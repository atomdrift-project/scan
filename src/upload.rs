//! Best-effort upload of scan results to a hopper instance.
//!
//! Mirrors the `/api/result` contract of postdoc's pull-based worker — the
//! same `ResultPayload` wire shape, the same zstd-compressed envelope — but
//! driven by a local `scan path` run instead of a poll loop. `scan path --hopper=<url>`
//! uses it to *renew* a sample hopper has already ingested with this build's
//! traits and model: hopper's `/api/result` is a lease-free `UPDATE ... WHERE
//! sha256 = ?`, so posting a result for an already-scanned SHA replaces its
//! stored analysis envelope (and an unknown SHA is a harmless no-op).
//!
//! Uploads run on a dedicated thread so blocking network I/O never stalls the
//! analysis pool. Network upload failures do not fail the local scan, but they
//! are surfaced as explicit errors so a successful local verdict cannot hide a
//! lost renewal. Failure to hand request-owned bytes to the uploader is
//! different: that is reported as a failed request rather than returning a
//! result known not to be persisted.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use reqwest::blocking::multipart::{Form, Part};
use serde::{Deserialize, Serialize};

mod queue;

use crate::engine::ScanResultEnvelope;
use crate::hopper::{
    Route, backoff, bearer, endpoints, error_chain, fuzz, is_permanent, parse_retry_after,
};

/// Hopper bounds the decompressed result body at 512 MiB (`maxResultBodyBytes`
/// in hopper's api.go); a larger document is truncated mid-stream and rejected
/// as invalid JSON, so an over-limit report is sent ML-verdict-only.
const HOPPER_MAX_RESULT_BODY_BYTES: usize = 512 << 20;

/// zstd's default level. Cleave reports are large, highly repetitive JSON that
/// zstd shrinks 3-5x on the wire; the compression cost is dwarfed by the
/// analysis that produced the payload.
const ZSTD_RESULT_LEVEL: i32 = 3;

/// Bound on resident jobs buffered ahead of the uploader thread. Overflow is
/// spooled to temporary files rather than blocking analysis or retaining large
/// reports in RAM. Shutdown still drains every accepted job.
const UPLOAD_QUEUE_DEPTH: usize = 16;

/// Cap on the uploader's reconciled-sha dedup set (~64-byte hex strings; the cap
/// bounds it near 10 MB). See the clear in the uploader loop.
const SEEN_SHAS_MAX: usize = 100_000;

/// Per-attempt request timeouts for `/api/upload`, separate from
/// [`REQUEST_TIMEOUT`] because that ceiling was sized for the small JSON
/// verdicts a renewal sends, not for streaming a multi-GiB artifact: with
/// artifacts up to [`HOPPER_MAX_UPLOAD_ARTIFACT_BYTES`], a legitimate transfer
/// on a mediocre link would outlast a short ceiling and be cut off as a timeout
/// rather than a real failure. The final ceiling matches hopper's own
/// `uploadBodyTimeout` (30 min) — no point the client giving up before the
/// server would have.
const UPLOAD_ATTEMPT_TIMEOUTS: [Duration; 4] = [
    Duration::from_secs(30),
    Duration::from_secs(120),
    Duration::from_secs(600),
    Duration::from_secs(1800),
];

/// Hopper's `maxUploadBytes` (`cmd/hopper/api.go`): artifact bytes above this
/// are rejected outright. hopper's handler checks `Content-Length` before
/// reading any body, but a client that starts streaming an oversized
/// multipart body anyway loses the race to read hopper's 413 — the
/// connection drops mid-write and reqwest surfaces it as "send failed
/// because receiver is gone", indistinguishable from a real network fault.
/// Checked client-side so an oversized dependency is skipped with one clear
/// log line instead of spending 4 retries on a request that can never
/// succeed (an over-cap upload lands in hopper with an empty `path`: the bytes
/// never arrive). Keep this in sync with hopper's cap — it can only shrink the
/// set of artifacts scan bothers attempting, never rescue one hopper would
/// reject.
const HOPPER_MAX_UPLOAD_ARTIFACT_BYTES: u64 = 8 << 30;

/// Hopper's ingestion lane header. A result renewed by `serve --hopper` is
/// one-shot: the caller that asked for the scan is already holding the verdict
/// in its own cache, so it will never ask again, and a renewal that does not
/// land means the artifact never enters the corpus at all. A fleet worker's
/// result is retryable for free — the job returns to the queue and is
/// dispatched again.
///
/// Hopper reserves ingestion slots for this lane so the retryable firehose
/// cannot starve the irreversible trickle. Declaring it is what claims the
/// reservation; a client that omits the header takes the worker lane.
const HOPPER_LANE_HEADER: &str = "X-Hopper-Lane";
const HOPPER_LANE_RENEW: &str = "renew";

/// Total wall-clock budget for renewing one result on hopper.
///
/// Hopper sheds result submissions with 503 + Retry-After when its ingestion
/// slots are saturated, and that saturation is driven by the worker fleet's
/// backlog — it can persist for many minutes. A budget of a few attempts over
/// seconds is not a retry so much as a coin flip: against a saturated hopper it
/// loses every renewal it is given.
const RENEW_BUDGET: Duration = Duration::from_secs(15 * 60);

/// Hopper's `validWorkerName` cap (`maxWorkerNameLen` in api.go).
const MAX_WORKER_NAME_LEN: usize = 64;

/// The JSON body POSTed to hopper's `/api/result`. The `{ml, llm?, raw}`
/// envelope is flattened onto the payload so the wire form is
/// `{sha256, worker, duration_ms, ml, llm, raw}` — byte-for-byte the shape
/// postdoc's pull-based worker sends, so hopper handles both identically.
#[derive(Debug, Serialize)]
pub struct ResultPayload {
    /// SHA-256 of the analyzed artifact.
    pub sha256: String,
    /// Name of the worker that produced the result.
    pub worker: String,
    /// Error reported instead of an analysis envelope, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Analysis duration in milliseconds.
    pub duration_ms: i64,
    /// The ML, LLM and raw report fields.
    #[serde(flatten)]
    pub envelope: Option<ScanResultEnvelope>,
}

/// Serialize and zstd-compress a result payload for upload. Returns the body
/// bytes and the `Content-Encoding` to advertise (`Some("zstd")` when
/// compression succeeded, `None` when it degraded to raw JSON). Returns `None`
/// only when serialization is unrecoverable — the result is then dropped.
///
/// If the serialized envelope exceeds hopper's body limit, the raw cleave report
/// is dropped and the ML verdict is sent alone: hopper still records the verdict
/// and only skips archive explosion, which beats losing the whole result.
pub fn encode_result_body(
    mut payload: ResultPayload,
    sha256: &str,
) -> Option<(Vec<u8>, Option<&'static str>)> {
    let json = serialize(&payload, sha256)?;
    let json = if json.len() > HOPPER_MAX_RESULT_BODY_BYTES {
        tracing::warn!(
            sha256 = %sha256,
            json_bytes = json.len(),
            limit_bytes = HOPPER_MAX_RESULT_BODY_BYTES,
            "upload: result JSON exceeds hopper's body limit; dropping raw report, posting ML verdict only",
        );
        // Empty the cleave report but keep the ml/llm verdict. An empty report
        // (not null) mirrors the envelope scan emits when there is no cleave
        // report, so the dropped-raw form stays a structurally valid envelope.
        if let Some(envelope) = payload.envelope.as_mut() {
            envelope.raw = std::sync::Arc::clone(crate::engine::empty_report());
        }
        // Moving/dropping the old buffer matters: shadowing it alone keeps
        // the oversized JSON allocation alive through the compression step.
        drop(json);
        serialize(&payload, sha256)?
    } else {
        json
    };
    // The JSON now owns everything needed for upload. Release the payload
    // before allocating the compressor and its output buffer.
    drop(payload);
    match zstd::encode_all(json.as_slice(), ZSTD_RESULT_LEVEL) {
        Ok(compressed) => Some((compressed, Some("zstd"))),
        Err(e) => {
            tracing::warn!(sha256 = %sha256, error = %e, "upload: zstd compress failed; sending uncompressed");
            Some((json, None))
        }
    }
}

fn serialize(payload: &ResultPayload, sha256: &str) -> Option<Vec<u8>> {
    serde_json::to_vec(payload)
        .inspect_err(|e| tracing::error!(sha256 = %sha256, error = %e, "upload: serialize failed"))
        .ok()
}

/// Worker identity tagged on uploaded results. Hopper's `validWorkerName`
/// requires a non-empty, space-free, printable-ASCII name no longer than 64
/// bytes; we derive it from the hostname (sanitized and truncated), falling back
/// to a fixed marker so the name is always valid.
#[must_use]
pub fn default_worker_name() -> String {
    let host = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_default();
    let sanitized: String = host
        .chars()
        .filter(char::is_ascii_graphic)
        .take(MAX_WORKER_NAME_LEN)
        .collect();
    if sanitized.is_empty() {
        "scan-fs".to_string()
    } else {
        sanitized
    }
}

/// Where an artifact's bytes can be loaded from, on demand — only after the
/// negotiation says hopper is missing them, so a known sha never reads a file or
/// decompresses a cache blob.
#[derive(Debug, Serialize, Deserialize)]
pub enum ArtifactBytes {
    /// The scanned file itself, read from disk.
    File(PathBuf),
    /// A fetched dependency, loaded from fletch's blob cache by its locator.
    Cached {
        /// The reference locator (PURL/URL) the cache keys the bytes under.
        locator: String,
    },
    /// Provenance only: hopper already holds these bytes, so none move (a
    /// registry fallback backfilling onto real content).
    None,
}

/// An artifact (the scanned file or a fetched dependency archive) offered to
/// hopper, with the provenance to record if hopper doesn't already have it.
/// Bytes are loaded lazily so the common "hopper already has it" case moves only
/// the 64-char sha across the wire, never the payload.
#[derive(Debug, Serialize, Deserialize)]
pub struct UploadArtifact {
    /// SHA-256 of the artifact's bytes — the negotiation and storage key.
    pub sha256: String,
    /// Size of the artifact's bytes, recorded in the provenance sidecar.
    pub size: u64,
    /// Filename hopper stores and sniffs the type from.
    pub filename: String,
    /// Where to load the bytes from, only if hopper turns out to need them.
    pub bytes: ArtifactBytes,
    /// Pre-serialized hopper `Sidecar` JSON (see [`crate::provenance::Upload`]).
    pub sidecar: Vec<u8>,
    /// Whether this artifact's provenance is worth backfilling onto a sample
    /// hopper already has the bytes for — true for fetched dependencies and
    /// map-backed roots carrying registry data, false for a plain local root
    /// whose sidecar contains only artifact + fetch identity.
    pub backfill: bool,
}

/// An artifact's bytes in a form every upload attempt can send without copying
/// them: a file reopened per attempt and streamed, or one shared buffer.
enum Body {
    File(PathBuf),
    Memory(bytes::Bytes),
}

impl Body {
    fn part(&self) -> std::io::Result<Part> {
        match self {
            Self::File(path) => {
                let file = std::fs::File::open(path)?;
                let len = file.metadata()?.len();
                Ok(Part::reader_with_length(file, len))
            }
            Self::Memory(bytes) => Ok(Part::reader_with_length(
                std::io::Cursor::new(bytes.clone()),
                bytes.len() as u64,
            )),
        }
    }
}

/// Work handed to the background uploader thread. Artifacts are reconciled before
/// a result so a never-seen top-level file's row exists before its verdict POST.
#[derive(Debug, Serialize, Deserialize)]
enum Job {
    /// Renew a verdict on hopper (the original `--upload` behavior). Encoding
    /// before handoff makes the job spoolable without reconstructing reports.
    Result {
        sha256: String,
        /// The package this artifact was analyzed as, when it was requested by
        /// one. Carried purely so the upload's log lines name the package the
        /// operator asked about rather than a digest they would have to
        /// resolve back to it by hand.
        purl: Option<String>,
        body: EncodedResult,
    },
    /// Ensure hopper has these artifacts' bytes+provenance, uploading only the
    /// ones it's missing.
    Artifacts {
        artifacts: Vec<UploadArtifact>,
        /// Durable staging directories owned by this job. They must outlive
        /// the analysis request's temporary upload directory and are removed
        /// after reconciliation, including when hopper already knows the SHA.
        cleanup_dirs: Vec<PathBuf>,
    },
    /// Mirror fetched dependencies into hopper as their own samples: bytes (only
    /// if missing) + provenance, then the verdict scan computed for each.
    Dependencies { deps: Vec<PreparedDependency> },
}

/// Wire-ready payload, identical whether resident or restored from the spool.
#[derive(Debug, Serialize, Deserialize)]
struct EncodedResult {
    bytes: Vec<u8>,
    zstd: bool,
}

fn prepare_result(
    worker: &str,
    sha256: &str,
    envelope: ScanResultEnvelope,
    tally: &Tally,
) -> Option<EncodedResult> {
    let payload = ResultPayload {
        sha256: sha256.to_owned(),
        worker: worker.to_owned(),
        error: None,
        duration_ms: 0,
        envelope: Some(envelope),
    };
    match encode_result_body(payload, sha256) {
        Some((bytes, encoding)) => Some(EncodedResult {
            bytes,
            zstd: encoding.is_some(),
        }),
        None => {
            tally.failed.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

/// Only upload inputs cross the queue, not the dependency's analysis graph.
#[derive(Debug, Serialize, Deserialize)]
struct PreparedDependency {
    sha256: String,
    locator: String,
    artifact: Option<UploadArtifact>,
    result: Option<EncodedResult>,
}

fn prepare_dependencies(
    worker: &str,
    deps: Vec<crate::engine::DepResult>,
    version: &str,
    analyzed_at: &str,
    tally: &Tally,
) -> Vec<PreparedDependency> {
    let collector = format!("scan+{worker}");
    deps.into_iter()
        .map(|dep| {
            let artifact = dep_artifact(&dep, &collector, analyzed_at);
            let result = crate::engine::dep_envelope(&dep, version, analyzed_at)
                .and_then(|envelope| prepare_result(worker, &dep.sha256, envelope, tally));
            PreparedDependency {
                sha256: dep.sha256,
                locator: dep.locator,
                artifact,
                result,
            }
        })
        .collect()
}

/// The two ends a renewal can reach, counted together because they are only
/// meaningful together: `uploaded` alone says nothing without `failed` beside
/// it, and a router reading one without the other would mistake a server that
/// files nothing for one with nothing to file.
#[derive(Debug, Default)]
struct Tally {
    /// Renewals hopper accepted.
    uploaded: AtomicUsize,
    /// Renewals that never reached hopper. Every one is a verdict that is lost,
    /// and until this counter existed the only trace was a warning in a log
    /// nobody was reading.
    failed: AtomicUsize,
}

/// Background uploader that POSTs scan results to hopper without blocking the
/// analysis threads. Created per `scan path --hopper` run; results are handed off
/// via [`Uploader::submit`] and flushed when the uploader is dropped.
#[derive(Debug)]
pub struct Uploader {
    /// `None` once flushed, or when the uploader thread failed to spawn.
    tx: Option<queue::Sender<Job>>,
    worker: Option<JoinHandle<()>>,
    /// Identity embedded into payloads before handing them to the queue.
    worker_name: String,
    /// Jobs accepted but not yet handled, including disk overflow.
    pending: Arc<AtomicUsize>,
    tally: Arc<Tally>,
}

/// A point-in-time view of the uploader, for `/_/stats`.
#[derive(Debug, Clone, Copy)]
pub struct UploadStats {
    /// Jobs accepted but not yet handled.
    pub pending: usize,
    /// Resident payload capacity; further queued jobs spill to disk.
    pub capacity: usize,
    /// Renewals that gave up after exhausting their retries.
    pub failed: usize,
    /// Renewals hopper accepted.
    pub uploaded: usize,
}

impl Uploader {
    /// Start a background uploader targeting `hopper_url`, tagging every result
    /// with `worker`. A failure to start — the thread will not spawn, or its
    /// HTTP client cannot be built — is logged, and every later submission then
    /// reports that the uploader stopped; the scan itself still completes.
    #[must_use]
    pub fn new(hopper_url: &str, worker: String) -> Self {
        crate::hopper::configure(hopper_url);
        let (tx, jobs) = queue::channel::<Job>(UPLOAD_QUEUE_DEPTH);
        let pending = Arc::new(AtomicUsize::new(0));
        let tally = Arc::new(Tally::default());
        let url = hopper_url.to_owned();
        let worker_name = worker.clone();
        let (thread_pending, thread_tally) = (Arc::clone(&pending), Arc::clone(&tally));
        let spawned = std::thread::Builder::new()
            .name("scan-upload".into())
            .spawn(move || {
                // On this thread, never the caller's: `serve` builds its
                // uploader inside its async runtime. Returning drops `jobs`,
                // so every submission says the uploader stopped.
                let Some(http) = crate::hopper::http() else {
                    return;
                };
                let hopper = Hopper::new(http.clone(), &url, worker, thread_tally);
                hopper.serve(jobs, &thread_pending);
            });
        let (tx, worker) = match spawned {
            Ok(handle) => {
                tracing::info!(hopper = %hopper_url, "upload: renewing results on hopper");
                (Some(tx), Some(handle))
            }
            Err(e) => {
                tracing::error!(error = %e, "upload: failed to spawn uploader thread; uploads disabled");
                (None, None)
            }
        };
        Self {
            tx,
            worker,
            worker_name,
            pending,
            tally,
        }
    }

    /// A point-in-time view of the upload queue.
    ///
    /// `pending` above capacity means the backlog can include disk-spooled jobs;
    /// it no longer means analyzers are blocked on network I/O. `failed` counts
    /// verdicts that could not be delivered.
    #[must_use]
    pub fn stats(&self) -> UploadStats {
        UploadStats {
            pending: self.pending.load(Ordering::Relaxed),
            capacity: UPLOAD_QUEUE_DEPTH,
            failed: self.tally.failed.load(Ordering::Relaxed),
            uploaded: self.tally.uploaded.load(Ordering::Relaxed),
        }
    }

    /// Hand a job to the uploader without waiting for network progress. When
    /// resident slots are full, only local spool I/O delays the caller. A spool
    /// failure is explicit, and returns ownership for staging cleanup.
    fn enqueue(&self, job: Job) -> Result<(), Job> {
        let Some(tx) = &self.tx else {
            return Err(job);
        };
        self.pending.fetch_add(1, Ordering::Relaxed);
        tx.send(job).map_err(|(job, error)| {
            self.pending.fetch_sub(1, Ordering::Relaxed);
            tracing::error!(%error, "upload: could not queue job");
            job
        })
    }

    /// Queue a result for upload. A failed handoff is logged and counted.
    pub fn submit(&self, sha256: String, purl: Option<String>, envelope: ScanResultEnvelope) {
        let Some(body) = prepare_result(&self.worker_name, &sha256, envelope, &self.tally) else {
            return;
        };
        let job = Job::Result { sha256, purl, body };
        if let Err(Job::Result { sha256, .. }) = self.enqueue(job) {
            self.tally.failed.fetch_add(1, Ordering::Relaxed);
            tracing::error!(
                sha256 = %sha256,
                "upload: could not queue result for hopper"
            );
        }
    }

    /// Queue artifacts (the scanned file and any fetched dependency archives) for
    /// content reconciliation: hopper is asked which it lacks, and only those are
    /// uploaded with their provenance. Submit before the matching [`Self::submit`] so a
    /// new top-level file's row exists before its verdict lands.
    pub fn submit_artifacts(&self, artifacts: Vec<UploadArtifact>) {
        if !self.enqueue_artifacts(artifacts, Vec::new()) {
            tracing::error!("upload: could not queue artifacts for hopper");
        }
    }

    /// Copy file-backed artifacts into a staging directory owned by the
    /// uploader, then queue them for reconciliation.
    ///
    /// Server uploads are initially held in a request-scoped [`tempfile::TempDir`].
    /// The analysis request is allowed to delete that directory as soon as the
    /// verdict is ready, while this uploader may still be waiting behind other
    /// jobs. Keeping an independent on-disk copy makes the handoff reliable
    /// without retaining a potentially large upload in memory or reading it
    /// before `/api/known` says hopper needs it.
    ///
    /// # Errors
    /// Returns a message when staging fails or the uploader has stopped.
    pub fn submit_artifacts_durable(
        &self,
        mut artifacts: Vec<UploadArtifact>,
    ) -> Result<(), String> {
        if artifacts.is_empty() {
            return Ok(());
        }

        let staging = tempfile::Builder::new()
            .prefix("scan-upload-")
            .tempdir()
            .map_err(|error| format!("create durable upload staging directory: {error}"))?;
        for (index, artifact) in artifacts.iter_mut().enumerate() {
            let ArtifactBytes::File(source) = &artifact.bytes else {
                continue;
            };
            let filename = source
                .file_name()
                .filter(|name| !name.is_empty())
                .map_or_else(
                    || format!("artifact-{index}"),
                    |name| name.to_string_lossy().into_owned(),
                );
            let destination = staging.path().join(format!("{index}-{filename}"));
            std::fs::copy(source, &destination).map_err(|error| {
                format!(
                    "copy {} into durable upload staging: {error}",
                    source.display()
                )
            })?;
            artifact.bytes = ArtifactBytes::File(destination);
        }

        // `keep` transfers directory ownership from TempDir to the queued job.
        // The paths stored in `artifacts` remain valid after this call.
        let staging_path = staging.keep();
        if self.enqueue_artifacts(artifacts, vec![staging_path]) {
            Ok(())
        } else {
            Err("could not queue durable artifacts for hopper".to_string())
        }
    }

    /// Persist one already-built artifact from bytes supplied by the caller,
    /// then queue it for reconciliation. This is used when Scan has a local
    /// verdict and can answer immediately, but Hopper may not yet have the
    /// artifact row that the verdict belongs to.
    ///
    /// # Errors
    /// Returns a message when staging fails or the uploader has stopped.
    pub fn submit_artifact_bytes_durable(
        &self,
        mut artifact: UploadArtifact,
        bytes: &[u8],
    ) -> Result<(), String> {
        let staging = tempfile::Builder::new()
            .prefix("scan-upload-")
            .tempdir()
            .map_err(|error| format!("create durable upload staging directory: {error}"))?;
        let destination = staging.path().join("artifact");
        std::fs::write(&destination, bytes)
            .map_err(|error| format!("write durable upload staging file: {error}"))?;
        artifact.bytes = ArtifactBytes::File(destination);

        let staging_path = staging.keep();
        if self.enqueue_artifacts(vec![artifact], vec![staging_path]) {
            Ok(())
        } else {
            Err("could not queue durable artifact for hopper".to_string())
        }
    }

    /// Queue fetched dependencies to mirror into hopper as their own samples:
    /// bytes (only if hopper lacks them) + provenance, then each dependency's
    /// verdict. Submit after the root [`Self::submit_artifacts`] and before the root
    /// [`Self::submit`] so the dependencies' rows exist before any verdict — the root's
    /// or their own — lands.
    pub fn submit_dependencies(
        &self,
        deps: Vec<crate::engine::DepResult>,
        version: &str,
        analyzed_at: &str,
    ) {
        if deps.is_empty() {
            return;
        }
        let job = Job::Dependencies {
            deps: prepare_dependencies(&self.worker_name, deps, version, analyzed_at, &self.tally),
        };
        if let Err(Job::Dependencies { deps, .. }) = self.enqueue(job) {
            self.tally.failed.fetch_add(
                deps.iter().filter(|d| d.result.is_some()).count(),
                Ordering::Relaxed,
            );
            tracing::error!(
                dependencies = deps.len(),
                "upload: could not queue dependencies for hopper"
            );
        }
    }

    /// Queue an artifact batch, reclaiming its staging directories when the
    /// queue is unavailable or closed: the thread cannot clean a job it never
    /// received.
    fn enqueue_artifacts(
        &self,
        artifacts: Vec<UploadArtifact>,
        cleanup_dirs: Vec<PathBuf>,
    ) -> bool {
        if artifacts.is_empty() {
            cleanup_upload_dirs(cleanup_dirs);
            return true;
        }
        match self.enqueue(Job::Artifacts {
            artifacts,
            cleanup_dirs,
        }) {
            Ok(()) => true,
            Err(job) => {
                if let Job::Artifacts { cleanup_dirs, .. } = job {
                    cleanup_upload_dirs(cleanup_dirs);
                }
                false
            }
        }
    }
}

impl Drop for Uploader {
    /// Stop accepting new results and wait for in-flight uploads to finish, so a
    /// scan's results are fully renewed before the process exits.
    fn drop(&mut self) {
        // Dropping the sender ends the thread's `for job in jobs` loop.
        self.tx = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// One hopper, as every call to it needs it: the HTTP client, each route at
/// every address `--hopper` named, the bearer token, and the name results are
/// filed under.
struct Hopper {
    http: reqwest::blocking::Client,
    result: Route,
    known: Route,
    upload: Route,
    token: Option<&'static str>,
    worker: String,
    tally: Arc<Tally>,
}

impl Hopper {
    fn new(
        http: reqwest::blocking::Client,
        hopper_url: &str,
        worker: String,
        tally: Arc<Tally>,
    ) -> Self {
        // One entry per address `--hopper` named, in preference order. A retry
        // walks down the list, so a replica that stops answering costs the
        // first attempt and the primary takes the rest.
        let bases = endpoints(hopper_url);
        Self {
            http,
            result: Route::new(&bases, "/api/result"),
            known: Route::new(&bases, "/api/known"),
            upload: Route::new(&bases, "/api/upload"),
            token: crate::hopper::token(),
            worker,
            tally,
        }
    }

    /// The uploader thread's loop: handle every queued job until the sender
    /// side is dropped.
    fn serve(&self, jobs: queue::Receiver<Job>, pending: &AtomicUsize) {
        // The same default blob cache scan fetched dependencies into, so a
        // missing dep's bytes are loaded locally rather than re-fetched.
        let cache = crate::fetch::open_blob_cache()
            .map_err(|e| tracing::warn!(error = %e, "upload: blob cache unavailable; dependency bytes will be re-fetched"))
            .ok();
        // Shas reconciled this run, so a dependency shared by many scanned
        // files is negotiated and uploaded at most once.
        let mut seen: HashSet<String> = HashSet::new();
        for job in jobs {
            // Handled below whatever the outcome; the depth is about the
            // queue, not about success.
            pending.fetch_sub(1, Ordering::Relaxed);
            let job = match job {
                Ok(job) => job,
                Err(error) => {
                    self.tally.failed.fetch_add(1, Ordering::Relaxed);
                    tracing::error!(%error, "upload: could not read spooled job");
                    continue;
                }
            };
            // Bound the dedup set: a long-lived `serve --hopper` process
            // reconciles an unbounded stream of unique shas. Clearing past the
            // cap only costs a redundant /known round-trip for shas negotiated
            // earlier.
            if seen.len() >= SEEN_SHAS_MAX {
                seen.clear();
            }
            match job {
                // Posted unconditionally. Asking first cost a round trip on
                // the same three-slot renew lane the post uses, to avoid a
                // write that hopper now declines in one indexed read — and the
                // answer aged the moment it arrived, because only the store is
                // ordered against the other producers pushing the same
                // dependency (hopper's `unchangedStore`).
                Job::Result { sha256, purl, body } => {
                    self.renew_encoded(&sha256, purl.as_deref(), body)
                }
                Job::Artifacts {
                    artifacts,
                    cleanup_dirs,
                } => {
                    self.reconcile(cache.as_ref(), &mut seen, artifacts);
                    cleanup_upload_dirs(cleanup_dirs);
                }
                Job::Dependencies { deps } => {
                    self.sync_prepared_dependencies(cache.as_ref(), &mut seen, deps)
                }
            }
        }
    }

    /// Reconcile a batch of artifacts against hopper: negotiate which it's
    /// missing (one `/api/known` round-trip), then upload only those — bytes
    /// plus provenance. `seen` dedups across batches so a dependency shared by
    /// many files is handled once. Best-effort throughout: a failure logs and
    /// the scan continues.
    fn reconcile(
        &self,
        cache: Option<&fletch::fetch::BlobCache>,
        seen: &mut HashSet<String>,
        artifacts: Vec<UploadArtifact>,
    ) {
        // Drop anything reconciled earlier this run; mark the rest seen now so a
        // later batch never re-negotiates them.
        let fresh: Vec<UploadArtifact> = artifacts
            .into_iter()
            .filter(|a| seen.insert(a.sha256.clone()))
            .collect();
        if fresh.is_empty() {
            return;
        }

        // The one question that gates the expensive byte transfer: which of these
        // does hopper already have? Everything it has, we never send.
        let shas: Vec<&str> = fresh.iter().map(|a| a.sha256.as_str()).collect();
        let known = self.known(&shas);

        for art in fresh {
            if known.contains(&art.sha256) {
                // hopper has the bytes. For a dependency, (re)send its provenance
                // so hopper refreshes the registry snapshot — the bytes never
                // move, only the small sidecar, and hopper preserves the original
                // discovery wrapper, updating just the registry data. Plain local
                // roots set `backfill` false; map-backed roots preserve and
                // refresh theirs.
                if art.backfill {
                    self.upload(&art, None);
                } else {
                    tracing::debug!(sha256 = %art.sha256, "upload: hopper already has artifact; skipping");
                }
                continue;
            }
            if art.size > HOPPER_MAX_UPLOAD_ARTIFACT_BYTES {
                // hopper will reject this outright (`maxUploadBytes` in
                // cmd/hopper/api.go) — and does so by dropping the connection
                // mid-write rather than returning a clean 413, so attempting it
                // would just spend the full retry budget on a guaranteed failure
                // and log a misleading "receiver is gone". Skip the transfer, not
                // the verdict: `sync_dependencies` still posts this dependency's
                // result, so hopper ends up with the row the comment above
                // describes (verdict, no bytes) — the same state a blob-cache
                // eviction produces, and no worse than it.
                tracing::warn!(
                    sha256 = %art.sha256,
                    file = %art.filename,
                    size = art.size,
                    limit = HOPPER_MAX_UPLOAD_ARTIFACT_BYTES,
                    "upload: artifact exceeds hopper's upload size cap; skipping bytes (verdict still posted)"
                );
                continue;
            }
            let body = match &art.bytes {
                ArtifactBytes::File(path) => path.is_file().then(|| Body::File(path.clone())),
                // The blob cache is size-capped and swept on a timer, so a
                // dependency's bytes can be evicted between the fetch that cached
                // them and this upload — a window that is a whole archive
                // analysis wide. Losing that race is how a dependency lands in
                // hopper as a row with a verdict and no bytes: analyzed,
                // uncontained, and therefore claimable, but with nothing any
                // worker can be served. Re-fetch rather than give up; the
                // artifact is content-addressed, so recovering it is always
                // possible while the registry serves it.
                ArtifactBytes::Cached { locator } => cache
                    .and_then(|c| c.load(locator))
                    .or_else(|| refetch_artifact(locator, &art.sha256))
                    .map(|bytes| Body::Memory(bytes.into())),
                ArtifactBytes::None => None,
            };
            let Some(body) = body else {
                tracing::warn!(sha256 = %art.sha256, file = %art.filename, "upload: artifact bytes unavailable; skipping");
                continue;
            };
            self.upload(&art, Some(&body));
        }
    }

    /// Mirror fetched dependencies into hopper as their own samples. For each
    /// dependency not already handled this run: ensure hopper has its bytes
    /// (uploaded only when missing) and provenance, then POST the verdict scan
    /// already computed for it. Best-effort throughout — a failure logs and the
    /// next dependency proceeds, exactly like the artifact reconciliation it
    /// builds on.
    fn sync_prepared_dependencies(
        &self,
        cache: Option<&fletch::fetch::BlobCache>,
        seen: &mut HashSet<String>,
        deps: Vec<PreparedDependency>,
    ) {
        // Each dependency is reconciled and verdict-posted once per run; a dependency
        // shared by many scanned files is handled the first time it is seen.
        let mut fresh: Vec<PreparedDependency> = deps
            .into_iter()
            .filter(|d| seen.insert(d.sha256.clone()))
            .collect();
        if fresh.is_empty() {
            return;
        }
        // Bytes + provenance first, so each dependency's row exists before its
        // verdict UPDATE (hopper's `/api/result` no-ops on a missing row). The
        // local seen set starts empty — the run-level dedup above already
        // removed repeats.
        let artifacts: Vec<UploadArtifact> =
            fresh.iter_mut().filter_map(|d| d.artifact.take()).collect();
        self.reconcile(cache, &mut HashSet::new(), artifacts);
        // Then the verdict for each dependency that has one, keyed by its content
        // sha. A dependency the embedded pass never reached carries none: its bytes
        // and provenance went up above, so hopper holds the artifact and can analyze
        // it, but scan posts no verdict it did not compute. Logged rather than
        // dropped silently — an unevaluated dependency is a coverage gap worth
        // seeing, not a routine skip.
        //
        // Every dependency's verdict is posted. Asking hopper first — the
        // `traits_version` probe this replaced — could only ever answer for the
        // instant it was asked: the popular dependencies it was meant to spare
        // (inherits, x/tools, setup-go) are exactly the ones several scans push at
        // once, so each probe returned "not current" and each scan posted anyway.
        // The store settles it now, in one indexed read, because the store is the
        // only place ordered against the other producers.
        for dep in fresh {
            let Some(body) = dep.result else {
                tracing::info!(
                    sha256 = %dep.sha256,
                    locator = %dep.locator,
                    "upload: dependency not evaluated; stored for analysis without a verdict"
                );
                continue;
            };
            // A dependency's locator is a PURL or a URL; only the former belongs
            // under a `purl` field, so a URL-sourced dependency logs by digest.
            let purl = dep
                .locator
                .starts_with("pkg:")
                .then_some(dep.locator.as_str());
            self.renew_encoded(&dep.sha256, purl, body);
        }
    }

    /// POST the batch existence probe (`/api/known`) and return the digests
    /// whose bytes hopper already holds. On any failure returns an empty set —
    /// the caller then treats every artifact as missing, which is the safe
    /// direction: a failed probe re-sends bytes hopper had, where the other way
    /// round would withhold an artifact it does not.
    ///
    /// One question, deliberately. Whether a verdict is already current is a
    /// different question about a different column, and its answer ages before
    /// it can be acted on — hopper decides that at the store (`unchangedStore`),
    /// where it is ordered against the other producers.
    fn known(&self, shas: &[&str]) -> HashSet<String> {
        #[derive(Serialize)]
        struct KnownRequest<'a> {
            sha256: &'a [&'a str],
        }
        #[derive(serde::Deserialize)]
        struct KnownResponse {
            #[serde(default)]
            known: Vec<String>,
        }
        // Each address in turn. Failing this probe is safe but not free: the
        // caller then treats every artifact as missing and pushes bytes hopper
        // already holds, so stopping at an unreachable replica would spend an
        // outage re-uploading the corpus to a primary that is up and one line down
        // the list. A decode failure is not retried elsewhere — the next address
        // runs the same build and would answer the same way.
        for url in self.known.each() {
            let resp = bearer(self.http.post(url), self.token)
                .json(&KnownRequest { sha256: shas })
                .send();
            match resp {
                Ok(resp) if resp.status().is_success() => {
                    return match resp.json::<KnownResponse>() {
                        Ok(kr) => kr.known.into_iter().collect(),
                        Err(e) => {
                            tracing::warn!(error = %error_chain(&e), "upload: known response decode failed");
                            HashSet::new()
                        }
                    };
                }
                Ok(resp) => {
                    tracing::warn!(endpoint = %url, status = %resp.status(), "upload: known probe non-success");
                }
                Err(e) => {
                    tracing::warn!(endpoint = %url, error = %error_chain(&e), "upload: known probe failed");
                }
            }
        }
        HashSet::new()
    }

    /// The multipart `/api/upload` body: the provenance part first, as hopper's
    /// handler requires, then the file part when there are bytes to send. Built
    /// afresh per attempt — a form is consumed by sending it — from bytes that
    /// are never copied.
    fn form(art: &UploadArtifact, body: Option<&Body>) -> std::io::Result<Form> {
        let provenance = Part::bytes(art.sidecar.clone())
            .mime_str("application/json")
            .map_err(std::io::Error::other)?;
        let form = Form::new().part("provenance", provenance);
        Ok(match body {
            Some(body) => form.part("file", body.part()?.file_name(art.filename.clone())),
            None => form,
        })
    }

    /// Store an artifact's provenance on hopper, with its bytes when `body` is
    /// given; without, hopper attaches the provenance to the sample it already
    /// holds. Retries transient failures on [`UPLOAD_ATTEMPT_TIMEOUTS`]; a
    /// permanent refusal stops at once. Returns whether hopper accepted it.
    fn upload(&self, art: &UploadArtifact, body: Option<&Body>) -> bool {
        let kind = if body.is_some() {
            "artifact"
        } else {
            "provenance backfill"
        };
        let mut retry_after = None;
        for (attempt, timeout) in UPLOAD_ATTEMPT_TIMEOUTS.into_iter().enumerate() {
            if attempt > 0 {
                std::thread::sleep(backoff(attempt, retry_after, fuzz()));
            }
            let form = match Self::form(art, body) {
                Ok(form) => form,
                Err(e) => {
                    tracing::warn!(sha256 = %art.sha256, kind, error = %e, "upload: artifact body unavailable; not retrying");
                    return false;
                }
            };
            let request = bearer(self.http.post(self.upload.at(attempt)), self.token)
                .timeout(timeout)
                .multipart(form);
            retry_after = None;
            match request.send() {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!(sha256 = %art.sha256, kind, file = %art.filename, size = art.size, "upload: stored on hopper");
                    return true;
                }
                Ok(resp) => {
                    let status = resp.status();
                    if is_permanent(status) {
                        let body = resp.text().unwrap_or_default();
                        tracing::error!(
                            sha256 = %art.sha256,
                            kind,
                            %status,
                            body = %body_excerpt(&body),
                            provenance = %body_excerpt(&String::from_utf8_lossy(&art.sidecar)),
                            "upload: hopper rejected artifact write; not retrying"
                        );
                        return false;
                    }
                    retry_after = parse_retry_after(resp.headers());
                    tracing::warn!(sha256 = %art.sha256, kind, %status, attempt, "upload: non-success response");
                }
                Err(e) => {
                    tracing::warn!(sha256 = %art.sha256, kind, error = %error_chain(&e), attempt, "upload: send failed");
                }
            }
        }
        tracing::error!(sha256 = %art.sha256, kind, attempts = UPLOAD_ATTEMPT_TIMEOUTS.len(), "upload: failed to write artifact to hopper; giving up after retries");
        false
    }

    /// POST one result to hopper, retrying transient failures within
    /// [`RENEW_BUDGET`]. A permanent refusal stops at once.
    fn renew_encoded(&self, sha256: &str, purl: Option<&str>, encoded: EncodedResult) {
        let encoding = encoded.zstd.then_some("zstd");
        // Shared, not copied, across attempts: a compressed report can be large.
        let body = bytes::Bytes::from(encoded.bytes);

        let started = Instant::now();
        let mut retry_after: Option<Duration> = None;
        for attempt in 0.. {
            if attempt > 0 {
                match renew_delay(attempt, retry_after, started.elapsed(), fuzz()) {
                    Some(delay) => std::thread::sleep(delay),
                    None => break,
                }
            }
            let mut request = bearer(self.http.post(self.result.at(attempt)), self.token)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                // Claim hopper's reserved lane: this renewal is one-shot, and the
                // caller is already holding the verdict in its cache.
                .header(HOPPER_LANE_HEADER, HOPPER_LANE_RENEW)
                .body(body.clone());
            if let Some(enc) = encoding {
                request = request.header(reqwest::header::CONTENT_ENCODING, enc);
            }
            retry_after = None;
            match request.send() {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!(
                        sha256 = %sha256,
                        purl,
                        attempt,
                        waited_ms = started.elapsed().as_millis(),
                        "upload: result renewed on hopper",
                    );
                    self.tally.uploaded.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Ok(resp) => {
                    let status = resp.status();
                    if is_permanent(status) {
                        let body = resp.text().unwrap_or_default();
                        self.tally.failed.fetch_add(1, Ordering::Relaxed);
                        tracing::error!(sha256 = %sha256, purl, %status, body = %body_excerpt(&body), "upload: hopper rejected result; not retrying");
                        return;
                    }
                    // Hopper sends Retry-After when it sheds; it knows when its
                    // slots free, so honour it rather than guessing shorter.
                    retry_after = parse_retry_after(resp.headers());
                    tracing::warn!(sha256 = %sha256, purl, %status, attempt, "upload: non-success response");
                }
                Err(e) => {
                    tracing::warn!(sha256 = %sha256, purl, error = %error_chain(&e), attempt, "upload: send failed");
                }
            }
        }
        tracing::error!(
            sha256 = %sha256,
            purl,
            budget_s = RENEW_BUDGET.as_secs(),
            "upload: failed to renew result on hopper; giving up after the renewal budget",
        );
        self.tally.failed.fetch_add(1, Ordering::Relaxed);
    }
}

/// Bound and flatten untrusted response bodies before writing them to logs.
fn body_excerpt(body: &str) -> String {
    const MAX_CHARS: usize = 512;
    let compact = body.replace(['\r', '\n', '\t'], " ");
    let mut excerpt: String = compact.chars().take(MAX_CHARS).collect();
    if compact.chars().count() > MAX_CHARS {
        excerpt.push_str("...");
    }
    excerpt
}

/// Remove uploader-owned staging directories after a job has been fully
/// reconciled. Failure is logged but does not turn a completed upload into a
/// failed scan; the directory is temporary and the next service cleanup can
/// remove an orphan left by an unusual filesystem error.
fn cleanup_upload_dirs(dirs: Vec<PathBuf>) {
    for dir in dirs {
        if let Err(error) = std::fs::remove_dir_all(&dir) {
            tracing::warn!(path = %dir.display(), error = %error, "upload: durable staging cleanup failed");
        }
    }
}

/// Mirror a result's fetched dependencies into a hopper instance from a caller
/// that has no [`Uploader`] (postdoc's pull-based worker). Dedups within this call,
/// and reconciles bytes + provenance before posting each verdict. Best-effort:
/// every failure is logged, never propagated, so a dependency sync never
/// disturbs the result it followed.
pub fn sync_result_dependencies(
    client: &reqwest::blocking::Client,
    base_url: &str,
    worker: &str,
    version: &str,
    analyzed_at: &str,
    cache: Option<&fletch::fetch::BlobCache>,
    deps: Vec<crate::engine::DepResult>,
) {
    if deps.is_empty() {
        return;
    }
    // Standalone reconciliation: nothing is watching the tally here.
    let hopper = Hopper::new(client.clone(), base_url, worker.to_owned(), Arc::default());
    let deps = prepare_dependencies(&hopper.worker, deps, version, analyzed_at, &hopper.tally);
    hopper.sync_prepared_dependencies(cache, &mut HashSet::new(), deps);
}

/// Build the upload artifact for a fetched dependency. Its bytes load from the
/// fetch blob cache only if hopper needs them; its sidecar uses the exact
/// registry snapshot already captured and analyzed, with no registry/cache
/// lookup on the upload path. `None`, logged, when the sidecar cannot be built.
fn dep_artifact(
    dep: &crate::engine::DepResult,
    collector: &str,
    now: &str,
) -> Option<UploadArtifact> {
    let filename = crate::engine::artifact_filename(&dep.url, &dep.locator);
    let upload = crate::provenance::Upload {
        filename: &filename,
        sha256: &dep.sha256,
        size_bytes: dep.size,
        collector,
        at: now,
        url: &dep.url,
        purl: dep
            .locator
            .starts_with("pkg:")
            .then_some(dep.locator.as_str()),
    };
    let sidecar = match &dep.provenance {
        Some(provenance) => upload.sidecar_from_provenance(provenance),
        None => upload.sidecar(None, &[]),
    };
    let sidecar = sidecar
        .map_err(|e| tracing::error!(sha256 = %dep.sha256, error = %e, "upload: dependency sidecar could not be built; not uploaded"))
        .ok()?;
    Some(UploadArtifact {
        sha256: dep.sha256.clone(),
        size: dep.size,
        sidecar,
        filename,
        bytes: ArtifactBytes::Cached {
            locator: dep.locator.clone(),
        },
        backfill: true,
    })
}

/// Re-fetch a dependency's bytes after the blob cache lost them, returning them
/// only if they still hash to the digest the verdict was computed over.
///
/// The digest check is not a formality. A locator is not always a pin: a
/// versionless PURL re-resolves to whatever the registry's `latest` is *now*,
/// and a tag can be moved. Uploading whatever comes back under the recorded
/// sha256 would file one artifact's bytes under another's identity — worse than
/// the missing bytes this is recovering from — so a mismatch is dropped loudly
/// and the artifact stays absent.
fn refetch_artifact(locator: &str, sha256: &str) -> Option<Vec<u8>> {
    let target = if locator.starts_with("pkg:") {
        fletch::RefLocator::Purl(locator.to_string())
    } else {
        fletch::RefLocator::Url(locator.to_string())
    };
    let (bytes, _, _) = match crate::fetch::fetch_one(target, false) {
        Ok(fetched) => fetched,
        Err(e) => {
            tracing::warn!(
                %locator, %sha256, error = %error_chain(&*e),
                "upload: dependency bytes gone from the cache and could not be re-fetched"
            );
            return None;
        }
    };
    if !bytes_match_digest(&bytes, sha256, locator) {
        return None;
    }
    tracing::info!(
        %locator, %sha256, bytes = bytes.len(),
        "upload: dependency bytes evicted from the cache; re-fetched for upload"
    );
    Some(bytes)
}

/// Whether `bytes` are the ones `sha256` names — the guard that keeps a
/// re-fetch from filing one artifact's content under another's identity.
///
/// Split out from [`refetch_artifact`] so the rule can be tested without a
/// network: it is the one step there that must never be relaxed, and a caller
/// that ever treats a mismatch as acceptable would corrupt the corpus silently.
fn bytes_match_digest(bytes: &[u8], sha256: &str, locator: &str) -> bool {
    use sha2::{Digest as _, Sha256};
    let got = format!("{:x}", Sha256::digest(bytes));
    if got != sha256 {
        tracing::warn!(
            %locator, expected = %sha256, got = %got,
            "upload: re-fetched dependency does not match the analyzed bytes; not uploading"
        );
        return false;
    }
    true
}

/// [`backoff`] within the renewal budget, or `None` once the budget is spent.
/// Never sleeps past the budget: a sleep that outlives it would turn the
/// ceiling into a lie and delay the give-up log.
fn renew_delay(
    attempt: usize,
    retry_after: Option<Duration>,
    elapsed: Duration,
    fuzz: f64,
) -> Option<Duration> {
    let remaining = RENEW_BUDGET
        .checked_sub(elapsed)
        .filter(|left| !left.is_zero())?;
    Some(backoff(attempt, retry_after, fuzz).min(remaining))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// A hopper with every route at `bases`, no token, and a plain client.
    fn hopper_at(bases: &str) -> Hopper {
        Hopper::new(
            reqwest::blocking::Client::new(),
            bases,
            "test".to_string(),
            Arc::default(),
        )
    }

    /// One request off `listener`, answered with `response`; returns the raw
    /// request. Reads the body by its Content-Length, which a sized upload
    /// must send.
    fn serve_one(
        listener: TcpListener,
        response: &'static [u8],
    ) -> std::thread::JoinHandle<Vec<u8>> {
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = Vec::new();
            let mut buf = [0u8; 8192];
            let body_start = loop {
                let n = stream.read(&mut buf).expect("read");
                assert!(n > 0, "connection closed before the headers ended");
                request.extend_from_slice(&buf[..n]);
                if let Some(at) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    break at + 4;
                }
            };
            let head = String::from_utf8_lossy(&request[..body_start]).to_ascii_lowercase();
            let length: usize = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .map_or(0, |v| v.trim().parse().expect("content-length"));
            while request.len() < body_start + length {
                let n = stream.read(&mut buf).expect("read body");
                assert!(n > 0, "connection closed mid-body");
                request.extend_from_slice(&buf[..n]);
            }
            stream.write_all(response).expect("respond");
            request
        })
    }

    const NO_CONTENT: &[u8] =
        b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

    fn artifact(sha: &str, size: u64, bytes: ArtifactBytes) -> UploadArtifact {
        UploadArtifact {
            sha256: sha.to_string(),
            size,
            filename: "x.bin".to_string(),
            bytes,
            sidecar: br#"{"schema_version":"1.0"}"#.to_vec(),
            backfill: false,
        }
    }

    /// The budget is the whole point: a renewal is one-shot, so it must outlive
    /// a saturated hopper rather than give up after a few seconds.
    #[test]
    fn renew_delay_respects_the_budget() {
        assert!(renew_delay(1, None, Duration::ZERO, 0.5).is_some());
        assert!(renew_delay(9, None, RENEW_BUDGET - Duration::from_secs(1), 0.5).is_some());
        assert!(renew_delay(9, None, RENEW_BUDGET, 0.5).is_none());
        assert!(renew_delay(9, None, RENEW_BUDGET + Duration::from_secs(1), 0.5).is_none());
    }

    /// A sleep must never outlive the budget, or the give-up log arrives late
    /// and the ceiling stops meaning anything.
    #[test]
    fn renew_delay_never_sleeps_past_the_budget() {
        let elapsed = RENEW_BUDGET - Duration::from_millis(400);
        let d = renew_delay(12, Some(Duration::from_secs(60)), elapsed, 1.0).unwrap();
        assert!(
            d <= Duration::from_millis(400),
            "slept past the budget: {d:?}"
        );
    }

    /// Hopper knows when its slots free; a shorter sleep just burns a
    /// slot-acquire on a pool that has already said it is full. Uploads and
    /// renewals share the rule.
    #[test]
    fn backoff_honours_retry_after_as_a_floor() {
        let hint = Duration::from_secs(30);
        assert!(backoff(1, Some(hint), 0.0) >= hint);
        assert!(renew_delay(1, Some(hint), Duration::ZERO, 0.0).unwrap() >= hint);
    }

    /// Claiming hopper's reserved lane is what keeps a one-shot renewal from
    /// competing with the retryable worker firehose.
    #[test]
    fn lane_header_matches_hoppers_contract() {
        assert_eq!(HOPPER_LANE_HEADER, "X-Hopper-Lane");
        assert_eq!(HOPPER_LANE_RENEW, "renew");
    }

    /// The wire body round-trips through zstd back to the exact JSON serde
    /// produced: this is the same shape hopper's `/api/result` decodes, so an fs
    /// upload is byte-identical to postdoc's worker uploading the same payload.
    #[test]
    fn encode_result_body_round_trips_through_zstd() {
        let payload = ResultPayload {
            sha256: "a".repeat(64),
            worker: "scan-fs".to_string(),
            error: None,
            duration_ms: 0,
            envelope: None,
        };
        let expected = serde_json::to_vec(&payload).unwrap();
        let (body, encoding) = encode_result_body(payload, "test").expect("encodes");
        assert_eq!(encoding, Some("zstd"));
        let decoded = zstd::decode_all(body.as_slice()).expect("valid zstd");
        assert_eq!(decoded, expected);

        // The flattened payload carries the transport fields and omits `error`
        // (skip_serializing_if), matching postdoc's worker's wire form.
        let value: serde_json::Value = serde_json::from_slice(&decoded).unwrap();
        assert_eq!(value["worker"], "scan-fs");
        assert_eq!(value["duration_ms"], 0);
        assert!(value.get("error").is_none());
    }

    #[test]
    fn encode_result_body_preserves_the_full_typed_report() {
        let raw: cleave::types::CompactReport = serde_json::from_value(serde_json::json!({
            "files": [
                {"id": 0, "path": "fixture.zip", "type": "zip", "sha": "root", "size": 128, "depth": 0},
                {"id": 1, "parent": 0, "path": format!("fixture.zip!!{}.py", "member".repeat(4096)),
                 "type": "python", "sha": "child", "size": 32, "depth": 1}
            ]
        }))
        .expect("typed raw report");
        assert_eq!(raw.files.len(), 2);
        let raw = crate::engine::StoredReport::new(&raw)
            .expect("serializes")
            .json;
        let raw_json = raw.get().to_owned();
        let payload = ResultPayload {
            sha256: "a".repeat(64),
            worker: "fixture".into(),
            error: None,
            duration_ms: 42,
            envelope: Some(ScanResultEnvelope {
                ml: crate::engine::MlSection {
                    v: "7",
                    probability: 0.0,
                    level: crate::model::Level::Clean,
                    conf: Some(100),
                    model_scores: Vec::new(),
                    skipped_models: Vec::new(),
                    version: "fixture".into(),
                    eng: "fixture",
                    analyzed_at: "2026-10-19T00:00:00Z".into(),
                    files: Vec::new(),
                    pids: None,
                    deleted: None,
                },
                llm: None,
                raw,
            }),
        };
        let expected = serde_json::to_vec(&payload).expect("expected JSON");
        // The stored report rides through the flattened envelope verbatim.
        assert!(
            String::from_utf8_lossy(&expected).ends_with(&format!(r#""raw":{raw_json}}}"#)),
            "raw is embedded as stored"
        );
        let (body, encoding) = encode_result_body(payload, "test").expect("encode full report");
        assert_eq!(encoding, Some("zstd"));
        assert_eq!(
            zstd::decode_all(body.as_slice()).expect("valid zstd"),
            expected
        );
    }

    #[test]
    fn default_worker_name_is_valid_for_hopper() {
        let name = default_worker_name();
        assert!(!name.is_empty());
        assert!(name.len() <= MAX_WORKER_NAME_LEN);
        // Mirrors hopper's `validWorkerName`: printable ASCII, no spaces.
        assert!(name.chars().all(|c| c.is_ascii_graphic()));
    }

    /// The `/api/known` probe walks the list like every other call.
    ///
    /// Its failure is safe — an unanswered probe makes the uploader treat
    /// every artifact as missing, and hopper's upsert is idempotent — but it
    /// is not free: that is the probe whose whole job is keeping bytes hopper
    /// already holds off the wire. Stopping at a dead replica would spend the
    /// outage re-uploading the corpus to a primary that was up the whole time.
    #[test]
    fn a_dead_replica_does_not_stop_the_known_probe() {
        // Port 1 refuses immediately, so this measures the decision rather
        // than a timeout.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server address");
        let server = serve_one(
            listener,
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
              Content-Length: 16\r\nConnection: close\r\n\r\n{\"known\":[\"aa\"]}",
        );
        let hopper = hopper_at(&format!("http://127.0.0.1:1,http://{addr}"));
        let known = hopper.known(&["aa", "bb"]);
        server.join().expect("server thread");
        assert!(known.contains("aa"), "the primary's answer was discarded");
        assert!(!known.contains("bb"), "hopper did not claim to hold bb");
    }

    #[test]
    fn upload_sends_configured_bearer_token() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server address");
        let server = serve_one(listener, NO_CONTENT);
        let mut hopper = hopper_at(&format!("http://{addr}"));
        hopper.token = Some("test-secret");
        let art = artifact(&"a".repeat(64), 1, ArtifactBytes::File(PathBuf::new()));
        assert!(hopper.upload(&art, None));
        let request = String::from_utf8_lossy(&server.join().expect("server thread")).into_owned();
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-secret\r\n"),
            "request headers: {request}"
        );
    }

    /// A file-backed artifact streams from disk with its length declared up
    /// front, behind the provenance part hopper's handler reads first.
    #[test]
    fn an_artifact_streams_from_disk_after_its_provenance() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("payload");
        std::fs::write(&path, b"the artifact bytes").expect("write");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server address");
        let server = serve_one(listener, NO_CONTENT);
        let hopper = hopper_at(&format!("http://{addr}"));
        let art = artifact(&"b".repeat(64), 18, ArtifactBytes::File(path.clone()));
        assert!(hopper.upload(&art, Some(&Body::File(path))));
        let request = server.join().expect("server thread");
        let text = String::from_utf8_lossy(&request);
        assert!(
            text.to_ascii_lowercase().contains("content-length:"),
            "{text}"
        );
        let provenance = text.find(r#"name="provenance""#).expect("provenance part");
        let file = text
            .find(r#"name="file"; filename="x.bin""#)
            .expect("file part");
        assert!(provenance < file, "provenance must precede the file");
        assert!(text.contains("the artifact bytes"), "{text}");
    }

    /// An artifact over `HOPPER_MAX_UPLOAD_ARTIFACT_BYTES` must never reach
    /// `/api/upload` at all — hopper would reject it outright, and the
    /// connection-drop that rejection produces looks identical to a real
    /// network fault (see the constant's doc comment). `reconcile` is expected
    /// to skip straight past it after the `/api/known` probe reports it
    /// missing, rather than attempting and retrying a doomed upload.
    #[test]
    fn oversized_artifact_skips_the_byte_upload() {
        let known_listener = TcpListener::bind("127.0.0.1:0").expect("bind known server");
        let known_addr = known_listener.local_addr().expect("known address");
        let known_server = serve_one(
            known_listener,
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
              Content-Length: 12\r\nConnection: close\r\n\r\n{\"known\":[]}",
        );

        // Bound but never accepted: a wrongly-attempted byte upload would
        // connect here, which the accept() below catches. Nonblocking so a
        // bug that skips the size check fails the assertion instead of
        // hanging the test for the full upload retry budget.
        let upload_listener = TcpListener::bind("127.0.0.1:0").expect("bind upload server");
        let upload_addr = upload_listener.local_addr().expect("upload address");
        upload_listener
            .set_nonblocking(true)
            .expect("nonblocking upload listener");

        let mut hopper = hopper_at(&format!("http://{known_addr}"));
        hopper.upload = Route::new(&[format!("http://{upload_addr}")], "/api/upload");
        let art = artifact(
            &"d".repeat(64),
            HOPPER_MAX_UPLOAD_ARTIFACT_BYTES + 1,
            ArtifactBytes::File(PathBuf::from("/nonexistent/huge.bin")),
        );
        hopper.reconcile(None, &mut HashSet::new(), vec![art]);
        known_server.join().expect("known server thread");

        match upload_listener.accept() {
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {} // nothing connected — correct
            Ok(_) => panic!("oversized artifact triggered a byte upload attempt"),
            Err(e) => panic!("unexpected accept error: {e}"),
        }
    }

    /// A job the queue refuses must leave no phantom behind in `pending`: the
    /// counter is what tells "nothing to file" from "filing is stuck".
    #[test]
    fn a_refused_job_does_not_leak_the_pending_count() {
        let (tx, jobs) = queue::channel::<Job>(UPLOAD_QUEUE_DEPTH);
        drop(jobs);
        let uploader = Uploader {
            tx: Some(tx),
            worker: None,
            worker_name: "test".to_owned(),
            pending: Arc::default(),
            tally: Arc::default(),
        };
        let dep = crate::engine::DepResult {
            sha256: "c".repeat(64),
            locator: "pkg:npm/x@1".to_string(),
            url: "https://example/x-1.tgz".to_string(),
            size: 1,
            provenance: None,
            verdict: None,
            members: crate::engine::MemberEvals::new(),
            report: None,
        };
        uploader.submit_dependencies(vec![dep], "v", "t");
        uploader.submit_artifacts(vec![artifact(
            &"e".repeat(64),
            1,
            ArtifactBytes::File(PathBuf::new()),
        )]);
        assert_eq!(uploader.stats().pending, 0);
    }

    /// A re-fetch recovers bytes the blob cache evicted, but a locator is not
    /// always a pin — a versionless PURL re-resolves to today's `latest`, and a
    /// tag can move. Only bytes that still hash to the analyzed digest may be
    /// uploaded under it; anything else would file one artifact's content under
    /// another's identity, which is worse than the absence being repaired.
    #[test]
    fn only_bytes_matching_the_analyzed_digest_are_uploaded() {
        let bytes = b"the exact bytes the verdict was computed over";
        let sha = {
            use sha2::{Digest as _, Sha256};
            format!("{:x}", Sha256::digest(bytes))
        };
        assert!(
            bytes_match_digest(bytes, &sha, "pkg:npm/x@1"),
            "the analyzed bytes must be accepted"
        );
        assert!(
            !bytes_match_digest(b"different bytes at the same locator", &sha, "pkg:npm/x@1"),
            "a moved tag or re-resolved range must not be filed under the old digest"
        );
        assert!(
            !bytes_match_digest(bytes, &"0".repeat(64), "pkg:npm/x@1"),
            "an unrelated digest must not accept these bytes"
        );
        // Empty content hashes to a real, well-known digest rather than to
        // nothing, so a truncated or zero-length re-fetch is a mismatch and not
        // an accidental pass.
        assert!(!bytes_match_digest(b"", &sha, "pkg:npm/x@1"));
    }

    /// A dependency's upload artifact loads its bytes lazily from the fetch cache
    /// (never eagerly), derives its stored filename from the resolved URL, and is
    /// marked backfillable so its captured registry provenance lands even when
    /// hopper already holds the bytes. Built here from an *unevaluated*
    /// dependency: the artifact is independent of the verdict, so bytes and
    /// provenance reach hopper even when scan has no verdict to post for them.
    /// The unsupported PURL proves artifact construction does not perform a
    /// registry lookup.
    #[test]
    fn dep_artifact_loads_bytes_lazily_and_is_backfillable() {
        let dep = crate::engine::DepResult {
            sha256: "c".repeat(64),
            locator: "pkg:bogus/x@1".to_string(),
            url: "https://example/x-1.tgz".to_string(),
            size: 99,
            provenance: Some(crate::provenance::RegistryProvenance::from_record_sources(
                fletch::Registry {
                    ecosystem: "bogus".to_string(),
                    name: "x".to_string(),
                    version: "1".to_string(),
                    ..fletch::Registry::default()
                },
                &[fletch::fetch::RecordedSource {
                    url: "https://registry.example/x".to_string(),
                    status: 200,
                    content_type: Some("application/json".to_string()),
                    size: 20,
                    bytes: Some(br#"{"provider_only":42}"#.to_vec()),
                }],
            )),
            verdict: None,
            members: crate::engine::MemberEvals::new(),
            report: None,
        };
        let art = dep_artifact(&dep, "scan+test", "2026-06-28T00:00:00Z").expect("sidecar builds");
        assert_eq!(art.sha256, "c".repeat(64));
        assert_eq!(art.size, 99);
        assert_eq!(
            art.filename, "x-1.tgz",
            "filename derived from the fetch URL"
        );
        assert!(art.backfill, "a dependency's provenance is backfillable");
        let sidecar: serde_json::Value = serde_json::from_slice(&art.sidecar).unwrap();
        assert_eq!(
            sidecar["registry"]["raw"][0]["body"]["provider_only"], 42,
            "upload uses the captured provider snapshot"
        );
        assert_eq!(sidecar["package"]["purl"], "pkg:bogus/x@1");
        assert!(
            matches!(&art.bytes, ArtifactBytes::Cached { locator } if locator == "pkg:bogus/x@1"),
            "bytes load lazily from the cache by locator",
        );
    }
}
