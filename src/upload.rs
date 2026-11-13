//! Best-effort upload of scan results to a hopper instance.
//!
//! Speaks hopper's `/v1/work/result` contract — the `ResultPayload` wire shape, a
//! zstd-compressed envelope — that its pull-based workers speak, but driven
//! by a local `scan path` run instead of a poll loop. `scan path --hopper=<url>`
//! uses it to *renew* a sample hopper has already ingested with this build's
//! traits and model: hopper's `/v1/work/result` is a lease-free `UPDATE ... WHERE
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
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use reqwest::blocking::multipart::{Form, Part};
use serde::{Deserialize, Serialize};

mod queue;

use crate::engine::{ScanResult, UploadDoc, UploadStamp};
use crate::hopper::{Route, bearer, endpoints, error_chain};
use crate::interpret::Interpretation;
use crate::retry::{backoff, fuzz, is_transient, parse_retry_after};

/// zstd's default level. Cleave reports are large, highly repetitive JSON that
/// zstd shrinks 3-5x on the wire; the compression cost is dwarfed by the
/// analysis that produced the payload.
const ZSTD_RESULT_LEVEL: i32 = 3;

/// The media type of a version 8 result body: JSON lines.
pub const RESULT_CONTENT_TYPE: &str = "application/x-ndjson";

/// Bound on resident jobs buffered ahead of the uploader thread. Overflow is
/// spooled to temporary files rather than blocking analysis or retaining large
/// reports in RAM. Shutdown still drains every accepted job.
const UPLOAD_QUEUE_DEPTH: usize = 16;

/// Match Hopper's three reserved renewal slots without occupying analysis lanes.
const RENEW_WORKERS: usize = 3;

#[derive(Debug, Serialize, Deserialize)]
struct Renewal {
    sha256: String,
    purl: Option<String>,
    body: EncodedResult,
}

/// Cap on the uploader's reconciled-sha dedup set (~64-byte hex strings; the cap
/// bounds it near 10 MB). See the clear in the uploader loop.
const SEEN_SHAS_MAX: usize = 100_000;

/// Per-attempt request timeouts for `/v1/intake/upload`, separate from
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
/// land means the artifact never enters the corpus at all. A queue worker's
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
/// slots are saturated, so a budget of a few attempts over seconds is not a
/// retry so much as a coin flip. But results renew one at a time on one
/// thread, and the CLI waits for the queue to drain before it exits: against a
/// hopper that is down rather than busy, every queued result spends this whole
/// budget, so it is a minute, not the length of an outage.
const RENEW_BUDGET: Duration = Duration::from_secs(60);

/// The least time one renewal attempt is given. Each attempt may run only
/// until the budget ends, so the last cannot outlast it by a whole request
/// timeout; but one started a moment before the end still gets long enough
/// to land.
const RENEW_MIN_ATTEMPT: Duration = Duration::from_secs(10);

/// Hopper's `validWorkerName` cap (`maxWorkerNameLen` in api.go).
const MAX_WORKER_NAME_LEN: usize = 64;

/// Compress a version 8 result body for upload, falling back to sending it
/// as it is when compression fails.
fn compress_body(body: Vec<u8>, sha256: &str) -> EncodedResult {
    match zstd::encode_all(body.as_slice(), ZSTD_RESULT_LEVEL) {
        Ok(bytes) => EncodedResult { bytes, zstd: true },
        Err(e) => {
            tracing::warn!(sha256 = %sha256, error = %e, "upload: zstd compress failed; sending uncompressed");
            EncodedResult {
                bytes: body,
                zstd: false,
            }
        }
    }
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

/// Encode one result's version 8 body under `sha256` and compress it. A
/// body that cannot be written is counted as a lost renewal.
fn prepare_result(
    doc: &UploadDoc,
    sha256: &str,
    llm: Option<&Interpretation>,
    stamp: &UploadStamp<'_>,
    tally: &Tally,
) -> Option<EncodedResult> {
    let Some(body) = doc.encode(sha256, llm, stamp) else {
        tally.failed.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    Some(compress_body(body, sha256))
}

/// `result`'s version 8 body under `sha256`, stamped with `worker` and the
/// analysis time `ms`, and compressed. A result classified without an upload body, or whose body
/// cannot be written, is counted as a lost renewal.
fn encode_result(
    result: &ScanResult,
    sha256: &str,
    worker: &str,
    ms: Option<u64>,
    tally: &Tally,
) -> Option<EncodedResult> {
    let Some(doc) = &result.upload else {
        tally.failed.fetch_add(1, Ordering::Relaxed);
        tracing::error!(sha256 = %sha256, "upload: result was not classified for upload");
        return None;
    };
    let at = crate::engine::analyzed_at(&result.analyzed_at);
    let stamp = UploadStamp {
        at: &at,
        model: &result.version,
        worker: Some(worker),
        ms,
    };
    prepare_result(doc, sha256, result.interpretation.as_ref(), &stamp, tally)
}

/// The body hopper's `/v1/work/result` takes for `result`, filed under `sha256`,
/// naming `worker` as its uploader and `ms` as the analysis time when the
/// caller timed it: version 8 JSON lines
/// ([`RESULT_CONTENT_TYPE`]), with the `Content-Encoding` they were
/// compressed with, if any. For callers that post results themselves, so
/// they send exactly what [`Uploader::submit`] sends. `None`, logged, when
/// the result was not classified for upload or its body cannot be written.
#[must_use]
pub fn encode_result_body(
    result: &ScanResult,
    sha256: &str,
    worker: &str,
    ms: Option<u64>,
) -> Option<(Vec<u8>, Option<&'static str>)> {
    let encoded = encode_result(result, sha256, worker, ms, &Tally::default())?;
    Some((encoded.bytes, encoded.zstd.then_some("zstd")))
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
    let at = crate::engine::analyzed_at(analyzed_at);
    let stamp = UploadStamp {
        at: &at,
        model: version,
        worker: Some(worker),
        ms: None,
    };
    deps.into_iter()
        .map(|dep| {
            let artifact = dep_artifact(&dep, &collector, analyzed_at);
            // No body without a verdict: scan posts none it did not compute.
            let result = dep
                .verdict
                .and(dep.upload.as_ref())
                .and_then(|doc| prepare_result(doc, &dep.sha256, None, &stamp, tally));
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
    /// so it is counted where an operator will see it, not only logged.
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

/// Why an artifact could not be handed to the uploader. Its message names
/// local paths: log it, do not serve it.
#[derive(Debug)]
pub enum StageError {
    /// The staging directory could not be created.
    Create(std::io::Error),
    /// A file could not be copied into the staging directory.
    Copy(PathBuf, std::io::Error),
    /// Caller-supplied bytes could not be written to the staging directory.
    Write(std::io::Error),
    /// The uploader has stopped, or its queue refused the job.
    Queue,
}

impl std::fmt::Display for StageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create(e) => write!(f, "create durable upload staging directory: {e}"),
            Self::Copy(path, e) => write!(
                f,
                "copy {} into durable upload staging: {e}",
                path.display()
            ),
            Self::Write(e) => write!(f, "write durable upload staging file: {e}"),
            Self::Queue => f.write_str("could not queue durable artifacts for hopper"),
        }
    }
}

impl std::error::Error for StageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Create(e) | Self::Copy(_, e) | Self::Write(e) => Some(e),
            Self::Queue => None,
        }
    }
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

    /// Queue `result` for upload, filed under `sha256`. A result classified
    /// without an upload body, or a failed handoff, is logged and counted.
    pub fn submit(&self, sha256: String, purl: Option<String>, result: &ScanResult) {
        let Some(body) = encode_result(result, &sha256, &self.worker_name, None, &self.tally)
        else {
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
    /// before `/v1/intake/held` says hopper needs it.
    ///
    /// # Errors
    /// Returns an error when staging fails or the uploader has stopped.
    pub fn submit_artifacts_durable(
        &self,
        mut artifacts: Vec<UploadArtifact>,
    ) -> Result<(), StageError> {
        if artifacts.is_empty() {
            return Ok(());
        }

        let staging = tempfile::Builder::new()
            .prefix("scan-upload-")
            .tempdir()
            .map_err(StageError::Create)?;
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
            std::fs::copy(source, &destination)
                .map_err(|error| StageError::Copy(source.clone(), error))?;
            artifact.bytes = ArtifactBytes::File(destination);
        }

        // `keep` transfers directory ownership from TempDir to the queued job.
        // The paths stored in `artifacts` remain valid after this call.
        let staging_path = staging.keep();
        if self.enqueue_artifacts(artifacts, vec![staging_path]) {
            Ok(())
        } else {
            Err(StageError::Queue)
        }
    }

    /// Persist one already-built artifact from bytes supplied by the caller,
    /// then queue it for reconciliation. This is used when Scan has a local
    /// verdict and can answer immediately, but Hopper may not yet have the
    /// artifact row that the verdict belongs to.
    ///
    /// # Errors
    /// Returns an error when staging fails or the uploader has stopped.
    pub fn submit_artifact_bytes_durable(
        &self,
        mut artifact: UploadArtifact,
        bytes: &[u8],
    ) -> Result<(), StageError> {
        let staging = tempfile::Builder::new()
            .prefix("scan-upload-")
            .tempdir()
            .map_err(StageError::Create)?;
        let destination = staging.path().join("artifact");
        std::fs::write(&destination, bytes).map_err(StageError::Write)?;
        artifact.bytes = ArtifactBytes::File(destination);

        let staging_path = staging.keep();
        if self.enqueue_artifacts(vec![artifact], vec![staging_path]) {
            Ok(())
        } else {
            Err(StageError::Queue)
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
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            tracing::error!("upload: uploader thread panicked; queued uploads were lost");
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
            result: Route::new(&bases, "/v1/work/result"),
            known: Route::new(&bases, "/v1/intake/held"),
            upload: Route::new(&bases, "/v1/intake/upload"),
            token: crate::hopper::token(),
            worker,
            tally,
        }
    }

    /// The uploader thread's loop: handle every queued job until the sender
    /// side is dropped.
    fn serve(&self, jobs: queue::Receiver<Job>, pending: &AtomicUsize) {
        std::thread::scope(|scope| {
            let mut senders = Vec::new();
            for lane in 0..RENEW_WORKERS {
                // One active and one resident queued payload per worker.
                // Overflow spills to disk: a stalled digest's FIFO lane must
                // not prevent the coordinator from feeding the other lanes.
                let (tx, rx) = queue::channel::<Renewal>(1);
                match std::thread::Builder::new()
                    .name(format!("scan-renew-{lane}"))
                    .spawn_scoped(scope, move || {
                        for job in rx {
                            match job {
                                Ok(job) => self.renew_encoded(&job.sha256, job.purl.as_deref(), job.body),
                                Err(error) => {
                                    self.tally.failed.fetch_add(1, Ordering::Relaxed);
                                    tracing::error!(%error, "upload: could not read spooled renewal");
                                }
                            }
                        }
                    }) {
                    Ok(_) => senders.push(tx),
                    Err(error) => tracing::warn!(%error, "upload: could not start renewal worker"),
                }
            }
            let publish = |job: Renewal| {
                if senders.is_empty() {
                    self.renew_encoded(&job.sha256, job.purl.as_deref(), job.body);
                    return;
                }
                // Repeated updates for one digest stay on one FIFO lane. A
                // slow older verdict must never overwrite its newer successor.
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                job.sha256.hash(&mut hash);
                // The remainder is below `senders.len()`, so it fits a usize.
                let lane =
                    usize::try_from(hash.finish() % senders.len() as u64).unwrap_or_default();
                if let Err((job, error)) = senders[lane].send(job) {
                    tracing::warn!(%error, "upload: renewal queue unavailable; sending inline");
                    self.renew_encoded(&job.sha256, job.purl.as_deref(), job.body);
                }
            };
            self.serve_jobs(jobs, pending, &publish);
            // Scope exit joins all workers after closing their channels, so
            // Uploader::drop still waits for every accepted verdict.
            drop(senders);
        });
    }

    fn serve_jobs(
        &self,
        jobs: queue::Receiver<Job>,
        pending: &AtomicUsize,
        publish: &impl Fn(Renewal),
    ) {
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
                // Posted unconditionally. Asking first would cost a round trip
                // on the same renew lane the post uses, to avoid a write hopper
                // declines in one indexed read — and the answer would age the
                // moment it arrived, because only the store is ordered against
                // the other producers pushing the same dependency.
                Job::Result { sha256, purl, body } => publish(Renewal { sha256, purl, body }),
                Job::Artifacts {
                    artifacts,
                    cleanup_dirs,
                } => {
                    self.reconcile(cache.as_ref(), &mut seen, artifacts);
                    cleanup_upload_dirs(cleanup_dirs);
                }
                Job::Dependencies { deps } => {
                    self.sync_prepared_dependencies(cache.as_ref(), &mut seen, deps, publish);
                }
            }
        }
    }

    /// Reconcile a batch of artifacts against hopper: negotiate which it's
    /// missing (one `/v1/intake/held` round-trip), then upload only those — bytes
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
        publish: &impl Fn(Renewal),
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
        // verdict UPDATE (hopper's `/v1/work/result` no-ops on a missing row). The
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
        // Every dependency's verdict is posted. Asking hopper first whether it
        // is current could only answer for the instant it was asked: the
        // popular dependencies such a probe would spare are exactly the ones
        // several scans push at once. Hopper's store settles it in one indexed
        // read, because the store is the only place ordered against the other
        // producers.
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
            let purl = dep.locator.starts_with("pkg:").then_some(dep.locator);
            publish(Renewal {
                sha256: dep.sha256,
                purl,
                body,
            });
        }
    }

    /// POST the batch existence probe (`/v1/intake/held`) and return the digests
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
            let resp = crate::retry::send(
                "hopper known probe",
                crate::hopper::LOOKUP_RETRY_BUDGET,
                |_| bearer(self.http.post(url), self.token).json(&KnownRequest { sha256: shas }),
            );
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

    /// The multipart `/v1/intake/upload` body: the provenance part first, as hopper's
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
                    if !is_transient(status) {
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
        let _span = tracing::info_span!("renew", sha256, purl).entered();

        let started = Instant::now();
        let mut attempts = 0;
        let outcome = crate::retry::send("hopper result renewal", RENEW_BUDGET, |attempt| {
            attempts = attempt;
            // Each attempt runs only until the budget ends, but never shorter
            // than RENEW_MIN_ATTEMPT, so a late one still has time to land.
            let remaining = RENEW_BUDGET.saturating_sub(started.elapsed());
            let request = bearer(self.http.post(self.result.at(attempt - 1)), self.token)
                .timeout(remaining.max(RENEW_MIN_ATTEMPT))
                .header(reqwest::header::CONTENT_TYPE, RESULT_CONTENT_TYPE)
                // Claim hopper's reserved lane: this renewal is one-shot, and the
                // caller is already holding the verdict in its cache.
                .header(HOPPER_LANE_HEADER, HOPPER_LANE_RENEW)
                .body(body.clone());
            match encoding {
                Some(enc) => request.header(reqwest::header::CONTENT_ENCODING, enc),
                None => request,
            }
        });
        match outcome {
            Ok(resp) if resp.status().is_success() => {
                tracing::info!(
                    sha256 = %sha256,
                    purl,
                    attempts,
                    waited_ms = started.elapsed().as_millis(),
                    "upload: result renewed on hopper",
                );
                self.tally.uploaded.fetch_add(1, Ordering::Relaxed);
            }
            Ok(resp) if !is_transient(resp.status()) => {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                self.tally.failed.fetch_add(1, Ordering::Relaxed);
                tracing::error!(sha256 = %sha256, purl, %status, body = %body_excerpt(&body), "upload: hopper rejected result; not retrying");
            }
            Ok(resp) => {
                self.tally.failed.fetch_add(1, Ordering::Relaxed);
                tracing::error!(
                    sha256 = %sha256,
                    purl,
                    status = %resp.status(),
                    attempts,
                    budget_s = RENEW_BUDGET.as_secs(),
                    "upload: failed to renew result on hopper; giving up after the renewal budget",
                );
            }
            Err(e) => {
                self.tally.failed.fetch_add(1, Ordering::Relaxed);
                tracing::error!(
                    sha256 = %sha256,
                    purl,
                    error = %error_chain(&e),
                    attempts,
                    budget_s = RENEW_BUDGET.as_secs(),
                    "upload: failed to renew result on hopper; giving up after the renewal budget",
                );
            }
        }
    }
}

/// Bound an untrusted response body and escape its control and bidi
/// characters before it reaches a log, where an ANSI sequence would rewrite
/// the operator's terminal.
fn body_excerpt(body: &str) -> String {
    const MAX_CHARS: usize = 512;
    let end = body
        .char_indices()
        .nth(MAX_CHARS)
        .map_or(body.len(), |(at, _)| at);
    let mut excerpt = crate::output::tty_text(body.get(..end).unwrap_or(body)).into_owned();
    if end < body.len() {
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
/// that has no [`Uploader`] (a pull-based queue worker). Dedups within this call,
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
    hopper.sync_prepared_dependencies(cache, &mut HashSet::new(), deps, &|job| {
        hopper.renew_encoded(&job.sha256, job.purl.as_deref(), job.body);
    });
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

#[cfg(test)]
mod tests;
