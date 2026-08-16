//! Best-effort upload of scan results to a hopper instance.
//!
//! Mirrors the pull-based worker's `/api/result` contract — the same
//! [`ResultPayload`] wire shape, the same zstd-compressed envelope — but driven
//! by a local `scan path` run instead of a poll loop. `scan path --hopper=<url>`
//! uses it to *renew* a sample hopper has already ingested with this build's
//! traits and model: hopper's `/api/result` is a lease-free `UPDATE ... WHERE
//! sha256 = ?`, so posting a result for an already-scanned SHA replaces its
//! stored cleave/litmus envelope (and an unknown SHA is a harmless no-op).
//!
//! Uploads run on a dedicated thread so blocking network I/O never stalls the
//! analysis pool, and every failure degrades to a logged warning — a scan never
//! fails because an upload did.

use std::collections::HashSet;
use std::path::PathBuf;
use std::thread::JoinHandle;
use std::time::Duration;

use serde::Serialize;

use crate::engine::ScanResultEnvelope;

/// Hopper bounds the decompressed result body at 512 MiB (`maxResultBodyBytes`
/// in hopper's api.go); a larger document is truncated mid-stream and rejected
/// as invalid JSON, so an over-limit report is sent ML-verdict-only.
const HOPPER_MAX_RESULT_BODY_BYTES: usize = 512 << 20;

/// zstd's default level. Cleave reports are large, highly repetitive JSON that
/// zstd shrinks 3-5x on the wire; the compression cost is dwarfed by the
/// analysis that produced the payload.
const ZSTD_RESULT_LEVEL: i32 = 3;

/// Bound on results buffered ahead of the uploader thread. A small queue applies
/// backpressure: a slow hopper throttles the scan rather than letting envelopes
/// (each up to hundreds of KB) accumulate unbounded in memory.
const UPLOAD_QUEUE_DEPTH: usize = 16;

/// Cap on the uploader's reconciled-sha dedup set (~64-byte hex strings; the cap
/// bounds it near 10 MB). See the clear in the uploader loop.
const SEEN_SHAS_MAX: usize = 100_000;

/// Per-attempt request timeouts, escalating. hopper's slow spells are usually
/// brief, so early attempts fail fast and get another try rather than pinning
/// the uploader for the full 120s ceiling each time; the final attempt keeps
/// the old ceiling so a genuinely slow-but-alive hopper still lands the write.
/// One entry per attempt — its length is the retry budget. Short, unlike the
/// worker's 20-minute budget: a renew is idempotent and the operator can
/// simply re-run.
const ATTEMPT_TIMEOUTS: [Duration; 4] = [
    Duration::from_secs(15),
    Duration::from_secs(30),
    Duration::from_secs(60),
    Duration::from_secs(120),
];

/// Request timeout per POST. Matches the worker so a wedged hopper can't pin an
/// uploader thread indefinitely.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Hopper's `validWorkerName` cap (`maxWorkerNameLen` in api.go).
const MAX_WORKER_NAME_LEN: usize = 64;

/// The JSON body POSTed to hopper's `/api/result`. The `{ml, llm?, raw}`
/// envelope is flattened onto the payload so the wire form is
/// `{sha256, worker, duration_ms, ml, llm, raw}` — byte-for-byte the shape the
/// pull-based worker sends, so hopper handles both identically.
#[derive(Serialize)]
pub(crate) struct ResultPayload {
    pub sha256: String,
    pub worker: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub duration_ms: i64,
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
pub(crate) fn encode_result_body(
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
        // Empty the cleave report but keep the ml/llm verdict. `{}` (not null)
        // mirrors the envelope litmus emits when there is no cleave report, so
        // the dropped-raw form stays a structurally valid envelope.
        if let Some(envelope) = payload.envelope.as_mut() {
            envelope.raw = cleave::types::CompactReport::default();
        }
        serialize(&payload, sha256)?
    } else {
        json
    };
    match zstd::encode_all(json.as_slice(), ZSTD_RESULT_LEVEL) {
        Ok(compressed) => Some((compressed, Some("zstd"))),
        Err(e) => {
            tracing::warn!(sha256 = %sha256, error = %e, "upload: zstd compress failed; sending uncompressed");
            Some((json, None))
        }
    }
}

fn serialize(payload: &ResultPayload, sha256: &str) -> Option<Vec<u8>> {
    match serde_json::to_vec(payload) {
        Ok(json) => Some(json),
        Err(e) => {
            tracing::error!(sha256 = %sha256, error = %e, "upload: serialize failed");
            None
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
#[derive(Debug)]
pub enum ArtifactBytes {
    /// The scanned file itself, read from disk.
    File(PathBuf),
    /// A fetched dependency, loaded from fletch's blob cache by its locator.
    Cached {
        /// The reference locator (PURL/URL) the cache keys the bytes under.
        locator: String,
    },
}

/// An artifact (the scanned file or a fetched dependency archive) offered to
/// hopper, with the provenance to record if hopper doesn't already have it.
/// Bytes are loaded lazily so the common "hopper already has it" case moves only
/// the 64-char sha across the wire, never the payload.
#[derive(Debug)]
pub struct UploadArtifact {
    /// SHA-256 of the artifact's bytes — the negotiation and storage key.
    pub sha256: String,
    /// Size of the artifact's bytes, recorded in the provenance sidecar.
    pub size: u64,
    /// Filename hopper stores and sniffs the type from.
    pub filename: String,
    /// Where to load the bytes from, only if hopper turns out to need them.
    pub bytes: ArtifactBytes,
    /// Pre-serialized hopper `Sidecar` JSON (see [`crate::provenance::build_sidecar`]).
    pub sidecar: Vec<u8>,
    /// Whether this artifact's provenance is worth backfilling onto a sample
    /// hopper already has the bytes for — true for fetched dependencies and
    /// map-backed roots carrying registry data, false for a plain local root
    /// whose sidecar contains only artifact + fetch identity.
    pub backfill: bool,
}

/// Work handed to the background uploader thread. Artifacts are reconciled before
/// a result so a never-seen top-level file's row exists before its verdict POST.
#[derive(Debug)]
enum Job {
    /// Renew a verdict on hopper (the original `--upload` behavior). Boxed: the
    /// envelope dwarfs the other variant, so an unboxed enum would bloat every
    /// queued job to its size.
    Result {
        sha256: String,
        envelope: Box<ScanResultEnvelope>,
    },
    /// Ensure hopper has these artifacts' bytes+provenance, uploading only the
    /// ones it's missing.
    Artifacts(Vec<UploadArtifact>),
    /// Mirror fetched dependencies into hopper as their own samples: bytes (only
    /// if missing) + provenance, then the verdict scan computed for each.
    Dependencies {
        deps: Vec<crate::engine::DepResult>,
        /// Model version and analysis time stamped on each dependency's verdict,
        /// carried from the parent result so the `ml` section is self-describing.
        version: String,
        analyzed_at: String,
    },
}

/// Background uploader that POSTs scan results to hopper without blocking the
/// analysis threads. Created per `scan path --hopper` run; results are handed off
/// via [`Uploader::submit`] and flushed when the uploader is dropped.
#[derive(Debug)]
pub struct Uploader {
    /// `None` once flushed, or when the uploader thread failed to spawn (uploads
    /// then degrade to silent no-ops rather than failing the scan).
    tx: Option<std::sync::mpsc::SyncSender<Job>>,
    worker: Option<JoinHandle<()>>,
}

impl Uploader {
    /// Start a background uploader targeting `hopper_url`, tagging every result
    /// with `worker`. Spawn failure is non-fatal: the returned uploader silently
    /// drops submissions so the scan still completes.
    #[must_use]
    pub fn new(hopper_url: &str, worker: String) -> Self {
        let base = hopper_url.trim_end_matches('/');
        let result_url = format!("{base}/api/result");
        let known_url = format!("{base}/api/known");
        let upload_url = format!("{base}/api/upload");
        let client = reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        let (tx, rx) = std::sync::mpsc::sync_channel::<Job>(UPLOAD_QUEUE_DEPTH);
        let handle = std::thread::Builder::new()
            .name("scan-upload".into())
            .spawn(move || {
                // The same default blob cache scan fetched dependencies into, so a
                // missing dep's bytes are loaded locally rather than re-fetched.
                let cache = fletch::fetch::BlobCache::open().ok();
                // Shas reconciled this run, so a dependency shared by many scanned
                // files is negotiated and uploaded at most once.
                let mut seen: HashSet<String> = HashSet::new();
                for job in rx {
                    // Bound the dedup set: a long-lived `serve --hopper` process
                    // reconciles an unbounded stream of unique shas, and this set
                    // otherwise grows forever. Clearing past the cap only costs a
                    // redundant /known round-trip for shas negotiated earlier.
                    if seen.len() >= SEEN_SHAS_MAX {
                        seen.clear();
                    }
                    match job {
                        Job::Result { sha256, envelope } => {
                            post_one(&client, &result_url, &worker, &sha256, *envelope);
                        }
                        Job::Artifacts(artifacts) => {
                            reconcile_artifacts(
                                &client,
                                &known_url,
                                &upload_url,
                                cache.as_ref(),
                                &mut seen,
                                artifacts,
                            );
                        }
                        Job::Dependencies {
                            deps,
                            version,
                            analyzed_at,
                        } => {
                            sync_dependencies(
                                &client,
                                &known_url,
                                &upload_url,
                                &result_url,
                                &worker,
                                &version,
                                &analyzed_at,
                                cache.as_ref(),
                                &mut seen,
                                deps,
                            );
                        }
                    }
                }
            });
        match handle {
            Ok(worker) => {
                tracing::info!(hopper = %hopper_url, "upload: renewing results on hopper");
                Self {
                    tx: Some(tx),
                    worker: Some(worker),
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "upload: failed to spawn uploader thread; uploads disabled");
                Self {
                    tx: None,
                    worker: None,
                }
            }
        }
    }

    /// Queue a result for upload. Blocks briefly when the upload queue is full
    /// (backpressure); a closed channel drops the result silently.
    pub fn submit(&self, sha256: String, envelope: ScanResultEnvelope) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Job::Result {
                sha256,
                envelope: Box::new(envelope),
            });
        }
    }

    /// Queue artifacts (the scanned file and any fetched dependency archives) for
    /// content reconciliation: hopper is asked which it lacks, and only those are
    /// uploaded with their provenance. Submit before the matching [`submit`] so a
    /// new top-level file's row exists before its verdict lands.
    pub fn submit_artifacts(&self, artifacts: Vec<UploadArtifact>) {
        if artifacts.is_empty() {
            return;
        }
        if let Some(tx) = &self.tx {
            let _ = tx.send(Job::Artifacts(artifacts));
        }
    }

    /// Queue fetched dependencies to mirror into hopper as their own samples:
    /// bytes (only if hopper lacks them) + provenance, then each dependency's
    /// verdict. Submit after the root [`submit_artifacts`] and before the root
    /// [`submit`] so the dependencies' rows exist before any verdict — the root's
    /// or their own — lands.
    pub fn submit_dependencies(
        &self,
        deps: Vec<crate::engine::DepResult>,
        version: String,
        analyzed_at: String,
    ) {
        if deps.is_empty() {
            return;
        }
        if let Some(tx) = &self.tx {
            let _ = tx.send(Job::Dependencies {
                deps,
                version,
                analyzed_at,
            });
        }
    }
}

impl Drop for Uploader {
    /// Stop accepting new results and wait for in-flight uploads to finish, so a
    /// scan's results are fully renewed before the process exits.
    fn drop(&mut self) {
        // Dropping the sender ends the thread's `for job in rx` loop.
        self.tx = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Flatten an error and its `source()` chain into one message. reqwest's
/// top-level `Display` is just "error sending request for url (...)"; the real
/// cause (connection refused, DNS failure, timeout) lives one or more links down
/// the chain, so log the whole chain to make a failed upload diagnosable.
pub(crate) fn error_chain(err: &dyn std::error::Error) -> String {
    use std::fmt::Write;
    let mut out = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        let _ = write!(out, ": {cause}");
        source = cause.source();
    }
    out
}

/// Reconcile a batch of artifacts against hopper: negotiate which it's missing
/// (one `/api/known` round-trip), then upload only those — bytes plus provenance.
/// The `seen` set dedups across batches so a dependency shared by many files is
/// handled once. Best-effort throughout: a failure logs and the scan continues.
fn reconcile_artifacts(
    client: &reqwest::blocking::Client,
    known_url: &str,
    upload_url: &str,
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

    // The only question that gates the expensive byte transfer: which of these
    // does hopper already have? Everything it has, we never send.
    let shas: Vec<&str> = fresh.iter().map(|a| a.sha256.as_str()).collect();
    let known = post_known(client, known_url, &shas);

    for art in fresh {
        if known.contains(&art.sha256) {
            // hopper has the bytes. For a dependency, (re)send its provenance so
            // hopper refreshes the registry snapshot — the bytes never move, only
            // the small sidecar, and hopper preserves the original discovery
            // wrapper, updating just the registry data. Plain local roots set
            // `backfill` false; map-backed roots preserve and refresh theirs.
            if art.backfill {
                upload_provenance_only(client, upload_url, &art);
            } else {
                tracing::debug!(sha256 = %art.sha256, "upload: hopper already has artifact; skipping");
            }
            continue;
        }
        let bytes = match &art.bytes {
            ArtifactBytes::File(path) => std::fs::read(path).ok(),
            ArtifactBytes::Cached { locator } => cache.and_then(|c| c.load(locator)),
        };
        let Some(bytes) = bytes else {
            tracing::warn!(sha256 = %art.sha256, file = %art.filename, "upload: artifact bytes unavailable; skipping");
            continue;
        };
        upload_one(client, upload_url, &art, &bytes);
    }
}

/// Mirror a result's fetched dependencies into a hopper instance from a caller
/// that has no [`Uploader`] (the pull-based worker). Builds the endpoint URLs
/// from `base_url`, dedups within this call, and reconciles bytes + provenance
/// before posting each verdict. Best-effort: every failure is logged, never
/// propagated, so a dependency sync never disturbs the result it followed.
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
    let base = base_url.trim_end_matches('/');
    let result_url = format!("{base}/api/result");
    let known_url = format!("{base}/api/known");
    let upload_url = format!("{base}/api/upload");
    let mut seen = HashSet::new();
    sync_dependencies(
        client,
        &known_url,
        &upload_url,
        &result_url,
        worker,
        version,
        analyzed_at,
        cache,
        &mut seen,
        deps,
    );
}

/// Mirror fetched dependencies into hopper as their own samples. For each
/// dependency not already handled this run: ensure hopper has its bytes (uploaded
/// only when missing) and provenance, then POST the verdict scan already computed
/// for it. Best-effort throughout — a failure logs and the next dependency
/// proceeds, exactly like the artifact reconciliation it builds on.
#[allow(clippy::too_many_arguments)]
fn sync_dependencies(
    client: &reqwest::blocking::Client,
    known_url: &str,
    upload_url: &str,
    result_url: &str,
    worker: &str,
    version: &str,
    analyzed_at: &str,
    cache: Option<&fletch::fetch::BlobCache>,
    seen: &mut HashSet<String>,
    deps: Vec<crate::engine::DepResult>,
) {
    // Each dependency is reconciled and verdict-posted once per run; a dependency
    // shared by many scanned files is handled the first time it is seen.
    let fresh: Vec<crate::engine::DepResult> = deps
        .into_iter()
        .filter(|d| seen.insert(d.sha256.clone()))
        .collect();
    if fresh.is_empty() {
        return;
    }
    let collector = format!("scan+{worker}");
    // Bytes + provenance first, so each dependency's row exists before its verdict
    // UPDATE (hopper's `/api/result` no-ops on a missing row). The local seen set
    // starts empty — the run-level dedup above already removed repeats.
    let artifacts: Vec<UploadArtifact> = fresh
        .iter()
        .map(|d| dep_artifact(d, &collector, analyzed_at))
        .collect();
    let mut local_seen = HashSet::new();
    reconcile_artifacts(
        client,
        known_url,
        upload_url,
        cache,
        &mut local_seen,
        artifacts,
    );
    // Then the verdict for each dependency that has one, keyed by its content
    // sha. A dependency the embedded pass never reached carries none: its bytes
    // and provenance went up above, so hopper holds the artifact and can analyze
    // it, but scan posts no verdict it did not compute. Logged rather than
    // dropped silently — an unevaluated dependency is a coverage gap worth
    // seeing, not a routine skip.
    for dep in fresh {
        let Some(envelope) = crate::engine::dep_envelope(&dep, version, analyzed_at) else {
            tracing::info!(
                sha256 = %dep.sha256,
                locator = %dep.locator,
                "upload: dependency not evaluated; stored for analysis without a verdict"
            );
            continue;
        };
        post_one(client, result_url, worker, &dep.sha256, envelope);
    }
}

/// Build the upload artifact for a fetched dependency. Its bytes load from the
/// fetch blob cache only if hopper needs them; its sidecar uses the exact
/// registry snapshot already captured and analyzed, with no registry/cache
/// lookup on the upload path.
fn dep_artifact(dep: &crate::engine::DepResult, collector: &str, now: &str) -> UploadArtifact {
    let filename = crate::engine::artifact_filename(&dep.url, &dep.locator);
    let purl = dep
        .locator
        .starts_with("pkg:")
        .then_some(dep.locator.as_str());
    let sidecar = if let Some(provenance) = &dep.provenance {
        crate::provenance::build_sidecar_from_provenance(
            &filename,
            &dep.sha256,
            dep.size,
            collector,
            now,
            &dep.url,
            purl.unwrap_or_default(),
            provenance,
        )
    } else {
        crate::provenance::build_sidecar(
            &filename,
            &dep.sha256,
            dep.size,
            collector,
            now,
            &dep.url,
            purl.unwrap_or_default(),
            None,
            &[],
        )
    };
    UploadArtifact {
        sha256: dep.sha256.clone(),
        size: dep.size,
        sidecar,
        filename,
        bytes: ArtifactBytes::Cached {
            locator: dep.locator.clone(),
        },
        backfill: true,
    }
}

/// POST the batch existence probe (`/api/known`) and return the subset hopper
/// already holds. On any failure returns an empty set — the caller then treats
/// every artifact as missing and tries to upload (hopper's upsert is idempotent),
/// which is the safe direction: we never skip a needed upload because the probe
/// failed.
fn post_known(
    client: &reqwest::blocking::Client,
    known_url: &str,
    shas: &[&str],
) -> HashSet<String> {
    #[derive(Serialize)]
    struct KnownRequest<'a> {
        sha256: &'a [&'a str],
    }
    #[derive(serde::Deserialize)]
    struct KnownResponse {
        #[serde(default)]
        known: Vec<String>,
    }
    let resp = client
        .post(known_url)
        .json(&KnownRequest { sha256: shas })
        .send();
    match resp {
        Ok(resp) if resp.status().is_success() => match resp.json::<KnownResponse>() {
            Ok(kr) => kr.known.into_iter().collect(),
            Err(e) => {
                tracing::warn!(error = %error_chain(&e), "upload: known response decode failed");
                HashSet::new()
            }
        },
        Ok(resp) => {
            tracing::warn!(status = %resp.status(), "upload: known probe non-success");
            HashSet::new()
        }
        Err(e) => {
            tracing::warn!(error = %error_chain(&e), "upload: known probe failed");
            HashSet::new()
        }
    }
}

/// Build the multipart provenance part from an artifact's sidecar.
fn provenance_part(art: &UploadArtifact) -> Option<reqwest::blocking::multipart::Part> {
    match reqwest::blocking::multipart::Part::bytes(art.sidecar.clone())
        .mime_str("application/json")
    {
        Ok(part) => Some(part),
        Err(e) => {
            tracing::warn!(sha256 = %art.sha256, error = %error_chain(&e), "upload: provenance part build failed");
            None
        }
    }
}

/// POST a multipart body to `/api/upload` with a short retry, rebuilding the
/// (non-`Clone`) form each attempt via `build_form`. `kind` labels the log lines.
/// Returns `true` on success. Best-effort: a 4xx (other than 408/429) is
/// permanent and stops immediately; the caller logs its own success detail.
fn post_upload(
    client: &reqwest::blocking::Client,
    upload_url: &str,
    sha256: &str,
    kind: &str,
    provenance: &[u8],
    build_form: impl Fn() -> Option<reqwest::blocking::multipart::Form>,
) -> bool {
    let token = std::env::var("HOPPER_UPLOAD_TOKEN")
        .ok()
        .filter(|value| !value.trim().is_empty());
    post_upload_with_token(
        client,
        upload_url,
        sha256,
        kind,
        provenance,
        token.as_deref(),
        build_form,
    )
}

#[allow(clippy::too_many_arguments)]
fn post_upload_with_token(
    client: &reqwest::blocking::Client,
    upload_url: &str,
    sha256: &str,
    kind: &str,
    provenance: &[u8],
    token: Option<&str>,
    build_form: impl Fn() -> Option<reqwest::blocking::multipart::Form>,
) -> bool {
    for (attempt, timeout) in ATTEMPT_TIMEOUTS.into_iter().enumerate() {
        if attempt > 0 {
            std::thread::sleep(Duration::from_secs(1 << (attempt - 1)));
        }
        let Some(form) = build_form() else {
            return false; // part build failed — unrecoverable
        };
        let mut request = client.post(upload_url).timeout(timeout).multipart(form);
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        match request.send() {
            Ok(resp) if resp.status().is_success() => return true,
            Ok(resp) => {
                let status = resp.status();
                if status.is_client_error()
                    && status != reqwest::StatusCode::REQUEST_TIMEOUT
                    && status != reqwest::StatusCode::TOO_MANY_REQUESTS
                {
                    let body = resp.text().unwrap_or_default();
                    tracing::warn!(
                        sha256 = %sha256,
                        kind,
                        %status,
                        body = %body,
                        provenance = %crate::worker::body_excerpt(&String::from_utf8_lossy(provenance)),
                        "upload: rejected by hopper; not retrying"
                    );
                    return false;
                }
                tracing::warn!(sha256 = %sha256, kind, %status, attempt, "upload: non-success response");
            }
            Err(e) => {
                tracing::warn!(sha256 = %sha256, kind, error = %error_chain(&e), attempt, "upload: send failed");
            }
        }
    }
    tracing::warn!(sha256 = %sha256, kind, attempts = ATTEMPT_TIMEOUTS.len(), "upload: giving up after retries");
    false
}

/// Upload one artifact's bytes + provenance via the multipart `/api/upload`. The
/// provenance part precedes the file part, as hopper's handler requires.
fn upload_one(
    client: &reqwest::blocking::Client,
    upload_url: &str,
    art: &UploadArtifact,
    bytes: &[u8],
) {
    use reqwest::blocking::multipart::{Form, Part};
    let ok = post_upload(
        client,
        upload_url,
        &art.sha256,
        "artifact",
        &art.sidecar,
        || {
            Some(Form::new().part("provenance", provenance_part(art)?).part(
                "file",
                Part::bytes(bytes.to_vec()).file_name(art.filename.clone()),
            ))
        },
    );
    if ok {
        tracing::debug!(sha256 = %art.sha256, file = %art.filename, size = art.size, "upload: artifact stored on hopper");
    }
}

/// Backfill a dependency's registry provenance onto a sample hopper already holds
/// the bytes for: the same multipart `/api/upload`, but with only the provenance
/// part (no file). hopper attaches it without moving any bytes.
fn upload_provenance_only(
    client: &reqwest::blocking::Client,
    upload_url: &str,
    art: &UploadArtifact,
) {
    use reqwest::blocking::multipart::Form;
    let ok = post_upload(
        client,
        upload_url,
        &art.sha256,
        "provenance backfill",
        &art.sidecar,
        || Some(Form::new().part("provenance", provenance_part(art)?)),
    );
    if ok {
        tracing::debug!(sha256 = %art.sha256, file = %art.filename, "upload: provenance backfilled on hopper");
    }
}

/// POST one result to hopper, retrying transient failures a few times. A 4xx
/// (other than 408/429) can never succeed on resend, so it stops immediately.
fn post_one(
    client: &reqwest::blocking::Client,
    result_url: &str,
    worker: &str,
    sha256: &str,
    envelope: ScanResultEnvelope,
) {
    let payload = ResultPayload {
        sha256: sha256.to_string(),
        worker: worker.to_string(),
        error: None,
        // fs renews don't track per-file analysis time; hopper treats this as
        // cosmetic. 0 keeps the wire shape identical to the worker's.
        duration_ms: 0,
        envelope: Some(envelope),
    };
    let Some((body, encoding)) = encode_result_body(payload, sha256) else {
        return;
    };

    for (attempt, timeout) in ATTEMPT_TIMEOUTS.into_iter().enumerate() {
        if attempt > 0 {
            // 1s, 2s, 4s — brief, since a renew is idempotent and re-runnable.
            std::thread::sleep(Duration::from_secs(1 << (attempt - 1)));
        }
        let mut request = client
            .post(result_url)
            .timeout(timeout)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.clone());
        if let Some(enc) = encoding {
            request = request.header(reqwest::header::CONTENT_ENCODING, enc);
        }
        match request.send() {
            Ok(resp) if resp.status().is_success() => {
                tracing::debug!(sha256 = %sha256, "upload: result renewed on hopper");
                return;
            }
            Ok(resp) => {
                let status = resp.status();
                if status.is_client_error()
                    && status != reqwest::StatusCode::REQUEST_TIMEOUT
                    && status != reqwest::StatusCode::TOO_MANY_REQUESTS
                {
                    let body = resp.text().unwrap_or_default();
                    tracing::warn!(sha256 = %sha256, %status, body = %body, "upload: rejected by hopper; not retrying");
                    return;
                }
                tracing::warn!(sha256 = %sha256, %status, attempt, "upload: non-success response");
            }
            Err(e) => {
                tracing::warn!(sha256 = %sha256, error = %error_chain(&e), attempt, "upload: send failed");
            }
        }
    }
    tracing::warn!(sha256 = %sha256, attempts = ATTEMPT_TIMEOUTS.len(), "upload: giving up after retries");
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// The wire body round-trips through zstd back to the exact JSON serde
    /// produced: this is the same shape hopper's `/api/result` decodes, so an fs
    /// upload is byte-identical to a worker upload of the same payload.
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
        // (skip_serializing_if), matching the worker's wire form.
        let value: serde_json::Value = serde_json::from_slice(&decoded).unwrap();
        assert_eq!(value["worker"], "scan-fs");
        assert_eq!(value["duration_ms"], 0);
        assert!(value.get("error").is_none());
    }

    #[test]
    fn default_worker_name_is_valid_for_hopper() {
        let name = default_worker_name();
        assert!(!name.is_empty());
        assert!(name.len() <= MAX_WORKER_NAME_LEN);
        // Mirrors hopper's `validWorkerName`: printable ASCII, no spaces.
        assert!(name.chars().all(|c| c.is_ascii_graphic()));
    }

    #[test]
    fn upload_sends_configured_bearer_token() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server address");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept upload");
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).expect("read upload");
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            stream
                .write_all(
                    b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .expect("write response");
            String::from_utf8_lossy(&request).into_owned()
        });

        let url = format!("http://{addr}/api/upload");
        let ok = post_upload_with_token(
            &reqwest::blocking::Client::new(),
            &url,
            &"a".repeat(64),
            "test",
            b"{}",
            Some("test-secret"),
            || Some(reqwest::blocking::multipart::Form::new().text("sha256", "a".repeat(64))),
        );
        assert!(ok);
        let request = server.join().expect("server thread");
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-secret\r\n"),
            "request headers: {request}"
        );
    }

    /// A dependency's upload artifact loads its bytes lazily from the fetch cache
    /// (never eagerly), derives its stored filename from the resolved URL, and is
    /// marked backfillable so its captured registry provenance lands even when
    /// hopper already holds the bytes. The test uses an unsupported PURL to prove
    /// artifact construction does not perform a registry lookup.
    ///
    /// Built here from an *unevaluated* dependency: the artifact is independent
    /// of the verdict, so bytes and provenance reach hopper even when scan has
    /// no verdict to post for them.
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
                    bytes: br#"{"provider_only":42}"#.to_vec(),
                }],
            )),
            verdict: None,
            members: crate::engine::MemberEvals::new(),
            raw: "{}".to_string(),
        };
        let art = dep_artifact(&dep, "scan+test", "2026-06-28T00:00:00Z");
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
        assert!(
            matches!(&art.bytes, ArtifactBytes::Cached { locator } if locator == "pkg:bogus/x@1"),
            "bytes load lazily from the cache by locator",
        );
    }
}
