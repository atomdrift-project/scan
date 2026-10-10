//! Best-effort upload of scan results to a hopper instance.
//!
//! Mirrors the pull-based worker's `/api/result` contract — the same
//! `ResultPayload` wire shape, the same zstd-compressed envelope — but driven
//! by a local `scan path` run instead of a poll loop. `scan path --hopper=<url>`
//! uses it to *renew* a sample hopper has already ingested with this build's
//! traits and model: hopper's `/api/result` is a lease-free `UPDATE ... WHERE
//! sha256 = ?`, so posting a result for an already-scanned SHA replaces its
//! stored cleave/litmus envelope (and an unknown SHA is a harmless no-op).
//!
//! Uploads run on a dedicated thread so blocking network I/O never stalls the
//! analysis pool. Network upload failures do not fail the local scan, but they
//! are surfaced as explicit errors so a successful local verdict cannot hide a
//! lost renewal. Failure to hand request-owned bytes to the uploader is
//! different: that is reported as a failed request rather than returning a
//! result known not to be persisted.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SendError, SyncSender};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use reqwest::StatusCode;
use reqwest::blocking::multipart::{Form, Part};
use serde::Serialize;

use crate::engine::ScanResultEnvelope;

/// A hopper bearer token and where it was found. The origin is a path or an
/// environment variable name — never the secret — so it is safe to log.
#[derive(Debug)]
struct Credential {
    token: String,
    origin: String,
}

/// The process's hopper credential, and the file it was (or would have been)
/// read from — resolved once, environment included.
///
/// `$HOPPER_TOKEN` wins, for callers that inject the token some other way;
/// otherwise it is the first non-empty line of `~/.tok/hopper`, unless
/// `$HOPPER_TOKEN_FILE` names another file — the same convention as
/// `~/.tok/openrouter` and `~/.tok/scan`. A locally supervised worker inherits
/// the service account's `HOME`, so it finds the file with no plumbing. The
/// variable names the file rather than the secret, so the token stays off argv
/// and out of the environment.
///
/// Resolved once per process: hopper reads its own copy once at startup too,
/// so rotation is a restart on both ends.
fn credentials() -> &'static (Option<Credential>, Option<PathBuf>) {
    static CREDENTIALS: OnceLock<(Option<Credential>, Option<PathBuf>)> = OnceLock::new();
    CREDENTIALS.get_or_init(|| {
        let env = std::env::var("HOPPER_TOKEN").ok();
        let path = std::env::var_os("HOPPER_TOKEN_FILE")
            .map(PathBuf::from)
            .filter(|path| !path.as_os_str().is_empty())
            .or_else(|| crate::interpret::tok_path("hopper"));
        (resolve_credential(env.as_deref(), path.as_deref()), path)
    })
}

/// Bearer token for hopper's API, or `None` when hopper is unauthenticated.
/// Never the origin, never logged.
#[must_use]
pub fn hopper_token() -> Option<&'static str> {
    credentials().0.as_ref().map(|c| c.token.as_str())
}

/// The process's blocking HTTP client for hopper, with [`REQUEST_TIMEOUT`] as
/// its default per-request ceiling.
///
/// Built on first use, on whichever plain thread first needs it — the uploader
/// thread or an analysis thread, never the async runtime `serve` builds its
/// uploader from, where reqwest refuses to build a blocking client and panics
/// dropping the half-built one. A static is never dropped there either. `None`,
/// said once, when it cannot be built: callers then stop rather than fall back
/// to a client without timeouts.
pub(crate) fn hopper_http() -> Option<&'static reqwest::blocking::Client> {
    static HTTP: OnceLock<Option<reqwest::blocking::Client>> = OnceLock::new();
    HTTP.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| {
                tracing::error!(error = %error_chain(&e), "cannot build the hopper HTTP client; hopper calls disabled");
            })
            .ok()
    })
    .as_ref()
}

/// Split `--hopper` into the endpoints to try, in preference order.
///
/// One address is the ordinary case. Several — comma-separated, as `SCAN_URL`
/// and `--allowed-dirs` already are — name the same hopper reached two ways:
/// put the replica first and the primary behind it, and a replica outage costs
/// a retry rather than a lost verdict. Reads and writes take the same list on
/// purpose. Routing them separately is a topology this worker would have to
/// know, and hopper's write relay exists precisely so it does not: a replica
/// answers lookups locally and forwards the renewals.
#[must_use]
pub fn endpoints(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|url| url.trim().trim_end_matches('/').trim().to_string())
        .filter(|url| !url.is_empty())
        .collect()
}

/// Warn, once per address, when the hopper token would cross the network in
/// cleartext: a plain `http://` hopper that is not on loopback. The token is
/// fleet-wide write access to the corpus, so it should only travel over TLS or
/// a network that encrypts on its own (a WireGuard/Tailscale address, which
/// this cannot tell apart and so still names).
pub fn warn_if_cleartext(raw: &str) {
    if hopper_token().is_none() {
        return;
    }
    for endpoint in endpoints(raw) {
        let Ok(url) = reqwest::Url::parse(&endpoint) else {
            continue;
        };
        let host = url
            .host_str()
            .unwrap_or_default()
            .trim_start_matches('[')
            .trim_end_matches(']');
        let loopback = host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback());
        if url.scheme() == "http" && !loopback {
            tracing::warn!(
                hopper = %endpoint,
                "the hopper token is sent to this address in cleartext; use https unless the network itself is encrypted"
            );
        }
    }
}

/// The one address a worker may poll: the primary, which [`endpoints`] puts
/// last.
///
/// Worker routes are the exception to the rule above. A replica answers
/// lookups and relays renewals, but it refuses `/api/next` and the plain
/// `/api/result` with a 403 even when its relay is enabled — the fleet's queue
/// is the primary's to hand out, and passing the worker firehose through a
/// replica helps no one. So there is nothing to fail over to here: the second
/// address is not another way to reach the same answer, it is the only one.
///
/// Returns `None` for an empty or blank `--hopper`, which is how the deploy
/// says "do not file results anywhere".
#[must_use]
pub fn worker_endpoint(raw: &str) -> Option<String> {
    endpoints(raw).pop()
}

/// One hopper route, at every address it can be reached.
///
/// Ordered as `--hopper` named them. A retry walks down the list rather than
/// hammering one address, so the second attempt after a replica stops answering
/// lands on the primary instead of on the same silence.
#[derive(Debug, Clone)]
struct Route(Vec<String>);

impl Route {
    fn new(bases: &[String], suffix: &str) -> Self {
        Self(bases.iter().map(|base| format!("{base}{suffix}")).collect())
    }

    /// The address to use on this attempt, clamped to the last: a budget longer
    /// than the list keeps retrying the final address rather than wrapping back
    /// to one already known to be failing.
    fn at(&self, attempt: usize) -> &str {
        let last = self.0.len().saturating_sub(1);
        self.0.get(attempt.min(last)).map_or("", String::as_str)
    }

    /// Every address, for a caller that tries each exactly once rather than
    /// retrying on a schedule.
    fn each(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }
}

/// The precedence behind [`credentials`], split out so it is testable without
/// touching process-wide environment or the `OnceLock`.
fn resolve_credential(env: Option<&str>, path: Option<&std::path::Path>) -> Option<Credential> {
    if let Some(value) = env.map(str::trim).filter(|value| !value.is_empty()) {
        return Some(Credential {
            token: value.to_string(),
            origin: "$HOPPER_TOKEN".to_string(),
        });
    }
    let path = path?;
    Some(Credential {
        token: crate::interpret::read_token_file(path)?,
        origin: path.display().to_string(),
    })
}

/// Point this process at `hopper_url`: report where the bearer token came from,
/// and arm the fleet-shared dependency precheck against that same host.
///
/// Both halves belong to one decision — "this is the hopper we talk to" — and
/// every entry point that reaches hopper calls this, so
/// `crate::corpus_precheck`'s "the hopper you submit to is the hopper you ask"
/// holds by construction. It used to hold only by each caller remembering, and
/// the pull worker did not: it logged the credential and never armed the
/// precheck, so it re-analyzed and re-mirrored dependencies the corpus already
/// held at this analyzer version — which hopper records as "result renewed with
/// no analyzer change; the re-analysis learned nothing".
pub fn use_hopper(hopper_url: &str) {
    log_hopper_credential();
    crate::corpus_precheck::configure(hopper_url);
}

/// Log where the hopper credential came from, or warn that there is none.
///
/// Hopper requires `Authorization: Bearer <token>` on every route and does not
/// exempt loopback, so an unauthenticated worker or `--hopper` upload is
/// rejected with 401 on every request. Say so once at startup rather than
/// leaving an operator to infer it from a retry loop.
fn log_hopper_credential() {
    match credentials() {
        (Some(credential), _) => {
            tracing::info!(source = %credential.origin, "hopper API token loaded");
        }
        (None, expected) => tracing::warn!(
            expected = %expected.as_deref().unwrap_or(std::path::Path::new("")).display(),
            "no hopper API token found; unless hopper runs unauthenticated every \
             request will be rejected with 401 — install the token at \
             ~/.tok/hopper (mode 0600) or set $HOPPER_TOKEN",
        ),
    }
}

/// Attach a bearer token to a blocking request, if there is one.
fn bearer(
    request: reqwest::blocking::RequestBuilder,
    token: Option<&str>,
) -> reqwest::blocking::RequestBuilder {
    match token {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

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

/// Per-attempt request timeouts for `/api/upload`, separate from
/// [`REQUEST_TIMEOUT`] because that ceiling was sized for the small JSON
/// verdicts a renewal sends, not for streaming a multi-GiB
/// artifact — with `HOPPER_MAX_UPLOAD_ARTIFACT_BYTES` raised to 8 GiB
/// (2026-08-27), a legitimate transfer on a merely mediocre link would blow
/// through 120s and get cut off as a timeout rather than a real failure.
/// The final ceiling matches hopper's own `uploadBodyTimeout` (30 min, same
/// change) — no point the client giving up before the server would have.
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
/// succeed (confirmed against hopper's `samples` table on 2026-08-27: every
/// artifact over the then-100-MiB cap that scan attempted to upload landed
/// with an empty `path` — the bytes never arrived — while everything under
/// it landed clean). Raised to 8 GiB alongside hopper's own cap the same
/// day; keep the two in sync — this can only shrink the set of artifacts
/// scan bothers attempting, never rescue one hopper would reject.
const HOPPER_MAX_UPLOAD_ARTIFACT_BYTES: u64 = 8 << 30;

/// Hopper's ingestion lane header. A result renewed by `serve --hopper` is
/// one-shot: the caller that asked for the scan is already holding the verdict
/// in its own cache, so it will never ask again, and a renewal that does not
/// land means the artifact never enters the corpus at all. A worker result is
/// retryable for free — the job returns to the queue and is dispatched again.
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
/// backlog — it can persist for many minutes. The old budget was four attempts
/// over ~16s, which is not a retry so much as a coin flip: measured against a
/// saturated hopper it lost every renewal it was given.
const RENEW_BUDGET: Duration = Duration::from_secs(15 * 60);

/// Ceiling on one backoff sleep, so a long budget still probes often enough to
/// catch a short window of free capacity rather than sleeping through it.
const RETRY_MAX_BACKOFF: Duration = Duration::from_secs(60);

/// Floor on one backoff sleep. Full jitter can draw near zero, and a retry
/// that fires instantly just spends a slot-acquire on a pool it was told is
/// full.
const RETRY_MIN_BACKOFF: Duration = Duration::from_millis(250);

/// Request timeout per POST. Matches the worker so a wedged hopper can't pin an
/// uploader thread indefinitely.
///
/// One ceiling for every attempt, deliberately: `/api/result` is not a request
/// a client can abandon cheaply. Hopper writes the parent and every archive
/// member in one transaction, and runs it on a context *detached* from the
/// request (`context.WithoutCancel`, `resultStoreTimeout` = 10 min in its
/// `api.go`) precisely so a client that gives up cannot discard completed
/// analysis. Timing out early therefore cancels nothing — it leaves the first
/// store running, holding one of hopper's few reserved renewal slots, and sends
/// a second copy of the same result to contend with it on the same rows. The
/// escalating 15s/30s/60s table this replaces was built on the opposite
/// assumption ("slow spells are brief, so fail fast and try again") and made
/// every store slower than 15 seconds look like a network fault.
///
/// Slower than 15 seconds is ordinary: hopper buckets its store phase out to
/// 60s (`hopper.result_phase.seconds`), and a loaded broker is slower still —
/// smaug's log for 2026-09-07/08 carries 1,509 ingestion sheds and 712 slow
/// slot-waits in 23 hours. The `send failed ... operation timed out` this
/// fixes was one such store: hopper accepted and committed it, and a later
/// attempt landed a second copy of the same verdict.
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
    /// Provenance only: hopper already holds these bytes, so none move (a
    /// registry fallback backfilling onto real content).
    None,
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
#[derive(Debug)]
enum Job {
    /// Renew a verdict on hopper (the original `--upload` behavior). Boxed: the
    /// envelope dwarfs the other variant, so an unboxed enum would bloat every
    /// queued job to its size.
    Result {
        sha256: String,
        /// The package this artifact was analyzed as, when it was requested by
        /// one. Carried purely so the upload's log lines name the package the
        /// operator asked about rather than a digest they would have to
        /// resolve back to it by hand.
        purl: Option<String>,
        envelope: Box<ScanResultEnvelope>,
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
    Dependencies {
        deps: Vec<crate::engine::DepResult>,
        /// Model version and analysis time stamped on each dependency's verdict,
        /// carried from the parent result so the `ml` section is self-describing.
        version: String,
        analyzed_at: String,
    },
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
    tx: Option<SyncSender<Job>>,
    worker: Option<JoinHandle<()>>,
    /// Jobs accepted but not yet handled. A `SyncSender` cannot be asked its
    /// depth, and this is the difference between "quiet because nothing needs
    /// filing" and "quiet because the filing is stuck".
    pending: Arc<AtomicUsize>,
    tally: Arc<Tally>,
}

/// A point-in-time view of the uploader, for `/_/stats`.
#[derive(Debug, Clone, Copy)]
pub struct UploadStats {
    /// Jobs accepted but not yet handled.
    pub pending: usize,
    /// Queue capacity; at this depth `submit` blocks the analysis thread.
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
        use_hopper(hopper_url);
        let (tx, jobs) = std::sync::mpsc::sync_channel::<Job>(UPLOAD_QUEUE_DEPTH);
        let pending = Arc::new(AtomicUsize::new(0));
        let tally = Arc::new(Tally::default());
        let url = hopper_url.to_owned();
        let (thread_pending, thread_tally) = (Arc::clone(&pending), Arc::clone(&tally));
        let spawned = std::thread::Builder::new()
            .name("scan-upload".into())
            .spawn(move || {
                // On this thread, never the caller's: `serve` builds its
                // uploader inside its async runtime. Returning drops `jobs`,
                // so every submission says the uploader stopped.
                let Some(http) = hopper_http() else {
                    return;
                };
                let hopper = Hopper::new(http.clone(), &url, worker, thread_tally);
                hopper.serve(jobs, &thread_pending);
            });
        match spawned {
            Ok(handle) => {
                tracing::info!(hopper = %hopper_url, "upload: renewing results on hopper");
                Self {
                    tx: Some(tx),
                    worker: Some(handle),
                    pending,
                    tally,
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "upload: failed to spawn uploader thread; uploads disabled");
                Self {
                    tx: None,
                    worker: None,
                    pending,
                    tally,
                }
            }
        }
    }

    /// A point-in-time view of the upload queue.
    ///
    /// `pending` at capacity means analyses are blocking on `submit`; `failed`
    /// climbing means verdicts are being computed and then lost, which no other
    /// signal reports.
    #[must_use]
    pub fn stats(&self) -> UploadStats {
        UploadStats {
            pending: self.pending.load(Ordering::Relaxed),
            capacity: UPLOAD_QUEUE_DEPTH,
            failed: self.tally.failed.load(Ordering::Relaxed),
            uploaded: self.tally.uploaded.load(Ordering::Relaxed),
        }
    }

    /// Hand a job to the uploader thread, blocking briefly when the queue is
    /// full (backpressure). The pending count covers exactly the jobs the
    /// thread will see; a job that cannot be queued comes back to the caller.
    fn enqueue(&self, job: Job) -> Result<(), Job> {
        let Some(tx) = &self.tx else {
            return Err(job);
        };
        self.pending.fetch_add(1, Ordering::Relaxed);
        tx.send(job).map_err(|SendError(job)| {
            self.pending.fetch_sub(1, Ordering::Relaxed);
            job
        })
    }

    /// Queue a result for upload. A stopped uploader drops the result, and
    /// says so.
    pub fn submit(&self, sha256: String, purl: Option<String>, envelope: ScanResultEnvelope) {
        let job = Job::Result {
            sha256,
            purl,
            envelope: Box::new(envelope),
        };
        if let Err(Job::Result { sha256, .. }) = self.enqueue(job) {
            tracing::error!(
                sha256 = %sha256,
                "upload: uploader stopped before result could be sent to hopper"
            );
        }
    }

    /// Queue artifacts (the scanned file and any fetched dependency archives) for
    /// content reconciliation: hopper is asked which it lacks, and only those are
    /// uploaded with their provenance. Submit before the matching [`Self::submit`] so a
    /// new top-level file's row exists before its verdict lands.
    pub fn submit_artifacts(&self, artifacts: Vec<UploadArtifact>) {
        if !self.enqueue_artifacts(artifacts, Vec::new()) {
            tracing::error!("upload: uploader stopped before artifacts could be queued");
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
            Err("uploader stopped before durable artifacts could be queued".to_string())
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
            Err("uploader stopped before durable artifact could be queued".to_string())
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
        version: String,
        analyzed_at: String,
    ) {
        if deps.is_empty() {
            return;
        }
        let job = Job::Dependencies {
            deps,
            version,
            analyzed_at,
        };
        if let Err(Job::Dependencies { deps, .. }) = self.enqueue(job) {
            tracing::error!(
                dependencies = deps.len(),
                "upload: uploader stopped before dependencies could be queued"
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

/// Whether hopper refused a request for good: any 4xx but 408 and 429, which
/// ask to be retried. Resending a permanent refusal can never succeed.
pub(crate) fn is_permanent(status: StatusCode) -> bool {
    status.is_client_error()
        && status != StatusCode::REQUEST_TIMEOUT
        && status != StatusCode::TOO_MANY_REQUESTS
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
            token: hopper_token(),
            worker,
            tally,
        }
    }

    /// The uploader thread's loop: handle every queued job until the sender
    /// side is dropped.
    fn serve(&self, jobs: Receiver<Job>, pending: &AtomicUsize) {
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
                Job::Result {
                    sha256,
                    purl,
                    envelope,
                } => self.renew(&sha256, purl.as_deref(), *envelope),
                Job::Artifacts {
                    artifacts,
                    cleanup_dirs,
                } => {
                    self.reconcile(cache.as_ref(), &mut seen, artifacts);
                    cleanup_upload_dirs(cleanup_dirs);
                }
                Job::Dependencies {
                    deps,
                    version,
                    analyzed_at,
                } => {
                    self.sync_dependencies(&version, &analyzed_at, cache.as_ref(), &mut seen, deps)
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
    fn sync_dependencies(
        &self,
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
        let collector = format!("scan+{}", self.worker);
        // Bytes + provenance first, so each dependency's row exists before its
        // verdict UPDATE (hopper's `/api/result` no-ops on a missing row). The
        // local seen set starts empty — the run-level dedup above already
        // removed repeats.
        let artifacts: Vec<UploadArtifact> = fresh
            .iter()
            .filter_map(|d| dep_artifact(d, &collector, analyzed_at))
            .collect();
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
            let envelope = match crate::engine::dep_envelope(&dep, version, analyzed_at) {
                Ok(Some(envelope)) => envelope,
                Ok(None) => {
                    tracing::info!(
                        sha256 = %dep.sha256,
                        locator = %dep.locator,
                        "upload: dependency not evaluated; stored for analysis without a verdict"
                    );
                    continue;
                }
                Err(error) => {
                    tracing::error!(
                        sha256 = %dep.sha256,
                        locator = %dep.locator,
                        error = format!("{error:#}"),
                        "upload: dependency report does not parse; verdict not posted"
                    );
                    continue;
                }
            };
            // A dependency's locator is a PURL or a URL; only the former belongs
            // under a `purl` field, so a URL-sourced dependency logs by digest.
            let purl = dep
                .locator
                .starts_with("pkg:")
                .then_some(dep.locator.as_str());
            self.renew(&dep.sha256, purl, envelope);
        }
    }

    /// POST the batch existence probe (`/api/known`) and return the digests
    /// whose bytes hopper already holds. On any failure returns an empty set —
    /// the caller then treats every artifact as missing, which is the safe
    /// direction: a failed probe re-sends bytes hopper had, where the other way
    /// round would withhold an artifact it does not.
    ///
    /// One question, deliberately. It also used to ask which verdicts were
    /// already current, which is a different question about a different column,
    /// and the answer aged before it could be acted on — hopper decides that at
    /// the store now (`unchangedStore`), where it is ordered against the other
    /// producers.
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
                            body = %body,
                            provenance = %crate::worker::body_excerpt(&String::from_utf8_lossy(&art.sidecar)),
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
    fn renew(&self, sha256: &str, purl: Option<&str>, envelope: ScanResultEnvelope) {
        let payload = ResultPayload {
            sha256: sha256.to_string(),
            worker: self.worker.clone(),
            error: None,
            // fs renews don't track per-file analysis time; hopper treats this as
            // cosmetic. 0 keeps the wire shape identical to the worker's.
            duration_ms: 0,
            envelope: Some(envelope),
        };
        let Some((body, encoding)) = encode_result_body(payload, sha256) else {
            self.tally.failed.fetch_add(1, Ordering::Relaxed);
            return;
        };
        // Shared, not copied, across attempts: a compressed report can be large.
        let body = bytes::Bytes::from(body);

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
                        tracing::error!(sha256 = %sha256, purl, %status, body = %body, "upload: hopper rejected result; not retrying");
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
/// that has no [`Uploader`] (the pull-based worker). Dedups within this call,
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
    hopper.sync_dependencies(version, analyzed_at, cache, &mut HashSet::new(), deps);
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

/// Look up whether hopper already holds real content for a purl coordinate —
/// synchronously, for `scan purl`/`scan url`'s registry-metadata fallback
/// (`crate::pkg::run`), which has no async runtime to bridge into
/// [`crate::server::corpus::Corpus`]'s async reader the server side of this
/// same fix uses. Returns the sha256 of the sample hopper already has
/// analyzed under this purl, if any; `None` on any failure or absence — the
/// caller's fallback is to post nothing, never to guess.
#[must_use]
pub(crate) fn known_sha_for_purl(
    client: &reqwest::blocking::Client,
    hopper_url: &str,
    purl: &str,
) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct LookupResponse {
        #[serde(default)]
        sha256: Option<String>,
    }
    for base in endpoints(hopper_url) {
        let request = client
            .get(format!("{base}/v1/lookup"))
            .query(&[("purl", purl)]);
        match bearer(request, hopper_token()).send() {
            Ok(resp) if resp.status().is_success() => {
                return resp
                    .json::<LookupResponse>()
                    .map_err(|e| tracing::warn!(endpoint = %base, error = %error_chain(&e), "upload: purl lookup answer unreadable"))
                    .ok()
                    .and_then(|r| r.sha256);
            }
            Ok(resp) => {
                tracing::warn!(endpoint = %base, status = %resp.status(), "upload: purl lookup non-success");
            }
            Err(e) => {
                tracing::warn!(endpoint = %base, error = %error_chain(&e), "upload: purl lookup failed");
            }
        }
    }
    None
}

/// Sleep before retry `attempt` (1-based) of any hopper write.
///
/// Exponential with full jitter: the sleep is drawn from `[0, ceiling)` where
/// the ceiling doubles per attempt up to [`RETRY_MAX_BACKOFF`]. The jitter
/// matters more than the growth here — every scan server writing to the same
/// saturated hopper would otherwise retry in lockstep and re-saturate it the
/// instant a slot frees.
///
/// `retry_after` is hopper's own hint and acts as a floor: returning before it
/// only spends a slot-acquire on a pool that just said it was full.
///
/// Pure, with the random draw passed in, so the policy is testable without
/// sleeping or seeding.
fn backoff(attempt: usize, retry_after: Option<Duration>, fuzz: f64) -> Duration {
    let ceiling = RETRY_MAX_BACKOFF.min(
        RETRY_MIN_BACKOFF
            .saturating_mul(1u32 << attempt.min(16))
            .max(RETRY_MIN_BACKOFF),
    );
    ceiling
        .mul_f64(fuzz.clamp(0.0, 1.0))
        .max(retry_after.unwrap_or(RETRY_MIN_BACKOFF))
        .max(RETRY_MIN_BACKOFF)
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

/// A uniform-ish draw in `[0, 1)` for backoff jitter.
///
/// Deliberately not a `rand` dependency: spreading retries needs decorrelation,
/// not statistical quality. Mixes the clock through splitmix64 so two uploader
/// threads starting in the same millisecond still diverge.
fn fuzz() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    // Seconds and sub-second nanos combined without going through u128, so
    // there is no truncating cast to explain away.
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        d.as_secs()
            .wrapping_shl(20)
            .wrapping_add(u64::from(d.subsec_nanos()))
    });
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // 53 bits is the mantissa width, so this maps onto [0, 1) without bias.
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Parse `Retry-After` in its delta-seconds form. The HTTP-date form and any
/// unparseable value yield `None`, leaving the caller on its own backoff.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let secs: u64 = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    (secs > 0).then(|| Duration::from_secs(secs).min(RETRY_MAX_BACKOFF))
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

    /// `--hopper` may name the same corpus twice: the replica first, the
    /// primary behind it. Reads and writes take the same list, because routing
    /// them apart is a topology this worker would have to know and hopper's
    /// write relay exists so that it does not.
    #[test]
    fn hopper_endpoints_are_a_preference_order() {
        assert_eq!(endpoints("https://ro/"), vec!["https://ro"]);
        assert_eq!(
            endpoints(" https://ro/ , http://rw:8081/ "),
            vec!["https://ro", "http://rw:8081"],
        );
        assert!(endpoints("").is_empty());
        assert!(endpoints(" , , ").is_empty());
    }

    /// The worker loop is the exception: hopper refuses `/api/next` on a
    /// replica with a 403 whether or not its relay is on, so a worker takes the
    /// primary — the last address — and nothing else. Handing it the raw string
    /// instead put the commas inside a hostname, and every poll for the life of
    /// the process failed with `invalid dns name`.
    #[test]
    fn a_worker_polls_the_primary_only() {
        assert_eq!(
            worker_endpoint("https://ro,https://rw").as_deref(),
            Some("https://rw"),
        );
        assert_eq!(
            worker_endpoint(" https://ro/ , https://rw/ ").as_deref(),
            Some("https://rw"),
        );
        // The ordinary single-address case is unchanged.
        assert_eq!(
            worker_endpoint("https://rw/").as_deref(),
            Some("https://rw")
        );
        // Nowhere to file results is a valid deploy, not an address.
        assert_eq!(worker_endpoint(""), None);
        assert_eq!(worker_endpoint(" , , "), None);
        // Whatever a worker polls, it is one address — never a list.
        for raw in ["https://ro,https://rw", "https://rw", " a , b , c "] {
            let picked = worker_endpoint(raw).expect("an address");
            assert!(
                !picked.contains(','),
                "a worker was handed a list: {picked}"
            );
        }
    }

    /// A retry walks down the list rather than hammering one address, so the
    /// attempt after a replica stops answering lands on the primary instead of
    /// on the same silence. Past the end it holds on the last: a budget longer
    /// than the list must not wrap back to an address already known to fail.
    #[test]
    fn a_retry_moves_to_the_next_address() {
        let bases = endpoints("https://ro,http://rw");
        let route = Route::new(&bases, "/api/result");
        assert_eq!(route.at(0), "https://ro/api/result");
        assert_eq!(route.at(1), "http://rw/api/result");
        assert_eq!(route.at(9), "http://rw/api/result");
    }

    /// One address is the ordinary case, and every attempt uses it.
    #[test]
    fn a_single_address_is_used_for_every_attempt() {
        let route = Route::new(&endpoints("https://only"), "/api/known");
        assert_eq!(route.at(0), "https://only/api/known");
        assert_eq!(route.at(5), "https://only/api/known");
    }

    /// The budget is the whole point: a renewal is one-shot, so it must outlive
    /// a saturated hopper rather than the ~16s the old four-attempt loop gave
    /// it.
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

    /// Full jitter: the draw scales the ceiling, so a fleet retrying against one
    /// saturated hopper spreads out instead of re-saturating it in lockstep.
    #[test]
    fn backoff_applies_full_jitter() {
        let low = backoff(10, None, 0.0);
        let high = backoff(10, None, 1.0);
        assert!(high > low, "jitter had no effect: {low:?} vs {high:?}");
        assert!(
            low >= RETRY_MIN_BACKOFF,
            "a near-zero draw must still back off: {low:?}"
        );
        assert!(high <= RETRY_MAX_BACKOFF, "exceeded the ceiling: {high:?}");
    }

    /// The ceiling grows with the attempt and then stops.
    #[test]
    fn backoff_grows_exponentially_then_caps() {
        let at = |n| backoff(n, None, 1.0);
        assert!(at(1) < at(3), "not growing: {:?} then {:?}", at(1), at(3));
        assert!(at(3) < at(6), "not growing: {:?} then {:?}", at(3), at(6));
        assert_eq!(at(20), RETRY_MAX_BACKOFF, "ceiling not enforced");
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

    /// 408 and 429 ask to be retried; every other 4xx is final.
    #[test]
    fn only_retryable_refusals_are_retried() {
        assert!(is_permanent(StatusCode::BAD_REQUEST));
        assert!(is_permanent(StatusCode::UNAUTHORIZED));
        assert!(is_permanent(StatusCode::PAYLOAD_TOO_LARGE));
        assert!(!is_permanent(StatusCode::REQUEST_TIMEOUT));
        assert!(!is_permanent(StatusCode::TOO_MANY_REQUESTS));
        assert!(!is_permanent(StatusCode::SERVICE_UNAVAILABLE));
    }

    #[test]
    fn parse_retry_after_forms() {
        use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};
        let with = |v: &str| {
            let mut h = HeaderMap::new();
            h.insert(RETRY_AFTER, HeaderValue::from_str(v).unwrap());
            parse_retry_after(&h)
        };
        assert_eq!(with("2"), Some(Duration::from_secs(2)));
        assert_eq!(with(" 5 "), Some(Duration::from_secs(5)));
        assert_eq!(with("0"), None);
        assert_eq!(with("-1"), None);
        // The HTTP-date form is legal but unparsed here; fall back to our own.
        assert_eq!(with("Wed, 21 Oct 2026 07:28:00 GMT"), None);
        // A hostile value must not park an uploader thread for hours.
        assert_eq!(with("86400"), Some(RETRY_MAX_BACKOFF));
        assert_eq!(parse_retry_after(&HeaderMap::new()), None);
    }

    #[test]
    fn fuzz_is_in_range_and_varies() {
        let draws: Vec<f64> = (0..64).map(|_| fuzz()).collect();
        assert!(draws.iter().all(|d| (0.0..1.0).contains(d)), "out of range");
        let distinct = draws
            .iter()
            .map(|d| d.to_bits())
            .collect::<std::collections::HashSet<_>>();
        assert!(distinct.len() > 1, "fuzz returned a constant");
    }

    /// Claiming hopper's reserved lane is what keeps a one-shot renewal from
    /// competing with the retryable worker firehose.
    #[test]
    fn lane_header_matches_hoppers_contract() {
        assert_eq!(HOPPER_LANE_HEADER, "X-Hopper-Lane");
        assert_eq!(HOPPER_LANE_RENEW, "renew");
    }

    /// `$HOPPER_TOKEN` wins, for callers that inject the token some other
    /// way; otherwise it comes from `~/.tok/hopper`. A blank env value is not
    /// a credential and must fall through to the file rather than suppress
    /// it.
    #[test]
    fn hopper_token_precedence() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("hopper");
        std::fs::write(&file, "from-file\n").expect("write token");
        let token = |env, path| resolve_credential(env, path).map(|c| c.token);

        assert_eq!(
            token(Some("from-env"), Some(&file)).as_deref(),
            Some("from-env")
        );
        assert_eq!(token(Some("  "), Some(&file)).as_deref(), Some("from-file"));
        assert_eq!(token(None, Some(&file)).as_deref(), Some("from-file"));
        assert_eq!(token(Some(" padded "), None).as_deref(), Some("padded"));
        assert_eq!(token(None, None), None);
        assert_eq!(token(None, Some(&dir.path().join("absent"))), None);
    }

    /// The logged origin names the source, never the secret.
    #[test]
    fn hopper_token_origin_names_its_source() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("hopper");
        std::fs::write(&file, "from-file\n").expect("write token");

        let from_env = resolve_credential(Some("from-env"), Some(&file)).expect("credential");
        assert_eq!(from_env.origin, "$HOPPER_TOKEN");

        let from_file = resolve_credential(None, Some(&file)).expect("credential");
        assert_eq!(from_file.origin, file.display().to_string());
        assert!(!from_file.origin.contains("from-file"));
    }

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
        let (tx, jobs) = std::sync::mpsc::sync_channel::<Job>(UPLOAD_QUEUE_DEPTH);
        drop(jobs);
        let uploader = Uploader {
            tx: Some(tx),
            worker: None,
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
            raw: "{}".to_string(),
        };
        uploader.submit_dependencies(vec![dep], "v".into(), "t".into());
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
            raw: "{}".to_string(),
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
