//! HTTP API server for litmus malware classification.
//!
//! Accepts artifacts — uploaded bytes, a package URL, an exact URL, or a local
//! path — runs cleave static analysis and the model, and answers with the
//! classification: the full scan envelope on the legacy routes, a decision on
//! `/v1`.
//!
//! Routes:
//!   GET  /_/health      — liveness (public; detail only when trusted)
//!   GET  /_/info        — build and capacity, for sizing a client
//!   GET  /_/stats       — live routing signals
//!   GET  /_/memory, /_/requests, /_/threads — diagnostics
//!   POST /_/reload      — reload the model bundle from disk (admin)
//!   POST /_/update      — pull models and traits, then reload (admin)
//!   GET  /lookup        — stored verdict by ?sha256= or ?purl= (no slot)
//!   GET  /status        — whether an analysis of an artifact is running
//!   POST /analyze       — upload a file, receive the scan envelope
//!   POST /analyze-purl  — fetch a PURL (registry provenance included) and analyze
//!   POST /analyze-path  — analyze a local path (loopback)
//!   GET  /v1/lookup     — decisions for one or many artifacts (no slot)
//!   POST /v1/analyze    — a decision, streamed if the analysis is slow
//!
//! [`ServerConfig`] is plain data: [`Startup::resolve`] builds it from a
//! command line, and [`build_app`] validates it before anything starts.

mod access;
mod acl;
mod analyze;
mod corpus;
mod decision;
mod diag;
mod error;
mod flight;
mod handlers;
mod idle;
mod latency;
mod v1;

pub use acl::{Cidr, TokenDigest, parse_cidr_list};

use crate::analysis::{ACTIVE_REQUESTS, ModelResources, RequestPhase};

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use std::future::Future;
use std::net::SocketAddr;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::{Duration, Instant};
use tokio::signal;
use tokio::sync::{Notify, Semaphore, watch};

use error::ApiError;

use crate::explain::ShapImportance;
use crate::model::{Model, Thresholds};

/// How the HTTP API server runs, settled before it starts.
///
/// Plain data: build one with struct update syntax over [`Default`], or
/// resolve one from a command line with [`Startup::resolve`].
/// [`build_app`] validates it.
///
/// ```
/// use scan::server::ServerConfig;
///
/// let config = ServerConfig {
///     model_dir: "/path/to/models".into(),
///     workers: 2,
///     ..ServerConfig::default()
/// };
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to listen on.
    pub bind: SocketAddr,
    /// Largest request body accepted, in bytes.
    pub max_body_size: usize,
    /// RSS past which requests are refused. `None` disables in-process
    /// throttling, for when a supervisor such as systemd `MemoryMax=` enforces
    /// a hard cap already.
    pub max_rss_bytes: Option<NonZeroU64>,
    /// Model bundle directory.
    pub model_dir: PathBuf,
    /// Manual probability cutoffs. `None` takes the bundle's level grid.
    pub thresholds: Option<Thresholds>,
    /// The operating point (0..=10000) the verdict thresholds come from.
    /// `None` with manual thresholds.
    pub level: Option<u16>,
    /// Per-rule time budget before cleave logs a slow rule.
    pub slow_rule_ms: u64,
    /// Directories `/analyze-path` may read. Empty refuses every request.
    pub allowed_dirs: Vec<PathBuf>,
    /// Where cleave extracts archive members for `/analyze-path` callers.
    pub extract_dir: Option<PathBuf>,
    /// Concurrent analyses; at least one. Past it requests are refused, not
    /// queued.
    pub workers: usize,
    /// Non-zero runs the companion pull worker beside the server (needs
    /// `hopper`). Kept as a count because `--idle-worker-slots` has always
    /// been one; the worker sizes itself.
    pub idle_worker_slots: usize,
    /// Peer networks besides loopback allowed to connect. `/analyze-path` is
    /// loopback-only regardless.
    pub allow_cidrs: Vec<Cidr>,
    /// Digest of the bearer token required on every route but `/_/health`;
    /// `None` leaves the API open. Loopback is not exempt: behind a Cloudflare
    /// tunnel every remote request arrives over loopback.
    pub auth_digest: Option<TokenDigest>,
    /// Per-request analysis timeout in seconds. Zero disables it.
    pub analysis_timeout_secs: u64,
    /// The LLM second opinion, when one is configured.
    pub interpret: Option<crate::interpret::InterpretConfig>,
    /// Which references a sample declares are followed. Off by default: an
    /// upload server driving outbound fetches is an SSRF-shaped exposure, so
    /// turning it on is an explicit operator decision.
    pub fetch: crate::fetch::FetchPolicy,
    /// Hopper API root(s). Enables result renewal, corpus deferral, and the
    /// companion idle worker.
    pub hopper: Option<String>,
    /// Passwords to try against encrypted archives.
    pub zip_passwords: crate::ArchivePasswords,
    /// Class-aware admission (`SCAN_SLOT_LANES=1`): payloads at or below this
    /// many bytes take the small lane. `None` keeps flat admission.
    pub slot_lanes: Option<u64>,
}

impl Default for ServerConfig {
    /// The `serve` command line's defaults, except `model_dir`, which has
    /// none: a caller always names one.
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 49999)),
            max_body_size: 100 * 1024 * 1024,
            max_rss_bytes: None,
            model_dir: PathBuf::new(),
            thresholds: None,
            level: None,
            slow_rule_ms: crate::cli::DEFAULT_SLOW_RULE_MS,
            allowed_dirs: Vec::new(),
            extract_dir: None,
            workers: 1,
            idle_worker_slots: 0,
            allow_cidrs: Vec::new(),
            auth_digest: None,
            analysis_timeout_secs: DEFAULT_ANALYSIS_TIMEOUT_SECS,
            interpret: None,
            fetch: crate::fetch::FetchPolicy::default(),
            hopper: None,
            zip_passwords: crate::ArchivePasswords::default(),
            slot_lanes: None,
        }
    }
}

impl ServerConfig {
    /// Check what plain data cannot enforce by itself.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid manual thresholds or zero workers.
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(t) = &self.thresholds {
            t.validate()
                .map_err(|error| anyhow::anyhow!("invalid thresholds: {error}"))?;
        }
        if self.workers == 0 {
            anyhow::bail!("workers must be >= 1");
        }
        Ok(())
    }

    /// The hopper root, or `None` when none is set or it is blank.
    fn hopper(&self) -> Option<&str> {
        self.hopper.as_deref().filter(|s| !s.trim().is_empty())
    }
}

/// What a caller supplies to start a server, before resolution.
///
/// [`ServerConfig`] is the resolved form. This is the unresolved one — the
/// shape a command line hands over, with `None` meaning "take the default"
/// rather than "off". [`Startup::resolve`] settles every default in one
/// place, so two binaries serving this API serve it identically.
#[derive(Debug)]
pub struct Startup {
    /// Address to listen on.
    pub bind: SocketAddr,
    /// Largest request body accepted, in megabytes.
    pub max_size_mb: usize,
    /// The process RSS ceiling (`--max-rss-gb`).
    pub max_rss: crate::memory::MaxRssPolicy,
    /// Comma-separated directories `/analyze-path` may read. Empty means the
    /// route refuses every request, which is the safe default.
    pub allowed_dirs: Option<String>,
    /// Where archive members are extracted for the caller to fetch.
    pub extract_dir: Option<PathBuf>,
    /// Concurrent request slots. `None` takes the worker default.
    pub workers: Option<NonZeroUsize>,
    /// Comma-separated CIDRs allowed to reach non-loopback routes.
    pub allow_cidr: Option<String>,
    /// File holding the bearer token. `None` disables authentication.
    pub token_file: Option<PathBuf>,
    /// Hopper API root. Enables result renewal, corpus deferral, and the
    /// companion idle worker.
    pub hopper: Option<String>,
    /// Slots for the companion idle worker. `None` takes half the request
    /// slots; ignored without `hopper`, since there would be nothing to claim.
    pub idle_worker_slots: Option<usize>,
    /// Per-request analysis timeout in seconds. Zero disables.
    pub analysis_timeout_secs: u64,
    /// Rules, model and analysis settings, shared with the pull worker.
    /// Fetching is off by default: a server driving outbound fetches is an
    /// SSRF-shaped exposure, so turning it on is an operator decision.
    pub rules: crate::worker::RulesStartup,
}

impl Startup {
    /// Settle every default, read the bearer token, and validate the result.
    ///
    /// # Errors
    ///
    /// Returns an error when the model bundle cannot be resolved, a CIDR does
    /// not parse, the thresholds or worker count are invalid, or `token_file`
    /// is set but missing, empty or unreadable. That last one fails closed on
    /// purpose: an operator who asked for authentication must never get an
    /// open server because a file went away.
    pub fn resolve(self) -> anyhow::Result<ServerConfig> {
        let rules = self.rules.resolve()?;

        // Canonicalized at startup so symlink-resolved request paths match in
        // the `starts_with` checks `/analyze-path` gates on.
        let allowed_dirs: Vec<PathBuf> = self
            .allowed_dirs
            .unwrap_or_default()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| {
                let path = PathBuf::from(s);
                path.canonicalize().unwrap_or(path)
            })
            .collect();

        let workers = self
            .workers
            .unwrap_or_else(crate::worker::default_workers)
            .get();

        let allow_cidrs = match self.allow_cidr {
            Some(ref list) => {
                parse_cidr_list(list).map_err(|e| anyhow::anyhow!("--allow-cidr: {e}"))?
            }
            None => Vec::new(),
        };

        let auth_digest = match self.token_file {
            Some(ref path) => {
                let token = crate::interpret::read_token_file(path).ok_or_else(|| {
                    anyhow::anyhow!(
                        "--token-file {}: missing, empty, or unreadable",
                        path.display()
                    )
                })?;
                let digest = TokenDigest::new(&token)
                    .map_err(|e| anyhow::anyhow!("--token-file {}: {e}", path.display()))?;
                // Name the file the running process actually read: after a
                // rotation the token in a file and the token in memory can
                // differ, and the 401 that follows is otherwise unreadable.
                tracing::info!(
                    token_file = %path.display(),
                    "bearer authentication enabled (token is read once, at startup)",
                );
                Some(digest)
            }
            None => None,
        };

        let hopper = self.hopper.filter(|s| !s.trim().is_empty());
        // Half the request slots for background work. Disabled without
        // hopper: nothing to claim.
        let idle_worker_slots = match (hopper.as_deref(), self.idle_worker_slots) {
            (None, _) => 0,
            (Some(_), Some(n)) => n.min(workers / 2),
            (Some(_), None) => workers / 2,
        };

        let config = ServerConfig {
            bind: self.bind,
            max_body_size: self.max_size_mb.saturating_mul(1024 * 1024),
            max_rss_bytes: self.max_rss.process_ceiling(),
            model_dir: rules.model_dir,
            thresholds: rules.thresholds,
            level: rules.level,
            slow_rule_ms: rules.slow_rule_ms,
            allowed_dirs,
            extract_dir: self.extract_dir,
            workers,
            idle_worker_slots,
            allow_cidrs,
            auth_digest,
            analysis_timeout_secs: self.analysis_timeout_secs,
            interpret: rules.interpret,
            fetch: rules.fetch,
            hopper,
            zip_passwords: rules.zip_passwords,
            slot_lanes: (std::env::var("SCAN_SLOT_LANES").as_deref() == Ok("1"))
                .then(crate::analysis::small_job_max_bytes),
        };
        config.validate()?;
        Ok(config)
    }
}

/// Default per-request analysis timeout: 34 minutes. Covers cold cleave scans
/// of large archives — and fetch-enabled scans whose dependency analysis can
/// far outlast the sample's own — while still preventing a pathological input
/// from pinning a slot forever. Override with `--analysis-timeout`.
pub const DEFAULT_ANALYSIS_TIMEOUT_SECS: u64 = 2040;

#[cfg(test)]
mod cpu_busy_tests {
    use super::cores_busy;
    use cleave::memory_tracker::CpuTime;

    #[test]
    fn cores_busy_is_the_busy_share_of_the_machine() {
        let a = CpuTime {
            busy: 1000,
            idle: 3000,
        };
        let b = CpuTime {
            busy: 1300,
            idle: 3100,
        };
        // 300 busy of 400 elapsed ticks on 16 CPUs: twelve cores' worth.
        assert_eq!(cores_busy(a, b, 16), Some(12.0));
        assert_eq!(cores_busy(a, a, 16), None, "no elapsed ticks, no answer");
        assert_eq!(
            cores_busy(b, a, 16),
            None,
            "a counter that ran backwards is not a reading"
        );
    }

    /// A second reader inside the window gets the standing answer instead of
    /// resetting the window to the few milliseconds since the first.
    #[test]
    fn a_second_reader_does_not_shrink_the_window() {
        let busy = super::CpuBusy::default();
        let Some(first) = cleave::memory_tracker::cpu_time() else {
            return; // No counters on this platform: nothing to window.
        };
        let at = std::time::Instant::now();
        *super::lock(&busy.last) = Some(super::CpuSample {
            at,
            counters: first,
            busy: Some(3.5),
        });
        assert_eq!(busy.sample(), Some(3.5));
        let kept = super::lock(&busy.last).as_ref().map(|s| s.at);
        assert_eq!(kept, Some(at), "an early read moved the window");
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn server_config_rejects_invalid_thresholds() {
        let config = ServerConfig {
            thresholds: Some(Thresholds {
                suspicious: -0.1,
                hostile: 0.9,
            }),
            ..ServerConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn server_config_accepts_the_defaults() {
        let config = ServerConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.level.is_none());
        assert!(config.max_rss_bytes.is_none());
    }

    #[test]
    fn server_config_rejects_zero_workers() {
        let config = ServerConfig {
            workers: 0,
            ..ServerConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn a_blank_hopper_is_no_hopper() {
        let config = ServerConfig {
            hopper: Some("  ".into()),
            ..ServerConfig::default()
        };
        assert_eq!(config.hopper(), None);
    }
}

/// One admitted analysis, as `/_/requests` and the watchdog report it.
struct InFlightRequest {
    name: String,
    size_bytes: u64,
    started_at: Instant,
    /// Shared with the blocking task; set to true to request cooperative cancellation.
    cancellation: Arc<AtomicBool>,
    /// Tracks the current analysis phase inside cleave/litmus. Updated at each
    /// major stage so `/_/requests` can report what a stuck request is doing.
    phase: RequestPhase,
    /// OS thread ID of the blocking thread servicing this request (0 until started).
    thread_id: AtomicU64,
}

impl InFlightRequest {
    fn new(
        name: &str,
        size_bytes: u64,
        cancellation: Arc<AtomicBool>,
        phase: RequestPhase,
    ) -> Self {
        Self {
            name: name.to_owned(),
            size_bytes,
            started_at: Instant::now(),
            cancellation,
            phase,
            thread_id: AtomicU64::new(0),
        }
    }
}

/// What an admitted analysis holds: a request slot and a core.
///
/// Slots are sized for a request's whole life, most of which is waiting on
/// the network; cores are sized to the rayon pool, the only capacity an
/// analysis really contends for. Reporting slots alone told the router
/// `slots_free=48` on a box whose 16 cores were all busy (2026-09-04), and
/// the analysis it sent there waited five minutes to start.
pub(super) struct AnalysisPermit {
    _slot: tokio::sync::OwnedSemaphorePermit,
    _cpu: tokio::sync::OwnedSemaphorePermit,
}

impl AnalysisPermit {
    pub(super) fn new(
        slot: tokio::sync::OwnedSemaphorePermit,
        cpu: tokio::sync::OwnedSemaphorePermit,
    ) -> Self {
        Self {
            _slot: slot,
            _cpu: cpu,
        }
    }
}

/// RAII guard over one admitted analysis. On drop — the analysis finished, or
/// the handler future was dropped on a client disconnect — it signals
/// cooperative cancellation to the blocking thread and removes the in-flight
/// entry, so neither the permits nor the entry can leak.
pub(super) struct RequestGuard {
    request_id: u64,
    state: Arc<AppState>,
    cancellation: Arc<AtomicBool>,
    /// Held here so the slot and core are released when the guard drops.
    _permit: AnalysisPermit,
    /// Keeps the idle worker frozen for the whole analysis, which outlives the
    /// handler whenever the client hangs up before it finishes.
    _busy: idle::BusyToken,
}

impl RequestGuard {
    /// Register an admitted analysis on `/_/requests` and hold its capacity
    /// until the guard drops.
    fn new(
        state: &Arc<AppState>,
        request_id: u64,
        entry: InFlightRequest,
        permit: AnalysisPermit,
    ) -> Self {
        let cancellation = Arc::clone(&entry.cancellation);
        state.in_flight.insert(request_id, entry);
        state.jobs.started.fetch_add(1, Ordering::Relaxed);
        ACTIVE_REQUESTS.fetch_add(1, Ordering::AcqRel);
        Self {
            request_id,
            state: Arc::clone(state),
            cancellation,
            _permit: permit,
            _busy: state.enter_busy(),
        }
    }

    /// Run `work` on a blocking thread, bounded by `--analysis-timeout`.
    ///
    /// The guard is dropped when the work returns. On timeout it follows the
    /// blocking task instead — tokio cannot stop a blocking thread, only ask it
    /// via the cancellation flag — so the thread's slot and core stay
    /// accounted for as long as it runs.
    pub(super) async fn run(
        self,
        work: impl FnOnce() -> anyhow::Result<crate::engine::ScanResult> + Send + 'static,
    ) -> AnalysisOutcome {
        let state = Arc::clone(&self.state);
        let id = self.request_id;
        let mut handle = tokio::task::spawn_blocking(move || {
            if let Some(entry) = state.in_flight.get(&id) {
                entry
                    .thread_id
                    .store(crate::thread_dump::os_thread_id(), Ordering::Relaxed);
            }
            let result = work();
            // A long-lived server returns its thread-local caches now and then.
            if id.is_multiple_of(100) {
                cleave::clear_all_thread_caches();
            }
            result
        });
        let timeout_secs = self.state.config.analysis_timeout_secs;
        let joined = if timeout_secs == 0 {
            handle.await
        } else {
            match tokio::time::timeout(Duration::from_secs(timeout_secs), &mut handle).await {
                Ok(joined) => joined,
                Err(_) => {
                    self.follow(handle);
                    return AnalysisOutcome::Timeout(timeout_secs);
                }
            }
        };
        match joined {
            Ok(result) => AnalysisOutcome::Ok(result.map(Box::new)),
            Err(e) => AnalysisOutcome::JoinError(e),
        }
    }

    /// Keep the slot and core until a timed-out blocking task returns.
    ///
    /// A blocking thread cannot be stopped, only asked: the cancellation flag,
    /// which cleave polls between members. Until it answers it is still using
    /// a core, so handing its permits back at the timeout is how a node
    /// reports capacity it does not have — 14 such orphans beside
    /// `slots_free=48` on one box. The guard follows the thread out instead,
    /// and `stuck_orphans` counts threads still running, not timeouts there
    /// have ever been.
    fn follow<T: Send + 'static>(self, task: tokio::task::JoinHandle<T>) {
        self.cancellation.store(true, Ordering::Release);
        self.state.stuck_orphans.fetch_add(1, Ordering::Relaxed);
        let tasks = Arc::clone(&self.state.tasks);
        tasks.spawn(async move {
            let _ = task.await;
            self.state.stuck_orphans.fetch_sub(1, Ordering::Relaxed);
            drop(self);
        });
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        // The permit is released with `_permit`; `_busy` thaws the worker if
        // this was the last holder.
        self.cancellation.store(true, Ordering::Release);
        self.state.in_flight.remove(&self.request_id);
        ACTIVE_REQUESTS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Outcome of a blocking analysis awaited with a bound.
///
/// `Ok` boxes the `ScanResult` (≈376 B) so the idle-path variants — `Timeout`
/// and `JoinError` — don't carry that much padding each.
#[derive(Debug)]
pub(super) enum AnalysisOutcome {
    /// Task completed (inner `Result` is the analyzer's result).
    Ok(anyhow::Result<Box<crate::engine::ScanResult>>),
    /// Task join failed (panic, runtime shutdown, etc.).
    JoinError(tokio::task::JoinError),
    /// Task exceeded the configured timeout. The blocking thread keeps running
    /// until cleave observes the cancellation flag, and keeps its slot and
    /// core until then.
    Timeout(u64),
}

/// Upper bounds, in bytes, of each size bucket; the last is open-ended.
/// Chosen around where behaviour actually changes: a source tarball, a typical
/// package, a large archive, and the multi-hundred-megabyte inputs whose member
/// expansion dominates everything else.
pub(crate) const SIZE_BUCKETS: [u64; 4] = [1 << 20, 16 << 20, 128 << 20, u64::MAX];

/// Human labels for [`SIZE_BUCKETS`], used as JSON keys on `/_/stats`.
pub(crate) const SIZE_BUCKET_NAMES: [&str; 4] = ["le_1mb", "le_16mb", "le_128mb", "gt_128mb"];

/// Completion totals for one class of work.
///
/// Keeps two views because they answer different questions. The running totals
/// are cumulative-with-aging and say what this server has done; the windowed
/// [`latency::Latency`] says what it is doing *now*, and is what a router
/// reads. See that module for why a percentile over a time window beats a mean
/// over a sample count.
#[derive(Debug, Default)]
pub(crate) struct JobBucket {
    pub(crate) count: AtomicU64,
    pub(crate) micros: AtomicU64,
    pub(crate) recent: latency::Latency,
}

/// How many completions the cumulative totals remember before they start
/// forgetting. The windowed view has its own, time-based expiry.
const JOB_BUCKET_MEMORY: u64 = 256;

impl JobBucket {
    /// Record one completion in both views.
    pub(crate) fn record(&self, micros: u64) {
        self.recent.record(micros);
        let n = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        self.micros.fetch_add(micros, Ordering::Relaxed);
        if n >= JOB_BUCKET_MEMORY {
            // Racy by construction: two threads crossing the line together may
            // both halve. That costs a little extra forgetting and nothing else,
            // which is a better trade here than a lock on the hot path.
            self.count.fetch_sub(n / 2, Ordering::Relaxed);
            let m = self.micros.load(Ordering::Relaxed);
            self.micros.fetch_sub(m / 2, Ordering::Relaxed);
        }
    }

    /// Samples in the aged totals, and their mean in milliseconds.
    pub(crate) fn mean_ms(&self) -> (u64, Option<u64>) {
        let n = self.count.load(Ordering::Relaxed);
        let ms = (n > 0).then(|| self.micros.load(Ordering::Relaxed) / n / 1_000);
        (n, ms)
    }

    /// The windowed view, in milliseconds, for `/_/stats`.
    pub(crate) fn recent(&self) -> diag::Recent {
        let s = self.recent.summary();
        diag::Recent {
            samples: s.samples,
            p80_ms: s.p80_micros.map(|us| us / 1_000),
            mean_ms: s.mean_micros.map(|us| us / 1_000),
        }
    }
}

/// PURL types tracked separately on `/_/stats`, plus a catch-all.
///
/// A PURL carries no size, so a router choosing a worker for one has nothing to
/// look up in [`SIZE_BUCKET_NAMES`] until the artifact has already been fetched
/// — by which point the choice is made. The type is the next best predictor and
/// is known up front: a golang pseudo-version resolves to a repository clone, an
/// npm package to a small tarball, and the two are not comparable work.
pub(crate) const PURL_TYPE_NAMES: [&str; 5] = ["cargo", "golang", "npm", "pypi", "other"];

/// The bucket index for `purl`, matching on the type between `pkg:` and `/`.
pub(crate) fn purl_type_bucket(purl: &str) -> usize {
    let rest = purl.strip_prefix("pkg:").unwrap_or(purl);
    let ty = rest.split('/').next().unwrap_or("");
    // Only the type is case-insensitive per the PURL spec; the rest is not
    // touched here because nothing downstream of this counter reads it.
    PURL_TYPE_NAMES
        .iter()
        .position(|n| ty.eq_ignore_ascii_case(n))
        // `other` is the last name and is never matched by a real type.
        .filter(|i| *i + 1 < PURL_TYPE_NAMES.len())
        .unwrap_or(PURL_TYPE_NAMES.len() - 1)
}

/// The bucket index for an artifact of `size` bytes.
pub(crate) fn size_bucket(size: u64) -> usize {
    SIZE_BUCKETS
        .iter()
        .position(|&bound| size <= bound)
        .unwrap_or(SIZE_BUCKETS.len() - 1)
}

/// Analyses this server has begun and completed, and the figures a router
/// reads to choose it.
///
/// Counted rather than sampled: a router wants "how big and how slow are this
/// server's jobs, typically", and totals divided at read time answer that
/// without keeping a window. `started` minus `completed` is also the honest
/// count of work that went in and never came out.
#[derive(Debug, Default)]
pub(super) struct Jobs {
    started: AtomicU64,
    completed: AtomicU64,
    bytes_total: AtomicU64,
    micros_total: AtomicU64,
    /// Per-size-bucket completion totals, for the size-aware half of routing.
    ///
    /// One scalar average is not enough to choose a server. The 12.5s-vs-90s
    /// spread measured across two scanners on the same artifact was a large
    /// archive's member analysis, not a constant handicap — a single number
    /// would brand a box "slow" when it is only slow at big inputs, and send
    /// every small package somewhere worse. A caller usually knows the size
    /// before it dispatches, so the useful answer is per bucket.
    by_size: [JobBucket; SIZE_BUCKETS.len()],
    by_type: [JobBucket; PURL_TYPE_NAMES.len()],
    /// The blended average, aged like the others. Separate from `completed`,
    /// which stays a true lifetime count for reporting: one answers "how fast
    /// is this server now", the other "how much has it done".
    ///
    /// Fresh analyses only — see `cached`. So are `by_size` and `by_type`.
    overall: JobBucket,
    /// Analyses answered from cleave's analysis cache.
    ///
    /// Kept apart from the fresh numbers because mixing them makes every
    /// average bimodal and therefore useless for prediction: the same artifact
    /// is milliseconds on a hit and minutes on a miss. A router choosing a
    /// worker for work it has not done wants the fresh figure; blending in
    /// cache hits only tells it how lucky this server has been.
    cached: JobBucket,
    /// `/lookup` service time. Near-constant — an index probe, not an analysis
    /// — and so the honest input for ordering the cheap-source race, where the
    /// analysis averages would be wrong by three orders of magnitude.
    lookups: JobBucket,
}

impl Jobs {
    /// Count one finished analysis. Every analyze route reports here, so
    /// `started` minus `completed` stays the work that never came out.
    fn finished(&self, result: &crate::engine::ScanResult, elapsed_ms: u64, purl: Option<&str>) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.bytes_total
            .fetch_add(result.size_bytes, Ordering::Relaxed);
        self.micros_total
            .fetch_add(elapsed_ms.saturating_mul(1_000), Ordering::Relaxed);
        // What the router averages is this server's own service time: the LLM
        // phase is left out, because the endpoint is shared by the whole fleet
        // and a contended one made every worker that asked it look slow.
        // Measured 2026-09-06: a 128-core box restarted, its first twenty
        // samples were probes that each waited on the endpoint, its p80 read
        // 82s, and it took 2 of the next 128 dispatches while a 4-core box
        // took 52.
        let micros = elapsed_ms
            .saturating_sub(result.interpret_ms)
            .saturating_mul(1_000);
        // Routing predicts the cost of work this server has *not* done, so
        // only fresh analyses feed the figures a router reads. A cache hit is
        // real and worth reporting, but it predicts nothing about the next
        // unseen artifact.
        if result.analysis_cached {
            self.cached.record(micros);
            return;
        }
        self.overall.record(micros);
        self.by_size[size_bucket(result.size_bytes)].record(micros);
        // By PURL type too, when the job was named by one: the only cost signal
        // a router has before dispatch for `?purl=` work.
        if let Some(purl) = purl {
            self.by_type[purl_type_bucket(purl)].record(micros);
        }
    }
}

/// Class-aware admission, when `SCAN_SLOT_LANES=1`.
///
/// The flat slot semaphore treats every analysis as equal, but a large archive
/// fans out across the whole shared rayon pool while a small file uses roughly
/// one thread — so `--workers` flat slots either under-admit smalls or
/// co-schedule whales that then fight for the pool (measured +55% wall on
/// whale co-residency). The lanes mirror the worker's cleave gate at the front
/// door: smalls get most permits, whales get few, and a full lane answers 429
/// with Retry-After instead of queueing — a whale's queue wait is minutes, so
/// the fleet routes it to an idle server; a small's wait is seconds, so callers
/// just retry.
#[derive(Debug)]
pub(super) struct SlotLanes {
    pub(super) whale: Arc<Semaphore>,
    pub(super) small: Arc<Semaphore>,
    /// Jobs at or below this size take the small lane (`SCAN_SMALL_JOB_MB`).
    /// Unknown size — a PURL or URL analysis whose payload has not been
    /// fetched yet — is a whale: those are almost always packages, and
    /// mis-classing a whale as small is the expensive direction.
    pub(super) small_max_bytes: u64,
}

impl SlotLanes {
    fn new(max_concurrent: usize, small_max_bytes: u64) -> Self {
        let whale_permits = (1 + max_concurrent / 8).min(max_concurrent);
        let small_permits = max_concurrent.saturating_sub(whale_permits).max(1);
        tracing::info!(
            whale_permits,
            small_permits,
            small_max_mb = small_max_bytes / (1024 * 1024),
            "slot lanes enabled: class-aware admission (SCAN_SLOT_LANES)"
        );
        Self {
            whale: Arc::new(Semaphore::new(whale_permits)),
            small: Arc::new(Semaphore::new(small_permits)),
            small_max_bytes,
        }
    }

    pub(super) fn available(&self) -> usize {
        self.whale.available_permits() + self.small.available_permits()
    }
}

/// Whether the model bundle is loaded.
enum Readiness {
    Starting,
    /// Startup failed. The reason is logged; it is never served.
    Failed(String),
    Ready(Arc<ModelResources>),
}

/// How many cached-upload repairs may hold their bytes at once. Each holds up
/// to `--max-size-mb`; a repeat past this is skipped, and the next repeat of
/// the same artifact repairs it instead.
const MAX_REPAIRS: usize = 4;

/// How long shutdown waits for analyses no client is waiting on.
/// Comfortably inside systemd's default 90-second stop timeout.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// Longest an upload may go without delivering a byte. An idle bound, not a
/// total one: a large artifact on a slow link still arrives, but a client
/// dripping a body (or stalling mid-upload) cannot hold its buffer and its
/// busy mark open indefinitely.
const BODY_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

/// The refusal for a body that stalled past [`BODY_IDLE_TIMEOUT`].
fn body_idle_timeout() -> ApiError {
    ApiError::new(
        axum::http::StatusCode::REQUEST_TIMEOUT,
        "body_timeout",
        "The request body stalled.",
    )
}

struct AppState {
    config: ServerConfig,
    /// Process uptime anchor — captured when the app is built, very close to
    /// process start. `/_/health` reports `now - started_at` as `uptime_secs`.
    started_at: Instant,
    readiness: RwLock<Readiness>,
    next_request_id: AtomicU64,
    /// One permit per `--workers` slot. Each analysis holds one for its whole
    /// life; the permit drops with the analysis or with its orphan follower,
    /// so a slot is always released — even on panic or runtime shutdown.
    slots: Arc<Semaphore>,
    /// One permit per rayon thread, shared with the idle worker: every analysis
    /// in this process, whoever asked for it, runs on the same pool.
    cpu: Arc<Semaphore>,
    lanes: Option<SlotLanes>,
    /// Tasks stuck past the timeout — still occupying a slot until the
    /// blocking thread finally returns. Tracked for observability only.
    stuck_orphans: AtomicUsize,
    /// Serializes `/_/reload` and `/_/update`. The guard is moved into the
    /// blocking work, so it is held for as long as that work runs.
    reload_lock: Arc<tokio::sync::Mutex<()>>,
    overloaded_since: Mutex<Option<Instant>>,
    in_flight: dashmap::DashMap<u64, InFlightRequest>,
    /// Machine-wide cores busy between consecutive `/_/stats` reads.
    cpu_busy: CpuBusy,
    /// Raised when the server stops, so background tasks wind down with it.
    /// Dropping the state drops the sender, which they read the same way.
    shutdown: watch::Sender<bool>,
    /// Work no request is waiting on — analyses whose callers hung up, their
    /// index and upload tails — counted so shutdown can drain it.
    tasks: Arc<Tasks>,
    jobs: Jobs,
    /// Requests outstanding, and the worker they freeze. See [`idle::Busy`].
    busy: Arc<idle::Busy>,
    /// The companion pull worker, when one is running.
    idle_worker: Option<Arc<idle::Worker>>,
    /// Analyses in progress, so concurrent requests for the same artifact
    /// share one run instead of each taking a slot. See [`flight`].
    flights: Arc<flight::Flights>,
    /// Bounds the cached-upload repairs in flight; see [`MAX_REPAIRS`].
    repairs: Arc<Semaphore>,
    /// Background hopper uploader (`--hopper`); `None` disables result renewal.
    uploader: Option<Arc<crate::upload::Uploader>>,
    /// The corpus behind this worker's index. `None` when no hopper is
    /// configured, which leaves a lookup answering from local knowledge alone.
    corpus: Option<Arc<corpus::Corpus>>,
}

/// A poisoned lock means a panic happened while it was held. Everything kept
/// under these locks is replaced whole, so the data stays usable.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl AppState {
    /// Analyses this server can start right now: a slot and a core for each.
    pub(super) fn available_analysis_permits(&self) -> usize {
        let slots = match &self.lanes {
            Some(lanes) => lanes.available(),
            None => self.slots.available_permits(),
        };
        slots.min(self.cpu.available_permits())
    }

    /// Analyses running now.
    pub(super) fn active_tasks(&self) -> usize {
        self.config
            .workers
            .saturating_sub(self.available_analysis_permits())
    }

    fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// The loaded bundle, or why there is none.
    pub(super) fn resources(&self) -> Result<Arc<ModelResources>, ApiError> {
        match &*self
            .readiness
            .read()
            .unwrap_or_else(PoisonError::into_inner)
        {
            Readiness::Ready(resources) => Ok(Arc::clone(resources)),
            Readiness::Starting => Err(ApiError::starting()),
            Readiness::Failed(_) => Err(ApiError::init_failed()),
        }
    }

    /// The reason startup failed, if it did.
    pub(super) fn init_failure(&self) -> Option<String> {
        match &*self
            .readiness
            .read()
            .unwrap_or_else(PoisonError::into_inner)
        {
            Readiness::Failed(message) => Some(message.clone()),
            _ => None,
        }
    }

    pub(super) fn is_ready(&self) -> bool {
        matches!(
            *self
                .readiness
                .read()
                .unwrap_or_else(PoisonError::into_inner),
            Readiness::Ready(_)
        )
    }

    /// Wrap a loaded model in the request-independent settings every analysis
    /// runs with. The one place a bundle is assembled.
    fn bundle(&self, model: Model, shap: Option<ShapImportance>) -> ModelResources {
        ModelResources {
            model,
            shap,
            interpret: self.config.interpret.clone(),
            fetch: self.config.fetch,
            zip_passwords: self.config.zip_passwords.clone(),
        }
    }

    /// Serve `resources` from now on. Returns whether a bundle was already
    /// being served.
    fn install(&self, resources: ModelResources) -> bool {
        let mut readiness = self
            .readiness
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let was_ready = matches!(*readiness, Readiness::Ready(_));
        *readiness = Readiness::Ready(Arc::new(resources));
        was_ready
    }

    fn fail_startup(&self, message: String) {
        tracing::error!("{message}");
        *self
            .readiness
            .write()
            .unwrap_or_else(PoisonError::into_inner) = Readiness::Failed(message);
    }

    /// Refuse a request this server cannot take on: startup failed, or memory
    /// is past the ceiling. A server still loading is refused later, where a
    /// slot is claimed, so an answer it already holds can still be served.
    pub(super) async fn admit_request(&self, request_id: u64) -> Result<(), ApiError> {
        if let Some(message) = self.init_failure() {
            tracing::error!(id = request_id, error = %message, "rejected: startup failed");
            return Err(ApiError::init_failed());
        }
        self.check_memory().await
    }

    /// Mark the server busy until the returned token drops, freezing the
    /// companion worker for exactly that window.
    ///
    /// Taken at handler entry — before the memory check, before the multipart
    /// parse, before the upload streams — because the cores have to be free by
    /// the time the analysis wants them, not by the time it starts.
    pub(super) fn enter_busy(&self) -> idle::BusyToken {
        self.busy.enter(self.idle_worker.as_ref())
    }

    /// Whether a request is outstanding right now. While one is, the
    /// companion worker is frozen.
    pub(super) fn is_busy(&self) -> bool {
        self.busy.is_busy()
    }

    /// Whether a companion worker process is alive right now.
    pub(super) fn idle_worker_running(&self) -> bool {
        self.idle_worker.as_ref().is_some_and(|w| w.is_running())
    }

    /// Cores of this box's current load that a request will not queue behind,
    /// published as `background_in_flight`.
    ///
    /// All of them, or none: the worker is frozen for the whole of every
    /// request, so there is no partial answer to give.
    ///
    /// Counted in *logical* cores, because that is the unit `cpu_busy_cores`
    /// is in — a router subtracts one from the other, and the two must agree.
    /// Reporting physical cores here read correctly on a host without SMT and
    /// wrongly on one with it: ionos measured `cpu_busy_cores=10.7` against
    /// `physical_cpus=6` with its worker busy, so subtracting 6 left 0.78 of
    /// phantom foreground load, and a fully saturated worker would have left
    /// exactly 1.0 — beamline's `HOST_PRESSURE_LIMIT`, past which the box is
    /// dropped from routing for work that steps aside the moment it arrives.
    pub(super) fn sheddable_cores(&self) -> usize {
        if self.idle_worker_running() && !self.is_busy() {
            std::thread::available_parallelism().map_or(0, std::num::NonZero::get)
        } else {
            0
        }
    }

    /// When the server first went over its memory ceiling, if it still is.
    pub(super) fn overload_mark(&self) -> std::sync::MutexGuard<'_, Option<Instant>> {
        lock(&self.overloaded_since)
    }

    /// Check RSS against the ceiling, reclaiming caches once before refusing.
    async fn check_memory(&self) -> Result<(), ApiError> {
        // Throttling disabled: the operator delegated OOM enforcement to an
        // external supervisor.
        let Some(max_rss_bytes) = self.config.max_rss_bytes.map(NonZeroU64::get) else {
            return Ok(());
        };
        let Some(rss) = cleave::memory_tracker::current_rss() else {
            return Ok(());
        };
        if rss <= max_rss_bytes {
            if self.overload_mark().take().is_some() {
                tracing::info!(
                    rss_mb = rss / 1024 / 1024,
                    "memory recovered below threshold"
                );
            }
            return Ok(());
        }

        tracing::info!(
            rss_mb = rss / 1024 / 1024,
            "memory pressure detected, clearing thread-local caches"
        );
        // Awaited before re-reading RSS: a fire-and-forget clear let the
        // re-read run first, logged memory freed that was not, and admitted
        // requests an overloaded worker could not service.
        if let Err(e) = tokio::task::spawn_blocking(cleave::clear_all_thread_caches).await {
            tracing::warn!(error = %e, "cache-clear task failed");
        }

        let Some(rss_after) = cleave::memory_tracker::current_rss() else {
            return Ok(());
        };
        if rss_after <= max_rss_bytes {
            self.overload_mark().take();
            tracing::info!(
                rss_before_mb = rss / 1024 / 1024,
                rss_after_mb = rss_after / 1024 / 1024,
                "cache clear freed memory, accepting request"
            );
            return Ok(());
        }

        // Still overloaded. Never terminate; requests are refused until memory
        // drops, and restarting is the operator's call.
        let since = *self.overload_mark().get_or_insert_with(Instant::now);
        tracing::warn!(
            rss_mb = rss_after / 1024 / 1024,
            max_rss_mb = max_rss_bytes / 1024 / 1024,
            overloaded_secs = since.elapsed().as_secs(),
            "server overloaded: high memory usage (even after cache clear)"
        );
        Err(ApiError::overloaded())
    }

    /// Stop background work, cancel what no client is waiting on, and wait a
    /// bounded time for it to wind down — so an analysis that finished as the
    /// server stopped still files its verdict.
    async fn drain(&self) {
        self.shutdown.send_replace(true);
        for entry in &self.in_flight {
            entry.cancellation.store(true, Ordering::Release);
        }
        if tokio::time::timeout(DRAIN_TIMEOUT, self.tasks.drained())
            .await
            .is_err()
        {
            tracing::warn!(
                remaining = self.tasks.live(),
                "shutdown: background work still running after {}s; abandoning it",
                DRAIN_TIMEOUT.as_secs()
            );
        }
    }
}

/// Work the server owns but no request waits on: counted so shutdown can wait
/// for it, where a bare `tokio::spawn` would be dropped mid-write.
#[derive(Default)]
pub(super) struct Tasks {
    live: AtomicUsize,
    idle: Notify,
}

impl Tasks {
    pub(super) fn spawn(self: &Arc<Self>, task: impl Future<Output = ()> + Send + 'static) {
        self.live.fetch_add(1, Ordering::AcqRel);
        let done = TaskDone(Arc::clone(self));
        tokio::spawn(async move {
            // Dropped on completion and on cancellation alike.
            let _done = done;
            task.await;
        });
    }

    fn live(&self) -> usize {
        self.live.load(Ordering::Acquire)
    }

    /// Resolves once no task is running.
    async fn drained(&self) {
        loop {
            // Registered before the count is read, so a task finishing in
            // between still wakes this waiter.
            let notified = self.idle.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            if self.live() == 0 {
                return;
            }
            notified.await;
        }
    }
}

struct TaskDone(Arc<Tasks>);

impl Drop for TaskDone {
    fn drop(&mut self) {
        if self.0.live.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.idle.notify_waiters();
        }
    }
}

/// Cores busy across the whole machine, averaged between two reads of
/// `/_/stats`.
///
/// The router polls stats every few seconds, so each poll sees the mean over
/// the interval since the last one — the window that matters for deciding
/// where the next analysis goes. Derived from the kernel's cumulative CPU
/// counters rather than the load average because the load average is not the
/// same number on every platform: Linux counts threads blocked on disk, FreeBSD
/// does not, and a scan host does a great deal of disk. `None` until two
/// reads exist, and on platforms with no counters; the caller then falls back
/// to `load1`.
#[derive(Default)]
pub(super) struct CpuBusy {
    last: Mutex<Option<CpuSample>>,
}

struct CpuSample {
    at: Instant,
    counters: cleave::memory_tracker::CpuTime,
    busy: Option<f64>,
}

/// Reads closer together than this share one window. Without it a second
/// poller — another router, an operator's curl — reset the window for the
/// first and handed it the few milliseconds in between.
const CPU_BUSY_MIN_WINDOW: Duration = Duration::from_secs(1);

impl CpuBusy {
    /// Logical cores busy since the previous window closed, the standing
    /// answer inside a window or when the counters have not moved, or `None`
    /// with nothing to compare yet.
    pub(super) fn sample(&self) -> Option<f64> {
        let mut last = lock(&self.last);
        if let Some(prev) = last.as_ref()
            && prev.at.elapsed() < CPU_BUSY_MIN_WINDOW
        {
            return prev.busy;
        }
        let now = cleave::memory_tracker::cpu_time()?;
        let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        let busy = last
            .as_ref()
            .and_then(|prev| cores_busy(prev.counters, now, cpus).or(prev.busy));
        *last = Some(CpuSample {
            at: Instant::now(),
            counters: now,
            busy,
        });
        busy
    }
}

/// `Δbusy / (Δbusy + Δidle)` of the machine, times its logical CPUs. `None`
/// when the counters have not advanced (two reads inside one tick) or ran
/// backwards (a counter reset), so the caller keeps its previous answer.
fn cores_busy(
    prev: cleave::memory_tracker::CpuTime,
    now: cleave::memory_tracker::CpuTime,
    cpus: usize,
) -> Option<f64> {
    let busy = now.busy.checked_sub(prev.busy)?;
    let idle = now.idle.checked_sub(prev.idle)?;
    let total = busy.checked_add(idle)?;
    (total > 0).then(|| cpus as f64 * busy as f64 / total as f64)
}

/// Build the axum [`Router`] and start loading resources in the background.
///
/// The router answers immediately; until the model bundle loads, the health
/// endpoint and the analyze endpoints answer 503. Useful for integration tests
/// that need the app without binding a port. Background tasks stop when the
/// router is dropped.
///
/// # Errors
///
/// Returns an error if the configuration is invalid.
pub async fn build_app(config: &ServerConfig) -> anyhow::Result<Router> {
    Ok(assemble(config.clone())?.0)
}

/// The router and the state behind it. Must run inside a tokio runtime.
fn assemble(config: ServerConfig) -> anyhow::Result<(Router, Arc<AppState>)> {
    config.validate()?;
    tracing::info!(model_dir = %config.model_dir.display(), "starting — resources loading in background");

    // CPU-bound cleave + ONNX work overlaps poorly across many threads, so a
    // smaller pool typically delivers higher aggregate throughput than 1/core.
    let max_concurrent = config.workers;
    let cores = crate::worker::cleave_concurrency(max_concurrent);
    tracing::info!(max_concurrent, cores, "concurrency limit set");

    let (shutdown, _) = watch::channel(false);
    let busy = Arc::new(idle::Busy::default());
    // Started before the models load, not after: it is its own process and
    // loads its own, so there is nothing here for it to wait on.
    let idle_worker = if config.idle_worker_slots > 0 {
        idle::start(
            config.hopper(),
            &crate::upload::default_worker_name(),
            &busy,
            shutdown.subscribe(),
        )
    } else {
        tracing::info!("idle worker disabled: --idle-worker-slots is 0");
        None
    };

    // Said once here rather than on every analysis: a server nobody gave a
    // hopper still answers, but every verdict it computes dies with the
    // process, and that is worth one line at startup instead of silence.
    let uploader = match config.hopper() {
        Some(url) => Some(Arc::new(crate::upload::Uploader::new(
            url,
            crate::upload::default_worker_name(),
        ))),
        None => {
            tracing::warn!(
                "no --hopper configured: analyzed results are kept in this \
                 process's verdict index only and are never uploaded",
            );
            None
        }
    };
    let corpus = corpus::Corpus::new(config.hopper());
    match &corpus {
        Some(c) => tracing::info!(addresses = %c.addresses(), "lookups defer to the corpus"),
        // Not a warning: a worker with no corpus behind it answers from its
        // own index, which is a whole deployment rather than a broken one.
        None => tracing::info!("no hopper configured: lookups answer from the local index alone"),
    }

    let state = Arc::new(AppState {
        started_at: Instant::now(),
        readiness: RwLock::new(Readiness::Starting),
        next_request_id: AtomicU64::new(1),
        slots: Arc::new(Semaphore::new(max_concurrent)),
        lanes: config
            .slot_lanes
            .map(|small_max_bytes| SlotLanes::new(max_concurrent, small_max_bytes)),
        cpu: Arc::new(Semaphore::new(cores)),
        cpu_busy: CpuBusy::default(),
        stuck_orphans: AtomicUsize::new(0),
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        overloaded_since: Mutex::new(None),
        flights: Arc::new(flight::Flights::default()),
        in_flight: dashmap::DashMap::new(),
        shutdown,
        tasks: Arc::new(Tasks::default()),
        jobs: Jobs::default(),
        busy,
        idle_worker,
        repairs: Arc::new(Semaphore::new(MAX_REPAIRS)),
        uploader,
        corpus,
        config,
    });

    state.tasks.spawn(load_resources(Arc::clone(&state)));
    spawn_watchdog(&state);

    // `{"purl": …}` needs a few hundred bytes; the router-wide upload limit
    // would let it buffer a full artifact's worth of JSON before rejecting it.
    const PURL_BODY_MAX: usize = 64 * 1024;

    // No ConcurrencyLimitLayer: each analyze handler refuses past capacity
    // with a 429 rather than queueing. Layers apply bottom-up, so the last
    // `.layer()` runs first per request; the ACL runs before the body limit so
    // a rejected peer never gets to upload bytes.
    let app = Router::new()
        .route("/_/health", get(diag::health))
        .route("/_/info", get(diag::info))
        .route("/_/stats", get(diag::stats))
        .route("/_/reload", post(handlers::reload))
        .route("/_/update", post(handlers::update))
        .route("/_/memory", get(diag::memory_stats))
        .route("/_/requests", get(diag::requests))
        .route("/_/threads", get(diag::threads))
        .route("/lookup", get(handlers::lookup))
        .route("/status", get(handlers::status))
        .route("/v1/lookup", get(v1::v1_lookup))
        .route("/v1/analyze", post(v1::v1_analyze))
        .route("/analyze", post(handlers::analyze))
        .route(
            "/analyze-purl",
            post(handlers::analyze_purl).layer(DefaultBodyLimit::max(PURL_BODY_MAX)),
        )
        .route("/analyze-path", post(handlers::analyze_path))
        .layer(DefaultBodyLimit::max(state.config.max_body_size))
        .layer(middleware::from_fn_with_state(Arc::clone(&state), acl::acl))
        // Outermost: every request gets an id and an access-log line, including
        // the ones the ACL rejects.
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            access::access_log,
        ))
        .with_state(Arc::clone(&state));

    Ok((app, state))
}

/// Load the model, SHAP data and traits concurrently, then serve them.
///
/// The verdict index opens alongside: it is not needed to become ready, but
/// opening it creates directories and prunes stale ones, which belongs here
/// rather than on the first lookup.
async fn load_resources(state: Arc<AppState>) {
    let init_start = Instant::now();
    tracing::info!("resource loader started (model + SHAP + YARA loading concurrently)");
    let _index = tokio::task::spawn_blocking(|| crate::lookup::global().is_some());

    // Each blocking closure reports queue_ms (time waiting for a thread)
    // separately from work_ms (time actually doing I/O and parsing).
    let spawned = Instant::now();
    let model_dir = state.config.model_dir.clone();
    let (thresholds, level) = (state.config.thresholds, state.config.level);
    let model_task = tokio::task::spawn_blocking(move || -> anyhow::Result<Model> {
        let queue_ms = spawned.elapsed().as_millis();
        let t = Instant::now();
        tracing::info!(queue_ms, "loading ONNX model and feature spec");
        let model = Model::load(&model_dir, thresholds, level)?;
        tracing::info!(
            queue_ms,
            work_ms = t.elapsed().as_millis(),
            spec_version = model.spec().version(),
            features = model.spec().total_features(),
            "ONNX model loaded",
        );
        Ok(model)
    });
    let shap_dir = state.config.model_dir.clone();
    let shap_task = tokio::task::spawn_blocking(move || load_shap(&shap_dir));
    let slow_rule_ms = state.config.slow_rule_ms;
    let yara_task = tokio::task::spawn_blocking(move || -> Result<(), String> {
        let queue_ms = spawned.elapsed().as_millis();
        let t = Instant::now();
        tracing::info!(queue_ms, "YARA warmup started");
        // The traits tree is the rule set every analysis runs against. Resolve
        // it before reporting ready: a server that answers `/_/health` with
        // "ok" while failing every analysis on a missing traits directory is
        // worse than one that never starts.
        let traits = cleave::traits_repo::try_resolve()?;
        tracing::info!(dir = %traits.display(), "cleave traits resolved");
        let opts = cleave::AnalysisOptions {
            slow_rule_ms,
            ..Default::default()
        };
        // A warmup: its result is not the point, the compiled rules are.
        let _ = cleave::analyze_file(std::path::Path::new("/dev/null"), &opts);
        tracing::info!(
            queue_ms,
            work_ms = t.elapsed().as_millis(),
            "YARA warmup complete",
        );
        Ok(())
    });

    match tokio::join!(model_task, shap_task, yara_task) {
        (Ok(Ok(model)), Ok(Ok(shap)), Ok(Ok(()))) => {
            let spec_version = model.spec().version();
            let features = model.spec().total_features();
            let shap_loaded = shap.is_some();
            state.install(state.bundle(model, shap));
            tracing::info!(
                total_ms = init_start.elapsed().as_millis(),
                spec_version,
                features,
                shap_loaded,
                "server ready",
            );
        }
        (Ok(Err(e)), _, _) => state.fail_startup(format!("failed to load model: {e:#}")),
        (Err(e), _, _) => state.fail_startup(format!("model load task panicked: {e}")),
        (_, Ok(Err(e)), _) => state.fail_startup(format!("failed to load SHAP data: {e:#}")),
        (_, Err(e), _) => state.fail_startup(format!("shap load task panicked: {e}")),
        (_, _, Ok(Err(e))) => state.fail_startup(format!("traits unavailable: {e}")),
        (_, _, Err(e)) => state.fail_startup(format!("yara warmup task panicked: {e}")),
    }
}

/// SHAP importances; `None` when the bundle ships none. A file that is
/// present but unreadable or stale is bad model metadata, so startup fails.
fn load_shap(model_dir: &std::path::Path) -> anyhow::Result<Option<ShapImportance>> {
    let t = Instant::now();
    let shap = ShapImportance::load(model_dir)?;
    tracing::info!(
        work_ms = t.elapsed().as_millis(),
        loaded = shap.is_some(),
        "SHAP data"
    );
    Ok(shap)
}

/// Periodically log stuck in-flight requests, and signal cooperative
/// cancellation to those past the cancel threshold so cleave can bail out of
/// slow rules. Never terminates the process; that is the operator's call.
///
/// The threshold follows the analysis timeout: at least the historical 10
/// minutes, and always past `--analysis-timeout` itself (the request has
/// already 504'd by then; this reaps the orphaned blocking thread). A timeout
/// of 0 is an explicit opt-out of time limits, so the watchdog only logs.
///
/// Holds the state weakly and stops on shutdown, so it never outlives the app.
fn spawn_watchdog(state: &Arc<AppState>) {
    let cancel_after_secs = match state.config.analysis_timeout_secs {
        0 => None,
        t => Some(t.max(600)),
    };
    let weak = Arc::downgrade(state);
    let mut stop = state.shutdown.subscribe();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = interval.tick() => {}
                // Shutdown raised, or the state (and with it the sender) gone.
                _ = stop.changed() => return,
            }
            let Some(state) = weak.upgrade() else {
                return;
            };
            let active = state.active_tasks();
            if active == 0 {
                continue;
            }
            let stuck = state.stuck_orphans.load(Ordering::Relaxed);
            for entry in &state.in_flight {
                let elapsed_secs = entry.started_at.elapsed().as_secs();
                let phase = entry.phase.get();
                let tid = entry.thread_id.load(Ordering::Relaxed);
                if cancel_after_secs.is_some_and(|t| elapsed_secs >= t) {
                    entry.cancellation.store(true, Ordering::Release);
                    tracing::error!(
                        request_id = entry.key(),
                        name = %entry.name,
                        elapsed_secs,
                        phase,
                        thread_id = tid,
                        stuck_orphans = stuck,
                        active_tasks = active,
                        "watchdog: task past cancel threshold — cancellation signalled",
                    );
                } else if elapsed_secs >= 120 {
                    tracing::warn!(
                        request_id = entry.key(),
                        name = %entry.name,
                        elapsed_secs,
                        phase,
                        thread_id = tid,
                        stuck_orphans = stuck,
                        active_tasks = active,
                        "watchdog: long-running task",
                    );
                }
            }
        }
    });
}

/// Start the HTTP server and serve until `SIGINT` or `SIGTERM`.
///
/// Binds the configured address, starts background resource loading, and on
/// a signal stops accepting, lets open requests finish, then drains the work
/// no client is waiting on.
///
/// # Errors
/// Returns an error if the configuration is invalid, the listening socket
/// cannot be bound, or the server fails while serving requests.
pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    // Warm cleave's YARA engine + capability mapper off the rayon pool before
    // the listener binds. The first request's analysis spawns rayon work; if
    // one of those rayon workers is the first to hit `yara_engine()`, init's
    // internal par_iter deadlocks against its peers parked on the OnceLock.
    // Prefetching from a non-rayon thread here avoids the race entirely.
    crate::engine::prefetch_cleave_resources();

    // Server mode processes many files over a long lifetime. Configure jemalloc
    // to aggressively return freed pages to the OS, preventing multi-GB RSS
    // growth from allocator fragmentation across thousands of analyses.
    cleave::memory_tracker::configure_jemalloc_low_memory();

    // Enforces the RSS ceiling on wall-clock time, independent of request
    // traffic: memory can grow between requests (fragmentation, background
    // YARA work). Skipped when throttling is disabled.
    let _rss_logger = config.max_rss_bytes.map(|limit| {
        cleave::memory_tracker::start_periodic_logging(Duration::from_secs(10), limit.get())
    });

    let (app, state) = assemble(config)?;
    let config = &state.config;

    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    eprintln!(
        "Listening on http://{} (max size: {} MB, starting up) — Press Ctrl+C to stop",
        config.bind,
        config.max_body_size / 1024 / 1024,
    );
    // The startup line is the record of what this process actually is: an
    // operator reading the log after a restart should not have to reconstruct
    // the running configuration from the unit file.
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        pid = std::process::id(),
        bind = %config.bind,
        max_body_mb = config.max_body_size / 1024 / 1024,
        analysis_timeout_secs = config.analysis_timeout_secs,
        allow_cidrs = config.allow_cidrs.len(),
        allowed_dirs = config.allowed_dirs.len(),
        authenticated = config.auth_digest.is_some(),
        "listening (resources loading in background)",
    );

    // An unauthenticated API is open to anyone who can reach the socket. Warn
    // unconditionally — a loopback bind is not evidence of safety, because a
    // Cloudflare tunnel terminates on loopback and puts the whole internet on
    // the other side of it.
    if config.auth_digest.is_none() {
        tracing::warn!(
            "no --token-file: the API is unauthenticated; any peer that reaches the socket can submit work",
        );
    }

    // /analyze-path reads any file under --allowed-dirs and is restricted to
    // loopback peers — but a tunnel makes every peer a loopback peer, so that
    // restriction stops protecting it. Leave --allowed-dirs empty unless the
    // host is genuinely local-only; with no allowed directory the route
    // rejects every request.
    if !config.allowed_dirs.is_empty() {
        tracing::warn!(
            allowed_dirs = config.allowed_dirs.len(),
            "--allowed-dirs is set: /analyze-path can read those directories for any peer reaching loopback, including through a tunnel",
        );
    }

    // Operator footgun: setting --allow-cidr while bound to loopback means
    // the CIDR list can never match (no remote peers can connect). Warn so
    // the operator notices before debugging "why is everyone getting 403?".
    if !config.allow_cidrs.is_empty() && config.bind.ip().is_loopback() {
        tracing::warn!(
            bind = %config.bind,
            "--allow-cidr is set but bind address is loopback; remote clients cannot connect (use --bind 0.0.0.0:PORT)",
        );
    }

    // ConnectInfo<SocketAddr> is required by the ACL middleware so it can
    // see the peer IP. Tests inject ConnectInfo manually on each Request.
    let stopping = Arc::clone(&state);
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        shutdown_signal().await;
        // Background tasks stop now; open requests finish first.
        stopping.shutdown.send_replace(true);
    })
    .await?;

    state.drain().await;
    tracing::info!("server shut down");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = signal::ctrl_c().await {
            tracing::warn!("failed to install Ctrl+C handler: {e}");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::warn!("failed to install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("received SIGINT"),
        () = terminate => tracing::info!("received SIGTERM"),
    }
}

#[cfg(test)]
mod size_bucket_tests {
    use super::{SIZE_BUCKET_NAMES, SIZE_BUCKETS, size_bucket};

    /// Every bucket has a label, or `/_/stats` would silently drop one.
    #[test]
    fn every_bucket_is_named() {
        assert_eq!(SIZE_BUCKETS.len(), SIZE_BUCKET_NAMES.len());
    }

    /// The boundaries are inclusive upper bounds and the last is open-ended, so
    /// no size — including zero and u64::MAX — can fall outside.
    #[test]
    fn every_size_lands_in_a_bucket() {
        for size in [0, 1, 1 << 20, (1 << 20) + 1, 16 << 20, 128 << 20, u64::MAX] {
            let i = size_bucket(size);
            assert!(
                i < SIZE_BUCKETS.len(),
                "size {size} fell outside the buckets"
            );
        }
    }

    /// Boundaries are inclusive: an artifact of exactly 1 MiB is a small one,
    /// not the first of the next class up.
    #[test]
    fn boundaries_are_inclusive_and_ordered() {
        assert_eq!(size_bucket(0), 0);
        assert_eq!(size_bucket(1 << 20), 0);
        assert_eq!(size_bucket((1 << 20) + 1), 1);
        assert_eq!(size_bucket(16 << 20), 1);
        assert_eq!(size_bucket((16 << 20) + 1), 2);
        assert_eq!(size_bucket(128 << 20), 2);
        assert_eq!(size_bucket((128 << 20) + 1), 3);
        assert_eq!(size_bucket(u64::MAX), 3);
        // Monotonic: a bigger artifact never lands in an earlier bucket.
        let mut last = 0;
        for size in [0_u64, 1 << 10, 1 << 20, 1 << 24, 1 << 27, 1 << 30, u64::MAX] {
            let i = size_bucket(size);
            assert!(i >= last, "bucket went backwards at {size}");
            last = i;
        }
    }
}

#[cfg(test)]
mod purl_type_tests {
    use super::{PURL_TYPE_NAMES, purl_type_bucket};

    #[test]
    fn known_types_get_their_own_bucket() {
        for (i, name) in PURL_TYPE_NAMES.iter().enumerate().take(4) {
            assert_eq!(purl_type_bucket(&format!("pkg:{name}/thing@1.0")), i);
        }
    }

    #[test]
    fn the_type_is_case_insensitive_and_pkg_is_optional() {
        assert_eq!(purl_type_bucket("pkg:PyPI/requests@2.0"), 3);
        assert_eq!(purl_type_bucket("npm/left-pad@1.0"), 2);
    }

    #[test]
    fn unknown_and_malformed_fall_into_other() {
        let other = PURL_TYPE_NAMES.len() - 1;
        assert_eq!(purl_type_bucket("pkg:maven/g/a@1"), other);
        assert_eq!(purl_type_bucket(""), other);
        // "other" is a bucket name, not a type: a PURL literally spelled that
        // way must not be mistaken for a real match on it.
        assert_eq!(purl_type_bucket("pkg:other/x@1"), other);
    }

    #[test]
    fn a_golang_module_path_keeps_its_slashes_out_of_the_type() {
        assert_eq!(
            purl_type_bucket("pkg:golang/github.com/spf13/cobra@v1.10.2"),
            1
        );
    }
}

#[cfg(test)]
mod job_bucket_tests {
    use super::{JOB_BUCKET_MEMORY, JobBucket};

    fn mean(b: &JobBucket) -> u64 {
        let n = b.count.load(std::sync::atomic::Ordering::Relaxed);
        b.micros.load(std::sync::atomic::Ordering::Relaxed) / n.max(1)
    }

    #[test]
    fn aging_preserves_the_mean_of_a_steady_stream() {
        let b = JobBucket::default();
        for _ in 0..JOB_BUCKET_MEMORY * 4 {
            b.record(1_000);
        }
        assert_eq!(mean(&b), 1_000, "halving must not shift a constant mean");
        assert!(
            b.count.load(std::sync::atomic::Ordering::Relaxed) <= JOB_BUCKET_MEMORY,
            "memory is unbounded",
        );
    }

    #[test]
    fn an_incident_is_forgotten_once_normal_work_resumes() {
        let b = JobBucket::default();
        // Normal, then an outage's worth of multi-minute jobs, then normal again.
        for _ in 0..200 {
            b.record(5_000_000); // 5s
        }
        for _ in 0..30 {
            b.record(3_300_000_000); // 55 min, the real figure from the outage
        }
        let poisoned = mean(&b);
        assert!(
            poisoned > 100_000_000,
            "test setup failed to poison the mean"
        );
        for _ in 0..JOB_BUCKET_MEMORY * 6 {
            b.record(5_000_000);
        }
        let recovered = mean(&b);
        assert!(
            recovered < 6_000_000,
            "still poisoned after recovery: {recovered}us (was {poisoned}us)",
        );
    }
}

#[cfg(test)]
mod job_bucket_recent_tests {
    use super::JobBucket;

    fn recent(b: &JobBucket) -> serde_json::Value {
        serde_json::to_value(b.recent()).expect("serializes")
    }

    // `recent` is what ships on /_/stats, and beamline indexes it by these
    // exact names. Asserting the shape here is what stops a rename from
    // silently demoting the router back to lifetime means — a failure that
    // looks like nothing at all from the outside.
    #[test]
    fn recent_publishes_the_keys_beamline_reads() {
        let b = JobBucket::default();
        b.record(9_000_000); // 9s
        let v = recent(&b);
        assert_eq!(v["samples"], 1);
        assert!(
            v["p80_ms"].is_number(),
            "p80_ms missing or not a number: {v}"
        );
        assert!(
            v["mean_ms"].is_number(),
            "mean_ms missing or not a number: {v}"
        );
    }

    #[test]
    fn recent_reports_an_untouched_bucket_as_empty_not_zero() {
        let v = recent(&JobBucket::default());
        assert_eq!(v["samples"], 0);
        assert!(
            v["p80_ms"].is_null(),
            "an unsampled class must not claim 0ms"
        );
    }

    // The cumulative and windowed views answer different questions and must
    // both advance: routing reads one, operators read the other.
    #[test]
    fn record_feeds_both_the_lifetime_and_the_windowed_view() {
        let b = JobBucket::default();
        for _ in 0..5 {
            b.record(2_000_000);
        }
        assert_eq!(b.count.load(std::sync::atomic::Ordering::Relaxed), 5);
        assert_eq!(recent(&b)["samples"], 5);
    }
}

#[cfg(test)]
mod tasks_tests {
    use super::Tasks;
    use std::sync::Arc;

    /// Drain waits for every owned task, including one that finishes after
    /// the drain began — the wakeup must not be lost between the count and
    /// the wait.
    #[tokio::test]
    async fn drain_waits_for_owned_work() {
        let tasks = Arc::new(Tasks::default());
        tasks.drained().await;

        let (go, wait) = tokio::sync::oneshot::channel::<()>();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&done);
        tasks.spawn(async move {
            let _ = wait.await;
            flag.store(true, std::sync::atomic::Ordering::Release);
        });
        assert_eq!(tasks.live(), 1);
        let _ = go.send(());
        tasks.drained().await;
        assert!(done.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(tasks.live(), 0);
    }
}
