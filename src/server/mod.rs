//! HTTP API server for litmus malware classification.
//!
//! Accepts file uploads via multipart/form-data, runs cleave static analysis
//! and ONNX model inference, and returns a unified JSON result including
//! classification, SHAP explanations, and the full cleave report.
//!
//! Routes:
//!   GET  /_/health      — liveness check
//!   GET  /lookup        — stored verdict by ?sha256= or ?purl= (no slot)
//!   POST /analyze       — upload a file, receive full classification JSON
//!   POST /analyze-purl  — fetch a PURL (registry provenance included) and analyze
//!   POST /analyze-path  — analyze a local path (loopback)
//!   POST /_/reload      — hot-reload model from disk
//!
//! [`ServerConfig`] keeps the public server surface intentionally small:
//! validated thresholds are supplied up front, and callers use accessors
//! rather than mutating fields after construction.

mod access;
mod acl;
mod corpus;
mod decision;
mod flight;
mod handlers;
mod latency;

pub use acl::{Cidr, TokenDigest, parse_cidr_list};
pub(crate) use handlers::classify_bytes;
pub(crate) use handlers::classify_file;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use std::net::SocketAddr;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};
use tokio::signal;

use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Model, Thresholds};

/// Immutable configuration for the HTTP API server.
///
/// Construct with [`ServerConfig::new`] so thresholds are validated before the
/// listener starts and background resource loading begins.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    bind: SocketAddr,
    max_body_size: usize,
    max_rss_bytes: Option<NonZeroU64>,
    model_dir: PathBuf,
    thresholds: Option<Thresholds>,
    slow_rule_ms: u64,
    allowed_dirs: Vec<PathBuf>,
    extract_dir: Option<PathBuf>,
    workers: usize,
    allow_cidrs: Vec<Cidr>,
    /// Bearer token required on every route except `/_/health`; `None`
    /// disables authentication. Only the digest is kept — see [`TokenDigest`].
    auth_digest: Option<TokenDigest>,
    level: Option<u16>,
    /// Per-request analysis timeout in seconds. 0 disables.
    analysis_timeout_secs: u64,
    interpret: Option<crate::interpret::InterpretConfig>,
    /// External-reference fetch policy. Off by default: an upload server driving
    /// outbound fetches is an SSRF-shaped exposure (the transport's resolver
    /// guards internal IPs, but enabling it is an explicit operator decision).
    fetch: crate::fetch::FetchPolicy,
    /// hopper master API root (`--hopper`); when set, every analyzed result —
    /// parent and members — is renewed on hopper's `/api/result`. `None` disables
    /// upload, leaving the server a pure analyze service.
    hopper: Option<String>,
    /// Additional passwords to try for encrypted archives.
    zip_passwords: crate::ArchivePasswords,
    /// Analysis slots the idle worker may use. `0` disables it. The value is
    /// capped at half of `workers`, leaving the other half for interactive
    /// analyses.
    ///
    /// Idle capacity is otherwise wasted: a scan server spends most of its life
    /// waiting for the next request while hopper holds a queue of work. The
    /// worker fills that gap and pauses the moment a request arrives.
    ///
    /// Deliberately fewer than `max_concurrent_tasks`: the difference is the
    /// interactive reserve. Pausing stops new claims but does not abandon a job
    /// already running, so without slots held back a request could still queue
    /// behind background work — the reserve is what keeps the answer prompt.
    idle_worker_slots: usize,
}

/// Default per-request analysis timeout: 34 minutes. Covers cold cleave scans
/// of large archives — and fetch-enabled scans whose dependency analysis can
/// far outlast the sample's own — while still preventing a pathological input
/// from pinning a slot forever. Override with `--analysis-timeout` /
/// [`ServerConfig::with_analysis_timeout`].
pub const DEFAULT_ANALYSIS_TIMEOUT_SECS: u64 = 2040;

/// After the most recent `/analyze` request, keep the embedded hopper worker
/// paused for this long before it starts claiming queue work again.
pub const IDLE_WORKER_QUIET_SECS: u64 = 7;

impl ServerConfig {
    /// Create a server configuration.
    ///
    /// `thresholds` may be `None` to use the model's recommended thresholds
    /// from `evaluation.json`, or `Some(t)` to override with explicit values.
    ///
    /// `max_body_size` and `max_rss_bytes` are byte counts. A `max_rss_bytes`
    /// of `0` disables in-process RSS throttling — the server will not reject
    /// requests on memory pressure (use this when an external supervisor like
    /// systemd `MemoryMax=` already enforces a hard cap).
    ///
    /// `workers` is the maximum number of concurrent analyses; requests beyond
    /// this are rejected with 503 by the per-handler hard gate.
    ///
    /// `allow_cidrs` lists peer networks (in addition to loopback) that may
    /// reach the server. The `/analyze-path` endpoint is always restricted
    /// to loopback regardless of this list.
    ///
    /// # Example
    /// ```
    /// use scan::server::ServerConfig;
    ///
    /// let config = ServerConfig::new(
    ///     "127.0.0.1:49999".parse()?,
    ///     100 * 1024 * 1024,
    ///     8 * 1024 * 1024 * 1024,
    ///     "/path/to/models",
    ///     None,
    ///     4_000,
    ///     vec![],
    ///     None,
    ///     2,
    ///     vec![],
    /// )?;
    ///
    /// assert_eq!(config.max_body_size(), 100 * 1024 * 1024);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[allow(clippy::too_many_arguments)] // ServerConfig is plumbed once at startup; a builder would add ceremony for no real benefit.
    pub fn new(
        bind: SocketAddr,
        max_body_size: usize,
        max_rss_bytes: u64,
        model_dir: impl Into<PathBuf>,
        thresholds: Option<Thresholds>,
        slow_rule_ms: u64,
        allowed_dirs: Vec<PathBuf>,
        extract_dir: Option<PathBuf>,
        workers: usize,
        allow_cidrs: Vec<Cidr>,
    ) -> anyhow::Result<Self> {
        if let Some(ref t) = thresholds {
            t.validate()
                .map_err(|error| anyhow::anyhow!("invalid thresholds: {error}"))?;
        }
        if workers == 0 {
            return Err(anyhow::anyhow!("workers must be >= 1"));
        }
        Ok(Self {
            bind,
            max_body_size,
            max_rss_bytes: NonZeroU64::new(max_rss_bytes),
            model_dir: model_dir.into(),
            thresholds,
            slow_rule_ms,
            allowed_dirs,
            extract_dir,
            workers,
            allow_cidrs,
            auth_digest: None,
            level: None,
            analysis_timeout_secs: DEFAULT_ANALYSIS_TIMEOUT_SECS,
            interpret: None,
            fetch: crate::fetch::FetchPolicy::default(),
            hopper: None,
            zip_passwords: crate::ArchivePasswords::default(),
            idle_worker_slots: 0,
        })
    }

    /// Attach a hopper master API root (`--hopper`); when set, the server renews
    /// every analyzed result (parent and members) on hopper's `/api/result`.
    #[must_use]
    pub fn with_hopper(mut self, hopper: Option<String>) -> Self {
        self.hopper = hopper.filter(|s| !s.trim().is_empty());
        self
    }

    /// Add passwords to try when cleave encounters encrypted archives.
    #[must_use]
    pub fn with_zip_passwords(mut self, passwords: impl Into<crate::ArchivePasswords>) -> Self {
        self.zip_passwords = passwords.into();
        self
    }

    /// Set how many analysis slots the idle worker may use; `0` disables it.
    ///
    /// Capped at twice its core budget rather than at half the server's slots.
    /// "Half the slots" meant half the machine when a slot was a core; since
    /// slots were sized at three per core (2026-09-03) it meant one and a half
    /// machines, and on a 128-core box the pull worker held 192 slots and 576
    /// claims. A slot's work ends at dispatch, so two per core is enough
    /// staging to keep the cores fed and no more claims than that in hand.
    #[must_use]
    pub fn with_idle_worker_slots(mut self, slots: usize) -> Self {
        let cores = idle_worker_cores(crate::worker::cleave_concurrency(self.workers));
        self.idle_worker_slots = slots.min(cores.saturating_mul(2));
        self
    }

    /// Analysis slots available to the idle worker.
    #[must_use]
    pub fn idle_worker_slots(&self) -> usize {
        self.idle_worker_slots
    }

    /// The configured hopper upload root, or `None` when `--hopper` was not set.
    #[must_use]
    pub fn hopper(&self) -> Option<&str> {
        self.hopper.as_deref()
    }

    /// Attach an LLM interpretation config (`--interpret`); `None` disables it.
    #[must_use]
    pub fn with_interpret(mut self, interpret: Option<crate::interpret::InterpretConfig>) -> Self {
        self.interpret = interpret;
        self
    }

    /// Set the external-reference fetch policy (off by default). Enabling it on
    /// the server makes uploaded samples drive outbound fetches.
    #[must_use]
    pub const fn with_fetch(mut self, policy: crate::fetch::FetchPolicy) -> Self {
        self.fetch = policy;
        self
    }

    /// The server's external-reference fetch policy.
    #[must_use]
    pub(crate) const fn fetch(&self) -> crate::fetch::FetchPolicy {
        self.fetch
    }

    /// LLM interpretation config, or `None` when `--interpret` was not set.
    #[must_use]
    pub fn interpret(&self) -> Option<&crate::interpret::InterpretConfig> {
        self.interpret.as_ref()
    }

    /// Attach the FPR severity level (0..=10000) that produced the resolved
    /// thresholds. Folded into `ml.lvl` in the JSON envelope.
    #[must_use]
    pub const fn with_level(mut self, level: Option<u16>) -> Self {
        self.level = level;
        self
    }

    /// Set the per-request analysis timeout in seconds (`--analysis-timeout`).
    /// 0 disables the timeout. Defaults to [`DEFAULT_ANALYSIS_TIMEOUT_SECS`].
    #[must_use]
    pub const fn with_analysis_timeout(mut self, secs: u64) -> Self {
        self.analysis_timeout_secs = secs;
        self
    }

    /// Per-request analysis timeout in seconds. 0 = disabled.
    #[must_use]
    pub const fn analysis_timeout_secs(&self) -> u64 {
        self.analysis_timeout_secs
    }

    /// Severity level (0..=10000) used to pick thresholds, or `None` for manual
    /// thresholds.
    #[must_use]
    pub const fn level(&self) -> Option<u16> {
        self.level
    }

    /// Directory for extracting archive members.
    #[must_use]
    pub fn extract_dir(&self) -> Option<&std::path::Path> {
        self.extract_dir.as_deref()
    }

    /// Address the HTTP server binds to.
    #[must_use]
    pub const fn bind(&self) -> SocketAddr {
        self.bind
    }

    /// Maximum request body size in bytes.
    #[must_use]
    pub const fn max_body_size(&self) -> usize {
        self.max_body_size
    }

    /// Maximum RSS before rejecting requests, or `None` when in-process RSS
    /// throttling is disabled (constructed with `0`).
    #[must_use]
    pub const fn max_rss_bytes(&self) -> Option<NonZeroU64> {
        self.max_rss_bytes
    }

    /// Directory containing model artifacts.
    #[must_use]
    pub fn model_dir(&self) -> &std::path::Path {
        &self.model_dir
    }

    /// Explicit threshold overrides, if any. `None` means use model defaults.
    #[must_use]
    pub const fn thresholds(&self) -> Option<Thresholds> {
        self.thresholds
    }

    /// Warn when a single cleave rule exceeds this duration in milliseconds.
    #[must_use]
    pub const fn slow_rule_ms(&self) -> u64 {
        self.slow_rule_ms
    }

    /// Directories allowed for `/analyze-path` requests.
    #[must_use]
    pub fn allowed_dirs(&self) -> &[PathBuf] {
        &self.allowed_dirs
    }

    /// Maximum number of concurrent analyses.
    #[must_use]
    pub const fn workers(&self) -> usize {
        self.workers
    }

    /// Networks (in addition to loopback) allowed to connect to the server.
    /// `/analyze-path` is always restricted to loopback regardless.
    #[must_use]
    pub fn allow_cidrs(&self) -> &[Cidr] {
        &self.allow_cidrs
    }

    /// Require `Authorization: Bearer <token>` on every route except
    /// `/_/health` (`--token-file`). `None` leaves the API unauthenticated.
    ///
    /// Loopback peers are **not** exempt: behind a Cloudflare tunnel,
    /// `cloudflared` connects over loopback, so every remote request arrives
    /// with a loopback peer address.
    #[must_use]
    pub const fn with_auth_token(mut self, digest: Option<TokenDigest>) -> Self {
        self.auth_digest = digest;
        self
    }

    /// Digest of the required bearer token, or `None` when the API is
    /// unauthenticated.
    #[must_use]
    pub const fn auth_digest(&self) -> Option<TokenDigest> {
        self.auth_digest
    }
}

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
}

#[cfg(test)]
mod idle_worker_cores_tests {
    use super::idle_worker_cores;

    /// The idle worker never gets the whole pool, and never zero of it.
    #[test]
    fn idle_worker_leaves_a_reserve_for_interactive_work() {
        assert_eq!(idle_worker_cores(16), 8);
        assert_eq!(idle_worker_cores(128), 64);
        assert_eq!(idle_worker_cores(3), 1);
        assert_eq!(idle_worker_cores(2), 1);
        assert_eq!(idle_worker_cores(1), 1);
        for cores in 1..=256 {
            let budget = idle_worker_cores(cores);
            assert!(budget >= 1, "cores={cores}");
            assert!(
                cores == 1 || budget < cores,
                "cores={cores} budget={budget}"
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod config_tests {
    use super::*;

    #[test]
    fn server_config_rejects_invalid_thresholds() {
        let result = ServerConfig::new(
            SocketAddr::from(([127, 0, 0, 1], 8081)),
            100 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            "/tmp/models",
            Some(Thresholds {
                suspicious: -0.1,
                hostile: 0.9,
            }),
            4_000,
            vec![],
            None,
            2,
            vec![],
        );

        assert!(result.is_err());
    }

    #[test]
    fn server_config_accepts_none_thresholds() {
        let result = ServerConfig::new(
            SocketAddr::from(([127, 0, 0, 1], 8081)),
            100 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            "/tmp/models",
            None,
            4_000,
            vec![],
            None,
            2,
            vec![],
        );

        assert!(result.is_ok());
    }

    #[test]
    fn server_config_rejects_zero_workers() {
        let result = ServerConfig::new(
            SocketAddr::from(([127, 0, 0, 1], 8081)),
            100 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            "/tmp/models",
            None,
            4_000,
            vec![],
            None,
            0,
            vec![],
        );

        assert!(result.is_err());
    }

    #[test]
    fn server_config_level_defaults_to_none() {
        let config = ServerConfig::new(
            SocketAddr::from(([127, 0, 0, 1], 8081)),
            100 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            "/tmp/models",
            None,
            4_000,
            vec![],
            None,
            2,
            vec![],
        )
        .expect("valid config");
        assert!(config.level().is_none());
    }

    #[test]
    fn server_config_with_level_persists() {
        let config = ServerConfig::new(
            SocketAddr::from(([127, 0, 0, 1], 8081)),
            100 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            "/tmp/models",
            None,
            4_000,
            vec![],
            None,
            2,
            vec![],
        )
        .expect("valid config")
        .with_level(Some(5));
        assert_eq!(config.level(), Some(5));
    }

    #[test]
    fn server_config_keeps_archive_passwords() {
        let config = ServerConfig::new(
            SocketAddr::from(([127, 0, 0, 1], 8081)),
            100 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            "/tmp/models",
            None,
            4_000,
            vec![],
            None,
            2,
            vec![],
        )
        .expect("valid config")
        .with_zip_passwords(vec!["private".to_string()]);
        assert_eq!(config.zip_passwords.as_slice(), ["private"]);
    }
}

#[derive(Debug)]
struct InFlightRequest {
    name: String,
    size_bytes: u64,
    started_at: Instant,
    /// Shared with the blocking task; set to true to request cooperative cancellation.
    cancellation: Arc<AtomicBool>,
    /// Tracks the current analysis phase inside cleave/litmus. Updated at each
    /// major stage so `/_/requests` can report what a stuck request is doing.
    phase: cleave::PhaseTracker,
    /// OS thread ID of the blocking thread servicing this request (0 until started).
    thread_id: AtomicU64,
}

/// RAII guard that cleans up a request slot when the handler future completes or
/// is dropped (e.g. on client disconnect). On drop it signals cooperative
/// cancellation to the blocking thread and removes the in-flight entry, ensuring
/// neither the semaphore slot nor the dashmap entry leaks even if axum cancels
/// the handler mid-flight.
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

pub(super) struct RequestGuard {
    request_id: u64,
    state: Arc<AppState>,
    cancellation: Arc<AtomicBool>,
    /// Held here so the slot and core are released when the guard drops.
    _permit: AnalysisPermit,
}

impl RequestGuard {
    fn new(
        request_id: u64,
        state: Arc<AppState>,
        cancellation: Arc<AtomicBool>,
        permit: AnalysisPermit,
    ) -> Self {
        state.jobs_started.fetch_add(1, Ordering::Relaxed);
        if let Some(pause) = &state.idle_pause {
            pause.store(true, Ordering::Release);
        }
        Self {
            request_id,
            state,
            cancellation,
            _permit: permit,
        }
    }
}

impl RequestGuard {
    /// Keep the slot and core until a timed-out blocking task returns.
    ///
    /// A blocking thread cannot be stopped, only asked: the cancellation flag,
    /// which cleave polls between members. Until it answers it is still using
    /// a core, so handing its permits back at the timeout is how a node
    /// reports capacity it does not have — 14 such orphans beside
    /// `slots_free=48` on one box. The guard follows the thread out instead,
    /// and `stuck_orphans` counts threads still running, not timeouts there
    /// have ever been.
    pub(super) fn follow<T: Send + 'static>(self, task: tokio::task::JoinHandle<T>) {
        self.cancellation.store(true, Ordering::Release);
        self.state.stuck_orphans.fetch_add(1, Ordering::Relaxed);
        tokio::spawn(async move {
            let _ = task.await;
            self.state.stuck_orphans.fetch_sub(1, Ordering::Relaxed);
            drop(self);
        });
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        // Signal the blocking thread to stop cooperatively, then remove the
        // in-flight entry. The permit is released automatically via _permit.
        self.cancellation.store(true, Ordering::Release);
        self.state.in_flight.remove(&self.request_id);
        // Resume the idle worker once the last interactive request is done.
        // Checked after the removal so a concurrent arrival cannot be missed:
        // that request raised the flag before its guard existed.
        if let Some(pause) = &self.state.idle_pause
            && self.state.in_flight.is_empty()
        {
            pause.store(false, Ordering::Release);
        }
    }
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

    /// The windowed view, in milliseconds, for `/_/stats`.
    pub(crate) fn recent_json(&self) -> serde_json::Value {
        let s = self.recent.summary();
        serde_json::json!({
            "samples": s.samples,
            "p80_ms": s.p80_micros.map(|us| us / 1_000),
            "mean_ms": s.mean_micros.map(|us| us / 1_000),
        })
    }
}

/// How much recent history the windowed estimates cover, for `/_/stats`. A
/// consumer that knows the window can tell "quiet worker" from "stale reading".
pub(crate) fn latency_window_secs() -> u64 {
    latency::WINDOW.as_secs()
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

#[derive(Debug)]
/// The loaded model bundle an analysis runs against: thresholds, the ML
/// ensemble, and the optional LLM and fetch policies attached to it.
///
/// Public because [`crate::worker::Embedded`] carries one — an idle worker
/// running inside a serve process shares the server's already-loaded models
/// rather than loading a second copy of the largest thing in the process.
pub struct ModelResources {
    pub(crate) model: Model,
    pub(crate) shap: Option<ShapImportance>,
    pub(crate) ctx: ExtractContext,
    /// LLM interpretation config (`--interpret`); `None` disables the pass.
    pub(crate) interpret: Option<crate::interpret::InterpretConfig>,
    /// External-reference fetch policy; default (empty) disables fetching.
    pub(crate) fetch: crate::fetch::FetchPolicy,
    /// Additional passwords to try for encrypted archives.
    pub(crate) zip_passwords: crate::ArchivePasswords,
}

/// Payloads at or below this many bytes count as small — for the slot lanes
/// and for [`small_pool_for`] alike. `SCAN_SMALL_JOB_MB`, the same knob the
/// worker's cleave gate reads; 1 MiB unless set.
fn small_job_max_bytes() -> u64 {
    std::env::var("SCAN_SMALL_JOB_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(1024 * 1024, |mb| mb.saturating_mul(1024 * 1024))
}

/// Threads for the small-analysis pool on a host with this many physical
/// cores: an eighth of them, never fewer than two (one thread cannot pipeline
/// a member window against its own producer) and never more than sixteen.
/// 4 cores → 2, 32 → 4, 64 → 8, 128 → 16.
#[must_use]
pub(crate) fn small_pool_threads(physical_cores: usize) -> usize {
    (physical_cores / 8).clamp(2, 16)
}

struct SmallPool {
    pool: rayon::ThreadPool,
    max_bytes: u64,
}

/// A second rayon pool for small analyses: the bulkhead that keeps a
/// 300-byte package from queueing behind a 40 MB one.
///
/// Every server analysis runs on a tokio blocking thread, so each inner
/// `par_iter` it issues — string extraction in stng, a member-window flush in
/// cleave — is *injected* into the global pool from outside, and rayon
/// workers take injected work only once their own deques are empty. While a
/// whale's thousands of member tasks are queued they never are. Measured
/// 2026-09-05 at concurrency 8 over 128 real PURLs: a package that analyzes
/// in 0.3s alone waited 16.8s in `Registry::in_worker_cold` for a worker, and
/// the p90 sat at 5.2–5.8s whether or not the LLM pass ran at all. Inside a
/// pool of its own, a small analysis's joins run on that pool's workers
/// through their local deques and never touch the injector.
///
/// Sized by [`small_pool_threads`] on the same stacks as the global pool. It
/// oversubscribes the CPU by that many threads while a whale saturates the
/// global pool — which is the point: the small work must not wait for it.
/// `SCAN_SMALL_POOL_THREADS` overrides the size (0 disables the pool);
/// `SCAN_SMALL_JOB_MB` says what counts as small. Built on first use.
static SMALL_POOL: OnceLock<Option<SmallPool>> = OnceLock::new();

fn build_small_pool() -> Option<SmallPool> {
    let threads = match std::env::var("SCAN_SMALL_POOL_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(0) => {
            tracing::info!("small analysis pool disabled (SCAN_SMALL_POOL_THREADS=0)");
            return None;
        }
        Some(n) => n,
        None => small_pool_threads(
            cleave::memory_tracker::physical_cpu_count()
                .or_else(|| {
                    std::thread::available_parallelism()
                        .ok()
                        .map(|n| n.get() / 2)
                })
                .unwrap_or(4),
        ),
    };
    let max_bytes = small_job_max_bytes();
    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(crate::RAYON_STACK_MB * 1024 * 1024)
        .thread_name(|i| format!("rayon-small-{i}"))
        .build()
    {
        Ok(pool) => {
            tracing::info!(
                threads,
                small_max_mb = max_bytes / (1024 * 1024),
                "small analysis pool ready: payloads at or below the cap run on their own rayon pool"
            );
            Some(SmallPool { pool, max_bytes })
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to build the small analysis pool; small payloads share the global pool");
            None
        }
    }
}

/// The pool a payload of `bytes` should analyze on, or `None` for the global
/// pool (a whale, or the small pool disabled).
pub(crate) fn small_pool_for(bytes: u64) -> Option<&'static rayon::ThreadPool> {
    SMALL_POOL
        .get_or_init(build_small_pool)
        .as_ref()
        .filter(|p| bytes <= p.max_bytes)
        .map(|p| &p.pool)
}

#[derive(Debug)]
/// See `AppState::lanes`.
pub(super) struct SlotLanes {
    pub(super) whale: Arc<tokio::sync::Semaphore>,
    pub(super) small: Arc<tokio::sync::Semaphore>,
    /// Jobs at or below this size take the small lane (`SCAN_SMALL_JOB_MB`,
    /// the same knob the worker's cleave gate reads). Unknown size — a PURL
    /// or URL analysis whose payload has not been fetched yet — is a whale:
    /// those are almost always packages, and mis-classing a whale as small
    /// is the expensive direction.
    pub(super) small_max_bytes: u64,
}

impl SlotLanes {
    fn from_env(max_concurrent: usize) -> Option<Self> {
        if std::env::var("SCAN_SLOT_LANES").as_deref() != Ok("1") {
            return None;
        }
        let whale_permits = (1 + max_concurrent / 8).min(max_concurrent);
        let small_permits = max_concurrent.saturating_sub(whale_permits).max(1);
        let small_max_bytes = small_job_max_bytes();
        tracing::info!(
            whale_permits,
            small_permits,
            small_max_mb = small_max_bytes / (1024 * 1024),
            "slot lanes enabled: class-aware admission (SCAN_SLOT_LANES)"
        );
        Some(Self {
            whale: Arc::new(tokio::sync::Semaphore::new(whale_permits)),
            small: Arc::new(tokio::sync::Semaphore::new(small_permits)),
            small_max_bytes,
        })
    }

    pub(super) fn available(&self) -> usize {
        self.whale.available_permits() + self.small.available_permits()
    }
}

struct AppState {
    max_upload_bytes: usize,
    /// Maximum RSS before rejecting requests; `None` disables throttling.
    max_rss_bytes: Option<NonZeroU64>,
    model_dir: PathBuf,
    threshold_overrides: Option<Thresholds>,
    slow_rule_ms: u64,
    level: Option<u16>,
    allowed_dirs: Vec<PathBuf>,
    extract_dir: Option<PathBuf>,
    allow_cidrs: Vec<Cidr>,
    /// Digest of the bearer token required by the ACL middleware; `None`
    /// disables authentication.
    auth_digest: Option<TokenDigest>,
    /// LLM interpretation config (`--interpret`); shared into every
    /// [`ModelResources`] so handlers can run the pass.
    interpret: Option<crate::interpret::InterpretConfig>,
    /// External-reference fetch policy; shared into every [`ModelResources`].
    fetch: crate::fetch::FetchPolicy,
    /// Additional passwords to try for encrypted archives.
    zip_passwords: crate::ArchivePasswords,
    /// Process uptime anchor — captured when build_app runs, very close to
    /// process start. /_/health reports `now - started_at` as uptime_secs.
    started_at: Instant,
    ready: AtomicBool,
    init_error: RwLock<Option<String>>,
    resources: RwLock<Option<Arc<ModelResources>>>,
    next_request_id: AtomicU64,
    /// Semaphore with max_concurrent_tasks permits. Each analysis handler acquires
    /// one OwnedSemaphorePermit before starting work; the permit is dropped when
    /// the analysis completes or when the orphan-cleanup task gives up. RAII
    /// semantics mean the slot is always released — even on panic or runtime shutdown.
    slots: Arc<tokio::sync::Semaphore>,
    /// One permit per rayon thread, shared with the idle worker: every analysis
    /// in this process, whoever asked for it, runs on the same pool.
    cpu: Arc<tokio::sync::Semaphore>,
    /// Class-aware admission (`SCAN_SLOT_LANES=1`): the flat `slots` semaphore
    /// treats every analysis as equal, but a large archive fans out across the
    /// whole shared rayon pool while a small file uses roughly one thread — so
    /// `--workers` flat slots either under-admit smalls or co-schedule whales
    /// that then fight for the pool (measured +55% wall on whale co-residency).
    /// The lanes mirror the worker's cleave gate at the front door: smalls
    /// (`< small_max_bytes`, the worker's 1 MiB small-job line) get most
    /// permits, whales get few, and a full lane answers 429 + Retry-After
    /// instead of queueing — a whale's queue wait is minutes, so the fleet
    /// routes it to an idle server; a small's wait is seconds, so callers just
    /// retry. `None` = lanes disabled, flat admission as before.
    lanes: Option<SlotLanes>,
    /// Tasks stuck past the grace period — still occupying a slot until the
    /// blocking thread finally returns. Tracked for observability only.
    stuck_orphans: AtomicUsize,
    /// Capacity of the slots semaphore. Requests are rejected with 503 when no
    /// permits are available, preventing orphaned blocking tasks from piling up
    /// and consuming unbounded memory.
    max_concurrent_tasks: usize,
    /// Per-request analysis timeout. `0` disables the timeout entirely.
    analysis_timeout_secs: u64,
    reload_lock: tokio::sync::Mutex<()>,
    overloaded_since: std::sync::Mutex<Option<Instant>>,
    in_flight: dashmap::DashMap<u64, InFlightRequest>,
    /// Hopper root, kept so the idle worker can claim from the same instance
    /// the uploader renews to.
    hopper: Option<String>,
    /// Analysis slots the idle worker may use; the rest are the interactive
    /// reserve. Zero disables it.
    idle_worker_slots: usize,
    /// Cores the idle worker may occupy at once — its own budget, below the
    /// server's, so the server can always start an analysis. See
    /// [`idle_worker_cores`].
    idle_worker_cores: usize,
    /// Pull-queue analyses the idle worker has in progress, tails included.
    idle_in_progress: Arc<AtomicUsize>,
    /// The idle worker's core budget as a semaphore; `idle_worker_cores` less
    /// its free permits is the cores pull work holds right now, published on
    /// `/_/stats` as `background_in_flight`.
    idle_cpu: Arc<tokio::sync::Semaphore>,
    /// Machine-wide cores busy between consecutive `/_/stats` reads.
    cpu_busy: CpuBusy,
    /// Raised once the HTTP server stops, so the idle worker winds down with it
    /// rather than outliving the thing it exists to fill the gaps of.
    shutdown: Arc<AtomicBool>,
    /// Per-size-bucket completion totals, for the size-aware half of routing.
    ///
    /// One scalar average is not enough to choose a server. The 12.5s-vs-90s
    /// spread measured across two scanners on the same artifact was a large
    /// archive's member analysis, not a constant handicap — a single number
    /// would brand a box "slow" when it is only slow at big inputs, and send
    /// every small package somewhere worse. A caller usually knows the size
    /// before it dispatches, so the useful answer is per bucket.
    job_buckets: [JobBucket; SIZE_BUCKETS.len()],
    job_types: [JobBucket; PURL_TYPE_NAMES.len()],
    /// The same per-size figures for work the idle worker did, kept apart
    /// from the request ones because they are not the same measurement.
    ///
    /// Request timings say how fast this server was on whatever a router chose
    /// to send it, which makes them useless for deciding whether that router
    /// chose well: a server nobody dispatches to reports nothing and stays
    /// unroutable, and one sent only small work looks fast at everything. Idle
    /// work is claimed from the same hopper queue by every server and is not
    /// selected by anybody's routing, so these are comparable across a fleet in
    /// the way request timings are not.
    ///
    /// Kept separate rather than merged: the idle worker stands down while
    /// interactive requests are in flight, so this is uncontended speed while
    /// `job_buckets` is speed under whatever load the server was carrying. Both
    /// are worth having, and averaging them together would describe neither.
    /// An `Arc` because the reporter outlives this borrow and must not hold the
    /// state that holds the reporter.
    idle_job_buckets: Arc<[JobBucket; SIZE_BUCKETS.len()]>,
    /// The blended average, aged like the others. Separate from
    /// `jobs_completed`, which stays a true lifetime count for reporting: one
    /// answers "how fast is this server now", the other "how much has it done".
    ///
    /// Fresh analyses only — see [`AppState::job_cached`]. So are
    /// `job_buckets` and `job_types`.
    job_overall: JobBucket,
    /// Analyses answered from this server's own verdict index.
    ///
    /// Kept apart from the fresh numbers because mixing them makes every
    /// average bimodal and therefore useless for prediction: the same artifact
    /// is milliseconds on a hit and minutes on a miss. A router choosing a
    /// worker for work it has not done wants the fresh figure; blending in
    /// cache hits only tells it how lucky this server has been.
    job_cached: JobBucket,
    /// `/lookup` service time. Near-constant — an index probe, not an analysis
    /// — and so the honest input for ordering the cheap-source race, where the
    /// analysis averages would be wrong by three orders of magnitude.
    lookups: JobBucket,
    /// Analyses this server has begun, completed, and the totals behind their
    /// averages.
    ///
    /// Counted rather than sampled: a router wants "how big and how slow are
    /// this server's jobs, typically", and totals divided at read time answer
    /// that without keeping a window. `started` minus `completed` is also the
    /// honest count of work that went in and never came out.
    jobs_started: AtomicU64,
    jobs_completed: AtomicU64,
    job_bytes_total: AtomicU64,
    job_micros_total: AtomicU64,
    /// Set once the idle worker has actually been spawned. Published on
    /// `/_/info`: "configured" and "running" are different states, and the gap
    /// between them is exactly where a silent early return hides.
    idle_worker_started: AtomicBool,
    /// Raised while any interactive request is in flight, so an embedded idle
    /// worker stops claiming queue work. `None` when no idle worker is running.
    ///
    /// Driven from [`RequestGuard`] rather than polled: the guard already
    /// brackets exactly the window that matters, and a poller would either lag
    /// a request's arrival or spin.
    idle_pause: Option<Arc<AtomicBool>>,
    /// Monotonic elapsed-time marker for the most recent analysis request.
    /// Unlike `idle_pause`, this also covers requests that are rejected before
    /// they acquire an analysis slot.
    last_analyze_request_ms: Arc<AtomicU64>,
    /// Analyses in progress, so concurrent requests for the same artifact
    /// share one run instead of each taking a slot. See [`flight`].
    flights: Arc<flight::Flights>,
    /// Background hopper uploader (`--hopper`); `None` disables result renewal.
    /// Shared across handlers; each analyzed result is queued to its own thread,
    /// so uploads never block the analyze response.
    uploader: Option<Arc<crate::upload::Uploader>>,
    /// The corpus behind this worker's index. `None` when no hopper is
    /// configured, which leaves a lookup answering from local knowledge alone.
    corpus: Option<Arc<corpus::Corpus>>,
}

impl AppState {
    /// Cores the idle worker holds at this moment: its budget less the permits
    /// it has not taken. Bounded by the budget, unlike the in-progress count,
    /// which includes tails waiting on the network and on rdu2 stood at 576
    /// against 96 cores (2026-09-05) — a discount that size told the router a
    /// fully busy box had nothing on it.
    pub(super) fn idle_cores_held(&self) -> usize {
        self.idle_worker_cores
            .saturating_sub(self.idle_cpu.available_permits())
    }

    /// Analyses this server can start right now: a slot and a core for each.
    pub(super) fn available_analysis_permits(&self) -> usize {
        let slots = match &self.lanes {
            Some(lanes) => lanes.available(),
            None => self.slots.available_permits(),
        };
        slots.min(self.cpu.available_permits())
    }

    fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Record activity from one of the analysis endpoints. The marker is
    /// elapsed milliseconds plus one so zero can mean "no request yet".
    pub(super) fn note_analyze_request(&self) {
        let elapsed_ms = self.started_at.elapsed().as_millis();
        let marker = u64::try_from(elapsed_ms)
            .unwrap_or(u64::MAX.saturating_sub(1))
            .saturating_add(1);
        self.last_analyze_request_ms
            .store(marker, Ordering::Release);
    }
}

/// Build the axum [`Router`] and start background resource loading.
///
/// The server is bound and begins accepting connections immediately.  Until
/// model resources finish loading the health endpoint returns 503 and the
/// analyze endpoint returns 503.  Resources load concurrently in a background
/// task; YARA is warmed up in a separate fire-and-forget task so it does not
/// delay readiness.
///
/// Useful for integration tests that need the app without binding to a port.
///
/// # Errors
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
    last: std::sync::Mutex<Option<(Instant, cleave::memory_tracker::CpuTime, Option<f64>)>>,
}

impl CpuBusy {
    /// Logical cores busy since the previous call, or the previous answer if
    /// the counters have not moved, or `None` with nothing to compare yet.
    pub(super) fn sample(&self) -> Option<f64> {
        let now = cleave::memory_tracker::cpu_time()?;
        let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        let mut last = self.last.lock().ok()?;
        let busy = match *last {
            Some((_, prev, previous)) => cores_busy(prev, now, cpus).or(previous),
            None => None,
        };
        *last = Some((Instant::now(), now, busy));
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

/// Cores the embedded idle worker may occupy: half the pool, and never none.
///
/// Half is the stated intent for background work, and this is the number
/// that delivers it: a core permit is held for one blocking classify, and
/// measured on rdu2 (2026-09-05) 96 permits kept 101 of 128 cores busy, so
/// permits track cores closely. The old cap was on slots, which stopped
/// meaning cores when slots were sized at three per core.
///
/// The server's interactive analyses and the idle worker's pull jobs feed one
/// rayon pool. The idle worker must not be able to fill it, or the server's
/// `slots_free` reads zero and the router stops sending — and the worker only
/// stands aside for requests that arrive. Half held back keeps the server
/// able to start on a saturated box; rayon absorbs the brief oversubscription
/// while the worker pauses (`IDLE_WORKER_QUIET_SECS`), which it does within a
/// tick of the first request. `.max(1)` yields one core on a one-core box
/// rather than zero, which would deadlock the worker's cleave gate.
fn idle_worker_cores(cores: usize) -> usize {
    (cores / 2).max(1)
}

/// Returns an error if the router cannot be assembled or background resource
/// initialization cannot be scheduled.
pub async fn build_app(config: &ServerConfig) -> anyhow::Result<Router> {
    tracing::info!(model_dir = %config.model_dir().display(), "starting — resources loading in background");

    // Concurrency limit comes from --workers (defaults to cores/2 in main.rs).
    // CPU-bound cleave + ONNX work overlaps poorly across many threads, so a
    // smaller pool typically delivers higher aggregate throughput than 1/core.
    let max_concurrent = config.workers();
    let cores = crate::worker::cleave_concurrency(max_concurrent);
    tracing::info!(
        max_concurrent,
        cores,
        idle_worker_cores = idle_worker_cores(cores),
        "concurrency limit set"
    );

    let state = Arc::new(AppState {
        max_upload_bytes: config.max_body_size(),
        max_rss_bytes: config.max_rss_bytes(),
        model_dir: config.model_dir().to_path_buf(),
        threshold_overrides: config.thresholds(),
        slow_rule_ms: config.slow_rule_ms(),
        level: config.level(),
        allowed_dirs: config.allowed_dirs().to_vec(),
        extract_dir: config.extract_dir().map(PathBuf::from),
        allow_cidrs: config.allow_cidrs().to_vec(),
        auth_digest: config.auth_digest(),
        interpret: config.interpret().cloned(),
        fetch: config.fetch(),
        zip_passwords: config.zip_passwords.clone(),
        started_at: Instant::now(),
        ready: AtomicBool::new(false),
        init_error: RwLock::new(None),
        resources: RwLock::new(None),
        next_request_id: AtomicU64::new(1),
        slots: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
        lanes: SlotLanes::from_env(max_concurrent),
        cpu: Arc::new(tokio::sync::Semaphore::new(cores)),
        idle_worker_cores: idle_worker_cores(cores),
        idle_in_progress: Arc::new(AtomicUsize::new(0)),
        idle_cpu: Arc::new(tokio::sync::Semaphore::new(idle_worker_cores(cores))),
        cpu_busy: CpuBusy::default(),
        stuck_orphans: AtomicUsize::new(0),
        max_concurrent_tasks: max_concurrent,
        analysis_timeout_secs: config.analysis_timeout_secs(),
        reload_lock: tokio::sync::Mutex::new(()),
        overloaded_since: std::sync::Mutex::new(None),
        flights: Arc::new(flight::Flights::default()),
        in_flight: dashmap::DashMap::new(),
        hopper: config.hopper().map(str::to_owned),
        idle_worker_slots: config.idle_worker_slots(),
        shutdown: Arc::new(AtomicBool::new(false)),
        idle_worker_started: AtomicBool::new(false),
        last_analyze_request_ms: Arc::new(AtomicU64::new(0)),
        job_buckets: Default::default(),
        job_types: Default::default(),
        idle_job_buckets: Arc::new(Default::default()),
        job_overall: Default::default(),
        job_cached: Default::default(),
        lookups: Default::default(),
        jobs_started: AtomicU64::new(0),
        jobs_completed: AtomicU64::new(0),
        job_bytes_total: AtomicU64::new(0),
        job_micros_total: AtomicU64::new(0),
        // Decided here because AppState lives behind an Arc and cannot be
        // amended later. The worker itself starts once the models are loaded.
        idle_pause: (config.idle_worker_slots() > 0 && config.hopper().is_some())
            .then(|| Arc::new(AtomicBool::new(false))),
        // Start the background uploader once when --hopper is set, so every
        // analyzed result (parent and members) is renewed on hopper without
        // blocking the analyze response. Said once here rather than on every
        // analysis: a server nobody configured a hopper for still answers, but
        // every verdict it computes dies with the process, and that is worth
        // one line at startup instead of silence.
        uploader: match config.hopper() {
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
        },
        corpus: {
            let corpus = corpus::Corpus::new(config.hopper());
            match &corpus {
                Some(c) => {
                    tracing::info!(addresses = %c.addresses(), "lookups defer to the corpus")
                }
                // Not a warning: a worker with no corpus behind it answers from
                // its own index, which is a whole deployment rather than a
                // broken one.
                None => tracing::info!(
                    "no hopper configured: lookups answer from the local index alone"
                ),
            }
            corpus
        },
    });

    // Background task: load model + SHAP + YARA concurrently, then mark ready.
    {
        // The idle worker fills the gaps around this server, so it winds down
        // with it. Awaiting the signal alongside axum's own graceful shutdown
        // is safe — signal streams deliver to every listener.
        {
            let stopping = Arc::clone(&state);
            tokio::spawn(async move {
                shutdown_signal().await;
                stopping.shutdown.store(true, Ordering::Release);
            });
        }

        let bg = Arc::clone(&state);
        let model_dir = config.model_dir().to_path_buf();
        let model_dir_shap = config.model_dir().to_path_buf();
        let thresholds = config.thresholds();
        let level = config.level();
        let slow_rule_ms = config.slow_rule_ms();
        tokio::spawn(async move {
            let init_start = Instant::now();
            tracing::info!("resource loader started (model + SHAP + YARA loading concurrently)");

            // Capture spawn times in the async context so each blocking closure
            // can report queue_ms (time waiting for a thread) separately from
            // work_ms (time actually doing I/O and parsing).
            let model_spawned_at = Instant::now();
            let model_task =
                tokio::task::spawn_blocking(move || -> anyhow::Result<(Model, ExtractContext)> {
                    let queue_ms = model_spawned_at.elapsed().as_millis();
                    let t = Instant::now();
                    tracing::info!(queue_ms, "loading ONNX model and feature spec");
                    let model = Model::load(&model_dir, thresholds, level)?;
                    let ctx = ExtractContext::new(model.spec());
                    tracing::info!(
                        queue_ms,
                        work_ms = t.elapsed().as_millis(),
                        spec_version = model.spec().version(),
                        features = model.spec().total_features(),
                        "ONNX model loaded",
                    );
                    Ok((model, ctx))
                });
            let shap_spawned_at = Instant::now();
            let shap_task = tokio::task::spawn_blocking(move || {
                let queue_ms = shap_spawned_at.elapsed().as_millis();
                let t = Instant::now();
                tracing::info!(queue_ms, "loading SHAP importance data");
                match ShapImportance::load(&model_dir_shap) {
                    Ok(shap) => {
                        tracing::info!(
                            queue_ms,
                            work_ms = t.elapsed().as_millis(),
                            "SHAP data loaded"
                        );
                        Some(shap)
                    }
                    Err(e) => {
                        tracing::warn!(
                            queue_ms,
                            work_ms = t.elapsed().as_millis(),
                            "SHAP data unavailable (explanations disabled): {e:#}"
                        );
                        None
                    }
                }
            });
            let yara_spawned_at = Instant::now();
            let yara_task = tokio::task::spawn_blocking(move || -> Result<(), String> {
                let queue_ms = yara_spawned_at.elapsed().as_millis();
                let t = Instant::now();
                tracing::info!(queue_ms, "YARA warmup started");
                // The traits tree is the rule set every analysis runs against.
                // Resolve it before reporting ready: a server that answers
                // `/_/health` with "ok" while failing every analysis on a
                // missing traits directory is worse than one that never starts.
                let traits = cleave::traits_repo::try_resolve()?;
                tracing::info!(dir = %traits.display(), "cleave traits resolved");
                let opts = cleave::AnalysisOptions {
                    slow_rule_ms,
                    ..Default::default()
                };
                let _ = cleave::analyze_file(std::path::Path::new("/dev/null"), &opts);
                tracing::info!(
                    queue_ms,
                    work_ms = t.elapsed().as_millis(),
                    "YARA warmup complete",
                );
                Ok(())
            });

            match tokio::join!(model_task, shap_task, yara_task) {
                (Ok(Ok((model, ctx))), Ok(shap), Ok(Ok(()))) => {
                    let spec_version = model.spec().version();
                    let features = model.spec().total_features();
                    let shap_loaded = shap.is_some();
                    tracing::info!("all resources ready, installing into AppState");
                    match bg.resources.write() {
                        Ok(mut lock) => {
                            let loaded = Arc::new(ModelResources {
                                model,
                                shap,
                                ctx,
                                interpret: bg.interpret.clone(),
                                fetch: bg.fetch,
                                zip_passwords: bg.zip_passwords.clone(),
                            });
                            *lock = Some(Arc::clone(&loaded));
                            if let Ok(mut init_error) = bg.init_error.write() {
                                *init_error = None;
                            }
                            bg.ready.store(true, Ordering::Release);
                            tracing::info!(
                                total_ms = init_start.elapsed().as_millis(),
                                spec_version,
                                features,
                                shap_loaded,
                                "server ready",
                            );
                            // Idle capacity is otherwise wasted. Started here
                            // rather than at bind time because it needs the
                            // loaded models — the same ones, not a second copy.
                            //
                            // The Arc is handed over rather than read back out
                            // of `bg.resources`: this scope still holds the
                            // write guard, and taking a read lock under it is a
                            // self-deadlock that would wedge the server the
                            // moment an idle worker was actually configured.
                            drop(lock);
                            spawn_idle_worker(&bg, &loaded);
                        }
                        Err(e) => tracing::error!("resources lock poisoned during init: {e}"),
                    }
                }
                (Ok(Err(e)), _, _) => {
                    record_init_failure(&bg, &format!("failed to load model: {e:#}"))
                }
                (Err(e), _, _) => {
                    record_init_failure(&bg, &format!("model load task panicked: {e}"))
                }
                (_, Err(e), _) => {
                    record_init_failure(&bg, &format!("shap load task panicked: {e}"))
                }
                (_, _, Ok(Err(e))) => record_init_failure(&bg, &format!("traits unavailable: {e}")),
                (_, _, Err(e)) => {
                    record_init_failure(&bg, &format!("yara warmup task panicked: {e}"))
                }
            }
        });
    }

    // Watchdog: periodically log about stuck in-flight requests. Signals cooperative
    // cancellation to tasks running past the cancel threshold so cleave can bail out
    // of slow YARA rules, but never terminates the process — that is left to the
    // operator. The threshold follows the configured analysis timeout: at least the
    // historical 10 minutes, and always past `--analysis-timeout` itself (the request
    // has already 504'd by then; this reaps the orphaned blocking thread). A timeout
    // of 0 is an explicit operator opt-out of time limits, so the watchdog only logs.
    {
        let watchdog = Arc::clone(&state);
        let cancel_after_secs = match config.analysis_timeout_secs() {
            0 => None,
            t => Some(t.max(600)),
        };
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let available = watchdog.available_analysis_permits();
                let active = watchdog.max_concurrent_tasks.saturating_sub(available);
                let stuck = watchdog.stuck_orphans.load(Ordering::Relaxed);

                if active == 0 {
                    continue;
                }

                // Log details for every long-running in-flight request.
                let now = Instant::now();
                for entry in watchdog.in_flight.iter() {
                    let elapsed_secs = now.duration_since(entry.started_at).as_secs();
                    let phase = entry.phase.get();
                    let tid = entry.thread_id.load(Ordering::Relaxed);
                    if cancel_after_secs.is_some_and(|t| elapsed_secs >= t) {
                        // Signal cooperative cancellation for very long tasks so
                        // cleave can exit slow YARA rules cleanly.
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

    // No ConcurrencyLimitLayer — the hard gate (active_tasks >= max_concurrent_tasks)
    // in each handler rejects immediately with 503. No silent queuing.
    // Hopper controls send rate via litmus-workers; litmus accepts or rejects.
    // Middleware order: layers are applied bottom-up, so the last `.layer()`
    // call wraps everything else and runs first per request. ACL runs before
    // the body limit so rejected peers don't get to upload bytes.
    let app = Router::new()
        .route("/_/health", get(handlers::health))
        .route("/_/info", get(handlers::info))
        .route("/_/stats", get(handlers::stats))
        .route("/_/reload", post(handlers::reload))
        .route("/_/update", post(handlers::update))
        .route("/_/memory", get(handlers::memory_stats))
        .route("/_/requests", get(handlers::requests))
        .route("/_/threads", get(handlers::threads))
        .route("/lookup", get(handlers::lookup))
        .route("/status", get(handlers::status))
        .route("/v1/lookup", get(handlers::v1_lookup))
        .route("/v1/analyze", post(handlers::v1_analyze))
        .route("/analyze", post(handlers::analyze))
        .route("/analyze-purl", post(handlers::analyze_purl))
        .route("/analyze-path", post(handlers::analyze_path))
        .layer(DefaultBodyLimit::max(config.max_body_size()))
        .layer(middleware::from_fn_with_state(Arc::clone(&state), acl::acl))
        // Outermost: every request gets an id and an access-log line, including
        // the ones the ACL rejects.
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            access::access_log,
        ))
        .with_state(state);

    Ok(app)
}

fn record_init_failure(state: &AppState, message: &str) {
    state.ready.store(false, Ordering::Release);
    if let Ok(mut init_error) = state.init_error.write() {
        *init_error = Some(message.to_string());
    }
    tracing::error!("{message}");
}

/// Start the HTTP server and block until shutdown.
///
/// This binds the configured socket address, starts background resource
/// loading, and serves requests until `SIGINT` or `SIGTERM`.
///
/// # Errors
/// Returns an error if the listening socket cannot be bound or the server
/// fails while serving requests.
pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    // Warm cleave's YARA engine + capability mapper off the rayon pool before
    // the listener binds. The first request's analysis spawns rayon work; if
    // one of those rayon workers is the first to hit `yara_engine()`, init's
    // internal par_iter deadlocks against its peers parked on the OnceLock.
    // Prefetching from a non-rayon thread here avoids the race entirely —
    // `prefetch_shared_resources` returns immediately and does the work in a
    // `std::thread::spawn`, so it doesn't delay startup.
    cleave::prefetch_shared_resources(true);

    // Server mode processes many files over a long lifetime. Configure jemalloc
    // to aggressively return freed pages to the OS, preventing multi-GB RSS
    // growth from allocator fragmentation across thousands of analyses.
    cleave::memory_tracker::configure_jemalloc_low_memory();

    // Watchdog thread: enforces the same RSS limit as check_memory_pressure on
    // wall-clock time, independent of request traffic. This catches memory
    // growth that happens between requests (e.g. jemalloc fragmentation or
    // background YARA work). Skipped when throttling is disabled.
    let _watchdog = config.max_rss_bytes().map(|limit| {
        cleave::memory_tracker::start_periodic_logging(
            std::time::Duration::from_secs(10),
            limit.get(),
        )
    });

    let app = build_app(&config).await?;

    let listener = tokio::net::TcpListener::bind(config.bind()).await?;
    eprintln!(
        "Listening on http://{} (max size: {} MB, starting up) — Press Ctrl+C to stop",
        config.bind(),
        config.max_body_size() / 1024 / 1024,
    );
    // The startup line is the record of what this process actually is: an
    // operator reading the log after a restart should not have to reconstruct
    // the running configuration from the unit file.
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        pid = std::process::id(),
        bind = %config.bind(),
        max_body_mb = config.max_body_size() / 1024 / 1024,
        analysis_timeout_secs = config.analysis_timeout_secs(),
        allow_cidrs = config.allow_cidrs().len(),
        allowed_dirs = config.allowed_dirs().len(),
        authenticated = config.auth_digest().is_some(),
        "listening (resources loading in background)",
    );

    // An unauthenticated API is open to anyone who can reach the socket. Warn
    // unconditionally — a loopback bind is not evidence of safety, because a
    // Cloudflare tunnel terminates on loopback and puts the whole internet on
    // the other side of it.
    if config.auth_digest().is_none() {
        tracing::warn!(
            "no --token-file: the API is unauthenticated; any peer that reaches the socket can submit work",
        );
    }

    // /analyze-path reads any file under --allowed-dirs and is restricted to
    // loopback peers — but a tunnel makes every peer a loopback peer, so that
    // restriction stops protecting it. Leave --allowed-dirs empty unless the
    // host is genuinely local-only; with no allowed directory the route
    // rejects every request.
    if !config.allowed_dirs().is_empty() {
        tracing::warn!(
            allowed_dirs = config.allowed_dirs().len(),
            "--allowed-dirs is set: /analyze-path can read those directories for any peer reaching loopback, including through a tunnel",
        );
    }

    // Operator footgun: setting --allow-cidr while bound to loopback means
    // the CIDR list can never match (no remote peers can connect). Warn so
    // the operator notices before debugging "why is everyone getting 403?".
    if !config.allow_cidrs().is_empty() && config.bind().ip().is_loopback() {
        tracing::warn!(
            bind = %config.bind(),
            "--allow-cidr is set but bind address is loopback; remote clients cannot connect (use --bind 0.0.0.0:PORT)",
        );
    }

    // ConnectInfo<SocketAddr> is required by the ACL middleware so it can
    // see the peer IP. Tests inject ConnectInfo manually on each Request.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    tracing::info!("server shut down");
    Ok(())
}

/// Start the embedded idle worker: fill unused analysis capacity with queue
/// work from hopper, and stand aside the moment a request arrives.
///
/// A scan server spends most of its life waiting. Meanwhile hopper holds a
/// backlog and the fleet's dedicated workers grind through it, so the idle
/// capacity here is pure waste — and, usefully, running the same queue work on
/// every server produces a continuous like-for-like measurement of how fast
/// each one actually is.
///
/// Interactive work always wins: [`RequestGuard`] raises the pause flag before
/// a request starts and lowers it when the last one finishes, and the worker's
/// prefetcher stops claiming while it is raised. Jobs already running are not
/// abandoned — that work is real, and a claim that dies is redispatched by
/// hopper anyway — so promptness comes from the slots held back for requests,
/// not from killing work mid-flight.
fn spawn_idle_worker(state: &Arc<AppState>, resources: &Arc<ModelResources>) {
    // Every exit says why. The first cut returned silently on three separate
    // paths, so a worker that never started was indistinguishable from one that
    // started and found nothing to do — and the only way to tell them apart was
    // to read the source.
    let Some(hopper) = state.hopper.clone() else {
        tracing::info!("idle worker disabled: no --hopper to claim work from");
        return;
    };
    // `--hopper` may name several addresses, replica first. Reads and renewals
    // take that whole list; a worker may not. Claiming work is the primary's
    // route alone, so this takes the one address rather than the string — which
    // a URL parser reads as a single very strange hostname.
    let Some(hopper) = crate::upload::worker_endpoint(&hopper) else {
        tracing::info!("idle worker disabled: --hopper names no address");
        return;
    };
    let Some(slots) = std::num::NonZeroUsize::new(state.idle_worker_slots) else {
        tracing::info!("idle worker disabled: --idle-worker-slots is 0");
        return;
    };
    let Some(pause) = state.idle_pause.clone() else {
        tracing::warn!(
            slots = slots.get(),
            "idle worker not started: no pause flag, so it could not yield to \
             requests — refusing rather than competing with them",
        );
        return;
    };
    let resources = Arc::clone(resources);

    tracing::info!(
        slots = slots.get(),
        reserved_for_requests = state.max_concurrent_tasks.saturating_sub(slots.get()),
        hopper = %hopper,
        "idle worker: filling spare capacity with hopper queue work",
    );
    state.idle_worker_started.store(true, Ordering::Release);

    let config = crate::worker::WorkerConfig {
        hopper_url: hopper,
        name: format!("{}-idle", crate::upload::default_worker_name()),
        workers: slots,
        poll_secs: 30,
        // Memory is the host's to manage: the server already bounds its own
        // concurrency, and a second RSS ceiling here would pause the worker on
        // the server's own footprint.
        max_rss_gb: 0,
        model_dir: state.model_dir.clone(),
        thresholds: None,
        data_dir: None,
        slow_rule_ms: state.slow_rule_ms,
        max_jobs: None,
        exit_if_empty: false,
        no_update: true,
        level: None,
        nice: 0,
        interpret: state.interpret.clone(),
        fetch: state.fetch,
        zip_passwords: state.zip_passwords.clone(),
        embedded: Some(crate::worker::Embedded {
            on_complete: Some({
                let buckets = Arc::clone(&state.idle_job_buckets);
                Arc::new(move |size_bytes: u64, micros: u64| {
                    buckets[size_bucket(size_bytes)].record(micros);
                })
            }),
            pause,
            shutdown: Arc::clone(&state.shutdown),
            // Its own core budget, deliberately not `state.cpu`. When the
            // idle worker drew from the server's pool it held every permit
            // whenever hopper had work, `available_analysis_permits` read
            // zero, `/_/stats` said `slots_free=0` with nothing in flight,
            // and beamline — correctly reading that as "at capacity" —
            // stopped sending. The worker yields to interactive traffic,
            // but only traffic that arrives, so the report starved the very
            // requests that would have made it true. Measured 2026-09-05:
            // three of the fleet's four servers unroutable for hours while
            // idle on the interactive path. See `idle_worker_cores`.
            cpu: Arc::clone(&state.idle_cpu),
            in_progress: Arc::clone(&state.idle_in_progress),
            resources,
            last_analyze_request_ms: Arc::clone(&state.last_analyze_request_ms),
            started_at: state.started_at,
            quiet_period: Duration::from_secs(IDLE_WORKER_QUIET_SECS),
        }),
    };
    tokio::spawn(async move {
        if let Err(e) = crate::worker::run(config).await {
            tracing::warn!(error = %e, "idle worker stopped");
        }
    });
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
        _ = ctrl_c => tracing::info!("received SIGINT"),
        _ = terminate => tracing::info!("received SIGTERM"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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

    // `recent_json` is what ships on /_/stats, and beamline indexes it by these
    // exact names. Asserting the shape here is what stops a rename from
    // silently demoting the router back to lifetime means — a failure that
    // looks like nothing at all from the outside.
    #[test]
    fn recent_json_publishes_the_keys_beamline_reads() {
        let b = JobBucket::default();
        b.record(9_000_000); // 9s
        let v = b.recent_json();
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
    fn recent_json_reports_an_untouched_bucket_as_empty_not_zero() {
        let v = JobBucket::default().recent_json();
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
        assert_eq!(b.recent_json()["samples"], 5);
    }
}

#[cfg(test)]
mod small_pool_tests {
    /// The small pool scales with the host and never outgrows what a whale
    /// leaves: an eighth of the cores, floor two, ceiling sixteen.
    #[test]
    fn small_pool_threads_scale_from_laptop_to_workstation() {
        assert_eq!(super::small_pool_threads(1), 2);
        assert_eq!(super::small_pool_threads(4), 2);
        assert_eq!(super::small_pool_threads(8), 2);
        assert_eq!(super::small_pool_threads(16), 2);
        assert_eq!(super::small_pool_threads(32), 4);
        assert_eq!(super::small_pool_threads(64), 8);
        assert_eq!(super::small_pool_threads(128), 16);
        assert_eq!(super::small_pool_threads(256), 16);
    }
}
