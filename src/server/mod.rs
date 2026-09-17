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
mod idle;
mod latency;

pub use acl::{Cidr, TokenDigest, parse_cidr_list};
pub(crate) use handlers::classify_bytes;
pub(crate) use handlers::classify_file;

use crate::memory::resolve_process_max_rss_bytes;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use std::net::SocketAddr;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
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
    /// Non-zero enables the companion pull worker; see
    /// [`ServerConfig::with_idle_worker_slots`].
    idle_worker_slots: usize,
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
    /// The `--max-rss-gb` flag as given: negative disables in-process
    /// throttling, zero takes the cgroup-aware limit, positive is a ceiling.
    pub max_rss_gb: i64,
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
    /// Traits bundle override.
    pub traits_dir: Option<PathBuf>,
    /// Force the startup refresh even when the local copy looks current
    /// (`-u`/`--update`).
    pub update: bool,
    /// Skip the startup model and traits refresh.
    pub no_update: bool,
    /// Hopper API root. Enables result renewal, corpus deferral, and the
    /// companion idle worker.
    pub hopper: Option<String>,
    /// Slots for the companion idle worker. `None` takes half the request
    /// slots; ignored without `hopper`, since there would be nothing to claim.
    pub idle_worker_slots: Option<usize>,
    /// Per-request analysis timeout in seconds. Zero disables.
    pub analysis_timeout_secs: u64,
    /// Model bundle. `None` resolves the installed one.
    pub model_dir: Option<PathBuf>,
    /// Operating point. `None` takes the bundle's own default.
    pub level: Option<u16>,
    /// Manual probability cutoffs, bypassing the level grid.
    pub thresholds: Option<Thresholds>,
    /// Per-rule time budget before cleave logs a slow rule.
    pub slow_rule_ms: u64,
    /// The LLM second opinion, when one is configured.
    pub interpret: Option<crate::interpret::InterpretConfig>,
    /// Whether to follow the references a sample declares. Off by default: a
    /// server driving outbound fetches is an SSRF-shaped exposure, so turning
    /// it on is an explicit operator decision.
    pub fetch: crate::fetch::FetchPolicy,
    /// Passwords to try against encrypted archives.
    pub zip_passwords: crate::ArchivePasswords,
}

impl Startup {
    /// Settle every default and read the bearer token.
    ///
    /// # Errors
    ///
    /// Returns an error when the model bundle cannot be resolved, a CIDR does
    /// not parse, or `token_file` is set but missing, empty or unreadable.
    /// That last one fails closed on purpose: an operator who asked for
    /// authentication must never get an open server because a file went away.
    pub fn resolve(self) -> anyhow::Result<ServerConfig> {
        use anyhow::Context as _;

        // Order is load-bearing and is why the refresh lives here rather than
        // at the call site. The override has to be applied first, or the
        // refresh installs into the default directory while `--traits-dir`
        // points at an empty one — and a server started that way comes up,
        // reports healthy, and fails every analysis.
        if let Some(dir) = self.traits_dir.as_ref() {
            cleave::traits_repo::set_override_dir(Some(dir.into()));
        }
        crate::refresh_rules_at_startup(self.update, self.no_update);

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

        let model_dir = match self.model_dir {
            Some(dir) => dir,
            None => crate::models_repo::model_dir().context("failed to resolve model directory")?,
        };
        // Manual cutoffs bypass the level grid, so no level applies then.
        let level = if self.thresholds.is_some() {
            None
        } else {
            Some(
                self.level
                    .or_else(|| crate::model::model_default_level(&model_dir))
                    .unwrap_or(crate::model::DEFAULT_SEVERITY_LEVEL),
            )
        };

        // Half the request slots for background work, capped again inside
        // `with_idle_worker_slots`. Disabled without hopper: nothing to claim.
        let idle_slots = match (self.hopper.as_deref(), self.idle_worker_slots) {
            (None, _) => 0,
            (Some(_), Some(n)) => n.min(workers / 2),
            (Some(_), None) => workers / 2,
        };

        Ok(ServerConfig::new(
            self.bind,
            self.max_size_mb.saturating_mul(1024 * 1024),
            resolve_process_max_rss_bytes(self.max_rss_gb),
            model_dir,
            self.thresholds,
            self.slow_rule_ms,
            allowed_dirs,
            self.extract_dir,
            workers,
            allow_cidrs,
        )?
        .with_level(level)
        .with_auth_token(auth_digest)
        .with_interpret(self.interpret)
        .with_fetch(self.fetch)
        .with_zip_passwords(self.zip_passwords)
        .with_hopper(self.hopper)
        .with_idle_worker_slots(idle_slots)
        .with_analysis_timeout(self.analysis_timeout_secs))
    }
}

/// Default per-request analysis timeout: 34 minutes. Covers cold cleave scans
/// of large archives — and fetch-enabled scans whose dependency analysis can
/// far outlast the sample's own — while still preventing a pathological input
/// from pinning a slot forever. Override with `--analysis-timeout` /
/// [`ServerConfig::with_analysis_timeout`].
pub const DEFAULT_ANALYSIS_TIMEOUT_SECS: u64 = 2040;

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

    /// Enable or disable the companion pull worker; `0` disables it.
    ///
    /// Retained as a count because `--idle-worker-slots` and the deploy's
    /// `IDLE=` have meant one for a long time, but the worker is its own
    /// process now and sizes itself like any standalone worker, so every
    /// non-zero value means the same thing.
    #[must_use]
    pub fn with_idle_worker_slots(mut self, slots: usize) -> Self {
        self.idle_worker_slots = slots;
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
/// A request's phase marker plus the timeline it leaves behind.
///
/// Wraps the [`cleave::PhaseTracker`] that `/_/requests` and the watchdog
/// read, and records when each phase began so the completion log can say
/// where a request's wall time went (`phases="purl:fetch=812 cleave:init=1930 …"`,
/// milliseconds per phase, repeats summed). That is what turns a latency
/// percentile into an attribution: fetch-bound, registry-bound or
/// analysis-bound, per request, without a profiler attached.
#[derive(Clone)]
pub(crate) struct RequestPhase(Arc<RequestPhaseInner>);

#[derive(Debug)]
struct RequestPhaseInner {
    tracker: cleave::PhaseTracker,
    started: Instant,
    /// `(phase, offset from `started`)` in the order the phases were entered.
    marks: Mutex<Vec<(String, Duration)>>,
}

/// More marks than this and the request is looping, not progressing; keep the
/// head so the timeline still shows how it started.
const PHASE_MARKS_MAX: usize = 64;

impl RequestPhase {
    pub(crate) fn with_label(label: impl Into<String>) -> Self {
        Self(Arc::new(RequestPhaseInner {
            tracker: cleave::PhaseTracker::with_label(label),
            started: Instant::now(),
            marks: Mutex::new(Vec::new()),
        }))
    }

    /// Enter `phase`: updates the shared tracker and stamps the timeline.
    pub(crate) fn set(&self, phase: &str) {
        self.0.tracker.set(phase);
        let at = self.0.started.elapsed();
        let mut marks = self
            .0
            .marks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if marks.len() < PHASE_MARKS_MAX && marks.last().is_none_or(|(last, _)| last != phase) {
            marks.push((phase.to_owned(), at));
        }
    }

    /// The current phase name, as the tracker reports it.
    pub(crate) fn get(&self) -> String {
        self.0.tracker.get()
    }

    /// The underlying tracker, for cleave's own phase reporting.
    pub(crate) fn tracker(&self) -> &cleave::PhaseTracker {
        &self.0.tracker
    }

    /// Milliseconds spent in each phase, first-entered order, as
    /// `name=ms name=ms …`. Time before the first mark is `pre`, so the
    /// figures sum to the request's elapsed time.
    pub(crate) fn timeline(&self) -> String {
        let now = self.0.started.elapsed();
        let marks = self
            .0
            .marks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let mut order: Vec<String> = Vec::new();
        let mut spent: std::collections::HashMap<String, u128> = std::collections::HashMap::new();
        let mut account = |name: &str, dur: Duration| {
            if !spent.contains_key(name) {
                order.push(name.to_owned());
            }
            *spent.entry(name.to_owned()).or_insert(0) += dur.as_millis();
        };
        if let Some((_, first)) = marks.first()
            && !first.is_zero()
        {
            account("pre", *first);
        }
        for (i, (name, at)) in marks.iter().enumerate() {
            let end = marks.get(i + 1).map_or(now, |(_, next)| *next);
            account(name, end.saturating_sub(*at));
        }
        order
            .iter()
            .map(|name| format!("{name}={}", spent.get(name).copied().unwrap_or(0)))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

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
    /// Keeps the idle worker frozen for the whole analysis, which outlives the
    /// handler whenever the client hangs up before it finishes.
    _busy: idle::BusyToken,
}

impl RequestGuard {
    fn new(
        request_id: u64,
        state: Arc<AppState>,
        cancellation: Arc<AtomicBool>,
        permit: AnalysisPermit,
    ) -> Self {
        state.jobs_started.fetch_add(1, Ordering::Relaxed);
        ACTIVE_REQUESTS.fetch_add(1, Ordering::AcqRel);
        let busy = state.enter_busy();
        Self {
            request_id,
            state,
            cancellation,
            _permit: permit,
            _busy: busy,
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
        ACTIVE_REQUESTS.fetch_sub(1, Ordering::AcqRel);
        // `_busy` thaws the worker here if this was the last holder. It is not
        // conditional on `in_flight` being empty: the counter already knows,
        // and it also counts the handlers that have not reached an analysis yet.
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
/// and for [`whale_lane_for`] alike. `SCAN_SMALL_JOB_MB`, the same knob the
/// worker's cleave gate reads; 1 MiB unless set.
fn small_job_max_bytes() -> u64 {
    std::env::var("SCAN_SMALL_JOB_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(1024 * 1024, |mb| mb.saturating_mul(1024 * 1024))
}

/// Threads for one whale's private pool on a host with this many physical
/// cores: a quarter of them, never fewer than two (one thread cannot pipeline
/// a member window against its own producer) and never more than sixteen.
/// 4 cores → 2, 64 → 16, 128 → 16. Several whales in flight each get their
/// own; the global pool keeps every core for everything else.
#[must_use]
pub(crate) fn whale_pool_threads(physical_cores: usize) -> usize {
    (physical_cores / 4).clamp(2, 16)
}

/// Threads for a small payload's private pool on a host with this many
/// physical cores: an eighth of them, 2–8. 4 cores → 2, 64 → 8, 128 → 8. A
/// package under the small cap has few members, so a wider pool mostly idles;
/// eight is where its p50 was best (0.29 s vs 0.44 s on 16, measured
/// 2026-09-05 at concurrency 8 over 256 PURLs).
#[must_use]
pub(crate) fn small_pool_threads(physical_cores: usize) -> usize {
    (physical_cores / 8).clamp(2, 8)
}

/// How many *big* whales analyze at once on a host with this many physical
/// cores; the rest wait their turn. An eighth of the cores, 1–8: 4 cores → 1,
/// 64 → 8, 128 → 8. With [`whale_pool_threads`] that bounds whale threads at
/// about two per physical core, oversubscribed enough to keep every core busy
/// while a whale is in a serial phase, not so much that the global pool's
/// small work loses its cores.
#[must_use]
pub(crate) fn whale_slots(physical_cores: usize) -> usize {
    (physical_cores / 8).clamp(1, 8)
}

/// Payloads above this many bytes are *big* whales, the ones that take a
/// slot from [`whale_slots`]. `SCAN_BIG_JOB_MB`; 8 MiB unless set. Between
/// `SCAN_SMALL_JOB_MB` and this a payload still gets a private pool, but
/// never waits: a 2 MiB package finishes in seconds, and making it queue
/// behind a 250 MiB one is the starvation this whole arrangement exists to
/// prevent.
fn big_job_min_bytes() -> u64 {
    std::env::var("SCAN_BIG_JOB_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(8 * 1024 * 1024, |mb| mb.saturating_mul(1024 * 1024))
}

/// Whale-lane sizing, read once from the environment.
/// Analyses admitted and not yet finished, process-wide — what a new lane
/// shares the cores with. Kept by [`RequestGuard`].
static ACTIVE_REQUESTS: AtomicUsize = AtomicUsize::new(0);

/// Threads a private lane of tier minimum `floor` gets when `active`
/// requests (this one included) are in flight on a host with `physical`
/// cores: an even share of the cores, never below the tier's floor and never
/// above the cores. Alone (`active <= 1`) the answer is `None`: the global
/// pool, every thread of it.
///
/// The tier floors (8 for small, 16 for whales) were chosen at concurrency 8
/// and are right there — 8 for small beat 16 on p50 — but they are the whole
/// story only when the box is full. At concurrency 1 the same 8-thread pool
/// left 120 threads idle: purls-128 measured p90 1.63 s on the fixed widths
/// against 1.12 s on the global pool, wall 164 → 111 s (2026-09-06). The
/// share reproduces both ends: 64 physical cores / 8 in flight = 8 (the
/// small floor), / 4 = 16, / 2 = 32, alone = everything.
#[must_use]
pub(crate) fn lane_threads(floor: usize, physical: usize, active: usize) -> Option<usize> {
    if active <= 1 {
        return None;
    }
    Some((physical / active).clamp(floor, physical.max(floor)))
}

struct WhaleConfig {
    /// Threads per whale pool; `0` sends whales to the global pool instead.
    threads: usize,
    /// Threads per small payload's pool; `0` keeps small payloads on the
    /// global pool.
    small_threads: usize,
    /// Physical cores, the numerator of the per-request share.
    physical: usize,
    /// `SCAN_LANE_SHARE=0` pins every lane at its tier floor (the widths
    /// measured at concurrency 8) instead of sharing the cores by load.
    share: bool,
    /// Big whales in flight at once.
    slots: usize,
    small_max_bytes: u64,
    big_min_bytes: u64,
}

static WHALE_CONFIG: OnceLock<WhaleConfig> = OnceLock::new();

fn whale_config() -> &'static WhaleConfig {
    WHALE_CONFIG.get_or_init(|| {
        let physical = cleave::memory_tracker::physical_cpu_count()
            .or_else(|| {
                std::thread::available_parallelism()
                    .ok()
                    .map(|n| n.get() / 2)
            })
            .unwrap_or(4);
        let env_usize = |key: &str| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
        };
        let threads =
            env_usize("SCAN_WHALE_POOL_THREADS").unwrap_or_else(|| whale_pool_threads(physical));
        let small_threads =
            env_usize("SCAN_SMALL_POOL_THREADS").unwrap_or_else(|| small_pool_threads(physical));
        let slots = env_usize("SCAN_WHALE_SLOTS")
            .unwrap_or_else(|| whale_slots(physical))
            .max(1);
        let small_max_bytes = small_job_max_bytes();
        let big_min_bytes = big_job_min_bytes().max(small_max_bytes);
        let share = std::env::var("SCAN_LANE_SHARE").as_deref() != Ok("0");
        if threads == 0 {
            tracing::info!("whale analysis pools disabled (SCAN_WHALE_POOL_THREADS=0)");
        } else {
            tracing::info!(
                threads_per_whale = threads,
                threads_per_small = small_threads,
                big_whale_slots = slots,
                whale_over_mb = small_max_bytes / (1024 * 1024),
                big_over_mb = big_min_bytes / (1024 * 1024),
                share,
                "analysis lanes ready: every payload runs on a private rayon pool sized to it"
            );
        }
        WhaleConfig {
            threads,
            small_threads,
            physical,
            share,
            slots,
            small_max_bytes,
            big_min_bytes,
        }
    })
}

/// Big whales in flight.
static WHALE_SLOTS_IN_USE: AtomicUsize = AtomicUsize::new(0);

/// One of [`whale_slots`], released on drop.
struct WhaleSlot;

/// Every big-whale slot is taken. Surfaced as 429 so the router places the
/// analysis on a worker with a free slot instead of queueing it here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WhaleSlotBusy {
    slots: usize,
}

impl std::fmt::Display for WhaleSlotBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "whale lane at capacity ({0}/{0} big analyses in flight)",
            self.slots
        )
    }
}

impl std::error::Error for WhaleSlotBusy {}

impl WhaleSlot {
    /// Take a slot now or refuse. This never waits: a request that reaches
    /// here is already streaming, so a queue is invisible to the router and
    /// the request sits behind whatever holds the slot for that whale's
    /// whole run (three wheels of 17–78 MB on 4-core scan-pdx, 2026-09-06:
    /// one held the only slot for 34 minutes and the other two timed out
    /// behind it while three other workers had room). A refusal ends the
    /// stream in milliseconds and the router tries the next worker.
    fn try_acquire(slots: usize) -> Result<Self, WhaleSlotBusy> {
        WHALE_SLOTS_IN_USE
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |in_use| {
                (in_use < slots).then_some(in_use + 1)
            })
            .map(|_| Self)
            .map_err(|_full| WhaleSlotBusy { slots })
    }

    /// Slots in use right now, for `/_/stats`.
    fn in_use() -> usize {
        WHALE_SLOTS_IN_USE.load(Ordering::Acquire)
    }
}

impl Drop for WhaleSlot {
    fn drop(&mut self) {
        WHALE_SLOTS_IN_USE.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A whale's private rayon pool for the life of one analysis: the bulkhead
/// that keeps a 40 MB package from stalling every 300-byte one that arrives
/// while it runs, and every 2 MB one from stalling behind it.
///
/// Every server analysis runs on a tokio blocking thread, so each inner
/// `par_iter` it issues — string extraction in stng, a member-window flush in
/// cleave — is *injected* into a rayon pool from outside, and rayon workers
/// take injected work only once their own deques are empty. While a whale's
/// thousands of member tasks sit in those deques they never are. Measured
/// 2026-09-05 at concurrency 8 over 128 real PURLs: a package that analyzes
/// in 0.3s alone waited 16.8s in `Registry::in_worker_cold` for a global-pool
/// worker, and the p90 sat at 5.2–6.9s whether or not the LLM pass ran.
///
/// A single *shared* whale pool was the first cut and fixed the small side
/// (p90 4.1s), but moved the starvation into the whale pool: three 36–254 MB
/// wheels in flight kept a 1.3 MB package waiting the whole 200s sweep, and
/// held each other to 170s+ where alone they take 25–40s. So each whale gets
/// its own pool ([`whale_pool_threads`] wide, dropped when the analysis
/// returns); nothing shares an injector with a whale. Big ones
/// (`SCAN_BIG_JOB_MB`) also take one of [`whale_slots`]; a burst of them is
/// refused past that count ([`WhaleSlotBusy`]) so the router spreads it over
/// the fleet instead of oversubscribing this host; the threads are cheap, the
/// cores are the budget.
///
/// `SCAN_WHALE_POOL_THREADS` sets the per-whale width (0 sends whales to the
/// global pool); `SCAN_WHALE_SLOTS` the big-whale concurrency;
/// `SCAN_SMALL_JOB_MB` (shared with the slot lanes) says where whale starts.
pub(crate) struct WhaleLane {
    pool: rayon::ThreadPool,
    /// Held for the pool's life when the payload is a big whale.
    _slot: Option<WhaleSlot>,
}

impl WhaleLane {
    /// Run `f` on this lane's pool, blocking until it returns.
    pub(crate) fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        self.pool.install(f)
    }
}

/// Big-whale slots in use and available, for `/_/stats`.
pub(crate) fn whale_slot_usage() -> (usize, usize) {
    (WhaleSlot::in_use(), whale_config().slots)
}

/// The lane a payload of `bytes` analyzes on: a private pool sized to it —
/// [`small_pool_threads`] wide at or below the small cap, [`whale_pool_threads`]
/// above it, after waiting for a slot if the payload is big — or `None` (the
/// global pool) when private pools are disabled for its size or fail to
/// build. The wait is the caller's phase to report.
///
/// Built per analysis and dropped after it: parking idle pools for reuse
/// (warm thread-local caches) measured as a wash, within noise on p50, p90
/// and throughput over 256 PURLs, so the simpler form stays.
///
/// Small payloads got private pools too once the whales had them: on the
/// global pool they compete for cleave's bounded inner-parallel owner slots
/// and half of them analyze serially at concurrency 8; on a pool of their
/// own each is parallel and exempt (`dedicated_pool`). Measured 2026-09-05
/// over 256 PURLs: throughput 203 → 230/min, mean 1.65 → 1.35 s.
///
/// Never for pull work. The idle worker's analyses already run on a pool of
/// their own — bounded, scheduled below the server's, and drawing on their
/// own parallelism slots — and a lane would undo all three: a lane pool's
/// threads are unmarked and at normal priority, and a big idle archive would
/// take one of the few whale slots and hold it for its whole slow run.
/// Measured 2026-09-06 on scan-lax: two of the two whale slots held by pull
/// work, and two interactive analyses waited in `whale:lane` until their
/// 30-minute budget ran out.
///
/// A big whale with every slot taken is refused ([`WhaleSlotBusy`]) rather
/// than queued; the handler turns that into a retry-later response.
pub(crate) fn whale_lane_for(bytes: u64) -> Result<Option<WhaleLane>, WhaleSlotBusy> {
    let cfg = whale_config();
    let floor = if bytes <= cfg.small_max_bytes {
        cfg.small_threads
    } else {
        cfg.threads
    };
    if floor == 0 {
        return Ok(None);
    }
    let threads = if cfg.share {
        // Alone on the box, the global pool is the widest lane there is.
        match lane_threads(floor, cfg.physical, ACTIVE_REQUESTS.load(Ordering::Acquire)) {
            Some(threads) => threads,
            None => return Ok(None),
        }
    } else {
        floor
    };
    let slot = if bytes > cfg.big_min_bytes {
        Some(WhaleSlot::try_acquire(cfg.slots)?)
    } else {
        None
    };
    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(crate::RAYON_STACK_MB * 1024 * 1024)
        .thread_name(move |i| format!("rayon-lane{threads}-{i}"))
        .build()
    {
        Ok(pool) => Ok(Some(WhaleLane { pool, _slot: slot })),
        Err(e) => {
            tracing::warn!(error = %e, bytes, "failed to build a private analysis pool; this payload shares the global pool");
            Ok(None)
        }
    }
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
    /// Raised while any interactive request is in flight, so an embedded idle
    /// worker stops claiming queue work. `None` when no idle worker is running.
    ///
    /// Driven from [`RequestGuard`] rather than polled: the guard already
    /// brackets exactly the window that matters, and a poller would either lag
    /// a request's arrival or spin.
    /// Requests outstanding, and the worker they freeze. See [`idle::Busy`].
    busy: Arc<idle::Busy>,
    /// The companion pull worker, when one is running.
    idle_worker: Option<Arc<idle::Worker>>,
    /// Monotonic elapsed-time marker for the most recent analysis request.
    /// Unlike `idle_pause`, this also covers requests that are rejected before
    /// they acquire an analysis slot.
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

    /// Mark the server busy until the returned token drops, freezing the
    /// companion worker for exactly that window.
    ///
    /// Taken at handler entry — before the memory check, before the multipart
    /// parse, before the upload streams — because the cores have to be free by
    /// the time the analysis wants them, not by the time it starts. The
    /// previous design raised its flag only once an analysis existed and
    /// covered the gap with a blanket seven-second quiet period that any
    /// request re-armed, including cache hits that did no work at all.
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

/// Returns an error if the router cannot be assembled or background resource
/// initialization cannot be scheduled.
pub async fn build_app(config: &ServerConfig) -> anyhow::Result<Router> {
    tracing::info!(model_dir = %config.model_dir().display(), "starting — resources loading in background");

    // Concurrency limit comes from --workers (defaults to cores/2 in main.rs).
    // CPU-bound cleave + ONNX work overlaps poorly across many threads, so a
    // smaller pool typically delivers higher aggregate throughput than 1/core.
    let max_concurrent = config.workers();
    let cores = crate::worker::cleave_concurrency(max_concurrent);
    tracing::info!(max_concurrent, cores, "concurrency limit set");

    // Built before the literal because the idle worker needs both: it is
    // frozen by `busy` and wound down by `shutdown`.
    let busy = Arc::new(idle::Busy::default());
    let shutdown = Arc::new(AtomicBool::new(false));
    // Started before the models load, not after: it is its own process and
    // loads its own, so there is nothing here for it to wait on.
    let idle_worker = if config.idle_worker_slots() > 0 {
        idle::start(
            config.hopper(),
            &crate::upload::default_worker_name(),
            &busy,
            &shutdown,
        )
    } else {
        tracing::info!("idle worker disabled: --idle-worker-slots is 0");
        None
    };

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
        cpu_busy: CpuBusy::default(),
        stuck_orphans: AtomicUsize::new(0),
        max_concurrent_tasks: max_concurrent,
        analysis_timeout_secs: config.analysis_timeout_secs(),
        reload_lock: tokio::sync::Mutex::new(()),
        overloaded_since: std::sync::Mutex::new(None),
        flights: Arc::new(flight::Flights::default()),
        in_flight: dashmap::DashMap::new(),
        hopper: config.hopper().map(str::to_owned),
        shutdown,
        job_buckets: Default::default(),
        job_types: Default::default(),
        job_overall: Default::default(),
        job_cached: Default::default(),
        lookups: Default::default(),
        jobs_started: AtomicU64::new(0),
        jobs_completed: AtomicU64::new(0),
        job_bytes_total: AtomicU64::new(0),
        job_micros_total: AtomicU64::new(0),
        // Decided here because AppState lives behind an Arc and cannot be
        // amended later. The worker itself starts once the models are loaded.
        busy,
        idle_worker,
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
                // Flush cleave's learned regex list so the next start of this
                // server prewarms the compiles this workload needed instead of
                // paying them on the first requests. Cheap when nothing new
                // compiled since the last tick.
                tokio::task::spawn_blocking(cleave::persist_regex_warm_memo);
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod whale_pool_tests {
    /// Per-whale pools scale with the host: a quarter of the cores, floor
    /// two, ceiling sixteen — the global pool keeps every core regardless.
    #[test]
    fn whale_pool_threads_scale_from_laptop_to_workstation() {
        assert_eq!(super::whale_pool_threads(1), 2);
        assert_eq!(super::whale_pool_threads(4), 2);
        assert_eq!(super::whale_pool_threads(8), 2);
        assert_eq!(super::whale_pool_threads(16), 4);
        assert_eq!(super::whale_pool_threads(64), 16);
        assert_eq!(super::whale_pool_threads(128), 16);
        assert_eq!(super::whale_pool_threads(256), 16);
    }

    /// Small-payload pools: an eighth of the cores, 2–8.
    #[test]
    fn small_pool_threads_scale_with_cores() {
        assert_eq!(super::small_pool_threads(1), 2);
        assert_eq!(super::small_pool_threads(4), 2);
        assert_eq!(super::small_pool_threads(16), 2);
        assert_eq!(super::small_pool_threads(32), 4);
        assert_eq!(super::small_pool_threads(64), 8);
        assert_eq!(super::small_pool_threads(256), 8);
    }

    /// Lane width follows the load: alone means the global pool, otherwise an
    /// even share of the cores floored at the tier width.
    #[test]
    fn lane_threads_share_cores_by_load() {
        assert_eq!(super::lane_threads(8, 64, 0), None);
        assert_eq!(super::lane_threads(8, 64, 1), None);
        assert_eq!(super::lane_threads(8, 64, 2), Some(32));
        assert_eq!(super::lane_threads(8, 64, 4), Some(16));
        assert_eq!(super::lane_threads(8, 64, 8), Some(8));
        assert_eq!(
            super::lane_threads(8, 64, 16),
            Some(8),
            "never below the floor"
        );
        assert_eq!(super::lane_threads(16, 64, 8), Some(16), "whale floor");
        assert_eq!(super::lane_threads(2, 4, 2), Some(2));
        assert_eq!(
            super::lane_threads(16, 4, 2),
            Some(16),
            "floor above the cores stays the floor"
        );
    }

    /// Big-whale slots: an eighth of the cores, 1–8, so slots × threads stays
    /// near two whale threads per core.
    #[test]
    fn whale_slots_scale_with_cores() {
        assert_eq!(super::whale_slots(1), 1);
        assert_eq!(super::whale_slots(4), 1);
        assert_eq!(super::whale_slots(8), 1);
        assert_eq!(super::whale_slots(16), 2);
        assert_eq!(super::whale_slots(64), 8);
        assert_eq!(super::whale_slots(128), 8);
        assert_eq!(super::whale_slots(256), 8);
        for cores in [4, 16, 64, 128] {
            assert!(
                super::whale_slots(cores) * super::whale_pool_threads(cores) <= 2 * cores.max(4)
            );
        }
    }

    /// A full slot table refuses instead of waiting, and a dropped slot is
    /// available again.
    #[test]
    fn whale_slot_refuses_when_full_and_frees_on_drop() {
        let held = super::WhaleSlot::try_acquire(1).expect("first slot");
        assert_eq!(super::WhaleSlot::in_use(), 1);
        assert_eq!(
            super::WhaleSlot::try_acquire(1).err(),
            Some(super::WhaleSlotBusy { slots: 1 }),
            "a full table refuses at once"
        );
        drop(held);
        assert_eq!(super::WhaleSlot::in_use(), 0);
        let again = super::WhaleSlot::try_acquire(1);
        assert!(again.is_ok(), "the slot is free again after drop");
        drop(again);
        assert_eq!(super::WhaleSlot::in_use(), 0);
    }
}

#[cfg(test)]
mod max_rss_tests {
    use crate::memory::{resolve_process_max_rss_bytes, resolve_worker_max_rss_gb};

    const GIB: u64 = 1024 * 1024 * 1024;

    /// A server and a worker read `--max-rss-gb` with the same vocabulary,
    /// differing only in the unit they answer in. Asserted from the server's
    /// side because `Startup::resolve` above is what feeds one of them, and a
    /// divergence would show up as a container sized by the wrong rule.
    #[test]
    fn max_rss_semantics_match_for_disabled_and_explicit_values() {
        assert_eq!(resolve_process_max_rss_bytes(-1), 0);
        assert_eq!(resolve_worker_max_rss_gb(-1), 0);

        assert_eq!(resolve_process_max_rss_bytes(3), 3 * GIB);
        assert_eq!(resolve_worker_max_rss_gb(3), 3);

        assert!(resolve_process_max_rss_bytes(0) > 0);
        assert!(resolve_worker_max_rss_gb(0) > 0);
    }
}
