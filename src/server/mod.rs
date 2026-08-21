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
mod flight;
mod handlers;

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
use std::sync::{Arc, RwLock};
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
}

/// Default per-request analysis timeout: 20 minutes. Covers cold cleave scans
/// of large archives — and fetch-enabled scans whose dependency analysis can
/// far outlast the sample's own — while still preventing a pathological input
/// from pinning a slot forever. Override with `--analysis-timeout` /
/// [`ServerConfig::with_analysis_timeout`].
pub const DEFAULT_ANALYSIS_TIMEOUT_SECS: u64 = 1200;

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
pub(super) struct RequestGuard {
    request_id: u64,
    state: Arc<AppState>,
    cancellation: Arc<AtomicBool>,
    /// Held here so the semaphore permit is released when the guard drops.
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl RequestGuard {
    fn new(
        request_id: u64,
        state: Arc<AppState>,
        cancellation: Arc<AtomicBool>,
        permit: tokio::sync::OwnedSemaphorePermit,
    ) -> Self {
        Self {
            request_id,
            state,
            cancellation,
            _permit: permit,
        }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        // Signal the blocking thread to stop cooperatively, then remove the
        // in-flight entry. The permit is released automatically via _permit.
        self.cancellation.store(true, Ordering::Release);
        self.state.in_flight.remove(&self.request_id);
    }
}

#[derive(Debug)]
pub(crate) struct ModelResources {
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

#[derive(Debug)]
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
    /// Analyses in progress, so concurrent requests for the same artifact
    /// share one run instead of each taking a slot. See [`flight`].
    flights: Arc<flight::Flights>,
    /// Background hopper uploader (`--hopper`); `None` disables result renewal.
    /// Shared across handlers; each analyzed result is queued to its own thread,
    /// so uploads never block the analyze response.
    uploader: Option<Arc<crate::upload::Uploader>>,
}

impl AppState {
    fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
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
/// Returns an error if the router cannot be assembled or background resource
/// initialization cannot be scheduled.
pub async fn build_app(config: &ServerConfig) -> anyhow::Result<Router> {
    tracing::info!(model_dir = %config.model_dir().display(), "starting — resources loading in background");

    // Concurrency limit comes from --workers (defaults to cores/2 in main.rs).
    // CPU-bound cleave + ONNX work overlaps poorly across many threads, so a
    // smaller pool typically delivers higher aggregate throughput than 1/core.
    let max_concurrent = config.workers();
    tracing::info!(max_concurrent, "concurrency limit set");

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
        stuck_orphans: AtomicUsize::new(0),
        max_concurrent_tasks: max_concurrent,
        analysis_timeout_secs: config.analysis_timeout_secs(),
        reload_lock: tokio::sync::Mutex::new(()),
        overloaded_since: std::sync::Mutex::new(None),
        flights: Arc::new(flight::Flights::default()),
        in_flight: dashmap::DashMap::new(),
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
    });

    // Background task: load model + SHAP + YARA concurrently, then mark ready.
    {
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
                            *lock = Some(Arc::new(ModelResources {
                                model,
                                shap,
                                ctx,
                                interpret: bg.interpret.clone(),
                                fetch: bg.fetch,
                                zip_passwords: bg.zip_passwords.clone(),
                            }));
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
                let available = watchdog.slots.available_permits();
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
        .route("/_/reload", post(handlers::reload))
        .route("/_/update", post(handlers::update))
        .route("/_/memory", get(handlers::memory_stats))
        .route("/_/requests", get(handlers::requests))
        .route("/_/threads", get(handlers::threads))
        .route("/lookup", get(handlers::lookup))
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
