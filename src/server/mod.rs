//! HTTP API server for litmus malware classification.
//!
//! Accepts file uploads via multipart/form-data, runs cleave static analysis
//! and ONNX model inference, and returns a unified JSON result including
//! classification, SHAP explanations, and the full cleave report.
//!
//! Routes:
//!   GET  /health   — liveness check
//!   POST /analyze  — upload a file, receive full classification JSON
//!   POST /reload   — hot-reload model from disk
//!
//! [`ServerConfig`] keeps the public server surface intentionally small:
//! validated thresholds are supplied up front, and callers use accessors
//! rather than mutating fields after construction.

mod acl;
mod handlers;

pub use acl::{parse_cidr_list, Cidr};

use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use std::net::SocketAddr;
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
    timeout_secs: u64,
    max_body_size: usize,
    max_rss_bytes: u64,
    model_dir: PathBuf,
    thresholds: Option<Thresholds>,
    slow_rule_ms: u64,
    allowed_dirs: Vec<PathBuf>,
    extract_dir: Option<PathBuf>,
    workers: usize,
    allow_cidrs: Vec<Cidr>,
}

impl ServerConfig {
    /// Create a server configuration.
    ///
    /// `thresholds` may be `None` to use the model's recommended thresholds
    /// from `evaluation.json`, or `Some(t)` to override with explicit values.
    ///
    /// `max_body_size` and `max_rss_bytes` are byte counts.
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
    /// use litmus::server::ServerConfig;
    ///
    /// let config = ServerConfig::new(
    ///     "127.0.0.1:49999".parse()?,
    ///     120,
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
    /// assert_eq!(config.timeout_secs(), 120);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[allow(clippy::too_many_arguments)] // ServerConfig is plumbed once at startup; a builder would add ceremony for no real benefit.
    pub fn new(
        bind: SocketAddr,
        timeout_secs: u64,
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
            timeout_secs,
            max_body_size,
            max_rss_bytes,
            model_dir: model_dir.into(),
            thresholds,
            slow_rule_ms,
            allowed_dirs,
            extract_dir,
            workers,
            allow_cidrs,
        })
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

    /// Per-request analysis timeout in seconds.
    #[must_use]
    pub const fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// Maximum request body size in bytes.
    #[must_use]
    pub const fn max_body_size(&self) -> usize {
        self.max_body_size
    }

    /// Maximum RSS before rejecting requests.
    #[must_use]
    pub const fn max_rss_bytes(&self) -> u64 {
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
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn server_config_rejects_invalid_thresholds() {
        let result = ServerConfig::new(
            SocketAddr::from(([127, 0, 0, 1], 8081)),
            120,
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
            120,
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
            120,
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
}

#[derive(Debug)]
struct InFlightRequest {
    name: String,
    size_bytes: u64,
    started_at: Instant,
    /// Set to true when the HTTP side returns 504 but the blocking task is still running.
    timed_out: AtomicBool,
    /// Shared with the blocking task; set to true to request cooperative cancellation.
    cancellation: Arc<AtomicBool>,
}

#[derive(Debug)]
struct ModelResources {
    model: Model,
    shap: Option<ShapImportance>,
    ctx: ExtractContext,
}

#[derive(Debug)]
struct AppState {
    timeout_secs: u64,
    max_upload_bytes: usize,
    max_rss_bytes: u64,
    model_dir: PathBuf,
    threshold_overrides: Option<Thresholds>,
    slow_rule_ms: u64,
    allowed_dirs: Vec<PathBuf>,
    extract_dir: Option<PathBuf>,
    allow_cidrs: Vec<Cidr>,
    /// Process uptime anchor — captured when build_app runs, very close to
    /// process start. /_/health reports `now - started_at` as uptime_secs.
    started_at: Instant,
    ready: AtomicBool,
    init_error: RwLock<Option<String>>,
    resources: RwLock<Option<Arc<ModelResources>>>,
    next_request_id: AtomicU64,
    active_tasks: AtomicUsize,
    /// Tasks that timed out, exceeded the 120s grace period, and had their slot
    /// forcibly recycled. Their threads are still running in the background.
    /// A non-zero value means something is seriously wrong.
    recycled_orphans: AtomicUsize,
    /// Hard cap on concurrent analysis tasks. Requests are rejected with 503
    /// when active_tasks >= this value, preventing orphaned blocking tasks
    /// from piling up and consuming unbounded memory.
    max_concurrent_tasks: usize,
    reload_lock: tokio::sync::Mutex<()>,
    overloaded_since: std::sync::Mutex<Option<Instant>>,
    /// When all slots first became occupied exclusively by orphaned tasks.
    /// Reset to None when any live task is present or load drops below max.
    saturated_since: std::sync::Mutex<Option<Instant>>,
    in_flight: dashmap::DashMap<u64, InFlightRequest>,
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
        timeout_secs: config.timeout_secs(),
        max_upload_bytes: config.max_body_size(),
        max_rss_bytes: config.max_rss_bytes(),
        model_dir: config.model_dir().to_path_buf(),
        threshold_overrides: config.thresholds(),
        slow_rule_ms: config.slow_rule_ms(),
        allowed_dirs: config.allowed_dirs().to_vec(),
        extract_dir: config.extract_dir().map(PathBuf::from),
        allow_cidrs: config.allow_cidrs().to_vec(),
        started_at: Instant::now(),
        ready: AtomicBool::new(false),
        init_error: RwLock::new(None),
        resources: RwLock::new(None),
        next_request_id: AtomicU64::new(1),
        active_tasks: AtomicUsize::new(0),
        recycled_orphans: AtomicUsize::new(0),
        max_concurrent_tasks: max_concurrent,
        reload_lock: tokio::sync::Mutex::new(()),
        overloaded_since: std::sync::Mutex::new(None),
        saturated_since: std::sync::Mutex::new(None),
        in_flight: dashmap::DashMap::new(),
    });

    // Background task: load model + SHAP + YARA concurrently, then mark ready.
    {
        let bg = Arc::clone(&state);
        let model_dir = config.model_dir().to_path_buf();
        let model_dir_shap = config.model_dir().to_path_buf();
        let thresholds = config.thresholds();
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
                    let model = Model::load(&model_dir, thresholds)?;
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
                            "SHAP data unavailable (explanations disabled): {e}"
                        );
                        None
                    }
                }
            });
            let yara_spawned_at = Instant::now();
            let yara_task = tokio::task::spawn_blocking(move || {
                let queue_ms = yara_spawned_at.elapsed().as_millis();
                let t = Instant::now();
                tracing::info!(queue_ms, "YARA warmup started");
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
            });

            match tokio::join!(model_task, shap_task, yara_task) {
                (Ok(Ok((model, ctx))), Ok(shap), Ok(())) => {
                    let spec_version = model.spec().version();
                    let features = model.spec().total_features();
                    let shap_loaded = shap.is_some();
                    tracing::info!("all resources ready, installing into AppState");
                    match bg.resources.write() {
                        Ok(mut lock) => {
                            *lock = Some(Arc::new(ModelResources { model, shap, ctx }));
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
                    record_init_failure(&bg, &format!("failed to load model: {e}"))
                }
                (Err(e), _, _) => {
                    record_init_failure(&bg, &format!("model load task panicked: {e}"))
                }
                (_, Err(e), _) => {
                    record_init_failure(&bg, &format!("shap load task panicked: {e}"))
                }
                (_, _, Err(e)) => {
                    record_init_failure(&bg, &format!("yara warmup task panicked: {e}"))
                }
            }
        });
    }

    // Watchdog: if every slot has been occupied solely by orphaned (timed-out) tasks
    // for 600 seconds, something is irreparably stuck. Force-cancel all in-flight tasks,
    // wait 30 s for cooperative shutdown, then exit so the process manager restarts us.
    {
        let watchdog = Arc::clone(&state);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let active = watchdog.active_tasks.load(Ordering::Relaxed);
                let is_full = active >= watchdog.max_concurrent_tasks;
                let timed_out_count = watchdog
                    .in_flight
                    .iter()
                    .filter(|e| e.timed_out.load(Ordering::Relaxed))
                    .count();
                let live = active.saturating_sub(timed_out_count);

                if !is_full || live > 0 {
                    // Healthy or recovering — reset the clock.
                    if let Ok(mut s) = watchdog.saturated_since.lock() {
                        *s = None;
                    }
                    continue;
                }

                // All slots consumed by orphaned tasks.
                let elapsed = match watchdog.saturated_since.lock() {
                    Ok(mut s) => s.get_or_insert_with(Instant::now).elapsed(),
                    Err(_) => continue,
                };

                if elapsed.as_secs() < 60 {
                    tracing::warn!(
                        active_tasks = active,
                        max_concurrent_tasks = watchdog.max_concurrent_tasks,
                        elapsed_secs = elapsed.as_secs(),
                        "watchdog: all slots orphaned — will force-cancel in {}s if not recovered",
                        60u64.saturating_sub(elapsed.as_secs()),
                    );
                    continue;
                }

                // 60s elapsed — force-cancel every in-flight task, then restart.
                tracing::error!(
                    active_tasks = active,
                    max_concurrent_tasks = watchdog.max_concurrent_tasks,
                    elapsed_secs = elapsed.as_secs(),
                    "watchdog: all slots orphaned for 60s — force-cancelling all tasks",
                );
                for entry in watchdog.in_flight.iter() {
                    entry.cancellation.store(true, Ordering::Relaxed);
                }
                tokio::time::sleep(Duration::from_secs(30)).await;
                tracing::error!("watchdog: restarting after forced-cancellation grace period");
                std::process::exit(1);
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
        .route("/analyze", post(handlers::analyze))
        .route("/analyze-path", post(handlers::analyze_path))
        .layer(DefaultBodyLimit::max(config.max_body_size()))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            acl::acl,
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
    // Server mode processes many files over a long lifetime. Configure jemalloc
    // to aggressively return freed pages to the OS, preventing multi-GB RSS
    // growth from allocator fragmentation across thousands of analyses.
    cleave::memory_tracker::configure_jemalloc_low_memory();

    // Watchdog thread: enforces the same RSS limit as check_memory_pressure on
    // wall-clock time, independent of request traffic. This catches memory
    // growth that happens between requests (e.g. jemalloc fragmentation or
    // background YARA work).
    let _watchdog = cleave::memory_tracker::start_periodic_logging(
        std::time::Duration::from_secs(10),
        config.max_rss_bytes(),
    );

    let app = build_app(&config).await?;

    let listener = tokio::net::TcpListener::bind(config.bind()).await?;
    eprintln!(
        "Listening on http://{} (timeout: {}s, max size: {} MB, starting up) — Press Ctrl+C to stop",
        config.bind(),
        config.timeout_secs(),
        config.max_body_size() / 1024 / 1024,
    );
    tracing::info!(
        bind = %config.bind(),
        timeout_secs = config.timeout_secs(),
        max_body_mb = config.max_body_size() / 1024 / 1024,
        allow_cidrs = config.allow_cidrs().len(),
        "listening (resources loading in background)",
    );

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
