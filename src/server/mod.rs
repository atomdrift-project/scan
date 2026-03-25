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

mod handlers;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::signal;
use tower::limit::ConcurrencyLimitLayer;

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
}

impl ServerConfig {
    /// Create a server configuration.
    ///
    /// `thresholds` may be `None` to use the model's recommended thresholds
    /// from `evaluation.json`, or `Some(t)` to override with explicit values.
    ///
    /// `max_body_size` and `max_rss_bytes` are byte counts.
    ///
    /// # Example
    /// ```
    /// use litmus::server::ServerConfig;
    ///
    /// let config = ServerConfig::new(
    ///     "127.0.0.1:8081".parse()?,
    ///     120,
    ///     100 * 1024 * 1024,
    ///     8 * 1024 * 1024 * 1024,
    ///     "/path/to/models",
    ///     None,
    ///     4_000,
    /// )?;
    ///
    /// assert_eq!(config.timeout_secs(), 120);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        bind: SocketAddr,
        timeout_secs: u64,
        max_body_size: usize,
        max_rss_bytes: u64,
        model_dir: impl Into<PathBuf>,
        thresholds: Option<Thresholds>,
        slow_rule_ms: u64,
    ) -> anyhow::Result<Self> {
        if let Some(ref t) = thresholds {
            t.validate()
                .map_err(|error| anyhow::anyhow!("invalid thresholds: {error}"))?;
        }
        Ok(Self {
            bind,
            timeout_secs,
            max_body_size,
            max_rss_bytes,
            model_dir: model_dir.into(),
            thresholds,
            slow_rule_ms,
        })
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
        );

        assert!(result.is_ok());
    }
}

#[derive(Debug)]
struct InFlightRequest {
    name: String,
    size_bytes: u64,
    started_at: Instant,
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
    ready: AtomicBool,
    resources: RwLock<Option<Arc<ModelResources>>>,
    next_request_id: AtomicU64,
    active_tasks: AtomicUsize,
    reload_lock: tokio::sync::Mutex<()>,
    overloaded_since: std::sync::Mutex<Option<Instant>>,
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

    let state = Arc::new(AppState {
        timeout_secs: config.timeout_secs(),
        max_upload_bytes: config.max_body_size(),
        max_rss_bytes: config.max_rss_bytes(),
        model_dir: config.model_dir().to_path_buf(),
        threshold_overrides: config.thresholds(),
        slow_rule_ms: config.slow_rule_ms(),
        ready: AtomicBool::new(false),
        resources: RwLock::new(None),
        next_request_id: AtomicU64::new(1),
        active_tasks: AtomicUsize::new(0),
        reload_lock: tokio::sync::Mutex::new(()),
        overloaded_since: std::sync::Mutex::new(None),
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
                (Ok(Err(e)), _, _) => tracing::error!("failed to load model: {e}"),
                (Err(e), _, _) => tracing::error!("model load task panicked: {e}"),
                (_, Err(e), _) => tracing::error!("shap load task panicked: {e}"),
                (_, _, Err(e)) => tracing::error!("yara warmup task panicked: {e}"),
            }
        });
    }

    let max_concurrent = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4)
        * 2;
    tracing::debug!(max_concurrent, "concurrency limit set");

    let analysis_routes = Router::new()
        .route("/analyze", post(handlers::analyze))
        .layer(ConcurrencyLimitLayer::new(max_concurrent));

    let app = Router::new()
        .route("/_/health", get(handlers::health))
        .route("/_/reload", post(handlers::reload))
        .route("/_/memory", get(handlers::memory_stats))
        .route("/_/requests", get(handlers::requests))
        .route("/_/threads", get(handlers::threads))
        .merge(analysis_routes)
        .layer(DefaultBodyLimit::max(config.max_body_size()))
        .with_state(state);

    Ok(app)
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
        "listening (resources loading in background)",
    );

    axum::serve(listener, app)
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
