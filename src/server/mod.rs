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

mod handlers;

use axum::routing::{get, post};
use axum::Router;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::signal;
use tower::limit::ConcurrencyLimitLayer;
use axum::extract::DefaultBodyLimit;

use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Model, Thresholds};

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind to (default: 127.0.0.1:8081).
    pub bind: SocketAddr,
    /// Per-request analysis timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum request body size in bytes.
    pub max_body_size: usize,
    /// Maximum RSS before rejecting requests (Linux only).
    pub max_rss_bytes: u64,
    /// Directory containing model.onnx, feature_spec.json, shap_importance.json.
    pub model_dir: PathBuf,
    /// Probability threshold for suspicious classification.
    pub threshold_suspicious: f32,
    /// Probability threshold for hostile classification.
    pub threshold_hostile: f32,
    /// Warn when a single cleave rule takes longer than this many milliseconds.
    pub slow_rule_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 8081)),
            timeout_secs: 120,
            max_body_size: 100 * 1024 * 1024,      // 100 MB
            max_rss_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            model_dir: PathBuf::new(),
            threshold_suspicious: Thresholds::DEFAULT_SUSPICIOUS,
            threshold_hostile: Thresholds::DEFAULT_HOSTILE,
            slow_rule_ms: 4000,
        }
    }
}

/// Metadata for a request currently in the analysis pipeline.
#[derive(Debug)]
pub struct InFlightRequest {
    /// Original upload filename.
    pub name: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// When analysis began (after upload completed).
    pub started_at: Instant,
}

/// Loaded model resources; swapped atomically on /reload.
#[derive(Debug)]
pub struct ModelResources {
    /// Loaded ONNX model.
    pub model: Model,
    /// SHAP importance data for explanations.
    pub shap: Option<ShapImportance>,
    /// Feature extraction context built from the model spec.
    pub ctx: ExtractContext,
}

/// Shared application state, held behind an `Arc`.
#[derive(Debug)]
pub struct AppState {
    /// Per-request analysis timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum upload size in bytes (defense-in-depth, mirrors DefaultBodyLimit).
    pub max_upload_bytes: usize,
    /// Maximum RSS before rejecting requests.
    pub max_rss_bytes: u64,
    /// Directory containing model artifacts.
    pub model_dir: PathBuf,
    /// Minimum probability to classify as suspicious.
    pub threshold_suspicious: f32,
    /// Minimum probability to classify as hostile.
    pub threshold_hostile: f32,
    /// Warn when a single cleave rule takes longer than this many milliseconds.
    pub slow_rule_ms: u64,
    /// True once model resources have been loaded and the server is ready to
    /// serve analysis requests.  Set with `Release` ordering; read with
    /// `Acquire` so the resources write is always visible before this flips.
    pub ready: AtomicBool,
    /// Current model resources; wrapped in `Arc` so handlers can snapshot
    /// without holding the lock across `await` points.  `None` while the
    /// server is still initialising.
    pub resources: RwLock<Option<Arc<ModelResources>>>,
    /// Monotonically increasing request counter.
    pub next_request_id: AtomicU64,
    /// Number of analysis tasks currently in flight.
    pub active_tasks: AtomicUsize,
    /// Serialises /reload — only one reload may run at a time.
    pub reload_lock: tokio::sync::Mutex<()>,
    /// Tracks when the server first entered a memory-overloaded state.
    pub overloaded_since: std::sync::Mutex<Option<Instant>>,
    /// Currently in-flight analysis requests, keyed by request ID.
    pub in_flight: dashmap::DashMap<u64, InFlightRequest>,
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
pub async fn build_app(config: &ServerConfig) -> anyhow::Result<Router> {
    tracing::info!(model_dir = %config.model_dir.display(), "starting — resources loading in background");

    let state = Arc::new(AppState {
        timeout_secs: config.timeout_secs,
        max_upload_bytes: config.max_body_size,
        max_rss_bytes: config.max_rss_bytes,
        model_dir: config.model_dir.clone(),
        threshold_suspicious: config.threshold_suspicious,
        threshold_hostile: config.threshold_hostile,
        slow_rule_ms: config.slow_rule_ms,
        ready: AtomicBool::new(false),
        resources: RwLock::new(None),
        next_request_id: AtomicU64::new(1),
        active_tasks: AtomicUsize::new(0),
        reload_lock: tokio::sync::Mutex::new(()),
        overloaded_since: std::sync::Mutex::new(None),
        in_flight: dashmap::DashMap::new(),
    });

    // Background task: load model + SHAP concurrently, then mark the server ready.
    {
        let bg = Arc::clone(&state);
        let model_dir = config.model_dir.clone();
        let model_dir_shap = config.model_dir.clone();
        let thresholds = Thresholds {
            suspicious: config.threshold_suspicious,
            hostile: config.threshold_hostile,
        };
        tokio::spawn(async move {
            let init_start = Instant::now();
            tracing::info!("resource loader started (model + SHAP loading concurrently)");

            // Capture spawn times in the async context so each blocking closure
            // can report queue_ms (time waiting for a thread) separately from
            // work_ms (time actually doing I/O and parsing).
            let model_spawned_at = Instant::now();
            let model_task = tokio::task::spawn_blocking(move || -> anyhow::Result<(Model, ExtractContext)> {
                let queue_ms = model_spawned_at.elapsed().as_millis();
                let t = Instant::now();
                tracing::info!(queue_ms, "loading ONNX model and feature spec");
                let model = Model::load(&model_dir, thresholds)?;
                let ctx = ExtractContext::new(&model.spec);
                tracing::info!(
                    queue_ms,
                    work_ms = t.elapsed().as_millis(),
                    spec_version = model.spec.version,
                    features = model.spec.total_features,
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
                        tracing::info!(queue_ms, work_ms = t.elapsed().as_millis(), "SHAP data loaded");
                        Some(shap)
                    }
                    Err(e) => {
                        tracing::warn!(queue_ms, work_ms = t.elapsed().as_millis(), "SHAP data unavailable (explanations disabled): {e}");
                        None
                    }
                }
            });

            match tokio::join!(model_task, shap_task) {
                (Ok(Ok((model, ctx))), Ok(shap)) => {
                    let spec_version = model.spec.version;
                    let features = model.spec.total_features;
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
                (Ok(Err(e)), _) => tracing::error!("failed to load model: {e}"),
                (Err(e), _) => tracing::error!("model load task panicked: {e}"),
                (_, Err(e)) => tracing::error!("shap load task panicked: {e}"),
            }
        });
    }

    // Fire-and-forget YARA warmup — does not block readiness.  If it finishes
    // before the first request arrives, great; otherwise the first request
    // naturally warms the rules.
    {
        let slow_rule_ms = config.slow_rule_ms;
        tokio::spawn(tokio::task::spawn_blocking(move || {
            let t = Instant::now();
            tracing::info!("YARA warmup started (non-blocking)");
            let opts = cleave::AnalysisOptions { slow_rule_ms, ..Default::default() };
            let _ = cleave::analyze_file(std::path::Path::new("/dev/null"), &opts);
            tracing::info!(elapsed_ms = t.elapsed().as_millis(), "YARA warmup complete");
        }));
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
        .layer(DefaultBodyLimit::max(config.max_body_size))
        .with_state(state);

    Ok(app)
}

/// Start the HTTP server. Blocks until SIGINT or SIGTERM.
pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    // Server mode processes many files over a long lifetime. Configure jemalloc
    // to aggressively return freed pages to the OS, preventing multi-GB RSS
    // growth from allocator fragmentation across thousands of analyses.
    cleave::memory_tracker::configure_jemalloc_low_memory();

    let app = build_app(&config).await?;

    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    eprintln!(
        "Listening on http://{} (timeout: {}s, max size: {} MB, starting up) — Press Ctrl+C to stop",
        config.bind,
        config.timeout_secs,
        config.max_body_size / 1024 / 1024,
    );
    tracing::info!(
        bind = %config.bind,
        timeout_secs = config.timeout_secs,
        max_body_mb = config.max_body_size / 1024 / 1024,
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
