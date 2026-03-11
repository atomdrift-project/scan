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
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
            threshold_suspicious: 0.75,
            threshold_hostile: 0.85,
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
    /// Current model resources; wrapped in `Arc` so handlers can snapshot
    /// without holding the lock across `await` points.
    pub resources: RwLock<Arc<ModelResources>>,
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

/// Start the HTTP server. Blocks until SIGINT or SIGTERM.
pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    tracing::info!(model_dir = %config.model_dir.display(), "loading resources");

    let thresholds = Thresholds {
        suspicious: config.threshold_suspicious,
        hostile: config.threshold_hostile,
    };

    // Load the ONNX model and warm up YARA/capability-mapper in parallel so
    // startup is not serialised on the slower of the two.
    let model_dir_for_model = config.model_dir.clone();
    let model_dir_for_shap = config.model_dir.clone();
    let slow_rule_ms = config.slow_rule_ms;

    let model_task = tokio::task::spawn_blocking(move || -> anyhow::Result<(Model, ExtractContext)> {
        let t = Instant::now();
        tracing::debug!("loading ONNX model and feature spec...");
        let model = Model::load(&model_dir_for_model, thresholds)?;
        let ctx = ExtractContext::new(&model.spec);
        tracing::info!(elapsed_ms = t.elapsed().as_millis(), "ONNX model loaded");
        Ok((model, ctx))
    });

    let warmup_task = tokio::task::spawn_blocking(move || {
        let t = Instant::now();
        tracing::debug!("warming up YARA rules and capability mapper...");
        let opts = cleave::AnalysisOptions { slow_rule_ms, ..Default::default() };
        let _ = cleave::analyze_file(std::path::Path::new("/dev/null"), &opts);
        tracing::info!(elapsed_ms = t.elapsed().as_millis(), "YARA warmup complete");
    });

    let shap_task = tokio::task::spawn_blocking(move || {
        tracing::debug!("loading SHAP importance data...");
        ShapImportance::load(&model_dir_for_shap).ok()
    });

    let (model_result, _, shap) = tokio::try_join!(model_task, warmup_task, shap_task)?;
    let (model, ctx) = model_result?;

    tracing::info!("all resources loaded");

    let state = Arc::new(AppState {
        timeout_secs: config.timeout_secs,
        max_rss_bytes: config.max_rss_bytes,
        model_dir: config.model_dir.clone(),
        threshold_suspicious: config.threshold_suspicious,
        threshold_hostile: config.threshold_hostile,
        slow_rule_ms: config.slow_rule_ms,
        resources: RwLock::new(Arc::new(ModelResources { model, shap, ctx })),
        next_request_id: AtomicU64::new(1),
        active_tasks: AtomicUsize::new(0),
        reload_lock: tokio::sync::Mutex::new(()),
        overloaded_since: std::sync::Mutex::new(None),
        in_flight: dashmap::DashMap::new(),
    });

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

    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    eprintln!(
        "Listening on http://{} (timeout: {}s, max size: {} MB) — Press Ctrl+C to stop",
        config.bind,
        config.timeout_secs,
        config.max_body_size / 1024 / 1024,
    );
    tracing::info!(
        bind = %config.bind,
        timeout_secs = config.timeout_secs,
        max_body_mb = config.max_body_size / 1024 / 1024,
        max_concurrent,
        "server ready",
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
