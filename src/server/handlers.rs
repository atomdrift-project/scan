//! HTTP request handlers for the litmus API server.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::Builder as TempBuilder;

use super::AppState;

fn analysis_error_response(error: &anyhow::Error) -> Response {
    let (status, message) = classify_analysis_error(error.root_cause().to_string().as_str());
    let detail = format!("{error:#}");

    (
        status,
        Json(if detail == message {
            serde_json::json!({ "error": message })
        } else {
            serde_json::json!({ "error": message, "detail": detail })
        }),
    )
        .into_response()
}

fn classify_analysis_error(message: &str) -> (StatusCode, String) {
    let normalized = message.to_ascii_lowercase();

    let status = if normalized.contains("unsupported file type")
        || normalized.contains("unsupported archive type")
        || normalized.contains("unsupported compression")
    {
        StatusCode::UNSUPPORTED_MEDIA_TYPE
    } else if normalized.contains("archive is encrypted but no passwords configured")
        || normalized.contains("invalid ")
        || normalized.contains("not a valid ")
        || normalized.contains("truncated")
        || normalized.contains("too small")
        || normalized.contains("out of bounds")
        || normalized.contains("empty package.json")
        || normalized.contains("maximum archive depth")
        || normalized.contains("maximum decode depth")
        || normalized.contains("exceeded maximum")
        || normalized.contains("file count limit exceeded")
    {
        StatusCode::UNPROCESSABLE_ENTITY
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };

    (status, message.to_string())
}

fn analysis_error_status(error: &anyhow::Error) -> StatusCode {
    classify_analysis_error(error.root_cause().to_string().as_str()).0
}
use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::Model;
use crate::scan::ScanResult;

/// GET /_/health — liveness check with memory and concurrency status.
/// Returns 503 while resources are still loading or when RSS exceeds the
/// configured limit. A fully-utilised worker pool returns 200 with
/// `status: "saturated"` — that's the target steady state, not a fault.
///
/// Every response carries `uptime_secs` (seconds since the server started)
/// so clients can detect restarts without polling a separate endpoint.
pub(super) async fn health(State(state): State<Arc<AppState>>) -> Response {
    let uptime_secs = state.started_at.elapsed().as_secs();

    if let Ok(init_error) = state.init_error.read() {
        if let Some(message) = init_error.as_ref() {
            tracing::error!("GET /_/health -> 503 (failed: {message})");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "failed",
                    "reason": "initialization_failed",
                    "uptime_secs": uptime_secs,
                })),
            )
                .into_response();
        }
    }

    if !state.ready.load(std::sync::atomic::Ordering::Acquire) {
        tracing::debug!("GET /_/health -> 503 (starting)");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "starting",
                "uptime_secs": uptime_secs,
            })),
        )
            .into_response();
    }

    let rss_bytes = cleave::memory_tracker::current_rss();
    let rss_mb = rss_bytes.map(|b| b / 1024 / 1024);
    let active_tasks = state
        .active_tasks
        .load(std::sync::atomic::Ordering::Relaxed);
    let overloaded = rss_bytes.map(|b| b > state.max_rss_bytes).unwrap_or(false);

    if overloaded {
        tracing::warn!("GET /_/health -> 503 (degraded, rss={rss_mb:?}MB)");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "degraded",
                "reason": "memory_pressure",
                "rss_mb": rss_mb,
                "max_rss_mb": state.max_rss_bytes / 1024 / 1024,
                "active_tasks": active_tasks,
                "uptime_secs": uptime_secs,
                "rayon_threads": rayon::current_num_threads(),
            })),
        )
            .into_response();
    }
    let max_tasks = state.max_concurrent_tasks;
    let orphaned_tasks = state
        .in_flight
        .iter()
        .filter(|e| e.timed_out.load(std::sync::atomic::Ordering::Relaxed))
        .count();
    let live_tasks = active_tasks.saturating_sub(orphaned_tasks);
    let load = if max_tasks > 0 {
        live_tasks as f64 / max_tasks as f64
    } else {
        0.0
    };
    // A fully-utilised worker pool is the *target* steady state, not a fault.
    // Report it as "saturated" with HTTP 200 so monitors can distinguish "all
    // slots busy" from real failures (memory pressure, stuck workers). The
    // /analyze endpoint still rejects with 503 when active >= max, so clients
    // back off correctly without /_/health pretending the server is unhealthy.
    let saturated = active_tasks >= max_tasks;
    if saturated {
        let oldest = state
            .in_flight
            .iter()
            .min_by_key(|e| e.started_at)
            .map(|e| (e.name.clone(), e.started_at.elapsed().as_secs()));
        tracing::debug!(
            active_tasks,
            live_tasks,
            orphaned_tasks,
            max_concurrent_tasks = max_tasks,
            oldest_task = ?oldest,
            "GET /_/health -> 200 (saturated)"
        );
        return Json(serde_json::json!({
            "status": "saturated",
            "reason": "thread_pool_saturated",
            "rss_mb": rss_mb,
            "active_tasks": active_tasks,
            "live_tasks": live_tasks,
            "orphaned_tasks": orphaned_tasks,
            "max_concurrent_tasks": max_tasks,
            "oldest_task": oldest.map(|(name, secs)| serde_json::json!({"name": name, "elapsed_secs": secs})),
            "load": load,
            "uptime_secs": uptime_secs,
            "rayon_threads": rayon::current_num_threads(),
        }))
            .into_response();
    }
    tracing::debug!("GET /_/health -> 200 (rss={rss_mb:?}MB, live={live_tasks}, orphaned={orphaned_tasks}, load={load:.2})");
    Json(serde_json::json!({
        "status": "ok",
        "rss_mb": rss_mb,
        "active_tasks": active_tasks,
        "live_tasks": live_tasks,
        "orphaned_tasks": orphaned_tasks,
        "max_concurrent_tasks": max_tasks,
        "load": load,
        "uptime_secs": uptime_secs,
        "rayon_threads": rayon::current_num_threads(),
    }))
    .into_response()
}

/// GET /_/info — static server capacity and version info.
///
/// Returned by litmus on startup so clients (e.g. hopper) can size their
/// per-node worker pools without hand-configuring slot counts, and so they
/// can compare model/traits commits across nodes for drift detection.
/// Always 200, independent of readiness — readiness is reported by /_/health.
pub(super) async fn info(State(state): State<Arc<AppState>>) -> Response {
    let cpus = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(0);
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "slots": state.max_concurrent_tasks,
        "cpus": cpus,
        "max_upload_mb": state.max_upload_bytes / 1024 / 1024,
        "max_rss_mb": state.max_rss_bytes / 1024 / 1024,
        "model_commit": crate::models_repo::version(),
        "traits_commit": cleave::traits_repo::version(),
    }))
    .into_response()
}

/// Outcome of [`do_model_reload`] — caller maps this to an HTTP response.
struct ReloadOutcome {
    elapsed_ms: u128,
}

/// Perform the cleave-traits reload + model load + atomic swap. Caller is
/// responsible for holding `state.reload_lock`. Shared by /_/reload and
/// /_/update so the load+swap dance lives in exactly one place.
async fn do_model_reload(
    state: &Arc<AppState>,
) -> Result<ReloadOutcome, (StatusCode, &'static str)> {
    let start = Instant::now();
    let model_dir = state.model_dir.clone();
    let thresholds = state.threshold_overrides;

    let result = tokio::task::spawn_blocking(move || {
        // Reload cleave traits first so the new model runs against fresh rules.
        if let Err(e) = cleave::reload_capability_mapper() {
            tracing::warn!("cleave trait reload failed (previous traits retained): {e}");
        } else {
            tracing::info!("cleave traits reloaded");
        }
        cleave::clear_all_thread_caches();

        let model = Model::load(&model_dir, thresholds)?;
        let shap = ShapImportance::load(&model_dir).ok();
        let ctx = ExtractContext::new(model.spec());
        Ok::<_, anyhow::Error>((model, shap, ctx))
    })
    .await;

    let elapsed_ms = start.elapsed().as_millis();

    let (model, shap, ctx) = match result {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            // Log internally but do not expose filesystem paths or model internals to callers.
            tracing::warn!("reload failed (previous model retained) in {elapsed_ms}ms: {e}");
            return Err((StatusCode::UNPROCESSABLE_ENTITY, "Failed to load model"));
        }
        Err(e) => {
            tracing::warn!("reload task join error: {e}");
            return Err((StatusCode::INTERNAL_SERVER_ERROR, "Internal error"));
        }
    };

    let spec_version = model.spec().version();
    let features = model.spec().total_features();
    let shap_loaded = shap.is_some();
    let was_ready = state.ready.load(std::sync::atomic::Ordering::Relaxed);

    match state.resources.write() {
        Ok(mut lock) => {
            *lock = Some(Arc::new(super::ModelResources { model, shap, ctx }));
            if let Ok(mut init_error) = state.init_error.write() {
                *init_error = None;
            }
            state
                .ready
                .store(true, std::sync::atomic::Ordering::Release);
            if was_ready {
                tracing::info!(
                    elapsed_ms,
                    spec_version,
                    features,
                    shap_loaded,
                    "model reloaded"
                );
            } else {
                tracing::info!(
                    elapsed_ms,
                    spec_version,
                    features,
                    shap_loaded,
                    "model loaded via reload — server now ready"
                );
            }
            Ok(ReloadOutcome { elapsed_ms })
        }
        Err(e) => {
            tracing::error!("write lock poisoned during reload: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, "Internal error"))
        }
    }
}

/// POST /reload — reload model from disk, swap atomically.
///
/// Only one reload may run at a time; concurrent calls receive 409.
pub(super) async fn reload(State(state): State<Arc<AppState>>) -> Response {
    tracing::info!("POST /_/reload");
    // Prevent concurrent reloads — each load allocates significant memory.
    let Ok(_guard) = state.reload_lock.try_lock() else {
        tracing::warn!("reload rejected: already in progress");
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "Reload already in progress"})),
        )
            .into_response();
    };

    match do_model_reload(&state).await {
        Ok(outcome) => Json(serde_json::json!({
            "status": "ok",
            "elapsed_ms": outcome.elapsed_ms,
        }))
        .into_response(),
        Err((status, msg)) => (status, Json(serde_json::json!({ "error": msg }))).into_response(),
    }
}

/// POST /_/update — pull latest models + traits from their git repos, then
/// reload. Both pulls are non-fatal: if either fails, the response reports
/// it but the reload still runs against whatever is currently on disk so
/// the operator gets the most-recent-good state. Shares the reload_lock
/// with /_/reload so concurrent calls receive 409.
pub(super) async fn update(State(state): State<Arc<AppState>>) -> Response {
    tracing::info!("POST /_/update");
    let Ok(_guard) = state.reload_lock.try_lock() else {
        tracing::warn!("update rejected: reload already in progress");
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "Reload already in progress"})),
        )
            .into_response();
    };

    // Run the two repo pulls on a blocking thread; they're synchronous git
    // and filesystem work. Both are non-fatal — log on failure and proceed
    // to the reload step regardless so we still pick up any partial state.
    let pulls = tokio::task::spawn_blocking(|| {
        let models_err = match crate::models_repo::update() {
            Ok(()) => None,
            Err(e) => {
                tracing::warn!("models update failed: {e}");
                Some(format!("{e}"))
            }
        };
        let traits_err = match cleave::traits_repo::update(false) {
            Ok(()) => None,
            Err(e) => {
                tracing::warn!("traits update failed: {e}");
                Some(format!("{e}"))
            }
        };
        (models_err, traits_err)
    })
    .await;

    let (models_err, traits_err) = match pulls {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("update pull task join error: {e}");
            (
                Some("task join failed".to_string()),
                Some("task join failed".to_string()),
            )
        }
    };

    match do_model_reload(&state).await {
        Ok(outcome) => Json(serde_json::json!({
            "status": "ok",
            "elapsed_ms": outcome.elapsed_ms,
            "models_updated": models_err.is_none(),
            "traits_updated": traits_err.is_none(),
            "models_error": models_err,
            "traits_error": traits_err,
            "version": env!("CARGO_PKG_VERSION"),
            "model_commit": crate::models_repo::version(),
            "traits_commit": cleave::traits_repo::version(),
        }))
        .into_response(),
        Err((status, msg)) => (
            status,
            Json(serde_json::json!({
                "status": "reload_failed",
                "error": msg,
                "models_updated": models_err.is_none(),
                "traits_updated": traits_err.is_none(),
                "models_error": models_err,
                "traits_error": traits_err,
            })),
        )
            .into_response(),
    }
}

/// RAII guard to ensure active_tasks and in_flight entries are cleaned up.
struct TaskGuard {
    state: Arc<AppState>,
    request_id: u64,
    start_time: Instant,
}

impl TaskGuard {
    fn new(state: Arc<AppState>, request_id: u64, name: String, size_bytes: u64) -> Self {
        state.active_tasks.fetch_add(1, Ordering::SeqCst);
        state.in_flight.insert(
            request_id,
            super::InFlightRequest {
                name,
                size_bytes,
                started_at: Instant::now(),
                timed_out: AtomicBool::new(false),
            },
        );
        Self {
            state,
            request_id,
            start_time: Instant::now(),
        }
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        let was_orphaned = if let Some(req) = self.state.in_flight.get(&self.request_id) {
            req.timed_out.load(Ordering::Relaxed)
        } else {
            false
        };

        self.state.active_tasks.fetch_sub(1, Ordering::SeqCst);
        self.state.in_flight.remove(&self.request_id);

        if was_orphaned {
            tracing::info!(
                id = self.request_id,
                duration_ms = self.start_time.elapsed().as_millis() as u64,
                "orphaned task finally finished and released slot"
            );
        }
    }
}

/// POST /analyze — accept multipart file upload, classify, return full JSON result.
pub(super) async fn analyze(
    State(state): State<Arc<AppState>>,
    mut multipart: axum::extract::Multipart,
) -> Response {
    let request_id = state.next_request_id();
    let request_start = Instant::now();

    tracing::info!(id = request_id, "--> POST /analyze");

    if let Ok(init_error) = state.init_error.read() {
        if let Some(message) = init_error.as_ref() {
            tracing::error!("analyze rejected: startup failed  id={request_id} error={message}");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "Server failed to initialize"})),
            )
                .into_response();
        }
    }

    if let Some(response) = check_memory_pressure(&state) {
        return response;
    }

    // Parse the first multipart field as the file.
    let mut field = match multipart.next_field().await {
        Ok(Some(f)) => f,
        Ok(None) => {
            tracing::warn!("bad request: no file field  id={request_id}");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "No file field in request"})),
            )
                .into_response();
        }
        Err(e) => {
            tracing::warn!("failed to parse multipart: {e}  id={request_id}");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid multipart data"})),
            )
                .into_response();
        }
    };

    // Sanitize the uploaded filename: replace any character outside [A-Za-z0-9_.-] with _,
    // collapse .. to prevent path traversal, and right-truncate to 63 characters so that
    // the extension is preserved. Used as both the `path` label and the temp file suffix
    // so that cleave can detect the file type from the extension.
    let filename: String = {
        let raw = field
            .file_name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("upload-{request_id}"));
        let sanitized: String = raw
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>()
            .replace("..", "__");
        if sanitized.len() > 63 {
            // Right-truncate: preserve the tail (and thus the extension).
            // All remaining chars are ASCII so byte indexing is safe.
            sanitized[sanitized.len() - 63..].to_string()
        } else {
            sanitized
        }
    };

    // Create temp file with the sanitized filename as suffix so cleave detects the file type.
    let suffix = format!("_{filename}");
    let temp_file =
        match tokio::task::spawn_blocking(move || TempBuilder::new().suffix(&suffix).tempfile())
            .await
        {
            Ok(Ok(f)) => f,
            Ok(Err(e)) => {
                tracing::warn!("failed to create temp file: {e}  id={request_id}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "Internal error"})),
                )
                    .into_response();
            }
            Err(e) => {
                tracing::warn!("temp file task join error (panic?): {e}  id={request_id}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "Internal error"})),
                )
                    .into_response();
            }
        };

    // Stream multipart field to the temp file.
    let path = temp_file.path().to_owned();
    let writer = match temp_file.reopen() {
        Ok(file) => file,
        Err(e) => {
            tracing::warn!("failed to reopen temp file for writing: {e}  id={request_id}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response();
        }
    };
    let mut tokio_file = tokio::fs::File::from_std(writer);

    let max_upload = state.max_upload_bytes;
    let mut file_size = 0usize;
    loop {
        match field.chunk().await {
            Ok(Some(chunk)) if !chunk.is_empty() => {
                file_size += chunk.len();
                if file_size > max_upload {
                    tracing::warn!(
                        "upload exceeded size limit: {file_size} > {max_upload}  id={request_id}"
                    );
                    return (
                        StatusCode::PAYLOAD_TOO_LARGE,
                        Json(serde_json::json!({"error": "File too large"})),
                    )
                        .into_response();
                }
                if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut tokio_file, &chunk).await {
                    tracing::warn!("failed to write chunk: {e}  id={request_id}");
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": "Failed to save file data"})),
                    )
                        .into_response();
                }
            }
            Ok(Some(_)) => continue,
            Ok(None) => break,
            Err(e) => {
                tracing::warn!("error reading multipart chunk: {e}  id={request_id}");
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": "Error reading upload data"})),
                )
                    .into_response();
            }
        }
    }

    if let Err(e) = tokio::io::AsyncWriteExt::flush(&mut tokio_file).await {
        tracing::warn!("failed to flush temp file: {e}  id={request_id}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Failed to save file data"})),
        )
            .into_response();
    }
    if let Err(e) = tokio_file.sync_all().await {
        tracing::warn!("failed to sync temp file: {e}  id={request_id}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Failed to save file data"})),
        )
            .into_response();
    }
    drop(tokio_file);

    if file_size == 0 {
        tracing::warn!("bad request: empty file  id={request_id}");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Empty file"})),
        )
            .into_response();
    }

    tracing::info!(
        id = request_id,
        filename = %filename,
        size_bytes = file_size,
        upload_ms = request_start.elapsed().as_millis() as u64,
        "received file, starting analysis",
    );

    // Snapshot the current model resources (Arc clone, no lock held during analysis).
    let resources = match state.resources.read() {
        Ok(lock) => match lock.as_ref() {
            Some(r) => Arc::clone(r),
            None => {
                tracing::debug!("analyze rejected: resources not yet loaded  id={request_id}");
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({"error": "Server starting up"})),
                )
                    .into_response();
            }
        },
        Err(e) => {
            tracing::error!("read lock poisoned: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response();
        }
    };

    // Hard gate: reject immediately if at capacity.
    let active = state
        .active_tasks
        .load(std::sync::atomic::Ordering::Relaxed);
    if active >= state.max_concurrent_tasks {
        tracing::warn!(
            id = request_id,
            filename = %filename,
            size_bytes = file_size,
            active_tasks = active,
            max = state.max_concurrent_tasks,
            "rejecting: at capacity"
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Server overloaded (too many active analyses)"})),
        )
            .into_response();
    }

    let slow_rule_ms = state.slow_rule_ms;
    let timeout_duration = Duration::from_secs(state.timeout_secs);

    // Claim a slot using the RAII guard.
    let guard = TaskGuard::new(
        Arc::clone(&state),
        request_id,
        filename.clone(),
        file_size as u64,
    );

    let filename_for_closure = filename.clone();
    let should_clear_caches = request_id.is_multiple_of(50);
    let cancellation = Arc::new(AtomicBool::new(false));
    let cancel_flag = Arc::clone(&cancellation);
    let mut handle = tokio::task::spawn_blocking(move || {
        let _moved_guard = guard;
        let result = classify_file(
            &path,
            &filename_for_closure,
            &resources,
            slow_rule_ms,
            None,
            Some(cancel_flag),
        );
        if should_clear_caches {
            cleave::clear_all_thread_caches();
        }
        drop(temp_file);
        result
    });

    let result = tokio::select! {
        res = &mut handle => Some(res),
        _ = tokio::time::sleep(timeout_duration) => {
            cancellation.store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(req) = state.in_flight.get(&request_id) {
                req.timed_out.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            None
        }
    };

    let elapsed_ms = request_start.elapsed().as_millis();

    match result {
        Some(Ok(Ok(scan_result))) => {
            tracing::info!(
                id = request_id,
                filename = %filename,
                size_bytes = file_size,
                elapsed_ms = elapsed_ms as u64,
                classification = %scan_result.classification,
                probability = scan_result.probability,
                "<-- 200 OK",
            );
            let mut resp = Json(scan_result.to_envelope()).into_response();
            resp.headers_mut()
                .insert("X-Total-Ms", (elapsed_ms as u64).into());
            resp
        }
        Some(Ok(Err(e))) => {
            let status = analysis_error_status(&e);
            tracing::warn!(id = request_id, elapsed_ms = elapsed_ms as u64, status = status.as_u16(), error = %e, "analysis failed");
            analysis_error_response(&e)
        }
        Some(Err(e)) => {
            tracing::warn!(id = request_id, elapsed_ms = elapsed_ms as u64, error = %e, "<-- 500 task join error (panic?)");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response()
        }
        None => {
            tracing::warn!(
                id = request_id,
                elapsed_ms = elapsed_ms as u64,
                "<-- 504 timeout after {}s",
                state.timeout_secs
            );
            (
                StatusCode::GATEWAY_TIMEOUT,
                Json(serde_json::json!({"error": "Analysis timed out"})),
            )
                .into_response()
        }
    }
}

/// Run the full cleave + litmus pipeline on `path`, returning a `ScanResult`.
///
/// Runs on a blocking thread. `label` is used as the `path` field in the result
/// (the original upload filename, not the temp file path).
fn classify_file(
    path: &std::path::Path,
    label: &str,
    resources: &super::ModelResources,
    slow_rule_ms: u64,
    extract_dir: Option<&std::path::Path>,
    cancellation: Option<Arc<AtomicBool>>,
) -> anyhow::Result<ScanResult> {
    use anyhow::Context as _;

    let sample_extraction =
        extract_dir.map(|d| cleave::SampleExtractionConfig::new(d.to_path_buf()));
    let opts = cleave::AnalysisOptions {
        slow_rule_ms,
        sample_extraction,
        cancellation: cancellation.clone(),
        ..Default::default()
    };
    let report =
        cleave::analyze_file(path, &opts).with_context(|| format!("cleave analysis of {label}"))?;

    let cr = crate::scan::classify_report(
        label,
        report,
        &resources.ctx,
        &resources.model,
        resources.shap.as_ref(),
        cancellation,
    )?;

    Ok(ScanResult {
        v: "4",
        classification: cr.classification,
        probability: cr.probability,
        thresholds: resources.model.thresholds(),
        version: crate::scan::model_version_string(resources.model.info()),
        analyzed_at: crate::scan::now_rfc3339(),
        cleave: Some(cr.report_json),
        pids: None,
        deleted: None,
        path: label.to_string(),
        finding_counts: cr.finding_counts,
        formula: cr.formula,
        reasons: cr.reasons,
        top_findings: cr.top_findings,
        file_type: cr.file_type,
        size_bytes: cr.size_bytes,
        sha256: cr.sha256,
        embedded_files: cr.embedded_files,
    })
}

/// Check RSS memory pressure. Returns a 503 response if the server is overloaded,
// --- /analyze-path endpoint ---

#[derive(serde::Deserialize)]
pub(super) struct AnalyzePathRequest {
    path: String,
}

/// POST /analyze-path — analyze a file by its on-disk path.
///
/// Accepts `{"path": "/full/path/to/file"}`. The path must be under one of
/// the directories specified by `--allowed-dirs`. Returns the same
/// `{"ml": {...}, "raw": {...}}` envelope as `/analyze`.
pub(super) async fn analyze_path(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnalyzePathRequest>,
) -> Response {
    let request_id = state.next_request_id();
    let request_start = Instant::now();

    let path = std::path::PathBuf::from(&req.path);

    // Validate the path is under an allowed directory.
    if state.allowed_dirs.is_empty() || !state.allowed_dirs.iter().any(|dir| path.starts_with(dir))
    {
        tracing::warn!(id = request_id, path = %req.path, "analyze-path rejected: not under allowed dirs");
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "Path not under allowed directories"})),
        )
            .into_response();
    }

    if !path.is_file() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "File not found"})),
        )
            .into_response();
    }

    // Check memory pressure.
    if let Some(resp) = check_memory_pressure(&state) {
        return resp;
    }

    // Ensure resources are loaded.
    let resources = {
        let guard = match state.resources.read() {
            Ok(g) => g,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "Internal error"})),
                )
                    .into_response();
            }
        };
        match guard.as_ref() {
            Some(r) => Arc::clone(r),
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({"error": "Server starting up"})),
                )
                    .into_response();
            }
        }
    };

    let file_size = path.metadata().map(|m| m.len()).unwrap_or(0);
    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    tracing::info!(
        id = request_id,
        path = %req.path,
        size_bytes = file_size,
        "--> POST /analyze-path",
    );

    // Hard gate: reject immediately if at capacity (including orphaned tasks).
    let active = state
        .active_tasks
        .load(std::sync::atomic::Ordering::Relaxed);
    if active >= state.max_concurrent_tasks {
        tracing::warn!(
            id = request_id,
            filename = %filename,
            size_bytes = file_size,
            active_tasks = active,
            max = state.max_concurrent_tasks,
            "rejecting: at capacity"
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Server overloaded (too many active analyses)"})),
        )
            .into_response();
    }

    let slow_rule_ms = state.slow_rule_ms;
    let timeout_duration = Duration::from_secs(state.timeout_secs);

    // Claim a slot using the RAII guard.
    let guard = TaskGuard::new(Arc::clone(&state), request_id, req.path.clone(), file_size);

    let should_clear_caches = request_id.is_multiple_of(50);
    let extract_dir = state.extract_dir.clone();
    let cancellation = Arc::new(AtomicBool::new(false));
    let cancel_flag = Arc::clone(&cancellation);
    let mut handle = tokio::task::spawn_blocking(move || {
        let _moved_guard = guard;
        let result = classify_file(
            &path,
            &filename,
            &resources,
            slow_rule_ms,
            extract_dir.as_deref(),
            Some(cancel_flag),
        );
        if should_clear_caches {
            cleave::clear_all_thread_caches();
        }
        result
    });

    let result = tokio::select! {
        res = &mut handle => Some(res),
        _ = tokio::time::sleep(timeout_duration) => {
            cancellation.store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(req) = state.in_flight.get(&request_id) {
                req.timed_out.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            None
        }
    };

    let elapsed_ms = request_start.elapsed().as_millis();

    match result {
        Some(Ok(Ok(mut scan_result))) => {
            // Inject extracted_path into the raw cleave JSON so cyclotron
            // knows where archive members were extracted on disk.
            if let (Some(ref extract_dir), Some(ref mut raw)) =
                (&state.extract_dir, &mut scan_result.cleave)
            {
                if let Some(fs) = raw.get("fs").and_then(|f| f.as_array()) {
                    if let Some(first) = fs
                        .first()
                        .and_then(|f| f.get("sha"))
                        .and_then(|s| s.as_str())
                    {
                        let short = &first[..first.len().min(6)];
                        let dir = extract_dir.join(short);
                        if dir.is_dir() {
                            raw.as_object_mut().map(|o| {
                                o.insert(
                                    "extracted_path".to_string(),
                                    serde_json::Value::String(dir.to_string_lossy().into_owned()),
                                )
                            });
                        }
                    }
                }
            }

            tracing::info!(
                id = request_id,
                elapsed_ms = elapsed_ms as u64,
                classification = %scan_result.classification,
                probability = scan_result.probability,
                "<-- 200 OK",
            );
            let mut resp = Json(scan_result.to_envelope()).into_response();
            resp.headers_mut()
                .insert("X-Total-Ms", (elapsed_ms as u64).into());
            resp
        }
        Some(Ok(Err(e))) => {
            let status = analysis_error_status(&e);
            tracing::warn!(id = request_id, elapsed_ms = elapsed_ms as u64, status = status.as_u16(), error = %e, "analysis failed");
            analysis_error_response(&e)
        }
        Some(Err(e)) => {
            tracing::warn!(id = request_id, elapsed_ms = elapsed_ms as u64, error = %e, "<-- 500 task join error (panic?)");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response()
        }
        None => {
            tracing::warn!(
                id = request_id,
                elapsed_ms = elapsed_ms as u64,
                "<-- 504 timeout after {}s",
                state.timeout_secs
            );
            (
                StatusCode::GATEWAY_TIMEOUT,
                Json(serde_json::json!({"error": "Analysis timed out"})),
            )
                .into_response()
        }
    }
}

/// Check RSS memory pressure and attempt recovery before rejecting requests.
///
/// Returns `Some(Response)` if the request should be rejected due to memory pressure,
/// or `None` if the server has enough memory to proceed.
fn check_memory_pressure(state: &AppState) -> Option<Response> {
    let rss = cleave::memory_tracker::current_rss()?;

    if rss <= state.max_rss_bytes {
        // Happy path: reset overload timer if set.
        if let Ok(mut overloaded) = state.overloaded_since.try_lock() {
            if overloaded.is_some() {
                tracing::info!(
                    rss_mb = rss / 1024 / 1024,
                    "memory recovered below threshold"
                );
                *overloaded = None;
            }
        }
        return None;
    }

    // Memory pressure detected — try to reclaim by clearing thread-local caches.
    tracing::info!(
        rss_mb = rss / 1024 / 1024,
        "memory pressure detected, clearing thread-local caches"
    );
    tokio::task::block_in_place(cleave::clear_all_thread_caches);

    // Re-check after clearing caches.
    let rss_after = cleave::memory_tracker::current_rss()?;
    if rss_after <= state.max_rss_bytes {
        if let Ok(mut overloaded) = state.overloaded_since.try_lock() {
            *overloaded = None;
        }
        tracing::info!(
            rss_before_mb = rss / 1024 / 1024,
            rss_after_mb = rss_after / 1024 / 1024,
            "cache clear freed memory, accepting request"
        );
        return None;
    }

    // Still overloaded — track duration and potentially terminate.
    let mut overloaded = state.overloaded_since.lock().ok()?;
    let since = *overloaded.get_or_insert_with(Instant::now);
    let overloaded_secs = since.elapsed().as_secs();

    if overloaded_secs > 30 {
        tracing::error!(
            rss_mb = rss_after / 1024 / 1024,
            overloaded_secs,
            "memory overload persisted >30s after cache clears, terminating"
        );
        std::process::exit(1);
    }

    tracing::warn!(
        rss_mb = rss_after / 1024 / 1024,
        max_rss_mb = state.max_rss_bytes / 1024 / 1024,
        overloaded_secs,
        "server overloaded: high memory usage (even after cache clear)"
    );
    Some(
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Server overloaded (memory)"})),
        )
            .into_response(),
    )
}

#[cfg(test)]
mod tests {
    use super::classify_analysis_error;
    use axum::http::StatusCode;

    #[test]
    fn classify_unsupported_file_type_as_415() {
        let (status, message) = classify_analysis_error("Unsupported file type: Unknown");
        assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_eq!(message, "Unsupported file type: Unknown");
    }

    #[test]
    fn classify_invalid_archive_as_422() {
        let (status, message) =
            classify_analysis_error("Archive is encrypted but no passwords configured");
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(message, "Archive is encrypted but no passwords configured");
    }

    #[test]
    fn classify_unexpected_failure_as_500() {
        let (status, _) = classify_analysis_error("model evaluation failed");
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }
}

/// GET /_/memory — memory diagnostics for all major structures.
///
/// `process.jemalloc` is null unless cleave was built with `--features jemalloc`.
/// When available, `jemalloc.allocated_mb` is the most useful leak indicator:
/// if it tracks RSS closely, you have a real leak; if RSS >> allocated, it's fragmentation.
pub(super) async fn memory_stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let rss_mb = cleave::memory_tracker::current_rss().map(|b| b / 1024 / 1024);

    let jemalloc = cleave::memory_tracker::jemalloc_stats().map(|s| {
        serde_json::json!({
            "allocated_mb":     s.allocated / 1024 / 1024,
            "active_mb":        s.active    / 1024 / 1024,
            "metadata_mb":      s.metadata  / 1024 / 1024,
            "resident_mb":      s.resident  / 1024 / 1024,
            "retained_mb":      s.retained  / 1024 / 1024,
            "fragmentation_mb": s.active.saturating_sub(s.allocated) / 1024 / 1024,
        })
    });

    Json(serde_json::json!({
        "process": {
            "rss_mb": rss_mb,
            "max_rss_mb": state.max_rss_bytes / 1024 / 1024,
            "jemalloc": jemalloc,
        },
        "server": {
            "active_tasks": state.active_tasks.load(Ordering::Relaxed),
            "max_concurrent_tasks": state.max_concurrent_tasks,
            "requests_total": state.next_request_id.load(Ordering::Relaxed),
            "timeout_secs": state.timeout_secs,
        },
        "thread_pools": {
            "rayon_threads": rayon::current_num_threads(),
        },
    }))
}

/// GET /_/requests — all analyses currently in flight, sorted by elapsed time descending.
pub(super) async fn requests(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let now = Instant::now();
    let mut entries: Vec<serde_json::Value> = state
        .in_flight
        .iter()
        .map(|e| {
            serde_json::json!({
                "request_id": e.key(),
                "name": e.name,
                "size_bytes": e.size_bytes,
                "elapsed_ms": now.duration_since(e.started_at).as_millis(),
                "timed_out": e.timed_out.load(std::sync::atomic::Ordering::Relaxed),
            })
        })
        .collect();

    entries.sort_by(|a, b| b["elapsed_ms"].as_u64().cmp(&a["elapsed_ms"].as_u64()));

    Json(serde_json::json!({
        "count": entries.len(),
        "requests": entries,
    }))
}

/// GET /_/threads — OS-level thread info for every thread in this process.
///
/// On Linux: thread name, state, and `wchan` (kernel function blocked in).
/// `wchan` values to watch for: `futex_wait*` = mutex deadlock, `do_epoll_wait` = healthy async.
/// On FreeBSD: equivalent via sysctl + kinfo_proc (`ki_wmesg` instead of wchan).
pub(super) async fn threads() -> Json<serde_json::Value> {
    let info = tokio::task::spawn_blocking(read_thread_info).await;
    let info = info.unwrap_or_else(|_| serde_json::json!({"error": "failed to read thread info"}));
    Json(info)
}

fn read_thread_info() -> serde_json::Value {
    #[cfg(target_os = "linux")]
    return read_thread_info_linux();

    #[cfg(target_os = "freebsd")]
    return read_thread_info_freebsd();

    #[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
    serde_json::json!({
        "note": "detailed thread info only available on Linux and FreeBSD",
        "rayon_threads": rayon::current_num_threads(),
    })
}

#[cfg(target_os = "linux")]
fn read_thread_info_linux() -> serde_json::Value {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return serde_json::json!({"error": "cannot read /proc/self/task"});
    };

    let mut threads: Vec<serde_json::Value> = tasks
        .flatten()
        .filter_map(|entry| {
            let base = entry.path();
            let tid: u32 = entry.file_name().to_string_lossy().parse().ok()?;

            let name = std::fs::read_to_string(base.join("comm"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();

            let wchan = std::fs::read_to_string(base.join("wchan"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();

            let mut state_str = String::new();
            let mut vol_switches: u64 = 0;
            let mut nonvol_switches: u64 = 0;
            if let Ok(status) = std::fs::read_to_string(base.join("status")) {
                for line in status.lines() {
                    if let Some(val) = line.strip_prefix("State:\t") {
                        state_str = val.to_string();
                    } else if let Some(val) = line.strip_prefix("voluntary_ctxt_switches:\t") {
                        vol_switches = val.trim().parse().unwrap_or(0);
                    } else if let Some(val) = line.strip_prefix("nonvoluntary_ctxt_switches:\t") {
                        nonvol_switches = val.trim().parse().unwrap_or(0);
                    }
                }
            }

            Some(serde_json::json!({
                "tid": tid,
                "name": name,
                "state": state_str,
                "wchan": wchan,
                "voluntary_context_switches": vol_switches,
                "nonvoluntary_context_switches": nonvol_switches,
            }))
        })
        .collect();

    threads.sort_by_key(|t| t["tid"].as_u64().unwrap_or(0));
    serde_json::json!({"count": threads.len(), "threads": threads})
}

#[cfg(target_os = "freebsd")]
fn read_thread_info_freebsd() -> serde_json::Value {
    use std::mem;

    let pid = unsafe { libc::getpid() };
    let mib: [libc::c_int; 4] = [
        libc::CTL_KERN,
        libc::KERN_PROC,
        libc::KERN_PROC_PID | libc::KERN_PROC_INC_THREAD,
        pid,
    ];

    let mut len: libc::size_t = 0;
    let ret = unsafe {
        libc::sysctl(
            mib.as_ptr(),
            4,
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return serde_json::json!({"error": "sysctl size query failed"});
    }

    len += len / 4; // 25% slack for new threads between calls
    let count = len / mem::size_of::<libc::kinfo_proc>();
    let mut procs: Vec<libc::kinfo_proc> = (0..count).map(|_| unsafe { mem::zeroed() }).collect();
    let mut actual_len = len;

    let ret = unsafe {
        libc::sysctl(
            mib.as_ptr(),
            4,
            procs.as_mut_ptr().cast(),
            &mut actual_len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return serde_json::json!({"error": "sysctl data query failed"});
    }

    procs.truncate(actual_len / mem::size_of::<libc::kinfo_proc>());

    let c_str = |buf: &[libc::c_char]| {
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    let state_str = |s: libc::c_char| match s as u8 {
        1 => "idle",
        2 => "running",
        3 => "sleeping",
        4 => "stopped",
        5 => "zombie",
        6 => "waiting",
        7 => "locked",
        _ => "unknown",
    };

    let mut threads: Vec<serde_json::Value> = procs
        .iter()
        .map(|p| {
            serde_json::json!({
                "tid": p.ki_tid,
                "name": c_str(&p.ki_tdname),
                "state": state_str(p.ki_stat),
                "wchan": c_str(&p.ki_wmesg),
            })
        })
        .collect();

    threads.sort_by_key(|t| t["tid"].as_u64().unwrap_or(0));
    serde_json::json!({"count": threads.len(), "threads": threads})
}
