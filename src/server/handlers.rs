//! HTTP request handlers for the litmus API server.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tempfile::Builder as TempBuilder;

use super::AppState;

/// Outcome of awaiting a `tokio::spawn_blocking` analysis task with a bound.
///
/// `Ok` boxes the `ScanResult` (≈376 B) so the idle-path variants — `Timeout`
/// and `JoinError` — don't carry that much padding each.
#[derive(Debug)]
enum AnalysisOutcome {
    /// Task completed (inner `Result` is the analyzer's result).
    Ok(anyhow::Result<Box<crate::scan::ScanResult>>),
    /// Task join failed (panic, runtime shutdown, etc.).
    JoinError(tokio::task::JoinError),
    /// Task exceeded the configured timeout. Slot is released; the blocking
    /// thread keeps running until cleave observes the cancellation flag.
    Timeout(u64),
}

/// Await a blocking analysis task with an optional per-request timeout.
///
/// On timeout, sets the cancellation flag (so cleave will exit at its next
/// checkpoint), increments `stuck_orphans` for observability, and returns
/// `Timeout`. The blocking task is **not** aborted — tokio can't force-stop
/// a blocking thread — but the HTTP slot is released so new work can land.
async fn await_with_timeout(
    handle: tokio::task::JoinHandle<anyhow::Result<crate::scan::ScanResult>>,
    timeout_secs: u64,
    cancellation: &AtomicBool,
    stuck_orphans: &AtomicUsize,
) -> AnalysisOutcome {
    if timeout_secs == 0 {
        return match handle.await {
            Ok(r) => AnalysisOutcome::Ok(r.map(Box::new)),
            Err(e) => AnalysisOutcome::JoinError(e),
        };
    }
    match tokio::time::timeout(Duration::from_secs(timeout_secs), handle).await {
        Ok(Ok(r)) => AnalysisOutcome::Ok(r.map(Box::new)),
        Ok(Err(e)) => AnalysisOutcome::JoinError(e),
        Err(_) => {
            cancellation.store(true, Ordering::Release);
            stuck_orphans.fetch_add(1, Ordering::Relaxed);
            AnalysisOutcome::Timeout(timeout_secs)
        }
    }
}

/// Return a platform-appropriate thread ID for the calling thread.
/// On Linux this is the TID (matches /proc/self/task/), on macOS/FreeBSD
/// it's the pthread ID, and elsewhere falls back to 0.
fn current_thread_id() -> u64 {
    #[cfg(target_os = "linux")]
    {
        unsafe { libc::syscall(libc::SYS_gettid) as u64 }
    }
    #[cfg(target_os = "macos")]
    {
        let mut tid: u64 = 0;
        // pthread_threadid_np gives a stable, unique-per-process thread ID on macOS.
        unsafe { libc::pthread_threadid_np(0, &mut tid) };
        tid
    }
    #[cfg(target_os = "freebsd")]
    {
        // thr_self writes the lwpid into the provided pointer.
        let mut tid: libc::c_long = 0;
        unsafe { libc::thr_self(&mut tid) };
        tid as u64
    }
    #[cfg(target_os = "openbsd")]
    {
        unsafe { libc::getthrid() as u64 }
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "openbsd"
    )))]
    {
        0
    }
}

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
        || normalized.contains("file name too long")
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

/// Returns the 1-minute system load average, or None on unsupported platforms.
fn system_load_avg() -> Option<f64> {
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "openbsd"
    ))]
    {
        let mut avg: [libc::c_double; 1] = [0.0];
        let ret = unsafe { libc::getloadavg(avg.as_mut_ptr(), 1) };
        if ret == 1 { Some(avg[0]) } else { None }
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "openbsd"
    )))]
    {
        None
    }
}

/// GET /_/health — liveness check with memory and concurrency status.
/// Returns 503 while resources are still loading or when RSS exceeds the
/// configured limit. A fully-utilised worker pool returns 200 with
/// `status: "saturated"` — that's the target steady state, not a fault.
///
/// Every response carries `uptime_secs` (seconds since the server started)
/// so clients can detect restarts without polling a separate endpoint.
pub(super) async fn health(State(state): State<Arc<AppState>>) -> Response {
    let uptime_secs = state.started_at.elapsed().as_secs();
    let load_avg = system_load_avg();

    if let Ok(init_error) = state.init_error.read()
        && let Some(message) = init_error.as_ref()
    {
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
    let max_rss_mb = state.max_rss_bytes.map(|n| n.get() / 1024 / 1024);
    let active_tasks = state
        .max_concurrent_tasks
        .saturating_sub(state.slots.available_permits());
    let overloaded = match (rss_bytes, state.max_rss_bytes) {
        (Some(rss), Some(limit)) => rss > limit.get(),
        _ => false,
    };

    if overloaded {
        tracing::warn!("GET /_/health -> 503 (degraded, rss={rss_mb:?}MB)");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "degraded",
                "reason": "memory_pressure",
                "rss_mb": rss_mb,
                "max_rss_mb": max_rss_mb,
                "active_tasks": active_tasks,
                "load_avg": load_avg,
                "uptime_secs": uptime_secs,
                "rayon_threads": rayon::current_num_threads(),
            })),
        )
            .into_response();
    }
    let max_tasks = state.max_concurrent_tasks;
    let stuck_orphans = state
        .stuck_orphans
        .load(std::sync::atomic::Ordering::Relaxed);

    // Tasks running longer than 120s — visible in /_/requests with full phase detail.
    let now = Instant::now();
    let long_running: Vec<serde_json::Value> = state
        .in_flight
        .iter()
        .filter_map(|e| {
            let elapsed_secs = now.duration_since(e.started_at).as_secs();
            if elapsed_secs >= 120 {
                Some(serde_json::json!({
                    "request_id": e.key(),
                    "name": e.name,
                    "elapsed_secs": elapsed_secs,
                    "phase": e.phase.get(),
                    "thread_id": e.thread_id.load(std::sync::atomic::Ordering::Relaxed),
                }))
            } else {
                None
            }
        })
        .collect();

    let load = if max_tasks > 0 {
        active_tasks as f64 / max_tasks as f64
    } else {
        0.0
    };
    // A fully-utilised worker pool is the *target* steady state, not a fault.
    // Report it as "saturated" with HTTP 200 so monitors can distinguish "all
    // slots busy" from real failures (memory pressure, stuck workers). The
    // /analyze endpoint still rejects with 503 when active >= max, so clients
    // back off correctly without /_/health pretending the server is unhealthy.
    if active_tasks >= max_tasks {
        let oldest = state
            .in_flight
            .iter()
            .min_by_key(|e| e.started_at)
            .map(|e| (e.name.clone(), e.started_at.elapsed().as_secs()));
        tracing::debug!(
            active_tasks,
            stuck_orphans,
            long_running = long_running.len(),
            max_concurrent_tasks = max_tasks,
            oldest_task = ?oldest,
            "GET /_/health -> 200 (saturated)"
        );
        return Json(serde_json::json!({
            "status": "saturated",
            "reason": "thread_pool_saturated",
            "rss_mb": rss_mb,
            "active_tasks": active_tasks,
            "stuck_orphans": stuck_orphans,
            "long_running_tasks": long_running,
            "max_concurrent_tasks": max_tasks,
            "oldest_task": oldest.map(|(name, secs)| serde_json::json!({"name": name, "elapsed_secs": secs})),
            "load": load,
            "load_avg": load_avg,
            "uptime_secs": uptime_secs,
            "rayon_threads": rayon::current_num_threads(),
        }))
            .into_response();
    }
    tracing::debug!(
        "GET /_/health -> 200 (rss={rss_mb:?}MB, active={active_tasks}, long_running={}, stuck_orphans={stuck_orphans}, load={load:.2})",
        long_running.len()
    );
    Json(serde_json::json!({
        "status": "ok",
        "rss_mb": rss_mb,
        "active_tasks": active_tasks,
        "stuck_orphans": stuck_orphans,
        "long_running_tasks": long_running,
        "max_concurrent_tasks": max_tasks,
        "load": load,
        "load_avg": load_avg,
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
    let total_mem_mb = cleave::memory_tracker::total_memory()
        .map(|bytes| bytes / 1024 / 1024)
        .unwrap_or(0);
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "slots": state.max_concurrent_tasks,
        "cpus": cpus,
        "max_upload_mb": state.max_upload_bytes / 1024 / 1024,
        "max_rss_mb": state.max_rss_bytes.map(|n| n.get() / 1024 / 1024),
        "total_mem_mb": total_mem_mb,
        "model_commit": crate::models_repo::version(),
        "traits_commit": cleave::traits_repo::version(),
    }))
    .into_response()
}

/// Outcome of [`do_model_reload`] — caller maps this to an HTTP response.
struct ReloadOutcome {
    elapsed_ms: u128,
    /// If trait reload failed, the error message. `None` means traits loaded OK.
    traits_reload_error: Option<String>,
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
    let level = state.level;

    const RELOAD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
    let result = tokio::time::timeout(
        RELOAD_TIMEOUT,
        tokio::task::spawn_blocking(move || {
            // Reload cleave traits first so the new model runs against fresh rules.
            let traits_reload_error = match cleave::reload_capability_mapper() {
                Err(e) => {
                    tracing::warn!("cleave trait reload failed (previous traits retained): {e}");
                    Some(e)
                }
                Ok(_) => {
                    tracing::info!("cleave traits reloaded");
                    None
                }
            };
            cleave::clear_all_thread_caches();

            let model = Model::load(&model_dir, thresholds, level)?;
            let shap = ShapImportance::load(&model_dir).ok();
            let ctx = ExtractContext::new(model.spec());
            Ok::<_, anyhow::Error>((model, shap, ctx, traits_reload_error))
        }),
    )
    .await;

    let elapsed_ms = start.elapsed().as_millis();

    let (model, shap, ctx, traits_reload_error) = match result {
        Ok(Ok(Ok(t))) => t,
        Ok(Ok(Err(e))) => {
            // Log internally but do not expose filesystem paths or model internals to callers.
            tracing::warn!("reload failed (previous model retained) in {elapsed_ms}ms: {e}");
            return Err((StatusCode::UNPROCESSABLE_ENTITY, "Failed to load model"));
        }
        Ok(Err(e)) => {
            tracing::warn!("reload task join error: {e}");
            return Err((StatusCode::INTERNAL_SERVER_ERROR, "Internal error"));
        }
        Err(_elapsed) => {
            tracing::warn!(
                "reload timed out after {}s (previous model retained)",
                RELOAD_TIMEOUT.as_secs()
            );
            return Err((StatusCode::GATEWAY_TIMEOUT, "Reload timed out"));
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
            Ok(ReloadOutcome {
                elapsed_ms,
                traits_reload_error,
            })
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
        Ok(outcome) => {
            let mut body = serde_json::json!({
                "status": "ok",
                "elapsed_ms": outcome.elapsed_ms,
            });
            if let Some(err) = &outcome.traits_reload_error {
                body["traits_reload_error"] = serde_json::json!(err);
            }
            Json(body).into_response()
        }
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
    //
    // The outer timeout is only a backstop around the sequential models +
    // traits updates. model_update validates each bundle (Model::load) before
    // swapping it in, so a broken bundle never lands on disk and there's nothing
    // to roll back.
    const PULL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(21 * 60);
    let pull_result = tokio::time::timeout(
        PULL_TIMEOUT,
        tokio::task::spawn_blocking(|| {
            let dir = crate::models_repo::install_target();
            let models_err = match crate::model_update::update(&dir, false) {
                Ok(()) => None,
                Err(e) => {
                    tracing::warn!("models update failed: {e}");
                    Some(e.to_string())
                }
            };
            let traits_err = match crate::traits_repo::update(false, false) {
                Ok(_) => None,
                Err(e) => {
                    tracing::warn!("traits update failed: {e}");
                    Some(e.to_string())
                }
            };
            (models_err, traits_err)
        }),
    )
    .await;

    let (models_err, traits_err) = match pull_result {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            tracing::warn!("update pull task join error: {e}");
            let msg = "task join failed".to_string();
            (Some(msg.clone()), Some(msg))
        }
        Err(_elapsed) => {
            tracing::warn!("update pull timed out after {}s", PULL_TIMEOUT.as_secs());
            let msg = format!("update timed out after {}s", PULL_TIMEOUT.as_secs());
            (Some(msg.clone()), Some(msg))
        }
    };

    match do_model_reload(&state).await {
        Ok(outcome) => {
            let mut body = serde_json::json!({
                "status": "ok",
                "elapsed_ms": outcome.elapsed_ms,
                "models_updated": models_err.is_none(),
                "traits_updated": traits_err.is_none(),
                "models_error": models_err,
                "traits_error": traits_err,
                "version": env!("CARGO_PKG_VERSION"),
                "model_commit": crate::models_repo::version(),
                "traits_commit": cleave::traits_repo::version(),
            });
            if let Some(err) = &outcome.traits_reload_error {
                body["traits_reload_error"] = serde_json::json!(err);
            }
            Json(body).into_response()
        }
        Err((status, msg)) => {
            // The in-memory model was not swapped, so requests continue against
            // the previous model. The on-disk bundle was validated before install,
            // so there's nothing to roll back — a reload failure here is in-memory.
            tracing::error!("model reload failed after update: {msg}");
            (
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
                .into_response()
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

    if let Ok(init_error) = state.init_error.read()
        && let Some(message) = init_error.as_ref()
    {
        tracing::error!("analyze rejected: startup failed  id={request_id} error={message}");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Server failed to initialize"})),
        )
            .into_response();
    }

    if let Some(response) = check_memory_pressure(&state).await {
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
            // All remaining chars are ASCII (filter above) so byte indexing is safe.
            #[allow(clippy::string_slice)]
            let s = sanitized[sanitized.len() - 63..].to_string();
            s
        } else {
            sanitized
        }
    };

    // Create a temp directory containing a file with the original filename so that
    // cleave's filename-based type detection works correctly (e.g. "package.json"
    // is recognized as PackageJson, not Unknown).
    let fname_for_temp = filename.clone();
    let temp_dir =
        match tokio::task::spawn_blocking(move || TempBuilder::new().prefix("litmus-").tempdir())
            .await
        {
            Ok(Ok(d)) => d,
            Ok(Err(e)) => {
                tracing::warn!("failed to create temp dir: {e}  id={request_id}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "Internal error"})),
                )
                    .into_response();
            }
            Err(e) => {
                tracing::warn!("temp dir task join error (panic?): {e}  id={request_id}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "Internal error"})),
                )
                    .into_response();
            }
        };

    // Stream multipart field to a file with the original name inside the temp dir.
    let path = temp_dir.path().join(&fname_for_temp);
    let writer = match std::fs::File::create(&path) {
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
        upload_ms = crate::duration_ms(request_start.elapsed()),
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

    // Claim a slot. OwnedSemaphorePermit is RAII: the slot is released when the
    // permit is dropped, even on panic or runtime shutdown — no manual fetch_sub needed.
    let Ok(permit) = Arc::clone(&state.slots).try_acquire_owned() else {
        let max = state.max_concurrent_tasks;
        tracing::warn!(
            id = request_id,
            filename = %filename,
            size_bytes = file_size,
            max,
            "rejecting: at capacity"
        );
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(
                serde_json::json!({"error": format!("At capacity ({max}/{max} active analyses)")}),
            ),
        )
            .into_response();
    };

    let slow_rule_ms = state.slow_rule_ms;

    let filename_for_closure = filename.clone();
    let should_clear_caches = request_id.is_multiple_of(100);
    let cancellation = Arc::new(AtomicBool::new(false));
    state.in_flight.insert(
        request_id,
        super::InFlightRequest {
            name: filename.clone(),
            size_bytes: file_size as u64,
            started_at: Instant::now(),
            cancellation: Arc::clone(&cancellation),
            phase: cleave::PhaseTracker::with_label(format!("req#{request_id} {filename}")),
            thread_id: AtomicU64::new(0),
        },
    );

    // RAII guard: if the client disconnects and axum drops this future, the guard's
    // Drop impl signals cancellation to the blocking thread and removes the in-flight
    // entry, so no slot or dashmap entry leaks.
    let guard = super::RequestGuard::new(
        request_id,
        Arc::clone(&state),
        Arc::clone(&cancellation),
        permit,
    );

    // Save the temp dir path so we can clean it up after the blocking task finishes.
    let temp_dir_path = temp_dir.path().to_path_buf();

    let cancel_flag = Arc::clone(&cancellation);
    let phase_state = Arc::clone(&state);
    let phase_tracker = phase_state
        .in_flight
        .get(&request_id)
        .map(|r| r.phase.clone());
    let handle = tokio::task::spawn_blocking(move || {
        // Record the OS thread servicing this request.
        if let Some(req) = phase_state.in_flight.get(&request_id) {
            req.thread_id.store(current_thread_id(), Ordering::Relaxed);
        }
        let result = classify_file(
            &path,
            &filename_for_closure,
            &resources,
            slow_rule_ms,
            None,
            Some(&cancel_flag),
            phase_tracker.as_ref(),
        );
        if should_clear_caches {
            cleave::clear_all_thread_caches();
        }
        drop(temp_dir);
        result
    });

    // Await to completion, bounded by the configured per-request timeout. If
    // the client disconnects, axum drops this future and guard.drop() fires,
    // signalling cancellation and releasing the slot. On timeout we signal
    // cancellation and return 504 — the blocking thread continues until
    // cleave notices the flag, but the slot is freed and `stuck_orphans` is
    // incremented so an operator can see zombie work.
    let result = await_with_timeout(
        handle,
        state.analysis_timeout_secs,
        &cancellation,
        &state.stuck_orphans,
    )
    .await;

    // Normal completion: drop the guard explicitly so its log context is clear.
    drop(guard);

    // Clean up temp directory (handles the case where drop(temp_dir) above didn't run).
    if let Err(e) = tokio::fs::remove_dir_all(&temp_dir_path).await {
        tracing::debug!(request_id, error = %e, "temp dir cleanup (may already be gone)");
    }

    let elapsed_ms = crate::duration_ms(request_start.elapsed());

    match result {
        AnalysisOutcome::Ok(Ok(scan_result)) => {
            let scan_result = *scan_result;
            tracing::info!(
                id = request_id,
                filename = %filename,
                size_bytes = file_size,
                elapsed_ms,
                classification = %scan_result.classification,
                probability = scan_result.probability,
                "<-- 200 OK",
            );
            let mut resp = Json(scan_result.into_envelope()).into_response();
            resp.headers_mut().insert("X-Total-Ms", elapsed_ms.into());
            resp
        }
        AnalysisOutcome::Ok(Err(e)) => {
            let status = analysis_error_status(&e);
            tracing::warn!(id = request_id, elapsed_ms, status = status.as_u16(), error = %e, "analysis failed");
            analysis_error_response(&e)
        }
        AnalysisOutcome::JoinError(e) => {
            tracing::warn!(id = request_id, elapsed_ms, error = %e, "<-- 500 task join error (panic?)");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response()
        }
        AnalysisOutcome::Timeout(secs) => {
            tracing::warn!(
                id = request_id,
                filename = %filename,
                elapsed_ms,
                timeout_secs = secs,
                "<-- 504 analysis timeout",
            );
            (
                StatusCode::GATEWAY_TIMEOUT,
                Json(serde_json::json!({
                    "error": "analysis timeout",
                    "timeout_secs": secs,
                })),
            )
                .into_response()
        }
    }
}

/// Run the full cleave + litmus pipeline on `path`, returning a `ScanResult`.
///
/// Runs on a blocking thread. `label` is used as the `path` field in the result
/// (the original upload filename, not the temp file path).
pub(crate) fn classify_file(
    path: &std::path::Path,
    label: &str,
    resources: &super::ModelResources,
    slow_rule_ms: u64,
    extract_dir: Option<&std::path::Path>,
    cancellation: Option<&Arc<AtomicBool>>,
    phase: Option<&cleave::PhaseTracker>,
) -> anyhow::Result<ScanResult> {
    use anyhow::Context as _;

    if let Some(p) = phase {
        p.set("cleave:init");
    }
    let sample_extraction =
        extract_dir.map(|d| cleave::SampleExtractionConfig::new(d.to_path_buf()));
    let opts = cleave::AnalysisOptions {
        slow_rule_ms,
        sample_extraction,
        cancellation: cancellation.cloned(),
        phase: phase.cloned(),
        ..Default::default()
    };
    let report =
        cleave::analyze_file(path, &opts).with_context(|| format!("cleave analysis of {label}"))?;

    // If the timeout fired while cleave was running, bail now rather than
    // burning CPU on feature extraction and model inference for a result
    // nobody is waiting for.
    if cancellation.is_some_and(|c| c.load(Ordering::Relaxed)) {
        anyhow::bail!("analysis cancelled");
    }

    if let Some(p) = phase {
        p.set("features+model");
    }
    let cr = crate::scan::classify_report(
        label,
        report,
        &resources.ctx,
        &resources.model,
        resources.shap.as_ref(),
        cancellation,
        Some(100),
    )?;

    Ok(scan_result_from(label, cr, resources))
}

/// Like [`classify_file`] but operates on in-memory data, avoiding disk I/O.
///
/// Takes ownership of `data` and hands it to `cleave::analyze_bytes_owned`,
/// which moves the buffer into the analysis pipeline instead of copying it.
/// At worker scale this eliminates one full-size memcpy per downloaded sample.
pub(crate) fn classify_bytes(
    data: Vec<u8>,
    label: &str,
    resources: &super::ModelResources,
    slow_rule_ms: u64,
    cancellation: Option<&Arc<AtomicBool>>,
    phase: Option<&cleave::PhaseTracker>,
) -> anyhow::Result<ScanResult> {
    use anyhow::Context as _;

    if let Some(p) = phase {
        p.set("cleave:init");
    }
    let opts = cleave::AnalysisOptions {
        slow_rule_ms,
        cancellation: cancellation.cloned(),
        phase: phase.cloned(),
        ..Default::default()
    };
    let report = cleave::analyze_bytes_owned(data, label, &opts)
        .with_context(|| format!("cleave analysis of {label}"))?;

    if cancellation.is_some_and(|c| c.load(Ordering::Relaxed)) {
        anyhow::bail!("analysis cancelled");
    }

    if let Some(p) = phase {
        p.set("features+model");
    }
    let cr = crate::scan::classify_report(
        label,
        report,
        &resources.ctx,
        &resources.model,
        resources.shap.as_ref(),
        cancellation,
        Some(100),
    )?;

    Ok(scan_result_from(label, cr, resources))
}

/// Build a [`ScanResult`] from a classified report.
fn scan_result_from(
    label: &str,
    cr: crate::scan::ClassifiedReport,
    resources: &super::ModelResources,
) -> ScanResult {
    ScanResult {
        v: "6",
        classification: cr.classification,
        probability: cr.probability,
        threshold: cr.threshold,
        l: cr.l,
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
        model_scores: cr.model_scores,
        skipped_models: cr.skipped_models,
        file_type: cr.file_type,
        size_bytes: cr.size_bytes,
        sha256: cr.sha256,
        embedded_files: cr.embedded_files,
    }
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

    let raw_path = std::path::PathBuf::from(&req.path);

    // Resolve symlinks and canonicalize BEFORE the allowed-dirs check to
    // prevent symlink-based path traversal (e.g., /allowed/link → /etc/shadow).
    let Ok(path) = raw_path.canonicalize() else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "File not found"})),
        )
            .into_response();
    };

    // Validate the canonical (symlink-resolved) path is under an allowed directory.
    if state.allowed_dirs.is_empty() || !state.allowed_dirs.iter().any(|dir| path.starts_with(dir))
    {
        tracing::warn!(id = request_id, path = %req.path, canonical = %path.display(), "analyze-path rejected: not under allowed dirs");
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
    if let Some(resp) = check_memory_pressure(&state).await {
        return resp;
    }

    // Ensure resources are loaded.
    let resources = {
        let Ok(guard) = state.resources.read() else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response();
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

    // Claim a slot — same RAII semaphore pattern as /analyze.
    let Ok(permit) = Arc::clone(&state.slots).try_acquire_owned() else {
        let max = state.max_concurrent_tasks;
        tracing::warn!(
            id = request_id,
            filename = %filename,
            size_bytes = file_size,
            max,
            "rejecting: at capacity"
        );
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(
                serde_json::json!({"error": format!("At capacity ({max}/{max} active analyses)")}),
            ),
        )
            .into_response();
    };

    let slow_rule_ms = state.slow_rule_ms;

    let should_clear_caches = request_id.is_multiple_of(100);
    let extract_dir = state.extract_dir.clone();
    let cancellation = Arc::new(AtomicBool::new(false));
    state.in_flight.insert(
        request_id,
        super::InFlightRequest {
            name: req.path.clone(),
            size_bytes: file_size,
            started_at: Instant::now(),
            cancellation: Arc::clone(&cancellation),
            phase: cleave::PhaseTracker::with_label(format!("req#{request_id} {}", req.path)),
            thread_id: AtomicU64::new(0),
        },
    );

    // RAII guard: signals cancellation and removes the in-flight entry on drop,
    // covering both normal completion and client-disconnect cancellation.
    let guard = super::RequestGuard::new(
        request_id,
        Arc::clone(&state),
        Arc::clone(&cancellation),
        permit,
    );

    let cancel_flag = Arc::clone(&cancellation);
    let phase_state = Arc::clone(&state);
    let phase_tracker = phase_state
        .in_flight
        .get(&request_id)
        .map(|r| r.phase.clone());
    let handle = tokio::task::spawn_blocking(move || {
        if let Some(req) = phase_state.in_flight.get(&request_id) {
            req.thread_id.store(current_thread_id(), Ordering::Relaxed);
        }
        let result = classify_file(
            &path,
            &filename,
            &resources,
            slow_rule_ms,
            extract_dir.as_deref(),
            Some(&cancel_flag),
            phase_tracker.as_ref(),
        );
        if should_clear_caches {
            cleave::clear_all_thread_caches();
        }
        result
    });

    // Await to completion, bounded by the configured per-request timeout.
    // See `analyze` for the timeout-drop-slot semantics.
    let result = await_with_timeout(
        handle,
        state.analysis_timeout_secs,
        &cancellation,
        &state.stuck_orphans,
    )
    .await;
    drop(guard);

    let elapsed_ms = crate::duration_ms(request_start.elapsed());

    match result {
        AnalysisOutcome::Ok(Ok(scan_result)) => {
            let mut scan_result = *scan_result;
            // Inject extracted_path into the raw cleave JSON so cyclotron
            // knows where archive members were extracted on disk.
            if let (Some(extract_dir), Some(raw)) = (&state.extract_dir, &mut scan_result.cleave)
                && let Some(fs) = raw.get("fs").and_then(|f| f.as_array())
                && let Some(first) = fs
                    .first()
                    .and_then(|f| f.get("sha"))
                    .and_then(|s| s.as_str())
            {
                // SHA hex is ASCII; byte slice is always a valid UTF-8 boundary.
                let short = first.get(..first.len().min(6)).unwrap_or(first);
                let dir = extract_dir.join(short);
                if dir.is_dir()
                    && let Some(o) = raw.as_object_mut()
                {
                    o.insert(
                        "extracted_path".to_string(),
                        serde_json::Value::String(dir.to_string_lossy().into_owned()),
                    );
                }
            }

            tracing::info!(
                id = request_id,
                elapsed_ms,
                classification = %scan_result.classification,
                probability = scan_result.probability,
                "<-- 200 OK",
            );
            let mut resp = Json(scan_result.into_envelope()).into_response();
            resp.headers_mut().insert("X-Total-Ms", elapsed_ms.into());
            resp
        }
        AnalysisOutcome::Ok(Err(e)) => {
            let status = analysis_error_status(&e);
            tracing::warn!(id = request_id, elapsed_ms, status = status.as_u16(), error = %e, "analysis failed");
            analysis_error_response(&e)
        }
        AnalysisOutcome::JoinError(e) => {
            tracing::warn!(id = request_id, elapsed_ms, error = %e, "<-- 500 task join error (panic?)");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response()
        }
        AnalysisOutcome::Timeout(secs) => {
            tracing::warn!(
                id = request_id,
                elapsed_ms,
                timeout_secs = secs,
                "<-- 504 analysis timeout",
            );
            (
                StatusCode::GATEWAY_TIMEOUT,
                Json(serde_json::json!({
                    "error": "analysis timeout",
                    "timeout_secs": secs,
                })),
            )
                .into_response()
        }
    }
}

/// Check RSS memory pressure and attempt recovery before rejecting requests.
///
/// Returns `Some(Response)` if the request should be rejected due to memory pressure,
/// or `None` if the server has enough memory to proceed.
async fn check_memory_pressure(state: &AppState) -> Option<Response> {
    // Throttling disabled: never reject on memory pressure. The operator has
    // delegated OOM enforcement to an external supervisor.
    let max_rss_bytes = state.max_rss_bytes?.get();
    let rss = cleave::memory_tracker::current_rss()?;

    if rss <= max_rss_bytes {
        // Happy path: reset overload timer if set.
        if let Ok(mut overloaded) = state.overloaded_since.try_lock()
            && overloaded.is_some()
        {
            tracing::info!(
                rss_mb = rss / 1024 / 1024,
                "memory recovered below threshold"
            );
            *overloaded = None;
        }
        return None;
    }

    // Memory pressure detected — try to reclaim by clearing thread-local caches.
    tracing::info!(
        rss_mb = rss / 1024 / 1024,
        "memory pressure detected, clearing thread-local caches"
    );
    // Await the clear before re-checking RSS. A prior version fired-and-forgot
    // via `drop(spawn_blocking(...))`, then immediately re-read RSS on the next
    // line — which produced a "cache clear freed memory" log before the clear
    // had actually run, and let overloaded workers admit requests they could
    // not service.
    if let Err(e) = tokio::task::spawn_blocking(cleave::clear_all_thread_caches).await {
        tracing::warn!(error = %e, "cache-clear task failed");
    }

    // Re-check after clearing caches.
    let rss_after = cleave::memory_tracker::current_rss()?;
    if rss_after <= max_rss_bytes {
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

    // Still overloaded — track duration and log. Never terminate; let the operator
    // decide when to restart. Requests continue to be rejected with 503 until
    // memory drops below the threshold.
    // Use try_lock: if a concurrent request holds the lock it is already recording
    // the overload timestamp, so it is safe to pass through rather than block a
    // tokio worker on a std::sync::Mutex.
    let since = *state
        .overloaded_since
        .try_lock()
        .ok()?
        .get_or_insert_with(Instant::now);
    let overloaded_secs = since.elapsed().as_secs();

    tracing::warn!(
        rss_mb = rss_after / 1024 / 1024,
        max_rss_mb = max_rss_bytes / 1024 / 1024,
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
            "max_rss_mb": state.max_rss_bytes.map(|n| n.get() / 1024 / 1024),
            "jemalloc": jemalloc,
        },
        "server": {
            "active_tasks": state.max_concurrent_tasks.saturating_sub(state.slots.available_permits()),
            "stuck_orphans": state.stuck_orphans.load(Ordering::Relaxed),
            "max_concurrent_tasks": state.max_concurrent_tasks,
            "requests_total": state.next_request_id.load(Ordering::Relaxed),
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
            let elapsed_ms = now.duration_since(e.started_at).as_millis();
            let phase = e.phase.get();
            let tid = e.thread_id.load(std::sync::atomic::Ordering::Relaxed);
            serde_json::json!({
                "request_id": e.key(),
                "name": e.name,
                "size_bytes": e.size_bytes,
                "elapsed_ms": elapsed_ms,
                "long_running": elapsed_ms >= 120_000,
                "phase": phase,
                "thread_id": if tid > 0 { Some(tid) } else { None },
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
