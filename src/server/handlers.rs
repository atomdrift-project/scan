//! HTTP request handlers for the litmus API server.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

use super::AppState;
use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Classification, Model, Thresholds};
use crate::scan::{
    count_findings_from_json, extract_top_findings_from_json, ScanResult,
    Thresholds as ScanThresholds,
};

/// GET /health
pub(super) async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// POST /reload — reload model from disk, swap atomically.
///
/// Only one reload may run at a time; concurrent calls receive 409.
pub(super) async fn reload(State(state): State<Arc<AppState>>) -> Response {
    // Prevent concurrent reloads — each load allocates significant memory.
    let _guard = match state.reload_lock.try_lock() {
        Ok(g) => g,
        Err(_) => {
            log::warn!("reload rejected: already in progress");
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({"error": "Reload already in progress"})),
            )
                .into_response();
        }
    };

    let start = Instant::now();
    let model_dir = state.model_dir.clone();
    let thresholds = Thresholds {
        suspicious: state.threshold_suspicious,
        hostile: state.threshold_hostile,
    };

    let result = tokio::task::spawn_blocking(move || {
        let model = Model::load(&model_dir, thresholds)?;
        let shap = ShapImportance::load(&model_dir).ok();
        let ctx = ExtractContext::new(&model.spec);
        Ok::<_, anyhow::Error>((model, shap, ctx))
    })
    .await;

    let elapsed_ms = start.elapsed().as_millis();

    match result {
        Ok(Ok((model, shap, ctx))) => {
            match state.resources.write() {
                Ok(mut lock) => {
                    *lock = Arc::new(super::ModelResources { model, shap, ctx });
                    log::info!("model reloaded in {elapsed_ms}ms");
                    Json(serde_json::json!({
                        "status": "ok",
                        "elapsed_ms": elapsed_ms,
                    }))
                    .into_response()
                }
                Err(e) => {
                    log::error!("write lock poisoned during reload: {e}");
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": "Internal error"})),
                    )
                        .into_response()
                }
            }
        }
        Ok(Err(e)) => {
            // Log internally but do not expose filesystem paths or model internals to callers.
            log::warn!("reload failed (previous model retained) in {elapsed_ms}ms: {e}");
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({
                    "error": "Failed to load model",
                    "elapsed_ms": elapsed_ms,
                })),
            )
                .into_response()
        }
        Err(e) => {
            log::warn!("reload task join error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
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

    log::info!("--> POST /analyze  id={request_id}");

    if let Some(response) = check_memory_pressure(&state) {
        return response;
    }

    // Parse the first multipart field as the file.
    let mut field = match multipart.next_field().await {
        Ok(Some(f)) => f,
        Ok(None) => {
            log::warn!("bad request: no file field  id={request_id}");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "No file field in request"})),
            )
                .into_response();
        }
        Err(e) => {
            log::warn!("failed to parse multipart: {e}  id={request_id}");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid multipart data"})),
            )
                .into_response();
        }
    };

    // Sanitize the uploaded filename: strip control characters, cap length.
    // Used only as the `path` label in the result — never interpreted as a filesystem path.
    let filename = field
        .file_name()
        .map(|s| s.chars().filter(|c| !c.is_control()).take(255).collect::<String>())
        .unwrap_or_else(|| format!("upload-{request_id}"));

    // Create temp file on the blocking thread pool.
    let temp_file = match tokio::task::spawn_blocking(NamedTempFile::new).await {
        Ok(Ok(f)) => f,
        Ok(Err(e)) => {
            log::warn!("failed to create temp file: {e}  id={request_id}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response();
        }
        Err(e) => {
            log::warn!("temp file task join error: {e}  id={request_id}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response();
        }
    };

    // Stream multipart field to the temp file.
    let path = temp_file.path().to_owned();
    let mut tokio_file = match tokio::fs::File::create(&path).await {
        Ok(f) => f,
        Err(e) => {
            log::warn!("failed to open temp file for writing: {e}  id={request_id}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response();
        }
    };

    let mut file_size = 0usize;
    loop {
        match field.chunk().await {
            Ok(Some(chunk)) if !chunk.is_empty() => {
                file_size += chunk.len();
                if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut tokio_file, &chunk).await
                {
                    log::warn!("failed to write chunk: {e}  id={request_id}");
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
                log::warn!("error reading multipart chunk: {e}  id={request_id}");
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": "Error reading upload data"})),
                )
                    .into_response();
            }
        }
    }

    if let Err(e) = tokio::io::AsyncWriteExt::flush(&mut tokio_file).await {
        log::warn!("failed to flush temp file: {e}  id={request_id}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Failed to save file data"})),
        )
            .into_response();
    }
    drop(tokio_file);

    if file_size == 0 {
        log::warn!("bad request: empty file  id={request_id}");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Empty file"})),
        )
            .into_response();
    }

    log::info!("starting analysis: filename={filename:?} size={file_size}  id={request_id}");

    // Snapshot the current model resources (Arc clone, no lock held during analysis).
    let resources = match state.resources.read() {
        Ok(lock) => Arc::clone(&*lock),
        Err(e) => {
            log::error!("read lock poisoned: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response();
        }
    };

    let timeout_duration = Duration::from_secs(state.timeout_secs);
    state.active_tasks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let slow_rule_ms = state.slow_rule_ms;
    let filename_for_closure = filename.clone();

    let mut handle = tokio::task::spawn_blocking(move || {
        classify_file(&path, &filename_for_closure, &resources, slow_rule_ms)
    });

    // Use timeout() rather than select! so a simultaneous completion + timeout
    // always prefers the completed result (no spurious 504s).
    let result = tokio::time::timeout(timeout_duration, &mut handle).await;

    let elapsed_ms = request_start.elapsed().as_millis();

    match result {
        Ok(join_result) => {
            state.active_tasks.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            drop(temp_file);

            match join_result {
                Ok(Ok(scan_result)) => {
                    log::info!(
                        "<-- 200 OK  id={request_id} filename={filename:?} size={file_size} \
                         elapsed_ms={elapsed_ms} classification={}",
                        scan_result.classification,
                    );
                    Json(scan_result).into_response()
                }
                Ok(Err(e)) => {
                    log::warn!(
                        "<-- 500 analysis failed  id={request_id} elapsed_ms={elapsed_ms}: {e}"
                    );
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": "Analysis failed"})),
                    )
                        .into_response()
                }
                Err(e) => {
                    log::warn!(
                        "<-- 500 task join error  id={request_id} elapsed_ms={elapsed_ms}: {e}"
                    );
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": "Internal error"})),
                    )
                        .into_response()
                }
            }
        }
        Err(_elapsed) => {
            // On timeout the blocking task keeps running; watch it to decrement the
            // counter and drop the temp file when it eventually finishes.
            let active = state.active_tasks.load(std::sync::atomic::Ordering::Relaxed);
            log::warn!(
                "analysis timed out after {}s  id={request_id} filename={filename:?} active={active}",
                state.timeout_secs,
            );
            let orphan_state = Arc::clone(&state);
            tokio::spawn(async move {
                let _ = handle.await;
                orphan_state
                    .active_tasks
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                drop(temp_file);
            });
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
) -> anyhow::Result<ScanResult> {
    use anyhow::Context as _;

    let opts = cleave::AnalysisOptions { slow_rule_ms, ..Default::default() };
    let mut report = cleave::analyze_file(path, &opts)
        .with_context(|| format!("cleave analysis of {label}"))?;

    let formula = cleave::formula_from_report(&report);
    report.finalize();

    let report_json = serde_json::to_value(&report).context("serializing cleave report")?;
    let mut features = resources.ctx.extract(&report_json);
    resources.model.spec.standardize(&mut features);
    let (probability, classification) = resources.model.predict(&features)?;

    let finding_counts = count_findings_from_json(&report_json);

    let (reasons, top_findings) = if classification != Classification::Benign {
        let r = resources
            .shap
            .as_ref()
            .map(|s| s.explain(&features, &resources.model.spec.feature_names, 5))
            .unwrap_or_default();
        let f = extract_top_findings_from_json(&report_json, &classification);
        (r, f)
    } else {
        (vec![], vec![])
    };

    let pf = report_json["files"]
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or(&report_json);
    let file_type = pf["file_type"].as_str().unwrap_or("unknown").to_string();
    let size_bytes = pf["size"].as_u64().unwrap_or(0);
    let sha256 = pf["sha256"].as_str().unwrap_or("").to_string();

    Ok(ScanResult {
        path: label.to_string(),
        classification,
        probability,
        thresholds: ScanThresholds {
            hostile: resources.model.thresholds.hostile,
            suspicious: resources.model.thresholds.suspicious,
        },
        finding_counts,
        formula,
        reasons,
        top_findings,
        file_type,
        size_bytes,
        sha256,
        cleave: Some(report_json),
        pids: None,
        deleted: None,
    })
}

/// Check RSS memory pressure. Returns a 503 response if the server is overloaded,
/// or `None` if memory is within limits. Linux-only; always returns `None` on other platforms.
fn check_memory_pressure(state: &AppState) -> Option<Response> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = state;
        return None;
    }

    #[cfg(target_os = "linux")]
    {
        let rss = current_rss_linux()?;

        if rss <= state.max_rss_bytes {
            // Happy path: reset overload timer if set.
            if let Ok(mut overloaded) = state.overloaded_since.try_lock() {
                if overloaded.is_some() {
                    log::info!("memory recovered: rss={}MB", rss / 1024 / 1024);
                    *overloaded = None;
                }
            }
            return None;
        }

        // Memory pressure: update overload timer.
        let mut overloaded = state.overloaded_since.lock().ok()?;
        let since = *overloaded.get_or_insert_with(Instant::now);
        let overloaded_secs = since.elapsed().as_secs();

        if overloaded_secs > 30 {
            log::error!(
                "memory overload persisted >30s (rss={}MB), terminating",
                rss / 1024 / 1024
            );
            std::process::exit(1);
        }

        log::warn!(
            "server overloaded: rss={}MB max={}MB overloaded_secs={overloaded_secs}",
            rss / 1024 / 1024,
            state.max_rss_bytes / 1024 / 1024,
        );
        Some(
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "Server overloaded (memory)"})),
            )
                .into_response(),
        )
    }
}

#[cfg(target_os = "linux")]
fn current_rss_linux() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // Format: "VmRSS:      123456 kB"
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}
