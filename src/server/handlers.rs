//! HTTP request handlers for the litmus API server.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tempfile::Builder as TempBuilder;

use super::AppState;
use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Model, Thresholds};
use crate::scan::{count_findings_from_json, extract_top_findings_from_json, ScanResult};

/// GET /_/health — liveness check with memory and concurrency status.
/// Returns 503 while resources are still loading or when RSS exceeds the configured limit.
pub(super) async fn health(State(state): State<Arc<AppState>>) -> Response {
    if !state.ready.load(std::sync::atomic::Ordering::Acquire) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "starting"})),
        )
            .into_response();
    }

    let rss_bytes = cleave::memory_tracker::current_rss();
    let rss_mb = rss_bytes.map(|b| b / 1024 / 1024);
    let active_tasks = state.active_tasks.load(std::sync::atomic::Ordering::Relaxed);
    let overloaded = rss_bytes
        .map(|b| b > state.max_rss_bytes)
        .unwrap_or(false);

    if overloaded {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "degraded",
                "reason": "memory_pressure",
                "rss_mb": rss_mb,
                "max_rss_mb": state.max_rss_bytes / 1024 / 1024,
                "active_tasks": active_tasks,
                "rayon_threads": rayon::current_num_threads(),
            })),
        )
            .into_response();
    }
    Json(serde_json::json!({
        "status": "ok",
        "rss_mb": rss_mb,
        "active_tasks": active_tasks,
        "rayon_threads": rayon::current_num_threads(),
    }))
    .into_response()
}

/// POST /reload — reload model from disk, swap atomically.
///
/// Only one reload may run at a time; concurrent calls receive 409.
pub(super) async fn reload(State(state): State<Arc<AppState>>) -> Response {
    // Prevent concurrent reloads — each load allocates significant memory.
    let Ok(_guard) = state.reload_lock.try_lock() else {
        tracing::warn!("reload rejected: already in progress");
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "Reload already in progress"})),
        )
            .into_response();
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
            let spec_version = model.spec.version;
            let features = model.spec.total_features;
            let shap_loaded = shap.is_some();
            let was_ready = state.ready.load(std::sync::atomic::Ordering::Relaxed);
            match state.resources.write() {
                Ok(mut lock) => {
                    *lock = Some(Arc::new(super::ModelResources { model, shap, ctx }));
                    state.ready.store(true, std::sync::atomic::Ordering::Release);
                    if was_ready {
                        tracing::info!(elapsed_ms, spec_version, features, shap_loaded, "model reloaded");
                    } else {
                        tracing::info!(elapsed_ms, spec_version, features, shap_loaded, "model loaded via reload — server now ready");
                    }
                    Json(serde_json::json!({
                        "status": "ok",
                        "elapsed_ms": elapsed_ms,
                    }))
                    .into_response()
                }
                Err(e) => {
                    tracing::error!("write lock poisoned during reload: {e}");
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
            tracing::warn!("reload failed (previous model retained) in {elapsed_ms}ms: {e}");
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
            tracing::warn!("reload task join error: {e}");
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

    tracing::info!("--> POST /analyze  id={request_id}");

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
            .map(|s| s.to_owned())
            .unwrap_or_else(|| format!("upload-{request_id}"));
        let sanitized: String = raw
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' { c } else { '_' })
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
    let temp_file = match tokio::task::spawn_blocking(move || {
        TempBuilder::new().suffix(&suffix).tempfile()
    })
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
            tracing::warn!("temp file task join error: {e}  id={request_id}");
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
            tracing::warn!("failed to open temp file for writing: {e}  id={request_id}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Internal error"})),
            )
                .into_response();
        }
    };

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
                if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut tokio_file, &chunk).await
                {
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
    drop(tokio_file);

    if file_size == 0 {
        tracing::warn!("bad request: empty file  id={request_id}");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Empty file"})),
        )
            .into_response();
    }

    tracing::info!("starting analysis: filename={filename:?} size={file_size}  id={request_id}");

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

    let timeout_duration = Duration::from_secs(state.timeout_secs);
    state.active_tasks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    state.in_flight.insert(
        request_id,
        super::InFlightRequest {
            name: filename.clone(),
            size_bytes: file_size as u64,
            started_at: Instant::now(),
        },
    );
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
            state.in_flight.remove(&request_id);
            drop(temp_file);

            match join_result {
                Ok(Ok(scan_result)) => {
                    tracing::info!(
                        "<-- 200 OK  id={request_id} filename={filename:?} size={file_size} \
                         elapsed_ms={elapsed_ms} classification={}",
                        scan_result.classification,
                    );
                    Json(scan_result).into_response()
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        "<-- 500 analysis failed  id={request_id} elapsed_ms={elapsed_ms}: {e}"
                    );
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": "Analysis failed"})),
                    )
                        .into_response()
                }
                Err(e) => {
                    tracing::warn!(
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
            tracing::warn!(
                "analysis timed out after {}s  id={request_id} filename={filename:?} active={active}",
                state.timeout_secs,
            );
            let orphan_state = Arc::clone(&state);
            tokio::spawn(async move {
                let _ = handle.await;
                orphan_state
                    .active_tasks
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                orphan_state.in_flight.remove(&request_id);
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

    let reasons = resources
        .shap
        .as_ref()
        .map(|s| s.explain(&features, &resources.model.spec.feature_names, 5))
        .unwrap_or_default();
    let top_findings = extract_top_findings_from_json(&report_json, &classification);

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
        thresholds: resources.model.thresholds,
        finding_counts,
        formula,
        reasons,
        top_findings,
        file_type,
        size_bytes,
        sha256,
        model: Some(resources.model.info.clone()),
        cleave: Some(report_json),
        pids: None,
        deleted: None,
    })
}

/// Check RSS memory pressure. Returns a 503 response if the server is overloaded,
/// or `None` if memory is within limits or RSS is unavailable on this platform.
fn check_memory_pressure(state: &AppState) -> Option<Response> {
    let rss = cleave::memory_tracker::current_rss()?;

    if rss <= state.max_rss_bytes {
        // Happy path: reset overload timer if set.
        if let Ok(mut overloaded) = state.overloaded_since.try_lock() {
            if overloaded.is_some() {
                tracing::info!("memory recovered: rss={}MB", rss / 1024 / 1024);
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
        tracing::error!(
            "memory overload persisted >30s (rss={}MB), terminating",
            rss / 1024 / 1024
        );
        std::process::exit(1);
    }

    tracing::warn!(
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
            serde_json::json!({
                "request_id": e.key(),
                "name": e.name,
                "size_bytes": e.size_bytes,
                "elapsed_ms": now.duration_since(e.started_at).as_millis(),
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
        libc::sysctl(mib.as_ptr(), 4, std::ptr::null_mut(), &mut len,
                     std::ptr::null_mut(), 0)
    };
    if ret != 0 {
        return serde_json::json!({"error": "sysctl size query failed"});
    }

    len += len / 4; // 25% slack for new threads between calls
    let count = len / mem::size_of::<libc::kinfo_proc>();
    let mut procs: Vec<libc::kinfo_proc> =
        (0..count).map(|_| unsafe { mem::zeroed() }).collect();
    let mut actual_len = len;

    let ret = unsafe {
        libc::sysctl(mib.as_ptr(), 4, procs.as_mut_ptr().cast(), &mut actual_len,
                     std::ptr::null_mut(), 0)
    };
    if ret != 0 {
        return serde_json::json!({"error": "sysctl data query failed"});
    }

    procs.truncate(actual_len / mem::size_of::<libc::kinfo_proc>());

    let c_str = |buf: &[libc::c_char]| {
        let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    let state_str = |s: libc::c_char| match s as u8 {
        1 => "idle", 2 => "running", 3 => "sleeping",
        4 => "stopped", 5 => "zombie", 6 => "waiting", 7 => "locked", _ => "unknown",
    };

    let mut threads: Vec<serde_json::Value> = procs
        .iter()
        .map(|p| serde_json::json!({
            "tid": p.ki_tid,
            "name": c_str(&p.ki_tdname),
            "state": state_str(p.ki_stat),
            "wchan": c_str(&p.ki_wmesg),
        }))
        .collect();

    threads.sort_by_key(|t| t["tid"].as_u64().unwrap_or(0));
    serde_json::json!({"count": threads.len(), "threads": threads})
}
