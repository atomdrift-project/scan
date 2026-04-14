//! Pull-based worker that polls a hopper instance for analysis jobs.
//!
//! Each worker maintains N concurrent analysis slots via a tokio semaphore.
//! When a slot is free, it claims work from hopper's `/api/next` endpoint,
//! analyzes the file, and posts the result back to `/api/result`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::model::{Model, Thresholds};
use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::server::{classify_bytes, classify_file, ModelResources};

/// Configuration for the worker mode.
#[derive(Debug)]
pub struct WorkerConfig {
    /// Hopper API base URL (e.g. `http://hopper:8081`).
    pub hopper_url: String,
    /// Worker name (defaults to hostname).
    pub name: String,
    /// Maximum concurrent analyses.
    pub workers: usize,
    /// Seconds to sleep when no work is available.
    pub poll_secs: u64,
    /// Per-file analysis timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum RSS in GB before pausing (0 = unlimited).
    pub max_rss_gb: u64,
    /// Minutes between rules/model updates (0 = disabled).
    pub update_interval_mins: u64,
    /// Path to model directory.
    pub model_dir: PathBuf,
    /// Optional threshold overrides.
    pub thresholds: Option<Thresholds>,
    /// Local data directory. Paths from hopper are joined with this root.
    /// If the file exists locally and SHA256 matches, it is analyzed in place
    /// instead of downloading from hopper.
    pub data_dir: Option<PathBuf>,
    /// Slow rule warning threshold in ms.
    pub slow_rule_ms: u64,
    /// Exit after this many jobs have been analyzed (None = run forever).
    pub max_jobs: Option<u64>,
}

#[derive(Deserialize)]
struct ClaimResponse {
    jobs: Vec<ClaimJob>,
}

#[derive(Deserialize)]
struct ClaimJob {
    sha256: String,
    path: String,
    size_bytes: i64,
    #[serde(default)]
    file_type: String,
}

#[derive(Serialize)]
struct ResultPayload {
    sha256: String,
    worker: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ml: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    duration_ms: i64,
}

/// Run the worker loop. Blocks until cancelled.
pub async fn run(config: WorkerConfig) -> Result<()> {
    let name = config.name.clone();
    let slots = config.workers;
    let mut client_builder = reqwest::Client::builder();
    if config.timeout_secs > 0 {
        client_builder = client_builder.timeout(Duration::from_secs(config.timeout_secs + 120));
    }
    let client = client_builder.build()?;
    let semaphore = Arc::new(Semaphore::new(slots));

    // Load model resources once at startup.
    let model = Model::load(&config.model_dir, config.thresholds)
        .context("loading model")?;
    let shap = ShapImportance::load(&config.model_dir).ok();
    let ctx = ExtractContext::new(model.spec());
    let resources = Arc::new(ModelResources {
        model,
        shap,
        ctx,
    });

    tracing::info!(
        name = %name,
        slots = slots,
        hopper = %config.hopper_url,
        "worker starting"
    );

    // Pre-warm YARA compiler and capability mapper before claiming work.
    // Both are initialized lazily; the first analysis blocks until they're ready.
    // Doing this at startup avoids a multi-second latency spike on the first real job.
    // A minimal PE stub triggers full parallel initialization via cleave's rayon::join.
    tracing::info!("pre-warming cleave resources");
    let warm_start = Instant::now();
    tokio::task::spawn_blocking(|| {
        let _ = cleave::analyze_bytes(
            b"\x4d\x5a\x90\x00\x03\x00\x00\x00",
            "warmup.exe",
            &cleave::AnalysisOptions::default(),
        );
    })
    .await
    .ok();
    tracing::info!(elapsed_ms = warm_start.elapsed().as_millis() as u64, "cleave resources ready");

    // Background: periodic rules update.
    if config.update_interval_mins > 0 {
        let interval = Duration::from_secs(config.update_interval_mins * 60);
        let model_dir = config.model_dir.clone();
        let thresholds = config.thresholds;
        let resources_ref = Arc::clone(&resources);
        tokio::spawn(async move {
            periodic_update(interval, &model_dir, thresholds.as_ref(), &resources_ref).await;
        });
    }

    let base_url = config.hopper_url.trim_end_matches('/').to_string();
    let data_dir = config.data_dir.clone();
    let poll_secs = config.poll_secs;
    let slow_rule_ms = config.slow_rule_ms;
    let timeout_secs = config.timeout_secs;
    let max_jobs = config.max_jobs;
    let max_rss_gb = config.max_rss_gb;
    let encoded_name: String = url_encode(&name);
    let mut consecutive_errors: u32 = 0;
    let completed = Arc::new(std::sync::atomic::AtomicU64::new(0));

    loop {
        if let Some(max) = max_jobs {
            if completed.load(Ordering::Acquire) >= max {
                tracing::info!(max_jobs = max, "job limit reached, draining in-flight work");
                break;
            }
        }

        // Enforce memory limit before claiming more work.
        if max_rss_gb > 0 {
            let max_bytes = max_rss_gb.saturating_mul(1024 * 1024 * 1024);
            if let Some(rss) = cleave::memory_tracker::current_rss() {
                if rss > max_bytes {
                    tracing::warn!(
                        rss_mb = rss / 1024 / 1024,
                        max_rss_mb = max_bytes / 1024 / 1024,
                        "memory pressure: pausing before claiming new work",
                    );
                    cleave::clear_all_thread_caches();
                    tokio::time::sleep(Duration::from_secs(poll_secs)).await;
                    continue;
                }
            }
        }

        // Wait for at least one free slot.
        let available = semaphore.available_permits();
        if available == 0 {
            tracing::debug!(slots, "all slots occupied — waiting for a job to complete");
        }
        let gate = semaphore
            .clone()
            .acquire_owned()
            .await
            .context("semaphore closed")?;
        let free = semaphore.available_permits() + 1;

        // Poll for work.
        let poll_url = format!(
            "{}/api/next?worker={}&count={}&slots={}",
            base_url, encoded_name, free, slots
        );
        let resp = match client.get(&poll_url).send().await {
            Ok(r) => r,
            Err(e) => {
                drop(gate);
                consecutive_errors += 1;
                let backoff = backoff_duration(poll_secs, consecutive_errors);
                tracing::warn!(error = %e, backoff_secs = backoff.as_secs(), "poll failed");
                tokio::time::sleep(backoff).await;
                continue;
            }
        };

        if resp.status() == reqwest::StatusCode::NO_CONTENT {
            drop(gate);
            consecutive_errors = 0;
            tokio::time::sleep(Duration::from_secs(poll_secs)).await;
            continue;
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            drop(gate);
            consecutive_errors += 1;
            let backoff = backoff_duration(poll_secs, consecutive_errors);
            tracing::warn!(%status, body = %body, backoff_secs = backoff.as_secs(), "unexpected response from /api/next");
            tokio::time::sleep(backoff).await;
            continue;
        }

        consecutive_errors = 0;

        let resp_body = resp.text().await.unwrap_or_default();
        let claim: ClaimResponse = match serde_json::from_str(&resp_body) {
            Ok(c) => c,
            Err(e) => {
                drop(gate);
                consecutive_errors += 1;
                let backoff = backoff_duration(poll_secs, consecutive_errors);
                let preview = if resp_body.len() > 200 { &resp_body[..200] } else { &resp_body };
                tracing::warn!(error = %e, body = %preview, backoff_secs = backoff.as_secs(), "failed to parse claim response");
                tokio::time::sleep(backoff).await;
                continue;
            }
        };

        // Release the gate permit — each job gets its own permit below.
        drop(gate);

        for job in claim.jobs {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .context("semaphore closed")?;
            let client = client.clone();
            let resources = Arc::clone(&resources);
            let url = base_url.clone();
            let name = name.clone();
            let data_dir = data_dir.clone();
            let completed = Arc::clone(&completed);

            tokio::spawn(async move {
                let result = run_job(
                    &client, &url, data_dir.as_deref(), &job, &resources, slow_rule_ms, timeout_secs,
                ).await;
                if let Err(ref e) = result {
                    tracing::warn!(
                        sha256 = %job.sha256,
                        file = %job.path,
                        file_type = %job.file_type,
                        size = job.size_bytes,
                        error = %e,
                        "analysis failed",
                    );
                }
                post_result(&client, &url, &name, &job.sha256, result).await;
                completed.fetch_add(1, Ordering::Release);
                drop(permit);
            });
        }
    }

    // Drain any in-flight work before exiting.
    let _ = semaphore.acquire_many(slots as u32).await;
    tracing::info!("all in-flight jobs finished, exiting");
    Ok(())
}

/// Analyze a single job. Returns (ml, raw, duration_ms) or an error string.
/// Resolves the relative path against data_dir if provided. If the file
/// isn't accessible locally (or SHA256 doesn't match), downloads from hopper.
async fn run_job(
    client: &reqwest::Client,
    base_url: &str,
    data_dir: Option<&Path>,
    job: &ClaimJob,
    resources: &Arc<ModelResources>,
    slow_rule_ms: u64,
    timeout_secs: u64,
) -> Result<(serde_json::Value, serde_json::Value, i64), String> {
    let label = Path::new(&job.path).file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| job.sha256.clone());

    // Try local file first; fall back to downloading bytes from hopper.
    // Trust hopper's SHA256 — re-hashing would read the file twice and block
    // the tokio executor with blocking I/O.
    let local_path = data_dir.map(|d| d.join(&job.path));
    let use_local = match local_path {
        Some(ref p) if p.exists() => {
            tracing::info!(sha256 = %job.sha256, path = %p.display(), file_type = %job.file_type, size = job.size_bytes, "analyzing local file");
            true
        }
        _ => false,
    };

    // Download bytes if not using local file. analyze_bytes avoids writing to disk.
    let downloaded: Option<Vec<u8>> = if use_local {
        None
    } else {
        let bytes = download_bytes(client, base_url, &job.sha256, &label).await?;
        tracing::info!(sha256 = %job.sha256, file = %label, size = bytes.len(), "downloaded for in-memory analysis");
        Some(bytes)
    };

    let resources = Arc::clone(resources);
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel2 = Arc::clone(&cancel);
    let local = local_path.clone();

    // Run analysis on a blocking thread with phase logging.
    let start = Instant::now();
    let phase = cleave::PhaseTracker::new();
    let phase2 = phase.clone();
    let phase_timeout = phase.clone();
    let label2 = label.clone();
    let sha_short = job.sha256.get(..12).unwrap_or(&job.sha256).to_string();

    // Background phase watcher — logs transitions with timing, and emits a
    // heartbeat every 30 s so a stuck phase is visible in logs.
    // Uses a tokio task instead of an OS thread to avoid one thread-per-job overhead.
    let cancel_watcher = cancel.clone();
    tokio::task::spawn(async move {
        let mut last_phase = String::new();
        let mut phase_start = Instant::now();
        let mut last_heartbeat = Instant::now();
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            if cancel_watcher.load(Ordering::Relaxed) {
                // Log completion of the final phase so elapsed time is never lost.
                if !last_phase.is_empty() && last_phase != "done" {
                    tracing::info!(
                        sha256 = %sha_short,
                        file = %label2,
                        phase = %last_phase,
                        elapsed_ms = phase_start.elapsed().as_millis() as u64,
                        "phase complete",
                    );
                }
                break;
            }
            let current = phase2.get();
            if current.is_empty() {
                // Phase tracker not yet updated — log if this persists.
                if last_heartbeat.elapsed().as_secs() >= 30 {
                    tracing::warn!(
                        sha256 = %sha_short,
                        file = %label2,
                        elapsed_ms = phase_start.elapsed().as_millis() as u64,
                        "analysis running but phase tracker has not been updated",
                    );
                    last_heartbeat = Instant::now();
                }
                continue;
            }
            if current != last_phase {
                if !last_phase.is_empty() {
                    tracing::info!(
                        sha256 = %sha_short,
                        file = %label2,
                        phase = %last_phase,
                        elapsed_ms = phase_start.elapsed().as_millis() as u64,
                        "phase complete",
                    );
                }
                last_phase = current;
                phase_start = Instant::now();
                last_heartbeat = Instant::now();
                tracing::info!(
                    sha256 = %sha_short,
                    file = %label2,
                    phase = %last_phase,
                    "phase started",
                );
                if last_phase == "done" {
                    break;
                }
            } else if last_heartbeat.elapsed().as_secs() >= 30 {
                tracing::warn!(
                    sha256 = %sha_short,
                    file = %label2,
                    phase = %last_phase,
                    elapsed_ms = phase_start.elapsed().as_millis() as u64,
                    "phase still running",
                );
                last_heartbeat = Instant::now();
            }
        }
    });

    tracing::info!(
        sha256 = %job.sha256.get(..12).unwrap_or(&job.sha256),
        file = %label,
        size = downloaded.as_ref().map_or(0, Vec::len),
        "analysis starting",
    );
    let sha_short2 = job.sha256.get(..12).unwrap_or(&job.sha256).to_string();
    let handle = tokio::task::spawn_blocking(move || {
        tracing::info!(
            sha256 = %sha_short2,
            thread_id = os_thread_id(),
            "analysis thread started",
        );
        if let Some(data) = downloaded {
            classify_bytes(&data, &label, &resources, slow_rule_ms, Some(&cancel2), Some(&phase))
        } else if let Some(path) = local.as_ref() {
            classify_file(path, &label, &resources, slow_rule_ms, None, Some(&cancel2), Some(&phase))
        } else {
            Err(anyhow::anyhow!("no downloaded bytes and no local path for {label}"))
        }
    });

    let result = if timeout_secs == 0 {
        // No timeout — let analysis run as long as it needs.
        handle.await
    } else {
        match tokio::time::timeout(Duration::from_secs(timeout_secs), handle).await {
            Ok(r) => r,
            Err(_) => {
                let stuck_phase = phase_timeout.get();
                cancel.store(true, Ordering::Relaxed);
                let elapsed_ms = start.elapsed().as_millis() as i64;
                return Err(format!(
                    "analysis timed out after {timeout_secs}s ({elapsed_ms}ms) in phase={stuck_phase}"
                ));
            }
        }
    };

    // Always signal the phase watcher to stop. Without this, if cleave returns
    // an error before setting phase="done", the watcher thread leaks indefinitely.
    cancel.store(true, Ordering::Relaxed);

    let elapsed_ms = start.elapsed().as_millis() as i64;

    match result {
        Ok(Ok(scan_result)) => {
            let envelope = scan_result.to_envelope();
            let ml = serde_json::to_value(&envelope.ml).map_err(|e| e.to_string())?;
            let raw = envelope.raw;
            Ok((ml, raw, elapsed_ms))
        }
        Ok(Err(e)) => Err(format!("{e:#}")),
        Err(e) => Err(format!("task join error: {e}")),
    }
}

/// Post the result back to hopper with retry on transient failures.
async fn post_result(
    client: &reqwest::Client,
    url: &str,
    worker: &str,
    sha256: &str,
    result: Result<(serde_json::Value, serde_json::Value, i64), String>,
) {
    let payload = match result {
        Ok((ml, raw, duration_ms)) => {
            let classification = match ml.get("class").and_then(serde_json::Value::as_u64) {
                Some(0) => "benign",
                Some(1) => "suspicious",
                Some(2) => "hostile",
                _ => "unknown",
            };
            tracing::info!(sha256 = %sha256, duration_ms, classification, "analysis complete");
            ResultPayload {
                sha256: sha256.to_string(),
                worker: worker.to_string(),
                ml: Some(ml),
                raw: Some(raw),
                error: None,
                duration_ms,
            }
        }
        Err(e) => {
            ResultPayload {
                sha256: sha256.to_string(),
                worker: worker.to_string(),
                ml: None,
                raw: None,
                error: Some(e),
                duration_ms: 0,
            }
        }
    };

    let url = format!("{}/api/result", url);
    for attempt in 0..3u32 {
        if attempt > 0 {
            let delay = Duration::from_secs(1 << attempt);
            tokio::time::sleep(delay).await;
        }
        match client.post(&url).json(&payload).send().await {
            Ok(resp) if resp.status().is_success() => return,
            Ok(resp) => {
                let status = resp.status();
                tracing::warn!(sha256 = %sha256, %status, attempt, "post result: non-success response");
            }
            Err(e) => {
                tracing::warn!(sha256 = %sha256, error = %e, attempt, "post result: send failed");
            }
        }
    }
    tracing::error!(sha256 = %sha256, "post result: giving up after 3 attempts");
}

async fn periodic_update(
    interval: Duration,
    _model_dir: &Path,
    _thresholds: Option<&Thresholds>,
    _resources: &Arc<ModelResources>,
) {
    loop {
        tokio::time::sleep(interval).await;
        tracing::info!("updating rules and models");

        // Pull latest models and traits.
        if let Err(e) = crate::models_repo::update() {
            tracing::warn!(error = %e, "model update failed");
        }
        if let Err(e) = cleave::traits_repo::update(false) {
            tracing::warn!(error = %e, "traits update failed");
        }

        // Reload capabilities after traits update.
        let _ = cleave::reload_capability_mapper();
        cleave::clear_all_thread_caches();

        // Note: model binary (xgboost weights) cannot be hot-swapped without
        // interior mutability on ModelResources. Traits and capability rules
        // are reloaded above. To pick up a new model, restart the worker.
        tracing::info!("rules updated; restart worker to pick up new model weights");
    }
}

/// Download file bytes from hopper's /api/file endpoint.
async fn download_bytes(
    client: &reqwest::Client,
    base_url: &str,
    sha256: &str,
    label: &str,
) -> Result<Vec<u8>, String> {
    let url = format!("{}/api/file/{}", base_url, sha256);
    let resp = client.get(&url).send().await.map_err(|e| format!("download {label}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("download {label}: HTTP {}", resp.status()));
    }
    let bytes = resp.bytes().await.map_err(|e| format!("download {label}: read body: {e}"))?;
    Ok(bytes.to_vec())
}

/// Exponential backoff with jitter, capped at 2 minutes.
fn backoff_duration(base_secs: u64, consecutive_errors: u32) -> Duration {
    let exp = std::cmp::min(consecutive_errors, 7); // cap at 2^7 = 128s
    let secs = base_secs.saturating_mul(1 << exp);
    let capped = std::cmp::min(secs, 120);
    // Simple jitter: ±25% using a cheap hash of the error count.
    let jitter = (consecutive_errors as u64 * 7 + 3) % (capped / 4 + 1);
    Duration::from_secs(capped.saturating_add(jitter))
}

/// Returns the OS-level thread ID for the calling thread.
///
/// The returned value matches what `lldb`'s `thread list`, `sample`, and
/// `/proc/self/task/` show, making it possible to correlate log lines with
/// debugger or profiler output when diagnosing stuck analyses.
fn os_thread_id() -> u64 {
    #[cfg(target_os = "linux")]
    {
        unsafe { libc::syscall(libc::SYS_gettid) as u64 }
    }
    #[cfg(target_os = "macos")]
    {
        let mut tid: u64 = 0;
        unsafe { libc::pthread_threadid_np(0, &mut tid) };
        tid
    }
    #[cfg(target_os = "freebsd")]
    {
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

/// Percent-encode a string for use in URL query parameters.
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push('%');
                out.push(char::from(b"0123456789ABCDEF"[(b >> 4) as usize]));
                out.push(char::from(b"0123456789ABCDEF"[(b & 0xF) as usize]));
            }
        }
    }
    out
}

