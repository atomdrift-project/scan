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
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.timeout_secs + 120))
        .build()?;
    let semaphore = Arc::new(Semaphore::new(slots));

    // Load model resources once at startup.
    let model = Model::load(&config.model_dir, config.thresholds.clone())
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

    let base_url = config.hopper_url.clone();
    let data_dir = config.data_dir.clone();
    let poll_secs = config.poll_secs;
    let slow_rule_ms = config.slow_rule_ms;
    let timeout_secs = config.timeout_secs;
    let encoded_name: String = url_encode(&name);
    let mut consecutive_errors: u32 = 0;

    loop {
        // Wait for at least one free slot.
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

            tokio::spawn(async move {
                let result = run_job(
                    &client, &url, data_dir.as_deref(), &job, &resources, slow_rule_ms, timeout_secs,
                ).await;
                if let Err(ref e) = result {
                    tracing::warn!(
                        sha256 = %job.sha256,
                        file_type = %job.file_type,
                        size = job.size_bytes,
                        error = %e,
                        "analysis failed",
                    );
                }
                post_result(&client, &url, &name, &job.sha256, result).await;
                drop(permit);
            });
        }
    }
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
    let local_path = data_dir.map(|d| d.join(&job.path));
    let use_local = match local_path {
        Some(ref p) if p.exists() => match verify_sha256(p, &job.sha256) {
            Ok(true) => {
                tracing::info!(sha256 = %job.sha256, path = %p.display(), file_type = %job.file_type, size = job.size_bytes, "analyzing local file");
                true
            }
            Ok(false) => {
                tracing::info!(sha256 = %job.sha256, path = %p.display(), "SHA256 mismatch, will download");
                false
            }
            Err(e) => {
                tracing::info!(sha256 = %job.sha256, path = %p.display(), error = %e, "hash check failed, will download");
                false
            }
        },
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
    let timeout = Duration::from_secs(timeout_secs);
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel2 = Arc::clone(&cancel);
    let local = local_path.clone();

    // Run analysis on a blocking thread with a timeout.
    let start = Instant::now();
    let handle = tokio::task::spawn_blocking(move || {
        if let Some(data) = downloaded {
            // In-memory analysis — no tempfile needed.
            classify_bytes(&data, &label, &resources, slow_rule_ms, Some(&cancel2))
        } else {
            // Local file on disk.
            let path = local.as_ref().expect("local path must be set");
            classify_file(path, &label, &resources, slow_rule_ms, None, Some(&cancel2), None)
        }
    });

    let result = tokio::time::timeout(timeout, handle).await;
    let elapsed_ms = start.elapsed().as_millis() as i64;

    match result {
        Ok(Ok(Ok(scan_result))) => {
            let envelope = scan_result.to_envelope();
            let ml = serde_json::to_value(&envelope.ml).map_err(|e| e.to_string())?;
            let raw = envelope.raw;
            Ok((ml, raw, elapsed_ms))
        }
        Ok(Ok(Err(e))) => Err(format!("{e:#}")),
        Ok(Err(e)) => Err(format!("task join error: {e}")),
        Err(_) => {
            cancel.store(true, Ordering::Relaxed);
            Err(format!("analysis timed out after {timeout_secs}s"))
        }
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
            tracing::debug!(sha256 = %sha256, duration_ms = duration_ms, "analysis complete");
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

/// Verify that a file's SHA256 matches the expected value.
fn verify_sha256(path: &Path, expected: &str) -> Result<bool, std::io::Error> {
    use sha2::{Sha256, Digest};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    let hash = format!("{:x}", hasher.finalize());
    Ok(hash == expected)
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

