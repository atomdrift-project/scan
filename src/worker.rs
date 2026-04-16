//! Pull-based worker that polls a hopper instance for analysis jobs.
//!
//! Each worker maintains N concurrent analysis slots via a tokio semaphore.
//! When a slot is free, it claims work from hopper's `/api/next` endpoint,
//! analyzes the file, and posts the result back to `/api/result`.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use crate::model::{Model, Thresholds};
use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::server::{classify_bytes, classify_file, ModelResources};

#[derive(Debug, Clone)]
struct IndexedLocalFile {
    path: PathBuf,
    size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LocalNameKey {
    parent_name: String,
    basename: String,
}

#[derive(Debug, Clone)]
struct CachedFileHash {
    size: u64,
    modified: Option<std::time::SystemTime>,
    sha256: String,
}

#[derive(Debug)]
struct LocalFileIndex {
    root: PathBuf,
    by_name: HashMap<LocalNameKey, Vec<IndexedLocalFile>>,
    verified_by_sha256: dashmap::DashMap<String, PathBuf>,
    hash_cache: dashmap::DashMap<PathBuf, CachedFileHash>,
}

impl LocalFileIndex {
    fn build(root: PathBuf) -> Result<Self> {
        let mut by_name: HashMap<LocalNameKey, Vec<IndexedLocalFile>> = HashMap::new();
        let mut stack = vec![root.clone()];
        let mut indexed_files = 0u64;

        while let Some(dir) = stack.pop() {
            let entries = match fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(e) => {
                    tracing::warn!(path = %dir.display(), error = %e, "failed to read local data directory entry");
                    continue;
                }
            };
            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(e) => {
                        tracing::warn!(path = %dir.display(), error = %e, "failed to enumerate local data directory entry");
                        continue;
                    }
                };
                let path = entry.path();
                let file_type = match entry.file_type() {
                    Ok(ft) => ft,
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e, "failed to read local file type");
                        continue;
                    }
                };
                if file_type.is_dir() {
                    stack.push(path);
                    continue;
                }
                if !file_type.is_file() {
                    continue;
                }

                let size = match entry.metadata() {
                    Ok(meta) => meta.len(),
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e, "failed to read local file metadata");
                        continue;
                    }
                };
                let Some(basename) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                let parent_name = path
                    .parent()
                    .and_then(Path::file_name)
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();
                by_name
                    .entry(LocalNameKey {
                        parent_name,
                        basename: basename.to_string(),
                    })
                    .or_default()
                    .push(IndexedLocalFile { path, size });
                indexed_files += 1;
            }
        }

        let distinct_names = by_name.len();
        tracing::info!(
            root = %root.display(),
            indexed_files,
            distinct_names,
            "built local sample index"
        );

        Ok(Self {
            root,
            by_name,
            verified_by_sha256: dashmap::DashMap::new(),
            hash_cache: dashmap::DashMap::new(),
        })
    }

    fn resolve(&self, requested_path: &str, sha256: &str, size_bytes: i64) -> Result<Option<PathBuf>> {
        if let Some(found) = self.verified_by_sha256.get(sha256) {
            let path = found.value().clone();
            if path.exists() && self.path_matches_sha256(&path, sha256)? {
                tracing::debug!(
                    sha256,
                    path = %path.display(),
                    "using cached local path for sha256"
                );
                return Ok(Some(path));
            }
            self.verified_by_sha256.remove(sha256);
        }

        let mut candidates = Vec::new();
        let requested = Path::new(requested_path);
        if requested.is_relative() {
            let direct = self.root.join(requested);
            if direct.exists() {
                candidates.push(direct);
            }
        }

        let basename = requested.file_name().and_then(|n| n.to_str()).unwrap_or(requested_path);
        let parent_name = requested
            .parent()
            .and_then(Path::file_name)
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let key = LocalNameKey {
            parent_name: parent_name.to_string(),
            basename: basename.to_string(),
        };
        if let Some(indexed) = self.by_name.get(&key) {
            let expected_size = u64::try_from(size_bytes).ok();
            candidates.extend(
                indexed
                    .iter()
                    .filter(|entry| expected_size.is_none_or(|size| entry.size == size))
                    .map(|entry| entry.path.clone()),
            );
        }

        candidates.sort();
        candidates.dedup();

        for candidate in candidates {
            if self.path_matches_sha256(&candidate, sha256)? {
                self.verified_by_sha256
                    .insert(sha256.to_string(), candidate.clone());
                return Ok(Some(candidate));
            }
        }

        Ok(None)
    }

    fn path_matches_sha256(&self, path: &Path, expected_sha256: &str) -> Result<bool> {
        let metadata = fs::metadata(path)
            .with_context(|| format!("reading metadata for {}", path.display()))?;
        let modified = metadata.modified().ok();
        if let Some(cached) = self.hash_cache.get(path) {
            if cached.size == metadata.len() && cached.modified == modified {
                return Ok(cached.sha256 == expected_sha256);
            }
        }

        let digest = sha256_file(path)?;
        self.hash_cache.insert(
            path.to_path_buf(),
            CachedFileHash {
                size: metadata.len(),
                modified,
                sha256: digest.clone(),
            },
        );
        Ok(digest == expected_sha256)
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read;

    let mut file =
        fs::File::open(path).with_context(|| format!("opening local file {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("reading local file {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

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

/// A job with its file data pre-downloaded (or marked for local access).
struct PrefetchedJob {
    job: ClaimJob,
    /// `Ok(None)` = use local file, `Ok(Some(bytes))` = downloaded, `Err` = download failed.
    data: std::result::Result<Option<Vec<u8>>, String>,
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
    // Background: periodic rules update.
    if config.update_interval_mins > 0 {
        let interval = Duration::from_secs(config.update_interval_mins * 60);
        tokio::spawn(async move {
            periodic_update(interval).await;
        });
    }

    let base_url = config.hopper_url.trim_end_matches('/').to_string();
    let data_dir = config.data_dir.clone();
    let local_index = data_dir
        .clone()
        .map(LocalFileIndex::build)
        .transpose()
        .context("building local data index")?
        .map(Arc::new);
    let poll_secs = config.poll_secs;
    let slow_rule_ms = config.slow_rule_ms;
    let timeout_secs = config.timeout_secs;
    let max_jobs = config.max_jobs;
    let max_rss_gb = config.max_rss_gb;
    let encoded_name: String = url_encode(&name);
    let mut consecutive_errors: u32 = 0;
    let completed = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let mut buffer: VecDeque<PrefetchedJob> = VecDeque::new();
    let mut buffer_bytes: usize = 0; // track memory held by prefetched data
    let max_buffer_bytes: usize = if cleave::memory_tracker::total_memory().unwrap_or(0) >= 16 * 1024 * 1024 * 1024 {
        1024 * 1024 * 1024 // 1 GiB on systems with >= 16 GiB RAM
    } else {
        512 * 1024 * 1024  // 512 MiB otherwise
    };
    // With local data, there's no download latency to hide — just claim
    // what we can run immediately. Without local data, prefetch 3x slots
    // so downloads overlap with analysis.
    let has_local_data = data_dir.is_some();
    let prefetch_count = if has_local_data { slots } else { (slots * 3).min(32) };
    let mut last_empty_poll = Instant::now() - Duration::from_secs(poll_secs + 1);

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
                    tokio::task::spawn_blocking(cleave::clear_all_thread_caches);
                    tokio::time::sleep(Duration::from_secs(poll_secs)).await;
                    continue;
                }
            }
        }

        // Refill the prefetch buffer when it's running low. Rate-limit polls
        // so we don't hammer hopper when it has no work.
        let should_poll = buffer.len() < slots
            && buffer_bytes < max_buffer_bytes
            && last_empty_poll.elapsed() >= Duration::from_secs(poll_secs);
        if should_poll {
            let poll_url = format!(
                "{}/api/next?worker={}&count={}&slots={}",
                base_url, encoded_name, prefetch_count, slots
            );
            tracing::info!(url = %poll_url, buffer = buffer.len(), "polling for work");
            let poll_start = Instant::now();

            match claim_and_prefetch(&client, &poll_url, &base_url, data_dir.as_deref()).await {
                Ok(None) => {
                    consecutive_errors = 0;
                    last_empty_poll = Instant::now();
                    if buffer.is_empty() {
                        tracing::debug!(elapsed_ms = crate::duration_ms(poll_start.elapsed()), poll_secs, "no work available, sleeping");
                        tokio::time::sleep(Duration::from_secs(poll_secs)).await;
                        continue;
                    }
                    // Buffer has work — keep dispatching.
                }
                Ok(Some(jobs)) => {
                    consecutive_errors = 0;
                    let n = jobs.len();
                    for pj in jobs {
                        buffer_bytes += pj.data.as_ref().map_or(0, |d| d.as_ref().map_or(0, |v| v.len()));
                        buffer.push_back(pj);
                    }
                    tracing::info!(
                        jobs = n,
                        buffer_mb = buffer_bytes / (1024 * 1024),
                        elapsed_ms = crate::duration_ms(poll_start.elapsed()),
                        "claimed and prefetched",
                    );
                }
                Err(e) => {
                    consecutive_errors += 1;
                    let backoff = backoff_duration(consecutive_errors);
                    tracing::warn!(error = %e, elapsed_ms = crate::duration_ms(poll_start.elapsed()), backoff_secs = backoff.as_secs(), "poll/prefetch failed");
                    if buffer.is_empty() {
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    // Buffer has work — keep dispatching despite poll failure.
                }
            }
        }

        // Dispatch the next prefetched job when a slot is free.
        let pj = match buffer.pop_front() {
            Some(pj) => {
                buffer_bytes -= pj.data.as_ref().map_or(0, |d| d.as_ref().map_or(0, |v| v.len()));
                pj
            }
            None => {
                // Buffer empty and poll rate-limited — wait before retrying.
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };

        // Wait for a free analysis slot.
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .context("semaphore closed")?;

        let client = client.clone();
        let resources = Arc::clone(&resources);
        let url = base_url.clone();
        let name = name.clone();
        let local_index = local_index.clone();
        let completed = Arc::clone(&completed);

        tokio::spawn(async move {
            let result = run_job(
                &client, &url, local_index.as_deref(), &pj.job, &resources,
                slow_rule_ms, timeout_secs, pj.data,
            ).await;
            if let Err(ref e) = result {
                tracing::warn!(
                    sha256 = %pj.job.sha256,
                    file = %pj.job.path,
                    file_type = %pj.job.file_type,
                    size = pj.job.size_bytes,
                    error = %e,
                    "analysis failed",
                );
            }
            post_result(&client, &url, &name, &pj.job.sha256, result).await;
            let n = completed.fetch_add(1, Ordering::Release) + 1;
            if n % 100 == 0 {
                tokio::task::spawn_blocking(cleave::clear_all_thread_caches);
            }
            drop(permit);
        });
    }

    // Drain any in-flight work before exiting.
    let _ = semaphore.acquire_many(u32::try_from(slots).unwrap_or(u32::MAX)).await;
    tracing::info!("all in-flight jobs finished, exiting");
    Ok(())
}

/// Claim jobs from hopper and prefetch file data for all of them concurrently.
/// Returns `Ok(None)` if no work is available (HTTP 204).
async fn claim_and_prefetch(
    client: &reqwest::Client,
    poll_url: &str,
    base_url: &str,
    data_dir: Option<&Path>,
) -> Result<Option<Vec<PrefetchedJob>>> {
    let resp = client.get(poll_url).send().await.context("poll request")?;

    if resp.status() == reqwest::StatusCode::NO_CONTENT {
        return Ok(None);
    }
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("unexpected response from /api/next: {status} {body}");
    }

    let resp_body = resp.text().await.context("read claim body")?;
    let claim: ClaimResponse = serde_json::from_str(&resp_body).context("parse claim response")?;

    if claim.jobs.is_empty() {
        return Ok(None);
    }

    // Prefetch all files concurrently.
    let mut set = tokio::task::JoinSet::new();
    for job in claim.jobs {
        let client = client.clone();
        let base_url = base_url.to_string();
        let data_dir = data_dir.map(Path::to_path_buf);
        set.spawn(async move {
            let local_path = data_dir.as_deref().map(|d| d.join(&job.path));
            let use_local = matches!(local_path, Some(ref p) if p.exists());
            let data = if use_local {
                Ok(None)
            } else {
                match download_bytes(&client, &base_url, &job.sha256, &job.path).await {
                    Ok(bytes) => Ok(Some(bytes)),
                    Err(e) => Err(e),
                }
            };
            PrefetchedJob { job, data }
        });
    }

    let mut prefetched = Vec::with_capacity(set.len());
    while let Some(result) = set.join_next().await {
        match result {
            Ok(pj) => prefetched.push(pj),
            Err(e) => tracing::warn!(error = %e, "prefetch task panicked"),
        }
    }
    Ok(Some(prefetched))
}

/// Analyze a single job. Returns (ml, raw, duration_ms) or an error string.
/// Resolves the job against the local data index if provided. If the file
/// isn't accessible locally (or SHA256 doesn't match), downloads from hopper.
async fn run_job(
    client: &reqwest::Client,
    base_url: &str,
    local_index: Option<&LocalFileIndex>,
    job: &ClaimJob,
    resources: &Arc<ModelResources>,
    slow_rule_ms: u64,
    timeout_secs: u64,
    prefetched: std::result::Result<Option<Vec<u8>>, String>,
) -> Result<(serde_json::Value, serde_json::Value, i64), String> {
    let label = Path::new(&job.path).file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| job.sha256.clone());

    // Try local file first; fall back to downloading bytes from hopper.
    // Exact-path hits are attempted first, then final-dir+basename+size lookup.
    let local_path = match local_index {
        Some(index) => index
            .resolve(&job.path, &job.sha256, job.size_bytes)
            .map_err(|e| e.to_string())?,
        None => None,
    };
    let use_local = match (local_index, local_path.as_ref()) {
        (_, Some(p)) => {
            tracing::info!(
                sha256 = %job.sha256,
                path = %p.display(),
                file_type = %job.file_type,
                size = job.size_bytes,
                "analyzing local file"
            );
            true
        }
        (Some(index), None) => {
            let parent = Path::new(&job.path)
                .parent()
                .and_then(Path::file_name)
                .and_then(|n| n.to_str())
                .unwrap_or("");
            tracing::warn!(
                sha256 = %job.sha256,
                requested_path = %job.path,
                data_root = %index.root.display(),
                parent_dir = %parent,
                basename = %label,
                file_type = %job.file_type,
                size = job.size_bytes,
                "local file not found under --data after exact-path and final-dir+basename+size lookup; downloading from hopper"
            );
            false
        }
        (None, None) => false,
    };

    // Use prefetched bytes, or fall back to downloading if prefetch failed.
    let downloaded: Option<Vec<u8>> = if use_local {
        None
    } else {
        match prefetched {
            Ok(Some(bytes)) => {
                tracing::info!(sha256 = %job.sha256, file = %label, size = bytes.len(), "using prefetched data");
                Some(bytes)
            }
            Ok(None) => None, // shouldn't happen for remote jobs, but handle gracefully
            Err(e) => {
                tracing::warn!(sha256 = %job.sha256, file = %label, error = %e, "prefetch failed, downloading directly");
                let bytes = download_bytes(client, base_url, &job.sha256, &job.path).await?;
                Some(bytes)
            }
        }
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
    let input_source = if use_local { "local" } else { "downloaded" };
    let input_size = if use_local {
        u64::try_from(job.size_bytes).unwrap_or(0)
    } else {
        downloaded.as_ref().map_or(0, |bytes| bytes.len() as u64)
    };

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
                        elapsed_ms = crate::duration_ms(phase_start.elapsed()),
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
                        elapsed_ms = crate::duration_ms(phase_start.elapsed()),
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
                        elapsed_ms = crate::duration_ms(phase_start.elapsed()),
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
                    elapsed_ms = crate::duration_ms(phase_start.elapsed()),
                    "phase still running",
                );
                last_heartbeat = Instant::now();
            }
        }
    });

    tracing::info!(
        sha256 = %job.sha256.get(..12).unwrap_or(&job.sha256),
        file = %label,
        source = input_source,
        size = input_size,
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
                #[allow(clippy::cast_sign_loss)]
                let elapsed_ms = crate::duration_ms(start.elapsed()) as i64;
                return Err(format!(
                    "analysis timed out after {timeout_secs}s ({elapsed_ms}ms) in phase={stuck_phase}"
                ));
            }
        }
    };

    // Always signal the phase watcher to stop. Without this, if cleave returns
    // an error before setting phase="done", the watcher thread leaks indefinitely.
    cancel.store(true, Ordering::Relaxed);

    #[allow(clippy::cast_sign_loss)]
    let elapsed_ms = crate::duration_ms(start.elapsed()) as i64;

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
    // Reuse the same exponential-backoff-with-jitter logic as poll failures.
    // base=2s gives delays of 2s, 4s, 8s, 16s, 32s across 6 attempts (~90s total).
    const MAX_ATTEMPTS: u32 = 6;
    for attempt in 0..MAX_ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(backoff_duration(attempt)).await;
        }
        tracing::debug!(sha256 = %sha256, attempt, "posting result to server");
        let post_start = Instant::now();
        match client.post(&url).json(&payload).send().await {
            Ok(resp) if resp.status().is_success() => {
                tracing::debug!(sha256 = %sha256, elapsed_ms = crate::duration_ms(post_start.elapsed()), "result posted");
                return;
            }
            Ok(resp) => {
                let status = resp.status();
                tracing::warn!(sha256 = %sha256, %status, elapsed_ms = crate::duration_ms(post_start.elapsed()), attempt, "post result: non-success response");
            }
            Err(e) => {
                tracing::warn!(sha256 = %sha256, error = %e, elapsed_ms = crate::duration_ms(post_start.elapsed()), attempt, "post result: send failed");
            }
        }
    }
    tracing::error!(sha256 = %sha256, "post result: giving up after {MAX_ATTEMPTS} attempts");
}

async fn periodic_update(interval: Duration) {
    loop {
        tokio::time::sleep(interval).await;
        tracing::info!("updating rules and models");

        // All update work runs on a single blocking thread under a hard timeout.
        // This ensures:
        //   - git and filesystem calls never block the async runtime
        //   - reload_capability_mapper / clear_all_thread_caches (which may
        //     contend with in-flight analysis tasks for internal cleave locks)
        //     are bounded and cannot deadlock the worker loop
        //   - rollback, if needed, is also on the same blocking thread so all
        //     git ops stay together
        //
        // If the timeout fires, the blocking thread is abandoned and the loop
        // continues; a future iteration will retry.
        const UPDATE_TIMEOUT: Duration = Duration::from_secs(150);
        match tokio::time::timeout(
            UPDATE_TIMEOUT,
            tokio::task::spawn_blocking(|| {
                let prev = match crate::models_repo::update() {
                    Ok(prev) => prev,
                    Err(e) => {
                        tracing::warn!(error = %e, "model update failed");
                        None
                    }
                };
                let traits_ok = match cleave::traits_repo::update(false) {
                    Ok(()) => true,
                    Err(e) => {
                        tracing::warn!(error = %e, "traits update failed");
                        false
                    }
                };

                if traits_ok {
                    if let Err(e) = cleave::reload_capability_mapper() {
                        tracing::warn!(error = %e, "capability mapper reload failed");
                    }
                    cleave::clear_all_thread_caches();
                } else if let Some(ref rev) = prev {
                    tracing::error!(rev, "traits update failed after models pull; rolling back models repo to previous commit");
                    if let Err(e) = crate::models_repo::rollback(rev) {
                        tracing::error!(error = %e, "models rollback failed");
                    }
                }
            }),
        )
        .await
        {
            Ok(Ok(())) => {
                // Note: model binary (xgboost weights) cannot be hot-swapped in
                // worker mode. Traits and capability rules are reloaded above.
                // To pick up a new model, restart the worker.
                tracing::info!("rules updated; restart worker to pick up new model weights");
            }
            Ok(Err(e)) => tracing::warn!(error = %e, "update task panicked"),
            Err(_) => tracing::warn!(
                "update timed out after {}s; continuing to serve requests",
                UPDATE_TIMEOUT.as_secs()
            ),
        }
    }
}

/// Download file bytes from hopper. Tries the fast `/data/{path}` endpoint
/// first (static file serving, no DB query). Falls back to `/api/file/{sha256}`
/// for backward compatibility with older hopper versions.
async fn download_bytes(
    client: &reqwest::Client,
    base_url: &str,
    sha256: &str,
    path: &str,
) -> Result<Vec<u8>, String> {
    let start = Instant::now();

    // Try path-based endpoint first (no DB lookup on hopper side).
    // Encode each path segment to handle filenames with spaces or special chars.
    let encoded_path: String = path.split('/')
        .map(|seg| url_encode(seg))
        .collect::<Vec<_>>()
        .join("/");
    let data_url = format!("{}/data/{}", base_url, encoded_path);
    tracing::debug!(sha256 = %sha256, url = %data_url, "downloading via /data/");
    let resp = client.get(&data_url).send().await.map_err(|e| format!("download {path}: {e}"))?;

    let resp = if resp.status() == reqwest::StatusCode::NOT_FOUND {
        // Fall back to legacy SHA256-based endpoint.
        let file_url = format!("{}/api/file/{}", base_url, sha256);
        tracing::debug!(sha256 = %sha256, url = %file_url, "falling back to /api/file/");
        client.get(&file_url).send().await.map_err(|e| format!("download {path}: {e}"))?
    } else {
        resp
    };

    if !resp.status().is_success() {
        return Err(format!("download {path}: HTTP {}", resp.status()));
    }
    let bytes = resp.bytes().await.map_err(|e| format!("download {path}: read body: {e}"))?;
    tracing::info!(
        sha256 = %sha256,
        file = %path,
        bytes = bytes.len(),
        elapsed_ms = crate::duration_ms(start.elapsed()),
        "download complete",
    );
    Ok(bytes.to_vec())
}

/// Exponential backoff with jitter, capped at 2 minutes.
/// Exponential backoff with jitter for hopper outage recovery.
/// Starts at 1s, doubles each attempt, caps at 60s. Jitter prevents
/// thundering herd when multiple workers reconnect simultaneously.
fn backoff_duration(consecutive_errors: u32) -> Duration {
    let exp = consecutive_errors.min(6); // cap at 2^6 = 64 → capped to 60
    let secs = 1u64.saturating_mul(1 << exp);
    let capped = secs.min(60);
    // Jitter: ±25% using a cheap deterministic hash of the error count.
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(path: &Path, data: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dirs");
        }
        let mut file = fs::File::create(path).expect("create file");
        file.write_all(data).expect("write file");
    }

    fn sha256_hex(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    #[test]
    fn local_index_resolves_exact_relative_path() {
        let root = tempfile::tempdir().expect("create temp dir");
        let rel = Path::new("good/repos/sample.bin");
        let bytes = b"sample-a";
        write_file(&root.path().join(rel), bytes);

        let index = LocalFileIndex::build(root.path().to_path_buf()).expect("build index");
        let resolved = index
            .resolve(
                rel.to_str().expect("utf8 rel path"),
                &sha256_hex(bytes),
                i64::try_from(bytes.len()).expect("len fits"),
            )
            .expect("resolve path");

        assert_eq!(resolved.as_deref(), Some(root.path().join(rel).as_path()));
    }

    #[test]
    fn local_index_resolves_by_final_dir_basename_and_size_for_absolute_requested_path() {
        let root = tempfile::tempdir().expect("create temp dir");
        let stored = root.path().join("bad/harvest/vxug/sample.bin");
        let bytes = b"sample-b";
        write_file(&stored, bytes);

        let index = LocalFileIndex::build(root.path().to_path_buf()).expect("build index");
        let resolved = index
            .resolve(
                "/srv/home/t/data/bad/harvest/vxug/sample.bin",
                &sha256_hex(bytes),
                i64::try_from(bytes.len()).expect("len fits"),
            )
            .expect("resolve path");

        assert_eq!(resolved.as_deref(), Some(stored.as_path()));
    }

    #[test]
    fn local_index_does_not_fallback_to_basename_only_when_final_dir_differs() {
        let root = tempfile::tempdir().expect("create temp dir");
        let stored = root.path().join("good/repos/sample.txt");
        let bytes = b"12345678";
        write_file(&stored, bytes);

        let index = LocalFileIndex::build(root.path().to_path_buf()).expect("build index");
        let resolved = index
            .resolve(
                "/srv/home/t/data/other/place/sample.txt",
                &sha256_hex(bytes),
                i64::try_from(bytes.len()).expect("len fits"),
            )
            .expect("resolve path");

        assert!(resolved.is_none());
    }

    #[test]
    fn local_index_rejects_sha_mismatch_even_when_final_dir_name_and_size_match() {
        let root = tempfile::tempdir().expect("create temp dir");
        let stored = root.path().join("good/repos/sample.txt");
        let bytes = b"12345678";
        write_file(&stored, bytes);

        let index = LocalFileIndex::build(root.path().to_path_buf()).expect("build index");
        let resolved = index
            .resolve(
                "/srv/home/t/data/good/repos/sample.txt",
                &sha256_hex(b"87654321"),
                i64::try_from(bytes.len()).expect("len fits"),
            )
            .expect("resolve path");

        assert!(resolved.is_none());
    }
}
