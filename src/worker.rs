//! Pull-based worker that polls a hopper instance for analysis jobs.
//!
//! Each worker maintains N concurrent analysis slots via a tokio semaphore.
//! When a slot is free, it claims work from hopper's `/api/next` endpoint,
//! analyzes the file, and posts the result back to `/api/result`.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Model, Thresholds};
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
    sha256: [u8; 32],
}

/// Stable handle into `LocalFileIndex::files`. `u32` supports 4 B indexed files,
/// far beyond any plausible data dir; halving the index width vs. `usize` lets
/// the secondary caches fit more per bucket.
type FileId = u32;

/// Identity hasher for SHA-256 digests. SHA-256 output is already uniformly
/// distributed, so we can skip hashing entirely and use any 8 bytes of the
/// digest as the hash code — dashmap shards and hashbrown buckets then spread
/// keys just as well as a wyhash/foldhash pass would, at zero cost per lookup.
#[derive(Default)]
struct Sha256IdentityHasher(u64);

impl Hasher for Sha256IdentityHasher {
    fn write(&mut self, bytes: &[u8]) {
        if let Some(first8) = bytes.first_chunk::<8>() {
            self.0 = u64::from_ne_bytes(*first8);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type Sha256IdentityBuildHasher = BuildHasherDefault<Sha256IdentityHasher>;

static NEXT_ANALYSIS_ID: AtomicU64 = AtomicU64::new(1);
static BLOCKING_STARTED_TOTAL: AtomicU64 = AtomicU64::new(0);
static BLOCKING_FINISHED_TOTAL: AtomicU64 = AtomicU64::new(0);
const RESOURCE_RENEWAL_INTERVAL: Duration = Duration::from_secs(10 * 60);

type ResourceHandle = Arc<RwLock<Arc<ModelResources>>>;

#[derive(Debug)]
struct LocalFileIndex {
    root: PathBuf,
    /// Every file found under `root` at startup — the single owner of each
    /// `PathBuf`. Secondary indexes refer to entries by `FileId`.
    files: Vec<IndexedLocalFile>,
    by_name: HashMap<LocalNameKey, Vec<FileId>>,
    /// SHA-256 → `FileId` for files whose content hash has been confirmed.
    verified_by_sha256: dashmap::DashMap<[u8; 32], FileId, Sha256IdentityBuildHasher>,
    /// Per-file lazily-populated hash cache, indexed by `FileId`. A boxed
    /// slice of `OnceLock` gives lock-free reads and a bounded, preallocated
    /// footprint (one slot per indexed file, regardless of how many are
    /// eventually hashed).
    hash_cache: Box<[OnceLock<CachedFileHash>]>,
}

impl LocalFileIndex {
    // Returns Result for forward-compatibility: individual dir-entry failures
    // are currently logged and skipped, but a future cap on I/O errors, a
    // permission-denied signal, or a root-missing fail-fast policy would want
    // to bubble up here.
    #[allow(clippy::unnecessary_wraps)]
    fn build(root: PathBuf) -> Result<Self> {
        let mut files: Vec<IndexedLocalFile> = Vec::new();
        let mut by_name: HashMap<LocalNameKey, Vec<FileId>> = HashMap::new();
        let mut stack = vec![root.clone()];

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

                // `FileId` is `u32`; bail out cleanly if a single data dir
                // somehow exceeds 4 B files rather than silently truncating.
                let Ok(file_id) = FileId::try_from(files.len()) else {
                    tracing::warn!(
                        root = %root.display(),
                        limit = FileId::MAX,
                        "local data index exceeds FileId capacity; ignoring remaining files",
                    );
                    break;
                };

                let basename = basename.to_string();
                files.push(IndexedLocalFile { path, size });
                by_name
                    .entry(LocalNameKey {
                        parent_name,
                        basename,
                    })
                    .or_default()
                    .push(file_id);
            }
        }

        let indexed_files = files.len();
        let distinct_names = by_name.len();
        tracing::info!(
            root = %root.display(),
            indexed_files,
            distinct_names,
            "built local sample index"
        );

        let hash_cache = (0..files.len())
            .map(|_| OnceLock::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            root,
            files,
            by_name,
            verified_by_sha256: dashmap::DashMap::with_hasher(Sha256IdentityBuildHasher::default()),
            hash_cache,
        })
    }

    fn resolve(
        &self,
        requested_path: &str,
        sha256: &str,
        size_bytes: i64,
    ) -> Result<Option<PathBuf>> {
        // Decode once at the boundary; all internal state is raw [u8; 32].
        let Some(expected) = sha256_from_hex(sha256) else {
            anyhow::bail!("expected 64-char hex sha256, got {:?}", sha256);
        };

        if let Some(found) = self.verified_by_sha256.get(&expected) {
            let file_id = *found.value();
            drop(found); // release the dashmap shard lock before any I/O
            if let Some(entry) = self.files.get(file_id as usize) {
                // One stat syscall instead of two: `path.exists()` + the later
                // `fs::metadata` inside `file_matches_sha256` used to hit the
                // filesystem twice for every local cache hit.
                if self.file_matches_sha256(file_id, entry, &expected)? {
                    tracing::debug!(
                        sha256,
                        path = %entry.path.display(),
                        "using cached local path for sha256"
                    );
                    return Ok(Some(entry.path.clone()));
                }
            }
            self.verified_by_sha256.remove(&expected);
        }

        let mut candidates: Vec<FileId> = Vec::new();
        let requested = Path::new(requested_path);
        if requested.is_relative() {
            let direct = self.root.join(requested);
            // Exact-path hits are still resolved via the name index so that
            // caches stay keyed by `FileId`. A disk-only match that isn't in
            // `by_name` is treated as absent (the index is the source of truth
            // for what this worker can analyze locally).
            if direct.exists()
                && let Some(id) = self.file_id_for_path(&direct)
            {
                candidates.push(id);
            }
        }

        let basename = requested
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(requested_path);
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
            candidates.extend(indexed.iter().copied().filter(|id| {
                self.files
                    .get(*id as usize)
                    .is_some_and(|entry| expected_size.is_none_or(|size| entry.size == size))
            }));
        }

        candidates.sort_unstable();
        candidates.dedup();

        for file_id in candidates {
            let Some(entry) = self.files.get(file_id as usize) else {
                continue;
            };
            if self.file_matches_sha256(file_id, entry, &expected)? {
                self.verified_by_sha256.insert(expected, file_id);
                return Ok(Some(entry.path.clone()));
            }
        }

        // Index miss — the file may have been added after the index was built
        // (e.g. newly harvested). Try the direct path on disk, verifying by
        // SHA-256 before trusting it. For absolute paths, also try
        // canonicalize() in case the path uses a symlinked prefix.
        let mut disk_candidates: Vec<PathBuf> = Vec::new();
        if requested.is_relative() {
            disk_candidates.push(self.root.join(requested));
        } else if requested.is_absolute() {
            disk_candidates.push(requested.to_path_buf());
            if let Ok(resolved) = requested.canonicalize()
                && resolved != requested
            {
                disk_candidates.push(resolved);
            }
        }
        let expected_size = u64::try_from(size_bytes).ok();
        for candidate in &disk_candidates {
            let meta = match fs::metadata(candidate) {
                Ok(m) if m.is_file() => m,
                _ => continue,
            };
            if expected_size.is_some_and(|sz| meta.len() != sz) {
                continue;
            }
            if let Ok(digest) = sha256_file(candidate)
                && digest == expected
            {
                tracing::info!(
                    sha256,
                    path = %candidate.display(),
                    "resolved file outside index by sha256 verification",
                );
                return Ok(Some(candidate.clone()));
            }
        }

        Ok(None)
    }

    fn file_id_for_path(&self, path: &Path) -> Option<FileId> {
        let basename = path.file_name().and_then(|n| n.to_str())?.to_string();
        let parent_name = path
            .parent()
            .and_then(Path::file_name)
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let key = LocalNameKey {
            parent_name,
            basename,
        };
        let candidates = self.by_name.get(&key)?;
        candidates
            .iter()
            .copied()
            .find(|id| self.files.get(*id as usize).is_some_and(|e| e.path == path))
    }

    fn file_matches_sha256(
        &self,
        file_id: FileId,
        entry: &IndexedLocalFile,
        expected: &[u8; 32],
    ) -> Result<bool> {
        // A missing file is not an error here — it means the cached entry is
        // stale and the caller should fall through to the filename-index path.
        let metadata = match fs::metadata(&entry.path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => {
                return Err(anyhow::Error::from(e)
                    .context(format!("reading metadata for {}", entry.path.display())));
            }
        };
        let modified = metadata.modified().ok();
        let size = metadata.len();

        // Stale entries (size/modified mismatch) fall through to re-hash.
        // `OnceLock` is write-once, so the re-hash pays no insert cost; in
        // practice files under `--data` don't rewrite, so this is rare.
        if let Some(slot) = self.hash_cache.get(file_id as usize)
            && let Some(cached) = slot.get()
            && cached.size == size
            && cached.modified == modified
        {
            return Ok(&cached.sha256 == expected);
        }

        let digest = sha256_file(&entry.path)?;
        if let Some(slot) = self.hash_cache.get(file_id as usize) {
            // Ignore the Err case: another thread raced us and won; its value
            // is equivalent (content-addressed), so drop ours silently.
            let _ = slot.set(CachedFileHash {
                size,
                modified,
                sha256: digest,
            });
        }
        Ok(&digest == expected)
    }
}

/// Decode a lowercase/uppercase hex SHA-256 string into raw bytes. Returns
/// `None` for any non-hex byte or wrong length — callers treat that as an
/// invalid job rather than propagating a structured error.
fn sha256_from_hex(hex: &str) -> Option<[u8; 32]> {
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        *slot = (hi << 4) | lo;
    }
    Some(out)
}

fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn sha256_file(path: &Path) -> Result<[u8; 32]> {
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
    Ok(hasher.finalize().into())
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
    /// Maximum RSS in GB before pausing (0 = unlimited).
    pub max_rss_gb: u64,
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
    /// FPR severity level (0..=19) that produced the thresholds, or `None` when
    /// manual thresholds were supplied. Surfaces as `ml.level` in the envelope.
    pub level: Option<u8>,
    /// Nice value applied to the process at startup (0 = leave unchanged).
    pub nice: i32,
}

fn load_model_resources(
    model_dir: &Path,
    thresholds: Option<Thresholds>,
    level: Option<u8>,
) -> Result<Arc<ModelResources>> {
    let model = Model::load(model_dir, thresholds).context("loading model")?;
    let shap = ShapImportance::load(model_dir).ok();
    let ctx = ExtractContext::new(model.spec());
    Ok(Arc::new(ModelResources {
        model,
        shap,
        ctx,
        level,
    }))
}

fn validate_and_load_resources(
    model_dir: &Path,
    thresholds: Option<Thresholds>,
    slow_rule_ms: u64,
    level: Option<u8>,
) -> Result<Arc<ModelResources>> {
    let validate_config = crate::ScanConfig::new(
        model_dir,
        crate::OutputFormat::Terminal,
        thresholds,
        crate::DisplayFilter::alerts_only(),
        slow_rule_ms,
        false,
    )?
    .with_level(level);
    crate::validate::run(&validate_config)?;
    load_model_resources(model_dir, thresholds, level)
}

/// Pull upstream rules and, **only if something actually changed**, re-validate
/// and reload the model bundle. Returns `Ok(None)` when both repos are already
/// up to date — a silent no-op so the periodic renewal doesn't flood the log
/// with a full validation pass every interval.
fn renew_resources_once(
    model_dir: &Path,
    thresholds: Option<Thresholds>,
    slow_rule_ms: u64,
    level: Option<u8>,
) -> Result<Option<Arc<ModelResources>>> {
    let (prev_models_head, models_after, models_changed) = match crate::models_repo::update(true) {
        Ok(prev) => {
            let after = crate::models_repo::version();
            let changed = prev.as_deref() != after.as_deref();
            (prev, after, changed)
        }
        Err(error) => {
            tracing::warn!(error = %error, "model renewal fetch failed; treating models as unchanged");
            (None, None, false)
        }
    };
    let traits_changed = match crate::traits_repo::update(false, true) {
        Ok(changed) => changed,
        Err(error) => {
            tracing::warn!(error = %error, "traits renewal fetch failed; treating traits as unchanged");
            false
        }
    };

    if !models_changed && !traits_changed {
        return Ok(None);
    }

    tracing::info!(
        models_changed,
        traits_changed,
        models_from = prev_models_head.as_deref().unwrap_or("none"),
        models_to = models_after.as_deref().unwrap_or("unknown"),
        "rules changed; revalidating bundle",
    );

    let resources = match validate_and_load_resources(
        model_dir,
        thresholds,
        slow_rule_ms,
        level,
    ) {
        Ok(resources) => resources,
        Err(error) => {
            if let Some(prev) = prev_models_head.as_deref() {
                tracing::error!(rollback_to = %prev, "renewed models failed validation; rolling back");
                if let Err(rollback_error) = crate::models_repo::rollback(prev) {
                    tracing::error!(error = %rollback_error, "model rollback after failed renewal failed");
                }
            }
            return Err(error);
        }
    };

    match cleave::reload_capability_mapper() {
        Ok((traits, composites)) => {
            tracing::info!(traits, composites, "cleave capability mapper renewed");
        }
        Err(error) => {
            if let Some(prev) = prev_models_head.as_deref() {
                tracing::error!(rollback_to = %prev, "capability mapper reload failed after renewal; rolling back models");
                if let Err(rollback_error) = crate::models_repo::rollback(prev) {
                    tracing::error!(error = %rollback_error, "model rollback after mapper reload failure failed");
                }
            }
            anyhow::bail!("reload cleave capability mapper: {error}");
        }
    }

    Ok(Some(resources))
}

fn current_resources(handle: &ResourceHandle) -> Result<Arc<ModelResources>> {
    let guard = handle
        .read()
        .map_err(|error| anyhow::anyhow!("worker resources lock poisoned: {error}"))?;
    Ok(Arc::clone(&guard))
}

fn spawn_resource_renewal_task(
    handle: ResourceHandle,
    model_dir: PathBuf,
    thresholds: Option<Thresholds>,
    slow_rule_ms: u64,
    level: Option<u8>,
    shutdown: Arc<AtomicBool>,
) {
    tokio::spawn(async move {
        loop {
            interruptible_sleep(RESOURCE_RENEWAL_INTERVAL, &shutdown).await;
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            tracing::debug!(
                interval_secs = RESOURCE_RENEWAL_INTERVAL.as_secs(),
                "worker resource renewal check starting",
            );
            let model_dir = model_dir.clone();
            let result = tokio::task::spawn_blocking(move || {
                renew_resources_once(&model_dir, thresholds, slow_rule_ms, level)
            })
            .await;

            let new_resources = match result {
                Ok(Ok(Some(resources))) => resources,
                Ok(Ok(None)) => {
                    // Nothing changed upstream — silent no-op.
                    continue;
                }
                Ok(Err(error)) => {
                    tracing::error!(error = %error, "worker resource renewal failed; keeping last-known-good resources");
                    continue;
                }
                Err(error) => {
                    tracing::error!(error = %error, "worker resource renewal task panicked; keeping last-known-good resources");
                    continue;
                }
            };

            let spec_version = new_resources.model.spec().version();
            let features = new_resources.model.spec().total_features();
            match handle.write() {
                Ok(mut guard) => {
                    *guard = new_resources;
                    tracing::info!(spec_version, features, "worker resources renewed");
                }
                Err(error) => {
                    tracing::error!(error = %error, "worker resources lock poisoned; renewal discarded");
                }
            }
        }
    });
}

/// A fixed grid of dedicated rayon thread pools — one per worker slot.
///
/// With N concurrent analyses competing for a single rayon pool the work-
/// stealing scheduler piles joins from unrelated analyses onto the same
/// thread; each thread ends up holding a deep stack of half-completed jobs
/// from every other caller and effective parallelism collapses. Giving each
/// worker slot its own isolated pool eliminates that cross-contamination:
/// an analysis's `par_iter` fan-out only ever touches its own pool.
///
/// The tokio semaphore already guarantees at most `slots` concurrent
/// analyses, so the free-list can never be starved in practice. The
/// condvar is kept as a cheap safety net.
pub(crate) struct WorkerPools {
    free: Mutex<Vec<(usize, Arc<rayon::ThreadPool>)>>,
    available: Condvar,
}

impl WorkerPools {
    /// Build `slots` pools, each sized `threads_per_slot`.
    ///
    /// Panics if rayon pool construction fails (OOM at startup).
    #[allow(clippy::expect_used)]
    pub(crate) fn new(slots: usize, threads_per_slot: usize) -> Arc<Self> {
        let slots = slots.max(1);
        let threads_per_slot = threads_per_slot.max(1);
        let free: Vec<(usize, Arc<rayon::ThreadPool>)> = (0..slots)
            .map(|i| {
                let p = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads_per_slot)
                    .thread_name(move |j| format!("cleave-{i}-{j}"))
                    .stack_size(64 * 1024 * 1024)
                    .build()
                    .expect("build cleave worker pool");
                (i, Arc::new(p))
            })
            .collect();
        Arc::new(Self {
            free: Mutex::new(free),
            available: Condvar::new(),
        })
    }

    /// Run `f` on a pool from the free-list, passing the slot index so callers
    /// can correlate logs with the `cleave-N-M` rayon thread names visible in
    /// stack samples.
    ///
    /// Panics if the pool mutex is poisoned (another worker panicked while
    /// holding the lock) — in that case the process state is unrecoverable.
    #[allow(clippy::expect_used)]
    pub(crate) fn install<T, F>(&self, f: F) -> T
    where
        F: FnOnce(usize) -> T + Send,
        T: Send,
    {
        let (slot, pool) = {
            let mut g = self.free.lock().expect("worker pool mutex poisoned");
            while g.is_empty() {
                g = self
                    .available
                    .wait(g)
                    .expect("worker pool condvar poisoned");
            }
            g.pop().expect("non-empty checked above")
        };
        let result = pool.install(|| f(slot));
        self.free
            .lock()
            .expect("worker pool mutex poisoned")
            .push((slot, pool));
        self.available.notify_one();
        result
    }
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
    // Typed `MlSection` instead of `serde_json::Value` so serialization walks the
    // struct once into HTTP body bytes; the prior shape allocated an intermediate
    // Value tree via `serde_json::to_value(&envelope.ml)` on every result post.
    #[serde(skip_serializing_if = "Option::is_none")]
    ml: Option<crate::scan::MlSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    duration_ms: i64,
}

/// A job with its file data pre-downloaded (or marked for local access).
struct PrefetchedJob {
    job: ClaimJob,
    /// `Ok(None)` = use local file, `Ok(Some(bytes))` = downloaded,
    /// `Err(Transient)` = download failed (fall back to direct download),
    /// `Err(Skipped)` = job rejected without attempting download (e.g. oversized);
    /// do not retry, post the error result directly.
    data: std::result::Result<Option<Vec<u8>>, PrefetchError>,
}

/// Why a prefetch did not produce bytes.
#[derive(Debug, Clone)]
enum PrefetchError {
    /// Download attempted and failed — `run_job` may retry via direct download.
    Transient(String),
    /// Download not attempted; treat as a permanent error for this worker.
    Skipped(String),
}

impl std::fmt::Display for PrefetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transient(m) | Self::Skipped(m) => f.write_str(m),
        }
    }
}

/// Upper bound on how long `run` will wait for in-flight analyses to drain
/// after a shutdown signal before exiting anyway. Cleave cancellation is
/// cooperative; a stuck rayon unpack can refuse to exit, and the operator
/// should not have to `kill -9` just because one file is wedged.
const SHUTDOWN_DRAIN_SECS: u64 = 60;

/// Poll the shutdown flag at ≤500 ms granularity so a signal interrupts any
/// sleep the main loop is parked in (no-work backoff, memory-pressure pause,
/// dispatch idle). Polling rather than `Notify` keeps the call sites simple
/// and avoids plumbing an extra Arc through every branch.
async fn interruptible_sleep(duration: Duration, shutdown: &AtomicBool) {
    let deadline = Instant::now() + duration;
    while !shutdown.load(Ordering::Relaxed) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return;
        }
        tokio::time::sleep(remaining.min(Duration::from_millis(500))).await;
    }
}

/// Spawn a task that flips `shutdown` when SIGINT, SIGTERM (unix), or Ctrl-C
/// (other platforms) arrive. Registration failures are logged, not fatal —
/// better to run without graceful shutdown than to refuse to start.
fn install_shutdown_handler(shutdown: Arc<AtomicBool>) {
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let sigterm = signal(SignalKind::terminate());
            let sigint = signal(SignalKind::interrupt());
            let (mut sigterm, mut sigint) = match (sigterm, sigint) {
                (Ok(t), Ok(i)) => (t, i),
                (Err(e), _) | (_, Err(e)) => {
                    tracing::warn!(error = %e, "failed to install signal handler; graceful shutdown disabled");
                    return;
                }
            };
            tokio::select! {
                _ = sigterm.recv() => tracing::info!("received SIGTERM, starting graceful shutdown"),
                _ = sigint.recv()  => tracing::info!("received SIGINT, starting graceful shutdown"),
            }
        }
        #[cfg(not(unix))]
        {
            if let Err(e) = tokio::signal::ctrl_c().await {
                tracing::warn!(error = %e, "ctrl_c handler failed; graceful shutdown disabled");
                return;
            }
            tracing::info!("received Ctrl-C, starting graceful shutdown");
        }
        shutdown.store(true, Ordering::Release);
    });
}

/// Apply a nice value to the current process. A no-op when `nice == 0`.
/// `setpriority` failure is logged but never fatal — an unprivileged process
/// cannot lower its nice value, and we'd rather run at the inherited priority
/// than refuse to start.
fn apply_nice(nice: i32) {
    if nice == 0 {
        return;
    }
    // SAFETY: setpriority(PRIO_PROCESS, 0, ...) targets the calling process
    // and has no memory effects. PRIO_PROCESS is POSIX; pid 0 means "self".
    let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, nice) };
    if rc == 0 {
        tracing::info!(nice, "set worker process nice value");
    } else {
        let err = std::io::Error::last_os_error();
        tracing::warn!(nice, error = %err, "setpriority failed; continuing at inherited priority");
    }
}

/// Run the worker loop. Blocks until cancelled.
pub async fn run(config: WorkerConfig) -> Result<()> {
    apply_nice(config.nice);
    // Arc<str> so every per-job dispatch clones an atomic refcount rather than
    // reallocating the worker name for each `tokio::spawn`.
    let name: Arc<str> = Arc::from(config.name.as_str());
    let slots = config.workers;
    // 120 s per request is long enough for cold cleave scans yet short enough
    // that a wedged hopper can't pin the worker indefinitely — without a
    // timeout the default is "no timeout", which defeats graceful shutdown.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let semaphore = Arc::new(Semaphore::new(slots));
    let shutdown = Arc::new(AtomicBool::new(false));
    install_shutdown_handler(Arc::clone(&shutdown));

    // Each slot's rayon pool must have enough threads to absorb cleave's
    // nested archive `par_iter` without starvation. With only 2 threads,
    // recursive tar→tar→member analysis commits both workers to depth-first
    // joins and the outer join has no thread left to reap — a true deadlock
    // that work-stealing cannot escape. 4 threads gives enough headroom for
    // the nested-join chain to always have a stealable worker.
    //
    // This trades some throughput (pools × 4 can exceed `num_cpus`) for
    // deadlock safety. The alternative — flattening cleave's archive
    // recursion to a single top-level par_iter — is the durable fix but
    // lives upstream.
    const MIN_THREADS_PER_SLOT: usize = 4;
    let threads_per_slot = (rayon::current_num_threads() / slots.max(1)).max(MIN_THREADS_PER_SLOT);
    let pools = WorkerPools::new(slots, threads_per_slot);

    tracing::info!(
        name = %name,
        slots = slots,
        threads_per_slot = threads_per_slot.max(1),
        hopper = %config.hopper_url,
        global_rayon_threads = rayon::current_num_threads(),
        pid = std::process::id(),
        "worker starting; send `kill -USR1 <pid>` for an all-thread backtrace",
    );

    // Start background rayon pool health monitoring.
    cleave::start_rayon_diagnostics();

    // Warm YARA + capability mapper on a non-rayon thread before any job is
    // dispatched. The variant (`true`) must match `AnalysisOptions::default()`
    // — otherwise the prefetch warms an engine nobody uses and the first real
    // analysis triggers a cold compile on a rayon worker, which deadlocks the
    // pool. See cleave::shared_resources::yara_engine for the contract.
    cleave::prefetch_shared_resources(true);

    let resources = load_model_resources(
        &config.model_dir,
        config.thresholds,
        config.level,
    )?;
    let resources: ResourceHandle = Arc::new(RwLock::new(resources));
    spawn_resource_renewal_task(
        Arc::clone(&resources),
        config.model_dir.clone(),
        config.thresholds,
        config.slow_rule_ms,
        config.level,
        Arc::clone(&shutdown),
    );

    // Arc<str> for the hopper URL — cloned per prefetched job and per dispatched
    // analysis; an atomic bump is far cheaper than a String reallocation.
    let base_url: Arc<str> = Arc::from(config.hopper_url.trim_end_matches('/'));
    let data_dir = config.data_dir.clone();
    let local_index = data_dir
        .clone()
        .map(LocalFileIndex::build)
        .transpose()
        .context("building local data index")?
        .map(Arc::new);
    let poll_secs = config.poll_secs;
    let slow_rule_ms = config.slow_rule_ms;
    let max_jobs = config.max_jobs;
    let max_rss_gb = config.max_rss_gb;
    let encoded_name: String = url_encode(&name);
    let available_tools = crate::tools::available_names().join(",");
    let mut consecutive_errors: u32 = 0;
    let completed = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let mut buffer: VecDeque<PrefetchedJob> = VecDeque::new();
    let mut buffer_bytes: usize = 0; // track memory held by prefetched data
    let max_buffer_bytes: usize =
        if cleave::memory_tracker::total_memory().unwrap_or(0) >= 16 * 1024 * 1024 * 1024 {
            1024 * 1024 * 1024 // 1 GiB on systems with >= 16 GiB RAM
        } else {
            512 * 1024 * 1024 // 512 MiB otherwise
        };
    // With local data, there's no download latency to hide — just claim
    // what we can run immediately. Without local data, prefetch 3x slots
    // so downloads overlap with analysis.
    let has_local_data = data_dir.is_some();
    let prefetch_count = if has_local_data {
        slots
    } else {
        (slots * 3).min(32)
    };
    let mut last_empty_poll = Instant::now() - Duration::from_secs(poll_secs + 1);
    let mut last_summary = Instant::now();

    loop {
        if shutdown.load(Ordering::Relaxed) {
            tracing::info!(
                buffered_jobs = buffer.len(),
                buffered_bytes = buffer_bytes,
                "shutdown signalled, draining in-flight work",
            );
            break;
        }

        if last_summary.elapsed() >= Duration::from_secs(60) {
            let started = BLOCKING_STARTED_TOTAL.load(Ordering::Relaxed);
            let finished = BLOCKING_FINISHED_TOTAL.load(Ordering::Relaxed);
            let inflight_blocking = started.saturating_sub(finished);
            let available_slots = semaphore.available_permits();
            let active_slots = slots.saturating_sub(available_slots);
            tracing::info!(
                rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / 1024 / 1024),
                queued_prefetch_jobs = buffer.len(),
                prefetch_buffer_mb = buffer_bytes / (1024 * 1024),
                active_slots,
                available_slots,
                blocking_started_total = started,
                blocking_finished_total = finished,
                inflight_blocking,
                completed = completed.load(Ordering::Acquire),
                "worker summary",
            );
            last_summary = Instant::now();
        }

        if let Some(max) = max_jobs
            && completed.load(Ordering::Acquire) >= max
        {
            tracing::info!(max_jobs = max, "job limit reached, draining in-flight work");
            break;
        }

        // Enforce memory limit before claiming more work.
        if max_rss_gb > 0 {
            let max_bytes = max_rss_gb.saturating_mul(1024 * 1024 * 1024);
            if let Some(rss) = cleave::memory_tracker::current_rss()
                && rss > max_bytes
            {
                tracing::warn!(
                    rss_mb = rss / 1024 / 1024,
                    max_rss_mb = max_bytes / 1024 / 1024,
                    "memory pressure: pausing before claiming new work",
                );
                if let Err(e) =
                    tokio::task::spawn_blocking(cleave::clear_all_thread_caches).await
                {
                    tracing::warn!(error = %e, "cache-clear task failed");
                }
                interruptible_sleep(Duration::from_secs(poll_secs), &shutdown).await;
                continue;
            }
        }

        // Refill the prefetch buffer when it's running low. Rate-limit polls
        // so we don't hammer hopper when it has no work.
        let should_poll = buffer.len() < slots
            && buffer_bytes < max_buffer_bytes
            && last_empty_poll.elapsed() >= Duration::from_secs(poll_secs);
        if should_poll {
            let mut poll_url = format!(
                "{}/api/next?worker={}&count={}&slots={}&version={}",
                base_url,
                encoded_name,
                prefetch_count,
                slots,
                env!("CARGO_PKG_VERSION"),
            );
            {
                use std::fmt::Write;
                // 5-char prefix matches hopper's litmusTraitsVersion()
                // truncation so the dashboard's stale-traits comparison
                // can string-equal the two.
                if let Some(traits) = cleave::traits_repo::version() {
                    let prefix: String = traits.chars().take(5).collect();
                    let _ = write!(poll_url, "&traits={}", prefix);
                }
                if let Some(rss) = cleave::memory_tracker::current_rss() {
                    let _ = write!(poll_url, "&rss_mb={}", rss / 1024 / 1024);
                }
                if let Some(load) = system_load_avg() {
                    let _ = write!(poll_url, "&load1={:.2}", load);
                }
                let _ = write!(poll_url, "&tools=");
                url_encode_into(&available_tools, &mut poll_url);
            }
            tracing::debug!(url = %poll_url, buffer = buffer.len(), "polling for work");
            let poll_start = Instant::now();

            // Cap the per-job download size so a single outsized payload can't
            // blow past the buffer budget. Pre-filtering on hopper's
            // `size_bytes` lets us reject without touching the network; jobs
            // without a size still download but are bounded by the client
            // timeout and the outer `buffer_bytes` gate.
            let max_single_bytes = max_buffer_bytes / 2;
            match claim_and_prefetch(
                &client,
                &poll_url,
                &base_url,
                data_dir.as_deref(),
                max_single_bytes,
            )
            .await
            {
                Ok(None) => {
                    consecutive_errors = 0;
                    last_empty_poll = Instant::now();
                    if buffer.is_empty() {
                        tracing::debug!(
                            elapsed_ms = crate::duration_ms(poll_start.elapsed()),
                            poll_secs,
                            "no work available, sleeping"
                        );
                        interruptible_sleep(Duration::from_secs(poll_secs), &shutdown).await;
                        continue;
                    }
                    // Buffer has work — keep dispatching.
                }
                Ok(Some(jobs)) => {
                    consecutive_errors = 0;
                    let n = jobs.len();
                    for pj in jobs {
                        buffer_bytes += pj
                            .data
                            .as_ref()
                            .map_or(0, |d| d.as_ref().map_or(0, Vec::len));
                        buffer.push_back(pj);
                    }
                    tracing::debug!(
                        jobs = n,
                        buffer_mb = buffer_bytes / (1024 * 1024),
                        elapsed_ms = crate::duration_ms(poll_start.elapsed()),
                        "claimed and prefetched",
                    );
                }
                Err(e) => {
                    consecutive_errors += 1;
                    let backoff = backoff_duration(consecutive_errors);
                    let error_chain = format!("{e:#}");
                    tracing::warn!(
                        url = %poll_url,
                        error = %error_chain,
                        error_debug = ?e,
                        elapsed_ms = crate::duration_ms(poll_start.elapsed()),
                        backoff_secs = backoff.as_secs(),
                        consecutive_errors,
                        "poll/prefetch failed",
                    );
                    if buffer.is_empty() {
                        interruptible_sleep(backoff, &shutdown).await;
                        continue;
                    }
                    // Buffer has work — keep dispatching despite poll failure.
                }
            }
        }

        // Dispatch the next prefetched job when a slot is free.
        let pj = match buffer.pop_front() {
            Some(pj) => {
                buffer_bytes -= pj
                    .data
                    .as_ref()
                    .map_or(0, |d| d.as_ref().map_or(0, Vec::len));
                pj
            }
            None => {
                // Buffer empty and poll rate-limited — wait before retrying.
                interruptible_sleep(Duration::from_millis(100), &shutdown).await;
                continue;
            }
        };

        // Wait for a free analysis slot. Also check memory pressure after
        // acquiring the permit — the RSS check at the top of the loop only
        // runs before claiming, but memory can grow while waiting for a slot.
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .context("semaphore closed")?;

        if max_rss_gb > 0 {
            let max_bytes = max_rss_gb.saturating_mul(1024 * 1024 * 1024);
            if let Some(rss) = cleave::memory_tracker::current_rss()
                && rss > max_bytes
            {
                tracing::warn!(
                    rss_mb = rss / 1024 / 1024,
                    max_rss_mb = max_bytes / 1024 / 1024,
                    "memory pressure before dispatch: clearing caches and pausing",
                );
                drop(permit);
                if let Err(e) =
                    tokio::task::spawn_blocking(cleave::clear_all_thread_caches).await
                {
                    tracing::warn!(error = %e, "cache-clear task failed");
                }
                interruptible_sleep(Duration::from_secs(poll_secs), &shutdown).await;
                // Put the job back at the front of the buffer.
                buffer_bytes += pj
                    .data
                    .as_ref()
                    .map_or(0, |d| d.as_ref().map_or(0, Vec::len));
                buffer.push_front(pj);
                continue;
            }
        }

        let client = client.clone();
        let resources = match current_resources(&resources) {
            Ok(resources) => resources,
            Err(error) => {
                tracing::error!(error = %error, "cannot snapshot worker resources; pausing");
                drop(permit);
                buffer_bytes += pj
                    .data
                    .as_ref()
                    .map_or(0, |d| d.as_ref().map_or(0, Vec::len));
                buffer.push_front(pj);
                interruptible_sleep(Duration::from_secs(poll_secs), &shutdown).await;
                continue;
            }
        };
        let url = Arc::clone(&base_url);
        let name = Arc::clone(&name);
        let local_index = local_index.clone();
        let completed = Arc::clone(&completed);
        let pools = Arc::clone(&pools);

        tokio::spawn(async move {
            let result = run_job(
                &client,
                &url,
                local_index.as_deref(),
                &pj.job,
                &resources,
                slow_rule_ms,
                pj.data,
                &pools,
            )
            .await;
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
            if n.is_multiple_of(100) {
                tokio::task::spawn_blocking(cleave::clear_all_thread_caches);
            }
            drop(permit);
        });
    }

    // Drain any in-flight work before exiting, but cap the wait so a stuck
    // cleave unpack can't indefinitely block shutdown. Jobs that haven't
    // finished by the timeout are still alive on the tokio blocking pool;
    // they'll complete (or be killed with the process) at shutdown time.
    let slot_count = u32::try_from(slots).unwrap_or(u32::MAX);
    let drain = semaphore.acquire_many(slot_count);
    match tokio::time::timeout(Duration::from_secs(SHUTDOWN_DRAIN_SECS), drain).await {
        Ok(_) => tracing::info!("all in-flight jobs finished, exiting"),
        Err(_) => {
            let still_running = slots.saturating_sub(semaphore.available_permits());
            tracing::warn!(
                still_running,
                drain_secs = SHUTDOWN_DRAIN_SECS,
                "drain timeout reached, exiting with in-flight analyses still running",
            );
        }
    }
    Ok(())
}

/// Claim jobs from hopper and prefetch file data for all of them concurrently.
/// Returns `Ok(None)` if no work is available (HTTP 204).
///
/// Jobs whose hopper-reported size exceeds `max_single_bytes` are not
/// downloaded — they're returned with an oversize error so the result posts
/// back to hopper immediately. This prevents a pathological single file from
/// blowing past the worker's prefetch memory budget.
async fn claim_and_prefetch(
    client: &reqwest::Client,
    poll_url: &str,
    base_url: &Arc<str>,
    data_dir: Option<&Path>,
    max_single_bytes: usize,
) -> Result<Option<Vec<PrefetchedJob>>> {
    let resp = client.get(poll_url).send().await.map_err(|e| {
        let error_text = e.to_string();
        let is_connect = e.is_connect();
        anyhow::Error::new(e).context(poll_request_context(poll_url, &error_text, is_connect))
    })?;

    if resp.status() == reqwest::StatusCode::NO_CONTENT {
        return Ok(None);
    }
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!(
            "poll request returned non-success: url={poll_url} status={status} body={}",
            body_excerpt(&body),
        );
    }

    let resp_body = resp
        .text()
        .await
        .with_context(|| format!("read claim body: url={poll_url}"))?;
    let claim: ClaimResponse = serde_json::from_str(&resp_body).with_context(|| {
        format!(
            "parse claim response: url={poll_url} body={}",
            body_excerpt(&resp_body),
        )
    })?;

    if claim.jobs.is_empty() {
        return Ok(None);
    }

    // Prefetch all files concurrently, but short-circuit any job whose reported
    // size exceeds `max_single_bytes` — those are dispatched as error results
    // rather than downloaded, so a 5 GiB outlier can't OOM the buffer.
    let mut set = tokio::task::JoinSet::new();
    let mut oversized: Vec<PrefetchedJob> = Vec::new();
    for job in claim.jobs {
        if u64::try_from(job.size_bytes).is_ok_and(|s| s > max_single_bytes as u64) {
            tracing::warn!(
                sha256 = %job.sha256,
                path = %job.path,
                size_bytes = job.size_bytes,
                max_single_bytes,
                "skipping oversized job; reporting error to hopper",
            );
            let err = PrefetchError::Skipped(format!(
                "file size {} exceeds per-job prefetch cap of {} bytes",
                job.size_bytes, max_single_bytes,
            ));
            oversized.push(PrefetchedJob {
                job,
                data: Err(err),
            });
            continue;
        }
        let client = client.clone();
        let base_url = Arc::clone(base_url);
        let data_dir = data_dir.map(Path::to_path_buf);
        set.spawn(async move {
            let local_path = data_dir.as_deref().map(|d| d.join(&job.path));
            let use_local = matches!(local_path, Some(ref p) if p.exists());
            let data = if use_local {
                Ok(None)
            } else {
                match download_bytes(&client, &base_url, &job.sha256, &job.path).await {
                    Ok(bytes) => Ok(Some(bytes)),
                    Err(e) => Err(PrefetchError::Transient(e)),
                }
            };
            PrefetchedJob { job, data }
        });
    }

    let mut prefetched = Vec::with_capacity(set.len() + oversized.len());
    prefetched.append(&mut oversized);
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
#[allow(clippy::too_many_arguments)]
async fn run_job(
    client: &reqwest::Client,
    base_url: &str,
    local_index: Option<&LocalFileIndex>,
    job: &ClaimJob,
    resources: &Arc<ModelResources>,
    slow_rule_ms: u64,
    prefetched: std::result::Result<Option<Vec<u8>>, PrefetchError>,
    pools: &Arc<WorkerPools>,
) -> Result<(crate::scan::MlSection, serde_json::Value, i64), String> {
    let analysis_id = NEXT_ANALYSIS_ID.fetch_add(1, Ordering::Relaxed);
    // `Arc<str>` so the watcher and the blocking closure share the basename
    // allocation instead of each cloning a fresh `String`.
    let label: Arc<str> = Path::new(&job.path)
        .file_name()
        .map(|n| Arc::from(n.to_string_lossy().as_ref()))
        .unwrap_or_else(|| Arc::from(job.sha256.as_str()));

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
            tracing::debug!(
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
                tracing::debug!(sha256 = %job.sha256, file = %label, size = bytes.len(), "using prefetched data");
                Some(bytes)
            }
            Ok(None) => None, // shouldn't happen for remote jobs, but handle gracefully
            Err(PrefetchError::Skipped(msg)) => {
                // Prefetch layer decided not to download this job (e.g. oversized);
                // fail the analysis immediately rather than retrying the fetch.
                return Err(msg);
            }
            Err(PrefetchError::Transient(e)) => {
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
    let sha_short: Arc<str> = Arc::from(job.sha256.get(..12).unwrap_or(&job.sha256));
    // Register the tracker with a descriptive label so cleave's rayon-diag
    // snapshot can name which analyses are in flight instead of just
    // reporting a count.
    let phase = cleave::PhaseTracker::with_label(format!("{sha_short} {label}"));
    let phase2 = phase.clone();
    let label2 = Arc::clone(&label);
    let label_for_blocking = Arc::clone(&label);
    // Pre-clone for the blocking closure before the watcher captures its copy.
    let sha_short2 = Arc::clone(&sha_short);
    let input_source = if use_local { "local" } else { "downloaded" };
    let input_size = if use_local {
        u64::try_from(job.size_bytes).unwrap_or(0)
    } else {
        downloaded.as_ref().map_or(0, |bytes| bytes.len() as u64)
    };

    // Background phase watcher — logs transitions with timing, and emits a
    // heartbeat every 30 s so a stuck phase is visible in logs.
    // Uses a tokio task instead of an OS thread to avoid one thread-per-job overhead.
    // The returned JoinHandle is aborted via RAII guard below so the watcher cannot
    // outlive this function even if the outer task is cancelled.
    let watcher_handle = tokio::task::spawn(async move {
        let mut last_phase = String::new();
        let mut phase_start = Instant::now();
        let mut slow_logged = false;
        let mut very_slow_logged = false;
        // 500 ms polling is fine-grained enough for the 60 s / 180 s slow-phase
        // thresholds below, and at 32 slots × 2 Hz the scheduler cost is a
        // fifth of the old 10 Hz poll.
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            let current = phase2.get();
            if current.is_empty() {
                // Phase tracker not yet updated — only surface this once it is
                // materially slow at the default log level.
                let elapsed = phase_start.elapsed();
                if elapsed.as_secs() >= 180 && !very_slow_logged {
                    tracing::warn!(
                        analysis_id,
                        sha256 = %sha_short,
                        file = %label2,
                        rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / 1024 / 1024),
                        elapsed_ms = crate::duration_ms(elapsed),
                        pid = std::process::id(),
                        "analysis running without phase updates for a very slow interval; \
                         send `kill -USR1 <pid>` for an all-thread backtrace",
                    );
                    very_slow_logged = true;
                } else if elapsed.as_secs() >= 60 && !slow_logged {
                    tracing::info!(
                        analysis_id,
                        sha256 = %sha_short,
                        file = %label2,
                        rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / 1024 / 1024),
                        elapsed_ms = crate::duration_ms(elapsed),
                        "analysis running without phase updates for a slow interval",
                    );
                    slow_logged = true;
                }
                continue;
            }
            if current != last_phase {
                if !last_phase.is_empty() {
                    tracing::debug!(
                        analysis_id,
                        sha256 = %sha_short,
                        file = %label2,
                        phase = %last_phase,
                        elapsed_ms = crate::duration_ms(phase_start.elapsed()),
                        "phase complete",
                    );
                }
                last_phase = current;
                phase_start = Instant::now();
                slow_logged = false;
                very_slow_logged = false;
                tracing::debug!(
                    analysis_id,
                    sha256 = %sha_short,
                    file = %label2,
                    phase = %last_phase,
                    "phase started",
                );
                if last_phase == "done" {
                    break;
                }
            } else {
                let elapsed = phase_start.elapsed();
                if elapsed.as_secs() >= 180 && !very_slow_logged {
                    tracing::warn!(
                        analysis_id,
                        sha256 = %sha_short,
                        file = %label2,
                        phase = %last_phase,
                        rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / 1024 / 1024),
                        elapsed_ms = crate::duration_ms(elapsed),
                        pid = std::process::id(),
                        "very slow phase; send `kill -USR1 <pid>` for an all-thread backtrace",
                    );
                    very_slow_logged = true;
                } else if elapsed.as_secs() >= 60 && !slow_logged {
                    tracing::info!(
                        analysis_id,
                        sha256 = %sha_short,
                        file = %label2,
                        phase = %last_phase,
                        rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / 1024 / 1024),
                        elapsed_ms = crate::duration_ms(elapsed),
                        "slow phase",
                    );
                    slow_logged = true;
                }
            }
        }
    });

    // RAII guard: aborts the watcher task whenever this function scope exits,
    // even if the outer tokio task was cancelled. Without it, a watcher whose
    // parent was dropped before the analysis returned would spin forever on its
    // 100 ms sleep loop.
    struct WatcherGuard(Option<tokio::task::JoinHandle<()>>);
    impl Drop for WatcherGuard {
        fn drop(&mut self) {
            if let Some(h) = self.0.take() {
                h.abort();
            }
        }
    }
    let _watcher_guard = WatcherGuard(Some(watcher_handle));

    tracing::debug!(
        analysis_id,
        sha256 = %job.sha256.get(..12).unwrap_or(&job.sha256),
        file = %label,
        source = input_source,
        size = input_size,
        "analysis starting",
    );
    let pools_for_blocking = Arc::clone(pools);
    let handle = tokio::task::spawn_blocking(move || {
        // Run the whole analysis inside this slot's dedicated rayon pool so
        // any `par_iter` fan-out stays local to this analysis and can't pile
        // joins onto sibling workers' stacks. Lifecycle logs are emitted from
        // INSIDE the install closure so the `thread_id` they report is the
        // rayon worker (cleave-N-M), the thread an operator should sample to
        // diagnose a wedged analysis. Sampling the outer tokio blocking
        // thread would just show it parked in a rayon `LockLatch`.
        pools_for_blocking.install(|slot| {
            let started = BLOCKING_STARTED_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
            let thread_id = os_thread_id();
            let inflight_blocking =
                started.saturating_sub(BLOCKING_FINISHED_TOTAL.load(Ordering::Relaxed));
            tracing::info!(
                analysis_id,
                sha256 = %sha_short2,
                file = %label_for_blocking,
                slot,
                thread_id,
                inflight_blocking,
                started_total = started,
                rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / 1024 / 1024),
                "analysis starting on rayon slot",
            );
            let result = if let Some(data) = downloaded {
                classify_bytes(
                    data,
                    &label_for_blocking,
                    &resources,
                    slow_rule_ms,
                    Some(&cancel2),
                    Some(&phase),
                )
            } else if let Some(path) = local.as_ref() {
                classify_file(
                    path,
                    &label_for_blocking,
                    &resources,
                    slow_rule_ms,
                    None,
                    Some(&cancel2),
                    Some(&phase),
                )
            } else {
                Err(anyhow::anyhow!(
                    "no downloaded bytes and no local path for {label_for_blocking}"
                ))
            };
            let finished = BLOCKING_FINISHED_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
            let inflight_blocking = BLOCKING_STARTED_TOTAL
                .load(Ordering::Relaxed)
                .saturating_sub(finished);
            tracing::debug!(
                analysis_id,
                sha256 = %sha_short2,
                slot,
                thread_id,
                inflight_blocking,
                finished_total = finished,
                rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / 1024 / 1024),
                elapsed_ms = crate::duration_ms(start.elapsed()),
                "analysis complete on rayon slot",
            );
            result
        })
    });

    let result = handle.await;

    #[allow(clippy::cast_sign_loss)]
    let elapsed_ms = crate::duration_ms(start.elapsed()) as i64;

    match result {
        Ok(Ok(scan_result)) => {
            let envelope = scan_result.into_envelope();
            Ok((envelope.ml, envelope.raw, elapsed_ms))
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
    result: Result<(crate::scan::MlSection, serde_json::Value, i64), String>,
) {
    let payload = match result {
        Ok((ml, raw, duration_ms)) => {
            let classification = match ml.classification {
                crate::model::Classification::Benign => "benign",
                crate::model::Classification::Suspicious => "suspicious",
                crate::model::Classification::Hostile => "hostile",
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
        Err(e) => ResultPayload {
            sha256: sha256.to_string(),
            worker: worker.to_string(),
            ml: None,
            raw: None,
            error: Some(e),
            duration_ms: 0,
        },
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

fn body_excerpt(body: &str) -> String {
    const MAX: usize = 512;
    let compact = body.replace(['\r', '\n', '\t'], " ");
    let mut out: String = compact.chars().take(MAX).collect();
    if compact.chars().count() > MAX {
        out.push_str("...");
    }
    out
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
    if path.is_empty() || path == "." {
        return Err(format!(
            "download {sha256}: empty path from hopper, cannot fetch"
        ));
    }

    let start = Instant::now();

    // Use path-based endpoint (static file serving, no DB query on hopper side).
    // Encode each path segment to handle filenames with spaces or special chars.
    // Build the URL in one pass — avoids the intermediate `Vec<String>` the
    // previous `split().map().collect().join()` chain allocated per download.
    let mut data_url = String::with_capacity(base_url.len() + 6 + path.len() * 2);
    data_url.push_str(base_url);
    data_url.push_str("/data/");
    let mut first = true;
    for segment in path.split('/') {
        if !first {
            data_url.push('/');
        }
        first = false;
        url_encode_into(segment, &mut data_url);
    }
    tracing::debug!(sha256 = %sha256, url = %data_url, "downloading via /data/");
    let resp =
        client.get(&data_url).send().await.map_err(|e| {
            format!("download failed: path={path} sha256={sha256} url={data_url}: {e}")
        })?;

    if resp.status().is_success() {
        let bytes = resp.bytes().await.map_err(|e| {
            format!("download body failed: path={path} sha256={sha256} url={data_url}: {e}")
        })?;
        tracing::info!(
            sha256 = %sha256,
            file = %path,
            bytes = bytes.len(),
            elapsed_ms = crate::duration_ms(start.elapsed()),
            "download complete via /data/",
        );
        return Ok(bytes.to_vec());
    }
    let data_status = resp.status();
    let data_body = resp
        .text()
        .await
        .map(|body| body_excerpt(&body))
        .unwrap_or_else(|e| format!("failed to read error body: {e}"));

    // /data/ failed — fall back to /api/file/{sha256} which does a DB lookup
    // by hash, so it works even when the relative path doesn't match hopper's
    // data root (e.g. different symlink resolution or data root migration).
    let api_url = format!("{base_url}/api/file/{sha256}");
    tracing::debug!(sha256 = %sha256, url = %api_url, "downloading via /api/file/ (fallback)");
    let resp = client.get(&api_url).send().await.map_err(|e| {
        format!("download fallback failed: path={path} sha256={sha256} url={api_url}: {e}")
    })?;

    if !resp.status().is_success() {
        let api_status = resp.status();
        let api_body = resp
            .text()
            .await
            .map(|body| body_excerpt(&body))
            .unwrap_or_else(|e| format!("failed to read error body: {e}"));
        return Err(format!(
            "download failed: path={path} sha256={sha256}; /data/ url={data_url} status={data_status} body={data_body}; /api/file/ url={api_url} status={api_status} body={api_body}",
        ));
    }
    let bytes = resp.bytes().await.map_err(|e| {
        format!("download fallback body failed: path={path} sha256={sha256} url={api_url}: {e}")
    })?;
    tracing::info!(
        sha256 = %sha256,
        file = %path,
        bytes = bytes.len(),
        elapsed_ms = crate::duration_ms(start.elapsed()),
        "download complete via /api/file/ (fallback)",
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
        if ret == 1 {
            Some(avg[0])
        } else {
            None
        }
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

/// Percent-encode a string for use in URL query parameters.
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    url_encode_into(s, &mut out);
    out
}

/// Append the percent-encoded form of `s` to `out`. Lets callers that build up
/// a URL piece-by-piece skip the per-segment `String` allocations that
/// `url_encode` would otherwise require.
fn url_encode_into(s: &str, out: &mut String) {
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
}

fn poll_request_context(url: &str, error_text: &str, is_connect: bool) -> String {
    let mut context = format!("poll request failed: url={url}");
    if is_connect && url.starts_with("https://") && error_text.contains("InvalidContentType") {
        context.push_str(
            " (HTTPS requested, but the peer did not speak TLS; hopper may be serving plain HTTP on this port. Try http://)",
        );
    }
    context
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

    /// Files added after the index was built should still be found via the
    /// disk fallback (relative path + SHA-256 verification).
    #[test]
    fn disk_fallback_resolves_file_added_after_index_build() {
        let root = tempfile::tempdir().expect("create temp dir");
        // Build index with an empty root — no files indexed.
        let index = LocalFileIndex::build(root.path().to_path_buf()).expect("build index");

        // Now add a file after the index was built.
        let rel = Path::new("unknown/harvest/new/crates/newpkg-1.0.crate");
        let bytes = b"newly-harvested-crate";
        write_file(&root.path().join(rel), bytes);

        let resolved = index
            .resolve(
                rel.to_str().expect("utf8"),
                &sha256_hex(bytes),
                i64::try_from(bytes.len()).expect("len fits"),
            )
            .expect("resolve path");

        assert_eq!(resolved.as_deref(), Some(root.path().join(rel).as_path()));
    }

    /// The disk fallback should reject a file whose SHA-256 doesn't match,
    /// even when the path exists on disk.
    #[test]
    fn disk_fallback_rejects_sha_mismatch() {
        let root = tempfile::tempdir().expect("create temp dir");
        let index = LocalFileIndex::build(root.path().to_path_buf()).expect("build index");

        let rel = Path::new("bad/malware/evil.bin");
        write_file(&root.path().join(rel), b"actual-content");

        let resolved = index
            .resolve(
                rel.to_str().expect("utf8"),
                &sha256_hex(b"different-content"),
                i64::try_from(14u64).expect("len fits"),
            )
            .expect("resolve path");

        assert!(resolved.is_none());
    }

    /// Absolute paths from the DB that exist on disk should be resolved
    /// via the disk fallback even when not in the index.
    #[test]
    fn disk_fallback_resolves_absolute_path_not_in_index() {
        let root = tempfile::tempdir().expect("create temp dir");
        let index = LocalFileIndex::build(root.path().to_path_buf()).expect("build index");

        // Create a file outside the index root (simulates an absolute DB path).
        let external = tempfile::tempdir().expect("create external dir");
        let path = external.path().join("wolfi/pkg-1.0.apk");
        let bytes = b"absolute-path-file";
        write_file(&path, bytes);

        let resolved = index
            .resolve(
                path.to_str().expect("utf8"),
                &sha256_hex(bytes),
                i64::try_from(bytes.len()).expect("len fits"),
            )
            .expect("resolve path");

        assert_eq!(resolved.as_deref(), Some(path.as_path()));
    }

    #[test]
    fn poll_request_context_hints_for_https_to_http_mismatch() {
        let context = poll_request_context(
            "https://10.9.8.5:8081/api/next",
            "client error (Connect): received corrupt message of type InvalidContentType",
            true,
        );
        assert!(context.contains("peer did not speak TLS"));
        assert!(context.contains("Try http://"));
    }

    #[test]
    fn poll_request_context_avoids_hint_for_other_errors() {
        let context = poll_request_context(
            "https://10.9.8.5:8081/api/next",
            "dns error: failed to lookup address information",
            true,
        );
        assert!(!context.contains("Try http://"));
    }
}
