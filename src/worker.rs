//! Pull-based worker that polls a hopper instance for analysis jobs.
//!
//! Each worker maintains N concurrent analysis slots via a tokio semaphore.
//! When a slot is free, it claims work from hopper's `/api/next` endpoint,
//! analyzes the file, and posts the result back to `/api/result`.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::{Semaphore, mpsc};

use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Model, Thresholds};
use crate::server::{ModelResources, classify_bytes, classify_file};
use crate::system_load_avg;

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
/// Cadence for the dedicated `/api/heartbeat` check-in. Fixed and independent of
/// the work-claim poll so a busy worker — prefetch buffer full, never polling
/// `/api/next` — still reports liveness, RSS, load, and queue depth on time.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);

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
    /// Exit cleanly once the hopper reports no further work and the prefetch
    /// queue has drained (for benchmarks / batch runs over a finite dataset).
    /// Unlike `max_jobs`, this does not depend on knowing the job count and
    /// cannot wedge the dispatch loop on a blocked claim.
    pub exit_if_empty: bool,
    /// FPR severity level (0..=10000) that produced the thresholds, or `None` when
    /// manual thresholds were supplied. Folded into `ml.lvl` in the envelope.
    pub level: Option<u16>,
    /// Nice value applied to the process at startup (0 = leave unchanged).
    pub nice: i32,
    /// Optional LLM interpretation config (`--interpret`); `None` disables the
    /// pass. Reattached to every reloaded `ModelResources` so renewals keep it.
    pub interpret: Option<crate::interpret::InterpretConfig>,
    /// External-reference fetch policy (`SCAN_FETCH`). Default (empty) keeps the
    /// worker fully offline; a non-empty policy makes every job fetch and
    /// re-analyze the references it discovers. Reattached to each reloaded
    /// `ModelResources` so renewals preserve it.
    pub fetch: crate::fetch::FetchPolicy,
}

fn load_model_resources(
    model_dir: &Path,
    thresholds: Option<Thresholds>,
    level: Option<u16>,
    interpret: Option<crate::interpret::InterpretConfig>,
    fetch: crate::fetch::FetchPolicy,
) -> Result<Arc<ModelResources>> {
    let model = Model::load(model_dir, thresholds, level).context("loading model")?;
    let shap = ShapImportance::load(model_dir).ok();
    let ctx = ExtractContext::new(model.spec());
    Ok(Arc::new(ModelResources {
        model,
        shap,
        ctx,
        interpret,
        // Per-job scanning honors the worker's fetch policy (`SCAN_FETCH`). The
        // fixed validate corpus never fetches — it runs through
        // `crate::validate::run`, which builds its own offline resources.
        fetch,
    }))
}

fn validate_and_load_resources(
    model_dir: &Path,
    thresholds: Option<Thresholds>,
    slow_rule_ms: u64,
    level: Option<u16>,
    interpret: Option<crate::interpret::InterpretConfig>,
    fetch: crate::fetch::FetchPolicy,
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
    crate::validate::run(&validate_config, false)?;
    load_model_resources(model_dir, thresholds, level, interpret, fetch)
}

/// Pull upstream rules and, **only if something actually changed**, re-validate
/// and reload the model bundle. Returns `Ok(None)` when both repos are already
/// up to date — a silent no-op so the periodic renewal doesn't flood the log
/// with a full validation pass every interval.
fn renew_resources_once(
    model_dir: &Path,
    thresholds: Option<Thresholds>,
    slow_rule_ms: u64,
    level: Option<u16>,
    interpret: Option<crate::interpret::InterpretConfig>,
    fetch: crate::fetch::FetchPolicy,
) -> Result<Option<Arc<ModelResources>>> {
    // model_update validates the freshly extracted bundle (Model::load) before
    // swapping it in, so a broken bundle never lands on disk — there's no
    // last-known-good state to roll back to. A combined-validation failure below
    // propagates; the worker keeps serving its current in-memory resources until
    // the next successful renewal or a restart.
    let dir = crate::models_repo::install_target();
    let before = crate::model_update::installed(&dir).map(|i| i.commit);
    let models_changed = match crate::model_update::update(&dir, false, false) {
        Ok(()) => before != crate::model_update::installed(&dir).map(|i| i.commit),
        Err(error) => {
            tracing::warn!(error = %error, "model renewal failed; treating models as unchanged");
            false
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
        "rules changed; revalidating bundle"
    );

    let resources =
        validate_and_load_resources(model_dir, thresholds, slow_rule_ms, level, interpret, fetch)?;

    let (traits, composites) = cleave::reload_capability_mapper()
        .map_err(|error| anyhow::anyhow!("reload cleave capability mapper: {error}"))?;
    tracing::info!(traits, composites, "cleave capability mapper renewed");

    Ok(Some(resources))
}

fn current_resources(handle: &ResourceHandle) -> Result<Arc<ModelResources>> {
    let guard = handle
        .read()
        .map_err(|error| anyhow::anyhow!("worker resources lock poisoned: {error}"))?;
    Ok(Arc::clone(&guard))
}

// Mirrors the resource-loading parameter set (model location, thresholds,
// level, interpret, fetch) plus the renewal handle/shutdown; threading them
// individually keeps the call chain explicit rather than introducing a struct
// used in exactly one place.
#[allow(clippy::too_many_arguments)]
fn spawn_resource_renewal_task(
    handle: ResourceHandle,
    model_dir: PathBuf,
    thresholds: Option<Thresholds>,
    slow_rule_ms: u64,
    level: Option<u16>,
    interpret: Option<crate::interpret::InterpretConfig>,
    fetch: crate::fetch::FetchPolicy,
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
            let interpret = interpret.clone();
            let result = tokio::task::spawn_blocking(move || {
                renew_resources_once(
                    &model_dir,
                    thresholds,
                    slow_rule_ms,
                    level,
                    interpret,
                    fetch,
                )
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

// Phase 2 (WORKER_POOL_PLAN.md): litmus no longer manages rayon. There is no
// per-slot pool grid — analyses run on the one process-global rayon pool that
// main installs (sized to the host's cores, 256 MB stacks). The tokio semaphore
// alone bounds concurrency; cleave's `par_iter` fan-out work-steals across the
// shared pool, so a single large archive can use the whole machine while total
// rayon threads stay capped at the pool size (not `slots × per-slot-threads`),
// which in turn caps cleave's per-thread YARA scanners.

/// How long a staged job may be passed over by smaller arrivals before SJF
/// dispatches it anyway. Bounds a big job's staging delay under a continuous
/// stream of small jobs, so SJF can't starve archives indefinitely.
///
/// Must sit well above typical *large-job service time*, not small-job time: on
/// the realworld dataset (medium/large analyses run 6–25 minutes) a 120 s bound
/// aged out every staged archive while slots ground through earlier work, and
/// the oldest-aged-first rule then preempted every small job — dispatch
/// degenerated to FIFO and the SJF latency win vanished. 15 minutes keeps the
/// guarantee (no archive waits forever behind a small-job stream) without
/// re-creating the starvation SJF exists to fix. `SCAN_SJF_MAX_WAIT_SECS`
/// overrides for experiments.
const SJF_MAX_STAGED_WAIT: Duration = Duration::from_secs(900);

/// Resolved aging bound: [`SJF_MAX_STAGED_WAIT`] unless overridden.
fn sjf_max_staged_wait() -> Duration {
    std::env::var("SCAN_SJF_MAX_WAIT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|s| *s > 0)
        .map_or(SJF_MAX_STAGED_WAIT, Duration::from_secs)
}

/// Size-aware dispatch (default): take the smallest staged job, not the oldest.
///
/// Sweeps everything currently in the prefetch channel into `reorder` (the
/// dispatch loop holds jobs the prefetcher has already claimed and downloaded),
/// then picks the smallest by hopper-reported size — unless something has waited
/// past [`SJF_MAX_STAGED_WAIT`], in which case the oldest such job goes first.
/// Blocks only when the window is empty. Returns `None` once the channel is
/// closed *and* the window is drained, matching `recv()`'s termination contract.
///
/// Pairs with hopper's size-interleaved Tier 1 handout: hopper guarantees every
/// claim batch carries a size mix, this picker guarantees the small ones go
/// first. Measured on the realworld dataset: median small-sample flow 15.3 →
/// 6.4 minutes at neutral wall. The known trade-off is at the tail — deferring
/// archives clusters them wherever small jobs thin out, which raised drain-mode
/// small p95 and peak RSS (the memory-admission gate is the backstop there).
/// `SCAN_SJF=0` restores FIFO dispatch.
async fn next_smallest_staged(
    rx: &mut mpsc::UnboundedReceiver<PrefetchedJob>,
    reorder: &mut Vec<(PrefetchedJob, Instant)>,
) -> Option<PrefetchedJob> {
    // Sweep all ready jobs into the window without blocking.
    while let Ok(pj) = rx.try_recv() {
        reorder.push((pj, Instant::now()));
    }
    // Empty window: block for the next arrival like FIFO would, then sweep
    // again so a burst that landed together is reordered together.
    if reorder.is_empty() {
        let first = rx.recv().await?;
        reorder.push((first, Instant::now()));
        while let Ok(pj) = rx.try_recv() {
            reorder.push((pj, Instant::now()));
        }
    }

    let now = Instant::now();
    let max_wait = sjf_max_staged_wait();
    let aged = reorder
        .iter()
        .enumerate()
        .filter(|(_, (_, staged_at))| now.duration_since(*staged_at) >= max_wait)
        .min_by_key(|(_, (_, staged_at))| *staged_at)
        .map(|(i, _)| i);
    #[allow(clippy::expect_used)]
    let idx = aged.unwrap_or_else(|| {
        reorder
            .iter()
            .enumerate()
            .min_by_key(|(_, (pj, _))| pj.job.size_bytes.max(0))
            .map(|(i, _)| i)
            .expect("reorder window is non-empty")
    });
    Some(reorder.swap_remove(idx).0)
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
    /// Whether hopper holds registry-metadata provenance for this sample. When
    /// set, the worker fetches it (`/api/provenance/{sha256}`) and reasons over
    /// the same registry facts a live `pkg`/`url` scan would — without a
    /// refetch. A second round-trip is wasted when absent, so hopper flags it
    /// here on the claim it already has to send.
    #[serde(default)]
    has_provenance: bool,
}

/// A job with its file data pre-downloaded (or marked for local access).
struct PrefetchedJob {
    job: ClaimJob,
    /// `Ok(None)` = use local file, `Ok(Some(bytes))` = downloaded,
    /// `Err(Transient)` = download failed (fall back to direct download),
    /// `Err(Skipped)` = job rejected without attempting download (e.g. oversized);
    /// do not retry, post the error result directly.
    data: std::result::Result<Option<bytes::Bytes>, PrefetchError>,
    /// Local-queue id assigned by the prefetcher when the job is staged; passed
    /// to `WorkerMetrics::complete` once analysis finishes. 0 until staged.
    queue_id: u64,
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
            use tokio::signal::unix::{SignalKind, signal};
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

/// Trailing window for the throughput and error-count metrics.
const METRICS_WINDOW: Duration = Duration::from_secs(15 * 60);

/// One-minute completion buckets for a trailing files-per-second rate. Bucketing
/// (vs. a list of every completion instant) keeps memory O(1) regardless of how
/// fast a worker churns. Slot `minute % 15` holds the count for that minute;
/// a slot whose stored minute is stale is reset on first use in the new minute.
struct RateWindow {
    buckets: [(u64, u32); 15],
}

impl RateWindow {
    fn new() -> Self {
        Self {
            buckets: [(u64::MAX, 0); 15],
        }
    }

    fn record(&mut self, minute: u64) {
        let slot = &mut self.buckets[(minute % 15) as usize];
        if slot.0 != minute {
            *slot = (minute, 0);
        }
        slot.1 += 1;
    }

    fn per_sec(&self, minute: u64) -> f64 {
        let total: u32 = self
            .buckets
            .iter()
            .filter(|(m, _)| *m != u64::MAX && minute.saturating_sub(*m) < 15)
            .map(|(_, c)| c)
            .sum();
        f64::from(total) / METRICS_WINDOW.as_secs() as f64
    }
}

/// Error instants within the trailing window plus the most recent message.
#[derive(Default)]
struct ErrorWindow {
    times: VecDeque<Instant>,
    last: Option<(Instant, String)>,
}

impl ErrorWindow {
    fn record(&mut self, msg: &str, now: Instant) {
        self.prune(now);
        self.times.push_back(now);
        self.last = Some((now, msg.to_string()));
    }

    fn prune(&mut self, now: Instant) {
        while let Some(&t) = self.times.front() {
            if now.duration_since(t) <= METRICS_WINDOW {
                break;
            }
            self.times.pop_front();
        }
    }
}

/// Point-in-time view of [`WorkerMetrics`], assembled for one heartbeat.
struct MetricsSnapshot {
    /// Age of the oldest item still in the local queue (staged or running).
    oldest_age: Option<Duration>,
    /// Time since the most recent job completion.
    last_completion_age: Option<Duration>,
    /// Files processed per second, averaged over the trailing window.
    files_per_sec: f64,
    /// Number of analysis errors within the trailing window.
    errors_recent: usize,
    /// Most recent error: how long ago it happened and its message.
    last_error: Option<(Duration, String)>,
}

/// Live per-worker metrics surfaced through the heartbeat. Updated on the job
/// hot path — `enqueue` when a sample is staged, `complete`/`record_error` when
/// analysis finishes — and snapshotted by the heartbeat task. The worker reports
/// ages and rates (never wall-clock timestamps), so clock skew between worker
/// and hopper can't distort the dashboard.
struct WorkerMetrics {
    /// Monotonic base for the throughput minute index.
    start: Instant,
    next_id: AtomicU64,
    /// Enqueue instant per in-queue item, keyed by id; oldest age = min elapsed.
    enqueued: Mutex<HashMap<u64, Instant>>,
    last_completion: Mutex<Option<Instant>>,
    throughput: Mutex<RateWindow>,
    errors: Mutex<ErrorWindow>,
}

// Each method holds one of the metrics mutexes only for a trivial,
// panic-free critical section; a poisoned lock means a prior holder panicked,
// which is already unrecoverable, so propagating via expect is correct.
#[allow(clippy::expect_used)]
impl WorkerMetrics {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            next_id: AtomicU64::new(0),
            enqueued: Mutex::new(HashMap::new()),
            last_completion: Mutex::new(None),
            throughput: Mutex::new(RateWindow::new()),
            errors: Mutex::new(ErrorWindow::default()),
        }
    }

    fn minute(&self) -> u64 {
        self.start.elapsed().as_secs() / 60
    }

    /// Register a freshly staged queue item; returns its id for `complete`.
    fn enqueue(&self) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.enqueued
            .lock()
            .expect("metrics mutex poisoned")
            .insert(id, Instant::now());
        id
    }

    /// Mark a queue item finished: drop it, stamp the completion, tick the rate.
    fn complete(&self, id: u64) {
        self.enqueued
            .lock()
            .expect("metrics mutex poisoned")
            .remove(&id);
        *self.last_completion.lock().expect("metrics mutex poisoned") = Some(Instant::now());
        let minute = self.minute();
        self.throughput
            .lock()
            .expect("metrics mutex poisoned")
            .record(minute);
    }

    fn record_error(&self, msg: &str) {
        self.errors
            .lock()
            .expect("metrics mutex poisoned")
            .record(msg, Instant::now());
    }

    fn snapshot(&self) -> MetricsSnapshot {
        let oldest_age = self
            .enqueued
            .lock()
            .expect("metrics mutex poisoned")
            .values()
            .min()
            .map(Instant::elapsed);
        let last_completion_age = self
            .last_completion
            .lock()
            .expect("metrics mutex poisoned")
            .map(|t| t.elapsed());
        let files_per_sec = self
            .throughput
            .lock()
            .expect("metrics mutex poisoned")
            .per_sec(self.minute());
        let (errors_recent, last_error) = {
            let mut errors = self.errors.lock().expect("metrics mutex poisoned");
            errors.prune(Instant::now());
            let last = errors.last.as_ref().map(|(t, m)| (t.elapsed(), m.clone()));
            (errors.times.len(), last)
        };
        MetricsSnapshot {
            oldest_age,
            last_completion_age,
            files_per_sec,
            errors_recent,
            last_error,
        }
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
    // Slot count bounds concurrency but not memory: a slot analysing a huge
    // archive holds it (plus expanded members) resident while a slot analysing a
    // 4 KB script holds nothing. This gate pauses admission on live memory
    // pressure — at the resolved `--max-rss-gb` ceiling (default 85% of RAM) —
    // so a burst of large archives serialises instead of co-residing and
    // exhausting memory. It pauses and reclaims; it never kills the worker.
    let admission = crate::admission::MemoryAdmission::new(
        config.max_rss_gb.saturating_mul(1024 * 1024 * 1024),
    );
    let shutdown = Arc::new(AtomicBool::new(false));
    install_shutdown_handler(Arc::clone(&shutdown));

    // Phase 2 thread model: `slots` (the tokio semaphore) bounds how many
    // analyses run concurrently; the one process-global rayon pool provides the
    // parallelism. Total rayon threads = the pool size, independent of slots —
    // no per-slot grid, no `slots × threads_per_slot` multiplication. This is
    // the headline number that caps cleave's per-thread YARA scanners.
    let global_rayon_threads = rayon::current_num_threads();
    tracing::info!(
        slots,
        rayon_threads = global_rayon_threads,
        "worker concurrency: up to {slots} analyses share one shared \
         {global_rayon_threads}-thread rayon pool (no per-slot pools)",
    );
    // Each in-flight analysis parks a coordinator on the pool and fans member
    // work into it; slots far beyond the pool size just queue analyses against
    // each other (observed: 16 slots on a 4-thread illumos zone → 5 s to run a
    // trivial rayon task, 28 KB jobs taking minutes). Likely a --workers value
    // copied from a larger host.
    if slots > global_rayon_threads.saturating_mul(2) {
        tracing::warn!(
            slots,
            rayon_threads = global_rayon_threads,
            "worker slots exceed 2x the rayon pool; analyses will queue against \
             each other for pool threads — lower --workers (or raise \
             CLEAVE_RAYON_THREADS) to restore throughput",
        );
    }

    tracing::info!(
        name = %name,
        slots = slots,
        hopper = %config.hopper_url,
        global_rayon_threads,
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
        config.interpret.clone(),
        config.fetch,
    )?;
    let resources: ResourceHandle = Arc::new(RwLock::new(resources));
    spawn_resource_renewal_task(
        Arc::clone(&resources),
        config.model_dir.clone(),
        config.thresholds,
        config.slow_rule_ms,
        config.level,
        config.interpret.clone(),
        config.fetch,
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
    let exit_if_empty = config.exit_if_empty;
    let encoded_name: String = url_encode(&name);
    let available_tools = crate::tools::available_names().join(",");
    let completed = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let metrics = Arc::new(WorkerMetrics::new());
    // Cloned before `available_tools`/`encoded_name` move into the prefetcher so
    // the heartbeat task can report the same identity on its own cadence.
    let heartbeat_tools = available_tools.clone();
    let heartbeat_name = url_encode(&name);

    let max_buffer_bytes: usize =
        if cleave::memory_tracker::total_memory().unwrap_or(0) >= 16 * 1024 * 1024 * 1024 {
            1024 * 1024 * 1024 // 1 GiB on systems with >= 16 GiB RAM
        } else {
            512 * 1024 * 1024 // 512 MiB otherwise
        };

    // Background prefetch keeps `1.1 × slots` samples staged at all times so a
    // free worker slot never waits on a download. The prefetcher polls and
    // downloads on its own task, pushing each sample into `rx` the instant its
    // download finishes; this loop only pulls ready samples and dispatches
    // them. `outstanding` (staged + in-flight) bounds the depth; `queued_bytes`
    // bounds staged payload memory against `max_buffer_bytes`.
    let (tx, mut rx) = mpsc::unbounded_channel::<PrefetchedJob>();
    let queued_bytes = Arc::new(AtomicUsize::new(0));
    let outstanding = Arc::new(AtomicUsize::new(0));
    // Size-aware dispatch (default on): the smallest staged job dispatches
    // first instead of FIFO, so tiny samples stop queueing behind multi-minute
    // archives — on the realworld benchmark with hopper's size-interleaved
    // handout this cut the median small-sample turnaround 15.3 → 6.4 minutes
    // at neutral wall time. SJF needs a window to reorder over, so it deepens
    // the prefetch target to 2× slots (the worker claims ahead of capacity and
    // self-optimizes locally; hopper's handout strategy stays simple) — a
    // 7-point depth sweep put the knee at 1.75–2× with nothing gained beyond.
    // `SCAN_SJF=0` restores FIFO dispatch; `SCAN_PREFETCH_DEPTH` overrides
    // the slots multiplier in either mode.
    let sjf = std::env::var("SCAN_SJF").ok().is_none_or(|v| v != "0");
    let depth_factor = std::env::var("SCAN_PREFETCH_DEPTH")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|f| *f >= 1.0)
        .unwrap_or(if sjf { 2.0 } else { 1.1 });
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let target_depth = ((slots as f64 * depth_factor).ceil() as usize).max(1);
    if sjf {
        tracing::info!(
            target_depth,
            max_staged_wait_s = sjf_max_staged_wait().as_secs(),
            "size-aware dispatch: smallest staged job first (SCAN_SJF=0 for FIFO)",
        );
    } else {
        tracing::info!(target_depth, "FIFO dispatch (SCAN_SJF=0)");
    }

    // Poll telemetry shared between the prefetcher (writer) and the heartbeat
    // task (reader) so a check-in can report why the worker is/isn't claiming.
    let poll_state = Arc::new(PollState::default());

    tokio::spawn(
        Prefetcher {
            client: client.clone(),
            base_url: Arc::clone(&base_url),
            data_dir: data_dir.clone(),
            encoded_name,
            available_tools,
            slots,
            max_single_bytes: max_buffer_bytes / 2,
            max_buffer_bytes,
            poll_secs,
            target_depth,
            metrics: Arc::clone(&metrics),
            poll_state: Arc::clone(&poll_state),
            exit_if_empty,
        }
        .run(
            tx,
            Arc::clone(&queued_bytes),
            Arc::clone(&outstanding),
            Arc::clone(&shutdown),
        ),
    );

    // The dispatch loop parks in `await`s, so emit the periodic summary from a
    // dedicated ticker reading the shared counters.
    {
        let semaphore = Arc::clone(&semaphore);
        let completed = Arc::clone(&completed);
        let outstanding = Arc::clone(&outstanding);
        let queued_bytes = Arc::clone(&queued_bytes);
        let shutdown = Arc::clone(&shutdown);
        // Default 60 s; `SCAN_HEARTBEAT_SECS` lowers it (min 1 s) so a short
        // benchmark run still emits a usable rss / active-slot time series.
        let heartbeat = Duration::from_secs(
            std::env::var("SCAN_HEARTBEAT_SECS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map_or(60, |s| s.max(1)),
        );
        tokio::spawn(async move {
            // Per-slot census state, carried across ticks.
            // `last_cpu`/`last_at`: previous CPU-seconds + wall sample, for cores-busy.
            // `stage_since`: when each analysis entered its current phase, so the
            // census can report time-in-stage (a phase that never advances is the
            // signature of a wedge). Pruned to the live set each tick.
            // `wedge_latched`: analyses already announced as wedged, so the
            // consolidated WEDGE event fires once per stuck analysis.
            let mut last_cpu = crate::inflight::process_cpu_secs();
            let mut last_at = Instant::now();
            let mut stage_since: std::collections::HashMap<u64, (String, Instant)> =
                std::collections::HashMap::new();
            let mut wedge_latched: std::collections::HashSet<u64> =
                std::collections::HashSet::new();
            // Slots whose analysis has run past this long are flagged as stuck and
            // logged at WARN so a wedge stands out in the stream. Tunable via
            // `SCAN_STUCK_WARN_SECS` (min 1) for noisy shards or test runs.
            let stuck_warn_secs = std::env::var("SCAN_STUCK_WARN_SECS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map_or(300, |s| s.max(1));
            // Cap census lines so a saturated worker can't flood the log; oldest
            // first means the most-stuck slots always appear.
            const CENSUS_MAX_LINES: usize = 64;
            // Scan for wedges at least this often even when the summary heartbeat
            // is longer, so a stuck slot self-documents promptly rather than
            // waiting for the next (possibly minute-long) summary.
            let wedge_check = heartbeat.min(Duration::from_secs(30));
            let mut last_summary = Instant::now();
            while !shutdown.load(Ordering::Relaxed) {
                interruptible_sleep(wedge_check, &shutdown).await;
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                let now = Instant::now();
                let cpu = crate::inflight::process_cpu_secs();
                let wall = now.duration_since(last_at).as_secs_f64().max(1e-6);
                // Average cores busy since the last tick. With slots full but this
                // near zero, the worker is blocked (locks / subprocess / I/O), not
                // grinding — the key blocked-vs-busy bit for triaging a wedge.
                let cpu_cores_busy = ((cpu - last_cpu) / wall).max(0.0);
                last_cpu = cpu;
                last_at = now;

                let census = crate::inflight::snapshot();
                let live: std::collections::HashSet<u64> =
                    census.iter().map(|e| e.analysis_id).collect();
                stage_since.retain(|id, _| live.contains(id));
                wedge_latched.retain(|id| live.contains(id));

                // Newly-stuck analyses: over the threshold and not yet announced.
                // Cheap (elapsed only); resolving wait-channels (which may fork
                // `ps` off Linux) is deferred until we know we need them.
                let newly_stuck: Vec<&std::sync::Arc<crate::inflight::Entry>> = census
                    .iter()
                    .filter(|e| now.duration_since(e.started).as_secs() >= stuck_warn_secs)
                    .filter(|e| !wedge_latched.contains(&e.analysis_id))
                    .collect();
                let summary_due = now.duration_since(last_summary) >= heartbeat;

                // Resolve wait-channels once per tick, only when something will
                // print them (a wedge fired, or the summary census is due).
                let wchans = if newly_stuck.is_empty() && !summary_due {
                    std::collections::HashMap::new()
                } else {
                    let tids: Vec<u64> = census
                        .iter()
                        .map(|e| e.thread_id.load(Ordering::Relaxed))
                        .filter(|&t| t != 0)
                        .collect();
                    crate::inflight::wait_channels(&tids)
                };
                let waiting_for = |entry: &crate::inflight::Entry, stage: &str| -> String {
                    let tid = entry.thread_id.load(Ordering::Relaxed);
                    wchans
                        .get(&tid)
                        .cloned()
                        .unwrap_or_else(|| format!("stage:{stage}"))
                };

                // Consolidated WEDGE event: fires once per stuck analysis, on the
                // wedge cadence, so a hang self-documents without waiting for the
                // summary heartbeat.
                if !newly_stuck.is_empty() {
                    // Aggregate every thread's wait-channel: for archive wedges the
                    // real blockage is on rayon workers, not the per-slot
                    // coordinator, so this names the resource classes the pool is
                    // stuck on (yara symbol / pipe_wait subprocess / futex lock).
                    let thread_waits = crate::inflight::format_wait_summary(
                        &crate::inflight::thread_wait_summary(),
                    );
                    tracing::warn!(
                        newly_stuck = newly_stuck.len(),
                        inflight = census.len(),
                        cpu_cores_busy = format!("{cpu_cores_busy:.1}"),
                        rayon_threads = global_rayon_threads,
                        stuck_threshold_s = stuck_warn_secs,
                        thread_waits,
                        "WEDGE DETECTED: analyses exceeded the stuck threshold; per-slot detail follows",
                    );
                    for entry in &newly_stuck {
                        wedge_latched.insert(entry.analysis_id);
                        let phase = entry.phase.get();
                        let stage = if phase.is_empty() {
                            "(starting)"
                        } else {
                            phase.as_str()
                        };
                        tracing::warn!(
                            analysis_id = entry.analysis_id,
                            sha256 = %entry.sha,
                            file = %entry.file,
                            size_bytes = entry.size_bytes,
                            file_type = %entry.file_type,
                            thread_id = entry.thread_id.load(Ordering::Relaxed),
                            stuck_for_ms = crate::duration_ms(now.duration_since(entry.started)),
                            stage,
                            waiting = waiting_for(entry, stage),
                            "WEDGE slot",
                        );
                    }
                    // Per-thread cleave breadcrumbs: which member each rayon
                    // worker is on. For an archive wedge the work is spread
                    // across the pool, so this names the member-level culprits the
                    // per-slot (coordinator) lines can't. Gated on the
                    // `cleave-breadcrumbs` feature, which requires a cleave build
                    // exposing `cleave::breadcrumb` (not yet in the released rev).
                    #[cfg(feature = "cleave-breadcrumbs")]
                    for crumb in cleave::breadcrumb::snapshot()
                        .into_iter()
                        .take(CENSUS_MAX_LINES)
                    {
                        tracing::warn!(
                            rayon_index = ?crumb.rayon_index,
                            thread_id = crumb.thread_id,
                            analyzer = crumb.analyzer,
                            target = %crumb.target,
                            age_ms = crate::duration_ms(crumb.age),
                            "WEDGE breadcrumb",
                        );
                    }
                }

                if !summary_due {
                    continue;
                }
                last_summary = now;

                let started = BLOCKING_STARTED_TOTAL.load(Ordering::Relaxed);
                let finished = BLOCKING_FINISHED_TOTAL.load(Ordering::Relaxed);
                let available_slots = semaphore.available_permits();
                tracing::info!(
                    rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / 1024 / 1024),
                    queued_prefetch_jobs = outstanding.load(Ordering::Relaxed),
                    prefetch_buffer_mb = queued_bytes.load(Ordering::Relaxed) / (1024 * 1024),
                    active_slots = slots.saturating_sub(available_slots),
                    available_slots,
                    cpu_cores_busy = format!("{cpu_cores_busy:.1}"),
                    load1 = system_load_avg().map(|load| format!("{load:.1}")),
                    rayon_threads = global_rayon_threads,
                    blocking_started_total = started,
                    blocking_finished_total = finished,
                    inflight_blocking = started.saturating_sub(finished),
                    completed = completed.load(Ordering::Acquire),
                    "worker summary",
                );

                // Per-slot census: one line per in-flight analysis — file, size,
                // how long it has been running, the stage it is in (and for how
                // long), the worker thread, and what a blocked thread is waiting
                // on. Lets an operator name a wedged slot from the log alone.
                for entry in census.iter().take(CENSUS_MAX_LINES) {
                    let phase = entry.phase.get();
                    let stage = if phase.is_empty() {
                        "(starting)"
                    } else {
                        phase.as_str()
                    };
                    let slot = stage_since
                        .entry(entry.analysis_id)
                        .or_insert_with(|| (phase.clone(), now));
                    if slot.0 != phase {
                        *slot = (phase.clone(), now);
                    }
                    let stage_elapsed = now.duration_since(slot.1);
                    let total_elapsed = now.duration_since(entry.started);
                    let thread_id = entry.thread_id.load(Ordering::Relaxed);
                    let waiting = waiting_for(entry, stage);
                    if total_elapsed.as_secs() >= stuck_warn_secs {
                        tracing::warn!(
                            analysis_id = entry.analysis_id,
                            sha256 = %entry.sha,
                            file = %entry.file,
                            size_bytes = entry.size_bytes,
                            file_type = %entry.file_type,
                            thread_id,
                            stuck_for_ms = crate::duration_ms(total_elapsed),
                            stage,
                            stage_for_ms = crate::duration_ms(stage_elapsed),
                            waiting,
                            "slot in-flight (STUCK)",
                        );
                    } else {
                        tracing::info!(
                            analysis_id = entry.analysis_id,
                            sha256 = %entry.sha,
                            file = %entry.file,
                            size_bytes = entry.size_bytes,
                            file_type = %entry.file_type,
                            thread_id,
                            elapsed_ms = crate::duration_ms(total_elapsed),
                            stage,
                            stage_for_ms = crate::duration_ms(stage_elapsed),
                            waiting,
                            "slot in-flight",
                        );
                    }
                }
                if census.len() > CENSUS_MAX_LINES {
                    tracing::info!(
                        truncated = census.len() - CENSUS_MAX_LINES,
                        shown = CENSUS_MAX_LINES,
                        "slot census truncated",
                    );
                }
            }
        });
    }

    // Dedicated check-in. The claim loop only contacts hopper via `/api/next`
    // when the prefetch buffer has room, so a saturated worker can go long
    // stretches without reporting. This task pings `/api/heartbeat` on a fixed
    // cadence regardless of buffer state, carrying live RSS, load, and an
    // accurate queue depth (staged backlog + running slots).
    {
        let client = client.clone();
        let base_url = Arc::clone(&base_url);
        let encoded_name = heartbeat_name;
        let available_tools = heartbeat_tools;
        let outstanding = Arc::clone(&outstanding);
        let semaphore = Arc::clone(&semaphore);
        let metrics = Arc::clone(&metrics);
        let shutdown = Arc::clone(&shutdown);
        let admission = Arc::clone(&admission);
        let poll_state = Arc::clone(&poll_state);
        const MIB: u64 = 1024 * 1024;
        tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                interruptible_sleep(HEARTBEAT_INTERVAL, &shutdown).await;
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                let report = HeartbeatReport {
                    slots,
                    active: slots.saturating_sub(semaphore.available_permits()),
                    queue: outstanding.load(Ordering::Acquire),
                    mem_reserved_mb: admission.reserved_bytes() as u64 / MIB,
                    mem_ceiling_mb: admission.ceiling_bytes() / MIB,
                    poll_age_s: metrics
                        .start
                        .elapsed()
                        .as_secs()
                        .saturating_sub(poll_state.last_poll_secs.load(Ordering::Acquire)),
                    last_want: poll_state.last_want.load(Ordering::Acquire),
                    last_claim: poll_state.last_claim.load(Ordering::Acquire),
                    buffer_room: poll_state.buffer_room.load(Ordering::Acquire),
                    metrics: metrics.snapshot(),
                };
                let url = heartbeat_url(&base_url, &encoded_name, &available_tools, &report);
                match client.get(&url).send().await {
                    Ok(resp) if resp.status().is_success() => {}
                    Ok(resp) => {
                        tracing::debug!(status = %resp.status(), "heartbeat: non-success response");
                    }
                    Err(e) => tracing::debug!(error = %e, "heartbeat request failed"),
                }
            }
        });
    }

    // SJF reorder window: jobs pulled off the channel but not yet dispatched,
    // each with its staging time for the anti-starvation age check.
    let mut reorder: Vec<(PrefetchedJob, Instant)> = Vec::new();
    loop {
        if shutdown.load(Ordering::Relaxed) {
            tracing::info!("shutdown signalled, draining in-flight work");
            break;
        }

        if let Some(max) = max_jobs
            && completed.load(Ordering::Acquire) >= max
        {
            tracing::info!(max_jobs = max, "job limit reached, draining in-flight work");
            shutdown.store(true, Ordering::Relaxed);
            break;
        }

        // Claim a worker slot, then take the next staged sample: work begins as
        // soon as both a free slot and a ready sample exist. An empty channel
        // means the prefetcher has exited (shutdown) — drain and stop.
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .context("semaphore closed")?;
        let next = if sjf {
            next_smallest_staged(&mut rx, &mut reorder).await
        } else {
            rx.recv().await
        };
        let Some(pj) = next else {
            drop(permit);
            shutdown.store(true, Ordering::Relaxed);
            break;
        };
        let staged_bytes = pj
            .data
            .as_ref()
            .map_or(0, |d| d.as_ref().map_or(0, bytes::Bytes::len));
        queued_bytes.fetch_sub(staged_bytes, Ordering::Release);
        outstanding.fetch_sub(1, Ordering::Release);

        let client = client.clone();
        let resources = match current_resources(&resources) {
            Ok(resources) => resources,
            Err(error) => {
                // Lock poisoned (a worker thread panicked). Report the job as
                // failed so hopper reassigns it instead of waiting out the
                // lease, then carry on.
                tracing::error!(error = %error, "cannot snapshot worker resources; failing job");
                drop(permit);
                let failure = format!("worker resource snapshot failed: {error}");
                metrics.record_error(&failure);
                post_result(&client, &base_url, &name, &pj.job.sha256, Err(failure)).await;
                metrics.complete(pj.queue_id);
                completed.fetch_add(1, Ordering::Release);
                continue;
            }
        };
        // Reserve this job's estimated memory footprint before it starts
        // expanding the archive. Blocks while the in-flight budget is full, so a
        // burst of large archives serialises rather than co-residing and
        // exhausting RAM. Held only for the duration of the analysis.
        let admission_guard = admission
            .admit(
                Arc::from(pj.job.sha256.as_str()),
                Arc::from(pj.job.path.as_str()),
                Arc::from(pj.job.file_type.as_str()),
                pj.job.size_bytes,
            )
            .await;

        let url = Arc::clone(&base_url);
        let name = Arc::clone(&name);
        let local_index = local_index.clone();
        let completed = Arc::clone(&completed);
        let metrics = Arc::clone(&metrics);

        tokio::spawn(async move {
            let result = run_job(
                &client,
                &url,
                local_index.as_deref(),
                &pj.job,
                &resources,
                slow_rule_ms,
                pj.data,
            )
            .await;
            // The archive and its expanded members are freed when `run_job`
            // returns; release the memory reservation now so the budget reopens
            // before the result round-trip to hopper, not after.
            drop(admission_guard);
            if let Err(ref e) = result {
                tracing::warn!(
                    sha256 = %pj.job.sha256,
                    file = %pj.job.path,
                    file_type = %pj.job.file_type,
                    size = pj.job.size_bytes,
                    error = %e,
                    "analysis failed",
                );
                metrics.record_error(&e.to_string());
            }
            post_result(&client, &url, &name, &pj.job.sha256, result).await;
            metrics.complete(pj.queue_id);
            let n = completed.fetch_add(1, Ordering::Release) + 1;
            if n.is_multiple_of(100) {
                tokio::task::spawn_blocking(cleave::clear_all_thread_caches);
            }
            drop(permit);
        });
    }

    // Drain any in-flight work before exiting. Each analysis releases its slot
    // permit only after posting its result, so acquiring every permit means all
    // results have reached hopper.
    //
    // On a graceful (SIGTERM) shutdown the wait is capped so a stuck cleave
    // unpack can't block shutdown indefinitely — hopper re-leases anything left
    // running. But `--exit-if-empty` is batch mode: the operator asked to
    // process a finite dataset to completion, so we wait unbounded — a 60 s cap
    // would silently drop the result of any analysis slower than the drain
    // window (e.g. a large archive), which on a finite run is a lost finding,
    // not a re-lease.
    let slot_count = u32::try_from(slots).unwrap_or(u32::MAX);
    let drain = semaphore.acquire_many(slot_count);
    if exit_if_empty {
        let _ = drain.await;
        tracing::info!("all in-flight jobs finished (batch drain), exiting");
    } else {
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
    }
    Ok(())
}

/// Background prefetcher: owns the polling and download side of the worker.
/// Keeps `target_depth` (`1.1 × slots`) samples staged in the dispatch channel,
/// downloading payloads concurrently and emitting each the moment it lands so a
/// free worker slot never blocks on the network.
struct Prefetcher {
    client: reqwest::Client,
    base_url: Arc<str>,
    data_dir: Option<PathBuf>,
    encoded_name: String,
    available_tools: String,
    slots: usize,
    /// Per-job download cap; oversized jobs are reported as errors rather than
    /// downloaded, so one huge payload can't blow the buffer budget.
    max_single_bytes: usize,
    /// Soft cap on total staged payload bytes.
    max_buffer_bytes: usize,
    poll_secs: u64,
    /// Staged + in-flight sample target (`1.1 × slots`).
    target_depth: usize,
    /// Shared metrics; the prefetcher stamps each sample's local-queue entry.
    metrics: Arc<WorkerMetrics>,
    /// Poll telemetry surfaced on the heartbeat (last want/claim, buffer room).
    poll_state: Arc<PollState>,
    /// Stop (closing the dispatch channel) when the hopper reports no work and
    /// the queue has drained — drives clean batch/benchmark termination.
    exit_if_empty: bool,
}

impl Prefetcher {
    /// Build the `/api/next` URL, attaching the live signals hopper uses to
    /// ration work: traits version, current RSS, 1-minute load, and tools.
    fn poll_url(&self, count: usize) -> String {
        use std::fmt::Write;
        let mut url = format!(
            "{}/api/next?worker={}&count={}&slots={}&version={}",
            self.base_url,
            self.encoded_name,
            count,
            self.slots,
            env!("CARGO_PKG_VERSION"),
        );
        // 5-char prefix matches hopper's litmusTraitsVersion() truncation so the
        // dashboard's stale-traits comparison can string-equal the two.
        if let Some(traits) = cleave::traits_repo::version() {
            let prefix: String = traits.chars().take(5).collect();
            let _ = write!(url, "&traits={}", prefix);
        }
        if let Some(rss) = cleave::memory_tracker::current_rss() {
            let _ = write!(url, "&rss_mb={}", rss / 1024 / 1024);
        }
        if let Some(load) = system_load_avg() {
            let _ = write!(url, "&load1={:.2}", load);
        }
        let _ = write!(url, "&tools=");
        url_encode_into(&self.available_tools, &mut url);
        url
    }

    /// Run until shutdown or the dispatch channel closes. `outstanding` tracks
    /// staged + in-flight samples to bound depth; `queued_bytes` tracks staged
    /// payload memory against `max_buffer_bytes`.
    async fn run(
        self,
        tx: mpsc::UnboundedSender<PrefetchedJob>,
        queued_bytes: Arc<AtomicUsize>,
        outstanding: Arc<AtomicUsize>,
        shutdown: Arc<AtomicBool>,
    ) {
        let mut consecutive_errors: u32 = 0;
        loop {
            if shutdown.load(Ordering::Relaxed) {
                return;
            }

            // Hold at the target depth and don't stage more bytes than the
            // budget allows. Either "buffer full" state: wait briefly for the
            // dispatch loop to drain, then re-evaluate.
            let room = self
                .target_depth
                .saturating_sub(outstanding.load(Ordering::Acquire));
            // Publish free buffer room every iteration: 0 here is the heartbeat's
            // signal that the worker is saturated and deliberately not polling.
            self.poll_state.buffer_room.store(room, Ordering::Release);
            let over_budget = queued_bytes.load(Ordering::Acquire) >= self.max_buffer_bytes;
            if room == 0 || over_budget {
                interruptible_sleep(Duration::from_millis(100), &shutdown).await;
                continue;
            }

            // Cap a single poll's burst to `slots` so concurrent downloads stay
            // bounded; the depth fills over a few polls.
            let count = room.min(self.slots);
            let url = self.poll_url(count);
            // Stamp the poll so the heartbeat can report poll age and want/claim.
            self.poll_state
                .last_poll_secs
                .store(self.metrics.start.elapsed().as_secs(), Ordering::Release);
            self.poll_state.last_want.store(count, Ordering::Release);
            match claim_jobs(&self.client, &url).await {
                Ok(None) => {
                    self.poll_state.last_claim.store(0, Ordering::Release);
                    consecutive_errors = 0;
                    // Batch/benchmark mode: once the hopper has no work AND the
                    // dispatch channel is drained (every claimed job picked up),
                    // stop. Returning drops `tx`, so the dispatch loop's
                    // `rx.recv()` yields `None` and runs its normal drain — which
                    // waits for any still-in-flight analyses — instead of
                    // blocking forever on a claim that will never arrive.
                    if self.exit_if_empty && outstanding.load(Ordering::Acquire) == 0 {
                        tracing::info!(
                            "hopper drained and queue empty; --exit-if-empty stopping prefetch",
                        );
                        return;
                    }
                    interruptible_sleep(Duration::from_secs(self.poll_secs), &shutdown).await;
                }
                Ok(Some(jobs)) => {
                    self.poll_state
                        .last_claim
                        .store(jobs.len(), Ordering::Release);
                    consecutive_errors = 0;
                    outstanding.fetch_add(jobs.len(), Ordering::Release);
                    let mut set = tokio::task::JoinSet::new();
                    for job in jobs {
                        let client = self.client.clone();
                        let base_url = Arc::clone(&self.base_url);
                        let data_dir = self.data_dir.clone();
                        let max_single_bytes = self.max_single_bytes;
                        set.spawn(async move {
                            prefetch_one(client, base_url, data_dir, max_single_bytes, job).await
                        });
                    }
                    while let Some(res) = set.join_next().await {
                        match res {
                            Ok(mut pj) => {
                                let bytes = pj
                                    .data
                                    .as_ref()
                                    .map_or(0, |d| d.as_ref().map_or(0, bytes::Bytes::len));
                                queued_bytes.fetch_add(bytes, Ordering::Release);
                                // Enters the local queue now; tracked until the
                                // dispatch loop finishes analysing it.
                                pj.queue_id = self.metrics.enqueue();
                                if tx.send(pj).is_err() {
                                    return; // dispatch loop gone
                                }
                            }
                            Err(e) => {
                                // Download task panicked; reclaim its depth slot.
                                outstanding.fetch_sub(1, Ordering::Release);
                                tracing::warn!(error = %e, "prefetch task panicked");
                            }
                        }
                    }
                }
                Err(e) => {
                    consecutive_errors += 1;
                    let backoff = backoff_duration(consecutive_errors);
                    tracing::warn!(
                        url = %url,
                        error = %format!("{e:#}"),
                        backoff_secs = backoff.as_secs(),
                        consecutive_errors,
                        "poll/prefetch failed",
                    );
                    interruptible_sleep(backoff, &shutdown).await;
                }
            }
        }
    }
}

/// Poll hopper's `/api/next` once. `Ok(None)` means no work is available now.
async fn claim_jobs(client: &reqwest::Client, poll_url: &str) -> Result<Option<Vec<ClaimJob>>> {
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
    Ok(Some(claim.jobs))
}

/// Download one claimed job's payload (or mark it for local access / rejection).
/// Oversized jobs are skipped without a download, local files are used in place,
/// and transient download failures fall through to `run_job`'s direct-download
/// retry.
async fn prefetch_one(
    client: reqwest::Client,
    base_url: Arc<str>,
    data_dir: Option<PathBuf>,
    max_single_bytes: usize,
    job: ClaimJob,
) -> PrefetchedJob {
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
        return PrefetchedJob {
            job,
            data: Err(err),
            queue_id: 0,
        };
    }

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
    PrefetchedJob {
        job,
        data,
        queue_id: 0,
    }
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
    prefetched: std::result::Result<Option<bytes::Bytes>, PrefetchError>,
) -> Result<(crate::engine::ScanResultEnvelope, Vec<crate::engine::DepResult>, i64), String> {
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
    let downloaded: Option<bytes::Bytes> = if use_local {
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

    // Registry metadata hopper collected for this sample at fetch time, so the
    // worker reasons over the same registry facts (age, custody, popularity,
    // deprecation) a live `pkg`/`url` scan fetches — without a refetch. Only
    // attempted when hopper flagged the sample as carrying it; best-effort, so a
    // miss never fails the scan. Consumed as stamped at collection time.
    let root_registry: Option<fletch::Registry> = if job.has_provenance {
        download_provenance(client, base_url, &job.sha256).await
    } else {
        None
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
    // Register this analysis in the live in-flight census so the periodic worker
    // summary can report its file, size, stage, time stuck, and what it is
    // waiting on. The guard deregisters it when `run_job` returns.
    let _inflight_census = crate::inflight::register(
        analysis_id,
        Arc::clone(&sha_short),
        Arc::clone(&label),
        u64::try_from(job.size_bytes).unwrap_or(0),
        Arc::from(job.file_type.as_str()),
        start,
        phase.clone(),
    );
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
    let handle = tokio::task::spawn_blocking(move || {
        // Runs on a tokio blocking thread; cleave's `par_iter` fan-out work-steals
        // across the shared global rayon pool. Lifecycle logs report `thread_id` —
        // the blocking thread an operator samples to find a wedged analysis; the
        // CPU work itself runs on the rayon pool threads.
        let started = BLOCKING_STARTED_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
        let thread_id = crate::thread_dump::os_thread_id();
        // Attach the worker thread to the live census so the periodic summary
        // can report which thread each in-flight analysis is wedged on and
        // read its kernel wait-channel.
        crate::inflight::set_thread_id(analysis_id, thread_id);
        // Register the blocking analysis thread for the SIGUSR1 thread dump
        // (rayon workers register via the pool's start handler).
        crate::thread_dump::register_self();
        let inflight_blocking =
            started.saturating_sub(BLOCKING_FINISHED_TOTAL.load(Ordering::Relaxed));
        tracing::info!(
            analysis_id,
            sha256 = %sha_short2,
            file = %label_for_blocking,
            thread_id,
            inflight_blocking,
            started_total = started,
            rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / 1024 / 1024),
            "analysis starting on worker thread",
        );
        // Record this analysis as in flight so the SIGABRT handler can name
        // it if a deep analysis overflows the stack and aborts the process.
        // The guard frees the slot on normal completion; an abort skips the
        // drop, leaving the entry live for the dump — exactly the suspect
        // set we want. See `crate::crash_dump`.
        let _inflight =
            crate::crash_dump::register(analysis_id, thread_id, &sha_short2, &label_for_blocking);
        let result = if let Some(data) = downloaded {
            classify_bytes(
                data,
                &label_for_blocking,
                &resources,
                slow_rule_ms,
                Some(&cancel2),
                Some(&phase),
                root_registry.as_ref(),
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
                root_registry.as_ref(),
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
            thread_id,
            inflight_blocking,
            finished_total = finished,
            rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / 1024 / 1024),
            elapsed_ms = crate::duration_ms(start.elapsed()),
            "analysis complete on worker thread",
        );
        result
    });

    let result = handle.await;

    #[allow(clippy::cast_sign_loss)]
    let elapsed_ms = crate::duration_ms(start.elapsed()) as i64;

    match result {
        Ok(Ok(mut scan_result)) => {
            let deps = std::mem::take(&mut scan_result.dependency_results);
            Ok((scan_result.into_envelope(), deps, elapsed_ms))
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
    result: Result<(crate::engine::ScanResultEnvelope, Vec<crate::engine::DepResult>, i64), String>,
) {
    // The hopper base URL, captured before `url` is shadowed by the result
    // endpoint below — fetched dependencies are mirrored against the same base.
    let base_url = url.to_string();
    // Fetched dependencies to mirror into hopper after this result is stored,
    // with the model version + analysis time their verdicts are stamped with.
    let mut dep_sync: Option<(Vec<crate::engine::DepResult>, String, String)> = None;
    let payload = match result {
        Ok((envelope, deps, duration_ms)) => {
            // v7 envelope no longer carries `class` on the wire; the verdict
            // is encoded in `lvl` (-1 = benign, anything else = hostile). The
            // suspicious band is consumer-side and not visible here.
            let verdict = if envelope.ml.level == Some(-1) {
                "benign"
            } else {
                "hostile"
            };
            tracing::info!(sha256 = %sha256, duration_ms, verdict, "analysis complete");
            if !deps.is_empty() {
                dep_sync = Some((
                    deps,
                    envelope.ml.version.clone(),
                    envelope.ml.analyzed_at.clone(),
                ));
            }
            crate::upload::ResultPayload {
                sha256: sha256.to_string(),
                worker: worker.to_string(),
                error: None,
                duration_ms,
                envelope: Some(envelope),
            }
        }
        Err(e) => crate::upload::ResultPayload {
            sha256: sha256.to_string(),
            worker: worker.to_string(),
            error: Some(e),
            duration_ms: 0,
            envelope: None,
        },
    };

    let url = format!("{}/api/result", url);

    // Serialize and compress once, then reuse the bytes across retries: cleave
    // reports are large, repetitive JSON that zstd shrinks 3-5x. Shared with the
    // local `scan path --hopper` uploader so both speak hopper's `/api/result`
    // byte-identically. `None` means serialization failed unrecoverably.
    let Some((body, encoding)) = crate::upload::encode_result_body(payload, sha256) else {
        return;
    };

    // Retry with the same exponential-backoff-with-jitter schedule as poll
    // failures (2s, 4s, 8s, 16s, 32s, then capped at ~60s) for up to
    // RETRY_BUDGET. Hopper only re-leases a dropped result after its 30-minute
    // claim expiry, so a ~20-minute retry window recovers most hopper restarts
    // and short outages without forcing a full re-analysis elsewhere. The post
    // is idempotent on hopper, so re-sending after an ambiguous timeout is safe.
    // Sleeps are deliberately not shutdown-interruptible: a worker shutting down
    // mid-retry loses at most one result, which the lease recovers anyway.
    const RETRY_BUDGET: Duration = Duration::from_secs(20 * 60);
    let started = Instant::now();
    let mut attempt: u32 = 0;
    loop {
        if attempt > 0 {
            tokio::time::sleep(backoff_duration(attempt)).await;
        }
        tracing::debug!(sha256 = %sha256, attempt, "posting result to server");
        let post_start = Instant::now();
        let mut request = client
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.clone());
        if let Some(enc) = encoding {
            request = request.header(reqwest::header::CONTENT_ENCODING, enc);
        }
        match request.send().await {
            Ok(resp) if resp.status().is_success() => {
                tracing::debug!(sha256 = %sha256, elapsed_ms = crate::duration_ms(post_start.elapsed()), attempt, "result posted");
                // The sample's row now exists on hopper; mirror its fetched
                // dependencies (bytes if missing, provenance, and verdict) as their
                // own samples. Best-effort and off the executor — never blocks or
                // fails the result that preceded it.
                if let Some((deps, version, analyzed_at)) = dep_sync.take() {
                    sync_worker_dependencies(
                        base_url.clone(),
                        worker.to_string(),
                        version,
                        analyzed_at,
                        deps,
                    )
                    .await;
                }
                return;
            }
            Ok(resp) => {
                let status = resp.status();
                let elapsed_ms = crate::duration_ms(post_start.elapsed());
                let body = resp.text().await.unwrap_or_default();
                // A 4xx means hopper rejected this exact payload; resending
                // identical bytes can never succeed, so retrying just burns
                // 20 minutes. 408 (timeout) and 429 (throttled) are the
                // transient exceptions.
                if status.is_client_error()
                    && status != reqwest::StatusCode::REQUEST_TIMEOUT
                    && status != reqwest::StatusCode::TOO_MANY_REQUESTS
                {
                    tracing::error!(sha256 = %sha256, %status, body = %body_excerpt(&body), elapsed_ms, attempt, "post result: rejected by server; not retrying");
                    return;
                }
                tracing::warn!(sha256 = %sha256, %status, body = %body_excerpt(&body), elapsed_ms, attempt, "post result: non-success response");
            }
            Err(e) => {
                tracing::warn!(sha256 = %sha256, error = %crate::upload::error_chain(&e), elapsed_ms = crate::duration_ms(post_start.elapsed()), attempt, "post result: send failed");
            }
        }
        attempt += 1;
        if started.elapsed() >= RETRY_BUDGET {
            break;
        }
    }
    tracing::error!(
        sha256 = %sha256,
        attempts = attempt,
        elapsed_s = started.elapsed().as_secs(),
        "post result: giving up after retry budget exhausted",
    );
}

/// Mirror a posted result's fetched dependencies into hopper as their own
/// samples, off the async executor. Each dependency's bytes come from the same
/// blob cache the analysis fetched them into (uploaded only if hopper lacks
/// them), paired with its provenance and the verdict scan already computed.
/// Best-effort: a blob cache that won't open, or any upload failure, is logged
/// inside the sync and never surfaced here.
async fn sync_worker_dependencies(
    base_url: String,
    worker: String,
    version: String,
    analyzed_at: String,
    deps: Vec<crate::engine::DepResult>,
) {
    let _ = tokio::task::spawn_blocking(move || {
        let cache = fletch::fetch::BlobCache::open().ok();
        crate::upload::sync_result_dependencies(
            dep_sync_client(),
            &base_url,
            &worker,
            &version,
            &analyzed_at,
            cache.as_ref(),
            deps,
        );
    })
    .await;
}

/// The shared blocking HTTP client for dependency mirroring, built once. Separate
/// from the worker's async client because the upload reconciliation is blocking
/// (it streams files and loads cache blobs); a clone is cheap (the client is an
/// `Arc` internally).
fn dep_sync_client() -> &'static reqwest::blocking::Client {
    static CLIENT: std::sync::OnceLock<reqwest::blocking::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new())
    })
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
) -> Result<bytes::Bytes, String> {
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
        return Ok(bytes);
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
    Ok(bytes)
}

/// Fetch the registry-metadata provenance hopper holds for `sha256`, returning
/// its normalized [`fletch::Registry`] record. Best-effort by design: an absent
/// record (HTTP 204), an unreachable hopper, or a malformed body all yield
/// `None` — registry provenance enriches a scan but must never fail one, exactly
/// as a live scan fails open when a registry lookup can't be made.
async fn download_provenance(
    client: &reqwest::Client,
    base_url: &str,
    sha256: &str,
) -> Option<fletch::Registry> {
    let url = format!("{base_url}/api/provenance/{sha256}");
    let resp = match client.get(&url).send().await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::debug!(sha256 = %sha256, error = %e, "provenance fetch failed");
            return None;
        }
    };
    if !resp.status().is_success() {
        tracing::debug!(sha256 = %sha256, status = %resp.status(), "no provenance");
        return None;
    }
    let body = match resp.bytes().await {
        Ok(body) => body,
        Err(e) => {
            tracing::debug!(sha256 = %sha256, error = %e, "provenance body read failed");
            return None;
        }
    };
    // 204 No Content (no stored provenance) arrives as an empty success body.
    if body.is_empty() {
        return None;
    }
    let record = crate::provenance::registry_record(&body);
    if let Some(reg) = &record {
        tracing::debug!(
            sha256 = %sha256,
            ecosystem = %reg.ecosystem,
            package = %reg.name,
            version = %reg.version,
            "registry provenance applied",
        );
    }
    record
}

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

/// Build the `/api/heartbeat` URL. Mirrors `Prefetcher::poll_url`'s live signals
/// (traits, RSS, load, tools) but claims no work and adds the worker's own queue
/// view: `active` running slots and `queue` staged-but-not-yet-dispatched jobs.
/// Live counts plus a metrics snapshot for one heartbeat.
/// Poll-side telemetry the prefetcher shares with the heartbeat task so a
/// check-in can explain *why* a worker isn't claiming: how full its buffer is,
/// what it last asked hopper for, what it got, and how long since it asked.
#[derive(Default)]
struct PollState {
    /// `WorkerMetrics::start`-relative seconds of the last `/api/next` attempt.
    /// A `poll_age` far past the poll cadence means the loop is wedged.
    last_poll_secs: AtomicU64,
    /// Jobs requested on the last poll.
    last_want: AtomicUsize,
    /// Jobs returned by the last poll (0 = hopper had nothing for this worker).
    last_claim: AtomicUsize,
    /// Free prefetch depth right now (`target_depth - outstanding`). 0 means the
    /// buffer is full, so the prefetcher is deliberately not polling — the
    /// signature of a worker saturated by slow jobs rather than starved.
    buffer_room: AtomicUsize,
}

struct HeartbeatReport {
    /// Configured analysis slots.
    slots: usize,
    /// Slots currently running an analysis.
    active: usize,
    /// Staged samples waiting for a free slot.
    queue: usize,
    /// In-flight memory reservation held by the admission gate, in MiB.
    mem_reserved_mb: u64,
    /// Memory ceiling that throttles intake (resolved `--max-rss-gb`), in MiB;
    /// 0 = gate disabled. This is the worker's RAM limit.
    mem_ceiling_mb: u64,
    /// Seconds since the prefetcher last polled `/api/next` (large = stalled).
    poll_age_s: u64,
    /// Jobs requested and returned on the last poll, and current free buffer room.
    last_want: usize,
    last_claim: usize,
    buffer_room: usize,
    metrics: MetricsSnapshot,
}

fn heartbeat_url(
    base_url: &str,
    encoded_name: &str,
    available_tools: &str,
    report: &HeartbeatReport,
) -> String {
    use std::fmt::Write;
    let metrics = &report.metrics;
    let mut url = format!(
        "{}/api/heartbeat?worker={}&slots={}&active={}&queue={}&version={}",
        base_url,
        encoded_name,
        report.slots,
        report.active,
        report.queue,
        env!("CARGO_PKG_VERSION"),
    );
    // 5-char prefix matches hopper's litmusTraitsVersion() truncation so the
    // dashboard's stale-traits comparison can string-equal the two.
    if let Some(traits) = cleave::traits_repo::version() {
        let prefix: String = traits.chars().take(5).collect();
        let _ = write!(url, "&traits={}", prefix);
    }
    if let Some(rss) = cleave::memory_tracker::current_rss() {
        let _ = write!(url, "&rss_mb={}", rss / 1024 / 1024);
    }
    if let Some(load) = system_load_avg() {
        let _ = write!(url, "&load1={:.2}", load);
    }
    // Local-queue metrics. Ages are sent in seconds (relative, not wall-clock)
    // so hopper renders "x ago" without depending on synchronised clocks.
    if let Some(age) = metrics.oldest_age {
        let _ = write!(url, "&oldest_s={}", age.as_secs());
    }
    if let Some(age) = metrics.last_completion_age {
        let _ = write!(url, "&done_age_s={}", age.as_secs());
    }
    let _ = write!(url, "&fps={:.3}", metrics.files_per_sec);
    let _ = write!(url, "&errs={}", metrics.errors_recent);
    if let Some((age, ref msg)) = metrics.last_error {
        let _ = write!(url, "&err_age_s={}&err=", age.as_secs());
        // Trim to keep the URL bounded; hopper only displays a short summary.
        let trimmed: String = msg.chars().take(200).collect();
        url_encode_into(&trimmed, &mut url);
    }
    // Admission / poll diagnostics: why the worker is (or isn't) claiming.
    // mem_ceiling_mb is the RAM limit that throttles intake; buffer_room=0 with
    // a large poll_age_s means it's saturated by slow jobs, not starved.
    let _ = write!(url, "&mem_reserved_mb={}", report.mem_reserved_mb);
    let _ = write!(url, "&mem_ceiling_mb={}", report.mem_ceiling_mb);
    let _ = write!(url, "&poll_age_s={}", report.poll_age_s);
    let _ = write!(url, "&want={}", report.last_want);
    let _ = write!(url, "&last_claim={}", report.last_claim);
    let _ = write!(url, "&buffer_room={}", report.buffer_room);
    let _ = write!(url, "&tools=");
    url_encode_into(available_tools, &mut url);
    url
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

    // Minimal in-process hopper: serves `/api/next` (returning exactly the
    // requested `count` of distinct jobs, an unlimited supply) and `/data/...`
    // (a fixed payload). Zero extra dependencies — just tokio, already in tree.
    async fn read_target(stream: &mut tokio::net::TcpStream) -> Option<String> {
        use tokio::io::AsyncReadExt;
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            let n = stream.read(&mut tmp).await.ok()?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
            if buf.len() > 16 * 1024 {
                break;
            }
        }
        let text = String::from_utf8_lossy(&buf);
        let line = text.lines().next()?;
        line.split_whitespace().nth(1).map(str::to_string)
    }

    async fn respond(stream: &mut tokio::net::TcpStream, status: &str, body: &[u8]) {
        use tokio::io::AsyncWriteExt;
        let head = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len(),
        );
        let _ = stream.write_all(head.as_bytes()).await;
        let _ = stream.write_all(body).await;
        let _ = stream.flush().await;
    }

    fn parse_count(target: &str) -> usize {
        target
            .split(['?', '&'])
            .find_map(|kv| kv.strip_prefix("count="))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }

    async fn wait_until(mut cond: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if cond() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        cond()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prefetcher_fills_to_target_backpressures_and_refills() {
        const PAYLOAD: &[u8] = b"payload";

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let api_calls = Arc::new(AtomicUsize::new(0));
        let next_id = Arc::new(AtomicUsize::new(0));
        {
            let api_calls = Arc::clone(&api_calls);
            let next_id = Arc::clone(&next_id);
            tokio::spawn(async move {
                loop {
                    let Ok((mut stream, _)) = listener.accept().await else {
                        break;
                    };
                    let api_calls = Arc::clone(&api_calls);
                    let next_id = Arc::clone(&next_id);
                    tokio::spawn(async move {
                        let Some(target) = read_target(&mut stream).await else {
                            return;
                        };
                        if target.starts_with("/api/next") {
                            api_calls.fetch_add(1, Ordering::Relaxed);
                            let count = parse_count(&target);
                            let jobs: Vec<_> = (0..count)
                                .map(|_| {
                                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                                    serde_json::json!({
                                        "sha256": format!("{id:064x}"),
                                        "path": format!("samples/s{id}.bin"),
                                        "size_bytes": PAYLOAD.len(),
                                        "file_type": "data",
                                    })
                                })
                                .collect();
                            let body = serde_json::json!({ "jobs": jobs }).to_string();
                            respond(&mut stream, "200 OK", body.as_bytes()).await;
                        } else if target.starts_with("/data/") {
                            respond(&mut stream, "200 OK", PAYLOAD).await;
                        } else {
                            respond(&mut stream, "404 Not Found", b"").await;
                        }
                    });
                }
            });
        }

        let slots = 3usize;
        let target_depth = slots * 3;
        let (tx, mut rx) = mpsc::unbounded_channel::<PrefetchedJob>();
        let queued_bytes = Arc::new(AtomicUsize::new(0));
        let outstanding = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));

        let handle = tokio::spawn(
            Prefetcher {
                client: reqwest::Client::new(),
                base_url: Arc::from(format!("http://127.0.0.1:{port}").as_str()),
                data_dir: None,
                encoded_name: "test".to_string(),
                available_tools: String::new(),
                slots,
                max_single_bytes: 1 << 20,
                max_buffer_bytes: 1 << 30,
                poll_secs: 1,
                target_depth,
                metrics: Arc::new(WorkerMetrics::new()),
                poll_state: Arc::new(PollState::default()),
                exit_if_empty: false,
            }
            .run(
                tx,
                Arc::clone(&queued_bytes),
                Arc::clone(&outstanding),
                Arc::clone(&shutdown),
            ),
        );

        // 1. Fills to exactly target_depth — every staged sample lands in the
        //    channel — and never overshoots the cap.
        assert!(
            wait_until(|| rx.len() == target_depth).await,
            "prefetcher did not fill to target_depth; channel len {}",
            rx.len(),
        );
        assert_eq!(outstanding.load(Ordering::Relaxed), target_depth);
        assert!(outstanding.load(Ordering::Relaxed) <= target_depth);

        // 2. Backpressure: once full it stops polling the hopper.
        let calls_when_full = api_calls.load(Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(
            api_calls.load(Ordering::Relaxed),
            calls_when_full,
            "prefetcher kept polling while the buffer was full",
        );

        // 3. Draining samples (as the dispatch loop would) frees room and it
        //    refills back to target, polling the hopper again.
        for _ in 0..slots {
            let pj = rx.recv().await.unwrap();
            assert_eq!(
                pj.data.unwrap().as_deref(),
                Some(PAYLOAD),
                "payload mismatch"
            );
            queued_bytes.fetch_sub(PAYLOAD.len(), Ordering::Release);
            outstanding.fetch_sub(1, Ordering::Release);
        }
        assert!(
            wait_until(|| outstanding.load(Ordering::Relaxed) == target_depth).await,
            "prefetcher did not refill after draining",
        );
        assert!(
            api_calls.load(Ordering::Relaxed) > calls_when_full,
            "prefetcher should have polled again to refill",
        );

        // 4. Shutdown stops the prefetcher and closes the channel.
        shutdown.store(true, Ordering::Relaxed);
        assert!(
            tokio::time::timeout(Duration::from_secs(5), handle)
                .await
                .is_ok(),
            "prefetcher did not exit on shutdown",
        );
        while rx.recv().await.is_some() {}
    }

    fn staged_pj(sha: &str, size_bytes: i64) -> PrefetchedJob {
        PrefetchedJob {
            job: ClaimJob {
                sha256: sha.to_string(),
                path: format!("{sha}.bin"),
                size_bytes,
                file_type: "data".to_string(),
                has_provenance: false,
            },
            data: Ok(None),
            queue_id: 0,
        }
    }

    #[tokio::test]
    async fn sjf_picks_smallest_staged_job_first() {
        let (tx, mut rx) = mpsc::unbounded_channel::<PrefetchedJob>();
        let mut reorder = Vec::new();
        tx.send(staged_pj("big", 500 * 1024 * 1024)).unwrap();
        tx.send(staged_pj("tiny", 4 * 1024)).unwrap();
        tx.send(staged_pj("mid", 8 * 1024 * 1024)).unwrap();

        let order = [
            next_smallest_staged(&mut rx, &mut reorder).await.unwrap(),
            next_smallest_staged(&mut rx, &mut reorder).await.unwrap(),
            next_smallest_staged(&mut rx, &mut reorder).await.unwrap(),
        ];
        let shas: Vec<&str> = order.iter().map(|pj| pj.job.sha256.as_str()).collect();
        assert_eq!(shas, ["tiny", "mid", "big"]);

        // Channel closed and window drained → None, like recv().
        drop(tx);
        assert!(next_smallest_staged(&mut rx, &mut reorder).await.is_none());
    }

    #[tokio::test]
    async fn sjf_drains_reorder_window_after_channel_close() {
        // Jobs staged in the window must still dispatch after the prefetcher
        // exits, or --exit-if-empty would drop the tail of the queue.
        let (tx, mut rx) = mpsc::unbounded_channel::<PrefetchedJob>();
        let mut reorder = Vec::new();
        tx.send(staged_pj("a", 100)).unwrap();
        tx.send(staged_pj("b", 50)).unwrap();
        drop(tx);

        assert_eq!(
            next_smallest_staged(&mut rx, &mut reorder)
                .await
                .unwrap()
                .job
                .sha256,
            "b"
        );
        assert_eq!(
            next_smallest_staged(&mut rx, &mut reorder)
                .await
                .unwrap()
                .job
                .sha256,
            "a"
        );
        assert!(next_smallest_staged(&mut rx, &mut reorder).await.is_none());
    }

    #[tokio::test]
    async fn sjf_ages_long_waiting_job_to_front() {
        let (tx, mut rx) = mpsc::unbounded_channel::<PrefetchedJob>();
        // A big job already staged longer than the aging bound beats a fresh
        // tiny job, so SJF cannot starve archives indefinitely.
        let mut reorder = vec![(
            staged_pj("old-big", 500 * 1024 * 1024),
            Instant::now() - SJF_MAX_STAGED_WAIT,
        )];
        tx.send(staged_pj("fresh-tiny", 4 * 1024)).unwrap();

        assert_eq!(
            next_smallest_staged(&mut rx, &mut reorder)
                .await
                .unwrap()
                .job
                .sha256,
            "old-big"
        );
    }

    #[test]
    fn rate_window_counts_only_the_trailing_window() {
        let mut w = RateWindow::new();
        // Three completions in minute 100, two in minute 101.
        w.record(100);
        w.record(100);
        w.record(100);
        w.record(101);
        w.record(101);
        // At minute 101 all five are within the trailing 15 one-minute buckets.
        assert!((w.per_sec(101) - 5.0 / 900.0).abs() < 1e-9);
        // The window spans diffs 0..14 (15 buckets). At minute 115 the minute-100
        // bucket has aged out (diff 15) but the minute-101 bucket (diff 14) holds,
        // leaving its two completions.
        assert!((w.per_sec(115) - 2.0 / 900.0).abs() < 1e-9);
        // Far in the future every bucket has aged out.
        assert_eq!(w.per_sec(200), 0.0);
    }

    #[test]
    fn rate_window_bucket_reuse_resets_stale_minute() {
        let mut w = RateWindow::new();
        w.record(0); // slot 0 holds minute 0
        w.record(15); // minute 15 reuses slot 0; must reset, not accumulate
        assert!((w.per_sec(15) - 1.0 / 900.0).abs() < 1e-9);
    }

    #[test]
    fn error_window_prunes_and_keeps_last() {
        let mut e = ErrorWindow::default();
        let base = Instant::now();
        e.record("old", base);
        e.record("recent", base + Duration::from_secs(60));
        // Pruning relative to a moment just past the window from `base` drops the
        // first error but keeps the second, while `last` still reflects "recent".
        e.prune(base + METRICS_WINDOW + Duration::from_secs(1));
        assert_eq!(e.times.len(), 1);
        assert_eq!(e.last.as_ref().map(|(_, m)| m.as_str()), Some("recent"));
    }

    #[test]
    fn worker_metrics_track_queue_completion_and_errors() {
        let m = WorkerMetrics::new();
        let a = m.enqueue();
        let _b = m.enqueue();
        // Two items queued; an oldest age exists.
        let snap = m.snapshot();
        assert!(snap.oldest_age.is_some());
        assert!(snap.last_completion_age.is_none());
        assert_eq!(snap.errors_recent, 0);

        m.record_error("boom");
        m.complete(a);
        let snap = m.snapshot();
        // One item still queued, one completion recorded, one error in window.
        assert!(snap.oldest_age.is_some());
        assert!(snap.last_completion_age.is_some());
        assert_eq!(snap.errors_recent, 1);
        assert_eq!(
            snap.last_error.as_ref().map(|(_, msg)| msg.as_str()),
            Some("boom")
        );
    }
}
