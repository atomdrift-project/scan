//! Pull-based worker that polls a hopper instance for analysis jobs.
//!
//! Shape (deliberately boring):
//!
//! ```text
//!   prefetcher ──► job channel ──► N worker tasks (N = --workers)
//! ```
//!
//! Each worker loops: take job → admit memory → cleave-gate → analyze → post.
//! There is no central dispatcher and no analysis-slot semaphore — the N tasks
//! *are* the concurrency limit. A permit is never held across a wait that does
//! not need it: cleave covers only the blocking classify; hopper I/O runs after
//! admit/cleave are dropped so a wedged hopper cannot freeze analysis.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::hash::{BuildHasherDefault, Hasher};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex as AsyncMutex, Semaphore, mpsc};
use tokio::task::JoinSet;

use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Model, Thresholds};
use crate::server::{ModelResources, classify_bytes, classify_file};
use crate::system_load_avg;
use crate::upload::{hopper_token, log_hopper_credential};

/// Attach the hopper bearer token to a request, if there is one.
///
/// Every call to hopper goes through this: hopper requires the token on all of
/// `/api/*` and `/data/`, and does not exempt loopback — a locally supervised
/// worker on the hopper host authenticates like any remote one. See
/// [`crate::upload::hopper_token`] for where the token comes from.
fn authed(request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match hopper_token() {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

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

/// How many top-level analyses may execute at once.
///
/// The N worker tasks bound jobs that may be claimed/in flight; this tighter
/// gate bounds memory-bandwidth-heavy cleave executions. Within that gate,
/// cleave gives only a bounded subset of sibling analyses access to nested
/// Rayon work while the other admitted analyses make serial progress. The
/// default scales with the pool (1/16, at least 1: eight analyses on the
/// 128-thread FreeBSD worker) and never exceeds `slots`;
/// `SCAN_CLEAVE_CONCURRENCY` remains an explicit production override.
///
/// Each worker waits on this gate *after* taking a job and *only* around the
/// blocking classify — never on a shared dispatch loop.
fn cleave_concurrency(slots: usize) -> usize {
    let override_value = std::env::var("SCAN_CLEAVE_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    cleave_concurrency_from(slots, rayon::current_num_threads(), override_value)
}

/// Pure gate sizing — `pool` is the Rayon thread count, `override_value` is
/// `SCAN_CLEAVE_CONCURRENCY` when set.
fn cleave_concurrency_from(slots: usize, pool: usize, override_value: Option<usize>) -> usize {
    let slots = slots.max(1);
    override_value.filter(|&value| value > 0).map_or_else(
        || (pool.max(1) / 16).clamp(1, slots),
        |value| value.min(slots),
    )
}

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

/// Resolve `requested_path` against `root` using nothing but the filesystem:
/// try the path as given (and, for an absolute path, its canonical form in case
/// the prefix is symlinked), then confirm size and SHA-256 before trusting the
/// match.
///
/// This is the whole data-serving path. Any sample still sitting where hopper
/// says it does resolves here, which is the overwhelmingly common case and
/// needs no index at all. `LocalFileIndex` exists only to catch the remainder —
/// samples that have moved out from under their recorded path — so it is a
/// best-effort accelerator layered on top of this, never a prerequisite for it.
fn resolve_on_disk(
    root: &Path,
    requested_path: &str,
    expected: &[u8; 32],
    size_bytes: i64,
) -> Option<PathBuf> {
    let requested = Path::new(requested_path);
    let mut candidates: Vec<PathBuf> = Vec::new();
    if requested.is_relative() {
        candidates.push(root.join(requested));
    } else if requested.is_absolute() {
        candidates.push(requested.to_path_buf());
        if let Ok(resolved) = requested.canonicalize()
            && resolved != requested
        {
            candidates.push(resolved);
        }
    }

    let expected_size = u64::try_from(size_bytes).ok();
    for candidate in &candidates {
        let meta = match fs::metadata(candidate) {
            Ok(m) if m.is_file() => m,
            _ => continue,
        };
        if expected_size.is_some_and(|sz| meta.len() != sz) {
            continue;
        }
        if let Ok(digest) = sha256_file(candidate)
            && digest == *expected
        {
            tracing::debug!(
                path = %candidate.display(),
                "resolved local file by path and sha256 verification",
            );
            return Some(candidate.clone());
        }
    }

    None
}

impl LocalFileIndex {
    /// Directories walked between progress lines. The walk is the longest
    /// single step in worker startup on a large corpus; without a periodic
    /// line there is no way to tell "still indexing" from "hung" from a log.
    const PROGRESS_EVERY_DIRS: usize = 25_000;

    /// Threads used for the startup walk. `read_dir` and the per-file `stat`
    /// are both I/O-bound, so this is queue depth for the storage device
    /// rather than CPU parallelism — one thread leaves any device with real
    /// seek latency almost entirely idle, which is what made a 3.5 M-file
    /// corpus on a spindle take longer to index than hopper's wedge timeout.
    fn walk_threads() -> usize {
        std::env::var("SCAN_INDEX_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(16)
    }

    // Returns Result for forward-compatibility: individual dir-entry failures
    // are currently logged and skipped, but a future cap on I/O errors, a
    // permission-denied signal, or a root-missing fail-fast policy would want
    // to bubble up here.
    fn build(root: PathBuf) -> Result<Self> {
        let started = Instant::now();
        let threads = Self::walk_threads();
        tracing::info!(root = %root.display(), threads, "indexing local samples");

        // A private pool, not the global one: these threads block on I/O for
        // as long as the walk runs, and the global pool is where cleave runs
        // analysis. Borrowing it here would park every in-flight job behind
        // the walk.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("sample-index-{i}"))
            .build()
            .context("building sample index thread pool")?;

        let batches: Mutex<Vec<Vec<IndexedLocalFile>>> = Mutex::new(Vec::new());
        let dirs_walked = AtomicUsize::new(0);
        pool.scope(|scope| {
            Self::walk_dir(&root, scope, &batches, &dirs_walked, started);
        });

        let mut files: Vec<IndexedLocalFile> = batches
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
            .into_iter()
            .flatten()
            .collect();

        // `FileId` is `u32`; drop the tail rather than silently truncating an
        // id if a single data dir somehow exceeds 4 B files.
        if files.len() > FileId::MAX as usize {
            tracing::warn!(
                root = %root.display(),
                found = files.len(),
                limit = FileId::MAX,
                "local data index exceeds FileId capacity; ignoring remaining files",
            );
            files.truncate(FileId::MAX as usize);
        }

        // Built serially from the merged file list. `FileId` is an opaque
        // handle and every lookup re-verifies size and content hash, so the
        // order the parallel walk happened to produce carries no meaning.
        let mut by_name: HashMap<LocalNameKey, Vec<FileId>> = HashMap::new();
        for (idx, file) in files.iter().enumerate() {
            let Some(basename) = file.path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let parent_name = file
                .path
                .parent()
                .and_then(Path::file_name)
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            // Bounded by the truncate above.
            #[allow(clippy::cast_possible_truncation)]
            let file_id = idx as FileId;
            by_name
                .entry(LocalNameKey {
                    parent_name,
                    basename: basename.to_string(),
                })
                .or_default()
                .push(file_id);
        }

        let indexed_files = files.len();
        let distinct_names = by_name.len();
        tracing::info!(
            root = %root.display(),
            indexed_files,
            distinct_names,
            dirs_walked = dirs_walked.load(Ordering::Relaxed),
            elapsed_s = started.elapsed().as_secs(),
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

    /// Index one directory, spawning a sibling task per subdirectory found.
    /// Files are collected per directory and merged in one shot, so the shared
    /// lock is taken once per directory rather than once per file.
    ///
    /// Symlinks are not followed: `file_type` reports them as neither dir nor
    /// file, so they are skipped and the walk cannot cycle.
    fn walk_dir<'scope>(
        dir: &Path,
        scope: &rayon::Scope<'scope>,
        batches: &'scope Mutex<Vec<Vec<IndexedLocalFile>>>,
        dirs_walked: &'scope AtomicUsize,
        started: Instant,
    ) {
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(e) => {
                tracing::warn!(path = %dir.display(), error = %e, "failed to read local data directory entry");
                return;
            }
        };

        let mut found: Vec<IndexedLocalFile> = Vec::new();
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    tracing::warn!(path = %dir.display(), error = %e, "failed to enumerate local data directory entry");
                    continue;
                }
            };
            let path = entry.path();
            // Served from the readdir buffer's `d_type` on Linux, so this
            // costs no syscall; only the size below needs a stat.
            let file_type = match entry.file_type() {
                Ok(ft) => ft,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "failed to read local file type");
                    continue;
                }
            };
            if file_type.is_dir() {
                scope.spawn(move |scope| {
                    Self::walk_dir(&path, scope, batches, dirs_walked, started);
                });
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
            found.push(IndexedLocalFile { path, size });
        }

        if !found.is_empty() {
            batches
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(found);
        }

        let walked = dirs_walked.fetch_add(1, Ordering::Relaxed) + 1;
        if walked.is_multiple_of(Self::PROGRESS_EVERY_DIRS) {
            tracing::info!(
                dirs_walked = walked,
                elapsed_s = started.elapsed().as_secs(),
                "indexing local samples",
            );
        }
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
        // (e.g. newly harvested), or never have moved in the first place.
        Ok(resolve_on_disk(
            &self.root,
            requested_path,
            &expected,
            size_bytes,
        ))
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
    pub workers: NonZeroUsize,
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
    /// Additional passwords to try for encrypted archives.
    pub zip_passwords: crate::ArchivePasswords,
    /// Set when this worker runs inside `atomscan serve` rather than as its own
    /// process. See [`Embedded`].
    pub embedded: Option<Embedded>,
}

/// Wiring for a worker running inside a serve process, filling idle capacity
/// with queue work.
///
/// Three things change when embedded, and each is a correctness issue rather
/// than a preference:
///
///   - **Signals and nice belong to the host.** Installing a second SIGTERM
///     handler, or renicing the process, would reach the server's own request
///     handling.
///   - **The models are already loaded.** Loading a second copy would double
///     the resident set of the largest thing in the process, on hosts we size
///     deliberately.
///   - **Interactive work comes first.** `pause` is raised while a user request
///     is in flight; the prefetcher stops claiming and the queue drains. It
///     does not abandon a job already running — that work is real and a claim
///     that dies is redispatched by hopper anyway — so responsiveness comes
///     from leaving slots free, not from killing work mid-flight.
#[derive(Clone)]
pub struct Embedded {
    /// Raised by the server while interactive requests are in flight.
    pub pause: Arc<AtomicBool>,
    /// The host's shutdown flag, so one signal stops both.
    pub shutdown: Arc<AtomicBool>,
    /// The server's already-loaded models.
    pub resources: Arc<ModelResources>,
    /// Elapsed-time marker for the most recent analysis request.
    pub last_analyze_request_ms: Arc<AtomicU64>,
    /// Same monotonic clock anchor used to produce the request marker.
    pub started_at: Instant,
    /// How long after an analysis request the idle worker must remain quiet.
    pub quiet_period: Duration,
}

// ModelResources carries no Debug, and dumping a model bundle into a log line
// would help nobody; report the state an operator can act on.
impl std::fmt::Debug for Embedded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedded")
            .field("paused", &self.pause.load(Ordering::Relaxed))
            .field("shutdown", &self.shutdown.load(Ordering::Relaxed))
            .field("quiet_period", &self.quiet_period)
            .finish_non_exhaustive()
    }
}

/// Settings that must survive every model load and periodic renewal unchanged.
#[derive(Debug)]
struct ResourceConfig {
    model_dir: PathBuf,
    thresholds: Option<Thresholds>,
    slow_rule_ms: u64,
    level: Option<u16>,
    interpret: Option<crate::interpret::InterpretConfig>,
    fetch: crate::fetch::FetchPolicy,
    zip_passwords: crate::ArchivePasswords,
}

fn load_model_resources(config: &ResourceConfig) -> Result<Arc<ModelResources>> {
    let model =
        Model::load(&config.model_dir, config.thresholds, config.level).context("loading model")?;
    let shap = ShapImportance::load(&config.model_dir).ok();
    let ctx = ExtractContext::new(model.spec());
    Ok(Arc::new(ModelResources {
        model,
        shap,
        ctx,
        interpret: config.interpret.clone(),
        // Per-job scanning honors the worker's fetch policy (`SCAN_FETCH`). The
        // fixed validate corpus never fetches — it runs through
        // `crate::validate::run`, which builds its own offline resources.
        fetch: config.fetch,
        zip_passwords: config.zip_passwords.clone(),
    }))
}

fn validate_and_load_resources(config: &ResourceConfig) -> Result<Arc<ModelResources>> {
    let validate_config = crate::ScanConfig::new(
        &config.model_dir,
        crate::OutputFormat::Terminal,
        config.thresholds,
        crate::DisplayFilter::alerts_only(),
        config.slow_rule_ms,
        false,
    )?
    .with_level(config.level)
    .with_zip_passwords(config.zip_passwords.clone());
    crate::validate::run(&validate_config, false)?;
    load_model_resources(config)
}

/// Pull upstream rules and, **only if something actually changed**, re-validate
/// and reload the model bundle. Returns `Ok(None)` when both repos are already
/// up to date — a silent no-op so the periodic renewal doesn't flood the log
/// with a full validation pass every interval.
fn renew_resources_once(config: &ResourceConfig) -> Result<Option<Arc<ModelResources>>> {
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

    let resources = validate_and_load_resources(config)?;

    let (traits, composites) = cleave::reload_capability_mapper()
        .map_err(|error| anyhow::anyhow!("reload cleave capability mapper: {error}"))?;
    tracing::info!(traits, composites, "cleave capability mapper renewed");

    Ok(Some(resources))
}

fn spawn_resource_renewal_task(
    handle: ResourceHandle,
    config: Arc<ResourceConfig>,
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
            let config = Arc::clone(&config);
            let result = tokio::task::spawn_blocking(move || renew_resources_once(&config)).await;

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
// per-slot pool grid — N pull-style worker tasks share the one process-global
// rayon pool that main installs (sized to the host's cores, 256 MB stacks).
// Cleave's `par_iter` fan-out work-steals across that pool, so a single large
// archive can use the whole machine while total rayon threads stay capped at
// the pool size (not `slots × per-slot-threads`), which in turn caps cleave's
// per-thread YARA scanners.

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

/// Shared job intake for the N worker tasks.
///
/// Tokio's `mpsc::Receiver` is single-consumer, so workers serialize briefly on
/// this mutex to pull the next SJF/FIFO job. The lock is held across an empty
/// `recv().await` — that is fine: with no work, every worker is idle anyway.
/// As soon as a job is taken the lock drops and analysis runs concurrently.
///
/// SJF policy matches the historical dispatcher: sweep the prefetch channel
/// into a reorder window, prefer the smallest sample, age out long-waiters
/// after [`SJF_MAX_STAGED_WAIT`]. `SCAN_SJF=0` restores FIFO.
struct JobSource {
    state: AsyncMutex<JobSourceState>,
}

struct JobSourceState {
    rx: mpsc::UnboundedReceiver<PrefetchedJob>,
    reorder: Vec<(PrefetchedJob, Instant)>,
    sjf: bool,
}

impl JobSource {
    fn new(rx: mpsc::UnboundedReceiver<PrefetchedJob>, sjf: bool) -> Self {
        Self {
            state: AsyncMutex::new(JobSourceState {
                rx,
                reorder: Vec::new(),
                sjf,
            }),
        }
    }

    async fn recv(&self) -> Option<PrefetchedJob> {
        let mut state = self.state.lock().await;
        if !state.sjf {
            return state.rx.recv().await;
        }
        // SJF via `&mut state` field access (not two simultaneous &mut borrows
        // of sibling fields into an async fn — that fails to compile).
        while let Ok(pj) = state.rx.try_recv() {
            state.reorder.push((pj, Instant::now()));
        }
        if state.reorder.is_empty() {
            let first = state.rx.recv().await?;
            state.reorder.push((first, Instant::now()));
            while let Ok(pj) = state.rx.try_recv() {
                state.reorder.push((pj, Instant::now()));
            }
        }
        pick_sjf_from_reorder(&mut state.reorder)
    }
}

/// Test-facing SJF picker over a bare channel + reorder window. Production
/// intake goes through [`JobSource::recv`], which inlines the same policy.
#[cfg(test)]
async fn next_smallest_staged(
    rx: &mut mpsc::UnboundedReceiver<PrefetchedJob>,
    reorder: &mut Vec<(PrefetchedJob, Instant)>,
) -> Option<PrefetchedJob> {
    while let Ok(pj) = rx.try_recv() {
        reorder.push((pj, Instant::now()));
    }
    if reorder.is_empty() {
        let first = rx.recv().await?;
        reorder.push((first, Instant::now()));
        while let Ok(pj) = rx.try_recv() {
            reorder.push((pj, Instant::now()));
        }
    }
    pick_sjf_from_reorder(reorder)
}

fn pick_sjf_from_reorder(reorder: &mut Vec<(PrefetchedJob, Instant)>) -> Option<PrefetchedJob> {
    let now = Instant::now();
    let max_wait = sjf_max_staged_wait();
    let aged = reorder
        .iter()
        .enumerate()
        .filter(|(_, (_, staged_at))| now.duration_since(*staged_at) >= max_wait)
        .min_by_key(|(_, (_, staged_at))| *staged_at)
        .map(|(i, _)| i);
    let idx = aged.or_else(|| {
        reorder
            .iter()
            .enumerate()
            .min_by_key(|(_, (pj, _))| pj.job.size_bytes.max(0))
            .map(|(i, _)| i)
    })?;
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
    /// `Ok(data)` = payload staged (in memory, spooled to disk, or local),
    /// `Err(Transient)` = download failed (fall back to direct download),
    /// `Err(Skipped)` = job rejected without attempting download (e.g. oversized);
    /// do not retry, post the error result directly.
    data: std::result::Result<PrefetchData, PrefetchError>,
    /// Local-queue id assigned by the prefetcher when the job is staged; passed
    /// to `WorkerMetrics::complete` once analysis finishes. 0 until staged.
    queue_id: u64,
}

/// Absolute per-job size cap, advertised to hopper as `max_bytes` so it never
/// hands out files no worker will analyze, and enforced locally as a backstop
/// for older hoppers. Anything at or below this is analyzable on any worker —
/// even a 16 GiB sample on an 8 GiB host — because oversized payloads stream to
/// the disk spool and take the file-path analysis route (mmap + on-disk archive
/// extraction) instead of being buffered in RAM. Hopper matches the rejection
/// message ("exceeds per-job" → skip='oversized'), so keep them in sync.
const MAX_JOB_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Where a staged job's payload lives until analysis.
enum PrefetchData {
    /// The file exists under `--data`; analyze it in place, nothing staged.
    Local,
    /// Downloaded into memory (small files); counted against the RAM buffer.
    Memory(bytes::Bytes),
    /// Streamed to a spool file on disk (files too big for the RAM buffer);
    /// counted against the disk spool budget until the payload drops.
    Spooled(SpooledPayload),
}

impl PrefetchData {
    /// Bytes this payload holds in RAM while staged (spooled and local payloads
    /// cost no buffer memory).
    fn staged_mem_bytes(&self) -> usize {
        match self {
            Self::Memory(b) => b.len(),
            Self::Local | Self::Spooled(_) => 0,
        }
    }
}

/// A payload streamed to disk. Dropping it deletes the spool file and releases
/// its reservation from the spool budget.
struct SpooledPayload {
    /// Deletes the file on drop. Declared before `spool` so the file is gone
    /// before the budget reopens.
    path: tempfile::TempPath,
    size: u64,
    spool: Arc<SpoolState>,
}

impl Drop for SpooledPayload {
    fn drop(&mut self) {
        self.spool.release(self.size);
    }
}

/// Disk spool for payloads too large to stage in the RAM buffer. Files between
/// `mem_threshold_bytes` and [`MAX_JOB_BYTES`] are streamed here and analyzed
/// via the file-path route (cleave memory-maps large files), so a 16 GiB sample
/// never has to fit in RAM.
#[derive(Debug)]
struct SpoolState {
    dir: PathBuf,
    /// Total bytes of concurrently-staged spool files allowed on disk.
    budget_bytes: u64,
    used: AtomicU64,
    /// Payloads at or below this stage in RAM; larger ones spool to disk.
    mem_threshold_bytes: usize,
    /// Free space the spool leaves on its filesystem beyond the file itself —
    /// cleave's archive extraction can write up to its 7 GiB guard on top of
    /// the spooled payload.
    disk_headroom_bytes: u64,
}

impl SpoolState {
    const DEFAULT_DISK_HEADROOM_BYTES: u64 = 8 * 1024 * 1024 * 1024;

    fn new(mem_threshold_bytes: usize) -> Self {
        let dir = std::env::var_os("SCAN_SPOOL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("scan-spool"));
        let budget_bytes = std::env::var("SCAN_SPOOL_BUDGET_GB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|gb| *gb > 0)
            .map_or(32 * 1024 * 1024 * 1024, |gb| gb * 1024 * 1024 * 1024);
        Self {
            dir,
            budget_bytes,
            used: AtomicU64::new(0),
            mem_threshold_bytes,
            disk_headroom_bytes: Self::DEFAULT_DISK_HEADROOM_BYTES,
        }
    }

    /// Reserve `size` bytes of spool space, or explain why not. Admission is
    /// gated on the concurrent-spool budget and on live free disk space; like
    /// the memory gate, an idle spool always admits one payload so a tight
    /// budget cannot starve large files forever.
    fn try_reserve(&self, size: u64) -> Result<(), String> {
        // The spool dir can vanish under a long-lived worker (a Windows %TEMP%
        // sweep removes it once it is empty), and `free_disk_bytes` returns
        // `None` for a missing path — silently skipping the disk check. Heal it
        // here so the check below measures the filesystem we will actually
        // write to.
        self.ensure_dir()?;
        let used = self.used.load(Ordering::Acquire);
        if used > 0 && used.saturating_add(size) > self.budget_bytes {
            return Err(format!(
                "spool budget full ({used} of {} bytes in use)",
                self.budget_bytes
            ));
        }
        if let Some(free) = free_disk_bytes(&self.dir)
            && free < size.saturating_add(self.disk_headroom_bytes)
        {
            return Err(format!(
                "insufficient free disk for spool: {free} bytes free, need {size} + {} headroom",
                self.disk_headroom_bytes
            ));
        }
        self.used.fetch_add(size, Ordering::AcqRel);
        Ok(())
    }

    fn release(&self, size: u64) {
        self.used.fetch_sub(size, Ordering::AcqRel);
    }

    /// Ensure the spool directory exists, creating it if it does not.
    ///
    /// Called on every spool admission and every spool write, not just at
    /// startup: the directory lives under `%TEMP%`/`/tmp`, and an OS temp sweep
    /// (Windows Storage Sense, `systemd-tmpfiles`) will delete it out from under
    /// a worker that has been up for days — it looks like an abandoned empty
    /// directory. Without this, a create-once spool leaves every payload above
    /// `mem_threshold_bytes` failing with "cannot create spool file" for the
    /// rest of the process's life, and the direct-download retry path fails the
    /// same way because it lands in the same missing directory.
    fn ensure_dir(&self) -> Result<(), String> {
        if self.dir.is_dir() {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| {
            format!(
                "cannot create spool dir {}: {e}",
                self.dir.display()
            )
        })
    }

    /// Create the spool directory and clear leftovers from crashed runs.
    /// Best-effort: only files older than a day are removed, so concurrent
    /// worker processes on the same host cannot delete each other's live
    /// spools.
    fn prepare(&self) {
        if let Err(e) = self.ensure_dir() {
            tracing::warn!(dir = %self.dir.display(), error = %e, "cannot create spool dir");
            return;
        }
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return;
        };
        let cutoff = Duration::from_secs(24 * 60 * 60);
        for entry in entries.flatten() {
            let stale = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age > cutoff);
            if stale && std::fs::remove_file(entry.path()).is_ok() {
                tracing::info!(path = %entry.path().display(), "removed stale spool file");
            }
        }
    }
}

/// Free bytes on the filesystem holding `path`, or `None` when unavailable.
#[cfg(unix)]
fn free_disk_bytes(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: c_path is a valid NUL-terminated path and stat is a valid
    // out-pointer for the duration of the call.
    if unsafe { libc::statvfs(c_path.as_ptr(), &raw mut stat) } != 0 {
        return None;
    }
    #[allow(clippy::unnecessary_cast)] // f_bavail/f_frsize widths vary by platform
    Some(stat.f_bavail as u64 * stat.f_frsize as u64)
}

#[cfg(not(unix))]
fn free_disk_bytes(_path: &Path) -> Option<u64> {
    None
}

/// Why a prefetch did not produce bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
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
/// after a shutdown signal before exiting anyway. Kept short so a redeploy is
/// snappy: every service supervisor (FreeBSD rc.d, systemd `TimeoutStopSec`,
/// launchd) SIGKILLs the process a few seconds after this as a backstop, so the
/// worker must exit within their grace window or be force-killed mid-drain.
/// Cleave cancellation is cooperative and a stuck rayon unpack can refuse to
/// exit; whatever does not finish here is re-leased by hopper, so exiting early
/// costs a re-scan, never a lost result. (Batch `--exit-if-empty` runs drain
/// unbounded instead — a finite dataset must complete, not re-lease.)
const SHUTDOWN_DRAIN_SECS: u64 = 15;

/// How long hopper must have nothing for this worker before the dry spell is
/// reported at WARN. A worker pointed at a healthy hopper should never sit this
/// long without a claim, so crossing it means something upstream is wrong — an
/// empty queue, a routing filter no sample matches, or a worker whose advertised
/// tools/`max_bytes` exclude it from everything queued.
///
/// Deliberately well above the 2 s poll cadence: brief gaps between batches are
/// normal and must not warn. Tunable via `SCAN_IDLE_WARN_SECS` (min 1).
const DEFAULT_IDLE_WARN_SECS: u64 = 120;

/// Re-warn cadence once a dry spell is already being reported, so a multi-hour
/// outage stays visible in the log without filling it at the poll rate.
const IDLE_WARN_REPEAT: Duration = Duration::from_secs(15 * 60);

/// Whether a dry spell of `dry` should be (re-)reported now.
///
/// Split out from the poll loop so the escalation policy — warn once on
/// crossing the threshold, then at [`IDLE_WARN_REPEAT`] — is testable without
/// driving a real prefetcher against a real hopper.
fn idle_warn_due(dry: Duration, since_last_warn: Option<Duration>, warn_after: Duration) -> bool {
    if dry < warn_after {
        return false;
    }
    match since_last_warn {
        // First crossing of the threshold for this dry spell.
        None => true,
        Some(since) => since >= IDLE_WARN_REPEAT,
    }
}

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

/// Park until `shutdown` is raised, polling at the same 500 ms granularity as
/// [`interruptible_sleep`]. A worker parks here for its whole life, which is
/// past what `interruptible_sleep`'s deadline arithmetic can express.
async fn wait_for_shutdown(shutdown: &AtomicBool) {
    while !shutdown.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(500)).await;
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
#[cfg(unix)]
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

/// Non-unix stub: `setpriority` is POSIX and has no direct Windows analogue —
/// the nearest equivalent, `SetPriorityClass`, works on coarse priority classes
/// rather than a nice value, so mapping one onto the other would be a guess.
/// Same posture as the unix path on failure: log and run at inherited priority
/// rather than refuse to start.
#[cfg(not(unix))]
fn apply_nice(nice: i32) {
    if nice != 0 {
        tracing::warn!(
            nice,
            "nice values are unsupported on this platform; continuing at inherited priority"
        );
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
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, Instant::now());
        id
    }

    /// Mark a queue item finished: drop it, stamp the completion, tick the rate.
    fn complete(&self, id: u64) {
        self.enqueued
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&id);
        *self
            .last_completion
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(Instant::now());
        let minute = self.minute();
        self.throughput
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .record(minute);
    }

    fn record_error(&self, msg: &str) {
        self.errors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .record(msg, Instant::now());
    }

    fn snapshot(&self) -> MetricsSnapshot {
        let oldest_age = self
            .enqueued
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .min()
            .map(Instant::elapsed);
        let last_completion_age = self
            .last_completion
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .map(|t| t.elapsed());
        let files_per_sec = self
            .throughput
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .per_sec(self.minute());
        let (errors_recent, last_error) = {
            let mut errors = self.errors.lock().unwrap_or_else(PoisonError::into_inner);
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
    // nice(2) is process-wide: renicing here would slow the server's own
    // request handling, not just this worker.
    if config.embedded.is_none() {
        apply_nice(config.nice);
    }
    // Arc<str> so every per-job dispatch clones an atomic refcount rather than
    // reallocating the worker name for each `tokio::spawn`.
    let name: Arc<str> = Arc::from(config.name.as_str());
    let slots = config.workers.get();
    // 120 s per request is long enough for cold cleave scans yet short enough
    // that a wedged hopper can't pin the worker indefinitely — without a
    // timeout the default is "no timeout", which defeats graceful shutdown.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    // Worker tasks (`slots`) claim and stage jobs; the cleave gate separately
    // bounds simultaneous memory-bandwidth-heavy analyses.
    let cleave_slots = cleave_concurrency(slots);
    let cleave_gate = Arc::new(Semaphore::new(cleave_slots));
    // Slot count bounds concurrency but not memory: a slot analysing a huge
    // archive holds it (plus expanded members) resident while a slot analysing a
    // 4 KB script holds nothing. This gate pauses admission on live memory
    // pressure — at the resolved `--max-rss-gb` ceiling (default 85% of RAM) —
    // so a burst of large archives serialises instead of co-residing and
    // exhausting memory. It pauses and reclaims; it never kills the worker.
    let admission = crate::admission::MemoryAdmission::new(
        config.max_rss_gb.saturating_mul(1024 * 1024 * 1024),
    );
    // Embedded: share the host's shutdown flag and leave its signal handler
    // alone. A second handler on the same signals would race the first.
    let shutdown = match &config.embedded {
        Some(embedded) => Arc::clone(&embedded.shutdown),
        None => {
            let flag = Arc::new(AtomicBool::new(false));
            install_shutdown_handler(Arc::clone(&flag));
            flag
        }
    };

    // Phase 2 thread model: `slots` long-lived worker tasks pull jobs; the one
    // process-global rayon pool provides member-level parallelism. Total rayon
    // threads = the pool size, independent of slots — no per-slot grid. This
    // is the headline number that caps cleave's per-thread YARA scanners.
    let global_rayon_threads = rayon::current_num_threads();
    tracing::info!(
        slots,
        cleave_slots,
        rayon_threads = global_rayon_threads,
        "worker concurrency: {slots} pull-style workers share one \
         {global_rayon_threads}-thread rayon pool (cleave gate={cleave_slots})",
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
    // Every hopper call carries this token, so a worker without one claims
    // nothing. Report the source (or its absence) before the first poll.
    log_hopper_credential();
    if cleave_slots < slots {
        tracing::warn!(
            slots,
            cleave_slots,
            "cleave entry gate allows {cleave_slots} simultaneous analyses; \
             other claimed worker slots deliberately wait at this gate. Set \
             SCAN_CLEAVE_CONCURRENCY to override",
        );
    }

    // Start background rayon pool health monitoring.
    cleave::start_rayon_diagnostics();

    // Warm YARA + capability mapper on a non-rayon thread before any job is
    // dispatched. The variant (`true`) must match `AnalysisOptions::default()`
    // — otherwise the prefetch warms an engine nobody uses and the first real
    // analysis triggers a cold compile on a rayon worker, which deadlocks the
    // pool. See cleave::shared_resources::yara_engine for the contract.
    cleave::prefetch_shared_resources(true);

    let resource_config = Arc::new(ResourceConfig {
        model_dir: config.model_dir.clone(),
        thresholds: config.thresholds,
        slow_rule_ms: config.slow_rule_ms,
        level: config.level,
        interpret: config.interpret.clone(),
        fetch: config.fetch,
        zip_passwords: config.zip_passwords.clone(),
    });
    let resources: ResourceHandle = match &config.embedded {
        // Share the server's models rather than loading a second copy, and
        // leave renewal to the host: its /_/reload owns the handle, and two
        // renewal tasks would reload the same directory on different clocks.
        Some(embedded) => Arc::new(RwLock::new(Arc::clone(&embedded.resources))),
        None => {
            let handle: ResourceHandle =
                Arc::new(RwLock::new(load_model_resources(&resource_config)?));
            spawn_resource_renewal_task(
                Arc::clone(&handle),
                resource_config,
                Arc::clone(&shutdown),
            );
            handle
        }
    };

    // Arc<str> for the hopper URL — cloned per prefetched job and per dispatched
    // analysis; an atomic bump is far cheaper than a String reallocation.
    let base_url: Arc<str> = Arc::from(config.hopper_url.trim_end_matches('/'));
    let data_dir = config.data_dir.clone();
    // Built off the startup path, on its own thread. The walk scales with the
    // corpus while hopper's liveness watchdog starts its clock the moment this
    // process spawns — blocking here is precisely how a worker gets killed for
    // being "wedged" before it has ever polled for work. Until the index
    // lands, jobs resolve as if no data dir were configured: the payload is
    // fetched from hopper, which is correct, merely slower than reading it off
    // local disk. `OnceLock` gives the dispatch path a lock-free read of a
    // value that is published exactly once.
    let local_index: Arc<OnceLock<LocalFileIndex>> = Arc::new(OnceLock::new());
    if let Some(root) = data_dir.clone() {
        let slot = Arc::clone(&local_index);
        std::thread::Builder::new()
            .name("sample-index".to_string())
            .spawn(move || match LocalFileIndex::build(root) {
                Ok(index) => {
                    if slot.set(index).is_err() {
                        tracing::error!("local sample index published twice");
                    }
                }
                Err(e) => tracing::error!(
                    error = %e,
                    "building local sample index failed; jobs will fetch payloads from hopper",
                ),
            })
            .context("spawning local sample index builder")?;
    }
    // Shared with every dispatched job so local resolution works from the
    // first poll, whether or not the index has landed yet.
    let data_root: Option<Arc<Path>> = data_dir.clone().map(Arc::<Path>::from);
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

    let big_worker = cleave::memory_tracker::total_memory().unwrap_or(0) >= 16 * 1024 * 1024 * 1024;
    let max_buffer_bytes: usize = if big_worker {
        1024 * 1024 * 1024 // 1 GiB on systems with >= 16 GiB RAM
    } else {
        512 * 1024 * 1024 // 512 MiB otherwise
    };
    // Largest file this worker will accept, advertised to hopper on /api/next so
    // it never routes files no worker can analyze. Every worker takes up to
    // MAX_JOB_BYTES regardless of RAM: payloads above `max_single_bytes` (half
    // the RAM buffer) stream to the disk spool and are analyzed via the
    // file-path route, where cleave memory-maps the sample and extracts archives
    // to disk, and the memory-admission gate serialises anything whose estimate
    // exceeds the ceiling. Hopper's filterCandidatesBySize honors this;
    // `prefetch_one` enforces the same ceiling locally as a backstop for older
    // hoppers.
    let advertised_max_bytes: usize = usize::try_from(MAX_JOB_BYTES).unwrap_or(usize::MAX);

    // Background prefetch keeps `1.1 × slots` samples staged at all times so a
    // free worker never waits on a download. The prefetcher polls and downloads
    // on its own task, pushing each sample into the job channel the instant its
    // download finishes; the N workers pull ready samples. `outstanding`
    // (staged + in-flight) bounds the depth; `queued_bytes` bounds staged
    // payload memory against `max_buffer_bytes`.
    let (tx, rx) = mpsc::unbounded_channel::<PrefetchedJob>();
    let queued_bytes = Arc::new(AtomicUsize::new(0));
    let outstanding = Arc::new(AtomicUsize::new(0));
    // Workers currently inside admit/analyze (not posting). Heartbeat/summary
    // use this instead of a slot semaphore.
    let analyzing = Arc::new(AtomicUsize::new(0));
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
    let jobs = Arc::new(JobSource::new(rx, sjf));
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
    // Embedded: hold exactly one claim, never stage ahead. A prefetched job is
    // a claim hopper believes is being worked on, and an idle worker that keeps
    // yielding to requests could sit on a staged job for a long time — hopper
    // would wait out the lease before redispatching it to a worker that could
    // have started immediately. Claiming only what it is about to analyze keeps
    // the queue honest.
    let target_depth = if config.embedded.is_some() {
        1
    } else {
        ((slots as f64 * depth_factor).ceil() as usize).max(1)
    };
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

    // Disk spool for payloads too large for the RAM buffer. Prepared once here
    // (creates the dir, sweeps stale files from crashed runs) and shared by the
    // prefetcher and the direct-download fallback in `run_job`. Prepared before
    // the prefetcher starts so the first spooled download never races the
    // directory creation; the one-time sweep is cheap enough to block startup.
    let spool = Arc::new(SpoolState::new(max_buffer_bytes / 2));
    spool.prepare();
    tracing::info!(
        spool_dir = %spool.dir.display(),
        spool_budget_gb = spool.budget_bytes / (1024 * 1024 * 1024),
        mem_threshold_mb = spool.mem_threshold_bytes / (1024 * 1024),
        max_job_gb = MAX_JOB_BYTES / (1024 * 1024 * 1024),
        "large payloads spool to disk (SCAN_SPOOL_DIR / SCAN_SPOOL_BUDGET_GB)",
    );

    let prefetch_task = tokio::spawn(
        Prefetcher {
            pause: config.embedded.as_ref().map(|e| Arc::clone(&e.pause)),
            last_analyze_request_ms: config
                .embedded
                .as_ref()
                .map(|e| Arc::clone(&e.last_analyze_request_ms)),
            activity_started_at: config.embedded.as_ref().map(|e| e.started_at),
            quiet_period: config.embedded.as_ref().map(|e| e.quiet_period),
            client: client.clone(),
            base_url: Arc::clone(&base_url),
            data_dir: data_dir.clone(),
            encoded_name,
            available_tools,
            slots,
            spool: Arc::clone(&spool),
            max_buffer_bytes,
            advertised_max_bytes,
            poll_secs,
            target_depth,
            metrics: Arc::clone(&metrics),
            poll_state: Arc::clone(&poll_state),
            exit_if_empty,
            idle_warn_after: Duration::from_secs(
                std::env::var("SCAN_IDLE_WARN_SECS")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())
                    .map_or(DEFAULT_IDLE_WARN_SECS, |s| s.max(1)),
            ),
        }
        .run(
            tx,
            Arc::clone(&queued_bytes),
            Arc::clone(&outstanding),
            Arc::clone(&shutdown),
        ),
    );

    // Workers park in `await`s, so emit the periodic summary from a dedicated
    // ticker reading the shared counters.
    {
        let analyzing = Arc::clone(&analyzing);
        let completed = Arc::clone(&completed);
        let outstanding = Arc::clone(&outstanding);
        let queued_bytes = Arc::clone(&queued_bytes);
        let shutdown = Arc::clone(&shutdown);
        // Poll telemetry so the summary can say *why* the worker is idle. It
        // already reaches hopper on the heartbeat; an operator reading worker
        // logs had no equivalent and could not tell "hopper has no work" from
        // "the poll loop is wedged" — both look like zero active slots.
        let poll_state = Arc::clone(&poll_state);
        let metrics_for_summary = Arc::clone(&metrics);
        // Default 60 s; `SCAN_HEARTBEAT_SECS` lowers it (min 1 s) so a short
        // benchmark run still emits a usable rss / active-slot time series.
        let heartbeat = Duration::from_secs(
            std::env::var("SCAN_HEARTBEAT_SECS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map_or(60, |s| s.max(1)),
        );
        // Optional short-cadence snapshots of cleave's per-Rayon-thread
        // breadcrumbs. This is deliberately separate from wedge detection:
        // stack overflow aborts synchronously and may happen before any wedge
        // threshold is reached. A recent snapshot is therefore the useful
        // evidence when the process dies without a warning.
        let breadcrumb_interval = std::env::var("SCAN_BREADCRUMB_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&s| s > 0)
            .map(Duration::from_secs);
        tokio::spawn(async move {
            tracing::debug!(
                heartbeat_secs = heartbeat.as_secs(),
                breadcrumb_secs = breadcrumb_interval.map(|d| d.as_secs()),
                "worker summary ticker armed"
            );
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
            // `idle_trimmed`: whether the allocator has already been asked to
            // hand back retained pages during the *current* idle stretch. Latched
            // so a worker parked on an empty hopper trims once, not once a
            // minute forever; cleared as soon as a slot picks work back up.
            let mut idle_trimmed = false;
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
            let tick_interval =
                breadcrumb_interval.map_or(wedge_check, |interval| wedge_check.min(interval));
            let mut last_summary = Instant::now();
            #[cfg(feature = "cleave-breadcrumbs")]
            let mut last_breadcrumb = Instant::now();
            while !shutdown.load(Ordering::Relaxed) {
                interruptible_sleep(tick_interval, &shutdown).await;
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                // A panic anywhere in this tick (census formatting, wait-channel
                // resolution) would otherwise kill this task silently and end
                // all summary/wedge telemetry for the rest of the worker's
                // life — observed once in production as the ticker going quiet
                // ~45 minutes before exit with the worker still running.
                // Contain it: log the panic, keep ticking.
                let tick = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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
                    #[cfg(feature = "cleave-breadcrumbs")]
                    let breadcrumb_due = breadcrumb_interval
                        .is_some_and(|interval| now.duration_since(last_breadcrumb) >= interval);

                    #[cfg(feature = "cleave-breadcrumbs")]
                    if breadcrumb_due {
                        for crumb in cleave::breadcrumb::snapshot()
                            .into_iter()
                            .take(CENSUS_MAX_LINES)
                        {
                            tracing::info!(
                                rayon_index = ?crumb.rayon_index,
                                thread_id = crumb.thread_id,
                                analyzer = crumb.analyzer,
                                target = %crumb.target,
                                age_ms = crate::duration_ms(crumb.age),
                                "RAYON breadcrumb snapshot",
                            );
                        }
                        last_breadcrumb = now;
                    }

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
                        // Nothing else this tick (the closure is one tick's body).
                        return;
                    }
                    last_summary = now;

                    let started = BLOCKING_STARTED_TOTAL.load(Ordering::Relaxed);
                    let finished = BLOCKING_FINISHED_TOTAL.load(Ordering::Relaxed);
                    let active_slots = analyzing.load(Ordering::Relaxed);
                    let available_slots = slots.saturating_sub(active_slots);
                    let heap = crate::heap_profile::stats();
                    let (regex_scratch_bytes, regex_scratch_budget_bytes) =
                        cleave::regex_scratch_usage();
                    tracing::info!(
                        rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / 1024 / 1024),
                        jemalloc_allocated_mb = heap.map(|stats| stats.allocated / (1024 * 1024)),
                        jemalloc_active_mb = heap.map(|stats| stats.active / (1024 * 1024)),
                        jemalloc_resident_mb = heap.map(|stats| stats.resident / (1024 * 1024)),
                        jemalloc_retained_mb = heap.map(|stats| stats.retained / (1024 * 1024)),
                        regex_scratch_mb = regex_scratch_bytes / (1024 * 1024),
                        regex_scratch_budget_mb = regex_scratch_budget_bytes / (1024 * 1024),
                        queued_prefetch_jobs = outstanding.load(Ordering::Relaxed),
                        prefetch_buffer_mb = queued_bytes.load(Ordering::Relaxed) / (1024 * 1024),
                        active_slots,
                        available_slots,
                        cpu_cores_busy = format!("{cpu_cores_busy:.1}"),
                        load1 = system_load_avg().map(|load| format!("{load:.1}")),
                        rayon_threads = global_rayon_threads,
                        blocking_started_total = started,
                        blocking_finished_total = finished,
                        inflight_blocking = started.saturating_sub(finished),
                        completed = completed.load(Ordering::Acquire),
                        corpus_checks = crate::corpus_precheck::counters().0,
                        corpus_skips = crate::corpus_precheck::counters().1,
                        // Why this worker is (or is not) claiming. `poll_age_s`
                        // far above the poll cadence means the loop is wedged;
                        // `last_claim=0` with a fresh `poll_age_s` and non-zero
                        // `buffer_room` means hopper simply has no work — an
                        // idle worker, not a stuck one.
                        poll_age_s = metrics_for_summary
                            .start
                            .elapsed()
                            .as_secs()
                            .saturating_sub(poll_state.last_poll_secs.load(Ordering::Acquire)),
                        last_want = poll_state.last_want.load(Ordering::Acquire),
                        last_claim = poll_state.last_claim.load(Ordering::Acquire),
                        buffer_room = poll_state.buffer_room.load(Ordering::Acquire),
                        "worker summary",
                    );

                    // Idle with memory still held: hand the allocator's retained
                    // pages back to the OS. This matters because the admission
                    // gate rations intake on *live process memory*, so pages the
                    // allocator is only holding throttle the next batch as
                    // effectively as pages in use. No-op on unix, where
                    // jemalloc's background thread already does it.
                    if active_slots == 0 {
                        if !idle_trimmed {
                            idle_trimmed = true;
                            tokio::task::spawn_blocking(|| {
                                let before = cleave::memory_tracker::current_rss();
                                cleave::clear_all_thread_caches();
                                crate::allocator::trim();
                                let after = cleave::memory_tracker::current_rss();
                                if let (Some(before), Some(after)) = (before, after) {
                                    tracing::info!(
                                        rss_before_mb = before / 1024 / 1024,
                                        rss_after_mb = after / 1024 / 1024,
                                        reclaimed_mb =
                                            before.saturating_sub(after) / 1024 / 1024,
                                        "idle: returned retained allocator pages to the OS",
                                    );
                                }
                            });
                        }
                    } else {
                        idle_trimmed = false;
                    }

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
                }));
                if let Err(panic) = tick {
                    let msg = panic
                        .downcast_ref::<&str>()
                        .map(|s| (*s).to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "non-string panic payload".to_string());
                    tracing::error!(panic = %msg, "worker summary tick panicked; ticker continues");
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
        let analyzing = Arc::clone(&analyzing);
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
                let active = analyzing.load(Ordering::Relaxed);
                let report = HeartbeatReport {
                    slots,
                    active,
                    queue: outstanding.load(Ordering::Acquire),
                    mem_reserved_mb: admission.reserved_bytes() / MIB,
                    mem_ceiling_mb: admission.ceiling_bytes() / MIB,
                    poll_age_s: metrics
                        .start
                        .elapsed()
                        .as_secs()
                        .saturating_sub(poll_state.last_poll_secs.load(Ordering::Acquire)),
                    last_want: poll_state.last_want.load(Ordering::Acquire),
                    last_claim: poll_state.last_claim.load(Ordering::Acquire),
                    buffer_room: poll_state.buffer_room.load(Ordering::Acquire),
                    active_shas: admission.in_flight_shas(),
                    metrics: metrics.snapshot(),
                };
                let url = heartbeat_url(&base_url, &encoded_name, &available_tools, &report);
                match authed(client.get(&url)).send().await {
                    Ok(resp) if resp.status().is_success() => {}
                    Ok(resp) => {
                        tracing::debug!(status = %resp.status(), "heartbeat: non-success response");
                    }
                    Err(e) => tracing::debug!(error = %e, "heartbeat request failed"),
                }
            }
        });
    }

    // N long-lived workers pull from the shared job source. No central
    // dispatcher, no analysis-slot semaphore — these tasks *are* the slots.
    let mut workers: JoinSet<()> = JoinSet::new();
    for worker_id in 0..slots {
        let client = client.clone();
        let base_url = Arc::clone(&base_url);
        let name = Arc::clone(&name);
        let local_index = Arc::clone(&local_index);
        let data_root = data_root.clone();
        let resources = Arc::clone(&resources);
        let jobs = Arc::clone(&jobs);
        let queued_bytes = Arc::clone(&queued_bytes);
        let outstanding = Arc::clone(&outstanding);
        let analyzing = Arc::clone(&analyzing);
        let completed = Arc::clone(&completed);
        let metrics = Arc::clone(&metrics);
        let spool = Arc::clone(&spool);
        let admission = Arc::clone(&admission);
        let cleave_gate = Arc::clone(&cleave_gate);
        let shutdown = Arc::clone(&shutdown);
        workers.spawn(async move {
            loop {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                if let Some(max) = max_jobs
                    && completed.load(Ordering::Acquire) >= max
                {
                    shutdown.store(true, Ordering::Relaxed);
                    break;
                }

                let Some(pj) = jobs.recv().await else {
                    // Prefetcher exited (channel closed). Do not raise shutdown:
                    // sibling workers and the summary ticker keep running until
                    // every in-flight analyze+post finishes.
                    break;
                };
                let staged_bytes = pj.data.as_ref().map_or(0, PrefetchData::staged_mem_bytes);
                queued_bytes.fetch_sub(staged_bytes, Ordering::Release);
                outstanding.fetch_sub(1, Ordering::Release);

                let snapshot: std::result::Result<Arc<ModelResources>, String> =
                    match resources.read() {
                        Ok(guard) => Ok(Arc::clone(&*guard)),
                        Err(error) => {
                            let error = anyhow::anyhow!("worker resources lock poisoned: {error}");
                            tracing::error!(
                                worker_id,
                                error = %error,
                                "cannot snapshot worker resources; failing job"
                            );
                            Err(format!("worker resource snapshot failed: {error}"))
                        }
                    };
                let snapshot = match snapshot {
                    Ok(snapshot) => snapshot,
                    Err(failure) => {
                        metrics.record_error(&failure);
                        post_result(&client, &base_url, &name, &pj.job.sha256, Err(failure)).await;
                        metrics.complete(pj.queue_id);
                        completed.fetch_add(1, Ordering::Release);
                        continue;
                    }
                };

                analyzing.fetch_add(1, Ordering::Release);
                // RAII-ish: always clear the analyzing count if we bail early.
                struct AnalyzingGuard(Arc<AtomicUsize>);
                impl Drop for AnalyzingGuard {
                    fn drop(&mut self) {
                        self.0.fetch_sub(1, Ordering::Release);
                    }
                }
                let _analyzing_guard = AnalyzingGuard(Arc::clone(&analyzing));

                let admission_guard = admission
                    .admit(
                        Arc::from(pj.job.sha256.as_str()),
                        Arc::from(pj.job.path.as_str()),
                        Arc::from(pj.job.file_type.as_str()),
                        pj.job.size_bytes,
                    )
                    .await;

                let result = run_job(
                    &client,
                    &base_url,
                    local_index.get(),
                    data_root.as_deref(),
                    &pj.job,
                    &snapshot,
                    Arc::clone(&cleave_gate),
                    slow_rule_ms,
                    &spool,
                    pj.data,
                )
                .await;
                drop(admission_guard);
                drop(_analyzing_guard);

                if let Err(ref e) = result {
                    tracing::warn!(
                        worker_id,
                        sha256 = %pj.job.sha256,
                        file = %pj.job.path,
                        file_type = %pj.job.file_type,
                        size = pj.job.size_bytes,
                        error = %e,
                        "analysis failed",
                    );
                    metrics.record_error(&e.to_string());
                }
                // Hopper I/O (including dep mirroring) runs with this worker
                // busy on post only — siblings keep analyzing.
                post_result(&client, &base_url, &name, &pj.job.sha256, result).await;
                metrics.complete(pj.queue_id);
                let n = completed.fetch_add(1, Ordering::Release) + 1;
                if n.is_multiple_of(100) {
                    // Clearing cleave's caches returns memory to the allocator;
                    // the trim is what returns it to the OS, which is the half
                    // the admission gate can actually see.
                    tokio::task::spawn_blocking(|| {
                        cleave::clear_all_thread_caches();
                        crate::allocator::trim();
                    });
                }
            }
        });
    }

    // A worker has no reason to stop on its own: it polls, analyses, posts, and
    // repeats. Park here until something actually asks it to stop, so the drain
    // below measures a shutdown deadline and not uptime — without this the cap
    // fires 15 s after startup and abandons eight healthy in-flight analyses.
    tokio::select! {
        // SIGTERM/SIGINT, or `--max-jobs` satisfied by the workers themselves.
        () = wait_for_shutdown(&shutdown) => {}
        // The prefetcher is the only source of work. It returns on shutdown and,
        // in `--exit-if-empty` mode, when the hopper runs dry; any other return
        // is a panic, which starves every slot forever. Say so and exit rather
        // than idle behind a heartbeat that still looks healthy — dropping `tx`
        // closes the dispatch channel, so the workers finish what is staged and
        // then stop on their own.
        res = prefetch_task => {
            if !exit_if_empty && !shutdown.load(Ordering::Relaxed) {
                match res {
                    Ok(()) => tracing::error!(
                        "prefetcher exited unexpectedly; no further jobs will be claimed"
                    ),
                    Err(e) => tracing::error!(
                        error = %e,
                        "prefetcher task died; no further jobs will be claimed"
                    ),
                }
            }
        }
    }

    // Wait for every worker to finish its current analyze+post. On SIGTERM the
    // wait is capped so a stuck cleave or wedged hopper cannot block shutdown —
    // hopper re-leases anything left running. `--exit-if-empty` waits unbounded.
    if exit_if_empty {
        while workers.join_next().await.is_some() {}
        tracing::info!("all in-flight jobs finished (batch drain), exiting");
    } else {
        let drain = async { while workers.join_next().await.is_some() {} };
        match tokio::time::timeout(Duration::from_secs(SHUTDOWN_DRAIN_SECS), drain).await {
            Ok(()) => tracing::info!("all in-flight jobs finished, exiting"),
            Err(_) => {
                let still_running = workers.len();
                tracing::warn!(
                    still_running,
                    drain_secs = SHUTDOWN_DRAIN_SECS,
                    "drain timeout reached, exiting with in-flight workers still running",
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
    /// Disk spool: payloads above its memory threshold stream to disk instead
    /// of the RAM buffer; jobs above [`MAX_JOB_BYTES`] are rejected outright.
    spool: Arc<SpoolState>,
    /// Soft cap on total staged payload bytes held in RAM.
    max_buffer_bytes: usize,
    /// Largest file this worker accepts ([`MAX_JOB_BYTES`]), sent to hopper as
    /// `max_bytes` so it routes only files this worker can analyze.
    advertised_max_bytes: usize,
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
    /// How long hopper must have nothing for this worker before the dry spell is
    /// reported at WARN. A field rather than an env read inside the poll loop so
    /// the escalation is exercisable from a test. See [`DEFAULT_IDLE_WARN_SECS`].
    idle_warn_after: Duration,
    /// Raised while the host server has interactive work in flight. The
    /// prefetcher is the single place work enters this worker, so gating here
    /// stops the whole pipeline at one point: staged jobs finish, slots drain,
    /// and nothing new is claimed until the flag clears.
    pause: Option<Arc<AtomicBool>>,
    /// Elapsed-time marker for the most recent host `/analyze` request.
    last_analyze_request_ms: Option<Arc<AtomicU64>>,
    /// Host clock anchor for interpreting `last_analyze_request_ms`.
    activity_started_at: Option<Instant>,
    /// Quiet period after host analysis traffic.
    quiet_period: Option<Duration>,
}

impl Prefetcher {
    fn recently_saw_analyze_request(&self) -> bool {
        let (Some(last), Some(started_at), Some(quiet_period)) = (
            self.last_analyze_request_ms.as_ref(),
            self.activity_started_at,
            self.quiet_period,
        ) else {
            return false;
        };
        let last_ms = last.load(Ordering::Acquire);
        if last_ms == 0 {
            return false;
        }
        let now_ms = u64::try_from(
            started_at
                .elapsed()
                .as_millis()
                .min(u128::from(u64::MAX.saturating_sub(1))),
        )
        .unwrap_or(u64::MAX.saturating_sub(1))
        .saturating_add(1);
        u128::from(now_ms.saturating_sub(last_ms)) < quiet_period.as_millis()
    }

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
        if self.advertised_max_bytes > 0 {
            let _ = write!(url, "&max_bytes={}", self.advertised_max_bytes);
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
        let mut paused_logged = false;
        // Dry-spell tracking. `last_productive` is the last moment this worker
        // had a reason to believe hopper had work for it — a successful claim,
        // or a deliberate decision not to ask (paused, or buffer full). Measuring
        // from there rather than from the last empty poll means a hopper that
        // trickles one job an hour still reads as starved, which it is.
        let mut last_productive = Instant::now();
        let mut dry_warned_at: Option<Instant> = None;
        loop {
            if shutdown.load(Ordering::Relaxed) {
                return;
            }

            // Interactive work has priority. Stop claiming — but do not abandon
            // what is already staged or running: that work is real, and hopper
            // redispatches a claim that dies anyway. Responsiveness comes from
            // the slots the server keeps for itself, not from killing jobs.
            let interactive = self
                .pause
                .as_ref()
                .is_some_and(|p| p.load(Ordering::Relaxed));
            let recently_active = self.recently_saw_analyze_request();
            if interactive || recently_active {
                if !paused_logged {
                    tracing::debug!(
                        interactive,
                        recently_active,
                        "idle worker paused: recent interactive analysis activity"
                    );
                    paused_logged = true;
                }
                self.poll_state.buffer_room.store(0, Ordering::Release);
                // Yielding to interactive work is not hopper being dry.
                last_productive = Instant::now();
                dry_warned_at = None;
                interruptible_sleep(Duration::from_millis(200), &shutdown).await;
                continue;
            }
            if paused_logged {
                tracing::debug!("idle worker resumed");
                paused_logged = false;
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
                // Full buffer: this worker is saturated, not starved.
                last_productive = Instant::now();
                dry_warned_at = None;
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
                    // Hopper answered, and had nothing. Rare on a healthy
                    // deployment, so say so loudly once the gap stops looking
                    // like the pause between batches. `--exit-if-empty` runs
                    // (batch/benchmark) drain to empty on purpose and are exempt.
                    let dry = last_productive.elapsed();
                    if !self.exit_if_empty
                        && idle_warn_due(
                            dry,
                            dry_warned_at.map(|at| at.elapsed()),
                            self.idle_warn_after,
                        )
                    {
                        dry_warned_at = Some(Instant::now());
                        tracing::warn!(
                            dry_s = dry.as_secs(),
                            hopper = %self.base_url,
                            worker = %self.encoded_name,
                            slots = self.slots,
                            wanted = count,
                            max_bytes = self.advertised_max_bytes,
                            tools = %self.available_tools,
                            traits = cleave::traits_repo::version()
                                .map(|t| t.chars().take(5).collect::<String>()),
                            "hopper has had no work for this worker — every analysis slot is                              idle. Check hopper's queue depth; if it is non-empty this worker                              is being filtered out of it, so compare the tools, max_bytes and                              traits above against what the queued samples require.",
                        );
                    }
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
                    // Close out a reported dry spell so the log shows the outage
                    // ending, not just beginning.
                    if dry_warned_at.take().is_some() {
                        tracing::info!(
                            dry_s = last_productive.elapsed().as_secs(),
                            claimed = jobs.len(),
                            "hopper has work again; resuming",
                        );
                    }
                    last_productive = Instant::now();
                    outstanding.fetch_add(jobs.len(), Ordering::Release);
                    let mut set = tokio::task::JoinSet::new();
                    for job in jobs {
                        let client = self.client.clone();
                        let base_url = Arc::clone(&self.base_url);
                        let data_dir = self.data_dir.clone();
                        let spool = Arc::clone(&self.spool);
                        set.spawn(async move {
                            prefetch_one(client, base_url, data_dir, spool, job).await
                        });
                    }
                    while let Some(res) = set.join_next().await {
                        match res {
                            Ok(mut pj) => {
                                let bytes =
                                    pj.data.as_ref().map_or(0, PrefetchData::staged_mem_bytes);
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
    let resp = authed(client.get(poll_url)).send().await.map_err(|e| {
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
/// Local files are used in place regardless of size, jobs above
/// [`MAX_JOB_BYTES`] are skipped without a download, payloads too large for the
/// RAM buffer stream to the disk spool, and transient download failures fall
/// through to `run_job`'s direct-download retry.
async fn prefetch_one(
    client: reqwest::Client,
    base_url: Arc<str>,
    data_dir: Option<PathBuf>,
    spool: Arc<SpoolState>,
    job: ClaimJob,
) -> PrefetchedJob {
    // `job.sha256` names the spool file (see `download_to_spool`), and
    // `tempfile`'s `prefix` is concatenated into the filename verbatim — it does
    // not reject path separators. A job whose digest is not 64 hex characters is
    // malformed no matter what, so refuse it here rather than let an arbitrary
    // string become a path component. Permanent: a bad digest never becomes good.
    if sha256_from_hex(&job.sha256).is_none() {
        tracing::warn!(
            sha256 = %job.sha256,
            path = %job.path,
            "refusing job: sha256 is not 64 hex characters",
        );
        let err = PrefetchError::Skipped(format!(
            "malformed sha256: expected 64 hex characters, got {:?}",
            job.sha256
        ));
        return PrefetchedJob {
            job,
            data: Err(err),
            queue_id: 0,
        };
    }

    // Local files need no download or staging, so no size check applies.
    let local_path = data_dir.as_deref().map(|d| d.join(&job.path));
    if matches!(local_path, Some(ref p) if p.exists()) {
        return PrefetchedJob {
            job,
            data: Ok(PrefetchData::Local),
            queue_id: 0,
        };
    }

    let size = u64::try_from(job.size_bytes).unwrap_or(0);
    if size > MAX_JOB_BYTES {
        tracing::warn!(
            sha256 = %job.sha256,
            path = %job.path,
            size_bytes = job.size_bytes,
            max_job_bytes = MAX_JOB_BYTES,
            "skipping oversized job; reporting error to hopper",
        );
        // "exceeds per-job" is matched by hopper's classifyResultError and
        // marks the sample skip='oversized' permanently.
        let err = PrefetchError::Skipped(format!(
            "file size {size} exceeds per-job cap of {MAX_JOB_BYTES} bytes",
        ));
        return PrefetchedJob {
            job,
            data: Err(err),
            queue_id: 0,
        };
    }

    let data = fetch_payload(&client, &base_url, &spool, &job)
        .await
        .map_err(PrefetchError::Transient);
    PrefetchedJob {
        job,
        data,
        queue_id: 0,
    }
}

/// Download a job's payload the size-appropriate way: into memory below the
/// spool threshold, streamed to a spool file above it. Shared by the prefetcher
/// and `run_job`'s direct-download fallback so both routes stay RAM-safe.
async fn fetch_payload(
    client: &reqwest::Client,
    base_url: &str,
    spool: &Arc<SpoolState>,
    job: &ClaimJob,
) -> Result<PrefetchData, String> {
    let size = u64::try_from(job.size_bytes).unwrap_or(0);
    if size <= spool.mem_threshold_bytes as u64 {
        return download_bytes(client, base_url, &job.sha256, &job.path)
            .await
            .map(PrefetchData::Memory);
    }
    spool.try_reserve(size).map_err(|reason| {
        format!(
            "cannot spool {size}-byte payload for {}: {reason}",
            job.sha256
        )
    })?;
    match download_to_spool(client, base_url, spool, &job.sha256, &job.path).await {
        Ok(path) => Ok(PrefetchData::Spooled(SpooledPayload {
            path,
            size,
            spool: Arc::clone(spool),
        })),
        Err(e) => {
            spool.release(size);
            Err(e)
        }
    }
}

/// Analyze a single job. Returns (ml, raw, duration_ms) or an error string.
///
/// Resolution order for the sample bytes: the local index when it is available,
/// otherwise `data_root` on its own via [`resolve_on_disk`], and failing both a
/// download from hopper. The index is an optional accelerator — it finds
/// samples whose recorded path has drifted — so its absence costs recall on
/// moved files, never the ability to read a sample that is where it should be.
///
/// The cleave gate is acquired only for the blocking analyze — after async
/// download/provenance — so hopper I/O cannot pin nested-Rayon capacity.
#[allow(clippy::too_many_arguments)]
async fn run_job(
    client: &reqwest::Client,
    base_url: &str,
    local_index: Option<&LocalFileIndex>,
    data_root: Option<&Path>,
    job: &ClaimJob,
    resources: &Arc<ModelResources>,
    cleave_gate: Arc<Semaphore>,
    slow_rule_ms: u64,
    spool: &Arc<SpoolState>,
    prefetched: std::result::Result<PrefetchData, PrefetchError>,
) -> Result<
    (
        crate::engine::ScanResultEnvelope,
        Vec<crate::engine::DepResult>,
        i64,
    ),
    String,
> {
    let analysis_id = NEXT_ANALYSIS_ID.fetch_add(1, Ordering::Relaxed);
    // `Arc<str>` so the watcher and the blocking closure share the basename
    // allocation instead of each cloning a fresh `String`.
    let label: Arc<str> = Path::new(&job.path)
        .file_name()
        .map(|n| Arc::from(n.to_string_lossy().as_ref()))
        .unwrap_or_else(|| Arc::from(job.sha256.as_str()));

    // Try local file first; fall back to downloading bytes from hopper.
    // Exact-path hits are attempted first, then final-dir+basename+size lookup.
    //
    // With no index — not built yet, or none configured — the filesystem alone
    // still resolves every sample that sits where hopper says it does. Gating
    // that on the index would mean downloading payloads we already have on
    // local disk for as long as the background walk takes to finish.
    let local_path = match (local_index, data_root) {
        (Some(index), _) => index
            .resolve(&job.path, &job.sha256, job.size_bytes)
            .map_err(|e| e.to_string())?,
        (None, Some(root)) => {
            let Some(expected) = sha256_from_hex(&job.sha256) else {
                return Err(format!("expected 64-char hex sha256, got {:?}", job.sha256));
            };
            resolve_on_disk(root, &job.path, &expected, job.size_bytes)
        }
        (None, None) => None,
    };
    let use_local = match (data_root, local_path.as_ref()) {
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
        (Some(root), None) => {
            let parent = Path::new(&job.path)
                .parent()
                .and_then(Path::file_name)
                .and_then(|n| n.to_str())
                .unwrap_or("");
            tracing::warn!(
                sha256 = %job.sha256,
                requested_path = %job.path,
                data_root = %root.display(),
                parent_dir = %parent,
                basename = %label,
                file_type = %job.file_type,
                size = job.size_bytes,
                indexed = local_index.is_some(),
                "local file not found under --data; downloading from hopper"
            );
            false
        }
        (None, None) => false,
    };

    // Use the prefetched payload, or fall back to downloading if prefetch
    // failed. The fallback goes through `fetch_payload`, so a payload too big
    // for the RAM buffer re-spools to disk instead of being buffered.
    let payload: Option<PrefetchData> = if use_local {
        None
    } else {
        match prefetched {
            Ok(PrefetchData::Local) => None, // prefetch saw a local file that the index can't resolve
            Ok(data) => {
                tracing::debug!(sha256 = %job.sha256, file = %label, size = job.size_bytes, "using prefetched data");
                Some(data)
            }
            Err(PrefetchError::Skipped(msg)) => {
                // Prefetch layer decided not to download this job (e.g. oversized);
                // fail the analysis immediately rather than retrying the fetch.
                return Err(msg);
            }
            Err(PrefetchError::Transient(e)) => {
                tracing::warn!(sha256 = %job.sha256, file = %label, error = %e, "prefetch failed, downloading directly");
                Some(fetch_payload(client, base_url, spool, job).await?)
            }
        }
    };

    // Registry metadata hopper collected for this sample at fetch time, so the
    // worker reasons over the same registry facts (age, custody, popularity,
    // deprecation) a live `pkg`/`url` scan fetches — without a refetch. Only
    // attempted when hopper flagged the sample as carrying it; best-effort, so a
    // miss never fails the scan. Consumed as stamped at collection time.
    let root_registry: Option<crate::provenance::RegistryProvenance> = if job.has_provenance {
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
    let input_source = match &payload {
        _ if use_local => "local",
        Some(PrefetchData::Memory(_)) => "downloaded",
        Some(PrefetchData::Spooled(_)) => "spooled",
        _ => "local",
    };
    let input_size = match &payload {
        Some(PrefetchData::Memory(bytes)) if !use_local => bytes.len() as u64,
        Some(PrefetchData::Spooled(spooled)) if !use_local => spooled.size,
        _ => u64::try_from(job.size_bytes).unwrap_or(0),
    };

    // Background phase watcher — logs transitions with timing, and emits a
    // heartbeat every 30 s so a stuck phase is visible in logs.
    // Uses a tokio task instead of an OS thread to avoid one thread-per-job overhead.
    // The returned JoinHandle is aborted via RAII guard below so the watcher cannot
    // outlive this function even if the outer task is cancelled.
    let sha_short_for_watcher = Arc::clone(&sha_short);
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
                        sha256 = %sha_short_for_watcher,
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
                        sha256 = %sha_short_for_watcher,
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
                        sha256 = %sha_short_for_watcher,
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
                    sha256 = %sha_short_for_watcher,
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
                        sha256 = %sha_short_for_watcher,
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
                        sha256 = %sha_short_for_watcher,
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

    // Nested-Rayon capacity is needed only for the blocking classify. Acquiring
    // earlier (or on the dispatch loop) pinned the gate across hopper downloads
    // and froze every other slot behind one whale's preamble.
    let gate_wait_start = Instant::now();
    let cleave_permit = Arc::clone(&cleave_gate)
        .acquire_owned()
        .await
        .map_err(|_closed| "cleave analysis gate closed".to_string())?;
    let gate_wait = gate_wait_start.elapsed();
    if gate_wait >= Duration::from_secs(1) {
        tracing::info!(
            sha256 = %job.sha256.get(..12).unwrap_or(&job.sha256),
            file = %job.path,
            wait_ms = crate::duration_ms(gate_wait),
            "analysis admitted to cleave after waiting for nested-work gate",
        );
    }

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
        // Keep the nested-work permit on this blocking thread for the whole
        // classify/fetch/graft. Dropping it from the async frame on cancel
        // would admit another tree while this one still owns Rayon workers.
        let _cleave_permit = cleave_permit;
        // Spooled payloads take the same file-path route as local files, so a
        // multi-GiB sample is memory-mapped rather than held in RAM. The
        // spooled payload is moved into this closure and dropped when it
        // returns, deleting the spool file and releasing its budget.
        // The worker's whole job is posting results (dependencies included)
        // back to hopper, so dependency capture is always on.
        let result = match (payload, local.as_ref()) {
            (Some(PrefetchData::Memory(data)), _) => classify_bytes(
                data,
                &label_for_blocking,
                &resources,
                slow_rule_ms,
                Some(&cancel2),
                Some(&phase),
                root_registry.as_ref(),
                true,
            ),
            (Some(PrefetchData::Spooled(spooled)), _) => classify_file(
                &spooled.path,
                &label_for_blocking,
                &resources,
                slow_rule_ms,
                None,
                Some(&cancel2),
                Some(&phase),
                root_registry.as_ref(),
                true,
            ),
            (_, Some(path)) => classify_file(
                path,
                &label_for_blocking,
                &resources,
                slow_rule_ms,
                None,
                Some(&cancel2),
                Some(&phase),
                root_registry.as_ref(),
                true,
            ),
            (None | Some(PrefetchData::Local), None) => Err(anyhow::anyhow!(
                "no downloaded bytes and no local path for {label_for_blocking}"
            )),
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
    result: Result<
        (
            crate::engine::ScanResultEnvelope,
            Vec<crate::engine::DepResult>,
            i64,
        ),
        String,
    >,
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
        let mut request = authed(client.post(&url))
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

/// Collapse whitespace and cap a body/blob to a single short line for a log
/// field, so a large payload can't bury the rest of the record. Shared with
/// `upload` (provenance sidecars are large registry documents).
pub(crate) fn body_excerpt(body: &str) -> String {
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
    let start = Instant::now();
    let (resp, route) = download_response(client, base_url, sha256, path).await?;
    let url = resp.url().to_string();
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("download body failed: path={path} sha256={sha256} url={url}: {e}"))?;
    verify_download_sha256(&bytes, sha256, path, &url, route)?;
    tracing::info!(
        sha256 = %sha256,
        file = %path,
        bytes = bytes.len(),
        elapsed_ms = crate::duration_ms(start.elapsed()),
        "download complete via {route}",
    );
    Ok(bytes)
}

/// Stream a payload to a new spool file instead of buffering it in RAM, so a
/// multi-GiB sample downloads with a constant memory footprint. Returns the
/// temp path; the file is deleted when the path drops. The caller reserves and
/// releases spool budget.
async fn download_to_spool(
    client: &reqwest::Client,
    base_url: &str,
    spool: &SpoolState,
    sha256: &str,
    path: &str,
) -> Result<tempfile::TempPath, String> {
    use tokio::io::AsyncWriteExt as _;

    // The digest becomes part of the spool filename below. `prefetch_one`
    // already rejects a malformed one, but this is the function that builds the
    // path, so it does not take that on trust — `tempfile` concatenates `prefix`
    // into the name verbatim, without rejecting path separators, so an unchecked
    // string here would be a traversal primitive out of the spool directory.
    // Checked before any I/O: a malformed job is not worth a request.
    if sha256_from_hex(sha256).is_none() {
        return Err(format!(
            "refusing to spool under a malformed sha256: {sha256:?}"
        ));
    }

    let start = Instant::now();
    let (mut resp, route) = download_response(client, base_url, sha256, path).await?;
    let url = resp.url().to_string();

    // Re-create the spool dir if an OS temp sweep removed it since startup;
    // otherwise every large payload fails here for the life of the process.
    spool.ensure_dir()?;
    let temp = tempfile::Builder::new()
        .prefix(sha256.get(..16).unwrap_or(sha256))
        .tempfile_in(&spool.dir)
        .map_err(|e| format!("cannot create spool file in {}: {e}", spool.dir.display()))?;
    let mut file = tokio::fs::File::from_std(
        temp.as_file()
            .try_clone()
            .map_err(|e| format!("cannot clone spool file handle: {e}"))?,
    );

    let mut written: u64 = 0;
    let mut hasher = Sha256::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| format!("download body failed: path={path} sha256={sha256} url={url}: {e}"))?
    {
        written += chunk.len() as u64;
        if written > MAX_JOB_BYTES {
            return Err(format!(
                "download exceeded per-job cap of {MAX_JOB_BYTES} bytes: path={path} sha256={sha256}",
            ));
        }
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("spool write failed: sha256={sha256}: {e}"))?;
        hasher.update(&chunk);
    }
    file.flush()
        .await
        .map_err(|e| format!("spool flush failed: sha256={sha256}: {e}"))?;
    verify_download_digest(hasher.finalize().into(), sha256, path, &url, route)?;
    tracing::info!(
        sha256 = %sha256,
        file = %path,
        bytes = written,
        elapsed_ms = crate::duration_ms(start.elapsed()),
        "download spooled to disk via {route}",
    );
    Ok(temp.into_temp_path())
}

fn verify_download_sha256(
    bytes: &[u8],
    expected_hex: &str,
    path: &str,
    url: &str,
    route: &str,
) -> Result<(), String> {
    verify_download_digest(Sha256::digest(bytes).into(), expected_hex, path, url, route)
}

fn verify_download_digest(
    actual: [u8; 32],
    expected_hex: &str,
    path: &str,
    url: &str,
    route: &str,
) -> Result<(), String> {
    let Some(expected) = sha256_from_hex(expected_hex) else {
        return Err(format!(
            "download has invalid expected sha256: path={path} sha256={expected_hex} url={url}"
        ));
    };
    if actual == expected {
        return Ok(());
    }
    let actual_hex = digest_hex(&actual);
    tracing::error!(
        expected_sha256 = %expected_hex,
        actual_sha256 = %actual_hex,
        file = %path,
        url = %url,
        route,
        "downloaded bytes failed sha256 verification"
    );
    Err(format!(
        "download sha256 mismatch: path={path} expected={expected_hex} actual={actual_hex} url={url}"
    ))
}

fn digest_hex(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

/// Open a download stream for a sample, trying the cheap path-based endpoint
/// first and falling back to the by-hash API. Returns the successful response
/// (headers read, body not yet consumed) and the route label for logs.
async fn download_response(
    client: &reqwest::Client,
    base_url: &str,
    sha256: &str,
    path: &str,
) -> Result<(reqwest::Response, &'static str), String> {
    if path.is_empty() || path == "." {
        return Err(format!(
            "download {sha256}: empty path from hopper, cannot fetch"
        ));
    }

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
    let resp = authed(client.get(&data_url))
        .send()
        .await
        .map_err(|e| format!("download failed: path={path} sha256={sha256} url={data_url}: {e}"))?;

    if resp.status().is_success() {
        return Ok((resp, "/data/"));
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
    let resp = authed(client.get(&api_url)).send().await.map_err(|e| {
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
    Ok((resp, "/api/file/ (fallback)"))
}

/// Fetch the registry-metadata provenance hopper holds for `sha256`, preserving
/// the complete JSON document alongside its normalized registry record.
/// Best-effort by design: an absent record (HTTP 204), an unreachable hopper,
/// or a malformed body all yield `None` — registry provenance enriches a scan
/// but must never fail one, exactly as a live scan fails open when a registry
/// lookup can't be made.
async fn download_provenance(
    client: &reqwest::Client,
    base_url: &str,
    sha256: &str,
) -> Option<crate::provenance::RegistryProvenance> {
    let url = format!("{base_url}/api/provenance/{sha256}");
    let resp = match authed(client.get(&url)).send().await {
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
    // `resp.bytes()` already owns a refcounted buffer; move it into provenance
    // so the complete hopper document survives without another full-size copy.
    let provenance = crate::provenance::RegistryProvenance::from_bytes(body);
    if let Some(provenance) = &provenance {
        let reg = &provenance.record;
        tracing::debug!(
            sha256 = %sha256,
            ecosystem = %reg.ecosystem,
            package = %reg.name,
            version = %reg.version,
            "registry provenance applied",
        );
    }
    provenance
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
    /// sha256 of every in-flight analysis, so hopper renews their claim leases
    /// and a multi-hour scan is not re-claimed mid-flight.
    active_shas: Vec<Arc<str>>,
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
    // In-progress sha256s (hex, comma-separated) so hopper renews their claim
    // leases. Bounded by slot count, so the query stays short.
    if !report.active_shas.is_empty() {
        let joined = report
            .active_shas
            .iter()
            .map(std::convert::AsRef::as_ref)
            .collect::<Vec<&str>>()
            .join(",");
        let _ = write!(url, "&active_shas={joined}");
    }
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

    /// The walk fans out one task per subdirectory and merges per-directory
    /// batches at the end, so `FileId` assignment crosses both directory and
    /// thread boundaries. Cover it with a tree wide and deep enough that the
    /// spawns genuinely interleave.
    #[test]
    fn local_index_finds_every_file_in_a_deep_wide_tree() {
        let root = tempfile::tempdir().expect("create temp dir");
        let mut expected = Vec::new();
        for branch in 0..8 {
            let mut rel = PathBuf::from(format!("branch{branch}"));
            for depth in 0..4 {
                rel = rel.join(format!("level{depth}"));
                let bytes = format!("sample-{branch}-{depth}").into_bytes();
                let file = rel.join(format!("s{branch}{depth}.bin"));
                write_file(&root.path().join(&file), &bytes);
                expected.push((file, bytes));
            }
        }

        let index = LocalFileIndex::build(root.path().to_path_buf()).expect("build index");
        assert_eq!(index.files.len(), expected.len());

        for (rel, bytes) in expected {
            let resolved = index
                .resolve(
                    rel.to_str().expect("utf8 rel path"),
                    &sha256_hex(&bytes),
                    i64::try_from(bytes.len()).expect("len fits"),
                )
                .expect("resolve path");
            assert_eq!(resolved.as_deref(), Some(root.path().join(&rel).as_path()));
        }
    }

    /// The index-free path is what actually serves data — before the index has
    /// been built, and on workers that never build one. Cover both path shapes
    /// and confirm the digest is what gates the match, not the path.
    #[test]
    fn resolve_on_disk_serves_samples_without_an_index() {
        let root = tempfile::tempdir().expect("create temp dir");
        let rel = Path::new("good/repos/sample.bin");
        let bytes = b"no-index-needed";
        let stored = root.path().join(rel);
        write_file(&stored, bytes);

        let expected = sha256_from_hex(&sha256_hex(bytes)).expect("decode sha256");
        let size = i64::try_from(bytes.len()).expect("len fits");

        // Relative path, joined onto the data root.
        assert_eq!(
            resolve_on_disk(
                root.path(),
                rel.to_str().expect("utf8 rel"),
                &expected,
                size
            )
            .as_deref(),
            Some(stored.as_path()),
        );
        // Absolute path, taken as given.
        assert_eq!(
            resolve_on_disk(
                root.path(),
                stored.to_str().expect("utf8 abs"),
                &expected,
                size,
            )
            .as_deref(),
            Some(stored.as_path()),
        );
        // A file whose content doesn't match the digest is never served.
        let wrong = sha256_from_hex(&sha256_hex(b"different bytes")).expect("decode sha256");
        assert!(
            resolve_on_disk(root.path(), rel.to_str().expect("utf8 rel"), &wrong, size).is_none()
        );
    }

    /// A symlinked directory must not be descended into. The serial walk got
    /// this for free by never following links; the parallel one has to keep
    /// that property or a cycle would spawn tasks forever.
    #[cfg(unix)]
    #[test]
    fn local_index_walk_does_not_follow_directory_symlinks() {
        let root = tempfile::tempdir().expect("create temp dir");
        let bytes = b"only-real-file";
        write_file(&root.path().join("real/sample.bin"), bytes);
        // A link back to the root would cycle if the walk followed it.
        std::os::unix::fs::symlink(root.path(), root.path().join("real/loop"))
            .expect("create symlink");

        let index = LocalFileIndex::build(root.path().to_path_buf()).expect("build index");

        assert_eq!(index.files.len(), 1);
        assert_eq!(index.files[0].path, root.path().join("real/sample.bin"));
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

    #[tokio::test]
    async fn hopper_provenance_response_reaches_analysis_losslessly() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock hopper");
        let addr = listener.local_addr().expect("mock hopper address");
        let body = br#"{"schema_version":"1.0","artifact":{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"fetch":{"collector":"forager","category":"bad","at":"2026-07-30T00:00:00Z"},"registry":{"record":{"ecosystem":"npm","name":"left-pad","version":"1.3.0"},"raw":{"provider_only":{"kept":true}}}}"#;
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept provenance request");
            let target = read_target(&mut stream).await.expect("request target");
            assert_eq!(
                target,
                "/api/provenance/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            );
            respond(&mut stream, "200 OK", body).await;
        });

        let provenance = download_provenance(
            &reqwest::Client::new(),
            &format!("http://{addr}"),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        .await
        .expect("worker applies provenance");
        assert_eq!(provenance.record.name, "left-pad");
        assert_eq!(provenance.raw().unwrap()["provider_only"]["kept"], true);
        server.await.expect("mock hopper task");
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

    fn claim_job(sha: &str, path: &str, size_bytes: i64) -> ClaimJob {
        ClaimJob {
            sha256: sha.to_string(),
            path: path.to_string(),
            size_bytes,
            file_type: "data".to_string(),
            has_provenance: false,
        }
    }

    #[tokio::test]
    async fn prefetch_one_rejects_jobs_over_max_job_bytes() {
        // Rejected before any download, so the unreachable base_url is never hit.
        let job = claim_job(&sha256_hex(b"huge"), "samples/huge.bin", (MAX_JOB_BYTES + 1) as i64);
        let pj = prefetch_one(
            reqwest::Client::new(),
            Arc::from("http://127.0.0.1:1"),
            None,
            test_spool(1 << 20),
            job,
        )
        .await;
        match pj.data {
            Err(PrefetchError::Skipped(msg)) => {
                // Hopper's classifyResultError matches this phrase to mark the
                // sample skip='oversized' — keep them in sync.
                assert!(msg.contains("exceeds per-job"), "message was: {msg}");
            }
            _ => panic!("job over MAX_JOB_BYTES must be skipped"),
        }
    }

    /// `job.sha256` becomes the spool filename, and `tempfile` concatenates a
    /// `prefix` into that name without rejecting path separators — so an
    /// unvalidated digest is a write-anywhere primitive. Hopper is authenticated,
    /// but the worker builds the path, so the worker checks it.
    #[tokio::test]
    async fn prefetch_one_refuses_a_sha256_that_is_not_hex() {
        let bad_digests: Vec<String> = vec![
            "../../../../evil".to_string(),
            r"..\..\evil".to_string(),
            "nul".to_string(),
            String::new(),
            // Right length, wrong alphabet.
            "z".repeat(64),
            // Hex but too short: `sha256.get(..16)` would still have yielded a name.
            "abc123".to_string(),
        ];
        for bad in &bad_digests {
            let job = claim_job(bad, "samples/x.bin", 16);
            let pj = prefetch_one(
                reqwest::Client::new(),
                // Unreachable: a malformed digest must be refused before any I/O.
                Arc::from("http://127.0.0.1:1"),
                None,
                test_spool(1 << 20),
                job,
            )
            .await;
            match pj.data {
                Err(PrefetchError::Skipped(msg)) => {
                    assert!(msg.contains("malformed sha256"), "message was: {msg}");
                }
                _ => panic!("sha256 {bad:?} must be refused before any I/O"),
            }
        }
    }

    /// The same check at the point the path is actually built, so the guard does
    /// not depend on every caller having validated first.
    #[tokio::test]
    async fn download_to_spool_refuses_a_malformed_sha256_before_touching_disk() {
        let parent = tempfile::tempdir().unwrap();
        let dir = parent.path().join("spool");
        let spool = SpoolState {
            dir: dir.clone(),
            budget_bytes: u64::MAX,
            used: AtomicU64::new(0),
            mem_threshold_bytes: 0,
            disk_headroom_bytes: 0,
        };
        let err = download_to_spool(
            &reqwest::Client::new(),
            "http://127.0.0.1:1",
            &spool,
            "../../escape",
            "samples/x.bin",
        )
        .await
        .expect_err("a malformed digest must not name a spool file");
        assert!(err.contains("malformed sha256"), "message was: {err}");
        // Nothing was created anywhere outside the spool dir.
        assert!(
            std::fs::read_dir(parent.path())
                .unwrap()
                .filter_map(Result::ok)
                .all(|e| e.path() == dir),
            "no stray entries beside the spool dir",
        );
    }

    #[tokio::test]
    async fn prefetch_one_uses_local_file_regardless_of_size() {
        // A local file needs no download or staging, so even a job bigger than
        // MAX_JOB_BYTES analyzes in place instead of being rejected.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("big.bin"), b"data").unwrap();
        let job = claim_job(&sha256_hex(b"data"), "big.bin", (MAX_JOB_BYTES + 1) as i64);
        let pj = prefetch_one(
            reqwest::Client::new(),
            Arc::from("http://127.0.0.1:1"),
            Some(dir.path().to_path_buf()),
            test_spool(1 << 20),
            job,
        )
        .await;
        assert!(matches!(pj.data, Ok(PrefetchData::Local)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prefetch_one_spools_large_payload_to_disk() {
        const PAYLOAD: &[u8] = b"a payload too big for the ram buffer";

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let Some(target) = read_target(&mut stream).await else {
                        return;
                    };
                    if target.starts_with("/data/") {
                        respond(&mut stream, "200 OK", PAYLOAD).await;
                    } else {
                        respond(&mut stream, "404 Not Found", b"").await;
                    }
                });
            }
        });

        // A 4-byte memory threshold forces the spool route.
        let spool = test_spool(4);
        let job = claim_job(
            &sha256_hex(PAYLOAD),
            "samples/big.bin",
            PAYLOAD.len() as i64,
        );
        let pj = prefetch_one(
            reqwest::Client::new(),
            Arc::from(format!("http://127.0.0.1:{port}").as_str()),
            None,
            Arc::clone(&spool),
            job,
        )
        .await;

        let data = pj.data.unwrap_or_else(|e| panic!("prefetch failed: {e}"));
        // Spooled payloads must not count against the RAM buffer.
        assert_eq!(data.staged_mem_bytes(), 0);
        let PrefetchData::Spooled(payload) = data else {
            panic!("payload above the memory threshold must spool to disk");
        };
        assert_eq!(std::fs::read(&payload.path).unwrap(), PAYLOAD);
        assert_eq!(spool.used.load(Ordering::Acquire), PAYLOAD.len() as u64);

        // Dropping the payload deletes the spool file and releases the budget.
        let spool_path = payload.path.to_path_buf();
        drop(payload);
        assert!(!spool_path.exists());
        assert_eq!(spool.used.load(Ordering::Acquire), 0);
    }

    /// A writer that collects formatted log output so a test can assert on what
    /// was actually emitted.
    #[derive(Clone, Default)]
    struct CapturedLog(Arc<std::sync::Mutex<Vec<u8>>>);

    impl CapturedLog {
        fn contains(&self, needle: &str) -> bool {
            self.0
                .lock()
                .map(|buf| String::from_utf8_lossy(&buf).contains(needle))
                .unwrap_or(false)
        }
    }

    impl std::io::Write for CapturedLog {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if let Ok(mut sink) = self.0.lock() {
                sink.extend_from_slice(buf);
            }
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Spawn a hopper that always answers `/api/next` with "no work", and run a
    /// prefetcher against it until `done` says stop. Returns whatever was logged.
    async fn run_against_empty_hopper(exit_if_empty: bool, done: &str) -> CapturedLog {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    if read_target(&mut stream).await.is_some() {
                        // 204: hopper is up and has nothing for this worker.
                        respond(&mut stream, "204 No Content", b"").await;
                    }
                });
            }
        });

        let captured = CapturedLog::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(captured.clone())
            .with_ansi(false)
            .finish();
        // Thread-local: this is a current-thread runtime, so every task polls on
        // this thread and picks up the subscriber.
        let _guard = tracing::subscriber::set_default(subscriber);

        let (tx, _rx) = mpsc::unbounded_channel::<PrefetchedJob>();
        let shutdown = Arc::new(AtomicBool::new(false));
        let handle = tokio::spawn(
            Prefetcher {
                pause: None,
                client: reqwest::Client::new(),
                base_url: Arc::from(format!("http://127.0.0.1:{port}").as_str()),
                data_dir: None,
                encoded_name: "test-worker".to_string(),
                available_tools: "7z".to_string(),
                slots: 4,
                spool: test_spool(1 << 20),
                max_buffer_bytes: 1 << 30,
                advertised_max_bytes: usize::try_from(MAX_JOB_BYTES).unwrap_or(usize::MAX),
                poll_secs: 1,
                target_depth: 8,
                metrics: Arc::new(WorkerMetrics::new()),
                poll_state: Arc::new(PollState::default()),
                exit_if_empty,
                // Standalone prefetcher: no embedded server to defer to, so the
                // quiet-period signals upstream added are all absent.
                last_analyze_request_ms: None,
                activity_started_at: None,
                quiet_period: None,
                // Far below the 1 s poll cadence, so the second empty poll trips it.
                idle_warn_after: Duration::from_millis(10),
            }
            .run(
                tx,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                Arc::clone(&shutdown),
            ),
        );

        let found = wait_until(|| captured.contains(done)).await;
        shutdown.store(true, Ordering::Relaxed);
        let _ = handle.await;
        if !found {
            // Not an assertion: the exempt case deliberately never logs.
            tracing::debug!("marker {done} never appeared");
        }
        captured
    }

    /// A worker whose hopper has nothing for it is a real problem — an empty
    /// queue, or a routing filter that excludes this worker from everything
    /// queued — and used to be indistinguishable from healthy idling in the log.
    #[tokio::test]
    async fn empty_hopper_is_reported_loudly() {
        let log = run_against_empty_hopper(false, "no work for this worker").await;
        assert!(
            log.contains("no work for this worker"),
            "a dry hopper must be reported",
        );
        assert!(log.contains("WARN"), "the dry-spell report must be at WARN");
        // The report has to carry the routing inputs, or an operator cannot tell
        // "the queue is empty" from "this worker is filtered out of a full queue".
        for field in ["dry_s", "hopper", "slots", "max_bytes", "tools"] {
            assert!(log.contains(field), "the report should carry `{field}`");
        }
    }

    /// `--exit-if-empty` (batch and benchmark runs) drains the hopper on purpose
    /// and then stops. Warning there would fire on every clean run.
    #[tokio::test]
    async fn batch_mode_draining_the_hopper_is_not_reported() {
        let log = run_against_empty_hopper(true, "--exit-if-empty stopping prefetch").await;
        assert!(
            !log.contains("no work for this worker"),
            "draining on purpose must not warn: {}",
            String::from_utf8_lossy(&log.0.lock().unwrap()),
        );
    }

    #[test]
    fn idle_warn_holds_until_the_threshold_then_repeats_on_a_slow_cadence() {
        let after = Duration::from_secs(120);

        // A gap shorter than the threshold is the normal pause between batches.
        assert!(!idle_warn_due(Duration::from_secs(0), None, after));
        assert!(!idle_warn_due(Duration::from_secs(119), None, after));

        // First crossing warns.
        assert!(idle_warn_due(Duration::from_secs(120), None, after));
        assert!(idle_warn_due(Duration::from_secs(9_000), None, after));

        // Already reported: stay quiet until the repeat cadence comes round, so
        // a long outage does not warn at the 2 s poll rate.
        assert!(!idle_warn_due(
            Duration::from_secs(300),
            Some(Duration::from_secs(0)),
            after,
        ));
        assert!(!idle_warn_due(
            Duration::from_secs(900),
            Some(IDLE_WARN_REPEAT - Duration::from_secs(1)),
            after,
        ));

        // ...and then re-warns, so an hours-long outage stays visible.
        assert!(idle_warn_due(
            Duration::from_secs(3_600),
            Some(IDLE_WARN_REPEAT),
            after,
        ));
    }

    /// The threshold is operator-tunable, so the policy must honour whatever it
    /// is handed rather than the default constant.
    #[test]
    fn idle_warn_respects_a_custom_threshold() {
        let after = Duration::from_secs(5);
        assert!(!idle_warn_due(Duration::from_secs(4), None, after));
        assert!(idle_warn_due(Duration::from_secs(5), None, after));
    }

    /// Regression: an OS temp sweep can delete the spool directory out from
    /// under a long-running worker (observed on Windows, where Storage Sense
    /// removes the empty directory under `%TEMP%`). The spool used to be created
    /// once at startup, so from that moment every payload above the memory
    /// threshold failed with `os error 3` for the rest of the process's life —
    /// including the "download directly" retry, which lands in the same missing
    /// directory. The spool must heal itself instead.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spool_recreates_its_directory_after_an_external_sweep() {
        const PAYLOAD: &[u8] = b"a payload too big for the ram buffer";

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let Some(target) = read_target(&mut stream).await else {
                        return;
                    };
                    if target.starts_with("/data/") {
                        respond(&mut stream, "200 OK", PAYLOAD).await;
                    } else {
                        respond(&mut stream, "404 Not Found", b"").await;
                    }
                });
            }
        });

        let parent = tempfile::tempdir().unwrap();
        let dir = parent.path().join("scan-spool");
        let spool = Arc::new(SpoolState {
            dir: dir.clone(),
            budget_bytes: u64::MAX,
            used: AtomicU64::new(0),
            // A 4-byte memory threshold forces the spool route.
            mem_threshold_bytes: 4,
            disk_headroom_bytes: 0,
        });
        spool.prepare();
        assert!(dir.is_dir(), "prepare() should create the spool dir");

        // The sweep.
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(!dir.exists());

        let job = claim_job(
            &sha256_hex(PAYLOAD),
            "samples/big.bin",
            PAYLOAD.len() as i64,
        );
        let pj = prefetch_one(
            reqwest::Client::new(),
            Arc::from(format!("http://127.0.0.1:{port}").as_str()),
            None,
            Arc::clone(&spool),
            job,
        )
        .await;

        let data = pj
            .data
            .unwrap_or_else(|e| panic!("spool must recreate its dir, got: {e}"));
        let PrefetchData::Spooled(payload) = data else {
            panic!("payload above the memory threshold must spool to disk");
        };
        assert_eq!(std::fs::read(&payload.path).unwrap(), PAYLOAD);
        assert!(dir.is_dir(), "the spool dir should have been recreated");
    }

    /// The free-disk gate reads the spool filesystem, and `free_disk_bytes`
    /// returns `None` for a path that does not exist — so a swept directory used
    /// to silently skip the check entirely. Reserving must heal the directory
    /// first, so the check measures the filesystem actually written to.
    #[test]
    fn try_reserve_recreates_a_swept_spool_directory() {
        let parent = tempfile::tempdir().unwrap();
        let dir = parent.path().join("scan-spool");
        let spool = SpoolState {
            dir: dir.clone(),
            budget_bytes: u64::MAX,
            used: AtomicU64::new(0),
            mem_threshold_bytes: 1 << 20,
            disk_headroom_bytes: 0,
        };
        assert!(!dir.exists());
        spool.try_reserve(1024).expect("reserve should heal the dir");
        assert!(dir.is_dir(), "try_reserve should have recreated the dir");
        assert_eq!(spool.used.load(Ordering::Acquire), 1024);
    }

    #[tokio::test]
    async fn download_bytes_rejects_sha256_mismatch() {
        const PAYLOAD: &[u8] = b"wrong bytes";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_target(&mut stream).await;
            respond(&mut stream, "200 OK", PAYLOAD).await;
        });
        let expected = sha256_hex(b"expected bytes");
        let err = download_bytes(
            &reqwest::Client::new(),
            &format!("http://{addr}"),
            &expected,
            "incoming/sample.bin",
        )
        .await
        .expect_err("mismatched bytes must fail");
        assert!(err.contains("sha256 mismatch"), "{err}");
        assert!(err.contains(&sha256_hex(PAYLOAD)), "{err}");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn download_to_spool_rejects_sha256_mismatch() {
        const PAYLOAD: &[u8] = b"wrong spooled bytes";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_target(&mut stream).await;
            respond(&mut stream, "200 OK", PAYLOAD).await;
        });
        let expected = sha256_hex(b"expected spooled bytes");
        let err = download_to_spool(
            &reqwest::Client::new(),
            &format!("http://{addr}"),
            &test_spool(0),
            &expected,
            "incoming/sample.bin",
        )
        .await
        .expect_err("mismatched spooled bytes must fail");
        assert!(err.contains("sha256 mismatch"), "{err}");
        assert!(err.contains(&sha256_hex(PAYLOAD)), "{err}");
        server.await.unwrap();
    }

    #[test]
    fn spool_budget_admits_when_idle_and_gates_when_busy() {
        let spool = SpoolState {
            dir: std::env::temp_dir(),
            budget_bytes: 100,
            used: AtomicU64::new(0),
            mem_threshold_bytes: 0,
            disk_headroom_bytes: 0,
        };
        // Idle spool admits even a payload beyond the budget (forward progress).
        assert!(spool.try_reserve(1000).is_ok());
        // Busy spool rejects anything that would exceed the budget...
        assert!(spool.try_reserve(1).is_err());
        // ...and reopens once the in-flight payload releases.
        spool.release(1000);
        assert!(spool.try_reserve(50).is_ok());
        assert!(spool.try_reserve(50).is_ok());
        assert!(spool.try_reserve(1).is_err());
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
                                        "sha256": sha256_hex(PAYLOAD),
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
                // Standalone worker: nothing to defer to.
                pause: None,
                last_analyze_request_ms: None,
                activity_started_at: None,
                quiet_period: None,
                client: reqwest::Client::new(),
                base_url: Arc::from(format!("http://127.0.0.1:{port}").as_str()),
                data_dir: None,
                encoded_name: "test".to_string(),
                available_tools: String::new(),
                slots,
                spool: test_spool(1 << 20),
                max_buffer_bytes: 1 << 30,
                advertised_max_bytes: usize::try_from(MAX_JOB_BYTES).unwrap_or(usize::MAX),
                poll_secs: 1,
                target_depth,
                metrics: Arc::new(WorkerMetrics::new()),
                poll_state: Arc::new(PollState::default()),
                exit_if_empty: false,
                idle_warn_after: Duration::from_secs(DEFAULT_IDLE_WARN_SECS),
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
            match pj.data.unwrap() {
                PrefetchData::Memory(bytes) => assert_eq!(&bytes[..], PAYLOAD, "payload mismatch"),
                PrefetchData::Local | PrefetchData::Spooled(_) => {
                    panic!("small payload should stage in memory")
                }
            }
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
            data: Ok(PrefetchData::Local),
            queue_id: 0,
        }
    }

    /// A spool over the system temp dir with an effectively unlimited budget
    /// and no free-disk requirement, so tests don't depend on host disk state.
    fn test_spool(mem_threshold_bytes: usize) -> Arc<SpoolState> {
        Arc::new(SpoolState {
            dir: std::env::temp_dir(),
            budget_bytes: u64::MAX,
            used: AtomicU64::new(0),
            mem_threshold_bytes,
            disk_headroom_bytes: 0,
        })
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

    #[test]
    fn cleave_concurrency_scales_with_pool_and_respects_slot_cap() {
        assert_eq!(cleave_concurrency_from(16, 32, None), 2);
        assert_eq!(cleave_concurrency_from(16, 8, None), 1);
        assert_eq!(cleave_concurrency_from(2, 64, None), 2);
        assert_eq!(cleave_concurrency_from(0, 32, None), 1);
    }

    #[test]
    fn cleave_concurrency_override_caps_at_slots() {
        assert_eq!(cleave_concurrency_from(4, 32, Some(64)), 4);
        assert_eq!(cleave_concurrency_from(16, 32, Some(2)), 2);
        // Zero / bogus overrides fall back to the pool formula.
        assert_eq!(cleave_concurrency_from(16, 32, Some(0)), 2);
    }

    #[test]
    fn os_thread_id_is_nonzero_on_supported_hosts() {
        let tid = crate::thread_dump::os_thread_id();
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "freebsd",
            target_os = "openbsd",
            target_os = "illumos",
            target_os = "solaris",
            windows
        ))]
        assert_ne!(tid, 0, "os_thread_id must resolve on this host");
        let _ = tid;
    }

    /// Reliability harness: mirrors the production worker loop's permit
    /// lifetimes without cleave/models/hopper. Each fake worker:
    ///   take job → analyzing++ → [optional cleave] → analyze → analyzing-- → post
    /// Regression targets are the Aug-18 hangs: a wedged post or a held cleave
    /// gate must not freeze sibling workers.
    async fn run_fake_workers<F, P>(
        jobs: Arc<JobSource>,
        n: usize,
        cleave: Arc<Semaphore>,
        analyzing: Arc<AtomicUsize>,
        completed: Arc<AtomicUsize>,
        analyze: F,
        post: P,
    ) where
        F: Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
            + Send
            + Sync
            + 'static,
        P: Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
            + Send
            + Sync
            + 'static,
    {
        let analyze = Arc::new(analyze);
        let post = Arc::new(post);
        let mut set = JoinSet::new();
        for _ in 0..n {
            let jobs = Arc::clone(&jobs);
            let cleave = Arc::clone(&cleave);
            let analyzing = Arc::clone(&analyzing);
            let completed = Arc::clone(&completed);
            let analyze = Arc::clone(&analyze);
            let post = Arc::clone(&post);
            set.spawn(async move {
                while let Some(pj) = jobs.recv().await {
                    let sha = pj.job.sha256.clone();
                    analyzing.fetch_add(1, Ordering::Release);
                    struct Guard(Arc<AtomicUsize>);
                    impl Drop for Guard {
                        fn drop(&mut self) {
                            self.0.fetch_sub(1, Ordering::Release);
                        }
                    }
                    let _guard = Guard(Arc::clone(&analyzing));
                    let permit = cleave.acquire().await.expect("cleave open");
                    analyze(sha.clone()).await;
                    drop(permit);
                    drop(_guard);
                    post(sha).await;
                    completed.fetch_add(1, Ordering::Release);
                }
            });
        }
        while set.join_next().await.is_some() {}
    }

    #[tokio::test]
    async fn job_source_serves_all_staged_jobs_to_concurrent_waiters() {
        let (tx, rx) = mpsc::unbounded_channel();
        for sha in ["a", "b", "c", "d"] {
            tx.send(staged_pj(sha, 10)).unwrap();
        }
        drop(tx);
        let jobs = Arc::new(JobSource::new(rx, false));
        let got = Arc::new(AtomicUsize::new(0));
        let mut set = JoinSet::new();
        for _ in 0..4 {
            let jobs = Arc::clone(&jobs);
            let got = Arc::clone(&got);
            set.spawn(async move {
                while jobs.recv().await.is_some() {
                    got.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        let deadline = tokio::time::sleep(Duration::from_secs(2));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                _ = set.join_next() => {
                    if set.is_empty() { break; }
                }
                _ = &mut deadline => panic!("JobSource did not drain under concurrent waiters"),
            }
        }
        assert_eq!(got.load(Ordering::Relaxed), 4);
    }

    /// Aug-17 regression: waiting on the cleave gate lived on the dispatch loop,
    /// so one held permit froze job intake. Siblings must still take jobs.
    #[tokio::test]
    async fn cleave_hold_does_not_block_sibling_job_intake() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(staged_pj("whale", 100)).unwrap();
        tx.send(staged_pj("sibling", 10)).unwrap();
        drop(tx);

        let jobs = Arc::new(JobSource::new(rx, false));
        let cleave = Arc::new(Semaphore::new(1));
        let analyzing = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        let whale_holds = Arc::new(tokio::sync::Notify::new());
        let release_whale = Arc::new(tokio::sync::Notify::new());

        let whale_holds2 = Arc::clone(&whale_holds);
        let release_whale2 = Arc::clone(&release_whale);
        let analyze = move |sha: String| {
            let whale_holds = Arc::clone(&whale_holds2);
            let release_whale = Arc::clone(&release_whale2);
            Box::pin(async move {
                if sha == "whale" {
                    whale_holds.notify_one();
                    release_whale.notified().await;
                }
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        };
        let post = |_sha: String| {
            Box::pin(async {}) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        };

        let workers = tokio::spawn(run_fake_workers(
            Arc::clone(&jobs),
            2,
            Arc::clone(&cleave),
            Arc::clone(&analyzing),
            Arc::clone(&completed),
            analyze,
            post,
        ));

        tokio::time::timeout(Duration::from_secs(2), whale_holds.notified())
            .await
            .expect("whale never entered analyze");
        // Harness bumps `analyzing` before cleave.acquire. While the whale holds
        // the only permit, the sibling must already be in that section (count
        // ≥ 2) — proving intake is not serialized on the gate.
        assert!(
            tokio::time::timeout(Duration::from_secs(2), async {
                while analyzing.load(Ordering::Acquire) < 2 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .is_ok(),
            "sibling failed to take a job while cleave was held (dispatch-loop regression)",
        );

        release_whale.notify_waiters();
        tokio::time::timeout(Duration::from_secs(2), workers)
            .await
            .expect("workers hung")
            .expect("workers panicked");
        assert_eq!(completed.load(Ordering::Relaxed), 2);
    }

    /// Aug-18 regression: post_result (dep sync / hopper timeouts) held the
    /// analysis slot, so a wedged hopper froze the worker. Post must run after
    /// analyzing drops so siblings keep moving.
    #[tokio::test]
    async fn post_hang_does_not_freeze_sibling_workers() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(staged_pj("hang-post", 100)).unwrap();
        tx.send(staged_pj("ok", 10)).unwrap();
        drop(tx);

        let jobs = Arc::new(JobSource::new(rx, false));
        let cleave = Arc::new(Semaphore::new(2));
        let analyzing = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        let ok_done = Arc::new(tokio::sync::Notify::new());
        let hang_entered_post = Arc::new(tokio::sync::Notify::new());

        let analyze = |_sha: String| {
            Box::pin(async {}) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        };
        let ok_done2 = Arc::clone(&ok_done);
        let hang_entered_post2 = Arc::clone(&hang_entered_post);
        let post = move |sha: String| {
            let ok_done = Arc::clone(&ok_done2);
            let hang_entered_post = Arc::clone(&hang_entered_post2);
            Box::pin(async move {
                if sha == "hang-post" {
                    hang_entered_post.notify_one();
                    std::future::pending::<()>().await;
                } else {
                    ok_done.notify_one();
                }
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        };

        let _workers = tokio::spawn(run_fake_workers(
            jobs,
            2,
            cleave,
            Arc::clone(&analyzing),
            Arc::clone(&completed),
            analyze,
            post,
        ));

        tokio::time::timeout(Duration::from_secs(2), hang_entered_post.notified())
            .await
            .expect("hanging post never started");
        // While one worker is wedged in post, analyzing must be 0 for that
        // worker — and the sibling must still complete.
        tokio::time::timeout(Duration::from_secs(2), ok_done.notified())
            .await
            .expect("sibling stuck behind wedged post (slot-held-across-post regression)");
        assert_eq!(
            completed.load(Ordering::Relaxed),
            1,
            "only the non-hanging job should have completed"
        );
        // Hanging worker is in post, not analyze.
        assert_eq!(analyzing.load(Ordering::Acquire), 0);
    }
}
