//! Pull-based worker that polls a hopper instance for analysis jobs.
//!
//! Shape (deliberately boring):
//!
//! ```text
//!   prefetcher ──► job channel ──► N slot tasks ──► per-job tails
//! ```
//!
//! Each slot loops: take a staged job → hand it to a tail → take the next. A
//! tail waits for the cleave gate and memory admission, analyzes, and posts.
//! There is no central dispatcher: the N slots *are* the claim limit, and a
//! permit is never held across a wait that does not need it — cleave covers
//! only the blocking classify, and hopper I/O runs after it is dropped, so a
//! wedged hopper cannot freeze analysis.
//!
//! [`run`] owns every task it starts. A stop (a signal, `--max-jobs`, the
//! stall watchdog) ends claiming; in-flight jobs get a short drain window to
//! finish, and whatever is left is aborted, which cancels its analysis.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::hash::{BuildHasherDefault, Hasher};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::Url;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex as AsyncMutex, Notify, OwnedSemaphorePermit, Semaphore, mpsc, watch};
use tokio::task::JoinSet;

use crate::admission::MemoryAdmission;
use crate::analysis::{ModelResources, classify_bytes, classify_file};
use crate::cli::Refresh;
use crate::explain::ShapImportance;
use crate::memory::MaxRssPolicy;
use crate::model::{Model, Thresholds};
use crate::system_load_avg;
use crate::upload::{hopper_token, use_hopper};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

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

/// Process-unique analysis id. Global rather than per-worker because the
/// in-flight census and the crash dump it keys are process-global too.
static NEXT_ANALYSIS_ID: AtomicU64 = AtomicU64::new(1);

/// How often a running worker pulls rule and model updates.
const RESOURCE_RENEWAL_INTERVAL: Duration = Duration::from_secs(10 * 60);

/// Cadence for the dedicated `/api/heartbeat` check-in. Fixed and independent of
/// the work-claim poll so a busy worker — prefetch buffer full, never polling
/// `/api/next` — still reports liveness, RSS, load, and queue depth on time.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);

/// Upper bound on how long [`run`] waits for in-flight analyses to drain after
/// a stop before cancelling them. Kept short so a redeploy is snappy: every
/// service supervisor (FreeBSD rc.d, systemd `TimeoutStopSec`, launchd)
/// SIGKILLs the process a few seconds after this as a backstop, so the worker
/// must exit within their grace window or be force-killed mid-drain. Whatever
/// does not finish here is re-leased by hopper, so exiting early costs a
/// re-scan, never a lost result. (Batch `--exit-if-empty` runs drain
/// unbounded instead — a finite dataset must complete, not re-lease.)
const SHUTDOWN_DRAIN: Duration = Duration::from_secs(15);

/// Cap on census lines per log event, so a saturated worker cannot flood the
/// log; the census is oldest first, so the most-stuck slots always appear.
const CENSUS_MAX_LINES: usize = 64;

/// Why [`run`] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// Asked to stop — a signal or `--max-jobs` — or `--exit-if-empty`
    /// drained the queue.
    Finished,
    /// The Rayon pool stopped making progress and cannot recover in process;
    /// the supervisor should restart the worker.
    Stalled,
}

impl Exit {
    /// The process exit status for this outcome: 0, or `EX_TEMPFAIL` (75)
    /// after a stall — transient, so a restart is the right response, and clear
    /// of the verdict codes (1 hostile, 2 suspicious) and the scan-error codes.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Finished => 0,
            Self::Stalled => 75,
        }
    }
}

/// What the progress watchdog concluded from one summary tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StallVerdict {
    /// Something completed, or there is nothing in flight to complete.
    Progressing,
    /// Slots are occupied and nothing has completed for the stuck threshold.
    Stalled,
    /// Stalled for so long that the pool will not recover; exit.
    Abort,
}

/// Classify one summary tick. `no_progress` is how long the worker has shown no
/// sign of life at all — nothing completed, no analysis changed stage, none
/// started, and no dependency payload finished — and an `abort_after` of
/// `None` disables the abort.
///
/// Idle is never a stall: with no slots occupied there is nothing to complete,
/// so a worker waiting on an empty queue must not be mistaken for a wedged one.
fn stall_verdict(
    active_slots: usize,
    no_progress: Duration,
    warn_after: Duration,
    abort_after: Option<Duration>,
) -> StallVerdict {
    if active_slots == 0 || no_progress < warn_after {
        return StallVerdict::Progressing;
    }
    if abort_after.is_some_and(|abort| no_progress >= abort.max(warn_after)) {
        return StallVerdict::Abort;
    }
    StallVerdict::Stalled
}

/// How many top-level analyses may execute at once, for the server: the pool
/// formula of [`cleave_concurrency_from`] with `SCAN_CLEAVE_CONCURRENCY` as the
/// override. The worker takes its override from [`WorkerTuning`] instead.
pub(crate) fn cleave_concurrency(slots: usize) -> usize {
    let override_value = std::env::var("SCAN_CLEAVE_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    cleave_concurrency_from(slots, rayon::current_num_threads(), override_value)
}

/// How many top-level analyses may execute at once.
///
/// The N worker tasks bound jobs that may be claimed/in flight; this tighter
/// gate bounds memory-bandwidth-heavy cleave executions. Within that gate,
/// cleave gives only a bounded subset of sibling analyses access to nested
/// Rayon work while the other admitted analyses make serial progress. The
/// default is the pool itself (at least 2) and never exceeds `slots`; an
/// override (`SCAN_CLEAVE_CONCURRENCY`) of 0 keeps the default.
///
/// Was 1/16 (one permit on a 16-thread host). Measured on the production
/// worker 2026-09-03: with one whale permit the pool ran at 2-3 of 16 cores
/// and every summary showed 11 of 16 slots queued behind it, because the
/// permit is held through the dependency fetch and (until [`CpuLease`]) the
/// LLM round trip, both network waits. A whale's own Rayon fan-out is bounded
/// by cleave's inner-work owner cap, and memory co-residency is already
/// guarded by [`crate::admission::MemoryAdmission`], so this lane no longer
/// needs to be the memory backstop it was sized as. 4 whale + 8 small permits
/// measured 5.3-5.4 cores; 8 measured 9-10; 12 measured 12.0 (75% of the pool);
/// 16 — every thread — measured 16.3, the pool saturated, at a 17.5 GB peak on
/// a 32 GB box with no memory-pressure warnings. Memory is the admission
/// gate's job (`admission.rs`), not this lane's.
///
/// Each worker waits on this gate *after* taking a job and *only* around the
/// blocking classify — never on a shared dispatch loop.
///
/// [`CpuLease`]: crate::engine::CpuLease
fn cleave_concurrency_from(slots: usize, pool: usize, override_value: Option<usize>) -> usize {
    let slots = slots.max(1);
    override_value.filter(|&value| value > 0).map_or_else(
        || pool.max(1).clamp(2.min(slots), slots),
        |value| value.min(slots),
    )
}

/// Small-lane width: the pool, 2..=64. Small jobs are cheap and mostly
/// single-threaded, so a pool's worth of them alongside the whales barely
/// touches the threads the whales are fanning out across; the ceiling keeps
/// a 128-thread host from running 128 blocking analyses on top of its own
/// fan-out. `SCAN_SMALL_LANE` overrides; 0 keeps the default. (Was a quarter,
/// 1..=8: four permits on a 16-thread host, each held for a 6-25 s LLM wait —
/// see [`crate::engine::CpuLease`].)
fn small_lane_from(pool: usize, override_value: Option<usize>) -> usize {
    override_value
        .filter(|&v| v > 0)
        .unwrap_or_else(|| pool.max(1).clamp(2, 64))
}

/// How many analyses may be past their slot at once (see [`slot_loop`]):
/// twice the slots, at least the slots; `SCAN_TAILS` overrides, 0 keeps the
/// default.
fn tail_cap_from(slots: usize, override_value: Option<usize>) -> usize {
    let slots = slots.max(1);
    override_value
        .filter(|&v| v > 0)
        .map_or(slots.saturating_mul(2), |v| v.max(slots))
}

/// The two admission lanes around the blocking cleave classify.
///
/// The whale gate (`cleave_concurrency`) exists to keep memory-bandwidth-heavy
/// archive analyses from co-residing; on hosts below 32 threads it is *one*.
/// Behind that one permit every claimed job used to queue — a 2 KB manifest
/// waited for whichever member of the current archive was slowest, and the
/// pool idled meanwhile. Measured on the 21-file `stuck` set (16 threads):
/// ~6 s of a 20 s run was small files serialized behind archives. Letting
/// only *small* jobs bypass the gate recovers that without the whale
/// co-residency that a wider gate brings (which measured +55% wall and
/// +1.9 GB on the whale-heavy long run — the pool starves).
#[derive(Debug)]
struct CleaveGate {
    whale: Arc<Semaphore>,
    small: Arc<Semaphore>,
    small_max_bytes: u64,
}

impl CleaveGate {
    fn new(whale_slots: usize, small_slots: usize, small_max_bytes: u64) -> Self {
        Self {
            whale: Arc::new(Semaphore::new(whale_slots)),
            small: Arc::new(Semaphore::new(small_slots)),
            small_max_bytes,
        }
    }

    /// Whether a job of `size` bytes takes the small lane.
    fn is_small(&self, size: u64) -> bool {
        self.small_max_bytes > 0 && size <= self.small_max_bytes
    }

    /// Wait for the lane a job of `size` bytes takes. The semaphores are never
    /// closed, so this only fails if that changes.
    async fn admit(&self, size: u64) -> Result<OwnedSemaphorePermit> {
        let lane = if self.is_small(size) {
            &self.small
        } else {
            &self.whale
        };
        Arc::clone(lane)
            .acquire_owned()
            .await
            .context("cleave analysis gate closed")
    }
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
    expected_size: Option<u64>,
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

    /// Walk `root` on `threads` threads (see [`WorkerTuning::index_threads`]).
    /// Unreadable entries are logged and skipped; only failing to start the
    /// walk is an error.
    fn build(root: PathBuf, threads: usize) -> Result<Self> {
        let started = Instant::now();
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
            // Bounded by the truncate above.
            let Ok(file_id) = FileId::try_from(idx) else {
                break;
            };
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
        expected_size: Option<u64>,
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
            expected_size,
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

/// Decode a 64-character hex SHA-256. `None` for any other length or a non-hex
/// byte — callers treat that as an invalid job. Stricter than
/// `burton::parse_sha256_hex`, which tolerates surrounding whitespace: the
/// digest also names files and URLs here, so it must be exactly 64 hex bytes.
fn sha256_from_hex(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    burton::parse_sha256_hex(hex)
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

// ---------------------------------------------------------------------------
// Startup resolution.
//
// A worker's memory ceiling, slot count, model bundle and operating point are
// resolved the same way whichever binary starts it. They live here rather than
// in a CLI's `main` so a second one cannot drift from the first.
// ---------------------------------------------------------------------------

/// Default slot count: three times the physical cores.
///
/// A slot spends most of its life waiting, not computing: the hopper
/// claim/prefetch round trips, the dependency fetch, and the LLM second
/// opinion are all network. Measured on the 16-core production worker
/// (2026-09-03): with 16 slots the pool never exceeded ~5 busy cores however
/// the cleave gate was sized, because every slot was parked in one of those
/// waits; 32 slots reached 5, 48 reached 9. Memory is bounded separately by
/// `admission::MemoryAdmission`, so extra slots cost only their prefetched
/// payload, and that buffer has its own budget.
#[must_use]
pub fn default_workers() -> NonZeroUsize {
    if let Some(cores) = cleave::memory_tracker::physical_cpu_count() {
        return NonZeroUsize::new(std::cmp::max(2, cores.saturating_mul(3)))
            .unwrap_or(NonZeroUsize::MIN);
    }
    let cores = cleave::memory_tracker::cpu_count().unwrap_or_else(|| {
        tracing::warn!(
            fallback = 4,
            "CPU count detection failed; defaulting worker basis to 4 cores",
        );
        4
    });
    // Logical count only: half of it approximates the physical cores, so the
    // same three-per-core default is 1.5x the logical count.
    NonZeroUsize::new(std::cmp::max(2, cores.saturating_mul(3) / 2)).unwrap_or(NonZeroUsize::MIN)
}

/// This host's name, or `"unknown"` when it cannot be read.
#[must_use]
pub fn default_worker_name() -> String {
    hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "unknown".to_string())
}

/// What a caller supplies to start a worker, before resolution.
///
/// [`WorkerConfig`] is the resolved form: every field settled, every default
/// taken. This is the unresolved one — the shape a command line hands over,
/// with `None` meaning "take the sensible default" rather than "off".
/// [`Startup::resolve`] turns one into the other, and is the only place those
/// defaults are decided, so two binaries starting a worker start it the same
/// way.
#[derive(Debug)]
pub struct Startup {
    /// Hopper's base URL. A worker uses one address, not the comma list
    /// `serve` accepts — a replica refuses worker routes outright, so an
    /// earlier address is not a fallback.
    pub hopper_url: String,
    /// Worker name. `None` takes the hostname.
    pub name: Option<String>,
    /// Concurrent analysis slots. `None` takes [`default_workers`].
    pub workers: Option<NonZeroUsize>,
    /// Pause between claim polls when hopper has no work.
    pub poll_interval: Duration,
    /// `--max-rss-gb`, resolved here against this host's (cgroup-aware) memory.
    pub max_rss: MaxRssPolicy,
    /// Nice value for the process; 0 leaves it unchanged.
    pub nice: i32,
    /// A sample tree this worker can read directly, skipping the download.
    pub data_dir: Option<PathBuf>,
    /// Stop after this many jobs. `None` runs until killed.
    pub max_jobs: Option<u64>,
    /// Exit rather than idle when hopper has nothing to claim.
    pub exit_if_empty: bool,
    /// Skip the trait-validation gate. A worker that starts with an
    /// incomplete rule set reports benign verdicts it has not earned, so this
    /// is for local work against on-disk rules, not for a fleet.
    pub no_validate: bool,
    /// Where the rules and the model come from.
    pub rules: RulesStartup,
}

impl Startup {
    /// Settle every default and prove the rule set is complete.
    ///
    /// Reads the `SCAN_*` tuning, resolves the memory ceiling, the slot count,
    /// the model bundle and the operating point, then runs the trait-validation
    /// gate unless [`Startup::no_validate`] waives it. The gate is the
    /// load-bearing part: a worker running a partial rule set answers benign
    /// for samples it never really examined, and does it quietly.
    ///
    /// # Errors
    ///
    /// Returns an error when a `SCAN_*` tuning variable does not parse, when
    /// the model bundle cannot be resolved, or when the trait-validation gate
    /// fails.
    pub fn resolve(self) -> Result<WorkerConfig> {
        // Before the refresh and validation, which take minutes: a typo in a
        // deploy file should fail the start at once.
        let tuning = WorkerTuning::from_env()?;
        let name = self.name.unwrap_or_else(default_worker_name);
        let workers = self.workers.unwrap_or_else(default_workers);
        let max_rss = self.max_rss.worker_ceiling();
        crate::memory::log_max_rss_resolution(
            "worker",
            self.max_rss,
            max_rss.map_or(0, NonZeroU64::get),
        );
        log_startup_diagnostics(&StartupDiagnostics {
            hopper_url: &self.hopper_url,
            name: &name,
            workers,
            poll_interval: self.poll_interval,
            max_rss_policy: self.max_rss,
            max_rss,
            data_dir: self.data_dir.as_deref(),
            max_jobs: self.max_jobs,
            traits_dir: self.rules.traits_dir.as_deref(),
            nice: self.nice,
        });

        let renew_rules = self.rules.refresh != Refresh::Skip;
        let rules = self.rules.resolve()?;
        if self.no_validate {
            tracing::warn!(
                "--no-validate: skipping the trait-validation gate; running \
                 against on-disk rules as-is",
            );
        } else {
            rules
                .validate()
                .context("worker startup validation failed")?;
        }

        Ok(WorkerConfig {
            hopper_url: self.hopper_url,
            name,
            workers,
            poll_interval: self.poll_interval,
            max_rss,
            data_dir: self.data_dir,
            max_jobs: self.max_jobs,
            exit_if_empty: self.exit_if_empty,
            renew_rules,
            nice: self.nice,
            rules,
            tuning,
        })
    }
}

/// Where a long-lived role's rules and model come from, before resolution.
///
/// Every daemon settles a model bundle, an operating point and the startup
/// refresh the same way; this is the shape it starts from, and
/// [`RulesStartup::resolve`] is where that happens.
#[derive(Debug)]
pub struct RulesStartup {
    /// Model bundle. `None` resolves the installed one.
    pub model_dir: Option<PathBuf>,
    /// Operating point in false positives per 100M. `None` takes the bundle's
    /// own default, then [`crate::model::DEFAULT_SEVERITY_LEVEL`]. Ignored
    /// when `thresholds` is set, which bypasses the level grid entirely.
    pub level: Option<u16>,
    /// Manual probability cutoffs, bypassing the level grid.
    pub thresholds: Option<Thresholds>,
    /// Traits bundle override (`--traits-dir`).
    pub traits_dir: Option<PathBuf>,
    /// The startup refresh (`-u` / `--no-update`). [`Refresh::Skip`] also
    /// pins the rules for the whole run.
    pub refresh: Refresh,
    /// Per-rule time budget before cleave logs a slow rule.
    pub slow_rule_ms: u64,
    /// The LLM second opinion, when one is configured.
    pub interpret: Option<crate::interpret::InterpretConfig>,
    /// Whether to follow the references a sample declares.
    pub fetch: crate::fetch::FetchPolicy,
    /// Passwords to try against encrypted archives.
    pub zip_passwords: crate::ArchivePasswords,
}

impl RulesStartup {
    /// Point cleave at the traits, refresh rules and models, and settle the
    /// bundle and operating point.
    ///
    /// # Errors
    ///
    /// Returns an error when no model bundle is named and none can be
    /// installed.
    pub fn resolve(self) -> Result<Rules> {
        // Order is load-bearing and is why the refresh lives here rather than
        // at the call site. The override has to be applied first, or the
        // refresh installs into the default directory while `--traits-dir`
        // points at an empty one — and a daemon started that way comes up,
        // reports healthy, and fails every analysis.
        if let Some(dir) = self.traits_dir.as_ref() {
            cleave::traits_repo::set_override_dir(Some(dir.into()));
        }
        crate::refresh_rules_at_startup(
            self.refresh == Refresh::Force,
            self.refresh == Refresh::Skip,
        );
        let model_dir = crate::cli::resolve_model_dir(self.model_dir)?;
        let level = crate::cli::operating_level(self.level, self.thresholds.is_some(), &model_dir);
        Ok(Rules {
            model_dir,
            level,
            thresholds: self.thresholds,
            slow_rule_ms: self.slow_rule_ms,
            interpret: self.interpret,
            fetch: self.fetch,
            zip_passwords: self.zip_passwords,
        })
    }
}

/// The rules and model a worker analyzes with: everything that must survive
/// every model load and periodic renewal unchanged.
#[derive(Debug)]
pub struct Rules {
    /// Model bundle directory.
    pub model_dir: PathBuf,
    /// The operating point that produced the thresholds, or `None` when
    /// manual thresholds were supplied.
    pub level: Option<u16>,
    /// Manual threshold overrides.
    pub thresholds: Option<Thresholds>,
    /// Slow rule warning threshold in ms.
    pub slow_rule_ms: u64,
    /// LLM interpretation; `None` disables the pass.
    pub interpret: Option<crate::interpret::InterpretConfig>,
    /// External-reference fetch policy for every job.
    pub fetch: crate::fetch::FetchPolicy,
    /// Additional passwords to try for encrypted archives.
    pub zip_passwords: crate::ArchivePasswords,
}

impl Rules {
    /// Run the trait-validation gate: the fixed benign corpus through these
    /// rules, offline.
    fn validate(&self) -> Result<()> {
        let config = crate::ScanConfig::new(
            &self.model_dir,
            crate::OutputFormat::Terminal,
            self.thresholds,
        )?
        .with_slow_rule_ms(self.slow_rule_ms)
        .with_level(self.level)
        .with_zip_passwords(self.zip_passwords.clone());
        crate::validate::run(&config, false)
    }

    /// Load the model bundle these rules name.
    fn load(&self) -> Result<Arc<ModelResources>> {
        let model =
            Model::load(&self.model_dir, self.thresholds, self.level).context("loading model")?;
        let shap = ShapImportance::load(&self.model_dir).context("loading SHAP data")?;
        Ok(Arc::new(ModelResources {
            model,
            shap,
            interpret: self.interpret.clone(),
            // Per-job scanning honors the worker's fetch policy. The validate
            // corpus never fetches — `crate::validate::run` builds its own
            // offline resources.
            fetch: self.fetch,
            zip_passwords: self.zip_passwords.clone(),
        }))
    }
}

/// A worker's resolved configuration, built by [`Startup::resolve`].
#[derive(Debug)]
pub struct WorkerConfig {
    /// Hopper API base URL (e.g. `http://hopper:8081`).
    pub hopper_url: String,
    /// Worker name, as hopper keys its claims.
    pub name: String,
    /// Concurrent claim slots.
    pub workers: NonZeroUsize,
    /// Pause between claim polls when hopper has no work.
    pub poll_interval: Duration,
    /// Memory ceiling that pauses admission, in bytes; `None` disables it.
    pub max_rss: Option<NonZeroU64>,
    /// Local data directory. Paths from hopper are joined with this root;
    /// a file that exists there and matches its SHA-256 is analyzed in place
    /// instead of downloaded.
    pub data_dir: Option<PathBuf>,
    /// Exit after this many jobs have been analyzed (`None` = run forever).
    pub max_jobs: Option<u64>,
    /// Exit cleanly once the hopper reports no further work and the prefetch
    /// queue has drained (for benchmarks / batch runs over a finite dataset).
    /// Unlike `max_jobs`, this does not depend on knowing the job count and
    /// cannot wedge the dispatch loop on a blocked claim.
    pub exit_if_empty: bool,
    /// Pull rule and model updates while running. Off under `--no-update`,
    /// which pins the on-disk rules for the whole run: a benchmark whose trait
    /// set swaps mid-run compares two rule sets, not two builds.
    pub renew_rules: bool,
    /// Nice value applied to the process at startup (0 = leave unchanged).
    pub nice: i32,
    /// The rules and model every job analyzes with.
    pub rules: Rules,
    /// Operator knobs from the `SCAN_*` environment.
    pub tuning: WorkerTuning,
}

/// Order in which staged jobs dispatch to slots (`SCAN_SJF`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DispatchOrder {
    /// Hopper handout order, unreordered (`SCAN_SJF=0`).
    Fifo,
    /// Smallest staged job first (default): protects small-job latency when a
    /// stream mixes sizes — a 2 KB manifest should not wait out an archive.
    Smallest,
    /// Largest staged job first (`SCAN_SJF=big`): LPT-style makespan trim for
    /// batch drains. The longest job bounds a batch's wall clock from below,
    /// so starting it as early as it is seen overlaps it with everything
    /// else; smallest-first provably ends the batch on the biggest job alone
    /// (measured: a 56 MB tgz ran solo for the last ~6 min of a 26-min
    /// 121-job drain). Latency-hostile on a live queue — meant for
    /// `--exit-if-empty` style batch runs.
    Largest,
}

impl std::str::FromStr for DispatchOrder {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "0" => Ok(Self::Fifo),
            "1" | "true" => Ok(Self::Smallest),
            "big" => Ok(Self::Largest),
            other => Err(format!("expected 0, 1 or big, got {other:?}")),
        }
    }
}

/// Operator knobs, each from one `SCAN_*` environment variable, read once at
/// startup. A value that does not parse fails the start rather than falling
/// back to a default: a typo in a deploy file must not quietly change how a
/// worker runs. `Default` is what an unset environment gives.
#[derive(Debug, Clone)]
pub struct WorkerTuning {
    /// `SCAN_CLEAVE_CONCURRENCY`: whale-gate permits; `None` or 0 sizes from
    /// the pool (see `cleave_concurrency_from`).
    pub cleave_concurrency: Option<usize>,
    /// `SCAN_SMALL_JOB_MB`: jobs at or below this size take the small lane
    /// instead of the whale gate; 0 disables the lane. 1 MiB by default.
    pub small_job_bytes: u64,
    /// `SCAN_SMALL_LANE`: small-lane permits; `None` or 0 sizes from the pool.
    pub small_lane: Option<usize>,
    /// `SCAN_TAILS`: analyses allowed past their slot; `None` or 0 is twice
    /// the slots.
    pub tails: Option<usize>,
    /// `SCAN_LLM_BACKLOG`: second opinions allowed to wait for the endpoint;
    /// `None` or 0 is four times the LLM client's in-flight cap.
    pub llm_backlog: Option<usize>,
    /// `SCAN_INDEX_THREADS`: threads for the `--data-dir` walk. `read_dir` and
    /// the per-file `stat` are I/O-bound, so this is queue depth for the
    /// storage device rather than CPU parallelism — one thread leaves any
    /// device with real seek latency almost entirely idle, which is what made
    /// a 3.5 M-file corpus on a spindle take longer to index than hopper's
    /// wedge timeout. 16 by default.
    pub index_threads: usize,
    /// `SCAN_SJF`: `0` is FIFO, `big` largest-first, anything else (`1`) the
    /// default smallest-first.
    pub dispatch_order: DispatchOrder,
    /// `SCAN_SJF_MAX_WAIT_SECS`: how long a staged job may be passed over by
    /// smaller arrivals before it dispatches anyway, so a stream of small jobs
    /// cannot starve archives indefinitely.
    ///
    /// Must sit well above typical *large-job service time*, not small-job
    /// time: on the realworld dataset (medium/large analyses run 6–25
    /// minutes) a 120 s bound aged out every staged archive while slots ground
    /// through earlier work, and the oldest-aged-first rule then preempted
    /// every small job — dispatch degenerated to FIFO and the SJF latency win
    /// vanished. 15 minutes by default.
    pub sjf_max_wait: Duration,
    /// `SCAN_PREFETCH_DEPTH`: staged jobs per slot, at least 1. `None` takes
    /// 2 under size-aware dispatch, which needs a window to reorder over (a
    /// 7-point sweep put the knee at 1.75–2× with nothing gained beyond), and
    /// 1.1 under FIFO.
    pub prefetch_depth: Option<f64>,
    /// `SCAN_SPOOL_DIR`: where payloads too big for the RAM buffer stream to.
    pub spool_dir: PathBuf,
    /// `SCAN_SPOOL_BUDGET_GB`: concurrently spooled bytes allowed on disk.
    pub spool_budget_bytes: u64,
    /// `SCAN_IDLE_WARN_SECS`: how long hopper may have nothing for this worker
    /// before the dry spell is reported at WARN. A worker pointed at a healthy
    /// hopper should never sit this long without a claim, so crossing it means
    /// something upstream is wrong — an empty queue, a routing filter no sample
    /// matches, or a worker whose advertised tools/`max_bytes` exclude it from
    /// everything queued. Well above the poll cadence, so the normal gaps
    /// between batches stay quiet. 2 minutes by default, at least 1 s.
    pub idle_warn_after: Duration,
    /// `SCAN_HEARTBEAT_SECS`: the worker summary cadence. 60 s by default; a
    /// short benchmark lowers it (at least 1 s) for a usable time series.
    pub summary_every: Duration,
    /// `SCAN_BREADCRUMB_SECS`: snapshot cleave's per-Rayon-thread breadcrumbs
    /// this often. Separate from wedge detection: a stack overflow aborts
    /// synchronously and can happen before any wedge threshold, so a recent
    /// snapshot is the evidence left behind. Off by default.
    pub breadcrumb_every: Option<Duration>,
    /// `SCAN_STUCK_WARN_SECS`: an analysis running this long is reported as a
    /// wedge, once. 5 minutes by default, at least 1 s.
    pub stuck_warn_after: Duration,
    /// `SCAN_STALL_WARN_SECS`: the whole worker showing no sign of life this
    /// long is a pool stall (see `stall_verdict`). 15 minutes by default, at
    /// least 1 s.
    pub stall_warn_after: Duration,
    /// `SCAN_STALL_ABORT_SECS`: a stall this long exits the process so the
    /// supervisor restarts it; 0 warns forever instead. 30 minutes by default
    /// — far longer than a healthy worker ever goes silent.
    pub stall_abort_after: Option<Duration>,
    /// `SCAN_ANALYSIS_TIMEOUT`: one job's analysis deadline, past which it is
    /// cancelled and reported to hopper as timed out; 0 disables it. The
    /// server's default request timeout unless set. Measured in time the
    /// process is running, so a worker frozen by its server does not wake to
    /// find every deadline spent.
    pub analysis_timeout: Option<Duration>,
}

impl Default for WorkerTuning {
    fn default() -> Self {
        Self {
            cleave_concurrency: None,
            small_job_bytes: MIB,
            small_lane: None,
            tails: None,
            llm_backlog: None,
            index_threads: 16,
            dispatch_order: DispatchOrder::Smallest,
            sjf_max_wait: Duration::from_secs(15 * 60),
            prefetch_depth: None,
            spool_dir: std::env::temp_dir().join("scan-spool"),
            spool_budget_bytes: 32 * GIB,
            idle_warn_after: Duration::from_secs(120),
            summary_every: Duration::from_secs(60),
            breadcrumb_every: None,
            stuck_warn_after: Duration::from_secs(300),
            stall_warn_after: Duration::from_secs(15 * 60),
            stall_abort_after: Some(Duration::from_secs(30 * 60)),
            analysis_timeout: Some(Duration::from_secs(
                crate::server::DEFAULT_ANALYSIS_TIMEOUT_SECS,
            )),
        }
    }
}

impl WorkerTuning {
    /// Read every knob from the process environment; unset ones keep their
    /// default.
    ///
    /// # Errors
    ///
    /// Returns an error naming the first variable that is set but does not
    /// parse, or `SCAN_PREFETCH_DEPTH` below 1.
    pub fn from_env() -> Result<Self> {
        let defaults = Self::default();
        let secs = |name: &str| -> Result<Option<Duration>> {
            Ok(env_var::<u64>(name)?.map(Duration::from_secs))
        };
        // Zero keeps the default for these, as it always has.
        let positive = |name: &str| -> Result<Option<usize>> {
            Ok(env_var::<usize>(name)?.filter(|&n| n > 0))
        };
        let at_least_one_sec = |name: &str, default: Duration| -> Result<Duration> {
            Ok(secs(name)?.map_or(default, |d| d.max(Duration::from_secs(1))))
        };
        // Zero disables these.
        let optional_secs = |name: &str, default: Option<Duration>| -> Result<Option<Duration>> {
            Ok(match secs(name)? {
                None => default,
                Some(d) => Some(d).filter(|d| !d.is_zero()),
            })
        };
        let prefetch_depth = env_var::<f64>("SCAN_PREFETCH_DEPTH")?;
        if let Some(depth) = prefetch_depth
            && (depth.is_nan() || depth < 1.0)
        {
            anyhow::bail!("SCAN_PREFETCH_DEPTH={depth}: must be at least 1");
        }
        Ok(Self {
            cleave_concurrency: env_var("SCAN_CLEAVE_CONCURRENCY")?,
            small_job_bytes: env_var::<u64>("SCAN_SMALL_JOB_MB")?
                .map_or(defaults.small_job_bytes, |mb| mb.saturating_mul(MIB)),
            small_lane: env_var("SCAN_SMALL_LANE")?,
            tails: env_var("SCAN_TAILS")?,
            llm_backlog: env_var("SCAN_LLM_BACKLOG")?,
            index_threads: positive("SCAN_INDEX_THREADS")?.unwrap_or(defaults.index_threads),
            dispatch_order: env_var("SCAN_SJF")?.unwrap_or(defaults.dispatch_order),
            sjf_max_wait: secs("SCAN_SJF_MAX_WAIT_SECS")?
                .filter(|d| !d.is_zero())
                .unwrap_or(defaults.sjf_max_wait),
            prefetch_depth,
            spool_dir: std::env::var_os("SCAN_SPOOL_DIR")
                .filter(|dir| !dir.is_empty())
                .map_or(defaults.spool_dir, PathBuf::from),
            spool_budget_bytes: env_var::<u64>("SCAN_SPOOL_BUDGET_GB")?
                .filter(|&gb| gb > 0)
                .map_or(defaults.spool_budget_bytes, |gb| gb.saturating_mul(GIB)),
            idle_warn_after: at_least_one_sec("SCAN_IDLE_WARN_SECS", defaults.idle_warn_after)?,
            summary_every: at_least_one_sec("SCAN_HEARTBEAT_SECS", defaults.summary_every)?,
            breadcrumb_every: optional_secs("SCAN_BREADCRUMB_SECS", None)?,
            stuck_warn_after: at_least_one_sec("SCAN_STUCK_WARN_SECS", defaults.stuck_warn_after)?,
            stall_warn_after: at_least_one_sec("SCAN_STALL_WARN_SECS", defaults.stall_warn_after)?,
            stall_abort_after: optional_secs("SCAN_STALL_ABORT_SECS", defaults.stall_abort_after)?,
            analysis_timeout: optional_secs("SCAN_ANALYSIS_TIMEOUT", defaults.analysis_timeout)?,
        })
    }

    /// Staged jobs per slot: [`WorkerTuning::prefetch_depth`], or the
    /// dispatch order's default.
    fn prefetch_depth(&self) -> f64 {
        self.prefetch_depth.unwrap_or(match self.dispatch_order {
            DispatchOrder::Fifo => 1.1,
            DispatchOrder::Smallest | DispatchOrder::Largest => 2.0,
        })
    }
}

/// `name` parsed as `T`; `None` when unset or empty. A value that is set but
/// does not parse is an error naming the variable.
fn env_var<T>(name: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(raw) = std::env::var_os(name) else {
        return Ok(None);
    };
    let Some(raw) = raw.to_str() else {
        anyhow::bail!("{name} is not valid UTF-8");
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    raw.parse()
        .map(Some)
        .map_err(|e| anyhow::anyhow!("{name}={raw:?}: {e}"))
}

/// What [`log_startup_diagnostics`] reports beyond the host's own memory view.
struct StartupDiagnostics<'a> {
    hopper_url: &'a str,
    name: &'a str,
    workers: NonZeroUsize,
    poll_interval: Duration,
    max_rss_policy: MaxRssPolicy,
    max_rss: Option<NonZeroU64>,
    data_dir: Option<&'a Path>,
    max_jobs: Option<u64>,
    traits_dir: Option<&'a Path>,
    nice: i32,
}

/// One line naming everything that sized this worker, before the refresh and
/// validation that can take minutes — so a worker that never comes up still
/// says what it was trying to be.
fn log_startup_diagnostics(d: &StartupDiagnostics<'_>) {
    use crate::memory::{cgroup_memory_diagnostics, proc_memtotal_mb, worker_memory_basis};

    let total_memory_mb = cleave::memory_tracker::total_memory().map(|b| b / MIB);
    let memory_limit_mb = cleave::memory_tracker::memory_limit() / MIB;
    let current_rss_mb = cleave::memory_tracker::current_rss().map(|b| b / MIB);
    let (proc_memtotal_mb, proc_memtotal_error) = match proc_memtotal_mb() {
        Ok(mb) => (Some(mb), None),
        Err(e) => (None, Some(e)),
    };
    let cgroup = cgroup_memory_diagnostics();
    let memory_basis = worker_memory_basis();

    if proc_memtotal_error.is_some() {
        if let Some(limit) = cgroup.effective_limit_bytes() {
            tracing::warn!(
                proc_memtotal_error = ?proc_memtotal_error,
                cgroup_memory_high = ?cgroup.memory_high,
                cgroup_memory_max = ?cgroup.memory_max,
                cgroup_effective_limit_mb = limit / MIB,
                auto_memory_basis_source = memory_basis.source,
                auto_memory_basis_mb = memory_basis.bytes / MIB,
                "proc meminfo unavailable; using shared memory detector for worker RSS auto-resolution",
            );
        } else if memory_basis.source == "fallback_16g" {
            tracing::warn!(
                proc_memtotal_error = ?proc_memtotal_error,
                "physical memory and cgroup memory limit unavailable; using 16 GiB fallback for worker RSS auto-resolution",
            );
        }
    }

    let max_rss_bytes = d.max_rss.map_or(0, NonZeroU64::get);
    tracing::info!(
        argv = ?redact_secrets(std::env::args()),
        hopper_url = d.hopper_url,
        worker_name = d.name,
        workers = d.workers.get(),
        poll_secs = d.poll_interval.as_secs(),
        max_rss_policy = ?d.max_rss_policy,
        resolved_max_rss_gb = max_rss_bytes / GIB,
        resolved_max_rss_mb = max_rss_bytes / MIB,
        rss_throttling_enabled = d.max_rss.is_some(),
        data_dir = ?d.data_dir,
        max_jobs = ?d.max_jobs,
        traits_dir = ?d.traits_dir,
        nice = d.nice,
        total_memory_mb = ?total_memory_mb,
        cleave_memory_limit_mb = memory_limit_mb,
        current_rss_mb = ?current_rss_mb,
        proc_memtotal_mb = ?proc_memtotal_mb,
        proc_memtotal_error = ?proc_memtotal_error,
        auto_memory_basis_source = memory_basis.source,
        auto_memory_basis_mb = memory_basis.bytes / MIB,
        cgroup_path = ?cgroup.path,
        cgroup_memory_current = ?cgroup.memory_current,
        cgroup_memory_current_mb = ?cgroup.memory_current_mb,
        cgroup_memory_high = ?cgroup.memory_high,
        cgroup_memory_high_mb = ?cgroup.memory_high_mb,
        cgroup_memory_max = ?cgroup.memory_max,
        cgroup_memory_max_mb = ?cgroup.memory_max_mb,
        "worker startup diagnostics",
    );
}

/// Flags whose value is a secret: an archive password or an LLM bearer token.
const SECRET_FLAGS: &[&str] = &["--zip-password", "--llm-key"];

/// The command line with every [`SECRET_FLAGS`] value replaced, fit for a log.
fn redact_secrets(args: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut args = args.into_iter();
    let mut redacted = Vec::new();
    while let Some(arg) = args.next() {
        if SECRET_FLAGS.contains(&arg.as_str()) {
            redacted.push(arg);
            if args.next().is_some() {
                redacted.push("<redacted>".to_string());
            }
        } else if let Some(flag) = SECRET_FLAGS
            .iter()
            .find(|f| arg.strip_prefix(**f).is_some_and(|v| v.starts_with('=')))
        {
            redacted.push(format!("{flag}=<redacted>"));
        } else {
            redacted.push(arg);
        }
    }
    redacted
}

/// Pull upstream rules and, **only if something actually changed**, re-validate
/// and reload the model bundle. Returns `Ok(None)` when both repos are already
/// up to date — a silent no-op so the periodic renewal doesn't flood the log
/// with a full validation pass every interval.
fn renew_resources_once(rules: &Rules) -> Result<Option<Arc<ModelResources>>> {
    // model_update validates the freshly extracted bundle (Model::load) before
    // swapping it in, so a broken bundle never lands on disk — there's no
    // last-known-good state to roll back to. A combined-validation failure below
    // propagates; the worker keeps serving its current in-memory resources until
    // the next successful renewal or a restart.
    let dir = crate::models_repo::install_target();
    let models_changed = match crate::model_update::update(&dir, false, false) {
        Ok(changed) => changed,
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

    rules.validate()?;
    let resources = rules.load()?;

    let (traits, composites) = cleave::reload_capability_mapper()
        .map_err(|error| anyhow::anyhow!("reload cleave capability mapper: {error}"))?;
    tracing::info!(traits, composites, "cleave capability mapper renewed");

    Ok(Some(resources))
}

/// Every [`RESOURCE_RENEWAL_INTERVAL`], pull rule and model updates and swap
/// in what changed. A failure keeps the last-known-good resources.
async fn renew_resources(handle: ResourceHandle, rules: Arc<Rules>, stop: Stop) {
    while !stop.sleep(RESOURCE_RENEWAL_INTERVAL).await {
        tracing::debug!(
            interval_secs = RESOURCE_RENEWAL_INTERVAL.as_secs(),
            "worker resource renewal check starting",
        );
        let rules = Arc::clone(&rules);
        let new_resources = match tokio::task::spawn_blocking(move || renew_resources_once(&rules))
            .await
        {
            Ok(Ok(Some(resources))) => resources,
            // Nothing changed upstream — silent no-op.
            Ok(Ok(None)) => continue,
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
        // A poisoned lock still holds a whole `Arc`; replacing it is safe.
        *handle.write().unwrap_or_else(PoisonError::into_inner) = new_resources;
        tracing::info!(spec_version, features, "worker resources renewed");
    }
}

// Phase 2 (WORKER_POOL_PLAN.md): litmus no longer manages rayon. There is no
// per-slot pool grid — N pull-style worker tasks share the one process-global
// rayon pool that main installs (sized to the host's cores, 256 MB stacks).
// Cleave's `par_iter` fan-out work-steals across that pool, so a single large
// archive can use the whole machine while total rayon threads stay capped at
// the pool size (not `slots × per-slot-threads`), which in turn caps cleave's
// per-thread YARA scanners.

/// Shared job intake for the N slot tasks.
///
/// Tokio's `mpsc::Receiver` is single-consumer, so slots serialize briefly on
/// this mutex to pull the next job. The lock is held across an empty
/// `recv().await` — that is fine: with no work, every slot is idle anyway.
/// As soon as a job is taken the lock drops and analysis runs concurrently.
///
/// Under size-aware dispatch it sweeps the prefetch channel into a reorder
/// window and picks from it (see [`pick_from_reorder`]).
struct JobSource {
    state: AsyncMutex<JobSourceState>,
    order: DispatchOrder,
    max_wait: Duration,
}

struct JobSourceState {
    rx: mpsc::UnboundedReceiver<PrefetchedJob>,
    reorder: Vec<(PrefetchedJob, Instant)>,
}

impl JobSource {
    fn new(
        rx: mpsc::UnboundedReceiver<PrefetchedJob>,
        order: DispatchOrder,
        max_wait: Duration,
    ) -> Self {
        Self {
            state: AsyncMutex::new(JobSourceState {
                rx,
                reorder: Vec::new(),
            }),
            order,
            max_wait,
        }
    }

    /// The next job to dispatch, or `None` once the prefetcher is gone and
    /// nothing is staged. Cancel-safe: a job swept into the reorder window
    /// stays there.
    async fn recv(&self) -> Option<PrefetchedJob> {
        let mut state = self.state.lock().await;
        if self.order == DispatchOrder::Fifo {
            return state.rx.recv().await;
        }
        let JobSourceState { rx, reorder } = &mut *state;
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
        let picked = pick_from_reorder(reorder, self.order, self.max_wait);
        drop(state);
        picked
    }
}

/// Hopper's claim tier for work something in the world has already called
/// malicious. Dispatched ahead of every other staged job.
const TIER_SIGHTED: &str = "sighted";

/// Take the next job from the reorder window: a sighted job first, then one
/// staged longer than `max_wait`, then by size in `order`.
fn pick_from_reorder(
    reorder: &mut Vec<(PrefetchedJob, Instant)>,
    order: DispatchOrder,
    max_wait: Duration,
) -> Option<PrefetchedJob> {
    // A sighted job outranks both the size sort and the aging bound, oldest
    // first among themselves.
    //
    // Hopper runs a priority ladder to decide this sample goes first -- the
    // registry may withdraw the artifact, and in the 2026-09-07 Shai-Hulud
    // reactivation it did so 4h25m after publication. Re-sorting that handout
    // by size discards the decision one step before it takes effect. Measured
    // 2026-09-08: a 12.4 MB sighted tarball sorted last under smallest-first
    // and waited 15 minutes, which is the SJF aging bound expiring, not queue
    // depth.
    //
    // Safe against starving the size policy because the tier is small by
    // construction and drains: hopper caps it at half of any one poll
    // (sightedMaxShare) and a row leaves it the moment it is analyzed. The
    // whole population was 54 rows against a 173k backlog when this was written.
    if let Some(idx) = reorder
        .iter()
        .enumerate()
        .filter(|(_, (pj, _))| pj.job.tier == TIER_SIGHTED)
        .min_by_key(|(_, (_, staged_at))| *staged_at)
        .map(|(i, _)| i)
    {
        return Some(reorder.swap_remove(idx).0);
    }
    let now = Instant::now();
    let aged = reorder
        .iter()
        .enumerate()
        .filter(|(_, (_, staged_at))| now.duration_since(*staged_at) >= max_wait)
        .min_by_key(|(_, (_, staged_at))| *staged_at)
        .map(|(i, _)| i);
    let idx = aged.or_else(|| {
        let sized = reorder
            .iter()
            .enumerate()
            .map(|(i, (pj, _))| (i, pj.job.size().unwrap_or(0)));
        match order {
            DispatchOrder::Largest => sized.max_by_key(|&(_, size)| size).map(|(i, _)| i),
            DispatchOrder::Smallest | DispatchOrder::Fifo => {
                sized.min_by_key(|&(_, size)| size).map(|(i, _)| i)
            }
        }
    })?;
    Some(reorder.swap_remove(idx).0)
}

#[derive(Deserialize)]
struct ClaimResponse {
    jobs: Vec<ClaimJob>,
}

#[derive(Debug, Deserialize)]
struct ClaimJob {
    sha256: String,
    path: String,
    /// As hopper sends it; read through [`ClaimJob::size`].
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
    /// Which of hopper's claim tiers this job came from ("sighted",
    /// "unanalyzed", "stale_traits", ...). Empty from a hopper that predates
    /// sending it, which is treated as ordinary work.
    #[serde(default)]
    tier: String,
}

impl ClaimJob {
    /// The sample's size, `None` when hopper sent a negative (unknown) one.
    fn size(&self) -> Option<u64> {
        u64::try_from(self.size_bytes).ok()
    }
}

/// A job with its file data pre-downloaded (or marked for local access).
struct PrefetchedJob {
    job: ClaimJob,
    /// `Ok(data)` = payload staged (in memory, spooled to disk, or local),
    /// `Err(Transient)` = download failed (fall back to direct download),
    /// `Err(Refused)` = job rejected without attempting download (e.g.
    /// oversized); do not retry, post the error result directly.
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
/// extraction) instead of being buffered in RAM.
const MAX_JOB_BYTES: u64 = 16 * GIB;

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

/// Why a claimed job was refused before any download. Permanent for this
/// sample: retrying cannot change the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Refusal {
    /// The digest is not 64 hex characters. It names the spool file, and
    /// `tempfile` splices a prefix into the name verbatim, so an unchecked
    /// string would be a path-traversal primitive.
    MalformedSha256(String),
    /// Larger than [`MAX_JOB_BYTES`].
    Oversized {
        /// Hopper's size for the sample.
        size: u64,
    },
}

impl std::fmt::Display for Refusal {
    /// Wire text: hopper's `classifyResultError` matches "exceeds per-job" to
    /// mark a sample skip='oversized' permanently.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedSha256(sha) => write!(
                f,
                "malformed sha256: expected 64 hex characters, got {sha:?}"
            ),
            Self::Oversized { size } => write!(
                f,
                "file size {size} exceeds per-job cap of {MAX_JOB_BYTES} bytes"
            ),
        }
    }
}

/// Why a prefetch did not produce bytes.
#[derive(Debug)]
enum PrefetchError {
    /// Download attempted and failed — `run_job` retries it directly.
    Transient(anyhow::Error),
    /// Download not attempted; permanent for this sample.
    Refused(Refusal),
}

impl std::fmt::Display for PrefetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transient(e) => write!(f, "{e:#}"),
            Self::Refused(refusal) => refusal.fmt(f),
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
    const DEFAULT_DISK_HEADROOM_BYTES: u64 = 8 * GIB;

    fn new(dir: PathBuf, budget_bytes: u64, mem_threshold_bytes: usize) -> Self {
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
    fn try_reserve(&self, size: u64) -> Result<()> {
        // The spool dir can vanish under a long-lived worker (a Windows %TEMP%
        // sweep removes it once it is empty), and `free_disk_bytes` returns
        // `None` for a missing path — silently skipping the disk check. Heal it
        // here so the check below measures the filesystem we will actually
        // write to.
        self.ensure_dir()?;
        if let Some(free) = free_disk_bytes(&self.dir)
            && free < size.saturating_add(self.disk_headroom_bytes)
        {
            anyhow::bail!(
                "insufficient free disk for spool: {free} bytes free, need {size} + {} headroom",
                self.disk_headroom_bytes
            );
        }
        // Check and reserve in one step: a separate load and add would let
        // two concurrent downloads both see room only one of them has.
        self.used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                (used == 0 || used.saturating_add(size) <= self.budget_bytes)
                    .then(|| used.saturating_add(size))
            })
            .map(drop)
            .map_err(|used| {
                anyhow::anyhow!(
                    "spool budget full ({used} of {} bytes in use)",
                    self.budget_bytes
                )
            })
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
    fn ensure_dir(&self) -> Result<()> {
        if self.dir.is_dir() {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("cannot create spool dir {}", self.dir.display()))
    }

    /// Create the spool directory and clear leftovers from crashed runs.
    /// Best-effort: only files older than a day are removed, so concurrent
    /// worker processes on the same host cannot delete each other's live
    /// spools.
    fn prepare(&self) {
        if let Err(e) = self.ensure_dir() {
            tracing::warn!(dir = %self.dir.display(), error = %format!("{e:#}"), "cannot create spool dir");
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
    // SAFETY: `statvfs` is a plain C struct, for which all-zero is a valid value.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: c_path is a valid NUL-terminated path and stat is a valid
    // out-pointer for the duration of the call.
    if unsafe { libc::statvfs(c_path.as_ptr(), &raw mut stat) } != 0 {
        return None;
    }
    #[allow(
        clippy::unnecessary_cast,
        reason = "f_bavail/f_frsize widths vary by platform"
    )]
    Some(stat.f_bavail as u64 * stat.f_frsize as u64)
}

#[cfg(not(unix))]
fn free_disk_bytes(_path: &Path) -> Option<u64> {
    None
}

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

/// The worker's stop signal, and why: waited on by every task without
/// polling. A stall overrides a plain stop — it must cut short even a drain
/// already under way — and nothing overrides a stall.
#[derive(Debug, Clone)]
struct Stop(watch::Sender<Option<Exit>>);

impl Stop {
    fn new() -> Self {
        Self(watch::Sender::new(None))
    }

    fn raise(&self, exit: Exit) {
        self.0.send_if_modified(|current| {
            let escalates = match *current {
                None => true,
                Some(Exit::Finished) => exit == Exit::Stalled,
                Some(Exit::Stalled) => false,
            };
            if escalates {
                *current = Some(exit);
            }
            escalates
        });
    }

    fn is_raised(&self) -> bool {
        self.0.borrow().is_some()
    }

    /// Why the worker is stopping; `Finished` if it was never asked to.
    fn exit(&self) -> Exit {
        self.0.borrow().unwrap_or(Exit::Finished)
    }

    /// Resolves once the signal is raised.
    async fn raised(&self) {
        let mut rx = self.0.subscribe();
        // `self` holds the sender, so the channel cannot close under us.
        let _ = rx.wait_for(Option::is_some).await;
    }

    /// Resolves once the worker has given up on a stalled pool.
    async fn stalled(&self) {
        let mut rx = self.0.subscribe();
        let _ = rx.wait_for(|exit| *exit == Some(Exit::Stalled)).await;
    }

    /// Sleep for `duration` or until the signal is raised, whichever is first.
    /// Returns whether it was raised.
    async fn sleep(&self, duration: Duration) -> bool {
        tokio::select! {
            () = tokio::time::sleep(duration) => self.is_raised(),
            () = self.raised() => true,
        }
    }
}

/// Raise `stop` on SIGTERM or SIGINT (Ctrl-C elsewhere). A registration
/// failure is logged, not fatal — better to run without graceful shutdown than
/// to refuse to start.
async fn stop_on_signal(stop: Stop) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let (mut sigterm, mut sigint) = match (
            signal(SignalKind::terminate()),
            signal(SignalKind::interrupt()),
        ) {
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
    stop.raise(Exit::Finished);
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
    // SAFETY: setpriority(PRIO_PROCESS, 0, ...) targets the caller and has no
    // memory effects; pid 0 means "self". On Linux, where nice is per-thread,
    // that is only the calling thread and the threads it starts afterwards.
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

/// State every worker task shares, behind one `Arc`.
struct WorkerShared {
    hopper: Hopper,
    tuning: WorkerTuning,
    /// Claim slots: the number of slot tasks.
    slots: usize,
    poll_interval: Duration,
    max_jobs: Option<u64>,
    exit_if_empty: bool,
    slow_rule_ms: u64,
    /// `--data-dir`: samples read in place.
    data_root: Option<PathBuf>,
    /// Built in the background from `data_root`. Until it lands, jobs resolve
    /// as if there were no index: the filesystem alone still finds every
    /// sample that sits where hopper says it does.
    local_index: OnceLock<LocalFileIndex>,
    /// Staged plus in-flight jobs the prefetcher keeps ahead of the slots.
    target_depth: usize,
    /// Cap on staged payload bytes held in RAM.
    max_buffer_bytes: usize,
    spool: Arc<SpoolState>,
    admission: Arc<MemoryAdmission>,
    cleave_gate: CleaveGate,
    /// Analyses allowed past their slot (see [`slot_loop`]).
    tails: Arc<Semaphore>,
    /// Second opinions allowed to wait for the LLM endpoint (see
    /// [`second_opinion`]).
    llm_queue: Arc<Semaphore>,
    /// Staged payload bytes held in RAM.
    queued_bytes: AtomicUsize,
    /// Staged plus in-flight jobs; bounds the prefetch depth.
    outstanding: AtomicUsize,
    /// Woken whenever a slot takes a staged job, so a prefetcher waiting for
    /// room re-checks at once.
    room: Notify,
    /// Jobs claimed and not yet finished, tails included. The heartbeat reports
    /// it as `active`: every one of them still holds its hopper claim.
    analyzing: AtomicUsize,
    /// Slots occupied: jobs between claim and hand-off to a tail. Bounded by
    /// `slots`, where `analyzing` is bounded by slots plus tails — so this, not
    /// `analyzing`, is what the summary measures against the slot total.
    /// Reporting the other read `active_slots=72` on a 24-slot worker
    /// (2026-09-05), which looked like a leak and was three bounded counts
    /// summed.
    dispatching: AtomicUsize,
    completed: AtomicU64,
    /// Analyses that reached, and that left, their blocking thread.
    blocking_started: AtomicU64,
    blocking_finished: AtomicU64,
    /// Second opinions run after the ML verdict was posted, skipped because
    /// the backlog was full, and re-posted because they changed it.
    llm_deferred: AtomicU64,
    llm_skipped: AtomicU64,
    llm_reposted: AtomicU64,
    metrics: WorkerMetrics,
    poll_state: PollState,
    stop: Stop,
}

/// A counter in [`WorkerShared`] that [`Held`] keeps one unit of.
#[derive(Debug, Clone, Copy)]
enum Gauge {
    Analyzing,
    Dispatching,
}

impl WorkerShared {
    /// Size every gate and buffer from `config` and the host, and log how.
    fn new(config: &WorkerConfig, hopper: Hopper) -> Self {
        let tuning = config.tuning.clone();
        let slots = config.workers.get();
        let pool = rayon::current_num_threads();
        let cleave_slots = cleave_concurrency_from(slots, pool, tuning.cleave_concurrency);
        let small_lane = small_lane_from(pool, tuning.small_lane);
        // A tail past its CPU work holds only its report, so twice the slots
        // is cheap, and without a bound a saturated LLM endpoint would grow
        // the backlog without limit.
        let tail_cap = tail_cap_from(slots, tuning.tails);
        // Four times the LLM client's in-flight cap keeps it fed without
        // letting a slow endpoint pile up reports without limit.
        let llm_backlog = tuning.llm_backlog.filter(|&v| v > 0).unwrap_or_else(|| {
            config
                .rules
                .interpret
                .as_ref()
                .map_or_else(crate::interpret::default_max_concurrency, |c| {
                    c.max_concurrency
                })
                .get()
                .saturating_mul(4)
        });
        let max_buffer_bytes = staged_buffer_bytes();
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a slot count times a small positive factor, rounded up"
        )]
        let target_depth = ((slots as f64 * tuning.prefetch_depth()).ceil() as usize).max(1);
        let spool = SpoolState::new(
            tuning.spool_dir.clone(),
            tuning.spool_budget_bytes,
            max_buffer_bytes / 2,
        );

        tracing::info!(
            llm_backlog,
            "two-phase post: LLM second opinions run after the ML verdict is posted (SCAN_LLM_BACKLOG)"
        );
        tracing::info!(
            tail_cap,
            "detached tails: analyses past memory admission run off their slot (SCAN_TAILS)",
        );
        tracing::info!(
            small_lane,
            small_job_mb = tuning.small_job_bytes / MIB,
            "small-job lane: jobs at or below the size limit bypass the cleave gate \
             (SCAN_SMALL_LANE / SCAN_SMALL_JOB_MB)",
        );
        tracing::info!(
            slots,
            cleave_slots,
            rayon_threads = pool,
            "worker concurrency: {slots} pull-style workers share one \
             {pool}-thread rayon pool (cleave gate={cleave_slots})",
        );
        // Each in-flight analysis parks a coordinator on the pool and fans
        // member work into it; slots far beyond the pool size just queue
        // analyses against each other (observed: 16 slots on a 4-thread
        // illumos zone → 5 s to run a trivial rayon task, 28 KB jobs taking
        // minutes). Likely a --workers value copied from a larger host.
        if slots > pool.saturating_mul(2) {
            tracing::warn!(
                slots,
                rayon_threads = pool,
                "worker slots exceed 2x the rayon pool; analyses will queue against \
                 each other for pool threads — lower --workers (or raise \
                 CLEAVE_RAYON_THREADS) to restore throughput",
            );
        }
        if cleave_slots < slots {
            tracing::warn!(
                slots,
                cleave_slots,
                "cleave entry gate allows {cleave_slots} simultaneous analyses; \
                 other claimed worker slots deliberately wait at this gate. Set \
                 SCAN_CLEAVE_CONCURRENCY to override",
            );
        }
        match tuning.dispatch_order {
            DispatchOrder::Smallest => tracing::info!(
                target_depth,
                max_staged_wait_s = tuning.sjf_max_wait.as_secs(),
                "size-aware dispatch: smallest staged job first (SCAN_SJF=0 for FIFO, SCAN_SJF=big for batch LPT)",
            ),
            DispatchOrder::Largest => tracing::info!(
                target_depth,
                max_staged_wait_s = tuning.sjf_max_wait.as_secs(),
                "size-aware dispatch: LARGEST staged job first (batch LPT; latency-hostile on a live queue)",
            ),
            DispatchOrder::Fifo => tracing::info!(target_depth, "FIFO dispatch (SCAN_SJF=0)"),
        }
        tracing::info!(
            spool_dir = %spool.dir.display(),
            spool_budget_gb = spool.budget_bytes / GIB,
            mem_threshold_mb = spool.mem_threshold_bytes as u64 / MIB,
            max_job_gb = MAX_JOB_BYTES / GIB,
            "large payloads spool to disk (SCAN_SPOOL_DIR / SCAN_SPOOL_BUDGET_GB)",
        );

        Self {
            hopper,
            slots,
            poll_interval: config.poll_interval,
            max_jobs: config.max_jobs,
            exit_if_empty: config.exit_if_empty,
            slow_rule_ms: config.rules.slow_rule_ms,
            data_root: config.data_dir.clone(),
            local_index: OnceLock::new(),
            target_depth,
            max_buffer_bytes,
            spool: Arc::new(spool),
            // Slot count bounds concurrency but not memory: a slot analysing a
            // huge archive holds it (plus expanded members) resident while a
            // slot analysing a 4 KB script holds nothing. This gate pauses
            // admission on live memory pressure at the resolved `--max-rss-gb`
            // ceiling, so a burst of large archives serialises instead of
            // co-residing. It pauses and reclaims; it never kills the worker.
            admission: MemoryAdmission::new(config.max_rss.map_or(0, NonZeroU64::get)),
            cleave_gate: CleaveGate::new(cleave_slots, small_lane, tuning.small_job_bytes),
            tails: Arc::new(Semaphore::new(tail_cap)),
            llm_queue: Arc::new(Semaphore::new(llm_backlog)),
            queued_bytes: AtomicUsize::new(0),
            outstanding: AtomicUsize::new(0),
            room: Notify::new(),
            analyzing: AtomicUsize::new(0),
            dispatching: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            blocking_started: AtomicU64::new(0),
            blocking_finished: AtomicU64::new(0),
            llm_deferred: AtomicU64::new(0),
            llm_skipped: AtomicU64::new(0),
            llm_reposted: AtomicU64::new(0),
            metrics: WorkerMetrics::new(),
            poll_state: PollState::default(),
            stop: Stop::new(),
            tuning,
        }
    }

    fn gauge(&self, gauge: Gauge) -> &AtomicUsize {
        match gauge {
            Gauge::Analyzing => &self.analyzing,
            Gauge::Dispatching => &self.dispatching,
        }
    }

    /// A slot took `pj` off the staging queue: give back its depth and buffer
    /// bytes, and wake the prefetcher if it was waiting for room.
    fn unstage(&self, pj: &PrefetchedJob) {
        let bytes = pj.data.as_ref().map_or(0, PrefetchData::staged_mem_bytes);
        self.queued_bytes.fetch_sub(bytes, Ordering::Release);
        self.outstanding.fetch_sub(1, Ordering::Release);
        self.room.notify_one();
    }
}

/// One unit of a [`Gauge`], given back on drop so no exit path can leak it.
struct Held {
    shared: Arc<WorkerShared>,
    gauge: Gauge,
}

impl Held {
    fn enter(shared: &Arc<WorkerShared>, gauge: Gauge) -> Self {
        shared.gauge(gauge).fetch_add(1, Ordering::Release);
        Self {
            shared: Arc::clone(shared),
            gauge,
        }
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.shared
            .gauge(self.gauge)
            .fetch_sub(1, Ordering::Release);
    }
}

/// Staged-payload budget: 1/16 of RAM, 512 MiB..=8 GiB. It bounds what the
/// prefetcher holds ahead of the slots, and the slot count scales with cores
/// (3x), so a fixed 1 GiB starved a large box while 1/16 of a 16 GB one is
/// the 1 GiB it always had.
fn staged_buffer_bytes() -> usize {
    let total = cleave::memory_tracker::total_memory().unwrap_or(16 * GIB);
    usize::try_from((total / 16).clamp(512 * MIB, 8 * GIB)).unwrap_or(usize::MAX)
}

/// Run the worker until it is told to stop, then drain.
///
/// Returns why it stopped; [`Exit::code`] is the status a binary should exit
/// with. Every task started here is owned here: the ones still running when it
/// returns are aborted, and an aborted job cancels its analysis.
///
/// # Errors
///
/// Returns an error when the hopper URL does not parse or the model bundle
/// fails to load.
pub async fn run(config: WorkerConfig) -> Result<Exit> {
    apply_nice(config.nice);
    tracing::info!(tuning = ?config.tuning, "worker tuning (SCAN_* environment)");
    let hopper = Hopper::new(&config.hopper_url, &config.name)?;
    // Every hopper call carries a token, so a worker without one claims
    // nothing. Report the source (or its absence) before the first poll, and
    // arm the dependency precheck against the hopper this worker claims from.
    use_hopper(&config.hopper_url);
    let shared = Arc::new(WorkerShared::new(&config, hopper));
    tracing::info!(
        name = %config.name,
        slots = shared.slots,
        hopper = %config.hopper_url,
        global_rayon_threads = rayon::current_num_threads(),
        pid = std::process::id(),
        "worker starting; send `kill -USR1 <pid>` for an all-thread backtrace",
    );

    warm_cleave();
    let resources: ResourceHandle = Arc::new(RwLock::new(config.rules.load()?));
    let rules = Arc::new(config.rules);

    let mut background = JoinSet::new();
    background.spawn(stop_on_signal(shared.stop.clone()));
    if config.renew_rules {
        background.spawn(renew_resources(
            Arc::clone(&resources),
            rules,
            shared.stop.clone(),
        ));
    } else {
        tracing::info!("--no-update: in-run model/traits renewal disabled");
    }
    if let Some(root) = config.data_dir {
        spawn_index_build(Arc::clone(&shared), root)?;
    }
    shared.spool.prepare();
    background.spawn(summary_loop(Arc::clone(&shared)));
    background.spawn(heartbeat_loop(Arc::clone(&shared)));

    let (tx, rx) = mpsc::unbounded_channel::<PrefetchedJob>();
    let mut prefetcher = tokio::spawn(prefetch_loop(Arc::clone(&shared), tx));
    let jobs = Arc::new(JobSource::new(
        rx,
        shared.tuning.dispatch_order,
        shared.tuning.sjf_max_wait,
    ));
    let mut slots = JoinSet::new();
    for slot in 0..shared.slots {
        slots.spawn(slot_loop(
            Arc::clone(&shared),
            Arc::clone(&jobs),
            Arc::clone(&resources),
            slot,
        ));
    }

    // A worker has no reason to stop on its own: it polls, analyses, posts, and
    // repeats. Park here until something asks it to, so the drain below
    // measures a shutdown deadline and not uptime.
    tokio::select! {
        // A signal, `--max-jobs` satisfied by the slots, or the stall watchdog.
        () = shared.stop.raised() => {}
        // The prefetcher is the only source of work. It returns on stop and,
        // in `--exit-if-empty` mode, when the hopper runs dry; any other return
        // is a panic, which starves every slot forever. Say so rather than idle
        // behind a heartbeat that still looks healthy — its channel is closed,
        // so the slots finish what is staged and then stop on their own.
        res = &mut prefetcher => {
            if !shared.exit_if_empty && !shared.stop.is_raised() {
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

    let exit = drain(&shared, slots).await;
    prefetcher.abort();
    background.shutdown().await;
    // End-of-run engine attribution (per-trait eval time under
    // CLEAVE_TRAIT_TIMING=1, raw-gate stats under CLEAVE_PHASE_STATS=1, regex
    // store churn). The CLI logs these per scan; a worker logs them once at
    // drain so a benchmark run ends with the aggregate.
    cleave::log_scan_stats();
    Ok(exit)
}

/// Warm cleave before the first job lands on a Rayon thread.
fn warm_cleave() {
    // Start background rayon pool health monitoring.
    cleave::start_rayon_diagnostics();
    // Warm YARA + capability mapper on a non-rayon thread before any job is
    // dispatched. The variant (`true`) must match `AnalysisOptions::default()`
    // — otherwise the prefetch warms an engine nobody uses and the first real
    // analysis triggers a cold compile on a rayon worker, which deadlocks the
    // pool. See cleave::shared_resources::yara_engine for the contract.
    crate::engine::prefetch_cleave_resources();
    // Then build every YARA bucket on a few background threads: buckets load
    // lazily per file type, and without this the first archive that touches
    // a PE member pays the ~3 s `pe` bucket JIT inside its analysis.
    cleave::prewarm_yara_buckets_background(true);
}

/// Build the `--data-dir` index on its own thread.
///
/// The walk scales with the corpus while hopper's liveness watchdog starts its
/// clock the moment this process spawns — blocking startup on it is precisely
/// how a worker gets killed for being "wedged" before it has ever polled for
/// work. Until the index lands, jobs resolve from the filesystem alone, or are
/// fetched from hopper, which is correct, merely slower. The thread ends on
/// its own when the walk does.
fn spawn_index_build(shared: Arc<WorkerShared>, root: PathBuf) -> Result<()> {
    std::thread::Builder::new()
        .name("sample-index".to_string())
        .spawn(
            move || match LocalFileIndex::build(root, shared.tuning.index_threads) {
                Ok(index) => {
                    if shared.local_index.set(index).is_err() {
                        tracing::error!("local sample index published twice");
                    }
                }
                Err(e) => tracing::error!(
                    error = %e,
                    "building local sample index failed; jobs will fetch payloads from hopper",
                ),
            },
        )
        .context("spawning local sample index builder")?;
    Ok(())
}

/// Let in-flight jobs finish — within [`SHUTDOWN_DRAIN`] unless this is a
/// batch run — then abort what is left, which cancels its analyses. A stall,
/// before or during the drain, skips the wait: a wedged pool will not drain,
/// and hopper re-leases its samples.
async fn drain(shared: &WorkerShared, mut slots: JoinSet<()>) -> Exit {
    let limit = (!shared.exit_if_empty).then_some(SHUTDOWN_DRAIN);
    let finished = async { while slots.join_next().await.is_some() {} };
    let deadline = async {
        match limit {
            Some(limit) => tokio::time::sleep(limit).await,
            None => std::future::pending().await,
        }
    };
    let drained = tokio::select! {
        () = finished => true,
        () = deadline => false,
        () = shared.stop.stalled() => false,
    };
    let exit = shared.stop.exit();
    if drained {
        if shared.exit_if_empty {
            tracing::info!("all in-flight jobs finished (batch drain), exiting");
        } else {
            tracing::info!("all in-flight jobs finished, exiting");
        }
    } else {
        if exit == Exit::Finished {
            tracing::warn!(
                still_running = slots.len(),
                drain_secs = SHUTDOWN_DRAIN.as_secs(),
                "drain timeout reached; cancelling in-flight analyses, which hopper re-leases",
            );
        }
        slots.shutdown().await;
    }
    exit
}

/// One claim slot: take the next staged job, hand it to a tail, repeat.
///
/// A slot's work ends at dispatch; what follows — the cleave-gate wait, the
/// memory reservation, the analysis, the LLM round trip, the hopper post — runs
/// as a tail so the slot can claim the next job while that one waits on the
/// network. On the production worker the tail is mostly waiting, and holding
/// the slot through it left the pool at 2-3 of 16 cores. Tails are bounded by
/// `tails`, not by the slot count. The slot owns its tails and waits for them
/// before it returns, so aborting a slot aborts — and cancels — what it
/// started.
async fn slot_loop(
    shared: Arc<WorkerShared>,
    jobs: Arc<JobSource>,
    resources: ResourceHandle,
    slot: usize,
) {
    let mut tails = JoinSet::new();
    loop {
        while let Some(done) = tails.try_join_next() {
            report_tail(done, slot);
        }
        if let Some(max) = shared.max_jobs
            && shared.completed.load(Ordering::Acquire) >= max
        {
            shared.stop.raise(Exit::Finished);
            break;
        }
        let pj = tokio::select! {
            biased;
            () = shared.stop.raised() => break,
            pj = jobs.recv() => pj,
        };
        // The prefetcher is gone and nothing is staged.
        let Some(pj) = pj else {
            break;
        };
        shared.unstage(&pj);
        let analyzing = Held::enter(&shared, Gauge::Analyzing);
        let _dispatching = Held::enter(&shared, Gauge::Dispatching);
        // Taken now, so a renewal mid-analysis cannot change the rules a job
        // started with. A poisoned lock still holds a whole `Arc`.
        let resources = Arc::clone(&*resources.read().unwrap_or_else(PoisonError::into_inner));
        let Ok(permit) = Arc::clone(&shared.tails).acquire_owned().await else {
            break;
        };
        tails.spawn(finish_job(
            Arc::clone(&shared),
            resources,
            pj,
            analyzing,
            permit,
            slot,
        ));
    }
    while let Some(done) = tails.join_next().await {
        report_tail(done, slot);
    }
}

/// A tail ended; say so if it panicked, which loses that job's post.
fn report_tail(done: std::result::Result<(), tokio::task::JoinError>, slot: usize) {
    if let Err(e) = done
        && e.is_panic()
    {
        tracing::error!(worker_id = slot, error = %e, "job tail panicked; its result was not posted");
    }
}

/// A job's tail: analyze, post the ML verdict, then — off the completion path
/// — the LLM second opinion. Holds its place in `analyzing` and its tail
/// permit until it returns.
async fn finish_job(
    shared: Arc<WorkerShared>,
    resources: Arc<ModelResources>,
    pj: PrefetchedJob,
    _analyzing: Held,
    _permit: OwnedSemaphorePermit,
    slot: usize,
) {
    let PrefetchedJob {
        job,
        data,
        queue_id,
    } = pj;
    let outcome = run_job(&shared, &resources, &job, data).await;
    if let Err(e) = &outcome {
        tracing::warn!(
            worker_id = slot,
            sha256 = %job.sha256,
            file = %job.path,
            file_type = %job.file_type,
            size = job.size_bytes,
            error = %e,
            "analysis failed",
        );
        shared.metrics.record_error(&e.to_string());
    }
    // Phase 1: post the ML verdict now. Keep a copy only when a second opinion
    // is pending, so a possible re-post has something to amend.
    let (post, later) = match outcome {
        Ok(out) => {
            let later = out
                .result
                .pending_llm
                .is_some()
                .then(|| (out.result.clone(), out.elapsed));
            let verdict = Verdict {
                envelope: out.result.into_envelope(),
                deps: out.deps,
                elapsed: out.elapsed,
            };
            (Post::Verdict(verdict), later)
        }
        Err(e) => (Post::Failed(e.to_string()), None),
    };
    shared.hopper.post_result(&job.sha256, post).await;
    shared.metrics.complete(queue_id);
    let n = shared.completed.fetch_add(1, Ordering::Release) + 1;
    // Checked here, where completions happen: a slot parked on an empty
    // queue would never see the count move.
    if shared.max_jobs.is_some_and(|max| n >= max) {
        shared.stop.raise(Exit::Finished);
    }
    if n.is_multiple_of(100) {
        // Clearing cleave's caches returns memory to the allocator; the trim is
        // what returns it to the OS, which is the half the admission gate can
        // actually see.
        let _ = tokio::task::spawn_blocking(reclaim_memory).await;
    }
    // Phase 2: the second opinion, off the completion path.
    if let Some((result, elapsed)) = later {
        second_opinion(&shared, &resources, result, elapsed, &job.sha256).await;
    }
}

/// Hand cleave's cached memory back to the allocator and the allocator's
/// retained pages back to the OS. Blocking.
fn reclaim_memory() {
    cleave::clear_all_thread_caches();
    crate::allocator::trim();
}

/// Drive the [`SummaryTicker`] until stop. Each tick runs on a blocking
/// thread: it reads `/proc` (or forks `ps` off Linux) and, on a stall abort,
/// takes a thread dump.
async fn summary_loop(shared: Arc<WorkerShared>) {
    let mut ticker = SummaryTicker::new(Arc::clone(&shared));
    let every = ticker.interval();
    tracing::debug!(
        heartbeat_secs = shared.tuning.summary_every.as_secs(),
        breadcrumb_secs = shared.tuning.breadcrumb_every.map(|d| d.as_secs()),
        "worker summary ticker armed"
    );
    while !shared.stop.sleep(every).await {
        // A panic anywhere in a tick (census formatting, wait-channel
        // resolution) would otherwise end all summary/wedge telemetry for the
        // rest of the worker's life — observed once in production as the
        // ticker going quiet ~45 minutes before exit with the worker still
        // running. Contain it: log the panic, keep ticking.
        let joined = tokio::task::spawn_blocking(move || {
            let verdict = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ticker.tick(Instant::now())
            }));
            (ticker, verdict)
        })
        .await;
        let verdict;
        (ticker, verdict) = match joined {
            Ok(done) => done,
            Err(e) => {
                tracing::error!(error = %e, "worker summary tick lost; ticker stops");
                return;
            }
        };
        match verdict {
            // The pool will not recover: stop, and exit as a stall.
            Ok(StallVerdict::Abort) => {
                shared.stop.raise(Exit::Stalled);
                return;
            }
            Ok(StallVerdict::Progressing | StallVerdict::Stalled) => {}
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_string());
                tracing::error!(panic = %msg, "worker summary tick panicked; ticker continues");
            }
        }
    }
}

/// The periodic worker summary, wedge census and stall watchdog. State
/// carried between ticks lives here; [`SummaryTicker::tick`] is one wake.
struct SummaryTicker {
    shared: Arc<WorkerShared>,
    rayon_threads: usize,
    /// Previous CPU-seconds and wall sample, for cores-busy.
    last_cpu: f64,
    last_at: Instant,
    /// When each analysis entered its current phase, so the census can report
    /// time-in-stage (a phase that never advances is the signature of a
    /// wedge). Pruned to the live set each tick.
    stage_since: HashMap<u64, (String, Instant)>,
    /// Analyses already announced as wedged, so the consolidated WEDGE event
    /// fires once per stuck analysis.
    wedge_latched: HashSet<u64>,
    /// Whether the allocator has already been asked to hand back retained
    /// pages during the *current* idle stretch. Latched so a worker parked on
    /// an empty hopper trims once, not once a minute forever.
    idle_trimmed: bool,
    last_summary: Instant,
    /// Stall detection: the completion and dependency counters as of the
    /// previous summary, and when the worker last showed any progress.
    last_finished_seen: u64,
    last_deps_seen: u64,
    progress_since: Instant,
    #[cfg(feature = "cleave-breadcrumbs")]
    last_breadcrumb: Instant,
}

impl SummaryTicker {
    fn new(shared: Arc<WorkerShared>) -> Self {
        let now = Instant::now();
        Self {
            last_finished_seen: shared.blocking_finished.load(Ordering::Relaxed),
            shared,
            rayon_threads: rayon::current_num_threads(),
            last_cpu: crate::inflight::process_cpu_secs(),
            last_at: now,
            stage_since: HashMap::new(),
            wedge_latched: HashSet::new(),
            idle_trimmed: false,
            last_summary: now,
            last_deps_seen: crate::fetch::payloads_analyzed_total(),
            progress_since: now,
            #[cfg(feature = "cleave-breadcrumbs")]
            last_breadcrumb: now,
        }
    }

    /// How often to wake: at least every 30 s, so a stuck slot self-documents
    /// promptly even when the summary itself is minutes apart, and at the
    /// breadcrumb cadence when one is set.
    fn interval(&self) -> Duration {
        let tuning = &self.shared.tuning;
        let wedge_check = tuning.summary_every.min(Duration::from_secs(30));
        tuning
            .breadcrumb_every
            .map_or(wedge_check, |interval| wedge_check.min(interval))
    }

    /// One wake: wedge reports and breadcrumbs every time, and when it is due
    /// the summary line, the stall check, the idle trim and the census.
    /// [`StallVerdict::Abort`] means the caller must stop the worker.
    fn tick(&mut self, now: Instant) -> StallVerdict {
        let cpu = crate::inflight::process_cpu_secs();
        let wall = now.duration_since(self.last_at).as_secs_f64().max(1e-6);
        // Average cores busy since the last tick. With slots full but this
        // near zero, the worker is blocked (locks / subprocess / I/O), not
        // grinding — the key blocked-vs-busy bit for triaging a wedge.
        let cpu_cores_busy = ((cpu - self.last_cpu) / wall).max(0.0);
        self.last_cpu = cpu;
        self.last_at = now;

        let census = crate::inflight::snapshot();
        let live: HashSet<u64> = census.iter().map(|e| e.analysis_id).collect();
        self.stage_since.retain(|id, _| live.contains(id));
        self.wedge_latched.retain(|id| live.contains(id));

        // Newly-stuck analyses: over the threshold and not yet announced.
        // Cheap (elapsed only); resolving wait-channels (which may fork `ps`
        // off Linux) is deferred until we know we need them.
        let stuck_after = self.shared.tuning.stuck_warn_after;
        let newly_stuck: Vec<&Arc<crate::inflight::Entry>> = census
            .iter()
            .filter(|e| now.duration_since(e.started) >= stuck_after)
            .filter(|e| !self.wedge_latched.contains(&e.analysis_id))
            .collect();
        let summary_due = now.duration_since(self.last_summary) >= self.shared.tuning.summary_every;
        self.breadcrumbs_if_due(now);

        // Resolve wait-channels once per tick, only when something will print
        // them (a wedge fired, or the summary census is due).
        let wchans = if newly_stuck.is_empty() && !summary_due {
            HashMap::new()
        } else {
            let tids: Vec<u64> = census
                .iter()
                .map(|e| e.thread_id.load(Ordering::Relaxed))
                .filter(|&t| t != 0)
                .collect();
            crate::inflight::wait_channels(&tids)
        };

        if !newly_stuck.is_empty() {
            self.report_wedges(&newly_stuck, census.len(), cpu_cores_busy, &wchans, now);
        }
        if !summary_due {
            return StallVerdict::Progressing;
        }
        self.last_summary = now;

        let progress = self.log_summary(cpu_cores_busy);
        let verdict = self.check_stall(&census, &progress, cpu_cores_busy, now);
        if verdict == StallVerdict::Abort {
            return verdict;
        }
        self.trim_if_idle();
        self.log_census(&census, &wchans, now);
        verdict
    }

    /// The `RAYON breadcrumb snapshot` lines, at `SCAN_BREADCRUMB_SECS`.
    #[cfg(feature = "cleave-breadcrumbs")]
    fn breadcrumbs_if_due(&mut self, now: Instant) {
        let due = self
            .shared
            .tuning
            .breadcrumb_every
            .is_some_and(|interval| now.duration_since(self.last_breadcrumb) >= interval);
        if !due {
            return;
        }
        log_breadcrumbs("RAYON breadcrumb snapshot", tracing::Level::INFO);
        self.last_breadcrumb = now;
    }

    #[cfg(not(feature = "cleave-breadcrumbs"))]
    fn breadcrumbs_if_due(&mut self, _now: Instant) {}

    /// Consolidated WEDGE event: fires once per stuck analysis, on the wedge
    /// cadence, so a hang self-documents without waiting for the summary.
    fn report_wedges(
        &mut self,
        newly_stuck: &[&Arc<crate::inflight::Entry>],
        inflight: usize,
        cpu_cores_busy: f64,
        wchans: &HashMap<u64, String>,
        now: Instant,
    ) {
        // Aggregate every thread's wait-channel: for archive wedges the real
        // blockage is on rayon workers, not the per-slot coordinator, so this
        // names the resource classes the pool is stuck on (yara symbol /
        // pipe_wait subprocess / futex lock).
        let thread_waits =
            crate::inflight::format_wait_summary(&crate::inflight::thread_wait_summary());
        tracing::warn!(
            newly_stuck = newly_stuck.len(),
            inflight,
            cpu_cores_busy = format!("{cpu_cores_busy:.1}"),
            rayon_threads = self.rayon_threads,
            stuck_threshold_s = self.shared.tuning.stuck_warn_after.as_secs(),
            thread_waits,
            "WEDGE DETECTED: analyses exceeded the stuck threshold; per-slot detail follows",
        );
        for entry in newly_stuck {
            self.wedge_latched.insert(entry.analysis_id);
            let phase = entry.phase.get();
            let stage = stage_name(&phase);
            tracing::warn!(
                analysis_id = entry.analysis_id,
                sha256 = %entry.sha,
                file = %entry.file,
                size_bytes = entry.size_bytes,
                file_type = %entry.file_type,
                thread_id = entry.thread_id.load(Ordering::Relaxed),
                stuck_for_ms = crate::duration_ms(now.duration_since(entry.started)),
                stage,
                waiting = waiting_for(wchans, entry, stage),
                "WEDGE slot",
            );
        }
        // Per-thread cleave breadcrumbs: which member each rayon worker is on.
        // For an archive wedge the work is spread across the pool, so this
        // names the member-level culprits the per-slot (coordinator) lines
        // can't.
        #[cfg(feature = "cleave-breadcrumbs")]
        log_breadcrumbs("WEDGE breadcrumb", tracing::Level::WARN);
    }

    /// The `worker summary` line. Returns the progress counters the stall
    /// check compares against.
    fn log_summary(&self, cpu_cores_busy: f64) -> Progress {
        let shared = &self.shared;
        let progress = Progress {
            started: shared.blocking_started.load(Ordering::Relaxed),
            finished: shared.blocking_finished.load(Ordering::Relaxed),
            deps_analyzed: crate::fetch::payloads_analyzed_total(),
            active_slots: shared.dispatching.load(Ordering::Relaxed),
        };
        // FreeBSD reads libc jemalloc via mallctl; everywhere else the bundled
        // tikv-jemalloc answers through cleave's ctl wrapper.
        let heap = crate::heap_profile::stats()
            .map(|s| {
                (
                    s.allocated as u64,
                    s.active as u64,
                    s.resident as u64,
                    s.retained as u64,
                )
            })
            .or_else(|| {
                cleave::memory_tracker::jemalloc_stats()
                    .map(|s| (s.allocated, s.active, s.resident, s.retained))
            });
        let (regex_scratch_bytes, regex_scratch_budget_bytes) = cleave::regex_scratch_usage();
        let [regex_str, regex_raw] = cleave::regex_store_usage();
        let (corpus_checks, corpus_skips) = crate::corpus_precheck::counters();
        let (purl_checks, purl_skips) = crate::corpus_precheck::purl_counters();
        tracing::info!(
            rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / MIB),
            jemalloc_allocated_mb = heap.map(|stats| stats.0 / MIB),
            jemalloc_active_mb = heap.map(|stats| stats.1 / MIB),
            jemalloc_resident_mb = heap.map(|stats| stats.2 / MIB),
            jemalloc_retained_mb = heap.map(|stats| stats.3 / MIB),
            regex_scratch_mb = regex_scratch_bytes as u64 / MIB,
            regex_scratch_budget_mb = regex_scratch_budget_bytes as u64 / MIB,
            regex_str_mb = regex_str.1 as u64 / MIB,
            regex_str_budget_mb = regex_str.2 as u64 / MIB,
            regex_str_entries = regex_str.0,
            regex_str_evictions = regex_str.3,
            regex_raw_mb = regex_raw.1 as u64 / MIB,
            regex_raw_budget_mb = regex_raw.2 as u64 / MIB,
            regex_raw_evictions = regex_raw.3,
            queued_prefetch_jobs = shared.outstanding.load(Ordering::Relaxed),
            prefetch_buffer_mb = shared.queued_bytes.load(Ordering::Relaxed) as u64 / MIB,
            active_slots = progress.active_slots,
            available_slots = shared.slots.saturating_sub(progress.active_slots),
            in_progress = shared.analyzing.load(Ordering::Relaxed),
            cpu_cores_busy = format!("{cpu_cores_busy:.1}"),
            load1 = system_load_avg().map(|load| format!("{load:.1}")),
            rayon_threads = self.rayon_threads,
            blocking_started_total = progress.started,
            blocking_finished_total = progress.finished,
            inflight_blocking = progress.started.saturating_sub(progress.finished),
            completed = shared.completed.load(Ordering::Acquire),
            deps_analyzed_total = progress.deps_analyzed,
            llm_deferred = shared.llm_deferred.load(Ordering::Relaxed),
            llm_skipped = shared.llm_skipped.load(Ordering::Relaxed),
            llm_reposted = shared.llm_reposted.load(Ordering::Relaxed),
            corpus_checks,
            corpus_skips,
            purl_checks,
            purl_skips,
            // Why this worker is (or is not) claiming. `poll_age_s` far above
            // the poll cadence means the loop is wedged; `last_claim=0` with a
            // fresh `poll_age_s` and non-zero `buffer_room` means hopper simply
            // has no work — an idle worker, not a stuck one.
            poll_age_s = shared.poll_age().as_secs(),
            last_want = shared.poll_state.last_want.load(Ordering::Acquire),
            last_claim = shared.poll_state.last_claim.load(Ordering::Acquire),
            buffer_room = shared.poll_state.buffer_room.load(Ordering::Acquire),
            "worker summary",
        );
        progress
    }

    /// The derived stall verdict.
    ///
    /// Every input is already on the summary line, but the conclusion is what
    /// an operator actually needs: slots full and `blocking_finished_total`
    /// not moving means nothing is completing at all. The per-slot census
    /// cannot say that — it names the *coordinator* thread of each analysis
    /// and the stage it entered, and the coordinators are precisely where the
    /// work is not. They are parked on a Rayon latch; the work is on the Rayon
    /// pool, which the census never names. So a stall is also the one moment
    /// worth spending a breadcrumb snapshot on: each line names the analyzer
    /// and member a Rayon worker is actually inside, which is the only view
    /// that points at a runaway leaf.
    ///
    /// Liveness is broader than "an analysis finished". A single whale
    /// legitimately holding every slot completes nothing for a long time while
    /// making steady progress, and killing that worker would be a false
    /// positive on a healthy machine — the one failure mode that would
    /// discredit the abort. So a stage transition counts as progress too (a
    /// working analysis walks archive:zip -> features+model -> done, while a
    /// wedged one sits on one stage for hours), as does an analysis the
    /// previous tick had not seen, and a finished dependency payload — a
    /// worker fetching transitive closures spends most of its life in the one
    /// `fetch+graft` stage (2026-09-07: twelve analyses there for an hour, each
    /// finishing dependency after dependency, and the abort killed a busy
    /// worker as a dead one).
    fn check_stall(
        &mut self,
        census: &[Arc<crate::inflight::Entry>],
        progress: &Progress,
        cpu_cores_busy: f64,
        now: Instant,
    ) -> StallVerdict {
        let stage_moved = census.iter().any(|entry| {
            self.stage_since
                .get(&entry.analysis_id)
                .is_none_or(|(seen, _)| *seen != entry.phase.get())
        });
        let deps_moved = progress.deps_analyzed != self.last_deps_seen;
        if progress.finished != self.last_finished_seen
            || progress.active_slots == 0
            || stage_moved
            || deps_moved
        {
            self.progress_since = now;
        }
        self.last_finished_seen = progress.finished;
        self.last_deps_seen = progress.deps_analyzed;
        let no_progress = now.duration_since(self.progress_since);
        let tuning = &self.shared.tuning;
        let verdict = stall_verdict(
            progress.active_slots,
            no_progress,
            tuning.stall_warn_after,
            tuning.stall_abort_after,
        );
        if verdict == StallVerdict::Progressing {
            return verdict;
        }
        tracing::warn!(
            no_progress_ms = crate::duration_ms(no_progress),
            completed_total = progress.finished,
            deps_analyzed_total = progress.deps_analyzed,
            active_slots = progress.active_slots,
            inflight_blocking = progress.started.saturating_sub(progress.finished),
            // Near zero with the slots full is the tell: the pool is parked on
            // latches, not grinding. Near one means a single runaway leaf is
            // holding it.
            cpu_cores_busy = format!("{cpu_cores_busy:.1}"),
            rayon_threads = self.rayon_threads,
            stall_warn_secs = tuning.stall_warn_after.as_secs(),
            "POOL STALLED: nothing has completed, changed stage, started, or \
             finished a dependency for the stall threshold; the Rayon pool \
             is not making progress. \
             Any breadcrumbs below name the analyzer and member each \
             Rayon worker is inside — a runaway leaf is among them",
        );
        #[cfg(feature = "cleave-breadcrumbs")]
        log_breadcrumbs("POOL STALLED breadcrumb", tracing::Level::WARN);
        if verdict == StallVerdict::Abort {
            self.log_abort(census, progress, no_progress, now);
        }
        verdict
    }

    /// Escalation: a pool this wedged does not recover.
    ///
    /// Rayon cannot preempt a running job, so once the workers' stacks have
    /// woven into one dependency chain behind a runaway leaf (2026-09-04: 11 of
    /// 12 threads with byte-identical stacks 32 minutes apart), nothing
    /// in-process can clear it. Cancellation is cooperative and a third-party
    /// parser never checks it. The only remaining lever is the process.
    ///
    /// Exiting is safe because hopper's claims are in-memory with expiring
    /// per-claim leases, and it resets a worker's claims when the process
    /// re-registers — losing claim state costs "wasted CPU, not corruption",
    /// because a repeat analysis is idempotent. `MaxClaimAttempts = 8` then
    /// stops a file that wedges workers from being handed out forever, so a
    /// poison sample cannot drive a restart loop.
    ///
    /// Every analysis thread's stack is dumped first: that is the artifact
    /// that names the runaway leaf.
    fn log_abort(
        &self,
        census: &[Arc<crate::inflight::Entry>],
        progress: &Progress,
        no_progress: Duration,
        now: Instant,
    ) {
        for entry in census.iter().take(CENSUS_MAX_LINES) {
            let phase = entry.phase.get();
            tracing::error!(
                analysis_id = entry.analysis_id,
                sha256 = %entry.sha,
                file = %entry.file,
                size_bytes = entry.size_bytes,
                thread_id = entry.thread_id.load(Ordering::Relaxed),
                stuck_for_ms = crate::duration_ms(now.duration_since(entry.started)),
                stage = stage_name(&phase),
                "STALL ABORT slot: in flight when the worker gave up",
            );
        }
        crate::thread_dump::dump_all_threads();
        tracing::error!(
            no_progress_ms = crate::duration_ms(no_progress),
            stall_abort_secs = self.shared.tuning.stall_abort_after.map(|d| d.as_secs()),
            completed_total = progress.finished,
            active_slots = progress.active_slots,
            exit_code = Exit::Stalled.code(),
            "STALL ABORT: nothing has completed, changed stage, started, or \
             finished a dependency for the abort threshold and a wedged \
             Rayon pool cannot recover in process; exiting so the supervisor \
             can restart. Claims expire hopper-side and the in-flight samples \
             above are handed out again",
        );
    }

    /// Idle with memory still held: hand the allocator's retained pages back
    /// to the OS, once per idle stretch. The admission gate rations intake on
    /// *live process memory*, so pages the allocator is only holding throttle
    /// the next batch as effectively as pages in use. Idle means nothing in
    /// flight at all — a tail still analyzing keeps the pool busy, and the
    /// Windows trim waits for every pool thread.
    fn trim_if_idle(&mut self) {
        if self.shared.analyzing.load(Ordering::Relaxed) != 0 {
            self.idle_trimmed = false;
            return;
        }
        if self.idle_trimmed {
            return;
        }
        self.idle_trimmed = true;
        let before = cleave::memory_tracker::current_rss();
        reclaim_memory();
        let after = cleave::memory_tracker::current_rss();
        if let (Some(before), Some(after)) = (before, after) {
            tracing::info!(
                rss_before_mb = before / MIB,
                rss_after_mb = after / MIB,
                reclaimed_mb = before.saturating_sub(after) / MIB,
                "idle: returned retained allocator pages to the OS",
            );
        }
    }

    /// Per-slot census: one line per in-flight analysis — file, size, how long
    /// it has been running, the stage it is in (and for how long), the worker
    /// thread, and what a blocked thread is waiting on. Lets an operator name
    /// a wedged slot from the log alone.
    fn log_census(
        &mut self,
        census: &[Arc<crate::inflight::Entry>],
        wchans: &HashMap<u64, String>,
        now: Instant,
    ) {
        let stuck_after = self.shared.tuning.stuck_warn_after;
        for entry in census.iter().take(CENSUS_MAX_LINES) {
            let phase = entry.phase.get();
            let stage = stage_name(&phase);
            let slot = self
                .stage_since
                .entry(entry.analysis_id)
                .or_insert_with(|| (phase.clone(), now));
            if slot.0 != phase {
                *slot = (phase.clone(), now);
            }
            let stage_elapsed = now.duration_since(slot.1);
            let total_elapsed = now.duration_since(entry.started);
            let thread_id = entry.thread_id.load(Ordering::Relaxed);
            let waiting = waiting_for(wchans, entry, stage);
            if total_elapsed >= stuck_after {
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
}

/// The counters one summary reports and the stall check compares.
struct Progress {
    started: u64,
    finished: u64,
    deps_analyzed: u64,
    active_slots: usize,
}

/// An analysis phase for a log line; `(starting)` before cleave reports one.
fn stage_name(phase: &str) -> &str {
    if phase.is_empty() {
        "(starting)"
    } else {
        phase
    }
}

/// What a census entry's thread is blocked on: its kernel wait channel when
/// one was read, otherwise the stage it is in.
fn waiting_for(
    wchans: &HashMap<u64, String>,
    entry: &crate::inflight::Entry,
    stage: &str,
) -> String {
    let tid = entry.thread_id.load(Ordering::Relaxed);
    wchans
        .get(&tid)
        .cloned()
        .unwrap_or_else(|| format!("stage:{stage}"))
}

/// One line per Rayon thread naming the analyzer and member it is inside,
/// capped at [`CENSUS_MAX_LINES`]. Needs a cleave build exposing
/// `cleave::breadcrumb`.
#[cfg(feature = "cleave-breadcrumbs")]
fn log_breadcrumbs(message: &'static str, level: tracing::Level) {
    for crumb in cleave::breadcrumb::snapshot()
        .into_iter()
        .take(CENSUS_MAX_LINES)
    {
        let age_ms = crate::duration_ms(crumb.age);
        if level == tracing::Level::INFO {
            tracing::info!(
                rayon_index = ?crumb.rayon_index,
                thread_id = crumb.thread_id,
                analyzer = crumb.analyzer,
                target = %crumb.target,
                age_ms,
                "{message}",
            );
        } else {
            tracing::warn!(
                rayon_index = ?crumb.rayon_index,
                thread_id = crumb.thread_id,
                analyzer = crumb.analyzer,
                target = %crumb.target,
                age_ms,
                "{message}",
            );
        }
    }
}

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

/// Live counts plus a metrics snapshot for one heartbeat.
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
    /// Time since the prefetcher last polled `/api/next` (large = stalled).
    poll_age: Duration,
    /// Jobs requested and returned on the last poll, and current free buffer room.
    last_want: usize,
    last_claim: usize,
    buffer_room: usize,
    /// sha256 of every in-flight analysis, so hopper renews their claim leases
    /// and a multi-hour scan is not re-claimed mid-flight.
    active_shas: Vec<Arc<str>>,
    metrics: MetricsSnapshot,
}

impl WorkerShared {
    /// Time since the prefetcher last polled `/api/next`.
    fn poll_age(&self) -> Duration {
        let last = Duration::from_secs(self.poll_state.last_poll_secs.load(Ordering::Acquire));
        self.metrics.start.elapsed().saturating_sub(last)
    }

    fn heartbeat_report(&self) -> HeartbeatReport {
        HeartbeatReport {
            slots: self.slots,
            active: self.analyzing.load(Ordering::Relaxed),
            queue: self.outstanding.load(Ordering::Acquire),
            mem_reserved_mb: self.admission.reserved_bytes() / MIB,
            mem_ceiling_mb: self.admission.ceiling_bytes() / MIB,
            poll_age: self.poll_age(),
            last_want: self.poll_state.last_want.load(Ordering::Acquire),
            last_claim: self.poll_state.last_claim.load(Ordering::Acquire),
            buffer_room: self.poll_state.buffer_room.load(Ordering::Acquire),
            active_shas: self.admission.in_flight_shas(),
            metrics: self.metrics.snapshot(),
        }
    }
}

/// Dedicated check-in. The claim loop only contacts hopper via `/api/next`
/// when the prefetch buffer has room, so a saturated worker can go long
/// stretches without reporting. This pings `/api/heartbeat` on a fixed cadence
/// regardless of buffer state, carrying live RSS, load, and an accurate queue
/// depth (staged backlog + running slots).
async fn heartbeat_loop(shared: Arc<WorkerShared>) {
    while !shared.stop.sleep(HEARTBEAT_INTERVAL).await {
        let url = shared.hopper.heartbeat_url(&shared.heartbeat_report());
        match shared.hopper.get(url).send().await {
            Ok(resp) if resp.status().is_success() => {}
            Ok(resp) => {
                tracing::debug!(status = %resp.status(), "heartbeat: non-success response");
            }
            Err(e) => tracing::debug!(error = %e, "heartbeat request failed"),
        }
    }
}

/// The claim side: keeps `target_depth` jobs staged ahead of the slots,
/// downloading payloads concurrently and sending each the moment it lands, so
/// a free slot never waits on the network. Runs until stop, or — with
/// `--exit-if-empty` — until hopper runs dry and the queue drains; returning
/// drops `tx`, which tells the slots no more work is coming.
async fn prefetch_loop(shared: Arc<WorkerShared>, tx: mpsc::UnboundedSender<PrefetchedJob>) {
    let mut consecutive_errors: u32 = 0;
    // Dry-spell tracking. `last_productive` is the last moment this worker had
    // a reason to believe hopper had work for it — a successful claim, or a
    // deliberate decision not to ask (buffer full). Measuring from there rather
    // than from the last empty poll means a hopper that trickles one job an
    // hour still reads as starved, which it is.
    let mut last_productive = Instant::now();
    let mut dry_warned_at: Option<Instant> = None;
    while !shared.stop.is_raised() {
        // Hold at the target depth and don't stage more bytes than the budget
        // allows. Either "buffer full" state waits for a slot to take a job.
        let room = shared
            .target_depth
            .saturating_sub(shared.outstanding.load(Ordering::Acquire));
        // Published every iteration: 0 is the heartbeat's signal that the
        // worker is saturated and deliberately not polling.
        shared.poll_state.buffer_room.store(room, Ordering::Release);
        let over_budget = shared.queued_bytes.load(Ordering::Acquire) >= shared.max_buffer_bytes;
        if room == 0 || over_budget {
            // Full buffer: this worker is saturated, not starved.
            last_productive = Instant::now();
            dry_warned_at = None;
            tokio::select! {
                () = shared.room.notified() => {}
                () = shared.stop.raised() => return,
            }
            continue;
        }

        // Cap a single poll's burst to `slots` so concurrent downloads stay
        // bounded; the depth fills over a few polls.
        let count = room.min(shared.slots);
        let url = shared.hopper.poll_url(count, shared.slots);
        // Stamp the poll so the heartbeat can report poll age and want/claim.
        shared
            .poll_state
            .last_poll_secs
            .store(shared.metrics.start.elapsed().as_secs(), Ordering::Release);
        shared.poll_state.last_want.store(count, Ordering::Release);
        match shared.hopper.claim(&url).await {
            Ok(None) => {
                shared.poll_state.last_claim.store(0, Ordering::Release);
                consecutive_errors = 0;
                // Hopper answered, and had nothing. Rare on a healthy
                // deployment, so say so loudly once the gap stops looking like
                // the pause between batches. `--exit-if-empty` runs
                // (batch/benchmark) drain to empty on purpose and are exempt.
                let dry = last_productive.elapsed();
                if !shared.exit_if_empty
                    && idle_warn_due(
                        dry,
                        dry_warned_at.map(|at| at.elapsed()),
                        shared.tuning.idle_warn_after,
                    )
                {
                    dry_warned_at = Some(Instant::now());
                    tracing::warn!(
                        dry_s = dry.as_secs(),
                        hopper = %shared.hopper.base,
                        worker = %shared.hopper.worker,
                        slots = shared.slots,
                        wanted = count,
                        max_bytes = MAX_JOB_BYTES,
                        tools = %shared.hopper.tools,
                        traits = cleave::traits_repo::version()
                            .map(|t| t.chars().take(5).collect::<String>()),
                        "hopper has had no work for this worker — every analysis slot is \
                         idle. Check hopper's queue depth; if it is non-empty this worker \
                         is being filtered out of it, so compare the tools, max_bytes and \
                         traits above against what the queued samples require.",
                    );
                }
                // Batch/benchmark mode: once the hopper has no work AND every
                // claimed job has been picked up, stop. Returning drops `tx`,
                // so the slots' `recv` yields `None` and the normal drain waits
                // for whatever is still in flight — instead of blocking forever
                // on a claim that will never arrive.
                if shared.exit_if_empty && shared.outstanding.load(Ordering::Acquire) == 0 {
                    tracing::info!(
                        "hopper drained and queue empty; --exit-if-empty stopping prefetch",
                    );
                    return;
                }
                shared.stop.sleep(shared.poll_interval).await;
            }
            Ok(Some(jobs)) => {
                shared
                    .poll_state
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
                if !stage_claimed(&shared, &tx, jobs).await {
                    return; // the slots are gone
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
                shared.stop.sleep(backoff).await;
            }
        }
    }
}

/// Download one poll's jobs concurrently and send each to the slots as it
/// lands. Returns `false` when the slots have gone away.
async fn stage_claimed(
    shared: &Arc<WorkerShared>,
    tx: &mpsc::UnboundedSender<PrefetchedJob>,
    jobs: Vec<ClaimJob>,
) -> bool {
    shared.outstanding.fetch_add(jobs.len(), Ordering::Release);
    let mut downloads = JoinSet::new();
    for job in jobs {
        let shared = Arc::clone(shared);
        downloads.spawn(async move {
            prefetch_one(
                &shared.hopper,
                shared.data_root.as_deref(),
                &shared.spool,
                job,
            )
            .await
        });
    }
    while let Some(res) = downloads.join_next().await {
        match res {
            Ok(mut pj) => {
                let bytes = pj.data.as_ref().map_or(0, PrefetchData::staged_mem_bytes);
                shared.queued_bytes.fetch_add(bytes, Ordering::Release);
                // Enters the local queue now; tracked until its tail finishes.
                pj.queue_id = shared.metrics.enqueue();
                if tx.send(pj).is_err() {
                    return false;
                }
            }
            Err(e) => {
                // Download task panicked; reclaim its depth slot.
                shared.outstanding.fetch_sub(1, Ordering::Release);
                tracing::warn!(error = %e, "prefetch task panicked");
            }
        }
    }
    true
}

/// Download one claimed job's payload (or mark it for local access / refusal).
/// Local files are used in place regardless of size, jobs above
/// [`MAX_JOB_BYTES`] are refused without a download, payloads too large for
/// the RAM buffer stream to the disk spool, and transient download failures
/// fall through to `run_job`'s direct-download retry.
async fn prefetch_one(
    hopper: &Hopper,
    data_dir: Option<&Path>,
    spool: &Arc<SpoolState>,
    job: ClaimJob,
) -> PrefetchedJob {
    let data = stage_payload(hopper, data_dir, spool, &job).await;
    PrefetchedJob {
        job,
        data,
        queue_id: 0,
    }
}

async fn stage_payload(
    hopper: &Hopper,
    data_dir: Option<&Path>,
    spool: &Arc<SpoolState>,
    job: &ClaimJob,
) -> std::result::Result<PrefetchData, PrefetchError> {
    // Refused before anything else, since the digest names the spool file:
    // a bad digest never becomes good.
    if sha256_from_hex(&job.sha256).is_none() {
        tracing::warn!(
            sha256 = %job.sha256,
            path = %job.path,
            "refusing job: sha256 is not 64 hex characters",
        );
        return Err(PrefetchError::Refused(Refusal::MalformedSha256(
            job.sha256.clone(),
        )));
    }

    // Local files need no download or staging, so no size check applies.
    if let Some(dir) = data_dir
        && tokio::fs::try_exists(dir.join(&job.path))
            .await
            .unwrap_or(false)
    {
        return Ok(PrefetchData::Local);
    }

    let size = job.size().unwrap_or(0);
    if size > MAX_JOB_BYTES {
        tracing::warn!(
            sha256 = %job.sha256,
            path = %job.path,
            size_bytes = job.size_bytes,
            max_job_bytes = MAX_JOB_BYTES,
            "skipping oversized job; reporting error to hopper",
        );
        return Err(PrefetchError::Refused(Refusal::Oversized { size }));
    }

    fetch_payload(hopper, spool, job)
        .await
        .map_err(PrefetchError::Transient)
}

/// Download a job's payload the size-appropriate way: into memory below the
/// spool threshold, streamed to a spool file above it. Shared by the prefetcher
/// and `run_job`'s direct-download fallback so both routes stay RAM-safe.
async fn fetch_payload(
    hopper: &Hopper,
    spool: &Arc<SpoolState>,
    job: &ClaimJob,
) -> Result<PrefetchData> {
    let size = job.size().unwrap_or(0);
    if size <= spool.mem_threshold_bytes as u64 {
        return hopper
            .download_bytes(&job.sha256, &job.path)
            .await
            .map(PrefetchData::Memory);
    }
    spool
        .try_reserve(size)
        .with_context(|| format!("cannot spool {size}-byte payload for {}", job.sha256))?;
    match hopper
        .download_to_spool(spool, &job.sha256, &job.path)
        .await
    {
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

/// Why a job produced no verdict. `Display` is the text posted to hopper,
/// whose `classifyResultError` sorts some of it by substring — "exceeds
/// per-job" marks a sample oversized, "analysis timed out" timed out, "no
/// local path" missing — so that wording is wire format.
#[derive(Debug)]
enum JobError {
    /// Refused before any download; permanent for this sample.
    Refused(Refusal),
    /// Ran past [`WorkerTuning::analysis_timeout`] and was cancelled.
    TimedOut(Duration),
    /// Anything else: a failed download, a cleave error, a panic.
    Failed(anyhow::Error),
}

impl std::fmt::Display for JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(refusal) => refusal.fmt(f),
            Self::TimedOut(limit) => write!(
                f,
                "analysis timed out after {}s (SCAN_ANALYSIS_TIMEOUT)",
                limit.as_secs()
            ),
            Self::Failed(e) => write!(f, "{e:#}"),
        }
    }
}

impl From<anyhow::Error> for JobError {
    fn from(e: anyhow::Error) -> Self {
        Self::Failed(e)
    }
}

/// A finished analysis, ready to post.
struct JobOutput {
    result: crate::engine::ScanResult,
    deps: Vec<crate::engine::DepResult>,
    elapsed: Duration,
}

/// How often an in-flight job wakes to check its deadline and log a slow
/// phase. Elapsed time is summed tick by tick, each tick capped at twice
/// this, so a worker frozen by its server (SIGSTOP, cgroup freeze) does not
/// wake to find every deadline spent.
const JOB_TICK: Duration = Duration::from_secs(15);

/// A phase this long is logged at INFO; three times this long at WARN, with a
/// pointer to the thread dump.
const SLOW_PHASE: Duration = Duration::from_secs(60);

/// Cancels an analysis when its job is dropped — an aborted tail must not
/// leave cleave running on a detached blocking thread. Harmless once the
/// analysis has returned.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Analyze a single job.
///
/// Resolution order for the sample bytes: the local index when it is available,
/// otherwise `--data-dir` on its own via [`resolve_on_disk`], and failing both a
/// download from hopper. The index is an optional accelerator — it finds
/// samples whose recorded path has drifted — so its absence costs recall on
/// moved files, never the ability to read a sample that is where it should be.
///
/// The cleave gate is acquired only for the blocking analyze — after async
/// download/provenance — so hopper I/O cannot pin nested-Rayon capacity.
async fn run_job(
    shared: &Arc<WorkerShared>,
    resources: &Arc<ModelResources>,
    job: &ClaimJob,
    prefetched: std::result::Result<PrefetchData, PrefetchError>,
) -> std::result::Result<JobOutput, JobError> {
    let analysis_id = NEXT_ANALYSIS_ID.fetch_add(1, Ordering::Relaxed);
    let label: Arc<str> = Path::new(&job.path)
        .file_name()
        .map(|n| Arc::from(n.to_string_lossy().as_ref()))
        .unwrap_or_else(|| Arc::from(job.sha256.as_str()));
    let sha_short: Arc<str> = Arc::from(job.sha256.get(..12).unwrap_or(&job.sha256));

    let local = locate(shared, job, &label).await?;
    // Use the prefetched payload, or download it now if prefetch failed. The
    // fallback goes through `fetch_payload`, so a payload too big for the RAM
    // buffer re-spools to disk instead of being buffered.
    let payload = match (&local, prefetched) {
        (Some(_), _) | (None, Ok(PrefetchData::Local)) => None,
        (None, Ok(data)) => {
            tracing::debug!(sha256 = %job.sha256, file = %label, size = job.size_bytes, "using prefetched data");
            Some(data)
        }
        (None, Err(PrefetchError::Refused(refusal))) => return Err(JobError::Refused(refusal)),
        (None, Err(PrefetchError::Transient(e))) => {
            tracing::warn!(sha256 = %job.sha256, file = %label, error = %format!("{e:#}"), "prefetch failed, downloading directly");
            Some(fetch_payload(&shared.hopper, &shared.spool, job).await?)
        }
    };

    // Registry metadata hopper collected for this sample at fetch time, so the
    // worker reasons over the same registry facts (age, custody, popularity,
    // deprecation) a live `pkg`/`url` scan fetches — without a refetch. Only
    // attempted when hopper flagged the sample as carrying it; best-effort, so a
    // miss never fails the scan. Consumed as stamped at collection time.
    let root_registry = if job.has_provenance {
        shared.hopper.download_provenance(&job.sha256).await
    } else {
        None
    };

    let input_size = match &payload {
        Some(PrefetchData::Memory(bytes)) => bytes.len() as u64,
        Some(PrefetchData::Spooled(spooled)) => spooled.size,
        Some(PrefetchData::Local) | None => job.size().unwrap_or(0),
    };
    let source = match &payload {
        Some(PrefetchData::Memory(_)) => "downloaded",
        Some(PrefetchData::Spooled(_)) => "spooled",
        Some(PrefetchData::Local) | None => "local",
    };
    let start = Instant::now();
    // Register the tracker with a descriptive label so cleave's rayon-diag
    // snapshot can name which analyses are in flight instead of just
    // reporting a count.
    let phase = crate::analysis::RequestPhase::with_label(format!("{sha_short} {label}"));
    // In the live census until this returns, so the periodic summary can
    // report its file, size, stage, time stuck, and what it is waiting on.
    let _census = crate::inflight::register(
        analysis_id,
        Arc::clone(&sha_short),
        Arc::clone(&label),
        input_size,
        Arc::from(job.file_type.as_str()),
        start,
        phase.tracker().clone(),
    );
    tracing::debug!(
        analysis_id,
        sha256 = %sha_short,
        file = %label,
        source,
        size = input_size,
        "analysis starting",
    );

    // Nested-Rayon capacity is needed only for the blocking classify. Acquiring
    // earlier (or on the dispatch loop) pinned the gate across hopper downloads
    // and froze every other slot behind one whale's preamble.
    let gate_wait_start = Instant::now();
    let small_lane = shared.cleave_gate.is_small(input_size);
    let cleave_permit = shared.cleave_gate.admit(input_size).await?;
    let gate_wait = gate_wait_start.elapsed();
    // Only now does this job cost memory: the gate's estimate predicts what an
    // analysis costs while it *runs*. Reserving before the cleave gate priced
    // every queued job as if it were already expanding an archive — on the
    // production worker four fifths of the committed memory belonged to jobs
    // parked in the CPU queue.
    let head = sniff(payload.as_ref(), local.as_deref()).await;
    let admission = shared
        .admission
        .admit(
            Arc::from(job.sha256.as_str()),
            Arc::from(job.path.as_str()),
            Arc::from(job.file_type.as_str()),
            job.size_bytes,
            head.as_deref(),
        )
        .await;
    if gate_wait >= Duration::from_secs(1) {
        tracing::info!(
            sha256 = %sha_short,
            file = %job.path,
            wait_ms = crate::duration_ms(gate_wait),
            small_lane,
            "analysis admitted to cleave after waiting for nested-work gate",
        );
    }

    let cancel = Arc::new(AtomicBool::new(false));
    let _cancel_on_drop = CancelOnDrop(Arc::clone(&cancel));
    let mut handle = {
        let shared = Arc::clone(shared);
        let resources = Arc::clone(resources);
        let cancel = Arc::clone(&cancel);
        let phase = phase.clone();
        let label = Arc::clone(&label);
        let sha_short = Arc::clone(&sha_short);
        tokio::task::spawn_blocking(move || {
            // Runs on a tokio blocking thread; cleave's `par_iter` fan-out
            // work-steals across the process-global rayon pool. Lifecycle logs
            // report `thread_id` — the blocking thread an operator samples to
            // find a wedged analysis; the CPU work itself runs on the rayon
            // pool threads.
            let started = shared.blocking_started.fetch_add(1, Ordering::Relaxed) + 1;
            let thread_id = crate::thread_dump::os_thread_id();
            // Attach the thread to the live census so the summary can report
            // which thread each in-flight analysis is wedged on and read its
            // kernel wait-channel.
            crate::inflight::set_thread_id(analysis_id, thread_id);
            // For the SIGUSR1 thread dump (rayon workers register via the
            // pool's start handler).
            crate::thread_dump::register_self();
            tracing::info!(
                analysis_id,
                sha256 = %sha_short,
                file = %label,
                thread_id,
                inflight_blocking = started
                    .saturating_sub(shared.blocking_finished.load(Ordering::Relaxed)),
                started_total = started,
                rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / MIB),
                "analysis starting on worker thread",
            );
            // Record this analysis as in flight so the SIGABRT handler can name
            // it if a deep analysis overflows the stack and aborts the process.
            // An abort skips the drop, leaving the entry live for the dump —
            // exactly the suspect set we want. See `crate::crash_dump`.
            let _inflight = crate::crash_dump::register(analysis_id, thread_id, &sha_short, &label);
            // The nested-work permit stays on this blocking thread through
            // classify/fetch/graft — dropping it from the async frame on cancel
            // would admit another tree while this one still owns Rayon workers
            // — and is handed to `classify_report` as a lease it releases right
            // before the LLM round trip, so the pool is not idle for a network
            // wait. The memory reservation goes with it. If classify bails
            // earlier the unused lease drops both.
            let cpu_lease: Option<crate::engine::CpuLease> = Some(Box::new(move || {
                drop(cleave_permit);
                drop(admission);
            }));
            // Spooled payloads take the same file-path route as local files, so
            // a multi-GiB sample is memory-mapped rather than held in RAM; the
            // payload drops when this returns, deleting the spool file. The
            // worker posts dependencies to hopper too, so it always captures
            // them.
            let analysis = crate::analysis::Analysis {
                cancellation: Some(&cancel),
                phase: Some(&phase),
                root_registry: root_registry.as_ref(),
                deps_for_upload: true,
                cpu_lease,
                ..crate::analysis::Analysis::new(&label, &resources, shared.slow_rule_ms)
            };
            let result = match (payload, local.as_ref()) {
                (Some(PrefetchData::Memory(data)), _) => classify_bytes(data, analysis),
                (Some(PrefetchData::Spooled(spooled)), _) => {
                    classify_file(&spooled.path, None, analysis)
                }
                (_, Some(path)) => classify_file(path, None, analysis),
                (None | Some(PrefetchData::Local), None) => Err(anyhow::anyhow!(
                    "no downloaded bytes and no local path for {label}"
                )),
            };
            let finished = shared.blocking_finished.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::debug!(
                analysis_id,
                sha256 = %sha_short,
                thread_id,
                inflight_blocking = shared
                    .blocking_started
                    .load(Ordering::Relaxed)
                    .saturating_sub(finished),
                finished_total = finished,
                rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / MIB),
                elapsed_ms = crate::duration_ms(start.elapsed()),
                phases = %phase.timeline(),
                "analysis complete on worker thread",
            );
            result
        })
    };

    // Wait for the analysis, waking every `JOB_TICK` to log a phase that has
    // gone slow and to enforce the deadline.
    let deadline = shared.tuning.analysis_timeout;
    let mut ticks = tokio::time::interval_at(tokio::time::Instant::now() + JOB_TICK, JOB_TICK);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut last_tick = Instant::now();
    let mut ran = Duration::ZERO;
    let mut slow = SlowPhase::default();
    let mut timed_out = None;
    let joined = loop {
        tokio::select! {
            biased;
            joined = &mut handle => break joined,
            _ = ticks.tick() => {
                let now = Instant::now();
                ran += now.duration_since(last_tick).min(2 * JOB_TICK);
                last_tick = now;
                let current = phase.get();
                slow.observe(&current, ran, analysis_id, &sha_short, &label);
                if let Some(limit) = deadline
                    && ran >= limit
                    && timed_out.is_none()
                {
                    timed_out = Some(limit);
                    cancel.store(true, Ordering::Release);
                    tracing::warn!(
                        analysis_id,
                        sha256 = %sha_short,
                        file = %label,
                        phase = stage_name(&current),
                        limit_s = limit.as_secs(),
                        "analysis exceeded its deadline (SCAN_ANALYSIS_TIMEOUT); cancelling",
                    );
                }
            }
        }
    };

    // A result that raced the cancel may be partial, so a timeout wins.
    if let Some(limit) = timed_out {
        return Err(JobError::TimedOut(limit));
    }
    match joined {
        Ok(Ok(mut result)) => {
            let deps = std::mem::take(&mut result.dependency_results);
            Ok(JobOutput {
                result,
                deps,
                elapsed: start.elapsed(),
            })
        }
        Ok(Err(e)) => Err(JobError::Failed(e)),
        Err(e) => Err(JobError::Failed(anyhow::anyhow!("task join error: {e}"))),
    }
}

/// Where the sample sits under `--data-dir`, if it is there. Confirming a
/// candidate hashes the whole file (up to [`MAX_JOB_BYTES`]), so the search
/// runs on a blocking thread.
async fn locate(
    shared: &Arc<WorkerShared>,
    job: &ClaimJob,
    label: &str,
) -> std::result::Result<Option<PathBuf>, JobError> {
    let Some(root) = shared.data_root.clone() else {
        return Ok(None);
    };
    let found = {
        let shared = Arc::clone(shared);
        let root = root.clone();
        let (path, sha256, size) = (job.path.clone(), job.sha256.clone(), job.size());
        tokio::task::spawn_blocking(move || match shared.local_index.get() {
            Some(index) => index.resolve(&path, &sha256, size),
            None => {
                let Some(expected) = sha256_from_hex(&sha256) else {
                    anyhow::bail!("expected 64-char hex sha256, got {sha256:?}");
                };
                Ok(resolve_on_disk(&root, &path, &expected, size))
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!("task join error: {e}"))??
    };
    match &found {
        Some(path) => tracing::debug!(
            sha256 = %job.sha256,
            path = %path.display(),
            file_type = %job.file_type,
            size = job.size_bytes,
            "analyzing local file"
        ),
        None => {
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
                indexed = shared.local_index.get().is_some(),
                "local file not found under --data; downloading from hopper"
            );
        }
    }
    Ok(found)
}

/// The leading bytes of a payload for the admission gate's archive sniff (see
/// `admission::looks_like_archive_bytes`). Best effort: an in-memory payload is
/// sliced, a file has its first `SNIFF_BYTES` read on a blocking thread, and
/// `None` means nothing was at hand.
async fn sniff(payload: Option<&PrefetchData>, local: Option<&Path>) -> Option<Vec<u8>> {
    let n = crate::admission::SNIFF_BYTES;
    let path = match (payload, local) {
        (Some(PrefetchData::Memory(bytes)), _) => {
            return Some(bytes[..bytes.len().min(n)].to_vec());
        }
        (Some(PrefetchData::Spooled(spooled)), _) => spooled.path.to_path_buf(),
        (_, Some(path)) => path.to_path_buf(),
        (Some(PrefetchData::Local) | None, None) => return None,
    };
    tokio::task::spawn_blocking(move || {
        use std::io::Read as _;
        let mut head = Vec::with_capacity(n);
        fs::File::open(path)
            .ok()?
            .take(n as u64)
            .read_to_end(&mut head)
            .ok()?;
        Some(head)
    })
    .await
    .ok()
    .flatten()
}

/// Tracks how long an analysis has sat in one phase, and says so once at
/// [`SLOW_PHASE`] and once more, louder, at three times that.
#[derive(Default)]
struct SlowPhase {
    phase: String,
    /// Running time when `phase` was first seen.
    since: Duration,
    logged: u8,
}

impl SlowPhase {
    fn observe(
        &mut self,
        current: &str,
        ran: Duration,
        analysis_id: u64,
        sha256: &str,
        file: &str,
    ) {
        if current != self.phase {
            current.clone_into(&mut self.phase);
            self.since = ran;
            self.logged = 0;
            return;
        }
        let in_phase = ran.saturating_sub(self.since);
        let phase = stage_name(current);
        if in_phase >= 3 * SLOW_PHASE && self.logged < 2 {
            self.logged = 2;
            tracing::warn!(
                analysis_id,
                sha256,
                file,
                phase,
                rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / MIB),
                elapsed_ms = crate::duration_ms(in_phase),
                pid = std::process::id(),
                "very slow phase; send `kill -USR1 <pid>` for an all-thread backtrace",
            );
        } else if in_phase >= SLOW_PHASE && self.logged < 1 {
            self.logged = 1;
            tracing::info!(
                analysis_id,
                sha256,
                file,
                phase,
                rss_mb = cleave::memory_tracker::current_rss().map(|rss| rss / MIB),
                elapsed_ms = crate::duration_ms(in_phase),
                "slow phase",
            );
        }
    }
}

/// Phase 2 of the two-phase post: the LLM second opinion, run after the ML
/// verdict is already on hopper, re-posting only if it changed anything.
/// `Required` admissions wait for backlog room; `Optional` ones are skipped
/// when there is none, so a saturated endpoint serves the cases that can
/// change a verdict.
async fn second_opinion(
    shared: &WorkerShared,
    resources: &Arc<ModelResources>,
    mut result: crate::engine::ScanResult,
    elapsed: Duration,
    sha256: &str,
) {
    let llm_queue = Arc::clone(&shared.llm_queue);
    let Some(permit) = (match result.pending_llm.as_ref().map(|p| p.admitted.tier) {
        Some(crate::interpret::LlmAdmission::Required) => llm_queue.acquire_owned().await.ok(),
        Some(crate::interpret::LlmAdmission::Optional) => {
            let permit = llm_queue.try_acquire_owned().ok();
            if permit.is_none() {
                shared.llm_skipped.fetch_add(1, Ordering::Relaxed);
            }
            permit
        }
        None => None,
    }) else {
        return;
    };
    shared.llm_deferred.fetch_add(1, Ordering::Relaxed);
    let resources = Arc::clone(resources);
    // The backlog permit covers the blocking LLM call and nothing after it.
    let amended = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let changed = resources.interpret.as_ref().is_some_and(|cfg| {
            crate::engine::apply_pending_interpretation(&mut result, cfg, &resources.model)
        });
        changed.then_some(result)
    })
    .await;
    if let Ok(Some(result)) = amended {
        shared.llm_reposted.fetch_add(1, Ordering::Relaxed);
        let verdict = Verdict {
            envelope: result.into_envelope(),
            deps: Vec::new(),
            elapsed,
        };
        shared
            .hopper
            .post_result(sha256, Post::Amended(verdict))
            .await;
    }
}

/// A verdict ready to post.
struct Verdict {
    envelope: crate::engine::ScanResultEnvelope,
    /// Fetched dependencies to mirror into hopper once the result is stored.
    deps: Vec<crate::engine::DepResult>,
    elapsed: Duration,
}

/// What a job posts to `/api/result`.
enum Post {
    /// The ML verdict, as soon as the analysis is done.
    Verdict(Verdict),
    /// The same result, amended by the LLM second opinion.
    Amended(Verdict),
    /// Why there is no verdict (a [`JobError`]'s text).
    Failed(String),
}

/// The hopper this worker claims from and posts to, and how it introduces
/// itself there.
#[derive(Debug)]
struct Hopper {
    /// Shared by every request; 120 s per request.
    client: reqwest::Client,
    /// Every route is a path under this.
    base: Url,
    /// Worker name, as hopper keys its claims.
    worker: String,
    /// The analyzer tools this host has, comma-separated, so hopper routes it
    /// only work it can finish.
    tools: String,
    /// Bearer token; hopper requires it on all of `/api/*` and `/data/`, and
    /// does not exempt loopback. See [`crate::upload::hopper_token`].
    token: Option<&'static str>,
}

impl Hopper {
    fn new(base: &str, worker: &str) -> Result<Self> {
        let base = Url::parse(base).with_context(|| format!("hopper URL {base:?}"))?;
        anyhow::ensure!(
            !base.cannot_be_a_base(),
            "hopper URL {base} cannot carry a path"
        );
        // 120 s per request is long enough for cold cleave scans yet short
        // enough that a wedged hopper can't pin the worker indefinitely —
        // without a timeout the default is "no timeout", which defeats graceful
        // shutdown.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("building the hopper HTTP client")?;
        Ok(Self {
            client,
            base,
            worker: worker.to_owned(),
            tools: crate::tools::available_names().join(","),
            token: hopper_token(),
        })
    }

    /// The base URL with `segments` appended to its path, each one
    /// percent-encoded, so a sample path cannot escape its route.
    fn url<'a>(&self, segments: impl IntoIterator<Item = &'a str>) -> Url {
        let mut url = self.base.clone();
        // `new` rejected a base that cannot carry a path.
        if let Ok(mut path) = url.path_segments_mut() {
            path.pop_if_empty().extend(segments);
        }
        url
    }

    fn get(&self, url: Url) -> reqwest::RequestBuilder {
        self.authed(self.client.get(url))
    }

    fn authed(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    /// What every poll and heartbeat tells hopper about this worker: who it
    /// is, which build and traits it runs, how loaded it is, and which tools it
    /// has. Hopper rations and routes work on these.
    fn identity(&self) -> Vec<(&'static str, String)> {
        let mut pairs = vec![
            ("worker", self.worker.clone()),
            ("version", env!("CARGO_PKG_VERSION").to_owned()),
        ];
        // 5-char prefix matches hopper's litmusTraitsVersion() truncation so
        // the dashboard's stale-traits comparison can string-equal the two.
        if let Some(traits) = cleave::traits_repo::version() {
            pairs.push(("traits", traits.chars().take(5).collect()));
        }
        if let Some(rss) = cleave::memory_tracker::current_rss() {
            pairs.push(("rss_mb", (rss / MIB).to_string()));
        }
        if let Some(load) = system_load_avg() {
            pairs.push(("load1", format!("{load:.2}")));
        }
        pairs.push(("tools", self.tools.clone()));
        pairs
    }

    /// The `/api/next` URL for a claim of `count` jobs. `max_bytes` keeps
    /// hopper from handing out files no worker here would analyze.
    fn poll_url(&self, count: usize, slots: usize) -> Url {
        let mut url = self.url(["api", "next"]);
        url.query_pairs_mut()
            .extend_pairs([
                ("count", count.to_string()),
                ("slots", slots.to_string()),
                ("max_bytes", MAX_JOB_BYTES.to_string()),
            ])
            .extend_pairs(self.identity());
        url
    }

    /// The `/api/heartbeat` URL: the poll's identity plus this worker's queue
    /// view and recent metrics. It claims no work.
    fn heartbeat_url(&self, report: &HeartbeatReport) -> Url {
        let metrics = &report.metrics;
        let mut pairs: Vec<(&'static str, String)> = vec![
            ("slots", report.slots.to_string()),
            ("active", report.active.to_string()),
            ("queue", report.queue.to_string()),
        ];
        // In-progress sha256s so hopper renews their claim leases. Bounded by
        // the slot count, so the query stays short.
        if !report.active_shas.is_empty() {
            let joined: Vec<&str> = report.active_shas.iter().map(AsRef::as_ref).collect();
            pairs.push(("active_shas", joined.join(",")));
        }
        // Ages are relative seconds, not wall-clock times, so hopper renders
        // "x ago" without depending on synchronised clocks.
        if let Some(age) = metrics.oldest_age {
            pairs.push(("oldest_s", age.as_secs().to_string()));
        }
        if let Some(age) = metrics.last_completion_age {
            pairs.push(("done_age_s", age.as_secs().to_string()));
        }
        pairs.push(("fps", format!("{:.3}", metrics.files_per_sec)));
        pairs.push(("errs", metrics.errors_recent.to_string()));
        if let Some((age, msg)) = &metrics.last_error {
            pairs.push(("err_age_s", age.as_secs().to_string()));
            // Trimmed to keep the URL bounded; hopper shows a short summary.
            pairs.push(("err", msg.chars().take(200).collect()));
        }
        // Why the worker is (or isn't) claiming. `mem_ceiling_mb` is the RAM
        // limit that throttles intake; `buffer_room=0` with a large
        // `poll_age_s` means it's saturated by slow jobs, not starved.
        pairs.extend([
            ("mem_reserved_mb", report.mem_reserved_mb.to_string()),
            ("mem_ceiling_mb", report.mem_ceiling_mb.to_string()),
            ("poll_age_s", report.poll_age.as_secs().to_string()),
            ("want", report.last_want.to_string()),
            ("last_claim", report.last_claim.to_string()),
            ("buffer_room", report.buffer_room.to_string()),
        ]);
        let mut url = self.url(["api", "heartbeat"]);
        url.query_pairs_mut()
            .extend_pairs(pairs)
            .extend_pairs(self.identity());
        url
    }

    /// Poll `/api/next` once. `Ok(None)` means no work is available now.
    async fn claim(&self, url: &Url) -> Result<Option<Vec<ClaimJob>>> {
        let resp = self.get(url.clone()).send().await.map_err(|e| {
            let error_text = e.to_string();
            let is_connect = e.is_connect();
            anyhow::Error::new(e).context(poll_request_context(
                url.as_str(),
                &error_text,
                is_connect,
            ))
        })?;
        if resp.status() == reqwest::StatusCode::NO_CONTENT {
            return Ok(None);
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "poll request returned non-success: url={url} status={status} body={}",
                body_excerpt(&body),
            );
        }
        let body = resp
            .text()
            .await
            .with_context(|| format!("read claim body: url={url}"))?;
        let claim: ClaimResponse = serde_json::from_str(&body).with_context(|| {
            format!(
                "parse claim response: url={url} body={}",
                body_excerpt(&body),
            )
        })?;
        Ok(Some(claim.jobs).filter(|jobs| !jobs.is_empty()))
    }

    /// Post a result, retrying transient failures. A posted verdict's fetched
    /// dependencies are then mirrored into hopper as their own samples.
    async fn post_result(&self, sha256: &str, post: Post) {
        let (verdict, amended) = match post {
            Post::Verdict(verdict) => (verdict, false),
            Post::Amended(verdict) => (verdict, true),
            Post::Failed(error) => {
                let payload = crate::upload::ResultPayload {
                    sha256: sha256.to_string(),
                    worker: self.worker.clone(),
                    error: Some(error),
                    duration_ms: 0,
                    envelope: None,
                };
                self.send_result(sha256, payload, None).await;
                return;
            }
        };
        let Verdict {
            envelope,
            deps,
            elapsed,
        } = verdict;
        // v7 envelope no longer carries `class` on the wire; the verdict is
        // encoded in `lvl` (clean = benign, anything else = hostile). The
        // suspicious band is consumer-side and not visible here.
        let class = if envelope.ml.level == crate::model::Level::Clean {
            "benign"
        } else {
            "hostile"
        };
        if amended {
            tracing::info!(sha256 = %sha256, verdict = class, "LLM amended the posted verdict");
        } else {
            tracing::info!(sha256 = %sha256, duration_ms = crate::duration_ms(elapsed), verdict = class, "analysis complete");
        }
        let deps = (!deps.is_empty()).then(|| {
            (
                deps,
                envelope.ml.version.clone(),
                envelope.ml.analyzed_at.clone(),
            )
        });
        let payload = crate::upload::ResultPayload {
            sha256: sha256.to_string(),
            worker: self.worker.clone(),
            error: None,
            duration_ms: i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX),
            envelope: Some(envelope),
        };
        self.send_result(sha256, payload, deps).await;
    }

    /// POST one result body to `/api/result`, retrying transient failures,
    /// then mirror `deps` once it is stored.
    async fn send_result(
        &self,
        sha256: &str,
        payload: crate::upload::ResultPayload,
        deps: Option<(Vec<crate::engine::DepResult>, String, String)>,
    ) {
        // Serialize and compress once, off the async threads (cleave reports
        // are large, repetitive JSON that zstd shrinks 3-5x), then reuse the
        // bytes across retries. Shared with the local `scan path --hopper`
        // uploader so both speak hopper's `/api/result` byte-identically.
        let encoded = {
            let sha256 = sha256.to_owned();
            tokio::task::spawn_blocking(move || crate::upload::encode_result_body(payload, &sha256))
                .await
        };
        // `None` means serialization failed unrecoverably; it is logged there.
        let Ok(Some((body, encoding))) = encoded else {
            return;
        };
        let body = bytes::Bytes::from(body);
        let url = self.url(["api", "result"]);

        // Retry with the same exponential-backoff-with-jitter schedule as poll
        // failures (2s, 4s, 8s, 16s, 32s, then capped at ~60s) for up to
        // RETRY_BUDGET. Hopper only re-leases a dropped result after its
        // 30-minute claim expiry, so a ~20-minute retry window recovers most
        // hopper restarts and short outages without forcing a full re-analysis
        // elsewhere. The post is idempotent on hopper, so re-sending after an
        // ambiguous timeout is safe. A shutdown aborts this tail mid-retry,
        // losing at most one result, which the lease recovers anyway.
        const RETRY_BUDGET: Duration = Duration::from_secs(20 * 60);
        let started = Instant::now();
        let mut attempt: u32 = 0;
        loop {
            if attempt > 0 {
                tokio::time::sleep(backoff_duration(attempt)).await;
            }
            tracing::debug!(sha256 = %sha256, attempt, "posting result to server");
            let post_start = Instant::now();
            let mut request = self
                .authed(self.client.post(url.clone()))
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body.clone());
            if let Some(enc) = encoding {
                request = request.header(reqwest::header::CONTENT_ENCODING, enc);
            }
            match request.send().await {
                Ok(resp) if resp.status().is_success() => {
                    tracing::debug!(sha256 = %sha256, elapsed_ms = crate::duration_ms(post_start.elapsed()), attempt, "result posted");
                    // The sample's row now exists on hopper; mirror its fetched
                    // dependencies (bytes if missing, provenance, and verdict)
                    // as their own samples. Best-effort, and never fails the
                    // result that preceded it.
                    if let Some((deps, version, analyzed_at)) = deps {
                        self.sync_dependencies(version, analyzed_at, deps).await;
                    }
                    return;
                }
                Ok(resp) => {
                    let status = resp.status();
                    let elapsed_ms = crate::duration_ms(post_start.elapsed());
                    let body = resp.text().await.unwrap_or_default();
                    // Resending a payload hopper refused for good can never
                    // succeed, so retrying would just burn 20 minutes.
                    if crate::upload::is_permanent(status) {
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
    /// samples, off the async threads. Each dependency's bytes come from the
    /// same blob cache the analysis fetched them into (uploaded only if hopper
    /// lacks them), paired with its provenance and the verdict scan already
    /// computed. Best-effort: failures are logged inside the sync.
    async fn sync_dependencies(
        &self,
        version: String,
        analyzed_at: String,
        deps: Vec<crate::engine::DepResult>,
    ) {
        let base = self.base.to_string();
        let worker = self.worker.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let Some(client) = crate::upload::hopper_http() else {
                return;
            };
            let cache = crate::fetch::open_blob_cache().ok();
            crate::upload::sync_result_dependencies(
                client,
                &base,
                &worker,
                &version,
                &analyzed_at,
                cache.as_ref(),
                deps,
            );
        })
        .await;
    }

    /// Download file bytes from hopper, verifying their SHA-256. Tries the
    /// fast `/data/{path}` endpoint first (static file serving, no DB query),
    /// falling back to `/api/file/{sha256}` for older hopper versions.
    async fn download_bytes(&self, sha256: &str, path: &str) -> Result<bytes::Bytes> {
        let start = Instant::now();
        let (resp, route) = self.download_response(sha256, path).await?;
        let url = resp.url().to_string();
        let bytes = resp.bytes().await.with_context(|| {
            format!("download body failed: path={path} sha256={sha256} url={url}")
        })?;
        // Up to half the staging buffer — gigabytes — so hashed off the async
        // threads.
        let digest: [u8; 32] = {
            let bytes = bytes.clone();
            tokio::task::spawn_blocking(move || Sha256::digest(&bytes).into())
                .await
                .context("hashing the download")?
        };
        verify_download_digest(digest, sha256, path, &url, route)?;
        tracing::info!(
            sha256 = %sha256,
            file = %path,
            bytes = bytes.len(),
            elapsed_ms = crate::duration_ms(start.elapsed()),
            "download complete via {route}",
        );
        Ok(bytes)
    }

    /// Stream a payload to a new spool file instead of buffering it in RAM, so
    /// a multi-GiB sample downloads with a constant memory footprint. Returns
    /// the temp path; the file is deleted when the path drops. The caller
    /// reserves and releases spool budget.
    async fn download_to_spool(
        &self,
        spool: &SpoolState,
        sha256: &str,
        path: &str,
    ) -> Result<tempfile::TempPath> {
        use tokio::io::AsyncWriteExt as _;

        // The digest becomes part of the spool filename below. `prefetch_one`
        // already rejects a malformed one, but this is the function that builds
        // the path, so it does not take that on trust — `tempfile` concatenates
        // `prefix` into the name verbatim, without rejecting path separators,
        // so an unchecked string here would be a traversal primitive out of the
        // spool directory. Checked before any I/O.
        if sha256_from_hex(sha256).is_none() {
            anyhow::bail!("refusing to spool under a malformed sha256: {sha256:?}");
        }

        let start = Instant::now();
        let (mut resp, route) = self.download_response(sha256, path).await?;
        let url = resp.url().to_string();

        // Re-create the spool dir if an OS temp sweep removed it since startup;
        // otherwise every large payload fails here for the life of the process.
        spool.ensure_dir()?;
        let temp = tempfile::Builder::new()
            .prefix(sha256.get(..16).unwrap_or(sha256))
            .tempfile_in(&spool.dir)
            .with_context(|| format!("cannot create spool file in {}", spool.dir.display()))?;
        let mut file = tokio::fs::File::from_std(
            temp.as_file()
                .try_clone()
                .context("cannot clone spool file handle")?,
        );

        let mut written: u64 = 0;
        let mut hasher = Sha256::new();
        while let Some(chunk) = resp.chunk().await.with_context(|| {
            format!("download body failed: path={path} sha256={sha256} url={url}")
        })? {
            written += chunk.len() as u64;
            if written > MAX_JOB_BYTES {
                anyhow::bail!(
                    "download exceeded per-job cap of {MAX_JOB_BYTES} bytes: path={path} sha256={sha256}",
                );
            }
            file.write_all(&chunk)
                .await
                .with_context(|| format!("spool write failed: sha256={sha256}"))?;
            hasher.update(&chunk);
        }
        file.flush()
            .await
            .with_context(|| format!("spool flush failed: sha256={sha256}"))?;
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

    /// Open a download stream for a sample, trying the cheap path-based
    /// endpoint first and falling back to the by-hash API. Returns the
    /// successful response (headers read, body not yet consumed) and the route
    /// label for logs.
    async fn download_response(
        &self,
        sha256: &str,
        path: &str,
    ) -> Result<(reqwest::Response, &'static str)> {
        if path.is_empty() || path == "." {
            anyhow::bail!("download {sha256}: empty path from hopper, cannot fetch");
        }

        // Path-based endpoint: static file serving, no DB query on hopper's
        // side. Each path segment is percent-encoded on its own.
        let data_url = self.url(std::iter::once("data").chain(path.split('/')));
        tracing::debug!(sha256 = %sha256, url = %data_url, "downloading via /data/");
        let resp = self.get(data_url.clone()).send().await.with_context(|| {
            format!("download failed: path={path} sha256={sha256} url={data_url}")
        })?;
        if resp.status().is_success() {
            return Ok((resp, "/data/"));
        }
        let data_status = resp.status();
        let data_body = resp
            .text()
            .await
            .map(|body| body_excerpt(&body))
            .unwrap_or_else(|e| format!("failed to read error body: {e}"));

        // /data/ failed — fall back to /api/file/{sha256}, which looks the
        // sample up by hash, so it works even when the relative path doesn't
        // match hopper's data root (a different symlink resolution or a data
        // root migration).
        let api_url = self.url(["api", "file", sha256]);
        tracing::debug!(sha256 = %sha256, url = %api_url, "downloading via /api/file/ (fallback)");
        let resp = self.get(api_url.clone()).send().await.with_context(|| {
            format!("download fallback failed: path={path} sha256={sha256} url={api_url}")
        })?;
        if !resp.status().is_success() {
            let api_status = resp.status();
            let api_body = resp
                .text()
                .await
                .map(|body| body_excerpt(&body))
                .unwrap_or_else(|e| format!("failed to read error body: {e}"));
            anyhow::bail!(
                "download failed: path={path} sha256={sha256}; /data/ url={data_url} status={data_status} body={data_body}; /api/file/ url={api_url} status={api_status} body={api_body}",
            );
        }
        Ok((resp, "/api/file/ (fallback)"))
    }

    /// Fetch the registry-metadata provenance hopper holds for `sha256`,
    /// preserving the complete JSON document alongside its normalized registry
    /// record. Best-effort by design: an absent record (HTTP 204), an
    /// unreachable hopper, or a malformed body all yield `None` — registry
    /// provenance enriches a scan but must never fail one, exactly as a live
    /// scan fails open when a registry lookup can't be made.
    async fn download_provenance(
        &self,
        sha256: &str,
    ) -> Option<crate::provenance::RegistryProvenance> {
        let url = self.url(["api", "provenance", sha256]);
        let resp = match self.get(url).send().await {
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
        // `resp.bytes()` already owns a refcounted buffer; move it into
        // provenance so the complete hopper document survives without another
        // full-size copy.
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

fn verify_download_digest(
    actual: [u8; 32],
    expected_hex: &str,
    path: &str,
    url: &str,
    route: &str,
) -> Result<()> {
    let Some(expected) = sha256_from_hex(expected_hex) else {
        anyhow::bail!(
            "download has invalid expected sha256: path={path} sha256={expected_hex} url={url}"
        );
    };
    if actual == expected {
        return Ok(());
    }
    let actual_hex = burton::hex(&actual);
    tracing::error!(
        expected_sha256 = %expected_hex,
        actual_sha256 = %actual_hex,
        file = %path,
        url = %url,
        route,
        "downloaded bytes failed sha256 verification"
    );
    anyhow::bail!(
        "download sha256 mismatch: path={path} expected={expected_hex} actual={actual_hex} url={url}"
    )
}

/// Exponential backoff for hopper outages: 1 s doubling to a 60 s cap, plus up
/// to a quarter more of random jitter, so a fleet that lost hopper together
/// does not reconnect in lockstep.
fn backoff_duration(consecutive_errors: u32) -> Duration {
    use std::hash::BuildHasher as _;
    let base = Duration::from_secs((1u64 << consecutive_errors.min(6)).min(60));
    let quarter_ms = u64::try_from(base.as_millis() / 4).unwrap_or(u64::MAX);
    // `RandomState` is keyed from the OS's randomness per process, so this is
    // a real draw without a dependency on `rand`.
    let draw = std::collections::hash_map::RandomState::new().hash_one(Instant::now());
    base + Duration::from_millis(draw % quarter_ms.saturating_add(1))
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
mod tests {

    #[test]
    fn idle_slots_are_never_a_stall() {
        // A worker waiting on an empty queue has nothing to complete; mistaking
        // that for a wedge would exit every idle worker on the fleet.
        assert_eq!(
            stall_verdict(0, secs(9999), secs(300), Some(secs(900))),
            StallVerdict::Progressing
        );
    }

    #[test]
    fn stall_warns_before_it_aborts() {
        // The shipped defaults: 15 minutes to warn, 30 to exit.
        let warn = secs(900);
        let abort = Some(secs(1800));
        assert_eq!(
            stall_verdict(4, secs(899), warn, abort),
            StallVerdict::Progressing
        );
        assert_eq!(
            stall_verdict(4, secs(900), warn, abort),
            StallVerdict::Stalled
        );
        assert_eq!(
            stall_verdict(4, secs(1799), warn, abort),
            StallVerdict::Stalled
        );
        assert_eq!(
            stall_verdict(4, secs(1800), warn, abort),
            StallVerdict::Abort
        );
    }

    #[test]
    fn zero_abort_threshold_disables_the_exit() {
        // The escape hatch: warn forever, never exit.
        assert_eq!(
            stall_verdict(4, secs(86_400), secs(900), None),
            StallVerdict::Stalled
        );
    }

    #[test]
    fn abort_can_never_fire_before_the_warn_it_escalates() {
        // An operator who sets the abort below the warn gets one Stalled tick
        // first, not a silent exit with no warning in the log above it.
        assert_eq!(
            stall_verdict(4, secs(10), secs(900), Some(secs(5))),
            StallVerdict::Progressing
        );
        assert_eq!(
            stall_verdict(4, secs(900), secs(900), Some(secs(5))),
            StallVerdict::Abort
        );
    }
    use super::*;
    use std::io::Write;

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    /// A worker configuration with no model behind it: enough for every task
    /// that never analyzes.
    fn test_config(base: &str) -> WorkerConfig {
        WorkerConfig {
            hopper_url: base.to_string(),
            name: "test-worker".to_string(),
            workers: NonZeroUsize::new(4).unwrap(),
            poll_interval: secs(1),
            max_rss: None,
            data_dir: None,
            max_jobs: None,
            exit_if_empty: false,
            renew_rules: false,
            nice: 0,
            rules: Rules {
                model_dir: PathBuf::new(),
                level: None,
                thresholds: None,
                slow_rule_ms: 4000,
                interpret: None,
                fetch: crate::fetch::FetchPolicy::default(),
                zip_passwords: crate::ArchivePasswords::default(),
            },
            tuning: WorkerTuning::default(),
        }
    }

    fn test_hopper(base: &str) -> Hopper {
        Hopper::new(base, "test-worker").unwrap()
    }

    /// Shared state for `config`, adjusted by `adjust` before it is shared.
    fn test_shared(
        config: &WorkerConfig,
        adjust: impl FnOnce(&mut WorkerShared),
    ) -> Arc<WorkerShared> {
        let mut shared = WorkerShared::new(config, test_hopper(&config.hopper_url));
        adjust(&mut shared);
        Arc::new(shared)
    }

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

        let index = LocalFileIndex::build(root.path().to_path_buf(), 4).expect("build index");
        let resolved = index
            .resolve(
                rel.to_str().expect("utf8 rel path"),
                &sha256_hex(bytes),
                Some(bytes.len() as u64),
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

        let index = LocalFileIndex::build(root.path().to_path_buf(), 4).expect("build index");
        assert_eq!(index.files.len(), expected.len());

        for (rel, bytes) in expected {
            let resolved = index
                .resolve(
                    rel.to_str().expect("utf8 rel path"),
                    &sha256_hex(&bytes),
                    Some(bytes.len() as u64),
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
        let size = Some(bytes.len() as u64);

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

        let index = LocalFileIndex::build(root.path().to_path_buf(), 4).expect("build index");

        assert_eq!(index.files.len(), 1);
        assert_eq!(index.files[0].path, root.path().join("real/sample.bin"));
    }

    #[test]
    fn local_index_resolves_by_final_dir_basename_and_size_for_absolute_requested_path() {
        let root = tempfile::tempdir().expect("create temp dir");
        let stored = root.path().join("bad/harvest/vxug/sample.bin");
        let bytes = b"sample-b";
        write_file(&stored, bytes);

        let index = LocalFileIndex::build(root.path().to_path_buf(), 4).expect("build index");
        let resolved = index
            .resolve(
                "/srv/home/t/data/bad/harvest/vxug/sample.bin",
                &sha256_hex(bytes),
                Some(bytes.len() as u64),
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

        let index = LocalFileIndex::build(root.path().to_path_buf(), 4).expect("build index");
        let resolved = index
            .resolve(
                "/srv/home/t/data/other/place/sample.txt",
                &sha256_hex(bytes),
                Some(bytes.len() as u64),
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

        let index = LocalFileIndex::build(root.path().to_path_buf(), 4).expect("build index");
        let resolved = index
            .resolve(
                "/srv/home/t/data/good/repos/sample.txt",
                &sha256_hex(b"87654321"),
                Some(bytes.len() as u64),
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
        let index = LocalFileIndex::build(root.path().to_path_buf(), 4).expect("build index");

        // Now add a file after the index was built.
        let rel = Path::new("unknown/harvest/new/crates/newpkg-1.0.crate");
        let bytes = b"newly-harvested-crate";
        write_file(&root.path().join(rel), bytes);

        let resolved = index
            .resolve(
                rel.to_str().expect("utf8"),
                &sha256_hex(bytes),
                Some(bytes.len() as u64),
            )
            .expect("resolve path");

        assert_eq!(resolved.as_deref(), Some(root.path().join(rel).as_path()));
    }

    /// The disk fallback should reject a file whose SHA-256 doesn't match,
    /// even when the path exists on disk.
    #[test]
    fn disk_fallback_rejects_sha_mismatch() {
        let root = tempfile::tempdir().expect("create temp dir");
        let index = LocalFileIndex::build(root.path().to_path_buf(), 4).expect("build index");

        let rel = Path::new("bad/malware/evil.bin");
        write_file(&root.path().join(rel), b"actual-content");

        let resolved = index
            .resolve(
                rel.to_str().expect("utf8"),
                &sha256_hex(b"different-content"),
                Some(14),
            )
            .expect("resolve path");

        assert!(resolved.is_none());
    }

    /// Absolute paths from the DB that exist on disk should be resolved
    /// via the disk fallback even when not in the index.
    #[test]
    fn disk_fallback_resolves_absolute_path_not_in_index() {
        let root = tempfile::tempdir().expect("create temp dir");
        let index = LocalFileIndex::build(root.path().to_path_buf(), 4).expect("build index");

        // Create a file outside the index root (simulates an absolute DB path).
        let external = tempfile::tempdir().expect("create external dir");
        let path = external.path().join("wolfi/pkg-1.0.apk");
        let bytes = b"absolute-path-file";
        write_file(&path, bytes);

        let resolved = index
            .resolve(
                path.to_str().expect("utf8"),
                &sha256_hex(bytes),
                Some(bytes.len() as u64),
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

        let provenance = test_hopper(&format!("http://{addr}"))
            .download_provenance("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
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
            tier: String::new(),
        }
    }

    #[tokio::test]
    async fn prefetch_one_rejects_jobs_over_max_job_bytes() {
        // Rejected before any download, so the unreachable base_url is never hit.
        let job = claim_job(
            &sha256_hex(b"huge"),
            "samples/huge.bin",
            (MAX_JOB_BYTES + 1) as i64,
        );
        let pj = prefetch_one(
            &test_hopper("http://127.0.0.1:1"),
            None,
            &test_spool(1 << 20),
            job,
        )
        .await;
        match pj.data {
            Err(PrefetchError::Refused(refusal)) => {
                let msg = refusal.to_string();
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
                // Unreachable: a malformed digest must be refused before any I/O.
                &test_hopper("http://127.0.0.1:1"),
                None,
                &test_spool(1 << 20),
                job,
            )
            .await;
            match pj.data {
                Err(PrefetchError::Refused(refusal)) => {
                    let msg = refusal.to_string();
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
        let err = test_hopper("http://127.0.0.1:1")
            .download_to_spool(&spool, "../../escape", "samples/x.bin")
            .await
            .expect_err("a malformed digest must not name a spool file")
            .to_string();
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
            &test_hopper("http://127.0.0.1:1"),
            Some(dir.path()),
            &test_spool(1 << 20),
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
            &test_hopper(&format!("http://127.0.0.1:{port}")),
            None,
            &spool,
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
        let mut config = test_config(&format!("http://127.0.0.1:{port}"));
        config.exit_if_empty = exit_if_empty;
        // Far below the 1 s poll cadence, so the second empty poll trips it.
        config.tuning.idle_warn_after = Duration::from_millis(10);
        let shared = test_shared(&config, |shared| {
            shared.spool = test_spool(1 << 20);
            shared.target_depth = 8;
            shared.max_buffer_bytes = 1 << 30;
        });
        let handle = tokio::spawn(prefetch_loop(Arc::clone(&shared), tx));

        let found = wait_until(|| captured.contains(done)).await;
        shared.stop.raise(Exit::Finished);
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
            &test_hopper(&format!("http://127.0.0.1:{port}")),
            None,
            &spool,
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
        spool
            .try_reserve(1024)
            .expect("reserve should heal the dir");
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
        let err = test_hopper(&format!("http://{addr}"))
            .download_bytes(&expected, "incoming/sample.bin")
            .await
            .expect_err("mismatched bytes must fail")
            .to_string();
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
        let err = test_hopper(&format!("http://{addr}"))
            .download_to_spool(&test_spool(0), &expected, "incoming/sample.bin")
            .await
            .expect_err("mismatched spooled bytes must fail")
            .to_string();
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
        let mut config = test_config(&format!("http://127.0.0.1:{port}"));
        config.workers = NonZeroUsize::new(slots).unwrap();
        let shared = test_shared(&config, |shared| {
            shared.spool = test_spool(1 << 20);
            shared.target_depth = target_depth;
            shared.max_buffer_bytes = 1 << 30;
        });
        let handle = tokio::spawn(prefetch_loop(Arc::clone(&shared), tx));

        // 1. Fills to exactly target_depth — every staged sample lands in the
        //    channel — and never overshoots the cap.
        assert!(
            wait_until(|| rx.len() == target_depth).await,
            "prefetcher did not fill to target_depth; channel len {}",
            rx.len(),
        );
        assert_eq!(shared.outstanding.load(Ordering::Relaxed), target_depth);

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
            shared.unstage(&pj);
            match pj.data.unwrap() {
                PrefetchData::Memory(bytes) => assert_eq!(&bytes[..], PAYLOAD, "payload mismatch"),
                PrefetchData::Local | PrefetchData::Spooled(_) => {
                    panic!("small payload should stage in memory")
                }
            }
        }
        assert!(
            wait_until(|| shared.outstanding.load(Ordering::Relaxed) == target_depth).await,
            "prefetcher did not refill after draining",
        );
        assert!(
            api_calls.load(Ordering::Relaxed) > calls_when_full,
            "prefetcher should have polled again to refill",
        );

        // 4. Stop ends the prefetcher and closes the channel.
        shared.stop.raise(Exit::Finished);
        assert!(
            tokio::time::timeout(Duration::from_secs(5), handle)
                .await
                .is_ok(),
            "prefetcher did not exit on shutdown",
        );
        while rx.recv().await.is_some() {}
    }

    fn staged_pj(sha: &str, size_bytes: i64) -> PrefetchedJob {
        staged_pj_tier(sha, size_bytes, "")
    }

    fn staged_pj_tier(sha: &str, size_bytes: i64, tier: &str) -> PrefetchedJob {
        PrefetchedJob {
            job: ClaimJob {
                sha256: sha.to_string(),
                path: format!("{sha}.bin"),
                size_bytes,
                file_type: "data".to_string(),
                has_provenance: false,
                tier: tier.to_string(),
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

    fn smallest_first(rx: mpsc::UnboundedReceiver<PrefetchedJob>) -> JobSource {
        JobSource::new(
            rx,
            DispatchOrder::Smallest,
            WorkerTuning::default().sjf_max_wait,
        )
    }

    async fn next_sha(jobs: &JobSource) -> String {
        jobs.recv().await.unwrap().job.sha256
    }

    #[tokio::test]
    async fn sjf_picks_smallest_staged_job_first() {
        let (tx, rx) = mpsc::unbounded_channel::<PrefetchedJob>();
        let jobs = smallest_first(rx);
        tx.send(staged_pj("big", 500 * 1024 * 1024)).unwrap();
        tx.send(staged_pj("tiny", 4 * 1024)).unwrap();
        tx.send(staged_pj("mid", 8 * 1024 * 1024)).unwrap();

        let order = [
            next_sha(&jobs).await,
            next_sha(&jobs).await,
            next_sha(&jobs).await,
        ];
        assert_eq!(order, ["tiny", "mid", "big"]);

        // Channel closed and window drained → None, like recv().
        drop(tx);
        assert!(jobs.recv().await.is_none());
    }

    #[tokio::test]
    async fn sjf_drains_reorder_window_after_channel_close() {
        // Jobs staged in the window must still dispatch after the prefetcher
        // exits, or --exit-if-empty would drop the tail of the queue.
        let (tx, rx) = mpsc::unbounded_channel::<PrefetchedJob>();
        let jobs = smallest_first(rx);
        tx.send(staged_pj("a", 100)).unwrap();
        tx.send(staged_pj("b", 50)).unwrap();
        drop(tx);

        assert_eq!(next_sha(&jobs).await, "b");
        assert_eq!(next_sha(&jobs).await, "a");
        assert!(jobs.recv().await.is_none());
    }

    /// Hopper decides a sighted sample goes first; the worker must not undo that
    /// by sorting it back down on size. A 12.4 MB sighted tarball sorted last
    /// under smallest-first on 2026-09-08 and waited out the 900s aging bound.
    #[tokio::test]
    async fn sighted_job_dispatches_before_smaller_ordinary_work() {
        let mut reorder = vec![
            (staged_pj("small", 10), Instant::now()),
            (
                staged_pj_tier("sighted-big", 12_400_000, TIER_SIGHTED),
                Instant::now(),
            ),
            (staged_pj("tiny", 1), Instant::now()),
        ];
        let got = pick_from_reorder(
            &mut reorder,
            DispatchOrder::Smallest,
            WorkerTuning::default().sjf_max_wait,
        )
        .unwrap();
        assert_eq!(
            got.job.sha256, "sighted-big",
            "a sighted job must outrank the size sort"
        );
        // The rest keeps smallest-first.
        let next = pick_from_reorder(
            &mut reorder,
            DispatchOrder::Smallest,
            WorkerTuning::default().sjf_max_wait,
        )
        .unwrap();
        assert_eq!(next.job.sha256, "tiny");
    }

    /// Two sighted jobs go oldest first, so the tier cannot starve within itself.
    #[tokio::test]
    async fn sighted_jobs_dispatch_oldest_first() {
        let older = Instant::now() - Duration::from_secs(60);
        let mut reorder = vec![
            (staged_pj_tier("newer", 10, TIER_SIGHTED), Instant::now()),
            (staged_pj_tier("older", 9_000_000, TIER_SIGHTED), older),
        ];
        let got = pick_from_reorder(
            &mut reorder,
            DispatchOrder::Smallest,
            WorkerTuning::default().sjf_max_wait,
        )
        .unwrap();
        assert_eq!(got.job.sha256, "older");
    }

    /// An unknown or absent tier is ordinary work, so an older hopper that sends
    /// no tier field behaves exactly as before.
    #[tokio::test]
    async fn untagged_jobs_keep_smallest_first() {
        let mut reorder = vec![
            (staged_pj("big", 5_000_000), Instant::now()),
            (staged_pj("small", 5), Instant::now()),
        ];
        let got = pick_from_reorder(
            &mut reorder,
            DispatchOrder::Smallest,
            WorkerTuning::default().sjf_max_wait,
        )
        .unwrap();
        assert_eq!(got.job.sha256, "small");
    }

    #[tokio::test]
    async fn sjf_ages_long_waiting_job_to_front() {
        let (tx, rx) = mpsc::unbounded_channel::<PrefetchedJob>();
        let jobs = smallest_first(rx);
        // A big job already staged longer than the aging bound beats a fresh
        // tiny job, so SJF cannot starve archives indefinitely.
        jobs.state.lock().await.reorder.push((
            staged_pj("old-big", 500 * 1024 * 1024),
            Instant::now() - jobs.max_wait,
        ));
        tx.send(staged_pj("fresh-tiny", 4 * 1024)).unwrap();

        assert_eq!(next_sha(&jobs).await, "old-big");
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
        assert_eq!(small_lane_from(16, None), 16);
        assert_eq!(small_lane_from(128, None), 64);
        assert_eq!(small_lane_from(2, None), 2);
        assert_eq!(small_lane_from(16, Some(2)), 2);
        assert_eq!(small_lane_from(16, Some(0)), 16);
        let gate = CleaveGate::new(1, 4, 1024 * 1024);
        assert!(gate.is_small(1024 * 1024));
        assert!(!gate.is_small(1024 * 1024 + 1));
        assert!(!CleaveGate::new(1, 4, 0).is_small(1), "0 disables the lane");
        assert_eq!(cleave_concurrency_from(16, 32, None), 16);
        assert_eq!(cleave_concurrency_from(16, 16, None), 16);
        assert_eq!(cleave_concurrency_from(16, 8, None), 8);
        assert_eq!(cleave_concurrency_from(2, 64, None), 2);
        assert_eq!(cleave_concurrency_from(0, 32, None), 1);
    }

    #[test]
    fn tail_cap_is_twice_the_slots_and_never_below_them() {
        assert_eq!(tail_cap_from(48, None), 96);
        assert_eq!(tail_cap_from(1, None), 2);
        assert_eq!(tail_cap_from(0, None), 2);
        assert_eq!(tail_cap_from(48, Some(200)), 200);
        // An override below the slot count would leave slots waiting on tails.
        assert_eq!(tail_cap_from(48, Some(10)), 48);
        assert_eq!(tail_cap_from(48, Some(0)), 96);
    }

    #[test]
    fn cleave_concurrency_override_caps_at_slots() {
        assert_eq!(cleave_concurrency_from(4, 32, Some(64)), 4);
        assert_eq!(cleave_concurrency_from(16, 32, Some(2)), 2);
        // Zero / bogus overrides fall back to the pool formula.
        assert_eq!(cleave_concurrency_from(16, 32, Some(0)), 16);
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
        let jobs = Arc::new(JobSource::new(
            rx,
            DispatchOrder::Fifo,
            WorkerTuning::default().sjf_max_wait,
        ));
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

        let jobs = Arc::new(JobSource::new(
            rx,
            DispatchOrder::Fifo,
            WorkerTuning::default().sjf_max_wait,
        ));
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

        let jobs = Arc::new(JobSource::new(
            rx,
            DispatchOrder::Fifo,
            WorkerTuning::default().sjf_max_wait,
        ));
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

    #[test]
    fn worker_diagnostics_redact_secrets() {
        let args = [
            "atomscan",
            "worker",
            "--zip-password",
            "secret one",
            "--zip-password=secret-two",
            "--verbose",
            "--llm-key",
            "sk-one",
            "--llm-key=sk-two",
        ]
        .map(str::to_string);

        assert_eq!(
            redact_secrets(args),
            [
                "atomscan",
                "worker",
                "--zip-password",
                "<redacted>",
                "--zip-password=<redacted>",
                "--verbose",
                "--llm-key",
                "<redacted>",
                "--llm-key=<redacted>",
            ]
        );
    }

    /// The reservation is one compare-and-swap: concurrent downloads must never
    /// reserve more than the budget between them. A load-then-add let two
    /// racing reservations both see room only one of them had.
    #[test]
    fn concurrent_spool_reservations_never_exceed_the_budget() {
        let parent = tempfile::tempdir().unwrap();
        for _ in 0..50 {
            let spool = SpoolState {
                dir: parent.path().to_path_buf(),
                budget_bytes: 100,
                used: AtomicU64::new(0),
                mem_threshold_bytes: 0,
                disk_headroom_bytes: 0,
            };
            let barrier = std::sync::Barrier::new(16);
            let admitted = AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..16 {
                    scope.spawn(|| {
                        barrier.wait();
                        if spool.try_reserve(30).is_ok() {
                            admitted.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
            });
            let admitted = admitted.load(Ordering::Relaxed);
            assert_eq!(
                admitted, 3,
                "a 100-byte budget holds three 30-byte payloads"
            );
            assert_eq!(spool.used.load(Ordering::Acquire), 90);
        }
    }

    #[test]
    fn backoff_doubles_to_a_cap_and_jitters_within_a_quarter() {
        for (errors, base) in [(1, 2), (2, 4), (5, 32), (6, 60), (40, 60)] {
            for _ in 0..20 {
                let d = backoff_duration(errors);
                assert!(d >= secs(base), "{errors} errors: {d:?} below {base}s");
                assert!(
                    d <= secs(base) + secs(base) / 4,
                    "{errors} errors: {d:?} above {base}s + 25%"
                );
            }
        }
        // Real jitter: a fleet that failed together must not retry together.
        let draws: HashSet<Duration> = (0..20).map(|_| backoff_duration(6)).collect();
        assert!(draws.len() > 1, "backoff jitter is not random: {draws:?}");
    }

    #[test]
    fn dispatch_order_parses_the_scan_sjf_values_and_rejects_others() {
        assert_eq!("0".parse::<DispatchOrder>(), Ok(DispatchOrder::Fifo));
        assert_eq!("1".parse::<DispatchOrder>(), Ok(DispatchOrder::Smallest));
        assert_eq!("big".parse::<DispatchOrder>(), Ok(DispatchOrder::Largest));
        assert!("smallest".parse::<DispatchOrder>().is_err());
    }

    #[test]
    fn a_stall_exits_tempfail_and_a_finish_exits_clean() {
        assert_eq!(Exit::Finished.code(), 0);
        assert_eq!(Exit::Stalled.code(), 75);
    }

    #[test]
    fn the_default_analysis_deadline_is_the_servers() {
        assert_eq!(
            WorkerTuning::default().analysis_timeout,
            Some(secs(crate::server::DEFAULT_ANALYSIS_TIMEOUT_SECS))
        );
    }

    /// Each sample-path segment is encoded on its own, so a path can neither
    /// climb out of `/data/` nor smuggle a query into the URL.
    #[test]
    fn data_urls_encode_each_path_segment() {
        let hopper = test_hopper("http://hopper.test:8081/");
        let url = hopper.url(std::iter::once("data").chain("a b/../c?d#e/f.bin".split('/')));
        assert_eq!(
            url.as_str(),
            "http://hopper.test:8081/data/a%20b/c%3Fd%23e/f.bin"
        );
        let prefixed = test_hopper("https://hops.example/hopper/");
        assert_eq!(
            prefixed.url(["api", "next"]).as_str(),
            "https://hops.example/hopper/api/next"
        );
    }

    #[test]
    fn poll_and_heartbeat_urls_carry_the_routing_signals() {
        let hopper = test_hopper("http://hopper.test");
        let poll = hopper.poll_url(3, 8);
        assert_eq!(poll.path(), "/api/next");
        let keys: HashSet<String> = poll.query_pairs().map(|(k, _)| k.into_owned()).collect();
        for key in ["count", "slots", "max_bytes", "worker", "version", "tools"] {
            assert!(keys.contains(key), "poll URL lacks {key}: {poll}");
        }
        assert!(poll.query_pairs().any(|(k, v)| k == "count" && v == "3"));

        let report = HeartbeatReport {
            slots: 8,
            active: 2,
            queue: 5,
            mem_reserved_mb: 1,
            mem_ceiling_mb: 2,
            poll_age: secs(3),
            last_want: 4,
            last_claim: 0,
            buffer_room: 6,
            active_shas: vec![Arc::from("aa"), Arc::from("bb")],
            metrics: MetricsSnapshot {
                oldest_age: Some(secs(9)),
                last_completion_age: None,
                files_per_sec: 0.5,
                errors_recent: 1,
                last_error: Some((secs(7), "boom & bust".to_string())),
            },
        };
        let beat = hopper.heartbeat_url(&report);
        assert_eq!(beat.path(), "/api/heartbeat");
        let pairs: HashMap<String, String> = beat
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        assert_eq!(pairs["active_shas"], "aa,bb");
        assert_eq!(pairs["err"], "boom & bust");
        assert_eq!(pairs["poll_age_s"], "3");
        assert_eq!(pairs["worker"], "test-worker");
    }

    #[tokio::test]
    async fn stop_wakes_every_waiter_without_polling() {
        let stop = Stop::new();
        let waiter = {
            let stop = stop.clone();
            tokio::spawn(async move { stop.sleep(secs(3600)).await })
        };
        tokio::task::yield_now().await;
        stop.raise(Exit::Finished);
        let raised = tokio::time::timeout(secs(5), waiter)
            .await
            .expect("a raised stop must end the sleep at once")
            .unwrap();
        assert!(raised);
        assert!(
            stop.sleep(secs(3600)).await,
            "an already-raised stop returns at once"
        );
    }

    /// A stall must cut short a drain already under way (a batch run waits on
    /// its drain forever otherwise), and a later plain stop cannot undo it.
    #[tokio::test]
    async fn a_stall_overrides_a_stop_and_not_the_reverse() {
        let stop = Stop::new();
        assert_eq!(stop.exit(), Exit::Finished);
        stop.raise(Exit::Finished);
        assert!(stop.is_raised());
        let stalled = {
            let stop = stop.clone();
            tokio::spawn(async move { stop.stalled().await })
        };
        tokio::task::yield_now().await;
        assert!(!stalled.is_finished(), "a plain stop is not a stall");
        stop.raise(Exit::Stalled);
        tokio::time::timeout(secs(5), stalled)
            .await
            .expect("a stall must wake the drain")
            .unwrap();
        stop.raise(Exit::Finished);
        assert_eq!(stop.exit(), Exit::Stalled);
    }

    /// Aborting a tail drops its `run_job`, and that must cancel the analysis
    /// running on its blocking thread — the drain's cancellation path.
    #[tokio::test]
    async fn an_aborted_job_cancels_its_analysis() {
        let cancel = Arc::new(AtomicBool::new(false));
        let task = {
            let guard = CancelOnDrop(Arc::clone(&cancel));
            tokio::spawn(async move {
                let _guard = guard;
                std::future::pending::<()>().await;
            })
        };
        tokio::task::yield_now().await;
        assert!(!cancel.load(Ordering::Acquire));
        task.abort();
        let _ = task.await;
        assert!(
            cancel.load(Ordering::Acquire),
            "abort must raise the cancel flag"
        );
    }
}
