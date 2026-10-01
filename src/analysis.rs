//! One analysis of a file or byte buffer against a loaded model bundle: the
//! pipeline the server and the pull worker share.
//!
//! [`classify_file`] and [`classify_bytes`] run cleave and then the model, and
//! return the wire [`ScanResult`]. [`ModelResources`] is the loaded bundle,
//! [`RequestPhase`] the per-request stage tracker, and the whale lanes decide
//! which rayon pool an analysis runs on.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};

use crate::engine::{ClassifyRequest, CpuLease, OutputNeeds, ScanResult};
use crate::explain::ShapImportance;
use crate::fetch::FetchPolicy;
use crate::model::Model;
use crate::provenance::RegistryProvenance;

/// One analysis: what to label it, which bundle to run, and the per-request
/// context the caller tracks it with. Build with [`Analysis::new`] and set the
/// rest with struct update syntax.
pub(crate) struct Analysis<'a> {
    /// The result's `path`: the upload's name, not a temp file path.
    pub(crate) label: &'a str,
    pub(crate) resources: &'a ModelResources,
    pub(crate) slow_rule_ms: u64,
    pub(crate) cancellation: Option<&'a Arc<AtomicBool>>,
    pub(crate) phase: Option<&'a RequestPhase>,
    /// Registry facts collected at fetch time (worker provenance or a
    /// caller's registry map), so the scan reasons over them without refetching.
    pub(crate) root_registry: Option<&'a RegistryProvenance>,
    /// Reference-following policy; [`Analysis::new`] takes the bundle's.
    pub(crate) follow: FetchPolicy,
    /// Capture each fetched dependency's standalone report, for callers that
    /// renew results on hopper.
    pub(crate) deps_for_upload: bool,
    pub(crate) cpu_lease: Option<CpuLease>,
}

impl<'a> Analysis<'a> {
    pub(crate) fn new(label: &'a str, resources: &'a ModelResources, slow_rule_ms: u64) -> Self {
        Self {
            label,
            resources,
            slow_rule_ms,
            cancellation: None,
            phase: None,
            root_registry: None,
            follow: resources.fetch,
            deps_for_upload: false,
            cpu_lease: None,
        }
    }

    fn mark(&self, phase: &str) {
        if let Some(p) = self.phase {
            p.set(phase);
        }
    }

    fn options(&self) -> cleave::AnalysisOptions {
        // Part of cleave's analysis cache key, not only a retention setting:
        // every daemon analysis must set it the same way, or the first one
        // caches under a different key than the rest.
        cleave::set_compact_member_retention(true);
        let mut opts = cleave::AnalysisOptions {
            slow_rule_ms: self.slow_rule_ms,
            cancellation: self.cancellation.cloned(),
            phase: self.phase.map(RequestPhase::tracker).cloned(),
            ..Default::default()
        };
        crate::engine::add_zip_passwords(&mut opts, self.resources.zip_passwords.as_slice());
        opts
    }
}

/// Analyze the file at `path`. Runs on a blocking thread. `extract_dir`, when
/// set, receives the samples cleave extracts.
pub(crate) fn classify_file(
    path: &Path,
    extract_dir: Option<&Path>,
    analysis: Analysis<'_>,
) -> Result<ScanResult> {
    analysis.mark("cleave:init");
    let mut opts = analysis.options();
    opts.sample_extraction =
        extract_dir.map(|d| cleave::SampleExtractionConfig::new(d.to_path_buf()));
    analysis.mark("cleave:analyze");
    let report = cleave::analyze_file(path, &opts)
        .with_context(|| format!("cleave analysis of {}", analysis.label))?;
    finish(report, analysis)
}

/// Analyze `data` in memory. cleave adopts the refcounted buffer, so a
/// downloaded sample is never copied.
pub(crate) fn classify_bytes(data: bytes::Bytes, analysis: Analysis<'_>) -> Result<ScanResult> {
    analysis.mark("cleave:init");
    let mut opts = analysis.options();
    // A whale analyzes on its own bounded pool so its members never fill the
    // global pool's deques and starve the small analyses that arrive while it
    // runs (see `whale_lane_for`). `install` blocks this thread until the
    // analysis returns, exactly as an inline call would.
    analysis.mark("whale:lane");
    let lane = whale_lane_for(data.len() as u64)?;
    // On its own pool the analysis is exempt from cleave's shared-pool
    // throttles; see `AnalysisOptions::dedicated_pool`.
    opts.dedicated_pool = lane.is_some();
    analysis.mark("cleave:analyze");
    let label = analysis.label;
    let analyze = move || cleave::analyze_bytes_shared(data, label, &opts);
    let report = match lane {
        Some(lane) => lane.install(analyze),
        None => analyze(),
    }
    .with_context(|| format!("cleave analysis of {label}"))?;
    finish(report, analysis)
}

/// Shared tail: honor a late cancellation, then classify and build the result.
fn finish(report: cleave::AnalysisReport, analysis: Analysis<'_>) -> Result<ScanResult> {
    // A timeout that fired while cleave ran: skip featurization and inference
    // for a result nobody is waiting for.
    if analysis
        .cancellation
        .is_some_and(|c| c.load(Ordering::Relaxed))
    {
        anyhow::bail!("analysis cancelled");
    }
    analysis.mark("classify:report");
    let r = analysis.resources;
    let cr = crate::engine::classify_report(
        report,
        ClassifyRequest {
            shap: r.shap.as_ref(),
            cancellation: analysis.cancellation.map(Arc::as_ref),
            interpret: r.interpret.as_ref(),
            fetch: analysis.follow,
            zip_passwords: r.zip_passwords.as_slice(),
            // The JSON envelope only: no renders, fetch log, or manifest.
            needs: OutputNeeds {
                deps_for_upload: analysis.deps_for_upload,
                ..OutputNeeds::default()
            },
            root_registry: analysis.root_registry,
            phase: analysis.phase.map(RequestPhase::tracker),
            cpu_lease: analysis.cpu_lease,
            // There is no file to re-read for the root imperative hunt; the
            // label is a best-effort path. Declared references still follow.
            ..ClassifyRequest::new(analysis.label, Path::new(analysis.label), &r.model)
        },
    )?;
    Ok(cr.into_scan_result(analysis.label.to_string(), &r.model, true))
}

#[derive(Debug)]
/// A request's phase marker plus the timeline it leaves behind.
///
/// Wraps the [`cleave::PhaseTracker`] that `/_/requests` and the watchdog
/// read, and records when each phase began so the completion log can say
/// where a request's wall time went (`phases="purl:fetch=812 cleave:init=1930 …"`,
/// milliseconds per phase, repeats summed). That is what turns a latency
/// percentile into an attribution: fetch-bound, registry-bound or
/// analysis-bound, per request, without a profiler attached.
#[derive(Clone)]
pub(crate) struct RequestPhase(Arc<RequestPhaseInner>);

#[derive(Debug)]
struct RequestPhaseInner {
    tracker: cleave::PhaseTracker,
    started: Instant,
    /// `(phase, offset from `started`)` in the order the phases were entered.
    marks: Mutex<Vec<(String, Duration)>>,
}

/// More marks than this and the request is looping, not progressing; keep the
/// head so the timeline still shows how it started.
const PHASE_MARKS_MAX: usize = 64;

impl RequestPhase {
    pub(crate) fn with_label(label: impl Into<String>) -> Self {
        Self(Arc::new(RequestPhaseInner {
            tracker: cleave::PhaseTracker::with_label(label),
            started: Instant::now(),
            marks: Mutex::new(Vec::new()),
        }))
    }

    /// Enter `phase`: updates the shared tracker and stamps the timeline.
    pub(crate) fn set(&self, phase: &str) {
        self.0.tracker.set(phase);
        let at = self.0.started.elapsed();
        let mut marks = self
            .0
            .marks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if marks.len() < PHASE_MARKS_MAX && marks.last().is_none_or(|(last, _)| last != phase) {
            marks.push((phase.to_owned(), at));
        }
    }

    /// The current phase name, as the tracker reports it.
    pub(crate) fn get(&self) -> String {
        self.0.tracker.get()
    }

    /// The underlying tracker, for cleave's own phase reporting.
    pub(crate) fn tracker(&self) -> &cleave::PhaseTracker {
        &self.0.tracker
    }

    /// Milliseconds spent in each phase, first-entered order, as
    /// `name=ms name=ms …`. Time before the first mark is `pre`, so the
    /// figures sum to the request's elapsed time.
    pub(crate) fn timeline(&self) -> String {
        let now = self.0.started.elapsed();
        let marks = self
            .0
            .marks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let mut order: Vec<String> = Vec::new();
        let mut spent: std::collections::HashMap<String, u128> = std::collections::HashMap::new();
        let mut account = |name: &str, dur: Duration| {
            if !spent.contains_key(name) {
                order.push(name.to_owned());
            }
            *spent.entry(name.to_owned()).or_insert(0) += dur.as_millis();
        };
        if let Some((_, first)) = marks.first()
            && !first.is_zero()
        {
            account("pre", *first);
        }
        for (i, (name, at)) in marks.iter().enumerate() {
            let end = marks.get(i + 1).map_or(now, |(_, next)| *next);
            account(name, end.saturating_sub(*at));
        }
        order
            .iter()
            .map(|name| format!("{name}={}", spent.get(name).copied().unwrap_or(0)))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[derive(Debug)]
/// The loaded model bundle an analysis runs against: thresholds, the ML
/// ensemble, and the optional LLM and fetch policies attached to it.
///
/// Shared by the server and the pull worker, each of which holds one behind an
/// `Arc` and swaps it on reload.
pub(crate) struct ModelResources {
    pub(crate) model: Model,
    pub(crate) shap: Option<ShapImportance>,
    /// LLM interpretation config (`--interpret`); `None` disables the pass.
    pub(crate) interpret: Option<crate::interpret::InterpretConfig>,
    /// External-reference fetch policy; default (empty) disables fetching.
    pub(crate) fetch: crate::fetch::FetchPolicy,
    /// Additional passwords to try for encrypted archives.
    pub(crate) zip_passwords: crate::ArchivePasswords,
}

/// Payloads at or below this many bytes count as small — for the slot lanes
/// and for [`whale_lane_for`] alike. `SCAN_SMALL_JOB_MB`, the same knob the
/// worker's cleave gate reads; 1 MiB unless set.
pub(crate) fn small_job_max_bytes() -> u64 {
    std::env::var("SCAN_SMALL_JOB_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(1024 * 1024, |mb| mb.saturating_mul(1024 * 1024))
}

/// Threads for one whale's private pool on a host with this many physical
/// cores: a quarter of them, never fewer than two (one thread cannot pipeline
/// a member window against its own producer) and never more than sixteen.
/// 4 cores → 2, 64 → 16, 128 → 16. Several whales in flight each get their
/// own; the global pool keeps every core for everything else.
#[must_use]
pub(crate) fn whale_pool_threads(physical_cores: usize) -> usize {
    (physical_cores / 4).clamp(2, 16)
}

/// Threads for a small payload's private pool on a host with this many
/// physical cores: an eighth of them, 2–8. 4 cores → 2, 64 → 8, 128 → 8. A
/// package under the small cap has few members, so a wider pool mostly idles;
/// eight is where its p50 was best (0.29 s vs 0.44 s on 16, measured
/// 2026-09-05 at concurrency 8 over 256 PURLs).
#[must_use]
pub(crate) fn small_pool_threads(physical_cores: usize) -> usize {
    (physical_cores / 8).clamp(2, 8)
}

/// How many *big* whales analyze at once on a host with this many physical
/// cores; the rest wait their turn. An eighth of the cores, 1–8: 4 cores → 1,
/// 64 → 8, 128 → 8. With [`whale_pool_threads`] that bounds whale threads at
/// about two per physical core, oversubscribed enough to keep every core busy
/// while a whale is in a serial phase, not so much that the global pool's
/// small work loses its cores.
#[must_use]
pub(crate) fn whale_slots(physical_cores: usize) -> usize {
    (physical_cores / 8).clamp(1, 8)
}

/// Payloads above this many bytes are *big* whales, the ones that take a
/// slot from [`whale_slots`]. `SCAN_BIG_JOB_MB`; 8 MiB unless set. Between
/// `SCAN_SMALL_JOB_MB` and this a payload still gets a private pool, but
/// never waits: a 2 MiB package finishes in seconds, and making it queue
/// behind a 250 MiB one is the starvation this whole arrangement exists to
/// prevent.
fn big_job_min_bytes() -> u64 {
    std::env::var("SCAN_BIG_JOB_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(8 * 1024 * 1024, |mb| mb.saturating_mul(1024 * 1024))
}

/// Whale-lane sizing, read once from the environment.
/// Analyses admitted and not yet finished, process-wide — what a new lane
/// shares the cores with. Kept by [`RequestGuard`].
pub(crate) static ACTIVE_REQUESTS: AtomicUsize = AtomicUsize::new(0);

/// Threads a private lane of tier minimum `floor` gets when `active`
/// requests (this one included) are in flight on a host with `physical`
/// cores: an even share of the cores, never below the tier's floor and never
/// above the cores. Alone (`active <= 1`) the answer is `None`: the global
/// pool, every thread of it.
///
/// The tier floors (8 for small, 16 for whales) were chosen at concurrency 8
/// and are right there — 8 for small beat 16 on p50 — but they are the whole
/// story only when the box is full. At concurrency 1 the same 8-thread pool
/// left 120 threads idle: purls-128 measured p90 1.63 s on the fixed widths
/// against 1.12 s on the global pool, wall 164 → 111 s (2026-09-06). The
/// share reproduces both ends: 64 physical cores / 8 in flight = 8 (the
/// small floor), / 4 = 16, / 2 = 32, alone = everything.
#[must_use]
pub(crate) fn lane_threads(floor: usize, physical: usize, active: usize) -> Option<usize> {
    if active <= 1 {
        return None;
    }
    Some((physical / active).clamp(floor, physical.max(floor)))
}

struct WhaleConfig {
    /// Threads per whale pool; `0` sends whales to the global pool instead.
    threads: usize,
    /// Threads per small payload's pool; `0` keeps small payloads on the
    /// global pool.
    small_threads: usize,
    /// Physical cores, the numerator of the per-request share.
    physical: usize,
    /// `SCAN_LANE_SHARE=0` pins every lane at its tier floor (the widths
    /// measured at concurrency 8) instead of sharing the cores by load.
    share: bool,
    /// Big whales in flight at once.
    slots: usize,
    small_max_bytes: u64,
    big_min_bytes: u64,
}

static WHALE_CONFIG: OnceLock<WhaleConfig> = OnceLock::new();

fn whale_config() -> &'static WhaleConfig {
    WHALE_CONFIG.get_or_init(|| {
        let physical = cleave::memory_tracker::physical_cpu_count()
            .or_else(|| {
                std::thread::available_parallelism()
                    .ok()
                    .map(|n| n.get() / 2)
            })
            .unwrap_or(4);
        let env_usize = |key: &str| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
        };
        let threads =
            env_usize("SCAN_WHALE_POOL_THREADS").unwrap_or_else(|| whale_pool_threads(physical));
        let small_threads =
            env_usize("SCAN_SMALL_POOL_THREADS").unwrap_or_else(|| small_pool_threads(physical));
        let slots = env_usize("SCAN_WHALE_SLOTS")
            .unwrap_or_else(|| whale_slots(physical))
            .max(1);
        let small_max_bytes = small_job_max_bytes();
        let big_min_bytes = big_job_min_bytes().max(small_max_bytes);
        let share = std::env::var("SCAN_LANE_SHARE").as_deref() != Ok("0");
        if threads == 0 {
            tracing::info!("whale analysis pools disabled (SCAN_WHALE_POOL_THREADS=0)");
        } else {
            tracing::info!(
                threads_per_whale = threads,
                threads_per_small = small_threads,
                big_whale_slots = slots,
                whale_over_mb = small_max_bytes / (1024 * 1024),
                big_over_mb = big_min_bytes / (1024 * 1024),
                share,
                "analysis lanes ready: every payload runs on a private rayon pool sized to it"
            );
        }
        WhaleConfig {
            threads,
            small_threads,
            physical,
            share,
            slots,
            small_max_bytes,
            big_min_bytes,
        }
    })
}

/// Big whales in flight.
static WHALE_SLOTS_IN_USE: AtomicUsize = AtomicUsize::new(0);

/// One of [`whale_slots`], released on drop.
struct WhaleSlot;

/// Every big-whale slot is taken. Surfaced as 429 so the router places the
/// analysis on a worker with a free slot instead of queueing it here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WhaleSlotBusy {
    slots: usize,
}

impl std::fmt::Display for WhaleSlotBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "whale lane at capacity ({0}/{0} big analyses in flight)",
            self.slots
        )
    }
}

impl std::error::Error for WhaleSlotBusy {}

impl WhaleSlot {
    /// Take a slot now or refuse. This never waits: a request that reaches
    /// here is already streaming, so a queue is invisible to the router and
    /// the request sits behind whatever holds the slot for that whale's
    /// whole run (three wheels of 17–78 MB on 4-core scan-pdx, 2026-09-06:
    /// one held the only slot for 34 minutes and the other two timed out
    /// behind it while three other workers had room). A refusal ends the
    /// stream in milliseconds and the router tries the next worker.
    fn try_acquire(slots: usize) -> Result<Self, WhaleSlotBusy> {
        WHALE_SLOTS_IN_USE
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |in_use| {
                (in_use < slots).then_some(in_use + 1)
            })
            .map(|_| Self)
            .map_err(|_full| WhaleSlotBusy { slots })
    }

    /// Slots in use right now, for `/_/stats`.
    fn in_use() -> usize {
        WHALE_SLOTS_IN_USE.load(Ordering::Acquire)
    }
}

impl Drop for WhaleSlot {
    fn drop(&mut self) {
        WHALE_SLOTS_IN_USE.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A whale's private rayon pool for the life of one analysis: the bulkhead
/// that keeps a 40 MB package from stalling every 300-byte one that arrives
/// while it runs, and every 2 MB one from stalling behind it.
///
/// Every server analysis runs on a tokio blocking thread, so each inner
/// `par_iter` it issues — string extraction in stng, a member-window flush in
/// cleave — is *injected* into a rayon pool from outside, and rayon workers
/// take injected work only once their own deques are empty. While a whale's
/// thousands of member tasks sit in those deques they never are. Measured
/// 2026-09-05 at concurrency 8 over 128 real PURLs: a package that analyzes
/// in 0.3s alone waited 16.8s in `Registry::in_worker_cold` for a global-pool
/// worker, and the p90 sat at 5.2–6.9s whether or not the LLM pass ran.
///
/// A single *shared* whale pool was the first cut and fixed the small side
/// (p90 4.1s), but moved the starvation into the whale pool: three 36–254 MB
/// wheels in flight kept a 1.3 MB package waiting the whole 200s sweep, and
/// held each other to 170s+ where alone they take 25–40s. So each whale gets
/// its own pool ([`whale_pool_threads`] wide, dropped when the analysis
/// returns); nothing shares an injector with a whale. Big ones
/// (`SCAN_BIG_JOB_MB`) also take one of [`whale_slots`]; a burst of them is
/// refused past that count ([`WhaleSlotBusy`]) so the router spreads it over
/// the fleet instead of oversubscribing this host; the threads are cheap, the
/// cores are the budget.
///
/// `SCAN_WHALE_POOL_THREADS` sets the per-whale width (0 sends whales to the
/// global pool); `SCAN_WHALE_SLOTS` the big-whale concurrency;
/// `SCAN_SMALL_JOB_MB` (shared with the slot lanes) says where whale starts.
pub(crate) struct WhaleLane {
    pool: rayon::ThreadPool,
    /// Held for the pool's life when the payload is a big whale.
    _slot: Option<WhaleSlot>,
}

impl WhaleLane {
    /// Run `f` on this lane's pool, blocking until it returns.
    pub(crate) fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        self.pool.install(f)
    }
}

/// Big-whale slots in use and available, for `/_/stats`.
pub(crate) fn whale_slot_usage() -> (usize, usize) {
    (WhaleSlot::in_use(), whale_config().slots)
}

/// The lane a payload of `bytes` analyzes on: a private pool sized to it —
/// [`small_pool_threads`] wide at or below the small cap, [`whale_pool_threads`]
/// above it, after waiting for a slot if the payload is big — or `None` (the
/// global pool) when private pools are disabled for its size or fail to
/// build. The wait is the caller's phase to report.
///
/// Built per analysis and dropped after it: parking idle pools for reuse
/// (warm thread-local caches) measured as a wash, within noise on p50, p90
/// and throughput over 256 PURLs, so the simpler form stays.
///
/// Small payloads got private pools too once the whales had them: on the
/// global pool they compete for cleave's bounded inner-parallel owner slots
/// and half of them analyze serially at concurrency 8; on a pool of their
/// own each is parallel and exempt (`dedicated_pool`). Measured 2026-09-05
/// over 256 PURLs: throughput 203 → 230/min, mean 1.65 → 1.35 s.
///
/// Never for pull work. The idle worker's analyses already run on a pool of
/// their own — bounded, scheduled below the server's, and drawing on their
/// own parallelism slots — and a lane would undo all three: a lane pool's
/// threads are unmarked and at normal priority, and a big idle archive would
/// take one of the few whale slots and hold it for its whole slow run.
/// Measured 2026-09-06 on scan-lax: two of the two whale slots held by pull
/// work, and two interactive analyses waited in `whale:lane` until their
/// 30-minute budget ran out.
///
/// A big whale with every slot taken is refused ([`WhaleSlotBusy`]) rather
/// than queued; the handler turns that into a retry-later response.
pub(crate) fn whale_lane_for(bytes: u64) -> Result<Option<WhaleLane>, WhaleSlotBusy> {
    let cfg = whale_config();
    let floor = if bytes <= cfg.small_max_bytes {
        cfg.small_threads
    } else {
        cfg.threads
    };
    if floor == 0 {
        return Ok(None);
    }
    let threads = if cfg.share {
        // Alone on the box, the global pool is the widest lane there is.
        match lane_threads(floor, cfg.physical, ACTIVE_REQUESTS.load(Ordering::Acquire)) {
            Some(threads) => threads,
            None => return Ok(None),
        }
    } else {
        floor
    };
    let slot = if bytes > cfg.big_min_bytes {
        Some(WhaleSlot::try_acquire(cfg.slots)?)
    } else {
        None
    };
    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(crate::RAYON_STACK_MB * 1024 * 1024)
        .thread_name(move |i| format!("rayon-lane{threads}-{i}"))
        .build()
    {
        Ok(pool) => Ok(Some(WhaleLane { pool, _slot: slot })),
        Err(e) => {
            tracing::warn!(error = %e, bytes, "failed to build a private analysis pool; this payload shares the global pool");
            Ok(None)
        }
    }
}

#[cfg(test)]
mod whale_pool_tests {
    /// Per-whale pools scale with the host: a quarter of the cores, floor
    /// two, ceiling sixteen — the global pool keeps every core regardless.
    #[test]
    fn whale_pool_threads_scale_from_laptop_to_workstation() {
        assert_eq!(super::whale_pool_threads(1), 2);
        assert_eq!(super::whale_pool_threads(4), 2);
        assert_eq!(super::whale_pool_threads(8), 2);
        assert_eq!(super::whale_pool_threads(16), 4);
        assert_eq!(super::whale_pool_threads(64), 16);
        assert_eq!(super::whale_pool_threads(128), 16);
        assert_eq!(super::whale_pool_threads(256), 16);
    }

    /// Small-payload pools: an eighth of the cores, 2–8.
    #[test]
    fn small_pool_threads_scale_with_cores() {
        assert_eq!(super::small_pool_threads(1), 2);
        assert_eq!(super::small_pool_threads(4), 2);
        assert_eq!(super::small_pool_threads(16), 2);
        assert_eq!(super::small_pool_threads(32), 4);
        assert_eq!(super::small_pool_threads(64), 8);
        assert_eq!(super::small_pool_threads(256), 8);
    }

    /// Lane width follows the load: alone means the global pool, otherwise an
    /// even share of the cores floored at the tier width.
    #[test]
    fn lane_threads_share_cores_by_load() {
        assert_eq!(super::lane_threads(8, 64, 0), None);
        assert_eq!(super::lane_threads(8, 64, 1), None);
        assert_eq!(super::lane_threads(8, 64, 2), Some(32));
        assert_eq!(super::lane_threads(8, 64, 4), Some(16));
        assert_eq!(super::lane_threads(8, 64, 8), Some(8));
        assert_eq!(
            super::lane_threads(8, 64, 16),
            Some(8),
            "never below the floor"
        );
        assert_eq!(super::lane_threads(16, 64, 8), Some(16), "whale floor");
        assert_eq!(super::lane_threads(2, 4, 2), Some(2));
        assert_eq!(
            super::lane_threads(16, 4, 2),
            Some(16),
            "floor above the cores stays the floor"
        );
    }

    /// Big-whale slots: an eighth of the cores, 1–8, so slots × threads stays
    /// near two whale threads per core.
    #[test]
    fn whale_slots_scale_with_cores() {
        assert_eq!(super::whale_slots(1), 1);
        assert_eq!(super::whale_slots(4), 1);
        assert_eq!(super::whale_slots(8), 1);
        assert_eq!(super::whale_slots(16), 2);
        assert_eq!(super::whale_slots(64), 8);
        assert_eq!(super::whale_slots(128), 8);
        assert_eq!(super::whale_slots(256), 8);
        for cores in [4, 16, 64, 128] {
            assert!(
                super::whale_slots(cores) * super::whale_pool_threads(cores) <= 2 * cores.max(4)
            );
        }
    }

    /// A full slot table refuses instead of waiting, and a dropped slot is
    /// available again.
    #[test]
    fn whale_slot_refuses_when_full_and_frees_on_drop() {
        let held = super::WhaleSlot::try_acquire(1).expect("first slot");
        assert_eq!(super::WhaleSlot::in_use(), 1);
        assert_eq!(
            super::WhaleSlot::try_acquire(1).err(),
            Some(super::WhaleSlotBusy { slots: 1 }),
            "a full table refuses at once"
        );
        drop(held);
        assert_eq!(super::WhaleSlot::in_use(), 0);
        let again = super::WhaleSlot::try_acquire(1);
        assert!(again.is_ok(), "the slot is free again after drop");
        drop(again);
        assert_eq!(super::WhaleSlot::in_use(), 0);
    }
}
