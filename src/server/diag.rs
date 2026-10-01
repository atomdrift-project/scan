//! The diagnostic routes: `/_/health`, `/_/info`, `/_/stats`, `/_/memory`,
//! `/_/requests`, `/_/threads`.
//!
//! Each body is a typed struct. Field order is the wire order, and consumers —
//! beamline's router, monitors, dashboards — index these by name, so
//! `key_sets_are_pinned` holds the names in place.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::Serialize;

use super::acl::Trusted;
use super::{AppState, JobBucket, PURL_TYPE_NAMES, SIZE_BUCKET_NAMES};

const MIB: u64 = 1024 * 1024;

fn rss_mb() -> Option<u64> {
    cleave::memory_tracker::current_rss().map(|b| b / MIB)
}

/// `/_/health` while the server cannot serve: still loading, or startup failed.
#[derive(Serialize)]
pub(super) struct HealthDown {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
    uptime_secs: u64,
}

/// `/_/health` past the memory ceiling.
#[derive(Serialize)]
pub(super) struct HealthDegraded {
    status: &'static str,
    reason: &'static str,
    rss_mb: Option<u64>,
    max_rss_mb: Option<u64>,
    active_tasks: usize,
    load_avg: Option<f64>,
    uptime_secs: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    rayon_threads: Option<usize>,
}

/// `/_/health` while serving. The fields after `reason` are diagnostic detail
/// that names samples in flight, sent only to a trusted request.
#[derive(Serialize)]
pub(super) struct HealthOk {
    status: &'static str,
    rss_mb: Option<u64>,
    max_rss_mb: Option<u64>,
    active_tasks: usize,
    max_concurrent_tasks: usize,
    load: f64,
    load_avg: Option<f64>,
    uptime_secs: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stuck_orphans: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    long_running_tasks: Option<Vec<LongRunning>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rayon_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    oldest_task: Option<OldestTask>,
}

#[derive(Serialize)]
pub(super) struct LongRunning {
    request_id: u64,
    name: String,
    elapsed_secs: u64,
    phase: String,
    thread_id: u64,
}

#[derive(Serialize)]
pub(super) struct OldestTask {
    name: String,
    elapsed_secs: u64,
}

/// GET /_/health — liveness check with memory and concurrency status.
/// Returns 503 while resources are still loading or when RSS exceeds the
/// configured limit. A fully-utilised worker pool returns 200 with
/// `status: "saturated"` — that's the target steady state, not a fault.
///
/// Every response carries `uptime_secs` (seconds since the server started)
/// so clients can detect restarts without polling a separate endpoint.
pub(super) async fn health(
    State(state): State<Arc<AppState>>,
    trusted: Option<Extension<Trusted>>,
) -> Response {
    // `/_/health` is the one route reachable without a bearer token, so that
    // tunnel and load-balancer probes work without holding a credential. The
    // liveness signal — status, memory, saturation — is public; the diagnostic
    // detail below it names the samples currently being analysed, so it is
    // added only for a request that authenticated (or when the server has no
    // token configured at all, which leaves the body as it always was).
    let trusted = trusted.is_some();
    let uptime_secs = state.started_at.elapsed().as_secs();

    if let Some(message) = state.init_failure() {
        tracing::error!("GET /_/health -> 503 (failed: {message})");
        let body = HealthDown {
            status: "failed",
            reason: Some("initialization_failed"),
            uptime_secs,
        };
        return (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
    }
    if !state.is_ready() {
        tracing::debug!("GET /_/health -> 503 (starting)");
        let body = HealthDown {
            status: "starting",
            reason: None,
            uptime_secs,
        };
        return (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
    }

    let load_avg = crate::system_load_avg();
    let rss_bytes = cleave::memory_tracker::current_rss();
    let rss_mb = rss_bytes.map(|b| b / MIB);
    let max_rss = state.config.max_rss_bytes.map(std::num::NonZeroU64::get);
    let max_rss_mb = max_rss.map(|b| b / MIB);
    let active_tasks = state.active_tasks();
    let overloaded = rss_bytes.zip(max_rss).is_some_and(|(rss, max)| rss > max);

    if overloaded {
        tracing::warn!("GET /_/health -> 503 (degraded, rss={rss_mb:?}MB)");
        let body = HealthDegraded {
            status: "degraded",
            reason: "memory_pressure",
            rss_mb,
            max_rss_mb,
            active_tasks,
            load_avg,
            uptime_secs,
            rayon_threads: trusted.then(rayon::current_num_threads),
        };
        return (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
    }
    let max_tasks = state.config.workers;
    let stuck_orphans = state.stuck_orphans.load(Ordering::Relaxed);

    // Tasks running longer than 120s — visible in /_/requests with full phase
    // detail. The count is always computed (it is cheap: `in_flight` holds at
    // most one entry per analysis slot, and the count feeds the log line), but
    // each entry names the sample being analysed, so the detail is built only
    // when it will actually be served.
    let now = Instant::now();
    let mut long_running_count = 0usize;
    let mut long_running = Vec::new();
    for entry in &state.in_flight {
        let elapsed_secs = now.duration_since(entry.started_at).as_secs();
        if elapsed_secs < 120 {
            continue;
        }
        long_running_count += 1;
        if trusted {
            long_running.push(LongRunning {
                request_id: *entry.key(),
                name: entry.name.clone(),
                elapsed_secs,
                phase: entry.phase.get(),
                thread_id: entry.thread_id.load(Ordering::Relaxed),
            });
        }
    }

    let load = if max_tasks > 0 {
        active_tasks as f64 / max_tasks as f64
    } else {
        0.0
    };
    // A fully-utilised worker pool is the *target* steady state, not a fault.
    // Report it as "saturated" with HTTP 200 so monitors can distinguish "all
    // slots busy" from real failures (memory pressure, stuck workers). The
    // analyze routes still refuse past capacity, so clients back off correctly
    // without /_/health pretending the server is unhealthy.
    let saturated = active_tasks >= max_tasks;
    let oldest = saturated
        .then(|| {
            state
                .in_flight
                .iter()
                .min_by_key(|e| e.started_at)
                .map(|e| (e.name.clone(), e.started_at.elapsed().as_secs()))
        })
        .flatten();

    if saturated {
        tracing::debug!(
            active_tasks,
            stuck_orphans,
            long_running = long_running_count,
            max_concurrent_tasks = max_tasks,
            oldest_task = ?oldest,
            "GET /_/health -> 200 (saturated)"
        );
    } else {
        tracing::debug!(
            "GET /_/health -> 200 (rss={rss_mb:?}MB, active={active_tasks}, long_running={long_running_count}, stuck_orphans={stuck_orphans}, load={load:.2})"
        );
    }

    Json(HealthOk {
        status: if saturated { "saturated" } else { "ok" },
        rss_mb,
        max_rss_mb,
        active_tasks,
        max_concurrent_tasks: max_tasks,
        load,
        load_avg,
        uptime_secs,
        reason: saturated.then_some("thread_pool_saturated"),
        stuck_orphans: trusted.then_some(stuck_orphans),
        long_running_tasks: trusted.then_some(long_running),
        rayon_threads: trusted.then(rayon::current_num_threads),
        oldest_task: oldest
            .filter(|_| trusted)
            .map(|(name, elapsed_secs)| OldestTask { name, elapsed_secs }),
    })
    .into_response()
}

/// `/_/info`.
#[derive(Serialize)]
pub(super) struct Info {
    version: &'static str,
    slots: usize,
    cpus: usize,
    max_upload_mb: usize,
    max_rss_mb: Option<u64>,
    total_mem_mb: u64,
    model_commit: Option<String>,
    traits_commit: Option<String>,
    /// Whether the idle worker is configured, running, and standing aside for
    /// a request. Published because its absence is otherwise invisible — it
    /// either starts or it does not, and until this existed the only evidence
    /// was a log line that never appeared.
    idle_worker: InfoIdleWorker,
}

#[derive(Serialize)]
pub(super) struct InfoIdleWorker {
    running: bool,
    frozen: bool,
    hopper: bool,
    interactive_in_flight: usize,
}

impl Info {
    /// Blocking: the commits are read from disk.
    fn collect(state: &AppState) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
            slots: state.config.workers,
            cpus: std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
            max_upload_mb: state.config.max_body_size / 1024 / 1024,
            max_rss_mb: state.config.max_rss_bytes.map(|n| n.get() / MIB),
            total_mem_mb: cleave::memory_tracker::total_memory().map_or(0, |bytes| bytes / MIB),
            model_commit: crate::models_repo::version(),
            traits_commit: cleave::traits_repo::version(),
            idle_worker: InfoIdleWorker {
                running: state.idle_worker_running(),
                frozen: state.is_busy(),
                hopper: state.config.hopper().is_some(),
                interactive_in_flight: state.in_flight.len(),
            },
        }
    }
}

/// Build a body on a blocking thread — it reads files — and serve it.
async fn collected<T: Serialize + Send + 'static>(
    state: Arc<AppState>,
    collect: fn(&AppState) -> T,
) -> Response {
    match tokio::task::spawn_blocking(move || collect(&state)).await {
        Ok(body) => Json(body).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "collecting a diagnostic body panicked");
            super::error::ApiError::internal().into_response()
        }
    }
}

/// GET /_/info — static server capacity and version info.
///
/// Read by clients (e.g. hopper) on startup so they can size their per-node
/// worker pools without hand-configuring slot counts, and so they can compare
/// model/traits commits across nodes for drift detection. Always 200,
/// independent of readiness — readiness is reported by /_/health.
pub(super) async fn info(State(state): State<Arc<AppState>>) -> Response {
    collected(state, Info::collect).await
}

/// A class of work's windowed latency, in milliseconds.
#[derive(Serialize)]
pub(crate) struct Recent {
    pub(crate) samples: u64,
    pub(crate) p80_ms: Option<u64>,
    pub(crate) mean_ms: Option<u64>,
}

/// One class of work on `/_/stats`.
#[derive(Serialize)]
pub(super) struct BucketStats {
    jobs: u64,
    avg_ms: Option<u64>,
    /// The one a router should read: a p80 over the last hour rather than a
    /// mean over the last few hundred jobs.
    recent: Recent,
}

/// Named buckets, as a JSON object keyed by name.
pub(super) struct Buckets<'a> {
    names: &'a [&'static str],
    buckets: &'a [JobBucket],
}

impl Serialize for Buckets<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.names.iter().zip(self.buckets).map(|(name, b)| {
            let (jobs, avg_ms) = b.mean_ms();
            (
                *name,
                BucketStats {
                    jobs,
                    avg_ms,
                    recent: b.recent(),
                },
            )
        }))
    }
}

#[derive(Serialize)]
pub(super) struct WhaleSlots {
    in_use: usize,
    max: usize,
}

#[derive(Serialize)]
pub(super) struct Uploads {
    pending: usize,
    capacity: usize,
    failed: usize,
    uploaded: usize,
}

#[derive(Serialize)]
pub(super) struct StatsIdleWorker {
    running: bool,
    frozen: bool,
}

/// `/_/stats`. The comment on each field says what a router reads it for.
#[derive(Serialize)]
pub(super) struct Stats<'a> {
    /// Where lookups this index could not answer actually went. A fleet
    /// quietly reading from the primary because the replica stopped answering
    /// is otherwise indistinguishable from one reading the replica, until the
    /// primary's load says so.
    corpus: Option<super::corpus::CorpusStats>,
    /// Saturation. The best routing signal available, because it is current
    /// rather than lagging: a latency average still reports health for the
    /// minute after a server takes on four large archives.
    slots: usize,
    slots_free: usize,
    in_flight: usize,
    /// Cores of this box's load that will not be there when a request
    /// arrives, for a router to subtract before calling the box saturated.
    /// Without it, load alone made a server full of sheddable work look
    /// identical to one full of requests (2026-09-05). See beamline's
    /// `foregroundPressure`.
    background_in_flight: usize,
    /// The basis `slots` was sized on, so a caller can read `load1` against
    /// the right denominator. Worth reporting because `in_flight` is this
    /// process's own count and nothing else's: measured on a 64-core node,
    /// `slots_free=64 in_flight=0` alongside `load1=50`.
    physical_cpus: Option<usize>,
    /// Logical cores the whole machine kept busy since the previous poll, from
    /// the kernel's CPU counters. Preferred over `load1` because it means the
    /// same thing on every platform. `null` until two polls exist or where the
    /// platform has no counters.
    cpu_busy_cores: Option<f64>,
    /// Big-whale slots: the one wait on the interactive path that is not a
    /// core or a slot, and the one that held two requests for their whole
    /// budget on 2026-09-06.
    whale_slots: WhaleSlots,
    overloaded: bool,
    stuck_orphans: usize,
    /// What this server has actually done. More useful for routing than a
    /// load average, which is a whole-host number that folds in every other
    /// tenant on the box.
    uptime_secs: u64,
    jobs_started: u64,
    jobs_completed: u64,
    /// Begun and never finished: timeouts, panics, and clients that hung up
    /// mid-analysis. Non-zero and climbing is the shape of a sick server.
    jobs_unfinished: u64,
    avg_job_bytes: Option<u64>,
    /// Fresh analyses only: what this server costs on work it has not seen.
    avg_job_ms: Option<u64>,
    /// Answered from cleave's analysis cache, kept out of the figures above
    /// so it cannot flatter a server into looking fast at work it never did.
    avg_job_ms_cached: Option<u64>,
    cached_samples: u64,
    /// Windowed p80, for the fleet-wide view, beside the lifetime means.
    recent: Recent,
    recent_lookup: Recent,
    latency_window_secs: u64,
    /// `/lookup` service time — an index probe, near-constant in the size of
    /// the artifact, and three orders of magnitude below an analysis.
    avg_lookup_ms: Option<u64>,
    /// Microseconds too: a healthy index probe rounds to 0ms, and a routing
    /// signal that is always zero is no signal at all.
    avg_lookup_us: Option<u64>,
    lookup_samples: u64,
    /// Lifetime, for the operator rather than the router.
    avg_job_ms_lifetime: Option<u64>,
    avg_job_samples: u64,
    /// By PURL type: a `?purl=` request has no size until the artifact is
    /// fetched, so this is what a router can compare on when the choice still
    /// matters.
    avg_job_ms_by_type: Buckets<'a>,
    /// By input size: a caller that knows how big the artifact is should
    /// compare servers at that size, not overall.
    avg_job_ms_by_size: Buckets<'a>,
    /// Memory headroom. A server near its ceiling is about to pause admission;
    /// a router should move away before that, not discover it by timing out.
    rss_mb: Option<u64>,
    max_rss_mb: Option<u64>,
    load1: Option<f64>,
    /// Capability, not speed. A server without 7z cannot read a DMG, so it
    /// returns a *weaker verdict* rather than a slower one.
    tools: Vec<&'static str>,
    max_upload_mb: usize,
    /// Verdict comparability. A scanner on stale traits produces an answer
    /// that should not be cached as authoritative alongside a current one.
    traits_commit: Option<String>,
    model_commit: Option<String>,
    ready: bool,
    /// Whether the verdicts it computes are actually reaching hopper.
    /// `failed` climbing means work is being done and then lost.
    uploads: Option<Uploads>,
    /// Whether spare capacity is genuinely spare.
    idle_worker: StatsIdleWorker,
}

impl<'a> Stats<'a> {
    /// Blocking: CPU counters, memory, tools and commits are read from the
    /// system and from disk.
    fn collect(state: &'a AppState) -> Self {
        let jobs = &state.jobs;
        let in_flight = state.in_flight.len();
        let started = jobs.started.load(Ordering::Relaxed);
        let completed = jobs.completed.load(Ordering::Relaxed);
        // Averages over completed jobs only: a job still running has
        // contributed no duration, and dividing by `started` would report
        // every busy server as faster than it is.
        let avg = |total: &std::sync::atomic::AtomicU64| {
            (completed > 0).then(|| total.load(Ordering::Relaxed) / completed)
        };
        let (recent_n, avg_job_ms) = jobs.overall.mean_ms();
        let (cached_n, avg_job_ms_cached) = jobs.cached.mean_ms();
        let lookup_n = jobs.lookups.count.load(Ordering::Relaxed);
        let lookup_micros = jobs.lookups.micros.load(Ordering::Relaxed);
        let (whale_in_use, whale_max) = crate::analysis::whale_slot_usage();
        Self {
            corpus: state.corpus.as_ref().map(|c| c.stats()),
            slots: state.config.workers,
            slots_free: state.available_analysis_permits(),
            in_flight,
            background_in_flight: state.sheddable_cores(),
            physical_cpus: cleave::memory_tracker::physical_cpu_count(),
            cpu_busy_cores: state.cpu_busy.sample(),
            whale_slots: WhaleSlots {
                in_use: whale_in_use,
                max: whale_max,
            },
            overloaded: state.overload_mark().is_some(),
            stuck_orphans: state.stuck_orphans.load(Ordering::Relaxed),
            uptime_secs: state.started_at.elapsed().as_secs(),
            jobs_started: started,
            jobs_completed: completed,
            jobs_unfinished: started
                .saturating_sub(completed)
                .saturating_sub(in_flight as u64),
            avg_job_bytes: avg(&jobs.bytes_total),
            avg_job_ms,
            avg_job_ms_cached,
            cached_samples: cached_n,
            recent: jobs.overall.recent(),
            recent_lookup: jobs.lookups.recent(),
            latency_window_secs: super::latency::WINDOW.as_secs(),
            avg_lookup_ms: (lookup_n > 0).then(|| lookup_micros / lookup_n / 1_000),
            avg_lookup_us: (lookup_n > 0).then(|| lookup_micros / lookup_n),
            lookup_samples: lookup_n,
            avg_job_ms_lifetime: avg(&jobs.micros_total).map(|us| us / 1_000),
            avg_job_samples: recent_n,
            avg_job_ms_by_type: Buckets {
                names: &PURL_TYPE_NAMES,
                buckets: &jobs.by_type,
            },
            avg_job_ms_by_size: Buckets {
                names: &SIZE_BUCKET_NAMES,
                buckets: &jobs.by_size,
            },
            rss_mb: rss_mb(),
            max_rss_mb: state.config.max_rss_bytes.map(|n| n.get() / MIB),
            load1: crate::system_load_avg(),
            tools: crate::tools::available_names(),
            max_upload_mb: state.config.max_body_size / 1024 / 1024,
            traits_commit: cleave::traits_repo::version(),
            model_commit: crate::models_repo::version(),
            ready: state.is_ready(),
            uploads: state.uploader.as_ref().map(|u| {
                let s = u.stats();
                Uploads {
                    pending: s.pending,
                    capacity: s.capacity,
                    failed: s.failed,
                    uploaded: s.uploaded,
                }
            }),
            idle_worker: StatsIdleWorker {
                running: state.idle_worker_running(),
                frozen: state.is_busy(),
            },
        }
    }
}

/// `GET /_/stats` — the live signals a router needs to choose this server.
///
/// Separate from `/_/info` because the two have different lifetimes: `/_/info`
/// reports what this build *is* and barely changes, while everything here moves
/// every second and must not be cached.
///
/// The vocabulary deliberately matches what the pull worker already advertises
/// to hopper on `/api/next` — slots, rss, load, max_bytes, traits, tools — so a
/// caller reasoning about which scanner to use is reading the same facts hopper
/// uses to ration work, rather than a second dialect of the same idea.
pub(super) async fn stats(State(state): State<Arc<AppState>>) -> Response {
    // Serialized on the blocking thread too: `Stats` borrows the state.
    match tokio::task::spawn_blocking(move || serde_json::to_vec(&Stats::collect(&state))).await {
        Ok(Ok(body)) => (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response(),
        Ok(Err(e)) => {
            tracing::error!(error = %e, "could not serialize /_/stats");
            super::error::ApiError::internal().into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, "collecting /_/stats panicked");
            super::error::ApiError::internal().into_response()
        }
    }
}

#[derive(Serialize)]
pub(super) struct Memory {
    process: MemoryProcess,
    server: MemoryServer,
    thread_pools: MemoryPools,
}

#[derive(Serialize)]
pub(super) struct MemoryProcess {
    rss_mb: Option<u64>,
    max_rss_mb: Option<u64>,
    jemalloc: Option<Jemalloc>,
}

#[derive(Serialize)]
pub(super) struct Jemalloc {
    allocated_mb: u64,
    active_mb: u64,
    metadata_mb: u64,
    resident_mb: u64,
    retained_mb: u64,
    fragmentation_mb: u64,
}

#[derive(Serialize)]
pub(super) struct MemoryServer {
    active_tasks: usize,
    stuck_orphans: usize,
    max_concurrent_tasks: usize,
    requests_total: u64,
}

#[derive(Serialize)]
pub(super) struct MemoryPools {
    rayon_threads: usize,
}

/// GET /_/memory — memory diagnostics for all major structures.
///
/// `process.jemalloc` is null unless cleave was built with `--features jemalloc`.
/// When available, `jemalloc.allocated_mb` is the most useful leak indicator:
/// if it tracks RSS closely, you have a real leak; if RSS >> allocated, it's fragmentation.
pub(super) async fn memory_stats(State(state): State<Arc<AppState>>) -> Json<Memory> {
    let jemalloc = cleave::memory_tracker::jemalloc_stats().map(|s| Jemalloc {
        allocated_mb: s.allocated / MIB,
        active_mb: s.active / MIB,
        metadata_mb: s.metadata / MIB,
        resident_mb: s.resident / MIB,
        retained_mb: s.retained / MIB,
        fragmentation_mb: s.active.saturating_sub(s.allocated) / MIB,
    });
    Json(Memory {
        process: MemoryProcess {
            rss_mb: rss_mb(),
            max_rss_mb: state.config.max_rss_bytes.map(|n| n.get() / MIB),
            jemalloc,
        },
        server: MemoryServer {
            active_tasks: state.active_tasks(),
            stuck_orphans: state.stuck_orphans.load(Ordering::Relaxed),
            max_concurrent_tasks: state.config.workers,
            requests_total: state.next_request_id.load(Ordering::Relaxed),
        },
        thread_pools: MemoryPools {
            rayon_threads: rayon::current_num_threads(),
        },
    })
}

#[derive(Serialize)]
pub(super) struct Requests {
    count: usize,
    /// Distinct runs; `attached` counts the requests riding them. The gap is
    /// duplicate work single-flight is absorbing.
    analyses: usize,
    attached: usize,
    requests: Vec<RequestEntry>,
}

#[derive(Serialize)]
pub(super) struct RequestEntry {
    request_id: u64,
    name: String,
    size_bytes: u64,
    elapsed_ms: u128,
    long_running: bool,
    phase: String,
    thread_id: Option<u64>,
}

/// GET /_/requests — all analyses currently in flight, longest-running first.
pub(super) async fn requests(State(state): State<Arc<AppState>>) -> Json<Requests> {
    let now = Instant::now();
    let mut entries: Vec<RequestEntry> = state
        .in_flight
        .iter()
        .map(|e| {
            let elapsed_ms = now.duration_since(e.started_at).as_millis();
            let tid = e.thread_id.load(Ordering::Relaxed);
            RequestEntry {
                request_id: *e.key(),
                name: e.name.clone(),
                size_bytes: e.size_bytes,
                elapsed_ms,
                long_running: elapsed_ms >= 120_000,
                phase: e.phase.get(),
                thread_id: (tid > 0).then_some(tid),
            }
        })
        .collect();
    entries.sort_by_key(|e| std::cmp::Reverse(e.elapsed_ms));
    let census = state.flights.census();
    Json(Requests {
        count: entries.len(),
        analyses: census.analyses,
        attached: census.attached,
        requests: entries,
    })
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

    // SAFETY: `getpid` takes no arguments and cannot fail.
    let pid = unsafe { libc::getpid() };
    let mib: [libc::c_int; 4] = [
        libc::CTL_KERN,
        libc::KERN_PROC,
        libc::KERN_PROC_PID | libc::KERN_PROC_INC_THREAD,
        pid,
    ];

    let mut len: libc::size_t = 0;
    // SAFETY: a size query: `mib` is a valid 4-element name, the output buffer
    // is null, and `len` receives the size the kernel would write.
    let ret = unsafe {
        libc::sysctl(
            mib.as_ptr(),
            4,
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return serde_json::json!({"error": "sysctl size query failed"});
    }

    len += len / 4; // 25% slack for new threads between calls
    let count = len / mem::size_of::<libc::kinfo_proc>();
    // SAFETY: `kinfo_proc` is a plain C struct for which all-zero bytes are a
    // valid value.
    let mut procs: Vec<libc::kinfo_proc> = (0..count).map(|_| unsafe { mem::zeroed() }).collect();
    let mut actual_len = count * mem::size_of::<libc::kinfo_proc>();

    // SAFETY: `procs` owns `actual_len` writable bytes, and the kernel writes
    // at most that many, reporting how many it wrote back in `actual_len`.
    let ret = unsafe {
        libc::sysctl(
            mib.as_ptr(),
            4,
            procs.as_mut_ptr().cast(),
            &mut actual_len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return serde_json::json!({"error": "sysctl data query failed"});
    }

    procs.truncate(actual_len / mem::size_of::<libc::kinfo_proc>());

    let c_str = |buf: &[libc::c_char]| {
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    let state_str = |s: libc::c_char| match s as u8 {
        1 => "idle",
        2 => "running",
        3 => "sleeping",
        4 => "stopped",
        5 => "zombie",
        6 => "waiting",
        7 => "locked",
        _ => "unknown",
    };

    let mut threads: Vec<serde_json::Value> = procs
        .iter()
        .map(|p| {
            serde_json::json!({
                "tid": p.ki_tid,
                "name": c_str(&p.ki_tdname),
                "state": state_str(p.ki_stat),
                "wchan": c_str(&p.ki_wmesg),
            })
        })
        .collect();

    threads.sort_by_key(|t| t["tid"].as_u64().unwrap_or(0));
    serde_json::json!({"count": threads.len(), "threads": threads})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ServerConfig;

    /// The keys of a serialized body, in wire order.
    fn keys<T: Serialize>(body: &T) -> Vec<String> {
        serde_json::to_value(body)
            .expect("serializes")
            .as_object()
            .expect("an object")
            .keys()
            .cloned()
            .collect()
    }

    /// Routers, monitors and dashboards index these bodies by name, so a key
    /// that moves or is renamed is an outage that looks like nothing. These
    /// lists were taken from the `json!` literals the structs replaced.
    #[tokio::test]
    async fn key_sets_are_pinned() {
        let (_app, state) = super::super::assemble(ServerConfig {
            model_dir: std::env::temp_dir(),
            ..ServerConfig::default()
        })
        .expect("app");

        assert_eq!(
            keys(&Stats::collect(&state)),
            [
                "corpus",
                "slots",
                "slots_free",
                "in_flight",
                "background_in_flight",
                "physical_cpus",
                "cpu_busy_cores",
                "whale_slots",
                "overloaded",
                "stuck_orphans",
                "uptime_secs",
                "jobs_started",
                "jobs_completed",
                "jobs_unfinished",
                "avg_job_bytes",
                "avg_job_ms",
                "avg_job_ms_cached",
                "cached_samples",
                "recent",
                "recent_lookup",
                "latency_window_secs",
                "avg_lookup_ms",
                "avg_lookup_us",
                "lookup_samples",
                "avg_job_ms_lifetime",
                "avg_job_samples",
                "avg_job_ms_by_type",
                "avg_job_ms_by_size",
                "rss_mb",
                "max_rss_mb",
                "load1",
                "tools",
                "max_upload_mb",
                "traits_commit",
                "model_commit",
                "ready",
                "uploads",
                "idle_worker",
            ]
        );
        let stats = serde_json::to_value(Stats::collect(&state)).expect("serializes");
        assert_eq!(
            stats["avg_job_ms_by_type"]
                .as_object()
                .expect("by type")
                .keys()
                .collect::<Vec<_>>(),
            ["cargo", "golang", "npm", "pypi", "other"]
        );
        assert_eq!(
            keys(&stats["avg_job_ms_by_size"]["le_1mb"]),
            ["jobs", "avg_ms", "recent"]
        );
        assert_eq!(keys(&stats["recent"]), ["samples", "p80_ms", "mean_ms"]);
        assert_eq!(keys(&stats["whale_slots"]), ["in_use", "max"]);
        assert_eq!(keys(&stats["idle_worker"]), ["running", "frozen"]);

        let info = Info::collect(&state);
        assert_eq!(
            keys(&info),
            [
                "version",
                "slots",
                "cpus",
                "max_upload_mb",
                "max_rss_mb",
                "total_mem_mb",
                "model_commit",
                "traits_commit",
                "idle_worker",
            ]
        );
        assert_eq!(
            keys(&info.idle_worker),
            ["running", "frozen", "hopper", "interactive_in_flight"]
        );

        let trusted_ok = HealthOk {
            status: "saturated",
            rss_mb: Some(1),
            max_rss_mb: None,
            active_tasks: 2,
            max_concurrent_tasks: 2,
            load: 1.0,
            load_avg: Some(0.5),
            uptime_secs: 3,
            reason: Some("thread_pool_saturated"),
            stuck_orphans: Some(0),
            long_running_tasks: Some(Vec::new()),
            rayon_threads: Some(4),
            oldest_task: Some(OldestTask {
                name: "a.zip".into(),
                elapsed_secs: 9,
            }),
        };
        assert_eq!(
            keys(&trusted_ok),
            [
                "status",
                "rss_mb",
                "max_rss_mb",
                "active_tasks",
                "max_concurrent_tasks",
                "load",
                "load_avg",
                "uptime_secs",
                "reason",
                "stuck_orphans",
                "long_running_tasks",
                "rayon_threads",
                "oldest_task",
            ]
        );
        let public_ok = HealthOk {
            reason: None,
            stuck_orphans: None,
            long_running_tasks: None,
            rayon_threads: None,
            oldest_task: None,
            ..trusted_ok
        };
        assert_eq!(
            keys(&public_ok),
            [
                "status",
                "rss_mb",
                "max_rss_mb",
                "active_tasks",
                "max_concurrent_tasks",
                "load",
                "load_avg",
                "uptime_secs",
            ]
        );
        let degraded = HealthDegraded {
            status: "degraded",
            reason: "memory_pressure",
            rss_mb: Some(1),
            max_rss_mb: Some(1),
            active_tasks: 0,
            load_avg: None,
            uptime_secs: 1,
            rayon_threads: None,
        };
        assert_eq!(
            keys(&degraded),
            [
                "status",
                "reason",
                "rss_mb",
                "max_rss_mb",
                "active_tasks",
                "load_avg",
                "uptime_secs",
            ]
        );
        let failed = HealthDown {
            status: "failed",
            reason: Some("initialization_failed"),
            uptime_secs: 1,
        };
        assert_eq!(keys(&failed), ["status", "reason", "uptime_secs"]);

        let running = super::super::handlers::StatusBody {
            state: "running",
            purl: None,
            url: None,
            sha256: Some("a".repeat(64)),
            elapsed_ms: Some(5),
            attached: Some(1),
        };
        assert_eq!(
            keys(&running),
            ["state", "purl", "url", "sha256", "elapsed_ms", "attached"]
        );
        let unknown = super::super::handlers::StatusBody {
            state: "unknown",
            elapsed_ms: None,
            attached: None,
            ..running
        };
        assert_eq!(keys(&unknown), ["state", "purl", "url", "sha256"]);
    }
}
