//! The HTTP API server.
//!
//! Accepts artifacts — uploaded bytes, a package URL, an exact URL, or a file
//! on this host — runs cleave static analysis and the model, and answers with
//! a decision.
//!
//! Routes:
//!   GET  /_/health      — liveness (public; detail only when trusted)
//!   GET  /_/info        — build and capacity, for sizing a client
//!   GET  /_/stats       — live routing signals
//!   GET  /_/memory, /_/requests, /_/threads — diagnostics
//!   POST /_/reload      — reload the model bundle from disk (admin)
//!   POST /_/update      — pull models and traits, then reload (admin)
//!   GET  /v1/lookup     — decisions for one or many artifacts (no slot)
//!   POST /v1/analyze    — a decision, streamed if the analysis is slow
//!
//! [`ServerConfig`] is plain data: [`Startup::resolve`] builds it from a
//! command line, and [`build_app`] validates it before anything starts.

mod access;
mod acl;
mod admin;
mod analyze;
mod config;
mod corpus;
mod cpu_busy;
mod decision;
mod diag;
mod error;
mod flight;
mod input;
mod jobs;
mod latency;
mod local;
mod tasks;
mod v1;

pub use acl::{Cidr, CidrError, TokenDigest, TokenError};
pub use config::{DEFAULT_ANALYSIS_TIMEOUT_SECS, ServerConfig, Startup};

use crate::analysis::{ActiveRequest, ModelResources, RequestPhase};

use axum::Router;
use axum::extract::{ConnectInfo, DefaultBodyLimit};
use axum::middleware;
use axum::routing::{get, post};
use std::future::Future;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::{Duration, Instant};
use tokio::signal;
use tokio::sync::{Semaphore, watch};

use cpu_busy::CpuBusy;
use error::ApiError;
use jobs::Jobs;
use tasks::Tasks;

use crate::explain::ShapImportance;
use crate::model::Model;

/// One admitted analysis, as `/_/requests` and the watchdog report it.
struct InFlightRequest {
    name: String,
    size_bytes: u64,
    started_at: Instant,
    /// Shared with the blocking task; set to true to request cooperative cancellation.
    cancellation: Arc<AtomicBool>,
    /// Tracks the current analysis phase inside cleave and scan. Updated at each
    /// major stage so `/_/requests` can report what a stuck request is doing.
    phase: RequestPhase,
    /// OS thread ID of the blocking thread servicing this request (0 until started).
    thread_id: AtomicU64,
}

impl InFlightRequest {
    fn new(
        name: &str,
        size_bytes: u64,
        cancellation: Arc<AtomicBool>,
        phase: RequestPhase,
    ) -> Self {
        Self {
            name: name.to_owned(),
            size_bytes,
            started_at: Instant::now(),
            cancellation,
            phase,
            thread_id: AtomicU64::new(0),
        }
    }
}

/// What an admitted analysis holds: a request slot and a core.
///
/// Slots are sized for a request's whole life, most of which is waiting on
/// the network; cores are sized to the rayon pool, the only capacity an
/// analysis really contends for. Reporting slots alone can tell a router a
/// server has free slots when every core is busy, and an analysis sent there
/// waits minutes to start.
pub(super) struct AnalysisPermit {
    _slot: tokio::sync::OwnedSemaphorePermit,
    _cpu: tokio::sync::OwnedSemaphorePermit,
}

/// RAII guard over one admitted analysis. On drop — the analysis finished, or
/// the handler future was dropped on a client disconnect — it signals
/// cooperative cancellation to the blocking thread and removes the in-flight
/// entry, so neither the permits nor the entry can leak.
pub(super) struct RequestGuard {
    request_id: u64,
    state: Arc<AppState>,
    cancellation: Arc<AtomicBool>,
    /// Held here so the slot and core are released when the guard drops.
    _permit: AnalysisPermit,
    /// Counts this analysis in the share of cores each lane gets.
    _active: ActiveRequest,
}

impl RequestGuard {
    /// Register an admitted analysis on `/_/requests` and hold its capacity
    /// until the guard drops.
    fn new(
        state: &Arc<AppState>,
        request_id: u64,
        entry: InFlightRequest,
        permit: AnalysisPermit,
    ) -> Self {
        let cancellation = Arc::clone(&entry.cancellation);
        state.in_flight.insert(request_id, entry);
        state.jobs.started.fetch_add(1, Ordering::Relaxed);
        Self {
            request_id,
            state: Arc::clone(state),
            cancellation,
            _permit: permit,
            _active: ActiveRequest::begin(),
        }
    }

    /// Run `work` on a blocking thread, bounded by `--analysis-timeout`.
    ///
    /// The guard is dropped when the work returns. On timeout it follows the
    /// blocking task instead — tokio cannot stop a blocking thread, only ask it
    /// via the cancellation flag — so the thread's slot and core stay
    /// accounted for as long as it runs.
    pub(super) async fn run(
        self,
        work: impl FnOnce() -> anyhow::Result<crate::engine::ScanResult> + Send + 'static,
    ) -> AnalysisOutcome {
        let state = Arc::clone(&self.state);
        let id = self.request_id;
        let mut handle = tokio::task::spawn_blocking(move || {
            if let Some(entry) = state.in_flight.get(&id) {
                entry
                    .thread_id
                    .store(crate::thread_dump::os_thread_id(), Ordering::Relaxed);
            }
            let result = work();
            // A long-lived server returns its thread-local caches now and then.
            if id.is_multiple_of(CACHE_CLEAR_EVERY) {
                cleave::clear_all_thread_caches();
            }
            result
        });
        let timeout_secs = self.state.config.analysis_timeout_secs;
        let joined = if timeout_secs == 0 {
            handle.await
        } else {
            match tokio::time::timeout(Duration::from_secs(timeout_secs), &mut handle).await {
                Ok(joined) => joined,
                Err(_) => {
                    self.follow(handle);
                    return AnalysisOutcome::Timeout(timeout_secs);
                }
            }
        };
        match joined {
            Ok(result) => AnalysisOutcome::Ok(result.map(Box::new)),
            Err(e) => AnalysisOutcome::JoinError(e),
        }
    }

    /// Keep the slot and core until a timed-out blocking task returns.
    ///
    /// A blocking thread cannot be stopped, only asked: the cancellation flag,
    /// which cleave polls between members. Until it answers it is still using
    /// a core, so handing its permits back at the timeout would report
    /// capacity the server does not have. The guard follows the thread out
    /// instead, and `stuck_orphans` counts threads still running, not timeouts
    /// there have ever been.
    fn follow<T: Send + 'static>(self, task: tokio::task::JoinHandle<T>) {
        self.cancellation.store(true, Ordering::Release);
        self.state.stuck_orphans.fetch_add(1, Ordering::Relaxed);
        let tasks = Arc::clone(&self.state.tasks);
        tasks.spawn(async move {
            if let Err(error) = task.await {
                tracing::error!(id = self.request_id, %error, "timed-out analysis task panicked");
            }
            self.state.stuck_orphans.fetch_sub(1, Ordering::Relaxed);
            drop(self);
        });
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        // The permit is released with `_permit`.
        self.cancellation.store(true, Ordering::Release);
        self.state.in_flight.remove(&self.request_id);
    }
}

/// Outcome of a blocking analysis awaited with a bound.
///
/// `Ok` boxes the `ScanResult` (≈376 B) so the idle-path variants — `Timeout`
/// and `JoinError` — don't carry that much padding each.
#[derive(Debug)]
pub(super) enum AnalysisOutcome {
    /// Task completed (inner `Result` is the analyzer's result).
    Ok(anyhow::Result<Box<crate::engine::ScanResult>>),
    /// Task join failed (panic, runtime shutdown, etc.).
    JoinError(tokio::task::JoinError),
    /// Task exceeded the configured timeout. The blocking thread keeps running
    /// until cleave observes the cancellation flag, and keeps its slot and
    /// core until then.
    Timeout(u64),
}

/// Whether the model bundle is loaded.
enum Readiness {
    Starting,
    /// Startup failed. The reason is logged; it is never served.
    Failed(String),
    Ready(Arc<ModelResources>),
}

/// How many artifacts may be held in memory outside an analysis slot at once:
/// cached-upload repairs and `refresh=1` downloads. Each holds up to
/// `--max-size-mb`, and nothing else bounds them.
const MAX_BUFFERED: usize = 4;

/// How long an analysis runs before the watchdog and `/_/health` call it
/// long-running.
const LONG_RUNNING_SECS: u64 = 120;

/// How often the watchdog looks for long-running analyses.
const WATCHDOG_INTERVAL: Duration = Duration::from_secs(30);

/// How often the RSS ceiling is checked, independent of request traffic.
const RSS_CHECK_INTERVAL: Duration = Duration::from_secs(10);

/// Every this many requests, the thread that ran one returns cleave's
/// thread-local caches, so a long-lived server does not hold them forever.
const CACHE_CLEAR_EVERY: u64 = 100;

/// How long shutdown waits for analyses no client is waiting on.
/// Comfortably inside systemd's default 90-second stop timeout.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

struct AppState {
    config: ServerConfig,
    /// Process uptime anchor — captured when the app is built, very close to
    /// process start. `/_/health` reports `now - started_at` as `uptime_secs`.
    started_at: Instant,
    readiness: RwLock<Readiness>,
    next_request_id: AtomicU64,
    /// One permit per `--workers` slot. Each analysis holds one for its whole
    /// life; the permit drops with the analysis or with its orphan follower,
    /// so a slot is always released — even on panic or runtime shutdown.
    slots: Arc<Semaphore>,
    /// One permit per rayon thread: every analysis in this process runs on the
    /// same pool.
    cpu: Arc<Semaphore>,
    /// Tasks stuck past the timeout — still occupying a slot until the
    /// blocking thread finally returns. Tracked for observability only.
    stuck_orphans: AtomicUsize,
    /// Serializes `/_/reload` and `/_/update`. The guard is moved into the
    /// blocking work, so it is held for as long as that work runs.
    reload_lock: Arc<tokio::sync::Mutex<()>>,
    overloaded_since: Mutex<Option<Instant>>,
    /// The status `/_/health` last answered, so a probe logs only a change.
    health: Mutex<&'static str>,
    in_flight: dashmap::DashMap<u64, InFlightRequest>,
    /// Machine-wide cores busy between consecutive `/_/stats` reads.
    cpu_busy: CpuBusy,
    /// Raised when the server stops, so background tasks wind down with it.
    /// Dropping the state drops the sender, which they read the same way.
    shutdown: watch::Sender<bool>,
    /// Work no request is waiting on — analyses whose callers hung up, their
    /// index and upload tails — counted so shutdown can drain it.
    tasks: Arc<Tasks>,
    jobs: Jobs,
    /// Analyses in progress, so concurrent requests for the same artifact
    /// share one run instead of each taking a slot. See [`flight`].
    flights: Arc<flight::Flights>,
    /// Bounds the artifacts held outside a slot; see [`MAX_BUFFERED`].
    buffered: Arc<Semaphore>,
    /// Background hopper uploader (`--hopper`); `None` disables result renewal.
    uploader: Option<Arc<crate::upload::Uploader>>,
    /// The corpus behind this server's index. `None` when no hopper is
    /// configured, which leaves a lookup answering from local knowledge alone.
    corpus: Option<Arc<corpus::Corpus>>,
}

/// A poisoned lock means a panic happened while it was held. Everything kept
/// under these locks is replaced or removed whole, so the data stays usable.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl AppState {
    /// Analyses this server can start right now: a slot and a core for each.
    pub(super) fn available_analysis_permits(&self) -> usize {
        self.slots
            .available_permits()
            .min(self.cpu.available_permits())
    }

    /// Analyses running now. Every analysis holds a slot for its whole run;
    /// the CPU gate is narrower than the slots, so it cannot count them.
    pub(super) fn active_tasks(&self) -> usize {
        self.config
            .workers
            .saturating_sub(self.slots.available_permits())
    }

    /// The loaded bundle, or why there is none.
    pub(super) fn resources(&self) -> Result<Arc<ModelResources>, ApiError> {
        match &*self
            .readiness
            .read()
            .unwrap_or_else(PoisonError::into_inner)
        {
            Readiness::Ready(resources) => Ok(Arc::clone(resources)),
            Readiness::Starting => Err(ApiError::starting()),
            Readiness::Failed(_) => Err(ApiError::init_failed()),
        }
    }

    /// The reason startup failed, if it did.
    pub(super) fn init_failure(&self) -> Option<String> {
        match &*self
            .readiness
            .read()
            .unwrap_or_else(PoisonError::into_inner)
        {
            Readiness::Failed(message) => Some(message.clone()),
            _ => None,
        }
    }

    pub(super) fn is_ready(&self) -> bool {
        matches!(
            *self
                .readiness
                .read()
                .unwrap_or_else(PoisonError::into_inner),
            Readiness::Ready(_)
        )
    }

    /// Wrap a loaded model in the request-independent settings every analysis
    /// runs with. The one place a bundle is assembled.
    fn bundle(&self, model: Model, shap: Option<ShapImportance>) -> ModelResources {
        ModelResources {
            model,
            shap,
            interpret: self.config.interpret.clone(),
            fetch: self.config.fetch,
            zip_passwords: self.config.zip_passwords.clone(),
        }
    }

    /// Serve `resources` from now on. Returns whether a bundle was already
    /// being served.
    fn install(&self, resources: ModelResources) -> bool {
        let mut readiness = self
            .readiness
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let was_ready = matches!(*readiness, Readiness::Ready(_));
        *readiness = Readiness::Ready(Arc::new(resources));
        was_ready
    }

    fn fail_startup(&self, message: String) {
        tracing::error!(error = %message, "startup failed");
        *self
            .readiness
            .write()
            .unwrap_or_else(PoisonError::into_inner) = Readiness::Failed(message);
    }

    /// Refuse a request this server cannot take on: startup failed, or memory
    /// is past the ceiling. A server still loading is refused later, where a
    /// slot is claimed, so an answer it already holds can still be served.
    pub(super) async fn admit_request(&self, request_id: u64) -> Result<(), ApiError> {
        if let Some(message) = self.init_failure() {
            tracing::error!(id = request_id, error = %message, "rejected: startup failed");
            return Err(ApiError::init_failed());
        }
        self.check_memory().await
    }

    /// When the server first went over its memory ceiling, if it still is.
    pub(super) fn overload_mark(&self) -> std::sync::MutexGuard<'_, Option<Instant>> {
        lock(&self.overloaded_since)
    }

    /// Check RSS against the ceiling, reclaiming caches once before refusing.
    async fn check_memory(&self) -> Result<(), ApiError> {
        // Throttling disabled: the operator delegated OOM enforcement to an
        // external supervisor.
        let Some(max_rss_bytes) = self.config.max_rss_bytes.map(NonZeroU64::get) else {
            return Ok(());
        };
        let Some(rss) = cleave::memory_tracker::current_rss() else {
            return Ok(());
        };
        if rss <= max_rss_bytes {
            if self.overload_mark().take().is_some() {
                tracing::info!(
                    rss_mb = rss / 1024 / 1024,
                    "memory recovered below threshold"
                );
            }
            return Ok(());
        }

        tracing::info!(
            rss_mb = rss / 1024 / 1024,
            "memory pressure detected, clearing thread-local caches"
        );
        // Awaited before re-reading RSS: a fire-and-forget clear would let the
        // re-read run first, log memory freed that was not, and admit
        // requests an overloaded server cannot service.
        if let Err(e) = tokio::task::spawn_blocking(cleave::clear_all_thread_caches).await {
            tracing::warn!(error = %e, "cache-clear task failed");
        }

        let Some(rss_after) = cleave::memory_tracker::current_rss() else {
            return Ok(());
        };
        if rss_after <= max_rss_bytes {
            self.overload_mark().take();
            tracing::info!(
                rss_before_mb = rss / 1024 / 1024,
                rss_after_mb = rss_after / 1024 / 1024,
                "cache clear freed memory, accepting request"
            );
            return Ok(());
        }

        // Still overloaded. Never terminate; requests are refused until memory
        // drops, and restarting is the operator's call.
        let since = *self.overload_mark().get_or_insert_with(Instant::now);
        tracing::warn!(
            rss_mb = rss_after / 1024 / 1024,
            max_rss_mb = max_rss_bytes / 1024 / 1024,
            overloaded_secs = since.elapsed().as_secs(),
            "server overloaded: high memory usage (even after cache clear)"
        );
        Err(ApiError::overloaded())
    }

    /// Stop background work and cancel every analysis in flight.
    fn stop(&self) {
        self.shutdown.send_replace(true);
        for entry in &self.in_flight {
            entry.cancellation.store(true, Ordering::Release);
        }
    }

    /// Stop, then wait a bounded time for owned background work to wind down —
    /// so an analysis that finished as the server stopped still files its
    /// verdict.
    async fn drain(&self) {
        self.stop();
        if tokio::time::timeout(DRAIN_TIMEOUT, self.tasks.drained())
            .await
            .is_err()
        {
            tracing::warn!(
                remaining = self.tasks.live(),
                timeout_secs = DRAIN_TIMEOUT.as_secs(),
                "shutdown: background work still running past the drain timeout; abandoning it"
            );
        }
    }
}

/// Build the axum [`Router`] and start loading resources in the background.
///
/// The router answers immediately; until the model bundle loads, the health
/// endpoint and the analyze endpoints answer 503. Useful for integration tests
/// that need the app without binding a port. Background tasks stop when the
/// router is dropped.
///
/// # Errors
///
/// Returns an error if the configuration is invalid.
pub async fn build_app(config: &ServerConfig) -> anyhow::Result<Router> {
    Ok(assemble(config.clone())?.0)
}

/// The router and the state behind it. Must run inside a tokio runtime.
fn assemble(config: ServerConfig) -> anyhow::Result<(Router, Arc<AppState>)> {
    config.validate()?;
    tracing::info!(model_dir = %config.model_dir.display(), "starting — resources loading in background");

    // CPU-bound cleave + ONNX work overlaps poorly across many threads, so a
    // smaller pool typically delivers higher aggregate throughput than 1/core.
    let max_concurrent = config.workers;
    let cores = crate::runtime::cleave_concurrency(max_concurrent);
    tracing::info!(max_concurrent, cores, "concurrency limit set");

    let (shutdown, _) = watch::channel(false);
    // Said once here rather than on every analysis: a server nobody gave a
    // hopper still answers, but every verdict it computes dies with the
    // process, and that is worth one line at startup instead of silence.
    let uploader = match config.hopper() {
        Some(url) => Some(Arc::new(crate::upload::Uploader::new(
            url,
            crate::upload::default_worker_name(),
        ))),
        None => {
            tracing::warn!(
                "no --hopper configured: analyzed results are kept in this \
                 process's verdict index only and are never uploaded",
            );
            None
        }
    };
    let corpus = corpus::Corpus::new(config.hopper());
    match &corpus {
        Some(c) => tracing::info!(addresses = %c.addresses(), "lookups defer to the corpus"),
        // Not a warning: a server with no corpus behind it answers from its
        // own index, which is a whole deployment rather than a broken one.
        None => tracing::info!("no hopper configured: lookups answer from the local index alone"),
    }

    let state = Arc::new(AppState {
        started_at: Instant::now(),
        readiness: RwLock::new(Readiness::Starting),
        next_request_id: AtomicU64::new(1),
        slots: Arc::new(Semaphore::new(max_concurrent)),
        cpu: Arc::new(Semaphore::new(cores)),
        cpu_busy: CpuBusy::default(),
        stuck_orphans: AtomicUsize::new(0),
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        overloaded_since: Mutex::new(None),
        health: Mutex::new("starting"),
        flights: Arc::new(flight::Flights::default()),
        in_flight: dashmap::DashMap::new(),
        shutdown,
        tasks: Arc::new(Tasks::default()),
        jobs: Jobs::default(),
        buffered: Arc::new(Semaphore::new(MAX_BUFFERED)),
        uploader,
        corpus,
        config,
    });

    state.tasks.spawn(load_resources(Arc::clone(&state)));
    spawn_watchdog(&state);

    // No ConcurrencyLimitLayer: each analyze handler refuses past capacity
    // with a 429 rather than queueing. Layers apply bottom-up, so the last
    // `.layer()` runs first per request; the ACL runs before the body limit so
    // a rejected peer never gets to upload bytes.
    let app = Router::new()
        .route("/_/health", get(diag::health))
        .route("/_/info", get(diag::info))
        .route("/_/stats", get(diag::stats))
        .route("/_/reload", post(admin::reload))
        .route("/_/update", post(admin::update))
        .route("/_/memory", get(diag::memory_stats))
        .route("/_/requests", get(diag::requests))
        .route("/_/threads", get(diag::threads))
        .route("/v1/lookup", get(v1::v1_lookup))
        .route("/v1/analyze", post(v1::v1_analyze))
        .layer(DefaultBodyLimit::max(state.config.max_body_size))
        .layer(middleware::from_fn_with_state(Arc::clone(&state), acl::acl))
        // Outermost: every request gets an id and an access-log line, including
        // the ones the ACL rejects.
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            access::access_log,
        ))
        .with_state(Arc::clone(&state));

    Ok((app, state))
}

/// Load the model, SHAP data and traits concurrently, then serve them.
///
/// The verdict index opens alongside: it is not needed to become ready, but
/// opening it creates directories and prunes stale ones, which belongs here
/// rather than on the first lookup.
async fn load_resources(state: Arc<AppState>) {
    let init_start = Instant::now();
    tracing::info!("resource loader started (model + SHAP + YARA loading concurrently)");
    state.tasks.spawn(async {
        if let Err(error) = tokio::task::spawn_blocking(crate::lookup::global).await {
            tracing::error!(%error, "opening the verdict index panicked");
        }
    });

    // Each blocking closure reports queue_ms (time waiting for a thread)
    // separately from work_ms (time actually doing I/O and parsing).
    let spawned = Instant::now();
    let model_dir = state.config.model_dir.clone();
    let (thresholds, level) = (state.config.thresholds, state.config.level);
    let model_task = tokio::task::spawn_blocking(move || -> anyhow::Result<Model> {
        let queue_ms = spawned.elapsed().as_millis();
        let t = Instant::now();
        tracing::info!(queue_ms, "loading ONNX model and feature spec");
        let model = Model::load(&model_dir, thresholds, level)?;
        tracing::info!(
            queue_ms,
            work_ms = t.elapsed().as_millis(),
            spec_version = model.spec().version(),
            features = model.spec().total_features(),
            "ONNX model loaded",
        );
        Ok(model)
    });
    let shap_dir = state.config.model_dir.clone();
    let shap_task = tokio::task::spawn_blocking(move || load_shap(&shap_dir));
    let slow_rule_ms = state.config.slow_rule_ms;
    let yara_task = tokio::task::spawn_blocking(move || -> Result<(), String> {
        let queue_ms = spawned.elapsed().as_millis();
        let t = Instant::now();
        tracing::info!(queue_ms, "YARA warmup started");
        // The traits tree is the rule set every analysis runs against. Resolve
        // it before reporting ready: a server that answers `/_/health` with
        // "ok" while failing every analysis on a missing traits directory is
        // worse than one that never starts.
        let traits = cleave::traits_repo::try_resolve().map_err(|e| format!("{e:#}"))?;
        tracing::info!(dir = %traits.display(), "cleave traits resolved");
        let opts = cleave::AnalysisOptions {
            compact_members: true,
            slow_rule_ms,
            ..Default::default()
        };
        // A warmup: its result is not the point, the compiled rules are.
        if let Err(error) = cleave::analyze_file(std::path::Path::new("/dev/null"), &opts) {
            tracing::debug!(error = format!("{error:#}"), "YARA warmup analysis failed");
        }
        tracing::info!(
            queue_ms,
            work_ms = t.elapsed().as_millis(),
            "YARA warmup complete",
        );
        Ok(())
    });

    match tokio::join!(model_task, shap_task, yara_task) {
        (Ok(Ok(model)), Ok(shap), Ok(Ok(()))) => {
            let spec_version = model.spec().version();
            let features = model.spec().total_features();
            let shap_loaded = shap.is_some();
            state.install(state.bundle(model, shap));
            tracing::info!(
                total_ms = init_start.elapsed().as_millis(),
                spec_version,
                features,
                shap_loaded,
                "server ready",
            );
        }
        (Ok(Err(e)), _, _) => state.fail_startup(format!("failed to load model: {e:#}")),
        (Err(e), _, _) => state.fail_startup(format!("model load task panicked: {e}")),
        (_, Err(e), _) => state.fail_startup(format!("shap load task panicked: {e}")),
        (_, _, Ok(Err(e))) => state.fail_startup(format!("traits unavailable: {e}")),
        (_, _, Err(e)) => state.fail_startup(format!("yara warmup task panicked: {e}")),
    }
}

/// SHAP importances; `None` when the bundle ships none or its file is
/// unusable — explanations are optional, so neither stops the server.
fn load_shap(model_dir: &std::path::Path) -> Option<ShapImportance> {
    let t = Instant::now();
    let shap = ShapImportance::load_or_warn(model_dir);
    tracing::info!(
        work_ms = t.elapsed().as_millis(),
        loaded = shap.is_some(),
        "SHAP data"
    );
    shap
}

/// Periodically log long-running in-flight requests. Never cancels or
/// terminates anything: `--analysis-timeout` already signalled cooperative
/// cancellation (see [`RequestGuard::run`]), so a request still here with its
/// flag raised is an orphan cleave has not yet noticed, and is logged as one.
///
/// Holds the state weakly and stops on shutdown, so it never outlives the app.
fn spawn_watchdog(state: &Arc<AppState>) {
    let weak = Arc::downgrade(state);
    let mut stop = state.shutdown.subscribe();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(WATCHDOG_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = interval.tick() => {}
                // Shutdown raised, or the state (and with it the sender) gone.
                _ = stop.changed() => return,
            }
            let Some(state) = weak.upgrade() else {
                return;
            };
            if state.in_flight.is_empty() {
                continue;
            }
            let active = state.active_tasks();
            let stuck = state.stuck_orphans.load(Ordering::Relaxed);
            for entry in &state.in_flight {
                let elapsed_secs = entry.started_at.elapsed().as_secs();
                if elapsed_secs < LONG_RUNNING_SECS {
                    continue;
                }
                let phase = entry.phase.current();
                let tid = entry.thread_id.load(Ordering::Relaxed);
                if entry.cancellation.load(Ordering::Acquire) {
                    tracing::error!(
                        request_id = entry.key(),
                        name = %entry.name,
                        elapsed_secs,
                        phase,
                        thread_id = tid,
                        stuck_orphans = stuck,
                        active_tasks = active,
                        "watchdog: timed-out task still running after cancellation",
                    );
                } else {
                    tracing::warn!(
                        request_id = entry.key(),
                        name = %entry.name,
                        elapsed_secs,
                        phase,
                        thread_id = tid,
                        stuck_orphans = stuck,
                        active_tasks = active,
                        "watchdog: long-running task",
                    );
                }
            }
        }
    });
}

/// Start the HTTP server and serve until `SIGINT` or `SIGTERM`.
///
/// Binds the configured address, starts background resource loading, and on
/// a signal stops accepting, lets open requests finish, then drains the work
/// no client is waiting on.
///
/// # Errors
/// Returns an error if the configuration is invalid, the listening socket
/// cannot be bound, or the server fails while serving requests.
pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    // Warm cleave's YARA engine + capability mapper off the rayon pool before
    // the listener binds. The first request's analysis spawns rayon work; if
    // one of those rayon workers is the first to hit `yara_engine()`, init's
    // internal par_iter deadlocks against its peers parked on the OnceLock.
    // Prefetching from a non-rayon thread here avoids the race entirely.
    crate::engine::prefetch_cleave_resources();

    // Server mode processes many files over a long lifetime. Configure jemalloc
    // to aggressively return freed pages to the OS, preventing multi-GB RSS
    // growth from allocator fragmentation across thousands of analyses.
    cleave::memory_tracker::configure_jemalloc_low_memory();

    // Enforces the RSS ceiling on wall-clock time, independent of request
    // traffic: memory can grow between requests (fragmentation, background
    // YARA work). Skipped when throttling is disabled. Sustained overload
    // exits so the supervisor restarts us before the kernel OOM-kills us.
    let _rss_logger = config.max_rss_bytes.map(|limit| {
        cleave::memory_tracker::start_periodic_logging(RSS_CHECK_INTERVAL, limit.get(), || {
            tracing::error!("RSS over the cap for >30s; exiting to avoid an OOM kill");
            std::process::exit(1);
        })
    });

    let (app, state) = assemble(config)?;
    let config = &state.config;

    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    // The startup line is the record of what this process actually is: an
    // operator reading the log after a restart should not have to reconstruct
    // the running configuration from the unit file.
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        pid = std::process::id(),
        bind = %config.bind,
        max_body_mb = config.max_body_size / 1024 / 1024,
        analysis_timeout_secs = config.analysis_timeout_secs,
        allow_cidrs = config.allow_cidrs.len(),
        local_paths = config.local_paths.len(),
        authenticated = config.auth_digest.is_some(),
        "listening (resources loading in background)",
    );

    // An unauthenticated API is open to anyone who can reach the socket. Warn
    // unconditionally — a loopback bind is not evidence of safety, because a
    // Cloudflare tunnel terminates on loopback and puts the whole internet on
    // the other side of it.
    if config.auth_digest.is_none() {
        tracing::warn!(
            "no --token-file: the API is unauthenticated; any peer that reaches the socket can submit work",
        );
    }

    // A loopback bind does not make local-path reads local: a tunnel delivers
    // remote callers over loopback too, and without a token nothing tells them
    // apart.
    if !config.local_paths.is_empty() && config.auth_digest.is_none() {
        tracing::warn!(
            local_paths = config.local_paths.len(),
            "--allow-local-paths without --token-file: any caller reaching loopback, including through a tunnel, can have files under those directories scanned",
        );
    }

    // Operator footgun: setting --allow-cidr while bound to loopback means
    // the CIDR list can never match (no remote peers can connect). Warn so
    // the operator notices before debugging "why is everyone getting 403?".
    if !config.allow_cidrs.is_empty() && config.bind.ip().is_loopback() {
        tracing::warn!(
            bind = %config.bind,
            "--allow-cidr is set but bind address is loopback; remote clients cannot connect (use --bind 0.0.0.0:PORT)",
        );
    }

    let stopping = Arc::clone(&state);
    serve(listener, app, async move {
        shutdown_signal().await;
        // Cancel analyses now, not after the last connection closes: a
        // streamed answer holds its connection open until its analysis ends.
        stopping.stop();
    })
    .await;

    state.drain().await;
    tracing::info!("server shut down");
    Ok(())
}

/// Most connections served at once. Past it, accepting waits: idle sockets
/// cannot exhaust file descriptors, and each analysis holds a slot of its own.
const MAX_CONNECTIONS: usize = 1024;

/// How long a client has to send its request headers.
const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// How long accepting pauses after a failed accept, most likely out of file
/// descriptors, where retrying at once would only spin.
const ACCEPT_RETRY_DELAY: Duration = Duration::from_secs(1);

/// Accept connections and serve `app` on them until `stop` resolves, then
/// wait up to [`DRAIN_TIMEOUT`] for open connections to finish.
///
/// Each request carries its peer address as [`ConnectInfo`], which the ACL
/// reads.
async fn serve(listener: tokio::net::TcpListener, app: Router, stop: impl Future<Output = ()>) {
    use hyper_util::rt::{TokioIo, TokioTimer};
    use tower::ServiceExt as _;

    let connections = Arc::new(Semaphore::new(MAX_CONNECTIONS));
    let graceful = hyper_util::server::graceful::GracefulShutdown::new();
    let mut stop = std::pin::pin!(stop);
    loop {
        // Held across the accept below: that is what makes a full server stop
        // accepting. The semaphore is never closed, so `Err` cannot happen.
        let permit = tokio::select! {
            permit = Arc::clone(&connections).acquire_owned() => match permit {
                Ok(permit) => permit,
                Err(_closed) => break,
            },
            () = &mut stop => break,
        };
        let (stream, peer) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(accepted) => accepted,
                Err(error) => {
                    drop(permit);
                    tracing::warn!(%error, "accept failed");
                    tokio::time::sleep(ACCEPT_RETRY_DELAY).await;
                    continue;
                }
            },
            () = &mut stop => break,
        };
        let service = app
            .clone()
            .map_request(move |mut req: axum::http::Request<_>| {
                req.extensions_mut().insert(ConnectInfo(peer));
                req
            });
        let conn = hyper::server::conn::http1::Builder::new()
            .timer(TokioTimer::new())
            .header_read_timeout(HEADER_READ_TIMEOUT)
            .serve_connection(
                TokioIo::new(stream),
                hyper_util::service::TowerToHyperService::new(service),
            );
        let conn = graceful.watch(conn);
        tokio::spawn(async move {
            let _permit = permit;
            if let Err(error) = conn.await {
                tracing::debug!(%peer, %error, "connection ended");
            }
        });
    }
    drop(listener);
    if tokio::time::timeout(DRAIN_TIMEOUT, graceful.shutdown())
        .await
        .is_err()
    {
        tracing::warn!(
            timeout_secs = DRAIN_TIMEOUT.as_secs(),
            "shutdown: connections still open past the drain timeout; closing them"
        );
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = signal::ctrl_c().await {
            tracing::warn!(error = %e, "failed to install Ctrl+C handler");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("received SIGINT"),
        () = terminate => tracing::info!("received SIGTERM"),
    }
}
