//! HTTP request handlers for the litmus API server.

use axum::extract::{Extension, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tempfile::Builder as TempBuilder;

use super::AppState;
use super::access::{RequestId, Subject, with_subject};
use super::acl::Trusted;
use super::corpus::{self, Reached};
use super::decision;
use super::flight::{Flight, FlightKey, Outcome};

/// Assemble a JSON object body from static keys.
///
/// Used by [`health`], which builds one body from a fixed public set and then
/// adds privileged keys for trusted requests.
fn object(
    pairs: impl IntoIterator<Item = (&'static str, serde_json::Value)>,
) -> serde_json::Map<String, serde_json::Value> {
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

/// Outcome of awaiting a `tokio::spawn_blocking` analysis task with a bound.
///
/// `Ok` boxes the `ScanResult` (≈376 B) so the idle-path variants — `Timeout`
/// and `JoinError` — don't carry that much padding each.
#[derive(Debug)]
enum AnalysisOutcome {
    /// Task completed (inner `Result` is the analyzer's result).
    Ok(anyhow::Result<Box<crate::engine::ScanResult>>),
    /// Task join failed (panic, runtime shutdown, etc.).
    JoinError(tokio::task::JoinError),
    /// Task exceeded the configured timeout. Slot is released; the blocking
    /// thread keeps running until cleave observes the cancellation flag.
    Timeout(u64),
}

/// Await a blocking analysis task with an optional per-request timeout.
///
/// On timeout, sets the cancellation flag (so cleave will exit at its next
/// checkpoint), increments `stuck_orphans` for observability, and returns
/// `Timeout`. The blocking task is **not** aborted — tokio can't force-stop
/// a blocking thread — but the HTTP slot is released so new work can land.
async fn await_with_timeout(
    handle: tokio::task::JoinHandle<anyhow::Result<crate::engine::ScanResult>>,
    timeout_secs: u64,
    cancellation: &AtomicBool,
    stuck_orphans: &AtomicUsize,
) -> AnalysisOutcome {
    if timeout_secs == 0 {
        return match handle.await {
            Ok(r) => AnalysisOutcome::Ok(r.map(Box::new)),
            Err(e) => AnalysisOutcome::JoinError(e),
        };
    }
    match tokio::time::timeout(Duration::from_secs(timeout_secs), handle).await {
        Ok(Ok(r)) => AnalysisOutcome::Ok(r.map(Box::new)),
        Ok(Err(e)) => AnalysisOutcome::JoinError(e),
        Err(_) => {
            cancellation.store(true, Ordering::Release);
            stuck_orphans.fetch_add(1, Ordering::Relaxed);
            AnalysisOutcome::Timeout(timeout_secs)
        }
    }
}

/// Return a platform-appropriate thread ID for the calling thread.
/// On Linux this is the TID (matches /proc/self/task/), on macOS/FreeBSD
/// it's the pthread ID, and elsewhere falls back to 0.
fn current_thread_id() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `gettid` takes no arguments and cannot fail; the tid is positive.
        u64::try_from(unsafe { libc::syscall(libc::SYS_gettid) }).unwrap_or(0)
    }
    #[cfg(target_os = "macos")]
    {
        let mut tid: u64 = 0;
        // pthread_threadid_np gives a stable, unique-per-process thread ID on macOS.
        unsafe { libc::pthread_threadid_np(0, &mut tid) };
        tid
    }
    #[cfg(target_os = "freebsd")]
    {
        // thr_self writes the lwpid into the provided pointer.
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

/// Classify an analysis error into its HTTP status and JSON body. Returns the
/// status alongside the response so callers can log it without reclassifying.
fn analysis_error_response(error: &anyhow::Error) -> (StatusCode, Response) {
    let (status, body) = analysis_error_body(error);
    (status, (status, Json(body)).into_response())
}

/// An anyhow error flattened to its whole `context: cause: root cause` chain.
///
/// `Display` on an `anyhow::Error` prints only the outermost context, so a log
/// line reads "cleave analysis of left-pad-1.3.0.tgz" and never says *why* it
/// failed. The alternate form carries every link, which is what makes a failed
/// request diagnosable from the server log alone.
fn error_chain(error: &anyhow::Error) -> String {
    format!("{error:#}")
}

/// The status and JSON body an analysis failure becomes.
///
/// Split out from [`analysis_error_response`] because a shared analysis has to
/// *store* its failure — `anyhow::Error` is not `Clone`, so the leader renders
/// it once and every follower replays the result.
fn analysis_error_body(error: &anyhow::Error) -> (StatusCode, serde_json::Value) {
    let message = error.root_cause().to_string();
    let detail = error_chain(error);
    // Classify over the whole chain, not just the root cause: "unsupported file
    // type" is often a middle link wrapped around an io error, and classifying
    // on the root alone reports those as 500.
    let status = classify_analysis_error(&detail);
    let body = if detail == message {
        serde_json::json!({ "error": message })
    } else {
        serde_json::json!({ "error": message, "detail": detail })
    };
    (status, body)
}

/// Where this result's analysis came from, for the completion log line.
///
/// `cached` means cleave replayed the whole report from its on-disk cache
/// (SQLite, keyed by content digest, options, and traits revision) rather than
/// running the pipeline. It survives restarts, so a fast response is not
/// evidence of a warm process. A request that instead rode *another request's*
/// in-flight run reports `shared=true` on its access line — that path never
/// reaches here, because only the leader logs the completion.
fn analysis_source(result: &ScanResult) -> &'static str {
    if result.analysis_cached {
        "cached"
    } else {
        "fresh"
    }
}

/// Where this result's LLM verdict came from, for the completion log line.
///
/// `--interpret` dominates a request's wall time when it actually queries the
/// endpoint and costs nothing when the verdict is replayed from the prompt
/// cache — a minute versus a tenth of a second on the same sample. Naming the
/// source turns that difference from a timing anomaly into a fact on the line.
/// `None` when no pass ran, which omits the field.
fn llm_source(interpretation: Option<&crate::interpret::Interpretation>) -> Option<&'static str> {
    let interpretation = interpretation?;
    Some(if interpretation.error.is_some() {
        "failed"
    } else if interpretation.cached {
        "cached"
    } else {
        "queried"
    })
}

/// Take the resources snapshot and the analysis slot a flight leader needs, or
/// the outcome to publish instead of running.
///
/// Followers claim neither: riding the leader's run rather than taking a slot
/// of their own is the whole point of sharing it.
fn claim_slot(
    state: &Arc<AppState>,
    request_id: u64,
    key: &FlightKey,
) -> Result<
    (
        Arc<super::ModelResources>,
        tokio::sync::OwnedSemaphorePermit,
    ),
    Outcome,
> {
    let resources = match state.resources.read() {
        Ok(lock) => match lock.as_ref() {
            Some(r) => Arc::clone(r),
            None => {
                tracing::debug!(id = request_id, "rejected: resources not yet loaded");
                return Err(Outcome::rendered(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Server starting up",
                ));
            }
        },
        Err(e) => {
            tracing::error!("read lock poisoned: {e}");
            return Err(Outcome::rendered(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal error",
            ));
        }
    };
    // Refusing past full, rather than queueing. A queue here would hide the one
    // fact the router most needs: a rejection is information, delivered in
    // milliseconds, and beamline answers it by promoting the next arm onto a
    // worker that has room. A queued request looks identical to a slow one from
    // outside, so the router keeps choosing a saturated worker while a idle one
    // waits — and the queued analysis still runs after somebody else has
    // already answered, which is the duplicate work single-flight exists to
    // prevent.
    let Ok(permit) = Arc::clone(&state.slots).try_acquire_owned() else {
        let max = state.max_concurrent_tasks;
        tracing::warn!(id = request_id, key = %key, max, "rejecting: at capacity");
        return Err(Outcome::rendered(
            StatusCode::TOO_MANY_REQUESTS,
            format!("At capacity ({max}/{max} active analyses)"),
        ));
    };
    Ok((resources, permit))
}

/// Fold an awaited analysis into the outcome every attached request shares.
fn flight_outcome(
    result: AnalysisOutcome,
    request_id: u64,
    elapsed_ms: u64,
    key: &FlightKey,
    state: &Arc<AppState>,
    persist: bool,
) -> Outcome {
    let uploader = state.uploader.as_ref();
    match result {
        AnalysisOutcome::Ok(Ok(scan_result)) => {
            tracing::info!(
                id = request_id,
                key = %key,
                elapsed_ms,
                classification = %scan_result.classification,
                probability = scan_result.probability,
                analysis = analysis_source(&scan_result),
                llm = llm_source(scan_result.interpretation.as_ref()),
                // Where this verdict goes next. `queued` hands it to the
                // uploader thread, whose own line reports whether it landed;
                // `disabled` means the server was started without --hopper and
                // the answer lives only in this process's index.
                hopper = if !persist {
                    "policy-specific"
                } else if uploader.is_some() {
                    "queued"
                } else {
                    "disabled"
                },
                "<-- 200 OK",
            );
            // Record the completion first: size and duration are what a
            // router averages to decide whether this server is a good choice
            // for the next artifact of a given size.
            state.jobs_completed.fetch_add(1, Ordering::Relaxed);
            state
                .job_bytes_total
                .fetch_add(scan_result.size_bytes, Ordering::Relaxed);
            let micros = elapsed_ms.saturating_mul(1_000);
            state.job_micros_total.fetch_add(micros, Ordering::Relaxed);
            // Routing predicts the cost of work this server has *not* done, so
            // only fresh analyses feed the figures a router reads. A cache hit
            // is real and worth reporting, but it predicts nothing about the
            // next unseen artifact.
            if scan_result.analysis_cached {
                state.job_cached.record(micros);
            } else {
                state.job_overall.record(micros);
                state.job_buckets[super::size_bucket(scan_result.size_bytes)].record(micros);
            }
            // Also by PURL type, when this job was named by one. That is the
            // only cost signal a router has before dispatch for `?purl=` work,
            // which is most of the traffic this fleet serves.
            if let Some(purl) = key.purl()
                && !scan_result.analysis_cached
            {
                state.job_types[super::purl_type_bucket(purl)].record(micros);
            }
            if persist {
                index_verdict(&scan_result, key.purl());
            }
            // Renew the verdict on hopper too, so it outlives this process and
            // this request. A caller that hangs up — or a proxy that gives up
            // at its own read timeout on a long run — still finds the answer
            // waiting on its next lookup, because the analysis was never the
            // connection's to lose. One clone per analysis, against a run
            // measured in seconds.
            //
            // Ordinarily that's under this result's own sha256 — but
            // `classify_purl`'s registry-metadata fallback sets `hopper_route`
            // to redirect onto a real sha hopper already holds, or to suppress
            // the post entirely when hopper holds nothing for the coordinate at
            // all. See [`crate::engine::HopperRoute`].
            let hopper_sha = match &scan_result.hopper_route {
                crate::engine::HopperRoute::Suppress => None,
                crate::engine::HopperRoute::Redirect(sha) => Some(sha.clone()),
                crate::engine::HopperRoute::Normal => Some(scan_result.sha256.clone()),
            };
            if persist
                && let Some(uploader) = uploader
                && let Some(hopper_sha) = hopper_sha
            {
                uploader.submit(
                    hopper_sha,
                    key.purl().map(str::to_owned),
                    (*scan_result).clone().into_envelope(),
                );
            }
            Outcome::Report(scan_result)
        }
        AnalysisOutcome::Ok(Err(e)) => {
            let (status, body) = analysis_error_body(&e);
            tracing::warn!(id = request_id, key = %key, elapsed_ms, status = status.as_u16(), error = %error_chain(&e), "<-- analysis failed");
            Outcome::Rendered { status, body }
        }
        AnalysisOutcome::JoinError(e) => {
            tracing::warn!(id = request_id, key = %key, elapsed_ms, error = %e, "<-- 500 task join error (panic?)");
            Outcome::rendered(StatusCode::INTERNAL_SERVER_ERROR, "Internal error")
        }
        AnalysisOutcome::Timeout(secs) => {
            tracing::warn!(id = request_id, key = %key, elapsed_ms, timeout_secs = secs, "<-- 504 analysis timeout");
            Outcome::Rendered {
                status: StatusCode::GATEWAY_TIMEOUT,
                body: serde_json::json!({ "error": "analysis timeout", "timeout_secs": secs }),
            }
        }
    }
}

/// Render the shared outcome as this request's response. `elapsed_ms` is the
/// caller's own wall time, so a follower reports how long *it* waited, `shared`
/// marks the response as one that rode another request's analysis, and `key`
/// names the artifact on the access line.
fn flight_response(outcome: &Outcome, elapsed_ms: u64, shared: bool, key: &FlightKey) -> Response {
    let mut resp = match outcome {
        Outcome::Report(result) => {
            let mut resp = Json(result.envelope_ref()).into_response();
            resp.headers_mut().insert("X-Total-Ms", elapsed_ms.into());
            resp
        }
        Outcome::Rendered { status, body } => (*status, Json(body)).into_response(),
    };
    if shared {
        resp.extensions_mut().insert(super::access::Shared);
    }
    resp.extensions_mut().insert(Subject::from(key));
    resp
}

/// Record what this analysis found, so a later lookup of the same artifact is
/// answerable without re-running it.
///
/// Off the response path: the write is small, but it opens the index on first
/// use (a `create_dir_all` plus a prune of stale ruleset namespaces), which has
/// no business happening on the reactor. Best-effort — a lookup that misses is
/// a normal answer, so nothing here is worth failing a request over.
fn index_verdict(result: &crate::engine::ScanResult, purl: Option<&str>) {
    let verdict = crate::lookup::Verdict::from_scan(result, purl);
    tokio::task::spawn_blocking(move || {
        if let Some(index) = crate::lookup::global() {
            index.put(&verdict);
        }
    });
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({"error": message.into()}))).into_response()
}

fn classify_analysis_error(message: &str) -> StatusCode {
    let normalized = message.to_ascii_lowercase();

    if normalized.contains("unsupported file type")
        || normalized.contains("unsupported archive type")
        || normalized.contains("unsupported compression")
    {
        StatusCode::UNSUPPORTED_MEDIA_TYPE
    } else if normalized.contains("archive is encrypted but no passwords configured")
        || normalized.contains("invalid ")
        || normalized.contains("not a valid ")
        || normalized.contains("truncated")
        || normalized.contains("corrupt")
        || normalized.contains("unexpected end of")
        || normalized.contains("too small")
        || normalized.contains("out of bounds")
        || normalized.contains("empty package.json")
        || normalized.contains("maximum archive depth")
        || normalized.contains("maximum decode depth")
        || normalized.contains("exceeded maximum")
        || normalized.contains("file count limit exceeded")
        || normalized.contains("file name too long")
    {
        StatusCode::UNPROCESSABLE_ENTITY
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}
use crate::engine::ScanResult;
use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::Model;
use crate::system_load_avg;

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
    let load_avg = system_load_avg();

    if let Ok(init_error) = state.init_error.read()
        && let Some(message) = init_error.as_ref()
    {
        tracing::error!("GET /_/health -> 503 (failed: {message})");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "failed",
                "reason": "initialization_failed",
                "uptime_secs": uptime_secs,
            })),
        )
            .into_response();
    }

    if !state.ready.load(std::sync::atomic::Ordering::Acquire) {
        tracing::debug!("GET /_/health -> 503 (starting)");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "starting",
                "uptime_secs": uptime_secs,
            })),
        )
            .into_response();
    }

    let rss_bytes = cleave::memory_tracker::current_rss();
    let rss_mb = rss_bytes.map(|b| b / 1024 / 1024);
    let max_rss_mb = state.max_rss_bytes.map(|n| n.get() / 1024 / 1024);
    let active_tasks = state
        .max_concurrent_tasks
        .saturating_sub(state.slots.available_permits());
    let overloaded = match (rss_bytes, state.max_rss_bytes) {
        (Some(rss), Some(limit)) => rss > limit.get(),
        _ => false,
    };

    if overloaded {
        tracing::warn!("GET /_/health -> 503 (degraded, rss={rss_mb:?}MB)");
        let mut body = object([
            ("status", "degraded".into()),
            ("reason", "memory_pressure".into()),
            ("rss_mb", rss_mb.into()),
            ("max_rss_mb", max_rss_mb.into()),
            ("active_tasks", active_tasks.into()),
            ("load_avg", load_avg.into()),
            ("uptime_secs", uptime_secs.into()),
        ]);
        if trusted {
            body.insert("rayon_threads".into(), rayon::current_num_threads().into());
        }
        return (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
    }
    let max_tasks = state.max_concurrent_tasks;
    let stuck_orphans = state
        .stuck_orphans
        .load(std::sync::atomic::Ordering::Relaxed);

    // Tasks running longer than 120s — visible in /_/requests with full phase
    // detail. The count is always computed (it is cheap: `in_flight` holds at
    // most one entry per analysis slot, and the count feeds the log line), but
    // each entry names the sample being analysed, so the detail is built only
    // when it will actually be served.
    let now = Instant::now();
    let mut long_running_count = 0usize;
    let mut long_running: Vec<serde_json::Value> = Vec::new();
    for entry in &state.in_flight {
        let elapsed_secs = now.duration_since(entry.started_at).as_secs();
        if elapsed_secs < 120 {
            continue;
        }
        long_running_count += 1;
        if trusted {
            long_running.push(serde_json::json!({
                "request_id": entry.key(),
                "name": entry.name,
                "elapsed_secs": elapsed_secs,
                "phase": entry.phase.get(),
                "thread_id": entry.thread_id.load(std::sync::atomic::Ordering::Relaxed),
            }));
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
    // /analyze endpoint still rejects with 503 when active >= max, so clients
    // back off correctly without /_/health pretending the server is unhealthy.
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

    let mut body = object([
        ("status", if saturated { "saturated" } else { "ok" }.into()),
        ("rss_mb", rss_mb.into()),
        ("max_rss_mb", max_rss_mb.into()),
        ("active_tasks", active_tasks.into()),
        ("max_concurrent_tasks", max_tasks.into()),
        ("load", load.into()),
        ("load_avg", load_avg.into()),
        ("uptime_secs", uptime_secs.into()),
    ]);
    if saturated {
        body.insert("reason".into(), "thread_pool_saturated".into());
    }
    if trusted {
        body.insert("stuck_orphans".into(), stuck_orphans.into());
        body.insert("long_running_tasks".into(), long_running.into());
        body.insert("rayon_threads".into(), rayon::current_num_threads().into());
        if let Some((name, elapsed_secs)) = oldest {
            body.insert(
                "oldest_task".into(),
                serde_json::json!({ "name": name, "elapsed_secs": elapsed_secs }),
            );
        }
    }
    Json(body).into_response()
}

/// GET /_/info — static server capacity and version info.
///
/// Returned by litmus on startup so clients (e.g. hopper) can size their
/// per-node worker pools without hand-configuring slot counts, and so they
/// can compare model/traits commits across nodes for drift detection.
/// Always 200, independent of readiness — readiness is reported by /_/health.
pub(super) async fn info(State(state): State<Arc<AppState>>) -> Response {
    let cpus = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(0);
    let total_mem_mb = cleave::memory_tracker::total_memory()
        .map(|bytes| bytes / 1024 / 1024)
        .unwrap_or(0);
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "slots": state.max_concurrent_tasks,
        "cpus": cpus,
        "max_upload_mb": state.max_upload_bytes / 1024 / 1024,
        "max_rss_mb": state.max_rss_bytes.map(|n| n.get() / 1024 / 1024),
        "total_mem_mb": total_mem_mb,
        "model_commit": crate::models_repo::version(),
        "traits_commit": cleave::traits_repo::version(),
        // Idle worker: whether it is configured, whether it is running, and
        // whether it is currently standing aside for a request. Published
        // because its absence is otherwise invisible — it either starts or it
        // does not, and until this existed the only evidence was a log line
        // that never appeared.
        "idle_worker": {
            "slots": state.idle_worker_slots,
            "hopper": state.hopper.is_some(),
            "started": state.idle_worker_started.load(Ordering::Relaxed),
            "paused": state
                .idle_pause
                .as_ref()
                .is_some_and(|p| p.load(Ordering::Relaxed)),
            "interactive_in_flight": state.in_flight.len(),
        },
    }))
    .into_response()
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
    let in_flight = state.in_flight.len();
    let free = state.slots.available_permits();
    let uploads = state.uploader.as_ref().map(|u| u.stats());
    let started = state.jobs_started.load(Ordering::Relaxed);
    let completed = state.jobs_completed.load(Ordering::Relaxed);
    let bytes = state.job_bytes_total.load(Ordering::Relaxed);
    let micros = state.job_micros_total.load(Ordering::Relaxed);
    // Aged, so a past incident stops steering present routing.
    let recent_n = state.job_overall.count.load(Ordering::Relaxed);
    let recent_micros = state.job_overall.micros.load(Ordering::Relaxed);
    let cached_n = state.job_cached.count.load(Ordering::Relaxed);
    let cached_micros = state.job_cached.micros.load(Ordering::Relaxed);
    let lookup_n = state.lookups.count.load(Ordering::Relaxed);
    let lookup_micros = state.lookups.micros.load(Ordering::Relaxed);
    // Averages over completed jobs only: a job still running has contributed no
    // duration, and dividing by `started` would report every busy server as
    // faster than it is.
    let avg = |total: u64| (completed > 0).then(|| total / completed);
    Json(serde_json::json!({
        // Saturation. The best routing signal available, because it is current
        // rather than lagging: a latency average still reports health for the
        // minute after a server takes on four large archives.
        // Where lookups this index could not answer actually went. A fleet
        // quietly reading from the primary because the replica stopped
        // answering is otherwise indistinguishable from one reading the
        // replica, until the primary's load says so.
        "corpus": state.corpus.as_ref().map(|c| c.stats()),
        "slots": state.max_concurrent_tasks,
        "slots_free": free,
        "in_flight": in_flight,
        // The basis `slots` was sized on, so a caller can read `load1` against
        // the right denominator. `/_/info` reports logical CPUs; slots are
        // sized on physical, and halving the denominator would halve the
        // apparent pressure.
        //
        // Worth reporting because `in_flight` is this process's own count and
        // nothing else's. A scan host commonly runs the pull worker beside this
        // server, and may run an ad-hoc analysis too: measured on a 64-core
        // node, `slots_free=64 in_flight=0` alongside `load1=50`. Every word of
        // that is true and the impression is wrong, and only `load1` against a
        // real core count corrects it.
        "physical_cpus": cleave::memory_tracker::physical_cpu_count(),
        "overloaded": state.overloaded_since.lock().is_ok_and(|g| g.is_some()),
        "stuck_orphans": state.stuck_orphans.load(Ordering::Relaxed),

        // What this server has actually done. More useful for routing than a
        // load average, which is a whole-host number that folds in every other
        // tenant on the box — real on a shared host, but not a statement about
        // this scanner. These are.
        "uptime_secs": state.started_at.elapsed().as_secs(),
        "jobs_started": started,
        "jobs_completed": completed,
        // Begun and never finished: timeouts, panics, and clients that hung up
        // mid-analysis. Non-zero and climbing is the shape of a sick server.
        "jobs_unfinished": started.saturating_sub(completed).saturating_sub(in_flight as u64),
        "avg_job_bytes": avg(bytes),
        // Fresh analyses only: what this server costs on work it has not seen.
        "avg_job_ms": (recent_n > 0).then(|| recent_micros / recent_n / 1_000),
        // Answered from this server's own index. Reported so an operator can
        // see the hit rate, and kept out of the figures above so it cannot
        // flatter a server into looking fast at work it never did.
        "avg_job_ms_cached": (cached_n > 0).then(|| cached_micros / cached_n / 1_000),
        "cached_samples": cached_n,
        // `/lookup` service time — an index probe, near-constant in the size of
        // the artifact, and three orders of magnitude below an analysis.
        // Windowed p80 for the fleet-wide view, beside the lifetime means.
        "recent": state.job_overall.recent_json(),
        "recent_lookup": state.lookups.recent_json(),
        "latency_window_secs": super::latency_window_secs(),
        "avg_lookup_ms": (lookup_n > 0).then(|| lookup_micros / lookup_n / 1_000),
        // Microseconds too: a healthy index probe rounds to 0ms, and a routing
        // signal that is always zero is no signal at all.
        "avg_lookup_us": (lookup_n > 0).then(|| lookup_micros / lookup_n),
        "lookup_samples": lookup_n,
        // Lifetime, for the operator rather than the router.
        "avg_job_ms_lifetime": avg(micros).map(|us| us / 1_000),
        "avg_job_samples": recent_n,
        // The same average, split by PURL type. A `?purl=` request has no size
        // until the artifact is fetched, so this is what a router can compare
        // on when the choice still matters. Types differ by more than an order
        // of magnitude: a golang pseudo-version is a repository clone.
        "avg_job_ms_by_type": super::PURL_TYPE_NAMES
            .iter()
            .zip(state.job_types.iter())
            .map(|(name, b)| {
                let n = b.count.load(Ordering::Relaxed);
                let ms = (n > 0).then(|| b.micros.load(Ordering::Relaxed) / n / 1_000);
                // `recent` is the one a router should read: a p80 over the last
                // hour rather than a mean over the last few hundred jobs.
                ((*name).to_string(), serde_json::json!({
                    "jobs": n, "avg_ms": ms, "recent": b.recent_json(),
                }))
            })
            .collect::<serde_json::Map<String, serde_json::Value>>(),

        // The same average, split by input size. A caller that knows how big
        // the artifact is should compare servers at that size, not overall.
        "avg_job_ms_by_size": super::SIZE_BUCKET_NAMES
            .iter()
            .zip(state.job_buckets.iter())
            .map(|(name, b)| {
                let n = b.count.load(Ordering::Relaxed);
                let ms = (n > 0).then(|| b.micros.load(Ordering::Relaxed) / n / 1_000);
                ((*name).to_string(), serde_json::json!({
                    "jobs": n, "avg_ms": ms, "recent": b.recent_json(),
                }))
            })
            .collect::<serde_json::Map<String, serde_json::Value>>(),

        // Memory headroom. A server near its ceiling is about to pause
        // admission; a router should move away before that, not discover it by
        // timing out.
        "rss_mb": cleave::memory_tracker::current_rss().map(|b| b / 1024 / 1024),
        "max_rss_mb": state.max_rss_bytes.map(|n| n.get() / 1024 / 1024),
        "load1": crate::system_load_avg(),

        // Capability, not speed. A server without 7z cannot read a DMG, so it
        // returns a *weaker verdict* rather than a slower one — routing on
        // latency alone would quietly pick it. Likewise an artifact past
        // max_upload_mb is refused, so sending it wastes a whole round trip.
        "tools": crate::tools::available_names(),
        "max_upload_mb": state.max_upload_bytes / 1024 / 1024,

        // Verdict comparability. A scanner on stale traits produces an answer
        // that should not be cached as authoritative alongside a current one.
        "traits_commit": cleave::traits_repo::version(),
        "model_commit": crate::models_repo::version(),
        "ready": state.ready.load(Ordering::Relaxed),

        // Whether the verdicts it computes are actually reaching hopper.
        // `failed` climbing means work is being done and then lost — the exact
        // failure that went unnoticed for weeks.
        "uploads": uploads.map(|u| serde_json::json!({
            "pending": u.pending,
            "capacity": u.capacity,
            "failed": u.failed,
            "uploaded": u.uploaded,
        })),

        // Whether spare capacity is genuinely spare.
        "idle_worker": {
            "slots": state.idle_worker_slots,
            "started": state.idle_worker_started.load(Ordering::Relaxed),
            "paused": state
                .idle_pause
                .as_ref()
                .is_some_and(|p| p.load(Ordering::Relaxed)),
        },
    }))
    .into_response()
}

/// Query string for `GET /lookup`. Exactly one of `sha256`, `purl`, or `url`.
///
/// Both identifiers travel as query parameters. A PURL's own grammar carries
/// `/`, `?` and `#` — `pkg:npm/@scope/name@1.0.0?arch=x64` in a path segment
/// would have everything from the `?` parsed as the *URL's* query and a
/// `#subpath` dropped by the client, silently keying on a different package —
/// and a digest gains nothing from a prettier URL that the other key cannot
/// have too.
#[derive(Debug, Default, serde::Deserialize)]
pub(super) struct LookupQuery {
    sha256: Option<String>,
    purl: Option<String>,
    url: Option<String>,
}

/// GET /lookup?sha256=… | ?purl=… — what we already know about an artifact.
///
/// Never analyzes: no slot, no fetch, and it answers while the model is still
/// loading. A caller that gets `404 unknown sample` asks for a real analysis
/// with `/analyze` or `/analyze-purl`.
pub(super) async fn lookup(
    State(state): State<Arc<AppState>>,
    Query(q): Query<LookupQuery>,
) -> Response {
    let started = Instant::now();
    let response = lookup_inner(&state, &q);
    // Timed here rather than inside each arm so every answer counts — a
    // rejection is as much a measure of this endpoint's speed as a hit.
    state
        .lookups
        .record(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
    response
}

fn lookup_inner(state: &Arc<AppState>, q: &LookupQuery) -> Response {
    let sha = q.sha256.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let purl = q.purl.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let url = q.url.as_deref().map(str::trim).filter(|s| !s.is_empty());
    // Every arm names its subject, so the request's access line says which
    // artifact was asked about — including the arms that reject, where the key
    // is the only way to tell a caller's bug from a caller's typo.
    match (sha, purl, url) {
        (None, None, None) => error_response(
            StatusCode::BAD_REQUEST,
            "provide sha256, purl, url, or both",
        ),
        (_, Some(_), Some(_)) | (Some(_), None, Some(_)) => error_response(
            StatusCode::BAD_REQUEST,
            "provide one locator plus an optional sha256",
        ),
        (Some(sha), Some(purl), None) => lookup_by_both(state, sha, purl),
        (Some(sha), None, None) => with_subject(lookup_by_sha(state, sha), Subject::sha256(sha)),
        (None, Some(purl), None) => lookup_by_purl(state, purl),
        (None, None, Some(url)) => lookup_by_url(url),
    }
}

fn lookup_by_url(raw: &str) -> Response {
    if !valid_http_url(raw) {
        return with_subject(
            error_response(StatusCode::BAD_REQUEST, "invalid url"),
            Subject::url(raw, None),
        );
    }
    // The legacy lookup route never analyzes or fetches. URL resolution is
    // provided by `/v1/analyze?url=...`; this route can only answer a URL once
    // the durable index has a record keyed by its resolved digest.
    let mut response = error_response(StatusCode::NOT_FOUND, "unknown sample");
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    with_subject(response, Subject::url(raw, None))
}

fn lookup_by_sha(state: &AppState, sha256: &str) -> Response {
    let Some(digest) = burton::parse_sha256_hex(sha256) else {
        return error_response(StatusCode::BAD_REQUEST, "invalid sha256");
    };
    let sha = sha256.to_ascii_lowercase();
    let decision = crate::bloom_repo::global()
        .as_deref()
        .map_or(crate::bloom_repo::Decision::Unknown, |lk| {
            lk.memo_sha256(&digest)
        });
    let verdict = crate::lookup::global().and_then(|index| index.get_sha(&sha));
    lookup_response(state, verdict.as_ref(), decision, &sha, None)
}

/// Answer for an artifact the caller can name both ways.
///
/// Both filters are consulted, because they are cheap — four in-memory probes,
/// already memoized — and because a caller who names both is asserting they are
/// one artifact, which makes each filter evidence about it. A key the other
/// missed is a hit neither would have produced alone, and a disagreement
/// between them lands on `Conflicted` instead of on whichever was asked first.
///
/// The digest stays the identity. Its stored verdict wins outright; the PURL's
/// is accepted only when it describes the same bytes, because a release whose
/// digest has moved is answering about a different artifact than the one asked
/// about. That check costs nothing here — the index already returns the digest
/// it resolved to.
fn lookup_by_both(state: &AppState, sha256: &str, raw: &str) -> Response {
    let Some(digest) = burton::parse_sha256_hex(sha256) else {
        return error_response(StatusCode::BAD_REQUEST, "invalid sha256");
    };
    let purl = match normalize_pkg_purl(raw) {
        Ok(purl) => purl,
        Err(message) => {
            return with_subject(
                error_response(StatusCode::BAD_REQUEST, message),
                Subject::purl(raw, None),
            );
        }
    };
    let sha = sha256.to_ascii_lowercase();

    let decision = crate::bloom_repo::global()
        .as_deref()
        .map_or(crate::bloom_repo::Decision::Unknown, |lk| {
            lk.decide_any(Some(&purl), Some(&digest))
        });

    let index = crate::lookup::global();
    let verdict = pick_verdict(
        index.and_then(|i| i.get_sha(&sha)),
        || index.and_then(|i| i.get_purl(&purl)),
        &sha,
    );

    let response = lookup_response(state, verdict.as_ref(), decision, &sha, Some(&purl));
    with_subject(response, Subject::purl(&purl, Some(&sha)))
}

/// Which stored verdict answers for a caller who named both keys.
///
/// Digest first, because a digest names exact bytes and a verdict filed under
/// it is about those bytes and nothing else. The PURL is consulted only when
/// the digest is unknown — lazily, so an exact hit costs no index lookup at all
/// — and its verdict is accepted only if it resolved to the same digest.
///
/// That last condition is the whole point of holding both keys. A release whose
/// artifact has changed under it still has a perfectly good verdict; it is just
/// a verdict about a different artifact than the caller asked about, and
/// serving it would answer a question nobody posed.
fn pick_verdict(
    by_sha: Option<crate::lookup::Verdict>,
    by_purl: impl FnOnce() -> Option<crate::lookup::Verdict>,
    sha: &str,
) -> Option<crate::lookup::Verdict> {
    if by_sha.is_some() {
        return by_sha;
    }
    by_purl().filter(|v| v.sha256.eq_ignore_ascii_case(sha))
}

fn lookup_by_purl(state: &AppState, raw: &str) -> Response {
    // `pkg:` is optional, as it is on /analyze-purl and `atomscan purl`, and
    // the canonical form is what the filters and the index are keyed by — so
    // `npm/left-pad@1.3.0` and `pkg:npm/left-pad@1.3.0` are one question.
    let purl = match normalize_pkg_purl(raw) {
        Ok(purl) => purl,
        // Unparseable: name what the caller actually sent, not the canonical
        // form there isn't one of.
        Err(message) => {
            return with_subject(
                error_response(StatusCode::BAD_REQUEST, message),
                Subject::purl(raw, None),
            );
        }
    };
    let decision = crate::bloom_repo::global()
        .as_deref()
        .map_or(crate::bloom_repo::Decision::Unknown, |lk| {
            lk.memo_purl(&purl)
        });
    let verdict = crate::lookup::global().and_then(|index| index.get_purl(&purl));
    let sha = verdict
        .as_ref()
        .map_or("", |v| v.sha256.as_str())
        .to_owned();
    let response = lookup_response(state, verdict.as_ref(), decision, &sha, Some(&purl));
    // The canonical PURL, plus the digest it resolved to when it hit — that
    // mapping is knowledge only this handler has.
    with_subject(response, Subject::purl(&purl, Some(&sha)))
}

/// How long a bloom-derived answer may be cached.
///
/// Two hours — deliberately longer than the hourly filter rebuild, so a derived
/// answer can outlive one cycle. That is the safe direction to be stale in: the
/// filters only ever ADD claims, so an answer that lags errs toward flagging an
/// artifact rather than clearing one. The cost is the reverse case — an artifact
/// analyzed and found clean can keep reading as cited until the entry ages out —
/// which is bounded, visible in the `bloom` field, and cheaper than paying a
/// hopper round trip on every repeat ask.
///
/// Still far below the 24h a measured verdict earns: that one is immutable for
/// the ruleset that produced it, and this one is not.
const BLOOM_DERIVED_MAX_AGE: u32 = 7200;

/// Render a lookup answer.
///
/// A stored verdict is a 200; an adverse bloom match with nothing stored is a
/// 200 carrying a *derived* answer (see [`crate::lookup::bloom_derived_view`]);
/// holding neither is a 404. The bloom decision rides on all three. That keeps
/// the kinds of knowledge distinguishable — a filter says who has claimed what,
/// an analysis says what the thing *is* — while still answering in one round
/// trip.
///
/// A derived answer carries no `eng`, which is how a consumer tells it from a
/// measurement, and how `/v1/analyze` knows it must still run: a citation is
/// exactly what that route exists to replace with a measurement, so it must
/// never stand in for one.
fn lookup_response(
    state: &AppState,
    verdict: Option<&crate::lookup::Verdict>,
    decision: crate::bloom_repo::Decision,
    sha256: &str,
    purl: Option<&str>,
) -> Response {
    // A token-protected answer must not be stored by a shared cache: it is
    // knowledge about a specific customer's artifact, not public data.
    let scope = if state.auth_digest.is_some() {
        "private"
    } else {
        "public"
    };
    let Some(verdict) = verdict else {
        // Nothing measured, but the filters may still have something to say. A
        // bloom match is answerable on its own — see `bloom_derived_view` — so
        // rather than answering "unknown" about a digest several operators call
        // malware, answer with what they say, marked as what it is. Mirrors
        // hopper's fromLedger, and saves the round trip to it.
        let mut synth_hits = Vec::new();
        if let Some(view) = crate::lookup::bloom_derived_view(
            decision,
            decision.as_str(),
            sha256.trim(),
            purl,
            &mut synth_hits,
        ) {
            let mut resp = Json(view).into_response();
            let headers = resp.headers_mut();
            // Emphatically NOT the 24h a measured verdict gets. This answer
            // stands on a filter that is rebuilt hourly and on a ledger that
            // moves underneath it, and it must stop being served the moment a
            // real analysis exists. hopper bounds its own ledger-derived
            // records the same way and for the same reason.
            if let Ok(value) = axum::http::HeaderValue::from_str(&format!(
                "{scope}, max-age={BLOOM_DERIVED_MAX_AGE}"
            )) {
                headers.insert(axum::http::header::CACHE_CONTROL, value);
            }
            if let Ok(value) = axum::http::HeaderValue::from_str(sha256.trim()) {
                headers.insert("X-SHA256", value);
            }
            headers.insert(
                "X-Scan-Source",
                axum::http::HeaderValue::from_static("scan:bloom"),
            );
            return resp;
        }
        // Nothing stored does not mean nothing happening: an analysis of this
        // very artifact may be minutes in. Saying so costs nothing — the caller
        // is already asking about this key, and the registry is a map lookup —
        // and it is what lets a caller who reconnects be routed back to the
        // worker already running their analysis instead of starting a second
        // one beside it. `/status` answers the same question on its own, for a
        // caller who has nothing else to ask.
        let running = purl
            .and_then(|p| state.flights.running(&FlightKey::Purl(p.to_string())))
            .or_else(|| {
                state
                    .flights
                    .running(&FlightKey::Sha(sha256.trim().to_ascii_lowercase()))
            });
        let mut body = serde_json::json!({
            "error": "unknown sample",
            "bloom": decision.as_str(),
        });
        if let Some(run) = running {
            body["analyzing"] = serde_json::json!({
                "elapsed_ms": crate::duration_ms(run.elapsed),
                "attached": run.attached,
            });
        }
        // A miss is not cacheable for any length of time: it becomes a hit the
        // moment anything analyzes this artifact.
        let mut resp = (StatusCode::NOT_FOUND, Json(body)).into_response();
        resp.headers_mut().insert(
            axum::http::header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store"),
        );
        resp.headers_mut().insert(
            "X-Scan-Source",
            axum::http::HeaderValue::from_static(lookup_source(None, decision)),
        );
        return resp;
    };
    let mut resp = Json(verdict.view(decision.as_str(), purl)).into_response();
    let headers = resp.headers_mut();
    // A verdict is immutable for the ruleset that produced it, and the ruleset
    // is part of the namespace it was read from — a rules or model update
    // lands in a fresh namespace, which reads as a miss rather than as this
    // answer going stale.
    if let Ok(value) = axum::http::HeaderValue::from_str(&format!("{scope}, max-age=86400")) {
        headers.insert(axum::http::header::CACHE_CONTROL, value);
    }
    if let Ok(value) = axum::http::HeaderValue::from_str(sha256.trim()) {
        headers.insert("X-SHA256", value);
    }
    headers.insert(
        "X-Scan-Source",
        axum::http::HeaderValue::from_static(lookup_source(Some(verdict), decision)),
    );
    resp
}

/// Explain which local knowledge produced a lookup answer. A stored verdict is
/// an analysis result; when there is no verdict, a non-unknown Bloom decision
/// is the only artifact-derived answer left. A lookup miss is therefore still
/// attributed to the Bloom/lookup layer, even when the Bloom decision is
/// `unknown`.
fn lookup_source(
    verdict: Option<&crate::lookup::Verdict>,
    _decision: crate::bloom_repo::Decision,
) -> &'static str {
    if verdict.is_some() {
        "scan:analysis"
    } else {
        "scan:bloom"
    }
}

/// Outcome of [`do_model_reload`] — caller maps this to an HTTP response.
struct ReloadOutcome {
    elapsed_ms: u128,
    /// If trait reload failed, the error message. `None` means traits loaded OK.
    traits_reload_error: Option<String>,
}

/// Perform the cleave-traits reload + model load + atomic swap. Caller is
/// responsible for holding `state.reload_lock`. Shared by /_/reload and
/// /_/update so the load+swap dance lives in exactly one place.
async fn do_model_reload(
    state: &Arc<AppState>,
) -> Result<ReloadOutcome, (StatusCode, &'static str)> {
    let start = Instant::now();
    let model_dir = state.model_dir.clone();
    let thresholds = state.threshold_overrides;
    let level = state.level;

    const RELOAD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
    let result = tokio::time::timeout(
        RELOAD_TIMEOUT,
        tokio::task::spawn_blocking(move || {
            // Reload cleave traits first so the new model runs against fresh rules.
            let traits_reload_error = match cleave::reload_capability_mapper() {
                Err(e) => {
                    tracing::warn!("cleave trait reload failed (previous traits retained): {e:#}");
                    Some(e)
                }
                Ok(_) => {
                    tracing::info!("cleave traits reloaded");
                    None
                }
            };
            cleave::clear_all_thread_caches();

            let model = Model::load(&model_dir, thresholds, level)?;
            let shap = ShapImportance::load(&model_dir).ok();
            let ctx = ExtractContext::new(model.spec());
            Ok::<_, anyhow::Error>((model, shap, ctx, traits_reload_error))
        }),
    )
    .await;

    let elapsed_ms = start.elapsed().as_millis();

    let (model, shap, ctx, traits_reload_error) = match result {
        Ok(Ok(Ok(t))) => t,
        Ok(Ok(Err(e))) => {
            // Log internally but do not expose filesystem paths or model internals to callers.
            tracing::warn!("reload failed (previous model retained) in {elapsed_ms}ms: {e:#}");
            return Err((StatusCode::UNPROCESSABLE_ENTITY, "Failed to load model"));
        }
        Ok(Err(e)) => {
            tracing::warn!("reload task join error: {e}");
            return Err((StatusCode::INTERNAL_SERVER_ERROR, "Internal error"));
        }
        Err(_elapsed) => {
            tracing::warn!(
                "reload timed out after {}s (previous model retained)",
                RELOAD_TIMEOUT.as_secs()
            );
            return Err((StatusCode::GATEWAY_TIMEOUT, "Reload timed out"));
        }
    };

    let spec_version = model.spec().version();
    let features = model.spec().total_features();
    let shap_loaded = shap.is_some();
    let was_ready = state.ready.load(std::sync::atomic::Ordering::Relaxed);

    match state.resources.write() {
        Ok(mut lock) => {
            *lock = Some(Arc::new(super::ModelResources {
                model,
                shap,
                ctx,
                interpret: state.interpret.clone(),
                fetch: state.fetch,
                zip_passwords: state.zip_passwords.clone(),
            }));
            if let Ok(mut init_error) = state.init_error.write() {
                *init_error = None;
            }
            state
                .ready
                .store(true, std::sync::atomic::Ordering::Release);
            if was_ready {
                tracing::info!(
                    elapsed_ms,
                    spec_version,
                    features,
                    shap_loaded,
                    "model reloaded"
                );
            } else {
                tracing::info!(
                    elapsed_ms,
                    spec_version,
                    features,
                    shap_loaded,
                    "model loaded via reload — server now ready"
                );
            }
            Ok(ReloadOutcome {
                elapsed_ms,
                traits_reload_error,
            })
        }
        Err(e) => {
            tracing::error!("write lock poisoned during reload: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, "Internal error"))
        }
    }
}

/// POST /reload — reload model from disk, swap atomically.
///
/// Only one reload may run at a time; concurrent calls receive 409.
pub(super) async fn reload(State(state): State<Arc<AppState>>) -> Response {
    tracing::info!("POST /_/reload");
    // Prevent concurrent reloads — each load allocates significant memory.
    let Ok(_guard) = state.reload_lock.try_lock() else {
        tracing::warn!("reload rejected: already in progress");
        return error_response(StatusCode::CONFLICT, "Reload already in progress");
    };

    match do_model_reload(&state).await {
        Ok(outcome) => {
            let mut body = serde_json::json!({
                "status": "ok",
                "elapsed_ms": outcome.elapsed_ms,
            });
            if let Some(err) = &outcome.traits_reload_error {
                body["traits_reload_error"] = serde_json::json!(err);
            }
            Json(body).into_response()
        }
        Err((status, msg)) => error_response(status, msg),
    }
}

/// POST /_/update — pull latest models + traits from their git repos, then
/// reload. Both pulls are non-fatal: if either fails, the response reports
/// it but the reload still runs against whatever is currently on disk so
/// the operator gets the most-recent-good state. Shares the reload_lock
/// with /_/reload so concurrent calls receive 409.
pub(super) async fn update(State(state): State<Arc<AppState>>) -> Response {
    tracing::info!("POST /_/update");
    let Ok(_guard) = state.reload_lock.try_lock() else {
        tracing::warn!("update rejected: reload already in progress");
        return error_response(StatusCode::CONFLICT, "Reload already in progress");
    };

    // Run the two repo pulls on a blocking thread; they're synchronous git
    // and filesystem work. Both are non-fatal — log on failure and proceed
    // to the reload step regardless so we still pick up any partial state.
    //
    // The outer timeout is only a backstop around the sequential models +
    // traits updates. model_update validates each bundle (Model::load) before
    // swapping it in, so a broken bundle never lands on disk and there's nothing
    // to roll back.
    const PULL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(21 * 60);
    let pull_result = tokio::time::timeout(
        PULL_TIMEOUT,
        tokio::task::spawn_blocking(|| {
            let dir = crate::models_repo::install_target();
            let models_err = match crate::model_update::update(&dir, false, false) {
                Ok(()) => None,
                Err(e) => {
                    tracing::warn!("models update failed: {e:#}");
                    Some(e.to_string())
                }
            };
            let traits_err = match crate::traits_repo::update(false, false) {
                Ok(_) => None,
                Err(e) => {
                    tracing::warn!("traits update failed: {e:#}");
                    Some(e.to_string())
                }
            };
            (models_err, traits_err)
        }),
    )
    .await;

    let (models_err, traits_err) = match pull_result {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            tracing::warn!("update pull task join error: {e}");
            let msg = "task join failed".to_string();
            (Some(msg.clone()), Some(msg))
        }
        Err(_elapsed) => {
            tracing::warn!("update pull timed out after {}s", PULL_TIMEOUT.as_secs());
            let msg = format!("update timed out after {}s", PULL_TIMEOUT.as_secs());
            (Some(msg.clone()), Some(msg))
        }
    };

    match do_model_reload(&state).await {
        Ok(outcome) => {
            let mut body = serde_json::json!({
                "status": "ok",
                "elapsed_ms": outcome.elapsed_ms,
                "models_updated": models_err.is_none(),
                "traits_updated": traits_err.is_none(),
                "models_error": models_err,
                "traits_error": traits_err,
                "version": env!("CARGO_PKG_VERSION"),
                "model_commit": crate::models_repo::version(),
                "traits_commit": cleave::traits_repo::version(),
            });
            if let Some(err) = &outcome.traits_reload_error {
                body["traits_reload_error"] = serde_json::json!(err);
            }
            Json(body).into_response()
        }
        Err((status, msg)) => {
            // The in-memory model was not swapped, so requests continue against
            // the previous model. The on-disk bundle was validated before install,
            // so there's nothing to roll back — a reload failure here is in-memory.
            tracing::error!("model reload failed after update: {msg}");
            (
                status,
                Json(serde_json::json!({
                    "status": "reload_failed",
                    "error": msg,
                    "models_updated": models_err.is_none(),
                    "traits_updated": traits_err.is_none(),
                    "models_error": models_err,
                    "traits_error": traits_err,
                })),
            )
                .into_response()
        }
    }
}

/// The name an upload is staged and reported under.
///
/// Every character outside `[A-Za-z0-9_.-]` becomes `_`, `..` collapses to
/// `__`, and the result keeps its last 63 bytes so the extension survives —
/// cleave detects file type from it, and the name is also the `path` label in
/// the report and in every log line about this request.
///
/// The filter is deliberately ASCII-only rather than Unicode-aware. Two
/// reasons, both of them the client's choice to make otherwise: a
/// `char::is_alphanumeric` filter keeps multi-byte characters, and the
/// right-truncation below would then slice mid-character and panic the
/// request; and a name that reaches logs and a filesystem path should not
/// carry homoglyphs, combining marks, or bidi-shaped text.
fn sanitize_upload_filename(raw: &str) -> String {
    let sanitized: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .replace("..", "__");
    // Every retained character is one ASCII byte, so this index is always a
    // character boundary.
    #[allow(clippy::string_slice)]
    if sanitized.len() > 63 {
        sanitized[sanitized.len() - 63..].to_string()
    } else {
        sanitized
    }
}

/// POST /analyze — accept multipart file upload, classify, return full JSON result.
pub(super) async fn analyze(
    State(state): State<Arc<AppState>>,
    request_id: Extension<RequestId>,
    mut multipart: axum::extract::Multipart,
) -> Response {
    let request_id = request_id.0.get();
    state.note_analyze_request();
    let request_start = Instant::now();

    tracing::info!(id = request_id, "--> POST /analyze");

    if let Ok(init_error) = state.init_error.read()
        && let Some(message) = init_error.as_ref()
    {
        tracing::error!(id = request_id, error = %message, "rejected: startup failed");
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "Server failed to initialize",
        );
    }

    if let Some(response) = check_memory_pressure(&state).await {
        return response;
    }

    // Parse the first multipart field as the file.
    let mut field = match multipart.next_field().await {
        Ok(Some(f)) => f,
        Ok(None) => {
            tracing::warn!(id = request_id, "bad request: no file field");
            return error_response(StatusCode::BAD_REQUEST, "No file field in request");
        }
        Err(e) => {
            tracing::warn!(id = request_id, error = %e, "bad request: unparseable multipart body");
            return error_response(StatusCode::BAD_REQUEST, "Invalid multipart data");
        }
    };

    // The staged name, used as the temp file's name (so cleave detects the
    // file type from its extension), as the `path` label in the report, and in
    // every log line about this request.
    let filename = match field.file_name() {
        Some(name) => sanitize_upload_filename(name),
        // Already within the sanitizer's alphabet.
        None => format!("upload-{request_id}"),
    };

    // Create a temp directory containing a file with the original filename so that
    // cleave's filename-based type detection works correctly (e.g. "package.json"
    // is recognized as PackageJson, not Unknown).
    let fname_for_temp = filename.clone();
    let temp_dir =
        match tokio::task::spawn_blocking(move || TempBuilder::new().prefix("scan-").tempdir())
            .await
        {
            Ok(Ok(d)) => d,
            Ok(Err(e)) => {
                tracing::warn!(id = request_id, error = %e, "failed to create temp dir");
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, "Internal error");
            }
            Err(e) => {
                tracing::warn!(id = request_id, error = %e, "temp dir task join error (panic?)");
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, "Internal error");
            }
        };

    // Stream multipart field to a file with the original name inside the temp dir.
    let path = temp_dir.path().join(&fname_for_temp);
    let writer = match std::fs::File::create(&path) {
        Ok(file) => file,
        Err(e) => {
            tracing::warn!(id = request_id, path = %path.display(), error = %e, "failed to open temp file for writing");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "Internal error");
        }
    };
    let mut tokio_file = tokio::fs::File::from_std(writer);

    let max_upload = state.max_upload_bytes;
    let mut file_size = 0usize;
    // Hashed as it streams — every byte is already in hand, and the digest is
    // what lets a second request for these bytes join the first one's analysis.
    let mut digest = Sha256::new();
    loop {
        match field.chunk().await {
            Ok(Some(chunk)) if !chunk.is_empty() => {
                file_size += chunk.len();
                if file_size > max_upload {
                    tracing::warn!(
                        id = request_id,
                        file_size,
                        max_upload,
                        "upload exceeded size limit",
                    );
                    return error_response(StatusCode::PAYLOAD_TOO_LARGE, "File too large");
                }
                digest.update(&chunk);
                if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut tokio_file, &chunk).await {
                    tracing::warn!(id = request_id, error = %e, "failed to write upload chunk");
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to save file data",
                    );
                }
            }
            Ok(Some(_)) => continue,
            Ok(None) => break,
            Err(e) => {
                tracing::warn!(id = request_id, error = %e, "failed to read multipart chunk");
                return error_response(StatusCode::BAD_REQUEST, "Error reading upload data");
            }
        }
    }

    if let Err(e) = tokio::io::AsyncWriteExt::flush(&mut tokio_file).await {
        tracing::warn!(id = request_id, error = %e, "failed to flush temp file");
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to save file data",
        );
    }
    if let Err(e) = tokio_file.sync_all().await {
        tracing::warn!(id = request_id, error = %e, "failed to sync temp file");
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to save file data",
        );
    }
    drop(tokio_file);

    if file_size == 0 {
        tracing::warn!(id = request_id, "bad request: empty file");
        return error_response(StatusCode::BAD_REQUEST, "Empty file");
    }

    let sha = format!("{:x}", digest.finalize());

    // Share the run with anyone already analyzing these exact bytes.
    let attachment = state.flights.join(FlightKey::Sha(sha.clone()));
    let leads = attachment.leads();
    if leads {
        tracing::info!(
            id = request_id,
            filename = %filename,
            size_bytes = file_size,
            sha256 = %sha,
            upload_ms = crate::duration_ms(request_start.elapsed()),
            "received file, starting analysis",
        );
        let publisher = state.flights.publisher(attachment.flight());
        match claim_slot(&state, request_id, attachment.flight().key()) {
            Err(outcome) => publisher.publish(outcome),
            Ok((resources, permit)) => {
                let follow = resources.fetch;
                let flight = Arc::clone(attachment.flight());
                let state = Arc::clone(&state);
                let upload = Upload {
                    dir: temp_dir,
                    path,
                    filename,
                    size_bytes: file_size,
                };
                // Detached, so the analysis outlives whichever request started
                // it: this client hanging up must not abandon the followers.
                tokio::spawn(async move {
                    publisher.publish(
                        run_file_analysis(
                            state,
                            request_id,
                            upload,
                            &flight,
                            resources,
                            permit,
                            RequestFollow {
                                policy: follow,
                                persist: true,
                            },
                        )
                        .await,
                    );
                });
            }
        }
    } else {
        tracing::info!(
            id = request_id,
            filename = %filename,
            size_bytes = file_size,
            sha256 = %sha,
            "received file, joined an analysis already in flight",
        );
        // These bytes are already being analyzed; ours are surplus.
        drop(temp_dir);
    }

    let outcome = attachment.flight().wait().await;
    flight_response(
        &outcome,
        crate::duration_ms(request_start.elapsed()),
        !leads,
        attachment.flight().key(),
    )
}

/// Bytes a request uploaded, staged on disk and waiting to be analyzed.
#[derive(Debug)]
struct Upload {
    /// Owns the directory holding [`Self::path`]; deleting it deletes the file.
    dir: tempfile::TempDir,
    /// The staged file, named so cleave can detect its type from the extension.
    path: std::path::PathBuf,
    /// Sanitized upload filename, used as the display path.
    filename: String,
    size_bytes: usize,
}

/// Request-scoped traversal and storage behavior. Keeping these together makes
/// it difficult to thread a custom follow policy into classification while
/// accidentally retaining the canonical cache-write behavior.
#[derive(Clone, Copy)]
struct RequestFollow {
    policy: crate::fetch::FetchPolicy,
    persist: bool,
}

/// Run one uploaded-file analysis on behalf of every request attached to
/// `flight`. Takes the staged upload and deletes it on the way out.
async fn run_file_analysis(
    state: Arc<AppState>,
    request_id: u64,
    upload: Upload,
    flight: &Arc<Flight>,
    resources: Arc<super::ModelResources>,
    permit: tokio::sync::OwnedSemaphorePermit,
    request_follow: RequestFollow,
) -> Outcome {
    let Upload {
        dir: temp_dir,
        path,
        filename,
        size_bytes: file_size,
    } = upload;
    let started = Instant::now();
    let slow_rule_ms = state.slow_rule_ms;
    let should_clear_caches = request_id.is_multiple_of(100);
    // The flight owns cancellation now: it is raised when the last attached
    // request goes away, not when any one of them does.
    let cancellation = flight.cancellation();
    state.in_flight.insert(
        request_id,
        super::InFlightRequest {
            name: filename.clone(),
            size_bytes: file_size as u64,
            started_at: Instant::now(),
            cancellation: Arc::clone(&cancellation),
            phase: cleave::PhaseTracker::with_label(format!("req#{request_id} {filename}")),
            thread_id: AtomicU64::new(0),
        },
    );
    let guard = super::RequestGuard::new(
        request_id,
        Arc::clone(&state),
        Arc::clone(&cancellation),
        permit,
    );

    let temp_dir_path = temp_dir.path().to_path_buf();
    let cancel_flag = Arc::clone(&cancellation);
    let phase_state = Arc::clone(&state);
    let phase_tracker = phase_state
        .in_flight
        .get(&request_id)
        .map(|r| r.phase.clone());
    // A policy-specific `--follow` override must not land in the shared
    // corpus, same rule every other route's artifact mirroring follows.
    let uploader = request_follow
        .persist
        .then(|| state.uploader.clone())
        .flatten();
    let handle = tokio::task::spawn_blocking(move || {
        // Record the OS thread servicing this analysis.
        if let Some(req) = phase_state.in_flight.get(&request_id) {
            req.thread_id.store(current_thread_id(), Ordering::Relaxed);
        }
        let result = classify_file_with_follow(
            &path,
            &filename,
            &resources,
            slow_rule_ms,
            None,
            Some(&cancel_flag),
            phase_tracker.as_ref(),
            None, // interactive upload carries no fetch-time registry provenance
            request_follow.policy,
            // /analyze returns the envelope and discards the result — only
            // /analyze-path renews results (and their dependencies) on hopper.
            false,
        );
        // Offer the artifact — bytes only, no registry provenance — right
        // alongside its verdict, same as every other route's bundle: after
        // analysis, not on receipt, so hopper never carries a claimable
        // bytes-with-no-verdict row for longer than it takes the uploader
        // thread to drain the queue (its upload-tier claim query drains
        // those first and ahead of everything else, so offering them earlier
        // would race this sample onto the worker fleet for a redundant
        // analysis). Queued before `drop(temp_dir)` below, not read yet — the
        // background uploader thread reads the path off disk when it
        // dequeues the job, which can still lose the temp dir on a deep
        // queue; best-effort, same as the graceful miss dependency mirroring
        // already tolerates on a blob-cache eviction.
        if let (Ok(scan_result), Some(uploader)) = (&result, &uploader) {
            uploader.submit_artifacts(crate::engine::collect_upload_artifacts(
                &path,
                &scan_result.sha256,
                scan_result.size_bytes,
                upload_collector(),
                None,
                None,
            ));
        }
        if should_clear_caches {
            cleave::clear_all_thread_caches();
        }
        drop(temp_dir);
        result
    });

    // Bounded by the configured per-request timeout. On timeout we signal
    // cancellation and report 504 — the blocking thread continues until cleave
    // notices the flag, but the slot is freed and `stuck_orphans` is
    // incremented so an operator can see zombie work.
    let result = await_with_timeout(
        handle,
        state.analysis_timeout_secs,
        &cancellation,
        &state.stuck_orphans,
    )
    .await;
    drop(guard);

    // Handles the case where drop(temp_dir) above didn't run.
    if let Err(e) = tokio::fs::remove_dir_all(&temp_dir_path).await {
        tracing::debug!(request_id, error = %e, "temp dir cleanup (may already be gone)");
    }

    flight_outcome(
        result,
        request_id,
        crate::duration_ms(started.elapsed()),
        flight.key(),
        &state,
        request_follow.persist,
    )
}

/// POST /analyze-purl — fetch a package by PURL and analyze it.
///
/// Scan looks up registry provenance itself (age, custody, downloads) and
/// grafts it into the report, the same path as `atomscan purl`. Beamline
/// calls this when a PURL is not in hopper; it is a full analysis and takes
/// a slot. Dependency fetch and LLM interpretation follow the process-wide
/// `--follow` / `--interpret` flags.
#[derive(serde::Deserialize)]
pub(super) struct AnalyzePurlRequest {
    purl: String,
}

pub(super) async fn analyze_purl(
    State(state): State<Arc<AppState>>,
    request_id: Extension<RequestId>,
    Json(req): Json<AnalyzePurlRequest>,
) -> Response {
    let request_id = request_id.0.get();
    state.note_analyze_request();
    let request_start = Instant::now();

    let purl = match normalize_pkg_purl(&req.purl) {
        Ok(p) => p,
        // No flight, so no key to take the subject from: name what the caller
        // sent, bounded, as the lookup route does.
        Err(msg) => {
            return with_subject(
                error_response(StatusCode::BAD_REQUEST, msg),
                Subject::purl(&req.purl, None),
            );
        }
    };

    if let Ok(init_error) = state.init_error.read()
        && let Some(message) = init_error.as_ref()
    {
        tracing::error!(id = request_id, error = %message, "rejected: startup failed");
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "Server failed to initialize",
        );
    }

    if let Some(response) = check_memory_pressure(&state).await {
        return response;
    }

    // Share the run with anyone already analyzing this PURL.
    let attachment = state.flights.join(FlightKey::Purl(purl.clone()));
    let leads = attachment.leads();
    if leads {
        tracing::info!(id = request_id, purl = %purl, "--> POST /analyze-purl");
        let publisher = state.flights.publisher(attachment.flight());
        match claim_slot(&state, request_id, attachment.flight().key()) {
            Err(outcome) => publisher.publish(outcome),
            Ok((resources, permit)) => {
                let follow = resources.fetch;
                let flight = Arc::clone(attachment.flight());
                let state = Arc::clone(&state);
                // Detached, so the analysis outlives whichever request started
                // it: this client hanging up must not abandon the followers.
                tokio::spawn(async move {
                    publisher.publish(
                        run_purl_analysis(
                            state,
                            request_id,
                            &purl,
                            &flight,
                            resources,
                            permit,
                            RequestFollow {
                                policy: follow,
                                persist: true,
                            },
                        )
                        .await,
                    );
                });
            }
        }
    } else {
        tracing::info!(
            id = request_id,
            purl = %purl,
            "--> POST /analyze-purl (joined an analysis already in flight)",
        );
    }

    let outcome = attachment.flight().wait().await;
    flight_response(
        &outcome,
        crate::duration_ms(request_start.elapsed()),
        !leads,
        attachment.flight().key(),
    )
}

/// Run one PURL analysis on behalf of every request attached to `flight`.
async fn run_purl_analysis(
    state: Arc<AppState>,
    request_id: u64,
    purl: &str,
    flight: &Arc<Flight>,
    resources: Arc<super::ModelResources>,
    permit: tokio::sync::OwnedSemaphorePermit,
    request_follow: RequestFollow,
) -> Outcome {
    let started = Instant::now();
    let slow_rule_ms = state.slow_rule_ms;
    let should_clear_caches = request_id.is_multiple_of(100);
    // The flight owns cancellation now: it is raised when the last attached
    // request goes away, not when any one of them does.
    let cancellation = flight.cancellation();
    state.in_flight.insert(
        request_id,
        super::InFlightRequest {
            name: purl.to_owned(),
            size_bytes: 0,
            started_at: Instant::now(),
            cancellation: Arc::clone(&cancellation),
            phase: cleave::PhaseTracker::with_label(format!("req#{request_id} {purl}")),
            thread_id: AtomicU64::new(0),
        },
    );
    let guard = super::RequestGuard::new(
        request_id,
        Arc::clone(&state),
        Arc::clone(&cancellation),
        permit,
    );

    let cancel_flag = Arc::clone(&cancellation);
    let phase_state = Arc::clone(&state);
    let phase_tracker = phase_state
        .in_flight
        .get(&request_id)
        .map(|r| r.phase.clone());
    let deps_for_upload = request_follow.persist && state.uploader.is_some();
    let uploader_for_artifacts = if request_follow.persist {
        state.uploader.clone()
    } else {
        None
    };
    let corpus_for_fallback = state.corpus.clone();
    let owned_purl = purl.to_owned();
    let handle = tokio::task::spawn_blocking(move || {
        if let Some(req) = phase_state.in_flight.get(&request_id) {
            req.thread_id.store(current_thread_id(), Ordering::Relaxed);
        }
        let result = classify_purl(
            &owned_purl,
            &resources,
            slow_rule_ms,
            Some(&cancel_flag),
            phase_tracker.as_ref(),
            deps_for_upload,
            uploader_for_artifacts.as_ref(),
            request_follow.policy,
            corpus_for_fallback.as_deref(),
        );
        if should_clear_caches {
            cleave::clear_all_thread_caches();
        }
        result
    });

    let result = await_with_timeout(
        handle,
        state.analysis_timeout_secs,
        &cancellation,
        &state.stuck_orphans,
    )
    .await;
    drop(guard);

    flight_outcome(
        result,
        request_id,
        crate::duration_ms(started.elapsed()),
        flight.key(),
        &state,
        request_follow.persist,
    )
}

/// Run one exact-URL analysis on behalf of every request attached to `flight`.
/// Unlike a PURL, the URL is fetched verbatim; the resulting ScanResult carries
/// the SHA-256 that Beamline uses to alias this URL to the canonical artifact.
async fn run_url_analysis(
    state: Arc<AppState>,
    request_id: u64,
    url: &str,
    flight: &Arc<Flight>,
    resources: Arc<super::ModelResources>,
    permit: tokio::sync::OwnedSemaphorePermit,
    request_follow: RequestFollow,
) -> Outcome {
    let started = Instant::now();
    let cancellation = flight.cancellation();
    state.in_flight.insert(
        request_id,
        super::InFlightRequest {
            name: url.to_owned(),
            size_bytes: 0,
            started_at: Instant::now(),
            cancellation: Arc::clone(&cancellation),
            phase: cleave::PhaseTracker::with_label(format!("req#{request_id} {url}")),
            thread_id: AtomicU64::new(0),
        },
    );
    let guard = super::RequestGuard::new(
        request_id,
        Arc::clone(&state),
        Arc::clone(&cancellation),
        permit,
    );
    let phase_state = Arc::clone(&state);
    let phase_tracker = phase_state
        .in_flight
        .get(&request_id)
        .map(|r| r.phase.clone());
    let cancel_flag = Arc::clone(&cancellation);
    let slow_rule_ms = state.slow_rule_ms;
    let analysis_timeout_secs = state.analysis_timeout_secs;
    let stuck_orphans = &state.stuck_orphans;
    let handle = tokio::task::spawn_blocking({
        let url = url.to_owned();
        let uploader = if request_follow.persist {
            state.uploader.clone()
        } else {
            None
        };
        let deps_for_upload = request_follow.persist && state.uploader.is_some();
        let policy = request_follow.policy;
        move || {
            if let Some(req) = phase_state.in_flight.get(&request_id) {
                req.thread_id.store(current_thread_id(), Ordering::Relaxed);
            }
            let result = classify_url(
                &url,
                &resources,
                slow_rule_ms,
                Some(&cancel_flag),
                phase_tracker.as_ref(),
                deps_for_upload,
                uploader.as_ref(),
                policy,
            );
            if request_id.is_multiple_of(100) {
                cleave::clear_all_thread_caches();
            }
            result
        }
    });
    let result =
        await_with_timeout(handle, analysis_timeout_secs, &cancellation, stuck_orphans).await;
    drop(guard);
    flight_outcome(
        result,
        request_id,
        crate::duration_ms(started.elapsed()),
        flight.key(),
        &state,
        request_follow.persist,
    )
}

/// Canonical `pkg:…` form, or a 400 message. Same prefixing rule as `atomscan purl`.
fn normalize_pkg_purl(raw: &str) -> Result<String, &'static str> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("missing purl");
    }
    let prefixed = if raw.starts_with("pkg:") {
        raw.to_string()
    } else {
        format!("pkg:{raw}")
    };
    fletch::purl::normalize(&prefixed).ok_or("not a package URL")
}

fn valid_http_url(raw: &str) -> bool {
    reqwest::Url::parse(raw)
        .map(|url| matches!(url.scheme(), "http" | "https"))
        .unwrap_or(false)
}

/// Fetch the PURL's artifact (and its registry record) then classify. Scan
/// looks up provenance itself — beamline does not supply it.
#[allow(clippy::too_many_arguments)] // one linear package analysis path
/// Offer a registry-metadata fallback's provenance to hopper without ever
/// posting the fallback's own content as a new sample: that content is the
/// registry's JSON record, not a real artifact, and it hashes differently
/// every time it's built (`with_age()`-derived fields are relative to the
/// call), so treating it as content-addressed mints hopper a fresh,
/// never-deduplicating row on every single fetch — confirmed 2026-08-27
/// against production (`lodash.once@4.1.1` and friends: 8-9 distinct shas for
/// 8-9 fetches of the identical coordinate, in under two hours).
///
/// Looks up whether hopper already holds *real* content for this purl (a
/// prior successful fetch, by this process or another producer). If so,
/// backfills this fresh registry metadata onto that existing sha as
/// provenance-only — no bytes move — and the caller's verdict should redirect
/// onto it too. If hopper has never seen this coordinate under any sha, there
/// is nothing to attach to; the caller's verdict is suppressed rather than
/// minting a placeholder that would just be more of the same churn.
///
/// Runs the corpus lookup via `block_on`: this is always called from a
/// `spawn_blocking` thread (`classify_purl` never runs on the async
/// executor), so blocking here costs nothing the caller isn't already paying.
fn offer_registry_fallback(
    corpus: Option<&corpus::Corpus>,
    uploader: Option<&Arc<crate::upload::Uploader>>,
    purl: &str,
    name: &str,
    registry_provenance: Option<&crate::provenance::RegistryProvenance>,
) -> crate::engine::HopperRoute {
    use crate::engine::HopperRoute;
    let Some(corpus) = corpus else {
        return HopperRoute::Suppress;
    };
    let (reached, _source) =
        tokio::runtime::Handle::current().block_on(corpus.known_with_source(None, Some(purl)));
    let Reached::Record(record) = reached else {
        return HopperRoute::Suppress;
    };
    let Some(real_sha) = record.sha256 else {
        return HopperRoute::Suppress;
    };
    if let Some(uploader) = uploader {
        let now = crate::engine::now_rfc3339();
        let sidecar = match registry_provenance {
            Some(provenance) => crate::provenance::build_sidecar_from_provenance(
                name,
                &real_sha,
                0,
                upload_collector(),
                &now,
                "",
                purl,
                provenance,
            ),
            None => crate::provenance::build_sidecar(
                name,
                &real_sha,
                0,
                upload_collector(),
                &now,
                "",
                purl,
                None,
                &[],
            ),
        };
        uploader.submit_artifacts(vec![crate::upload::UploadArtifact {
            sha256: real_sha.clone(),
            size: 0,
            filename: name.to_string(),
            bytes: crate::upload::ArtifactBytes::File(std::path::PathBuf::new()),
            sidecar,
            backfill: true,
        }]);
    }
    HopperRoute::Redirect(real_sha)
}

#[allow(clippy::too_many_arguments)]
fn classify_purl(
    purl: &str,
    resources: &super::ModelResources,
    slow_rule_ms: u64,
    cancellation: Option<&Arc<AtomicBool>>,
    phase: Option<&cleave::PhaseTracker>,
    deps_for_upload: bool,
    uploader: Option<&Arc<crate::upload::Uploader>>,
    follow: crate::fetch::FetchPolicy,
    corpus: Option<&corpus::Corpus>,
) -> anyhow::Result<crate::engine::ScanResult> {
    use fletch::RefLocator;

    let locator = RefLocator::Purl(purl.to_string());
    // This happens before cleave sees the initial bytes. Keep it distinct from
    // `fetch+graft`, which is the later dependency-fetch phase inside the
    // report pipeline.
    if let Some(p) = phase {
        p.set("purl:registry");
    }
    let (registry, registry_sources) = crate::fetch::registry_with_sources(&locator);
    let registry_provenance = registry.clone().map(|record| {
        crate::provenance::RegistryProvenance::from_record_sources(record, &registry_sources)
    });

    if let Some(reg) = &registry
        && reg.version_removed == Some(true)
        && let Some((name, bytes)) = crate::fetch::registry_document(reg)
    {
        if let Some(p) = phase {
            p.set("purl:registry-document");
        }
        let hopper_route =
            offer_registry_fallback(corpus, uploader, purl, &name, registry_provenance.as_ref());
        let mut result = classify_bytes_with_follow(
            bytes::Bytes::from(bytes),
            &name,
            resources,
            slow_rule_ms,
            cancellation,
            phase,
            registry_provenance.as_ref(),
            follow,
            deps_for_upload,
        )?;
        result.hopper_route = hopper_route;
        return Ok(result);
    }

    if let Some(p) = phase {
        p.set("purl:payload");
    }
    let (bytes, name, rec) = match crate::fetch::fetch_one(locator, false) {
        Ok(t) => t,
        Err(e) => match registry.as_ref().and_then(crate::fetch::registry_document) {
            Some((name, bytes)) => {
                if let Some(p) = phase {
                    p.set("purl:registry-document");
                }
                let hopper_route = offer_registry_fallback(
                    corpus,
                    uploader,
                    purl,
                    &name,
                    registry_provenance.as_ref(),
                );
                let mut result = classify_bytes_with_follow(
                    bytes::Bytes::from(bytes),
                    &name,
                    resources,
                    slow_rule_ms,
                    cancellation,
                    phase,
                    registry_provenance.as_ref(),
                    follow,
                    deps_for_upload,
                )?;
                result.hopper_route = hopper_route;
                return Ok(result);
            }
            None => return Err(e),
        },
    };

    let result = classify_bytes_with_follow(
        bytes::Bytes::from(bytes),
        &name,
        resources,
        slow_rule_ms,
        cancellation,
        phase,
        registry_provenance.as_ref(),
        follow,
        deps_for_upload,
    )?;

    // Offer the artifact — bytes, registry record, and fetch provenance —
    // before its verdict, exactly as the CLI (`scan purl --hopper`) and the
    // pull worker do. Deliberately after analysis, not on fetch: hopper's
    // upload-tier claim query drains bytes-with-no-verdict rows first and
    // ahead of everything else, so offering them before this process has its
    // own verdict in hand would race the sample onto the worker fleet's claim
    // queue for a redundant analysis. Hopper drops a result for a SHA it
    // never ingested, so the verdict alone lands nowhere either — queuing
    // artifacts then result keeps the row unclaimable for only the width of
    // the queue, not the width of an analysis.
    if let Some(uploader) = uploader {
        uploader.submit_artifacts(crate::engine::collect_upload_artifacts(
            std::path::Path::new(&name),
            &result.sha256,
            result.size_bytes,
            upload_collector(),
            registry_provenance.as_ref(),
            Some(&rec),
        ));
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn classify_url(
    url: &str,
    resources: &super::ModelResources,
    slow_rule_ms: u64,
    cancellation: Option<&Arc<AtomicBool>>,
    phase: Option<&cleave::PhaseTracker>,
    deps_for_upload: bool,
    uploader: Option<&Arc<crate::upload::Uploader>>,
    follow: crate::fetch::FetchPolicy,
) -> anyhow::Result<crate::engine::ScanResult> {
    use fletch::RefLocator;

    if let Some(p) = phase {
        p.set("url:payload");
    }
    let (bytes, name, rec) = crate::fetch::fetch_one(RefLocator::Url(url.to_owned()), false)
        .map_err(|e| anyhow::anyhow!(e))?;
    let result = classify_bytes_with_follow(
        bytes::Bytes::from(bytes),
        &name,
        resources,
        slow_rule_ms,
        cancellation,
        phase,
        None,
        follow,
        deps_for_upload,
    )?;
    // See the matching comment in `classify_purl` on why this waits for the
    // verdict rather than firing on fetch.
    if let Some(uploader) = uploader {
        uploader.submit_artifacts(crate::engine::collect_upload_artifacts(
            std::path::Path::new(&name),
            &result.sha256,
            result.size_bytes,
            upload_collector(),
            None,
            Some(&rec),
        ));
    }
    Ok(result)
}

/// The collector name recorded on every artifact this server files, matching
/// the CLI's `scan+<worker>` form so hopper's provenance reads the same
/// whichever path ingested the sample.
fn upload_collector() -> &'static str {
    static COLLECTOR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    COLLECTOR.get_or_init(|| format!("scan+{}", crate::upload::default_worker_name()))
}

/// Run the full cleave + litmus pipeline on `path`, returning a `ScanResult`.
///
/// Runs on a blocking thread. `label` is used as the `path` field in the result
/// (the original upload filename, not the temp file path).
#[allow(clippy::too_many_arguments)] // one linear analysis path; splitting would only scatter it
pub(crate) fn classify_file(
    path: &std::path::Path,
    label: &str,
    resources: &super::ModelResources,
    slow_rule_ms: u64,
    extract_dir: Option<&std::path::Path>,
    cancellation: Option<&Arc<AtomicBool>>,
    phase: Option<&cleave::PhaseTracker>,
    root_registry: Option<&crate::provenance::RegistryProvenance>,
    deps_for_upload: bool,
) -> anyhow::Result<ScanResult> {
    classify_file_with_follow(
        path,
        label,
        resources,
        slow_rule_ms,
        extract_dir,
        cancellation,
        phase,
        root_registry,
        resources.fetch,
        deps_for_upload,
    )
}

#[allow(clippy::too_many_arguments)]
fn classify_file_with_follow(
    path: &std::path::Path,
    label: &str,
    resources: &super::ModelResources,
    slow_rule_ms: u64,
    extract_dir: Option<&std::path::Path>,
    cancellation: Option<&Arc<AtomicBool>>,
    phase: Option<&cleave::PhaseTracker>,
    root_registry: Option<&crate::provenance::RegistryProvenance>,
    follow: crate::fetch::FetchPolicy,
    deps_for_upload: bool,
) -> anyhow::Result<ScanResult> {
    use anyhow::Context as _;

    if let Some(p) = phase {
        p.set("cleave:init");
    }
    // Every scan entry point declares this before its first analysis (see
    // `engine::classify_*`); the daemons reach cleave through here, so this is
    // theirs. It is not just a memory setting: it is part of cleave's analysis
    // cache key, so a daemon that left it at the default analyzed its first
    // sample under one key and every later one under another — orphaning that
    // first entry, and making the first response's member shape differ from the
    // rest.
    cleave::set_compact_member_retention(true); // compact projection only
    let sample_extraction =
        extract_dir.map(|d| cleave::SampleExtractionConfig::new(d.to_path_buf()));
    let mut opts = cleave::AnalysisOptions {
        slow_rule_ms,
        sample_extraction,
        cancellation: cancellation.cloned(),
        phase: phase.cloned(),
        ..Default::default()
    };
    crate::engine::add_zip_passwords(&mut opts, resources.zip_passwords.as_slice());
    let report =
        cleave::analyze_file(path, &opts).with_context(|| format!("cleave analysis of {label}"))?;
    finish_classify(
        label,
        report,
        resources,
        cancellation,
        phase,
        root_registry,
        follow,
        deps_for_upload,
    )
}

/// Like [`classify_file`] but operates on in-memory data, avoiding disk I/O.
///
/// Adopts the refcounted `data` buffer via `cleave::analyze_bytes_shared`, which
/// moves it into the analysis pipeline with no copy. At worker scale this
/// eliminates one full-size memcpy per downloaded sample.
#[allow(clippy::too_many_arguments)] // one linear analysis path; splitting would only scatter it
pub(crate) fn classify_bytes(
    data: bytes::Bytes,
    label: &str,
    resources: &super::ModelResources,
    slow_rule_ms: u64,
    cancellation: Option<&Arc<AtomicBool>>,
    phase: Option<&cleave::PhaseTracker>,
    root_registry: Option<&crate::provenance::RegistryProvenance>,
    deps_for_upload: bool,
) -> anyhow::Result<ScanResult> {
    classify_bytes_with_follow(
        data,
        label,
        resources,
        slow_rule_ms,
        cancellation,
        phase,
        root_registry,
        resources.fetch,
        deps_for_upload,
    )
}

#[allow(clippy::too_many_arguments)]
fn classify_bytes_with_follow(
    data: bytes::Bytes,
    label: &str,
    resources: &super::ModelResources,
    slow_rule_ms: u64,
    cancellation: Option<&Arc<AtomicBool>>,
    phase: Option<&cleave::PhaseTracker>,
    root_registry: Option<&crate::provenance::RegistryProvenance>,
    follow: crate::fetch::FetchPolicy,
    deps_for_upload: bool,
) -> anyhow::Result<ScanResult> {
    use anyhow::Context as _;

    if let Some(p) = phase {
        p.set("cleave:init");
    }
    // See `classify_file`: this is part of cleave's cache key, not only a
    // retention setting.
    cleave::set_compact_member_retention(true); // compact projection only
    let mut opts = cleave::AnalysisOptions {
        slow_rule_ms,
        cancellation: cancellation.cloned(),
        phase: phase.cloned(),
        ..Default::default()
    };
    crate::engine::add_zip_passwords(&mut opts, resources.zip_passwords.as_slice());
    let report = cleave::analyze_bytes_shared(data, label, &opts)
        .with_context(|| format!("cleave analysis of {label}"))?;
    finish_classify(
        label,
        report,
        resources,
        cancellation,
        phase,
        root_registry,
        follow,
        deps_for_upload,
    )
}

/// Shared tail of [`classify_file`]/[`classify_bytes`]: honor a late cancellation,
/// run feature extraction + model inference, and assemble the [`ScanResult`].
/// `deps_for_upload` marks callers that renew results on hopper and therefore
/// need per-dependency standalone reports captured (worker, `--hopper` server).
#[allow(clippy::too_many_arguments)] // shared linear tail for file and byte analyses
fn finish_classify(
    label: &str,
    report: cleave::AnalysisReport,
    resources: &super::ModelResources,
    cancellation: Option<&Arc<AtomicBool>>,
    phase: Option<&cleave::PhaseTracker>,
    root_registry: Option<&crate::provenance::RegistryProvenance>,
    follow: crate::fetch::FetchPolicy,
    deps_for_upload: bool,
) -> anyhow::Result<ScanResult> {
    // If the timeout fired while cleave was running, bail now rather than
    // burning CPU on feature extraction and model inference for a result
    // nobody is waiting for.
    if cancellation.is_some_and(|c| c.load(Ordering::Relaxed)) {
        anyhow::bail!("analysis cancelled");
    }

    // classify_report stages its own census labels from here: "fetch+graft"
    // through dependency fetch+analysis, then "features+model".
    let cr = crate::engine::classify_report(
        label,
        report,
        &resources.ctx,
        &resources.model,
        resources.shap.as_ref(),
        cancellation,
        &cleave::output::TinyOpts::tiny(),
        resources.interpret.as_ref(),
        // The server analyzes uploaded bytes, not a disk file; the root
        // imperative hunt (which re-reads the path) is therefore skipped, but
        // declared references from the report are still fetched when the
        // operator enabled it. `label` is a best-effort path for that hunt.
        std::path::Path::new(label),
        follow,
        resources.zip_passwords.as_slice(),
        // Server output is the JSON envelope: no renders, no fetch log, no
        // manifest listing. Dependency results are captured only for callers
        // that renew results on hopper (worker, `serve --hopper`).
        crate::engine::OutputNeeds {
            deps_for_upload,
            ..Default::default()
        },
        // Registry metadata collected at fetch time (worker provenance /
        // `--registry-map`), so a hopper-sourced scan reasons over the same
        // registry facts a live `pkg`/`url` scan fetches — without refetching.
        root_registry,
        None, // uploaded bytes have no scan-side acquisition fetch record
        None, // server returns JSON; the inline terminal bloom flag doesn't apply
        phase,
    )?;

    Ok(scan_result_from(label, cr, resources))
}

/// Build a [`ScanResult`] from a classified report.
fn scan_result_from(
    label: &str,
    cr: crate::engine::ClassifiedReport,
    resources: &super::ModelResources,
) -> ScanResult {
    ScanResult {
        v: "7",
        classification: cr.classification,
        probability: cr.probability,
        threshold: cr.threshold,
        level: cr.level,
        analysis_cached: cr.analysis_cached,
        version: crate::engine::model_version_string(resources.model.info()),
        analyzed_at: crate::engine::now_rfc3339(),
        cleave: Some(cr.report),
        pids: None,
        deleted: None,
        path: label.to_string(),
        finding_counts: cr.finding_counts,
        formula: cr.formula,
        reasons: cr.reasons,
        top_findings: cr.top_findings,
        model_scores: cr.model_scores,
        skipped_models: cr.skipped_models,
        file_type: cr.file_type,
        size_bytes: cr.size_bytes,
        sha256: cr.sha256,
        embedded_files: cr.embedded_files,
        rendered_context: cr.rendered_context,
        interpretation: cr.interpretation,
        dependency_results: cr.dependency_results,
        bloom_mark: None,
        hopper_route: crate::engine::HopperRoute::Normal,
    }
}

// --- /analyze-path endpoint ---

#[derive(serde::Deserialize)]
pub(super) struct AnalyzePathRequest {
    path: String,
    /// Optional registry provenance for this file, in any shape
    /// [`crate::provenance::registry_provenance`] accepts (a hopper sidecar, a
    /// bare fletch envelope, or a normalized record). This is the server-side
    /// equivalent of the CLI's `--registry-map` entry for the same sha: it lets
    /// a caller that already holds the facts — promoter fetches them from
    /// hopper — hand them over instead of making the scan refetch or go without.
    /// Absent or unparseable means the scan simply runs without registry facts,
    /// exactly as it did before this field existed.
    #[serde(default)]
    registry: Option<Box<serde_json::value::RawValue>>,
}

/// POST /analyze-path — analyze a file by its on-disk path.
///
/// Accepts `{"path": "/full/path/to/file", "registry": {...}}` (registry
/// optional). The path must be under one of the directories specified by
/// `--allowed-dirs`. Returns the same `{"ml": {...}, "raw": {...}}` envelope as
/// `/analyze`.
pub(super) async fn analyze_path(
    State(state): State<Arc<AppState>>,
    Extension(request_id): Extension<RequestId>,
    Json(req): Json<AnalyzePathRequest>,
) -> Response {
    state.note_analyze_request();
    // Attached around the whole handler rather than at each return: this route
    // rejects from several places — not found, not under an allowed dir, under
    // memory pressure — and a rejected path is the one an operator most needs
    // named. The path is as the caller wrote it; where the canonical form
    // differs, the rejection line below carries both.
    let subject = Subject::path(&req.path);
    with_subject(
        analyze_path_inner(state, request_id.get(), req).await,
        subject,
    )
}

async fn analyze_path_inner(
    state: Arc<AppState>,
    request_id: u64,
    req: AnalyzePathRequest,
) -> Response {
    let request_start = Instant::now();

    let raw_path = std::path::PathBuf::from(&req.path);

    // Resolve symlinks and canonicalize BEFORE the allowed-dirs check to
    // prevent symlink-based path traversal (e.g., /allowed/link → /etc/shadow).
    let Ok(path) = raw_path.canonicalize() else {
        return error_response(StatusCode::NOT_FOUND, "File not found");
    };

    // Validate the canonical (symlink-resolved) path is under an allowed directory.
    if state.allowed_dirs.is_empty() || !state.allowed_dirs.iter().any(|dir| path.starts_with(dir))
    {
        tracing::warn!(id = request_id, path = %req.path, canonical = %path.display(), "analyze-path rejected: not under allowed dirs");
        return error_response(StatusCode::FORBIDDEN, "Path not under allowed directories");
    }

    if !path.is_file() {
        return error_response(StatusCode::NOT_FOUND, "File not found");
    }

    // Check memory pressure.
    if let Some(resp) = check_memory_pressure(&state).await {
        return resp;
    }

    // Ensure resources are loaded.
    let resources = {
        let Ok(guard) = state.resources.read() else {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "Internal error");
        };
        match guard.as_ref() {
            Some(r) => Arc::clone(r),
            None => {
                return error_response(StatusCode::SERVICE_UNAVAILABLE, "Server starting up");
            }
        }
    };

    let file_size = path.metadata().map(|m| m.len()).unwrap_or(0);
    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    tracing::info!(
        id = request_id,
        path = %req.path,
        size_bytes = file_size,
        "--> POST /analyze-path",
    );

    // Claim a slot — same RAII semaphore pattern as /analyze.
    let Ok(permit) = Arc::clone(&state.slots).try_acquire_owned() else {
        let max = state.max_concurrent_tasks;
        tracing::warn!(
            id = request_id,
            filename = %filename,
            size_bytes = file_size,
            max,
            "rejecting: at capacity"
        );
        return error_response(
            StatusCode::TOO_MANY_REQUESTS,
            format!("At capacity ({max}/{max} active analyses)"),
        );
    };

    let slow_rule_ms = state.slow_rule_ms;

    let should_clear_caches = request_id.is_multiple_of(100);
    let extract_dir = state.extract_dir.clone();
    let cancellation = Arc::new(AtomicBool::new(false));
    state.in_flight.insert(
        request_id,
        super::InFlightRequest {
            name: req.path.clone(),
            size_bytes: file_size,
            started_at: Instant::now(),
            cancellation: Arc::clone(&cancellation),
            phase: cleave::PhaseTracker::with_label(format!("req#{request_id} {}", req.path)),
            thread_id: AtomicU64::new(0),
        },
    );

    // RAII guard: signals cancellation and removes the in-flight entry on drop,
    // covering both normal completion and client-disconnect cancellation.
    let guard = super::RequestGuard::new(
        request_id,
        Arc::clone(&state),
        Arc::clone(&cancellation),
        permit,
    );

    // Keep a path clone for the hopper renewal (the analysis closure moves
    // `path`); only when --hopper is configured, so the common case pays nothing.
    let upload_path = state.uploader.as_ref().map(|_| path.clone());
    let deps_for_upload = upload_path.is_some();
    let cancel_flag = Arc::clone(&cancellation);
    let phase_state = Arc::clone(&state);
    let phase_tracker = phase_state
        .in_flight
        .get(&request_id)
        .map(|r| r.phase.clone());

    // Registry provenance the caller supplied for this file, parsed before the
    // analysis thread starts so a malformed document costs nothing downstream.
    // Provenance enriches a scan but is never required, so an unparseable
    // document degrades to a warning and a registry-less scan rather than a 400.
    let root_registry = req.registry.as_ref().and_then(|raw| {
        let provenance = crate::provenance::registry_provenance(raw.get().as_bytes());
        if provenance.is_none() {
            tracing::warn!(
                id = request_id,
                path = %req.path,
                "analyze-path registry provenance carries no record; scanning without it",
            );
        }
        provenance
    });

    let handle = tokio::task::spawn_blocking(move || {
        if let Some(req) = phase_state.in_flight.get(&request_id) {
            req.thread_id.store(current_thread_id(), Ordering::Relaxed);
        }
        let result = classify_file(
            &path,
            &filename,
            &resources,
            slow_rule_ms,
            extract_dir.as_deref(),
            Some(&cancel_flag),
            phase_tracker.as_ref(),
            root_registry.as_ref(), // caller-supplied, the server-side `--registry-map` equivalent
            deps_for_upload,        // dependencies ride the hopper renewal below
        );
        if should_clear_caches {
            cleave::clear_all_thread_caches();
        }
        result
    });

    // Await to completion, bounded by the configured per-request timeout.
    // See `analyze` for the timeout-drop-slot semantics.
    let result = await_with_timeout(
        handle,
        state.analysis_timeout_secs,
        &cancellation,
        &state.stuck_orphans,
    )
    .await;
    drop(guard);

    let elapsed_ms = crate::duration_ms(request_start.elapsed());

    match result {
        AnalysisOutcome::Ok(Ok(scan_result)) => {
            let mut scan_result = *scan_result;
            let llm = llm_source(scan_result.interpretation.as_ref());
            let analysis = analysis_source(&scan_result);
            // Record where archive members were extracted on disk, so cyclotron
            // can open them.
            if let (Some(extract_dir), Some(raw)) = (&state.extract_dir, &mut scan_result.cleave)
                && let Some(first) = raw.files.first().map(|f| f.sha.as_str())
            {
                // SHA hex is ASCII; byte slice is always a valid UTF-8 boundary.
                let short = first.get(..first.len().min(6)).unwrap_or(first);
                let dir = extract_dir.join(short);
                if dir.is_dir() {
                    raw.extracted_path = Some(dir.to_string_lossy().into_owned());
                }
            }

            tracing::info!(
                id = request_id,
                path = %req.path,
                elapsed_ms,
                classification = %scan_result.classification,
                probability = scan_result.probability,
                analysis,
                llm,
                // Where this verdict goes next; see the same field on the
                // flight path. A local file has no locator, so the uploader's
                // own line names it by digest alone.
                hopper = if state.uploader.is_some() {
                    "queued"
                } else {
                    "disabled"
                },
                "<-- 200 OK",
            );
            // Renew the result on hopper when --hopper is set. Serialize the
            // response body first so the (possibly large) envelope moves to the
            // uploader without a clone; the renewal runs off the executor since
            // collect_upload_artifacts reads sidecars from disk.
            index_verdict(&scan_result, None);
            let sha256 = scan_result.sha256.clone();
            let size = scan_result.size_bytes;
            let deps = std::mem::take(&mut scan_result.dependency_results);
            let envelope = scan_result.into_envelope();
            let body = serde_json::to_vec(&envelope).unwrap_or_else(|_| b"{}".to_vec());
            if let (Some(uploader), Some(path)) = (&state.uploader, upload_path) {
                let uploader = Arc::clone(uploader);
                tokio::task::spawn_blocking(move || {
                    crate::engine::upload_scan_result(
                        &uploader, &path, sha256, size, None, None, deps, envelope,
                    );
                });
            }
            let mut resp = (
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                body,
            )
                .into_response();
            resp.headers_mut().insert("X-Total-Ms", elapsed_ms.into());
            resp
        }
        AnalysisOutcome::Ok(Err(e)) => {
            let (status, response) = analysis_error_response(&e);
            tracing::warn!(id = request_id, path = %req.path, elapsed_ms, status = status.as_u16(), error = %error_chain(&e), "<-- analysis failed");
            response
        }
        AnalysisOutcome::JoinError(e) => {
            tracing::warn!(id = request_id, elapsed_ms, error = %e, "<-- 500 task join error (panic?)");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "Internal error")
        }
        AnalysisOutcome::Timeout(secs) => {
            tracing::warn!(
                id = request_id,
                elapsed_ms,
                timeout_secs = secs,
                "<-- 504 analysis timeout",
            );
            (
                StatusCode::GATEWAY_TIMEOUT,
                Json(serde_json::json!({
                    "error": "analysis timeout",
                    "timeout_secs": secs,
                })),
            )
                .into_response()
        }
    }
}

/// Check RSS memory pressure and attempt recovery before rejecting requests.
///
/// Returns `Some(Response)` if the request should be rejected due to memory pressure,
/// or `None` if the server has enough memory to proceed.
async fn check_memory_pressure(state: &AppState) -> Option<Response> {
    // Throttling disabled: never reject on memory pressure. The operator has
    // delegated OOM enforcement to an external supervisor.
    let max_rss_bytes = state.max_rss_bytes?.get();
    let rss = cleave::memory_tracker::current_rss()?;

    if rss <= max_rss_bytes {
        // Happy path: reset overload timer if set.
        if let Ok(mut overloaded) = state.overloaded_since.try_lock()
            && overloaded.is_some()
        {
            tracing::info!(
                rss_mb = rss / 1024 / 1024,
                "memory recovered below threshold"
            );
            *overloaded = None;
        }
        return None;
    }

    // Memory pressure detected — try to reclaim by clearing thread-local caches.
    tracing::info!(
        rss_mb = rss / 1024 / 1024,
        "memory pressure detected, clearing thread-local caches"
    );
    // Await the clear before re-checking RSS. A prior version fired-and-forgot
    // via `drop(spawn_blocking(...))`, then immediately re-read RSS on the next
    // line — which produced a "cache clear freed memory" log before the clear
    // had actually run, and let overloaded workers admit requests they could
    // not service.
    if let Err(e) = tokio::task::spawn_blocking(cleave::clear_all_thread_caches).await {
        tracing::warn!(error = %e, "cache-clear task failed");
    }

    // Re-check after clearing caches.
    let rss_after = cleave::memory_tracker::current_rss()?;
    if rss_after <= max_rss_bytes {
        if let Ok(mut overloaded) = state.overloaded_since.try_lock() {
            *overloaded = None;
        }
        tracing::info!(
            rss_before_mb = rss / 1024 / 1024,
            rss_after_mb = rss_after / 1024 / 1024,
            "cache clear freed memory, accepting request"
        );
        return None;
    }

    // Still overloaded — track duration and log. Never terminate; let the operator
    // decide when to restart. Requests continue to be rejected with 503 until
    // memory drops below the threshold.
    // Use try_lock: if a concurrent request holds the lock it is already recording
    // the overload timestamp, so it is safe to pass through rather than block a
    // tokio worker on a std::sync::Mutex.
    let since = *state
        .overloaded_since
        .try_lock()
        .ok()?
        .get_or_insert_with(Instant::now);
    let overloaded_secs = since.elapsed().as_secs();

    tracing::warn!(
        rss_mb = rss_after / 1024 / 1024,
        max_rss_mb = max_rss_bytes / 1024 / 1024,
        overloaded_secs,
        "server overloaded: high memory usage (even after cache clear)"
    );
    Some(error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "Server overloaded (memory)",
    ))
}

/// GET /_/memory — memory diagnostics for all major structures.
///
/// `process.jemalloc` is null unless cleave was built with `--features jemalloc`.
/// When available, `jemalloc.allocated_mb` is the most useful leak indicator:
/// if it tracks RSS closely, you have a real leak; if RSS >> allocated, it's fragmentation.
pub(super) async fn memory_stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let rss_mb = cleave::memory_tracker::current_rss().map(|b| b / 1024 / 1024);

    let jemalloc = cleave::memory_tracker::jemalloc_stats().map(|s| {
        serde_json::json!({
            "allocated_mb":     s.allocated / 1024 / 1024,
            "active_mb":        s.active    / 1024 / 1024,
            "metadata_mb":      s.metadata  / 1024 / 1024,
            "resident_mb":      s.resident  / 1024 / 1024,
            "retained_mb":      s.retained  / 1024 / 1024,
            "fragmentation_mb": s.active.saturating_sub(s.allocated) / 1024 / 1024,
        })
    });

    Json(serde_json::json!({
        "process": {
            "rss_mb": rss_mb,
            "max_rss_mb": state.max_rss_bytes.map(|n| n.get() / 1024 / 1024),
            "jemalloc": jemalloc,
        },
        "server": {
            "active_tasks": state.max_concurrent_tasks.saturating_sub(state.slots.available_permits()),
            "stuck_orphans": state.stuck_orphans.load(Ordering::Relaxed),
            "max_concurrent_tasks": state.max_concurrent_tasks,
            "requests_total": state.next_request_id.load(Ordering::Relaxed),
        },
        "thread_pools": {
            "rayon_threads": rayon::current_num_threads(),
        },
    }))
}

/// GET /_/requests — all analyses currently in flight, sorted by elapsed time descending.
/// GET /status?sha256=… | ?purl=… — where an analysis of this artifact stands.
///
/// Exists for the caller whose connection did not survive the analysis. The run
/// keeps going here when a proxy gives up at its own ceiling, but from outside
/// a run in progress and a run that never started are both `404 unknown sample`
/// on /lookup. That ambiguity is the whole problem: it is the difference
/// between waiting a little longer and paying for a twenty-minute analysis
/// twice.
///
/// Running is reported before complete, so a caller is never told to go away
/// while a run it could ride is still live. The reverse order has a window —
/// between a flight publishing and its verdict reaching the index — where a
/// live run reads as `unknown`.
///
/// `lost` is deliberately not a state. It is the caller's own inference: they
/// dispatched, their connection died, and this answers `unknown`. Reporting it
/// here would mean keeping a graveyard of every run that ever ended, to tell a
/// caller something they already know.
pub(super) async fn status(
    State(state): State<Arc<AppState>>,
    Query(q): Query<LookupQuery>,
) -> Response {
    let sha = q.sha256.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let raw = q.purl.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let raw_url = q.url.as_deref().map(str::trim).filter(|s| !s.is_empty());
    if raw.is_some() && raw_url.is_some() {
        return error_response(StatusCode::BAD_REQUEST, "provide purl or url, not both");
    }
    if let Some(url) = raw_url
        && !valid_http_url(url)
    {
        return error_response(StatusCode::BAD_REQUEST, "invalid url");
    }
    if sha.is_none() && raw.is_none() && raw_url.is_none() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "provide sha256, purl, url, or both",
        );
    }
    // The canonical form is what a flight is keyed by, so an uncanonical spelling
    // must not read as a different artifact — the same rule /lookup follows.
    let purl = match raw.map(normalize_pkg_purl) {
        Some(Ok(purl)) => Some(purl),
        Some(Err(message)) => return error_response(StatusCode::BAD_REQUEST, message),
        None => None,
    };
    let sha = sha.map(str::to_ascii_lowercase);
    let url = raw_url.map(str::to_owned);

    let running = purl
        .as_deref()
        .and_then(|p| state.flights.running(&FlightKey::Purl(p.to_string())))
        .or_else(|| {
            url.as_deref()
                .and_then(|u| state.flights.running(&FlightKey::Url(u.to_string())))
        })
        .or_else(|| {
            sha.as_deref()
                .and_then(|s| state.flights.running(&FlightKey::Sha(s.to_string())))
        });
    if let Some(run) = running {
        return Json(serde_json::json!({
            "state": "running",
            "purl": purl,
            "url": url,
            "sha256": sha,
            "elapsed_ms": crate::duration_ms(run.elapsed),
            "attached": run.attached,
        }))
        .into_response();
    }

    let index = crate::lookup::global();
    let complete = index.is_some_and(|index| {
        purl.as_deref().is_some_and(|p| index.get_purl(p).is_some())
            || sha.as_deref().is_some_and(|s| index.get_sha(s).is_some())
    });
    Json(serde_json::json!({
        "state": if complete { "complete" } else { "unknown" },
        "purl": purl,
        "url": url,
        "sha256": sha,
    }))
    .into_response()
}

/// Query for `GET /v1/lookup`. `purl` and `url` repeat; `sha256` names one artifact.
///
/// Parsed from the raw query rather than through `Query<T>`: a repeated key is
/// a sequence, and `serde_urlencoded` — what axum's `Query` is built on —
/// cannot deserialize one. It silently rejects the whole request instead, which
/// would make `?purl=a&purl=b` a 400 with no explanation.
pub(super) struct V1LookupQuery {
    purl: Vec<String>,
    url: Vec<String>,
    sha256: Option<String>,
    /// How many false positives per 100 million benign files the caller will
    /// tolerate. Chosen by them, unlike `fires_at`, which is measured.
    false_positive_budget: Option<u16>,
    /// A budget that was sent but is not a number. Held rather than silently
    /// defaulted: a caller who meant to loosen their budget and got the strict
    /// default back would see verdicts they never asked for and never learn why.
    bad_budget: Option<String>,
    /// Whether the caller insists on a fresh run.
    ///
    /// `/v1/analyze` answers from a verdict it already holds, which is what
    /// makes asking twice cheap. Somebody re-checking an artifact under a new
    /// engine needs a way to say so, and without one the only way to force a
    /// re-analysis would be to have no verdict — which is not a state a caller
    /// can arrange. Meaningless to `/v1/lookup`, which never analyzes.
    force: bool,
    /// Whether the caller wants the authoritative answer rather than the cheap
    /// one.
    ///
    /// Distinct from [`Self::force`], which is about spending an analysis slot.
    /// This is about which layer may answer: the bloom filters are membership
    /// rebuilt on a schedule, so a caller who needs current truth — reading
    /// after a write, or checking whether a revocation has landed — must be able
    /// to say "not from a filter". It bypasses both bloom paths and applies to
    /// `/v1/lookup` as much as to `/v1/analyze`, because a stale bless is a
    /// lookup problem too.
    ///
    /// Spelled to match hopper's own escape hatch (`?fresh=1`), so one word
    /// means the same thing at both hops.
    fresh: bool,
    /// Which references discovered inside the root artifact the caller wants
    /// followed. Repeated keys and comma-separated values are both accepted.
    /// Empty means use the deployment policy.
    follow: Vec<String>,
}

/// The spellings that opt in to a boolean flag. Anything else — including a
/// bare `=` — leaves the default in place, because the reading that costs
/// something must never be reached by an ambiguous value.
fn affirmative(value: &str) -> bool {
    matches!(value, "1" | "true" | "yes")
}

/// `X-Hopper-Fresh`, the header spelling of `?fresh=1`.
///
/// Named for hopper's own escape hatch rather than for scan, because it is the
/// same request travelling: a caller sets it once and every hop that can answer
/// from something cheaper stands down. Accepts the same spellings the query
/// parameter does — hopper itself only reads `1`, and accepting a superset here
/// costs nothing and surprises nobody.
fn header_wants_fresh(headers: &axum::http::HeaderMap) -> bool {
    headers
        .get("x-hopper-fresh")
        .and_then(|v| v.to_str().ok())
        .is_some_and(affirmative)
}

impl V1LookupQuery {
    /// Fold the header alias into the parsed query. Either spelling opts in;
    /// neither can opt back out, so a proxy that adds the header cannot be
    /// defeated by a stale `fresh=0` further down the chain.
    fn with_fresh_header(mut self, headers: &axum::http::HeaderMap) -> Self {
        self.fresh = self.fresh || header_wants_fresh(headers);
        self
    }

    fn parse(raw: Option<&str>) -> Self {
        let mut q = Self {
            purl: Vec::new(),
            url: Vec::new(),
            sha256: None,
            false_positive_budget: None,
            bad_budget: None,
            force: false,
            fresh: false,
            follow: Vec::new(),
        };
        for (key, value) in form_urlencoded::parse(raw.unwrap_or("").as_bytes()) {
            match key.as_ref() {
                "purl" => q.purl.push(value.into_owned()),
                "url" => q.url.push(value.into_owned()),
                "sha256" => q.sha256 = Some(value.into_owned()),
                "false_positive_budget" => match value.parse::<u16>() {
                    Ok(n) => q.false_positive_budget = Some(n),
                    Err(_) => q.bad_budget = Some(value.into_owned()),
                },
                // Only an affirmative spelling forces a run. Anything else —
                // including `force=0` and a bare `force=` — leaves the cheap
                // path in place, because the expensive reading of an ambiguous
                // value is the one that burns an analysis slot.
                "force" => q.force = matches!(value.as_ref(), "1" | "true" | "yes"),
                // Same affirmative-only rule as `force`, for the same reason:
                // an ambiguous value must not silently change which layer
                // answers.
                "fresh" => q.fresh = affirmative(value.as_ref()),
                "follow" => q.follow.push(value.into_owned()),
                // Unknown parameters are ignored, so a caller can carry their
                // own tracing keys through without us rejecting the request.
                _ => {}
            }
        }
        q
    }
}

/// Resolve a request's follow selection. The configured policy supplies the
/// default and the operational limits; an explicit request replaces only the
/// selected reference categories after its syntax has been validated.
fn v1_follow_policy(
    q: &V1LookupQuery,
    configured: crate::fetch::FetchPolicy,
) -> Result<crate::fetch::FetchPolicy, (&'static str, String)> {
    if q.follow.is_empty() {
        return Ok(configured);
    }
    let selected = crate::fetch::FetchPolicy::parse_follow(&q.follow.join(","))
        .map_err(|message| ("invalid_follow_policy", message))?;
    Ok(configured.with_selection(selected))
}

/// How many packages one URL may name.
///
/// A PURL runs about fifty characters encoded, so fifty of them sits well
/// inside every intermediary's URL limit with room to spare. Past this the
/// answer is POST, and the error says so rather than leaving it to be
/// discovered by a truncated query string.
const V1_MAX_KEYS: usize = 50;

/// How long the ordinary response still applies.
///
/// Only long enough to catch an outcome that needed no work to reach — a
/// refusal, or a run that was already finished when this request joined it.
/// Capacity is refused the instant a slot is asked for, which is what keeps
/// `429 At capacity` a real 429 the router can act on rather than a decision
/// buried in a 200 body.
const V1_ANALYZE_GRACE: Duration = Duration::from_millis(250);

/// When the first progress frame goes out, for an analysis still running.
const V1_PROGRESS_FIRST: Duration = Duration::from_secs(1);

/// How often progress is reported after that.
const V1_PROGRESS_EVERY: Duration = Duration::from_secs(5);

/// Analyze the bytes a caller sent, rather than a package they named.
///
/// The digest is the identity, so two callers uploading the same artifact share
/// one analysis exactly as two callers naming one PURL do. `?purl=` may still
/// accompany the bytes: scan grafts the registry provenance onto the report,
/// and it is echoed in each finding's `pkg`.
async fn v1_analyze_bytes(
    state: &Arc<AppState>,
    request_id: u64,
    q: &V1LookupQuery,
    headers: axum::http::HeaderMap,
    bytes: bytes::Bytes,
    request_start: Instant,
    request_follow: RequestFollow,
) -> Response {
    if let Ok(init_error) = state.init_error.read()
        && let Some(message) = init_error.as_ref()
    {
        tracing::error!(id = request_id, error = %message, "rejected: startup failed");
        return v1_error(StatusCode::SERVICE_UNAVAILABLE, "starting", message);
    }
    if let Some(response) = check_memory_pressure(state).await {
        return response;
    }

    let budget = q
        .false_positive_budget
        .unwrap_or_else(|| decision::default_budget(state.level));
    // The name only decides how cleave types the bytes, so a caller that sends
    // none still gets an analysis — of an artifact typed by content rather than
    // by extension.
    let filename = headers
        .get("x-filename")
        .and_then(|v| v.to_str().ok())
        .map(sanitize_upload_filename)
        .unwrap_or_else(|| format!("upload-{request_id}"));

    // Read whole rather than streamed to disk, and bounded by the same limit
    // the multipart path enforces: this is the small-artifact route — a caller
    // with gigabytes has a registry to publish to — and having every byte in
    // hand is what lets the digest be known before a temp file exists, so an
    // artifact already being analyzed costs no disk at all.
    let sha = format!("{:x}", Sha256::digest(&bytes));

    // Kept apart on purpose: `purl` is the key everything downstream is stored
    // and looked up by, `asked` is what the caller typed and what the answer is
    // spelled with. See `V1Decision::asked_about`.
    let asked = q.purl.first().map(String::as_str);
    let purl = match asked.map(normalize_pkg_purl) {
        Some(Ok(purl)) => Some(purl),
        Some(Err(message)) => {
            return v1_error(StatusCode::BAD_REQUEST, "invalid_purl", message);
        }
        None => None,
    };

    // Already answered?
    //
    // The digest is in hand before any temp file exists — which is the reason
    // the bytes are read whole rather than streamed — so this costs one index
    // probe against the very artifact the caller sent. Without it, re-uploading
    // something this worker has already analyzed pays for the whole analysis
    // again, which is the same omission the named path above carried until the
    // day this did.
    //
    // Resolved by digest and by digest ALONE. The PURL must not reach the
    // resolver here, and this is not a style preference — passing it was a
    // false negative with a CVE's shape.
    //
    // `v1_decide` hands both keys to the corpus, and hopper answers a
    // `?sha256=…&purl=…` query on either. So an upload of arbitrary bytes
    // carrying `?purl=` of a package the corpus knows came back with *that
    // package's* verdict: measured against production, 25 bytes of text sent as
    // `?purl=pkg:npm/chalk@5.3.0` were answered `allow` under chalk's digest,
    // and the bytes were never looked at. Anything can be laundered through a
    // reputable coordinate that way, which is precisely the attack this route
    // exists to catch.
    //
    // The index path was already safe — `pick_verdict` accepts a PURL's verdict
    // only when it describes the same bytes — and the corpus path had no such
    // guard. Now neither needs one: what is asked about is the digest, which is
    // the only thing an upload actually names. The caller's PURL is grafted
    // back on afterwards as provenance, which is all it ever was.
    //
    // Only a real verdict short-circuits. `unknown` means nobody has analyzed
    // these bytes, which is why the caller sent them, and `unavailable` means
    // we could not find out — turning either into an answer would report on
    // work never done.
    if !q.force && request_follow.persist {
        if let Ok((decided, source)) = v1_decide(state, Some(&sha), None, None, budget, q.fresh)
            .await
            .map(|(d, source)| (d.asked_about(asked), source))
            && decided.is_answerable()
        {
            let elapsed = crate::duration_ms(request_start.elapsed());
            tracing::info!(
                id = request_id,
                sha256 = %sha,
                size_bytes = bytes.len(),
                ms = elapsed,
                kind = if decided.is_verdict() { "verdict" } else { "derived" },
                "--> POST /v1/analyze (bytes; answered from what we already knew; no slot spent)"
            );
            let mut resp = Json(decided).into_response();
            resp.headers_mut().insert("X-Total-Ms", elapsed.into());
            resp.headers_mut().insert(
                "X-Scan-Source",
                axum::http::HeaderValue::from_static(source),
            );
            resp.extensions_mut().insert(match purl.as_deref() {
                Some(named) => Subject::purl(named, Some(&sha)),
                None => Subject::sha256(&sha),
            });
            return resp;
        }
        tracing::info!(
            id = request_id,
            sha256 = %sha,
            "no verdict held for these bytes; analyzing"
        );
    }

    let attachment = state.flights.join(FlightKey::sha_follow(
        sha.clone(),
        request_follow.policy.selection_bits(),
        state.fetch.selection_bits(),
    ));
    let leads = attachment.leads();
    if leads {
        tracing::info!(id = request_id, sha256 = %sha, size_bytes = bytes.len(), filename = %filename, "--> POST /v1/analyze (bytes)");
        let publisher = state.flights.publisher(attachment.flight());
        match claim_slot(state, request_id, attachment.flight().key()) {
            Err(outcome) => publisher.publish(outcome),
            Ok((resources, permit)) => match stage_upload(request_id, &filename, &bytes).await {
                Err(outcome) => publisher.publish(outcome),
                Ok(upload) => {
                    let flight = Arc::clone(attachment.flight());
                    let state = Arc::clone(state);
                    tokio::spawn(async move {
                        publisher.publish(
                            run_file_analysis(
                                state,
                                request_id,
                                upload,
                                &flight,
                                resources,
                                permit,
                                request_follow,
                            )
                            .await,
                        );
                    });
                }
            },
        }
    } else {
        tracing::info!(id = request_id, sha256 = %sha, "--> POST /v1/analyze (bytes; joined a run already in flight)");
    }

    let flight = Arc::clone(attachment.flight());
    // The digest labels the request in logs; it is never the locator.
    let named = Named {
        subject: purl.clone().unwrap_or_else(|| sha.clone()),
        key: purl,
        asked: asked.map(str::to_owned),
        is_url: false,
    };
    let follow_name = request_follow.policy.follow_name();
    match tokio::time::timeout(V1_ANALYZE_GRACE, flight.wait()).await {
        Ok(outcome) => {
            let elapsed = crate::duration_ms(request_start.elapsed());
            v1_outcome_response(
                &outcome,
                &named,
                budget,
                elapsed,
                !leads,
                follow_name.as_deref(),
            )
        }
        Err(_) => {
            tracing::info!(id = request_id, sha256 = %sha, "answering as a stream");
            v1_streamed(
                Arc::clone(state),
                attachment,
                flight,
                named,
                budget,
                request_start,
                follow_name,
            )
        }
    }
}

/// Write the uploaded bytes into a temp directory, named so cleave detects the
/// artifact's type from its extension.
async fn stage_upload(request_id: u64, filename: &str, bytes: &[u8]) -> Result<Upload, Outcome> {
    let dir =
        match tokio::task::spawn_blocking(|| TempBuilder::new().prefix("scan-").tempdir()).await {
            Ok(Ok(dir)) => dir,
            Ok(Err(e)) => {
                tracing::warn!(id = request_id, error = %e, "failed to create temp dir");
                return Err(Outcome::rendered(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal error",
                ));
            }
            Err(e) => {
                tracing::warn!(id = request_id, error = %e, "temp dir task join error (panic?)");
                return Err(Outcome::rendered(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal error",
                ));
            }
        };
    let path = dir.path().join(filename);
    if let Err(e) = tokio::fs::write(&path, bytes).await {
        tracing::warn!(id = request_id, path = %path.display(), error = %e, "failed to write upload");
        return Err(Outcome::rendered(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to save file data",
        ));
    }
    Ok(Upload {
        dir,
        path,
        filename: filename.to_string(),
        size_bytes: bytes.len(),
    })
}

/// POST /v1/analyze — analyze an artifact and answer with a decision.
///
/// The whole point of this route over `/analyze-purl` is that it survives being
/// slow. A proxy between us and the caller gives up on a silent connection —
/// measured at 125 seconds in front of this fleet — and tears it down, which
/// costs the caller an analysis that in fact completed: the worker finishes,
/// files its verdict, and answers the next asker in milliseconds, but the reply
/// to *this* request had nowhere to go.
///
/// So the answer starts before it is known. Nothing is sent for the first
/// [`V1_ANALYZE_GRACE`], because most analyses finish inside it and deserve an
/// ordinary response with an ordinary status code. Past that the response
/// begins — headers, then a space every [`V1_HEARTBEAT`] — and the connection
/// stops being idle, so nothing between here and the caller has cause to cut
/// it. The decision follows whenever the analysis lands.
pub(super) async fn v1_analyze(
    State(state): State<Arc<AppState>>,
    request_id: Extension<RequestId>,
    raw: axum::extract::RawQuery,
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
) -> Response {
    let request_id = request_id.0.get();
    let request_start = Instant::now();
    let q = V1LookupQuery::parse(raw.0.as_deref()).with_fresh_header(&headers);

    if !q.purl.is_empty() && !q.url.is_empty() {
        return v1_error(
            StatusCode::BAD_REQUEST,
            "multiple_locators",
            "Use ?purl= or ?url=, not both.",
        );
    }

    // Two ways to name an artifact, and the artifact itself is one of them: a
    // caller holding bytes nobody has published — a build output, something
    // pulled from a mirror, a file off disk — has nothing to locate them by.
    //
    // Which one is meant is decided by what arrived, not by a header the caller
    // has to remember: bytes are an artifact, and no bytes means the package
    // named in the query. `Content-Type` says nothing here that the presence of
    // a body does not, and requiring it only turns a correct request into a
    // 415 for a reason the caller cannot see.
    // Checked before anything is read: a budget that is not a number is the
    // caller's mistake whichever way they named the artifact, and answering it
    // only after the body has been read would make the same request a 400 or a
    // 503 depending on whether the model happened to be loaded.
    if let Some(bad) = q.bad_budget.as_deref() {
        return v1_error(
            StatusCode::BAD_REQUEST,
            "invalid_false_positive_budget",
            &format!("false_positive_budget must be a whole number from 0 to 65535, not {bad:?}."),
        );
    }
    let follow = match v1_follow_policy(&q, state.fetch) {
        Ok(policy) => policy,
        Err((code, message)) => return v1_error(StatusCode::BAD_REQUEST, code, &message),
    };
    // Everything this server analyses is filed, whatever policy produced it.
    //
    // This used to be `follow == state.fetch`: a result was shared only when the
    // caller's policy happened to equal the one this box was started with. The
    // intent was sound — the corpus holds one verdict per artifact, and a
    // narrower verdict overwriting a wider one would under-report it — but the
    // test was against a local default nobody coordinates. Beamline resolves
    // `references` for a PURL while an unflagged server defaults to
    // `dependencies,references`, so no ordinary request ever matched, and every
    // verdict the fleet produced was dropped: no bytes offered, no dependencies
    // mirrored, no result posted. The corpus could not grow from its own
    // traffic.
    //
    // Filing everything trades that for the opposite risk — a `follow=none`
    // verdict can now land on top of a wider one — and takes it knowingly,
    // because a corpus that records a shallower answer than it might have is
    // worth more than one that records nothing at all.
    //
    // TODO(t): Refactor the data model to allow realtime follow reassembly.
    // Dependencies, references, and CI actions belong in their own tables
    // rather than folded into one verdict; a caller's `follow=` is then a view
    // assembled from what is stored, and the question of which policy owns the
    // row stops being asked. Until then hopper is deliberately policy-blind and
    // the last writer wins.
    let request_follow = RequestFollow {
        policy: follow,
        persist: true,
    };
    let budget = q
        .false_positive_budget
        .unwrap_or_else(|| decision::default_budget(state.level));
    let Ok(bytes) = axum::body::to_bytes(body, state.max_upload_bytes).await else {
        tracing::warn!(
            id = request_id,
            max = state.max_upload_bytes,
            "upload exceeded size limit"
        );
        return v1_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "artifact_too_large",
            &format!(
                "The artifact exceeds the {} byte limit.",
                state.max_upload_bytes
            ),
        );
    };
    if !bytes.is_empty() {
        if !q.url.is_empty() {
            return v1_error(
                StatusCode::BAD_REQUEST,
                "url_with_body",
                "Use either an exact url or an uploaded artifact, not both.",
            );
        }
        return v1_analyze_bytes(
            &state,
            request_id,
            &q,
            headers,
            bytes,
            request_start,
            request_follow,
        )
        .await;
    }
    if let Some(raw_url) = q.url.first() {
        if q.url.len() > 1 {
            return v1_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "too_many_packages",
                "Only one exact url may be analyzed per request.",
            );
        }
        let url = raw_url.trim();
        if !valid_http_url(url) {
            return v1_error(
                StatusCode::BAD_REQUEST,
                "invalid_url",
                "url must be an absolute http or https URL.",
            );
        }
        let url = url.to_owned();
        if let Ok(init_error) = state.init_error.read()
            && let Some(message) = init_error.as_ref()
        {
            tracing::error!(id = request_id, error = %message, "rejected: startup failed");
            return v1_error(StatusCode::SERVICE_UNAVAILABLE, "starting", message);
        }
        if let Some(response) = check_memory_pressure(&state).await {
            return response;
        }
        let attachment = state.flights.join(FlightKey::url_follow(
            url.clone(),
            request_follow.policy.selection_bits(),
            state.fetch.selection_bits(),
        ));
        let leads = attachment.leads();
        if leads {
            tracing::info!(id = request_id, url = %url, "--> POST /v1/analyze");
            let publisher = state.flights.publisher(attachment.flight());
            match claim_slot(&state, request_id, attachment.flight().key()) {
                Err(outcome) => publisher.publish(outcome),
                Ok((resources, permit)) => {
                    let flight = Arc::clone(attachment.flight());
                    let state = Arc::clone(&state);
                    let url_for_task = url.clone();
                    tokio::spawn(async move {
                        publisher.publish(
                            run_url_analysis(
                                state,
                                request_id,
                                &url_for_task,
                                &flight,
                                resources,
                                permit,
                                request_follow,
                            )
                            .await,
                        );
                    });
                }
            }
        } else {
            tracing::info!(id = request_id, url = %url, "--> POST /v1/analyze (joined a run already in flight)");
        }
        let flight = Arc::clone(attachment.flight());
        let about = Named {
            key: Some(url.clone()),
            asked: Some(url.clone()),
            subject: url,
            is_url: true,
        };
        let follow_name = request_follow.policy.follow_name();
        return match tokio::time::timeout(V1_ANALYZE_GRACE, flight.wait()).await {
            Ok(outcome) => {
                let elapsed = crate::duration_ms(request_start.elapsed());
                v1_outcome_response(
                    &outcome,
                    &about,
                    budget,
                    elapsed,
                    !leads,
                    follow_name.as_deref(),
                )
            }
            Err(_) => {
                tracing::info!(id = request_id, "answering as a stream");
                v1_streamed(
                    Arc::clone(&state),
                    attachment,
                    flight,
                    about,
                    budget,
                    request_start,
                    follow_name,
                )
            }
        };
    }

    let Some(named) = q.purl.first() else {
        return v1_error(
            StatusCode::BAD_REQUEST,
            "missing_package",
            "Name an artifact with ?purl= or ?url=, or send it as the body.",
        );
    };
    let req = AnalyzePurlRequest {
        purl: named.clone(),
    };
    let purl = match normalize_pkg_purl(&req.purl) {
        Ok(purl) => purl,
        Err(message) => {
            return with_subject(
                v1_error(StatusCode::BAD_REQUEST, "invalid_purl", message),
                Subject::purl(&req.purl, None),
            );
        }
    };

    if let Ok(init_error) = state.init_error.read()
        && let Some(message) = init_error.as_ref()
    {
        tracing::error!(id = request_id, error = %message, "rejected: startup failed");
        return v1_error(StatusCode::SERVICE_UNAVAILABLE, "starting", message);
    }
    if let Some(response) = check_memory_pressure(&state).await {
        return response;
    }

    // Already answered?
    //
    // This is the expensive door into the question `/v1/lookup` answers
    // cheaply, and until now it never asked: every caller paid a full
    // download-and-classify for an artifact this worker already held a verdict
    // for. Measured against production, three consecutive analyses of
    // pkg:cargo/tokio@1.40.0 ran 291s, 161s and 116s while `/v1/lookup`
    // answered the same question from the index in a single hop.
    //
    // Resolved exactly the way the lookup resolves it — same normalization,
    // same index-then-corpus order, same budget — because two routes answering
    // one question differently is worse than either answer alone. The index is
    // consulted first and costs nothing; only a miss reaches the corpus.
    //
    // Only a real verdict short-circuits. `unknown` means nobody has analyzed
    // this, which is the whole reason the caller is here, and `unavailable`
    // means we could not find out — turning that into a refusal to work would
    // make a corpus outage look like an answer. Both fall through and run.
    if !q.force && request_follow.persist {
        if let Ok((decided, source)) =
            v1_resolve(&state, None, Some(&req.purl), None, budget, q.fresh).await
            && decided.is_answerable()
        {
            let elapsed = crate::duration_ms(request_start.elapsed());
            tracing::info!(
                id = request_id,
                purl = %purl,
                ms = elapsed,
                kind = if decided.is_verdict() { "verdict" } else { "derived" },
                "--> POST /v1/analyze (answered from what we already knew; no slot spent)"
            );
            let mut resp = Json(decided).into_response();
            resp.headers_mut().insert("X-Total-Ms", elapsed.into());
            resp.headers_mut().insert(
                "X-Scan-Source",
                axum::http::HeaderValue::from_static(source),
            );
            resp.extensions_mut().insert(Subject::purl(&purl, None));
            return resp;
        }
        tracing::info!(
            id = request_id,
            purl = %purl,
            "no verdict held; analyzing"
        );
    }

    let attachment = state.flights.join(FlightKey::purl_follow(
        purl.clone(),
        request_follow.policy.selection_bits(),
        state.fetch.selection_bits(),
    ));
    let leads = attachment.leads();
    if leads {
        tracing::info!(id = request_id, purl = %purl, "--> POST /v1/analyze");
        let publisher = state.flights.publisher(attachment.flight());
        match claim_slot(&state, request_id, attachment.flight().key()) {
            Err(outcome) => publisher.publish(outcome),
            Ok((resources, permit)) => {
                let flight = Arc::clone(attachment.flight());
                let state = Arc::clone(&state);
                let purl = purl.clone();
                tokio::spawn(async move {
                    publisher.publish(
                        run_purl_analysis(
                            state,
                            request_id,
                            &purl,
                            &flight,
                            resources,
                            permit,
                            request_follow,
                        )
                        .await,
                    );
                });
            }
        }
    } else {
        tracing::info!(id = request_id, purl = %purl, "--> POST /v1/analyze (joined a run already in flight)");
    }

    let flight = Arc::clone(attachment.flight());
    // Named, not analyzed from bytes, so there is always a coordinate: the
    // normalized one keys everything, and `req.purl` is what the caller typed.
    let about = Named {
        key: Some(purl.clone()),
        asked: Some(req.purl.clone()),
        subject: purl.clone(),
        is_url: false,
    };
    // Inside the grace window the ordinary response still applies, which is what
    // keeps `429 At capacity` a real 429 the router can act on rather than a
    // decision buried in a 200 body. Capacity is refused the moment a slot is
    // asked for, so it never reaches the streaming path.
    let follow_name = request_follow.policy.follow_name();
    match tokio::time::timeout(V1_ANALYZE_GRACE, flight.wait()).await {
        Ok(outcome) => {
            let elapsed = crate::duration_ms(request_start.elapsed());
            v1_outcome_response(
                &outcome,
                &about,
                budget,
                elapsed,
                !leads,
                follow_name.as_deref(),
            )
        }
        Err(_) => {
            tracing::info!(id = request_id, purl = %purl, "answering as a stream");
            v1_streamed(
                Arc::clone(&state),
                attachment,
                flight,
                about,
                budget,
                request_start,
                follow_name,
            )
        }
    }
}

/// The package a request is about, in the three spellings that are not
/// interchangeable.
///
/// Collapsing them into one string is what produced the bug this exists to
/// prevent: `key` is what the index and the corpus are keyed by, `asked` is
/// what the caller typed and what the answer is spelled with, and `subject`
/// labels logs and progress — the digest, when an upload named no coordinate
/// at all.
struct Named {
    key: Option<String>,
    asked: Option<String>,
    subject: String,
    is_url: bool,
}

/// A finished analysis, as a decision.
fn v1_outcome_response(
    outcome: &Outcome,
    named: &Named,
    budget: u16,
    elapsed_ms: u64,
    shared: bool,
    follow: Option<&str>,
) -> Response {
    let (purl, asked) = (named.key.as_deref(), named.asked.as_deref());
    let subject = named.subject.as_str();
    let mut resp = match outcome {
        Outcome::Report(result) => {
            // An upload has no locator, and the digest is not one: passing it
            // here would put a sha256 in the `purl` field, where /v1/lookup
            // reports null for the same artifact. The two routes answer with
            // one shape or neither is trustworthy.
            // The verdict is stored under the normalized key; only the answer
            // going back out is spelled the caller's way.
            let verdict_key = (!named.is_url).then_some(purl).flatten();
            let verdict = crate::lookup::Verdict::from_scan(result, verdict_key);
            let decided = if named.is_url {
                V1Decision::stored(&verdict, None, budget).asked_about_url(asked)
            } else {
                V1Decision::stored(&verdict, purl, budget).asked_about(asked)
            };
            let mut resp = Json(decided).into_response();
            resp.headers_mut().insert("X-Total-Ms", elapsed_ms.into());
            // Whether this answer cost an analysis. The route is the same
            // either way, but a run served from the analysis cache did no work
            // and one that reached the pipeline did — and a caller measuring
            // what its fleet spends cannot tell those apart from the route.
            resp.headers_mut().insert(
                "X-Scan-Source",
                axum::http::HeaderValue::from_static(if result.analysis_cached {
                    "scan:cached"
                } else {
                    "scan:analysis"
                }),
            );
            resp
        }
        // A refusal keeps its status: the caller's router uses it to send the
        // work somewhere that can take it, which a decision in a 200 body
        // cannot be made to do.
        Outcome::Rendered { status, body } => (*status, Json(body)).into_response(),
    };
    // Which question this answer answers. The caller resolved a policy before
    // asking, but only this server knows what it applied on top of its own
    // configuration, and the answer has to be filed under what was measured
    // rather than what was requested.
    //
    // On refusals too. A refusal files nothing, but a caller correlating a
    // retry — or an operator asking why a fleet records nothing — should not
    // have to infer which policy was in play from which reply it happened to
    // get.
    if let Some(name) = follow
        && let Ok(value) = axum::http::HeaderValue::from_str(name)
    {
        resp.headers_mut().insert("X-Scan-Follow", value);
    }
    if shared {
        resp.extensions_mut().insert(super::access::Shared);
    }
    resp.extensions_mut().insert(if named.is_url {
        Subject::url(subject, None)
    } else {
        Subject::purl(subject, None)
    });
    resp
}

/// The same answer, delivered as a stream that reports progress until the
/// analysis lands.
///
/// Newline-delimited JSON: zero or more progress frames, then the decision.
/// A caller reads lines until one carries `decision`, and that is the answer;
/// an analysis that finishes before the first frame is due emits nothing but
/// the decision, so a fast call still looks like a single JSON object and still
/// parses as one.
///
/// Progress is real rather than a keepalive. The phase a run is in is already
/// tracked for the watchdog, so saying it costs nothing and turns a silent
/// connection into one a caller can watch — which is also what stops anything
/// between here and them from concluding the connection is idle and cutting it.
///
/// Committing to `200` here is the trade: the status goes out before the
/// outcome is known, so a failure past the grace window arrives as a decision
/// of `unavailable` rather than a 5xx. That is the v1 contract either way — a
/// caller reads `decision`, not the status line — and the alternative on this
/// path is not a truthful 504 but a severed connection and no answer at all.
fn v1_streamed(
    state: Arc<AppState>,
    attachment: super::flight::Attachment,
    flight: Arc<Flight>,
    named: Named,
    budget: u16,
    request_start: Instant,
    follow: Option<String>,
) -> Response {
    let Named {
        key: purl,
        asked,
        subject,
        is_url,
    } = named;
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
    // Progress frames name the package too, and a caller reading the stream
    // correlates on the same field the decision carries. An upload has no
    // coordinate at all, so identify it as sha256 rather than putting the
    // digest in the purl field.
    let labelled_locator = asked.clone().unwrap_or_else(|| subject.clone());
    let labelled = subject.clone();
    tokio::spawn(async move {
        // Held for the life of the stream: an attachment dropped early would
        // tell the flight nobody is waiting on this analysis while somebody is.
        let _attachment = attachment;
        let mut waiting = std::pin::pin!(flight.wait());
        let mut next = V1_PROGRESS_FIRST;
        let outcome = loop {
            tokio::select! {
                outcome = &mut waiting => break outcome,
                () = tokio::time::sleep(next) => {
                    next = V1_PROGRESS_EVERY;
                    let mut frame = serde_json::json!({
                        "state": "analyzing",
                        "elapsed_ms": crate::duration_ms(request_start.elapsed()),
                        "phase": v1_phase_of(&state, &subject),
                    });
                    let field = if is_url {
                        "url"
                    } else if asked.is_some() {
                        "purl"
                    } else {
                        "sha256"
                    };
                    frame[field] = serde_json::Value::String(labelled_locator.clone());
                    // A caller that has gone away shows up here as a closed
                    // channel, which ends the stream. The analysis keeps going:
                    // it is not this connection's to lose.
                    if !v1_send(&tx, &frame).await {
                        return;
                    }
                }
            }
        };
        let elapsed = crate::duration_ms(request_start.elapsed());
        let decided = match outcome.as_ref() {
            Outcome::Report(result) => {
                // As on the unstreamed path: an upload has no locator, and the
                // digest is not one.
                let verdict_key = (!is_url).then_some(purl.as_deref()).flatten();
                let verdict = crate::lookup::Verdict::from_scan(result, verdict_key);
                if is_url {
                    V1Decision::stored(&verdict, None, budget).asked_about_url(asked.as_deref())
                } else {
                    V1Decision::stored(&verdict, purl.as_deref(), budget)
                        .asked_about(asked.as_deref())
                }
            }
            Outcome::Rendered { status, .. } => {
                tracing::warn!(subject = %subject, status = status.as_u16(), elapsed_ms = elapsed, "streamed analysis failed");
                if is_url {
                    V1Decision::unavailable(None, None).asked_about_url(asked.as_deref())
                } else {
                    V1Decision::unavailable(None, purl.as_deref()).asked_about(asked.as_deref())
                }
            }
        };
        v1_send(&tx, &decided).await;
    });

    let mut resp = Response::new(axum::body::Body::from_stream(ChannelStream(rx)));
    let headers = resp.headers_mut();
    // NDJSON, because the body is a sequence rather than one document. A caller
    // that only wants the answer reads the last line.
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/x-ndjson"),
    );
    // Nothing may buffer this: a proxy that holds the bytes back to measure the
    // body defeats the only thing progress frames are for.
    headers.insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    headers.insert(
        "X-Accel-Buffering",
        axum::http::HeaderValue::from_static("no"),
    );
    headers.insert(
        "X-Scan-Source",
        axum::http::HeaderValue::from_static("scan:analysis"),
    );
    // See the matching header in `v1_outcome_response`. Sent on the stream's
    // own headers, which go out before the first progress frame, so a caller
    // knows how to file the decision before the decision arrives.
    if let Some(name) = follow
        && let Ok(value) = axum::http::HeaderValue::from_str(&name)
    {
        headers.insert("X-Scan-Follow", value);
    }
    resp.extensions_mut().insert(if is_url {
        Subject::url(&labelled, None)
    } else {
        Subject::purl(&labelled, None)
    });
    resp
}

/// Write one NDJSON line. Reports whether the caller is still there.
async fn v1_send<T: serde::Serialize>(
    tx: &tokio::sync::mpsc::Sender<Result<bytes::Bytes, std::io::Error>>,
    frame: &T,
) -> bool {
    let Ok(mut line) = serde_json::to_vec(frame) else {
        return true;
    };
    line.push(b'\n');
    tx.send(Ok(bytes::Bytes::from(line))).await.is_ok()
}

/// Which phase this artifact's run is in, if it is one of ours.
///
/// Read from the same registry the watchdog reports from. A follower riding
/// somebody else's run does not know their request id, so the lookup is by the
/// artifact rather than the request — there is at most one run per key, which
/// is what single-flight guarantees.
fn v1_phase_of(state: &Arc<AppState>, purl: &str) -> Option<String> {
    state
        .in_flight
        .iter()
        .find(|entry| entry.name == purl)
        .map(|entry| entry.phase.get())
}

/// An mpsc receiver as a body stream. Hand-written so the crate takes the
/// `Stream` trait alone rather than all of futures-util for one adapter.
struct ChannelStream(tokio::sync::mpsc::Receiver<Result<bytes::Bytes, std::io::Error>>);

impl futures_core::Stream for ChannelStream {
    type Item = Result<bytes::Bytes, std::io::Error>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.0.poll_recv(cx)
    }
}

/// GET /v1/lookup — what we know, at the caller's threshold. Never analyzes.
///
/// Answers a single object when one package is named and an array when `purl`
/// repeats, so the shape follows the shape of the question rather than the data:
/// a caller that always asks about one always gets one, and a caller that always
/// asks about many always gets many. Neither ever has to branch on what came
/// back.
pub(super) async fn v1_lookup(
    State(state): State<Arc<AppState>>,
    raw: axum::extract::RawQuery,
    headers: axum::http::HeaderMap,
) -> Response {
    let started = Instant::now();
    let q = V1LookupQuery::parse(raw.0.as_deref()).with_fresh_header(&headers);
    let response = v1_lookup_inner(&state, &q).await;
    state
        .lookups
        .record(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
    response
}

async fn v1_lookup_inner(state: &Arc<AppState>, q: &V1LookupQuery) -> Response {
    if let Some(bad) = q.bad_budget.as_deref() {
        return v1_error(
            StatusCode::BAD_REQUEST,
            "invalid_false_positive_budget",
            &format!("false_positive_budget must be a whole number from 0 to 65535, not {bad:?}."),
        );
    }
    let budget = q
        .false_positive_budget
        .unwrap_or_else(|| decision::default_budget(state.level));
    let sha = q.sha256.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let purls: Vec<&str> = q
        .purl
        .iter()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();
    let urls: Vec<&str> = q
        .url
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect();

    if !purls.is_empty() && !urls.is_empty() {
        return v1_error(
            StatusCode::BAD_REQUEST,
            "multiple_locators",
            "Use ?purl= or ?url=, not both.",
        );
    }
    if urls.iter().any(|url| !valid_http_url(url)) {
        return v1_error(
            StatusCode::BAD_REQUEST,
            "invalid_url",
            "url must be an absolute http or https URL.",
        );
    }
    if sha.is_none() && purls.is_empty() && urls.is_empty() {
        return v1_error(
            StatusCode::BAD_REQUEST,
            "missing_package",
            "Name an artifact with ?purl=, ?url=, or ?sha256=.",
        );
    }
    if purls.len().max(urls.len()) > V1_MAX_KEYS {
        return v1_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "too_many_packages",
            &format!(
                "{} packages exceeds the limit of {V1_MAX_KEYS} for a URL. Use POST /v1/lookup.",
                purls.len().max(urls.len())
            ),
        );
    }

    if let Some(url) = urls.first() {
        if urls.len() > 1 {
            return v1_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "too_many_packages",
                "Only one exact url may be analyzed per request.",
            );
        }
        return match v1_resolve(state, sha, None, Some(url), budget, q.fresh).await {
            Ok((decided, source)) => {
                let mut resp = Json(decided).into_response();
                resp.headers_mut().insert(
                    "X-Scan-Source",
                    axum::http::HeaderValue::from_static(source),
                );
                resp
            }
            Err(response) => *response,
        };
    }

    // One package named two ways is one question, so a lone sha256 and a lone
    // purl resolve together rather than as two entries.
    if purls.len() <= 1 {
        let purl = purls.first().copied();
        return match v1_resolve(state, sha, purl, None, budget, q.fresh).await {
            Ok((decided, source)) => {
                let mut resp = Json(decided).into_response();
                resp.headers_mut().insert(
                    "X-Scan-Source",
                    axum::http::HeaderValue::from_static(source),
                );
                resp
            }
            Err(response) => *response,
        };
    }

    let mut out = Vec::with_capacity(purls.len());
    let mut source = None;
    for purl in purls {
        match v1_resolve(state, None, Some(purl), None, budget, q.fresh).await {
            Ok((decided, row_source)) => {
                source = Some(match source {
                    None => row_source,
                    Some(previous) if previous == row_source => previous,
                    Some(_) => "scan:analysis",
                });
                out.push(decided);
            }
            Err(response) => return *response,
        }
    }
    let mut resp = Json(out).into_response();
    resp.headers_mut().insert(
        "X-Scan-Source",
        axum::http::HeaderValue::from_static(source.unwrap_or("scan:analysis")),
    );
    resp
}

/// One package, resolved and decided, answered about the coordinate the caller
/// named. `Err` is a request-level rejection: a key we cannot parse is the
/// caller's mistake and stops the whole call, because answering the rest would
/// hide it.
async fn v1_resolve(
    state: &Arc<AppState>,
    sha: Option<&str>,
    raw_purl: Option<&str>,
    raw_url: Option<&str>,
    budget: u16,
    fresh: bool,
) -> Result<(V1Decision, &'static str), Box<Response>> {
    let (decided, source) = v1_decide(state, sha, raw_purl, raw_url, budget, fresh).await?;
    Ok((
        decided.asked_about(raw_purl).asked_about_url(raw_url),
        source,
    ))
}

/// The decision itself, in this worker's own vocabulary: every key here is the
/// normalized one, because that is what the index and the corpus are keyed by.
/// The filters' opinion of an artifact named by a digest, a PURL, or both.
///
/// Both keys are evidence about one artifact, so both are supplied and `burton`
/// combines them: the worst claim against either wins, and a blessing needs all
/// of them. A caller who names both is asserting they are the same thing, so a
/// bless on one beside a claim on the other is a contradiction, not a coin flip.
fn bloom_decision(sha: Option<&str>, purl: Option<&str>) -> crate::bloom_repo::Decision {
    use crate::bloom_repo::Decision;
    let Some(lk) = crate::bloom_repo::global() else {
        return Decision::Unknown;
    };
    lk.decide_any(purl, sha.and_then(burton::parse_sha256_hex).as_ref())
}

async fn v1_decide(
    state: &Arc<AppState>,
    sha: Option<&str>,
    raw_purl: Option<&str>,
    raw_url: Option<&str>,
    budget: u16,
    fresh: bool,
) -> Result<(V1Decision, &'static str), Box<Response>> {
    let purl = match raw_purl.map(normalize_pkg_purl) {
        Some(Ok(purl)) => Some(purl),
        Some(Err(message)) => {
            return Err(Box::new(v1_error(
                StatusCode::BAD_REQUEST,
                "invalid_purl",
                message,
            )));
        }
        None => None,
    };
    let url = match raw_url {
        Some(url) if !valid_http_url(url) => {
            return Err(Box::new(v1_error(
                StatusCode::BAD_REQUEST,
                "invalid_url",
                "url must be an absolute http or https URL.",
            )));
        }
        Some(url) => Some(url.trim().to_owned()),
        None => None,
    };
    let sha = match sha {
        Some(sha) if burton::parse_sha256_hex(sha).is_none() => {
            return Err(Box::new(v1_error(
                StatusCode::BAD_REQUEST,
                "invalid_sha256",
                "sha256 must be 64 hexadecimal characters.",
            )));
        }
        Some(sha) => Some(sha.to_ascii_lowercase()),
        None => None,
    };

    // A missing index is not an empty one. Reporting `unknown` here would tell
    // the caller nobody has analyzed this package, when what is true is that we
    // cannot say — and those two carry different policies at the other end.
    // This is the whole reason `unavailable` is a separate value.
    let Some(index) = crate::lookup::global() else {
        return Ok((
            V1Decision::unavailable(sha.as_deref(), purl.as_deref()),
            "none",
        ));
    };
    let index = Some(index);
    let by_sha = sha
        .as_deref()
        .and_then(|s| index.as_ref().and_then(|i| i.get_sha(s)));
    // The digest is the identity: a PURL's verdict is accepted only when it
    // describes the same bytes, because a release whose digest has moved is an
    // answer about a different artifact than the one asked about.
    let verdict = match (&by_sha, &sha) {
        (Some(_), _) => by_sha,
        (None, Some(sha)) => pick_verdict(
            None,
            || {
                purl.as_deref()
                    .and_then(|p| index.as_ref().and_then(|i| i.get_purl(p)))
            },
            sha,
        ),
        (None, None) => purl
            .as_deref()
            .and_then(|p| index.as_ref().and_then(|i| i.get_purl(p))),
    };

    if let Some(verdict) = verdict.as_ref() {
        return Ok((
            V1Decision::stored(verdict, purl.as_deref(), budget).with_url(url.as_deref()),
            // Held, not produced. This used to report `scan:analysis` — the same
            // value a fresh run reports — which made an instant index hit and a
            // ninety-second analysis indistinguishable to anything counting
            // cache layers, and every miss looked like a hit.
            "scan:index",
        ));
    }

    // Nothing measured here. Ask the filters before the network: they are the
    // cheapest knowledge in the process, and for a blessed artifact they are
    // the whole answer.
    //
    // Unless the caller asked for the authoritative answer. A filter is
    // membership rebuilt on a schedule, so it is exactly what somebody reading
    // after a write — or checking whether a revocation has landed — needs
    // bypassed. Withholding the decision here disables both bloom paths at
    // once: the fast bless below, and the derived fallback after the corpus.
    let bloom = if fresh {
        crate::bloom_repo::Decision::Unknown
    } else {
        bloom_decision(sha.as_deref(), purl.as_deref())
    };

    // A bless answers immediately and does not pay the hopper round trip. The
    // exposure is a bless that has gone stale — but that is bounded by the
    // filter rebuild, because `good` is rebuilt as `good − (bad ∪ sighted)` and
    // the bad channel is the designed revocation path. It is also the bargain
    // the local scan path already takes: `bloom_skip_predicate` skips the
    // download outright on a good hit, without asking anyone.
    if bloom == crate::bloom_repo::Decision::Skip
        && let Some(d) = V1Decision::bloom(bloom, sha.as_deref(), purl.as_deref(), budget)
    {
        return Ok((d.with_url(url.as_deref()), "scan:bloom"));
    }

    // Not in this worker's index. The corpus behind it may still know, and a
    // caller should not have to learn that two services exist in order to get
    // one answer — so ask, rather than reporting an absence that is only ours.
    //
    // Measured beats derived: a filter claim is a floor, and hopper may hold
    // the real level, the real findings and the sentence a person reads. Only
    // when it holds nothing does the filter's own claim stand in.
    let Some(corpus) = state.corpus.as_ref() else {
        return Ok((
            V1Decision::bloom(bloom, sha.as_deref(), purl.as_deref(), budget)
                .unwrap_or_else(|| V1Decision::unanalyzed(sha.as_deref(), purl.as_deref()))
                .with_url(url.as_deref()),
            "scan:bloom",
        ));
    };
    let (reached, source) = corpus
        .known_with_source(sha.as_deref(), purl.as_deref())
        .await;
    let source = source.map_or("none", |source| match source {
        corpus::CorpusSource::Replica => "scan:replica",
        corpus::CorpusSource::Primary => "scan:primary",
    });
    Ok((
        match reached {
            Reached::Record(record) => {
                V1Decision::corpus(&record, sha.as_deref(), purl.as_deref(), budget)
                    .with_url(url.as_deref())
            }
            // The corpus holds nothing either. A filter claim is the last thing
            // we know, and answering `unanalyzed` about a digest several operators
            // call malware is a worse answer than saying who says so.
            Reached::Nothing => V1Decision::bloom(bloom, sha.as_deref(), purl.as_deref(), budget)
                .unwrap_or_else(|| V1Decision::unanalyzed(sha.as_deref(), purl.as_deref()))
                .with_url(url.as_deref()),
            // The corpus could not answer, so neither can we. Emphatically not
            // `unanalyzed`: that would tell the caller nobody has analyzed this
            // package, which is a claim about the package rather than about us, and
            // the one that lets a gate fail open during an outage.
            Reached::Unreachable => {
                V1Decision::unavailable(sha.as_deref(), purl.as_deref()).with_url(url.as_deref())
            }
        },
        source,
    ))
}

/// One decision, as it goes on the wire.
///
/// Every field is always present. A key that is unknown is `null` and a list
/// that is empty is `[]`, never absent — a caller writes one code path against
/// a shape that does not move, and a generated type has no optionals to unwrap
/// that are really just "we had nothing to say".
#[derive(serde::Serialize)]
pub(super) struct V1Decision {
    decision: decision::Decision,
    purl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    sha256: Option<String>,
    severity: Option<decision::Severity>,
    /// The tightest false-positive budget per 100 million benign files at which
    /// this artifact grades hostile — lower being worse, and `-1` meaning it
    /// fires at none. A property of the file and the model: measured, never
    /// chosen, which is what separates it from the caller's own
    /// `false_positive_budget` that it is compared against. Present so a caller
    /// can tune that budget against real numbers.
    fires_at: Option<i32>,
    reason: Option<String>,
    findings: Vec<V1Finding>,
    engine_version: Option<String>,
    analyzed_at: Option<String>,
}

impl V1Decision {
    /// Nobody has analyzed this artifact. Nothing is wrong; there is simply no
    /// answer, and what a caller does about that is their policy to set.
    fn unanalyzed(sha: Option<&str>, purl: Option<&str>) -> Self {
        Self::empty(decision::Decision::Unanalyzed, sha, purl)
    }

    /// We could not answer. Deliberately carries no severity, no level and no
    /// findings: this decision is about us, not about the artifact, and a
    /// caller must not be able to read anything into it.
    fn unavailable(sha: Option<&str>, purl: Option<&str>) -> Self {
        Self::empty(decision::Decision::Unavailable, sha, purl)
    }

    /// What the filters alone justify, for an artifact no stored verdict and no
    /// corpus record covers. `None` when the filters had no opinion.
    ///
    /// `engine_version` and `analyzed_at` stay `None`, which is the whole
    /// contract: an engine is what separates a measurement from a citation, so
    /// a caller (and beamline's cache) can tell this from a scan we ran, and
    /// `/v1/analyze` is free to replace it with a real one.
    ///
    /// Levels come from [`crate::lookup::bloom_claim`] — the loosest each tier
    /// can justify, since a filter carries membership and not a measurement.
    fn bloom(
        d: crate::bloom_repo::Decision,
        sha: Option<&str>,
        purl: Option<&str>,
        budget: u16,
    ) -> Option<Self> {
        let claim = crate::lookup::bloom_claim(d)?;
        let fires_at = claim
            .as_ref()
            .map_or(crate::lookup::BENIGN_LEVEL, |c| c.lvl);
        let (decided, severity) = decision::decide(Some(fires_at), budget);
        Some(Self {
            decision: decided,
            purl: purl.map(str::to_owned),
            url: None,
            sha256: sha.map(str::to_owned),
            severity: Some(severity),
            fires_at: Some(fires_at),
            reason: claim.as_ref().map(|c| c.desc.to_owned()),
            findings: claim
                .as_ref()
                .map(|c| V1Finding::from_bloom(c, purl))
                .into_iter()
                .collect(),
            engine_version: None,
            analyzed_at: None,
        })
    }

    fn empty(decided: decision::Decision, sha: Option<&str>, purl: Option<&str>) -> Self {
        Self {
            decision: decided,
            purl: purl.map(str::to_owned),
            url: None,
            sha256: sha.map(str::to_owned),
            severity: None,
            fires_at: None,
            reason: None,
            findings: Vec::new(),
            engine_version: None,
            analyzed_at: None,
        }
    }

    /// Whether this says something about the artifact rather than about us,
    /// and says it because an engine of ours measured it.
    ///
    /// `unanalyzed` reports that nobody has analyzed it, which is precisely what
    /// `/v1/analyze` exists to fix, and `unavailable` reports that we could not
    /// find out. Neither may stand in for a run.
    ///
    /// Nor may a level derived from threat-feed citations, and that one is the
    /// easy miss: it carries a real `decision`, so it reads as a verdict at
    /// every glance. Standing in for the run would mean an artifact nobody has
    /// analyzed never gets analyzed — the caller is told `block`, the corpus
    /// learns nothing, and the gap the derived level papers over stays open for
    /// good. An engine is exactly what separates a measurement from a citation,
    /// so an engine is what is asked for.
    fn is_verdict(&self) -> bool {
        !matches!(
            self.decision,
            decision::Decision::Unanalyzed | decision::Decision::Unavailable
        ) && self.engine_version.is_some()
    }

    /// Anything that answers the caller's question, measured or not.
    ///
    /// Wider than [`Self::is_verdict`] on purpose, and the difference is a
    /// policy choice rather than an oversight. `is_verdict` remains the strict
    /// question — is this a measurement of ours — and downstream still asks it
    /// by looking for an engine. This one governs whether `/v1/analyze` may
    /// answer at all, where the operator's judgement is that a fast answer from
    /// what we already know beats spending a slot to rediscover it.
    ///
    /// The cost is real and worth naming: for an artifact nobody has analyzed
    /// and a feed has cited, this answers from the citation and the analysis
    /// never happens, so the corpus does not learn. `?fresh=1` is the escape
    /// hatch for a caller who needs the measurement, and the derived answer
    /// still carries no `engine_version`, so nothing downstream mistakes it for
    /// one.
    ///
    /// `unanalyzed` and `unavailable` are excluded exactly as before: the first is
    /// what `/v1/analyze` exists to fix, the second is a statement about us.
    fn is_answerable(&self) -> bool {
        !matches!(
            self.decision,
            decision::Decision::Unanalyzed | decision::Decision::Unavailable
        ) && self.fires_at.is_some()
    }

    /// A verdict this worker holds in its own index.
    ///
    /// Takes no digest: a stored verdict always carries its own, and it is the
    /// artifact's identity rather than whatever the caller happened to type.
    fn stored(v: &crate::lookup::Verdict, purl: Option<&str>, budget: u16) -> Self {
        let (decided, severity) = decision::decide(v.lvl, budget);
        Self {
            decision: decided,
            // The verdict names the artifact it is about; the caller's spelling
            // only fills in what it could not.
            purl: v.purl.clone().or_else(|| purl.map(str::to_owned)),
            url: None,
            sha256: Some(v.sha256.clone()),
            severity: Some(severity),
            fires_at: v.lvl,
            reason: v.why.clone(),
            findings: V1Finding::worth_reporting(v.hits.iter().map(V1Finding::from_hit)),
            engine_version: Some(v.eng.clone()),
            analyzed_at: Some(v.at.clone()),
        }
    }

    /// A record the corpus holds. Decided here rather than there: hopper stores
    /// what an artifact is, and turning that into allow or block is policy this
    /// worker owns, so the same budget produces the same answer whichever side
    /// of the index the record came from.
    fn corpus(
        r: &corpus::CorpusRecord,
        sha: Option<&str>,
        purl: Option<&str>,
        budget: u16,
    ) -> Self {
        let (decided, severity) = decision::decide(r.fires_at, budget);
        Self {
            decision: decided,
            purl: r.purl.clone().or_else(|| purl.map(str::to_owned)),
            url: None,
            // Empty is absent. A record standing on threat-feed citations for a
            // package nobody has analyzed names no bytes, and the corpus sends
            // the field as "" rather than omitting it — which would put an
            // empty string where the wire contract says string|null, and where
            // a caller comparing digests to prove two spellings are one thing
            // would find them equal.
            sha256: r
                .sha256
                .clone()
                .filter(|s| !s.is_empty())
                .or_else(|| sha.map(str::to_owned)),
            severity: Some(severity),
            fires_at: r.fires_at,
            reason: r.reason.clone(),
            findings: V1Finding::worth_reporting(r.findings.iter().map(V1Finding::from_corpus)),
            engine_version: r.engine_version.clone(),
            analyzed_at: r.analyzed_at.clone(),
        }
    }

    /// Answer about the coordinate the caller named, in the spelling they
    /// named it.
    ///
    /// Every key is normalized before it is looked up — PyPI folds `.` and `_`
    /// to `-` per PEP 503, npm scopes are unwrapped, a bare `npm/left-pad` gets
    /// its `pkg:` — and echoing the normalized form back is how
    /// `pkg:pypi/info.gianlucacosta.eos.core@2.0.2` came home answered about
    /// `pkg:pypi/info-gianlucacosta-eos-core@2.0.2`. The same package, and a
    /// caller has no way to know that without implementing PEP 503 themselves.
    ///
    /// That matters most where it is least visible: a lookup may name fifty
    /// packages and the reply is a list, so `purl` is what a caller matches
    /// response to request by. Rewriting the spelling breaks that silently, and
    /// only for the names that happen to contain a `.` or a `_`.
    ///
    /// So the field answers "the package you asked about" and the caller's
    /// bytes are returned unaltered. `sha256` remains the identity, and it is
    /// the field to compare when two spellings must be proven to be one thing.
    fn asked_about(mut self, asked: Option<&str>) -> Self {
        if let Some(asked) = asked {
            self.purl = Some(asked.to_owned());
        }
        self
    }

    fn asked_about_url(mut self, asked: Option<&str>) -> Self {
        if let Some(asked) = asked {
            self.url = Some(asked.to_owned());
        }
        self
    }

    fn with_url(mut self, url: Option<&str>) -> Self {
        if let Some(url) = url {
            self.url = Some(url.to_owned());
        }
        self
    }
}

/// One finding on the wire.
///
/// Fed from this worker's index or from the corpus, which know different
/// amounts about the same thing: a stored hit carries the file and offset it
/// fired on, while the corpus keeps only the trait and its criticality — those
/// details live in the one column a lookup must not read.
///
/// `id` and `crit` are therefore the only fields always present, and the rest
/// are omitted when there is nothing to say rather than sent as null. That is
/// the opposite of the rule the enclosing decision object follows, and the two
/// differ because the questions do. A decision has a FIXED set of things it
/// answers, so a caller writes one code path against nine keys that never move
/// and `"engine_version": null` is itself the answer to "which engine". A
/// finding has no such set: how much is known about one varies by where it came
/// from, four nulls per corpus finding is most of the object, and a reader
/// checking `desc` has to handle absence anyway.
#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct V1Finding {
    id: String,
    crit: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pkg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    desc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    off: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    line: Option<u64>,
}

impl V1Finding {
    fn from_hit(h: &crate::lookup::Hit) -> Self {
        let some = |s: &String| (!s.is_empty()).then(|| s.clone());
        Self {
            id: h.id.clone(),
            crit: h.crit,
            file: some(&h.file),
            pkg: some(&h.pkg),
            desc: some(&h.desc),
            off: h.off,
            line: h.line,
        }
    }

    /// A finding synthesized from a filter hit. Carries no `file`, `off` or
    /// `line` — a filter knows membership and nothing about where anything
    /// fired — which is the same shape the corpus sends for a citation.
    fn from_bloom(c: &crate::lookup::BloomClaim, purl: Option<&str>) -> Self {
        Self {
            id: c.id.to_owned(),
            crit: c.crit,
            file: None,
            pkg: purl.map(str::to_owned),
            desc: Some(c.desc.to_owned()),
            off: None,
            line: None,
        }
    }

    fn from_corpus(f: &corpus::CorpusFinding) -> Self {
        Self {
            id: f.id.clone(),
            crit: f.crit,
            file: None,
            pkg: None,
            desc: f.desc.clone(),
            off: None,
            line: None,
        }
    }

    /// The findings worth putting on the wire: the strongest few, worst first.
    ///
    /// The corpus already decides this in a trigger — `crit >= 4`, ordered by
    /// criticality, at most three — and a decision answered from this worker's
    /// own index has to land on the same set, or one artifact reads differently
    /// depending on which side of the index happened to answer. Applying it
    /// here rather than trusting each source keeps the two in step by
    /// construction; on corpus records it is a no-op.
    ///
    /// A benign artifact clears the bar with nothing, and that is the intended
    /// answer rather than a gap to fill: ten "Rust test marker" hits explain
    /// nothing about an allow, and listing them invites a caller to read
    /// significance into noise.
    fn worth_reporting(all: impl Iterator<Item = Self>) -> Vec<Self> {
        let mut kept: Vec<Self> = all.filter(|f| f.crit >= REPORT_MIN_CRIT).collect();
        // Stable, so equal criticalities keep the order their source listed
        // them in — the same tiebreak as the trigger's `ORDER BY crit DESC,
        // ord`.
        kept.sort_by_key(|f| std::cmp::Reverse(f.crit));
        kept.truncate(REPORT_LIMIT);
        kept
    }
}

/// Suspicious and above. Below this a trait is an observation, not a reason.
const REPORT_MIN_CRIT: u8 = 4;

/// Enough to show why, few enough to read. Matches the corpus's `LIMIT 3`.
const REPORT_LIMIT: usize = 3;

/// A v1 error. `code` is stable and machine-readable; `message` is for humans
/// and may be reworded freely.
fn v1_error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": { "code": code, "message": message }
        })),
    )
        .into_response()
}

pub(super) async fn requests(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let now = Instant::now();
    let mut entries: Vec<serde_json::Value> = state
        .in_flight
        .iter()
        .map(|e| {
            let elapsed_ms = now.duration_since(e.started_at).as_millis();
            let phase = e.phase.get();
            let tid = e.thread_id.load(std::sync::atomic::Ordering::Relaxed);
            serde_json::json!({
                "request_id": e.key(),
                "name": e.name,
                "size_bytes": e.size_bytes,
                "elapsed_ms": elapsed_ms,
                "long_running": elapsed_ms >= 120_000,
                "phase": phase,
                "thread_id": if tid > 0 { Some(tid) } else { None },
            })
        })
        .collect();

    entries.sort_by(|a, b| b["elapsed_ms"].as_u64().cmp(&a["elapsed_ms"].as_u64()));

    // `analyses` counts distinct runs and `attached` counts the requests riding
    // them; the gap is duplicate work single-flight is absorbing.
    let census = state.flights.census();
    Json(serde_json::json!({
        "count": entries.len(),
        "analyses": census.analyses,
        "attached": census.attached,
        "requests": entries,
    }))
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

    let pid = unsafe { libc::getpid() };
    let mib: [libc::c_int; 4] = [
        libc::CTL_KERN,
        libc::KERN_PROC,
        libc::KERN_PROC_PID | libc::KERN_PROC_INC_THREAD,
        pid,
    ];

    let mut len: libc::size_t = 0;
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
    let mut procs: Vec<libc::kinfo_proc> = (0..count).map(|_| unsafe { mem::zeroed() }).collect();
    let mut actual_len = len;

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
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{Outcome, classify_analysis_error, flight_response};

    /// A registry-metadata fallback must never mint hopper a placeholder row:
    /// with nothing known about the purl (a 404 from `/v1/lookup`), the result
    /// is `Suppress` — nothing gets posted for it. See
    /// [`super::offer_registry_fallback`]'s doc comment for why (the fallback's
    /// own content hashes differently on every fetch).
    #[tokio::test]
    async fn unknown_purl_suppresses_the_registry_fallback() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock corpus");
        let addr = listener.local_addr().expect("addr");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept lookup");
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            stream
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .expect("write 404");
        });

        let corpus = super::super::corpus::Corpus::new(Some(&format!("http://{addr}")))
            .expect("corpus configured");
        let route = tokio::task::spawn_blocking(move || {
            super::offer_registry_fallback(
                Some(&corpus),
                None,
                "pkg:npm/never-seen@0.0.0",
                "never-seen@0.0.0.registry.json",
                None,
            )
        })
        .await
        .expect("task");
        server.join().expect("server thread");

        assert!(
            matches!(route, crate::engine::HopperRoute::Suppress),
            "an unknown coordinate must suppress the post, not mint a row: {route:?}"
        );
    }

    /// A registry-metadata fallback for a purl hopper already holds real
    /// content for redirects onto that sha instead of minting a new one.
    #[tokio::test]
    async fn known_purl_redirects_the_registry_fallback() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock corpus");
        let addr = listener.local_addr().expect("addr");
        let real_sha = "b".repeat(64);
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept lookup");
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let body = format!(r#"{{"sha256":"{real_sha}"}}"#);
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len(),
            );
            stream.write_all(head.as_bytes()).expect("write head");
            stream.write_all(body.as_bytes()).expect("write body");
        });

        let corpus = super::super::corpus::Corpus::new(Some(&format!("http://{addr}")))
            .expect("corpus configured");
        let route = tokio::task::spawn_blocking(move || {
            super::offer_registry_fallback(
                Some(&corpus),
                None,
                "pkg:npm/known-package@1.0.0",
                "known-package@1.0.0.registry.json",
                None,
            )
        })
        .await
        .expect("task");
        server.join().expect("server thread");

        match route {
            crate::engine::HopperRoute::Redirect(sha) => assert_eq!(sha, "b".repeat(64)),
            other => panic!("expected a redirect onto the known sha, got {other:?}"),
        }
    }
    use axum::http::StatusCode;

    /// A decision that is about us rather than about the artifact must never
    /// stand in for a run.
    ///
    /// `/v1/analyze` short-circuits on this predicate, so an `unavailable`
    /// slipping through would turn a corpus outage into a silent refusal to
    /// analyze anything — the caller would be told we could not find out, about
    /// work we never attempted. `unknown` is the same mistake in the other
    /// direction: it reports that nobody has analyzed the artifact, which is
    /// precisely the state the caller asked us to change.
    /// A finding says only what is known about it.
    ///
    /// The decision object around it keeps every key at all times; a finding
    /// does not, because how much is known about one varies by where it came
    /// from. A corpus finding knows the trait and its criticality and nothing
    /// else — sending four nulls to say so is most of the object.
    #[test]
    fn a_finding_omits_what_it_does_not_know() {
        use super::super::corpus::CorpusFinding;
        use super::V1Finding;
        let corpus = V1Finding::from_corpus(&CorpusFinding {
            id: "intel/feed/malicious".into(),
            crit: 5,
            desc: Some("Cited as malicious by 3 independent sources.".into()),
        });
        let json = serde_json::to_value(&corpus).expect("serialize");
        let obj = json.as_object().expect("object");
        assert!(obj.contains_key("id"), "id is the finding's identity");
        assert!(obj.contains_key("crit"), "crit is always known");
        assert!(obj.contains_key("desc"), "a desc that exists must be sent");
        for absent in ["file", "pkg", "off", "line"] {
            assert!(
                !obj.contains_key(absent),
                "{absent} is unknown here and must be omitted, not null"
            );
        }
        // Never null: absent is how "nothing to say" is spelled in a finding.
        assert!(
            obj.values().all(|v| !v.is_null()),
            "a null survived into a finding: {json}"
        );
    }

    /// What may answer `/v1/analyze` without spending a slot.
    ///
    /// The rule is "anything that actually answers the question", which is
    /// wider than "a measurement of ours" — an operator's call, made because a
    /// fast answer from what we already know beats rediscovering it. What stays
    /// excluded is what was always excluded: `unanalyzed`, which is the very thing
    /// the route exists to fix, and `unavailable`, which is a statement about us
    /// rather than about the artifact.
    #[test]
    fn only_a_real_answer_may_replace_an_analysis() {
        use super::V1Decision;
        use super::decision::Decision;
        let purl = Some("pkg:npm/left-pad@1.3.0");
        assert!(!V1Decision::unanalyzed(None, purl).is_answerable());
        assert!(!V1Decision::unavailable(None, purl).is_answerable());

        // A decision with no level answers nothing, whatever it is labelled.
        assert!(!V1Decision::empty(Decision::Block, None, purl).is_answerable());

        let measured = |d| {
            let mut v = V1Decision::empty(d, None, purl);
            v.engine_version = Some("2.8.0".into());
            v.fires_at = Some(10);
            v
        };
        assert!(measured(Decision::Allow).is_answerable());
        assert!(measured(Decision::Block).is_answerable());
    }

    /// A derived answer may answer, but must never claim to be a measurement:
    /// the absent engine is what stops it being cached as one downstream, and
    /// what `?fresh=1` exists to get past.
    #[test]
    fn a_derived_answer_answers_without_claiming_an_engine() {
        use super::V1Decision;
        let purl = Some("pkg:npm/left-pad@1.3.0");
        for d in [
            crate::bloom_repo::Decision::Skip,
            crate::bloom_repo::Decision::SightedHostile,
            crate::bloom_repo::Decision::SightedSuspicious,
            crate::bloom_repo::Decision::KnownBad,
        ] {
            let derived = V1Decision::bloom(d, None, purl, 25).expect("answerable");
            assert!(derived.is_answerable(), "{d:?}");
            assert!(!derived.is_verdict(), "{d:?} is not a measurement");
            assert!(derived.engine_version.is_none(), "{d:?}");
            assert!(derived.analyzed_at.is_none(), "{d:?}");
        }
        // No filter had an opinion: nothing to answer with.
        assert!(V1Decision::bloom(crate::bloom_repo::Decision::Unknown, None, purl, 25).is_none());
    }

    /// `fresh` is opt-in on the same affirmative-only terms as `force`, and is
    /// a separate question from it: `force` spends a slot, `fresh` chooses
    /// which layer may answer. A caller can want either without the other.
    #[test]
    fn only_an_affirmative_fresh_bypasses_the_filters() {
        use super::V1LookupQuery;
        let q = |raw| V1LookupQuery::parse(Some(raw));
        assert!(q("purl=pkg:npm/left-pad@1.3.0&fresh=1").fresh);
        assert!(q("purl=pkg:npm/left-pad@1.3.0&fresh=true").fresh);
        assert!(q("purl=pkg:npm/left-pad@1.3.0&fresh=yes").fresh);
        assert!(!q("purl=pkg:npm/left-pad@1.3.0&fresh=0").fresh);
        assert!(!q("purl=pkg:npm/left-pad@1.3.0&fresh=").fresh);
        assert!(!q("purl=pkg:npm/left-pad@1.3.0").fresh);

        // Independent of `force`, in both directions.
        let both = q("purl=pkg:npm/left-pad@1.3.0&fresh=1&force=0");
        assert!(both.fresh && !both.force);
        let other = q("purl=pkg:npm/left-pad@1.3.0&fresh=0&force=1");
        assert!(!other.fresh && other.force);
    }

    /// The header spelling is an alias, not an override: either opts in, and a
    /// proxy that adds the header cannot be defeated by a stale `fresh=0`
    /// further down the chain.
    #[test]
    fn the_fresh_header_is_an_alias_that_only_opts_in() {
        use super::V1LookupQuery;
        let headers = |value: Option<&str>| {
            let mut h = axum::http::HeaderMap::new();
            if let Some(v) = value {
                h.insert(
                    "x-hopper-fresh",
                    axum::http::HeaderValue::from_str(v).expect("header value"),
                );
            }
            h
        };
        let q = |raw, header| {
            V1LookupQuery::parse(Some(raw))
                .with_fresh_header(&headers(header))
                .fresh
        };
        let base = "purl=pkg:npm/left-pad@1.3.0";
        assert!(q(base, Some("1")), "the header alone opts in");
        assert!(q(base, Some("true")));
        assert!(!q(base, Some("0")), "a negative header is not an opt-in");
        assert!(!q(base, None));
        // Neither spelling can opt back out of the other.
        assert!(q("purl=pkg:npm/left-pad@1.3.0&fresh=1", Some("0")));
        assert!(q("purl=pkg:npm/left-pad@1.3.0&fresh=0", Some("1")));
    }

    /// Forcing a fresh run is opt-in, and only an affirmative spelling opts in.
    /// The expensive reading of an ambiguous value is the one that burns an
    /// analysis slot, so anything else leaves the cheap path in place.
    #[test]
    fn only_an_affirmative_force_spends_a_slot() {
        use super::V1LookupQuery;
        let q = |raw| V1LookupQuery::parse(Some(raw)).force;
        assert!(q("purl=pkg:npm/left-pad@1.3.0&force=1"));
        assert!(q("purl=pkg:npm/left-pad@1.3.0&force=true"));
        assert!(q("purl=pkg:npm/left-pad@1.3.0&force=yes"));
        assert!(!q("purl=pkg:npm/left-pad@1.3.0&force=0"));
        assert!(!q("purl=pkg:npm/left-pad@1.3.0&force=false"));
        assert!(!q("purl=pkg:npm/left-pad@1.3.0&force="));
        assert!(!q("purl=pkg:npm/left-pad@1.3.0"));
    }

    #[test]
    fn follow_policy_repeats_union_and_overrides_the_server_default() {
        use super::{V1LookupQuery, v1_follow_policy};
        use crate::fetch::FetchPolicy;

        let configured: FetchPolicy = "all".parse().unwrap();
        let query = V1LookupQuery::parse(Some(
            "purl=pkg:npm/app@1.0.0&follow=references&follow=ci-actions",
        ));
        assert_eq!(query.follow, ["references", "ci-actions"]);
        let effective = v1_follow_policy(&query, configured).expect("valid selection");
        assert!(effective.urls && effective.packages && effective.deps && effective.ci);

        let dependencies_only: FetchPolicy = "dependencies".parse().unwrap();
        let references = V1LookupQuery::parse(Some("follow=references"));
        let effective = v1_follow_policy(&references, dependencies_only)
            .expect("a request may override the configured categories");
        assert!(effective.urls && effective.packages);
        assert!(!effective.deps && !effective.ci);

        let legacy = V1LookupQuery::parse(Some("follow=deps"));
        assert!(v1_follow_policy(&legacy, configured).is_err());
    }

    /// One shape, whichever route answered and whatever it found.
    ///
    /// A caller writes one parser against nine keys and reads `decision` to
    /// know what happened. That only holds if every way of producing a decision
    /// produces the same keys — a field present on a lookup and absent on an
    /// analysis is a field nobody can rely on, and the difference would show up
    /// as an intermittent null rather than as an error.
    #[test]
    fn every_decision_has_the_same_shape() {
        use super::super::corpus::{CorpusFinding, CorpusRecord};
        use super::V1Decision;
        use crate::lookup::{Hit, Verdict};

        let stored = Verdict {
            sha256: "a".repeat(64),
            lvl: Some(3),
            eng: "2.8.0".into(),
            at: "2026-08-01T00:00:00Z".into(),
            purl: Some("pkg:npm/evil@1.0.0".into()),
            why: Some("Reverse shell in postinstall.".into()),
            hits: vec![Hit {
                id: "objectives/c2/backdoor".into(),
                crit: 5,
                file: "lib/install.js".into(),
                pkg: String::new(),
                desc: "Spawns bash".into(),
                off: Some(109),
                line: Some(12),
            }],
        };
        let from_corpus = CorpusRecord {
            sha256: Some("a".repeat(64)),
            purl: Some("pkg:npm/evil@1.0.0".into()),
            fires_at: Some(3),
            engine_version: Some("2.8.0".into()),
            analyzed_at: Some("2026-08-01T00:00:00Z".into()),
            reason: Some("Reverse shell in postinstall.".into()),
            findings: vec![CorpusFinding {
                id: "objectives/c2/backdoor".into(),
                crit: 5,
                desc: None,
            }],
        };

        let shapes = [
            // What /v1/lookup and /v1/analyze both answer with on a hit.
            V1Decision::stored(&stored, Some("pkg:npm/evil@1.0.0"), 25),
            // What a lookup answers with when the corpus knew instead.
            V1Decision::corpus(&from_corpus, None, Some("pkg:npm/evil@1.0.0"), 25),
            V1Decision::unanalyzed(None, Some("pkg:npm/evil@1.0.0")),
            V1Decision::unavailable(None, Some("pkg:npm/evil@1.0.0")),
        ];

        let keys = |d: &V1Decision| -> Vec<String> {
            let v = serde_json::to_value(d).expect("serializes");
            let mut k: Vec<String> = v
                .as_object()
                .expect("an object")
                .keys()
                .map(String::clone)
                .collect();
            k.sort();
            k
        };
        let expected = keys(&shapes[0]);
        assert_eq!(expected.len(), 9, "the shape changed: {expected:?}");
        for shape in &shapes[1..] {
            assert_eq!(
                keys(shape),
                expected,
                "a decision answered with different keys"
            );
        }

        // A finding, by contrast, does NOT keep one shape across the two
        // sources, and that is deliberate. The decision object answers a fixed
        // set of questions, so its keys never move; a finding's content depends
        // on where it came from, and the corpus holds no file or offset at all.
        // Sending four nulls to say so is most of the object, so absence is how
        // "nothing to say" is spelled here.
        let finding_keys = |d: &V1Decision| -> Vec<String> {
            let v = serde_json::to_value(d).expect("serializes");
            let mut k: Vec<String> = v["findings"][0]
                .as_object()
                .expect("a finding")
                .keys()
                .map(String::clone)
                .collect();
            k.sort();
            k
        };
        let stored_finding = finding_keys(&shapes[0]);
        let corpus_finding = finding_keys(&shapes[1]);
        assert_eq!(
            corpus_finding,
            ["crit", "id"],
            "a corpus finding must carry only what it knows",
        );
        // Whatever a finding does carry, it is never a null.
        for shape in &shapes[..2] {
            let v = serde_json::to_value(shape).expect("serializes");
            assert!(
                v["findings"][0]
                    .as_object()
                    .expect("a finding")
                    .values()
                    .all(|x| !x.is_null()),
                "a null survived into a finding: {}",
                v["findings"][0]
            );
        }
        // The identity and severity are the two a caller may always rely on,
        // whichever side of the index answered.
        for id_or_crit in ["id", "crit"] {
            assert!(
                stored_finding.iter().any(|k| k == id_or_crit)
                    && corpus_finding.iter().any(|k| k == id_or_crit),
                "{id_or_crit} must be present on every finding",
            );
        }
    }

    /// A caller gets an answer about the package they named, spelled the way
    /// they named it.
    ///
    /// Found in production: `pkg:pypi/info.gianlucacosta.eos.core@2.0.2` came
    /// back answered about `pkg:pypi/info-gianlucacosta-eos-core@2.0.2`. Both
    /// name the same project — PEP 503 folds `.` and `_` to `-` — but a caller
    /// cannot know that without implementing PEP 503, and a lookup that names
    /// fifty packages is matched to its request by this field. Rewriting it
    /// breaks that correlation silently, and only for names with a `.` or `_`
    /// in them.
    #[test]
    fn a_decision_is_spelled_the_way_the_caller_asked() {
        use super::V1Decision;
        use crate::lookup::Verdict;

        let asked = "pkg:pypi/info.gianlucacosta.eos.core@2.0.2";
        let normalized = "pkg:pypi/info-gianlucacosta-eos-core@2.0.2";
        assert_eq!(
            super::normalize_pkg_purl(asked).as_deref(),
            Ok(normalized),
            "the premise: these two spellings are one package",
        );

        // PEP 503 lowercases as well as folding separators, and that half was
        // sighted separately in production: `pkg:pypi/ImportanceScore@1.2` came
        // back answered about `pkg:pypi/importancescore@1.2`. Same cause, and a
        // caller whose package name has no `.` or `_` in it at all.
        let mixed = "pkg:pypi/ImportanceScore@1.2";
        assert_eq!(
            super::normalize_pkg_purl(mixed).as_deref(),
            Ok("pkg:pypi/importancescore@1.2"),
            "the premise: case folds too",
        );
        let cased = V1Decision::unanalyzed(None, Some("pkg:pypi/importancescore@1.2"))
            .asked_about(Some(mixed));
        assert_eq!(
            serde_json::to_value(&cased).expect("serializes")["purl"],
            mixed,
            "the caller's capitalization was rewritten",
        );

        let stored = Verdict {
            sha256: "a".repeat(64),
            lvl: Some(-1),
            eng: "2.8.0".into(),
            at: "2026-08-01T00:00:00Z".into(),
            // What the index holds, which is always the normalized key.
            purl: Some(normalized.to_string()),
            why: None,
            hits: Vec::new(),
        };

        // Every kind of decision, since a caller correlating a list of fifty
        // gets whichever kind we happen to have.
        let decisions = [
            V1Decision::stored(&stored, Some(normalized), 25).asked_about(Some(asked)),
            V1Decision::unanalyzed(None, Some(normalized)).asked_about(Some(asked)),
            V1Decision::unavailable(None, Some(normalized)).asked_about(Some(asked)),
        ];
        for d in &decisions {
            let v = serde_json::to_value(d).expect("serializes");
            assert_eq!(
                v["purl"], asked,
                "answered about a different spelling than was asked about",
            );
        }

        // The digest still names the artifact, and is what proves two
        // spellings are one thing.
        let v = serde_json::to_value(&decisions[0]).expect("serializes");
        assert_eq!(v["sha256"], "a".repeat(64));

        // A lookup by digest alone has no spelling to echo, so the stored one
        // stands rather than becoming null.
        let by_sha = V1Decision::stored(&stored, None, 25).asked_about(None);
        let v = serde_json::to_value(&by_sha).expect("serializes");
        assert_eq!(v["purl"], normalized);
    }

    /// Findings are evidence for the decision, not a dump of everything the
    /// scanner noticed. A benign crate matched ten "Rust test marker" traits at
    /// `crit: 3`; answering an `allow` with all ten invites a caller to read
    /// significance into noise, and it disagreed with the same artifact looked
    /// up from the corpus, where the trigger had already cut them.
    #[test]
    fn only_the_strongest_few_findings_reach_the_wire() {
        use super::super::corpus::{CorpusFinding, CorpusRecord};
        use super::V1Decision;
        use crate::lookup::{Hit, Verdict};

        let hit = |id: &str, crit: u8| Hit {
            id: id.into(),
            crit,
            file: "lib/install.js".into(),
            pkg: String::new(),
            desc: String::new(),
            off: None,
            line: None,
        };
        let ids =
            |d: &V1Decision| -> Vec<String> { d.findings.iter().map(|f| f.id.clone()).collect() };

        let benign = Verdict {
            sha256: "a".repeat(64),
            lvl: Some(-1),
            eng: "2.8.0".into(),
            at: "2026-08-01T00:00:00Z".into(),
            purl: Some("pkg:cargo/tokio@1.40.0".into()),
            why: None,
            hits: (0..10)
                .map(|i| hit(&format!("testing/harness::{i}"), 3))
                .collect(),
        };
        let d = V1Decision::stored(&benign, None, 25);
        assert_eq!(d.decision, super::decision::Decision::Allow);
        assert!(
            ids(&d).is_empty(),
            "sub-threshold traits were reported as evidence: {:?}",
            ids(&d),
        );

        // Worst first, capped at three, and equal criticalities keep the order
        // their source listed them in.
        let noisy = Verdict {
            hits: vec![
                hit("weak", 4),
                hit("worst", 6),
                hit("dropped", 3),
                hit("strong-a", 5),
                hit("strong-b", 5),
                hit("cut", 4),
            ],
            ..benign
        };
        assert_eq!(
            ids(&V1Decision::stored(&noisy, None, 25)),
            ["worst", "strong-a", "strong-b"],
        );

        // The corpus applies the same rule in a trigger, so passing it through
        // here changes nothing — which is the point: one artifact reads the
        // same whichever side of the index answered.
        let record = CorpusRecord {
            sha256: Some("a".repeat(64)),
            purl: Some("pkg:cargo/tokio@1.40.0".into()),
            fires_at: Some(-1),
            engine_version: Some("2.8.0".into()),
            analyzed_at: Some("2026-08-01T00:00:00Z".into()),
            reason: None,
            findings: (0..10)
                .map(|i| CorpusFinding {
                    id: format!("testing/harness::{i}"),
                    crit: 3,
                    desc: None,
                })
                .collect(),
        };
        assert!(
            ids(&V1Decision::corpus(&record, None, None, 25)).is_empty(),
            "a corpus record reported findings a stored verdict would have cut",
        );
    }

    /// `unavailable` is a statement about us, not about the artifact, so nothing
    /// about the artifact may ride along on one. A caller that could read a
    /// severity or a budget out of a failed lookup would eventually branch on
    /// it, and would then be treating our outage as evidence.
    #[test]
    fn an_unavailable_decision_carries_nothing_about_the_package() {
        let d = super::V1Decision::unavailable(Some("a"), Some("pkg:npm/x@1.0.0"));
        let v = serde_json::to_value(&d).expect("serializes");
        assert_eq!(v["decision"], "unavailable");
        assert_eq!(v["purl"], "pkg:npm/x@1.0.0");
        for empty in [
            "severity",
            "fires_at",
            "reason",
            "engine_version",
            "analyzed_at",
        ] {
            assert!(
                v[empty].is_null(),
                "{empty} leaked into an unavailable decision"
            );
        }
        assert_eq!(v["findings"].as_array().map(Vec::len), Some(0));
    }

    /// The `llm=` field separates a minute-long endpoint query from a replay of
    /// the prompt cache, which are otherwise distinguishable only by timing.
    #[test]
    fn llm_source_names_where_the_verdict_came_from() {
        use crate::interpret::Interpretation;

        let pass = |cached, error: Option<&str>| Interpretation {
            corroborated: false,
            grade: None,
            outcome: crate::Classification::Benign,
            blended: 0.1,
            interpretation: String::new(),
            model: "m".to_string(),
            error: error.map(str::to_string),
            analyzer_directed: false,
            cached,
        };
        assert_eq!(super::llm_source(None), None, "no pass ran");
        assert_eq!(super::llm_source(Some(&pass(false, None))), Some("queried"));
        assert_eq!(super::llm_source(Some(&pass(true, None))), Some("cached"));
        assert_eq!(
            super::llm_source(Some(&pass(true, Some("timeout")))),
            Some("failed"),
            "a failed pass is not a cache hit even if a stale entry existed",
        );
    }

    /// A follower renders the leader's failure verbatim: same status, same body,
    /// no second analysis and no second error.
    #[tokio::test]
    async fn a_replayed_failure_keeps_its_status_and_body() {
        let outcome = Outcome::Rendered {
            status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
            body: serde_json::json!({ "error": "Unsupported file type", "detail": "nope" }),
        };
        let key = super::FlightKey::Sha("f".repeat(64));
        let response = flight_response(&outcome, 42, false, &key);
        assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        // Even a replayed failure names its artifact on the access line.
        assert!(
            response
                .extensions()
                .get::<super::super::access::Subject>()
                .is_some(),
            "flight responses carry their subject",
        );

        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let parsed: serde_json::Value = serde_json::from_slice(&body).expect("parse body");
        assert_eq!(parsed["error"], "Unsupported file type");
        assert_eq!(parsed["detail"], "nope");
    }

    /// A hostile upload filename must not be able to panic the request. A
    /// Unicode-aware filter kept multi-byte characters, and the tail
    /// truncation then sliced one in half.
    #[test]
    fn sanitize_upload_filename_survives_multibyte_names() {
        let raw = "\u{3041}".repeat(80) + ".zip";
        let name = super::sanitize_upload_filename(&raw);
        assert_eq!(name.len(), 63);
        assert!(name.is_ascii(), "{name}");
        assert!(name.ends_with(".zip"), "the extension must survive: {name}");
    }

    #[test]
    fn sanitize_upload_filename_defuses_paths_and_control_characters() {
        assert_eq!(
            super::sanitize_upload_filename("../../etc/shadow"),
            "______etc_shadow"
        );
        assert_eq!(
            super::sanitize_upload_filename("a\nb\r\u{202e}gpj.exe"),
            "a_b__gpj.exe"
        );
        // A name already inside the alphabet is left exactly as it is.
        assert_eq!(
            super::sanitize_upload_filename("left-pad-1.3.0.tgz"),
            "left-pad-1.3.0.tgz"
        );
    }

    #[test]
    fn classify_unsupported_file_type_as_415() {
        assert_eq!(
            classify_analysis_error("Unsupported file type: Unknown"),
            StatusCode::UNSUPPORTED_MEDIA_TYPE
        );
    }

    #[test]
    fn classify_invalid_archive_as_422() {
        assert_eq!(
            classify_analysis_error("Archive is encrypted but no passwords configured"),
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    #[test]
    fn classify_unexpected_failure_as_500() {
        assert_eq!(
            classify_analysis_error("model evaluation failed"),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    /// A malformed upload is the caller's problem, not a server fault: cleave
    /// reports it from deep in the archive reader, so it arrives as a 422 only
    /// because the whole chain is classified.
    #[test]
    fn classify_corrupt_archive_as_422() {
        assert_eq!(
            classify_analysis_error(
                "cleave analysis of bad.tgz: Failed to read tar entry: corrupt deflate stream"
            ),
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    /// The wrapping context is part of what gets classified: a "truncated"
    /// cause buried under `cleave analysis of x.tgz` is still a 422, not a 500.
    #[test]
    fn classify_reads_the_whole_error_chain() {
        assert_eq!(
            classify_analysis_error("cleave analysis of x.tgz: truncated gzip stream"),
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    /// A request without `registry` still parses — the field is optional, so
    /// callers predating it (and files with no hopper record) are unaffected.
    #[test]
    fn analyze_path_registry_is_optional() {
        let req: super::AnalyzePathRequest =
            serde_json::from_str(r#"{"path":"/tmp/x/a.tgz"}"#).expect("parses without registry");
        assert_eq!(req.path, "/tmp/x/a.tgz");
        assert!(req.registry.is_none());
    }

    /// A supplied record round-trips through the same provenance parser the
    /// CLI's `--registry-map` entries use, so both scan paths accept the exact
    /// document hopper hands promoter.
    #[test]
    fn analyze_path_registry_parses_as_provenance() {
        let body = r#"{"path":"/tmp/x/a.tgz","registry":{"ecosystem":"npm","name":"left-pad","version":"1.3.0"}}"#;
        let req: super::AnalyzePathRequest = serde_json::from_str(body).expect("parses");
        let raw = req.registry.expect("registry present");
        let provenance = crate::provenance::registry_provenance(raw.get().as_bytes())
            .expect("a bare normalized record is one of the accepted shapes");
        assert_eq!(provenance.record.name, "left-pad");
    }

    /// Provenance enriches a scan but is never required, so a document that
    /// carries no recoverable record degrades to `None` (scan without registry
    /// facts) rather than failing the request.
    #[test]
    fn analyze_path_unparseable_registry_degrades_to_none() {
        let body = r#"{"path":"/tmp/x/a.tgz","registry":[1,2,3]}"#;
        let req: super::AnalyzePathRequest = serde_json::from_str(body).expect("parses");
        let raw = req.registry.expect("registry present");
        assert!(crate::provenance::registry_provenance(raw.get().as_bytes()).is_none());
    }

    /// The lookup routes read their decision straight off the filters, so the
    /// fixture coverage lives with the filters rather than with a handler.
    #[test]
    fn filters_answer_skip_known_bad_and_unknown() {
        use crate::bloom_repo::{Decision, KEY_SCHEME, Lookup, purl_key};
        use burton::{KeySets, Record, Tier};

        let tmp = tempfile::tempdir().expect("tempdir");
        let mut good_sha = [0u8; 32];
        good_sha[0] = 1;
        let mut bad_sha = [0u8; 32];
        bad_sha[0] = 2;

        let mut sets = KeySets::new();
        sets.insert(
            Tier::Good,
            Record {
                purl: purl_key("pkg:npm/good@1"),
                sha256: Some(good_sha),
            },
        );
        sets.insert(
            Tier::Bad,
            Record {
                purl: purl_key("pkg:npm/evil@1"),
                sha256: Some(bad_sha),
            },
        );
        burton::build::write_bundle(
            tmp.path(),
            &sets.into_filters(1e-9),
            "2026-08-31",
            KEY_SCHEME,
        )
        .expect("write bundle");

        let lk = Lookup::load_from(tmp.path());
        assert_eq!(lk.memo_purl("pkg:npm/good@1"), Decision::Skip);
        assert_eq!(lk.memo_purl("pkg:npm/evil@1"), Decision::KnownBad);
        assert_eq!(lk.memo_sha256(&good_sha), Decision::Skip);
        let mut unseen = [0u8; 32];
        unseen[0] = 0xab;
        assert_eq!(lk.memo_sha256(&unseen), Decision::Unknown);
    }

    /// Without filters installed every key is `unknown` — fail closed, never
    /// an error, so a lookup still answers.
    #[test]
    fn filters_absent_reads_unknown() {
        use crate::bloom_repo::{Decision, Lookup};
        let lk = Lookup::default();
        assert_eq!(lk.memo_purl("pkg:npm/left-pad@1.3.0"), Decision::Unknown);
        assert_eq!(lk.memo_sha256(&[7u8; 32]), Decision::Unknown);
    }

    #[test]
    fn normalize_pkg_purl_accepts_bare_and_full() {
        assert_eq!(
            super::normalize_pkg_purl("pkg:npm/left-pad@1.3.0").unwrap(),
            "pkg:npm/left-pad@1.3.0"
        );
        assert_eq!(
            super::normalize_pkg_purl("npm/left-pad@1.3.0").unwrap(),
            "pkg:npm/left-pad@1.3.0"
        );
        assert!(super::normalize_pkg_purl("").is_err());
        assert!(super::normalize_pkg_purl("not a purl").is_err());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod artifact_upload_tests {
    use super::*;

    /// The collector must read the same whichever path ingested the sample, so
    /// hopper's provenance does not fork by ingest route.
    #[test]
    fn collector_matches_the_cli_form() {
        let got = upload_collector();
        assert!(got.starts_with("scan+"), "collector = {got}");
        assert_eq!(got, upload_collector(), "collector is not stable");
    }

    /// A fetched root's bytes live in fletch's blob cache, keyed by locator —
    /// not on disk. Building the artifact from the fetch record is what makes
    /// the upload possible at all.
    ///
    /// It also pins a dependency on the caller: the PURL slot hopper projects
    /// into its queryable `purl_base` column is filled only when the locator
    /// carries the `pkg:` prefix. Beamline forwards the canonical form, so this
    /// holds today; if that ever changes, the artifact still uploads but
    /// `/api/sample?purl=` stops finding it, which is exactly the silent gap
    /// this whole path exists to close.
    #[test]
    fn fetched_root_offers_cached_bytes_and_its_purl() {
        let record = |locator: &str| {
            serde_json::from_value::<fletch::fetch::FetchRecord>(serde_json::json!({
                "locator": locator,
                "resolved_url": "https://crates.io/api/v1/crates/libc/0.2.101/download",
                "fetched_at": 0,
                "cached": false,
                "outcome": "ok",
            }))
            .expect("FetchRecord")
        };
        let build = |rec: &fletch::fetch::FetchRecord| {
            crate::engine::collect_upload_artifacts(
                std::path::Path::new("libc-0.2.101.crate"),
                &"a".repeat(64),
                1234,
                "scan+test",
                None,
                Some(rec),
            )
        };

        let arts = build(&record("pkg:cargo/libc@0.2.101"));
        assert_eq!(arts.len(), 1);
        let art = &arts[0];
        assert_eq!(art.size, 1234);
        assert!(
            matches!(art.bytes, crate::upload::ArtifactBytes::Cached { .. }),
            "a fetched root must take its bytes from the blob cache, not a path",
        );
        assert!(
            art.backfill,
            "a PURL-identified artifact is worth backfilling onto an existing sample",
        );
        let sidecar: serde_json::Value =
            serde_json::from_slice(&art.sidecar).expect("sidecar json");
        assert_eq!(
            sidecar["package"]["purl"], "pkg:cargo/libc@0.2.101",
            "the PURL must reach the sidecar slot hopper reads into purl_base",
        );
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod pick_verdict_tests {
    use super::pick_verdict;
    use crate::lookup::Verdict;

    fn verdict(sha: &str, eng: &str) -> Verdict {
        Verdict {
            sha256: sha.to_owned(),
            lvl: Some(-1),
            eng: eng.to_owned(),
            at: "2026-01-01T00:00:00Z".to_owned(),
            purl: None,
            why: None,
            hits: Vec::new(),
        }
    }

    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OTHER: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[test]
    fn a_digest_hit_wins_and_costs_no_purl_lookup() {
        let mut asked = false;
        let got = pick_verdict(
            Some(verdict(SHA, "by-sha")),
            || {
                asked = true;
                Some(verdict(SHA, "by-purl"))
            },
            SHA,
        );
        assert_eq!(got.expect("verdict").eng, "by-sha");
        assert!(!asked, "an exact hit must not cost a second index lookup");
    }

    // The second chance: the index knows the release even though these exact
    // bytes are new to it, and the digests agree.
    #[test]
    fn the_purl_answers_when_the_digest_is_unknown() {
        let got = pick_verdict(None, || Some(verdict(SHA, "by-purl")), SHA);
        assert_eq!(got.expect("verdict").eng, "by-purl");
    }

    // The guard the pair exists for: the release resolved to other bytes.
    #[test]
    fn a_purl_verdict_for_other_bytes_is_refused() {
        let got = pick_verdict(None, || Some(verdict(OTHER, "different-artifact")), SHA);
        assert!(
            got.is_none(),
            "served a verdict about bytes nobody asked about"
        );
    }

    #[test]
    fn digest_comparison_is_case_insensitive() {
        let got = pick_verdict(None, || Some(verdict(&SHA.to_uppercase(), "by-purl")), SHA);
        assert!(got.is_some(), "hex case must not decide identity");
    }

    #[test]
    fn neither_key_known_is_no_verdict() {
        assert!(pick_verdict(None, || None, SHA).is_none());
    }
}
