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
use super::access::RequestId;
use super::acl::Trusted;
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
) -> Outcome {
    match result {
        AnalysisOutcome::Ok(Ok(scan_result)) => {
            tracing::info!(
                id = request_id,
                key = %key,
                elapsed_ms,
                classification = %scan_result.classification,
                probability = scan_result.probability,
                "<-- 200 OK",
            );
            index_verdict(&scan_result, key.purl());
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
/// caller's own wall time, so a follower reports how long *it* waited.
fn flight_response(outcome: &Outcome, elapsed_ms: u64) -> Response {
    match outcome {
        Outcome::Report(result) => {
            let mut resp = Json(result.envelope_ref()).into_response();
            resp.headers_mut().insert("X-Total-Ms", elapsed_ms.into());
            resp
        }
        Outcome::Rendered { status, body } => (*status, Json(body)).into_response(),
    }
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
    }))
    .into_response()
}

/// Query string for `GET /purl`.
///
/// The PURL travels as a query parameter rather than a path segment because its
/// own grammar carries `/`, `?` and `#`: `pkg:npm/@scope/name@1.0.0?arch=x64`
/// in a path would have everything from the `?` parsed as the *URL's* query and
/// a `#subpath` dropped by the client, silently keying on a different package.
/// Qualifiers are load-bearing here — `fletch::purl::identity` keeps them.
#[derive(Debug, Default, serde::Deserialize)]
pub(super) struct PurlQuery {
    purl: Option<String>,
}

/// GET /sha256/{sha256} — what we already know about an artifact.
///
/// Never analyzes: no slot, no fetch, and it answers while the model is still
/// loading. A caller that gets `404 unknown sample` asks for a real analysis
/// with `/analyze` or `/analyze-purl`.
pub(super) async fn lookup_sha(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(sha256): axum::extract::Path<String>,
) -> Response {
    let Some(digest) = crate::bloom::parse_sha256_hex(sha256.trim()) else {
        return error_response(StatusCode::BAD_REQUEST, "invalid sha256");
    };
    let sha = sha256.trim().to_ascii_lowercase();
    let decision = crate::bloom_repo::global()
        .as_deref()
        .map_or(crate::bloom_repo::Decision::Unknown, |lk| {
            lk.memo_sha256(&digest)
        });
    let verdict = crate::lookup::global().and_then(|index| index.get_sha(&sha));
    lookup_response(&state, verdict.as_ref(), decision, &sha, None)
}

/// GET /purl?purl=… — what we already know about a package.
///
/// Same contract as [`lookup_sha`]: local knowledge only, never an analysis.
pub(super) async fn lookup_purl(
    State(state): State<Arc<AppState>>,
    Query(q): Query<PurlQuery>,
) -> Response {
    let purl = q.purl.as_deref().map(str::trim).unwrap_or_default();
    if purl.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "missing purl");
    }
    if fletch::purl::identity(purl).is_none() {
        return error_response(StatusCode::BAD_REQUEST, "invalid purl");
    }
    let decision = crate::bloom_repo::global()
        .as_deref()
        .map_or(crate::bloom_repo::Decision::Unknown, |lk| lk.memo_purl(purl));
    let verdict = crate::lookup::global().and_then(|index| index.get_purl(purl));
    let sha = verdict.as_ref().map_or("", |v| v.sha256.as_str()).to_owned();
    lookup_response(&state, verdict.as_ref(), decision, &sha, Some(purl))
}

/// Render a lookup answer.
///
/// A stored verdict is a 200; holding nothing is a 404, and the bloom decision
/// rides on both. That keeps the two kinds of knowledge distinguishable — a
/// filter says "probably not worth scanning", an analysis says what the thing
/// *is* — while still answering both questions in one round trip.
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
        // A miss is not cacheable for any length of time: it becomes a hit the
        // moment anything analyzes this artifact.
        let mut resp = (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "unknown sample",
                "bloom": decision.as_str(),
            })),
        )
            .into_response();
        resp.headers_mut().insert(
            axum::http::header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store"),
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
    headers.insert("X-Scan-Source", axum::http::HeaderValue::from_static("index"));
    resp
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
    if attachment.leads() {
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
                        run_file_analysis(state, request_id, upload, &flight, resources, permit)
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
    flight_response(&outcome, crate::duration_ms(request_start.elapsed()))
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

/// Run one uploaded-file analysis on behalf of every request attached to
/// `flight`. Takes the staged upload and deletes it on the way out.
async fn run_file_analysis(
    state: Arc<AppState>,
    request_id: u64,
    upload: Upload,
    flight: &Arc<Flight>,
    resources: Arc<super::ModelResources>,
    permit: tokio::sync::OwnedSemaphorePermit,
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
    let handle = tokio::task::spawn_blocking(move || {
        // Record the OS thread servicing this analysis.
        if let Some(req) = phase_state.in_flight.get(&request_id) {
            req.thread_id.store(current_thread_id(), Ordering::Relaxed);
        }
        let result = classify_file(
            &path,
            &filename,
            &resources,
            slow_rule_ms,
            None,
            Some(&cancel_flag),
            phase_tracker.as_ref(),
            None, // interactive upload carries no fetch-time registry provenance
            // /analyze returns the envelope and discards the result — only
            // /analyze-path renews results (and their dependencies) on hopper.
            false,
        );
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
    )
}

/// POST /analyze-purl — fetch a package by PURL and analyze it.
///
/// Scan looks up registry provenance itself (age, custody, downloads) and
/// grafts it into the report, the same path as `atomscan purl`. Beamline
/// calls this when a PURL is not in hopper; it is a full analysis and takes
/// a slot. Dependency fetch and LLM interpretation follow the process-wide
/// `--fetch` / `--interpret` flags.
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
    let request_start = Instant::now();

    let purl = match normalize_pkg_purl(&req.purl) {
        Ok(p) => p,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, msg),
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
    if attachment.leads() {
        tracing::info!(id = request_id, purl = %purl, "--> POST /analyze-purl");
        let publisher = state.flights.publisher(attachment.flight());
        match claim_slot(&state, request_id, attachment.flight().key()) {
            Err(outcome) => publisher.publish(outcome),
            Ok((resources, permit)) => {
                let flight = Arc::clone(attachment.flight());
                let state = Arc::clone(&state);
                // Detached, so the analysis outlives whichever request started
                // it: this client hanging up must not abandon the followers.
                tokio::spawn(async move {
                    publisher.publish(
                        run_purl_analysis(state, request_id, &purl, &flight, resources, permit)
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
    flight_response(&outcome, crate::duration_ms(request_start.elapsed()))
}

/// Run one PURL analysis on behalf of every request attached to `flight`.
async fn run_purl_analysis(
    state: Arc<AppState>,
    request_id: u64,
    purl: &str,
    flight: &Arc<Flight>,
    resources: Arc<super::ModelResources>,
    permit: tokio::sync::OwnedSemaphorePermit,
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
    let deps_for_upload = state.uploader.is_some();
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

/// Fetch the PURL's artifact (and its registry record) then classify. Scan
/// looks up provenance itself — beamline does not supply it.
fn classify_purl(
    purl: &str,
    resources: &super::ModelResources,
    slow_rule_ms: u64,
    cancellation: Option<&Arc<AtomicBool>>,
    phase: Option<&cleave::PhaseTracker>,
    deps_for_upload: bool,
) -> anyhow::Result<crate::engine::ScanResult> {
    use fletch::RefLocator;

    let locator = RefLocator::Purl(purl.to_string());
    let (registry, registry_sources) = crate::fetch::registry_with_sources(&locator);
    let registry_provenance = registry.clone().map(|record| {
        crate::provenance::RegistryProvenance::from_record_sources(record, &registry_sources)
    });

    if let Some(reg) = &registry
        && reg.version_removed == Some(true)
        && let Some((name, bytes)) = crate::fetch::registry_document(reg)
    {
        return classify_bytes(
            bytes::Bytes::from(bytes),
            &name,
            resources,
            slow_rule_ms,
            cancellation,
            phase,
            registry_provenance.as_ref(),
            deps_for_upload,
        );
    }

    let (bytes, name, _rec) = match crate::fetch::fetch_one(locator, false) {
        Ok(t) => t,
        Err(e) => match registry.as_ref().and_then(crate::fetch::registry_document) {
            Some((name, bytes)) => {
                return classify_bytes(
                    bytes::Bytes::from(bytes),
                    &name,
                    resources,
                    slow_rule_ms,
                    cancellation,
                    phase,
                    registry_provenance.as_ref(),
                    deps_for_upload,
                );
            }
            None => return Err(e),
        },
    };

    classify_bytes(
        bytes::Bytes::from(bytes),
        &name,
        resources,
        slow_rule_ms,
        cancellation,
        phase,
        registry_provenance.as_ref(),
        deps_for_upload,
    )
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
    use anyhow::Context as _;

    if let Some(p) = phase {
        p.set("cleave:init");
    }
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
    use anyhow::Context as _;

    if let Some(p) = phase {
        p.set("cleave:init");
    }
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
        deps_for_upload,
    )
}

/// Shared tail of [`classify_file`]/[`classify_bytes`]: honor a late cancellation,
/// run feature extraction + model inference, and assemble the [`ScanResult`].
/// `deps_for_upload` marks callers that renew results on hopper and therefore
/// need per-dependency standalone reports captured (worker, `--hopper` server).
fn finish_classify(
    label: &str,
    report: cleave::AnalysisReport,
    resources: &super::ModelResources,
    cancellation: Option<&Arc<AtomicBool>>,
    phase: Option<&cleave::PhaseTracker>,
    root_registry: Option<&crate::provenance::RegistryProvenance>,
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
        resources.fetch,
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
    request_id: Extension<RequestId>,
    Json(req): Json<AnalyzePathRequest>,
) -> Response {
    let request_id = request_id.0.get();
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
                elapsed_ms,
                classification = %scan_result.classification,
                probability = scan_result.probability,
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
    use axum::http::StatusCode;

    /// A follower renders the leader's failure verbatim: same status, same body,
    /// no second analysis and no second error.
    #[tokio::test]
    async fn a_replayed_failure_keeps_its_status_and_body() {
        let outcome = Outcome::Rendered {
            status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
            body: serde_json::json!({ "error": "Unsupported file type", "detail": "nope" }),
        };
        let response = flight_response(&outcome, 42);
        assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

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
        use crate::bloom::{Record, generate};
        use crate::bloom_repo::{Decision, Lookup};

        let tmp = tempfile::tempdir().expect("tempdir");
        let mut good_sha = [0u8; 32];
        good_sha[0] = 1;
        let mut bad_sha = [0u8; 32];
        bad_sha[0] = 2;
        for f in generate(
            vec![
                Record {
                    purl: Some("pkg:npm/good@1".into()),
                    sha256: Some(good_sha),
                },
                Record {
                    purl: Some("pkg:npm/evil@1".into()),
                    sha256: Some(bad_sha),
                },
            ],
            vec![Record {
                purl: Some("pkg:npm/evil@1".into()),
                sha256: Some(bad_sha),
            }],
            1e-9,
        ) {
            let path = tmp.path().join(format!("{}.adbl", f.artifact_stem()));
            std::fs::write(path, f.to_bytes()).expect("write filter");
        }
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
