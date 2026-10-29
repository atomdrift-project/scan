//! The `/v1` routes: a decision a firewall can act on, for an artifact named
//! by package URL, exact URL or digest, or sent as bytes.
//!
//! Every error here is the v1 shape, `{"error": {"code", "message"}}` — see
//! [`ApiError::response`] — whichever layer produced it.

use axum::body::HttpBody as _;
use axum::extract::{Extension, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::AppState;
use super::access::{RequestId, Subject, with_subject};
use super::analyze::{self, Job, RequestFollow, index_query};
use super::corpus::{self, Reached};
use super::decision::{self, V1Decision};
use super::error::ApiError;
use super::flight::{Artifact, Attachment, FlightKey, Outcome};
use super::input::{normalize_pkg_purl, sanitize_upload_filename, valid_http_url};
use crate::lookup::pick_verdict;

/// Query for `GET /v1/lookup` and `POST /v1/analyze`. `purl`, `url` and
/// `sha256` repeat; one `sha256` names one artifact, and only a lookup takes
/// several.
///
/// Parsed from the raw query rather than through `Query<T>`: a repeated key is
/// a sequence, and `serde_urlencoded` — what axum's `Query` is built on —
/// cannot deserialize one. It silently rejects the whole request instead, which
/// would make `?purl=a&purl=b` a 400 with no explanation.
#[derive(Default)]
pub(super) struct V1Query {
    purl: Vec<String>,
    url: Vec<String>,
    sha256: Vec<String>,
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
    /// Bypass outer verdict caches and reconcile a SHA with Hopper. Hopper's
    /// result is reusable only when it was produced by this server's traits;
    /// otherwise the immutable sample bytes are fetched and analyzed here.
    refresh: bool,
    /// A file on this host to analyze (`/v1/analyze` only, and only when the
    /// server was started with `--allow-local-paths`).
    path: Vec<String>,
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

fn matching_traits_version(stored: Option<&str>, local: Option<&str>) -> bool {
    stored
        .zip(local)
        .is_some_and(|(stored, local)| stored == local)
}

impl V1Query {
    /// Fold the header alias into the parsed query. Either spelling opts in;
    /// neither can opt back out, so a proxy that adds the header cannot be
    /// defeated by a stale `fresh=0` further down the chain.
    ///
    /// `X-Hopper-Fresh` is named for hopper's own escape hatch rather than for
    /// scan, because it is the same request travelling: a caller sets it once
    /// and every hop that can answer from something cheaper stands down. It
    /// accepts the same spellings the query parameter does — hopper itself
    /// only reads `1`, and accepting a superset here costs nothing and
    /// surprises nobody.
    fn with_fresh_header(mut self, headers: &HeaderMap) -> Self {
        self.fresh = self.fresh
            || headers
                .get("x-hopper-fresh")
                .and_then(|v| v.to_str().ok())
                .is_some_and(affirmative);
        self
    }

    fn parse(raw: Option<&str>) -> Self {
        let mut q = Self::default();
        for (key, value) in form_urlencoded::parse(raw.unwrap_or("").as_bytes()) {
            match key.as_ref() {
                "purl" => q.purl.push(value.into_owned()),
                "url" => q.url.push(value.into_owned()),
                "sha256" => q.sha256.push(value.into_owned()),
                "false_positive_budget" => match value.parse::<u16>() {
                    Ok(n) => q.false_positive_budget = Some(n),
                    Err(_) => q.bad_budget = Some(value.into_owned()),
                },
                // Only an affirmative spelling forces a run, because the
                // expensive reading of an ambiguous value is the one that
                // burns an analysis slot. `fresh` follows the same rule, for
                // the same reason: an ambiguous value must not silently change
                // which layer answers.
                "force" => q.force = affirmative(value.as_ref()),
                "fresh" => q.fresh = affirmative(value.as_ref()),
                // This is intentionally stricter than the older boolean
                // spellings: `refresh=1` is the one public wire contract.
                "refresh" => q.refresh = value.as_ref() == "1",
                "path" => q.path.push(value.into_owned()),
                "follow" => q.follow.push(value.into_owned()),
                // Unknown parameters are ignored, so a caller can carry their
                // own tracing keys through without us rejecting the request.
                _ => {}
            }
        }
        q
    }

    /// The caller's budget, or a 400 naming what they sent instead.
    fn budget(&self, server_level: Option<u16>) -> Result<u16, ApiError> {
        if let Some(bad) = self.bad_budget.as_deref() {
            return Err(ApiError::bad_request(
                "invalid_false_positive_budget",
                format!(
                    "false_positive_budget must be a whole number from 0 to 65535, not {bad:?}."
                ),
            ));
        }
        Ok(self
            .false_positive_budget
            .unwrap_or_else(|| decision::default_budget(server_level)))
    }
}

/// Resolve a request's follow selection: the configured policy, restricted to
/// what the request names. A request may follow less than the operator allows;
/// asking for more is refused rather than quietly trimmed, so a caller never
/// mistakes a shallower answer for the one it asked for.
fn v1_follow_policy(
    q: &V1Query,
    configured: crate::fetch::FetchPolicy,
) -> Result<crate::fetch::FetchPolicy, ApiError> {
    if q.follow.is_empty() {
        return Ok(configured);
    }
    let selected = crate::fetch::FetchPolicy::parse_follow(&q.follow.join(","))
        .map_err(|message| ApiError::bad_request("invalid_follow_policy", message))?;
    configured.restricted_to(selected).ok_or_else(|| {
        ApiError::bad_request(
            "follow_not_allowed",
            format!(
                "This server follows {}; follow= may only narrow that.",
                configured
                    .follow_name()
                    .unwrap_or_else(|| "a custom selection".to_owned())
            ),
        )
    })
}

/// Longest an upload may go without delivering a byte. An idle bound, not a
/// total one: a large artifact on a slow link still arrives, but a client
/// dripping a body (or stalling mid-upload) cannot hold its buffer and its
/// busy mark open indefinitely.
const BODY_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

/// How many packages one URL may name.
///
/// A PURL runs about fifty characters encoded, so fifty of them sits well
/// inside every intermediary's URL limit with room to spare. Past this the
/// answer is POST, and the error says so rather than leaving it to be
/// discovered by a truncated query string.
const V1_MAX_KEYS: usize = 50;

/// How many of one `/v1/lookup`'s packages ask the corpus at once.
const V1_LOOKUP_CONCURRENCY: usize = 8;

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

/// One `/v1/analyze` request, its query validated before any body is read.
struct V1Analyze {
    request_id: u64,
    started: Instant,
    q: V1Query,
    budget: u16,
    follow: RequestFollow,
    /// The package as the caller spelled it, and its normalized key.
    purl: Option<(String, String)>,
    /// The exact URL, validated.
    url: Option<String>,
    /// A file on this host, absolute; the server reads it only if it lies
    /// under `--allow-local-paths`.
    path: Option<std::path::PathBuf>,
}

impl V1Analyze {
    /// Validate everything the query says. A refusal is a finished response:
    /// an unparseable PURL names what the caller sent on the access line.
    fn parse(state: &AppState, request_id: u64, q: V1Query) -> Result<Self, Box<Response>> {
        let started = Instant::now();
        let reject =
            |code, message: &'static str| Box::new(ApiError::bad_request(code, message).response());
        let locators = usize::from(!q.purl.is_empty())
            + usize::from(!q.url.is_empty())
            + usize::from(!q.sha256.is_empty())
            + usize::from(!q.path.is_empty());
        if locators > 1 || q.sha256.len() > 1 || q.path.len() > 1 {
            return Err(reject(
                "multiple_locators",
                "Use one of ?purl=, ?url=, ?sha256= or ?path=.",
            ));
        }
        // Refused before anything looks at the path, so a server that does not
        // read local files never touches the filesystem on a caller's behalf.
        let path = match q.path.first() {
            None => None,
            Some(_) if state.config.local_paths.is_empty() => {
                return Err(Box::new(
                    ApiError::new(
                        StatusCode::FORBIDDEN,
                        "local_paths_disabled",
                        "This server does not read local files (see --allow-local-paths).",
                    )
                    .response(),
                ));
            }
            Some(raw) => {
                let path = std::path::PathBuf::from(raw);
                if !path.is_absolute() || raw.contains('\0') {
                    return Err(reject(
                        "invalid_path",
                        "path must be an absolute file path.",
                    ));
                }
                Some(path)
            }
        };
        if let Some(raw_sha) = q.sha256.first()
            && burton::parse_sha256_hex(raw_sha).is_none()
        {
            return Err(reject(
                "invalid_sha256",
                "sha256 must be 64 hexadecimal characters.",
            ));
        }
        if !q.sha256.is_empty() && !q.refresh {
            return Err(reject(
                "refresh_required",
                "?sha256= on /v1/analyze requires refresh=1.",
            ));
        }
        if q.refresh && q.sha256.is_empty() {
            return Err(if q.purl.is_empty() && q.url.is_empty() {
                reject("missing_sha256", "refresh=1 requires ?sha256=.")
            } else {
                reject(
                    "multiple_locators",
                    "refresh=1 names exactly one artifact with ?sha256=.",
                )
            });
        }
        // A budget that is not a number is the caller's mistake whichever way
        // they named the artifact, and answering it only after the body has
        // been read would make the same request a 400 or a 503 depending on
        // whether the model happened to be loaded.
        let budget = q
            .budget(state.config.level)
            .map_err(|e| Box::new(e.response()))?;
        // Everything this server analyzes is filed, whatever policy produced
        // it, and a narrower request may be answered from what a wider one
        // filed: a shallower verdict kept is worth more than none. The cost is
        // that the index and hopper hold one verdict per artifact, whichever
        // policy filed it first; holding one per policy is a data-model
        // change, not a flag.
        let follow = RequestFollow {
            policy: v1_follow_policy(&q, state.config.fetch).map_err(|e| Box::new(e.response()))?,
            refresh: q.refresh,
        };
        let url = match q.url.as_slice() {
            [] => None,
            [url] => {
                let url = url.trim();
                if !valid_http_url(url) {
                    return Err(reject(
                        "invalid_url",
                        "url must be an absolute http or https URL.",
                    ));
                }
                Some(url.to_owned())
            }
            _ => {
                return Err(reject(
                    "too_many_packages",
                    "Only one exact url may be analyzed per request.",
                ));
            }
        };
        // Kept apart on purpose: the normalized key is what everything
        // downstream is stored and looked up by, the caller's spelling is what
        // the answer is spelled with. See `V1Decision::asked_about`.
        let purl = match q.purl.first() {
            None => None,
            Some(asked) => match normalize_pkg_purl(asked) {
                Ok(purl) => Some((asked.clone(), purl)),
                Err(message) => {
                    return Err(Box::new(with_subject(
                        ApiError::bad_request("invalid_purl", message).response(),
                        Subject::purl(asked, None),
                    )));
                }
            },
        };
        Ok(Self {
            request_id,
            started,
            q,
            budget,
            follow,
            purl,
            url,
            path,
        })
    }

    fn elapsed_ms(&self) -> u64 {
        crate::duration_ms(self.started.elapsed())
    }
}

/// POST /v1/analyze — analyze an artifact and answer with a decision.
///
/// The whole point of this streamed answer is that it survives being
/// slow. A proxy between us and the caller gives up on a silent connection —
/// after as little as 125 seconds — and tears it down, which costs the
/// caller an analysis that in fact completed: the server finishes,
/// files its verdict, and answers the next asker in milliseconds, but the reply
/// to *this* request had nowhere to go.
///
/// So the answer starts before it is known. Nothing is sent for the first
/// [`V1_ANALYZE_GRACE`], because most analyses finish inside it and deserve an
/// ordinary response with an ordinary status code. Past that the response
/// begins — headers, then a progress frame after [`V1_PROGRESS_FIRST`] and
/// every [`V1_PROGRESS_EVERY`] — and the connection stops being idle, so
/// nothing between here and the caller has cause to cut it. The decision
/// follows whenever the analysis lands.
///
/// A locator can name an artifact, and the artifact itself is another way in:
/// a caller holding bytes nobody has published has nothing to locate them by.
/// Which one is meant is decided by what arrived, not by a header the caller
/// has to remember: bytes are an artifact, and no bytes means the package
/// named in the query.
pub(super) async fn v1_analyze(
    State(state): State<Arc<AppState>>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    raw: axum::extract::RawQuery,
    headers: HeaderMap,
    body: axum::body::Body,
) -> Response {
    let q = V1Query::parse(raw.0.as_deref()).with_fresh_header(&headers);
    let req = match V1Analyze::parse(&state, request_id, q) {
        Ok(req) => req,
        Err(rejected) => return *rejected,
    };
    if let Some(path) = req.path.as_deref() {
        if !body.is_end_stream() {
            return ApiError::bad_request(
                "path_with_body",
                "Use either ?path= or an uploaded artifact, not both.",
            )
            .response();
        }
        return analyze_path(&state, &req, path).await;
    }

    // Memory is checked before the body is read: an artifact of up to
    // `--max-size-mb` is read only by a server with room for it. Full admission
    // waits until it has to analyze, so a verdict already held for these bytes
    // is answered even by a server whose startup failed.
    if !body.is_end_stream()
        && let Err(refusal) = state.check_memory().await
    {
        return refusal.response();
    }
    let (bytes, sha) = match read_body(body, state.config.max_body_size).await {
        Ok(read) => read,
        Err(refusal) => {
            tracing::warn!(
                id = request_id,
                status = refusal.status.as_u16(),
                "request body refused"
            );
            return refusal.response();
        }
    };
    if !bytes.is_empty() {
        if req.q.refresh {
            return ApiError::bad_request(
                "sha256_with_body",
                "Use either ?sha256=&refresh=1 or an uploaded artifact, not both.",
            )
            .response();
        }
        if req.url.is_some() {
            return ApiError::bad_request(
                "url_with_body",
                "Use either an exact url or an uploaded artifact, not both.",
            )
            .response();
        }
        return analyze_bytes(&state, &req, &headers, bytes, sha).await;
    }
    if req.q.refresh {
        return refresh(&state, &req, headers).await;
    }
    if let Some(url) = req.url.clone() {
        return analyze_url(&state, &req, url).await;
    }
    if let Some((asked, purl)) = req.purl.clone() {
        return analyze_purl(&state, &req, asked, purl).await;
    }
    ApiError::bad_request(
        "missing_package",
        "Name an artifact with ?purl= or ?url=, or send it as the body.",
    )
    .response()
}

/// Read a request body of at most `limit` bytes, hashing it as it arrives so
/// the digest costs no second pass over the bytes, and none of it at once.
async fn read_body(
    body: axum::body::Body,
    limit: usize,
) -> Result<(bytes::Bytes, String), ApiError> {
    let mut stream = body.into_data_stream();
    let mut digest = Sha256::new();
    let mut buf = bytes::BytesMut::new();
    loop {
        let next = std::future::poll_fn(|cx| {
            futures_core::Stream::poll_next(std::pin::Pin::new(&mut stream), cx)
        });
        let Ok(next) = tokio::time::timeout(BODY_IDLE_TIMEOUT, next).await else {
            return Err(ApiError::new(
                StatusCode::REQUEST_TIMEOUT,
                "body_timeout",
                "The request body stalled.",
            ));
        };
        let Some(chunk) = next else {
            break;
        };
        let chunk = chunk.map_err(|e| {
            tracing::warn!(error = %e, "request body could not be read");
            ApiError::bad_request("unreadable_body", "The request body could not be read.")
        })?;
        if buf.len().saturating_add(chunk.len()) > limit {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "artifact_too_large",
                format!("The artifact exceeds the {limit} byte limit."),
            ));
        }
        digest.update(&chunk);
        buf.extend_from_slice(&chunk);
    }
    Ok((buf.freeze(), format!("{:x}", digest.finalize())))
}

/// Analyze the bytes a caller sent, rather than a package they named.
///
/// The digest is the identity, so two callers uploading the same artifact share
/// one analysis exactly as two callers naming one PURL do. `?purl=` may still
/// accompany the bytes: scan grafts the registry provenance onto the report,
/// and it is echoed in each finding's `pkg`.
async fn analyze_bytes(
    state: &Arc<AppState>,
    req: &V1Analyze,
    headers: &HeaderMap,
    bytes: bytes::Bytes,
    sha: String,
) -> Response {
    let request_id = req.request_id;
    // The name only decides how cleave types the bytes, so a caller that sends
    // none still gets an analysis — of an artifact typed by content rather than
    // by extension.
    let filename = headers
        .get("x-filename")
        .and_then(|v| v.to_str().ok())
        .map(sanitize_upload_filename)
        // `""` and `.` name the staging directory itself: staging would fail,
        // and with it every request sharing this flight.
        .filter(|name| !name.is_empty() && name != ".")
        .unwrap_or_else(|| format!("upload-{request_id}"));
    let asked = req.purl.as_ref().map(|(asked, _)| asked.as_str());
    let purl = req.purl.as_ref().map(|(_, purl)| purl.clone());

    // Already answered? The digest is in hand before any temp file exists, so
    // this costs one index probe against the very artifact the caller sent;
    // without it, re-uploading something this server has already analyzed
    // pays for the whole analysis again.
    //
    // Resolved by digest and by digest ALONE. The PURL must not reach the
    // resolver here, and this is not a style preference — passing it would be
    // a false negative with a CVE's shape. Hopper answers a `?sha256=…&purl=…`
    // query on either key, so an upload of arbitrary bytes carrying `?purl=` of
    // a package the corpus knows would come back with *that package's*
    // verdict: 25 bytes of text sent as `?purl=pkg:npm/chalk@5.3.0` would be
    // answered `allow` under chalk's digest without the bytes ever being
    // looked at. Anything can be laundered through a
    // reputable coordinate that way, which is precisely the attack this route
    // exists to catch. What is asked about is the digest, which is the only
    // thing an upload actually names; the caller's PURL is grafted back on
    // afterwards as provenance, which is all it ever was.
    //
    // Only an answer short-circuits. `unanalyzed` means nobody has analyzed
    // these bytes, which is why the caller sent them, and `unavailable` means
    // we could not find out — turning either into an answer would report on
    // work never done.
    if !req.q.force && !req.q.refresh {
        let key = Key {
            sha: Some(sha.clone()),
            ..Key::default()
        };
        let (decided, source) = decide(state, &key, req.budget, req.q.fresh).await;
        let decided = decided.asked_about(asked);
        if decided.is_answerable() {
            tracing::info!(
                id = request_id,
                sha256 = %sha,
                size_bytes = bytes.len(),
                ms = req.elapsed_ms(),
                kind = if decided.is_verdict() { "verdict" } else { "derived" },
                "--> POST /v1/analyze (bytes; answered from what we already knew; no slot spent)"
            );
            // A local verdict is not enough to satisfy a persisted upload: the
            // original analysis may have lost its request-scoped bytes before
            // the asynchronous uploader read them. Re-offer them, so a cached
            // answer can repair a missing hopper row without another analysis.
            repair(state, request_id, &filename, &sha, bytes);
            let mut resp = answered(decided, source, req.elapsed_ms());
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

    if let Err(refusal) = state.admit_request(request_id).await {
        return refused(req, &refusal);
    }

    let attachment = state.flights.join(FlightKey::new(
        Artifact::Sha(sha.clone()),
        req.follow.policy.selection_bits(),
        state.config.fetch.selection_bits(),
    ));
    if attachment.leads() {
        tracing::info!(id = request_id, sha256 = %sha, size_bytes = bytes.len(), filename = %filename, "--> POST /v1/analyze (bytes)");
        analyze::lead(
            state,
            request_id,
            attachment.flight(),
            Job::Bytes { filename, bytes },
            req.follow,
        );
    } else {
        tracing::info!(id = request_id, sha256 = %sha, "--> POST /v1/analyze (bytes; joined a run already in flight)");
    }
    // The digest labels the request in logs; it is never the locator.
    let named = Named {
        subject: purl.clone().unwrap_or(sha),
        key: purl,
        asked: asked.map(str::to_owned),
        is_url: false,
    };
    answer(req, attachment, named).await
}

/// Analyze a file on this host, as an upload of the bytes read from it.
async fn analyze_path(state: &Arc<AppState>, req: &V1Analyze, path: &std::path::Path) -> Response {
    if let Err(refusal) = state.check_memory().await {
        return refusal.response();
    }
    let request_id = req.request_id;
    let allowed = state.config.local_paths.clone();
    let limit = state.config.max_body_size;
    let owned = path.to_owned();
    let asked = path.to_string_lossy();
    let file = match tokio::task::spawn_blocking(move || {
        super::local::read(&owned, &allowed, limit)
    })
    .await
    {
        Ok(Ok(file)) => file,
        Ok(Err(refusal)) => return with_subject(refusal.response(), Subject::path(&asked, None)),
        Err(error) => {
            tracing::error!(id = request_id, %error, "reading a local path panicked");
            return ApiError::internal().response();
        }
    };
    tracing::info!(id = request_id, path = ?path, sha256 = %file.sha256, size_bytes = file.bytes.len(), "--> POST /v1/analyze (path)");
    let mut headers = HeaderMap::new();
    if let Ok(name) = HeaderValue::from_str(&file.filename) {
        headers.insert("x-filename", name);
    }
    let subject = Subject::path(&asked, Some(&file.sha256));
    with_subject(
        analyze_bytes(state, req, &headers, file.bytes, file.sha256).await,
        subject,
    )
}

/// Re-offer bytes already analyzed to hopper, off the reactor and at most
/// [`super::MAX_BUFFERED`] at once: each holds the whole upload until it is
/// staged, and a repeat past the bound is skipped — the next repeat of the
/// same artifact repairs it instead.
fn repair(state: &AppState, request_id: u64, filename: &str, sha: &str, bytes: bytes::Bytes) {
    let Some(uploader) = state.uploader.clone() else {
        return;
    };
    let Ok(permit) = Arc::clone(&state.buffered).try_acquire_owned() else {
        tracing::debug!(id = request_id, sha256 = %sha, "repair skipped: enough already in flight");
        return;
    };
    let (filename, sha) = (filename.to_owned(), sha.to_owned());
    state.tasks.spawn(async move {
        let _ = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let Some(artifact) = crate::engine::collect_upload_artifacts(
                std::path::Path::new(&filename),
                &sha,
                bytes.len() as u64,
                crate::engine::upload_collector(),
                None,
                None,
            )
            .into_iter()
            .next() else {
                return;
            };
            if let Err(error) = uploader.submit_artifact_bytes_durable(artifact, &bytes) {
                tracing::error!(
                    id = request_id,
                    sha256 = %sha,
                    %error,
                    "upload: could not stage cached artifact for hopper"
                );
            }
        })
        .await;
    });
}

/// `?sha256=…&refresh=1`: reconcile a digest with hopper, analyzing it here
/// when hopper's verdict is missing or from other traits.
async fn refresh(state: &Arc<AppState>, req: &V1Analyze, mut headers: HeaderMap) -> Response {
    let request_id = req.request_id;
    // Trimmed as validation trimmed it, so the digest matches the index and
    // hopper's keys.
    let Some(sha) = req
        .q
        .sha256
        .first()
        .map(|sha| sha.trim().to_ascii_lowercase())
    else {
        return ApiError::bad_request("missing_sha256", "refresh=1 requires ?sha256=.").response();
    };
    // The local index is already namespaced by the currently loaded traits
    // version. It is the cheapest and most reliable answer, and a refresh must
    // not turn a valid Scan-local result into a Hopper dependency merely to
    // prove what Scan already knows.
    let key = sha.clone();
    if let Some(verdict) = index_query(move |index| index.get_sha(&key))
        .await
        .flatten()
    {
        let mut resp = answered(
            V1Decision::stored(&verdict, None, req.budget),
            "scan:index",
            req.elapsed_ms(),
        );
        resp.extensions_mut().insert(Subject::sha256(&sha));
        tracing::info!(id = request_id, sha256 = %sha, "refresh answered from Scan's current local index");
        return resp;
    }

    let Some(corpus) = state.corpus.as_ref() else {
        return ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "hopper_unavailable",
            "refresh requires a configured Hopper corpus.",
        )
        .response();
    };

    let (reached, source) = corpus
        .known_with_source(Some(&sha), None, /* fresh = */ true)
        .await;
    if let Reached::Record(record) = reached {
        // Reads the installed traits on first use, so off the reactor.
        let local = tokio::task::spawn_blocking(crate::corpus_precheck::local_traits)
            .await
            .ok()
            .flatten();
        if matching_traits_version(record.traits_version.as_deref(), local) {
            let decided = V1Decision::corpus(&record, Some(&sha), None, req.budget);
            let mut resp = answered(decided, corpus_source(source), req.elapsed_ms());
            insert_follow(
                resp.headers_mut(),
                req.follow.policy.follow_name().as_deref(),
            );
            resp.extensions_mut().insert(Subject::sha256(&sha));
            tracing::info!(id = request_id, sha256 = %sha, traits_version = ?record.traits_version, "refresh answered from Hopper at the current traits version");
            return resp;
        }
    }

    // The sample is held in memory until the analysis takes it, so the memory
    // check and the bound on buffered artifacts come before the download.
    let Ok(_buffered) = Arc::clone(&state.buffered).try_acquire_owned() else {
        return ApiError::at_capacity("Too many artifacts downloading", None, Some(5)).response();
    };
    if let Err(refusal) = state.admit_request(request_id).await {
        return refused(req, &refusal);
    }
    let sample = match corpus.sample(&sha, state.config.max_body_size).await {
        Ok(sample) => sample,
        Err(message) => {
            // The detail names internal addresses; it stays in the log.
            tracing::warn!(id = request_id, sha256 = %sha, error = %message, "refresh could not fetch sample bytes from Hopper");
            return ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "sample_unavailable",
                "Hopper could not serve the sample.",
            )
            .response();
        }
    };
    if sample.sha256 != sha {
        tracing::error!(id = request_id, sha256 = %sha, fetched_sha256 = %sample.sha256, "Hopper returned bytes with the wrong digest");
        return ApiError::new(
            StatusCode::BAD_GATEWAY,
            "sample_digest_mismatch",
            "Hopper returned bytes that do not match the requested SHA-256.",
        )
        .response();
    }
    let short_sha: String = sha.chars().take(12).collect();
    if let Ok(value) = HeaderValue::from_str(&format!("refresh-{short_sha}")) {
        headers.insert("x-filename", value);
    }
    tracing::info!(id = request_id, sha256 = %sha, size_bytes = sample.bytes.len(), "Hopper verdict missing or stale; analyzing fetched bytes");
    analyze_bytes(state, req, &headers, sample.bytes, sample.sha256).await
}

/// Analyze an exact URL, fetched verbatim.
async fn analyze_url(state: &Arc<AppState>, req: &V1Analyze, url: String) -> Response {
    let request_id = req.request_id;
    if let Err(refusal) = state.admit_request(request_id).await {
        return refused(req, &refusal);
    }
    let attachment = state.flights.join(FlightKey::new(
        Artifact::Url(url.clone()),
        req.follow.policy.selection_bits(),
        state.config.fetch.selection_bits(),
    ));
    if attachment.leads() {
        tracing::info!(id = request_id, url = %url, "--> POST /v1/analyze");
        analyze::lead(
            state,
            request_id,
            attachment.flight(),
            Job::Url(url.clone()),
            req.follow,
        );
    } else {
        tracing::info!(id = request_id, url = %url, "--> POST /v1/analyze (joined a run already in flight)");
    }
    let named = Named {
        key: Some(url.clone()),
        asked: Some(url.clone()),
        subject: url,
        is_url: true,
    };
    answer(req, attachment, named).await
}

/// Analyze a package named by PURL, unless what we already know answers it.
async fn analyze_purl(
    state: &Arc<AppState>,
    req: &V1Analyze,
    asked: String,
    purl: String,
) -> Response {
    let request_id = req.request_id;

    // Already answered?
    //
    // This is the expensive door into the question `/v1/lookup` answers
    // cheaply. Without asking first, every caller would pay a full download-
    // and-classify — minutes for a large crate — for an artifact this server
    // already holds a verdict for, while `/v1/lookup` answers the same
    // question from the index in a single hop.
    //
    // Resolved exactly the way the lookup resolves it — same normalization,
    // same index-then-corpus order, same budget — because two routes answering
    // one question differently is worse than either answer alone. Only an
    // answer short-circuits: `unanalyzed` is the whole reason the caller is
    // here, and `unavailable` means we could not find out — turning that into
    // a refusal to work would make a corpus outage look like an answer.
    if !req.q.force && !req.q.refresh {
        let key = Key {
            purl: Some(purl.clone()),
            ..Key::default()
        };
        let (decided, source) = decide(state, &key, req.budget, req.q.fresh).await;
        let decided = decided.asked_about(Some(&asked));
        if decided.is_answerable() {
            tracing::info!(
                id = request_id,
                purl = %purl,
                ms = req.elapsed_ms(),
                kind = if decided.is_verdict() { "verdict" } else { "derived" },
                "--> POST /v1/analyze (answered from what we already knew; no slot spent)"
            );
            let mut resp = answered(decided, source, req.elapsed_ms());
            resp.extensions_mut().insert(Subject::purl(&purl, None));
            return resp;
        }
        tracing::info!(id = request_id, purl = %purl, "no verdict held; analyzing");
    }

    // Admitted only once it has to analyze: there is no body to protect, and
    // an answer already held costs nothing to serve, even from a server whose
    // startup failed.
    if let Err(refusal) = state.admit_request(request_id).await {
        return refused(req, &refusal);
    }

    let attachment = state.flights.join(FlightKey::new(
        Artifact::Purl(purl.clone()),
        req.follow.policy.selection_bits(),
        state.config.fetch.selection_bits(),
    ));
    if attachment.leads() {
        tracing::info!(id = request_id, purl = %purl, "--> POST /v1/analyze");
        analyze::lead(
            state,
            request_id,
            attachment.flight(),
            Job::Purl(purl.clone()),
            req.follow,
        );
    } else {
        tracing::info!(id = request_id, purl = %purl, "--> POST /v1/analyze (joined a run already in flight)");
    }
    // Named, not analyzed from bytes, so there is always a coordinate: the
    // normalized one keys everything, and `asked` is what the caller typed.
    let named = Named {
        key: Some(purl.clone()),
        asked: Some(asked),
        subject: purl,
        is_url: false,
    };
    answer(req, attachment, named).await
}

/// `X-Scan-Follow`: the follow policy a decision was made under, so a caller
/// can file it, and a retry need not infer which policy was in play.
fn insert_follow(headers: &mut HeaderMap, follow: Option<&str>) {
    if let Some(value) = follow.and_then(|name| HeaderValue::from_str(name).ok()) {
        headers.insert("X-Scan-Follow", value);
    }
}

/// A request refused admission, named with the follow policy it would have
/// run under — as every other analyze answer is, refusals included.
fn refused(req: &V1Analyze, refusal: &ApiError) -> Response {
    let mut resp = refusal.response();
    insert_follow(
        resp.headers_mut(),
        req.follow.policy.follow_name().as_deref(),
    );
    resp
}

/// A decision answered without an analysis.
fn answered(decided: V1Decision, source: &'static str, elapsed_ms: u64) -> Response {
    let mut resp = Json(decided).into_response();
    resp.headers_mut().insert("X-Total-Ms", elapsed_ms.into());
    resp.headers_mut()
        .insert("X-Scan-Source", HeaderValue::from_static(source));
    resp
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

impl Named {
    /// The field a frame names the artifact in.
    fn locator_field(&self) -> &'static str {
        if self.is_url {
            "url"
        } else if self.asked.is_some() {
            "purl"
        } else {
            "sha256"
        }
    }

    /// The artifact as a frame names it: the caller's spelling, or the digest
    /// for an upload with no coordinate.
    fn locator(&self) -> &str {
        self.asked.as_deref().unwrap_or(&self.subject)
    }

    /// The access-line subject: an upload with no coordinate is named by its
    /// digest, never logged as a `purl`.
    fn access_subject(&self) -> Subject {
        match (self.is_url, self.key.as_deref()) {
            (true, _) => Subject::url(&self.subject),
            (false, Some(purl)) => Subject::purl(purl, None),
            (false, None) => Subject::sha256(&self.subject),
        }
    }

    /// A finished analysis, as a decision about this artifact.
    ///
    /// An upload has no locator, and the digest is not one: passing it here
    /// would put a sha256 in the `purl` field, where `/v1/lookup` reports null
    /// for the same artifact. The verdict is stored under the normalized key;
    /// only the answer going back out is spelled the caller's way.
    fn decision(&self, result: &crate::engine::ScanResult, budget: u16) -> V1Decision {
        let key = self.key.as_deref();
        if self.is_url {
            let verdict = crate::lookup::Verdict::from_scan(result, None);
            V1Decision::stored(&verdict, None, budget).with_url(self.asked.as_deref())
        } else {
            let verdict = crate::lookup::Verdict::from_scan(result, key);
            V1Decision::stored(&verdict, key, budget).asked_about(self.asked.as_deref())
        }
    }
}

/// The per-request framing an answer renders with: the budget the decision
/// cites, the clock its `elapsed_ms` fields count from, the follow policy
/// named on its headers.
struct V1Framing {
    request_id: u64,
    budget: u16,
    started: Instant,
    follow: Option<String>,
}

/// Answer within the grace window if the analysis lands inside it, otherwise
/// as a stream.
async fn answer(req: &V1Analyze, attachment: Attachment, named: Named) -> Response {
    let framing = V1Framing {
        request_id: req.request_id,
        budget: req.budget,
        started: req.started,
        follow: req.follow.policy.follow_name(),
    };
    // Inside the grace window the ordinary response still applies, which is
    // what keeps `429 At capacity` a real 429 the router can act on rather than
    // a decision buried in a 200 body.
    let waited = tokio::time::timeout(V1_ANALYZE_GRACE, attachment.flight().wait()).await;
    match waited {
        Ok(outcome) => outcome_response(&outcome, &named, &framing, !attachment.leads()),
        Err(_) => {
            tracing::info!(id = req.request_id, subject = %named.subject, "answering as a stream");
            streamed(attachment, named, framing)
        }
    }
}

/// A finished analysis, as a decision.
fn outcome_response(
    outcome: &Outcome,
    named: &Named,
    framing: &V1Framing,
    shared: bool,
) -> Response {
    let mut resp = match outcome {
        Outcome::Report(result) => {
            let mut resp = Json(named.decision(result, framing.budget)).into_response();
            resp.headers_mut().insert(
                "X-Total-Ms",
                crate::duration_ms(framing.started.elapsed()).into(),
            );
            // Whether this answer cost an analysis. The route is the same
            // either way, but a run served from the analysis cache did no work
            // and one that reached the pipeline did — and a caller measuring
            // what its fleet spends cannot tell those apart from the route.
            resp.headers_mut().insert(
                "X-Scan-Source",
                HeaderValue::from_static(if result.analysis_cached {
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
        Outcome::Failed(refusal) => refusal.response(),
    };
    // Which question this answer answers. The caller resolved a policy before
    // asking, but only this server knows what it applied on top of its own
    // configuration, and the answer has to be filed under what was measured
    // rather than what was requested. On refusals too: a caller correlating a
    // retry should not have to infer which policy was in play.
    insert_follow(resp.headers_mut(), framing.follow.as_deref());
    // The normalized coordinate, so a caching caller can file one entry per
    // artifact rather than one per spelling.
    //
    // The body is spelled with `asked` on purpose: an answer should come back
    // in the words the question was put in. But a cache keyed on those words
    // holds `v4.4.0+incompatible` and `v4.4.0%2Bincompatible` as two packages
    // and buys the same analysis twice. Only this server knows they are one
    // coordinate, because only this server ran them through the normalizer,
    // so it is this server that has to say so.
    if let Some(key) = named.key.as_deref()
        && let Ok(value) = HeaderValue::from_str(key)
    {
        resp.headers_mut().insert("X-Scan-Purl", value);
    }
    if shared {
        resp.extensions_mut().insert(super::access::Shared);
    }
    resp.extensions_mut().insert(named.access_subject());
    resp
}

/// The `reason` a streamed failure carries, when the status says the artifact
/// itself could not be obtained rather than that this fleet fell over.
///
/// 422 is what [`ApiError::from_analysis`] renders for
/// [`crate::fetch::Unretrievable`]; 413 is an artifact past the size limit.
/// Both are facts about the package that will not change if the caller retries
/// against a healthier server, and both are what poppy files under download
/// failures rather than its error rate. Every other status — 500, 504, 429, a
/// server starting up — is about us, and stays an unqualified outage.
fn unretrievable_reason(status: StatusCode) -> Option<&'static str> {
    match status {
        StatusCode::UNPROCESSABLE_ENTITY => Some("unretrievable"),
        StatusCode::PAYLOAD_TOO_LARGE => Some("too_large"),
        _ => None,
    }
}

/// A progress frame for a run still going. The phase is read off the flight,
/// so a follower reports the run it rides however that run is named.
fn progress_frame(attachment: &Attachment, named: &Named, started: Instant) -> serde_json::Value {
    let mut frame = serde_json::json!({
        "state": "analyzing",
        "elapsed_ms": crate::duration_ms(started.elapsed()),
        "phase": attachment.flight().phase(),
    });
    frame[named.locator_field()] = serde_json::Value::String(named.locator().to_owned());
    frame
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
/// The access line marks the response `streamed`, and the stream logs its own
/// outcome under the same id.
fn streamed(attachment: Attachment, named: Named, framing: V1Framing) -> Response {
    let V1Framing {
        request_id,
        budget,
        started,
        follow,
    } = framing;
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
    let canonical = named.key.clone();
    let access_subject = named.access_subject();
    tokio::spawn(async move {
        let flight = Arc::clone(attachment.flight());
        let mut waiting = std::pin::pin!(flight.wait());
        let mut next = V1_PROGRESS_FIRST;
        let outcome = loop {
            tokio::select! {
                outcome = &mut waiting => break outcome,
                () = tokio::time::sleep(next) => {
                    next = V1_PROGRESS_EVERY;
                    // A caller that has gone away shows up here as a closed
                    // channel, which ends the stream. The analysis keeps going:
                    // it is not this connection's to lose.
                    if !v1_send(&tx, &progress_frame(&attachment, &named, started)).await {
                        tracing::info!(id = request_id, subject = %named.subject, "stream closed by the caller; the analysis continues");
                        return;
                    }
                }
            }
        };
        // Held to here: an attachment dropped early would tell the flight
        // nobody is waiting on this analysis while somebody is.
        drop(attachment);
        let elapsed = crate::duration_ms(started.elapsed());
        let decided = match outcome.as_ref() {
            Outcome::Report(result) => {
                tracing::info!(id = request_id, subject = %named.subject, elapsed_ms = elapsed, "streamed analysis answered");
                named.decision(result, budget)
            }
            Outcome::Failed(refusal) if refusal.status == StatusCode::TOO_MANY_REQUESTS => {
                // Refused after the stream began (a big whale found every slot
                // taken). Not a decision: the stream ends without one, which a
                // router reads as "hand this to a server with room". A decision
                // here would be final and wrong — the package was never looked
                // at.
                tracing::warn!(id = request_id, subject = %named.subject, elapsed_ms = elapsed, "streamed analysis refused; ending without a decision");
                let mut frame = serde_json::json!({
                    "state": "refused",
                    "elapsed_ms": elapsed,
                });
                if let (Some(frame), serde_json::Value::Object(body)) =
                    (frame.as_object_mut(), refusal.body_json())
                {
                    frame.extend(body);
                }
                frame[named.locator_field()] =
                    serde_json::Value::String(named.locator().to_owned());
                v1_send(&tx, &frame).await;
                return;
            }
            Outcome::Failed(refusal) => {
                tracing::warn!(id = request_id, subject = %named.subject, status = refusal.status.as_u16(), elapsed_ms = elapsed, "streamed analysis failed");
                // An artifact nobody can download is not an outage, and the
                // difference is already drawn on the unstreamed path, where an
                // unretrievable package is a 422. Collapsing every failure into
                // a bare `unavailable` would erase it for every streamed caller,
                // which is all of them: poppy asks with `Accept:
                // application/x-ndjson`, so it never sees the 422 and would
                // score a dead package — say, a deleted GitHub repo whose
                // versions proxy.golang.org still lists but cannot serve — as a
                // beamline outage.
                //
                // The decision stays `unavailable`: we still did not answer,
                // and a caller must not read an assessment into it. What
                // changes is that `reason` says whose failure it was, which
                // beamline already relays verbatim.
                let reason = unretrievable_reason(refusal.status);
                if named.is_url {
                    V1Decision::unavailable(None, None)
                        .with_url(named.asked.as_deref())
                        .because(reason)
                } else {
                    V1Decision::unavailable(None, named.key.as_deref())
                        .asked_about(named.asked.as_deref())
                        .because(reason)
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
        HeaderValue::from_static("application/x-ndjson"),
    );
    // Nothing may buffer this: a proxy that holds the bytes back to measure the
    // body defeats the only thing progress frames are for.
    headers.insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("no-store"),
    );
    headers.insert("X-Accel-Buffering", HeaderValue::from_static("no"));
    headers.insert("X-Scan-Source", HeaderValue::from_static("scan:analysis"));
    // See `outcome_response`. These headers go out before the first progress
    // frame, so a caller knows how to file the decision before it arrives.
    insert_follow(headers, follow.as_deref());
    if let Some(key) = canonical.as_deref()
        && let Ok(value) = HeaderValue::from_str(key)
    {
        headers.insert("X-Scan-Purl", value);
    }
    resp.extensions_mut().insert(access_subject);
    resp.extensions_mut().insert(super::access::Streamed);
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
    headers: HeaderMap,
) -> Response {
    let started = Instant::now();
    let q = V1Query::parse(raw.0.as_deref()).with_fresh_header(&headers);
    let response = v1_lookup_inner(&state, &q)
        .await
        .unwrap_or_else(|refusal| refusal.response());
    state
        .jobs
        .lookups
        .record(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
    response
}

/// The values given, trimmed, with blanks dropped.
fn nonblank(values: &[String]) -> Vec<&str> {
    values
        .iter()
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .collect()
}

async fn v1_lookup_inner(state: &Arc<AppState>, q: &V1Query) -> Result<Response, ApiError> {
    let budget = q.budget(state.config.level)?;
    let shas = nonblank(&q.sha256);
    // One digest pairs with a PURL naming the same bytes. Several digests are
    // several artifacts, listed the way several PURLs are.
    let sha = match shas.as_slice() {
        [one] => Some(*one),
        _ => None,
    };
    let purls = nonblank(&q.purl);
    let urls = nonblank(&q.url);

    if !purls.is_empty() && !urls.is_empty() {
        return Err(ApiError::bad_request(
            "multiple_locators",
            "Use ?purl= or ?url=, not both.",
        ));
    }
    if shas.len() > 1 && (!purls.is_empty() || !urls.is_empty()) {
        return Err(ApiError::bad_request(
            "multiple_locators",
            "Several sha256 values name several artifacts; send them without ?purl= or ?url=.",
        ));
    }
    if urls.iter().any(|url| !valid_http_url(url)) {
        return Err(ApiError::bad_request(
            "invalid_url",
            "url must be an absolute http or https URL.",
        ));
    }
    if shas.is_empty() && purls.is_empty() && urls.is_empty() {
        return Err(ApiError::bad_request(
            "missing_package",
            "Name an artifact with ?purl=, ?url=, or ?sha256=.",
        ));
    }
    let named = purls.len().max(urls.len()).max(shas.len());
    if named > V1_MAX_KEYS {
        return Err(ApiError::bad_request(
            "too_many_packages",
            format!("{named} packages exceeds the limit of {V1_MAX_KEYS} per request."),
        ));
    }

    if let Some(&url) = urls.first() {
        if urls.len() > 1 {
            return Err(ApiError::bad_request(
                "too_many_packages",
                "Only one exact url may be looked up per request.",
            ));
        }
        let key = Key::parse(sha, None, Some(url))?;
        let (decided, source) = decide(state, &key, budget, q.fresh).await;
        return Ok(sourced(Json(decided.with_url(Some(url))), source));
    }

    // One package named two ways is one question, so a lone sha256 and a lone
    // purl resolve together rather than as two entries.
    if shas.len() <= 1 && purls.len() <= 1 {
        let purl = purls.first().copied();
        let key = Key::parse(sha, purl, None)?;
        let (decided, source) = decide(state, &key, budget, q.fresh).await;
        return Ok(sourced(Json(decided.asked_about(purl)), source));
    }

    // A key we cannot parse is the caller's mistake and stops the whole call,
    // because answering the rest would hide it.
    let (keys, asked): (Vec<Key>, Vec<Option<&str>>) = if shas.len() > 1 {
        (
            shas.iter()
                .map(|sha| Key::parse(Some(sha), None, None))
                .collect::<Result<_, _>>()?,
            vec![None; shas.len()],
        )
    } else {
        (
            purls
                .iter()
                .map(|purl| Key::parse(None, Some(purl), None))
                .collect::<Result<_, _>>()?,
            purls.iter().copied().map(Some).collect(),
        )
    };
    let decided = decide_many(state, keys, budget, q.fresh).await;
    let mut source = None;
    let mut out = Vec::with_capacity(decided.len());
    for ((decided, row_source), purl) in decided.into_iter().zip(asked) {
        source = Some(match source {
            None => row_source,
            Some(previous) if previous == row_source => previous,
            Some(_) => "scan:analysis",
        });
        out.push(decided.asked_about(purl));
    }
    Ok(sourced(Json(out), source.unwrap_or("scan:analysis")))
}

fn sourced(body: impl IntoResponse, source: &'static str) -> Response {
    let mut resp = body.into_response();
    resp.headers_mut()
        .insert("X-Scan-Source", HeaderValue::from_static(source));
    resp
}

/// One artifact a v1 request names, every key normalized: what the index and
/// the corpus are keyed by.
#[derive(Clone, Debug, Default)]
struct Key {
    sha: Option<String>,
    purl: Option<String>,
    url: Option<String>,
}

impl Key {
    fn parse(sha: Option<&str>, purl: Option<&str>, url: Option<&str>) -> Result<Self, ApiError> {
        let purl = purl
            .map(normalize_pkg_purl)
            .transpose()
            .map_err(|message| ApiError::bad_request("invalid_purl", message))?;
        let url = match url {
            Some(url) if !valid_http_url(url) => {
                return Err(ApiError::bad_request(
                    "invalid_url",
                    "url must be an absolute http or https URL.",
                ));
            }
            Some(url) => Some(url.trim().to_owned()),
            None => None,
        };
        let sha = match sha {
            Some(sha) if burton::parse_sha256_hex(sha).is_none() => {
                return Err(ApiError::bad_request(
                    "invalid_sha256",
                    "sha256 must be 64 hexadecimal characters.",
                ));
            }
            Some(sha) => Some(sha.to_ascii_lowercase()),
            None => None,
        };
        Ok(Self { sha, purl, url })
    }

    /// A versioned PURL names every payload of that release. Only the corpus
    /// can choose across the fleet; a local alias covers just one artifact.
    fn needs_corpus(&self, configured: bool) -> bool {
        configured
            && self.sha.is_none()
            && self.purl.as_deref().is_some_and(|purl| {
                fletch::purl::Purl::parse(purl).is_ok_and(|purl| purl.version().is_some())
            })
    }

    fn unavailable(&self) -> (V1Decision, &'static str) {
        (
            V1Decision::unavailable(self.sha.as_deref(), self.purl.as_deref())
                .with_url(self.url.as_deref()),
            "none",
        )
    }

    fn stored(&self, verdict: &crate::lookup::Verdict, budget: u16) -> (V1Decision, &'static str) {
        (
            V1Decision::stored(verdict, self.purl.as_deref(), budget).with_url(self.url.as_deref()),
            // Held, not produced. Reporting `scan:analysis` — the value a
            // fresh run reports — would make an instant index hit and a
            // ninety-second analysis indistinguishable to anything counting
            // cache layers.
            "scan:index",
        )
    }
}

/// The verdict this server holds for `key`. The digest is the identity: a
/// PURL's verdict is accepted only when it describes the same bytes, because a
/// release whose digest has moved is an answer about a different artifact than
/// the one asked about. Blocking: the index is files.
fn held(index: &crate::lookup::Index, key: &Key, corpus: bool) -> Option<crate::lookup::Verdict> {
    if key.needs_corpus(corpus) {
        return None;
    }
    let by_purl = || key.purl.as_deref().and_then(|p| index.get_purl(p));
    match key.sha.as_deref() {
        Some(sha) => pick_verdict(index.get_sha(sha), by_purl, sha),
        None => by_purl(),
    }
}

/// The decision for one artifact, in this server's own vocabulary.
///
/// A missing index is not an empty one. Reporting `unanalyzed` then would tell
/// the caller nobody has analyzed this package, when what is true is that we
/// cannot say — and those two carry different policies at the other end. That
/// is the whole reason `unavailable` is a separate value.
async fn decide(
    state: &AppState,
    key: &Key,
    budget: u16,
    fresh: bool,
) -> (V1Decision, &'static str) {
    if key.needs_corpus(state.corpus.is_some()) {
        return decide_unheld(state, key, budget, fresh).await;
    }
    let lookup = key.clone();
    match index_query(move |index| held(index, &lookup, false)).await {
        None => key.unavailable(),
        Some(Some(verdict)) => key.stored(&verdict, budget),
        Some(None) => decide_unheld(state, key, budget, fresh).await,
    }
}

/// Decisions for many artifacts: one trip to the blocking pool for every index
/// read, then the corpus asked for the rest, a bounded number at a time.
async fn decide_many(
    state: &Arc<AppState>,
    keys: Vec<Key>,
    budget: u16,
    fresh: bool,
) -> Vec<(V1Decision, &'static str)> {
    // A row whose task fails stays `unavailable`: we could not find out.
    let mut out: Vec<_> = keys.iter().map(Key::unavailable).collect();
    let lookup = keys.clone();
    let corpus = state.corpus.is_some();
    let verdicts = index_query(move |index| {
        lookup
            .iter()
            .map(|key| held(index, key, corpus))
            .collect::<Vec<_>>()
    })
    .await;
    let index_available = verdicts.is_some();
    let verdicts = verdicts.unwrap_or_else(|| vec![None; keys.len()]);
    let mut misses = Vec::new();
    for (i, (key, verdict)) in keys.into_iter().zip(verdicts).enumerate() {
        match verdict {
            Some(verdict) => out[i] = key.stored(&verdict, budget),
            None if index_available || key.needs_corpus(corpus) => misses.push((i, key)),
            None => {} // An unavailable local index cannot answer an exact-artifact lookup.
        }
    }

    let mut pending = misses.into_iter();
    let mut running = tokio::task::JoinSet::new();
    let start = |running: &mut tokio::task::JoinSet<_>, (i, key): (usize, Key)| {
        let state = Arc::clone(state);
        running.spawn(async move { (i, decide_unheld(&state, &key, budget, fresh).await) });
    };
    for job in pending.by_ref().take(V1_LOOKUP_CONCURRENCY) {
        start(&mut running, job);
    }
    while let Some(joined) = running.join_next().await {
        match joined {
            Ok((i, decided)) => out[i] = decided,
            Err(e) => {
                tracing::error!(error = %e, "a v1 lookup row panicked; answering unavailable")
            }
        }
        if let Some(job) = pending.next() {
            start(&mut running, job);
        }
    }
    out
}

fn corpus_source(source: Option<corpus::CorpusSource>) -> &'static str {
    match source {
        Some(corpus::CorpusSource::Replica) => "scan:replica",
        Some(corpus::CorpusSource::Primary) => "scan:primary",
        None => "none",
    }
}

/// The decision for an artifact this server's index does not hold: the
/// filters, then the corpus.
async fn decide_unheld(
    state: &AppState,
    key: &Key,
    budget: u16,
    fresh: bool,
) -> (V1Decision, &'static str) {
    let (sha, purl, url) = (key.sha.as_deref(), key.purl.as_deref(), key.url.as_deref());
    // Nothing measured here. Ask the filters before the network: they are the
    // cheapest knowledge in the process, and for a blessed artifact they are
    // the whole answer.
    //
    // Unless the caller asked for the authoritative answer. A filter is
    // membership rebuilt on a schedule, so it is exactly what somebody reading
    // after a write — or checking whether a revocation has landed — needs
    // bypassed. Withholding the decision here disables both bloom paths at
    // once: the fast bless below, and the derived fallback after the corpus.
    //
    // Both keys are evidence about one artifact, so both are supplied and
    // `burton` combines them: the worst claim against either wins, and a
    // blessing needs all of them. A caller who names both is asserting they
    // are the same thing, so a bless on one beside a claim on the other is a
    // contradiction, not a coin flip.
    let bloom = match crate::bloom_repo::global() {
        Some(lk) if !fresh => lk.decide_any(purl, sha.and_then(burton::parse_sha256_hex).as_ref()),
        _ => crate::bloom_repo::Decision::Unknown,
    };

    // A bless answers immediately and does not pay the hopper round trip. The
    // exposure is a bless that has gone stale — but that is bounded by the
    // filter rebuild, because `good` is rebuilt as `good − (bad ∪ sighted)` and
    // the bad channel is the designed revocation path. It is also the bargain
    // the local scan path already takes: `bloom_skip_predicate` skips the
    // download outright on a good hit, without asking anyone.
    if bloom == crate::bloom_repo::Decision::Skip
        && !key.needs_corpus(state.corpus.is_some())
        && let Some(d) = V1Decision::bloom(bloom, sha, purl, budget)
    {
        return (d.with_url(url), "scan:bloom");
    }

    let filters_or_unanalyzed = || {
        V1Decision::bloom(bloom, sha, purl, budget)
            .unwrap_or_else(|| V1Decision::unanalyzed(sha, purl))
            .with_url(url)
    };

    // Not in this server's index. The corpus behind it may still know, and a
    // caller should not have to learn that two services exist in order to get
    // one answer — so ask, rather than reporting an absence that is only ours.
    //
    // Measured beats derived: a filter claim is a floor, and hopper may hold
    // the real level, the real findings and the sentence a person reads. Only
    // when it holds nothing does the filter's own claim stand in.
    let Some(corpus) = state.corpus.as_ref() else {
        return (filters_or_unanalyzed(), "scan:bloom");
    };
    let (reached, source) = corpus
        .known_with_source(sha, purl, /* fresh = */ false)
        .await;
    let decided = match reached {
        Reached::Record(record) => V1Decision::corpus(&record, sha, purl, budget).with_url(url),
        // The corpus holds nothing either. A filter claim is the last thing we
        // know, and answering `unanalyzed` about a digest several operators
        // call malware is a worse answer than saying who says so.
        Reached::Nothing => filters_or_unanalyzed(),
        // The corpus could not answer, so neither can we. Emphatically not
        // `unanalyzed`: that would tell the caller nobody has analyzed this
        // package, which is a claim about the package rather than about us, and
        // the one that lets a gate fail open during an outage.
        Reached::Unreachable => V1Decision::unavailable(sha, purl).with_url(url),
    };
    (decided, corpus_source(source))
}

#[cfg(test)]
mod tests;
