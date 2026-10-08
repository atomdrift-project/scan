//! The `/v1` routes: a decision a firewall can act on, for an artifact named
//! by package URL, exact URL or digest, or sent as bytes.
//!
//! Every error here is the v1 shape, `{"error": {"code", "message"}}` — see
//! [`ApiError::v1`] — whichever layer produced it.

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
use super::decision;
use super::error::ApiError;
use super::flight::{Attachment, FlightKey, Outcome};
use super::handlers::{normalize_pkg_purl, pick_verdict, sanitize_upload_filename, valid_http_url};
use crate::model::Level;

/// Query for `GET /v1/lookup` and `POST /v1/analyze`. `purl` and `url` repeat;
/// `sha256` names one artifact.
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
    /// Bypass outer verdict caches and reconcile a SHA with Hopper. Hopper's
    /// result is reusable only when it was produced by this worker's traits;
    /// otherwise the immutable sample bytes are fetched and analyzed here.
    refresh: bool,
    /// Return the complete scan envelope instead of the compact v1 decision.
    /// A full envelope cannot be reconstructed from the verdict index, so this
    /// also bypasses decision-only fast paths.
    full: bool,
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

/// `X-Hopper-Fresh`, the header spelling of `?fresh=1`.
///
/// Named for hopper's own escape hatch rather than for scan, because it is the
/// same request travelling: a caller sets it once and every hop that can answer
/// from something cheaper stands down. Accepts the same spellings the query
/// parameter does — hopper itself only reads `1`, and accepting a superset here
/// costs nothing and surprises nobody.
fn header_wants_fresh(headers: &HeaderMap) -> bool {
    headers
        .get("x-hopper-fresh")
        .and_then(|v| v.to_str().ok())
        .is_some_and(affirmative)
}

impl V1LookupQuery {
    /// Fold the header alias into the parsed query. Either spelling opts in;
    /// neither can opt back out, so a proxy that adds the header cannot be
    /// defeated by a stale `fresh=0` further down the chain.
    fn with_fresh_header(mut self, headers: &HeaderMap) -> Self {
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
            refresh: false,
            full: false,
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
                "full" => q.full = value.as_ref() == "1",
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

/// Resolve a request's follow selection. The configured policy supplies the
/// default and the operational limits; an explicit request replaces only the
/// selected reference categories after its syntax has been validated.
fn v1_follow_policy(
    q: &V1LookupQuery,
    configured: crate::fetch::FetchPolicy,
) -> Result<crate::fetch::FetchPolicy, ApiError> {
    if q.follow.is_empty() {
        return Ok(configured);
    }
    let selected = crate::fetch::FetchPolicy::parse_follow(&q.follow.join(","))
        .map_err(|message| ApiError::bad_request("invalid_follow_policy", message))?;
    Ok(configured.with_selection(selected))
}

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
    q: V1LookupQuery,
    budget: u16,
    follow: RequestFollow,
    /// The package as the caller spelled it, and its normalized key.
    purl: Option<(String, String)>,
    /// The exact URL, validated.
    url: Option<String>,
}

impl V1Analyze {
    /// Validate everything the query says. A refusal is a finished response:
    /// an unparseable PURL names what the caller sent on the access line.
    fn parse(state: &AppState, request_id: u64, q: V1LookupQuery) -> Result<Self, Box<Response>> {
        let started = Instant::now();
        let reject =
            |code, message: &'static str| Box::new(ApiError::bad_request(code, message).v1());
        let locators = usize::from(!q.purl.is_empty())
            + usize::from(!q.url.is_empty())
            + usize::from(q.sha256.is_some());
        if locators > 1 {
            return Err(reject(
                "multiple_locators",
                "Use ?purl=, ?url=, or ?sha256=, not more than one.",
            ));
        }
        if let Some(raw_sha) = q.sha256.as_deref()
            && burton::parse_sha256_hex(raw_sha).is_none()
        {
            return Err(reject(
                "invalid_sha256",
                "sha256 must be 64 hexadecimal characters.",
            ));
        }
        if q.sha256.is_some() && !q.refresh {
            return Err(reject(
                "refresh_required",
                "?sha256= on /v1/analyze requires refresh=1.",
            ));
        }
        if q.refresh && q.sha256.is_none() {
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
        let budget = q.budget(state.config.level).map_err(|e| Box::new(e.v1()))?;
        // Everything this server analyses is filed, whatever policy produced
        // it: a corpus that records a shallower answer than it might have is
        // worth more than one that records nothing at all.
        //
        // TODO(t): Refactor the data model to allow realtime follow reassembly.
        // Dependencies, references, and CI actions belong in their own tables
        // rather than folded into one verdict; a caller's `follow=` is then a
        // view assembled from what is stored, and the question of which policy
        // owns the row stops being asked. Until then hopper is deliberately
        // policy-blind and the last writer wins.
        let follow = RequestFollow {
            policy: v1_follow_policy(&q, state.config.fetch).map_err(|e| Box::new(e.v1()))?,
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
                return Err(Box::new(
                    ApiError::new(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "too_many_packages",
                        "Only one exact url may be analyzed per request.",
                    )
                    .v1(),
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
                        ApiError::bad_request("invalid_purl", message).v1(),
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
        })
    }

    fn elapsed_ms(&self) -> u64 {
        crate::duration_ms(self.started.elapsed())
    }
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
    Extension(request_id): Extension<RequestId>,
    raw: axum::extract::RawQuery,
    headers: HeaderMap,
    body: axum::body::Body,
) -> Response {
    let request_id = request_id.get();
    let q = V1LookupQuery::parse(raw.0.as_deref()).with_fresh_header(&headers);
    let req = match V1Analyze::parse(&state, request_id, q) {
        Ok(req) => req,
        Err(rejected) => return *rejected,
    };

    // Memory is checked before the body is read: an artifact of up to
    // `--max-size-mb` is read only by a server with room for it. Full admission
    // waits until it has to analyze, so a verdict already held for these bytes
    // is answered even by a server whose startup failed. A refresh may be
    // answered from what is already held, so it is admitted once it has to
    // analyze.
    if !req.q.refresh
        && !body.is_end_stream()
        && let Err(refusal) = state.check_memory().await
    {
        return refusal.v1();
    }
    let (bytes, sha) = match read_body(body, state.config.max_body_size).await {
        Ok(read) => read,
        Err(refusal) => {
            tracing::warn!(
                id = request_id,
                status = refusal.status.as_u16(),
                "request body refused"
            );
            return refusal.v1();
        }
    };
    if !bytes.is_empty() {
        if req.q.refresh {
            return ApiError::bad_request(
                "sha256_with_body",
                "Use either ?sha256=&refresh=1 or an uploaded artifact, not both.",
            )
            .v1();
        }
        if req.url.is_some() {
            return ApiError::bad_request(
                "url_with_body",
                "Use either an exact url or an uploaded artifact, not both.",
            )
            .v1();
        }
        return analyze_bytes(&state, &req, &headers, bytes, sha).await;
    }
    if req.q.refresh {
        return refresh(&state, &req, headers).await;
    }
    if let Some(url) = req.url.clone() {
        return analyze_url(&state, &req, url).await;
    }
    if req.purl.is_some() {
        return analyze_purl(&state, &req).await;
    }
    ApiError::bad_request(
        "missing_package",
        "Name an artifact with ?purl= or ?url=, or send it as the body.",
    )
    .v1()
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
    while let Some(chunk) = std::future::poll_fn(|cx| {
        futures_core::Stream::poll_next(std::pin::Pin::new(&mut stream), cx)
    })
    .await
    {
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
        .unwrap_or_else(|| format!("upload-{request_id}"));
    let asked = req.purl.as_ref().map(|(asked, _)| asked.as_str());
    let purl = req.purl.as_ref().map(|(_, purl)| purl.clone());

    // Already answered? The digest is in hand before any temp file exists, so
    // this costs one index probe against the very artifact the caller sent;
    // without it, re-uploading something this worker has already analyzed
    // pays for the whole analysis again.
    //
    // Resolved by digest and by digest ALONE. The PURL must not reach the
    // resolver here, and this is not a style preference — passing it was a
    // false negative with a CVE's shape. Hopper answers a `?sha256=…&purl=…`
    // query on either key, so an upload of arbitrary bytes carrying `?purl=` of
    // a package the corpus knows came back with *that package's* verdict:
    // measured against production, 25 bytes of text sent as
    // `?purl=pkg:npm/chalk@5.3.0` were answered `allow` under chalk's digest,
    // and the bytes were never looked at. Anything can be laundered through a
    // reputable coordinate that way, which is precisely the attack this route
    // exists to catch. What is asked about is the digest, which is the only
    // thing an upload actually names; the caller's PURL is grafted back on
    // afterwards as provenance, which is all it ever was.
    //
    // Only an answer short-circuits. `unanalyzed` means nobody has analyzed
    // these bytes, which is why the caller sent them, and `unavailable` means
    // we could not find out — turning either into an answer would report on
    // work never done.
    if !req.q.full && !req.q.force && !req.q.refresh {
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

    let attachment = state.flights.join(FlightKey::sha_follow(
        sha.clone(),
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

/// Re-offer bytes already analyzed to hopper, off the reactor and at most
/// [`super::MAX_REPAIRS`] at once: each holds the whole upload until it is
/// staged, and a repeat past the bound is skipped — the next repeat of the
/// same artifact repairs it instead.
fn repair(state: &AppState, request_id: u64, filename: &str, sha: &str, bytes: bytes::Bytes) {
    let Some(uploader) = state.uploader.clone() else {
        return;
    };
    let Ok(permit) = Arc::clone(&state.repairs).try_acquire_owned() else {
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
    let Some(sha) = req.q.sha256.as_deref().map(str::to_ascii_lowercase) else {
        return ApiError::bad_request("missing_sha256", "refresh=1 requires ?sha256=.").v1();
    };
    // The local index is already namespaced by the currently loaded traits
    // version. It is the cheapest and most reliable answer, and a refresh must
    // not turn a valid Scan-local result into a Hopper dependency merely to
    // prove what Scan already knows.
    if !req.q.full {
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
    }

    let Some(corpus) = state.corpus.as_ref() else {
        return ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "hopper_unavailable",
            "refresh requires a configured Hopper corpus.",
        )
        .v1();
    };

    let (reached, source) = corpus.known_fresh_with_source(Some(&sha), None).await;
    if !req.q.full
        && let Reached::Record(record) = reached
    {
        // Reads the installed traits on first use, so off the reactor.
        let local = tokio::task::spawn_blocking(crate::corpus_precheck::local_traits)
            .await
            .ok()
            .flatten();
        if matching_traits_version(record.traits_version.as_deref(), local) {
            let decided = V1Decision::corpus(&record, Some(&sha), None, req.budget);
            let mut resp = answered(decided, corpus_source(source), req.elapsed_ms());
            if let Some(name) = req.follow.policy.follow_name()
                && let Ok(value) = HeaderValue::from_str(&name)
            {
                resp.headers_mut().insert("X-Scan-Follow", value);
            }
            resp.extensions_mut().insert(Subject::sha256(&sha));
            tracing::info!(id = request_id, sha256 = %sha, traits_version = ?record.traits_version, "refresh answered from Hopper at the current traits version");
            return resp;
        }
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
            .v1();
        }
    };
    if sample.sha256 != sha {
        tracing::error!(id = request_id, sha256 = %sha, fetched_sha256 = %sample.sha256, "Hopper returned bytes with the wrong digest");
        return ApiError::new(
            StatusCode::BAD_GATEWAY,
            "sample_digest_mismatch",
            "Hopper returned bytes that do not match the requested SHA-256.",
        )
        .v1();
    }
    if let Err(refusal) = state.admit_request(request_id).await {
        return refusal.v1();
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
        return refusal.v1();
    }
    let attachment = state.flights.join(FlightKey::url_follow(
        url.clone(),
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
async fn analyze_purl(state: &Arc<AppState>, req: &V1Analyze) -> Response {
    let request_id = req.request_id;
    let Some((asked, purl)) = req.purl.clone() else {
        return ApiError::bad_request("missing_package", "Name an artifact with ?purl=.").v1();
    };

    // Already answered?
    //
    // This is the expensive door into the question `/v1/lookup` answers
    // cheaply, and until it asked, every caller paid a full download-and-
    // classify for an artifact this worker already held a verdict for.
    // Measured against production, three consecutive analyses of
    // pkg:cargo/tokio@1.40.0 ran 291s, 161s and 116s while `/v1/lookup`
    // answered the same question from the index in a single hop.
    //
    // Resolved exactly the way the lookup resolves it — same normalization,
    // same index-then-corpus order, same budget — because two routes answering
    // one question differently is worse than either answer alone. Only an
    // answer short-circuits: `unanalyzed` is the whole reason the caller is
    // here, and `unavailable` means we could not find out — turning that into
    // a refusal to work would make a corpus outage look like an answer.
    if !req.q.full && !req.q.force && !req.q.refresh {
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

    let attachment = state.flights.join(FlightKey::purl_follow(
        purl.clone(),
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

/// A request refused admission, named with the follow policy it would have
/// run under — as every other analyze answer is, refusals included.
fn refused(req: &V1Analyze, refusal: &ApiError) -> Response {
    let mut resp = refusal.v1();
    if let Some(name) = req.follow.policy.follow_name()
        && let Ok(value) = HeaderValue::from_str(&name)
    {
        resp.headers_mut().insert("X-Scan-Follow", value);
    }
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

    fn access_subject(&self) -> Subject {
        if self.is_url {
            Subject::url(&self.subject, None)
        } else {
            Subject::purl(&self.subject, None)
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
            V1Decision::stored(&verdict, None, budget).asked_about_url(self.asked.as_deref())
        } else {
            let verdict = crate::lookup::Verdict::from_scan(result, key);
            V1Decision::stored(&verdict, key, budget).asked_about(self.asked.as_deref())
        }
    }
}

/// A successful full v1 response keeps the ordinary terminal marker while
/// exposing the complete scan envelope at the same top level.
#[derive(serde::Serialize)]
struct V1FullResult {
    status: &'static str,
    #[serde(flatten)]
    envelope: crate::engine::ScanResultEnvelope,
}

impl V1FullResult {
    fn from_scan(result: &crate::engine::ScanResult) -> Self {
        Self {
            status: "analyzed",
            envelope: result.to_envelope(),
        }
    }
}

/// The per-request framing an answer renders with: the budget the decision
/// cites, the clock its `elapsed_ms` fields count from, the follow policy
/// named on its headers, and whether the terminal frame is the full report or
/// the verdict.
struct V1Framing {
    request_id: u64,
    budget: u16,
    started: Instant,
    follow: Option<String>,
    full: bool,
}

/// Answer within the grace window if the analysis lands inside it, otherwise
/// as a stream.
async fn answer(req: &V1Analyze, attachment: Attachment, named: Named) -> Response {
    let framing = V1Framing {
        request_id: req.request_id,
        budget: req.budget,
        started: req.started,
        follow: req.follow.policy.follow_name(),
        full: req.q.full,
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
            let mut resp = if framing.full {
                Json(V1FullResult::from_scan(result)).into_response()
            } else {
                Json(named.decision(result, framing.budget)).into_response()
            };
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
        Outcome::Failed(refusal) => refusal.v1(),
    };
    // Which question this answer answers. The caller resolved a policy before
    // asking, but only this server knows what it applied on top of its own
    // configuration, and the answer has to be filed under what was measured
    // rather than what was requested. On refusals too: a caller correlating a
    // retry should not have to infer which policy was in play.
    if let Some(name) = framing.follow.as_deref()
        && let Ok(value) = HeaderValue::from_str(name)
    {
        resp.headers_mut().insert("X-Scan-Follow", value);
    }
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
/// against a healthier worker, and both are what poppy files under download
/// failures rather than its error rate. Every other status — 500, 504, 429, a
/// worker starting up — is about us, and stays an unqualified outage.
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
        full,
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
                if full {
                    v1_send(&tx, &V1FullResult::from_scan(result)).await;
                    return;
                }
                named.decision(result, budget)
            }
            Outcome::Failed(refusal) if refusal.status == StatusCode::TOO_MANY_REQUESTS => {
                // Refused after the stream began (a big whale found every slot
                // taken). Not a decision: the stream ends without one, which a
                // router reads as "hand this to a worker with room". A decision
                // here would be final and wrong — the package was never looked
                // at.
                tracing::warn!(id = request_id, subject = %named.subject, elapsed_ms = elapsed, "streamed analysis refused; ending without a decision");
                let mut frame = serde_json::json!({
                    "state": "refused",
                    "elapsed_ms": elapsed,
                });
                if let (Some(frame), serde_json::Value::Object(body)) =
                    (frame.as_object_mut(), refusal.v1_body())
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
                // a bare `unavailable` put that back for every streamed caller,
                // which is all of them: poppy asks with `Accept:
                // application/x-ndjson`, so it never saw the 422 and scored a
                // dead package as a beamline outage. On 2026-09-17 one deleted
                // GitHub repo — whose twelve versions proxy.golang.org still
                // lists from cache and can no longer serve — paged the fleet
                // that way.
                //
                // The decision stays `unavailable`: we still did not answer,
                // and a caller must not read an assessment into it. What
                // changes is that `reason` says whose failure it was, which
                // beamline already relays verbatim.
                let reason = unretrievable_reason(refusal.status);
                if named.is_url {
                    V1Decision::unavailable(None, None)
                        .asked_about_url(named.asked.as_deref())
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
    if let Some(name) = follow
        && let Ok(value) = HeaderValue::from_str(&name)
    {
        headers.insert("X-Scan-Follow", value);
    }
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
    let q = V1LookupQuery::parse(raw.0.as_deref()).with_fresh_header(&headers);
    let response = v1_lookup_inner(&state, &q)
        .await
        .unwrap_or_else(|refusal| refusal.v1());
    state
        .jobs
        .lookups
        .record(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
    response
}

async fn v1_lookup_inner(state: &Arc<AppState>, q: &V1LookupQuery) -> Result<Response, ApiError> {
    let budget = q.budget(state.config.level)?;
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
        return Err(ApiError::bad_request(
            "multiple_locators",
            "Use ?purl= or ?url=, not both.",
        ));
    }
    if urls.iter().any(|url| !valid_http_url(url)) {
        return Err(ApiError::bad_request(
            "invalid_url",
            "url must be an absolute http or https URL.",
        ));
    }
    if sha.is_none() && purls.is_empty() && urls.is_empty() {
        return Err(ApiError::bad_request(
            "missing_package",
            "Name an artifact with ?purl=, ?url=, or ?sha256=.",
        ));
    }
    let named = purls.len().max(urls.len());
    if named > V1_MAX_KEYS {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "too_many_packages",
            format!(
                "{named} packages exceeds the limit of {V1_MAX_KEYS} for a URL. Use POST /v1/lookup."
            ),
        ));
    }

    if let Some(&url) = urls.first() {
        if urls.len() > 1 {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "too_many_packages",
                "Only one exact url may be analyzed per request.",
            ));
        }
        let key = Key::parse(sha, None, Some(url))?;
        let (decided, source) = decide(state, &key, budget, q.fresh).await;
        return Ok(sourced(Json(decided.asked_about_url(Some(url))), source));
    }

    // One package named two ways is one question, so a lone sha256 and a lone
    // purl resolve together rather than as two entries.
    if purls.len() <= 1 {
        let purl = purls.first().copied();
        let key = Key::parse(sha, purl, None)?;
        let (decided, source) = decide(state, &key, budget, q.fresh).await;
        return Ok(sourced(Json(decided.asked_about(purl)), source));
    }

    // A key we cannot parse is the caller's mistake and stops the whole call,
    // because answering the rest would hide it.
    let keys = purls
        .iter()
        .map(|purl| Key::parse(None, Some(purl), None))
        .collect::<Result<Vec<_>, _>>()?;
    let decided = decide_many(state, keys, budget, q.fresh).await;
    let mut source = None;
    let mut out = Vec::with_capacity(decided.len());
    for ((decided, row_source), purl) in decided.into_iter().zip(&purls) {
        source = Some(match source {
            None => row_source,
            Some(previous) if previous == row_source => previous,
            Some(_) => "scan:analysis",
        });
        out.push(decided.asked_about(Some(purl)));
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
            // Held, not produced. This used to report `scan:analysis` — the
            // same value a fresh run reports — which made an instant index hit
            // and a ninety-second analysis indistinguishable to anything
            // counting cache layers.
            "scan:index",
        )
    }
}

/// The verdict this worker holds for `key`. The digest is the identity: a
/// PURL's verdict is accepted only when it describes the same bytes, because a
/// release whose digest has moved is an answer about a different artifact than
/// the one asked about. Blocking: the index is files.
fn held(index: &crate::lookup::Index, key: &Key) -> Option<crate::lookup::Verdict> {
    let by_purl = || key.purl.as_deref().and_then(|p| index.get_purl(p));
    match key.sha.as_deref() {
        Some(sha) => pick_verdict(index.get_sha(sha), by_purl, sha),
        None => by_purl(),
    }
}

/// The decision for one artifact, in this worker's own vocabulary.
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
    let lookup = key.clone();
    match index_query(move |index| held(index, &lookup)).await {
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
    let Some(verdicts) = index_query(move |index| {
        lookup
            .iter()
            .map(|key| held(index, key))
            .collect::<Vec<_>>()
    })
    .await
    else {
        return out;
    };
    let mut misses = Vec::new();
    for (i, (key, verdict)) in keys.into_iter().zip(verdicts).enumerate() {
        match verdict {
            Some(verdict) => out[i] = key.stored(&verdict, budget),
            None => misses.push((i, key)),
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

fn corpus_source(source: Option<corpus::CorpusSource>) -> &'static str {
    match source {
        Some(corpus::CorpusSource::Replica) => "scan:replica",
        Some(corpus::CorpusSource::Primary) => "scan:primary",
        None => "none",
    }
}

/// The decision for an artifact this worker's index does not hold: the
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
    let bloom = if fresh {
        crate::bloom_repo::Decision::Unknown
    } else {
        bloom_decision(sha, purl)
    };

    // A bless answers immediately and does not pay the hopper round trip. The
    // exposure is a bless that has gone stale — but that is bounded by the
    // filter rebuild, because `good` is rebuilt as `good − (bad ∪ sighted)` and
    // the bad channel is the designed revocation path. It is also the bargain
    // the local scan path already takes: `bloom_skip_predicate` skips the
    // download outright on a good hit, without asking anyone.
    if bloom == crate::bloom_repo::Decision::Skip
        && let Some(d) = V1Decision::bloom(bloom, sha, purl, budget)
    {
        return (d.with_url(url), "scan:bloom");
    }

    let filters_or_unanalyzed = || {
        V1Decision::bloom(bloom, sha, purl, budget)
            .unwrap_or_else(|| V1Decision::unanalyzed(sha, purl))
            .with_url(url)
    };

    // Not in this worker's index. The corpus behind it may still know, and a
    // caller should not have to learn that two services exist in order to get
    // one answer — so ask, rather than reporting an absence that is only ours.
    //
    // Measured beats derived: a filter claim is a floor, and hopper may hold
    // the real level, the real findings and the sentence a person reads. Only
    // when it holds nothing does the filter's own claim stand in.
    let Some(corpus) = state.corpus.as_ref() else {
        return (filters_or_unanalyzed(), "scan:bloom");
    };
    let (reached, source) = corpus.known_with_source(sha, purl).await;
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
    /// can tune that budget against real numbers; `null` when there is none.
    fires_at: Level,
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
        let fires_at = claim.lvl;
        let (decided, severity) = decision::decide(fires_at, budget);
        Some(Self {
            decision: decided,
            purl: purl.map(str::to_owned),
            url: None,
            sha256: sha.map(str::to_owned),
            severity: Some(severity),
            fires_at,
            reason: claim.finding.as_ref().map(|f| f.desc.to_owned()),
            findings: claim
                .finding
                .as_ref()
                .map(|f| V1Finding::from_bloom(f, purl))
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
            fires_at: Level::Manual,
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
        ) && !self.fires_at.is_manual()
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

    /// Attach the reason an `unavailable` came about. `None` leaves the
    /// decision exactly as it was, so the common outage keeps its bare shape.
    fn because(mut self, reason: Option<&'static str>) -> Self {
        if let Some(reason) = reason {
            self.reason = Some(reason.to_owned());
        }
        self
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
    fn from_bloom(c: &crate::lookup::BloomFinding, purl: Option<&str>) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A finding says only what is known about it.
    ///
    /// The decision object around it keeps every key at all times; a finding
    /// does not, because how much is known about one varies by where it came
    /// from. A corpus finding knows the trait and its criticality and nothing
    /// else — sending four nulls to say so is most of the object.
    #[test]
    fn a_finding_omits_what_it_does_not_know() {
        use super::corpus::CorpusFinding;
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
        use super::decision::Decision;
        let purl = Some("pkg:npm/left-pad@1.3.0");
        assert!(!V1Decision::unanalyzed(None, purl).is_answerable());
        assert!(!V1Decision::unavailable(None, purl).is_answerable());

        // A decision with no level answers nothing, whatever it is labelled.
        assert!(!V1Decision::empty(Decision::Block, None, purl).is_answerable());

        let measured = |d| {
            let mut v = V1Decision::empty(d, None, purl);
            v.engine_version = Some("2.8.0".into());
            v.fires_at = Level::At(10);
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
        let headers = |value: Option<&str>| {
            let mut h = HeaderMap::new();
            if let Some(v) = value {
                h.insert(
                    "x-hopper-fresh",
                    HeaderValue::from_str(v).expect("header value"),
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
    fn refresh_has_one_unambiguous_wire_spelling() {
        let q = |raw: &str| -> bool { V1LookupQuery::parse(Some(raw)).refresh };
        assert!(q(
            "sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa&refresh=1"
        ));
        for value in ["", "0", "true", "yes", "2"] {
            let raw = format!(
                "sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa&refresh={value}"
            );
            assert!(!q(&raw));
        }
    }

    #[test]
    fn full_has_one_unambiguous_wire_spelling() {
        let q = |raw: &str| -> bool { V1LookupQuery::parse(Some(raw)).full };
        assert!(q("purl=pkg:npm/left-pad@1.3.0&full=1"));
        for value in ["", "0", "true", "yes", "2"] {
            let raw = format!("purl=pkg:npm/left-pad@1.3.0&full={value}");
            assert!(!q(&raw));
        }
    }

    #[test]
    fn refresh_reuses_only_an_explicit_matching_traits_version() {
        assert!(matching_traits_version(Some("abc12"), Some("abc12")));
        assert!(!matching_traits_version(Some("old00"), Some("abc12")));
        assert!(!matching_traits_version(None, Some("abc12")));
        assert!(!matching_traits_version(Some("abc12"), None));
        assert!(!matching_traits_version(None, None));
    }

    #[test]
    fn follow_policy_repeats_union_and_overrides_the_server_default() {
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
        use super::corpus::{CorpusFinding, CorpusRecord};
        use crate::lookup::{Hit, Verdict};

        let stored = Verdict {
            sha256: "a".repeat(64),
            lvl: Level::At(3),
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
            fires_at: Level::At(3),
            engine_version: Some("2.8.0".into()),
            traits_version: None,
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
        use crate::lookup::Verdict;

        let asked = "pkg:pypi/info.gianlucacosta.eos.core@2.0.2";
        let normalized = "pkg:pypi/info-gianlucacosta-eos-core@2.0.2";
        assert_eq!(
            normalize_pkg_purl(asked).as_deref(),
            Ok(normalized),
            "the premise: these two spellings are one package",
        );

        // PEP 503 lowercases as well as folding separators, and that half was
        // sighted separately in production: `pkg:pypi/ImportanceScore@1.2` came
        // back answered about `pkg:pypi/importancescore@1.2`. Same cause, and a
        // caller whose package name has no `.` or `_` in it at all.
        let mixed = "pkg:pypi/ImportanceScore@1.2";
        assert_eq!(
            normalize_pkg_purl(mixed).as_deref(),
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
            lvl: Level::Clean,
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
        use super::corpus::{CorpusFinding, CorpusRecord};
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
            lvl: Level::Clean,
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
            fires_at: Level::Clean,
            engine_version: Some("2.8.0".into()),
            traits_version: None,
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
        let d = V1Decision::unavailable(Some("a"), Some("pkg:npm/x@1.0.0"));
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

    /// The line this draws is the one [`ApiError::from_analysis`] already
    /// draws for unstreamed callers: an artifact nobody can download is the
    /// package's failure, and everything else is ours. Poppy only ever reads
    /// the stream, so until this existed a dead package reached it as an
    /// outage and went into the fleet error rate — which is how one deleted Go
    /// repo paged the fleet on 2026-09-17.
    #[test]
    fn only_the_artifacts_own_failures_are_named() {
        assert_eq!(
            unretrievable_reason(StatusCode::UNPROCESSABLE_ENTITY),
            Some("unretrievable"),
        );
        assert_eq!(
            unretrievable_reason(StatusCode::PAYLOAD_TOO_LARGE),
            Some("too_large"),
        );

        // Ours. These stay an unqualified outage, because retrying them against
        // a healthier worker is exactly the right thing for a caller to do.
        for status in [
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::GATEWAY_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::NOT_FOUND,
        ] {
            assert_eq!(unretrievable_reason(status), None, "status {status}");
        }
    }

    /// `because` qualifies an unavailable without promoting it: the decision is
    /// still that we did not answer, so nothing may read an assessment from it.
    #[test]
    fn a_reason_qualifies_but_does_not_become_a_verdict() {
        let bare = V1Decision::unavailable(None, Some("pkg:golang/example.com/x@v1.0.0"));
        assert_eq!(bare.reason, None);

        let named = V1Decision::unavailable(None, Some("pkg:golang/example.com/x@v1.0.0"))
            .because(unretrievable_reason(StatusCode::UNPROCESSABLE_ENTITY));
        assert_eq!(named.reason.as_deref(), Some("unretrievable"));
        assert_eq!(named.decision, super::decision::Decision::Unavailable);
        assert_eq!(named.severity, None);
        assert_eq!(named.fires_at, Level::Manual);
        assert!(named.findings.is_empty());

        // No reason leaves the decision byte-for-byte what it was.
        let untouched = V1Decision::unavailable(None, None).because(None);
        assert_eq!(untouched.reason, None);
    }

    /// A streamed upload's progress names the phase of the run it rides. The
    /// upload's `/_/requests` entry is named by its filename while its subject
    /// is the digest, so the old by-name search reported `phase: null` for
    /// every streamed upload.
    #[test]
    fn a_streamed_uploads_progress_carries_its_phase() {
        let flights = Arc::new(super::super::flight::Flights::default());
        let sha = "c".repeat(64);
        let leader = flights.join(FlightKey::Sha(sha.clone()));
        let follower = flights.join(FlightKey::Sha(sha.clone()));
        let phase = crate::analysis::RequestPhase::with_label("req#3 upload-3");
        leader.flight().set_phase(phase.clone());
        phase.set("cleave:analyze");

        let named = Named {
            key: None,
            asked: None,
            subject: sha.clone(),
            is_url: false,
        };
        let frame = progress_frame(&follower, &named, Instant::now());
        assert_eq!(frame["state"], "analyzing");
        assert_eq!(frame["phase"], "cleave:analyze");
        assert_eq!(frame["sha256"], sha, "an upload is named by its digest");
    }
}
