//! `GET /v1/lookup`, and the resolution `/v1/analyze` shares with it: what
//! this server, its filters and the corpus behind it already know about an
//! artifact, without analyzing anything.

use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue};
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;
use std::time::Instant;

use super::V1Query;
use crate::lookup::pick_verdict;
use crate::server::AppState;
use crate::server::analyze::index_query;
use crate::server::corpus::{self, Reached};
use crate::server::decision::V1Decision;
use crate::server::error::ApiError;
use crate::server::input::{normalize_pkg_purl, valid_http_url};

/// How many packages one URL may name.
///
/// A PURL runs about fifty characters encoded, so fifty of them sits well
/// inside every intermediary's URL limit with room to spare. Past this the
/// answer is POST, and the error says so rather than leaving it to be
/// discovered by a truncated query string.
const V1_MAX_KEYS: usize = 50;

/// How many of one `/v1/lookup`'s packages ask the corpus at once.
const V1_LOOKUP_CONCURRENCY: usize = 8;

/// GET /v1/lookup — what we know, at the caller's threshold. Never analyzes.
///
/// Answers a single object when one package is named and an array when `purl`
/// repeats, so the shape follows the shape of the question rather than the data:
/// a caller that always asks about one always gets one, and a caller that always
/// asks about many always gets many. Neither ever has to branch on what came
/// back.
pub(in crate::server) async fn v1_lookup(
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
    // A digest pairs with one PURL; beside several it would name none of
    // them, and answering by PURL alone would silently drop it.
    if !shas.is_empty() && purls.len() > 1 {
        return Err(ApiError::bad_request(
            "multiple_locators",
            "A sha256 pairs with one purl; send several purls without ?sha256=.",
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
        // Answered by its digest, when one came with it. A URL alone is
        // answered by the bytes it last served: for good when it parses into
        // an immutable PURL, otherwise only within the index's
        // MUTABLE_ALIAS_TTL of that fetch.
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
    let source = decided
        .iter()
        .map(|&(_, source)| source)
        .max()
        .unwrap_or(Source::None);
    let out: Vec<_> = decided
        .into_iter()
        .zip(asked)
        .map(|((decided, _), purl)| decided.asked_about(purl))
        .collect();
    Ok(sourced(Json(out), source))
}

fn sourced(body: impl IntoResponse, source: Source) -> Response {
    let mut resp = body.into_response();
    resp.headers_mut().insert("X-Scan-Source", source.into());
    resp
}

/// Where an answer came from, for `X-Scan-Source`.
///
/// Declared in order of what it costs to be answered there, so a batch
/// reports its deepest row:
/// that is what the request cost. `None` reached no layer at all, so it stands
/// only when no row was answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Source {
    None,
    Index,
    Cached,
    Bloom,
    Replica,
    Primary,
    Analysis,
}

impl Source {
    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Index => "scan:index",
            Self::Cached => "scan:cached",
            Self::Bloom => "scan:bloom",
            Self::Replica => "scan:replica",
            Self::Primary => "scan:primary",
            Self::Analysis => "scan:analysis",
        }
    }
}

impl From<Source> for HeaderValue {
    fn from(source: Source) -> Self {
        Self::from_static(source.as_str())
    }
}

/// One artifact a v1 request names, every key normalized: what the index and
/// the corpus are keyed by.
#[derive(Clone, Debug, Default)]
pub(super) struct Key {
    pub(super) sha: Option<String>,
    pub(super) purl: Option<String>,
    pub(super) url: Option<String>,
}

impl Key {
    pub(super) fn parse(
        sha: Option<&str>,
        purl: Option<&str>,
        url: Option<&str>,
    ) -> Result<Self, ApiError> {
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
    /// can choose among every server's answers; a local alias covers just one
    /// artifact.
    pub(super) fn needs_corpus(&self, configured: bool) -> bool {
        configured
            && self.sha.is_none()
            && self.purl.as_deref().is_some_and(|purl| {
                fletch::purl::Purl::parse(purl).is_ok_and(|purl| purl.version().is_some())
            })
    }

    fn unavailable(&self) -> (V1Decision, Source) {
        (
            V1Decision::unavailable(self.sha.as_deref(), self.purl.as_deref())
                .with_url(self.url.as_deref()),
            Source::None,
        )
    }

    fn stored(&self, verdict: &crate::lookup::Verdict, budget: u16) -> (V1Decision, Source) {
        (
            V1Decision::stored(verdict, self.purl.as_deref(), budget).with_url(self.url.as_deref()),
            // Held, not produced. Reporting `scan:analysis` — the value a
            // fresh run reports — would make an instant index hit and a
            // ninety-second analysis indistinguishable to anything counting
            // cache layers.
            Source::Index,
        )
    }
}

/// The verdict this server holds for `key`. The digest is the identity: a
/// PURL's verdict is accepted only when it describes the same bytes, because a
/// release whose digest has moved is an answer about a different artifact than
/// the one asked about. Blocking: the index is files.
pub(super) fn held(
    index: &crate::lookup::Index,
    key: &Key,
    corpus: bool,
) -> Option<crate::lookup::Verdict> {
    if key.needs_corpus(corpus) {
        return None;
    }
    let by_purl = || key.purl.as_deref().and_then(|p| index.purl(p));
    match key.sha.as_deref() {
        Some(sha) => pick_verdict(index.sha(sha), by_purl, sha),
        None => by_purl().or_else(|| key.url.as_deref().and_then(|u| index.url(u))),
    }
}

/// The decision for one artifact, in this server's own vocabulary.
///
/// A missing index is not an empty one. Reporting `unanalyzed` then would tell
/// the caller nobody has analyzed this package, when what is true is that we
/// cannot say — and those two carry different policies at the other end. That
/// is the whole reason `unavailable` is a separate value.
pub(super) async fn decide(
    state: &AppState,
    key: &Key,
    budget: u16,
    fresh: bool,
) -> (V1Decision, Source) {
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
pub(super) async fn decide_many(
    state: &Arc<AppState>,
    keys: Vec<Key>,
    budget: u16,
    fresh: bool,
) -> Vec<(V1Decision, Source)> {
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

pub(super) fn corpus_source(source: Option<corpus::CorpusSource>) -> Source {
    match source {
        Some(corpus::CorpusSource::Replica) => Source::Replica,
        Some(corpus::CorpusSource::Primary) => Source::Primary,
        None => Source::None,
    }
}

/// The decision for an artifact this server's index does not hold: the
/// filters, then the corpus.
async fn decide_unheld(
    state: &AppState,
    key: &Key,
    budget: u16,
    fresh: bool,
) -> (V1Decision, Source) {
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
        return (d.with_url(url), Source::Bloom);
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
        // Credit the filters only when they answered; an `unanalyzed` row
        // was answered by nothing.
        return match V1Decision::bloom(bloom, sha, purl, budget) {
            Some(decided) => (decided.with_url(url), Source::Bloom),
            None => (
                V1Decision::unanalyzed(sha, purl).with_url(url),
                Source::None,
            ),
        };
    };
    let (reached, source) = corpus.known_with_source(sha, purl, fresh).await;
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
