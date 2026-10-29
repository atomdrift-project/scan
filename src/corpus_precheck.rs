//! Fleet-shared skip for fetched-dependency analysis, backed by hopper's corpus.
//!
//! `crate::analysis_cache` already memoizes dependency analyses — but per
//! host, namespaced by ruleset version. A fleet release therefore invalidates
//! every host's cache at once, and each host rebuilds a private copy of the
//! same shared dependency universe: thousands of redundant re-scans per hour,
//! every one stored by hopper with "result renewed with no analyzer change;
//! the re-analysis learned nothing".
//!
//! This module asks the one cache the whole fleet shares — hopper — before
//! analyzing a fetched payload or resolving a package coordinate:
//!
//!   1. TRAITS MATCH: the stored verdict was produced under this process's own
//!      analyzer version (`traits_version` equals our 5-char traits commit,
//!      the same truncation the /api/next heartbeat sends). Re-analysis by the
//!      same analyzer learns nothing — hopper logs exactly that when it
//!      happens — so this skips ANY verdict, hostile included.
//!   2. BENIGN AND FRESH: `fires_at` is `-1` (clean), analyzed within the last 30 days
//!      (see `DEFAULT_MAX_AGE_DAYS` for why that long), regardless of
//!      analyzer version. This applies only to package-coordinate lookups
//!      before fetching. Once bytes have been fetched, another analyzer's
//!      clean verdict cannot replace analysis under the current rules.
//!      The coarse rule that keeps working through a
//!      mixed-version fleet or a release that just bumped every traits hash:
//!      dependency universes are overwhelmingly benign.
//!
//! Anything neither rule covers — not found, not fresh, hostile under a
//! different analyzer, unreachable — falls through to a normal analysis:
//! fail-open, never fail-closed.
//!
//! Enabled automatically whenever `--hopper` is: [`configure`] is called with
//! the process's own hopper URL whenever an uploader is built, so there is no
//! second setting to keep in sync — the hopper you submit to is the hopper you
//! ask. (Hopper serves `/v1/lookup` from an in-memory pool on both the primary
//! and the read replica, so whichever the process talks to answers cheaply.)
//! No `--hopper`, no precheck. Auth reuses the process's hopper bearer token
//! (see [`crate::hopper::token`]).
//!
//! # isotope13
//!
//! `--enriched` scans arm the same precheck against isotope13's API instead
//! ([`configure_isotope13`]), with the subscription token. The wire is the
//! same `/v1/lookup`; what changes is whose judgment it is. Hopper is a corpus
//! of other runs, so only the same analyzer's verdict is ours. isotope13 is the
//! service the subscriber asked to judge for them, so an analyzed verdict
//! stands whatever produced it — with one bound kept from rule 2: a clean
//! verdict older than the max age is analyzed again rather than trusted. A
//! refused token or an exhausted quota opens the breaker: the scan carries on
//! locally, which is always a complete answer.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, PoisonError, RwLock};
use std::time::Duration;

use crate::hopper::{Finding, Record, Status};
use crate::model::Level;

/// Lookups attempted (a hit or a miss, but the wire was asked).
static CHECKS: AtomicU64 = AtomicU64::new(0);
/// Analyses skipped because the corpus held a matching-analyzer verdict.
static SKIPS: AtomicU64 = AtomicU64::new(0);
/// PURLs asked of hopper by the pre-fetch batch negotiation.
static PURL_CHECKS: AtomicU64 = AtomicU64::new(0);
/// PURLs whose fetch+analysis were skipped on hopper's standing verdict.
static PURL_SKIPS: AtomicU64 = AtomicU64::new(0);

const BREAKER_LIMIT: u32 = 5;
const BREAKER_COOLDOWN_SECS: u64 = 300;

/// Per-request ceiling on a lookup. It sits on the analysis path, so a slow
/// replica must cost seconds, not the pipeline.
const LOOKUP_TIMEOUT: Duration = Duration::from_secs(2);

/// PURLs per `/v1/lookup` request: hopper's documented cap.
const PURL_BATCH: usize = 50;

/// How fresh a benign verdict must be to stand in for a re-analysis.
///
/// 30 rather than something tighter because this window is NOT the safety net
/// against a benign verdict going bad — the threat-feed path is (a cited
/// dependency is force-rescanned via cyclotron regardless of this cache), and
/// hopper's own stale-traits rescan makes corpus rows re-analysis-eligible
/// after 30 days (--rescan-age, aligned with this window), resetting analyzed_at whenever it drains to
/// them. A verdict only ever reaches this age if every corpus refresh channel
/// left it alone; the residual exposure is a detector improvement on an
/// uncited, never-requeued dep, which self-heals when the rescan tier reaches
/// it. Known gap, deliberately not this module's job: nothing yet refreshes a
/// package when its dependency RECORDS change.
const DEFAULT_MAX_AGE_DAYS: u64 = 30;

/// Consecutive transport failures, kept as a half-open circuit breaker. At
/// [`BREAKER_LIMIT`] it opens: a dead replica must cost one log line, not a
/// per-dependency connect timeout inside the analysis pipeline. After
/// [`BREAKER_COOLDOWN_SECS`] one caller probes again; a success closes it, a
/// failure restarts the cooldown.
#[derive(Debug, Default)]
struct Breaker {
    failures: AtomicU32,
    /// When the breaker last opened or probed, in Unix seconds.
    opened_at: AtomicU64,
}

impl Breaker {
    /// Whether a lookup may go on the wire at `now`: the breaker is closed, or
    /// it has been open a full cooldown and this caller won the one probe.
    fn allows(&self, now: u64) -> bool {
        if self.failures.load(Ordering::Relaxed) < BREAKER_LIMIT {
            return true;
        }
        let opened = self.opened_at.load(Ordering::Relaxed);
        now >= opened.saturating_add(BREAKER_COOLDOWN_SECS)
            && self
                .opened_at
                .compare_exchange(opened, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
    }

    /// Open the breaker now, whatever the count: the backend said no, and every
    /// later ask before the cooldown would hear the same. Returns whether this
    /// call opened it, so the reason is logged once.
    fn trip(&self, now: u64) -> bool {
        self.opened_at.store(now, Ordering::Relaxed);
        self.failures.swap(BREAKER_LIMIT, Ordering::Relaxed) < BREAKER_LIMIT
    }

    /// The wire answered: close the breaker.
    fn succeeded(&self) {
        self.failures.store(0, Ordering::Relaxed);
    }

    /// Count a transport failure at `now`; returns whether the breaker is open.
    fn failed(&self, now: u64, what: &str, error: &dyn std::fmt::Display) -> bool {
        let n = self
            .failures
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        if n < BREAKER_LIMIT {
            return false;
        }
        self.opened_at.store(now, Ordering::Relaxed);
        if n == BREAKER_LIMIT {
            tracing::warn!(
                error = %error,
                "{what}: {BREAKER_LIMIT} consecutive transport failures; \
                 pausing for {BREAKER_COOLDOWN_SECS}s"
            );
        }
        true
    }
}

/// The hopper this process asks, and how it weighs the answers.
#[derive(Debug)]
pub struct Precheck {
    lookup_url: String,
    max_age: Duration,
    breaker: Breaker,
    backend: Backend,
}

/// Who answers, which decides how it is authenticated and whose judgment an
/// answer is.
#[derive(Debug)]
enum Backend {
    /// The hopper this process submits to. Only a same-analyzer verdict is ours.
    Hopper,
    /// isotope13's API under `--enriched`. Its analyzed verdicts are the answer.
    Isotope13(cleave::enriched::Token),
}

/// The armed precheck. Every [`configure`] replaces it, so the last hopper a
/// process was pointed at is the one it asks.
static ARMED: RwLock<Option<Arc<Precheck>>> = RwLock::new(None);

/// Arm the precheck against isotope13's API, for an `--enriched` scan in
/// `--analyze=auto`. A later [`configure`] (`--hopper`) replaces it: the
/// hopper a process submits to is still the one it asks.
pub fn configure_isotope13(token: cleave::enriched::Token) {
    let max_age_days = DEFAULT_MAX_AGE_DAYS;
    let precheck = Precheck {
        lookup_url: format!("{}/v1/lookup", crate::remote::API_URL),
        max_age: Duration::from_secs(max_age_days.saturating_mul(86_400)),
        breaker: Breaker::default(),
        backend: Backend::Isotope13(token),
    };
    tracing::info!(
        event = "remote.armed",
        url = %precheck.lookup_url,
        tid = %precheck.tid(),
        max_age_days,
        "isotope13 lookups enabled: known artifacts are answered without local analysis"
    );
    *ARMED.write().unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(precheck));
}

/// Arm the precheck against the hopper this process already talks to. Called
/// where the uploader is built — the one place every `--hopper` mode passes
/// through — so enablement follows `--hopper` with no second setting.
pub fn configure(hopper_base_url: &str) {
    let max_age_days = DEFAULT_MAX_AGE_DAYS;
    let precheck = Precheck::new(
        hopper_base_url,
        Duration::from_secs(max_age_days.saturating_mul(86_400)),
    );
    let mut armed = ARMED.write().unwrap_or_else(PoisonError::into_inner);
    if let Some(p) = &precheck
        && armed
            .as_ref()
            .is_none_or(|old| old.lookup_url != p.lookup_url)
    {
        tracing::info!(
            url = %p.lookup_url,
            max_age_days,
            "corpus precheck enabled: matching-analyzer verdicts may be reused; fresh benign package coordinates may skip fetching"
        );
    }
    *armed = precheck.map(Arc::new);
}

/// The armed precheck, or `None` when this process has no hopper.
pub fn armed() -> Option<Arc<Precheck>> {
    ARMED.read().unwrap_or_else(PoisonError::into_inner).clone()
}

/// Downloaded bytes require current-analyzer coverage; package coordinates
/// may use the bounded fresh-benign shortcut before any download takes place.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lookup {
    DownloadedBytes,
    PackageCoordinate,
}

/// A verdict this build may adopt as its own: the corpus produced it under the
/// analyzer we are running, so re-deriving it locally would reach the same
/// answer. `fires_at` is the tightest false-positive budget at which the
/// artifact grades hostile; grading it against a caller's budget is
/// Postdoc's request handler's job, never this module's.
#[derive(Debug, Clone, Default)]
pub struct Verdict {
    pub fires_at: Level,
    pub reason: Option<String>,
    pub findings: Vec<Finding>,
    /// When it was analyzed (RFC 3339), when the answer said.
    pub analyzed_at: Option<String>,
    /// The scanner build that produced it, when the answer said.
    pub engine_version: Option<String>,
}

impl From<Record> for Verdict {
    fn from(rec: Record) -> Self {
        Self {
            fires_at: rec.fires_at,
            reason: rec.reason,
            findings: rec.findings,
            analyzed_at: rec.analyzed_at,
            engine_version: rec.engine_version,
        }
    }
}

/// What the corpus lets us skip, and whether it also hands us an answer.
///
/// The distinction is the whole point: a verdict is only ours to report when it
/// was produced by the analyzer we are running. Anything else is evidence about
/// someone else's judgment — usable for a package-coordinate fetch shortcut,
/// not a substitute for analyzing bytes already downloaded by this scan.
#[derive(Debug, Clone)]
pub enum Standing {
    /// Rule 1: the same analyzer already judged these bytes. Adopt its verdict.
    Adopt(Verdict),
    /// Rule 2: benign under some other analyzer, and fresh. Skip the work; there
    /// is nothing to report, which for a benign artifact is the whole verdict.
    SkipBenign,
    /// Neither rule applies. Analyze.
    Analyze,
}

impl Standing {
    /// Whether this standing spares us the analysis at all.
    #[must_use]
    pub const fn skips_analysis(&self) -> bool {
        !matches!(self, Self::Analyze)
    }
}

/// A dependency the batch PURL negotiation answered for: the content sha that
/// records the fetch edge, and what the corpus lets us skip for it — never
/// [`Standing::Analyze`].
#[derive(Debug, Clone)]
pub struct PurlHit {
    pub sha: String,
    pub standing: Standing,
}

/// This process's 5-char traits commit prefix — the same value the /api/next
/// heartbeat sends, truncated the way hopper truncates, so string equality
/// against a stored `traits_version` means "same analyzer".
pub fn local_traits() -> Option<&'static str> {
    static TRAITS: OnceLock<Option<String>> = OnceLock::new();
    TRAITS
        .get_or_init(|| cleave::traits_repo::version().map(|v| v.chars().take(5).collect()))
        .as_deref()
}

impl Precheck {
    /// A precheck against the first hopper `hopper_base_url` names, or `None`
    /// when it names none.
    fn new(hopper_base_url: &str, max_age: Duration) -> Option<Self> {
        // HOPPER may be a comma list, replica first ("https://ro,…"): lookups
        // belong on the first entry — the replica when one is named, which is
        // exactly where a cheap read should land.
        let base = crate::hopper::endpoints(hopper_base_url)
            .into_iter()
            .next()?;
        Some(Self {
            lookup_url: format!("{base}/v1/lookup"),
            max_age,
            breaker: Breaker::default(),
            backend: Backend::Hopper,
        })
    }

    /// Whether this precheck asks isotope13 (`--enriched`) rather than hopper.
    pub fn is_isotope13(&self) -> bool {
        matches!(self.backend, Backend::Isotope13(_))
    }

    /// Per-request ceiling: tight for a hopper replica on the fleet's network,
    /// roomier for a public API across the internet.
    fn timeout(&self) -> Duration {
        match self.backend {
            Backend::Hopper => LOOKUP_TIMEOUT,
            Backend::Isotope13(_) => crate::remote::LOOKUP_TIMEOUT,
        }
    }

    /// The token's fingerprint, for logs; empty for hopper, whose credential
    /// is logged where it is loaded.
    fn tid(&self) -> String {
        match &self.backend {
            Backend::Hopper => String::new(),
            Backend::Isotope13(token) => token.fingerprint(),
        }
    }

    fn http(&self) -> Option<&'static reqwest::blocking::Client> {
        match self.backend {
            Backend::Hopper => crate::hopper::http(),
            Backend::Isotope13(_) => crate::remote::http(),
        }
    }

    /// Send `request`, authenticated, and feed the outcome to the breaker.
    fn send(
        &self,
        request: reqwest::blocking::RequestBuilder,
        what: &str,
    ) -> Option<reqwest::blocking::Response> {
        match self.authed(request).send() {
            Ok(r) => {
                self.breaker.succeeded();
                Some(r)
            }
            Err(e) => {
                self.breaker.failed(crate::civil::now_secs(), what, &e);
                None
            }
        }
    }

    fn authed(
        &self,
        request: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        match &self.backend {
            Backend::Hopper => match crate::hopper::token() {
                Some(token) => request.bearer_auth(token),
                None => request,
            },
            Backend::Isotope13(token) => request.bearer_auth(token.secret()),
        }
    }

    /// Whether to read this answer at all. Every non-200 means "analyze it";
    /// from isotope13, a refused token or an exhausted quota also means "stop
    /// asking", because every later lookup in the run would get the same
    /// answer.
    fn answered(&self, status: reqwest::StatusCode, what: &str) -> bool {
        if status == reqwest::StatusCode::OK {
            return true;
        }
        if matches!(self.backend, Backend::Isotope13(_))
            && matches!(status.as_u16(), 401 | 403 | 429)
        {
            let reason = if status.as_u16() == 429 {
                "quota"
            } else {
                "token_refused"
            };
            if self.breaker.trip(crate::civil::now_secs()) {
                tracing::warn!(
                    event = "remote.disabled",
                    reason,
                    status = status.as_u16(),
                    tid = %self.tid(),
                    "{what}: isotope13 lookups paused; analyzing locally"
                );
            }
        }
        false
    }

    /// The policy for this backend's answers.
    fn judge(&self, rec: Record, lookup: Lookup) -> Standing {
        match self.backend {
            Backend::Hopper => standing_of(
                rec,
                local_traits(),
                self.max_age.as_secs(),
                crate::civil::now_secs(),
                lookup,
            ),
            Backend::Isotope13(_) => {
                service_standing(rec, self.max_age.as_secs(), crate::civil::now_secs())
            }
        }
    }

    /// What hopper's corpus already holds for these downloaded bytes: a verdict
    /// this build may adopt (same analyzer), or [`Standing::Analyze`]. A clean
    /// verdict from another analyzer does not cover the current rules. Any
    /// failure — unreachable, non-200, unparseable — also means analyze.
    pub fn standing(&self, content_sha: &str) -> Standing {
        if content_sha.len() != 64 || !self.breaker.allows(crate::civil::now_secs()) {
            return Standing::Analyze;
        }
        let Some(http) = self.http() else {
            return Standing::Analyze;
        };
        CHECKS.fetch_add(1, Ordering::Relaxed);
        let request = http
            .get(&self.lookup_url)
            .timeout(self.timeout())
            .query(&[("sha256", content_sha)]);
        let Some(resp) = self.send(request, "corpus precheck") else {
            return Standing::Analyze;
        };
        if !self.answered(resp.status(), "corpus precheck") {
            return Standing::Analyze; // 404 unknown, 202 bytes-only, 401/5xx — all "scan it".
        }
        let rec = match resp.json::<Record>() {
            Ok(rec) => rec,
            Err(e) => {
                tracing::debug!(error = %e, "corpus precheck: unreadable lookup answer; analyzing");
                return Standing::Analyze;
            }
        };
        let standing = self.judge(rec, Lookup::DownloadedBytes);
        if standing.skips_analysis() {
            SKIPS.fetch_add(1, Ordering::Relaxed);
        }
        standing
    }

    /// Batch digest negotiation: which of these artifacts — local files, named
    /// by their content — does the backend already judge? Returns
    /// `sha256 → standing` for every one it answers with a standing verdict,
    /// under the same policy as [`Self::standing`]. An answer is paired by its
    /// own `sha256` and must name a digest that was asked, so an answer about
    /// other bytes can never stand in for this file. Fail-open everywhere; an
    /// open breaker stops the batch.
    pub fn shas(&self, shas: &[String]) -> HashMap<String, Standing> {
        let mut out = HashMap::new();
        let Some(http) = self.http() else {
            return out;
        };
        for chunk in shas.chunks(PURL_BATCH) {
            if !self.breaker.allows(crate::civil::now_secs()) {
                break;
            }
            CHECKS.fetch_add(chunk.len() as u64, Ordering::Relaxed);
            let lookup = http.get(&self.lookup_url).timeout(self.timeout());
            let request = chunk
                .iter()
                .fold(lookup, |req, sha| req.query(&[("sha256", sha.as_str())]));
            let started = std::time::Instant::now();
            let Some(resp) = self.send(request, "sha precheck") else {
                continue;
            };
            let status = resp.status();
            if !self.answered(status, "sha precheck") {
                tracing::debug!(
                    keys = chunk.len(),
                    status = status.as_u16(),
                    "sha precheck: no answer"
                );
                continue;
            }
            let before = out.len();
            match resp.bytes() {
                Ok(body) => self.collect_sha_hits(chunk, &body, &mut out),
                Err(e) => tracing::debug!(error = %e, "sha precheck: unreadable lookup answer"),
            }
            let hits = out.len() - before;
            SKIPS.fetch_add(hits as u64, Ordering::Relaxed);
            tracing::info!(
                event = "remote.lookup",
                kind = "sha256",
                keys = chunk.len(),
                hits,
                misses = chunk.len() - hits,
                status = status.as_u16(),
                ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            );
        }
        out
    }

    /// Fold one digest batch answer into `out`, keyed by the answer's own
    /// `sha256`, which must be one that was asked.
    fn collect_sha_hits(&self, asked: &[String], body: &[u8], out: &mut HashMap<String, Standing>) {
        let Some(items) = answer_items(body) else {
            tracing::debug!("sha precheck: unreadable lookup answer");
            return;
        };
        for item in items {
            let Ok(mut answer) = serde_json::from_str::<Record>(item.get()) else {
                continue;
            };
            let Some(sha) = answer.sha256.take().map(|d| d.to_ascii_lowercase()) else {
                continue;
            };
            if !asked.iter().any(|a| a.eq_ignore_ascii_case(&sha)) {
                tracing::warn!(event = "remote.lookup", mismatched = %sha, "answer names a digest that was not asked; ignored");
                continue;
            }
            let standing = self.judge(answer, Lookup::DownloadedBytes);
            if standing.skips_analysis() {
                out.insert(sha, standing);
            }
        }
    }

    /// Batch PURL negotiation: which of these dependency PURLs does hopper hold a
    /// standing verdict for? Returns `purl → hit` for every entry that satisfies
    /// the same two rules as [`Self::standing`] — the caller skips the FETCH as
    /// well as the analysis for those, which the per-sha precheck cannot (a
    /// registry PURL's content sha is only learned by downloading it). An answer
    /// with no usable sha is dropped — the fetch edge (`source → content sha`)
    /// must stay recordable — so the dependency falls through to a normal fetch.
    /// Fail-open everywhere; an open breaker stops the batch.
    pub fn purls(&self, purls: &[String]) -> HashMap<String, PurlHit> {
        let mut out = HashMap::new();
        if purls.is_empty() {
            return out;
        }
        let Some(http) = self.http() else {
            return out;
        };
        for chunk in purls.chunks(PURL_BATCH) {
            // Re-checked per chunk: a trip here or on another thread stops the
            // rest, which would only time out too.
            if !self.breaker.allows(crate::civil::now_secs()) {
                break;
            }
            PURL_CHECKS.fetch_add(chunk.len() as u64, Ordering::Relaxed);
            let lookup = http.get(&self.lookup_url).timeout(self.timeout());
            let request = chunk
                .iter()
                .fold(lookup, |req, purl| req.query(&[("purl", purl.as_str())]));
            let Some(resp) = self.send(request, "purl precheck") else {
                continue;
            };
            if !self.answered(resp.status(), "purl precheck") {
                continue;
            }
            match resp.bytes() {
                Ok(body) => self.collect_hits(chunk, &body, &mut out),
                Err(e) => tracing::debug!(error = %e, "purl precheck: unreadable lookup answer"),
            }
        }
        out
    }

    /// Fold one batch answer into `out`. One purl answers with one object,
    /// several with a list in the order asked; both are tolerated, and the
    /// answer's own `purl` field is preferred over its position. Each item is
    /// parsed on its own, so one malformed answer costs only its dependency.
    fn collect_hits(&self, asked: &[String], body: &[u8], out: &mut HashMap<String, PurlHit>) {
        let Some(items) = answer_items(body) else {
            tracing::debug!("purl precheck: unreadable lookup answer");
            return;
        };
        for (i, item) in items.iter().enumerate() {
            let Ok(mut answer) = serde_json::from_str::<Record>(item.get()) else {
                continue;
            };
            let (sha, purl) = (answer.sha256.take(), answer.purl.take());
            let standing = self.judge(answer, Lookup::PackageCoordinate);
            if !standing.skips_analysis() {
                continue;
            }
            let Some(sha) =
                sha.filter(|d| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()))
            else {
                continue;
            };
            let Some(purl) = purl.or_else(|| asked.get(i).cloned()) else {
                continue;
            };
            PURL_SKIPS.fetch_add(1, Ordering::Relaxed);
            out.insert(
                purl,
                PurlHit {
                    sha: sha.to_ascii_lowercase(),
                    standing,
                },
            );
        }
    }
}

/// The verdict in one `/v1/analyze` answer line, and the digest it names:
/// isotope13's judgment of an artifact it was asked to analyze, so it stands
/// as long as it is an analyzed verdict with a level.
///
/// # Errors
/// An unreadable line, or an answer that is not an analyzed verdict.
pub fn analyzed_verdict(line: &str) -> Result<(String, Verdict), String> {
    #[derive(serde::Deserialize)]
    struct Answer {
        cause: Option<String>,
        #[serde(flatten)]
        record: Record,
    }
    let answer: Answer =
        serde_json::from_str(line).map_err(|e| format!("unreadable analysis answer: {e}"))?;
    let mut rec = answer.record;
    if rec.status != Some(Status::Analyzed) || rec.fires_at == Level::Manual {
        return Err(format!(
            "isotope13 could not analyze it ({}{})",
            rec.status.map_or("", Status::as_str),
            answer.cause.map(|c| format!(": {c}")).unwrap_or_default()
        ));
    }
    let sha = rec.sha256.take().unwrap_or_default().to_ascii_lowercase();
    Ok((sha, rec.into()))
}

/// One answer as a list of items: several keys answer with a list, one with an
/// object. `None` when the body is not JSON of either shape.
fn answer_items(body: &[u8]) -> Option<Vec<&serde_json::value::RawValue>> {
    if body.trim_ascii_start().starts_with(b"[") {
        serde_json::from_slice(body).ok()
    } else {
        serde_json::from_slice(body).ok().map(|one| vec![one])
    }
}

/// isotope13's policy, pure for the same reason as [`standing_of`]: an
/// analyzed verdict is adopted — the service's judgment is what an enriched
/// scan asked for — except a clean one older than `max_age_s`, which is
/// analyzed again rather than trusted past its freshness. Anything not
/// `analyzed`, or with no level, is analyzed here.
/// Most a record's `analyzed_at` may lie ahead of this clock (skew between
/// hosts) and still count as when it was analyzed.
const MAX_CLOCK_SKEW_S: u64 = 300;

/// Whether a clean verdict analyzed at `at` is within `max_age_s` of `now`. A
/// time further ahead than clock skew explains is not a time at all: read as
/// "now", it would keep the verdict fresh until that date.
fn fresh_at(at: u64, max_age_s: u64, now: u64) -> bool {
    at <= now.saturating_add(MAX_CLOCK_SKEW_S) && now.saturating_sub(at) <= max_age_s
}

fn service_standing(rec: Record, max_age_s: u64, now: u64) -> Standing {
    if rec.status != Some(Status::Analyzed) || rec.fires_at == Level::Manual {
        return Standing::Analyze;
    }
    if rec.fires_at == Level::Clean {
        let fresh = rec
            .analyzed_at
            .as_deref()
            .and_then(parse_rfc3339_epoch)
            .is_some_and(|at| fresh_at(at, max_age_s, now));
        if !fresh {
            return Standing::Analyze;
        }
    }
    Standing::Adopt(rec.into())
}

/// The policy, pure so the tests can hold it still: rule 1 (same analyzer,
/// verdict adopted) or, for package coordinates only, rule 2 (benign and fresh).
fn standing_of(
    rec: Record,
    my_traits: Option<&str>,
    max_age_s: u64,
    now: u64,
    lookup: Lookup,
) -> Standing {
    // Rule 1: same analyzer already judged these bytes. A verdict is only a
    // dedupe key when it EXISTS — fires_at is null for a record that was
    // never classified, and traits equality on an unclassified record would
    // skip an analysis that never happened.
    if rec.fires_at != Level::Manual
        && let (Some(mine), Some(theirs)) = (my_traits, rec.traits_version.as_deref())
        && !mine.is_empty()
        && mine == theirs
    {
        return Standing::Adopt(rec.into());
    }
    // Downloaded bytes need current-analyzer coverage. Returning SkipBenign
    // here would discard the payload report, lose its next-hop references,
    // and make the caller report "analysis unavailable" for a successful fetch.
    if lookup == Lookup::DownloadedBytes {
        return Standing::Analyze;
    }

    // Rule 2: benign, and fresh enough that staleness is bounded. Deliberately
    // NOT adopted: a different analyzer's judgment is not this scan's finding.
    // It can hide nothing — the rule requires a clean record.
    if rec.fires_at != Level::Clean {
        return Standing::Analyze;
    }
    let Some(at) = rec.analyzed_at.as_deref().and_then(parse_rfc3339_epoch) else {
        return Standing::Analyze;
    };
    if fresh_at(at, max_age_s, now) {
        Standing::SkipBenign
    } else {
        Standing::Analyze
    }
}

/// `(lookups attempted, analyses skipped)` since process start, for a caller's
/// summary line.
pub fn counters() -> (u64, u64) {
    (
        CHECKS.load(Ordering::Relaxed),
        SKIPS.load(Ordering::Relaxed),
    )
}

/// `(purl_checks, purl_skips)` lifetime counters for the batch negotiation.
pub fn purl_counters() -> (u64, u64) {
    (
        PURL_CHECKS.load(Ordering::Relaxed),
        PURL_SKIPS.load(Ordering::Relaxed),
    )
}

/// Parse an RFC 3339 UTC timestamp — `2026-08-23T23:00:44Z`, optionally with
/// fractional seconds, which are ignored — to a Unix epoch. Offsets other than
/// `Z` are rejected: hopper emits UTC, and a wrong-but-plausible parse here
/// would silently misjudge freshness.
fn parse_rfc3339_epoch(s: &str) -> Option<u64> {
    let s = s.strip_suffix('Z')?;
    let (whole, fraction) = s.split_once('.').unwrap_or((s, "0"));
    if fraction.is_empty() || !fraction.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let b = whole.as_bytes();
    if b.len() != 19
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
    {
        return None;
    }
    let num = |r: std::ops::Range<usize>| -> Option<u32> {
        let field = whole.get(r)?;
        // Digits only: `str::parse` would also take a sign.
        if !field.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        field.parse().ok()
    };
    crate::civil::utc_epoch(
        num(0..4)?,
        num(5..7)?,
        num(8..10)?,
        num(11..13)?,
        num(14..16)?,
        num(17..19)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fixed_points_and_what_the_engine_writes() {
        for (s, want) in [
            ("1970-01-01T00:00:00Z", 0),
            ("2000-02-29T12:00:00Z", 951_825_600),
            ("2026-08-23T23:00:44Z", 1_787_526_044),
            ("2026-08-23T23:00:44.123456Z", 1_787_526_044),
            ("2026-12-31T23:59:59Z", 1_798_761_599),
            ("2100-03-01T00:00:00Z", 4_107_542_400),
        ] {
            assert_eq!(parse_rfc3339_epoch(s), Some(want), "{s}");
        }
        // The writer this parser inverts, not a copy of it.
        let before = crate::civil::now_secs();
        let parsed = parse_rfc3339_epoch(&crate::engine::now_rfc3339()).expect("parses");
        assert!(
            (before..=crate::civil::now_secs()).contains(&parsed),
            "{parsed} vs {before}"
        );
    }

    #[test]
    fn rejects_offsets_and_garbage() {
        for s in [
            "2026-08-23T23:00:44+02:00", // non-UTC offset: refuse, don't misjudge
            "2026-08-23 23:00:44Z",      // space separator
            "not-a-time",
            "",
            "2026-13-01T00:00:00Z",        // month 13
            "2026-08-23T23:00:44junkZ",    // junk between seconds and Z
            "2026-08-23T23:00:44.Z",       // empty fraction
            "2026-08-23T23:00:44.12x4Z",   // non-digit fraction
            "2026-+8-23T23:00:44Z",        // a sign is not a digit
            "2026-08-23T23:00:44.123+00Z", // offset hidden in the fraction
        ] {
            assert_eq!(parse_rfc3339_epoch(s), None, "{s}");
        }
    }

    /// Closed, open after the limit, half-open after the cooldown: exactly one
    /// probe goes out, a success closes the breaker, a failure re-arms the wait.
    #[test]
    fn the_breaker_opens_probes_once_and_closes_on_success() {
        let breaker = Breaker::default();
        let t0 = 1_000_000;
        for i in 1..BREAKER_LIMIT {
            assert!(breaker.allows(t0));
            assert!(!breaker.failed(t0, "test", &"refused"), "open after {i}");
        }
        assert!(breaker.failed(t0, "test", &"refused"), "the limit opens it");
        assert!(!breaker.allows(t0 + 1), "open: nothing goes out");
        let later = t0 + BREAKER_COOLDOWN_SECS;
        assert!(breaker.allows(later), "cooled down: one probe");
        assert!(!breaker.allows(later), "and only one");
        // The probe failed: the cooldown starts over from the probe.
        assert!(breaker.failed(later, "test", &"refused"));
        assert!(!breaker.allows(later + BREAKER_COOLDOWN_SECS - 1));
        assert!(breaker.allows(later + BREAKER_COOLDOWN_SECS));
        // That probe succeeded: closed again, for everyone.
        breaker.succeeded();
        assert!(breaker.allows(later + BREAKER_COOLDOWN_SECS));
        assert!(breaker.allows(later + BREAKER_COOLDOWN_SECS));
    }

    /// A breaker another caller opened stops the batch before its first chunk.
    #[test]
    fn an_open_breaker_sends_no_purl_lookups() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let precheck = Precheck::new(
            &format!("http://{}", listener.local_addr().expect("addr")),
            Duration::from_secs(86_400),
        )
        .expect("a hopper");
        for _ in 0..BREAKER_LIMIT {
            precheck
                .breaker
                .failed(crate::civil::now_secs(), "test", &"refused");
        }
        let purls: Vec<String> = (0..120).map(|i| format!("pkg:npm/p{i}@1")).collect();
        assert!(precheck.purls(&purls).is_empty());
        assert!(
            matches!(listener.accept(), Err(e) if e.kind() == std::io::ErrorKind::WouldBlock),
            "an open breaker must keep the batch off the wire"
        );
    }

    #[test]
    fn a_batch_answer_pairs_by_purl_and_drops_unusable_items() {
        let precheck =
            Precheck::new("http://unused", Duration::from_secs(86_400)).expect("a hopper");
        let fresh = crate::engine::now_rfc3339();
        let sha = "AB".repeat(32);
        let body = format!(
            r#"[{{"purl":"pkg:npm/b@1","sha256":"{sha}","fires_at":-1,"analyzed_at":"{fresh}"}},
                {{"sha256":"{sha}","fires_at":-1,"analyzed_at":"{fresh}"}},
                {{"purl":"pkg:npm/c@1","sha256":"short","fires_at":-1,"analyzed_at":"{fresh}"}},
                {{"purl":"pkg:npm/d@1","findings":[{{"id":"no-crit"}}]}}]"#
        );
        let asked: Vec<String> = ["pkg:npm/a@1", "pkg:npm/x@1", "pkg:npm/c@1", "pkg:npm/d@1"]
            .map(String::from)
            .to_vec();
        let mut out = HashMap::new();
        precheck.collect_hits(&asked, body.as_bytes(), &mut out);
        // Named by its own `purl`, then by position; an unusable sha and a
        // malformed item each drop only themselves.
        assert_eq!(out.len(), 2, "{:?}", out.keys().collect::<Vec<_>>());
        assert_eq!(out["pkg:npm/b@1"].sha, sha.to_ascii_lowercase());
        assert!(matches!(out["pkg:npm/x@1"].standing, Standing::SkipBenign));
    }

    #[test]
    fn policy_two_rules() {
        let rec = |fires: Level, tv: Option<&str>, at: Option<&str>| Record {
            fires_at: fires,
            analyzed_at: at.map(String::from),
            traits_version: tv.map(String::from),
            reason: None,
            findings: Vec::new(),
            ..Record::default()
        };
        let now = parse_rfc3339_epoch("2026-08-24T00:00:00Z").unwrap();
        let week = 7 * 86_400;
        let fresh = Some("2026-08-23T00:00:00Z"); // 1 day old
        let stale = Some("2026-08-01T00:00:00Z"); // 23 days old
        let standing = |r: Record, mine: Option<&str>| {
            standing_of(r, mine, week, now, Lookup::PackageCoordinate)
        };

        // Rule 1: the same analyzer's verdict is adopted, hostile included, at
        // any age — it is the verdict this build would have computed.
        let adopted = standing(rec(Level::At(3), Some("b8c1c"), stale), Some("b8c1c"));
        assert!(
            matches!(&adopted, Standing::Adopt(v) if v.fires_at == Level::At(3)),
            "expected the stored verdict, got {adopted:?}"
        );
        // ...but never on a record with no verdict at all.
        assert!(matches!(
            standing(rec(Level::Manual, Some("b8c1c"), fresh), Some("b8c1c")),
            Standing::Analyze
        ));
        // A non-benign verdict from a DIFFERENT analyzer is neither adopted nor
        // skipped: it is not ours to report, and rule 2 does not cover it.
        assert!(matches!(
            standing(rec(Level::At(3), Some("f6eaa"), fresh), Some("b8c1c")),
            Standing::Analyze
        ));
        // Empty-string traits (the member-row gap) must not match anything.
        assert!(matches!(
            standing(rec(Level::At(3), Some(""), fresh), Some("")),
            Standing::Analyze
        ));

        // Rule 2: benign and fresh skips the work across analyzers, and adopts
        // nothing — there is no finding to carry.
        assert!(matches!(
            standing(rec(Level::Clean, Some("f6eaa"), fresh), Some("b8c1c")),
            Standing::SkipBenign
        ));
        // ...including when the stored row has no traits at all.
        assert!(matches!(
            standing(rec(Level::Clean, None, fresh), Some("b8c1c")),
            Standing::SkipBenign
        ));
        // ...but not stale, and not without a timestamp.
        assert!(matches!(
            standing(rec(Level::Clean, None, stale), Some("b8c1c")),
            Standing::Analyze
        ));
        assert!(matches!(
            standing(rec(Level::Clean, None, None), Some("b8c1c")),
            Standing::Analyze
        ));
        // A benign verdict from our own analyzer is adopted, not merely skipped:
        // "clean, and we can say so" outranks "clean enough not to re-run".
        assert!(matches!(
            standing(rec(Level::Clean, Some("b8c1c"), fresh), Some("b8c1c")),
            Standing::Adopt(_)
        ));
    }

    /// The adopted verdict carries what a reader needs: the level, the sentence,
    /// and the findings behind it.
    #[test]
    fn an_adopted_verdict_keeps_the_findings() {
        let rec: Record = serde_json::from_str(
            r#"{"sha256":"ab","fires_at":2,"traits_version":"b8c1c",
                "analyzed_at":"2026-08-23T23:00:44Z","reason":"steals credentials",
                "findings":[{"id":"objectives/exfil::env","crit":5},
                            {"id":"feed/osv","desc":"cited by OSV","crit":4}]}"#,
        )
        .expect("parse");
        let now = parse_rfc3339_epoch("2026-08-24T00:00:00Z").unwrap();
        let v = match standing_of(rec, Some("b8c1c"), 7 * 86_400, now, Lookup::DownloadedBytes) {
            Standing::Adopt(v) => Some(v),
            _ => None,
        }
        .expect("the same analyzer must adopt");
        assert_eq!(v.fires_at, Level::At(2));
        assert_eq!(v.reason.as_deref(), Some("steals credentials"));
        assert_eq!(v.findings.len(), 2);
        assert_eq!(v.findings[0].crit, 5);
        assert_eq!(v.findings[1].desc.as_deref(), Some("cited by OSV"));
    }

    #[test]
    fn downloaded_bytes_require_current_analyzer_even_when_corpus_is_fresh_and_clean() {
        let now = parse_rfc3339_epoch("2026-10-16T00:00:00Z").unwrap();
        let record = |traits: Option<&str>| Record {
            fires_at: Level::Clean,
            analyzed_at: Some("2026-10-15T00:00:00Z".to_string()),
            traits_version: traits.map(String::from),
            reason: None,
            findings: Vec::new(),
            ..Record::default()
        };
        for traits in [Some("older"), Some(""), None] {
            assert!(matches!(
                standing_of(
                    record(traits),
                    Some("local"),
                    30 * 86_400,
                    now,
                    Lookup::DownloadedBytes
                ),
                Standing::Analyze
            ));
            assert!(matches!(
                standing_of(
                    record(traits),
                    Some("local"),
                    30 * 86_400,
                    now,
                    Lookup::PackageCoordinate
                ),
                Standing::SkipBenign
            ));
        }
        assert!(matches!(
            standing_of(
                record(Some("local")),
                Some("local"),
                30 * 86_400,
                now,
                Lookup::DownloadedBytes
            ),
            Standing::Adopt(_)
        ));
    }

    fn isotope13() -> Precheck {
        Precheck {
            lookup_url: "https://unused.invalid/v1/lookup".into(),
            max_age: Duration::from_secs(30 * 86_400),
            breaker: Breaker::default(),
            backend: Backend::Isotope13(
                cleave::enriched::Token::parse("i13_test_token").expect("a token"),
            ),
        }
    }

    /// isotope13 is the judge an enriched scan asked for: an analyzed verdict
    /// stands whatever produced it, but a stale clean one, an unanalyzed one,
    /// or one with no level is analyzed here.
    #[test]
    fn isotope13_adopts_analyzed_verdicts_but_not_stale_clean_ones() {
        let now = parse_rfc3339_epoch("2026-10-22T00:00:00Z").unwrap();
        let month = 30 * 86_400;
        let rec = |status: &str, fires: Level, at: &str| Record {
            fires_at: fires,
            analyzed_at: Some(at.to_string()),
            status: serde_json::from_value(status.into()).unwrap(),
            ..Record::default()
        };
        let fresh = "2026-10-21T00:00:00Z";
        let stale = "2026-08-01T00:00:00Z";
        assert!(matches!(
            service_standing(rec("analyzed", Level::At(3), stale), month, now),
            Standing::Adopt(v) if v.fires_at == Level::At(3)
        ));
        assert!(matches!(
            service_standing(rec("analyzed", Level::Clean, fresh), month, now),
            Standing::Adopt(v) if v.fires_at == Level::Clean
        ));
        for analyze in [
            rec("analyzed", Level::Clean, stale),
            rec("analyzed", Level::Manual, fresh),
            rec("unanalyzed", Level::At(3), fresh),
            rec("unavailable", Level::Clean, fresh),
            Record {
                fires_at: Level::At(3),
                ..Record::default()
            },
        ] {
            assert!(matches!(
                service_standing(analyze, month, now),
                Standing::Analyze
            ));
        }
    }

    /// An answer is paired by its own digest, which must be one that was asked:
    /// a verdict about other bytes can never stand in for a file.
    #[test]
    fn a_digest_answer_must_name_a_digest_that_was_asked() {
        let precheck = isotope13();
        let fresh = crate::engine::now_rfc3339();
        let (a, b, other) = ("a".repeat(64), "b".repeat(64), "c".repeat(64));
        let body = format!(
            r#"[{{"status":"analyzed","sha256":"{}","fires_at":3,"analyzed_at":"{fresh}"}},
                {{"status":"unanalyzed","sha256":"{b}"}},
                {{"status":"analyzed","sha256":"{other}","fires_at":3,"analyzed_at":"{fresh}"}}]"#,
            a.to_uppercase()
        );
        let mut out = HashMap::new();
        precheck.collect_sha_hits(&[a.clone(), b], body.as_bytes(), &mut out);
        assert_eq!(out.keys().collect::<Vec<_>>(), vec![&a]);
        assert!(matches!(&out[&a], Standing::Adopt(v) if v.fires_at == Level::At(3)));
    }

    /// A refused token or an exhausted quota stops the asking at once; any
    /// other failure is one batch analyzed here.
    #[test]
    fn a_refusal_from_isotope13_stops_the_asking() {
        for status in [401, 403, 429] {
            let precheck = isotope13();
            let code = reqwest::StatusCode::from_u16(status).unwrap();
            assert!(!precheck.answered(code, "test"));
            assert!(
                !precheck.breaker.allows(crate::civil::now_secs()),
                "{status} opens the breaker"
            );
        }
        let precheck = isotope13();
        assert!(!precheck.answered(reqwest::StatusCode::BAD_GATEWAY, "test"));
        assert!(
            precheck.breaker.allows(crate::civil::now_secs()),
            "a 502 is one failed batch"
        );
        assert!(precheck.answered(reqwest::StatusCode::OK, "test"));

        // Hopper's refusals keep their old meaning: analyze, keep asking.
        let hopper = Precheck::new("http://unused", Duration::from_secs(1)).unwrap();
        assert!(!hopper.answered(reqwest::StatusCode::UNAUTHORIZED, "test"));
        assert!(hopper.breaker.allows(crate::civil::now_secs()));
    }

    #[test]
    fn an_analysis_answer_is_a_verdict_only_when_analyzed() {
        let sha = "D".repeat(64);
        let (got, v) = analyzed_verdict(&format!(
            r#"{{"status":"analyzed","sha256":"{sha}","fires_at":-1,"engine_version":"2.13.0","analyzed_at":"2026-10-22T00:00:00Z","findings":[]}}"#
        ))
        .unwrap();
        assert_eq!(got, sha.to_ascii_lowercase());
        assert_eq!(v.fires_at, Level::Clean);
        assert_eq!(v.engine_version.as_deref(), Some("2.13.0"));

        let err = analyzed_verdict(r#"{"status":"unavailable","cause":"no workers"}"#).unwrap_err();
        assert!(err.contains("unavailable: no workers"), "{err}");
        assert!(
            analyzed_verdict(r#"{"status":"analyzed"}"#).is_err(),
            "no level"
        );
        assert!(analyzed_verdict("not json").is_err());
    }

    #[test]
    fn policy_reads_only_the_two_fields() {
        // The response may carry fields we have never heard of.
        let rec: Record = serde_json::from_str(
            r#"{"sha256":"ab","fires_at":-1,"engine_version":"2.8.0",
                "traits_version":"b8c1c","analyzed_at":"2026-08-23T23:00:44Z",
                "findings":[],"brand_new_field":true}"#,
        )
        .expect("parse");
        assert_eq!(rec.fires_at, Level::Clean);
        assert!(rec.analyzed_at.is_some());
        assert!(rec.findings.is_empty());
    }
}
