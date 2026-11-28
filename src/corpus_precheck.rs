//! A shared skip for fetched-dependency analysis, backed by hopper's corpus.
//!
//! `crate::analysis_cache` memoizes dependency analyses per host, keyed by
//! ruleset version, so a ruleset change invalidates every host's cache at
//! once and each host re-analyzes the same dependencies. Hopper holds the one
//! cache every scanner shares, so this module asks it first, before analyzing
//! a fetched payload or resolving a package coordinate:
//!
//!   1. SAME ANALYZER: the stored verdict was produced under this process's
//!      traits (`traits_version` equals our 5-char traits commit). Analyzing
//!      the same bytes under the same rules reaches the same verdict, so this
//!      adopts any verdict, hostile included.
//!   2. BENIGN AND FRESH: `fires_at` is `-1` (clean) and the record was
//!      analyzed within `DEFAULT_MAX_AGE_DAYS`, whatever analyzer produced
//!      it. This applies only to package coordinates, before fetching: once
//!      bytes are in hand, another analyzer's clean verdict cannot stand in
//!      for analysis under the current rules. It keeps working while
//!      scanners run different trait versions, because dependency universes
//!      are overwhelmingly benign.
//!
//! Anything neither rule covers — not found, not fresh, hostile under a
//! different analyzer, unreachable — falls through to a normal analysis:
//! fail-open, never fail-closed.
//!
//! Armed whenever `--hopper` is: `configure` runs wherever an uploader is
//! built, so the hopper a process files to is the hopper it asks, with no
//! second setting to keep in sync. No `--hopper`, no precheck. Requests carry
//! the process's hopper bearer token ([`crate::hopper::token`]).
//!
//! # Atomdrift Uplink
//!
//! A scan that asks Uplink (any but `--local`) arms the same precheck against Uplink's API
//! instead (`configure_uplink`): with the subscription token on a paid plan,
//! with no credential otherwise. The wire is the same `/v1/lookup`; what
//! changes is whose judgment it is. Hopper is a corpus of other runs, so only
//! the same analyzer's verdict is ours. Uplink is the service the scan asked
//! to judge for it, so an analyzed verdict stands whatever produced it — with
//! one bound kept from rule 2: a clean verdict older than the max age is
//! analyzed again rather than trusted.
//!
//! Uplink not answering is never a scan's problem: local analysis is always a
//! complete answer. An unreachable or failing service is not asked again for
//! the rest of the run, and nothing is said of it. A spent quota (`429`) is
//! said once, and Uplink is left alone until the `Retry-After` it gave has
//! passed, in this run and in later ones.

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

/// The longest a `Retry-After` may hold Uplink lookups off: its quotas turn
/// over on the clock hour, so nothing honest asks for longer.
const QUOTA_HOLD_MAX_SECS: u64 = 3600;

/// Per-request ceiling on a lookup. It sits on the analysis path, so a slow
/// replica must cost seconds, not the pipeline.
const LOOKUP_TIMEOUT: Duration = Duration::from_secs(2);

/// PURLs per `/v1/lookup` request: hopper's documented cap.
const PURL_BATCH: usize = 50;

/// How fresh a benign verdict must be to stand in for a re-analysis.
///
/// This window is not the safety net against a benign verdict going bad:
/// hopper re-analyzes a dependency a threat feed cites whatever this cache
/// holds, and makes every row re-analysis-eligible after the same 30 days,
/// resetting `analyzed_at` when it does. A verdict reaches this age only if
/// every refresh left it alone, so the remaining exposure is a detector
/// improvement on an uncited dependency, which the periodic re-analysis
/// catches up with. A package whose dependency records change is not
/// refreshed by anything here.
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

    /// Open the breaker until `until`, in Unix seconds, whatever the count:
    /// the backend said no, or is not there, and every ask before then would
    /// hear the same. `u64::MAX` is the rest of the run. Returns whether this
    /// call opened it, so the reason is said once.
    fn hold(&self, until: u64) -> bool {
        self.opened_at.store(
            until.saturating_sub(BREAKER_COOLDOWN_SECS),
            Ordering::Relaxed,
        );
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
                what,
                %error,
                failures = BREAKER_LIMIT,
                cooldown_secs = BREAKER_COOLDOWN_SECS,
                "consecutive transport failures; pausing lookups"
            );
        }
        true
    }
}

/// The hopper this process asks, and how it weighs the answers.
#[derive(Debug)]
pub(crate) struct Precheck {
    lookup_url: String,
    max_age: Duration,
    breaker: Breaker,
    backend: Backend,
    /// Where a spent quota's end is remembered between runs, so the next run
    /// does not ask only to be refused again. `None` remembers nothing.
    quota_file: Option<std::path::PathBuf>,
}

/// Who answers, which decides how it is authenticated and whose judgment an
/// answer is.
#[derive(Debug)]
enum Backend {
    /// The hopper this process submits to. Only a same-analyzer verdict is ours.
    Hopper,
    /// Uplink's API, with a subscriber's token or with no credential. Its
    /// analyzed verdicts are the answer.
    Uplink(Option<cleave::feed::Token>),
}

/// The armed precheck.
///
/// Process-wide on purpose: every fetch path in the process consults it, and
/// none of them is handed configuration. Written only by [`configure`] and
/// [`configure_uplink`], which run at startup before any scan; each replaces
/// what was there, so the last backend a process was pointed at is the one
/// it asks. Read only through [`armed`], which hands out a snapshot: a
/// precheck in use is never mutated, only replaced.
static ARMED: RwLock<Option<Arc<Precheck>>> = RwLock::new(None);

/// [`DEFAULT_MAX_AGE_DAYS`] as a duration.
const DEFAULT_MAX_AGE: Duration =
    Duration::from_secs(DEFAULT_MAX_AGE_DAYS * crate::civil::SECS_PER_DAY.unsigned_abs());

/// Arm the precheck against Uplink's API, for a scan that asks it:
/// the subscribers' host with `token`, the open one without. A later
/// [`configure`] (`--hopper`) replaces it: the hopper a process submits to is
/// still the one it asks.
pub(crate) fn configure_uplink(token: Option<cleave::feed::Token>) {
    let precheck = Precheck::uplink(token);
    let now = crate::civil::now_secs();
    if let Some(until) = precheck.remembered_quota_hold(now) {
        precheck.breaker.hold(until);
        tracing::debug!(
            event = "remote.disabled",
            reason = "quota",
            until,
            "Uplink's quota was spent by an earlier run; analyzing locally until it turns over"
        );
    }
    tracing::info!(
        event = "remote.armed",
        url = %precheck.lookup_url,
        tid = %precheck.tid(),
        max_age_days = DEFAULT_MAX_AGE_DAYS,
        "Uplink lookups enabled: known artifacts are answered without local analysis"
    );
    *ARMED.write().unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(precheck));
}

/// Arm the precheck against the hopper this process already talks to. Called
/// where the uploader is built — the one place every `--hopper` mode passes
/// through — so enablement follows `--hopper` with no second setting.
pub(crate) fn configure(hopper_base_url: &str) {
    let precheck = Precheck::new(hopper_base_url, DEFAULT_MAX_AGE);
    let mut armed = ARMED.write().unwrap_or_else(PoisonError::into_inner);
    if let Some(p) = &precheck
        && armed
            .as_ref()
            .is_none_or(|old| old.lookup_url != p.lookup_url)
    {
        tracing::info!(
            url = %p.lookup_url,
            max_age_days = DEFAULT_MAX_AGE_DAYS,
            "corpus precheck enabled: matching-analyzer verdicts may be reused; fresh benign package coordinates may skip fetching"
        );
    }
    *armed = precheck.map(Arc::new);
}

/// The armed precheck, or `None` when this process has no hopper.
pub(crate) fn armed() -> Option<Arc<Precheck>> {
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
/// artifact grades hostile; grading it against a caller's budget is the
/// caller's job, never this module's.
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
    pub(crate) const fn skips_analysis(&self) -> bool {
        !matches!(self, Self::Analyze)
    }
}

/// A dependency the batch PURL negotiation answered for: the content sha that
/// records the fetch edge, and what the corpus lets us skip for it — never
/// [`Standing::Analyze`].
#[derive(Debug, Clone)]
pub(crate) struct PurlHit {
    pub(crate) sha: String,
    pub(crate) standing: Standing,
}

/// This process's traits commit, truncated to the 5 characters hopper stores
/// in `traits_version`, so string equality against a stored record means
/// "same analyzer".
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
        // `--hopper` may be a comma list, replica first: lookups belong on the
        // first entry, which is where a cheap read should land.
        let base = crate::hopper::endpoints(hopper_base_url)
            .into_iter()
            .next()?;
        Some(Self {
            lookup_url: format!("{base}/v1/lookup"),
            max_age,
            breaker: Breaker::default(),
            backend: Backend::Hopper,
            quota_file: None,
        })
    }

    /// A precheck against Uplink: the subscribers' host with `token`, the
    /// open one without. The host follows the credential, so a token is sent
    /// to [`crate::remote::API_URL`] and nowhere else.
    fn uplink(token: Option<cleave::feed::Token>) -> Self {
        // The two are counted apart: by organization, and by address.
        let (api, quota) = if token.is_some() {
            (crate::remote::API_URL, "uplink-quota")
        } else {
            (crate::remote::OPEN_API_URL, "uplink-quota-anonymous")
        };
        Self {
            lookup_url: format!("{api}/v1/lookup"),
            max_age: DEFAULT_MAX_AGE,
            breaker: Breaker::default(),
            backend: Backend::Uplink(token),
            quota_file: crate::cache_dir().map(|dir| dir.join(quota)),
        }
    }

    /// When the quota an earlier run spent turns over, if that is still ahead
    /// of `now`. Bounded like a fresh `Retry-After`, so a damaged or planted
    /// file can cost an hour of lookups and no more.
    fn remembered_quota_hold(&self, now: u64) -> Option<u64> {
        let text = std::fs::read_to_string(self.quota_file.as_ref()?).ok()?;
        let until: u64 = text.trim().parse().ok()?;
        (until > now).then(|| until.min(now.saturating_add(QUOTA_HOLD_MAX_SECS)))
    }

    /// Whether this precheck asks Uplink rather than hopper.
    pub(crate) fn is_uplink(&self) -> bool {
        matches!(self.backend, Backend::Uplink(_))
    }

    /// Per-request ceiling: tight for a hopper on the operator's own network,
    /// roomier for a public API across the internet.
    fn timeout(&self) -> Duration {
        match self.backend {
            Backend::Hopper => LOOKUP_TIMEOUT,
            Backend::Uplink(_) => crate::remote::LOOKUP_TIMEOUT,
        }
    }

    /// The token's fingerprint, for logs; empty for hopper, whose credential
    /// is logged where it is loaded.
    fn tid(&self) -> String {
        match &self.backend {
            Backend::Hopper => String::new(),
            Backend::Uplink(None) => "anonymous".to_string(),
            Backend::Uplink(Some(token)) => token.fingerprint(),
        }
    }

    fn http(&self) -> Option<&'static reqwest::blocking::Client> {
        match self.backend {
            Backend::Hopper => crate::hopper::http(),
            Backend::Uplink(_) => crate::remote::http(),
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
                match self.backend {
                    Backend::Hopper => {
                        self.breaker.failed(crate::civil::now_secs(), what, &e);
                    }
                    // Not there for one ask is not there for the run, and the
                    // scan says nothing of it.
                    Backend::Uplink(_) => {
                        if self.breaker.hold(u64::MAX) {
                            tracing::debug!(
                                event = "remote.disabled",
                                reason = "unreachable",
                                error = %e,
                                what,
                                "Uplink did not answer; analyzing locally for the rest of the run"
                            );
                        }
                    }
                }
                None
            }
        }
    }

    fn authed(
        &self,
        request: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        match &self.backend {
            Backend::Hopper => crate::hopper::bearer(request, crate::hopper::token()),
            Backend::Uplink(None) => request,
            Backend::Uplink(Some(token)) => request.bearer_auth(token.secret()),
        }
    }

    /// Read the acquisition coordinate stored for these exact bytes in Hopper.
    pub(crate) fn artifact_purl(&self, sha: &str) -> Option<fletch::purl::Purl> {
        let base = self.lookup_url.strip_suffix("/v1/lookup")?;
        if self.is_uplink() || !self.breaker.allows(crate::civil::now_secs()) {
            return None;
        }
        let request = self
            .http()?
            .get(format!("{base}/v1/files/{sha}/provenance"))
            .timeout(self.timeout());
        let response = self
            .send(request, "artifact provenance")?
            .error_for_status()
            .ok()?;
        crate::provenance::artifact_purl(&response.bytes().ok()?, sha)
    }

    /// Whether to read this answer at all. Every non-200 means "analyze it".
    /// From Uplink it also means "stop asking": a spent quota until the
    /// `retry_after` seconds it named have passed, a refusal or a failing
    /// service for the rest of the run, because every later lookup would hear
    /// the same.
    fn answered(&self, status: reqwest::StatusCode, retry_after: Option<u64>, what: &str) -> bool {
        if status == reqwest::StatusCode::OK {
            return true;
        }
        if !self.is_uplink() {
            return false;
        }
        let (status, tid) = (status.as_u16(), self.tid());
        match status {
            429 => {
                let until = crate::civil::now_secs().saturating_add(
                    retry_after
                        .unwrap_or(BREAKER_COOLDOWN_SECS)
                        .min(QUOTA_HOLD_MAX_SECS),
                );
                if self.breaker.hold(until) {
                    tracing::info!(
                        event = "remote.disabled",
                        reason = "quota",
                        until,
                        %tid,
                        what,
                        "Uplink's quota is spent; analyzing locally until it turns over"
                    );
                    self.quota_spent(until);
                }
            }
            401 | 403 if self.breaker.hold(u64::MAX) => {
                tracing::warn!(
                    event = "remote.disabled",
                    reason = "refused",
                    status,
                    %tid,
                    what,
                    "Uplink refused the lookup; analyzing locally"
                );
            }
            500..=599 if self.breaker.hold(u64::MAX) => {
                tracing::debug!(
                    event = "remote.disabled",
                    reason = "unavailable",
                    status,
                    what,
                    "Uplink is failing; analyzing locally for the rest of the run"
                );
            }
            _ => {}
        }
        false
    }

    /// Remember that the quota is spent until `until`, and say so: once in
    /// this process, and not again in a later run that reads the file.
    fn quota_spent(&self, until: u64) {
        if let Some(path) = &self.quota_file
            && let Err(error) = std::fs::write(path, until.to_string())
        {
            tracing::debug!(path = %path.display(), %error, "cannot remember Uplink's quota hold");
        }
        static SAID: std::sync::Once = std::sync::Once::new();
        let notice = match self.backend {
            Backend::Uplink(Some(_)) => {
                "* Uplink lookup quota exceeded; falling back to offline mode."
            }
            _ => {
                "* Anonymous lookup quota exceeded; falling back to offline mode. For faster scans, sign in with 'atomscan login'."
            }
        };
        SAID.call_once(|| crate::engine::print_above_bar(|| eprintln!("{notice}")));
    }

    /// The policy for this backend's answers. A pending record — bytes held,
    /// never analyzed — has no verdict to stand on.
    fn judge(&self, rec: Record, lookup: Lookup) -> Standing {
        if rec.is_pending() {
            return Standing::Analyze;
        }
        match self.backend {
            Backend::Hopper => standing_of(
                rec,
                local_traits(),
                self.max_age.as_secs(),
                crate::civil::now_secs(),
                lookup,
            ),
            Backend::Uplink(_) => {
                service_standing(rec, self.max_age.as_secs(), crate::civil::now_secs())
            }
        }
    }

    /// What hopper's corpus already holds for these downloaded bytes: a verdict
    /// this build may adopt (same analyzer), or [`Standing::Analyze`]. A clean
    /// verdict from another analyzer does not cover the current rules. Any
    /// failure — unreachable, non-200, unparseable, or an answer about bytes
    /// other than `content_sha` — also means analyze.
    pub(crate) fn standing(&self, content_sha: &str) -> Standing {
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
        if !self.answered(resp.status(), retry_after(&resp), "corpus precheck") {
            return Standing::Analyze; // 404 unknown, 401/5xx — all "scan it".
        }
        let rec = match resp.json::<Record>() {
            Ok(rec) => rec,
            Err(e) => {
                tracing::debug!(error = %e, "corpus precheck: unreadable lookup answer; analyzing");
                return Standing::Analyze;
            }
        };
        if rec
            .sha256
            .as_deref()
            .is_none_or(|sha| !sha.eq_ignore_ascii_case(content_sha))
        {
            tracing::warn!(
                event = "remote.lookup",
                asked = content_sha,
                answered = rec.sha256.as_deref().unwrap_or(""),
                "answer does not name the digest asked; analyzing"
            );
            return Standing::Analyze;
        }
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
    /// other bytes can never stand in for this file. Also the digests the
    /// backend did answer for, with nothing that stands: what it does not know,
    /// as opposed to what was never asked or never answered. Fail-open
    /// everywhere; an open breaker stops the batch.
    pub(crate) fn shas(&self, shas: &[String]) -> (HashMap<String, Standing>, Vec<String>) {
        let (mut out, mut unknown) = (HashMap::new(), Vec::new());
        let Some(http) = self.http() else {
            return (out, unknown);
        };
        // Uplink answers a repeated `sha256` as a batch; hopper's lookup
        // reads only the first, so every further digest in a chunk would go
        // unanswered and be fetched and analyzed for nothing.
        let batch = if self.is_uplink() { PURL_BATCH } else { 1 };
        for chunk in shas.chunks(batch) {
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
            if !self.answered(status, retry_after(&resp), "sha precheck") {
                tracing::debug!(
                    keys = chunk.len(),
                    status = status.as_u16(),
                    "sha precheck: no answer"
                );
                continue;
            }
            let before = out.len();
            let read = match resp.bytes() {
                Ok(body) => self.collect_sha_hits(chunk, &body, &mut out),
                Err(e) => {
                    tracing::debug!(error = %e, "sha precheck: unreadable lookup answer");
                    false
                }
            };
            if read {
                unknown.extend(
                    chunk
                        .iter()
                        .filter(|sha| !out.contains_key(&sha.to_ascii_lowercase()))
                        .cloned(),
                );
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
        (out, unknown)
    }

    /// Fold one digest batch answer into `out`, keyed by the answer's own
    /// `sha256`, which must be one that was asked.
    /// Returns whether the answer could be read at all.
    fn collect_sha_hits(
        &self,
        asked: &[String],
        body: &[u8],
        out: &mut HashMap<String, Standing>,
    ) -> bool {
        let Some(items) = answer_items(body) else {
            tracing::debug!("sha precheck: unreadable lookup answer");
            return false;
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
        true
    }

    /// Batch PURL negotiation: which of these dependency PURLs does hopper hold a
    /// standing verdict for? Returns `purl → hit` for every entry that satisfies
    /// the same two rules as [`Self::standing`] — the caller skips the FETCH as
    /// well as the analysis for those, which the per-sha precheck cannot (a
    /// registry PURL's content sha is only learned by downloading it). An answer
    /// with no usable sha is dropped — the fetch edge (`source → content sha`)
    /// must stay recordable — so the dependency falls through to a normal fetch.
    /// Fail-open everywhere; an open breaker stops the batch.
    pub(crate) fn purls(&self, purls: &[String]) -> HashMap<String, PurlHit> {
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
            if !self.answered(resp.status(), retry_after(&resp), "purl precheck") {
                continue;
            }
            match resp.bytes() {
                Ok(body) => self.collect_hits(chunk, &body, &mut out),
                Err(e) => tracing::debug!(error = %e, "purl precheck: unreadable lookup answer"),
            }
        }
        out
    }

    /// Fold one batch answer into `out`, keyed by the purl as asked. A lone
    /// purl's answer is about it whatever spelling hopper echoes back (it
    /// canonicalizes); in a batch, unknown purls are omitted, so position means
    /// nothing and an answer counts only when it names a purl that was asked.
    /// Each item is parsed on its own, so one malformed answer costs only its
    /// dependency.
    fn collect_hits(&self, asked: &[String], body: &[u8], out: &mut HashMap<String, PurlHit>) {
        let Some(items) = answer_items(body) else {
            tracing::debug!("purl precheck: unreadable lookup answer");
            return;
        };
        for item in &items {
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
            let purl = match asked {
                [only] => only.clone(),
                _ => match purl.filter(|p| asked.contains(p)) {
                    Some(purl) => purl,
                    None => continue,
                },
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

/// One answer as a list of items: several keys answer with a list (hopper's
/// wrapped as `{"records": [...]}`), one with an object. `None` when the body
/// is not JSON of any of those shapes.
fn answer_items(body: &[u8]) -> Option<Vec<&serde_json::value::RawValue>> {
    #[derive(serde::Deserialize)]
    struct Batch<'a> {
        #[serde(borrow)]
        records: Vec<&'a serde_json::value::RawValue>,
    }
    if body.trim_ascii_start().starts_with(b"[") {
        return serde_json::from_slice(body).ok();
    }
    let one: &serde_json::value::RawValue = serde_json::from_slice(body).ok()?;
    match serde_json::from_str::<Batch<'_>>(one.get()) {
        Ok(batch) => Some(batch.records),
        Err(_) => Some(vec![one]),
    }
}

/// The whole seconds a refusal asks the caller to wait, when it says.
fn retry_after(resp: &reqwest::blocking::Response) -> Option<u64> {
    resp.headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// Uplink's policy, pure for the same reason as [`standing_of`]: an
/// analyzed verdict is adopted — the service's judgment is what a paid
/// scan asked for — except a clean one older than `max_age_s`, which is
/// analyzed again rather than trusted past its freshness. Anything not
/// `analyzed`, or with no level, is analyzed here.
fn service_standing(rec: Record, max_age_s: u64, now: u64) -> Standing {
    if rec.status != Some(Status::Analyzed) || rec.fires_at == Level::Manual {
        return Standing::Analyze;
    }
    if rec.fires_at == Level::Clean {
        let fresh = rec
            .analyzed_at
            .as_deref()
            .and_then(parse_rfc3339_epoch)
            .is_some_and(|at| crate::civil::fresh_at(at, max_age_s, now));
        if !fresh {
            return Standing::Analyze;
        }
    }
    Standing::Adopt(rec.into())
}

/// Rule 1: whether this analyzer — traits version `mine` — already judged the
/// bytes `rec` describes, so its verdict may stand in for an analysis.
///
/// A verdict is only a dedupe key when it EXISTS: `fires_at` is null for a
/// record that was never classified, and traits equality on an unclassified
/// record would skip an analysis that never happened. An unknown local
/// version matches nothing.
///
/// The model bundle is deliberately not part of the key: traits change every
/// few hours and models about weekly, so a model fix reaches reused verdicts
/// with the next traits change.
#[must_use]
pub fn same_analyzer(rec: &Record, mine: Option<&str>) -> bool {
    rec.fires_at != Level::Manual
        && mine.is_some_and(|mine| !mine.is_empty() && rec.traits_version.as_deref() == Some(mine))
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
    if same_analyzer(&rec, my_traits) {
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
    if crate::civil::fresh_at(at, max_age_s, now) {
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
pub(crate) fn parse_rfc3339_epoch(s: &str) -> Option<u64> {
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

    const DAY: u64 = crate::civil::SECS_PER_DAY.unsigned_abs();

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
            Duration::from_secs(DAY),
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
        let precheck = Precheck::new("http://unused", Duration::from_secs(DAY)).expect("a hopper");
        let fresh = crate::engine::now_rfc3339();
        let sha = "AB".repeat(32);
        let body = format!(
            r#"{{"records":[{{"purl":"pkg:npm/b@1","sha256":"{sha}","fires_at":-1,"analyzed_at":"{fresh}"}},
                {{"sha256":"{sha}","fires_at":-1,"analyzed_at":"{fresh}"}},
                {{"purl":"pkg:npm/c@1","sha256":"short","fires_at":-1,"analyzed_at":"{fresh}"}},
                {{"purl":"pkg:npm/d@1","findings":[{{"id":"no-crit"}}]}}]}}"#
        );
        let asked: Vec<String> = ["pkg:npm/a@1", "pkg:npm/x@1", "pkg:npm/c@1", "pkg:npm/d@1"]
            .map(String::from)
            .to_vec();
        let mut out = HashMap::new();
        precheck.collect_hits(&asked, body.as_bytes(), &mut out);
        // Hopper omits unknown purls from a batch, so position names nothing:
        // an answer about a purl nobody asked, or naming none, pairs with no
        // one. An unusable sha and a malformed item each drop only themselves.
        assert!(out.is_empty(), "{:?}", out.keys().collect::<Vec<_>>());
    }

    #[test]
    fn an_answer_is_keyed_by_the_purl_as_asked() {
        let precheck = Precheck::new("http://unused", Duration::from_secs(DAY)).expect("a hopper");
        let fresh = crate::engine::now_rfc3339();
        let sha = "ab".repeat(32);
        let answer = |purl: &str| {
            format!(r#"{{"purl":"{purl}","sha256":"{sha}","fires_at":-1,"analyzed_at":"{fresh}"}}"#)
        };
        // A batch echoes each purl as the caller spelled it.
        let asked = ["pkg:npm/a@1", "pkg:npm/b@1"].map(String::from).to_vec();
        let mut out = HashMap::new();
        let body = format!(r#"{{"records":[{}]}}"#, answer("pkg:npm/b@1"));
        precheck.collect_hits(&asked, body.as_bytes(), &mut out);
        assert_eq!(out.keys().collect::<Vec<_>>(), ["pkg:npm/b@1"]);
        // A lone purl comes back canonicalized; it still answers what was asked.
        let asked = vec!["npm/c@1".to_string()];
        let mut out = HashMap::new();
        precheck.collect_hits(&asked, answer("pkg:npm/c@1").as_bytes(), &mut out);
        assert_eq!(out.keys().collect::<Vec<_>>(), ["npm/c@1"]);
        // Bytes held but never analyzed are no verdict, whatever else it says.
        let pending = answer("pkg:npm/c@1").replacen('{', r#"{"state":"pending","#, 1);
        let mut out = HashMap::new();
        precheck.collect_hits(&asked, pending.as_bytes(), &mut out);
        assert!(out.is_empty(), "a pending record was taken as a verdict");
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
        let week = 7 * DAY;
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
        let v = match standing_of(rec, Some("b8c1c"), 7 * DAY, now, Lookup::DownloadedBytes) {
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
                    30 * DAY,
                    now,
                    Lookup::DownloadedBytes
                ),
                Standing::Analyze
            ));
            assert!(matches!(
                standing_of(
                    record(traits),
                    Some("local"),
                    30 * DAY,
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
                30 * DAY,
                now,
                Lookup::DownloadedBytes
            ),
            Standing::Adopt(_)
        ));
    }

    fn uplink() -> Precheck {
        Precheck {
            lookup_url: "https://unused.invalid/v1/lookup".into(),
            max_age: Duration::from_secs(30 * DAY),
            breaker: Breaker::default(),
            backend: Backend::Uplink(Some(
                cleave::feed::Token::parse("i13_test_token").expect("a token"),
            )),
            quota_file: None,
        }
    }

    /// The host follows the credential: a token is sent to the subscribers'
    /// host and no other, and a scan without one asks the open host and sends
    /// nothing that identifies it.
    #[test]
    fn only_a_subscriber_sends_a_credential() {
        let http = crate::remote::http().expect("a client");
        let sent = |precheck: &Precheck| {
            precheck
                .authed(http.get(&precheck.lookup_url))
                .build()
                .expect("a request")
        };
        let open = Precheck::uplink(None);
        assert!(open.is_uplink());
        assert_eq!(open.tid(), "anonymous");
        let request = sent(&open);
        assert_eq!(
            request.url().as_str(),
            "https://api.atomdrift.org/v1/lookup"
        );
        assert!(
            request
                .headers()
                .get(reqwest::header::AUTHORIZATION)
                .is_none()
        );
        assert!(request.headers().get(reqwest::header::COOKIE).is_none());

        let token = cleave::feed::Token::parse("i13_test_token").expect("a token");
        let paid = Precheck::uplink(Some(token));
        let request = sent(&paid);
        assert_eq!(
            request.url().as_str(),
            "https://api.atomdrift.com/v1/lookup"
        );
        assert_eq!(
            request.headers().get(reqwest::header::AUTHORIZATION),
            Some(&reqwest::header::HeaderValue::from_static(
                "Bearer i13_test_token"
            ))
        );
        assert_ne!(paid.tid(), "anonymous");
    }

    /// Uplink is the judge the scan asked for: an analyzed verdict
    /// stands whatever produced it, but a stale clean one, an unanalyzed one,
    /// or one with no level is analyzed here.
    #[test]
    fn uplink_adopts_analyzed_verdicts_but_not_stale_clean_ones() {
        let now = parse_rfc3339_epoch("2026-10-22T00:00:00Z").unwrap();
        let month = 30 * DAY;
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
        let precheck = uplink();
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

    /// A held breaker lets nothing out before its time, one probe after it,
    /// and nothing at all when held for the run.
    #[test]
    fn a_held_breaker_stays_shut_until_its_time() {
        let t0 = 1_000_000;
        let breaker = Breaker::default();
        assert!(breaker.hold(t0 + 600), "the first hold opens it");
        assert!(!breaker.hold(t0 + 600), "and is the only one to say so");
        assert!(!breaker.allows(t0));
        assert!(!breaker.allows(t0 + 599));
        assert!(breaker.allows(t0 + 600), "its time has come: one probe");
        assert!(!breaker.allows(t0 + 600), "and only one");
        breaker.succeeded();
        assert!(breaker.allows(t0 + 601));

        let breaker = Breaker::default();
        assert!(breaker.hold(u64::MAX));
        assert!(!breaker.allows(t0));
        assert!(!breaker.allows(u64::MAX - 1), "held for the run");
    }

    /// Uplink saying no, or failing, stops the asking for the rest of the
    /// run; an answer that is merely not a verdict does not.
    #[test]
    fn a_refusal_or_a_failure_from_uplink_stops_the_asking() {
        let far = crate::civil::now_secs() + 100 * BREAKER_COOLDOWN_SECS;
        for status in [401, 403, 500, 502, 503, 504] {
            let precheck = uplink();
            let code = reqwest::StatusCode::from_u16(status).unwrap();
            assert!(!precheck.answered(code, None, "test"));
            assert!(
                !precheck.breaker.allows(far),
                "{status} ends the asking for the run"
            );
        }
        for status in [400, 404, 413] {
            let precheck = uplink();
            let code = reqwest::StatusCode::from_u16(status).unwrap();
            assert!(!precheck.answered(code, None, "test"));
            assert!(
                precheck.breaker.allows(crate::civil::now_secs()),
                "{status} is one unanswered batch"
            );
        }
        assert!(uplink().answered(reqwest::StatusCode::OK, None, "test"));

        // Hopper's refusals keep their old meaning: analyze, keep asking.
        let hopper = Precheck::new("http://unused", Duration::from_secs(1)).unwrap();
        for status in [401, 429, 503] {
            let code = reqwest::StatusCode::from_u16(status).unwrap();
            assert!(!hopper.answered(code, Some(600), "test"));
            assert!(hopper.breaker.allows(crate::civil::now_secs()), "{status}");
        }
    }

    /// A spent quota holds lookups off for the `Retry-After` it came with: no
    /// longer than an hour whatever it says, and a cooldown when it says
    /// nothing.
    #[test]
    fn a_spent_quota_is_left_alone_until_retry_after() {
        let too_many = reqwest::StatusCode::TOO_MANY_REQUESTS;
        for (retry_after, held, free) in [
            (Some(600), 590, 610),
            (
                Some(86_400),
                QUOTA_HOLD_MAX_SECS - 10,
                QUOTA_HOLD_MAX_SECS + 10,
            ),
            (None, BREAKER_COOLDOWN_SECS - 10, BREAKER_COOLDOWN_SECS + 10),
        ] {
            let precheck = uplink();
            let now = crate::civil::now_secs();
            assert!(!precheck.answered(too_many, retry_after, "test"));
            assert!(!precheck.breaker.allows(now), "{retry_after:?}");
            assert!(!precheck.breaker.allows(now + held), "{retry_after:?}");
            assert!(precheck.breaker.allows(now + free), "{retry_after:?}");
        }
    }

    /// The end of a spent quota outlives the run that spent it, so the next
    /// run does not ask only to be refused; a file that says more than a
    /// quota could, or nothing readable, is not believed.
    #[test]
    fn a_spent_quota_is_remembered_by_the_next_run() {
        let dir = tempfile::tempdir().expect("tmp");
        let quota_file = dir.path().join("uplink-quota-anonymous");
        let fresh = || Precheck {
            quota_file: Some(quota_file.clone()),
            backend: Backend::Uplink(None),
            ..uplink()
        };
        let now = crate::civil::now_secs();
        assert_eq!(fresh().remembered_quota_hold(now), None, "nothing spent");

        assert!(!fresh().answered(reqwest::StatusCode::TOO_MANY_REQUESTS, Some(600), "test"));
        let until = fresh().remembered_quota_hold(now).expect("remembered");
        assert!((now + 600..=now + 605).contains(&until), "{until} vs {now}");
        assert_eq!(fresh().remembered_quota_hold(until), None, "turned over");

        std::fs::write(&quota_file, (now + 30 * 86_400).to_string()).expect("write");
        assert_eq!(
            fresh().remembered_quota_hold(now),
            Some(now + QUOTA_HOLD_MAX_SECS),
            "no quota holds for longer than an hour"
        );
        for junk in ["", "soon", "-5", "18446744073709551616"] {
            std::fs::write(&quota_file, junk).expect("write");
            assert_eq!(fresh().remembered_quota_hold(now), None, "{junk:?}");
        }
    }

    /// An Uplink that cannot be reached costs one failed ask: nothing more
    /// goes out for the rest of the run, and every artifact is analyzed here.
    #[test]
    fn an_unreachable_uplink_is_not_asked_again() {
        let precheck = Precheck {
            // Nothing listens on a reserved port of this machine.
            lookup_url: "https://127.0.0.1:1/v1/lookup".into(),
            ..uplink()
        };
        let sha = "a".repeat(64);
        assert!(matches!(precheck.standing(&sha), Standing::Analyze));
        let far = crate::civil::now_secs() + 100 * BREAKER_COOLDOWN_SECS;
        assert!(!precheck.breaker.allows(far), "held for the run");
        let (standings, unknown) = precheck.shas(std::slice::from_ref(&sha));
        assert!(standings.is_empty());
        assert!(unknown.is_empty(), "unanswered is not unknown");
        assert!(
            precheck
                .purls(&["pkg:npm/left-pad@1.3.0".to_string()])
                .is_empty()
        );
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
        assert_eq!(rec.findings, []);
    }
}
