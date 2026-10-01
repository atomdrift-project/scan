//! Fleet-shared skip for fetched-dependency analysis, backed by hopper's corpus.
//!
//! [`crate::analysis_cache`] already memoizes dependency analyses — but per
//! worker, namespaced by ruleset version. A fleet release therefore invalidates
//! every worker's cache at once, and each of a dozen workers rebuilds a private
//! copy of the same shared dependency universe: measured 2026-08-23 as
//! thousands of redundant re-scans per hour, every one stored by hopper with
//! "result renewed with no analyzer change; the re-analysis learned nothing".
//!
//! This module asks the one cache the whole fleet shares — hopper — before
//! analyzing a fetched payload. Two independent rules, either sufficient:
//!
//!   1. TRAITS MATCH: the stored verdict was produced under this worker's own
//!      analyzer version (`traits_version` equals our 5-char traits commit,
//!      the same truncation the /api/next heartbeat sends). Re-analysis by the
//!      same analyzer learns nothing — hopper logs exactly that when it
//!      happens — so this skips ANY verdict, hostile included.
//!   2. BENIGN AND FRESH: `fires_at` is `-1` (clean), analyzed within the last 30 days
//!      (see `DEFAULT_MAX_AGE_DAYS` for why that long), regardless of
//!      analyzer version. The coarse rule that keeps working through a
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
//! (see [`crate::upload::hopper_token`]).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, PoisonError, RwLock};
use std::time::Duration;

use crate::fetch::unix_now;
use crate::model::Level;

/// Lookups attempted (a hit or a miss, but the wire was asked).
static CHECKS: AtomicU64 = AtomicU64::new(0);
/// Analyses skipped because the corpus already held a benign, fresh verdict.
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
/// `SCAN_CORPUS_MAX_AGE_DAYS` overrides.
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
/// package when its dependency RECORDS change (2026-08-24).
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
pub(crate) struct Precheck {
    lookup_url: String,
    max_age: Duration,
    /// `SCAN_PURL_PRECHECK=0` turns off only the batch PURL negotiation.
    purls: bool,
    breaker: Breaker,
}

/// The armed precheck. Every [`configure`] replaces it, so the last hopper a
/// process was pointed at is the one it asks.
static ARMED: RwLock<Option<Arc<Precheck>>> = RwLock::new(None);

/// Arm the precheck against the hopper this process already talks to. Called
/// where the uploader is built — the one place every `--hopper` mode passes
/// through — so enablement follows `--hopper` with no second setting.
pub(crate) fn configure(hopper_base_url: &str) {
    let max_age_days = match std::env::var("SCAN_CORPUS_MAX_AGE_DAYS") {
        Ok(days) => days.trim().parse().unwrap_or_else(|_| {
            tracing::warn!(value = %days, default = DEFAULT_MAX_AGE_DAYS, "SCAN_CORPUS_MAX_AGE_DAYS is not a number of days; using the default");
            DEFAULT_MAX_AGE_DAYS
        }),
        Err(_) => DEFAULT_MAX_AGE_DAYS,
    };
    let purls = std::env::var("SCAN_PURL_PRECHECK").as_deref() != Ok("0");
    let precheck = Precheck::new(
        hopper_base_url,
        Duration::from_secs(max_age_days.saturating_mul(86_400)),
        purls,
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
            "corpus precheck enabled: same-analyzer or benign+fresh dependencies will not be re-analyzed"
        );
    }
    *armed = precheck.map(Arc::new);
}

/// The armed precheck, or `None` when this process has no hopper.
pub(crate) fn armed() -> Option<Arc<Precheck>> {
    ARMED.read().unwrap_or_else(PoisonError::into_inner).clone()
}

fn authed(request: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
    match crate::upload::hopper_token() {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

/// The fields the policy reads, plus the verdict it may adopt. Everything else
/// in the record is ignored, so the response shape may grow freely.
#[derive(serde::Deserialize)]
struct Record {
    fires_at: Level,
    analyzed_at: Option<String>,
    traits_version: Option<String>,
    reason: Option<String>,
    #[serde(default)]
    findings: Vec<Finding>,
}

/// One answer of a batch PURL lookup: the record plus the two keys that pair
/// it with its dependency.
#[derive(serde::Deserialize)]
struct PurlAnswer {
    sha256: Option<String>,
    purl: Option<String>,
    #[serde(flatten)]
    record: Record,
}

/// One of the corpus's strongest traits for an artifact, as `/v1/lookup`
/// reports it: a stable id, its criticality (4 suspicious, 5 hostile), and a
/// sentence for the findings that are not the analyzer's own.
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Finding {
    pub id: String,
    #[serde(default)]
    pub desc: String,
    pub crit: u32,
}

/// A verdict this build may adopt as its own: the corpus produced it under the
/// analyzer we are running, so re-deriving it locally would reach the same
/// answer. `fires_at` is the tightest false-positive budget at which the
/// artifact grades hostile; grading it against a caller's budget is
/// [`crate::server::decision::decide`]'s job, never this module's.
#[derive(Debug, Clone)]
pub(crate) struct Verdict {
    pub fires_at: Level,
    pub reason: Option<String>,
    pub findings: Vec<Finding>,
}

/// What the corpus lets us skip, and whether it also hands us an answer.
///
/// The distinction is the whole point: a verdict is only ours to report when it
/// was produced by the analyzer we are running. Anything else is evidence about
/// someone else's judgment — good enough to skip re-deriving a benign result,
/// not good enough to publish as this scan's finding.
#[derive(Debug, Clone)]
pub(crate) enum Standing {
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
    pub(crate) const fn skips_analysis(&self) -> bool {
        !matches!(self, Self::Analyze)
    }
}

/// A dependency the batch PURL negotiation answered for: the content sha that
/// records the fetch edge, and what the corpus lets us skip for it — never
/// [`Standing::Analyze`].
#[derive(Debug, Clone)]
pub(crate) struct PurlHit {
    pub sha: String,
    pub standing: Standing,
}

/// This worker's 5-char traits commit prefix — the same value the /api/next
/// heartbeat sends, truncated the way hopper truncates, so string equality
/// against a stored `traits_version` means "same analyzer".
pub(crate) fn local_traits() -> Option<&'static str> {
    static TRAITS: OnceLock<Option<String>> = OnceLock::new();
    TRAITS
        .get_or_init(|| cleave::traits_repo::version().map(|v| v.chars().take(5).collect()))
        .as_deref()
}

impl Precheck {
    /// A precheck against the first hopper `hopper_base_url` names, or `None`
    /// when it names none.
    fn new(hopper_base_url: &str, max_age: Duration, purls: bool) -> Option<Self> {
        // HOPPER may be a comma list, replica first ("https://ro,…"): lookups
        // belong on the first entry — the replica when one is named, which is
        // exactly where a cheap read should land.
        let base = crate::upload::endpoints(hopper_base_url)
            .into_iter()
            .next()?;
        Some(Self {
            lookup_url: format!("{base}/v1/lookup"),
            max_age,
            purls,
            breaker: Breaker::default(),
        })
    }

    /// What hopper's corpus already holds for these exact bytes: a verdict this
    /// build may adopt (same analyzer), a benign result worth skipping but not
    /// reporting, or nothing. Any failure — unreachable, non-200, unparseable,
    /// neither rule met — is [`Standing::Analyze`].
    pub(crate) fn standing(&self, content_sha: &str) -> Standing {
        if content_sha.len() != 64 || !self.breaker.allows(unix_now()) {
            return Standing::Analyze;
        }
        let Some(http) = crate::upload::hopper_http() else {
            return Standing::Analyze;
        };
        CHECKS.fetch_add(1, Ordering::Relaxed);
        let request = http
            .get(&self.lookup_url)
            .timeout(LOOKUP_TIMEOUT)
            .query(&[("sha256", content_sha)]);
        let resp = match authed(request).send() {
            Ok(r) => {
                self.breaker.succeeded();
                r
            }
            Err(e) => {
                self.breaker.failed(unix_now(), "corpus precheck", &e);
                return Standing::Analyze;
            }
        };
        if resp.status() != reqwest::StatusCode::OK {
            return Standing::Analyze; // 404 unknown, 202 bytes-only, 401/5xx — all "scan it".
        }
        let rec = match resp.json::<Record>() {
            Ok(rec) => rec,
            Err(e) => {
                tracing::debug!(error = %e, "corpus precheck: unreadable lookup answer; analyzing");
                return Standing::Analyze;
            }
        };
        let standing = standing_of(rec, local_traits(), self.max_age.as_secs(), unix_now());
        if standing.skips_analysis() {
            SKIPS.fetch_add(1, Ordering::Relaxed);
        }
        standing
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
        if !self.purls || purls.is_empty() {
            return out;
        }
        let Some(http) = crate::upload::hopper_http() else {
            return out;
        };
        for chunk in purls.chunks(PURL_BATCH) {
            // Re-checked per chunk: a trip here or on another thread stops the
            // rest, which would only time out too.
            if !self.breaker.allows(unix_now()) {
                break;
            }
            PURL_CHECKS.fetch_add(chunk.len() as u64, Ordering::Relaxed);
            let lookup = http.get(&self.lookup_url).timeout(LOOKUP_TIMEOUT);
            let request = chunk
                .iter()
                .fold(lookup, |req, purl| req.query(&[("purl", purl.as_str())]));
            let resp = match authed(request).send() {
                Ok(r) => {
                    self.breaker.succeeded();
                    r
                }
                Err(e) => {
                    self.breaker.failed(unix_now(), "purl precheck", &e);
                    continue;
                }
            };
            if resp.status() != reqwest::StatusCode::OK {
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
        let items: Result<Vec<&serde_json::value::RawValue>, _> =
            if body.trim_ascii_start().starts_with(b"[") {
                serde_json::from_slice(body)
            } else {
                serde_json::from_slice(body).map(|one| vec![one])
            };
        let items = match items {
            Ok(items) => items,
            Err(e) => {
                tracing::debug!(error = %e, "purl precheck: unreadable lookup answer");
                return;
            }
        };
        for (i, item) in items.iter().enumerate() {
            let Ok(answer) = serde_json::from_str::<PurlAnswer>(item.get()) else {
                continue;
            };
            let standing = standing_of(
                answer.record,
                local_traits(),
                self.max_age.as_secs(),
                unix_now(),
            );
            if !standing.skips_analysis() {
                continue;
            }
            let Some(sha) = answer
                .sha256
                .filter(|d| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()))
            else {
                continue;
            };
            let Some(purl) = answer.purl.or_else(|| asked.get(i).cloned()) else {
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

/// The policy, pure so the tests can hold it still: rule 1 (same analyzer,
/// verdict adopted) or rule 2 (benign and fresh, work skipped).
fn standing_of(rec: Record, my_traits: Option<&str>, max_age_s: u64, now: u64) -> Standing {
    // Rule 1: same analyzer already judged these bytes. A verdict is only a
    // dedupe key when it EXISTS — fires_at is null for a record that was
    // never classified, and traits equality on an unclassified record would
    // skip an analysis that never happened.
    if rec.fires_at != Level::Manual
        && let (Some(mine), Some(theirs)) = (my_traits, rec.traits_version.as_deref())
        && !mine.is_empty()
        && mine == theirs
    {
        return Standing::Adopt(Verdict {
            fires_at: rec.fires_at,
            reason: rec.reason,
            findings: rec.findings,
        });
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
    if now.saturating_sub(at) <= max_age_s {
        Standing::SkipBenign
    } else {
        Standing::Analyze
    }
}

/// `(lookups attempted, analyses skipped)` since process start, for the worker
/// summary line.
pub(crate) fn counters() -> (u64, u64) {
    (
        CHECKS.load(Ordering::Relaxed),
        SKIPS.load(Ordering::Relaxed),
    )
}

/// `(purl_checks, purl_skips)` lifetime counters for the batch negotiation.
pub(crate) fn purl_counters() -> (u64, u64) {
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
    crate::fetch::utc_epoch(
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
        let before = unix_now();
        let parsed = parse_rfc3339_epoch(&crate::engine::now_rfc3339()).expect("parses");
        assert!(
            (before..=unix_now()).contains(&parsed),
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
            true,
        )
        .expect("a hopper");
        for _ in 0..BREAKER_LIMIT {
            precheck.breaker.failed(unix_now(), "test", &"refused");
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
            Precheck::new("http://unused", Duration::from_secs(86_400), true).expect("a hopper");
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
        };
        let now = parse_rfc3339_epoch("2026-08-24T00:00:00Z").unwrap();
        let week = 7 * 86_400;
        let fresh = Some("2026-08-23T00:00:00Z"); // 1 day old
        let stale = Some("2026-08-01T00:00:00Z"); // 23 days old
        let standing = |r: Record, mine: Option<&str>| standing_of(r, mine, week, now);

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
        let v = match standing_of(rec, Some("b8c1c"), 7 * 86_400, now) {
            Standing::Adopt(v) => Some(v),
            _ => None,
        }
        .expect("the same analyzer must adopt");
        assert_eq!(v.fires_at, Level::At(2));
        assert_eq!(v.reason.as_deref(), Some("steals credentials"));
        assert_eq!(v.findings.len(), 2);
        assert_eq!(v.findings[0].crit, 5);
        assert_eq!(v.findings[1].desc, "cited by OSV");
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
