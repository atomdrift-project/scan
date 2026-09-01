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
//!   2. BENIGN AND FRESH: `fires_at == -1`, analyzed within the last 30 days
//!      (see [`DEFAULT_MAX_AGE_DAYS`] for why that long), regardless of
//!      analyzer version. The coarse rule that keeps working through a
//!      mixed-version fleet or a release that just bumped every traits hash:
//!      dependency universes are overwhelmingly benign.
//!
//! Anything neither rule covers — not found, not fresh, hostile under a
//! different analyzer, unreachable — falls through to a normal analysis:
//! fail-open, never fail-closed.
//!
//! Enabled automatically whenever `--hopper` is: [`configure`] is called with
//! the process's own hopper URL when the uploader is built, so there is no
//! second setting to keep in sync — the hopper you submit to is the hopper you
//! ask. (Hopper serves `/v1/lookup` from an in-memory pool on both the primary
//! and the read replica, so whichever the process talks to answers cheaply.)
//! No `--hopper`, no precheck. Auth reuses the process's hopper bearer token
//! (see [`crate::upload::bearer_token`]).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Lookups attempted (a hit or a miss, but the wire was asked).
static CHECKS: AtomicU64 = AtomicU64::new(0);
/// Analyses skipped because the corpus already held a benign, fresh verdict.
static SKIPS: AtomicU64 = AtomicU64::new(0);
/// PURLs asked of hopper by the pre-fetch batch negotiation.
static PURL_CHECKS: AtomicU64 = AtomicU64::new(0);
/// PURLs whose fetch+analysis were skipped on hopper's standing verdict.
static PURL_SKIPS: AtomicU64 = AtomicU64::new(0);
/// Consecutive transport failures. At [`BREAKER_LIMIT`] the precheck disables
/// itself for the life of the process: a dead replica must cost one log line,
/// not a per-dependency connect timeout inside the analysis pipeline.
static FAILURES: AtomicU32 = AtomicU32::new(0);

const BREAKER_LIMIT: u32 = 5;

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

struct Precheck {
    lookup_url: String,
    client: reqwest::blocking::Client,
    max_age: Duration,
}

static INSTANCE: OnceLock<Option<Precheck>> = OnceLock::new();

/// Arm the precheck against the hopper this process already talks to. Called
/// where the uploader is built — the one place every `--hopper` mode passes
/// through — so enablement follows `--hopper` with no second setting. First
/// caller wins; later calls (a server rebuilding its uploader) are no-ops.
pub(crate) fn configure(hopper_base_url: &str) {
    INSTANCE.get_or_init(|| {
        // HOPPER may be a comma list, replica first ("https://ro,…"): lookups
        // belong on the first entry — the replica when one is named, which is
        // exactly where a cheap read should land.
        let base = hopper_base_url
            .split(',')
            .next()
            .unwrap_or_default()
            .trim()
            .trim_end_matches('/');
        if base.is_empty() {
            return None;
        }
        let days = std::env::var("SCAN_CORPUS_MAX_AGE_DAYS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_AGE_DAYS);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .ok()?;
        tracing::info!(
            url = %base,
            max_age_days = days,
            "corpus precheck enabled: same-analyzer or benign+fresh dependencies will not be re-analyzed"
        );
        Some(Precheck {
            lookup_url: format!("{base}/v1/lookup"),
            client,
            max_age: Duration::from_secs(days * 86_400),
        })
    });
}

fn instance() -> Option<&'static Precheck> {
    INSTANCE.get().and_then(|o| o.as_ref())
}

/// The three fields the policy reads. Everything else in the record is
/// ignored, so the response shape may grow freely.
#[derive(serde::Deserialize)]
struct Record {
    fires_at: Option<i64>,
    analyzed_at: Option<String>,
    traits_version: Option<String>,
}

/// This worker's 5-char traits commit prefix — the same value the /api/next
/// heartbeat sends, truncated the way hopper truncates, so string equality
/// against a stored `traits_version` means "same analyzer".
fn local_traits() -> Option<&'static str> {
    static TRAITS: OnceLock<Option<String>> = OnceLock::new();
    TRAITS
        .get_or_init(|| cleave::traits_repo::version().map(|v| v.chars().take(5).collect()))
        .as_deref()
}

/// True when hopper's corpus already holds a verdict for these exact bytes
/// that re-analysis could not improve on — same analyzer, or benign and
/// fresh (see the module doc). Any failure — unreachable, non-200,
/// unparseable, neither rule met — is false.
pub(crate) fn skip_reanalysis(content_sha: &str) -> bool {
    let Some(p) = instance() else { return false };
    if content_sha.len() != 64 || FAILURES.load(Ordering::Relaxed) >= BREAKER_LIMIT {
        return false;
    }
    CHECKS.fetch_add(1, Ordering::Relaxed);

    let mut req = p
        .client
        .get(&p.lookup_url)
        .query(&[("sha256", content_sha)]);
    if let Some(token) = crate::upload::bearer_token() {
        req = req.bearer_auth(token);
    }
    let resp = match req.send() {
        Ok(r) => {
            FAILURES.store(0, Ordering::Relaxed);
            r
        }
        Err(e) => {
            let n = FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
            if n == BREAKER_LIMIT {
                tracing::warn!(
                    error = %e,
                    "corpus precheck: {BREAKER_LIMIT} consecutive transport failures; \
                     disabling for the rest of this process"
                );
            }
            return false;
        }
    };
    if resp.status() != reqwest::StatusCode::OK {
        return false; // 404 unknown, 202 bytes-only, 401/5xx — all mean "scan it".
    }
    let Ok(rec) = resp.json::<Record>() else {
        return false;
    };
    if !verdict_stands(&rec, local_traits(), p.max_age.as_secs(), now_epoch()) {
        return false;
    }
    SKIPS.fetch_add(1, Ordering::Relaxed);
    true
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// The policy, pure so the tests can hold it still: rule 1 (same analyzer)
/// or rule 2 (benign and fresh).
fn verdict_stands(rec: &Record, my_traits: Option<&str>, max_age_s: u64, now: u64) -> bool {
    // Rule 1: same analyzer already judged these bytes. A verdict is only a
    // dedupe key when it EXISTS — fires_at is null for a record that was
    // never classified, and traits equality on an unclassified record would
    // skip an analysis that never happened.
    if rec.fires_at.is_some()
        && let (Some(mine), Some(theirs)) = (my_traits, rec.traits_version.as_deref())
        && !mine.is_empty()
        && mine == theirs
    {
        return true;
    }
    // Rule 2: benign, and fresh enough that staleness is bounded.
    if rec.fires_at != Some(-1) {
        return false;
    }
    let Some(at) = rec.analyzed_at.as_deref().and_then(parse_rfc3339_epoch) else {
        return false;
    };
    now.saturating_sub(at) <= max_age_s
}

/// (lookups attempted, analyses skipped) since process start, for the worker
/// summary line.
/// Batch PURL negotiation: which of these dependency PURLs does hopper hold a
/// standing verdict for? Returns `purl → content sha256` for every entry that
/// satisfies the same two rules as [`skip_reanalysis`] — the caller skips the
/// FETCH as well as the analysis for those, which the per-sha precheck cannot
/// (a registry PURL's content sha is only learned by downloading it). Batched
/// 50 to a request (hopper's documented cap). An answer with no usable sha is
/// dropped — the fetch-edge (`source → content sha`) must stay recordable — so
/// the dependency falls through to a normal fetch. Fail-open everywhere, and
/// `SCAN_PURL_PRECHECK=0` disables just this half.
pub(crate) fn precheck_purls(purls: &[String]) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    let Some(p) = instance() else { return out };
    if purls.is_empty()
        || FAILURES.load(Ordering::Relaxed) >= BREAKER_LIMIT
        || std::env::var("SCAN_PURL_PRECHECK").as_deref() == Ok("0")
    {
        return out;
    }
    PURL_CHECKS.fetch_add(purls.len() as u64, Ordering::Relaxed);
    for chunk in purls.chunks(50) {
        let mut req = p.client.get(&p.lookup_url);
        for purl in chunk {
            req = req.query(&[("purl", purl.as_str())]);
        }
        if let Some(token) = crate::upload::bearer_token() {
            req = req.bearer_auth(token);
        }
        let resp = match req.send() {
            Ok(r) => {
                FAILURES.store(0, Ordering::Relaxed);
                r
            }
            Err(e) => {
                let n = FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
                if n == BREAKER_LIMIT {
                    tracing::warn!(
                        error = %e,
                        "purl precheck: {BREAKER_LIMIT} consecutive transport failures;                          disabling for the rest of this process"
                    );
                }
                continue;
            }
        };
        if resp.status() != reqwest::StatusCode::OK {
            continue;
        }
        let Ok(value) = resp.json::<serde_json::Value>() else {
            continue;
        };
        // One purl answers with one object, several with a list in the order
        // asked; tolerate both, and prefer the answer's own `purl` field over
        // positional matching when present.
        let items: Vec<&serde_json::Value> = match value.as_array() {
            Some(list) => list.iter().collect(),
            None => vec![&value],
        };
        for (i, item) in items.iter().enumerate() {
            let rec = Record {
                fires_at: item.get("fires_at").and_then(serde_json::Value::as_i64),
                analyzed_at: item
                    .get("analyzed_at")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                traits_version: item
                    .get("traits_version")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
            };
            if !verdict_stands(&rec, local_traits(), p.max_age.as_secs(), now_epoch()) {
                continue;
            }
            let Some(sha) = item
                .get("sha256")
                .and_then(serde_json::Value::as_str)
                .filter(|d| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()))
            else {
                continue;
            };
            let purl = item
                .get("purl")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .or_else(|| chunk.get(i).cloned());
            if let Some(purl) = purl {
                PURL_SKIPS.fetch_add(1, Ordering::Relaxed);
                out.insert(purl, sha.to_ascii_lowercase());
            }
        }
    }
    out
}

/// `(purl_checks, purl_skips)` lifetime counters for the batch negotiation.
pub(crate) fn purl_counters() -> (u64, u64) {
    (
        PURL_CHECKS.load(Ordering::Relaxed),
        PURL_SKIPS.load(Ordering::Relaxed),
    )
}

pub(crate) fn counters() -> (u64, u64) {
    (
        CHECKS.load(Ordering::Relaxed),
        SKIPS.load(Ordering::Relaxed),
    )
}

/// Parse an RFC 3339 UTC timestamp ("2026-08-23T23:00:44Z", fractional seconds
/// tolerated and ignored) to a Unix epoch. The inverse of
/// [`crate::engine::now_rfc3339`]'s civil-date math (Howard Hinnant's
/// days-from-civil), hand-rolled for the same reason: no time crate. Offsets
/// other than Z are rejected — hopper emits UTC and a wrong-but-plausible
/// parse here would silently misjudge freshness.
fn parse_rfc3339_epoch(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    if b.len() < 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
    {
        return None;
    }
    if b[b.len() - 1] != b'Z' {
        return None;
    }
    let num = |r: std::ops::Range<usize>| -> Option<u64> { s.get(r)?.parse().ok() };
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (h, mi, sec) = (num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || sec > 60 {
        return None;
    }
    // days-from-civil (Hinnant), the inverse of now_rfc3339's civil-from-days.
    let y = y as i64 - i64::from(mo <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400) as u64;
    let mp = if mo > 2 { mo - 3 } else { mo + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe as i64 - 719_468;
    u64::try_from(days * 86_400 + (h * 3600 + mi * 60 + sec) as i64).ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The parser must invert engine::now_rfc3339 exactly: format an epoch,
    /// parse it back, and land on the same second — across month/era edges.
    #[test]
    fn parses_what_now_rfc3339_formats() {
        for s in [
            "1970-01-01T00:00:00Z",
            "2000-02-29T12:00:00Z",
            "2026-08-23T23:00:44Z",
            "2026-12-31T23:59:59Z",
            "2100-03-01T00:00:00Z",
        ] {
            let epoch = parse_rfc3339_epoch(s).expect(s);
            // Reformat via the same civil-date math the writer uses.
            let days = epoch / 86_400;
            let z = days as i64 + 719_468;
            let era = z.div_euclid(146_097);
            let doe = z.rem_euclid(146_097) as u64;
            let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
            let y = yoe as i64 + era * 400;
            let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            let mp = (5 * doy + 2) / 153;
            let d = doy - (153 * mp + 2) / 5 + 1;
            let mo = if mp < 10 { mp + 3 } else { mp - 9 };
            let y = if mo <= 2 { y + 1 } else { y };
            let tod = epoch % 86_400;
            let back = format!(
                "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}Z",
                tod / 3600,
                (tod % 3600) / 60,
                tod % 60
            );
            assert_eq!(back, s);
        }
    }

    #[test]
    fn rejects_offsets_and_garbage() {
        for s in [
            "2026-08-23T23:00:44+02:00", // non-UTC offset: refuse, don't misjudge
            "2026-08-23 23:00:44Z",      // space separator
            "not-a-time",
            "",
            "2026-13-01T00:00:00Z", // month 13
        ] {
            assert_eq!(parse_rfc3339_epoch(s), None, "{s}");
        }
    }

    #[test]
    fn policy_two_rules() {
        let rec = |fires: Option<i64>, tv: Option<&str>, at: Option<&str>| Record {
            fires_at: fires,
            analyzed_at: at.map(String::from),
            traits_version: tv.map(String::from),
        };
        let now = parse_rfc3339_epoch("2026-08-24T00:00:00Z").unwrap();
        let week = 7 * 86_400;
        let fresh = Some("2026-08-23T00:00:00Z"); // 1 day old
        let stale = Some("2026-08-01T00:00:00Z"); // 23 days old

        // Rule 1: same analyzer skips any verdict, hostile included, any age.
        assert!(verdict_stands(
            &rec(Some(3), Some("b8c1c"), stale),
            Some("b8c1c"),
            week,
            now
        ));
        // ...but never on a record with no verdict at all.
        assert!(!verdict_stands(
            &rec(None, Some("b8c1c"), fresh),
            Some("b8c1c"),
            week,
            now
        ));
        // ...and never across analyzers for a non-benign verdict.
        assert!(!verdict_stands(
            &rec(Some(3), Some("f6eaa"), fresh),
            Some("b8c1c"),
            week,
            now
        ));
        // Empty-string traits (the member-row gap) must not match anything.
        assert!(!verdict_stands(
            &rec(Some(3), Some(""), fresh),
            Some(""),
            week,
            now
        ));

        // Rule 2: benign and fresh skips across analyzers...
        assert!(verdict_stands(
            &rec(Some(-1), Some("f6eaa"), fresh),
            Some("b8c1c"),
            week,
            now
        ));
        // ...including when the stored row has no traits at all.
        assert!(verdict_stands(
            &rec(Some(-1), None, fresh),
            Some("b8c1c"),
            week,
            now
        ));
        // ...but not stale, and not without a timestamp.
        assert!(!verdict_stands(
            &rec(Some(-1), None, stale),
            Some("b8c1c"),
            week,
            now
        ));
        assert!(!verdict_stands(
            &rec(Some(-1), None, None),
            Some("b8c1c"),
            week,
            now
        ));
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
        assert_eq!(rec.fires_at, Some(-1));
        assert!(rec.analyzed_at.is_some());
    }
}
