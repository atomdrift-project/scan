//! The one retry policy for every blocking call this process makes to another
//! service: hopper, isotope13, the update bucket.
//!
//! A transient failure — the connection dropped or timed out, or the server
//! answered 408, 429 or 5xx — is retried with exponential backoff and full
//! jitter, honouring `Retry-After`, until the caller's budget is spent. Any
//! other answer is final at once: a 404 or 401 says the same thing every time.
//!
//! The budget is the caller's, because what a retry costs depends on what is
//! waiting: a renewal on the uploader's own thread can afford a minute, a
//! lookup with a safe fallback only seconds. Calls that already degrade by
//! other means retry nothing here — the corpus precheck is an optimization
//! behind a circuit breaker and fails open; the LLM client fails over between
//! endpoints behind per-endpoint breakers; the update notice is best-effort.

use std::time::{Duration, Instant};

use reqwest::StatusCode;
use reqwest::blocking::{RequestBuilder, Response};

/// Ceiling on one backoff sleep, so a long budget still probes often enough to
/// catch a short window of free capacity rather than sleeping through it.
pub(crate) const RETRY_MAX_BACKOFF: Duration = Duration::from_secs(60);

/// Floor on one backoff sleep. Full jitter can draw near zero, and a retry
/// that fires instantly just spends a slot-acquire on a pool it was told is
/// full.
pub(crate) const RETRY_MIN_BACKOFF: Duration = Duration::from_millis(250);

/// Sleep before retry `attempt` (1-based) of any call.
///
/// Exponential with full jitter: the sleep is drawn from `[0, ceiling)` where
/// the ceiling doubles per attempt up to [`RETRY_MAX_BACKOFF`]. The jitter
/// matters more than the growth here — every scan server writing to the same
/// saturated hopper would otherwise retry in lockstep and re-saturate it the
/// instant a slot frees.
///
/// `retry_after` is the server's own hint and acts as a floor: returning before
/// it only spends a slot-acquire on a pool that just said it was full.
///
/// Pure, with the random draw passed in, so the policy is testable without
/// sleeping or seeding.
pub(crate) fn backoff(attempt: usize, retry_after: Option<Duration>, fuzz: f64) -> Duration {
    let ceiling = RETRY_MAX_BACKOFF.min(RETRY_MIN_BACKOFF.saturating_mul(1u32 << attempt.min(16)));
    ceiling
        .mul_f64(fuzz.clamp(0.0, 1.0))
        .max(retry_after.unwrap_or(RETRY_MIN_BACKOFF))
        .max(RETRY_MIN_BACKOFF)
}

/// The sleep before retry `attempt` once `elapsed` of `budget` is spent: the
/// [`backoff`], cut short at the budget, or `None` once nothing is left. Never
/// sleeps past the budget: a sleep that outlives it would turn the ceiling into
/// a lie and delay the give-up log.
pub(crate) fn delay_within(
    budget: Duration,
    attempt: usize,
    retry_after: Option<Duration>,
    elapsed: Duration,
    fuzz: f64,
) -> Option<Duration> {
    let remaining = budget.checked_sub(elapsed).filter(|left| !left.is_zero())?;
    Some(backoff(attempt, retry_after, fuzz).min(remaining))
}

/// A uniform-ish draw in `[0, 1)` for backoff jitter.
///
/// Deliberately not a `rand` dependency: spreading retries needs decorrelation,
/// not statistical quality. Mixes the clock through splitmix64 so two uploader
/// threads starting in the same millisecond still diverge.
pub(crate) fn fuzz() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    // Seconds and sub-second nanos combined without going through u128, so
    // there is no truncating cast to explain away.
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        d.as_secs()
            .wrapping_shl(20)
            .wrapping_add(u64::from(d.subsec_nanos()))
    });
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // 53 bits is the mantissa width, so this maps onto [0, 1) without bias.
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Parse `Retry-After` in its delta-seconds form. The HTTP-date form and any
/// unparseable value yield `None`, leaving the caller on its own backoff.
pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let secs: u64 = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    (secs > 0).then(|| Duration::from_secs(secs).min(RETRY_MAX_BACKOFF))
}

/// Whether an answer is worth asking again: a timeout, rate limiting, or the
/// server being unwell.
pub(crate) fn is_transient(status: StatusCode) -> bool {
    status.is_server_error()
        || status == StatusCode::REQUEST_TIMEOUT
        || status == StatusCode::TOO_MANY_REQUESTS
}

/// Send the request `build` makes for each attempt (1-based, so a caller can
/// walk an address list), retrying transient failures within `budget`.
///
/// Returns the first answer that is not transient — a success or a permanent
/// refusal, for the caller to read — or, once the budget is spent, the last
/// attempt's outcome. A zero budget makes exactly one attempt. `what` names
/// the call in the log; run it inside a span carrying the caller's own fields
/// (the artifact, the package) and every retry line carries them too.
pub(crate) fn send(
    what: &str,
    budget: Duration,
    mut build: impl FnMut(usize) -> RequestBuilder,
) -> reqwest::Result<Response> {
    let started = Instant::now();
    let mut attempt = 0;
    loop {
        attempt += 1;
        let outcome = build(attempt).send();
        let (retry_after, failure) = match &outcome {
            Ok(resp) if !is_transient(resp.status()) => return outcome,
            Ok(resp) => (parse_retry_after(resp.headers()), resp.status().to_string()),
            // A request that could not be built, or a redirect refused by
            // policy, fails the same way every time.
            Err(e) if e.is_builder() || e.is_redirect() => return outcome,
            Err(e) => (None, crate::hopper::error_chain(e)),
        };
        let Some(delay) = delay_within(budget, attempt, retry_after, started.elapsed(), fuzz())
        else {
            if !budget.is_zero() {
                tracing::warn!(
                    what,
                    attempts = attempt,
                    budget_s = budget.as_secs(),
                    error = %failure,
                    "transient failure persisted past the retry budget; giving up",
                );
            }
            return outcome;
        };
        tracing::warn!(
            what,
            attempt,
            delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
            error = %failure,
            "transient failure; retrying",
        );
        // Release the connection before sleeping on it.
        drop(outcome);
        std::thread::sleep(delay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    /// Full jitter: the draw scales the ceiling, so a fleet retrying against one
    /// saturated hopper spreads out instead of re-saturating it in lockstep.
    #[test]
    fn backoff_applies_full_jitter() {
        let low = backoff(10, None, 0.0);
        let high = backoff(10, None, 1.0);
        assert!(high > low, "jitter had no effect: {low:?} vs {high:?}");
        assert!(
            low >= RETRY_MIN_BACKOFF,
            "a near-zero draw must still back off: {low:?}"
        );
        assert!(high <= RETRY_MAX_BACKOFF, "exceeded the ceiling: {high:?}");
    }

    /// The ceiling grows with the attempt and then stops.
    #[test]
    fn backoff_grows_exponentially_then_caps() {
        let at = |n| backoff(n, None, 1.0);
        assert!(at(1) < at(3), "not growing: {:?} then {:?}", at(1), at(3));
        assert!(at(3) < at(6), "not growing: {:?} then {:?}", at(3), at(6));
        assert_eq!(at(20), RETRY_MAX_BACKOFF, "ceiling not enforced");
    }

    #[test]
    fn parse_retry_after_forms() {
        use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};
        let with = |v: &str| {
            let mut h = HeaderMap::new();
            h.insert(RETRY_AFTER, HeaderValue::from_str(v).unwrap());
            parse_retry_after(&h)
        };
        assert_eq!(with("2"), Some(Duration::from_secs(2)));
        assert_eq!(with(" 5 "), Some(Duration::from_secs(5)));
        assert_eq!(with("0"), None);
        assert_eq!(with("-1"), None);
        // The HTTP-date form is legal but unparsed here; fall back to our own.
        assert_eq!(with("Wed, 21 Oct 2026 07:28:00 GMT"), None);
        // A hostile value must not park an uploader thread for hours.
        assert_eq!(with("86400"), Some(RETRY_MAX_BACKOFF));
        assert_eq!(parse_retry_after(&HeaderMap::new()), None);
    }

    #[test]
    fn fuzz_is_in_range_and_varies() {
        let draws: Vec<f64> = (0..64).map(|_| fuzz()).collect();
        assert!(draws.iter().all(|d| (0.0..1.0).contains(d)), "out of range");
        let distinct = draws
            .iter()
            .map(|d| d.to_bits())
            .collect::<std::collections::HashSet<_>>();
        assert!(distinct.len() > 1, "fuzz returned a constant");
    }

    /// 408, 429 and 5xx ask to be retried; every other answer is final.
    #[test]
    fn only_transient_answers_are_retried() {
        for status in [408, 429, 500, 502, 503, 504] {
            assert!(
                is_transient(StatusCode::from_u16(status).unwrap()),
                "{status}"
            );
        }
        for status in [200, 301, 400, 401, 403, 404, 413] {
            assert!(
                !is_transient(StatusCode::from_u16(status).unwrap()),
                "{status}"
            );
        }
    }

    /// Answer each connection to a fresh listener with the next of `statuses`,
    /// counting the requests that arrive.
    fn serve(statuses: &'static [u16]) -> (String, std::thread::JoinHandle<usize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            for &status in statuses {
                let (connection, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(connection);
                let mut line = String::new();
                while line != "\r\n" {
                    line.clear();
                    assert!(reader.read_line(&mut line).unwrap() > 0);
                }
                write!(
                    reader.get_mut(),
                    "HTTP/1.1 {status} X\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                .unwrap();
            }
            statuses.len()
        });
        (url, server)
    }

    /// A transient answer is asked again, and the first final one returned.
    #[test]
    fn a_transient_failure_is_retried_until_it_clears() {
        let (url, server) = serve(&[503, 429, 200]);
        let client = reqwest::blocking::Client::new();
        let resp = send("test", Duration::from_secs(30), |_| client.get(&url)).unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(server.join().unwrap(), 3);
    }

    /// A permanent refusal is returned at once, and a zero budget makes one
    /// attempt whatever the answer.
    #[test]
    fn a_final_answer_or_an_empty_budget_stops_at_once() {
        let (url, server) = serve(&[404]);
        let client = reqwest::blocking::Client::new();
        let resp = send("test", Duration::from_secs(30), |_| client.get(&url)).unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        server.join().unwrap();

        let (url, server) = serve(&[503]);
        let resp = send("test", Duration::ZERO, |_| client.get(&url)).unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        server.join().unwrap();
    }
}
