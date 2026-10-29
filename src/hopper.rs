//! The hopper wire: the credential, the HTTP client, and the addresses
//! `--hopper` names. Retries follow `crate::retry`, as every call out does.
//!
//! Hopper holds the corpus: every result this process files, and every verdict
//! it may reuse instead of analyzing again. [`crate::upload`] files results,
//! [`crate::corpus_precheck`] asks before a dependency is analyzed, and the
//! server's corpus reader answers lookups the local index cannot; all three
//! reach hopper through what is here.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

use reqwest::StatusCode;

/// The process's hopper token, and the file it was (or would have been)
/// read from — resolved once, environment included.
///
/// The first non-empty line of `~/.tok/hopper` — the same convention as
/// `~/.tok/openrouter` and `~/.tok/scan`. A supervised service inherits its
/// account's `HOME`, so it finds the file with no plumbing, and the token
/// stays off argv and out of the environment.
///
/// Resolved once per process: hopper reads its own copy once at startup too,
/// so rotation is a restart on both ends.
fn credentials() -> &'static (Option<String>, Option<PathBuf>) {
    static CREDENTIALS: OnceLock<(Option<String>, Option<PathBuf>)> = OnceLock::new();
    CREDENTIALS.get_or_init(|| {
        let path = crate::interpret::tok_path("hopper");
        (
            path.as_deref().and_then(crate::interpret::read_token_file),
            path,
        )
    })
}

/// Bearer token for hopper's API, or `None` when hopper is unauthenticated.
/// Never the origin, never logged.
#[must_use]
pub fn token() -> Option<&'static str> {
    credentials().0.as_deref()
}

/// The process's blocking HTTP client for hopper, with `REQUEST_TIMEOUT` as
/// its default per-request ceiling.
///
/// Built on first use, on whichever plain thread first needs it — the uploader
/// thread or an analysis thread, never the async runtime `serve` builds its
/// uploader from, where reqwest refuses to build a blocking client and panics
/// dropping the half-built one. A static is never dropped there either. `None`,
/// said once, when it cannot be built: callers then stop rather than fall back
/// to a client without timeouts.
pub fn http() -> Option<&'static reqwest::blocking::Client> {
    static HTTP: OnceLock<Option<reqwest::blocking::Client>> = OnceLock::new();
    HTTP.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            // A blackholed address must fail fast so the retry moves on to the
            // next endpoint, rather than spend the transfer ceiling connecting.
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .map_err(|e| {
                tracing::error!(error = %error_chain(&e), "cannot build the hopper HTTP client; hopper calls disabled");
            })
            .ok()
    })
    .as_ref()
}

/// Split `--hopper` into the endpoints to try, in preference order.
///
/// One address is the ordinary case. Several, comma-separated, name the same
/// hopper reached two ways: put the replica first and the primary behind it,
/// and a replica outage costs a retry rather than a lost verdict. Reads and
/// writes take the same list on purpose. Routing them separately is a topology
/// this server would have to
/// know, and hopper's write relay exists precisely so it does not: a replica
/// answers lookups locally and forwards the renewals.
#[must_use]
pub fn endpoints(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|url| url.trim().trim_end_matches('/').trim().to_string())
        .filter(|url| !url.is_empty())
        .collect()
}

/// Warn, once per address, when the hopper token would cross the network in
/// cleartext: a plain `http://` hopper that is not on loopback. The token is
/// fleet-wide write access to the corpus, so it should only travel over TLS or
/// a network that encrypts on its own (a WireGuard/Tailscale address, which
/// this cannot tell apart and so still names).
pub fn warn_if_cleartext(raw: &str) {
    if token().is_none() {
        return;
    }
    for endpoint in endpoints(raw) {
        let Ok(url) = reqwest::Url::parse(&endpoint) else {
            continue;
        };
        let host = url
            .host_str()
            .unwrap_or_default()
            .trim_start_matches('[')
            .trim_end_matches(']');
        let loopback = host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback());
        if url.scheme() == "http" && !loopback {
            tracing::warn!(
                hopper = %endpoint,
                "the hopper token is sent to this address in cleartext; use https unless the network itself is encrypted"
            );
        }
    }
}

/// The one address a queue worker (postdoc's, for one) may poll: the primary,
/// which [`endpoints`] puts last.
///
/// Worker routes are the exception to the rule above. A replica answers
/// lookups and relays renewals, but it refuses `/api/next` and the plain
/// `/api/result` with a 403 even when its relay is enabled — the fleet's queue
/// is the primary's to hand out, and passing the worker firehose through a
/// replica helps no one. So there is nothing to fail over to here: the second
/// address is not another way to reach the same answer, it is the only one.
///
/// Returns `None` for an empty or blank `--hopper`, which is how the deploy
/// says "do not file results anywhere".
#[must_use]
pub fn worker_endpoint(raw: &str) -> Option<String> {
    endpoints(raw).pop()
}

/// One hopper route, at every address it can be reached.
///
/// Ordered as `--hopper` named them. A retry walks down the list rather than
/// hammering one address, so the second attempt after a replica stops answering
/// lands on the primary instead of on the same silence.
#[derive(Debug, Clone)]
pub(crate) struct Route(Vec<String>);

impl Route {
    pub(crate) fn new(bases: &[String], suffix: &str) -> Self {
        Self(bases.iter().map(|base| format!("{base}{suffix}")).collect())
    }

    /// The address to use on this attempt, clamped to the last: a budget longer
    /// than the list keeps retrying the final address rather than wrapping back
    /// to one already known to be failing.
    pub(crate) fn at(&self, attempt: usize) -> &str {
        let last = self.0.len().saturating_sub(1);
        self.0.get(attempt.min(last)).map_or("", String::as_str)
    }

    /// Every address, for a caller that tries each exactly once rather than
    /// retrying on a schedule.
    pub(crate) fn each(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }
}

/// Point this process at `hopper_url`: report where the bearer token came
/// from, and arm [`crate::corpus_precheck`] against the same hopper. Every
/// entry point that reaches hopper calls this, so the hopper a process files
/// to is the hopper it asks.
pub fn configure(hopper_url: &str) {
    log_hopper_credential();
    crate::corpus_precheck::configure(hopper_url);
}

/// Log where the hopper credential came from, or warn that there is none.
///
/// Hopper requires `Authorization: Bearer <token>` on every route and does not
/// exempt loopback, so an unauthenticated process is rejected with 401 on
/// every request. Say so once at startup rather than
/// leaving an operator to infer it from a retry loop.
fn log_hopper_credential() {
    let (token, path) = credentials();
    let path = path
        .as_deref()
        .unwrap_or(std::path::Path::new(""))
        .display();
    match token {
        Some(_) => tracing::info!(source = %path, "hopper API token loaded"),
        None => tracing::warn!(
            expected = %path,
            "no hopper API token found; unless hopper runs unauthenticated every \
             request will be rejected with 401 — install the token at \
             ~/.tok/hopper (mode 0600)",
        ),
    }
}

/// Attach a bearer token to a blocking request, if there is one.
pub(crate) fn bearer(
    request: reqwest::blocking::RequestBuilder,
    token: Option<&str>,
) -> reqwest::blocking::RequestBuilder {
    match token {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

/// Request timeout per call, the same for every attempt.
///
/// `/api/result` is not a request a client can abandon cheaply. Hopper writes
/// the parent and every archive member in one transaction, on a context
/// detached from the request, so a client that gives up cancels nothing: the
/// first store keeps running and holding one of hopper's reserved renewal
/// slots, and a retry sends a second copy to contend with it on the same rows.
/// A store slower than 15 seconds is ordinary under load, so the ceiling is
/// generous.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// How long any hopper call may spend establishing its connection.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Flatten an error and its `source()` chain into one message. reqwest's
/// top-level `Display` is just "error sending request for url (...)"; the real
/// cause (connection refused, DNS failure, timeout) lives one or more links down
/// the chain, so log the whole chain to make a failed upload diagnosable.
pub fn error_chain(err: &dyn std::error::Error) -> String {
    use std::fmt::Write;
    let mut out = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        let _ = write!(out, ": {cause}");
        source = cause.source();
    }
    out
}

/// Whether hopper refused a request for good: any 4xx but 408 and 429, which
/// ask to be retried. Resending a permanent refusal can never succeed.
#[must_use]
pub fn is_permanent(status: StatusCode) -> bool {
    status.is_client_error()
        && status != StatusCode::REQUEST_TIMEOUT
        && status != StatusCode::TOO_MANY_REQUESTS
}

/// How long a hopper lookup with a safe fallback retries one address before
/// moving to the next: long enough to ride out a restart or a shed, short
/// enough that an outage costs seconds, not minutes, on the path waiting on it.
pub(crate) const LOOKUP_RETRY_BUDGET: Duration = Duration::from_secs(15);

/// The sha256 of the content hopper holds under `purl`, asking each address
/// in turn and retrying transient failures within `LOOKUP_RETRY_BUDGET`;
/// `None` on any failure or absence, so a caller posts nothing rather than
/// guess.
#[must_use]
pub fn sha_for_purl(hopper_url: &str, purl: &str) -> Option<String> {
    let client = http()?;
    #[derive(serde::Deserialize)]
    struct LookupResponse {
        #[serde(default)]
        sha256: Option<String>,
    }
    let _span = tracing::info_span!("purl_lookup", purl).entered();
    for base in endpoints(hopper_url) {
        let url = format!("{base}/v1/lookup");
        let lookup = crate::retry::send("hopper purl lookup", LOOKUP_RETRY_BUDGET, |_| {
            bearer(client.get(&url).query(&[("purl", purl)]), token())
        });
        match lookup {
            Ok(resp) if resp.status().is_success() => {
                return resp
                    .json::<LookupResponse>()
                    .map_err(|e| tracing::warn!(endpoint = %base, error = %error_chain(&e), "hopper: purl lookup answer unreadable"))
                    .ok()
                    .and_then(|r| r.sha256);
            }
            Ok(resp) => {
                tracing::warn!(endpoint = %base, status = %resp.status(), "hopper: purl lookup non-success");
            }
            Err(e) => {
                tracing::warn!(endpoint = %base, error = %error_chain(&e), "hopper: purl lookup failed");
            }
        }
    }
    None
}

/// One artifact as `/v1/lookup` reports it: hopper's corpus, and isotope13's
/// API, which speaks the same wire.
///
/// Every field is optional because the corpus spans several eras of stored
/// record: `fires_at` in particular is absent for rows written before levels
/// existed, and a missing level must stay missing rather than become a zero.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Record {
    /// The artifact's digest, lowercase hex.
    #[serde(default)]
    pub sha256: Option<String>,
    /// The package it was looked up as, when it was.
    #[serde(default)]
    pub purl: Option<String>,
    /// Whether the artifact was analyzed. Hopper's own answers carry none.
    #[serde(default)]
    pub status: Option<Status>,
    /// The tightest false-positive budget at which it grades hostile.
    #[serde(default)]
    pub fires_at: crate::model::Level,
    /// The scanner build that analyzed it.
    #[serde(default)]
    pub engine_version: Option<String>,
    /// The traits commit it was analyzed under, abbreviated.
    #[serde(default)]
    pub traits_version: Option<String>,
    /// When it was analyzed, RFC 3339.
    #[serde(default)]
    pub analyzed_at: Option<String>,
    /// Why, in a sentence, when the verdict was not the analyzer's own.
    #[serde(default)]
    pub reason: Option<String>,
    /// Its strongest traits.
    #[serde(default)]
    pub findings: Vec<Finding>,
}

/// Whether a looked-up artifact was analyzed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Analyzed; the record carries the verdict.
    Analyzed,
    /// Held, but nobody has analyzed it.
    Unanalyzed,
    /// It could not be fetched or analyzed.
    Unavailable,
    /// A status this build does not know.
    #[serde(other)]
    Unknown,
}

impl Status {
    /// The wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Analyzed => "analyzed",
            Self::Unanalyzed => "unanalyzed",
            Self::Unavailable => "unavailable",
            Self::Unknown => "unknown",
        }
    }
}

/// One of an artifact's strongest traits, as the corpus stores them.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Finding {
    /// The trait id.
    pub id: String,
    /// Criticality rank: 4 suspicious, 5 hostile.
    #[serde(default)]
    pub crit: u8,
    /// A sentence for a person, present only on findings the analyzer did not
    /// produce. A trait id describes itself; a citation does not.
    #[serde(default)]
    pub desc: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `--hopper` may name the same corpus twice: the replica first, the
    /// primary behind it. Reads and writes take the same list, because routing
    /// them apart is a topology this process would have to know and hopper's
    /// write relay exists so that it does not.
    #[test]
    fn hopper_endpoints_are_a_preference_order() {
        assert_eq!(endpoints("https://ro/"), vec!["https://ro"]);
        assert_eq!(
            endpoints(" https://ro/ , http://rw:8081/ "),
            vec!["https://ro", "http://rw:8081"],
        );
        assert!(endpoints("").is_empty());
        assert!(endpoints(" , , ").is_empty());
    }

    /// Queue polling is the exception: hopper refuses `/api/next` on a replica
    /// with a 403 whether or not its relay is on, so a worker takes the primary
    /// — the last address — and nothing else. Handing it the raw string would
    /// put the commas inside a hostname and fail every poll with
    /// `invalid dns name`.
    #[test]
    fn a_worker_polls_the_primary_only() {
        assert_eq!(
            worker_endpoint("https://ro,https://rw").as_deref(),
            Some("https://rw"),
        );
        assert_eq!(
            worker_endpoint(" https://ro/ , https://rw/ ").as_deref(),
            Some("https://rw"),
        );
        // The ordinary single-address case is unchanged.
        assert_eq!(
            worker_endpoint("https://rw/").as_deref(),
            Some("https://rw")
        );
        // Nowhere to file results is a valid deploy, not an address.
        assert_eq!(worker_endpoint(""), None);
        assert_eq!(worker_endpoint(" , , "), None);
        // Whatever a worker polls, it is one address — never a list.
        for raw in ["https://ro,https://rw", "https://rw", " a , b , c "] {
            let picked = worker_endpoint(raw).expect("an address");
            assert!(
                !picked.contains(','),
                "a worker was handed a list: {picked}"
            );
        }
    }

    /// A retry walks down the list rather than hammering one address, so the
    /// attempt after a replica stops answering lands on the primary instead of
    /// on the same silence. Past the end it holds on the last: a budget longer
    /// than the list must not wrap back to an address already known to fail.
    #[test]
    fn a_retry_moves_to_the_next_address() {
        let bases = endpoints("https://ro,http://rw");
        let route = Route::new(&bases, "/api/result");
        assert_eq!(route.at(0), "https://ro/api/result");
        assert_eq!(route.at(1), "http://rw/api/result");
        assert_eq!(route.at(9), "http://rw/api/result");
    }

    /// One address is the ordinary case, and every attempt uses it.
    #[test]
    fn a_single_address_is_used_for_every_attempt() {
        let route = Route::new(&endpoints("https://only"), "/api/known");
        assert_eq!(route.at(0), "https://only/api/known");
        assert_eq!(route.at(5), "https://only/api/known");
    }

    /// 408 and 429 ask to be retried; every other 4xx is final.
    #[test]
    fn only_retryable_refusals_are_retried() {
        assert!(is_permanent(StatusCode::BAD_REQUEST));
        assert!(is_permanent(StatusCode::UNAUTHORIZED));
        assert!(is_permanent(StatusCode::PAYLOAD_TOO_LARGE));
        assert!(!is_permanent(StatusCode::REQUEST_TIMEOUT));
        assert!(!is_permanent(StatusCode::TOO_MANY_REQUESTS));
        assert!(!is_permanent(StatusCode::SERVICE_UNAVAILABLE));
    }
}
