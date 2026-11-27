//! The hopper wire: the credential, the HTTP client, and the addresses
//! `--hopper` names. Retries follow `crate::retry`, as every call out does.
//!
//! Hopper holds the corpus: every result this process files, and every verdict
//! it may reuse instead of analyzing again. [`crate::upload`] files results,
//! [`crate::corpus_precheck`] asks before a dependency is analyzed, and the
//! server's corpus reader answers lookups the local index cannot; all three
//! reach hopper through what is here.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
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
            // Hopper refuses a client that does not name itself.
            .user_agent(concat!("atomdrift-scan/", env!("CARGO_PKG_VERSION")))
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
/// hopper reached more than one way: put the replica first and the primary
/// behind it, and a replica outage costs a lookup a retry rather than an
/// answer. Only reads walk this list; writes go to one primary address, chosen
/// once at startup ([`worker_endpoint`]).
#[must_use]
pub fn endpoints(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|url| url.trim().trim_end_matches('/').trim().to_string())
        .filter(|url| !url.is_empty())
        .collect()
}

/// Whether a credential sent to `url` crosses the network in cleartext: plain
/// `http://` to anything but loopback. A WireGuard or Tailscale address, which
/// encrypts on its own, cannot be told apart and so still counts.
pub(crate) fn is_cleartext(url: &reqwest::Url) -> bool {
    let host = url
        .host_str()
        .unwrap_or_default()
        .trim_start_matches('[')
        .trim_end_matches(']');
    let loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    url.scheme() == "http" && !loopback
}

/// Warn, once per address, when the hopper token would cross the network in
/// cleartext (see `is_cleartext`). The token is write access to the corpus,
/// so it should only travel over TLS or a network that encrypts on its own.
pub fn warn_if_cleartext(raw: &str) {
    if token().is_none() {
        return;
    }
    for endpoint in endpoints(raw) {
        if reqwest::Url::parse(&endpoint).is_ok_and(|url| is_cleartext(&url)) {
            tracing::warn!(
                hopper = %endpoint,
                "the hopper token is sent to this address in cleartext; use https unless the network itself is encrypted"
            );
        }
    }
}

/// The one address writes go to: the first `--hopper` address that says it is
/// the primary.
///
/// Queue polls, results, renewals and uploads all land here. A replica serves
/// reads; a write sent to it either costs an extra hop through its relay or,
/// without one, is refused outright, and it refuses the work queue even with a
/// relay. So writes skip every replica in the list.
///
/// The list may name the primary more than one way, a LAN address ahead of a
/// public one, and writes then take the first that answers. That is decided
/// once, at the first call for this list: each address is asked
/// `GET /_/replica` in turn, so one that cannot be reached costs a single
/// `ROLE_PROBE_TIMEOUT` at startup rather than a timeout on every write. The
/// choice holds until the process restarts. When no address answers as the
/// primary (hopper is down or still starting), writes go to the last address,
/// the primary by the ordering [`endpoints`] asks for.
///
/// A single address is used as given, without asking. Returns `None` for an
/// empty or blank `--hopper`, which is how the deploy says "do not file results
/// anywhere".
#[must_use]
pub fn worker_endpoint(raw: &str) -> Option<String> {
    let mut endpoints = endpoints(raw);
    if endpoints.len() <= 1 {
        return endpoints.pop();
    }
    static CHOSEN: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());
    // Held across the probe so concurrent first callers ask once, not each.
    let mut chosen = CHOSEN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(primary) = chosen.get(raw) {
        return Some(primary.clone());
    }
    // On a thread of its own: the blocking client panics when it is built or
    // used on an async runtime's thread, and a caller may be on one.
    let primary = std::thread::scope(|s| s.spawn(|| choose_primary(&endpoints, probe_role)).join())
        .ok()
        .or_else(|| endpoints.last().cloned())?;
    chosen.insert(raw.to_owned(), primary.clone());
    drop(chosen);
    Some(primary)
}

/// How long the startup check may spend asking one address what it is.
const ROLE_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// What one `--hopper` address said it is.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Role {
    Primary,
    Replica,
    /// No answer that names a role: unreachable, refused, or still starting.
    Unknown(String),
}

/// The first of `endpoints` that `role` calls the primary, else the last.
fn choose_primary(endpoints: &[String], role: impl Fn(&str) -> Role) -> String {
    for endpoint in endpoints {
        match role(endpoint) {
            Role::Primary => {
                tracing::info!(hopper = %endpoint, "hopper writes go to this address");
                return endpoint.clone();
            }
            Role::Replica => {
                tracing::info!(hopper = %endpoint, "hopper address is a replica; writes skip it");
            }
            Role::Unknown(reason) => {
                tracing::warn!(hopper = %endpoint, reason = %reason, "hopper address did not say it is the primary; writes skip it");
            }
        }
    }
    let fallback = endpoints.last().cloned().unwrap_or_default();
    tracing::warn!(
        hopper = %fallback,
        "no hopper address answered as the primary; writes go to the last address until restart",
    );
    fallback
}

/// Ask `endpoint` whether it is a replica.
fn probe_role(endpoint: &str) -> Role {
    #[derive(serde::Deserialize)]
    struct Status {
        replica: bool,
    }
    let Some(http) = http() else {
        return Role::Unknown("no HTTP client".into());
    };
    let request = http
        .get(format!("{endpoint}/_/replica"))
        .timeout(ROLE_PROBE_TIMEOUT);
    let response = match bearer(request, token()).send() {
        Ok(response) => response,
        Err(e) => return Role::Unknown(error_chain(&e)),
    };
    let status = response.status();
    if !status.is_success() {
        return Role::Unknown(format!("HTTP {status}"));
    }
    match response.json::<Status>() {
        Ok(Status { replica: true }) => Role::Replica,
        Ok(Status { replica: false }) => Role::Primary,
        Err(e) => Role::Unknown(error_chain(&e)),
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
/// `/v1/work/result` is not a request a client can abandon cheaply. Hopper writes
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
    status.is_client_error() && !crate::retry::is_transient(status)
}

/// How long a hopper lookup with a safe fallback retries one address before
/// moving to the next: long enough to ride out a restart or a shed, short
/// enough that an outage costs seconds, not minutes, on the path waiting on it.
pub(crate) const LOOKUP_RETRY_BUDGET: Duration = Duration::from_secs(15);

/// The sha256 of the content hopper holds under `purl`, asking each address
/// in turn and retrying transient failures within `LOOKUP_RETRY_BUDGET`;
/// `None` on any failure or absence, so a caller posts nothing rather than
/// guess. A pending record (bytes held, never analyzed) counts as absence.
#[must_use]
pub(crate) fn sha_for_purl(hopper_url: &str, purl: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct LookupResponse {
        #[serde(default)]
        sha256: Option<String>,
        #[serde(default)]
        state: Option<State>,
    }
    let client = http()?;
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
                    .filter(|r| r.state != Some(State::Pending))
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

/// One artifact as `/v1/lookup` reports it: hopper's corpus, and Uplink's
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
    /// Hopper's own word on the same question; Uplink's answers carry none.
    #[serde(default)]
    pub state: Option<State>,
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

impl Record {
    /// Whether hopper holds the artifact's bytes but nobody has analyzed
    /// them: an answer from the corpus, but no verdict.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.state == Some(State::Pending)
    }
}

/// Hopper's `state` for a looked-up artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    /// Analyzed; the record carries the verdict.
    Analyzed,
    /// The bytes are held, but nothing has analyzed them yet.
    Pending,
    /// A state this build does not know.
    #[serde(other)]
    Unknown,
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
    use std::sync::Arc;

    use super::*;

    /// `--hopper` may name the same corpus more than once: the replica first,
    /// the primary behind it. Reads walk the list in order.
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

    /// Queue polling is the exception: hopper refuses `/v1/work/next` on a replica
    /// with a 403 whether or not its relay is on, so a worker takes one primary
    /// address and nothing else. Handing it the raw string would put the commas
    /// inside a hostname and fail every poll with `invalid dns name`.
    #[test]
    fn a_worker_polls_one_address() {
        // The ordinary single-address case is used as given, without asking.
        assert_eq!(
            worker_endpoint("https://rw/").as_deref(),
            Some("https://rw")
        );
        // Nowhere to file results is a valid deploy, not an address.
        assert_eq!(worker_endpoint(""), None);
        assert_eq!(worker_endpoint(" , , "), None);
    }

    fn roles<'a>(table: &'a [(&'a str, Role)]) -> impl Fn(&str) -> Role + 'a {
        move |endpoint| {
            table
                .iter()
                .find(|(name, _)| *name == endpoint)
                .map_or(Role::Unknown("not in the table".into()), |(_, role)| {
                    role.clone()
                })
        }
    }

    fn list(raw: &str) -> Vec<String> {
        endpoints(raw)
    }

    /// Public replica first, then the primary on the LAN, then the same
    /// primary through the public tunnel: writes take the LAN.
    #[test]
    fn writes_take_the_first_address_that_is_the_primary() {
        let table = [
            ("https://ro", Role::Replica),
            ("http://lan", Role::Primary),
            ("https://rw", Role::Primary),
        ];
        assert_eq!(
            choose_primary(&list("https://ro,http://lan,https://rw"), roles(&table)),
            "http://lan",
        );
    }

    /// The LAN path is down at startup: writes take the public address.
    #[test]
    fn an_unreachable_primary_address_falls_through_to_the_next() {
        let table = [
            ("https://ro", Role::Replica),
            ("http://lan", Role::Unknown("connection refused".into())),
            ("https://rw", Role::Primary),
        ];
        assert_eq!(
            choose_primary(&list("https://ro,http://lan,https://rw"), roles(&table)),
            "https://rw",
        );
    }

    /// Hopper down or starting: nothing answers, and writes go to the last
    /// address, as they did before the check.
    #[test]
    fn with_no_answer_writes_go_to_the_last_address() {
        let table = [("https://ro", Role::Replica)];
        assert_eq!(
            choose_primary(&list("https://ro,http://lan,https://rw"), roles(&table)),
            "https://rw",
        );
    }

    /// Answer every connection on `listener` with `body`, counting them.
    fn answer_role(body: &'static str) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("address"));
        let asked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = Arc::clone(&asked);
        std::thread::spawn(move || {
            for mut stream in listener.incoming().map_while(Result::ok) {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (base, asked)
    }

    /// End to end over HTTP: the replica and an unreachable address are
    /// skipped, the primary is chosen, and a second call asks nobody.
    #[test]
    fn the_primary_is_found_once_per_process() {
        let (replica, replica_asked) = answer_role(r#"{"replica":true,"lag_seconds":0}"#);
        let (primary, primary_asked) = answer_role(r#"{"replica":false,"lag_seconds":null}"#);
        // Bound then dropped: nothing listens there, so the probe is refused.
        let dead = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
            format!("http://{}", listener.local_addr().expect("address"))
        };
        let raw = format!("{replica},{dead},{primary}");
        assert_eq!(worker_endpoint(&raw), Some(primary.clone()));
        assert_eq!(worker_endpoint(&raw), Some(primary));
        assert_eq!(replica_asked.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(primary_asked.load(std::sync::atomic::Ordering::SeqCst), 1);
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
