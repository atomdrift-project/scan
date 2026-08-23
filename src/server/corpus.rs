//! Deferring a lookup to hopper when the local index has nothing.
//!
//! A scan worker holds a verdict index and a set of filters; hopper holds the
//! corpus behind them. Asking one question and getting one answer is the point:
//! a caller should not have to know that two services exist, nor which of them
//! to reconcile against the other, so a worker that does not know finds out
//! rather than reporting absence.
//!
//! Reads go to the replica first and fall back to the primary. The replica is
//! rested after a failure rather than retried on the next request, because the
//! cost of an unreachable endpoint is a timeout, and paying that in front of
//! every lookup would be worse than the outage it is reacting to.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// How long a hopper read may take before the endpoint counts as unreachable.
///
/// Short on purpose. Hopper answers this route from an in-memory record pool
/// and never reads its largest column, so a slow answer means something is
/// wrong rather than something is big — and there is a caller waiting on the
/// far end of two hops.
const READ_TIMEOUT: Duration = Duration::from_secs(2);

/// How long the replica stops leading after a failure.
///
/// Long enough that a restart or a failover is not fronted by a timeout on
/// every lookup, short enough that a recovered replica takes the load back
/// without an operator doing anything.
const REPLICA_REST: Duration = Duration::from_secs(10);

/// What hopper said about an artifact.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Reached {
    /// The corpus holds a record for it.
    Record(Box<CorpusRecord>),
    /// The corpus answered and holds nothing, or holds bytes nobody has
    /// analyzed. Either way there is no verdict — and nothing is wrong.
    Nothing,
    /// No endpoint could be reached. Says nothing about the artifact, which is
    /// the whole reason it is distinct from [`Self::Nothing`].
    Unreachable,
}

/// One artifact as hopper's `/v1/lookup` reports it.
///
/// Every field is optional because the corpus spans several eras of stored
/// record: `fires_at` in particular is absent for rows written before levels
/// existed, and a missing level must stay missing rather than become a zero.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub(crate) struct CorpusRecord {
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub purl: Option<String>,
    #[serde(default)]
    pub fires_at: Option<i32>,
    #[serde(default)]
    pub engine_version: Option<String>,
    #[serde(default)]
    pub analyzed_at: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub findings: Vec<CorpusFinding>,
}

/// One of the artifact's strongest traits, as the corpus stores them.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub(crate) struct CorpusFinding {
    pub id: String,
    #[serde(default)]
    pub crit: u8,
}

/// The corpus behind this worker's index.
#[derive(Debug)]
pub(crate) struct Corpus {
    replica: Option<String>,
    primary: Option<String>,
    client: reqwest::Client,
    /// When the replica may lead again, or `None` while it is leading.
    rested: Mutex<Option<Instant>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Corpus {
    /// A corpus reader, or `None` when neither endpoint is configured and there
    /// is nothing behind this worker to defer to.
    ///
    /// `replica` is `--hopper-read`, `primary` is `--hopper`. Either alone is a
    /// valid deployment: a worker with only the primary reads from what it
    /// writes to, and one with only a replica reads and never writes.
    pub(crate) fn new(replica: Option<String>, primary: Option<String>) -> Option<Arc<Self>> {
        // Whitespace first, then trailing slashes: a config set to blank is
        // not an endpoint, and treating it as one would put a doomed request in
        // front of every lookup.
        let trim = |s: Option<String>| {
            s.map(|s| s.trim().trim_end_matches('/').trim().to_string())
                .filter(|s| !s.is_empty())
        };
        let (replica, primary) = (trim(replica), trim(primary));
        if replica.is_none() && primary.is_none() {
            return None;
        }
        let client = reqwest::Client::builder()
            .timeout(READ_TIMEOUT)
            .build()
            .unwrap_or_default();
        Some(Arc::new(Self {
            replica,
            primary,
            client,
            rested: Mutex::new(None),
        }))
    }

    /// The endpoints to try, in order.
    ///
    /// A rested replica is tried last rather than not at all: if the primary is
    /// also down, an endpoint we believe to be failing is still a better answer
    /// than none, and trying it costs nothing when the primary succeeds first.
    fn order_at(&self, now: Instant) -> Vec<&str> {
        let resting = lock(&self.rested).is_some_and(|until| now < until);
        match (self.replica.as_deref(), self.primary.as_deref()) {
            (Some(replica), Some(primary)) if resting => vec![primary, replica],
            (Some(replica), Some(primary)) => vec![replica, primary],
            (Some(only), None) | (None, Some(only)) => vec![only],
            (None, None) => Vec::new(),
        }
    }

    /// Record how an endpoint behaved. Only the replica's standing changes:
    /// the primary is where everything falls back to, so demoting it would
    /// leave nowhere to fall.
    fn note_at(&self, base: &str, reachable: bool, now: Instant) {
        if self.replica.as_deref() != Some(base) {
            return;
        }
        let mut rested = lock(&self.rested);
        *rested = if reachable {
            None
        } else {
            Some(now + REPLICA_REST)
        };
    }

    /// Ask the corpus about an artifact, trying each endpoint in turn.
    pub(crate) async fn known(&self, sha: Option<&str>, purl: Option<&str>) -> Reached {
        let mut query = Vec::with_capacity(2);
        if let Some(sha) = sha {
            query.push(format!("sha256={sha}"));
        }
        if let Some(purl) = purl {
            query.push(format!("purl={}", percent_encode(purl)));
        }
        if query.is_empty() {
            return Reached::Nothing;
        }
        let path = format!("/v1/lookup?{}", query.join("&"));

        for base in self.order_at(Instant::now()) {
            match self.ask(base, &path).await {
                // An answer, whichever kind. A 404 from the replica is taken as
                // the answer rather than re-asked at the primary: "we hold
                // nothing" is the common reply for a caller gating installs,
                // and confirming every one of them against the primary would
                // put the whole miss load back on the machine the replica
                // exists to spare. The cost is a narrow window after a write
                // where replication lag reads as absence.
                Some(reached) => {
                    self.note_at(base, true, Instant::now());
                    return reached;
                }
                None => {
                    self.note_at(base, false, Instant::now());
                    tracing::warn!(endpoint = %base, "corpus unreachable");
                }
            }
        }
        Reached::Unreachable
    }

    /// One endpoint's answer, or `None` when it could not give one.
    async fn ask(&self, base: &str, path: &str) -> Option<Reached> {
        let mut request = self.client.get(format!("{base}{path}"));
        if let Some(token) = crate::upload::hopper_token() {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.ok()?;
        match response.status().as_u16() {
            200 => match response.json::<CorpusRecord>().await {
                Ok(record) => Some(Reached::Record(Box::new(record))),
                // Reachable, but not speaking our protocol. Not a reason to try
                // the other endpoint, which is running the same build.
                Err(error) => {
                    tracing::warn!(endpoint = %base, %error, "corpus record did not parse");
                    Some(Reached::Nothing)
                }
            },
            // 404 is "nothing stored"; 202 is "held, nobody has looked at it".
            // Neither is a verdict, and neither is a failure.
            404 | 202 => Some(Reached::Nothing),
            // 4xx is this request being wrong, which the next endpoint would
            // also say. 5xx is the endpoint being unwell, which it might not.
            status if (400..500).contains(&status) => {
                tracing::warn!(endpoint = %base, status, "corpus refused the query");
                Some(Reached::Nothing)
            }
            status => {
                tracing::warn!(endpoint = %base, status, "corpus returned an error");
                None
            }
        }
    }
}

/// Percent-encode a PURL for a query string. A PURL's own grammar carries `?`,
/// `#` and `&`, every one of which would otherwise end the value early and ask
/// about a different artifact than the caller named.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(char::from(byte));
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn corpus(replica: Option<&str>, primary: Option<&str>) -> Arc<Corpus> {
        Corpus::new(replica.map(str::to_string), primary.map(str::to_string)).expect("configured")
    }

    #[test]
    fn nothing_configured_is_no_corpus() {
        assert!(Corpus::new(None, None).is_none());
        assert!(Corpus::new(Some("  ".into()), None).is_none());
    }

    /// Reads belong on the replica; the primary is where they fall back to.
    #[test]
    fn the_replica_leads() {
        let c = corpus(Some("http://ro"), Some("http://rw"));
        assert_eq!(c.order_at(Instant::now()), vec!["http://ro", "http://rw"]);
    }

    /// A failing replica must not be retried in front of every lookup: the cost
    /// of an unreachable endpoint is a timeout, and paying it on the way to
    /// every answer is worse than the outage it reacts to.
    #[test]
    fn a_failed_replica_stops_leading_then_takes_it_back() {
        let c = corpus(Some("http://ro"), Some("http://rw"));
        let t0 = Instant::now();

        c.note_at("http://ro", false, t0);
        assert_eq!(
            c.order_at(t0),
            vec!["http://rw", "http://ro"],
            "a failing replica still leads",
        );

        // Tried last rather than not at all: if the primary is down too, an
        // endpoint we doubt beats no endpoint.
        assert_eq!(
            c.order_at(t0 + REPLICA_REST - Duration::from_millis(1))
                .first(),
            Some(&"http://rw")
        );
        assert_eq!(
            c.order_at(t0 + REPLICA_REST).first(),
            Some(&"http://ro"),
            "a rested replica never took the load back",
        );
    }

    /// A replica that answers is trusted again immediately — the rest is a
    /// reaction to failure, not a penalty to serve out.
    #[test]
    fn a_recovered_replica_leads_again_at_once() {
        let c = corpus(Some("http://ro"), Some("http://rw"));
        let t0 = Instant::now();
        c.note_at("http://ro", false, t0);
        c.note_at("http://ro", true, t0);
        assert_eq!(c.order_at(t0).first(), Some(&"http://ro"));
    }

    /// The primary is the floor. Demoting it on a failure would leave the
    /// fallback order pointing at nothing.
    #[test]
    fn the_primary_is_never_demoted() {
        let c = corpus(Some("http://ro"), Some("http://rw"));
        let t0 = Instant::now();
        c.note_at("http://rw", false, t0);
        assert_eq!(c.order_at(t0), vec!["http://ro", "http://rw"]);
    }

    #[test]
    fn one_endpoint_is_a_valid_deployment() {
        assert_eq!(
            corpus(Some("http://ro"), None).order_at(Instant::now()),
            vec!["http://ro"]
        );
        assert_eq!(
            corpus(None, Some("http://rw")).order_at(Instant::now()),
            vec!["http://rw"]
        );
    }

    /// A PURL carries `?`, `#` and `&` in its own grammar. Left raw, everything
    /// after the first of them is parsed as somebody else's query parameter and
    /// the corpus is asked about a different artifact than the caller named.
    #[test]
    fn a_purl_survives_the_query_string() {
        assert_eq!(
            percent_encode("pkg:npm/@scope/name@1.0.0?arch=x64#sub"),
            "pkg%3Anpm%2F%40scope%2Fname%401.0.0%3Farch%3Dx64%23sub",
        );
    }
}
