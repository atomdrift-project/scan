//! Deferring a lookup to hopper when the local index has nothing.
//!
//! A scan worker holds a verdict index and a set of filters; hopper holds the
//! corpus behind them. Asking one question and getting one answer is the point:
//! a caller should not have to know that two services exist, nor which of them
//! to reconcile against the other, so a worker that does not know finds out
//! rather than reporting absence.
//!
//! `--hopper` may name several addresses for the same corpus, in preference
//! order: put the replica first and the primary behind it. Reads and writes
//! take the same list, because routing them separately is a topology this
//! worker would have to know and hopper's write relay exists so that it does
//! not — a replica answers lookups locally and forwards the renewals.
//!
//! A head that fails is rested rather than retried on the next request: the
//! cost of an unreachable address is a timeout, and paying that in front of
//! every lookup would be worse than the outage it is reacting to.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// How long a hopper read may take before the endpoint counts as unreachable.
///
/// Short on purpose. Hopper answers this route from an in-memory record pool
/// and never reads its largest column, so a slow answer means something is
/// wrong rather than something is big — and there is a caller waiting on the
/// far end of two hops.
const READ_TIMEOUT: Duration = Duration::from_secs(2);

/// How stale the preferred address may be before its silence stops counting as
/// an answer.
///
/// A replica 404 is taken as "nothing known" so that the miss load — which for
/// a caller gating installs is most of the load — stays off the primary. That
/// bargain assumes the replica is current. When it is not, a verdict the
/// primary already holds reads as absence: analyze a package, get `block`, look
/// it up a minute later and be told nobody has ever seen it.
///
/// Measured at 27 minutes in production, so the assumption needs enforcing
/// rather than asserting. 60s is far above healthy replication, which runs
/// sub-second, and far below any lag that could hide a verdict a caller just
/// asked us to produce.
const MAX_REPLICA_LAG: Duration = Duration::from_secs(60);

/// How long a lag reading is trusted before asking again. Short enough to catch
/// a replica falling behind, long enough that it costs one request a minute
/// rather than one per lookup.
const LAG_TTL: Duration = Duration::from_secs(30);

/// How long the preferred address stops leading after a failure.
///
/// Long enough that a restart or a failover is not fronted by a timeout on
/// every lookup, short enough that a recovered address takes the load back
/// without an operator doing anything.
const REST: Duration = Duration::from_secs(10);

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
    /// Every address the corpus can be reached at, in preference order.
    bases: Vec<String>,
    /// Per-address traffic, parallel to `bases`.
    ///
    /// Failover is otherwise invisible: a fleet quietly reading from the
    /// primary because the replica stopped answering looks exactly like one
    /// reading from the replica, and the first sign is the primary's load. This
    /// is what turns that into a number an operator can see before it becomes a
    /// page.
    traffic: Vec<Address>,
    client: reqwest::Client,
    /// When the preferred address may lead again, or `None` while it leads.
    rested: Mutex<Option<Instant>>,
    /// The preferred address's data lag when last asked, and when we asked.
    /// `None` inside means it could not tell us.
    lag: Mutex<Option<(Instant, Option<Duration>)>>,
    /// What the corpus said, across every address.
    found: AtomicU64,
    nothing: AtomicU64,
    unreachable: AtomicU64,
}

/// One address's share of the traffic.
#[derive(Debug, Default)]
struct Address {
    asked: AtomicU64,
    answered: AtomicU64,
    failed: AtomicU64,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Corpus {
    /// A corpus reader, or `None` when `--hopper` named nothing and there is
    /// nothing behind this worker to defer to.
    pub(crate) fn new(hopper: Option<&str>) -> Option<Arc<Self>> {
        let bases = crate::upload::endpoints(hopper.unwrap_or(""));
        if bases.is_empty() {
            return None;
        }
        let client = reqwest::Client::builder()
            .timeout(READ_TIMEOUT)
            .build()
            .unwrap_or_default();
        let traffic = bases.iter().map(|_| Address::default()).collect();
        Some(Arc::new(Self {
            bases,
            traffic,
            client,
            rested: Mutex::new(None),
            lag: Mutex::new(None),
            found: AtomicU64::new(0),
            nothing: AtomicU64::new(0),
            unreachable: AtomicU64::new(0),
        }))
    }

    /// The addresses this corpus can be reached at, for a startup log line.
    pub(crate) fn addresses(&self) -> String {
        self.bases.join(", ")
    }

    /// The addresses to try, in order.
    ///
    /// A rested head goes last rather than being dropped: if everything behind
    /// it is also down, an address we believe to be failing still beats no
    /// address, and trying it costs nothing when something earlier answers.
    fn order_at(&self, now: Instant) -> Vec<&str> {
        let mut order: Vec<&str> = self.bases.iter().map(String::as_str).collect();
        if order.len() > 1 && lock(&self.rested).is_some_and(|until| now < until) {
            order.rotate_left(1);
        }
        order
    }

    /// Record how an address behaved. Only the preferred one's standing
    /// changes: everything else is where it falls back to, and demoting those
    /// would leave nowhere to fall.
    fn note_at(&self, base: &str, reachable: bool, now: Instant) {
        if self.bases.first().map(String::as_str) != Some(base) {
            return;
        }
        let mut rested = lock(&self.rested);
        *rested = if reachable { None } else { Some(now + REST) };
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
        let mut path = format!("/v1/lookup?{}", query.join("&"));
        // A replica too far behind cannot be believed about an absence, so its
        // reads are relayed to the primary until it catches up. `fresh=1` is
        // hopper's own read-after-write hatch and a no-op on the primary, which
        // has no relay to forward to — so this is safe to send to whichever
        // address answers.
        if self.preferred_is_stale().await {
            path.push_str("&fresh=1");
        }

        for base in self.order_at(Instant::now()) {
            self.count(base, |a| &a.asked);
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
                    self.count(base, |a| &a.answered);
                    match &reached {
                        Reached::Record(_) => &self.found,
                        _ => &self.nothing,
                    }
                    .fetch_add(1, Ordering::Relaxed);
                    return reached;
                }
                None => {
                    self.note_at(base, false, Instant::now());
                    self.count(base, |a| &a.failed);
                    tracing::warn!(endpoint = %base, "corpus unreachable");
                }
            }
        }
        // Every address failed, which the per-address warnings above say one at
        // a time and nobody reads that way. Say it once, plainly: this is the
        // case where a caller gating installs gets `unavailable` and has to
        // decide what to do without us.
        tracing::error!(
            endpoints = %self.addresses(),
            "corpus unreachable at every address; answering unavailable",
        );
        self.unreachable.fetch_add(1, Ordering::Relaxed);
        Reached::Unreachable
    }

    /// Whether the preferred address is too far behind to be trusted about an
    /// absence.
    ///
    /// Only the preferred address is asked: it is the one whose 404 we would
    /// otherwise believe, and the addresses behind it are where a stale reading
    /// sends the traffic. A reading is cached for [`LAG_TTL`] so this costs a
    /// request a minute rather than one per lookup.
    ///
    /// An address that cannot tell us its lag is trusted as before. The other
    /// choice — treat silence as staleness — relays every read to the primary
    /// the moment this endpoint has a bad day, which is the load the replica
    /// exists to prevent. A single address is never stale by this measure:
    /// there is nowhere to send the traffic instead.
    async fn preferred_is_stale(&self) -> bool {
        if self.bases.len() < 2 {
            return false;
        }
        let Some(base) = self.bases.first() else {
            return false;
        };
        let now = Instant::now();
        let cached = lock(&self.lag)
            .filter(|(at, _)| now.duration_since(*at) < LAG_TTL)
            .map(|(_, lag)| lag);
        let lag = match cached {
            Some(lag) => lag,
            None => {
                let fresh = self.ask_lag(base).await;
                *lock(&self.lag) = Some((now, fresh));
                if let Some(lag) = fresh.filter(|lag| *lag >= MAX_REPLICA_LAG) {
                    tracing::warn!(
                        endpoint = %base,
                        lag_secs = lag.as_secs(),
                        max_secs = MAX_REPLICA_LAG.as_secs(),
                        "corpus replica is behind; relaying reads to the primary",
                    );
                }
                fresh
            }
        };
        lag.is_some_and(|lag| lag >= MAX_REPLICA_LAG)
    }

    /// One address's reported data lag, or `None` when it did not say.
    async fn ask_lag(&self, base: &str) -> Option<Duration> {
        #[derive(serde::Deserialize)]
        struct Status {
            #[serde(default)]
            lag_seconds: Option<i64>,
        }
        let mut request = self.client.get(format!("{base}/_/replica"));
        if let Some(token) = crate::upload::hopper_token() {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.ok()?;
        if !response.status().is_success() {
            return None;
        }
        let status = response.json::<Status>().await.ok()?;
        status
            .lag_seconds
            .filter(|secs| *secs >= 0)
            .and_then(|secs| u64::try_from(secs).ok())
            .map(Duration::from_secs)
    }

    /// Bump one of an address's counters.
    fn count(&self, base: &str, pick: impl Fn(&Address) -> &AtomicU64) {
        if let Some(i) = self.bases.iter().position(|b| b == base) {
            pick(&self.traffic[i]).fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A snapshot for `/_/stats`.
    pub(crate) fn stats(&self) -> serde_json::Value {
        let by_address: Vec<_> = self
            .bases
            .iter()
            .zip(&self.traffic)
            .map(|(base, a)| {
                serde_json::json!({
                    "address": base,
                    "asked": a.asked.load(Ordering::Relaxed),
                    "answered": a.answered.load(Ordering::Relaxed),
                    "failed": a.failed.load(Ordering::Relaxed),
                })
            })
            .collect();
        let resting = lock(&self.rested).is_some_and(|until| Instant::now() < until);
        serde_json::json!({
            // Every deferral, and what came back. `unreachable` is the one that
            // reaches a caller as a decision about us rather than the artifact.
            "found": self.found.load(Ordering::Relaxed),
            "nothing": self.nothing.load(Ordering::Relaxed),
            "unreachable": self.unreachable.load(Ordering::Relaxed),
            // Non-zero `failed` on the first address with traffic on a later one
            // is a failover in progress, whether or not anyone has noticed.
            "preferred_resting": resting,
            "by_address": by_address,
        })
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
            // Being turned away is about this endpoint, not about the query:
            // a credential can be wrong for one address and right for the
            // next, and 403 is exactly how a replica declines a route it will
            // not serve. Answering "nothing known" here would be a claim about
            // the artifact made on the strength of never having asked, and it
            // would strand every lookup on a replica whose token went stale.
            401 | 403 => {
                tracing::warn!(endpoint = %base, status = response.status().as_u16(), "corpus turned us away");
                None
            }
            // Any other 4xx is this request being wrong, which the next
            // endpoint would also say. 5xx is the endpoint being unwell, which
            // it might not.
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

    fn corpus(addresses: &str) -> Arc<Corpus> {
        Corpus::new(Some(addresses)).expect("configured")
    }

    #[test]
    fn nothing_configured_is_no_corpus() {
        assert!(Corpus::new(None).is_none());
        // A blank setting is not an address, and treating it as one would put a
        // doomed request in front of every lookup.
        assert!(Corpus::new(Some("  ")).is_none());
        assert!(Corpus::new(Some(" , , ")).is_none());
    }

    /// Reads belong on the replica; the primary is where they fall back to.
    #[test]
    fn the_replica_leads() {
        let c = corpus("http://ro, http://rw/");
        assert_eq!(c.order_at(Instant::now()), vec!["http://ro", "http://rw"]);
    }

    /// A failing replica must not be retried in front of every lookup: the cost
    /// of an unreachable endpoint is a timeout, and paying it on the way to
    /// every answer is worse than the outage it reacts to.
    #[test]
    fn a_failed_replica_stops_leading_then_takes_it_back() {
        let c = corpus("http://ro, http://rw/");
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
            c.order_at(t0 + REST - Duration::from_millis(1)).first(),
            Some(&"http://rw")
        );
        assert_eq!(
            c.order_at(t0 + REST).first(),
            Some(&"http://ro"),
            "a rested replica never took the load back",
        );
    }

    /// A replica that answers is trusted again immediately — the rest is a
    /// reaction to failure, not a penalty to serve out.
    #[test]
    fn a_recovered_replica_leads_again_at_once() {
        let c = corpus("http://ro, http://rw/");
        let t0 = Instant::now();
        c.note_at("http://ro", false, t0);
        c.note_at("http://ro", true, t0);
        assert_eq!(c.order_at(t0).first(), Some(&"http://ro"));
    }

    /// The primary is the floor. Demoting it on a failure would leave the
    /// fallback order pointing at nothing.
    #[test]
    fn the_primary_is_never_demoted() {
        let c = corpus("http://ro, http://rw/");
        let t0 = Instant::now();
        c.note_at("http://rw", false, t0);
        assert_eq!(c.order_at(t0), vec!["http://ro", "http://rw"]);
    }

    /// One address is the ordinary deployment, and it must not be demoted into
    /// nothing: with nowhere to fall back to, resting the only address we have
    /// would answer `unavailable` for the whole cooldown.
    #[test]
    fn one_address_is_a_valid_deployment() {
        let c = corpus("http://only");
        let t0 = Instant::now();
        assert_eq!(c.order_at(t0), vec!["http://only"]);
        c.note_at("http://only", false, t0);
        assert_eq!(
            c.order_at(t0),
            vec!["http://only"],
            "the only address stopped being tried",
        );
    }

    /// The distinction the whole reliability contract rests on, at the layer
    /// that decides it. A corpus that cannot be reached must report exactly
    /// that: `Nothing` would become `unknown` one layer up and tell a caller
    /// nobody has analyzed the package — a claim about the package rather than
    /// about us, and the one that lets a gate fail open during an outage.
    #[tokio::test]
    async fn an_unreachable_corpus_reports_itself() {
        // Port 1 refuses immediately, so this measures the decision rather than
        // a timeout.
        let c = corpus("http://127.0.0.1:1");
        assert_eq!(
            c.known(None, Some("pkg:npm/left-pad@1.3.0")).await,
            Reached::Unreachable,
        );
    }

    /// And a failed read rests the address, so the next lookup does not pay the
    /// same timeout again before falling back.
    #[tokio::test]
    async fn a_failed_read_rests_the_address() {
        let c = corpus("http://127.0.0.1:1, http://127.0.0.1:2");
        let before = c.order_at(Instant::now());
        assert_eq!(before.first(), Some(&"http://127.0.0.1:1"));
        let _ = c.known(None, Some("pkg:npm/left-pad@1.3.0")).await;
        assert_eq!(
            c.order_at(Instant::now()).first(),
            Some(&"http://127.0.0.1:2"),
            "the address that just failed is still being tried first",
        );
    }

    /// An endpoint that answers every request with `status`, for the failover
    /// tests. Stays up for the life of the test: a corpus may ask it twice.
    fn endpoint(status: &'static str, body: &'static str) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let _ = stream.read(&mut [0u8; 2048]);
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len(),
                );
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    /// A replica too far behind cannot be believed about an absence.
    ///
    /// The bargain is that a replica's 404 is taken as the answer so the miss
    /// load stays off the primary. It holds only while the replica is current.
    /// Measured at 27 minutes behind in production, during which a verdict the
    /// primary already held read as `unknown` — analyze a package, get a
    /// decision, look it up a minute later and be told nobody has seen it.
    #[tokio::test]
    async fn a_replica_that_is_behind_has_its_reads_relayed() {
        let behind = endpoint("200 OK", r#"{"replica":true,"lag_seconds":1620}"#);
        let c = corpus(&format!("{behind},http://127.0.0.1:1"));
        assert!(c.preferred_is_stale().await, "27 minutes behind was believed");

        // Cached, so this costs a request a minute and not one per lookup.
        assert!(c.preferred_is_stale().await);
    }

    /// A replica that is keeping up is believed, which is the whole point of
    /// having one.
    #[tokio::test]
    async fn a_current_replica_is_still_trusted() {
        let current = endpoint("200 OK", r#"{"replica":true,"lag_seconds":2}"#);
        let c = corpus(&format!("{current},http://127.0.0.1:1"));
        assert!(!c.preferred_is_stale().await);
    }

    /// And one that cannot say is trusted as before. Treating silence as
    /// staleness would relay every read to the primary the moment this endpoint
    /// had a bad day — the exact load the replica exists to prevent.
    #[tokio::test]
    async fn an_address_that_cannot_report_its_lag_is_trusted() {
        let quiet = endpoint("404 Not Found", r#"{}"#);
        let c = corpus(&format!("{quiet},http://127.0.0.1:1"));
        assert!(!c.preferred_is_stale().await);

        // Unreachable is the same answer, not a worse one.
        let c = corpus("http://127.0.0.1:1,http://127.0.0.1:2");
        assert!(!c.preferred_is_stale().await);
    }

    /// A single address is never stale by this measure: `fresh=1` would ask the
    /// same machine the same question, and there is nowhere else to send it.
    #[tokio::test]
    async fn one_address_is_never_relayed_past_itself() {
        let behind = endpoint("200 OK", r#"{"replica":true,"lag_seconds":9999}"#);
        let c = corpus(&behind);
        assert!(!c.preferred_is_stale().await);
    }

    /// A stale token on the replica must not read as an empty corpus.
    ///
    /// 401 and 403 are the endpoint declining us, not the corpus reporting on
    /// the artifact — and a credential can be wrong for one address and right
    /// for the next. Taking either as "nothing known" would answer `unknown`
    /// for every package in the corpus, indefinitely, without ever asking the
    /// primary sitting one line down the list.
    #[tokio::test]
    async fn a_refused_credential_falls_through_to_the_primary() {
        for status in ["401 Unauthorized", "403 Forbidden"] {
            let replica = endpoint(status, r#"{"error":"nope"}"#);
            let primary = endpoint("200 OK", r#"{"sha256":"abc","fires_at":-1}"#);
            let c = corpus(&format!("{replica},{primary}"));
            let reached = c.known(None, Some("pkg:npm/left-pad@1.3.0")).await;
            let Reached::Record(record) = reached else {
                panic!("{status} at the replica became {reached:?}");
            };
            assert_eq!(record.sha256.as_deref(), Some("abc"));
        }
    }

    /// And when every address turns us away there is no answer to give. Saying
    /// `Nothing` there would be a statement about the artifact resting on
    /// nothing but our own inability to ask.
    #[tokio::test]
    async fn a_corpus_that_refuses_everywhere_is_unreachable() {
        let a = endpoint("401 Unauthorized", r#"{"error":"nope"}"#);
        let b = endpoint("403 Forbidden", r#"{"error":"nope"}"#);
        let c = corpus(&format!("{a},{b}"));
        assert_eq!(
            c.known(None, Some("pkg:npm/left-pad@1.3.0")).await,
            Reached::Unreachable,
        );
    }

    /// A malformed query, though, really is wrong everywhere: the next address
    /// is running the same build and would say the same thing. Failing over on
    /// it would double every bad request against the primary.
    #[tokio::test]
    async fn a_bad_request_is_not_retried_elsewhere() {
        let replica = endpoint("400 Bad Request", r#"{"error":"nope"}"#);
        let primary = endpoint("200 OK", r#"{"sha256":"abc","fires_at":-1}"#);
        let c = corpus(&format!("{replica},{primary}"));
        assert_eq!(
            c.known(None, Some("pkg:npm/left-pad@1.3.0")).await,
            Reached::Nothing,
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
