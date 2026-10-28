//! Single-flight de-duplication of concurrent identical analyses.
//!
//! Two clients asking about the same artifact should cost one analysis, not
//! two. Beamline already collapses duplicate lookups, but only within one
//! Worker isolate: it runs many, and a client that retries after a `202`
//! usually lands on a different one, so the same sha can arrive here several
//! times while the first analysis is still running. Each duplicate would take a
//! slot of its own and push the server toward `429 At capacity` for work it is
//! already doing.
//!
//! The first request for a key *leads*: it runs the analysis in a server-owned
//! task and publishes the outcome. Later requests *follow* — they take no slot
//! and wait for that same outcome. Because the analysis outlives any one
//! request, the leader hanging up does not abandon the followers, and nobody
//! hanging up cancels it: its verdict is filed either way (see
//! [`Attachment`]'s `Drop`).

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use tokio::sync::watch;

use super::error::ApiError;
use crate::analysis::RequestPhase;
use crate::engine::ScanResult;

/// What a flight analyzes.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) enum Artifact {
    /// Lowercase hex SHA-256 of uploaded bytes.
    ///
    /// The bytes are the identity; the upload filename is not part of it, so
    /// concurrent requests for one artifact share a single analysis and the
    /// first one in decides the answer. cleave does type the file by its
    /// extension, so a follower that named the same bytes differently gets the
    /// leader's framing of them — an acceptable trade for one analysis per
    /// artifact, since it takes a deliberate race to arrange.
    Sha(String),
    /// A package URL, canonical.
    Purl(String),
    /// An exact fetch URL.
    Url(String),
}

/// What makes two requests the same piece of work: the artifact, and the
/// follow policy it is analyzed under.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct FlightKey {
    pub(super) artifact: Artifact,
    /// The request's follow selection, when it is not the server's own;
    /// `None` keeps requests under the server's policy together.
    pub(super) follow: Option<u8>,
}

impl FlightKey {
    /// The key for `artifact` analyzed under `requested`, scoped by policy only
    /// when that differs from the server's `configured` one.
    pub(super) fn new(artifact: Artifact, requested: u8, configured: u8) -> Self {
        Self {
            artifact,
            follow: (requested != configured).then_some(requested),
        }
    }

    /// The package this analysis was requested by, when it was requested by
    /// one. Uploads carry a digest and a filename, never a locator.
    pub(super) fn purl(&self) -> Option<&str> {
        match &self.artifact {
            Artifact::Purl(purl) => Some(purl),
            Artifact::Sha(_) | Artifact::Url(_) => None,
        }
    }
}

impl From<Artifact> for FlightKey {
    /// `artifact` under the server's own policy.
    fn from(artifact: Artifact) -> Self {
        Self {
            artifact,
            follow: None,
        }
    }
}

impl From<&FlightKey> for super::access::Subject {
    /// The key a flight was created under is exactly the artifact its requests
    /// were about, so the access line takes it verbatim: uploads are named by
    /// the digest of their bytes, PURL analyses by the canonical locator.
    fn from(key: &FlightKey) -> Self {
        match &key.artifact {
            Artifact::Sha(sha) => Self::sha256(sha),
            Artifact::Purl(purl) => Self::purl(purl, None),
            Artifact::Url(url) => Self::url(url, None),
        }
    }
}

impl fmt::Display for FlightKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.artifact {
            Artifact::Sha(sha) => write!(f, "sha:{sha}")?,
            Artifact::Purl(purl) => write!(f, "purl:{purl}")?,
            Artifact::Url(url) => write!(f, "url:{url}")?,
        }
        match self.follow {
            Some(follow) => write!(f, " follow:{follow:04b}"),
            None => Ok(()),
        }
    }
}

/// The finished analysis, shared by every request attached to a flight.
#[derive(Debug)]
pub(super) enum Outcome {
    /// A completed report. Each waiter serializes it through
    /// [`ScanResult::envelope_ref`], which borrows the cleave report rather
    /// than cloning it — the report can be hundreds of kilobytes.
    Report(Box<ScanResult>),
    /// Anything that ends in an error response. `anyhow::Error` is not `Clone`,
    /// so a failure, a timeout, or a panic is turned into its answer once by
    /// the leader, and each waiter renders it in its own route's shape.
    Failed(ApiError),
}

impl Outcome {
    /// The outcome published when a leader's task dies without producing one.
    /// Without it every follower would wait out its own client timeout.
    fn abandoned() -> Self {
        Self::Failed(ApiError {
            message: "analysis did not complete".into(),
            ..ApiError::internal()
        })
    }
}

/// One analysis that concurrent identical requests share.
#[derive(Debug)]
pub(super) struct Flight {
    key: FlightKey,
    /// Raised when the analysis outruns `--analysis-timeout` or the server
    /// shuts down; cleave polls it at its checkpoints and stops. Callers
    /// leaving does *not* raise it — see [`Attachment`]'s `Drop` for why the
    /// work outlives them.
    cancellation: Arc<AtomicBool>,
    /// The run's phase tracker, once the leader has started it. A follower
    /// streaming progress reads the phase here: it does not know the leader's
    /// request, and the run's display name is not its key.
    phase: OnceLock<RequestPhase>,
    /// Holds `None` until the analysis finishes.
    outcome: watch::Sender<Option<Arc<Outcome>>>,
}

impl Flight {
    fn new(key: FlightKey) -> Self {
        let (outcome, _) = watch::channel(None);
        Self {
            key,
            cancellation: Arc::new(AtomicBool::new(false)),
            phase: OnceLock::new(),
            outcome,
        }
    }

    /// Record the phase tracker of the run the leader started.
    pub(super) fn set_phase(&self, phase: RequestPhase) {
        // Only the leader sets it, once.
        let _ = self.phase.set(phase);
    }

    /// The phase the analysis is in, once it has started.
    pub(super) fn phase(&self) -> Option<String> {
        self.phase.get().map(RequestPhase::get)
    }

    /// The key this analysis is shared under.
    pub(super) const fn key(&self) -> &FlightKey {
        &self.key
    }

    /// Cancellation flag for the analysis, polled by cleave at its checkpoints.
    pub(super) fn cancellation(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancellation)
    }

    /// Wait for the analysis to finish.
    pub(super) async fn wait(&self) -> Arc<Outcome> {
        let mut rx = self.outcome.subscribe();
        loop {
            let settled = rx.borrow_and_update().clone();
            if let Some(outcome) = settled {
                return outcome;
            }
            if rx.changed().await.is_err() {
                // The sender lives in this `Flight`, which the caller holds
                // through its attachment, so this is unreachable in practice.
                return Arc::new(Outcome::abandoned());
            }
        }
    }
}

/// Registry of analyses currently in progress.
#[derive(Debug, Default)]
pub(super) struct Flights {
    /// Attachment counts live inside the map so that joining and leaving are
    /// atomic with respect to each other: a flight cannot gain a follower
    /// between its count reaching zero and its removal.
    live: Mutex<HashMap<FlightKey, Live>>,
}

#[derive(Debug)]
struct Live {
    flight: Arc<Flight>,
    attached: usize,
}

/// A poisoned lock here means a panic happened while the map was borrowed. The
/// map is only ever inserted into and removed from under this lock, so there is
/// no half-updated invariant to recover from and the data stays usable.
fn lock(live: &Mutex<HashMap<FlightKey, Live>>) -> MutexGuard<'_, HashMap<FlightKey, Live>> {
    live.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Flights {
    /// Attach to the analysis for `key`, starting one if it is not already
    /// running. The returned [`Attachment`] detaches on drop.
    pub(super) fn join(self: &Arc<Self>, key: FlightKey) -> Attachment {
        let mut live = lock(&self.live);
        let (flight, leader) = match live.get_mut(&key) {
            Some(entry) => {
                entry.attached += 1;
                (Arc::clone(&entry.flight), false)
            }
            None => {
                let flight = Arc::new(Flight::new(key.clone()));
                live.insert(
                    key,
                    Live {
                        flight: Arc::clone(&flight),
                        attached: 1,
                    },
                );
                (flight, true)
            }
        };
        drop(live);
        Attachment {
            flights: Arc::clone(self),
            flight,
            leader,
        }
    }

    /// Hand the leader the sole right to publish this flight's outcome.
    pub(super) fn publisher(self: &Arc<Self>, flight: &Arc<Flight>) -> Publisher {
        Publisher {
            flights: Arc::clone(self),
            flight: Arc::clone(flight),
            published: false,
        }
    }

    /// Snapshot how much work de-duplication is saving right now.
    pub(super) fn census(&self) -> Census {
        let live = lock(&self.live);
        let census = Census {
            analyses: live.len(),
            attached: live.values().map(|entry| entry.attached).sum(),
        };
        drop(live);
        census
    }

    /// Retire `flight` and wake everyone waiting on it.
    fn finish(&self, flight: &Arc<Flight>, outcome: Outcome) {
        let mut live = lock(&self.live);
        // Only retire our own flight: if it was already retired, a later
        // request may have started a fresh one under the same key.
        if live
            .get(flight.key())
            .is_some_and(|entry| Arc::ptr_eq(&entry.flight, flight))
        {
            live.remove(flight.key());
        }
        drop(live);
        // `send_replace` stores the value even with no receivers attached yet,
        // so a follower that subscribes a moment later still sees it.
        let _ = flight.outcome.send_replace(Some(Arc::new(outcome)));
    }
}

/// How many analyses are running and how many requests are riding them. The
/// gap between the two is work that is not being repeated.
#[derive(Clone, Copy, Debug)]
pub(super) struct Census {
    /// Distinct analyses in progress.
    pub(super) analyses: usize,
    /// Requests attached to them — at least one apiece.
    pub(super) attached: usize,
}

/// One request's claim on a flight. Dropping it detaches; the analysis runs on
/// regardless, so a caller who reconnects can rejoin it.
#[derive(Debug)]
pub(super) struct Attachment {
    flights: Arc<Flights>,
    flight: Arc<Flight>,
    leader: bool,
}

impl Attachment {
    /// True for the one request that must run the analysis.
    pub(super) const fn leads(&self) -> bool {
        self.leader
    }

    /// The shared analysis this request is attached to.
    pub(super) const fn flight(&self) -> &Arc<Flight> {
        &self.flight
    }
}

impl Drop for Attachment {
    fn drop(&mut self) {
        let mut live = lock(&self.flights.live);
        let key = self.flight.key();
        match live.get_mut(key) {
            // A later request may have started a fresh flight under this key
            // after ours retired; its count is none of our business.
            Some(entry) if Arc::ptr_eq(&entry.flight, &self.flight) => {
                entry.attached -= 1;
            }
            _ => return,
        }
        drop(live);
        // Deliberately *not* cancelled when the last request leaves. An
        // analysis is worth finishing with nobody waiting: its verdict is
        // indexed locally and renewed on hopper, so a caller that hung up — or
        // a proxy that timed out at two minutes on a twenty-minute run — finds
        // the answer waiting rather than starting again from nothing. The
        // watchdog's `--analysis-timeout` is what bounds a runaway.
        //
        // The flight is left registered for the same reason. Retiring it here
        // would leave the analysis running but unreachable, so the caller who
        // reconnected — the one this is all for — would key a second flight and
        // pay for a run already minutes in. Only [`Flights::finish`] retires a
        // flight, and every flight reaches it: the leader publishes, and a
        // leader that dies without publishing has its outcome published by the
        // [`Publisher`] it dropped.
    }
}

/// The leader's sole right to publish a flight's outcome.
///
/// If it is dropped without publishing — the analysis task panicked, or the
/// runtime shut down under it — an internal-error outcome goes out instead, so
/// followers get an answer rather than waiting out their own timeouts.
#[derive(Debug)]
pub(super) struct Publisher {
    flights: Arc<Flights>,
    flight: Arc<Flight>,
    published: bool,
}

impl Publisher {
    /// Publish the analysis outcome to every attached request.
    pub(super) fn publish(mut self, outcome: Outcome) {
        self.flights.finish(&self.flight, outcome);
        self.published = true;
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        if !self.published {
            self.flights.finish(&self.flight, Outcome::abandoned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn key() -> FlightKey {
        FlightKey::from(Artifact::Sha("a".repeat(64)))
    }

    use axum::http::StatusCode;

    fn rendered(status: StatusCode) -> Outcome {
        Outcome::Failed(ApiError::new(status, "test", "x"))
    }

    fn status_of(outcome: &Outcome) -> Option<StatusCode> {
        match outcome {
            Outcome::Failed(e) => Some(e.status),
            Outcome::Report(_) => None,
        }
    }

    #[test]
    fn first_request_leads_and_the_rest_follow() {
        let flights = Arc::new(Flights::default());
        let leader = flights.join(key());
        let follower = flights.join(key());
        assert!(leader.leads());
        assert!(!follower.leads());
        assert!(Arc::ptr_eq(leader.flight(), follower.flight()));
        assert_eq!(flights.census().analyses, 1);
    }

    /// The same bytes are one analysis however they were named: the digest is
    /// the whole identity, so a second uploader follows rather than paying for
    /// a duplicate run.
    #[test]
    fn the_same_bytes_under_another_name_share_one_analysis() {
        let flights = Arc::new(Flights::default());
        let zip = flights.join(key());
        let txt = flights.join(FlightKey::from(Artifact::Sha("a".repeat(64))));
        assert!(zip.leads());
        assert!(!txt.leads(), "the same bytes follow the run already going");
        assert_eq!(flights.census().analyses, 1);
    }

    #[test]
    fn a_different_key_is_a_different_flight() {
        let flights = Arc::new(Flights::default());
        let a = flights.join(key());
        let b = flights.join(FlightKey::from(Artifact::Purl(
            "pkg:npm/left-pad@1.3.0".into(),
        )));
        assert!(a.leads());
        assert!(b.leads());
        assert_eq!(flights.census().analyses, 2);
    }

    #[test]
    fn different_follow_policies_do_not_share_an_analysis() {
        let flights = Arc::new(Flights::default());
        let sha = "a".repeat(64);
        let dependencies = flights.join(FlightKey {
            artifact: Artifact::Sha(sha.clone()),
            follow: Some(0b0100),
        });
        let none = flights.join(FlightKey {
            artifact: Artifact::Sha(sha),
            follow: Some(0),
        });
        assert!(dependencies.leads());
        assert!(none.leads());
        assert_eq!(flights.census().analyses, 2);
    }

    #[test]
    fn an_analysis_outlives_every_request_that_was_waiting_on_it() {
        let flights = Arc::new(Flights::default());
        let leader = flights.join(key());
        let follower = flights.join(key());
        let flight = Arc::clone(leader.flight());
        let cancellation = flight.cancellation();

        // The leader hanging up must not stop work the follower still wants.
        drop(leader);
        assert!(!cancellation.load(Ordering::Acquire));
        assert_eq!(flights.census().analyses, 1);

        // Nor does the last one leaving: the verdict is indexed locally and
        // renewed on hopper, so the work is not the connection's to lose. Only
        // the watchdog's --analysis-timeout stops a runaway.
        drop(follower);
        assert!(
            !cancellation.load(Ordering::Acquire),
            "a departed caller must not cancel work that still has somewhere to go",
        );
        assert_eq!(
            flights.census().analyses,
            1,
            "the flight stays reachable, so a caller who comes back rejoins it",
        );
    }

    /// The reconnect guarantee. A proxy that gives up at its own ceiling takes
    /// every attachment down with it, but the analysis is still running — so
    /// the caller coming back must ride that run rather than key a second one
    /// for work already minutes old. Retiring the flight when the count hit
    /// zero left the analysis running but unreachable, which is the worst of
    /// both: still burning a slot, and paid for twice.
    #[test]
    fn a_caller_who_reconnects_rejoins_the_running_analysis() {
        let flights = Arc::new(Flights::default());
        let cut = flights.join(key());
        let flight = Arc::clone(cut.flight());
        drop(cut);

        let back = flights.join(key());
        assert!(!back.leads(), "the reconnect started a second analysis");
        assert!(Arc::ptr_eq(&flight, back.flight()));
        assert_eq!(flights.census().attached, 1);

        // And it still ends: publishing retires the flight exactly as before,
        // so a flight nobody rejoins cannot linger past its own outcome.
        flights.publisher(&flight).publish(rendered(StatusCode::OK));
        assert_eq!(flights.census().analyses, 0);
        assert!(
            flights.join(key()).leads(),
            "a finished flight was rejoined"
        );
    }

    /// Reattaching is only worth anything if the verdict actually arrives, so
    /// the caller who reconnects is served by the run they rejoined.
    #[tokio::test]
    async fn a_reconnecting_caller_is_served_by_the_run_it_rejoined() {
        let flights = Arc::new(Flights::default());
        let cut = flights.join(key());
        let flight = Arc::clone(cut.flight());
        let publisher = flights.publisher(&flight);
        drop(cut);

        let back = flights.join(key());
        publisher.publish(rendered(StatusCode::OK));
        let outcome = back.flight().wait().await;
        assert_eq!(status_of(&outcome), Some(StatusCode::OK));
    }

    #[test]
    fn a_retired_flight_is_not_rejoined() {
        let flights = Arc::new(Flights::default());
        let first = flights.join(key());
        flights
            .publisher(first.flight())
            .publish(rendered(StatusCode::OK));
        assert_eq!(flights.census().analyses, 0);

        // The next request starts fresh rather than attaching to a finished run.
        let second = flights.join(key());
        assert!(second.leads());
        assert!(!Arc::ptr_eq(first.flight(), second.flight()));

        // Dropping the retired attachment must not disturb the new flight.
        drop(first);
        assert_eq!(flights.census().analyses, 1);
        assert!(!second.flight().cancellation().load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn followers_receive_the_leader_s_outcome() {
        let flights = Arc::new(Flights::default());
        let leader = flights.join(key());
        let follower = flights.join(key());

        flights
            .publisher(leader.flight())
            .publish(rendered(StatusCode::UNSUPPORTED_MEDIA_TYPE));

        for attachment in [&leader, &follower] {
            let outcome = attachment.flight().wait().await;
            assert_eq!(
                status_of(&outcome),
                Some(StatusCode::UNSUPPORTED_MEDIA_TYPE)
            );
        }
    }

    #[tokio::test]
    async fn a_leader_that_dies_still_answers_its_followers() {
        let flights = Arc::new(Flights::default());
        let leader = flights.join(key());
        let follower = flights.join(key());

        // Task died before publishing: the guard speaks for it.
        drop(flights.publisher(leader.flight()));

        let outcome = follower.flight().wait().await;
        assert_eq!(status_of(&outcome), Some(StatusCode::INTERNAL_SERVER_ERROR));
    }

    /// A follower streaming progress reads the leader's phase through the
    /// flight, not by searching `/_/requests` for an entry named like the
    /// flight's subject: an upload's entry is named by its filename and its
    /// subject is the digest, so that search finds nothing for an upload.

    #[test]
    fn a_follower_reads_the_phase_of_the_run_it_rides() {
        let flights = Arc::new(Flights::default());
        let leader = flights.join(key());
        let follower = flights.join(key());
        assert_eq!(follower.flight().phase(), None, "nothing has started");

        let phase = RequestPhase::with_label("req#7 upload.zip");
        leader.flight().set_phase(phase.clone());
        phase.set("cleave:analyze");
        assert_eq!(follower.flight().phase().as_deref(), Some("cleave:analyze"));
    }

    #[tokio::test]
    async fn a_follower_that_subscribes_late_still_sees_the_outcome() {
        let flights = Arc::new(Flights::default());
        let leader = flights.join(key());
        let follower = flights.join(key());

        flights
            .publisher(leader.flight())
            .publish(rendered(StatusCode::OK));

        // Nobody was awaiting `wait()` when the outcome landed.
        let outcome = follower.flight().wait().await;
        assert_eq!(status_of(&outcome), Some(StatusCode::OK));
    }
}
