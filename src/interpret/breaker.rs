//! Bounding LLM calls: the in-flight cap, the foreground reserve, and the
//! per-endpoint breakers that fail over quickly.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use super::LlmCaller;

/// LLM calls in flight. Each caller waits until the count is below its own
/// limit, so the cap is the caller's `max_concurrency`, not whichever
/// configuration happened to ask first.
pub(super) struct InFlight {
    pub(super) count: Mutex<usize>,
    cv: Condvar,
}

/// The process's one pool: every caller shares the endpoint's capacity.
pub(super) static IN_FLIGHT: InFlight = InFlight::new();

/// RAII permit; releases its slot on drop.
pub(super) struct Permit<'a>(&'a InFlight);

/// In-flight LLM calls background work may hold at once, per process: a
/// quarter of the pool (or `over`, `SCAN_LLM_BACKGROUND_CONCURRENCY`), at
/// least one, never all of it.
///
/// Everything above the cap stays free for foreground callers, so a request
/// with someone waiting on it queues only behind other such requests, never
/// behind background work. A pool of one reserves nothing; there, background
/// work simply takes turns.
///
/// The share is small because an endpoint serving several clients splits each
/// prefill step's token budget across its *running* requests whatever their
/// priority; priority only orders the waiting queue. Background calls holding
/// most of the pool would add seconds to every foreground prompt. Nobody waits
/// on background work, so it drains at whatever rate the endpoint leaves it.
pub(super) fn background_cap(max: usize, over: Option<NonZeroUsize>) -> usize {
    let cap = over.map_or_else(|| (max / 4).max(1), NonZeroUsize::get);
    if max < 2 { max } else { cap.min(max - 1) }
}

impl InFlight {
    pub(super) const fn new() -> Self {
        Self {
            count: Mutex::new(0),
            cv: Condvar::new(),
        }
    }

    /// Wait for a slot under `caller`'s limit: `max` for a foreground caller,
    /// [`background_cap`] of it for background work.
    pub(super) fn acquire(
        &self,
        max: NonZeroUsize,
        background: Option<NonZeroUsize>,
        caller: LlmCaller,
    ) -> Permit<'_> {
        let limit = match caller {
            LlmCaller::Foreground => max.get(),
            LlmCaller::Background => background_cap(max.get(), background),
        };
        let mut count = self.count.lock().unwrap_or_else(PoisonError::into_inner);
        while *count >= limit {
            count = self.cv.wait(count).unwrap_or_else(PoisonError::into_inner);
        }
        *count += 1;
        drop(count);
        Permit(self)
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        {
            let mut count = self.0.count.lock().unwrap_or_else(PoisonError::into_inner);
            *count -= 1;
        } // release the lock before waking waiters
        // All of them, not one: a background waiter woken while the count
        // sits at its lower limit goes straight back to sleep, and the
        // foreground waiter it was woken instead of would hang.
        self.0.cv.notify_all();
    }
}

// ── per-endpoint breaker ────────────────────────────────────────────────────
// Without a breaker, a dead primary in front of a live fallback costs every
// call the primary's timeout before the fallback grades it, and a saturated
// primary keeps prefilling requests whose client has already given up.
//
// So each endpoint carries its own breaker. Consecutive transport failures
// open it; while open the endpoint is skipped outright and the next one in
// the chain is tried at once. When every endpoint is open, a call fails
// immediately instead of waiting. After the cooldown exactly one call probes
// an endpoint, and that call's outcome closes the breaker or re-arms the
// cooldown.

/// How long a tripped endpoint is skipped before one request probes it again.
pub(super) const BREAKER_COOLDOWN: Duration = Duration::from_secs(60);
/// Consecutive transport failures that trip an endpoint's breaker. One is a
/// blip; three in a row against a chain that answers in seconds is an outage.
pub(super) const BREAKER_TRIP_AFTER: u32 = 3;

#[derive(Default)]
pub(super) struct BreakerState {
    /// Transport failures since the endpoint last answered, probes included.
    pub(super) consecutive_failures: u32,
    /// `None` while closed.
    open: Option<Opened>,
}

/// An open breaker.
struct Opened {
    /// When it opened, or when its last probe failed.
    since: Instant,
    /// A half-open probe is in flight: other callers keep skipping until it
    /// reports back.
    probing: bool,
}

impl Opened {
    fn now() -> Self {
        Self {
            since: Instant::now(),
            probing: false,
        }
    }
}

/// What the breaker says to do with an endpoint right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Admit {
    /// Closed: send.
    Send,
    /// Open past its cooldown and nobody probing yet: this call is the probe.
    Probe,
    /// Open: skip it.
    Skip,
}

/// Per-endpoint breakers, keyed by base URL. Tests build their own so the
/// process-wide instance never carries one test's failures into another.
pub(super) struct Breakers {
    pub(super) inner: Mutex<HashMap<String, BreakerState>>,
    pub(super) cooldown: Duration,
}

pub(super) static BREAKERS: OnceLock<Breakers> = OnceLock::new();

impl Breakers {
    pub(super) fn new(cooldown: Duration) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            cooldown,
        }
    }

    /// Run `f` on one endpoint's state, holding the registry lock only for
    /// the duration of the call.
    fn with_state<R>(&self, endpoint: &str, f: impl FnOnce(&mut BreakerState) -> R) -> R {
        let mut g = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let out = f(g.entry(endpoint.to_string()).or_default());
        drop(g);
        out
    }

    pub(super) fn admit(&self, endpoint: &str) -> Admit {
        let cooldown = self.cooldown;
        self.with_state(endpoint, |st| match &mut st.open {
            None => Admit::Send,
            Some(open) if !open.probing && open.since.elapsed() >= cooldown => {
                open.probing = true;
                Admit::Probe
            }
            Some(_) => Admit::Skip,
        })
    }

    /// The endpoint answered. Any HTTP response counts, a refusal included: a
    /// 4xx is the model's problem, not the transport's, and it proves the host
    /// is there. Returns whether this closed an open breaker.
    pub(super) fn answered(&self, endpoint: &str) -> bool {
        self.with_state(endpoint, |st| {
            let was_open = st.open.is_some();
            *st = BreakerState::default();
            was_open
        })
    }

    /// A transport failure (unreachable, timeout, 5xx). Returns whether this
    /// call opened the breaker; a failed probe re-arms the cooldown instead.
    pub(super) fn failed(&self, endpoint: &str) -> bool {
        self.with_state(endpoint, |st| {
            st.consecutive_failures += 1;
            match &mut st.open {
                Some(open) => {
                    *open = Opened::now();
                    false
                }
                None if st.consecutive_failures >= BREAKER_TRIP_AFTER => {
                    st.open = Some(Opened::now());
                    true
                }
                None => false,
            }
        })
    }
}
