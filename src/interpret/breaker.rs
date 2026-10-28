//! Bounding LLM calls: the in-flight cap, the foreground reserve, and the
//! per-endpoint and whole-chain breakers that fail over quickly.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// LLM calls in flight. Each caller waits until the count is below its own
/// limit, so the cap is the caller's `max_concurrency`, not whichever
/// configuration happened to ask first.
pub(super) struct InFlight {
    pub(super) count: Mutex<usize>,
    pub(super) cv: Condvar,
}

/// The process's one pool: every caller shares the endpoint's capacity.
pub(super) static IN_FLIGHT: InFlight = InFlight::new();

/// RAII permit; releases its slot on drop.
pub(super) struct Permit<'a>(&'a InFlight);

/// In-flight LLM calls background work may hold at once, per process: a
/// quarter of the pool (or `over`, `SCAN_LLM_BACKGROUND_CONCURRENCY`), at
/// least one, never all of it.
///
/// A process does not compete only with itself: the fleet's workers share one
/// GPU with serve, and vLLM splits every prefill step's token budget across
/// *running* requests whatever their priority; priority only orders the
/// waiting queue. Background calls that hold most of the pool therefore add
/// seconds to every foreground prompt, at any priority. Nobody waits on a
/// queue job, so background work takes the small share and drains at whatever
/// rate the GPU leaves it.
pub(super) fn background_cap(max: usize, over: Option<NonZeroUsize>) -> usize {
    let cap = over.map_or((max / 4).max(1), NonZeroUsize::get);
    if max < 2 { max } else { cap.min(max - 1) }
}

/// Permits background work may never take: everything above
/// [`background_cap`] stays free for foreground callers, so a serve request
/// queues only behind other requests, never behind background work. A pool of one
/// reserves nothing; there, background work simply takes turns.
#[cfg(test)]
pub(super) fn foreground_reserve(max: usize) -> usize {
    max - background_cap(max, None)
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
// The health gate below is chain-wide: it asks whether *any* endpoint answers.
// It cannot see the case this breaker exists for — a dead primary in front of
// a live fallback — where the chain is healthy and every call still pays the
// primary's timeout before the fallback grades. With the primary saturated,
// every request spends the full timeout there before the fallback answers in
// seconds, and the abandoned requests are still prefilled on the host after
// the client is gone.
//
// So each endpoint carries its own breaker. Consecutive transport failures
// open it; while open the endpoint is skipped outright and the next one in
// the chain is tried at once. After the cooldown exactly one call probes it,
// and that call's outcome closes the breaker or re-arms the cooldown.

/// How long a tripped endpoint is skipped before one request probes it again.
pub const BREAKER_COOLDOWN: Duration = Duration::from_secs(60);
/// Consecutive transport failures that trip an endpoint's breaker. One is a
/// blip; three in a row against a chain that answers in seconds is an outage.
pub(super) const BREAKER_TRIP_AFTER: u32 = 3;

#[derive(Default)]
pub(super) struct BreakerState {
    pub(super) consecutive_failures: u32,
    /// When the breaker opened; `None` while closed.
    pub(super) opened_at: Option<Instant>,
    /// A half-open probe is in flight: other callers keep skipping until it
    /// reports back.
    pub(super) probing: bool,
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
pub(crate) struct Breakers {
    pub(super) inner: Mutex<HashMap<String, BreakerState>>,
    pub(super) cooldown: Duration,
}

pub(super) static BREAKERS: OnceLock<Breakers> = OnceLock::new();

pub(super) fn breakers() -> &'static Breakers {
    BREAKERS.get_or_init(|| Breakers::new(BREAKER_COOLDOWN))
}

impl Breakers {
    pub(super) fn new(cooldown: Duration) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            cooldown,
        }
    }

    /// Run `f` on one endpoint's state, holding the registry lock only for
    /// the duration of the call.
    pub(super) fn with_state<R>(
        &self,
        endpoint: &str,
        f: impl FnOnce(&mut BreakerState) -> R,
    ) -> R {
        let mut g = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let out = f(g.entry(endpoint.to_string()).or_default());
        drop(g);
        out
    }

    pub(super) fn admit(&self, endpoint: &str) -> Admit {
        let cooldown = self.cooldown;
        self.with_state(endpoint, |st| match st.opened_at {
            None => Admit::Send,
            Some(at) if !st.probing && at.elapsed() >= cooldown => {
                st.probing = true;
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
            let was_open = st.opened_at.is_some();
            *st = BreakerState::default();
            was_open
        })
    }

    /// A transport failure (unreachable, timeout, 5xx). Returns whether this
    /// call opened the breaker; a failed probe re-arms the cooldown instead.
    pub(super) fn failed(&self, endpoint: &str) -> bool {
        self.with_state(endpoint, |st| {
            st.consecutive_failures += 1;
            st.probing = false;
            if st.opened_at.is_some() {
                st.opened_at = Some(Instant::now());
                return false;
            }
            if st.consecutive_failures >= BREAKER_TRIP_AFTER {
                st.opened_at = Some(Instant::now());
                return true;
            }
            false
        })
    }
}

// ── endpoint health (circuit breaker) ───────────────────────────────────────
// A process-wide breaker over the LLM endpoint. It starts *closed* (optimistic):
// the first request goes straight through. A transport failure opens it; while
// open, callers don't fire doomed requests — exactly one of them probes
// `GET /models` every `HEALTH_RETRY_INTERVAL` and the rest wait, until the
// endpoint recovers or each caller's budget (the request timeout) elapses.

pub(super) struct Health {
    pub(super) inner: Mutex<HealthState>,
    pub(super) cv: Condvar,
}

pub(super) struct HealthState {
    /// Breaker closed (`true`) vs open (`false`). Starts closed.
    pub(super) healthy: bool,
    /// Whether a caller currently owns the probe loop (so only one probes).
    pub(super) probing: bool,
}

pub(super) static HEALTH: OnceLock<Health> = OnceLock::new();

pub(super) fn health() -> &'static Health {
    HEALTH.get_or_init(|| Health {
        inner: Mutex::new(HealthState {
            healthy: true,
            probing: false,
        }),
        cv: Condvar::new(),
    })
}

impl Health {
    /// Record the outcome of a real request: a success closes the breaker, a
    /// transport failure opens it. Wakes any waiters.
    pub(super) fn set(&self, healthy: bool) {
        {
            let mut g = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            g.healthy = healthy;
        } // release the lock before waking waiters
        self.cv.notify_all();
    }

    /// Block until the endpoint is healthy or `budget` elapses; returns whether
    /// it ended up healthy. When closed, returns immediately. When open, the
    /// first caller becomes the sole prober (re-probing every `retry`); others
    /// wait for its result. `probe` is injected so the state machine is testable
    /// without a network.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the guard is held across the condvar loop on purpose"
    )]
    pub(super) fn wait_until_healthy(
        &self,
        budget: Duration,
        retry: Duration,
        probe: &dyn Fn() -> bool,
    ) -> bool {
        let deadline = Instant::now() + budget;
        let mut g = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if g.healthy {
                return true;
            }
            if g.probing {
                // Someone else is probing — wait for their result (or the budget).
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return g.healthy;
                }
                g = self
                    .cv
                    .wait_timeout(g, remaining)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
                continue;
            }
            // Become the sole prober for the whole recovery loop.
            g.probing = true;
            let healthy = loop {
                drop(g);
                let ok = probe();
                g = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
                g.healthy = ok;
                if ok {
                    break true;
                }
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break false;
                }
                // Nap before the next probe, releasing the lock so a concurrent
                // request's `set(true)` can wake us early.
                g = self
                    .cv
                    .wait_timeout(g, retry.min(remaining))
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
                if g.healthy {
                    break true;
                }
            };
            g.probing = false;
            self.cv.notify_all();
            return healthy;
        }
    }
}
