//! Live memory-pressure admission control for analyses.
//!
//! The worker runs up to `slots` analyses concurrently, but slot count says
//! nothing about *memory*: a burst of large archives can co-reside and exhaust
//! RAM — the failure mode that OOM-killed hosts (many archives admitted at once
//! → swap death; every thread then stalled on page-faults, which surfaced as
//! bogus "rule took 30000ms" timings).
//!
//! This gate pauses admission — it never kills the process — once memory reaches
//! a ceiling (the resolved `--max-rss-gb`, default 85% of RAM). Two checks gate
//! each new job, both keyed on signals every supported platform exposes:
//!
//! * **Predictive** — each in-flight analysis reserves a flat per-slot estimate
//!   (1.5 GB by default). A burst commits its full reservation the instant it is
//!   admitted, so the gate closes *before* the archives expand, not after.
//! * **Reactive** — live memory usage (`total − MemAvailable` on Linux, this
//!   process's RSS elsewhere) must leave room for one more slot. This catches
//!   estimates that ran low and pressure from other processes on the host.
//!
//! One always-admit hatch (when nothing is in flight) guarantees forward
//! progress even on a host too small to fit a single slot's estimate.
//!
//! When the gate closes it logs the per-slot breakdown and clears caches once,
//! then re-polls until memory frees. Admission is acquired from the single
//! dispatch loop, so the check-then-reserve sequence races nothing; releases
//! happen on the per-job tokio tasks and only ever *decrease* the reservation,
//! so they cannot cause over-commit.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB: u64 = 1024 * 1024;

/// Assumed peak resident footprint of one analysis slot. Archives decompress and
/// each source member spawns a tree-sitter parse tree, so a slot's true peak is
/// wildly file-dependent and routinely far above its on-disk size; a flat,
/// deliberately pessimistic estimate bounds concurrency safely without pretending
/// to predict per-file blowup. Override with `LITMUS_PER_SLOT_ESTIMATE_MB`.
const DEFAULT_PER_SLOT_ESTIMATE_BYTES: usize = 1536 * 1024 * 1024;

/// Re-poll interval while waiting for memory to free (bounds feedback latency
/// even if no release wakes us).
const REPOLL_INTERVAL: Duration = Duration::from_secs(1);

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse().ok()
}

/// Live memory usage in bytes toward the ceiling, or `None` when no source is
/// readable. Prefers the system-wide figure (`total − MemAvailable`) where the
/// platform exposes live availability (Linux), so the gate respects pressure
/// from *other* processes too; elsewhere falls back to this process's RSS, which
/// every supported platform exposes. The default `used_fn`; tests inject a value.
fn live_used() -> Option<u64> {
    match (
        cleave::memory_tracker::total_memory(),
        cleave::memory_tracker::available_memory(),
    ) {
        (Some(total), Some(avail)) => Some(total.saturating_sub(avail)),
        _ => cleave::memory_tracker::current_rss(),
    }
}

/// One in-flight analysis, recorded for the per-slot memory diagnostics.
#[derive(Debug)]
struct Inflight {
    id: u64,
    sha256: Arc<str>,
    path: Arc<str>,
    file_type: Arc<str>,
    on_disk_bytes: u64,
    admitted: Instant,
}

/// Live memory-pressure gate over all in-flight analyses.
#[derive(Debug)]
pub struct MemoryAdmission {
    /// Memory ceiling in bytes. Admission pauses once committed reservations or
    /// live usage would push past it. `0` disables the gate (slot-limited only),
    /// matching a disabled `--max-rss-gb`.
    ceiling_bytes: u64,
    /// Flat per-slot reservation; see [`DEFAULT_PER_SLOT_ESTIMATE_BYTES`].
    est_bytes: usize,
    /// Sum of in-flight reservations.
    reserved: AtomicUsize,
    next_id: AtomicU64,
    inflight: Mutex<Vec<Inflight>>,
    /// Woken whenever a reservation is released.
    released: Notify,
    /// Live-usage source; a field so tests can inject a fixed value.
    used_fn: fn() -> Option<u64>,
}

impl MemoryAdmission {
    /// Build the gate. `ceiling_bytes` is the resolved `--max-rss-gb` (default
    /// auto = 85% of RAM); `0` disables proactive throttling so an operator who
    /// opts out, or an unsupported platform, degrades to slot-limited dispatch.
    pub fn new(ceiling_bytes: u64) -> Arc<Self> {
        let est_bytes = env_usize("LITMUS_PER_SLOT_ESTIMATE_MB")
            .and_then(|mb| mb.checked_mul(1024 * 1024))
            .filter(|b| *b > 0)
            .unwrap_or(DEFAULT_PER_SLOT_ESTIMATE_BYTES);

        if ceiling_bytes == 0 {
            tracing::warn!(
                "memory admission disabled (max-rss 0); dispatch is slot-limited only"
            );
        } else {
            tracing::info!(
                ceiling_gb = ceiling_bytes as f64 / GIB,
                per_slot_estimate_gb = est_bytes as f64 / GIB,
                "live memory admission enabled",
            );
        }

        Arc::new(Self {
            ceiling_bytes,
            est_bytes,
            reserved: AtomicUsize::new(0),
            next_id: AtomicU64::new(0),
            inflight: Mutex::new(Vec::new()),
            released: Notify::new(),
            used_fn: live_used,
        })
    }

    /// Acquire admission, awaiting free memory asynchronously (worker dispatch
    /// loop). The returned guard releases the reservation on drop. `on_disk_bytes`
    /// is the hopper-reported size, recorded only for the per-slot diagnostics.
    pub async fn admit(
        self: &Arc<Self>,
        sha256: Arc<str>,
        path: Arc<str>,
        file_type: Arc<str>,
        on_disk_bytes: i64,
    ) -> AdmissionGuard {
        let started = Instant::now();
        let mut paused = false;

        loop {
            if self.try_reserve() {
                return self.register(sha256, path, file_type, on_disk_bytes, started, paused);
            }
            if !paused {
                // First time the gate closes for this job: surface where memory
                // is tied up, then reclaim what the caches are holding. Both run
                // once per pause episode — the dispatch loop admits serially, so
                // this cannot thrash.
                self.log_inflight("admission paused: memory at saturation");
                paused = true;
                if let Err(e) = tokio::task::spawn_blocking(cleave::clear_all_thread_caches).await {
                    tracing::warn!(error = %e, "cache-clear task failed");
                }
            }
            // Wait for a release, but wake periodically to re-poll live usage
            // even when no reservation is freed.
            tokio::select! {
                () = self.released.notified() => {}
                () = tokio::time::sleep(REPOLL_INTERVAL) => {}
            }
        }
    }

    /// Reserve one slot if memory allows. Gated on committed reservations and on
    /// live usage, both against `ceiling_bytes`; an always-admit hatch (nothing
    /// in flight) guarantees forward progress on a host too small for one slot.
    ///
    /// Single caller per loop iteration; releases only decrease `reserved`.
    fn try_reserve(&self) -> bool {
        let est = self.est_bytes;

        // Disabled: dispatch is bounded by slot count alone.
        if self.ceiling_bytes == 0 {
            self.reserved.fetch_add(est, Ordering::AcqRel);
            return true;
        }

        let reserved = self.reserved.load(Ordering::Acquire);
        let used = (self.used_fn)();

        // Forward-progress hatch: with nothing of ours in flight, admit one slot
        // unless the host is *already* past the ceiling (pressure from elsewhere).
        if reserved == 0 {
            if used.is_some_and(|u| u > self.ceiling_bytes) {
                return false;
            }
            self.reserved.store(est, Ordering::Release);
            return true;
        }

        // Predictive: committed reservations must leave room for one more slot.
        if (reserved as u64).saturating_add(est as u64) > self.ceiling_bytes {
            return false;
        }

        // Reactive: live usage must leave room for one more slot, catching
        // estimates that ran low and pressure from other processes.
        if let Some(used) = used
            && used.saturating_add(est as u64) > self.ceiling_bytes
        {
            return false;
        }

        self.reserved.store(reserved + est, Ordering::Release);
        true
    }

    /// Record an admitted job (its `est` is already reserved) and return its guard.
    fn register(
        self: &Arc<Self>,
        sha256: Arc<str>,
        path: Arc<str>,
        file_type: Arc<str>,
        on_disk_bytes: i64,
        started: Instant,
        paused: bool,
    ) -> AdmissionGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        #[allow(clippy::expect_used)]
        self.inflight
            .lock()
            .expect("admission registry mutex poisoned")
            .push(Inflight {
                id,
                sha256,
                path,
                file_type,
                on_disk_bytes: u64::try_from(on_disk_bytes).unwrap_or(0),
                admitted: Instant::now(),
            });
        if paused {
            tracing::info!(
                waited_s = started.elapsed().as_secs(),
                "job admitted after memory-pressure pause",
            );
        }
        AdmissionGuard {
            admission: Arc::clone(self),
            id,
            est: self.est_bytes,
        }
    }

    fn release(&self, id: u64, est: usize) {
        self.reserved.fetch_sub(est, Ordering::AcqRel);
        #[allow(clippy::expect_used)]
        self.inflight
            .lock()
            .expect("admission registry mutex poisoned")
            .retain(|j| j.id != id);
        self.released.notify_waiters();
    }

    /// Log a per-slot breakdown of where in-flight memory is tied up: total
    /// reserved vs ceiling vs live usage, then one line per in-flight analysis
    /// (oldest first).
    pub fn log_inflight(&self, reason: &str) {
        #[allow(clippy::expect_used)]
        let mut jobs: Vec<_> = {
            let guard = self
                .inflight
                .lock()
                .expect("admission registry mutex poisoned");
            guard
                .iter()
                .map(|j| {
                    (
                        Arc::clone(&j.sha256),
                        Arc::clone(&j.path),
                        Arc::clone(&j.file_type),
                        j.on_disk_bytes,
                        j.admitted.elapsed(),
                    )
                })
                .collect()
        };
        jobs.sort_by_key(|j| std::cmp::Reverse(j.4));

        let reserved = self.reserved.load(Ordering::Acquire);
        let used = (self.used_fn)().unwrap_or(0);
        tracing::warn!(
            reason,
            inflight = jobs.len(),
            reserved_gb = reserved as f64 / GIB,
            ceiling_gb = self.ceiling_bytes as f64 / GIB,
            used_gb = used as f64 / GIB,
            "in-flight memory admission: per-slot breakdown follows",
        );
        for (slot, (sha256, path, file_type, on_disk, age)) in jobs.iter().enumerate() {
            tracing::warn!(
                slot,
                sha256 = %sha256,
                path = %path,
                file_type = %file_type,
                on_disk_mb = on_disk / MIB,
                age_s = age.as_secs(),
                "in-flight analysis",
            );
        }
    }
}

/// RAII reservation. Dropping it frees the budget and wakes one waiter.
#[derive(Debug)]
pub struct AdmissionGuard {
    admission: Arc<MemoryAdmission>,
    id: u64,
    est: usize,
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        self.admission.release(self.id, self.est);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const GB: usize = 1024 * 1024 * 1024;

    fn no_used() -> Option<u64> {
        None
    }

    /// 9 GB live usage — leaves < 1.5 GB below a 10 GB ceiling.
    fn used_9gb() -> Option<u64> {
        Some(9 * GB as u64)
    }

    /// 11 GB live usage — already past a 10 GB ceiling.
    fn used_11gb() -> Option<u64> {
        Some(11 * GB as u64)
    }

    fn gate(ceiling: usize, est: usize, used_fn: fn() -> Option<u64>) -> Arc<MemoryAdmission> {
        Arc::new(MemoryAdmission {
            ceiling_bytes: ceiling as u64,
            est_bytes: est,
            reserved: AtomicUsize::new(0),
            next_id: AtomicU64::new(0),
            inflight: Mutex::new(Vec::new()),
            released: Notify::new(),
            used_fn,
        })
    }

    #[tokio::test]
    async fn reservation_releases_on_drop() {
        let g = gate(10 * GB, 1536 * 1024 * 1024, no_used);
        let guard = g.admit("a".into(), "p".into(), "bin".into(), 40).await;
        assert_eq!(g.reserved.load(Ordering::Acquire), g.est_bytes);
        assert_eq!(g.inflight.lock().unwrap().len(), 1);
        drop(guard);
        assert_eq!(g.reserved.load(Ordering::Acquire), 0);
        assert!(g.inflight.lock().unwrap().is_empty());
    }

    #[test]
    fn disabled_ceiling_always_admits() {
        let g = gate(0, GB, used_11gb);
        assert!(g.try_reserve());
        assert!(g.try_reserve());
        assert_eq!(g.reserved.load(Ordering::Acquire), 2 * GB);
    }

    #[test]
    fn lone_job_admits_below_ceiling_but_not_when_already_over() {
        // Nothing in flight, host under the ceiling → the forward-progress hatch
        // admits even when one slot's estimate alone exceeds the ceiling.
        let g = gate(GB, 2 * GB, no_used);
        assert!(g.try_reserve());

        // Nothing in flight but the host is already past the ceiling (other
        // processes) → pause rather than pile on.
        let g = gate(10 * GB, GB, used_11gb);
        assert!(!g.try_reserve());
    }

    #[test]
    fn predictive_cap_pauses_a_burst_before_it_allocates() {
        // No live signal, so only committed reservations gate. With a 1.5 GB
        // estimate and a 10 GB ceiling, the 7th slot would commit 10.5 GB.
        let est = 1536 * 1024 * 1024;
        let g = gate(10 * GB, est, no_used);
        let mut admitted = 0;
        while g.try_reserve() {
            admitted += 1;
            assert!(admitted < 100, "predictive cap never engaged");
        }
        assert_eq!(admitted, 6);
        assert!((g.reserved.load(Ordering::Acquire) as u64) + est as u64 > g.ceiling_bytes);
    }

    #[test]
    fn reactive_gate_pauses_when_live_usage_leaves_no_room() {
        // One slot already in flight; live usage at 9 GB leaves < 1.5 GB under
        // the 10 GB ceiling, so the next slot must wait even though reservations
        // alone would fit.
        let est = 1536 * 1024 * 1024;
        let g = gate(10 * GB, est, used_9gb);
        g.reserved.store(est, Ordering::Release);
        assert!(!g.try_reserve());
    }
}
