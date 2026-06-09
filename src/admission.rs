//! Size-aware in-flight admission control for archive analyses.
//!
//! The worker runs up to `slots` analyses concurrently, but slot count says
//! nothing about *memory*: a slot analysing a multi-gigabyte archive holds the
//! whole archive plus its expanded members resident, while a slot analysing a
//! 4 KB script holds almost nothing. Bounding concurrency by slot count alone
//! lets a burst of large archives co-reside and exhaust RAM — the failure mode
//! that OOM-killed a 256 GB host (127 archives admitted at once → swap death;
//! every thread then stalled on page-faults, which surfaced as bogus
//! "rule took 30000ms" timings).
//!
//! This gate caps the *aggregate estimated footprint* of in-flight analyses to
//! a fraction (default 80%) of physical RAM. Each job reserves
//! `on_disk_bytes × expansion_factor`; a job is admitted only when its
//! reservation fits the remaining budget AND live RSS leaves room. A burst of
//! large archives therefore serialises instead of piling up, while small jobs
//! still fill every slot. One always-admit escape hatch (when nothing is in
//! flight) guarantees forward progress even for a single archive larger than
//! the whole budget.
//!
//! Admission is acquired from the single dispatch loop, so the check-then-
//! reserve sequence races nothing; releases happen on the per-job tokio tasks
//! and only ever *decrease* the reservation, so they cannot cause over-commit.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB: u64 = 1024 * 1024;

/// Fraction of physical RAM the in-flight estimate may reach when *live*
/// available memory can't be read (the only-thing-to-go-by fallback). When
/// available memory is readable it governs instead — see `try_reserve`.
/// Matches the worker's 85% RSS policy (`resolve_worker_max_rss_gb`) so the
/// admission budget and the reactive RSS gate agree on the ceiling.
const DEFAULT_BUDGET_FRACTION: f64 = 0.85;

/// Headroom left free when admitting a lone job against live available memory,
/// so a single large archive makes progress without allocating to the edge.
const LONE_HEADROOM_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// A lone job that never fits live available memory is force-admitted after this
/// long, so the pipeline can't wedge forever on one oversized file.
const FORCE_ADMIT_AFTER: Duration = Duration::from_secs(300);

/// Multiplier applied to a job's on-disk size to estimate its peak resident
/// footprint during analysis. Archives decompress and each source member spawns
/// a tree-sitter parse tree (often 10–50× the member), but only transiently and
/// one member at a time, so the *whole-archive* peak is far below the sum of
/// per-member worst cases. 8× is a deliberately conservative whole-archive
/// estimate; underestimates are caught by the live-RSS gate below.
const DEFAULT_EXPANSION_FACTOR: u64 = 8;

/// Every reservation counts at least this much, so thousands of tiny jobs still
/// register aggregate pressure rather than rounding to zero.
const MIN_ESTIMATE_BYTES: usize = 8 * 1024 * 1024;

/// How long a job may wait for budget before we log the in-flight breakdown.
const STALL_LOG_AFTER: Duration = Duration::from_secs(2);

/// Re-poll interval while waiting for budget (bounds RSS-feedback latency even
/// if no release wakes us).
const REPOLL_INTERVAL: Duration = Duration::from_secs(1);

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok()?.parse().ok()
}

fn env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok()?.parse().ok()
}

/// One in-flight analysis, recorded for the per-slot memory diagnostics.
#[derive(Debug)]
struct Inflight {
    id: u64,
    sha256: Arc<str>,
    path: Arc<str>,
    file_type: Arc<str>,
    on_disk_bytes: u64,
    est_bytes: usize,
    admitted: Instant,
}

/// Aggregate byte budget over all in-flight analyses.
#[derive(Debug)]
pub struct MemoryAdmission {
    /// Ceiling for the summed reservation, in bytes, used only when live
    /// available memory is unknown. `usize::MAX` disables the proactive gate
    /// (only the live-RSS backstop applies).
    budget_bytes: usize,
    /// Live available memory to keep free when admitting a non-lone job, so
    /// concurrent litmus processes leave headroom for each other. `total -
    /// budget`; 0 when total memory is unknown.
    free_floor_bytes: u64,
    expansion_factor: u64,
    reserved: AtomicUsize,
    next_id: AtomicU64,
    inflight: Mutex<Vec<Inflight>>,
    /// Woken whenever a reservation is released.
    released: Notify,
    /// Live available-memory source; a field so tests can inject a fixed value.
    available_fn: fn() -> Option<u64>,
}

/// Live available memory in bytes, or `None` where no live source is readable.
/// The default `available_fn`; tests inject a fixed value instead.
fn available_bytes() -> Option<u64> {
    cleave::memory_tracker::available_memory()
}

impl MemoryAdmission {
    /// Build the gate from physical RAM. Returns a disabled gate (no proactive
    /// throttling) when total memory is unknown, so an unsupported platform
    /// degrades to the previous behaviour rather than refusing all work.
    pub fn new() -> Arc<Self> {
        let fraction = env_f64("LITMUS_MEM_BUDGET_FRAC")
            .filter(|f| *f > 0.0 && *f <= 1.0)
            .unwrap_or(DEFAULT_BUDGET_FRACTION);
        let expansion_factor = env_u64("LITMUS_ARCHIVE_EXPANSION")
            .filter(|f| *f >= 1)
            .unwrap_or(DEFAULT_EXPANSION_FACTOR);

        let (budget_bytes, free_floor_bytes) = match cleave::memory_tracker::total_memory() {
            Some(total) if total > 0 => {
                // total × fraction is provably in [0, total] ⊆ [0, u64::MAX],
                // which fits usize on the 64-bit targets litmus runs on.
                #[allow(
                    clippy::cast_precision_loss,
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss
                )]
                let budget = (total as f64 * fraction) as usize;
                let free_floor = total.saturating_sub(budget as u64);
                tracing::info!(
                    total_gb = total as f64 / GIB,
                    budget_gb = budget as f64 / GIB,
                    free_floor_gb = free_floor as f64 / GIB,
                    fraction,
                    expansion_factor,
                    "in-flight memory admission enabled",
                );
                (budget, free_floor)
            }
            _ => {
                tracing::warn!(
                    "total physical memory unknown; in-flight memory admission \
                     falls back to live available memory only"
                );
                (usize::MAX, 0)
            }
        };

        Arc::new(Self {
            budget_bytes,
            free_floor_bytes,
            expansion_factor,
            reserved: AtomicUsize::new(0),
            next_id: AtomicU64::new(0),
            inflight: Mutex::new(Vec::new()),
            released: Notify::new(),
            available_fn: available_bytes,
        })
    }

    /// Estimate a job's peak resident footprint from its on-disk size.
    fn estimate(&self, on_disk_bytes: u64) -> usize {
        let raw = on_disk_bytes.saturating_mul(self.expansion_factor);
        let est = usize::try_from(raw).unwrap_or(usize::MAX);
        // Cap a single estimate at the budget: a giant archive still admits via
        // the `reserved == 0` escape hatch instead of being unsatisfiable.
        est.max(MIN_ESTIMATE_BYTES).min(self.budget_bytes)
    }

    /// Acquire admission, awaiting budget asynchronously (worker dispatch loop).
    /// The returned guard releases the reservation on drop. `on_disk_bytes` is
    /// the hopper-reported size; negative/zero is treated as the minimum estimate.
    pub async fn admit(
        self: &Arc<Self>,
        sha256: Arc<str>,
        path: Arc<str>,
        file_type: Arc<str>,
        on_disk_bytes: i64,
    ) -> AdmissionGuard {
        let est = self.estimate(u64::try_from(on_disk_bytes).unwrap_or(0));
        let started = Instant::now();
        let mut logged_stall = false;

        loop {
            if self.try_reserve(est, started.elapsed() >= FORCE_ADMIT_AFTER) {
                return self.register(
                    sha256,
                    path,
                    file_type,
                    on_disk_bytes,
                    est,
                    started,
                    logged_stall,
                );
            }
            if !logged_stall && started.elapsed() >= STALL_LOG_AFTER {
                self.log_inflight("dispatch stalled: in-flight memory budget reached");
                logged_stall = true;
            }
            // Wait for a release, but wake periodically to re-poll live available
            // memory even when no reservation is freed.
            tokio::select! {
                () = self.released.notified() => {}
                () = tokio::time::sleep(REPOLL_INTERVAL) => {}
            }
        }
    }

    /// Synchronous twin of [`Self::admit`] for the CLI scan path, which analyses
    /// files on a rayon pool with no async runtime. Blocks the calling rayon
    /// worker until the job fits, polling for releases by sibling workers. The
    /// `reserved == 0` escape hatch guarantees at least one worker always
    /// proceeds, so the pool can't deadlock.
    pub fn admit_blocking(
        self: &Arc<Self>,
        sha256: Arc<str>,
        path: Arc<str>,
        file_type: Arc<str>,
        on_disk_bytes: i64,
    ) -> AdmissionGuard {
        let est = self.estimate(u64::try_from(on_disk_bytes).unwrap_or(0));
        let started = Instant::now();
        let mut logged_stall = false;

        loop {
            if self.try_reserve(est, started.elapsed() >= FORCE_ADMIT_AFTER) {
                return self.register(
                    sha256,
                    path,
                    file_type,
                    on_disk_bytes,
                    est,
                    started,
                    logged_stall,
                );
            }
            if !logged_stall && started.elapsed() >= STALL_LOG_AFTER {
                self.log_inflight("scan stalled: in-flight memory budget reached");
                logged_stall = true;
            }
            std::thread::sleep(REPOLL_INTERVAL);
        }
    }

    /// Record an admitted job (its `est` is already reserved) and return its guard.
    /// Private two-call-site helper; the arguments mirror `Inflight`'s fields
    /// plus the wait bookkeeping, so a parameter struct would just restate it.
    #[allow(clippy::too_many_arguments)]
    fn register(
        self: &Arc<Self>,
        sha256: Arc<str>,
        path: Arc<str>,
        file_type: Arc<str>,
        on_disk_bytes: i64,
        est: usize,
        started: Instant,
        logged_stall: bool,
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
                est_bytes: est,
                admitted: Instant::now(),
            });
        if logged_stall {
            tracing::info!(
                waited_s = started.elapsed().as_secs(),
                est_mb = est as u64 / MIB,
                "job admitted after memory-budget wait",
            );
        }
        AdmissionGuard {
            admission: Arc::clone(self),
            id,
            est,
        }
    }

    /// Reserve `est` if it fits. Gated on *live available memory* when readable
    /// (so concurrent litmus processes leave room for each other), falling back
    /// to the aggregate `budget_bytes` ceiling plus the live-RSS backstop when it
    /// isn't. `force_lone` admits a lone job unconditionally — the bounded escape
    /// hatch for a single file that never fits live memory.
    ///
    /// Single caller per loop iteration; releases only decrease `reserved`.
    fn try_reserve(&self, est: usize, force_lone: bool) -> bool {
        let reserved = self.reserved.load(Ordering::Acquire);
        let available = (self.available_fn)();

        // Lone job: admit if it fits live available memory with headroom, so a
        // single large archive makes progress without co-residing with anything.
        // Unknown available memory, or a long wait (`force_lone`), admits
        // unconditionally — the bounded escape hatch that guarantees progress.
        if reserved == 0 {
            let fits = match available {
                Some(avail) => (est as u64).saturating_add(LONE_HEADROOM_BYTES) <= avail,
                None => true,
            };
            if fits || force_lone {
                self.reserved.store(est, Ordering::Release);
                return true;
            }
            return false;
        }

        // Aggregate reservation ceiling: caps in-flight commitments before their
        // allocations land (usize::MAX when total memory is unknown → disabled).
        if reserved.saturating_add(est) > self.budget_bytes {
            return false;
        }

        match available {
            // Cross-process live gate: admitting must leave free_floor available,
            // so a burst of large archives serialises across all litmus processes
            // instead of exhausting RAM. `available` reflects the whole host.
            Some(avail) => {
                if (est as u64).saturating_add(self.free_floor_bytes) > avail {
                    return false;
                }
            }
            // No live signal: own-RSS backstop catches reservation underestimates.
            None => {
                if let Some(rss) = cleave::memory_tracker::current_rss() {
                    let rss = usize::try_from(rss).unwrap_or(usize::MAX);
                    if rss.saturating_add(est) > self.budget_bytes {
                        return false;
                    }
                }
            }
        }

        self.reserved.store(reserved + est, Ordering::Release);
        true
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
    /// reserved vs budget vs live RSS, then one line per in-flight analysis
    /// (largest reservation first).
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
                        j.est_bytes,
                        j.admitted.elapsed(),
                    )
                })
                .collect()
        };
        jobs.sort_by_key(|j| std::cmp::Reverse(j.4));

        let reserved = self.reserved.load(Ordering::Acquire);
        let rss = cleave::memory_tracker::current_rss().unwrap_or(0);
        tracing::warn!(
            reason,
            inflight = jobs.len(),
            reserved_gb = reserved as f64 / GIB,
            budget_gb = self.budget_bytes as f64 / GIB,
            rss_gb = rss as f64 / GIB,
            "in-flight memory admission: per-slot breakdown follows",
        );
        for (slot, (sha256, path, file_type, on_disk, est, age)) in jobs.iter().enumerate() {
            tracing::warn!(
                slot,
                sha256 = %sha256,
                path = %path,
                file_type = %file_type,
                on_disk_mb = on_disk / MIB,
                reserved_mb = *est as u64 / MIB,
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
// `MB` is 1 MiB in an i64 (matching `admit`'s `on_disk_bytes`); casting it to
// usize/u64 in budget fixtures can never truncate or change sign.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod tests {
    use super::*;

    fn no_available() -> Option<u64> {
        None
    }

    fn gate(budget: usize, factor: u64) -> Arc<MemoryAdmission> {
        // Default gate: available memory unknown → falls back to the budget
        // ceiling, matching the pre-live-gate behaviour these tests assert.
        gate_avail(budget, factor, 0, no_available)
    }

    fn gate_avail(
        budget: usize,
        factor: u64,
        free_floor: u64,
        available_fn: fn() -> Option<u64>,
    ) -> Arc<MemoryAdmission> {
        Arc::new(MemoryAdmission {
            budget_bytes: budget,
            free_floor_bytes: free_floor,
            expansion_factor: factor,
            reserved: AtomicUsize::new(0),
            next_id: AtomicU64::new(0),
            inflight: Mutex::new(Vec::new()),
            released: Notify::new(),
            available_fn,
        })
    }

    #[test]
    fn estimate_applies_factor_floor_and_cap() {
        let g = gate(1_000_000_000, 8);
        // Floor: tiny file rounds up to MIN_ESTIMATE_BYTES.
        assert_eq!(g.estimate(1), MIN_ESTIMATE_BYTES);
        // Factor: 100 MB on disk × 8.
        assert_eq!(g.estimate(100 * 1024 * 1024), 800 * 1024 * 1024);
        // Cap: never exceeds the budget.
        assert_eq!(g.estimate(u64::MAX), 1_000_000_000);
    }

    // Sizes must clear the MIN_ESTIMATE_BYTES floor to exercise the budget math
    // rather than the floor; work in megabytes.
    const MB: i64 = 1024 * 1024;

    #[tokio::test]
    async fn reservation_releases_on_drop() {
        let g = gate(100 * MB as usize, 1);
        let guard = g.admit("a".into(), "p".into(), "bin".into(), 40 * MB).await;
        assert_eq!(g.reserved.load(Ordering::Acquire), 40 * MB as usize);
        assert_eq!(g.inflight.lock().unwrap().len(), 1);
        drop(guard);
        assert_eq!(g.reserved.load(Ordering::Acquire), 0);
        assert!(g.inflight.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn over_budget_job_waits_then_admits_after_release() {
        // budget 100 MB, factor 1: first 80 MB job fits, second must wait.
        let g = gate(100 * MB as usize, 1);
        let first = g.admit("a".into(), "p".into(), "bin".into(), 80 * MB).await;
        assert_eq!(g.reserved.load(Ordering::Acquire), 80 * MB as usize);

        let g2 = Arc::clone(&g);
        let waiter = tokio::spawn(async move {
            // 40 MB would push reserved to 120 MB > 100 MB; blocks until `first`
            // frees and the escape hatch (reserved == 0) re-opens admission.
            let _guard = g2
                .admit("b".into(), "q".into(), "bin".into(), 40 * MB)
                .await;
            g2.reserved.load(Ordering::Acquire)
        });

        // Give the waiter a moment to block, then release the first job.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !waiter.is_finished(),
            "second job should be waiting on budget"
        );
        drop(first);

        let reserved_after = waiter.await.unwrap();
        assert_eq!(reserved_after, 40 * MB as usize);
    }

    #[tokio::test]
    async fn single_oversized_job_admits_via_escape_hatch() {
        // Job estimate exceeds the whole budget, but nothing is in flight, so
        // it must still admit (forward progress).
        let g = gate(100 * MB as usize, 1);
        let _guard = g
            .admit("a".into(), "p".into(), "bin".into(), 10_000 * MB)
            .await;
        // Estimate capped at budget.
        assert_eq!(g.reserved.load(Ordering::Acquire), 100 * MB as usize);
    }

    // Signature is fixed by `MemoryAdmission.available_fn: fn() -> Option<u64>`.
    #[allow(clippy::unnecessary_wraps)]
    fn avail_200mb() -> Option<u64> {
        Some(200 * MB as u64)
    }

    #[test]
    fn live_available_gate_serialises_large_jobs() {
        // Budget is not the limiter; the live gate (free_floor 100 MB, available
        // 200 MB) is. With 50 MB already in flight:
        let g = gate_avail(10_000 * MB as usize, 1, 100 * MB as u64, avail_200mb);
        g.reserved.store(50 * MB as usize, Ordering::Release);
        // 150 MB job: 150 + 100 floor = 250 MB > 200 available → denied.
        assert!(!g.try_reserve(150 * MB as usize, false));
        // 50 MB job: 50 + 100 = 150 MB ≤ 200 available → admitted.
        assert!(g.try_reserve(50 * MB as usize, false));
    }

    #[test]
    fn lone_job_waits_until_forced_when_it_exceeds_available() {
        // A lone job needs LONE_HEADROOM_BYTES (2 GiB) below available, so a
        // 100 MB job against 200 MB available does not fit by that rule.
        let g = gate_avail(10_000 * MB as usize, 1, 0, avail_200mb);
        assert!(!g.try_reserve(100 * MB as usize, false));
        // Forced (the bounded escape hatch after a long wait) admits anyway.
        assert!(g.try_reserve(100 * MB as usize, true));
        assert_eq!(g.reserved.load(Ordering::Acquire), 100 * MB as usize);
    }
}
