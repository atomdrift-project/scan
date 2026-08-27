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
//! * **Predictive** — each in-flight analysis reserves an estimated footprint.
//!   Archive-shaped jobs reserve more than flat files because they expand into
//!   member reports and parse trees. A burst commits its full reservation the
//!   instant it is admitted, so the gate closes *before* archives expand, not
//!   after.
//! * **Reactive** — live memory usage (`total − MemAvailable` on Linux, this
//!   process's RSS elsewhere) must leave room for the next reservation. This
//!   catches estimates that ran low and pressure from other processes on the
//!   host.
//!
//! One always-admit hatch (when nothing is in flight) guarantees forward
//! progress even on a host too small to fit a single slot's estimate.
//!
//! When the gate closes it logs the per-slot breakdown and clears caches once,
//! then re-polls until memory frees. Admission is acquired concurrently from
//! every worker task (and every paused waiter retries on the repoll tick), so
//! the check-then-reserve sequence runs as a CAS loop on `reserved`; releases
//! happen on the per-job tokio tasks and saturate at zero, so no interleaving
//! can wrap the reservation below zero and wedge the gate shut.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB: u64 = 1024 * 1024;

/// Assumed peak resident footprint of a small, non-archive analysis.
const DEFAULT_FLAT_ESTIMATE_BYTES: u64 = 512 * 1024 * 1024;

/// Minimum reservation for archive-shaped jobs. Archives decompress and each
/// source member can spawn a tree-sitter parse tree, so their true peak is often
/// far above on-disk size.
const MIN_ARCHIVE_ESTIMATE_BYTES: u64 = 1536 * 1024 * 1024;

/// Upper bound for the size-scaled archive estimate. Jobs larger than the
/// ceiling still run via the one-job forward-progress hatch.
const MAX_ARCHIVE_ESTIMATE_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// On-disk archive bytes are a weak lower bound for in-memory member state; this
/// multiplier is intentionally pessimistic for large bundles while still letting
/// small package archives co-reside.
const ARCHIVE_ESTIMATE_MULTIPLIER: u64 = 64;

/// Re-poll interval while waiting for memory to free (bounds feedback latency
/// even if no release wakes us).
const REPOLL_INTERVAL: Duration = Duration::from_secs(1);

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

fn dynamic_estimate_bytes(path: &str, file_type: &str, on_disk_bytes: i64) -> u64 {
    let bytes = u64::try_from(on_disk_bytes).unwrap_or(0);
    if looks_like_archive(path, file_type) {
        let scaled = bytes
            .saturating_mul(ARCHIVE_ESTIMATE_MULTIPLIER)
            .saturating_add(MIN_ARCHIVE_ESTIMATE_BYTES)
            .min(MAX_ARCHIVE_ESTIMATE_BYTES);
        scaled.max(MIN_ARCHIVE_ESTIMATE_BYTES)
    } else {
        DEFAULT_FLAT_ESTIMATE_BYTES
    }
}

fn looks_like_archive(path: &str, file_type: &str) -> bool {
    let ft = file_type.trim().to_ascii_lowercase();
    matches!(
        ft.as_str(),
        "7z" | "apk"
            | "archive"
            | "bz2"
            | "conda"
            | "crate"
            | "crx"
            | "deb"
            | "dmg"
            | "ear"
            | "gem"
            | "gz"
            | "jar"
            | "nupkg"
            | "pkg"
            | "rar"
            | "rpm"
            | "tar"
            | "tar.bz2"
            | "tar.gz"
            | "tar.xz"
            | "tbz2"
            | "tgz"
            | "txz"
            | "vsix"
            | "war"
            | "whl"
            | "xpi"
            | "xz"
            | "zip"
            | "zst"
    ) || {
        let p = path.to_ascii_lowercase();
        ARCHIVE_SUFFIXES.iter().any(|suffix| p.ends_with(suffix))
    }
}

const ARCHIVE_SUFFIXES: &[&str] = &[
    ".7z", ".apk", ".bz2", ".conda", ".crate", ".crx", ".deb", ".dmg", ".ear", ".gem", ".gz",
    ".jar", ".nupkg", ".pkg", ".rar", ".rpm", ".tar", ".tar.bz2", ".tar.gz", ".tar.xz", ".tbz2",
    ".tgz", ".txz", ".vsix", ".war", ".whl", ".xpi", ".xz", ".zip", ".zst",
];

/// One in-flight analysis, recorded for the per-slot memory diagnostics.
#[derive(Debug)]
struct Inflight {
    id: u64,
    sha256: Arc<str>,
    path: Arc<str>,
    file_type: Arc<str>,
    on_disk_bytes: u64,
    est_bytes: u64,
    admitted: Instant,
}

/// Live memory-pressure gate over all in-flight analyses.
#[derive(Debug)]
pub struct MemoryAdmission {
    /// Memory ceiling in bytes. Admission pauses once committed reservations or
    /// live usage would push past it. `0` disables the gate (slot-limited only),
    /// matching a disabled `--max-rss-gb`.
    ceiling_bytes: u64,
    /// Optional fixed reservation override. When absent, reservations are
    /// estimated from the job path/type/size.
    fixed_est_bytes: Option<u64>,
    /// Sum of in-flight reservations.
    reserved: AtomicU64,
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
        let fixed_est_bytes = std::env::var("SCAN_PER_SLOT_ESTIMATE_MB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .and_then(|mb| mb.checked_mul(1024 * 1024))
            .filter(|b| *b > 0);

        if ceiling_bytes == 0 {
            tracing::warn!("memory admission disabled (max-rss 0); dispatch is slot-limited only");
        } else {
            match fixed_est_bytes {
                Some(est_bytes) => tracing::info!(
                    ceiling_gb = ceiling_bytes as f64 / GIB,
                    per_slot_estimate_gb = est_bytes as f64 / GIB,
                    "live memory admission enabled with fixed per-slot estimate",
                ),
                None => tracing::info!(
                    ceiling_gb = ceiling_bytes as f64 / GIB,
                    flat_estimate_gb = DEFAULT_FLAT_ESTIMATE_BYTES as f64 / GIB,
                    min_archive_estimate_gb = MIN_ARCHIVE_ESTIMATE_BYTES as f64 / GIB,
                    max_archive_estimate_gb = MAX_ARCHIVE_ESTIMATE_BYTES as f64 / GIB,
                    archive_multiplier = ARCHIVE_ESTIMATE_MULTIPLIER,
                    "live memory admission enabled with dynamic estimates",
                ),
            }
        }

        Arc::new(Self {
            ceiling_bytes,
            fixed_est_bytes,
            reserved: AtomicU64::new(0),
            next_id: AtomicU64::new(0),
            inflight: Mutex::new(Vec::new()),
            released: Notify::new(),
            used_fn: live_used,
        })
    }

    /// Acquire admission, awaiting free memory asynchronously (worker dispatch
    /// loop). The returned guard releases the reservation on drop. `on_disk_bytes`
    /// Current sum of in-flight memory reservations, in bytes. Surfaced on the
    /// worker heartbeat so hopper can see how close to the ceiling a worker is
    /// running (and thus whether memory admission is about to pause intake).
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved.load(Ordering::Acquire)
    }

    /// Configured memory ceiling in bytes — the resolved `--max-rss-gb` that
    /// throttles intake. `0` means the gate is disabled (slot-limited only).
    pub fn ceiling_bytes(&self) -> u64 {
        self.ceiling_bytes
    }

    /// The sha256 of every analysis currently in flight. Reported on the worker
    /// heartbeat so hopper can renew these claims' leases: a multi-hour scan
    /// must not have its claim expire and be re-issued to another worker.
    pub fn in_flight_shas(&self) -> Vec<Arc<str>> {
        self.inflight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|j| Arc::clone(&j.sha256))
            .collect()
    }

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
        let est = self
            .fixed_est_bytes
            .unwrap_or_else(|| dynamic_estimate_bytes(&path, &file_type, on_disk_bytes));

        loop {
            if self.try_reserve(est) {
                let guard = self.register(sha256, path, file_type, on_disk_bytes, est);
                if paused {
                    tracing::info!(
                        waited_s = started.elapsed().as_secs(),
                        "job admitted after memory-pressure pause",
                    );
                }
                return guard;
            }
            if !paused {
                // First time the gate closes for this job: surface where memory
                // is tied up, then reclaim what the caches are holding. Both run
                // once per pause episode — the dispatch loop admits serially, so
                // this cannot thrash.
                self.log_inflight("admission paused: memory at saturation");
                paused = true;
                // Clearing the caches only moves memory back to the allocator.
                // This gate re-polls *live process memory*, so unless those pages
                // also go back to the OS the pause sees no improvement and waits
                // out the full re-poll for nothing.
                let reclaim = tokio::task::spawn_blocking(|| {
                    cleave::clear_all_thread_caches();
                    crate::allocator::trim();
                });
                if let Err(e) = reclaim.await {
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

    /// Reserve one job if memory allows. Gated on committed reservations and on
    /// live usage, both against `ceiling_bytes`; an always-admit hatch (nothing
    /// in flight) guarantees forward progress on a host too small for one job or
    /// whose allocator has retained RSS after earlier jobs.
    ///
    /// Called concurrently from every worker task, and races per-job releases,
    /// so the check-then-reserve must be a CAS: a plain load+store here loses
    /// concurrent updates, and one lost add is enough to later wrap `reserved`
    /// below zero — which reads as ~2^64 and closes the gate permanently.
    fn try_reserve(&self, est: u64) -> bool {
        // Disabled: dispatch is bounded by slot count alone.
        if self.ceiling_bytes == 0 {
            self.reserved.fetch_add(est, Ordering::AcqRel);
            return true;
        }

        let used = (self.used_fn)();
        let mut reserved = self.reserved.load(Ordering::Acquire);
        loop {
            // Forward-progress hatch: with nothing of ours in flight, admit one
            // job even when the job estimate exceeds the ceiling, or when RSS
            // stayed high because the allocator retained freed arenas. The CAS
            // below admits exactly one job through the hatch even when several
            // waiters observe zero at once.
            if reserved != 0 {
                // Predictive: committed reservations must leave room for this job.
                if reserved.saturating_add(est) > self.ceiling_bytes {
                    return false;
                }

                // Reactive: live usage must leave room for this job, catching
                // estimates that ran low and pressure from other processes.
                if let Some(used) = used
                    && used.saturating_add(est) > self.ceiling_bytes
                {
                    return false;
                }
            }

            match self.reserved.compare_exchange_weak(
                reserved,
                reserved.saturating_add(est),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => reserved = actual,
            }
        }
    }

    /// Record an admitted job (its `est` is already reserved) and return its guard.
    fn register(
        self: &Arc<Self>,
        sha256: Arc<str>,
        path: Arc<str>,
        file_type: Arc<str>,
        on_disk_bytes: i64,
        est_bytes: u64,
    ) -> AdmissionGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.inflight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Inflight {
                id,
                sha256,
                path,
                file_type,
                on_disk_bytes: u64::try_from(on_disk_bytes).unwrap_or(0),
                est_bytes,
                admitted: Instant::now(),
            });
        AdmissionGuard {
            admission: Arc::clone(self),
            id,
            est: est_bytes,
        }
    }

    fn release(&self, id: u64, est: u64) {
        // Saturating: an accounting bug must degrade to a reservation leaked
        // toward zero, never a wrap below zero — a wrapped counter reads as
        // ~2^64 and pauses admission forever (seen live on smaug 2026-08-22).
        let _ = self
            .reserved
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |r| {
                Some(r.saturating_sub(est))
            });
        self.inflight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|j| j.id != id);
        self.released.notify_waiters();
    }

    /// Log a per-slot breakdown of where in-flight memory is tied up: total
    /// reserved vs ceiling vs live usage, then one line per in-flight analysis
    /// (oldest first).
    pub fn log_inflight(&self, reason: &str) {
        let mut jobs: Vec<_> = {
            let guard = self
                .inflight
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        let used = (self.used_fn)().unwrap_or(0);
        tracing::warn!(
            reason,
            inflight = jobs.len(),
            reserved_gb = reserved as f64 / GIB,
            ceiling_gb = self.ceiling_bytes as f64 / GIB,
            used_gb = used as f64 / GIB,
            "in-flight memory admission: per-slot breakdown follows",
        );
        for (slot, (sha256, path, file_type, on_disk, est, age)) in jobs.iter().enumerate() {
            tracing::warn!(
                slot,
                sha256 = %sha256,
                path = %path,
                file_type = %file_type,
                on_disk_mb = on_disk / MIB,
                est_gb = *est as f64 / GIB,
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
    est: u64,
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        self.admission.release(self.id, self.est);
    }
}

#[cfg(test)]
// `used_9gb`/`used_11gb` must keep the `Option<u64>` return to match the
// `fn() -> Option<u64>` pointer `gate` takes (shared with `no_used` → None).
#[allow(clippy::unwrap_used, clippy::unnecessary_wraps)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    fn no_used() -> Option<u64> {
        None
    }

    // Signatures are fixed by `MemoryAdmission.used_fn: fn() -> Option<u64>`.
    /// 9 GB live usage — leaves < 1.5 GB below a 10 GB ceiling.
    #[allow(clippy::unnecessary_wraps)]
    fn used_9gb() -> Option<u64> {
        Some(9 * GB)
    }

    /// 11 GB live usage — already past a 10 GB ceiling.
    #[allow(clippy::unnecessary_wraps)]
    fn used_11gb() -> Option<u64> {
        Some(11 * GB)
    }

    fn gate(ceiling: u64, est: u64, used_fn: fn() -> Option<u64>) -> Arc<MemoryAdmission> {
        Arc::new(MemoryAdmission {
            ceiling_bytes: ceiling,
            fixed_est_bytes: Some(est),
            reserved: AtomicU64::new(0),
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
        assert_eq!(
            g.reserved.load(Ordering::Acquire),
            g.fixed_est_bytes.unwrap()
        );
        assert_eq!(g.inflight.lock().unwrap().len(), 1);
        drop(guard);
        assert_eq!(g.reserved.load(Ordering::Acquire), 0);
        assert!(g.inflight.lock().unwrap().is_empty());
    }

    #[test]
    fn disabled_ceiling_always_admits() {
        let g = gate(0, GB, used_11gb);
        assert!(g.try_reserve(GB));
        assert!(g.try_reserve(GB));
        assert_eq!(g.reserved.load(Ordering::Acquire), 2 * GB);
    }

    #[test]
    fn lone_job_always_admits_for_forward_progress() {
        // Nothing in flight, host under the ceiling → the forward-progress hatch
        // admits even when one slot's estimate alone exceeds the ceiling.
        let g = gate(GB, 2 * GB, no_used);
        assert!(g.try_reserve(2 * GB));

        // Nothing in flight but RSS is already past the ceiling, commonly from
        // allocator-retained arenas after earlier archive jobs. Still admit one
        // job so a low ceiling cannot deadlock the worker.
        let g = gate(10 * GB, GB, used_11gb);
        assert!(g.try_reserve(GB));
    }

    #[test]
    fn predictive_cap_pauses_a_burst_before_it_allocates() {
        // No live signal, so only committed reservations gate. With a 1.5 GB
        // estimate and a 10 GB ceiling, the 7th slot would commit 10.5 GB.
        let est = 1536 * 1024 * 1024;
        let g = gate(10 * GB, est, no_used);
        let mut admitted = 0;
        while g.try_reserve(est) {
            admitted += 1;
            assert!(admitted < 100, "predictive cap never engaged");
        }
        assert_eq!(admitted, 6);
        assert!(g.reserved.load(Ordering::Acquire) + est > g.ceiling_bytes);
    }

    #[test]
    fn reactive_gate_pauses_when_live_usage_leaves_no_room() {
        // One slot already in flight; live usage at 9 GB leaves < 1.5 GB under
        // the 10 GB ceiling, so the next slot must wait even though reservations
        // alone would fit.
        let est = 1536 * 1024 * 1024;
        let g = gate(10 * GB, est, used_9gb);
        g.reserved.store(est, Ordering::Release);
        assert!(!g.try_reserve(est));
    }

    #[test]
    fn dynamic_estimate_scales_archives_by_size() {
        let small_zip = dynamic_estimate_bytes("pkg.zip", "data", 10 * 1024 * 1024);
        let big_zip = dynamic_estimate_bytes("pkg.zip", "data", 100 * 1024 * 1024);
        assert!(small_zip >= MIN_ARCHIVE_ESTIMATE_BYTES);
        assert!(big_zip > small_zip);
        assert!(big_zip <= MAX_ARCHIVE_ESTIMATE_BYTES);
        assert_eq!(
            dynamic_estimate_bytes("plain.js", "javascript", 10 * 1024 * 1024),
            DEFAULT_FLAT_ESTIMATE_BYTES,
        );
    }

    #[test]
    fn concurrent_reserve_release_never_wraps_reserved() {
        // Regression: try_reserve used a load+store, so a release landing
        // between them was overwritten; the corresponding guards still
        // subtracted their full estimate and `reserved` wrapped below zero,
        // closing the gate permanently. With the CAS every add is recorded,
        // so after all guards release the counter must be exactly zero.
        let est = GB;
        let g = gate(4 * GB, est, no_used);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..1000 {
                        if g.try_reserve(est) {
                            g.release(u64::MAX, est);
                        }
                    }
                });
            }
        });
        let reserved = g.reserved.load(Ordering::Acquire);
        assert_eq!(reserved, 0, "reserved leaked or wrapped: {reserved:#x}");
    }

    #[test]
    fn reservations_can_exceed_the_32_bit_address_range() {
        let est = 5 * GB;
        let g = gate(10 * GB, est, no_used);

        assert!(est > u64::from(u32::MAX));
        assert!(g.try_reserve(est));
        assert!(g.try_reserve(est));
        assert!(!g.try_reserve(1));
        assert_eq!(g.reserved.load(Ordering::Acquire), 10 * GB);
    }
}
