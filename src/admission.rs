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
//! * **Reactive** — this process's live RSS must leave room for the next
//!   reservation. This catches estimates that ran low without conflating the
//!   worker's `--max-rss-gb` ceiling with unrelated host-wide memory use.
//! * **Host** — the kernel's live *available* memory must leave a floor after
//!   the next reservation. The ceiling is a policy on this process; the host
//!   can still be out of memory because of other tenants, page cache that
//!   will not be reclaimed in time, or estimates that ran low everywhere at
//!   once. When the kernel says there is no room, there is no room.
//!
//! Archive detection uses the job's path suffix and hopper's `file_type`, and
//! — when the payload is at hand — its leading bytes: hopper hands out
//! suffix-less names (`v1`, a bare sha, `tool@v3.2.0`) for tarballs and zips
//! that would otherwise be reserved as 512 MiB flat files and expand into
//! several GB of member analysis.
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
///
/// 256 MiB, down from 512. Measured on the production worker 2026-09-03 with
/// 48 slots: reservations summed to 25.7-26.0 GB against the 26 GB ceiling
/// and paused admission ~200 times per 10-minute run, while the process's
/// real working set peaked at 15.3 GB. Most of that gap was this floor: a
/// 4 KB manifest or a 100 KB script reserved half a gigabyte. The RSS check
/// in `try_reserve` remains the reactive backstop for a real overrun.
const DEFAULT_FLAT_ESTIMATE_BYTES: u64 = 256 * 1024 * 1024;

/// Minimum reservation for archive-shaped jobs. Archives decompress and each
/// source member can spawn a tree-sitter parse tree, so their true peak is often
/// far above on-disk size.
///
/// 512 MiB, down from 1.5 GiB, and the multiplier below 16x, down from 64x
/// (2026-09-03). The old figures summed to the 26 GB ceiling at ~20 archives
/// in flight while the process peaked at 14 GB, so the estimate — not memory
/// — was the throughput ceiling. At these figures, with every pool thread
/// admitted, the worker measured a 17.5 GB peak on a 32 GB box with no
/// memory-pressure warnings and the pool saturated. The RSS ceiling and the
/// host floor stay as the reactive backstops for the archive this does not
/// fit.
const MIN_ARCHIVE_ESTIMATE_BYTES: u64 = 512 * 1024 * 1024;

/// Upper bound for the size-scaled archive estimate. Jobs larger than the
/// ceiling still run via the one-job forward-progress hatch.
const MAX_ARCHIVE_ESTIMATE_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// On-disk archive bytes are a weak lower bound for in-memory member state; this
/// multiplier is intentionally pessimistic for large bundles while still letting
/// small package archives co-reside.
const ARCHIVE_ESTIMATE_MULTIPLIER: u64 = 16;

/// Re-poll interval while waiting for memory to free (bounds feedback latency
/// even if no release wakes us).
const REPOLL_INTERVAL: Duration = Duration::from_secs(1);

/// Host available memory the gate refuses to eat into: 1/16 of RAM (6.25%),
/// at least 1 GiB — 2 GiB on a 32 GiB host, 8 GiB on 128 GiB.
///
/// The floor guards against the kernel's reclaim/paging storm, not against
/// allocation failure (Windows fails allocations on the commit limit, which
/// the pagefile backs; Linux overcommits). Windows starts trimming working
/// sets and paging hard below roughly 1 GiB available, and `ullAvailPhys` /
/// `MemAvailable` both count reclaimable cache as available, so 1 GiB of
/// headroom above that threshold is the real margin. An earlier 10% floor
/// left ~3 GiB idle on a 32 GiB desktop where the `--max-rss-gb` ceiling
/// (85%) already bounds this process. `SCAN_HOST_FLOOR_MB` overrides
/// (0 disables the host check).
fn host_floor_bytes() -> u64 {
    if let Some(mb) = std::env::var("SCAN_HOST_FLOOR_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        return mb.saturating_mul(MIB);
    }
    let total = cleave::memory_tracker::total_memory().unwrap_or(16 * 1024 * MIB);
    (total / 16).max(1024 * MIB)
}

/// Live available host memory in bytes, or `None` where the platform has no
/// live source. The default `available_fn`; tests inject a value.
fn live_available() -> Option<u64> {
    cleave::memory_tracker::available_memory()
}

/// Bytes of a payload worth sniffing for an archive signature. The tar magic
/// sits at offset 257, everything else in the first 8 bytes.
pub const SNIFF_BYTES: usize = 512;

/// Archive signature check on a payload's leading bytes. Only formats whose
/// analysis expands into a member walk matter here; a wrong `false` costs a
/// flat estimate (today's behavior), a wrong `true` costs one pessimistic
/// reservation.
#[must_use]
pub fn looks_like_archive_bytes(head: &[u8]) -> bool {
    const SIGS: &[&[u8]] = &[
        b"\x1f\x8b",           // gzip
        b"PK\x03\x04",         // zip (and jar/whl/nupkg/vsix wrappers)
        b"\xfd7zXZ\x00",       // xz
        b"BZh",                // bzip2
        b"\x28\xb5\x2f\xfd",   // zstd
        b"7z\xbc\xaf\x27\x1c", // 7z
        b"Rar!\x1a\x07",       // rar
        b"!<arch>\n",          // ar (deb)
        b"\xed\xab\xee\xdb",   // rpm
        b"Cr24",               // crx
        b"xar!",               // xar (pkg)
    ];
    if SIGS.iter().any(|sig| head.starts_with(sig)) {
        return true;
    }
    // ustar / GNU tar: "ustar" at offset 257.
    head.get(257..262) == Some(b"ustar".as_slice())
}

/// This process's live resident memory in bytes toward the `--max-rss-gb`
/// ceiling, or `None` when the platform cannot read it. The old implementation
/// preferred `total - available` whenever both happened to be implemented. On
/// FreeBSD that measured the whole host (108 GiB in the affected run) while the
/// worker itself used 1.5 GiB, causing a 64 GiB process ceiling to admit only
/// one job. The default `used_fn`; tests inject a value.
fn live_used() -> Option<u64> {
    cleave::memory_tracker::current_rss()
}

fn dynamic_estimate_bytes(
    path: &str,
    file_type: &str,
    on_disk_bytes: i64,
    head: Option<&[u8]>,
) -> u64 {
    let bytes = u64::try_from(on_disk_bytes).unwrap_or(0);
    if looks_like_archive(path, file_type) || head.is_some_and(looks_like_archive_bytes) {
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
    /// Live host-available-memory source; a field so tests can inject a value.
    available_fn: fn() -> Option<u64>,
    /// Host available memory the gate will not reserve into (0 = no host check).
    host_floor_bytes: u64,
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

        let host_floor_bytes = host_floor_bytes();
        if ceiling_bytes != 0 {
            match live_available() {
                Some(available) => tracing::info!(
                    host_floor_gb = host_floor_bytes as f64 / GIB,
                    host_available_gb = available as f64 / GIB,
                    "host available-memory check enabled",
                ),
                None => tracing::info!(
                    "host available-memory check unavailable on this platform; \
                     admission keys on process RSS and reservations only",
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
            available_fn: live_available,
            host_floor_bytes,
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
    /// `head` is the payload's leading bytes when the caller has them (see
    /// [`SNIFF_BYTES`]); they refine the archive estimate for suffix-less names.
    pub async fn admit(
        self: &Arc<Self>,
        sha256: Arc<str>,
        path: Arc<str>,
        file_type: Arc<str>,
        on_disk_bytes: i64,
        head: Option<&[u8]>,
    ) -> AdmissionGuard {
        let started = Instant::now();
        let mut paused = false;
        let est = self
            .fixed_est_bytes
            .unwrap_or_else(|| dynamic_estimate_bytes(&path, &file_type, on_disk_bytes, head));

        loop {
            let refusal = self.try_reserve(est).err();
            if refusal.is_none() {
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
                if let Some(Refusal::Host { available }) = refusal {
                    // The host itself is out of room — not our ceiling, not our
                    // reservations. This is the condition that precedes an OOM
                    // kill, so it is an error, not a routine pause.
                    tracing::error!(
                        sha256 = %sha256,
                        path = %path,
                        host_available_mb = available / MIB,
                        host_floor_mb = self.host_floor_bytes / MIB,
                        job_est_mb = est / MIB,
                        "host memory below floor: refusing to start analysis until \
                         memory frees (about to OOM otherwise)",
                    );
                }
                self.log_inflight(match refusal {
                    Some(Refusal::Host { .. }) => "admission paused: host memory below floor",
                    Some(Refusal::Rss { .. }) => "admission paused: process RSS at ceiling",
                    Some(Refusal::Reserved { .. }) | None => {
                        "admission paused: reservations at ceiling"
                    }
                });
                paused = true;
                // Reclaim only when memory is actually short (host floor or
                // process RSS). A `Reserved` refusal is the predictive cap —
                // the sum of pessimistic per-job estimates — and says nothing
                // about live pressure; clearing there wiped ~1.2 GB of
                // compiled trait regexes (44k engines) on every pause, which
                // then recompiled while the queue drained. Measured on the
                // poppy worker benchmark: 8 such pauses per run at 2–3.8 GB
                // used against an 11 GB ceiling.
                //
                // Clearing the caches only moves memory back to the allocator.
                // This gate re-polls *live process memory*, so unless those pages
                // also go back to the OS the pause sees no improvement and waits
                // out the full re-poll for nothing.
                if matches!(refusal, Some(Refusal::Host { .. } | Refusal::Rss { .. })) {
                    let reclaim = tokio::task::spawn_blocking(|| {
                        cleave::clear_all_thread_caches();
                        crate::allocator::trim();
                    });
                    if let Err(e) = reclaim.await {
                        tracing::warn!(error = %e, "cache-clear task failed");
                    }
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
    fn try_reserve(&self, est: u64) -> Result<(), Refusal> {
        // Disabled: dispatch is bounded by slot count alone.
        if self.ceiling_bytes == 0 {
            self.reserved.fetch_add(est, Ordering::AcqRel);
            return Ok(());
        }

        let used = (self.used_fn)();
        let available = if self.host_floor_bytes == 0 {
            None
        } else {
            (self.available_fn)()
        };
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
                    return Err(Refusal::Reserved { reserved });
                }

                // Reactive: live usage must leave room for this job, catching
                // estimates that ran low and pressure from other processes.
                if let Some(used) = used
                    && used.saturating_add(est) > self.ceiling_bytes
                {
                    return Err(Refusal::Rss { used });
                }

                // Host: the kernel must still have the floor left after this
                // job's estimate. Independent of the ceiling — a host shared
                // with other tenants, or whose earlier estimates all ran low,
                // can be out of memory while every check above passes.
                if let Some(available) = available
                    && available.saturating_sub(est) < self.host_floor_bytes
                {
                    return Err(Refusal::Host { available });
                }
            }

            match self.reserved.compare_exchange_weak(
                reserved,
                reserved.saturating_add(est),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
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
        let host_available_gb = (self.available_fn)().map(|a| a as f64 / GIB);
        tracing::warn!(
            reason,
            inflight = jobs.len(),
            reserved_gb = reserved as f64 / GIB,
            ceiling_gb = self.ceiling_bytes as f64 / GIB,
            used_gb = used as f64 / GIB,
            host_available_gb,
            host_floor_gb = self.host_floor_bytes as f64 / GIB,
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

/// Why `try_reserve` would not admit a job right now. `Host` is the one that
/// precedes an OOM kill and is logged at error level; the other two are the
/// gate doing its routine job against this process's own ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    /// Committed reservations plus this estimate exceed the ceiling.
    Reserved { reserved: u64 },
    /// Live process RSS plus this estimate exceeds the ceiling.
    Rss { used: u64 },
    /// The kernel's available memory minus this estimate is below the floor.
    Host { available: u64 },
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
        gate_with_host(ceiling, est, used_fn, no_used, 0)
    }

    fn gate_with_host(
        ceiling: u64,
        est: u64,
        used_fn: fn() -> Option<u64>,
        available_fn: fn() -> Option<u64>,
        host_floor_bytes: u64,
    ) -> Arc<MemoryAdmission> {
        Arc::new(MemoryAdmission {
            ceiling_bytes: ceiling,
            fixed_est_bytes: Some(est),
            reserved: AtomicU64::new(0),
            next_id: AtomicU64::new(0),
            inflight: Mutex::new(Vec::new()),
            released: Notify::new(),
            used_fn,
            available_fn,
            host_floor_bytes,
        })
    }

    /// 3 GB available on the host — below floor + a 2 GB estimate.
    #[allow(clippy::unnecessary_wraps)]
    fn avail_3gb() -> Option<u64> {
        Some(3 * GB)
    }

    /// 20 GB available on the host — ample.
    #[allow(clippy::unnecessary_wraps)]
    fn avail_20gb() -> Option<u64> {
        Some(20 * GB)
    }

    #[test]
    fn host_floor_refuses_when_kernel_has_no_room() {
        // Ceiling and RSS both say fine; the host says 3 GB left. One job is
        // in flight, so the hatch is closed: the second must wait.
        let g = gate_with_host(64 * GB, 2 * GB, no_used, avail_3gb, 2 * GB);
        assert!(
            g.try_reserve(2 * GB).is_ok(),
            "first job always admits (hatch)"
        );
        assert_eq!(
            g.try_reserve(2 * GB),
            Err(Refusal::Host { available: 3 * GB }),
            "3 GB avail - 2 GB est < 2 GB floor"
        );
        let ample = gate_with_host(64 * GB, 2 * GB, no_used, avail_20gb, 2 * GB);
        assert!(ample.try_reserve(2 * GB).is_ok());
        assert!(
            ample.try_reserve(2 * GB).is_ok(),
            "20 GB available leaves the floor"
        );
    }

    #[test]
    fn host_floor_zero_disables_host_check() {
        let g = gate_with_host(64 * GB, 2 * GB, no_used, avail_3gb, 0);
        assert!(g.try_reserve(2 * GB).is_ok());
        assert!(g.try_reserve(2 * GB).is_ok());
    }

    #[test]
    fn archive_bytes_signatures_are_recognised() {
        assert!(looks_like_archive_bytes(b"\x1f\x8b\x08\x00rest"));
        assert!(looks_like_archive_bytes(b"PK\x03\x04\x14\x00"));
        assert!(looks_like_archive_bytes(b"\xfd7zXZ\x00\x00"));
        let mut tar = vec![0u8; 512];
        tar[257..262].copy_from_slice(b"ustar");
        assert!(looks_like_archive_bytes(&tar));
        assert!(!looks_like_archive_bytes(b"\x7fELF\x02\x01\x01"));
        assert!(!looks_like_archive_bytes(b"MZ\x90\x00"));
        assert!(!looks_like_archive_bytes(b""));
    }

    #[test]
    fn suffix_less_tarball_estimates_as_archive_when_sniffed() {
        // `v1`: a 37 MB gzip tarball hopper hands out with no suffix. Path and
        // type say "flat"; the bytes say archive.
        let flat = dynamic_estimate_bytes("v1", "data", 37 * 1024 * 1024, None);
        assert_eq!(flat, DEFAULT_FLAT_ESTIMATE_BYTES);
        let sniffed = dynamic_estimate_bytes("v1", "data", 37 * 1024 * 1024, Some(b"\x1f\x8b\x08"));
        assert!(sniffed >= MIN_ARCHIVE_ESTIMATE_BYTES);
        assert!(sniffed > flat);
    }

    #[tokio::test]
    async fn reservation_releases_on_drop() {
        let g = gate(10 * GB, 1536 * 1024 * 1024, no_used);
        let guard = g
            .admit("a".into(), "p".into(), "bin".into(), 40, None)
            .await;
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
        assert!(g.try_reserve(GB).is_ok());
        assert!(g.try_reserve(GB).is_ok());
        assert_eq!(g.reserved.load(Ordering::Acquire), 2 * GB);
    }

    #[test]
    fn lone_job_always_admits_for_forward_progress() {
        // Nothing in flight, host under the ceiling → the forward-progress hatch
        // admits even when one slot's estimate alone exceeds the ceiling.
        let g = gate(GB, 2 * GB, no_used);
        assert!(g.try_reserve(2 * GB).is_ok());

        // Nothing in flight but RSS is already past the ceiling, commonly from
        // allocator-retained arenas after earlier archive jobs. Still admit one
        // job so a low ceiling cannot deadlock the worker.
        let g = gate(10 * GB, GB, used_11gb);
        assert!(g.try_reserve(GB).is_ok());
    }

    #[test]
    fn predictive_cap_pauses_a_burst_before_it_allocates() {
        // No live signal, so only committed reservations gate. With a 1.5 GB
        // estimate and a 10 GB ceiling, the 7th slot would commit 10.5 GB.
        let est = 1536 * 1024 * 1024;
        let g = gate(10 * GB, est, no_used);
        let mut admitted = 0;
        while g.try_reserve(est).is_ok() {
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
        assert!(g.try_reserve(est).is_err());
    }

    #[test]
    fn dynamic_estimate_scales_archives_by_size() {
        let small_zip = dynamic_estimate_bytes("pkg.zip", "data", 10 * 1024 * 1024, None);
        let big_zip = dynamic_estimate_bytes("pkg.zip", "data", 100 * 1024 * 1024, None);
        assert!(small_zip >= MIN_ARCHIVE_ESTIMATE_BYTES);
        assert!(big_zip > small_zip);
        assert!(big_zip <= MAX_ARCHIVE_ESTIMATE_BYTES);
        assert_eq!(
            dynamic_estimate_bytes("plain.js", "javascript", 10 * 1024 * 1024, None),
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
                        if g.try_reserve(est).is_ok() {
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
        assert!(g.try_reserve(est).is_ok());
        assert!(g.try_reserve(est).is_ok());
        assert!(g.try_reserve(1).is_err());
        assert_eq!(g.reserved.load(Ordering::Acquire), 10 * GB);
    }
}
