//! Live registry of in-flight analyses for the periodic worker census.
//!
//! Distinct from [`crate::crash_dump`]: that registry is async-signal-safe and
//! frozen at register time, written straight to stderr from the `SIGABRT`
//! handler. This one is read from the normal summary task every heartbeat to
//! answer "what is each slot doing, and is it stuck?" — per-slot file, size,
//! total elapsed, current stage (phase) and time in that stage, the worker
//! thread, and a best-effort hint at what a blocked analysis is waiting on.
//!
//! It holds a clone of each analysis's [`cleave::PhaseTracker`], so the census
//! reads the live phase without duplicating cleave's phase plumbing. Cost is one
//! `Mutex<Vec>` push/scan per analysis and per heartbeat — negligible against a
//! multi-second analysis, and never on the hot per-member path.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// One in-flight analysis, registered for the duration of [`run_job`].
///
/// [`run_job`]: crate::worker
#[derive(Debug)]
pub struct Entry {
    /// Monotonic per-process analysis id, matching the lifecycle log lines.
    pub analysis_id: u64,
    /// Shortened sha256, for cross-referencing the lifecycle logs.
    pub sha: Arc<str>,
    /// Display name (basename) of the file under analysis.
    pub file: Arc<str>,
    /// Hopper-reported size in bytes.
    pub size_bytes: u64,
    /// Hopper-reported file type (e.g. `elf`, `zip`).
    pub file_type: Arc<str>,
    /// OS thread id of the blocking worker thread this analysis runs on; `0`
    /// until it lands on one (it is registered before `spawn_blocking` starts).
    /// On Windows this is `GetCurrentThreadId`; on Unix it matches `/proc`/ps.
    pub thread_id: AtomicU64,
    /// When the analysis started, for total-elapsed reporting.
    pub started: Instant,
    /// Live phase tracker, shared with cleave; `phase.get()` is the current stage.
    pub phase: cleave::PhaseTracker,
}

fn registry() -> &'static Mutex<Vec<Arc<Entry>>> {
    static REGISTRY: OnceLock<Mutex<Vec<Arc<Entry>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Removes its entry from the registry when the analysis finishes or its task is
/// dropped. A poisoned lock is tolerated: the entry simply lingers until the
/// next successful lock prunes it (the census filters defensively too).
#[derive(Debug)]
pub struct Guard(u64);

impl Drop for Guard {
    fn drop(&mut self) {
        if let Ok(mut reg) = registry().lock() {
            reg.retain(|e| e.analysis_id != self.0);
        }
    }
}

/// Record an analysis as in flight. The returned guard deregisters it on drop.
#[must_use]
pub fn register(
    analysis_id: u64,
    sha: Arc<str>,
    file: Arc<str>,
    size_bytes: u64,
    file_type: Arc<str>,
    started: Instant,
    phase: cleave::PhaseTracker,
) -> Guard {
    let entry = Arc::new(Entry {
        analysis_id,
        sha,
        file,
        size_bytes,
        file_type,
        thread_id: AtomicU64::new(0),
        started,
        phase,
    });
    if let Ok(mut reg) = registry().lock() {
        reg.push(entry);
    }
    Guard(analysis_id)
}

/// Attach the OS thread id once the analysis lands on a blocking worker thread.
pub fn set_thread_id(analysis_id: u64, thread_id: u64) {
    if let Ok(reg) = registry().lock()
        && let Some(entry) = reg.iter().find(|e| e.analysis_id == analysis_id)
    {
        entry.thread_id.store(thread_id, Ordering::Relaxed);
    }
}

/// Snapshot the current in-flight set (cheap `Arc` clones), oldest first so the
/// most-stuck analyses lead the census.
#[must_use]
pub fn snapshot() -> Vec<Arc<Entry>> {
    let mut rows = registry().lock().map(|r| r.clone()).unwrap_or_default();
    rows.sort_by_key(|e| e.started);
    rows
}

/// Best-effort name of what a blocked thread is waiting on: its kernel
/// wait-channel (e.g. `futex`, `pipe_wait`, a lock symbol). Read straight from
/// procfs on Linux — no shell-out, async-cheap. Returns `None` when the thread
/// is running, the channel is unavailable, or the platform doesn't expose it, so
/// the caller falls back to the analysis phase. Other platforms resolve
/// wait-channels in batch via [`wait_channels`].
#[must_use]
pub fn wait_channel(thread_id: u64) -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        if thread_id == 0 {
            return None;
        }
        let raw = std::fs::read_to_string(format!("/proc/self/task/{thread_id}/wchan")).ok()?;
        let chan = raw.trim();
        if chan.is_empty() || chan == "0" {
            None
        } else {
            Some(chan.to_string())
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = thread_id;
        None
    }
}

/// Resolve the wait-channel of many worker threads at once, keyed by OS thread
/// id. Batching matters off Linux, where the only portable source is one `ps`
/// invocation per census rather than one syscall per thread:
///
/// - **Linux**: per-thread `read` of `/proc/self/task/<tid>/wchan` (no fork).
/// - **FreeBSD**: one `ps -H -o tid=,wchan=` lists every thread's wait channel.
/// - **illumos/Solaris**: one `ps -L -o lwp=,s=,wchan=`; the value carries the
///   LWP state plus wchan address (illumos has no symbolic wchan, but the state
///   still says sleeping-vs-runnable and the address disambiguates lock sites).
/// - **macOS / other**: empty — no per-thread wait channel is exposed, so the
///   census falls back to the analysis stage.
///
/// Threads that are running (no wait channel) are simply absent from the map.
#[must_use]
pub fn wait_channels(thread_ids: &[u64]) -> std::collections::HashMap<u64, String> {
    // `mut` is used on platforms whose arm populates the map (Linux) or replaces
    // it (FreeBSD/illumos); macOS and others leave it empty.
    #[allow(unused_mut)]
    let mut map = std::collections::HashMap::new();
    if thread_ids.is_empty() {
        return map;
    }
    #[cfg(target_os = "linux")]
    {
        for &tid in thread_ids {
            if let Some(chan) = wait_channel(tid) {
                map.insert(tid, chan);
            }
        }
    }
    // BSD/SysV `ps` treat everything after the first `=` in an -o argument as
    // literal header text (so `-o tid=,wchan=` is ONE column named ",wchan=").
    // Each column must be its own -o flag to get null headers. Verified on
    // FreeBSD 14: `-o tid= -o wchan=` emits `tid wchan` pairs matching thr_self.
    #[cfg(target_os = "freebsd")]
    {
        let pid = std::process::id().to_string();
        if let Some(found) = ps_thread_channels(&["-p", &pid, "-H", "-o", "tid=", "-o", "wchan="]) {
            map = filter_to(found, thread_ids);
        }
    }
    #[cfg(any(target_os = "illumos", target_os = "solaris"))]
    {
        let pid = std::process::id().to_string();
        if let Some(found) =
            ps_thread_channels(&["-L", "-o", "lwp=", "-o", "s=", "-o", "wchan=", "-p", &pid])
        {
            map = filter_to(found, thread_ids);
        }
    }
    map
}

/// Keep only the requested thread ids from a process-wide `ps` result.
#[cfg(any(target_os = "freebsd", target_os = "illumos", target_os = "solaris"))]
fn filter_to(
    mut found: std::collections::HashMap<u64, String>,
    thread_ids: &[u64],
) -> std::collections::HashMap<u64, String> {
    let wanted: std::collections::HashSet<u64> = thread_ids.iter().copied().collect();
    found.retain(|id, _| wanted.contains(id));
    found
}

/// Run `ps` and parse each line as `<thread-id> <wait-channel...>`: the first
/// whitespace token is the id, the remainder (trimmed) is the channel. Lines
/// whose channel is empty or `-` (running) are dropped. Best-effort: any failure
/// returns `None` so the caller falls back to the analysis stage.
#[cfg(any(target_os = "freebsd", target_os = "illumos", target_os = "solaris"))]
fn ps_thread_channels(args: &[&str]) -> Option<std::collections::HashMap<u64, String>> {
    let output = std::process::Command::new("ps").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut map = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        let mut parts = line.splitn(2, char::is_whitespace);
        let Some(id) = parts.next().and_then(|s| s.trim().parse::<u64>().ok()) else {
            continue;
        };
        let channel = parts.next().map(str::trim).unwrap_or_default();
        if !channel.is_empty() && channel != "-" {
            map.insert(id, channel.to_string());
        }
    }
    Some(map)
}

/// Tally the kernel wait-channel of every thread in this process, keyed by
/// channel.
///
/// The per-slot census tracks only the analysis's coordinator thread, but an
/// archive fans its members across the whole rayon pool — so a wedged archive's
/// real blockage is on `rayon-*` workers the census never names. This aggregate
/// fills that gap: on a wedge it shows the *resource classes* the process is
/// stuck on (e.g. many threads in a YARA/wasmtime symbol, some in `pipe_wait` on
/// a subprocess, some in `futex` on a lock), turning the archive black box into
/// a resource breakdown. Running threads (no wait channel) tally under
/// `(running)`, so a high count there reads as CPU-busy rather than blocked.
///
/// Linux reads procfs directly (no fork); FreeBSD/illumos shell one `ps`. Empty
/// on platforms with no per-thread wait channel (e.g. macOS).
#[must_use]
pub fn thread_wait_summary() -> std::collections::BTreeMap<String, usize> {
    // `mut` is used on platforms whose arm populates the tally; macOS leaves it empty.
    #[allow(unused_mut)]
    let mut tally = std::collections::BTreeMap::new();
    #[cfg(target_os = "linux")]
    if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
        for tid in entries.flatten() {
            let chan = std::fs::read_to_string(tid.path().join("wchan")).unwrap_or_default();
            *tally.entry(normalize_channel(chan.trim())).or_insert(0) += 1;
        }
    }
    #[cfg(any(target_os = "freebsd", target_os = "illumos", target_os = "solaris"))]
    {
        let pid = std::process::id().to_string();
        #[cfg(target_os = "freebsd")]
        let args = ["-p", &pid, "-H", "-o", "wchan="];
        #[cfg(any(target_os = "illumos", target_os = "solaris"))]
        let args = ["-L", "-o", "wchan=", "-p", &pid];
        if let Ok(output) = std::process::Command::new("ps").args(args).output()
            && output.status.success()
        {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                *tally.entry(normalize_channel(line.trim())).or_insert(0) += 1;
            }
        }
    }
    tally
}

/// Map an empty / running wait channel to a stable `(running)` key.
#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "illumos",
    target_os = "solaris"
))]
fn normalize_channel(chan: &str) -> String {
    if chan.is_empty() || chan == "0" || chan == "-" {
        "(running)".to_string()
    } else {
        chan.to_string()
    }
}

/// Render a [`thread_wait_summary`] tally as `chan=count` pairs, busiest first,
/// for a single log field.
#[must_use]
pub fn format_wait_summary(tally: &std::collections::BTreeMap<String, usize>) -> String {
    let mut pairs: Vec<(&String, &usize)> = tally.iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    pairs
        .iter()
        .map(|(chan, count)| format!("{chan}={count}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Total user+system CPU seconds consumed by this process so far (`RUSAGE_SELF`).
///
/// Sampling the delta across two heartbeats yields average cores busy, which
/// distinguishes a worker grinding through heavy work (cores busy) from one
/// wedged on locks, subprocesses, or I/O (cores ≈ idle while slots stay full).
#[must_use]
pub fn process_cpu_secs() -> f64 {
    #[cfg(unix)]
    {
        // SAFETY: getrusage into a zeroed rusage is the documented usage; we only
        // read the two timeval fields it always populates.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
            return 0.0;
        }
        let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1_000_000.0;
        secs(usage.ru_utime) + secs(usage.ru_stime)
    }
    #[cfg(windows)]
    {
        // FILETIME is 100 ns intervals; sum user + kernel like getrusage.
        #[repr(C)]
        #[derive(Clone, Copy, Default)]
        struct FileTime {
            low: u32,
            high: u32,
        }
        impl FileTime {
            fn as_secs_f64(self) -> f64 {
                let ticks = (u64::from(self.high) << 32) | u64::from(self.low);
                ticks as f64 / 10_000_000.0
            }
        }
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
            fn GetProcessTimes(
                process: *mut core::ffi::c_void,
                creation: *mut FileTime,
                exit: *mut FileTime,
                kernel: *mut FileTime,
                user: *mut FileTime,
            ) -> i32;
        }
        let mut creation = FileTime::default();
        let mut exit = FileTime::default();
        let mut kernel = FileTime::default();
        let mut user = FileTime::default();
        // SAFETY: all four out-pointers are valid locals; handle is the process
        // pseudo-handle (-1) which GetProcessTimes accepts.
        let ok = unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut creation,
                &mut exit,
                &mut kernel,
                &mut user,
            )
        };
        if ok == 0 {
            return 0.0;
        }
        user.as_secs_f64() + kernel.as_secs_f64()
    }
    #[cfg(not(any(unix, windows)))]
    {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u64) -> Guard {
        register(
            id,
            Arc::from("deadbeef"),
            Arc::from("evil.whl"),
            4096,
            Arc::from("zip"),
            Instant::now(),
            cleave::PhaseTracker::new(),
        )
    }

    #[test]
    fn register_then_guard_drop_frees_entry() {
        // A distinctive id so concurrent tests don't collide in the shared registry.
        let id = 0xCEED_0001;
        let present = || snapshot().iter().any(|e| e.analysis_id == id);
        assert!(!present(), "precondition: id not already registered");
        {
            let _g = entry(id);
            assert!(present(), "register must add the entry");
        }
        assert!(!present(), "dropping the guard must remove the entry");
    }

    #[test]
    fn set_thread_id_updates_live_entry() {
        let id = 0xCEED_0002;
        let _g = entry(id);
        set_thread_id(id, 4242);
        let tid = snapshot()
            .iter()
            .find(|e| e.analysis_id == id)
            .map(|e| e.thread_id.load(Ordering::Relaxed));
        assert_eq!(tid, Some(4242));
    }

    #[test]
    fn snapshot_is_oldest_first() {
        let id_old = 0xCEED_0003;
        let id_new = 0xCEED_0004;
        let _old = entry(id_old);
        let _new = entry(id_new);
        let rows = snapshot();
        let pos = |id| rows.iter().position(|e| e.analysis_id == id);
        assert!(pos(id_old) < pos(id_new), "older entry must sort first");
    }

    #[test]
    fn wait_channel_is_none_for_unknown_thread() {
        // tid 0 is never a real worker thread; on every platform this yields None
        // so the census falls back to the analysis stage.
        assert_eq!(wait_channel(0), None);
    }

    #[test]
    fn wait_channels_empty_input_is_empty() {
        // No threads to resolve must never fork `ps` or touch procfs.
        assert!(wait_channels(&[]).is_empty());
        // tid 0 resolves to nothing on every platform.
        assert!(!wait_channels(&[0]).contains_key(&0));
    }

    #[test]
    fn format_wait_summary_orders_by_count_then_name() {
        let mut tally = std::collections::BTreeMap::new();
        tally.insert("futex".to_string(), 3);
        tally.insert("pipe_wait".to_string(), 5);
        tally.insert("(running)".to_string(), 1);
        assert_eq!(
            format_wait_summary(&tally),
            "pipe_wait=5 futex=3 (running)=1"
        );
    }

    #[test]
    fn thread_wait_summary_never_panics() {
        // Contents are platform-dependent (empty on macOS); it must never panic
        // and counts must be positive where present.
        for count in thread_wait_summary().values() {
            assert!(*count > 0);
        }
    }

    #[test]
    fn process_cpu_secs_is_monotonic_nonnegative() {
        let a = process_cpu_secs();
        let mut x = 0u64;
        for i in 0..2_000_000 {
            x = x.wrapping_add(i);
        }
        assert!(x > 0);
        let b = process_cpu_secs();
        assert!(a >= 0.0 && b >= a, "cpu seconds must not go backwards");
    }
}
