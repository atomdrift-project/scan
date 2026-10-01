//! Process-wide setup a binary running this analysis stack must perform.
//!
//! These are properties of the *process*, not of any one scan: which signals
//! are blocked, how wide the Rayon pool is, which environment variables the
//! downstream crates will read. A second binary that runs the same workload
//! without them does not merely lose a convenience — it loses the behaviour the
//! fleet was tuned around, quietly, and only under load.
//!
//! The one thing that cannot live here is the global allocator. A
//! `#[global_allocator]` static has to be declared in the binary crate, so each
//! binary declares its own against [`cleave::JEMALLOC_CONF`], the shared tuning
//! string. Everything else is here so it is written once.

/// `SCAN_NO_ANALYSIS_CACHE=1` — one switch that disables every cache of our
/// own analysis across the stack: filefacts file metadata, stng extracted
/// strings, cleave analysis results, and scan's analysis envelope + LLM
/// verdicts. Download caches (fletch registry metadata) and rule-compilation
/// caches (YARA, trait mapper) stay on — they hold inputs, not analysis.
///
/// Implemented by filling in each layer's own env var, so per-layer semantics
/// stay defined in one place and child processes inherit the policy. Only
/// unset vars are filled in: a per-layer var the operator set explicitly
/// always wins over the umbrella.
fn propagate_no_analysis_cache() {
    let on = std::env::var("SCAN_NO_ANALYSIS_CACHE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    if !on {
        return;
    }
    // CLEAVE_SKIP_CACHE=1 also drags cleave's YARA rule-compilation cache with
    // it (legacy behavior); pin that cache back on — recompiling rules costs
    // 4-18s per process and compiles rules, not sample analysis.
    let defaults = [
        ("FILEFACTS_CACHE", "0"),
        ("STNG_STRING_CACHE", "0"),
        ("CLEAVE_SKIP_CACHE", "1"),
        ("CLEAVE_SKIP_YARA_CACHE", "0"),
        ("SCAN_ANALYSIS_CACHE", "0"),
    ];
    for (key, value) in defaults {
        if std::env::var_os(key).is_none() {
            // SAFETY: the caller's documented precondition is that no thread
            // has been spawned, so no concurrent environment access can race.
            unsafe { std::env::set_var(key, value) };
        }
    }
}

/// Accept the older spelling of the reference-following variables.
///
/// `SCAN_FETCH`/`SCAN_FETCH_DEPTH` predate `SCAN_FOLLOW`/`SCAN_FOLLOW_DEPTH`.
/// A deployment still setting the old names must not silently stop following
/// references, so the old name fills the new one when the new one is unset.
/// The canonical variable wins when both are present.
fn propagate_follow_aliases() {
    for (canonical, legacy) in [
        ("SCAN_FOLLOW", "SCAN_FETCH"),
        ("SCAN_FOLLOW_DEPTH", "SCAN_FETCH_DEPTH"),
    ] {
        if std::env::var_os(canonical).is_none()
            && let Some(value) = std::env::var_os(legacy)
        {
            // SAFETY: as above — before any thread exists.
            unsafe { std::env::set_var(canonical, value) };
        }
    }
}

/// Raise the soft open-file limit as far as the hard limit allows.
///
/// systemd starts every service at a soft `RLIMIT_NOFILE` of 1024 (the hard
/// limit is 524288), and unlike Go, Rust's runtime never lifts it. A worker
/// holds several descriptors per in-flight analysis — the sample, its extracted
/// members, rizin's pipes — so 1024 is a ceiling on concurrency that nothing
/// reports. uruk-hai's idle worker sat pinned at exactly 1024 with ~1,170
/// analyses in flight (2026-09-25), logging ~680 EMFILE errors a minute, and
/// every symptom was somewhere else: each file the process tried to open
/// failed, so `traits_repo::version()` read no sidecar (every report went out
/// without `rev`, hopper could not skip re-posts, and the dashboard showed no
/// traits), `/proc/self` was unreadable (no RSS), and `TempDir`'s drop, which
/// ignores errors, could not remove what it extracted — 66k leaked
/// `cleave-archive-*` dirs exhausted the tmpfs's inodes and turned every new
/// extraction into "No space left on device".
///
/// Tries the hard limit first; where the kernel refuses it (Linux caps at
/// `fs.nr_open` even when the hard limit is unlimited, macOS at
/// `kern.maxfilesperproc`) it falls back to smaller ceilings. Never lowers the
/// limit. Children inherit it, which is what covers the server's idle worker.
fn raise_open_file_limit() {
    #[cfg(unix)]
    // SAFETY: getrlimit/setrlimit read and write only the struct passed in.
    unsafe {
        let mut lim: libc::rlimit = std::mem::zeroed();
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) != 0 {
            return;
        }
        for target in [lim.rlim_max, 1 << 20, 10_240] {
            if target <= lim.rlim_cur || target > lim.rlim_max {
                continue;
            }
            let raised = libc::rlimit {
                rlim_cur: target,
                rlim_max: lim.rlim_max,
            };
            if libc::setrlimit(libc::RLIMIT_NOFILE, &raised) == 0 {
                return;
            }
        }
    }
}

/// Whether [`install`] blocked `SIGUSR1`. The dump thread may only `sigwait`
/// for a blocked signal; unblocked, its default disposition kills the process.
#[cfg(unix)]
static SIGUSR1_BLOCKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Block `SIGUSR1` in the calling thread, so every thread spawned after it
/// inherits the mask and the dump thread can consume the signal by `sigwait`.
#[cfg(unix)]
fn block_sigusr1() {
    // SAFETY: `sigemptyset`/`sigaddset` initialize the mask before
    // `pthread_sigmask` reads it, and all three touch only that local.
    let rc = unsafe {
        let mut mask: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut mask);
        libc::sigaddset(&mut mask, libc::SIGUSR1);
        libc::pthread_sigmask(libc::SIG_BLOCK, &mask, std::ptr::null_mut())
    };
    if rc == 0 {
        SIGUSR1_BLOCKED.store(true, std::sync::atomic::Ordering::Release);
    } else {
        // Logging is not up yet; stderr is all there is.
        eprintln!("warning: cannot block SIGUSR1 (error {rc}); thread dumps are disabled");
    }
}

/// Establish the process-wide state the analysis stack expects.
///
/// Blocks `SIGUSR1` so every later thread inherits the blocked mask and the
/// dedicated thread can consume it by `sigwait` — without this the default
/// disposition kills the process on a thread dump. Lifts the open-file limit
/// (see `raise_open_file_limit`). Then settles the environment the
/// downstream crates read.
///
/// # Safety
///
/// Call once, as the first statement of `main`, before spawning any thread and
/// before reading any of the environment variables involved. This sets
/// environment variables, which is undefined behaviour if another thread is
/// concurrently reading or writing the environment.
pub unsafe fn install() {
    #[cfg(unix)]
    block_sigusr1();
    raise_open_file_limit();
    propagate_no_analysis_cache();
    propagate_follow_aliases();
}

/// Install the crash dump, the thread dump and the heap profiler, and start
/// the thread that serves `SIGUSR1` dumps.
///
/// Separate from [`install`] because it spawns a thread and logs: it runs
/// after `install`, whose signal mask that thread must inherit, and after the
/// tracing subscriber is up.
pub fn install_diagnostics() {
    crate::crash_dump::install();
    crate::thread_dump::install();
    crate::heap_profile::install();
    #[cfg(unix)]
    spawn_sigusr1_thread();
}

/// Serve `SIGUSR1` (the Linux stand-in for BSD's `SIGINFO`): an all-thread
/// backtrace and, when enabled, a heap profile — captured in-process, so it
/// works in jails where no debugger can attach.
///
/// The thread lives as long as the process. It parks in `sigwait`, so there is
/// nothing to join, and process exit ends it.
#[cfg(unix)]
fn spawn_sigusr1_thread() {
    if !SIGUSR1_BLOCKED.load(std::sync::atomic::Ordering::Acquire) {
        tracing::warn!("SIGUSR1 is not blocked; not starting the thread-dump listener");
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("sigusr1".into())
        .spawn(|| {
            // SAFETY: as in `block_sigusr1`: the calls initialize and read
            // only this local mask.
            let mask = unsafe {
                let mut mask: libc::sigset_t = std::mem::zeroed();
                libc::sigemptyset(&mut mask);
                libc::sigaddset(&mut mask, libc::SIGUSR1);
                mask
            };
            loop {
                let mut sig: libc::c_int = 0;
                // SAFETY: `mask` is initialized and holds only a signal every
                // thread has blocked; `sig` is a valid out-pointer.
                if unsafe { libc::sigwait(&mask, &mut sig) } != 0 {
                    continue;
                }
                crate::thread_dump::dump_all_threads();
                crate::heap_profile::dump_on_signal();
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "cannot start the SIGUSR1 listener; thread dumps are disabled");
    }
}

/// Build the process-wide Rayon pool cleave's analysis fans out on, and say
/// how it was sized.
///
/// Every binary running this stack needs the same pool: the width the fleet
/// was measured at, stacks deep enough for nested archive analysis, and each
/// worker registered for the thread dump and demoted one scheduling notch. A
/// pool built any other way fails quietly, under load.
///
/// `CLEAVE_RAYON_THREADS` overrides the width.
pub fn install_rayon_pool() {
    let override_threads = std::env::var("CLEAVE_RAYON_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0);
    let physical = cleave::memory_tracker::physical_cpu_count();
    let logical = cleave::memory_tracker::cpu_count();
    let detected_cores = pool_cores(physical, logical, smt_pool_vendor());
    let threads = override_threads.unwrap_or_else(|| {
        detected_cores.unwrap_or_else(|| {
            tracing::warn!(
                fallback = RAYON_FALLBACK_THREADS,
                "CPU count detection failed; rayon pool DOWNGRADED to \
                 {RAYON_FALLBACK_THREADS} threads. On a many-core host this throttles \
                 throughput and oversubscribes workers — set CLEAVE_RAYON_THREADS to the \
                 core count.",
            );
            RAYON_FALLBACK_THREADS
        })
    });
    // 256 MB stacks: cleave's archive analysis is nested-parallel, and a rayon
    // worker blocked in an inner join steals other pending tasks — including
    // other in-flight analyses' tasks on this shared pool — and runs them on
    // top of its current stack. Frames from independent deep analyses stack
    // up, so the headroom must cover several, not one (64 MB overflowed in
    // production with 4 large archives in flight). Stacks are virtual memory;
    // only pages actually touched are committed, so the cost of the extra
    // headroom is address space, not RSS.
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(crate::RAYON_STACK_MB * 1024 * 1024)
        .thread_name(|i| format!("rayon-{i}"))
        // Each worker registers for the SIGUSR1 in-process thread dump (no
        // debugger can attach in the production jails) and steps one
        // scheduling notch below normal (see `demote_current_thread`).
        .start_handler(|_| {
            crate::thread_dump::register_self();
            crate::thread_priority::demote_current_thread();
        })
        .build_global()
    {
        tracing::warn!(error = %e, "failed to install global rayon pool; using default");
    }
    let active_threads = rayon::current_num_threads();
    // A pool smaller than the detected core count is a resource downgrade that
    // oversubscribes the worker slots — surface it loudly so it is diagnosable
    // from a single log line.
    if let Some(cores) = detected_cores
        && active_threads < cores
    {
        tracing::warn!(
            rayon_threads = active_threads,
            detected_cores = cores,
            "rayon pool smaller than detected cores; worker slots may oversubscribe the CPU",
        );
    }
    tracing::info!(
        threads = active_threads,
        detected_cores,
        stack_mb = crate::RAYON_STACK_MB,
        "rayon pool ready"
    );
}

/// Pool width when CPU detection fails entirely.
const RAYON_FALLBACK_THREADS: usize = 4;

/// Below this many physical cores an SMT host takes its logical count.
const SMALL_HOST_CORES: usize = 8;

/// The pool width for this host's shape, before any override.
///
/// On a wide host, physical cores win: logical SMT siblings (32 on a 16-core
/// host) oversubscribe archive-member analysis, and S2 dropped 56.6 s → 51.8 s
/// at 16 threads. On a small SMT host the opposite holds: the pool is the only
/// parallelism there is, and it idles whenever the one admitted archive is in a
/// serial phase (tar/gzip read). Measured 2026-08-30 on a 4c/8t AMD Ryzen 3400G
/// over the 121-job poppy worker benchmark: 4 threads 34:13, 5 → 32:47,
/// 6 → 31:15, 8 → 27:43 (−19%), peak RSS flat (4.3–4.6 GB), outputs identical.
/// The wider pool is taken only where that class of sibling exists (see
/// [`smt_pool_vendor`]).
fn pool_cores(physical: Option<usize>, logical: Option<usize>, smt_vendor: bool) -> Option<usize> {
    match (physical, logical) {
        (Some(p), Some(l)) if p < SMALL_HOST_CORES && l > p && smt_vendor => Some(l),
        (Some(p), _) => Some(p),
        (None, l) => l,
    }
}

/// Whether this host's SMT siblings are the kind the small-host pool policy
/// was measured on: x86 hyperthreads from AMD (measured) or Intel (same
/// design). Apple silicon never qualifies — its extra "logical" capacity is
/// efficiency cores, certified not to help this workload — and neither does
/// an unknown vendor.
fn smt_pool_vendor() -> bool {
    if cfg!(target_vendor = "apple") || !cfg!(target_arch = "x86_64") {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .is_ok_and(|info| info.contains("AuthenticAMD") || info.contains("GenuineIntel"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        // x86_64 FreeBSD/Windows/illumos hosts: AMD/Intel is the only x86 SMT
        // there is; the arch check above already excludes everything else.
        true
    }
}

#[cfg(test)]
mod pool_tests {
    use super::pool_cores;

    #[test]
    fn small_smt_hosts_take_their_siblings_and_wide_hosts_do_not() {
        assert_eq!(pool_cores(Some(4), Some(8), true), Some(8));
        assert_eq!(pool_cores(Some(4), Some(8), false), Some(4));
        assert_eq!(pool_cores(Some(16), Some(32), true), Some(16));
        assert_eq!(pool_cores(Some(4), Some(4), true), Some(4));
        assert_eq!(pool_cores(None, Some(6), true), Some(6));
        assert_eq!(pool_cores(None, None, true), None);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::raise_open_file_limit;

    fn nofile() -> (libc::rlim_t, libc::rlim_t) {
        // SAFETY: getrlimit writes only the struct passed in.
        unsafe {
            let mut lim: libc::rlimit = std::mem::zeroed();
            assert_eq!(libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim), 0);
            (lim.rlim_cur, lim.rlim_max)
        }
    }

    #[test]
    fn open_file_limit_is_raised_never_lowered() {
        let (before, hard) = nofile();
        raise_open_file_limit();
        let (after, hard_after) = nofile();
        assert!(after >= before, "soft limit lowered: {before} -> {after}");
        assert_eq!(hard, hard_after, "the hard limit is not ours to change");
        // Where there was headroom, some of it must have been taken: staying at
        // a low soft limit is exactly the failure this exists to prevent.
        if hard > before && before < 10_240 {
            assert!(
                after > before,
                "soft limit left at {before} under a hard limit of {hard}"
            );
        }
    }
}
