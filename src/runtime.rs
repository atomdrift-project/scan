//! Process-wide setup a binary running this analysis stack must perform.
//!
//! These are properties of the *process*, not of any one scan: which signals
//! are blocked, whether a debugger can attach, which environment variables the
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

/// Establish the process-wide state the analysis stack expects.
///
/// Blocks `SIGUSR1` so every later thread inherits the blocked mask and the
/// dedicated handler can consume it by `sigwait` — without this the default
/// disposition kills the process on a thread dump. Permits a forked debugger
/// to attach under `yama.ptrace_scope=1`, which is what makes a live stack
/// obtainable from a wedged worker. Lifts the open-file limit (see
/// [`raise_open_file_limit`]). Then settles the environment the downstream
/// crates read.
///
/// # Safety
///
/// Call once, as the first statement of `main`, before spawning any thread and
/// before reading any of the environment variables involved. This sets
/// environment variables, which is undefined behaviour if another thread is
/// concurrently reading or writing the environment.
pub unsafe fn install() {
    #[cfg(unix)]
    // SAFETY: `sigemptyset`/`sigaddset` initialize the mask they are given
    // before `pthread_sigmask` reads it, and the caller guarantees this is the
    // only thread.
    unsafe {
        let mut mask: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut mask);
        libc::sigaddset(&mut mask, libc::SIGUSR1);
        libc::pthread_sigmask(libc::SIG_BLOCK, &mask, std::ptr::null_mut());
    }

    #[cfg(target_os = "linux")]
    // SAFETY: `prctl` with `PR_SET_PTRACER` takes scalars and touches no
    // memory this process owns.
    unsafe {
        libc::prctl(libc::PR_SET_PTRACER, libc::PR_SET_PTRACER_ANY, 0, 0, 0);
    }

    raise_open_file_limit();
    propagate_no_analysis_cache();
    propagate_follow_aliases();
}

/// Install the crash and thread-dump handlers a long-lived daemon needs.
///
/// Separate from [`install`] because it spawns a thread: it has to run after
/// the signal mask is set, and it is the point past which [`install`]'s
/// precondition no longer holds.
pub fn install_diagnostics() {
    crate::crash_dump::install();
    crate::thread_dump::install();
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
