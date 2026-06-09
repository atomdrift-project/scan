//! In-process all-thread backtraces for the `SIGUSR1` wedge dump.
//!
//! The previous `SIGUSR1` handler shelled out to `lldb`/`gdb` to attach to the
//! worker — which fails exactly where it is needed: macOS SIP blocks self-attach,
//! and the production FreeBSD bastille jails don't grant ptrace. This module
//! captures the stacks of the analysis threads from *inside* the process, with no
//! debugger and no ptrace, so a wedge can be diagnosed in the jail.
//!
//! Mechanism (Linux/FreeBSD): every analysis thread (the rayon pool workers and
//! the `spawn_blocking` analysis threads) registers its OS thread id via
//! [`register_self`]. On `SIGUSR1` the handler thread calls [`dump_all_threads`],
//! which signals each registered thread with `SIGUSR2`; the [`on_capture`]
//! handler — running on the target thread — walks its own stack with
//! [`backtrace::trace_unsynchronized`] and stores raw instruction pointers into a
//! preallocated slot (no allocation, no locks: async-signal-safe). The
//! coordinator then symbolizes those pointers (allocation is fine here, off the
//! signal path) and writes them to stderr.
//!
//! It is strictly operator-triggered (never automatic), so its blast radius is a
//! manual diagnostic on an already-wedged process. On platforms without per-thread
//! signalling (macOS) it prints a pointer to the breadcrumb/wait-channel census
//! instead.

use std::io::Write;
use std::sync::{Mutex, OnceLock};

/// Registry of analysis-thread OS ids to signal during a dump. Long-lived
/// threads (rayon pool, pooled `spawn_blocking`) register once; ids are never
/// removed, so a since-exited thread merely fails to signal (harmless).
fn registry() -> &'static Mutex<std::collections::BTreeSet<u64>> {
    static REGISTRY: OnceLock<Mutex<std::collections::BTreeSet<u64>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(std::collections::BTreeSet::new()))
}

/// Register the calling thread as an analysis thread to include in dumps. Cheap
/// and idempotent; call it from each rayon worker's start handler and at the top
/// of each blocking analysis.
pub fn register_self() {
    if let Ok(mut reg) = registry().lock() {
        reg.insert(os_thread_id());
    }
}

/// OS-level thread id for the calling thread.
///
/// Matches what `/proc/<pid>/task`, `ps`, and a debugger report, so log lines,
/// the in-flight census, and the wait-channel lookup can all correlate on it.
/// The single canonical implementation for the crate.
pub(crate) fn os_thread_id() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `gettid` takes no arguments and cannot fail.
        unsafe { libc::syscall(libc::SYS_gettid) as u64 }
    }
    #[cfg(target_os = "macos")]
    {
        let mut tid: u64 = 0;
        // SAFETY: writes this thread's id into a local; handle 0 means self.
        unsafe { libc::pthread_threadid_np(0, &mut tid) };
        tid
    }
    #[cfg(target_os = "freebsd")]
    {
        let mut tid: libc::c_long = 0;
        // SAFETY: `thr_self` writes the calling thread's id into the pointee.
        unsafe { libc::thr_self(&mut tid) };
        u64::try_from(tid).unwrap_or(0)
    }
    #[cfg(target_os = "openbsd")]
    {
        // SAFETY: `getthrid` takes no arguments and cannot fail.
        unsafe { libc::getthrid() as u64 }
    }
    #[cfg(any(target_os = "illumos", target_os = "solaris"))]
    {
        // The LWP id matches `/proc/<pid>/lwp/<id>` and the `lwp` column of
        // `ps -L`; the `libc` crate doesn't bind `_lwp_self(2)`, so declare it.
        unsafe extern "C" {
            fn _lwp_self() -> libc::id_t;
        }
        // SAFETY: `_lwp_self` takes no arguments and cannot fail.
        unsafe { _lwp_self() as u64 }
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "illumos",
        target_os = "solaris"
    )))]
    {
        0
    }
}

// --- Signal-based capture (Linux/FreeBSD) ----------------------------------

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
mod capture {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    /// Signal used to ask a thread to record its own stack.
    pub(super) const CAPTURE_SIGNAL: libc::c_int = libc::SIGUSR2;

    /// Capacity of the capture arena. Threads beyond this are simply not
    /// recorded (the count of attempted signals is still reported).
    pub(super) const MAX_THREADS: usize = 512;
    /// Frames captured per thread; deep cleave recursion is bounded elsewhere.
    pub(super) const MAX_FRAMES: usize = 64;

    /// Only true while a dump is in progress, so a stray `SIGUSR2` is ignored.
    pub(super) static CAPTURING: AtomicBool = AtomicBool::new(false);
    /// Next free slot; each signalled thread claims one with `fetch_add`.
    pub(super) static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);
    /// Threads that have finished writing their slot (or overflowed the arena).
    pub(super) static DONE: AtomicUsize = AtomicUsize::new(0);

    pub(super) static SLOT_TID: [AtomicU64; MAX_THREADS] =
        [const { AtomicU64::new(0) }; MAX_THREADS];
    pub(super) static SLOT_NFRAMES: [AtomicUsize; MAX_THREADS] =
        [const { AtomicUsize::new(0) }; MAX_THREADS];
    pub(super) static SLOT_FRAMES: [[AtomicUsize; MAX_FRAMES]; MAX_THREADS] =
        [const { [const { AtomicUsize::new(0) }; MAX_FRAMES] }; MAX_THREADS];

    /// `SIGUSR2` handler: records the interrupted thread's own stack.
    ///
    /// Async-signal-safe by construction: it only does atomic operations and a
    /// frame-pointer/unwind walk into preallocated storage — no allocation, no
    /// locks, no stdio.
    pub(super) extern "C" fn on_capture(_sig: libc::c_int) {
        if !CAPTURING.load(Ordering::Acquire) {
            return;
        }
        let idx = NEXT_SLOT.fetch_add(1, Ordering::AcqRel);
        if idx >= MAX_THREADS {
            DONE.fetch_add(1, Ordering::AcqRel);
            return;
        }
        SLOT_TID[idx].store(super::os_thread_id(), Ordering::Relaxed);
        let mut frames = 0usize;
        // SAFETY: `trace_unsynchronized` is the documented primitive for capturing
        // the current thread's stack from a signal handler; the closure only
        // stores integers into preallocated atomics.
        unsafe {
            backtrace::trace_unsynchronized(|frame| {
                if frames < MAX_FRAMES {
                    SLOT_FRAMES[idx][frames].store(frame.ip() as usize, Ordering::Relaxed);
                    frames += 1;
                    true
                } else {
                    false
                }
            });
        }
        SLOT_NFRAMES[idx].store(frames, Ordering::Release);
        DONE.fetch_add(1, Ordering::AcqRel);
    }

    /// Deliver [`CAPTURE_SIGNAL`] to one thread by OS id. Returns whether it was
    /// accepted (a since-exited thread fails harmlessly).
    pub(super) fn signal_thread(tid: u64) -> bool {
        let Ok(tid) = libc::c_long::try_from(tid) else {
            return false;
        };
        #[cfg(target_os = "linux")]
        {
            let pid = libc::c_long::from(unsafe { libc::getpid() });
            // SAFETY: `tgkill(tgid, tid, sig)` with our own pid and a valid signal.
            let ret = unsafe {
                libc::syscall(
                    libc::SYS_tgkill,
                    pid,
                    tid,
                    libc::c_long::from(CAPTURE_SIGNAL),
                )
            };
            ret == 0
        }
        #[cfg(target_os = "freebsd")]
        {
            // SAFETY: `thr_kill(id, sig)` targets a thread in our own process.
            unsafe { libc::thr_kill(tid, CAPTURE_SIGNAL) == 0 }
        }
    }
}

/// Install the capture-signal handler. Idempotent; call once at startup, before
/// the `SIGUSR1` dump path can run. No-op on platforms without per-thread
/// signalling.
pub fn install() {
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    {
        // SAFETY: a zeroed `sigaction` with our handler and an empty mask is a
        // valid disposition; `SA_RESTART` keeps interrupted syscalls transparent.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = capture::on_capture as usize;
            libc::sigemptyset(&mut action.sa_mask);
            action.sa_flags = libc::SA_RESTART;
            libc::sigaction(capture::CAPTURE_SIGNAL, &action, std::ptr::null_mut());
        }
    }
}

/// Capture and write every registered analysis thread's backtrace to stderr.
/// Operator-triggered (via `SIGUSR1`); safe to call when the worker is wedged.
pub fn dump_all_threads() {
    let mut out = std::io::stderr().lock();
    let pid = std::process::id();
    let _ = writeln!(out, "\n--- litmus in-process thread dump (pid {pid}) ---");
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    {
        write_registered_backtraces(&mut out);
    }
    #[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
    {
        let _ = writeln!(
            out,
            "  all-thread capture is unsupported on this platform; consult the \
             WEDGE breadcrumb (per-member) and thread_waits (per-thread wait \
             channel) fields in the worker summary instead",
        );
    }
    let _ = writeln!(out, "--- end thread dump ---\n");
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
fn write_registered_backtraces(out: &mut impl Write) {
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    let self_tid = os_thread_id();
    let targets: Vec<u64> = registry()
        .lock()
        .map(|reg| reg.iter().copied().filter(|&t| t != self_tid).collect())
        .unwrap_or_default();

    capture::NEXT_SLOT.store(0, Ordering::Release);
    capture::DONE.store(0, Ordering::Release);
    capture::CAPTURING.store(true, Ordering::Release);

    let signalled = targets
        .iter()
        .filter(|&&tid| capture::signal_thread(tid))
        .count();

    // Wait for the signalled threads to record their slots, bounded so a thread
    // that never returns from the kernel can't stall the dump.
    let deadline = Instant::now() + Duration::from_secs(2);
    while capture::DONE.load(Ordering::Acquire) < signalled && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    capture::CAPTURING.store(false, Ordering::Release);

    let captured = capture::NEXT_SLOT
        .load(Ordering::Acquire)
        .min(capture::MAX_THREADS);
    let _ = writeln!(
        out,
        "  {captured}/{signalled} analysis threads captured (registered={})",
        targets.len(),
    );
    for slot in 0..captured {
        let tid = capture::SLOT_TID[slot].load(Ordering::Relaxed);
        let frames = capture::SLOT_NFRAMES[slot]
            .load(Ordering::Acquire)
            .min(capture::MAX_FRAMES);
        let _ = writeln!(out, "  thread tid={tid}:");
        for frame in 0..frames {
            let ip =
                capture::SLOT_FRAMES[slot][frame].load(Ordering::Relaxed) as *mut std::ffi::c_void;
            let mut printed = false;
            // SAFETY: `resolve` only reads symbol tables for the address; off the
            // signal path, so allocation inside the callback is fine.
            unsafe {
                backtrace::resolve(ip, |symbol| {
                    let name = symbol
                        .name()
                        .map_or_else(|| "<unknown>".to_string(), |n| n.to_string());
                    match (symbol.filename(), symbol.lineno()) {
                        (Some(file), Some(line)) => {
                            let _ = writeln!(out, "    {ip:p} {name} ({}:{line})", file.display());
                        }
                        _ => {
                            let _ = writeln!(out, "    {ip:p} {name}");
                        }
                    }
                    printed = true;
                });
            }
            if !printed {
                let _ = writeln!(out, "    {ip:p} <unresolved>");
            }
        }
    }
}
