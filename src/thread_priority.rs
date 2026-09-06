//! Scheduling priority for the threads a pool runs on.
//!
//! Two rayon pools share every core of a serve host: the global pool the
//! server's own analyses fan out on, and the idle pull worker's pool. Core
//! budgets decide how many threads each may run; they do not decide who wins
//! a core when both want it. The kernel scheduler does, and it favours
//! whichever thread it is told to. Demoting the pull worker's threads is what
//! lets its budget be generous: when an interactive analysis becomes
//! runnable it takes the core, and the pull work resumes in the gaps.

/// Lower the calling thread's scheduling priority by `steps` notches below
/// where it is now. Never needs privilege; a failure leaves the thread as it
/// was.
///
/// Linux: `setpriority` on the thread id, which is per-thread nice there.
/// FreeBSD: the idle scheduling class via `rtprio_thread`, which is the
/// per-thread control that platform offers (`setpriority` there is for the
/// whole process). Windows: `SetThreadPriority` one class down. Elsewhere a
/// no-op.
pub fn demote_current_thread(steps: u8) {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: PRIO_PROCESS with a tid is Linux's per-thread nice; both
        // calls touch no memory and a non-zero return is ignored by design.
        unsafe {
            let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
            let current = libc::getpriority(libc::PRIO_PROCESS, tid);
            libc::setpriority(
                libc::PRIO_PROCESS,
                tid,
                (current + i32::from(steps)).min(19),
            );
        }
    }
    #[cfg(target_os = "freebsd")]
    {
        // FreeBSD has no per-thread nice. Its idle class runs only when no
        // timeshare thread wants the CPU, which is exactly the relationship
        // pull work should have to interactive work. `prio` orders idle
        // threads among themselves, 0 highest through 31; `steps` maps onto
        // that so a caller asking for "more demoted" still gets it.
        // SAFETY: rtprio_thread writes nothing we own and validates its
        // arguments; a failure returns non-zero and the thread stays as it was.
        unsafe {
            let mut rtp = libc::rtprio {
                type_: libc::RTP_PRIO_IDLE,
                prio: u16::from(steps).min(31),
            };
            let lwpid = libc::pthread_getthreadid_np() as libc::lwpid_t;
            libc::rtprio_thread(libc::RTP_SET, lwpid, &raw mut rtp);
        }
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut core::ffi::c_void;
            fn SetThreadPriority(thread: *mut core::ffi::c_void, priority: i32) -> i32;
        }
        const THREAD_PRIORITY_LOWEST: i32 = -2;
        const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
        let priority = if steps > 1 {
            THREAD_PRIORITY_LOWEST
        } else {
            THREAD_PRIORITY_BELOW_NORMAL
        };
        // SAFETY: both calls take the pseudo-handle of the calling thread and
        // touch no memory we own.
        unsafe {
            SetThreadPriority(GetCurrentThread(), priority);
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "freebsd", windows)))]
    {
        let _ = steps;
    }
}
