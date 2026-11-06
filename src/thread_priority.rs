//! Scheduling priority for the analysis pool's threads.
//!
//! Demoting Rayon workers lets external CPU-bound helpers make progress when
//! the pool saturates the host, while workers still use idle CPU time.

/// Run the calling thread one scheduling notch below where it is now.
///
/// Raising a nice value never needs privilege, and a failure leaves the thread
/// as it was, which is only a missed optimization — so it is logged at debug
/// and otherwise ignored.
///
/// Linux and Windows have a per-thread priority. Elsewhere this is a no-op:
/// FreeBSD's `setpriority` acts on the whole process.
pub(crate) fn demote_current_thread() {
    #[cfg(target_os = "linux")]
    {
        // A tid always fits in id_t; the fallible conversion only keeps the
        // cast lint honest, and on failure the thread stays as it was.
        // SAFETY: `gettid` takes no arguments and cannot fail.
        let Ok(tid) = libc::id_t::try_from(unsafe { libc::syscall(libc::SYS_gettid) }) else {
            return;
        };
        // PRIO_PROCESS with a tid is Linux's per-thread nice. `getpriority`
        // cannot fail for our own tid; its -1 is a nice value, not an error.
        // SAFETY: both calls take scalars and touch no memory.
        let rc = unsafe {
            let current = libc::getpriority(libc::PRIO_PROCESS, tid);
            libc::setpriority(libc::PRIO_PROCESS, tid, (current + 1).min(19))
        };
        if rc != 0 {
            tracing::debug!(
                error = %std::io::Error::last_os_error(),
                "cannot lower pool thread priority",
            );
        }
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut core::ffi::c_void;
            fn SetThreadPriority(thread: *mut core::ffi::c_void, priority: i32) -> i32;
        }
        const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
        // SAFETY: both calls take the pseudo-handle of the calling thread and
        // touch no memory we own.
        if unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) } == 0 {
            tracing::debug!(
                error = %std::io::Error::last_os_error(),
                "cannot lower pool thread priority",
            );
        }
    }
}
