//! Crash-time dump of the analyses in flight when the process aborts.
//!
//! A stack overflow deep inside cleave's work-stealing analysis aborts the
//! whole worker process (the Rust runtime prints `has overflowed its stack`
//! then `abort()`s → `SIGABRT`). The per-analysis lifecycle logs name every
//! file that *started*, but not which of the in-flight set was on a CPU when
//! the stack blew — so an operator is left diffing "analysis starting" against
//! "analysis complete" across thousands of lines to guess the culprit.
//!
//! This keeps a small, lock-free registry of the analyses currently running on
//! a worker thread. From an async-signal-safe `SIGABRT` handler it writes that
//! set straight to stderr, so the abort log names the suspects directly. It is
//! a diagnostic backstop: the real fix for any given overflow is to bound the
//! recursion that caused it (see filefacts' `MAX_AST_DEPTH`). Because the
//! shared rayon pool lets one thread steal another analysis's work, the
//! offending file is one of the in-flight set, not necessarily the last to
//! start — dumping them all is the honest answer.

pub use imp::{Guard, install, register};

#[cfg(unix)]
mod imp {
    use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};

    /// Maximum in-flight analyses we can record at once. Worker slots sit far
    /// below this; the slack absorbs any transient over-subscription, and an
    /// overflow past it merely drops a line from the dump (best-effort).
    const SLOTS: usize = 512;
    /// Bytes reserved per slot for the preformatted description line. Long
    /// paths are truncated to fit; the head of the path still identifies it.
    const LINE: usize = 256;

    /// Sentinel id marking a slot that is claimed but whose line is not yet
    /// written. The handler skips it so it never prints a half-filled buffer.
    /// Analysis ids start at 1 and increment, so `u64::MAX` never collides.
    const CLAIMING: u64 = u64::MAX;

    struct Slot {
        /// `0` = free, [`CLAIMING`] = being filled, anything else = the live
        /// analysis id (published last, with `Release`, so a reader that sees a
        /// real id also sees a fully written `line`/`len`).
        id: AtomicU64,
        len: AtomicUsize,
        line: [AtomicU8; LINE],
    }

    impl Slot {
        const fn new() -> Self {
            Slot {
                id: AtomicU64::new(0),
                len: AtomicUsize::new(0),
                line: [const { AtomicU8::new(0) }; LINE],
            }
        }
    }

    static REGISTRY: [Slot; SLOTS] = [const { Slot::new() }; SLOTS];

    /// Frees its slot when the analysis finishes (or its thread unwinds). A
    /// stack overflow `abort()`s without unwinding, so the entry is still live
    /// when the handler runs — which is exactly what we want to report.
    #[derive(Debug)]
    pub struct Guard {
        slot: Option<usize>,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            if let Some(idx) = self.slot {
                REGISTRY[idx].id.store(0, Ordering::Release);
            }
        }
    }

    /// Record an analysis as in flight. The line is formatted here, on the
    /// normal worker thread; the signal handler only ever copies bytes and
    /// calls `write(2)`, both async-signal-safe.
    pub fn register(analysis_id: u64, thread_id: u64, sha: &str, file: &str) -> Guard {
        let mut buf = [0u8; LINE];
        let n = format_line(&mut buf, analysis_id, thread_id, sha, file);

        for (idx, slot) in REGISTRY.iter().enumerate() {
            // Claim with the sentinel first so a concurrent dump never sees a
            // real id before the line bytes are in place.
            if slot
                .id
                .compare_exchange(0, CLAIMING, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                for (i, &b) in buf[..n].iter().enumerate() {
                    slot.line[i].store(b, Ordering::Relaxed);
                }
                slot.len.store(n, Ordering::Release);
                slot.id.store(analysis_id, Ordering::Release);
                return Guard { slot: Some(idx) };
            }
        }
        // Registry full: skip silently. The dump simply omits this analysis.
        Guard { slot: None }
    }

    /// Append the description into `buf`, truncating to its capacity. No
    /// allocation: a fixed cursor over a stack array.
    fn format_line(buf: &mut [u8; LINE], id: u64, tid: u64, sha: &str, file: &str) -> usize {
        let mut pos = 0;
        let mut put = |bytes: &[u8]| {
            let take = bytes.len().min(buf.len().saturating_sub(pos));
            buf[pos..pos + take].copy_from_slice(&bytes[..take]);
            pos += take;
        };
        put(b"  [inflight] analysis_id=");
        put(itoa(id).as_slice());
        put(b" thread_id=");
        put(itoa(tid).as_slice());
        put(b" sha=");
        put(sha.as_bytes());
        put(b" file=");
        put(file.as_bytes());
        put(b"\n");
        pos
    }

    /// Decimal-format a `u64` into a small inline buffer (max 20 digits).
    /// Returns a `(buf, len)` pair as a tiny stack type so callers can slice it.
    struct Itoa {
        buf: [u8; 20],
        start: usize,
    }
    impl Itoa {
        fn as_slice(&self) -> &[u8] {
            &self.buf[self.start..]
        }
    }
    fn itoa(mut v: u64) -> Itoa {
        let mut buf = [0u8; 20];
        let mut i = buf.len();
        loop {
            i -= 1;
            buf[i] = b'0' + (v % 10) as u8;
            v /= 10;
            if v == 0 {
                break;
            }
        }
        Itoa { buf, start: i }
    }

    /// Install the `SIGABRT` handler. Idempotent enough to call once at start.
    ///
    /// We hook only `SIGABRT`, not `SIGSEGV`: the Rust runtime owns the
    /// `SIGSEGV` handler that detects the overflow and prints the
    /// `has overflowed its stack` line, then calls `abort()`. Catching the
    /// resulting `SIGABRT` adds our dump *after* that message while preserving
    /// it, and also covers panics-as-abort and OOM aborts.
    pub fn install() {
        // SAFETY: a zeroed `sigaction` is a valid empty disposition; we set the
        // handler, an empty mask, and run on the alternate signal stack the
        // Rust runtime already established (the faulting thread's own stack is
        // exhausted). `SA_RESETHAND` restores the default after we run so the
        // closing `raise` actually aborts (and can still dump core).
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_abort as *const () as usize;
            libc::sigemptyset(&mut sa.sa_mask);
            sa.sa_flags = libc::SA_ONSTACK | libc::SA_RESETHAND;
            libc::sigaction(libc::SIGABRT, &sa, std::ptr::null_mut());
        }
    }

    extern "C" fn on_abort(_sig: libc::c_int) {
        // Async-signal-safe only: atomic loads, a stack copy, and `write(2)`.
        write_all(b"\n--- scan aborting: in-flight analyses (one is the likely culprit) ---\n");
        for slot in REGISTRY.iter() {
            let id = slot.id.load(Ordering::Acquire);
            if id == 0 || id == CLAIMING {
                continue;
            }
            let len = slot.len.load(Ordering::Acquire).min(LINE);
            // Copy out via atomic loads rather than aliasing the atomics as a
            // byte slice — keeps the read race-free and strictly defined.
            let mut local = [0u8; LINE];
            for (i, dst) in local[..len].iter_mut().enumerate() {
                *dst = slot.line[i].load(Ordering::Relaxed);
            }
            write_all(&local[..len]);
        }
        write_all(b"--- end in-flight dump ---\n");

        // SA_RESETHAND has restored SIG_DFL; re-raise to abort for real.
        unsafe {
            libc::raise(libc::SIGABRT);
        }
    }

    /// `write(2)` the whole buffer to stderr, retrying short writes. Ignores
    /// errors — there is nothing useful to do with them mid-abort.
    fn write_all(mut bytes: &[u8]) {
        while !bytes.is_empty() {
            let n = unsafe { libc::write(libc::STDERR_FILENO, bytes.as_ptr().cast(), bytes.len()) };
            if n <= 0 {
                break;
            }
            bytes = &bytes[n.cast_unsigned()..];
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn itoa_formats_decimal() {
            assert_eq!(itoa(0).as_slice(), b"0");
            assert_eq!(itoa(1).as_slice(), b"1");
            assert_eq!(itoa(497).as_slice(), b"497");
            assert_eq!(itoa(u64::MAX).as_slice(), b"18446744073709551615");
        }

        #[test]
        fn format_line_is_complete_and_bounded() {
            let mut buf = [0u8; LINE];
            let n = format_line(&mut buf, 497, 19_272_535, "5a5ddec16368", "evil.whl");
            assert_eq!(
                &buf[..n],
                b"  [inflight] analysis_id=497 thread_id=19272535 sha=5a5ddec16368 file=evil.whl\n"
            );
        }

        #[test]
        fn format_line_truncates_an_overlong_path() {
            let long = "a/".repeat(LINE); // far longer than the buffer
            let mut buf = [0u8; LINE];
            let n = format_line(&mut buf, 1, 2, "deadbeef", &long);
            assert_eq!(n, LINE, "truncates to capacity, never overflows");
        }

        #[test]
        fn register_occupies_then_frees_a_slot() {
            // A distinctive id we can find in the registry regardless of other
            // entries left by concurrent tests.
            let id = 0xA11CE_u64;
            let occupied = || REGISTRY.iter().any(|s| s.id.load(Ordering::Acquire) == id);
            assert!(!occupied(), "precondition: id not already present");
            {
                let _g = register(id, 7, "cafebabe", "f.bin");
                assert!(occupied(), "register must mark a slot live");
            }
            assert!(!occupied(), "dropping the guard must free the slot");
        }
    }
}

#[cfg(not(unix))]
mod imp {
    /// No-op guard on non-unix targets; the registry and handler are unix-only.
    #[derive(Debug)]
    pub struct Guard;
    /// Record an analysis as in flight. No-op here: there is no crash-dump
    /// registry or signal handler to feed on non-unix targets.
    #[must_use]
    pub fn register(_id: u64, _thread_id: u64, _sha: &str, _file: &str) -> Guard {
        Guard
    }
    /// Install the crash-dump handler. No-op here: non-unix targets have
    /// no `SIGABRT` to hook.
    pub fn install() {}
}
