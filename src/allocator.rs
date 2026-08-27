//! Return freed-but-retained allocator pages to the OS.
//!
//! Both allocators this binary links keep freed pages mapped so the next
//! allocation is cheap. That is the right default while work is flowing and the
//! wrong one for a worker whose intake is gated on its own RSS: the admission
//! gate reads live process memory ([`crate::admission`]), so pages the allocator
//! is merely holding count against the ceiling exactly as if they were live, and
//! a worker can throttle itself on memory nothing is using.
//!
//! The two platforms need different amounts of help:
//!
//! - **unix (jemalloc)** — configured via `cleave::JEMALLOC_CONF` with
//!   `dirty_decay_ms:1000,retain:false,background_thread:true`. A background
//!   thread purges on its own schedule whether or not the process is doing
//!   anything, so there is nothing to drive from here.
//! - **windows (mimalloc)** — no equivalent. mimalloc has no background purge
//!   thread, and its purge is driven by allocator activity, so a worker that
//!   finishes a batch of large samples and then goes idle keeps every retained
//!   segment for as long as it stays idle. Measured on a 16-slot worker: 12.6 GB
//!   working set / 16.4 GB private bytes held flat across 40 minutes with zero
//!   active slots, against a 26 GB admission ceiling.
//!
//! So this module is a real operation on Windows and a documented no-op
//! elsewhere. Call it when the worker goes idle or after dropping large caches —
//! not on a hot path.

/// Return retained allocator pages to the OS. Blocking and potentially slow;
/// call from `spawn_blocking`, never from an async context and never from a
/// rayon worker — it broadcasts across the rayon pool, so calling it from inside
/// that pool would have a thread wait on itself.
///
/// No-op unless this is a Windows build using mimalloc.
#[cfg(all(windows, not(feature = "crt-heap")))]
pub fn trim() {
    // `mi_collect(true)` does two things (mimalloc v3 `theap.c`): it collects
    // the *calling thread's* heap, and it then runs `_mi_arenas_collect` with a
    // forced purge, which is program-wide. The arena purge is what actually
    // hands pages back, but it can only hand back what has already drained out
    // of the per-thread heaps — and on this worker the bulk of the memory was
    // allocated on the rayon analysis threads, not here.
    //
    // Hence the broadcast: run the per-thread half on every rayon worker so
    // their free pages reach the arenas, then a final call for this thread.
    //
    // `broadcast` waits for every rayon worker to pick up the closure, so this
    // blocks while the pool is busy. That is why callers gate on an idle worker;
    // on a genuinely wedged rayon thread (which the summary ticker detects and
    // reports separately) this parks one blocking thread rather than spinning.
    rayon::broadcast(|_| unsafe { libmimalloc_sys::mi_collect(true) });
    // SAFETY: `mi_collect` takes no pointers and is safe to call from any
    // thread at any time; it is `unsafe` only by virtue of being an `extern "C"`
    // declaration.
    unsafe { libmimalloc_sys::mi_collect(true) };
}

/// Return retained allocator pages to the OS. See the Windows definition; on
/// unix jemalloc's `background_thread` already does this on its own schedule,
/// and on a `crt-heap` build there is no knob to reach for.
#[cfg(not(all(windows, not(feature = "crt-heap"))))]
pub fn trim() {}

#[cfg(test)]
mod tests {
    /// The `#[global_allocator]` this crate ships lives in `src/main.rs`, so it
    /// applies to `atomscan` and not to the lib test harness — which would leave
    /// the test below measuring the system allocator's retention rather than
    /// mimalloc's, and asserting nothing about the code it covers. Install the
    /// same allocator here so the test exercises the real one.
    ///
    /// Scoped to `cfg(test)` on the lib target, so it cannot collide with the
    /// binary's own declaration; integration tests in `tests/` link the lib as a
    /// normal dependency and are unaffected.
    #[cfg(all(windows, not(feature = "crt-heap")))]
    #[global_allocator]
    static TEST_GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

    /// Proves the mechanism this module exists for: memory freed on the rayon
    /// analysis threads stays resident until [`super::trim`] asks for it back.
    ///
    /// `#[ignore]` because it asserts on process RSS — it needs a couple of GB
    /// of headroom and a machine that is not under memory pressure, neither of
    /// which a shared CI runner guarantees. Run it by hand when touching this
    /// module:
    ///
    /// ```text
    /// cargo test -p atomdrift-scan --release allocator::tests -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "asserts on process RSS; run manually, see the doc comment"]
    fn trim_returns_rayon_thread_allocations_to_the_os() {
        const CHUNK: usize = 64 * 1024 * 1024;
        const CHUNKS_PER_THREAD: usize = 4;

        let rss = || cleave::memory_tracker::current_rss().unwrap_or(0) / 1024 / 1024;
        let threads = rayon::current_num_threads();
        let baseline = rss();

        // Allocate on the rayon pool, touching every page so it is really
        // resident, then drop it all inside the same broadcast.
        rayon::broadcast(|_| {
            let mut held: Vec<Vec<u8>> = Vec::new();
            for _ in 0..CHUNKS_PER_THREAD {
                held.push(vec![0xA5u8; CHUNK]);
            }
            std::hint::black_box(&held);
        });
        let after_free = rss();

        super::trim();
        let after_trim = rss();

        println!(
            "threads={threads} baseline={baseline}MB after_free={after_free}MB \
             after_trim={after_trim}MB"
        );

        // The point of the test: freeing is not returning. If the allocator had
        // already handed everything back on drop there would be nothing here to
        // fix, and this module would be dead weight.
        let retained = after_free.saturating_sub(baseline);
        let reclaimed = after_free.saturating_sub(after_trim);
        println!("retained_after_free={retained}MB reclaimed_by_trim={reclaimed}MB");

        #[cfg(all(windows, not(feature = "crt-heap")))]
        {
            assert!(
                retained > 256,
                "expected the freed rayon allocations to stay resident \
                 (baseline={baseline}MB after_free={after_free}MB); \
                 if this fails the allocator changed its retention policy",
            );
            assert!(
                reclaimed * 2 > retained,
                "trim should return most of what was retained, got \
                 {reclaimed}MB of {retained}MB",
            );
        }
    }
}
