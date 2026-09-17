//! One Ctrl-C handler for the whole process.
//!
//! `ctrlc::set_handler` installs exactly once: a second call returns
//! `MultipleHandlers` and installs nothing. Three scan entry points —
//! [`crate::engine::run`], [`crate::engine::run_paths`] and [`crate::ps::run`] —
//! each installed their own handler over their own cancellation flag, and each
//! discarded the error. Whichever ran first won and kept a flag nobody would
//! look at again.
//!
//! `scan sys` runs two of them: a process scan, then a file scan over the
//! persistence and temp locations. Ctrl-C during the file phase reached the
//! *process* phase's flag, which had already finished, so the first interrupt
//! printed "finishing current process…" and cancelled nothing; only a second
//! one stopped the scan, by exiting outright and skipping the graceful path
//! entirely. An embedder calling `Analyzer::scan_path` more than once hit the
//! same thing.
//!
//! So the handler is installed once and owns a registry. Each scan registers
//! its own flag for as long as it runs, which is what keeps one scan's
//! interrupt from leaking into the next.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

/// Scans that may still be cancelled, each with the notice to print when it is.
///
/// Held weakly: a scan deregisters by dropping the flag it was handed, so no
/// guard type has to be threaded through the entry points, and a scan that
/// panics cannot leave a live registration behind.
static SCANS: Mutex<Vec<(Weak<AtomicBool>, &'static str)>> = Mutex::new(Vec::new());

/// Installed on the first [`arm`] call and never replaced.
static HANDLER: OnceLock<()> = OnceLock::new();

/// A cancellation flag for one scan, with Ctrl-C wired to it.
///
/// `notice` is what the user sees on the first interrupt — it names the unit of
/// work the scan will finish before stopping, so it differs per entry point.
///
/// The returned flag is registered for as long as it lives. Callers pass clones
/// into the work (cleave's `AnalysisOptions::cancellation`) and drop them when
/// the scan ends, which deregisters it.
pub(crate) fn arm(notice: &'static str) -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    {
        let mut scans = SCANS.lock().unwrap_or_else(PoisonError::into_inner);
        scans.retain(|(scan, _)| scan.strong_count() > 0);
        scans.push((Arc::downgrade(&flag), notice));
    }
    // `set_handler` fails only for reasons that cannot occur twice — an
    // already-installed handler, or a platform refusal — and `OnceLock` already
    // excludes the first. A refusal leaves the process on the default
    // disposition, which is what it had before this call, so there is nothing
    // to recover: say so once and carry on.
    HANDLER.get_or_init(|| {
        if let Err(e) = ctrlc::set_handler(interrupted) {
            tracing::warn!(
                error = %e,
                "could not install Ctrl-C handler; interrupts will not be graceful",
            );
        }
    });
    flag
}

/// Ask every live scan to stop, and report which ones this interrupt reached.
///
/// Empty means the interrupt cancelled nothing — every flag was already set, or
/// no scan is registered. Separate from [`interrupted`] because that one exits
/// the process, which a test cannot call.
fn cancel_all() -> Vec<&'static str> {
    let mut notices: Vec<&'static str> = Vec::new();
    // Scoped so the registry is unlocked before the caller prints, which takes
    // the progress bar's lock and must never do so holding this one.
    {
        let mut scans = SCANS.lock().unwrap_or_else(PoisonError::into_inner);
        scans.retain(|(scan, _)| scan.strong_count() > 0);
        for (scan, notice) in scans.iter() {
            // `swap` rather than `store`: only the interrupt that actually
            // flips a flag announces itself, so two scans sharing a notice
            // print it once and a flag already set stays quiet.
            if let Some(flag) = scan.upgrade()
                && !flag.swap(true, Ordering::Relaxed)
                && !notices.contains(notice)
            {
                notices.push(notice);
            }
        }
    }
    notices
}

/// Cancel every running scan; on an interrupt that cancels nothing, exit.
///
/// Runs on the `ctrlc` crate's own thread. The first Ctrl-C flips each live
/// flag and returns, leaving the scans to finish the file they are on. A
/// second one flips nothing — every flag is already set — and that is the
/// signal to stop waiting. An interrupt arriving when no scan is registered
/// takes the same exit, since there is no work left to finish gracefully.
fn interrupted() {
    let notices = cancel_all();
    if notices.is_empty() {
        // Cleave runs each rizin in its own process group, so SIGINT on the
        // terminal never reaches them — without an explicit kill here, every
        // in-flight child would outlive us as an orphan.
        cleave::kill_all_rizin_groups();
        std::process::exit(130);
    }
    for notice in notices {
        crate::engine::print_above_bar(|| eprintln!("\n{notice}"));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    // The registry is process-wide by design, and `cargo test` runs these in
    // one process alongside everything else that arms a scan. So each test
    // asserts on the flag it holds -- which no other test can un-set -- rather
    // than on what its own `cancel_all` happened to announce, which a sibling
    // test's interrupt can legitimately have claimed first.

    /// The defect this module exists for: the second scan in a process got no
    /// handler at all, because `set_handler` had already been called and the
    /// error was discarded. Registration, not installation, is what has to
    /// happen per scan.
    #[test]
    fn every_armed_scan_is_cancelled_not_just_the_first() {
        let first = arm("first");
        let second = arm("second");

        cancel_all();

        assert!(
            first.load(Ordering::Relaxed),
            "first scan was not cancelled"
        );
        assert!(
            second.load(Ordering::Relaxed),
            "second scan was not cancelled",
        );
    }

    /// An interrupt that reaches nothing is what tells [`interrupted`] to stop
    /// waiting and exit, so "already cancelled" must report empty rather than
    /// announcing itself a second time.
    #[test]
    fn a_second_interrupt_reaches_nothing() {
        let scan = arm("repeat-probe");

        cancel_all();
        assert!(
            scan.load(Ordering::Relaxed),
            "the first interrupt must cancel the scan",
        );
        assert!(
            !cancel_all().contains(&"repeat-probe"),
            "an already-cancelled scan must not be announced again",
        );
    }

    /// A finished scan must not stay in the registry, or `scan sys` would print
    /// its notice for a Ctrl-C belonging to the phase after it — which is the
    /// symptom the old per-entry-point handlers produced.
    #[test]
    fn a_finished_scan_deregisters_itself() {
        const TAG: &str = "deregister-probe";

        drop(arm(TAG));
        // Arming again prunes; the finished scan must not survive it.
        let live = arm("live-probe");

        let stale = SCANS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(_, notice)| *notice == TAG)
            .count();
        assert_eq!(stale, 0, "a finished scan is still registered");

        cancel_all();
        assert!(
            live.load(Ordering::Relaxed),
            "the live scan was not reached"
        );
    }
}
