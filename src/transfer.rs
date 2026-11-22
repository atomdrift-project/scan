//! Byte counts of the rules downloads in flight, per bundle, for the one
//! terminal line that reports them.
//!
//! Every bundle download registers here as it starts and counts its bytes as
//! they arrive; nothing here draws. A process that never shows the status line
//! (a server, a redirected run) pays one uncontended lock per body chunk.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// What one bundle has fetched so far, summed over its files.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Tally {
    /// Bytes received.
    pub(crate) done: u64,
    /// Sum of the announced sizes; `None` once any file came unsized, since a
    /// partial total would make the percentage and ETA lie.
    pub(crate) total: Option<u64>,
    /// Files still downloading.
    pub(crate) open: u32,
}

/// Bundles in the order their first download started.
static TALLIES: Mutex<Vec<(&'static str, Tally)>> = Mutex::new(Vec::new());

/// Counters only: a holder that panicked leaves nothing to repair.
fn tallies() -> MutexGuard<'static, Vec<(&'static str, Tally)>> {
    TALLIES.lock().unwrap_or_else(PoisonError::into_inner)
}

fn update(bundle: &'static str, f: impl FnOnce(&mut Tally)) {
    let mut tallies = tallies();
    let index = tallies
        .iter()
        .position(|(name, _)| *name == bundle)
        .unwrap_or_else(|| {
            tallies.push((
                bundle,
                Tally {
                    total: Some(0),
                    ..Tally::default()
                },
            ));
            tallies.len() - 1
        });
    f(&mut tallies[index].1);
}

/// Forget every count, so a new refresh reports only its own downloads.
pub(crate) fn reset() {
    tallies().clear();
}

/// Every bundle that has downloaded anything since the last [`reset`].
pub(crate) fn snapshot() -> Vec<(&'static str, Tally)> {
    tallies().clone()
}

/// One file of `bundle` downloading; counted until dropped, failed or not.
#[derive(Debug)]
pub(crate) struct Transfer {
    bundle: &'static str,
}

impl Transfer {
    /// Start counting a file of `bundle` whose size is `size`, when known.
    pub(crate) fn begin(bundle: &'static str, size: Option<u64>) -> Self {
        update(bundle, |tally| {
            tally.open += 1;
            tally.total = tally.total.zip(size).map(|(sum, size)| sum + size);
        });
        Self { bundle }
    }

    /// Count `bytes` more received.
    pub(crate) fn advance(&self, bytes: u64) {
        update(self.bundle, |tally| tally.done += bytes);
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        update(self.bundle, |tally| {
            tally.open = tally.open.saturating_sub(1)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test owns the process-wide registry, so parallel tests cannot race
    /// on it.
    #[test]
    fn transfers_sum_per_bundle_and_an_unsized_file_drops_the_total() {
        reset();
        let a = Transfer::begin("Bloom filters", Some(100));
        let b = Transfer::begin("Bloom filters", Some(50));
        a.advance(40);
        b.advance(10);
        let tally = snapshot()[0].1;
        assert_eq!(
            tally,
            Tally {
                done: 50,
                total: Some(150),
                open: 2
            }
        );
        drop(a);
        assert_eq!(snapshot()[0].1.open, 1);

        let c = Transfer::begin("Models", None);
        c.advance(7);
        drop((b, c));
        let all = snapshot();
        assert_eq!(all[0].0, "Bloom filters");
        assert_eq!(all[0].1.open, 0);
        assert_eq!(
            all[1],
            (
                "Models",
                Tally {
                    done: 7,
                    total: None,
                    open: 0
                }
            )
        );
        reset();
        assert!(snapshot().is_empty());
    }
}
