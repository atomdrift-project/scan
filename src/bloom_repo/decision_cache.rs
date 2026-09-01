//! Bounded LRU of bloom membership decisions.
//!
//! Used only by the lookup routes ([`super::Lookup::memo_sha256`] /
//! [`super::Lookup::memo_purl`]). Scan-time probes go straight to the filters
//! so a unique-file crawl cannot evict the lookup working set.
//!
//! One mutex around the map: a hit returns under the lock, a miss computes
//! under the same lock. That is the single-flight — bloom probes are cheap
//! enough that waiting a few microseconds is simpler than a condvar protocol.

use super::Decision;
use lru::LruCache;
use std::borrow::Borrow;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::{Mutex, PoisonError};

/// Entries per lookup kind (SHA-256 and PURL each get their own map).
pub(super) const CAP: usize = 4096;

pub(super) struct DecisionCache<K> {
    lru: Mutex<LruCache<K, Decision>>,
}

impl<K> std::fmt::Debug for DecisionCache<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never take the lock to print: a Debug impl must not be able to block.
        f.debug_struct("DecisionCache").finish_non_exhaustive()
    }
}

impl<K: Eq + Hash> DecisionCache<K> {
    pub(super) fn new(entries: usize) -> Self {
        let cap = NonZeroUsize::new(entries).unwrap_or(NonZeroUsize::MIN);
        Self {
            lru: Mutex::new(LruCache::new(cap)),
        }
    }

    /// Return a cached decision, or run `compute` once for this key.
    ///
    /// `query` is borrowed on the hit path so a PURL lookup does not allocate
    /// on a warm key. The owned `K` is built only on a miss (`ToOwned`).
    pub(super) fn get_or_insert<Q>(&self, query: &Q, compute: impl FnOnce() -> Decision) -> Decision
    where
        K: Borrow<Q>,
        Q: ToOwned<Owned = K> + Hash + Eq + ?Sized,
    {
        let mut lru = self.lru.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(&d) = lru.get(query) {
            return d;
        }
        let d = compute();
        lru.put(query.to_owned(), d);
        d
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::thread;

    #[test]
    fn second_get_does_not_recompute() {
        let cache = DecisionCache::<String>::new(8);
        let computes = AtomicU32::new(0);
        let run = || {
            cache.get_or_insert("pkg:npm/left-pad@1.3.0", || {
                computes.fetch_add(1, Ordering::SeqCst);
                Decision::Skip
            })
        };
        assert_eq!(run(), Decision::Skip);
        assert_eq!(run(), Decision::Skip);
        assert_eq!(computes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn lru_evicts_oldest_past_cap() {
        let cache = DecisionCache::<u8>::new(2);
        let computes = AtomicU32::new(0);
        let run = |k: u8| {
            cache.get_or_insert(&k, || {
                computes.fetch_add(1, Ordering::SeqCst);
                Decision::Unknown
            })
        };
        let _ = run(1);
        let _ = run(2);
        let _ = run(3); // evicts 1
        let after_fill = computes.load(Ordering::SeqCst);
        assert_eq!(after_fill, 3);
        let _ = run(1); // miss — 1 was evicted
        assert_eq!(computes.load(Ordering::SeqCst), after_fill + 1);
        let _ = run(3); // still hot
        assert_eq!(computes.load(Ordering::SeqCst), after_fill + 1);
    }

    #[test]
    fn concurrent_misses_share_one_compute() {
        let cache = DecisionCache::<[u8; 32]>::new(8);
        let digest = [7u8; 32];
        let computes = AtomicU32::new(0);
        let barrier = Barrier::new(8);
        thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    barrier.wait();
                    cache.get_or_insert(&digest, || {
                        computes.fetch_add(1, Ordering::SeqCst);
                        Decision::KnownBad
                    })
                });
            }
        });
        assert_eq!(computes.load(Ordering::SeqCst), 1);
    }
}
