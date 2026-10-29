//! Background reclamation of the on-disk caches a scan process populates.
//!
//! scan is the process that most heavily fills these caches, and its long-lived
//! server mode never exits — so a startup-only sweep would never fire
//! again, and cleave only prunes the shared caches on a *directory* scan, not
//! the per-payload analysis a daemon runs. [`start`] therefore picks a one-shot
//! sweep for CLI runs and a recurring loop for daemons.
//!
//! The mechanism is scan's `cache_sweep` module (best-effort, self-gated to once a
//! day, non-blocking): one detached thread that dies with the process. scan's
//! caches share one budget capped at 30 days and 2 GiB. fletch's blob cache and
//! filefacts' own cache are swept by those crates, since only they know their
//! layout and budget.

use std::time::Duration;

use crate::cache_sweep::{self, Budget, Root};

/// Re-sweep interval for daemon modes. The daily marker makes most wakes a
/// single `stat`, so this only bounds how soon a newly-oversized cache is seen.
const DAEMON_SWEEP_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);

/// Start cache reclamation for this process. `daemon` = a long-running mode that
/// never exits (sweep must recur); otherwise a one-shot CLI that sweeps once at
/// startup, racing the real work.
pub fn start(daemon: bool) {
    let budgets = vec![budget()];
    // fletch's blob cache: sharded, a blob and its sidecar evicted together,
    // flat leftovers of an older layout reclaimed, `FLETCH_CACHE_*` overrides.
    let refs = vec![fletch::cache_sweep::refs_budget()];
    if daemon {
        cache_sweep::spawn_periodic(budgets, DAEMON_SWEEP_INTERVAL);
        fletch::cache_sweep::spawn_periodic(refs, DAEMON_SWEEP_INTERVAL);
    } else {
        cache_sweep::spawn(budgets);
        fletch::cache_sweep::spawn(refs);
    }
}

/// The caches scan owns, under one budget.
fn budget() -> Budget {
    // scan's own caches: the analysis snapshot store
    // (`analysis/<version>/<key>.zst`, depth 2), the lookup verdict index
    // (`lookup/<version>/<sha>.json` and its `<key>.purl` aliases, also depth
    // 2) and the LLM verdict cache (`interpret/<hash>.json`, depth 1), sharing
    // one ceiling.
    let mut scan_roots = Vec::new();
    if let Some(path) = crate::analysis_cache::cache_base() {
        scan_roots.push(Root { path, depth: 2 });
    }
    if let Some(path) = crate::lookup::index_base() {
        scan_roots.push(Root { path, depth: 2 });
    }
    if let Some(path) = crate::interpret::cache_base() {
        scan_roots.push(Root { path, depth: 1 });
    }
    Budget {
        label: "scan",
        roots: scan_roots,
        max_age: Duration::from_secs(30 * 24 * 60 * 60),
        max_bytes: 2 * 1024 * 1024 * 1024,
        max_entries: filefacts::cache::DEFAULT_MAX_ITEMS,
    }
}
