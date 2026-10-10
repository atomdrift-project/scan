//! Background reclamation of the on-disk caches a scan process populates.
//!
//! scan is the process that most heavily fills these caches, and its long-lived
//! worker/server modes never exit — so a startup-only sweep would never fire
//! again, and cleave only prunes the shared caches on a *directory* scan, not
//! the per-payload analysis a daemon runs. [`start`] therefore picks a one-shot
//! sweep for CLI runs and a recurring loop for daemons.
//!
//! The mechanism is filefacts' `cache_sweep` (best-effort, self-gated to once a
//! day, non-blocking): one detached thread that dies with the process. Each component
//! gets one budget capped at 30 days and 2 GiB (both env-overridable). fletch's
//! blob cache is the exception: fletch sweeps it with its own copy of the same
//! mechanism, since only fletch knows its layout and budget.

use std::time::Duration;

use filefacts::cache_sweep::{self, Budget, Root};

/// Re-sweep interval for daemon modes. The daily marker makes most wakes a
/// single `stat`, so this only bounds how soon a newly-oversized cache is seen.
const DAEMON_SWEEP_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);

/// Start cache reclamation for this process. `daemon` = a serve/worker mode that
/// never exits (sweep must recur); otherwise a one-shot CLI that sweeps once at
/// startup, racing the real work.
pub fn start(daemon: bool) {
    let budgets = budgets();
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

/// Every cache a scan process is responsible for, one budget per component.
fn budgets() -> Vec<Budget> {
    // The string/rizin caches older stng releases filled, aging out now that
    // filefacts caches strings itself; honours a relocated STNG_STRING_CACHE_DIR.
    let mut out = vec![cache_sweep::legacy_stng_budget()];

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
    out.push(Budget {
        label: "scan",
        roots: scan_roots,
        max_age: cache_sweep::max_age_from_env("SCAN_CACHE_TTL_DAYS"),
        max_bytes: cache_sweep::max_bytes_from_env("SCAN_CACHE_MAX_BYTES"),
        max_entries: cache_sweep::max_entries_from_env("SCAN_CACHE_MAX_ENTRIES"),
    });

    out
}
