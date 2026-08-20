//! Background reclamation of the on-disk caches a scan process populates.
//!
//! scan is the process that most heavily fills these caches, and its long-lived
//! worker/server modes never exit — so a startup-only sweep would never fire
//! again, and cleave only prunes the shared caches on a *directory* scan, not
//! the per-payload analysis a daemon runs. [`start`] therefore picks a one-shot
//! sweep for CLI runs and a recurring loop for daemons.
//!
//! The mechanism is stng's `cache_sweep` (best-effort, self-gated to once a day,
//! non-blocking): one detached thread that dies with the process. Each component
//! gets one budget capped at 30 days and 2 GiB (both env-overridable).

use std::time::Duration;

use stng::{Budget, Root};

/// Re-sweep interval for daemon modes. The daily marker makes most wakes a
/// single `stat`, so this only bounds how soon a newly-oversized cache is seen.
const DAEMON_SWEEP_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);

/// Start cache reclamation for this process. `daemon` = a serve/worker mode that
/// never exits (sweep must recur); otherwise a one-shot CLI that sweeps once at
/// startup, racing the real work.
pub fn start(daemon: bool) {
    let budgets = budgets();
    if daemon {
        stng::spawn_periodic(budgets, DAEMON_SWEEP_INTERVAL);
    } else {
        stng::spawn(budgets);
    }
}

/// Every cache a scan process is responsible for, one budget per component.
fn budgets() -> Vec<Budget> {
    // stng's string/r2 caches — via stng's own helpers, so a relocated
    // STNG_STRING_CACHE_DIR is swept correctly (no path drift).
    let mut out = vec![stng::stng_budget()];

    // scan's own caches: the analysis snapshot store
    // (`analysis/<version>/<sha>.zst`, depth 2), the lookup verdict index
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
        max_age: stng::cache_sweep::max_age_from_env("SCAN_CACHE_TTL_DAYS"),
        max_bytes: stng::cache_sweep::max_bytes_from_env("SCAN_CACHE_MAX_BYTES"),
    });

    // fletch's blob cache. scan is the fetcher's main driver, but links fletch at
    // a pinned rev without `fetch::refs_dir`, so the path is mirrored here
    // (`…/fletch/refs`). refs/ has no env override, so nothing drifts. Fetched
    // artifacts are large, so this tier gets a 10 GiB default (matching fletch's
    // own `refs_budget`), overridable via `FLETCH_CACHE_MAX_BYTES`.
    if let Some(base) = dirs::cache_dir() {
        out.push(Budget {
            label: "fletch",
            roots: vec![Root {
                path: base.join("fletch").join("refs"),
                depth: 1,
            }],
            max_age: stng::cache_sweep::max_age_from_env("FLETCH_CACHE_TTL_DAYS"),
            max_bytes: stng::cache_sweep::max_bytes_from_env_or(
                "FLETCH_CACHE_MAX_BYTES",
                FLETCH_CACHE_MAX_BYTES_DEFAULT,
            ),
        });
    }

    out
}

/// Default ceiling for fletch's blob cache — 10 GiB, matching fletch's own
/// `cache_sweep::FLETCH_DEFAULT_MAX_BYTES`. Kept in step by hand because scan
/// links fletch at a rev that doesn't expose it.
const FLETCH_CACHE_MAX_BYTES_DEFAULT: u64 = 10 * 1024 * 1024 * 1024;
