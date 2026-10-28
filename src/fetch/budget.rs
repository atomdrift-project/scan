//! How much one root, and the whole run, may fetch: counts, bytes and
//! wall-clock time.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// A fetch allowance: live fetches and bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Allowance {
    pub(super) fetches: usize,
    pub(super) bytes: u64,
}

/// The per-execution fetch budget (`--fetch-max-total-*`), shared by every
/// fetch phase in the process. Concurrent scans draw on it, so a fetch reserves
/// its allowance before touching the network and refunds what it did not use:
/// two scans can never both spend the last of it. Left unlimited in long-lived
/// server modes, where each job is bounded by its own per-root budget instead.
#[derive(Debug)]
pub(super) struct TotalBudget {
    pub(super) fetches: AtomicUsize,
    pub(super) bytes: AtomicU64,
}

pub(super) static TOTAL_BUDGET: TotalBudget = TotalBudget::new(usize::MAX, u64::MAX);

impl TotalBudget {
    pub(super) const fn new(fetches: usize, bytes: u64) -> Self {
        Self {
            fetches: AtomicUsize::new(fetches),
            bytes: AtomicU64::new(bytes),
        }
    }

    /// Take up to `want` of each, as much as is left. The grant is the
    /// caller's to spend, and to [`Self::settle`] when done.
    pub(super) fn reserve(&self, want: Allowance) -> Allowance {
        let fetches = self
            .fetches
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                Some(left - left.min(want.fetches))
            })
            .unwrap_or_else(|left| left);
        let bytes = self
            .bytes
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                Some(left - left.min(want.bytes))
            })
            .unwrap_or_else(|left| left);
        Allowance {
            fetches: fetches.min(want.fetches),
            bytes: bytes.min(want.bytes),
        }
    }

    /// Settle a grant against what was spent: the unused part goes back, and
    /// any overshoot is taken too — fletch's byte cap is best-effort, stopping
    /// only after the fetch that crossed it. The count cap is exact.
    pub(super) fn settle(&self, granted: Allowance, spent: Allowance) {
        let _ = self
            .fetches
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                Some(left.saturating_add(granted.fetches.saturating_sub(spent.fetches)))
            });
        let _ = self
            .bytes
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                Some(if spent.bytes <= granted.bytes {
                    left.saturating_add(granted.bytes - spent.bytes)
                } else {
                    left.saturating_sub(spent.bytes - granted.bytes)
                })
            });
    }
}

/// Set the per-execution fetch ceiling; see
/// [`crate::settings::Settings::fetch_total`].
pub(crate) fn set_total_budget(max_fetches: usize, max_bytes: u64) {
    TOTAL_BUDGET.fetches.store(max_fetches, Ordering::Relaxed);
    TOTAL_BUDGET.bytes.store(max_bytes, Ordering::Relaxed);
}

/// What one scanned artifact may still fetch, across all of its hops and
/// declaring files: `--fetch-max-file-fetches` live dependency/package
/// fetches, `--fetch-max-urls` live URL fetches, and `--fetch-max-file-size`
/// bytes. Cache hits are free and uncounted, so a warm re-run is never
/// throttled; references past a cap become `BudgetExceeded`, never silently
/// dropped.
#[derive(Debug, Clone, Copy)]
pub(super) struct RootBudget {
    pub(super) deps: usize,
    pub(super) urls: usize,
    pub(super) bytes: u64,
}

/// The two fetch classes, each with its own count cap: declared dependencies
/// and command-mentioned packages, and opportunistic raw URLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FetchClass {
    Deps,
    Urls,
}

impl RootBudget {
    pub(super) const fn new(policy: &FetchPolicy) -> Self {
        Self {
            deps: policy.max_file_fetches,
            urls: policy.max_url_fetches,
            bytes: policy.max_file_bytes,
        }
    }

    /// What one fetch of `class` may ask the total budget for.
    pub(super) const fn want(&self, class: FetchClass) -> Allowance {
        Allowance {
            fetches: match class {
                FetchClass::Deps => self.deps,
                FetchClass::Urls => self.urls,
            },
            bytes: self.bytes,
        }
    }

    pub(super) fn spend(&mut self, class: FetchClass, spent: Allowance) {
        let fetches = match class {
            FetchClass::Deps => &mut self.deps,
            FetchClass::Urls => &mut self.urls,
        };
        *fetches = fetches.saturating_sub(spent.fetches);
        self.bytes = self.bytes.saturating_sub(spent.bytes);
    }
}

/// Dependency payloads whose analysis has finished, process-wide, whatever the
/// outcome (analyzed, cache hit, corpus skip, nothing to analyze). Postdoc's
/// worker reads this from its stall ticker as a liveness signal: a top-level
/// analysis walking a large transitive closure completes nothing and changes no
/// stage for a long time while making steady progress payload by payload, and
/// without this counter that walk is indistinguishable from a wedged pool.
pub(super) static PAYLOADS_ANALYZED_TOTAL: AtomicU64 = AtomicU64::new(0);

/// How many dependency payloads have finished analysis in this process.
pub fn payloads_analyzed_total() -> u64 {
    PAYLOADS_ANALYZED_TOTAL.load(Ordering::Relaxed)
}

/// Describe the count budget that can clip one fetch class. The per-file cap
/// is normally the limiting value; a smaller process-wide remainder takes
/// precedence so the notice names the budget that actually stopped the work.
pub(super) fn fetch_count_budget_notice(
    per_file_flag: &str,
    per_file_limit: usize,
    total_remaining: usize,
) -> String {
    if total_remaining < per_file_limit {
        format!(
            "Skipping remaining fetches, hit fetch budget (--fetch-max-total-fetches={total_remaining})"
        )
    } else {
        format!("Skipping remaining fetches, hit fetch budget ({per_file_flag}={per_file_limit})")
    }
}

/// The live network work in a batch. Cache hits and budget-clipped edges are
/// free of every budget.
pub(super) fn live_fetch_usage(records: &[FetchRecord]) -> Allowance {
    records
        .iter()
        .filter(|record| record.counts_against_budget())
        .fold(
            Allowance {
                fetches: 0,
                bytes: 0,
            },
            |spent, record| Allowance {
                fetches: spent.fetches + 1,
                bytes: spent.bytes.saturating_add(record.size.unwrap_or(0)),
            },
        )
}

/// The wall-clock ceiling on one artifact's fetch phase (`--fetch-timeout`).
///
/// The count and byte budgets bound how much a scan pulls, not how long pulling
/// takes: a wide tree of slow or rate-limited registries keeps every count
/// budget's room while holding the scan open indefinitely. Checked at group
/// boundaries — a group already fetching finishes, so no download is torn
/// mid-transfer and no partially-analyzed payload reaches the report — and once
/// passed it stays passed, so no later hop restarts the sweep.
#[derive(Debug)]
pub(super) struct Deadline {
    /// `None` for no cap: a zero duration, or one no `Instant` can hold.
    pub(super) at: Option<Instant>,
    pub(super) limit: Duration,
    pub(super) passed: bool,
}

impl Deadline {
    pub(super) fn new(limit: Duration) -> Self {
        Self {
            at: (!limit.is_zero())
                .then(|| Instant::now().checked_add(limit))
                .flatten(),
            limit,
            passed: false,
        }
    }

    /// Whether the cap has passed, said once when it first does.
    pub(super) fn passed(&mut self) -> bool {
        if !self.passed && self.at.is_some_and(|at| Instant::now() >= at) {
            self.passed = true;
            tracing::warn!(
                timeout_secs = self.limit.as_secs(),
                "fetch time cap reached (--fetch-timeout); remaining references not followed"
            );
        }
        self.passed
    }
}
