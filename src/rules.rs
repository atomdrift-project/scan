//! The rules, model and concurrency a long-lived server runs with, settled once
//! at startup so every binary serving the API resolves them the same way.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::Result;

use crate::cli::Refresh;
use crate::model::Thresholds;

/// Where a server's rules and model come from, before resolution.
#[derive(Debug)]
pub struct RulesStartup {
    /// Model bundle. `None` resolves the installed one.
    pub model_dir: Option<PathBuf>,
    /// Operating point in false positives per 100M. `None` takes the bundle's
    /// own default, then [`crate::model::DEFAULT_SEVERITY_LEVEL`]. Ignored
    /// when `thresholds` is set, which bypasses the level grid entirely.
    pub level: Option<u16>,
    /// Manual probability cutoffs, bypassing the level grid.
    pub thresholds: Option<Thresholds>,
    /// Traits bundle override (`--traits-dir`).
    pub traits_dir: Option<PathBuf>,
    /// The startup refresh (`-u` / `--no-update`). [`Refresh::Skip`] also
    /// pins the rules for the whole run.
    pub refresh: Refresh,
    /// Per-rule time budget before cleave logs a slow rule.
    pub slow_rule_ms: u64,
    /// The LLM second opinion, when one is configured.
    pub interpret: Option<crate::interpret::InterpretConfig>,
    /// Whether to follow the references a sample declares.
    pub fetch: crate::fetch::FetchPolicy,
    /// Passwords to try against encrypted archives.
    pub zip_passwords: crate::ArchivePasswords,
}

impl RulesStartup {
    /// Point cleave at the traits, refresh rules and models, and settle the
    /// bundle and operating point.
    ///
    /// # Errors
    ///
    /// Returns an error when no model bundle is named and none can be
    /// installed.
    pub fn resolve(self) -> Result<Rules> {
        // The override comes first: otherwise the refresh installs into the
        // default directory while `--traits-dir` points at an empty one, and
        // the server comes up healthy and fails every analysis.
        if let Some(dir) = self.traits_dir.as_ref() {
            cleave::traits_repo::set_override_dir(Some(dir.into()));
        }
        crate::refresh_rules_at_startup(
            self.refresh == Refresh::Force,
            self.refresh == Refresh::Skip,
        )?;
        let model_dir = crate::cli::resolve_model_dir(self.model_dir)?;
        let level = crate::cli::operating_level(self.level, self.thresholds.is_some(), &model_dir);
        Ok(Rules {
            model_dir,
            level,
            thresholds: self.thresholds,
            slow_rule_ms: self.slow_rule_ms,
            interpret: self.interpret,
            fetch: self.fetch,
            zip_passwords: self.zip_passwords,
        })
    }
}

/// The resolved rules and model: everything that must survive every model
/// load unchanged.
#[derive(Debug)]
pub struct Rules {
    /// Model bundle directory.
    pub model_dir: PathBuf,
    /// The operating point that produced the thresholds, or `None` when
    /// manual thresholds were supplied.
    pub level: Option<u16>,
    /// Manual threshold overrides.
    pub thresholds: Option<Thresholds>,
    /// Slow rule warning threshold in ms.
    pub slow_rule_ms: u64,
    /// LLM interpretation; `None` disables the pass.
    pub interpret: Option<crate::interpret::InterpretConfig>,
    /// External-reference fetch policy.
    pub fetch: crate::fetch::FetchPolicy,
    /// Additional passwords to try for encrypted archives.
    pub zip_passwords: crate::ArchivePasswords,
}

/// Default request slots: three per physical core.
///
/// A slot spends much of its life on the network — dependency fetches and the
/// LLM second opinion — so one per core leaves the CPU idle. Memory is bounded
/// by the RSS ceiling, not by the slot count.
#[must_use]
pub fn default_workers() -> NonZeroUsize {
    if let Some(cores) = cleave::memory_tracker::physical_cpu_count() {
        return NonZeroUsize::new(std::cmp::max(2, cores.saturating_mul(3)))
            .unwrap_or(NonZeroUsize::MIN);
    }
    let cores = cleave::memory_tracker::cpu_count().unwrap_or_else(|| {
        tracing::warn!(
            fallback = 4,
            "CPU count detection failed; defaulting slot basis to 4 cores",
        );
        4
    });
    // A logical count is about twice the physical one.
    NonZeroUsize::new(std::cmp::max(2, cores.saturating_mul(3) / 2)).unwrap_or(NonZeroUsize::MIN)
}

/// How many top-level analyses may execute at once: the rayon pool (at least
/// 2), never more than `slots`. `SCAN_CLEAVE_CONCURRENCY` overrides; 0 keeps
/// the default.
///
/// Slots bound the requests in flight; this tighter gate bounds the
/// memory-bandwidth-heavy cleave executions among them, held only around the
/// blocking classify.
pub(crate) fn cleave_concurrency(slots: usize) -> usize {
    let override_value = std::env::var("SCAN_CLEAVE_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    cleave_concurrency_from(slots, rayon::current_num_threads(), override_value)
}

fn cleave_concurrency_from(slots: usize, pool: usize, override_value: Option<usize>) -> usize {
    let slots = slots.max(1);
    override_value.filter(|&value| value > 0).map_or_else(
        || pool.max(1).clamp(2.min(slots), slots),
        |value| value.min(slots),
    )
}

#[cfg(test)]
mod tests {
    use super::cleave_concurrency_from;

    #[test]
    fn cleave_concurrency_scales_with_pool_and_respects_slot_cap() {
        assert_eq!(cleave_concurrency_from(16, 32, None), 16);
        assert_eq!(cleave_concurrency_from(16, 16, None), 16);
        assert_eq!(cleave_concurrency_from(16, 8, None), 8);
        assert_eq!(cleave_concurrency_from(2, 64, None), 2);
        assert_eq!(cleave_concurrency_from(0, 32, None), 1);
    }

    #[test]
    fn cleave_concurrency_override_caps_at_slots() {
        assert_eq!(cleave_concurrency_from(4, 32, Some(64)), 4);
        assert_eq!(cleave_concurrency_from(16, 32, Some(2)), 2);
        assert_eq!(cleave_concurrency_from(16, 32, Some(0)), 16);
    }
}
