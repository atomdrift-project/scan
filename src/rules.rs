//! The rules, model and concurrency a long-lived server runs with, settled once
//! at startup so every binary serving the API resolves them the same way; and
//! the one refresh that keeps every rule source current.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

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
        refresh_rules_at_startup(
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

/// One source of the rules a scan runs with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// cleave's trait definitions.
    Traits,
    /// The classifier bundle.
    Models,
    /// Scan's known-good and known-bad filters.
    Bloom,
    /// The NSRL allowlist.
    Nsrl,
}

impl Source {
    /// Every source, in the order a report lists them.
    pub const ALL: [Self; 4] = [Self::Traits, Self::Models, Self::Bloom, Self::Nsrl];

    /// The name logs and reports use.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Traits => "traits",
            Self::Models => "models",
            Self::Bloom => "bloom",
            Self::Nsrl => "nsrl",
        }
    }

    /// Whether a scan can go on without this source. The filters only let a
    /// scan skip work; without them every file is analyzed.
    #[must_use]
    pub const fn optional(self) -> bool {
        matches!(self, Self::Bloom | Self::Nsrl)
    }

    fn update(self, models_dir: Option<&Path>, force: bool, quiet: bool) -> Result<bool> {
        match self {
            Self::Traits => crate::traits_repo::update(force, quiet),
            Self::Models => match models_dir {
                Some(dir) => crate::model_update::update(dir, force, quiet),
                None => crate::model_update::update(
                    &crate::models_repo::install_target()?,
                    force,
                    quiet,
                ),
            },
            Self::Bloom => {
                crate::bloom_update::update(&crate::bloom_repo::install_dir()?, force, quiet)
            }
            Self::Nsrl => crate::nsrl_repo::update(&crate::nsrl_repo::install_dir()?, force, quiet),
        }
    }
}

/// What one [`refresh`] did: per source, whether it installed anything.
#[derive(Debug)]
pub struct Refreshed(Vec<(Source, Result<bool>)>);

impl Refreshed {
    /// Whether any source installed something new.
    #[must_use]
    pub fn changed(&self) -> bool {
        self.0.iter().any(|(_, result)| matches!(result, Ok(true)))
    }

    /// Whether `source` installed something new.
    #[must_use]
    pub fn changed_source(&self, source: Source) -> bool {
        self.0
            .iter()
            .any(|(s, result)| *s == source && matches!(result, Ok(true)))
    }

    /// The sources that failed to update, with why.
    pub fn failures(&self) -> impl Iterator<Item = (Source, &anyhow::Error)> {
        self.0
            .iter()
            .filter_map(|(source, result)| result.as_ref().err().map(|e| (*source, e)))
    }

    /// Why `--enriched` cannot be honoured, when the traits refresh says so.
    #[must_use]
    pub fn enriched_refusal(&self) -> Option<&str> {
        self.failures()
            .find_map(|(_, e)| crate::traits_repo::enriched_refusal(e))
    }
}

/// Bring `sources` up to date, concurrently: the slow path is one network
/// round-trip, not four, and one source failing never blocks the others.
/// `models_dir` overrides the installed model bundle's directory. `quiet` is
/// a background refresh; see `crate::bundle::update`. What a failure
/// means, and how loudly to say so, is the caller's to decide.
#[must_use]
pub fn refresh(
    sources: &[Source],
    models_dir: Option<&Path>,
    force: bool,
    quiet: bool,
) -> Refreshed {
    let results = std::thread::scope(|s| {
        let running: Vec<_> = sources
            .iter()
            .map(|&source| {
                (
                    source,
                    s.spawn(move || source.update(models_dir, force, quiet)),
                )
            })
            .collect();
        running
            .into_iter()
            .map(|(source, handle)| {
                let result = handle
                    .join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("{} update panicked", source.name())));
                tracing::debug!(
                    source = source.name(),
                    ok = result.is_ok(),
                    "rules refreshed"
                );
                (source, result)
            })
            .collect()
    });
    Refreshed(results)
}

/// Refresh every rule source as a long-lived process starts, unless
/// `no_update`. Every failure but one is ridden out on the installed rules: a
/// host in a disconnected environment must still start.
///
/// This is also what fills a `--traits-dir` naming an empty directory on a
/// fresh deploy; without it a server comes up healthy and fails every analysis.
///
/// # Errors
/// The one failure that is fatal: `--enriched` was asked for and the token is
/// missing or refused. A daemon that came up anyway would serve rules nobody
/// deployed it with.
pub fn refresh_rules_at_startup(force: bool, no_update: bool) -> Result<()> {
    if no_update {
        tracing::warn!("--no-update: skipping the startup rules refresh");
        return Ok(());
    }
    let refreshed = refresh(&Source::ALL, None, force, true);
    for (source, e) in refreshed.failures() {
        tracing::warn!(
            source = source.name(),
            error = format!("{e:#}"),
            "startup rules refresh failed"
        );
    }
    match refreshed.enriched_refusal() {
        Some(why) => anyhow::bail!("enriched rules unavailable: {why}"),
        None => Ok(()),
    }
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
/// 2), never more than `slots`.
///
/// Slots bound the requests in flight; this tighter gate bounds the
/// memory-bandwidth-heavy cleave executions among them, held only around the
/// blocking classify.
pub(crate) fn cleave_concurrency(slots: usize) -> usize {
    cleave_concurrency_from(slots, rayon::current_num_threads())
}

fn cleave_concurrency_from(slots: usize, pool: usize) -> usize {
    let slots = slots.max(1);
    pool.max(1).clamp(2.min(slots), slots)
}

#[cfg(test)]
mod tests {
    use super::cleave_concurrency_from;

    #[test]
    fn cleave_concurrency_scales_with_pool_and_respects_slot_cap() {
        assert_eq!(cleave_concurrency_from(16, 32), 16);
        assert_eq!(cleave_concurrency_from(16, 16), 16);
        assert_eq!(cleave_concurrency_from(16, 8), 8);
        assert_eq!(cleave_concurrency_from(2, 64), 2);
        assert_eq!(cleave_concurrency_from(0, 32), 1);
    }
}
