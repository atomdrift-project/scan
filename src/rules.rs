//! The rules and model a long-lived server runs with, settled once at startup
//! so every binary serving the API resolves them the same way; and the one
//! refresh that keeps every rule source current.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::cli::Refresh;
use crate::model::Thresholds;

/// Where a server's rules and model come from, before resolution.
#[derive(Debug)]
pub struct RulesStartup {
    /// Model bundle. `None` resolves the installed one.
    pub model_dir: Option<PathBuf>,
    /// Pool-adjusted FP level (one FP plus (L-1)/100M above L0). `None` takes the bundle's
    /// own default, then [`crate::model::DEFAULT_SEVERITY_LEVEL`]. Ignored
    /// when `thresholds` is set, which bypasses the level grid entirely.
    pub level: Option<u16>,
    /// Manual probability cutoffs, bypassing the level grid.
    pub thresholds: Option<Thresholds>,
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
        refresh_rules_at_startup(self.refresh)?;
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
    pub(crate) fn changed(&self) -> bool {
        self.0.iter().any(|(_, result)| matches!(result, Ok(true)))
    }

    /// Whether `source` installed something new.
    #[must_use]
    pub(crate) fn changed_source(&self, source: Source) -> bool {
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

    /// Why `--uplink` cannot be honoured, when the traits refresh says so.
    #[must_use]
    pub(crate) fn enriched_refusal(&self) -> Option<&str> {
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
        // Collected so every update is spawned before the first join; a lazy
        // chain would run them one after another.
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
/// [`Refresh::Skip`]; [`Refresh::IfStale`] still fetches, since a daemon only
/// starts on deploy or restart. Every failure but one is ridden out on the installed rules: a
/// host in a disconnected environment must still start.
///
/// This is also what installs the traits on a fresh deploy; without it a
/// server comes up healthy and fails every analysis.
///
/// # Errors
/// The one failure that is fatal: `--uplink` was asked for and the token is
/// missing or refused. A daemon that came up anyway would serve rules nobody
/// deployed it with.
pub fn refresh_rules_at_startup(refresh: Refresh) -> Result<()> {
    if refresh == Refresh::Skip {
        tracing::warn!("--no-update: skipping the startup rules refresh");
        return Ok(());
    }
    let refreshed = self::refresh(&Source::ALL, None, refresh == Refresh::Force, true);
    for (source, e) in refreshed.failures() {
        tracing::warn!(
            source = source.name(),
            error = format!("{e:#}"),
            "startup rules refresh failed"
        );
    }
    match refreshed.enriched_refusal() {
        Some(why) => anyhow::bail!("uplink rules unavailable: {why}"),
        None => Ok(()),
    }
}
