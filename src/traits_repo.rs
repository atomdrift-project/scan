//! Thin wrapper over cleave's R2-backed trait updater.
//!
//! cleave traits are distributed as signed `.tar.zst` bundles from the update
//! bucket (`cleave::rule_update`); this module resolves the install dir and
//! delegates, so `scan update-rules` fetches traits the same way `cleave
//! update-rules` does. No git.

use anyhow::Result;
use std::path::PathBuf;

/// Make cleave use an already-installed traits directory instead of attempting
/// to fetch during resource initialization.
///
/// This protects first-run litmus startup: cleave sees no explicit traits dir
/// and would otherwise try to install into its default data directory during a
/// scan. Point it at an existing tree if one is present.
pub fn prepare_runtime_env() {
    if cleave::traits_repo::override_dir().is_some() {
        return;
    }
    if let Some(value) = std::env::var_os("CLEAVE_TRAITS_DIR")
        && !value.is_empty()
    {
        return;
    }
    let default = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("cleave")
        .join("traits");
    if let Some(path) = [PathBuf::from("traits"), default].into_iter().find(|path| {
        path.is_dir()
            && (path.join("objectives").is_dir()
                || path.join("micro-behaviors").is_dir()
                || path.join("metadata").is_dir())
    }) {
        tracing::debug!(path = %path.display(), "using existing cleave traits directory");
        cleave::traits_repo::set_override_dir(Some(path));
    }
}

/// Install or refresh cleave traits from the R2 bundle, the same path
/// `cleave update-rules` uses. Returns `true` when the installed commit changed.
pub fn update(force: bool, quiet: bool) -> Result<bool> {
    prepare_runtime_env();
    let dir = cleave::traits_repo::install_target();
    let before = cleave::rule_update::installed(&dir).map(|i| i.commit);
    cleave::rule_update::update(&dir, force, quiet)
        .map_err(|e| anyhow::anyhow!("traits update failed: {e}"))?;
    let after = cleave::rule_update::installed(&dir).map(|i| i.commit);
    Ok(before != after)
}

/// Report whether newer traits are available without applying them.
pub fn check_updates() -> Result<()> {
    prepare_runtime_env();
    let dir = cleave::traits_repo::install_target();
    cleave::rule_update::check(&dir).map_err(|e| anyhow::anyhow!("{e}"))
}
