//! Thin wrapper over cleave's R2-backed trait updater.
//!
//! cleave traits are distributed as `.tar.zst` bundles from the update bucket
//! (`cleave::rule_update`), verified by sha256 against the bucket's own
//! manifest; that guards against corruption, not a compromised bucket (they
//! are not signed). This module resolves the install dir and delegates, so
//! `scan update-rules` fetches traits the same way `cleave update-rules` does.

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Pin cleave to the installed traits tree unless the caller chose one.
///
/// With no explicit directory, cleave prefers a `traits/` in the working
/// directory over the installed tree, so scanning inside an untrusted checkout
/// would let that checkout supply (or hollow out) the rules. `CLEAVE_TRAITS_DIR`
/// still selects a development tree. With nothing installed and no `traits/`
/// here, cleave is left to bootstrap-install into the data directory.
pub fn prepare_runtime_env() {
    let env_set = std::env::var_os("CLEAVE_TRAITS_DIR").is_some_and(|v| !v.is_empty());
    if env_set || cleave::traits_repo::override_dir().is_some() {
        return;
    }
    // cleave's own default (`traits_repo::default_traits_dir`, private there).
    let installed = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("atomdrift")
        .join("cleave")
        .join("traits");
    let local = looks_like_traits(Path::new("traits"));
    if !local && !looks_like_traits(&installed) {
        return;
    }
    if local {
        tracing::debug!(
            path = %installed.display(),
            "ignoring traits/ in the working directory; set CLEAVE_TRAITS_DIR to use it"
        );
    }
    cleave::traits_repo::set_override_dir(Some(installed));
}

/// The test cleave applies to a candidate traits directory.
fn looks_like_traits(path: &Path) -> bool {
    path.is_dir()
        && (path.join("objectives").is_dir()
            || path.join("micro-behaviors").is_dir()
            || path.join("metadata").is_dir())
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
