//! Thin wrapper over cleave's R2-backed trait updater.
//!
//! cleave traits are distributed as `.tar.zst` bundles from the update bucket
//! (`cleave::rule_update`), verified by sha256 against the bucket's own
//! manifest; that guards against corruption, not a compromised bucket (they
//! are not signed). This module resolves the install dir and delegates, so
//! `scan update-rules` fetches traits the same way `cleave update-rules` does.

use anyhow::Result;
use std::path::Path;

/// Pin cleave to the installed traits tree unless the caller chose one.
///
/// With no explicit directory, cleave prefers a `traits/` in the working
/// directory over the installed tree, so scanning inside an untrusted checkout
/// would let that checkout supply (or hollow out) the rules. `CLEAVE_TRAITS_DIR`
/// still selects a development tree. With nothing installed and no `traits/`
/// here, cleave is left to bootstrap-install into the data directory.
pub fn prepare_runtime_env() {
    let env_set = crate::env::text(crate::env::CLEAVE_TRAITS_DIR).is_some();
    if env_set || cleave::traits_repo::override_dir().is_some() {
        return;
    }
    // `traits-enriched` when this run is Enriched, `traits-oss` otherwise.
    let installed = cleave::traits_repo::installed_dir(cleave::enriched::enabled());
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
///
/// The target is read again afterwards: a refused token without `--enriched`
/// falls the run back to the open-source tree, and it is that tree's commit
/// that then answers.
pub fn update(force: bool, quiet: bool) -> Result<bool> {
    prepare_runtime_env();
    let commit =
        || cleave::rule_update::installed(&cleave::traits_repo::install_target()).map(|i| i.commit);
    let before = commit();
    // Kept as the source, not flattened into text: auto-update tells a refused
    // enriched token (an error) from an unreachable bucket (not one) by it.
    cleave::rule_update::refresh(force, quiet)
        .map_err(|e| anyhow::Error::new(e).context("traits update failed"))?;
    Ok(before != commit())
}

/// Why `--enriched` cannot be honoured, when that is what `error` says: no
/// token, or one the rules channel refused. Without `--enriched` a refusal
/// never gets this far — the run has fallen back to the open-source rules.
/// Every other update failure — the network, the bucket, the disk — is one a
/// host rides out on the rules it has.
#[must_use]
pub fn enriched_refusal(error: &anyhow::Error) -> Option<&str> {
    match error.downcast_ref::<cleave::rule_update::Error>()? {
        cleave::rule_update::Error::Enriched(why) => Some(why),
        cleave::rule_update::Error::Other(_) => None,
    }
}

/// Report whether newer traits are available without applying them.
pub fn check_updates() -> Result<()> {
    prepare_runtime_env();
    let dir = cleave::traits_repo::install_target();
    cleave::rule_update::check(&dir).map_err(anyhow::Error::new)
}
