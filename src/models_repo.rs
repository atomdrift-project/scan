//! Manages the external models repository (clone, update, resolve).
//!
//! Models live in a separate git repository. This module handles locating,
//! auto-cloning, and updating that repository so users don't need to manage
//! it manually.
//!
//! Resolution order:
//! 1. `LITMUS_MODELS_DIR` env var
//! 2. Platform data directory (auto-cloned from upstream)
//!
//! For development, symlink the data directory to a local checkout:
//!   ln -sfn ~/dev/atomdrift/litmus-models \
//!     ~/Library/Application\ Support/litmus/models

use crate::git_cmd;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

const MODELS_REPO_URL: &str = "https://codeberg.org/atomdrift/litmus-models.git";
const CURRENT_MODEL: &str = "scan-v16";
const STALENESS_DAYS: u64 = 30;
const GIT_TIMEOUT: Duration = Duration::from_secs(60);

/// Resolve the models base directory, auto-cloning if necessary.
///
/// Returns the base directory (not the versioned subdirectory).
/// Call `model_dir()` to get the path suitable for `Model::load`.
pub(crate) fn resolve_and_ensure() -> Result<PathBuf> {
    if let Ok(explicit) = std::env::var("LITMUS_MODELS_DIR") {
        let p = PathBuf::from(&explicit);
        if p.is_dir() {
            tracing::debug!("Using models from LITMUS_MODELS_DIR={}", p.display());
            return Ok(p);
        }
        anyhow::bail!("LITMUS_MODELS_DIR={explicit} does not exist");
    }

    let data_dir = default_models_dir();
    if has_models(&data_dir) {
        tracing::debug!("Using models from {}", data_dir.display());
        if let Some(days) = days_since_last_commit(&data_dir) {
            if days > STALENESS_DAYS {
                eprintln!(
                    "Note: Models last updated {days} days ago. Run 'litmus update-rules' to refresh."
                );
            }
        }
        return Ok(data_dir);
    }

    if is_git_repo(&data_dir) {
        let model_path = data_dir.join(CURRENT_MODEL);
        let missing: Vec<String> = MODEL_ARTIFACTS
            .iter()
            .filter(|f| !model_path.join(f).exists())
            .map(|f| format!("{CURRENT_MODEL}/{f}"))
            .collect();
        eprintln!(
            "Models exist at {} but missing: {} — removing in 30s (Ctrl-C to abort)...",
            data_dir.display(),
            missing.join(", "),
        );
        std::thread::sleep(std::time::Duration::from_secs(30));
        std::fs::remove_dir_all(&data_dir).with_context(|| {
            format!(
                "failed to remove incomplete models at {}",
                data_dir.display()
            )
        })?;
    } else {
        eprintln!("First run: downloading litmus models...");
    }
    clone_repo(&data_dir).with_context(|| {
        format!(
            "failed to download models\n\nEnsure 'git' is installed, or manually clone:\n  git clone {MODELS_REPO_URL} \"{}\"",
            data_dir.display()
        )
    })?;
    eprintln!("Models ready ({})", data_dir.join(CURRENT_MODEL).display());
    Ok(data_dir)
}

/// Return the model directory suitable for `Model::load`.
pub fn model_dir() -> Result<PathBuf> {
    resolve_and_ensure().map(|base| base.join(CURRENT_MODEL))
}

/// Update the models repository, cloning it first if necessary.
///
/// Returns the pre-update HEAD commit so the caller can roll back if the new
/// models turn out to be broken.  The returned value is `None` when the repo
/// was freshly cloned (nothing to roll back to) or when HEAD could not be read.
pub fn update() -> Result<Option<String>> {
    let base = current_models_dir();
    if ensure_repo(&base)? {
        return Ok(None);
    }

    let before = git_cmd::short_head(&base);
    eprintln!("Updating models...");

    let base_str = base.to_string_lossy();
    let output = git_cmd::run(&["-C", &base_str, "pull", "--ff-only"], GIT_TIMEOUT)?;

    if !output.status.success() {
        anyhow::bail!("update failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    let after = git_cmd::short_head(&base).unwrap_or_default();
    let before_str = before.as_deref().unwrap_or_default();
    if before_str == after {
        eprintln!("Already up to date ({after}).");
    } else {
        eprintln!("Updated: {before_str} -> {after}");
        if let Ok(diff) = git_cmd::run(
            &[
                "-C",
                &base_str,
                "diff",
                "--stat",
                &format!("{before_str}..{after}"),
            ],
            GIT_TIMEOUT,
        ) {
            if diff.status.success() {
                let summary = String::from_utf8_lossy(&diff.stdout);
                if !summary.is_empty() {
                    eprint!("{summary}");
                }
            }
        }
    }
    Ok(before)
}

/// Roll back the models repository to a previously recorded commit.
///
/// Called when a pulled update fails to load so the disk is returned to the
/// last-known-good state for future reloads.
pub fn rollback(rev: &str) -> Result<()> {
    let base = current_models_dir();
    let broken = git_cmd::short_head(&base).unwrap_or_else(|| "unknown".to_string());
    tracing::error!(broken_commit = %broken, rollback_to = %rev, "rolling back models repo");
    let base_str = base.to_string_lossy();
    let output = git_cmd::run(&["-C", &base_str, "reset", "--hard", rev], GIT_TIMEOUT)?;
    if !output.status.success() {
        anyhow::bail!(
            "rollback to {rev} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    tracing::info!(broken_commit = %broken, rolled_back_to = %rev, "models repo rolled back");
    Ok(())
}

/// Check for updates without applying them.
pub fn check_updates() -> Result<()> {
    let base = current_models_dir();
    if ensure_repo(&base)? {
        return Ok(());
    }

    let base_str = base.to_string_lossy();
    let fetch = git_cmd::run(&["-C", &base_str, "fetch", "--dry-run"], GIT_TIMEOUT)
        .context("git fetch failed")?;

    let local = git_cmd::short_head(&base).unwrap_or_default();
    let stderr = String::from_utf8_lossy(&fetch.stderr);
    if fetch.status.success() && stderr.trim().is_empty() {
        eprintln!("Models are up to date ({local}).");
    } else {
        eprintln!("Updates available (current: {local}). Run 'litmus update-rules' to update.");
    }

    if let Some(days) = days_since_last_commit(&base) {
        eprintln!("Last updated: {days} day(s) ago.");
    }
    Ok(())
}

/// Get the short commit hash of the current models HEAD, if available.
#[must_use]
pub fn version() -> Option<String> {
    git_cmd::short_head(&current_models_dir())
}

fn default_models_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("litmus")
        .join("models")
}

/// Resolve the models directory currently in use (without auto-cloning).
fn current_models_dir() -> PathBuf {
    if let Ok(explicit) = std::env::var("LITMUS_MODELS_DIR") {
        return PathBuf::from(explicit);
    }
    default_models_dir()
}

/// Ensure the models repo exists at `base`, cloning if necessary.
/// Returns `Ok(true)` if a fresh clone was performed.
fn ensure_repo(base: &Path) -> Result<bool> {
    if is_git_repo(base) {
        return Ok(false);
    }
    eprintln!("Models not found — cloning...");
    clone_repo(base).with_context(|| {
        format!(
            "failed to clone models\n\nEnsure 'git' is installed, or manually clone:\n  git clone {MODELS_REPO_URL} \"{}\"",
            base.display()
        )
    })?;
    eprintln!(
        "Models ready ({}).",
        git_cmd::short_head(base).unwrap_or_default()
    );
    Ok(true)
}

const MODEL_ARTIFACTS: &[&str] = &["model.json", "feature_spec.json"];

fn has_models(path: &Path) -> bool {
    let model = path.join(CURRENT_MODEL);
    MODEL_ARTIFACTS.iter().all(|f| model.join(f).exists())
}

fn clone_repo(target: &Path) -> std::io::Result<()> {
    Command::new("git").arg("--version").output().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("git is not installed or not in PATH: {e}"),
        )
    })?;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let output = Command::new("git")
        .args(["clone", "--depth", "1", MODELS_REPO_URL])
        .arg(target)
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    Ok(())
}

fn days_since_last_commit(path: &Path) -> Option<u64> {
    let output = Command::new("git")
        .args(["-C", &path.to_string_lossy(), "log", "-1", "--format=%ct"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let timestamp: i64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs() as i64;
    Some(((now - timestamp) / 86400).max(0).cast_unsigned())
}

fn is_git_repo(path: &Path) -> bool {
    // Follow symlinks: canonicalize then check .git
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    resolved.join(".git").exists()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_models_dir_ends_with_litmus_models() {
        let dir = default_models_dir();
        assert!(dir.ends_with("litmus/models"));
    }

    #[test]
    fn has_models_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!has_models(tmp.path()));
    }

    #[test]
    fn has_models_with_current_model_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join(CURRENT_MODEL);
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("model.json"), b"").unwrap();
        std::fs::write(model.join("feature_spec.json"), b"{}").unwrap();
        assert!(has_models(tmp.path()));
    }

    #[test]
    fn env_var_overrides_default() {
        // SAFETY: set_var is unsound under parallel test execution.
        // This test runs in its own process via `cargo test -- --test-threads=1`
        // or is acceptable because LITMUS_MODELS_DIR is only read here.
        unsafe {
            let original = std::env::var("LITMUS_MODELS_DIR").ok();
            std::env::set_var("LITMUS_MODELS_DIR", "/tmp/test-models");
            let result = current_models_dir();
            assert_eq!(result, PathBuf::from("/tmp/test-models"));
            match original {
                Some(v) => std::env::set_var("LITMUS_MODELS_DIR", v),
                None => std::env::remove_var("LITMUS_MODELS_DIR"),
            }
        }
    }
}
