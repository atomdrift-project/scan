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

use std::path::{Path, PathBuf};
use std::process::Command;

const MODELS_REPO_URL: &str = "https://codeberg.org/atomdrift/litmus-models.git";
const CURRENT_VERSION: &str = "v1";
const DEFAULT_MODEL: &str = "default";
const STALENESS_DAYS: u64 = 30;

/// Resolve the models base directory, auto-cloning if necessary.
///
/// Returns the base directory (not the versioned subdirectory).
/// Call `model_dir()` to get the path suitable for `Model::load`.
#[must_use]
pub fn resolve_and_ensure() -> PathBuf {
    if let Ok(explicit) = std::env::var("LITMUS_MODELS_DIR") {
        let p = PathBuf::from(&explicit);
        if p.is_dir() {
            tracing::debug!("Using models from LITMUS_MODELS_DIR={}", p.display());
            return p;
        }
        eprintln!("Error: LITMUS_MODELS_DIR={explicit} does not exist");
        std::process::exit(1);
    }

    let data_dir = default_models_dir();
    if has_models(&data_dir) {
        tracing::debug!("Using models from {}", data_dir.display());
        check_staleness(&data_dir);
        return data_dir;
    }

    eprintln!("First run: downloading litmus models...");
    if let Err(e) = clone_repo(&data_dir) {
        eprintln!("Error: Failed to download models: {e}");
        eprintln!();
        eprintln!("Ensure 'git' is installed, or manually clone:");
        eprintln!("  git clone {MODELS_REPO_URL} \"{}\"", data_dir.display());
        std::process::exit(1);
    }
    eprintln!("Models ready. Continuing...");
    data_dir
}

/// Return the model directory suitable for `Model::load`.
#[must_use]
pub fn model_dir() -> PathBuf {
    resolve_and_ensure().join(CURRENT_VERSION).join(DEFAULT_MODEL)
}

/// Update the models repository.
pub fn update() -> Result<(), String> {
    let base = current_models_dir();
    if !is_git_repo(&base) {
        return Err(format!(
            "Models directory is not a git repository: {}\n\
             (if using a symlink, ensure the target is a git repo)",
            base.display()
        ));
    }

    let before = short_head(&base).unwrap_or_default();
    eprintln!("Updating models...");

    let output = Command::new("git")
        .args(["-C", &base.to_string_lossy(), "pull", "--ff-only"])
        .output()
        .map_err(|e| format!("git pull failed: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "Update failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let after = short_head(&base).unwrap_or_default();
    if before == after {
        eprintln!("Already up to date ({after}).");
    } else {
        eprintln!("Updated: {before} -> {after}");
        if let Ok(diff) = Command::new("git")
            .args([
                "-C",
                &base.to_string_lossy(),
                "diff",
                "--stat",
                &format!("{before}..{after}"),
            ])
            .output()
        {
            if diff.status.success() {
                let summary = String::from_utf8_lossy(&diff.stdout);
                if !summary.is_empty() {
                    eprint!("{summary}");
                }
            }
        }
    }
    Ok(())
}

/// Check for updates without applying them.
pub fn check_updates() -> Result<(), String> {
    let base = current_models_dir();
    if !is_git_repo(&base) {
        return Err(format!(
            "Models directory is not a git repository: {}",
            base.display()
        ));
    }

    let fetch = Command::new("git")
        .args(["-C", &base.to_string_lossy(), "fetch", "--dry-run"])
        .output()
        .map_err(|e| format!("git fetch failed: {e}"))?;

    let local = short_head(&base).unwrap_or_default();
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
    short_head(&current_models_dir())
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

fn has_models(path: &Path) -> bool {
    let model = path.join(CURRENT_VERSION).join(DEFAULT_MODEL);
    model.join("model.onnx").exists() && model.join("feature_spec.json").exists()
}

fn clone_repo(target: &Path) -> std::io::Result<()> {
    check_git_available()?;
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

fn check_staleness(path: &Path) {
    if let Some(days) = days_since_last_commit(path) {
        if days > STALENESS_DAYS {
            eprintln!(
                "Note: Models last updated {days} days ago. Run 'litmus update-rules' to refresh."
            );
        }
    }
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
    Some(((now - timestamp) / 86400).max(0) as u64)
}

fn short_head(path: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["-C", &path.to_string_lossy(), "rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

fn is_git_repo(path: &Path) -> bool {
    // Follow symlinks: canonicalize then check .git
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    resolved.join(".git").exists()
}

fn check_git_available() -> std::io::Result<()> {
    Command::new("git")
        .arg("--version")
        .output()
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "git is not installed or not in PATH",
            )
        })?;
    Ok(())
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
    fn has_models_with_v1_default_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("v1").join("default");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("model.onnx"), b"").unwrap();
        std::fs::write(model.join("feature_spec.json"), b"{}").unwrap();
        assert!(has_models(tmp.path()));
    }

    #[test]
    fn env_var_overrides_default() {
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
