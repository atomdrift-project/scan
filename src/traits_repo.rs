//! Timeout-bounded wrapper over `cleave::traits_repo` update operations.
//!
//! cleave's own `traits_repo::update` invokes `git` without a timeout, so a
//! hung `ssh` PIN prompt, a dead TCP connection, or an unresponsive server
//! can leave the pull blocked for hours. This module performs the same
//! pull/check via `git_cmd::run` (process-group-killable, non-interactive).
//!
//! Initial clone still delegates to cleave: the clone path runs on first
//! invocation only, and keeping that logic in cleave avoids duplicating
//! the repo URL and resolution rules.

use crate::git_cmd;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Make cleave use an already-installed traits directory instead of attempting
/// an auto-clone during resource initialization.
///
/// This protects first-run litmus startup from a common race/failure mode:
/// cleave sees no explicit traits dir, tries to clone into its default data
/// directory, and git fails because that directory already exists and is
/// non-empty from a prior cleave/litmus invocation.
pub fn prepare_runtime_env() {
    if cleave::traits_repo::override_dir().is_some() {
        return;
    }
    if let Some(value) = std::env::var_os("CLEAVE_TRAITS_DIR")
        && !value.is_empty()
    {
        return;
    }
    if let Some(path) = existing_traits_dir() {
        tracing::debug!(path = %path.display(), "using existing cleave traits directory");
        cleave::traits_repo::set_override_dir(Some(path));
    }
}

fn existing_traits_dir() -> Option<PathBuf> {
    candidate_traits_dirs()
        .into_iter()
        .find(|path| looks_like_traits_dir(path))
}

fn candidate_traits_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![PathBuf::from("traits")];
    dirs.push(default_traits_dir());
    dirs
}

fn default_traits_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("cleave")
        .join("traits")
}

fn looks_like_traits_dir(path: &Path) -> bool {
    path.is_dir()
        && (path.join("objectives").is_dir()
            || path.join("micro-behaviors").is_dir()
            || path.join("metadata").is_dir())
}

/// Pull (or force-reset) the traits repo, bounded by [`git_cmd::NETWORK_TIMEOUT`].
///
/// Falls back to `cleave::traits_repo::update` when no local checkout
/// exists yet so the fresh-install clone path stays in one place.
pub fn update(force: bool) -> Result<()> {
    prepare_runtime_env();
    let Ok(traits_dir) = cleave::traits_repo::try_resolve() else {
        return cleave::traits_repo::update(force)
            .map_err(|e| anyhow::anyhow!("traits update failed: {e}"));
    };

    let path = traits_dir.to_string_lossy().into_owned();
    let before = git_cmd::short_head(&traits_dir);

    if force {
        eprintln!("Force-updating traits from upstream...");
        let fetch = git_cmd::run(&["-C", &path, "fetch", "origin"], git_cmd::NETWORK_TIMEOUT)
            .context("git fetch failed")?;
        if !fetch.status.success() {
            anyhow::bail!("git fetch failed: {}", git_cmd::format_failure(&fetch));
        }
        let reset = git_cmd::run(
            &["-C", &path, "reset", "--hard", "origin/HEAD"],
            git_cmd::NETWORK_TIMEOUT,
        )
        .context("git reset failed")?;
        if !reset.status.success() {
            anyhow::bail!("git reset failed: {}", git_cmd::format_failure(&reset));
        }
    } else {
        eprintln!("Updating traits...");
        let pull = git_cmd::run(
            &["-C", &path, "pull", "--ff-only"],
            git_cmd::NETWORK_TIMEOUT,
        )
            .context("git pull failed")?;
        if !pull.status.success() {
            anyhow::bail!("traits update failed: {}", git_cmd::format_failure(&pull));
        }
    }

    let after = git_cmd::short_head(&traits_dir).unwrap_or_default();
    let before_str = before.as_deref().unwrap_or_default();
    if before_str == after {
        eprintln!("Already up to date ({after}).");
    } else {
        eprintln!("Updated: {before_str} -> {after}");
        if let Ok(diff) = git_cmd::run(
            &[
                "-C",
                &path,
                "diff",
                "--stat",
                &format!("{before_str}..{after}"),
            ],
            git_cmd::NETWORK_TIMEOUT,
        ) && diff.status.success()
        {
            let summary = String::from_utf8_lossy(&diff.stdout);
            if !summary.is_empty() {
                eprint!("{summary}");
            }
        }
    }
    Ok(())
}

/// Check for updates without applying them, bounded by [`git_cmd::NETWORK_TIMEOUT`].
pub fn check_updates() -> Result<()> {
    prepare_runtime_env();
    let traits_dir: PathBuf =
        cleave::traits_repo::try_resolve().map_err(|e| anyhow::anyhow!("{e}"))?;
    let path = traits_dir.to_string_lossy().into_owned();

    let fetch = git_cmd::run(
        &["-C", &path, "fetch", "--dry-run"],
        git_cmd::NETWORK_TIMEOUT,
    )
    .context("git fetch failed")?;

    let local = git_cmd::short_head(&traits_dir).unwrap_or_default();
    let stderr = String::from_utf8_lossy(&fetch.stderr);
    if fetch.status.success() && stderr.trim().is_empty() {
        eprintln!("Traits are up to date ({local}).");
    } else {
        eprintln!("Updates available (current: {local}). Run 'litmus update-rules' to update.");
    }
    Ok(())
}
