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
use std::path::PathBuf;
use std::time::Duration;

const GIT_TIMEOUT: Duration = Duration::from_secs(60);

/// Pull (or force-reset) the traits repo, bounded by [`GIT_TIMEOUT`].
///
/// Falls back to `cleave::traits_repo::update` when no local checkout
/// exists yet so the fresh-install clone path stays in one place.
pub fn update(force: bool) -> Result<()> {
    let Ok(traits_dir) = cleave::traits_repo::try_resolve() else {
        return cleave::traits_repo::update(force)
            .map_err(|e| anyhow::anyhow!("traits update failed: {e}"));
    };

    let path = traits_dir.to_string_lossy().into_owned();
    let before = git_cmd::short_head(&traits_dir);

    if force {
        eprintln!("Force-updating traits from upstream...");
        let fetch = git_cmd::run(&["-C", &path, "fetch", "origin"], GIT_TIMEOUT)
            .context("git fetch failed")?;
        if !fetch.status.success() {
            anyhow::bail!(
                "git fetch failed: {}",
                String::from_utf8_lossy(&fetch.stderr)
            );
        }
        let reset = git_cmd::run(
            &["-C", &path, "reset", "--hard", "origin/HEAD"],
            GIT_TIMEOUT,
        )
        .context("git reset failed")?;
        if !reset.status.success() {
            anyhow::bail!(
                "git reset failed: {}",
                String::from_utf8_lossy(&reset.stderr)
            );
        }
    } else {
        eprintln!("Updating traits...");
        let pull = git_cmd::run(&["-C", &path, "pull", "--ff-only"], GIT_TIMEOUT)
            .context("git pull failed")?;
        if !pull.status.success() {
            anyhow::bail!(
                "traits update failed: {}",
                String::from_utf8_lossy(&pull.stderr)
            );
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
    Ok(())
}

/// Check for updates without applying them, bounded by [`GIT_TIMEOUT`].
pub fn check_updates() -> Result<()> {
    let traits_dir: PathBuf =
        cleave::traits_repo::try_resolve().map_err(|e| anyhow::anyhow!("{e}"))?;
    let path = traits_dir.to_string_lossy().into_owned();

    let fetch = git_cmd::run(&["-C", &path, "fetch", "--dry-run"], GIT_TIMEOUT)
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
