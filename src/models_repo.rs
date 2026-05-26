//! Manages the external models repository (clone, update, resolve).
//!
//! Models live in a separate git repository whose checkout root *is* the
//! model bundle: `feature_spec.json`, `config.json`, `evaluation.json`, and
//! either `model.json` (XGBoost) or `model.txt` (LightGBM) sit at the top
//! level. The model "version" is the git ref the checkout points at, not a
//! subdirectory name.
//!
//! On-disk layout under `dirs::data_dir()/litmus/models/`:
//!
//! ```text
//! models/
//!   azoth/        ← default bundle, all file types
//!   azoth-pe/     ← (future) PE-only specialist
//!   azoth-elf/    ← (future) ELF-only specialist
//!   …
//! ```
//!
//! The directory name comes from the last path segment of the configured
//! upstream URL (`.../azoth.git` → `azoth`), so swapping `LITMUS_MODELS_REPO`
//! to a per-filetype variant gives it a sibling directory automatically.
//!
//! Resolution order:
//! 1. `LITMUS_MODELS_DIR` env var (points at a ready-to-load bundle directory)
//! 2. The bundle directory derived from the upstream URL, auto-cloned/updated
//!
//! Configurable via env (or CLI flags that overwrite these vars at startup):
//! - `LITMUS_MODELS_REPO`: upstream URL (default: codeberg.org/atomdrift/azoth.git).
//!   May embed a ref via the `URL#ref` syntax (`...azoth.git#v1`); the
//!   trailing fragment overrides `LITMUS_MODELS_REF`.
//! - `LITMUS_MODELS_REF`:  branch or tag to track (default: the remote's
//!   default branch). If the configured ref is missing on the remote we warn
//!   and fall back to whatever branch `HEAD` points at.
//!
//! For development, symlink the bundle path to a local working tree:
//!   ln -sfn ~/dev/atomdrift/azoth ~/.local/share/litmus/models/azoth

use crate::git_cmd;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_MODELS_REPO_URL: &str = "https://codeberg.org/atomdrift/azoth.git";
const STALENESS_DAYS: u64 = 30;

/// Sentinel used internally when the user pinned no ref. The clone path
/// translates this to "omit `--branch`" and the fetch path to "fetch
/// whatever the remote's HEAD points at".
const REMOTE_DEFAULT_REF: &str = "";

/// Files that must be present in a complete bundle. The model file is checked
/// separately because we accept either `model.json` or `model.txt`.
const REQUIRED_ARTIFACTS: &[&str] = &["feature_spec.json"];
const MODEL_FILES: &[&str] = &["model.json", "model.txt"];

/// Resolve the configured upstream URL and ref.
///
/// Resolution order:
/// 1. `LITMUS_MODELS_REPO=URL#ref` — the trailing fragment, if present, is the ref.
/// 2. `LITMUS_MODELS_REF=...` — used only if no `#ref` is embedded in the URL.
/// 3. Built-in default URL plus `REMOTE_DEFAULT_REF` (= "track the remote's
///    default branch").
fn configured_repo() -> (String, String) {
    let raw =
        std::env::var("LITMUS_MODELS_REPO").unwrap_or_else(|_| DEFAULT_MODELS_REPO_URL.to_owned());
    let (url, embedded_ref) = match raw.rsplit_once('#') {
        Some((url, ref_name)) => (url.to_owned(), Some(ref_name.to_owned())),
        None => (raw, None),
    };
    let ref_name = embedded_ref
        .or_else(|| std::env::var("LITMUS_MODELS_REF").ok())
        .unwrap_or_else(|| REMOTE_DEFAULT_REF.to_owned());
    (url, ref_name)
}

/// Resolve the models bundle directory, auto-cloning if necessary.
///
/// Returns the path suitable for [`crate::model::Model::load`].
pub fn model_dir() -> Result<PathBuf> {
    if let Ok(explicit) = std::env::var("LITMUS_MODELS_DIR") {
        let p = PathBuf::from(&explicit);
        if p.is_dir() {
            tracing::debug!("Using models from LITMUS_MODELS_DIR={}", p.display());
            return Ok(p);
        }
        anyhow::bail!("LITMUS_MODELS_DIR={explicit} does not exist");
    }

    let data_dir = default_models_dir();
    let (repo_url, ref_name) = configured_repo();

    if is_git_repo(&data_dir)
        && let Some(existing) = current_remote_url(&data_dir)
        && !urls_match(&existing, &repo_url)
    {
        anyhow::bail!(
            "configured models repo ({repo_url}) differs from on-disk repo ({existing}) at {}; \
             run 'litmus update-rules' or set LITMUS_MODELS_REPO/LITMUS_MODELS_DIR explicitly",
            data_dir.display()
        );
    }

    if has_models(&data_dir) {
        tracing::debug!("Using models from {}", data_dir.display());
        if let Some(days) = days_since_last_commit(&data_dir)
            && days > STALENESS_DAYS
        {
            eprintln!(
                "Note: Models last updated {days} days ago. Run 'litmus update-rules' to refresh."
            );
        }
        return Ok(data_dir);
    }

    if is_git_repo(&data_dir) {
        let missing = missing_artifacts(&data_dir);
        anyhow::bail!(
            "models exist at {} but are incomplete: missing {}. \
             Run 'litmus update-rules' to refresh, or set LITMUS_MODELS_DIR to a complete bundle.",
            data_dir.display(),
            missing.join(", ")
        );
    } else {
        eprintln!("First run: downloading litmus models...");
    }
    clone_repo(&data_dir, &repo_url, &ref_name).with_context(|| {
        format!(
            "failed to download models\n\nEnsure 'git' is installed, or manually clone:\n  git clone {repo_url} \"{}\"",
            data_dir.display()
        )
    })?;
    eprintln!("Models ready ({})", data_dir.display());
    Ok(data_dir)
}

/// Update the models repository, cloning it first if necessary.
///
/// Returns the pre-update HEAD commit so the caller can roll back if the new
/// models turn out to be broken. The returned value is `None` when the repo
/// was freshly cloned (nothing to roll back to) or when HEAD could not be read.
pub fn update() -> Result<Option<String>> {
    let base = current_models_dir();
    let (repo_url, ref_name) = configured_repo();

    if ensure_repo(&base, &repo_url, &ref_name)? {
        return Ok(None);
    }

    let before = git_cmd::short_head(&base);
    let effective_ref = resolve_ref(&repo_url, &ref_name)?;
    eprintln!(
        "Updating models (ref={display})...",
        display = display_ref(&effective_ref)
    );

    let base_str = base.to_string_lossy();
    let fetch_args: Vec<&str> = if effective_ref.is_empty() {
        vec!["-C", &base_str, "fetch", "origin"]
    } else {
        vec!["-C", &base_str, "fetch", "origin", &effective_ref]
    };
    let fetch = git_cmd::run(&fetch_args, git_cmd::NETWORK_TIMEOUT)?;
    if !fetch.status.success() {
        anyhow::bail!("fetch failed: {}", git_cmd::format_failure(&fetch));
    }

    // Force the working tree to the fetched ref. We use `reset --hard
    // FETCH_HEAD` rather than `checkout`/`pull --ff-only` so the ref-name can
    // refer to either a branch or a tag (or even an arbitrary commit) without
    // needing a different command per case, and so a manually edited bundle
    // gets discarded the same way a half-applied pull would have been.
    let reset = git_cmd::run(
        &["-C", &base_str, "reset", "--hard", "FETCH_HEAD"],
        git_cmd::NETWORK_TIMEOUT,
    )?;
    if !reset.status.success() {
        anyhow::bail!(
            "reset to {} failed: {}",
            display_ref(&effective_ref),
            git_cmd::format_failure(&reset)
        );
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
            git_cmd::NETWORK_TIMEOUT,
        ) && diff.status.success()
        {
            let summary = String::from_utf8_lossy(&diff.stdout);
            if !summary.is_empty() {
                eprint!("{summary}");
            }
        }
    }
    Ok(before)
}

/// Roll back the models repository to a previously recorded commit.
pub fn rollback(rev: &str) -> Result<()> {
    let base = current_models_dir();
    let broken = git_cmd::short_head(&base).unwrap_or_else(|| "unknown".to_string());
    tracing::error!(broken_commit = %broken, rollback_to = %rev, "rolling back models repo");
    let base_str = base.to_string_lossy();
    let output = git_cmd::run(
        &["-C", &base_str, "reset", "--hard", rev],
        git_cmd::NETWORK_TIMEOUT,
    )?;
    if !output.status.success() {
        anyhow::bail!(
            "rollback to {rev} failed: {}",
            git_cmd::format_failure(&output)
        );
    }
    tracing::info!(broken_commit = %broken, rolled_back_to = %rev, "models repo rolled back");
    Ok(())
}

/// Check for updates without applying them.
pub fn check_updates() -> Result<()> {
    let base = current_models_dir();
    let (repo_url, ref_name) = configured_repo();
    if ensure_repo(&base, &repo_url, &ref_name)? {
        return Ok(());
    }

    let effective_ref = resolve_ref(&repo_url, &ref_name)?;
    let base_str = base.to_string_lossy();
    let mut args: Vec<&str> = vec!["-C", &base_str, "fetch", "--dry-run", "origin"];
    if !effective_ref.is_empty() {
        args.push(&effective_ref);
    }
    let fetch = git_cmd::run(&args, git_cmd::NETWORK_TIMEOUT).context("git fetch failed")?;

    let local = git_cmd::short_head(&base).unwrap_or_default();
    let stderr = String::from_utf8_lossy(&fetch.stderr);
    let display = display_ref(&effective_ref);
    if fetch.status.success() && stderr.trim().is_empty() {
        eprintln!("Models are up to date ({local}, ref={display}).");
    } else {
        eprintln!("Updates available (current: {local}, ref={display}). Run 'litmus update-rules' to update.");
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

/// Default on-disk path for the model bundle.
///
/// Layout: `<data_dir>/litmus/models/<bundle>` where `<bundle>` is derived
/// from the last path segment of the configured upstream URL. This means a
/// per-filetype variant (e.g. `…/azoth-pe.git`) lands at a sibling directory
/// (`…/models/azoth-pe`) without any further configuration.
///
/// On a dev machine you can `ln -sfn /path/to/working/tree
/// ~/.local/share/litmus/models/azoth` to share one tree between the
/// trainer and the scanner without round-tripping through git.
fn default_models_dir() -> PathBuf {
    let (url, _) = configured_repo();
    let bundle = bundle_name_from_url(&url).unwrap_or_else(|| "azoth".to_owned());
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("litmus")
        .join("models")
        .join(bundle)
}

/// Extract the bundle directory name from a git URL: the last path segment
/// with any trailing `.git` and `/` stripped.
fn bundle_name_from_url(url: &str) -> Option<String> {
    let stripped = url
        .trim_end_matches('/')
        .strip_suffix(".git")
        .unwrap_or_else(|| {
            // Fall back to the unstripped trim if there's no `.git` suffix.
            url.trim_end_matches('/')
        });
    let segment = stripped.rsplit(&['/', ':']).next()?;
    if segment.is_empty() {
        None
    } else {
        Some(segment.to_owned())
    }
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
fn ensure_repo(base: &Path, repo_url: &str, ref_name: &str) -> Result<bool> {
    if is_git_repo(base) {
        return Ok(false);
    }
    eprintln!("Models not found — cloning {repo_url}...");
    clone_repo(base, repo_url, ref_name).with_context(|| {
        format!(
            "failed to clone models\n\nEnsure 'git' is installed, or manually clone:\n  git clone {repo_url} \"{}\"",
            base.display()
        )
    })?;
    eprintln!(
        "Models ready ({}).",
        git_cmd::short_head(base).unwrap_or_default()
    );
    Ok(true)
}

/// Find which required artifacts are missing from the bundle. Returned for
/// human-readable warnings on incomplete checkouts.
fn missing_artifacts(path: &Path) -> Vec<String> {
    if has_ensemble_layout(path) {
        return Vec::new();
    }
    let mut missing: Vec<String> = REQUIRED_ARTIFACTS
        .iter()
        .filter(|f| !path.join(f).exists())
        .map(|f| (*f).to_owned())
        .collect();
    if !MODEL_FILES.iter().any(|f| path.join(f).exists()) {
        missing.push(format!("one of [{}]", MODEL_FILES.join(", ")));
    }
    missing
}

/// True if the directory looks like a complete bundle litmus can load.
///
/// Two layouts are accepted: an ensemble bundle (`general/` subdirectory
/// containing the required artifacts) or a legacy single-bundle (artifacts
/// at the root). See `model.rs` for the loader-side details.
fn has_models(path: &Path) -> bool {
    has_ensemble_layout(path) || has_single_bundle_layout(path)
}

fn has_single_bundle_layout(path: &Path) -> bool {
    REQUIRED_ARTIFACTS.iter().all(|f| path.join(f).exists()) && has_model_artifact(path)
}

fn has_model_artifact(path: &Path) -> bool {
    if MODEL_FILES.iter().any(|f| path.join(f).exists()) {
        return true;
    }
    let models = path.join("models");
    let Ok(entries) = std::fs::read_dir(models) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        let path = entry.path();
        if !path.is_file() {
            return false;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            return false;
        };
        name.starts_with("seed_")
            && matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("json" | "txt")
            )
    })
}

fn has_ensemble_layout(path: &Path) -> bool {
    let general = path.join("general");
    general.is_dir() && has_single_bundle_layout(&general)
}

fn clone_repo(target: &Path, repo_url: &str, ref_name: &str) -> Result<()> {
    Command::new("git")
        .arg("--version")
        .output()
        .map_err(|e| anyhow::anyhow!("git is not installed or not in PATH: {e}"))?;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let effective_ref = resolve_ref(repo_url, ref_name)?;
    let target_str = target.to_string_lossy();
    let mut args = vec!["clone", "--depth", "1"];
    if !effective_ref.is_empty() {
        args.extend(["--branch", &effective_ref]);
    }
    args.push(repo_url);
    args.push(&target_str);
    let output = git_cmd::run(&args, git_cmd::NETWORK_TIMEOUT).context("git clone failed")?;
    if !output.status.success() {
        anyhow::bail!("git clone failed: {}", git_cmd::format_failure(&output));
    }
    Ok(())
}

/// Pre-flight ref check: if the user pinned a ref that the remote doesn't
/// expose, warn once and treat it as "default branch". Returning the empty
/// string signals callers to omit `--branch` / fetch the remote's HEAD.
fn resolve_ref(repo_url: &str, ref_name: &str) -> Result<String> {
    if ref_name.is_empty() {
        return Ok(String::new());
    }
    let output = git_cmd::run(
        &["ls-remote", "--exit-code", repo_url, ref_name],
        git_cmd::NETWORK_TIMEOUT,
    )
    .context("checking remote models ref")?;
    let exists = output.status.success();
    if exists {
        Ok(ref_name.to_owned())
    } else {
        eprintln!(
            "Note: ref {ref_name:?} not found on {repo_url}; using the remote's default branch."
        );
        Ok(String::new())
    }
}

/// Render a ref for human-readable log lines: empty → `<default>`.
fn display_ref(ref_name: &str) -> &str {
    if ref_name.is_empty() {
        "<default>"
    } else {
        ref_name
    }
}

fn current_remote_url(path: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["-C", &path.to_string_lossy(), "remote", "get-url", "origin"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Compare two git URLs that point at the same repository. Normalizes:
///   - trailing `/` and `.git`
///   - protocol prefix (`ssh://`, `https://`, `http://`, `git://`)
///   - `user@` prefix on ssh URLs
///   - `git@host:path` ↔ `ssh://git@host/path` (the `:` host/path separator
///     is rewritten to `/` so both forms compare equal)
///
/// Anything left over after this is "host + path", which is what actually
/// identifies the repo.
fn urls_match(a: &str, b: &str) -> bool {
    fn normalize(s: &str) -> String {
        let s = s.trim().trim_end_matches('/');
        let s = s.strip_suffix(".git").unwrap_or(s);
        let s = s
            .strip_prefix("ssh://")
            .or_else(|| s.strip_prefix("https://"))
            .or_else(|| s.strip_prefix("http://"))
            .or_else(|| s.strip_prefix("git://"))
            .unwrap_or(s);
        let s = s.split_once('@').map_or(s, |(_, after)| after);
        // `git@host:path` uses ':' as the host/path separator.
        s.replacen(':', "/", 1)
    }
    normalize(a) == normalize(b)
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

    // The path-derivation logic is covered by `bundle_name_from_url_handles_common_shapes`.
    // A `default_models_dir()` test would also be desirable but it depends on
    // ambient `LITMUS_MODELS_REPO` / `LITMUS_MODELS_REF` env vars, which other
    // tests in this module mutate concurrently — leading to a flake under the
    // default parallel test runner. Not worth a serial-test dep just for this.

    #[test]
    fn bundle_name_from_url_handles_common_shapes() {
        assert_eq!(
            bundle_name_from_url("https://codeberg.org/atomdrift/azoth.git").as_deref(),
            Some("azoth")
        );
        assert_eq!(
            bundle_name_from_url("https://codeberg.org/atomdrift/azoth-pe.git/").as_deref(),
            Some("azoth-pe")
        );
        assert_eq!(
            bundle_name_from_url("git@codeberg.org:atomdrift/azoth-elf.git").as_deref(),
            Some("azoth-elf")
        );
        assert_eq!(
            bundle_name_from_url("https://example.com/foo").as_deref(),
            Some("foo")
        );
    }

    #[test]
    fn has_models_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!has_models(tmp.path()));
    }

    #[test]
    fn has_models_accepts_xgboost_layout() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("model.json"), b"").unwrap();
        std::fs::write(tmp.path().join("feature_spec.json"), b"{}").unwrap();
        assert!(has_models(tmp.path()));
    }

    #[test]
    fn has_models_accepts_ensemble_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let general = tmp.path().join("general");
        std::fs::create_dir_all(&general).unwrap();
        std::fs::write(general.join("model.txt"), b"").unwrap();
        std::fs::write(general.join("feature_spec.json"), b"{}").unwrap();
        // Top-level config.json is optional for the resolver's "is this a
        // complete bundle?" check; the loader will read it if present.
        assert!(has_models(tmp.path()));
    }

    #[test]
    fn has_models_accepts_multiseed_ensemble_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("general").join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(tmp.path().join("general").join("feature_spec.json"), b"{}").unwrap();
        std::fs::write(models.join("seed_42.txt"), b"").unwrap();
        assert!(has_models(tmp.path()));
    }

    #[test]
    fn has_models_rejects_ensemble_with_empty_general() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("general")).unwrap();
        // general/ exists but contains no artifacts → not a valid bundle.
        assert!(!has_models(tmp.path()));
    }

    #[test]
    fn has_models_accepts_lightgbm_layout() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("model.txt"), b"").unwrap();
        std::fs::write(tmp.path().join("feature_spec.json"), b"{}").unwrap();
        assert!(has_models(tmp.path()));
    }

    #[test]
    fn has_models_rejects_missing_model_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("feature_spec.json"), b"{}").unwrap();
        assert!(!has_models(tmp.path()));
    }

    #[test]
    fn urls_match_normalizes_dot_git_and_slash() {
        assert!(urls_match(
            "https://codeberg.org/atomdrift/azoth.git",
            "https://codeberg.org/atomdrift/azoth"
        ));
        assert!(urls_match(
            "https://codeberg.org/atomdrift/azoth/",
            "https://codeberg.org/atomdrift/azoth.git"
        ));
        assert!(!urls_match(
            "https://codeberg.org/atomdrift/azoth.git",
            "https://codeberg.org/atomdrift/litmus-models.git"
        ));
    }

    #[test]
    fn urls_match_treats_ssh_and_https_as_equivalent() {
        // The on-disk remote is often ssh:// (push-friendly) while the
        // canonical default we ship is https:// (no key required). These
        // forms point at the same repo and must compare equal so the
        // url-mismatch reclone path doesn't fire spuriously.
        assert!(urls_match(
            "https://codeberg.org/atomdrift/azoth.git",
            "ssh://git@codeberg.org/atomdrift/azoth.git",
        ));
        assert!(urls_match(
            "https://codeberg.org/atomdrift/azoth.git",
            "git@codeberg.org:atomdrift/azoth.git",
        ));
        assert!(!urls_match(
            "https://codeberg.org/atomdrift/azoth.git",
            "git@codeberg.org:atomdrift/azoth-pe.git",
        ));
    }

    #[test]
    fn configured_repo_splits_url_fragment() {
        // SAFETY: env-var manipulation in tests; these vars are only read here.
        unsafe {
            let saved_repo = std::env::var("LITMUS_MODELS_REPO").ok();
            let saved_ref = std::env::var("LITMUS_MODELS_REF").ok();

            // URL#ref takes precedence over the separate REF env var.
            std::env::set_var("LITMUS_MODELS_REPO", "https://example.com/foo.git#v3");
            std::env::set_var("LITMUS_MODELS_REF", "v9");
            let (url, r) = configured_repo();
            assert_eq!(url, "https://example.com/foo.git");
            assert_eq!(r, "v3");

            // No fragment → REF env var wins.
            std::env::set_var("LITMUS_MODELS_REPO", "https://example.com/bar.git");
            std::env::set_var("LITMUS_MODELS_REF", "v9");
            let (url, r) = configured_repo();
            assert_eq!(url, "https://example.com/bar.git");
            assert_eq!(r, "v9");

            // Neither set → empty (= remote default branch).
            std::env::remove_var("LITMUS_MODELS_REPO");
            std::env::remove_var("LITMUS_MODELS_REF");
            let (_, r) = configured_repo();
            assert_eq!(r, "");

            // Restore.
            match saved_repo {
                Some(v) => std::env::set_var("LITMUS_MODELS_REPO", v),
                None => std::env::remove_var("LITMUS_MODELS_REPO"),
            }
            match saved_ref {
                Some(v) => std::env::set_var("LITMUS_MODELS_REF", v),
                None => std::env::remove_var("LITMUS_MODELS_REF"),
            }
        }
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
