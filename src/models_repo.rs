//! Locates the model bundle and installs/updates it from the R2 bundle.
//!
//! Models are distributed as signed `.tar.zst` bundles from the update bucket —
//! see [`crate::model_update`], which `scan update-rules` drives. This module
//! handles *resolution* (where the bundle lives) and the first-run bootstrap
//! install. The bundle root *is* what [`crate::model::Model::load`] reads:
//! `feature_spec.json` + `model.onnx` (or `models/seed_*.onnx`) at the top, or a
//! `general/` ensemble subdirectory. No git.
//!
//! On-disk layout under `dirs::data_dir()/atomdrift/scan/models/<bundle>` where
//! `<bundle>` comes from the last path segment of `SCAN_MODELS_REPO`
//! (`.../azoth.git` → `azoth`), so a per-filetype variant lands at a sibling dir.
//!
//! Resolution order:
//! 1. `SCAN_MODELS_DIR` env var (a ready-to-load bundle directory)
//! 2. The bundle directory derived from the upstream URL (bootstrapped if missing)
//!
//! For development, symlink the bundle path to a local working tree:
//!   ln -sfn ~/dev/atomdrift/azoth ~/.local/share/atomdrift/scan/models/azoth

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const DEFAULT_MODELS_REPO_URL: &str = "https://github.com/atomdrift-project/azoth.git";

/// Files that must be present in a complete bundle. The model file is checked
/// separately because it may live at the top level or under `models/`.
const REQUIRED_ARTIFACTS: &[&str] = &["feature_spec.json"];
/// litmus is ONNX-only — the native LightGBM/XGBoost loaders were retired.
const MODEL_FILES: &[&str] = &["model.onnx"];

/// Resolve the configured upstream URL: `SCAN_MODELS_REPO` or the built-in
/// default. Used only to derive the on-disk bundle directory name. Any `#ref`
/// fragment is dropped — bundles are version-keyed R2 downloads, so a ref no
/// longer selects anything (see [`crate::model_update`]).
fn models_repo_url() -> String {
    let raw =
        std::env::var("SCAN_MODELS_REPO").unwrap_or_else(|_| DEFAULT_MODELS_REPO_URL.to_owned());
    match raw.rsplit_once('#') {
        Some((url, _)) => url.to_owned(),
        None => raw,
    }
}

/// Resolve the models bundle directory, bootstrap-installing if necessary.
///
/// Returns the path suitable for [`crate::model::Model::load`].
pub fn model_dir() -> Result<PathBuf> {
    if let Ok(explicit) = std::env::var("SCAN_MODELS_DIR") {
        let p = PathBuf::from(&explicit);
        if p.is_dir() {
            tracing::debug!("Using models from SCAN_MODELS_DIR={}", p.display());
            return Ok(p);
        }
        anyhow::bail!("SCAN_MODELS_DIR={explicit} does not exist");
    }

    let data_dir = default_models_dir();
    if has_models(&data_dir) {
        tracing::debug!("Using models from {}", data_dir.display());
        return Ok(data_dir);
    }

    eprintln!("First run: downloading Atomdrift Scan models...");
    crate::model_update::update(&data_dir, false, false)
        .with_context(|| format!("failed to install models to {}", data_dir.display()))?;
    Ok(data_dir)
}

/// Get the installed model commit (short), from the bundle's sidecar.
#[must_use]
pub fn version() -> Option<String> {
    crate::model_update::installed(&current_models_dir())
        .map(|i| i.commit.chars().take(12).collect())
}

/// Directory the updater should install into: `SCAN_MODELS_DIR` override or
/// the default bundle path. Public, and doesn't require the dir to exist.
#[must_use]
pub fn install_target() -> PathBuf {
    current_models_dir()
}

/// Feature dimensionality of the installed model: the number of inputs each
/// classifier consumes, read straight from `feature_spec.json`'s
/// `total_features` without loading any ONNX graph. `None` when no model is
/// installed or the spec is unreadable. Used by `scan version`.
#[must_use]
pub fn feature_dimension() -> Option<usize> {
    #[derive(serde::Deserialize)]
    struct Dim {
        total_features: usize,
    }
    let base = current_models_dir();
    let spec = [
        base.join("general").join("feature_spec.json"),
        base.join("feature_spec.json"),
    ]
    .into_iter()
    .find(|p| p.is_file())?;
    let text = std::fs::read_to_string(spec).ok()?;
    serde_json::from_str::<Dim>(&text)
        .ok()
        .map(|d| d.total_features)
}

/// Number of ONNX models in the installed bundle — the ensemble's route models
/// (`general` plus the per-filetype and per-filegroup specialists). `None` when
/// no model is installed. Used by `scan version`.
#[must_use]
pub fn model_count() -> Option<usize> {
    let base = current_models_dir();
    if !base.is_dir() {
        return None;
    }
    let n = walkdir::WalkDir::new(&base)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("onnx"))
        .count();
    (n > 0).then_some(n)
}

/// Default on-disk path for the model bundle: `<data_dir>/atomdrift/scan/models/<bundle>`,
/// where `<bundle>` is the last path segment of the configured upstream URL.
fn default_models_dir() -> PathBuf {
    let url = models_repo_url();
    let bundle = bundle_name_from_url(&url).unwrap_or_else(|| "azoth".to_owned());
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("atomdrift")
        .join("scan")
        .join("models")
        .join(bundle)
}

/// Extract the bundle directory name from a git URL: the last path segment
/// with any trailing `.git` and `/` stripped.
fn bundle_name_from_url(url: &str) -> Option<String> {
    let trimmed = url.trim_end_matches('/');
    let stripped = trimmed.strip_suffix(".git").unwrap_or(trimmed);
    let segment = stripped.rsplit(&['/', ':']).next()?;
    if segment.is_empty() {
        None
    } else {
        Some(segment.to_owned())
    }
}

/// Resolve the models directory currently in use (without bootstrapping).
fn current_models_dir() -> PathBuf {
    if let Ok(explicit) = std::env::var("SCAN_MODELS_DIR") {
        return PathBuf::from(explicit);
    }
    default_models_dir()
}

/// True if the directory looks like a complete bundle litmus can load.
///
/// Two layouts are accepted: an ensemble bundle (`general/` subdirectory
/// containing the required artifacts) or a legacy single-bundle (artifacts
/// at the root). See `model.rs` for the loader-side details.
fn has_models(path: &Path) -> bool {
    let general = path.join("general");
    (general.is_dir() && has_single_bundle_layout(&general)) || has_single_bundle_layout(path)
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
        name.starts_with("seed_") && path.extension().and_then(|ext| ext.to_str()) == Some("onnx")
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn bundle_name_from_url_handles_common_shapes() {
        assert_eq!(
            bundle_name_from_url("https://github.com/atomdrift-project/azoth.git").as_deref(),
            Some("azoth")
        );
        assert_eq!(
            bundle_name_from_url("https://github.com/atomdrift-project/azoth-pe.git/").as_deref(),
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
    fn has_models_accepts_onnx_single_layout() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("model.onnx"), b"").unwrap();
        std::fs::write(tmp.path().join("feature_spec.json"), b"{}").unwrap();
        assert!(has_models(tmp.path()));
    }

    #[test]
    fn has_models_accepts_ensemble_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let general = tmp.path().join("general");
        std::fs::create_dir_all(&general).unwrap();
        std::fs::write(general.join("model.onnx"), b"").unwrap();
        std::fs::write(general.join("feature_spec.json"), b"{}").unwrap();
        assert!(has_models(tmp.path()));
    }

    #[test]
    fn has_models_accepts_multiseed_ensemble_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("general").join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(tmp.path().join("general").join("feature_spec.json"), b"{}").unwrap();
        std::fs::write(models.join("seed_42.onnx"), b"").unwrap();
        assert!(has_models(tmp.path()));
    }

    #[test]
    fn has_models_rejects_ensemble_with_empty_general() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("general")).unwrap();
        assert!(!has_models(tmp.path()));
    }

    #[test]
    fn has_models_rejects_native_model_files() {
        for native in ["model.txt", "model.json"] {
            let tmp = tempfile::tempdir().unwrap();
            std::fs::write(tmp.path().join(native), b"").unwrap();
            std::fs::write(tmp.path().join("feature_spec.json"), b"{}").unwrap();
            assert!(!has_models(tmp.path()), "{native} should be rejected");
        }
    }

    #[test]
    fn has_models_rejects_missing_model_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("feature_spec.json"), b"{}").unwrap();
        assert!(!has_models(tmp.path()));
    }

    #[test]
    fn models_repo_url_resolves_and_drops_fragment() {
        // SAFETY: env-var manipulation in tests; this var is only read here.
        unsafe {
            let saved_repo = std::env::var("SCAN_MODELS_REPO").ok();

            // A `#ref` fragment is stripped so the bundle name stays clean.
            std::env::set_var("SCAN_MODELS_REPO", "https://example.com/foo.git#v3");
            assert_eq!(models_repo_url(), "https://example.com/foo.git");

            // A plain URL passes through unchanged.
            std::env::set_var("SCAN_MODELS_REPO", "https://example.com/bar.git");
            assert_eq!(models_repo_url(), "https://example.com/bar.git");

            // Unset falls back to the built-in default.
            std::env::remove_var("SCAN_MODELS_REPO");
            assert_eq!(models_repo_url(), DEFAULT_MODELS_REPO_URL);

            match saved_repo {
                Some(v) => std::env::set_var("SCAN_MODELS_REPO", v),
                None => std::env::remove_var("SCAN_MODELS_REPO"),
            }
        }
    }

    #[test]
    fn env_var_overrides_default() {
        // SAFETY: set_var is unsound under parallel test execution; SCAN_MODELS_DIR
        // is only read here.
        unsafe {
            let original = std::env::var("SCAN_MODELS_DIR").ok();
            std::env::set_var("SCAN_MODELS_DIR", "/tmp/test-models");
            let result = current_models_dir();
            assert_eq!(result, PathBuf::from("/tmp/test-models"));
            match original {
                Some(v) => std::env::set_var("SCAN_MODELS_DIR", v),
                None => std::env::remove_var("SCAN_MODELS_DIR"),
            }
        }
    }
}
