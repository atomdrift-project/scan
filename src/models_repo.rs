//! Locates the model bundle and installs/updates it from the R2 bundle.
//!
//! Models are distributed as `.tar.zst` bundles from the update bucket,
//! verified by sha256 against the bucket's own (unsigned) manifest — see
//! [`crate::model_update`], which `scan update-rules` drives. This module
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

/// Resolve the models bundle directory, bootstrap-installing (a download) if
/// nothing is installed yet.
///
/// Returns the path suitable for [`crate::model::Model::load`].
///
/// # Errors
/// Returns an error if `SCAN_MODELS_DIR` names a missing directory, or the
/// first-run install fails.
pub fn ensure_model_dir() -> Result<PathBuf> {
    if let Ok(explicit) = std::env::var("SCAN_MODELS_DIR") {
        let p = PathBuf::from(&explicit);
        if p.is_dir() {
            tracing::debug!("Using models from SCAN_MODELS_DIR={}", p.display());
            return Ok(p);
        }
        anyhow::bail!("SCAN_MODELS_DIR={explicit} does not exist");
    }

    let data_dir = default_models_dir();
    // `settle`: an update between its swap's two renames is not a first run.
    if has_models(&data_dir) || (crate::model_update::settle(&data_dir) && has_models(&data_dir)) {
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
    crate::model_update::installed(&install_target()).map(|i| i.commit.chars().take(12).collect())
}

/// The models directory in use: the `SCAN_MODELS_DIR` override (which the
/// updater leaves alone) or the default bundle path the updater fills. Doesn't
/// bootstrap or require the dir to exist.
#[must_use]
pub fn install_target() -> PathBuf {
    std::env::var_os("SCAN_MODELS_DIR").map_or_else(default_models_dir, PathBuf::from)
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
    let base = install_target();
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
    let base = install_target();
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
/// where `<bundle>` is named by `SCAN_MODELS_REPO` (see [`bundle_name`]).
fn default_models_dir() -> PathBuf {
    let repo = std::env::var("SCAN_MODELS_REPO");
    let bundle = bundle_name(repo.as_deref().unwrap_or(DEFAULT_MODELS_REPO_URL));
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("atomdrift")
        .join("scan")
        .join("models")
        .join(bundle.unwrap_or("azoth"))
}

/// The bundle directory name for an upstream git URL: its last path segment,
/// without a trailing `.git` or `/`. A `#ref` fragment is dropped — bundles are
/// version-keyed R2 downloads (see [`crate::model_update`]), so a ref selects
/// nothing.
fn bundle_name(url: &str) -> Option<&str> {
    let url = url.split_once('#').map_or(url, |(url, _)| url);
    let trimmed = url.trim_end_matches('/');
    let stripped = trimmed.strip_suffix(".git").unwrap_or(trimmed);
    stripped.rsplit(['/', ':']).next().filter(|s| !s.is_empty())
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
mod tests {
    use super::*;

    #[test]
    fn bundle_name_handles_common_shapes() {
        assert_eq!(
            bundle_name("https://github.com/atomdrift-project/azoth.git"),
            Some("azoth")
        );
        assert_eq!(
            bundle_name("https://github.com/atomdrift-project/azoth-pe.git/"),
            Some("azoth-pe")
        );
        assert_eq!(
            bundle_name("git@codeberg.org:atomdrift/azoth-elf.git"),
            Some("azoth-elf")
        );
        assert_eq!(bundle_name("https://example.com/foo"), Some("foo"));
        assert_eq!(bundle_name("https://example.com/foo.git#v3"), Some("foo"));
        assert_eq!(bundle_name("https://example.com/"), Some("example.com"));
        assert_eq!(bundle_name(""), None);
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
}
