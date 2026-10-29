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
//! On-disk layout: `dirs::data_dir()/atomdrift/scan/models/azoth`.
//!
//! Resolution order:
//! 1. `SCAN_MODELS_DIR` env var (a ready-to-load bundle directory)
//! 2. The default bundle directory (bootstrapped if missing)
//!
//! For development, symlink the bundle path to a local working tree:
//!   ln -sfn ~/dev/atomdrift/azoth ~/.local/share/atomdrift/scan/models/azoth

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Files that must be present in a complete bundle. The model file is checked
/// separately because it may live at the top level or under `models/`.
const REQUIRED_ARTIFACTS: &[&str] = &["feature_spec.json"];
/// Scan loads ONNX models only; there is no native LightGBM/XGBoost loader.
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
    if let Some(explicit) = crate::env::path(crate::env::SCAN_MODELS_DIR) {
        if explicit.is_dir() {
            tracing::debug!("Using models from SCAN_MODELS_DIR={}", explicit.display());
            return Ok(explicit);
        }
        anyhow::bail!("SCAN_MODELS_DIR={} does not exist", explicit.display());
    }

    let data_dir = default_models_dir()?;
    // `settle`: an update between its swap's two renames is not a first run.
    if has_models(&data_dir) || (crate::bundle::settle(&data_dir) && has_models(&data_dir)) {
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
    crate::model_update::installed(&install_target().ok()?)
        .map(|i| i.commit.chars().take(12).collect())
}

/// The models directory in use: the `SCAN_MODELS_DIR` override (which the
/// updater leaves alone) or the default bundle path the updater fills. Doesn't
/// bootstrap or require the dir to exist.
///
/// # Errors
/// When there is no override and no data directory; see [`crate::data_dir`].
pub fn install_target() -> Result<PathBuf> {
    crate::env::path(crate::env::SCAN_MODELS_DIR).map_or_else(default_models_dir, Ok)
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
    let base = install_target().ok()?;
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
    let base = install_target().ok()?;
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

/// Default on-disk path for the model bundle: `<data_dir>/atomdrift/scan/models/azoth`.
fn default_models_dir() -> Result<PathBuf> {
    Ok(crate::data_dir()?.join("models").join("azoth"))
}

/// True if the directory looks like a complete bundle scan can load.
///
/// Two layouts are accepted: an ensemble bundle (`general/` subdirectory
/// containing the required artifacts) or a legacy single-bundle (artifacts
/// at the root). See `model.rs` for the loader-side details.
fn has_models(path: &Path) -> bool {
    has_single_bundle_layout(&path.join("general")) || has_single_bundle_layout(path)
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
