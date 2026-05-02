//! Integration test for the model-backend dispatch in `litmus::model::Model`.
//!
//! Verifies that:
//!   * a directory containing `model.json` loads via the XGBoost backend.
//!   * a directory containing `model.txt` loads via the LightGBM backend.
//!   * a directory containing both fails with an "ambiguous bundle" error.
//!   * a directory containing neither fails with an "incomplete bundle" error.
//!
//! Each test points `Model::load` at a fresh tempdir populated from a
//! known-good real bundle, so the dispatch logic is exercised end-to-end
//! including feature-spec loading, threshold resolution, and feature-count
//! cross-validation between the spec and the backend.
//!
//! These tests are `#[ignore]`d by default because they depend on real model
//! bundles existing on disk. To run:
//!
//! ```sh
//! LITMUS_XGBOOST_BUNDLE=/home/t/collimator/out \
//! LITMUS_LIGHTGBM_BUNDLE=/home/t/collimator/out/models/azoth-light-full-leaves96-cpu \
//!     cargo test --test backend_dispatch -- --ignored
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use litmus::model::Model;

fn xgboost_bundle() -> Option<PathBuf> {
    let p = std::env::var("LITMUS_XGBOOST_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/t/collimator/out"));
    (p.join("model.json").is_file() && p.join("feature_spec.json").is_file()).then_some(p)
}

fn lightgbm_bundle() -> Option<PathBuf> {
    let p = std::env::var("LITMUS_LIGHTGBM_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/home/t/collimator/out/models/azoth-light-full-leaves96-cpu")
        });
    (p.join("model.txt").is_file() && p.join("feature_spec.json").is_file()).then_some(p)
}

fn copy_bundle(src: &Path, files: &[&str]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    for f in files {
        std::fs::copy(src.join(f), dir.path().join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    dir
}

#[test]
#[ignore = "needs LITMUS_XGBOOST_BUNDLE pointing at a model.json+feature_spec.json bundle"]
fn xgboost_bundle_dispatches_to_xgboost_backend() {
    let Some(src) = xgboost_bundle() else {
        panic!("LITMUS_XGBOOST_BUNDLE not set or missing artifacts");
    };
    let dir = copy_bundle(&src, &["model.json", "feature_spec.json"]);
    let model = Model::load(dir.path(), None).expect("load XGBoost bundle");
    assert_eq!(model.backend_kind(), "xgboost");
    assert!(model.spec().total_features() > 0);

    // Predict zeros — should not panic and should produce a finite probability.
    let zeros = vec![0.0f32; model.spec().total_features()];
    let (prob, _class) = model.predict(&zeros).expect("predict zeros");
    assert!(prob.is_finite() && (0.0..=1.0).contains(&prob));
}

#[test]
#[ignore = "needs LITMUS_LIGHTGBM_BUNDLE pointing at a model.txt+feature_spec.json bundle"]
fn lightgbm_bundle_dispatches_to_lightgbm_backend() {
    let Some(src) = lightgbm_bundle() else {
        panic!("LITMUS_LIGHTGBM_BUNDLE not set or missing artifacts");
    };
    let dir = copy_bundle(&src, &["model.txt", "feature_spec.json"]);
    let model = Model::load(dir.path(), None).expect("load LightGBM bundle");
    assert_eq!(model.backend_kind(), "lightgbm");
    assert!(model.spec().total_features() > 0);

    let zeros = vec![0.0f32; model.spec().total_features()];
    let (prob, _class) = model.predict(&zeros).expect("predict zeros");
    assert!(prob.is_finite() && (0.0..=1.0).contains(&prob));
}

#[test]
#[ignore = "needs both bundles available"]
fn ambiguous_bundle_with_both_model_files_is_rejected() {
    let Some(xgb) = xgboost_bundle() else {
        panic!("LITMUS_XGBOOST_BUNDLE not set");
    };
    let Some(lgb) = lightgbm_bundle() else {
        panic!("LITMUS_LIGHTGBM_BUNDLE not set");
    };
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(xgb.join("model.json"), dir.path().join("model.json")).unwrap();
    std::fs::copy(lgb.join("model.txt"), dir.path().join("model.txt")).unwrap();
    std::fs::copy(
        xgb.join("feature_spec.json"),
        dir.path().join("feature_spec.json"),
    )
    .unwrap();
    let err = Model::load(dir.path(), None).expect_err("ambiguous bundle must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("ambiguous"),
        "error should mention 'ambiguous': {msg}"
    );
}

#[test]
fn empty_bundle_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let err = Model::load(dir.path(), None).expect_err("empty dir must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("incomplete") || msg.contains("missing"),
        "error should mention incomplete/missing: {msg}"
    );
}

#[test]
fn bundle_with_spec_but_no_model_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("feature_spec.json"), b"{}").unwrap();
    let err = Model::load(dir.path(), None).expect_err("spec-only must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("model.json") && msg.contains("model.txt"),
        "error should mention both filenames: {msg}"
    );
}
