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

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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
        msg.contains("model.onnx") && msg.contains("model.txt") && msg.contains("model.json"),
        "error should mention all three filenames: {msg}"
    );
}

fn onnx_bundle() -> Option<PathBuf> {
    let p = std::env::var("LITMUS_ONNX_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/t/azoth/filetypes/pe"));
    (p.join("model.onnx").is_file() && p.join("feature_spec.json").is_file()).then_some(p)
}

#[test]
#[ignore = "needs LITMUS_ONNX_BUNDLE pointing at a model.onnx+feature_spec.json bundle (default: /home/t/azoth/filetypes/pe)"]
fn onnx_bundle_dispatches_to_onnx_backend() {
    let Some(src) = onnx_bundle() else {
        panic!("LITMUS_ONNX_BUNDLE not set or missing artifacts");
    };
    let dir = copy_bundle(&src, &["model.onnx", "feature_spec.json"]);
    let model = Model::load(dir.path(), None).expect("load ONNX bundle");
    assert_eq!(model.backend_kind(), "onnx");
    assert!(model.spec().total_features() > 0);

    let zeros = vec![0.0f32; model.spec().total_features()];
    let (prob, _class) = model.predict(&zeros).expect("predict zeros");
    assert!(prob.is_finite() && (0.0..=1.0).contains(&prob));
}

#[test]
#[ignore = "needs both LITMUS_ONNX_BUNDLE and LITMUS_LIGHTGBM_BUNDLE pointing at the same trained route"]
fn onnx_is_preferred_over_txt_when_both_present() {
    // When a bundle ships both model.onnx and model.txt for the same
    // trained route (transitional ONNX-migration state), the .onnx is
    // loaded and the .txt is ignored. Compare predictions on a zero
    // vector to a separate txt-only load: they should match within
    // float32 noise (1e-5 is the bundle-wide parity bound the
    // collimator-side convert script enforces).
    let Some(onnx_src) = onnx_bundle() else {
        panic!("LITMUS_ONNX_BUNDLE not set");
    };
    let lgb_src = onnx_src.clone(); // PE bundle ships both .onnx and .txt today.
    if !lgb_src.join("model.txt").is_file() {
        panic!("expected {} to ship both model.onnx and model.txt for this test", lgb_src.display());
    }

    let hybrid_dir = copy_bundle(&onnx_src, &["model.onnx", "model.txt", "feature_spec.json"]);
    let hybrid = Model::load(hybrid_dir.path(), None).expect("hybrid bundle loads");
    assert_eq!(hybrid.backend_kind(), "onnx", "hybrid bundle should prefer ONNX");

    let txt_dir = copy_bundle(&lgb_src, &["model.txt", "feature_spec.json"]);
    let txt = Model::load(txt_dir.path(), None).expect("txt-only bundle loads");
    assert_eq!(txt.backend_kind(), "lightgbm");

    let zeros = vec![0.0f32; hybrid.spec().total_features()];
    let (p_onnx, _) = hybrid.predict(&zeros).expect("predict onnx");
    let (p_txt, _) = txt.predict(&zeros).expect("predict txt");
    let delta = (p_onnx - p_txt).abs();
    assert!(
        delta < 1e-5,
        "ONNX/.txt parity broken on zero vector: onnx={p_onnx} txt={p_txt} delta={delta:.3e}",
    );
}

#[test]
#[ignore = "needs LITMUS_LIGHTGBM_BUNDLE pointing at a model.txt+feature_spec.json bundle"]
fn multi_seed_lightgbm_bundle_loads_and_predicts() {
    // Two seed members loaded from the same source model.txt — averaging two
    // identical models must produce a probability identical to either member's
    // single prediction (within f32 noise). This confirms the K-member load
    // path and predict path are wired correctly without needing two distinct
    // trained models to compare against.
    let Some(src) = lightgbm_bundle() else {
        panic!("LITMUS_LIGHTGBM_BUNDLE not set or missing artifacts");
    };
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        src.join("feature_spec.json"),
        dir.path().join("feature_spec.json"),
    )
    .unwrap();
    let models_dir = dir.path().join("models");
    std::fs::create_dir(&models_dir).unwrap();
    std::fs::copy(src.join("model.txt"), models_dir.join("seed_42.txt")).unwrap();
    std::fs::copy(src.join("model.txt"), models_dir.join("seed_43.txt")).unwrap();
    // Bystander file — must be ignored, not loaded.
    std::fs::write(models_dir.join("README.md"), b"hi").unwrap();

    let model = Model::load(dir.path(), None).expect("load multi-seed bundle");
    assert_eq!(model.backend_kind(), "lightgbm");
    let zeros = vec![0.0f32; model.spec().total_features()];
    let (prob, _class) = model.predict(&zeros).expect("predict zeros");
    assert!(prob.is_finite() && (0.0..=1.0).contains(&prob));

    // Compare to the single-model prediction — must match within f32 noise.
    let single = copy_bundle(&src, &["model.txt", "feature_spec.json"]);
    let single_model = Model::load(single.path(), None).expect("load single bundle");
    let (single_prob, _) = single_model.predict(&zeros).expect("predict zeros");
    assert!(
        (prob - single_prob).abs() < 1e-6,
        "K=2 (identical members) prediction {prob} != K=1 prediction {single_prob}"
    );
}

#[test]
#[ignore = "needs LITMUS_LIGHTGBM_BUNDLE pointing at a model.txt+feature_spec.json bundle"]
fn ambiguous_legacy_plus_multi_seed_layout_is_rejected() {
    // A bundle with BOTH `model.txt` and `models/seed_*.txt` is ambiguous —
    // the deploy script should choose one layout. We refuse to silently pick.
    let Some(src) = lightgbm_bundle() else {
        panic!("LITMUS_LIGHTGBM_BUNDLE not set or missing artifacts");
    };
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        src.join("feature_spec.json"),
        dir.path().join("feature_spec.json"),
    )
    .unwrap();
    std::fs::copy(src.join("model.txt"), dir.path().join("model.txt")).unwrap();
    let models_dir = dir.path().join("models");
    std::fs::create_dir(&models_dir).unwrap();
    std::fs::copy(src.join("model.txt"), models_dir.join("seed_42.txt")).unwrap();

    let err = Model::load(dir.path(), None).expect_err("ambiguous layout must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("ambiguous"),
        "error should mention 'ambiguous': {msg}"
    );
}

#[test]
fn empty_models_dir_falls_through_to_incomplete_error() {
    // A bundle with `models/` but no recognized seed files (and no legacy
    // model.txt either) must error out with the same "incomplete" path so the
    // operator gets a clear signal.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("feature_spec.json"), b"{}").unwrap();
    std::fs::create_dir(dir.path().join("models")).unwrap();
    std::fs::write(dir.path().join("models").join("README.md"), b"hi").unwrap();

    let err = Model::load(dir.path(), None).expect_err("empty models dir must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("incomplete") || msg.contains("missing"),
        "error should mention incomplete/missing: {msg}"
    );
}
