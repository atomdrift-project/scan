//! Integration test for the model-backend dispatch in `litmus::model::Model`.
//!
//! litmus is ONNX-only: the native LightGBM (`.txt`) and XGBoost (`.json`)
//! loaders were retired. These tests verify that:
//!   * a directory containing `model.onnx` loads via the ONNX backend.
//!   * multi-seed `models/seed_*.onnx` load and average correctly.
//!   * a directory with both `model.onnx` and `models/` is ambiguous.
//!   * a native `.txt`/`.json` (or empty) bundle is rejected.
//!
//! The ONNX tests are `#[ignore]`d by default because they depend on a real
//! ONNX bundle on disk. To run:
//!
//! ```sh
//! LITMUS_ONNX_BUNDLE=/home/t/azoth/filetypes/pe \
//!     cargo test --test backend_dispatch -- --ignored
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use litmus::model::Model;

fn onnx_bundle() -> Option<PathBuf> {
    let p = std::env::var("LITMUS_ONNX_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/t/azoth/filetypes/pe"));
    (p.join("model.onnx").is_file() && p.join("feature_spec.json").is_file()).then_some(p)
}

fn copy_bundle(src: &Path, files: &[&str]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    for f in files {
        std::fs::copy(src.join(f), dir.path().join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    dir
}

#[test]
#[ignore = "needs LITMUS_ONNX_BUNDLE pointing at a model.onnx+feature_spec.json bundle (default: /home/t/azoth/filetypes/pe)"]
fn onnx_bundle_dispatches_to_onnx_backend() {
    let Some(src) = onnx_bundle() else {
        panic!("LITMUS_ONNX_BUNDLE not set or missing artifacts");
    };
    let dir = copy_bundle(&src, &["model.onnx", "feature_spec.json"]);
    let model = Model::load(dir.path(), None, None).expect("load ONNX bundle");
    assert_eq!(model.backend_kind(), "onnx");
    assert!(model.spec().total_features() > 0);

    let zeros = vec![0.0f32; model.spec().total_features()];
    let (prob, _class) = model.predict(&zeros).expect("predict zeros");
    assert!(prob.is_finite() && (0.0..=1.0).contains(&prob));
}

#[test]
#[ignore = "needs LITMUS_ONNX_BUNDLE pointing at a model.onnx+feature_spec.json bundle"]
fn multi_seed_onnx_bundle_loads_and_predicts() {
    // Two seed members loaded from the same source model.onnx — averaging two
    // identical models must produce a probability identical to either member's
    // single prediction (within f32 noise). Confirms the K-member load and
    // predict paths are wired correctly without needing two distinct models.
    let Some(src) = onnx_bundle() else {
        panic!("LITMUS_ONNX_BUNDLE not set or missing artifacts");
    };
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        src.join("feature_spec.json"),
        dir.path().join("feature_spec.json"),
    )
    .unwrap();
    let models_dir = dir.path().join("models");
    std::fs::create_dir(&models_dir).unwrap();
    std::fs::copy(src.join("model.onnx"), models_dir.join("seed_42.onnx")).unwrap();
    std::fs::copy(src.join("model.onnx"), models_dir.join("seed_43.onnx")).unwrap();
    // Bystander file — must be ignored, not loaded.
    std::fs::write(models_dir.join("README.md"), b"hi").unwrap();

    let model = Model::load(dir.path(), None, None).expect("load multi-seed bundle");
    assert_eq!(model.backend_kind(), "onnx");
    let zeros = vec![0.0f32; model.spec().total_features()];
    let (prob, _class) = model.predict(&zeros).expect("predict zeros");
    assert!(prob.is_finite() && (0.0..=1.0).contains(&prob));

    let single = copy_bundle(&src, &["model.onnx", "feature_spec.json"]);
    let single_model = Model::load(single.path(), None, None).expect("load single bundle");
    let (single_prob, _) = single_model.predict(&zeros).expect("predict zeros");
    assert!(
        (prob - single_prob).abs() < 1e-6,
        "K=2 (identical members) prediction {prob} != K=1 prediction {single_prob}"
    );
}

#[test]
#[ignore = "needs LITMUS_ONNX_BUNDLE pointing at a model.onnx+feature_spec.json bundle"]
fn ambiguous_onnx_legacy_plus_multi_seed_layout_is_rejected() {
    // A bundle with BOTH top-level `model.onnx` and `models/seed_*.onnx` is
    // ambiguous — the deploy should choose one layout. We refuse to guess.
    let Some(src) = onnx_bundle() else {
        panic!("LITMUS_ONNX_BUNDLE not set or missing artifacts");
    };
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        src.join("feature_spec.json"),
        dir.path().join("feature_spec.json"),
    )
    .unwrap();
    std::fs::copy(src.join("model.onnx"), dir.path().join("model.onnx")).unwrap();
    let models_dir = dir.path().join("models");
    std::fs::create_dir(&models_dir).unwrap();
    std::fs::copy(src.join("model.onnx"), models_dir.join("seed_42.onnx")).unwrap();

    let err = Model::load(dir.path(), None, None).expect_err("ambiguous layout must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("ambiguous"),
        "error should mention 'ambiguous': {msg}"
    );
}

#[test]
fn native_txt_model_is_rejected() {
    // The LightGBM/XGBoost loaders are gone: a `.txt`/`.json`-only bundle is
    // no longer loadable. (Resolution happens before the model is parsed, so
    // stub contents are fine.)
    for native in ["model.txt", "model.json"] {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("feature_spec.json"), b"{}").unwrap();
        std::fs::write(dir.path().join(native), b"stub").unwrap();
        let err = Model::load(dir.path(), None, None).expect_err("native model must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("incomplete") || msg.contains("model.onnx"),
            "error should point at the missing model.onnx: {msg}"
        );
    }
}

#[test]
fn empty_bundle_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let err = Model::load(dir.path(), None, None).expect_err("empty dir must error");
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
    let err = Model::load(dir.path(), None, None).expect_err("spec-only must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("model.onnx"),
        "error should mention the missing model.onnx: {msg}"
    );
}

#[test]
fn empty_models_dir_falls_through_to_incomplete_error() {
    // A bundle with `models/` but no recognized seed files (and no legacy
    // model.onnx either) must error out with the same "incomplete" path so the
    // operator gets a clear signal.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("feature_spec.json"), b"{}").unwrap();
    std::fs::create_dir(dir.path().join("models")).unwrap();
    std::fs::write(dir.path().join("models").join("README.md"), b"hi").unwrap();

    let err = Model::load(dir.path(), None, None).expect_err("empty models dir must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("incomplete") || msg.contains("missing"),
        "error should mention incomplete/missing: {msg}"
    );
}
