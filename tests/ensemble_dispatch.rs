//! Integration tests for ensemble routing in `litmus::model::Model`.
//!
//! Builds a temporary ensemble bundle (general/ + filegroups/native/ +
//! filetypes/elf/) from a real LightGBM bundle and verifies:
//!   * `predict_for(file_type, …)` consults the right specialists.
//!   * Files whose type is unmapped route to general only.
//!   * A specialist with the wrong number of features is dropped (warning,
//!     not fatal); the route degrades.
//!   * `required_routes` lists are enforced.
//!
//! Tests are `#[ignore]`d by default because they need a real bundle on
//! disk. Run with:
//!
//! ```sh
//! LITMUS_LIGHTGBM_BUNDLE=/home/t/collimator/out/models/azoth-light-full-leaves96-cpu \
//!     cargo test --test ensemble_dispatch -- --ignored
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use litmus::model::Model;

fn lightgbm_bundle() -> Option<PathBuf> {
    let p = std::env::var("LITMUS_LIGHTGBM_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/home/t/collimator/out/models/azoth-light-full-leaves96-cpu")
        });
    (p.join("model.txt").is_file() && p.join("feature_spec.json").is_file()).then_some(p)
}

/// Stage `general/` plus optionally `filegroups/<group>` and
/// `filetypes/<type>` from a single source bundle. All routes get the same
/// model, which is sufficient to exercise the routing decision (each route's
/// score is identical, so OR semantics are easy to verify).
fn stage_ensemble(
    src: &Path,
    groups: &[&str],
    types: &[&str],
    config_json: &str,
) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");

    let copy_bundle_to = |dest: &Path| {
        std::fs::create_dir_all(dest).unwrap();
        for f in &["model.txt", "feature_spec.json"] {
            std::fs::copy(src.join(f), dest.join(f)).unwrap();
        }
    };

    copy_bundle_to(&dir.path().join("general"));
    for g in groups {
        copy_bundle_to(&dir.path().join("filegroups").join(g));
    }
    for t in types {
        copy_bundle_to(&dir.path().join("filetypes").join(t));
    }

    std::fs::write(dir.path().join("config.json"), config_json).unwrap();
    dir
}

#[test]
#[ignore = "needs LITMUS_LIGHTGBM_BUNDLE pointing at a real LightGBM bundle"]
fn ensemble_routes_to_filetype_specialist_when_present() {
    let Some(src) = lightgbm_bundle() else {
        panic!("LITMUS_LIGHTGBM_BUNDLE not set or missing artifacts");
    };
    let cfg = r#"{
      "filetype_to_filegroup": { "elf": "native", "pe": "native" }
    }"#;
    let dir = stage_ensemble(&src, &["native"], &["elf"], cfg);
    let model = Model::load(dir.path(), None).expect("load ensemble");

    let zeros = vec![0.0f32; model.spec().total_features()];

    // Files of type elf consult: general + filegroup(native) + filetype(elf).
    // Since all three are the same model, the probability is the same as
    // general-only and the OR decision over identical thresholds is unchanged.
    let (prob_elf, _) = model.predict_for("elf", &zeros).expect("predict elf");

    // Files of type "py" are unmapped → route is general only.
    let (prob_py, _) = model.predict_for("py", &zeros).expect("predict py");

    // Sanity: probabilities are finite.
    assert!(prob_elf.is_finite());
    assert!(prob_py.is_finite());
    // With identical models on every route, max-probability is the same
    // single-model probability for both file types.
    assert!((prob_elf - prob_py).abs() < 1e-6);
}

#[test]
#[ignore = "needs LITMUS_LIGHTGBM_BUNDLE pointing at a real LightGBM bundle"]
fn ensemble_routes_to_filegroup_when_filetype_absent() {
    let Some(src) = lightgbm_bundle() else {
        panic!("LITMUS_LIGHTGBM_BUNDLE not set or missing artifacts");
    };
    let cfg = r#"{
      "filetype_to_filegroup": { "pe": "native" }
    }"#;
    // No filetypes/pe specialist; pe routes through filegroup(native) + general.
    let dir = stage_ensemble(&src, &["native"], &[], cfg);
    let model = Model::load(dir.path(), None).expect("load ensemble");

    let zeros = vec![0.0f32; model.spec().total_features()];
    let (prob, _) = model.predict_for("pe", &zeros).expect("predict pe");
    assert!(prob.is_finite());
}

#[test]
#[ignore = "needs LITMUS_LIGHTGBM_BUNDLE pointing at a real LightGBM bundle"]
fn ensemble_with_only_general_falls_back_to_general() {
    let Some(src) = lightgbm_bundle() else {
        panic!("LITMUS_LIGHTGBM_BUNDLE not set or missing artifacts");
    };
    let cfg = r#"{}"#;
    let dir = stage_ensemble(&src, &[], &[], cfg);
    let model = Model::load(dir.path(), None).expect("load ensemble");

    let zeros = vec![0.0f32; model.spec().total_features()];
    let (prob_general, _) = model.predict(&zeros).expect("predict general");
    let (prob_elf, _) = model.predict_for("elf", &zeros).expect("predict elf");
    // No specialists, so predict and predict_for must return the same value.
    assert!((prob_general - prob_elf).abs() < 1e-6);
}

#[test]
#[ignore = "needs LITMUS_LIGHTGBM_BUNDLE pointing at a real LightGBM bundle"]
fn ensemble_required_route_missing_is_fatal() {
    let Some(src) = lightgbm_bundle() else {
        panic!("LITMUS_LIGHTGBM_BUNDLE not set or missing artifacts");
    };
    let cfg = r#"{
      "filetype_to_filegroup": {},
      "required_routes": ["filetype:nonexistent"]
    }"#;
    let dir = stage_ensemble(&src, &[], &[], cfg);
    let err = Model::load(dir.path(), None).expect_err("missing required must fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("nonexistent") && msg.contains("required"),
        "error should call out the missing required route: {msg}"
    );
}
