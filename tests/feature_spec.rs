//! Verify the feature spec version matches what litmus was compiled against.
//!
//! Catches model/code version drift at CI time rather than in production.

use litmus::features::{FeatureSpec, EXPECTED_SPEC_VERSION};

fn model_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LITMUS_MODELS_DIR") {
        return std::path::PathBuf::from(d);
    }
    litmus::models_repo::model_dir().expect("failed to resolve model directory")
}

#[test]
fn spec_version_matches_expected() {
    let spec_path = model_dir().join("feature_spec.json");
    if !spec_path.exists() {
        panic!(
            "SKIPPED (would hide failures): {} not found — set LITMUS_MODELS_DIR or clone the models repo",
            spec_path.display()
        );
    }

    let spec = FeatureSpec::load(&spec_path)
        .unwrap_or_else(|e| panic!("failed to load feature spec: {e}"));

    assert_eq!(
        spec.version, EXPECTED_SPEC_VERSION,
        "feature_spec.json is v{} but litmus expects v{EXPECTED_SPEC_VERSION} — \
         update litmus feature extraction or retrain the model",
        spec.version,
    );
    assert!(
        spec.total_features > 0,
        "feature spec has 0 features — corrupt spec?",
    );
    assert!(
        !spec.presence_vocab.is_empty(),
        "feature spec has no presence_vocab — wrong spec version?",
    );
}
