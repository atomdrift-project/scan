//! Verify the feature spec version matches what litmus was compiled against.
//!
//! Catches model/code version drift at CI time rather than in production.

use anyhow::{Context, Result};
use litmus::features::{EXPECTED_MODEL_ABI_VERSION, EXPECTED_SPEC_VERSION, FeatureSpec};

fn model_dir() -> Result<std::path::PathBuf> {
    std::env::var("LITMUS_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .context("set LITMUS_MODELS_DIR to run integration tests against real model artifacts")
}

#[test]
fn spec_version_matches_expected() -> Result<()> {
    if std::env::var_os("LITMUS_MODELS_DIR").is_none() {
        eprintln!("skipping: LITMUS_MODELS_DIR is not set");
        return Ok(());
    }

    let dir = model_dir()?;
    let root_spec = dir.join("feature_spec.json");
    let ensemble_spec = dir.join("general").join("feature_spec.json");
    let spec_path = if root_spec.exists() {
        root_spec
    } else {
        ensemble_spec
    };
    assert!(
        spec_path.exists(),
        "{} not found — set LITMUS_MODELS_DIR or clone the models repo",
        spec_path.display()
    );

    let spec = FeatureSpec::load(&spec_path).context("failed to load feature spec")?;

    assert_eq!(
        spec.version(),
        EXPECTED_SPEC_VERSION,
        "feature_spec.json is v{} but litmus expects v{EXPECTED_SPEC_VERSION} — \
         update litmus feature extraction or retrain the model",
        spec.version(),
    );
    assert_eq!(
        spec.abi_version(),
        EXPECTED_MODEL_ABI_VERSION,
        "feature_spec.json ABI is v{} but litmus expects ABI v{EXPECTED_MODEL_ABI_VERSION}",
        spec.abi_version(),
    );
    assert!(
        spec.total_features() > 0,
        "feature spec has 0 features — corrupt spec?",
    );
    assert!(
        !spec.presence_vocab().is_empty(),
        "feature spec has no presence_vocab — wrong spec version?",
    );
    Ok(())
}
