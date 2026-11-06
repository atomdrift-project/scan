//! Verify the feature spec version matches what atomscan was compiled against.
//!
//! Catches model/code version drift at CI time rather than in production.

use anyhow::{Context, Result};
use scan::features::{EXPECTED_MODEL_ABI_VERSION, EXPECTED_SPEC_VERSION, FeatureSpec};

fn model_dir() -> Result<std::path::PathBuf> {
    scan::env::path(scan::env::SCAN_MODELS_DIR)
        .context("set SCAN_MODELS_DIR to run integration tests against real model artifacts")
}

#[test]
#[ignore = "needs SCAN_MODELS_DIR pointing at a model bundle"]
fn spec_version_matches_expected() -> Result<()> {
    let spec_path = model_dir()?.join("general").join("feature_spec.json");
    assert!(
        spec_path.exists(),
        "{} not found — set SCAN_MODELS_DIR or clone the models repo",
        spec_path.display()
    );

    let spec = FeatureSpec::load(&spec_path).context("failed to load feature spec")?;

    assert_eq!(
        spec.version(),
        EXPECTED_SPEC_VERSION,
        "feature_spec.json is v{} but atomscan expects v{EXPECTED_SPEC_VERSION} — \
         update atomscan feature extraction or retrain the model",
        spec.version(),
    );
    assert_eq!(
        spec.abi_version(),
        EXPECTED_MODEL_ABI_VERSION,
        "feature_spec.json ABI is v{} but atomscan expects ABI v{EXPECTED_MODEL_ABI_VERSION}",
        spec.abi_version(),
    );
    // `load` has checked the layout itself, including a non-empty
    // presence vocabulary.
    assert!(
        spec.total_features() > 0,
        "feature spec has 0 features — corrupt spec?",
    );
    Ok(())
}
