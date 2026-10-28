//! `Analyzer`'s builders must change what the analyzer classifies with.
//!
//! They used to rewrite only the analyzer's config, while the model it scans
//! with kept the thresholds it was loaded under, so a caller who set a cutoff
//! got no error and no effect.

use anyhow::Result;
use scan::{Analyzer, Thresholds};

/// A one-feature model bundle that loads without any external artifacts.
fn bundle() -> Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("model.onnx"),
        include_bytes!("fixtures/cli-model.onnx"),
    )?;
    std::fs::write(
        dir.path().join("feature_spec.json"),
        serde_json::to_vec(&serde_json::json!({
            "version": scan::features::EXPECTED_SPEC_VERSION,
            "abi_version": scan::features::EXPECTED_MODEL_ABI_VERSION,
            "presence_vocab": ["metadata"],
            "feature_names": ["present:metadata"],
            "total_features": 1,
            "standardized": false,
            "metric_transforms": {}
        }))?,
    )?;
    std::fs::write(
        dir.path().join("config.json"),
        br#"{"suspicious":0.9,"hostile":0.99}"#,
    )?;
    Ok(dir)
}

#[test]
fn with_thresholds_reaches_the_model() -> Result<()> {
    let dir = bundle()?;
    let wanted = Thresholds {
        suspicious: 0.2,
        hostile: 0.3,
    };
    let analyzer = Analyzer::load(dir.path())
        .expect("load")
        .with_thresholds(wanted)
        .expect("valid thresholds");
    assert_eq!(analyzer.model().thresholds(), wanted);
    Ok(())
}

#[test]
fn invalid_thresholds_are_refused() -> Result<()> {
    let dir = bundle()?;
    let inverted = Thresholds {
        suspicious: 0.9,
        hostile: 0.1,
    };
    assert!(
        Analyzer::load(dir.path())
            .expect("load")
            .with_thresholds(inverted)
            .is_err()
    );
    Ok(())
}
