//! The smallest model bundle that loads: a `general/` route holding
//! `fixtures/cli-model.onnx`, a one-feature tree ensemble, and the spec that
//! names its feature; and a `config.json` whose level grid puts the general
//! route at suspicious 0.9 and hostile 0.99 at the default level.

use std::io;
use std::path::Path;

/// Write the bundle into `dir`, creating it.
pub(crate) fn write(dir: &Path) -> io::Result<()> {
    let general = dir.join("general");
    std::fs::create_dir_all(&general)?;
    std::fs::write(
        general.join("model.onnx"),
        include_bytes!("../fixtures/cli-model.onnx"),
    )?;
    let spec = serde_json::json!({
        "version": scan::features::EXPECTED_SPEC_VERSION,
        "abi_version": scan::features::EXPECTED_MODEL_ABI_VERSION,
        "base_feature_abi": 21,
        "trait_path_depth": 3,
        "presence_vocab": ["metadata"],
        "feature_names": ["present:metadata"],
        "total_features": 1,
        "standardized": false,
        "metric_transforms": {}
    });
    std::fs::write(
        general.join("feature_spec.json"),
        serde_json::to_vec(&spec).map_err(io::Error::other)?,
    )?;
    // Hostile at L25; the suspicious band reads the loosest row within
    // 100 × 25 = L2500.
    std::fs::write(
        dir.join("config.json"),
        br#"{"levels": [
            {"level": 25, "hostile": {"thresholds": {"general": 0.99}}},
            {"level": 2500, "hostile": {"thresholds": {"general": 0.9}}}
        ]}"#,
    )
}
