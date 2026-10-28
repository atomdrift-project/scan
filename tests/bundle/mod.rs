//! The smallest model bundle that loads: `fixtures/cli-model.onnx`, a
//! one-feature tree ensemble, and the spec that names its feature.

use std::io;
use std::path::Path;

/// Write `model.onnx` and `feature_spec.json` into `dir`, creating it.
pub(crate) fn write(dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(
        dir.join("model.onnx"),
        include_bytes!("../fixtures/cli-model.onnx"),
    )?;
    let spec = serde_json::json!({
        "version": scan::features::EXPECTED_SPEC_VERSION,
        "abi_version": scan::features::EXPECTED_MODEL_ABI_VERSION,
        "presence_vocab": ["metadata"],
        "feature_names": ["present:metadata"],
        "total_features": 1,
        "standardized": false,
        "metric_transforms": {}
    });
    std::fs::write(
        dir.join("feature_spec.json"),
        serde_json::to_vec(&spec).map_err(io::Error::other)?,
    )
}
