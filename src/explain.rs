//! Approximate SHAP explanations using global feature importance.
//!
//! Cross-references globally important features (from shap_importance.json)
//! with per-file active features to explain why a file was flagged.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::features::FeatureSpec;

/// Emit the feature-space mismatch warning at most once per process (it would
/// otherwise fire per file on a directory scan).
static SHAP_MISMATCH_WARNED: AtomicBool = AtomicBool::new(false);

/// `shap_importance.json` as collimator writes it.
#[derive(Debug, serde::Deserialize)]
struct ShapFile {
    #[serde(default)]
    top_features: Vec<ShapFeature>,
    /// Provenance: SHA-256 of the ordered feature-name list the SHAP was
    /// computed against (collimator stamps it). Absent on legacy files.
    feature_names_sha256: Option<String>,
}

/// A single feature importance entry from shap_importance.json.
#[derive(Debug, Clone, serde::Deserialize)]
struct ShapFeature {
    name: String,
    importance: f64,
}

/// One important feature, resolved to its slot in the model's feature vector.
#[derive(Debug, Clone)]
struct BoundFeature {
    slot: usize,
    name: String,
    importance: f64,
    description: String,
}

/// Global SHAP importance, bound to the feature space it was computed for.
#[derive(Debug, Clone)]
pub struct ShapImportance {
    /// The important features present in the spec, most important first.
    features: Vec<BoundFeature>,
    /// Length of the feature vector the slots index.
    n_features: usize,
}

/// SHA-256 over the ordered feature names, newline-joined — must match
/// `collimator.explain.feature_names_digest` byte-for-byte.
fn feature_names_digest(feature_names: &[String]) -> String {
    let mut hasher = Sha256::new();
    for (i, name) in feature_names.iter().enumerate() {
        if i > 0 {
            hasher.update(b"\n");
        }
        hasher.update(name.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// A reason why a file was flagged.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Reason {
    /// Feature name (e.g., "crit_count:suspicious")
    pub feature: String,
    /// Global SHAP importance of this feature
    pub importance: f64,
    /// The feature's value for this file
    pub value: f64,
    /// Human-readable description
    pub description: String,
}

impl ShapImportance {
    /// Load shap_importance.json for the model directory and bind it to the
    /// feature spec beside it.
    ///
    /// Reasons are computed in the general feature space (`Model::spec()`), so
    /// the matching file in a routed bundle is the general route's
    /// (`<dir>/general/shap_importance.json`). Falls back to a root-level file
    /// for legacy single-model layouts. (Per-route SHAP for other routes is
    /// kept under each route dir for offline analysis; attributing a specific
    /// winning route in-scan would require explaining in that route's feature
    /// space — a larger change tracked separately.)
    ///
    /// A bundle without a SHAP file is `Ok(None)`: most ship none.
    ///
    /// # Errors
    /// When the file or its feature spec cannot be read, or it was not
    /// computed against that spec — a stale file (an older model) would
    /// otherwise yield misleading reasons.
    pub fn load(model_dir: &Path) -> Result<Option<Self>> {
        let Some(dir) = [model_dir.join("general"), model_dir.to_path_buf()]
            .into_iter()
            .find(|d| d.join("shap_importance.json").is_file())
        else {
            return Ok(None);
        };
        let path = dir.join("shap_importance.json");
        let data = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let file: ShapFile =
            serde_json::from_slice(&data).with_context(|| format!("parsing {}", path.display()))?;
        let spec = FeatureSpec::load(&dir.join("feature_spec.json"))
            .with_context(|| format!("loading the feature spec beside {}", path.display()))?;
        let shap = Self::bind(file, spec.feature_names()).inspect_err(|e| {
            if !SHAP_MISMATCH_WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!("ignoring {}: {e}", path.display());
            }
        })?;
        tracing::info!("loaded {} SHAP importance features", shap.features.len());
        Ok(Some(shap))
    }

    /// Resolve `file`'s features to slots in `feature_names`, which must be
    /// the feature space it was computed against.
    fn bind(file: ShapFile, feature_names: &[String]) -> Result<Self> {
        match &file.feature_names_sha256 {
            Some(stamp) if *stamp == feature_names_digest(feature_names) => {}
            Some(_) => anyhow::bail!("feature space does not match the loaded model (stale SHAP)"),
            None => anyhow::bail!("no provenance stamp (regenerate with `make azoth-shap`)"),
        }
        let slots: HashMap<&str, usize> = feature_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let mut features: Vec<BoundFeature> = file
            .top_features
            .into_iter()
            .filter_map(|f| {
                Some(BoundFeature {
                    slot: *slots.get(f.name.as_str())?,
                    description: describe_feature(&f.name),
                    name: f.name,
                    importance: f.importance,
                })
            })
            .collect();
        features.sort_by(|a, b| b.importance.total_cmp(&a.importance));
        Ok(Self {
            features,
            n_features: feature_names.len(),
        })
    }

    /// Explain why a file was flagged: the important features active in
    /// `feature_values` (a vector in the spec the SHAP file was bound to at
    /// load), by descending importance. A vector of another length gets none.
    #[must_use]
    pub fn explain(&self, feature_values: &[f32]) -> Vec<Reason> {
        if feature_values.len() != self.n_features {
            if !SHAP_MISMATCH_WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    expected = self.n_features,
                    got = feature_values.len(),
                    "ignoring shap_importance.json: feature vector is not the space it was loaded for"
                );
            }
            return Vec::new();
        }
        self.features
            .iter()
            .filter_map(|f| {
                let value = f64::from(*feature_values.get(f.slot)?);
                (value != 0.0).then(|| Reason {
                    feature: f.name.clone(),
                    importance: f.importance,
                    value,
                    description: f.description.clone(),
                })
            })
            .collect()
    }
}

/// Generate a human-readable description for a feature name.
fn describe_feature(name: &str) -> String {
    // v12 path×tier binary features: "path:objectives/evasion/process:hostile"
    if let Some(rest) = name.strip_prefix("path:") {
        if let Some((path, tier)) = rest.rsplit_once(':') {
            let short = path
                .strip_prefix("objectives/")
                .or_else(|| path.strip_prefix("micro-behaviors/"))
                .or_else(|| path.strip_prefix("well-known/"))
                .or_else(|| path.strip_prefix("metadata/"))
                .unwrap_or(path)
                .replace('/', " > ");
            return format!("{short} [{tier}]");
        }
        return format!("path: {}", rest.replace('/', " > "));
    }
    if let Some(rest) = name.strip_prefix("agg:") {
        return format!("aggregate: {}", rest.replace('_', " "));
    }
    if let Some(rest) = name.strip_prefix("ext:") {
        return format!("external: {}", rest.replace('_', " "));
    }
    if let Some(rest) = name.strip_prefix("metrics:") {
        return format!("metric: {}", rest.replace('_', " "));
    }
    if let Some(rest) = name.strip_prefix("filetype:") {
        return format!("file type: {rest}");
    }
    if let Some(rest) = name.strip_prefix("struct:") {
        return format!("structural: {}", rest.replace('_', " "));
    }
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| (*n).to_string()).collect()
    }

    fn shap_file(stamp: Option<String>) -> ShapFile {
        ShapFile {
            top_features: vec![
                ShapFeature {
                    name: "agg:max_crit".to_string(),
                    importance: 0.5,
                },
                ShapFeature {
                    name: "struct:zero_findings".to_string(),
                    importance: 0.9,
                },
                ShapFeature {
                    name: "not:in_spec".to_string(),
                    importance: 2.0,
                },
            ],
            feature_names_sha256: stamp,
        }
    }

    #[test]
    fn binds_once_and_explains_active_features_by_importance() {
        let spec = names(&["agg:max_crit", "struct:zero_findings", "ext:has_yara_match"]);
        let shap = ShapImportance::bind(shap_file(Some(feature_names_digest(&spec))), &spec)
            .expect("stamped for this spec");
        let reasons = shap.explain(&[5.0, 1.0, 0.0]);
        let got: Vec<(&str, f64)> = reasons
            .iter()
            .map(|r| (r.feature.as_str(), r.value))
            .collect();
        assert_eq!(got, [("struct:zero_findings", 1.0), ("agg:max_crit", 5.0)]);
        assert_eq!(reasons[1].description, "aggregate: max crit");
        // An inactive feature gives no reason.
        assert_eq!(shap.explain(&[0.0, 1.0, 0.0]).len(), 1);
        // A vector from another feature space gives none at all.
        assert!(shap.explain(&[1.0, 1.0]).is_empty());
    }

    #[test]
    fn stale_or_unstamped_shap_is_refused() {
        let spec = names(&["agg:max_crit"]);
        let stale = feature_names_digest(&names(&["agg:other"]));
        assert!(ShapImportance::bind(shap_file(Some(stale)), &spec).is_err());
        assert!(ShapImportance::bind(shap_file(None), &spec).is_err());
    }
}
