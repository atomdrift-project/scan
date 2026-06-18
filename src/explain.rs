//! Approximate SHAP explanations using global feature importance.
//!
//! Cross-references globally important features (from shap_importance.json)
//! with per-file active features to explain why a file was flagged.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Emit the stale-SHAP warning at most once per process (it would otherwise
/// fire per file on a directory scan).
static SHAP_MISMATCH_WARNED: AtomicBool = AtomicBool::new(false);

/// A single feature importance entry from shap_importance.json.
#[derive(Debug, Clone, serde::Deserialize)]
struct ShapFeature {
    name: String,
    importance: f64,
}

/// Global SHAP importance data.
#[derive(Debug, Clone)]
pub struct ShapImportance {
    features: Vec<ShapFeature>,
    /// Provenance: SHA-256 of the ordered feature-name list the SHAP was
    /// computed against (collimator stamps it). `None` for legacy/unstamped
    /// files. A SHAP file is only valid for its exact feature space, so we
    /// refuse to apply one whose digest doesn't match the loaded model.
    feature_names_sha256: Option<String>,
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
    /// Load shap_importance.json for the model directory.
    ///
    /// Reasons are computed in the general feature space (`Model::spec()`), so
    /// the matching file in a routed bundle is the general route's
    /// (`<dir>/general/shap_importance.json`). Falls back to a root-level file
    /// for legacy single-model layouts. (Per-route SHAP for other routes is
    /// kept under each route dir for offline analysis; attributing a specific
    /// winning route in-scan would require explaining in that route's feature
    /// space — a larger change tracked separately.)
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = [
            model_dir.join("general").join("shap_importance.json"),
            model_dir.join("shap_importance.json"),
        ]
        .into_iter()
        .find(|p| p.is_file())
        .with_context(|| format!("no shap_importance.json under {}", model_dir.display()))?;
        let data = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&data).context("parsing SHAP data")?;
        let features: Vec<ShapFeature> = v["top_features"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|f| serde_json::from_value(f.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();
        let feature_names_sha256 = v["feature_names_sha256"].as_str().map(ToString::to_string);

        tracing::info!("loaded {} SHAP importance features", features.len());
        Ok(Self {
            features,
            feature_names_sha256,
        })
    }

    /// Explain why a file was flagged by cross-referencing active features
    /// with global importance, sorted by descending importance.
    #[must_use]
    pub fn explain(&self, feature_values: &[f32], feature_names: &[String]) -> Vec<Reason> {
        // Provenance gate: only apply SHAP computed against this exact feature
        // space. A stale file (different/old model) would otherwise yield
        // misleading reasons — the silent failure that masked a 6-week-old
        // SHAP. Mismatch or a missing stamp ⇒ no reasons, warn once.
        let valid = match &self.feature_names_sha256 {
            Some(stamp) => stamp == &feature_names_digest(feature_names),
            None => false,
        };
        if !valid {
            if !SHAP_MISMATCH_WARNED.swap(true, Ordering::Relaxed) {
                let reason = if self.feature_names_sha256.is_some() {
                    "feature space does not match the loaded model (stale SHAP)"
                } else {
                    "no provenance stamp (regenerate with `make azoth-shap`)"
                };
                tracing::warn!("ignoring shap_importance.json: {reason}");
            }
            return Vec::new();
        }

        let name_to_idx: std::collections::HashMap<&str, usize> = feature_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        let mut reasons: Vec<Reason> = self
            .features
            .iter()
            .filter_map(|shap| {
                let idx = name_to_idx.get(shap.name.as_str())?;
                let value = *feature_values.get(*idx)? as f64;
                if value == 0.0 {
                    return None;
                }
                Some(Reason {
                    feature: shap.name.clone(),
                    importance: shap.importance,
                    value,
                    description: describe_feature(&shap.name),
                })
            })
            .collect();

        reasons.sort_by(|a, b| b.importance.total_cmp(&a.importance));
        reasons
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
