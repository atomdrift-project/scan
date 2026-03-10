//! Approximate SHAP explanations using global feature importance.
//!
//! Cross-references globally important features (from shap_importance.json)
//! with per-file active features to explain why a file was flagged.

use anyhow::{Context, Result};
use std::path::Path;

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
    /// Load from shap_importance.json in model directory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("shap_importance.json");
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

        log::info!("loaded {} SHAP importance features", features.len());
        Ok(Self { features })
    }

    /// Explain why a file was flagged by cross-referencing active features
    /// with global importance. Returns top N reasons.
    #[must_use]
    pub fn explain(
        &self,
        feature_values: &[f32],
        feature_names: &[String],
        max_reasons: usize,
    ) -> Vec<Reason> {
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
                let value = feature_values[*idx] as f64;
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

        reasons.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reasons.truncate(max_reasons);
        reasons
    }
}

/// Generate a human-readable description for a feature name.
fn describe_feature(name: &str) -> String {
    // v12 path×tier binary features: "path:objectives/evasion/process:hostile"
    if let Some(rest) = name.strip_prefix("path:") {
        if let Some((path, tier)) = rest.rsplit_once(':') {
            let short = humanize_path(path);
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

/// Shorten hierarchical paths to human-readable form.
fn humanize_path(path: &str) -> String {
    let short = path
        .strip_prefix("objectives/")
        .or_else(|| path.strip_prefix("micro-behaviors/"))
        .or_else(|| path.strip_prefix("well-known/"))
        .or_else(|| path.strip_prefix("metadata/"))
        .unwrap_or(path);
    short.replace('/', " > ")
}
