//! Feature extraction from cleave v3 AnalysisReport JSON.
//!
//! Mirrors the feature extraction in collimator/src/collimator/features.py (v12)
//! exactly, using the same feature_spec.json vocabulary to produce identical
//! feature vectors.
//!
//! Feature groups:
//!   1. Path × Tier: binary features for hierarchical paths × crit tiers
//!   2. Path Aggregates: attack breadth, context breadth, ratios (8)
//!   3. Third-Party / Well-Known Summary: aggregated match signals (6)
//!   4. Key Metrics: curated binary/text/PE metrics (16)
//!   5. File Type: one-hot
//!   6. Structural: anomalies + finding count (4)

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Criticality ordinals (must match collimator CRITICALITY_ORDINAL).
fn crit_ordinal(crit: &str) -> u32 {
    match crit {
        "filtered" => 0,
        "component" => 1,
        "notable" => 3,
        "suspicious" => 4,
        "hostile" => 5,
        _ => 2,
    }
}

/// Tier thresholds used for path×tier binary features (notable=3, suspicious=4, hostile=5).
const TIERS: &[(&str, u32)] = &[("notable", 3), ("suspicious", 4), ("hostile", 5)];

/// Top-level categories counted as "attack" paths for aggregate features.
const ATTACK_TOPS: &[&str] = &["objectives"];
/// Top-level categories counted as "context" (benign indicator) paths.
const CONTEXT_TOPS: &[&str] = &["metadata"];

/// Key metrics extracted from the report's `metrics` object.
/// Each entry is (group, field, use_log1p).
const KEY_METRICS: &[(&str, &str, bool)] = &[
    // Binary structure
    ("binary", "overall_entropy", false),
    ("binary", "code_entropy", false),
    ("binary", "code_to_data_ratio", false),
    ("binary", "function_count", true),
    ("binary", "complexity_per_kb", false),
    ("binary", "max_complexity", false),
    ("binary", "normalized_string_count", false),
    ("binary", "high_entropy_regions", false),
    // Text analysis
    ("text", "char_entropy", false),
    ("text", "unique_chars", true),
    ("text", "whitespace_ratio", false),
    ("text", "most_common_ratio", false),
    ("text", "total_lines", true),
    // String analysis
    ("strings", "avg_entropy", false),
    // PE-specific
    ("pe", "rsrc_entropy", false),
    ("pe", "rsrc_size", true),
];

/// Feature specification loaded from feature_spec.json (v12).
#[derive(Debug, Clone)]
pub struct FeatureSpec {
    /// Spec format version (expected: 12).
    pub version: u32,
    /// Entries like "objectives/evasion/process:hostile".
    pub path_vocab: Vec<String>,
    /// File type vocabulary for one-hot encoding.
    pub filetype_vocab: Vec<String>,
    /// Names of all features in the vector, in order.
    pub feature_names: Vec<String>,
    /// Total number of features in the vector.
    pub total_features: usize,
    /// Per-feature means for z-score standardization (from training).
    pub feature_means: Option<Vec<f32>>,
    /// Per-feature standard deviations for z-score standardization (from training).
    pub feature_stds: Option<Vec<f32>>,
}

impl FeatureSpec {
    /// Load feature specification from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).context("reading feature spec")?;
        let v: serde_json::Value = serde_json::from_str(&data).context("parsing feature spec")?;

        let version = v["version"].as_u64().unwrap_or(12) as u32;
        if version < 12 {
            tracing::warn!(
                "feature spec version {} (expected 12); models trained with older collimator are not compatible",
                version,
            );
        }

        Ok(Self {
            version,
            path_vocab: json_string_array(&v["path_vocab"]),
            filetype_vocab: json_string_array(&v["filetype_vocab"]),
            feature_names: json_string_array(&v["feature_names"]),
            total_features: v["total_features"].as_u64().unwrap_or(0) as usize,
            feature_means: json_f32_array(&v["feature_means"]),
            feature_stds: json_f32_array(&v["feature_stds"]),
        })
    }

    /// Apply z-score standardization using training statistics.
    /// Features that were constant during training (mean=0, std=1) are zeroed
    /// to prevent catastrophic misclassification from unseen raw values.
    pub fn standardize(&self, features: &mut [f32]) {
        let (Some(means), Some(stds)) = (self.feature_means.as_ref(), self.feature_stds.as_ref()) else { return };

        for i in 0..features.len().min(means.len()) {
            let m = means[i];
            let s = stds[i];
            if m == 0.0 && s == 1.0 {
                features[i] = 0.0;
            } else {
                features[i] = (features[i] - m) / s;
            }
        }
    }
}

fn json_string_array(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .map(|arr| arr.iter().filter_map(|s| s.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

fn json_f32_array(v: &serde_json::Value) -> Option<Vec<f32>> {
    v.as_array().map(|arr| {
        arr.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()
    })
}

/// Pre-built lookup tables for fast repeated extraction against a spec.
#[derive(Debug)]
pub struct ExtractContext {
    /// Maps path string -> list of (feature_index_in_combo_block, tier_ordinal).
    path_tiers: HashMap<String, Vec<(usize, u32)>>,
    n_combos: usize,
    ft_lookup: HashMap<String, usize>,
    n_ft: usize,
    /// Total number of features in the output vector.
    pub total_features: usize,
}

impl ExtractContext {
    /// Build lookup tables from a feature specification.
    #[must_use]
    pub fn new(spec: &FeatureSpec) -> Self {
        let mut path_tiers: HashMap<String, Vec<(usize, u32)>> = HashMap::new();
        for (i, combo) in spec.path_vocab.iter().enumerate() {
            // combo is "objectives/evasion/process:hostile"
            if let Some((path, tier_name)) = combo.rsplit_once(':') {
                if let Some(&tier_ord) = TIERS.iter().find(|&&(t, _)| t == tier_name).map(|(_, o)| o) {
                    path_tiers.entry(path.to_string()).or_default().push((i, tier_ord));
                }
            }
        }

        let ft_lookup: HashMap<String, usize> = spec
            .filetype_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        Self {
            path_tiers,
            n_combos: spec.path_vocab.len(),
            ft_lookup,
            n_ft: spec.filetype_vocab.len(),
            total_features: spec.total_features,
        }
    }

    /// Extract features from a cleave AnalysisReport serialized as JSON.
    #[must_use]
    pub fn extract(&self, report: &serde_json::Value) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.total_features];
        self.extract_into(report, &mut vec);
        vec
    }

    fn extract_into(&self, report: &serde_json::Value, vec: &mut [f32]) {
        let pf = primary_file(report);
        let findings = pf_array(pf, "findings");
        let mut offset = 0usize;

        // -------------------------------------------------------------------
        // Group 1: Path × Tier binary features (n_combos)
        // -------------------------------------------------------------------
        let path_offset = offset;
        offset += self.n_combos;

        // Per-path max crit across all findings (at all hierarchy levels).
        let mut sample_paths: HashMap<&str, u32> = HashMap::new();
        let mut third_party_max_crit = 0u32;
        let mut third_party_count = 0u32;
        let mut well_known_max_crit = 0u32;
        let mut well_known_hostile = 0u32;
        let mut well_known_suspicious = 0u32;
        let mut has_yara = false;

        for finding in &findings {
            let fid = finding["id"].as_str().unwrap_or("");
            if fid.is_empty() {
                continue;
            }
            let crit_ord = crit_ordinal(finding["crit"].as_str().unwrap_or("baseline"));

            let top = fid.split('/').next().unwrap_or("");
            if top == "third_party" {
                third_party_count += 1;
                if crit_ord > third_party_max_crit {
                    third_party_max_crit = crit_ord;
                }
                has_yara = true;
            } else if top == "well-known" {
                if crit_ord > well_known_max_crit {
                    well_known_max_crit = crit_ord;
                }
                if crit_ord >= 5 {
                    well_known_hostile += 1;
                } else if crit_ord >= 4 {
                    well_known_suspicious += 1;
                }
            }

            for path in finding_paths(fid) {
                let entry = sample_paths.entry(path).or_insert(0);
                if crit_ord > *entry {
                    *entry = crit_ord;
                }
            }
        }

        // Set binary path×tier features.
        for (path, &max_ord) in &sample_paths {
            if let Some(entries) = self.path_tiers.get(*path) {
                for &(feat_idx, tier_ord) in entries {
                    if max_ord >= tier_ord {
                        vec[path_offset + feat_idx] = 1.0;
                    }
                }
            }
        }

        // -------------------------------------------------------------------
        // Group 2: Path Aggregates (8 features)
        // -------------------------------------------------------------------
        let agg_offset = offset;
        offset += 8;

        let mut attack_breadth_notable = 0u32;
        let mut attack_breadth_suspicious = 0u32;
        let mut attack_max_crit = 0u32;
        let mut micro_breadth_notable = 0u32;
        let mut context_breadth = 0u32;
        let mut total_active = 0u32;

        for (path, &max_ord) in &sample_paths {
            // Only count 3-level paths (two '/' separators) to avoid double-counting.
            if path.chars().filter(|&c| c == '/').count() < 2 {
                continue;
            }
            if max_ord < 3 {
                continue;
            }
            total_active += 1;
            let top = path.split('/').next().unwrap_or("");
            if ATTACK_TOPS.contains(&top) {
                attack_breadth_notable += 1;
                if max_ord >= 4 {
                    attack_breadth_suspicious += 1;
                }
                if max_ord > attack_max_crit {
                    attack_max_crit = max_ord;
                }
            } else if CONTEXT_TOPS.contains(&top) {
                context_breadth += 1;
            } else if top == "micro-behaviors" {
                micro_breadth_notable += 1;
            }
        }

        vec[agg_offset] = attack_breadth_notable as f32;
        vec[agg_offset + 1] = attack_breadth_suspicious as f32;
        vec[agg_offset + 2] = attack_max_crit as f32;
        vec[agg_offset + 3] = micro_breadth_notable as f32;
        vec[agg_offset + 4] = context_breadth as f32;
        vec[agg_offset + 5] = attack_breadth_notable as f32 / (context_breadth as f32 + 1.0);
        vec[agg_offset + 6] = (total_active as f32 + 1.0).ln();
        vec[agg_offset + 7] = attack_breadth_notable as f32
            * (context_breadth as f32 / total_active.max(1) as f32);

        // -------------------------------------------------------------------
        // Group 3: Third-Party / Well-Known Summary (6 features)
        // -------------------------------------------------------------------
        let ext_offset = offset;
        offset += 6;

        vec[ext_offset] = third_party_max_crit as f32;
        vec[ext_offset + 1] = (third_party_count as f32 + 1.0).ln();
        vec[ext_offset + 2] = well_known_max_crit as f32;
        vec[ext_offset + 3] = well_known_hostile as f32;
        vec[ext_offset + 4] = well_known_suspicious as f32;
        vec[ext_offset + 5] = if has_yara { 1.0 } else { 0.0 };

        // -------------------------------------------------------------------
        // Group 4: Key Metrics (16 features)
        // -------------------------------------------------------------------
        let metrics = pf.get("metrics").unwrap_or(&serde_json::Value::Null);
        for &(group, fname, use_log) in KEY_METRICS {
            let val = metrics
                .get(group)
                .and_then(|g| g.get(fname))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0) as f32;
            vec[offset] = if use_log { (val.abs() + 1.0).ln() } else { val };
            offset += 1;
        }

        // -------------------------------------------------------------------
        // Group 5: File Type one-hot
        // -------------------------------------------------------------------
        let file_type = pf["file_type"].as_str().unwrap_or("");
        if let Some(&idx) = self.ft_lookup.get(file_type) {
            vec[offset + idx] = 1.0;
        }
        offset += self.n_ft;

        // -------------------------------------------------------------------
        // Group 6: Structural (4 features)
        // -------------------------------------------------------------------
        let file_size = pf["size"].as_u64().unwrap_or(0);
        let is_binary = matches!(file_type, "pe" | "elf" | "macho");
        let imports = pf_array(pf, "imports");

        vec[offset] = if is_binary && file_size < 20_000 { 1.0 } else { 0.0 };
        vec[offset + 1] = if imports.is_empty() { 1.0 } else { 0.0 };
        vec[offset + 2] = if findings.is_empty() { 1.0 } else { 0.0 };
        vec[offset + 3] = (findings.len() as f32 + 1.0).ln();
    }
}

/// Return the primary (first) file entry from a v3 report.
fn primary_file(report: &serde_json::Value) -> &serde_json::Value {
    report["files"]
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or(&serde_json::Value::Null)
}

/// Get an array field as a slice of Value refs.
fn pf_array<'a>(pf: &'a serde_json::Value, key: &str) -> Vec<&'a serde_json::Value> {
    pf[key].as_array().map(|a| a.iter().collect()).unwrap_or_default()
}

/// Extract hierarchical path prefixes (1, 2, up to 3 levels) from a finding ID.
///
/// Mirrors collimator's `_finding_paths`: splits the base (before `::`) by `/`
/// and returns at most 3 prefixes — parts[:1], parts[:2], parts[:3].
///
/// "objectives/evasion/process/injection::technique-x"
///     -> ["objectives", "objectives/evasion", "objectives/evasion/process"]
/// "metadata/format::no-functions"
///     -> ["metadata", "metadata/format"]
fn finding_paths(finding_id: &str) -> impl Iterator<Item = &str> {
    let base = finding_id.split("::").next().unwrap_or(finding_id);
    // Collect slash byte positions (up to 2, giving us up to 3 components).
    let mut slash_ends: [usize; 2] = [0; 2];
    let mut n_slashes = 0usize;
    for (i, ch) in base.char_indices() {
        if ch == '/' {
            if n_slashes < 2 {
                slash_ends[n_slashes] = i;
            }
            n_slashes += 1;
        }
    }
    // Determine the end of the 3rd component (everything up to the 3rd slash, or end).
    let third_end = if n_slashes >= 3 {
        // Find position of the 3rd slash.
        let mut count = 0;
        let mut pos = base.len();
        for (i, ch) in base.char_indices() {
            if ch == '/' {
                count += 1;
                if count == 3 {
                    pos = i;
                    break;
                }
            }
        }
        pos
    } else {
        base.len()
    };

    FindingPaths {
        base,
        slash_ends,
        n_slashes: n_slashes.min(2),
        third_end,
        step: 0,
    }
}

struct FindingPaths<'a> {
    base: &'a str,
    slash_ends: [usize; 2],
    n_slashes: usize,
    third_end: usize,
    step: usize,
}

impl<'a> Iterator for FindingPaths<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.step {
            // 1-component prefix: up to first slash (or full base if no slashes).
            0 => {
                let end = if self.n_slashes >= 1 { self.slash_ends[0] } else { self.base.len() };
                Some(&self.base[..end])
            }
            // 2-component prefix: up to second slash.
            1 if self.n_slashes >= 1 => {
                let end = if self.n_slashes >= 2 { self.slash_ends[1] } else { self.base.len() };
                Some(&self.base[..end])
            }
            // 3-component prefix: up to third slash (or end).
            2 if self.n_slashes >= 2 => Some(&self.base[..self.third_end]),
            _ => return None,
        };
        self.step += 1;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(id: &str) -> Vec<&str> {
        finding_paths(id).collect()
    }

    #[test]
    fn test_finding_paths_deep() {
        assert_eq!(
            paths("objectives/evasion/process/injection::technique-x"),
            vec!["objectives", "objectives/evasion", "objectives/evasion/process"],
        );
    }

    #[test]
    fn test_finding_paths_two_levels() {
        assert_eq!(
            paths("metadata/format::no-functions"),
            vec!["metadata", "metadata/format"],
        );
    }

    #[test]
    fn test_finding_paths_one_level() {
        assert_eq!(paths("standalone"), vec!["standalone"]);
    }

    #[test]
    fn test_finding_paths_exactly_three() {
        assert_eq!(
            paths("objectives/evasion/process::technique"),
            vec!["objectives", "objectives/evasion", "objectives/evasion/process"],
        );
    }

    #[test]
    fn test_crit_ordinal() {
        assert_eq!(crit_ordinal("hostile"), 5);
        assert_eq!(crit_ordinal("suspicious"), 4);
        assert_eq!(crit_ordinal("notable"), 3);
        assert_eq!(crit_ordinal("baseline"), 2);
        assert_eq!(crit_ordinal("unknown"), 2);
    }

    #[test]
    fn test_standardize() {
        let spec = FeatureSpec {
            version: 12,
            path_vocab: vec![],
            filetype_vocab: vec![],
            feature_names: vec![],
            total_features: 3,
            feature_means: Some(vec![0.0, 1.0, 2.0]),
            feature_stds: Some(vec![1.0, 2.0, 0.5]),
        };
        let mut features = vec![1.0, 3.0, 3.0];
        spec.standardize(&mut features);
        // dead feature (mean=0, std=1): zeroed
        assert_eq!(features[0], 0.0);
        // (3.0 - 1.0) / 2.0 = 1.0
        assert!((features[1] - 1.0).abs() < 1e-6);
        // (3.0 - 2.0) / 0.5 = 2.0
        assert!((features[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_extract_empty_report() {
        let spec = FeatureSpec {
            version: 12,
            path_vocab: vec!["objectives/evasion/process:hostile".to_string()],
            filetype_vocab: vec!["sh".to_string()],
            feature_names: vec![],
            total_features: 1 + 8 + 6 + 16 + 1 + 4,
            feature_means: None,
            feature_stds: None,
        };
        let ctx = ExtractContext::new(&spec);
        let report = serde_json::json!({"files": [{"file_type": "sh", "size": 100}]});
        let features = ctx.extract(&report);
        assert_eq!(features.len(), spec.total_features);
        // path feature: 0 (no findings)
        assert_eq!(features[0], 0.0);
        // zero_findings structural feature
        let struct_offset = 1 + 8 + 6 + 16 + 1;
        assert_eq!(features[struct_offset + 2], 1.0); // zero_findings
        // filetype sh one-hot
        let ft_offset = 1 + 8 + 6 + 16;
        assert_eq!(features[ft_offset], 1.0);
    }
}
