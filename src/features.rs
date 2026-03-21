//! Feature extraction from cleave v3 AnalysisReport JSON.
//!
//! Mirrors the feature extraction in collimator/src/collimator/features.py (v13)
//! exactly, using the same feature_spec.json vocabulary to produce identical
//! feature vectors.
//!
//! Feature groups (v13):
//!   1. Presence: binary (1.0 if path exists at crit ≥ baseline)
//!   2. Max Criticality: ordinal (0–5) per path
//!   3. Aggregates: breadth, concentration, escalation ratios (8)
//!   4. Third-Party / Well-Known Summary (6)
//!   5. Key Metrics: curated binary/text/PE metrics (16)
//!   6. File Type: one-hot
//!   7. Structural: anomalies + finding count (4)

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Feature spec version this build was compiled against.
/// Must match the version in the loaded feature_spec.json.
pub const EXPECTED_SPEC_VERSION: u32 = 13;

/// Minimum finding confidence for inclusion (matches collimator MIN_CONFIDENCE).
const MIN_CONFIDENCE: f64 = 0.65;

/// Criticality ordinals (must match collimator CRITICALITY_ORDINAL).
pub(crate) fn crit_ordinal(crit: &str) -> u32 {
    match crit {
        "filtered" => 0,
        "component" => 1,
        "notable" => 3,
        "suspicious" => 4,
        "hostile" => 5,
        _ => 2, // baseline and unknown
    }
}

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

/// Feature specification loaded from feature_spec.json (v13).
#[derive(Debug, Clone)]
pub struct FeatureSpec {
    /// Spec format version (expected: 13).
    version: u32,
    /// Path vocabulary for presence/maxcrit features.
    presence_vocab: Vec<String>,
    /// File type vocabulary for one-hot encoding.
    filetype_vocab: Vec<String>,
    /// Names of all features in the vector, in order.
    feature_names: Vec<String>,
    /// Total number of features in the vector.
    total_features: usize,
    /// Per-feature means for z-score standardization (from training).
    feature_means: Option<Vec<f32>>,
    /// Per-feature standard deviations for z-score standardization (from training).
    feature_stds: Option<Vec<f32>>,
    /// Whether the model was trained on standardized features. When false,
    /// inference should use raw features directly (no z-score transform).
    standardized: bool,
}

impl FeatureSpec {
    /// Load feature specification from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).context("reading feature spec")?;
        let v: serde_json::Value = serde_json::from_str(&data).context("parsing feature spec")?;

        let version = v["version"].as_u64().unwrap_or(0) as u32;
        if version != EXPECTED_SPEC_VERSION {
            anyhow::bail!(
                "feature spec version mismatch: spec is v{version} but litmus requires v{EXPECTED_SPEC_VERSION} — \
                 the model was trained with a different collimator version and cannot be used with this build",
            );
        }

        Ok(Self {
            version,
            presence_vocab: json_string_array(&v["presence_vocab"]),
            filetype_vocab: json_string_array(&v["filetype_vocab"]),
            feature_names: json_string_array(&v["feature_names"]),
            total_features: v["total_features"].as_u64().unwrap_or(0) as usize,
            feature_means: json_f32_array(&v["feature_means"]),
            feature_stds: json_f32_array(&v["feature_stds"]),
            standardized: v["standardized"].as_bool().unwrap_or(true),
        })
    }

    /// Spec format version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Path vocabulary for presence and max-criticality features.
    #[must_use]
    pub fn presence_vocab(&self) -> &[String] {
        &self.presence_vocab
    }

    /// Names of all features in model input order.
    #[must_use]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Total feature count expected by the model.
    #[must_use]
    pub const fn total_features(&self) -> usize {
        self.total_features
    }

    /// Whether raw features should be standardized before inference.
    #[must_use]
    pub const fn is_standardized(&self) -> bool {
        self.standardized
    }

    /// Apply z-score standardization using training statistics.
    /// No-op if the model was trained on raw features (standardized=false).
    /// Features that were constant during training (mean=0, std=1) are zeroed
    /// to prevent catastrophic misclassification from unseen raw values.
    pub fn standardize(&self, features: &mut [f32]) {
        if !self.standardized {
            return;
        }
        let (Some(means), Some(stds)) = (self.feature_means.as_ref(), self.feature_stds.as_ref())
        else {
            return;
        };

        for (feature, (&m, &s)) in features.iter_mut().zip(means.iter().zip(stds.iter())) {
            if s.abs() <= f32::EPSILON
                || (m.abs() <= f32::EPSILON && (s - 1.0).abs() <= f32::EPSILON)
            {
                // Dead feature: constant during training, or zero std.
                // Zero it to prevent NaN/inf from propagating.
                *feature = 0.0;
            } else {
                *feature = (*feature - m) / s;
            }
        }
    }
}

fn json_string_array(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn json_f32_array(v: &serde_json::Value) -> Option<Vec<f32>> {
    v.as_array().map(|arr| {
        arr.iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect()
    })
}

/// Pre-built lookup tables for fast repeated extraction against a spec.
#[derive(Debug)]
pub(crate) struct ExtractContext {
    /// Maps path string -> index in presence_vocab.
    presence_lookup: HashMap<String, usize>,
    n_presence: usize,
    ft_lookup: HashMap<String, usize>,
    n_ft: usize,
    /// Total number of features in the output vector.
    total_features: usize,
}

impl ExtractContext {
    /// Build lookup tables from a feature specification.
    #[must_use]
    pub(crate) fn new(spec: &FeatureSpec) -> Self {
        let presence_lookup: HashMap<String, usize> = spec
            .presence_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        let ft_lookup: HashMap<String, usize> = spec
            .filetype_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        Self {
            presence_lookup,
            n_presence: spec.presence_vocab.len(),
            ft_lookup,
            n_ft: spec.filetype_vocab.len(),
            total_features: spec.total_features,
        }
    }

    /// Extract features from a cleave AnalysisReport serialized as JSON.
    #[must_use]
    pub(crate) fn extract(&self, report: &serde_json::Value) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.total_features];
        self.extract_into(report, &mut vec);
        vec
    }

    fn extract_into(&self, report: &serde_json::Value, vec: &mut [f32]) {
        let pf = primary_file(report);
        let findings = pf_array(pf, "findings");
        let mut offsets = FeatureCursor::default();
        let finding_summary = summarize_findings(&findings);

        // -------------------------------------------------------------------
        // Group 1: Presence features (n_presence binary features)
        // Set to 1.0 if path exists at criticality ≥ baseline (ordinal 2).
        // -------------------------------------------------------------------
        let presence_offset = offsets.take(self.n_presence);
        self.write_presence_features(&finding_summary.sample_paths, vec, presence_offset);

        // -------------------------------------------------------------------
        // Group 2: Max criticality features (n_presence ordinal features)
        // Stores the max criticality ordinal (0–5) per path.
        // -------------------------------------------------------------------
        let maxcrit_offset = offsets.take(self.n_presence);
        self.write_max_crit_features(&finding_summary.sample_paths, vec, maxcrit_offset);

        // -------------------------------------------------------------------
        // Group 3: Aggregates (8 features)
        // -------------------------------------------------------------------
        let agg_offset = offsets.take(8);
        write_aggregate_features(&finding_summary.sample_paths, vec, agg_offset);

        // -------------------------------------------------------------------
        // Group 4: Third-Party / Well-Known Summary (6 features)
        // -------------------------------------------------------------------
        let ext_offset = offsets.take(6);
        write_external_summary_features(&finding_summary, vec, ext_offset);

        // -------------------------------------------------------------------
        // Group 5: Key Metrics (16 features)
        // -------------------------------------------------------------------
        let metrics_offset = offsets.take(KEY_METRICS.len());
        write_metric_features(pf, vec, metrics_offset);

        // -------------------------------------------------------------------
        // Group 6: File Type one-hot
        // -------------------------------------------------------------------
        let file_type = pf["file_type"].as_str().unwrap_or("");
        let file_type_offset = offsets.take(self.n_ft);
        self.write_file_type_features(file_type, vec, file_type_offset);

        // -------------------------------------------------------------------
        // Group 7: Structural (4 features)
        // -------------------------------------------------------------------
        let file_size = pf["size"].as_u64().unwrap_or(0);
        let is_binary = matches!(file_type, "pe" | "elf" | "macho");
        let imports = pf_array(pf, "imports");
        let structural_offset = offsets.take(4);
        write_structural_features(
            vec,
            structural_offset,
            is_binary,
            file_size,
            imports.is_empty(),
            finding_summary.filtered_finding_count,
        );
    }

    fn write_presence_features(
        &self,
        sample_paths: &HashMap<&str, u32>,
        vec: &mut [f32],
        offset: usize,
    ) {
        for (path, &max_ord) in sample_paths {
            if max_ord >= 2 {
                if let Some(&idx) = self.presence_lookup.get(*path) {
                    vec[offset + idx] = 1.0;
                }
            }
        }
    }

    fn write_max_crit_features(
        &self,
        sample_paths: &HashMap<&str, u32>,
        vec: &mut [f32],
        offset: usize,
    ) {
        for (path, &max_ord) in sample_paths {
            if let Some(&idx) = self.presence_lookup.get(*path) {
                vec[offset + idx] = max_ord as f32;
            }
        }
    }

    fn write_file_type_features(&self, file_type: &str, vec: &mut [f32], offset: usize) {
        if let Some(&idx) = self.ft_lookup.get(file_type) {
            vec[offset + idx] = 1.0;
        }
    }
}

#[derive(Debug, Default)]
struct FeatureCursor {
    offset: usize,
}

impl FeatureCursor {
    fn take(&mut self, width: usize) -> usize {
        let start = self.offset;
        self.offset += width;
        start
    }
}

#[derive(Debug, Default)]
struct FindingSummary<'a> {
    sample_paths: HashMap<&'a str, u32>,
    filtered_finding_count: u32,
    third_party_max_crit: u32,
    third_party_count: u32,
    well_known_max_crit: u32,
    well_known_hostile: u32,
    well_known_suspicious: u32,
    has_yara: bool,
}

fn summarize_findings<'a>(findings: &[&'a serde_json::Value]) -> FindingSummary<'a> {
    let mut summary = FindingSummary::default();

    for finding in findings {
        let fid = finding["id"].as_str().unwrap_or("");
        if fid.is_empty() {
            continue;
        }

        let conf = finding["conf"].as_f64().unwrap_or(1.0);
        if conf < MIN_CONFIDENCE {
            continue;
        }
        summary.filtered_finding_count += 1;

        let crit_ord = crit_ordinal(finding["crit"].as_str().unwrap_or("baseline"));
        let top = fid.split('/').next().unwrap_or("");

        match top {
            "third_party" => {
                summary.third_party_count += 1;
                summary.third_party_max_crit = summary.third_party_max_crit.max(crit_ord);
                summary.has_yara = true;
            }
            "well-known" => {
                summary.well_known_max_crit = summary.well_known_max_crit.max(crit_ord);
                if crit_ord >= 5 {
                    summary.well_known_hostile += 1;
                } else if crit_ord >= 4 {
                    summary.well_known_suspicious += 1;
                }
            }
            _ => {}
        }

        for path in finding_paths(fid) {
            let entry = summary.sample_paths.entry(path).or_insert(0);
            *entry = (*entry).max(crit_ord);
        }
    }

    summary
}

fn write_aggregate_features(sample_paths: &HashMap<&str, u32>, vec: &mut [f32], offset: usize) {
    let mut max_crit = 0u32;
    let mut categories: HashSet<&str> = HashSet::new();
    let mut path_breadth_any = 0u32;
    let mut total_active = 0u32;
    let mut breadth_notable = 0u32;
    let mut breadth_suspicious = 0u32;
    let mut breadth_hostile = 0u32;
    let mut breadth_notable_only = 0u32;

    for (path, &max_ord) in sample_paths {
        let path_depth = path.chars().filter(|&c| c == '/').count();
        if max_ord >= 2 {
            let top = path.split('/').next().unwrap_or("");
            categories.insert(top);
            if path_depth >= 2 {
                path_breadth_any += 1;
            }
        }

        if path_depth < 2 || max_ord < 3 {
            continue;
        }
        total_active += 1;
        breadth_notable += 1;
        max_crit = max_crit.max(max_ord);
        if max_ord >= 4 {
            breadth_suspicious += 1;
        }
        if max_ord >= 5 {
            breadth_hostile += 1;
        } else if max_ord == 3 {
            breadth_notable_only += 1;
        }
    }

    vec[offset] = max_crit as f32;
    vec[offset + 1] = categories.len() as f32;
    vec[offset + 2] = (path_breadth_any as f32 + 1.0).ln();
    vec[offset + 3] = (total_active as f32 + 1.0).ln();
    vec[offset + 4] = breadth_suspicious as f32 / path_breadth_any.max(1) as f32;
    vec[offset + 5] = breadth_hostile as f32 / path_breadth_any.max(1) as f32;
    vec[offset + 6] = breadth_suspicious as f32 / breadth_notable.max(1) as f32;
    vec[offset + 7] = breadth_notable_only as f32 / breadth_notable.max(1) as f32;
}

fn write_external_summary_features(summary: &FindingSummary<'_>, vec: &mut [f32], offset: usize) {
    vec[offset] = summary.third_party_max_crit as f32;
    vec[offset + 1] = (summary.third_party_count as f32 + 1.0).ln();
    vec[offset + 2] = summary.well_known_max_crit as f32;
    vec[offset + 3] = summary.well_known_hostile as f32;
    vec[offset + 4] = summary.well_known_suspicious as f32;
    vec[offset + 5] = if summary.has_yara { 1.0 } else { 0.0 };
}

fn write_metric_features(pf: &serde_json::Value, vec: &mut [f32], mut offset: usize) {
    let metrics = pf.get("metrics").unwrap_or(&serde_json::Value::Null);
    for &(group, field_name, use_log) in KEY_METRICS {
        let value = metrics
            .get(group)
            .and_then(|group_metrics| group_metrics.get(field_name))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;
        vec[offset] = if use_log {
            (value.abs() + 1.0).ln()
        } else {
            value
        };
        offset += 1;
    }
}

fn write_structural_features(
    vec: &mut [f32],
    offset: usize,
    is_binary: bool,
    file_size: u64,
    imports_empty: bool,
    filtered_finding_count: u32,
) {
    vec[offset] = if is_binary && file_size < 20_000 {
        1.0
    } else {
        0.0
    };
    vec[offset + 1] = if imports_empty { 1.0 } else { 0.0 };
    vec[offset + 2] = if filtered_finding_count == 0 {
        1.0
    } else {
        0.0
    };
    vec[offset + 3] = (filtered_finding_count as f32 + 1.0).ln();
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
    pf[key]
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_default()
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
                let end = if self.n_slashes >= 1 {
                    self.slash_ends[0]
                } else {
                    self.base.len()
                };
                Some(&self.base[..end])
            }
            // 2-component prefix: up to second slash.
            1 if self.n_slashes >= 1 => {
                let end = if self.n_slashes >= 2 {
                    self.slash_ends[1]
                } else {
                    self.base.len()
                };
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
            vec![
                "objectives",
                "objectives/evasion",
                "objectives/evasion/process"
            ],
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
            vec![
                "objectives",
                "objectives/evasion",
                "objectives/evasion/process"
            ],
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
            version: 13,
            presence_vocab: vec![],
            filetype_vocab: vec![],
            feature_names: vec![],
            total_features: 3,
            feature_means: Some(vec![0.0, 1.0, 2.0]),
            feature_stds: Some(vec![1.0, 2.0, 0.5]),
            standardized: true,
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
            version: 13,
            presence_vocab: vec!["objectives/evasion/process".to_string()],
            filetype_vocab: vec!["sh".to_string()],
            feature_names: vec![],
            // 1 presence + 1 maxcrit + 8 agg + 6 ext + 16 metrics + 1 filetype + 4 struct
            total_features: 1 + 1 + 8 + 6 + 16 + 1 + 4,
            feature_means: None,
            feature_stds: None,
            standardized: false,
        };
        let ctx = ExtractContext::new(&spec);
        let report = serde_json::json!({"files": [{"file_type": "sh", "size": 100}]});
        let features = ctx.extract(&report);
        assert_eq!(features.len(), spec.total_features);
        // presence feature: 0 (no findings)
        assert_eq!(features[0], 0.0);
        // maxcrit feature: 0 (no findings)
        assert_eq!(features[1], 0.0);
        // filetype sh one-hot
        let ft_offset = 1 + 1 + 8 + 6 + 16;
        assert_eq!(features[ft_offset], 1.0);
        // zero_findings structural feature
        let struct_offset = ft_offset + 1;
        assert_eq!(features[struct_offset + 2], 1.0);
    }

    #[test]
    fn test_extract_with_findings() {
        let spec = FeatureSpec {
            version: 13,
            presence_vocab: vec![
                "objectives".to_string(),
                "objectives/evasion".to_string(),
                "objectives/evasion/process".to_string(),
            ],
            filetype_vocab: vec!["php".to_string()],
            feature_names: vec![],
            total_features: 3 + 3 + 8 + 6 + 16 + 1 + 4,
            feature_means: None,
            feature_stds: None,
            standardized: false,
        };
        let ctx = ExtractContext::new(&spec);
        let report = serde_json::json!({
            "files": [{
                "file_type": "php",
                "size": 1000,
                "findings": [
                    {"id": "objectives/evasion/process/injection::test", "crit": "hostile", "conf": 0.9},
                ],
                "metrics": {},
            }]
        });
        let features = ctx.extract(&report);

        // All 3 presence features should be 1.0 (hostile ≥ baseline)
        assert_eq!(features[0], 1.0); // present:objectives
        assert_eq!(features[1], 1.0); // present:objectives/evasion
        assert_eq!(features[2], 1.0); // present:objectives/evasion/process

        // All 3 maxcrit features should be 5.0 (hostile)
        assert_eq!(features[3], 5.0); // maxcrit:objectives
        assert_eq!(features[4], 5.0); // maxcrit:objectives/evasion
        assert_eq!(features[5], 5.0); // maxcrit:objectives/evasion/process

        // agg:max_crit = 5.0
        assert_eq!(features[6], 5.0);
    }

    #[test]
    fn test_confidence_filtering() {
        let spec = FeatureSpec {
            version: 13,
            presence_vocab: vec!["objectives".to_string()],
            filetype_vocab: vec![],
            feature_names: vec![],
            total_features: 1 + 1 + 8 + 6 + 16 + 4,
            feature_means: None,
            feature_stds: None,
            standardized: false,
        };
        let ctx = ExtractContext::new(&spec);
        let report = serde_json::json!({
            "files": [{
                "file_type": "sh",
                "size": 100,
                "findings": [
                    {"id": "objectives/evasion::test", "crit": "hostile", "conf": 0.3},
                ],
                "metrics": {},
            }]
        });
        let features = ctx.extract(&report);
        // Low confidence finding should be skipped
        assert_eq!(features[0], 0.0); // present:objectives
        assert_eq!(features[1], 0.0); // maxcrit:objectives
    }
}
