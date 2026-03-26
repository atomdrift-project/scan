//! Feature extraction from cleave v3 AnalysisReport JSON.
//!
//! Mirrors the feature extraction in collimator/src/collimator/features.py (v15)
//! exactly, using the same feature_spec.json vocabulary to produce identical
//! feature vectors.
//!
//! Feature groups (v15):
//!   1. Presence: binary (1.0 if path exists at crit >= baseline)
//!   2. Max Criticality: ordinal (0-5) per path
//!   3. Aggregates: breadth, concentration, finding-density, hostile-escalation signals (25)
//!   4. Third-Party / Well-Known Summary (6)
//!   5. Key Metrics: curated binary/text/PE metrics (16)
//!   6. File Type: multi-hot across all files
//!   7. Structural: report/container context (7)

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Feature spec version this build was compiled against.
/// Must match the version in the loaded feature_spec.json.
pub const EXPECTED_SPEC_VERSION: u32 = 15;
/// Stable model ABI version shared with collimator.
/// Keep this in sync with EXPECTED_SPEC_VERSION for a single compatibility number.
pub const EXPECTED_MODEL_ABI_VERSION: u32 = EXPECTED_SPEC_VERSION;

/// Minimum finding confidence for inclusion (matches collimator MIN_CONFIDENCE).
const MIN_CONFIDENCE: f64 = 0.65;

/// Number of riskiest files to summarize for top-k aggregate features.
const TOP_K_RISK_FILES: usize = 1;

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

/// Feature specification loaded from feature_spec.json (v15).
#[derive(Debug, Clone)]
pub struct FeatureSpec {
    /// Spec format version (expected: 15).
    version: u32,
    /// Stable preprocessing/inference ABI version.
    abi_version: u32,
    /// Path vocabulary for presence/maxcrit features.
    presence_vocab: Vec<String>,
    /// File type vocabulary for multi-hot encoding.
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

#[derive(Debug, serde::Deserialize)]
struct RawFeatureSpec {
    #[serde(default)]
    version: u32,
    #[serde(default = "default_abi_version")]
    abi_version: u32,
    #[serde(default)]
    presence_vocab: Vec<String>,
    #[serde(default)]
    filetype_vocab: Vec<String>,
    #[serde(default)]
    feature_names: Vec<String>,
    #[serde(default)]
    total_features: usize,
    feature_means: Option<Vec<f32>>,
    feature_stds: Option<Vec<f32>>,
    #[serde(default = "default_standardized")]
    standardized: bool,
}

const fn default_standardized() -> bool {
    true
}

const fn default_abi_version() -> u32 {
    EXPECTED_MODEL_ABI_VERSION
}

impl FeatureSpec {
    /// Load feature specification from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).context("reading feature spec")?;
        let raw: RawFeatureSpec = serde_json::from_str(&data).context("parsing feature spec")?;

        if raw.version != EXPECTED_SPEC_VERSION {
            anyhow::bail!(
                "feature spec version mismatch: this installed model uses spec v{}, but this litmus build requires v{EXPECTED_SPEC_VERSION}. \
                 The model is incompatible with this build. Run 'litmus update-rules' to install a matching model bundle.",
                raw.version,
            );
        }

        let spec = Self {
            version: raw.version,
            abi_version: raw.abi_version,
            presence_vocab: raw.presence_vocab,
            filetype_vocab: raw.filetype_vocab,
            feature_names: raw.feature_names,
            total_features: raw.total_features,
            feature_means: raw.feature_means,
            feature_stds: raw.feature_stds,
            standardized: raw.standardized,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Spec format version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Stable model ABI version.
    #[must_use]
    pub const fn abi_version(&self) -> u32 {
        self.abi_version
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

    fn validate(&self) -> Result<()> {
        if self.presence_vocab.is_empty() {
            anyhow::bail!("feature spec missing presence_vocab entries");
        }
        if self.feature_names.len() != self.total_features {
            anyhow::bail!(
                "feature spec feature_names length {} does not match total_features {}",
                self.feature_names.len(),
                self.total_features
            );
        }
        if self.abi_version != EXPECTED_MODEL_ABI_VERSION {
            anyhow::bail!(
                "feature spec ABI mismatch: spec has ABI v{} but litmus requires ABI v{}",
                self.abi_version,
                EXPECTED_MODEL_ABI_VERSION
            );
        }

        let expected_feature_names =
            build_expected_feature_names(&self.presence_vocab, &self.filetype_vocab);
        if self.feature_names != expected_feature_names {
            anyhow::bail!(
                "feature spec feature_names do not match the expected v{EXPECTED_SPEC_VERSION} layout"
            );
        }

        match (
            self.feature_means.as_ref(),
            self.feature_stds.as_ref(),
            self.standardized,
        ) {
            (Some(means), Some(stds), _) => {
                if means.len() != self.total_features || stds.len() != self.total_features {
                    anyhow::bail!(
                        "feature spec standardization stats must each have {} entries (got means={}, stds={})",
                        self.total_features,
                        means.len(),
                        stds.len()
                    );
                }
            }
            (None, None, false) => {}
            (None, None, true) => {
                anyhow::bail!("feature spec is marked standardized but feature_means/feature_stds are missing");
            }
            (Some(_), None, _) | (None, Some(_), _) => {
                anyhow::bail!(
                    "feature spec must include both feature_means and feature_stds together"
                );
            }
        }

        Ok(())
    }
}

fn build_expected_feature_names(
    presence_vocab: &[String],
    filetype_vocab: &[String],
) -> Vec<String> {
    let mut feature_names = Vec::with_capacity(
        presence_vocab.len() * 2 + 25 + 6 + KEY_METRICS.len() + filetype_vocab.len() + 7,
    );

    for path in presence_vocab {
        feature_names.push(format!("present:{path}"));
    }

    for path in presence_vocab {
        feature_names.push(format!("maxcrit:{path}"));
    }

    feature_names.extend([
        "agg:max_crit".to_string(),
        "agg:category_breadth".to_string(),
        "agg:path_breadth_any".to_string(),
        "agg:total_active_paths".to_string(),
        "agg:suspicious_concentration".to_string(),
        "agg:hostile_concentration".to_string(),
        "agg:escalation_rate".to_string(),
        "agg:notable_only_fraction".to_string(),
        "agg:notable_findings_log".to_string(),
        "agg:suspicious_findings_log".to_string(),
        "agg:hostile_findings_log".to_string(),
        "agg:notable_finding_ratio".to_string(),
        "agg:suspicious_finding_ratio".to_string(),
        "agg:hostile_finding_ratio".to_string(),
        "agg:unique_suspicious_ids_log".to_string(),
        "agg:unique_hostile_ids_log".to_string(),
        format!("agg:top{TOP_K_RISK_FILES}_file_suspicious_ratio_sum"),
        format!("agg:top{TOP_K_RISK_FILES}_file_hostile_ratio_sum"),
        format!("agg:top{TOP_K_RISK_FILES}_file_suspicious_findings_log"),
        format!("agg:top{TOP_K_RISK_FILES}_file_hostile_findings_log"),
        "agg:hostile_escalation_rate".to_string(),
        "agg:hostile_share_of_suspicious".to_string(),
        "agg:suspicious_finding_escalation_rate".to_string(),
        "agg:hostile_finding_escalation_rate".to_string(),
        "agg:hostile_share_of_suspicious_findings".to_string(),
    ]);

    feature_names.extend([
        "ext:third_party_max_crit".to_string(),
        "ext:third_party_count".to_string(),
        "ext:well_known_max_crit".to_string(),
        "ext:well_known_hostile_count".to_string(),
        "ext:well_known_suspicious_count".to_string(),
        "ext:has_yara_match".to_string(),
    ]);

    for &(group, field_name, _) in KEY_METRICS {
        feature_names.push(format!("metrics:{group}_{field_name}"));
    }

    for filetype in filetype_vocab {
        feature_names.push(format!("filetype:{filetype}"));
    }

    feature_names.extend([
        "struct:tiny_executable".to_string(),
        "struct:no_imports".to_string(),
        "struct:zero_findings".to_string(),
        "struct:finding_count_log".to_string(),
        "struct:file_count_log".to_string(),
        "struct:inner_file_count_log".to_string(),
        "struct:stealth_potential".to_string(),
    ]);

    feature_names
}

/// Pre-built lookup tables for fast repeated extraction against a spec.
#[derive(Debug)]
pub struct ExtractContext {
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
    /// Build lookup tables from a feature specification.
    pub fn new(spec: &FeatureSpec) -> Self {
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
    /// Extract features from a cleave AnalysisReport serialized as JSON.
    pub fn extract(&self, report: &serde_json::Value) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.total_features];
        self.extract_into(report, &mut vec);
        vec
    }

    fn extract_into(&self, report: &serde_json::Value, vec: &mut [f32]) {
        let empty_obj = serde_json::Value::Object(serde_json::Map::new());
        let raw_files = report_files(report);
        // Match Python: if no files, substitute a single empty object so
        // structural features always have at least one file entry.
        let files: Vec<&serde_json::Value> = if raw_files.is_empty() {
            vec![&empty_obj]
        } else {
            raw_files
        };
        let summary = summarize_report_files(&files);
        let merged_metrics = merge_metric_values(&files);
        let mut offsets = FeatureCursor::default();

        // -------------------------------------------------------------------
        // Group 1: Presence features (n_presence binary features)
        // Set to 1.0 if path exists at criticality >= baseline (ordinal 2).
        // -------------------------------------------------------------------
        let presence_offset = offsets.take(self.n_presence);
        self.write_presence_features(&summary.sample_paths, vec, presence_offset);

        // -------------------------------------------------------------------
        // Group 2: Max criticality features (n_presence ordinal features)
        // Stores the max criticality ordinal (0-5) per path.
        // -------------------------------------------------------------------
        let maxcrit_offset = offsets.take(self.n_presence);
        self.write_max_crit_features(&summary.sample_paths, vec, maxcrit_offset);

        // -------------------------------------------------------------------
        // Group 3: Aggregates (25 features)
        // -------------------------------------------------------------------
        let agg_offset = offsets.take(25);
        write_aggregate_features(&summary, &files, vec, agg_offset);

        // -------------------------------------------------------------------
        // Group 4: Third-Party / Well-Known Summary (6 features)
        // -------------------------------------------------------------------
        let ext_offset = offsets.take(6);
        write_external_summary_features(&summary, vec, ext_offset);

        // -------------------------------------------------------------------
        // Group 5: Key Metrics (16 features)
        // -------------------------------------------------------------------
        let metrics_offset = offsets.take(KEY_METRICS.len());
        write_metric_features(&merged_metrics, vec, metrics_offset);

        // -------------------------------------------------------------------
        // Group 6: File Type multi-hot across all files
        // -------------------------------------------------------------------
        let file_type_offset = offsets.take(self.n_ft);
        self.write_file_type_features(&files, vec, file_type_offset);

        // -------------------------------------------------------------------
        // Group 7: Structural (7 features)
        // -------------------------------------------------------------------
        let structural_offset = offsets.take(7);
        write_structural_features(
            vec,
            structural_offset,
            &files,
            summary.filtered_finding_count,
        );
    }

    fn write_presence_features(
        &self,
        sample_paths: &HashMap<String, u32>,
        vec: &mut [f32],
        offset: usize,
    ) {
        for (path, &max_ord) in sample_paths {
            if max_ord >= 2 {
                if let Some(&idx) = self.presence_lookup.get(path.as_str()) {
                    vec[offset + idx] = 1.0;
                }
            }
        }
    }

    fn write_max_crit_features(
        &self,
        sample_paths: &HashMap<String, u32>,
        vec: &mut [f32],
        offset: usize,
    ) {
        for (path, &max_ord) in sample_paths {
            if let Some(&idx) = self.presence_lookup.get(path.as_str()) {
                vec[offset + idx] = max_ord as f32;
            }
        }
    }

    fn write_file_type_features(
        &self,
        files: &[&serde_json::Value],
        vec: &mut [f32],
        offset: usize,
    ) {
        for file_entry in files {
            let ft = file_entry["file_type"].as_str().unwrap_or("");
            if let Some(&idx) = self.ft_lookup.get(ft) {
                vec[offset + idx] = 1.0;
            }
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
struct FindingSummary {
    sample_paths: HashMap<String, u32>,
    filtered_finding_count: u32,
    notable_finding_count: u32,
    suspicious_finding_count: u32,
    hostile_finding_count: u32,
    unique_suspicious_ids: usize,
    unique_hostile_ids: usize,
    third_party_max_crit: u32,
    third_party_count: u32,
    well_known_max_crit: u32,
    well_known_hostile: u32,
    well_known_suspicious: u32,
    has_yara: bool,
}

/// Summarize findings from a single file entry.
fn summarize_findings(findings: &[&serde_json::Value]) -> FindingSummary {
    let mut summary = FindingSummary::default();
    let mut suspicious_ids: HashSet<&str> = HashSet::new();
    let mut hostile_ids: HashSet<&str> = HashSet::new();

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

        if crit_ord >= 3 {
            summary.notable_finding_count += 1;
        }
        if crit_ord >= 4 {
            summary.suspicious_finding_count += 1;
            suspicious_ids.insert(fid);
        }
        if crit_ord >= 5 {
            summary.hostile_finding_count += 1;
            hostile_ids.insert(fid);
        }

        let top = fid.split('/').next().unwrap_or("");
        match top {
            "third_party" => {
                summary.third_party_count += 1;
                summary.third_party_max_crit = summary.third_party_max_crit.max(crit_ord);
                if fid.starts_with("third_party/yara") {
                    summary.has_yara = true;
                }
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
            let entry = summary.sample_paths.entry(path.to_owned()).or_insert(0);
            *entry = (*entry).max(crit_ord);
        }
    }

    summary.unique_suspicious_ids = suspicious_ids.len();
    summary.unique_hostile_ids = hostile_ids.len();
    summary
}

/// Aggregate findings across every file in the report (v15 default aggregation).
fn summarize_report_files(files: &[&serde_json::Value]) -> FindingSummary {
    let mut combined = FindingSummary::default();
    let mut all_suspicious_ids: HashSet<String> = HashSet::new();
    let mut all_hostile_ids: HashSet<String> = HashSet::new();

    for file_entry in files {
        let findings: Vec<&serde_json::Value> = file_entry["findings"]
            .as_array()
            .map(|a| a.iter().collect())
            .unwrap_or_default();
        let file_summary = summarize_findings(&findings);

        combined.filtered_finding_count += file_summary.filtered_finding_count;
        combined.notable_finding_count += file_summary.notable_finding_count;
        combined.suspicious_finding_count += file_summary.suspicious_finding_count;
        combined.hostile_finding_count += file_summary.hostile_finding_count;
        combined.third_party_count += file_summary.third_party_count;
        combined.well_known_hostile += file_summary.well_known_hostile;
        combined.well_known_suspicious += file_summary.well_known_suspicious;
        combined.third_party_max_crit = combined
            .third_party_max_crit
            .max(file_summary.third_party_max_crit);
        combined.well_known_max_crit = combined
            .well_known_max_crit
            .max(file_summary.well_known_max_crit);
        combined.has_yara = combined.has_yara || file_summary.has_yara;

        for (path, max_ord) in &file_summary.sample_paths {
            let entry = combined.sample_paths.entry(path.clone()).or_insert(0);
            *entry = (*entry).max(*max_ord);
        }

        // Collect unique IDs across all files for dedup.
        for finding in &findings {
            let fid = finding["id"].as_str().unwrap_or("");
            if fid.is_empty() {
                continue;
            }
            let conf = finding["conf"].as_f64().unwrap_or(1.0);
            if conf < MIN_CONFIDENCE {
                continue;
            }
            let crit_ord = crit_ordinal(finding["crit"].as_str().unwrap_or("baseline"));
            if crit_ord >= 4 {
                all_suspicious_ids.insert(fid.to_owned());
            }
            if crit_ord >= 5 {
                all_hostile_ids.insert(fid.to_owned());
            }
        }
    }

    combined.unique_suspicious_ids = all_suspicious_ids.len();
    combined.unique_hostile_ids = all_hostile_ids.len();
    combined
}

/// Per-file risk statistics for top-k aggregation.
struct FileRiskStats {
    suspicious_ratio: f32,
    hostile_ratio: f32,
    suspicious_findings: u32,
    hostile_findings: u32,
}

fn file_risk_stats(file_entry: &serde_json::Value) -> FileRiskStats {
    let findings: Vec<&serde_json::Value> = file_entry["findings"]
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    let summary = summarize_findings(&findings);
    let denom = summary.filtered_finding_count.max(1) as f32;
    FileRiskStats {
        suspicious_ratio: summary.suspicious_finding_count as f32 / denom,
        hostile_ratio: summary.hostile_finding_count as f32 / denom,
        suspicious_findings: summary.suspicious_finding_count,
        hostile_findings: summary.hostile_finding_count,
    }
}

fn topk_file_risk_features(files: &[&serde_json::Value]) -> (f32, f32, f32, f32) {
    if files.is_empty() || TOP_K_RISK_FILES == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut stats: Vec<FileRiskStats> = files.iter().map(|f| file_risk_stats(f)).collect();

    // Top by suspicious ratio.
    stats.sort_by(|a, b| {
        (
            b.suspicious_ratio,
            b.suspicious_findings,
            b.hostile_ratio,
            b.hostile_findings,
        )
            .partial_cmp(&(
                a.suspicious_ratio,
                a.suspicious_findings,
                a.hostile_ratio,
                a.hostile_findings,
            ))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_susp: Vec<&FileRiskStats> = stats.iter().take(TOP_K_RISK_FILES).collect();
    let susp_ratio_sum: f32 = top_susp.iter().map(|s| s.suspicious_ratio).sum();
    let susp_findings_log =
        (top_susp.iter().map(|s| s.suspicious_findings).sum::<u32>() as f32 + 1.0).ln();

    // Top by hostile ratio.
    stats.sort_by(|a, b| {
        (
            b.hostile_ratio,
            b.hostile_findings,
            b.suspicious_ratio,
            b.suspicious_findings,
        )
            .partial_cmp(&(
                a.hostile_ratio,
                a.hostile_findings,
                a.suspicious_ratio,
                a.suspicious_findings,
            ))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_host: Vec<&FileRiskStats> = stats.iter().take(TOP_K_RISK_FILES).collect();
    let host_ratio_sum: f32 = top_host.iter().map(|s| s.hostile_ratio).sum();
    let host_findings_log =
        (top_host.iter().map(|s| s.hostile_findings).sum::<u32>() as f32 + 1.0).ln();

    (
        susp_ratio_sum,
        host_ratio_sum,
        susp_findings_log,
        host_findings_log,
    )
}

fn write_aggregate_features(
    summary: &FindingSummary,
    files: &[&serde_json::Value],
    vec: &mut [f32],
    offset: usize,
) {
    let sample_paths = &summary.sample_paths;
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

    // Original 8 breadth/concentration features.
    vec[offset] = max_crit as f32;
    vec[offset + 1] = categories.len() as f32;
    vec[offset + 2] = (path_breadth_any as f32 + 1.0).ln();
    vec[offset + 3] = (total_active as f32 + 1.0).ln();
    vec[offset + 4] = breadth_suspicious as f32 / path_breadth_any.max(1) as f32;
    vec[offset + 5] = breadth_hostile as f32 / path_breadth_any.max(1) as f32;
    vec[offset + 6] = breadth_suspicious as f32 / breadth_notable.max(1) as f32;
    vec[offset + 7] = breadth_notable_only as f32 / breadth_notable.max(1) as f32;

    // Base aggregate features.
    vec[offset + 8] = (summary.notable_finding_count as f32 + 1.0).ln();
    vec[offset + 9] = (summary.suspicious_finding_count as f32 + 1.0).ln();
    vec[offset + 10] = (summary.hostile_finding_count as f32 + 1.0).ln();
    let filtered = summary.filtered_finding_count.max(1) as f32;
    vec[offset + 11] = summary.notable_finding_count as f32 / filtered;
    vec[offset + 12] = summary.suspicious_finding_count as f32 / filtered;
    vec[offset + 13] = summary.hostile_finding_count as f32 / filtered;
    vec[offset + 14] = (summary.unique_suspicious_ids as f32 + 1.0).ln();
    vec[offset + 15] = (summary.unique_hostile_ids as f32 + 1.0).ln();
    let (susp_ratio, host_ratio, susp_log, host_log) = topk_file_risk_features(files);
    vec[offset + 16] = susp_ratio;
    vec[offset + 17] = host_ratio;
    vec[offset + 18] = susp_log;
    vec[offset + 19] = host_log;
    vec[offset + 20] = breadth_hostile as f32 / breadth_notable.max(1) as f32;
    vec[offset + 21] = breadth_hostile as f32 / breadth_suspicious.max(1) as f32;
    vec[offset + 22] = summary.suspicious_finding_count as f32 / summary.notable_finding_count.max(1) as f32;
    vec[offset + 23] = summary.hostile_finding_count as f32 / summary.notable_finding_count.max(1) as f32;
    vec[offset + 24] = summary.hostile_finding_count as f32 / summary.suspicious_finding_count.max(1) as f32;
}

fn write_external_summary_features(summary: &FindingSummary, vec: &mut [f32], offset: usize) {
    vec[offset] = summary.third_party_max_crit as f32;
    vec[offset + 1] = (summary.third_party_count as f32 + 1.0).ln();
    vec[offset + 2] = summary.well_known_max_crit as f32;
    vec[offset + 3] = summary.well_known_hostile as f32;
    vec[offset + 4] = summary.well_known_suspicious as f32;
    vec[offset + 5] = if summary.has_yara { 1.0 } else { 0.0 };
}

/// Merge per-file metrics into report-level maxima.
fn merge_metric_values(files: &[&serde_json::Value]) -> serde_json::Value {
    let mut merged: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for file_entry in files {
        let metrics = match file_entry.get("metrics") {
            Some(m) if m.is_object() => m,
            _ => continue,
        };
        let Some(metrics_obj) = metrics.as_object() else {
            continue;
        };
        for (group, fields) in metrics_obj {
            let Some(fields_obj) = fields.as_object() else {
                continue;
            };
            let group_map = merged.entry(group.clone()).or_default();
            for (fname, raw_value) in fields_obj {
                let val = raw_value.as_f64().unwrap_or(0.0);
                let entry = group_map.entry(fname.clone()).or_insert(f64::NEG_INFINITY);
                if val > *entry {
                    *entry = val;
                }
            }
        }
    }
    serde_json::to_value(&merged).unwrap_or(serde_json::Value::Null)
}

fn write_metric_features(metrics: &serde_json::Value, vec: &mut [f32], mut offset: usize) {
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
    files: &[&serde_json::Value],
    filtered_finding_count: u32,
) {
    let binary_like = ["pe", "elf", "macho"];
    let mut any_tiny_binary = false;
    let mut import_candidates = 0u32;
    let mut importless_candidates = 0u32;
    let mut max_entropy: f64 = 0.0;

    for file_entry in files {
        let ft = file_entry["file_type"].as_str().unwrap_or("");
        let size = file_entry["size"].as_u64().unwrap_or(0);
        if binary_like.contains(&ft) && size < 20_000 {
            any_tiny_binary = true;
        }
        if file_entry.get("imports").is_some() {
            import_candidates += 1;
            let imports_empty = file_entry["imports"].as_array().is_none_or(Vec::is_empty);
            if imports_empty {
                importless_candidates += 1;
            }
        }
        // Track max binary entropy for stealth_potential.
        let entropy = file_entry
            .get("metrics")
            .and_then(|m| m.get("binary"))
            .and_then(|b| b.get("overall_entropy"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        if entropy > max_entropy {
            max_entropy = entropy;
        }
    }

    vec[offset] = if any_tiny_binary { 1.0 } else { 0.0 };
    vec[offset + 1] = if import_candidates > 0 && importless_candidates == import_candidates {
        1.0
    } else {
        0.0
    };
    vec[offset + 2] = if filtered_finding_count == 0 {
        1.0
    } else {
        0.0
    };
    vec[offset + 3] = (filtered_finding_count as f32 + 1.0).ln();
    let file_count = files.len() as f32;
    vec[offset + 4] = (file_count + 1.0).ln();
    vec[offset + 5] = ((file_count - 1.0).max(0.0) + 1.0).ln();
    // stealth_potential: high entropy (packed/encrypted) with very few findings.
    vec[offset + 6] = if filtered_finding_count < 5 && max_entropy > 6.5 {
        1.0
    } else {
        0.0
    };
}

/// Return all file entries from a v3 report.
fn report_files(report: &serde_json::Value) -> Vec<&serde_json::Value> {
    report["files"]
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
    use std::io::Write;

    fn paths(id: &str) -> Vec<&str> {
        finding_paths(id).collect()
    }

    fn valid_feature_names(presence_vocab: &[String], filetype_vocab: &[String]) -> String {
        build_expected_feature_names(presence_vocab, filetype_vocab)
            .into_iter()
            .map(|name| format!("\"{name}\""))
            .collect::<Vec<_>>()
            .join(",")
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
            version: 15,
            abi_version: 15,
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
            version: 15,
            abi_version: 15,
            presence_vocab: vec!["objectives/evasion/process".to_string()],
            filetype_vocab: vec!["sh".to_string()],
            feature_names: vec![],
            // 1 presence + 1 maxcrit + 25 agg + 6 ext + 16 metrics + 1 filetype + 7 struct
            total_features: 1 + 1 + 25 + 6 + 16 + 1 + 7,
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
        let ft_offset = 1 + 1 + 25 + 6 + 16;
        assert_eq!(features[ft_offset], 1.0);
        // zero_findings structural feature
        let struct_offset = ft_offset + 1;
        assert_eq!(features[struct_offset + 2], 1.0);
        // file_count_log = log1p(1) for single file
        assert!((features[struct_offset + 4] - (2.0f32).ln()).abs() < 1e-6);
        // inner_file_count_log = log1p(0) = 0 for single file
        assert_eq!(features[struct_offset + 5], 0.0);
    }

    #[test]
    fn test_extract_with_findings() {
        let spec = FeatureSpec {
            version: 15,
            abi_version: 15,
            presence_vocab: vec![
                "objectives".to_string(),
                "objectives/evasion".to_string(),
                "objectives/evasion/process".to_string(),
            ],
            filetype_vocab: vec!["php".to_string()],
            feature_names: vec![],
            total_features: 3 + 3 + 25 + 6 + 16 + 1 + 7,
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

        // All 3 presence features should be 1.0 (hostile >= baseline)
        assert_eq!(features[0], 1.0); // present:objectives
        assert_eq!(features[1], 1.0); // present:objectives/evasion
        assert_eq!(features[2], 1.0); // present:objectives/evasion/process

        // All 3 maxcrit features should be 5.0 (hostile)
        assert_eq!(features[3], 5.0); // maxcrit:objectives
        assert_eq!(features[4], 5.0); // maxcrit:objectives/evasion
        assert_eq!(features[5], 5.0); // maxcrit:objectives/evasion/process

        // agg:max_crit = 5.0
        assert_eq!(features[6], 5.0);

        // agg:hostile_findings_log = log1p(1)
        assert!((features[6 + 10] - (2.0f32).ln()).abs() < 1e-6);
        // agg:hostile_finding_ratio = 1/1
        assert!((features[6 + 13] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_filtering() {
        let spec = FeatureSpec {
            version: 15,
            abi_version: 15,
            presence_vocab: vec!["objectives".to_string()],
            filetype_vocab: vec![],
            feature_names: vec![],
            total_features: 1 + 1 + 25 + 6 + 16 + 7,
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

    #[test]
    fn test_multi_file_aggregation() {
        let spec = FeatureSpec {
            version: 15,
            abi_version: 15,
            presence_vocab: vec![
                "objectives".to_string(),
                "objectives/evasion".to_string(),
                "metadata".to_string(),
                "metadata/format".to_string(),
            ],
            filetype_vocab: vec!["pe".to_string(), "sh".to_string()],
            feature_names: vec![],
            total_features: 4 + 4 + 25 + 6 + 16 + 2 + 7,
            feature_means: None,
            feature_stds: None,
            standardized: false,
        };
        let ctx = ExtractContext::new(&spec);
        let report = serde_json::json!({
            "files": [
                {
                    "file_type": "pe",
                    "size": 1000,
                    "findings": [
                        {"id": "objectives/evasion/process::test", "crit": "hostile", "conf": 0.9},
                    ],
                    "metrics": {"binary": {"overall_entropy": 7.5}},
                },
                {
                    "file_type": "sh",
                    "size": 200,
                    "findings": [
                        {"id": "metadata/format::no-functions", "crit": "notable", "conf": 0.8},
                    ],
                    "metrics": {"binary": {"overall_entropy": 3.0}},
                },
            ]
        });
        let features = ctx.extract(&report);

        // Both file types should be set (multi-hot)
        let ft_offset = 4 + 4 + 25 + 6 + 16;
        assert_eq!(features[ft_offset], 1.0); // pe
        assert_eq!(features[ft_offset + 1], 1.0); // sh

        // Presence from both files should be merged
        assert_eq!(features[0], 1.0); // present:objectives (from file 1)
        assert_eq!(features[1], 1.0); // present:objectives/evasion (from file 1)
        assert_eq!(features[2], 1.0); // present:metadata (from file 2)
        assert_eq!(features[3], 1.0); // present:metadata/format (from file 2)

        // Metrics should take max: binary.overall_entropy = max(7.5, 3.0) = 7.5
        let metrics_offset = 4 + 4 + 25 + 6;
        assert!((features[metrics_offset] - 7.5).abs() < 1e-6);

        // file_count_log = log1p(2)
        let struct_offset = ft_offset + 2;
        assert!((features[struct_offset + 4] - (3.0f32).ln()).abs() < 1e-6);
        // inner_file_count_log = log1p(1)
        assert!((features[struct_offset + 5] - (2.0f32).ln()).abs() < 1e-6);
    }

    #[test]
    fn test_has_yara_only_for_yara_prefix() {
        let spec = FeatureSpec {
            version: 15,
            abi_version: 15,
            presence_vocab: vec!["third_party".to_string()],
            filetype_vocab: vec![],
            feature_names: vec![],
            total_features: 1 + 1 + 25 + 6 + 16 + 7,
            feature_means: None,
            feature_stds: None,
            standardized: false,
        };
        let ctx = ExtractContext::new(&spec);

        // third_party but not yara -> has_yara should be false
        let report = serde_json::json!({
            "files": [{
                "file_type": "pe",
                "size": 100,
                "findings": [
                    {"id": "third_party/clamav::match", "crit": "hostile", "conf": 0.9},
                ],
                "metrics": {},
            }]
        });
        let features = ctx.extract(&report);
        let ext_offset = 1 + 1 + 25;
        assert_eq!(features[ext_offset + 5], 0.0); // has_yara = false

        // third_party/yara -> has_yara should be true
        let report = serde_json::json!({
            "files": [{
                "file_type": "pe",
                "size": 100,
                "findings": [
                    {"id": "third_party/yara/rule::match", "crit": "hostile", "conf": 0.9},
                ],
                "metrics": {},
            }]
        });
        let features = ctx.extract(&report);
        assert_eq!(features[ext_offset + 5], 1.0); // has_yara = true
    }

    #[test]
    fn test_load_rejects_missing_feature_names() -> Result<()> {
        let mut file = tempfile::NamedTempFile::new()?;
        writeln!(
            file,
            "{{\"version\":15,\"presence_vocab\":[\"objectives\"],\"filetype_vocab\":[\"sh\"],\"total_features\":51,\"standardized\":false}}"
        )?;

        let Err(err) = FeatureSpec::load(file.path()) else {
            anyhow::bail!("missing feature_names should be rejected");
        };
        assert!(err.to_string().contains("feature_names length"));
        Ok(())
    }

    #[test]
    fn test_load_rejects_unexpected_feature_layout() -> Result<()> {
        let presence_vocab = vec!["objectives".to_string()];
        let filetype_vocab = vec!["sh".to_string()];
        let mut feature_names = build_expected_feature_names(&presence_vocab, &filetype_vocab);
        let Some(last_feature_name) = feature_names.last_mut() else {
            anyhow::bail!("expected non-empty feature list");
        };
        *last_feature_name = "struct:wrong_name".to_string();
        let total_features = feature_names.len();
        let mut file = tempfile::NamedTempFile::new()?;
        writeln!(
            file,
            "{{\"version\":15,\"abi_version\":15,\"presence_vocab\":[\"objectives\"],\"filetype_vocab\":[\"sh\"],\"feature_names\":[{}],\"total_features\":{},\"standardized\":false}}",
            feature_names
                .into_iter()
                .map(|name| format!("\"{name}\""))
                .collect::<Vec<_>>()
                .join(","),
            total_features
        )?;

        let Err(err) = FeatureSpec::load(file.path()) else {
            anyhow::bail!("unexpected feature layout should be rejected");
        };
        assert!(err.to_string().contains("expected v15 layout"));
        Ok(())
    }

    #[test]
    fn test_load_rejects_mismatched_spec_version_with_update_guidance() -> Result<()> {
        let mut file = tempfile::NamedTempFile::new()?;
        writeln!(
            file,
            "{{\"version\":14,\"abi_version\":14,\"presence_vocab\":[\"objectives\"],\"filetype_vocab\":[\"sh\"],\"feature_names\":[],\"total_features\":0,\"standardized\":false}}"
        )?;

        let Err(err) = FeatureSpec::load(file.path()) else {
            anyhow::bail!("mismatched spec version should be rejected");
        };
        let message = err.to_string();
        assert!(message.contains("feature spec version mismatch"));
        assert!(message.contains("Run 'litmus update-rules'"));
        Ok(())
    }

    #[test]
    fn test_load_rejects_mismatched_standardization_stats() -> Result<()> {
        let presence_vocab = vec!["objectives".to_string()];
        let filetype_vocab = vec!["sh".to_string()];
        let feature_names = valid_feature_names(&presence_vocab, &filetype_vocab);
        let total_features = build_expected_feature_names(&presence_vocab, &filetype_vocab).len();
        let mut file = tempfile::NamedTempFile::new()?;
        writeln!(
            file,
            "{{\"version\":15,\"abi_version\":15,\"presence_vocab\":[\"objectives\"],\"filetype_vocab\":[\"sh\"],\"feature_names\":[{}],\"total_features\":{},\"feature_means\":[0.0],\"feature_stds\":[1.0],\"standardized\":true}}",
            feature_names,
            total_features
        )?;

        let Err(err) = FeatureSpec::load(file.path()) else {
            anyhow::bail!("mismatched standardization stats should be rejected");
        };
        assert!(err.to_string().contains("standardization stats"));
        Ok(())
    }
}
