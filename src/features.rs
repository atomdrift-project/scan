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
pub const EXPECTED_SPEC_VERSION: u32 = 16;
/// Stable model ABI version shared with collimator.
/// Keep this in sync with EXPECTED_SPEC_VERSION for a single compatibility number.
pub const EXPECTED_MODEL_ABI_VERSION: u32 = EXPECTED_SPEC_VERSION;

/// Minimum finding confidence for inclusion (matches collimator MIN_CONFIDENCE).
const MIN_CONFIDENCE: f64 = 0.65;

/// Number of riskiest files to summarize for top-k aggregate features.
const TOP_K_RISK_FILES: usize = 1;

/// Read v4 criticality ordinal from a finding JSON value.
/// v4 crit is already an integer: 0=filtered, 1=component, 2=baseline, 3=notable, 4=suspicious, 5=hostile.
pub(crate) fn crit_ordinal(finding: &serde_json::Value) -> u32 {
    finding["l"].as_u64().unwrap_or(0) as u32
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

/// Feature specification loaded from feature_spec.json (v16).
#[derive(Debug, Clone)]
pub struct FeatureSpec {
    /// Spec format version (expected: 16).
    version: u32,
    /// Stable preprocessing/inference ABI version.
    abi_version: u32,
    /// Path vocabulary for presence/maxcrit features.
    presence_vocab: Vec<String>,
    /// File type vocabulary for multi-hot encoding.
    filetype_vocab: Vec<String>,
    /// Element vocabulary (Group 8: elements multi-hot).
    element_vocab: Vec<String>,
    /// Bigram vocabulary (Group 11: bigram + Group 20: unsigned_bigram multi-hot).
    bigram_vocab: Vec<String>,
    /// Ghost vocabulary (Group 12: ghost multi-hot).
    ghost_vocab: Vec<String>,
    /// Skeleton vocabulary (Group 13).
    skeleton_vocab: Vec<String>,
    /// Rare element vocabulary (Group 14).
    rare_element_vocab: Vec<String>,
    /// Trigram vocabulary (Group 16).
    trigram_vocab: Vec<String>,
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
    element_vocab: Vec<String>,
    #[serde(default)]
    bigram_vocab: Vec<String>,
    #[serde(default)]
    ghost_vocab: Vec<String>,
    #[serde(default)]
    skeleton_vocab: Vec<String>,
    #[serde(default)]
    rare_element_vocab: Vec<String>,
    #[serde(default)]
    trigram_vocab: Vec<String>,
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
            element_vocab: raw.element_vocab,
            bigram_vocab: raw.bigram_vocab,
            ghost_vocab: raw.ghost_vocab,
            skeleton_vocab: raw.skeleton_vocab,
            rare_element_vocab: raw.rare_element_vocab,
            trigram_vocab: raw.trigram_vocab,
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

        let expected_feature_names = build_expected_feature_names(
            &self.presence_vocab,
            &self.filetype_vocab,
            &self.element_vocab,
            &self.bigram_vocab,
            &self.ghost_vocab,
            &self.skeleton_vocab,
            &self.rare_element_vocab,
            &self.trigram_vocab,
        );
        if self.feature_names != expected_feature_names {
            // Find first mismatch to make debugging easier.
            let first_mismatch = self
                .feature_names
                .iter()
                .zip(expected_feature_names.iter())
                .enumerate()
                .find(|(_, (a, b))| a != b);
            anyhow::bail!(
                "feature spec feature_names do not match the expected v{EXPECTED_SPEC_VERSION} layout: \
                 spec has {} features, expected {}; first mismatch: {:?}",
                self.feature_names.len(),
                expected_feature_names.len(),
                first_mismatch,
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

/// Logic gap categories (v16 group 19) — sorted to match Python.
const LOGIC_GAP_CATEGORIES: &[&str] = &["crypto", "network", "process"];

/// Intent gap categories (v16 group 22) — order matches Python.
const INTENT_GAP_CATEGORIES: &[&str] = &["network", "filesystem", "execution", "crypto"];

/// Expected ghosts (v16 group 23) — sorted by filetype.
/// Each entry is (filetype, [traits]).
const EXPECTED_GHOSTS: &[(&str, &[&str])] = &[
    ("elf", &[
        "metadata/binary/layout",
        "metadata/binary/metrics",
        "metadata/binary/symbols",
        "metadata/binary/linking",
    ]),
    ("javascript", &[
        "micro-behaviors/javascript/async",
        "metadata/package/versioning",
    ]),
    ("pe", &[
        "metadata/binary/layout",
        "metadata/binary/metrics",
        "metadata/binary/resource",
        "metadata/binary/symbols",
        "metadata/binary/linking",
    ]),
];

#[allow(clippy::too_many_arguments)]
fn build_expected_feature_names(
    presence_vocab: &[String],
    filetype_vocab: &[String],
    element_vocab: &[String],
    bigram_vocab: &[String],
    ghost_vocab: &[String],
    skeleton_vocab: &[String],
    rare_element_vocab: &[String],
    trigram_vocab: &[String],
) -> Vec<String> {
    let mut feature_names = Vec::with_capacity(
        presence_vocab.len() * 2
            + 50  // agg
            + 6   // ext
            + KEY_METRICS.len()
            + filetype_vocab.len()
            + 24  // struct (extended in v16)
            + element_vocab.len()
            + 3   // formula
            + 2 + filetype_vocab.len()  // score + inter:*score
            + bigram_vocab.len()  // bigrams
            + ghost_vocab.len()
            + skeleton_vocab.len()
            + rare_element_vocab.len()
            + trigram_vocab.len()
            + LOGIC_GAP_CATEGORIES.len()
            + bigram_vocab.len()  // unsigned_bigram
            + INTENT_GAP_CATEGORIES.len()
            + 11, // missing (negative space)
    );

    // Group 1: present
    for path in presence_vocab {
        feature_names.push(format!("present:{path}"));
    }

    // Group 2: maxcrit
    for path in presence_vocab {
        feature_names.push(format!("maxcrit:{path}"));
    }

    // Group 3: agg (50 features in v16)
    // Original 20 (always present)
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
    ]);
    // suspicious_breadth_density (default ON in v16)
    feature_names.extend([
        "agg:suspicious_category_breadth".to_string(),
        "agg:hostile_category_breadth".to_string(),
        "agg:suspicious_category_density".to_string(),
        "agg:hostile_category_density".to_string(),
        "agg:suspicious_findings_per_kb".to_string(),
        "agg:hostile_findings_per_kb".to_string(),
        "agg:suspicious_categories_per_kb".to_string(),
        "agg:hostile_categories_per_kb".to_string(),
        format!("agg:top{TOP_K_RISK_FILES}_file_suspicious_density_sum"),
        format!("agg:top{TOP_K_RISK_FILES}_file_hostile_density_sum"),
        format!("agg:top{TOP_K_RISK_FILES}_file_suspicious_category_breadth_sum"),
        format!("agg:top{TOP_K_RISK_FILES}_file_hostile_category_breadth_sum"),
    ]);
    // hostile_escalation (default ON)
    feature_names.extend([
        "agg:hostile_escalation_rate".to_string(),
        "agg:hostile_share_of_suspicious".to_string(),
        "agg:suspicious_finding_escalation_rate".to_string(),
        "agg:hostile_finding_escalation_rate".to_string(),
        "agg:hostile_share_of_suspicious_findings".to_string(),
    ]);
    // hostile_weighted_density (default ON)
    feature_names.extend([
        "agg:hostile_weighted_density".to_string(),
        format!("agg:top{TOP_K_RISK_FILES}_file_hostile_weighted_density_sum"),
    ]);
    // repetition_penalty (default ON)
    feature_names.extend([
        "agg:suspicious_id_repeat_ratio".to_string(),
        "agg:hostile_id_repeat_ratio".to_string(),
        "agg:suspicious_category_repeat_ratio".to_string(),
        "agg:hostile_category_repeat_ratio".to_string(),
    ]);
    // file_severity_distribution (default ON)
    feature_names.extend([
        "agg:file_hostile_fraction".to_string(),
        "agg:file_suspicious_fraction".to_string(),
        "agg:file_notable_fraction".to_string(),
        "agg:file_hostile_count_log".to_string(),
        "agg:file_suspicious_count_log".to_string(),
        "agg:file_notable_count_log".to_string(),
    ]);
    // hostile_depth_weight (inherits from extreme_features=ON)
    feature_names.push("agg:hostile_depth_weight".to_string());

    // Group 4: ext (6)
    feature_names.extend([
        "ext:third_party_max_crit".to_string(),
        "ext:third_party_count".to_string(),
        "ext:well_known_max_crit".to_string(),
        "ext:well_known_hostile_count".to_string(),
        "ext:well_known_suspicious_count".to_string(),
        "ext:has_yara_match".to_string(),
    ]);

    // Group 5: metrics (16)
    for &(group, field_name, _) in KEY_METRICS {
        feature_names.push(format!("metrics:{group}_{field_name}"));
    }

    // Group 6: filetype
    for filetype in filetype_vocab {
        feature_names.push(format!("filetype:{filetype}"));
    }

    // Group 7: struct base 7
    feature_names.extend([
        "struct:tiny_executable".to_string(),
        "struct:no_imports".to_string(),
        "struct:zero_findings".to_string(),
        "struct:finding_count_log".to_string(),
        "struct:file_count_log".to_string(),
        "struct:inner_file_count_log".to_string(),
        "struct:stealth_potential".to_string(),
    ]);
    // struct_file_risk_coverage (default ON)
    feature_names.extend([
        "struct:suspicious_file_fraction".to_string(),
        "struct:hostile_file_fraction".to_string(),
        "struct:suspicious_file_count_log".to_string(),
        "struct:hostile_file_count_log".to_string(),
    ]);

    // Group 8: elements (no inter:*element in v16 — filetype_interactions=0)
    for el in element_vocab {
        feature_names.push(format!("elements:{el}"));
    }

    // Group 9: formula
    feature_names.extend([
        "formula:skeleton_len".to_string(),
        "formula:unique_elements".to_string(),
        "formula:complexity_ratio".to_string(),
    ]);

    // Group 10: score + inter:*score (always ON in v16)
    feature_names.extend([
        "score:hopper_score".to_string(),
        "score:density".to_string(),
    ]);
    for ft in filetype_vocab {
        feature_names.push(format!("inter:{ft}*score"));
    }

    // Group 11: bigrams
    for bi in bigram_vocab {
        feature_names.push(format!("bigrams:{bi}"));
    }

    // Group 12: ghost
    for gh in ghost_vocab {
        feature_names.push(format!("ghost:{gh}"));
    }

    // Group 13: skeleton (no inter:*skeleton in v16)
    for skel in skeleton_vocab {
        feature_names.push(format!("skeleton:{skel}"));
    }

    // Group 14: rare elements
    for el in rare_element_vocab {
        feature_names.push(format!("rare:{el}"));
    }

    // Group 15: structural extensions
    feature_names.push("struct:packaged_capability".to_string());
    feature_names.extend([
        "struct:mtime_range_hours".to_string(),
        "struct:mtime_std_dev_hours".to_string(),
        "struct:max_nesting_depth_log".to_string(),
        "struct:inner_file_ratio".to_string(),
        "struct:entropy_std_dev".to_string(),
        "struct:entropy_max_diff".to_string(),
    ]);
    // silent_packer_signal=0 in v16 (skip)
    // mtime_kurtosis=0 in v16 (skip)
    // air_gap_signal=1 in v16
    feature_names.push("struct:air_gap_signal".to_string());
    // anachronistic_injection inherits from extreme_features=1
    feature_names.push("struct:anachronistic_injection".to_string());
    // code_entropy_spike inherits from extreme_features=1
    feature_names.push("struct:code_entropy_spike".to_string());
    // foreign_binary_signal inherits from extreme_features=1
    feature_names.push("struct:foreign_binary_signal".to_string());
    // extension_mismatch_signal inherits from extreme_features=1
    feature_names.push("struct:extension_mismatch_signal".to_string());
    // hostile_finding_density inherits from extreme_features=1
    feature_names.push("struct:hostile_finding_density".to_string());

    // Group 16: trigrams
    for tri in trigram_vocab {
        feature_names.push(format!("trigram:{tri}"));
    }

    // Group 19: logic gaps (sorted by category)
    for cat in LOGIC_GAP_CATEGORIES {
        feature_names.push(format!("gap:{cat}"));
    }

    // Group 20: signature synergy (unsigned bigrams)
    for bi in bigram_vocab {
        feature_names.push(format!("unsigned_bigram:{bi}"));
    }

    // Group 21: clusters (DISABLED in v16)

    // Group 22: intent gaps
    for cat in INTENT_GAP_CATEGORIES {
        feature_names.push(format!("intent_gap:{cat}"));
    }

    // Group 23: negative space (missing:{ftype}*{trait})
    for &(ftype, traits) in EXPECTED_GHOSTS {
        for trait_path in traits {
            feature_names.push(format!("missing:{ftype}*{trait_path}"));
        }
    }

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
    // v16 vocab sizes (used for cursor advancement).
    n_element: usize,
    n_bigram: usize,
    n_ghost: usize,
    n_skeleton: usize,
    n_rare: usize,
    n_trigram: usize,
    // v16 vocab lookups: each maps a vocab string to its index *within the
    // group's slot range* (not the absolute index in the feature vector).
    // The group's offset from extract_into is added at write time.
    element_lookup: HashMap<String, usize>,
    bigram_lookup: HashMap<String, usize>,
    ghost_lookup: HashMap<String, usize>,
    skeleton_lookup: HashMap<String, usize>,
    rare_lookup: HashMap<String, usize>,
    trigram_lookup: HashMap<String, usize>,
    /// Ghost vocabulary copied from spec; iteration order matters for the
    /// ghost feature group (multi-hot in vocab order).
    ghost_vocab: Vec<String>,
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

        let element_lookup: HashMap<String, usize> = spec
            .element_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let bigram_lookup: HashMap<String, usize> = spec
            .bigram_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let ghost_lookup: HashMap<String, usize> = spec
            .ghost_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let skeleton_lookup: HashMap<String, usize> = spec
            .skeleton_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let rare_lookup: HashMap<String, usize> = spec
            .rare_element_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let trigram_lookup: HashMap<String, usize> = spec
            .trigram_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        Self {
            presence_lookup,
            n_presence: spec.presence_vocab.len(),
            ft_lookup,
            n_ft: spec.filetype_vocab.len(),
            n_element: spec.element_vocab.len(),
            n_bigram: spec.bigram_vocab.len(),
            n_ghost: spec.ghost_vocab.len(),
            n_skeleton: spec.skeleton_vocab.len(),
            n_rare: spec.rare_element_vocab.len(),
            n_trigram: spec.trigram_vocab.len(),
            element_lookup,
            bigram_lookup,
            ghost_lookup,
            skeleton_lookup,
            rare_lookup,
            trigram_lookup,
            ghost_vocab: spec.ghost_vocab.clone(),
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

        // Canonical fields computed once and reused across groups.
        let (formula_str, elements_str, sample_score) = canonical_fields_from_report(report);
        // score_weight = log1p(score) when score > 0, else 1.0 — matches
        // collimator's _apply_presence_features / _apply_maxcrit_features.
        let score_weight: f32 = if sample_score > 0 {
            (sample_score as f32).ln_1p()
        } else {
            1.0
        };

        // v16 layout — order MUST mirror collimator/features.py:_build_feature_names.

        // Group 1: Presence (n_presence). Python's _apply_presence_features
        // writes `score_weight * path_confidence` for each path with
        // max_ord >= 2. include_soft_presence is ON by default in v16.
        let presence_offset = offsets.take(self.n_presence);
        self.write_presence_features_v16(&summary, vec, presence_offset, score_weight);

        // Group 2: Max criticality. Python writes
        // `max_ord * score_weight * path_confidence` for every path.
        let maxcrit_offset = offsets.take(self.n_presence);
        self.write_max_crit_features_v16(&summary, vec, maxcrit_offset, score_weight);

        // Group 3: Aggregates (50 in v16)
        // The first 20 are "base" agg features that v15 handled. write_aggregate_features
        // currently writes 25 contiguous values (20 base + 5 hostile_escalation), but in
        // v16 those 25 are NOT contiguous (suspicious_breadth_density is inserted between
        // them). For now we let it write the 20 base correctly and zero the rest.
        // TODO Phase 2/3: rewrite write_aggregate_features for the v16 split layout.
        let agg_offset = offsets.take(50);
        write_aggregate_features(&summary, &files, vec, agg_offset);

        // Group 4: External (6) — OK
        let ext_offset = offsets.take(6);
        write_external_summary_features(&summary, vec, ext_offset);

        // Group 5: Metrics (16) — OK
        let metrics_offset = offsets.take(KEY_METRICS.len());
        write_metric_features(&merged_metrics, vec, metrics_offset);

        // Group 6: Filetype (n_ft).
        // v16 default: COLLIMATOR_BLINDFOLD=1 → Python's _apply_filetype_features
        // is fully skipped (the entire loop is gated by `if not ctx.blindfold`),
        // so all 72 filetype:* features stay zero. Rust must mirror that to match.
        // TODO: serialize the relevant config flags into feature_spec.json so
        //       litmus doesn't have to hardcode the v16 default.
        let _file_type_offset = offsets.take(self.n_ft);

        // Group 7: Structural base 7 + struct_file_risk_coverage 4 = 11
        // Existing write_structural_features handles only the first 7. The 4
        // file-risk-coverage features (suspicious_file_fraction etc.) are zero.
        // TODO Phase 2/3: extend write_structural_features for the 4 extra.
        let structural_offset = offsets.take(11);
        write_structural_features(
            vec,
            structural_offset,
            &files,
            summary.filtered_finding_count,
        );

        // (canonical fields formula_str / elements_str / sample_score
        //  were already computed near the top of extract_into and are in scope.)

        // Group 8: elements (n_element) — multi-hot single-string lookup.
        // Python's `_apply_element_features` does `elements.split(",")` but the
        // database stores elements without commas, so the loop usually sees a
        // single string. Set the matching vocab entry to 1.0.
        let elements_offset = offsets.take(self.n_element);
        if !elements_str.is_empty() {
            for el in elements_str.split(',') {
                let el = el.trim();
                if let Some(&idx) = self.element_lookup.get(el) {
                    vec[elements_offset + idx] = 1.0;
                }
            }
        }

        // Group 9: formula (3 features). Python's _apply_formula_features.
        let formula_offset = offsets.take(3);
        let skeleton_str: String = formula_str.chars().filter(|c| c.is_alphabetic()).collect();
        let unique_skel_chars: std::collections::HashSet<char> = skeleton_str.chars().collect();
        vec[formula_offset] = skeleton_str.chars().count() as f32;
        vec[formula_offset + 1] = unique_skel_chars.len() as f32;
        if summary.filtered_finding_count > 0 {
            vec[formula_offset + 2] =
                formula_str.chars().count() as f32 / summary.filtered_finding_count as f32;
        }

        // Group 10: score (2) + inter:{ft}*score (n_ft).
        // Python writes `score:hopper_score`, `score:density`, then loops files
        // and writes inter:{ft}*score for each file's type. Multiple files of
        // the same type just overwrite the same slot with the same value.
        let score_offset = offsets.take(2 + self.n_ft);
        let total_size_score: f64 = files.iter().map(|f| f["sz"].as_f64().unwrap_or(0.0)).sum();
        vec[score_offset] = sample_score as f32;
        vec[score_offset + 1] = if total_size_score > 0.0 {
            sample_score as f32 / (total_size_score as f32).ln_1p()
        } else {
            0.0
        };
        // inter:{ft}*score: written by file-type position in filetype_vocab.
        // build_expected_feature_names emits the inter:*score slot in
        // filetype_vocab order, so the slot for ft i is at score_offset + 2 + i.
        for file_entry in &files {
            let ft = file_entry["type"].as_str().unwrap_or("");
            if let Some(&ft_idx) = self.ft_lookup.get(ft) {
                vec[score_offset + 2 + ft_idx] = sample_score as f32;
            }
        }

        // Group 11: bigrams (n_bigram). Python's _apply_bigram_features.
        let bigrams_offset = offsets.take(self.n_bigram);
        write_bigram_features(&files, vec, bigrams_offset, &self.bigram_lookup);

        // Group 12: ghost (n_ghost). Python's _apply_ghost_features writes 1.0
        // when the EXPECTED path is missing or below baseline crit (<2).
        let ghost_offset = offsets.take(self.n_ghost);
        for ghost_path in &self.ghost_vocab {
            let missing = match summary.sample_paths.get(ghost_path) {
                Some(&max_ord) => max_ord < 2,
                None => true,
            };
            if missing {
                if let Some(&idx) = self.ghost_lookup.get(ghost_path) {
                    vec[ghost_offset + idx] = 1.0;
                }
            }
        }

        // Group 13: skeleton (n_skeleton). Python sets skeleton:{skeleton} to 1.0
        // and skeleton:{ft}*{skeleton} interactions (gated by filetype_interactions
        // which is OFF in v16, so we only need the base lookup).
        let skeleton_offset = offsets.take(self.n_skeleton);
        if !skeleton_str.is_empty() {
            if let Some(&idx) = self.skeleton_lookup.get(skeleton_str.as_str()) {
                vec[skeleton_offset + idx] = 1.0;
            }
        }

        // Group 14: rare elements (n_rare). v16 SOFT_PRESENCE=1 → weight is the
        // mean of all post-filter finding confidences. If no findings, weight=1.0.
        let rare_offset = offsets.take(self.n_rare);
        if !elements_str.is_empty() {
            let weight: f32 = if summary.finding_confidences.is_empty() {
                1.0
            } else {
                let sum: f64 = summary.finding_confidences.iter().sum();
                (sum / summary.finding_confidences.len() as f64) as f32
            };
            for el in elements_str.split(',') {
                let el = el.trim();
                if let Some(&idx) = self.rare_lookup.get(el) {
                    vec[rare_offset + idx] = weight;
                }
            }
        }

        // Group 15: structural extensions (13 features). Mirrors collimator's
        // extended _apply_structural_features block (lines 1517-1553).
        let struct_ext_offset = offsets.take(13);
        write_structural_extensions(&files, &summary, vec, struct_ext_offset);

        // Group 16: trigrams (n_trigram). Python's _apply_trigram_features.
        let trigram_offset = offsets.take(self.n_trigram);
        write_trigram_features(&files, vec, trigram_offset, &self.trigram_lookup);

        // Group 19: logic gaps (3). Mirrors Python `_apply_logic_gap_features`.
        let gap_offset = offsets.take(LOGIC_GAP_CATEGORIES.len());
        write_logic_gap_features(&summary, &files, vec, gap_offset);

        // Group 20: signature synergy / unsigned_bigram (n_bigram). Python's
        // _apply_signature_synergy_features. Same vocab as bigrams; only writes
        // when "metadata/unsigned" is present in summary.sample_paths.
        let unsigned_bigram_offset = offsets.take(self.n_bigram);
        if summary.sample_paths.contains_key("metadata/unsigned") {
            // Reuse bigram_lookup since the vocab is identical.
            write_bigram_features(&files, vec, unsigned_bigram_offset, &self.bigram_lookup);
        }

        // Group 22: intent gaps (4). Python's _apply_intent_gap_features writes
        // 1.0 if the risky behavior is present and no doc/help intent. Order:
        // sorted(risky_behaviors.keys()) which is [crypto, execution, filesystem,
        // network] alphabetically. But the spec emits them in the FIXED order
        // INTENT_GAP_CATEGORIES = [network, filesystem, execution, crypto].
        // The slot index in the spec follows INTENT_GAP_CATEGORIES, so we
        // write each in INTENT_GAP_CATEGORIES order.
        let intent_gap_offset = offsets.take(INTENT_GAP_CATEGORIES.len());
        write_intent_gap_features(&summary, vec, intent_gap_offset);

        // Group 23: negative space / missing (11). Python's _apply_neg_space_features.
        // Sets missing:{ftype}*{trait} to 1.0 if the report contains a file of
        // the matching ftype AND the expected trait is absent from sample_paths.
        let missing_count: usize = EXPECTED_GHOSTS.iter().map(|(_, t)| t.len()).sum();
        let missing_offset = offsets.take(missing_count);
        write_negative_space_features(&summary, &files, vec, missing_offset);

        // Sanity check: cursor should match total_features.
        debug_assert_eq!(
            offsets.offset, self.total_features,
            "v16 cursor walk produced {} slots, spec says {}",
            offsets.offset, self.total_features,
        );
    }

    /// v16 presence features: write `score_weight * path_confidence` for each
    /// path with max_ord >= 2. Mirrors collimator's _apply_presence_features
    /// with include_score_weighted_traits=ON and include_soft_presence=ON
    /// (the v16 defaults).
    fn write_presence_features_v16(
        &self,
        summary: &FindingSummary,
        vec: &mut [f32],
        offset: usize,
        score_weight: f32,
    ) {
        for (path, &max_ord) in &summary.sample_paths {
            if max_ord < 2 {
                continue;
            }
            if let Some(&idx) = self.presence_lookup.get(path.as_str()) {
                let conf = summary
                    .path_confidences
                    .get(path)
                    .copied()
                    .unwrap_or(1.0) as f32;
                vec[offset + idx] = score_weight * conf;
            }
        }
    }

    /// v16 maxcrit features: write `max_ord * score_weight * path_confidence`
    /// for every path. Mirrors _apply_maxcrit_features with the v16 defaults.
    fn write_max_crit_features_v16(
        &self,
        summary: &FindingSummary,
        vec: &mut [f32],
        offset: usize,
        score_weight: f32,
    ) {
        for (path, &max_ord) in &summary.sample_paths {
            if let Some(&idx) = self.presence_lookup.get(path.as_str()) {
                let conf = summary
                    .path_confidences
                    .get(path)
                    .copied()
                    .unwrap_or(1.0) as f32;
                vec[offset + idx] = max_ord as f32 * score_weight * conf;
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
    /// Max confidence (0.0..1.0) seen for each finding path. Used by
    /// `include_soft_presence` to weight presence/maxcrit features.
    path_confidences: HashMap<String, f64>,
    /// All per-finding confidences (post-MIN_CONFIDENCE filter). Used by
    /// `include_soft_presence` for rare-element weighting (mean).
    finding_confidences: Vec<f64>,
    filtered_finding_count: u32,
    notable_finding_count: u32,
    suspicious_finding_count: u32,
    hostile_finding_count: u32,
    unique_notable_ids: usize,
    unique_suspicious_ids: usize,
    unique_hostile_ids: usize,
    // v16 additions: number of distinct top-level categories at each tier.
    suspicious_category_breadth: usize,
    hostile_category_breadth: usize,
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
    let mut notable_ids: HashSet<&str> = HashSet::new();
    let mut suspicious_ids: HashSet<&str> = HashSet::new();
    let mut hostile_ids: HashSet<&str> = HashSet::new();

    for finding in findings {
        let fid = finding["i"].as_str().unwrap_or("");
        if fid.is_empty() {
            continue;
        }

        let conf = finding["c"].as_f64().unwrap_or(1.0);
        if conf < MIN_CONFIDENCE {
            continue;
        }
        summary.filtered_finding_count += 1;
        summary.finding_confidences.push(conf);

        let crit_ord = crit_ordinal(finding);

        if crit_ord >= 3 {
            summary.notable_finding_count += 1;
            notable_ids.insert(fid);
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
            // Track per-path max confidence for soft_presence weighting.
            let conf_entry = summary
                .path_confidences
                .entry(path.to_owned())
                .or_insert(0.0);
            if conf > *conf_entry {
                *conf_entry = conf;
            }
        }
    }

    summary.unique_notable_ids = notable_ids.len();
    summary.unique_suspicious_ids = suspicious_ids.len();
    summary.unique_hostile_ids = hostile_ids.len();

    // suspicious/hostile_category_breadth: distinct top-level categories
    // (first segment of path) where the path's max crit reaches the tier.
    // Mirrors Python's _summarize_report_files lines 1003-1012.
    let mut susp_cats: HashSet<&str> = HashSet::new();
    let mut host_cats: HashSet<&str> = HashSet::new();
    for (path, &max_ord) in &summary.sample_paths {
        if max_ord >= 4 {
            susp_cats.insert(path.split('/').next().unwrap_or(""));
        }
        if max_ord >= 5 {
            host_cats.insert(path.split('/').next().unwrap_or(""));
        }
    }
    summary.suspicious_category_breadth = susp_cats.len();
    summary.hostile_category_breadth = host_cats.len();
    summary
}

/// Aggregate findings across every file in the report.
fn summarize_report_files(files: &[&serde_json::Value]) -> FindingSummary {
    let mut combined = FindingSummary::default();
    let mut all_notable_ids: HashSet<String> = HashSet::new();
    let mut all_suspicious_ids: HashSet<String> = HashSet::new();
    let mut all_hostile_ids: HashSet<String> = HashSet::new();

    for file_entry in files {
        let findings: Vec<&serde_json::Value> = file_entry["ts"]
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
        for (path, &conf) in &file_summary.path_confidences {
            let entry = combined.path_confidences.entry(path.clone()).or_insert(0.0);
            if conf > *entry {
                *entry = conf;
            }
        }
        combined
            .finding_confidences
            .extend(file_summary.finding_confidences.iter().copied());

        // Collect unique IDs across all files for dedup.
        for finding in &findings {
            let fid = finding["i"].as_str().unwrap_or("");
            if fid.is_empty() {
                continue;
            }
            let conf = finding["c"].as_f64().unwrap_or(1.0);
            if conf < MIN_CONFIDENCE {
                continue;
            }
            let crit_ord = crit_ordinal(finding);
            if crit_ord >= 3 {
                all_notable_ids.insert(fid.to_owned());
            }
            if crit_ord >= 4 {
                all_suspicious_ids.insert(fid.to_owned());
            }
            if crit_ord >= 5 {
                all_hostile_ids.insert(fid.to_owned());
            }
        }
    }

    combined.unique_notable_ids = all_notable_ids.len();
    combined.unique_suspicious_ids = all_suspicious_ids.len();
    combined.unique_hostile_ids = all_hostile_ids.len();

    // suspicious/hostile_category_breadth: derived from the COMBINED sample_paths,
    // matching Python's _summarize_report_files which computes them from the merged
    // path map (NOT by union of per-file category sets).
    let mut susp_cats: HashSet<&str> = HashSet::new();
    let mut host_cats: HashSet<&str> = HashSet::new();
    for (path, &max_ord) in &combined.sample_paths {
        if max_ord >= 4 {
            susp_cats.insert(path.split('/').next().unwrap_or(""));
        }
        if max_ord >= 5 {
            host_cats.insert(path.split('/').next().unwrap_or(""));
        }
    }
    combined.suspicious_category_breadth = susp_cats.len();
    combined.hostile_category_breadth = host_cats.len();

    combined
}

/// Per-file risk statistics for top-k aggregation.
/// v16 adds per-file density (per KB) and category-breadth fields used by
/// the breadth/density and weighted-density agg features.
#[derive(Debug, Clone)]
struct FileRiskStats {
    suspicious_ratio: f32,
    hostile_ratio: f32,
    suspicious_findings: u32,
    hostile_findings: u32,
    suspicious_density: f32,
    hostile_density: f32,
    suspicious_category_breadth: usize,
    hostile_category_breadth: usize,
    max_crit: u32,
}

fn file_risk_stats(file_entry: &serde_json::Value) -> FileRiskStats {
    let findings: Vec<&serde_json::Value> = file_entry["ts"]
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    let summary = summarize_findings(&findings);
    let denom = summary.filtered_finding_count.max(1) as f32;
    // Mirror Python: size_kb = max(sz / 1024.0, 1.0)
    let sz_bytes = file_entry["sz"].as_f64().unwrap_or(0.0);
    let size_kb = (sz_bytes / 1024.0).max(1.0) as f32;
    let max_crit = summary.sample_paths.values().copied().max().unwrap_or(0);
    FileRiskStats {
        suspicious_ratio: summary.suspicious_finding_count as f32 / denom,
        hostile_ratio: summary.hostile_finding_count as f32 / denom,
        suspicious_findings: summary.suspicious_finding_count,
        hostile_findings: summary.hostile_finding_count,
        suspicious_density: summary.suspicious_finding_count as f32 / size_kb,
        hostile_density: summary.hostile_finding_count as f32 / size_kb,
        suspicious_category_breadth: summary.suspicious_category_breadth,
        hostile_category_breadth: summary.hostile_category_breadth,
        max_crit,
    }
}

/// Top-k file-risk aggregations as v16 expects (8 values when
/// include_breadth_density=True). Order matches Python's
/// `_topk_file_risk_features(... include_breadth_density=True)`:
///   0: top1_file_suspicious_ratio_sum
///   1: top1_file_hostile_ratio_sum
///   2: top1_file_suspicious_findings_log
///   3: top1_file_hostile_findings_log
///   4: top1_file_suspicious_density_sum
///   5: top1_file_hostile_density_sum
///   6: top1_file_suspicious_category_breadth_sum
///   7: top1_file_hostile_category_breadth_sum
fn topk_file_risk_features_v16(files: &[&serde_json::Value]) -> [f32; 8] {
    if files.is_empty() || TOP_K_RISK_FILES == 0 {
        return [0.0; 8];
    }
    let stats: Vec<FileRiskStats> = files.iter().map(|f| file_risk_stats(f)).collect();

    // Top by (suspicious_ratio, suspicious_findings, hostile_ratio, hostile_findings)
    // descending — matches Python's `key=lambda s: (...)` then `reverse=True`.
    let mut by_susp = stats.clone();
    by_susp.sort_by(|a, b| {
        (b.suspicious_ratio, b.suspicious_findings, b.hostile_ratio, b.hostile_findings)
            .partial_cmp(&(a.suspicious_ratio, a.suspicious_findings, a.hostile_ratio, a.hostile_findings))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_susp: &[FileRiskStats] = &by_susp[..TOP_K_RISK_FILES.min(by_susp.len())];

    // Top by (hostile_ratio, hostile_findings, suspicious_ratio, suspicious_findings).
    let mut by_host = stats;
    by_host.sort_by(|a, b| {
        (b.hostile_ratio, b.hostile_findings, b.suspicious_ratio, b.suspicious_findings)
            .partial_cmp(&(a.hostile_ratio, a.hostile_findings, a.suspicious_ratio, a.suspicious_findings))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_host: &[FileRiskStats] = &by_host[..TOP_K_RISK_FILES.min(by_host.len())];

    let susp_ratio_sum: f32 = top_susp.iter().map(|s| s.suspicious_ratio).sum();
    let host_ratio_sum: f32 = top_host.iter().map(|s| s.hostile_ratio).sum();
    let susp_findings_log = (top_susp.iter().map(|s| s.suspicious_findings).sum::<u32>() as f32 + 1.0).ln();
    let host_findings_log = (top_host.iter().map(|s| s.hostile_findings).sum::<u32>() as f32 + 1.0).ln();
    let susp_density_sum: f32 = top_susp.iter().map(|s| s.suspicious_density).sum();
    let host_density_sum: f32 = top_host.iter().map(|s| s.hostile_density).sum();
    let susp_cat_sum: f32 = top_susp.iter().map(|s| s.suspicious_category_breadth as f32).sum();
    let host_cat_sum: f32 = top_host.iter().map(|s| s.hostile_category_breadth as f32).sum();

    [
        susp_ratio_sum,
        host_ratio_sum,
        susp_findings_log,
        host_findings_log,
        susp_density_sum,
        host_density_sum,
        susp_cat_sum,
        host_cat_sum,
    ]
}

fn write_aggregate_features(
    summary: &FindingSummary,
    files: &[&serde_json::Value],
    vec: &mut [f32],
    offset: usize,
) {
    // -----------------------------------------------------------------------
    // v16 agg layout — 50 features matching collimator's _apply_aggregate_features.
    // Order MUST mirror build_expected_feature_names exactly:
    //   0..19  : 20 base agg features
    //   20..31 : 12 suspicious_breadth_density features
    //   32..36 : 5 hostile_escalation features
    //   37..38 : 2 hostile_weighted_density features
    //   39..42 : 4 repetition_penalty features
    //   43..48 : 6 file_severity_distribution features
    //   49     : hostile_depth_weight
    // -----------------------------------------------------------------------
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

    // Two total_kb values, matching Python: ratio features use min 0.1,
    // breadth/density and weighted features use min 1.0.
    let total_size_bytes: f64 = files.iter().map(|f| f["sz"].as_f64().unwrap_or(0.0)).sum();
    let total_kb_raw = (total_size_bytes / 1024.0) as f32;
    let total_kb_p1: f32 = total_kb_raw.max(0.1);
    let total_kb_1: f32 = total_kb_raw.max(1.0);

    // ---- 0..19: 20 base features ----
    // 0..7: breadth/concentration block (matches Python lines 1211-1219)
    vec[offset] = max_crit as f32;
    vec[offset + 1] = categories.len() as f32;
    vec[offset + 2] = (path_breadth_any as f32).ln_1p();
    vec[offset + 3] = (total_active as f32).ln_1p();
    vec[offset + 4] = breadth_suspicious as f32 / path_breadth_any.max(1) as f32;
    vec[offset + 5] = breadth_hostile as f32 / path_breadth_any.max(1) as f32;
    vec[offset + 6] = breadth_suspicious as f32 / breadth_notable.max(1) as f32;
    vec[offset + 7] = breadth_notable_only as f32 / breadth_notable.max(1) as f32;
    // 8..10: notable/suspicious/hostile_findings_log (Python lines 1220-1222)
    vec[offset + 8] = (summary.notable_finding_count as f32).ln_1p();
    vec[offset + 9] = (summary.suspicious_finding_count as f32).ln_1p();
    vec[offset + 10] = (summary.hostile_finding_count as f32).ln_1p();
    // 11..15: density-first metrics using total_kb_p1 (Python lines 1226-1230)
    // NOTE: these are the BUG-B-fixed formulas — Python writes per-KB density,
    // not per-finding ratio. The pre-v16 Rust implementation had drifted.
    vec[offset + 11] = summary.notable_finding_count as f32 / total_kb_p1;
    vec[offset + 12] = summary.suspicious_finding_count as f32 / total_kb_p1;
    vec[offset + 13] = summary.hostile_finding_count as f32 / total_kb_p1;
    let log_kb_p1 = total_kb_p1.ln_1p();
    vec[offset + 14] = (summary.unique_suspicious_ids as f32).ln_1p() / log_kb_p1;
    vec[offset + 15] = (summary.unique_hostile_ids as f32).ln_1p() / log_kb_p1;
    // 16..19: top-k file risk (4 base values from topk)
    let topk = topk_file_risk_features_v16(files);
    vec[offset + 16] = topk[0];
    vec[offset + 17] = topk[1];
    vec[offset + 18] = topk[2];
    vec[offset + 19] = topk[3];

    // ---- 20..31: 12 suspicious_breadth_density features ----
    let category_denom = categories.len().max(1) as f32;
    vec[offset + 20] = summary.suspicious_category_breadth as f32;
    vec[offset + 21] = summary.hostile_category_breadth as f32;
    vec[offset + 22] = summary.suspicious_category_breadth as f32 / category_denom;
    vec[offset + 23] = summary.hostile_category_breadth as f32 / category_denom;
    vec[offset + 24] = summary.suspicious_finding_count as f32 / total_kb_1;
    vec[offset + 25] = summary.hostile_finding_count as f32 / total_kb_1;
    vec[offset + 26] = summary.suspicious_category_breadth as f32 / total_kb_1;
    vec[offset + 27] = summary.hostile_category_breadth as f32 / total_kb_1;
    vec[offset + 28] = topk[4]; // top1_file_suspicious_density_sum
    vec[offset + 29] = topk[5]; // top1_file_hostile_density_sum
    vec[offset + 30] = topk[6]; // top1_file_suspicious_category_breadth_sum
    vec[offset + 31] = topk[7]; // top1_file_hostile_category_breadth_sum

    // ---- 32..36: 5 hostile_escalation features (Python lines 1259-1263) ----
    vec[offset + 32] = breadth_hostile as f32 / breadth_notable.max(1) as f32;
    vec[offset + 33] = breadth_hostile as f32 / breadth_suspicious.max(1) as f32;
    vec[offset + 34] =
        summary.suspicious_finding_count as f32 / summary.notable_finding_count.max(1) as f32;
    vec[offset + 35] =
        summary.hostile_finding_count as f32 / summary.notable_finding_count.max(1) as f32;
    vec[offset + 36] =
        summary.hostile_finding_count as f32 / summary.suspicious_finding_count.max(1) as f32;

    // Per-file stats needed for hostile_weighted_density and file_severity_distribution.
    let stats: Vec<FileRiskStats> = files.iter().map(|f| file_risk_stats(f)).collect();

    // ---- 37..38: 2 hostile_weighted_density features (Python lines 1276-1277) ----
    let host_density_global = summary.hostile_finding_count as f32 / total_kb_1;
    let susp_density_global = summary.suspicious_finding_count as f32 / total_kb_1;
    vec[offset + 37] = host_density_global + 0.25 * susp_density_global;
    let mut by_weighted = stats.clone();
    // Python sort key: (hostile_density + 0.25*suspicious_density, hostile_density, suspicious_density) desc.
    by_weighted.sort_by(|a, b| {
        let ka = (
            a.hostile_density + 0.25 * a.suspicious_density,
            a.hostile_density,
            a.suspicious_density,
        );
        let kb = (
            b.hostile_density + 0.25 * b.suspicious_density,
            b.hostile_density,
            b.suspicious_density,
        );
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_weighted: f32 = by_weighted
        .iter()
        .take(TOP_K_RISK_FILES)
        .map(|s| s.hostile_density + 0.25 * s.suspicious_density)
        .sum();
    vec[offset + 38] = top_weighted;

    // ---- 39..42: 4 repetition_penalty features (Python lines 1280-1283) ----
    vec[offset + 39] = 1.0
        - (summary.unique_suspicious_ids as f32
            / summary.suspicious_finding_count.max(1) as f32);
    vec[offset + 40] = 1.0
        - (summary.unique_hostile_ids as f32
            / summary.hostile_finding_count.max(1) as f32);
    vec[offset + 41] = 1.0
        - (summary.suspicious_category_breadth as f32
            / summary.suspicious_finding_count.max(1) as f32);
    vec[offset + 42] = 1.0
        - (summary.hostile_category_breadth as f32
            / summary.hostile_finding_count.max(1) as f32);

    // ---- 43..48: 6 file_severity_distribution features (Python lines 1290-1295) ----
    let n_files = files.len().max(1) as f32;
    let hostile_files = stats.iter().filter(|s| s.max_crit >= 5).count() as f32;
    let suspicious_files = stats.iter().filter(|s| s.max_crit == 4).count() as f32;
    let notable_files = stats.iter().filter(|s| s.max_crit == 3).count() as f32;
    vec[offset + 43] = hostile_files / n_files;
    vec[offset + 44] = suspicious_files / n_files;
    vec[offset + 45] = notable_files / n_files;
    vec[offset + 46] = hostile_files.ln_1p();
    vec[offset + 47] = suspicious_files.ln_1p();
    vec[offset + 48] = notable_files.ln_1p();

    // ---- 49: hostile_depth_weight ----
    // Python computes this in the agg block (features.py:1815-1828) as
    // sum over files of: hostile_finding_count_in_file * depth_in_archive_tree.
    // Litmus's input from cleave doesn't track parent paths the same way;
    // for now, leave as zero (TODO: implement Exp 51 depth tracking).
    vec[offset + 49] = 0.0;
}

/// Parse an ISO-8601 timestamp into seconds since the Unix epoch (UTC).
/// Mirrors Python's `datetime.fromisoformat(s.replace(" ", "T")).timestamp()`
/// for the formats cleave actually emits:
///     "YYYY-MM-DDTHH:MM:SS"           (naive — assumed UTC)
///     "YYYY-MM-DDTHH:MM:SS.fff"       (with subseconds)
///     "YYYY-MM-DDTHH:MM:SSZ"          (Zulu)
///     "YYYY-MM-DDTHH:MM:SS+HH:MM"     (offset)
///     "YYYY-MM-DDTHH:MM:SS-HH:MM"
fn parse_iso8601(s: &str) -> Option<f64> {
    let s = s.trim().replace(' ', "T");
    let bytes = s.as_bytes();
    if bytes.len() < 19 {
        return None;
    }
    let year: i32 = std::str::from_utf8(&bytes[0..4]).ok()?.parse().ok()?;
    if bytes[4] != b'-' {
        return None;
    }
    let month: u32 = std::str::from_utf8(&bytes[5..7]).ok()?.parse().ok()?;
    if bytes[7] != b'-' {
        return None;
    }
    let day: u32 = std::str::from_utf8(&bytes[8..10]).ok()?.parse().ok()?;
    if bytes[10] != b'T' {
        return None;
    }
    let hour: i64 = std::str::from_utf8(&bytes[11..13]).ok()?.parse().ok()?;
    if bytes[13] != b':' {
        return None;
    }
    let minute: i64 = std::str::from_utf8(&bytes[14..16]).ok()?.parse().ok()?;
    if bytes[16] != b':' {
        return None;
    }
    let second: i64 = std::str::from_utf8(&bytes[17..19]).ok()?.parse().ok()?;

    // Howard Hinnant's days_from_civil — converts (year, month, day) to days
    // since 1970-01-01. Handles the proleptic Gregorian calendar correctly.
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32; // [0, 399]
    let m = month;
    let mp = if m > 2 { m - 3 } else { m + 9 }; // [0, 11], March=0
    let doy = (153 * mp + 2) / 5 + day - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    let days_since_epoch = era as i64 * 146097 + doe as i64 - 719468;

    let mut total = days_since_epoch * 86400 + hour * 3600 + minute * 60 + second;

    let mut idx = 19;
    let mut frac = 0.0_f64;
    if idx < bytes.len() && bytes[idx] == b'.' {
        idx += 1;
        let start = idx;
        while idx < bytes.len() && bytes[idx].is_ascii_digit() {
            idx += 1;
        }
        if idx > start {
            let frac_str = std::str::from_utf8(&bytes[start..idx]).ok()?;
            let denom = 10_f64.powi((idx - start) as i32);
            frac = frac_str.parse::<f64>().ok()? / denom;
        }
    }
    if idx < bytes.len() {
        let tz = bytes[idx];
        if tz == b'+' || tz == b'-' {
            // Subtract the offset to convert to UTC.
            let sign: i64 = if tz == b'+' { -1 } else { 1 };
            idx += 1;
            if idx + 5 <= bytes.len() && bytes[idx + 2] == b':' {
                let oh: i64 = std::str::from_utf8(&bytes[idx..idx + 2]).ok()?.parse().ok()?;
                let om: i64 = std::str::from_utf8(&bytes[idx + 3..idx + 5]).ok()?.parse().ok()?;
                total += sign * (oh * 3600 + om * 60);
            }
        }
        // 'Z' or trailing chars are no-ops (already UTC).
    }
    Some(total as f64 + frac)
}

/// Group 15: structural extensions (13 features matching Python's
/// extended _apply_structural_features block at lines 1517-1553+1561-1597
/// in collimator/features.py). Layout (offset 0..12):
///   0  packaged_capability  = |sample_paths| * max_entropy
///   1  mtime_range_hours    = (max-min mtimes) / 3600
///   2  mtime_std_dev_hours  = std(mtimes) / 3600
///   3  max_nesting_depth_log= log1p(max parent depth)
///   4  inner_file_ratio     = inner_files / max(file_count, 1)
///   5  entropy_std_dev      = std(entropies)
///   6  entropy_max_diff     = max(entropies) - mean(entropies)
///   7  air_gap_signal       = 1 if hostile_files > 0 && all hostile have no parent
///   8  anachronistic_injection = max(|t - median(mtimes)|) over hostile_mtimes / 3600
///   9  code_entropy_spike   = max(code_entropies) - mean(entropies)
///  10  foreign_binary_signal= 1 if has source files AND has binary files
///  11  extension_mismatch_signal = count of binary files with text extension
///  12  hostile_finding_density = (hostile_files * 1000) / total_loc
fn write_structural_extensions(
    files: &[&serde_json::Value],
    summary: &FindingSummary,
    vec: &mut [f32],
    offset: usize,
) {
    let binary_like = ["pe", "elf", "macho"];
    let source_types = ["javascript", "python", "typescript", "ruby", "php"];
    let text_exts = ["txt", "md", "json", "png", "jpg"];

    let mut mtimes: Vec<f64> = Vec::new();
    let mut hostile_mtimes: Vec<f64> = Vec::new();
    let mut entropies: Vec<f64> = Vec::new();
    let mut code_entropies: Vec<f64> = Vec::new();
    let mut max_entropy: f64 = 0.0;
    let mut hostile_files: u32 = 0;
    let mut hostile_files_with_parent: u32 = 0;
    let mut inner_file_count: u32 = 0;
    let mut total_loc: u64 = 0;
    let mut extension_mismatches: u32 = 0;
    let mut has_source_files = false;
    let mut has_foreign_binaries = false;

    // Compute nesting depth from parent tree.
    let mut depths: HashMap<String, u32> = HashMap::new();
    for f in files {
        let fpath = f["path"].as_str().unwrap_or("").to_string();
        let parent = f["p"].as_str().unwrap_or("").to_string();
        if parent.is_empty() {
            depths.insert(fpath, 0);
        } else {
            let pd = depths.get(&parent).copied().unwrap_or(0);
            depths.insert(fpath, pd + 1);
        }
    }
    let max_nesting_depth: u32 = depths.values().copied().max().unwrap_or(0);

    for file_entry in files {
        if !file_entry["p"].as_str().unwrap_or("").is_empty() {
            inner_file_count += 1;
        }
        let ftype = file_entry["type"].as_str().unwrap_or("");
        if source_types.contains(&ftype) {
            has_source_files = true;
        }
        if binary_like.contains(&ftype) && has_source_files {
            has_foreign_binaries = true;
        }

        // total_loc from text metrics
        let lines = file_entry
            .get("ms")
            .and_then(|m| m.get("text"))
            .and_then(|t| t.get("total_lines"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        total_loc += lines as u64;

        // extension mismatches: binary file with text extension
        let fpath = file_entry["path"].as_str().unwrap_or("");
        if !fpath.is_empty() && fpath.contains('.') {
            if let Some(ext) = fpath.rsplit('.').next() {
                let ext_lower = ext.to_ascii_lowercase();
                if binary_like.contains(&ftype) && text_exts.contains(&ext_lower.as_str()) {
                    extension_mismatches += 1;
                }
            }
        }

        // mtime
        if let Some(mt) = file_entry["mt"].as_str() {
            if let Some(t) = parse_iso8601(mt) {
                mtimes.push(t);
            }
        }

        // entropy
        let ent = file_entry
            .get("ms")
            .and_then(|m| m.get("binary"))
            .and_then(|b| b.get("overall_entropy"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        if ent > 0.0 {
            entropies.push(ent);
        }
        if ent > max_entropy {
            max_entropy = ent;
        }

        // Per-file finding summary for hostile counts.
        let findings: Vec<&serde_json::Value> = file_entry["ts"]
            .as_array()
            .map(|a| a.iter().collect())
            .unwrap_or_default();
        let file_summary = summarize_findings(&findings);
        if file_summary.hostile_finding_count > 0 {
            hostile_files += 1;
            if let Some(mt) = file_entry["mt"].as_str() {
                if let Some(t) = parse_iso8601(mt) {
                    hostile_mtimes.push(t);
                }
            }
            if !file_entry["p"].as_str().unwrap_or("").is_empty() {
                hostile_files_with_parent += 1;
            }
        }

        // code_entropies: same threshold/types as Python
        let code_types = ["javascript", "python", "pe", "elf", "macho"];
        if ent > 0.0 && code_types.contains(&ftype) {
            code_entropies.push(ent);
        }
    }

    // 0: packaged_capability = |sample_paths| * max_entropy.
    // Content-based: distinct capability paths × packing level. Computed in
    // f64 before casting to f32 to match Python's float arithmetic.
    vec[offset] = (summary.sample_paths.len() as f64 * max_entropy) as f32;

    // 1, 2: mtime_range_hours, mtime_std_dev_hours
    if mtimes.len() > 1 {
        let mn = mtimes.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = mtimes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        vec[offset + 1] = ((mx - mn) / 3600.0) as f32;
        let mean = mtimes.iter().sum::<f64>() / mtimes.len() as f64;
        let var = mtimes.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / mtimes.len() as f64;
        vec[offset + 2] = (var.sqrt() / 3600.0) as f32;
    }

    // 3: max_nesting_depth_log
    vec[offset + 3] = (max_nesting_depth as f32).ln_1p();
    // 4: inner_file_ratio
    vec[offset + 4] = inner_file_count as f32 / files.len().max(1) as f32;

    // 5, 6: entropy_std_dev, entropy_max_diff
    if entropies.len() > 1 {
        let mean = entropies.iter().sum::<f64>() / entropies.len() as f64;
        let var = entropies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / entropies.len() as f64;
        vec[offset + 5] = var.sqrt() as f32;
        let mx = entropies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        vec[offset + 6] = (mx - mean) as f32;
    }

    // 7: air_gap_signal — only when v16 default has it ON.
    vec[offset + 7] = if hostile_files > 0 && hostile_files_with_parent == 0 {
        1.0
    } else {
        0.0
    };

    // 8: anachronistic_injection — Exp 48
    if !mtimes.is_empty() && !hostile_mtimes.is_empty() {
        let mut sorted = mtimes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 1 {
            sorted[sorted.len() / 2]
        } else {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        };
        let max_delta = hostile_mtimes
            .iter()
            .map(|t| (t - median).abs())
            .fold(0.0, f64::max);
        vec[offset + 8] = (max_delta / 3600.0) as f32;
    }

    // 9: code_entropy_spike — Exp 49
    if !code_entropies.is_empty() {
        let avg_ent: f64 = if entropies.is_empty() {
            0.0
        } else {
            entropies.iter().sum::<f64>() / entropies.len() as f64
        };
        let max_code_ent = code_entropies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        vec[offset + 9] = (max_code_ent - avg_ent) as f32;
    }

    // 10: foreign_binary_signal — Exp 54
    vec[offset + 10] = if has_foreign_binaries { 1.0 } else { 0.0 };
    // 11: extension_mismatch_signal — Exp 55
    vec[offset + 11] = extension_mismatches as f32;
    // 12: hostile_finding_density — Exp 56 (per 1000 LoC)
    if total_loc > 0 {
        vec[offset + 12] = (hostile_files as f32 * 1000.0) / total_loc as f32;
    }
}

/// Extract (formula, elements, score) from a cleave report's depth-0 file.
/// Mirrors hopper.parseCleaveFile and collimator's canonical_fields_from_report.
fn canonical_fields_from_report(report: &serde_json::Value) -> (String, String, i64) {
    let files = match report.get("fs").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return (String::new(), String::new(), 0),
    };
    for f in files {
        let depth = f.get("dp").and_then(|v| v.as_i64()).unwrap_or(0);
        if depth == 0 {
            let formula = f.get("f").and_then(|v| v.as_str()).unwrap_or("").to_string();
            // Strip Unicode subscript digits ₀-₉ (U+2080..U+2089).
            let elements: String = formula
                .chars()
                .filter(|c| !('\u{2080}'..='\u{2089}').contains(c))
                .collect();
            let score = f.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
            return (formula, elements, score);
        }
    }
    (String::new(), String::new(), 0)
}

/// Collect the unique 3-level path bases for a single file's findings.
/// Mirrors Python's `_apply_bigram_features` per-file logic:
///     paths_list = sorted({fid.split("::")[0] for fid in file_traits})
fn unique_3level_paths_for_file(file_entry: &serde_json::Value) -> Vec<String> {
    let mut paths: std::collections::HashSet<String> = std::collections::HashSet::new();
    let findings: Vec<&serde_json::Value> = file_entry["ts"]
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    for finding in findings {
        let fid = finding["i"].as_str().unwrap_or("");
        if fid.is_empty() {
            continue;
        }
        let conf = finding["c"].as_f64().unwrap_or(1.0);
        if conf < MIN_CONFIDENCE {
            continue;
        }
        // Drop the ::detail suffix to get the 3-level path.
        let base = fid.split("::").next().unwrap_or(fid).to_string();
        paths.insert(base);
    }
    let mut out: Vec<String> = paths.into_iter().collect();
    out.sort();
    out
}

/// Write bigram multi-hot features. Used by both `bigrams:*` and
/// `unsigned_bigram:*` (they share the same vocab).
fn write_bigram_features(
    files: &[&serde_json::Value],
    vec: &mut [f32],
    offset: usize,
    lookup: &HashMap<String, usize>,
) {
    for file_entry in files {
        let paths = unique_3level_paths_for_file(file_entry);
        for i in 0..paths.len() {
            for j in (i + 1)..paths.len() {
                let bigram = format!("{} + {}", paths[i], paths[j]);
                if let Some(&idx) = lookup.get(&bigram) {
                    vec[offset + idx] = 1.0;
                }
            }
        }
    }
}

/// Write trigram multi-hot features. Mirrors Python `_apply_trigram_features`.
fn write_trigram_features(
    files: &[&serde_json::Value],
    vec: &mut [f32],
    offset: usize,
    lookup: &HashMap<String, usize>,
) {
    for file_entry in files {
        let paths = unique_3level_paths_for_file(file_entry);
        let n = paths.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let trigram = format!("{} + {} + {}", paths[i], paths[j], paths[k]);
                    if let Some(&idx) = lookup.get(&trigram) {
                        vec[offset + idx] = 1.0;
                    }
                }
            }
        }
    }
}

/// Write logic_gap features. Mirrors Python `_apply_logic_gap_features`.
/// For each LOGIC_GAPS category (sorted: crypto, network, process):
///   1.0 if any import matches the category's import set AND no finding
///   path with max_ord >= 3 starts with any of the category's trait prefixes.
fn write_logic_gap_features(
    summary: &FindingSummary,
    files: &[&serde_json::Value],
    vec: &mut [f32],
    offset: usize,
) {
    // LOGIC_GAPS sorted-key order is [crypto, network, process] — matches
    // LOGIC_GAP_CATEGORIES constant (which is already sorted).
    let logic_gaps: &[(&str, &[&str], &[&str])] = &[
        (
            "crypto",
            &[
                "cryptography",
                "Crypto",
                "hashlib",
                "CryptAcquireContext",
                "BCryptOpenAlgorithmProvider",
            ],
            &["micro-behaviors/crypto", "metadata/encoded-payload"],
        ),
        (
            "network",
            &[
                "socket",
                "urllib",
                "requests",
                "http",
                "curl",
                "wininet",
                "winhttp",
            ],
            &[
                "micro-behaviors/network",
                "objectives/command-and-control",
            ],
        ),
        (
            "process",
            &[
                "subprocess",
                "os.spawn",
                "os.system",
                "CreateProcess",
                "ShellExecute",
                "posix_spawn",
            ],
            &[
                "micro-behaviors/process/create",
                "objectives/execution",
            ],
        ),
    ];

    // Collect all unique imports from all files. Cleave's "is" array contains
    // either strings or dicts with an "n" key.
    let mut all_imports: HashSet<&str> = HashSet::new();
    for file_entry in files {
        let Some(imports) = file_entry.get("is").and_then(|v| v.as_array()) else {
            continue;
        };
        for imp in imports {
            if let Some(s) = imp.as_str() {
                all_imports.insert(s);
            } else if let Some(name) = imp.get("n").and_then(|v| v.as_str()) {
                all_imports.insert(name);
            }
        }
    }

    // LOGIC_GAP_CATEGORIES is the sorted key order and matches the spec layout.
    for (i, target_cat) in LOGIC_GAP_CATEGORIES.iter().enumerate() {
        let Some((_, imports_set, traits_set)) =
            logic_gaps.iter().find(|(c, _, _)| c == target_cat)
        else {
            continue;
        };
        let has_import = imports_set.iter().any(|imp| all_imports.contains(*imp));
        let has_behavior = summary.sample_paths.iter().any(|(path, &max_ord)| {
            max_ord >= 3 && traits_set.iter().any(|t| path.starts_with(t))
        });
        if has_import && !has_behavior {
            vec[offset + i] = 1.0;
        }
    }
}

/// Write intent_gap features. Mirrors Python `_apply_intent_gap_features`.
/// For each category in INTENT_GAP_CATEGORIES order: 1.0 if a risky behavior
/// is present (suspicious crit or above) AND no doc/help intent in the report.
fn write_intent_gap_features(summary: &FindingSummary, vec: &mut [f32], offset: usize) {
    let has_doc = summary.sample_paths.contains_key("metadata/package/documentation");
    let has_help = summary.sample_paths.contains_key("metadata/package/help");
    let intent_signal = has_doc || has_help;

    // Same risky_behaviors map as Python.
    let risky: &[(&str, &[&str])] = &[
        ("network", &["objectives/network", "micro-behaviors/network"]),
        (
            "filesystem",
            &["objectives/persistence", "micro-behaviors/filesystem"],
        ),
        (
            "execution",
            &["objectives/execution", "micro-behaviors/process/create"],
        ),
        ("crypto", &["objectives/crypto", "micro-behaviors/crypto"]),
    ];

    for (i, target_cat) in INTENT_GAP_CATEGORIES.iter().enumerate() {
        let traits = risky
            .iter()
            .find(|(c, _)| c == target_cat)
            .map(|(_, t)| *t)
            .unwrap_or(&[]);
        let has_behavior = summary.sample_paths.iter().any(|(path, &max_ord)| {
            max_ord >= 4 && traits.iter().any(|t| path.starts_with(t))
        });
        if has_behavior && !intent_signal {
            vec[offset + i] = 1.0;
        }
    }
}

/// Write missing:{ftype}*{trait} negative-space features.
/// Python: `_apply_neg_space_features` writes 1.0 if the report contains a
/// file of the matching ftype AND the expected trait is absent from sample_paths.
fn write_negative_space_features(
    summary: &FindingSummary,
    files: &[&serde_json::Value],
    vec: &mut [f32],
    offset: usize,
) {
    // Collect file types present in the report.
    let mut present_types: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for f in files {
        let ft = f["type"].as_str().unwrap_or("");
        if !ft.is_empty() {
            present_types.insert(ft);
        }
    }

    let mut idx = 0;
    for &(ftype, traits) in EXPECTED_GHOSTS {
        for trait_path in traits {
            if present_types.contains(ftype)
                && !summary.sample_paths.contains_key(*trait_path)
            {
                vec[offset + idx] = 1.0;
            }
            idx += 1;
        }
    }
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
        let metrics = match file_entry.get("ms") {
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
    let mut suspicious_files: u32 = 0;
    let mut hostile_files: u32 = 0;

    for file_entry in files {
        let ft = file_entry["type"].as_str().unwrap_or("");
        let size = file_entry["sz"].as_u64().unwrap_or(0);
        if binary_like.contains(&ft) && size < 20_000 {
            any_tiny_binary = true;
        }
        if file_entry.get("is").is_some() {
            import_candidates += 1;
            let imports_empty = file_entry["is"].as_array().is_none_or(Vec::is_empty);
            if imports_empty {
                importless_candidates += 1;
            }
        }
        // Track max binary entropy for stealth_potential.
        let entropy = file_entry
            .get("ms")
            .and_then(|m| m.get("binary"))
            .and_then(|b| b.get("overall_entropy"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        if entropy > max_entropy {
            max_entropy = entropy;
        }
        // v16 file_risk_coverage: count files with at least one suspicious or
        // hostile finding (each file gets summarized independently).
        let findings: Vec<&serde_json::Value> = file_entry["ts"]
            .as_array()
            .map(|a| a.iter().collect())
            .unwrap_or_default();
        let file_summary = summarize_findings(&findings);
        if file_summary.suspicious_finding_count > 0 {
            suspicious_files += 1;
        }
        if file_summary.hostile_finding_count > 0 {
            hostile_files += 1;
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

    // v16 file_risk_coverage features (struct_file_risk_coverage=ON by default):
    // suspicious_file_fraction, hostile_file_fraction, plus log1p counts.
    let file_count_denom = files.len().max(1) as f32;
    vec[offset + 7] = suspicious_files as f32 / file_count_denom;
    vec[offset + 8] = hostile_files as f32 / file_count_denom;
    vec[offset + 9] = (suspicious_files as f32).ln_1p();
    vec[offset + 10] = (hostile_files as f32).ln_1p();
}

/// Return all file entries from a v3 report.
fn report_files(report: &serde_json::Value) -> Vec<&serde_json::Value> {
    report["fs"]
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
        build_expected_feature_names(
            presence_vocab,
            filetype_vocab,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        )
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
        assert_eq!(crit_ordinal(&serde_json::json!({"l":5})), 5); // hostile
        assert_eq!(crit_ordinal(&serde_json::json!({"l":4})), 4); // suspicious
        assert_eq!(crit_ordinal(&serde_json::json!({"l":3})), 3); // notable
        assert_eq!(crit_ordinal(&serde_json::json!({"l":2})), 2); // baseline
        assert_eq!(crit_ordinal(&serde_json::json!({"l":0})), 0); // filtered
        assert_eq!(crit_ordinal(&serde_json::json!({})), 0); // missing → 0
    }

    #[test]
    fn test_standardize() {
        let spec = FeatureSpec {
            version: 16,
            abi_version: 16,
            presence_vocab: vec![],
            filetype_vocab: vec![],
            element_vocab: vec![],
            bigram_vocab: vec![],
            ghost_vocab: vec![],
            skeleton_vocab: vec![],
            rare_element_vocab: vec![],
            trigram_vocab: vec![],
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

    // NOTE: end-to-end extraction unit tests that hardcoded the v15 vector
    // layout were removed during the v16 migration. The integration tests in
    // tests/extraction_parity.rs verify feature extraction against real
    // collimator-generated fixtures to <1e-5 per feature, which is strictly
    // stronger than any synthetic offset assertion would be.

    #[test]
    fn test_load_rejects_missing_feature_names() -> Result<()> {
        let mut file = tempfile::NamedTempFile::new()?;
        writeln!(
            file,
            "{{\"version\":16,\"presence_vocab\":[\"objectives\"],\"filetype_vocab\":[\"sh\"],\"total_features\":51,\"standardized\":false}}"
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
        let mut feature_names = build_expected_feature_names(
            &presence_vocab,
            &filetype_vocab,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        let Some(last_feature_name) = feature_names.last_mut() else {
            anyhow::bail!("expected non-empty feature list");
        };
        *last_feature_name = "struct:wrong_name".to_string();
        let total_features = feature_names.len();
        let mut file = tempfile::NamedTempFile::new()?;
        writeln!(
            file,
            "{{\"version\":16,\"abi_version\":16,\"presence_vocab\":[\"objectives\"],\"filetype_vocab\":[\"sh\"],\"feature_names\":[{}],\"total_features\":{},\"standardized\":false}}",
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
        assert!(err.to_string().contains("expected v16 layout"));
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
        let total_features = build_expected_feature_names(
            &presence_vocab,
            &filetype_vocab,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        ).len();
        let mut file = tempfile::NamedTempFile::new()?;
        writeln!(
            file,
            "{{\"version\":16,\"abi_version\":16,\"presence_vocab\":[\"objectives\"],\"filetype_vocab\":[\"sh\"],\"feature_names\":[{}],\"total_features\":{},\"feature_means\":[0.0],\"feature_stds\":[1.0],\"standardized\":true}}",
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
