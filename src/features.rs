//! Feature extraction from cleave v3 AnalysisReport JSON.
//!
//! Mirrors the feature extraction in collimator/src/collimator/features.py (v16)
//! exactly, using the same feature_spec.json vocabulary to produce identical
//! feature vectors.
//!
//! Feature assignment uses [`FeatureWriter`] which maps feature names to indices
//! via the spec's `feature_names` list. Features not in the spec (disabled groups)
//! are silently skipped — no errors, no wasted space.

// All feature vectors use f32 to match the model's input dtype. The f64→f32
// narrowing throughout this file is intentional and safe: feature values are
// counts, ratios, or scores that fit well within f32 range.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Name-based feature assignment. Wraps a feature vec + name→index lookup.
/// Features not in the spec (disabled groups, unknown names) are silently
/// skipped — this is how disabled feature groups "just work" without special
/// handling.
struct FeatureWriter<'a> {
    vec: &'a mut [f32],
    lookup: &'a HashMap<String, usize>,
}

impl FeatureWriter<'_> {
    /// Set a feature by name. No-op if the feature isn't in the spec.
    #[inline]
    fn set(&mut self, name: &str, value: f32) {
        if let Some(&idx) = self.lookup.get(name) {
            self.vec[idx] = value;
        }
    }
}

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
    version: u32,
    abi_version: u32,
    presence_vocab: Vec<String>,
    filetype_vocab: Vec<String>,
    element_vocab: Vec<String>,
    bigram_vocab: Vec<String>,
    ghost_vocab: Vec<String>,
    skeleton_vocab: Vec<String>,
    rare_element_vocab: Vec<String>,
    trigram_vocab: Vec<String>,
    metric_vocab: Vec<String>,
    crit_unigram_vocab: Vec<String>,
    crit_bigram_vocab: Vec<String>,
    crit_trigram_vocab: Vec<String>,
    attack_bigram_vocab: Vec<String>,
    attack_trigram_vocab: Vec<String>,
    mbc_bigram_vocab: Vec<String>,
    mbc_trigram_vocab: Vec<String>,
    feature_names: Vec<String>,
    total_features: usize,
    feature_means: Option<Vec<f32>>,
    feature_stds: Option<Vec<f32>>,
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
    metric_vocab: Vec<String>,
    #[serde(default)]
    crit_unigram_vocab: Vec<String>,
    #[serde(default)]
    crit_bigram_vocab: Vec<String>,
    #[serde(default)]
    crit_trigram_vocab: Vec<String>,
    #[serde(default)]
    attack_bigram_vocab: Vec<String>,
    #[serde(default)]
    attack_trigram_vocab: Vec<String>,
    #[serde(default)]
    mbc_bigram_vocab: Vec<String>,
    #[serde(default)]
    mbc_trigram_vocab: Vec<String>,
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
            metric_vocab: raw.metric_vocab,
            crit_unigram_vocab: raw.crit_unigram_vocab,
            crit_bigram_vocab: raw.crit_bigram_vocab,
            crit_trigram_vocab: raw.crit_trigram_vocab,
            attack_bigram_vocab: raw.attack_bigram_vocab,
            attack_trigram_vocab: raw.attack_trigram_vocab,
            mbc_bigram_vocab: raw.mbc_bigram_vocab,
            mbc_trigram_vocab: raw.mbc_trigram_vocab,
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
            &self.metric_vocab,
            &self.crit_unigram_vocab,
            &self.crit_bigram_vocab,
            &self.crit_trigram_vocab,
            &self.attack_bigram_vocab,
            &self.attack_trigram_vocab,
            &self.mbc_bigram_vocab,
            &self.mbc_trigram_vocab,
        );
        if self.feature_names != expected_feature_names {
            // Feature groups can be disabled during training (COLLIMATOR_DISABLE_FEATURE_GROUPS),
            // producing fewer features than the full layout. Verify that every feature in the
            // spec exists in the expected layout — if so, it's a valid subset.
            let expected_set: std::collections::HashSet<&str> = expected_feature_names.iter().map(String::as_str).collect();
            let all_known = self.feature_names.iter().all(|n| expected_set.contains(n.as_str()));
            if !all_known {
                let first_unknown = self.feature_names.iter().find(|n| !expected_set.contains(n.as_str()));
                anyhow::bail!(
                    "feature spec contains unknown feature {:?} not in the expected v{EXPECTED_SPEC_VERSION} layout \
                     (spec has {} features, expected max {})",
                    first_unknown,
                    self.feature_names.len(),
                    expected_feature_names.len(),
                );
            }
            // Valid subset — some feature groups were disabled during training.
            tracing::info!(
                "feature spec has {} features (expected {} with all groups enabled) — {} groups disabled",
                self.feature_names.len(),
                expected_feature_names.len(),
                expected_feature_names.len() - self.feature_names.len(),
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

/// Logic gap categories (v16 group 19).
const LOGIC_GAP_CATEGORIES: &[&str] = &["crypto", "network", "process"];

/// Intent gap categories (v16 group 22).
const INTENT_GAP_CATEGORIES: &[&str] = &["network", "filesystem", "execution", "crypto"];

/// Expected ghosts (v16 group 23).
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
    metric_vocab: &[String],
    crit_unigram_vocab: &[String],
    crit_bigram_vocab: &[String],
    crit_trigram_vocab: &[String],
    attack_bigram_vocab: &[String],
    attack_trigram_vocab: &[String],
    mbc_bigram_vocab: &[String],
    mbc_trigram_vocab: &[String],
) -> Vec<String> {
    let mut feature_names = Vec::with_capacity(20000); // overestimate to avoid reallocs

    // Group 1: present
    for path in presence_vocab {
        feature_names.push(format!("present:{path}"));
    }

    // Group 2: maxcrit
    for path in presence_vocab {
        feature_names.push(format!("maxcrit:{path}"));
    }

    // Group 3: agg (50 features in v16)
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
        "agg:hostile_escalation_rate".to_string(),
        "agg:hostile_share_of_suspicious".to_string(),
        "agg:suspicious_finding_escalation_rate".to_string(),
        "agg:hostile_finding_escalation_rate".to_string(),
        "agg:hostile_share_of_suspicious_findings".to_string(),
        "agg:hostile_weighted_density".to_string(),
        format!("agg:top{TOP_K_RISK_FILES}_file_hostile_weighted_density_sum"),
        "agg:suspicious_id_repeat_ratio".to_string(),
        "agg:hostile_id_repeat_ratio".to_string(),
        "agg:suspicious_category_repeat_ratio".to_string(),
        "agg:hostile_category_repeat_ratio".to_string(),
        "agg:file_hostile_fraction".to_string(),
        "agg:file_suspicious_fraction".to_string(),
        "agg:file_notable_fraction".to_string(),
        "agg:file_hostile_count_log".to_string(),
        "agg:file_suspicious_count_log".to_string(),
        "agg:file_notable_count_log".to_string(),
        "agg:hostile_depth_weight".to_string(),
        "agg:suspicious_2level_breadth".to_string(),
        "agg:hostile_2level_breadth".to_string(),
        "agg:objectives_breadth".to_string(),
        "agg:attack_technique_count".to_string(),
        "agg:attack_tactic_count".to_string(),
        "agg:mbc_behavior_count".to_string(),
        "agg:has_attack_and_objective".to_string(),
    ]);

    // Crit-category n-grams (vocab-driven)
    for cu in crit_unigram_vocab { feature_names.push(format!("crit:{cu}")); }
    for cb in crit_bigram_vocab { feature_names.push(format!("critbi:{cb}")); }
    for ct in crit_trigram_vocab { feature_names.push(format!("crittri:{ct}")); }

    // ATT&CK/MBC code n-grams (vocab-driven)
    for ab in attack_bigram_vocab { feature_names.push(format!("atkbi:{ab}")); }
    for at in attack_trigram_vocab { feature_names.push(format!("atktri:{at}")); }
    for mb in mbc_bigram_vocab { feature_names.push(format!("mbcbi:{mb}")); }
    for mt in mbc_trigram_vocab { feature_names.push(format!("mbctri:{mt}")); }

    // Group 4: ext (6)
    feature_names.extend([
        "ext:third_party_max_crit".to_string(),
        "ext:third_party_count".to_string(),
        "ext:well_known_max_crit".to_string(),
        "ext:well_known_hostile_count".to_string(),
        "ext:well_known_suspicious_count".to_string(),
        "ext:has_yara_match".to_string(),
    ]);

    // Group 5: metrics (16 base + dynamic extended vocab)
    for &(group, field_name, _) in KEY_METRICS {
        feature_names.push(format!("metrics:{group}_{field_name}"));
    }
    for mk in metric_vocab {
        feature_names.push(format!("metrics:{mk}"));
    }

    // Group 6: filetype
    for filetype in filetype_vocab {
        feature_names.push(format!("filetype:{filetype}"));
    }

    // Group 7: struct base 7 + extensions
    feature_names.extend([
        "struct:tiny_executable".to_string(),
        "struct:no_imports".to_string(),
        "struct:zero_findings".to_string(),
        "struct:finding_count_log".to_string(),
        "struct:file_count_log".to_string(),
        "struct:inner_file_count_log".to_string(),
        "struct:stealth_potential".to_string(),
        "struct:suspicious_file_fraction".to_string(),
        "struct:hostile_file_fraction".to_string(),
        "struct:suspicious_file_count_log".to_string(),
        "struct:hostile_file_count_log".to_string(),
    ]);

    // Group 8: elements
    for el in element_vocab {
        feature_names.push(format!("elements:{el}"));
    }

    // Group 9: formula
    feature_names.extend([
        "formula:skeleton_len".to_string(),
        "formula:unique_elements".to_string(),
        "formula:complexity_ratio".to_string(),
    ]);

    // Group 10: score + inter
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

    // Group 13: skeleton
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
    feature_names.push("struct:air_gap_signal".to_string());
    feature_names.push("struct:anachronistic_injection".to_string());
    feature_names.push("struct:code_entropy_spike".to_string());
    feature_names.push("struct:foreign_binary_signal".to_string());
    feature_names.push("struct:extension_mismatch_signal".to_string());
    feature_names.push("struct:hostile_finding_density".to_string());

    // Group 16: trigrams
    for tri in trigram_vocab {
        feature_names.push(format!("trigram:{tri}"));
    }

    // Group 19: logic gaps
    for cat in LOGIC_GAP_CATEGORIES {
        feature_names.push(format!("gap:{cat}"));
    }

    // Group 20: unsigned bigrams
    for bi in bigram_vocab {
        feature_names.push(format!("unsigned_bigram:{bi}"));
    }

    // Group 22: intent gaps
    for cat in INTENT_GAP_CATEGORIES {
        feature_names.push(format!("intent_gap:{cat}"));
    }

    // Group 23: negative space
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
    presence_lookup: HashMap<String, usize>,
    n_presence: usize,
    n_ft: usize,
    n_element: usize,
    n_bigram: usize,
    n_ghost: usize,
    n_skeleton: usize,
    n_rare: usize,
    n_trigram: usize,
    n_ext_metrics: usize,
    metric_vocab: Vec<String>,
    n_crit_unigram: usize,
    n_crit_bigram: usize,
    n_crit_trigram: usize,
    n_atk_bigram: usize,
    n_atk_trigram: usize,
    n_mbc_bigram: usize,
    n_mbc_trigram: usize,
    /// Global feature name → index lookup for vocab-based features.
    absolute_lookup: HashMap<String, usize>,
    ghost_vocab: Vec<String>,
    total_features: usize,

    // Optimized bigram/trigram lookups
    path_to_id: HashMap<String, u32>,
    bigram_id_lookup: HashMap<(u32, u32), usize>,
    trigram_id_lookup: HashMap<(u32, u32, u32), usize>,
}

impl ExtractContext {
    /// Build lookup tables from a feature specification.
    #[must_use]
    pub fn new(spec: &FeatureSpec) -> Self {
        let presence_lookup: HashMap<String, usize> = spec
            .presence_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        // Optimized bigram/trigram lookups
        let mut path_to_id = HashMap::new();
        let mut next_id = 0u32;
        let mut get_id = |path: &str| {
            if let Some(&id) = path_to_id.get(path) {
                id
            } else {
                let id = next_id;
                path_to_id.insert(path.to_string(), id);
                next_id += 1;
                id
            }
        };

        let mut bigram_id_lookup = HashMap::new();
        for (i, bi_str) in spec.bigram_vocab.iter().enumerate() {
            let parts: Vec<&str> = bi_str.split(" + ").collect();
            if parts.len() == 2 {
                let id1 = get_id(parts[0]);
                let id2 = get_id(parts[1]);
                bigram_id_lookup.insert((id1.min(id2), id1.max(id2)), i);
            }
        }

        let mut trigram_id_lookup = HashMap::new();
        for (i, tri_str) in spec.trigram_vocab.iter().enumerate() {
            let parts: Vec<&str> = tri_str.split(" + ").collect();
            if parts.len() == 3 {
                let mut ids = [get_id(parts[0]), get_id(parts[1]), get_id(parts[2])];
                ids.sort();
                trigram_id_lookup.insert((ids[0], ids[1], ids[2]), i);
            }
        }

        Self {
            presence_lookup,
            n_presence: spec.presence_vocab.len(),
            n_ft: spec.filetype_vocab.len(),
            n_element: spec.element_vocab.len(),
            n_bigram: spec.bigram_vocab.len(),
            n_ghost: spec.ghost_vocab.len(),
            n_skeleton: spec.skeleton_vocab.len(),
            n_rare: spec.rare_element_vocab.len(),
            n_trigram: spec.trigram_vocab.len(),
            n_ext_metrics: spec.metric_vocab.len(),
            metric_vocab: spec.metric_vocab.clone(),
            n_crit_unigram: spec.crit_unigram_vocab.len(),
            n_crit_bigram: spec.crit_bigram_vocab.len(),
            n_crit_trigram: spec.crit_trigram_vocab.len(),
            n_atk_bigram: spec.attack_bigram_vocab.len(),
            n_atk_trigram: spec.attack_trigram_vocab.len(),
            n_mbc_bigram: spec.mbc_bigram_vocab.len(),
            n_mbc_trigram: spec.mbc_trigram_vocab.len(),
            absolute_lookup: spec.feature_names.iter().enumerate()
                .map(|(i, n)| (n.clone(), i))
                .collect(),
            ghost_vocab: spec.ghost_vocab.clone(),
            total_features: spec.total_features,
            path_to_id,
            bigram_id_lookup,
            trigram_id_lookup,
        }
    }

    /// Extract features from a cleave AnalysisReport serialized as JSON.
    #[must_use]
    pub fn extract(&self, report: &serde_json::Value) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.total_features];
        self.extract_report_into(report, &mut vec);
        vec
    }

    /// Extract features for a single compact file entry.
    #[must_use]
    pub fn extract_file(&self, file: &serde_json::Value) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.total_features];
        self.extract_files_into(std::slice::from_ref(&file), Some(file), &mut vec);
        vec
    }

    fn extract_report_into(&self, report: &serde_json::Value, vec: &mut [f32]) {
        let raw_files = report_files(report);
        let primary_file = primary_file(report);
        self.extract_files_into(&raw_files, primary_file, vec);
    }

    fn extract_files_into(
        &self,
        raw_files: &[&serde_json::Value],
        primary_file: Option<&serde_json::Value>,
        vec: &mut [f32],
    ) {
        // Small warm-cache reports are cheaper to summarize serially than to
        // fan out into rayon jobs.
        let file_summaries: Vec<FileSummary> = if raw_files.len() < 8 {
            raw_files.iter().map(|&f| FileSummary::new(f)).collect()
        } else {
            raw_files
                .par_iter()
                .map(|&f| FileSummary::new(f))
                .collect()
        };

        // If no files, we need at least one empty summary for structural logic.
        let summaries = if file_summaries.is_empty() {
            vec![FileSummary::default()]
        } else {
            file_summaries
        };

        let combined = summarize_report_summaries(&summaries);
        let merged_metrics = merge_metric_summaries(&summaries);
        let mut offsets = FeatureCursor::default();

        let (formula_str, elements_str, sample_score) =
            canonical_fields_from_primary_file(primary_file);
        let score_weight: f32 = if sample_score > 0 {
            (sample_score as f32).ln_1p()
        } else {
            1.0
        };

        // G1: Presence
        let presence_offset = offsets.take(self.n_presence);
        self.write_presence_features_v16(&combined, vec, presence_offset, score_weight);

        // G2: Max crit
        let maxcrit_offset = offsets.take(self.n_presence);
        self.write_max_crit_features_v16(&combined, vec, maxcrit_offset, score_weight);

        // G3: Aggregates
        let n_crit = self.n_crit_unigram + self.n_crit_bigram + self.n_crit_trigram;
        let n_code_ngrams = self.n_atk_bigram + self.n_atk_trigram + self.n_mbc_bigram + self.n_mbc_trigram;
        offsets.take(57 + n_crit + n_code_ngrams);
        write_aggregate_features(&combined, &summaries, &mut FeatureWriter { vec, lookup: &self.absolute_lookup });

        // Crit-category n-grams
        {
            let crit_categories: HashSet<&str> = [
                "objectives", "well-known", "supply-chain", "anti-analysis", "anti-static",
                "command-and-control", "evasion", "execution", "exfiltration",
            ].into_iter().collect();
            let crit_pfx = |c: u32| match c { 5 => "h", 4 => "s", _ => "n" };
            let mut pmc: HashMap<String, u32> = HashMap::new();
            for (path, &mo) in &combined.sample_paths {
                if mo < 3 { continue; }
                let parts: Vec<&str> = path.split('/').collect();
                if !crit_categories.contains(parts[0]) { continue; }
                let key = if parts.len() >= 2 { format!("{}/{}", parts[0], parts[1]) } else { parts[0].to_string() };
                let e = pmc.entry(key).or_insert(0);
                *e = (*e).max(mo);
            }
            let mut tokens: Vec<String> = pmc.iter().map(|(k, &c)| format!("{}:{k}", crit_pfx(c))).collect();
            tokens.sort();
            let lookup = &self.absolute_lookup;
            for t in &tokens { if let Some(&i) = lookup.get(&format!("crit:{t}")) { vec[i] = 1.0; } }

            // Safety: trigrams are O(N^3), bigrams O(N^2). Cap N to avoid complexity bombs on bloated reports.
            if tokens.len() <= 512 {
                for (i, t1) in tokens.iter().enumerate() {
                    for t2 in &tokens[i+1..] {
                        if let Some(&idx) = lookup.get(&format!("critbi:{t1} + {t2}")) { vec[idx] = 1.0; }
                    }
                    if tokens.len() <= 128 {
                        for j in i+1..tokens.len() {
                            for t3 in &tokens[j+1..] {
                                if let Some(&idx) = lookup.get(&format!("crittri:{t1} + {} + {t3}", tokens[j])) { vec[idx] = 1.0; }
                            }
                        }
                    }
                }
            } else {
                tracing::warn!(tokens = tokens.len(), "too many unique crit tokens; skipping n-gram generation");
            }
        }

        // ATT&CK/MBC code n-grams
        {
            let mut attacks: HashSet<String> = HashSet::new();
            let mut mbcs: HashSet<String> = HashSet::new();
            for s in &summaries {
                for f in &s.raw_findings {
                    if let Some(a) = f.get("a").and_then(|v| v.as_str()) { attacks.insert(a.to_string()); }
                    if let Some(m) = f.get("m").and_then(|v| v.as_str()) { mbcs.insert(m.to_string()); }
                }
            }
            let lookup = &self.absolute_lookup;
            let mut sa: Vec<_> = attacks.iter().collect(); sa.sort();
            if sa.len() <= 512 {
                for (i, a1) in sa.iter().enumerate() {
                    for (j, a2) in sa[i+1..].iter().enumerate() {
                        if let Some(&idx) = lookup.get(&format!("atkbi:{a1} + {a2}")) { vec[idx] = 1.0; }
                        if sa.len() <= 128 {
                            for a3 in &sa[i+1+j+1..] {
                                if let Some(&idx) = lookup.get(&format!("atktri:{a1} + {a2} + {a3}")) { vec[idx] = 1.0; }
                            }
                        }
                    }
                }
            } else {
                tracing::warn!(tokens = sa.len(), "too many unique ATT&CK tokens; skipping n-gram generation");
            }

            let mut sm: Vec<_> = mbcs.iter().collect(); sm.sort();
            if sm.len() <= 512 {
                for (i, m1) in sm.iter().enumerate() {
                    for (j, m2) in sm[i+1..].iter().enumerate() {
                        if let Some(&idx) = lookup.get(&format!("mbcbi:{m1} + {m2}")) { vec[idx] = 1.0; }
                        if sm.len() <= 128 {
                            for m3 in &sm[i+1+j+1..] {
                                if let Some(&idx) = lookup.get(&format!("mbctri:{m1} + {m2} + {m3}")) { vec[idx] = 1.0; }
                            }
                        }
                    }
                }
            } else {
                tracing::warn!(tokens = sm.len(), "too many unique MBC tokens; skipping n-gram generation");
            }
        }

        // G4: External (name-based)
        offsets.take(6); // reserve space for offset tracking compatibility
        write_external_summary_features(&combined, &mut FeatureWriter { vec, lookup: &self.absolute_lookup });

        // G5: Metrics (base + extended vocab)
        offsets.take(KEY_METRICS.len() + self.n_ext_metrics);
        write_metric_features(&merged_metrics, &mut FeatureWriter { vec, lookup: &self.absolute_lookup }, &self.metric_vocab);

        // G6: Filetype (blindfolded in v16)
        let _file_type_offset = offsets.take(self.n_ft);

        // G7: Structural
        offsets.take(11);
        write_structural_features(&mut FeatureWriter { vec, lookup: &self.absolute_lookup }, &summaries, combined.filtered_finding_count);

        // G8: Elements
        offsets.take(self.n_element);
        {
            let w = &mut FeatureWriter { vec, lookup: &self.absolute_lookup };
            if !elements_str.is_empty() {
                for el in elements_str.split(',') {
                    let el = el.trim();
                    w.set(&format!("elements:{el}"), 1.0);
                }
            }

            // G9: Formula
            let skeleton_str: String = formula_str.chars().filter(|c| c.is_alphabetic()).collect();
            let unique_skel_chars: HashSet<char> = skeleton_str.chars().collect();
            w.set("formula:skeleton_len", skeleton_str.chars().count() as f32);
            w.set("formula:unique_elements", unique_skel_chars.len() as f32);
            if combined.filtered_finding_count > 0 {
                w.set("formula:complexity_ratio", formula_str.chars().count() as f32 / combined.filtered_finding_count as f32);
            }

            // G10: Score
            let total_size_bytes: f64 = summaries.iter().map(|s| s.size_bytes).sum();
            w.set("score:hopper_score", sample_score as f32);
            w.set("score:density", if total_size_bytes > 0.0 {
                sample_score as f32 / (total_size_bytes as f32).ln_1p()
            } else { 0.0 });
            for s in &summaries {
                w.set(&format!("inter:{}*score", s.file_type), sample_score as f32);
            }

            // G12: Ghost
            for ghost_path in &self.ghost_vocab {
                let missing = match combined.sample_paths.get(ghost_path) {
                    Some(&max_ord) => max_ord < 2,
                    None => true,
                };
                if missing {
                    w.set(&format!("ghost:{ghost_path}"), 1.0);
                }
            }

            // G13: Skeleton
            if !skeleton_str.is_empty() {
                w.set(&format!("skeleton:{skeleton_str}"), 1.0);
            }

            // G14: Rare Elements
            if !elements_str.is_empty() {
                let weight: f32 = if combined.finding_confidences.is_empty() {
                    1.0
                } else {
                    let sum: f64 = combined.finding_confidences.iter().sum();
                    (sum / combined.finding_confidences.len() as f64) as f32
                };
                for el in elements_str.split(',') {
                    let el = el.trim();
                    w.set(&format!("rare:{el}"), weight);
                }
            }
        }
        offsets.take(3); // formula
        offsets.take(2 + self.n_ft); // score

        // G11: Bigrams (optimized — still uses offset for performance)
        let bigrams_offset = offsets.take(self.n_bigram);
        self.write_bigram_features_optimized(&summaries, vec, bigrams_offset);

        offsets.take(self.n_ghost); // ghost (now name-based above)
        offsets.take(self.n_skeleton); // skeleton (now name-based above)
        offsets.take(self.n_rare); // rare (now name-based above)

        // G15: Structural Extensions
        offsets.take(13);
        write_structural_extensions(&summaries, &combined, &mut FeatureWriter { vec, lookup: &self.absolute_lookup });

        // G16: Trigrams (optimized)
        let trigram_offset = offsets.take(self.n_trigram);
        self.write_trigram_features_optimized(&summaries, vec, trigram_offset);

        // G19: Logic Gaps
        offsets.take(LOGIC_GAP_CATEGORIES.len());
        write_logic_gap_features(&combined, &summaries, &mut FeatureWriter { vec, lookup: &self.absolute_lookup });

        // G20: Signature Synergy
        let unsigned_bigram_offset = offsets.take(self.n_bigram);
        if combined.sample_paths.contains_key("metadata/unsigned") {
            self.write_bigram_features_optimized(&summaries, vec, unsigned_bigram_offset);
        }

        // G22: Intent Gaps
        offsets.take(INTENT_GAP_CATEGORIES.len());
        write_intent_gap_features(&combined, &mut FeatureWriter { vec, lookup: &self.absolute_lookup });

        // G23: Negative Space
        let missing_count: usize = EXPECTED_GHOSTS.iter().map(|(_, t)| t.len()).sum();
        offsets.take(missing_count);
        write_negative_space_features(&combined, &summaries, &mut FeatureWriter { vec, lookup: &self.absolute_lookup });

        debug_assert_eq!(offsets.offset, self.total_features);
    }

    fn write_presence_features_v16(&self, summary: &FindingSummary, vec: &mut [f32], offset: usize, score_weight: f32) {
        for (path, &max_ord) in &summary.sample_paths {
            if max_ord >= 2 {
                if let Some(&idx) = self.presence_lookup.get(path.as_str()) {
                    let conf = summary.path_confidences.get(path).copied().unwrap_or(1.0) as f32;
                    vec[offset + idx] = score_weight * conf;
                }
            }
        }
    }

    fn write_max_crit_features_v16(&self, summary: &FindingSummary, vec: &mut [f32], offset: usize, score_weight: f32) {
        for (path, &max_ord) in &summary.sample_paths {
            if let Some(&idx) = self.presence_lookup.get(path.as_str()) {
                let conf = summary.path_confidences.get(path).copied().unwrap_or(1.0) as f32;
                vec[offset + idx] = max_ord as f32 * score_weight * conf;
            }
        }
    }

    fn write_bigram_features_optimized(&self, summaries: &[FileSummary], vec: &mut [f32], offset: usize) {
        for s in summaries {
            let ids: Vec<u32> = s.unique_3level_paths.iter().filter_map(|p| self.path_to_id.get(p).copied()).collect();
            if ids.len() > 512 {
                tracing::warn!(path = %s.path, tokens = ids.len(), "too many unique paths; skipping bigram generation for file");
                continue;
            }
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let key = (ids[i].min(ids[j]), ids[i].max(ids[j]));
                    if let Some(&idx) = self.bigram_id_lookup.get(&key) {
                        vec[offset + idx] = 1.0;
                    }
                }
            }
        }
    }

    fn write_trigram_features_optimized(&self, summaries: &[FileSummary], vec: &mut [f32], offset: usize) {
        for s in summaries {
            let ids: Vec<u32> = s.unique_3level_paths.iter().filter_map(|p| self.path_to_id.get(p).copied()).collect();
            let n = ids.len();
            if n > 128 {
                tracing::warn!(path = %s.path, tokens = n, "too many unique paths; skipping trigram generation for file");
                continue;
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        let mut sorted = [ids[i], ids[j], ids[k]];
                        sorted.sort();
                        if let Some(&idx) = self.trigram_id_lookup.get(&(sorted[0], sorted[1], sorted[2])) {
                            vec[offset + idx] = 1.0;
                        }
                    }
                }
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

/// Pre-calculated data for a single file entry in a report.
#[derive(Debug, Clone, Default)]
struct FileSummary {
    path: String,
    parent: String,
    file_type: String,
    size_bytes: f64,
    mtime: Option<f64>,
    overall_entropy: f64,
    metrics: HashMap<String, HashMap<String, f64>>,
    findings: FindingSummary,
    risk: FileRiskStats,
    unique_3level_paths: Vec<String>,
    imports: HashSet<String>,
    /// (finding_id, confidence, crit_ordinal) for cross-file unique ID dedup.
    raw_findings: Vec<serde_json::Value>,
    /// Whether the "is" key exists in the cleave JSON (even if empty).
    has_imports_key: bool,
}

impl FileSummary {
    fn new(file_entry: &serde_json::Value) -> Self {
        let findings_raw: Vec<&serde_json::Value> = file_entry["ts"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
        let findings = summarize_findings(&findings_raw);
        
        let size_bytes = file_entry["sz"].as_f64().unwrap_or(0.0);
        let size_kb = (size_bytes / 1024.0).max(1.0) as f32;
        let denom = findings.filtered_finding_count.max(1) as f32;
        let max_crit = findings.sample_paths.values().copied().max().unwrap_or(0);
        
        let risk = FileRiskStats {
            suspicious_ratio: findings.suspicious_finding_count as f32 / denom,
            hostile_ratio: findings.hostile_finding_count as f32 / denom,
            suspicious_findings: findings.suspicious_finding_count,
            hostile_findings: findings.hostile_finding_count,
            suspicious_density: findings.suspicious_finding_count as f32 / size_kb,
            hostile_density: findings.hostile_finding_count as f32 / size_kb,
            suspicious_category_breadth: findings.suspicious_category_breadth,
            hostile_category_breadth: findings.hostile_category_breadth,
            max_crit,
        };

        // N-gram paths: full base path (depth=0), filtered by min_crit=2
        // (baseline+). Must match Python's _ngram_paths_for_file behavior.
        let mut unique_3level_paths: Vec<String> = findings_raw
            .iter()
            .filter_map(|f| {
                let fid = f["i"].as_str()?;
                let conf = f["c"].as_f64().unwrap_or(1.0);
                let crit = crit_ordinal(f);
                (conf >= MIN_CONFIDENCE && crit >= 2)
                    .then(|| fid.split("::").next().unwrap_or(fid).to_string())
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        unique_3level_paths.sort();

        let metrics: HashMap<String, HashMap<String, f64>> = file_entry
            .get("ms")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(group, fields)| {
                        let group_map = fields
                            .as_object()?
                            .iter()
                            .filter_map(|(k, v)| {
                                // Handle numbers, booleans (true→1.0), and skip strings/nulls.
                                let val = v.as_f64()
                                    .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))?;
                                Some((k.clone(), val))
                            })
                            .collect();
                        Some((group.clone(), group_map))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let overall_entropy = metrics.get("binary").and_then(|m| m.get("overall_entropy")).copied().unwrap_or(0.0);

        let imports: HashSet<String> = file_entry
            .get("is")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|imp| {
                imp.as_str()
                    .or_else(|| imp.get("n").and_then(|v| v.as_str()))
            })
            .map(str::to_string)
            .collect();

        let raw_findings: Vec<serde_json::Value> = findings_raw.iter().map(|f| (*f).clone()).collect();
        let has_imports_key = file_entry.get("is").is_some();

        Self {
            path: file_entry["path"].as_str().unwrap_or("").to_string(),
            parent: file_entry["p"].as_str().unwrap_or("").to_string(),
            file_type: file_entry["type"].as_str().unwrap_or("").to_string(),
            size_bytes,
            mtime: file_entry["mt"].as_str().and_then(parse_iso8601),
            overall_entropy,
            metrics,
            findings,
            risk,
            unique_3level_paths,
            imports,
            raw_findings,
            has_imports_key,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct FindingSummary {
    sample_paths: HashMap<String, u32>,
    path_confidences: HashMap<String, f64>,
    finding_confidences: Vec<f64>,
    filtered_finding_count: u32,
    notable_finding_count: u32,
    suspicious_finding_count: u32,
    hostile_finding_count: u32,
    unique_notable_ids: usize,
    unique_suspicious_ids: usize,
    unique_hostile_ids: usize,
    suspicious_category_breadth: usize,
    hostile_category_breadth: usize,
    third_party_max_crit: u32,
    third_party_count: u32,
    well_known_max_crit: u32,
    well_known_hostile: u32,
    well_known_suspicious: u32,
    has_yara: bool,
}

fn summarize_findings(findings: &[&serde_json::Value]) -> FindingSummary {
    let mut summary = FindingSummary::default();
    let mut notable_ids: HashSet<&str> = HashSet::new();
    let mut suspicious_ids: HashSet<&str> = HashSet::new();
    let mut hostile_ids: HashSet<&str> = HashSet::new();

    for finding in findings {
        let fid = finding["i"].as_str().unwrap_or("");
        if fid.is_empty() { continue; }
        let conf = finding["c"].as_f64().unwrap_or(1.0);
        if conf < MIN_CONFIDENCE { continue; }
        
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
                if fid.starts_with("third_party/yara") { summary.has_yara = true; }
            }
            "well-known" => {
                summary.well_known_max_crit = summary.well_known_max_crit.max(crit_ord);
                if crit_ord >= 5 { summary.well_known_hostile += 1; }
                else if crit_ord >= 4 { summary.well_known_suspicious += 1; }
            }
            _ => {}
        }

        // `entry(path.to_owned())` would allocate a fresh `String` on every
        // call regardless of whether the key already exists. Most findings
        // share path prefixes, so hitting an existing entry is the common
        // case — `get_mut` + conditional `insert` pays for the allocation
        // only on first insert.
        for path in finding_paths(fid) {
            match summary.sample_paths.get_mut(path) {
                Some(v) => *v = (*v).max(crit_ord),
                None => { summary.sample_paths.insert(path.to_owned(), crit_ord); }
            }
            match summary.path_confidences.get_mut(path) {
                Some(v) => *v = v.max(conf),
                None => { summary.path_confidences.insert(path.to_owned(), conf); }
            }
        }
    }

    summary.unique_notable_ids = notable_ids.len();
    summary.unique_suspicious_ids = suspicious_ids.len();
    summary.unique_hostile_ids = hostile_ids.len();

    let mut susp_cats: HashSet<&str> = HashSet::new();
    let mut host_cats: HashSet<&str> = HashSet::new();
    for (path, &max_ord) in &summary.sample_paths {
        if max_ord >= 4 { susp_cats.insert(path.split('/').next().unwrap_or("")); }
        if max_ord >= 5 { host_cats.insert(path.split('/').next().unwrap_or("")); }
    }
    summary.suspicious_category_breadth = susp_cats.len();
    summary.hostile_category_breadth = host_cats.len();
    summary
}

fn summarize_report_summaries(summaries: &[FileSummary]) -> FindingSummary {
    let mut combined = FindingSummary::default();

    // Deduplicate unique IDs across all files (matching Python's second pass
    // in _summarize_report_files which iterates all findings again). The
    // finding IDs borrow from `summaries` for the lifetime of this function,
    // so the sets hold `&str` — matching `summarize_findings`'s shape — and
    // avoid one `String` allocation per qualifying finding per criticality
    // tier.
    let mut notable_ids: HashSet<&str> = HashSet::new();
    let mut suspicious_ids: HashSet<&str> = HashSet::new();
    let mut hostile_ids: HashSet<&str> = HashSet::new();

    for s in summaries {
        let fs = &s.findings;
        combined.filtered_finding_count += fs.filtered_finding_count;
        combined.notable_finding_count += fs.notable_finding_count;
        combined.suspicious_finding_count += fs.suspicious_finding_count;
        combined.hostile_finding_count += fs.hostile_finding_count;
        combined.third_party_count += fs.third_party_count;
        combined.well_known_hostile += fs.well_known_hostile;
        combined.well_known_suspicious += fs.well_known_suspicious;
        combined.third_party_max_crit = combined.third_party_max_crit.max(fs.third_party_max_crit);
        combined.well_known_max_crit = combined.well_known_max_crit.max(fs.well_known_max_crit);
        combined.has_yara |= fs.has_yara;

        // Same motivation as `summarize_findings`: skip the `String` clone
        // when the path already aggregates into `combined`.
        for (path, max_ord) in &fs.sample_paths {
            match combined.sample_paths.get_mut(path) {
                Some(v) => *v = (*v).max(*max_ord),
                None => { combined.sample_paths.insert(path.clone(), *max_ord); }
            }
        }
        for (path, &conf) in &fs.path_confidences {
            match combined.path_confidences.get_mut(path) {
                Some(v) => *v = v.max(conf),
                None => { combined.path_confidences.insert(path.clone(), conf); }
            }
        }
        combined.finding_confidences.extend(fs.finding_confidences.iter().copied());

        // Re-scan raw findings to deduplicate unique IDs across files.
        for finding in &s.raw_findings {
            let fid = finding["i"].as_str().unwrap_or("");
            if fid.is_empty() { continue; }
            let conf = finding["c"].as_f64().unwrap_or(1.0);
            if conf < MIN_CONFIDENCE { continue; }
            let crit = crit_ordinal(finding);
            if crit >= 3 { notable_ids.insert(fid); }
            if crit >= 4 { suspicious_ids.insert(fid); }
            if crit >= 5 { hostile_ids.insert(fid); }
        }
    }

    combined.unique_notable_ids = notable_ids.len();
    combined.unique_suspicious_ids = suspicious_ids.len();
    combined.unique_hostile_ids = hostile_ids.len();

    let mut susp_cats: HashSet<&str> = HashSet::new();
    let mut host_cats: HashSet<&str> = HashSet::new();
    for (path, &max_ord) in &combined.sample_paths {
        if max_ord >= 4 { susp_cats.insert(path.split('/').next().unwrap_or("")); }
        if max_ord >= 5 { host_cats.insert(path.split('/').next().unwrap_or("")); }
    }
    combined.suspicious_category_breadth = susp_cats.len();
    combined.hostile_category_breadth = host_cats.len();
    combined
}

#[derive(Debug, Clone, Default)]
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

fn write_aggregate_features(summary: &FindingSummary, summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
    let mut max_crit = 0u32;
    let mut categories: HashSet<&str> = HashSet::new();
    let mut path_breadth_any = 0u32;
    let mut total_active = 0u32;
    let mut breadth_notable = 0u32;
    let mut breadth_suspicious = 0u32;
    let mut breadth_hostile = 0u32;
    let mut breadth_notable_only = 0u32;

    for (path, &max_ord) in &summary.sample_paths {
        let path_depth = path.chars().filter(|&c| c == '/').count();
        if max_ord >= 2 {
            categories.insert(path.split('/').next().unwrap_or(""));
            if path_depth >= 2 { path_breadth_any += 1; }
        }
        if path_depth < 2 || max_ord < 3 { continue; }
        total_active += 1;
        breadth_notable += 1;
        max_crit = max_crit.max(max_ord);
        if max_ord >= 4 { breadth_suspicious += 1; }
        if max_ord >= 5 { breadth_hostile += 1; }
        else if max_ord == 3 { breadth_notable_only += 1; }
    }

    let total_size_bytes: f64 = summaries.iter().map(|s| s.size_bytes).sum();
    let total_kb_raw = (total_size_bytes / 1024.0) as f32;
    let total_kb_p1 = total_kb_raw.max(0.1);
    let total_kb_1 = total_kb_raw.max(1.0);

    w.set("agg:max_crit", max_crit as f32);
    w.set("agg:category_breadth", categories.len() as f32);
    w.set("agg:path_breadth_any", (path_breadth_any as f32).ln_1p());
    w.set("agg:total_active_paths", (total_active as f32).ln_1p());
    w.set("agg:suspicious_concentration", breadth_suspicious as f32 / path_breadth_any.max(1) as f32);
    w.set("agg:hostile_concentration", breadth_hostile as f32 / path_breadth_any.max(1) as f32);
    w.set("agg:escalation_rate", breadth_suspicious as f32 / breadth_notable.max(1) as f32);
    w.set("agg:notable_only_fraction", breadth_notable_only as f32 / breadth_notable.max(1) as f32);
    w.set("agg:notable_findings_log", (summary.notable_finding_count as f32).ln_1p());
    w.set("agg:suspicious_findings_log", (summary.suspicious_finding_count as f32).ln_1p());
    w.set("agg:hostile_findings_log", (summary.hostile_finding_count as f32).ln_1p());
    w.set("agg:notable_finding_ratio", summary.notable_finding_count as f32 / total_kb_p1);
    w.set("agg:suspicious_finding_ratio", summary.suspicious_finding_count as f32 / total_kb_p1);
    w.set("agg:hostile_finding_ratio", summary.hostile_finding_count as f32 / total_kb_p1);
    let log_kb_p1 = total_kb_p1.ln_1p();
    w.set("agg:unique_suspicious_ids_log", (summary.unique_suspicious_ids as f32).ln_1p() / log_kb_p1);
    w.set("agg:unique_hostile_ids_log", (summary.unique_hostile_ids as f32).ln_1p() / log_kb_p1);

    let topk = topk_file_risk_features_from_summaries(summaries);
    w.set("agg:top1_file_suspicious_ratio_sum", topk[0]);
    w.set("agg:top1_file_hostile_ratio_sum", topk[1]);
    w.set("agg:top1_file_suspicious_findings_log", topk[2]);
    w.set("agg:top1_file_hostile_findings_log", topk[3]);

    let category_denom = categories.len().max(1) as f32;
    w.set("agg:suspicious_category_breadth", summary.suspicious_category_breadth as f32);
    w.set("agg:hostile_category_breadth", summary.hostile_category_breadth as f32);
    w.set("agg:suspicious_category_density", summary.suspicious_category_breadth as f32 / category_denom);
    w.set("agg:hostile_category_density", summary.hostile_category_breadth as f32 / category_denom);
    w.set("agg:suspicious_findings_per_kb", summary.suspicious_finding_count as f32 / total_kb_1);
    w.set("agg:hostile_findings_per_kb", summary.hostile_finding_count as f32 / total_kb_1);
    w.set("agg:suspicious_categories_per_kb", summary.suspicious_category_breadth as f32 / total_kb_1);
    w.set("agg:hostile_categories_per_kb", summary.hostile_category_breadth as f32 / total_kb_1);
    w.set("agg:top1_file_suspicious_density_sum", topk[4]);
    w.set("agg:top1_file_hostile_density_sum", topk[5]);
    w.set("agg:top1_file_suspicious_category_breadth_sum", topk[6]);
    w.set("agg:top1_file_hostile_category_breadth_sum", topk[7]);

    w.set("agg:hostile_escalation_rate", breadth_hostile as f32 / breadth_notable.max(1) as f32);
    w.set("agg:hostile_share_of_suspicious", breadth_hostile as f32 / breadth_suspicious.max(1) as f32);
    w.set("agg:suspicious_finding_escalation_rate", summary.suspicious_finding_count as f32 / summary.notable_finding_count.max(1) as f32);
    w.set("agg:hostile_finding_escalation_rate", summary.hostile_finding_count as f32 / summary.notable_finding_count.max(1) as f32);
    w.set("agg:hostile_share_of_suspicious_findings", summary.hostile_finding_count as f32 / summary.suspicious_finding_count.max(1) as f32);

    let host_density_global = summary.hostile_finding_count as f32 / total_kb_1;
    let susp_density_global = summary.suspicious_finding_count as f32 / total_kb_1;
    w.set("agg:hostile_weighted_density", host_density_global + 0.25 * susp_density_global);

    let mut stats: Vec<FileRiskStats> = summaries.iter().map(|s| s.risk.clone()).collect();
    stats.sort_by(|a, b| {
        let ka = (a.hostile_density + 0.25 * a.suspicious_density, a.hostile_density, a.suspicious_density);
        let kb = (b.hostile_density + 0.25 * b.suspicious_density, b.hostile_density, b.suspicious_density);
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_weighted: f32 = stats.iter().take(TOP_K_RISK_FILES).map(|s| s.hostile_density + 0.25 * s.suspicious_density).sum();
    w.set("agg:top1_file_hostile_weighted_density_sum", top_weighted);

    w.set("agg:suspicious_id_repeat_ratio", 1.0 - (summary.unique_suspicious_ids as f32 / summary.suspicious_finding_count.max(1) as f32));
    w.set("agg:hostile_id_repeat_ratio", 1.0 - (summary.unique_hostile_ids as f32 / summary.hostile_finding_count.max(1) as f32));
    w.set("agg:suspicious_category_repeat_ratio", 1.0 - (summary.suspicious_category_breadth as f32 / summary.suspicious_finding_count.max(1) as f32));
    w.set("agg:hostile_category_repeat_ratio", 1.0 - (summary.hostile_category_breadth as f32 / summary.hostile_finding_count.max(1) as f32));

    let n_files = summaries.len().max(1) as f32;
    let hostile_files = summaries.iter().filter(|s| s.risk.max_crit >= 5).count() as f32;
    let suspicious_files = summaries.iter().filter(|s| s.risk.max_crit == 4).count() as f32;
    let notable_files = summaries.iter().filter(|s| s.risk.max_crit == 3).count() as f32;
    w.set("agg:file_hostile_fraction", hostile_files / n_files);
    w.set("agg:file_suspicious_fraction", suspicious_files / n_files);
    w.set("agg:file_notable_fraction", notable_files / n_files);
    w.set("agg:file_hostile_count_log", hostile_files.ln_1p());
    w.set("agg:file_suspicious_count_log", suspicious_files.ln_1p());
    w.set("agg:file_notable_count_log", notable_files.ln_1p());
    w.set("agg:hostile_depth_weight", 0.0);

    // 2-level breadth features.
    let mut suspicious_2level: HashSet<String> = HashSet::new();
    let mut hostile_2level: HashSet<String> = HashSet::new();
    let mut objectives_2level: HashSet<String> = HashSet::new();
    for (path, &max_ord) in &summary.sample_paths {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() >= 2 {
            let two_level = format!("{}/{}", parts[0], parts[1]);
            if max_ord >= 4 { suspicious_2level.insert(two_level.clone()); }
            if max_ord >= 5 { hostile_2level.insert(two_level.clone()); }
            if parts[0] == "objectives" && max_ord >= 2 { objectives_2level.insert(two_level); }
        }
    }
    w.set("agg:suspicious_2level_breadth", suspicious_2level.len() as f32);
    w.set("agg:hostile_2level_breadth", hostile_2level.len() as f32);
    w.set("agg:objectives_breadth", objectives_2level.len() as f32);

    // ATT&CK / MBC features from 'a' and 'm' fields in findings.
    let mut attack_techniques: HashSet<String> = HashSet::new();
    let mut mbc_behaviors: HashSet<String> = HashSet::new();
    for s in summaries {
        for finding in &s.raw_findings {
            if let Some(a) = finding.get("a").and_then(|v| v.as_str()) {
                attack_techniques.insert(a.to_string());
            }
            if let Some(m) = finding.get("m").and_then(|v| v.as_str()) {
                mbc_behaviors.insert(m.to_string());
            }
        }
    }
    w.set("agg:attack_technique_count", attack_techniques.len() as f32);
    #[allow(clippy::string_slice)]
    let tactic_prefixes: HashSet<&str> = attack_techniques.iter()
        .filter(|t| t.starts_with('T') && t.len() >= 4)
        .map(|t| &t[..4])
        .collect();
    w.set("agg:attack_tactic_count", tactic_prefixes.len() as f32);
    w.set("agg:mbc_behavior_count", mbc_behaviors.len() as f32);
    let has_objectives = summary.sample_paths.keys().any(|p| p.starts_with("objectives/"));
    w.set("agg:has_attack_and_objective", if !attack_techniques.is_empty() && has_objectives { 1.0 } else { 0.0 });
}

fn topk_file_risk_features_from_summaries(summaries: &[FileSummary]) -> [f32; 8] {
    if summaries.is_empty() || TOP_K_RISK_FILES == 0 { return [0.0; 8]; }
    let mut stats: Vec<FileRiskStats> = summaries.iter().map(|s| s.risk.clone()).collect();
    
    let mut by_susp = stats.clone();
    by_susp.sort_by(|a, b| {
        (b.suspicious_ratio, b.suspicious_findings, b.hostile_ratio, b.hostile_findings)
            .partial_cmp(&(a.suspicious_ratio, a.suspicious_findings, a.hostile_ratio, a.hostile_findings))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_susp = &by_susp[..TOP_K_RISK_FILES.min(by_susp.len())];

    stats.sort_by(|a, b| {
        (b.hostile_ratio, b.hostile_findings, b.suspicious_ratio, b.suspicious_findings)
            .partial_cmp(&(a.hostile_ratio, a.hostile_findings, a.suspicious_ratio, a.suspicious_findings))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_host = &stats[..TOP_K_RISK_FILES.min(stats.len())];

    [
        top_susp.iter().map(|s| s.suspicious_ratio).sum(),
        top_host.iter().map(|s| s.hostile_ratio).sum(),
        (top_susp.iter().map(|s| s.suspicious_findings).sum::<u32>() as f32 + 1.0).ln(),
        (top_host.iter().map(|s| s.hostile_findings).sum::<u32>() as f32 + 1.0).ln(),
        top_susp.iter().map(|s| s.suspicious_density).sum(),
        top_host.iter().map(|s| s.hostile_density).sum(),
        top_susp.iter().map(|s| s.suspicious_category_breadth as f32).sum(),
        top_host.iter().map(|s| s.hostile_category_breadth as f32).sum(),
    ]
}

fn parse_iso8601(s: &str) -> Option<f64> {
    let s = s.trim().replace(' ', "T");
    let bytes = s.as_bytes();
    if bytes.len() < 19 { return None; }
    let year: i32 = std::str::from_utf8(&bytes[0..4]).ok()?.parse().ok()?;
    let month: u32 = std::str::from_utf8(&bytes[5..7]).ok()?.parse().ok()?;
    let day: u32 = std::str::from_utf8(&bytes[8..10]).ok()?.parse().ok()?;
    let hour: i64 = std::str::from_utf8(&bytes[11..13]).ok()?.parse().ok()?;
    let minute: i64 = std::str::from_utf8(&bytes[14..16]).ok()?.parse().ok()?;
    let second: i64 = std::str::from_utf8(&bytes[17..19]).ok()?.parse().ok()?;

    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days_since_epoch = era as i64 * 146097 + doe as i64 - 719468;

    let mut total = days_since_epoch * 86400 + hour * 3600 + minute * 60 + second;

    let mut idx = 19;
    let mut frac = 0.0_f64;
    if idx < bytes.len() && bytes[idx] == b'.' {
        idx += 1;
        let start = idx;
        while idx < bytes.len() && bytes[idx].is_ascii_digit() { idx += 1; }
        if idx > start {
            let frac_str = std::str::from_utf8(&bytes[start..idx]).ok()?;
            frac = frac_str.parse::<f64>().ok()? / 10_f64.powi((idx - start) as i32);
        }
    }
    if idx < bytes.len() {
        let tz = bytes[idx];
        if tz == b'+' || tz == b'-' {
            let sign: i64 = if tz == b'+' { -1 } else { 1 };
            idx += 1;
            if idx + 5 <= bytes.len() && bytes[idx + 2] == b':' {
                let oh: i64 = std::str::from_utf8(&bytes[idx..idx + 2]).ok()?.parse().ok()?;
                let om: i64 = std::str::from_utf8(&bytes[idx + 3..idx + 5]).ok()?.parse().ok()?;
                total += sign * (oh * 3600 + om * 60);
            }
        }
    }
    Some(total as f64 + frac)
}

fn write_structural_extensions(summaries: &[FileSummary], combined: &FindingSummary, w: &mut FeatureWriter<'_>) {
    let binary_like = ["pe", "elf", "macho"];
    let source_types = ["javascript", "python", "typescript", "ruby", "php"];
    let text_exts = ["txt", "md", "json", "png", "jpg"];

    let mut mtimes = Vec::new();
    let mut hostile_mtimes = Vec::new();
    let mut entropies = Vec::new();
    let mut code_entropies = Vec::new();
    let mut max_entropy = 0.0_f64;
    let mut hostile_files = 0;
    let mut hostile_files_with_parent = 0;
    let mut inner_file_count = 0;
    let mut total_loc = 0;
    let mut extension_mismatches = 0;
    let mut has_source_files = false;
    let mut has_foreign_binaries = false;

    let mut depths = HashMap::new();
    for s in summaries {
        if s.parent.is_empty() { depths.insert(&s.path, 0); }
        else {
            let pd = depths.get(&s.parent).copied().unwrap_or(0);
            depths.insert(&s.path, pd + 1);
        }
    }
    let max_nesting_depth = depths.values().copied().max().unwrap_or(0);

    for s in summaries {
        if !s.parent.is_empty() { inner_file_count += 1; }
        if source_types.contains(&s.file_type.as_str()) { has_source_files = true; }
        if binary_like.contains(&s.file_type.as_str()) && has_source_files { has_foreign_binaries = true; }

        let lines = s.metrics.get("text").and_then(|t| t.get("total_lines")).copied().unwrap_or(0.0);
        total_loc += lines as u64;

        if !s.path.is_empty() && s.path.contains('.') {
            if let Some(ext) = s.path.rsplit('.').next() {
                if binary_like.contains(&s.file_type.as_str()) && text_exts.contains(&ext.to_ascii_lowercase().as_str()) {
                    extension_mismatches += 1;
                }
            }
        }

        if let Some(t) = s.mtime { mtimes.push(t); }
        if s.overall_entropy > 0.0 { entropies.push(s.overall_entropy); }
        max_entropy = max_entropy.max(s.overall_entropy);

        if s.findings.hostile_finding_count > 0 {
            hostile_files += 1;
            if let Some(t) = s.mtime { hostile_mtimes.push(t); }
            if !s.parent.is_empty() { hostile_files_with_parent += 1; }
        }

        let code_types = ["javascript", "python", "pe", "elf", "macho"];
        if s.overall_entropy > 0.0 && code_types.contains(&s.file_type.as_str()) {
            code_entropies.push(s.overall_entropy);
        }
    }

    w.set("struct:packaged_capability", (combined.sample_paths.len() as f64 * max_entropy) as f32);
    if mtimes.len() > 1 {
        let mn = mtimes.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = mtimes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        w.set("struct:mtime_range_hours", ((mx - mn) / 3600.0) as f32);
        let mean = mtimes.iter().sum::<f64>() / mtimes.len() as f64;
        let var = mtimes.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / mtimes.len() as f64;
        w.set("struct:mtime_stddev_hours", (var.sqrt() / 3600.0) as f32);
    }
    w.set("struct:max_nesting_depth_log", (max_nesting_depth as f32).ln_1p());
    w.set("struct:inner_file_ratio", inner_file_count as f32 / summaries.len().max(1) as f32);
    if entropies.len() > 1 {
        let mean = entropies.iter().sum::<f64>() / entropies.len() as f64;
        let var = entropies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / entropies.len() as f64;
        w.set("struct:entropy_std_dev", var.sqrt() as f32);
        let mx = entropies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        w.set("struct:entropy_max_diff", (mx - mean) as f32);
    }
    w.set("struct:air_gap_signal", f32::from(hostile_files > 0 && hostile_files_with_parent == 0));
    if !mtimes.is_empty() && !hostile_mtimes.is_empty() {
        let mut sorted = mtimes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 1 { sorted[sorted.len() / 2] }
                     else { (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0 };
        let max_delta = hostile_mtimes.iter().map(|t| (t - median).abs()).fold(0.0, f64::max);
        w.set("struct:anachronistic_injection", (max_delta / 3600.0) as f32);
    }
    if !code_entropies.is_empty() {
        let avg_ent = if entropies.is_empty() { 0.0 } else { entropies.iter().sum::<f64>() / entropies.len() as f64 };
        let max_code_ent = code_entropies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        w.set("struct:code_entropy_spike", (max_code_ent - avg_ent) as f32);
    }
    w.set("struct:foreign_binary_signal", f32::from(has_foreign_binaries));
    w.set("struct:extension_mismatch_signal", extension_mismatches as f32);
    if total_loc > 0 { w.set("struct:hostile_finding_density", (hostile_files as f32 * 1000.0) / total_loc as f32); }
}

fn canonical_fields_from_primary_file(file: Option<&serde_json::Value>) -> (String, String, i64) {
    let Some(file) = file else {
        return (String::new(), String::new(), 0);
    };
    let formula = file
        .get("f")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    let elements: String = formula
        .chars()
        .filter(|c| !('\u{2080}'..='\u{2089}').contains(c))
        .collect();
    let score = file
        .get("x")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    (formula, elements, score)
}

fn primary_file(report: &serde_json::Value) -> Option<&serde_json::Value> {
    report
        .get("fs")
        .and_then(serde_json::Value::as_array)
        .and_then(|files| {
            files
                .iter()
                .find(|file| file.get("dp").and_then(serde_json::Value::as_i64).unwrap_or(0) == 0)
        })
}

fn write_logic_gap_features(summary: &FindingSummary, summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
    let logic_gaps: &[(&str, &[&str], &[&str])] = &[
        ("crypto", &["cryptography", "Crypto", "hashlib", "CryptAcquireContext", "BCryptOpenAlgorithmProvider"], &["micro-behaviors/crypto", "metadata/encoded-payload"]),
        ("network", &["socket", "urllib", "requests", "http", "curl", "wininet", "winhttp"], &["micro-behaviors/network", "objectives/command-and-control"]),
        ("process", &["subprocess", "os.spawn", "os.system", "CreateProcess", "ShellExecute", "posix_spawn"], &["micro-behaviors/process/create", "objectives/execution"]),
    ];

    let mut all_imports: HashSet<&str> = HashSet::new();
    for s in summaries { for imp in &s.imports { all_imports.insert(imp.as_str()); } }

    for target_cat in LOGIC_GAP_CATEGORIES {
        if let Some((_, imports_set, traits_set)) = logic_gaps.iter().find(|(c, _, _)| c == target_cat) {
            let has_import = imports_set.iter().any(|imp| all_imports.contains(*imp));
            let has_behavior = summary.sample_paths.iter().any(|(path, &max_ord)| max_ord >= 3 && traits_set.iter().any(|t| path.starts_with(t)));
            if has_import && !has_behavior {
                w.set(&format!("gap:{target_cat}"), 1.0);
            }
        }
    }
}

fn write_intent_gap_features(summary: &FindingSummary, w: &mut FeatureWriter<'_>) {
    let intent_signal = summary.sample_paths.contains_key("metadata/package/documentation") || summary.sample_paths.contains_key("metadata/package/help");
    let risky: &[(&str, &[&str])] = &[
        ("network", &["objectives/network", "micro-behaviors/network"]),
        ("filesystem", &["objectives/persistence", "micro-behaviors/filesystem"]),
        ("execution", &["objectives/execution", "micro-behaviors/process/create"]),
        ("crypto", &["objectives/crypto", "micro-behaviors/crypto"]),
    ];
    for target_cat in INTENT_GAP_CATEGORIES {
        let traits = risky.iter().find(|(c, _)| c == target_cat).map(|(_, t)| *t).unwrap_or(&[]);
        let has_behavior = summary.sample_paths.iter().any(|(path, &max_ord)| max_ord >= 4 && traits.iter().any(|t| path.starts_with(t)));
        if has_behavior && !intent_signal {
            w.set(&format!("intent_gap:{target_cat}"), 1.0);
        }
    }
}

fn write_negative_space_features(summary: &FindingSummary, summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
    let mut present_types: HashSet<&str> = HashSet::new();
    for s in summaries { if !s.file_type.is_empty() { present_types.insert(s.file_type.as_str()); } }
    for &(ftype, traits) in EXPECTED_GHOSTS {
        for trait_path in traits {
            if present_types.contains(ftype) && !summary.sample_paths.contains_key(*trait_path) {
                w.set(&format!("missing:{ftype}*{trait_path}"), 1.0);
            }
        }
    }
}

fn write_external_summary_features(summary: &FindingSummary, w: &mut FeatureWriter<'_>) {
    w.set("ext:third_party_max_crit", summary.third_party_max_crit as f32);
    w.set("ext:third_party_count", (summary.third_party_count as f32 + 1.0).ln());
    w.set("ext:well_known_max_crit", summary.well_known_max_crit as f32);
    w.set("ext:well_known_hostile_count", summary.well_known_hostile as f32);
    w.set("ext:well_known_suspicious_count", summary.well_known_suspicious as f32);
    w.set("ext:has_yara_match", f32::from(summary.has_yara));
}

fn merge_metric_summaries(summaries: &[FileSummary]) -> serde_json::Value {
    let mut merged: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for s in summaries {
        for (group, fields) in &s.metrics {
            let group_map = merged.entry(group.clone()).or_default();
            for (fname, &val) in fields {
                let e = group_map.entry(fname.clone()).or_insert(f64::NEG_INFINITY);
                *e = f64::max(*e, val);
            }
        }
    }
    serde_json::to_value(&merged).unwrap_or(serde_json::Value::Null)
}

fn write_metric_features(metrics: &serde_json::Value, w: &mut FeatureWriter<'_>, metric_vocab: &[String]) {
    let base_keys: HashSet<String> = KEY_METRICS.iter()
        .map(|&(g, f, _)| format!("{g}_{f}"))
        .collect();

    // Base KEY_METRICS with explicit log transforms.
    for &(group, field_name, use_log) in KEY_METRICS {
        let value = metrics.get(group).and_then(|g| g.get(field_name)).and_then(serde_json::Value::as_f64).unwrap_or(0.0) as f32;
        let val = if use_log { (value.abs() + 1.0).ln() } else { value };
        w.set(&format!("metrics:{group}_{field_name}"), val);
    }

    // Extended metrics from dynamic vocab — skip keys already in KEY_METRICS.
    for mk in metric_vocab {
        let parts: Vec<&str> = mk.splitn(2, '_').collect();
        if parts.len() == 2 && !base_keys.contains(mk.as_str()) {
            let value = metrics.get(parts[0])
                .and_then(|g| g.get(parts[1]))
                .and_then(|v| v.as_f64().or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 })))
                .unwrap_or(0.0) as f32;
            let use_log = ["count", "size", "total", "bytes", "length"]
                .iter()
                .any(|word| parts[1].contains(word));
            let val = if use_log { (value.abs() + 1.0).ln() } else { value };
            w.set(&format!("metrics:{mk}"), val);
        }
    }
}

fn write_structural_features(w: &mut FeatureWriter<'_>, summaries: &[FileSummary], filtered_finding_count: u32) {
    let binary_like = ["pe", "elf", "macho"];
    let mut any_tiny_binary = false;
    let mut import_candidates = 0;
    let mut importless_candidates = 0;
    let mut max_entropy = 0.0_f64;
    let mut suspicious_files = 0;
    let mut hostile_files = 0;

    for s in summaries {
        if binary_like.contains(&s.file_type.as_str()) && s.size_bytes < 20_000.0 { any_tiny_binary = true; }
        if s.has_imports_key {
            import_candidates += 1;
            if s.imports.is_empty() { importless_candidates += 1; }
        }
        max_entropy = max_entropy.max(s.overall_entropy);
        if s.findings.suspicious_finding_count > 0 { suspicious_files += 1; }
        if s.findings.hostile_finding_count > 0 { hostile_files += 1; }
    }

    w.set("struct:tiny_executable", f32::from(any_tiny_binary));
    w.set("struct:no_imports", f32::from(import_candidates > 0 && importless_candidates == import_candidates));
    w.set("struct:no_findings", f32::from(filtered_finding_count == 0));
    w.set("struct:finding_count_log", (filtered_finding_count as f32 + 1.0).ln());
    let file_count = summaries.len() as f32;
    w.set("struct:file_count_log", (file_count + 1.0).ln());
    w.set("struct:inner_file_count_log", ((file_count - 1.0).max(0.0) + 1.0).ln());
    w.set("struct:stealth_potential", f32::from(filtered_finding_count < 5 && max_entropy > 6.5));
    let denom = summaries.len().max(1) as f32;
    w.set("struct:suspicious_file_fraction", suspicious_files as f32 / denom);
    w.set("struct:hostile_file_fraction", hostile_files as f32 / denom);
    w.set("struct:suspicious_file_count_log", (suspicious_files as f32).ln_1p());
    w.set("struct:hostile_file_count_log", (hostile_files as f32).ln_1p());
}

fn report_files(report: &serde_json::Value) -> Vec<&serde_json::Value> {
    report["fs"].as_array().map(|a| a.iter().collect()).unwrap_or_default()
}

fn finding_paths(finding_id: &str) -> FindingPaths<'_> {
    let base = finding_id.split("::").next().unwrap_or(finding_id);
    let mut slash_ends = [0; 2];
    let mut n_slashes = 0;
    for (i, ch) in base.char_indices() {
        if ch == '/' {
            if n_slashes < 2 { slash_ends[n_slashes] = i; }
            n_slashes += 1;
        }
    }
    let third_end = if n_slashes >= 3 {
        let mut count = 0;
        let mut pos = base.len();
        for (i, ch) in base.char_indices() {
            if ch == '/' { count += 1; if count == 3 { pos = i; break; } }
        }
        pos
    } else { base.len() };

    FindingPaths { base, slash_ends, n_slashes: n_slashes.min(2), third_end, step: 0 }
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
    // slash_ends and third_end are byte offsets of ASCII '/' characters, which are
    // always single-byte UTF-8 codepoints — so these slices are valid UTF-8 boundaries.
    #[allow(clippy::string_slice)]
    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.step {
            0 => Some(if self.n_slashes >= 1 { &self.base[..self.slash_ends[0]] } else { self.base }),
            1 if self.n_slashes >= 1 => Some(if self.n_slashes >= 2 { &self.base[..self.slash_ends[1]] } else { self.base }),
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

    fn paths(id: &str) -> Vec<&str> { finding_paths(id).collect() }

    #[test]
    fn test_finding_paths_deep() {
        assert_eq!(paths("objectives/evasion/process/injection::technique-x"), vec!["objectives", "objectives/evasion", "objectives/evasion/process"]);
    }

    #[test]
    fn test_finding_paths_two_levels() {
        assert_eq!(paths("metadata/format::no-functions"), vec!["metadata", "metadata/format"]);
    }

    #[test]
    fn test_finding_paths_one_level() {
        assert_eq!(paths("standalone"), vec!["standalone"]);
    }

    #[test]
    fn test_crit_ordinal() {
        assert_eq!(crit_ordinal(&serde_json::json!({"l":5})), 5);
        assert_eq!(crit_ordinal(&serde_json::json!({"l":0})), 0);
        assert_eq!(crit_ordinal(&serde_json::json!({})), 0);
    }

    #[test]
    fn test_standardize() {
        let spec = FeatureSpec {
            version: 16, abi_version: 16, presence_vocab: vec![], filetype_vocab: vec![], element_vocab: vec![], bigram_vocab: vec![], ghost_vocab: vec![], skeleton_vocab: vec![], rare_element_vocab: vec![], trigram_vocab: vec![], metric_vocab: vec![],
            crit_unigram_vocab: vec![], crit_bigram_vocab: vec![], crit_trigram_vocab: vec![],
            attack_bigram_vocab: vec![], attack_trigram_vocab: vec![],
            mbc_bigram_vocab: vec![], mbc_trigram_vocab: vec![],
            feature_names: vec![], total_features: 3,
            feature_means: Some(vec![0.0, 1.0, 2.0]), feature_stds: Some(vec![1.0, 2.0, 0.5]), standardized: true,
        };
        let mut features = vec![1.0, 3.0, 3.0];
        spec.standardize(&mut features);
        assert_eq!(features[0], 0.0);
        assert!((features[1] - 1.0).abs() < 1e-6);
        assert!((features[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_rejects_missing_feature_names() -> Result<()> {
        let mut file = tempfile::NamedTempFile::new()?;
        writeln!(file, "{{\"version\":16,\"presence_vocab\":[\"objectives\"],\"filetype_vocab\":[\"sh\"],\"total_features\":51,\"standardized\":false}}")?;
        let Err(err) = FeatureSpec::load(file.path()) else { anyhow::bail!("missing feature_names should be rejected"); };
        assert!(err.to_string().contains("feature_names length"));
        Ok(())
    }
}
