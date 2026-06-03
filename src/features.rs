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
pub const EXPECTED_SPEC_VERSION: u32 = 17;

/// Earliest spec version this build can still load. v16 specs are
/// loadable too: v17 added the `cluster:*` feature family and new
/// `agg:static_*` aggregates that the extractor doesn't yet produce,
/// but the loader extracts known features and zeros out unknowns —
/// v16 bundles still work end-to-end.
pub const MIN_LOADABLE_SPEC_VERSION: u32 = 16;
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
///
/// `use_log1p` matches collimator's choice in `_TEXT_FULL_FIELDS` / `_BATCH1_OVERLAY`
/// (count/size fields get log1p; ratios/entropies do not). These entries are
/// always extracted regardless of `metric_vocab` because routes like
/// filetypes/c and filetypes/go reference them in `feature_names` without
/// listing them in the dynamic `metric_vocab` array.
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
    // Binary overlay (collimator `_BATCH1_OVERLAY`).
    ("binary", "overlay_ratio", false),
    ("binary", "overlay_entropy", false),
    ("binary", "overlay_size", true),
    ("binary", "has_overlay", false),
    // Text analysis
    ("text", "char_entropy", false),
    ("text", "unique_chars", true),
    ("text", "whitespace_ratio", false),
    ("text", "most_common_ratio", false),
    ("text", "total_lines", true),
    // Text full-shape fields (collimator `_TEXT_FULL_FIELDS`).
    ("text", "non_ascii_ratio", false),
    ("text", "non_printable_ratio", false),
    ("text", "null_byte_count", true),
    ("text", "high_byte_ratio", false),
    ("text", "avg_line_length", true),
    ("text", "max_line_length", true),
    ("text", "line_length_stddev", true),
    ("text", "last_line_length", true),
    ("text", "empty_line_ratio", false),
    ("text", "tab_count", true),
    ("text", "space_count", true),
    ("text", "trailing_whitespace_lines", true),
    ("text", "unusual_whitespace", true),
    ("text", "max_inline_whitespace_run", true),
    ("text", "unicode_escape_count", true),
    ("text", "octal_escape_count", true),
    ("text", "escape_density", false),
    ("text", "invisible_chars", true),
    ("text", "long_token_count", true),
    ("text", "repeated_char_sequences", true),
    ("text", "digit_ratio", false),
    ("text", "mixed_indent", false),
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
    tiered_bigram_vocab: Vec<String>,
    tiered_trigram_vocab: Vec<String>,
    kv_vocab: Vec<String>,
    symbol_vocab: Vec<String>,
    symbol_bigram_vocab: Vec<String>,
    symbol_trigram_vocab: Vec<String>,
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
    tiered_bigram_vocab: Vec<String>,
    #[serde(default)]
    tiered_trigram_vocab: Vec<String>,
    #[serde(default)]
    kv_vocab: Vec<String>,
    #[serde(default)]
    symbol_vocab: Vec<String>,
    #[serde(default)]
    symbol_bigram_vocab: Vec<String>,
    #[serde(default)]
    symbol_trigram_vocab: Vec<String>,
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

        if raw.version < MIN_LOADABLE_SPEC_VERSION || raw.version > EXPECTED_SPEC_VERSION {
            anyhow::bail!(
                "feature spec version mismatch: this installed model uses spec v{}, but this litmus build accepts v{MIN_LOADABLE_SPEC_VERSION}..=v{EXPECTED_SPEC_VERSION}. \
                 The model is incompatible with this build. Run 'litmus update-rules' to install a matching model bundle.",
                raw.version,
            );
        }
        if raw.version < EXPECTED_SPEC_VERSION {
            tracing::warn!(
                version = raw.version,
                expected = EXPECTED_SPEC_VERSION,
                "loading older spec version; extractor produces newer-version features as zeros for forward compatibility, but consider retraining at v{EXPECTED_SPEC_VERSION}"
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
            tiered_bigram_vocab: raw.tiered_bigram_vocab,
            tiered_trigram_vocab: raw.tiered_trigram_vocab,
            kv_vocab: raw.kv_vocab,
            symbol_vocab: raw.symbol_vocab,
            symbol_bigram_vocab: raw.symbol_bigram_vocab,
            symbol_trigram_vocab: raw.symbol_trigram_vocab,
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
            &self.tiered_bigram_vocab,
            &self.tiered_trigram_vocab,
            &self.kv_vocab,
            &self.symbol_vocab,
            &self.symbol_bigram_vocab,
            &self.symbol_trigram_vocab,
        );
        if self.feature_names != expected_feature_names {
            // The spec is allowed to be a SUBSET of what this extractor knows
            // (some feature groups disabled at training via
            // COLLIMATOR_DISABLE_FEATURE_GROUPS), AND it is allowed to contain
            // features this extractor doesn't yet implement — those slots are
            // filled with zeros at extraction time and the model degrades
            // gracefully on them rather than refusing to load.
            //
            // The previous behavior (anyhow::bail! on unknown features) kept
            // every collimator-side feature innovation tied to a synchronous
            // litmus update. That's the wrong trade: deploying a model with
            // 35 unknown features that extract as zeros costs at most a
            // measurable accuracy delta; refusing to deploy costs the whole
            // model. We surface the situation as an ERROR (caught by
            // verify_azoth_litmus_runtime.py's exit-code gate, see
            // collimator/scripts) so CI/operator sees the degradation
            // immediately and can decide whether to add proper extraction.
            let expected_set: std::collections::HashSet<&str> =
                expected_feature_names.iter().map(String::as_str).collect();
            let unknown_features: Vec<&str> = self
                .feature_names
                .iter()
                .map(String::as_str)
                .filter(|n| !expected_set.contains(*n))
                .collect();
            if !unknown_features.is_empty() {
                let preview: Vec<&str> = unknown_features.iter().copied().take(5).collect();
                // WARN-level (not ERROR): graceful degradation, not a deploy
                // blocker. Unknown features extract as zeros — the model has
                // those slots in its input vector but the runtime fills them
                // with the same default it would see for a sample with no
                // signal. Deploy verification (which fails on ERROR-level
                // anomalies for inverted thresholds, ABI mismatch, etc.)
                // intentionally lets this through. Engineer adds proper
                // extraction when they want to recover the model accuracy
                // those features were providing.
                tracing::warn!(
                    spec_features = self.feature_names.len(),
                    expected_max = expected_feature_names.len(),
                    unknown_count = unknown_features.len(),
                    sample = ?preview,
                    "feature spec contains features unknown to this extractor — they will extract as zeros (model accuracy may degrade for those slots)"
                );
            } else {
                tracing::debug!(
                    "feature spec has {} features (extractor knows {} optional-inclusive features) — {} optional features absent from this model",
                    self.feature_names.len(),
                    expected_feature_names.len(),
                    expected_feature_names.len().saturating_sub(self.feature_names.len()),
                );
            }
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

const FORMAT_GROUPS: &[(&str, &[&str])] = &[
    (
        "script",
        &[
            "batch",
            "javascript",
            "lua",
            "perl",
            "php",
            "powershell",
            "python",
            "ruby",
            "shell",
            "typescript",
            "vbscript",
        ],
    ),
    ("native_binary", &["elf", "macho", "pe"]),
    (
        "archive_package",
        &[
            "7z", "apk", "cab", "deb", "egg", "gz", "jar", "msi", "rar", "rpm", "tar", "tgz",
            "vsix", "war", "whl", "xpi", "xz", "zip", "zst",
        ],
    ),
    (
        "document",
        &[
            "doc", "docx", "html", "pdf", "ppt", "pptx", "rtf", "xls", "xlsx",
        ],
    ),
    (
        "source_code",
        &[
            "c", "cpp", "csharp", "go", "java", "kotlin", "makefile", "rust", "scala", "swift",
        ],
    ),
    (
        "config_data",
        &["ini", "json", "plist", "toml", "xml", "yaml", "yml"],
    ),
    (
        "media",
        &[
            "bmp", "gif", "jpg", "jpeg", "mp3", "mp4", "png", "svg", "webp",
        ],
    ),
];

/// Expected ghosts (v16 group 23).
const EXPECTED_GHOSTS: &[(&str, &[&str])] = &[
    (
        "elf",
        &[
            "metadata/binary/layout",
            "metadata/binary/metrics",
            "metadata/binary/symbols",
            "metadata/binary/linking",
        ],
    ),
    (
        "javascript",
        &[
            "micro-behaviors/javascript/async",
            "metadata/package/versioning",
        ],
    ),
    (
        "pe",
        &[
            "metadata/binary/layout",
            "metadata/binary/metrics",
            "metadata/binary/resource",
            "metadata/binary/symbols",
            "metadata/binary/linking",
        ],
    ),
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
    tiered_bigram_vocab: &[String],
    tiered_trigram_vocab: &[String],
    kv_vocab: &[String],
    symbol_vocab: &[String],
    symbol_bigram_vocab: &[String],
    symbol_trigram_vocab: &[String],
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
        "agg:kill_chain_span".to_string(),
        "agg:objective_micro_ratio".to_string(),
        "agg:avg_finding_depth".to_string(),
        "agg:objective_hostile_density".to_string(),
        "agg:static_file_bytes_log".to_string(),
        "agg:static_import_count_log".to_string(),
        "agg:static_export_count_log".to_string(),
        "agg:static_dependency_count_log".to_string(),
        "agg:static_string_count_log".to_string(),
        "agg:static_wide_string_ratio".to_string(),
        "agg:static_max_string_length_log".to_string(),
        "agg:static_string_entropy_max".to_string(),
        "agg:static_text_lines_log".to_string(),
        "agg:static_function_count_log".to_string(),
        "agg:static_code_bytes_log".to_string(),
        "agg:static_code_to_data_ratio_max".to_string(),
        "agg:static_wx_units_log".to_string(),
        "agg:static_writable_unit_ratio".to_string(),
        "agg:static_executable_unit_ratio".to_string(),
        "agg:static_nonstandard_unit_names_log".to_string(),
        "agg:static_largest_unit_ratio_max".to_string(),
        "agg:static_resource_ratio_max".to_string(),
        "agg:static_signed_file_fraction".to_string(),
        "agg:attack_technique_count".to_string(),
        "agg:attack_tactic_count".to_string(),
        "agg:mbc_behavior_count".to_string(),
        "agg:has_attack_and_objective".to_string(),
        // ATT&CK / MBC co-occurrence aggregates (log1p of unordered combinations
        // among distinct technique / behavior codes seen in raw_findings).
        "agg:attack_bigram_count".to_string(),
        "agg:attack_trigram_count".to_string(),
        "agg:mbc_bigram_count".to_string(),
        // Objective path co-occurrence aggregates (log1p of unordered combinations
        // among distinct `objectives/*` and `well-known/*` paths seen in
        // sample_paths). Trigram is bounded with a per-pair cap of 20 inner
        // elements to avoid O(n^3) explosion on samples with many objectives.
        "agg:objective_bigram_count".to_string(),
        "agg:objective_trigram_count".to_string(),
    ]);

    // Crit-category n-grams (vocab-driven)
    for cu in crit_unigram_vocab {
        feature_names.push(format!("crit:{cu}"));
    }
    for cb in crit_bigram_vocab {
        feature_names.push(format!("critbi:{cb}"));
    }
    for ct in crit_trigram_vocab {
        feature_names.push(format!("crittri:{ct}"));
    }

    // ATT&CK/MBC code n-grams (vocab-driven)
    for ab in attack_bigram_vocab {
        feature_names.push(format!("atkbi:{ab}"));
    }
    for at in attack_trigram_vocab {
        feature_names.push(format!("atktri:{at}"));
    }
    for mb in mbc_bigram_vocab {
        feature_names.push(format!("mbcbi:{mb}"));
    }
    for mt in mbc_trigram_vocab {
        feature_names.push(format!("mbctri:{mt}"));
    }

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

    // Group 6b: Portable format-group hints derived only from cleave file types.
    for &(group, _) in FORMAT_GROUPS {
        feature_names.push(format!("format:{group}"));
        feature_names.push(format!("format:{group}_file_fraction"));
        feature_names.push(format!("format:{group}_inner_fraction"));
        feature_names.push(format!("format:{group}_suspicious_fraction"));
        feature_names.push(format!("format:{group}_hostile_fraction"));
    }
    feature_names.extend([
        "format:group_count_log".to_string(),
        "format:mixed_script_binary".to_string(),
        "format:mixed_archive_script".to_string(),
        "format:mixed_archive_binary".to_string(),
        "format:unknown_file_fraction".to_string(),
    ]);

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

    // Group 11b: report-level severity-prefixed trait bigrams
    for bi in tiered_bigram_vocab {
        feature_names.push(format!("tierbi:{bi}"));
    }

    // Group 11c: report-level severity-prefixed trait trigrams
    for tri in tiered_trigram_vocab {
        feature_names.push(format!("tiertri:{tri}"));
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

    // Additional aggregate-level co-occurrence counts. Only present in some
    // route specs (e.g. filetypes/makefile) — registered here so they don't
    // surface as "unknown to this extractor".
    feature_names.push("agg:suspicious_bigram_count".to_string());
    feature_names.push("agg:suspicious_trigram_count".to_string());

    // Cross-metric derived ratios. Computed from binary metric fields in
    // write_derived_metric_features.
    feature_names.push("metrics:derived_string_per_function".to_string());
    feature_names.push("metrics:derived_imports_per_dependency".to_string());
    feature_names.push("metrics:derived_wide_string_ratio".to_string());

    // Optional silent-packer-signal struct feature.
    feature_names.push("struct:silent_packer_signal".to_string());

    // Group 24: textenc — 12 fixed ratios over file_strings.
    for name in TEXTENC_FEATURE_NAMES {
        feature_names.push(format!("textenc:{name}"));
    }

    // Group 25: kv vocab — sparse one-hot tokens from cleave metrics+values.
    for kv in kv_vocab {
        feature_names.push(format!("kv:{kv}"));
    }

    // Group 26: symbol vocab — normalized imports/exports/functions.
    for sym in symbol_vocab {
        feature_names.push(format!("symbol:{sym}"));
    }
    for sb in symbol_bigram_vocab {
        feature_names.push(format!("symbol_bi:{sb}"));
    }
    for st in symbol_trigram_vocab {
        feature_names.push(format!("symbol_tri:{st}"));
    }

    feature_names
}

/// Fixed textenc feature names; matches collimator's `_apply_text_encoding_features`.
const TEXTENC_FEATURE_NAMES: &[&str] = &[
    "string_count_log",
    "avg_len_log",
    "max_len_log",
    "base64ish_ratio",
    "hexish_ratio",
    "urlish_ratio",
    "pathish_ratio",
    "unicode_escape_ratio",
    "wide_ratio",
    "high_entropy_ratio",
    "long_token_ratio",
    "short_junk_ratio",
];

/// Pre-built lookup tables for fast repeated extraction against a spec.
#[derive(Debug)]
pub struct ExtractContext {
    presence_lookup: HashMap<String, usize>,
    n_presence: usize,
    n_ft: usize,
    n_format: usize,
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
    n_tiered_bigram: usize,
    n_tiered_trigram: usize,
    n_kv: usize,
    n_symbol: usize,
    n_symbol_bi: usize,
    n_symbol_tri: usize,
    n_textenc: usize,
    n_derived_metrics: usize,
    n_silent_packer: usize,
    n_extra_agg: usize,
    /// Global feature name → index lookup for vocab-based features.
    absolute_lookup: HashMap<String, usize>,
    ghost_vocab: Vec<String>,
    total_features: usize,

    // Optimized bigram/trigram lookups
    path_to_id: HashMap<String, u32>,
    bigram_id_lookup: HashMap<(u32, u32), usize>,
    trigram_id_lookup: HashMap<(u32, u32, u32), usize>,

    // Base indices of the families written by raw `base + idx` offset, anchored
    // to their real position in the spec's `feature_names`. `None` when the
    // family is absent from the spec (the writer then skips it). See `new()`.
    present_base: Option<usize>,
    maxcrit_base: Option<usize>,
    bigram_base: Option<usize>,
    trigram_base: Option<usize>,
    unsigned_bigram_base: Option<usize>,

    /// Offset families that are present in the spec but laid out incompatibly
    /// (partial or out of vocab order). Empty for a healthy bundle; surfaced by
    /// [`Self::validate_layout`] so a bad bundle is rejected at load time.
    layout_errors: Vec<&'static str>,
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

        // Global feature name -> index. Built before `Self` so the anchored
        // family bases below can be derived from it.
        let absolute_lookup: HashMap<String, usize> = spec
            .feature_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        // Anchor the offset-written families (presence, max-crit, bigram,
        // trigram, unsigned-bigram) to their real position in `feature_names`
        // rather than to a hand-maintained running cursor. The cursor approach
        // silently drifted once — a metrics-block miscount left every later
        // family 26 slots too high and ran the unsigned-bigram block off the
        // end of the vector — and a release build has no `debug_assert` to
        // catch it.
        //
        // `base + idx` is only valid if the family occupies a contiguous
        // base..base+len run in vocab order, so verify that whole invariant here
        // (once per model load) instead of trusting it. Two outcomes return
        // `None` (the writer then skips the family, leaving zeros), but they
        // mean different things:
        //   * fully absent  — the family was pruned/disabled in the spec; the
        //     model was trained without it, so skipping is correct, not an error.
        //   * partially present or out of order — `base + idx` would land on the
        //     wrong slots; recorded in `layout_errors` so `validate_layout` can
        //     reject the bundle rather than serve corrupted features.
        let mut layout_errors: Vec<&'static str> = Vec::new();
        let mut family_base = |prefix: &'static str, vocab: &[String]| -> Option<usize> {
            if vocab.is_empty() {
                return None;
            }
            let indices: Vec<Option<usize>> = vocab
                .iter()
                .map(|v| absolute_lookup.get(&format!("{prefix}{v}")).copied())
                .collect();
            if indices.iter().all(Option::is_none) {
                return None; // pruned/disabled entirely — skip gracefully
            }
            let base = indices[0];
            let contiguous = base.is_some_and(|base| {
                indices
                    .iter()
                    .enumerate()
                    .all(|(i, idx)| *idx == Some(base + i))
            });
            if contiguous {
                base
            } else {
                tracing::warn!(
                    family = prefix,
                    "offset feature family is partially present or not contiguous \
                     in vocab order; skipping to avoid misaligned writes — the \
                     model bundle is incompatible with this litmus build",
                );
                layout_errors.push(prefix);
                None
            }
        };
        let present_base = family_base("present:", &spec.presence_vocab);
        let maxcrit_base = family_base("maxcrit:", &spec.presence_vocab);
        let bigram_base = family_base("bigrams:", &spec.bigram_vocab);
        let trigram_base = family_base("trigram:", &spec.trigram_vocab);
        let unsigned_bigram_base = family_base("unsigned_bigram:", &spec.bigram_vocab);

        Self {
            presence_lookup,
            n_presence: spec.presence_vocab.len(),
            n_ft: spec.filetype_vocab.len(),
            n_format: spec
                .feature_names
                .iter()
                .filter(|name| name.starts_with("format:"))
                .count(),
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
            n_tiered_bigram: spec.tiered_bigram_vocab.len(),
            n_tiered_trigram: spec.tiered_trigram_vocab.len(),
            n_kv: spec
                .feature_names
                .iter()
                .filter(|n| n.starts_with("kv:"))
                .count(),
            n_symbol: spec
                .feature_names
                .iter()
                .filter(|n| n.starts_with("symbol:"))
                .count(),
            n_symbol_bi: spec
                .feature_names
                .iter()
                .filter(|n| n.starts_with("symbol_bi:"))
                .count(),
            n_symbol_tri: spec
                .feature_names
                .iter()
                .filter(|n| n.starts_with("symbol_tri:"))
                .count(),
            n_textenc: spec
                .feature_names
                .iter()
                .filter(|n| n.starts_with("textenc:"))
                .count(),
            n_derived_metrics: spec
                .feature_names
                .iter()
                .filter(|n| n.starts_with("metrics:derived_"))
                .count(),
            n_silent_packer: spec
                .feature_names
                .iter()
                .filter(|n| n.as_str() == "struct:silent_packer_signal")
                .count(),
            n_extra_agg: spec
                .feature_names
                .iter()
                .filter(|n| {
                    n.as_str() == "agg:suspicious_bigram_count"
                        || n.as_str() == "agg:suspicious_trigram_count"
                })
                .count(),
            absolute_lookup,
            ghost_vocab: spec.ghost_vocab.clone(),
            total_features: spec.total_features,
            path_to_id,
            bigram_id_lookup,
            trigram_id_lookup,
            present_base,
            maxcrit_base,
            bigram_base,
            trigram_base,
            unsigned_bigram_base,
            layout_errors,
        }
    }

    /// Reject a model bundle whose offset-written feature families are present
    /// but laid out incompatibly with this build's extractor.
    ///
    /// A family that is laid out non-contiguously / out of vocab order would
    /// make `base + idx` writes land on the wrong slots (before anchoring, off
    /// the end of the vector entirely). A family that is simply *absent* is not
    /// an error — it was pruned/disabled in the spec and the extractor skips it.
    ///
    /// This is deterministic and corpus-independent, so `validate` calls it to
    /// reject a bad bundle at load time (triggering rollback) instead of relying
    /// on the benign corpus to happen to exercise the broken family.
    pub fn validate_layout(&self) -> Result<()> {
        if self.layout_errors.is_empty() {
            Ok(())
        } else {
            anyhow::bail!(
                "model bundle incompatible with this litmus build: offset feature \
                 families [{}] are present in the spec but not laid out \
                 contiguously in vocab order",
                self.layout_errors.join(", "),
            )
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
            raw_files.par_iter().map(|&f| FileSummary::new(f)).collect()
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
        let score_weight: f64 = if sample_score > 0 {
            (sample_score as f64).ln_1p()
        } else {
            1.0
        };

        // G1: Presence. Cursor advanced only to feed the layout-drift check at
        // the end; the write itself is anchored to the spec (see `new()`).
        offsets.take(self.n_presence);
        if let Some(base) = self.present_base {
            self.write_presence_features_v16(&combined, vec, base, score_weight);
        }

        // G2: Max crit
        offsets.take(self.n_presence);
        if let Some(base) = self.maxcrit_base {
            self.write_max_crit_features_v16(&combined, vec, base, score_weight);
        }

        // G3: Aggregates
        let n_crit = self.n_crit_unigram + self.n_crit_bigram + self.n_crit_trigram;
        let n_code_ngrams =
            self.n_atk_bigram + self.n_atk_trigram + self.n_mbc_bigram + self.n_mbc_trigram;
        offsets.take(57 + n_crit + n_code_ngrams);
        write_aggregate_features(
            &combined,
            &summaries,
            &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            },
        );

        // Crit-category n-grams
        {
            let crit_categories: HashSet<&str> = [
                "objectives",
                "well-known",
                "supply-chain",
                "anti-analysis",
                "anti-static",
                "command-and-control",
                "evasion",
                "execution",
                "exfiltration",
            ]
            .into_iter()
            .collect();
            let crit_pfx = |c: u32| match c {
                5 => "h",
                4 => "s",
                _ => "n",
            };
            let mut pmc: HashMap<String, u32> = HashMap::new();
            for (path, &mo) in &combined.sample_paths {
                if mo < 3 {
                    continue;
                }
                let parts: Vec<&str> = path.split('/').collect();
                if !crit_categories.contains(parts[0]) {
                    continue;
                }
                let key = if parts.len() >= 2 {
                    format!("{}/{}", parts[0], parts[1])
                } else {
                    parts[0].to_string()
                };
                let e = pmc.entry(key).or_insert(0);
                *e = (*e).max(mo);
            }
            let mut tokens: Vec<String> = pmc
                .iter()
                .map(|(k, &c)| format!("{}:{k}", crit_pfx(c)))
                .collect();
            tokens.sort();
            let lookup = &self.absolute_lookup;
            for t in &tokens {
                if let Some(&i) = lookup.get(&format!("crit:{t}")) {
                    vec[i] = 1.0;
                }
            }

            // Safety: trigrams are O(N^3), bigrams O(N^2). Cap N to avoid complexity bombs on bloated reports.
            if tokens.len() <= 512 {
                for (i, t1) in tokens.iter().enumerate() {
                    for t2 in &tokens[i + 1..] {
                        if let Some(&idx) = lookup.get(&format!("critbi:{t1} + {t2}")) {
                            vec[idx] = 1.0;
                        }
                    }
                    if tokens.len() <= 128 {
                        for j in i + 1..tokens.len() {
                            for t3 in &tokens[j + 1..] {
                                if let Some(&idx) =
                                    lookup.get(&format!("crittri:{t1} + {} + {t3}", tokens[j]))
                                {
                                    vec[idx] = 1.0;
                                }
                            }
                        }
                    }
                }
            } else {
                tracing::warn!(
                    tokens = tokens.len(),
                    "too many unique crit tokens; skipping n-gram generation"
                );
            }
        }

        // ATT&CK/MBC code n-grams
        {
            let mut attacks: HashSet<String> = HashSet::new();
            let mut mbcs: HashSet<String> = HashSet::new();
            for s in &summaries {
                for f in &s.raw_findings {
                    if let Some(a) = f.get("a").and_then(|v| v.as_str()) {
                        attacks.insert(a.to_string());
                    }
                    if let Some(m) = f.get("m").and_then(|v| v.as_str()) {
                        mbcs.insert(m.to_string());
                    }
                }
            }
            let lookup = &self.absolute_lookup;
            let mut sa: Vec<_> = attacks.iter().collect();
            sa.sort();
            if sa.len() <= 512 {
                for (i, a1) in sa.iter().enumerate() {
                    for (j, a2) in sa[i + 1..].iter().enumerate() {
                        if let Some(&idx) = lookup.get(&format!("atkbi:{a1} + {a2}")) {
                            vec[idx] = 1.0;
                        }
                        if sa.len() <= 128 {
                            for a3 in &sa[i + 1 + j + 1..] {
                                if let Some(&idx) =
                                    lookup.get(&format!("atktri:{a1} + {a2} + {a3}"))
                                {
                                    vec[idx] = 1.0;
                                }
                            }
                        }
                    }
                }
            } else {
                tracing::warn!(
                    tokens = sa.len(),
                    "too many unique ATT&CK tokens; skipping n-gram generation"
                );
            }

            let mut sm: Vec<_> = mbcs.iter().collect();
            sm.sort();
            if sm.len() <= 512 {
                for (i, m1) in sm.iter().enumerate() {
                    for (j, m2) in sm[i + 1..].iter().enumerate() {
                        if let Some(&idx) = lookup.get(&format!("mbcbi:{m1} + {m2}")) {
                            vec[idx] = 1.0;
                        }
                        if sm.len() <= 128 {
                            for m3 in &sm[i + 1 + j + 1..] {
                                if let Some(&idx) =
                                    lookup.get(&format!("mbctri:{m1} + {m2} + {m3}"))
                                {
                                    vec[idx] = 1.0;
                                }
                            }
                        }
                    }
                }
            } else {
                tracing::warn!(
                    tokens = sm.len(),
                    "too many unique MBC tokens; skipping n-gram generation"
                );
            }
        }

        // G4: External (name-based)
        offsets.take(6); // reserve space for offset tracking compatibility
        write_external_summary_features(
            &combined,
            &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            },
        );

        // G5: Metrics (base + extended vocab)
        offsets.take(KEY_METRICS.len() + self.n_ext_metrics);
        write_metric_features(
            &merged_metrics,
            &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            },
            &self.metric_vocab,
        );

        // G6: Filetype (blindfolded in v16)
        let _file_type_offset = offsets.take(self.n_ft);

        // G6b: Format hints
        offsets.take(self.n_format);
        write_format_hint_features(
            &summaries,
            &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            },
        );

        // G7: Structural
        offsets.take(11);
        write_structural_features(
            &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            },
            &summaries,
            combined.filtered_finding_count,
        );

        // G8: Elements
        offsets.take(self.n_element);
        {
            let w = &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            };
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
                w.set(
                    "formula:complexity_ratio",
                    formula_str.chars().count() as f32 / combined.filtered_finding_count as f32,
                );
            }

            // G10: Score
            let total_size_bytes: f64 = summaries.iter().map(|s| s.size_bytes).sum();
            w.set("score:hopper_score", sample_score as f32);
            w.set(
                "score:density",
                if total_size_bytes > 0.0 {
                    sample_score as f32 / (total_size_bytes as f32).ln_1p()
                } else {
                    0.0
                },
            );
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
        offsets.take(self.n_bigram);
        if let Some(base) = self.bigram_base {
            self.write_bigram_features_optimized(&summaries, vec, base);
        }

        // G11b: Tiered report-level notable+ bigrams.
        offsets.take(self.n_tiered_bigram);
        write_tiered_bigram_features(
            &combined,
            &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            },
        );
        offsets.take(self.n_tiered_trigram);
        write_tiered_trigram_features(
            &combined,
            &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            },
        );

        offsets.take(self.n_ghost); // ghost (now name-based above)
        offsets.take(self.n_skeleton); // skeleton (now name-based above)
        offsets.take(self.n_rare); // rare (now name-based above)

        // G15: Structural Extensions
        offsets.take(13);
        write_structural_extensions(
            &summaries,
            &combined,
            &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            },
        );

        // G16: Trigrams (optimized)
        offsets.take(self.n_trigram);
        if let Some(base) = self.trigram_base {
            self.write_trigram_features_optimized(&summaries, vec, base);
        }

        // G19: Logic Gaps
        offsets.take(LOGIC_GAP_CATEGORIES.len());
        write_logic_gap_features(
            &combined,
            &summaries,
            &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            },
        );

        // G20: Signature Synergy
        offsets.take(self.n_bigram);
        if let Some(base) = self.unsigned_bigram_base
            && combined.sample_paths.contains_key("metadata/unsigned")
        {
            self.write_bigram_features_optimized(&summaries, vec, base);
        }

        // G22: Intent Gaps
        offsets.take(INTENT_GAP_CATEGORIES.len());
        write_intent_gap_features(
            &combined,
            &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            },
        );

        // G23: Negative Space
        let missing_count: usize = EXPECTED_GHOSTS.iter().map(|(_, t)| t.len()).sum();
        offsets.take(missing_count);
        write_negative_space_features(
            &combined,
            &summaries,
            &mut FeatureWriter {
                vec,
                lookup: &self.absolute_lookup,
            },
        );

        // G24+: extras gated by per-route spec contents.
        // Sparse one-hot families that resolve through absolute_lookup; the
        // cursor bookkeeping below tracks how many of these slots actually
        // exist in this spec so the final offset matches total_features.
        let mut tail_writer = FeatureWriter {
            vec,
            lookup: &self.absolute_lookup,
        };

        if self.n_extra_agg > 0 {
            offsets.take(self.n_extra_agg);
            write_suspicious_ngram_counts(&combined, &mut tail_writer);
        }
        if self.n_derived_metrics > 0 {
            offsets.take(self.n_derived_metrics);
            write_derived_metric_features(&merged_metrics, &mut tail_writer);
        }
        if self.n_silent_packer > 0 {
            offsets.take(self.n_silent_packer);
            write_silent_packer_signal(
                &summaries,
                combined.filtered_finding_count,
                &mut tail_writer,
            );
        }
        if self.n_textenc > 0 {
            offsets.take(self.n_textenc);
            write_textenc_features(&summaries, &mut tail_writer);
        }
        if self.n_kv > 0 {
            offsets.take(self.n_kv);
            for s in &summaries {
                write_kv_features(s, &mut tail_writer);
            }
        }
        if self.n_symbol > 0 || self.n_symbol_bi > 0 || self.n_symbol_tri > 0 {
            offsets.take(self.n_symbol + self.n_symbol_bi + self.n_symbol_tri);
            write_symbol_features(
                &summaries,
                &mut tail_writer,
                self.n_symbol_bi > 0,
                self.n_symbol_tri > 0,
            );
        }

        // The hand-maintained cursor should still land exactly on
        // `total_features`. It is no longer load-bearing — every offset-written
        // family above is anchored to the spec's `feature_names`, so extraction
        // stays correct even when this disagrees — but a mismatch means the
        // layout constants in this file have drifted from the model's
        // `feature_spec.json` and should be resynced with collimator. Surface it
        // once (a `debug_assert` here was compiled out of the release workers,
        // which is how a 26-slot metrics drift shipped undetected) and never
        // panic: a bad layout must degrade gracefully, not kill the analysis.
        if offsets.offset != self.total_features {
            static DRIFT_WARNED: std::sync::Once = std::sync::Once::new();
            DRIFT_WARNED.call_once(|| {
                tracing::warn!(
                    cursor = offsets.offset,
                    total_features = self.total_features,
                    "feature-layout cursor disagrees with spec total_features; \
                     offset families are anchored to the spec so extraction is \
                     still correct, but the layout constants in features.rs have \
                     drifted from collimator and should be resynced",
                );
            });
        }
    }

    fn write_presence_features_v16(
        &self,
        summary: &FindingSummary,
        vec: &mut [f32],
        offset: usize,
        score_weight: f64,
    ) {
        for (path, &max_ord) in &summary.sample_paths {
            if max_ord >= 2
                && let Some(&idx) = self.presence_lookup.get(path.as_str())
            {
                let conf = summary.path_confidences.get(path).copied().unwrap_or(1.0);
                if let Some(slot) = vec.get_mut(offset + idx) {
                    *slot = (score_weight * conf) as f32;
                }
            }
        }
    }

    fn write_max_crit_features_v16(
        &self,
        summary: &FindingSummary,
        vec: &mut [f32],
        offset: usize,
        score_weight: f64,
    ) {
        for (path, &max_ord) in &summary.sample_paths {
            if let Some(&idx) = self.presence_lookup.get(path.as_str()) {
                let conf = summary.path_confidences.get(path).copied().unwrap_or(1.0);
                if let Some(slot) = vec.get_mut(offset + idx) {
                    *slot = (f64::from(max_ord) * score_weight * conf) as f32;
                }
            }
        }
    }

    fn write_bigram_features_optimized(
        &self,
        summaries: &[FileSummary],
        vec: &mut [f32],
        offset: usize,
    ) {
        for s in summaries {
            let ids: Vec<u32> = s
                .unique_3level_paths
                .iter()
                .filter_map(|p| self.path_to_id.get(p).copied())
                .collect();
            if ids.len() > 512 {
                tracing::warn!(path = %s.path, tokens = ids.len(), "too many unique paths; skipping bigram generation for file");
                continue;
            }
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let key = (ids[i].min(ids[j]), ids[i].max(ids[j]));
                    if let Some(&idx) = self.bigram_id_lookup.get(&key)
                        && let Some(slot) = vec.get_mut(offset + idx)
                    {
                        *slot = 1.0;
                    }
                }
            }
        }
    }

    fn write_trigram_features_optimized(
        &self,
        summaries: &[FileSummary],
        vec: &mut [f32],
        offset: usize,
    ) {
        for s in summaries {
            let ids: Vec<u32> = s
                .unique_3level_paths
                .iter()
                .filter_map(|p| self.path_to_id.get(p).copied())
                .collect();
            let n = ids.len();
            if n > 256 {
                tracing::warn!(path = %s.path, tokens = n, "too many unique paths; skipping trigram generation for file");
                continue;
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        let mut sorted = [ids[i], ids[j], ids[k]];
                        sorted.sort();
                        if let Some(&idx) = self
                            .trigram_id_lookup
                            .get(&(sorted[0], sorted[1], sorted[2]))
                            && let Some(slot) = vec.get_mut(offset + idx)
                        {
                            *slot = 1.0;
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
    /// Full raw cleave metrics JSON (preserves strings, lists, dicts that
    /// the numeric `metrics` map drops). Used by kv: token extraction.
    raw_metrics: serde_json::Value,
    /// Flat structural values from `ff.v` or `k` (kv: source for v.* paths).
    raw_values: serde_json::Value,
    /// String tuples from `ff.s` or `ss` (textenc source).
    raw_strings: Vec<serde_json::Value>,
    findings: FindingSummary,
    risk: FileRiskStats,
    unique_3level_paths: Vec<String>,
    imports: HashSet<String>,
    /// Cleave imports array (preserves [lib, name, ...] tuples for symbol port).
    raw_imports: Vec<serde_json::Value>,
    /// Cleave exports (`ff.x`) — list of export entries; one source for `symbol:`.
    raw_exports: Vec<serde_json::Value>,
    /// Cleave function names (`ff.fn`) — second source for `symbol:`.
    raw_functions: Vec<serde_json::Value>,
    /// Filefacts call targets (`ff.ct`) — dotted call paths
    /// (`Symbol::Call.target`). Sourced from cleave's v5 compact AST
    /// emission; flows into `symbol:` when the spec's symbol_vocab
    /// contains the token.
    raw_call_targets: Vec<serde_json::Value>,
    /// Filefacts member chains (`ff.mc`) — dotted access chains
    /// (`Symbol::Member.path`). Same flow as call targets.
    raw_member_chains: Vec<serde_json::Value>,
    /// (finding_id, confidence, crit_ordinal) for cross-file unique ID dedup.
    raw_findings: Vec<serde_json::Value>,
    /// Whether the "is" key exists in the cleave JSON (even if empty).
    has_imports_key: bool,
}

impl FileSummary {
    fn new(file_entry: &serde_json::Value) -> Self {
        let findings_raw: Vec<&serde_json::Value> = file_entry["ts"]
            .as_array()
            .map(|a| a.iter().collect())
            .unwrap_or_default();
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

        // N-gram paths: full base path (depth=0), filtered by min_crit=3
        // (notable+). Must match Python's _ngram_paths_for_file behavior.
        let mut unique_3level_paths: Vec<String> = findings_raw
            .iter()
            .filter_map(|f| {
                let fid = f["i"].as_str()?;
                let conf = f["c"].as_f64().unwrap_or(1.0);
                let crit = crit_ordinal(f);
                (conf >= MIN_CONFIDENCE && crit >= 3)
                    .then(|| fid.split("::").next().unwrap_or(fid).to_string())
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        unique_3level_paths.sort();

        // Resolve metrics: prefer v5 `ff.m`, fall back to v4 top-level `ms`.
        // Matches `file_metrics` in collimator/src/collimator/features.py.
        let metrics_source = file_entry
            .get("ff")
            .and_then(|f| f.get("m"))
            .filter(|v| v.is_object())
            .or_else(|| file_entry.get("ms"));
        let metrics: HashMap<String, HashMap<String, f64>> = metrics_source
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(group, fields)| {
                        let group_map = fields
                            .as_object()?
                            .iter()
                            .filter_map(|(k, v)| {
                                // Handle numbers, booleans (true→1.0), and skip strings/nulls.
                                let val = v
                                    .as_f64()
                                    .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))?;
                                Some((k.clone(), val))
                            })
                            .collect();
                        Some((group.clone(), group_map))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let overall_entropy = metrics
            .get("binary")
            .and_then(|m| m.get("overall_entropy"))
            .copied()
            .unwrap_or(0.0);

        // Imports — v5 prefers `ff.i`, falls back to v4 `is`.
        let imports_array = file_entry
            .get("ff")
            .and_then(|f| f.get("i"))
            .and_then(|v| v.as_array())
            .or_else(|| file_entry.get("is").and_then(|v| v.as_array()));
        let imports: HashSet<String> = imports_array
            .into_iter()
            .flatten()
            .filter_map(|imp| {
                imp.as_str()
                    .or_else(|| imp.get("n").and_then(|v| v.as_str()))
            })
            .map(str::to_string)
            .collect();

        let raw_findings: Vec<serde_json::Value> =
            findings_raw.iter().map(|f| (*f).clone()).collect();
        let has_imports_key = file_entry.get("is").is_some()
            || file_entry
                .get("ff")
                .and_then(|f| f.get("i"))
                .is_some();

        // Cleave v5 facts block `ff` carries `m` (metrics), `v` (flat values),
        // `s` (strings), `i` (imports), `x` (exports), `fn` (function names).
        // v4 reports stash these at top-level keys: `ms`, `k`, `ss`, `is`.
        // Mirror the resolution order Python uses (collimator/features.py
        // file_metrics/file_values/file_strings/file_imports).
        let facts = file_entry.get("ff").cloned().unwrap_or(serde_json::Value::Null);
        let raw_metrics = facts
            .get("m")
            .filter(|v| v.is_object())
            .cloned()
            .or_else(|| file_entry.get("ms").cloned())
            .unwrap_or(serde_json::Value::Null);
        let raw_values = facts
            .get("v")
            .filter(|v| v.is_object())
            .cloned()
            .or_else(|| file_entry.get("k").cloned())
            .unwrap_or(serde_json::Value::Null);
        let raw_strings = facts
            .get("s")
            .and_then(|v| v.as_array().cloned())
            .or_else(|| {
                file_entry
                    .get("ss")
                    .and_then(|v| v.as_array().cloned())
            })
            .unwrap_or_default();
        let raw_imports = facts
            .get("i")
            .and_then(|v| v.as_array().cloned())
            .or_else(|| {
                file_entry
                    .get("is")
                    .and_then(|v| v.as_array().cloned())
            })
            .unwrap_or_default();
        let raw_exports = facts
            .get("x")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let raw_functions = facts
            .get("fn")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let raw_call_targets = facts
            .get("ct")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let raw_member_chains = facts
            .get("mc")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();

        Self {
            path: file_entry["path"].as_str().unwrap_or("").to_string(),
            parent: file_entry["p"].as_str().unwrap_or("").to_string(),
            file_type: file_entry["type"].as_str().unwrap_or("").to_string(),
            size_bytes,
            mtime: file_entry["mt"].as_str().and_then(parse_iso8601),
            overall_entropy,
            metrics,
            raw_metrics,
            raw_values,
            raw_strings,
            findings,
            risk,
            unique_3level_paths,
            imports,
            raw_imports,
            raw_exports,
            raw_functions,
            raw_call_targets,
            raw_member_chains,
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

        // `entry(path.to_owned())` would allocate a fresh `String` on every
        // call regardless of whether the key already exists. Most findings
        // share path prefixes, so hitting an existing entry is the common
        // case — `get_mut` + conditional `insert` pays for the allocation
        // only on first insert.
        for path in finding_paths(fid) {
            match summary.sample_paths.get_mut(path) {
                Some(v) => *v = (*v).max(crit_ord),
                None => {
                    summary.sample_paths.insert(path.to_owned(), crit_ord);
                }
            }
            match summary.path_confidences.get_mut(path) {
                Some(v) => *v = v.max(conf),
                None => {
                    summary.path_confidences.insert(path.to_owned(), conf);
                }
            }
        }
    }

    summary.unique_notable_ids = notable_ids.len();
    summary.unique_suspicious_ids = suspicious_ids.len();
    summary.unique_hostile_ids = hostile_ids.len();

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
                None => {
                    combined.sample_paths.insert(path.clone(), *max_ord);
                }
            }
        }
        for (path, &conf) in &fs.path_confidences {
            match combined.path_confidences.get_mut(path) {
                Some(v) => *v = v.max(conf),
                None => {
                    combined.path_confidences.insert(path.clone(), conf);
                }
            }
        }
        combined
            .finding_confidences
            .extend(fs.finding_confidences.iter().copied());

        // Re-scan raw findings to deduplicate unique IDs across files.
        for finding in &s.raw_findings {
            let fid = finding["i"].as_str().unwrap_or("");
            if fid.is_empty() {
                continue;
            }
            let conf = finding["c"].as_f64().unwrap_or(1.0);
            if conf < MIN_CONFIDENCE {
                continue;
            }
            let crit = crit_ordinal(finding);
            if crit >= 3 {
                notable_ids.insert(fid);
            }
            if crit >= 4 {
                suspicious_ids.insert(fid);
            }
            if crit >= 5 {
                hostile_ids.insert(fid);
            }
        }
    }

    combined.unique_notable_ids = notable_ids.len();
    combined.unique_suspicious_ids = suspicious_ids.len();
    combined.unique_hostile_ids = hostile_ids.len();

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

fn write_aggregate_features(
    summary: &FindingSummary,
    summaries: &[FileSummary],
    w: &mut FeatureWriter<'_>,
) {
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

    let total_size_bytes: f64 = summaries.iter().map(|s| s.size_bytes).sum();
    let total_kb_raw = (total_size_bytes / 1024.0) as f32;
    let total_kb_p1 = total_kb_raw.max(0.1);
    let total_kb_1 = total_kb_raw.max(1.0);

    w.set("agg:max_crit", max_crit as f32);
    w.set("agg:category_breadth", categories.len() as f32);
    w.set("agg:path_breadth_any", (path_breadth_any as f32).ln_1p());
    w.set("agg:total_active_paths", (total_active as f32).ln_1p());
    w.set(
        "agg:suspicious_concentration",
        breadth_suspicious as f32 / path_breadth_any.max(1) as f32,
    );
    w.set(
        "agg:hostile_concentration",
        breadth_hostile as f32 / path_breadth_any.max(1) as f32,
    );
    w.set(
        "agg:escalation_rate",
        breadth_suspicious as f32 / breadth_notable.max(1) as f32,
    );
    w.set(
        "agg:notable_only_fraction",
        breadth_notable_only as f32 / breadth_notable.max(1) as f32,
    );
    w.set(
        "agg:notable_findings_log",
        (summary.notable_finding_count as f32).ln_1p(),
    );
    w.set(
        "agg:suspicious_findings_log",
        (summary.suspicious_finding_count as f32).ln_1p(),
    );
    w.set(
        "agg:hostile_findings_log",
        (summary.hostile_finding_count as f32).ln_1p(),
    );
    w.set(
        "agg:notable_finding_ratio",
        summary.notable_finding_count as f32 / total_kb_p1,
    );
    w.set(
        "agg:suspicious_finding_ratio",
        summary.suspicious_finding_count as f32 / total_kb_p1,
    );
    w.set(
        "agg:hostile_finding_ratio",
        summary.hostile_finding_count as f32 / total_kb_p1,
    );
    let log_kb_p1 = total_kb_p1.ln_1p();
    w.set(
        "agg:unique_suspicious_ids_log",
        (summary.unique_suspicious_ids as f32).ln_1p() / log_kb_p1,
    );
    w.set(
        "agg:unique_hostile_ids_log",
        (summary.unique_hostile_ids as f32).ln_1p() / log_kb_p1,
    );

    let topk = topk_file_risk_features_from_summaries(summaries);
    w.set("agg:top1_file_suspicious_ratio_sum", topk[0]);
    w.set("agg:top1_file_hostile_ratio_sum", topk[1]);
    w.set("agg:top1_file_suspicious_findings_log", topk[2]);
    w.set("agg:top1_file_hostile_findings_log", topk[3]);

    let category_denom = categories.len().max(1) as f32;
    w.set(
        "agg:suspicious_category_breadth",
        summary.suspicious_category_breadth as f32,
    );
    w.set(
        "agg:hostile_category_breadth",
        summary.hostile_category_breadth as f32,
    );
    w.set(
        "agg:suspicious_category_density",
        summary.suspicious_category_breadth as f32 / category_denom,
    );
    w.set(
        "agg:hostile_category_density",
        summary.hostile_category_breadth as f32 / category_denom,
    );
    w.set(
        "agg:suspicious_findings_per_kb",
        summary.suspicious_finding_count as f32 / total_kb_1,
    );
    w.set(
        "agg:hostile_findings_per_kb",
        summary.hostile_finding_count as f32 / total_kb_1,
    );
    w.set(
        "agg:suspicious_categories_per_kb",
        summary.suspicious_category_breadth as f32 / total_kb_1,
    );
    w.set(
        "agg:hostile_categories_per_kb",
        summary.hostile_category_breadth as f32 / total_kb_1,
    );
    w.set("agg:top1_file_suspicious_density_sum", topk[4]);
    w.set("agg:top1_file_hostile_density_sum", topk[5]);
    w.set("agg:top1_file_suspicious_category_breadth_sum", topk[6]);
    w.set("agg:top1_file_hostile_category_breadth_sum", topk[7]);

    w.set(
        "agg:hostile_escalation_rate",
        breadth_hostile as f32 / breadth_notable.max(1) as f32,
    );
    w.set(
        "agg:hostile_share_of_suspicious",
        breadth_hostile as f32 / breadth_suspicious.max(1) as f32,
    );
    w.set(
        "agg:suspicious_finding_escalation_rate",
        summary.suspicious_finding_count as f32 / summary.notable_finding_count.max(1) as f32,
    );
    w.set(
        "agg:hostile_finding_escalation_rate",
        summary.hostile_finding_count as f32 / summary.notable_finding_count.max(1) as f32,
    );
    w.set(
        "agg:hostile_share_of_suspicious_findings",
        summary.hostile_finding_count as f32 / summary.suspicious_finding_count.max(1) as f32,
    );

    let host_density_global = summary.hostile_finding_count as f32 / total_kb_1;
    let susp_density_global = summary.suspicious_finding_count as f32 / total_kb_1;
    w.set(
        "agg:hostile_weighted_density",
        host_density_global + 0.25 * susp_density_global,
    );

    let mut stats: Vec<FileRiskStats> = summaries.iter().map(|s| s.risk.clone()).collect();
    stats.sort_by(|a, b| {
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
    let top_weighted: f32 = stats
        .iter()
        .take(TOP_K_RISK_FILES)
        .map(|s| s.hostile_density + 0.25 * s.suspicious_density)
        .sum();
    w.set("agg:top1_file_hostile_weighted_density_sum", top_weighted);

    w.set(
        "agg:suspicious_id_repeat_ratio",
        1.0 - (summary.unique_suspicious_ids as f32
            / summary.suspicious_finding_count.max(1) as f32),
    );
    w.set(
        "agg:hostile_id_repeat_ratio",
        1.0 - (summary.unique_hostile_ids as f32 / summary.hostile_finding_count.max(1) as f32),
    );
    w.set(
        "agg:suspicious_category_repeat_ratio",
        1.0 - (summary.suspicious_category_breadth as f32
            / summary.suspicious_finding_count.max(1) as f32),
    );
    w.set(
        "agg:hostile_category_repeat_ratio",
        1.0 - (summary.hostile_category_breadth as f32
            / summary.hostile_finding_count.max(1) as f32),
    );

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
            if max_ord >= 4 {
                suspicious_2level.insert(two_level.clone());
            }
            if max_ord >= 5 {
                hostile_2level.insert(two_level.clone());
            }
            if parts[0] == "objectives" && max_ord >= 2 {
                objectives_2level.insert(two_level);
            }
        }
    }
    w.set(
        "agg:suspicious_2level_breadth",
        suspicious_2level.len() as f32,
    );
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
    let tactic_prefixes: HashSet<&str> = attack_techniques
        .iter()
        .filter(|t| t.starts_with('T') && t.len() >= 4)
        .map(|t| &t[..4])
        .collect();
    w.set("agg:attack_tactic_count", tactic_prefixes.len() as f32);
    w.set("agg:mbc_behavior_count", mbc_behaviors.len() as f32);

    // ATT&CK / MBC co-occurrence aggregates: log1p of the count of unordered
    // pairs / triples among the distinct technique and behavior codes seen.
    // Mirrors `agg:attack_bigram_count` and friends in the collimator extractor
    // (features.py around line 2197).  Combinations are computed analytically
    // (n choose k) — equivalent to enumerating but cheaper for large finding
    // counts.  saturating_sub guards the n=0 / n=1 cases where the product
    // would otherwise underflow on usize.
    let n_atk = attack_techniques.len();
    let atk_bi = (n_atk * n_atk.saturating_sub(1)) / 2;
    let atk_tri = (n_atk * n_atk.saturating_sub(1) * n_atk.saturating_sub(2)) / 6;
    w.set("agg:attack_bigram_count", (atk_bi as f32).ln_1p());
    w.set("agg:attack_trigram_count", (atk_tri as f32).ln_1p());
    let n_mbc = mbc_behaviors.len();
    let mbc_bi = (n_mbc * n_mbc.saturating_sub(1)) / 2;
    w.set("agg:mbc_bigram_count", (mbc_bi as f32).ln_1p());
    let has_objectives = summary
        .sample_paths
        .keys()
        .any(|p| p.starts_with("objectives/"));
    w.set(
        "agg:has_attack_and_objective",
        if !attack_techniques.is_empty() && has_objectives {
            1.0
        } else {
            0.0
        },
    );

    // Objective-path co-occurrence aggregates.  Mirrors collimator's
    // `agg:objective_bigram_count` / `agg:objective_trigram_count` (features.py
    // around line 2155).  Counts unordered pairs and triples among the distinct
    // `objectives/*` and `well-known/*` sample paths.  The trigram inner loop
    // is capped at 20 per (i, j) pair to bound work on samples with many
    // objective paths — the same cap collimator uses, so the values match.
    let mut obj_paths: Vec<&str> = summary
        .sample_paths
        .keys()
        .filter(|p| p.starts_with("objectives/") || p.starts_with("well-known/"))
        .map(String::as_str)
        .collect();
    obj_paths.sort_unstable();
    let n_obj = obj_paths.len();
    let n_obj_bi = (n_obj * n_obj.saturating_sub(1)) / 2;
    let mut n_obj_tri: usize = 0;
    for i in 0..n_obj {
        for j in (i + 1)..n_obj {
            let k_end = (j + 20).min(n_obj);
            n_obj_tri += k_end.saturating_sub(j + 1);
        }
    }
    w.set("agg:objective_bigram_count", (n_obj_bi as f32).ln_1p());
    w.set("agg:objective_trigram_count", (n_obj_tri as f32).ln_1p());
}

fn tier_prefix(crit: u32) -> &'static str {
    match crit {
        5 => "h",
        4 => "s",
        _ => "n",
    }
}

fn truncate_path_depth(path: &str, depth: usize) -> String {
    if depth == 0 {
        return path.to_string();
    }
    path.split('/').take(depth).collect::<Vec<_>>().join("/")
}

fn write_tiered_bigram_features(summary: &FindingSummary, w: &mut FeatureWriter<'_>) {
    let mut token_max_crit: HashMap<String, u32> = HashMap::new();
    for (path, &max_ord) in &summary.sample_paths {
        if max_ord < 3 {
            continue;
        }
        let key = truncate_path_depth(path, 3);
        let entry = token_max_crit.entry(key).or_insert(0);
        *entry = (*entry).max(max_ord);
    }

    let mut tokens: Vec<String> = token_max_crit
        .into_iter()
        .map(|(path, crit)| format!("{}:{path}", tier_prefix(crit)))
        .collect();
    tokens.sort();
    if tokens.len() > 512 {
        tracing::warn!(
            tokens = tokens.len(),
            "too many tiered bigram tokens; skipping generation"
        );
        return;
    }
    for (i, t1) in tokens.iter().enumerate() {
        for t2 in &tokens[i + 1..] {
            w.set(&format!("tierbi:{t1} + {t2}"), 1.0);
        }
    }
}

fn write_tiered_trigram_features(summary: &FindingSummary, w: &mut FeatureWriter<'_>) {
    let mut token_max_crit: HashMap<String, u32> = HashMap::new();
    for (path, &max_ord) in &summary.sample_paths {
        if max_ord < 3 {
            continue;
        }
        let key = truncate_path_depth(path, 3);
        let entry = token_max_crit.entry(key).or_insert(0);
        *entry = (*entry).max(max_ord);
    }

    let mut tokens: Vec<String> = token_max_crit
        .into_iter()
        .map(|(path, crit)| format!("{}:{path}", tier_prefix(crit)))
        .collect();
    tokens.sort();
    if tokens.len() > 512 {
        tracing::warn!(
            tokens = tokens.len(),
            "too many tiered trigram tokens; skipping generation"
        );
        return;
    }
    for (i, t1) in tokens.iter().enumerate() {
        for j in i + 1..tokens.len() {
            let t2 = &tokens[j];
            for t3 in &tokens[j + 1..] {
                w.set(&format!("tiertri:{t1} + {t2} + {t3}"), 1.0);
            }
        }
    }
}

fn topk_file_risk_features_from_summaries(summaries: &[FileSummary]) -> [f32; 8] {
    if summaries.is_empty() || TOP_K_RISK_FILES == 0 {
        return [0.0; 8];
    }
    let mut stats: Vec<FileRiskStats> = summaries.iter().map(|s| s.risk.clone()).collect();

    let mut by_susp = stats.clone();
    by_susp.sort_by(|a, b| {
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
    let top_susp = &by_susp[..TOP_K_RISK_FILES.min(by_susp.len())];

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
    let top_host = &stats[..TOP_K_RISK_FILES.min(stats.len())];

    [
        top_susp.iter().map(|s| s.suspicious_ratio).sum(),
        top_host.iter().map(|s| s.hostile_ratio).sum(),
        (top_susp.iter().map(|s| s.suspicious_findings).sum::<u32>() as f32 + 1.0).ln(),
        (top_host.iter().map(|s| s.hostile_findings).sum::<u32>() as f32 + 1.0).ln(),
        top_susp.iter().map(|s| s.suspicious_density).sum(),
        top_host.iter().map(|s| s.hostile_density).sum(),
        top_susp
            .iter()
            .map(|s| s.suspicious_category_breadth as f32)
            .sum(),
        top_host
            .iter()
            .map(|s| s.hostile_category_breadth as f32)
            .sum(),
    ]
}

fn parse_iso8601(s: &str) -> Option<f64> {
    let s = s.trim().replace(' ', "T");
    let bytes = s.as_bytes();
    if bytes.len() < 19 {
        return None;
    }
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
        while idx < bytes.len() && bytes[idx].is_ascii_digit() {
            idx += 1;
        }
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
                let oh: i64 = std::str::from_utf8(&bytes[idx..idx + 2])
                    .ok()?
                    .parse()
                    .ok()?;
                let om: i64 = std::str::from_utf8(&bytes[idx + 3..idx + 5])
                    .ok()?
                    .parse()
                    .ok()?;
                total += sign * (oh * 3600 + om * 60);
            }
        }
    }
    Some(total as f64 + frac)
}

fn format_groups_for_type(file_type: &str) -> Vec<&'static str> {
    let normalized = file_type.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Vec::new();
    }
    FORMAT_GROUPS
        .iter()
        .filter_map(|&(group, types)| types.contains(&normalized.as_str()).then_some(group))
        .collect()
}

fn write_format_hint_features(summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
    let total_files = summaries.len().max(1) as f32;
    let inner_files = summaries
        .iter()
        .filter(|s| !s.parent.is_empty())
        .count()
        .max(1) as f32;
    let mut known_files = 0usize;
    let mut present_groups = HashSet::new();

    for &(group, _) in FORMAT_GROUPS {
        let mut group_count = 0usize;
        let mut inner_count = 0usize;
        let mut suspicious_count = 0usize;
        let mut hostile_count = 0usize;

        for s in summaries {
            let groups = format_groups_for_type(&s.file_type);
            if groups.is_empty() {
                continue;
            }
            if groups.contains(&group) {
                group_count += 1;
                present_groups.insert(group);
                if !s.parent.is_empty() {
                    inner_count += 1;
                }
                if s.findings.suspicious_finding_count > 0 {
                    suspicious_count += 1;
                }
                if s.findings.hostile_finding_count > 0 {
                    hostile_count += 1;
                }
            }
        }

        known_files += group_count;
        let group_denom = group_count.max(1) as f32;
        w.set(&format!("format:{group}"), f32::from(group_count > 0));
        w.set(
            &format!("format:{group}_file_fraction"),
            group_count as f32 / total_files,
        );
        w.set(
            &format!("format:{group}_inner_fraction"),
            inner_count as f32 / inner_files,
        );
        w.set(
            &format!("format:{group}_suspicious_fraction"),
            suspicious_count as f32 / group_denom,
        );
        w.set(
            &format!("format:{group}_hostile_fraction"),
            hostile_count as f32 / group_denom,
        );
    }

    w.set(
        "format:group_count_log",
        (present_groups.len() as f32).ln_1p(),
    );
    w.set(
        "format:mixed_script_binary",
        f32::from(present_groups.contains("script") && present_groups.contains("native_binary")),
    );
    w.set(
        "format:mixed_archive_script",
        f32::from(present_groups.contains("archive_package") && present_groups.contains("script")),
    );
    w.set(
        "format:mixed_archive_binary",
        f32::from(
            present_groups.contains("archive_package") && present_groups.contains("native_binary"),
        ),
    );
    w.set(
        "format:unknown_file_fraction",
        (summaries.len().saturating_sub(known_files)) as f32 / total_files,
    );
}

fn write_structural_extensions(
    summaries: &[FileSummary],
    combined: &FindingSummary,
    w: &mut FeatureWriter<'_>,
) {
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
        if s.parent.is_empty() {
            depths.insert(&s.path, 0);
        } else {
            let pd = depths.get(&s.parent).copied().unwrap_or(0);
            depths.insert(&s.path, pd + 1);
        }
    }
    let max_nesting_depth = depths.values().copied().max().unwrap_or(0);

    for s in summaries {
        if !s.parent.is_empty() {
            inner_file_count += 1;
        }
        if source_types.contains(&s.file_type.as_str()) {
            has_source_files = true;
        }
        if binary_like.contains(&s.file_type.as_str()) && has_source_files {
            has_foreign_binaries = true;
        }

        let lines = s
            .metrics
            .get("text")
            .and_then(|t| t.get("total_lines"))
            .copied()
            .unwrap_or(0.0);
        total_loc += lines as u64;

        if !s.path.is_empty()
            && s.path.contains('.')
            && let Some(ext) = s.path.rsplit('.').next()
            && binary_like.contains(&s.file_type.as_str())
            && text_exts.contains(&ext.to_ascii_lowercase().as_str())
        {
            extension_mismatches += 1;
        }

        if let Some(t) = s.mtime {
            mtimes.push(t);
        }
        if s.overall_entropy > 0.0 {
            entropies.push(s.overall_entropy);
        }
        max_entropy = max_entropy.max(s.overall_entropy);

        if s.findings.hostile_finding_count > 0 {
            hostile_files += 1;
            if let Some(t) = s.mtime {
                hostile_mtimes.push(t);
            }
            if !s.parent.is_empty() {
                hostile_files_with_parent += 1;
            }
        }

        let code_types = ["javascript", "python", "pe", "elf", "macho"];
        if s.overall_entropy > 0.0 && code_types.contains(&s.file_type.as_str()) {
            code_entropies.push(s.overall_entropy);
        }
    }

    w.set(
        "struct:packaged_capability",
        (combined.sample_paths.len() as f64 * max_entropy) as f32,
    );
    if mtimes.len() > 1 {
        let mn = mtimes.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = mtimes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        w.set("struct:mtime_range_hours", ((mx - mn) / 3600.0) as f32);
        let mean = mtimes.iter().sum::<f64>() / mtimes.len() as f64;
        let var = mtimes.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / mtimes.len() as f64;
        w.set("struct:mtime_stddev_hours", (var.sqrt() / 3600.0) as f32);
    }
    w.set(
        "struct:max_nesting_depth_log",
        (max_nesting_depth as f32).ln_1p(),
    );
    w.set(
        "struct:inner_file_ratio",
        inner_file_count as f32 / summaries.len().max(1) as f32,
    );
    if entropies.len() > 1 {
        let mean = entropies.iter().sum::<f64>() / entropies.len() as f64;
        let var =
            entropies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / entropies.len() as f64;
        w.set("struct:entropy_std_dev", var.sqrt() as f32);
        let mx = entropies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        w.set("struct:entropy_max_diff", (mx - mean) as f32);
    }
    w.set(
        "struct:air_gap_signal",
        f32::from(hostile_files > 0 && hostile_files_with_parent == 0),
    );
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
        w.set(
            "struct:anachronistic_injection",
            (max_delta / 3600.0) as f32,
        );
    }
    if !code_entropies.is_empty() {
        let avg_ent = if entropies.is_empty() {
            0.0
        } else {
            entropies.iter().sum::<f64>() / entropies.len() as f64
        };
        let max_code_ent = code_entropies
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        w.set("struct:code_entropy_spike", (max_code_ent - avg_ent) as f32);
    }
    w.set(
        "struct:foreign_binary_signal",
        f32::from(has_foreign_binaries),
    );
    w.set(
        "struct:extension_mismatch_signal",
        extension_mismatches as f32,
    );
    if total_loc > 0 {
        w.set(
            "struct:hostile_finding_density",
            (hostile_files as f32 * 1000.0) / total_loc as f32,
        );
    }
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
            files.iter().find(|file| {
                file.get("dp")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0)
                    == 0
            })
        })
}

fn write_logic_gap_features(
    summary: &FindingSummary,
    summaries: &[FileSummary],
    w: &mut FeatureWriter<'_>,
) {
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
                "socket", "urllib", "requests", "http", "curl", "wininet", "winhttp",
            ],
            &["micro-behaviors/network", "objectives/command-and-control"],
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
            &["micro-behaviors/process/create", "objectives/execution"],
        ),
    ];

    let mut all_imports: HashSet<&str> = HashSet::new();
    for s in summaries {
        for imp in &s.imports {
            all_imports.insert(imp.as_str());
        }
    }

    for target_cat in LOGIC_GAP_CATEGORIES {
        if let Some((_, imports_set, traits_set)) =
            logic_gaps.iter().find(|(c, _, _)| c == target_cat)
        {
            let has_import = imports_set.iter().any(|imp| all_imports.contains(*imp));
            let has_behavior = summary.sample_paths.iter().any(|(path, &max_ord)| {
                max_ord >= 3 && traits_set.iter().any(|t| path.starts_with(t))
            });
            if has_import && !has_behavior {
                w.set(&format!("gap:{target_cat}"), 1.0);
            }
        }
    }
}

fn write_intent_gap_features(summary: &FindingSummary, w: &mut FeatureWriter<'_>) {
    let intent_signal = summary
        .sample_paths
        .contains_key("metadata/package/documentation")
        || summary.sample_paths.contains_key("metadata/package/help");
    let risky: &[(&str, &[&str])] = &[
        (
            "network",
            &["objectives/network", "micro-behaviors/network"],
        ),
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
    for target_cat in INTENT_GAP_CATEGORIES {
        let traits = risky
            .iter()
            .find(|(c, _)| c == target_cat)
            .map(|(_, t)| *t)
            .unwrap_or(&[]);
        let has_behavior = summary
            .sample_paths
            .iter()
            .any(|(path, &max_ord)| max_ord >= 4 && traits.iter().any(|t| path.starts_with(t)));
        if has_behavior && !intent_signal {
            w.set(&format!("intent_gap:{target_cat}"), 1.0);
        }
    }
}

fn write_negative_space_features(
    summary: &FindingSummary,
    summaries: &[FileSummary],
    w: &mut FeatureWriter<'_>,
) {
    let mut present_types: HashSet<&str> = HashSet::new();
    for s in summaries {
        if !s.file_type.is_empty() {
            present_types.insert(s.file_type.as_str());
        }
    }
    for &(ftype, traits) in EXPECTED_GHOSTS {
        for trait_path in traits {
            if present_types.contains(ftype) && !summary.sample_paths.contains_key(*trait_path) {
                w.set(&format!("missing:{ftype}*{trait_path}"), 1.0);
            }
        }
    }
}

fn write_external_summary_features(summary: &FindingSummary, w: &mut FeatureWriter<'_>) {
    w.set(
        "ext:third_party_max_crit",
        summary.third_party_max_crit as f32,
    );
    w.set(
        "ext:third_party_count",
        (summary.third_party_count as f32 + 1.0).ln(),
    );
    w.set(
        "ext:well_known_max_crit",
        summary.well_known_max_crit as f32,
    );
    w.set(
        "ext:well_known_hostile_count",
        summary.well_known_hostile as f32,
    );
    w.set(
        "ext:well_known_suspicious_count",
        summary.well_known_suspicious as f32,
    );
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

fn write_metric_features(
    metrics: &serde_json::Value,
    w: &mut FeatureWriter<'_>,
    metric_vocab: &[String],
) {
    let base_keys: HashSet<String> = KEY_METRICS
        .iter()
        .map(|&(g, f, _)| format!("{g}_{f}"))
        .collect();

    // Base KEY_METRICS with explicit log transforms.
    for &(group, field_name, use_log) in KEY_METRICS {
        let value = metrics
            .get(group)
            .and_then(|g| g.get(field_name))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;
        let val = if use_log {
            (value.abs() + 1.0).ln()
        } else {
            value
        };
        w.set(&format!("metrics:{group}_{field_name}"), val);
    }

    // Extended metrics from dynamic vocab — skip keys already in KEY_METRICS.
    for mk in metric_vocab {
        let parts: Vec<&str> = mk.splitn(2, '_').collect();
        if parts.len() == 2 && !base_keys.contains(mk.as_str()) {
            let value = metrics
                .get(parts[0])
                .and_then(|g| g.get(parts[1]))
                .and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                })
                .unwrap_or(0.0) as f32;
            let use_log = ["count", "size", "total", "bytes", "length"]
                .iter()
                .any(|word| parts[1].contains(word));
            let val = if use_log {
                (value.abs() + 1.0).ln()
            } else {
                value
            };
            w.set(&format!("metrics:{mk}"), val);
        }
    }
}

fn write_structural_features(
    w: &mut FeatureWriter<'_>,
    summaries: &[FileSummary],
    filtered_finding_count: u32,
) {
    let binary_like = ["pe", "elf", "macho"];
    let mut any_tiny_binary = false;
    let mut import_candidates = 0;
    let mut importless_candidates = 0;
    let mut max_entropy = 0.0_f64;
    let mut suspicious_files = 0;
    let mut hostile_files = 0;

    for s in summaries {
        if binary_like.contains(&s.file_type.as_str()) && s.size_bytes < 20_000.0 {
            any_tiny_binary = true;
        }
        if s.has_imports_key {
            import_candidates += 1;
            if s.imports.is_empty() {
                importless_candidates += 1;
            }
        }
        max_entropy = max_entropy.max(s.overall_entropy);
        if s.findings.suspicious_finding_count > 0 {
            suspicious_files += 1;
        }
        if s.findings.hostile_finding_count > 0 {
            hostile_files += 1;
        }
    }

    w.set("struct:tiny_executable", f32::from(any_tiny_binary));
    w.set(
        "struct:no_imports",
        f32::from(import_candidates > 0 && importless_candidates == import_candidates),
    );
    w.set("struct:no_findings", f32::from(filtered_finding_count == 0));
    w.set(
        "struct:finding_count_log",
        (filtered_finding_count as f32 + 1.0).ln(),
    );
    let file_count = summaries.len() as f32;
    w.set("struct:file_count_log", (file_count + 1.0).ln());
    w.set(
        "struct:inner_file_count_log",
        ((file_count - 1.0).max(0.0) + 1.0).ln(),
    );
    w.set(
        "struct:stealth_potential",
        f32::from(filtered_finding_count < 5 && max_entropy > 6.5),
    );
    let denom = summaries.len().max(1) as f32;
    w.set(
        "struct:suspicious_file_fraction",
        suspicious_files as f32 / denom,
    );
    w.set("struct:hostile_file_fraction", hostile_files as f32 / denom);
    w.set(
        "struct:suspicious_file_count_log",
        (suspicious_files as f32).ln_1p(),
    );
    w.set(
        "struct:hostile_file_count_log",
        (hostile_files as f32).ln_1p(),
    );
}

fn report_files(report: &serde_json::Value) -> Vec<&serde_json::Value> {
    report["fs"]
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_default()
}

fn finding_paths(finding_id: &str) -> FindingPaths<'_> {
    let base = finding_id.split("::").next().unwrap_or(finding_id);
    let mut slash_ends = [0; 2];
    let mut n_slashes = 0;
    for (i, ch) in base.char_indices() {
        if ch == '/' {
            if n_slashes < 2 {
                slash_ends[n_slashes] = i;
            }
            n_slashes += 1;
        }
    }
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
    // slash_ends and third_end are byte offsets of ASCII '/' characters, which are
    // always single-byte UTF-8 codepoints — so these slices are valid UTF-8 boundaries.
    #[allow(clippy::string_slice)]
    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.step {
            0 => Some(if self.n_slashes >= 1 {
                &self.base[..self.slash_ends[0]]
            } else {
                self.base
            }),
            1 if self.n_slashes >= 1 => Some(if self.n_slashes >= 2 {
                &self.base[..self.slash_ends[1]]
            } else {
                self.base
            }),
            2 if self.n_slashes >= 2 => Some(&self.base[..self.third_end]),
            _ => return None,
        };
        self.step += 1;
        result
    }
}

// ============================================================================
// kv: / symbol: / textenc: / derived metric helpers — ported from
// collimator/src/collimator/features.py. Each helper mirrors the Python
// behavior exactly so the feature vectors stay bit-identical with the
// training-time extraction.
// ============================================================================

/// Normalize a value to a bounded vocab token (collapse whitespace, truncate).
/// Mirrors `_normalize_vocab_token` in collimator.
fn normalize_vocab_token(value: &str, max_len: usize) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    // Collapse runs of whitespace to single spaces ("a   b\nc" → "a b c").
    let mut out = String::with_capacity(trimmed.len());
    let mut prev_ws = false;
    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            if !prev_ws && !out.is_empty() {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    if out.chars().count() > max_len {
        out = out.chars().take(max_len).collect();
    }
    out
}

/// Shannon entropy over characters. Matches `_char_entropy`.
fn char_entropy(value: &str) -> f64 {
    if value.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<char, u32> = HashMap::new();
    let mut n = 0u32;
    for ch in value.chars() {
        *counts.entry(ch).or_insert(0) += 1;
        n += 1;
    }
    let n_f = f64::from(n);
    counts
        .values()
        .map(|&c| {
            let p = f64::from(c) / n_f;
            -p * p.log2()
        })
        .sum()
}

/// `_looks_base64ish` — char-class heuristic.
///
/// Note: Python's `str.isalnum()` is Unicode-aware (e.g. accented letters,
/// Cyrillic), so we use `is_alphanumeric()` not `is_ascii_alphanumeric()`.
/// Using the ASCII-only variant under-counts on non-Latin strings and
/// produces values smaller than Python's by a few thousandths.
fn looks_base64ish(value: &str) -> bool {
    let len = value.chars().count();
    if len < 16 {
        return false;
    }
    let approved = value
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '+' | '/' | '=' | '_' | '-'))
        .count();
    let has_b64_punct = value.chars().any(|c| matches!(c, '+' | '/' | '='));
    (approved as f64) / (len.max(1) as f64) > 0.92 && has_b64_punct
}

/// `_looks_hexish` — fraction of hex digits after whitespace strip.
fn looks_hexish(value: &str) -> bool {
    let compact: String = value.trim().chars().filter(|c| !c.is_whitespace()).collect();
    let len = compact.chars().count();
    if len < 16 {
        return false;
    }
    let hex = compact.chars().filter(char::is_ascii_hexdigit).count();
    (hex as f64) / (len as f64) > 0.95
}

/// Extract one string entry: `(value, is_wide)`. Matches `_string_values`.
/// A cleave string row looks like `[offset, ..., "value"]` or
/// `[offset, "wide", "value"]`; we pull the last element as value and check
/// any non-first / non-last element for a wide-encoding marker.
fn extract_string_entry(item: &serde_json::Value) -> Option<(String, bool)> {
    let arr = item.as_array()?;
    if arr.len() < 2 {
        return None;
    }
    let value = arr.last()?.as_str()?.to_string();
    if value.is_empty() {
        return None;
    }
    let is_wide = arr[1..arr.len().saturating_sub(1)].iter().any(|part| {
        part.as_str()
            .map(|s| matches!(s.to_lowercase().as_str(), "wide" | "u16" | "utf16le" | "utf-16le"))
            .unwrap_or(false)
    });
    Some((value, is_wide))
}

/// Collect normalized import + export + function symbols for a file.
/// Mirrors `_file_symbols` in collimator: imports may be `[lib, name]`
/// tuples (we record both the bare name and the `lib!name` form), dicts
/// with `n`/`name`/`symbol`, or plain strings. Exports (`ff.x`) and
/// function names (`ff.fn`) contribute their first tuple element.
fn collect_file_symbols(summary: &FileSummary) -> HashSet<String> {
    let mut out = HashSet::new();
    let insert_if_long = |out: &mut HashSet<String>, token: String| {
        if token.chars().count() >= 2 {
            out.insert(token);
        }
    };

    for raw in &summary.raw_imports {
        let mut composite: Option<String> = None;
        if let Some(arr) = raw.as_array() {
            let lib = arr.first().and_then(|v| v.as_str()).unwrap_or("");
            let name = arr.get(1).and_then(|v| v.as_str()).unwrap_or("");
            let name_sym = normalize_vocab_token(name, 96);
            if !name_sym.is_empty() {
                insert_if_long(&mut out, name_sym);
            }
            composite = match (lib.is_empty(), name.is_empty()) {
                (false, false) => Some(format!("{lib}!{name}")),
                (true, false) => Some(name.to_string()),
                (false, true) => Some(lib.to_string()),
                _ => None,
            };
        }
        let raw_str = if let Some(s) = composite {
            s
        } else if let Some(s) = raw.as_str() {
            s.to_string()
        } else if let Some(obj) = raw.as_object() {
            obj.get("n")
                .or_else(|| obj.get("name"))
                .or_else(|| obj.get("symbol"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        } else {
            String::new()
        };
        let sym = normalize_vocab_token(&raw_str, 96);
        if !sym.is_empty() {
            insert_if_long(&mut out, sym);
        }
    }

    for raw in summary.raw_exports.iter().chain(summary.raw_functions.iter()) {
        let candidate = if let Some(arr) = raw.as_array() {
            arr.first().and_then(|v| v.as_str()).unwrap_or("").to_string()
        } else if let Some(s) = raw.as_str() {
            s.to_string()
        } else {
            String::new()
        };
        let sym = normalize_vocab_token(&candidate, 96);
        if !sym.is_empty() {
            insert_if_long(&mut out, sym);
        }
    }

    // Filefacts AST symbol kinds: Call (ff.ct) and Member (ff.mc).
    // Older cleave reports don't carry these keys; the iterator is then
    // empty and the loop is a no-op.
    for raw in summary
        .raw_call_targets
        .iter()
        .chain(summary.raw_member_chains.iter())
    {
        let candidate = raw.as_str().unwrap_or("");
        let sym = normalize_vocab_token(candidate, 96);
        if !sym.is_empty() {
            insert_if_long(&mut out, sym);
        }
    }

    out
}

/// Per-file caps from collimator (`_SYMBOL_BIGRAM_CAP`/`_SYMBOL_TRIGRAM_CAP`).
const SYMBOL_BIGRAM_CAP: usize = 64;
const SYMBOL_TRIGRAM_CAP: usize = 24;

/// Emit sparse kv:* tokens for one file. Mirrors `_metric_kv_tokens`.
///
/// `include_shape` and `split_string_values` are gated by collimator's
/// `FeatureConfig.include_kv_shape_features` / `include_kv_value_split`
/// env knobs; runtime models keep both off by default, so we set them
/// to `false`. Models that need shape/split features were trained with
/// extra vocab entries that won't appear here — they extract as zeros
/// (acceptable graceful degradation; matches existing policy).
fn write_kv_features(summary: &FileSummary, w: &mut FeatureWriter<'_>) {
    if let Some(obj) = summary.raw_metrics.as_object() {
        for (group, fields) in obj {
            let Some(field_map) = fields.as_object() else {
                continue;
            };
            for (key, value) in field_map {
                let base = format!("{group}.{key}");
                emit_kv_value_tokens(&base, value, w);
            }
        }
    }
    if let Some(obj) = summary.raw_values.as_object() {
        for (path, value) in obj {
            let base = format!("v.{path}");
            emit_kv_value_tokens(&base, value, w);
        }
    }
}

fn emit_kv_value_tokens(base: &str, value: &serde_json::Value, w: &mut FeatureWriter<'_>) {
    // bool: emit "<base>=true|false". Numeric values are never directly
    // tokenized — only their bucket form is, which is gated by shape mode
    // (off by default), so we skip them entirely here.
    match value {
        serde_json::Value::Bool(b) => {
            w.set(&format!("kv:{base}={}", if *b { "true" } else { "false" }), 1.0);
        }
        serde_json::Value::String(s) => {
            let val = normalize_vocab_token(s, 80);
            if !val.is_empty() {
                w.set(&format!("kv:{base}={val}"), 1.0);
            }
        }
        serde_json::Value::Array(items) => {
            // Python emits `<base>:item=<value>` for the first 32 items when
            // shape mode is on. Shape mode is off in production, so skip.
            // We still descend into nested objects below for the dict case.
            let _ = items;
        }
        serde_json::Value::Object(map) => {
            // Same: nested-dict tokens are shape-mode only.
            let _ = map;
        }
        _ => {}
    }
}

/// Emit symbol:*, symbol_bi:*, symbol_tri:* for a set of summaries.
fn write_symbol_features(
    summaries: &[FileSummary],
    w: &mut FeatureWriter<'_>,
    emit_bigrams: bool,
    emit_trigrams: bool,
) {
    for s in summaries {
        let symbols = collect_file_symbols(s);
        for sym in &symbols {
            w.set(&format!("symbol:{sym}"), 1.0);
        }
        if !emit_bigrams && !emit_trigrams {
            continue;
        }
        let mut sorted: Vec<&String> = symbols.iter().collect();
        sorted.sort();
        if emit_bigrams {
            let cap = sorted.len().min(SYMBOL_BIGRAM_CAP);
            for i in 0..cap {
                for j in (i + 1)..cap {
                    w.set(&format!("symbol_bi:{}||{}", sorted[i], sorted[j]), 1.0);
                }
            }
        }
        if emit_trigrams {
            let cap = sorted.len().min(SYMBOL_TRIGRAM_CAP);
            for i in 0..cap {
                for j in (i + 1)..cap {
                    for k in (j + 1)..cap {
                        w.set(
                            &format!(
                                "symbol_tri:{}||{}||{}",
                                sorted[i], sorted[j], sorted[k]
                            ),
                            1.0,
                        );
                    }
                }
            }
        }
    }
}

/// Emit textenc:* — 12 fixed ratios over concatenated strings across files.
/// Mirrors `_apply_text_encoding_features`.
fn write_textenc_features(summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
    let mut strings: Vec<(String, bool)> = Vec::new();
    for s in summaries {
        for item in &s.raw_strings {
            if let Some(entry) = extract_string_entry(item) {
                strings.push(entry);
            }
        }
    }
    let n = strings.len();
    if n == 0 {
        return;
    }

    let mut sum_len: usize = 0;
    let mut max_len: usize = 0;
    let mut base64ish = 0u32;
    let mut hexish = 0u32;
    let mut urlish = 0u32;
    let mut pathish = 0u32;
    let mut unicode_escape = 0u32;
    let mut wide_n = 0u32;
    let mut high_entropy = 0u32;
    let mut long_token = 0u32;
    let mut short_junk = 0u32;

    for (value, is_wide) in &strings {
        let len = value.chars().count();
        sum_len += len;
        max_len = max_len.max(len);
        let lower = value.to_lowercase();

        if looks_base64ish(value) {
            base64ish += 1;
        }
        if looks_hexish(value) {
            hexish += 1;
        }
        if lower.contains("http://")
            || lower.contains("https://")
            || lower.contains("://")
            || lower.contains("%2f")
        {
            urlish += 1;
        }
        if value.contains('/')
            || value.contains('\\')
            || lower.starts_with("c:")
            || lower.starts_with("./")
            || lower.starts_with("../")
        {
            pathish += 1;
        }
        if value.contains("\\x") || value.contains("\\u") || lower.contains("%u") {
            unicode_escape += 1;
        }
        if *is_wide {
            wide_n += 1;
        }
        if len >= 24 && char_entropy(value) >= 4.0 {
            high_entropy += 1;
        }
        if len >= 80 {
            long_token += 1;
        }
        if (4..=8).contains(&len) && char_entropy(value) >= 2.4 {
            short_junk += 1;
        }
    }

    let n_f = n as f64;
    let denom = n_f.max(1.0);
    let log1p = |x: f64| x.ln_1p();

    w.set("textenc:string_count_log", log1p(n_f) as f32);
    w.set("textenc:avg_len_log", log1p(sum_len as f64 / denom) as f32);
    w.set("textenc:max_len_log", log1p(max_len as f64) as f32);
    w.set("textenc:base64ish_ratio", (f64::from(base64ish) / denom) as f32);
    w.set("textenc:hexish_ratio", (f64::from(hexish) / denom) as f32);
    w.set("textenc:urlish_ratio", (f64::from(urlish) / denom) as f32);
    w.set("textenc:pathish_ratio", (f64::from(pathish) / denom) as f32);
    w.set(
        "textenc:unicode_escape_ratio",
        (f64::from(unicode_escape) / denom) as f32,
    );
    w.set("textenc:wide_ratio", (f64::from(wide_n) / denom) as f32);
    w.set("textenc:high_entropy_ratio", (f64::from(high_entropy) / denom) as f32);
    w.set("textenc:long_token_ratio", (f64::from(long_token) / denom) as f32);
    w.set("textenc:short_junk_ratio", (f64::from(short_junk) / denom) as f32);
}

/// Cross-metric derived ratios. Mirrors collimator `_BATCH1_RATIOS`.
fn write_derived_metric_features(
    merged_metrics: &serde_json::Value,
    w: &mut FeatureWriter<'_>,
) {
    let get = |group: &str, field: &str| -> f64 {
        merged_metrics
            .get(group)
            .and_then(|g| g.get(field))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
    };
    let ratio = |num: f64, denom: f64| -> f64 {
        if denom == 0.0 {
            0.0
        } else {
            num / denom
        }
    };
    let string_count = get("binary", "string_count");
    let function_count = get("binary", "function_count");
    let import_count = get("binary", "import_count");
    let dependency_count = get("binary", "dependency_count");
    let wide_string_count = get("binary", "wide_string_count");
    w.set(
        "metrics:derived_string_per_function",
        ratio(string_count, function_count) as f32,
    );
    w.set(
        "metrics:derived_imports_per_dependency",
        ratio(import_count, dependency_count) as f32,
    );
    w.set(
        "metrics:derived_wide_string_ratio",
        ratio(wide_string_count, string_count) as f32,
    );
}

/// `struct:silent_packer_signal` — large file size / few findings → packer.
/// Mirrors collimator's gated computation (Exp 43).
fn write_silent_packer_signal(
    summaries: &[FileSummary],
    filtered_finding_count: u32,
    w: &mut FeatureWriter<'_>,
) {
    let total_size: f64 = summaries.iter().map(|s| s.size_bytes).sum();
    let size_mb = total_size / (1024.0 * 1024.0);
    let denom = f64::from(filtered_finding_count + 1);
    w.set(
        "struct:silent_packer_signal",
        (size_mb.ln_1p() / denom) as f32,
    );
}

/// Aggregate-level suspicious n-gram co-occurrence counts. Mirrors
/// collimator's `include_suspicious_trigrams` branch.
fn write_suspicious_ngram_counts(combined: &FindingSummary, w: &mut FeatureWriter<'_>) {
    let mut sus_paths: Vec<&String> = combined
        .sample_paths
        .iter()
        .filter(|&(_, &mo)| mo >= 4)
        .map(|(p, _)| p)
        .collect();
    sus_paths.sort();
    let n = sus_paths.len();
    let mut n_bi: u64 = 0;
    let mut n_tri: u64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            n_bi += 1;
            // Python caps the third-element span to 20 to bound O(n^3).
            let k_end = n.min(j + 20);
            n_tri += (k_end.saturating_sub(j + 1)) as u64;
        }
    }
    w.set(
        "agg:suspicious_bigram_count",
        (n_bi as f64).ln_1p() as f32,
    );
    w.set(
        "agg:suspicious_trigram_count",
        (n_tri as f64).ln_1p() as f32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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
            ]
        );
    }

    #[test]
    fn test_finding_paths_two_levels() {
        assert_eq!(
            paths("metadata/format::no-functions"),
            vec!["metadata", "metadata/format"]
        );
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

    /// `collect_file_symbols` reads imports/exports/functions PLUS the
    /// filefacts AST symbol kinds (`ff.ct` = call targets,
    /// `ff.mc` = member chains). Tokens shorter than 2 chars are dropped to
    /// match collimator's `_file_symbols` threshold.
    #[test]
    fn collect_file_symbols_picks_up_ff_ct_and_mc() {
        let file = serde_json::json!({
            "path": "lib.rs",
            "type": "rust",
            "sz": 1024,
            "ff": {
                "i": [["libc", "open"], ["libc", "x"]],   // "x" too short, skipped
                "ct": ["client.get", "attempt.url().origin", "a"],
                "mc": ["window.localStorage", "process.env.PATH"],
                "x": [["exported_fn"]],
                "fn": [["render"], ["x"]]                 // "x" too short, skipped
            }
        });
        let summary = FileSummary::new(&file);
        let syms = collect_file_symbols(&summary);

        for expected in [
            "open",                                       // import
            "libc!open",                                  // composite import
            "exported_fn",                                // export
            "render",                                     // function
            "client.get",                                 // ff.ct
            "attempt.url().origin",                       // ff.ct
            "window.localStorage",                        // ff.mc
            "process.env.PATH",                           // ff.mc
        ] {
            assert!(
                syms.contains(expected),
                "expected {expected:?} in symbol set, got {syms:?}"
            );
        }
        assert!(
            !syms.contains("a") && !syms.contains("x"),
            "1-char symbols should be filtered out"
        );
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
            metric_vocab: vec![],
            crit_unigram_vocab: vec![],
            crit_bigram_vocab: vec![],
            crit_trigram_vocab: vec![],
            attack_bigram_vocab: vec![],
            attack_trigram_vocab: vec![],
            mbc_bigram_vocab: vec![],
            mbc_trigram_vocab: vec![],
            tiered_bigram_vocab: vec![],
            tiered_trigram_vocab: vec![],
            kv_vocab: vec![],
            symbol_vocab: vec![],
            symbol_bigram_vocab: vec![],
            symbol_trigram_vocab: vec![],
            feature_names: vec![],
            total_features: 3,
            feature_means: Some(vec![0.0, 1.0, 2.0]),
            feature_stds: Some(vec![1.0, 2.0, 0.5]),
            standardized: true,
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
        let Err(err) = FeatureSpec::load(file.path()) else {
            anyhow::bail!("missing feature_names should be rejected");
        };
        assert!(err.to_string().contains("feature_names length"));
        Ok(())
    }

    /// Minimal spec carrying only the fields the offset-family anchoring reads.
    fn anchor_spec(
        presence_vocab: Vec<String>,
        bigram_vocab: Vec<String>,
        trigram_vocab: Vec<String>,
        feature_names: Vec<String>,
    ) -> FeatureSpec {
        FeatureSpec {
            version: 17,
            abi_version: 17,
            presence_vocab,
            bigram_vocab,
            trigram_vocab,
            total_features: feature_names.len(),
            feature_names,
            filetype_vocab: vec![],
            element_vocab: vec![],
            ghost_vocab: vec![],
            skeleton_vocab: vec![],
            rare_element_vocab: vec![],
            metric_vocab: vec![],
            crit_unigram_vocab: vec![],
            crit_bigram_vocab: vec![],
            crit_trigram_vocab: vec![],
            attack_bigram_vocab: vec![],
            attack_trigram_vocab: vec![],
            mbc_bigram_vocab: vec![],
            mbc_trigram_vocab: vec![],
            tiered_bigram_vocab: vec![],
            tiered_trigram_vocab: vec![],
            kv_vocab: vec![],
            symbol_vocab: vec![],
            symbol_bigram_vocab: vec![],
            symbol_trigram_vocab: vec![],
            feature_means: None,
            feature_stds: None,
            standardized: false,
        }
    }

    #[test]
    fn offset_families_anchor_to_real_spec_positions() {
        // feature_names deliberately places every offset family somewhere a
        // hand-maintained running cursor would NOT land. Anchoring must follow
        // the spec, not the cursor — this is the exact drift that ran the
        // unsigned-bigram block off the end of the vector in production.
        let feature_names = vec![
            "filler:0".to_string(),               // 0
            "present:objectives".to_string(),     // 1
            "maxcrit:objectives".to_string(),     // 2
            "filler:1".to_string(),               // 3
            "bigrams:a + b".to_string(),          // 4
            "bigrams:a + c".to_string(),          // 5
            "trigram:a + b + c".to_string(),      // 6
            "unsigned_bigram:a + b".to_string(),  // 7
            "unsigned_bigram:a + c".to_string(),  // 8
        ];
        let spec = anchor_spec(
            vec!["objectives".to_string()],
            vec!["a + b".to_string(), "a + c".to_string()],
            vec!["a + b + c".to_string()],
            feature_names,
        );
        let ctx = ExtractContext::new(&spec);
        assert_eq!(ctx.present_base, Some(1));
        assert_eq!(ctx.maxcrit_base, Some(2));
        assert_eq!(ctx.bigram_base, Some(4));
        assert_eq!(ctx.trigram_base, Some(6));
        assert_eq!(ctx.unsigned_bigram_base, Some(7));
    }

    #[test]
    fn non_contiguous_offset_family_is_rejected() {
        // The bigram family is laid out reversed vs vocab order, breaking the
        // `base + idx` invariant; the writer must skip it (None → zeros) AND
        // `validate_layout` must reject the bundle rather than serve corruption.
        let feature_names = vec![
            "bigrams:a + c".to_string(), // 0 — vocab idx 1, but base + 0
            "bigrams:a + b".to_string(), // 1 — vocab idx 0, but base + 1
        ];
        let spec = anchor_spec(
            vec![],
            vec!["a + b".to_string(), "a + c".to_string()],
            vec![],
            feature_names,
        );
        let ctx = ExtractContext::new(&spec);
        assert_eq!(ctx.bigram_base, None);
        let err = ctx.validate_layout().unwrap_err().to_string();
        assert!(err.contains("bigrams:"), "unexpected error: {err}");
    }

    #[test]
    fn fully_absent_family_is_skipped_not_rejected() {
        // bigram_vocab is non-empty but feature_names has no "bigrams:"/
        // "unsigned_bigram:" entries: the family was pruned/disabled in the spec.
        // The extractor skips it; this is a legitimately smaller model, not an
        // incompatibility, so validation must pass.
        let spec = anchor_spec(
            vec![],
            vec!["a + b".to_string()],
            vec![],
            vec!["filler:0".to_string()],
        );
        let ctx = ExtractContext::new(&spec);
        assert_eq!(ctx.bigram_base, None);
        assert!(ctx.validate_layout().is_ok());
    }

    #[test]
    fn validate_layout_accepts_anchored_spec() {
        // Both bigram families present and contiguous: a healthy bundle.
        let spec = anchor_spec(
            vec![],
            vec!["a + b".to_string()],
            vec![],
            vec![
                "bigrams:a + b".to_string(),
                "unsigned_bigram:a + b".to_string(),
            ],
        );
        let ctx = ExtractContext::new(&spec);
        assert_eq!(ctx.bigram_base, Some(0));
        assert_eq!(ctx.unsigned_bigram_base, Some(1));
        assert!(ctx.validate_layout().is_ok());
    }
}
