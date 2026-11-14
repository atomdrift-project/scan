//! The feature spec a model was trained with: its vocabularies, groups and
//! layout, and the checks that a spec matches what this extractor emits.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};

use super::names::{
    AGG_FEATURES, EXT_FEATURES, FORMAT_FEATURE_NAMES, FORMAT_SUMMARY_FEATURES, FORMULA_FEATURES,
    GAP_FEATURE_NAMES, INTENT_GAP_FEATURE_NAMES, MISSING_FEATURE_NAMES, SCORE_FEATURES,
    STRUCT_EXTENSION_FEATURES, STRUCT_FEATURES, TAIL_FEATURES, TEXTENC_FEATURES,
    metric_base_feature_names,
};
use super::structure::known_metric_getter;
use super::{
    EXPECTED_MODEL_ABI_VERSION, EXPECTED_SPEC_VERSION, EXTENDED_KEYWORD_ABI, KEYWORD_ABI,
    METRIC_TRANSFORM_ABI, MIN_LOADABLE_SPEC_VERSION, MIN_MODEL_ABI_VERSION,
};

/// Key metrics extracted from the report's `metrics` object.
/// Each entry is (group, field, use_log1p).
///
/// `use_log1p` matches training: count and size fields get log1p, ratios and
/// entropies do not. These entries are
/// always extracted regardless of `metric_vocab` because routes like
/// filetypes/c and filetypes/go reference them in `feature_names` without
/// listing them in the dynamic `metric_vocab` array.
pub(super) const KEY_METRICS: &[(&str, &str, bool)] = &[
    // Binary structure
    ("binary", "overall_entropy", false),
    ("binary", "code_entropy", false),
    ("binary", "code_to_data_ratio", false),
    ("binary", "function_count", true),
    ("binary", "complexity_per_kb", false),
    ("binary", "max_complexity", false),
    ("binary", "normalized_string_count", false),
    ("binary", "high_entropy_regions", false),
    // Binary overlay
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
    // Text full-shape fields
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

/// A keyword family: where its words come from, and the `<source>:` prefix
/// of its feature names. Keys a spec's `keyword_vocabs`, so a source this
/// build does not know is a parse error rather than a silently empty family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize)]
pub(super) enum KeywordSource {
    /// Words of the rule part of a finding ID (after `::`, or its last segment).
    #[serde(rename = "ruleword")]
    RuleId,
    /// Words of finding descriptions.
    #[serde(rename = "descword")]
    Description,
    /// Words of the whole finding ID.
    #[serde(rename = "fullword")]
    FullId,
    /// The whole-ID and description words, pooled.
    #[serde(rename = "ruletextword")]
    IdAndDescription,
    /// Words of the decoded context windows.
    #[serde(rename = "contextword")]
    Context,
}

impl KeywordSource {
    /// Every source, in feature-layout order.
    pub(super) const ALL: [Self; 5] = [
        Self::RuleId,
        Self::Description,
        Self::FullId,
        Self::IdAndDescription,
        Self::Context,
    ];

    /// The source's name as a spec spells it, which prefixes its features.
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::RuleId => "ruleword",
            Self::Description => "descword",
            Self::FullId => "fullword",
            Self::IdAndDescription => "ruletextword",
            Self::Context => "contextword",
        }
    }

    /// The first ABI that reads this source.
    const fn min_abi(self) -> u32 {
        match self {
            Self::RuleId | Self::Description => KEYWORD_ABI,
            Self::FullId | Self::IdAndDescription | Self::Context => EXTENDED_KEYWORD_ABI,
        }
    }
}

/// Feature specification loaded from `feature_spec.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct FeatureSpec {
    pub(super) version: u32,
    pub(super) abi_version: u32,
    #[serde(default)]
    pub(super) base_feature_abi: Option<u32>,
    #[serde(default)]
    pub(super) keyword_vocabs: HashMap<KeywordSource, Vec<String>>,
    #[serde(default)]
    pub(super) presence_vocab: Vec<String>,
    #[serde(default)]
    pub(super) filetype_vocab: Vec<String>,
    #[serde(default)]
    pub(super) element_vocab: Vec<String>,
    #[serde(default)]
    pub(super) bigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) ghost_vocab: Vec<String>,
    #[serde(default)]
    pub(super) skeleton_vocab: Vec<String>,
    #[serde(default)]
    pub(super) rare_element_vocab: Vec<String>,
    #[serde(default)]
    pub(super) trigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) metric_vocab: Vec<String>,
    #[serde(default)]
    pub(super) metric_transforms: Option<HashMap<String, String>>,
    #[serde(default)]
    pub(super) metric_getters: Vec<String>,
    #[serde(default)]
    pub(super) metric_reference_year: u32,
    #[serde(default)]
    pub(super) metric_correlations: HashMap<String, [[String; 2]; 2]>,
    #[serde(default)]
    pub(super) crit_unigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) crit_bigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) crit_trigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) attack_bigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) attack_trigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) mbc_bigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) mbc_trigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) tiered_bigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) tiered_trigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) tiered_quadgram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) kv_vocab: Vec<String>,
    #[serde(default)]
    pub(super) symbol_vocab: Vec<String>,
    #[serde(default)]
    pub(super) symbol_bigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) symbol_trigram_vocab: Vec<String>,
    #[serde(default)]
    pub(super) feature_names: Vec<String>,
    #[serde(default)]
    pub(super) total_features: usize,
    pub(super) feature_means: Option<Vec<f32>>,
    pub(super) feature_stds: Option<Vec<f32>>,
    #[serde(default = "default_standardized")]
    pub(super) standardized: bool,
    pub(super) trait_path_depth: usize,
    #[serde(default = "default_tiered_min_crit")]
    pub(super) tiered_min_crit: [u32; 3],
    #[serde(default)]
    pub(super) tiered_branch_min_crit: HashMap<String, u32>,
}

pub(super) const fn default_tiered_min_crit() -> [u32; 3] {
    [3; 3]
}

const fn default_standardized() -> bool {
    true
}

impl FeatureSpec {
    /// Load feature specification from a JSON file.
    ///
    /// # Errors
    /// When the file cannot be read or parsed, its version is outside what
    /// this build reads, or it fails [`Self::validate`].
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).context("reading feature spec")?;
        let spec: Self = serde_json::from_str(&data).context("parsing feature spec")?;

        if !(MIN_LOADABLE_SPEC_VERSION..=EXPECTED_SPEC_VERSION).contains(&spec.version) {
            anyhow::bail!(
                "feature spec version mismatch: this installed model uses spec v{}, but this Atomdrift Scan build accepts v{MIN_LOADABLE_SPEC_VERSION}..=v{EXPECTED_SPEC_VERSION}. \
                 The model is incompatible with this build. Run 'atomscan update-rules' to install a matching model bundle.",
                spec.version,
            );
        }
        // Older-but-loadable specs are normal while an ensemble is retrained
        // piecemeal; `atomscan version` and `update-rules` surface it as a hint.
        if spec.version < EXPECTED_SPEC_VERSION {
            tracing::debug!(
                version = spec.version,
                expected = EXPECTED_SPEC_VERSION,
                "loading an older feature spec; features added since its version are absent from the model",
            );
        }
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

    /// Apply z-score standardization using training statistics.
    ///
    /// As in training, a feature whose stats are exactly `(mean 0, std 1)` was
    /// constant during training and is zeroed rather than passed through as a
    /// raw value the model never saw. A zero std is zeroed too, instead of
    /// dividing by it.
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

    /// The ABI whose base (non-keyword) feature semantics this spec uses.
    pub(super) fn feature_semantics_abi(&self) -> u32 {
        self.base_feature_abi.unwrap_or(self.abi_version)
    }

    /// Check the spec is one this extractor can fill: a supported ABI, its
    /// keyword families and metric contract, and a consistent layout.
    pub(super) fn validate(&self) -> Result<()> {
        if !(MIN_MODEL_ABI_VERSION..=EXPECTED_MODEL_ABI_VERSION).contains(&self.abi_version) {
            anyhow::bail!(
                "feature spec ABI mismatch: spec has ABI v{} but Atomdrift Scan requires ABI v{MIN_MODEL_ABI_VERSION}..=v{EXPECTED_MODEL_ABI_VERSION}",
                self.abi_version,
            );
        }
        self.validate_keywords()?;
        anyhow::ensure!(
            self.trait_path_depth > 0,
            "trait_path_depth must be positive"
        );
        anyhow::ensure!(
            !self.presence_vocab.is_empty(),
            "feature spec missing presence_vocab entries"
        );
        anyhow::ensure!(
            self.feature_names.len() == self.total_features,
            "feature spec feature_names length {} does not match total_features {}",
            self.feature_names.len(),
            self.total_features
        );
        self.validate_metrics()?;
        self.validate_standardization()?;
        self.note_unknown_features();
        Ok(())
    }

    /// Keyword families need a keyword ABI, a base ABI this build extracts,
    /// and only the sources their ABI defines.
    fn validate_keywords(&self) -> Result<()> {
        let abi = self.abi_version;
        if abi < KEYWORD_ABI {
            anyhow::ensure!(
                self.base_feature_abi.is_none() && self.keyword_vocabs.is_empty(),
                "keyword features require ABI v{KEYWORD_ABI} or later (spec is ABI v{abi})"
            );
            return Ok(());
        }
        anyhow::ensure!(
            self.base_feature_abi
                .is_some_and(|base| (MIN_MODEL_ABI_VERSION..=METRIC_TRANSFORM_ABI).contains(&base)),
            "an ABI v{abi} spec requires base_feature_abi in {MIN_MODEL_ABI_VERSION}..={METRIC_TRANSFORM_ABI} (got {:?})",
            self.base_feature_abi,
        );
        for source in self.keyword_vocabs.keys() {
            anyhow::ensure!(
                abi >= source.min_abi(),
                "keyword source {} requires ABI v{} (spec is ABI v{abi})",
                source.name(),
                source.min_abi(),
            );
        }
        Ok(())
    }

    /// From ABI 21 every `metrics:` feature names its transform, and every
    /// metric getter is one this build computes.
    fn validate_metrics(&self) -> Result<()> {
        let Some(transforms) = &self.metric_transforms else {
            anyhow::ensure!(
                self.feature_semantics_abi() < METRIC_TRANSFORM_ABI,
                "an ABI v{METRIC_TRANSFORM_ABI} feature spec requires metric_transforms"
            );
            return Ok(());
        };
        for name in self
            .feature_names
            .iter()
            .filter_map(|n| n.strip_prefix("metrics:"))
        {
            anyhow::ensure!(
                matches!(
                    transforms.get(name).map(String::as_str),
                    Some("identity" | "log1p_abs")
                ),
                "missing or invalid metric transform: {name}"
            );
        }
        anyhow::ensure!(
            !self.metric_getters.iter().any(|g| g == "pe_year_distance")
                || (1970..=9999).contains(&self.metric_reference_year),
            "pe_year_distance requires its training reference year"
        );
        for getter in &self.metric_getters {
            anyhow::ensure!(
                known_metric_getter(getter)
                    || self
                        .metric_correlations
                        .get(getter)
                        .is_some_and(|pair| pair.iter().flatten().all(|s| !s.is_empty())),
                "unknown metric getter: {getter}"
            );
        }
        Ok(())
    }

    /// Standardization stats come as a pair, one entry per feature, and a
    /// standardized spec must carry them.
    fn validate_standardization(&self) -> Result<()> {
        match (
            self.feature_means.as_ref(),
            self.feature_stds.as_ref(),
            self.standardized,
        ) {
            (Some(means), Some(stds), _) => anyhow::ensure!(
                means.len() == self.total_features && stds.len() == self.total_features,
                "feature spec standardization stats must each have {} entries (got means={}, stds={})",
                self.total_features,
                means.len(),
                stds.len()
            ),
            (None, None, false) => {}
            (None, None, true) => anyhow::bail!(
                "feature spec is marked standardized but feature_means/feature_stds are missing"
            ),
            (Some(_), None, _) | (None, Some(_), _) => anyhow::bail!(
                "feature spec must include both feature_means and feature_stds together"
            ),
        }
        Ok(())
    }

    /// Log how the spec's layout differs from what this extractor emits.
    ///
    /// A spec may be a subset of the extractor's layout (feature groups
    /// disabled at training), and it may name features this build does not
    /// extract: those slots stay zero, as for a sample with no signal. A load
    /// warns and continues, so a model with a few unknown features costs some
    /// accuracy rather than the whole model; `atomscan validate` treats the
    /// same list as fatal (see [`Self::degraded_feature_names`]).
    fn note_unknown_features(&self) {
        let expected = self.expected_feature_names();
        if self.feature_names == expected {
            return;
        }
        let unknown = self.unknown_feature_names(&expected);
        if unknown.is_empty() {
            tracing::debug!(
                spec_features = self.feature_names.len(),
                extractor_features = expected.len(),
                absent_optional = expected.len().saturating_sub(self.feature_names.len()),
                "feature spec is a subset of the extractor's layout",
            );
        } else {
            tracing::warn!(
                spec_features = self.feature_names.len(),
                expected_max = expected.len(),
                unknown_count = unknown.len(),
                sample = ?unknown.iter().take(5).collect::<Vec<_>>(),
                "feature spec contains features unknown to this extractor — they will extract as zeros (model accuracy may degrade for those slots)"
            );
        }
    }

    /// Feature names this spec declares that the extractor built into this
    /// binary does not produce. `expected` is the extractor's full
    /// feature-name list. Each returned slot extracts to zero at runtime, so
    /// the model receives no signal where it expects some. Empty in the normal
    /// case (the spec is a clean subset of what the extractor knows).
    fn unknown_feature_names<'a>(&'a self, expected: &[String]) -> Vec<&'a str> {
        let expected_set: HashSet<&str> = expected.iter().map(String::as_str).collect();
        self.feature_names
            .iter()
            .map(String::as_str)
            .filter(|n| !expected_set.contains(*n))
            .collect()
    }

    /// Feature names this spec declares that this build's extractor cannot
    /// produce; each extracts to zero, silently degrading the model on those
    /// slots relative to its training. A non-empty result means the loaded
    /// model is degraded — `atomscan validate` treats this as fatal, while a
    /// normal scan only WARNs (the model still degrades gracefully). This is
    /// distinct from *absent optional* features (the spec being a strict subset
    /// of what the extractor knows, the normal case), which never appear here.
    #[must_use]
    pub fn degraded_feature_names(&self) -> Vec<String> {
        let expected_feature_names = self.expected_feature_names();
        self.unknown_feature_names(&expected_feature_names)
            .iter()
            .map(|name| (*name).to_string())
            .collect()
    }

    /// The full feature-name list this extractor emits for the spec's
    /// vocabularies — the canonical layout that `feature_names` is checked
    /// against. The fixed-name families come from the same tables that
    /// [`FeatureWriter::set`] checks its names against.
    fn expected_feature_names(&self) -> Vec<String> {
        fn vocab(names: &mut Vec<String>, prefix: &str, vocab: &[String]) {
            names.extend(vocab.iter().map(|v| format!("{prefix}{v}")));
        }
        fn fixed(names: &mut Vec<String>, fixed: &[&str]) {
            names.extend(fixed.iter().map(|n| (*n).to_string()));
        }
        let mut names = Vec::with_capacity(self.total_features.max(1024));
        vocab(&mut names, "present:", &self.presence_vocab);
        vocab(&mut names, "maxcrit:", &self.presence_vocab);
        fixed(&mut names, AGG_FEATURES);
        vocab(&mut names, "crit:", &self.crit_unigram_vocab);
        vocab(&mut names, "critbi:", &self.crit_bigram_vocab);
        vocab(&mut names, "crittri:", &self.crit_trigram_vocab);
        vocab(&mut names, "atkbi:", &self.attack_bigram_vocab);
        vocab(&mut names, "atktri:", &self.attack_trigram_vocab);
        vocab(&mut names, "mbcbi:", &self.mbc_bigram_vocab);
        vocab(&mut names, "mbctri:", &self.mbc_trigram_vocab);
        fixed(&mut names, EXT_FEATURES);
        names.extend(metric_base_feature_names());
        vocab(&mut names, "metrics:", &self.metric_vocab);
        vocab(&mut names, "filetype:", &self.filetype_vocab);
        names.extend(FORMAT_FEATURE_NAMES.iter().flatten().cloned());
        fixed(&mut names, FORMAT_SUMMARY_FEATURES);
        fixed(&mut names, STRUCT_FEATURES);
        vocab(&mut names, "elements:", &self.element_vocab);
        fixed(&mut names, FORMULA_FEATURES);
        fixed(&mut names, SCORE_FEATURES);
        names.extend(
            self.filetype_vocab
                .iter()
                .map(|ft| format!("inter:{ft}*score")),
        );
        vocab(&mut names, "bigrams:", &self.bigram_vocab);
        vocab(&mut names, "tierbi:", &self.tiered_bigram_vocab);
        vocab(&mut names, "tiertri:", &self.tiered_trigram_vocab);
        vocab(&mut names, "tierquad:", &self.tiered_quadgram_vocab);
        names.extend(
            self.feature_names
                .iter()
                .filter(|n| n.starts_with("mbc:"))
                .cloned(),
        );
        vocab(&mut names, "ghost:", &self.ghost_vocab);
        vocab(&mut names, "skeleton:", &self.skeleton_vocab);
        vocab(&mut names, "rare:", &self.rare_element_vocab);
        fixed(&mut names, STRUCT_EXTENSION_FEATURES);
        vocab(&mut names, "trigram:", &self.trigram_vocab);
        names.extend(GAP_FEATURE_NAMES.iter().cloned());
        vocab(&mut names, "unsigned_bigram:", &self.bigram_vocab);
        names.extend(INTENT_GAP_FEATURE_NAMES.iter().cloned());
        names.extend(MISSING_FEATURE_NAMES.iter().cloned());
        fixed(&mut names, TAIL_FEATURES);
        fixed(&mut names, TEXTENC_FEATURES);
        vocab(&mut names, "kv:", &self.kv_vocab);
        vocab(&mut names, "symbol:", &self.symbol_vocab);
        vocab(&mut names, "symbol_bi:", &self.symbol_bigram_vocab);
        vocab(&mut names, "symbol_tri:", &self.symbol_trigram_vocab);
        for source in KeywordSource::ALL {
            if let Some(words) = self.keyword_vocabs.get(&source) {
                vocab(&mut names, &format!("{}:", source.name()), words);
            }
        }
        names
    }
}

/// Logic gap categories (group 19).
pub(super) const LOGIC_GAP_CATEGORIES: &[&str] = &["crypto", "network", "process"];

/// Intent gap categories (group 22).
pub(super) const INTENT_GAP_CATEGORIES: &[&str] = &["network", "filesystem", "execution", "crypto"];
