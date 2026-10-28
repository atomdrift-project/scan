//! Feature extraction from cleave's compact report.
//!
//! Mirrors the feature extraction in collimator/src/collimator/features.py (v16)
//! exactly, using the same feature_spec.json vocabulary to produce identical
//! feature vectors.
//!
//! Feature assignment uses `FeatureWriter`, which maps feature names to the
//! slots of the spec's `feature_names` list. A name the spec does not carry
//! (a disabled group, or a vocabulary token the model never saw) writes
//! nothing — that is how pruned and partial specs "just work".

// All feature vectors use f32 to match the model's input dtype. The f64→f32
// narrowing throughout this file is intentional and safe: feature values are
// counts, ratios, or scores that fit well within f32 range.
#![expect(
    clippy::cast_possible_truncation,
    reason = "feature values are narrowed to the model's f32 input dtype"
)]

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

// An ensemble can load many specs of the same version, including specialists
// initialized concurrently during validation. Report the compatibility note
// once per version for the process, not once per route.
static WARNED_OLD_SPEC_VERSIONS: LazyLock<Mutex<HashSet<u32>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Name-based feature assignment: one route's feature vector plus the spec's
/// name → slot lookup.
struct FeatureWriter<'a> {
    vec: &'a mut [f32],
    lookup: &'a HashMap<String, usize>,
}

impl FeatureWriter<'_> {
    /// Set a fixed-name feature. The spec may have pruned it (then this is a
    /// no-op), but the name must be one the extractor declares in
    /// [`FIXED_FEATURE_NAMES`]: a misspelled name would otherwise never fire
    /// and never fail.
    #[inline]
    fn set(&mut self, name: &str, value: f32) {
        debug_assert!(
            FIXED_FEATURE_NAMES.contains(name),
            "feature {name:?} is not declared in the extractor's layout"
        );
        self.set_token(name, value);
    }

    /// Set a vocabulary-driven feature (`elements:`, `kv:`, n-grams, …). Most
    /// candidate tokens are not in the model's vocabulary, so a miss is the
    /// normal case.
    #[inline]
    fn set_token(&mut self, name: &str, value: f32) {
        if let Some(&idx) = self.lookup.get(name) {
            self.vec[idx] = value;
        }
    }

    /// [`Self::set_token`] for `<prefix><token>`, built in the reusable `key`.
    fn set_prefixed(&mut self, key: &mut String, prefix: &str, token: &str, value: f32) {
        key.clear();
        key.push_str(prefix);
        key.push_str(token);
        self.set_token(key, value);
    }

    /// Set a feature whose slot was resolved when the context was built.
    #[inline]
    fn set_slot(&mut self, slot: usize, value: f32) {
        if let Some(cell) = self.vec.get_mut(slot) {
            *cell = value;
        }
    }
}

/// Feature spec version this build was compiled against.
/// Must match the version in the loaded feature_spec.json.
///
/// v19 reads per-model trait_path_depth, preserving depth-3 defaults for older
/// bundles. v18 grows the extended-metrics block by `ast_depth_capped` (collimator's
/// ALWAYS_KEEP_METRICS allowlist keeps that rare anti-analysis signal past the
/// frequency prune). The metric is read by name from the spec vocab, so older
/// v16/v17 bundles still load — the extractor just doesn't place a slot they
/// don't list.
// v20 retains strongest tiered tokens; older models keep overflow-drop semantics.
pub const EXPECTED_SPEC_VERSION: u32 = 21;

/// Earliest spec version this build can still load. v16/v17 specs are
/// loadable too: each newer version added feature families/columns the
/// extractor produces, but the loader extracts known features and zeros out
/// unknowns, so older bundles still work end-to-end.
pub const MIN_LOADABLE_SPEC_VERSION: u32 = 16;
/// Stable model ABI version shared with collimator.
/// Keep this in sync with EXPECTED_SPEC_VERSION for a single compatibility number.
pub const EXPECTED_MODEL_ABI_VERSION: u32 = EXPECTED_SPEC_VERSION;

/// Minimum finding confidence for inclusion (matches collimator MIN_CONFIDENCE).
const MIN_CONFIDENCE: f64 = 0.65;

/// Criticality ordinals as a compact report carries them
/// ([`cleave::Criticality::rank`]).
const CRIT_BASELINE: u32 = 2;
const CRIT_NOTABLE: u32 = 3;
const CRIT_SUSPICIOUS: u32 = 4;
const CRIT_HOSTILE: u32 = 5;

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
    metric_transforms: Option<HashMap<String, String>>,
    metric_getters: Vec<String>,
    metric_reference_year: u32,
    metric_correlations: HashMap<String, [[String; 2]; 2]>,
    crit_unigram_vocab: Vec<String>,
    crit_bigram_vocab: Vec<String>,
    crit_trigram_vocab: Vec<String>,
    attack_bigram_vocab: Vec<String>,
    attack_trigram_vocab: Vec<String>,
    mbc_bigram_vocab: Vec<String>,
    mbc_trigram_vocab: Vec<String>,
    tiered_bigram_vocab: Vec<String>,
    tiered_trigram_vocab: Vec<String>,
    tiered_quadgram_vocab: Vec<String>,
    kv_vocab: Vec<String>,
    symbol_vocab: Vec<String>,
    symbol_bigram_vocab: Vec<String>,
    symbol_trigram_vocab: Vec<String>,
    feature_names: Vec<String>,
    total_features: usize,
    feature_means: Option<Vec<f32>>,
    feature_stds: Option<Vec<f32>>,
    standardized: bool,
    trait_path_depth: usize,
    tiered_min_crit: [u32; 3],
    tiered_branch_min_crit: HashMap<String, u32>,
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
    metric_transforms: Option<HashMap<String, String>>,
    #[serde(default)]
    metric_getters: Vec<String>,
    #[serde(default)]
    metric_reference_year: u32,
    #[serde(default)]
    metric_correlations: HashMap<String, [[String; 2]; 2]>,
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
    tiered_quadgram_vocab: Vec<String>,
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
    #[serde(default = "default_trait_path_depth")]
    trait_path_depth: usize,
    #[serde(default = "default_tiered_min_crit")]
    tiered_min_crit: [u32; 3],
    #[serde(default)]
    tiered_branch_min_crit: HashMap<String, u32>,
}

const fn default_tiered_min_crit() -> [u32; 3] {
    [3; 3]
}

fn default_trait_path_depth() -> usize {
    3
}

const fn default_standardized() -> bool {
    true
}

const fn default_abi_version() -> u32 {
    0 // An omitted ABI inherits the spec version, never the serving version.
}

impl FeatureSpec {
    /// Load feature specification from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).context("reading feature spec")?;
        let raw: RawFeatureSpec = serde_json::from_str(&data).context("parsing feature spec")?;

        if raw.version < MIN_LOADABLE_SPEC_VERSION || raw.version > EXPECTED_SPEC_VERSION {
            anyhow::bail!(
                "feature spec version mismatch: this installed model uses spec v{}, but this Atomdrift Scan build accepts v{MIN_LOADABLE_SPEC_VERSION}..=v{EXPECTED_SPEC_VERSION}. \
                 The model is incompatible with this build. Run 'atomscan update-rules' to install a matching model bundle.",
                raw.version,
            );
        }
        if raw.version < EXPECTED_SPEC_VERSION
            && WARNED_OLD_SPEC_VERSIONS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(raw.version)
        {
            tracing::warn!(
                version = raw.version,
                expected = EXPECTED_SPEC_VERSION,
                "loading older feature spec; features added since this version are absent from the model; retrain at v{EXPECTED_SPEC_VERSION} to use them"
            );
        }

        let spec = Self {
            version: raw.version,
            abi_version: if raw.abi_version == 0 {
                raw.version
            } else {
                raw.abi_version
            },
            presence_vocab: raw.presence_vocab,
            filetype_vocab: raw.filetype_vocab,
            element_vocab: raw.element_vocab,
            bigram_vocab: raw.bigram_vocab,
            ghost_vocab: raw.ghost_vocab,
            skeleton_vocab: raw.skeleton_vocab,
            rare_element_vocab: raw.rare_element_vocab,
            trigram_vocab: raw.trigram_vocab,
            metric_vocab: raw.metric_vocab,
            metric_transforms: raw.metric_transforms,
            metric_getters: raw.metric_getters,
            metric_reference_year: raw.metric_reference_year,
            metric_correlations: raw.metric_correlations,
            crit_unigram_vocab: raw.crit_unigram_vocab,
            crit_bigram_vocab: raw.crit_bigram_vocab,
            crit_trigram_vocab: raw.crit_trigram_vocab,
            attack_bigram_vocab: raw.attack_bigram_vocab,
            attack_trigram_vocab: raw.attack_trigram_vocab,
            mbc_bigram_vocab: raw.mbc_bigram_vocab,
            mbc_trigram_vocab: raw.mbc_trigram_vocab,
            tiered_bigram_vocab: raw.tiered_bigram_vocab,
            tiered_trigram_vocab: raw.tiered_trigram_vocab,
            tiered_quadgram_vocab: raw.tiered_quadgram_vocab,
            kv_vocab: raw.kv_vocab,
            symbol_vocab: raw.symbol_vocab,
            symbol_bigram_vocab: raw.symbol_bigram_vocab,
            symbol_trigram_vocab: raw.symbol_trigram_vocab,
            feature_names: raw.feature_names,
            total_features: raw.total_features,
            feature_means: raw.feature_means,
            feature_stds: raw.feature_stds,
            standardized: raw.standardized,
            trait_path_depth: raw.trait_path_depth,
            tiered_min_crit: raw.tiered_min_crit,
            tiered_branch_min_crit: raw.tiered_branch_min_crit,
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

    /// Apply z-score standardization using training statistics.
    ///
    /// Mirrors collimator's `standardize`: a feature whose training stats are
    /// exactly `(mean 0, std 1)` was constant during training, and collimator
    /// zeroes it rather than pass through a raw value the model never saw. A
    /// zero std is zeroed too, instead of dividing by it.
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
        anyhow::ensure!(
            self.trait_path_depth > 0,
            "trait_path_depth must be positive"
        );
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
        anyhow::ensure!(
            self.abi_version < 21 || self.metric_transforms.is_some(),
            "v21 feature spec requires metric_transforms"
        );
        if let Some(transforms) = &self.metric_transforms {
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
        }

        if !(18..=EXPECTED_MODEL_ABI_VERSION).contains(&self.abi_version) {
            anyhow::bail!(
                "feature spec ABI mismatch: spec has ABI v{} but Atomdrift Scan requires ABI v{}",
                self.abi_version,
                EXPECTED_MODEL_ABI_VERSION
            );
        }

        let expected_feature_names = self.expected_feature_names();
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
            let unknown_features = self.unknown_feature_names(&expected_feature_names);
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
                    expected_feature_names
                        .len()
                        .saturating_sub(self.feature_names.len()),
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
                anyhow::bail!(
                    "feature spec is marked standardized but feature_means/feature_stds are missing"
                );
            }
            (Some(_), None, _) | (None, Some(_), _) => {
                anyhow::bail!(
                    "feature spec must include both feature_means and feature_stds together"
                );
            }
        }

        Ok(())
    }

    /// Feature names this spec declares that the extractor built into this
    /// binary does not produce. `expected` is the extractor's full
    /// feature-name list. Each returned slot extracts to zero at runtime, so
    /// the model receives no signal where it expects some. Empty in the normal
    /// case (the spec is a clean subset of what the extractor knows).
    fn unknown_feature_names<'a>(&'a self, expected: &[String]) -> Vec<&'a str> {
        let expected_set: std::collections::HashSet<&str> =
            expected.iter().map(String::as_str).collect();
        self.feature_names
            .iter()
            .map(String::as_str)
            .filter(|n| !expected_set.contains(*n))
            .collect()
    }

    /// Feature names this spec declares that this build's extractor cannot
    /// produce; each extracts to zero, silently degrading the model on those
    /// slots relative to its training. A non-empty result means the loaded
    /// model is degraded — `litmus validate` treats this as fatal, while a
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
        names
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

/// Group 3: report-level aggregates.
const AGG_FEATURES: &[&str] = &[
    "agg:max_crit",
    "agg:category_breadth",
    "agg:path_breadth_any",
    "agg:total_active_paths",
    "agg:suspicious_concentration",
    "agg:hostile_concentration",
    "agg:escalation_rate",
    "agg:notable_only_fraction",
    "agg:notable_findings_log",
    "agg:suspicious_findings_log",
    "agg:hostile_findings_log",
    "agg:notable_finding_ratio",
    "agg:suspicious_finding_ratio",
    "agg:hostile_finding_ratio",
    "agg:unique_suspicious_ids_log",
    "agg:unique_hostile_ids_log",
    // The `top1` block summarizes the single riskiest file; see
    // `topk_file_risk_features_from_summaries`.
    "agg:top1_file_suspicious_ratio_sum",
    "agg:top1_file_hostile_ratio_sum",
    "agg:top1_file_suspicious_findings_log",
    "agg:top1_file_hostile_findings_log",
    "agg:suspicious_category_breadth",
    "agg:hostile_category_breadth",
    "agg:suspicious_category_density",
    "agg:hostile_category_density",
    "agg:suspicious_findings_per_kb",
    "agg:hostile_findings_per_kb",
    "agg:suspicious_categories_per_kb",
    "agg:hostile_categories_per_kb",
    "agg:top1_file_suspicious_density_sum",
    "agg:top1_file_hostile_density_sum",
    "agg:top1_file_suspicious_category_breadth_sum",
    "agg:top1_file_hostile_category_breadth_sum",
    // Size-invariant crit-tier severity fractions (collimator's
    // include_severity_fractions group).
    "agg:confidence_mean",
    "agg:confidence_std",
    "agg:confidence_skew",
    "agg:confidence_kurtosis",
    "agg:trait_id_lexical_distance",
    "agg:docobf_obfuscation_count",
    "agg:docobf_eval_count",
    "agg:docobf_lure_count",
    "agg:docobf_total_count",
    "agg:docobf_has_any",
    "agg:crit3_finding_fraction",
    "agg:crit4_finding_fraction",
    "agg:hostile_finding_fraction",
    "agg:severe_to_mundane_ratio",
    "agg:crit4_present",
    "agg:hostile_escalation_rate",
    "agg:hostile_share_of_suspicious",
    "agg:suspicious_finding_escalation_rate",
    "agg:hostile_finding_escalation_rate",
    "agg:hostile_share_of_suspicious_findings",
    "agg:hostile_weighted_density",
    "agg:top1_file_hostile_weighted_density_sum",
    "agg:suspicious_id_repeat_ratio",
    "agg:hostile_id_repeat_ratio",
    "agg:suspicious_category_repeat_ratio",
    "agg:hostile_category_repeat_ratio",
    "agg:file_hostile_fraction",
    "agg:file_suspicious_fraction",
    "agg:file_notable_fraction",
    "agg:file_hostile_count_log",
    "agg:file_suspicious_count_log",
    "agg:file_notable_count_log",
    "agg:hostile_depth_weight",
    "agg:suspicious_2level_breadth",
    "agg:hostile_2level_breadth",
    "agg:objectives_breadth",
    // From here to `static_signed_file_fraction`, collimator computes these
    // and litmus does not: they are declared so a spec carrying them is not
    // reported as degraded, and extract as zero.
    "agg:kill_chain_span",
    "agg:objective_micro_ratio",
    "agg:avg_finding_depth",
    "agg:objective_hostile_density",
    "agg:static_file_bytes_log",
    "agg:static_import_count_log",
    "agg:static_export_count_log",
    "agg:static_dependency_count_log",
    "agg:static_string_count_log",
    "agg:static_wide_string_ratio",
    "agg:static_max_string_length_log",
    "agg:static_string_entropy_max",
    "agg:static_text_lines_log",
    "agg:static_function_count_log",
    "agg:static_code_bytes_log",
    "agg:static_code_to_data_ratio_max",
    "agg:static_wx_units_log",
    "agg:static_writable_unit_ratio",
    "agg:static_executable_unit_ratio",
    "agg:static_nonstandard_unit_names_log",
    "agg:static_largest_unit_ratio_max",
    "agg:static_resource_ratio_max",
    "agg:static_signed_file_fraction",
    "agg:attack_technique_count",
    "agg:attack_tactic_count",
    "agg:mbc_behavior_count",
    "agg:has_attack_and_objective",
    // ATT&CK / MBC co-occurrence aggregates (log1p of unordered combinations
    // among distinct technique / behavior codes seen in raw_findings).
    "agg:attack_bigram_count",
    "agg:attack_trigram_count",
    "agg:mbc_bigram_count",
    // Objective path co-occurrence aggregates (log1p of unordered combinations
    // among distinct `objectives/*` and `well-known/*` paths seen in
    // sample_paths). Trigram is bounded with a per-pair cap of 20 inner
    // elements to avoid O(n^3) explosion on samples with many objectives.
    "agg:objective_bigram_count",
    "agg:objective_trigram_count",
];

/// Group 4: external-signal summary.
const EXT_FEATURES: &[&str] = &[
    "ext:third_party_max_crit",
    "ext:third_party_count",
    "ext:well_known_max_crit",
    "ext:well_known_hostile_count",
    "ext:well_known_suspicious_count",
    "ext:has_yara_match",
];

/// Group 6b: portable format-group hints, after the per-group block.
const FORMAT_SUMMARY_FEATURES: &[&str] = &[
    "format:group_count_log",
    "format:mixed_script_binary",
    "format:mixed_archive_script",
    "format:mixed_archive_binary",
    "format:unknown_file_fraction",
];

/// Group 7: structure.
const STRUCT_FEATURES: &[&str] = &[
    "struct:tiny_executable",
    "struct:no_imports",
    "struct:zero_findings",
    "struct:finding_count_log",
    "struct:file_count_log",
    "struct:inner_file_count_log",
    "struct:stealth_potential",
    "struct:suspicious_file_fraction",
    "struct:hostile_file_fraction",
    "struct:suspicious_file_count_log",
    "struct:hostile_file_count_log",
];

/// Group 9: formula.
const FORMULA_FEATURES: &[&str] = &[
    "formula:skeleton_len",
    "formula:unique_elements",
    "formula:complexity_ratio",
];

/// Group 10: score (the `inter:` family follows it).
const SCORE_FEATURES: &[&str] = &["score:hopper_score", "score:density"];

/// Group 15: structural extensions.
const STRUCT_EXTENSION_FEATURES: &[&str] = &[
    "struct:packaged_capability",
    "struct:mtime_range_hours",
    "struct:mtime_std_dev_hours",
    "struct:max_nesting_depth_log",
    "struct:inner_file_ratio",
    "struct:entropy_std_dev",
    "struct:entropy_max_diff",
    "struct:air_gap_signal",
    "struct:anachronistic_injection",
    "struct:code_entropy_spike",
    "struct:foreign_binary_signal",
    "struct:extension_mismatch_signal",
    "struct:hostile_finding_density",
];

/// Features only some route specs carry (e.g. filetypes/makefile): suspicious
/// co-occurrence counts, cross-metric ratios, and the silent-packer signal.
const TAIL_FEATURES: &[&str] = &[
    "agg:suspicious_bigram_count",
    "agg:suspicious_trigram_count",
    "metrics:derived_string_per_function",
    "metrics:derived_imports_per_dependency",
    "metrics:derived_wide_string_ratio",
    "struct:silent_packer_signal",
];

/// Group 24: textenc — collimator's `_apply_text_encoding_features` ratios
/// over a file's strings. The compact report carries no strings, so these
/// always extract as zero; they are declared so a spec carrying them is not
/// reported as degraded.
const TEXTENC_FEATURES: &[&str] = &[
    "textenc:string_count_log",
    "textenc:avg_len_log",
    "textenc:max_len_log",
    "textenc:base64ish_ratio",
    "textenc:hexish_ratio",
    "textenc:urlish_ratio",
    "textenc:pathish_ratio",
    "textenc:unicode_escape_ratio",
    "textenc:wide_ratio",
    "textenc:high_entropy_ratio",
    "textenc:long_token_ratio",
    "textenc:short_junk_ratio",
];

/// Group 5: `metrics:<group>_<field>` for every [`KEY_METRICS`] entry.
fn metric_base_feature_names() -> impl Iterator<Item = String> {
    KEY_METRICS
        .iter()
        .map(|&(group, field, _)| format!("metrics:{group}_{field}"))
}

/// Group 6b: `format:<group>` and its four fractions, per [`FORMAT_GROUPS`] entry.
static FORMAT_FEATURE_NAMES: LazyLock<Vec<[String; 5]>> = LazyLock::new(|| {
    FORMAT_GROUPS
        .iter()
        .map(|&(group, _)| {
            [
                format!("format:{group}"),
                format!("format:{group}_file_fraction"),
                format!("format:{group}_inner_fraction"),
                format!("format:{group}_suspicious_fraction"),
                format!("format:{group}_hostile_fraction"),
            ]
        })
        .collect()
});

/// Group 19: `gap:<category>`, per [`LOGIC_GAP_CATEGORIES`] entry.
static GAP_FEATURE_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    LOGIC_GAP_CATEGORIES
        .iter()
        .map(|cat| format!("gap:{cat}"))
        .collect()
});

/// Group 22: `intent_gap:<category>`, per [`INTENT_GAP_CATEGORIES`] entry.
static INTENT_GAP_FEATURE_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    INTENT_GAP_CATEGORIES
        .iter()
        .map(|cat| format!("intent_gap:{cat}"))
        .collect()
});

/// Group 23: `missing:<type>*<trait>`, in [`EXPECTED_GHOSTS`] order.
static MISSING_FEATURE_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    EXPECTED_GHOSTS
        .iter()
        .flat_map(|&(ftype, traits)| traits.iter().map(move |t| format!("missing:{ftype}*{t}")))
        .collect()
});

/// Every fixed-name feature: the layout minus its vocabulary-driven families.
/// The single list [`FeatureWriter::set`] checks names against.
static FIXED_FEATURE_NAMES: LazyLock<HashSet<String>> = LazyLock::new(|| {
    [
        AGG_FEATURES,
        EXT_FEATURES,
        FORMAT_SUMMARY_FEATURES,
        STRUCT_FEATURES,
        FORMULA_FEATURES,
        SCORE_FEATURES,
        STRUCT_EXTENSION_FEATURES,
        TAIL_FEATURES,
        TEXTENC_FEATURES,
    ]
    .concat()
    .into_iter()
    .map(str::to_string)
    .chain(metric_base_feature_names())
    .chain(FORMAT_FEATURE_NAMES.iter().flatten().cloned())
    .chain(GAP_FEATURE_NAMES.iter().cloned())
    .chain(INTENT_GAP_FEATURE_NAMES.iter().cloned())
    .chain(MISSING_FEATURE_NAMES.iter().cloned())
    .collect()
});

/// One `metrics:` feature resolved to its slot: the value at
/// `metrics[group][field]`, optionally `ln(|v| + 1)`-scaled.
#[derive(Debug)]
struct MetricSlot {
    slot: usize,
    group: String,
    field: String,
    log: bool,
    getter: Option<String>,
    correlation: Option<[[String; 2]; 2]>,
}

/// Pre-built lookup tables for fast repeated extraction against a spec.
#[derive(Debug)]
pub struct ExtractContext {
    ranked_ngrams: bool,
    relative_structure: bool,
    explicit_metrics: bool,
    metric_reference_year: u32,
    trait_path_depth: usize,
    tiered_min_crit: [u32; 3],
    tiered_branch_min_crit: HashMap<String, u32>,
    presence_lookup: HashMap<String, usize>,
    /// Global feature name → index lookup for name-written features.
    absolute_lookup: HashMap<String, usize>,
    /// Group 5: the base [`KEY_METRICS`] then the spec's extended metric
    /// vocab, each resolved to its slot (absent ones dropped).
    metric_slots: Vec<MetricSlot>,
    /// Group 12: `ghost:` vocab paths that have a slot in this spec.
    ghost_slots: Vec<(String, usize)>,
    total_features: usize,
    /// Optional families this spec carries; their inputs are only gathered
    /// (and their writers only run) when a route needs them.
    kv: bool,
    kv_shape: bool,
    kv_split: bool,
    /// Any of the `symbol:`, `symbol_bi:` and `symbol_tri:` families.
    symbols: bool,
    symbol_bigrams: bool,
    symbol_trigrams: bool,
    suspicious_ngram_counts: bool,
    optional_finding_features: bool,
    tiered_slots: Vec<(usize, Vec<String>)>,

    // Optimized bigram/trigram lookups
    path_to_id: HashMap<String, u32>,
    bigram_id_lookup: HashMap<(u32, u32), usize>,
    trigram_id_lookup: HashMap<(u32, u32, u32), usize>,

    // Per-family map from vocab index -> the real `feature_names` slot for that
    // member, resolved by NAME in `new()`. `None` at a position means that vocab
    // entry has no slot in this spec (pruned) and is skipped. Indexed by the same
    // vocab index the runtime lookups (`presence_lookup`, `bigram_id_lookup`,
    // `trigram_id_lookup`) produce, so a family that is a SUBSET of its (possibly
    // shared) vocab still writes each present member to its true slot — no
    // contiguous `base + idx` block required. See `new()`.
    present_slots: Vec<Option<usize>>,
    maxcrit_slots: Vec<Option<usize>>,
    bigram_slots: Vec<Option<usize>>,
    trigram_slots: Vec<Option<usize>>,
    unsigned_bigram_slots: Vec<Option<usize>>,
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

        let absolute_lookup: HashMap<String, usize> = spec
            .feature_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        // Resolve each offset-written family (presence, max-crit, bigram, trigram,
        // unsigned-bigram) to the real `feature_names` slot of every vocab member,
        // by NAME, rather than assuming a contiguous `base + idx` run. This is
        // robust to any model layout the spec presents:
        //   * fully absent  — the family was pruned/disabled in the spec; every
        //     entry maps to `None` and the writer skips it (model trained without
        //     it — correct, not an error).
        //   * a SUBSET of its (often shared) vocab — e.g. an allowlist prune drops
        //     some `maxcrit:`/`unsigned_bigram:` names while keeping the
        //     `present:`/`bigrams:` ones that share the same vocab. The dropped
        //     entries map to `None`; the present ones each map to their true slot,
        //     so writes never land on the wrong feature. No 1:1 contiguous block
        //     is required, so litmus accepts pruned/reordered bundles instead of
        //     rejecting them.
        let family_slots = |prefix: &str, vocab: &[String]| -> Vec<Option<usize>> {
            vocab
                .iter()
                .map(|v| absolute_lookup.get(&format!("{prefix}{v}")).copied())
                .collect()
        };
        let present_slots = family_slots("present:", &spec.presence_vocab);
        let maxcrit_slots = family_slots("maxcrit:", &spec.presence_vocab);
        let bigram_slots = family_slots("bigrams:", &spec.bigram_vocab);
        let trigram_slots = family_slots("trigram:", &spec.trigram_vocab);
        let unsigned_bigram_slots = family_slots("unsigned_bigram:", &spec.bigram_vocab);

        let has_prefix = |prefix: &str| spec.feature_names.iter().any(|n| n.starts_with(prefix));
        let has_name = |name: &str| absolute_lookup.contains_key(name);

        Self {
            trait_path_depth: spec.trait_path_depth,
            ranked_ngrams: spec.abi_version >= 20,
            relative_structure: spec.abi_version >= 21,
            explicit_metrics: spec.metric_transforms.is_some(),
            metric_reference_year: spec.metric_reference_year,
            tiered_min_crit: spec.tiered_min_crit,
            tiered_branch_min_crit: spec.tiered_branch_min_crit.clone(),
            metric_slots: metric_slots(&absolute_lookup, spec),
            ghost_slots: spec
                .ghost_vocab
                .iter()
                .filter_map(|path| {
                    let slot = *absolute_lookup.get(&format!("ghost:{path}"))?;
                    Some((path.clone(), slot))
                })
                .collect(),
            total_features: spec.total_features,
            kv: has_prefix("kv:"),
            kv_shape: spec
                .kv_vocab
                .iter()
                .any(|v| v.split('=').next().is_some_and(|key| key.contains(':'))),
            kv_split: spec.kv_vocab.iter().any(|v| v.contains("=part:")),
            symbols: has_prefix("symbol:") || has_prefix("symbol_bi:") || has_prefix("symbol_tri:"),
            symbol_bigrams: has_prefix("symbol_bi:"),
            symbol_trigrams: has_prefix("symbol_tri:"),
            tiered_slots: spec
                .feature_names
                .iter()
                .enumerate()
                .filter_map(|(slot, name)| {
                    let (family, value) = name.split_once(':')?;
                    ["tierbi", "tiertri", "tierquad"]
                        .contains(&family)
                        .then(|| (slot, value.split(" + ").map(str::to_owned).collect()))
                })
                .collect(),
            optional_finding_features: has_prefix("mbc:")
                || has_prefix("agg:docobf_")
                || [
                    "agg:confidence_mean",
                    "agg:confidence_std",
                    "agg:confidence_skew",
                    "agg:confidence_kurtosis",
                    "agg:trait_id_lexical_distance",
                ]
                .iter()
                .any(|name| has_name(name)),
            suspicious_ngram_counts: has_name("agg:suspicious_bigram_count")
                || has_name("agg:suspicious_trigram_count"),
            presence_lookup,
            absolute_lookup,
            path_to_id,
            bigram_id_lookup,
            trigram_id_lookup,
            present_slots,
            maxcrit_slots,
            bigram_slots,
            trigram_slots,
            unsigned_bigram_slots,
        }
    }

    /// Extract this route's feature vector from a cleave compact report.
    #[must_use]
    pub fn extract(&self, report: &cleave::types::CompactReport) -> Vec<f32> {
        let parsed = ParsedReport::from_compact_report(report, self.raw_needs(), None);
        self.extract_from_parsed(&parsed)
    }

    /// This route's active raw-subtree needs (which optional families it emits).
    pub(crate) const fn raw_needs(&self) -> RawNeeds {
        RawNeeds {
            kv: self.kv,
            kv_shape: self.kv_shape,
            kv_split: self.kv_split,
            symbol: self.symbols,
        }
    }

    /// Allocate and fill this route's feature vector from a shared [`ParsedReport`].
    pub(crate) fn extract_from_parsed(&self, parsed: &ParsedReport) -> Vec<f32> {
        if parsed.trait_path_depth != self.trait_path_depth {
            let adjusted = parsed.at_trait_path_depth(self.trait_path_depth);
            return self.extract_from_parsed(&adjusted);
        }
        let mut vec = vec![0.0f32; self.total_features];
        self.write_features(parsed, &mut vec);
        vec
    }

    /// Emit this route's features into `vec` from an already-parsed report.
    /// The parse is route-independent (see [`ParsedReport`]); only the writes
    /// below depend on this context's vocab/slots. Families are written in
    /// layout order.
    fn write_features(&self, parsed: &ParsedReport, vec: &mut [f32]) {
        let summaries = parsed.summaries.as_slice();
        let combined = &parsed.combined;
        let w = &mut FeatureWriter {
            vec,
            lookup: &self.absolute_lookup,
        };

        self.write_path_features(combined, parsed.sample_score, w); // G1, G2
        write_aggregate_features(parsed, w); // G3
        if self.optional_finding_features {
            write_optional_finding_features(parsed, w);
        }
        write_crit_ngrams(&parsed.crit_tokens, w);
        write_ngrams(w, &parsed.attack_codes, "atkbi:", "atktri:", "ATT&CK");
        write_ngrams(w, &parsed.mbc_codes, "mbcbi:", "mbctri:", "MBC");
        write_external_summary_features(combined, w); // G4
        self.write_metric_features(&parsed.merged_metrics, summaries, w); // G5
        // G6 (filetype one-hots) is blindfolded since v16.
        write_format_hint_features(summaries, w); // G6b
        write_structural_features(w, summaries, combined.filtered_finding_count); // G7
        self.write_formula_features(parsed, w); // G8–G10, G12–G14
        self.write_bigram_features(summaries, w, &self.bigram_slots); // G11
        for (index, order) in [2, 3, 4].into_iter().enumerate() {
            if !self
                .tiered_slots
                .iter()
                .any(|(_, parts)| parts.len() == order)
            {
                continue;
            }
            let tokens = if self.ranked_ngrams {
                std::borrow::Cow::Owned(ranked_tiered_path_tokens(
                    combined,
                    self.tiered_min_crit[index],
                    &self.tiered_branch_min_crit,
                    order,
                ))
            } else if self.tiered_min_crit[index] == 3 && self.tiered_branch_min_crit.is_empty() {
                std::borrow::Cow::Borrowed(parsed.tiered_tokens.as_slice())
            } else {
                std::borrow::Cow::Owned(tiered_path_tokens_with_options(
                    combined,
                    self.tiered_min_crit[index],
                    &self.tiered_branch_min_crit,
                ))
            };
            if tokens.len() > if order == 4 { 64 } else { 512 } {
                continue;
            }
            for (slot, parts) in &self.tiered_slots {
                if parts.len() == order
                    && parts.iter().all(|part| tokens.binary_search(part).is_ok())
                {
                    w.set_slot(*slot, 1.0);
                }
            }
        }
        write_structural_extensions(summaries, combined, self.relative_structure, w); // G15
        self.write_trigram_features(summaries, w); // G16
        write_logic_gap_features(combined, summaries, w); // G19
        // G20: signature synergy — the bigram block again, for unsigned samples.
        if combined.sample_paths.contains_key("metadata/unsigned") {
            self.write_bigram_features(summaries, w, &self.unsigned_bigram_slots);
        }
        write_intent_gap_features(combined, w); // G22
        write_negative_space_features(combined, summaries, w); // G23

        // Families only some route specs carry. G24 (textenc) has no input in
        // the compact report and always extracts as zero.
        if self.suspicious_ngram_counts {
            write_suspicious_ngram_counts(combined, w);
        }
        if !self.explicit_metrics {
            write_derived_metric_features(&parsed.merged_metrics, w);
        }
        write_silent_packer_signal(summaries, combined.filtered_finding_count, w);
        if self.kv {
            for s in summaries {
                for token in &s.kv_tokens {
                    w.set_token(token, 1.0);
                }
            }
        }
        if self.symbols {
            write_symbol_features(summaries, w, self.symbol_bigrams, self.symbol_trigrams);
        }
    }

    /// G1 presence and G2 max-criticality, per sample path, weighted by the
    /// sample's risk score and the path's best confidence.
    fn write_path_features(
        &self,
        summary: &FindingSummary,
        sample_score: i64,
        w: &mut FeatureWriter<'_>,
    ) {
        let score_weight: f64 = if sample_score > 0 {
            (sample_score as f64).ln_1p()
        } else {
            1.0
        };
        for (path, &max_ord) in &summary.sample_paths {
            if max_ord >= CRIT_BASELINE
                && let Some(&idx) = self.presence_lookup.get(path.as_str())
                && let Some(&Some(slot)) = self.present_slots.get(idx)
            {
                let conf = summary.path_confidences.get(path).copied().unwrap_or(1.0);
                w.set_slot(slot, (score_weight * conf) as f32);
            }
        }
        for (path, &max_ord) in &summary.sample_paths {
            if let Some(&idx) = self.presence_lookup.get(path.as_str())
                && let Some(&Some(slot)) = self.maxcrit_slots.get(idx)
            {
                let conf = summary.path_confidences.get(path).copied().unwrap_or(1.0);
                w.set_slot(slot, (f64::from(max_ord) * score_weight * conf) as f32);
            }
        }
    }

    /// G5: the base [`KEY_METRICS`] then the spec's extended metric vocab.
    fn write_metric_features(
        &self,
        metrics: &MetricMap,
        summaries: &[FileSummary],
        w: &mut FeatureWriter<'_>,
    ) {
        for m in &self.metric_slots {
            let value = if let Some([[ga, ka], [gb, kb]]) = &m.correlation {
                let get = |g: &String, k: &String| {
                    metrics
                        .get(g)
                        .and_then(|fields| fields.get(k))
                        .copied()
                        .unwrap_or(0.0)
                };
                get(ga, ka) * get(gb, kb)
            } else if let Some(getter) = &m.getter {
                derived_metric(getter, metrics, summaries, self.metric_reference_year)
            } else {
                metrics
                    .get(&m.group)
                    .and_then(|g| g.get(&m.field))
                    .copied()
                    .unwrap_or(0.0)
            };
            // Legacy models retain their f32-before-log contract.
            let output = if self.explicit_metrics {
                (if m.log { value.abs().ln_1p() } else { value }) as f32
            } else {
                let value = value as f32;
                if m.log {
                    (value.abs() + 1.0).ln()
                } else {
                    value
                }
            };
            w.set_slot(m.slot, output);
        }
    }

    /// G8 elements, G9 formula, G10 score, G12 ghost, G13 skeleton and G14
    /// rare elements: the families read off the primary file's formula and
    /// risk score.
    fn write_formula_features(&self, parsed: &ParsedReport, w: &mut FeatureWriter<'_>) {
        let combined = &parsed.combined;
        let formula_str = parsed.formula_str.as_str();
        let elements_str = parsed.elements_str.as_str();
        let sample_score = parsed.sample_score;
        let mut key = String::new();

        // G8: Elements
        if !elements_str.is_empty() {
            for el in elements_str.split(',') {
                w.set_prefixed(&mut key, "elements:", el.trim(), 1.0);
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
        let total_size_bytes: f64 = parsed.summaries.iter().map(|s| s.size_bytes).sum();
        w.set("score:hopper_score", sample_score as f32);
        w.set(
            "score:density",
            if total_size_bytes > 0.0 {
                sample_score as f32 / (total_size_bytes as f32).ln_1p()
            } else {
                0.0
            },
        );
        for s in &parsed.summaries {
            key.clear();
            let _ = write!(key, "inter:{}*score", s.file_type);
            w.set_token(&key, sample_score as f32);
        }

        // G12: Ghost — an expected path the sample never reached at baseline.
        for (ghost_path, slot) in &self.ghost_slots {
            let missing = combined
                .sample_paths
                .get(ghost_path)
                .is_none_or(|&max_ord| max_ord < CRIT_BASELINE);
            if missing {
                w.set_slot(*slot, 1.0);
            }
        }

        // G13: Skeleton
        if !skeleton_str.is_empty() {
            w.set_prefixed(&mut key, "skeleton:", &skeleton_str, 1.0);
        }

        // G14: Rare Elements, weighted by the mean finding confidence.
        if !elements_str.is_empty() {
            let weight: f32 = if combined.finding_confidences.is_empty() {
                1.0
            } else {
                let sum: f64 = combined.finding_confidences.iter().sum();
                (sum / combined.finding_confidences.len() as f64) as f32
            };
            for el in elements_str.split(',') {
                w.set_prefixed(&mut key, "rare:", el.trim(), weight);
            }
        }
    }

    /// Path-pair features: a vocab bigram is set when both of its 3-level
    /// paths fired in the same file. `slots` picks the family (`bigrams:` or
    /// `unsigned_bigram:`), which share one vocab.
    fn write_bigram_features(
        &self,
        summaries: &[FileSummary],
        w: &mut FeatureWriter<'_>,
        slots: &[Option<usize>],
    ) {
        for s in summaries {
            let ranked;
            let paths = if self.ranked_ngrams {
                ranked = ranked_ngram_paths(&s.raw_findings, 512);
                &ranked
            } else {
                &s.unique_3level_paths
            };
            let ids: Vec<u32> = paths
                .iter()
                .filter_map(|p| self.path_to_id.get(p).copied())
                .collect();
            if ids.len() > 512 {
                tracing::warn!(path = %s.path, tokens = ids.len(), "too many unique paths; skipping bigram generation for file");
                continue;
            }
            let present: HashSet<_> = ids.into_iter().collect();
            for (&(a, b), &idx) in &self.bigram_id_lookup {
                if present.contains(&a)
                    && present.contains(&b)
                    && let Some(&Some(slot)) = slots.get(idx)
                {
                    w.set_slot(slot, 1.0);
                }
            }
        }
    }

    /// Path-triple features, as [`Self::write_bigram_features`].
    fn write_trigram_features(&self, summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
        for s in summaries {
            let ranked;
            let paths = if self.ranked_ngrams {
                ranked = ranked_ngram_paths(&s.raw_findings, 128);
                &ranked
            } else {
                &s.unique_3level_paths
            };
            let ids: Vec<u32> = paths
                .iter()
                .filter_map(|p| self.path_to_id.get(p).copied())
                .collect();
            let n = ids.len();
            if n > 256 {
                tracing::warn!(path = %s.path, tokens = n, "too many unique paths; skipping trigram generation for file");
                continue;
            }
            let present: HashSet<_> = ids.into_iter().collect();
            for (&(a, b, c), &idx) in &self.trigram_id_lookup {
                if present.contains(&a)
                    && present.contains(&b)
                    && present.contains(&c)
                    && let Some(&Some(slot)) = self.trigram_slots.get(idx)
                {
                    w.set_slot(slot, 1.0);
                }
            }
        }
    }
}

/// Resolve the `metrics:` family to slots: every [`KEY_METRICS`] entry, then
/// each extended-vocab key (`<group>_<field>`) that is not a base metric.
/// Extended keys are log-scaled when the field names a count or size.
fn metric_slots(lookup: &HashMap<String, usize>, spec: &FeatureSpec) -> Vec<MetricSlot> {
    if let Some(transforms) = &spec.metric_transforms {
        return lookup
            .iter()
            .filter_map(|(name, &slot)| {
                let suffix = name.strip_prefix("metrics:")?;
                let (group, field) = suffix.split_once('_')?;
                Some(MetricSlot {
                    slot,
                    group: group.into(),
                    field: field.into(),
                    log: transforms.get(suffix).is_some_and(|t| t == "log1p_abs"),
                    getter: spec
                        .metric_getters
                        .iter()
                        .any(|g| g == suffix)
                        .then(|| suffix.to_owned()),
                    correlation: spec.metric_correlations.get(suffix).cloned(),
                })
            })
            .collect();
    }
    let metric_vocab = &spec.metric_vocab;
    let base = KEY_METRICS.iter().map(|&(group, field, log)| {
        (
            format!("metrics:{group}_{field}"),
            group.to_string(),
            field.to_string(),
            log,
        )
    });
    let base_keys: HashSet<String> = KEY_METRICS
        .iter()
        .map(|&(group, field, _)| format!("{group}_{field}"))
        .collect();
    let extended = metric_vocab.iter().filter_map(|key| {
        let (group, field) = key.split_once('_')?;
        if base_keys.contains(key.as_str()) {
            return None;
        }
        let log = ["count", "size", "total", "bytes", "length"]
            .iter()
            .any(|word| field.contains(word));
        Some((
            format!("metrics:{key}"),
            group.to_string(),
            field.to_string(),
            log,
        ))
    });
    base.chain(extended)
        .filter_map(|(name, group, field, log)| {
            Some(MetricSlot {
                slot: *lookup.get(&name)?,
                group,
                field,
                log,
                getter: None,
                correlation: None,
            })
        })
        .collect()
}

/// The finding fields retained for cross-file aggregation: ID dedup and
/// ATT&CK/MBC code rollups. Keeping just these avoids cloning the whole
/// finding JSON subtree for every file.
#[derive(Debug, Clone, Default, PartialEq)]
struct RawFinding {
    id: String,
    conf: f64,
    crit: u32,
    atk: Option<String>,
    mbc: Option<String>,
}

/// Select base paths before filtering against a model vocabulary, as in training.
fn ranked_ngram_paths(findings: &[RawFinding], limit: usize) -> Vec<String> {
    let mut scores: HashMap<&str, f64> = HashMap::new();
    for finding in findings {
        if finding.id.is_empty()
            || finding.conf.is_nan()
            || finding.conf < MIN_CONFIDENCE
            || finding.crit < 3
        {
            continue;
        }
        let path = finding.id.split("::").next().unwrap_or("");
        let score = if finding.conf.is_finite() {
            f64::from(finding.crit) * f64::from(finding.conf.min(1.0) as f32)
        } else {
            0.0
        };
        scores
            .entry(path)
            .and_modify(|v| *v = v.max(score))
            .or_insert(score);
    }
    let mut paths: Vec<_> = scores.keys().copied().collect();
    if paths.len() > limit {
        paths.sort_unstable_by(|a, b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(b)));
        paths.truncate(limit);
    }
    paths.sort_unstable();
    paths.into_iter().map(str::to_owned).collect()
}

/// Pre-calculated data for a single file entry in a report.
///
/// Compact placement metadata supplies archive structure. Timestamps and text
/// encodings remain unavailable in the retained-facts contract.
#[derive(Debug, Clone, Default, PartialEq)]
struct FileSummary {
    id: Option<u32>,
    parent: Option<u32>,
    depth: u32,
    entry_section_nontext: bool,
    path: String,
    file_type: String,
    size_bytes: f64,
    overall_entropy: f64,
    metrics: HashMap<String, HashMap<String, f64>>,
    findings: FindingSummary,
    risk: FileRiskStats,
    unique_3level_paths: Vec<String>,
    /// Import tokens (`name` and `lib!name`) for the logic-gap features.
    imports: HashSet<String>,
    /// Full `kv:` feature names from the metrics block. Empty unless a route
    /// reads the `kv:` family.
    kv_tokens: Vec<String>,
    /// Normalized import/export/function/AST symbols, sorted and unique.
    /// Empty unless a route reads the `symbol:` family.
    symbols: Vec<String>,
    /// Findings reduced to the fields used for cross-file aggregation.
    raw_findings: Vec<RawFinding>,
}

/// Which optional feature families are active for this route, so
/// [`FileSummary::from_compact`] only gathers the inputs they read.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RawNeeds {
    kv: bool,
    kv_shape: bool,
    kv_split: bool,
    symbol: bool,
}

impl RawNeeds {
    /// All optional inputs any specialist feature family may read.
    pub(crate) const fn all() -> Self {
        Self {
            kv: true,
            kv_shape: true,
            kv_split: true,
            symbol: true,
        }
    }

    /// The needs of any of two consumers — used to parse a report once for the
    /// general pass and every route, gathering an input if *any* of them reads it.
    /// `RawNeeds::default()` (no families) is the identity for folding these.
    pub(crate) const fn union(self, other: Self) -> Self {
        Self {
            kv: self.kv || other.kv,
            kv_shape: self.kv_shape || other.kv_shape,
            kv_split: self.kv_split || other.kv_split,
            symbol: self.symbol || other.symbol,
        }
    }
}

/// A report parsed into the route-independent summaries the feature writers read.
///
/// Building this is the expensive part of extraction (per-file `FileSummary`
/// construction, cross-file finding/metric aggregation, token sets). It depends
/// only on the report and the union of active families, never on a specific
/// route's vocab — so the general pass and every ML route share one
/// `ParsedReport` instead of each re-summarizing the same report.
#[derive(Debug)]
pub(crate) struct ParsedReport {
    trait_path_depth: usize,
    depth_variants: std::sync::Mutex<HashMap<usize, std::sync::Arc<ParsedReport>>>,
    summaries: Vec<FileSummary>,
    combined: FindingSummary,
    merged_metrics: MetricMap,
    formula_str: String,
    elements_str: String,
    sample_score: i64,
    /// Sorted `tier:category[/sub]` tokens for the `crit:` n-grams.
    crit_tokens: Vec<String>,
    /// Sorted `tier:path` tokens for the `tierbi:`/`tiertri:` families.
    tiered_tokens: Vec<String>,
    /// Distinct ATT&CK technique and MBC behavior codes, sorted.
    attack_codes: Vec<String>,
    mbc_codes: Vec<String>,
}

impl ParsedReport {
    /// Reuse decoded facts while rebuilding depth-dependent summaries for a
    /// route. The shared default-depth parse remains untouched for other routes.
    fn at_trait_path_depth(&self, depth: usize) -> std::sync::Arc<Self> {
        let mut variants = self
            .depth_variants
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::sync::Arc::clone(
            variants
                .entry(depth)
                .or_insert_with(|| std::sync::Arc::new(self.rebuild_at_trait_path_depth(depth))),
        )
    }

    fn rebuild_at_trait_path_depth(&self, depth: usize) -> Self {
        let mut summaries = self.summaries.clone();
        for file in &mut summaries {
            let views: Vec<_> = file
                .raw_findings
                .iter()
                .map(|f| FindingView {
                    id: &f.id,
                    conf: f.conf,
                    crit: f.crit,
                })
                .collect();
            file.findings = summarize_findings_at_depth(&views, depth);
            file.risk.suspicious_category_breadth = file.findings.suspicious_category_breadth;
            file.risk.hostile_category_breadth = file.findings.hostile_category_breadth;
        }
        let mut adjusted =
            Self::from_summaries(summaries, Some((&self.formula_str, self.sample_score)));
        adjusted.trait_path_depth = depth;
        adjusted
    }

    /// Featurize a typed compact report.
    ///
    /// `keep` restricts the pass to the sample's own files by sha — used when
    /// fetched payloads have been grafted in and must not dilute the sample's
    /// own aggregate. `None` featurizes every file. The primary file is the
    /// first entry at depth 0, resolved within whatever subset survives `keep`.
    pub(crate) fn from_compact_report(
        report: &cleave::types::CompactReport,
        needs: RawNeeds,
        keep: Option<&HashSet<String>>,
    ) -> Self {
        let files: Vec<&cleave::types::CompactFile> = match keep {
            Some(shas) => report
                .files
                .iter()
                .filter(|f| shas.contains(&f.sha))
                .collect(),
            None => report.files.iter().collect(),
        };
        Self::from_compact_files(&files, needs)
    }

    /// Featurize a single compact file entry (an archive member) on its own.
    pub(crate) fn from_compact_file(file: &cleave::types::CompactFile, needs: RawNeeds) -> Self {
        // Placement depth belongs to the enclosing archive. When grading this
        // member alone, its own metadata must match its training database row.
        Self::from_summaries(
            vec![FileSummary::from_compact(file, needs)],
            Some((file.formula.as_deref().unwrap_or(""), file.risk)),
        )
    }

    fn from_compact_files(files: &[&cleave::types::CompactFile], needs: RawNeeds) -> Self {
        // Small warm-cache reports are cheaper to summarize serially than to fan
        // out into rayon jobs.
        let file_summaries: Vec<FileSummary> = if files.len() < 8 {
            files
                .iter()
                .map(|&f| FileSummary::from_compact(f, needs))
                .collect()
        } else {
            files
                .par_iter()
                .map(|&f| FileSummary::from_compact(f, needs))
                .collect()
        };
        let primary = files
            .iter()
            .copied()
            .find(|f| f.depth == 0)
            .map(|f| (f.formula.as_deref().unwrap_or(""), f.risk));
        Self::from_summaries(file_summaries, primary)
    }

    fn from_summaries(file_summaries: Vec<FileSummary>, primary: Option<(&str, i64)>) -> Self {
        let summaries = if file_summaries.is_empty() {
            vec![FileSummary::default()]
        } else {
            file_summaries
        };
        let (formula_str, elements_str, sample_score) = primary.map_or_else(
            || (String::new(), String::new(), 0),
            |(f, risk)| {
                (
                    f.to_owned(),
                    f.chars()
                        .filter(|c| !('\u{2080}'..='\u{2089}').contains(c))
                        .collect(),
                    risk,
                )
            },
        );

        let combined = summarize_report_summaries(&summaries);
        let merged_metrics = merge_metric_summaries(&summaries);
        let crit_tokens = crit_category_tokens(&combined);
        let tiered_tokens = tiered_path_tokens(&combined);
        let mut attack_codes: HashSet<&str> = HashSet::new();
        let mut mbc_codes: HashSet<&str> = HashSet::new();
        for finding in summaries.iter().flat_map(|s| &s.raw_findings) {
            if let Some(a) = &finding.atk {
                attack_codes.insert(a);
            }
            if let Some(m) = &finding.mbc {
                mbc_codes.insert(m);
            }
        }
        let sorted = |codes: HashSet<&str>| {
            let mut codes: Vec<String> = codes.into_iter().map(str::to_string).collect();
            codes.sort();
            codes
        };
        let attack_codes = sorted(attack_codes);
        let mbc_codes = sorted(mbc_codes);
        Self {
            trait_path_depth: default_trait_path_depth(),
            depth_variants: std::sync::Mutex::default(),
            summaries,
            combined,
            merged_metrics,
            formula_str,
            elements_str,
            sample_score,
            crit_tokens,
            tiered_tokens,
            attack_codes,
            mbc_codes,
        }
    }
}

impl FileSummary {
    /// Summarize one compact file entry.
    fn from_compact(file: &cleave::types::CompactFile, needs: RawNeeds) -> Self {
        let raw_findings: Vec<RawFinding> = file
            .findings
            .iter()
            .map(|f| RawFinding {
                id: f.id.clone(),
                conf: f64::from(f.confidence),
                crit: u32::from(f.criticality),
                atk: f.attack.clone(),
                mbc: f.mbc.clone(),
            })
            .collect();
        let facts = &file.facts;
        let finding_views: Vec<FindingView<'_>> = raw_findings
            .iter()
            .map(|f| FindingView {
                id: &f.id,
                conf: f.conf,
                crit: f.crit,
            })
            .collect();
        let findings = summarize_findings(&finding_views);

        let size_bytes = file.size as f64;
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

        let mut unique_3level_paths: Vec<String> = finding_views
            .iter()
            .filter(|f| f.conf >= MIN_CONFIDENCE && f.crit >= CRIT_NOTABLE)
            .map(|f| f.id.split("::").next().unwrap_or(f.id).to_string())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        unique_3level_paths.sort();

        let metrics_value = facts
            .metrics
            .as_ref()
            .map(|m| &m.0)
            .filter(|v| v.is_object());
        let metrics = numeric_metrics(metrics_value);
        let overall_entropy = metrics
            .get("binary")
            .and_then(|m| m.get("overall_entropy"))
            .copied()
            .unwrap_or(0.0);

        let imports: HashSet<String> = facts
            .imports
            .iter()
            .flat_map(|imp| import_tokens_from_parts(&imp.library, &imp.name))
            .collect();

        Self {
            id: Some(file.id),
            parent: file.parent,
            depth: file.depth,
            entry_section_nontext: metrics_value
                .and_then(|v| v.get("pe"))
                .and_then(|v| v.get("entry_section"))
                .and_then(serde_json::Value::as_str)
                .is_some_and(|v| !v.is_empty() && v != ".text"),
            path: file.path.clone(),
            file_type: file.file_type.clone(),
            size_bytes,
            overall_entropy,
            metrics,
            findings,
            risk,
            unique_3level_paths,
            imports,
            kv_tokens: if needs.kv {
                kv_tokens_with_options(metrics_value, needs.kv_shape, needs.kv_split)
            } else {
                Vec::new()
            },
            symbols: if needs.symbol {
                file_symbols(facts)
            } else {
                Vec::new()
            },
            raw_findings,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
struct FindingSummary {
    sample_paths: HashMap<String, u32>,
    path_confidences: HashMap<String, f64>,
    path_scores: HashMap<String, f64>,
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

// Compatibility aliases for metrics cleave has renamed since August 2026. Keep
// in sync with collimator.features._METRIC_RENAMES: existing names (even
// null/zero) win. Keep every entry: hopper keeps reports from every generation.
const METRIC_RENAMES: &[(&str, &str, &str, &str)] = &[
    ("archive", "total_compressed", "archive", "compressed_size"),
    (
        "archive",
        "total_uncompressed",
        "archive",
        "uncompressed_size",
    ),
    ("ast", "array_literal_max_length", "ast", "max_array_length"),
    (
        "ast",
        "numeric_array_max_length",
        "ast",
        "max_numeric_array",
    ),
    (
        "ast",
        "numeric_sequence_max_length",
        "ast",
        "max_numeric_sequence",
    ),
    (
        "ast",
        "string_concat_chain_max_length",
        "ast",
        "max_concat_chain",
    ),
    (
        "binary",
        "embedded_archive_count",
        "binary",
        "embedded_archives",
    ),
    (
        "binary",
        "embedded_binary_count",
        "binary",
        "embedded_binaries",
    ),
    ("binary", "embedded_file_count", "binary", "embedded_files"),
    ("binary", "overall_entropy", "file", "entropy"),
    (
        "binary",
        "total_basic_blocks",
        "binary",
        "basic_block_count",
    ),
    ("comments", "base64_in_comments", "comments", "base64"),
    ("comments", "code_in_comments", "comments", "code"),
    ("comments", "empty_comments", "comments", "empty"),
    (
        "comments",
        "high_entropy_comments",
        "comments",
        "high_entropy",
    ),
    ("comments", "total", "comments", "count"),
    ("elf", "gnu_stack_section_absent", "elf", "no_gnu_stack"),
    (
        "elf",
        "stripped_but_symtab_present",
        "elf",
        "stripped_with_symtab",
    ),
    ("file", "size_bytes", "file", "size"),
    (
        "functions",
        "code_in_functions_ratio",
        "functions",
        "code_ratio",
    ),
    ("functions", "density_per_100_lines", "functions", "density"),
    ("functions", "nested_functions", "functions", "nested"),
    ("identifiers", "total", "identifiers", "count"),
    ("identifiers", "unique_count", "identifiers", "unique"),
    ("imports", "aliased_imports", "imports", "aliased"),
    ("imports", "dynamic_imports", "imports", "dynamic"),
    ("imports", "relative_imports", "imports", "relative"),
    ("imports", "unique_modules", "imports", "count"),
    ("imports", "wildcard_imports", "imports", "wildcard"),
    ("pe", "headers_size_bytes", "pe", "headers_size"),
    ("strings", "hex_strings", "strings", "hex"),
    ("strings", "shell_command_strings", "strings", "shell"),
    ("strings", "sql_strings", "strings", "sql"),
    ("strings", "total", "strings", "count"),
    ("strings", "total_bytes", "strings", "bytes"),
    (
        "strings",
        "unicode_heavy_strings",
        "strings",
        "unicode_heavy",
    ),
    ("strings", "url_encoded_strings", "strings", "url_encoded"),
    ("strings", "very_long_strings", "strings", "very_long"),
    ("text", "high_byte_ratio", "text", "non_ascii_ratio"),
    (
        "text",
        "max_inline_whitespace_run",
        "text",
        "max_whitespace_run",
    ),
    ("text", "most_common_char_codepoint", "text", "top_char"),
    ("text", "most_common_char_is_null", "text", "top_char_null"),
    ("text", "most_common_ratio", "text", "top_char_ratio"),
    ("text", "total_lines", "text", "lines"),
    ("ast", "member_depth_max", "ast", "max_member_depth"),
    (
        "binary",
        "high_entropy_strings",
        "binary",
        "high_entropy_string_count",
    ),
    (
        "binary",
        "normalized_string_count",
        "text",
        "normalized_string_count",
    ),
    ("comments", "url_in_comments", "comments", "url_count"),
    ("elf", "build_id_present", "elf", "has_build_id"),
    ("elf", "class_bits", "elf", "bits"),
    ("elf", "debuglink_present", "elf", "has_debuglink"),
    ("elf", "entry_point", "elf", "entry"),
    ("elf", "gnu_hash_present", "elf", "has_gnu_hash"),
    ("elf", "hidden_symbols", "elf", "hidden_symbol_count"),
    (
        "elf",
        "load_segment_max_p_filesz",
        "elf",
        "load_segment_max_file_size",
    ),
    (
        "elf",
        "load_segment_max_p_memsz",
        "elf",
        "load_segment_max_memory_size",
    ),
    ("pe", "certificate_table_size", "pe", "cert_table_size"),
    ("pe", "delay_load_imports", "pe", "delay_import_count"),
    ("pe", "manifest_present", "pe", "has_manifest"),
    ("pe", "size_of_headers", "pe", "headers_size"),
    ("pe", "size_of_image", "pe", "image_size"),
    ("pe", "tls_callbacks", "pe", "tls_callback_count"),
    ("pe", "version_info_present", "pe", "has_version_info"),
    ("text", "max_ws_run", "text", "max_whitespace_run"),
    ("apk", "declared_size", "apk", "installed_size"),
    ("apk", "depend_count", "apk", "dependency_count"),
    ("archive", "prefix_bytes", "archive", "leading_bytes"),
    (
        "archive",
        "timing.mtime_dominant_fraction",
        "archive",
        "timing.mtime_dominant_ratio",
    ),
    ("ast", "max_array_len", "ast", "max_array_length"),
    ("ast", "max_numeric_seq", "ast", "max_numeric_sequence"),
    ("binary", "basic_blocks", "binary", "basic_block_count"),
    ("binary", "huge_func_count", "binary", "huge_function_count"),
    ("binary", "leaf_func_count", "binary", "leaf_function_count"),
    ("binary", "tiny_func_count", "binary", "tiny_function_count"),
    (
        "class",
        "constant_pool_size",
        "class",
        "constant_pool_count",
    ),
    ("deb", "depends_count", "deb", "dependency_count"),
    ("deb", "installed_size", "deb", "installed_size_kib"),
    (
        "dmg",
        "total_uncompressed_bytes",
        "dmg",
        "uncompressed_size",
    ),
    (
        "elf",
        "compressed_sections_count",
        "elf",
        "compressed_section_count",
    ),
    (
        "elf",
        "has_direct_loader_dep",
        "elf",
        "has_direct_loader_dependency",
    ),
    (
        "iso",
        "blank_identifier_fields",
        "iso",
        "blank_identifier_count",
    ),
    ("iso", "dir_count", "iso", "directory_count"),
    ("iso", "udf.dir_count", "iso", "udf.directory_count"),
    ("jpeg", "appended_bytes", "jpeg", "trailing_bytes"),
    (
        "lnk",
        "args_leading_spaces",
        "lnk",
        "arguments_leading_spaces",
    ),
    ("lnk", "args_leading_tabs", "lnk", "arguments_leading_tabs"),
    (
        "lnk",
        "args_max_whitespace_run",
        "lnk",
        "arguments_max_whitespace_run",
    ),
    (
        "lnk",
        "args_whitespace_total",
        "lnk",
        "arguments_whitespace_count",
    ),
    (
        "office",
        "vba.mean_identifier_length",
        "office",
        "vba.avg_identifier_length",
    ),
    ("office", "vba_project_size", "office", "vba.project_size"),
    (
        "pdf",
        "decoded_form_value_max_len",
        "pdf",
        "decoded_form_value_max_length",
    ),
    ("pdf", "leading_bytes_before_header", "pdf", "leading_bytes"),
    ("pdf", "trailing_bytes_after_eof", "pdf", "trailing_bytes"),
    (
        "pe",
        "clr.strong_name_sig_size",
        "pe",
        "clr.strong_name_signature_size",
    ),
    ("pe", "recovered_exports", "pe", "recovered_export_count"),
    (
        "pe",
        "recovered_functions",
        "pe",
        "recovered_function_count",
    ),
    ("pe", "recovered_imports", "pe", "recovered_import_count"),
    ("pe", "recovered_sections", "pe", "recovered_section_count"),
    (
        "pe",
        "reloc_overhang_bytes",
        "pe",
        "relocation_overhang_bytes",
    ),
    (
        "pe",
        "reloc_overhang_ratio",
        "pe",
        "relocation_overhang_ratio",
    ),
    ("png", "chunks_after_iend", "png", "trailing_chunk_count"),
    ("rar", "encrypted_header", "rar", "headers_encrypted"),
    (
        "registry",
        "first_published_at",
        "registry",
        "first_published_unix",
    ),
    ("registry", "maintainers", "registry", "maintainer_count"),
    (
        "registry",
        "previous_published_at",
        "registry",
        "previous_published_unix",
    ),
    ("registry", "published_at", "registry", "published_unix"),
    ("rtf", "group_depth_max", "rtf", "max_group_depth"),
    ("scpt", "calls", "scpt", "call_count"),
    ("scpt", "decoded", "scpt", "decoded_count"),
    ("scpt", "events", "scpt", "event_count"),
    ("scpt", "handlers", "scpt", "handler_count"),
    ("sections", "entropy_max", "sections", "max_entropy"),
    ("sections", "entropy_mean", "sections", "avg_entropy"),
    ("source", "ast_walk_panic", "source", "ast_walk_panicked"),
    ("source", "extract_panic", "source", "extract_panicked"),
];

type MetricAliasKey = (&'static str, &'static str);

// Same equivalence classes/order as collimator: canonical spelling first,
// then historical names in the table's order. Never overwrite explicit data.
static METRIC_EQUIVALENTS: LazyLock<Vec<Vec<MetricAliasKey>>> = LazyLock::new(|| {
    let renames: HashMap<MetricAliasKey, MetricAliasKey> = METRIC_RENAMES
        .iter()
        .map(|&(og, ok, ng, nk)| ((og, ok), (ng, nk)))
        .collect();
    let mut groups: Vec<Vec<MetricAliasKey>> = Vec::new();
    let mut positions = HashMap::new();
    for &(og, ok, _, _) in METRIC_RENAMES {
        let old = (og, ok);
        let mut canonical = old;
        let mut seen = HashSet::new();
        while let Some(&next) = renames.get(&canonical) {
            assert!(seen.insert(canonical), "cyclic metric rename");
            canonical = next;
        }
        let index = *positions.entry(canonical).or_insert_with(|| {
            groups.push(vec![canonical]);
            groups.len() - 1
        });
        if !groups[index].contains(&old) {
            groups[index].push(old);
        }
    }
    groups
});

/// Borrow numeric/categorical facts under old and new metric spellings.
/// Missing aliases are filled; explicit values and malformed groups win.
fn metric_fields(
    source: Option<&serde_json::Value>,
) -> impl Iterator<Item = (&str, &str, &serde_json::Value)> {
    let groups = source.and_then(serde_json::Value::as_object);
    let original = groups
        .into_iter()
        .flat_map(|groups| groups.iter())
        .flat_map(|(group, fields)| {
            fields.as_object().into_iter().flat_map(move |fields| {
                fields
                    .iter()
                    .map(move |(key, value)| (group.as_str(), key.as_str(), value))
            })
        });
    let aliases = METRIC_EQUIVALENTS.iter().flat_map(move |equivalents| {
        let value = equivalents
            .iter()
            .find_map(|&(group, key)| groups?.get(group)?.as_object()?.get(key));
        equivalents.iter().filter_map(move |&(group, key)| {
            let groups = groups?;
            if let Some(fields) = groups.get(group)
                && fields.as_object()?.contains_key(key)
            {
                return None;
            }
            Some((group, key, value?))
        })
    });
    original.chain(aliases)
}

/// Keep numbers and booleans (`true` → 1.0), dropping strings/nulls.
fn numeric_metrics(source: Option<&serde_json::Value>) -> MetricMap {
    let mut result = MetricMap::new();
    for (group, key, value) in metric_fields(source) {
        if let Some(number) = value
            .as_f64()
            .or_else(|| value.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
        {
            result
                .entry(group.to_owned())
                .or_default()
                .insert(key.to_owned(), number);
        }
    }
    result
}

/// The tokens a `(library, name)` import pair contributes: the bare name and
/// the `lib!name` form, each subject to collimator's per-token `len >= 2`
/// gate.
fn import_tokens_from_parts(lib: &str, name: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(2);
    if name.len() >= 2 {
        out.push(name.to_string());
    }
    let combined = if !lib.is_empty() && !name.is_empty() {
        format!("{lib}!{name}")
    } else if name.is_empty() {
        lib.to_string()
    } else {
        name.to_string()
    };
    if combined.len() >= 2 && !out.contains(&combined) {
        out.push(combined);
    }
    out
}

/// The whole of a finding that scoring reads: its id, confidence and
/// criticality ordinal.
#[derive(Debug, Clone, Copy)]
struct FindingView<'a> {
    id: &'a str,
    conf: f64,
    crit: u32,
}

fn summarize_findings(findings: &[FindingView<'_>]) -> FindingSummary {
    summarize_findings_at_depth(findings, default_trait_path_depth())
}

fn summarize_findings_at_depth(findings: &[FindingView<'_>], depth: usize) -> FindingSummary {
    let mut summary = FindingSummary::default();
    let mut notable_ids: HashSet<&str> = HashSet::new();
    let mut suspicious_ids: HashSet<&str> = HashSet::new();
    let mut hostile_ids: HashSet<&str> = HashSet::new();

    for finding in findings {
        let fid = finding.id;
        if fid.is_empty() {
            continue;
        }
        let conf = finding.conf;
        if conf < MIN_CONFIDENCE {
            continue;
        }

        summary.filtered_finding_count += 1;
        summary.finding_confidences.push(conf);
        let crit_ord = finding.crit;

        if crit_ord >= CRIT_NOTABLE {
            summary.notable_finding_count += 1;
            notable_ids.insert(fid);
        }
        if crit_ord >= CRIT_SUSPICIOUS {
            summary.suspicious_finding_count += 1;
            suspicious_ids.insert(fid);
        }
        if crit_ord >= CRIT_HOSTILE {
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
                if crit_ord >= CRIT_HOSTILE {
                    summary.well_known_hostile += 1;
                } else if crit_ord >= CRIT_SUSPICIOUS {
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
        let score = if conf.is_finite() {
            f64::from(crit_ord) * f64::from(conf.clamp(0.0, 1.0) as f32)
        } else {
            0.0
        };
        for path in finding_paths(fid, depth) {
            match summary.path_scores.get_mut(path) {
                Some(v) => *v = v.max(score),
                None => {
                    summary.path_scores.insert(path.to_owned(), score);
                }
            }
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
    (
        summary.suspicious_category_breadth,
        summary.hostile_category_breadth,
    ) = category_breadth(&summary.sample_paths);
    summary
}

/// Distinct top-level categories reached at suspicious and at hostile
/// criticality.
fn category_breadth(sample_paths: &HashMap<String, u32>) -> (usize, usize) {
    let mut susp_cats: HashSet<&str> = HashSet::new();
    let mut host_cats: HashSet<&str> = HashSet::new();
    for (path, &max_ord) in sample_paths {
        if max_ord >= CRIT_SUSPICIOUS {
            susp_cats.insert(path.split('/').next().unwrap_or(""));
        }
        if max_ord >= CRIT_HOSTILE {
            host_cats.insert(path.split('/').next().unwrap_or(""));
        }
    }
    (susp_cats.len(), host_cats.len())
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
        for (path, &score) in &fs.path_scores {
            match combined.path_scores.get_mut(path) {
                Some(v) => *v = v.max(score),
                None => {
                    combined.path_scores.insert(path.clone(), score);
                }
            }
        }
        combined
            .finding_confidences
            .extend(fs.finding_confidences.iter().copied());

        // Re-scan raw findings to deduplicate unique IDs across files.
        for finding in &s.raw_findings {
            let fid = finding.id.as_str();
            if fid.is_empty() {
                continue;
            }
            if finding.conf < MIN_CONFIDENCE {
                continue;
            }
            let crit = finding.crit;
            if crit >= CRIT_NOTABLE {
                notable_ids.insert(fid);
            }
            if crit >= CRIT_SUSPICIOUS {
                suspicious_ids.insert(fid);
            }
            if crit >= CRIT_HOSTILE {
                hostile_ids.insert(fid);
            }
        }
    }

    combined.unique_notable_ids = notable_ids.len();
    combined.unique_suspicious_ids = suspicious_ids.len();
    combined.unique_hostile_ids = hostile_ids.len();
    (
        combined.suspicious_category_breadth,
        combined.hostile_category_breadth,
    ) = category_breadth(&combined.sample_paths);
    combined
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
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

/// G3: report-level aggregates.
/// Optional finding features consume the retained raw fields without another parse.
fn write_optional_finding_features(parsed: &ParsedReport, w: &mut FeatureWriter<'_>) {
    for code in &parsed.mbc_codes {
        w.set_token(&format!("mbc:{code}"), 1.0);
    }
    let needs_moments = [
        "agg:confidence_mean",
        "agg:confidence_std",
        "agg:confidence_skew",
        "agg:confidence_kurtosis",
    ]
    .iter()
    .any(|name| w.lookup.contains_key(*name));
    if needs_moments {
        let values: Vec<f64> = parsed
            .summaries
            .iter()
            .flat_map(|s| &s.raw_findings)
            .filter(|f| !f.id.is_empty() && f.conf >= MIN_CONFIDENCE)
            .map(|f| f.conf)
            .collect();
        if !values.is_empty() {
            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let variance = values.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt();
            let (skew, kurtosis) = if std > 1e-9 {
                (
                    values.iter().map(|c| (c - mean).powi(3)).sum::<f64>() / n / std.powi(3),
                    values.iter().map(|c| (c - mean).powi(4)).sum::<f64>() / n / std.powi(4) - 3.0,
                )
            } else {
                (0.0, 0.0)
            };
            w.set("agg:confidence_mean", mean as f32);
            w.set("agg:confidence_std", std as f32);
            w.set("agg:confidence_skew", skew as f32);
            w.set("agg:confidence_kurtosis", kurtosis as f32);
        }
    }
    if w.lookup.contains_key("agg:trait_id_lexical_distance") {
        let mut ids: Vec<&str> = parsed
            .summaries
            .iter()
            .flat_map(|s| &s.raw_findings)
            .map(|f| f.id.as_str())
            .filter(|s| !s.is_empty())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        let total: usize = ids
            .windows(2)
            .map(|pair| lexical_distance(pair[0], pair[1]))
            .sum();
        let average = if ids.len() > 1 {
            total as f64 / (ids.len() - 1) as f64
        } else {
            0.0
        };
        w.set("agg:trait_id_lexical_distance", average as f32);
    }
    if [
        "agg:docobf_obfuscation_count",
        "agg:docobf_eval_count",
        "agg:docobf_lure_count",
        "agg:docobf_total_count",
        "agg:docobf_has_any",
    ]
    .iter()
    .any(|name| w.lookup.contains_key(*name))
    {
        let mut counts = [0u32; 3];
        let prefixes = [
            "objectives/anti-static/obfuscation/document",
            "objectives/execution/interpreter/eval",
            "objectives/execution/lure/document",
        ];
        for finding in parsed.summaries.iter().flat_map(|s| &s.raw_findings) {
            if let Some(index) = prefixes.iter().position(|p| finding.id.starts_with(p)) {
                counts[index] += 1;
            }
        }
        let total = counts.iter().sum::<u32>();
        w.set("agg:docobf_obfuscation_count", counts[0] as f32);
        w.set("agg:docobf_eval_count", counts[1] as f32);
        w.set("agg:docobf_lure_count", counts[2] as f32);
        w.set("agg:docobf_total_count", total as f32);
        w.set("agg:docobf_has_any", f32::from(total > 0));
    }
}

fn lexical_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let previous = row[j + 1];
            row[j + 1] = (row[j] + 1)
                .min(previous + 1)
                .min(diagonal + usize::from(ca != cb));
            diagonal = previous;
        }
    }
    row[b.len()]
}

fn write_aggregate_features(parsed: &ParsedReport, w: &mut FeatureWriter<'_>) {
    let summary = &parsed.combined;
    let summaries = parsed.summaries.as_slice();
    let breadth = PathBreadth::new(summary);
    write_breadth_features(summary, &breadth, w);
    write_finding_volume_features(summary, summaries, w);
    write_top_file_features(summaries, w);
    write_file_tier_features(summaries, w);
    write_code_features(summary, &parsed.attack_codes, &parsed.mbc_codes, w);
}

/// How widely the sample's findings spread over the trait taxonomy, counted
/// over distinct sample paths at each criticality tier.
struct PathBreadth {
    max_crit: u32,
    /// Top-level categories reached at baseline or above.
    categories: usize,
    /// Paths at depth >= 2 reached at baseline or above.
    any: u32,
    /// Paths at depth >= 2 reached at notable or above, and the split of
    /// those by their highest tier.
    notable: u32,
    suspicious: u32,
    hostile: u32,
    notable_only: u32,
}

impl PathBreadth {
    fn new(summary: &FindingSummary) -> Self {
        let mut breadth = Self {
            max_crit: 0,
            categories: 0,
            any: 0,
            notable: 0,
            suspicious: 0,
            hostile: 0,
            notable_only: 0,
        };
        let mut categories: HashSet<&str> = HashSet::new();
        for (path, &max_ord) in &summary.sample_paths {
            let path_depth = path.chars().filter(|&c| c == '/').count();
            if max_ord >= CRIT_BASELINE {
                categories.insert(path.split('/').next().unwrap_or(""));
                if path_depth >= 2 {
                    breadth.any += 1;
                }
            }
            if path_depth < 2 || max_ord < CRIT_NOTABLE {
                continue;
            }
            breadth.notable += 1;
            breadth.max_crit = breadth.max_crit.max(max_ord);
            if max_ord >= CRIT_SUSPICIOUS {
                breadth.suspicious += 1;
            }
            if max_ord >= CRIT_HOSTILE {
                breadth.hostile += 1;
            } else if max_ord == CRIT_NOTABLE {
                breadth.notable_only += 1;
            }
        }
        breadth.categories = categories.len();
        breadth
    }
}

/// Breadth over the taxonomy: categories, active paths, and how far their
/// criticality escalates.
fn write_breadth_features(
    summary: &FindingSummary,
    breadth: &PathBreadth,
    w: &mut FeatureWriter<'_>,
) {
    w.set("agg:max_crit", breadth.max_crit as f32);
    w.set("agg:category_breadth", breadth.categories as f32);
    w.set("agg:path_breadth_any", (breadth.any as f32).ln_1p());
    w.set("agg:total_active_paths", (breadth.notable as f32).ln_1p());
    w.set(
        "agg:suspicious_concentration",
        breadth.suspicious as f32 / breadth.any.max(1) as f32,
    );
    w.set(
        "agg:hostile_concentration",
        breadth.hostile as f32 / breadth.any.max(1) as f32,
    );
    w.set(
        "agg:escalation_rate",
        breadth.suspicious as f32 / breadth.notable.max(1) as f32,
    );
    w.set(
        "agg:notable_only_fraction",
        breadth.notable_only as f32 / breadth.notable.max(1) as f32,
    );
    w.set(
        "agg:hostile_escalation_rate",
        breadth.hostile as f32 / breadth.notable.max(1) as f32,
    );
    w.set(
        "agg:hostile_share_of_suspicious",
        breadth.hostile as f32 / breadth.suspicious.max(1) as f32,
    );
    let category_denom = breadth.categories.max(1) as f32;
    w.set(
        "agg:suspicious_category_density",
        summary.suspicious_category_breadth as f32 / category_denom,
    );
    w.set(
        "agg:hostile_category_density",
        summary.hostile_category_breadth as f32 / category_denom,
    );

    // 2-level breadth features.
    let mut suspicious_2level: HashSet<String> = HashSet::new();
    let mut hostile_2level: HashSet<String> = HashSet::new();
    let mut objectives_2level: HashSet<String> = HashSet::new();
    for (path, &max_ord) in &summary.sample_paths {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() >= 2 {
            let two_level = format!("{}/{}", parts[0], parts[1]);
            if max_ord >= CRIT_SUSPICIOUS {
                suspicious_2level.insert(two_level.clone());
            }
            if max_ord >= CRIT_HOSTILE {
                hostile_2level.insert(two_level.clone());
            }
            if parts[0] == "objectives" && max_ord >= CRIT_BASELINE {
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
}

/// Finding counts at each tier: as logs, per KB, as shares of each other,
/// and how often the same id or category repeats.
fn write_finding_volume_features(
    summary: &FindingSummary,
    summaries: &[FileSummary],
    w: &mut FeatureWriter<'_>,
) {
    let total_size_bytes: f64 = summaries.iter().map(|s| s.size_bytes).sum();
    let total_kb_raw = (total_size_bytes / 1024.0) as f32;
    let total_kb_p1 = total_kb_raw.max(0.1);
    let total_kb_1 = total_kb_raw.max(1.0);

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

    // Size-invariant crit-tier severity fractions — mirror of collimator's
    // include_severity_fractions block (count of crit>=N findings as a share of
    // ALL findings, not per-KB).
    let total_findings = summary.filtered_finding_count.max(1) as f32;
    let mundane = summary
        .filtered_finding_count
        .saturating_sub(summary.notable_finding_count)
        .max(1) as f32;
    w.set(
        "agg:crit3_finding_fraction",
        summary.notable_finding_count as f32 / total_findings,
    );
    w.set(
        "agg:crit4_finding_fraction",
        summary.suspicious_finding_count as f32 / total_findings,
    );
    w.set(
        "agg:hostile_finding_fraction",
        summary.hostile_finding_count as f32 / total_findings,
    );
    w.set(
        "agg:severe_to_mundane_ratio",
        summary.notable_finding_count as f32 / mundane,
    );
    w.set(
        "agg:crit4_present",
        f32::from(summary.suspicious_finding_count > 0),
    );

    w.set(
        "agg:suspicious_category_breadth",
        summary.suspicious_category_breadth as f32,
    );
    w.set(
        "agg:hostile_category_breadth",
        summary.hostile_category_breadth as f32,
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
}

/// The `top1` block: the single riskiest file by each ordering.
fn write_top_file_features(summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
    let topk = topk_file_risk_features_from_summaries(summaries);
    w.set("agg:top1_file_suspicious_ratio_sum", topk[0]);
    w.set("agg:top1_file_hostile_ratio_sum", topk[1]);
    w.set("agg:top1_file_suspicious_findings_log", topk[2]);
    w.set("agg:top1_file_hostile_findings_log", topk[3]);
    w.set("agg:top1_file_suspicious_density_sum", topk[4]);
    w.set("agg:top1_file_hostile_density_sum", topk[5]);
    w.set("agg:top1_file_suspicious_category_breadth_sum", topk[6]);
    w.set("agg:top1_file_hostile_category_breadth_sum", topk[7]);

    // The "top-k weighted sum" is the single max; min_by over the same
    // descending comparator picks the same first-maximum file a stable
    // sort + take(1) would.
    let top_weighted = summaries
        .iter()
        .map(|s| &s.risk)
        .min_by(|a, b| {
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
        })
        .map_or(0.0, |s| s.hostile_density + 0.25 * s.suspicious_density);
    w.set("agg:top1_file_hostile_weighted_density_sum", top_weighted);
}

/// How many files sit in each criticality tier.
fn write_file_tier_features(summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
    let n_files = summaries.len().max(1) as f32;
    let hostile_files = summaries
        .iter()
        .filter(|s| s.risk.max_crit >= CRIT_HOSTILE)
        .count() as f32;
    let suspicious_files = summaries
        .iter()
        .filter(|s| s.risk.max_crit == CRIT_SUSPICIOUS)
        .count() as f32;
    let notable_files = summaries
        .iter()
        .filter(|s| s.risk.max_crit == CRIT_NOTABLE)
        .count() as f32;
    w.set("agg:file_hostile_fraction", hostile_files / n_files);
    w.set("agg:file_suspicious_fraction", suspicious_files / n_files);
    w.set("agg:file_notable_fraction", notable_files / n_files);
    w.set("agg:file_hostile_count_log", hostile_files.ln_1p());
    w.set("agg:file_suspicious_count_log", suspicious_files.ln_1p());
    w.set("agg:file_notable_count_log", notable_files.ln_1p());
    w.set("agg:hostile_depth_weight", 0.0);
}

/// ATT&CK / MBC code counts and the objective-path co-occurrence counts.
fn write_code_features(
    summary: &FindingSummary,
    attack_techniques: &[String],
    mbc_behaviors: &[String],
    w: &mut FeatureWriter<'_>,
) {
    w.set("agg:attack_technique_count", attack_techniques.len() as f32);
    // An ATT&CK technique ID is ASCII ("T1059.001"), but `atk` is whatever the
    // finding carried and nothing validates it, so a multi-byte codepoint can
    // sit inside the first four bytes. `len() >= 4` counts bytes and says
    // nothing about where characters begin: "Tü─…" is T(0) ü(1..3) ─(3..6), so
    // a byte-4 slice lands mid-character and panics, taking the whole analysis
    // task with it. `get` yields None on a non-boundary index instead.
    let tactic_prefixes: HashSet<&str> = attack_techniques
        .iter()
        .filter_map(|t| t.get(..4))
        .filter(|prefix| prefix.starts_with('T'))
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
        f32::from(!attack_techniques.is_empty() && has_objectives),
    );

    // Objective-path co-occurrence aggregates.  Mirrors collimator's
    // `agg:objective_bigram_count` / `agg:objective_trigram_count` (features.py
    // around line 2155).  Counts unordered pairs and triples among the distinct
    // `objectives/*` and `well-known/*` sample paths.  The trigram inner loop
    // is capped at 20 per (i, j) pair to bound work on samples with many
    // objective paths — the same cap collimator uses, so the values match.
    let n_obj = summary
        .sample_paths
        .keys()
        .filter(|p| p.starts_with("objectives/") || p.starts_with("well-known/"))
        .count();
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
        CRIT_HOSTILE => "h",
        CRIT_SUSPICIOUS => "s",
        _ => "n",
    }
}

/// Categories whose notable-or-above paths feed the `crit:` n-grams.
const CRIT_CATEGORIES: &[&str] = &[
    "objectives",
    "well-known",
    "supply-chain",
    "anti-analysis",
    "anti-static",
    "command-and-control",
    "evasion",
    "execution",
    "exfiltration",
];

/// Sorted `tier:category[/sub]` tokens: each notable-or-above sample path in
/// a [`CRIT_CATEGORIES`] category, cut to two levels, at its highest tier.
fn crit_category_tokens(summary: &FindingSummary) -> Vec<String> {
    let mut max_crit: HashMap<String, u32> = HashMap::new();
    for (path, &mo) in &summary.sample_paths {
        if mo < CRIT_NOTABLE {
            continue;
        }
        let parts: Vec<&str> = path.split('/').collect();
        if !CRIT_CATEGORIES.contains(&parts[0]) {
            continue;
        }
        let key = if parts.len() >= 2 {
            format!("{}/{}", parts[0], parts[1])
        } else {
            parts[0].to_string()
        };
        let e = max_crit.entry(key).or_insert(0);
        *e = (*e).max(mo);
    }
    let mut tokens: Vec<String> = max_crit
        .iter()
        .map(|(k, &c)| format!("{}:{k}", tier_prefix(c)))
        .collect();
    tokens.sort();
    tokens
}

/// Sorted `tier:path` tokens from prefixes at the model's trait depth,
/// shared by the tiered bigram and trigram feature builders.
fn tiered_path_tokens(summary: &FindingSummary) -> Vec<String> {
    tiered_path_tokens_with_options(summary, 3, &HashMap::new())
}

fn tiered_path_tokens_with_options(
    summary: &FindingSummary,
    minimum: u32,
    branches: &HashMap<String, u32>,
) -> Vec<String> {
    let mut tokens: Vec<String> = summary
        .sample_paths
        .iter()
        .filter_map(|(path, &crit)| {
            let branch = path.split('/').next().unwrap_or("");
            let floor = branches.get(branch).copied().unwrap_or(minimum);
            (crit > 0 && crit >= floor).then(|| format!("{}:{path}", tier_prefix(crit)))
        })
        .collect();
    tokens.sort();
    tokens
}

/// Keep real finding evidence, with deterministic lexical ties. The score for
/// each path was accumulated as max(severity * confidence), never a product
/// of unrelated maxima. Match collimator's v20 token budgets exactly.
fn ranked_tiered_path_tokens(
    summary: &FindingSummary,
    minimum: u32,
    branches: &HashMap<String, u32>,
    order: usize,
) -> Vec<String> {
    let mut tokens = tiered_path_tokens_with_options(summary, minimum, branches);
    let limit = match order {
        2 => 512,
        3 => 128,
        _ => 64,
    };
    if tokens.len() > limit {
        // Each token is `<tier>:<path>`; rank by the path's score.
        let score = |token: &str| {
            let path = token.split_once(':').map_or(token, |(_, path)| path);
            summary.path_scores.get(path).copied().unwrap_or(0.0)
        };
        tokens.sort_unstable_by(|a, b| {
            let (score_a, score_b) = (score(a), score(b));
            score_b.total_cmp(&score_a).then_with(|| a.cmp(b))
        });
        tokens.truncate(limit);
        tokens.sort();
    }
    tokens
}

/// `crit:` unigrams over the category tokens, then their n-grams.
fn write_crit_ngrams(tokens: &[String], w: &mut FeatureWriter<'_>) {
    let mut key = String::new();
    for t in tokens {
        key.clear();
        let _ = write!(key, "crit:{t}");
        w.set_token(&key, 1.0);
    }
    write_ngrams(w, tokens, "critbi:", "crittri:", "crit");
}

/// Set the pair (`<bi>a + b`) and triple (`<tri>a + b + c`) tokens over
/// sorted, distinct `tokens`. Pairs are O(n²) and triples O(n³), so a
/// bloated report skips both past 512 tokens and triples past 128.
fn write_ngrams(w: &mut FeatureWriter<'_>, tokens: &[String], bi: &str, tri: &str, what: &str) {
    if tokens.len() > 512 {
        tracing::warn!(
            tokens = tokens.len(),
            "too many unique {what} tokens; skipping n-gram generation"
        );
        return;
    }
    let mut key = String::new();
    for (i, t1) in tokens.iter().enumerate() {
        for (j, t2) in tokens.iter().enumerate().skip(i + 1) {
            key.clear();
            let _ = write!(key, "{bi}{t1} + {t2}");
            w.set_token(&key, 1.0);
            if tokens.len() <= 128 {
                for t3 in &tokens[j + 1..] {
                    key.clear();
                    let _ = write!(key, "{tri}{t1} + {t2} + {t3}");
                    w.set_token(&key, 1.0);
                }
            }
        }
    }
}

fn topk_file_risk_features_from_summaries(summaries: &[FileSummary]) -> [f32; 8] {
    let top_susp = summaries.iter().map(|s| &s.risk).min_by(|a, b| {
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
    let top_host = summaries.iter().map(|s| &s.risk).min_by(|a, b| {
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

    let (Some(top_susp), Some(top_host)) = (top_susp, top_host) else {
        return [0.0; 8];
    };

    [
        top_susp.suspicious_ratio,
        top_host.hostile_ratio,
        (top_susp.suspicious_findings as f32 + 1.0).ln(),
        (top_host.hostile_findings as f32 + 1.0).ln(),
        top_susp.suspicious_density,
        top_host.hostile_density,
        top_susp.suspicious_category_breadth as f32,
        top_host.hostile_category_breadth as f32,
    ]
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

/// G6b: portable format-group hints derived only from cleave file types.
fn write_format_hint_features(summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
    let total_files = summaries.len().max(1) as f32;
    let mut known_files = 0usize;
    let mut present_groups = HashSet::new();

    // Each file's format groups, resolved once — the loop below is O(groups × files).
    let file_groups: Vec<Vec<&'static str>> = summaries
        .iter()
        .map(|s| format_groups_for_type(&s.file_type))
        .collect();

    for (&(group, _), names) in FORMAT_GROUPS.iter().zip(FORMAT_FEATURE_NAMES.iter()) {
        let mut group_count = 0usize;
        let mut suspicious_count = 0usize;
        let mut hostile_count = 0usize;

        for (s, groups) in summaries.iter().zip(&file_groups) {
            if groups.contains(&group) {
                group_count += 1;
                present_groups.insert(group);
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
        let [
            present,
            file_fraction,
            inner_fraction,
            suspicious_fraction,
            hostile_fraction,
        ] = names;
        w.set(present, f32::from(group_count > 0));
        w.set(file_fraction, group_count as f32 / total_files);
        // No compact file has a parent link, so no file is "inner".
        w.set(inner_fraction, 0.0);
        w.set(suspicious_fraction, suspicious_count as f32 / group_denom);
        w.set(hostile_fraction, hostile_count as f32 / group_denom);
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

/// Resolve each parent edge once; malformed cycles have no depth evidence.
fn relative_depths(files: &[FileSummary]) -> Vec<u32> {
    let by_id: HashMap<_, _> = files
        .iter()
        .enumerate()
        .filter_map(|(i, f)| Some((f.id?, i)))
        .collect();
    let mut depths = vec![None; files.len()];
    for start in 0..files.len() {
        let mut chain = Vec::new();
        let mut seen = HashSet::new();
        let mut i = start;
        while depths[i].is_none() {
            if !seen.insert(i) {
                for &(j, _) in &chain {
                    depths[j] = Some(0);
                }
                break;
            }
            if files[i].depth > 0 {
                depths[i] = Some(files[i].depth);
                break;
            }
            let parent = files[i]
                .parent
                .and_then(|p| by_id.get(&p).copied())
                .filter(|&p| p != i);
            let Some(parent) = parent else {
                depths[i] = Some(0);
                break;
            };
            chain.push((i, parent));
            i = parent;
        }
        for &(j, parent) in chain.iter().rev() {
            if depths[j].is_none() {
                depths[j] = Some(depths[parent].unwrap_or(0u32).saturating_add(1));
            }
        }
    }
    let base = depths.iter().flatten().copied().min().unwrap_or(0);
    depths
        .into_iter()
        .map(|d| d.unwrap_or(0).saturating_sub(base))
        .collect()
}

/// G15: structure relative to the graded artifact (legacy bundles keep zeros).
fn write_structural_extensions(
    summaries: &[FileSummary],
    combined: &FindingSummary,
    relative: bool,
    w: &mut FeatureWriter<'_>,
) {
    let binary_like = ["pe", "elf", "macho"];
    let source_types = ["javascript", "python", "typescript", "ruby", "php"];
    let text_exts = ["txt", "md", "json", "png", "jpg"];
    let code_types = ["javascript", "python", "pe", "elf", "macho"];

    let mut entropies = Vec::new();
    let mut code_entropies = Vec::new();
    let mut max_entropy = 0.0_f64;
    let mut hostile_files = 0;
    let mut total_loc: u64 = 0;
    let mut extension_mismatches = 0;
    let mut has_source_files = relative
        && summaries
            .iter()
            .any(|s| source_types.contains(&s.file_type.as_str()));
    let mut has_foreign_binaries = false;

    for s in summaries {
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
        #[expect(clippy::cast_sign_loss, reason = "a line count is never negative")]
        let lines = lines as u64;
        total_loc += lines;

        if !s.path.is_empty()
            && s.path.contains('.')
            && let Some(ext) = s.path.rsplit('.').next()
            && binary_like.contains(&s.file_type.as_str())
            && text_exts.contains(&ext.to_ascii_lowercase().as_str())
        {
            extension_mismatches += 1;
        }

        if s.overall_entropy > 0.0 {
            entropies.push(s.overall_entropy);
        }
        max_entropy = max_entropy.max(s.overall_entropy);

        if s.findings.hostile_finding_count > 0 {
            hostile_files += 1;
        }

        if s.overall_entropy > 0.0 && code_types.contains(&s.file_type.as_str()) {
            code_entropies.push(s.overall_entropy);
        }
    }

    w.set(
        "struct:packaged_capability",
        (combined.sample_paths.len() as f64 * max_entropy) as f32,
    );
    let depths = if relative {
        relative_depths(summaries)
    } else {
        vec![0; summaries.len()]
    };
    w.set(
        "struct:max_nesting_depth_log",
        f64::from(*depths.iter().max().unwrap_or(&0)).ln_1p() as f32,
    );
    w.set(
        "struct:inner_file_ratio",
        depths.iter().filter(|&&d| d > 0).count() as f32 / summaries.len().max(1) as f32,
    );
    if relative {
        w.set(
            "agg:hostile_depth_weight",
            summaries
                .iter()
                .zip(&depths)
                .map(|(s, &d)| f64::from(s.findings.hostile_finding_count) * f64::from(d))
                .sum::<f64>() as f32,
        );
    }
    if entropies.len() > 1 {
        let mean = entropies.iter().sum::<f64>() / entropies.len() as f64;
        let var =
            entropies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / entropies.len() as f64;
        w.set("struct:entropy_std_dev", var.sqrt() as f32);
        let mx = entropies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        w.set("struct:entropy_max_diff", (mx - mean) as f32);
    }
    w.set(
        "struct:air_gap_signal",
        f32::from(
            hostile_files > 0
                && !summaries
                    .iter()
                    .zip(&depths)
                    .any(|(s, &d)| d > 0 && s.findings.hostile_finding_count > 0),
        ),
    );
    if !code_entropies.is_empty() {
        let avg_ent = if entropies.is_empty() {
            0.0
        } else {
            entropies.iter().sum::<f64>() / entropies.len() as f64
        };
        let max_code_ent = code_entropies
            .iter()
            .copied()
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

/// G19: a capability imported but never exercised (no notable behavior).
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

    for (target_cat, name) in LOGIC_GAP_CATEGORIES.iter().zip(GAP_FEATURE_NAMES.iter()) {
        if let Some((_, imports_set, traits_set)) =
            logic_gaps.iter().find(|(c, _, _)| c == target_cat)
        {
            let has_import = imports_set.iter().any(|imp| all_imports.contains(*imp));
            let has_behavior = summary.sample_paths.iter().any(|(path, &max_ord)| {
                max_ord >= CRIT_NOTABLE && traits_set.iter().any(|t| path.starts_with(t))
            });
            if has_import && !has_behavior {
                w.set(name, 1.0);
            }
        }
    }
}

/// G22: a risky behavior with no documentation explaining it.
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
    for (target_cat, name) in INTENT_GAP_CATEGORIES
        .iter()
        .zip(INTENT_GAP_FEATURE_NAMES.iter())
    {
        let traits = risky
            .iter()
            .find(|(c, _)| c == target_cat)
            .map(|(_, t)| *t)
            .unwrap_or(&[]);
        let has_behavior = summary.sample_paths.iter().any(|(path, &max_ord)| {
            max_ord >= CRIT_SUSPICIOUS && traits.iter().any(|t| path.starts_with(t))
        });
        if has_behavior && !intent_signal {
            w.set(name, 1.0);
        }
    }
}

/// G23: negative space — a file type present without a trait it should have.
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
    let expected = EXPECTED_GHOSTS
        .iter()
        .flat_map(|&(ftype, traits)| traits.iter().map(move |t| (ftype, *t)));
    for ((ftype, trait_path), name) in expected.zip(MISSING_FEATURE_NAMES.iter()) {
        if present_types.contains(ftype) && !summary.sample_paths.contains_key(trait_path) {
            w.set(name, 1.0);
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

type MetricMap = HashMap<String, HashMap<String, f64>>;

fn merge_metric_summaries(summaries: &[FileSummary]) -> MetricMap {
    let mut merged: MetricMap = HashMap::new();
    for s in summaries {
        for (group, fields) in &s.metrics {
            let group_map = merged.entry(group.clone()).or_default();
            for (fname, &val) in fields {
                let e = group_map.entry(fname.clone()).or_insert(f64::NEG_INFINITY);
                *e = f64::max(*e, val);
            }
        }
    }
    merged
}

/// G7: structure.
fn write_structural_features(
    w: &mut FeatureWriter<'_>,
    summaries: &[FileSummary],
    filtered_finding_count: u32,
) {
    let binary_like = ["pe", "elf", "macho"];
    let mut any_tiny_binary = false;
    let mut max_entropy = 0.0_f64;
    let mut suspicious_files = 0;
    let mut hostile_files = 0;

    for s in summaries {
        if binary_like.contains(&s.file_type.as_str()) && s.size_bytes < 20_000.0 {
            any_tiny_binary = true;
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
    // Collimator sets this when every file with an imports key imports
    // nothing; the compact schema has no such key, so it is always 0.
    w.set("struct:no_imports", 0.0);
    w.set(
        "struct:zero_findings",
        f32::from(filtered_finding_count == 0),
    );
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

fn finding_paths(finding_id: &str, depth: usize) -> FindingPaths<'_> {
    let base = finding_id.split("::").next().unwrap_or(finding_id);
    FindingPaths {
        base,
        slashes: base.match_indices('/'),
        remaining: depth,
        finished: false,
    }
}

struct FindingPaths<'a> {
    base: &'a str,
    slashes: std::str::MatchIndices<'a, char>,
    remaining: usize,
    finished: bool,
}

impl<'a> Iterator for FindingPaths<'a> {
    type Item = &'a str;
    #[expect(
        clippy::string_slice,
        reason = "ASCII slash offsets are UTF-8 boundaries"
    )]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 || self.finished {
            return None;
        }
        self.remaining -= 1;
        let end = self.slashes.next().map_or_else(
            || {
                self.finished = true;
                self.base.len()
            },
            |(i, _)| i,
        );
        Some(&self.base[..end])
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

fn count_bucket(n: usize) -> &'static str {
    match n {
        0 => "0",
        1 => "1",
        2..=3 => "2_3",
        4..=7 => "4_7",
        8..=15 => "8_15",
        16..=31 => "16_31",
        32..=63 => "32_63",
        64..=127 => "64_127",
        _ => "128_plus",
    }
}

fn number_bucket(n: f64) -> String {
    if !n.is_finite() {
        return "nonfinite".into();
    }
    if n == 0.0 {
        return "zero".into();
    }
    let magnitude = n.abs();
    let bucket = if magnitude < 1.0 {
        "lt1"
    } else if magnitude < 10.0 {
        "1_9"
    } else if magnitude < 100.0 {
        "10_99"
    } else if magnitude < 1000.0 {
        "100_999"
    } else if magnitude < 10000.0 {
        "1k_9k"
    } else if magnitude < 1000000.0 {
        "10k_999k"
    } else {
        "1m_plus"
    };
    format!("{}_{bucket}", if n < 0.0 { "neg" } else { "pos" })
}

/// Emit only bounded categorical/shape tokens; the model selects its slots.
fn kv_tokens_with_options(
    metrics: Option<&serde_json::Value>,
    shape: bool,
    split: bool,
) -> Vec<String> {
    use serde_json::Value;
    let mut tokens = Vec::new();
    for (group, key, value) in metric_fields(metrics) {
        let base = format!("kv:{group}.{key}");
        if shape {
            tokens.push(format!("{base}:exists"));
        }
        match value {
            Value::Bool(b) => {
                tokens.push(format!("{base}={b}"));
                if shape {
                    tokens.push(format!("{base}:bool={b}"));
                }
            }
            Value::String(s) => {
                let val = normalize_vocab_token(s, 80);
                if !val.is_empty() {
                    tokens.push(format!("{base}={val}"));
                }
                if shape {
                    tokens.push(format!("{base}:strlen={}", count_bucket(s.chars().count())));
                    tokens.push(format!(
                        "{base}:{}",
                        if s.is_empty() { "empty" } else { "nonempty" }
                    ));
                }
                if split {
                    for part in s.split(|c: char| c.is_whitespace() || ",;:/\\|".contains(c)) {
                        let part = normalize_vocab_token(part, 64);
                        if !part.is_empty() {
                            tokens.push(format!("{base}=part:{part}"));
                        }
                    }
                }
            }
            Value::Number(n) if shape => {
                if let Some(n) = n.as_f64() {
                    tokens.push(format!("{base}:number={}", number_bucket(n)));
                }
            }
            Value::Array(items) if shape => {
                tokens.push(format!("{base}:len={}", count_bucket(items.len())));
                if items.is_empty() {
                    tokens.push(format!("{base}:empty"));
                }
            }
            Value::Object(items) if shape => {
                tokens.push(format!("{base}:len={}", count_bucket(items.len())));
                if items.is_empty() {
                    tokens.push(format!("{base}:empty"));
                }
            }
            Value::Null if shape => tokens.push(format!("{base}:empty")),
            _ => {}
        }
    }
    tokens
}

#[cfg(test)]
fn kv_tokens(metrics: Option<&serde_json::Value>) -> Vec<String> {
    kv_tokens_with_options(metrics, false, false)
}

/// A file's normalized symbols, sorted and unique. Mirrors `_file_symbols`
/// in collimator: an import contributes its bare name and its `lib!name`
/// form (or whichever half it has); exports, function names, AST call
/// targets and member chains contribute their name. Tokens shorter than two
/// characters are dropped.
fn file_symbols(facts: &cleave::types::CompactFacts) -> Vec<String> {
    let mut out = HashSet::new();
    let mut insert = |raw: &str| {
        let token = normalize_vocab_token(raw, 96);
        if token.chars().count() >= 2 {
            out.insert(token);
        }
    };
    for imp in &facts.imports {
        let (lib, name) = (imp.library.as_str(), imp.name.as_str());
        insert(name);
        match (lib.is_empty(), name.is_empty()) {
            (false, false) => insert(&format!("{lib}!{name}")),
            (true, false) => insert(name),
            (false, true) => insert(lib),
            (true, true) => {}
        }
    }
    for name in facts
        .exports
        .iter()
        .map(|e| e.name.as_str())
        .chain(facts.functions.iter().map(|f| f.name.as_str()))
        .chain(facts.targets.iter().map(String::as_str))
        .chain(facts.members.iter().map(String::as_str))
    {
        insert(name);
    }
    let mut symbols: Vec<String> = out.into_iter().collect();
    symbols.sort();
    symbols
}

/// Per-file caps from collimator (`_SYMBOL_BIGRAM_CAP`/`_SYMBOL_TRIGRAM_CAP`).
const SYMBOL_BIGRAM_CAP: usize = 64;
const SYMBOL_TRIGRAM_CAP: usize = 24;

/// Emit symbol:*, symbol_bi:*, symbol_tri:* for a set of summaries.
fn write_symbol_features(
    summaries: &[FileSummary],
    w: &mut FeatureWriter<'_>,
    emit_bigrams: bool,
    emit_trigrams: bool,
) {
    let mut key = String::new();
    for s in summaries {
        let sorted = s.symbols.as_slice();
        for sym in sorted {
            key.clear();
            let _ = write!(key, "symbol:{sym}");
            w.set_token(&key, 1.0);
        }
        if emit_bigrams {
            let cap = sorted.len().min(SYMBOL_BIGRAM_CAP);
            for i in 0..cap {
                for j in (i + 1)..cap {
                    key.clear();
                    let _ = write!(key, "symbol_bi:{}||{}", sorted[i], sorted[j]);
                    w.set_token(&key, 1.0);
                }
            }
        }
        if emit_trigrams {
            let cap = sorted.len().min(SYMBOL_TRIGRAM_CAP);
            for i in 0..cap {
                for j in (i + 1)..cap {
                    for k in (j + 1)..cap {
                        key.clear();
                        let _ = write!(
                            key,
                            "symbol_tri:{}||{}||{}",
                            sorted[i], sorted[j], sorted[k]
                        );
                        w.set_token(&key, 1.0);
                    }
                }
            }
        }
    }
}

fn known_metric_getter(name: &str) -> bool {
    KEY_METRICS
        .iter()
        .any(|(g, f, _)| name == format!("{g}_{f}"))
        || matches!(
            name,
            "pe_is_dotnet"
                | "pe_linker_major_version"
                | "pe_subsystem"
                | "pe_checksum_missing"
                | "pe_entry_section_nontext"
                | "pe_year_distance"
                | "pe_year_pre_2000"
                | "pe_year_future"
                | "text_mixed_indent"
                | "binary_has_overlay"
                | "binary_nonstandard_section_name_count"
                | "derived_string_per_function"
                | "derived_imports_per_dependency"
                | "derived_wide_string_ratio"
                | "derived_imports_per_kb"
                | "derived_sections_per_kb"
                | "derived_strings_per_kb"
                | "text_lines_over_200"
                | "text_lines_over_500"
                | "text_lines_over_1000"
                | "text_lines_in_200_499"
                | "text_lines_in_500_999"
        )
}

fn derived_metric(name: &str, metrics: &MetricMap, files: &[FileSummary], year: u32) -> f64 {
    let get = |group: &str, field: &str| {
        metrics
            .get(group)
            .and_then(|g| g.get(field))
            .copied()
            .unwrap_or(0.0)
    };
    let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
    match name {
        "pe_year_distance" => {
            if get("pe", "timestamp_year") > 0.0 {
                (f64::from(year) - get("pe", "timestamp_year")).abs()
            } else {
                0.0
            }
        }
        "pe_entry_section_nontext" => f64::from(files.iter().any(|f| f.entry_section_nontext)),
        "pe_year_pre_2000" => {
            f64::from(get("pe", "timestamp_year") > 0.0 && get("pe", "timestamp_year") < 2000.0)
        }
        "pe_year_future" => f64::from(get("pe", "timestamp_year") > 2030.0),
        "derived_string_per_function" => ratio(
            get("binary", "string_count"),
            get("binary", "function_count"),
        ),
        "derived_imports_per_dependency" => ratio(
            get("binary", "import_count"),
            get("binary", "dependency_count"),
        ),
        "derived_wide_string_ratio" => ratio(
            get("binary", "wide_string_count"),
            get("binary", "string_count"),
        ),
        "derived_imports_per_kb" => ratio(
            get("binary", "import_count"),
            get("binary", "file_size") / 1024.0,
        ),
        "derived_sections_per_kb" => ratio(
            get("binary", "section_count"),
            get("binary", "file_size") / 1024.0,
        ),
        "derived_strings_per_kb" => ratio(
            get("binary", "string_count"),
            get("binary", "file_size") / 1024.0,
        ),
        "text_lines_in_200_499" => {
            (get("text", "lines_over_200") - get("text", "lines_over_500")).max(0.0)
        }
        "text_lines_in_500_999" => {
            (get("text", "lines_over_500") - get("text", "lines_over_1000")).max(0.0)
        }
        _ => {
            let (g, f) = name.split_once('_').unwrap_or(("", ""));
            let value = get(g, f);
            if matches!(
                name,
                "pe_is_dotnet" | "pe_checksum_missing" | "text_mixed_indent" | "binary_has_overlay"
            ) {
                f64::from(value != 0.0)
            } else {
                value
            }
        }
    }
}

/// Cross-metric derived ratios. Mirrors collimator `_BATCH1_RATIOS`.
fn write_derived_metric_features(merged_metrics: &MetricMap, w: &mut FeatureWriter<'_>) {
    let get = |group: &str, field: &str| -> f64 {
        merged_metrics
            .get(group)
            .and_then(|g| g.get(field))
            .copied()
            .unwrap_or(0.0)
    };
    let ratio = |num: f64, denom: f64| -> f64 { if denom == 0.0 { 0.0 } else { num / denom } };
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
    w.set("agg:suspicious_bigram_count", (n_bi as f64).ln_1p() as f32);
    w.set(
        "agg:suspicious_trigram_count",
        (n_tri as f64).ln_1p() as f32,
    );
}

#[cfg(test)]
mod reader_equivalence_tests {
    use super::{FileSummary, RawNeeds};
    use cleave::types::{CompactReport, compact_from_files};

    /// Build a report with the shapes that actually vary: findings at several
    /// criticalities and confidences, imports, metrics, a nested member.
    fn sample_report() -> CompactReport {
        use cleave::types::{Criticality, FindingKind};
        let mut files = Vec::new();
        for (id, path, crit, conf) in [
            (0u32, "pkg.zip", Criticality::Notable, 0.9_f32),
            (1, "pkg.zip!!lib/net.py", Criticality::Suspicious, 0.7),
            (2, "pkg.zip!!lib/quiet.py", Criticality::Baseline, 0.5),
        ] {
            let mut fa = cleave::FileAnalysis {
                id,
                path: path.to_string(),
                file_type: "python".to_string(),
                sha256: format!("{id:064}"),
                size: 2048 * u64::from(id + 1),
                ..Default::default()
            };
            let mut f = cleave::types::Finding::new(
                "objectives/execution/shell::sh".to_string(),
                FindingKind::Capability,
                "spawns a shell".to_string(),
                conf,
            );
            f.crit = crit;
            f.attack = Some("T1059".into());
            fa.findings.push(f);
            if id > 0 {
                fa.depth = 1;
                fa.parent_id = Some(0);
            }
            files.push(fa);
        }
        compact_from_files(&files, cleave::types::Facts::Full)
    }

    /// The typed reader's per-file output, pinned. These values feed the model
    /// directly, so drift here moves ML output silently rather than failing —
    /// which is why they are asserted rather than reviewed.
    #[test]
    fn typed_reader_summarizes_every_file() {
        let report = sample_report();
        for needs in [RawNeeds::default(), RawNeeds::all()] {
            let summaries: Vec<FileSummary> = report
                .files
                .iter()
                .map(|f| FileSummary::from_compact(f, needs))
                .collect();
            assert_eq!(summaries.len(), report.files.len());
            for (summary, file) in summaries.iter().zip(&report.files) {
                assert_eq!(
                    summary.raw_findings.len(),
                    file.findings.len(),
                    "every finding on {} must reach the summary",
                    file.path
                );
                for (raw, finding) in summary.raw_findings.iter().zip(&file.findings) {
                    assert_eq!(raw.id, finding.id);
                    assert_eq!(raw.crit, u32::from(finding.criticality));
                    // The encoder omits `conf` at 0.5/0.0 and readers restore a
                    // missing `conf` as 1.0 — preserved deliberately, since it
                    // is what every deployed model was calibrated against.
                    assert_eq!(raw.conf, f64::from(finding.confidence));
                }
            }
        }
    }

    /// Imports live in the hand-encoded `facts` block, so the tokens derived
    /// from them are the likeliest thing to drift.
    #[test]
    fn typed_reader_derives_import_tokens() {
        let mut fa = cleave::FileAnalysis {
            id: 0,
            path: "app.exe".to_string(),
            file_type: "pe".to_string(),
            sha256: "d".repeat(64),
            size: 65536,
            ..Default::default()
        };
        fa.imports = vec![cleave::types::Import {
            symbol: "CreateProcessW".to_string(),
            library: Some("kernel32.dll".to_string()),
            ..Default::default()
        }];
        let report = compact_from_files(&[fa], cleave::types::Facts::Full);
        let summary = FileSummary::from_compact(&report.files[0], RawNeeds::all());
        // Both forms: LOGIC_GAPS matches the bare name, the vocab the qualified.
        assert!(summary.imports.contains("CreateProcessW"));
        assert!(summary.imports.contains("kernel32.dll!CreateProcessW"));
    }
}

#[cfg(test)]
mod import_token_tests {
    use super::import_tokens_from_parts;

    /// The tokens a `(library, name)` pair contributes. Both forms matter:
    /// LOGIC_GAPS matches bare names (`socket`, `CreateProcess`), while the
    /// symbol vocab carries the `lib!name` form.
    #[test]
    fn yields_bare_name_and_lib_qualified() {
        let t = import_tokens_from_parts("kernel32", "CreateProcessW");
        assert!(t.contains(&"CreateProcessW".to_string()));
        assert!(t.contains(&"kernel32!CreateProcessW".to_string()));
    }

    /// The `len >= 2` gate is applied per token, exactly as collimator applies
    /// it: a one-character *name* is dropped on its own but still contributes
    /// the qualified `lib!name` token, which does pass. Matching collimator
    /// here is the whole point — it defines the features the model was trained
    /// on, so "more sensible" behaviour would be a mismatch.
    #[test]
    fn length_gate_is_per_token_not_per_entry() {
        assert_eq!(import_tokens_from_parts("x", "a"), vec!["x!a"]);
        assert!(import_tokens_from_parts("", "").is_empty());
    }

    /// An ordinal-only import has no name; the library alone is the token.
    #[test]
    fn library_only_entry_falls_back_to_library() {
        assert_eq!(import_tokens_from_parts("ws2_32", ""), vec!["ws2_32"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn ranked_tokens_keep_strongest_with_stable_ties() {
        let mut summary = FindingSummary::default();
        for i in (0..600).rev() {
            let path = format!("path{i:04}");
            summary.sample_paths.insert(path.clone(), 3);
            summary.path_scores.insert(path, 3.0);
        }
        summary.sample_paths.insert("strong".into(), 5);
        summary.path_scores.insert("strong".into(), 4.5);
        for (order, limit) in [(2, 512), (3, 128), (4, 64)] {
            let tokens = ranked_tiered_path_tokens(&summary, 3, &HashMap::new(), order);
            let mut expected: Vec<_> = (0..limit - 1).map(|i| format!("n:path{i:04}")).collect();
            expected.push("h:strong".into());
            expected.sort();
            assert_eq!(tokens, expected);
        }
    }

    #[test]
    fn ranked_score_uses_paired_severity_and_confidence() {
        let summary = summarize_findings(&[
            FindingView {
                id: "a/b::one",
                crit: 5,
                conf: 0.7,
            },
            FindingView {
                id: "a/b::two",
                crit: 3,
                conf: 1.0,
            },
            FindingView {
                id: "a/b::low",
                crit: 5,
                conf: 0.1,
            },
        ]);
        assert_eq!(summary.path_scores["a/b"], 5.0 * f64::from(0.7_f32));
        assert_eq!(summary.path_scores["a"], 5.0 * f64::from(0.7_f32));
    }

    #[test]
    fn ranked_base_paths_select_before_vocabulary_filtering() {
        let findings: Vec<_> = (0..600)
            .rev()
            .map(|i| RawFinding {
                id: format!("path{i:04}::trait"),
                conf: 1.0,
                crit: 3,
                ..RawFinding::default()
            })
            .collect();
        assert_eq!(
            ranked_ngram_paths(&findings, 128),
            (0..128).map(|i| format!("path{i:04}")).collect::<Vec<_>>()
        );
    }

    fn paths(id: &str) -> Vec<&str> {
        finding_paths(id, default_trait_path_depth()).collect()
    }

    /// An ATT&CK ID whose first four bytes straddle a codepoint must not take
    /// the analysis task down with it. `atk` is copied straight from a finding
    /// and nothing validates it, so hostile or mojibake input reaches the
    /// tactic-prefix slice directly; in production this panicked 38 analyses
    /// with "end byte index 4 is not a char boundary".
    #[test]
    fn attack_tactic_prefix_tolerates_non_ascii_technique_id() {
        // T(0) ü(1..3) ─(3..6) — byte 4 lands inside the box char.
        let codes = ["T\u{fc}\u{2500}x".to_string(), "T1059.001".to_string()];
        let lookup: HashMap<String, usize> = [("agg:attack_tactic_count".to_string(), 0)]
            .into_iter()
            .collect();
        let mut vec = vec![0.0f32; 1];
        write_code_features(
            &FindingSummary::default(),
            &codes,
            &[],
            &mut FeatureWriter {
                vec: &mut vec,
                lookup: &lookup,
            },
        );
        // The unusable ID is skipped; the well-formed one still counts.
        assert_eq!(
            vec[0], 1.0,
            "expected only the ASCII technique to yield a tactic prefix"
        );
    }

    /// The ordinals the writers compare against are cleave's criticality ranks.
    #[test]
    fn criticality_ordinals_match_cleave_ranks() {
        use cleave::Criticality;
        assert_eq!(u32::from(Criticality::Baseline.rank()), CRIT_BASELINE);
        assert_eq!(u32::from(Criticality::Notable.rank()), CRIT_NOTABLE);
        assert_eq!(u32::from(Criticality::Suspicious.rank()), CRIT_SUSPICIOUS);
        assert_eq!(u32::from(Criticality::Hostile.rank()), CRIT_HOSTILE);
    }

    /// `kv:` tokens come from boolean and string metrics only, as in
    /// collimator with shape mode off.
    #[test]
    fn kv_tokens_cover_booleans_and_strings_only() {
        let metrics = serde_json::json!({
            "pe": {"signed": true, "subsystem": "  windows   gui ", "sections": 4},
            "text": "not a group",
        });
        let mut tokens = kv_tokens(Some(&metrics));
        tokens.sort();
        assert_eq!(tokens, ["kv:pe.signed=true", "kv:pe.subsystem=windows gui"]);
        assert!(kv_tokens(None).is_empty());
    }

    #[test]
    fn trait_depth_spec_defaults_and_reads_explicit_value() {
        let old: RawFeatureSpec = serde_json::from_str("{}").expect("legacy defaults");
        assert_eq!(old.trait_path_depth, 3);
        let new: RawFeatureSpec =
            serde_json::from_str(r#"{"trait_path_depth":5}"#).expect("depth field");
        assert_eq!(new.trait_path_depth, 5);
    }

    #[test]
    fn trait_depth_prefixes_are_configurable() {
        let id = "objectives/evasion/process/injection::method";
        assert_eq!(finding_paths(id, 1).collect::<Vec<_>>(), ["objectives"]);
        assert_eq!(
            finding_paths(id, 2).collect::<Vec<_>>(),
            ["objectives", "objectives/evasion"]
        );
        assert_eq!(
            finding_paths(id, 4).collect::<Vec<_>>(),
            [
                "objectives",
                "objectives/evasion",
                "objectives/evasion/process",
                "objectives/evasion/process/injection",
            ]
        );
        assert_eq!(
            finding_paths("métadata/format::x", 12).collect::<Vec<_>>(),
            ["métadata", "métadata/format"]
        );
    }

    #[test]
    fn renamed_metrics_resolve_for_numbers_and_categorical_tokens() {
        for &(old_group, old_key, new_group, new_key) in METRIC_RENAMES {
            for value in [
                serde_json::json!(5.27),
                serde_json::json!(true),
                serde_json::json!("signed"),
            ] {
                let mut metrics = serde_json::json!({});
                metrics[new_group] = serde_json::json!({new_key: value});
                let original = metrics.clone();
                let fields: Vec<_> = metric_fields(Some(&metrics)).collect();
                assert!(fields.contains(&(old_group, old_key, &value)));
                if value.is_number() || value.is_boolean() {
                    assert_eq!(
                        numeric_metrics(Some(&metrics))[old_group][old_key],
                        if value.is_boolean() { 1.0 } else { 5.27 }
                    );
                } else {
                    assert!(
                        kv_tokens(Some(&metrics))
                            .contains(&format!("kv:{old_group}.{old_key}=signed"))
                    );
                }
                assert!(
                    kv_tokens_with_options(Some(&metrics), true, false)
                        .contains(&format!("kv:{old_group}.{old_key}:exists"))
                );
                assert_eq!(metrics, original, "aliasing must not mutate the report");
            }
        }
    }

    #[test]
    fn historical_metrics_fill_v9_names_and_sibling_aliases() {
        for &(old_group, old_key, new_group, new_key) in METRIC_RENAMES {
            for value in [
                serde_json::json!(0),
                serde_json::Value::Null,
                serde_json::json!(false),
                serde_json::json!(5.27),
                serde_json::json!("signed"),
            ] {
                let metrics = serde_json::json!({old_group: {old_key: value}});
                let original = metrics.clone();
                let fields: Vec<_> = metric_fields(Some(&metrics)).collect();
                assert!(fields.contains(&(new_group, new_key, &value)));
                assert_eq!(metrics, original);
            }
        }
        let metrics = serde_json::json!({"ast": {"max_array_len": 0, "max_array_length": 12}});
        let fields: Vec<_> = metric_fields(Some(&metrics)).collect();
        assert!(fields.contains(&("ast", "array_literal_max_length", &serde_json::json!(12))));
        assert!(fields.contains(&("ast", "max_array_len", &serde_json::json!(0))));
    }

    #[test]
    fn metric_aliases_preserve_explicit_values_and_malformed_groups() {
        for value in [
            serde_json::json!(0),
            serde_json::Value::Null,
            serde_json::json!(false),
            serde_json::json!("old"),
        ] {
            let metrics = serde_json::json!({"file": {"entropy": 5.27}, "binary": {"overall_entropy": value}});
            let fields: Vec<_> = metric_fields(Some(&metrics)).collect();
            let old: Vec<_> = fields
                .iter()
                .filter(|(g, k, _)| *g == "binary" && *k == "overall_entropy")
                .collect();
            assert_eq!(old.len(), 1);
            assert_eq!(old[0].2, &value);
        }
        for group in [
            serde_json::Value::Null,
            serde_json::json!(3),
            serde_json::json!([]),
        ] {
            let metrics = serde_json::json!({"file": {"entropy": 5.27}, "binary": group});
            assert!(
                !metric_fields(Some(&metrics))
                    .any(|(g, k, _)| g == "binary" && k == "overall_entropy")
            );
        }
        assert_eq!(metric_fields(None).count(), 0);
    }

    #[test]
    fn member_metadata_is_independent_of_archive_depth() {
        let names = [
            "present:objectives",
            "maxcrit:objectives",
            "score:hopper_score",
            "score:density",
            "inter:javascript*score",
            "formula:skeleton_len",
            "formula:unique_elements",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let spec = anchor_spec(vec!["objectives".into()], vec![], vec![], names);
        let ctx = ExtractContext::new(&spec);
        for score in [0, 124] {
            for formula in [None, Some("O₂H₄")] {
                let mut file: cleave::types::CompactFile =
                    serde_json::from_value(serde_json::json!({
                        "id": 7, "sha": "member", "path": "pkg.vsix!!main.js",
                        "type": "javascript", "size": 2048, "risk": score,
                        "mol": formula,
                        "traits": [{"id": "objectives/execution::shell", "conf": 1.0, "crit": 4}]
                    }))
                    .expect("member fixture");
                let expected = ctx.extract_from_parsed(&ParsedReport::from_compact_files(
                    &[&file],
                    ctx.raw_needs(),
                ));
                for depth in [0, 1, 3] {
                    file.depth = depth;
                    let parsed = ParsedReport::from_compact_file(&file, ctx.raw_needs());
                    assert_eq!(parsed.sample_score, score);
                    assert_eq!(parsed.formula_str, formula.unwrap_or(""));
                    assert_eq!(
                        parsed.elements_str,
                        if formula.is_some() { "OH" } else { "" }
                    );
                    assert_eq!(ctx.extract_from_parsed(&parsed), expected, "depth {depth}");
                }
            }
        }
    }

    #[test]
    fn archive_depths_are_relative_order_independent_and_cycle_safe() {
        let root = FileSummary {
            id: Some(10),
            depth: 2,
            ..Default::default()
        };
        let child = FileSummary {
            id: Some(20),
            parent: Some(10),
            depth: 3,
            ..Default::default()
        };
        let nested = FileSummary {
            id: Some(30),
            parent: Some(20),
            ..Default::default()
        };
        assert_eq!(
            relative_depths(&[root.clone(), child.clone(), nested.clone()]),
            vec![0, 1, 2]
        );
        assert_eq!(
            relative_depths(&[nested, child.clone(), root]),
            vec![2, 1, 0]
        );
        assert_eq!(relative_depths(&[child]), vec![0]);
        let cycle = [
            FileSummary {
                id: Some(1),
                parent: Some(2),
                ..Default::default()
            },
            FileSummary {
                id: Some(2),
                parent: Some(1),
                ..Default::default()
            },
        ];
        assert_eq!(relative_depths(&cycle), vec![0, 0]);
        let chain: Vec<_> = (0..5000u32)
            .rev()
            .map(|id| FileSummary {
                id: Some(id),
                parent: id.checked_sub(1),
                ..Default::default()
            })
            .collect();
        assert_eq!(relative_depths(&chain)[0], 4999);
    }

    #[test]
    fn explicit_metric_transform_overrides_legacy_defaults() -> Result<()> {
        for (transform, expected) in [("identity", 9.0_f32), ("log1p_abs", 10.0_f32.ln())] {
            let mut file = tempfile::NamedTempFile::new()?;
            let value = serde_json::json!({
                "version":21,"abi_version":21,"presence_vocab":["metadata"],
                "feature_names":["metrics:text_max_inline_whitespace_run"],"total_features":1,
                "metric_transforms":{"text_max_inline_whitespace_run":transform},"standardized":false
            });
            serde_json::to_writer(&mut file, &value)?;
            let spec = FeatureSpec::load(file.path())?;
            let report = serde_json::from_value(serde_json::json!({"v":"8","files":[{
                "id":0,"path":"test.go","type":"go","sha":"fixture","size":100,
                "facts":{"metrics":{"text":{"max_inline_whitespace_run":9}}}
            }]}))?;
            assert!((ExtractContext::new(&spec).extract(&report)[0] - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn new_metric_contract_is_required_and_legacy_abi_defaults_to_version() -> Result<()> {
        for (version, transforms, succeeds) in [
            (20, None, true),
            (21, None, false),
            (21, Some("identity"), true),
            (21, Some("log10"), false),
        ] {
            let mut file = tempfile::NamedTempFile::new()?;
            let mut value = serde_json::json!({"version":version,"presence_vocab":["metadata"],
                "feature_names":["metrics:text_max_inline_whitespace_run"],"total_features":1,"standardized":false});
            if let Some(t) = transforms {
                value["metric_transforms"] =
                    serde_json::json!({"text_max_inline_whitespace_run":t});
            }
            serde_json::to_writer(&mut file, &value)?;
            let result = FeatureSpec::load(file.path());
            assert_eq!(result.is_ok(), succeeds, "{value}");
            if let Ok(spec) = result {
                assert_eq!(spec.abi_version, version);
            }
        }
        Ok(())
    }

    #[test]
    fn aggregate_metadata_still_comes_only_from_root() {
        let report: cleave::types::CompactReport = serde_json::from_value(
            serde_json::json!({"v": "8", "files": [
                {"id": 1, "sha": "member", "path": "pkg.vsix!!main.js", "type": "javascript", "size": 2048,
                 "depth": 1, "risk": 124, "mol": "O₂"},
                {"id": 0, "sha": "root", "path": "pkg.vsix", "type": "vsix", "size": 4096,
                 "depth": 0, "risk": 10, "mol": "H₄"}
            ]}),
        ).expect("aggregate fixture");
        let parsed = ParsedReport::from_compact_report(&report, RawNeeds::all(), None);
        assert_eq!(parsed.sample_score, 10);
        assert_eq!(parsed.formula_str, "H₄");
        assert_eq!(parsed.summaries.len(), 2);
        let keep = ["member".to_owned()].into_iter().collect();
        let filtered = ParsedReport::from_compact_report(&report, RawNeeds::all(), Some(&keep));
        assert_eq!(filtered.sample_score, 0);
        assert!(filtered.formula_str.is_empty());
        assert_eq!(filtered.summaries.len(), 1);
    }

    /// Unlike report parity, this exercises the entry point used to grade an
    /// embedded file. Expected vectors must come from its training DB row.
    #[test]
    #[ignore = "needs SCAN_MODELS_DIR with a member training extraction fixture"]
    fn member_features_match_training_fixture() -> Result<()> {
        #[derive(serde::Deserialize)]
        struct Sample {
            report: cleave::types::CompactReport,
            expected_features: Vec<f32>,
        }
        #[derive(serde::Deserialize)]
        struct Fixture {
            samples: Vec<Sample>,
        }
        let dir = std::path::PathBuf::from(std::env::var("SCAN_MODELS_DIR")?);
        let spec = FeatureSpec::load(&dir.join("feature_spec.json"))?;
        let fixture: Fixture =
            serde_json::from_slice(&std::fs::read(dir.join("extraction_fixture.json"))?)?;
        assert!(!fixture.samples.is_empty());
        let ctx = ExtractContext::new(&spec);
        for sample in fixture.samples {
            assert_eq!(
                sample.report.files.len(),
                1,
                "fixture must contain only the graded member"
            );
            let parsed = ParsedReport::from_compact_file(&sample.report.files[0], ctx.raw_needs());
            let actual = ctx.extract_from_parsed(&parsed);
            assert_eq!(actual.len(), sample.expected_features.len());
            for (i, (actual, expected)) in actual.iter().zip(&sample.expected_features).enumerate()
            {
                assert!(
                    actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= 1e-5,
                    "{}: serving={actual}, training={expected}",
                    spec.feature_names[i]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn trait_depth_is_per_route_for_shared_report() {
        let file: cleave::types::CompactFile = serde_json::from_value(serde_json::json!({
            "id": 0, "sha": "fixture", "path": "fixture", "type": "go", "size": 1,
            "traits": [{"id": "objectives/evasion/process/injection::method", "conf": 1.0, "crit": 4}]
        }))
        .expect("compact fixture");
        let names = vec![
            "present:objectives/evasion/process/injection".to_string(),
            "maxcrit:objectives/evasion/process/injection".to_string(),
            "tierbi:s:objectives + s:objectives/evasion/process/injection".to_string(),
            "tiertri:s:objectives + s:objectives/evasion + s:objectives/evasion/process/injection"
                .to_string(),
        ];
        let mut spec = anchor_spec(
            vec!["objectives/evasion/process/injection".to_string()],
            vec![],
            vec![],
            names,
        );
        let shared = ParsedReport::from_compact_file(&file, RawNeeds::all());
        spec.trait_path_depth = 4;
        let deep = ExtractContext::new(&spec);
        assert_eq!(deep.extract_from_parsed(&shared), [1.0, 4.0, 1.0, 1.0]);
        assert!(std::sync::Arc::ptr_eq(
            &shared.at_trait_path_depth(4),
            &shared.at_trait_path_depth(4)
        ));
        spec.trait_path_depth = 3;
        let default = ExtractContext::new(&spec);
        assert_eq!(default.extract_from_parsed(&shared), [0.0; 4]);
        assert_eq!(deep.extract_from_parsed(&shared), [1.0, 4.0, 1.0, 1.0]);
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

    /// `file_symbols` reads imports/exports/functions PLUS the
    /// filefacts AST symbol kinds (`tgt` = call targets, `mbr` = member
    /// chains). Tokens shorter than 2 chars are dropped to match collimator's
    /// `_file_symbols` threshold.
    #[test]
    fn file_symbols_picks_up_targets_and_members() {
        let file: cleave::types::CompactFile = serde_json::from_value(serde_json::json!({
            "id": 0,
            "path": "lib.rs",
            "type": "rust",
            "sha": "a".repeat(64),
            "size": 1024,
            "facts": {
                "imp": [["libc", "open"], ["libc", "x"]],   // "x" too short, skipped
                "tgt": ["client.get", "attempt.url().origin", "a"],
                "mbr": ["window.localStorage", "process.env.PATH"],
                "exp": [["exported_fn"]],
                "funcs": [["render"], ["x"]]                // "x" too short, skipped
            }
        }))
        .expect("fixture is a valid compact file");
        let syms = file_symbols(&file.facts);
        assert!(
            syms.is_sorted(),
            "symbols are sorted for the n-gram writers"
        );
        assert_eq!(
            FileSummary::from_compact(&file, RawNeeds::all()).symbols,
            syms
        );

        for expected in [
            "open",                 // import
            "libc!open",            // composite import
            "exported_fn",          // export
            "render",               // function
            "client.get",           // target
            "attempt.url().origin", // target
            "window.localStorage",  // member chain
            "process.env.PATH",     // member chain
        ] {
            assert!(
                syms.iter().any(|s| s == expected),
                "expected {expected:?} in symbol set, got {syms:?}"
            );
        }
        assert!(
            syms.iter().all(|s| s != "a" && s != "x"),
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
            metric_transforms: None,
            metric_getters: Vec::new(),
            metric_reference_year: 0,
            metric_correlations: HashMap::new(),
            crit_unigram_vocab: vec![],
            crit_bigram_vocab: vec![],
            crit_trigram_vocab: vec![],
            attack_bigram_vocab: vec![],
            attack_trigram_vocab: vec![],
            mbc_bigram_vocab: vec![],
            mbc_trigram_vocab: vec![],
            tiered_bigram_vocab: vec![],
            tiered_trigram_vocab: vec![],
            tiered_quadgram_vocab: vec![],
            kv_vocab: vec![],
            symbol_vocab: vec![],
            symbol_bigram_vocab: vec![],
            symbol_trigram_vocab: vec![],
            feature_names: vec![],
            total_features: 3,
            feature_means: Some(vec![0.0, 1.0, 2.0]),
            feature_stds: Some(vec![1.0, 2.0, 0.5]),
            standardized: true,
            trait_path_depth: default_trait_path_depth(),
            tiered_min_crit: default_tiered_min_crit(),
            tiered_branch_min_crit: HashMap::new(),
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
            metric_transforms: None,
            metric_getters: Vec::new(),
            metric_reference_year: 0,
            metric_correlations: HashMap::new(),
            crit_unigram_vocab: vec![],
            crit_bigram_vocab: vec![],
            crit_trigram_vocab: vec![],
            attack_bigram_vocab: vec![],
            attack_trigram_vocab: vec![],
            mbc_bigram_vocab: vec![],
            mbc_trigram_vocab: vec![],
            tiered_bigram_vocab: vec![],
            tiered_trigram_vocab: vec![],
            tiered_quadgram_vocab: vec![],
            kv_vocab: vec![],
            symbol_vocab: vec![],
            symbol_bigram_vocab: vec![],
            symbol_trigram_vocab: vec![],
            feature_means: None,
            feature_stds: None,
            standardized: false,
            trait_path_depth: default_trait_path_depth(),
            tiered_min_crit: default_tiered_min_crit(),
            tiered_branch_min_crit: HashMap::new(),
        }
    }

    #[test]
    fn offset_families_anchor_to_real_spec_positions() {
        // feature_names deliberately places every offset family somewhere a
        // hand-maintained running cursor would NOT land. Anchoring must follow
        // the spec, not the cursor — this is the exact drift that ran the
        // unsigned-bigram block off the end of the vector in production.
        let feature_names = vec![
            "filler:0".to_string(),              // 0
            "present:objectives".to_string(),    // 1
            "maxcrit:objectives".to_string(),    // 2
            "filler:1".to_string(),              // 3
            "bigrams:a + b".to_string(),         // 4
            "bigrams:a + c".to_string(),         // 5
            "trigram:a + b + c".to_string(),     // 6
            "unsigned_bigram:a + b".to_string(), // 7
            "unsigned_bigram:a + c".to_string(), // 8
        ];
        let spec = anchor_spec(
            vec!["objectives".to_string()],
            vec!["a + b".to_string(), "a + c".to_string()],
            vec!["a + b + c".to_string()],
            feature_names,
        );
        let ctx = ExtractContext::new(&spec);
        // Each family member resolves to its real feature_names slot by name,
        // wherever a running cursor would have landed.
        assert_eq!(ctx.present_slots, vec![Some(1)]);
        assert_eq!(ctx.maxcrit_slots, vec![Some(2)]);
        assert_eq!(ctx.bigram_slots, vec![Some(4), Some(5)]);
        assert_eq!(ctx.trigram_slots, vec![Some(6)]);
        assert_eq!(ctx.unsigned_bigram_slots, vec![Some(7), Some(8)]);
    }

    #[test]
    fn non_contiguous_offset_family_resolves_each_member_to_its_slot() {
        // The bigram family is laid out REVERSED vs vocab order. The old
        // `base + idx` scheme couldn't represent this and rejected the bundle;
        // the per-member slot map resolves each vocab entry to its true slot
        // (vocab idx 0 -> slot 1, idx 1 -> slot 0), so the bundle is accepted and
        // writes land correctly.
        let feature_names = vec![
            "bigrams:a + c".to_string(), // 0 — vocab idx 1
            "bigrams:a + b".to_string(), // 1 — vocab idx 0
        ];
        let spec = anchor_spec(
            vec![],
            vec!["a + b".to_string(), "a + c".to_string()],
            vec![],
            feature_names,
        );
        let ctx = ExtractContext::new(&spec);
        assert_eq!(ctx.bigram_slots, vec![Some(1), Some(0)]);
    }

    #[test]
    fn shared_vocab_subset_family_maps_only_present_members() {
        // present: and maxcrit: share presence_vocab, but an allowlist prune
        // dropped a maxcrit: name while keeping its present: counterpart. The
        // dropped entry maps to None (skipped); the kept ones resolve to their
        // true slots. Previously this misaligned `base + idx` and was rejected —
        // it is the exact real-world bundle that motivated the slot map.
        let feature_names = vec![
            "present:a".to_string(), // 0
            "present:b".to_string(), // 1
            "maxcrit:b".to_string(), // 2  (maxcrit:a was pruned)
        ];
        let spec = anchor_spec(
            vec!["a".to_string(), "b".to_string()],
            vec![],
            vec![],
            feature_names,
        );
        let ctx = ExtractContext::new(&spec);
        assert_eq!(ctx.present_slots, vec![Some(0), Some(1)]);
        assert_eq!(ctx.maxcrit_slots, vec![None, Some(2)]);
    }

    #[test]
    fn fully_absent_family_is_skipped_not_rejected() {
        // bigram_vocab is non-empty but feature_names has no "bigrams:"/
        // "unsigned_bigram:" entries: the family was pruned/disabled in the spec.
        // Every entry maps to None and the extractor skips it — a legitimately
        // smaller model, not an incompatibility, so validation passes.
        let spec = anchor_spec(
            vec![],
            vec!["a + b".to_string()],
            vec![],
            vec!["filler:0".to_string()],
        );
        let ctx = ExtractContext::new(&spec);
        assert_eq!(ctx.bigram_slots, vec![None]);
    }
}
