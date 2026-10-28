//! The feature spec a model was trained with: its vocabularies, groups and
//! layout, and the checks that a spec matches what this extractor emits.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// Key metrics extracted from the report's `metrics` object.
/// Each entry is (group, field, use_log1p).
///
/// `use_log1p` matches collimator's choice in `_TEXT_FULL_FIELDS` / `_BATCH1_OVERLAY`
/// (count/size fields get log1p; ratios/entropies do not). These entries are
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
    pub(super) version: u32,
    pub(super) abi_version: u32,
    pub(super) presence_vocab: Vec<String>,
    pub(super) filetype_vocab: Vec<String>,
    pub(super) element_vocab: Vec<String>,
    pub(super) bigram_vocab: Vec<String>,
    pub(super) ghost_vocab: Vec<String>,
    pub(super) skeleton_vocab: Vec<String>,
    pub(super) rare_element_vocab: Vec<String>,
    pub(super) trigram_vocab: Vec<String>,
    pub(super) metric_vocab: Vec<String>,
    pub(super) metric_transforms: Option<HashMap<String, String>>,
    pub(super) metric_getters: Vec<String>,
    pub(super) metric_reference_year: u32,
    pub(super) metric_correlations: HashMap<String, [[String; 2]; 2]>,
    pub(super) crit_unigram_vocab: Vec<String>,
    pub(super) crit_bigram_vocab: Vec<String>,
    pub(super) crit_trigram_vocab: Vec<String>,
    pub(super) attack_bigram_vocab: Vec<String>,
    pub(super) attack_trigram_vocab: Vec<String>,
    pub(super) mbc_bigram_vocab: Vec<String>,
    pub(super) mbc_trigram_vocab: Vec<String>,
    pub(super) tiered_bigram_vocab: Vec<String>,
    pub(super) tiered_trigram_vocab: Vec<String>,
    pub(super) tiered_quadgram_vocab: Vec<String>,
    pub(super) kv_vocab: Vec<String>,
    pub(super) symbol_vocab: Vec<String>,
    pub(super) symbol_bigram_vocab: Vec<String>,
    pub(super) symbol_trigram_vocab: Vec<String>,
    pub(super) feature_names: Vec<String>,
    pub(super) total_features: usize,
    pub(super) feature_means: Option<Vec<f32>>,
    pub(super) feature_stds: Option<Vec<f32>>,
    pub(super) standardized: bool,
    pub(super) trait_path_depth: usize,
    pub(super) tiered_min_crit: [u32; 3],
    pub(super) tiered_branch_min_crit: HashMap<String, u32>,
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct RawFeatureSpec {
    #[serde(default)]
    pub(super) version: u32,
    #[serde(default = "default_abi_version")]
    pub(super) abi_version: u32,
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
    #[serde(default = "default_trait_path_depth")]
    pub(super) trait_path_depth: usize,
    #[serde(default = "default_tiered_min_crit")]
    pub(super) tiered_min_crit: [u32; 3],
    #[serde(default)]
    pub(super) tiered_branch_min_crit: HashMap<String, u32>,
}

pub(super) const fn default_tiered_min_crit() -> [u32; 3] {
    [3; 3]
}

pub(super) fn default_trait_path_depth() -> usize {
    3
}

pub(super) const fn default_standardized() -> bool {
    true
}

pub(super) const fn default_abi_version() -> u32 {
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

    pub(super) fn validate(&self) -> Result<()> {
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
            // Refusing to load on unknown features would tie every
            // collimator-side feature innovation to a synchronous scan
            // update. That's the wrong trade: deploying a model with
            // dozens of unknown features that extract as zeros costs at most a
            // measurable accuracy delta; refusing to deploy costs the whole
            // model. So the load warns, naming the features, and
            // `atomscan validate` treats the same list as fatal; an operator
            // sees the degradation and decides whether to add extraction.
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
    pub(super) fn unknown_feature_names<'a>(&'a self, expected: &[String]) -> Vec<&'a str> {
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
    pub(super) fn expected_feature_names(&self) -> Vec<String> {
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
pub(super) const LOGIC_GAP_CATEGORIES: &[&str] = &["crypto", "network", "process"];

/// Intent gap categories (v16 group 22).
pub(super) const INTENT_GAP_CATEGORIES: &[&str] = &["network", "filesystem", "execution", "crypto"];
