//! Feature extraction from cleave's compact report.
//!
//! Produces the same feature vectors as the training pipeline's extractor,
//! driven by the vocabularies of the model's `feature_spec.json`.
//!
//! Feature assignment uses `FeatureWriter`, which maps feature names to the
//! slots of the spec's `feature_names` list. A name the spec does not carry
//! (a disabled group, or a vocabulary token the model never saw) writes
//! nothing — that is how pruned and partial specs "just work".

use std::collections::HashMap;

use self::names::FIXED_FEATURE_NAMES;

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

/// Newest feature spec version this build reads.
///
/// Each version adds to its predecessor: v21 requires `metric_transforms` and
/// measures archive structure relative to the sample; v22 adds the rule and
/// description keyword families over an explicit `base_feature_abi`; v23 adds
/// full-ID, pooled ID/description and decoded context keywords.
pub const EXPECTED_SPEC_VERSION: u32 = 23;

/// Earliest spec format this build parses: the deployed model's.
const MIN_LOADABLE_SPEC_VERSION: u32 = 20;
/// Stable model ABI version, one number with [`EXPECTED_SPEC_VERSION`].
pub const EXPECTED_MODEL_ABI_VERSION: u32 = EXPECTED_SPEC_VERSION;
/// Oldest model ABI this build loads.
const MIN_MODEL_ABI_VERSION: u32 = 20;
/// First ABI with explicit metric transforms and relative archive depths.
const METRIC_TRANSFORM_ABI: u32 = 21;
/// First ABI with keyword families: rule-ID and description words.
const KEYWORD_ABI: u32 = 22;
/// First ABI with the extended keyword families, which also reads a
/// finding with no description as empty text rather than refusing it.
const EXTENDED_KEYWORD_ABI: u32 = 23;

/// Minimum finding confidence for inclusion, as in training.
const MIN_CONFIDENCE: f64 = 0.65;

/// Criticality ordinals as a compact report carries them: the values of
/// [`cleave::Criticality::rank`], which features use as numbers. Not the enum
/// discriminants, which cleave does not promise to keep;
/// `criticality_ordinals_match_cleave_ranks` pins these to `rank`.
pub(crate) const CRIT_BASELINE: u32 = 2;
pub(crate) const CRIT_NOTABLE: u32 = 3;
pub(crate) const CRIT_SUSPICIOUS: u32 = 4;
pub(crate) const CRIT_HOSTILE: u32 = 5;

/// The path of a trait id — `objectives/exec/shell` of
/// `objectives/exec/shell::bash` — or the whole id when it names no rule.
#[must_use]
pub(crate) fn trait_path(id: &str) -> &str {
    id.split_once("::").map_or(id, |(path, _)| path)
}

mod context;
mod keywords;
mod names;
mod report;
mod spec;
mod structure;
mod write;

pub use self::context::ExtractContext;
pub(crate) use self::report::{ParsedReport, RawNeeds};
pub use self::spec::FeatureSpec;

#[cfg(test)]
mod reader_equivalence_tests;

#[cfg(test)]
mod import_token_tests;

#[cfg(test)]
mod tests;
