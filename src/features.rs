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

/// Newest feature spec version this build reads.
///
/// v21 requires `metric_transforms`. v20 retains the strongest tiered tokens
/// (older models keep overflow-drop semantics). v19 reads per-model
/// `trait_path_depth`, defaulting to depth 3. v18 adds `ast_depth_capped` to
/// the extended metrics, which are read by name from the spec vocab.
pub const EXPECTED_SPEC_VERSION: u32 = 21;

/// Earliest spec format this build parses. The real floor is the model ABI:
/// an unset `abi_version` defaults to the spec version, so a v16/v17 spec
/// loads only if it declares an ABI of at least [`MIN_MODEL_ABI_VERSION`].
pub const MIN_LOADABLE_SPEC_VERSION: u32 = 16;
/// Stable model ABI version shared with collimator.
/// Keep this in sync with EXPECTED_SPEC_VERSION for a single compatibility number.
pub const EXPECTED_MODEL_ABI_VERSION: u32 = EXPECTED_SPEC_VERSION;
/// Oldest model ABI this build loads.
pub const MIN_MODEL_ABI_VERSION: u32 = 18;

/// Minimum finding confidence for inclusion (matches collimator MIN_CONFIDENCE).
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
mod names;
mod report;
mod spec;
mod structure;
mod write;

pub(crate) use self::report::*;
pub use self::{context::*, spec::*};
#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use self::{names::*, structure::*, write::*};

#[cfg(test)]
mod reader_equivalence_tests;

#[cfg(test)]
mod import_token_tests;

#[cfg(test)]
mod tests;
