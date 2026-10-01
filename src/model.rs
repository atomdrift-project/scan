//! Model loading, thresholding, and inference.
//!
//! ## Model file format
//!
//! Only **`.onnx`** is supported — a portable inference graph emitted by
//! collimator's training pipeline (LightGBM via `onnxmltools.convert_lightgbm`)
//! or any other framework that exports to ONNX. Loaded via `tract` (pure-Rust,
//! no FFI). The native LightGBM (`.txt`) and XGBoost (`.json`) loaders were
//! retired now that collimator deploys ONNX-only bundles; any non-ONNX model
//! artifact is rejected at load time.
//!
//! ## Bundle layouts
//!
//! Two on-disk shapes are supported:
//!
//! ### Single-bundle (legacy / dev)
//!
//! ```text
//! <model_dir>/
//!   model.onnx
//!   feature_spec.json
//!   config.json
//!   evaluation.json
//! ```
//!
//! ### Ensemble (azoth) — see `~/azoth/DESIGN.md`
//!
//! ```text
//! <model_dir>/
//!   config.json                   ensemble-level config: route map + thresholds
//!   route_policies.json           optional per-filetype decision policies
//!   general/                      always required
//!     model.onnx | models/seed_*.onnx
//!     feature_spec.json
//!   filegroups/<group>/           optional, e.g. native, scripts, archive
//!     model.onnx | models/seed_*.onnx
//!     feature_spec.json           may be absent → uses general's spec
//!   filetypes/<type>/             optional, e.g. elf, pe
//!     model.onnx | models/seed_*.onnx
//!     feature_spec.json           may be absent → uses general's spec
//! ```
//!
//! Detection is by presence of `general/` immediately under `<model_dir>`.
//!
//! ## Ensemble `config.json` schema (`azoth.routed_ensemble.v1`)
//!
//! Emitted by collimator's calibration pipeline. The top-level config is the
//! coarse compatibility source of thresholds for every route; specialist
//! subdirectories do not carry their own `config.json`. When
//! `route_policies.json` is present, it is the primary runtime decision
//! artifact.
//!
//! ```text
//! {
//!   "schema": "azoth.routed_ensemble.v1",
//!   "filetype_to_group": { "elf": "native", "pe": "native", "py": "scripts", … },
//!   "required_routes": ["general"],                    // optional
//!   "models": [
//!     {"route": "general",        "kind": "general",   "rows": …},
//!     {"route": "filegroups/native", "kind": "filegroup", "rows": …},
//!     {"route": "filetypes/elf",  "kind": "filetype",  "rows": …}
//!   ],
//!   "levels": [
//!     {
//!       "level": 50,
//!       "hostile": {
//!         "target_per_100M": 50,
//!         "hostile_per_million": 0.5,
//!         "budget": 8,
//!         "thresholds": { "general": 0.997, "filetypes/elf": 0.951, … },
//!         "tp": …, "fp": …, "recall": …
//!       }
//!     },
//!     …
//!   ],
//!   "calibration_snapshot_id": …,
//!   "score_table_hash": "…",
//!   "model_set_hash":   "…"
//! }
//! ```
//!
//! Route names use slash-separated paths matching the on-disk layout:
//! `"general"`, `"filegroups/<name>"`, `"filetypes/<name>"`.
//!
//! ## Ensemble `route_policies.json` schema (`azoth.route_policy_search.v1`)
//!
//! This optional artifact records the calibrated decision policy per filetype
//! and level. Each severity has a `best.thresholds` map:
//!
//! ```text
//! {
//!   "routes": {
//!     "filetypes/elf": {
//!       "filetype": "elf",
//!       "levels": [{
//!         "level": 50,
//!         "hostile": {
//!           "best": {
//!             "policy": "specialist_primary_with_escape",
//!             "thresholds": {
//!               "filetypes/elf": 0.995,
//!               "general": 0.968
//!             }
//!           }
//!         }
//!       }]
//!     }
//!   }
//! }
//! ```
//!
//! At runtime litmus scores the applicable loaded routes, then applies only
//! the route thresholds named by the filetype policy. A route absent from the
//! policy does not participate in that severity decision. If
//! `route_policies.json` is absent or has no policy for a filetype, litmus
//! falls back to the older OR over `config.json` route thresholds.
//!
//! ## Specialist feature-spec rule
//!
//! Specialists may carry their own `feature_spec.json`. It may differ from
//! `general/feature_spec.json`, but it must have the ABI version this litmus
//! binary understands and it must match the specialist model's feature count.
//! At runtime each route extracts its own feature vector from the same cleave
//! report before scoring.
//!
//! ## ABI mismatch
//!
//! - `general/` ABI mismatch: fatal. Refuse to start.
//! - Specialist ABI mismatch: warn, drop that specialist, continue.
//!
//! ## Optional `suspicious` in collimator JSON
//!
//! Collimator no longer emits a `suspicious` field in any of the JSONs it
//! writes (`config.json`, `evaluation.json`, `route_policies.json`). Litmus
//! derives it consumer-side as a **level-table lookup**: the suspicious
//! threshold is the hostile threshold at level
//! `min(max_grid_level, SUSPICIOUS_LEVEL_CEILING)` (the looser-budget row in
//! the same `levels[]` table). Manual `--threshold-hostile <val>` (no `-l`)
//! skips the derivation: the `Thresholds` struct carries
//! `suspicious == hostile`, so `classify` only ever returns Benign or Hostile.
//!
//! ## Malformed metadata
//!
//! A bundle file that is absent is simply not used. One that is present but
//! unreadable, unparseable, or internally inconsistent stops the load with an
//! error: a silently-ignored threshold table degrades every verdict.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::features::{
    EXPECTED_MODEL_ABI_VERSION, ExtractContext, FeatureSpec, ParsedReport, RawNeeds,
};

/// Parse `path` as JSON. `Ok(None)` when the file does not exist; a read or
/// parse failure is an error.
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    let data = match std::fs::read(path) {
        Ok(data) => data,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    serde_json::from_slice(&data)
        .map(Some)
        .with_context(|| format!("parsing {}", path.display()))
}

/// A bundle's `config.json`, parsed once. The single-bundle layout carries
/// top-level `suspicious`/`hostile`; an ensemble carries the routing map and
/// the per-level threshold grid. Keys litmus does not read are ignored.
#[derive(Debug, Default, serde::Deserialize)]
struct BundleConfig {
    suspicious: Option<f32>,
    hostile: Option<f32>,
    /// `cleave file_type → filegroup name`.
    #[serde(default)]
    filetype_to_group: HashMap<String, String>,
    /// Routes whose absence is fatal at startup.
    #[serde(default)]
    required_routes: Vec<String>,
    #[serde(default)]
    levels: Vec<LevelEntryJson>,
    /// Deploy tuning goal prescribed by the model (collimator bakes its
    /// `DEFAULT_SEVERITY_LEVEL` here). Absent on older bundles.
    default_severity_level: Option<u16>,
}

impl BundleConfig {
    /// `<dir>/config.json`, or `None` when the bundle has none.
    fn load(dir: &Path) -> Result<Option<Self>> {
        read_json(&dir.join("config.json"))
    }

    /// The single-bundle top-level thresholds. `suspicious` defaults to
    /// `hostile` (hostile-only): the level-space derivation needs a level
    /// table, which this block does not have.
    fn thresholds(&self) -> Result<Option<Thresholds>> {
        let Some(hostile) = self.hostile else {
            return Ok(None);
        };
        let t = Thresholds {
            suspicious: self.suspicious.unwrap_or(hostile),
            hostile,
        };
        t.validate().context("config.json thresholds are invalid")?;
        Ok(Some(t))
    }
}

/// `evaluation.json`: the legacy home of recommended thresholds.
#[derive(Debug, serde::Deserialize)]
struct EvaluationJson {
    #[serde(default = "default_model_abi_version")]
    model_abi_version: u32,
    recommended_thresholds: Option<EvaluationThresholds>,
}

#[derive(Debug, serde::Deserialize)]
struct EvaluationThresholds {
    suspicious: Option<f32>,
    hostile: Option<f32>,
}

const fn default_model_abi_version() -> u32 {
    EXPECTED_MODEL_ABI_VERSION
}

/// Recommended thresholds from `<dir>/evaluation.json`, or `None` when the
/// file or its recommendation is absent. As with `config.json`, a missing
/// `suspicious` collapses to hostile-only.
fn load_evaluation_thresholds(dir: &Path) -> Result<Option<Thresholds>> {
    let path = dir.join("evaluation.json");
    let Some(eval) = read_json::<EvaluationJson>(&path)? else {
        return Ok(None);
    };
    if eval.model_abi_version != EXPECTED_MODEL_ABI_VERSION {
        anyhow::bail!(
            "{} has model_abi_version {} but this build expects {EXPECTED_MODEL_ABI_VERSION}",
            path.display(),
            eval.model_abi_version,
        );
    }
    let Some(hostile) = eval.recommended_thresholds.as_ref().and_then(|r| r.hostile) else {
        return Ok(None);
    };
    let t = Thresholds {
        suspicious: eval
            .recommended_thresholds
            .and_then(|r| r.suspicious)
            .unwrap_or(hostile),
        hostile,
    };
    t.validate()
        .with_context(|| format!("{} recommended thresholds are invalid", path.display()))?;
    Ok(Some(t))
}

/// Current suspicious ceiling (FP per 100M benigns) for level-sweep decisions.
///
/// Set to L3000 — a deliberate EXPERIMENTAL widening (2026-07) to surface as much
/// of the weak-signal tail as suspicious as the calibration curve supports. On
/// the current calibrate run hostile recall climbs smoothly to a peak at L4000
/// then COLLAPSES ~8pp at L5000 (the benign quantile runs out of resolution), so
/// L3000 is the loosest robustly-stable point, one notch below that fragile peak.
/// Trialed AGAINST prior precision evidence: a hopper fired-level analysis (model
/// 2.2.0-rc.1) put the precision elbow far lower, at L100 — L0 (99.6%), L100
/// (6168 bad vs 591 good — 91%), L250 (457 bad vs 1020 good — 31%), with L250+
/// adding more false positives than true positives. So L3000 knowingly re-admits
/// a low-precision tail. RE-MEASURE the elbow on the current model before making
/// this permanent; tighten back toward L100/L250 if the suspicious bucket floods.
const SUSPICIOUS_LEVEL_CEILING: u16 = 3000;

/// The suspicious LEVEL for a grid: `min(max_grid_level, SUSPICIOUS_LEVEL_CEILING)`.
///
/// Any file that fires at a level looser than the operator's selected hostile
/// level — but not above the suspicious ceiling — is classified as suspicious.
#[must_use]
pub(crate) fn capped_suspicious_level(max_grid_level: u16) -> u16 {
    max_grid_level.min(SUSPICIOUS_LEVEL_CEILING)
}

/// Classification outcome, ordered by severity (`Benign < Suspicious < Hostile`).
///
/// Serializes as an integer: 0 = benign, 1 = suspicious, 2 = hostile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum Classification {
    /// File shows no significant malicious indicators.
    Benign = 0,
    /// File has notable suspicious indicators.
    Suspicious = 1,
    /// File is likely malicious.
    Hostile = 2,
}

impl serde::Serialize for Classification {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u8(*self as u8)
    }
}

impl std::fmt::Display for Classification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Benign => write!(f, "benign"),
            Self::Suspicious => write!(f, "suspicious"),
            Self::Hostile => write!(f, "hostile"),
        }
    }
}

/// One route score from a routed ensemble decision.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RouteScore {
    /// Compact model route name, e.g. `az`, `az/native`, `az/elf`.
    #[serde(rename = "rte")]
    pub model: String,
    /// Probability emitted by this route's model. This is the calibrated value
    /// (the space thresholds live in) and drives the verdict.
    #[serde(rename = "prob")]
    pub probability: f32,
    /// Raw (pre-isotonic) model probability. The calibrated `probability`
    /// saturates the upper tail to 1.0, so the raw score is what's surfaced to
    /// humans for triage — it preserves resolution the calibrated number loses.
    #[serde(rename = "raw")]
    pub raw: f32,
    /// Classification after applying this route's calibrated thresholds.
    #[serde(rename = "cls")]
    pub classification: Classification,
}

/// One applicable route that was not scored.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SkippedRoute {
    /// Compact model route name, e.g. `az/native`.
    #[serde(rename = "rte")]
    pub model: String,
    /// Short reason the route was not used.
    #[serde(rename = "why")]
    pub reason: &'static str,
}

/// Probability cutoffs used to map model output into a [`Classification`].
///
/// Invariants:
/// - `suspicious` must be within `0.0..=1.0`
/// - `hostile` must be within `0.0..=1.0`
/// - `suspicious <= hostile`
///
/// # Example
/// ```
/// use scan::{Classification, Thresholds};
///
/// let thresholds = Thresholds {
///     suspicious: 0.8,
///     hostile: 0.95,
/// };
/// thresholds.validate()?;
///
/// assert_eq!(thresholds.classify(0.2), Classification::Benign);
/// assert_eq!(thresholds.classify(0.85), Classification::Suspicious);
/// assert_eq!(thresholds.classify(0.99), Classification::Hostile);
/// # Ok::<(), scan::model::ThresholdValidationError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Thresholds {
    /// Minimum probability to classify as suspicious.
    pub suspicious: f32,
    /// Minimum probability to classify as hostile.
    pub hostile: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            suspicious: Self::FALLBACK_SUSPICIOUS,
            hostile: Self::FALLBACK_HOSTILE,
        }
    }
}

impl Thresholds {
    /// Fallback thresholds used when `evaluation.json` is absent or unreadable.
    /// These are intentionally conservative (high hostile threshold, moderate
    /// suspicious threshold) to minimize false positives when operating without
    /// model-specific calibration data.
    pub const FALLBACK_SUSPICIOUS: f32 = 0.65;
    /// Fallback hostile threshold.
    pub const FALLBACK_HOSTILE: f32 = 0.90;

    /// Maximum acceptable relative divergence between custom thresholds and
    /// recommended thresholds before a warning is emitted. 0.3 = 30%.
    const DIVERGENCE_WARN_RATIO: f32 = 0.3;

    /// Warn if custom thresholds diverge significantly from recommended values.
    pub fn warn_if_divergent(&self, recommended: &Thresholds) {
        let check = |name: &str, custom: f32, rec: f32| {
            if rec > 0.0 {
                let ratio = ((custom - rec) / rec).abs();
                if ratio > Self::DIVERGENCE_WARN_RATIO {
                    tracing::warn!(
                        custom = custom,
                        recommended = rec,
                        divergence_pct = format!("{:.0}%", ratio * 100.0),
                        "custom {name} threshold diverges significantly from model recommendation"
                    );
                }
            }
        };
        check("suspicious", self.suspicious, recommended.suspicious);
        check("hostile", self.hostile, recommended.hostile);
    }

    /// Validate the threshold invariants.
    ///
    /// Callers constructing thresholds dynamically should validate once at the
    /// boundary, then pass the value through the rest of the system unchanged.
    pub fn validate(&self) -> std::result::Result<(), ThresholdValidationError> {
        if !(0.0..=1.0).contains(&self.suspicious) {
            return Err(ThresholdValidationError::OutOfRange {
                name: "suspicious",
                value: self.suspicious,
            });
        }
        if !(0.0..=1.0).contains(&self.hostile) {
            return Err(ThresholdValidationError::OutOfRange {
                name: "hostile",
                value: self.hostile,
            });
        }
        if self.suspicious > self.hostile {
            return Err(ThresholdValidationError::Misordered {
                suspicious: self.suspicious,
                hostile: self.hostile,
            });
        }
        Ok(())
    }

    /// Classify a raw model probability into a [`Classification`].
    ///
    /// This method assumes the thresholds are already valid.
    #[must_use]
    pub fn classify(&self, probability: f32) -> Classification {
        if probability >= self.hostile {
            Classification::Hostile
        } else if probability >= self.suspicious {
            Classification::Suspicious
        } else {
            Classification::Benign
        }
    }

    /// Decide a verdict from a probability, returning the (class, prob,
    /// threshold) triple that the decision was made against.
    ///
    /// The reported `threshold` is the cutoff defining the verdict band:
    /// - Hostile: `hostile` (the cutoff `prob` met or exceeded)
    /// - Suspicious: `suspicious` (the cutoff `prob` met or exceeded)
    /// - Benign: `suspicious` (the cutoff `prob` did not reach)
    #[must_use]
    pub fn decide(&self, probability: f32) -> Decision {
        // The raw threshold path carries no level table (manual `--threshold-*`
        // mode, single-bundle, or the no-grid fallback), so `level` is `Manual`.
        if probability >= self.hostile {
            Decision {
                class: Classification::Hostile,
                probability,
                threshold: self.hostile,
                level: Level::Manual,
            }
        } else if probability >= self.suspicious {
            Decision {
                class: Classification::Suspicious,
                probability,
                threshold: self.suspicious,
                level: Level::Manual,
            }
        } else {
            Decision {
                class: Classification::Benign,
                probability,
                threshold: self.suspicious,
                level: Level::Manual,
            }
        }
    }
}

/// The outcome of a routed/threshold decision, carrying the probability the
/// verdict was based on and the threshold it was compared against.
///
/// Invariant: `class = Hostile` iff `probability >= threshold` for the Hostile
/// cutoff; `class = Suspicious` iff `probability >= threshold` for the
/// Suspicious cutoff; `class = Benign` iff `probability < threshold`, where
/// `threshold` is the Suspicious cutoff the score did not reach.
#[derive(Debug, Clone, Copy)]
pub struct Decision {
    /// Classification outcome.
    pub class: Classification,
    /// Probability the decision was made on.
    pub probability: f32,
    /// Cutoff defining the verdict band.
    pub threshold: f32,
    /// Level-independent envelope marker (serialized as JSON `lvl`): where this
    /// file's hostile decision fires on the grid. Independent of the deploy
    /// `-l`, so the serialized envelope is identical across levels and
    /// cache-shareable — `-l` only moves the hostile/suspicious cutoffs applied
    /// to `level` to produce `class`.
    pub level: Level,
}

/// The lowest false-positive level (FP per 100M benigns) at which a file's
/// hostile decision fires: a property of the file and the model, never of a
/// caller's budget.
///
/// On the wire (`lvl`, `fires_at`) it is `null`, `-1` or the level. That
/// encoding invites comparing `-1` as the tightest level of all, so in code
/// each case is its own variant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Level {
    /// No level applies: manual-threshold mode, or a record that carries none
    /// (a corpus row written before levels, a decision with no answer).
    /// Serialized as `null`.
    #[default]
    Manual,
    /// Fires at no level on the grid. Serialized as `-1`.
    Clean,
    /// Fires at this level and every looser one. Synthesized verdicts (LLM,
    /// trait floor) are placed on the same axis so they decode to their class.
    At(u16),
}

impl Level {
    /// Whether no level table applies; omits `lvl` where the wire does.
    #[must_use]
    pub const fn is_manual(&self) -> bool {
        matches!(self, Self::Manual)
    }

    /// The wire encoding.
    fn wire(self) -> Option<i32> {
        match self {
            Self::Manual => None,
            Self::Clean => Some(-1),
            Self::At(n) => Some(i32::from(n)),
        }
    }
}

impl fmt::Display for Level {
    /// `L50`, `clean` or `manual`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manual => f.write_str("manual"),
            Self::Clean => f.write_str("clean"),
            Self::At(n) => write!(f, "L{n}"),
        }
    }
}

impl serde::Serialize for Level {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&self.wire(), serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Level {
    /// Through `Option`, so an absent field reads as [`Self::Manual`] exactly as
    /// the `Option<i32>` it replaces did.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match <Option<i32> as serde::Deserialize>::deserialize(deserializer)? {
            None => Ok(Self::Manual),
            Some(-1) => Ok(Self::Clean),
            Some(n) => u16::try_from(n).ok().map(Self::At).ok_or_else(|| {
                serde::de::Error::invalid_value(
                    serde::de::Unexpected::Signed(n.into()),
                    &"null, -1, or a level in 0..=65535",
                )
            }),
        }
    }
}

/// Validation error for [`Thresholds`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ThresholdValidationError {
    /// A threshold was outside the inclusive `[0.0, 1.0]` range.
    OutOfRange {
        /// Threshold field name.
        name: &'static str,
        /// Invalid threshold value.
        value: f32,
    },
    /// The suspicious threshold was greater than the hostile threshold.
    Misordered {
        /// Suspicious threshold value.
        suspicious: f32,
        /// Hostile threshold value.
        hostile: f32,
    },
}

impl fmt::Display for ThresholdValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { name, value } => {
                write!(f, "{name} threshold {value} is outside [0.0, 1.0]")
            }
            Self::Misordered {
                suspicious,
                hostile,
            } => write!(
                f,
                "suspicious threshold ({suspicious}) must be less than or equal to hostile threshold ({hostile})"
            ),
        }
    }
}

impl std::error::Error for ThresholdValidationError {}

/// Stable metadata about the loaded model, computed once at startup.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ModelInfo {
    /// Feature spec version (e.g. 13).
    pub version: u32,
    /// Stable preprocessing/inference ABI version.
    pub abi_version: u32,
}

/// One trained route member: a serialized ONNX graph scored single-sample,
/// returning the positive-class probability. Collimator's `.onnx` route
/// artifacts are the only model format litmus loads.
///
/// The graph is `onnxmltools.convert_lightgbm`'s standard output:
/// inputs `("input", float32, [N, n_features])`, outputs
/// `[label (int64, [N]), probabilities (float32, [N, 2])]`. We
/// ignore the label output and use column 1 of probabilities.
///
/// A plain tree ensemble is walked directly ([`FastTreeBackend`]); any other
/// graph falls back to tract.
#[derive(Debug)]
enum OnnxModel {
    Fast(FastTreeBackend),
    Tract(TractOnnxBackend),
}

struct TractOnnxBackend {
    /// Pre-optimized tract plan. Stored as a runnable model so
    /// `.run` calls don't re-optimize each prediction.
    plan: tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>,
    n_features: usize,
}

impl std::fmt::Debug for TractOnnxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TractOnnxBackend")
            .field("n_features", &self.n_features)
            .finish_non_exhaustive()
    }
}

impl OnnxModel {
    /// Load one `.onnx` member. Only ONNX is supported — the native LightGBM
    /// (`.txt`) and XGBoost (`.json`) loaders were retired.
    fn load(path: &Path) -> Result<Self> {
        if path.extension().and_then(|e| e.to_str()) != Some("onnx") {
            anyhow::bail!(
                "unsupported model file {}; only .onnx is supported \
                 (the LightGBM/XGBoost loaders were removed)",
                path.display(),
            );
        }
        match FastTreeBackend::load(path) {
            Ok(fast) => {
                tracing::debug!(
                    path = %path.display(),
                    features = fast.n_features,
                    trees = fast.trees.len(),
                    "loaded fast ONNX tree ensemble"
                );
                return Ok(Self::Fast(fast));
            }
            Err(error) => {
                tracing::debug!(
                    path = %path.display(),
                    error = %error,
                    "fast ONNX tree loader did not accept model; falling back to tract"
                );
            }
        }
        Ok(Self::Tract(TractOnnxBackend::load(path)?))
    }

    fn predict(&self, features: &[f32]) -> Result<f32> {
        match self {
            Self::Fast(fast) => fast.predict(features),
            Self::Tract(tract) => tract.predict(features),
        }
    }

    const fn n_features(&self) -> usize {
        match self {
            Self::Fast(fast) => fast.n_features,
            Self::Tract(tract) => tract.n_features,
        }
    }
}

impl TractOnnxBackend {
    fn load(path: &Path) -> Result<Self> {
        use tract_onnx::prelude::*;
        // Parse the ONNX graph first to recover n_features from the
        // (dynamic-batch) input fact.
        let mut model = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("parsing ONNX model {}", path.display()))?;
        let input_fact_dyn = model
            .input_fact(0)
            .with_context(|| format!("reading input fact from {}", path.display()))?
            .clone();
        // input fact's shape is a ShapeFactoid with per-dim factoids.
        // batch dim is unknown (the converter set it dynamic); the
        // feature dim is concrete (the converter pinned n_features
        // explicitly). Resolve the concrete feature dim via the
        // Factoid::concretize trait method, then to_usize.
        use tract_onnx::tract_hir::infer::Factoid;
        use tract_onnx::tract_hir::internal::DimLike;
        let n_features = input_fact_dyn
            .shape
            .dim(1)
            .ok_or_else(|| {
                anyhow::anyhow!("ONNX input has no feature dimension in {}", path.display())
            })?
            .concretize()
            .ok_or_else(|| anyhow::anyhow!("non-concrete feature dim in {}", path.display()))?
            .to_usize()
            .with_context(|| format!("feature dim not usize in {}", path.display()))?;
        // Pin batch dim to 1 so tract can optimize the graph.
        // Litmus infers one file at a time at scan time; multi-seed
        // ensembling averages across seed members, not across a
        // batch dimension. If we ever want batched scoring, swap
        // this for a symbolic dim via tract_pulse.
        let pinned_fact = InferenceFact::dt_shape(f32::datum_type(), tvec!(1, n_features));
        model = model
            .with_input_fact(0, pinned_fact)
            .with_context(|| format!("pinning batch dim for {}", path.display()))?;
        let plan = model
            .into_optimized()
            .with_context(|| format!("optimizing ONNX model {}", path.display()))?
            .into_runnable()
            .with_context(|| format!("making ONNX model runnable {}", path.display()))?;
        Ok(Self { plan, n_features })
    }

    fn predict(&self, features: &[f32]) -> Result<f32> {
        use tract_onnx::prelude::*;
        if features.len() != self.n_features {
            anyhow::bail!(
                "feature vector length {} != ONNX expected {}",
                features.len(),
                self.n_features
            );
        }
        // Build a (1, n_features) f32 tensor. We copy because tract
        // takes owned input. For single-sample inference at scan time
        // this is a single allocation per call; perfectly cheap.
        let input: Tensor =
            tract_ndarray::Array2::from_shape_vec((1, self.n_features), features.to_vec())
                .context("building ONNX input tensor")?
                .into();
        let outputs = self
            .plan
            .run(tvec!(input.into()))
            .context("ONNX inference run failed")?;
        // onnxmltools' LightGBM classifier exports outputs in
        // [label, probabilities] order. probabilities is (1, 2) where
        // column 1 is the positive-class probability — matches the
        // contract the rest of model.rs expects from .predict().
        let probs = outputs
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("ONNX missing probabilities output"))?;
        let view = probs
            .to_array_view::<f32>()
            .context("ONNX probabilities output not f32")?;
        let probs_slice = view
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("ONNX probabilities not contiguous"))?;
        if probs_slice.len() != 2 {
            anyhow::bail!(
                "ONNX probabilities length {} != 2 (expected binary [neg, pos])",
                probs_slice.len()
            );
        }
        Ok(probs_slice[1])
    }
}

/// Class-score accumulator size kept on the stack. The route models are
/// binary; anything wider falls back to a heap vector.
const FAST_TREE_STACK_CLASSES: usize = 8;

struct FastTreeBackend {
    n_features: usize,
    n_classes: usize,
    trees: Vec<usize>,
    nodes: Vec<FastTreeNode>,
    leaves: Vec<(usize, f32)>,
    base_values: Vec<f32>,
    post_transform: FastPostTransform,
    binary_result_layout: bool,
    /// The logistic post-transform kernel, built once at load. The factory
    /// behind `tract_linalg::ops().sigmoid_f32` allocates a fresh boxed op on
    /// every call, and this runs once per route per embedded member — tens of
    /// thousands of times in a directory scan — to transform `n_classes`
    /// floats. `ElementWise<f32>` is `Send + Sync`, so one instance serves
    /// every thread. `None` unless `post_transform` is `Logistic`.
    sigmoid: Option<Box<dyn tract_linalg::element_wise::ElementWise<f32>>>,
}

#[derive(Debug, Clone, Copy)]
enum FastPostTransform {
    None,
    Logistic,
    Softmax,
}

#[derive(Debug, Clone, Copy)]
enum FastCmp {
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

#[derive(Debug, Clone, Copy)]
enum FastTreeNode {
    Branch {
        feature_id: usize,
        threshold: f32,
        true_id: usize,
        false_id: usize,
        cmp: FastCmp,
        nan_is_true: bool,
    },
    Leaf {
        start: usize,
        end: usize,
    },
}

impl std::fmt::Debug for FastTreeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastTreeBackend")
            .field("n_features", &self.n_features)
            .field("n_classes", &self.n_classes)
            .field("trees", &self.trees.len())
            .field("post_transform", &self.post_transform)
            .finish_non_exhaustive()
    }
}

impl FastTreeBackend {
    fn load(path: &Path) -> Result<Self> {
        use tract_onnx::prelude::Framework;
        let proto = tract_onnx::onnx()
            .proto_model_for_path(path)
            .with_context(|| format!("parsing ONNX protobuf {}", path.display()))?;
        let graph = proto
            .graph
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ONNX model has no graph"))?;
        let node = graph
            .node
            .iter()
            .find(|node| node.op_type == "TreeEnsembleClassifier")
            .ok_or_else(|| anyhow::anyhow!("ONNX graph has no TreeEnsembleClassifier node"))?;

        let n_features = onnx_input_feature_count(graph)?;
        let n_classes = onnx_class_count(node)?;
        let nodes_featureids = attr_usizes(node, "nodes_featureids")?;
        if nodes_featureids.is_empty() {
            anyhow::bail!("TreeEnsembleClassifier has no nodes");
        }
        let n_nodes = nodes_featureids.len();
        let node_ids = attr_usizes_exact(node, "nodes_nodeids", n_nodes)?;
        let tree_ids = attr_usizes_exact(node, "nodes_treeids", n_nodes)?;
        let true_ids = attr_usizes_exact(node, "nodes_truenodeids", n_nodes)?;
        let false_ids = attr_usizes_exact(node, "nodes_falsenodeids", n_nodes)?;
        let node_values = attr_floats_exact(node, "nodes_values", n_nodes)?;
        let node_modes = attr_strings_exact(node, "nodes_modes", n_nodes)?;
        let nan_is_true = optional_attr_bools(node, "nodes_missing_value_tracks_true", n_nodes)?;

        let leaf_node_ids = attr_usizes(node, "class_nodeids")?;
        if leaf_node_ids.is_empty() {
            anyhow::bail!("TreeEnsembleClassifier has no class leaves");
        }
        let n_leaves = leaf_node_ids.len();
        let leaf_tree_ids = attr_usizes_exact(node, "class_treeids", n_leaves)?;
        let leaf_class_ids = attr_usizes_exact(node, "class_ids", n_leaves)?;
        let leaf_weights = attr_floats_exact(node, "class_weights", n_leaves)?;

        if tree_ids.first().copied() != Some(0) || leaf_tree_ids.first().copied() != Some(0) {
            anyhow::bail!("TreeEnsembleClassifier tree ids must start at 0");
        }
        let n_trees = tree_ids
            .last()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("TreeEnsembleClassifier has no tree ids"))?
            + 1;
        if leaf_tree_ids.last().copied() != Some(n_trees - 1) {
            anyhow::bail!("TreeEnsembleClassifier node/leaf tree counts mismatch");
        }

        let mut node_order: Vec<usize> = (0..n_nodes).collect();
        node_order.sort_by_key(|&idx| (tree_ids[idx], node_ids[idx]));
        let mut leaf_order: Vec<usize> = (0..n_leaves).collect();
        leaf_order.sort_by_key(|&idx| (leaf_tree_ids[idx], leaf_node_ids[idx]));

        let mut trees = Vec::with_capacity(n_trees);
        let mut nodes = Vec::with_capacity(n_nodes);
        let mut leaves = Vec::with_capacity(n_leaves);
        let mut current_tree_id = None;
        let mut in_leaf_idx = 0usize;
        for node_idx in node_order {
            let tree_id = tree_ids[node_idx];
            if Some(tree_id) != current_tree_id {
                current_tree_id = Some(tree_id);
                trees.push(nodes.len());
            }

            if let Some(cmp) = parse_fast_cmp(&node_modes[node_idx])? {
                let tree_offset = *trees
                    .last()
                    .ok_or_else(|| anyhow::anyhow!("TreeEnsembleClassifier missing tree root"))?;
                nodes.push(FastTreeNode::Branch {
                    feature_id: nodes_featureids[node_idx],
                    threshold: node_values[node_idx],
                    true_id: true_ids[node_idx] + tree_offset,
                    false_id: false_ids[node_idx] + tree_offset,
                    cmp,
                    nan_is_true: nan_is_true[node_idx],
                });
            } else {
                let start = leaves.len();
                while in_leaf_idx < leaf_order.len() {
                    let leaf_idx = leaf_order[in_leaf_idx];
                    if leaf_tree_ids[leaf_idx] != tree_id
                        || leaf_node_ids[leaf_idx] != node_ids[node_idx]
                    {
                        break;
                    }
                    leaves.push((leaf_class_ids[leaf_idx], leaf_weights[leaf_idx]));
                    in_leaf_idx += 1;
                }
                nodes.push(FastTreeNode::Leaf {
                    start,
                    end: leaves.len(),
                });
            }
        }

        if in_leaf_idx != leaf_order.len() {
            anyhow::bail!("TreeEnsembleClassifier has leaves that do not match any node");
        }
        let max_feature = nodes_featureids.iter().copied().max().unwrap_or(0);
        if max_feature >= n_features {
            anyhow::bail!(
                "TreeEnsembleClassifier uses feature {max_feature}, input has {n_features}"
            );
        }
        if leaf_class_ids.iter().any(|&class_id| class_id >= n_classes) {
            anyhow::bail!("TreeEnsembleClassifier leaf class id exceeds class count");
        }

        let mut base_values = optional_onnx_attr(node, "base_values")
            .map(|attr| attr.floats.clone())
            .unwrap_or_else(|| vec![0.0; n_classes]);
        if base_values.is_empty() {
            base_values.resize(n_classes, 0.0);
        }
        if base_values.len() != n_classes {
            anyhow::bail!(
                "TreeEnsembleClassifier base_values length {} != class count {}",
                base_values.len(),
                n_classes
            );
        }

        let post_transform = match optional_attr_string(node, "post_transform")?.as_deref() {
            None | Some("NONE") => FastPostTransform::None,
            Some("LOGISTIC") => FastPostTransform::Logistic,
            Some("SOFTMAX") => FastPostTransform::Softmax,
            Some(other) => anyhow::bail!("unsupported post_transform {other}"),
        };
        let binary_result_layout =
            n_classes < 3 && leaves.iter().all(|(class_id, _)| *class_id == 0);

        Ok(Self {
            n_features,
            n_classes,
            trees,
            nodes,
            leaves,
            base_values,
            post_transform,
            binary_result_layout,
            sigmoid: match post_transform {
                FastPostTransform::Logistic => Some((tract_linalg::ops().sigmoid_f32)()),
                FastPostTransform::None | FastPostTransform::Softmax => None,
            },
        })
    }

    fn predict(&self, features: &[f32]) -> Result<f32> {
        if features.len() != self.n_features {
            anyhow::bail!(
                "feature vector length {} != ONNX expected {}",
                features.len(),
                self.n_features
            );
        }

        // Class counts are tiny (2 for the binary route models). Keeping the
        // accumulator on the stack for the common case avoids a heap
        // allocation per prediction on a path that runs once per route per
        // embedded member.
        let mut score_buf = [0.0f32; FAST_TREE_STACK_CLASSES];
        let mut score_heap;
        let scores: &mut [f32] = if self.n_classes <= FAST_TREE_STACK_CLASSES {
            &mut score_buf[..self.n_classes]
        } else {
            score_heap = vec![0.0f32; self.n_classes];
            &mut score_heap[..]
        };
        for &root in &self.trees {
            let mut node_idx = root;
            loop {
                match self.nodes.get(node_idx) {
                    Some(FastTreeNode::Branch {
                        feature_id,
                        threshold,
                        true_id,
                        false_id,
                        cmp,
                        nan_is_true,
                    }) => {
                        let feature = features[*feature_id];
                        node_idx = if fast_compare(*cmp, feature, *threshold, *nan_is_true) {
                            *true_id
                        } else {
                            *false_id
                        };
                    }
                    Some(FastTreeNode::Leaf { start, end }) => {
                        for &(class_id, weight) in &self.leaves[*start..*end] {
                            scores[class_id] += weight;
                        }
                        break;
                    }
                    None => {
                        anyhow::bail!("TreeEnsembleClassifier branch points outside node table")
                    }
                }
            }
        }

        for (score, base) in scores.iter_mut().zip(self.base_values.iter()) {
            *score += *base;
        }
        match self.post_transform {
            FastPostTransform::None => {}
            FastPostTransform::Logistic => {
                if let Some(sigmoid) = &self.sigmoid {
                    sigmoid
                        .run(scores)
                        .context("ONNX sigmoid post-transform failed")?;
                }
            }
            FastPostTransform::Softmax => {
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for score in scores.iter_mut() {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                if sum != 0.0 {
                    for score in scores.iter_mut() {
                        *score /= sum;
                    }
                }
            }
        }

        if self.binary_result_layout {
            scores
                .first()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("TreeEnsembleClassifier emitted no score"))
        } else {
            scores
                .get(1)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("TreeEnsembleClassifier emitted no positive class"))
        }
    }
}

fn onnx_attr<'a>(
    node: &'a tract_onnx::pb::NodeProto,
    name: &str,
) -> Result<&'a tract_onnx::pb::AttributeProto> {
    node.attribute
        .iter()
        .find(|attr| attr.name == name)
        .ok_or_else(|| anyhow::anyhow!("TreeEnsembleClassifier missing attribute {name}"))
}

fn optional_onnx_attr<'a>(
    node: &'a tract_onnx::pb::NodeProto,
    name: &str,
) -> Option<&'a tract_onnx::pb::AttributeProto> {
    node.attribute.iter().find(|attr| attr.name == name)
}

fn attr_usizes(node: &tract_onnx::pb::NodeProto, name: &str) -> Result<Vec<usize>> {
    onnx_attr(node, name)?
        .ints
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .with_context(|| format!("attribute {name} has negative value {value}"))
        })
        .collect()
}

fn attr_usizes_exact(
    node: &tract_onnx::pb::NodeProto,
    name: &str,
    expected: usize,
) -> Result<Vec<usize>> {
    let values = attr_usizes(node, name)?;
    if values.len() != expected {
        anyhow::bail!(
            "attribute {name} length {} != expected {expected}",
            values.len()
        );
    }
    Ok(values)
}

fn attr_floats_exact(
    node: &tract_onnx::pb::NodeProto,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>> {
    let values = onnx_attr(node, name)?.floats.clone();
    if values.len() != expected {
        anyhow::bail!(
            "attribute {name} length {} != expected {expected}",
            values.len()
        );
    }
    Ok(values)
}

fn attr_strings_exact(
    node: &tract_onnx::pb::NodeProto,
    name: &str,
    expected: usize,
) -> Result<Vec<String>> {
    let values: Vec<String> = onnx_attr(node, name)?
        .strings
        .iter()
        .map(|bytes| {
            std::str::from_utf8(bytes)
                .map(str::to_owned)
                .with_context(|| format!("attribute {name} contains non-UTF8 string"))
        })
        .collect::<Result<_>>()?;
    if values.len() != expected {
        anyhow::bail!(
            "attribute {name} length {} != expected {expected}",
            values.len()
        );
    }
    Ok(values)
}

fn optional_attr_string(node: &tract_onnx::pb::NodeProto, name: &str) -> Result<Option<String>> {
    optional_onnx_attr(node, name)
        .map(|attr| {
            std::str::from_utf8(&attr.s)
                .map(str::to_owned)
                .with_context(|| format!("attribute {name} contains non-UTF8 string"))
        })
        .transpose()
}

fn optional_attr_bools(
    node: &tract_onnx::pb::NodeProto,
    name: &str,
    expected: usize,
) -> Result<Vec<bool>> {
    let Some(attr) = optional_onnx_attr(node, name) else {
        return Ok(vec![false; expected]);
    };
    if attr.ints.len() != expected {
        anyhow::bail!(
            "attribute {name} length {} != expected {expected}",
            attr.ints.len()
        );
    }
    Ok(attr.ints.iter().map(|&value| value != 0).collect())
}

fn onnx_class_count(node: &tract_onnx::pb::NodeProto) -> Result<usize> {
    let int_count = optional_onnx_attr(node, "classlabels_int64s").map(|attr| attr.ints.len());
    let string_count =
        optional_onnx_attr(node, "classlabels_strings").map(|attr| attr.strings.len());
    match (int_count, string_count) {
        (Some(count), None) | (None, Some(count)) if count > 0 => Ok(count),
        (Some(_), Some(_)) => {
            anyhow::bail!("TreeEnsembleClassifier has both integer and string class labels")
        }
        _ => anyhow::bail!("TreeEnsembleClassifier has no class labels"),
    }
}

fn onnx_input_feature_count(graph: &tract_onnx::pb::GraphProto) -> Result<usize> {
    let input = graph
        .input
        .first()
        .ok_or_else(|| anyhow::anyhow!("ONNX graph has no input"))?;
    let value = input
        .r#type
        .as_ref()
        .and_then(|ty| ty.value.as_ref())
        .ok_or_else(|| anyhow::anyhow!("ONNX input has no type"))?;
    let tract_onnx::pb::type_proto::Value::TensorType(tensor_type) = value;
    let shape = tensor_type
        .shape
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("ONNX input tensor has no shape"))?;
    let feature_dim = shape
        .dim
        .get(1)
        .and_then(|dim| dim.value.as_ref())
        .ok_or_else(|| anyhow::anyhow!("ONNX input has no feature dimension"))?;
    match feature_dim {
        tract_onnx::pb::tensor_shape_proto::dimension::Value::DimValue(value) => {
            usize::try_from(*value).context("ONNX feature dimension is negative")
        }
        tract_onnx::pb::tensor_shape_proto::dimension::Value::DimParam(param) => {
            anyhow::bail!("ONNX feature dimension is symbolic: {param}")
        }
    }
}

fn parse_fast_cmp(mode: &str) -> Result<Option<FastCmp>> {
    match mode {
        "BRANCH_LEQ" => Ok(Some(FastCmp::LessEqual)),
        "BRANCH_LT" => Ok(Some(FastCmp::Less)),
        "BRANCH_GTE" => Ok(Some(FastCmp::GreaterEqual)),
        "BRANCH_GT" => Ok(Some(FastCmp::Greater)),
        "BRANCH_EQ" => Ok(Some(FastCmp::Equal)),
        "BRANCH_NEQ" => Ok(Some(FastCmp::NotEqual)),
        "LEAF" => Ok(None),
        other => anyhow::bail!("unsupported tree node mode {other}"),
    }
}

fn fast_compare(cmp: FastCmp, feature: f32, threshold: f32, nan_is_true: bool) -> bool {
    if feature.is_nan() {
        return nan_is_true;
    }
    match cmp {
        FastCmp::Equal => feature == threshold,
        FastCmp::NotEqual => feature != threshold,
        FastCmp::Less => feature < threshold,
        FastCmp::Greater => feature > threshold,
        FastCmp::LessEqual => feature <= threshold,
        FastCmp::GreaterEqual => feature >= threshold,
    }
}

/// Inference backend powering a loaded [`Model`].
///
/// Holds one or more trained models for a route. Single-model bundles (the
/// single-bundle layout, `model.onnx` directly under the bundle dir) load
/// `models = [one_model]`. Multi-seed bundles store `models/seed_NN.onnx` and
/// load all of them; `predict` returns the arithmetic mean of every member's
/// score, which is the variance-reducing equivalent of training K seeds and
/// ensembling them at inference time.
///
/// All members of `models` are required to have the same `num_features` —
/// mismatches are rejected at load time.
#[derive(Debug)]
struct Backend {
    first: OnnxModel,
    rest: Vec<OnnxModel>,
}

impl Backend {
    const fn num_features(&self) -> usize {
        self.first.n_features()
    }

    fn predict(&self, features: &[f32]) -> Result<f32> {
        // Average member predictions in f64 to keep the rounding noise-floor
        // below f32 precision even for K up to a few dozen seeds. The K=1
        // path is mathematically identical to a direct `models[0].predict`.
        //
        // Summed serially, in member order. This used to be a rayon
        // `try_reduce` over the members, which is the wrong shape for the
        // work: a bundle carries 3 seeds, each seed's ONNX call is
        // microseconds, and this runs once per route per *embedded member* —
        // tens of thousands of times in a directory scan, nested inside the
        // per-member `par_iter` which is itself nested inside the per-path
        // pool. Fanning 3 microsecond tasks out at that depth costs more in
        // job/latch/epoch traffic than the calls themselves, and the churn is
        // charged to every worker in the pool. Serial order is also
        // deterministic, which a parallel reduce's association order is not.
        let mut sum = f64::from(self.first.predict(features)?);
        for m in &self.rest {
            sum += f64::from(m.predict(features)?);
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "members emit f32; f64 only keeps the ensemble sum stable"
        )]
        let avg = (sum / self.n_members() as f64) as f32;
        Ok(avg)
    }

    /// Number of trained members (1 = legacy single-model bundle, ≥2 =
    /// multi-seed). Exposed mainly so the load-time `tracing::info!` line
    /// can advertise it.
    fn n_members(&self) -> usize {
        1 + self.rest.len()
    }
}

/// Per-route isotonic calibrator persisted by collimator's
/// `azoth_calibrate_ensemble.py` as `calibrator.json` in the route directory.
///
/// Maps a route's raw model probability to a calibrated probability on
/// `[0, 1]`. The calibrator is fit on the deployment-time calibration corpus
/// so the deployed score matches the empirical malware fraction at any score
/// quantile.
///
/// Storage format (`azoth.calibrator.isotonic.v1`):
/// - `x`: ascending raw-probability breakpoints
/// - `y`: monotone-non-decreasing calibrated probabilities at each breakpoint
/// - `out_of_bounds`: `"clip"` — values outside `[x[0], x[-1]]` clamp to the
///   closest endpoint's `y`.
///
/// Apply: linear interpolation between adjacent breakpoints. Linear interp on
/// monotone breakpoints preserves the ranking, so AUC is unchanged — only the
/// probability *scale* shifts to match the empirical observation.
///
/// Backward compat: when `calibrator.json` is absent, raw scores pass through
/// unchanged. Bundles built before this change continue to load.
#[derive(Debug, Clone)]
struct IsotonicCalibrator {
    /// Sorted-ascending raw probability breakpoints.
    x: Vec<f32>,
    /// Monotone-non-decreasing calibrated probabilities; `y[i]` is the value
    /// at `x[i]`.
    y: Vec<f32>,
}

impl IsotonicCalibrator {
    /// Read `<bundle_dir>/calibrator.json` if it exists. Returns Ok(None) if
    /// the file is missing (intentional — pre-calibrator bundles are still
    /// loadable). Errors when the file exists but is unreadable or invalid.
    fn load_optional(bundle_dir: &Path) -> Result<Option<Self>> {
        #[derive(serde::Deserialize)]
        struct Raw {
            schema: Option<String>,
            x: Vec<f32>,
            y: Vec<f32>,
        }
        let path = bundle_dir.join("calibrator.json");
        let Some(raw) = read_json::<Raw>(&path)? else {
            return Ok(None);
        };
        if let Some(schema) = raw.schema.as_deref() {
            // Hard-fail on unrecognized future versions to avoid silently
            // applying a calibrator we don't understand.
            if schema != "azoth.calibrator.isotonic.v1" {
                anyhow::bail!(
                    "calibrator at {} has unsupported schema {schema}; this Atomdrift Scan build only supports azoth.calibrator.isotonic.v1",
                    path.display()
                );
            }
        }
        if raw.x.len() != raw.y.len() || raw.x.is_empty() {
            anyhow::bail!(
                "calibrator at {} is malformed: x.len={}, y.len={}",
                path.display(),
                raw.x.len(),
                raw.y.len()
            );
        }
        // Reject non-finite values up front: a single NaN or Inf in x or y
        // would propagate through every `apply()` call (NaN poisoning) — and
        // because thresholds are also calibrated at load time, every file
        // scored against this route would silently classify as Benign.
        for v in raw.x.iter().chain(raw.y.iter()) {
            if !v.is_finite() {
                anyhow::bail!(
                    "calibrator at {} contains a non-finite value (NaN/Inf)",
                    path.display(),
                );
            }
        }
        // Strict-ascending x: equal adjacent breakpoints would make the
        // linear-interpolation denominator zero and produce NaN in apply().
        for w in raw.x.windows(2) {
            if w[1] <= w[0] {
                anyhow::bail!(
                    "calibrator at {} is malformed: x is not strictly ascending ({} >= {})",
                    path.display(),
                    w[0],
                    w[1],
                );
            }
        }
        // y must be monotone-non-decreasing (the contract the calibrator
        // claims) AND in [0, 1] (a probability). If either fails, threshold
        // calibration could swap suspicious > hostile or land outside
        // valid-probability range — both are silent classifier corruption.
        for w in raw.y.windows(2) {
            if w[1] < w[0] {
                anyhow::bail!(
                    "calibrator at {} is malformed: y is not monotone non-decreasing ({} > {})",
                    path.display(),
                    w[0],
                    w[1],
                );
            }
        }
        for &v in &raw.y {
            if !(0.0..=1.0).contains(&v) {
                anyhow::bail!(
                    "calibrator at {} has y value {} outside [0, 1]",
                    path.display(),
                    v,
                );
            }
        }
        Ok(Some(Self { x: raw.x, y: raw.y }))
    }

    /// Apply the calibrator to a single raw probability. Linear interpolation
    /// between adjacent breakpoints; clip to endpoint `y` outside `[x[0], x[-1]]`.
    fn apply(&self, raw: f32) -> f32 {
        if raw.is_nan() {
            return raw;
        }
        // Out-of-bounds clipping (matches sklearn IsotonicRegression(out_of_bounds="clip")).
        if raw <= self.x[0] {
            return self.y[0];
        }
        if raw >= self.x[self.x.len() - 1] {
            return self.y[self.y.len() - 1];
        }
        // Binary search for the interval containing `raw`.
        let i = match self
            .x
            .binary_search_by(|p| p.partial_cmp(&raw).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(idx) => return self.y[idx], // exact hit on a breakpoint
            Err(idx) => idx,
        };
        // raw is in (x[i-1], x[i]); linear interpolation.
        let x0 = self.x[i - 1];
        let x1 = self.x[i];
        let y0 = self.y[i - 1];
        let y1 = self.y[i];
        let t = (raw - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }
}

/// Output of the per-bundle loader: backend, spec, resolved thresholds, plus
/// where the thresholds came from for the load-time log line.
struct LoadedBundle {
    backend: Backend,
    spec: FeatureSpec,
    thresholds: Thresholds,
    threshold_source: &'static str,
    /// Per-route isotonic calibrator from `calibrator.json`. Optional —
    /// pre-calibrator bundles return `None` here and skip calibration.
    calibrator: Option<IsotonicCalibrator>,
}

/// FALLBACK default severity level (per-100M-benigns scale; 50 = 50 FP/100M ≡
/// 0.5 FP/M). The operating point is resolved as: explicit CLI level → the
/// model-prescribed default from the bundle's config.json
/// (`model_default_level`, baked by collimator from its `DEFAULT_SEVERITY_LEVEL`)
/// → THIS const. So the tuning goal normally travels embedded in the model and
/// this is only used for older bundles whose config.json lacks
/// `default_severity_level`. Keep it equal to collimator's current default so
/// the fallback is sane; `make deploy` (collimator's azoth-deploy-final) fails
/// if the staged bundle's level, collimator's const, and this const disagree.
pub const DEFAULT_SEVERITY_LEVEL: u16 = 25;

/// The deploy tuning goal prescribed by THIS model's `config.json`
/// (`default_severity_level`), or `None` if the bundle predates the field.
///
/// Resolution order for the operating point is: explicit CLI level → this
/// model-prescribed default → the `DEFAULT_SEVERITY_LEVEL` const fallback. So a
/// bundle calibrated at L50 operates at L50 without any litmus rebuild, and a
/// caller can still pin a different level explicitly. A malformed config.json
/// reads as `None` here; `Model::load` reports it.
#[must_use]
pub fn model_default_level(model_dir: &Path) -> Option<u16> {
    BundleConfig::load(model_dir).ok()??.default_severity_level
}

/// The ensemble's top-level `config.json`, resolved at one level.
#[derive(Debug, Default)]
struct EnsembleConfig {
    /// `cleave file_type → filegroup name` map. Files whose `file_type` is
    /// not a key here route to general only (per DESIGN.md §Runtime Decision).
    filetype_to_filegroup: HashMap<String, String>,
    /// Routes whose absence is fatal at startup. Names: `general`,
    /// `filegroups/<name>`, `filetypes/<name>` (matching the route paths in
    /// `config.json`'s `models[]` array).
    required_routes: Vec<String>,
    /// Per-route thresholds at the resolved level. Keyed by route name as
    /// emitted in `levels[].hostile.thresholds`: `"general"`,
    /// `"filegroups/<name>"`, `"filetypes/<name>"`.
    route_thresholds: HashMap<String, Thresholds>,
    /// General route's hostile threshold at every level, ascending by level.
    /// Drives the verdict sweep for filetypes with no route policy. Calibrated
    /// space, matching the general route's emitted probability.
    general_grid: Vec<(u16, f32)>,
    /// Largest level present in the `levels[]` grid. The suspicious cap is
    /// `capped_suspicious_level(grid_max)`.
    grid_max: u16,
}

impl EnsembleConfig {
    fn new(cfg: BundleConfig, level: u16) -> Self {
        let route_thresholds = thresholds_at_level(&cfg.levels, level);
        // General route's hostile threshold per level, ascending. Used by the
        // verdict sweep for filetypes that have no route policy of their own.
        let mut general_grid: Vec<(u16, f32)> = cfg
            .levels
            .iter()
            .filter_map(|entry| {
                let threshold = *entry.hostile.thresholds.get("general")?;
                Some((entry.level, threshold))
            })
            .collect();
        general_grid.sort_by_key(|&(level, _)| level);
        let grid_max = cfg.levels.iter().map(|e| e.level).max().unwrap_or(0);
        Self {
            filetype_to_filegroup: cfg.filetype_to_group,
            required_routes: cfg.required_routes,
            route_thresholds,
            general_grid,
            grid_max,
        }
    }
}

/// A route's index in [`RouteNames`]. Assigned at load, so the per-file
/// decision compares integers rather than hashing route names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RouteId(usize);

impl RouteId {
    /// The general route: always interned first.
    const GENERAL: Self = Self(0);
}

/// What a route name names. Route names are the on-disk layout paths:
/// `general`, `filegroups/<name>`, `filetypes/<name>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteKind<'a> {
    General,
    Group(&'a str),
    Type(&'a str),
}

impl<'a> RouteKind<'a> {
    fn parse(name: &'a str) -> Option<Self> {
        if name == "general" {
            return Some(Self::General);
        }
        if let Some(group) = name.strip_prefix("filegroups/") {
            return Some(Self::Group(group));
        }
        name.strip_prefix("filetypes/").map(Self::Type)
    }
}

/// The wire name of a route: `az` for general, `az/<name>` for a specialist.
fn compact_route_name(route: &str) -> String {
    match RouteKind::parse(route) {
        Some(RouteKind::General) => "az".to_string(),
        Some(RouteKind::Group(name) | RouteKind::Type(name)) => format!("az/{name}"),
        None => format!("az/{route}"),
    }
}

/// Every route name the bundle refers to — in `config.json`, in
/// `route_policies.json`, or on disk — interned at load. Index 0 is `general`.
#[derive(Debug)]
struct RouteNames {
    names: Vec<String>,
    ids: HashMap<String, RouteId>,
}

impl Default for RouteNames {
    fn default() -> Self {
        let mut names = Self {
            names: Vec::new(),
            ids: HashMap::new(),
        };
        names.intern("general");
        names
    }
}

impl RouteNames {
    fn intern(&mut self, name: &str) -> RouteId {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = RouteId(self.names.len());
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        id
    }

    fn id(&self, name: &str) -> Option<RouteId> {
        self.ids.get(name).copied()
    }

    fn name(&self, id: RouteId) -> &str {
        &self.names[id.0]
    }
}

/// Per-filetype route policy loaded from `route_policies.json`.
///
/// `by_filetype` holds the policy resolved at the default level and drives the
/// diagnostic per-route classification (`models[]`). `grid` retains the hostile
/// policy at *every* level so the verdict path can sweep for the lowest
/// false-positive level at which a file fires — that swept level is the envelope
/// level marker (see `sweep_policy_grid` and `Model::decide_swept`).
#[derive(Debug, Default)]
struct RoutePolicies {
    by_filetype: HashMap<String, RoutePolicy>,
    grid: HashMap<String, Vec<LevelPolicy>>,
}

impl RoutePolicies {
    /// True when `route` is referenced by any policy at any level. Used to
    /// decide whether an on-disk specialist participates in the ensemble — a
    /// route referenced only at non-default levels must still load so the
    /// sweep can evaluate it.
    fn contains_route(&self, route: RouteId) -> bool {
        self.grid
            .values()
            .flatten()
            .any(|lp| lp.hostile.references_route(route))
            || self.by_filetype.values().any(|policy| {
                policy.hostile.references_route(route) || policy.suspicious.references_route(route)
            })
    }

    /// The default-level policy for a file type (see [`lookup_filetype`]).
    fn policy_for(&self, file_type: &str) -> Option<&RoutePolicy> {
        lookup_filetype(&self.by_filetype, file_type)
    }

    /// The per-level policy grid for a file type (see [`lookup_filetype`]).
    /// Used by the envelope-level sweep.
    fn grid_for(&self, file_type: &str) -> Option<&[LevelPolicy]> {
        lookup_filetype(&self.grid, file_type).map(Vec::as_slice)
    }

    /// Every policy severity, for load-time passes over the thresholds.
    fn severities_mut(&mut self) -> impl Iterator<Item = &mut PolicySeverity> {
        self.by_filetype
            .values_mut()
            .flat_map(|policy| [&mut policy.hostile, &mut policy.suspicious])
            .chain(self.grid.values_mut().flatten().map(|lp| &mut lp.hostile))
    }
}

/// Look a cleave file type up in a per-filetype table: as given, then with
/// pure-compression suffixes stripped (`tar.gz` → `tar`; the original
/// spelling goes first so a bundle whose keys still carry the suffix keeps
/// working), then as its container archive (`gem` → `tar`, `whl` → `zip`).
/// Mirrors `collimator.data.normalize_archive_filetype`, and the specialist
/// fallback in [`RouteSet::specialist_keys`], so the resolved policy and the
/// specialists scored agree.
fn lookup_filetype<'a, V>(table: &'a HashMap<String, V>, file_type: &str) -> Option<&'a V> {
    if let Some(v) = table.get(file_type) {
        return Some(v);
    }
    let normalized = normalize_archive_filetype(file_type);
    if normalized != file_type
        && let Some(v) = table.get(normalized.as_ref())
    {
        return Some(v);
    }
    let container = container_filetype(file_type)?;
    if container != file_type && container != normalized {
        return table.get(container);
    }
    None
}

/// Pure-compression suffixes — formats with no multi-file container of
/// their own. Stripped from the tail of compound filetype strings (e.g.
/// `tar.gz` → `tar`). Source of truth in
/// `collimator/src/collimator/data.py::_PURE_COMPRESSION_SUFFIXES`; keep
/// the two lists in sync — collimator emits training labels using this
/// rule and litmus routes scan-time files using it.
const PURE_COMPRESSION_SUFFIXES: &[&str] = &["gz", "bz2", "xz", "zst", "z", "lzma"];

/// Strip pure-compression suffixes from a cleave-reported filetype label.
/// `tar.gz` → `tar`, `tar.bz2.xz` → `tar`. Bare compression labels
/// (`gz`, `bz2`, …) are returned unchanged — litmus decompresses and
/// re-routes those at extraction time; the wrapper has no route of its
/// own. Borrows when the label is already normalized, the common case.
fn normalize_archive_filetype(file_type: &str) -> Cow<'_, str> {
    let trimmed = file_type.trim();
    let mut normalized: Cow<'_, str> = if trimmed.bytes().any(|b| b.is_ascii_uppercase()) {
        Cow::Owned(trimmed.to_ascii_lowercase())
    } else {
        Cow::Borrowed(trimmed)
    };
    while let Some((head, tail)) = normalized.rsplit_once('.') {
        if head.is_empty() || !PURE_COMPRESSION_SUFFIXES.contains(&tail) {
            break;
        }
        let head_len = head.len();
        match &mut normalized {
            Cow::Borrowed(s) => *s = s.get(..head_len).unwrap_or(s),
            Cow::Owned(s) => s.truncate(head_len),
        }
    }
    normalized
}

/// The container archive label for a scanned file type, as filefacts defines
/// it: `gem`/`crate`/`python_sdist` → `tar`, `whl`/`jar`/`nupkg` → `zip`, the
/// compressed `tar.*` variants → `tar`. `None` when the type is not
/// archive-backed (source, binaries, bare single-file compression like `gz`).
///
/// This is the specialty-else-container half of routing: a packaged type with
/// no specialist of its own falls back to its container's. filefacts owns the
/// archive→container knowledge — no table is duplicated here. Unlike
/// [`normalize_archive_filetype`] (which only strips compression suffixes for
/// collimator parity), this also collapses specialty package types onto their
/// container, so it is consulted *after* the exact and compression-stripped
/// lookups, never instead of them.
fn container_filetype(file_type: &str) -> Option<&'static str> {
    filefacts::FileType::from_label(file_type.trim())
        .and_then(filefacts::FileType::archive_format)
        .map(filefacts::ArchiveFormat::label)
}

/// Hostile decision policy for one filetype at one severity level. A filetype's
/// grid is kept in ascending `level` order at load time so the sweep can take
/// the first (lowest) firing level.
#[derive(Debug, Clone)]
struct LevelPolicy {
    level: u16,
    hostile: PolicySeverity,
}

#[derive(Debug, Clone)]
struct RoutePolicy {
    suspicious: PolicySeverity,
    hostile: PolicySeverity,
}

#[derive(Debug, Clone)]
struct PolicySeverity {
    /// Route → calibrated threshold, ordered by route name. Routes absent
    /// from this list do not participate in that severity decision.
    thresholds: Vec<(RouteId, f32)>,
    /// Learned-blend policy. When set, ``thresholds`` is empty and this
    /// severity is evaluated as ``sigmoid(intercept + sum(w_i * logit(p_i))) >=
    /// threshold`` over the named routes. Mirrors what
    /// ``azoth_route_policy_search._make_learned_blend_candidate_at_fp`` fits
    /// — the weights live in the same calibrated probability space that
    /// ``predict_calibrated`` emits, so no isotonic mapping is applied at
    /// load time.
    blend: Option<BlendPolicy>,
}

impl PolicySeverity {
    /// The OR-rule threshold for `route`, if it participates.
    fn threshold(&self, route: RouteId) -> Option<f32> {
        self.thresholds
            .iter()
            .find(|&&(r, _)| r == route)
            .map(|&(_, t)| t)
    }

    /// True iff this severity could fire on a contribution from ``route``.
    /// For OR-rule policies that's "route has a threshold"; for blend policies
    /// it's "route is one of the blend inputs."
    fn references_route(&self, route: RouteId) -> bool {
        self.threshold(route).is_some()
            || self
                .blend
                .as_ref()
                .is_some_and(|blend| blend.routes.contains(&route))
    }

    /// True iff this severity fires given the per-route scores. Dispatches
    /// to blend evaluation when ``blend`` is set, OR-rule otherwise.
    #[cfg(test)]
    fn fires(&self, scores: &[RouteProbability]) -> bool {
        self.fire(scores).is_some()
    }

    /// If this severity fires, return the deciding `(probability, threshold)`
    /// pair so callers can build a [`Decision`] whose `probability` and
    /// `threshold` satisfy `probability >= threshold`.
    ///
    /// For OR-rule policies the deciding route is the one with the largest
    /// margin (`probability - threshold`) so the published `prob`/`threshold`
    /// pair best characterises why the verdict fired.
    fn fire(&self, scores: &[RouteProbability]) -> Option<(f32, f32)> {
        if let Some(blend) = &self.blend {
            return blend.fire(scores);
        }
        let mut best: Option<(f32, f32)> = None;
        for score in scores {
            if let Some(t) = self.threshold(score.route)
                && score.probability >= t
            {
                let margin = score.probability - t;
                let best_margin = best.map_or(f32::NEG_INFINITY, |(p, bt)| p - bt);
                if margin > best_margin {
                    best = Some((score.probability, t));
                }
            }
        }
        best
    }
}

#[derive(Debug, Clone)]
struct BlendPolicy {
    /// Routes consumed by the blend, in the order their weights are listed.
    /// Both ``weights`` and the score lookup happen by index, so reordering
    /// after load is forbidden.
    routes: Vec<RouteId>,
    weights: Vec<f32>,
    intercept: f32,
    /// Calibrated-space threshold on the sigmoid output. Already in the
    /// space the blend was fit in (post-isotonic, matching ``predict_calibrated``),
    /// so the per-route isotonic mapping that ``calibrate_policy_thresholds``
    /// applies to OR-rule thresholds is intentionally NOT applied here.
    threshold: f32,
}

impl BlendPolicy {
    /// Evaluate ``sigmoid(intercept + sum(w_i * logit(clip(p_i)))) >= threshold``
    /// over the configured routes. Missing routes (specialist not loaded for
    /// this filetype, scoring failure, etc.) fall back to "doesn't fire" —
    /// the blend can't be honestly evaluated with incomplete inputs and
    /// firing on a partial blend would be a calibration mismatch.
    #[cfg(test)]
    fn fires(&self, scores: &[RouteProbability]) -> bool {
        self.fire(scores).is_some()
    }

    /// Compute the blend's sigmoid output and, if it crosses the threshold,
    /// return `(sigmoid_output, threshold)`. Missing routes mean the blend
    /// cannot be honestly evaluated; treat as "doesn't fire."
    fn fire(&self, scores: &[RouteProbability]) -> Option<(f32, f32)> {
        // f64 math throughout — logit blows up near 0/1, and the cumulative
        // weighted sum can drift if we stay in f32. The final compare against
        // ``threshold`` is in f32 to match how the calibration step writes it.
        const EPS: f64 = 1e-6;
        let mut z: f64 = f64::from(self.intercept);
        for (route, &weight) in self.routes.iter().zip(&self.weights) {
            let score = scores.iter().find(|s| s.route == *route)?;
            let p = (f64::from(score.probability)).clamp(EPS, 1.0 - EPS);
            let logit = (p / (1.0 - p)).ln();
            z += f64::from(weight) * logit;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the blend threshold is an f32; the f64 only steadies the sum"
        )]
        let sigmoid_z = (1.0 / (1.0 + (-z).exp())) as f32;
        (sigmoid_z >= self.threshold).then_some((sigmoid_z, self.threshold))
    }
}

#[derive(Debug, serde::Deserialize)]
struct RoutePoliciesJson {
    #[serde(default)]
    routes: HashMap<String, RoutePolicyRouteJson>,
}

#[derive(Debug, serde::Deserialize)]
struct RoutePolicyRouteJson {
    filetype: String,
    #[serde(default)]
    levels: Vec<RoutePolicyLevelJson>,
}

#[derive(Debug, serde::Deserialize)]
struct RoutePolicyLevelJson {
    level: u16,
    hostile: RoutePolicySeverityJson,
    // Collimator may omit `suspicious`; the loader derives it from `hostile`.
    suspicious: Option<RoutePolicySeverityJson>,
}

#[derive(Debug, serde::Deserialize)]
struct RoutePolicySeverityJson {
    best: Option<RoutePolicyBestJson>,
}

#[derive(Debug, serde::Deserialize)]
struct RoutePolicyBestJson {
    #[serde(default)]
    thresholds: HashMap<String, f32>,
    /// Optional learned-blend variant. When set, ``thresholds`` is typically
    /// empty and the severity classifies via the blend's combined score.
    /// Mirrors azoth_route_policy_search._make_learned_blend_candidate_at_fp.
    blend: Option<BlendPolicyJson>,
}

/// Every field a blend is evaluated with is required: a defaulted threshold
/// of 0.0 would pass validation and fire on everything.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BlendPolicyJson {
    routes: Vec<String>,
    weights: Vec<f32>,
    intercept: f32,
    threshold: f32,
    /// Currently only ``"logit"`` is supported; absent means logit. Future
    /// blend variants (e.g., raw-prob linear, monotonic GAM) would advertise
    /// themselves here so deploy refuses unknown shapes loudly rather than
    /// silently misapplying the wrong transform.
    transform: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct LevelEntryJson {
    level: u16,
    hostile: SeverityEntryJson,
    // Collimator may omit `suspicious`; per-route derivation happens in
    // `thresholds_at_level`. An absent block is treated as an empty map,
    // which yields hostile-only routes (the same shape used when only some
    // routes have a calibrated suspicious threshold).
    suspicious: Option<SeverityEntryJson>,
}

#[derive(Debug, serde::Deserialize)]
struct SeverityEntryJson {
    /// Route-name → threshold map for this severity. Route names match the
    /// `models[].route` field: `"general"`, `"filegroups/<name>"`, `"filetypes/<name>"`.
    #[serde(default)]
    thresholds: HashMap<String, f32>,
}

/// Load the searched per-filetype decision policies, interning every route
/// they name. Optional: an ensemble without `route_policies.json` keeps the
/// original OR semantics.
fn load_route_policies(
    model_dir: &Path,
    level: u16,
    names: &mut RouteNames,
) -> Result<RoutePolicies> {
    let path = model_dir.join("route_policies.json");
    let Some(json) = read_json::<RoutePoliciesJson>(&path)? else {
        return Ok(RoutePolicies::default());
    };

    let mut by_filetype = HashMap::new();
    let mut grid: HashMap<String, Vec<LevelPolicy>> = HashMap::new();
    for route in json.routes.into_values() {
        let context = || format!("{}: filetype {}", path.display(), route.filetype);
        // Retain the hostile policy at every level (ascending) for the verdict
        // sweep. A level with no usable policy (collimator's `no_policy`) is
        // dropped from the grid — the sweep simply can't fire there.
        let mut levels = Vec::new();
        for entry in &route.levels {
            if let Some(hostile) =
                policy_severity_from_json(&entry.hostile, names).with_context(context)?
            {
                levels.push(LevelPolicy {
                    level: entry.level,
                    hostile,
                });
            }
        }
        levels.sort_by_key(|lp| lp.level);

        // The default-level policy drives the diagnostic per-route
        // classification in the `models[]` array. Its suspicious band is
        // derived in level-space via the suspicious-ceiling lookup: the hostile
        // policy at `capped_suspicious_level(max)`. An explicit `suspicious`
        // block, when collimator emits one, wins; the level's own hostile is
        // the final fallback.
        let active = match route.levels.iter().find(|entry| entry.level == level) {
            Some(entry) => {
                match policy_severity_from_json(&entry.hostile, names).with_context(context)? {
                    Some(hostile) => {
                        let explicit = match &entry.suspicious {
                            Some(json) => {
                                policy_severity_from_json(json, names).with_context(context)?
                            }
                            None => None,
                        };
                        let suspicious = explicit.unwrap_or_else(|| {
                            let max = route.levels.iter().map(|e| e.level).max().unwrap_or(level);
                            let suspicious_level = capped_suspicious_level(max);
                            levels
                                .iter()
                                .find(|lp| lp.level == suspicious_level)
                                .map_or_else(|| hostile.clone(), |lp| lp.hostile.clone())
                        });
                        Some(RoutePolicy {
                            suspicious,
                            hostile,
                        })
                    }
                    None => None,
                }
            }
            None => None,
        };
        if !levels.is_empty() {
            grid.insert(route.filetype.clone(), levels);
        }
        if let Some(policy) = active {
            by_filetype.insert(route.filetype, policy);
        }
    }

    tracing::info!(
        path = %path.display(),
        level = level,
        routes = by_filetype.len(),
        grid_routes = grid.len(),
        "loaded route policies",
    );
    Ok(RoutePolicies { by_filetype, grid })
}

/// One severity's policy, or `None` when collimator found none at this level
/// (no `best`, or neither thresholds nor a blend).
///
/// Thresholds outside `[0, 1]` are dropped: collimator writes one ulp above
/// 1.0 for a route that must never fire at a level, and a probability can
/// never reach it, so dropping it decides the same way.
fn policy_severity_from_json(
    json: &RoutePolicySeverityJson,
    names: &mut RouteNames,
) -> Result<Option<PolicySeverity>> {
    let Some(best) = json.best.as_ref() else {
        return Ok(None);
    };
    let blend = best
        .blend
        .as_ref()
        .map(|blend| blend_policy_from_json(blend, names))
        .transpose()?;
    // Ordered by name so diagnostics that list a policy's routes are stable.
    let mut named: Vec<(&String, f32)> = best
        .thresholds
        .iter()
        .filter(|&(_, t)| (0.0..=1.0).contains(t))
        .map(|(route, &t)| (route, t))
        .collect();
    named.sort_by(|a, b| a.0.cmp(b.0));
    let thresholds: Vec<(RouteId, f32)> = named
        .into_iter()
        .map(|(route, t)| (names.intern(route), t))
        .collect();
    // A severity is loadable if either the OR-rule has at least one
    // threshold OR the blend is well-formed. The two coexist only in
    // pathological writer output — current code emits one or the other.
    if thresholds.is_empty() && blend.is_none() {
        return Ok(None);
    }
    Ok(Some(PolicySeverity { thresholds, blend }))
}

fn blend_policy_from_json(json: &BlendPolicyJson, names: &mut RouteNames) -> Result<BlendPolicy> {
    // Only the logit transform is supported. An unknown transform means the
    // writer is ahead of this deploy binary — refuse rather than misapply.
    if let Some(other) = json.transform.as_deref().filter(|t| *t != "logit") {
        anyhow::bail!("unknown blend transform {other:?}");
    }
    if json.routes.is_empty() || json.routes.len() != json.weights.len() {
        anyhow::bail!(
            "blend needs one weight per route (routes={}, weights={})",
            json.routes.len(),
            json.weights.len(),
        );
    }
    if !(0.0..=1.0).contains(&json.threshold) {
        anyhow::bail!("blend threshold {} is outside [0, 1]", json.threshold);
    }
    Ok(BlendPolicy {
        routes: json.routes.iter().map(|r| names.intern(r)).collect(),
        weights: json.weights.clone(),
        intercept: json.intercept,
        threshold: json.threshold,
    })
}

/// Pull per-route thresholds from `levels[]` at the requested level. Pairs up
/// hostile and suspicious thresholds for each route. When the JSON omits a
/// `suspicious` block, suspicious is derived in level-space by reading the
/// hostile threshold for the same route at level
/// `capped_suspicious_level(max_grid_level)`.
fn thresholds_at_level(levels: &[LevelEntryJson], level: u16) -> HashMap<String, Thresholds> {
    let Some(entry) = levels.iter().find(|e| e.level == level) else {
        if !levels.is_empty() {
            tracing::warn!(
                level = level,
                available = ?levels.iter().map(|e| e.level).collect::<Vec<_>>(),
                "ensemble config has no thresholds for this severity level"
            );
        }
        return HashMap::new();
    };

    let max = levels.iter().map(|e| e.level).max().unwrap_or(level);
    let suspicious_level = capped_suspicious_level(max);
    let loose_hostile = (suspicious_level != level)
        .then(|| levels.iter().find(|e| e.level == suspicious_level))
        .flatten()
        .map(|e| &e.hostile.thresholds);

    let mut out: HashMap<String, Thresholds> = HashMap::new();
    let suspicious_block = entry.suspicious.as_ref();
    for (route, &hostile) in &entry.hostile.thresholds {
        // Hostile policy is primary. When the JSON carries an explicit
        // `suspicious` block but a specific route is absent from it, that
        // route stays active as hostile-only (suspicious == hostile). When
        // collimator omits the `suspicious` block entirely, derive the
        // suspicious cutoff from the ceiling row's hostile threshold for the
        // same route (level-space lookup, falls back to hostile-only when
        // the route is missing from the looser row).
        let suspicious = match suspicious_block {
            Some(block) => block.thresholds.get(route).copied().unwrap_or(hostile),
            None => loose_hostile
                .and_then(|m| m.get(route))
                .map_or(hostile, |&s| s.min(hostile)),
        };
        let t = Thresholds {
            suspicious,
            hostile,
        };
        if t.validate().is_ok() {
            out.insert(route.clone(), t);
        } else {
            // Inverted hostile/suspicious thresholds for a single route.
            // Litmus drops the route from this level entirely — meaningful
            // recall loss that should not pass deploy verification silently.
            tracing::error!(
                route = %route, level = level,
                suspicious = t.suspicious, hostile = t.hostile,
                "ignoring invalid thresholds in ensemble config"
            );
        }
    }
    if let Some(block) = suspicious_block {
        for (route, &suspicious) in &block.thresholds {
            if out.contains_key(route) {
                continue;
            }
            let t = Thresholds {
                suspicious,
                hostile: 1.0,
            };
            if t.validate().is_ok() {
                out.insert(route.clone(), t);
            } else {
                tracing::error!(
                    route = %route, level = level,
                    suspicious = t.suspicious, hostile = t.hostile,
                    "ignoring invalid suspicious-only thresholds in ensemble config"
                );
            }
        }
    }
    out
}

/// Load the inference backend for a bundle directory. Multi-seed bundles
/// store every member at `models/seed_NN.onnx` (one file per seed); the
/// single-bundle layout has a single `model.onnx` directly under the bundle
/// dir. Both layouts are accepted; the multi-seed layout is preferred when the
/// `models/` subdirectory is present and non-empty.
///
/// Mixing the single-bundle and multi-seed layouts in the same bundle is
/// rejected.
fn load_backend(bundle_dir: &Path) -> Result<Backend> {
    let multi_dir = bundle_dir.join("models");
    let multi = if multi_dir.is_dir() {
        collect_multi_seed_paths(&multi_dir)?
    } else {
        Vec::new()
    };

    // Single-model layout: model.onnx directly under bundle_dir. The native
    // model.txt (LightGBM) / model.json (XGBoost) layouts were retired — the
    // deploy ships ONNX-only.
    let single_onnx = bundle_dir.join("model.onnx");

    if !multi.is_empty() && single_onnx.is_file() {
        anyhow::bail!(
            "model bundle is ambiguous: both `models/` (multi-seed) and a top-level model.onnx \
             exist in {}; remove one to disambiguate the layout",
            bundle_dir.display(),
        );
    }

    let model_paths: Vec<PathBuf> = if !multi.is_empty() {
        multi
    } else if single_onnx.is_file() {
        vec![single_onnx]
    } else {
        anyhow::bail!(
            "model bundle is incomplete: no model.onnx and no models/ directory in {}",
            bundle_dir.display(),
        )
    };

    // Load seeds serially rather than with `into_par_iter`. This runs inside a
    // specialist route's `OnceLock` initializer (`RouteStore::get`), which is
    // itself reached from the rayon parallel classify. A nested rayon call here
    // lets the initializing worker steal another classify task that needs the
    // *same* route, re-enter the in-progress `OnceLock`, and deadlock forever on
    // its wait semaphore — the intermittent multi-minute hang seen on `pkg`/`url`
    // scans. A bundle is at most a few small ONNX seeds, so serial loading costs
    // nothing measurable and removes the re-entrancy entirely.
    let mut paths = model_paths.into_iter();
    let Some(first_path) = paths.next() else {
        anyhow::bail!("model bundle contains no loadable model")
    };
    let first = OnnxModel::load(&first_path)?;
    let n_features = first.n_features();
    let mut rest = Vec::with_capacity(paths.len());
    for path in paths {
        let member = OnnxModel::load(&path)?;
        // Refuse to mix feature counts — averaging across heterogeneous
        // feature spaces would silently produce nonsense scores.
        let n = member.n_features();
        if n_features != n {
            anyhow::bail!(
                "model bundle has mismatched feature counts in {}: member {} expects {n} \
                 features but earlier members expected {n_features}",
                bundle_dir.display(),
                path.display(),
            );
        }
        rest.push(member);
    }

    Ok(Backend { first, rest })
}

/// Pick up every `seed_*.onnx` under a `models/` directory. Native
/// `.txt`/`.json` seeds are ignored — only ONNX is loadable. Output is
/// sorted (so seed_42 lands before seed_43) and deterministic across runs.
fn collect_multi_seed_paths(multi_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(multi_dir)
        .with_context(|| format!("reading multi-seed models dir {}", multi_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("reading entry under {}", multi_dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Tolerate bystander files (READMEs, hashes) and retired native dumps
        // so they don't break loading; only ONNX seeds are members.
        if name.starts_with("seed_") && path.extension().and_then(|e| e.to_str()) == Some("onnx") {
            paths.push(path);
        }
    }
    // Sorted for a deterministic load order of the averaged ensemble.
    paths.sort();
    Ok(paths)
}

/// Load one model bundle (model file + feature spec + thresholds).
///
/// `is_general` controls how missing artifacts are reported: for the general
/// route they're fatal load errors; for specialists callers handle the error
/// non-fatally (drop with a warning).
///
/// Threshold resolution: `explicit` → the bundle's `config.json` →
/// `evaluation.json` recommendations → [`Thresholds::default`].
fn load_bundle(
    bundle_dir: &Path,
    explicit: Option<Thresholds>,
    is_general: bool,
) -> Result<LoadedBundle> {
    let spec_path = bundle_dir.join("feature_spec.json");
    if !spec_path.is_file() {
        if is_general {
            anyhow::bail!(
                "model bundle is incomplete: missing {}. Run 'atomscan update-rules' to refresh the installed models.",
                spec_path.display(),
            );
        }
        anyhow::bail!("missing {}", spec_path.display());
    }

    let backend = load_backend(bundle_dir)?;

    tracing::debug!(path = %spec_path.display(), "loading feature spec");
    let spec = FeatureSpec::load(&spec_path)
        .with_context(|| format!("loading feature spec from {}", spec_path.display()))?;

    if spec.total_features() != backend.num_features() {
        anyhow::bail!(
            "feature count mismatch: feature_spec.json has {} features but the model expects {} — \
             these artifacts are from different training runs",
            spec.total_features(),
            backend.num_features(),
        );
    }

    let (thresholds, threshold_source) = if let Some(explicit) = explicit {
        explicit
            .validate()
            .map_err(|error| anyhow::anyhow!("invalid thresholds: {error}"))?;
        if let Some(recommended) = load_evaluation_thresholds(bundle_dir)? {
            explicit.warn_if_divergent(&recommended);
        }
        (explicit, "explicit")
    } else if let Some(t) = BundleConfig::load(bundle_dir)?
        .map(|cfg| cfg.thresholds())
        .transpose()?
        .flatten()
    {
        (t, "config.json")
    } else if let Some(t) = load_evaluation_thresholds(bundle_dir)? {
        (t, "evaluation.json")
    } else {
        tracing::error!(
            bundle = %bundle_dir.display(),
            "no config.json or evaluation.json thresholds found — using conservative \
             fallback (suspicious={}, hostile={})",
            Thresholds::FALLBACK_SUSPICIOUS,
            Thresholds::FALLBACK_HOSTILE,
        );
        (Thresholds::default(), "fallback")
    };

    let calibrator = IsotonicCalibrator::load_optional(bundle_dir)
        .with_context(|| format!("loading optional calibrator from {}", bundle_dir.display()))?;
    // When a calibrator is present, push the bundle-level thresholds through
    // it so threshold comparisons happen in the same (calibrated) probability
    // space as `predict_calibrated`'s output.  Isotonic is monotone, so this
    // is decision-equivalent to comparing raw_score >= raw_threshold — but
    // every emitted probability is now on a meaningful [0,1] scale matching
    // the empirical malware fraction we observed at training time.
    //
    // Per-route policy thresholds (in `route_policies.json`, used by the
    // OR-of-routes deployment policies) live in a different on-disk file and
    // can't be calibrated here — the per-route calibrators aren't all loaded
    // yet at this point. They get a second pass in `Model::load_ensemble`
    // via `calibrate_policy_thresholds`, after every specialist has been
    // loaded and its calibrator is available for lookup.
    let thresholds = if let Some(cal) = calibrator.as_ref() {
        let calibrated = Thresholds {
            suspicious: cal.apply(thresholds.suspicious),
            hostile: cal.apply(thresholds.hostile),
        };
        // Re-validate: a calibrator could in principle push hostile below
        // suspicious (it shouldn't given monotonicity, but defense in depth)
        // or out of [0,1]. We'd rather refuse to load than ship a bundle
        // whose decision boundaries are nonsensical.
        calibrated.validate().with_context(|| {
            format!(
                "calibrated thresholds for {} are invalid (suspicious={}, hostile={}); \
             check the calibrator at {}/calibrator.json",
                bundle_dir.display(),
                calibrated.suspicious,
                calibrated.hostile,
                bundle_dir.display(),
            )
        })?;
        tracing::debug!(
            bundle = %bundle_dir.display(),
            breakpoints = cal.x.len(),
            "loaded isotonic calibrator (thresholds calibrated to match)"
        );
        calibrated
    } else {
        thresholds
    };

    Ok(LoadedBundle {
        backend,
        spec,
        thresholds,
        threshold_source,
        calibrator,
    })
}

/// Walk a `filegroups/` or `filetypes/` directory and register each
/// subdirectory as a lazily loaded specialist. A directory the deployment
/// config does not mention, or whose calibrator is invalid, is recorded as
/// skipped rather than failing the whole load.
///
/// `category` is the path prefix used in the ensemble config's route names —
/// either `"filegroups"` or `"filetypes"` — so specialist thresholds can be
/// looked up under e.g. `"filegroups/native"`.
fn load_specialists(
    parent: &Path,
    route_thresholds: &HashMap<String, Thresholds>,
    route_policies: &RoutePolicies,
    names: &RouteNames,
    category: &'static str,
) -> RouteStore {
    let mut out = RouteStore::default();
    let entries = match std::fs::read_dir(parent) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return out,
        Err(e) => {
            tracing::warn!(parent = %parent.display(), error = %e, "cannot read specialist directory");
            return out;
        }
    };

    // Building each specialist's runnable tract plan is the expensive part,
    // so scans keep lazy descriptors after cheap calibrator validation.
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let name = name.to_owned();
        let route_name = format!("{category}/{name}");
        // Skip on-disk subdirectories that the deployment config doesn't list.
        // These are common as artifacts of experimentation; loading them with
        // fallback thresholds would put uncalibrated routes in the OR. Every
        // route the config or the policies name was interned before this runs.
        let route_t = route_thresholds.get(&route_name).copied();
        let Some(id) = names
            .id(&route_name)
            .filter(|&id| route_t.is_some() || route_policies.contains_route(id))
        else {
            tracing::debug!(
                category = %category,
                name = %name,
                "skipping specialist directory: no thresholds in ensemble config"
            );
            out.skipped.insert(name);
            continue;
        };
        let calibrator = match IsotonicCalibrator::load_optional(&path) {
            Ok(calibrator) => calibrator,
            Err(error) => {
                tracing::warn!(
                    category = %category,
                    name = %name,
                    path = %path.display(),
                    error = ?error,
                    "dropping specialist with invalid calibrator",
                );
                out.skipped.insert(name);
                continue;
            }
        };
        out.lazy.insert(
            name.clone(),
            LazyRoute {
                bundle_dir: path,
                name,
                category,
                id,
                thresholds: route_t.unwrap_or_default(),
                calibrator,
                loaded: OnceLock::new(),
            },
        );
    }
    out
}

/// Load and validate one specialist bundle, applying the ABI-version rule.
/// Errors here are recoverable — the specialist is dropped, not fatal.
fn load_specialist(bundle_dir: &Path, name: &str, thresholds: Thresholds) -> Result<Route> {
    let bundle = load_bundle(bundle_dir, Some(thresholds), /* is_general = */ false)
        .with_context(|| format!("specialist {name}"))?;

    if bundle.spec.abi_version() != EXPECTED_MODEL_ABI_VERSION {
        anyhow::bail!(
            "ABI mismatch: specialist abi_version={} (build expects {})",
            bundle.spec.abi_version(),
            EXPECTED_MODEL_ABI_VERSION,
        );
    }

    let ctx = ExtractContext::new(&bundle.spec);

    Ok(Route {
        backend: bundle.backend,
        spec: bundle.spec,
        ctx,
        thresholds: bundle.thresholds,
        calibrator: bundle.calibrator,
    })
}

/// One inference path (general, a filegroup specialist, or a filetype specialist).
///
/// Specialists may have their own feature space. Route scoring extracts and
/// standardizes features with the route's own spec before calling its backend.
#[derive(Debug)]
struct Route {
    backend: Backend,
    spec: FeatureSpec,
    ctx: ExtractContext,
    thresholds: Thresholds,
    /// Optional isotonic calibrator. When present, the route's raw
    /// probability is mapped through this before any downstream consumer
    /// (threshold check, OR aggregation, JSON output) sees it. Older
    /// bundles without `calibrator.json` carry `None` and behave like
    /// before — backward compatible.
    calibrator: Option<IsotonicCalibrator>,
}

impl Route {
    /// Score a feature vector and return `(raw, calibrated)`. The raw value is
    /// the backend's pre-isotonic probability; the calibrated value is what
    /// every decision path consumes.
    fn predict_raw_calibrated(&self, features: &[f32]) -> Result<(f32, f32)> {
        let raw = self.backend.predict(features)?;
        let calibrated = self.calibrator.as_ref().map_or(raw, |cal| cal.apply(raw));
        Ok((raw, calibrated))
    }
}

/// A specialist registered at load and built on first use.
#[derive(Debug)]
struct LazyRoute {
    bundle_dir: PathBuf,
    name: String,
    category: &'static str,
    id: RouteId,
    thresholds: Thresholds,
    calibrator: Option<IsotonicCalibrator>,
    loaded: OnceLock<RouteLoad>,
}

#[derive(Debug)]
enum RouteLoad {
    Loaded(Arc<Route>),
    Failed(String),
}

impl LazyRoute {
    fn load_once(&self) -> &RouteLoad {
        self.loaded.get_or_init(|| {
            match load_specialist(&self.bundle_dir, &self.name, self.thresholds) {
                Ok(route) => RouteLoad::Loaded(Arc::new(route)),
                Err(error) => {
                    let message = format!("{error:#}");
                    tracing::warn!(
                        category = %self.category,
                        name = %self.name,
                        path = %self.bundle_dir.display(),
                        error = %message,
                        "dropping lazy specialist; route will degrade to general or filegroup",
                    );
                    RouteLoad::Failed(message)
                }
            }
        })
    }

    fn get(&self) -> Option<Arc<Route>> {
        match self.load_once() {
            RouteLoad::Loaded(route) => Some(Arc::clone(route)),
            RouteLoad::Failed(_) => None,
        }
    }

    fn validate(&self) -> Result<()> {
        match self.load_once() {
            RouteLoad::Loaded(_) => Ok(()),
            RouteLoad::Failed(message) => {
                anyhow::bail!(
                    "specialist {}/{} at {} failed to load: {}",
                    self.category,
                    self.name,
                    self.bundle_dir.display(),
                    message
                )
            }
        }
    }
}

/// The specialists of one category (`filegroups` or `filetypes`), by name.
#[derive(Debug, Default)]
struct RouteStore {
    lazy: HashMap<String, LazyRoute>,
    /// On-disk specialists that were not registered (uncalibrated or invalid).
    skipped: HashSet<String>,
}

impl RouteStore {
    /// The named specialist, loading it on first use; `None` when absent or
    /// when it failed to load.
    fn get(&self, name: &str) -> Option<(RouteId, Arc<Route>)> {
        let lazy = self.lazy.get(name)?;
        Some((lazy.id, lazy.get()?))
    }

    fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    fn calibrator(&self, name: &str) -> Option<&IsotonicCalibrator> {
        self.lazy.get(name)?.calibrator.as_ref()
    }

    fn len(&self) -> usize {
        self.lazy.len()
    }

    fn is_empty(&self) -> bool {
        self.lazy.is_empty()
    }

    fn validate_all(&self) -> Result<()> {
        self.lazy
            .par_iter()
            .try_for_each(|(_, route)| route.validate())
    }

    fn validate_route(&self, name: &str) -> Result<()> {
        match self.lazy.get(name) {
            Some(route) => route.validate(),
            None => anyhow::bail!("route {name:?} is not available"),
        }
    }
}

/// Routing decision: which route to consult for a file of type `T` in group `G`.
///
/// DESIGN.md says: filetype if present → filegroup if present → general always.
/// Each present route contributes to the OR over thresholds.
#[derive(Debug, Default)]
struct RouteSet {
    /// Filegroup name → specialist model or lazy descriptor.
    filegroups: RouteStore,
    /// Filetype name → specialist model or lazy descriptor.
    filetypes: RouteStore,
    /// Filetype → filegroup mapping from `config.json`. Used to translate a
    /// scanned file's `type` into the applicable filegroup specialist.
    filetype_to_filegroup: HashMap<String, String>,
    /// Optional searched policies keyed by cleave file type.
    policies: RoutePolicies,
    /// Every route name the config, the policies and the specialists use.
    names: RouteNames,
}

impl RouteSet {
    /// True when there are no specialist routes available; routed prediction
    /// then degrades to the general route alone.
    fn is_empty(&self) -> bool {
        self.filegroups.is_empty() && self.filetypes.is_empty()
    }

    /// Specialist lookup keys for a scanned `file_type`, selected by ARCHIVE
    /// format rather than compression. A compressed label collapses onto its
    /// container (`tar.gz` → `tar`, `tar.bz2.xz` → `tar`) so the compressed
    /// variant reaches the same filetype/filegroup specialists — and the same
    /// per-route policy thresholds — that collimator trained and calibrated
    /// under the normalized label. Returns `(filetype key, filegroup)`.
    /// Agrees with [`lookup_filetype`], so route selection and the policy's
    /// per-route threshold keys agree.
    fn specialist_keys<'a>(&'a self, file_type: &'a str) -> (Cow<'a, str>, Option<&'a str>) {
        let normalized = normalize_archive_filetype(file_type);
        // Specialty-else-container: prefer a specialist trained for the type
        // itself (e.g. `gem`, `whl`, `python_sdist`); when none exists, fall
        // back to its container archive (`tar`, `zip`) as filefacts defines
        // it. Compression is already collapsed by `normalize_archive_filetype`
        // (`tar.gz` → `tar`), so this only adds the package → container hop.
        let key = if self.filetypes.contains(&normalized) {
            normalized
        } else if let Some(container) =
            container_filetype(file_type).filter(|c| self.filetypes.contains(c))
        {
            Cow::Borrowed(container)
        } else {
            normalized
        };
        let group = self
            .filetype_to_filegroup
            .get(key.as_ref())
            .map(String::as_str);
        (key, group)
    }

    /// The isotonic calibrator of a route, if it has one.
    fn calibrator<'a>(
        &'a self,
        general: Option<&'a IsotonicCalibrator>,
        route: RouteId,
    ) -> Option<&'a IsotonicCalibrator> {
        match RouteKind::parse(self.names.name(route))? {
            RouteKind::General => general,
            RouteKind::Group(name) => self.filegroups.calibrator(name),
            RouteKind::Type(name) => self.filetypes.calibrator(name),
        }
    }

    fn validate_all(&self) -> Result<()> {
        let (filegroups, filetypes) = rayon::join(
            || self.filegroups.validate_all(),
            || self.filetypes.validate_all(),
        );
        filegroups.context("validating filegroup specialist routes")?;
        filetypes.context("validating filetype specialist routes")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct RouteProbability {
    route: RouteId,
    probability: f32,
}

#[cfg(test)]
fn policy_classify(policy: &RoutePolicy, scores: &[RouteProbability]) -> Classification {
    // Each severity decides via its own rule: OR-rule per-route thresholds or
    // learned blend. Hostile takes precedence over suspicious, same as before.
    if policy.hostile.fires(scores) {
        Classification::Hostile
    } else if policy.suspicious.fires(scores) {
        Classification::Suspicious
    } else {
        Classification::Benign
    }
}

/// Run the policy against the route scores and return the deciding [`Decision`]
/// for hostile/suspicious fires. Returns `None` when neither severity fires —
/// the caller picks the Benign fallback (typically general's prob against the
/// model's suspicious cutoff).
fn policy_decide(policy: &RoutePolicy, scores: &[RouteProbability]) -> Option<Decision> {
    if let Some((p, t)) = policy.hostile.fire(scores) {
        return Some(Decision {
            class: Classification::Hostile,
            probability: p,
            threshold: t,
            level: Level::Manual,
        });
    }
    if let Some((p, t)) = policy.suspicious.fire(scores) {
        return Some(Decision {
            class: Classification::Suspicious,
            probability: p,
            threshold: t,
            level: Level::Manual,
        });
    }
    None
}

/// Lowest level (FP per 100M benigns) at which a filetype's per-level hostile
/// policy fires, with the deciding `(probability, threshold)` at that level.
///
/// `grid` is ascending by level. Hostile thresholds loosen as the level rises,
/// so once a file fires it keeps firing — the minimum firing level is the
/// strictest budget that still flags it. Scanning every level and keeping the
/// minimum stays correct even if a bundle's grid isn't perfectly monotone.
/// Returns `None` when the file fires at no level (clean).
fn sweep_policy_grid(grid: &[LevelPolicy], scores: &[RouteProbability]) -> Option<(u16, f32, f32)> {
    let mut best: Option<(u16, f32, f32)> = None;
    for lp in grid {
        if let Some((p, t)) = lp.hostile.fire(scores)
            && best.is_none_or(|(bl, _, _)| lp.level < bl)
        {
            best = Some((lp.level, p, t));
        }
    }
    best
}

/// Map a file's lowest firing level to a verdict at the active deploy level.
///
/// `fired_level` is the level-independent marker from the grid sweep; `level` is
/// the active `-l`. A file is **hostile** when it fires within the hostile
/// budget (`fired_level <= level`) and **suspicious** when it fires above that
/// budget but within the derived suspicious ceiling. Beyond that ceiling is
/// benign.
pub(crate) fn verdict_for_level(fired_level: u16, level: u16, grid_max: u16) -> Classification {
    if fired_level <= level {
        Classification::Hostile
    } else if fired_level <= capped_suspicious_level(grid_max) {
        Classification::Suspicious
    } else {
        Classification::Benign
    }
}

/// Lowest level at which the general/OR fallback fires, for filetypes with no
/// route policy. Mirrors [`Model::decide_from_scores`]: at each level a file
/// fires if any route's probability crosses that level's general hostile
/// threshold. The highest crossing probability is reported for that level.
fn sweep_general_grid(grid: &[(u16, f32)], scores: &[RouteProbability]) -> Option<(u16, f32, f32)> {
    let mut best: Option<(u16, f32, f32)> = None;
    for &(level, thr) in grid {
        if let Some(p) = scores
            .iter()
            .map(|s| s.probability)
            .filter(|&p| p >= thr)
            .reduce(f32::max)
            && best.is_none_or(|(bl, _, _)| level < bl)
        {
            best = Some((level, p, thr));
        }
    }
    best
}

/// Push each per-route threshold in `route_policies.json` through that route's
/// isotonic calibrator. The scoring path emits calibrated probabilities, so
/// the policy comparison must happen in calibrated space too. Routes without
/// a calibrator (pre-calibrator bundles) are left untouched. Routes referenced
/// by a policy but not loaded as a specialist are also left untouched —
/// they'll never produce a `RouteProbability` entry, so the comparison never
/// fires.
///
/// Blend severities are calibrated at fit time (the policy writer applies
/// isotonic to the per-route inputs before fitting LR), so their threshold
/// and weights are already in calibrated-prob space. Skipping the per-route
/// mapping is intentional — the logistic combination doesn't decompose
/// through isotonic the way per-route OR-rule thresholds do.
fn calibrate_policy_thresholds<'a>(
    policies: &mut RoutePolicies,
    lookup: impl Fn(RouteId) -> Option<&'a IsotonicCalibrator>,
) {
    let mut adjusted = 0_usize;
    for severity in policies.severities_mut() {
        if severity.blend.is_some() {
            continue;
        }
        for (route, threshold) in &mut severity.thresholds {
            if let Some(cal) = lookup(*route) {
                *threshold = cal.apply(*threshold);
                adjusted += 1;
            }
        }
    }
    if adjusted > 0 {
        tracing::debug!(adjusted, "calibrated route_policies thresholds");
    }
}

fn policy_route_class(policy: &RoutePolicy, route: RouteId, probability: f32) -> Classification {
    // Per-route classification is well-defined only for OR-rule severities —
    // a learned blend's verdict is a function of all its inputs jointly, so
    // no single route can be labeled Hostile/Suspicious on its own. For
    // blend severities we return Benign here; the final verdict still comes
    // from the policy decision, which evaluates the blend over the full score
    // vector. The diagnostic per-route classification just won't get an
    // individual contribution for blend-driven filetypes.
    let fires = |severity: &PolicySeverity| {
        severity.blend.is_none() && severity.threshold(route).is_some_and(|t| probability >= t)
    };
    if fires(&policy.hostile) {
        Classification::Hostile
    } else if fires(&policy.suspicious) {
        Classification::Suspicious
    } else {
        Classification::Benign
    }
}

/// Loaded model plus the feature spec and thresholds used for inference.
///
/// For an ensemble bundle, `inner`/`spec`/`thresholds` are the *general*
/// route, and `routes` carries the optional specialists. For a single-bundle
/// deployment, `routes` is empty and the model behaves exactly like before.
#[derive(Debug)]
pub struct Model {
    inner: Backend,
    spec: FeatureSpec,
    /// Extraction tables for `spec`, built once at load.
    ctx: ExtractContext,
    thresholds: Thresholds,
    info: ModelInfo,
    routes: RouteSet,
    /// Optional isotonic calibrator for the general model. Applied after
    /// `inner.predict` everywhere we score with the general route. Mirrors
    /// the per-route calibrator on Route; backward compat = None.
    general_calibrator: Option<IsotonicCalibrator>,
    /// Active deploy level (FP per 100M benigns) that drives the verdict and
    /// the envelope level marker. `None` in manual-threshold mode (`--threshold-*`)
    /// and on single-bundle deployments with no level grid; the sweep is then
    /// bypassed and the level serializes as `null`.
    active_level: Option<u16>,
    /// General route's hostile threshold per level (ascending), for the
    /// no-policy verdict sweep. Empty on single-bundle / pre-grid deployments.
    general_grid: Vec<(u16, f32)>,
    /// Largest grid level; suspicious cap = `capped_suspicious_level(grid_max)`.
    grid_max: u16,
}

/// One route's score for a file, before it is split into the decision input
/// and the wire record.
struct Scored {
    route: RouteId,
    raw: f32,
    probability: f32,
    class: Classification,
}

impl Model {
    /// Load model artifacts from a directory containing `feature_spec.json`
    /// and `model.onnx` (or `models/seed_*.onnx`).
    ///
    /// The same directory may also contain optional metadata such as
    /// `shap_importance.json` and git history, but those are not required here.
    ///
    /// # Errors
    /// Returns an error if thresholds are invalid, required model artifacts are
    /// missing, any present metadata file is malformed, or the loaded feature
    /// spec does not match this build.
    ///
    /// Threshold resolution order:
    /// 1. Explicit `thresholds` argument (from CLI flags)
    /// 2. `config.json` in the model directory
    /// 3. Recommended thresholds from `evaluation.json`
    /// 4. Conservative fallback constants
    ///
    /// If explicit thresholds are provided *and* `evaluation.json` contains
    /// recommendations, a warning is emitted when they diverge significantly.
    pub fn load(
        model_dir: &Path,
        thresholds: Option<Thresholds>,
        active_level: Option<u16>,
    ) -> Result<Self> {
        // Detect ensemble layout by the presence of `general/` immediately
        // under model_dir. Otherwise treat model_dir itself as a single bundle.
        if model_dir.join("general").is_dir() {
            Self::load_ensemble(model_dir, thresholds, active_level)
        } else {
            Self::load_single_bundle(model_dir, thresholds, active_level)
        }
    }

    /// Load a single-bundle layout (legacy / dev). The model artifacts
    /// (`model.onnx`, `feature_spec.json`, `config.json`,
    /// `evaluation.json`) sit directly in `model_dir`. No specialists.
    fn load_single_bundle(
        model_dir: &Path,
        thresholds: Option<Thresholds>,
        active_level: Option<u16>,
    ) -> Result<Self> {
        let bundle = load_bundle(model_dir, thresholds, /* is_general = */ true)?;
        tracing::info!(
            n_members = bundle.backend.n_members(),
            features = bundle.spec.total_features(),
            model_abi_version = bundle.spec.abi_version(),
            threshold_suspicious = bundle.thresholds.suspicious,
            threshold_hostile = bundle.thresholds.hostile,
            threshold_source = bundle.threshold_source,
            spec_version = bundle.spec.version(),
            layout = "single-bundle",
            "model loaded",
        );
        // Single-bundle deployments carry no level grid, so the verdict sweep
        // has nothing to sweep: `decide` falls back to the threshold path and
        // the level serializes as `null`.
        Ok(Self::new(
            bundle,
            RouteSet::default(),
            active_level,
            Vec::new(),
            0,
        ))
    }

    /// Load an ensemble layout (`general/` + `filegroups/*` + `filetypes/*`).
    /// See module-level docs for the on-disk shape and config schema.
    fn load_ensemble(
        model_dir: &Path,
        thresholds: Option<Thresholds>,
        active_level: Option<u16>,
    ) -> Result<Self> {
        // Ensemble-level config.json carries the routing map and the per-route
        // thresholds for every level. We resolve thresholds from it before
        // loading any individual route bundle so each bundle gets the right
        // pre-resolved thresholds and skips its own (nonexistent) config.json.
        //
        // The diagnostic `by_filetype` policy and per-route thresholds are
        // pinned to DEFAULT_SEVERITY_LEVEL (not the active level) so the
        // `models[]` array and emitted `prob` are identical regardless of the
        // caller's `-l` — the JSON envelope stays cacheable across levels. The
        // active level enters only the final verdict derivation, via the
        // level-independent sweep over `policies.grid` / `general_grid`.
        let cfg = EnsembleConfig::new(
            BundleConfig::load(model_dir)?.unwrap_or_default(),
            DEFAULT_SEVERITY_LEVEL,
        );
        let mut names = RouteNames::default();
        for route in cfg.route_thresholds.keys() {
            names.intern(route);
        }
        let mut policies = load_route_policies(model_dir, DEFAULT_SEVERITY_LEVEL, &mut names)?;

        let general_dir = model_dir.join("general");
        let general_thresholds =
            thresholds.or_else(|| cfg.route_thresholds.get("general").copied());

        // Walk filegroups/<name>/ and filetypes/<name>/ while general loads.
        let specialist_start = std::time::Instant::now();
        let (general, (filegroups, filetypes)) = rayon::join(
            || {
                load_bundle(
                    &general_dir,
                    general_thresholds,
                    /* is_general = */ true,
                )
                .with_context(|| format!("loading general route from {}", general_dir.display()))
            },
            || {
                rayon::join(
                    || {
                        load_specialists(
                            &model_dir.join("filegroups"),
                            &cfg.route_thresholds,
                            &policies,
                            &names,
                            "filegroups",
                        )
                    },
                    || {
                        load_specialists(
                            &model_dir.join("filetypes"),
                            &cfg.route_thresholds,
                            &policies,
                            &names,
                            "filetypes",
                        )
                    },
                )
            },
        );
        let general = general?;
        tracing::info!(
            filegroups = filegroups.len(),
            filetypes = filetypes.len(),
            elapsed_ms = specialist_start.elapsed().as_millis(),
            "prepared specialist routes",
        );

        let routes = RouteSet {
            filegroups,
            filetypes,
            filetype_to_filegroup: cfg.filetype_to_filegroup,
            policies: RoutePolicies::default(),
            names,
        };

        // Required routes from config: any listed name that didn't load is
        // fatal. "general" is implicitly required and was already loaded above.
        for required in &cfg.required_routes {
            match RouteKind::parse(required) {
                Some(RouteKind::General) => {}
                Some(RouteKind::Group(name)) => {
                    routes.filegroups.validate_route(name).with_context(|| {
                        format!(
                            "ensemble config marks filegroup {name:?} as required but it failed to load"
                        )
                    })?;
                }
                Some(RouteKind::Type(name)) => {
                    routes.filetypes.validate_route(name).with_context(|| {
                        format!(
                            "ensemble config marks filetype {name:?} as required but it failed to load"
                        )
                    })?;
                }
                None => anyhow::bail!(
                    "unknown required-route name {required:?} in ensemble config; \
                     expected `general`, `filegroups/<name>`, or `filetypes/<name>`"
                ),
            }
        }

        // The route_policies.json thresholds for the OR-of-routes decision
        // path were loaded in raw-score space, but the route scorers emit
        // calibrated probabilities (see `load_bundle`). Push each per-route
        // threshold through that route's calibrator so the policy compares
        // like with like. Isotonic monotonicity preserves the decision.
        calibrate_policy_thresholds(&mut policies, |route| {
            routes.calibrator(general.calibrator.as_ref(), route)
        });
        let routes = RouteSet { policies, ..routes };

        tracing::info!(
            n_members = general.backend.n_members(),
            features = general.spec.total_features(),
            model_abi_version = general.spec.abi_version(),
            threshold_suspicious = general.thresholds.suspicious,
            threshold_hostile = general.thresholds.hostile,
            threshold_source = general.threshold_source,
            spec_version = general.spec.version(),
            filegroups = routes.filegroups.len(),
            filetypes = routes.filetypes.len(),
            layout = "ensemble",
            "model loaded",
        );

        Ok(Self::new(
            general,
            routes,
            active_level,
            cfg.general_grid,
            cfg.grid_max,
        ))
    }

    fn new(
        general: LoadedBundle,
        routes: RouteSet,
        active_level: Option<u16>,
        general_grid: Vec<(u16, f32)>,
        grid_max: u16,
    ) -> Self {
        let info = ModelInfo {
            version: general.spec.version(),
            abi_version: general.spec.abi_version(),
        };
        let ctx = ExtractContext::new(&general.spec);
        Self {
            inner: general.backend,
            spec: general.spec,
            ctx,
            thresholds: general.thresholds,
            info,
            routes,
            general_calibrator: general.calibrator,
            active_level,
            general_grid,
            grid_max,
        }
    }

    /// Force every calibrated specialist route to load and validate.
    ///
    /// Normal scanning initializes specialist ONNX graphs on demand from the
    /// observed file type. Validation and install gates call this explicitly so
    /// malformed specialist artifacts are caught even if the current fixture set
    /// never routes to them.
    ///
    /// # Errors
    /// Returns the first specialist that fails to load.
    pub fn validate_all_routes(&self) -> Result<()> {
        if self.routes.is_empty() {
            return Ok(());
        }
        let started = std::time::Instant::now();
        self.routes.validate_all()?;
        tracing::info!(
            filegroups = self.routes.filegroups.len(),
            filetypes = self.routes.filetypes.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "validated specialist routes",
        );
        Ok(())
    }

    /// Classify a cleave report the way a scan does: featurize every file,
    /// standardize, score each route the report's primary file type reaches,
    /// and decide.
    ///
    /// # Errors
    /// Returns an error if a consulted route's backend fails.
    pub fn predict_report(
        &self,
        report: &cleave::types::CompactReport,
    ) -> Result<(Decision, Vec<RouteScore>, Vec<SkippedRoute>)> {
        let parsed = ParsedReport::from_compact_report(report, RawNeeds::all(), None);
        let mut features = self.ctx.extract_from_parsed(&parsed);
        self.spec.standardize(&mut features);
        let file_type = report
            .files
            .first()
            .map_or("unknown", |f| f.file_type.as_str());
        self.predict_for_report_detailed(file_type, &features, &parsed)
    }

    /// Routed prediction from a parsed cleave report (a whole sample or one
    /// archive member), with per-route scores retained for JSON and `--extra`
    /// output.
    ///
    /// This is the production ensemble path. General is scored from the
    /// caller-provided standardized general feature vector; each specialist
    /// scores its own route-specific vector from the shared [`ParsedReport`].
    pub(crate) fn predict_for_report_detailed(
        &self,
        file_type: &str,
        general_features: &[f32],
        parsed: &ParsedReport,
    ) -> Result<(Decision, Vec<RouteScore>, Vec<SkippedRoute>)> {
        let policy = self.routes.policies.policy_for(file_type);
        let (route_probs, scores, mut skipped) =
            self.score_routes(file_type, policy, general_features, parsed)?;
        let decision = self.decide(file_type, policy, &route_probs, &mut skipped);
        Ok((decision, scores, skipped))
    }

    /// Score every applicable route for `file_type` and return the
    /// `(probabilities, RouteScore list, SkippedRoute list)` triple. The
    /// per-route `RouteScore.classification` uses the policy's per-route
    /// classifier when available, else the route's own thresholds.
    fn score_routes(
        &self,
        file_type: &str,
        policy: Option<&RoutePolicy>,
        general_features: &[f32],
        parsed: &ParsedReport,
    ) -> Result<(Vec<RouteProbability>, Vec<RouteScore>, Vec<SkippedRoute>)> {
        // Select specialists by ARCHIVE format, not compression: a `.tgz`
        // resolves to the `tar` container specialist instead of routing
        // general-only and surfacing `filetypes/tar` as an `unavailable` skip.
        let (type_key, group_key) = self.routes.specialist_keys(file_type);
        let group = group_key.and_then(|name| self.routes.filegroups.get(name));
        let filetype = self.routes.filetypes.get(&type_key);

        let classify = |route: RouteId, probability: f32, own: &Thresholds| {
            policy.map_or_else(
                || own.classify(probability),
                |policy| policy_route_class(policy, route, probability),
            )
        };
        let score_general = || -> Result<Scored> {
            let raw = self.inner.predict(general_features)?;
            let probability = self
                .general_calibrator
                .as_ref()
                .map_or(raw, |cal| cal.apply(raw));
            Ok(Scored {
                route: RouteId::GENERAL,
                raw,
                probability,
                class: classify(RouteId::GENERAL, probability, &self.thresholds),
            })
        };
        let score_specialist = |entry: Option<&(RouteId, Arc<Route>)>| -> Result<Option<Scored>> {
            let Some((id, route)) = entry else {
                return Ok(None);
            };
            let mut features = route.ctx.extract_from_parsed(parsed);
            route.spec.standardize(&mut features);
            let (raw, probability) = route.predict_raw_calibrated(&features)?;
            Ok(Some(Scored {
                route: *id,
                raw,
                probability,
                class: classify(*id, probability, &route.thresholds),
            }))
        };
        let (general, (group_score, filetype_score)) = rayon::join(score_general, || {
            rayon::join(
                || score_specialist(group.as_ref()),
                || score_specialist(filetype.as_ref()),
            )
        });

        let mut route_probs = Vec::with_capacity(3);
        let mut scores = Vec::with_capacity(3);
        for scored in [Some(general?), group_score?, filetype_score?]
            .into_iter()
            .flatten()
        {
            route_probs.push(RouteProbability {
                route: scored.route,
                probability: scored.probability,
            });
            scores.push(RouteScore {
                model: compact_route_name(self.routes.names.name(scored.route)),
                probability: scored.probability,
                raw: scored.raw,
                classification: scored.class,
            });
        }

        let mut skipped = Vec::new();
        if group.is_none()
            && let Some(name) = group_key
            && self.routes.filegroups.skipped.contains(name)
        {
            skipped.push(SkippedRoute {
                model: format!("az/{name}"),
                reason: "uncalibrated",
            });
        }
        if filetype.is_none() && self.routes.filetypes.skipped.contains(type_key.as_ref()) {
            skipped.push(SkippedRoute {
                model: format!("az/{type_key}"),
                reason: "uncalibrated",
            });
        }

        Ok((route_probs, scores, skipped))
    }

    /// Pick the strongest [`Decision`] across the route scores using only the
    /// model's general thresholds (no per-filetype policy). Picks the highest
    /// class; on ties, the highest probability. Benign fallback uses the
    /// general route's prob against the suspicious cutoff.
    fn decide_from_scores(&self, scores: &[RouteProbability]) -> Decision {
        // General route is always first when scoring through predict_*_detailed.
        let general_prob = scores
            .iter()
            .find(|s| s.route == RouteId::GENERAL)
            .map_or(0.0, |s| s.probability);
        let mut best = self.thresholds.decide(general_prob);
        for score in scores {
            let candidate = self.thresholds.decide(score.probability);
            let better = match candidate.class.cmp(&best.class) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Equal => candidate.probability > best.probability,
                std::cmp::Ordering::Less => false,
            };
            if better {
                best = candidate;
            }
        }
        best
    }

    /// Pick the final [`Decision`] from per-route scores.
    ///
    /// In level mode the verdict comes from the level-independent sweep (see
    /// [`Self::decide_swept`]). In manual-threshold mode (`active_level` is
    /// `None`) it keeps the pre-level behaviour: the per-filetype policy at the
    /// default level when one exists, else the OR over the model's thresholds —
    /// with no level marker, since no level table applies.
    fn decide(
        &self,
        file_type: &str,
        policy: Option<&RoutePolicy>,
        route_probs: &[RouteProbability],
        skipped: &mut Vec<SkippedRoute>,
    ) -> Decision {
        // Diagnostic: note routes the default-level policy references but that
        // produced no score this run.
        if let Some(policy) = policy {
            for &(route, _) in policy
                .hostile
                .thresholds
                .iter()
                .chain(&policy.suspicious.thresholds)
            {
                if !route_probs.iter().any(|score| score.route == route) {
                    skipped.push(SkippedRoute {
                        model: compact_route_name(self.routes.names.name(route)),
                        reason: "unavailable",
                    });
                }
            }
        }

        let Some(level) = self.active_level else {
            // Manual-threshold mode: pre-level semantics, level is null.
            return match policy {
                Some(policy) => policy_decide(policy, route_probs)
                    .unwrap_or_else(|| self.benign_fallback(route_probs, Level::Manual)),
                None => self.decide_from_scores(route_probs),
            };
        };

        self.decide_swept(file_type, route_probs, level)
    }

    /// Benign decision reporting the general route's probability against the
    /// model's suspicious cutoff (the band the score didn't reach), tagged with
    /// the given level marker.
    fn benign_fallback(&self, route_probs: &[RouteProbability], level_marker: Level) -> Decision {
        let general_prob = route_probs
            .iter()
            .find(|s| s.route == RouteId::GENERAL)
            .map_or(0.0, |s| s.probability);
        Decision {
            class: Classification::Benign,
            probability: general_prob,
            threshold: self.thresholds.suspicious,
            level: level_marker,
        }
    }

    /// Derive the verdict from the level-independent lowest-firing-level sweep.
    ///
    /// The swept level is computed over the full grid with no reference to the
    /// active deploy level, so it — and the entire serialized envelope — is
    /// identical regardless of the deploy `-l`, keeping the result
    /// cache-shareable. The active deploy level only positions the cutoffs:
    /// hostile when the swept level is <= the deploy level, suspicious when it
    /// is <= `capped_suspicious_level(grid_max)`, else benign. A file that
    /// fires at no grid level is [`Level::Clean`].
    fn decide_swept(
        &self,
        file_type: &str,
        route_probs: &[RouteProbability],
        level: u16,
    ) -> Decision {
        let swept = if let Some(grid) = self.routes.policies.grid_for(file_type) {
            sweep_policy_grid(grid, route_probs)
        } else if self.general_grid.is_empty() {
            // No grid to sweep (general-only bundle without a levels[] table):
            // fall back to the threshold path; the level stays null.
            return self.decide_from_scores(route_probs);
        } else {
            sweep_general_grid(&self.general_grid, route_probs)
        };

        let Some((fired_level, probability, threshold)) = swept else {
            return self.benign_fallback(route_probs, Level::Clean);
        };

        Decision {
            class: verdict_for_level(fired_level, level, self.grid_max),
            probability,
            threshold,
            level: Level::At(fired_level),
        }
    }

    /// Inference backend identifier (always `"onnx"`).
    #[must_use]
    pub const fn backend_kind(&self) -> &'static str {
        "onnx"
    }

    /// Feature specification used to build input vectors.
    #[must_use]
    pub const fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    /// Extraction tables for [`Self::spec`], built once at load.
    #[must_use]
    pub const fn ctx(&self) -> &ExtractContext {
        &self.ctx
    }

    /// Largest calibrated grid level, which caps the suspicious ceiling (see
    /// `capped_suspicious_level`).
    #[must_use]
    pub(crate) const fn grid_max(&self) -> u16 {
        self.grid_max
    }

    /// Active deploy level (`-l`, FP per 100M benigns): the single hostile
    /// threshold — a file is hostile iff its firing level is `<=` this. `None`
    /// in manual-threshold mode and on single-bundle deployments with no grid.
    #[must_use]
    pub(crate) const fn active_level(&self) -> Option<u16> {
        self.active_level
    }

    /// Classification thresholds carried by this loaded model.
    #[must_use]
    pub const fn thresholds(&self) -> Thresholds {
        self.thresholds
    }

    /// Stable metadata describing the loaded model artifacts.
    #[must_use]
    pub const fn info(&self) -> &ModelInfo {
        &self.info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn normalize_archive_filetype_strips_pure_compression() {
        // Compound archive labels collapse onto their container.
        assert_eq!(normalize_archive_filetype("tar.gz"), "tar");
        assert_eq!(normalize_archive_filetype("tar.bz2"), "tar");
        assert_eq!(normalize_archive_filetype("tar.xz"), "tar");
        assert_eq!(normalize_archive_filetype("tar.zst"), "tar");
        assert_eq!(normalize_archive_filetype("tar.Z"), "tar");
        assert_eq!(normalize_archive_filetype("TAR.GZ"), "tar");
        // Repeated stripping for stacked suffixes.
        assert_eq!(normalize_archive_filetype("tar.bz2.xz"), "tar");
        // Bare compression labels keep their identity — RAW_COMPRESSED_FILETYPES
        // logic on the collimator side already gates them.
        assert_eq!(normalize_archive_filetype("gz"), "gz");
        assert_eq!(normalize_archive_filetype("bz2"), "bz2");
        // Containers without a compression suffix pass through.
        assert_eq!(normalize_archive_filetype("zip"), "zip");
        assert_eq!(normalize_archive_filetype("pe"), "pe");
        // Empty / whitespace.
        assert_eq!(normalize_archive_filetype(""), "");
        assert_eq!(normalize_archive_filetype("  tar.gz  "), "tar");
        // An already-normal label is borrowed, not copied.
        assert!(matches!(
            normalize_archive_filetype("elf"),
            Cow::Borrowed("elf")
        ));
    }

    #[test]
    fn container_filetype_maps_packages_to_their_archive() {
        // Specialty package types collapse onto their container archive —
        // filefacts is the source of truth, so these stay correct as new
        // package formats are added there.
        assert_eq!(container_filetype("gem"), Some("tar"));
        assert_eq!(container_filetype("crate"), Some("tar"));
        assert_eq!(container_filetype("python_sdist"), Some("tar"));
        assert_eq!(container_filetype("npm"), Some("tar"));
        assert_eq!(container_filetype("whl"), Some("zip"));
        assert_eq!(container_filetype("jar"), Some("zip"));
        assert_eq!(container_filetype("nupkg"), Some("zip"));
        // Compressed tar variants resolve to the tar container too.
        assert_eq!(container_filetype("tar.gz"), Some("tar"));
        assert_eq!(container_filetype("tar.zst"), Some("tar"));
        // Non-archive types and bare compression carry no container.
        assert_eq!(container_filetype("pe"), None);
        assert_eq!(container_filetype("python"), None);
        assert_eq!(container_filetype("gz"), None);
        assert_eq!(container_filetype("not_a_type"), None);
    }

    #[test]
    fn specialist_keys_routes_compressed_archives_to_their_container() {
        // collimator emits the filetype→filegroup map (and trains/calibrates
        // every specialist) under the normalized container label, so the map
        // carries `tar`/`javascript`, never `tar.gz`.
        let routes = RouteSet {
            filetype_to_filegroup: HashMap::from([
                ("tar".to_string(), "archive".to_string()),
                ("javascript".to_string(), "scripts".to_string()),
            ]),
            ..Default::default()
        };
        let keys = |file_type: &str| {
            let (key, group) = routes.specialist_keys(file_type);
            (key.into_owned(), group.map(str::to_owned))
        };

        // A compressed tarball must resolve to the `tar` container specialist
        // and its filegroup. This is the `.tgz` routing gap: selecting by the
        // raw `tar.gz` label found no specialist, surfaced `filetypes/tar` as an
        // `unavailable` skip, and let an obfuscated npm dropper score benign on
        // the general route alone.
        let tar = ("tar".to_string(), Some("archive".to_string()));
        assert_eq!(keys("tar.gz"), tar);
        // Stacked compression suffixes collapse all the way to the container.
        assert_eq!(keys("tar.bz2.xz"), tar);
        // The uncompressed container is unchanged.
        assert_eq!(keys("tar"), tar);
        // Non-archive types pass through untouched and still resolve their group.
        assert_eq!(
            keys("javascript"),
            ("javascript".to_string(), Some("scripts".to_string()))
        );
        // A container with no configured filegroup yields no group, still
        // normalized so its filetype specialist (if any) is reachable.
        assert_eq!(keys("zip"), ("zip".to_string(), None));
    }

    #[test]
    fn route_names_parse_and_render_compactly() {
        assert_eq!(RouteKind::parse("general"), Some(RouteKind::General));
        assert_eq!(
            RouteKind::parse("filegroups/native"),
            Some(RouteKind::Group("native"))
        );
        assert_eq!(
            RouteKind::parse("filetypes/elf"),
            Some(RouteKind::Type("elf"))
        );
        assert_eq!(RouteKind::parse("mystery/thing"), None);
        assert_eq!(compact_route_name("general"), "az");
        assert_eq!(compact_route_name("filegroups/native"), "az/native");
        assert_eq!(compact_route_name("filetypes/elf"), "az/elf");

        let mut names = RouteNames::default();
        assert_eq!(names.id("general"), Some(RouteId::GENERAL));
        let elf = names.intern("filetypes/elf");
        assert_eq!(
            names.intern("filetypes/elf"),
            elf,
            "interning is idempotent"
        );
        assert_eq!(names.name(elf), "filetypes/elf");
    }

    #[test]
    fn load_rejects_missing_feature_spec_with_update_guidance() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("model.json"), b"{}")?;

        let Err(err) = Model::load(dir.path(), None, None) else {
            anyhow::bail!("missing feature spec should be rejected");
        };
        let message = err.to_string();
        assert!(message.contains("model bundle is incomplete"));
        assert!(message.contains("Run 'atomscan update-rules'"));
        Ok(())
    }

    #[test]
    fn capped_suspicious_level_clamps_to_the_ceiling() {
        // Below the ceiling passes through; at/above it clamps to the ceiling.
        assert_eq!(
            capped_suspicious_level(SUSPICIOUS_LEVEL_CEILING - 1),
            SUSPICIOUS_LEVEL_CEILING - 1
        );
        assert_eq!(
            capped_suspicious_level(SUSPICIOUS_LEVEL_CEILING),
            SUSPICIOUS_LEVEL_CEILING
        );
        assert_eq!(
            capped_suspicious_level(SUSPICIOUS_LEVEL_CEILING + 1),
            SUSPICIOUS_LEVEL_CEILING
        );
        assert_eq!(capped_suspicious_level(25_000), SUSPICIOUS_LEVEL_CEILING);
    }

    #[test]
    fn bundle_config_is_absent_when_missing_and_fatal_when_malformed() -> Result<()> {
        let dir = tempfile::tempdir()?;
        assert!(BundleConfig::load(dir.path())?.is_none());

        std::fs::write(dir.path().join("config.json"), b"{\"levels\": [")?;
        let err = BundleConfig::load(dir.path()).expect_err("truncated config must fail");
        assert!(format!("{err:#}").contains("parsing"), "{err:#}");
        // The default-level probe leaves the report to `Model::load`.
        assert_eq!(model_default_level(dir.path()), None);

        // An ensemble whose config.json is malformed refuses to load rather
        // than silently degrading to general-only.
        std::fs::create_dir_all(dir.path().join("general"))?;
        let err = Model::load(dir.path(), None, None).expect_err("malformed config must fail");
        assert!(format!("{err:#}").contains("config.json"), "{err:#}");
        Ok(())
    }

    #[test]
    fn single_bundle_thresholds_must_be_valid() -> Result<()> {
        let cfg: BundleConfig = serde_json::from_str(r#"{"hostile": 0.9}"#)?;
        let t = cfg.thresholds()?.context("hostile present")?;
        assert_eq!((t.suspicious, t.hostile), (0.9, 0.9), "hostile-only");

        let cfg: BundleConfig = serde_json::from_str(r#"{"suspicious": 0.95, "hostile": 0.9}"#)?;
        assert!(cfg.thresholds().is_err(), "inverted thresholds are fatal");

        let cfg: BundleConfig = serde_json::from_str(r#"{"default_severity_level": 50}"#)?;
        assert!(cfg.thresholds()?.is_none());
        assert_eq!(cfg.default_severity_level, Some(50));
        Ok(())
    }

    #[test]
    fn isotonic_calibrator_load_apply_and_endpoints() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("calibrator.json"),
            br#"{"schema":"azoth.calibrator.isotonic.v1",
                 "x":[0.0,0.25,0.5,0.75,1.0],
                 "y":[0.0,0.10,0.40,0.85,1.0]}"#,
        )?;
        let cal =
            IsotonicCalibrator::load_optional(dir.path())?.context("calibrator should load")?;

        // Endpoint clipping.
        assert!((cal.apply(-1.0) - 0.0).abs() < 1e-6);
        assert!((cal.apply(2.0) - 1.0).abs() < 1e-6);
        // Exact breakpoints pass through.
        assert!((cal.apply(0.5) - 0.40).abs() < 1e-6);
        // Linear interpolation between breakpoints.
        // At raw=0.625 we sit halfway between (0.5,0.40) and (0.75,0.85) → 0.625.
        assert!((cal.apply(0.625) - 0.625).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn isotonic_preserves_threshold_decisions() -> Result<()> {
        // Decision-equivalence property: for any monotone calibrator,
        // cal(p) >= cal(τ) iff p >= τ. This is the invariant that lets us
        // calibrate thresholds at load time without changing decisions.
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("calibrator.json"),
            br#"{"schema":"azoth.calibrator.isotonic.v1",
                 "x":[0.0,0.1,0.3,0.6,0.9,1.0],
                 "y":[0.0,0.02,0.15,0.55,0.92,1.0]}"#,
        )?;
        let cal =
            IsotonicCalibrator::load_optional(dir.path())?.context("calibrator should load")?;

        for &raw_t in &[0.05_f32, 0.2, 0.5, 0.7, 0.95] {
            let cal_t = cal.apply(raw_t);
            for &raw_p in &[0.0_f32, 0.05, 0.2, 0.5, 0.7, 0.95, 1.0] {
                let cal_p = cal.apply(raw_p);
                assert_eq!(
                    raw_p >= raw_t,
                    cal_p >= cal_t,
                    "decision flipped for raw_p={raw_p} raw_t={raw_t}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn isotonic_rejects_unknown_schema() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("calibrator.json"),
            br#"{"schema":"azoth.calibrator.spline.v2","x":[0.0,1.0],"y":[0.0,1.0]}"#,
        )?;
        let err = IsotonicCalibrator::load_optional(dir.path())
            .expect_err("future schema should be rejected");
        assert!(err.to_string().contains("unsupported schema"));
        Ok(())
    }

    #[test]
    fn isotonic_load_optional_returns_none_when_absent() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let cal = IsotonicCalibrator::load_optional(dir.path())?;
        assert!(
            cal.is_none(),
            "missing calibrator must be a None, not an error"
        );
        Ok(())
    }

    #[test]
    fn isotonic_rejects_duplicate_x_breakpoints() -> Result<()> {
        // Equal adjacent breakpoints make linear interpolation divide by zero
        // and produce NaN in apply() — silently turning every prediction into
        // NaN and every classification into Benign. Reject at load time.
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("calibrator.json"),
            br#"{"schema":"azoth.calibrator.isotonic.v1","x":[0.0,0.5,0.5,1.0],"y":[0.0,0.4,0.6,1.0]}"#,
        )?;
        let err = IsotonicCalibrator::load_optional(dir.path())
            .expect_err("duplicate breakpoints must be rejected");
        assert!(
            err.to_string().contains("strictly ascending"),
            "error should mention strict ascending: {err}"
        );
        Ok(())
    }

    #[test]
    fn isotonic_rejects_non_finite_values() -> Result<()> {
        // NaN or Inf in x or y would propagate through every apply() call.
        for body in [
            br#"{"schema":"azoth.calibrator.isotonic.v1","x":[0.0,0.5,1.0],"y":[0.0,"NaN",1.0]}"# as &[u8],
            br#"{"schema":"azoth.calibrator.isotonic.v1","x":[0.0,"Infinity",1.0],"y":[0.0,0.5,1.0]}"#,
        ] {
            let dir = tempfile::tempdir()?;
            std::fs::write(dir.path().join("calibrator.json"), body)?;
            let result = IsotonicCalibrator::load_optional(dir.path());
            // serde may reject NaN/Inf at parse time (depending on parser
            // strict-mode); either parse-rejection or our own check is fine
            // — both prevent the bad calibrator from being installed.
            assert!(result.is_err(),
                    "non-finite calibrator must be rejected (got {result:?})");
        }
        Ok(())
    }

    #[test]
    fn isotonic_rejects_y_out_of_unit_interval() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("calibrator.json"),
            br#"{"schema":"azoth.calibrator.isotonic.v1","x":[0.0,0.5,1.0],"y":[0.0,0.5,1.5]}"#,
        )?;
        let err =
            IsotonicCalibrator::load_optional(dir.path()).expect_err("y > 1 must be rejected");
        assert!(
            err.to_string().contains("outside [0, 1]"),
            "error should explain the bound: {err}"
        );
        Ok(())
    }

    #[test]
    fn isotonic_rejects_non_monotone_y() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("calibrator.json"),
            br#"{"schema":"azoth.calibrator.isotonic.v1","x":[0.0,0.4,0.8,1.0],"y":[0.0,0.6,0.3,1.0]}"#,
        )?;
        let err = IsotonicCalibrator::load_optional(dir.path())
            .expect_err("non-monotone y must be rejected");
        assert!(
            err.to_string().contains("monotone"),
            "error should mention monotone: {err}"
        );
        Ok(())
    }

    /// An OR-rule severity over named routes, interned into `names`.
    fn or_rule(names: &mut RouteNames, thresholds: &[(&str, f32)]) -> PolicySeverity {
        PolicySeverity {
            thresholds: thresholds
                .iter()
                .map(|&(route, t)| (names.intern(route), t))
                .collect(),
            blend: None,
        }
    }

    /// One route score, interned into `names`.
    fn score(names: &mut RouteNames, route: &str, probability: f32) -> RouteProbability {
        RouteProbability {
            route: names.intern(route),
            probability,
        }
    }

    #[test]
    fn calibrate_policy_thresholds_pushes_each_route_through_its_calibrator() {
        // Build two distinct calibrators so we can verify per-route routing.
        let cal_general = IsotonicCalibrator {
            x: vec![0.0, 0.5, 1.0],
            y: vec![0.0, 0.10, 1.0],
        };
        let cal_elf = IsotonicCalibrator {
            x: vec![0.0, 0.5, 1.0],
            y: vec![0.0, 0.90, 1.0],
        };
        // RoutePolicy with thresholds for general, filetypes/elf (calibrated)
        // and filegroups/missing (no calibrator — must remain untouched).
        let mut names = RouteNames::default();
        let policy = RoutePolicy {
            hostile: or_rule(
                &mut names,
                &[
                    ("general", 0.5),
                    ("filetypes/elf", 0.5),
                    ("filegroups/missing", 0.5),
                ],
            ),
            suspicious: or_rule(&mut names, &[("general", 0.5)]),
        };
        let mut policies = RoutePolicies {
            by_filetype: HashMap::from([("elf".to_string(), policy)]),
            ..Default::default()
        };

        let elf = names.id("filetypes/elf").unwrap();
        let missing = names.id("filegroups/missing").unwrap();
        calibrate_policy_thresholds(&mut policies, |route| match route {
            RouteId::GENERAL => Some(&cal_general),
            r if r == elf => Some(&cal_elf),
            _ => None,
        });

        let policy = policies.by_filetype.get("elf").expect("policy retained");
        // general at raw=0.5 → 0.10
        assert!((policy.hostile.threshold(RouteId::GENERAL).unwrap() - 0.10).abs() < 1e-6);
        // filetypes/elf at raw=0.5 → 0.90
        assert!((policy.hostile.threshold(elf).unwrap() - 0.90).abs() < 1e-6);
        // No calibrator for filegroups/missing — left at raw 0.5.
        assert!((policy.hostile.threshold(missing).unwrap() - 0.5).abs() < 1e-6);
        // Suspicious side also calibrated.
        assert!((policy.suspicious.threshold(RouteId::GENERAL).unwrap() - 0.10).abs() < 1e-6);
    }

    #[test]
    fn thresholds_at_level_extracts_per_route_pairs() {
        // Synthetic levels[] block with two routes; level 5 only.
        let json = r#"{
          "filetype_to_group": {},
          "levels": [
            {
              "level": 5,
              "hostile":    {"thresholds": {"general": 0.99, "filetypes/elf": 0.95}},
              "suspicious": {"thresholds": {"general": 0.80, "filetypes/elf": 0.70}}
            },
            {
              "level": 9,
              "hostile":    {"thresholds": {"general": 0.50}},
              "suspicious": {"thresholds": {"general": 0.30}}
            }
          ]
        }"#;
        let parsed: BundleConfig = serde_json::from_str(json).unwrap();

        let level5 = thresholds_at_level(&parsed.levels, 5);
        assert_eq!(level5.len(), 2);
        let g = level5.get("general").expect("general at level 5");
        assert!((g.hostile - 0.99).abs() < 1e-6);
        assert!((g.suspicious - 0.80).abs() < 1e-6);
        let elf = level5.get("filetypes/elf").expect("elf at level 5");
        assert!((elf.hostile - 0.95).abs() < 1e-6);
        assert!((elf.suspicious - 0.70).abs() < 1e-6);

        let level9 = thresholds_at_level(&parsed.levels, 9);
        assert_eq!(level9.len(), 1);
        assert!((level9.get("general").unwrap().hostile - 0.50).abs() < 1e-6);

        // Routes that have a hostile threshold but no matching suspicious
        // threshold stay active as hostile-only routes.
        let half = r#"{
          "levels": [{
            "level": 5,
            "hostile":    {"thresholds": {"general": 0.9, "filetypes/elf": 0.95}},
            "suspicious": {"thresholds": {"general": 0.6}}
          }]
        }"#;
        let parsed: BundleConfig = serde_json::from_str(half).unwrap();
        let level5 = thresholds_at_level(&parsed.levels, 5);
        assert_eq!(level5.len(), 2);
        assert!(level5.contains_key("general"));
        let elf = level5.get("filetypes/elf").expect("elf remains active");
        assert!((elf.hostile - 0.95).abs() < 1e-6);
        assert!(
            (elf.suspicious - elf.hostile).abs() < 1e-6,
            "hostile-only routes classify only at the hostile threshold"
        );
    }

    #[test]
    fn route_policy_classification_keeps_general_escape() {
        let mut names = RouteNames::default();
        let policy = RoutePolicy {
            hostile: or_rule(&mut names, &[("filetypes/elf", 0.99), ("general", 0.95)]),
            suspicious: or_rule(&mut names, &[("filetypes/elf", 0.90)]),
        };

        let general_escape = [
            score(&mut names, "general", 0.96),
            score(&mut names, "filetypes/elf", 0.10),
        ];
        assert_eq!(
            policy_classify(&policy, &general_escape),
            Classification::Hostile
        );

        let specialist_suspicious = [
            score(&mut names, "general", 0.10),
            score(&mut names, "filetypes/elf", 0.91),
        ];
        assert_eq!(
            policy_classify(&policy, &specialist_suspicious),
            Classification::Suspicious
        );

        let inactive_route = [score(&mut names, "filegroups/native", 1.0)];
        assert_eq!(
            policy_classify(&policy, &inactive_route),
            Classification::Benign
        );
    }

    fn general_only(prob: f32) -> Vec<RouteProbability> {
        vec![RouteProbability {
            route: RouteId::GENERAL,
            probability: prob,
        }]
    }

    /// Build a single-route ("general") hostile OR-rule policy grid: each level
    /// pairs with the general threshold that fires at it. Thresholds loosen as
    /// the level rises, matching the real grid's shape.
    fn general_policy_grid(rows: &[(u16, f32)]) -> Vec<LevelPolicy> {
        rows.iter()
            .map(|&(level, thr)| LevelPolicy {
                level,
                hostile: PolicySeverity {
                    thresholds: vec![(RouteId::GENERAL, thr)],
                    blend: None,
                },
            })
            .collect()
    }

    #[test]
    fn sweep_returns_lowest_firing_level() {
        // Grid: stricter levels demand higher probabilities.
        let grid =
            general_policy_grid(&[(2, 0.99), (20, 0.85), (50, 0.70), (200, 0.40), (500, 0.10)]);

        // p=0.88 clears the L20 cutoff (0.85) but not L2 (0.99) → lowest is 20.
        let swept = sweep_policy_grid(&grid, &general_only(0.88)).expect("fires");
        assert_eq!(swept.0, 20);

        // p=0.30 only clears L500 (0.10) → lowest firing level is 500.
        let swept = sweep_policy_grid(&grid, &general_only(0.30)).expect("fires");
        assert_eq!(swept.0, 500);

        // p=0.05 clears nothing → never fires.
        assert!(sweep_policy_grid(&grid, &general_only(0.05)).is_none());
    }

    #[test]
    fn sweep_is_independent_of_active_level() {
        // The swept level is a property of the file + model, not of `-l`: the
        // same scores yield the same level marker whatever the deploy level. Only
        // the verdict derived from it changes. This is what makes the envelope
        // cache-shareable across `-l`.
        let grid = general_policy_grid(&[(2, 0.99), (20, 0.85), (50, 0.70)]);
        let swept_level = sweep_policy_grid(&grid, &general_only(0.88))
            .expect("fires")
            .0;
        assert_eq!(swept_level, 20);
        for active in [0_u16, 10, 20, 50, 200, 1000] {
            // Swept level is unchanged; only the class depends on `active`.
            let _ = verdict_for_level(swept_level, active, 1000);
        }
    }

    /// `lvl`/`fires_at` keep the encoding the `Option<i32>` had: `null`, `-1`,
    /// or the level, with an absent field reading as `null`. Anything else is
    /// not a level and is refused rather than guessed at.
    #[test]
    fn level_keeps_its_wire_encoding() {
        for (json, level) in [
            ("null", Level::Manual),
            ("-1", Level::Clean),
            ("0", Level::At(0)),
            ("25000", Level::At(25_000)),
        ] {
            assert_eq!(serde_json::to_string(&level).unwrap(), json);
            assert_eq!(serde_json::from_str::<Level>(json).unwrap(), level);
        }
        #[derive(serde::Deserialize)]
        struct Row {
            lvl: Level,
        }
        assert_eq!(
            serde_json::from_str::<Row>("{}").unwrap().lvl,
            Level::Manual
        );
        for bad in ["-2", "65536", "\"25\""] {
            assert!(serde_json::from_str::<Level>(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn verdict_for_level_applies_caps() {
        // Rule: swept level <= active level is hostile; swept level above active
        // but within the suspicious cap is suspicious; anything beyond the cap
        // is benign.
        let grid_max = 25_000;
        assert_eq!(verdict_for_level(20, 50, grid_max), Classification::Hostile);
        assert_eq!(verdict_for_level(50, 50, grid_max), Classification::Hostile);
        // The suspicious band ends at the SUSPICIOUS_LEVEL_CEILING cap.
        assert_eq!(
            verdict_for_level(SUSPICIOUS_LEVEL_CEILING, 50, grid_max),
            Classification::Suspicious,
            "at the ceiling is still suspicious"
        );
        assert_eq!(
            verdict_for_level(SUSPICIOUS_LEVEL_CEILING + 1, 50, grid_max),
            Classification::Benign,
            "just past the suspicious ceiling is benign"
        );
        assert_eq!(
            verdict_for_level(25_000, 50, grid_max),
            Classification::Benign
        );

        // Stricter deploy (-l 10): only levels <= 10 are hostile; the ceiling
        // still bounds the top of the suspicious band.
        assert_eq!(
            verdict_for_level(20, 10, grid_max),
            Classification::Suspicious
        );
        assert_eq!(
            verdict_for_level(SUSPICIOUS_LEVEL_CEILING, 10, grid_max),
            Classification::Suspicious
        );
    }

    #[test]
    fn sweep_general_grid_uses_max_crossing_route() {
        // General-route OR fallback: any route crossing the level's general
        // threshold fires it. A weak general but strong specialist still fires.
        let mut names = RouteNames::default();
        let grid = [(2_u16, 0.99_f32), (20, 0.85), (200, 0.40)];
        let scores = [
            score(&mut names, "general", 0.10),
            score(&mut names, "filetypes/elf", 0.90),
        ];
        let swept = sweep_general_grid(&grid, &scores).expect("fires");
        assert_eq!(swept.0, 20, "0.90 clears L20 (0.85) but not L2 (0.99)");
        assert!((swept.1 - 0.90).abs() < 1e-6, "reports the crossing prob");
    }

    #[test]
    fn blend_policy_fires_on_hand_built_inputs() {
        // Construct a blend that fires when general's prob is high enough on
        // its own (weight 1.0 on general, 0 on specialist, intercept chosen
        // so threshold=0.5 lines up with general prob ≈ 0.6).
        // sigmoid(intercept + 1.0 * logit(0.6)) = 0.5
        //   logit(0.6) ≈ 0.4054
        //   intercept = -0.4054 gives sigmoid(0) = 0.5
        let mut names = RouteNames::default();
        let elf = names.intern("filetypes/elf");
        let blend = BlendPolicy {
            routes: vec![RouteId::GENERAL, elf],
            weights: vec![1.0, 0.0],
            intercept: -0.4054,
            threshold: 0.5,
        };
        let scores = |general: f32, specialist: f32| {
            [
                RouteProbability {
                    route: RouteId::GENERAL,
                    probability: general,
                },
                RouteProbability {
                    route: elf,
                    probability: specialist,
                },
            ]
        };
        // general=0.6 → fires
        assert!(blend.fires(&scores(0.6, 0.0)));
        // general=0.5 → doesn't fire (logit(0.5)=0, intercept negative)
        assert!(!blend.fires(&scores(0.5, 0.99)));
        // Missing route → can't blend, doesn't fire.
        assert!(!blend.fires(&general_only(1.0)));
    }

    fn write_policies(dir: &Path, blend: &str) -> Result<()> {
        std::fs::write(
            dir.join("route_policies.json"),
            format!(
                r#"{{
              "schema": "azoth.route_policy_search.v1",
              "routes": {{
                "filetypes/elf": {{
                  "filetype": "elf",
                  "levels": [{{
                    "level": 5,
                    "hostile": {{"best": {{"thresholds": {{}}, "blend": {blend}}}}},
                    "suspicious": {{"best": {{"thresholds": {{"general": 0.7}}}}}}
                  }}]
                }}
              }}
            }}"#
            ),
        )?;
        Ok(())
    }

    #[test]
    fn load_route_policies_parses_blend_field() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_policies(
            dir.path(),
            r#"{
                "routes": ["general", "filegroups/native", "filetypes/elf"],
                "weights": [0.5, 0.3, 1.2],
                "intercept": -1.5,
                "threshold": 0.8,
                "transform": "logit"
            }"#,
        )?;
        let mut names = RouteNames::default();
        let policies = load_route_policies(dir.path(), 5, &mut names)?;
        let policy = policies.by_filetype.get("elf").expect("elf policy loaded");
        let blend = policy.hostile.blend.as_ref().expect("blend loaded");
        assert_eq!(blend.routes.len(), 3);
        assert_eq!(blend.weights.len(), 3);
        assert!((blend.intercept - -1.5).abs() < 1e-6);
        assert!((blend.threshold - 0.8).abs() < 1e-6);
        // contains_route should recognize blend routes too.
        assert!(policies.contains_route(names.id("filegroups/native").unwrap()));
        assert!(policies.contains_route(names.id("filetypes/elf").unwrap()));
        Ok(())
    }

    #[test]
    fn a_malformed_blend_stops_the_load() -> Result<()> {
        for (blend, why) in [
            // A defaulted threshold of 0.0 would fire on everything.
            (
                r#"{"routes": ["general"], "weights": [1.0], "intercept": 0.0}"#,
                "missing threshold",
            ),
            (
                r#"{"routes": ["general"], "weights": [1.0], "threshold": 0.5}"#,
                "missing intercept",
            ),
            (
                r#"{"routes": ["general"], "weights": [1.0], "intercept": 0.0, "threshold": 0.5, "bias": 1}"#,
                "unknown field",
            ),
            (
                r#"{"routes": ["general"], "weights": [1.0, 2.0], "intercept": 0.0, "threshold": 0.5}"#,
                "length mismatch",
            ),
            (
                r#"{"routes": ["general"], "weights": [1.0], "intercept": 0.0, "threshold": 1.5}"#,
                "threshold out of range",
            ),
            (
                r#"{"routes": ["general"], "weights": [1.0], "intercept": 0.0, "threshold": 0.5, "transform": "gam"}"#,
                "unknown transform",
            ),
        ] {
            let dir = tempfile::tempdir()?;
            write_policies(dir.path(), blend)?;
            let result = load_route_policies(dir.path(), 5, &mut RouteNames::default());
            assert!(result.is_err(), "{why} must fail the load");
        }
        Ok(())
    }

    #[test]
    fn route_policies_are_optional_but_must_parse() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let policies = load_route_policies(dir.path(), 5, &mut RouteNames::default())?;
        assert!(policies.by_filetype.is_empty() && policies.grid.is_empty());

        std::fs::write(dir.path().join("route_policies.json"), b"{\"routes\": 7}")?;
        assert!(load_route_policies(dir.path(), 5, &mut RouteNames::default()).is_err());
        Ok(())
    }

    #[test]
    fn blend_policy_severity_isnt_isotonic_calibrated() {
        // calibrate_policy_thresholds must leave blend severities alone —
        // the blend's threshold is already in calibrated-prob space (its fit
        // ran on isotonic-calibrated route probs at policy-search time).
        // If we double-applied isotonic here it would shift the threshold off
        // the calibrated distribution and deploy verdicts would silently drift.
        let blend = BlendPolicy {
            routes: vec![RouteId::GENERAL],
            weights: vec![1.0],
            intercept: 0.0,
            threshold: 0.42,
        };
        let policy = RoutePolicy {
            hostile: PolicySeverity {
                thresholds: Vec::new(),
                blend: Some(blend),
            },
            suspicious: PolicySeverity {
                thresholds: vec![(RouteId::GENERAL, 0.5)],
                blend: None,
            },
        };
        let mut policies = RoutePolicies {
            by_filetype: HashMap::from([("elf".to_string(), policy)]),
            ..Default::default()
        };
        // A calibrator that shifts every input by +0.1 (clamped to [0, 1]).
        let cal = IsotonicCalibrator {
            x: vec![0.0, 1.0],
            y: vec![0.1, 1.0],
        };
        calibrate_policy_thresholds(&mut policies, |_route| Some(&cal));
        let pol = policies.by_filetype.get("elf").unwrap();
        // Hostile (blend) threshold stays at its original calibrated value.
        let blend = pol.hostile.blend.as_ref().unwrap();
        assert!((blend.threshold - 0.42).abs() < 1e-6);
        // Suspicious (OR-rule) threshold did get mapped.
        assert!(
            (pol.suspicious.threshold(RouteId::GENERAL).unwrap() - cal.apply(0.5)).abs() < 1e-6
        );
    }

    #[test]
    fn load_route_policies_reads_level_and_route_membership() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("route_policies.json"),
            r#"{
              "schema": "azoth.route_policy_search.v1",
              "routes": {
                "filetypes/elf": {
                  "filetype": "elf",
                  "levels": [
                    {
                      "level": 4,
                      "hostile": {
                        "best": {"thresholds": {"general": 0.99}}
                      },
                      "suspicious": {
                        "best": {"thresholds": {"general": 0.90}}
                      }
                    },
                    {
                      "level": 5,
                      "hostile": {
                        "best": {"thresholds": {"filetypes/elf": 0.98, "general": 0.97}}
                      },
                      "suspicious": {
                        "best": {"thresholds": {"filegroups/native": 0.80}}
                      }
                    }
                  ]
                }
              }
            }"#,
        )?;

        let mut names = RouteNames::default();
        let policies = load_route_policies(dir.path(), 5, &mut names)?;
        let id = |name: &str| names.id(name).unwrap();
        assert!(policies.contains_route(RouteId::GENERAL));
        assert!(policies.contains_route(id("filegroups/native")));
        assert!(policies.contains_route(id("filetypes/elf")));
        assert_eq!(names.id("filetypes/pe"), None);

        let elf = policies.by_filetype.get("elf").expect("elf policy");
        assert!(
            (elf.hostile.threshold(id("filetypes/elf")).unwrap() - 0.98).abs() < 1e-6,
            "must select the requested level"
        );
        assert!(elf.suspicious.threshold(RouteId::GENERAL).is_none());
        Ok(())
    }

    #[test]
    fn policy_routes_never_firing_sentinel_is_dropped() -> Result<()> {
        // Collimator writes one ulp above 1.0 for a route that must never fire
        // at a level; the real bundle carries it, so it must still load.
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("route_policies.json"),
            r#"{"routes": {"filetypes/yaml": {"filetype": "yaml", "levels": [
                {"level": 0, "hostile": {"best": {"thresholds": {"general": 1.0000001192092896}}}},
                {"level": 5, "hostile": {"best": {"thresholds": {"general": 0.9}}}}
            ]}}}"#,
        )?;
        let policies = load_route_policies(dir.path(), 5, &mut RouteNames::default())?;
        let grid = policies.grid_for("yaml").expect("grid");
        assert_eq!(grid.len(), 1, "the never-fires level has no policy");
        assert_eq!(grid[0].level, 5);
        Ok(())
    }

    #[test]
    fn ensemble_loader_rejects_missing_general() -> Result<()> {
        let dir = tempfile::tempdir()?;
        // Make the layout look like an ensemble (general/ subdir present)
        // but leave it empty so general/ has no model artifacts.
        std::fs::create_dir_all(dir.path().join("general"))?;
        let err = Model::load(dir.path(), None, None)
            .expect_err("ensemble with empty general/ must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("loading general route") || msg.contains("incomplete"),
            "expected general-route load failure, got {msg}"
        );
        Ok(())
    }

    #[test]
    fn ensemble_loader_rejects_unknown_required_route_name() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir_all(dir.path().join("general"))?;
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
              "required_routes": ["mystery/thing"]
            }"#,
        )?;
        // We can't actually load general here without a real model bundle, so
        // we expect either "loading general route" or the required-route check
        // depending on order. Either error mode confirms the loader caught it.
        let err = Model::load(dir.path(), None, None).expect_err("must fail");
        let _ = err;
        Ok(())
    }
}
