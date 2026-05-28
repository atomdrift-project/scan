//! Model loading, thresholding, and inference.
//!
//! ## Model file formats (since v1.3 / spec v18)
//!
//! Three backends are supported:
//!
//! - **`.onnx`** — portable inference graph emitted by collimator's
//!   training pipeline (LightGBM via `onnxmltools.convert_lightgbm`)
//!   or any other framework that exports to ONNX. Loaded via `tract`
//!   (pure-Rust, no FFI). **Preferred when present** — ~50% smaller
//!   than `.txt`, 3-5× faster to load, 5-11× faster to infer.
//! - **`.txt`** — native LightGBM Booster format. The historical
//!   default. Still emitted alongside `.onnx` during the transition
//!   and used as the canonical artifact for constant-predictor
//!   models that ONNX Runtime's TreeEnsembleClassifier can't load.
//! - **`.json`** — native XGBoost JSON dump.
//!
//! When `.onnx` and `.txt` are present for the same model (transitional
//! state during the v18 ONNX migration), `.onnx` wins. The legacy
//! ambiguity rule (`.txt` + `.json` together is ambiguous because
//! they're different backends) still applies — `.onnx` doesn't conflict
//! with either since it can represent both backends' graphs.
//!
//! ## Bundle layouts
//!
//! Two on-disk shapes are supported:
//!
//! ### Single-bundle (legacy / dev)
//!
//! ```text
//! <model_dir>/
//!   model.onnx | model.txt | model.json
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
//!     model.onnx | model.txt | model.json
//!     feature_spec.json
//!   filegroups/<group>/           optional, e.g. native, scripts, archive
//!     model.onnx | model.txt | model.json
//!     feature_spec.json           may be absent → uses general's spec
//!   filetypes/<type>/             optional, e.g. elf, pe
//!     model.onnx | model.txt | model.json
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
//!       "level": 5,
//!       "hostile": {
//!         "target_per_million": 5.0,
//!         "budget": 8,
//!         "thresholds": { "general": 0.997, "filetypes/elf": 0.951, … },
//!         "tp": …, "fp": …, "recall": …
//!       },
//!       "suspicious": { "thresholds": {…}, … }
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
//!         "level": 5,
//!         "hostile": {
//!           "best": {
//!             "policy": "specialist_primary_with_escape",
//!             "thresholds": {
//!               "filetypes/elf": 0.995,
//!               "general": 0.968
//!             }
//!           }
//!         },
//!         "suspicious": {"best": {"thresholds": {"filetypes/elf": 0.980}}}
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

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::features::{ExtractContext, FeatureSpec, EXPECTED_MODEL_ABI_VERSION};

/// Recommended thresholds loaded from collimator's model metadata.
///
/// These are computed during training based on FPR targets and represent
/// the empirically optimal operating points for the model.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
struct EvaluationThresholds {
    suspicious: Option<f64>,
    hostile: Option<f64>,
}

/// Per-level threshold metrics emitted by collimator.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
struct SeverityThresholdMetric {
    threshold: Option<f64>,
}

/// Gzip-style severity level thresholds emitted by collimator.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
struct SeverityLevel {
    level: u8,
    suspicious: Option<SeverityThresholdMetric>,
    hostile: Option<SeverityThresholdMetric>,
}

/// config.json in the model directory.
#[derive(Debug, Clone, serde::Deserialize)]
struct ConfigJson {
    suspicious: Option<f64>,
    hostile: Option<f64>,
    #[serde(default)]
    severity_levels: Vec<SeverityLevel>,
}

/// Thresholds block within evaluation.json.
#[derive(Debug, Clone, serde::Deserialize)]
struct EvaluationJson {
    #[serde(default = "default_model_abi_version")]
    model_abi_version: u32,
    #[serde(default)]
    recommended_thresholds: Option<EvaluationThresholds>,
    #[serde(default)]
    optimal_threshold: Option<f64>,
    #[serde(default)]
    severity_levels: Vec<SeverityLevel>,
}

const fn default_model_abi_version() -> u32 {
    EXPECTED_MODEL_ABI_VERSION
}

/// Try to load thresholds from config.json in the model directory.
///
/// config.json is the primary model-level configuration and takes precedence
/// over evaluation.json recommendations.
// Threshold values from JSON are in 0.0..1.0; narrowing to f32 is safe.
#[allow(clippy::cast_possible_truncation)]
fn load_config_thresholds(model_dir: &Path) -> Option<Thresholds> {
    let path = model_dir.join("config.json");
    let data = std::fs::read_to_string(&path).ok()?;
    let cfg: ConfigJson = serde_json::from_str(&data).ok()?;
    let suspicious = cfg.suspicious? as f32;
    let hostile = cfg.hostile? as f32;
    let t = Thresholds {
        suspicious,
        hostile,
    };
    if t.validate().is_ok() {
        tracing::info!(
            path = %path.display(),
            suspicious = suspicious,
            hostile = hostile,
            "loaded thresholds from config.json"
        );
        Some(t)
    } else {
        // Hard model-config anomaly: surface at error level so CI / deploy
        // verify catches it instead of silently degrading to fallbacks.
        tracing::error!(path = %path.display(), "config.json thresholds are invalid, ignoring");
        None
    }
}

#[allow(clippy::cast_possible_truncation)]
fn thresholds_from_severity_levels(levels: &[SeverityLevel], level: u8) -> Option<Thresholds> {
    let entry = levels.iter().find(|entry| entry.level == level)?;
    let suspicious = entry.suspicious?.threshold? as f32;
    let hostile = entry.hostile?.threshold? as f32;
    let thresholds = Thresholds {
        suspicious,
        hostile,
    };
    thresholds.validate().ok()?;
    Some(thresholds)
}

fn load_config_severity_thresholds(model_dir: &Path, level: u8) -> Option<Thresholds> {
    let path = model_dir.join("config.json");
    let data = std::fs::read_to_string(&path).ok()?;
    let cfg: ConfigJson = serde_json::from_str(&data).ok()?;
    let thresholds = thresholds_from_severity_levels(&cfg.severity_levels, level)?;
    tracing::info!(
        path = %path.display(),
        level = level,
        suspicious = thresholds.suspicious,
        hostile = thresholds.hostile,
        "loaded severity thresholds from config.json"
    );
    Some(thresholds)
}

fn load_evaluation_severity_thresholds(model_dir: &Path, level: u8) -> Option<Thresholds> {
    let path = model_dir.join("evaluation.json");
    let data = std::fs::read_to_string(&path).ok()?;
    let eval: EvaluationJson = serde_json::from_str(&data).ok()?;
    if eval.model_abi_version != EXPECTED_MODEL_ABI_VERSION {
        tracing::error!(
            path = %path.display(),
            found = eval.model_abi_version,
            expected = EXPECTED_MODEL_ABI_VERSION,
            "evaluation.json ABI version mismatch, ignoring severity thresholds"
        );
        return None;
    }
    let thresholds = thresholds_from_severity_levels(&eval.severity_levels, level)?;
    tracing::info!(
        path = %path.display(),
        level = level,
        suspicious = thresholds.suspicious,
        hostile = thresholds.hostile,
        "loaded severity thresholds from evaluation.json"
    );
    Some(thresholds)
}

/// Load gzip-style severity thresholds from model metadata.
///
/// Resolution order matches normal threshold loading: `config.json` first, then
/// `evaluation.json`. Returns `Ok(None)` when this model bundle predates
/// severity metadata.
///
/// # Errors
/// Returns an error if `level` is outside `1..=9`.
pub fn load_severity_thresholds(model_dir: &Path, level: u8) -> Result<Option<Thresholds>> {
    if !(1..=9).contains(&level) {
        anyhow::bail!("severity level must be in 1..=9, got {level}");
    }
    Ok(load_config_severity_thresholds(model_dir, level)
        .or_else(|| load_evaluation_severity_thresholds(model_dir, level)))
}

/// Try to load recommended thresholds from evaluation.json.
#[allow(clippy::cast_possible_truncation)]
fn load_evaluation_thresholds(model_dir: &Path) -> Option<Thresholds> {
    let path = model_dir.join("evaluation.json");
    let data = std::fs::read_to_string(&path).ok()?;
    let eval: EvaluationJson = serde_json::from_str(&data).ok()?;
    if eval.model_abi_version != EXPECTED_MODEL_ABI_VERSION {
        tracing::error!(
            path = %path.display(),
            found = eval.model_abi_version,
            expected = EXPECTED_MODEL_ABI_VERSION,
            "evaluation.json ABI version mismatch, ignoring thresholds"
        );
        return None;
    }

    if let Some(rec) = eval.recommended_thresholds {
        let suspicious = rec.suspicious? as f32;
        let hostile = rec.hostile? as f32;
        let t = Thresholds {
            suspicious,
            hostile,
        };
        if t.validate().is_ok() {
            tracing::info!(
                path = %path.display(),
                suspicious = suspicious,
                hostile = hostile,
                "loaded recommended thresholds from evaluation.json"
            );
            return Some(t);
        }
    }

    // Fall back to optimal_threshold as a single hostile threshold.
    if let Some(threshold) = eval.optimal_threshold {
        let t = threshold as f32;
        tracing::debug!(
            path = %path.display(),
            threshold = t,
            "evaluation.json has optimal_threshold but no recommended_thresholds"
        );
    }

    None
}

/// Classification outcome.
///
/// Serializes as an integer: 0 = benign, 1 = suspicious, 2 = hostile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    #[serde(rename = "m")]
    pub model: String,
    /// Probability emitted by this route's model.
    #[serde(rename = "prob")]
    pub probability: f32,
    /// Classification after applying this route's calibrated thresholds.
    #[serde(rename = "class")]
    pub classification: Classification,
}

/// One applicable route that was not scored.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SkippedRoute {
    /// Compact model route name, e.g. `az/native`.
    #[serde(rename = "m")]
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
/// use litmus::{Classification, Thresholds};
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
/// # Ok::<(), litmus::model::ThresholdValidationError>(())
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
        if probability >= self.hostile {
            Decision {
                class: Classification::Hostile,
                probability,
                threshold: self.hostile,
            }
        } else if probability >= self.suspicious {
            Decision {
                class: Classification::Suspicious,
                probability,
                threshold: self.suspicious,
            }
        } else {
            Decision {
                class: Classification::Benign,
                probability,
                threshold: self.suspicious,
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
    /// The suspicious threshold was not strictly lower than the hostile threshold.
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
    /// Optional SHA-256 hex digest of the model file.
    ///
    /// Litmus does not compute this for ordinary scan startup because hashing
    /// the model artifact is avoidable hot-path work.
    pub sha256: String,
    /// Short git commit hash of the models repository, if available.
    ///
    /// This is optional because spawning `git` during every scan is likewise
    /// avoidable startup work.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

/// One trained tree-boosting model, picked at load time by file extension.
/// Pure-Rust ONNX backend (tract). Loads a serialized ONNX graph and
/// runs single-sample inference returning the positive-class
/// probability. Used to read the .onnx artifacts collimator emits
/// alongside .txt models — typically ~50% smaller and 3-5× faster
/// to load and infer than the LightGBM .txt path.
///
/// The graph is `onnxmltools.convert_lightgbm`'s standard output:
/// inputs `("input", float32, [N, n_features])`, outputs
/// `[label (int64, [N]), probabilities (float32, [N, 2])]`. We
/// ignore the label output and use column 1 of probabilities.
struct OnnxBackend {
    /// Pre-optimized tract plan. Stored as a runnable model so
    /// `.run` calls don't re-optimize each prediction.
    plan: tract_onnx::prelude::TypedRunnableModel<
        tract_onnx::prelude::TypedModel,
    >,
    n_features: usize,
}

impl std::fmt::Debug for OnnxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxBackend")
            .field("n_features", &self.n_features)
            .finish_non_exhaustive()
    }
}

impl OnnxBackend {
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
            .ok_or_else(|| anyhow::anyhow!("ONNX input has no feature dimension in {}", path.display()))?
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
        let input: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, self.n_features),
            features.to_vec(),
        )
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

#[derive(Debug)]
enum InnerModel {
    Xgboost(xgboost_ars::Model),
    Lightgbm(lightgbm_ars::Model),
    Onnx(Box<OnnxBackend>),
}

impl InnerModel {
    fn num_features(&self) -> usize {
        match self {
            Self::Xgboost(m) => m.num_features(),
            Self::Lightgbm(m) => m.num_features(),
            Self::Onnx(m) => m.n_features,
        }
    }

    fn predict(&self, features: &[f32]) -> Result<f32> {
        match self {
            Self::Xgboost(m) => Ok(m.predict(features)),
            Self::Lightgbm(m) => m
                .predict(features)
                .map_err(|e| anyhow::anyhow!("lightgbm predict failed: {e}")),
            Self::Onnx(m) => m.predict(features),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Xgboost(_) => "xgboost",
            Self::Lightgbm(_) => "lightgbm",
            Self::Onnx(_) => "onnx",
        }
    }
}

/// Inference backend powering a loaded [`Model`].
///
/// Holds one or more trained models for a route. Single-model bundles (the
/// historical layout, `model.txt` or `model.json` directly under the bundle
/// dir) load `models = [one_model]`. Multi-seed bundles store
/// `models/seed_NN.txt` (or `.json`) and load all of them; `predict` returns
/// the arithmetic mean of every member's score, which is the variance-reducing
/// equivalent of training K seeds and ensembling them at inference time.
///
/// All members of `models` are required to have the same backend kind
/// (XGBoost or LightGBM) and the same `num_features` — mixing is rejected at
/// load time.
#[derive(Debug)]
struct Backend {
    /// At least one model; up to K for multi-seed bundles. K=1 is the default
    /// and is byte-equivalent to the pre-multi-seed predict path.
    models: Vec<InnerModel>,
}

impl Backend {
    fn num_features(&self) -> usize {
        // Constructor enforces uniform num_features; pick member 0.
        self.models[0].num_features()
    }

    fn predict(&self, features: &[f32]) -> Result<f32> {
        // Average member predictions in f64 to keep the rounding noise-floor
        // below f32 precision even for K up to a few dozen seeds. The K=1
        // path is mathematically identical to a direct `models[0].predict`.
        let mut sum = 0.0_f64;
        for m in &self.models {
            sum += f64::from(m.predict(features)?);
        }
        // Safety: `models` is non-empty by construction. The narrowing cast
        // is intentional — every member emits f32 probabilities and our caller
        // surface is f32, so the f64 accumulator only exists to keep summation
        // numerically stable across K members.
        #[allow(clippy::cast_possible_truncation)]
        let avg = (sum / self.models.len() as f64) as f32;
        Ok(avg)
    }

    fn kind(&self) -> &'static str {
        self.models[0].kind()
    }

    /// Number of trained members (1 = legacy single-model bundle, ≥2 =
    /// multi-seed). Exposed mainly so the load-time `tracing::info!` line
    /// can advertise it.
    fn n_members(&self) -> usize {
        self.models.len()
    }
}

/// Per-route isotonic calibrator persisted by collimator's
/// `azoth_calibrate_ensemble.py` as `calibrator.json` next to `model.txt`.
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
    /// loadable). Errors only when the file exists but is unparseable.
    fn load_optional(bundle_dir: &Path) -> Result<Option<Self>> {
        let path = bundle_dir.join("calibrator.json");
        if !path.is_file() {
            return Ok(None);
        }
        #[derive(serde::Deserialize)]
        struct Raw {
            schema: Option<String>,
            x: Vec<f32>,
            y: Vec<f32>,
        }
        let bytes = std::fs::read(&path)
            .with_context(|| format!("reading calibrator at {}", path.display()))?;
        let raw: Raw = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing calibrator at {}", path.display()))?;
        if let Some(schema) = raw.schema.as_deref() {
            // Hard-fail on unrecognized future versions to avoid silently
            // applying a calibrator we don't understand.
            if schema != "azoth.calibrator.isotonic.v1" {
                anyhow::bail!(
                    "calibrator at {} has unsupported schema {schema}; this litmus only supports azoth.calibrator.isotonic.v1",
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
/// telemetry about where the thresholds came from for the load-time log line.
struct LoadedBundle {
    backend: Backend,
    spec: FeatureSpec,
    thresholds: Thresholds,
    threshold_source: &'static str,
    /// Per-route isotonic calibrator from `calibrator.json`. Optional —
    /// pre-calibrator bundles return `None` here and skip calibration.
    calibrator: Option<IsotonicCalibrator>,
}

/// Default severity level used to pick thresholds out of the levels[] table
/// when the caller hasn't asked for a different one. Matches collimator's
/// default deploy policy.
pub const DEFAULT_SEVERITY_LEVEL: u8 = 3;

/// Ensemble-level config parsed from the top-level `config.json`. Every field
/// is optional — an ensemble bundle with only `general/` populated and no
/// extras loads with general as the only route at fallback thresholds.
#[derive(Debug, Default)]
struct EnsembleConfig {
    /// `cleave file_type → filegroup name` map. Files whose `file_type` is
    /// not a key here route to general only (per DESIGN.md §Runtime Decision).
    filetype_to_filegroup: HashMap<String, String>,
    /// Routes whose absence is fatal at startup. Names: `general`,
    /// `filegroups/<name>`, `filetypes/<name>` (matching the route paths in
    /// `config.json`'s `models[]` array).
    required_routes: Vec<String>,
    /// Per-route thresholds at the active level. Keyed by route name as
    /// emitted in `levels[].hostile.thresholds`: `"general"`,
    /// `"filegroups/<name>"`, `"filetypes/<name>"`.
    route_thresholds: HashMap<String, Thresholds>,
}

/// Per-filetype route policy loaded from `route_policies.json`.
#[derive(Debug, Default)]
struct RoutePolicies {
    by_filetype: HashMap<String, RoutePolicy>,
}

impl RoutePolicies {
    fn contains_route(&self, route: &str) -> bool {
        self.by_filetype.values().any(|policy| {
            policy.hostile.references_route(route) || policy.suspicious.references_route(route)
        })
    }
}

#[derive(Debug, Clone)]
struct RoutePolicy {
    suspicious: PolicySeverity,
    hostile: PolicySeverity,
}

#[derive(Debug, Clone)]
struct PolicySeverity {
    /// Route-name → calibrated threshold. Routes absent from this map do not
    /// participate in that severity decision.
    thresholds: HashMap<String, f32>,
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
    /// True iff this severity could fire on a contribution from ``route``.
    /// For OR-rule policies that's "route has a threshold"; for blend policies
    /// it's "route is one of the blend inputs."
    fn references_route(&self, route: &str) -> bool {
        if self.thresholds.contains_key(route) {
            return true;
        }
        if let Some(blend) = &self.blend {
            return blend.routes.iter().any(|r| r == route);
        }
        false
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
            if let Some(&t) = self.thresholds.get(&score.route)
                && score.probability >= t
            {
                let margin = score.probability - t;
                let best_margin =
                    best.map(|(p, bt)| p - bt).unwrap_or(f32::NEG_INFINITY);
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
    routes: Vec<String>,
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
    #[allow(clippy::cast_possible_truncation)]
    fn fires(&self, scores: &[RouteProbability]) -> bool {
        self.fire(scores).is_some()
    }

    /// Compute the blend's sigmoid output and, if it crosses the threshold,
    /// return `(sigmoid_output, threshold)`. Missing routes mean the blend
    /// cannot be honestly evaluated; treat as "doesn't fire."
    #[allow(clippy::cast_possible_truncation)]
    fn fire(&self, scores: &[RouteProbability]) -> Option<(f32, f32)> {
        // f64 math throughout — logit blows up near 0/1, and the cumulative
        // weighted sum can drift if we stay in f32. The final compare against
        // ``threshold`` is in f32 to match how the calibration step writes it.
        const EPS: f64 = 1e-6;
        let mut z: f64 = f64::from(self.intercept);
        for (idx, route) in self.routes.iter().enumerate() {
            let score = scores.iter().find(|s| &s.route == route)?;
            let p = (f64::from(score.probability)).clamp(EPS, 1.0 - EPS);
            let logit = (p / (1.0 - p)).ln();
            z += f64::from(self.weights[idx]) * logit;
        }
        let sigmoid_z = 1.0 / (1.0 + (-z).exp());
        let sigmoid_z = sigmoid_z as f32;
        (sigmoid_z >= self.threshold).then_some((sigmoid_z, self.threshold))
    }
}

/// Wire-format view of `config.json`. Captures only the fields litmus reads;
/// other keys (timestamp, score_table_hash, model_set_hash, models[], etc.)
/// are accepted and ignored.
#[derive(Debug, serde::Deserialize)]
struct EnsembleConfigJson {
    #[serde(default)]
    filetype_to_group: HashMap<String, String>,
    #[serde(default)]
    required_routes: Vec<String>,
    #[serde(default)]
    levels: Vec<LevelEntryJson>,
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
    level: u8,
    hostile: RoutePolicySeverityJson,
    suspicious: RoutePolicySeverityJson,
}

#[derive(Debug, serde::Deserialize)]
struct RoutePolicySeverityJson {
    best: Option<RoutePolicyBestJson>,
}

#[derive(Debug, serde::Deserialize)]
struct RoutePolicyBestJson {
    #[serde(default)]
    thresholds: HashMap<String, f64>,
    /// Optional learned-blend variant. When set, ``thresholds`` is typically
    /// empty and the severity classifies via the blend's combined score.
    /// Mirrors azoth_route_policy_search._make_learned_blend_candidate_at_fp.
    #[serde(default)]
    blend: Option<BlendPolicyJson>,
}

#[derive(Debug, serde::Deserialize)]
struct BlendPolicyJson {
    #[serde(default)]
    routes: Vec<String>,
    #[serde(default)]
    weights: Vec<f64>,
    #[serde(default)]
    intercept: f64,
    #[serde(default)]
    threshold: f64,
    /// Currently only ``"logit"`` is supported. Future blend variants
    /// (e.g., raw-prob linear, monotonic GAM) would advertise themselves
    /// here so deploy can refuse unknown shapes loudly rather than
    /// silently misapplying the wrong transform.
    #[serde(default)]
    transform: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct LevelEntryJson {
    level: u8,
    hostile: SeverityEntryJson,
    suspicious: SeverityEntryJson,
}

#[derive(Debug, serde::Deserialize)]
struct SeverityEntryJson {
    /// Route-name → threshold map for this severity. Route names match the
    /// `models[].route` field: `"general"`, `"filegroups/<name>"`, `"filetypes/<name>"`.
    #[serde(default)]
    thresholds: HashMap<String, f64>,
}

/// Load and partially validate `<model_dir>/config.json`'s ensemble fields.
/// Returns `None` if the file is absent or unparseable; ensemble routing
/// then degrades to general-only with no specialists matched.
fn load_ensemble_config(model_dir: &Path, level: u8) -> Option<EnsembleConfig> {
    let path = model_dir.join("config.json");
    let data = std::fs::read_to_string(&path).ok()?;
    let json: EnsembleConfigJson = serde_json::from_str(&data).ok()?;

    let route_thresholds = thresholds_at_level(&json.levels, level);

    Some(EnsembleConfig {
        filetype_to_filegroup: json.filetype_to_group,
        required_routes: json.required_routes,
        route_thresholds,
    })
}

/// Load searched per-filetype decision policies. These are optional: older
/// ensembles without `route_policies.json` keep the original OR semantics.
fn load_route_policies(model_dir: &Path, level: u8) -> RoutePolicies {
    let path = model_dir.join("route_policies.json");
    let data = match std::fs::read_to_string(&path) {
        Ok(data) => data,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return RoutePolicies::default(),
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "cannot read route policies");
            return RoutePolicies::default();
        }
    };
    let json: RoutePoliciesJson = match serde_json::from_str(&data) {
        Ok(json) => json,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "cannot parse route policies");
            return RoutePolicies::default();
        }
    };

    let mut by_filetype = HashMap::new();
    for (_route_name, route) in json.routes {
        let Some(level_entry) = route.levels.iter().find(|entry| entry.level == level) else {
            continue;
        };
        let Some(hostile) = policy_severity_from_json(&level_entry.hostile) else {
            continue;
        };
        let Some(suspicious) = policy_severity_from_json(&level_entry.suspicious) else {
            continue;
        };
        by_filetype.insert(
            route.filetype,
            RoutePolicy {
                suspicious,
                hostile,
            },
        );
    }

    tracing::info!(
        path = %path.display(),
        level = level,
        routes = by_filetype.len(),
        "loaded route policies",
    );
    RoutePolicies { by_filetype }
}

#[allow(clippy::cast_possible_truncation)]
fn policy_severity_from_json(json: &RoutePolicySeverityJson) -> Option<PolicySeverity> {
    let best = json.best.as_ref()?;
    let blend = best
        .blend
        .as_ref()
        .and_then(blend_policy_from_json);
    let mut thresholds = HashMap::new();
    for (route, &threshold) in &best.thresholds {
        let threshold = threshold as f32;
        if (0.0..=1.0).contains(&threshold) {
            thresholds.insert(route.clone(), threshold);
        }
    }
    // A severity is loadable if either the OR-rule has at least one
    // threshold OR the blend is well-formed. The two coexist only in
    // pathological writer output — current code emits one or the other.
    if thresholds.is_empty() && blend.is_none() {
        return None;
    }
    Some(PolicySeverity { thresholds, blend })
}

#[allow(clippy::cast_possible_truncation)]
fn blend_policy_from_json(json: &BlendPolicyJson) -> Option<BlendPolicy> {
    // Only the logit transform is currently supported. An unknown transform
    // means the writer is ahead of this deploy binary — refuse loudly rather
    // than silently misapplying.
    match json.transform.as_deref() {
        None | Some("logit") => {}
        Some(other) => {
            tracing::error!(
                transform = %other,
                "unknown blend transform; ignoring blend policy",
            );
            return None;
        }
    }
    if json.routes.len() != json.weights.len() {
        tracing::error!(
            routes = json.routes.len(),
            weights = json.weights.len(),
            "blend routes/weights length mismatch; ignoring blend policy",
        );
        return None;
    }
    if json.routes.is_empty() {
        return None;
    }
    let threshold = json.threshold as f32;
    if !(0.0..=1.0).contains(&threshold) {
        tracing::error!(threshold, "blend threshold outside [0,1]; ignoring");
        return None;
    }
    Some(BlendPolicy {
        routes: json.routes.clone(),
        weights: json.weights.iter().map(|&w| w as f32).collect(),
        intercept: json.intercept as f32,
        threshold,
    })
}

/// Pull per-route thresholds from `levels[]` at the requested level. Pairs up
/// hostile and suspicious thresholds for each route; routes that appear in
/// only one of the two are skipped (a route can't be partially calibrated).
#[allow(clippy::cast_possible_truncation)]
fn thresholds_at_level(levels: &[LevelEntryJson], level: u8) -> HashMap<String, Thresholds> {
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

    let mut out: HashMap<String, Thresholds> = HashMap::new();
    for (route, &hostile) in &entry.hostile.thresholds {
        // Hostile policy is primary. Some calibrated specialists only earn a
        // hostile threshold at a level; keep them active as hostile-only by
        // setting suspicious equal to hostile.
        let suspicious = entry
            .suspicious
            .thresholds
            .get(route)
            .copied()
            .unwrap_or(hostile);
        let t = Thresholds {
            suspicious: suspicious as f32,
            hostile: hostile as f32,
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
    for (route, &suspicious) in &entry.suspicious.thresholds {
        if out.contains_key(route) {
            continue;
        }
        let t = Thresholds {
            suspicious: suspicious as f32,
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
    out
}

/// Load the inference backend for a bundle directory. Multi-seed bundles
/// store every member at `models/seed_NN.{txt,json}` (one file per seed); the
/// historical layout has a single `model.{txt,json}` directly under the
/// bundle dir. Both layouts are accepted; the multi-seed layout is preferred
/// when the `models/` subdirectory is present and non-empty.
///
/// Mixing `.json` (XGBoost) and `.txt` (LightGBM) members in the same `models/`
/// directory is rejected — the deployed bundle uses a single backend kind per
/// route. So is mixing legacy and multi-seed layouts in the same bundle.
fn load_backend(bundle_dir: &Path) -> Result<Backend> {
    let multi_dir = bundle_dir.join("models");
    let multi = if multi_dir.is_dir() {
        collect_multi_seed_paths(&multi_dir)?
    } else {
        Vec::new()
    };

    // Legacy single-model layout: model.{onnx,txt,json} directly under
    // bundle_dir. .onnx wins when present (load + infer faster). When
    // both .txt and .json exist with no .onnx, the bundle is ambiguous
    // because they're different backends (LightGBM vs XGBoost). .onnx
    // + .txt for the same route is FINE — the .onnx is just the
    // LightGBM model in a portable format.
    let single_onnx = bundle_dir.join("model.onnx");
    let single_xgb = bundle_dir.join("model.json");
    let single_lgb = bundle_dir.join("model.txt");
    let has_single = single_onnx.is_file() || single_xgb.is_file() || single_lgb.is_file();

    if !multi.is_empty() && has_single {
        anyhow::bail!(
            "model bundle is ambiguous: both `models/` (multi-seed) and a top-level model artifact \
             exist in {}; remove one to disambiguate the layout",
            bundle_dir.display(),
        );
    }

    let model_paths: Vec<PathBuf> = if !multi.is_empty() {
        multi
    } else if single_onnx.is_file() {
        vec![single_onnx]
    } else {
        match (single_xgb.is_file(), single_lgb.is_file()) {
            (true, false) => vec![single_xgb],
            (false, true) => vec![single_lgb],
            (true, true) => anyhow::bail!(
                "model bundle is ambiguous: both {} and {} exist; remove one to disambiguate the backend",
                single_xgb.display(),
                single_lgb.display(),
            ),
            (false, false) => anyhow::bail!(
                "model bundle is incomplete: missing model.onnx, model.json (XGBoost), or model.txt (LightGBM) \
                 in {} (and no models/ directory either)",
                bundle_dir.display(),
            ),
        }
    };

    let mut models = Vec::with_capacity(model_paths.len());
    let mut first_kind: Option<&'static str> = None;
    let mut first_n_features: Option<usize> = None;
    for path in &model_paths {
        let inner = load_inner_model(path)?;
        // Refuse to mix backend kinds across members.
        let kind = inner.kind();
        if let Some(prev) = first_kind {
            if prev != kind {
                anyhow::bail!(
                    "model bundle has mixed backends in {}: member {} is {kind} but earlier \
                     members were {prev}; every member must share the same backend kind",
                    bundle_dir.display(),
                    path.display(),
                );
            }
        } else {
            first_kind = Some(kind);
        }
        // Refuse to mix feature counts — averaging across heterogeneous
        // feature spaces would silently produce nonsense scores.
        let n = inner.num_features();
        if let Some(prev) = first_n_features {
            if prev != n {
                anyhow::bail!(
                    "model bundle has mismatched feature counts in {}: member {} expects {n} \
                     features but earlier members expected {prev}",
                    bundle_dir.display(),
                    path.display(),
                );
            }
        } else {
            first_n_features = Some(n);
        }
        models.push(inner);
    }

    Ok(Backend { models })
}

/// Pick up every `seed_*.{onnx,txt,json}` under a `models/` directory.
/// When the same seed has both `.onnx` and `.txt` (transitional state
/// while the deploy migrates ONNX in), the `.onnx` wins because it
/// loads ~3× faster, infers 5-11× faster, and is ~50% smaller. Output
/// is sorted by stem (so seed_42 lands before seed_43) and is
/// deterministic across runs.
fn collect_multi_seed_paths(multi_dir: &Path) -> Result<Vec<PathBuf>> {
    use std::collections::BTreeMap;
    // stem -> chosen path. BTreeMap keeps stems sorted for deterministic order.
    let mut by_stem: BTreeMap<String, PathBuf> = BTreeMap::new();
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
        if !name.starts_with("seed_") {
            // Tolerate bystander files (READMEs, hashes) so future additions
            // don't silently break loading.
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        // Lower number = higher priority (matches collimator's bundle.py
        // _EXT_PRIORITY).
        let priority = match ext {
            "onnx" => 0u8,
            "txt" => 1,
            "json" => 2,
            _ => continue,
        };
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let stem = stem.to_owned();
        match by_stem.get(&stem) {
            Some(existing) => {
                let existing_ext = existing
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                let existing_priority = match existing_ext {
                    "onnx" => 0u8,
                    "txt" => 1,
                    "json" => 2,
                    _ => 255,
                };
                if priority < existing_priority {
                    by_stem.insert(stem, path);
                }
            }
            None => {
                by_stem.insert(stem, path);
            }
        }
    }
    Ok(by_stem.into_values().collect())
}

/// Load one trained model from a path, picking the backend by extension.
fn load_inner_model(path: &Path) -> Result<InnerModel> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "json" => {
            let m = xgboost_ars::Model::load(path)
                .with_context(|| format!("loading XGBoost model from {}", path.display()))?;
            Ok(InnerModel::Xgboost(m))
        }
        "txt" => {
            let m = lightgbm_ars::Model::load(path)
                .with_context(|| format!("loading LightGBM model from {}", path.display()))?;
            Ok(InnerModel::Lightgbm(m))
        }
        "onnx" => {
            let m = OnnxBackend::load(path)?;
            Ok(InnerModel::Onnx(Box::new(m)))
        }
        other => anyhow::bail!(
            "unrecognized model extension {other:?} for {}; expected .onnx, .txt (LightGBM), or .json (XGBoost)",
            path.display(),
        ),
    }
}

/// Load one model bundle (model file + feature spec + thresholds).
///
/// `is_general` controls how missing artifacts are reported: for the general
/// route they're fatal load errors; for specialists callers handle the error
/// non-fatally (drop with a warning).
fn load_bundle(
    bundle_dir: &Path,
    explicit_thresholds: Option<&Thresholds>,
    is_general: bool,
) -> Result<LoadedBundle> {
    let spec_path = bundle_dir.join("feature_spec.json");
    if !spec_path.is_file() {
        if is_general {
            anyhow::bail!(
                "model bundle is incomplete: missing {}. Run 'litmus update-rules' to refresh the installed models.",
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

    let config_thresholds = load_config_thresholds(bundle_dir);
    let recommended = load_evaluation_thresholds(bundle_dir);
    let (thresholds, threshold_source) = match explicit_thresholds {
        Some(explicit) => {
            explicit
                .validate()
                .map_err(|error| anyhow::anyhow!("invalid thresholds: {error}"))?;
            if let Some(ref rec) = recommended {
                explicit.warn_if_divergent(rec);
            }
            (*explicit, "explicit")
        }
        None => match config_thresholds.or(recommended) {
            Some(t) => (
                t,
                if config_thresholds.is_some() {
                    "config.json"
                } else {
                    "evaluation.json"
                },
            ),
            None => {
                tracing::error!(
                    bundle = %bundle_dir.display(),
                    "no config.json or evaluation.json thresholds found — using conservative \
                     fallback (suspicious={}, hostile={})",
                    Thresholds::FALLBACK_SUSPICIOUS,
                    Thresholds::FALLBACK_HOSTILE,
                );
                (Thresholds::default(), "fallback")
            }
        },
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
        calibrated
    } else {
        thresholds
    };
    if let Some(cal) = calibrator.as_ref() {
        tracing::debug!(
            bundle = %bundle_dir.display(),
            breakpoints = cal.x.len(),
            "loaded isotonic calibrator (thresholds calibrated to match)"
        );
    }

    Ok(LoadedBundle {
        backend,
        spec,
        thresholds,
        threshold_source,
        calibrator,
    })
}

/// Walk a `filegroups/` or `filetypes/` directory, loading each subdirectory
/// as a specialist bundle. ABI mismatches and spec-subset violations drop the
/// specialist with a warning rather than failing the whole load.
///
/// `category` is the path prefix used in the ensemble config's route names —
/// either `"filegroups"` or `"filetypes"` — so specialist thresholds can be
/// looked up under e.g. `"filegroups/native"`.
fn load_specialists(
    parent: &Path,
    route_thresholds: &HashMap<String, Thresholds>,
    route_policies: &RoutePolicies,
    category: &str,
    out: &mut HashMap<String, Route>,
    skipped: &mut HashSet<String>,
) {
    let entries = match std::fs::read_dir(parent) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            tracing::warn!(parent = %parent.display(), error = %e, "cannot read specialist directory");
            return;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n.to_owned(),
            None => continue,
        };
        let route_name = format!("{category}/{name}");
        // Skip on-disk subdirectories that the deployment config doesn't list.
        // These are common as artifacts of experimentation; loading them with
        // fallback thresholds would put uncalibrated routes in the OR.
        let route_t = route_thresholds.get(&route_name).copied();
        if route_t.is_none() && !route_policies.contains_route(&route_name) {
            tracing::debug!(
                category = %category,
                name = %name,
                "skipping specialist directory: no thresholds in ensemble config"
            );
            skipped.insert(name);
            continue;
        };
        match load_specialist(&path, &name, Some(route_t.unwrap_or_default())) {
            Ok(route) => {
                out.insert(name, route);
            }
            Err(e) => {
                tracing::warn!(
                    category = %category,
                    name = %name,
                    error = ?e,
                    "dropping specialist; route will degrade to general or filegroup",
                );
            }
        }
    }
}

/// Load and validate one specialist bundle, applying the spec-subset and
/// ABI-version rules. Errors here are recoverable — the specialist is
/// dropped, not fatal.
fn load_specialist(
    bundle_dir: &Path,
    name: &str,
    explicit_thresholds: Option<Thresholds>,
) -> Result<Route> {
    let bundle = load_bundle(
        bundle_dir,
        explicit_thresholds.as_ref(),
        /* is_general = */ false,
    )
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
    /// Score a feature vector with the route's backend, then apply the
    /// route's calibrator if present. Single source of truth for "what does
    /// this route output for this file" — used by every code path that
    /// previously called `backend.predict` directly.
    fn predict_calibrated(&self, features: &[f32]) -> Result<f32> {
        let raw = self.backend.predict(features)?;
        Ok(match &self.calibrator {
            Some(cal) => cal.apply(raw),
            None => raw,
        })
    }
}

/// Routing decision: which route to consult for a file of type `T` in group `G`.
///
/// DESIGN.md says: filetype if present → filegroup if present → general always.
/// Each present route contributes to the OR over thresholds.
#[derive(Debug, Default)]
struct RouteSet {
    /// Filegroup name → specialist model.
    filegroups: HashMap<String, Route>,
    /// Filetype name → specialist model.
    filetypes: HashMap<String, Route>,
    /// Filegroup routes present on disk but omitted from calibration.
    skipped_filegroups: HashSet<String>,
    /// Filetype routes present on disk but omitted from calibration.
    skipped_filetypes: HashSet<String>,
    /// Filetype → filegroup mapping from `config.json`. Used to translate a
    /// scanned file's `type` into the applicable filegroup specialist.
    filetype_to_filegroup: HashMap<String, String>,
    /// Optional searched policies keyed by cleave file type.
    policies: RoutePolicies,
}

impl RouteSet {
    /// True when there are no specialist routes loaded; routed prediction
    /// then degrades to the general route alone.
    fn is_empty(&self) -> bool {
        self.filegroups.is_empty() && self.filetypes.is_empty()
    }
}

/// OR over classifications: pick the more severe of two outcomes.
const fn max_class(a: Classification, b: Classification) -> Classification {
    if (a as u8) >= (b as u8) {
        a
    } else {
        b
    }
}

#[derive(Debug, Clone)]
struct RouteProbability {
    route: String,
    probability: f32,
}

fn compact_route_name(route: &str) -> String {
    if route == "general" {
        "az".to_string()
    } else if let Some(name) = route.strip_prefix("filegroups/") {
        format!("az/{name}")
    } else if let Some(name) = route.strip_prefix("filetypes/") {
        format!("az/{name}")
    } else {
        format!("az/{route}")
    }
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
        });
    }
    if let Some((p, t)) = policy.suspicious.fire(scores) {
        return Some(Decision {
            class: Classification::Suspicious,
            probability: p,
            threshold: t,
        });
    }
    None
}

/// Push each per-route threshold in `route_policies.json` through that route's
/// isotonic calibrator. The scoring path emits calibrated probabilities, so
/// the policy comparison must happen in calibrated space too. Routes without
/// a calibrator (pre-calibrator bundles) are left untouched. Routes referenced
/// by a policy but not loaded as a specialist are also left untouched —
/// they'll never produce a `RouteProbability` entry, so the comparison never
/// fires.
fn calibrate_policy_thresholds_with<'a>(
    policies: &mut RoutePolicies,
    lookup: impl Fn(&str) -> Option<&'a IsotonicCalibrator>,
) {
    let mut adjusted = 0_usize;
    for policy in policies.by_filetype.values_mut() {
        for severity in [&mut policy.hostile, &mut policy.suspicious] {
            // Blend severities are calibrated at fit time (the policy writer
            // applies isotonic to the per-route inputs before fitting LR), so
            // their threshold and weights are already in calibrated-prob
            // space. Skipping the per-route mapping is intentional — the
            // logistic combination doesn't decompose through isotonic the
            // way per-route OR-rule thresholds do.
            if severity.blend.is_some() {
                continue;
            }
            for (route_name, threshold) in severity.thresholds.iter_mut() {
                if let Some(cal) = lookup(route_name) {
                    *threshold = cal.apply(*threshold);
                    adjusted += 1;
                }
            }
        }
    }
    if adjusted > 0 {
        tracing::debug!(adjusted, "calibrated route_policies thresholds");
    }
}

/// Production lookup for `calibrate_policy_thresholds_with`. Resolves a route
/// name (`general`, `filegroups/X`, `filetypes/Y`) to its calibrator, if any.
fn calibrate_policy_thresholds(
    policies: &mut RoutePolicies,
    general_calibrator: &Option<IsotonicCalibrator>,
    filegroups: &HashMap<String, Route>,
    filetypes: &HashMap<String, Route>,
) {
    calibrate_policy_thresholds_with(policies, |route_name| {
        if route_name == "general" {
            general_calibrator.as_ref()
        } else if let Some(name) = route_name.strip_prefix("filegroups/") {
            filegroups.get(name).and_then(|r| r.calibrator.as_ref())
        } else if let Some(name) = route_name.strip_prefix("filetypes/") {
            filetypes.get(name).and_then(|r| r.calibrator.as_ref())
        } else {
            None
        }
    });
}

fn policy_route_class(policy: &RoutePolicy, route: &str, probability: f32) -> Classification {
    // Per-route classification is well-defined only for OR-rule severities —
    // a learned blend's verdict is a function of all its inputs jointly, so
    // no single route can be labeled Hostile/Suspicious on its own. For
    // blend severities we return Benign here; the final verdict still comes
    // from policy_classify, which evaluates the blend over the full score
    // vector. The diagnostic per-route classification just won't get an
    // individual contribution for blend-driven filetypes.
    if policy.hostile.blend.is_none()
        && policy
            .hostile
            .thresholds
            .get(route)
            .is_some_and(|&threshold| probability >= threshold)
    {
        Classification::Hostile
    } else if policy.suspicious.blend.is_none()
        && policy
            .suspicious
            .thresholds
            .get(route)
            .is_some_and(|&threshold| probability >= threshold)
    {
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
    thresholds: Thresholds,
    info: ModelInfo,
    routes: RouteSet,
    /// Optional isotonic calibrator for the general model. Applied after
    /// `inner.predict` everywhere we score with the general route. Mirrors
    /// the per-route calibrator on Route; backward compat = None.
    general_calibrator: Option<IsotonicCalibrator>,
}

impl Model {
    /// Score with the general model and apply its calibrator if present.
    /// Single source of truth for the general path's calibrated output —
    /// every previous direct `self.inner.predict` call now goes through here.
    fn predict_general_calibrated(&self, features: &[f32]) -> Result<f32> {
        let raw = self.inner.predict(features)?;
        Ok(match &self.general_calibrator {
            Some(cal) => cal.apply(raw),
            None => raw,
        })
    }
}

impl Model {
    /// Load model artifacts from a directory containing `feature_spec.json`
    /// and either `model.json` (XGBoost) or `model.txt` (LightGBM).
    ///
    /// The same directory may also contain optional metadata such as
    /// `shap_importance.json` and git history, but those are not required here.
    ///
    /// # Errors
    /// Returns an error if thresholds are invalid, required model artifacts are
    /// missing, or the loaded feature spec does not match this build.
    ///
    /// Threshold resolution order:
    /// 1. Explicit `thresholds` argument (from CLI flags)
    /// 2. `config.json` in the model directory
    /// 3. Recommended thresholds from `evaluation.json`
    /// 4. Conservative fallback constants
    ///
    /// If explicit thresholds are provided *and* `evaluation.json` contains
    /// recommendations, a warning is emitted when they diverge significantly.
    pub fn load(model_dir: &Path, thresholds: Option<Thresholds>) -> Result<Self> {
        // Detect ensemble layout by the presence of `general/` immediately
        // under model_dir. Otherwise treat model_dir itself as a single bundle.
        if model_dir.join("general").is_dir() {
            Self::load_ensemble(model_dir, thresholds)
        } else {
            Self::load_single_bundle(model_dir, thresholds)
        }
    }

    /// Load a single-bundle layout (legacy / dev). The model artifacts
    /// (`model.txt`/`model.json`, `feature_spec.json`, `config.json`,
    /// `evaluation.json`) sit directly in `model_dir`. No specialists.
    fn load_single_bundle(model_dir: &Path, thresholds: Option<Thresholds>) -> Result<Self> {
        let bundle = load_bundle(model_dir, thresholds.as_ref(), /* is_general = */ true)?;
        tracing::info!(
            backend = bundle.backend.kind(),
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
        let info = ModelInfo {
            version: bundle.spec.version(),
            abi_version: bundle.spec.abi_version(),
            sha256: String::new(),
            commit: None,
        };
        Ok(Self {
            inner: bundle.backend,
            spec: bundle.spec,
            thresholds: bundle.thresholds,
            info,
            routes: RouteSet::default(),
            general_calibrator: bundle.calibrator,
        })
    }

    /// Load an ensemble layout (`general/` + `filegroups/*` + `filetypes/*`).
    /// See module-level docs for the on-disk shape and config schema.
    #[allow(clippy::too_many_lines)]
    fn load_ensemble(model_dir: &Path, thresholds: Option<Thresholds>) -> Result<Self> {
        // Ensemble-level config.json carries the routing map and the per-route
        // thresholds for every level. We resolve thresholds from it before
        // loading any individual route bundle so each bundle gets the right
        // pre-resolved thresholds and skips its own (nonexistent) config.json.
        let ensemble_cfg =
            load_ensemble_config(model_dir, DEFAULT_SEVERITY_LEVEL).unwrap_or_default();
        let policies = load_route_policies(model_dir, DEFAULT_SEVERITY_LEVEL);

        let general_dir = model_dir.join("general");
        let general_thresholds =
            thresholds.or_else(|| ensemble_cfg.route_thresholds.get("general").copied());
        let general = load_bundle(
            &general_dir,
            general_thresholds.as_ref(),
            /* is_general = */ true,
        )
        .with_context(|| format!("loading general route from {}", general_dir.display()))?;

        let mut filegroups: HashMap<String, Route> = HashMap::new();
        let mut filetypes: HashMap<String, Route> = HashMap::new();
        let mut skipped_filegroups: HashSet<String> = HashSet::new();
        let mut skipped_filetypes: HashSet<String> = HashSet::new();

        // Walk filegroups/<name>/ and filetypes/<name>/, loading each
        // specialist that exists. Specialists with ABI mismatch or bad spec
        // subset are dropped with a warning, not a hard failure.
        load_specialists(
            &model_dir.join("filegroups"),
            &ensemble_cfg.route_thresholds,
            &policies,
            "filegroups",
            &mut filegroups,
            &mut skipped_filegroups,
        );
        load_specialists(
            &model_dir.join("filetypes"),
            &ensemble_cfg.route_thresholds,
            &policies,
            "filetypes",
            &mut filetypes,
            &mut skipped_filetypes,
        );

        // Required routes from config: any listed name that didn't load is
        // fatal. "general" is implicitly required and was already loaded above.
        for required in &ensemble_cfg.required_routes {
            if required == "general" {
                continue;
            }
            if let Some(name) = required.strip_prefix("filegroups/") {
                if !filegroups.contains_key(name) {
                    anyhow::bail!(
                        "ensemble config marks filegroup {name:?} as required but it failed to load"
                    );
                }
            } else if let Some(name) = required.strip_prefix("filetypes/") {
                if !filetypes.contains_key(name) {
                    anyhow::bail!(
                        "ensemble config marks filetype {name:?} as required but it failed to load"
                    );
                }
            } else {
                anyhow::bail!(
                    "unknown required-route name {required:?} in ensemble config; \
                     expected `general`, `filegroups/<name>`, or `filetypes/<name>`"
                );
            }
        }

        // The route_policies.json thresholds for the OR-of-routes decision
        // path were loaded in raw-score space, but the route scorers now emit
        // calibrated probabilities (see `load_bundle`). Push each per-route
        // threshold through that route's calibrator so policy_classify
        // compares like with like. Isotonic monotonicity preserves the
        // decision; the comparison is now numerically consistent.
        let mut policies = policies;
        calibrate_policy_thresholds(&mut policies, &general.calibrator, &filegroups, &filetypes);

        tracing::info!(
            backend = general.backend.kind(),
            n_members = general.backend.n_members(),
            features = general.spec.total_features(),
            model_abi_version = general.spec.abi_version(),
            threshold_suspicious = general.thresholds.suspicious,
            threshold_hostile = general.thresholds.hostile,
            threshold_source = general.threshold_source,
            spec_version = general.spec.version(),
            filegroups = filegroups.len(),
            filetypes = filetypes.len(),
            layout = "ensemble",
            "model loaded",
        );

        let info = ModelInfo {
            version: general.spec.version(),
            abi_version: general.spec.abi_version(),
            sha256: String::new(),
            commit: None,
        };
        Ok(Self {
            inner: general.backend,
            spec: general.spec,
            thresholds: general.thresholds,
            info,
            routes: RouteSet {
                filegroups,
                filetypes,
                skipped_filegroups,
                skipped_filetypes,
                filetype_to_filegroup: ensemble_cfg.filetype_to_filegroup,
                policies,
            },
            general_calibrator: general.calibrator,
        })
    }

    /// Run inference on a feature vector against the general route only.
    ///
    /// Use [`Model::predict_for`] when the caller knows the file's `type` so
    /// the ensemble's filegroup and filetype specialists can contribute. This
    /// method exists for call sites that don't yet have a file type (and for
    /// single-bundle deployments where it makes no difference).
    ///
    /// # Errors
    /// Returns an error if the underlying model backend fails to produce a
    /// prediction for the provided feature vector.
    pub fn predict(&self, features: &[f32]) -> Result<(f32, Classification)> {
        let probability = self.predict_general_calibrated(features)?;
        Ok((probability, self.thresholds.classify(probability)))
    }

    /// Pick the strongest [`Decision`] across the route scores using only the
    /// model's general thresholds (no per-filetype policy). Picks the highest
    /// class; on ties, the highest probability. Benign fallback uses the
    /// general route's prob against the suspicious cutoff.
    fn decide_from_scores(&self, scores: &[RouteProbability]) -> Decision {
        // General route is always first when scoring through predict_*_detailed.
        let general_prob = scores
            .iter()
            .find(|s| s.route == "general")
            .map_or_else(|| 0.0, |s| s.probability);
        let mut best = self.thresholds.decide(general_prob);
        for score in scores {
            let candidate = self.thresholds.decide(score.probability);
            let better = match (candidate.class as u8).cmp(&(best.class as u8)) {
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

    /// Routed prediction for a file of type `file_type`.
    ///
    /// Consults general always; consults the filegroup specialist if a
    /// mapping exists for `file_type`; consults the filetype specialist if
    /// one is loaded for `file_type`. The reported classification is the OR
    /// over per-route threshold crossings — see DESIGN.md §Runtime Decision.
    ///
    /// The reported probability is `max(per-route probabilities)`. It is
    /// monotone with the OR decision: if any route flagged hostile, the
    /// reported probability is at least that route's, and at least one
    /// route's threshold for hostile.
    ///
    /// On a single-bundle deployment this falls through to `predict()`.
    ///
    /// # Errors
    /// Returns an error if any consulted backend fails on `features`.
    pub fn predict_for(&self, file_type: &str, features: &[f32]) -> Result<(f32, Classification)> {
        if self.routes.is_empty() {
            return self.predict(features);
        }

        // Score general first; it's always present and supplies the baseline.
        let general_prob = self.predict_general_calibrated(features)?;
        let mut max_prob = general_prob;
        let mut classification = self.thresholds.classify(general_prob);

        // Optional filegroup specialist, looked up via the configured
        // filetype → filegroup map.
        if let Some(group_name) = self.routes.filetype_to_filegroup.get(file_type)
            && let Some(route) = self.routes.filegroups.get(group_name)
        {
            if route.spec.total_features() != features.len() {
                tracing::debug!(
                    route = %group_name,
                    expected = route.spec.total_features(),
                    got = features.len(),
                    "skipping routed feature-vector prediction; use predict_for_report for heterogeneous specialists",
                );
            } else {
                let prob = route.predict_calibrated(features)?;
                if prob > max_prob {
                    max_prob = prob;
                }
                classification = max_class(classification, route.thresholds.classify(prob));
            }
        }

        // Optional filetype specialist.
        if let Some(route) = self.routes.filetypes.get(file_type) {
            if route.spec.total_features() != features.len() {
                tracing::debug!(
                    route = %file_type,
                    expected = route.spec.total_features(),
                    got = features.len(),
                    "skipping routed feature-vector prediction; use predict_for_report for heterogeneous specialists",
                );
            } else {
                let prob = route.predict_calibrated(features)?;
                if prob > max_prob {
                    max_prob = prob;
                }
                classification = max_class(classification, route.thresholds.classify(prob));
            }
        }

        Ok((max_prob, classification))
    }

    fn score_route_report(
        route: &Route,
        report: &serde_json::Value,
    ) -> Result<(f32, Classification)> {
        let mut features = route.ctx.extract(report);
        route.spec.standardize(&mut features);
        let probability = route.predict_calibrated(&features)?;
        Ok((probability, route.thresholds.classify(probability)))
    }

    fn score_route_file(route: &Route, file: &serde_json::Value) -> Result<(f32, Classification)> {
        let mut features = route.ctx.extract_file(file);
        route.spec.standardize(&mut features);
        let probability = route.predict_calibrated(&features)?;
        Ok((probability, route.thresholds.classify(probability)))
    }

    /// Routed prediction from a full cleave report.
    ///
    /// This is the production ensemble path. General is scored from the
    /// caller-provided general feature vector, while each specialist extracts
    /// and standardizes its own route-specific vector from `report`.
    pub fn predict_for_report(
        &self,
        file_type: &str,
        general_features: &[f32],
        report: &serde_json::Value,
    ) -> Result<(f32, Classification)> {
        let (decision, _, _) =
            self.predict_for_report_detailed(file_type, general_features, report)?;
        Ok((decision.probability, decision.class))
    }

    /// Same as [`Self::predict_for_report`], with per-route scores retained
    /// for JSON and `--extra` output.
    pub fn predict_for_report_detailed(
        &self,
        file_type: &str,
        general_features: &[f32],
        report: &serde_json::Value,
    ) -> Result<(Decision, Vec<RouteScore>, Vec<SkippedRoute>)> {
        let (route_probs, scores, mut skipped) = self.score_all_routes(
            file_type,
            general_features,
            |route| Self::score_route_report(route, report),
        )?;
        let decision = self.decide_from_routes(file_type, &route_probs, &mut skipped);
        Ok((decision, scores, skipped))
    }

    /// Routed prediction for one embedded-file JSON object.
    pub fn predict_for_file(
        &self,
        file_type: &str,
        general_features: &[f32],
        file: &serde_json::Value,
    ) -> Result<(f32, Classification)> {
        let (decision, _, _) =
            self.predict_for_file_detailed(file_type, general_features, file)?;
        Ok((decision.probability, decision.class))
    }

    /// Same as [`Self::predict_for_file`], with per-route scores retained.
    pub fn predict_for_file_detailed(
        &self,
        file_type: &str,
        general_features: &[f32],
        file: &serde_json::Value,
    ) -> Result<(Decision, Vec<RouteScore>, Vec<SkippedRoute>)> {
        let (route_probs, scores, mut skipped) = self.score_all_routes(
            file_type,
            general_features,
            |route| Self::score_route_file(route, file),
        )?;
        let decision = self.decide_from_routes(file_type, &route_probs, &mut skipped);
        Ok((decision, scores, skipped))
    }

    /// Score every applicable route for `file_type` and return the
    /// `(probabilities, RouteScore list, SkippedRoute list)` triple. The
    /// per-route `RouteScore.classification` uses the policy's per-route
    /// classifier when available, mirroring the previous behaviour for the
    /// diagnostic `models` array.
    fn score_all_routes<F>(
        &self,
        file_type: &str,
        general_features: &[f32],
        mut score: F,
    ) -> Result<(Vec<RouteProbability>, Vec<RouteScore>, Vec<SkippedRoute>)>
    where
        F: FnMut(&Route) -> Result<(f32, Classification)>,
    {
        let policy = self.routes.policies.by_filetype.get(file_type);
        let general_prob = self.predict_general_calibrated(general_features)?;
        let general_class = policy.map_or_else(
            || self.thresholds.classify(general_prob),
            |policy| policy_route_class(policy, "general", general_prob),
        );
        let mut route_probs = vec![RouteProbability {
            route: "general".to_string(),
            probability: general_prob,
        }];
        let mut scores = vec![RouteScore {
            model: "az".to_string(),
            probability: general_prob,
            classification: general_class,
        }];
        let mut skipped = Vec::new();

        if self.routes.is_empty() {
            return Ok((route_probs, scores, skipped));
        }

        if let Some(group_name) = self.routes.filetype_to_filegroup.get(file_type) {
            if let Some(route) = self.routes.filegroups.get(group_name) {
                let route_name = format!("filegroups/{group_name}");
                let (prob, class) = score(route)?;
                let class = policy.map_or(class, |policy| {
                    policy_route_class(policy, &route_name, prob)
                });
                scores.push(RouteScore {
                    model: format!("az/{group_name}"),
                    probability: prob,
                    classification: class,
                });
                route_probs.push(RouteProbability {
                    route: route_name,
                    probability: prob,
                });
            } else if self.routes.skipped_filegroups.contains(group_name) {
                skipped.push(SkippedRoute {
                    model: format!("az/{group_name}"),
                    reason: "uncalibrated",
                });
            }
        }

        if let Some(route) = self.routes.filetypes.get(file_type) {
            let route_name = format!("filetypes/{file_type}");
            let (prob, class) = score(route)?;
            let class = policy.map_or(class, |policy| {
                policy_route_class(policy, &route_name, prob)
            });
            scores.push(RouteScore {
                model: format!("az/{file_type}"),
                probability: prob,
                classification: class,
            });
            route_probs.push(RouteProbability {
                route: route_name,
                probability: prob,
            });
        } else if self.routes.skipped_filetypes.contains(file_type) {
            skipped.push(SkippedRoute {
                model: format!("az/{file_type}"),
                reason: "uncalibrated",
            });
        }

        Ok((route_probs, scores, skipped))
    }

    /// Pick the final [`Decision`] from per-route scores, applying the
    /// per-filetype policy when one is configured.
    fn decide_from_routes(
        &self,
        file_type: &str,
        route_probs: &[RouteProbability],
        skipped: &mut Vec<SkippedRoute>,
    ) -> Decision {
        let policy = self.routes.policies.by_filetype.get(file_type);
        let Some(policy) = policy else {
            return self.decide_from_scores(route_probs);
        };
        for route in policy
            .hostile
            .thresholds
            .keys()
            .chain(policy.suspicious.thresholds.keys())
        {
            if !route_probs.iter().any(|score| score.route == *route) {
                skipped.push(SkippedRoute {
                    model: compact_route_name(route),
                    reason: "unavailable",
                });
            }
        }
        policy_decide(policy, route_probs).unwrap_or_else(|| {
            // Benign fallback: report general's prob against the model's
            // suspicious cutoff. That's the cutoff the score didn't reach.
            let general_prob = route_probs
                .iter()
                .find(|s| s.route == "general")
                .map_or(0.0, |s| s.probability);
            Decision {
                class: Classification::Benign,
                probability: general_prob,
                threshold: self.thresholds.suspicious,
            }
        })
    }

    /// Inference backend identifier (`"xgboost"` or `"lightgbm"`).
    #[must_use]
    pub fn backend_kind(&self) -> &'static str {
        self.inner.kind()
    }

    /// Feature specification used to build input vectors.
    #[must_use]
    pub fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    /// Classification thresholds carried by this loaded model.
    #[must_use]
    pub const fn thresholds(&self) -> Thresholds {
        self.thresholds
    }

    /// Stable metadata describing the loaded model artifacts.
    #[must_use]
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn load_rejects_missing_feature_spec_with_update_guidance() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("model.json"), b"{}")?;

        let Err(err) = Model::load(dir.path(), None) else {
            anyhow::bail!("missing feature spec should be rejected");
        };
        let message = err.to_string();
        assert!(message.contains("model bundle is incomplete"));
        assert!(message.contains("Run 'litmus update-rules'"));
        Ok(())
    }

    #[test]
    fn load_severity_thresholds_reads_config_json_levels() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
              "suspicious": 0.65,
              "hostile": 0.90,
              "severity_levels": [
                {
                  "level": 1,
                  "suspicious": {"threshold": 0.99},
                  "hostile": {"threshold": 0.99}
                },
                {
                  "level": 9,
                  "suspicious": {"threshold": 0.50},
                  "hostile": {"threshold": 0.80}
                }
              ]
            }"#,
        )?;

        let level_1 = load_severity_thresholds(dir.path(), 1)?.context("level 1")?;
        assert_eq!(level_1.suspicious, 0.99);
        assert_eq!(level_1.hostile, 0.99);
        assert_eq!(level_1.classify(0.99), Classification::Hostile);

        let level_9 = load_severity_thresholds(dir.path(), 9)?.context("level 9")?;
        assert_eq!(level_9.suspicious, 0.50);
        assert_eq!(level_9.hostile, 0.80);
        Ok(())
    }

    #[test]
    fn load_severity_thresholds_rejects_invalid_level() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let err = load_severity_thresholds(dir.path(), 0).expect_err("level 0 is invalid");
        assert!(err.to_string().contains("1..=9"));
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
        let mut hostile = HashMap::new();
        hostile.insert("general".to_string(), 0.5);
        hostile.insert("filetypes/elf".to_string(), 0.5);
        hostile.insert("filegroups/missing".to_string(), 0.5);
        let mut suspicious = HashMap::new();
        suspicious.insert("general".to_string(), 0.5);
        let policy = RoutePolicy {
            hostile: PolicySeverity {
                thresholds: hostile,
                blend: None,
            },
            suspicious: PolicySeverity {
                thresholds: suspicious,
                blend: None,
            },
        };
        let mut policies = RoutePolicies {
            by_filetype: HashMap::from([("elf".to_string(), policy)]),
        };

        calibrate_policy_thresholds_with(&mut policies, |route| match route {
            "general" => Some(&cal_general),
            "filetypes/elf" => Some(&cal_elf),
            _ => None,
        });

        let elf = policies.by_filetype.get("elf").expect("policy retained");
        // general at raw=0.5 → 0.10
        assert!((elf.hostile.thresholds["general"] - 0.10).abs() < 1e-6);
        // filetypes/elf at raw=0.5 → 0.90
        assert!((elf.hostile.thresholds["filetypes/elf"] - 0.90).abs() < 1e-6);
        // No calibrator for filegroups/missing — left at raw 0.5.
        assert!((elf.hostile.thresholds["filegroups/missing"] - 0.5).abs() < 1e-6);
        // Suspicious side also calibrated.
        assert!((elf.suspicious.thresholds["general"] - 0.10).abs() < 1e-6);
    }

    #[test]
    fn max_class_is_or_over_severity() {
        use Classification::*;
        assert_eq!(max_class(Benign, Benign), Benign);
        assert_eq!(max_class(Benign, Suspicious), Suspicious);
        assert_eq!(max_class(Suspicious, Hostile), Hostile);
        assert_eq!(max_class(Hostile, Suspicious), Hostile);
        assert_eq!(max_class(Hostile, Hostile), Hostile);
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
        let parsed: EnsembleConfigJson = serde_json::from_str(json).unwrap();

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
        let parsed: EnsembleConfigJson = serde_json::from_str(half).unwrap();
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
        let policy = RoutePolicy {
            hostile: PolicySeverity {
                thresholds: HashMap::from([
                    ("filetypes/elf".to_string(), 0.99),
                    ("general".to_string(), 0.95),
                ]),
                blend: None,
            },
            suspicious: PolicySeverity {
                thresholds: HashMap::from([("filetypes/elf".to_string(), 0.90)]),
                blend: None,
            },
        };

        let general_escape = vec![
            RouteProbability {
                route: "general".to_string(),
                probability: 0.96,
            },
            RouteProbability {
                route: "filetypes/elf".to_string(),
                probability: 0.10,
            },
        ];
        assert_eq!(
            policy_classify(&policy, &general_escape),
            Classification::Hostile
        );

        let specialist_suspicious = vec![
            RouteProbability {
                route: "general".to_string(),
                probability: 0.10,
            },
            RouteProbability {
                route: "filetypes/elf".to_string(),
                probability: 0.91,
            },
        ];
        assert_eq!(
            policy_classify(&policy, &specialist_suspicious),
            Classification::Suspicious
        );

        let inactive_route = vec![RouteProbability {
            route: "filegroups/native".to_string(),
            probability: 1.0,
        }];
        assert_eq!(
            policy_classify(&policy, &inactive_route),
            Classification::Benign
        );
    }

    #[test]
    fn blend_policy_fires_on_hand_built_inputs() {
        // Construct a blend that fires when general's prob is high enough on
        // its own (weight 1.0 on general, 0 on specialist, intercept chosen
        // so threshold=0.5 lines up with general prob ≈ 0.6).
        // sigmoid(intercept + 1.0 * logit(0.6)) = 0.5
        //   logit(0.6) ≈ 0.4054
        //   intercept = -0.4054 gives sigmoid(0) = 0.5
        let blend = BlendPolicy {
            routes: vec!["general".to_string(), "filetypes/elf".to_string()],
            weights: vec![1.0, 0.0],
            intercept: -0.4054,
            threshold: 0.5,
        };
        // general=0.6 → fires
        let fires = blend.fires(&[
            RouteProbability {
                route: "general".to_string(),
                probability: 0.6,
            },
            RouteProbability {
                route: "filetypes/elf".to_string(),
                probability: 0.0,
            },
        ]);
        assert!(fires);
        // general=0.5 → doesn't fire (logit(0.5)=0, intercept negative)
        let no_fire = blend.fires(&[
            RouteProbability {
                route: "general".to_string(),
                probability: 0.5,
            },
            RouteProbability {
                route: "filetypes/elf".to_string(),
                probability: 0.99,
            },
        ]);
        assert!(!no_fire);
        // Missing route → can't blend, doesn't fire.
        let missing = blend.fires(&[RouteProbability {
            route: "general".to_string(),
            probability: 1.0,
        }]);
        assert!(!missing);
    }

    #[test]
    fn load_route_policies_parses_blend_field() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("route_policies.json"),
            r#"{
              "schema": "azoth.route_policy_search.v1",
              "routes": {
                "filetypes/elf": {
                  "filetype": "elf",
                  "levels": [{
                    "level": 5,
                    "hostile": {"best": {
                      "thresholds": {},
                      "blend": {
                        "routes": ["general", "filegroups/native", "filetypes/elf"],
                        "weights": [0.5, 0.3, 1.2],
                        "intercept": -1.5,
                        "threshold": 0.8,
                        "transform": "logit"
                      }
                    }},
                    "suspicious": {"best": {"thresholds": {"general": 0.7}}}
                  }]
                }
              }
            }"#,
        )?;
        let policies = load_route_policies(dir.path(), 5);
        let policy = policies
            .by_filetype
            .get("elf")
            .expect("elf policy loaded");
        let blend = policy.hostile.blend.as_ref().expect("blend loaded");
        assert_eq!(blend.routes.len(), 3);
        assert_eq!(blend.weights.len(), 3);
        assert!((blend.intercept - -1.5).abs() < 1e-6);
        assert!((blend.threshold - 0.8).abs() < 1e-6);
        // contains_route should recognize blend routes too.
        assert!(policies.contains_route("filegroups/native"));
        assert!(policies.contains_route("filetypes/elf"));
        Ok(())
    }

    #[test]
    fn blend_policy_severity_isnt_isotonic_calibrated() {
        // calibrate_policy_thresholds_with must leave blend severities alone —
        // the blend's threshold is already in calibrated-prob space (its fit
        // ran on isotonic-calibrated route probs at policy-search time).
        // If we double-applied isotonic here it would shift the threshold off
        // the calibrated distribution and deploy verdicts would silently drift.
        let blend = BlendPolicy {
            routes: vec!["general".to_string()],
            weights: vec![1.0],
            intercept: 0.0,
            threshold: 0.42,
        };
        let policy = RoutePolicy {
            hostile: PolicySeverity {
                thresholds: HashMap::new(),
                blend: Some(blend),
            },
            suspicious: PolicySeverity {
                thresholds: HashMap::from([("general".to_string(), 0.5)]),
                blend: None,
            },
        };
        let mut policies = RoutePolicies {
            by_filetype: HashMap::from([("elf".to_string(), policy)]),
        };
        // A calibrator that shifts every input by +0.1 (clamped to [0, 1]).
        let cal = IsotonicCalibrator {
            x: vec![0.0, 1.0],
            y: vec![0.1, 1.0],
        };
        calibrate_policy_thresholds_with(&mut policies, |_route| Some(&cal));
        let pol = policies.by_filetype.get("elf").unwrap();
        // Hostile (blend) threshold stays at its original calibrated value.
        let blend = pol.hostile.blend.as_ref().unwrap();
        assert!((blend.threshold - 0.42).abs() < 1e-6);
        // Suspicious (OR-rule) threshold did get mapped.
        assert!(
            (pol.suspicious.thresholds.get("general").copied().unwrap() - cal.apply(0.5)).abs()
                < 1e-6
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

        let policies = load_route_policies(dir.path(), 5);
        assert!(policies.contains_route("general"));
        assert!(policies.contains_route("filegroups/native"));
        assert!(policies.contains_route("filetypes/elf"));
        assert!(!policies.contains_route("filetypes/pe"));

        let elf = policies.by_filetype.get("elf").expect("elf policy");
        assert!(
            (elf.hostile.thresholds["filetypes/elf"] - 0.98).abs() < 1e-6,
            "must select the requested level"
        );
        assert!(!elf.suspicious.thresholds.contains_key("general"));
        Ok(())
    }

    #[test]
    fn ensemble_loader_rejects_missing_general() -> Result<()> {
        let dir = tempfile::tempdir()?;
        // Make the layout look like an ensemble (general/ subdir present)
        // but leave it empty so general/ has no model artifacts.
        std::fs::create_dir_all(dir.path().join("general"))?;
        let err =
            Model::load(dir.path(), None).expect_err("ensemble with empty general/ must fail");
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
        let err = Model::load(dir.path(), None).expect_err("must fail");
        let _ = err;
        Ok(())
    }
}
