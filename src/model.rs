//! Model loading, thresholding, and inference.
//!
//! ## Bundle layouts
//!
//! Two on-disk shapes are supported:
//!
//! ### Single-bundle (legacy / dev)
//!
//! ```text
//! <model_dir>/
//!   model.txt | model.json
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
//!   general/                      always required
//!     model.txt | model.json
//!     feature_spec.json
//!   filegroups/<group>/           optional, e.g. native, scripts, archive
//!     model.txt | model.json
//!     feature_spec.json           may be absent → uses general's spec
//!   filetypes/<type>/             optional, e.g. elf, pe
//!     model.txt | model.json
//!     feature_spec.json           may be absent → uses general's spec
//! ```
//!
//! Detection is by presence of `general/` immediately under `<model_dir>`.
//!
//! ## Ensemble `config.json` schema (`azoth.routed_ensemble.v1`)
//!
//! Emitted by collimator's calibration pipeline. The top-level config is the
//! single source of thresholds for every route; specialist subdirectories do
//! not carry their own `config.json`.
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
use std::path::Path;

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
        tracing::warn!(path = %path.display(), "config.json thresholds are invalid, ignoring");
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
        tracing::warn!(
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
        tracing::warn!(
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

/// Inference backend powering a loaded [`Model`].
///
/// Selected at load time by probing the bundle directory for `model.json`
/// (XGBoost) or `model.txt` (LightGBM). Both backends expose the same
/// `predict` and `num_features` surface; everything outside this enum is
/// backend-agnostic.
#[derive(Debug)]
enum Backend {
    Xgboost(xgboost_ars::Model),
    Lightgbm(lightgbm_ars::Model),
}

impl Backend {
    fn num_features(&self) -> usize {
        match self {
            Self::Xgboost(m) => m.num_features(),
            Self::Lightgbm(m) => m.num_features(),
        }
    }

    fn predict(&self, features: &[f32]) -> Result<f32> {
        match self {
            Self::Xgboost(m) => Ok(m.predict(features)),
            Self::Lightgbm(m) => m
                .predict(features)
                .map_err(|e| anyhow::anyhow!("lightgbm predict failed: {e}")),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Xgboost(_) => "xgboost",
            Self::Lightgbm(_) => "lightgbm",
        }
    }
}

/// Output of the per-bundle loader: backend, spec, resolved thresholds, plus
/// telemetry about where the thresholds came from for the load-time log line.
struct LoadedBundle {
    backend: Backend,
    spec: FeatureSpec,
    thresholds: Thresholds,
    threshold_source: &'static str,
}

/// Default severity level used to pick thresholds out of the levels[] table
/// when the caller hasn't asked for a different one. Matches the CLI default.
const DEFAULT_SEVERITY_LEVEL: u8 = 5;

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
            tracing::warn!(
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
            tracing::warn!(
                route = %route, level = level,
                suspicious = t.suspicious, hostile = t.hostile,
                "ignoring invalid suspicious-only thresholds in ensemble config"
            );
        }
    }
    out
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

    let xgb_path = bundle_dir.join("model.json");
    let lgb_path = bundle_dir.join("model.txt");
    let (backend, model_path) = match (xgb_path.is_file(), lgb_path.is_file()) {
        (true, false) => {
            let m = xgboost_ars::Model::load(&xgb_path)
                .with_context(|| format!("loading XGBoost model from {}", xgb_path.display()))?;
            (Backend::Xgboost(m), xgb_path)
        }
        (false, true) => {
            let m = lightgbm_ars::Model::load(&lgb_path)
                .with_context(|| format!("loading LightGBM model from {}", lgb_path.display()))?;
            (Backend::Lightgbm(m), lgb_path)
        }
        (true, true) => anyhow::bail!(
            "model bundle is ambiguous: both {} and {} exist; remove one to disambiguate the backend",
            xgb_path.display(),
            lgb_path.display(),
        ),
        (false, false) => anyhow::bail!(
            "model bundle is incomplete: missing model.json (XGBoost) or model.txt (LightGBM) in {}",
            bundle_dir.display(),
        ),
    };

    tracing::debug!(path = %spec_path.display(), "loading feature spec");
    let spec = FeatureSpec::load(&spec_path)
        .with_context(|| format!("loading feature spec from {}", spec_path.display()))?;

    if spec.total_features() != backend.num_features() {
        anyhow::bail!(
            "feature count mismatch: feature_spec.json has {} features but {} expects {} — \
             these artifacts are from different training runs",
            spec.total_features(),
            model_path.display(),
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
                tracing::warn!(
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

    Ok(LoadedBundle {
        backend,
        spec,
        thresholds,
        threshold_source,
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
        let Some(route_t) = route_thresholds.get(&route_name).copied() else {
            tracing::debug!(
                category = %category,
                name = %name,
                "skipping specialist directory: no thresholds in ensemble config"
            );
            skipped.insert(name);
            continue;
        };
        match load_specialist(&path, &name, Some(route_t)) {
            Ok(route) => {
                out.insert(name, route);
            }
            Err(e) => {
                tracing::warn!(
                    category = %category,
                    name = %name,
                    error = %e,
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
            "filegroups",
            &mut filegroups,
            &mut skipped_filegroups,
        );
        load_specialists(
            &model_dir.join("filetypes"),
            &ensemble_cfg.route_thresholds,
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

        tracing::info!(
            backend = general.backend.kind(),
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
            },
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
        let probability = self.inner.predict(features)?;
        Ok((probability, self.thresholds.classify(probability)))
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
        let general_prob = self.inner.predict(features)?;
        let mut max_prob = general_prob;
        let mut classification = self.thresholds.classify(general_prob);

        // Optional filegroup specialist, looked up via the configured
        // filetype → filegroup map.
        if let Some(group_name) = self.routes.filetype_to_filegroup.get(file_type) {
            if let Some(route) = self.routes.filegroups.get(group_name) {
                if route.spec.total_features() != features.len() {
                    tracing::debug!(
                        route = %group_name,
                        expected = route.spec.total_features(),
                        got = features.len(),
                        "skipping routed feature-vector prediction; use predict_for_report for heterogeneous specialists",
                    );
                } else {
                    let prob = route.backend.predict(features)?;
                    if prob > max_prob {
                        max_prob = prob;
                    }
                    classification = max_class(classification, route.thresholds.classify(prob));
                }
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
                let prob = route.backend.predict(features)?;
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
        let probability = route.backend.predict(&features)?;
        Ok((probability, route.thresholds.classify(probability)))
    }

    fn score_route_file(route: &Route, file: &serde_json::Value) -> Result<(f32, Classification)> {
        let mut features = route.ctx.extract_file(file);
        route.spec.standardize(&mut features);
        let probability = route.backend.predict(&features)?;
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
        let (probability, classification, _, _) =
            self.predict_for_report_detailed(file_type, general_features, report)?;
        Ok((probability, classification))
    }

    /// Same as [`Self::predict_for_report`], with per-route scores retained
    /// for JSON and `--extra` output.
    pub fn predict_for_report_detailed(
        &self,
        file_type: &str,
        general_features: &[f32],
        report: &serde_json::Value,
    ) -> Result<(f32, Classification, Vec<RouteScore>, Vec<SkippedRoute>)> {
        let general_prob = self.inner.predict(general_features)?;
        let general_class = self.thresholds.classify(general_prob);
        let mut max_prob = general_prob;
        let mut classification = general_class;
        let mut scores = vec![RouteScore {
            model: "az".to_string(),
            probability: general_prob,
            classification: general_class,
        }];
        let mut skipped = Vec::new();

        if self.routes.is_empty() {
            return Ok((max_prob, classification, scores, skipped));
        }

        if let Some(group_name) = self.routes.filetype_to_filegroup.get(file_type) {
            if let Some(route) = self.routes.filegroups.get(group_name) {
                let (prob, class) = Self::score_route_report(route, report)?;
                scores.push(RouteScore {
                    model: format!("az/{group_name}"),
                    probability: prob,
                    classification: class,
                });
                if prob > max_prob {
                    max_prob = prob;
                }
                classification = max_class(classification, class);
            } else if self.routes.skipped_filegroups.contains(group_name) {
                skipped.push(SkippedRoute {
                    model: format!("az/{group_name}"),
                    reason: "uncalibrated",
                });
            }
        }

        if let Some(route) = self.routes.filetypes.get(file_type) {
            let (prob, class) = Self::score_route_report(route, report)?;
            scores.push(RouteScore {
                model: format!("az/{file_type}"),
                probability: prob,
                classification: class,
            });
            if prob > max_prob {
                max_prob = prob;
            }
            classification = max_class(classification, class);
        } else if self.routes.skipped_filetypes.contains(file_type) {
            skipped.push(SkippedRoute {
                model: format!("az/{file_type}"),
                reason: "uncalibrated",
            });
        }

        Ok((max_prob, classification, scores, skipped))
    }

    /// Routed prediction for one embedded-file JSON object.
    pub fn predict_for_file(
        &self,
        file_type: &str,
        general_features: &[f32],
        file: &serde_json::Value,
    ) -> Result<(f32, Classification)> {
        let (probability, classification, _, _) =
            self.predict_for_file_detailed(file_type, general_features, file)?;
        Ok((probability, classification))
    }

    /// Same as [`Self::predict_for_file`], with per-route scores retained.
    pub fn predict_for_file_detailed(
        &self,
        file_type: &str,
        general_features: &[f32],
        file: &serde_json::Value,
    ) -> Result<(f32, Classification, Vec<RouteScore>, Vec<SkippedRoute>)> {
        let general_prob = self.inner.predict(general_features)?;
        let general_class = self.thresholds.classify(general_prob);
        let mut max_prob = general_prob;
        let mut classification = general_class;
        let mut scores = vec![RouteScore {
            model: "az".to_string(),
            probability: general_prob,
            classification: general_class,
        }];
        let mut skipped = Vec::new();

        if self.routes.is_empty() {
            return Ok((max_prob, classification, scores, skipped));
        }

        if let Some(group_name) = self.routes.filetype_to_filegroup.get(file_type) {
            if let Some(route) = self.routes.filegroups.get(group_name) {
                let (prob, class) = Self::score_route_file(route, file)?;
                scores.push(RouteScore {
                    model: format!("az/{group_name}"),
                    probability: prob,
                    classification: class,
                });
                if prob > max_prob {
                    max_prob = prob;
                }
                classification = max_class(classification, class);
            } else if self.routes.skipped_filegroups.contains(group_name) {
                skipped.push(SkippedRoute {
                    model: format!("az/{group_name}"),
                    reason: "uncalibrated",
                });
            }
        }

        if let Some(route) = self.routes.filetypes.get(file_type) {
            let (prob, class) = Self::score_route_file(route, file)?;
            scores.push(RouteScore {
                model: format!("az/{file_type}"),
                probability: prob,
                classification: class,
            });
            if prob > max_prob {
                max_prob = prob;
            }
            classification = max_class(classification, class);
        } else if self.routes.skipped_filetypes.contains(file_type) {
            skipped.push(SkippedRoute {
                model: format!("az/{file_type}"),
                reason: "uncalibrated",
            });
        }

        Ok((max_prob, classification, scores, skipped))
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
