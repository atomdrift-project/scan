//! Model loading, thresholding, and inference.

use anyhow::{Context, Result};
use std::fmt;
use std::path::Path;

use crate::features::{FeatureSpec, EXPECTED_MODEL_ABI_VERSION};

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

/// Loaded model plus the feature spec and thresholds used for inference.
#[derive(Debug)]
pub struct Model {
    inner: xgboost_ars::Model,
    spec: FeatureSpec,
    thresholds: Thresholds,
    info: ModelInfo,
}

impl Model {
    /// Load model artifacts from a directory containing `model.json` and
    /// `feature_spec.json`.
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
        let model_path = model_dir.join("model.json");
        let spec_path = model_dir.join("feature_spec.json");

        if !spec_path.is_file() {
            anyhow::bail!(
                "model bundle is incomplete: missing {}. Run 'litmus update-rules' to refresh the installed models.",
                spec_path.display(),
            );
        }
        if !model_path.is_file() {
            anyhow::bail!(
                "model bundle is incomplete: missing {}. Run 'litmus update-rules' to refresh the installed models.",
                model_path.display(),
            );
        }

        tracing::debug!(path = %spec_path.display(), "loading feature spec");
        let spec = FeatureSpec::load(&spec_path)
            .with_context(|| format!("loading feature spec from {}", spec_path.display()))?;
        tracing::debug!(features = spec.total_features(), "feature spec loaded");

        tracing::debug!(path = %model_path.display(), "loading model");
        let inner = xgboost_ars::Model::load(&model_path)
            .with_context(|| format!("loading model from {}", model_path.display()))?;
        tracing::debug!("model loaded");

        // Cross-validate feature counts between spec and model.
        if spec.total_features() != inner.num_features() {
            anyhow::bail!(
                "feature count mismatch: feature_spec.json has {} features but model.json \
                 expects {} — these artifacts are from different training runs",
                spec.total_features(),
                inner.num_features(),
            );
        }

        // Resolve thresholds: explicit > config.json > evaluation.json > fallback.
        let config_thresholds = load_config_thresholds(model_dir);
        let recommended = load_evaluation_thresholds(model_dir);
        let effective_thresholds = match thresholds {
            Some(explicit) => {
                explicit
                    .validate()
                    .map_err(|error| anyhow::anyhow!("invalid thresholds: {error}"))?;
                if let Some(ref rec) = recommended {
                    explicit.warn_if_divergent(rec);
                }
                explicit
            }
            None => config_thresholds.or(recommended).unwrap_or_else(|| {
                tracing::warn!(
                    "no config.json or evaluation.json thresholds found — using conservative \
                     fallback (suspicious={}, hostile={})",
                    Thresholds::FALLBACK_SUSPICIOUS,
                    Thresholds::FALLBACK_HOSTILE,
                );
                Thresholds {
                    suspicious: Thresholds::FALLBACK_SUSPICIOUS,
                    hostile: Thresholds::FALLBACK_HOSTILE,
                }
            }),
        };

        tracing::info!(
            features = spec.total_features(),
            model_abi_version = spec.abi_version(),
            threshold_suspicious = effective_thresholds.suspicious,
            threshold_hostile = effective_thresholds.hostile,
            threshold_source = if thresholds.is_some() {
                "explicit"
            } else if config_thresholds.is_some() {
                "config.json"
            } else if recommended.is_some() {
                "evaluation.json"
            } else {
                "fallback"
            },
            spec_version = spec.version(),
            "model loaded",
        );

        let info = ModelInfo {
            version: spec.version(),
            abi_version: spec.abi_version(),
            sha256: String::new(),
            commit: None,
        };

        Ok(Self {
            inner,
            spec,
            thresholds: effective_thresholds,
            info,
        })
    }

    /// Run inference on a feature vector.
    ///
    /// Returns the raw probability together with the derived classification.
    /// The caller is responsible for ensuring the feature vector matches the
    /// loaded [`FeatureSpec`] shape and ordering.
    ///
    /// # Errors
    /// Returns an error if the underlying model backend fails to produce a
    /// prediction for the provided feature vector.
    pub fn predict(&self, features: &[f32]) -> Result<(f32, Classification)> {
        let probability = self.inner.predict(features);
        Ok((probability, self.thresholds.classify(probability)))
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
}
