//! Model loading, thresholding, and inference.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::Read;
use std::path::Path;
use std::process::Command;

use crate::features::FeatureSpec;

/// Classification outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Classification {
    /// File shows no significant malicious indicators.
    Benign,
    /// File has notable suspicious indicators.
    Suspicious,
    /// File is likely malicious.
    Hostile,
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
/// - `suspicious < hostile`
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
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Thresholds {
    /// Minimum probability to classify as suspicious.
    pub suspicious: f32,
    /// Minimum probability to classify as hostile.
    pub hostile: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            suspicious: Self::DEFAULT_SUSPICIOUS,
            hostile: Self::DEFAULT_HOSTILE,
        }
    }
}

impl Thresholds {
    /// Default suspicious threshold.
    pub const DEFAULT_SUSPICIOUS: f32 = 0.975;
    /// Default hostile threshold.
    pub const DEFAULT_HOSTILE: f32 = 0.99;

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
        if self.suspicious >= self.hostile {
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
                "suspicious threshold ({suspicious}) must be less than hostile threshold ({hostile})"
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
    /// SHA-256 hex digest of the model file.
    pub sha256: String,
    /// Short git commit hash of the models repository, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

/// Loaded model plus the feature spec and thresholds used for inference.
#[derive(Debug)]
pub struct Model {
    inner: xgboost_native::Model,
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
    pub fn load(model_dir: &Path, thresholds: Thresholds) -> Result<Self> {
        thresholds
            .validate()
            .map_err(|error| anyhow::anyhow!("invalid thresholds: {error}"))?;

        let model_path = model_dir.join("model.json");
        let spec_path = model_dir.join("feature_spec.json");

        tracing::debug!(path = %spec_path.display(), "loading feature spec");
        let spec = FeatureSpec::load(&spec_path)
            .with_context(|| format!("loading feature spec from {}", spec_path.display()))?;
        tracing::debug!(features = spec.total_features(), "feature spec loaded");

        tracing::debug!(path = %model_path.display(), "loading model");
        let inner = xgboost_native::Model::load(&model_path)
            .with_context(|| format!("loading model from {}", model_path.display()))?;
        tracing::debug!("model loaded");

        let model_sha256 = {
            let mut file = std::fs::File::open(&model_path)
                .with_context(|| format!("opening {}", model_path.display()))?;
            let mut hasher = Sha256::new();
            let mut buf = [0u8; 65536];
            loop {
                let n = file.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
            }
            format!("{:x}", hasher.finalize())
        };

        let commit = Command::new("git")
            .args([
                "-C",
                &model_dir.to_string_lossy(),
                "rev-parse",
                "--short",
                "HEAD",
            ])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

        tracing::info!(
            features = spec.total_features(),
            threshold_suspicious = thresholds.suspicious,
            threshold_hostile = thresholds.hostile,
            model_sha256 = %model_sha256,
            model_commit = ?commit,
            spec_version = spec.version(),
            "model loaded",
        );

        let info = ModelInfo {
            version: spec.version(),
            sha256: model_sha256,
            commit,
        };

        Ok(Self {
            inner,
            spec,
            thresholds,
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
