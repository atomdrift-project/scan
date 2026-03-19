//! Model loading and inference.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
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

/// Thresholds for classification.
#[derive(Debug, Clone, Copy, serde::Serialize)]
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

    /// Validate that thresholds are sane: both in [0.0, 1.0] and suspicious < hostile.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.suspicious) {
            return Err(format!(
                "suspicious threshold {} is outside [0.0, 1.0]",
                self.suspicious
            ));
        }
        if !(0.0..=1.0).contains(&self.hostile) {
            return Err(format!(
                "hostile threshold {} is outside [0.0, 1.0]",
                self.hostile
            ));
        }
        if self.suspicious >= self.hostile {
            return Err(format!(
                "suspicious threshold ({}) must be less than hostile threshold ({})",
                self.suspicious, self.hostile
            ));
        }
        Ok(())
    }

    /// Classify a probability score into a [`Classification`].
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

/// Metadata about the loaded model, computed once at startup.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
    /// Feature spec version (e.g. 13).
    pub version: u32,
    /// SHA-256 hex digest of the model file.
    pub sha256: String,
    /// Short git commit hash of the models repository, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

/// Loaded model ready for inference.
#[derive(Debug)]
pub struct Model {
    inner: xgboost_native::Model,
    /// Feature specification used to build input vectors.
    pub spec: FeatureSpec,
    /// Classification thresholds.
    pub thresholds: Thresholds,
    /// Model metadata (version, sha256, commit).
    pub info: ModelInfo,
}

impl Model {
    /// Load model from a directory containing model.json and feature_spec.json.
    #[must_use]
    pub fn load(model_dir: &Path, thresholds: Thresholds) -> Result<Self> {
        thresholds
            .validate()
            .map_err(|e| anyhow::anyhow!("invalid thresholds: {e}"))?;

        let model_path = model_dir.join("model.json");
        let spec_path = model_dir.join("feature_spec.json");

        tracing::debug!(path = %spec_path.display(), "loading feature spec");
        let spec = FeatureSpec::load(&spec_path)
            .with_context(|| format!("loading feature spec from {}", spec_path.display()))?;
        tracing::debug!(features = spec.total_features, "feature spec loaded");

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
            features = spec.total_features,
            threshold_suspicious = thresholds.suspicious,
            threshold_hostile = thresholds.hostile,
            model_sha256 = %model_sha256,
            model_commit = ?commit,
            spec_version = spec.version,
            "model loaded",
        );

        let info = ModelInfo {
            version: spec.version,
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

    /// Run inference on a feature vector. Returns (probability, classification).
    pub fn predict(&self, features: &[f32]) -> Result<(f32, Classification)> {
        let probability = self.inner.predict(features);
        Ok((probability, self.thresholds.classify(probability)))
    }
}
