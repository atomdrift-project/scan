//! Model loading and inference.

use anyhow::{Context, Result};
use std::path::Path;

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
#[derive(Debug, Clone, Copy)]
pub struct Thresholds {
    /// Minimum probability to classify as suspicious.
    pub suspicious: f32,
    /// Minimum probability to classify as hostile.
    pub hostile: f32,
}

impl Thresholds {
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

/// Loaded model ready for inference.
#[derive(Debug)]
pub struct Model {
    inner: xgboost_native::Model,
    /// Feature specification used to build input vectors.
    pub spec: FeatureSpec,
    /// Classification thresholds.
    pub thresholds: Thresholds,
}

impl Model {
    /// Load model from a directory containing model.json and feature_spec.json.
    pub fn load(model_dir: &Path, thresholds: Thresholds) -> Result<Self> {
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

        tracing::info!(
            features = spec.total_features,
            threshold_suspicious = thresholds.suspicious,
            threshold_hostile = thresholds.hostile,
            "model loaded",
        );

        Ok(Self {
            inner,
            spec,
            thresholds,
        })
    }

    /// Run inference on a feature vector. Returns (probability, classification).
    pub fn predict(&self, features: &[f32]) -> Result<(f32, Classification)> {
        let probability = self.inner.predict(features);
        Ok((probability, self.thresholds.classify(probability)))
    }
}
