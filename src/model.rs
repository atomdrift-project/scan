//! ONNX model loading and inference.

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Mutex;

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
/// Session::run requires &mut self, so we wrap in Mutex for thread safety.
pub struct Model {
    session: Mutex<ort::session::Session>,
    /// Feature specification used to build input vectors.
    pub spec: FeatureSpec,
    /// Classification thresholds.
    pub thresholds: Thresholds,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("spec", &self.spec)
            .field("thresholds", &self.thresholds)
            .finish_non_exhaustive()
    }
}

impl Model {
    /// Load model from a directory containing model.onnx and feature_spec.json.
    pub fn load(model_dir: &Path, thresholds: Thresholds) -> Result<Self> {
        let onnx_path = model_dir.join("model.onnx");
        let spec_path = model_dir.join("feature_spec.json");

        let spec = FeatureSpec::load(&spec_path)
            .with_context(|| format!("loading feature spec from {}", spec_path.display()))?;

        let mut builder = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("creating ONNX session builder: {e}"))?;

        builder = builder
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("setting intra threads: {e}"))?;

        let session = builder
            .commit_from_file(&onnx_path)
            .map_err(|e| anyhow::anyhow!("loading ONNX model from {}: {e}", onnx_path.display()))?;

        tracing::info!(
            features = spec.total_features,
            threshold_suspicious = thresholds.suspicious,
            threshold_hostile = thresholds.hostile,
            "model loaded",
        );

        Ok(Self {
            session: Mutex::new(session),
            spec,
            thresholds,
        })
    }

    /// Run inference on a feature vector. Returns (probability, classification).
    pub fn predict(&self, features: &[f32]) -> Result<(f32, Classification)> {
        let input = ort::value::Tensor::from_array(([1, features.len()], features.to_vec()))
            .map_err(|e| anyhow::anyhow!("creating input tensor: {e}"))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("locking session: {e}"))?;

        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| anyhow::anyhow!("running inference: {e}"))?;

        // XGBoost ONNX: outputs[0] = label(i64), outputs[1] = probabilities(f32, [batch, 2])
        let (_shape, data) = outputs
            .get("probabilities")
            .ok_or_else(|| anyhow::anyhow!("model has no 'probabilities' output"))?
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("extracting probabilities tensor: {e}"))?;
        // Shape [1, 2]: [benign_prob, malware_prob] — take class 1.
        let probability = data.get(1).copied().unwrap_or(0.0);

        Ok((probability, self.thresholds.classify(probability)))
    }
}
