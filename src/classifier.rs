//! XGBoost model inference for malware classification.
//!
//! Implements native XGBoost JSON model parsing and tree traversal.
//! Zero external dependencies - no ONNX runtime needed.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Classification result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classification {
    Benign,
    Suspicious,
    Hostile,
}

impl std::fmt::Display for Classification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Classification::Benign => write!(f, "BENIGN"),
            Classification::Suspicious => write!(f, "SUSPICIOUS"),
            Classification::Hostile => write!(f, "HOSTILE"),
        }
    }
}

/// Result of classification
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// The classification category
    pub classification: Classification,
    /// Probability of hostile class (0.0-1.0)
    pub probability: f32,
    /// Confidence in the classification (distance from nearest threshold)
    #[allow(dead_code)]
    pub confidence: f32,
    /// Threshold used for hostile classification
    pub hostile_threshold: f32,
    /// Threshold used for suspicious classification
    pub suspicious_threshold: f32,
}

/// XGBoost decision tree
#[derive(Debug, Clone, Deserialize)]
struct Tree {
    base_weights: Vec<f64>,
    left_children: Vec<i32>,
    right_children: Vec<i32>,
    split_conditions: Vec<f64>,
    split_indices: Vec<usize>,
    default_left: Vec<i32>,
}

impl Tree {
    /// Traverse the tree and return the leaf value
    fn predict(&self, features: &[f32]) -> f64 {
        let mut node_idx = 0usize;
        loop {
            // Check if current node is a leaf (left child = -1)
            if self.left_children[node_idx] == -1 {
                return self.base_weights[node_idx];
            }

            // Get split feature and condition
            let split_idx = self.split_indices[node_idx];
            let split_cond = self.split_conditions[node_idx];

            // Get feature value (treat out-of-bounds as NaN)
            let feature_val = if split_idx < features.len() {
                features[split_idx] as f64
            } else {
                f64::NAN
            };

            // Determine which child to visit
            node_idx = if feature_val.is_nan() {
                // Use default direction for missing values
                if self.default_left[node_idx] == 1 {
                    self.left_children[node_idx] as usize
                } else {
                    self.right_children[node_idx] as usize
                }
            } else if feature_val < split_cond {
                self.left_children[node_idx] as usize
            } else {
                self.right_children[node_idx] as usize
            };
        }
    }
}

/// XGBoost model for binary classification
#[derive(Debug, Clone)]
pub struct Classifier {
    trees: Vec<Tree>,
    #[allow(dead_code)]
    feature_names: Vec<String>,
    hostile_threshold: f32,
    suspicious_threshold: f32,
}

impl Classifier {
    /// Load classifier from XGBoost JSON model file
    pub fn load(model_path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(model_path)
            .context(format!("Failed to read model file: {:?}", model_path))?;

        Self::from_json(&data)
    }

    /// Load classifier from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        // Parse the XGBoost JSON format
        #[derive(Deserialize)]
        struct RawModel {
            learner: Learner,
        }

        #[derive(Deserialize)]
        struct Learner {
            feature_names: Option<Vec<String>>,
            gradient_booster: GradientBooster,
        }

        #[derive(Deserialize)]
        struct GradientBooster {
            model: GBTreeModel,
        }

        #[derive(Deserialize)]
        struct GBTreeModel {
            trees: Vec<Tree>,
        }

        let raw: RawModel =
            serde_json::from_str(json).context("Failed to parse XGBoost JSON model")?;

        let feature_names = raw.learner.feature_names.unwrap_or_default();
        let trees = raw.learner.gradient_booster.model.trees;

        Ok(Self {
            trees,
            feature_names,
            hostile_threshold: 0.80,
            suspicious_threshold: 0.40,
        })
    }

    /// Try to load from default locations
    pub fn load_default() -> Result<Self> {
        // Check LITMUS_MODEL_PATH environment variable
        if let Ok(path) = std::env::var("LITMUS_MODEL_PATH") {
            let model_path = Path::new(&path);
            if model_path.exists() {
                return Self::load(model_path);
            }
        }

        // Check ~/.litmus/models/litmus_v1.json
        if let Some(home) = dirs::home_dir() {
            let model_path = home.join(".litmus").join("models").join("litmus_v1.json");
            if model_path.exists() {
                return Self::load(&model_path);
            }
        }

        // Check ./models/litmus_v1.json
        let model_path = Path::new("models/litmus_v1.json");
        if model_path.exists() {
            return Self::load(model_path);
        }

        anyhow::bail!(
            "No model found. Set LITMUS_MODEL_PATH or place model at ~/.litmus/models/litmus_v1.json"
        )
    }

    /// Set the hostile threshold
    pub fn with_hostile_threshold(mut self, threshold: f32) -> Self {
        self.hostile_threshold = threshold;
        self
    }

    /// Set both thresholds
    #[allow(dead_code)]
    pub fn with_thresholds(mut self, hostile: f32, suspicious: f32) -> Self {
        self.hostile_threshold = hostile;
        self.suspicious_threshold = suspicious;
        self
    }

    /// Classify a feature vector
    pub fn classify(&self, features: &[f32]) -> Result<ClassificationResult> {
        // Sum all tree predictions (binary classification uses single output)
        let mut raw_score: f64 = 0.0;
        for tree in &self.trees {
            raw_score += tree.predict(features);
        }

        // Apply sigmoid for binary classification
        let probability = sigmoid(raw_score) as f32;

        // Determine classification based on thresholds
        let classification = if probability > self.hostile_threshold {
            Classification::Hostile
        } else if probability > self.suspicious_threshold {
            Classification::Suspicious
        } else {
            Classification::Benign
        };

        // Calculate confidence as distance from nearest threshold
        let confidence = match classification {
            Classification::Hostile => probability - self.hostile_threshold,
            Classification::Suspicious => {
                let dist_to_hostile = self.hostile_threshold - probability;
                let dist_to_suspicious = probability - self.suspicious_threshold;
                dist_to_hostile.min(dist_to_suspicious)
            }
            Classification::Benign => self.suspicious_threshold - probability,
        };

        Ok(ClassificationResult {
            classification,
            probability,
            confidence,
            hostile_threshold: self.hostile_threshold,
            suspicious_threshold: self.suspicious_threshold,
        })
    }

    /// Get feature names (for explainability)
    #[allow(dead_code)]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Classify using a feature map (for compatibility with named features)
    #[allow(dead_code)]
    pub fn classify_map(&self, features: &HashMap<String, f32>) -> Result<ClassificationResult> {
        // Build feature vector in correct order
        let mut fvec = vec![0.0f32; self.feature_names.len()];
        for (i, name) in self.feature_names.iter().enumerate() {
            if let Some(&val) = features.get(name) {
                fvec[i] = val;
            }
        }
        self.classify(&fvec)
    }
}

/// Sigmoid function for binary classification
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_classification_thresholds() {
        // Test that thresholds work correctly
        let classifier = Classifier {
            trees: vec![],
            feature_names: vec![],
            hostile_threshold: 0.80,
            suspicious_threshold: 0.40,
        };

        // With no trees, raw_score = 0, sigmoid(0) = 0.5
        // 0.5 is between 0.40 and 0.80, so SUSPICIOUS
        let result = classifier.classify(&[]).unwrap();
        assert_eq!(result.classification, Classification::Suspicious);
        assert!((result.probability - 0.5).abs() < 0.01);
    }
}
