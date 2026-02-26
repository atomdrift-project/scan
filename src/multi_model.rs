//! Multi-model classifier that routes to per-filetype models.

use crate::classifier::{Classifier, ClassificationResult};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Model registry that maps file types to models
#[derive(Debug, Clone, Deserialize)]
pub struct ModelRegistry {
    models: HashMap<String, ModelMetadata>,
    default_model: String,

    #[serde(skip)]
    model_dir: PathBuf,

    #[serde(skip)]
    cache: HashMap<String, Classifier>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelMetadata {
    model_file: String,
    #[allow(dead_code)]
    evaluation_file: String,
    #[allow(dead_code)]
    n_samples: usize,
    #[allow(dead_code)]
    n_benign: usize,
    #[allow(dead_code)]
    n_malware: usize,
    #[allow(dead_code)]
    roc_auc: f64,
    #[allow(dead_code)]
    f1: f64,
    optimal_threshold: f64,
}

impl ModelRegistry {
    /// Load model registry from JSON file
    pub fn load(registry_path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(registry_path)
            .context(format!("Failed to read registry: {:?}", registry_path))?;

        let mut registry: Self = serde_json::from_str(&data)
            .context("Failed to parse model registry JSON")?;

        registry.model_dir = registry_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        Ok(registry)
    }

    /// Try to load from default locations
    pub fn load_default() -> Result<Self> {
        // Check LITMUS_MODEL_PATH environment variable
        if let Ok(path) = std::env::var("LITMUS_MODEL_PATH") {
            let model_dir = Path::new(&path).parent().unwrap_or_else(|| Path::new("."));
            let registry_path = model_dir.join("model_registry.json");
            if registry_path.exists() {
                return Self::load(&registry_path);
            }
        }

        // Check ~/.litmus/models/model_registry.json
        if let Some(home) = dirs::home_dir() {
            let registry_path = home
                .join(".litmus")
                .join("models")
                .join("model_registry.json");
            if registry_path.exists() {
                return Self::load(&registry_path);
            }
        }

        // Check ./models/model_registry.json
        let registry_path = Path::new("models/model_registry.json");
        if registry_path.exists() {
            return Self::load(registry_path);
        }

        anyhow::bail!("No model registry found. Train per-filetype models or use single model.")
    }

    /// Get model for a specific file type
    fn get_model(&mut self, file_type: &str) -> Result<&Classifier> {
        // Check if already cached
        if self.cache.contains_key(file_type) {
            return Ok(self.cache.get(file_type).unwrap());
        }

        // Look up model metadata
        let model_file = if let Some(meta) = self.models.get(file_type) {
            &meta.model_file
        } else {
            // Fall back to default model
            &self.default_model
        };

        // Load model
        let model_path = self.model_dir.join(model_file);
        let mut classifier = Classifier::load(&model_path)
            .context(format!("Failed to load model for {}", file_type))?;

        // Set optimal threshold if available
        if let Some(meta) = self.models.get(file_type) {
            classifier = classifier.with_hostile_threshold(meta.optimal_threshold as f32);
        }

        // Cache for future use
        self.cache.insert(file_type.to_string(), classifier);

        Ok(self.cache.get(file_type).unwrap())
    }

    /// Classify a file with the appropriate model
    pub fn classify(
        &mut self,
        file_type: &str,
        features: &[f32],
    ) -> Result<ClassificationResult> {
        let classifier = self.get_model(file_type)?;
        classifier.classify(features)
    }

    /// Check if registry is empty (should use default classifier instead)
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

/// Detect file type from file path or extension
pub fn detect_filetype(path: &Path) -> String {
    let path_str = path.to_string_lossy();
    let path_lower = path_str.to_lowercase();

    // Check for common extensions
    if path_lower.ends_with(".py") {
        "python".to_string()
    } else if path_lower.ends_with(".js") {
        "javascript".to_string()
    } else if path_lower.ends_with(".exe") || path_lower.ends_with(".dll") || path_lower.ends_with(".sys") {
        "pe".to_string()
    } else if path_lower.ends_with(".elf") || path_lower.contains("/bin/") || path_lower.contains("/sbin/") {
        "elf".to_string()
    } else if path_lower.ends_with(".sh") || path_lower.ends_with(".bash") {
        "shell".to_string()
    } else if path_lower.ends_with(".rb") {
        "ruby".to_string()
    } else if path_lower.ends_with(".go") {
        "go".to_string()
    } else if path_lower.ends_with(".rs") {
        "rust".to_string()
    } else if path_lower.ends_with(".jar") {
        "java".to_string()
    } else if path_lower.ends_with(".apk") {
        "android".to_string()
    } else {
        "unknown".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_filetype() {
        assert_eq!(detect_filetype(Path::new("test.py")), "python");
        assert_eq!(detect_filetype(Path::new("test.js")), "javascript");
        assert_eq!(detect_filetype(Path::new("test.exe")), "pe");
        assert_eq!(detect_filetype(Path::new("/bin/ls")), "elf");
        assert_eq!(detect_filetype(Path::new("test.sh")), "shell");
        assert_eq!(detect_filetype(Path::new("unknown.txt")), "unknown");
    }
}
