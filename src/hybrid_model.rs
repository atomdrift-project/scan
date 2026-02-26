//! Hybrid classifier that intelligently routes between per-filetype and single models.
//!
//! Uses per-filetype models when they have sufficient training data (500+ malware samples),
//! falls back to single unified model for rare or poorly-represented file types.

use crate::classifier::{Classifier, ClassificationResult};
use crate::multi_model::{ModelRegistry, detect_filetype};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Minimum malware samples required for a per-filetype model to be considered "well-trained"
const MIN_MALWARE_SAMPLES: usize = 500;

/// Hybrid classifier that routes intelligently between per-type and single models
pub struct HybridClassifier {
    /// Per-filetype model registry (optional)
    registry: Option<ModelRegistry>,
    /// Single unified model (fallback)
    single_model: Classifier,
    /// Metadata about which models are well-trained
    well_trained_types: HashMap<String, usize>,
}

impl HybridClassifier {
    /// Load hybrid classifier with both per-type and single models
    pub fn load_default() -> Result<Self> {
        // Load single model (required)
        let single_model = Classifier::load_default()
            .context("Failed to load single model - required for hybrid classifier")?;

        // Try to load per-filetype registry (optional)
        let (registry, well_trained_types) = match ModelRegistry::load_default() {
            Ok(reg) => {
                let trained = Self::identify_well_trained_models(&reg)?;
                (Some(reg), trained)
            }
            Err(_) => {
                // No registry found - will use single model for all classifications
                (None, HashMap::new())
            }
        };

        Ok(Self {
            registry,
            single_model,
            well_trained_types,
        })
    }

    /// Identify which per-filetype models have sufficient training data
    fn identify_well_trained_models(_registry: &ModelRegistry) -> Result<HashMap<String, usize>> {
        let registry_path = Self::find_registry_path()?;
        let registry_dir = registry_path.parent().unwrap_or_else(|| Path::new("."));

        // Load model_registry.json to get sample counts
        let registry_json_path = registry_dir.join("model_registry.json");
        let data = std::fs::read_to_string(&registry_json_path)
            .context("Failed to read model_registry.json")?;

        #[derive(Deserialize)]
        struct RegistryData {
            models: HashMap<String, ModelMetadata>,
        }

        #[derive(Deserialize)]
        struct ModelMetadata {
            n_malware: usize,
        }

        let registry_data: RegistryData = serde_json::from_str(&data)
            .context("Failed to parse model_registry.json")?;

        // Filter models with sufficient malware samples
        let mut well_trained = HashMap::new();
        for (ftype, metadata) in registry_data.models {
            if metadata.n_malware >= MIN_MALWARE_SAMPLES {
                well_trained.insert(ftype, metadata.n_malware);
            }
        }

        Ok(well_trained)
    }

    /// Find the registry path (same logic as ModelRegistry)
    fn find_registry_path() -> Result<std::path::PathBuf> {
        // Check LITMUS_MODEL_PATH environment variable
        if let Ok(path) = std::env::var("LITMUS_MODEL_PATH") {
            let model_dir = Path::new(&path).parent().unwrap_or_else(|| Path::new("."));
            let registry_path = model_dir.join("model_registry.json");
            if registry_path.exists() {
                return Ok(registry_path);
            }
        }

        // Check ~/.litmus/models/model_registry.json
        if let Some(home) = dirs::home_dir() {
            let registry_path = home
                .join(".litmus")
                .join("models")
                .join("model_registry.json");
            if registry_path.exists() {
                return Ok(registry_path);
            }
        }

        // Check ./models/model_registry.json
        let registry_path = Path::new("models/model_registry.json");
        if registry_path.exists() {
            return Ok(registry_path.to_path_buf());
        }

        anyhow::bail!("No model registry found")
    }

    /// Classify using the most appropriate model
    pub fn classify(&mut self, path: &Path, features: &[f32]) -> Result<ClassificationResult> {
        let file_type = detect_filetype(path);

        // Check if we have a well-trained per-type model for this file type
        if self.well_trained_types.contains_key(&file_type) {
            if let Some(ref mut registry) = self.registry {
                // Use per-filetype model (eliminates file type bias)
                return registry.classify(&file_type, features);
            }
        }

        // Fall back to single model for rare/poorly-trained types
        self.single_model.classify(features)
    }

    /// Get statistics about the hybrid classifier
    pub fn stats(&self) -> HybridStats {
        HybridStats {
            has_registry: self.registry.is_some(),
            well_trained_types: self.well_trained_types.clone(),
            uses_single_model_fallback: true,
        }
    }

    /// Set hostile threshold for the single model
    pub fn with_hostile_threshold(mut self, threshold: f32) -> Self {
        self.single_model = self.single_model.with_hostile_threshold(threshold);
        self
    }
}

/// Statistics about the hybrid classifier
#[derive(Debug, Clone)]
pub struct HybridStats {
    /// Whether per-filetype registry is available
    pub has_registry: bool,
    /// File types with well-trained models (and their malware sample counts)
    pub well_trained_types: HashMap<String, usize>,
    /// Whether single model is used as fallback
    pub uses_single_model_fallback: bool,
}

impl std::fmt::Display for HybridStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Hybrid Classifier Statistics:")?;
        writeln!(f, "  Per-filetype registry: {}", if self.has_registry { "available" } else { "not found" })?;
        writeln!(f, "  Well-trained models: {}", self.well_trained_types.len())?;

        if !self.well_trained_types.is_empty() {
            writeln!(f, "\n  File types with per-type models ({}+ malware samples):", MIN_MALWARE_SAMPLES)?;
            let mut types: Vec<_> = self.well_trained_types.iter().collect();
            types.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
            for (ftype, count) in types {
                writeln!(f, "    {}: {} malware samples", ftype, count)?;
            }
        }

        writeln!(f, "\n  Fallback: {}", if self.uses_single_model_fallback {
            "single model (for rare types)"
        } else {
            "none"
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_samples_threshold() {
        assert!(MIN_MALWARE_SAMPLES >= 500, "Minimum samples should be at least 500");
    }
}
