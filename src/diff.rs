//! Diff command implementation - compare versions for supply chain attack detection.
//!
//! Uses cleave's DiffAnalyzer for comprehensive diff analysis and compares
//! ML classifications between versions to detect supply chain attacks.

use crate::classifier::{Classification, Classifier, ClassificationResult};
use crate::features::FeatureExtractor;
use crate::multi_model::{ModelRegistry, detect_filetype};
use crate::output;
use crate::OutputFormat;
use anyhow::{Context, Result};
use cleave::diff::DiffAnalyzer;
use cleave::DiffReport;
use std::path::Path;

/// Risk level for supply chain changes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::Low => write!(f, "LOW"),
            RiskLevel::Medium => write!(f, "MEDIUM"),
            RiskLevel::High => write!(f, "HIGH"),
            RiskLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Result of diff analysis
pub struct DiffResult {
    pub old_classification: Option<crate::classifier::ClassificationResult>,
    pub new_classification: Option<crate::classifier::ClassificationResult>,
    pub risk_level: RiskLevel,
    /// What triggered the risk level
    pub risk_source: RiskSource,
    pub new_capabilities: Vec<String>,
    pub removed_capabilities: Vec<String>,
    pub feature_deltas: Vec<(String, f32, f32)>, // (name, old_value, new_value)
}

/// Source of the risk assessment
#[derive(Debug, Clone)]
pub enum RiskSource {
    /// Full file classification changed to hostile/suspicious
    FileClassification,
    /// New hostile findings from cleave
    NewHostileFindings(usize),
    /// New suspicious findings from cleave
    NewSuspiciousFindings(usize),
    /// Binary metrics anomaly (huge functions, size spike)
    BinaryAnomaly(String),
    /// Dangerous capability categories added
    DangerousCapabilities(usize),
    /// No significant risk detected
    None,
}

/// Classify features using either multi-model registry or single model
fn classify_features(path: &Path, features: &[f32], threshold: f32) -> Result<Option<ClassificationResult>> {
    // Try multi-model registry first (preferred for per-filetype models)
    match ModelRegistry::load_default() {
        Ok(mut registry) => {
            if registry.is_empty() {
                // Registry exists but has no models, fall back to single model
                return classify_with_single_model(features, threshold);
            }

            let file_type = detect_filetype(path);
            match registry.classify(&file_type, features) {
                Ok(result) => Ok(Some(result)),
                Err(e) => {
                    eprintln!("Warning: Multi-model classification failed: {}", e);
                    classify_with_single_model(features, threshold)
                }
            }
        }
        Err(_) => {
            // No registry found, use single model (backward compatibility)
            classify_with_single_model(features, threshold)
        }
    }
}

/// Classify using single model (backward compatibility)
fn classify_with_single_model(features: &[f32], threshold: f32) -> Result<Option<ClassificationResult>> {
    match Classifier::load_default() {
        Ok(classifier) => {
            let classifier = classifier.with_hostile_threshold(threshold);
            Ok(Some(classifier.classify(features)?))
        }
        Err(_) => {
            Ok(None)
        }
    }
}

/// Run the diff command
pub fn run(old_path: &Path, new_path: &Path, threshold: f32, format: OutputFormat) -> Result<()> {
    if !old_path.exists() {
        anyhow::bail!("Old path does not exist: {:?}", old_path);
    }
    if !new_path.exists() {
        anyhow::bail!("New path does not exist: {:?}", new_path);
    }

    // Use cleave's DiffAnalyzer for comprehensive diff
    let diff_analyzer = DiffAnalyzer::new(old_path, new_path);
    let diff_report: DiffReport = diff_analyzer.analyze()
        .context("cleave diff analysis failed")?;

    // Also get full reports for feature extraction and ML classification
    let options = cleave::AnalysisOptions {
        disable_yara: false,
        disable_radare2: false,
        disable_upx: false,
        ..Default::default()
    };

    let old_report = cleave::analyze_file(old_path, &options)
        .context(format!("cleave analysis failed for old path: {:?}", old_path))?;

    let new_report = cleave::analyze_file(new_path, &options)
        .context(format!("cleave analysis failed for new path: {:?}", new_path))?;

    // Extract features
    let extractor = FeatureExtractor::load_default().unwrap_or_else(|e| {
        eprintln!("Warning: Could not load vocabulary: {}. Using defaults.", e);
        FeatureExtractor::new()
    });
    let old_features = extractor.extract_with_path(&old_report, Some(old_path));
    let new_features = extractor.extract_with_path(&new_report, Some(new_path));

    // Classify both versions using appropriate models
    let old_result = classify_features(old_path, old_features.as_slice(), threshold)?;
    let new_result = classify_features(new_path, new_features.as_slice(), threshold)?;

    // Find capability changes from cleave's diff report
    let new_capabilities: Vec<String> = diff_report
        .modified_analysis
        .iter()
        .flat_map(|ma| ma.new_capabilities.iter().map(|f| f.id.clone()))
        .collect();
    let removed_capabilities: Vec<String> = diff_report
        .modified_analysis
        .iter()
        .flat_map(|ma| ma.removed_capabilities.iter().map(|f| f.id.clone()))
        .collect();

    // Calculate feature deltas (top changes)
    let mut feature_deltas: Vec<(String, f32, f32)> = Vec::new();
    let min_len = old_features.len().min(new_features.len());
    for i in 0..min_len {
        let old_val = old_features.values[i];
        let new_val = new_features.values[i];
        let delta = (new_val - old_val).abs();
        if delta > 0.01 {
            feature_deltas.push((old_features.names[i].clone(), old_val, new_val));
        }
    }
    feature_deltas.sort_by(|a, b| {
        let delta_a = (a.2 - a.1).abs();
        let delta_b = (b.2 - b.1).abs();
        delta_b.partial_cmp(&delta_a).unwrap()
    });
    feature_deltas.truncate(15);

    // Assess risk level
    let (risk_level, risk_source) = assess_risk(
        old_result.as_ref(),
        new_result.as_ref(),
        &new_capabilities,
        &old_report,
        &new_report,
    );

    let diff_result = DiffResult {
        old_classification: old_result,
        new_classification: new_result,
        risk_level,
        risk_source,
        new_capabilities,
        removed_capabilities,
        feature_deltas,
    };

    match format {
        OutputFormat::Json => {
            output::print_diff_json(old_path, new_path, &diff_result)?;
        }
        OutputFormat::Terminal => {
            output::print_diff_terminal(old_path, new_path, &diff_result)?;
        }
    }

    Ok(())
}

/// Assess supply chain risk based on classification and capability changes
/// Returns (RiskLevel, RiskSource) so we can show what triggered the assessment
fn assess_risk(
    old_result: Option<&crate::classifier::ClassificationResult>,
    new_result: Option<&crate::classifier::ClassificationResult>,
    new_capabilities: &[String],
    old_report: &cleave::AnalysisReport,
    new_report: &cleave::AnalysisReport,
) -> (RiskLevel, RiskSource) {
    // Check for classification change to hostile
    if let (Some(old), Some(new)) = (old_result, new_result) {
        if new.classification == Classification::Hostile
            && old.classification != Classification::Hostile
        {
            return (RiskLevel::Critical, RiskSource::FileClassification);
        }
        if new.classification == Classification::Suspicious
            && old.classification == Classification::Benign
            && new.probability > 0.6
        {
            return (RiskLevel::High, RiskSource::FileClassification);
        }
    }

    // Check for new hostile findings (strongest signal from cleave)
    let old_hostile: std::collections::HashSet<_> = old_report
        .findings
        .iter()
        .filter(|f| f.crit == cleave::Criticality::Hostile)
        .map(|f| &f.id)
        .collect();
    let new_hostile_count = new_report
        .findings
        .iter()
        .filter(|f| f.crit == cleave::Criticality::Hostile && !old_hostile.contains(&f.id))
        .count();

    if new_hostile_count > 0 {
        return (RiskLevel::Critical, RiskSource::NewHostileFindings(new_hostile_count));
    }

    // Check for new suspicious findings
    let old_suspicious: std::collections::HashSet<_> = old_report
        .findings
        .iter()
        .filter(|f| f.crit == cleave::Criticality::Suspicious)
        .map(|f| &f.id)
        .collect();
    let new_suspicious_count = new_report
        .findings
        .iter()
        .filter(|f| f.crit == cleave::Criticality::Suspicious && !old_suspicious.contains(&f.id))
        .count();

    if new_suspicious_count >= 3 {
        return (RiskLevel::Critical, RiskSource::NewSuspiciousFindings(new_suspicious_count));
    }
    if new_suspicious_count >= 1 {
        return (RiskLevel::High, RiskSource::NewSuspiciousFindings(new_suspicious_count));
    }

    // Check for huge function injection (xz went from 0 to 4 huge functions)
    let old_huge_funcs = old_report
        .metrics
        .as_ref()
        .and_then(|m| m.binary.as_ref())
        .map(|b| b.huge_functions)
        .unwrap_or(0);
    let new_huge_funcs = new_report
        .metrics
        .as_ref()
        .and_then(|m| m.binary.as_ref())
        .map(|b| b.huge_functions)
        .unwrap_or(0);

    if new_huge_funcs > old_huge_funcs && new_huge_funcs >= 2 {
        return (
            RiskLevel::Critical,
            RiskSource::BinaryAnomaly(format!("huge_functions: {} → {}", old_huge_funcs, new_huge_funcs)),
        );
    }
    if new_huge_funcs > old_huge_funcs {
        return (
            RiskLevel::High,
            RiskSource::BinaryAnomaly(format!("huge_functions: {} → {}", old_huge_funcs, new_huge_funcs)),
        );
    }

    // Check for function size anomaly (xz went from avg 681 to 2360 bytes)
    let old_avg_func_size = old_report
        .metrics
        .as_ref()
        .and_then(|m| m.binary.as_ref())
        .map(|b| b.avg_function_size)
        .unwrap_or(0.0);
    let new_avg_func_size = new_report
        .metrics
        .as_ref()
        .and_then(|m| m.binary.as_ref())
        .map(|b| b.avg_function_size)
        .unwrap_or(0.0);

    if old_avg_func_size > 100.0 && new_avg_func_size > old_avg_func_size * 2.0 {
        return (
            RiskLevel::High,
            RiskSource::BinaryAnomaly(format!(
                "avg_function_size: {:.0} → {:.0} ({}x increase)",
                old_avg_func_size,
                new_avg_func_size,
                new_avg_func_size / old_avg_func_size
            )),
        );
    }

    // Check for dangerous new capabilities
    let dangerous_prefixes = [
        "exec/", "c2/", "credential/", "exfil/", "anti-analysis/", "persist/",
    ];

    let dangerous_new_caps = new_capabilities
        .iter()
        .filter(|c| dangerous_prefixes.iter().any(|p| c.starts_with(p)))
        .count();

    if dangerous_new_caps >= 3 {
        return (RiskLevel::Critical, RiskSource::DangerousCapabilities(dangerous_new_caps));
    }
    if dangerous_new_caps >= 1 {
        return (RiskLevel::High, RiskSource::DangerousCapabilities(dangerous_new_caps));
    }

    // Check for obfuscation increase
    let old_obf = old_report
        .source_code_metrics
        .as_ref()
        .map(|m| m.obfuscation_indicators)
        .unwrap_or(0);
    let new_obf = new_report
        .source_code_metrics
        .as_ref()
        .map(|m| m.obfuscation_indicators)
        .unwrap_or(0);

    if new_obf > old_obf + 5 {
        return (
            RiskLevel::High,
            RiskSource::BinaryAnomaly(format!("obfuscation: {} → {}", old_obf, new_obf)),
        );
    }

    // Check for significant entropy increase
    let old_entropy = old_report
        .metrics
        .as_ref()
        .and_then(|m| m.text.as_ref())
        .map(|t| t.char_entropy)
        .unwrap_or(0.0);
    let new_entropy = new_report
        .metrics
        .as_ref()
        .and_then(|m| m.text.as_ref())
        .map(|t| t.char_entropy)
        .unwrap_or(0.0);

    if new_entropy > old_entropy + 1.0 {
        return (
            RiskLevel::Medium,
            RiskSource::BinaryAnomaly(format!("entropy: {:.2} → {:.2}", old_entropy, new_entropy)),
        );
    }

    // Any new capabilities at all is at least medium
    if !new_capabilities.is_empty() {
        return (RiskLevel::Medium, RiskSource::DangerousCapabilities(new_capabilities.len()));
    }

    (RiskLevel::Low, RiskSource::None)
}
