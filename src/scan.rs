//! Scan command implementation - classify files as benign/suspicious/hostile.

use crate::classifier::{Classifier, ClassificationResult};
use crate::features::FeatureExtractor;
use crate::hybrid_model::HybridClassifier;
use crate::multi_model::{ModelRegistry, detect_filetype};
use crate::output;
use crate::OutputFormat;
use anyhow::{Context, Result};
use cleave::types::{AnalysisReport, Criticality};
use std::path::Path;
use std::process::Command;

/// Classify features using either hybrid, multi-model registry, or single model
fn classify_features(path: &Path, features: &[f32], threshold: f32) -> Result<Option<ClassificationResult>> {
    // Try hybrid classifier first (intelligently routes between per-type and single models)
    match HybridClassifier::load_default() {
        Ok(hybrid) => {
            let mut hybrid = hybrid.with_hostile_threshold(threshold);

            // Print stats if in verbose mode
            if std::env::var("LITMUS_VERBOSE").is_ok() {
                eprintln!("{}", hybrid.stats());
            }

            match hybrid.classify(path, features) {
                Ok(result) => return Ok(Some(result)),
                Err(e) => {
                    eprintln!("Warning: Hybrid classifier failed: {}", e);
                    eprintln!("Trying fallback methods...");
                }
            }
        }
        Err(_) => {
            // Hybrid not available, try older approaches
        }
    }

    // Fall back to multi-model registry
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
                    eprintln!("Falling back to single model");
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
        Err(e) => {
            eprintln!("Warning: Could not load ML model: {}", e);
            eprintln!("Showing cleave analysis without ML classification.\n");
            Ok(None)
        }
    }
}

/// Parse criticality level from string
fn parse_criticality(s: &str) -> Result<Criticality> {
    match s.to_lowercase().as_str() {
        "hostile" => Ok(Criticality::Hostile),
        "suspicious" => Ok(Criticality::Suspicious),
        "notable" => Ok(Criticality::Notable),
        "baseline" => Ok(Criticality::Baseline),
        "component" => Ok(Criticality::Component),
        "filtered" => Ok(Criticality::Filtered),
        _ => anyhow::bail!("Invalid criticality level: {}. Valid values: hostile, suspicious, notable, baseline, component, filtered", s),
    }
}

/// Get the highest criticality level from findings in this report
fn report_criticality(report: &AnalysisReport) -> Option<Criticality> {
    report.findings.iter().map(|f| &f.crit).max().copied()
}

/// Get the highest criticality level across all files in the report
/// Uses v2 flat files array which includes archive contents
fn highest_criticality(report: &AnalysisReport) -> Option<Criticality> {
    // Check top-level findings
    let mut max_crit = report_criticality(report);

    // Check all files in v2 array (includes archive contents)
    for file in &report.files {
        if let Some(file_max) = file.findings.iter().map(|f| f.crit).max() {
            max_crit = Some(match max_crit {
                Some(current) => current.max(file_max),
                None => file_max,
            });
        }
    }

    max_crit
}

/// Scan a single file and check error-if condition
fn scan_single_file(
    path: &Path,
    options: &cleave::AnalysisOptions,
    explain: bool,
    threshold: f32,
    format: OutputFormat,
    error_crits: &[Criticality],
) -> Result<bool> {
    let report = cleave::analyze_file(path, options)
        .context(format!("cleave analysis failed for {:?}", path))?;

    // Extract features (pass path for byte histogram)
    let extractor = FeatureExtractor::load_default().unwrap_or_else(|e| {
        eprintln!("Warning: Could not load vocabulary: {}. Using defaults.", e);
        FeatureExtractor::new()
    });
    let features = extractor.extract_with_path(&report, Some(path));

    // Try to classify using either multi-model or single model
    let result = classify_features(path, features.as_slice(), threshold)?;

    // Output results
    match format {
        OutputFormat::Json => {
            output::print_scan_json(path, &report, result.as_ref(), &features)?;
        }
        OutputFormat::Terminal => {
            output::print_scan_terminal(path, &report, result.as_ref())?;
        }
    }

    // Run SHAP explanation if requested
    if explain {
        if result.is_some() {
            run_shap_explanation(&features)?;
        } else {
            eprintln!("\nCannot run SHAP explanation without loaded model.");
        }
    }

    // Check if highest criticality matches error-if list
    // This recursively checks sub_reports (files within archives)
    if !error_crits.is_empty() {
        if let Some(highest) = highest_criticality(&report) {
            if error_crits.contains(&highest) {
                return Ok(true); // Matches error condition
            }
        }
    }

    Ok(false) // Does not match error condition
}

/// Run the scan command
pub fn run(path: &Path, explain: bool, threshold: f32, format: OutputFormat, error_if: &[String]) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Path does not exist: {:?}", path);
    }

    // Parse error-if criticality levels
    let error_crits: Vec<Criticality> = error_if
        .iter()
        .map(|s| parse_criticality(s))
        .collect::<Result<Vec<_>>>()?;

    // Analyze with cleave (handles archives recursively via sub_reports)
    let options = cleave::AnalysisOptions {
        disable_yara: false,
        disable_radare2: true, // Skip radare2 for speed
        disable_upx: false,
        ..Default::default()
    };

    // Handle directories by walking recursively (like cleave scan does)
    if path.is_dir() {
        use walkdir::WalkDir;

        let mut all_files = Vec::new();
        for entry in WalkDir::new(path)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                let file_name = e.file_name().to_string_lossy();
                !file_name.starts_with(".git")
            })
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                all_files.push(entry.path().to_path_buf());
            }
        }

        eprintln!("Found {} files to scan\n", all_files.len());

        let mut error_files = Vec::new();
        for file_path in all_files {
            match scan_single_file(&file_path, &options, explain, threshold, format, &error_crits) {
                Ok(true) => {
                    error_files.push(file_path.display().to_string());
                }
                Ok(false) => {}
                Err(e) => {
                    eprintln!("Error scanning {}: {}", file_path.display(), e);
                }
            }
        }

        // If any files matched error-if condition, exit with error
        if !error_files.is_empty() {
            let mut msg = format!("Error: {} file(s) match --error-if condition:\n", error_files.len());
            for file in error_files.iter().take(10) {
                msg.push_str(&format!("  - {}\n", file));
            }
            if error_files.len() > 10 {
                msg.push_str(&format!("  ... and {} more files\n", error_files.len() - 10));
            }
            anyhow::bail!(msg);
        }

        return Ok(());
    }

    // Single file
    let matches_error = scan_single_file(path, &options, explain, threshold, format, &error_crits)?;

    if matches_error {
        anyhow::bail!(
            "Error: highest criticality level matches --error-if condition"
        );
    }

    Ok(())
}

/// Find Python interpreter (prefer venv)
fn find_python() -> String {
    // Check for venv python in training directory
    let venv_python = std::path::PathBuf::from("training/.venv/bin/python");
    if venv_python.exists() {
        return venv_python.display().to_string();
    }
    // Fall back to system python
    "python3".to_string()
}

/// Shell out to Python for SHAP explanations
fn run_shap_explanation(features: &crate::features::FeatureVector) -> Result<()> {
    // Find explain.py
    let explain_script = find_explain_script()?;

    // Find model files
    let model_path = find_model_path("litmus_v1.json")?; // XGBoost native format for SHAP
    let feature_names_path = find_model_path("feature_names.json")?;

    // Serialize features to JSON
    let features_json = serde_json::to_string(&features.values)?;

    // Run Python script with venv if available
    let python = find_python();
    let output = Command::new(&python)
        .arg(&explain_script)
        .arg(&features_json)
        .arg(&model_path)
        .arg(&feature_names_path)
        .output()
        .context("Failed to run SHAP explanation script. Is Python installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("SHAP explanation failed: {}", stderr);
    }

    // Parse and display SHAP output
    let shap_output: serde_json::Value = serde_json::from_slice(&output.stdout)
        .context("Failed to parse SHAP output")?;

    println!("\nTop Contributing Features (SHAP):");
    if let Some(top_features) = shap_output.get("top_features").and_then(|v| v.as_array()) {
        for (i, feature) in top_features.iter().enumerate() {
            let name = feature.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            let value = feature.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let shap = feature.get("shap").and_then(|v| v.as_f64()).unwrap_or(0.0);

            let sign = if shap >= 0.0 { "+" } else { "" };
            println!(
                "  {:2}. {:30} {}{:.3}  ({:.2})",
                i + 1,
                name,
                sign,
                shap,
                value
            );
        }
    }

    if let Some(base) = shap_output.get("base_value").and_then(|v| v.as_f64()) {
        println!("\nBase value: {:.3} (prior probability)", base);
    }

    Ok(())
}

/// Find the explain.py script
fn find_explain_script() -> Result<std::path::PathBuf> {
    // Check LITMUS_TRAINING_PATH
    if let Ok(path) = std::env::var("LITMUS_TRAINING_PATH") {
        let script = std::path::PathBuf::from(path).join("explain.py");
        if script.exists() {
            return Ok(script);
        }
    }

    // Check ~/.litmus/training/explain.py
    if let Some(home) = dirs::home_dir() {
        let script = home.join(".litmus").join("training").join("explain.py");
        if script.exists() {
            return Ok(script);
        }
    }

    // Check ./training/explain.py
    let script = std::path::PathBuf::from("training/explain.py");
    if script.exists() {
        return Ok(script);
    }

    anyhow::bail!(
        "Could not find explain.py. Set LITMUS_TRAINING_PATH or place in ~/.litmus/training/"
    )
}

/// Find a model file
fn find_model_path(filename: &str) -> Result<std::path::PathBuf> {
    // Check LITMUS_MODEL_PATH directory
    if let Ok(path) = std::env::var("LITMUS_MODEL_PATH") {
        let model_dir = std::path::Path::new(&path).parent().unwrap_or(std::path::Path::new("."));
        let file = model_dir.join(filename);
        if file.exists() {
            return Ok(file);
        }
    }

    // Check ~/.litmus/models/
    if let Some(home) = dirs::home_dir() {
        let file = home.join(".litmus").join("models").join(filename);
        if file.exists() {
            return Ok(file);
        }
    }

    // Check ./models/
    let file = std::path::PathBuf::from("models").join(filename);
    if file.exists() {
        return Ok(file);
    }

    anyhow::bail!("Could not find model file: {}", filename)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cleave::types::{Finding, TargetInfo};

    fn create_finding(crit: Criticality, id: &str) -> Finding {
        Finding {
            id: id.to_string(),
            kind: cleave::types::FindingKind::Capability,
            desc: "test finding".to_string(),
            conf: 1.0,
            crit,
            mbc: None,
            attack: None,
            trait_refs: vec![],
            evidence: vec![],
            match_count: 0,
            source_file: None,
        }
    }

    fn create_test_report(findings: Vec<(Criticality, &str)>) -> AnalysisReport {
        let mut report = AnalysisReport::new(TargetInfo {
            path: "test.tar".to_string(),
            file_type: "archive".to_string(),
            size_bytes: 1024,
            sha256: "test".to_string(),
            architectures: None,
        });

        report.findings = findings
            .into_iter()
            .map(|(crit, id)| create_finding(crit, id))
            .collect();

        report
    }

    #[test]
    fn test_highest_criticality_no_findings() {
        let report = create_test_report(vec![]);
        assert_eq!(highest_criticality(&report), None);
    }

    #[test]
    fn test_highest_criticality_single_finding() {
        let report = create_test_report(vec![(Criticality::Suspicious, "test/suspicious")]);
        assert_eq!(highest_criticality(&report), Some(Criticality::Suspicious));
    }

    #[test]
    fn test_highest_criticality_multiple_findings() {
        let report = create_test_report(vec![
            (Criticality::Notable, "test/notable"),
            (Criticality::Hostile, "test/hostile"),
            (Criticality::Suspicious, "test/suspicious"),
        ]);
        assert_eq!(highest_criticality(&report), Some(Criticality::Hostile));
    }

    #[test]
    fn test_highest_criticality_ordering() {
        // Verify criticality ordering: Hostile > Suspicious > Notable > Baseline > Component > Filtered
        assert!(Criticality::Hostile > Criticality::Suspicious);
        assert!(Criticality::Suspicious > Criticality::Notable);
        assert!(Criticality::Notable > Criticality::Baseline);
        assert!(Criticality::Baseline > Criticality::Filtered);
    }

    #[test]
    fn test_parse_criticality() {
        assert_eq!(parse_criticality("hostile").unwrap(), Criticality::Hostile);
        assert_eq!(parse_criticality("HOSTILE").unwrap(), Criticality::Hostile);
        assert_eq!(parse_criticality("Suspicious").unwrap(), Criticality::Suspicious);
        assert_eq!(parse_criticality("notable").unwrap(), Criticality::Notable);
        assert_eq!(parse_criticality("baseline").unwrap(), Criticality::Baseline);
        assert_eq!(parse_criticality("filtered").unwrap(), Criticality::Filtered);

        assert!(parse_criticality("invalid").is_err());
        assert!(parse_criticality("").is_err());
    }
}
