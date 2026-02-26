//! Output formatting for terminal and JSON.

use crate::classifier::{Classification, ClassificationResult};
use crate::diff::{DiffResult, RiskLevel, RiskSource};
use crate::features::FeatureVector;
use anyhow::Result;
use colored::Colorize;
use cleave::types::{AnalysisReport, Criticality};
use std::path::Path;

/// Print scan results to terminal
pub fn print_scan_terminal(
    path: &Path,
    report: &AnalysisReport,
    result: Option<&ClassificationResult>,
) -> Result<()> {
    println!("litmus v{}\n", env!("CARGO_PKG_VERSION"));
    println!("File: {}", path.display());
    println!("Type: {}", report.target.file_type);
    println!("Size: {} bytes", report.target.size_bytes);
    println!();

    // Classification result
    if let Some(r) = result {
        let class_str = match r.classification {
            Classification::Hostile => "HOSTILE".red().bold(),
            Classification::Suspicious => "SUSPICIOUS".yellow().bold(),
            Classification::Benign => "BENIGN".green().bold(),
        };

        println!("Classification: {}", class_str);
        println!(
            "Probability: {:.2} (threshold: {:.2})",
            r.probability, r.hostile_threshold
        );
        println!();
    } else {
        println!("Classification: {} (no model loaded)", "UNKNOWN".dimmed());
        println!();
    }

    // Key capabilities from cleave
    let hostile: Vec<_> = report
        .findings
        .iter()
        .filter(|f| f.crit == Criticality::Hostile)
        .collect();
    let suspicious: Vec<_> = report
        .findings
        .iter()
        .filter(|f| f.crit == Criticality::Suspicious)
        .collect();
    let notable: Vec<_> = report
        .findings
        .iter()
        .filter(|f| f.crit == Criticality::Notable)
        .collect();

    if !hostile.is_empty() || !suspicious.is_empty() || !notable.is_empty() {
        println!("Key Capabilities Detected:");

        for f in hostile.iter().take(5) {
            println!("  {} {}", "[HOSTILE]".red(), f.id);
        }
        for f in suspicious.iter().take(5) {
            println!("  {} {}", "[Suspicious]".yellow(), f.id);
        }
        for f in notable.iter().take(5) {
            println!("  {} {}", "[Notable]".blue(), f.id);
        }

        let total = hostile.len() + suspicious.len() + notable.len();
        if total > 15 {
            println!("  ... and {} more", total - 15);
        }
        println!();
    }

    if result.is_some() {
        println!(
            "{}",
            "Run with --explain for detailed SHAP analysis (requires Python)".dimmed()
        );
    }

    Ok(())
}

/// Print scan results as JSON
pub fn print_scan_json(
    path: &Path,
    report: &AnalysisReport,
    result: Option<&ClassificationResult>,
    features: &FeatureVector,
) -> Result<()> {
    let output = serde_json::json!({
        "path": path.display().to_string(),
        "file_type": report.target.file_type,
        "size_bytes": report.target.size_bytes,
        "sha256": report.target.sha256,
        "classification": result.map(|r| match r.classification {
            Classification::Hostile => "hostile",
            Classification::Suspicious => "suspicious",
            Classification::Benign => "benign",
        }),
        "probability": result.map(|r| r.probability),
        "hostile_threshold": result.map(|r| r.hostile_threshold),
        "suspicious_threshold": result.map(|r| r.suspicious_threshold),
        "findings": report.findings.iter().map(|f| {
            serde_json::json!({
                "id": f.id,
                "crit": format!("{:?}", f.crit),
                "desc": f.desc,
            })
        }).collect::<Vec<_>>(),
        "feature_count": features.len(),
        "cleave": report,
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

/// Print diff results to terminal
pub fn print_diff_terminal(old_path: &Path, new_path: &Path, result: &DiffResult) -> Result<()> {
    println!("litmus diff - Supply Chain Analysis\n");
    println!(
        "Comparing: {} → {}",
        old_path.display(),
        new_path.display()
    );
    println!();

    // Classification change
    if let (Some(old), Some(new)) = (&result.old_classification, &result.new_classification) {
        let old_str = format_classification(old.classification, old.probability);
        let new_str = format_classification(new.classification, new.probability);
        println!("Classification Change: {} → {}", old_str, new_str);
    }

    // Risk level with source
    let risk_str = match result.risk_level {
        RiskLevel::Critical => "CRITICAL".red().bold(),
        RiskLevel::High => "HIGH".red(),
        RiskLevel::Medium => "MEDIUM".yellow(),
        RiskLevel::Low => "LOW".green(),
    };
    let source_str = format_risk_source(&result.risk_source);
    println!("Risk Level: {} ({})", risk_str, source_str);
    println!();

    // New capabilities
    if !result.new_capabilities.is_empty() {
        println!("New Capabilities:");
        for cap in &result.new_capabilities {
            println!("  {} {}", "+".green(), cap);
        }
        println!();
    }

    // Removed capabilities
    if !result.removed_capabilities.is_empty() {
        println!("Removed Capabilities:");
        for cap in &result.removed_capabilities {
            println!("  {} {}", "-".red(), cap);
        }
        println!();
    }

    // Feature deltas
    if !result.feature_deltas.is_empty() {
        println!("Feature Delta (top changes):");
        for (name, old_val, new_val) in &result.feature_deltas {
            println!("  {}: {:.2} → {:.2}", name, old_val, new_val);
        }
        println!();
    }

    // Recommendation
    match result.risk_level {
        RiskLevel::Critical => {
            println!(
                "{}",
                "Recommendation: DO NOT UPGRADE - Likely supply chain compromise"
                    .red()
                    .bold()
            );
        }
        RiskLevel::High => {
            println!(
                "{}",
                "Recommendation: Manual review required before upgrade"
                    .yellow()
                    .bold()
            );
        }
        RiskLevel::Medium => {
            println!(
                "{}",
                "Recommendation: Review new capabilities before upgrade".yellow()
            );
        }
        RiskLevel::Low => {
            println!("{}", "Recommendation: Changes appear safe".green());
        }
    }

    Ok(())
}

/// Print diff results as JSON
pub fn print_diff_json(old_path: &Path, new_path: &Path, result: &DiffResult) -> Result<()> {
    let output = serde_json::json!({
        "old_path": old_path.display().to_string(),
        "new_path": new_path.display().to_string(),
        "old_classification": result.old_classification.as_ref().map(|r| {
            serde_json::json!({
                "class": format!("{:?}", r.classification),
                "probability": r.probability,
            })
        }),
        "new_classification": result.new_classification.as_ref().map(|r| {
            serde_json::json!({
                "class": format!("{:?}", r.classification),
                "probability": r.probability,
            })
        }),
        "risk_level": format!("{:?}", result.risk_level),
        "new_capabilities": result.new_capabilities,
        "removed_capabilities": result.removed_capabilities,
        "feature_deltas": result.feature_deltas.iter().map(|(name, old, new)| {
            serde_json::json!({
                "name": name,
                "old_value": old,
                "new_value": new,
            })
        }).collect::<Vec<_>>(),
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

/// Format classification with color
fn format_classification(class: Classification, prob: f32) -> String {
    let label = match class {
        Classification::Hostile => "HOSTILE".red().to_string(),
        Classification::Suspicious => "SUSPICIOUS".yellow().to_string(),
        Classification::Benign => "BENIGN".green().to_string(),
    };
    format!("{} ({:.2})", label, prob)
}

/// Format risk source for display
fn format_risk_source(source: &RiskSource) -> String {
    match source {
        RiskSource::FileClassification => "file ML classification".to_string(),
        RiskSource::NewHostileFindings(n) => format!("{} new hostile findings", n),
        RiskSource::NewSuspiciousFindings(n) => format!("{} new suspicious findings", n),
        RiskSource::BinaryAnomaly(detail) => format!("binary anomaly: {}", detail),
        RiskSource::DangerousCapabilities(n) => format!("{} dangerous capabilities", n),
        RiskSource::None => "no significant changes".to_string(),
    }
}
