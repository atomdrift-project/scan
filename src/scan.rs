//! Recursive directory scanning and file classification.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Classification, Model, ModelInfo, Thresholds};
use crate::OutputFormat;

pub use crate::explain::Reason;

/// Which classifications to display.
#[derive(Debug, Clone)]
pub struct DisplayFilter {
    /// Show hostile files.
    pub hostile: bool,
    /// Show suspicious files.
    pub suspicious: bool,
    /// Show benign files.
    pub benign: bool,
}

impl DisplayFilter {
    /// Returns true if the filter includes the given classification.
    #[must_use]
    pub fn shows(&self, c: &Classification) -> bool {
        match c {
            Classification::Hostile => self.hostile,
            Classification::Suspicious => self.suspicious,
            Classification::Benign => self.benign,
        }
    }
}

impl Default for DisplayFilter {
    fn default() -> Self {
        Self {
            hostile: true,
            suspicious: true,
            benign: false,
        }
    }
}

/// Scan configuration.
#[derive(Debug)]
pub struct ScanConfig {
    /// Directory containing model.onnx and feature_spec.json.
    pub model_dir: std::path::PathBuf,
    /// Output format.
    pub format: OutputFormat,
    /// Minimum probability to classify as suspicious.
    pub threshold_suspicious: f32,
    /// Minimum probability to classify as hostile.
    pub threshold_hostile: f32,
    /// Which classifications to display.
    pub filter: DisplayFilter,
    /// Warn when a single rule takes longer than this many milliseconds (default: 4000).
    pub slow_rule_ms: u64,
}

/// Summary of scan results.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScanSummary {
    /// Total number of files analyzed.
    pub total_files: u32,
    /// Number of hostile files.
    pub hostile: u32,
    /// Number of suspicious files.
    pub suspicious: u32,
    /// Number of benign files.
    pub benign: u32,
    /// Number of files that could not be analyzed.
    pub errors: u32,
    /// Wall-clock duration of the scan in milliseconds.
    pub duration_ms: u64,
}

/// Finding counts by criticality level from cleave.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct FindingCounts {
    /// Hostile-criticality findings.
    pub hostile: u32,
    /// Suspicious-criticality findings.
    pub suspicious: u32,
    /// Notable-criticality findings.
    pub notable: u32,
    /// Baseline-criticality findings.
    pub baseline: u32,
}

/// Result for a single scanned file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScanResult {
    /// Path to the analyzed file.
    pub path: String,
    /// Model classification outcome.
    pub classification: Classification,
    /// Raw malware probability from the model.
    pub probability: f32,
    /// Thresholds used for classification.
    pub thresholds: Thresholds,
    /// Breakdown of findings by criticality (terminal display only).
    #[serde(skip)]
    pub finding_counts: FindingCounts,
    /// Cleave formula string summarizing findings.
    pub formula: String,
    /// Top SHAP-based reasons for the classification.
    pub reasons: Vec<Reason>,
    /// Top findings from cleave at the relevant criticality level.
    pub top_findings: Vec<TopFinding>,
    /// Detected file type (e.g. "pe", "elf", "sh").
    pub file_type: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// SHA-256 hex digest of the file.
    pub sha256: String,
    /// Model metadata (JSON mode only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelInfo>,
    /// Full cleave report (JSON mode only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cleave: Option<serde_json::Value>,
    /// PIDs running this binary (process scan only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pids: Option<Vec<u32>>,
    /// Whether the binary was deleted from disk (process scan only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted: Option<bool>,
}

/// A notable finding from cleave at the highest relevant criticality.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TopFinding {
    /// Finding identifier (e.g. "objectives/evasion/process::injection").
    pub id: String,
    /// Criticality level (e.g. "hostile", "suspicious").
    pub crit: String,
    /// Human-readable description of the finding.
    pub desc: String,
}

pub(crate) const SPINNER: &[char] = &[
    '\u{2800}', '\u{2801}', '\u{2809}', '\u{2819}', '\u{281B}', '\u{281E}', '\u{2816}', '\u{2812}',
    '\u{2810}', '\u{2800}',
];

/// Progress state shared between threads.
pub(crate) struct Progress {
    analyzed: AtomicU32,
    total: u32,
    start: Instant,
}

impl Progress {
    pub(crate) fn new(total: u32) -> Self {
        Self {
            analyzed: AtomicU32::new(0),
            total,
            start: Instant::now(),
        }
    }

    pub(crate) fn increment(&self) {
        self.analyzed.fetch_add(1, Ordering::Relaxed);
        self.draw();
    }

    /// Redraw progress line without incrementing (after printing a result).
    pub(crate) fn redraw(&self) {
        self.draw();
    }

    fn draw(&self) {
        let done = self.analyzed.load(Ordering::Relaxed);
        let elapsed = self.start.elapsed().as_secs_f64();
        let rate = done as f64 / elapsed.max(0.001);
        let remaining = (self.total - done) as f64 / rate.max(0.001);

        let frame = SPINNER[done as usize % SPINNER.len()];
        let bar_w = 20;
        let filled = (done as usize * bar_w / self.total.max(1) as usize).min(bar_w);
        let bar: String = (0..bar_w)
            .map(|i| {
                if i < filled {
                    '\u{2501}' // ━
                } else if i == filled {
                    '\u{2578}' // ╸
                } else {
                    '\u{2500}' // ─
                }
            })
            .collect();

        let filled_str: String = bar.chars().take(filled + 1).collect();
        let dim_str: String = bar.chars().skip(filled + 1).collect();

        eprint!(
            "\r  \x1b[38;2;100;180;255m{frame}\x1b[0m \x1b[38;2;80;160;220m{filled_str}\x1b[38;2;50;50;50m{dim_str}\x1b[0m  \x1b[38;2;160;160;160m{done}/{total}  {rate:.0}/s  {eta}\x1b[0m   ",
            total = self.total,
            eta = format_eta(remaining),
        );
        let _ = std::io::stderr().flush();
    }

    pub(crate) fn finish(&self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let done = self.analyzed.load(Ordering::Relaxed);
        let rate = done as f64 / elapsed.max(0.001);
        eprint!(
            "\r\x1b[2K  \x1b[38;2;80;220;80m\u{2713}\x1b[0m  \x1b[38;2;160;160;160m{done} files in {elapsed:.1}s ({rate:.0}/s)\x1b[0m\n",
        );
        let _ = std::io::stderr().flush();
    }
}

fn format_eta(secs: f64) -> String {
    if secs < 1.0 {
        "<1s".to_string()
    } else if secs < 60.0 {
        format!("~{:.0}s", secs)
    } else {
        format!("~{}m{:.0}s", (secs / 60.0) as u32, secs % 60.0)
    }
}

/// Run a scan against a path (file or directory).
pub fn run(path: &Path, config: &ScanConfig) -> Result<ScanSummary> {
    let model = Model::load(
        &config.model_dir,
        crate::model::Thresholds {
            suspicious: config.threshold_suspicious,
            hostile: config.threshold_hostile,
        },
    )?;

    let shap = ShapImportance::load(&config.model_dir).ok();
    let ctx = ExtractContext::new(&model.spec);
    let cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms,
        ..Default::default()
    };
    let is_terminal = matches!(config.format, OutputFormat::Terminal);
    let scan_start = Instant::now();

    // Single-file path: handle directly without the directory streaming API.
    if path.is_file() {
        let result = analyze_single(path, &cleave_opts, &ctx, &model, shap.as_ref(), config);
        let (mut hostile, mut suspicious, mut benign, mut errors) = (0u32, 0u32, 0u32, 0u32);
        let stdout = Mutex::new(std::io::stdout());
        match result {
            Ok(r) => {
                match r.classification {
                    Classification::Hostile => hostile += 1,
                    Classification::Suspicious => suspicious += 1,
                    Classification::Benign => benign += 1,
                }
                if config.format == OutputFormat::Json || config.filter.shows(&r.classification) {
                    emit_result(&r, config, false, &stdout);
                }
            }
            Err(e) => {
                tracing::warn!("error analyzing {}: {}", path.display(), e);
                errors += 1;
            }
        }
        let summary = ScanSummary {
            total_files: 1,
            hostile,
            suspicious,
            benign,
            errors,
            duration_ms: scan_start.elapsed().as_millis() as u64,
        };
        if is_terminal {
            crate::output::print_summary(&summary);
        }
        return Ok(summary);
    }

    if !path.is_dir() {
        anyhow::bail!("path does not exist: {}", path.display());
    }

    // Directory scan: delegate walking and parallel analysis to cleave, which
    // loads CapabilityMapper and YARA once and streams results via callback.
    let total_files: OnceLock<u32> = OnceLock::new();
    let hostile_count = AtomicU32::new(0);
    let suspicious_count = AtomicU32::new(0);
    let benign_count = AtomicU32::new(0);
    let error_count = AtomicU32::new(0);
    let stdout = Mutex::new(std::io::stdout());
    let progress: OnceLock<Progress> = OnceLock::new();

    cleave::scan_directory(path, &cleave_opts, |event| match event {
        cleave::ScanEvent::Start { total } => {
            let _ = total_files.set(total as u32);
            if is_terminal && total > 1 {
                crate::output::print_header(path, total);
                let _ = progress.set(Progress::new(total as u32));
            }
        }
        cleave::ScanEvent::File {
            path: ref file_path,
            result,
        } => {
            let scan_result = result.and_then(|report| {
                process_report(file_path, report, &ctx, &model, shap.as_ref(), config)
            });
            let prog = progress.get();
            if let Some(p) = prog {
                p.increment();
            }
            match scan_result {
                Ok(r) => {
                    match r.classification {
                        Classification::Hostile => {
                            hostile_count.fetch_add(1, Ordering::Relaxed);
                        }
                        Classification::Suspicious => {
                            suspicious_count.fetch_add(1, Ordering::Relaxed);
                        }
                        Classification::Benign => {
                            benign_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    if config.format == OutputFormat::Json || config.filter.shows(&r.classification)
                    {
                        emit_result(&r, config, prog.is_some(), &stdout);
                        if let Some(p) = prog {
                            p.redraw();
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("error analyzing {}: {}", file_path.display(), e);
                    error_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    })?;

    if let Some(p) = progress.get() {
        p.finish();
    }

    let hostile = hostile_count.load(Ordering::Relaxed);
    let suspicious = suspicious_count.load(Ordering::Relaxed);
    let benign = benign_count.load(Ordering::Relaxed);
    let errors = error_count.load(Ordering::Relaxed);
    let summary = ScanSummary {
        total_files: total_files
            .get()
            .copied()
            .unwrap_or(hostile + suspicious + benign + errors),
        hostile,
        suspicious,
        benign,
        errors,
        duration_ms: scan_start.elapsed().as_millis() as u64,
    };

    if is_terminal {
        crate::output::print_summary(&summary);
    }

    Ok(summary)
}

/// Emit a scan result to the appropriate output channel.
fn emit_result(
    r: &ScanResult,
    config: &ScanConfig,
    show_progress: bool,
    stdout: &Mutex<std::io::Stdout>,
) {
    match config.format {
        OutputFormat::Terminal => {
            crate::output::print_file_result_streaming(r, show_progress);
        }
        OutputFormat::Json => match serde_json::to_string(r) {
            Ok(line) => {
                if let Ok(mut out) = stdout.lock() {
                    let _ = writeln!(out, "{line}");
                }
            }
            Err(e) => {
                tracing::error!(path = %r.path, "failed to serialize scan result: {e}");
            }
        },
    }
}

/// Intermediate classification result from the model pipeline.
/// Produced by `classify_report`, consumed when building a `ScanResult`.
pub(crate) struct ClassifiedReport {
    pub(crate) classification: Classification,
    pub(crate) probability: f32,
    pub(crate) finding_counts: FindingCounts,
    pub(crate) formula: String,
    pub(crate) reasons: Vec<Reason>,
    pub(crate) top_findings: Vec<TopFinding>,
    pub(crate) file_type: String,
    pub(crate) size_bytes: u64,
    pub(crate) sha256: String,
    pub(crate) report_json: serde_json::Value,
}

/// Run the full cleave-finalize + model inference pipeline on a report.
/// This is the single authoritative inference path used by scan, ps, and the server.
pub(crate) fn classify_report(
    label: &str,
    mut report: cleave::AnalysisReport,
    ctx: &ExtractContext,
    model: &Model,
    shap: Option<&ShapImportance>,
) -> Result<ClassifiedReport> {
    let formula = cleave::formula_from_report(&report);
    report.finalize();

    let report_json = serde_json::to_value(&report).context("serializing cleave report")?;
    let mut features = ctx.extract(&report_json);
    let nonzero = features.iter().filter(|&&v| v != 0.0).count();
    model.spec.standardize(&mut features);
    let (probability, classification) = model.predict(&features)?;

    let finding_counts = count_findings_from_json(&report_json);

    tracing::debug!(
        path = %label,
        classification = ?classification,
        probability = format!("{:.4}", probability),
        features_nonzero = nonzero,
        features_total = features.len(),
        findings_hostile = finding_counts.hostile,
        findings_suspicious = finding_counts.suspicious,
        findings_notable = finding_counts.notable,
        findings_baseline = finding_counts.baseline,
        formula = %formula,
        "classified file",
    );

    let (reasons, top_findings) = if classification != Classification::Benign {
        let r = shap
            .map(|s| s.explain(&features, &model.spec.feature_names, 5))
            .unwrap_or_default();
        let f = extract_top_findings_from_json(&report_json, &classification);
        (r, f)
    } else {
        (vec![], vec![])
    };

    let pf = report_json["files"]
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or(&report_json);
    let file_type = pf["file_type"].as_str().unwrap_or("unknown").to_string();
    let size_bytes = pf["size"].as_u64().unwrap_or(0);
    let sha256 = pf["sha256"].as_str().unwrap_or("").to_string();

    Ok(ClassifiedReport {
        classification,
        probability,
        finding_counts,
        formula,
        reasons,
        top_findings,
        file_type,
        size_bytes,
        sha256,
        report_json,
    })
}

/// Apply litmus model inference to a cleave report. Always returns a ScanResult
/// (even for benign); the caller decides whether to display it.
fn process_report(
    path: &Path,
    report: cleave::AnalysisReport,
    ctx: &ExtractContext,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
) -> Result<ScanResult> {
    let cr = classify_report(&path.display().to_string(), report, ctx, model, shap)?;
    let is_json = matches!(config.format, OutputFormat::Json);

    Ok(ScanResult {
        path: path.display().to_string(),
        classification: cr.classification,
        probability: cr.probability,
        thresholds: Thresholds {
            suspicious: config.threshold_suspicious,
            hostile: config.threshold_hostile,
        },
        finding_counts: cr.finding_counts,
        formula: cr.formula,
        reasons: cr.reasons,
        top_findings: cr.top_findings,
        file_type: cr.file_type,
        size_bytes: cr.size_bytes,
        sha256: cr.sha256,
        model: if is_json {
            Some(model.info.clone())
        } else {
            None
        },
        cleave: if is_json { Some(cr.report_json) } else { None },
        pids: None,
        deleted: None,
    })
}

/// Count cleave findings by criticality level.
#[must_use]
pub fn count_findings_from_json(report: &serde_json::Value) -> FindingCounts {
    let findings = report["findings"].as_array().or_else(|| {
        report["files"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|f| f["findings"].as_array())
    });

    let Some(findings) = findings else {
        return FindingCounts::default();
    };

    let mut counts = FindingCounts::default();
    for f in findings {
        match f["crit"].as_str().unwrap_or("baseline") {
            "hostile" => counts.hostile += 1,
            "suspicious" => counts.suspicious += 1,
            "notable" => counts.notable += 1,
            _ => counts.baseline += 1,
        }
    }
    counts
}

/// Analyze a single file end-to-end (cleave + litmus model).
fn analyze_single(
    path: &Path,
    cleave_opts: &cleave::AnalysisOptions,
    ctx: &ExtractContext,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
) -> Result<ScanResult> {
    let report = cleave::analyze_file(path, cleave_opts)
        .with_context(|| format!("cleave analysis of {}", path.display()))?;
    process_report(path, report, ctx, model, shap, config)
}

/// Extract top findings at the highest criticality level present.
#[must_use]
pub fn extract_top_findings_from_json(
    report: &serde_json::Value,
    classification: &Classification,
) -> Vec<TopFinding> {
    let findings = report["findings"]
        .as_array()
        .or_else(|| {
            report["files"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(|f| f["findings"].as_array())
        })
        .cloned()
        .unwrap_or_default();

    let min_crit = match classification {
        Classification::Hostile => "hostile",
        Classification::Suspicious => "suspicious",
        Classification::Benign => "baseline",
    };

    let mut relevant: Vec<TopFinding> = findings
        .iter()
        .filter(|f| {
            let crit = f["crit"].as_str().unwrap_or("baseline");
            crit_ordinal(crit) >= crit_ordinal(min_crit)
        })
        .map(|f| TopFinding {
            id: f["id"].as_str().unwrap_or("").to_string(),
            crit: f["crit"].as_str().unwrap_or("baseline").to_string(),
            desc: f["desc"].as_str().unwrap_or("").to_string(),
        })
        .collect();

    // Fall back to suspicious if no hostile findings.
    if relevant.is_empty() && min_crit == "hostile" {
        relevant = findings
            .iter()
            .filter(|f| crit_ordinal(f["crit"].as_str().unwrap_or("baseline")) >= 4)
            .map(|f| TopFinding {
                id: f["id"].as_str().unwrap_or("").to_string(),
                crit: f["crit"].as_str().unwrap_or("baseline").to_string(),
                desc: f["desc"].as_str().unwrap_or("").to_string(),
            })
            .collect();
    }

    // Deduplicate by base ID.
    let mut seen = std::collections::HashSet::new();
    relevant.retain(|f| {
        let base = f.id.split("::").next().unwrap_or(&f.id);
        seen.insert(base.to_string())
    });

    relevant.sort_by(|a, b| crit_ordinal(&b.crit).cmp(&crit_ordinal(&a.crit)));
    relevant.truncate(5);
    relevant
}

fn crit_ordinal(crit: &str) -> u32 {
    crate::features::crit_ordinal(crit)
}
