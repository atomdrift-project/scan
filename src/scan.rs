//! Recursive directory scanning and file classification.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Classification, Model};
use crate::OutputFormat;

pub use crate::explain::Reason;

/// Which classifications to display.
#[derive(Debug, Clone)]
pub struct DisplayFilter {
    pub hostile: bool,
    pub suspicious: bool,
    pub benign: bool,
}

impl DisplayFilter {
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
        Self { hostile: true, suspicious: true, benign: false }
    }
}

/// Scan configuration.
pub struct ScanConfig {
    pub model_dir: std::path::PathBuf,
    pub format: OutputFormat,
    pub threshold_suspicious: f32,
    pub threshold_hostile: f32,
    pub filter: DisplayFilter,
    pub verbose: bool,
    /// Warn when a single rule takes longer than this many milliseconds (default: 4000).
    pub slow_rule_ms: u64,
}

/// Summary of scan results.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScanSummary {
    pub total_files: u32,
    pub hostile: u32,
    pub suspicious: u32,
    pub benign: u32,
    pub errors: u32,
    pub duration_ms: u64,
}

/// Finding counts by criticality level from cleave.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct FindingCounts {
    pub hostile: u32,
    pub suspicious: u32,
    pub notable: u32,
    pub baseline: u32,
}

/// Result for a single scanned file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScanResult {
    pub path: String,
    pub classification: Classification,
    pub probability: f32,
    pub thresholds: Thresholds,
    pub finding_counts: FindingCounts,
    pub formula: String,
    pub reasons: Vec<Reason>,
    pub top_findings: Vec<TopFinding>,
    pub file_type: String,
    pub size_bytes: u64,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cleave: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted: Option<bool>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Thresholds {
    pub hostile: f32,
    pub suspicious: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TopFinding {
    pub id: String,
    pub crit: String,
    pub desc: String,
}

const SPINNER: &[char] = &['\u{2800}', '\u{2801}', '\u{2809}', '\u{2819}', '\u{281B}', '\u{281E}', '\u{2816}', '\u{2812}', '\u{2810}', '\u{2800}'];

/// Progress state shared between threads.
struct Progress {
    analyzed: AtomicU32,
    total: u32,
    start: Instant,
}

impl Progress {
    fn new(total: u32) -> Self {
        Self {
            analyzed: AtomicU32::new(0),
            total,
            start: Instant::now(),
        }
    }

    fn increment(&self) {
        self.analyzed.fetch_add(1, Ordering::Relaxed);
        self.draw();
    }

    /// Redraw progress line without incrementing (after printing a result).
    fn redraw(&self) {
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

    fn finish(&self) {
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
    let cleave_opts = cleave::AnalysisOptions { slow_rule_ms: config.slow_rule_ms, ..Default::default() };
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
                if config.filter.shows(&r.classification) {
                    emit_result(&r, config, false, &stdout);
                }
            }
            Err(e) => {
                log::warn!("error analyzing {}: {}", path.display(), e);
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
            let scan_result =
                result.and_then(|report| process_report(file_path, report, &ctx, &model, shap.as_ref(), config));
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
                    if config.filter.shows(&r.classification) {
                        emit_result(&r, config, prog.is_some(), &stdout);
                        if let Some(p) = prog {
                            p.redraw();
                        }
                    }
                }
                Err(e) => {
                    log::warn!("error analyzing {}: {}", file_path.display(), e);
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
        total_files: total_files.get().copied().unwrap_or(hostile + suspicious + benign + errors),
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
fn emit_result(r: &ScanResult, config: &ScanConfig, show_progress: bool, stdout: &Mutex<std::io::Stdout>) {
    match config.format {
        OutputFormat::Terminal => {
            crate::output::print_file_result_streaming(r, show_progress, config.verbose);
        }
        OutputFormat::Json => {
            if let Ok(line) = serde_json::to_string(r) {
                let mut out = stdout.lock().unwrap();
                let _ = writeln!(out, "{line}");
            }
        }
    }
}

/// Apply litmus model inference to a cleave report. Always returns a ScanResult
/// (even for benign); the caller decides whether to display it.
fn process_report(
    path: &Path,
    mut report: cleave::AnalysisReport,
    ctx: &ExtractContext,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
) -> Result<ScanResult> {
    // Compute formula before finalize (formula_from_report reads root-level findings).
    let formula = cleave::formula_from_report(&report);

    // Finalize moves all data into files[0], matching the structure the model was
    // trained on (collimator processes finalized cleave output). Without this,
    // primary_file() returns Null and the entire feature vector is zeros.
    report.finalize();

    let report_json = serde_json::to_value(&report).context("serializing cleave report")?;
    let mut features = ctx.extract(&report_json);
    model.spec.standardize(&mut features);
    let (probability, classification) = model.predict(&features)?;

    let finding_counts = count_findings_from_json(&report_json);

    // Only compute expensive extras for non-benign files.
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

    let cleave_json = match config.format {
        OutputFormat::Json => Some(report_json),
        OutputFormat::Terminal => None,
    };

    Ok(ScanResult {
        path: path.display().to_string(),
        classification,
        probability,
        thresholds: Thresholds {
            hostile: config.threshold_hostile,
            suspicious: config.threshold_suspicious,
        },
        finding_counts,
        formula,
        reasons,
        top_findings,
        file_type,
        size_bytes,
        sha256,
        cleave: cleave_json,
        pids: None,
        deleted: None,
    })
}

/// Count cleave findings by criticality level.
pub fn count_findings_from_json(report: &serde_json::Value) -> FindingCounts {
    let findings = report["findings"]
        .as_array()
        .or_else(|| {
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
    match crit {
        "filtered" => 0,
        "component" => 1,
        "baseline" => 2,
        "notable" => 3,
        "suspicious" => 4,
        "hostile" => 5,
        _ => 2,
    }
}
