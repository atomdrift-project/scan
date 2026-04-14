//! Recursive file-system scanning and classification.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Classification, Model, Thresholds};
use crate::OutputFormat;

pub use crate::explain::Reason;

/// Terminal display policy for scan results.
///
/// This affects human-readable output only. JSON output still emits every
/// scanned file so downstream consumers receive a complete event stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayFilter {
    hostile: bool,
    suspicious: bool,
    benign: bool,
}

impl DisplayFilter {
    /// Create a filter with explicit inclusion flags for each classification.
    #[must_use]
    pub const fn new(hostile: bool, suspicious: bool, benign: bool) -> Self {
        Self {
            hostile,
            suspicious,
            benign,
        }
    }

    /// Include only hostile and suspicious files.
    #[must_use]
    pub const fn alerts_only() -> Self {
        Self::new(true, true, false)
    }

    /// Include every classification.
    #[must_use]
    pub const fn all() -> Self {
        Self::new(true, true, true)
    }

    /// Returns true if the filter includes the given classification.
    #[must_use]
    pub fn shows(&self, c: &Classification) -> bool {
        match c {
            Classification::Hostile => self.hostile,
            Classification::Suspicious => self.suspicious,
            Classification::Benign => self.benign,
        }
    }

    /// Returns whether hostile files are included.
    #[must_use]
    pub const fn shows_hostile(&self) -> bool {
        self.hostile
    }

    /// Returns whether suspicious files are included.
    #[must_use]
    pub const fn shows_suspicious(&self) -> bool {
        self.suspicious
    }

    /// Returns whether benign files are included.
    #[must_use]
    pub const fn shows_benign(&self) -> bool {
        self.benign
    }
}

impl Default for DisplayFilter {
    fn default() -> Self {
        Self::alerts_only()
    }
}

/// Immutable configuration for file-system and process scans.
///
/// Use [`ScanConfig::new`] so threshold invariants are validated before work
/// begins. After construction the value is read-only and can be shared freely.
#[derive(Debug)]
pub struct ScanConfig {
    model_dir: PathBuf,
    format: OutputFormat,
    thresholds: Option<Thresholds>,
    filter: DisplayFilter,
    slow_rule_ms: u64,
    scan_threads: Option<usize>,
    extra: bool,
}

impl ScanConfig {
    /// Create a scan configuration.
    ///
    /// `thresholds` may be `None` to use the model's recommended thresholds
    /// from `evaluation.json`, or `Some(t)` to override with explicit values.
    ///
    /// `slow_rule_ms` is advisory logging only; it does not cancel analysis.
    ///
    /// # Example
    /// ```
    /// use litmus::{DisplayFilter, OutputFormat, ScanConfig, Thresholds};
    ///
    /// let config = ScanConfig::new(
    ///     "/path/to/models",
    ///     OutputFormat::Terminal,
    ///     None,
    ///     DisplayFilter::alerts_only(),
    ///     4_000,
    ///     None,
    ///     false,
    /// )?;
    ///
    /// assert_eq!(config.format(), OutputFormat::Terminal);
    /// assert!(config.filter().shows_hostile());
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        model_dir: impl Into<PathBuf>,
        format: OutputFormat,
        thresholds: Option<Thresholds>,
        filter: DisplayFilter,
        slow_rule_ms: u64,
        scan_threads: Option<usize>,
        extra: bool,
    ) -> Result<Self> {
        if let Some(ref t) = thresholds {
            t.validate()
                .map_err(|error| anyhow::anyhow!("invalid thresholds: {error}"))?;
        }
        Ok(Self {
            model_dir: model_dir.into(),
            format,
            thresholds,
            filter,
            slow_rule_ms,
            scan_threads,
            extra,
        })
    }

    /// Directory containing `model.json` and `feature_spec.json`.
    #[must_use]
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Output format for emitted results.
    #[must_use]
    pub const fn format(&self) -> OutputFormat {
        self.format
    }

    /// Explicit threshold overrides, if any. `None` means use model defaults.
    #[must_use]
    pub const fn thresholds(&self) -> Option<Thresholds> {
        self.thresholds
    }

    /// Filter controlling which classifications are printed in terminal mode.
    #[must_use]
    pub const fn filter(&self) -> DisplayFilter {
        self.filter
    }

    /// Warn when a single rule exceeds this duration in milliseconds.
    #[must_use]
    pub const fn slow_rule_ms(&self) -> u64 {
        self.slow_rule_ms
    }

    /// Number of parallel cleave scan workers to use for directory scans.
    #[must_use]
    pub const fn scan_threads(&self) -> Option<usize> {
        self.scan_threads
    }

    /// Whether to show extra debug info (raw probability, SHAP values) in terminal output.
    #[must_use]
    pub const fn extra(&self) -> bool {
        self.extra
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn scan_config_rejects_invalid_thresholds() {
        let result = ScanConfig::new(
            "/tmp/models",
            OutputFormat::Terminal,
            Some(Thresholds {
                suspicious: 0.99,
                hostile: 0.50,
            }),
            DisplayFilter::alerts_only(),
            4_000,
            None,
            false,
        );

        assert!(result.is_err());
    }
}

/// Aggregate counters for a completed scan.
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

/// Classification result for a single analyzed file or executable.
///
/// In terminal mode only a subset of results may be shown, but in JSON mode
/// every scanned item is emitted as a `ScanResult`.
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// Schema version.
    pub v: &'static str,
    /// Model classification outcome.
    pub classification: Classification,
    /// Raw malware probability from the model.
    pub probability: f32,
    /// Thresholds.
    pub thresholds: Thresholds,
    /// Model version identifier (spec version, ABI version, model hash prefix).
    pub version: String,
    /// UTC timestamp of when this analysis was performed (RFC 3339).
    pub analyzed_at: String,
    /// Full cleave report (unmutated).
    pub cleave: Option<serde_json::Value>,
    /// PIDs running this binary (process scan only).
    pub pids: Option<Vec<u32>>,
    /// Whether the binary was deleted from disk (process scan only).
    pub deleted: Option<bool>,
    /// Display path (original filename or scanned path).
    pub path: String,
    /// Finding counts by severity level.
    pub finding_counts: FindingCounts,
    /// Molecular formula.
    pub formula: String,
    /// SHAP explanation reasons.
    pub reasons: Vec<Reason>,
    /// Top findings for display.
    pub top_findings: Vec<TopFinding>,
    /// Detected file type.
    pub file_type: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// SHA-256 hex digest.
    pub sha256: String,
    /// Per-file ML evaluations for archive members.
    pub embedded_files: Vec<EmbeddedFile>,
}

/// A representative cleave finding surfaced alongside a classification.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TopFinding {
    /// Finding identifier (e.g. "objectives/evasion/process::injection").
    pub id: String,
    /// Criticality ordinal (0=filtered .. 5=hostile).
    pub crit: u32,
    /// Human-readable description of the finding.
    pub desc: String,
}

/// A file embedded within an archive or self-extracting executable.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbeddedFile {
    /// Relative path within the archive (portion after "!!" delimiter).
    pub path: String,
    /// Detected file type.
    pub file_type: String,
    /// Model classification for this embedded file.
    pub classification: Classification,
    /// Raw model probability for this embedded file.
    pub probability: f32,
    /// Molecular formula for this embedded file.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub formula: String,
    /// Top findings for this embedded file.
    pub top_findings: Vec<TopFinding>,
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

/// Run a scan against a file or directory tree.
///
/// A file path is analyzed directly. A directory path is walked recursively via
/// `cleave`, with results streamed as they complete.
///
/// # Errors
/// Returns an error if the target path does not exist, model artifacts cannot
/// be loaded, or `cleave` analysis fails for the overall scan operation.
pub fn run(path: &Path, config: &ScanConfig) -> Result<ScanSummary> {
    let model = Model::load(config.model_dir(), config.thresholds())?;
    let model_version = model_version_string(model.info());

    let shap = ShapImportance::load(config.model_dir()).ok();
    let ctx = ExtractContext::new(model.spec());
    let cancellation = Arc::new(AtomicBool::new(false));
    let ctrlc_flag = Arc::clone(&cancellation);
    let _ = ctrlc::set_handler(move || {
        if ctrlc_flag.load(Ordering::Relaxed) {
            // Second ctrl-c: hard exit.
            std::process::exit(130);
        }
        eprintln!("\nInterrupted — finishing current file…");
        ctrlc_flag.store(true, Ordering::Relaxed);
    });
    let cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        scan_threads: config.scan_threads().unwrap_or(0),
        cancellation: Some(Arc::clone(&cancellation)),
        ..Default::default()
    };
    let is_terminal = matches!(config.format(), OutputFormat::Terminal);
    let scan_start = Instant::now();

    // Single-file path: handle directly without the directory streaming API.
    if path.is_file() {
        let result = analyze_single(
            path,
            &cleave_opts,
            &ctx,
            &model,
            &model_version,
            shap.as_ref(),
            config,
        );
        let (mut hostile, mut suspicious, mut benign, mut errors) = (0u32, 0u32, 0u32, 0u32);
        let stdout = Mutex::new(std::io::stdout());
        match result {
            Ok(r) => {
                match r.classification {
                    Classification::Hostile => hostile += 1,
                    Classification::Suspicious => suspicious += 1,
                    Classification::Benign => benign += 1,
                }
                if config.format() == OutputFormat::Json || config.filter().shows(&r.classification)
                {
                    emit_result(&r, config, false, &stdout);
                }
            }
            Err(e) => {
                let msg = crate::tools::enrich_error(&e)
                    .unwrap_or_else(|| format!("{e:#}"));
                tracing::warn!("error analyzing {}: {}", path.display(), msg);
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
                process_report(
                    file_path,
                    report,
                    &ctx,
                    &model,
                    &model_version,
                    shap.as_ref(),
                    config,
                    cleave_opts.cancellation.as_ref(),
                )
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
                    if config.format() == OutputFormat::Json
                        || config.filter().shows(&r.classification)
                    {
                        emit_result(&r, config, prog.is_some(), &stdout);
                        if let Some(p) = prog {
                            p.redraw();
                        }
                    }
                }
                Err(e) => {
                    let msg = crate::tools::enrich_error(&e)
                        .unwrap_or_else(|| format!("{e:#}"));
                    tracing::warn!("error analyzing {}: {}", file_path.display(), msg);
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
    match config.format() {
        OutputFormat::Terminal => {
            crate::output::print_file_result_streaming(r, show_progress, config.extra());
        }
        OutputFormat::Json => match serde_json::to_string(&r.to_envelope()) {
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
    pub(crate) embedded_files: Vec<EmbeddedFile>,
    pub(crate) report_json: serde_json::Value,
}

/// Run the full cleave-finalize + model inference pipeline on a report.
/// This is the single authoritative inference path used by scan, ps, and the server.
#[allow(clippy::needless_pass_by_value)] // Arc clones at call sites are negligible; ownership simplifies callers.
pub(crate) fn classify_report(
    label: &str,
    mut report: cleave::AnalysisReport,
    ctx: &ExtractContext,
    model: &Model,
    shap: Option<&ShapImportance>,
    cancellation: Option<&Arc<AtomicBool>>,
) -> Result<ClassifiedReport> {
    report.finalize();
    let compact = cleave::types::compact::compact_from_files(&report.files);
    let formula = compact
        .fs
        .first()
        .and_then(|file| file.f.clone())
        .unwrap_or_default();

    let report_json = serde_json::to_value(&compact).context("serializing cleave report")?;
    let mut raw_features = ctx.extract(&report_json);
    let nonzero = raw_features.iter().filter(|&&v| v != 0.0).count();
    let expected = model.spec().total_features();
    if raw_features.len() != expected {
        anyhow::bail!(
            "feature vector length mismatch: got {} expected {} — model/feature_spec out of sync",
            raw_features.len(),
            expected,
        );
    }
    let (probability, classification, reasons) = if let Some(shap) = shap {
        let mut features = raw_features.clone();
        model.spec().standardize(&mut features);
        let (probability, classification) = model.predict(&features)?;
        let reasons = shap.explain(&raw_features, model.spec().feature_names());
        (probability, classification, reasons)
    } else {
        model.spec().standardize(&mut raw_features);
        let (probability, classification) = model.predict(&raw_features)?;
        (probability, classification, Vec::new())
    };

    let finding_counts = count_findings_from_json(&report_json);

    tracing::debug!(
        path = %label,
        classification = ?classification,
        probability = format!("{:.4}", probability),
        features_nonzero = nonzero,
        features_total = expected,
        findings_hostile = finding_counts.hostile,
        findings_suspicious = finding_counts.suspicious,
        findings_notable = finding_counts.notable,
        findings_baseline = finding_counts.baseline,
        formula = %formula,
        "classified file",
    );

    let pf = report_json["fs"]
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or(&report_json);
    let file_type = pf["type"].as_str().unwrap_or("unknown").to_string();
    let size_bytes = pf["sz"].as_u64().unwrap_or(0);
    let sha256 = pf["sha"].as_str().unwrap_or("").to_string();

    // Extract embedded files (archive members at depth > 0), run each through
    // the model individually, and elevate the parent if any embedded file scores higher.
    // Limit to top 100 embedded entries to prevent resource exhaustion.
    let embedded_entries: Vec<&serde_json::Value> = report_json["fs"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|f| f["dp"].as_u64().unwrap_or(0) > 0)
        .take(100)
        .collect();

    let mut embedded_files: Vec<EmbeddedFile> = Vec::with_capacity(embedded_entries.len());
    let mut max_probability = probability;
    let mut max_classification = classification;

    for ef in &embedded_entries {
        if let Some(c) = cancellation {
            if c.load(Ordering::Relaxed) {
                anyhow::bail!("analysis cancelled during embedded file processing");
            }
        }

        let mut ef_features = ctx.extract_file(ef);
        model.spec().standardize(&mut ef_features);
        let (ef_prob, ef_class) = model
            .predict(&ef_features)
            .unwrap_or((0.0, Classification::Benign));

        let full_path = ef["path"].as_str().unwrap_or("");
        let rel_path = full_path
            .rsplit_once("!!")
            .map(|(_, r)| r)
            .unwrap_or(full_path)
            .to_string();

        let ef_top_findings: Vec<TopFinding> = ef["ts"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|ff| crit_ordinal(ff) >= 4)
            .take(3)
            .map(|ff| TopFinding {
                id: ff["i"].as_str().unwrap_or("").to_string(),
                crit: ff["l"].as_u64().unwrap_or(0) as u32,
                desc: ff["d"].as_str().unwrap_or("").to_string(),
            })
            .collect();

        tracing::debug!(
            parent = %label,
            embedded_path = %rel_path,
            probability = format!("{:.4}", ef_prob),
            classification = ?ef_class,
            "classified embedded file",
        );

        if ef_prob > max_probability {
            max_probability = ef_prob;
            max_classification = ef_class;
        }

        embedded_files.push(EmbeddedFile {
            path: rel_path,
            file_type: ef["type"].as_str().unwrap_or("unknown").to_string(),
            classification: ef_class,
            probability: ef_prob,
            formula: ef["f"].as_str().unwrap_or("").to_string(),
            top_findings: ef_top_findings,
        });
    }

    embedded_files.sort_by(|a, b| b.probability.total_cmp(&a.probability));
    embedded_files.truncate(10);

    // If an embedded file scored higher, elevate the parent classification.
    let (classification, probability) = if max_probability > probability {
        tracing::info!(
            path = %label,
            original_probability = format!("{:.4}", probability),
            elevated_probability = format!("{:.4}", max_probability),
            elevated_classification = ?max_classification,
            "elevated archive classification due to embedded file",
        );
        (max_classification, max_probability)
    } else {
        (classification, probability)
    };

    let top_findings = extract_top_findings_from_json(&report_json, &classification);

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
        embedded_files,
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
    model_version: &str,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
    cancellation: Option<&Arc<AtomicBool>>,
) -> Result<ScanResult> {
    let cr = classify_report(
        &path.display().to_string(),
        report,
        ctx,
        model,
        shap,
        cancellation,
    )?;
    let is_json = matches!(config.format(), OutputFormat::Json);

    // Include raw cleave report for JSON output (unmutated — ML scores go in the ml section).
    let cleave = if is_json { Some(cr.report_json) } else { None };

    let thresholds = model.thresholds();

    Ok(ScanResult {
        v: "4",
        classification: cr.classification,
        probability: cr.probability,
        thresholds,
        version: model_version.to_string(),
        analyzed_at: now_rfc3339(),
        cleave,
        pids: None,
        deleted: None,
        path: path.display().to_string(),
        finding_counts: cr.finding_counts,
        formula: cr.formula,
        reasons: cr.reasons,
        top_findings: cr.top_findings,
        file_type: cr.file_type,
        size_bytes: cr.size_bytes,
        sha256: cr.sha256,
        embedded_files: cr.embedded_files,
    })
}

/// Count cleave findings by criticality level from either a top-level report or
/// the primary file entry inside that report.
#[must_use]
pub fn count_findings_from_json(report: &serde_json::Value) -> FindingCounts {
    let findings = report["ts"].as_array().or_else(|| {
        report["fs"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|f| f["ts"].as_array())
    });

    let Some(findings) = findings else {
        return FindingCounts::default();
    };

    let mut counts = FindingCounts::default();
    for f in findings {
        match f["l"].as_u64().unwrap_or(0) {
            5 => counts.hostile += 1,
            4 => counts.suspicious += 1,
            3 => counts.notable += 1,
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
    model_version: &str,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
) -> Result<ScanResult> {
    let report = cleave::analyze_file(path, cleave_opts)
        .with_context(|| format!("cleave analysis of {}", path.display()))?;
    process_report(
        path,
        report,
        ctx,
        model,
        model_version,
        shap,
        config,
        cleave_opts.cancellation.as_ref(),
    )
}

/// Extract a small set of human-facing findings relevant to the classification.
#[must_use]
pub fn extract_top_findings_from_json(
    report: &serde_json::Value,
    classification: &Classification,
) -> Vec<TopFinding> {
    let findings = report["ts"]
        .as_array()
        .or_else(|| {
            report["fs"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(|f| f["ts"].as_array())
        })
        .cloned()
        .unwrap_or_default();

    let min_crit: u32 = match classification {
        Classification::Hostile => 5,
        Classification::Suspicious | Classification::Benign => 4,
    };

    let mut relevant: Vec<TopFinding> = findings
        .iter()
        .filter(|f| crit_ordinal(f) >= min_crit)
        .map(|f| TopFinding {
            id: f["i"].as_str().unwrap_or("").to_string(),
            crit: f["l"].as_u64().unwrap_or(0) as u32,
            desc: f["d"].as_str().unwrap_or("").to_string(),
        })
        .collect();

    // Fall back to suspicious if no hostile findings.
    if relevant.is_empty() && min_crit == 5 {
        relevant = findings
            .iter()
            .filter(|f| crit_ordinal(f) >= 4)
            .map(|f| TopFinding {
                id: f["i"].as_str().unwrap_or("").to_string(),
                crit: f["l"].as_u64().unwrap_or(0) as u32,
                desc: f["d"].as_str().unwrap_or("").to_string(),
            })
            .collect();
    }

    // Deduplicate by base ID.
    let mut seen = std::collections::HashSet::new();
    relevant.retain(|f| {
        let base = f.id.split("::").next().unwrap_or(&f.id);
        seen.insert(base.to_string())
    });

    relevant.sort_by(|a, b| b.crit.cmp(&a.crit));
    relevant.truncate(5);
    relevant
}

use crate::features::crit_ordinal;

/// Build a compact model version string from ModelInfo.
/// Format: "v{spec_version}.{abi_version}-{sha256_prefix}" or with commit if available.
pub(crate) fn model_version_string(info: &crate::model::ModelInfo) -> String {
    if info.sha256.is_empty() {
        return format!("v{}.{}", info.version, info.abi_version);
    }
    let sha_prefix = if info.sha256.len() >= 8 {
        &info.sha256[..8]
    } else {
        &info.sha256
    };
    match &info.commit {
        Some(commit) => format!(
            "v{}.{}-{}-{}",
            info.version, info.abi_version, sha_prefix, commit
        ),
        None => format!("v{}.{}-{}", info.version, info.abi_version, sha_prefix),
    }
}

/// Return the current time as an RFC 3339 string in UTC.
pub(crate) fn now_rfc3339() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Manual UTC breakdown — avoids external time crate dependency.
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    // Civil date from day count (algorithm from Howard Hinnant).
    let z = days as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Serialize Thresholds as [suspicious, hostile] array for v4 JSON.
fn serialize_thresholds<S>(t: &Thresholds, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::Serialize;
    [t.suspicious, t.hostile].serialize(serializer)
}

/// Build per-file ML classification entries for the `ml.fs` array.
///
/// Each entry contains `{id, class, prob}` keyed by the cleave `fs[].id` field.
/// The root file (dp=0) gets the parent classification; embedded files get their
/// individual scores matched by path suffix.
fn build_ml_fs(
    report_json: &serde_json::Value,
    classification: &Classification,
    probability: f32,
    embedded_files: &[EmbeddedFile],
) -> Vec<serde_json::Value> {
    let Some(fs) = report_json.get("fs").and_then(|v| v.as_array()) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(fs.len());
    for (idx, entry) in fs.iter().enumerate() {
        let id = entry
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(idx as u64);
        let depth = entry.get("dp").and_then(serde_json::Value::as_u64).unwrap_or(0);

        let (cls, prob) = if depth == 0 {
            (classification, probability)
        } else {
            let path = entry.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let suffix = path.rsplit_once("!!").map(|(_, r)| r).unwrap_or(path);
            embedded_files
                .iter()
                .find(|ef| ef.path == suffix)
                .map(|ef| (&ef.classification, ef.probability))
                .unwrap_or((classification, probability))
        };

        out.push(serde_json::json!({"id": id, "class": cls, "prob": prob}));
    }
    out
}

/// Top-level JSON envelope: `{"ml": {...}, "raw": {...}}`.
#[derive(Debug, serde::Serialize)]
pub struct ScanResultEnvelope {
    /// ML classification section.
    pub ml: MlSection,
    /// Raw cleave analysis report.
    pub raw: serde_json::Value,
}

/// The `ml` section of the response envelope.
#[derive(Debug, serde::Serialize)]
pub struct MlSection {
    v: &'static str,
    #[serde(rename = "class")]
    classification: Classification,
    #[serde(rename = "prob")]
    probability: f32,
    #[serde(serialize_with = "serialize_thresholds")]
    thresholds: Thresholds,
    version: String,
    analyzed_at: String,
    fs: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    deleted: Option<bool>,
}

impl ScanResult {
    /// Build the `{"ml": {...}, "raw": {...}}` envelope for JSON output.
    #[must_use]
    pub fn to_envelope(&self) -> ScanResultEnvelope {
        let raw = self.cleave.clone().unwrap_or(serde_json::json!({}));
        let ml_fs = build_ml_fs(
            &raw,
            &self.classification,
            self.probability,
            &self.embedded_files,
        );
        ScanResultEnvelope {
            ml: MlSection {
                v: self.v,
                classification: self.classification,
                probability: self.probability,
                thresholds: self.thresholds,
                version: self.version.clone(),
                analyzed_at: self.analyzed_at.clone(),
                fs: ml_fs,
                pids: self.pids.clone(),
                deleted: self.deleted,
            },
            raw,
        }
    }
}
