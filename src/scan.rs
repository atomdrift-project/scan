//! Recursive file-system scanning and classification.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::OutputFormat;
use crate::explain::ShapImportance;
use crate::features::{ExtractContext, crit_ordinal};
use crate::model::{Classification, Decision, Model, RouteScore, SkippedRoute, Thresholds};

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
    extra: bool,
    level: Option<u16>,
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
            extra,
            level: None,
        })
    }

    /// Attach the severity level that produced the resolved thresholds.
    ///
    /// `None` indicates manual thresholds (no level applies); `Some(n)` is the
    /// 0..=10000 level that was used to pick `thresholds` from the model's
    /// `severity_levels[]` table. Folded into `ml.l` in the JSON envelope (which
    /// also encodes the benign verdict via the `-1` sentinel) so downstream
    /// consumers can correlate verdicts with FPR severity.
    #[must_use]
    pub const fn with_level(mut self, level: Option<u16>) -> Self {
        self.level = level;
        self
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

    /// Whether to show extra debug info (raw probability, SHAP values) in terminal output.
    #[must_use]
    pub const fn extra(&self) -> bool {
        self.extra
    }

    /// Severity level (0..=10000) used to pick thresholds, or `None` when manual
    /// thresholds were supplied via `--suspicious-threshold` / `--hostile-threshold`.
    #[must_use]
    pub const fn level(&self) -> Option<u16> {
        self.level
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
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
            false,
        );

        assert!(result.is_err());
    }

    #[test]
    fn scan_config_level_defaults_to_none() {
        let config = ScanConfig::new(
            "/tmp/models",
            OutputFormat::Terminal,
            None,
            DisplayFilter::alerts_only(),
            4_000,
            false,
        )
        .expect("valid config");
        assert!(config.level().is_none());
    }

    #[test]
    fn scan_config_with_level_persists() {
        let config = ScanConfig::new(
            "/tmp/models",
            OutputFormat::Terminal,
            None,
            DisplayFilter::alerts_only(),
            4_000,
            false,
        )
        .expect("valid config")
        .with_level(Some(7));
        assert_eq!(config.level(), Some(7));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod trait_floor_tests {
    use super::*;
    use serde_json::json;

    fn benign() -> Decision {
        Decision {
            class: Classification::Benign,
            probability: 0.1,
            threshold: 0.65,
            l: Some(-1),
        }
    }

    /// A report with `n` findings at crit `crit`/confidence `conf`, padded with
    /// `pad` baseline (crit-0) findings so the total/fraction can be controlled.
    fn report(crit: u64, conf: f64, n: usize, pad: usize) -> serde_json::Value {
        let mut ts: Vec<serde_json::Value> = (0..n).map(|_| json!({"l": crit, "c": conf})).collect();
        ts.extend((0..pad).map(|_| json!({"l": 0, "c": 0.9})));
        json!({ "ts": ts })
    }

    #[test]
    fn one_confident_crit5_escalates_to_grid_max_plus_1() {
        let mut d = benign();
        apply_trait_floor(&mut d, &report(5, 0.8, 1, 0), 100, "test");
        assert_eq!(d.class, Classification::Suspicious);
        assert_eq!(d.l, Some(101));
    }

    #[test]
    fn low_confidence_crit5_is_ignored() {
        let mut d = benign();
        // c < 0.76 → not counted, stays benign.
        apply_trait_floor(&mut d, &report(5, 0.5, 3, 0), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn two_confident_crit4_above_fraction_escalates_to_grid_max_plus_2() {
        let mut d = benign();
        // 4 confident crit-4 out of 4 total → fraction 1.0 >= 0.05.
        apply_trait_floor(&mut d, &report(4, 0.9, 4, 0), 100, "test");
        assert_eq!(d.class, Classification::Suspicious);
        assert_eq!(d.l, Some(102));
    }

    #[test]
    fn confident_crit4_below_fraction_stays_benign() {
        let mut d = benign();
        // 2 confident crit-4 diluted by 200 baseline findings → fraction ~0.01.
        apply_trait_floor(&mut d, &report(4, 0.9, 2, 200), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn low_confidence_crit4_does_not_count_toward_pair() {
        let mut d = benign();
        // Only one confident crit-4; the other two are below threshold.
        let report = json!({"ts": [
            {"l": 4, "c": 0.9},
            {"l": 4, "c": 0.5},
            {"l": 4, "c": 0.6},
        ]});
        apply_trait_floor(&mut d, &report, 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn missing_confidence_defaults_below_threshold() {
        let mut d = benign();
        // `c` omitted → DEFAULT_TRAIT_CONFIDENCE (0.5) < 0.76, so it never counts.
        let report = json!({"ts": [{"l": 5}, {"l": 5}]});
        apply_trait_floor(&mut d, &report, 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn never_lowers_a_non_benign_verdict() {
        let mut d = benign();
        d.class = Classification::Hostile;
        d.l = Some(50);
        apply_trait_floor(&mut d, &report(5, 0.9, 5, 0), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.l, Some(50));
    }

    #[test]
    fn reads_findings_from_embedded_fs_shape() {
        let mut d = benign();
        let report = json!({"fs": [{"ts": [{"l": 5, "c": 0.8}]}]});
        apply_trait_floor(&mut d, &report, 100, "test");
        assert_eq!(d.class, Classification::Suspicious);
        assert_eq!(d.l, Some(101));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod envelope_tests {
    use super::*;

    fn base_result() -> ScanResult {
        ScanResult {
            v: "6",
            classification: Classification::Benign,
            probability: 0.10,
            threshold: 0.65,
            // Benign that never fires at any grid level.
            l: Some(-1),
            version: "test".to_string(),
            analyzed_at: "2026-04-16T00:00:00Z".to_string(),
            cleave: None,
            pids: None,
            deleted: None,
            path: "/tmp/x".to_string(),
            finding_counts: FindingCounts::default(),
            formula: String::new(),
            reasons: Vec::new(),
            top_findings: Vec::new(),
            model_scores: Vec::new(),
            skipped_models: Vec::new(),
            file_type: "unknown".to_string(),
            size_bytes: 0,
            sha256: String::new(),
            embedded_files: Vec::new(),
        }
    }

    #[test]
    fn envelope_serializes_l_and_drops_legacy_fields() {
        // `ml.l` is the model's level-independent marker, serialized verbatim
        // (`-1` for a file that never fires). The dropped v5 fields must not
        // appear anywhere in the envelope.
        let r = base_result();
        let json = serde_json::to_value(r.to_envelope()).expect("serialize");
        assert_eq!(json["ml"]["v"].as_str(), Some("6"));
        assert_eq!(json["ml"]["l"].as_i64(), Some(-1));
        for dropped in [
            "class",
            "threshold",
            "level",
            "thresholds",
            "oclass",
            "oprob",
        ] {
            assert!(
                json["ml"].get(dropped).is_none(),
                "v6 envelope must not emit `{dropped}`"
            );
        }
    }

    #[test]
    fn envelope_emits_null_l_in_manual_mode() {
        let mut r = base_result();
        r.l = None;
        let json = serde_json::to_value(r.to_envelope()).expect("serialize");
        assert!(
            json["ml"]["l"].is_null(),
            "manual-threshold mode (no level table) serializes l as null"
        );
    }

    #[test]
    fn envelope_emits_firing_level() {
        let mut r = base_result();
        r.classification = Classification::Hostile;
        r.probability = 0.99;
        r.l = Some(7);
        let json = serde_json::to_value(r.to_envelope()).expect("serialize");
        assert_eq!(json["ml"]["l"].as_i64(), Some(7));
    }

    #[test]
    fn envelope_l_is_independent_of_verdict() {
        // A file the model only flags at a high level reports that true level
        // even when the active caps render it benign — the envelope (hence the
        // cache key) is identical regardless of the deploy `-l`.
        let mut r = base_result();
        r.classification = Classification::Benign;
        r.l = Some(500);
        let json = serde_json::to_value(r.to_envelope()).expect("serialize");
        assert_eq!(json["ml"]["l"].as_i64(), Some(500));
    }

    #[test]
    fn envelope_per_file_l_reflects_each_member() {
        // Each `fs[]` row reports its own file's lowest-firing-level: the root
        // carries the envelope `l`, members their own (matched by path suffix).
        let mut r = base_result();
        r.l = Some(20);
        r.probability = 0.97;
        r.cleave = Some(serde_json::json!({
            "fs": [
                {"id": 0, "dp": 0, "path": "/tmp/x"},
                {"id": 1, "dp": 1, "path": "/tmp/x!!evil.sh"},
                {"id": 2, "dp": 1, "path": "/tmp/x!!readme.txt"},
            ]
        }));
        let member = |path: &str, l: Option<i32>, prob: f32| EmbeddedFile {
            path: path.to_string(),
            file_type: "unknown".to_string(),
            classification: Classification::Benign,
            probability: prob,
            threshold: 0.8,
            l,
            model_scores: Vec::new(),
            skipped_models: Vec::new(),
            formula: String::new(),
            top_findings: Vec::new(),
        };
        r.embedded_files = vec![
            member("evil.sh", Some(2), 0.99),
            member("readme.txt", Some(-1), 0.01),
        ];
        let json = serde_json::to_value(r.to_envelope()).expect("serialize");
        let fs = json["ml"]["fs"].as_array().expect("fs array");
        assert_eq!(fs[0]["l"].as_i64(), Some(20), "root row carries envelope l");
        assert_eq!(fs[1]["l"].as_i64(), Some(2), "evil.sh reports its own l");
        assert_eq!(
            fs[2]["l"].as_i64(),
            Some(-1),
            "readme.txt reports its own l"
        );
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
    /// Probability the verdict was decided on.
    pub probability: f32,
    /// Cutoff defining the verdict band — the same value `probability` was
    /// compared against to produce `classification`.
    pub threshold: f32,
    /// Level-independent envelope marker (`ml.l`): the lowest false-positive
    /// level (FP per 100M benigns) at which this file's hostile decision fires.
    /// `Some(-1)` = never fires (clean); `Some(0..=10000)` = the firing level;
    /// `None` = manual-threshold mode. Independent of the deploy `-l`, so the
    /// envelope is identical across levels and cache-shareable — `-l` only moves
    /// the cutoffs that turn `l` into `classification`.
    pub l: Option<i32>,
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
    /// Per-model route scores from the routed ensemble.
    pub model_scores: Vec<RouteScore>,
    /// Applicable model routes skipped by the routed ensemble.
    pub skipped_models: Vec<SkippedRoute>,
}

/// A representative cleave finding surfaced alongside a classification.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TopFinding {
    /// Finding identifier (e.g. "objectives/evasion/process::injection").
    pub id: String,
    /// Criticality ordinal (0=filtered .. 5=hostile).
    pub crit: u32,
    /// Cleave-assigned confidence in `[0.0, 1.0]`. Cleave omits the field
    /// from compact JSON when it equals its default (0.5); the [`From`] impl
    /// restores that default so downstream consumers always see a value.
    pub conf: f32,
    /// Human-readable description of the finding.
    pub desc: String,
}

/// Default cleave trait confidence when the JSON `c` field is omitted.
/// Mirrors `DEFAULT_CONF` in `cleave::types::compact`.
const DEFAULT_TRAIT_CONFIDENCE: f32 = 0.5;

impl From<&serde_json::Value> for TopFinding {
    fn from(f: &serde_json::Value) -> Self {
        // Cleave's compact-v4 omits `c` when it equals the 0.5 default. The
        // float values are bucketed to two decimals so the f64→f32 down-cast
        // is exact for every value the analyzer actually emits.
        #[allow(clippy::cast_possible_truncation)]
        let conf = f["c"]
            .as_f64()
            .map_or(DEFAULT_TRAIT_CONFIDENCE, |x| x as f32);
        Self {
            id: f["i"].as_str().unwrap_or("").to_string(),
            crit: crit_ordinal(f),
            conf,
            desc: f["d"].as_str().unwrap_or("").to_string(),
        }
    }
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
    /// Cutoff that defined this embedded file's verdict band.
    pub threshold: f32,
    /// Level-independent lowest-firing-level marker for this member. See
    /// [`ScanResult::l`].
    pub l: Option<i32>,
    /// Per-model route scores for this embedded file.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub model_scores: Vec<RouteScore>,
    /// Applicable model routes skipped for this embedded file.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub skipped_models: Vec<SkippedRoute>,
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

// secs is always positive finite (computed from elapsed/rate); the f64→u32 cast
// is safe for any realistic scan duration.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
    let model = Model::load(config.model_dir(), config.thresholds(), config.level())?;

    let shap = ShapImportance::load(config.model_dir()).ok();
    let ctx = ExtractContext::new(model.spec());
    let cancellation = Arc::new(AtomicBool::new(false));
    let ctrlc_flag = Arc::clone(&cancellation);
    let _ = ctrlc::set_handler(move || {
        if ctrlc_flag.load(Ordering::Relaxed) {
            // Second ctrl-c: reap rizin workers, then hard exit. Cleave runs
            // each rizin in its own process group, so SIGINT on the terminal
            // never reaches them — without an explicit SIGKILL here, every
            // in-flight child would outlive us as an orphan.
            cleave::kill_all_rizin_groups();
            std::process::exit(130);
        }
        eprintln!("\nInterrupted — finishing current file…");
        ctrlc_flag.store(true, Ordering::Relaxed);
    });
    let cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        cancellation: Some(Arc::clone(&cancellation)),
        ..Default::default()
    };
    let is_terminal = matches!(config.format(), OutputFormat::Terminal);
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
                if config.format() == OutputFormat::Json || config.filter().shows(&r.classification)
                {
                    emit_result(&r, config, false, &stdout);
                }
            }
            Err(e) => {
                let msg = crate::tools::enrich_error(&e).unwrap_or_else(|| format!("{e:#}"));
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
            duration_ms: crate::duration_ms(scan_start.elapsed()),
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
            if let Some(total) = total {
                let total32 = u32::try_from(total).unwrap_or(u32::MAX);
                let _ = total_files.set(total32);
                if is_terminal && total > 1 {
                    crate::output::print_header(path, total);
                    let _ = progress.set(Progress::new(total32));
                }
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
                    let msg = crate::tools::enrich_error(&e).unwrap_or_else(|| format!("{e:#}"));
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
        duration_ms: crate::duration_ms(scan_start.elapsed()),
    };

    if is_terminal {
        crate::output::print_summary(&summary);
    }

    Ok(summary)
}

/// Aggregate verdict counters shared across the parallel scan workers.
#[derive(Default)]
struct Tally {
    hostile: AtomicU32,
    suspicious: AtomicU32,
    benign: AtomicU32,
    errors: AtomicU32,
}

impl Tally {
    /// Record one classified file against the matching counter.
    fn count(&self, classification: Classification) {
        let counter = match classification {
            Classification::Hostile => &self.hostile,
            Classification::Suspicious => &self.suspicious,
            Classification::Benign => &self.benign,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Assemble the final [`ScanSummary`]. Every analyzed file lands in exactly
    /// one of the four counters, so their sum is the total file count.
    fn summary(&self, scan_start: Instant) -> ScanSummary {
        let hostile = self.hostile.load(Ordering::Relaxed);
        let suspicious = self.suspicious.load(Ordering::Relaxed);
        let benign = self.benign.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);
        ScanSummary {
            total_files: hostile + suspicious + benign + errors,
            hostile,
            suspicious,
            benign,
            errors,
            duration_ms: crate::duration_ms(scan_start.elapsed()),
        }
    }
}

/// Run the ML pipeline on one cleave result and record the verdict.
///
/// Shared by the file-batch and per-directory streams in [`run_paths`]. Called
/// from rayon worker threads, so every shared input is behind `&`/atomics.
#[allow(clippy::too_many_arguments)]
fn record_file_result(
    file_path: &Path,
    cleave_result: Result<cleave::AnalysisReport>,
    ctx: &ExtractContext,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
    cancellation: Option<&Arc<AtomicBool>>,
    tally: &Tally,
    stdout: &Mutex<std::io::Stdout>,
) {
    let scan_result = cleave_result.and_then(|report| {
        process_report(file_path, report, ctx, model, shap, config, cancellation)
    });
    match scan_result {
        Ok(r) => {
            tally.count(r.classification);
            if config.format() == OutputFormat::Json || config.filter().shows(&r.classification) {
                emit_result(&r, config, false, stdout);
            }
        }
        Err(e) => {
            let msg = crate::tools::enrich_error(&e).unwrap_or_else(|| format!("{e:#}"));
            tracing::warn!("error analyzing {}: {}", file_path.display(), msg);
            tally.errors.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Scan a set of file and directory paths, classifying every file.
///
/// Explicit files are analyzed together as one parallel batch (via
/// [`cleave::scan_files`]); each directory argument is streamed through cleave's
/// recursive walker. Both feed a single shared [`Tally`], so the returned
/// [`ScanSummary`] aggregates across every path. Results print in completion
/// order, not argument order.
///
/// A path that is neither a file nor a directory is logged and counted as an
/// error; the remaining paths are still scanned.
///
/// # Errors
/// Propagates model-load and cleave setup failures. Per-file analysis errors
/// are recorded in the summary, not returned.
pub fn run_paths(paths: &[PathBuf], config: &ScanConfig) -> Result<ScanSummary> {
    let model = Model::load(config.model_dir(), config.thresholds(), config.level())?;
    let shap = ShapImportance::load(config.model_dir()).ok();
    let ctx = ExtractContext::new(model.spec());

    let cancellation = Arc::new(AtomicBool::new(false));
    let ctrlc_flag = Arc::clone(&cancellation);
    let _ = ctrlc::set_handler(move || {
        if ctrlc_flag.load(Ordering::Relaxed) {
            // Second ctrl-c: reap rizin workers, then hard exit. See `run`.
            cleave::kill_all_rizin_groups();
            std::process::exit(130);
        }
        eprintln!("\nInterrupted — finishing current file…");
        ctrlc_flag.store(true, Ordering::Relaxed);
    });

    let cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        cancellation: Some(Arc::clone(&cancellation)),
        ..Default::default()
    };
    let is_terminal = matches!(config.format(), OutputFormat::Terminal);
    let scan_start = Instant::now();

    let tally = Tally::default();
    let stdout = Mutex::new(std::io::stdout());

    // Partition the requested paths: explicit files become one parallel batch,
    // each directory is walked separately, and anything else is an error.
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    for path in paths {
        if path.is_file() {
            files.push(path.clone());
        } else if path.is_dir() {
            dirs.push(path.clone());
        } else {
            tracing::warn!("path does not exist: {}", path.display());
            tally.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    let record = |file_path: &Path, result: Result<cleave::AnalysisReport>| {
        record_file_result(
            file_path,
            result,
            &ctx,
            &model,
            shap.as_ref(),
            config,
            Some(&cancellation),
            &tally,
            &stdout,
        );
    };

    if !files.is_empty() {
        cleave::scan_files(&files, &cleave_opts, |event| {
            if let cleave::ScanEvent::File { path, result } = event {
                record(&path, *result);
            }
        })?;
    }

    for dir in &dirs {
        cleave::scan_directory(dir, &cleave_opts, |event| {
            if let cleave::ScanEvent::File { path, result } = event {
                record(&path, *result);
            }
        })?;
    }

    let summary = tally.summary(scan_start);
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
        OutputFormat::Json => {
            let envelope = r.envelope_ref();
            let Ok(mut out) = stdout.lock() else {
                return;
            };
            if let Err(e) = serde_json::to_writer(&mut *out, &envelope) {
                tracing::error!(path = %r.path, "failed to serialize scan result: {e}");
                return;
            }
            let _ = out.write_all(b"\n");
        }
    }
}

/// Intermediate classification result from the model pipeline.
/// Produced by `classify_report`, consumed when building a `ScanResult`.
pub(crate) struct ClassifiedReport {
    pub(crate) classification: Classification,
    pub(crate) probability: f32,
    pub(crate) threshold: f32,
    /// Level-independent lowest-firing-level marker for the root file. See
    /// [`ScanResult::l`].
    pub(crate) l: Option<i32>,
    pub(crate) finding_counts: FindingCounts,
    pub(crate) formula: String,
    pub(crate) reasons: Vec<Reason>,
    pub(crate) top_findings: Vec<TopFinding>,
    pub(crate) model_scores: Vec<RouteScore>,
    pub(crate) skipped_models: Vec<SkippedRoute>,
    pub(crate) file_type: String,
    pub(crate) size_bytes: u64,
    pub(crate) sha256: String,
    pub(crate) embedded_files: Vec<EmbeddedFile>,
    pub(crate) report_json: serde_json::Value,
}

/// crit-4 fraction gate for the trait floor's suspicious arm. A sparse, severe
/// dropper (an npm install-hook beacon: the embedded package.json is ~0.15, its
/// .tgz container ~0.08) clears it; a busy benign binary with a couple of
/// incidental crit-4 findings among hundreds (bash/sh ~0.01) does not. Measured
/// on a /usr/bin sample — see the trait-floor prevalence check.
const TRAIT_FLOOR_CRIT4_FRACTION: f32 = 0.05;

/// Minimum cleave confidence (`c`) for a finding to count toward the trait floor.
/// Low-confidence high-crit findings are exactly the incidental ones that fire on
/// busy benign binaries (e.g. a couple of speculative crit-4s among hundreds of
/// findings), so the floor only acts on evidence cleave is sure about. Measured:
/// at 0.76 the dropper keeps all 4 crit-4s and the PE keeps all 3 crit-5s, while
/// no /usr/bin benign trips the crit-5 arm.
const TRAIT_FLOOR_MIN_CONFIDENCE: f32 = 0.76;

/// Confidence-filtered crit-5/crit-4 tallies plus the file's total finding count.
/// The crit tiers count only findings with `c >= TRAIT_FLOOR_MIN_CONFIDENCE`; the
/// total is every finding (the fraction's denominator is the file's whole activity,
/// not just its confident severe traits).
struct TraitFloorCounts {
    hostile: u32,
    suspicious: u32,
    total: u32,
}

fn trait_floor_counts(report: &serde_json::Value) -> TraitFloorCounts {
    let findings = report["ts"].as_array().or_else(|| {
        report["fs"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|f| f["ts"].as_array())
    });
    let mut out = TraitFloorCounts {
        hostile: 0,
        suspicious: 0,
        total: 0,
    };
    let Some(findings) = findings else {
        return out;
    };
    for f in findings {
        out.total += 1;
        let conf = f["c"]
            .as_f64()
            .map_or(DEFAULT_TRAIT_CONFIDENCE, |x| x as f32);
        if conf < TRAIT_FLOOR_MIN_CONFIDENCE {
            continue;
        }
        match f["l"].as_u64().unwrap_or(0) {
            5 => out.hostile += 1,
            4 => out.suspicious += 1,
            _ => {}
        }
    }
    out
}

/// Trait floor: escalate a model-**Benign** verdict to **Suspicious** when cleave
/// surfaced high-criticality evidence the model didn't act on, recorded as an
/// off-grid synthetic level so the signal is preserved and its source is legible:
///   - `>= 1` hostile (crit-5) trait → `grid_max + 1` (crit-5 is high precision: ~0% benign FP)
///   - `>= 2` suspicious (crit-4) traits AND crit-4 fraction `>= TRAIT_FLOOR_CRIT4_FRACTION` → `grid_max + 2`
///
/// Only confident findings count (`c >= TRAIT_FLOOR_MIN_CONFIDENCE`); the crit-4
/// fraction's denominator is the file's *total* finding count, so a sparse, severe
/// dropper clears it while a busy benign binary with a couple of incidental
/// crit-4s does not.
///
/// Only raises Benign → Suspicious; never lowers a model verdict (a file the
/// model already graded suspicious/hostile is left untouched). `verdict_for_level`
/// maps any `l > active_level` to Suspicious, so the synthetic levels classify
/// correctly without changing the verdict logic.
fn apply_trait_floor(
    decision: &mut Decision,
    report: &serde_json::Value,
    grid_max: u16,
    label: &str,
) {
    if decision.class != Classification::Benign {
        return;
    }
    let counts = trait_floor_counts(report);
    if counts.hostile >= 1 {
        decision.class = Classification::Suspicious;
        decision.l = Some(i32::from(grid_max) + 1);
        // Loud by design: the model graded this benign yet cleave is confident it
        // carries a hostile (crit-5) trait. If the models are doing their job this
        // is extraordinary — every occurrence is a model gap worth investigating.
        tracing::warn!(
            path = %label,
            arm = "crit5",
            confident_hostile = counts.hostile,
            synthetic_level = i32::from(grid_max) + 1,
            "TRAIT FLOOR: model said benign but cleave found a confident hostile trait — escalated to suspicious",
        );
        return;
    }
    if counts.suspicious >= 2
        && counts.total > 0
        && counts.suspicious as f32 / counts.total as f32 >= TRAIT_FLOOR_CRIT4_FRACTION
    {
        decision.class = Classification::Suspicious;
        decision.l = Some(i32::from(grid_max) + 2);
        tracing::warn!(
            path = %label,
            arm = "crit4_fraction",
            confident_suspicious = counts.suspicious,
            total_findings = counts.total,
            crit4_fraction = format!(
                "{:.3}",
                counts.suspicious as f32 / counts.total as f32
            ),
            synthetic_level = i32::from(grid_max) + 2,
            "TRAIT FLOOR: model said benign but cleave found a sparse cluster of confident suspicious traits — escalated to suspicious",
        );
    }
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
    embedded_file_limit: Option<usize>,
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
    // SHAP explanations use raw (unstandardized) values, so when SHAP is
    // enabled we run `explain()` first, then standardize `raw_features` in place
    // for `predict()`. Avoids cloning the whole feature vector.
    let reasons = shap
        .map(|s| s.explain(&raw_features, model.spec().feature_names()))
        .unwrap_or_default();
    model.spec().standardize(&mut raw_features);

    // The cleave file_type drives ensemble routing; pull it from the top-level
    // file in the report. Single-bundle deployments ignore it via predict_for's
    // fast path.
    let pf = report_json["fs"]
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or(&report_json);
    let file_type = pf["type"].as_str().unwrap_or("unknown").to_string();
    let size_bytes = pf["sz"].as_u64().unwrap_or(0);
    let sha256 = pf["sha"].as_str().unwrap_or("").to_string();

    let (mut decision, model_scores, skipped_models) =
        model.predict_for_report_detailed(&file_type, &raw_features, &report_json)?;

    let finding_counts = count_findings_from_json(&report_json);

    tracing::debug!(
        path = %label,
        file_type = %file_type,
        classification = ?decision.class,
        probability = format!("{:.4}", decision.probability),
        threshold = format!("{:.4}", decision.threshold),
        features_nonzero = nonzero,
        features_total = expected,
        findings_hostile = finding_counts.hostile,
        findings_suspicious = finding_counts.suspicious,
        findings_notable = finding_counts.notable,
        findings_baseline = finding_counts.baseline,
        formula = %formula,
        "classified file",
    );

    // Trait floor: a screaming cleave signal the model graded benign is escalated
    // to suspicious (off-grid synthetic level). Applied per-file so an archive
    // member's evidence elevates its container via decision_outranks below.
    apply_trait_floor(&mut decision, &report_json, model.grid_max(), label);

    // Extract embedded files (archive members at depth > 0), run each through
    // the model individually, and elevate the parent if any embedded file's
    // decision outranks it. Ordinary scans cap embedded work to prevent
    // resource exhaustion; validation passes None so every Cleave-produced
    // embedded file is checked.
    let embedded_iter = report_json["fs"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|f| f["dp"].as_u64().unwrap_or(0) > 0);
    let embedded_entries: Vec<&serde_json::Value> = match embedded_file_limit {
        Some(limit) => embedded_iter.take(limit).collect(),
        None => embedded_iter.collect(),
    };

    let mut embedded_files: Vec<EmbeddedFile> = Vec::with_capacity(embedded_entries.len());
    let mut max_decision: Decision = decision;

    for ef in &embedded_entries {
        if let Some(c) = cancellation
            && c.load(Ordering::Relaxed)
        {
            anyhow::bail!("analysis cancelled during embedded file processing");
        }

        let mut ef_features = ctx.extract_file(ef);
        model.spec().standardize(&mut ef_features);
        let ef_type = ef["type"].as_str().unwrap_or("unknown");
        let (mut ef_decision, ef_model_scores, ef_skipped_models) = model
            .predict_for_file_detailed(ef_type, &ef_features, ef)
            .unwrap_or((
                Decision {
                    class: Classification::Benign,
                    probability: 0.0,
                    threshold: model.thresholds().suspicious,
                    l: None,
                },
                Vec::new(),
                Vec::new(),
            ));
        // Trait floor on the member's own findings — a sparse, severe dropper
        // (the npm install-hook beacon lives in the embedded package.json) scores
        // above the crit-4 fraction gate even when the container's findings dilute
        // it; the floored member then elevates the container below.
        apply_trait_floor(
            &mut ef_decision,
            ef,
            model.grid_max(),
            ef["path"].as_str().unwrap_or(label),
        );

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
            .map(TopFinding::from)
            .collect();

        tracing::debug!(
            parent = %label,
            embedded_path = %rel_path,
            probability = format!("{:.4}", ef_decision.probability),
            classification = ?ef_decision.class,
            "classified embedded file",
        );

        if decision_outranks(&ef_decision, &max_decision) {
            max_decision = ef_decision;
        }

        embedded_files.push(EmbeddedFile {
            path: rel_path,
            file_type: ef["type"].as_str().unwrap_or("unknown").to_string(),
            classification: ef_decision.class,
            probability: ef_decision.probability,
            threshold: ef_decision.threshold,
            l: ef_decision.l,
            model_scores: ef_model_scores,
            skipped_models: ef_skipped_models,
            formula: ef["f"].as_str().unwrap_or("").to_string(),
            top_findings: ef_top_findings,
        });
    }

    embedded_files.sort_by(|a, b| b.probability.total_cmp(&a.probability));
    if embedded_file_limit.is_some() {
        embedded_files.truncate(10);
    }

    // If an embedded file's decision outranks the parent, elevate.
    let final_decision = if decision_outranks(&max_decision, &decision) {
        tracing::info!(
            path = %label,
            original_probability = format!("{:.4}", decision.probability),
            elevated_probability = format!("{:.4}", max_decision.probability),
            elevated_classification = ?max_decision.class,
            elevated_threshold = format!("{:.4}", max_decision.threshold),
            "elevated archive classification due to embedded file",
        );
        max_decision
    } else {
        decision
    };

    let top_findings = extract_top_findings_from_json(&report_json, &final_decision.class);

    Ok(ClassifiedReport {
        classification: final_decision.class,
        probability: final_decision.probability,
        threshold: final_decision.threshold,
        l: final_decision.l,
        finding_counts,
        formula,
        reasons,
        top_findings,
        model_scores,
        skipped_models,
        file_type,
        size_bytes,
        sha256,
        embedded_files,
        report_json,
    })
}

/// Returns `true` when `candidate` should replace `current` as the dominant
/// decision: higher class wins; on ties, higher probability wins.
fn decision_outranks(candidate: &Decision, current: &Decision) -> bool {
    match (candidate.class as u8).cmp(&(current.class as u8)) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => candidate.probability > current.probability,
        std::cmp::Ordering::Less => false,
    }
}

/// Apply litmus model inference to a cleave report. Always returns a ScanResult
/// (even for benign); the caller decides whether to display it.
pub(crate) fn process_report(
    path: &Path,
    report: cleave::AnalysisReport,
    ctx: &ExtractContext,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
    cancellation: Option<&Arc<AtomicBool>>,
) -> Result<ScanResult> {
    let path_display = path.display().to_string();
    let cr = classify_report(
        &path_display,
        report,
        ctx,
        model,
        shap,
        cancellation,
        Some(100),
    )?;
    let is_json = matches!(config.format(), OutputFormat::Json);

    // Include raw cleave report for JSON output (unmutated — ML scores go in the ml section).
    let cleave = if is_json { Some(cr.report_json) } else { None };

    Ok(ScanResult {
        v: "6",
        classification: cr.classification,
        probability: cr.probability,
        threshold: cr.threshold,
        l: cr.l,
        version: model_version_string(model.info()),
        analyzed_at: now_rfc3339(),
        cleave,
        pids: None,
        deleted: None,
        path: path_display,
        finding_counts: cr.finding_counts,
        formula: cr.formula,
        reasons: cr.reasons,
        top_findings: cr.top_findings,
        model_scores: cr.model_scores,
        skipped_models: cr.skipped_models,
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
        shap,
        config,
        cleave_opts.cancellation.as_ref(),
    )
}

/// Classify a payload held entirely in memory.
///
/// This is the in-process analog of [`run`] for callers that already own the
/// bytes (HTTP proxies, S3 fetchers, fuzz harnesses, etc.). It skips the
/// filesystem round-trip used by the on-disk path and relies on cleave's
/// SHA256-keyed analysis cache to short-circuit repeated payloads.
///
/// `filename` is advisory only — cleave uses the extension as a type hint and
/// the value is echoed back in [`ScanResult::path`]. Pass something
/// human-meaningful (e.g. the URL the bytes came from) for logging.
///
/// Unlike [`run`], no `ScanSummary` aggregation happens here; one call =
/// one [`ScanResult`]. Callers that need bulk counters should aggregate.
///
/// # Errors
/// Propagates cleave analysis failures, model inference errors, and feature
/// spec mismatches.
pub fn scan_bytes(
    data: Vec<u8>,
    filename: &str,
    model: &Model,
    ctx: &ExtractContext,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
) -> Result<ScanResult> {
    let cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        ..Default::default()
    };
    let report = cleave::analyze_bytes_owned(data, filename, &cleave_opts)
        .with_context(|| format!("cleave analysis of {filename}"))?;
    process_report(
        std::path::Path::new(filename),
        report,
        ctx,
        model,
        shap,
        config,
        None,
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
        .map(TopFinding::from)
        .collect();

    // Fall back to suspicious-level findings if no hostile-level findings.
    if relevant.is_empty() && min_crit == 5 {
        relevant = findings
            .iter()
            .filter(|f| crit_ordinal(f) >= 4)
            .map(TopFinding::from)
            .collect();
    }

    // Deduplicate by base ID.
    let mut seen = std::collections::HashSet::new();
    relevant.retain(|f| {
        let base = f.id.split("::").next().unwrap_or(&f.id);
        seen.insert(base.to_string())
    });

    relevant.sort_by_key(|f| std::cmp::Reverse(f.crit));
    relevant.truncate(5);
    relevant
}

/// Crate-private helper for [`ScanResult::top_traits`]. Walks the cleave
/// compact report directly because we don't (yet) hold it in typed form on
/// `ScanResult`. Pulls from `report.ts` (single-file analyses) or, failing
/// that, `report.fs[0].ts` (compact-v4 envelopes).
///
/// Returns `Vec<TopFinding>` with `conf` populated, sorted by
/// `crit × conf` descending, deduplicated by base id, truncated to `n`.
#[allow(clippy::cast_precision_loss)]
fn top_traits_by_score(report: &serde_json::Value, n: usize) -> Vec<TopFinding> {
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

    let mut scored: Vec<TopFinding> = findings
        .iter()
        .filter_map(|f| {
            let finding = TopFinding::from(f);
            // Drop filtered (0) and component-tier (1) traits — they're noise
            // for ranking. Anything with no id is also worthless.
            if finding.crit < 2 || finding.id.is_empty() {
                return None;
            }
            Some(finding)
        })
        .collect();

    // Sort first, then dedupe by base id — that way each family keeps its
    // highest-scored representative.
    scored.sort_by(|a, b| {
        let sa = (a.crit as f32) * a.conf;
        let sb = (b.crit as f32) * b.conf;
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut seen = std::collections::HashSet::new();
    scored.retain(|t| {
        let base = t.id.split("::").next().unwrap_or(&t.id).to_owned();
        seen.insert(base)
    });
    scored.truncate(n);
    scored
}

/// Build a compact model version string from ModelInfo.
/// Format: "v{spec_version}.{abi_version}-{sha256_prefix}" or with commit if available.
pub(crate) fn model_version_string(info: &crate::model::ModelInfo) -> String {
    if info.sha256.is_empty() {
        return format!("v{}.{}", info.version, info.abi_version);
    }
    let sha_prefix = if info.sha256.len() >= 8 {
        info.sha256.get(..8).unwrap_or(&info.sha256)
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

fn route_scores_empty(scores: &[crate::model::RouteScore]) -> bool {
    scores.is_empty()
}

fn skipped_routes_empty(routes: &[crate::model::SkippedRoute]) -> bool {
    routes.is_empty()
}

/// Build per-file ML classification entries for the `ml.fs` array.
///
/// Each entry is `{id, prob, l}` keyed by the cleave `fs[].id` field. The root
/// file (`dp=0`) carries the envelope's probability and `l`; embedded archive
/// members are matched by path suffix and report their *own* probability and
/// lowest-firing-level `l` — every row's `l` is therefore the level-independent
/// marker for that specific file. A member with no recorded evaluation (e.g.
/// truncated past the embedded-file cap) falls back to the root values.
fn build_ml_fs(
    report_json: &serde_json::Value,
    root_prob: f32,
    root_l: Option<i32>,
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
        let depth = entry
            .get("dp")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);

        let (prob, l) = if depth == 0 {
            (root_prob, root_l)
        } else {
            let path = entry.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let suffix = path.rsplit_once("!!").map(|(_, r)| r).unwrap_or(path);
            embedded_files
                .iter()
                .find(|ef| ef.path == suffix)
                .map(|ef| (ef.probability, ef.l))
                .unwrap_or((root_prob, root_l))
        };

        out.push(serde_json::json!({
            "id": id,
            "prob": prob,
            "l": l,
        }));
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
    pub(crate) v: &'static str,
    #[serde(rename = "prob")]
    pub(crate) probability: f32,
    /// Resolved verdict marker, always serialized (including as `null`):
    /// - `Some(-1)` → benign.
    /// - `Some(0..=10000)` → hostile; the per-100M-benigns level that selected
    ///   the firing threshold.
    /// - `None` → hostile; manual `--threshold-hostile` / `--threshold-suspicious`
    ///   were used and no level applies.
    #[serde(rename = "l")]
    pub(crate) l: Option<i32>,
    #[serde(rename = "models", skip_serializing_if = "Vec::is_empty")]
    pub(crate) model_scores: Vec<crate::model::RouteScore>,
    #[serde(rename = "skip", skip_serializing_if = "Vec::is_empty")]
    pub(crate) skipped_models: Vec<crate::model::SkippedRoute>,
    pub(crate) version: String,
    pub(crate) analyzed_at: String,
    pub(crate) fs: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) deleted: Option<bool>,
}

/// Zero-copy envelope view serialized directly from `&ScanResult`. Borrows the
/// cleave JSON and the owned `String` fields, so the per-file JSON output path
/// avoids cloning the cleave report (which can be hundreds of KB for archives).
#[derive(Debug, serde::Serialize)]
pub struct ScanResultEnvelopeRef<'a> {
    /// ML classification section (borrowed).
    pub ml: MlSectionRef<'a>,
    /// Raw cleave analysis report (borrowed).
    pub raw: &'a serde_json::Value,
}

/// Borrowed counterpart of [`MlSection`].
#[derive(Debug, serde::Serialize)]
pub struct MlSectionRef<'a> {
    pub(crate) v: &'static str,
    #[serde(rename = "prob")]
    pub(crate) probability: f32,
    /// See [`MlSection::l`] for the encoding of this field.
    #[serde(rename = "l")]
    pub(crate) l: Option<i32>,
    #[serde(rename = "models", skip_serializing_if = "route_scores_empty")]
    pub(crate) model_scores: &'a [crate::model::RouteScore],
    #[serde(rename = "skip", skip_serializing_if = "skipped_routes_empty")]
    pub(crate) skipped_models: &'a [crate::model::SkippedRoute],
    pub(crate) version: &'a str,
    pub(crate) analyzed_at: &'a str,
    pub(crate) fs: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pids: Option<&'a [u32]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) deleted: Option<bool>,
}

impl ScanResult {
    /// Return up to `n` cleave traits from this result, ranked by
    /// `criticality × confidence` (highest first).
    ///
    /// Filtered traits (criticality 0) and component-tier traits (criticality 1)
    /// are dropped; remaining traits are deduplicated by their first
    /// `::`-delimited segment so families (`shell::bash`, `shell::sh`) don't
    /// crowd each other out.
    ///
    /// Returns an empty `Vec` when the result has no attached cleave report
    /// (i.e. `cleave` is `None` because the analyzer was configured with
    /// [`OutputFormat::Terminal`]).
    #[must_use]
    pub fn top_traits(&self, n: usize) -> Vec<TopFinding> {
        self.cleave
            .as_ref()
            .map(|r| top_traits_by_score(r, n))
            .unwrap_or_default()
    }

    /// Build the `{"ml": {...}, "raw": {...}}` envelope for JSON output.
    ///
    /// Prefer [`Self::into_envelope`] when the caller owns the `ScanResult`.
    /// `to_envelope` clones the full `cleave` report (potentially multi-MB
    /// for archive analyses); the move variant avoids that entirely.
    #[must_use]
    pub fn to_envelope(&self) -> ScanResultEnvelope {
        let raw = self.cleave.clone().unwrap_or(serde_json::json!({}));
        let l = self.l;
        let ml_fs = build_ml_fs(&raw, self.probability, l, &self.embedded_files);
        ScanResultEnvelope {
            ml: MlSection {
                v: self.v,
                probability: self.probability,
                l,
                model_scores: self.model_scores.clone(),
                skipped_models: self.skipped_models.clone(),
                version: self.version.clone(),
                analyzed_at: self.analyzed_at.clone(),
                fs: ml_fs,
                pids: self.pids.clone(),
                deleted: self.deleted,
            },
            raw,
        }
    }

    /// Build the envelope, consuming the `ScanResult`. Avoids cloning the
    /// cleave report and owned string fields — the hot path for worker and
    /// server handlers, which drop the result immediately after building the
    /// envelope.
    #[must_use]
    pub fn into_envelope(self) -> ScanResultEnvelope {
        let raw = self.cleave.unwrap_or(serde_json::json!({}));
        let l = self.l;
        let ml_fs = build_ml_fs(&raw, self.probability, l, &self.embedded_files);
        ScanResultEnvelope {
            ml: MlSection {
                v: self.v,
                probability: self.probability,
                l,
                model_scores: self.model_scores,
                skipped_models: self.skipped_models,
                version: self.version,
                analyzed_at: self.analyzed_at,
                fs: ml_fs,
                pids: self.pids,
                deleted: self.deleted,
            },
            raw,
        }
    }

    /// Zero-copy envelope view over `&self`. Use this on the JSON output hot
    /// path (e.g. [`crate::scan`] and [`crate::ps`] `emit_result`) — it avoids
    /// cloning the cleave report and all owned string fields. Prefer
    /// [`Self::into_envelope`] when the caller can give up ownership.
    #[must_use]
    pub fn envelope_ref(&self) -> ScanResultEnvelopeRef<'_> {
        static EMPTY_RAW: OnceLock<serde_json::Value> = OnceLock::new();
        let raw: &serde_json::Value = match &self.cleave {
            Some(v) => v,
            None => EMPTY_RAW.get_or_init(|| serde_json::json!({})),
        };
        let l = self.l;
        let ml_fs = build_ml_fs(raw, self.probability, l, &self.embedded_files);
        ScanResultEnvelopeRef {
            ml: MlSectionRef {
                v: self.v,
                probability: self.probability,
                l,
                model_scores: &self.model_scores,
                skipped_models: &self.skipped_models,
                version: &self.version,
                analyzed_at: &self.analyzed_at,
                fs: ml_fs,
                pids: self.pids.as_deref(),
                deleted: self.deleted,
            },
            raw,
        }
    }
}
