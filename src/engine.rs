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
use crate::{json_alias, json_alias_array, json_alias_str};

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
    interpret: Option<crate::interpret::InterpretConfig>,
    fetch: crate::fetch::FetchPolicy,
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
    /// use scan::{Classification, DisplayFilter, OutputFormat, ScanConfig, Thresholds};
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
    /// assert!(config.filter().shows(&Classification::Hostile));
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
            interpret: None,
            fetch: crate::fetch::FetchPolicy::default(),
        })
    }

    /// Set the external-reference fetch policy: which kinds of reference
    /// (registry packages, bare URLs) discovered in analyzed files to fetch,
    /// re-analyze, and graft into the report. The default [`FetchPolicy`]
    /// selects nothing and disables fetching.
    #[must_use]
    pub const fn with_fetch(mut self, policy: crate::fetch::FetchPolicy) -> Self {
        self.fetch = policy;
        self
    }

    /// The external-reference fetch policy (off by default).
    #[must_use]
    pub(crate) const fn fetch_policy(&self) -> crate::fetch::FetchPolicy {
        self.fetch
    }

    /// Attach the severity level that produced the resolved thresholds.
    ///
    /// `None` indicates manual thresholds (no level applies); `Some(n)` is the
    /// 0..=25000 level that was used to pick `thresholds` from the model's
    /// `severity_levels[]` table. Folded into `ml.lvl` in the JSON envelope (which
    /// also encodes the benign verdict via the `-1` sentinel) so downstream
    /// consumers can correlate verdicts with FPR severity.
    #[must_use]
    pub const fn with_level(mut self, level: Option<u16>) -> Self {
        self.level = level;
        self
    }

    /// Attach an LLM interpretation config (`--interpret`). `None` disables the
    /// pass; callers like `validate` always leave it unset.
    #[must_use]
    pub fn with_interpret(mut self, interpret: Option<crate::interpret::InterpretConfig>) -> Self {
        self.interpret = interpret;
        self
    }

    /// LLM interpretation config, or `None` when `--interpret` was not set.
    #[must_use]
    pub fn interpret(&self) -> Option<&crate::interpret::InterpretConfig> {
        self.interpret.as_ref()
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

    /// Severity level (0..=25000) used to pick thresholds, or `None` when manual
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
            level: Some(-1),
        }
    }

    /// A report with `n` findings at crit `crit`/confidence `conf`, padded with
    /// `pad` baseline (crit-0) findings so the total/fraction can be controlled.
    fn report(crit: u64, conf: f64, n: usize, pad: usize) -> serde_json::Value {
        let mut ts: Vec<serde_json::Value> = (0..n)
            .map(|_| json!({"crit": crit, "conf": conf}))
            .collect();
        ts.extend((0..pad).map(|_| json!({"crit": 0, "conf": 0.9})));
        json!({ "find": ts })
    }

    #[test]
    fn one_confident_crit5_escalates_to_grid_max_plus_1() {
        let mut d = benign();
        apply_trait_floor(&mut d, &report(5, 0.8, 1, 0), 100, "test");
        assert_eq!(d.class, Classification::Suspicious);
        assert_eq!(d.level, Some(101));
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
        assert_eq!(d.level, Some(102));
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
        let report = json!({"find": [
            {"crit": 4, "conf": 0.9},
            {"crit": 4, "conf": 0.5},
            {"crit": 4, "conf": 0.6},
        ]});
        apply_trait_floor(&mut d, &report, 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn missing_confidence_defaults_below_threshold() {
        let mut d = benign();
        // `conf` omitted → DEFAULT_TRAIT_CONFIDENCE (0.5) < 0.76, so it never counts.
        let report = json!({"find": [{"crit": 5}, {"crit": 5}]});
        apply_trait_floor(&mut d, &report, 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn never_lowers_a_non_benign_verdict() {
        let mut d = benign();
        d.class = Classification::Hostile;
        d.level = Some(50);
        apply_trait_floor(&mut d, &report(5, 0.9, 5, 0), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.level, Some(50));
    }

    #[test]
    fn reads_findings_from_embedded_files_shape() {
        let mut d = benign();
        let report = json!({"files": [{"find": [{"crit": 5, "conf": 0.8}]}]});
        apply_trait_floor(&mut d, &report, 100, "test");
        assert_eq!(d.class, Classification::Suspicious);
        assert_eq!(d.level, Some(101));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod envelope_tests {
    use super::*;

    fn base_result() -> ScanResult {
        ScanResult {
            v: "7",
            classification: Classification::Benign,
            probability: 0.10,
            threshold: 0.65,
            // Benign that never fires at any grid level.
            level: Some(-1),
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
            rendered_context: String::new(),
            interpretation: None,
        }
    }

    #[test]
    fn level_confidence_maps_known_grid_and_override_markers() {
        let cases = [
            (None, None),
            (Some(-1), Some(0)),
            (Some(0), Some(100)),
            (Some(1), Some(99)),
            (Some(2), Some(98)),
            (Some(5), Some(95)),
            (Some(50), Some(90)),
            (Some(25000), Some(29)),
            (Some(25001), Some(28)),
            (Some(25002), Some(27)),
            (Some(50000), Some(17)),
            (Some(50001), Some(16)),
            (Some(50002), Some(15)),
        ];
        for (level, want) in cases {
            assert_eq!(level_confidence(level), want, "level {level:?}");
        }
    }

    #[test]
    fn envelope_serializes_lvl_and_drops_legacy_fields() {
        // `ml.lvl` is the model's level-independent marker, serialized verbatim
        // (`-1` for a file that never fires). The dropped v5 fields must not
        // appear anywhere in the envelope.
        let r = base_result();
        let json = serde_json::to_value(r.to_envelope()).expect("serialize");
        assert_eq!(json["ml"]["v"].as_str(), Some("7"));
        assert_eq!(json["ml"]["lvl"].as_i64(), Some(-1));
        assert_eq!(json["ml"]["conf"].as_u64(), Some(0));
        for dropped in [
            "class",
            "l",
            "threshold",
            "level",
            "thresholds",
            "oclass",
            "oprob",
        ] {
            assert!(
                json["ml"].get(dropped).is_none(),
                "v7 envelope must not emit `{dropped}`"
            );
        }
    }

    #[test]
    fn envelope_emits_null_lvl_in_manual_mode() {
        let mut r = base_result();
        r.level = None;
        let json = serde_json::to_value(r.to_envelope()).expect("serialize");
        assert!(
            json["ml"]["lvl"].is_null(),
            "manual-threshold mode (no level table) serializes lvl as null"
        );
        assert!(
            json["ml"]["conf"].is_null(),
            "manual-threshold mode serializes conf as null"
        );
    }

    #[test]
    fn envelope_emits_firing_level() {
        let mut r = base_result();
        r.classification = Classification::Hostile;
        r.probability = 0.99;
        r.level = Some(7);
        let json = serde_json::to_value(r.to_envelope()).expect("serialize");
        assert_eq!(json["ml"]["lvl"].as_i64(), Some(7));
        assert_eq!(json["ml"]["conf"].as_u64(), Some(94));
    }

    #[test]
    fn envelope_level_is_independent_of_verdict() {
        // A file the model only flags at a high level reports that true level
        // even when the active caps render it benign — the envelope (hence the
        // cache key) is identical regardless of the deploy `-l`.
        let mut r = base_result();
        r.classification = Classification::Benign;
        r.level = Some(500);
        let json = serde_json::to_value(r.to_envelope()).expect("serialize");
        assert_eq!(json["ml"]["lvl"].as_i64(), Some(500));
    }

    #[test]
    fn envelope_per_file_level_reflects_each_member() {
        // Each `files[]` row reports its own file's lowest-firing-level: the root
        // carries the envelope `lvl`, members their own (matched by path suffix).
        let mut r = base_result();
        r.level = Some(20);
        r.probability = 0.97;
        r.cleave = Some(serde_json::json!({
            "files": [
                {"id": 0, "dp": 0, "path": "/tmp/x", "type": "zip"},
                {"id": 1, "dp": 1, "path": "/tmp/x!!evil.sh", "type": "shell"},
                {"id": 2, "dp": 1, "path": "/tmp/x!!readme.txt", "type": "text"},
            ]
        }));
        let member = |path: &str, level: Option<i32>, prob: f32| EmbeddedFile {
            path: path.to_string(),
            file_type: "unknown".to_string(),
            classification: Classification::Benign,
            probability: prob,
            threshold: 0.8,
            level,
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
        let files = json["ml"]["files"].as_array().expect("files array");
        assert_eq!(
            files[0]["lvl"].as_i64(),
            Some(20),
            "root row carries envelope lvl"
        );
        assert_eq!(
            files[1]["lvl"].as_i64(),
            Some(2),
            "evil.sh reports its own lvl"
        );
        assert_eq!(
            files[2]["lvl"].as_i64(),
            Some(-1),
            "readme.txt reports its own lvl"
        );
        assert_eq!(
            files[0]["type"].as_str(),
            Some("zip"),
            "each row carries its file type"
        );
        assert_eq!(files[1]["type"].as_str(), Some("shell"));
        assert_eq!(files[2]["type"].as_str(), Some("text"));
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

/// Map a level marker (`ml.lvl`) to a pessimistic human-facing confidence percent.
///
/// This is a display/export confidence, not the model probability (`ml.prob`) and
/// not a posterior probability. The table is intentionally integer-valued and
/// strictly separated across the calibrated deploy grid. `25001` and `25002`
/// are off-grid trait-floor override markers (`grid_max + 1/2`) and sit just
/// below the current L25000 tail. `50001`/`50002` are reserved for the same
/// meaning if the calibrated grid grows to L50000.
#[must_use]
pub const fn level_confidence(level: Option<i32>) -> Option<u8> {
    match level {
        None => None,
        Some(n) if n < 0 => Some(0),
        Some(0) => Some(100),
        Some(1) => Some(99),
        Some(2) => Some(98),
        Some(3) => Some(97),
        Some(4) => Some(96),
        Some(5) => Some(95),
        Some(10) => Some(94),
        Some(20) => Some(93),
        Some(30) => Some(92),
        Some(40) => Some(91),
        Some(50) => Some(90),
        Some(60) => Some(89),
        Some(70) => Some(88),
        Some(80) => Some(87),
        Some(90) => Some(86),
        Some(100) => Some(85),
        Some(200) => Some(82),
        Some(300) => Some(80),
        Some(500) => Some(78),
        Some(1000) => Some(75),
        Some(2000) => Some(66),
        Some(5000) => Some(54),
        Some(7500) => Some(49),
        Some(10000) => Some(45),
        Some(15000) => Some(38),
        Some(20000) => Some(33),
        Some(25000) => Some(29),
        Some(25001) => Some(28),
        Some(25002) => Some(27),
        Some(50000) => Some(17),
        Some(50001) => Some(16),
        Some(50002) => Some(15),
        Some(n) if n > 50002 => Some(15),
        Some(n) if n > 25002 => Some(26),
        Some(n) if n > 25000 => Some(28),
        Some(n) if n > 20000 => Some(29),
        Some(n) if n > 15000 => Some(33),
        Some(n) if n > 10000 => Some(38),
        Some(n) if n > 7500 => Some(45),
        Some(n) if n > 5000 => Some(49),
        Some(n) if n > 2000 => Some(54),
        Some(n) if n > 1000 => Some(66),
        Some(n) if n > 500 => Some(75),
        Some(n) if n > 300 => Some(78),
        Some(n) if n > 200 => Some(80),
        Some(n) if n > 100 => Some(82),
        Some(n) if n > 90 => Some(85),
        Some(n) if n > 80 => Some(86),
        Some(n) if n > 70 => Some(87),
        Some(n) if n > 60 => Some(88),
        Some(n) if n > 50 => Some(89),
        Some(n) if n > 40 => Some(90),
        Some(n) if n > 30 => Some(91),
        Some(n) if n > 20 => Some(92),
        Some(n) if n > 10 => Some(93),
        Some(n) if n > 5 => Some(94),
        Some(_) => Some(95),
    }
}

/// Synthetic false-positive level for an LLM-driven verdict, used when the blend
/// shifts the class away from ML's. Hostile gets the stronger (stricter) level,
/// suspicious a loose review level, benign the clean marker — mapping through
/// [`level_confidence`] to ≈85% / ≈80% / 0%.
const fn interpreted_level(outcome: Classification) -> i32 {
    match outcome {
        Classification::Hostile => 100,
        Classification::Suspicious => 250,
        Classification::Benign => -1,
    }
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
    /// Level-independent envelope marker (`ml.lvl`): the lowest false-positive
    /// level (FP per 100M benigns) at which this file's hostile decision fires.
    /// `Some(-1)` = never fires (clean); `Some(0..=25000)` = the firing level;
    /// `None` = manual-threshold mode. Independent of the deploy `-l`, so the
    /// envelope is identical across levels and cache-shareable — `-l` only moves
    /// the cutoffs that turn `level` into `classification`.
    pub level: Option<i32>,
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
    /// cleave's rendered context view (the annotated hex/source block), captured
    /// while the typed report was in scope. Body-only for the terminal format
    /// (litmus draws the headline); header-prefixed for `--format tiny`.
    pub rendered_context: String,
    /// Optional LLM interpretation blended with the ML verdict (`--interpret`).
    /// Serialized as the response `llm` section; `None` when interpretation was
    /// disabled or gated out.
    pub interpretation: Option<crate::interpret::Interpretation>,
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

/// Default cleave trait confidence when the JSON `conf`/`c` field is omitted.
/// Mirrors `DEFAULT_CONF` in `cleave::types::compact`.
const DEFAULT_TRAIT_CONFIDENCE: f32 = 0.5;

/// The cleave findings array, taken from a single-file report (`find`/`ts`) or,
/// failing that, the first file entry of a compact envelope (`files[0].find`).
fn report_findings(report: &serde_json::Value) -> Option<&Vec<serde_json::Value>> {
    json_alias_array(report, &["find", "ts"]).or_else(|| {
        json_alias_array(report, &["files", "fs"])
            .and_then(|a| a.first())
            .and_then(|f| json_alias_array(f, &["find", "ts"]))
    })
}

impl From<&serde_json::Value> for TopFinding {
    fn from(f: &serde_json::Value) -> Self {
        // Cleave omits `conf` when it equals the 0.5 default. The
        // float values are bucketed to two decimals so the f64→f32 down-cast
        // is exact for every value the analyzer actually emits.
        #[allow(clippy::cast_possible_truncation)]
        let conf = json_alias(f, &["conf", "c"])
            .and_then(serde_json::Value::as_f64)
            .map_or(DEFAULT_TRAIT_CONFIDENCE, |x| x as f32);
        Self {
            id: json_alias_str(f, &["id", "i"]).unwrap_or("").to_string(),
            crit: crit_ordinal(f),
            conf,
            desc: json_alias_str(f, &["desc", "d"]).unwrap_or("").to_string(),
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
    /// [`ScanResult::level`].
    pub level: Option<i32>,
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
        let tally = Tally::default();
        let stdout = Mutex::new(std::io::stdout());
        let cleave_result = cleave::analyze_file(path, &cleave_opts)
            .with_context(|| format!("cleave analysis of {}", path.display()));
        record_file_result(
            path,
            cleave_result,
            &ctx,
            &model,
            shap.as_ref(),
            config,
            cleave_opts.cancellation.as_ref(),
            &tally,
            &stdout,
            None,
        );
        let summary = tally.summary(scan_start);
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
    let tally = Tally::default();
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
            record_file_result(
                file_path,
                *result,
                &ctx,
                &model,
                shap.as_ref(),
                config,
                cleave_opts.cancellation.as_ref(),
                &tally,
                &stdout,
                progress.get(),
            );
        }
    })?;

    if let Some(p) = progress.get() {
        p.finish();
    }

    // Every analyzed file is tallied, so the tally's sum is the total — but
    // prefer cleave's upfront count when the walk reported one.
    let mut summary = tally.summary(scan_start);
    if let Some(&total) = total_files.get() {
        summary.total_files = total;
    }

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
/// The single home for every scan's per-file recording: [`run`]'s single-file
/// and directory scans and [`run_paths`]'s file-batch and per-directory streams.
/// Called from rayon worker threads, so every shared input is behind `&`/atomics.
/// `progress`, when present, is advanced per file and redrawn after each shown
/// result.
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
    progress: Option<&Progress>,
) {
    let scan_result = cleave_result.and_then(|report| {
        process_report(file_path, report, ctx, model, shap, config, cancellation)
    });
    if let Some(p) = progress {
        p.increment();
    }
    match scan_result {
        Ok(r) => {
            tally.count(r.classification);
            if config.format() == OutputFormat::Json || config.filter().shows(&r.classification) {
                emit_result(&r, config, progress.is_some(), stdout);
                if let Some(p) = progress {
                    p.redraw();
                }
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
            None,
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
        // The terminal view's full render (litmus badge + cleave header/body)
        // was built in `classify_report`; write it verbatim.
        OutputFormat::Terminal => {
            let _ = show_progress;
            let Ok(mut out) = stdout.lock() else {
                return;
            };
            let _ = out.write_all(r.rendered_context.as_bytes());
            // `--extra`: append the full ML diagnostics that explain the grade —
            // which route drove it and the top SHAP features behind it. The
            // terminal view's cleave body only shows static findings; route
            // scores + reasons are computed but were previously dropped here.
            if config.extra() {
                write_extra_diagnostics(&mut *out, r);
            }
        }
        // `--format tiny` prefixes the machine verdict line, never colored.
        OutputFormat::Tiny => {
            let Ok(mut out) = stdout.lock() else {
                return;
            };
            write_tiny(&mut *out, r);
        }
    }
}

/// The cleave context density litmus renders at. `--format tiny` uses cleave's
/// full tiny (machine/LLM). The terminal view is cut from cleave's own terminal
/// render — same rich header and body — but capped at the top 3 traits (cleave
/// shows all notable+), with litmus adding a verdict badge + subtitle on top.
pub(crate) fn tiny_opts_for(config: &ScanConfig) -> cleave::output::TinyOpts {
    if matches!(config.format(), OutputFormat::Tiny) {
        cleave::output::TinyOpts::tiny()
    } else {
        cleave::output::TinyOpts {
            top_n: 3,
            // Only the hit lines/rows — no surrounding context, no `⋯` gap
            // markers, no padding rows in the hex view.
            context_lines: Some(1),
            full_context: false,
            header: cleave::output::HeaderStyle::Rich,
            ..cleave::output::TinyOpts::terminal()
        }
    }
}

/// Append full ML diagnostics (route scores + SHAP reasons) under the rendered
/// context, for `--extra`. Shows which route drove the grade and the top SHAP
/// features behind the top-level featurization. Embedded archive members list
/// their route scores; per-member SHAP reasons are not computed (reasons exist
/// only for the top-level file), so to attribute an embedded hit, scan the
/// extracted member directly — it then becomes the top-level file.
pub(crate) fn write_extra_diagnostics(out: &mut dyn std::io::Write, r: &ScanResult) {
    if !r.model_scores.is_empty() {
        let _ = writeln!(
            out,
            "  routes (raw): {}",
            crate::output::format_route_scores(&r.model_scores),
        );
    }
    if !r.reasons.is_empty() {
        let _ = writeln!(out, "  shap (top features by importance):");
        for reason in r.reasons.iter().take(12) {
            let _ = writeln!(
                out,
                "    imp={:.4} val={:.4}  {}  [{}]",
                reason.importance, reason.value, reason.feature, reason.description,
            );
        }
    }
    for ef in &r.embedded_files {
        if !ef.model_scores.is_empty() {
            let _ = writeln!(
                out,
                "  embedded {} ({}): routes (raw) {}",
                ef.path,
                ef.file_type,
                crate::output::format_route_scores(&ef.model_scores),
            );
        }
    }
}

/// Write litmus's `--format tiny` view (machine/LLM-facing, never colored): one
/// ML-verdict line — the gate, calibrated confidence, matched false-positive
/// level — then cleave's annotated context.
pub(crate) fn write_tiny(out: &mut dyn std::io::Write, r: &ScanResult) {
    let class = r.classification.to_string();
    let verdict = if matches!(r.classification, Classification::Benign) {
        format!("ascan {class} confidence={:.3}\n", r.probability)
    } else {
        let fp_level = r.level.map_or_else(|| "-".to_string(), |n| format!("L{n}"));
        format!(
            "ascan {class} confidence={:.3} fp-level={fp_level}\n",
            r.probability,
        )
    };
    let _ = out.write_all(verdict.as_bytes());
    if let Some(llm) = &r.interpretation {
        let _ = out.write_all(format_llm_line(llm, false).as_bytes());
    }
    let _ = out.write_all(r.rendered_context.as_bytes());
}

/// One-line LLM verdict, colored by the blended outcome. Shown under the ML
/// verdict in terminal output when `--interpret` produced a result. `color` is
/// false for `--format tiny` (LLM-facing output is never colored).
pub(crate) fn format_llm_line(llm: &crate::interpret::Interpretation, color: bool) -> String {
    use colored::Colorize;
    if !color {
        if let Some(err) = &llm.error {
            return format!("llm error  {err}\n");
        }
        let review = if llm.review { "  ⚠ review" } else { "" };
        let grade = llm.grade.map_or("?", crate::interpret::LlmGrade::as_str);
        return format!(
            "llm {grade} → {} blended={:.3}{review}  {}\n",
            llm.outcome, llm.blended, llm.interpretation,
        );
    }
    if let Some(err) = &llm.error {
        return format!(
            "llm {}  {}\n",
            "error".truecolor(255, 175, 0),
            err.bright_black(),
        );
    }
    let (r, g, b) = match llm.outcome {
        Classification::Hostile => (215, 95, 95),
        Classification::Suspicious => (255, 175, 0),
        Classification::Benign => (95, 175, 95),
    };
    let outcome = llm.outcome.to_string().truecolor(r, g, b).bold();
    let review = if llm.review {
        "  ⚠ review".truecolor(255, 175, 0).to_string()
    } else {
        String::new()
    };
    let grade = llm.grade.map_or("?", crate::interpret::LlmGrade::as_str);
    format!(
        "llm {} → {outcome} blended={:.3}{review}  {}\n",
        grade.bright_black(),
        llm.blended,
        llm.interpretation.bright_black(),
    )
}

/// Intermediate classification result from the model pipeline.
/// Produced by `classify_report`, consumed when building a `ScanResult`.
pub(crate) struct ClassifiedReport {
    pub(crate) classification: Classification,
    pub(crate) probability: f32,
    pub(crate) threshold: f32,
    /// Level-independent lowest-firing-level marker for the root file. See
    /// [`ScanResult::level`].
    pub(crate) level: Option<i32>,
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
    /// cleave's rendered context view (annotated hex/source block).
    pub(crate) rendered_context: String,
    /// Optional LLM interpretation blended with the ML verdict (`--interpret`).
    pub(crate) interpretation: Option<crate::interpret::Interpretation>,
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
    let findings = report_findings(report);
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
        #[allow(clippy::cast_possible_truncation)]
        let conf = json_alias(f, &["conf", "c"])
            .and_then(serde_json::Value::as_f64)
            .map_or(DEFAULT_TRAIT_CONFIDENCE, |x| x as f32);
        if conf < TRAIT_FLOOR_MIN_CONFIDENCE {
            continue;
        }
        match crit_ordinal(f) {
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
/// model already graded suspicious/hostile is left untouched). Synthetic levels
/// sit above the model grid so the envelope records that this was a trait-floor
/// escalation rather than an ordinary swept model level.
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
        decision.level = Some(i32::from(grid_max) + 1);
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
        decision.level = Some(i32::from(grid_max) + 2);
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
#[allow(clippy::too_many_arguments)] // single authoritative inference path; bundling would just shift the noise.
pub(crate) fn classify_report(
    label: &str,
    mut report: cleave::AnalysisReport,
    ctx: &ExtractContext,
    model: &Model,
    shap: Option<&ShapImportance>,
    cancellation: Option<&Arc<AtomicBool>>,
    embedded_file_limit: Option<usize>,
    tiny_opts: &cleave::output::TinyOpts,
    interpret: Option<&crate::interpret::InterpretConfig>,
    root_path: &Path,
    fetch: crate::fetch::FetchPolicy,
) -> Result<ClassifiedReport> {
    report.finalize();
    // Fetch the external references the analysis surfaced and graft each payload
    // into report.files as a uniform node — after finalize() (which populates
    // files[] and the per-file declared references) and before featurization, so
    // fetched content feeds the verdict like any other file. Off unless the
    // policy selects a kind. The returned edges are attached to report_json below.
    let fetch_edges = crate::fetch::orchestrate(&mut report, root_path, fetch);
    // Drop component/baseline traits no composite fired on before the report is
    // summarized, featurized, and posted. finalize() has already inherited and
    // re-evaluated composites up the whole archive/embedding chain, so stripping
    // here never starves a parent composite of its building blocks. This shrinks
    // the raw report posted to hopper (large archive reports otherwise blow past
    // its body limit) and is the report the model is now featurized from.
    report.strip_unmatched_traits();
    let compact = cleave::types::compact::compact_from_files(&report.files);
    validate_report_references(label, &compact);
    let formula = compact
        .files
        .first()
        .and_then(|file| file.formula.clone())
        .unwrap_or_default();

    let mut report_json = serde_json::to_value(&compact).context("serializing cleave report")?;
    // Attach the fetch edge log at report level (`source_sha256 → content_sha256`
    // per reference). Report-level, not per-file: a fetch is a per-event
    // observation, so it never falsely dedups when content is exploded by hash.
    if !fetch_edges.is_empty()
        && let Some(obj) = report_json.as_object_mut()
    {
        obj.insert(
            "fetched".to_string(),
            serde_json::to_value(&fetch_edges).unwrap_or_default(),
        );
    }
    // Parse the report once (with the needs of the general pass and every route),
    // then share it — each route differs only in which features it writes, not in
    // how the report is summarized. The same union covers embedded members below.
    let needs = ctx.raw_needs().union(model.route_needs_union());
    let parsed = crate::features::ParsedReport::from_report(&report_json, needs);
    let mut raw_features = ctx.extract_from_parsed(&parsed);
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
    let pf = json_alias_array(&report_json, &["files", "fs"])
        .and_then(|a| a.first())
        .unwrap_or(&report_json);
    let file_type = pf["type"].as_str().unwrap_or("unknown").to_string();
    let size_bytes = json_alias(pf, &["size", "sz"])
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let sha256 = pf["sha"].as_str().unwrap_or("").to_string();

    let (mut decision, model_scores, skipped_models) =
        model.predict_for_report_detailed(&file_type, &raw_features, &parsed)?;

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
    let embedded_iter = json_alias_array(&report_json, &["files", "fs"])
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

        let ef_parsed = crate::features::ParsedReport::from_file(ef, needs);
        let mut ef_features = ctx.extract_from_parsed(&ef_parsed);
        model.spec().standardize(&mut ef_features);
        let ef_type = ef["type"].as_str().unwrap_or("unknown");
        let (mut ef_decision, ef_model_scores, ef_skipped_models) = model
            .predict_for_file_detailed(ef_type, &ef_features, &ef_parsed)
            .unwrap_or((
                Decision {
                    class: Classification::Benign,
                    probability: 0.0,
                    threshold: model.thresholds().suspicious,
                    level: None,
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

        let ef_top_findings: Vec<TopFinding> = json_alias_array(ef, &["find", "ts"])
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
            level: ef_decision.level,
            model_scores: ef_model_scores,
            skipped_models: ef_skipped_models,
            formula: json_alias_str(ef, &["mol", "f"]).unwrap_or("").to_string(),
            top_findings: ef_top_findings,
        });
    }

    embedded_files.sort_by(|a, b| b.probability.total_cmp(&a.probability));
    if embedded_file_limit.is_some() {
        embedded_files.truncate(10);
    }

    // If an embedded file's decision outranks the parent, elevate.
    let mut final_decision = if decision_outranks(&max_decision, &decision) {
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

    // LLM second opinion (root file only). Render the full cleave tiny context
    // for the model regardless of the display density, then blend.
    let interpretation = interpret.and_then(|cfg| {
        if final_decision.probability < cfg.min_prob {
            tracing::debug!(
                path = %label,
                probability = format!("{:.4}", final_decision.probability),
                cutoff = format!("{:.4}", cfg.min_prob),
                "below --interpret-min-prob cutoff; skipping LLM interpretation",
            );
            return None;
        }
        let llm_ctx = crate::interpret::sanitize_context(&cleave::output::format_context(
            &report,
            &cleave::output::TinyOpts::tiny(),
        ));
        let interp = crate::interpret::interpret(
            cfg,
            &llm_ctx,
            final_decision.class,
            final_decision.probability,
        )?;
        if let Some(grade) = interp.grade {
            tracing::info!(
                file = %label,
                sha256 = %sha256,
                grade = grade.as_str(),
                outcome = %interp.outcome,
                conf = format!("{:.4}", interp.blended),
                review = interp.review,
                interpretation = %interp.interpretation,
                "fetched LLM interpretation",
            );
        }
        Some(interp)
    });

    // Adopt the blended verdict as the effective one when the LLM out-read ML
    // (escalating a missed threat, or clearing an ML false positive). The `ml`
    // section reflects litmus's final answer; the LLM's raw grade + rationale
    // stay in the `llm` section. A synthetic "interpreted" level surfaces an
    // escalation (L100 hostile / L250 suspicious) or clears a benign (L-1).
    if let Some(interp) = &interpretation
        && interp.grade.is_some()
        && interp.outcome as u8 != final_decision.class as u8
    {
        tracing::warn!(
            path = %label,
            ml = ?final_decision.class,
            outcome = ?interp.outcome,
            grade = interp.grade.map_or("?", crate::interpret::LlmGrade::as_str),
            conf = format!("{:.4}", interp.blended),
            review = interp.review,
            reason = %interp.interpretation,
            "LLM interpretation shifted the verdict",
        );
        final_decision.class = interp.outcome;
        final_decision.probability = interp.blended;
        final_decision.level = Some(interpreted_level(interp.outcome));
    }

    // Render cleave's context view now, while the typed (finalized) report is in
    // scope and the verdict (incl. any interpretation) is known. The terminal
    // view extends cleave's rich render with litmus's verdict badge + subtitle;
    // `--format tiny` uses cleave's machine render verbatim.
    let rendered_context = if tiny_opts.header == cleave::output::HeaderStyle::Rich {
        let (badge, badge_w) = crate::output::terminal_badge(
            &final_decision.class,
            final_decision.probability,
            final_decision.threshold,
            final_decision.level,
        );
        // The filename starts after the stamp and one separator space.
        let indent = badge_w + 1;
        let trailer = crate::output::terminal_trailer(&reasons, interpretation.as_ref());
        let subtitle = crate::output::terminal_subtitle(&sha256, indent);
        let adorn = cleave::output::HeaderBadge {
            badge: Some(&badge),
            trailer: trailer.as_deref(),
            subtitle: subtitle.as_deref(),
        };
        let body = cleave::output::format_context_badged(&report, tiny_opts, adorn);
        if body.is_empty() {
            // No notable+ traits to anchor a header on: still surface the
            // verdict headline so a flagged file is never silent.
            let trail = trailer.map(|t| format!(" {t}")).unwrap_or_default();
            let mut head = format!("{badge} {label}{trail}\n");
            if let Some(sub) = &subtitle {
                head.push_str(sub);
                head.push('\n');
            }
            head
        } else {
            body
        }
    } else {
        cleave::output::format_context(&report, tiny_opts)
    };

    Ok(ClassifiedReport {
        classification: final_decision.class,
        probability: final_decision.probability,
        threshold: final_decision.threshold,
        level: final_decision.level,
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
        rendered_context,
        interpretation,
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
        &tiny_opts_for(config),
        config.interpret(),
        path,
        config.fetch_policy(),
    )?;
    let is_json = matches!(config.format(), OutputFormat::Json);

    // Include raw cleave report for JSON output (unmutated — ML scores go in the ml section).
    let cleave = if is_json { Some(cr.report_json) } else { None };

    Ok(ScanResult {
        v: "7",
        classification: cr.classification,
        probability: cr.probability,
        threshold: cr.threshold,
        level: cr.level,
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
        rendered_context: cr.rendered_context,
        interpretation: cr.interpretation,
    })
}

/// Count cleave findings by criticality level from either a top-level report or
/// the primary file entry inside that report.
#[must_use]
pub fn count_findings_from_json(report: &serde_json::Value) -> FindingCounts {
    let findings = report_findings(report);

    let Some(findings) = findings else {
        return FindingCounts::default();
    };

    let mut counts = FindingCounts::default();
    for f in findings {
        match crit_ordinal(f) {
            5 => counts.hostile += 1,
            4 => counts.suspicious += 1,
            3 => counts.notable += 1,
            _ => counts.baseline += 1,
        }
    }
    counts
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
    let findings = report_findings(report).map(Vec::as_slice).unwrap_or(&[]);

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

/// Build a compact model version string from ModelInfo.
/// Format: "v{spec_version}.{abi_version}-{sha256_prefix}" or with commit if available.
pub(crate) fn model_version_string(info: &crate::model::ModelInfo) -> String {
    if info.sha256.is_empty() {
        return format!("v{}.{}", info.version, info.abi_version);
    }
    // get(..8) yields None when the string is shorter than 8 bytes.
    let sha_prefix = info.sha256.get(..8).unwrap_or(&info.sha256);
    match &info.commit {
        Some(commit) => format!(
            "v{}.{}-{}-{}",
            info.version, info.abi_version, sha_prefix, commit
        ),
        None => format!("v{}.{}-{}", info.version, info.abi_version, sha_prefix),
    }
}

/// Verify every cross-reference in the compact report resolves to an emitted
/// file. A finding's `from[].file` entries index into `files[]`; if a member is
/// ever dropped or renumbered without remapping
/// these, the index dangles and the trait renders downstream (hopper/prism) with
/// no file context — the silent defect that left older reports with orphaned
/// composites. We can't repair it here, but a producer-side log turns it into a
/// visible signal instead of a mystery on the rendering side. The check is a
/// HashSet membership scan over findings — negligible beside model inference.
fn validate_report_references(label: &str, report: &cleave::types::compact::CompactReport) {
    let ids: std::collections::HashSet<u32> = report.files.iter().map(|f| f.id).collect();
    let mut dangling = 0usize;
    let mut sample: Vec<String> = Vec::new();
    let mut note = |trait_id: &str, missing: u32| {
        dangling += 1;
        if sample.len() < 3 {
            sample.push(format!("{trait_id}->#{missing}"));
        }
    };
    for file in &report.files {
        for finding in &file.findings {
            // v8 merged the old `src` (inherited single source) and `sources[]`
            // (cross-file composite members) into one `from: Vec<CompactSource>`.
            for s in &finding.from {
                if !ids.contains(&s.file) {
                    note(&finding.id, s.file);
                }
            }
        }
    }
    if dangling > 0 {
        tracing::error!(
            label,
            dangling,
            files = report.files.len(),
            examples = %sample.join(", "),
            "compact report integrity: cross-file references point at file ids not in files[]; \
             affected traits will render without file context downstream"
        );
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

/// Build per-file ML classification entries for the `ml.files` array.
///
/// Each entry is `{id, type, prob, lvl, conf}` keyed by the cleave `files[].id` field. The root
/// file (`dp=0`) carries the envelope's probability and `lvl`; embedded archive
/// members are matched by path suffix and report their *own* probability and
/// lowest-firing-level `lvl` — every row's `lvl` is therefore the level-independent
/// marker for that specific file. A member with no recorded evaluation (e.g.
/// truncated past the embedded-file cap) falls back to the root values.
fn build_ml_files(
    report_json: &serde_json::Value,
    root_prob: f32,
    root_level: Option<i32>,
    embedded_files: &[EmbeddedFile],
) -> Vec<serde_json::Value> {
    let Some(report_files) = json_alias_array(report_json, &["files", "fs"]) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(report_files.len());
    for (idx, entry) in report_files.iter().enumerate() {
        let id = entry
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(idx as u64);
        let depth = entry
            .get("dp")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let file_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");

        let (prob, file_level) = if depth == 0 {
            (root_prob, root_level)
        } else {
            let path = entry.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let suffix = path.rsplit_once("!!").map(|(_, r)| r).unwrap_or(path);
            embedded_files
                .iter()
                .find(|ef| ef.path == suffix)
                .map(|ef| (ef.probability, ef.level))
                .unwrap_or((root_prob, root_level))
        };

        out.push(serde_json::json!({
            "id": id,
            "type": file_type,
            "prob": prob,
            "lvl": file_level,
            "conf": level_confidence(file_level),
        }));
    }
    out
}

/// Top-level JSON envelope: `{"ml": {...}, "raw": {...}}`.
#[derive(Debug, serde::Serialize)]
pub struct ScanResultEnvelope {
    /// ML classification section.
    pub ml: MlSection,
    /// LLM interpretation section (`--interpret`); omitted when not run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm: Option<crate::interpret::Interpretation>,
    /// Raw cleave analysis report.
    pub raw: serde_json::Value,
}

/// This scan binary's build version, stamped into every `ml` envelope so a
/// stored result records which engine produced it. Distinct from `version` (the
/// ML model) and `raw.tv` (the traits-repo commit); together they pin the build
/// that generated a report, which is otherwise unrecoverable from the JSON.
pub(crate) const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The `ml` section of the response envelope.
#[derive(Debug, serde::Serialize)]
pub struct MlSection {
    pub(crate) v: &'static str,
    #[serde(rename = "prob")]
    pub(crate) probability: f32,
    /// Resolved verdict marker, always serialized (including as `null`):
    /// - `Some(-1)` → benign.
    /// - `Some(0..=25000)` → hostile; the per-100M-benigns level that selected
    ///   the firing threshold.
    /// - `None` → hostile; manual `--threshold-hostile` / `--threshold-suspicious`
    ///   were used and no level applies.
    #[serde(rename = "lvl")]
    pub(crate) level: Option<i32>,
    /// Pessimistic integer confidence percent derived from `level`; `null` when no
    /// level table applies (manual-threshold mode).
    pub(crate) conf: Option<u8>,
    #[serde(rename = "mods", skip_serializing_if = "Vec::is_empty")]
    pub(crate) model_scores: Vec<crate::model::RouteScore>,
    #[serde(rename = "skip", skip_serializing_if = "Vec::is_empty")]
    pub(crate) skipped_models: Vec<crate::model::SkippedRoute>,
    pub(crate) version: String,
    /// Scan engine build (`CARGO_PKG_VERSION`) that produced this report.
    pub(crate) eng: &'static str,
    pub(crate) analyzed_at: String,
    #[serde(rename = "files")]
    pub(crate) files: Vec<serde_json::Value>,
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
    /// LLM interpretation section (`--interpret`); omitted when not run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm: Option<&'a crate::interpret::Interpretation>,
    /// Raw cleave analysis report (borrowed).
    pub raw: &'a serde_json::Value,
}

/// Borrowed counterpart of [`MlSection`].
#[derive(Debug, serde::Serialize)]
pub struct MlSectionRef<'a> {
    pub(crate) v: &'static str,
    #[serde(rename = "prob")]
    pub(crate) probability: f32,
    /// See [`MlSection::level`] for the encoding of this field.
    #[serde(rename = "lvl")]
    pub(crate) level: Option<i32>,
    /// See [`MlSection::conf`].
    pub(crate) conf: Option<u8>,
    #[serde(rename = "mods", skip_serializing_if = "route_scores_empty")]
    pub(crate) model_scores: &'a [crate::model::RouteScore],
    #[serde(rename = "skip", skip_serializing_if = "skipped_routes_empty")]
    pub(crate) skipped_models: &'a [crate::model::SkippedRoute],
    pub(crate) version: &'a str,
    /// Scan engine build (`CARGO_PKG_VERSION`) that produced this report.
    pub(crate) eng: &'static str,
    pub(crate) analyzed_at: &'a str,
    #[serde(rename = "files")]
    pub(crate) files: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pids: Option<&'a [u32]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) deleted: Option<bool>,
}

impl ScanResult {
    /// Build the `{"ml": {...}, "raw": {...}}` envelope for JSON output.
    ///
    /// Prefer [`Self::into_envelope`] when the caller owns the `ScanResult`.
    /// `to_envelope` clones the full `cleave` report (potentially multi-MB
    /// for archive analyses); the move variant avoids that entirely.
    #[must_use]
    pub fn to_envelope(&self) -> ScanResultEnvelope {
        let raw = self.cleave.clone().unwrap_or(serde_json::json!({}));
        let level = self.level;
        let ml_files = build_ml_files(&raw, self.probability, level, &self.embedded_files);
        ScanResultEnvelope {
            ml: MlSection {
                v: self.v,
                probability: self.probability,
                level,
                conf: level_confidence(level),
                model_scores: self.model_scores.clone(),
                skipped_models: self.skipped_models.clone(),
                version: self.version.clone(),
                eng: ENGINE_VERSION,
                analyzed_at: self.analyzed_at.clone(),
                files: ml_files,
                pids: self.pids.clone(),
                deleted: self.deleted,
            },
            llm: self.interpretation.clone(),
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
        let level = self.level;
        let ml_files = build_ml_files(&raw, self.probability, level, &self.embedded_files);
        ScanResultEnvelope {
            ml: MlSection {
                v: self.v,
                probability: self.probability,
                level,
                conf: level_confidence(level),
                model_scores: self.model_scores,
                skipped_models: self.skipped_models,
                version: self.version,
                eng: ENGINE_VERSION,
                analyzed_at: self.analyzed_at,
                files: ml_files,
                pids: self.pids,
                deleted: self.deleted,
            },
            llm: self.interpretation,
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
        let level = self.level;
        let ml_files = build_ml_files(raw, self.probability, level, &self.embedded_files);
        ScanResultEnvelopeRef {
            ml: MlSectionRef {
                v: self.v,
                probability: self.probability,
                level,
                conf: level_confidence(level),
                model_scores: &self.model_scores,
                skipped_models: &self.skipped_models,
                version: &self.version,
                eng: ENGINE_VERSION,
                analyzed_at: &self.analyzed_at,
                files: ml_files,
                pids: self.pids.as_deref(),
                deleted: self.deleted,
            },
            llm: self.interpretation.as_ref(),
            raw,
        }
    }
}
