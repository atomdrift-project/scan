//! Scanning and classification: the CLI scan drivers ([`run`], [`run_paths`],
//! [`run_bytes`]), the pipeline every front end classifies through
//! (`classify_report`), and the result and JSON envelope it produces.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;

use crate::OutputFormat;
use crate::bloom_repo::Lookup;
use crate::model::{Classification, Thresholds};

pub use crate::explain::Reason;

mod envelope;
mod hopper;
mod pipeline;
mod progress;
mod render_cards;
mod render_context;
mod retention;
mod run;
mod verdict;

pub use self::envelope::{
    DepResult, EmbeddedFile, HopperRoute, MemberEvals, MlSection, ScanResult, ScanResultEnvelope,
    ScanResultEnvelopeRef, TopFinding,
};
pub use self::pipeline::{PendingLlm, count_findings, extract_top_findings};
pub use self::run::{run, run_bytes, run_paths, scan_bytes, scan_file};
pub use self::verdict::{
    FloorArm, FloorDecision, level_confidence, synthesized_level, trait_floor,
};

pub(crate) use self::envelope::dep_envelope;
pub(crate) use self::hopper::{
    Origin, artifact_filename, collect_upload_artifacts, registry_fallback_artifact,
    upload_collector, upload_scan_result,
};
pub(crate) use self::pipeline::{
    ClassifiedReport, ClassifyRequest, CpuLease, OutputNeeds, apply_pending_interpretation,
    classify_report,
};
pub(crate) use self::progress::{
    PROGRESS_TICK, Progress, SPINNER, bar_active, external_fetch_finished, external_fetch_started,
    print_above_bar, term_dims,
};
pub(crate) use self::render_cards::{format_llm_line, write_tiny};
pub(crate) use self::render_context::recategorize_annotations;
pub(crate) use self::run::{
    DetectionCounts, bloom_gate_fresh, detection_counts, detection_counts_from, tiny_opts_for,
};

use self::render_context::INTERPRET_PRIMARY_BUDGET_BYTES;
use self::retention::UNANALYZED_MEMBER_RISK;

/// Terminal display policy for scan results.
///
/// This affects human-readable output only. JSON output still emits every
/// scanned file so downstream consumers receive a complete event stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayFilter {
    /// Show hostile files.
    pub hostile: bool,
    /// Show suspicious files.
    pub suspicious: bool,
    /// Show benign files.
    pub benign: bool,
}

impl DisplayFilter {
    /// Include only hostile and suspicious files.
    #[must_use]
    pub const fn alerts_only() -> Self {
        Self {
            hostile: true,
            suspicious: true,
            benign: false,
        }
    }

    /// Include every classification.
    #[must_use]
    pub const fn all() -> Self {
        Self {
            hostile: true,
            suspicious: true,
            benign: true,
        }
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

    /// Returns true when every classification is admitted (`--show=all`). This is
    /// the cue to emit a complete archive manifest in JSON output — every member,
    /// including the ones cleave never analyzed because they carry no findings.
    #[must_use]
    pub fn is_all(&self) -> bool {
        self.hostile && self.suspicious && self.benign
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
    hopper: Option<String>,
    zip_passwords: crate::ArchivePasswords,
    mode: crate::Mode,
    bloom: Option<Arc<Lookup>>,
}

static RIZIN_TIMEOUT: OnceLock<Duration> = OnceLock::new();

/// Fix the wall-clock limit (`--rizin-timeout-secs`) on each Rizin run of every
/// analysis this process starts. Call once at startup, before any analysis; a
/// later call is ignored.
pub fn set_rizin_timeout(timeout: Duration) {
    let _ = RIZIN_TIMEOUT.set(timeout);
}

/// The Rizin limit for an [`cleave::AnalysisOptions`]: the configured one, or
/// `None` (cleave's default) when the process never set it.
pub(crate) fn rizin_timeout() -> Option<Duration> {
    RIZIN_TIMEOUT.get().copied()
}

pub(crate) fn add_zip_passwords(options: &mut cleave::AnalysisOptions, passwords: &[String]) {
    for password in passwords {
        if !options
            .zip_passwords
            .iter()
            .any(|existing| existing == password)
        {
            options.zip_passwords.push(password.clone());
        }
    }
}

impl ScanConfig {
    /// Create a scan configuration that shows alerts only, with the default
    /// slow-rule warning and no extra diagnostics; the `with_*` methods
    /// change the rest.
    ///
    /// `thresholds` may be `None` to use the model's recommended thresholds
    /// from `evaluation.json`, or `Some(t)` to override with explicit values.
    ///
    /// # Example
    /// ```
    /// use scan::{Classification, OutputFormat, ScanConfig};
    ///
    /// let config = ScanConfig::new("/path/to/models", OutputFormat::Terminal, None)?;
    ///
    /// assert_eq!(config.format(), OutputFormat::Terminal);
    /// assert!(config.filter().shows(&Classification::Hostile));
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        model_dir: impl Into<PathBuf>,
        format: OutputFormat,
        thresholds: Option<Thresholds>,
    ) -> Result<Self> {
        if let Some(ref t) = thresholds {
            t.validate()
                .map_err(|error| anyhow::anyhow!("invalid thresholds: {error}"))?;
        }
        Ok(Self {
            model_dir: model_dir.into(),
            format,
            thresholds,
            filter: DisplayFilter::alerts_only(),
            slow_rule_ms: crate::cli::DEFAULT_SLOW_RULE_MS,
            extra: false,
            level: None,
            interpret: None,
            fetch: crate::fetch::FetchPolicy::default(),
            hopper: None,
            zip_passwords: crate::ArchivePasswords::default(),
            // Bloom short-circuiting is opt-in via `with_bloom`; an unconfigured
            // config runs a full scan (slow mode), so server/fs paths are unaffected.
            mode: crate::Mode::Slow,
            bloom: None,
        })
    }

    /// Which classifications terminal output shows (`--show`).
    #[must_use]
    pub const fn with_filter(mut self, filter: DisplayFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Warn about any single cleave rule slower than `ms`. Advisory logging
    /// only; it does not cancel analysis.
    #[must_use]
    pub const fn with_slow_rule_ms(mut self, ms: u64) -> Self {
        self.slow_rule_ms = ms;
        self
    }

    /// Print extra per-file diagnostics (`--extra`).
    #[must_use]
    pub const fn with_extra(mut self, extra: bool) -> Self {
        self.extra = extra;
        self
    }

    /// Upload (renew) each scan result on the hopper instance at `url` by POSTing
    /// its envelope to `/api/result`. `None` (default) disables uploading. Used by
    /// `scan path --hopper`; failures are reported as errors but never affect the
    /// scan's outcome.
    #[must_use]
    pub fn with_hopper(mut self, url: Option<String>) -> Self {
        self.hopper = url;
        self
    }

    /// Add passwords to try when cleave encounters encrypted archives.
    /// Cleave's built-in common sample passwords remain enabled.
    #[must_use]
    pub fn with_zip_passwords(mut self, passwords: impl Into<crate::ArchivePasswords>) -> Self {
        self.zip_passwords = passwords.into();
        self
    }

    /// Additional archive passwords supplied by the caller.
    #[must_use]
    pub(crate) fn zip_passwords(&self) -> &[String] {
        self.zip_passwords.as_slice()
    }

    /// Hopper base URL to renew results on, or `None` when uploading is disabled.
    #[must_use]
    pub(crate) fn hopper(&self) -> Option<&str> {
        self.hopper.as_deref()
    }

    /// Set the external-reference fetch policy: which kinds of reference
    /// (registry packages, bare URLs) discovered in analyzed files to fetch,
    /// re-analyze, and graft into the report. The default [`FetchPolicy`](crate::fetch::FetchPolicy)
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
    /// `severity_levels[]` table. It moves the hostile/suspicious cutoffs only:
    /// the envelope's `ml.lvl` is each file's own fired level, independent of it.
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

    /// Enable bloom-filter short-circuiting: `mode` selects how aggressively the
    /// local known-good/known-bad filters are consulted, and `lookup` carries the
    /// loaded filters. Unset, a config stays in [`crate::Mode::Slow`] (full scan).
    #[must_use]
    pub fn with_bloom(mut self, mode: crate::Mode, lookup: Arc<Lookup>) -> Self {
        self.mode = mode;
        self.bloom = Some(lookup);
        self
    }

    /// The scan execution mode (defaults to [`crate::Mode::Slow`]).
    #[must_use]
    pub const fn mode(&self) -> crate::Mode {
        self.mode
    }

    /// The loaded bloom filters, or `None` in slow mode / when none are synced.
    #[must_use]
    pub(crate) fn bloom(&self) -> Option<&Lookup> {
        self.bloom.as_deref()
    }

    /// A shared handle to the bloom filters, for the cleave skip predicate (which
    /// must be `'static`). `None` when bloom is disabled.
    #[must_use]
    pub(crate) fn bloom_arc(&self) -> Option<Arc<Lookup>> {
        self.bloom.clone()
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
#[derive(Debug, Clone, Default, serde::Serialize, PartialEq)]
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

/// Warm cleave's YARA engine and capability mapper from a non-rayon thread,
/// so the first analysis cannot race a rayon worker into the one-time init.
/// Every scan entry point calls this before its first analysis.
///
/// It never asks cleave for its regex prewarm. That compiles every pattern in
/// cleave's warm memo, the union of everything any scan sharing the cache has
/// ever compiled (~79k programs on 2026-09-25), when a scan only needs the few
/// its file types reach, compiled lazily on first use. Measured that day on a
/// quiet host: a one-shot 12-byte file went from 4.9 GiB peak and 44-48 s CPU
/// to 0.8 GiB and 7-8 s, an 80 KB npm package from 4.9 GiB and ~50 s CPU to
/// 1.5 GiB and 14 s, both faster in wall. A long-lived server answering 100
/// mixed files, sequentially or four at a time, saw no first-request or
/// throughput gain from it, only 2.6-2.8 GiB more peak RSS.
pub fn prefetch_cleave_resources() {
    cleave::prefetch_shared_resources_with(true, false);
}

/// The model version recorded on every result: `v{spec_version}.{abi_version}`.
pub(crate) fn model_version_string(info: &crate::model::ModelInfo) -> String {
    format!("v{}.{}", info.version, info.abi_version)
}

/// The current time as an RFC 3339 string in UTC (`2026-10-01T13:02:22Z`).
pub(crate) fn now_rfc3339() -> String {
    crate::civil::rfc3339(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
    )
}

/// This scan build, stamped into every `ml` envelope (`eng`) so a stored result
/// records which engine produced it. Distinct from `version` (the ML model) and
/// `raw.tv` (the traits-repo commit); together they pin the build behind a report.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The `ml.v` envelope schema this build writes.
pub(crate) const SCHEMA_VERSION: &str = "7";

/// A compact finding's criticality. The compact form stores
/// [`cleave::Criticality::rank`]; this reads it back.
fn compact_crit(finding: &cleave::types::CompactTrait) -> cleave::Criticality {
    use cleave::Criticality;
    match finding.criticality {
        0 => Criticality::Filtered,
        1 => Criticality::Component,
        2 => Criticality::Baseline,
        3 => Criticality::Notable,
        4 => Criticality::Suspicious,
        _ => Criticality::Hostile,
    }
}

/// The delimiter cleave writes between archive layers in a report path
/// (`root.zip!!member.tgz!!inner/file`). Directory separators inside one layer
/// stay `/`.
const ARCHIVE_DELIMITER: &str = "!!";

/// The innermost layer of an archive path: the member's path inside its own
/// container. A path with no layers is returned whole.
fn archive_leaf(path: &str) -> &str {
    path.rsplit(ARCHIVE_DELIMITER).next().unwrap_or(path)
}

/// Whether `path` lies inside the archive at `container`, at any depth.
fn is_inside(path: &str, container: &str) -> bool {
    path.strip_prefix(container)
        .is_some_and(|rest| rest.starts_with(ARCHIVE_DELIMITER))
}

/// Engine settings an operator sets through the environment. They are
/// deploy-time switches, not request input, so they are read once per process.
#[derive(Debug, Clone)]
pub(crate) struct Tuning {
    /// `SCAN_INTERPRET_DUMP_DIR`: also write each LLM render to
    /// `<dir>/<sha256>.render`, for offline prompt tuning (`hacks/interpret-tune`).
    pub(crate) interpret_dump_dir: Option<PathBuf>,
    /// `SCAN_KEEP_ALL_MEMBERS=1`: store whole reports, skipping retention.
    pub(crate) keep_all_members: bool,
    /// `SCAN_INTERPRET_BUDGET_BYTES`: the primary subject's render budget for
    /// the LLM; `0` disables the cap.
    pub(crate) interpret_budget_bytes: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            interpret_dump_dir: None,
            keep_all_members: false,
            interpret_budget_bytes: INTERPRET_PRIMARY_BUDGET_BYTES,
        }
    }
}

impl Tuning {
    fn from_env() -> Self {
        let default = Self::default();
        Self {
            interpret_dump_dir: std::env::var_os("SCAN_INTERPRET_DUMP_DIR").map(PathBuf::from),
            keep_all_members: std::env::var("SCAN_KEEP_ALL_MEMBERS").as_deref() == Ok("1"),
            interpret_budget_bytes: std::env::var("SCAN_INTERPRET_BUDGET_BYTES")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(default.interpret_budget_bytes),
        }
    }

    /// This process's settings, read from the environment on first use.
    pub(crate) fn get() -> &'static Self {
        static TUNING: OnceLock<Tuning> = OnceLock::new();
        TUNING.get_or_init(Self::from_env)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_all_only_when_every_class_shown() {
        assert!(DisplayFilter::all().is_all());
        assert!(!DisplayFilter::alerts_only().is_all());
        assert!(
            !DisplayFilter {
                benign: false,
                ..DisplayFilter::all()
            }
            .is_all()
        );
    }

    #[test]
    fn scan_config_rejects_invalid_thresholds() {
        let result = ScanConfig::new(
            "/tmp/models",
            OutputFormat::Terminal,
            Some(Thresholds {
                suspicious: 0.99,
                hostile: 0.50,
            }),
        );

        assert!(result.is_err());
    }

    #[test]
    fn scan_config_level_defaults_to_none() {
        let config =
            ScanConfig::new("/tmp/models", OutputFormat::Terminal, None).expect("valid config");
        assert!(config.level().is_none());
    }

    #[test]
    fn scan_config_with_level_persists() {
        let config = ScanConfig::new("/tmp/models", OutputFormat::Terminal, None)
            .expect("valid config")
            .with_level(Some(7));
        assert_eq!(config.level(), Some(7));
    }

    #[test]
    fn archive_passwords_extend_defaults_without_duplicates() {
        let mut options = cleave::AnalysisOptions::default();
        let default_password = options.zip_passwords[0].clone();

        add_zip_passwords(
            &mut options,
            &[default_password.clone(), "private".into(), "private".into()],
        );

        assert_eq!(
            options
                .zip_passwords
                .iter()
                .filter(|password| *password == &default_password)
                .count(),
            1
        );
        assert_eq!(
            options
                .zip_passwords
                .iter()
                .filter(|password| password.as_str() == "private")
                .count(),
            1
        );
    }
}
