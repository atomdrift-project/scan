//! Recursive file-system scanning and classification.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::Mode;
use crate::OutputFormat;
use crate::bloom_repo::{Decision as BloomDecision, Lookup};
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
    mode: crate::Mode,
    bloom: Option<Arc<Lookup>>,
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
            hopper: None,
            // Bloom short-circuiting is opt-in via `with_bloom`; an unconfigured
            // config runs a full scan (slow mode), so server/fs paths are unaffected.
            mode: crate::Mode::Slow,
            bloom: None,
        })
    }

    /// Upload (renew) each scan result on the hopper instance at `url` by POSTing
    /// its envelope to `/api/result`. `None` (default) disables uploading. Used by
    /// `scan path --hopper`; failures degrade to logged warnings and never affect
    /// the scan's outcome.
    #[must_use]
    pub fn with_hopper(mut self, url: Option<String>) -> Self {
        self.hopper = url;
        self
    }

    /// Hopper base URL to renew results on, or `None` when uploading is disabled.
    #[must_use]
    pub(crate) fn hopper(&self) -> Option<&str> {
        self.hopper.as_deref()
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

    /// Enable bloom-filter short-circuiting: `mode` selects how aggressively the
    /// local known-good/known-bad filters are consulted, and `lookup` carries the
    /// loaded filters. Unset, a config stays in [`crate::Mode::Slow`] (full scan).
    #[must_use]
    pub fn with_bloom(mut self, mode: crate::Mode, lookup: Lookup) -> Self {
        let lookup = Arc::new(lookup);
        // Publish process-wide so the dependency-fetch reporter can label deps
        // good/bad without threading config into the fetch subsystem.
        crate::bloom_repo::set_global(Arc::clone(&lookup));
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod manifest_tests {
    use super::*;

    fn stub(path: &str, ty: &str, sha: &str, size: u64) -> ArchiveMemberStub {
        ArchiveMemberStub {
            path: path.to_string(),
            file_type: ty.to_string(),
            sha256: sha.to_string(),
            size_bytes: size,
        }
    }

    #[test]
    fn appends_unanalyzed_members_skipping_already_analyzed() {
        // files[0] is the root archive; files[1] an analyzed member with a sha.
        let mut report = serde_json::json!({
            "files": [
                {"id": 0, "path": "app.zip", "type": "zip", "sha": "root", "size": 4096},
                {"id": 1, "path": "app.zip!!evil.sh", "type": "shell", "sha": "aaa", "size": 200, "risk": 9},
            ]
        });
        let members = vec![
            // Already analyzed (matches sha "aaa") — must be skipped, no duplicate.
            stub("evil.sh", "shell", "aaa", 200),
            // Never analyzed — must be appended as a listing-only entry.
            stub("README.md", "markdown", "bbb", 1024),
            // Nested member: single `!` becomes `!!`, depth counts the levels.
            stub("inner.tar!logo.png", "png", "ccc", 8192),
        ];
        append_unanalyzed_members(&mut report, &members);

        let files = report["files"].as_array().unwrap();
        assert_eq!(files.len(), 4, "two listing-only members appended");

        let readme = &files[2];
        assert_eq!(readme["id"], 2);
        assert_eq!(readme["path"], "app.zip!!README.md");
        assert_eq!(readme["type"], "markdown");
        assert_eq!(readme["size"], 1024);
        assert_eq!(readme["depth"], 1);
        assert_eq!(readme["risk"], -1, "sentinel marks the member unanalyzed");

        let logo = &files[3];
        assert_eq!(logo["path"], "app.zip!!inner.tar!!logo.png");
        assert_eq!(logo["depth"], 2);
        assert_eq!(logo["risk"], -1);
    }

    #[test]
    fn build_ml_files_drops_listing_only_members() {
        // A root file, an analyzed member, and a listing-only member (risk -1).
        let report = serde_json::json!({
            "files": [
                {"id": 0, "path": "app.zip", "type": "zip", "depth": 0},
                {"id": 1, "path": "app.zip!!evil.sh", "type": "shell", "depth": 1, "risk": 9},
                {"id": 2, "path": "app.zip!!README.md", "type": "markdown", "depth": 1, "risk": -1},
            ]
        });
        let ml = build_ml_files(&report, 0.9, Some(100), &[]);
        let ids: Vec<u64> = ml.iter().map(|f| f["id"].as_u64().unwrap()).collect();
        assert_eq!(
            ids,
            vec![0, 1],
            "listing-only member is excluded from ml.files"
        );
    }

    #[test]
    fn is_all_only_when_every_class_shown() {
        assert!(DisplayFilter::all().is_all());
        assert!(!DisplayFilter::alerts_only().is_all());
        assert!(!DisplayFilter::new(true, true, false).is_all());
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

    #[test]
    fn file_touched_within_flags_fresh_and_ignores_old_and_missing() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("x.txt");
        std::fs::write(&f, b"hi").unwrap();
        let now = SystemTime::now();
        // Just-written file is within the window.
        assert!(file_touched_within(&f, KNOWN_GOOD_RESCAN_SECS, now));
        // Evaluated from a clock 10 days ahead, every timestamp is well outside
        // the 48h window — the not-recent (skip-eligible) case.
        let later = now + Duration::from_secs(10 * 86_400);
        assert!(!file_touched_within(&f, KNOWN_GOOD_RESCAN_SECS, later));
        // Unreadable metadata (e.g. a fetched artifact's synthetic path) is treated
        // as not-recent, so the normal known-good skip still applies.
        assert!(!file_touched_within(
            &dir.path().join("does-not-exist"),
            KNOWN_GOOD_RESCAN_SECS,
            now
        ));
    }

    #[test]
    fn collect_upload_artifacts_offers_root_only() {
        // Fetched dependencies are mirrored separately (bytes + provenance + their
        // own verdict) via the uploader's dependency path, so this offers only the
        // scanned file itself — a local artifact hopper may never have seen.
        let arts =
            collect_upload_artifacts(Path::new("/tmp/proj.tgz"), &"a".repeat(64), 10, "scan+test");

        assert_eq!(arts.len(), 1, "only the scanned file is offered here");
        assert_eq!(arts[0].sha256, "a".repeat(64));
        assert!(matches!(
            arts[0].bytes,
            crate::upload::ArtifactBytes::File(_)
        ));
        assert!(
            !arts[0].backfill,
            "root file's thin sidecar is not backfilled"
        );
        assert_eq!(arts[0].filename, "proj.tgz", "filename is the file's name");
    }

    #[test]
    fn dep_envelope_carries_verdict_and_report() {
        // A dependency's verdict envelope encodes its aggregate level as `ml.lvl`
        // and passes its standalone report through as `raw`, exactly the shape a
        // first-hand scan posts — so hopper records it identically.
        let dep = DepResult {
            sha256: "d".repeat(64),
            locator: "pkg:npm/evil@1.0.0".to_string(),
            url: "https://reg/evil-1.0.0.tgz".to_string(),
            size: 1234,
            level: Some(100),
            probability: 0.97,
            raw: serde_json::json!({"v": "8", "files": [{"sha": "d".repeat(64), "type": "npm"}]}),
        };
        let env = dep_envelope(dep, "model-9", "2026-06-28T00:00:00Z");
        assert_eq!(env.ml.level, Some(100), "aggregate verdict rides in ml.lvl");
        assert_eq!(env.ml.version, "model-9");
        assert_eq!(env.ml.analyzed_at, "2026-06-28T00:00:00Z");
        assert_eq!(
            env.raw["files"][0]["type"], "npm",
            "the dependency's own report is the result raw, so hopper keeps its FileType",
        );
    }

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
            dependency_results: Vec::new(),
            bloom_mark: None,
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
    fn interpreted_escalation_levels_stay_within_their_class_band() {
        use crate::model::{capped_suspicious_level, verdict_for_level};

        let hostile_lvl = interpreted_level(Classification::Hostile);
        let susp_lvl = interpreted_level(Classification::Suspicious);

        // Both are real (non-benign) levels, and a hostile escalation sits
        // strictly tighter than a suspicious one.
        assert!(hostile_lvl >= 0 && susp_lvl >= 0);
        assert!(
            hostile_lvl < susp_lvl,
            "a hostile escalation (L{hostile_lvl}) must be tighter than a suspicious one (L{susp_lvl})"
        );

        // A benign->suspicious escalation must never exceed the suspicious
        // ceiling, or a consumer that reclassifies from `lvl` (e.g. hopper)
        // would drop the escalated verdict to benign.
        let ceiling = capped_suspicious_level(u16::MAX); // == SUSPICIOUS_LEVEL_CEILING
        assert!(
            (susp_lvl as u16) <= ceiling,
            "suspicious escalation L{susp_lvl} must sit within the L{ceiling} ceiling"
        );

        // Across every deploy level we operate at — hopper's L4 critical line,
        // the L5 hostile/suspicious boundary, collimator's L50 default — the
        // synthetic level must reclassify into the class the LLM lifted it to:
        // suspicious->hostile escalations stay inside the hostile spectrum, and
        // benign->suspicious escalations stay inside the suspicious band.
        let grid_max = 25_000;
        for deploy in [4_u16, 5, 50] {
            assert_eq!(
                verdict_for_level(hostile_lvl as u16, deploy, grid_max),
                Classification::Hostile,
                "hostile escalation must read hostile at deploy L{deploy}"
            );
            assert_eq!(
                verdict_for_level(susp_lvl as u16, deploy, grid_max),
                Classification::Suspicious,
                "suspicious escalation must read suspicious at deploy L{deploy}"
            );
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
/// shifts the class away from ML's. An escalation lands at the *weak edge* of the
/// target class so it fits squarely inside it (one step tighter than the boundary
/// for margin): hostile just inside the L5 hostile line (L4), suspicious just
/// inside the L100 suspicious ceiling (L99), benign the clean marker. This keeps
/// the synthetic level in-band for consumers that reclassify from `lvl` (e.g.
/// hopper): L4 reads hostile, L99 reads suspicious under the L100 ceiling — where
/// the old L100/L250 would have mis-read as suspicious / benign respectively.
/// Maps through [`level_confidence`] to ≈95% / ≈85% / 0%; keep the two
/// `*_ESCALATION_CONF` constants in `interpret.rs` in sync.
const fn interpreted_level(outcome: Classification) -> i32 {
    match outcome {
        Classification::Hostile => 4,
        Classification::Suspicious => 99,
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
    /// Fetched dependencies to mirror into hopper as their own samples. Empty
    /// unless the scan fetched dependencies; consumed by the upload paths and
    /// never serialized into this result's own envelope.
    pub dependency_results: Vec<DepResult>,
    /// The bloom status flag for a known-bad/conflicted file: drives the inline
    /// 🚩/🏴 mark in the terminal header and the `bloom=` token on the `--format
    /// tiny` line. `None` for unremarkable files; a terminal-UI concern only, so
    /// it is never serialized into the JSON envelope.
    pub bloom_mark: Option<crate::output::BloomMark>,
}

/// A fetched dependency to mirror into hopper as its own sample: the aggregate
/// verdict scan computed for it during the parent analysis, plus its standalone
/// compact cleave report. Provenance and bytes are recovered from the fetch blob
/// cache at upload time (keyed by `locator`), so they never travel in the result.
#[derive(Debug, Clone)]
pub struct DepResult {
    /// SHA-256 of the dependency's bytes — its identity and `/api/result` key.
    pub sha256: String,
    /// The reference locator (PURL/URL) the bytes were fetched from.
    pub locator: String,
    /// The URL the locator resolved to — drives the stored filename/type sniff.
    pub url: String,
    /// Size of the dependency's bytes, recorded in the provenance sidecar.
    pub size: u64,
    /// Aggregate verdict marker (`ml.lvl`): the dependency's container elevated by
    /// its worst member, exactly as a first-hand scan of the same bytes resolves.
    pub level: Option<i32>,
    /// Probability the aggregate verdict was decided on.
    pub probability: f32,
    /// The dependency's own compact cleave report — the `raw` for its result.
    pub raw: serde_json::Value,
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

/// Heartbeat redraw cadence. A background thread redraws on this interval so
/// the spinner keeps animating — and the long-tail notice can appear — even
/// while no file completes (a single slow rizin analysis can stall for minutes).
const PROGRESS_TICK: Duration = Duration::from_millis(125);

/// Files-remaining threshold below which a stall counts as the "long tail".
const TAIL_FILES: u32 = 4;

/// How long the tail must sit without advancing before we reassure the user
/// that the scan is grinding, not hung (milliseconds).
const TAIL_STALL_MS: u128 = 150;

/// The long-tail reassurance text (plain; colorized at render time).
const TAIL_MESSAGE: &str =
    "\u{2014} on the final long tail of difficult reverse engineering; please be patient\u{2026}";

/// Build the dim, space-prefixed notice that fits within `budget` visible
/// columns, truncating with an ellipsis. Empty when there isn't room for a
/// meaningful slice (a very narrow terminal) — the bar alone still renders.
fn fit_notice(msg: &str, budget: usize) -> String {
    const MIN: usize = 12;
    if budget < MIN {
        return String::new();
    }
    let text = if msg.chars().count() <= budget {
        msg.to_string()
    } else {
        let mut t: String = msg.chars().take(budget - 1).collect();
        t.push('\u{2026}');
        t
    };
    format!(" \x1b[38;2;120;120;120m{text}\x1b[0m")
}

/// Terminal width in columns, for capping the progress line. Falls back to 80
/// when the width can't be queried (not a tty, or the ioctl fails).
fn term_cols() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: TIOCGWINSZ fills a zero-initialised `winsize`; we trust the
        // result only when the ioctl reports success and a non-zero width.
        unsafe {
            let mut ws: libc::winsize = std::mem::zeroed();
            if libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
                return ws.ws_col as usize;
            }
        }
    }
    80
}

/// Shared progress state. Lives behind an `Arc` so the heartbeat thread can hold
/// a clone independent of the `Progress` handle the scan loop borrows.
struct Inner {
    analyzed: AtomicU32,
    total: u32,
    start: Instant,
    /// Elapsed millis at the most recent `increment`; lets `render` measure how
    /// long the count has been frozen.
    last_advance_ms: AtomicU64,
    /// Spinner frame counter, advanced once per render so the spinner animates
    /// on the heartbeat regardless of completions.
    tick: AtomicU32,
    /// Terminal width (columns), sampled once at construction. Caps the
    /// long-tail notice so the bar line never wraps — a wrapped line couldn't be
    /// erased in place by the `\r\x1b[2K` redraw/clear.
    term_cols: usize,
    /// Set when the scan is done: the heartbeat thread exits and no further
    /// render runs, so a late redraw can never clobber the final summary line.
    stopped: AtomicBool,
    /// Serialises renders from the rayon workers and the heartbeat thread; their
    /// `\r`-prefixed writes must never interleave.
    draw_lock: Mutex<()>,
}

/// The active terminal progress bar, published so incidental stderr writers —
/// chiefly the fetch progress block (`report_fetch`/`report_skip` and their
/// header) — can print *above* the bar instead of grafting onto or racing its
/// `\r`-parked line. Only the single-process terminal CLI ever installs a bar;
/// server/JSON modes run none and leave this `None`. A `Weak` so a finished bar
/// is never kept alive; [`print_above_bar`] upgrades it per call.
static ACTIVE_BAR: Mutex<Option<Weak<Inner>>> = Mutex::new(None);

/// Run `print` (which writes one or more *complete* newline-terminated lines to
/// stderr) so its output lands cleanly above the progress bar rather than
/// interleaving with it. When a live bar is active, the bar line is erased and
/// `print` runs under the bar's `draw_lock`, so neither the heartbeat thread nor
/// a parallel worker can repaint the bar between the erase and the write; the
/// bar redraws itself on the next `increment`/heartbeat tick, beneath the lines
/// just printed. With no active bar (server/JSON modes, or after the scan ends)
/// the closure simply runs. Callers must not already hold `draw_lock`.
pub(crate) fn print_above_bar(print: impl FnOnce()) {
    // Upgrade to an owned Arc and drop the registry guard before touching
    // draw_lock, so the two locks are never held at once (no ordering cycle).
    let bar = ACTIVE_BAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .and_then(Weak::upgrade);
    match bar {
        Some(inner) => {
            let _guard = inner
                .draw_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Erase the bar only if it's still live; once stopped, `finish`
            // already cleared the line and the cursor is at a fresh column 0.
            if !inner.stopped.load(Ordering::Relaxed) {
                eprint!("\r\x1b[2K");
                let _ = std::io::stderr().flush();
            }
            print();
        }
        None => print(),
    }
}

impl Inner {
    fn draw(&self) {
        let _guard = self
            .draw_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.stopped.load(Ordering::Relaxed) {
            return;
        }
        self.render();
    }

    fn render(&self) {
        let done = self.analyzed.load(Ordering::Relaxed);
        let elapsed = self.start.elapsed();
        let rate = f64::from(done) / elapsed.as_secs_f64().max(0.001);
        let eta = f64::from(self.total - done) / rate.max(0.001);

        let frame = SPINNER[self.tick.fetch_add(1, Ordering::Relaxed) as usize % SPINNER.len()];
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

        let stats = format!("{done}/{}  {:.0}/s  {}", self.total, rate, format_eta(eta));

        // Long-tail reassurance: when only a few files remain and the count has
        // not moved for a while, the scan is almost certainly deep in a slow
        // reverse-engineering pass, not hung. Appended to the bar line (not a
        // separate line) so the bar's own clear erases it — the note shows while
        // the tail is stalled and vanishes the instant a file completes or the
        // scan ends. Capped to the terminal width so it can never wrap.
        let left = self.total - done;
        let stalled_ms = elapsed
            .as_millis()
            .saturating_sub(u128::from(self.last_advance_ms.load(Ordering::Relaxed)));
        let note = if (1..=TAIL_FILES).contains(&left) && stalled_ms >= TAIL_STALL_MS {
            // Fixed visible prefix: 1 indent + spinner + space + bar + 2 spaces.
            let used = 25 + stats.chars().count();
            fit_notice(TAIL_MESSAGE, self.term_cols.saturating_sub(used + 1))
        } else {
            String::new()
        };

        // `\x1b[K` erases to end of line — robustly clears a wider previous
        // frame (a longer ETA, or the notice once it's gone) without padding.
        eprint!(
            "\r \x1b[38;2;100;180;255m{frame}\x1b[0m \x1b[38;2;80;160;220m{filled_str}\x1b[38;2;50;50;50m{dim_str}\x1b[0m  \x1b[38;2;160;160;160m{stats}\x1b[0m{note}\x1b[K",
        );
        let _ = std::io::stderr().flush();
    }

    /// Halt the heartbeat and wait out any in-flight render, so the caller can
    /// write a final line the ticker can no longer overwrite.
    fn quiesce(&self) {
        self.stopped.store(true, Ordering::Relaxed);
        let _guard = self
            .draw_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

/// Progress bar handle held by the scan loop. Dropping it (or calling `finish`
/// / `quiesce`) stops the background heartbeat thread.
pub(crate) struct Progress {
    inner: Arc<Inner>,
}

impl Progress {
    pub(crate) fn new(total: u32) -> Self {
        let inner = Arc::new(Inner {
            analyzed: AtomicU32::new(0),
            total,
            start: Instant::now(),
            last_advance_ms: AtomicU64::new(0),
            tick: AtomicU32::new(0),
            term_cols: term_cols(),
            stopped: AtomicBool::new(false),
            draw_lock: Mutex::new(()),
        });
        // Publish this bar so the fetch progress block can print above it (see
        // `print_above_bar`). Overwrites any prior registration — the terminal
        // CLI runs one bar at a time — and is cleared on drop.
        *ACTIVE_BAR
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::downgrade(&inner));
        // Detached heartbeat: it observes `stopped` and exits within one tick of
        // `quiesce`/drop. A spawn failure just means no heartbeat — the per-file
        // redraws still drive the bar.
        let beat = Arc::clone(&inner);
        let _ = std::thread::Builder::new()
            .name("scan-progress".into())
            .spawn(move || {
                while !beat.stopped.load(Ordering::Relaxed) {
                    std::thread::sleep(PROGRESS_TICK);
                    beat.draw();
                }
            });
        Self { inner }
    }

    pub(crate) fn increment(&self) {
        self.inner.analyzed.fetch_add(1, Ordering::Relaxed);
        let ms = u64::try_from(self.inner.start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.inner.last_advance_ms.store(ms, Ordering::Relaxed);
        self.inner.draw();
    }

    /// Erase the bar, run `print` (which writes a file's result to stdout), then
    /// redraw the bar beneath it — all while holding the draw lock so the
    /// heartbeat thread cannot repaint the bar between the erase and the write.
    /// Without the leading erase the result is grafted onto the bar line the
    /// last render left on screen (cursor parked at its end, no newline).
    pub(crate) fn around_result(&self, print: impl FnOnce()) {
        let inner = &self.inner;
        let _guard = inner
            .draw_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
        print();
        if !inner.stopped.load(Ordering::Relaxed) {
            inner.render();
        }
    }

    /// Stop the heartbeat without printing anything, for callers that render
    /// their own final line in the bar's place (e.g. `ps`).
    pub(crate) fn quiesce(&self) {
        self.inner.quiesce();
    }

    /// Stop the heartbeat and erase the progress bar, leaving the cursor at the
    /// start of its now-blank line so the caller can write the closing summary in
    /// its place. The bar deliberately prints no completion line of its own — the
    /// summary is the single end statement (one verdict, one duration), so a
    /// separate "N files in Xs" line here would only repeat it.
    pub(crate) fn finish(&self) {
        self.inner.quiesce();
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        // Safety net: stop the heartbeat even on an early return or error that
        // skipped `finish`/`quiesce`.
        self.inner.stopped.store(true, Ordering::Relaxed);
        // Unpublish so a later `print_above_bar` doesn't try to erase a bar that
        // no longer owns the terminal. A stale `Weak` would upgrade to `None`
        // once `Inner` frees, but the heartbeat thread holds a clone until it
        // observes `stopped`, so clear eagerly rather than rely on that timing.
        *ACTIVE_BAR
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
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

/// Apply a bloom-filter decision before any expensive analysis. Emits the
/// verdict and returns `Some(summary)` when the artifact is short-circuited (a
/// known-good skip, or — in fast mode — a bloom-only verdict); `None` to proceed.
pub(crate) fn bloom_gate(
    config: &ScanConfig,
    label: &str,
    decision: BloomDecision,
) -> Option<ScanSummary> {
    use crate::output::BloomVerdict;
    let fast = config.mode() == Mode::Fast;
    let format = config.format();
    // Known-good/unscanned are benign-tier: print the per-file line only when
    // benign output is requested, so a large directory scan isn't spammed. The
    // aggregate tally always reports them in the summary regardless.
    let show_quiet = format == OutputFormat::Json || config.filter().shows(&Classification::Benign);
    // Tally for end-of-scan observability. `unscanned` only matters for Unknown.
    crate::bloom_repo::record(decision, fast && decision == BloomDecision::Unknown);
    match decision {
        BloomDecision::Skip => {
            if show_quiet {
                crate::output::print_bloom_verdict(label, BloomVerdict::KnownGood, format);
            }
            Some(one_file_summary(0, 0, 1))
        }
        // Known-bad and conflicted are flags, not short-circuits: they run full
        // analysis in every mode — the scan is the verdict — and the flag rides
        // the normal result inline (see `BloomMark`), so no separate banner is
        // printed here. The caller derives the mark from this same decision.
        BloomDecision::KnownBad => None,
        BloomDecision::Conflicted => {
            // Build-time subtraction removes bad keys from good, so a conflict can
            // only mean filter version skew or a producer bug — it should never
            // happen. Always scan, but make the noise loud (in the logs).
            tracing::warn!(
                label,
                "bloom conflict: key is in BOTH the good and bad filters (should never happen) — scanning"
            );
            None
        }
        // Unknown is left unscanned only in bloom-only fast mode; otherwise scanned.
        BloomDecision::Unknown => fast.then(|| {
            if show_quiet {
                crate::output::print_bloom_verdict(label, BloomVerdict::Unscanned, format);
            }
            one_file_summary(0, 0, 0)
        }),
    }
}

/// The bloom short-circuit for a scan target, honoring the local known-good
/// freshness override: a known-good file created, status-changed, or modified
/// within the last 48h is scanned on its own merits rather than skipped on its
/// bloom vouch. Returns `Some(summary)` when the target is short-circuited
/// (counted, not scanned) and `None` when it must be scanned. Used by process
/// scans, which decide before analysis; path scans inline the same override in
/// [`record_file_result`], where cleave has already produced a minimal report by
/// the time the decision is known and a fresh known-good is re-analyzed in place.
pub(crate) fn bloom_gate_fresh(
    config: &ScanConfig,
    path: &Path,
    decision: BloomDecision,
) -> Option<ScanSummary> {
    if decision == BloomDecision::Skip
        && file_touched_within(path, KNOWN_GOOD_RESCAN_SECS, SystemTime::now())
    {
        return None;
    }
    bloom_gate(config, &path.display().to_string(), decision)
}

/// Build the cleave skip predicate from the active bloom filters: a file whose
/// sha256 is known-good (or, in fast mode, simply unknown) is skipped before
/// analysis. Known-bad and conflicted return `false` so they are still analyzed;
/// every file's verdict line is emitted later in [`record_file_result`], which
/// re-derives the decision from the report's sha. `None` when bloom is disabled.
fn bloom_skip_predicate(config: &ScanConfig) -> Option<cleave::SkipPredicate> {
    let lookup = config.bloom_arc()?;
    let fast = config.mode() == Mode::Fast;
    Some(cleave::SkipPredicate(Arc::new(
        move |sha_hex: &str, path: &Path| {
            let Some(digest) = crate::bloom::parse_sha256_hex(sha_hex) else {
                return false;
            };
            match lookup.decide_sha256(&digest) {
                // Known-good is trusted and skipped, unless the file was created,
                // status-changed, or modified within the last 48h — a fresh
                // known-good is analyzed on its own merits (recent activity, and a
                // guard against a bloom false-positive on a freshly planted file),
                // so cleave analyzes it once here rather than skipping then
                // re-analyzing in `record_file_result`.
                BloomDecision::Skip => {
                    !file_touched_within(path, KNOWN_GOOD_RESCAN_SECS, SystemTime::now())
                }
                BloomDecision::Unknown => fast,
                BloomDecision::KnownBad | BloomDecision::Conflicted => false,
            }
        },
    )))
}

/// A one-file [`ScanSummary`] for a bloom-decided artifact that was not analyzed.
const fn one_file_summary(hostile: u32, suspicious: u32, benign: u32) -> ScanSummary {
    ScanSummary {
        total_files: 1,
        hostile,
        suspicious,
        benign,
        errors: 0,
        duration_ms: 0,
    }
}

/// Enumerate the regular files under `dir` for a directory scan, mirroring the
/// structural filters cleave applies during its own walk: skip `.git*` trees,
/// keep only regular files, never follow symlinks. This reads no file contents —
/// it is a cheap `readdir` pass whose purpose is to learn the file count (and
/// the list) upfront, so the progress bar has a denominator before analysis
/// begins. The list is handed to [`cleave::scan_paths`], which still applies its
/// program-type and size filters per file, so a few of these may be analyzed
/// away without a verdict (the bar tops out just under 100%, then the summary
/// reports the true total).
fn discover_files(dir: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !e.file_name().to_string_lossy().starts_with(".git"))
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.path().to_path_buf())
        .collect()
}

/// Total detection rules resident in memory, folding in `bloom_rules` bloom
/// signatures: YAML traits + composite rules + YARA rules + `bloom_rules`.
/// `cleave::version_info()` loads (or reuses) the shared resources. Split out so
/// the auto-update checkmark — which knows its bloom count without a
/// [`ScanConfig`] — reports the same figure as the banner.
pub(crate) fn detection_rule_count_from(bloom_rules: u64) -> u64 {
    let info = cleave::version_info();
    info.trait_count as u64 + info.composite_count as u64 + info.yara_rules as u64 + bloom_rules
}

/// Total detection rules resident in memory after resource load — YAML traits +
/// composite rules + YARA rules + bloom signatures — for the scan banner. Reads
/// already-loaded counts, so it adds no work. Shared by [`run`], [`run_paths`],
/// and the `ps` subcommand so every banner reports the same figure.
pub(crate) fn detection_rule_count(config: &ScanConfig) -> u64 {
    let bloom_rules = config
        .bloom()
        .map_or(0, crate::bloom_repo::Lookup::rule_count);
    detection_rule_count_from(bloom_rules)
}

/// Run a scan against a file or directory tree.
///
/// A file path is analyzed directly. A directory path is walked once by
/// [`discover_files`] to learn the file count upfront (for the progress bar and
/// ETA), then the discovered list is analyzed in parallel via
/// [`cleave::scan_paths`], with results streamed as they complete.
///
/// # Errors
/// Returns an error if the target path does not exist, model artifacts cannot
/// be loaded, or `cleave` analysis fails for the overall scan operation.
pub fn run(path: &Path, config: &ScanConfig) -> Result<ScanSummary> {
    prefetch_cleave_resources();

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
        skip_predicate: bloom_skip_predicate(config),
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
            None,
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

    // Directory scan: walk the tree ourselves first so the file count is known
    // before any analysis starts — cleave's own walk streams and reports the
    // total only once it finishes, leaving a progress bar with no denominator.
    // The discovered list is handed to cleave::scan_paths, which applies the
    // same filtering as scan_directory, so the tree is walked exactly once.
    let files = discover_files(path);
    let total = u32::try_from(files.len()).unwrap_or(u32::MAX);
    let tally = Tally::default();
    let stdout = Mutex::new(std::io::stdout());
    let progress = (is_terminal && files.len() > 1).then(|| {
        crate::output::print_banner(detection_rule_count(config));
        Progress::new(total)
    });

    cleave::scan_paths(files, &cleave_opts, |event| {
        if let cleave::ScanEvent::File {
            path: ref file_path,
            result,
        } = event
        {
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
                progress.as_ref(),
                None,
                None,
            );
        }
    })?;

    if let Some(p) = &progress {
        p.finish();
    }

    let summary = tally.summary(scan_start);
    if is_terminal {
        crate::output::print_summary(&summary);
    }

    Ok(summary)
}

/// Analyze in-memory bytes — a fetched artifact — exactly as a single local
/// file: classify, render, and tally. `label` is the display path (the URL or
/// PURL); `name` drives cleave's extension-based type detection. Powers the
/// `pkg`/`url` subcommands. Honors `config`'s fetch policy, so the fetched
/// package's own declared references are followed when `--fetch` is set.
pub fn run_bytes(
    label: &str,
    name: &str,
    bytes: Vec<u8>,
    config: &ScanConfig,
    root_registry: Option<&fletch::Registry>,
) -> Result<ScanSummary> {
    prefetch_cleave_resources();

    let model = Model::load(config.model_dir(), config.thresholds(), config.level())?;
    let shap = ShapImportance::load(config.model_dir()).ok();
    let ctx = ExtractContext::new(model.spec());
    // Content-digest short-circuit happens inside cleave via the skip predicate,
    // complementing the PURL gate in `pkg.rs`: it catches known *content* even
    // when the package locator was not in the filter. `record_file_result` emits
    // the verdict from the minimal report cleave returns on a skip.
    let cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        skip_predicate: bloom_skip_predicate(config),
        ..Default::default()
    };
    let is_terminal = matches!(config.format(), OutputFormat::Terminal);
    let scan_start = Instant::now();
    let tally = Tally::default();
    let stdout = Mutex::new(std::io::stdout());

    let report = cleave::analyze_bytes_owned(bytes, name, &cleave_opts)
        .with_context(|| format!("cleave analysis of {label}"));
    record_file_result(
        Path::new(label),
        report,
        &ctx,
        &model,
        shap.as_ref(),
        config,
        None,
        &tally,
        &stdout,
        None,
        None,
        root_registry,
    );

    let summary = tally.summary(scan_start);
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
            // Only files that were actually analyzed count as scanned. Errors
            // (missing paths, read failures) are reported separately so a path
            // that was never opened is never tallied as a clean scan.
            total_files: hostile + suspicious + benign,
            hostile,
            suspicious,
            benign,
            errors,
            duration_ms: crate::duration_ms(scan_start.elapsed()),
        }
    }
}

/// Freshness window for the local known-good re-scan: a known-good file created,
/// changed, or modified this recently is scanned on its own merits rather than
/// skipped. Mirrors the dependency freshness window in [`crate::fetch`].
const KNOWN_GOOD_RESCAN_SECS: u64 = crate::fetch::FRESH_WINDOW_SECS;

/// Whether the file at `path` was created (btime), status-changed (ctime), or
/// modified (mtime) within the last `window` seconds. A known-good file this
/// fresh is re-scanned rather than trusted on its bloom vouch — recent activity,
/// and a guard against a bloom false-positive on a freshly planted file. A
/// timestamp in the future (clock skew) counts as recent; unreadable metadata
/// (e.g. a fetched artifact's synthetic path) counts as not-recent.
fn file_touched_within(path: &Path, window: u64, now: SystemTime) -> bool {
    let Ok(md) = std::fs::metadata(path) else {
        return false;
    };
    let recent = |t: SystemTime| now.duration_since(t).map_or(true, |d| d.as_secs() <= window);
    // created()/modified() are unsupported on some platforms/filesystems.
    if md.created().is_ok_and(recent) || md.modified().is_ok_and(recent) {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let ctime = md.ctime();
        if ctime >= 0 && recent(UNIX_EPOCH + Duration::from_secs(ctime.unsigned_abs())) {
            return true;
        }
    }
    false
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
    uploader: Option<&crate::upload::Uploader>,
    root_registry: Option<&fletch::Registry>,
) {
    // Bloom verdict, re-derived from the root sha cleave computed. A file cleave
    // skipped at our request (known-good, or fast-mode unknown) arrives as a
    // minimal report and is short-circuited here; a known-bad/conflicted file was
    // analyzed and is flagged here before its full verdict is rendered below.
    // The bloom flag for the (still-scanned) known-bad/conflicted result, derived
    // from the same decision `bloom_gate` acts on. Known-good/unknown short-circuit
    // above and never reach the scanned render, so they leave no mark.
    let mut bloom_mark = None;
    // Resolve the bloom decision from the sha cleave computed.
    let decision = if let Some(lookup) = config.bloom()
        && let Ok(report) = &cleave_result
        && let Some(digest) = crate::bloom::parse_sha256_hex(&report.target.sha256)
    {
        Some(lookup.decide_sha256(&digest))
    } else {
        None
    };
    if let Some(decision) = decision {
        // A known-good file created/changed/modified within the last 48h is scanned
        // on its own merits: the skip predicate declined to skip it, so cleave has
        // already produced a full report — fall through to the normal scan (no bloom
        // mark, since `from_decision(Skip)` is `None`). A stale known-good (and, in
        // fast mode, an unknown) arrived as a minimal report and is counted here
        // without an ML pass.
        let fresh_known_good = decision == BloomDecision::Skip
            && file_touched_within(file_path, KNOWN_GOOD_RESCAN_SECS, SystemTime::now());
        if !fresh_known_good {
            if let Some(summary) = bloom_gate(config, &file_path.display().to_string(), decision) {
                if summary.benign > 0 {
                    tally.count(Classification::Benign);
                }
                if let Some(p) = progress {
                    p.increment();
                }
                return;
            }
            bloom_mark = crate::output::BloomMark::from_decision(decision);
        }
    }

    let scan_result = cleave_result.and_then(|report| {
        process_report(
            file_path,
            report,
            ctx,
            model,
            shap,
            config,
            cancellation,
            root_registry,
            bloom_mark,
        )
    });
    if let Some(p) = progress {
        p.increment();
    }
    match scan_result {
        Ok(mut r) => {
            tally.count(r.classification);
            // A bloom-flagged (known-bad/conflicted) file is always surfaced — its
            // flag replaces the old unconditional banner, so the benign filter must
            // not swallow a known-bad file the model happens to rate benign.
            if config.format() == OutputFormat::Json
                || r.bloom_mark.is_some()
                || config.filter().shows(&r.classification)
            {
                if let Some(p) = progress {
                    p.around_result(|| emit_result(&r, config, true, stdout));
                } else {
                    emit_result(&r, config, false, stdout);
                }
            }
            // Renew this result on hopper if `--hopper` was set. Done after
            // emit so local output is never delayed by the handoff; the upload
            // itself runs on the uploader's own thread. Only successful results
            // are sent — an error envelope has an empty file type, which hopper
            // treats as a delete. `into_envelope` consumes `r`, so it goes last.
            if let Some(uploader) = uploader {
                let sha256 = r.sha256.clone();
                let size = r.size_bytes;
                let deps = std::mem::take(&mut r.dependency_results);
                let envelope = r.into_envelope();
                upload_scan_result(uploader, file_path, sha256, size, deps, envelope);
            }
        }
        Err(e) => {
            let msg = crate::tools::enrich_error(&e).unwrap_or_else(|| format!("{e:#}"));
            tracing::error!("error analyzing {}: {}", file_path.display(), msg);
            tally.errors.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Renew one scan result on hopper: ensure hopper has the scanned file and any
/// fetched dependency archives (with provenance), then renew the verdict. Used
/// by both the CLI `--hopper` path and the serve-mode `--hopper` upload. The
/// artifacts are queued before the result so a never-seen top-level file's row
/// exists before its verdict lands. Blocking (reads sidecars from disk); callers
/// on an async runtime must run it off the executor.
pub(crate) fn upload_scan_result(
    uploader: &crate::upload::Uploader,
    file_path: &Path,
    sha256: String,
    size_bytes: u64,
    dependency_results: Vec<DepResult>,
    envelope: ScanResultEnvelope,
) {
    let artifacts = collect_upload_artifacts(file_path, &sha256, size_bytes, upload_collector());
    uploader.submit_artifacts(artifacts);
    // Each fetched dependency, mirrored into hopper as its own sample: bytes (only
    // if missing) + provenance + the verdict scan computed for it. Queued before
    // the root verdict so a dependency's row exists before its own verdict lands.
    if !dependency_results.is_empty() {
        uploader.submit_dependencies(
            dependency_results,
            envelope.ml.version.clone(),
            envelope.ml.analyzed_at.clone(),
        );
    }
    uploader.submit(sha256, envelope);
}

/// Build the `/api/result` envelope for a fetched dependency: the standalone
/// cleave report scan captured for it as `raw`, and the aggregate verdict it
/// computed as the `ml` section — the same shape a first-hand scan of those bytes
/// would post, so hopper records the dependency exactly as if it had been scanned
/// directly. `version`/`analyzed_at` are the parent run's, identifying the build.
#[must_use]
pub(crate) fn dep_envelope(
    dep: DepResult,
    version: &str,
    analyzed_at: &str,
) -> ScanResultEnvelope {
    let level = dep.level;
    let ml_files = build_ml_files(&dep.raw, dep.probability, level, &[]);
    ScanResultEnvelope {
        ml: MlSection {
            v: "7",
            probability: dep.probability,
            level,
            conf: level_confidence(level),
            model_scores: Vec::new(),
            skipped_models: Vec::new(),
            version: version.to_string(),
            eng: ENGINE_VERSION,
            analyzed_at: analyzed_at.to_string(),
            files: ml_files,
            pids: None,
            deleted: None,
        },
        llm: None,
        raw: dep.raw,
    }
}

/// The `fetch.collector` identity stamped on every sidecar this process uploads,
/// computed once. Mirrors hopper's collector convention (`forager+<id>`,
/// `prism`) so a sample's origin is legible.
fn upload_collector() -> &'static str {
    static COLLECTOR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    COLLECTOR.get_or_init(|| format!("scan+{}", crate::upload::default_worker_name()))
}

/// The scanned file itself, offered to hopper so a `--upload` run can store a
/// locally-analyzed file hopper has never seen. Just the one artifact — fetched
/// dependencies are mirrored separately (bytes, provenance, *and* verdict) by
/// [`crate::upload::Uploader::submit_dependencies`], so they never ride here.
fn collect_upload_artifacts(
    file_path: &Path,
    sha256: &str,
    size_bytes: u64,
    collector: &str,
) -> Vec<crate::upload::UploadArtifact> {
    use crate::upload::{ArtifactBytes, UploadArtifact};
    let now = now_rfc3339();

    // The scanned file: bytes from disk, no registry (a local, un-fetched artifact).
    let root_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file")
        .to_string();
    vec![UploadArtifact {
        sha256: sha256.to_string(),
        size: size_bytes,
        filename: root_name.clone(),
        bytes: ArtifactBytes::File(file_path.to_path_buf()),
        sidecar: crate::provenance::build_sidecar(
            &root_name,
            sha256,
            size_bytes,
            collector,
            &now,
            "",
            "",
            None,
            &[],
        ),
        // The root file's sidecar carries no registry/PURL, so there's nothing
        // worth backfilling onto a copy hopper already has.
        backfill: false,
    }]
}

/// A filename for an uploaded artifact: the last path segment of the fetch URL
/// (query/fragment stripped), falling back to the locator's tail with PURL
/// punctuation flattened. hopper uses it for the stored filename and type sniff.
pub(crate) fn artifact_filename(url: &str, locator: &str) -> String {
    let from_url = url
        .rsplit('/')
        .next()
        .map(|seg| seg.split(['?', '#']).next().unwrap_or(seg))
        .filter(|seg| !seg.is_empty());
    if let Some(name) = from_url {
        return name.to_string();
    }
    locator
        .rsplit('/')
        .next()
        .unwrap_or(locator)
        .replace(['@', ':'], "-")
}

/// SHA-256 (hex) of a file's contents, used to match a scanned file to its entry
/// in a `--registry-map`. `None` on any I/O error — the file just scans without
/// registry provenance.
fn sha256_file_hex(path: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).ok()?;
    Some(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
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
pub fn run_paths(
    paths: &[PathBuf],
    config: &ScanConfig,
    registry_map: Option<&std::collections::HashMap<String, fletch::Registry>>,
) -> Result<ScanSummary> {
    prefetch_cleave_resources();

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
        skip_predicate: bloom_skip_predicate(config),
        ..Default::default()
    };
    let is_terminal = matches!(config.format(), OutputFormat::Terminal);
    let scan_start = Instant::now();

    let tally = Tally::default();
    let stdout = Mutex::new(std::io::stdout());

    // Partition the requested paths: explicit files become one unfiltered batch
    // (a named file is always analyzed), each directory is walked upfront into
    // its file list, and anything else is an error. Walking the directories here
    // — rather than letting cleave stream them — lets us learn the total file
    // count before analysis starts, so the progress bar has a denominator.
    let mut files = Vec::new();
    let mut dir_files = Vec::new();
    for path in paths {
        if path.is_file() {
            files.push(path.clone());
        } else if path.is_dir() {
            dir_files.extend(discover_files(path));
        } else {
            tracing::error!("path does not exist: {}", path.display());
            tally.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    let total = u32::try_from(files.len() + dir_files.len()).unwrap_or(u32::MAX);
    let progress = (is_terminal && total > 1).then(|| {
        crate::output::print_banner(detection_rule_count(config));
        Progress::new(total)
    });

    // `--hopper`: renew every result on hopper as it completes. The uploader
    // owns a background thread; dropping it (below, after the scan closures
    // release their borrow) flushes and joins in-flight uploads.
    let uploader = config
        .hopper()
        .map(|url| crate::upload::Uploader::new(url, crate::upload::default_worker_name()));

    {
        let record = |file_path: &Path, result: Result<cleave::AnalysisReport>| {
            // Per-file provenance: match this artifact's content sha to its registry
            // record in the map. A file with no entry simply scans without it.
            let root_registry =
                registry_map.and_then(|m| sha256_file_hex(file_path).and_then(|sha| m.get(&sha)));
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
                progress.as_ref(),
                uploader.as_ref(),
                root_registry,
            );
        };

        if !files.is_empty() {
            cleave::scan_files(&files, &cleave_opts, |event| {
                if let cleave::ScanEvent::File { path, result } = event {
                    record(&path, *result);
                }
            })?;
        }

        if !dir_files.is_empty() {
            cleave::scan_paths(dir_files, &cleave_opts, |event| {
                if let cleave::ScanEvent::File { path, result } = event {
                    record(&path, *result);
                }
            })?;
        }
    }

    if let Some(p) = &progress {
        p.finish();
    }
    // `record` (and its borrow of `uploader`) is now out of scope; flush and
    // join the uploader before reporting the summary so every result is renewed.
    drop(uploader);

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
    // The bloom flag rides the machine line as `bloom=known-bad|conflicted` — the
    // machine-readable analog of the terminal 🚩/🏴 (the JSON envelope is untouched).
    let bloom = r
        .bloom_mark
        .map_or_else(String::new, |m| format!(" bloom={}", m.tiny_str()));
    let verdict = if matches!(r.classification, Classification::Benign) {
        format!("scan {class} confidence={:.3}{bloom}\n", r.probability)
    } else {
        let fp_level = r.level.map_or_else(|| "-".to_string(), |n| format!("L{n}"));
        format!(
            "scan {class} confidence={:.3} fp-level={fp_level}{bloom}\n",
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

#[allow(clippy::too_many_arguments)]
fn render_terminal_context(
    report: &cleave::AnalysisReport,
    tiny_opts: &cleave::output::TinyOpts,
    decision: &Decision,
    reasons: &[Reason],
    interpretation: Option<&crate::interpret::Interpretation>,
    sha256: &str,
    label: &str,
    bloom_mark: Option<crate::output::BloomMark>,
) -> String {
    let (badge, badge_w) = crate::output::terminal_badge(
        &decision.class,
        decision.probability,
        decision.threshold,
        decision.level,
        bloom_mark,
    );
    // The filename starts after the stamp and one separator space.
    let indent = badge_w + 1;
    let trailer = crate::output::terminal_trailer(reasons, interpretation);
    let subtitle = crate::output::terminal_subtitle(sha256, indent);
    let adorn = cleave::output::HeaderBadge {
        badge: Some(&badge),
        trailer: trailer.as_deref(),
        subtitle: subtitle.as_deref(),
    };
    let body = cleave::output::format_context_badged(report, tiny_opts, adorn);
    if body.is_empty() {
        // No notable+ traits to anchor a header on: still surface the verdict
        // headline so a flagged file is never silent.
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
    /// Fetched dependencies to mirror into hopper as their own samples.
    pub(crate) dependency_results: Vec<DepResult>,
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
    fetch_progress: bool,
    render_context: bool,
    list_all_members: bool,
    // The root artifact's own registry metadata, for the one-shot `pkg:`/`url`
    // path. `None` for ordinary scans. Grafted as a child `registry` node (so it
    // trains like any other file) and correlated with the artifact via a
    // `scope: package` composite, mirroring what `fetch::orchestrate` does per
    // fetched dependency.
    root_registry: Option<&fletch::Registry>,
    // The bloom status flag (🚩 known-bad / 🏴 conflicted), rendered inline in the
    // terminal header. `None` for unflagged files.
    bloom_mark: Option<crate::output::BloomMark>,
) -> Result<ClassifiedReport> {
    // Capture every archive member — including the ones cleave catalogues but
    // never analyzes (docs, data files, images: non-program members it skips by
    // default) — before `finalize()` clears `archive_contents`. With `--show=all`
    // JSON output these are surfaced as listing-only entries below so the manifest
    // is complete; otherwise the snapshot stays empty and nothing changes.
    let listed_members: Vec<ArchiveMemberStub> = if list_all_members {
        report
            .archive_contents
            .iter()
            .map(|e| ArchiveMemberStub {
                path: e.path.clone(),
                file_type: e.file_type.clone(),
                sha256: e.sha256.clone(),
                size_bytes: e.size_bytes,
            })
            .collect()
    } else {
        Vec::new()
    };
    report.finalize();
    // The sha256s of the sample's own files, captured before fetching grafts any
    // external payload onto report.files. The sample's own verdict is featurized
    // from these alone (see `root_json` below): fetched content is external —
    // reached via a reference — and may *escalate* the verdict through the
    // per-file embedded path, but must never dilute the sample's own aggregate
    // (a benign fetched dependency would otherwise mask a hostile manifest).
    let own_shas: std::collections::HashSet<String> =
        report.files.iter().map(|f| f.sha256.clone()).collect();
    // Fetch the external references the analysis surfaced and graft each payload
    // into report.files as a uniform node — after finalize() (which populates
    // files[] and the per-file declared references) and before featurization, so
    // fetched content feeds the verdict like any other file. Off unless the
    // policy selects a kind. The returned edges are attached to report_json below.
    let (fetch_edges, fetched_deps) =
        crate::fetch::orchestrate(&mut report, root_path, fetch, fetch_progress);
    // One-shot `pkg:`/`url`: graft the root artifact's own registry metadata as a
    // child `registry` node and correlate the two with a `scope: package`
    // composite. The `--fetch` path does the equivalent per fetched dependency
    // inside `orchestrate`; here the registry record is the root's own. Runs
    // after `orchestrate` (so the registry node sits outside the sample's own
    // `own_shas` aggregate, like other grafted content) and before `strip` (so a
    // package composite's `trait_refs` keep their building-block traits).
    if let Some(reg) = root_registry {
        crate::fetch::graft_root_registry(&mut report, reg);
    }
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
    // Parse the report once with every optional raw subtree any specialist may
    // read, then share it across the root and embedded-file scoring passes.
    // Specialist ONNX graphs still load on demand; this only keeps archive
    // members with different filetypes from missing route-specific raw fields.
    let needs = ctx.raw_needs().union(crate::features::RawNeeds::all());
    // The sample's own decision featurizes its own files only. With nothing
    // fetched this is the whole report; otherwise drop the grafted payloads so
    // they can't dilute the aggregate (they still classify individually via the
    // embedded path on the full `report_json`, where a hostile one elevates).
    let root_json = if fetch_edges.is_empty() {
        std::borrow::Cow::Borrowed(&report_json)
    } else {
        let mut rj = report_json.clone();
        if let Some(files) = rj
            .get_mut("files")
            .and_then(serde_json::Value::as_array_mut)
        {
            files.retain(|f| {
                f.get("sha")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|s| own_shas.contains(s))
            });
        }
        std::borrow::Cow::Owned(rj)
    };
    let parsed = crate::features::ParsedReport::from_report(&root_json, needs);
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
        .filter(|f| {
            json_alias(f, &["depth", "dp"])
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                > 0
        });
    let embedded_entries: Vec<&serde_json::Value> = match embedded_file_limit {
        Some(limit) => embedded_iter.take(limit).collect(),
        None => embedded_iter.collect(),
    };

    let mut embedded_files: Vec<EmbeddedFile> = Vec::with_capacity(embedded_entries.len());
    let mut max_decision: Decision = decision;

    // Fetched payloads keyed by the sha of the content retrieved, with the
    // declaring file + the byte the reference sits at. When the embedded pass
    // classifies one of these hostile/suspicious, that verdict is pinned back
    // onto the declaring manifest at the reference byte (below the loop).
    let fetched_by_content: std::collections::HashMap<&str, (&str, Option<u64>, &str)> =
        fetch_edges
            .iter()
            .filter_map(|r| {
                r.content_sha256.as_deref().map(|c| {
                    (
                        c,
                        (
                            r.source_sha256.as_str(),
                            r.source_offset,
                            r.locator.as_str(),
                        ),
                    )
                })
            })
            .collect();
    // (declaring sha, reference offset, dependency locator, confirmed class).
    let mut dep_backrefs: Vec<(String, Option<u64>, String, Classification)> = Vec::new();

    // Attribute each fetched dependency's files to it so the embedded pass's
    // per-node decisions roll up into the dependency's own aggregate verdict —
    // its container elevated by its worst member, exactly as a first-hand scan of
    // those bytes resolves. No extra model work: the decisions are the ones the
    // loop already computes for the merged report.
    let sha_to_dep: std::collections::HashMap<&str, usize> = fetched_deps
        .iter()
        .enumerate()
        .flat_map(|(i, d)| d.member_shas.iter().map(move |s| (s.as_str(), i)))
        .collect();
    let mut dep_decisions: Vec<Option<Decision>> = vec![None; fetched_deps.len()];

    // Feature-extract and model-score members in parallel: reports with
    // thousands of embedded files (nested npm tarballs, fetched dependency
    // trees) previously ran this loop serially on one rayon worker — on
    // member-heavy archives that single-threaded model pass, not cleave's
    // analysis, was the scan's wall-clock tail. Per-member work is pure
    // (&Model is already shared across rayon workers by the outer scan);
    // the fold below stays single-threaded and in entry order, so tie-break
    // semantics (`decision_outranks` keeps the earliest on ties) and output
    // ordering are byte-identical to the serial loop.
    struct EmbeddedScored<'j> {
        ef: &'j serde_json::Value,
        decision: Decision,
        model_scores: Vec<RouteScore>,
        skipped_models: Vec<SkippedRoute>,
    }
    let scored: Vec<EmbeddedScored<'_>> = {
        use rayon::prelude::*;
        embedded_entries
            .par_iter()
            .map(|&ef| {
                // Cancellation: skip the expensive work; the post-pass bail
                // below surfaces the cancellation before results are used.
                if let Some(c) = cancellation
                    && c.load(Ordering::Relaxed)
                {
                    return EmbeddedScored {
                        ef,
                        decision: Decision {
                            class: Classification::Benign,
                            probability: 0.0,
                            threshold: model.thresholds().suspicious,
                            level: None,
                        },
                        model_scores: Vec::new(),
                        skipped_models: Vec::new(),
                    };
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
                // Trait floor on the member's own findings — a sparse, severe
                // dropper (the npm install-hook beacon lives in the embedded
                // package.json) scores above the crit-4 fraction gate even when
                // the container's findings dilute it; the floored member then
                // elevates the container below.
                apply_trait_floor(
                    &mut ef_decision,
                    ef,
                    model.grid_max(),
                    ef["path"].as_str().unwrap_or(label),
                );
                EmbeddedScored {
                    ef,
                    decision: ef_decision,
                    model_scores: ef_model_scores,
                    skipped_models: ef_skipped_models,
                }
            })
            .collect()
    };
    if let Some(c) = cancellation
        && c.load(Ordering::Relaxed)
    {
        anyhow::bail!("analysis cancelled during embedded file processing");
    }

    for item in scored {
        let EmbeddedScored {
            ef,
            decision: ef_decision,
            model_scores: ef_model_scores,
            skipped_models: ef_skipped_models,
        } = item;

        // Roll this node's decision into the aggregate verdict of the fetched
        // dependency it belongs to (the container or one of its members).
        if let Some(sha) = ef["sha"].as_str()
            && let Some(&di) = sha_to_dep.get(sha)
            && dep_decisions[di]
                .as_ref()
                .is_none_or(|cur| decision_outranks(&ef_decision, cur))
        {
            dep_decisions[di] = Some(ef_decision);
        }

        // A fetched dependency that classifies hostile/suspicious is pinned back
        // onto the file that declared it (request: the manifest names the bad
        // dependency at its byte). Keyed by content sha so it matches the
        // retrieved payload node.
        if matches!(
            ef_decision.class,
            Classification::Suspicious | Classification::Hostile
        ) && let Some(sha) = ef["sha"].as_str()
            && let Some(&(src_sha, src_off, locator)) = fetched_by_content.get(sha)
        {
            dep_backrefs.push((
                src_sha.to_string(),
                src_off,
                locator.to_string(),
                ef_decision.class,
            ));
        }

        let full_path = ef["path"].as_str().unwrap_or("");
        let rel_path = full_path
            .rsplit_once("!!")
            .map(|(_, r)| r)
            .unwrap_or(full_path)
            .to_string();

        let ef_top_findings: Vec<TopFinding> = json_alias_array(ef, &["traits", "find", "ts"])
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

    // Pin each confirmed hostile/suspicious fetched dependency back onto the
    // file that declared it, at the reference byte — so the manifest carries a
    // citable trait naming the bad dependency, and prism's galaxy view can light
    // the offending edge. Display/attribution only; the verdict was already
    // elevated by the embedded pass above.
    for (src_sha, src_off, locator, class) in &dep_backrefs {
        inject_dependency_backref(
            &mut report_json,
            &report,
            src_sha,
            *src_off,
            locator,
            *class,
        );
    }

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
    // escalation (L4 hostile / L99 suspicious) or clears a benign (L-1).
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
    let rendered_context = if !render_context {
        String::new()
    } else if tiny_opts.header == cleave::output::HeaderStyle::Rich {
        render_terminal_context(
            &report,
            tiny_opts,
            &final_decision,
            &reasons,
            interpretation.as_ref(),
            &sha256,
            label,
            bloom_mark,
        )
    } else {
        cleave::output::format_context(&report, tiny_opts)
    };

    // Surface the archive members cleave catalogued but never analyzed, so a
    // `--show=all` JSON manifest lists every file (path/type/size) — not just the
    // ones that produced findings. Appended last, after featurization and the
    // embedded-file pass have consumed `report_json`, so the listing never feeds
    // the model. Empty unless `--show=all` requested the manifest.
    if !listed_members.is_empty() {
        append_unanalyzed_members(&mut report_json, &listed_members);
    }

    // Mirror each fetched dependency into hopper as its own sample: its standalone
    // report plus the aggregate verdict harvested above. A dependency the embedded
    // pass never reached (e.g. truncated by the embedded-file cap) defaults to
    // benign rather than inventing a verdict.
    let dependency_results: Vec<DepResult> = fetched_deps
        .into_iter()
        .zip(dep_decisions)
        .map(|(dep, decision)| {
            let decision = decision.unwrap_or(Decision {
                class: Classification::Benign,
                probability: 0.0,
                threshold: 0.0,
                level: Some(-1),
            });
            DepResult {
                sha256: dep.content_sha,
                locator: dep.locator,
                url: dep.url,
                size: dep.size,
                level: decision.level,
                probability: decision.probability,
                raw: dep.raw,
            }
        })
        .collect();

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
        dependency_results,
        rendered_context,
        interpretation,
    })
}

/// A minimal archive-member record captured before `finalize()` discards
/// `archive_contents`. Carries only what a `--show=all` listing needs.
struct ArchiveMemberStub {
    /// Archive-relative path (single `!` between nesting levels, as cleave
    /// records it in `archive_contents`).
    path: String,
    /// Detected file type (e.g. "markdown", "png").
    file_type: String,
    /// SHA256 of the member's contents — the key used to skip members that were
    /// already analyzed and therefore already present in `files[]`.
    sha256: String,
    /// Uncompressed size in bytes.
    size_bytes: u64,
}

/// Append the archive members cleave never analyzed to `report_json["files"]` as
/// listing-only entries (`id`/`path`/`type`/`sha`/`size`/`depth`). Members that
/// were analyzed — matched by sha256 — are already present and skipped. Each
/// appended entry carries a sentinel `risk` of -1 so a consumer can tell an
/// unanalyzed listing (-1) apart from an analyzed member that simply produced no
/// traits (0); [`build_ml_files`] reads the same sentinel to keep these out of
/// the classified `ml.files` array.
fn append_unanalyzed_members(report_json: &mut serde_json::Value, members: &[ArchiveMemberStub]) {
    let Some(files) = report_json
        .get_mut("files")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };
    let mut seen: std::collections::HashSet<String> = files
        .iter()
        .filter_map(|f| f.get("sha").and_then(|v| v.as_str()).map(str::to_owned))
        .collect();
    // Compact paths join nesting levels with `!!` under the root file's path;
    // `archive_contents` paths are archive-relative with single `!`. Re-root and
    // normalize so the listing entries match their analyzed siblings.
    let root_path = files
        .first()
        .and_then(|f| f.get("path"))
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_owned();
    let mut next_id = files
        .iter()
        .filter_map(|f| f.get("id").and_then(serde_json::Value::as_u64))
        .max()
        .map_or(0, |m| m + 1);
    for m in members {
        if m.sha256.is_empty() || !seen.insert(m.sha256.clone()) {
            continue;
        }
        let path = format!("{root_path}!!{}", m.path.replace('!', "!!"));
        let depth = m.path.matches('!').count() as u64 + 1;
        files.push(serde_json::json!({
            "id": next_id,
            "path": path,
            "type": m.file_type,
            "sha": m.sha256,
            "size": m.size_bytes,
            "depth": depth,
            "risk": -1,
        }));
        next_id += 1;
    }
}

/// Pin a fetched dependency's verdict onto the file that declared it: push a
/// synthetic trait onto the declaring file's compact entry, at the reference's
/// byte span, naming the dependency and its class. The verdict was already
/// elevated upstream; this is the citable back-reference (and the galaxy edge
/// prism lights). The span length comes from the declared reference's evidence
/// in the typed report; the offset is the reference site.
fn inject_dependency_backref(
    report_json: &mut serde_json::Value,
    report: &cleave::AnalysisReport,
    source_sha: &str,
    source_offset: Option<u64>,
    locator: &str,
    class: Classification,
) {
    let (crit, word) = match class {
        Classification::Hostile => (5u8, "hostile"),
        _ => (4u8, "suspicious"),
    };
    let off = source_offset.unwrap_or(0);
    let len = report
        .files
        .iter()
        .find(|f| f.sha256 == source_sha)
        .and_then(|f| f.filefacts.as_ref())
        .and_then(|ff| ff.references.iter().find(|r| r.offset == off))
        .map_or(1, |r| u64::try_from(r.evidence.len()).unwrap_or(u64::MAX));
    // Friendly name: drop the `pkg:npm/` prefix and decode the `%40` scope.
    let dep = locator
        .strip_prefix("pkg:npm/")
        .unwrap_or(locator)
        .replace("%40", "@");
    let new_trait = serde_json::json!({
        "id": "fetch/dependency-verdict",
        "crit": crit,
        "desc": format!("Fetched dependency {dep} classified {word}"),
        "spans": [[off, len]],
    });
    let Some(files) = report_json.get_mut("files").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for f in files {
        if f.get("sha").and_then(serde_json::Value::as_str) != Some(source_sha) {
            continue;
        }
        match f
            .get_mut("traits")
            .and_then(serde_json::Value::as_array_mut)
        {
            Some(traits) => traits.push(new_trait),
            None => {
                if let Some(obj) = f.as_object_mut() {
                    obj.insert("traits".to_string(), serde_json::json!([new_trait]));
                }
            }
        }
        return;
    }
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
#[allow(clippy::too_many_arguments)] // mirrors classify_report's wide single-path signature
pub(crate) fn process_report(
    path: &Path,
    report: cleave::AnalysisReport,
    ctx: &ExtractContext,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
    cancellation: Option<&Arc<AtomicBool>>,
    root_registry: Option<&fletch::Registry>,
    bloom_mark: Option<crate::output::BloomMark>,
) -> Result<ScanResult> {
    let path_display = path.display().to_string();
    let is_json = matches!(config.format(), OutputFormat::Json);
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
        // Render the live-vs-cache fetch log only on the interactive terminal
        // path; JSON and tiny output stay machine-clean (the edges ride along in
        // the JSON `fetched` array regardless).
        matches!(config.format(), OutputFormat::Terminal),
        !is_json,
        // `--show=all` with JSON output: list every archive member, even the
        // no-finding ones cleave skipped analyzing.
        config.filter().is_all() && is_json,
        root_registry,
        bloom_mark,
    )?;

    // Retain the raw cleave report for JSON output, and whenever uploading to
    // hopper — the renewed result must carry the full report (so hopper stores it
    // and explodes archive members), and the fetch edges inside it drive the
    // content reconciliation in `record_file_result`. Without this, a terminal
    // `--upload` would post a verdict with an empty report.
    let cleave = if is_json || config.hopper().is_some() {
        Some(cr.report_json)
    } else {
        None
    };

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
        dependency_results: cr.dependency_results,
        bloom_mark,
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
    prefetch_cleave_resources();

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
        None,
        None,
    )
}

fn prefetch_cleave_resources() {
    // Keep cleave's cold-start work off rayon workers where possible. Cleave's
    // loaders are worker-safe, but these library entry points are also used by
    // pkg/url scans and fetched payload analysis where visible latency matters.
    cleave::prefetch_shared_resources(true);
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
        // Listing-only members (added for `--show=all`) were never analyzed, so
        // they carry no ML verdict; their sentinel `risk` of -1 keeps them out of
        // the classified `ml.files` while they remain in the raw `files` manifest.
        if entry.get("risk").and_then(serde_json::Value::as_i64) == Some(-1) {
            continue;
        }
        let id = entry
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(idx as u64);
        let depth = json_alias(entry, &["depth", "dp"])
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
