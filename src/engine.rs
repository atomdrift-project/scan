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
        let ml = build_ml_files(&report, 0.9, Some(100), &MemberEvals::new());
        let ids: Vec<u64> = ml.iter().map(|f| f["id"].as_u64().unwrap()).collect();
        assert_eq!(
            ids,
            vec![0, 1],
            "listing-only member is excluded from ml.files"
        );
    }

    #[test]
    fn build_ml_files_never_stamps_members_with_root_verdict() {
        // A hostile container with three members: one evaluated benign, one
        // evaluated with the same basename as another node, one never
        // evaluated. No member may inherit the root's verdict — hopper mirrors
        // these entries into each member's own sample row, and a member can
        // occur in many containers.
        let report = serde_json::json!({
            "files": [
                {"id": 0, "path": "evil.elf", "type": "elf", "depth": 0},
                {"id": 1, "path": "compatibility", "type": "unknown", "depth": 1},
                {"id": 2, "path": "a!!page", "type": "unknown", "depth": 2},
                {"id": 3, "path": "b!!page", "type": "unknown", "depth": 2},
                {"id": 4, "path": "never-scored", "type": "unknown", "depth": 1},
            ]
        });
        let member = |id: u64, path: &str, prob: f32, level: i32| EmbeddedFile {
            id,
            sha256: String::new(),
            path: path.to_string(),
            file_type: "unknown".to_string(),
            classification: Classification::Benign,
            probability: prob,
            threshold: 0.8,
            level: Some(level),
            model_scores: Vec::new(),
            skipped_models: Vec::new(),
            formula: String::new(),
            top_findings: Vec::new(),
        };
        let evaluated = MemberEvals::from([
            (1, member(1, "compatibility", 0.00001, -1)),
            (2, member(2, "page", 0.00002, -1)),
            (3, member(3, "page", 0.7, 3000)),
        ]);
        let ml = build_ml_files(&report, 0.99, Some(0), &evaluated);

        assert!(
            (ml[0]["prob"].as_f64().unwrap() - 0.99).abs() < 1e-6,
            "root keeps its own"
        );
        assert!(
            (ml[1]["prob"].as_f64().unwrap() - 0.00001).abs() < 1e-9,
            "evaluated member reports its own probability, not the root's"
        );
        assert_eq!(ml[1]["lvl"].as_i64(), Some(-1));
        // Same basename, different nodes: id keys the join, so each reports
        // its own evaluation (the old path-suffix match returned the first).
        assert!((ml[2]["prob"].as_f64().unwrap() - 0.00002).abs() < 1e-9);
        assert_eq!(ml[3]["lvl"].as_i64(), Some(3000));
        // Never evaluated: no verdict fields at all — absence, not inheritance.
        assert_eq!(ml[4]["id"].as_u64(), Some(4));
        assert!(
            ml[4].get("prob").is_none()
                && ml[4].get("lvl").is_none()
                && ml[4].get("conf").is_none(),
            "unevaluated member carries no fabricated verdict"
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
    fn one_confident_crit5_escalates_to_hostile_band() {
        let mut d = benign();
        apply_trait_floor(&mut d, &report(5, 0.8, 1, 0), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.probability, 0.8);
        assert_eq!(d.level, Some(50));
    }

    #[test]
    fn low_confidence_crit5_is_ignored() {
        let mut d = benign();
        // c < 0.76 → not counted, stays benign.
        apply_trait_floor(&mut d, &report(5, 0.5, 3, 0), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn two_confident_crit4_above_fraction_escalates_to_suspicious_band() {
        let mut d = benign();
        // 4 confident crit-4 out of 4 total → fraction 1.0 >= 0.05.
        apply_trait_floor(&mut d, &report(4, 0.9, 4, 0), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Suspicious);
        assert_eq!(d.probability, 0.9);
        assert_eq!(d.level, Some(100));
    }

    #[test]
    fn confident_crit4_below_fraction_stays_benign() {
        let mut d = benign();
        // 2 confident crit-4 diluted by 200 baseline findings → fraction ~0.01.
        apply_trait_floor(&mut d, &report(4, 0.9, 2, 200), Some(50), 100, "test");
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
        apply_trait_floor(&mut d, &report, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn missing_confidence_defaults_below_threshold() {
        let mut d = benign();
        // `conf` omitted → DEFAULT_TRAIT_CONFIDENCE (0.5) < 0.76, so it never counts.
        let report = json!({"find": [{"crit": 5}, {"crit": 5}]});
        apply_trait_floor(&mut d, &report, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn never_lowers_a_non_benign_verdict() {
        let mut d = benign();
        d.class = Classification::Hostile;
        d.level = Some(50);
        apply_trait_floor(&mut d, &report(5, 0.9, 5, 0), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.level, Some(50));
    }

    #[test]
    fn reads_findings_from_embedded_files_shape() {
        let mut d = benign();
        let report = json!({"files": [{"find": [{"crit": 5, "conf": 0.8}]}]});
        apply_trait_floor(&mut d, &report, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.level, Some(50));
    }

    #[test]
    fn reads_findings_from_current_compact_traits_shape() {
        let mut d = benign();
        let report = json!({"files": [{"traits": [{"crit": 5, "conf": 0.98}]}]});
        apply_trait_floor(&mut d, &report, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.probability, 0.98);
        assert_eq!(d.level, Some(50));
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
        let arts = collect_upload_artifacts(
            Path::new("/tmp/proj.tgz"),
            &"a".repeat(64),
            10,
            "scan+test",
            None,
        );

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
    fn collect_upload_artifacts_backfills_preserved_root_provenance() {
        let provenance = crate::provenance::registry_provenance(
            br#"{"record":{"ecosystem":"npm","name":"proj","version":"1.0.0"},"sources":[{"url":"https://registry.example/proj","status":200,"body":{"provider_only":42}}]}"#,
        )
        .unwrap();
        let arts = collect_upload_artifacts(
            Path::new("/tmp/proj.tgz"),
            &"a".repeat(64),
            10,
            "scan+test",
            Some(&provenance),
        );
        let sidecar: serde_json::Value = serde_json::from_slice(&arts[0].sidecar).unwrap();
        assert!(arts[0].backfill);
        assert_eq!(sidecar["registry"]["raw"][0]["body"]["provider_only"], 42);
    }

    /// A dependency carrying the verdict scan computed for it.
    fn evaluated_dep() -> DepResult {
        DepResult {
            sha256: "d".repeat(64),
            locator: "pkg:npm/evil@1.0.0".to_string(),
            url: "https://reg/evil-1.0.0.tgz".to_string(),
            size: 1234,
            provenance: None,
            verdict: Some(Decision {
                class: Classification::Hostile,
                probability: 0.97,
                threshold: 0.65,
                level: Some(100),
            }),
            members: MemberEvals::new(),
            raw: serde_json::json!({"v": "8", "files": [{"sha": "d".repeat(64), "type": "npm"}]})
                .to_string(),
        }
    }

    #[test]
    fn dep_envelope_carries_verdict_and_report() {
        // A dependency's verdict envelope encodes its aggregate level as `ml.lvl`
        // and passes its standalone report through as `raw`, so hopper keeps the
        // dependency's own analysis rather than a slice of its parent's.
        let env = dep_envelope(&evaluated_dep(), "model-9", "2026-06-28T00:00:00Z")
            .expect("a dependency with a verdict yields an envelope");
        assert_eq!(env.ml.level, Some(100), "aggregate verdict rides in ml.lvl");
        assert_eq!(env.ml.probability, 0.97);
        assert_eq!(env.ml.version, "model-9");
        assert_eq!(env.ml.analyzed_at, "2026-06-28T00:00:00Z");
        assert_eq!(
            env.raw["files"][0]["type"], "npm",
            "the dependency's own report is the result raw, so hopper keeps its FileType",
        );
    }

    #[test]
    fn dep_envelope_absent_without_a_verdict() {
        // A dependency the embedded pass never reached has no verdict, so there is
        // no result to post. Inventing a benign one would be indistinguishable
        // from a real evaluation — and would bless the package in the known-good
        // bloom filter, suppressing every future fetch of it. The bytes and
        // provenance still upload, so hopper holds the artifact and can analyze it.
        let dep = DepResult {
            verdict: None,
            ..evaluated_dep()
        };
        assert!(
            dep_envelope(&dep, "model-9", "2026-06-28T00:00:00Z").is_none(),
            "an unevaluated dependency posts no verdict",
        );
    }

    pub(super) fn base_result() -> ScanResult {
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
            embedded_files: MemberEvals::new(),
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
    fn interpreted_level_pins_to_the_single_band_boundaries() {
        use crate::model::{capped_suspicious_level, verdict_for_level};
        use Classification::{Benign, Hostile, Suspicious};

        // There is exactly one hostile and one suspicious threshold on the grid,
        // so an interpreted level is pinned to the loosest rung of the target
        // band — derived from the model's live thresholds (active level +
        // suspicious ceiling), never hardcoded, so it tracks the boundary even if
        // we move the hostile level or the suspicious ceiling.
        let grid_max = 30_000;
        for deploy in [4_u16, 5, 25, 50] {
            // Escalation to hostile lands on the active deploy level: the loosest
            // rung still inside the hostile budget.
            assert_eq!(
                interpreted_level(Some(deploy), grid_max, Hostile),
                Some(i32::from(deploy))
            );
            // Hold/downgrade to suspicious lands on the suspicious ceiling.
            assert_eq!(
                interpreted_level(Some(deploy), grid_max, Suspicious),
                Some(i32::from(capped_suspicious_level(grid_max)))
            );
            // Each round-trips through the model's own classifier into exactly the
            // class the LLM lifted it to — the guarantee that keeps a `lvl`-based
            // consumer (e.g. hopper) from re-reading an escalation as the wrong class.
            let hostile_lvl =
                u16::try_from(interpreted_level(Some(deploy), grid_max, Hostile).unwrap()).unwrap();
            let susp_lvl =
                u16::try_from(interpreted_level(Some(deploy), grid_max, Suspicious).unwrap())
                    .unwrap();
            assert_eq!(verdict_for_level(hostile_lvl, deploy, grid_max), Hostile);
            assert_eq!(verdict_for_level(susp_lvl, deploy, grid_max), Suspicious);
        }
        // A grid tighter than the ceiling caps the suspicious rung at grid_max.
        assert_eq!(
            interpreted_level(Some(25), 2_000, Suspicious),
            Some(i32::from(capped_suspicious_level(2_000)))
        );
        // Benign is the clean marker regardless of grid (even in manual mode).
        assert_eq!(interpreted_level(Some(25), grid_max, Benign), Some(-1));
        assert_eq!(interpreted_level(None, 0, Benign), Some(-1));
        // Manual-threshold mode (no grid): no synthetic hostile/suspicious level.
        assert_eq!(interpreted_level(None, 0, Hostile), None);
        assert_eq!(interpreted_level(None, 0, Suspicious), None);
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
        let member = |id: u64, path: &str, level: Option<i32>, prob: f32| EmbeddedFile {
            id,
            sha256: String::new(),
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
        r.embedded_files = MemberEvals::from([
            (1, member(1, "evil.sh", Some(2), 0.99)),
            (2, member(2, "readme.txt", Some(-1), 0.01)),
        ]);
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

/// False-positive level for an LLM-driven verdict, used when the blend shifts the
/// class away from ML's. There is exactly one hostile threshold and one
/// suspicious threshold on the grid, so the interpreted level is pinned to the
/// *loosest rung of the target band* — the boundary the model itself uses:
/// - **Hostile** → the active deploy level (`-l`): the weakest level still inside
///   the hostile budget. Anything stricter is also hostile, so the band's max
///   threshold is the honest place for an LLM escalation that ML didn't reach.
/// - **Suspicious** → `capped_suspicious_level(grid_max)` = `min(grid_max, ceiling)`:
///   the suspicious ceiling, the weakest suspicious rung.
/// - **Benign** → `-1`, the clean marker.
///
/// Deriving from the model's `active_level` / `grid_max` keeps the interpreted
/// level in lockstep with the model's real thresholds — the same values
/// [`crate::model::verdict_for_level`] classifies against — so escalation and
/// de-escalation land in the right band even if the hostile line or suspicious
/// ceiling moves. Since `ml.conf` is [`level_confidence`] of the level, the
/// displayed confidence tracks automatically. `active_level` is `None` in
/// manual-threshold mode (no grid); hostile/suspicious then return `None`,
/// matching how a genuine ML verdict serializes its level there.
fn interpreted_level(
    active_level: Option<u16>,
    grid_max: u16,
    outcome: Classification,
) -> Option<i32> {
    match outcome {
        Classification::Benign => Some(-1),
        Classification::Hostile => active_level.map(i32::from),
        Classification::Suspicious => {
            active_level.map(|_| i32::from(crate::model::capped_suspicious_level(grid_max)))
        }
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
    /// Per-file ML evaluations for archive members, keyed by node id.
    /// See [`MemberEvals`] — the single source of truth for member verdicts.
    pub embedded_files: MemberEvals,
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
/// compact cleave report and the registry provenance captured during analysis.
/// Only artifact bytes are recovered lazily from the fetch blob cache at upload
/// time.
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
    /// The exact registry snapshot used during analysis. `None` for URL fetches,
    /// unsupported registries, or failed lookups. Cloning is cheap because the
    /// opaque document is refcounted compact bytes.
    pub provenance: Option<crate::provenance::RegistryProvenance>,
    /// The dependency's aggregate verdict: its container elevated by its worst
    /// member, exactly as a first-hand scan of the same bytes resolves. `None`
    /// when the embedded pass never reached it — its bytes and provenance are
    /// still stored, so hopper can analyze it, but scan posts no verdict it did
    /// not compute. A fabricated one would be indistinguishable from a real
    /// evaluation, and a fabricated *benign* would bless the package in the
    /// known-good bloom filter, suppressing every future fetch of it.
    pub verdict: Option<Decision>,
    /// This dependency's own per-member evaluations, keyed by node id — the same
    /// table a first-hand scan carries in `ScanResult::embedded_files`, and what
    /// `ml.files` is built from.
    ///
    /// Without it every member of a dependency reached hopper with no verdict at
    /// all, while the members of a directly-scanned package got theirs. Bounded
    /// by EMBEDDED_FILE_LIMIT, so the retained size is capped per dependency
    /// rather than growing with the report.
    pub members: MemberEvals,
    /// The dependency's own compact cleave report as JSON text — the `raw`
    /// for its result, parsed transiently at envelope build (see
    /// `FetchedDependency::raw` for why text form).
    pub raw: String,
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
/// failing that, the first file entry of a compact envelope (`files[0].traits`).
fn report_findings(report: &serde_json::Value) -> Option<&Vec<serde_json::Value>> {
    json_alias_array(report, &["traits", "find", "ts"]).or_else(|| {
        json_alias_array(report, &["files", "fs"])
            .and_then(|a| a.first())
            .and_then(|f| json_alias_array(f, &["traits", "find", "ts"]))
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

/// Every member evaluation a scan produced, keyed by cleave node id — the
/// single source of truth. `ml.files`, container elevation, dependency
/// verdicts, and the diagnostics views all derive from this table. A node
/// absent here was not evaluated; no consumer may invent a verdict for it
/// (a member can occur in many containers, and hopper mirrors per-member
/// entries into the member's own sample row).
pub type MemberEvals = std::collections::BTreeMap<u64, EmbeddedFile>;

/// The worst (highest-outranking) member evaluation, or `None` when no member
/// was evaluated. Iteration is id order == report entry order, so ties keep
/// the earliest member exactly as the old in-loop fold did.
fn worst_member(evals: &MemberEvals) -> Option<Decision> {
    evals
        .values()
        .map(EmbeddedFile::decision)
        .reduce(|best, d| {
            if decision_outranks(&d, &best) {
                d
            } else {
                best
            }
        })
}

/// A file embedded within an archive or self-extracting executable.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbeddedFile {
    /// The cleave `files[].id` of this member — the stable key that ties this
    /// evaluation back to its report node. Paths are not unique (many members
    /// share a basename; fetched payloads collide across pages), so id is the
    /// only safe join key for `ml.files`.
    pub id: u64,
    /// SHA-256 of this member's bytes — the content key sha-keyed consumers
    /// (dependency roll-up, fetch backrefs) join on.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub sha256: String,
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

impl EmbeddedFile {
    /// This member's verdict as a [`Decision`], for outranking comparisons.
    fn decision(&self) -> Decision {
        Decision {
            class: self.classification,
            probability: self.probability,
            threshold: self.threshold,
            level: self.level,
        }
    }

    /// The `n` highest-probability evaluations, for diagnostics displays.
    /// A view — the table itself is never sorted or truncated.
    pub(crate) fn top_offenders(evals: &MemberEvals, n: usize) -> Vec<&EmbeddedFile> {
        let mut v: Vec<&EmbeddedFile> = evals.values().collect();
        v.sort_by(|a, b| b.probability.total_cmp(&a.probability));
        v.truncate(n);
        v
    }
}

pub(crate) const SPINNER: &[char] = &[
    '\u{2800}', '\u{2801}', '\u{2809}', '\u{2819}', '\u{281B}', '\u{281E}', '\u{2816}', '\u{2812}',
    '\u{2810}', '\u{2800}',
];

/// Heartbeat redraw cadence. A background thread redraws on this interval so
/// the spinner keeps animating — and the long-tail notice can appear — even
/// while no file completes (a single slow rizin analysis can stall for minutes).
/// Shared with [`crate::deptree`], whose live tree animates on the same cadence.
pub(crate) const PROGRESS_TICK: Duration = Duration::from_millis(125);

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

/// Terminal size as `(cols, rows)`, for capping the progress line and bounding
/// the live dependency tree. Falls back to `(80, 24)` when the size can't be
/// queried (not a tty, or the ioctl fails).
pub(crate) fn term_dims() -> (usize, usize) {
    #[cfg(unix)]
    {
        // SAFETY: TIOCGWINSZ fills a zero-initialised `winsize`; we trust the
        // result only when the ioctl reports success and a non-zero width.
        unsafe {
            let mut ws: libc::winsize = std::mem::zeroed();
            if libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
                let rows = if ws.ws_row > 0 {
                    ws.ws_row as usize
                } else {
                    24
                };
                return (ws.ws_col as usize, rows);
            }
        }
    }
    (80, 24)
}

/// Terminal width in columns, for capping the progress line. Falls back to 80
/// when the width can't be queried (not a tty, or the ioctl fails).
fn term_cols() -> usize {
    term_dims().0
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

/// Whether an interactive scan progress bar currently owns the terminal. The
/// live dependency tree ([`crate::deptree`]) defers to it: a multi-file scan's
/// bar keeps the streamed fetch log (which coexists with the bar via
/// [`print_above_bar`]), so only a single-artifact scan — where no bar is live —
/// takes over stderr with an in-place tree.
pub(crate) fn bar_active() -> bool {
    ACTIVE_BAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .and_then(Weak::upgrade)
        .is_some()
}

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
    root_registry: Option<&crate::provenance::RegistryProvenance>,
    root_fetch: Option<&fletch::fetch::FetchRecord>,
) -> Result<ScanSummary> {
    // Deliberately no `prefetch_cleave_resources()` here. This one-shot single-
    // artifact path (`pkg:`/`url`) never fans out across rayon, and its analyses
    // usually hit cleave's report cache — so the capability mapper's match
    // indexes (a multi-hundred-ms regex build) are left to build lazily only if
    // an analysis actually misses, and are skipped entirely on a warm scan. The
    // lazy build then runs on this main thread, so there is no rayon re-entrancy
    // to guard against. Directory/worker paths still prefetch (they do fan out).

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
        root_fetch,
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
    let recent = |t: SystemTime| {
        now.duration_since(t)
            .map_or(true, |d| d.as_secs() <= window)
    };
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
    root_registry: Option<&crate::provenance::RegistryProvenance>,
    root_fetch: Option<&fletch::fetch::FetchRecord>,
) {
    // Bloom verdict, re-derived from the root sha cleave computed. A file cleave
    // skipped at our request (a stale known-good, or fast-mode unknown) arrives as
    // a minimal report and is short-circuited here; a known-bad/conflicted file —
    // or a fresh known-good scanned on its own merits — was analyzed and carries a
    // provenance marker on its SHA-256 line, derived from the same decision.
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
        // already produced a full report — fall through to the normal scan and mark
        // it ✓ known-good. A stale known-good (and, in fast mode, an unknown)
        // arrived as a minimal report and is counted here without an ML pass.
        let fresh_known_good = decision == BloomDecision::Skip
            && file_touched_within(file_path, KNOWN_GOOD_RESCAN_SECS, SystemTime::now());
        if !fresh_known_good
            && let Some(summary) = bloom_gate(config, &file_path.display().to_string(), decision)
        {
            if summary.benign > 0 {
                tally.count(Classification::Benign);
            }
            if let Some(p) = progress {
                p.increment();
            }
            return;
        }
        // Known-bad/conflicted, or a fresh known-good — annotate the SHA-256 line.
        // Non-fast unknown maps to `None` (unremarkable, matched neither set).
        bloom_mark = crate::output::BloomMark::from_decision(decision);
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
            root_fetch,
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
            // JSON, tiny, and interpret are machine/LLM payload formats: they
            // emit every scanned file — `--show` gates only the terminal view.
            if matches!(
                config.format(),
                OutputFormat::Json | OutputFormat::Tiny | OutputFormat::Interpret
            ) || r.bloom_mark.is_some()
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
                upload_scan_result(
                    uploader,
                    file_path,
                    sha256,
                    size,
                    root_registry,
                    deps,
                    envelope,
                );
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
    root_provenance: Option<&crate::provenance::RegistryProvenance>,
    dependency_results: Vec<DepResult>,
    envelope: ScanResultEnvelope,
) {
    let artifacts = collect_upload_artifacts(
        file_path,
        &sha256,
        size_bytes,
        upload_collector(),
        root_provenance,
    );
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

/// Feature-extract and model-score a report's embedded files — the archive
/// members at depth > 0 — returning one evaluation per entry.
///
/// Extracted so a fetched dependency can be graded on its own report rather
/// than as a tail region of the report it was grafted into. Sharing the parent's
/// pass meant sharing its embedded-file budget, and a dependency past the cap
/// came back with no evaluation at all.
///
/// Per-member work is pure and runs in parallel: reports with thousands of
/// embedded files (nested npm tarballs, fetched dependency trees) previously ran
/// this serially on one rayon worker, and on member-heavy archives that pass —
/// not cleave's analysis — was the scan's wall-clock tail.
fn score_embedded_files(
    entries: &[&serde_json::Value],
    label: &str,
    needs: crate::features::RawNeeds,
    ctx: &ExtractContext,
    model: &Model,
    cancellation: Option<&Arc<AtomicBool>>,
) -> Vec<EmbeddedFile> {
    use rayon::prelude::*;
    entries
        .par_iter()
        .map(|&ef| {
            // Cancellation: skip the expensive work; the post-pass bail
            // below surfaces the cancellation before results are used.
            let (ef_decision, ef_model_scores, ef_skipped_models) =
                if cancellation.is_some_and(|c| c.load(Ordering::Relaxed)) {
                    (
                        Decision {
                            class: Classification::Benign,
                            probability: 0.0,
                            threshold: model.thresholds().suspicious,
                            level: None,
                        },
                        Vec::new(),
                        Vec::new(),
                    )
                } else {
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
                        model.active_level(),
                        model.grid_max(),
                        ef["path"].as_str().unwrap_or(label),
                    );
                    (ef_decision, ef_model_scores, ef_skipped_models)
                };

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
            EmbeddedFile {
                // u64::MAX when the node has no id: it then matches no report
                // node rather than falsely matching id 0 (the root).
                id: ef["id"].as_u64().unwrap_or(u64::MAX),
                sha256: ef["sha"].as_str().unwrap_or("").to_string(),
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
            }
        })
        .collect()
}

/// How many embedded files one report contributes to the model pass. Bounds the
/// per-report work on an archive with thousands of members.
///
/// Per report, not per scan: a fetched dependency is graded on its own report and
/// gets its own budget. Sharing the parent's — the whole merged tree competing
/// for one allowance, dependencies appended last — meant a large package's
/// dependencies fell off the end and came back ungraded.
pub(crate) const EMBEDDED_FILE_LIMIT: usize = 100;

/// Grade a fetched dependency on its own standalone report: the container's own
/// verdict, elevated by its worst member. That is exactly how a first-hand scan
/// of the same bytes resolves, which is the point — a dependency is an artifact
/// someone else's manifest happened to name, not a region of the report it was
/// grafted into.
///
/// It was previously graded by attributing the parent's embedded pass back to it,
/// which made its verdict depend on where it landed in the parent's file list and
/// whether the shared budget reached that far.
///
/// The report is parsed here and dropped on return: only the verdict outlives the
/// call, so grading a dependency costs a transient parse rather than a retained
/// tree — the same trade `dep_envelope` makes when it builds the POST body.
///
/// `None` when the report will not parse or the feature vector does not match the
/// model; the caller reports no verdict rather than inventing one.
fn classify_dependency(
    raw: &str,
    label: &str,
    ctx: &ExtractContext,
    model: &Model,
    cancellation: Option<&Arc<AtomicBool>>,
) -> Option<(Decision, MemberEvals)> {
    let report_json: serde_json::Value = serde_json::from_str(raw).ok()?;
    let needs = ctx.raw_needs().union(crate::features::RawNeeds::all());

    // The container's own verdict. No own_shas filtering: this report is the one
    // cleave produced for the dependency's bytes alone, before anything was
    // grafted onto it, so every file in it is the dependency's own.
    let parsed = crate::features::ParsedReport::from_report(&report_json, needs);
    let mut features = ctx.extract_from_parsed(&parsed);
    if features.len() != model.spec().total_features() {
        return None;
    }
    model.spec().standardize(&mut features);
    let pf = json_alias_array(&report_json, &["files", "fs"])
        .and_then(|a| a.first())
        .unwrap_or(&report_json);
    let file_type = pf["type"].as_str().unwrap_or("unknown");
    let (mut decision, _, _) = model
        .predict_for_report_detailed(file_type, &features, &parsed)
        .ok()?;
    apply_trait_floor(
        &mut decision,
        &report_json,
        model.active_level(),
        model.grid_max(),
        label,
    );

    // Its own members, on its own budget, elevating it as they would any
    // container.
    let entries: Vec<&serde_json::Value> = json_alias_array(&report_json, &["files", "fs"])
        .into_iter()
        .flatten()
        .filter(|f| {
            json_alias(f, &["depth", "dp"])
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                > 0
        })
        .take(EMBEDDED_FILE_LIMIT)
        .collect();
    let mut members = MemberEvals::new();
    for ef in score_embedded_files(&entries, label, needs, ctx, model, cancellation) {
        let member = ef.decision();
        if decision_outranks(&member, &decision) {
            decision = member;
        }
        members.insert(ef.id, ef);
    }
    Some((decision, members))
}

/// Build the `/api/result` envelope for a fetched dependency: the standalone
/// cleave report scan captured for it as `raw`, and the aggregate verdict it
/// computed as the `ml` section — the same shape a first-hand scan of those bytes
/// would post, so hopper records the dependency exactly as if it had been scanned
/// directly. `version`/`analyzed_at` are the parent run's, identifying the build.
///
/// `None` when the dependency carries no verdict (see [`DepResult::verdict`]) —
/// there is nothing to post, and scan does not invent one.
#[must_use]
pub(crate) fn dep_envelope(
    dep: &DepResult,
    version: &str,
    analyzed_at: &str,
) -> Option<ScanResultEnvelope> {
    let verdict = dep.verdict?;
    let level = verdict.level;
    // The report text becomes a `Value` only here, for the short-lived
    // envelope being POSTed — not for the job-long retention window.
    let raw: serde_json::Value =
        serde_json::from_str(&dep.raw).unwrap_or_else(|_| serde_json::json!({}));
    let ml_files = build_ml_files(&raw, verdict.probability, level, &dep.members);
    Some(ScanResultEnvelope {
        ml: MlSection {
            v: "7",
            probability: verdict.probability,
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
        raw,
    })
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
    root_provenance: Option<&crate::provenance::RegistryProvenance>,
) -> Vec<crate::upload::UploadArtifact> {
    use crate::upload::{ArtifactBytes, UploadArtifact};
    let now = now_rfc3339();

    // The scanned file: bytes from disk, no registry (a local, un-fetched artifact).
    let root_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file")
        .to_string();
    let (sidecar, backfill) = if let Some(provenance) = root_provenance {
        (
            crate::provenance::build_sidecar_from_provenance(
                &root_name, sha256, size_bytes, collector, &now, "", "", provenance,
            ),
            true,
        )
    } else {
        (
            crate::provenance::build_sidecar(
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
            false,
        )
    };
    vec![UploadArtifact {
        sha256: sha256.to_string(),
        size: size_bytes,
        filename: root_name.clone(),
        bytes: ArtifactBytes::File(file_path.to_path_buf()),
        sidecar,
        // A map-backed root can carry registry data worth adding to a sample
        // hopper already has; a plain local file's thin sidecar cannot.
        backfill,
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
    Some(
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    )
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
    registry_map: Option<&std::collections::HashMap<String, crate::provenance::RegistryProvenance>>,
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
        // `--format interpret` is the LLM payload verbatim — no verdict line.
        OutputFormat::Interpret => {
            let Ok(mut out) = stdout.lock() else {
                return;
            };
            let _ = out.write_all(r.rendered_context.as_bytes());
        }
    }
}

/// The cleave context density litmus renders at. `--format tiny` uses cleave's
/// full tiny (machine/LLM). The terminal view is cut from cleave's own terminal
/// render — same rich header and body — but capped at the top 3 traits (cleave
/// shows all notable+), with litmus adding a verdict badge + subtitle on top.
pub(crate) fn tiny_opts_for(config: &ScanConfig) -> cleave::output::TinyOpts {
    if matches!(
        config.format(),
        OutputFormat::Tiny | OutputFormat::Interpret
    ) {
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
    // A top-10 view of the evaluation table; the table itself is complete.
    for ef in EmbeddedFile::top_offenders(&r.embedded_files, 10) {
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
        let grade = llm.grade.map_or("?", crate::interpret::LlmGrade::as_str);
        return format!(
            "llm {grade} → {} blended={:.3}  {}\n",
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
    let grade = llm.grade.map_or("?", crate::interpret::LlmGrade::as_str);
    format!(
        "llm {} → {outcome} blended={:.3}  {}\n",
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
    );
    // The filename starts after the stamp and one separator space.
    let indent = badge_w + 1;
    let trailer = crate::output::terminal_trailer(reasons, interpretation);
    // The bloom provenance marker trails the SHA-256 line — the field the filters
    // matched on — rather than the verdict badge.
    let subtitle = crate::output::terminal_subtitle(sha256, indent, bloom_mark);
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

/// Wall-clock of each post-analysis phase inside `classify_report`, in
/// milliseconds. Static analysis (cleave/filefacts) runs *before* this and is not
/// included — subtract `total_ms` from the caller's whole-invocation elapsed to
/// isolate it. Logged per root file on the CLI path (see `process_report`) so a
/// slow scan is self-diagnosing: which phase ate the wall clock (usually the LLM
/// `interpret_ms` when an endpoint is contended, or `fetch_ms` on a wide `--fetch`).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PhaseTimings {
    pub(crate) fetch_ms: u64,
    pub(crate) interpret_ms: u64,
    pub(crate) render_ms: u64,
    pub(crate) total_ms: u64,
}

/// Intermediate classification result from the model pipeline.
/// Produced by `classify_report`, consumed when building a `ScanResult`.
pub(crate) struct ClassifiedReport {
    pub(crate) phase_ms: PhaseTimings,
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
    pub(crate) embedded_files: MemberEvals,
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
    hostile_confidence: f32,
    suspicious_confidence: f32,
    total: u32,
}

fn trait_floor_counts(report: &serde_json::Value) -> TraitFloorCounts {
    let findings = report_findings(report);
    let mut out = TraitFloorCounts {
        hostile: 0,
        suspicious: 0,
        hostile_confidence: 0.0,
        suspicious_confidence: 0.0,
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
            5 => {
                out.hostile += 1;
                out.hostile_confidence = out.hostile_confidence.max(conf);
            }
            4 => {
                out.suspicious += 1;
                out.suspicious_confidence = out.suspicious_confidence.max(conf);
            }
            _ => {}
        }
    }
    out
}

/// Trait floor: override a model-**Benign** verdict when cleave surfaced
/// high-criticality evidence the model did not act on:
///   - one or more hostile (crit-5) traits → **Hostile**
///   - two or more suspicious (crit-4) traits whose fraction clears
///     `TRAIT_FLOOR_CRIT4_FRACTION` → **Suspicious**
///
/// Only confident findings count (`c >= TRAIT_FLOOR_MIN_CONFIDENCE`); the crit-4
/// fraction's denominator is the file's *total* finding count, so a sparse, severe
/// dropper clears it while a busy benign binary with a couple of incidental
/// crit-4s does not.
///
/// Never lowers a model verdict. Override levels are pinned to the same band
/// boundaries used by ordinary and interpreted verdicts, so a level-only
/// downstream consumer cannot reinterpret the override as another class.
fn apply_trait_floor(
    decision: &mut Decision,
    report: &serde_json::Value,
    active_level: Option<u16>,
    grid_max: u16,
    label: &str,
) {
    if decision.class != Classification::Benign {
        return;
    }
    let counts = trait_floor_counts(report);
    if counts.hostile >= 1 {
        decision.class = Classification::Hostile;
        decision.probability = counts.hostile_confidence;
        decision.level = interpreted_level(active_level, grid_max, Classification::Hostile);
        // Loud by design: the model graded this benign yet cleave is confident it
        // carries a hostile (crit-5) trait. If the models are doing their job this
        // is extraordinary — every occurrence is a model gap worth investigating.
        tracing::warn!(
            path = %label,
            arm = "crit5",
            confident_hostile = counts.hostile,
            trait_confidence = format!("{:.3}", counts.hostile_confidence),
            level = ?decision.level,
            "TRAIT FLOOR: model said benign but cleave found a confident hostile trait — escalated to hostile",
        );
        return;
    }
    if counts.suspicious >= 2
        && counts.total > 0
        && counts.suspicious as f32 / counts.total as f32 >= TRAIT_FLOOR_CRIT4_FRACTION
    {
        decision.class = Classification::Suspicious;
        decision.probability = counts.suspicious_confidence;
        decision.level = interpreted_level(active_level, grid_max, Classification::Suspicious);
        tracing::warn!(
            path = %label,
            arm = "crit4_fraction",
            confident_suspicious = counts.suspicious,
            total_findings = counts.total,
            crit4_fraction = format!(
                "{:.3}",
                counts.suspicious as f32 / counts.total as f32
            ),
            trait_confidence = format!("{:.3}", counts.suspicious_confidence),
            level = ?decision.level,
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
    // `--format interpret`: render `rendered_context` as the LLM query payload —
    // byte-for-byte the user message a live `--interpret` query sends (the
    // sanitized tiny render), without the system prompt. Independent of
    // `interpret`, which controls actually querying.
    llm_view: bool,
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
    root_registry: Option<&crate::provenance::RegistryProvenance>,
    // Acquisition provenance for a one-shot `pkg:`/`url` root. Ordinary local
    // scans have no fetch record; their path + content digest still identify
    // their origin in interpret output.
    root_fetch: Option<&fletch::fetch::FetchRecord>,
    // The bloom status flag (🚩 known-bad / 🏴 conflicted), rendered inline in the
    // terminal header. `None` for unflagged files.
    bloom_mark: Option<crate::output::BloomMark>,
) -> Result<ClassifiedReport> {
    // Capture every archive member — including the ones cleave catalogues but
    // never analyzes (docs, data files, images: non-program members it skips by
    // default) — before `finalize()` clears `archive_contents`. With `--show=all`
    // JSON output these are surfaced as listing-only entries below so the manifest
    // is complete; otherwise the snapshot stays empty and nothing changes.
    // Phase stopwatches for the per-file timing line (see PhaseTimings). total
    // spans this whole function (post-analysis: fetch → ML → interpret → render);
    // each phase is timed at its call site below.
    let classify_start = Instant::now();
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
    let fetch_start = Instant::now();
    let (fetch_edges, fetched_deps, dependency_registries) =
        crate::fetch::orchestrate(&mut report, root_path, fetch, fetch_progress);
    let fetch_ms = crate::duration_ms(fetch_start.elapsed());
    // One-shot `pkg:`/`url`: graft the root artifact's own registry metadata as a
    // child `registry` node and correlate the two with a `scope: package`
    // composite. The `--fetch` path does the equivalent per fetched dependency
    // inside `orchestrate`; here the registry record is the root's own. Runs
    // after `orchestrate` (so the registry node sits outside the sample's own
    // `own_shas` aggregate, like other grafted content) and before `strip` (so a
    // package composite's `trait_refs` keep their building-block traits).
    // A metadata-only `pkg` fallback analyzes the `*.registry.json` document as
    // the root itself. Preserve its provenance for interpret, but do not graft a
    // duplicate copy beneath it. Ordinary package artifacts and map/worker
    // inputs still receive the registry sidecar node.
    if let Some(reg) = root_registry
        && root_needs_registry_graft(&report)
    {
        crate::fetch::graft_root_registry(&mut report, &reg.record);
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
    apply_trait_floor(
        &mut decision,
        &report_json,
        model.active_level(),
        model.grid_max(),
        label,
    );

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

    // Fetched payloads keyed by the sha of the content retrieved, with the
    // declaring file + the byte the reference sits at. When the embedded pass
    // classifies one of these hostile/suspicious, that verdict is pinned back
    // onto the declaring manifest at the reference byte (below the pass).
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

    // Feature-extract and model-score members in parallel: reports with
    // thousands of embedded files (nested npm tarballs, fetched dependency
    // trees) previously ran this loop serially on one rayon worker — on
    // member-heavy archives that single-threaded model pass, not cleave's
    // analysis, was the scan's wall-clock tail. Per-member work is pure
    // (&Model is already shared across rayon workers by the outer scan).
    let scored: Vec<EmbeddedFile> =
        score_embedded_files(&embedded_entries, label, needs, ctx, model, cancellation);
    if let Some(c) = cancellation
        && c.load(Ordering::Relaxed)
    {
        anyhow::bail!("analysis cancelled during embedded file processing");
    }

    // The single source of truth for member verdicts: every evaluation, keyed
    // by node id. ml.files, container elevation, dependency verdicts, and the
    // diagnostics views all derive from this table; a node absent from it was
    // not evaluated, and no consumer may invent a verdict for it. Ids ascend
    // in report order, so BTreeMap iteration preserves entry order and the
    // earliest-wins tie-break of `decision_outranks` folds below.
    let mut member_evals = MemberEvals::new();
    for ef in scored {
        tracing::debug!(
            parent = %label,
            embedded_path = %ef.path,
            probability = format!("{:.4}", ef.probability),
            classification = ?ef.classification,
            "classified embedded file",
        );
        member_evals.insert(ef.id, ef);
    }

    // A fetched dependency that classifies hostile/suspicious is pinned back
    // onto the file that declared it (request: the manifest names the bad
    // dependency at its byte). Keyed by content sha so it matches the
    // retrieved payload node.
    let dep_backrefs: Vec<DepBackref> = member_evals
        .values()
        .filter(|ef| {
            matches!(
                ef.classification,
                Classification::Suspicious | Classification::Hostile
            )
        })
        .filter_map(|ef| {
            let &(src_sha, src_off, locator) = fetched_by_content.get(ef.sha256.as_str())?;
            Some(DepBackref {
                source_sha: src_sha.to_string(),
                source_offset: src_off,
                locator: locator.to_string(),
                dep_sha: ef.sha256.clone(),
                dep_type: ef.file_type.clone(),
                class: ef.classification,
            })
        })
        .collect();

    // Elevate the container by its worst member, exactly as before — derived
    // from the table instead of tracked during the loop.
    let max_decision = worst_member(&member_evals)
        .filter(|worst| decision_outranks(worst, &decision))
        .unwrap_or(decision);

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
    for backref in &dep_backrefs {
        inject_dependency_backref(&mut report_json, &report, backref);
    }

    let top_findings = extract_top_findings_from_json(&report_json, &final_decision.class);

    // Mirror each fetched dependency into hopper as its own sample: its standalone
    // report plus the verdict scan computed for it. Graded on its own report by
    // classify_dependency, not attributed back from this report's embedded pass —
    // so its verdict is what a first-hand scan of those bytes produces, rather
    // than a function of where it landed in the merged file list.
    let provenance_by_locator: std::collections::HashMap<
        &str,
        &crate::provenance::RegistryProvenance,
    > = dependency_registries
        .iter()
        .map(|registry| (registry.locator.as_str(), &registry.provenance))
        .collect();
    let dependency_results: Vec<DepResult> = fetched_deps
        .into_iter()
        .map(|dep| {
            let graded = classify_dependency(&dep.raw, &dep.locator, ctx, model, cancellation);
            let (verdict, members) = match graded {
                Some((v, m)) => (Some(v), m),
                None => (None, MemberEvals::new()),
            };
            let provenance = provenance_by_locator
                .get(dep.locator.as_str())
                .map(|provenance| (**provenance).clone());
            DepResult {
                verdict,
                members,
                sha256: dep.content_sha,
                locator: dep.locator,
                url: dep.url,
                size: dep.size,
                provenance,
                raw: dep.raw,
            }
        })
        .collect();

    // LLM second opinion (root file only). Build the cleave tiny render once: it
    // feeds the model (below) and, when `SCAN_INTERPRET_DUMP_DIR` is set, is
    // written to `<dir>/<sha256>.render` — the raw (untransformed) render, so the
    // prompt-tuning harness (`hacks/interpret-tune`) can sweep render variants
    // offline from one scan, independent of whether `--interpret` is on.
    let dump_dir = std::env::var_os("SCAN_INTERPRET_DUMP_DIR");
    let llm_ctx = (interpret.is_some() || dump_dir.is_some() || llm_view).then(|| {
        let rendered = render_interpret_context(
            label,
            &sha256,
            root_fetch,
            root_registry,
            &fetch_edges,
            &dependency_results,
            &dependency_registries,
            &report,
        );
        crate::interpret::sanitize_context(&rendered)
    });
    if let (Some(dir), Some(ctx)) = (dump_dir, llm_ctx.as_deref()) {
        let dir = std::path::Path::new(&dir);
        if std::fs::create_dir_all(dir).is_ok() {
            let _ = std::fs::write(dir.join(format!("{sha256}.render")), ctx);
        }
    }
    let interpret_start = Instant::now();
    let interpretation = interpret.and_then(|cfg| {
        // The gate lives in `interpret::interpret`: it runs when ML clears the
        // probability floor OR cleave surfaced a suspicious/hostile finding ML
        // under-weighted (so an ML-blind packed binary still gets a second
        // opinion); it returns `None` when gated out or on any failure.
        let interp = crate::interpret::interpret(
            cfg,
            llm_ctx.as_deref()?,
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
                interpretation = %interp.interpretation,
                "fetched LLM interpretation",
            );
        }
        Some(interp)
    });
    // Dominant suspect for a slow contended run: the LLM round-trip (queue wait +
    // generation) against a shared endpoint. Zero when `--interpret` is off or gated.
    let interpret_ms = crate::duration_ms(interpret_start.elapsed());

    // Adopt the blended verdict as the effective one when the LLM out-read ML
    // (escalating a missed threat, or clearing an ML false positive). The `ml`
    // section reflects litmus's final answer; the LLM's raw grade + rationale
    // stay in the `llm` section. The interpreted level is pinned to the target
    // band's loosest rung (see `interpreted_level`): the active hostile level for
    // an escalation, the suspicious ceiling for a hold/downgrade, L-1 for benign.
    if let Some(interp) = &interpretation
        && interp.grade.is_some()
        && interp.outcome as u8 != final_decision.class as u8
    {
        // INFO, not WARN: an LLM override of the ML verdict is normal operation,
        // not a fault. (It also kept surfacing as the last stderr line a caller
        // grabbed when a slow run was externally killed, making a benign shift look
        // like a crash cause.)
        tracing::info!(
            path = %label,
            ml = ?final_decision.class,
            outcome = ?interp.outcome,
            grade = interp.grade.map_or("?", crate::interpret::LlmGrade::as_str),
            conf = format!("{:.4}", interp.blended),
            reason = %interp.interpretation,
            "LLM interpretation shifted the verdict",
        );
        final_decision.class = interp.outcome;
        final_decision.probability = interp.blended;
        final_decision.level =
            interpreted_level(model.active_level(), model.grid_max(), interp.outcome);
    }

    // Render cleave's context view now, while the typed (finalized) report is in
    // scope and the verdict (incl. any interpretation) is known. The terminal
    // view extends cleave's rich render with litmus's verdict badge + subtitle;
    // `--format tiny` uses cleave's machine render verbatim.
    let render_start = Instant::now();
    let mut rendered_context = if !render_context {
        String::new()
    } else if llm_view {
        // `--format interpret`: byte-for-byte the user message the live
        // `--interpret` query sends (built above) — the sanitized render,
        // annotations included — just without the system prompt.
        llm_ctx.clone().unwrap_or_default()
    } else if tiny_opts.header == cleave::output::HeaderStyle::Rich {
        let registry_ids: std::collections::HashSet<u32> =
            dependency_registries.iter().map(|r| r.file_id).collect();
        let by_id: std::collections::HashMap<u32, &cleave::FileAnalysis> =
            report.files.iter().map(|f| (f.id, f)).collect();
        let mut primary = report.clone();
        primary.files.retain(|file| {
            fetched_root_id(file, &by_id).is_none() && !registry_ids.contains(&file.id)
        });
        let mut rendered = render_terminal_context(
            &primary,
            tiny_opts,
            &final_decision,
            &reasons,
            interpretation.as_ref(),
            &sha256,
            label,
            bloom_mark,
        );
        if let Some(fetched) = render_terminal_fetch_context(
            &fetch_edges,
            &dependency_results,
            &dependency_registries,
            &report,
            tiny_opts,
        ) {
            rendered.push_str(&fetched);
        }
        rendered
    } else {
        cleave::output::format_context(&report, tiny_opts)
    };
    // The dependency appendix goes to every text format, not just the LLM view.
    // It is the only place a render states that a verdict was inherited from
    // something the sample merely pointed at — naming each dependency, the
    // locator it came from, its own classification, and its elevated findings.
    // Terminal and tiny were left to infer all of that from grafted nodes, which
    // is what made `--format tiny` and `--format interpret` produce the same body
    // with only one of them explaining itself.
    //
    // llm_view already has it: built into llm_ctx above, before sanitize, so the
    // bytes sent to the model stay exactly what that path renders.
    if render_context
        && !llm_view
        && tiny_opts.header != cleave::output::HeaderStyle::Rich
        && let Some(deps) = render_dependency_context(&fetch_edges, &dependency_results, &report)
    {
        rendered_context.push_str(&deps);
    }
    let render_ms = crate::duration_ms(render_start.elapsed());

    // Surface the archive members cleave catalogued but never analyzed, so a
    // `--show=all` JSON manifest lists every file (path/type/size) — not just the
    // ones that produced findings. Appended last, after featurization and the
    // embedded-file pass have consumed `report_json`, so the listing never feeds
    // the model. Empty unless `--show=all` requested the manifest.
    if !listed_members.is_empty() {
        append_unanalyzed_members(&mut report_json, &listed_members);
    }

    Ok(ClassifiedReport {
        phase_ms: PhaseTimings {
            fetch_ms,
            interpret_ms,
            render_ms,
            total_ms: crate::duration_ms(classify_start.elapsed()),
        },
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
        embedded_files: member_evals,
        report_json,
        dependency_results,
        rendered_context,
        interpretation,
    })
}

/// Whether registry metadata is external context that needs grafting beneath
/// the root. A metadata-only package fallback already analyzes the registry
/// document as its root and must not receive a duplicate sidecar node.
fn root_needs_registry_graft(report: &cleave::AnalysisReport) -> bool {
    !report
        .files
        .first()
        .is_some_and(|file| file.file_type == "registry")
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

/// One confirmed hostile/suspicious fetched dependency, ready to pin back onto
/// the file that declared it: the declaring file and reference byte, plus the
/// dependency's identity — locator (PURL/URL), content sha, sniffed file type,
/// and class.
struct DepBackref {
    source_sha: String,
    source_offset: Option<u64>,
    locator: String,
    dep_sha: String,
    dep_type: String,
    class: Classification,
}

/// Pin a fetched dependency's verdict onto the file that declared it — a synthetic
/// trait at the reference's byte span naming the dependency (purl), its content sha,
/// and its class — then roll that trait up every containing archive to the depth-0
/// root, exactly as cleave propagates a member's own traits.
///
/// Without the roll-up the verdict lands only on the manifest (depth > 0), so
/// depth-0-scoped consumers — a triage query's `max_crit`, a caller's `Detected()`
/// heuristic — miss a package whose malice lives entirely in a fetched dependency
/// (a benign wrapper around a hostile transitive dep). The declaring file cites the
/// reference byte span; rolled-up ancestors carry the verdict without a cross-file
/// span. Compact paths nest with `!!`, so an ancestor is any file whose path is a
/// `!!`-boundary prefix of the declaring file's.
///
/// The trait carries the dependency's identity twice: `desc` is prose for humans
/// and the LLM context; `dep` ({locator, sha, type}) is the machine-readable copy
/// that hopper forwards opaquely so prism can render a specific, clickable feed
/// chip ("depends on hostile npm: zaboodle v1.49" → /file/{sha}) without parsing
/// the sentence.
fn inject_dependency_backref(
    report_json: &mut serde_json::Value,
    report: &cleave::AnalysisReport,
    backref: &DepBackref,
) {
    let (crit, sev) = match backref.class {
        Classification::Hostile => (5u8, "Malicious"),
        _ => (4u8, "Suspicious"),
    };
    let off = backref.source_offset.unwrap_or(0);
    let len = report
        .files
        .iter()
        .find(|f| f.sha256 == backref.source_sha)
        .and_then(|f| f.filefacts.as_ref())
        .and_then(|ff| ff.references.iter().find(|r| r.offset == off))
        .map_or(1, |r| u64::try_from(r.evidence.len()).unwrap_or(u64::MAX));
    let desc = format!(
        "{sev} dependency: {} | {}",
        backref.locator, backref.dep_sha
    );
    let dep = serde_json::json!({
        "locator": backref.locator,
        "sha": backref.dep_sha,
        "type": backref.dep_type,
    });

    let Some(files) = report_json.get_mut("files").and_then(|v| v.as_array_mut()) else {
        return;
    };

    // The declaring file's compact path locates every container above it.
    let decl_path = files
        .iter()
        .find(|f| {
            f.get("sha").and_then(serde_json::Value::as_str) == Some(backref.source_sha.as_str())
        })
        .and_then(|f| f.get("path").and_then(serde_json::Value::as_str))
        .map(str::to_owned);

    for f in files.iter_mut() {
        let is_decl =
            f.get("sha").and_then(serde_json::Value::as_str) == Some(backref.source_sha.as_str());
        let is_ancestor = decl_path.as_deref().is_some_and(|dp| {
            f.get("path")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|p| dp.starts_with(&format!("{p}!!")))
        });
        if !is_decl && !is_ancestor {
            continue;
        }
        // The precise declaring file cites the reference byte span; a rolled-up
        // ancestor carries the verdict without a (meaningless) cross-file span.
        let new_trait = if is_decl {
            serde_json::json!({"id": "fetch/dependency-verdict", "crit": crit, "desc": desc.clone(), "dep": dep.clone(), "spans": [[off, len]]})
        } else {
            serde_json::json!({"id": "fetch/dependency-verdict", "crit": crit, "desc": desc.clone(), "dep": dep.clone()})
        };
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
    }
}

/// Cap on elevated-finding lines rendered per fetched dependency in the LLM
/// context appendix; a dependency with more still shows its worst, and the
/// omission is stated so the model never mistakes the cut for completeness.
const MAX_DEP_FINDING_LINES: usize = 12;

/// Raw provider documents smaller than this stay verbatim in interpret
/// provenance. Larger registry responses (notably npm packuments) are projected
/// structurally around the requested version instead of byte-truncated, keeping
/// valid JSON and the provider-only fields an LLM may need.
const MAX_INTERPRET_RAW_REGISTRY_BYTES: usize = 4 * 1024;

/// Render the package-aware user message used by `--interpret` and
/// `--format interpret`.
///
/// Cleave's merged report contains the primary artifact, fetched artifacts, and
/// registry sidecars in one flat file list. Rendering that list directly makes
/// fetched packages look like archive members and puts their traits before the
/// appendix that explains where they came from. Build one subject block at a
/// time instead: compact provenance first, then only that package's cleave
/// context. Dependencies are omitted unless suspicious/hostile or a notable+
/// match is tied to their registry provenance (directly or through composite
/// sources).
#[allow(clippy::too_many_arguments)]
fn render_interpret_context(
    label: &str,
    sha256: &str,
    root_fetch: Option<&fletch::fetch::FetchRecord>,
    root_registry: Option<&crate::provenance::RegistryProvenance>,
    fetch_edges: &[fletch::fetch::FetchRecord],
    deps: &[DepResult],
    registries: &[crate::fetch::DependencyRegistry],
    report: &cleave::AnalysisReport,
) -> String {
    use std::fmt::Write as _;

    let by_id: std::collections::HashMap<u32, &cleave::FileAnalysis> =
        report.files.iter().map(|f| (f.id, f)).collect();
    let mut fetched_root_by_file = std::collections::HashMap::new();
    for file in &report.files {
        if let Some(root) = fetched_root_id(file, &by_id) {
            fetched_root_by_file.insert(file.id, root);
        }
    }
    let registry_ids: std::collections::HashSet<u32> =
        registries.iter().map(|r| r.file_id).collect();

    // Provider bodies can recur when several package subjects came from the
    // same registry document. Inline the first and make every later occurrence
    // a digest reference.
    let mut raw_seen = std::collections::HashSet::new();
    let mut out = String::new();

    let mut primary = report.clone();
    primary
        .files
        .retain(|f| !fetched_root_by_file.contains_key(&f.id) && !registry_ids.contains(&f.id));
    let primary_context =
        cleave::output::format_context(&primary, &cleave::output::TinyOpts::tiny());
    let _ = writeln!(out, "== PRIMARY {label} ==");
    let primary_provenance =
        primary_provenance(label, sha256, root_fetch, root_registry, &mut raw_seen);
    let _ = writeln!(
        out,
        "provenance={}",
        serde_json::to_string(&primary_provenance).unwrap_or_else(|_| "{}".to_string())
    );
    out.push_str(&primary_context);

    let mut shown_registry_ids = std::collections::HashSet::new();
    let mut visited_roots = std::collections::HashSet::new();
    let mut candidate_count = 0_usize;
    let mut shown_count = 0_usize;
    for rec in fetch_edges.iter().filter(|r| r.content_sha256.is_some()) {
        let content_sha = rec.content_sha256.as_deref().unwrap_or_default();
        let Some(root) = report
            .files
            .iter()
            .find(|f| f.sha256 == content_sha && f.rel == cleave::types::Rel::Fetched)
        else {
            continue;
        };
        if !visited_roots.insert(root.id) {
            continue;
        }
        candidate_count += 1;
        let graded = deps.iter().find(|d| d.sha256 == content_sha);
        let registry = registries.iter().find(|r| r.locator == rec.locator);
        let member_ids: std::collections::HashSet<u32> = fetched_root_by_file
            .iter()
            .filter_map(|(&file, &fetched_root)| (fetched_root == root.id).then_some(file))
            .collect();
        let severe_finding = report
            .files
            .iter()
            .filter(|f| member_ids.contains(&f.id))
            .flat_map(|f| &f.findings)
            .any(|f| f.crit >= cleave::Criticality::Suspicious);
        let severe_verdict = graded.and_then(|d| d.verdict).is_some_and(|v| {
            matches!(
                v.class,
                Classification::Suspicious | Classification::Hostile
            )
        });
        let provenance_hit = registry.is_some_and(|registry| {
            provenance_has_notable_match(report, &member_ids, registry.file_id)
        });
        if !severe_finding && !severe_verdict && !provenance_hit {
            continue;
        }
        shown_count += 1;

        let class = match graded.and_then(|d| d.verdict).map(|v| v.class) {
            Some(Classification::Hostile) => "hostile",
            Some(Classification::Suspicious) => "suspicious",
            Some(Classification::Benign) => "benign",
            None => "not-evaluated",
        };
        let subject = if rec.kind == fletch::RefKind::Dependency {
            "DEP"
        } else {
            "FETCH"
        };
        let _ = writeln!(out, "\n== {subject} {} class={class} ==", rec.locator);
        let provenance = dependency_provenance(rec, registry, report, &mut raw_seen);
        let _ = writeln!(
            out,
            "provenance={}",
            serde_json::to_string(&provenance).unwrap_or_else(|_| "{}".to_string())
        );

        let mut view = report.clone();
        view.files
            .retain(|f| member_ids.contains(&f.id) || registry.is_some_and(|r| r.file_id == f.id));
        out.push_str(&cleave::output::format_context(
            &view,
            &cleave::output::TinyOpts::tiny(),
        ));
        if let Some(registry) = registry {
            shown_registry_ids.insert(registry.file_id);
        }
    }

    // A removed or age-gated dependency may have no artifact subtree at all.
    // Its atomic provenance finding is still evidence and gets its own subject.
    for registry in registries {
        if shown_registry_ids.contains(&registry.file_id) {
            continue;
        }
        if fetch_edges
            .iter()
            .any(|edge| edge.locator == registry.locator && edge.content_sha256.is_some())
        {
            continue;
        }
        candidate_count += 1;
        let member_ids = std::collections::HashSet::new();
        if !provenance_has_notable_match(report, &member_ids, registry.file_id) {
            continue;
        }
        shown_registry_ids.insert(registry.file_id);
        shown_count += 1;
        let status = registry.artifact_skip.unwrap_or("registry-only");
        let _ = writeln!(out, "\n== DEP {} artifact={status} ==", registry.locator);
        let provenance =
            render_registry_provenance(&registry.locator, &registry.provenance, &mut raw_seen);
        let _ = writeln!(
            out,
            "provenance={}",
            serde_json::to_string(&provenance).unwrap_or_else(|_| "{}".to_string())
        );
        let mut view = report.clone();
        view.files.retain(|f| f.id == registry.file_id);
        out.push_str(&cleave::output::format_context(
            &view,
            &cleave::output::TinyOpts::tiny(),
        ));
    }

    let omitted = candidate_count.saturating_sub(shown_count);
    if omitted > 0 {
        let _ = writeln!(out, "\ndeps_omitted={omitted}");
    }
    out
}

fn fetched_root_id<'a>(
    file: &'a cleave::FileAnalysis,
    by_id: &std::collections::HashMap<u32, &'a cleave::FileAnalysis>,
) -> Option<u32> {
    let mut current = file;
    for _ in 0..=by_id.len() {
        if current.rel == cleave::types::Rel::Fetched {
            return Some(current.id);
        }
        current = *by_id.get(&current.parent_id?)?;
    }
    None
}

/// True for a notable+ atomic match on the registry node, or a notable+
/// composite on the dependency artifact whose resolved sources include it.
fn provenance_has_notable_match(
    report: &cleave::AnalysisReport,
    artifact_ids: &std::collections::HashSet<u32>,
    registry_id: u32,
) -> bool {
    report.files.iter().any(|file| {
        if file.id == registry_id
            && file
                .findings
                .iter()
                .any(|f| f.crit >= cleave::Criticality::Notable)
        {
            return true;
        }
        artifact_ids.contains(&file.id)
            && file.findings.iter().any(|finding| {
                finding.crit >= cleave::Criticality::Notable
                    && file
                        .composite_sources
                        .get(&finding.id)
                        .is_some_and(|sources| sources.iter().any(|s| s.file == registry_id))
            })
    })
}

fn primary_provenance(
    label: &str,
    sha256: &str,
    fetch: Option<&fletch::fetch::FetchRecord>,
    registry: Option<&crate::provenance::RegistryProvenance>,
    raw_seen: &mut std::collections::HashSet<String>,
) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    if let Some(fetch) = fetch {
        out.insert("fetch".to_string(), compact_fetch_record(fetch, None));
    } else {
        out.insert(
            "artifact".to_string(),
            serde_json::json!({"path": label, "sha256": sha256}),
        );
    }
    if let Some(registry) = registry {
        let locator = fetch.map_or("", |f| f.locator.as_str());
        out.insert(
            "registry".to_string(),
            render_registry_provenance(locator, registry, raw_seen),
        );
    }
    serde_json::Value::Object(out)
}

fn dependency_provenance(
    fetch: &fletch::fetch::FetchRecord,
    registry: Option<&crate::fetch::DependencyRegistry>,
    report: &cleave::AnalysisReport,
    raw_seen: &mut std::collections::HashSet<String>,
) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    out.insert(
        "fetch".to_string(),
        compact_fetch_record(fetch, Some(report)),
    );
    if let Some(registry) = registry {
        out.insert(
            "registry".to_string(),
            render_registry_provenance(&registry.locator, &registry.provenance, raw_seen),
        );
    }
    serde_json::Value::Object(out)
}

/// Fetch provenance minus response headers and declaring-file hashes. Those are
/// high-volume and either irrelevant to interpretation or already rendered as
/// source paths; URLs, redirects, timing, cache/pin state, and content identity
/// remain.
fn compact_fetch_record(
    fetch: &fletch::fetch::FetchRecord,
    report: Option<&cleave::AnalysisReport>,
) -> serde_json::Value {
    let Ok(mut value) = serde_json::to_value(fetch) else {
        return serde_json::Value::Null;
    };
    if let Some(obj) = value.as_object_mut() {
        obj.remove("headers");
        obj.remove("source_sha256");
        if let Some(source_path) = report.and_then(|report| {
            report
                .files
                .iter()
                .find(|file| file.sha256 == fetch.source_sha256)
                .map(|file| file.path.as_str())
        }) {
            obj.insert(
                "source_path".to_string(),
                serde_json::Value::String(source_path.to_string()),
            );
        }
    }
    value
}

fn render_registry_provenance(
    _locator: &str,
    provenance: &crate::provenance::RegistryProvenance,
    raw_seen: &mut std::collections::HashSet<String>,
) -> serde_json::Value {
    let registry = &provenance.record;
    let mut out = serde_json::Map::new();
    out.insert("record".to_string(), sparse_registry_record(registry));
    if let Some(raw) = provenance.raw() {
        out.insert(
            "raw".to_string(),
            compact_registry_raw(&raw, registry, raw_seen),
        );
    }
    serde_json::Value::Object(out)
}

fn sparse_registry_record(registry: &fletch::Registry) -> serde_json::Value {
    let mut value = serde_json::to_value(registry).unwrap_or(serde_json::Value::Null);
    remove_empty_json(&mut value);
    value
}

fn remove_empty_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(obj) => {
            for child in obj.values_mut() {
                remove_empty_json(child);
            }
            obj.retain(|_, value| {
                !value.is_null()
                    && !matches!(value, serde_json::Value::String(s) if s.is_empty())
                    && !matches!(value, serde_json::Value::Array(a) if a.is_empty())
                    && !matches!(value, serde_json::Value::Object(o) if o.is_empty())
            });
        }
        serde_json::Value::Array(array) => {
            for child in array.iter_mut() {
                remove_empty_json(child);
            }
        }
        _ => {}
    }
}

fn compact_registry_raw(
    raw: &serde_json::Value,
    registry: &fletch::Registry,
    raw_seen: &mut std::collections::HashSet<String>,
) -> serde_json::Value {
    if let Some(sources) = raw.as_array() {
        return serde_json::Value::Array(
            sources
                .iter()
                .map(|source| compact_registry_source(source, registry, raw_seen))
                .collect(),
        );
    }
    compact_registry_source(raw, registry, raw_seen)
}

fn compact_registry_source(
    source: &serde_json::Value,
    registry: &fletch::Registry,
    raw_seen: &mut std::collections::HashSet<String>,
) -> serde_json::Value {
    use sha2::{Digest as _, Sha256};

    let body = source.get("body").unwrap_or(source);
    let body_bytes = serde_json::to_vec(body).unwrap_or_default();
    let digest = format!("{:x}", Sha256::digest(&body_bytes));
    let duplicate = !raw_seen.insert(digest.clone());
    let mut out = serde_json::Map::new();
    for key in ["url", "status", "content_type"] {
        if let Some(value) = source.get(key) {
            out.insert(key.to_string(), value.clone());
        }
    }
    out.insert("sha256".to_string(), serde_json::json!(digest));
    if duplicate {
        out.insert("deduplicated".to_string(), serde_json::Value::Bool(true));
        return serde_json::Value::Object(out);
    }

    if source.get("body_b64").is_some() {
        out.insert(
            "body_b64".to_string(),
            serde_json::json!("<preserved in provenance>"),
        );
        out.insert(
            "original_bytes".to_string(),
            serde_json::json!(body_bytes.len()),
        );
    } else if body_bytes.len() <= MAX_INTERPRET_RAW_REGISTRY_BYTES {
        out.insert("body".to_string(), body.clone());
    } else {
        out.insert(
            "body_projection".to_string(),
            focus_registry_json(body, &registry.version, 0),
        );
        out.insert(
            "original_bytes".to_string(),
            serde_json::json!(body_bytes.len()),
        );
    }
    serde_json::Value::Object(out)
}

/// Preserve provider-specific data while bounding huge maps/arrays. Version
/// maps retain the requested release; ordinary objects keep their first useful
/// fields, and long text is visibly abbreviated.
fn focus_registry_json(
    value: &serde_json::Value,
    version: &str,
    depth: usize,
) -> serde_json::Value {
    if depth >= 5 {
        return serde_json::json!("…");
    }
    match value {
        serde_json::Value::Object(obj) => {
            let mut out = serde_json::Map::new();
            let version_map = !version.is_empty() && obj.contains_key(version);
            // Stable semantic priority first, then a few provider-specific keys
            // so projection does not collapse back to only normalized fields.
            for key in [
                version,
                "name",
                "id",
                "version",
                "info",
                "crate",
                "package",
                "dist-tags",
                "time",
                "versions",
                "releases",
                "urls",
                "scripts",
                "dist",
                "_npmUser",
                "maintainers",
                "author",
                "description",
                "repository",
                "homepage",
                "deprecated",
                "unpublished",
                "modified",
                "created",
            ] {
                if out.len() >= 12 {
                    break;
                }
                if !key.is_empty()
                    && let Some(child) = obj.get(key)
                {
                    out.insert(
                        key.to_string(),
                        focus_registry_json(child, version, depth + 1),
                    );
                }
            }
            if !version_map {
                for (key, child) in obj {
                    if out.len() >= 12 {
                        break;
                    }
                    if !out.contains_key(key) {
                        out.insert(key.clone(), focus_registry_json(child, version, depth + 1));
                    }
                }
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(array) => {
            let mut selected: Vec<&serde_json::Value> = array
                .iter()
                .filter(|item| {
                    item.as_object().is_some_and(|obj| {
                        ["version", "num", "name"]
                            .iter()
                            .any(|key| obj.get(*key).and_then(|v| v.as_str()) == Some(version))
                    })
                })
                .collect();
            if selected.is_empty() {
                selected.extend(array.iter().take(4));
            }
            serde_json::Value::Array(
                selected
                    .into_iter()
                    .take(4)
                    .map(|child| focus_registry_json(child, version, depth + 1))
                    .collect(),
            )
        }
        serde_json::Value::String(text) if text.len() > 384 => {
            let mut end = 384;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            serde_json::Value::String(format!("{}…", text.get(..end).unwrap_or(text)))
        }
        _ => value.clone(),
    }
}

/// Human-facing counterpart to [`render_interpret_context`]. It uses the same
/// subject selection and keeps provenance before traits, but prints only
/// processed acquisition/registry fields. Raw provider JSON remains exclusive
/// to the compact interpret payload.
fn render_terminal_fetch_context(
    fetch_edges: &[fletch::fetch::FetchRecord],
    deps: &[DepResult],
    registries: &[crate::fetch::DependencyRegistry],
    report: &cleave::AnalysisReport,
    opts: &cleave::output::TinyOpts,
) -> Option<String> {
    use std::fmt::Write as _;

    let by_id: std::collections::HashMap<u32, &cleave::FileAnalysis> =
        report.files.iter().map(|f| (f.id, f)).collect();
    let fetched_root_by_file: std::collections::HashMap<u32, u32> = report
        .files
        .iter()
        .filter_map(|file| fetched_root_id(file, &by_id).map(|root| (file.id, root)))
        .collect();
    let mut visited_roots = std::collections::HashSet::new();
    let mut shown_registry_ids = std::collections::HashSet::new();
    let mut candidate_count = 0_usize;
    let mut shown_count = 0_usize;
    let mut out = String::new();

    for rec in fetch_edges.iter().filter(|r| r.content_sha256.is_some()) {
        let content_sha = rec.content_sha256.as_deref().unwrap_or_default();
        let Some(root) = report
            .files
            .iter()
            .find(|f| f.sha256 == content_sha && f.rel == cleave::types::Rel::Fetched)
        else {
            continue;
        };
        if !visited_roots.insert(root.id) {
            continue;
        }
        candidate_count += 1;
        let graded = deps.iter().find(|d| d.sha256 == content_sha);
        let registry = registries.iter().find(|r| r.locator == rec.locator);
        let member_ids: std::collections::HashSet<u32> = fetched_root_by_file
            .iter()
            .filter_map(|(&file, &fetched_root)| (fetched_root == root.id).then_some(file))
            .collect();
        let severe_finding = report
            .files
            .iter()
            .filter(|file| member_ids.contains(&file.id))
            .flat_map(|file| &file.findings)
            .any(|finding| finding.crit >= cleave::Criticality::Suspicious);
        let severe_verdict = graded.and_then(|d| d.verdict).is_some_and(|verdict| {
            matches!(
                verdict.class,
                Classification::Suspicious | Classification::Hostile
            )
        });
        let provenance_hit = registry.is_some_and(|registry| {
            provenance_has_notable_match(report, &member_ids, registry.file_id)
        });
        if !severe_finding && !severe_verdict && !provenance_hit {
            continue;
        }
        shown_count += 1;

        let subject = if rec.kind == fletch::RefKind::Dependency {
            "dependency"
        } else {
            "URL fetch"
        };
        match graded.and_then(|d| d.verdict) {
            Some(verdict) => {
                let class = verdict.class.to_string().to_ascii_lowercase();
                let _ = writeln!(
                    out,
                    "\n  ↳ {:.0}% {} · {class} {subject}",
                    verdict.probability * 100.0,
                    rec.locator
                );
            }
            None => {
                let _ = writeln!(out, "\n  ↳ {} · {subject} · not evaluated", rec.locator);
            }
        }
        write_terminal_fetch_provenance(&mut out, rec, report);
        let mut signals = Vec::new();
        if rec.pin_verified == Some(false) {
            signals.push("checksum mismatch");
        }
        if let Some(registry) = registry {
            write_terminal_registry_provenance(&mut out, registry, &mut signals);
            shown_registry_ids.insert(registry.file_id);
        } else if !signals.is_empty() {
            let _ = writeln!(out, "      signals    {}", signals.join(" · "));
        }

        let mut view = report.clone();
        view.files.retain(|file| {
            member_ids.contains(&file.id) || registry.is_some_and(|r| r.file_id == file.id)
        });
        out.push_str(&cleave::output::format_context(&view, opts));
    }

    for registry in registries {
        if shown_registry_ids.contains(&registry.file_id)
            || fetch_edges
                .iter()
                .any(|edge| edge.locator == registry.locator && edge.content_sha256.is_some())
        {
            continue;
        }
        candidate_count += 1;
        if !provenance_has_notable_match(
            report,
            &std::collections::HashSet::new(),
            registry.file_id,
        ) {
            continue;
        }
        shown_count += 1;
        let status = registry.artifact_skip.unwrap_or("registry only");
        let _ = writeln!(out, "\n  ↳ {} · {status}", registry.locator);
        let mut signals = Vec::new();
        write_terminal_registry_provenance(&mut out, registry, &mut signals);
        let mut view = report.clone();
        view.files.retain(|file| file.id == registry.file_id);
        out.push_str(&cleave::output::format_context(&view, opts));
    }

    let omitted = candidate_count.saturating_sub(shown_count);
    if omitted > 0 {
        let _ = writeln!(out, "\n  {omitted} unremarkable fetched artifacts omitted");
    }
    (!out.is_empty()).then_some(out)
}

fn write_terminal_fetch_provenance(
    out: &mut String,
    rec: &fletch::fetch::FetchRecord,
    report: &cleave::AnalysisReport,
) {
    use std::fmt::Write as _;

    let source = report
        .files
        .iter()
        .find(|file| file.sha256 == rec.source_sha256)
        .map_or("<unknown>", |file| file.path.as_str());
    let _ = write!(out, "      from       {source}");
    if let Some(offset) = rec.source_offset {
        let _ = write!(out, " @ byte {offset}");
    }
    out.push('\n');
    if !rec.resolved_url.is_empty() && rec.resolved_url != rec.locator {
        let _ = writeln!(out, "      resolved   {}", rec.resolved_url);
    }
    if let Some(final_url) = rec
        .final_url
        .as_deref()
        .filter(|url| *url != rec.resolved_url && *url != rec.locator)
    {
        let _ = writeln!(out, "      final      {final_url}");
    }
    if let Some(content_sha256) = &rec.content_sha256 {
        let _ = writeln!(out, "      sha256     {content_sha256}");
    }
}

fn write_terminal_registry_provenance(
    out: &mut String,
    registry: &crate::fetch::DependencyRegistry,
    signals: &mut Vec<&'static str>,
) {
    use std::fmt::Write as _;

    let record = &registry.provenance.record;
    let mut summary = Vec::new();
    if let Some(age) = record.age_days {
        summary.push(format!("{age}d old"));
    }
    if let Some(downloads) = record.downloads_recent.or(record.downloads_total) {
        summary.push(format!("{downloads} downloads"));
    }
    if let Some(maintainers) = record.maintainers {
        let noun = if maintainers == 1 {
            "maintainer"
        } else {
            "maintainers"
        };
        summary.push(format!("{maintainers} {noun}"));
    }
    if !summary.is_empty() {
        let _ = writeln!(out, "      registry   {}", summary.join(" · "));
    }
    if record.version_removed == Some(true) {
        signals.push("version removed");
    }
    if record.security_hold == Some(true) {
        signals.push("security hold");
    }
    if record.publisher_in_maintainers == Some(false) {
        signals.push("publisher not in maintainers");
    }
    if record.publisher_verified == Some(false) {
        signals.push("publisher unverified");
    }
    if record.has_install_script == Some(true) {
        signals.push("install script");
    }
    if let Some(deprecated) = record.deprecated.as_deref() {
        let _ = writeln!(out, "      deprecated {deprecated}");
    }
    if !signals.is_empty() {
        let _ = writeln!(out, "      signals    {}", signals.join(" · "));
    }
    if let Some(repository) = record.repository.as_deref() {
        let _ = writeln!(out, "      upstream   {repository}");
    }
    for url in registry.provenance.source_urls() {
        let _ = writeln!(out, "      metadata   {url}");
    }
}

/// Render the fetched-dependencies appendix for the LLM interpret context.
///
/// Fetched payloads are grafted into the report under synthetic paths (a UUID
/// or purl), so in the main render they are indistinguishable from the
/// sample's own archive members — and the `fetch/dependency-verdict` trait
/// that elevates the sample is injected into the compact JSON *after* the
/// render, so the context never explains a dependency-driven verdict. This
/// appendix is that explanation, kept clearly separate from the archive-member
/// view: one block per fetched reference naming its locator (URL/PURL), how
/// the sample referenced it (binding kind + declaring file + byte offset),
/// the model's classification of the fetched bytes, and the suspicious+
/// findings on the dependency's own files (in the render's `# SEV` annotation
/// grammar, so the interpret gates parse them like any other finding).
/// `None` when nothing was fetched.
fn render_dependency_context(
    fetch_edges: &[fletch::fetch::FetchRecord],
    deps: &[DepResult],
    report: &cleave::AnalysisReport,
) -> Option<String> {
    use std::fmt::Write as _;
    // Only edges that landed bytes have a grafted node to describe.
    let landed: Vec<_> = fetch_edges
        .iter()
        .filter(|r| r.content_sha256.is_some())
        .collect();
    if landed.is_empty() {
        return None;
    }
    let mut out = String::new();
    out.push_str(
        "\n== FETCHED DEPENDENCIES ==\n\
         The scan followed references declared by this sample and retrieved the content below.\n\
         Each payload was analyzed and appears above under its locator — these files are\n\
         EXTERNAL retrieved content, not members of the sample's own archive. A hostile or\n\
         suspicious dependency elevates the sample's verdict (fetch/dependency-verdict).\n",
    );
    for rec in landed {
        let content_sha = rec.content_sha256.as_deref().unwrap_or_default();
        let root = report.files.iter().find(|f| f.sha256 == content_sha);
        // The verdict scan computed for this dependency, from the same graded
        // results it uploads. Reading it from the parent's embedded pass instead
        // made the render disagree with the record: that pass is bounded by the
        // parent's budget and ordered by the merged file list, so a dependency it
        // never reached printed no classification at all — the appendix read as
        // if the dependency were unremarkable.
        let graded = deps.iter().find(|d| d.sha256 == content_sha);
        let _ = writeln!(out, "\ndependency: {}", rec.locator);
        if !rec.resolved_url.is_empty() && rec.resolved_url != rec.locator {
            let _ = writeln!(out, "  resolved url: {}", rec.resolved_url);
        }
        let source = report
            .files
            .iter()
            .find(|f| f.sha256 == rec.source_sha256)
            .map_or("<unknown file>", |f| f.path.as_str());
        let _ = write!(
            out,
            "  referenced: {} in {source}",
            ref_kind_phrase(&rec.kind)
        );
        if let Some(off) = rec.source_offset {
            let _ = write!(out, " @ byte {off}");
        }
        out.push('\n');
        let _ = writeln!(out, "  content sha256: {content_sha}");
        if let Some(root) = root {
            let _ = writeln!(out, "  analyzed above as: {}", root.path);
        }
        match graded.and_then(|d| d.verdict) {
            Some(v) => {
                let class = match v.class {
                    Classification::Hostile => "hostile",
                    Classification::Suspicious => "suspicious",
                    Classification::Benign => "benign",
                };
                let _ = writeln!(out, "  classification: {class} (p={:.2})", v.probability);
            }
            // Say so rather than printing nothing: an ungraded dependency and an
            // unremarkable one are different, and only one of them is a coverage
            // gap worth chasing.
            None => out.push_str("  classification: not evaluated\n"),
        }
        let Some(root) = root else { continue };
        // The dependency's own elevated findings: the root node plus every
        // member below it, deduped by id, worst first.
        let member_prefix = format!("{}!!", root.path);
        let mut elevated: Vec<(&cleave::FileAnalysis, &cleave::Finding)> = report
            .files
            .iter()
            .filter(|f| f.path == root.path || f.path.starts_with(&member_prefix))
            .flat_map(|f| f.findings.iter().map(move |fd| (f, fd)))
            .filter(|(_, fd)| fd.crit >= cleave::Criticality::Suspicious)
            .collect();
        elevated.sort_by_key(|(_, fd)| std::cmp::Reverse(fd.crit));
        let mut seen = std::collections::HashSet::new();
        elevated.retain(|(_, fd)| seen.insert(fd.id.as_str()));
        if elevated.is_empty() {
            continue;
        }
        out.push_str("  elevated findings on this dependency's files:\n");
        let total = elevated.len();
        for (f, fd) in elevated.into_iter().take(MAX_DEP_FINDING_LINES) {
            let sev = if fd.crit >= cleave::Criticality::Hostile {
                'H'
            } else {
                'S'
            };
            let _ = write!(out, "  # {sev} {}", fd.id);
            if !fd.desc.is_empty() {
                let _ = write!(out, " — {}", fd.desc);
            }
            let _ = writeln!(out, " [{}]", f.path);
        }
        if total > MAX_DEP_FINDING_LINES {
            let _ = writeln!(
                out,
                "  … {} more elevated findings omitted",
                total - MAX_DEP_FINDING_LINES
            );
        }
    }
    Some(out)
}

/// How a fetch edge's declaring reference binds the sample to the dependency,
/// as prose for the LLM context.
fn ref_kind_phrase(kind: &fletch::RefKind) -> &'static str {
    match kind {
        fletch::RefKind::Dependency => "declared as a dependency",
        fletch::RefKind::Command => "named by an install command",
        fletch::RefKind::UrlFetch => "fetched from a URL",
        fletch::RefKind::Repository => "named as the source repository",
        fletch::RefKind::Local => "a local reference",
        _ => "referenced",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod dependency_grading_tests {
    use super::*;

    /// Grading needs a real model to score against. Point SCAN_MODELS_DIR at a
    /// bundle to run these; without one there is nothing to exercise, so they
    /// skip rather than fail. Mirrors the SCAN_ONNX_BUNDLE convention in
    /// tests/backend_dispatch.rs.
    fn model_bundle() -> Option<std::path::PathBuf> {
        let p = std::env::var("SCAN_MODELS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("/home/t/azoth/filetypes/npm"));
        (p.join("model.onnx").is_file() && p.join("feature_spec.json").is_file()).then_some(p)
    }

    /// One dependency node, plus `members` javascript members beneath it.
    fn dep_report(members: usize, member_traits: &str) -> String {
        let files: Vec<serde_json::Value> = std::iter::once(serde_json::json!({
            "id": 0, "sha": "d".repeat(64), "type": "npm", "dp": 0, "path": "evil-1.0.0.tgz"
        }))
        .chain((0..members).map(|i| {
            let mut f = serde_json::json!({
                "id": i + 1, "sha": format!("{:064x}", i), "type": "javascript",
                "dp": 1, "path": format!("evil-1.0.0.tgz!!lib/{i}.js"),
            });
            if !member_traits.is_empty() {
                f["traits"] = serde_json::from_str(member_traits).unwrap();
            }
            f
        }))
        .collect();
        serde_json::json!({"v": "8", "files": files}).to_string()
    }

    /// The regression this whole path exists for: a dependency is graded from its
    /// own report, so it comes back with a verdict. It used to be graded by
    /// attributing the parent's embedded pass back to it, and one that fell
    /// outside the parent's shared budget got no evaluation at all — which was
    /// then uploaded as a confident benign.
    #[test]
    fn grades_a_dependency_on_its_own_report() {
        let Some(dir) = model_bundle() else {
            eprintln!("skipping: no model bundle (set SCAN_MODELS_DIR)");
            return;
        };
        let model = Model::load(&dir, None, None).expect("load model bundle");
        let ctx = ExtractContext::new(model.spec());

        // Far more members than one report's budget: previously the tail of a
        // merged report went ungraded, now the budget is this report's own.
        let raw = dep_report(EMBEDDED_FILE_LIMIT * 2, "");
        let verdict = classify_dependency(&raw, "pkg:npm/evil@1.0.0", &ctx, &model, None);
        assert!(
            verdict.is_some(),
            "a well-formed dependency report must produce a verdict, never silence",
        );
    }

    /// A report that will not parse yields no verdict. The caller then uploads the
    /// bytes and posts nothing, leaving hopper to grade it — rather than inventing
    /// a benign that would be indistinguishable from a real evaluation.
    #[test]
    fn declines_to_grade_an_unparseable_report() {
        let Some(dir) = model_bundle() else {
            eprintln!("skipping: no model bundle (set SCAN_MODELS_DIR)");
            return;
        };
        let model = Model::load(&dir, None, None).expect("load model bundle");
        let ctx = ExtractContext::new(model.spec());
        assert!(
            classify_dependency("{not json", "pkg:npm/x@1", &ctx, &model, None).is_none(),
            "an unparseable report must yield no verdict",
        );
    }

    /// Members elevate their container: a dependency whose members carry confident
    /// hostile findings must not rank below the same dependency with clean ones.
    /// Relative, not absolute — the thresholds are the model's business.
    #[test]
    fn severe_members_do_not_rank_below_clean_ones() {
        let Some(dir) = model_bundle() else {
            eprintln!("skipping: no model bundle (set SCAN_MODELS_DIR)");
            return;
        };
        let model = Model::load(&dir, None, None).expect("load model bundle");
        let ctx = ExtractContext::new(model.spec());

        let (clean, _) = classify_dependency(&dep_report(3, ""), "pkg:npm/a@1", &ctx, &model, None)
            .expect("clean report grades");
        let (severe, _) = classify_dependency(
            &dep_report(3, r#"[{"crit":5,"conf":1.0},{"crit":5,"conf":0.98}]"#),
            "pkg:npm/b@1",
            &ctx,
            &model,
            None,
        )
        .expect("severe report grades");

        assert!(
            !decision_outranks(&clean, &severe),
            "clean members outranked hostile ones: clean={:?} severe={:?}",
            clean.class,
            severe.class,
        );
    }

    /// The dependency backref is a conclusion about *another* artifact, pinned
    /// onto the file that named it. It must never reach featurization: a
    /// synthetic crit-5 trait describing someone else's bytes, fed as a feature
    /// of these ones, teaches the model that declaring dependencies is itself
    /// malicious.
    ///
    /// Nothing in the types enforces that — it holds only because the injection
    /// sits after the feature pass in classify_report. This pins the ordering so
    /// a future edit that moves either one fails here instead of silently
    /// contaminating training data. Delete this test only by making the
    /// distinction explicit in the trait itself.
    #[test]
    fn dependency_backref_is_injected_after_featurization() {
        let src = include_str!("engine.rs");
        let featurize = src
            .find("let parsed = crate::features::ParsedReport::from_report(&root_json, needs);")
            .expect("root featurization call");
        let inject = src
            .find("inject_dependency_backref(&mut report_json, &report, backref);")
            .expect("backref injection call");
        assert!(
            featurize < inject,
            "dependency backrefs are injected before featurization — the model would \
             train on synthetic traits describing other artifacts",
        );
    }

    /// The acceptance test for the whole dependency path: what hopper stores for
    /// a fetched dependency must be the same *shape* as what it stores when the
    /// same bytes are scanned directly with `scan purl`. Anything a first-hand
    /// scan fills and the dependency path leaves empty is a field hopper, the
    /// bloom pool, and prism silently lose for every dependency.
    ///
    /// Shape, not values: the two paths legitimately differ in what they measure
    /// (a dependency borrows the parent run's model version and timestamp, and
    /// carries no LLM interpretation — the interpret pass runs on the root only).
    /// Those are asserted as *known* differences, so adding a new one has to be
    /// deliberate rather than accidental.
    #[test]
    fn dependency_envelope_matches_a_direct_scan() {
        let Some(dir) = model_bundle() else {
            eprintln!("skipping: no model bundle (set SCAN_MODELS_DIR)");
            return;
        };
        let model = Model::load(&dir, None, None).expect("load model bundle");
        let ctx = ExtractContext::new(model.spec());

        let raw = dep_report(3, r#"[{"crit":5,"conf":1.0}]"#);
        let (verdict, members) =
            classify_dependency(&raw, "pkg:npm/evil@1.0.0", &ctx, &model, None)
                .expect("dependency grades");

        let dep = DepResult {
            sha256: "d".repeat(64),
            locator: "pkg:npm/evil@1.0.0".to_string(),
            url: "https://reg.test/evil-1.0.0.tgz".to_string(),
            size: 1234,
            provenance: None,
            verdict: Some(verdict),
            members,
            raw: raw.clone(),
        };
        let dep_env = dep_envelope(&dep, "model-9", "2026-06-28T00:00:00Z")
            .expect("a graded dependency yields an envelope");

        // The same bytes as a first-hand scan would produce them.
        let direct = ScanResult {
            v: "7",
            classification: verdict.class,
            probability: verdict.probability,
            threshold: verdict.threshold,
            level: verdict.level,
            version: "model-9".to_string(),
            analyzed_at: "2026-06-28T00:00:00Z".to_string(),
            cleave: Some(serde_json::from_str(&raw).expect("report parses")),
            embedded_files: dep.members.clone(),
            ..crate::engine::envelope_tests::base_result()
        };
        let direct_env = direct.to_envelope();

        assert_eq!(
            dep_env.ml.level, direct_env.ml.level,
            "verdict marker must match a direct scan",
        );
        assert_eq!(
            dep_env.ml.probability, direct_env.ml.probability,
            "probability must match a direct scan",
        );
        assert_eq!(
            dep_env.ml.conf, direct_env.ml.conf,
            "confidence must match a direct scan",
        );
        assert_eq!(
            dep_env.raw, direct_env.raw,
            "the stored report must be the dependency's own, unmodified",
        );

        // The regression this test exists for: ml.files carried no verdicts for a
        // dependency's members, because the envelope was built with an empty eval
        // table. hopper mirrors these into each member's own sample row, so every
        // member of every dependency was stored ungraded.
        assert_eq!(
            dep_env.ml.files, direct_env.ml.files,
            "per-member verdicts must match a direct scan",
        );
        assert!(
            dep_env.ml.files.iter().any(|f| f.get("prob").is_some()),
            "at least one member must carry a verdict: {:?}",
            dep_env.ml.files,
        );

        // Known, deliberate differences. A dependency borrows the parent run's
        // identity because it was graded by that run, and the interpret pass runs
        // on the root only.
        assert!(
            dep_env.llm.is_none(),
            "dependencies carry no interpretation"
        );
        assert_eq!(dep_env.ml.version, direct_env.ml.version);
        assert_eq!(dep_env.ml.analyzed_at, direct_env.ml.analyzed_at);
    }

    /// A dependency's own budget is per report, so a report far larger than the
    /// limit still contributes exactly the limit — rather than whatever was left
    /// after the parent's members had taken theirs.
    #[test]
    fn embedded_budget_is_per_report() {
        let members: Vec<serde_json::Value> = (0..EMBEDDED_FILE_LIMIT * 3)
            .map(|i| serde_json::json!({"id": i + 1, "sha": "m".repeat(64), "type": "javascript", "dp": 1}))
            .collect();
        let report = serde_json::json!({
            "v": "8",
            "files": std::iter::once(serde_json::json!({"id": 0, "sha": "d".repeat(64), "type": "npm", "dp": 0}))
                .chain(members)
                .collect::<Vec<_>>(),
        });
        let entries: Vec<&serde_json::Value> = json_alias_array(&report, &["files", "fs"])
            .into_iter()
            .flatten()
            .filter(|f| {
                json_alias(f, &["depth", "dp"])
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    > 0
            })
            .take(EMBEDDED_FILE_LIMIT)
            .collect();
        assert_eq!(
            entries.len(),
            EMBEDDED_FILE_LIMIT,
            "a dependency grades its own members up to its own limit",
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod dep_backref_tests {
    use super::*;

    /// The injector only reads `report.files` to size the reference span (and
    /// falls back to a 1-byte span when the declaring file isn't there), so an
    /// empty report exercises everything but the span length.
    fn empty_report() -> cleave::AnalysisReport {
        serde_json::from_value(serde_json::json!({"version": "3"})).unwrap()
    }

    fn backref(class: Classification) -> DepBackref {
        DepBackref {
            source_sha: "s".repeat(64),
            source_offset: Some(42),
            locator: "pkg:npm/zaboodle@1.49".to_string(),
            dep_sha: "d".repeat(64),
            dep_type: "javascript".to_string(),
            class,
        }
    }

    /// Compact report: a depth-0 archive containing the declaring manifest and
    /// an unrelated sibling.
    fn compact_fixture() -> serde_json::Value {
        serde_json::json!({"files": [
            {"sha": "r".repeat(64), "path": "pkg.tgz", "traits": [{"id": "existing/trait", "crit": 1}]},
            {"sha": "s".repeat(64), "path": "pkg.tgz!!package.json"},
            {"sha": "o".repeat(64), "path": "pkg.tgz!!README.md"},
        ]})
    }

    #[test]
    fn declarer_gets_span_and_structured_dep() {
        let mut report_json = compact_fixture();
        inject_dependency_backref(
            &mut report_json,
            &empty_report(),
            &backref(Classification::Hostile),
        );

        let t = &report_json["files"][1]["traits"][0];
        assert_eq!(t["id"], "fetch/dependency-verdict");
        assert_eq!(t["crit"], 5, "hostile dependency pins at crit 5");
        assert_eq!(
            t["desc"],
            format!(
                "Malicious dependency: pkg:npm/zaboodle@1.49 | {}",
                "d".repeat(64)
            ),
            "desc stays prose for the traits tab and LLM context",
        );
        assert_eq!(t["dep"]["locator"], "pkg:npm/zaboodle@1.49");
        assert_eq!(t["dep"]["sha"], "d".repeat(64));
        assert_eq!(t["dep"]["type"], "javascript");
        assert_eq!(
            t["spans"][0][0], 42,
            "declaring file cites the reference byte"
        );
    }

    #[test]
    fn ancestor_carries_dep_without_span_and_siblings_stay_clean() {
        let mut report_json = compact_fixture();
        inject_dependency_backref(
            &mut report_json,
            &empty_report(),
            &backref(Classification::Hostile),
        );

        let root_traits = report_json["files"][0]["traits"].as_array().unwrap();
        assert_eq!(root_traits.len(), 2, "rolled up alongside existing traits");
        let rt = &root_traits[1];
        assert_eq!(rt["id"], "fetch/dependency-verdict");
        assert_eq!(
            rt["dep"]["sha"],
            "d".repeat(64),
            "dep identity rolls up intact"
        );
        assert!(
            rt.get("spans").is_none(),
            "a rolled-up ancestor carries no cross-file span",
        );
        assert!(
            report_json["files"][2].get("traits").is_none(),
            "unrelated sibling is untouched",
        );
    }

    #[test]
    fn suspicious_dependency_pins_at_crit_4() {
        let mut report_json = compact_fixture();
        inject_dependency_backref(
            &mut report_json,
            &empty_report(),
            &backref(Classification::Suspicious),
        );

        let t = &report_json["files"][1]["traits"][0];
        assert_eq!(t["crit"], 4);
        assert_eq!(
            t["desc"],
            format!(
                "Suspicious dependency: pkg:npm/zaboodle@1.49 | {}",
                "d".repeat(64)
            ),
        );
        assert_eq!(
            t["dep"]["type"], "javascript",
            "dep rides on suspicious too"
        );
    }

    #[test]
    fn url_locator_flows_through_verbatim() {
        let mut report_json = compact_fixture();
        let mut b = backref(Classification::Hostile);
        b.locator = "http://x.y.z/x.exe".to_string();
        b.dep_type = "pe".to_string();
        inject_dependency_backref(&mut report_json, &empty_report(), &b);

        let t = &report_json["files"][1]["traits"][0];
        assert_eq!(t["dep"]["locator"], "http://x.y.z/x.exe");
        assert_eq!(t["dep"]["type"], "pe");
    }

    /// A report shaped like a real fetched-dependency scan: the sample's
    /// manifest declared a reference, and the fetched payload was grafted
    /// under a synthetic path with a member of its own.
    fn dep_render_fixture() -> (
        Vec<fletch::fetch::FetchRecord>,
        Vec<DepResult>,
        cleave::AnalysisReport,
    ) {
        let mk_file =
            |sha: &str, path: &str, findings: Vec<cleave::Finding>| cleave::FileAnalysis {
                sha256: sha.to_string(),
                path: path.to_string(),
                findings,
                ..cleave::FileAnalysis::default()
            };
        let finding = |id: &str, desc: &str, crit: cleave::Criticality| cleave::Finding {
            id: id.to_string(),
            desc: desc.to_string(),
            crit,
            ..cleave::Finding::default()
        };
        let mut report = empty_report();
        report.files = vec![
            mk_file("r".repeat(64).as_str(), "pkg.src.tar.gz", vec![]),
            mk_file("s".repeat(64).as_str(), "pkg.src.tar.gz!!.SRCINFO", vec![]),
            mk_file(
                "d".repeat(64).as_str(),
                "d420381f-dep",
                vec![finding(
                    "objectives/persistence/x::implant",
                    "drops an implant",
                    cleave::Criticality::Hostile,
                )],
            ),
            mk_file(
                "m".repeat(64).as_str(),
                "d420381f-dep!!configure",
                vec![
                    finding(
                        "micro-behaviors/net/y::beacon",
                        "beacons out",
                        cleave::Criticality::Suspicious,
                    ),
                    finding(
                        "meta/z::noise",
                        "notable only",
                        cleave::Criticality::Notable,
                    ),
                ],
            ),
        ];
        let edge: fletch::fetch::FetchRecord = serde_json::from_value(serde_json::json!({
            "source_sha256": "s".repeat(64),
            "source_offset": 132,
            "kind": "dependency",
            "locator": "https://example.com/dep-1.0.tar.gz",
            "content_sha256": "d".repeat(64),
            "fetched_at": 1,
            "cached": false,
            "outcome": "ok",
        }))
        .unwrap();
        // The graded dependency, as classify_dependency produced it — the same
        // results the upload path posts, so the render and the record cannot
        // disagree.
        let deps = vec![DepResult {
            sha256: "d".repeat(64),
            locator: "https://example.com/dep-1.0.tar.gz".to_string(),
            url: "https://example.com/dep-1.0.tar.gz".to_string(),
            size: 0,
            provenance: None,
            verdict: Some(Decision {
                class: Classification::Hostile,
                probability: 0.97,
                threshold: 0.5,
                level: None,
            }),
            members: MemberEvals::new(),
            raw: "{}".to_string(),
        }];
        (vec![edge], deps, report)
    }

    /// The appendix names the locator, the declaring file + reference kind +
    /// byte offset, the fetched bytes' classification, and the dependency's
    /// suspicious+ findings in the `# SEV` annotation grammar — while notable
    /// findings and the sample's own files stay out of it.
    #[test]
    fn dependency_context_names_locator_reference_and_elevated_findings() {
        let (edges, deps, report) = dep_render_fixture();
        let ctx = render_dependency_context(&edges, &deps, &report).unwrap();
        for want in [
            "== FETCHED DEPENDENCIES ==",
            "dependency: https://example.com/dep-1.0.tar.gz",
            "referenced: declared as a dependency in pkg.src.tar.gz!!.SRCINFO @ byte 132",
            "analyzed above as: d420381f-dep",
            "classification: hostile (p=0.97)",
            "# H objectives/persistence/x::implant — drops an implant [d420381f-dep]",
            "# S micro-behaviors/net/y::beacon — beacons out [d420381f-dep!!configure]",
        ] {
            assert!(ctx.contains(want), "missing {want:?} in:\n{ctx}");
        }
        assert!(
            !ctx.contains("meta/z::noise"),
            "notable findings must stay out of the appendix:\n{ctx}"
        );
        // The interpret gate must see the appendix's elevated markers.
        assert!(
            crate::interpret::sanitize_context(&ctx)
                .lines()
                .any(|l| l.trim_start().starts_with("# H "))
        );
    }

    /// A dependency scan could not grade says so, rather than printing nothing.
    /// The appendix used to read its verdict from the parent's embedded pass,
    /// which is bounded by the parent's budget — so a dependency that pass never
    /// reached silently lost its classification line and read as unremarkable.
    /// Now the render and the uploaded record share one source, so the only way
    /// to omit a verdict is for there genuinely not to be one.
    #[test]
    fn dependency_context_says_when_a_dependency_was_not_graded() {
        let (edges, mut deps, report) = dep_render_fixture();
        deps[0].verdict = None;
        let ctx = render_dependency_context(&edges, &deps, &report).unwrap();
        assert!(
            ctx.contains("classification: not evaluated"),
            "an ungraded dependency must be named as such, got: {ctx:?}",
        );
    }

    /// No landed fetches → no appendix; an edge whose fetch failed (no
    /// content sha) contributes nothing.
    #[test]
    fn dependency_context_absent_without_landed_fetches() {
        let (mut edges, deps, report) = dep_render_fixture();
        assert!(render_dependency_context(&[], &deps, &report).is_none());
        edges[0].content_sha256 = None;
        assert!(render_dependency_context(&edges, &deps, &report).is_none());
    }

    #[test]
    fn interpret_context_puts_provenance_before_each_packages_traits() {
        let finding = |id: &str, desc: &str, crit: cleave::Criticality| cleave::Finding {
            id: id.to_string(),
            desc: desc.to_string(),
            crit,
            ..cleave::Finding::default()
        };
        let mut root = cleave::FileAnalysis {
            id: 0,
            path: "root.tgz".to_string(),
            sha256: "r".repeat(64),
            ..cleave::FileAnalysis::default()
        };
        root.findings.push(finding(
            "root/notable",
            "primary package finding",
            cleave::Criticality::Notable,
        ));
        let mut dep = cleave::FileAnalysis {
            id: 1,
            parent_id: Some(0),
            depth: 1,
            rel: cleave::types::Rel::Fetched,
            path: "dep.tgz".to_string(),
            sha256: "d".repeat(64),
            ..cleave::FileAnalysis::default()
        };
        dep.findings.push(finding(
            "dep/hostile",
            "dependency package finding",
            cleave::Criticality::Hostile,
        ));
        let mut registry_file = cleave::FileAnalysis {
            id: 2,
            parent_id: Some(0),
            depth: 1,
            rel: cleave::types::Rel::Registry,
            role: cleave::types::Role::Sidecar,
            path: "dep@1.registry.json".to_string(),
            sha256: "g".repeat(64),
            ..cleave::FileAnalysis::default()
        };
        registry_file.findings.push(finding(
            "registry/new",
            "new package",
            cleave::Criticality::Notable,
        ));
        let mut report = empty_report();
        report.files = vec![root, dep, registry_file];
        let edge: fletch::fetch::FetchRecord = serde_json::from_value(serde_json::json!({
            "source_sha256": "r".repeat(64),
            "kind": "dependency",
            "locator": "pkg:test/dep@1",
            "resolved_url": "https://example.test/dep.tgz",
            "content_sha256": "d".repeat(64),
            "fetched_at": 1,
            "cached": true,
            "outcome": "ok",
        }))
        .unwrap();
        let deps = vec![DepResult {
            sha256: "d".repeat(64),
            locator: "pkg:test/dep@1".to_string(),
            url: "https://example.test/dep.tgz".to_string(),
            size: 0,
            provenance: None,
            verdict: Some(Decision {
                class: Classification::Hostile,
                probability: 0.97,
                threshold: 0.5,
                level: None,
            }),
            members: MemberEvals::new(),
            raw: "{}".to_string(),
        }];
        let registries = vec![crate::fetch::DependencyRegistry {
            locator: "pkg:test/dep@1".to_string(),
            provenance: crate::provenance::RegistryProvenance::from_record_sources(
                fletch::Registry {
                    ecosystem: "test".to_string(),
                    name: "dep".to_string(),
                    version: "1".to_string(),
                    ..fletch::Registry::default()
                },
                &[fletch::fetch::RecordedSource {
                    url: "https://registry.example/dep".to_string(),
                    status: 200,
                    content_type: Some("application/json".to_string()),
                    bytes: br#"{"provider_only":{"kept":true}}"#.to_vec(),
                }],
            ),
            file_id: 2,
            artifact_skip: None,
        }];
        let edges = vec![edge];

        let ctx = render_interpret_context(
            "root.tgz",
            &"r".repeat(64),
            None,
            None,
            &edges,
            &deps,
            &registries,
            &report,
        );
        let primary_provenance = ctx.find("provenance=").unwrap();
        let primary_trait = ctx.find("primary package finding").unwrap();
        let dep_header = ctx.find("== DEP pkg:test/dep@1").unwrap();
        let dep_provenance =
            ctx.get(dep_header..).unwrap().find("provenance=").unwrap() + dep_header;
        let dep_trait = ctx.find("dependency package finding").unwrap();
        assert!(primary_provenance < primary_trait);
        assert!(primary_trait < dep_header);
        assert!(dep_provenance < dep_trait);
        assert_eq!(ctx.matches("dependency package finding").count(), 1);
        assert!(ctx.contains(r#""record":{"ecosystem":"test","name":"dep","version":"1"}"#));
        assert!(
            ctx.contains(r#""provider_only":{"kept":true}"#),
            "interpret projection must derive from preserved raw provenance: {ctx}"
        );
        assert!(ctx.contains(r#""source_path":"root.tgz""#));
        assert!(
            !ctx.contains(":null"),
            "sparse records must omit nulls: {ctx}"
        );

        let terminal = render_terminal_fetch_context(
            &edges,
            &deps,
            &registries,
            &report,
            &cleave::output::TinyOpts::terminal(),
        )
        .expect("hostile dependency is shown");
        let provenance = terminal.find("      from       root.tgz").unwrap();
        let finding = terminal.find("dependency package finding").unwrap();
        assert!(provenance < finding);
        assert_eq!(terminal.matches("pkg:test/dep@1").count(), 1);
        assert!(
            terminal.contains(&"d".repeat(64)),
            "hash must stay complete"
        );
        assert!(!terminal.contains("transfer"));
        assert!(!terminal.contains("cache:"));
        assert!(terminal.contains("metadata   https://registry.example/dep"));
    }

    #[test]
    fn metadata_only_root_keeps_provenance_without_a_duplicate_graft() {
        let mut report = empty_report();
        report.files = vec![cleave::FileAnalysis {
            id: 0,
            path: "removed@1.registry.json".to_string(),
            file_type: "registry".to_string(),
            sha256: "a".repeat(64),
            ..cleave::FileAnalysis::default()
        }];
        assert!(
            !root_needs_registry_graft(&report),
            "the registry document is already the analyzed root"
        );
        let provenance = crate::provenance::RegistryProvenance::from_record_sources(
            fletch::Registry {
                ecosystem: "test".to_string(),
                name: "removed".to_string(),
                version: "1".to_string(),
                version_removed: Some(true),
                ..fletch::Registry::default()
            },
            &[fletch::fetch::RecordedSource {
                url: "https://registry.example/removed".to_string(),
                status: 200,
                content_type: Some("application/json".to_string()),
                bytes: br#"{"provider_only":{"kept":true}}"#.to_vec(),
            }],
        );
        let ctx = render_interpret_context(
            "removed@1.registry.json",
            &"a".repeat(64),
            None,
            Some(&provenance),
            &[],
            &[],
            &[],
            &report,
        );
        assert!(ctx.contains(r#""provider_only":{"kept":true}"#));
        assert_eq!(ctx.matches("== PRIMARY").count(), 1);

        report.files[0].file_type = "npm".to_string();
        assert!(root_needs_registry_graft(&report));
    }

    #[test]
    fn notable_composite_with_registry_source_selects_dependency_provenance() {
        let registry_id = 9;
        let mut artifact = cleave::FileAnalysis {
            id: 4,
            ..cleave::FileAnalysis::default()
        };
        artifact.findings.push(cleave::Finding {
            id: "package/composite".to_string(),
            crit: cleave::Criticality::Notable,
            ..cleave::Finding::default()
        });
        let mut artifact_json = serde_json::to_value(&artifact).unwrap();
        artifact_json["composite_sources"] = serde_json::json!({
            "package/composite": [{"file": registry_id}]
        });
        let artifact: cleave::FileAnalysis = serde_json::from_value(artifact_json).unwrap();
        let registry = cleave::FileAnalysis {
            id: registry_id,
            rel: cleave::types::Rel::Registry,
            role: cleave::types::Role::Sidecar,
            ..cleave::FileAnalysis::default()
        };
        let mut report = empty_report();
        report.files = vec![artifact, registry];
        assert!(provenance_has_notable_match(
            &report,
            &[4].into_iter().collect(),
            registry_id,
        ));
    }

    #[test]
    fn large_raw_registry_json_is_version_focused_and_valid() {
        let body = serde_json::json!({
            "name": "dep",
            "versions": (0..100)
                .map(|n| (format!("1.0.{n}"), serde_json::json!({"version": format!("1.0.{n}")})))
                .collect::<serde_json::Map<String, serde_json::Value>>(),
            "provider_extra": (0..100)
                .map(|n| (format!("field_{n}"), serde_json::Value::String("x".repeat(2_000))))
                .collect::<serde_json::Map<String, serde_json::Value>>(),
        });
        let projected = focus_registry_json(&body, "1.0.42", 0);
        assert_eq!(projected["versions"]["1.0.42"]["version"], "1.0.42");
        assert!(
            projected["versions"].as_object().unwrap().len() < 10,
            "large version maps should not flood interpret context"
        );
        let encoded = serde_json::to_vec(&projected).expect("projection remains valid JSON");
        assert!(
            encoded.len() < 8 * 1024,
            "projection should stay comfortably bounded, got {} bytes",
            encoded.len()
        );
    }

    #[test]
    fn terminal_url_fetch_omits_duplicate_urls_and_never_shortens_sha256() {
        let sha = "a".repeat(64);
        let mut rec: fletch::fetch::FetchRecord = serde_json::from_value(serde_json::json!({
            "source_sha256": "r".repeat(64),
            "source_offset": 42,
            "kind": "url_fetch",
            "locator": "https://example.test/stage.sh",
            "resolved_url": "https://example.test/stage.sh",
            "final_url": "https://example.test/stage.sh",
            "content_sha256": sha,
            "fetched_at": 1,
            "cached": false,
            "outcome": "ok",
        }))
        .unwrap();
        let mut report = empty_report();
        report.files = vec![cleave::FileAnalysis {
            id: 0,
            path: "dropper.sh".to_string(),
            sha256: "r".repeat(64),
            ..cleave::FileAnalysis::default()
        }];
        let mut out = String::new();
        write_terminal_fetch_provenance(&mut out, &rec, &report);
        assert!(out.contains("from       dropper.sh @ byte 42"));
        assert!(out.contains(&format!("sha256     {sha}")));
        assert!(!out.contains("resolved"));
        assert!(!out.contains("final"));

        rec.final_url = Some("https://cdn.example.test/stage.sh".to_string());
        out.clear();
        write_terminal_fetch_provenance(&mut out, &rec, &report);
        assert!(out.contains("final      https://cdn.example.test/stage.sh"));
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
    root_registry: Option<&crate::provenance::RegistryProvenance>,
    root_fetch: Option<&fletch::fetch::FetchRecord>,
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
        Some(EMBEDDED_FILE_LIMIT),
        &tiny_opts_for(config),
        config.interpret(),
        // `--format interpret` emits byte-for-byte what a live `--interpret`
        // query sends as its user message — the sanitized render, annotations
        // included — without the system prompt (which hedges the descriptions
        // as fallible; a downstream consumer should frame them likewise).
        matches!(config.format(), OutputFormat::Interpret),
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
        root_fetch,
        bloom_mark,
    )?;

    // Per-file phase timing (CLI path only — serve logs its own per-sample line).
    // Makes a slow scan self-diagnosing: total_ms is the post-analysis wall clock,
    // and the phase split says where it went. Subtract total_ms from the caller's
    // whole-invocation elapsed (e.g. gauntlet's per-sample time) to get the static
    // cleave/filefacts share, which is measured upstream of here.
    let pm = cr.phase_ms;
    tracing::info!(
        path = %path_display,
        sha256 = %cr.sha256,
        file_type = %cr.file_type,
        class = ?cr.classification,
        fetch_ms = pm.fetch_ms,
        interpret_ms = pm.interpret_ms,
        render_ms = pm.render_ms,
        total_ms = pm.total_ms,
        "scan phases complete",
    );

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
        None,
    )
}

/// Scan a payload that already lives on disk, returning the same [`ScanResult`]
/// as [`scan_bytes`]. cleave memory-maps the file, so peak resident memory stays
/// bounded regardless of the payload's size — the entry point for large,
/// streamed-to-disk responses that must not be buffered whole in RAM.
///
/// `read_path` is the on-disk file to analyze; `filename` is the logical name
/// (e.g. the originating URL) echoed into the report and used for the label,
/// exactly as in [`scan_bytes`]. `read_path`'s extension still drives cleave's
/// type detection, so callers should give the temporary file a name that carries
/// the payload's real extension.
///
/// # Errors
/// Propagates cleave analysis failures and model inference errors.
pub fn scan_file(
    read_path: &Path,
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
    let report = cleave::analyze_file(read_path, &cleave_opts)
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

/// Structural-integrity counts for a compact report. All zero on a well-formed
/// report; each non-zero field is logged once (not per finding) as a
/// producer-side signal. One HashSet scan over findings — negligible beside
/// model inference.
#[derive(Debug, Default, PartialEq, Eq)]
struct ReportIntegrity {
    /// `from[].file` entries that don't resolve to an emitted file id — the
    /// trait then renders downstream (hopper/prism) with no file context.
    dangling_refs: usize,
    /// `role:sidecar` files with no `pid`. A sidecar is metadata *about* a
    /// parent node, so an absent parent means it describes nothing.
    orphan_sidecars: usize,
}

/// Verify the compact report's structural invariants and return the counts so
/// callers and tests can assert on them. A finding's `from[].file` entries index
/// into `files[]`; if a member is dropped or renumbered without remapping these,
/// the index dangles and the trait renders downstream (hopper/prism) with no file
/// context. A `role:sidecar` must also name a parent (see [`ReportIntegrity`]).
/// We can't repair these here, but a producer-side log turns each into a visible
/// signal instead of a mystery on the rendering side.
fn validate_report_references(
    label: &str,
    report: &cleave::types::compact::CompactReport,
) -> ReportIntegrity {
    let ids: std::collections::HashSet<u32> = report.files.iter().map(|f| f.id).collect();

    let mut integ = ReportIntegrity::default();
    let mut ref_sample: Vec<String> = Vec::new();
    for file in &report.files {
        // Invariant 3: a sidecar must describe a parent node.
        if matches!(file.role, cleave::types::Role::Sidecar) && file.parent.is_none() {
            integ.orphan_sidecars += 1;
        }
        for finding in &file.findings {
            // v8 merged the old `src` (inherited single source) and `sources[]`
            // (cross-file composite members) into one `from: Vec<CompactSource>`.
            for s in &finding.from {
                if !ids.contains(&s.file) {
                    integ.dangling_refs += 1;
                    if ref_sample.len() < 3 {
                        ref_sample.push(format!("{}->#{}", finding.id, s.file));
                    }
                }
            }
        }
    }

    if integ.dangling_refs > 0 {
        tracing::error!(
            label,
            dangling = integ.dangling_refs,
            files = report.files.len(),
            examples = %ref_sample.join(", "),
            "compact report integrity: cross-file references point at file ids not in files[]; \
             affected traits will render without file context downstream"
        );
    }
    if integ.orphan_sidecars > 0 {
        tracing::error!(
            label,
            orphan_sidecars = integ.orphan_sidecars,
            "compact report integrity: role:sidecar files without a pid; a sidecar must \
             describe a parent node"
        );
    }
    integ
}

#[cfg(test)]
mod integrity_tests {
    use super::validate_report_references;
    use cleave::types::{FileAnalysis, Role};

    fn file(id: u32, path: &str, ftype: &str, sha: &str, parent: Option<u32>) -> FileAnalysis {
        FileAnalysis {
            id,
            path: path.into(),
            file_type: ftype.into(),
            sha256: sha.into(),
            size: 10,
            parent_id: parent,
            ..Default::default()
        }
    }

    #[test]
    fn flags_orphan_sidecar() {
        // A sidecar with no `pid` describes nothing — the one thing invariant 3
        // rejects. The ordinary parent/member pair around it stays clean.
        let mut orphan = file(2, "reg", "registry", "s2", None);
        orphan.role = Role::Sidecar;
        let report = cleave::types::compact::compact_from_files(&[
            file(0, "a.tar", "tar", "s0", None),
            file(1, "a.tar!!x.py", "python", "s1", Some(0)),
            orphan,
        ]);
        let integ = validate_report_references("test", &report);
        assert_eq!(integ.orphan_sidecars, 1, "orphan sidecar");
        assert_eq!(integ.dangling_refs, 0, "no dangling refs");
    }

    #[test]
    fn clean_report_has_zero_integrity_counts() {
        // A sidecar that correctly names its parent is fine.
        let mut sidecar = file(1, "reg", "registry", "s1", Some(0));
        sidecar.role = Role::Sidecar;
        let report = cleave::types::compact::compact_from_files(&[
            file(0, "a.tar", "tar", "s0", None),
            sidecar,
        ]);
        let integ = validate_report_references("test", &report);
        assert_eq!(integ, super::ReportIntegrity::default());
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
/// Each entry is `{id, type, prob, lvl, conf}` keyed by the cleave `files[].id`
/// field. The root file (`dp=0`) carries the envelope's probability and `lvl`;
/// embedded archive members are matched by id and report their *own*
/// probability and lowest-firing-level `lvl` — every row's `lvl` is therefore
/// the level-independent marker for that specific file. A member with no
/// recorded evaluation (e.g. truncated past the embedded-file cap) emits only
/// `{id, type}` — no verdict fields. It must NEVER inherit the root's verdict:
/// a member can occur in many containers, and hopper mirrors these entries
/// into the member's own sample row (`litmusResultForMember`), so a fabricated
/// value becomes that file's global grade everywhere it appears.
fn build_ml_files(
    report_json: &serde_json::Value,
    root_prob: f32,
    root_level: Option<i32>,
    embedded_files: &MemberEvals,
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

        let evaluation = if depth == 0 {
            Some((root_prob, root_level))
        } else {
            embedded_files.get(&id).map(|ef| (ef.probability, ef.level))
        };

        match evaluation {
            Some((prob, file_level)) => out.push(serde_json::json!({
                "id": id,
                "type": file_type,
                "prob": prob,
                "lvl": file_level,
                "conf": level_confidence(file_level),
            })),
            // No verdict fields: consumers (hopper's forMember) treat a
            // prob-less entry as "not analyzed", which is the truth.
            None => out.push(serde_json::json!({
                "id": id,
                "type": file_type,
            })),
        }
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
