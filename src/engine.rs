//! Recursive file-system scanning and classification.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant, SystemTime};

use crate::Mode;
use crate::OutputFormat;
use crate::bloom_repo::{Decision as BloomDecision, Lookup};
use crate::explain::ShapImportance;
use crate::features::ExtractContext;
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
            zip_passwords: crate::ArchivePasswords::default(),
            // Bloom short-circuiting is opt-in via `with_bloom`; an unconfigured
            // config runs a full scan (slow mode), so server/fs paths are unaffected.
            mode: crate::Mode::Slow,
            bloom: None,
        })
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

    /// Decode a wire-shaped fixture into the typed report the pipeline carries,
    /// so these tests exercise the same reader production does.
    fn report(v: serde_json::Value) -> cleave::types::CompactReport {
        serde_json::from_value(v).unwrap()
    }

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
        let mut report = report(serde_json::json!({
            "files": [
                {"id": 0, "path": "app.zip", "type": "zip", "sha": "root", "size": 4096},
                {"id": 1, "path": "app.zip!!evil.sh", "type": "shell", "sha": "aaa", "size": 200, "risk": 9},
            ]
        }));
        let members = vec![
            // Already analyzed (matches sha "aaa") — must be skipped, no duplicate.
            stub("evil.sh", "shell", "aaa", 200),
            // Never analyzed — must be appended as a listing-only entry.
            stub("README.md", "markdown", "bbb", 1024),
            // Nested member: single `!` becomes `!!`, depth counts the levels.
            stub("inner.tar!logo.png", "png", "ccc", 8192),
        ];
        append_unanalyzed_members(&mut report, &members);

        let files = &report.files;
        assert_eq!(files.len(), 4, "two listing-only members appended");

        let readme = &files[2];
        assert_eq!(readme.id, 2);
        assert_eq!(readme.path, "app.zip!!README.md");
        assert_eq!(readme.file_type, "markdown");
        assert_eq!(readme.size, 1024);
        assert_eq!(readme.depth, 1);
        assert_eq!(
            readme.risk, UNANALYZED_MEMBER_RISK,
            "sentinel marks the member unanalyzed"
        );

        let logo = &files[3];
        assert_eq!(logo.path, "app.zip!!inner.tar!!logo.png");
        assert_eq!(logo.depth, 2);
        assert_eq!(logo.risk, UNANALYZED_MEMBER_RISK);
    }

    #[test]
    fn build_ml_files_drops_listing_only_members() {
        // A root file, an analyzed member, and a listing-only member (risk -1).
        let report = report(serde_json::json!({
            "files": [
                {"id": 0, "path": "app.zip", "type": "zip", "sha": "r", "size": 4096, "depth": 0},
                {"id": 1, "path": "app.zip!!evil.sh", "type": "shell", "sha": "a", "size": 200, "depth": 1, "risk": 9},
                {"id": 2, "path": "app.zip!!README.md", "type": "markdown", "sha": "b", "size": 1024, "depth": 1, "risk": -1},
            ]
        }));
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
        let report = report(serde_json::json!({
            "files": [
                {"id": 0, "path": "evil.elf", "type": "elf", "sha": "0", "size": 9, "depth": 0},
                {"id": 1, "path": "compatibility", "type": "unknown", "sha": "1", "size": 1, "depth": 1},
                {"id": 2, "path": "a!!page", "type": "unknown", "sha": "2", "size": 1, "depth": 2},
                {"id": 3, "path": "b!!page", "type": "unknown", "sha": "3", "size": 1, "depth": 2},
                {"id": 4, "path": "never-scored", "type": "unknown", "sha": "4", "size": 1, "depth": 1},
            ]
        }));
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod trait_floor_tests {
    use super::*;

    fn benign() -> Decision {
        Decision {
            class: Classification::Benign,
            probability: 0.1,
            threshold: 0.65,
            level: Some(-1),
        }
    }

    /// One finding, in the family `objectives/<area>/<n>` — distinct `area`
    /// values put findings in distinct families.
    fn finding(area: &str, crit: u8, conf: f32) -> cleave::types::CompactTrait {
        cleave::types::CompactTrait {
            id: format!("objectives/{area}/sub/leaf::trait-{crit}-{conf}"),
            criticality: crit,
            confidence: conf,
            ..cleave::types::CompactTrait::default()
        }
    }

    /// `n` findings at `crit`/`conf`, each in its own family.
    fn spread(crit: u8, conf: f32, n: usize) -> Vec<cleave::types::CompactTrait> {
        (0..n)
            .map(|i| finding(&format!("area{i}"), crit, conf))
            .collect()
    }

    /// `n` findings at `crit`/`conf`, all in one family — the shape a single
    /// behavior described several ways produces.
    fn clustered(crit: u8, conf: f32, n: usize) -> Vec<cleave::types::CompactTrait> {
        (0..n)
            .map(|i| cleave::types::CompactTrait {
                id: format!("objectives/evasion/process/hook/inline::variant-{i}"),
                criticality: crit,
                confidence: conf,
                ..cleave::types::CompactTrait::default()
            })
            .collect()
    }

    #[test]
    fn recategorizing_an_annotation_survives_multibyte_descriptions() {
        // A description opening with a multi-byte character used to split the
        // severity off at a computed byte index, landing inside the character.
        let line =
            "  // H über-loader resolves imports (objectives/anti-static/obfuscation/string::x)";
        assert_eq!(
            recategorize_annotation(line).as_deref(),
            Some("  // Possible anti-static/obfuscation — über-loader resolves imports"),
        );
        // Multi-byte content anywhere else is fine too, and a whole render of it
        // must not panic.
        let render = "== PRIMARY x ==\n# S 4:2 naïve café résumé (micro-behaviors/data/encode::y)\n  körper\n";
        assert!(
            recategorize_annotations(render).contains("Possible data/encode — naïve café résumé")
        );
        // `well-known/` keeps its full depth: the family name is the finding.
        assert_eq!(
            recategorize_annotation(
                "// S Owhit new-tab wallpaper extension identity (well-known/unwanted/newtab-wallpaper-adware/owhit::identity)"
            )
            .as_deref(),
            Some("// Possible unwanted/newtab-wallpaper-adware/owhit — Owhit new-tab wallpaper extension identity"),
        );
        // ...while an objectives/ path is still cut to two, so a technique
        // taxonomy does not turn into a wall of near-identical labels.
        assert_eq!(
            trait_category("objectives/evasion/process/injection/hollowing::x"),
            "evasion/process"
        );
        assert_eq!(
            trait_category("well-known/malware/dropper/nemucod/obfuscation::y"),
            "malware/dropper/nemucod/obfuscation"
        );

        // Non-annotation lines pass through untouched.
        assert_eq!(recategorize_annotation("  let x = 1;"), None);
        assert_eq!(recategorize_annotation("# H no trait id here"), None);
    }

    #[test]
    fn recategorizing_covers_every_annotation_the_render_can_emit() {
        // `interpret::parse_annotation` decides what *is* an annotation, over the
        // marker set `// -- #` and the grades `HSNBCF`. Anything it admits and
        // this does not reaches the grader with its grade letter intact — the one
        // thing presenting observations instead of verdicts exists to prevent.
        assert_eq!(
            recategorize_annotation(
                "-- C .NET set_Item reference (micro-behaviors/data/manipulation::setter)"
            )
            .as_deref(),
            Some("-- Possible data/manipulation — .NET set_Item reference"),
        );
        assert_eq!(
            recategorize_annotation("// F 9:1 packed section (metadata/binary/packer::upx)")
                .as_deref(),
            // The `line:col` pointer survives ahead of the category — it locates
            // the finding rather than grading it.
            Some("// 9:1 Possible binary/packer — packed section"),
        );
        // A third-party signature has no prose and no parenthesized id: the path
        // *is* the body. Left alone, `// H Detects Quasar RAT (third_party/…)`
        // and its bare cousin were the loudest grades still leaking through.
        assert_eq!(
            recategorize_annotation(
                "// H third_party/elastic/Linux_Trojan_Ladvix/linux/trojan/ladvix"
            )
            .as_deref(),
            Some("// Possible elastic/Linux_Trojan_Ladvix"),
        );
        assert_eq!(
            recategorize_annotation("// H Detects Quasar RAT (third_party/SigBase/Quasar/RAT)")
                .as_deref(),
            Some("// Possible SigBase/Quasar — Detects Quasar RAT"),
        );
        // A parenthetical that is prose, not a trait id, still leaves the line be.
        assert_eq!(
            recategorize_annotation("# S writes a file (see below)"),
            None
        );
    }

    #[test]
    fn family_is_the_first_two_hierarchy_segments() {
        assert_eq!(
            trait_family("objectives/evasion/process/hook/inline::rust-inline-hook-hijack"),
            "objectives/evasion"
        );
        // One behavior spelled several ways shares a family...
        assert_eq!(
            trait_family("objectives/evasion/process/hook/inline::rust-inline-hook-hijack"),
            trait_family("objectives/evasion/process/hook/inline::rust-hook-byte-copy"),
        );
        // ...and so does one behavior restated under sibling techniques, which
        // is what a four-deep family missed: `ansible-core`'s base64
        // decode-and-execute counted once per subdirectory it was spelled in,
        // and three such counts cleared a diversity test meant to require three
        // independent witnesses.
        assert_eq!(
            trait_family("objectives/anti-static/obfuscation/string/reconstruct::a"),
            trait_family("objectives/anti-static/obfuscation/eval/scripting::b"),
        );
        // Distinct objectives stay distinct.
        assert_ne!(
            trait_family("objectives/evasion/process/hook/inline::a"),
            trait_family("objectives/supply-chain/install-hook/npm::b"),
        );
        // The accepted cost: `darkglitch`'s two genuinely distinct backdoor
        // capabilities now share a family, so a verdict resting on that pair
        // alone no longer clears the diversity test. See
        // [`TRAIT_FLOOR_FAMILY_DEPTH`].
        assert_eq!(
            trait_family("objectives/command-and-control/backdoor/rat/multi::a"),
            trait_family("objectives/command-and-control/backdoor/tasking/filesystem::b"),
        );
        // A short or path-less id is its own family, never a shared bucket.
        assert_eq!(
            trait_family("micro-behaviors/mem::x"),
            "micro-behaviors/mem"
        );
        assert_eq!(trait_family("bare-id"), "bare-id");
        // The leaf trait id is never part of the family, at any path length —
        // otherwise every trait would be its own family and the diversity test
        // would pass on any cluster of near-duplicates.
        for id in [
            "objectives/evasion/process/hook/inline::rust-inline-hook-hijack",
            "objectives/a/b::leaf",
            "micro-behaviors/mem::x",
            "a::b",
        ] {
            assert!(
                !trait_family(id).contains("::"),
                "family kept the trait id: {id} -> {}",
                trait_family(id)
            );
        }
        // Two traits differing only in their leaf are one family.
        assert_eq!(
            trait_family("objectives/a/b/c/d::one"),
            trait_family("objectives/a/b/c/d::two"),
        );
    }

    #[test]
    fn two_crit5_with_a_third_severe_escalates_to_hostile_band() {
        let mut d = benign();
        let mut ts = spread(5, 0.8, 2);
        ts.push(finding("supply-chain", 4, 0.9));
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        // The probability is the strongest *crit-5*, not the strongest finding.
        assert_eq!(d.probability, 0.8);
        assert_eq!(d.level, Some(50));
    }

    #[test]
    fn a_lone_crit5_stays_benign() {
        let mut d = benign();
        // One hostile trait cannot carry a verdict, however confident: nothing
        // corroborates it. This is the shape that graded WannaCry hostile off a
        // single finding — and static-keys benign code with it.
        apply_trait_floor(&mut d, &spread(5, 0.98, 1), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn a_crit5_with_one_thin_corroborator_stays_benign() {
        let mut d = benign();
        // Two findings in two families still falls short of the severe count.
        // This is the real pair that marked a stock `libwebp.dll` hostile: an
        // RMM signature plus a generic PE-layout anomaly.
        let ts = vec![
            finding("command-and-control", 5, 0.9),
            finding("binary-anomaly", 4, 0.97),
        ];
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn two_crit5_corroborated_across_families_escalates_to_hostile_band() {
        let mut d = benign();
        // Two anchors plus a further severe finding.
        let mut ts = spread(5, 0.98, 2);
        ts.extend([finding("execution", 4, 0.94)]);
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.probability, 0.98);
        assert_eq!(d.level, Some(50));
    }

    #[test]
    fn two_families_no_longer_reach_the_hostile_arm() {
        // Enough anchors and enough severe findings, but drawn from only two
        // kinds of behavior — the `PyAutoIt` shape, where input synthesis and
        // window manipulation are the package's whole purpose. Both arms now ask
        // for three (see [`TRAIT_FLOOR_HOSTILE_FAMILIES`]).
        let mut d = benign();
        let mut ts = spread(5, 0.98, 2);
        ts.extend([finding("area0", 4, 0.94), finding("area1", 4, 0.92)]);
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_ne!(d.class, Classification::Hostile);

        // A third family is what earns it.
        let mut wider = benign();
        let mut ts = spread(5, 0.98, 2);
        ts.extend([finding("execution", 4, 0.94)]);
        apply_trait_floor(&mut wider, &ts, Some(50), 100, "test");
        assert_eq!(wider.class, Classification::Hostile);
    }

    #[test]
    fn a_lone_crit5_beside_crit4s_no_longer_reaches_the_hostile_arm() {
        // The `rex-powershell` shape, and the one `ansible-core` fired on: a
        // single crit-5 anchor with two crit-4s beside it. An accepted
        // regression — see [`TRAIT_FLOOR_HOSTILE_CRIT5`]. It does not go
        // unremarked, only unblocked: the suspicious arm still has it.
        let mut d = benign();
        let mut ts = spread(5, 0.98, 1);
        ts.extend([finding("evasion", 4, 0.8), finding("execution", 4, 0.94)]);
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_ne!(d.class, Classification::Hostile);
    }

    #[test]
    fn recognized_software_needs_one_more_anchor_before_the_hostile_arm() {
        let evidence = || {
            let mut ts = spread(5, 0.98, 2);
            ts.extend([finding("execution", 4, 0.94)]);
            ts
        };
        // Two anchors clear the arm for a sample cleave cannot name...
        let mut unknown = benign();
        apply_trait_floor(&mut unknown, &evidence(), Some(50), 100, "test");
        assert_eq!(unknown.class, Classification::Hostile);

        // ...and the same evidence does not, once it can. Blocking a named
        // application on evidence this thin is what promoted `ansible-core`.
        let mut known = benign();
        let mut with_id = evidence();
        with_id.push(cleave::types::CompactTrait {
            id: "well-known/app/infrastructure/ansible::module-utils-path".to_string(),
            criticality: 1,
            confidence: 0.95,
            ..cleave::types::CompactTrait::default()
        });
        apply_trait_floor(&mut known, &with_id, Some(50), 100, "test");
        assert_ne!(known.class, Classification::Hostile);

        // Recognizing a sample as *malware* is not recognition that holds back.
        let mut named_malware = benign();
        let mut with_malware_id = evidence();
        with_malware_id.push(cleave::types::CompactTrait {
            id: "well-known/malware/rat/darkglitch::tasking".to_string(),
            criticality: 1,
            confidence: 0.95,
            ..cleave::types::CompactTrait::default()
        });
        apply_trait_floor(&mut named_malware, &with_malware_id, Some(50), 100, "test");
        assert_eq!(named_malware.class, Classification::Hostile);
    }

    #[test]
    fn three_crit4_in_two_families_stays_benign() {
        let mut d = benign();
        // Count alone would clear the suspicious arm; breadth does not.
        let ts = vec![
            finding("evasion", 4, 0.9),
            finding("evasion", 4, 0.92),
            finding("execution", 4, 0.88),
        ];
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn two_crit5_without_further_corroboration_stays_benign() {
        let mut d = benign();
        apply_trait_floor(&mut d, &spread(5, 0.9, 2), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn a_single_family_cluster_cannot_corroborate_itself() {
        let mut d = benign();
        // Three confident crit-5s, all from `objectives/evasion/process` —
        // counts clear, diversity does not. This is the static-keys shape.
        apply_trait_floor(&mut d, &clustered(5, 0.98, 3), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
        // Same for the suspicious arm.
        let mut d = benign();
        apply_trait_floor(&mut d, &clustered(4, 0.93, 4), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn low_confidence_crit5_is_ignored() {
        let mut d = benign();
        // c < 0.76 → not counted, stays benign.
        apply_trait_floor(&mut d, &spread(5, 0.5, 3), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn three_confident_crit4_across_families_escalates_to_suspicious_band() {
        let mut d = benign();
        apply_trait_floor(&mut d, &spread(4, 0.9, 3), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Suspicious);
        assert_eq!(d.probability, 0.9);
        // Placed inside the band (51..=100) rather than pinned to its weakest
        // rung — see `interpreted_level`. No crit-5 here, so it sits at the
        // uncorroborated midpoint.
        let lvl = d.level.expect("a floored verdict carries a level");
        assert!(lvl > 50 && lvl < 100, "expected inside the band, got {lvl}");
    }

    #[test]
    fn two_confident_crit4_stays_benign() {
        let mut d = benign();
        apply_trait_floor(&mut d, &spread(4, 0.9, 2), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn a_busy_file_is_not_diluted_out_of_an_escalation() {
        let mut d = benign();
        // Three confident crit-4s in distinct families, among 200 baseline
        // findings. The fraction gate this replaced scored ~0.015 here and
        // stayed benign; activity elsewhere in the file is not evidence about
        // these three.
        let mut ts = spread(4, 0.9, 3);
        ts.extend((0..200).map(|i| finding(&format!("noise{i}"), 0, 0.9)));
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Suspicious);
    }

    #[test]
    fn low_confidence_crit4_does_not_count_toward_the_trio() {
        let mut d = benign();
        // Only one confident crit-4; the other two are below threshold.
        let ts = vec![
            finding("a", 4, 0.9),
            finding("b", 4, 0.5),
            finding("c", 4, 0.6),
        ];
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    /// cleave now always writes `conf`, but reports from builds that omitted it
    /// still decode — as 0.0, "no confidence recorded", which is below the 0.76
    /// gate. So unscored crit-5s in an old report never trip the floor.
    #[test]
    fn confidence_omitted_by_an_older_build_decodes_below_threshold() {
        let mut d = benign();
        let ts: Vec<cleave::types::CompactTrait> = serde_json::from_value(serde_json::json!([
            {"id": "objectives/a/b/c::x", "crit": 5},
            {"id": "objectives/d/e/f::y", "crit": 5},
            {"id": "objectives/g/h/i::z", "crit": 4},
        ]))
        .unwrap();
        assert_eq!(
            ts[0].confidence, 0.0,
            "an omitted conf records no confidence"
        );
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn never_lowers_a_non_benign_verdict() {
        let mut d = benign();
        d.class = Classification::Hostile;
        d.level = Some(50);
        apply_trait_floor(&mut d, &spread(4, 0.9, 5), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.level, Some(50));
    }

    #[test]
    fn floors_on_the_root_files_own_findings() {
        // `root_findings` is what the classify path passes: a report's findings
        // live on files[0], never at report level.
        let mut d = benign();
        let report: cleave::types::CompactReport = serde_json::from_value(serde_json::json!({
            "files": [{"id": 0, "path": "x", "type": "elf", "sha": "s", "size": 1,
                       "traits": [
                           {"id": "objectives/command-and-control/backdoor/a::t1", "crit": 5, "conf": 0.98},
                           {"id": "objectives/persistence/service/b::t2", "crit": 5, "conf": 0.9},
                           {"id": "objectives/discovery/env-vars/c::t3", "crit": 4, "conf": 0.8}
                       ]}]
        }))
        .unwrap();
        apply_trait_floor(&mut d, root_findings(&report), Some(50), 100, "test");
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

    /// A target the operator named is never handed to cleave with a skip
    /// predicate: the bless is a bulk shortcut for directory walks and fetched
    /// dependencies, and answering "scan this file" from one returns a lookup
    /// where an analysis was asked for. The directory-walk options keep theirs.
    #[test]
    fn named_target_opts_drop_the_skip_predicate_and_keep_everything_else() {
        let walk = cleave::AnalysisOptions {
            slow_rule_ms: 1234,
            skip_predicate: Some(cleave::SkipPredicate(Arc::new(|_, _| true))),
            ..Default::default()
        };
        assert!(
            walk.skip_predicate.is_some(),
            "a directory walk keeps the known-good shortcut"
        );

        let named = named_target_opts(&walk);
        assert!(
            named.skip_predicate.is_none(),
            "a named target must be analyzed on its own merits"
        );
        // Only the predicate moves; everything else the caller configured has to
        // survive, or a named file would be analyzed under different settings
        // than the same file found by walking its parent directory.
        assert_eq!(named.slow_rule_ms, walk.slow_rule_ms);
    }

    /// A fetched package carries its locator to the uploader; a fetched URL and
    /// a local path carry none, because neither names a package. The verdict's
    /// identity and the sidecar's package slot read the same rule.
    #[test]
    fn fetched_purl_is_only_a_package_locator() {
        // Built from JSON rather than a struct literal: FetchRecord has a
        // dozen fields this rule does not read, and listing them here would
        // make the test break on every unrelated field added to it.
        let record = |locator: &str| {
            serde_json::from_value::<fletch::fetch::FetchRecord>(serde_json::json!({
                "locator": locator,
                "fetched_at": 0,
                "cached": false,
                "outcome": "ok",
            }))
            .expect("FetchRecord from locator alone")
        };
        assert_eq!(
            fetched_purl(Some(&record("pkg:npm/left-pad@1.3.0"))),
            Some("pkg:npm/left-pad@1.3.0"),
        );
        assert_eq!(
            fetched_purl(Some(&record("https://example.com/a.tgz"))),
            None
        );
        assert_eq!(fetched_purl(None), None);

        // The logged identity recovers a package from a registry URL too, so a
        // `scan url` of a tarball reports what a reader recognises. An
        // arbitrary URL still identifies nothing.
        assert_eq!(
            artifact_purl(Some(&record(
                "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz"
            ))),
            Some("pkg:npm/left-pad@1.3.0".to_string()),
        );
        assert_eq!(
            artifact_purl(Some(&record("pkg:npm/left-pad@1.3.0"))),
            Some("pkg:npm/left-pad@1.3.0".to_string()),
        );
        assert_eq!(
            artifact_purl(Some(&record("https://example.com/a.tgz"))),
            None
        );
        assert_eq!(artifact_purl(None), None);
    }

    /// The sidecar hopper stores binds the package a URL-fetched artifact
    /// belongs to, so `scan url <registry tarball>` and the equivalent `scan
    /// purl` deposit the same identity rather than one bound row and one
    /// anonymous one.
    #[test]
    fn collect_upload_artifacts_binds_a_recovered_package() {
        let fetch = |locator: &str, url: &str| {
            serde_json::from_value::<fletch::fetch::FetchRecord>(serde_json::json!({
                "locator": locator,
                "resolved_url": url,
                "fetched_at": 0,
                "cached": false,
                "outcome": "ok",
            }))
            .expect("FetchRecord")
        };
        let package_of = |rec: &fletch::fetch::FetchRecord| {
            let arts = collect_upload_artifacts(
                Path::new("ignored"),
                &"a".repeat(64),
                10,
                "t",
                None,
                Some(rec),
            );
            let sidecar: serde_json::Value =
                serde_json::from_slice(&arts[0].sidecar).expect("sidecar json");
            sidecar["package"].clone()
        };

        let url = "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz";
        assert_eq!(
            package_of(&fetch(url, url))["purl"],
            serde_json::json!("pkg:npm/left-pad@1.3.0"),
            "a registry URL is stored under the package it names",
        );
        assert_eq!(
            package_of(&fetch("pkg:npm/left-pad@1.3.0", url))["purl"],
            serde_json::json!("pkg:npm/left-pad@1.3.0"),
            "and matches what the PURL spelling stores",
        );
        // A URL that names no package must not acquire an invented identity.
        assert!(
            package_of(&fetch(
                "https://example.com/a.tgz",
                "https://example.com/a.tgz"
            ))
            .is_null(),
            "no package slot without a package",
        );
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
            None,
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
            raw: serde_json::json!({"v": "8", "files": [
                {"id": 0, "path": "evil-1.0.0.tgz", "size": 1234, "sha": "d".repeat(64), "type": "npm"}
            ]})
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
            env.raw.files.first().map(|f| f.file_type.as_str()),
            Some("npm"),
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
            analysis_cached: false,
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
            pending_llm: None,
            dependency_results: Vec::new(),
            bloom_mark: None,
            hopper_route: HopperRoute::Normal,
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
    fn interpreted_level_places_a_verdict_inside_its_band() {
        use crate::model::{capped_suspicious_level, verdict_for_level};
        use Classification::{Benign, Hostile, Suspicious};

        let grid_max = 30_000;
        let ceiling = i32::from(capped_suspicious_level(grid_max));
        for deploy in [4_u16, 5, 25, 50] {
            // Escalation to hostile lands on the active deploy level: the loosest
            // rung still inside the hostile budget.
            assert_eq!(
                interpreted_level(Some(deploy), grid_max, Hostile, true),
                Some(i32::from(deploy))
            );

            // Suspicious lands *within* the band, not on its weakest rung, so two
            // interpreted verdicts of differing strength no longer collapse onto
            // one number.
            let alone = interpreted_level(Some(deploy), grid_max, Suspicious, false).unwrap();
            let corroborated = interpreted_level(Some(deploy), grid_max, Suspicious, true).unwrap();
            assert!(
                i32::from(deploy) < corroborated && corroborated < alone && alone < ceiling,
                "deploy {deploy}: expected deploy < {corroborated} < {alone} < {ceiling}",
            );

            // The round trip is the load-bearing part: whatever level we synthesize,
            // the model's own classifier must read it back as the class we lifted
            // the sample to, or a `lvl`-only consumer (hopper) sees a different
            // verdict than we published.
            for lvl in [
                interpreted_level(Some(deploy), grid_max, Hostile, true).unwrap(),
                alone,
                corroborated,
            ] {
                let lvl = u16::try_from(lvl).unwrap();
                assert_ne!(
                    verdict_for_level(lvl, deploy, grid_max),
                    Benign,
                    "deploy {deploy}: level {lvl} read back as benign",
                );
            }
            assert_eq!(
                verdict_for_level(u16::try_from(alone).unwrap(), deploy, grid_max),
                Suspicious
            );
            assert_eq!(
                verdict_for_level(u16::try_from(corroborated).unwrap(), deploy, grid_max),
                Suspicious
            );
        }
        // A grid tighter than the ceiling keeps the placement inside it.
        let tight = interpreted_level(Some(25), 2_000, Suspicious, false).unwrap();
        assert!(tight <= i32::from(capped_suspicious_level(2_000)) && tight > 25);
        // A deploy level at or above the ceiling leaves no band to place within.
        assert_eq!(
            interpreted_level(Some(3_000), grid_max, Suspicious, false),
            Some(ceiling)
        );
        // Benign is the clean marker regardless of grid (even in manual mode).
        assert_eq!(
            interpreted_level(Some(25), grid_max, Benign, false),
            Some(-1)
        );
        assert_eq!(interpreted_level(None, 0, Benign, false), Some(-1));
        // Manual-threshold mode (no grid): no synthetic hostile/suspicious level.
        assert_eq!(interpreted_level(None, 0, Hostile, true), None);
        assert_eq!(interpreted_level(None, 0, Suspicious, false), None);
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
        r.cleave = Some(
            serde_json::from_value(serde_json::json!({
                "files": [
                    {"id": 0, "size": 1, "depth": 0, "sha": "a", "path": "/tmp/x", "type": "zip"},
                    {"id": 1, "size": 1, "depth": 1, "sha": "b", "path": "/tmp/x!!evil.sh", "type": "shell"},
                    {"id": 2, "size": 1, "depth": 1, "sha": "c", "path": "/tmp/x!!readme.txt", "type": "text"},
                ]
            }))
            .unwrap(),
        );
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
/// Where a hostile verdict lands when the LLM clears it.
///
/// Positioned by how deep ML fired rather than pinned: the level ML reached is
/// the budget for how far one contrary opinion may move it. A file that fired at
/// the loosest hostile rung barely survived the boundary, so a clear pushes it
/// most of the way across the suspicious band; a file that fired near the
/// strictest rung is moved barely past it.
///
/// Geometric, for the same reason as [`interpreted_level`] — the axis is. At the
/// shipped `-l 25` an ML `L1` lands at `L31`, an `L12` at `L248`.
///
/// `L0` never reaches here: [`Evidence::may_cross`] refuses the crossing outright.
#[allow(clippy::cast_possible_truncation)] // bounded by `ceiling` (<= grid_max)
fn softened_level(ml_level: Option<i32>, active_level: Option<u16>, grid_max: u16) -> Option<i32> {
    let active = active_level?;
    let ceiling = i32::from(crate::model::capped_suspicious_level(grid_max));
    let floor = i32::from(active).saturating_add(1);
    if floor >= ceiling {
        return Some(ceiling);
    }
    // No level to scale by (manual-threshold mode) falls back to the midpoint.
    let fraction = match ml_level {
        Some(lvl) if lvl > 0 && active > 0 => (f64::from(lvl) / f64::from(active)).clamp(0.0, 1.0),
        _ => 0.5,
    };
    let placed = f64::from(floor) * (f64::from(ceiling) / f64::from(floor)).powf(fraction);
    Some((placed.round() as i32).clamp(floor, ceiling))
}

#[allow(clippy::cast_possible_truncation)] // bounded by `ceiling` (<= grid_max)
fn interpreted_level(
    active_level: Option<u16>,
    grid_max: u16,
    outcome: Classification,
    corroborated: bool,
) -> Option<i32> {
    match outcome {
        Classification::Benign => Some(-1),
        Classification::Hostile => active_level.map(i32::from),
        Classification::Suspicious => active_level.map(|active| {
            let ceiling = i32::from(crate::model::capped_suspicious_level(grid_max));
            let floor = i32::from(active).saturating_add(1);
            if floor >= ceiling {
                return ceiling;
            }
            // Placed *within* the band rather than at its weakest rung.
            //
            // Pinning every interpreted-suspicious verdict to the ceiling put all
            // of them on one number, which threw away the ordering the level axis
            // exists to carry: a sample cleave independently flagged and a sample
            // resting on the LLM's word alone both read as 3000.
            //
            // Geometric, because the axis is: `level_confidence` compresses
            // 200→82, 1000→75, 2000→66, 5000→54, so a *linear* midpoint of
            // 26..3000 sits at 1513 and reads as barely-suspicious. The geometric
            // one lands near 279, which is where the middle of the band actually
            // is in confidence terms.
            //
            // Corroboration moves it a quarter of the way in instead of half —
            // nearer the hostile boundary, because two detectors agreeing is a
            // stronger claim than one.
            let fraction = if corroborated { 0.25 } else { 0.5 };
            let placed = f64::from(floor) * (f64::from(ceiling) / f64::from(floor)).powf(fraction);
            (placed.round() as i32).clamp(floor, ceiling)
        }),
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
    pub cleave: Option<cleave::types::CompactReport>,
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
    /// Human result card for terminal output, or cleave's annotated context for
    /// the tiny/interpret formats. Built while the typed report is still in scope.
    pub rendered_context: String,
    /// Optional LLM interpretation blended with the ML verdict (`--interpret`).
    /// Serialized as the response `llm` section; `None` when interpretation was
    /// disabled or gated out.
    pub interpretation: Option<crate::interpret::Interpretation>,
    /// A second opinion still to run; see [`PendingLlm`]. Never serialized.
    pub pending_llm: Option<PendingLlm>,
    /// Whether cleave replayed this analysis from its on-disk cache instead of
    /// running the pipeline. Not serialized — it describes how this run reached
    /// the verdict, not the verdict — but it is what tells an operator whether a
    /// fast response was cached work or a fast file.
    pub analysis_cached: bool,
    /// Fetched dependencies to mirror into hopper as their own samples. Empty
    /// unless the scan fetched dependencies; consumed by the upload paths and
    /// never serialized into this result's own envelope.
    pub dependency_results: Vec<DepResult>,
    /// The bloom status flag for a known-bad/conflicted file: drives the inline
    /// 🚩/🏴 mark in the terminal header and the `bloom=` token on the `--format
    /// tiny` line. `None` for unremarkable files; a terminal-UI concern only, so
    /// it is never serialized into the JSON envelope.
    pub bloom_mark: Option<crate::output::BloomMark>,
    /// Where this result's verdict should land on hopper, when that differs
    /// from the ordinary "post under `sha256`" rule. Not serialized — like
    /// [`analysis_cached`](Self::analysis_cached), it describes how this
    /// result reached hopper, not the verdict itself. `Normal` for every
    /// ordinary analysis; only `classify_purl`'s registry-metadata fallback
    /// (real artifact bytes unfetchable) sets the other variants, because
    /// that fallback's own content — the registry's JSON record, not a real
    /// artifact — hashes differently on every fetch and would otherwise mint
    /// hopper a fresh, never-deduplicating row each time it fires.
    pub hopper_route: HopperRoute,
}

/// See [`ScanResult::hopper_route`].
#[derive(Debug, Clone, Default)]
pub enum HopperRoute {
    /// Post the verdict under this result's own `sha256`, as always.
    #[default]
    Normal,
    /// Post the verdict under this sha256 instead of the result's own —
    /// hopper already holds real content for the requested coordinate under
    /// a different, stable sha256, and the registry-metadata verdict backs
    /// onto that row rather than minting a new one.
    Redirect(String),
    /// Post nothing to hopper for this result. Hopper has never seen real
    /// content for the requested coordinate, so there is nothing to attach
    /// a verdict to that would not just be more of the same churn.
    Suppress,
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
    /// all, while the members of a directly-scanned package got theirs.
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

/// The findings of a report's root file — the ones that describe the artifact
/// scan was asked about, as opposed to its members.
fn root_findings(report: &cleave::types::CompactReport) -> &[cleave::types::CompactTrait] {
    report.files.first().map_or(&[], |f| &f.findings)
}

impl From<&cleave::types::CompactTrait> for TopFinding {
    fn from(f: &cleave::types::CompactTrait) -> Self {
        Self {
            id: f.id.clone(),
            crit: u32::from(f.criticality),
            conf: f.confidence,
            desc: f.description.clone(),
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
    external_dependencies: AtomicU32,
    external_urls: AtomicU32,
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
/// chiefly actionable fetch failures/skips — can print *above* the bar instead
/// of grafting onto or racing its `\r`-parked line. Only the single-process
/// terminal CLI ever installs a bar;
/// server/JSON modes run none and leave this `None`. A `Weak` so a finished bar
/// is never kept alive; [`print_above_bar`] upgrades it per call.
static ACTIVE_BAR: Mutex<Option<Weak<Inner>>> = Mutex::new(None);

/// Whether an interactive scan progress bar currently owns the terminal. The
/// live dependency tree ([`crate::deptree`]) defers to it: a multi-file scan's
/// bar owns external-fetch status, so only a single-artifact scan — where no bar
/// is live — takes over stderr with an in-place tree.
pub(crate) fn bar_active() -> bool {
    ACTIVE_BAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .and_then(Weak::upgrade)
        .is_some()
}

/// Add one file's active external-reference work to the main scan bar. The
/// counters are process-wide only while that bar is alive; concurrent files
/// contribute to the same compact status note.
pub(crate) fn external_fetch_started(dependencies: usize, urls: usize) {
    let bar = ACTIVE_BAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .and_then(Weak::upgrade);
    let Some(inner) = bar else { return };
    let dependencies = u32::try_from(dependencies).unwrap_or(u32::MAX);
    let urls = u32::try_from(urls).unwrap_or(u32::MAX);
    inner
        .external_dependencies
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            Some(n.saturating_add(dependencies))
        })
        .ok();
    inner
        .external_urls
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            Some(n.saturating_add(urls))
        })
        .ok();
}

/// Remove one file's external-reference work from the main scan bar.
pub(crate) fn external_fetch_finished(dependencies: usize, urls: usize) {
    let bar = ACTIVE_BAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .and_then(Weak::upgrade);
    let Some(inner) = bar else { return };
    let dependencies = u32::try_from(dependencies).unwrap_or(u32::MAX);
    let urls = u32::try_from(urls).unwrap_or(u32::MAX);
    inner
        .external_dependencies
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            Some(n.saturating_sub(dependencies))
        })
        .ok();
    inner
        .external_urls
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            Some(n.saturating_sub(urls))
        })
        .ok();
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
        let external_dependencies = self.external_dependencies.load(Ordering::Relaxed);
        let external_urls = self.external_urls.load(Ordering::Relaxed);
        let elapsed = self.start.elapsed();

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

        let stats = progress_stats(done, self.total, elapsed);

        // Long-tail reassurance: when only a few files remain and the count has
        // not moved for a while, the scan is almost certainly deep in a slow
        // reverse-engineering pass, not hung. Appended to the bar line (not a
        // separate line) so the bar's own clear erases it — the note shows while
        // the tail is stalled and vanishes the instant a file completes or the
        // scan ends. Capped to the terminal width so it can never wrap.
        let left = self.total.saturating_sub(done);
        let stalled_ms = elapsed
            .as_millis()
            .saturating_sub(u128::from(self.last_advance_ms.load(Ordering::Relaxed)));
        let note_text = external_fetch_note(external_dependencies, external_urls).or_else(|| {
            ((1..=TAIL_FILES).contains(&left) && stalled_ms >= TAIL_STALL_MS)
                .then(|| TAIL_MESSAGE.to_string())
        });
        let used = 25 + stats.chars().count();
        let note = note_text.map_or_else(String::new, |text| {
            fit_notice(&text, self.term_cols.saturating_sub(used + 1))
        });

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

/// Stable right-hand progress text. There is no rate before the first file
/// completes, so keep that otherwise-nonsensical initial ETA qualitative.
fn progress_stats(done: u32, total: u32, elapsed: Duration) -> String {
    if done == 0 {
        return format!("{done}/{total}  Estimating\u{2026}");
    }

    let rate = f64::from(done) / elapsed.as_secs_f64().max(0.001);
    let eta = f64::from(total.saturating_sub(done)) / rate.max(0.001);
    format!("{done}/{total}  {rate:.0}/s  {}", format_eta(eta))
}

/// Compact status text for external work currently shared by the scan's
/// concurrent workers. Keep it short enough to coexist with the file counter.
fn external_fetch_note(dependencies: u32, urls: u32) -> Option<String> {
    fn noun(count: u32, singular: &str, plural: &str) -> String {
        format!("{count} {}", if count == 1 { singular } else { plural })
    }

    match (dependencies, urls) {
        (0, 0) => None,
        (0, urls) => Some(format!("fetching {}", noun(urls, "URL", "URLs"))),
        (dependencies, 0) => Some(format!(
            "fetching {}",
            noun(dependencies, "dependency", "dependencies")
        )),
        (dependencies, urls) => Some(format!(
            "fetching {} · {}",
            noun(dependencies, "dependency", "dependencies"),
            noun(urls, "URL", "URLs")
        )),
    }
}

/// A single-artifact scan has no file-count denominator to fill a progress bar,
/// but the analysis of one archive can still take many seconds (extraction,
/// disassembly, per-member scoring) with nothing on screen. This is a minimal
/// animated spinner for that gap — `⠙ scanning demo.zip · 12s` redrawn in place
/// on stderr — so the scan visibly *works* rather than appearing to hang. It is
/// deliberately not a [`Progress`] bar and never registers in `ACTIVE_BAR`, so
/// the live dependency tree still takes over stderr during the fetch phase.
pub(crate) struct Spinner {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Spinner {
    /// Start spinning next to `label`, or `None` when stderr isn't a terminal
    /// (piped/redirected output draws nothing, matching the progress bar).
    pub(crate) fn start(label: String) -> Option<Self> {
        use std::io::IsTerminal as _;
        if !std::io::stderr().is_terminal() {
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let start = Instant::now();
        let handle = std::thread::Builder::new()
            .name("scan-spinner".into())
            .spawn(move || {
                let mut tick = 0usize;
                // The distinct archive members seen entering analysis. cleave
                // fans members across the rayon pool and names each on a
                // per-thread breadcrumb; sampling those each tick lets us show a
                // live count and the member currently in hand — real progress
                // through a deeply nested archive, where one file expands to
                // thousands of members with nothing else to count. It undercounts
                // members that begin and finish between two samples, so the count
                // is prefixed `~`.
                let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                let mut current = String::new();
                while !flag.load(Ordering::Relaxed) {
                    let frame = SPINNER[tick % SPINNER.len()];
                    let secs = start.elapsed().as_secs();
                    // One sample of the per-thread breadcrumbs: grow the distinct
                    // member set (the live count) and show the oldest in-flight
                    // member (most likely the slow one holding up the scan).
                    let members = cleave::breadcrumb::snapshot();
                    for crumb in &members {
                        if crumb.analyzer == "member" {
                            seen.insert(crumb.target.clone());
                        }
                    }
                    if let Some(oldest) = members.iter().find(|c| c.analyzer == "member") {
                        current = oldest.target.clone();
                    }
                    let detail = if seen.is_empty() {
                        String::new()
                    } else {
                        let member = spinner_tail(&current, 48);
                        format!(
                            "  \x1b[38;2;120;120;120m~{} members\x1b[0m  \x1b[38;2;80;80;80m{member}\x1b[0m",
                            seen.len()
                        )
                    };
                    eprint!(
                        "\r\x1b[2K \x1b[38;2;100;180;255m{frame}\x1b[0m  \x1b[38;2;160;160;160mscanning {label}\x1b[0m{detail}  \x1b[38;2;80;80;80m{secs}s\x1b[0m"
                    );
                    let _ = std::io::stderr().flush();
                    tick += 1;
                    std::thread::sleep(PROGRESS_TICK);
                }
            })
            .ok()?;
        Some(Self {
            stop,
            handle: Some(handle),
        })
    }
}

/// Keep the last `width` characters of a member path — its filename and nearest
/// directories, the telling part — eliding the head with a leading `…`. A short
/// path is returned unchanged.
fn spinner_tail(path: &str, width: usize) -> String {
    let count = path.chars().count();
    if count <= width {
        return path.to_string();
    }
    let tail: String = path.chars().skip(count - width + 1).collect();
    format!("\u{2026}{tail}")
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        // Erase the spinner line so the next output starts clean.
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
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
            external_dependencies: AtomicU32::new(0),
            external_urls: AtomicU32::new(0),
            total,
            start: Instant::now(),
            last_advance_ms: AtomicU64::new(0),
            tick: AtomicU32::new(0),
            term_cols: term_cols(),
            stopped: AtomicBool::new(false),
            draw_lock: Mutex::new(()),
        });
        // Publish this bar so fetch status can join it and actionable failures
        // can print above it (see `print_above_bar`). Overwrites any prior
        // registration — the terminal
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

#[cfg(test)]
mod progress_render_tests {
    use super::*;

    #[test]
    fn eta_waits_for_a_real_sample() {
        assert_eq!(
            progress_stats(0, 100, Duration::from_secs(30)),
            "0/100  Estimating…",
            "elapsed time alone cannot produce a rate"
        );
    }

    #[test]
    fn eta_appears_after_the_first_completed_file() {
        let stats = progress_stats(1, 100, Duration::from_millis(500));
        assert!(!stats.contains("Estimating"));
        assert!(stats.contains("/s"));
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
        // A sighting is the same shape of answer as known-bad — a flag that
        // runs the full analysis — but says somebody else reported it rather
        // than that we measured it. The distinction rides the inline mark.
        BloomDecision::KnownBad
        | BloomDecision::SightedHostile
        | BloomDecision::SightedSuspicious => None,
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

/// The analysis options for a target the operator named by path, which is the
/// shared options with the bloom skip predicate removed.
///
/// A named target is always analyzed on its own merits, so cleave must not
/// short-circuit it into a minimal report. The gate in [`record_file_result`]
/// makes the same call for the same reason; both are needed, because they act at
/// different points — the predicate decides whether analysis runs at all, the
/// gate decides whether the result is counted without an ML pass.
fn named_target_opts(opts: &cleave::AnalysisOptions) -> cleave::AnalysisOptions {
    cleave::AnalysisOptions {
        skip_predicate: None,
        ..opts.clone()
    }
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
            let Some(digest) = burton::parse_sha256_hex(sha_hex) else {
                return false;
            };
            let decision = lookup.decide(&burton::Artifact::sha256(&digest));
            // Known-good is trusted and skipped, unless the file was created,
            // status-changed, or modified within the last 48h — a fresh
            // known-good is analyzed on its own merits (recent activity, and a
            // guard against a bloom false positive on a freshly planted file),
            // so cleave analyzes it once here rather than skipping and then
            // re-analyzing in `record_file_result`.
            //
            // Every other decision is analyzed, adverse or not: in fast mode an
            // unknown is skipped as well, which is what fast mode means.
            if decision.may_skip() {
                !file_touched_within(path, KNOWN_GOOD_RESCAN_SECS, SystemTime::now())
            } else {
                fast && decision == BloomDecision::Unknown
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

/// The two distinct detection inventories shown in the banner. Bloom entries
/// are SHA-256/PURL signatures; traits, composites, and YARA are actual rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DetectionCounts {
    pub(crate) hashes_and_purls: u64,
    pub(crate) rules: u64,
}

/// Detection inventory after resource load. `cleave::version_info()` loads (or
/// reuses) shared resources; `hashes_and_purls` comes from the already-loaded
/// Bloom repository.
pub(crate) fn detection_counts_from(hashes_and_purls: u64) -> DetectionCounts {
    let info = cleave::version_info();
    DetectionCounts {
        hashes_and_purls,
        rules: info.trait_count as u64 + info.composite_count as u64 + info.yara_rules as u64,
    }
}

/// Detection inventory already resident in memory for the scan banner. Shared
/// by [`run`], [`run_paths`], and `ps` so every scanner reports the same counts.
pub(crate) fn detection_counts(config: &ScanConfig) -> DetectionCounts {
    let hashes_and_purls = config
        .bloom()
        .map_or(0, crate::bloom_repo::Lookup::rule_count);
    detection_counts_from(hashes_and_purls)
}

/// Run a scan against a file or directory tree.
///
/// A file path is analyzed directly. A directory path is walked once by
/// `discover_files` to learn the file count upfront (for the progress bar and
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
        crate::engine::print_above_bar(|| eprintln!("\nInterrupted — finishing current file…"));
        ctrlc_flag.store(true, Ordering::Relaxed);
    });
    // scan consumes only the compact projection of member nodes; let cleave
    // drop fold-time fields that exist solely for the full v3 schema.
    cleave::set_compact_member_retention(true);
    let mut cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        cancellation: Some(Arc::clone(&cancellation)),
        skip_predicate: bloom_skip_predicate(config),
        ..Default::default()
    };
    add_zip_passwords(&mut cleave_opts, config.zip_passwords());
    let is_terminal = matches!(config.format(), OutputFormat::Terminal);
    let scan_start = Instant::now();

    // Single-file path: handle directly without the directory streaming API.
    if path.is_file() {
        let tally = Tally::default();
        let stdout = Mutex::new(std::io::stdout());
        // One artifact has no file-count bar, but its static analysis
        // (extraction, disassembly) is the silent long pole — spin a marker so
        // the scan visibly works. Dropped before `record_file_result` so its
        // fetch tree and the final render own the terminal cleanly.
        let spinner = is_terminal.then(|| {
            let label = path.file_name().map_or_else(
                || path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            Spinner::start(label)
        });
        let cleave_result = cleave::analyze_file(path, &named_target_opts(&cleave_opts))
            .with_context(|| format!("cleave analysis of {}", path.display()));
        drop(spinner);
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
            true,
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
        crate::output::print_banner(detection_counts(config));
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
                false,
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
    // `Some(purl)` when `bytes` is a registry-metadata document standing in
    // for a package that couldn't be fetched (`pkg.rs`'s two fallback
    // branches) — never a real artifact. Its bytes hash differently on every
    // fetch, so `record_file_result` must not post this run's own sha256 to
    // hopper; see `HopperRoute`'s doc comment for the full story (the
    // server-side twin of this fix).
    registry_fallback_purl: Option<&str>,
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
    cleave::set_compact_member_retention(true); // compact projection only
    let mut cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        skip_predicate: bloom_skip_predicate(config),
        ..Default::default()
    };
    add_zip_passwords(&mut cleave_opts, config.zip_passwords());
    let is_terminal = matches!(config.format(), OutputFormat::Terminal);
    let scan_start = Instant::now();
    let tally = Tally::default();
    let stdout = Mutex::new(std::io::stdout());

    let report = cleave::analyze_bytes_owned(bytes, name, &cleave_opts)
        .with_context(|| format!("cleave analysis of {label}"));
    // `--hopper` on `url`/`purl`: a fetched artifact is the never-before-seen
    // case, so the uploader matters more here than on a local path scan —
    // `record_file_result` routes through `upload_scan_result`, which offers the
    // bytes+sidecar over `/api/known` before the verdict POST, and hopper drops
    // a result for a SHA it never ingested. Dropped at the end of the function,
    // which flushes and joins the background thread before we return.
    let uploader = config
        .hopper()
        .map(|url| crate::upload::Uploader::new(url, crate::upload::default_worker_name()));
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
        uploader.as_ref(),
        root_registry,
        root_fetch,
        // The url/purl the operator named — the fetched artifact itself, not a
        // dependency of it.
        true,
        registry_fallback_purl,
    );
    drop(uploader);

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
/// skipped.
///
/// Deliberately its own value rather than the dependency window in
/// [`crate::fetch`]. The two answer different questions: that one asks how long
/// a *registry release* stays too new to trust a vouch for, and is sized in
/// hours because a published version is immutable; this one asks how recently a
/// *local file* was written, and is the guard against a bloom false-positive
/// shielding something a live intrusion just planted. Shrinking it to match the
/// dependency window would narrow that guard for no reason connected to it.
const KNOWN_GOOD_RESCAN_SECS: u64 = 48 * 3_600;

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
        if ctime >= 0 && recent(std::time::UNIX_EPOCH + Duration::from_secs(ctime.unsigned_abs())) {
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
    named: bool,
    registry_fallback_purl: Option<&str>,
) {
    // Bloom verdict, re-derived from the root sha cleave computed. A file cleave
    // skipped at our request (a stale known-good, or fast-mode unknown) arrives as
    // a minimal report and is short-circuited here; a known-bad/conflicted file —
    // or a fresh known-good scanned on its own merits — was analyzed and carries a
    // provenance marker on its SHA-256 line, derived from the same decision.
    //
    // `named` marks a target the operator asked for by path, which is never
    // answered from a bless — see the `scan_anyway` comment below.
    let mut bloom_mark = None;
    // Resolve the bloom decision from the sha cleave computed.
    let decision = if let Some(lookup) = config.bloom()
        && let Ok(report) = &cleave_result
        && let Some(digest) = burton::parse_sha256_hex(&report.target.sha256)
    {
        Some(lookup.decide_sha256(&digest))
    } else {
        None
    };
    if let Some(decision) = decision {
        // Two known-good files are still scanned on their own merits, and in both
        // cases the skip predicate already declined to skip, so cleave has
        // produced a full report — fall through to the normal scan and mark it
        // ✓ known-good. A stale known-good (and, in fast mode, an unknown)
        // arrived as a minimal report and is counted here without an ML pass.
        //
        //   - a file created/changed/modified within KNOWN_GOOD_RESCAN_SECS, and
        //   - a file the operator named on the command line.
        //
        // The second is a policy, not an optimization: a bless is a bulk
        // shortcut for dependencies and directory walks, and returning one where
        // a scan was asked for answers a different question than the one put to
        // us. The mark is still emitted, so a named known-bad still says so.
        let scan_anyway = decision == BloomDecision::Skip
            && (named || file_touched_within(file_path, KNOWN_GOOD_RESCAN_SECS, SystemTime::now()));
        if !scan_anyway
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
            // An adversely bloom-marked file is always surfaced — its flag
            // replaces the old unconditional banner, so the benign filter must
            // not swallow a known-bad file the model happens to rate benign. A
            // fresh known-good that was rescanned and remained benign still
            // obeys the ordinary display filter.
            // JSON, tiny, and interpret are machine/LLM payload formats: they
            // emit every scanned file — `--show` gates only the terminal view.
            if matches!(
                config.format(),
                OutputFormat::Json | OutputFormat::Tiny | OutputFormat::Interpret
            ) || r
                .bloom_mark
                .is_some_and(crate::output::BloomMark::forces_terminal_display)
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
                if let Some(purl) = registry_fallback_purl {
                    // `file_path`'s bytes are the registry's own JSON record
                    // standing in for a package that couldn't be fetched — not
                    // a real artifact, and it hashes differently on every
                    // fetch (see the fallback's own doc comment in `pkg.rs`).
                    // Posting `r.sha256` here would mint hopper an unbounded
                    // stream of never-deduplicating rows, exactly the bug
                    // fixed server-side in `classify_purl`/`HopperRoute`. This
                    // is that fix's CLI twin: redirect onto real content
                    // hopper already holds for the purl, or post nothing.
                    let client = reqwest::blocking::Client::new();
                    if let Some(hopper_url) = config.hopper()
                        && let Some(real_sha) =
                            crate::upload::known_sha_for_purl(&client, hopper_url, purl)
                    {
                        let now = now_rfc3339();
                        let collector = format!("scan+{}", crate::upload::default_worker_name());
                        let filename = file_path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("file")
                            .to_string();
                        let sidecar = match root_registry {
                            Some(provenance) => crate::provenance::build_sidecar_from_provenance(
                                &filename, &real_sha, 0, &collector, &now, "", purl, provenance,
                            ),
                            None => crate::provenance::build_sidecar(
                                &filename,
                                &real_sha,
                                0,
                                &collector,
                                &now,
                                "",
                                purl,
                                None,
                                &[],
                            ),
                        };
                        uploader.submit_artifacts(vec![crate::upload::UploadArtifact {
                            sha256: real_sha,
                            size: 0,
                            filename,
                            bytes: crate::upload::ArtifactBytes::File(std::path::PathBuf::new()),
                            sidecar,
                            backfill: true,
                        }]);
                    }
                } else {
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
                        root_fetch,
                        deps,
                        envelope,
                    );
                }
            }
        }
        Err(e) => {
            if cancellation.is_some_and(|c| c.load(Ordering::Relaxed)) {
                // Ctrl-C makes in-flight cleave work return cancellation errors
                // on every worker. Cancellation is expected control flow, not a
                // failed file; keep it out of both the log and the error tally.
                return;
            }
            let msg = crate::tools::enrich_error(&e).unwrap_or_else(|| format!("{e:#}"));
            tracing::error!("error analyzing {}: {}", file_path.display(), msg);
            // A failed file still gets a line on stdout under `--format json`, so a
            // consumer reading the NDJSON sees *that* it failed and why, instead of
            // inferring it from a record that never arrives. This does not make the
            // failure anything other than a failure: the tally below still counts it,
            // the process still exits 3, and nothing is handed to hopper — an error
            // envelope carries an empty file type, which hopper reads as a delete.
            //
            // `raw.files` is deliberately empty, and that is load-bearing for
            // compatibility. Readers that predate this record skip a fileless entry
            // and report the sample as having produced no result — which is exactly
            // right. A record carrying a file entry would instead be scored, and an
            // absent `ml.lvl` decodes to 0 in a JSON reader that defaults its
            // numbers, which reads as *hostile*. Keep the list empty.
            if matches!(config.format(), OutputFormat::Json) {
                emit_error_record(file_path, &msg, stdout);
            }
            tally.errors.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The `err` section of an error record: the file that failed, and why.
#[derive(Debug, serde::Serialize)]
struct ErrSection<'a> {
    path: &'a str,
    msg: &'a str,
    /// Scan engine build that produced this record, mirroring [`MlSectionRef::eng`]
    /// so an error line is attributable to a build like a successful one is.
    eng: &'static str,
}

/// An error line: `{"err": {...}, "raw": {"files": []}}`.
///
/// Shaped to sit in the same NDJSON stream as [`ScanResultEnvelopeRef`] without
/// being mistaken for one. See the call site for why `raw.files` must stay empty.
#[derive(Debug, serde::Serialize)]
struct ErrorRecord<'a> {
    err: ErrSection<'a>,
    raw: EmptyRaw,
}

#[derive(Debug, serde::Serialize)]
struct EmptyRaw {
    files: [(); 0],
}

fn emit_error_record(file_path: &Path, msg: &str, stdout: &Mutex<std::io::Stdout>) {
    let path = file_path.display().to_string();
    let record = ErrorRecord {
        err: ErrSection {
            path: &path,
            msg,
            eng: ENGINE_VERSION,
        },
        raw: EmptyRaw { files: [] },
    };
    let Ok(mut out) = stdout.lock() else {
        return;
    };
    if serde_json::to_writer(&mut *out, &record).is_err() {
        // Nothing further to do: the failure is already logged to stderr and
        // counted, and this line is the redundant copy.
        return;
    }
    let _ = out.write_all(b"\n");
}

/// Renew one scan result on hopper: ensure hopper has the scanned file and any
/// fetched dependency archives (with provenance), then renew the verdict. Used
/// by both the CLI `--hopper` path and the serve-mode `--hopper` upload. The
/// artifacts are queued before the result so a never-seen top-level file's row
/// exists before its verdict lands. Blocking (reads sidecars from disk); callers
/// on an async runtime must run it off the executor.
#[allow(clippy::too_many_arguments)] // a flat hand-off of one result's parts; a struct would only indirect it
pub(crate) fn upload_scan_result(
    uploader: &crate::upload::Uploader,
    file_path: &Path,
    sha256: String,
    size_bytes: u64,
    root_provenance: Option<&crate::provenance::RegistryProvenance>,
    root_fetch: Option<&fletch::fetch::FetchRecord>,
    dependency_results: Vec<DepResult>,
    envelope: ScanResultEnvelope,
) {
    static COLLECTOR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let artifacts = collect_upload_artifacts(
        file_path,
        &sha256,
        size_bytes,
        COLLECTOR.get_or_init(|| format!("scan+{}", crate::upload::default_worker_name())),
        root_provenance,
        root_fetch,
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
    // `scan purl` names a package outright and a registry URL still identifies
    // one; a local path or an arbitrary URL identifies nothing, and the
    // uploader names those by digest alone.
    uploader.submit(sha256, artifact_purl(root_fetch), envelope);
}

/// Feature-extract and model-score a report's embedded files — the archive
/// members at depth > 0 — returning one evaluation per entry.
///
/// Extracted so a fetched dependency can be graded on its own report rather
/// than deriving its standalone verdict from the parent report.
///
/// Per-member work is pure and runs in parallel: reports with thousands of
/// embedded files (nested npm tarballs, fetched dependency trees) previously ran
/// this serially on one rayon worker, and on member-heavy archives that pass —
/// not cleave's analysis — was the scan's wall-clock tail.
fn score_embedded_files(
    entries: &[&cleave::types::CompactFile],
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
                    let ef_parsed = crate::features::ParsedReport::from_compact_file(ef, needs);
                    let mut ef_features = ctx.extract_from_parsed(&ef_parsed);
                    model.spec().standardize(&mut ef_features);
                    let (mut ef_decision, ef_model_scores, ef_skipped_models) = model
                        .predict_for_file_detailed(&ef.file_type, &ef_features, &ef_parsed)
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
                        &ef.findings,
                        model.active_level(),
                        model.grid_max(),
                        if ef.path.is_empty() { label } else { &ef.path },
                    );
                    (ef_decision, ef_model_scores, ef_skipped_models)
                };

            let rel_path = ef
                .path
                .rsplit_once("!!")
                .map_or(ef.path.as_str(), |(_, r)| r)
                .to_string();
            let ef_top_findings: Vec<TopFinding> = ef
                .findings
                .iter()
                .filter(|ff| ff.criticality >= 4)
                .take(3)
                .map(TopFinding::from)
                .collect();
            EmbeddedFile {
                id: u64::from(ef.id),
                sha256: ef.sha.clone(),
                path: rel_path,
                file_type: ef.file_type.clone(),
                classification: ef_decision.class,
                probability: ef_decision.probability,
                threshold: ef_decision.threshold,
                level: ef_decision.level,
                model_scores: ef_model_scores,
                skipped_models: ef_skipped_models,
                formula: ef.formula.clone().unwrap_or_default(),
                top_findings: ef_top_findings,
            }
        })
        .collect()
}

/// Every analyzed embedded node receives an individual model verdict.
///
/// This deliberately has no count limit: selecting only an archive prefix makes
/// malware detection depend on member ordering and lets a hostile tail member,
/// fetched dependency, or registry security sidecar evade container elevation.
fn embedded_entries(
    report: &cleave::types::CompactReport,
) -> impl Iterator<Item = &cleave::types::CompactFile> {
    report.files.iter().filter(|file| file.depth > 0)
}

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
    let report: cleave::types::CompactReport = serde_json::from_str(raw).ok()?;
    let needs = ctx.raw_needs().union(crate::features::RawNeeds::all());

    // The container's own verdict. No own_shas filtering: this report is the one
    // cleave produced for the dependency's bytes alone, before anything was
    // grafted onto it, so every file in it is the dependency's own.
    let parsed = crate::features::ParsedReport::from_compact_report(&report, needs, None);
    let mut features = ctx.extract_from_parsed(&parsed);
    if features.len() != model.spec().total_features() {
        return None;
    }
    model.spec().standardize(&mut features);
    let file_type = report
        .files
        .first()
        .map_or("unknown", |f| f.file_type.as_str());
    let (mut decision, _, _) = model
        .predict_for_report_detailed(file_type, &features, &parsed)
        .ok()?;
    apply_trait_floor(
        &mut decision,
        root_findings(&report),
        model.active_level(),
        model.grid_max(),
        label,
    );

    // Every member elevates the dependency as it would in a first-hand scan.
    let entries: Vec<&cleave::types::CompactFile> = embedded_entries(&report).collect();
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
    // The report text is decoded only here, for the short-lived envelope being
    // POSTed — not for the job-long retention window.
    let raw: cleave::types::CompactReport = serde_json::from_str(&dep.raw).unwrap_or_default();
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

/// The scanned file itself, offered to hopper so a `--upload` run can store a
/// locally-analyzed file hopper has never seen. Just the one artifact — fetched
/// dependencies are mirrored separately (bytes, provenance, *and* verdict) by
/// [`crate::upload::Uploader::submit_dependencies`], so they never ride here.
/// The package locator an artifact was fetched as, when it was fetched by one.
///
/// A fetch record's locator is either a PURL or a plain URL, and only the
/// former is a package identity. Shared by the sidecar's package slot — which
/// hopper projects into its queryable `purl_base` column — and by the verdict
/// handed to the uploader, so the bytes and the verdict can never disagree
/// about what this artifact is.
fn fetched_purl(root_fetch: Option<&fletch::fetch::FetchRecord>) -> Option<&str> {
    root_fetch
        .map(|rec| rec.locator.as_str())
        .filter(|locator| locator.starts_with("pkg:"))
}

/// The package an artifact *is*, as far as anything here can tell.
///
/// Broader than [`fetched_purl`], which asks only whether the *request* named a
/// package. This also recovers the coordinate from a registry URL, because
/// `scan url https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz` is a
/// request about a package whether or not it was spelled as one — and hopper
/// should store it under that package either way, or the same artifact lands
/// twice depending on which spelling the operator reached for.
///
/// Feeds both the sidecar's package slot — which hopper projects into its
/// queryable `purl_base` column — and the identity the uploader logs.
fn artifact_purl(root_fetch: Option<&fletch::fetch::FetchRecord>) -> Option<String> {
    if let Some(purl) = fetched_purl(root_fetch) {
        return Some(purl.to_string());
    }
    fletch::purl::url_to_purl(&root_fetch?.locator)
}

pub(crate) fn collect_upload_artifacts(
    file_path: &Path,
    sha256: &str,
    size_bytes: u64,
    collector: &str,
    root_provenance: Option<&crate::provenance::RegistryProvenance>,
    root_fetch: Option<&fletch::fetch::FetchRecord>,
) -> Vec<crate::upload::UploadArtifact> {
    use crate::upload::{ArtifactBytes, UploadArtifact};
    let now = now_rfc3339();

    // For a fetched root (`scan url|purl`), `file_path` is the display locator,
    // not a readable file: it takes its name from the fetch URL, its bytes from
    // fletch's blob cache (where the fetch stored them), that URL for the
    // sidecar's fetch slot, and — when the locator is a PURL — the package slot
    // hopper projects into its queryable purl_base column. A local `scan path`
    // root reads from disk and claims none of the rest.
    let purl = artifact_purl(root_fetch).unwrap_or_default();
    let purl = purl.as_str();
    let (root_name, bytes, url) = match root_fetch {
        Some(rec) => {
            let url = rec.final_url.as_deref().unwrap_or(&rec.resolved_url);
            (
                artifact_filename(url, &rec.locator),
                ArtifactBytes::Cached {
                    locator: rec.locator.clone(),
                },
                url,
            )
        }
        None => (
            file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("file")
                .to_string(),
            ArtifactBytes::File(file_path.to_path_buf()),
            "",
        ),
    };
    let sidecar = if let Some(provenance) = root_provenance {
        crate::provenance::build_sidecar_from_provenance(
            &root_name, sha256, size_bytes, collector, &now, url, purl, provenance,
        )
    } else {
        crate::provenance::build_sidecar(
            &root_name,
            sha256,
            size_bytes,
            collector,
            &now,
            url,
            purl,
            None,
            &[],
        )
    };
    vec![UploadArtifact {
        sha256: sha256.to_string(),
        size: size_bytes,
        filename: root_name,
        bytes,
        sidecar,
        // Registry data or a PURL identity is worth backfilling onto a sample
        // hopper already has; a plain local file's thin sidecar is not.
        backfill: root_provenance.is_some() || !purl.is_empty(),
    }]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod upload_artifact_tests {
    use super::collect_upload_artifacts;
    use crate::upload::ArtifactBytes;
    use std::path::Path;

    fn fetch_record(locator: &str, resolved: &str) -> fletch::fetch::FetchRecord {
        serde_json::from_value(serde_json::json!({
            "locator": locator,
            "resolved_url": resolved,
            "fetched_at": 0,
            "cached": false,
            "outcome": "ok",
        }))
        .expect("minimal FetchRecord")
    }

    #[test]
    fn fetched_purl_root_carries_package_identity() {
        let rec = fetch_record(
            "pkg:npm/lodash@4.17.21",
            "https://registry.npmjs.org/lodash/-/lodash-4.17.21.tgz",
        );
        let arts = collect_upload_artifacts(
            Path::new("pkg:npm/lodash@4.17.21"),
            "aa11",
            42,
            "scan+test",
            None,
            Some(&rec),
        );
        let art = &arts[0];
        assert!(art.backfill, "a PURL identity is worth backfilling");
        assert_eq!(
            art.filename, "lodash-4.17.21.tgz",
            "named from the fetch URL"
        );
        assert!(
            matches!(&art.bytes, ArtifactBytes::Cached { locator } if locator == "pkg:npm/lodash@4.17.21"),
            "fetched root loads from the blob cache, not the display label"
        );
        let sidecar: serde_json::Value = serde_json::from_slice(&art.sidecar).unwrap();
        assert_eq!(sidecar["package"]["purl"], "pkg:npm/lodash@4.17.21");
        assert_eq!(
            sidecar["fetch"]["url"],
            "https://registry.npmjs.org/lodash/-/lodash-4.17.21.tgz"
        );
        assert_eq!(sidecar["artifact"]["filename"], "lodash-4.17.21.tgz");
    }

    #[test]
    fn fetched_url_root_has_no_package_slot() {
        let rec = fetch_record(
            "https://example.com/tool.zip",
            "https://example.com/tool.zip",
        );
        let arts = collect_upload_artifacts(
            Path::new("https://example.com/tool.zip"),
            "bb22",
            7,
            "scan+test",
            None,
            Some(&rec),
        );
        let art = &arts[0];
        assert!(
            !art.backfill,
            "a bare URL fetch carries no registry identity"
        );
        assert_eq!(art.filename, "tool.zip");
        let sidecar: serde_json::Value = serde_json::from_slice(&art.sidecar).unwrap();
        assert!(
            sidecar.get("package").is_none(),
            "no PURL, no package claim"
        );
        assert_eq!(sidecar["fetch"]["url"], "https://example.com/tool.zip");
    }

    #[test]
    fn hostile_redirect_targets_cannot_shape_the_stored_filename() {
        use super::{MAX_ARTIFACT_FILENAME, artifact_filename};

        // A redirect picks the final URL, so every one of these is reachable by
        // an attacker who controls (or compromises) the server we fetch from.
        for (url, expect) in [
            // Encoded separators a consumer might decode back into a path.
            (
                "https://e.com/..%2f..%2fetc%2fpasswd",
                "_2f.._2fetc_2fpasswd",
            ),
            // Windows separators in the final segment.
            ("https://e.com/a\\..\\..\\system32\\x.dll", "x.dll"),
            // Bare path components.
            ("https://e.com/..", "artifact"),
            ("https://e.com/.", "artifact"),
            // A right-to-left override disguising the real extension.
            ("https://e.com/invoice\u{202e}gpj.exe", "invoice_gpj.exe"),
            // Control characters that would forge a log line.
            ("https://e.com/a\nb\rc", "a_b_c"),
            // Leading dash reads as a flag; leading dot hides the file.
            ("https://e.com/-rf", "rf"),
            ("https://e.com/.ssh", "ssh"),
            // Ordinary names are untouched.
            ("https://e.com/lodash-4.17.21.tgz", "lodash-4.17.21.tgz"),
            ("https://e.com/x_1+2~3.tar.gz", "x_1+2~3.tar.gz"),
        ] {
            assert_eq!(artifact_filename(url, "pkg:npm/x@1"), expect, "url {url:?}");
        }

        // Length is bounded regardless of what the server sends.
        let long = format!("https://e.com/{}", "a".repeat(4096));
        assert_eq!(artifact_filename(&long, "").len(), MAX_ARTIFACT_FILENAME);

        // A hostile locator gets the same treatment when no URL resolved.
        assert_eq!(artifact_filename("", "pkg:npm/../../etc@1"), "etc-1");
    }

    #[test]
    fn local_root_keeps_thin_sidecar_and_disk_bytes() {
        let arts = collect_upload_artifacts(
            Path::new("/tmp/sample.exe"),
            "cc33",
            1,
            "scan+test",
            None,
            None,
        );
        let art = &arts[0];
        assert!(!art.backfill);
        assert_eq!(art.filename, "sample.exe");
        assert!(matches!(&art.bytes, ArtifactBytes::File(p) if p == Path::new("/tmp/sample.exe")));
        let sidecar: serde_json::Value = serde_json::from_slice(&art.sidecar).unwrap();
        assert!(sidecar.get("package").is_none());
        assert_eq!(sidecar["fetch"]["url"], "");
    }
}

/// Longest stored filename we will emit. Well under the 255-byte cap every
/// mainstream filesystem imposes, leaving a consumer room for its own prefix.
const MAX_ARTIFACT_FILENAME: usize = 128;

/// A filename for an uploaded artifact: the last path segment of the fetch URL
/// (query/fragment stripped), falling back to the locator's tail with PURL
/// punctuation flattened. hopper uses it for the stored filename and type sniff.
///
/// Both inputs are hostile. A redirect chooses `url`'s final segment, and a
/// locator can come from references discovered inside the sample, so the result
/// is [sanitized](sanitize_artifact_filename) before it leaves this process.
pub(crate) fn artifact_filename(url: &str, locator: &str) -> String {
    // Split on the Windows separator too: a consumer that resolves `a\..\..\b`
    // as a path must not be handed one.
    let from_url = url
        .rsplit(['/', '\\'])
        .next()
        .map(|seg| seg.split(['?', '#']).next().unwrap_or(seg))
        .filter(|seg| !seg.is_empty());
    let raw = match from_url {
        Some(name) => std::borrow::Cow::Borrowed(name),
        None => std::borrow::Cow::Owned(
            locator
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(locator)
                .replace(['@', ':'], "-"),
        ),
    };
    sanitize_artifact_filename(&raw)
}

/// Reduce an untrusted path segment to a name that is inert everywhere it
/// lands: hopper's stored filename, a `Content-Disposition` field, a log line,
/// and an analyst's screen.
///
/// An allowlist rather than a denylist, because the interesting attacks are the
/// ones we would forget to enumerate: `%2f` that a consumer later decodes into
/// a separator, a `U+202E` right-to-left override that makes a `.exe` render as
/// a `.png` to the analyst reading the verdict, a control character that forges
/// a log line, a 64 KiB segment. Everything outside `[A-Za-z0-9.-_+~]` → `_`.
///
/// Leading dots and dashes go too: they produce hidden files, the `.`/`..` path
/// components, and flag-like arguments. Registry filenames are ASCII in
/// practice, so this is lossless for real packages.
fn sanitize_artifact_filename(raw: &str) -> String {
    let safe: String = raw
        .chars()
        .take(MAX_ARTIFACT_FILENAME)
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '-' | '_' | '+' | '~' => c,
            _ => '_',
        })
        .collect();
    match safe.trim_start_matches(['.', '-']) {
        // Nothing survived: a segment of only dots/dashes, or empty to begin
        // with. Name it rather than emit "" for a consumer to interpret.
        "" => "artifact".to_string(),
        trimmed => trimmed.to_string(),
    }
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
/// recursive walker. Both feed a single shared `Tally`, so the returned
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
        crate::engine::print_above_bar(|| eprintln!("\nInterrupted — finishing current file…"));
        ctrlc_flag.store(true, Ordering::Relaxed);
    });

    cleave::set_compact_member_retention(true); // compact projection only
    let mut cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        cancellation: Some(Arc::clone(&cancellation)),
        skip_predicate: bloom_skip_predicate(config),
        ..Default::default()
    };
    add_zip_passwords(&mut cleave_opts, config.zip_passwords());
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
        crate::output::print_banner(detection_counts(config));
        Progress::new(total)
    });

    // `--hopper`: renew every result on hopper as it completes. The uploader
    // owns a background thread; dropping it (below, after the scan closures
    // release their borrow) flushes and joins in-flight uploads.
    let uploader = config
        .hopper()
        .map(|url| crate::upload::Uploader::new(url, crate::upload::default_worker_name()));

    {
        let record = |file_path: &Path, result: Result<cleave::AnalysisReport>, named: bool| {
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
                named,
                None,
            );
        };

        // A single artifact has no file-count bar (total == 1), so its static
        // analysis — extraction, disassembly, per-member scoring — would run with
        // nothing on screen for however many seconds it takes. Analyze it
        // directly with a spinner instead of through the streaming API, dropping
        // the spinner before `record` so its fetch tree and final render own the
        // terminal cleanly.
        if is_terminal && files.len() == 1 && dir_files.is_empty() {
            let path = &files[0];
            let label = path.file_name().map_or_else(
                || path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            let spinner = Spinner::start(label);
            let result = cleave::analyze_file(path, &named_target_opts(&cleave_opts))
                .with_context(|| format!("cleave analysis of {}", path.display()));
            drop(spinner);
            record(path, result, true);
        } else {
            // The two batches differ in exactly one way: a file the operator
            // named is analyzed unconditionally, while a file found by walking a
            // directory they named is eligible for the known-good shortcut. That
            // is the whole point of the partition above — the bloom is a bulk
            // optimization, and a named path is not bulk.
            if !files.is_empty() {
                cleave::scan_files(&files, &named_target_opts(&cleave_opts), |event| {
                    if let cleave::ScanEvent::File { path, result } = event {
                        record(&path, *result, true);
                    }
                })?;
            }

            if !dir_files.is_empty() {
                cleave::scan_paths(dir_files, &cleave_opts, |event| {
                    if let cleave::ScanEvent::File { path, result } = event {
                        record(&path, *result, false);
                    }
                })?;
            }
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
        // The terminal result card was built in `classify_report`; write it
        // verbatim, then flush before the stderr footer.
        OutputFormat::Terminal => {
            let _ = show_progress;
            let Ok(mut out) = stdout.lock() else {
                return;
            };
            let _ = out.write_all(r.rendered_context.as_bytes());
            if !r.rendered_context.ends_with('\n') {
                let _ = out.write_all(b"\n");
            }
            // `--extra`: append the full ML diagnostics that explain the grade —
            // which route drove it and the top SHAP features behind it. The
            // compact trait grid only shows static findings; route scores +
            // reasons are computed but otherwise omitted here.
            if config.extra() {
                write_extra_diagnostics(&mut *out, r);
            }
            // The closing summary is written to stderr. Commit this whole card
            // first so stdio buffering cannot strand its final trait beneath the
            // footer (especially visible for a single-file scan).
            let _ = out.flush();
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

/// Cleave context density for machine/LLM output. The primary terminal artifact
/// and its fetched appendix use Scan's own global three-trait cards.
pub(crate) fn tiny_opts_for(config: &ScanConfig) -> cleave::output::TinyOpts {
    if matches!(
        config.format(),
        OutputFormat::Tiny | OutputFormat::Interpret
    ) {
        cleave::output::TinyOpts::tiny()
    } else {
        cleave::output::TinyOpts {
            // Keep five findings per file in machine-facing compact output.
            top_n: 5,
            always_crit: None,
            // Focus on suspicious+ (plus their composite legs) whenever any
            // fired; a merged capture window renders only selected rows, and a
            // suspicious+ hit keeps one trailing row/line of context — the
            // continuation tends to carry the payoff. Unremarkable dependency
            // files fall back to their notable top-five.
            focus_crit: Some(cleave::Criticality::Suspicious),
            // Card layout keeps the compact render headerless.
            card: true,
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
    // Lead with the level, as `output::print_extra` does. It is the only place
    // the loose tail above the suspicious ceiling (L3000..=L25000) is legible:
    // those files grade benign, so nothing else in this render names the rung
    // they fired on.
    let _ = writeln!(out, "  level: {}", crate::output::format_level(r.level),);
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
        // Benign is not always "fired nowhere": everything above the suspicious
        // ceiling (L3000) grades benign while still carrying the rung it fired
        // on, up to the grid max (L25000). Report that rung — dropping it hid
        // the whole loose tail from the LLM-facing render. A file that fired at
        // no level (`-1`) has nothing to report and keeps the bare line.
        match r.level {
            Some(lvl) if lvl >= 0 => format!(
                "scan {class} confidence={:.3} fp-level=L{lvl}{bloom}\n",
                r.probability,
            ),
            _ => format!("scan {class} confidence={:.3}{bloom}\n", r.probability),
        }
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

fn terminal_safe_text(text: &str) -> String {
    crate::deptree::strip_ansi(text)
        .chars()
        .filter(|c| !c.is_control())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn terminal_identity_tokens(text: &str) -> Vec<String> {
    let mut normalized = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            normalized.extend(ch.to_lowercase());
        } else {
            normalized.push(' ');
        }
    }
    normalized.split_whitespace().map(str::to_owned).collect()
}

fn terminal_label_contains_identity(label: &str, identity: &str) -> bool {
    let label = Path::new(label)
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or(label);
    let label = terminal_identity_tokens(label);
    let identity = terminal_identity_tokens(identity);
    !identity.is_empty()
        && label
            .windows(identity.len())
            .any(|candidate| candidate == identity.as_slice())
}

fn terminal_identity_summary(report: &cleave::AnalysisReport, label: &str) -> Option<String> {
    let identity = report.files.first()?.identity.as_ref()?;
    let (what, title) = if let Some(claim) = &identity.title {
        (claim.value.clone(), true)
    } else if let Some(claim) = &identity.name {
        let mut name = claim.value.clone();
        if let Some(version) = &identity.version {
            name.push(' ');
            name.push_str(&version.value);
        }
        (name, false)
    } else if let Some(claim) = &identity.identifier {
        (claim.value.clone(), false)
    } else {
        return None;
    };
    let what = terminal_safe_text(&what);
    if what.is_empty() {
        return None;
    }

    let detail = identity
        .organization
        .as_ref()
        .map(|c| c.value.as_str())
        .or_else(|| identity.producer.as_ref().map(|c| c.value.as_str()))
        .map(terminal_safe_text)
        .filter(|d| !d.is_empty() && !d.eq_ignore_ascii_case(&what));

    // A package name/version already spelled by its filename contributes
    // nothing. Compare semantic tokens so archive punctuation and suffixes do
    // not defeat the check (`nordpass 1.0.2` == `nordpass-1.0.2.tgz`). A
    // document title or identity carrying producer information still earns the
    // line because it adds a useful claim about the artifact.
    if !title && detail.is_none() && terminal_label_contains_identity(label, &what) {
        return None;
    }

    let what = if title { format!("“{what}”") } else { what };
    Some(detail.map_or(what.clone(), |d| format!("{what} · {d}")))
}

fn terminal_finding_path(path: &str) -> String {
    if let Some(decoded) = decoded_region_display_path(path) {
        return decoded;
    }
    let path = collapse_decoded_dup(path);
    let leaf = path.rsplit(ARCHIVE_DELIMITER).next().unwrap_or(&path);
    let name = leaf.rsplit('/').next().unwrap_or(leaf);
    terminal_safe_text(name)
}

fn terminal_note_anchor(file: &cleave::FileAnalysis, finding_id: &str) -> Option<String> {
    for line in &file.context {
        let Some(note) = line.notes.iter().find(|n| n.id.as_str() == finding_id) else {
            continue;
        };
        if let Some(base_line) = line.line {
            let relative = note.off.saturating_sub(line.loc);
            let upto = usize::try_from(relative)
                .unwrap_or(usize::MAX)
                .min(line.data.len());
            let added = line.data[..upto].iter().filter(|&&b| b == b'\n').count();
            let added = u64::try_from(added).unwrap_or(u64::MAX);
            return Some(format!(":{}", base_line.saturating_add(added)));
        }
        // Byte offsets are precise but add little triage value in the compact
        // trait grid. Keep source lines when available; otherwise the filename
        // is the useful anchor.
        return None;
    }
    None
}

fn terminal_finding_location(
    report: &cleave::AnalysisReport,
    file: &cleave::FileAnalysis,
    finding: &cleave::Finding,
) -> String {
    let by_id: std::collections::HashMap<u32, &cleave::FileAnalysis> =
        report.files.iter().map(|f| (f.id, f)).collect();
    let primary_id = report.files.first().map(|f| f.id);
    if let Some(source) = file
        .composite_sources
        .get(finding.id.as_str())
        .and_then(|sources| {
            sources
                .iter()
                .find(|s| s.line.is_some() || s.offset.is_some())
                .or_else(|| sources.first())
        })
        && let Some(source_file) = by_id.get(&source.file)
    {
        if Some(source_file.id) == primary_id {
            return source
                .line
                .map_or_else(String::new, |line| format!("line {line}"));
        }
        let mut location = terminal_finding_path(&source_file.path);
        if let Some(line) = source.line {
            location.push_str(&format!(":{line}"));
        }
        return location;
    }

    if Some(file.id) == primary_id {
        return terminal_note_anchor(file, finding.id.as_str())
            .map_or_else(String::new, |anchor| {
                format!("line {}", anchor.trim_start_matches(':'))
            });
    }

    let mut location = terminal_finding_path(&file.path);
    if let Some(anchor) = terminal_note_anchor(file, finding.id.as_str()) {
        location.push_str(&anchor);
    }
    location
}

/// Whether a finding belongs to the file that lists it.
///
/// `src` alone is not the test. It marks an *inherited copy* — the finding was
/// located in a member below and that member will report it — but a cross-file
/// composite carries source provenance too (`composite_sources` records the
/// members it drew from) while being native to no member at all: it exists only
/// on the container. Filtering on `src.is_none()` therefore dropped every
/// container-scope conclusion from this summary, so localstack-core's
/// `aws-instance-launch-with-user-data` was absent from the terminal view while
/// sitting in the JSON. cleave's `select_ids` and `compact.rs` draw the same
/// distinction.
fn finding_is_native(file: &cleave::FileAnalysis, finding: &cleave::Finding) -> bool {
    finding.src.is_none() || file.composite_sources.contains_key(finding.id.as_str())
}

fn terminal_top_traits(report: &cleave::AnalysisReport) -> Vec<crate::output::TerminalTrait> {
    let mut deepest: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for file in &report.files {
        for finding in file.findings.iter().filter(|f| finding_is_native(file, f)) {
            deepest
                .entry(finding.id.as_str())
                .and_modify(|depth| *depth = (*depth).max(file.depth))
                .or_insert(file.depth);
        }
    }

    let mut ranked: Vec<(&cleave::FileAnalysis, &cleave::Finding)> = report
        .files
        .iter()
        .flat_map(|file| {
            let deepest = &deepest;
            file.findings.iter().filter_map(move |finding| {
                (finding_is_native(file, finding)
                    && finding.crit >= cleave::Criticality::Notable
                    && deepest
                        .get(finding.id.as_str())
                        .is_none_or(|depth| *depth == file.depth))
                .then_some((file, finding))
            })
        })
        .collect();
    // Prefer conclusions made at the artifact's shallower layers: a CHM-level
    // dropper conclusion summarizes its embedded HTML primitive, for example.
    // Confidence resolves peers at the same layer. Within one severity, take one
    // conclusion from each behavioral family before spending another row on a
    // close sibling. A weaker tier never displaces an available stronger one.
    ranked.sort_by(|(file_a, a), (file_b, b)| {
        b.crit
            .rank()
            .cmp(&a.crit.rank())
            .then_with(|| file_a.depth.cmp(&file_b.depth))
            .then_with(|| b.conf.total_cmp(&a.conf))
    });

    let mut selected = Vec::with_capacity(3);
    let mut seen_ids = std::collections::HashSet::new();
    let mut seen_families = std::collections::HashSet::new();
    for criticality in [
        cleave::Criticality::Hostile,
        cleave::Criticality::Suspicious,
        cleave::Criticality::Notable,
    ] {
        for diversify in [true, false] {
            for &(file, finding) in &ranked {
                if finding.crit != criticality {
                    continue;
                }
                let full_id = finding.id.as_str();
                let base_id = full_id.split_once("::").map_or(full_id, |(base, _)| base);
                if seen_ids.contains(base_id) {
                    continue;
                }
                let family = trait_family(full_id);
                if diversify && seen_families.contains(family) {
                    continue;
                }
                let description = if finding.desc.is_empty() {
                    base_id
                        .rsplit('/')
                        .next()
                        .unwrap_or(base_id)
                        .replace('-', " ")
                } else {
                    terminal_safe_text(finding.desc.as_str())
                };
                if description.is_empty() {
                    continue;
                }
                seen_ids.insert(base_id);
                seen_families.insert(family);
                selected.push(crate::output::TerminalTrait {
                    criticality: finding.crit,
                    description,
                    location: terminal_finding_location(report, file, finding),
                });
                // A hostile conclusion is sufficient for triage. Additional
                // rows at the same or weaker severity only repeat support for
                // a verdict the first row already makes unambiguously.
                if criticality == cleave::Criticality::Hostile || selected.len() == 3 {
                    return selected;
                }
            }
        }
    }
    selected
}

#[allow(clippy::too_many_arguments)]
fn render_terminal_context(
    report: &cleave::AnalysisReport,
    decision: &Decision,
    reasons: &[Reason],
    interpretation: Option<&crate::interpret::Interpretation>,
    sha256: &str,
    label: &str,
    bloom_mark: Option<crate::output::BloomMark>,
    member_evals: &MemberEvals,
) -> String {
    // The card frame, top-down: verdict rule → artifact → optional claimed
    // identity → SHA-256 → optional model reason → the three
    // strongest traits across the whole artifact. Plain (piped) output keeps the
    // same information as unframed grep-able lines.
    let root = report.files.first();
    let file_type = root.map(|f| f.file_type.as_str()).unwrap_or_default();
    let size = root.map_or(0, |f| f.size);
    let is_container = report.files.iter().any(|f| f.depth > 0);

    // A grab-bag archive (several independent packages zipped together) reads
    // far better as a stack of per-package verdict cards than as one inherited
    // verdict over a flat member list. When the archive holds two or more
    // packages that independently scored suspicious+, switch to that layout.
    if is_container
        && let Some(cards) = render_archive_cards(
            report,
            decision,
            interpretation,
            sha256,
            label,
            bloom_mark,
            member_evals,
        )
    {
        return cards;
    }

    let color = colored::control::SHOULD_COLORIZE.should_colorize();

    // Each card *leads* with its blank separator (setting it off from the
    // banner or the previous card) and ends flush — the footer brings its own
    // spacing — so a clean scan's quiet summary still hugs the banner.
    let mut head = String::from("\n");
    if color {
        head.push_str(&crate::output::terminal_rule(
            &decision.class,
            decision.probability,
            decision.threshold,
            decision.level,
            cleave::output::terminal_width(),
        ));
        head.push('\n');
    } else {
        let (stamp, _) = crate::output::terminal_badge(
            &decision.class,
            decision.probability,
            decision.threshold,
            decision.level,
        );
        head.push_str(&stamp);
        head.push(' ');
    }
    head.push_str(&crate::output::terminal_artifact_line(
        label,
        file_type,
        size,
        is_container,
    ));
    head.push('\n');
    if let Some(identity) = terminal_identity_summary(report, label)
        .as_deref()
        .and_then(crate::output::terminal_identity_line)
    {
        head.push_str(&identity);
        head.push('\n');
    }
    if let Some(hash) = crate::output::terminal_hash_line(sha256, bloom_mark) {
        head.push_str(&hash);
        head.push('\n');
    }
    if let Some(interp) = crate::output::terminal_interpretation(interpretation, 1) {
        head.push_str(&interp);
        head.push('\n');
    } else if let Some(trailer) = crate::output::terminal_trailer(reasons) {
        // Without an LLM, keep the model's compact explanation in the same slot.
        head.push_str(&trailer);
        head.push('\n');
    }

    let traits = terminal_top_traits(report);
    let body = crate::output::terminal_trait_rows(&traits, cleave::output::terminal_width());
    if !body.trim().is_empty() {
        head.push_str(&body);
    }
    // Flush ending: the next card (or the footer) supplies the separator.
    while head.ends_with("\n\n") {
        head.pop();
    }
    head
}

/// The delimiter cleave inserts between archive layers in a file path
/// (`root.zip!!member.tgz!!inner/file`). Directory separators *within* one
/// archive stay `/`; only a nested-archive boundary is `!!`.
const ARCHIVE_DELIMITER: &str = "!!";

/// The id of the depth-1 ancestor of `file` — the top-level package it belongs
/// to inside the root archive — or `None` for the root itself. `parent_id` is
/// unreliable here (it is dropped for non-container members during compaction),
/// so membership is decided by path: a file belongs to the depth-1 package whose
/// path is the longest archive-prefix of its own.
fn top_package_id(file: &cleave::FileAnalysis, roots: &[(u32, &str)]) -> Option<u32> {
    if file.depth == 0 {
        return None;
    }
    if file.depth == 1 {
        return Some(file.id);
    }
    roots
        .iter()
        .filter(|(_, root)| {
            file.path
                .strip_prefix(*root)
                .is_some_and(|rest| rest.starts_with(ARCHIVE_DELIMITER))
        })
        .max_by_key(|(_, root)| root.len())
        .map(|(id, _)| *id)
}

/// Render a multi-package archive as a stack of independent verdict cards.
///
/// A "grab-bag" archive — several unrelated packages zipped together — is badly
/// served by one inherited verdict over a flat member list: it collapses N
/// distinct malicious packages into a single `HOSTILE` line and buries which
/// file is which. Instead each top-level package inside the archive that scored
/// suspicious+ on its own is framed as its own card (verdict stamp, package
/// name, its findings), clearly nested under the archive banner. The archive's
/// own card is added only when it carries a non-inherited hostile finding of its
/// own, or outscores every package it contains.
///
/// `None` (fall back to the single-card render) when the archive holds fewer
/// than two independently-notable packages — an ordinary single-package scan is
/// better as the one classic card.
#[allow(clippy::too_many_arguments)]
fn render_archive_cards(
    report: &cleave::AnalysisReport,
    decision: &Decision,
    interpretation: Option<&crate::interpret::Interpretation>,
    sha256: &str,
    label: &str,
    bloom_mark: Option<crate::output::BloomMark>,
    member_evals: &MemberEvals,
) -> Option<String> {
    let by_id: std::collections::HashMap<u32, &cleave::FileAnalysis> =
        report.files.iter().map(|f| (f.id, f)).collect();

    // Group every file under its top-level package (its depth-1 ancestor).
    let roots: Vec<(u32, &str)> = report
        .files
        .iter()
        .filter(|f| f.depth == 1)
        .map(|f| (f.id, f.path.as_str()))
        .collect();
    let mut members_of: std::collections::BTreeMap<u32, Vec<u32>> =
        std::collections::BTreeMap::new();
    for file in &report.files {
        if let Some(pkg) = top_package_id(file, &roots) {
            members_of.entry(pkg).or_default().push(file.id);
        }
    }
    if members_of.is_empty() {
        return None;
    }

    // Each package's verdict is the worst independent verdict among its files —
    // exactly how a first-hand scan of that package alone would resolve.
    struct Package {
        id: u32,
        decision: Decision,
        members: Vec<u32>,
    }
    let mut packages: Vec<Package> = members_of
        .into_iter()
        .filter_map(|(pkg, members)| {
            let decision = members
                .iter()
                .filter_map(|id| {
                    member_evals
                        .get(&u64::from(*id))
                        .map(EmbeddedFile::decision)
                })
                .reduce(|best, d| {
                    if decision_outranks(&d, &best) {
                        d
                    } else {
                        best
                    }
                })?;
            Some(Package {
                id: pkg,
                decision,
                members,
            })
        })
        .collect();

    // Banner tally over every package; the cards below spell out only the
    // suspicious+ ones.
    // Worst first, so the reader meets the most dangerous package immediately.
    packages.sort_by(|a, b| {
        if decision_outranks(&a.decision, &b.decision) {
            std::cmp::Ordering::Less
        } else if decision_outranks(&b.decision, &a.decision) {
            std::cmp::Ordering::Greater
        } else {
            a.id.cmp(&b.id)
        }
    });
    let notable: Vec<&Package> = packages
        .iter()
        .filter(|p| {
            matches!(
                p.decision.class,
                Classification::Suspicious | Classification::Hostile
            )
        })
        .collect();

    // Does a root finding belong to the archive itself? It must be non-inherited
    // (native to the container — no `src` child pointing into a member) *and*
    // native to nothing below it: a trait that also fired natively on some member
    // (a package's own atomic trait re-evaluated at container scope, or a
    // composite native to one package's container node) is that member's story,
    // already told by its card. What survives is genuinely the archive's own — an
    // atomic match on the container's own bytes, or a composite that spans
    // packages and so is native to no single one. This mirrors cleave's own
    // "native deeper down belongs to the member" rule.
    let member_native: std::collections::HashSet<&str> = report
        .files
        .iter()
        .filter(|f| f.depth > 0)
        .flat_map(|f| {
            f.findings
                .iter()
                .filter(|x| x.src.is_none())
                .map(|x| x.id.as_str())
        })
        .collect();
    let root_id = report.files.first().map(|f| f.id);
    let is_archive_own =
        |f: &cleave::Finding| f.src.is_none() && !member_native.contains(f.id.as_str());
    let archive_card = report.files.first().is_some_and(|root| {
        root.findings
            .iter()
            .any(|f| f.crit >= cleave::Criticality::Hostile && is_archive_own(f))
    });

    // Fall back to the single classic card unless this is genuinely a grab-bag
    // (two or more independently-notable packages).
    if notable.len() < 2 {
        return None;
    }

    let color = colored::control::SHOULD_COLORIZE.should_colorize();
    let root = report.files.first();
    let file_type = root.map(|f| f.file_type.as_str()).unwrap_or_default();
    let size = root.map_or(0, |f| f.size);

    // ── Banner: verdict rule → 📦 name · TYPE · size → hash ──
    let mut out = String::from("\n");
    if color {
        out.push_str(&crate::output::terminal_rule(
            &decision.class,
            decision.probability,
            decision.threshold,
            decision.level,
            cleave::output::terminal_width(),
        ));
        out.push('\n');
    } else {
        let (stamp, _) = crate::output::terminal_badge(
            &decision.class,
            decision.probability,
            decision.threshold,
            decision.level,
        );
        out.push_str(&stamp);
        out.push(' ');
    }
    out.push_str(&crate::output::terminal_artifact_line(
        label, file_type, size, true,
    ));
    out.push('\n');
    if let Some(identity) = terminal_identity_summary(report, label)
        .as_deref()
        .and_then(crate::output::terminal_identity_line)
    {
        out.push_str(&identity);
        out.push('\n');
    }
    if let Some(hash) = crate::output::terminal_hash_line(sha256, bloom_mark) {
        out.push_str(&hash);
        out.push('\n');
    }
    if let Some(interp) = crate::output::terminal_interpretation(interpretation, 1) {
        out.push_str(&interp);
        out.push('\n');
    }

    // Decoded regions are children of the named artifact, not sibling packages.
    // Give them a compact branch tree; real archive members retain the stronger
    // package cards used for grab-bag archives.
    let embedded_tree = notable.iter().all(|pkg| {
        by_id
            .get(&pkg.id)
            .is_some_and(|file| decoded_region_display_path(&file.path).is_some())
    });

    // ── One independently notable child, worst first ──
    let render_child = |pkg: &Package, last: bool| -> String {
        let file = by_id.get(&pkg.id);
        let name = file.map_or_else(String::new, |f| package_display_path(&f.path));
        let ptype = file.map(|f| f.file_type.as_str()).unwrap_or_default();
        let psize = file.map_or(0, |f| f.size);
        let member_ids: std::collections::HashSet<u32> = pkg.members.iter().copied().collect();

        // Rank over every member in the package, then spend exactly three rows
        // on its strongest distinct traits. A large package reads like one
        // artifact instead of a transcript of its member traversal.
        let mut view = report.clone();
        view.files.retain(|f| member_ids.contains(&f.id));
        for f in &mut view.files {
            f.path = collapse_decoded_dup(&f.path);
        }
        let traits = terminal_top_traits(&view);
        let rows = crate::output::terminal_trait_rows(
            &traits,
            cleave::output::terminal_width().saturating_sub(2),
        );
        if embedded_tree {
            crate::output::terminal_embedded_branch(
                &pkg.decision.class,
                &name,
                ptype,
                psize,
                &rows,
                last,
            )
        } else {
            crate::output::terminal_card(&pkg.decision.class, &name, ptype, psize, &rows)
        }
    };

    // The root's own conclusion belongs directly beneath the root metadata in a
    // decoded tree. A second card repeating the root filename would imply a
    // sibling object where none exists.
    if archive_card && let Some((rid, root)) = root_id.zip(report.files.first()) {
        // The archive's own findings, and the member files their cross-package
        // trails point at — the members must stay in the view so those `↳` legs
        // resolve to real paths, but only the root's findings block is kept.
        let own: std::collections::HashSet<&str> = root
            .findings
            .iter()
            .filter(|f| is_archive_own(f))
            .map(|f| f.id.as_str())
            .collect();
        let referenced: std::collections::HashSet<u32> = own
            .iter()
            .filter_map(|id| root.composite_sources.get(*id))
            .flatten()
            .map(|s| s.file)
            .collect();
        let mut view = report.clone();
        view.files
            .retain(|f| f.id == rid || referenced.contains(&f.id));
        if let Some(v) = view.files.iter_mut().find(|f| f.id == rid) {
            v.findings.retain(|f| own.contains(f.id.as_str()));
        }
        let traits = terminal_top_traits(&view);
        let rows = crate::output::terminal_trait_rows(
            &traits,
            cleave::output::terminal_width().saturating_sub(2),
        );
        if !rows.trim().is_empty() {
            if embedded_tree {
                out.push_str(&rows);
                out.push('\n');
            } else {
                out.push('\n');
                out.push_str(&crate::output::terminal_card(
                    &decision.class,
                    label,
                    file_type,
                    size,
                    &rows,
                ));
            }
        }
    }

    for (index, pkg) in notable.iter().enumerate() {
        if !embedded_tree {
            out.push('\n');
        }
        out.push_str(&render_child(pkg, index + 1 == notable.len()));
    }

    // Note quiet packages omitted from the detailed cards.
    let omitted = packages.len().saturating_sub(notable.len());
    if omitted > 0 {
        let plural = if omitted == 1 { "package" } else { "packages" };
        if color {
            let dim = "\x1b[38;2;100;100;100m";
            out.push_str(&format!(
                "\n {dim}{omitted} clean {plural} not shown\x1b[0m\n"
            ));
        } else {
            out.push_str(&format!("\n {omitted} clean {plural} not shown\n"));
        }
    }

    while out.ends_with("\n\n") {
        out.pop();
    }
    Some(out)
}

/// Collapse cleave's decoded-region path duplication for display. A region
/// decoded out of a member (a unicode-escape/base64 blob) is pathed as
/// `…!!MEMBER!!MEMBER##encoding@off` — the member repeats around the archive
/// delimiter — which renders as an alarming doubled path. When the segment
/// before `##` is immediately preceded by an identical `!!MEMBER`, drop the
/// duplicate so it reads as `MEMBER##encoding@off`. A no-op for any other path.
fn collapse_decoded_dup(path: &str) -> String {
    let Some(enc) = path.rfind("##") else {
        return path.to_string();
    };
    let (head, tail) = path.split_at(enc);
    // The last archive segment before the encoding marker, and everything before
    // it. If that preceding text ends with an identical `!!<segment>`, the member
    // is doubled — keep one copy.
    let Some((rest, seg)) = head.rsplit_once("!!") else {
        return path.to_string();
    };
    if !seg.is_empty()
        && (rest == seg
            || rest
                .strip_suffix(seg)
                .is_some_and(|before| before.ends_with("!!")))
    {
        return format!("{rest}{tail}");
    }
    path.to_string()
}

/// Turn cleave's decoded-region suffix into a compact relationship label.
/// The parent artifact already owns its filename in the summary, so a direct
/// child reads `embedded base64 @ 11096`; a decoded archive member retains only
/// the useful member leaf: `install.js · embedded unicode escape @ 20`.
fn decoded_region_display_path(path: &str) -> Option<String> {
    let path = collapse_decoded_dup(path);
    let (parent, region) = path.rsplit_once("##")?;
    let (encoding, offset) = region.rsplit_once('@')?;
    if encoding.is_empty() || offset.is_empty() {
        return None;
    }
    let encoding = terminal_safe_text(&encoding.replace('-', " "));
    let offset = terminal_safe_text(offset);
    if encoding.is_empty() || offset.is_empty() {
        return None;
    }
    let decoded = format!("embedded {encoding} @ {offset}");
    if parent.contains(ARCHIVE_DELIMITER) {
        let member = terminal_finding_path(parent);
        if !member.is_empty() {
            return Some(format!("{member} \u{00b7} {decoded}"));
        }
    }
    Some(decoded)
}

/// The path of a package relative to the root archive: everything after the
/// first archive delimiter, deeper nesting shown as `/`. `demo.zip!!a.tgz` →
/// `a.tgz`; a bare root path is returned as-is.
fn package_display_path(path: &str) -> String {
    if let Some(decoded) = decoded_region_display_path(path) {
        return decoded;
    }
    path.split_once("!!")
        .map_or_else(|| path.to_string(), |(_, m)| m.replace("!!", "/"))
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
    pub(crate) report: cleave::types::CompactReport,
    /// Fetched dependencies to mirror into hopper as their own samples.
    pub(crate) dependency_results: Vec<DepResult>,
    /// Pre-rendered terminal card or machine/LLM context payload.
    pub(crate) rendered_context: String,
    /// Optional LLM interpretation blended with the ML verdict (`--interpret`).
    pub(crate) interpretation: Option<crate::interpret::Interpretation>,
    /// Set instead of `interpretation` when the caller runs the LLM step
    /// itself; see [`PendingLlm`].
    pub(crate) pending_llm: Option<PendingLlm>,
    /// Whether cleave replayed this analysis from its on-disk cache rather than
    /// running the pipeline. See [`ScanResult::analysis_cached`].
    pub(crate) analysis_cached: bool,
}

/// Confident crit-5 findings the hostile arm needs — the anchor the rest of the
/// arm corroborates.
///
/// Was `1`, on the reasoning that one suffices *because* it is corroborated:
/// real malware routinely carries a single hostile trait beside supporting
/// crit-4s (`rex-powershell`, `openclaude`, and a rhadamanthys coinminer all
/// have exactly one), and an earlier attempt at two lost all three.
///
/// Raised to `2` (2026-08-27) after the single-anchor arm was observed marking
/// model-clean legitimate packages hostile on one trait that describes the
/// package's own function: `ansible-core` on a PowerShell base64-exec trait —
/// which is how its Windows connection plugin works — with `confident_hostile=1,
/// severe=3, families=3`, and Cherry Studio (`v2.0.9.tar.gz`) at `1/4/4`. Both
/// fired the arm at its exact minimum. A single crit-5 is one witness however
/// many crit-4s sit beside it, and the crit-4s in both cases were describing the
/// same legitimate behavior from other angles.
///
/// The named regressions above are the known cost; re-measure them alongside the
/// gauntlet missed-sample pool before relaxing this again.
const TRAIT_FLOOR_HOSTILE_CRIT5: u32 = 2;

/// Confident severe findings (crit-5 *or* crit-4) the hostile arm needs in
/// total. Rejects the thin pair that family diversity alone admits: a
/// ScreenConnect RMM signature plus `pe-large-without-material-section` is two
/// findings in two families, but a generic PE-layout anomaly is not evidence of
/// malice, and that pair was marking a stock `libwebp.dll` hostile.
const TRAIT_FLOOR_HOSTILE_SEVERE: u32 = 3;

/// Distinct trait families the hostile arm's severe findings must span.
/// Counting alone treats one behavior described three ways as three independent
/// witnesses: the static-keys false positive presented as
/// `rust-inline-hook-hijack`, `rust-hook-byte-copy`, and
/// `rust-mprotect-hook-patch` — three findings, one directory, overlapping
/// regexes over the same two tokens. Corroboration has to come from somewhere
/// else in the tree to be corroboration at all.
///
/// Counted over both severe tiers, so crit-5 and crit-4 families together have
/// to reach it — the same set the arm's `severe()` total draws from.
///
/// Was two. The argument for two was that every observed false positive was a
/// *single*-family cluster, already rejected, while three would have discarded
/// `darkglitch` — a Python RAT whose hostile traits span only `backdoor/rat` and
/// `backdoor/tasking`. That argument was made against a four-deep family, and
/// [`TRAIT_FLOOR_FAMILY_DEPTH`] is now two: `darkglitch`'s pair collapses to one
/// family at this depth regardless, so three no longer costs what it did.
///
/// Raised to three (2026-08-27) because two families is a materially weaker claim
/// once a family is a *kind* of behavior rather than a technique. `PyAutoIt` — a
/// legitimate AutoIt wrapper — reached the hostile arm at `confident_hostile=4,
/// severe=5, families=2`, on the input-synthesis and window-manipulation traits
/// that are AutoIt's entire purpose. It also makes the two arms consistent: the
/// suspicious arm has always required three (see
/// [`TRAIT_FLOOR_SUSPICIOUS_FAMILIES`]), and the hostile arm demanding *less*
/// diversity than the suspicious one had no principle behind it.
const TRAIT_FLOOR_HOSTILE_FAMILIES: usize = 3;

/// Distinct trait families the suspicious arm's crit-4 findings must span.
/// Subsumes a count — n families need n findings — so this is the arm's only
/// threshold.
///
/// Set above [`TRAIT_FLOOR_HOSTILE_FAMILIES`] deliberately. The hostile arm has
/// a confident crit-5 anchoring it; this arm has nothing but the breadth of its
/// own evidence, so it has to be broader to make the same claim.
const TRAIT_FLOOR_SUSPICIOUS_FAMILIES: usize = 3;

/// Trait-hierarchy depth that defines a family: `objectives/anti-static` rather
/// than a leaf path or the finer `objectives/anti-static/obfuscation/string`.
/// One definition, used by both arms.
///
/// Was four (`objectives/evasion/process/hook`), chosen as the shallowest depth
/// that gave nothing up: at three, the static-keys cluster correctly collapsed to
/// one family, but so did `darkglitch` — a Python RAT whose three hostile traits
/// are genuinely distinct capabilities under `command-and-control/backdoor`
/// (`rat/multi` and `tasking/filesystem`) — and it lost its verdict entirely.
///
/// Shallowed to two (2026-08-27). Four proved too fine to be corroboration on the
/// gauntlet false-positive pool: `ansible-core` reached the hostile arm with
/// three "independent" families that are all one idea — a PowerShell base64
/// decode-and-execute, which is how its Windows connection plugin works, counted
/// once per subdirectory it was spelled in. Two makes a family a *kind* of
/// behavior rather than a technique, so restating one behavior at different
/// depths can no longer corroborate itself.
///
/// The darkglitch class of verdict is the known cost, and it is the thing to
/// re-measure first if the missed-sample pool regresses.
const TRAIT_FLOOR_FAMILY_DEPTH: usize = 2;

/// The family a finding belongs to: the first [`TRAIT_FLOOR_FAMILY_DEPTH`]
/// segments of its hierarchy path. A trait id is `path::leaf`
/// (`objectives/evasion/process/hook/inline::rust-inline-hook-hijack`); an id
/// carrying no path, or a shorter one, is its own family rather than joining a
/// catch-all bucket that would let unrelated traits corroborate each other.
fn trait_family(id: &str) -> &str {
    let path = id.split("::").next().unwrap_or(id);
    path.match_indices('/')
        .nth(TRAIT_FLOOR_FAMILY_DEPTH - 1)
        // `cut` indexes an ASCII '/', so it is a char boundary by construction;
        // `get` keeps that fact from resting on a panic.
        .and_then(|(cut, _)| path.get(..cut))
        .unwrap_or(path)
}

/// Minimum cleave confidence (`c`) for a finding to count toward the trait floor.
/// Low-confidence high-crit findings are exactly the incidental ones that fire on
/// busy benign binaries (e.g. a couple of speculative crit-4s among hundreds of
/// findings), so the floor only acts on evidence cleave is sure about. Measured:
/// at 0.76 the dropper keeps all 4 crit-4s and the PE keeps all 3 crit-5s, while
/// no /usr/bin benign trips the crit-5 arm.
const TRAIT_FLOOR_MIN_CONFIDENCE: f32 = 0.76;

/// Confidence-filtered crit-5/crit-4 tallies, with the trait families each tier
/// drew from. Only findings scoring `>= TRAIT_FLOOR_MIN_CONFIDENCE` are counted
/// at all, so an unscored or hedged trait contributes to neither arm.
struct TraitFloorCounts {
    hostile: u32,
    suspicious: u32,
    hostile_confidence: f32,
    suspicious_confidence: f32,
    /// Families across both severe tiers — the hostile arm's diversity test.
    severe_families: std::collections::HashSet<String>,
    /// Families among the crit-4s alone — the suspicious arm's.
    suspicious_families: std::collections::HashSet<String>,
}

impl TraitFloorCounts {
    /// Confident severe findings across both tiers.
    const fn severe(&self) -> u32 {
        self.hostile + self.suspicious
    }
}

/// `well-known/` subtrees that positively identify a sample as a *named piece of
/// software* rather than as a threat. `malware/` and `unwanted/` are pointedly
/// absent: those recognizers identify a sample too, but identifying it as
/// malware is not a reason to hold back.
const KNOWN_SOFTWARE_PREFIXES: [&str; 5] = [
    "well-known/app/",
    "well-known/lib/",
    "well-known/tool/",
    "well-known/game/",
    "well-known/dual-use/",
];

/// Whether cleave confidently recognized this sample as a named application,
/// library, tool or known dual-use utility.
///
/// Used to raise — never to waive — the hostile arm's anchor requirement. A
/// sample cleave can name is one whose hostile traits are far more likely to be
/// describing the program's own advertised function: `ansible-core`'s PowerShell
/// base64-exec *is* its Windows connection plugin, and eight
/// `well-known/app/infrastructure` Ansible recognizers fire on the same render
/// while the floor promotes it to hostile on that one trait.
fn identified_as_known_software(findings: &[cleave::types::CompactTrait]) -> bool {
    findings.iter().any(|f| {
        f.confidence >= TRAIT_FLOOR_MIN_CONFIDENCE
            && KNOWN_SOFTWARE_PREFIXES
                .iter()
                .any(|prefix| f.id.starts_with(prefix))
    })
}

fn trait_floor_counts(findings: &[cleave::types::CompactTrait]) -> TraitFloorCounts {
    let mut out = TraitFloorCounts {
        hostile: 0,
        suspicious: 0,
        hostile_confidence: 0.0,
        suspicious_confidence: 0.0,
        severe_families: std::collections::HashSet::new(),
        suspicious_families: std::collections::HashSet::new(),
    };
    for f in findings {
        let conf = f.confidence;
        if conf < TRAIT_FLOOR_MIN_CONFIDENCE {
            continue;
        }
        let family = trait_family(&f.id);
        match f.criticality {
            5 => {
                out.hostile += 1;
                out.hostile_confidence = out.hostile_confidence.max(conf);
                out.severe_families.insert(family.to_string());
            }
            4 => {
                out.suspicious += 1;
                out.suspicious_confidence = out.suspicious_confidence.max(conf);
                out.severe_families.insert(family.to_string());
                out.suspicious_families.insert(family.to_string());
            }
            _ => {}
        }
    }
    out
}

/// Trait floor: override a model-**Benign** verdict when cleave surfaced
/// high-criticality evidence the model did not act on:
///   - a hostile (crit-5) trait, corroborated by enough further severe findings
///     to reach [`TRAIT_FLOOR_HOSTILE_SEVERE`] across
///     [`TRAIT_FLOOR_HOSTILE_FAMILIES`] trait families → **Hostile**
///   - suspicious (crit-4) traits spanning
///     [`TRAIT_FLOOR_SUSPICIOUS_FAMILIES`] families → **Suspicious**
///
/// Both arms count only confident findings (`c >= TRAIT_FLOOR_MIN_CONFIDENCE`),
/// and both measure evidence by the families it comes from, so one behavior
/// spelled several ways cannot corroborate itself.
///
/// This is a backstop for model misses, not a second opinion: it fires only on
/// a model-Benign verdict, and every threshold above is deliberately set where
/// a lone mislabeled trait — or a cluster of near-duplicate ones — cannot reach
/// it alone.
///
/// Never lowers a model verdict. Override levels are pinned to the same band
/// boundaries used by ordinary and interpreted verdicts, so a level-only
/// downstream consumer cannot reinterpret the override as another class.
fn apply_trait_floor(
    decision: &mut Decision,
    findings: &[cleave::types::CompactTrait],
    active_level: Option<u16>,
    grid_max: u16,
    label: &str,
) {
    if decision.class != Classification::Benign {
        return;
    }
    let counts = trait_floor_counts(findings);
    // Recognized software has to clear a higher anchor before the floor will
    // *block* it. Not an exemption — one extra confident crit-5, which a genuine
    // compromise adding its own hostile behavior still reaches, while a lone
    // trait describing the program's own advertised function no longer promotes
    // a model-clean sample straight past review. Falling short here drops
    // through to the suspicious arm below, which is where "a human should look"
    // belongs.
    let required_crit5 = if identified_as_known_software(findings) {
        TRAIT_FLOOR_HOSTILE_CRIT5.saturating_add(1)
    } else {
        TRAIT_FLOOR_HOSTILE_CRIT5
    };
    if counts.hostile >= required_crit5
        && counts.severe() >= TRAIT_FLOOR_HOSTILE_SEVERE
        && counts.severe_families.len() >= TRAIT_FLOOR_HOSTILE_FAMILIES
    {
        decision.class = Classification::Hostile;
        decision.probability = counts.hostile_confidence;
        decision.level = interpreted_level(active_level, grid_max, Classification::Hostile, true);
        // The model graded this benign yet cleave is confident it carries a
        // hostile (crit-5) trait — a model gap worth investigating. INFO keeps
        // it visible in serve/worker mode (scan=info) without spamming default
        // CLI runs (scan=warn).
        tracing::info!(
            path = %label,
            arm = "crit5",
            confident_hostile = counts.hostile,
            confident_severe = counts.severe(),
            families = counts.severe_families.len(),
            trait_confidence = format!("{:.3}", counts.hostile_confidence),
            level = ?decision.level,
            "TRAIT FLOOR: model said benign but cleave found corroborated hostile traits — escalated to hostile",
        );
        return;
    }
    if counts.suspicious_families.len() >= TRAIT_FLOOR_SUSPICIOUS_FAMILIES {
        decision.class = Classification::Suspicious;
        decision.probability = counts.suspicious_confidence;
        // The crit-4 arm by definition lacked the crit-5 anchor, but a lone
        // confident hostile trait may still be present and is corroboration.
        decision.level = interpreted_level(
            active_level,
            grid_max,
            Classification::Suspicious,
            counts.hostile > 0,
        );
        tracing::info!(
            path = %label,
            arm = "crit4",
            confident_suspicious = counts.suspicious,
            families = counts.suspicious_families.len(),
            trait_confidence = format!("{:.3}", counts.suspicious_confidence),
            level = ?decision.level,
            "TRAIT FLOOR: model said benign but cleave found corroborated suspicious traits — escalated to suspicious",
        );
    }
}

/// Which optional output surfaces the caller will read. `classify_report`
/// skips building anything no consumer looks at. The default is none of them —
/// the bare JSON-verdict shape (server and validation).
#[derive(Clone, Copy, Default)]
pub(crate) struct OutputNeeds {
    /// Render `rendered_context` as the LLM query payload (`--format
    /// interpret`): byte-for-byte the user message a live `--interpret` query
    /// sends (the sanitized tiny render), without the system prompt.
    /// Independent of `interpret`, which controls actually querying.
    pub llm_view: bool,
    /// Show the live fetch log / dependency tree (interactive terminal only).
    pub fetch_progress: bool,
    /// Build the rendered terminal/tiny context body.
    pub render_context: bool,
    /// List never-analyzed archive members (`--show=all` JSON manifest).
    pub list_all_members: bool,
    /// Capture and grade each fetched dependency's standalone report
    /// (`dependency_results`) for hopper renewal. Without this — or one of the
    /// render/LLM surfaces above — a scan drops them unread, so the capture
    /// and the per-dependency model pass are skipped entirely.
    pub deps_for_upload: bool,
}

/// Fold an LLM interpretation into the verdict: the three ways a second
/// opinion may move class, probability and level. Shared by the inline
/// path in [`classify_report`] and the deferred one in
/// [`apply_pending_interpretation`], so both land on the same answer.
fn blend_interpretation(
    label: &str,
    model: &Model,
    interp: &crate::interpret::Interpretation,
    class: &mut Classification,
    probability: &mut f32,
    lvl: &mut Option<i32>,
) {
    // Adopt the blended verdict as the effective one when the LLM out-read ML
    // (escalating a missed threat, or clearing an ML false positive). The `ml`
    // section reflects litmus's final answer; the LLM's raw grade + rationale
    // stay in the `llm` section. The interpreted level is pinned to the target
    // band's loosest rung (see `interpreted_level`): the active hostile level for
    // an escalation, the suspicious ceiling for a hold/downgrade, L-1 for benign.
    if interp.grade.is_some() && interp.outcome as u8 != *class as u8 {
        // INFO, not WARN: an LLM override of the ML verdict is normal operation,
        // not a fault. (It also kept surfacing as the last stderr line a caller
        // grabbed when a slow run was externally killed, making a benign shift look
        // like a crash cause.)
        tracing::info!(
            path = %label,
            ml = ?*class,
            outcome = ?interp.outcome,
            grade = interp.grade.map_or("?", crate::interpret::LlmGrade::as_str),
            conf = format!("{:.4}", interp.blended),
            reason = %interp.interpretation,
            "LLM interpretation shifted the verdict",
        );
        let ml_class = *class;
        let ml_level = *lvl;
        *class = interp.outcome;
        *probability = interp.blended;
        *lvl = if ml_class == Classification::Hostile
            && interp.outcome == Classification::Suspicious
        {
            // A cleared hostile is placed by how deep ML fired, not pinned to the
            // band's edge: the level ML reached is the budget for how far one
            // contrary opinion may move it.
            softened_level(ml_level, model.active_level(), model.grid_max())
        } else {
            interpreted_level(
                model.active_level(),
                model.grid_max(),
                interp.outcome,
                interp.corroborated,
            )
        };
    } else if interp.grade == Some(crate::interpret::LlmGrade::Benign)
        && *class == Classification::Hostile
        && *lvl == Some(0)
    {
        // `may_cross` refused to move a verdict off the grid's tightest budget on
        // one contrary opinion, and that stands — but the disagreement is still
        // evidence, so the verdict gives up the depth it cannot justify and sits
        // on the weakest hostile rung instead. Still blocked, still reviewed.
        let weakened = model.active_level().map(i32::from);
        tracing::info!(
            path = %label,
            from = 0,
            to = ?weakened,
            reason = %interp.interpretation,
            "LLM cleared an L0 hostile — held in band, moved to the weakest rung",
        );
        *lvl = weakened;
    } else if interp.grade == Some(crate::interpret::LlmGrade::Hostile)
        && *class == Classification::Hostile
        && let Some(level) = *lvl
        && level > 0
    {
        // Both detectors independently said hostile, so the class does not move —
        // but agreement is still evidence, and leaving the verdict on the rung ML
        // happened to stop at understates it. Halve the level: deeper into the
        // hostile band, bounded at 0, and never out of it.
        //
        // What it refines is usually already an assertion rather than a
        // measurement: a floor-driven hostile is pinned to the *weakest* hostile
        // rung by `interpreted_level`, which is why every L25 in the gauntlet
        // missed pool carries a floor probability (0.98/0.99) rather than a model
        // one. On a genuinely swept level it does overwrite measured data, and a
        // stricter deploy than this one will read the halved value as hostile
        // where the sweep alone would have said suspicious.
        let strengthened = level / 2;
        tracing::info!(
            path = %label,
            from = level,
            to = strengthened,
            "both detectors agree hostile — verdict moved deeper into the band",
        );
        *lvl = Some(strengthened);
    }
}

/// Hands the caller's CPU admission back before the LLM round trip.
///
/// The worker admits an analysis through a gate sized for the Rayon pool
/// and holds that permit on the blocking thread for the whole classify.
/// The LLM second opinion at the end of [`classify_report`] is a 2-8 s
/// network wait that needs no CPU, and on the production worker it was the
/// majority of every permit's lifetime: 16 slots, 5 permits, ~2 of 16 cores
/// busy. Calling the lease there lets the next analysis start its CPU work
/// while this one waits on the endpoint. `None` for callers with no gate.
pub(crate) type CpuLease = Box<dyn FnOnce() + Send>;

/// An LLM second opinion the caller has agreed to run itself, after the
/// analysis has been posted with its ML verdict. Produced by
/// `classify_report` instead of calling the model when a `CpuLease` is
/// given: that caller owns the post-CPU tail and can post the ML result now
/// and the LLM's amendment later (`apply_pending_interpretation`), so the
/// endpoint's latency stops sitting on the completion path.
#[derive(Debug, Clone)]
pub struct PendingLlm {
    /// The sanitized render the model reads.
    pub ctx: String,
    /// Whether this one may be skipped when the endpoint is saturated.
    pub admission: crate::interpret::LlmAdmission,
    /// cleave's own severity, read once from the typed report.
    pub findings: crate::interpret::FindingSeverity,
}

/// Run a deferred second opinion and fold it into `result` exactly as
/// [`classify_report`] would have inline. Returns whether anything changed
/// and therefore needs re-posting: the LLM section itself is new information,
/// so any interpretation counts. `false` when nothing was pending, the gate
/// declined, or the endpoint failed.
pub(crate) fn apply_pending_interpretation(
    result: &mut ScanResult,
    cfg: &crate::interpret::InterpretConfig,
    model: &Model,
) -> bool {
    let Some(pending) = result.pending_llm.take() else {
        return false;
    };
    let Some(interp) = crate::interpret::interpret(
        cfg,
        &pending.ctx,
        result.classification,
        result.probability,
        crate::interpret::LevelContext {
            fired: result.level,
            active: model.active_level(),
            grid_max: model.grid_max(),
        },
        pending.findings,
        // Deferred means nobody is on the line: a worker job, or serve's own
        // idle puller. It takes its permit behind foreground callers.
        crate::interpret::LlmCaller::Background,
    ) else {
        return false;
    };
    if let Some(grade) = interp.grade {
        tracing::info!(
            file = %result.path,
            sha256 = %result.sha256,
            grade = grade.as_str(),
            outcome = %interp.outcome,
            conf = format!("{:.4}", interp.blended),
            cached = interp.cached,
            interpretation = %interp.interpretation,
            "LLM interpretation",
        );
    }
    blend_interpretation(
        &result.path,
        model,
        &interp,
        &mut result.classification,
        &mut result.probability,
        &mut result.level,
    );
    result.interpretation = Some(interp);
    true
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
    tiny_opts: &cleave::output::TinyOpts,
    interpret: Option<&crate::interpret::InterpretConfig>,
    root_path: &Path,
    fetch: crate::fetch::FetchPolicy,
    zip_passwords: &[String],
    needs: OutputNeeds,
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
    // Live stage tracker for the worker/server census. Without the split set
    // here, the whole of classify_report reported as "features+model" — and
    // dependency fetch+analysis, the expensive half, was repeatedly
    // misattributed to featurization during triage.
    phase: Option<&cleave::PhaseTracker>,
    cpu_lease: Option<CpuLease>,
) -> Result<ClassifiedReport> {
    // Read before the pipeline consumes `report`: whether cleave produced this
    // analysis or replayed it from its cache is the difference between a
    // request that did the work and one that did not.
    let analysis_cached = report.cache_hit;
    let OutputNeeds {
        llm_view,
        fetch_progress,
        render_context,
        list_all_members,
        deps_for_upload,
    } = needs;
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
    //
    // Standalone per-dependency captures (and the classify_dependency grading
    // below) exist for hopper uploads, the LLM query payload, and the rendered
    // dependency appendix. When none of those consumers is active — a plain
    // JSON scan with no upload target — skip the capture up front.
    let dump_dir = std::env::var_os("SCAN_INTERPRET_DUMP_DIR");
    let need_dep_results =
        deps_for_upload || interpret.is_some() || llm_view || render_context || dump_dir.is_some();
    if let Some(p) = phase {
        p.set("fetch+graft");
    }
    let fetch_start = Instant::now();
    let (fetch_edges, fetched_deps, dependency_registries) = crate::fetch::orchestrate(
        &mut report,
        root_path,
        fetch,
        fetch_progress,
        need_dep_results,
        zip_passwords,
    );
    let fetch_ms = crate::duration_ms(fetch_start.elapsed());
    if let Some(p) = phase {
        p.set("features+model");
    }
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
    // Dependency backrefs need only the cited reference's span length. Capture
    // that scalar now instead of keeping every file's filefacts graph alive
    // through JSON conversion and the per-member model pass.
    let fetch_ref_lengths: std::collections::HashMap<(&str, u64), u64> = fetch_edges
        .iter()
        .filter_map(|edge| {
            let off = edge.source_offset?;
            let len = report
                .files
                .iter()
                .find(|file| file.sha256 == edge.source_sha256)
                .and_then(|file| file.filefacts.as_ref())
                .and_then(|facts| {
                    facts
                        .references
                        .iter()
                        .find(|reference| reference.offset == off)
                })
                .map_or(1, |reference| {
                    u64::try_from(reference.evidence.len()).unwrap_or(u64::MAX)
                });
            Some(((edge.source_sha256.as_str(), off), len))
        })
        .collect();
    // Text/LLM renderers consume the typed report. Plain JSON does not: compact
    // conversion has already retained precisely the traits, metrics, symbols,
    // references, and context it emits. Release the much wider cleave graph
    // before featurizing thousands of compact member nodes.
    //
    // Released *during conversion* when nothing later renders the typed
    // report: the consuming variant drops each `FileAnalysis` as its compact
    // projection is built, so the full typed graph and the full compact copy
    // never co-reside (the borrowing version held both until the whole
    // conversion finished — on a member-heavy sample the typed graph is the
    // single largest live allocation in the process). The rest of the typed
    // report is reset right after for the same reason. Freeing all of it
    // after the `to_value` instead made the report, the compact copy, and the
    // JSON DOM all peak together (MiniMax: a ~3 GB step at the very end of an
    // already 12.8 GB run, for a report whose serialized form is 490 MB).
    let keep_typed_report = render_context || interpret.is_some() || llm_view || dump_dir.is_some();
    let mut compact = if keep_typed_report {
        cleave::types::compact::compact_from_files(&report.files)
    } else {
        cleave::types::compact::compact_from_files_consuming(std::mem::take(&mut report.files))
    };
    if !keep_typed_report {
        let target = report.target.clone();
        report = cleave::AnalysisReport::new(target);
    }
    validate_report_references(label, &compact);
    let formula = compact
        .files
        .first()
        .and_then(|file| file.formula.clone())
        .unwrap_or_default();

    // Attach the fetch edge log at report level (`source_sha256 → content_sha256`
    // per reference). Report-level, not per-file: a fetch is a per-event
    // observation, so it never falsely dedups when content is exploded by hash.
    if !fetch_edges.is_empty() {
        compact.fetched = fetch_edges
            .iter()
            .map(|e| serde_json::to_value(e).unwrap_or_default())
            .collect();
    }
    // Parse the report once with every optional raw subtree any specialist may
    // read, then share it across the root and embedded-file scoring passes.
    // Specialist ONNX graphs still load on demand; this only keeps archive
    // members with different filetypes from missing route-specific raw fields.
    let needs = ctx.raw_needs().union(crate::features::RawNeeds::all());
    // The sample's own decision featurizes its own files only. With nothing
    // fetched this is the whole report; otherwise drop the grafted payloads so
    // they can't dilute the aggregate (they still classify individually via the
    // embedded pass over the full report, where a hostile one elevates).
    let parsed = crate::features::ParsedReport::from_compact_report(
        &compact,
        needs,
        (!fetch_edges.is_empty()).then_some(&own_shas),
    );
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
    let pf = compact.files.first();
    let file_type = pf.map_or_else(|| "unknown".to_string(), |f| f.file_type.clone());
    let size_bytes = pf.map_or(0, |f| f.size);
    let sha256 = pf.map(|f| f.sha.clone()).unwrap_or_default();

    let (mut decision, model_scores, skipped_models) =
        model.predict_for_report_detailed(&file_type, &raw_features, &parsed)?;

    let finding_counts = count_findings(&compact);

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
        root_findings(&compact),
        model.active_level(),
        model.grid_max(),
        label,
    );

    // Extract every embedded file (archive members at depth > 0), run each
    // through the model individually, and elevate the parent if any embedded
    // file's decision outranks it. A count cap here would make the result depend
    // on archive ordering and permit a hostile tail member to evade elevation.
    let embedded_entries: Vec<&cleave::types::CompactFile> = embedded_entries(&compact).collect();

    // Fetched payloads keyed by the sha of the content retrieved, with the
    // declaring file + the byte the reference sits at. When the embedded pass
    // classifies one of these hostile/suspicious, that verdict is pinned back
    // onto the declaring manifest at the reference byte (below the pass).
    let fetched_by_content: std::collections::HashMap<&str, (&str, Option<u64>, u64, &str)> =
        fetch_edges
            .iter()
            .filter_map(|r| {
                r.content_sha256.as_deref().map(|c| {
                    let source_len = r
                        .source_offset
                        .and_then(|off| {
                            fetch_ref_lengths
                                .get(&(r.source_sha256.as_str(), off))
                                .copied()
                        })
                        .unwrap_or(1);
                    (
                        c,
                        (
                            r.source_sha256.as_str(),
                            r.source_offset,
                            source_len,
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
            let &(src_sha, src_off, source_len, locator) =
                fetched_by_content.get(ef.sha256.as_str())?;
            Some(DepBackref {
                source_sha: src_sha.to_string(),
                source_offset: src_off,
                source_len,
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
        inject_dependency_backref(&mut compact, backref);
    }

    let top_findings = extract_top_findings(root_findings(&compact), &final_decision.class);

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
            // Same retention rubric as the root report: grading above consumed
            // the complete standalone report; the stored/mirrored body keeps
            // only the nodes someone will read again.
            let raw = match serde_json::from_str::<cleave::types::CompactReport>(&dep.raw) {
                Ok(mut report) => {
                    apply_report_retention(&mut report);
                    serde_json::to_string(&report).unwrap_or(dep.raw)
                }
                Err(_) => dep.raw,
            };
            DepResult {
                verdict,
                members,
                sha256: dep.content_sha,
                locator: dep.locator,
                url: dep.url,
                size: dep.size,
                provenance,
                raw,
            }
        })
        .collect();

    // LLM second opinion (root file only). Build the cleave tiny render once: it
    // feeds the model (below) and, when `SCAN_INTERPRET_DUMP_DIR` is set, is
    // written to `<dir>/<sha256>.render` — the raw (untransformed) render, so the
    // prompt-tuning harness (`hacks/interpret-tune`) can sweep render variants
    // offline from one scan, independent of whether `--interpret` is on.
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
    // CPU work is done: featurized, scored, dependencies graded. What follows
    // is a network wait (the LLM) and light rendering. Give the admission
    // permit back now so the pool is not idle for the round trip.
    // A caller that gave a lease owns the tail: it posts the ML verdict now
    // and runs the LLM step from `pending_llm` afterwards.
    let owner_runs_llm = cpu_lease.is_some();
    if let Some(release) = cpu_lease {
        release();
    }
    // Census label for the round trip. Without it the wait was reported as
    // "features+model", which hid that the pool was idle for the network.
    if let (Some(p), true) = (phase, interpret.is_some() && !owner_runs_llm) {
        p.set("interpret");
    }
    let interpret_start = Instant::now();
    let mut pending_llm: Option<PendingLlm> = None;
    let interpretation =
        if owner_runs_llm && let (Some(cfg), Some(ctx)) = (interpret, llm_ctx.as_deref()) {
            let findings = crate::interpret::FindingSeverity::from_report(&report);
            pending_llm = crate::interpret::admission(
                cfg,
                final_decision.class,
                final_decision.probability,
                crate::interpret::LevelContext {
                    fired: final_decision.level,
                    active: model.active_level(),
                    grid_max: model.grid_max(),
                },
                findings,
                ctx,
            )
            .map(|admission| PendingLlm {
                ctx: ctx.to_string(),
                admission,
                findings,
            });
            None
        } else {
            interpret.and_then(|cfg| {
                // The gate lives in `interpret::interpret`: it runs when ML fired at or
                // below the cutoff level OR cleave surfaced a suspicious/hostile finding ML
                // under-weighted (so an ML-blind packed binary still gets a second
                // opinion); it returns `None` when gated out or on any failure.
                let interp = crate::interpret::interpret(
                    cfg,
                    llm_ctx.as_deref()?,
                    final_decision.class,
                    final_decision.probability,
                    // Where ML placed this file on the calibrated FP axis, which is what
                    // bounds how far the LLM may move the band (see `interpret::blend`).
                    crate::interpret::LevelContext {
                        fired: final_decision.level,
                        active: model.active_level(),
                        grid_max: model.grid_max(),
                    },
                    // cleave's own verdict, read from the structured report rather than
                    // re-parsed out of the render it produced. See `FindingSeverity`.
                    crate::interpret::FindingSeverity::from_report(&report),
                    // Inline means a caller is waiting: a serve request or an
                    // interactive scan. It never queues behind background work.
                    crate::interpret::LlmCaller::Foreground,
                )?;
                if let Some(grade) = interp.grade {
                    tracing::info!(
                        file = %label,
                        sha256 = %sha256,
                        grade = grade.as_str(),
                        outcome = %interp.outcome,
                        conf = format!("{:.4}", interp.blended),
                        cached = interp.cached,
                        interpretation = %interp.interpretation,
                        "LLM interpretation",
                    );
                }
                Some(interp)
            })
        };
    // Dominant suspect for a slow contended run: the LLM round-trip (queue wait +
    // generation) against a shared endpoint. Zero when `--interpret` is off or gated.
    let interpret_ms = crate::duration_ms(interpret_start.elapsed());

    if let Some(interp) = &interpretation {
        blend_interpretation(
            label,
            model,
            interp,
            &mut final_decision.class,
            &mut final_decision.probability,
            &mut final_decision.level,
        );
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
        // `--interpret` query sends — the sanitized render with its annotations
        // recategorized, just without the system prompt. The recategorization is
        // applied here and in `interpret::interpret`, never in
        // `render_interpret_context` itself: scan parses that render back for the
        // LLM admission gate and for `Evidence` (`has_elevated_finding`,
        // `has_hostile_finding`, `render_mostly_readable`), all of which key on
        // the `SEV` letter this transform removes. Stripping it upstream silently
        // withdrew samples from the gate — measured as seven true positives
        // dropping from a caught verdict to `lvl = -1`.
        recategorize_annotations(&llm_ctx.unwrap_or_default())
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
            &final_decision,
            &reasons,
            interpretation.as_ref(),
            &sha256,
            label,
            bloom_mark,
            &member_evals,
        );
        if let Some(fetched) = render_terminal_fetch_context(
            &fetch_edges,
            &dependency_results,
            &dependency_registries,
            &report,
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

    // Retention rubric: everything above — featurization, the model, the
    // embedded pass, renders — consumed the complete report; what remains is
    // what gets stored (hopper bodies, JSON output). Sub-signal member nodes
    // are the bulk of that weight and nobody reads them again, so they are
    // dropped here (see `apply_report_retention`). A `--show=all` manifest
    // request is an explicit ask for the complete listing, so it opts out.
    if !list_all_members {
        apply_report_retention(&mut compact);
    }

    // Surface the archive members cleave catalogued but never analyzed, so a
    // `--show=all` JSON manifest lists every file (path/type/size) — not just the
    // ones that produced findings. Appended last, after featurization and the
    // embedded-file pass have consumed `report_json`, so the listing never feeds
    // the model. Empty unless `--show=all` requested the manifest.
    if !listed_members.is_empty() {
        append_unanalyzed_members(&mut compact, &listed_members);
    }

    Ok(ClassifiedReport {
        phase_ms: PhaseTimings {
            fetch_ms,
            interpret_ms,
            render_ms,
            total_ms: crate::duration_ms(classify_start.elapsed()),
        },
        pending_llm,
        classification: final_decision.class,
        probability: final_decision.probability,
        threshold: final_decision.threshold,
        level: final_decision.level,
        analysis_cached,
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
        report: compact,
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

/// Post-classification retention rubric (operator policy, 2026-08-01): a
/// stored report keeps a member node only when someone will plausibly read it
/// again. A node survives when it is
///
/// 1. the root (depth 0 / id 0),
/// 2. among the top 3 nodes by `risk` in this report,
/// 3. carrying its own crit ≥ 3 (notable+) trait,
/// 4. cited by any crit ≥ 3 finding's `from` list (composite contributors), or
/// 5. provenance skeleton: a `registry` record or a fetch-edge placeholder
///    (empty type) — dependency provenance sidecars reference these by
///    `file_id`, so they must not dangle.
///
/// A node failing every rule but carrying identity claims (`ident`: a PE
/// version resource, a bundle manifest, a signer) is stripped, not dropped:
/// its analysis payload — `facts`/`ctx`/`traits`, the bytes retention exists
/// to shed — is cleared, and the listing skeleton plus `ident` survives in a
/// few hundred bytes. hopper projects those claims into queryable identity
/// columns, and a member dropped here never becomes a hopper row at all.
/// Capped at [`MAX_IDENTITY_ONLY_NODES`] — it is the one rule whose survivor
/// count an attacker controls directly.
///
/// Everything else is removed outright — no stub rows. On the benchmark
/// corpus that is ~77% of nodes and ~56% of stored report bytes, almost all
/// of it `facts`/`ctx` that only trait evaluation and featurization (both
/// already complete) ever consume. Collimator's retraining featurizer reads
/// `facts` from the nodes that remain. `SCAN_KEEP_ALL_MEMBERS=1` disables
/// the rubric for debugging or corpus captures.
/// Ceiling on identity-only survivors, the one retention rule an attacker can
/// drive per-member: `ident` comes from the sample itself (a PE version
/// resource, a bundle manifest), so an archive of a million near-identical
/// tiny signed-looking members — which compresses to almost nothing — would
/// otherwise turn a rubric meant to *shed* stored bytes into an amplifier.
/// Every other rule is bounded by interestingness (top-3, cited-by-finding,
/// notable trait). Far above any real installer; only a hostile archive or a
/// pathological corpus sample reaches it.
const MAX_IDENTITY_ONLY_NODES: usize = 4096;

pub(crate) fn apply_report_retention(report: &mut cleave::types::CompactReport) {
    if std::env::var("SCAN_KEEP_ALL_MEMBERS").as_deref() == Ok("1") {
        return;
    }
    let files = &mut report.files;

    // Contributors cited by any notable+ finding, across all nodes.
    let mut cited: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for f in files.iter() {
        for t in f.findings.iter().filter(|t| t.criticality >= 3) {
            for r in &t.from {
                cited.insert(r.file);
            }
        }
    }
    // Top 3 by risk (ties keep earlier nodes — stable sort on a stable list).
    let mut by_risk: Vec<(u32, i64)> = files.iter().map(|f| (f.id, f.risk)).collect();
    by_risk.sort_by_key(|&(_, risk)| std::cmp::Reverse(risk));
    let top3: std::collections::HashSet<u32> = by_risk.iter().take(3).map(|&(id, _)| id).collect();

    let mut identity_only = 0usize;
    files.retain_mut(|f| {
        if f.depth == 0
            || f.id == 0
            || top3.contains(&f.id)
            || cited.contains(&f.id)
            || f.file_type.is_empty()
            || f.file_type == "registry"
            || f.findings.iter().any(|t| t.criticality >= 3)
        {
            return true;
        }
        if f.identity.as_ref().is_some_and(|i| !i.is_empty()) {
            identity_only += 1;
            if identity_only > MAX_IDENTITY_ONLY_NODES {
                return false;
            }
            f.formula = None;
            f.findings = Vec::new();
            f.refs = Vec::new();
            f.context = Vec::new();
            f.facts = cleave::types::CompactFacts::default();
            return true;
        }
        false
    });
    if identity_only > MAX_IDENTITY_ONLY_NODES {
        tracing::warn!(
            kept = MAX_IDENTITY_ONLY_NODES,
            dropped = identity_only - MAX_IDENTITY_ONLY_NODES,
            "report retention: identity-only members over cap; report truncated"
        );
    }
}

/// Sentinel `risk` for an archive member scan listed but never analyzed. Keeps
/// an unanalyzed listing (-1) distinguishable from an analyzed member that
/// simply produced no traits (0), so no consumer mistakes silence for a clean
/// result.
pub(crate) const UNANALYZED_MEMBER_RISK: i64 = -1;

/// Append the archive members cleave never analyzed to `report_json["files"]` as
/// listing-only entries (`id`/`path`/`type`/`sha`/`size`/`depth`). Members that
/// were analyzed — matched by sha256 — are already present and skipped. Each
/// appended entry carries a sentinel `risk` of -1 so a consumer can tell an
/// unanalyzed listing (-1) apart from an analyzed member that simply produced no
/// traits (0); [`build_ml_files`] reads the same sentinel to keep these out of
/// the classified `ml.files` array.
fn append_unanalyzed_members(
    report: &mut cleave::types::CompactReport,
    members: &[ArchiveMemberStub],
) {
    let files = &mut report.files;
    let mut seen: std::collections::HashSet<String> = files.iter().map(|f| f.sha.clone()).collect();
    // Compact paths join nesting levels with `!!` under the root file's path;
    // `archive_contents` paths are archive-relative with single `!`. Re-root and
    // normalize so the listing entries match their analyzed siblings.
    let root_path = files.first().map(|f| f.path.clone()).unwrap_or_default();
    let mut next_id = files.iter().map(|f| f.id).max().map_or(0, |m| m + 1);
    for m in members {
        if m.sha256.is_empty() || !seen.insert(m.sha256.clone()) {
            continue;
        }
        files.push(cleave::types::CompactFile {
            id: next_id,
            path: format!("{root_path}!!{}", m.path.replace('!', "!!")),
            file_type: m.file_type.clone(),
            sha: m.sha256.clone(),
            size: m.size_bytes,
            risk: UNANALYZED_MEMBER_RISK,
            depth: u32::try_from(m.path.matches('!').count() + 1).unwrap_or(u32::MAX),
            ..cleave::types::CompactFile::default()
        });
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
    source_len: u64,
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
fn inject_dependency_backref(report: &mut cleave::types::CompactReport, backref: &DepBackref) {
    let (crit, sev) = match backref.class {
        Classification::Hostile => (5u8, "Malicious"),
        _ => (4u8, "Suspicious"),
    };
    let desc = format!(
        "{sev} dependency: {} | {}",
        backref.locator, backref.dep_sha
    );
    let dep = cleave::types::CompactDep {
        locator: backref.locator.clone(),
        sha: backref.dep_sha.clone(),
        file_type: backref.dep_type.clone(),
    };
    let span = [backref.source_offset.unwrap_or(0), backref.source_len];

    // The declaring file's compact path locates every container above it.
    let decl_path = report
        .files
        .iter()
        .find(|f| f.sha == backref.source_sha)
        .map(|f| f.path.clone());

    for f in &mut report.files {
        let is_decl = f.sha == backref.source_sha;
        let is_ancestor = decl_path
            .as_deref()
            .is_some_and(|dp| dp.starts_with(&format!("{}!!", f.path)));
        if !is_decl && !is_ancestor {
            continue;
        }
        f.findings.push(cleave::types::CompactTrait {
            id: DEP_VERDICT_TRAIT_ID.to_string(),
            criticality: crit,
            description: desc.clone(),
            dep: Some(dep.clone()),
            // The precise declaring file cites the reference byte span; a
            // rolled-up ancestor carries the verdict without a (meaningless)
            // cross-file span.
            ev: if is_decl { vec![span] } else { Vec::new() },
            ..cleave::types::CompactTrait::default()
        });
    }
}

/// Trait id for the synthetic finding [`inject_dependency_backref`] pins onto a
/// manifest that declared a hostile or suspicious dependency.
const DEP_VERDICT_TRAIT_ID: &str = "fetch/dependency-verdict";

/// Cap on elevated-finding lines rendered per fetched dependency in the LLM
/// context appendix; a dependency with more still shows its worst, and the
/// omission is stated so the model never mistakes the cut for completeness.
const MAX_DEP_FINDING_LINES: usize = 12;

/// Raw provider documents smaller than this stay verbatim in interpret
/// provenance. Larger registry responses (notably npm packuments) are projected
/// structurally around the requested version instead of byte-truncated, keeping
/// valid JSON and the provider-only fields an LLM may need.
const MAX_INTERPRET_RAW_REGISTRY_BYTES: usize = 4 * 1024;

/// Dependency subjects rendered for the LLM, worst first.
///
/// Every subject that clears the gate costs a provenance line plus a cleave
/// context block — around 4 KB each. A container image dependency-closure
/// produces hundreds: the render of `library/kibana:9.4.0` ran to 1.21 MB, of
/// which 983 KB (81%) was 257 dependency subjects, and the endpoint rejected the
/// whole prompt with a 400 for exceeding its context window. The artifact's own
/// contents were only 211 KB. Two subjects keep the evidence that changes a
/// verdict — a hostile or suspicious dependency — while the rest are accounted
/// for by `deps_omitted`, which the model already sees.
const MAX_INTERPRET_DEP_SUBJECTS: usize = 2;

/// Sort key for [`MAX_INTERPRET_DEP_SUBJECTS`], most risk first.
///
/// A dependency is ranked by the verdict scan already computed for it, on the
/// same envelope every other verdict in this system uses: `level` is the lowest
/// false-positive budget (FP per 100M benigns) at which its hostile decision
/// still fires, so a *lower* level is the more confident call and `-1` never
/// fires at any level. Probability breaks ties within a level.
///
/// Dependencies the embedded pass never graded carry no decision to rank
/// (`DepResult::verdict` is `None` by design rather than fabricated), so a
/// suspicious-or-worse member trait is the only risk signal they have, and a
/// subject admitted on registry provenance alone has none.
fn dep_subject_risk(verdict: Option<Decision>, severe_finding: bool) -> (u8, i32, u32) {
    // A probability is finite and non-negative, and IEEE-754 orders such floats
    // identically to their bit patterns — an exact tiebreaker without a lossy
    // cast, and one that keeps the whole key `Ord`.
    let prob_key = |d: Decision| d.probability.max(0.0).to_bits();
    match verdict {
        // Negated so a level of 0 (fires even at the strictest budget) outranks
        // a level of 3000 (fires only when 3000 FP per 100M is acceptable).
        Some(d) if d.level.is_some_and(|l| l >= 0) => {
            (3, -d.level.unwrap_or_default(), prob_key(d))
        }
        _ if severe_finding => (2, 0, 0),
        Some(d) => (1, 0, prob_key(d)),
        None => (0, 0, 0),
    }
}

/// Upper bound, in bytes, on the PRIMARY subject's rendered context before its
/// weakest members are dropped from the prompt.
///
/// The render is one prompt, and the GPU pays for every token of it. Measured
/// on the fleet's vLLM over 31k requests (2026-09-05): prompts over 20k tokens
/// were 5.6% of requests and ~31% of all prefill, and the outliers were
/// structural — a jar inside a wheel fanning out to 83 `.class` members, a
/// lalsuite wheel with 163 Mach-O members each carrying a hex window — not
/// evidence anyone reads. The render tokenizes at ~2 bytes/token on those
/// (hex rows and JSON), so 96 KiB is roughly 32–48k tokens, past the p95 of
/// what the model was being asked to read and under the point where one
/// request holds the KV cache against everyone else.
///
/// `SCAN_INTERPRET_BUDGET_BYTES` overrides it; `0` disables the cap.
pub(crate) const INTERPRET_PRIMARY_BUDGET_BYTES: usize = 96 * 1024;

fn interpret_primary_budget() -> usize {
    std::env::var("SCAN_INTERPRET_BUDGET_BYTES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(INTERPRET_PRIMARY_BUDGET_BYTES)
}

/// Render `primary` for the LLM, dropping its weakest members until the render
/// fits in `budget` bytes. Returns the render and how many members went.
///
/// What may go: a member (never the root) none of whose findings reached
/// suspicious. Everything at or above that line is the evidence the grade rests
/// on — `docs/interpret-tuning.md`: cut hinting and metadata, never evidence —
/// and stays whatever it costs, so an archive that is *all* suspicious members
/// is rendered whole. Among the droppable, the ones with the least to say go
/// first: lowest peak criticality, then fewest findings, discovery order among
/// equals. A dropped member takes its own nested members with it, so the render
/// never shows a child under a container it no longer lists.
///
/// `budget == 0` disables the cap.
fn budget_primary_context(primary: &mut cleave::AnalysisReport, budget: usize) -> (String, usize) {
    use cleave::output::{TinyOpts, format_context};

    let mut rendered = format_context(primary, &TinyOpts::tiny());
    if budget == 0 || rendered.len() <= budget {
        return (rendered, 0);
    }
    let mut order: Vec<((cleave::Criticality, usize), u32)> = primary
        .files
        .iter()
        .filter(|f| f.depth > 0 && f.id != 0)
        .filter(|f| {
            f.findings
                .iter()
                .all(|t| t.crit < cleave::Criticality::Suspicious)
        })
        .map(|f| {
            let peak = f.findings.iter().map(|t| t.crit).max().unwrap_or_default();
            ((peak, f.findings.len()), f.id)
        })
        .collect();
    order.sort_by_key(|(key, _)| *key);

    let mut dropped = 0usize;
    while rendered.len() > budget && !order.is_empty() {
        // Drop in proportion to the overshoot, so a 200-member archive converges
        // in a handful of renders rather than two hundred.
        let per_file = rendered.len() / primary.files.len().max(1);
        let n = ((rendered.len() - budget) / per_file.max(1)).clamp(1, order.len());
        let mut victims: std::collections::HashSet<u32> =
            order.drain(..n).map(|(_, id)| id).collect();
        // Take nested members of a victim along, however deep.
        loop {
            let before = victims.len();
            for f in &primary.files {
                if f.parent_id.is_some_and(|p| victims.contains(&p)) {
                    victims.insert(f.id);
                }
            }
            if victims.len() == before {
                break;
            }
        }
        order.retain(|(_, id)| !victims.contains(id));
        let before = primary.files.len();
        primary.files.retain(|f| !victims.contains(&f.id));
        dropped += before - primary.files.len();
        rendered = format_context(primary, &TinyOpts::tiny());
    }
    (rendered, dropped)
}

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
    let (primary_context, members_omitted) =
        budget_primary_context(&mut primary, interpret_primary_budget());
    if members_omitted > 0 {
        tracing::info!(
            label,
            members_omitted,
            budget_bytes = interpret_primary_budget(),
            "interpret render over budget; weakest members dropped"
        );
    }
    let _ = writeln!(out, "== PRIMARY {label} ==");
    let now_secs = scan_now_secs();
    let primary_provenance = slim_provenance_for_interpret(
        primary_provenance(label, sha256, root_fetch, root_registry, &mut raw_seen),
        now_secs,
    );
    let primary_provenance_line =
        serde_json::to_string(&primary_provenance).unwrap_or_else(|_| "{}".to_string());
    let _ = writeln!(out, "provenance={primary_provenance_line}");
    out.push_str(&primary_context);
    if members_omitted > 0 {
        let _ = writeln!(out, "members_omitted={members_omitted} (prompt budget)");
    }
    // Restated after the findings. One `provenance=` line at the head of a render
    // that runs to tens of KB is not read: the grader infers identity from
    // filenames and source instead, and the registry's claim never enters the
    // judgment at all. The same bytes at the tail are read — measured on a
    // mislabelled sample, where head-only provenance graded benign ("standard LDAP
    // gem", describing the code and ignoring the claim) and the tail restatement
    // caught it ("registry mismatch").
    // Only the registry record is restated, never the whole block: `raw` is the
    // bulk of it and repeating that would cost more than the restatement buys.
    // The claim is what the grader needs a second look at.
    let identity = primary_provenance
        .get("registry")
        .and_then(|registry| registry.get("record"))
        .filter(|record| record.as_object().is_some_and(|r| !r.is_empty()));
    if let Some(identity) = identity {
        let line = serde_json::to_string(&serde_json::json!({"registry": {"record": identity}}))
            .unwrap_or_else(|_| "{}".to_string());
        let _ = writeln!(
            out,
            "\n== SUBJECT IDENTITY (registry claim for PRIMARY) ==\nprovenance={line}"
        );
    }

    let mut shown_registry_ids = std::collections::HashSet::new();
    let mut visited_roots = std::collections::HashSet::new();
    let mut candidate_count = 0_usize;
    // Subjects are buffered rather than appended, so the worst
    // `MAX_INTERPRET_DEP_SUBJECTS` can be chosen once both loops have run.
    let mut subjects: Vec<((u8, i32, u32), String)> = Vec::new();
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

        let mut out = String::new();
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
        let provenance = slim_provenance_for_interpret(
            dependency_provenance(rec, registry, report, &mut raw_seen),
            now_secs,
        );
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
        subjects.push((
            dep_subject_risk(graded.and_then(|d| d.verdict), severe_finding),
            out,
        ));
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
        let mut out = String::new();
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
        // No artifact was analyzed, so there is no verdict and no member trait.
        subjects.push((dep_subject_risk(None, false), out));
    }

    // Worst first, and a stable sort so equal-risk subjects keep discovery
    // order. Only the top few are spent on: the render is one prompt, and an
    // over-budget prompt is refused whole rather than truncated by the endpoint.
    subjects.sort_by_key(|(risk, _)| std::cmp::Reverse(*risk));
    let shown_count = subjects.len().min(MAX_INTERPRET_DEP_SUBJECTS);
    for (_, block) in subjects.iter().take(MAX_INTERPRET_DEP_SUBJECTS) {
        out.push_str(block);
    }

    let omitted = candidate_count.saturating_sub(shown_count);
    if omitted > 0 {
        let _ = writeln!(out, "\ndeps_omitted={omitted}");
    }
    out
}

/// Rewrite each finding annotation from a graded conclusion into a categorized
/// observation, for the LLM view only.
///
/// cleave announces a finding as `# SEV LOC desc (trait::id)`. Both the severity
/// letter and the prose are the analyzer's *answer*, and handing the answer to a
/// second opinion asked to check it produces agreement, not review: the model
/// summarizes the highest-severity assertion instead of reading the bytes under
/// it. Measured on the poppy/gauntlet false positives, a .NET single-file bundle
/// whose overlay carries the CLR graded hostile under every prompt, provenance
/// and carve-out we tried, and benign as soon as the annotation stopped asserting
/// "process-hollowing API chain" outright.
///
/// The rewrite keeps the description and drops the grade:
///
/// ```text
/// // H Dynamically resolved process-hollowing API chain (objectives/evasion/process/injection/hollowing::…)
/// // Possible evasion/process — Dynamically resolved process-hollowing API chain
/// ```
///
/// Dropping the description instead was also measured, and costs recall exactly
/// where `docs/interpret-tuning.md` predicts: on packed binaries the prose is the
/// only readable signal, and a real dropper went benign without it. So the prose
/// stays and only its authority is removed.
///
/// The terminal view is untouched — this is the machine/LLM render alone.
pub(crate) fn recategorize_annotations(rendered: &str) -> String {
    let mut out = String::with_capacity(rendered.len());
    for line in rendered.split_inclusive('\n') {
        match recategorize_annotation(line.trim_end_matches('\n')) {
            Some(rewritten) => {
                out.push_str(&rewritten);
                if line.ends_with('\n') {
                    out.push('\n');
                }
            }
            None => out.push_str(line),
        }
    }
    out
}

/// One annotation line, or `None` when the line is not one.
fn recategorize_annotation(line: &str) -> Option<String> {
    let indent_len = line.len() - line.trim_start().len();
    let (indent, rest) = line.split_at(indent_len);
    // The marker set must match `interpret::parse_annotation`, which is what
    // decides a line *is* an annotation: any marker it recognizes and this one
    // does not passes through with its grade letter intact, which is precisely
    // what recategorizing is meant to prevent.
    let (comment, rest) = ["//", "--", "#"]
        .into_iter()
        .find_map(|m| Some((m, rest.strip_prefix(m)?.strip_prefix(' ')?)))?;
    // `SEV ` — one grade letter then a space. Parsed by chars rather than byte
    // offsets: an annotation body can open with a multi-byte character, and
    // splitting at a computed byte index lands inside it and panics (observed on
    // browser-extension samples whose descriptions begin with 'ü').
    let mut head = rest.chars();
    let sev = head.next()?;
    // `C` and `F` belong here too — same reason as the marker set above.
    if !matches!(sev, 'H' | 'S' | 'N' | 'B' | 'C' | 'F') || head.next() != Some(' ') {
        return None;
    }
    // `sev` is ASCII and the char after it is a space, so this index is a
    // boundary by construction; `get` keeps that from resting on a panic.
    let rest = rest.get(sev.len_utf8() + 1..)?;
    // A third-party signature carries no prose: the id *is* the body, unwrapped —
    // `// H third_party/elastic/Linux_Trojan_Ladvix/linux/trojan/ladvix`. Naming
    // its category is the whole rewrite; there is no description to keep.
    if !rest.contains('(') && !rest.contains(char::is_whitespace) && rest.contains('/') {
        return Some(format!(
            "{indent}{comment} Possible {}",
            trait_category(rest)
        ));
    }
    // Otherwise the trait id is the trailing parenthesized group; without one
    // there is no category to name and the line is left alone.
    let open = rest.rfind(" (")?;
    // `open` indexes a two-byte ASCII " (", so both edges are boundaries by
    // construction; `get` says so without resting on a panic, the same way the
    // severity split above does.
    let inner = rest.get(open + 2..)?.strip_suffix(')')?;
    // A trait id is a slash path, optionally with a `::rule` suffix — never prose.
    // Requiring `::` alone would exempt every `third_party/` signature, which is
    // where the loudest grades live: `// H Detects Quasar RAT (third_party/…)`
    // would keep its H and reach the grader as a verdict.
    if inner.contains('(') || inner.contains(')') || inner.contains(char::is_whitespace) {
        return None;
    }
    if !inner.contains("::") && !inner.contains('/') {
        return None;
    }
    let body = rest.get(..open)?;
    // `LOC ` (a `line:col` or `@offset`) stays; it is a pointer, not a verdict.
    let (loc, desc) = match body.split_once(' ') {
        Some((head, tail))
            if head.starts_with('@')
                || head.split_once(':').is_some_and(|(a, b)| {
                    !a.is_empty()
                        && a.chars().all(|c| c.is_ascii_digit())
                        && b.chars().all(|c| c.is_ascii_digit())
                }) =>
        {
            (format!("{head} "), tail)
        }
        _ => (String::new(), body),
    };
    let category = trait_category(inner);
    Some(format!(
        "{indent}{comment} {loc}Possible {category} — {desc}"
    ))
}

/// The family a trait belongs to: the two path components below its namespace,
/// e.g. `objectives/evasion/process/injection/hollowing::x` → `evasion/process`.
/// Broad on purpose — it should place the match, not characterize it.
fn trait_category(trait_id: &str) -> String {
    let path = trait_id.split("::").next().unwrap_or(trait_id);
    let mut parts = path.split('/').filter(|p| !p.is_empty());
    let Some(namespace) = parts.next() else {
        return "pattern".to_string();
    };
    let rest: Vec<&str> = parts.collect();
    if rest.is_empty() {
        return namespace.to_string();
    }
    // `well-known/` is the exception: its depth is not a taxonomy of technique,
    // it is an *identity* — `unwanted/newtab-wallpaper-adware/owhit` names the
    // family, and the family is the whole finding. Truncating it to two
    // components throws away the only part that decides the verdict, which is how
    // a Chrome extension cleave had already recognized as newtab wallpaper adware
    // reached the grader as an unremarkable observation and was cleared.
    if namespace == "well-known" {
        return rest.join("/");
    }
    match rest.as_slice() {
        [a, b, ..] => format!("{a}/{b}"),
        [a] => (*a).to_string(),
        [] => "pattern".to_string(),
    }
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
                        .get(finding.id.as_str())
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
    let mut record = serde_json::to_value(registry).unwrap_or(serde_json::Value::Null);
    remove_empty_json(&mut record);
    out.insert("record".to_string(), record);
    if let Some(raw) = provenance.raw() {
        out.insert(
            "raw".to_string(),
            compact_registry_raw(&raw, registry, raw_seen),
        );
    }
    serde_json::Value::Object(out)
}

/// Trim a rendered provenance block down to what the grader actually reads.
///
/// The full block runs 5–9 KB per subject — on a small package that is ~74% of
/// the whole render — and almost all of it is `raw`: provider response bodies
/// and version history the LLM never uses to grade. What it does use is the
/// package's *claimed identity* (does the observed behavior have a plausible
/// role in what this package says it is for?) and the handful of signals that
/// say whether that claim is worth anything, so the record is projected to those
/// and nothing else.
///
/// `page` is recomputed from `first_published_at`/`published_at` against scan
/// time rather than read from the record's own `age_days`. An offline
/// `--registry-map` sidecar freezes `age_days` at *collection* time, and a
/// corpus collected within days of publication carries `0` for everything —
/// which would tell the grader that every long-established package is brand
/// new, exactly inverting the signal. `downloads_recent` is likewise dropped
/// when zero rather than reported as zero, since the sidecar path leaves it
/// unset rather than measured.
fn slim_provenance_for_interpret(
    mut provenance: serde_json::Value,
    now_secs: i64,
) -> serde_json::Value {
    let Some(registry) = provenance
        .get_mut("registry")
        .and_then(|r| r.as_object_mut())
    else {
        return provenance;
    };
    // The provider documents are dropped and the record is projected. `raw` was
    // kept until 2026-09-05 so that provider-only fields could reach the grader;
    // measured over 113 rendered PURLs it was 30% of every prompt token — a
    // packument projection of `time`, `dist`, `versions`, `_rev` — while every
    // identity signal it carried is summarised by the record fletch derives
    // from it. An A/B on the shipped prompt with `raw` removed graded the same
    // samples identically (`hacks/interpret-tune/tune.py --templates noraw`).
    registry.remove("raw");
    if let Some(record) = registry.get("record") {
        let slim = project_registry_record(record, now_secs);
        registry.insert("record".to_string(), slim);
    }
    provenance
}

/// The package's registry identity, projected for the grader.
///
/// Everything fletch knows about *who published this and whether anyone uses
/// it*: name, title and description, publisher and maintainers, repository,
/// download counts, package and version age, release cadence, and the flags a
/// registry raises on a package it has already acted on. These are what let a
/// model recognise a typosquat, a dependency-confusion placeholder, or a
/// hijacked publisher — the supply-chain cases our rules cannot enumerate —
/// so the projection keeps all of them, in words rather than single letters,
/// at a few dozen tokens. The provider documents behind them (`raw`) are not
/// kept: measured 2026-09-05 over 113 rendered PURLs they were 30% of every
/// prompt token and carried nothing the record does not summarise.
///
/// Reads the record through its serialized keys rather than the `fletch::Registry`
/// struct: these are the wire names every consumer of the envelope already sees,
/// and ecosystems populate different subsets of them (a gem record carries
/// downloads but no release count; a pypi record carries both). Absent and
/// empty values are omitted; a zero download count is kept, because zero is
/// the signal.
fn project_registry_record(record: &serde_json::Value, now_secs: i64) -> serde_json::Value {
    let get_str = |key: &str| {
        record
            .get(key)
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty())
    };
    let get_i64 = |key: &str| record.get(key).and_then(serde_json::Value::as_i64);
    let get_true = |key: &str| record.get(key).and_then(serde_json::Value::as_bool) == Some(true);
    let truncate = |s: &str, max: usize| {
        if s.chars().count() <= max {
            s.to_string()
        } else {
            let head: String = s.chars().take(max).collect();
            format!("{head}…")
        }
    };

    let mut out = serde_json::Map::new();
    let mut put = |key: &str, value: serde_json::Value| {
        out.insert(key.to_string(), value);
    };
    let version = get_str("version").unwrap_or("");
    if let Some(name) = get_str("name") {
        let eco = get_str("ecosystem").unwrap_or("");
        let mut id = String::new();
        if !eco.is_empty() {
            id.push_str(eco);
            id.push('/');
        }
        id.push_str(name);
        if !version.is_empty() {
            id.push('@');
            id.push_str(version);
        }
        put("package", serde_json::json!(id));
    }
    for (key, max) in [("title", 120), ("description", 300)] {
        if let Some(text) = get_str(key) {
            put(key, serde_json::json!(truncate(text, max)));
        }
    }
    // Who. `publisher` is the account that pushed this version; `author` is
    // whatever the manifest claims. Both matter when they disagree.
    for (key, max) in [
        ("publisher", 60),
        ("author", 60),
        ("publisher_email_domain", 80),
    ] {
        if let Some(text) = get_str(key) {
            put(key, serde_json::json!(truncate(text, max)));
        }
    }
    if let Some(n) = get_i64("maintainers") {
        put("maintainers", serde_json::json!(n));
    }
    if get_true("publisher_in_maintainers") {
        put("publisher_in_maintainers", serde_json::json!(true));
    }
    if get_true("publisher_verified") {
        put("publisher_verified", serde_json::json!(true));
    }
    // Where it claims to come from.
    for key in ["repository", "homepage", "license"] {
        if let Some(text) = get_str(key) {
            put(key, serde_json::json!(truncate(text, 200)));
        }
    }
    // Whether anyone uses it.
    for key in ["downloads_total", "downloads_recent"] {
        if let Some(n) = get_i64(key) {
            put(key, serde_json::json!(n));
        }
    }
    // Two distinct ages, both measured at scan time. `package_age_days` is how
    // long the *package* has existed and is the credibility signal;
    // `version_age_days` is how long this *version* has, a freshness signal.
    // Folding them into one would report a decade-old gem as days old whenever
    // it had just cut a release — inverting exactly the signal that matters.
    let days_since = |ts: i64| (now_secs > ts).then(|| (now_secs - ts) / 86_400);
    if let Some(days) = get_i64("first_published_at").and_then(days_since) {
        put("package_age_days", serde_json::json!(days));
    }
    if let Some(days) = get_i64("published_at").and_then(days_since) {
        put("version_age_days", serde_json::json!(days));
    }
    if let Some(days) = get_i64("previous_published_at").and_then(days_since) {
        put("previous_release_age_days", serde_json::json!(days));
    }
    for key in ["release_count", "releases_24h", "releases_48h"] {
        if let Some(n) = get_i64(key) {
            put(key, serde_json::json!(n));
        }
    }
    if let Some(latest) = get_str("latest_version") {
        put("latest_version", serde_json::json!(latest));
        if !version.is_empty() && latest != version {
            put("is_latest", serde_json::json!(false));
        }
    }
    // What the registry itself has flagged. Only when true: a sea of `false`
    // is noise, and an absent flag reads the same as a false one.
    for key in [
        "has_install_script",
        "security_hold",
        "version_removed",
        "deprecated",
    ] {
        if get_true(key) {
            put(key, serde_json::json!(true));
        }
    }
    if let Some(text) = get_str("deprecated") {
        put("deprecated", serde_json::json!(truncate(text, 120)));
    }
    for key in ["unpacked_size", "file_count"] {
        if let Some(n) = get_i64(key).filter(|n| *n > 0) {
            put(key, serde_json::json!(n));
        }
    }
    if let Some(vulns) = get_i64("vulnerability_count").filter(|v| *v > 0) {
        put("vulnerability_count", serde_json::json!(vulns));
    }
    serde_json::Value::Object(out)
}

/// Seconds since the Unix epoch, or `0` when the clock is before it.
fn scan_now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
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
    let mut out = String::new();
    struct FetchBranch {
        verdict: Option<(Classification, f32)>,
        subject: String,
        source: String,
        body: String,
    }
    let mut branches = Vec::new();

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
        let graded = deps.iter().find(|d| d.sha256 == content_sha);
        let registry = registries.iter().find(|r| r.locator == rec.locator);
        let member_ids: std::collections::HashSet<u32> = fetched_root_by_file
            .iter()
            .filter_map(|(&file, &fetched_root)| (fetched_root == root.id).then_some(file))
            .collect();
        // Only hostile dependencies earn a provenance block: a benign-but-old or
        // merely-notable dependency is exactly the noise that buried the real
        // ones. Hostility is a hostile verdict on the fetched bytes, or a hostile
        // finding on one of its members.
        let hostile_finding = report
            .files
            .iter()
            .filter(|file| member_ids.contains(&file.id))
            .flat_map(|file| &file.findings)
            .any(|finding| finding.crit >= cleave::Criticality::Hostile);
        let hostile_verdict = graded
            .and_then(|d| d.verdict)
            .is_some_and(|verdict| verdict.class == Classification::Hostile);
        if !hostile_finding && !hostile_verdict {
            continue;
        }

        let subject = if rec.kind == fletch::RefKind::Dependency {
            "dependency"
        } else {
            "external URL"
        };
        let verdict = graded
            .and_then(|d| d.verdict)
            .map(|verdict| (verdict.class, verdict.probability));
        let source = terminal_fetch_source(rec, report);
        let mut body = String::new();
        let _ = writeln!(
            body,
            "{}",
            crate::output::terminal_reference_locator_row(&rec.locator)
        );
        write_terminal_fetch_redirects(&mut body, rec);
        let _ = writeln!(
            body,
            "{}",
            crate::output::terminal_reference_hash_row(content_sha, &root.file_type, root.size,)
        );
        let mut signals = Vec::new();
        if rec.pin_verified == Some(false) {
            signals.push("checksum mismatch");
        }
        if let Some(registry) = registry {
            write_terminal_registry_provenance(&mut body, registry, &mut signals, true);
            shown_registry_ids.insert(registry.file_id);
        } else if !signals.is_empty() {
            let _ = writeln!(body, " \u{00b7}   signals  {}", signals.join(" · "));
        }

        let mut view = report.clone();
        view.files.retain(|file| {
            member_ids.contains(&file.id) || registry.is_some_and(|r| r.file_id == file.id)
        });
        let traits = terminal_top_traits(&view);
        let rows = crate::output::terminal_trait_rows(
            &traits,
            cleave::output::terminal_width().saturating_sub(3),
        );
        if !rows.is_empty() {
            for row in rows.lines() {
                let _ = writeln!(body, "{row}");
            }
        }
        branches.push(FetchBranch {
            verdict,
            subject: subject.to_string(),
            source,
            body,
        });
    }

    for (index, branch) in branches.iter().enumerate() {
        if index == 0 {
            out.push('\n');
        }
        out.push_str(&crate::output::terminal_reference_branch(
            branch
                .verdict
                .as_ref()
                .map(|(classification, probability)| (classification, *probability)),
            &branch.subject,
            &branch.source,
            &branch.body,
            index + 1 == branches.len(),
        ));
    }

    for registry in registries {
        if shown_registry_ids.contains(&registry.file_id)
            || fetch_edges
                .iter()
                .any(|edge| edge.locator == registry.locator && edge.content_sha256.is_some())
        {
            continue;
        }
        // A registry-only entry (no bytes fetched, so no verdict) is shown only
        // when the registry itself flags it hostile — a pulled version or a
        // security hold. "Older than fetch age limit" and other benign states are
        // not findings; they were the bulk of the old noise.
        let record = &registry.provenance.record;
        let hostile_signal =
            record.version_removed == Some(true) || record.security_hold == Some(true);
        if !hostile_signal {
            continue;
        }
        let status = registry
            .artifact_skip
            .unwrap_or("REGISTRY ONLY")
            .to_uppercase();
        let _ = writeln!(
            out,
            "\n{}",
            crate::output::terminal_reference_status_heading(&status, "REGISTRY")
        );
        let _ = writeln!(
            out,
            "{}",
            crate::output::terminal_reference_locator(&registry.locator)
        );
        let mut signals = Vec::new();
        write_terminal_registry_provenance(&mut out, registry, &mut signals, false);
        let mut view = report.clone();
        view.files.retain(|file| file.id == registry.file_id);
        let traits = terminal_top_traits(&view);
        let rows = crate::output::terminal_trait_rows(
            &traits,
            cleave::output::terminal_width().saturating_sub(3),
        );
        if !rows.is_empty() {
            for row in rows.lines() {
                let _ = writeln!(out, "   {row}");
            }
        }
    }

    // Nothing is footnoted for the artifacts that stayed quiet. Their count
    // says nothing at the decision point — the streamed fetch summary already
    // reports how many were retrieved — and a section that ends by naming what
    // it declined to show reads as withheld evidence rather than a clean pass.
    (!out.is_empty()).then_some(out)
}

fn terminal_fetch_source(
    rec: &fletch::fetch::FetchRecord,
    report: &cleave::AnalysisReport,
) -> String {
    let source_file = report
        .files
        .iter()
        .find(|file| file.sha256 == rec.source_sha256);
    source_file.map_or_else(
        || "<unknown>".to_string(),
        |file| {
            if file.depth == 0 {
                "this file".to_string()
            } else {
                terminal_finding_path(&file.path)
            }
        },
    )
}

fn write_terminal_fetch_redirects(out: &mut String, rec: &fletch::fetch::FetchRecord) {
    use std::fmt::Write as _;

    if !rec.resolved_url.is_empty() && rec.resolved_url != rec.locator {
        let _ = writeln!(out, " \u{00b7}   resolved  {}", rec.resolved_url);
    }
    if let Some(final_url) = rec
        .final_url
        .as_deref()
        .filter(|url| *url != rec.resolved_url && *url != rec.locator)
    {
        let _ = writeln!(out, " \u{00b7}   final  {final_url}");
    }
}

fn write_terminal_registry_provenance(
    out: &mut String,
    registry: &crate::fetch::DependencyRegistry,
    signals: &mut Vec<&'static str>,
    marker_rows: bool,
) {
    use std::fmt::Write as _;

    let record = &registry.provenance.record;
    let prefix = if marker_rows { " \u{00b7}   " } else { "    " };
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
        let _ = writeln!(out, "{prefix}registry  {}", summary.join(" · "));
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
        let _ = writeln!(out, "{prefix}deprecated  {deprecated}");
    }
    if !signals.is_empty() {
        let _ = writeln!(out, "{prefix}signals  {}", signals.join(" · "));
    }
    if let Some(repository) = record.repository.as_deref() {
        let _ = writeln!(out, "{prefix}upstream  {repository}");
    }
    for url in registry.provenance.source_urls() {
        let _ = writeln!(out, "{prefix}metadata  {url}");
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
mod dep_subject_risk_tests {
    use super::*;

    fn decision(level: i32, probability: f32) -> Decision {
        Decision {
            class: Classification::Hostile,
            probability,
            threshold: 0.5,
            level: Some(level),
        }
    }

    /// The whole point of the ordering: `level` is a false-positive budget, so
    /// the dependency that fires at the *strictest* budget is the riskiest.
    #[test]
    fn lower_level_outranks_higher_level() {
        assert!(
            dep_subject_risk(Some(decision(0, 0.99)), false)
                > dep_subject_risk(Some(decision(3000, 0.99)), false)
        );
    }

    #[test]
    fn probability_breaks_ties_within_a_level() {
        assert!(
            dep_subject_risk(Some(decision(25, 0.97)), false)
                > dep_subject_risk(Some(decision(25, 0.96)), false)
        );
    }

    /// A level of -1 never fires, so a graded-clean dependency must rank below
    /// an ungraded one carrying a suspicious-or-worse member trait.
    #[test]
    fn clean_verdict_ranks_below_a_severe_finding() {
        assert!(dep_subject_risk(None, true) > dep_subject_risk(Some(decision(-1, 0.01)), false));
    }

    /// Registry-provenance-only subjects have no risk signal at all.
    #[test]
    fn ungraded_and_unremarkable_ranks_last() {
        assert!(dep_subject_risk(Some(decision(-1, 0.01)), false) > dep_subject_risk(None, false));
        assert!(dep_subject_risk(None, true) > dep_subject_risk(None, false));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod card_render_tests {
    use super::*;

    fn terminal_report(value: serde_json::Value) -> cleave::AnalysisReport {
        serde_json::from_value(value).expect("valid terminal report fixture")
    }

    #[test]
    fn terminal_hostile_trait_stops_global_selection() {
        let report = terminal_report(serde_json::json!({
            "version": "3",
            "files": [
                {
                    "id": 0, "path": "sample.zip", "depth": 0,
                    "file_type": "zip", "sha256": "root", "size": 10,
                    "findings": [
                        {"id": "objectives/dropper::one", "desc": "Archive dropper", "conf": 0.99, "crit": "hostile"},
                        {"id": "objectives/member::copy", "desc": "Inherited copy", "conf": 1.0, "crit": "hostile", "src": 1}
                    ],
                    "composite_sources": {
                        "objectives/dropper::one": [{"file": 1, "line": 42}]
                    }
                },
                {
                    "id": 1, "path": "sample.zip!!nested/agent.py", "depth": 1,
                    "file_type": "python", "sha256": "member", "size": 8,
                    "findings": [
                        {"id": "objectives/member::copy", "desc": "Remote terminal agent", "conf": 0.98, "crit": "hostile"},
                        {"id": "micro-behaviors/evasion::one", "desc": "Hides execution", "conf": 0.9, "crit": "suspicious"},
                        {"id": "metadata/package::one", "desc": "Routine package metadata", "conf": 1.0, "crit": "notable"}
                    ]
                }
            ]
        }));

        let traits = terminal_top_traits(&report);
        assert_eq!(traits.len(), 1);
        assert_eq!(traits[0].description, "Archive dropper");
        assert_eq!(traits[0].location, "agent.py:42");
        assert!(traits.iter().all(|t| t.description != "Inherited copy"));
    }

    #[test]
    fn terminal_non_hostile_traits_still_fill_three_rows() {
        let report = terminal_report(serde_json::json!({
            "version": "3",
            "files": [{
                "id": 0, "path": "sample.bin", "depth": 0,
                "file_type": "binary", "sha256": "root", "size": 10,
                "findings": [
                    {"id": "capabilities/network::one", "desc": "Contacts remote host", "conf": 0.99, "crit": "suspicious"},
                    {"id": "evasion/packing::one", "desc": "Packed executable", "conf": 0.98, "crit": "suspicious"},
                    {"id": "metadata/identity::one", "desc": "Unsigned identity", "conf": 0.97, "crit": "notable"},
                    {"id": "metadata/toolchain::one", "desc": "Compiler metadata", "conf": 0.96, "crit": "notable"}
                ]
            }]
        }));

        let traits = terminal_top_traits(&report);
        assert_eq!(traits.len(), 3);
        assert_eq!(traits[0].description, "Contacts remote host");
        assert_eq!(traits[1].description, "Packed executable");
        assert_eq!(traits[2].description, "Unsigned identity");
    }

    #[test]
    fn terminal_trait_location_omits_byte_offset() {
        let report = terminal_report(serde_json::json!({
            "version": "3",
            "files": [
                {
                    "id": 0, "path": "sample.zip", "depth": 0,
                    "file_type": "zip", "sha256": "root", "size": 10,
                    "findings": [
                        {"id": "evasion/packing::one", "desc": "Packed executable", "conf": 0.99, "crit": "suspicious"}
                    ],
                    "composite_sources": {
                        "evasion/packing::one": [{"file": 1, "offset": 1114110}]
                    }
                },
                {
                    "id": 1, "path": "sample.zip!!payload.exe", "depth": 1,
                    "file_type": "pe", "sha256": "member", "size": 8
                }
            ]
        }));

        let traits = terminal_top_traits(&report);
        assert_eq!(traits.len(), 1);
        assert_eq!(traits[0].location, "payload.exe");
        assert!(!traits[0].location.contains("@0x"));
    }

    #[test]
    fn terminal_identity_prefers_a_document_title() {
        let report = terminal_report(serde_json::json!({
            "version": "3",
            "files": [{
                "id": 0, "path": "invoice.docx", "depth": 0,
                "file_type": "docx", "sha256": "root", "size": 10,
                "identity": {
                    "title": {"value": "Quarterly Results", "source": "office.title", "verified": false},
                    "producer": {"value": "Microsoft Word", "source": "office.app", "verified": false},
                    "trust": "unsigned"
                }
            }]
        }));
        assert_eq!(
            terminal_identity_summary(&report, "invoice.docx").as_deref(),
            Some("“Quarterly Results” · Microsoft Word")
        );
    }

    #[test]
    fn terminal_identity_hides_name_and_version_already_in_filename() {
        for (label, name, version) in [
            ("nordpass-1.0.2.tgz", "nordpass", "1.0.2"),
            (
                "/tmp/atomscan-2.5.0-aarch64-apple-darwin.tar.gz",
                "atomscan",
                "2.5.0-aarch64-apple-darwin",
            ),
        ] {
            let report = terminal_report(serde_json::json!({
                "version": "3",
                "files": [{
                    "id": 0, "path": label, "depth": 0,
                    "file_type": "archive", "sha256": "root", "size": 10,
                    "identity": {
                        "name": {"value": name, "source": "package.name", "verified": false},
                        "version": {"value": version, "source": "package.version", "verified": false},
                        "trust": "unsigned"
                    }
                }]
            }));
            assert_eq!(terminal_identity_summary(&report, label), None);
        }
    }

    #[test]
    fn terminal_identity_keeps_additional_producer_information() {
        let report = terminal_report(serde_json::json!({
            "version": "3",
            "files": [{
                "id": 0, "path": "agent-1.2.3.tgz", "depth": 0,
                "file_type": "npm", "sha256": "root", "size": 10,
                "identity": {
                    "name": {"value": "agent", "source": "package.name", "verified": false},
                    "version": {"value": "1.2.3", "source": "package.version", "verified": false},
                    "producer": {"value": "Example Labs", "source": "package.author", "verified": false},
                    "trust": "unsigned"
                }
            }]
        }));
        assert_eq!(
            terminal_identity_summary(&report, "agent-1.2.3.tgz").as_deref(),
            Some("agent 1.2.3 · Example Labs")
        );
    }

    #[test]
    fn collapse_decoded_dup_drops_the_repeated_member() {
        // A decoded region: the member repeats around the archive delimiter.
        assert_eq!(
            collapse_decoded_dup("root!!pkg.tar!!a/b/server.js!!a/b/server.js##unicode-escape@224"),
            "root!!pkg.tar!!a/b/server.js##unicode-escape@224"
        );
        // No `##` marker, or no doubling: returned unchanged.
        assert_eq!(
            collapse_decoded_dup("root!!pkg!!a/b.js"),
            "root!!pkg!!a/b.js"
        );
        assert_eq!(
            collapse_decoded_dup("root!!pkg!!a/b.js##base64@0"),
            "root!!pkg!!a/b.js##base64@0"
        );
        // A decoder may repeat the root itself around the delimiter.
        assert_eq!(
            collapse_decoded_dup("root.sh!!root.sh##base64@1"),
            "root.sh##base64@1"
        );
    }

    #[test]
    fn decoded_regions_use_relationship_labels() {
        assert_eq!(
            decoded_region_display_path("/tmp/sample.sh##base64@11096").as_deref(),
            Some("embedded base64 @ 11096")
        );
        assert_eq!(
            decoded_region_display_path("root.zip!!scripts/install.js##unicode-escape@20")
                .as_deref(),
            Some("install.js \u{00b7} embedded unicode escape @ 20")
        );
        assert_eq!(
            package_display_path("/tmp/sample.sh##base64@21"),
            "embedded base64 @ 21"
        );
    }

    #[test]
    fn spinner_tail_keeps_the_filename_end() {
        assert_eq!(spinner_tail("short.py", 48), "short.py");
        let long = "animica/stratum_pool/_data/aicf_rag/chunks/deeply/nested/file.json";
        let tail = spinner_tail(long, 20);
        assert!(tail.starts_with('\u{2026}'));
        assert!(tail.ends_with("file.json"));
        assert_eq!(tail.chars().count(), 20);
    }

    #[test]
    fn external_fetch_note_stays_compact_and_grammatical() {
        assert_eq!(external_fetch_note(0, 0), None);
        assert_eq!(
            external_fetch_note(1, 1).as_deref(),
            Some("fetching 1 dependency · 1 URL")
        );
        assert_eq!(
            external_fetch_note(0, 2).as_deref(),
            Some("fetching 2 URLs")
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod dependency_grading_tests {
    use super::*;

    /// The former production cutoff. Tests deliberately put evidence beyond it
    /// so reintroducing a prefix-only grading policy cannot pass unnoticed.
    const FORMER_EMBEDDED_FILE_LIMIT: usize = 100;

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

    /// A dependency is graded from its own report and every member receives a
    /// verdict, including members beyond the former production cutoff.
    #[test]
    fn grades_every_dependency_member() {
        let Some(dir) = model_bundle() else {
            eprintln!("skipping: no model bundle (set SCAN_MODELS_DIR)");
            return;
        };
        let model = Model::load(&dir, None, None).expect("load model bundle");
        let ctx = ExtractContext::new(model.spec());

        let count = FORMER_EMBEDDED_FILE_LIMIT * 2;
        let raw = dep_report(count, "");
        let (_verdict, members) =
            classify_dependency(&raw, "pkg:npm/evil@1.0.0", &ctx, &model, None)
                .expect("a well-formed dependency report must produce a verdict");
        assert_eq!(
            members.len(),
            count,
            "every dependency member must be graded"
        );
        let tail_id = u64::try_from(count).expect("test member count fits u64");
        assert!(
            members.contains_key(&tail_id),
            "the final member, beyond the former cutoff, must have a verdict"
        );
    }

    /// Exact regression for a large dependency closure: the security-held
    /// dependency sidecar comes after hundreds of ordinary dependency nodes.
    /// It must receive a verdict and become the member that elevates the parent.
    #[test]
    fn security_held_dependency_at_high_index_elevates_parent() {
        let Some(dir) = model_bundle() else {
            eprintln!("skipping: no model bundle (set SCAN_MODELS_DIR)");
            return;
        };
        let model = Model::load(&dir, None, None).expect("load model bundle");
        let ctx = ExtractContext::new(model.spec());

        // Mirrors the observed Polymarket topology: async-mutex-lock's hostile
        // registry sidecar was node 347, well beyond the former 100-node cap.
        let tail_id = 347u32;
        let files: Vec<serde_json::Value> = std::iter::once(serde_json::json!({
            "id": 0, "path": "parent.tgz", "sha": "p".repeat(64),
            "type": "npm", "size": 1, "depth": 0
        }))
        .chain((1..tail_id).map(|id| serde_json::json!({
            "id": id, "path": format!("ordinary-dependency-{id}.registry.json"),
            "sha": format!("{id:064x}"), "type": "registry", "size": 1,
            "depth": 2, "rel": "registry"
        })))
        .chain(std::iter::once(serde_json::json!({
            "id": tail_id,
            "path": "async-mutex-lock@5.3.1.registry.json",
            "sha": "a".repeat(64), "type": "registry", "size": 768,
            "depth": 2, "rel": "registry",
            "traits": [{
                "id": "objectives/supply-chain/impersonation/registry/publish::registry-takedown-security-hold",
                "crit": 5, "conf": 0.99,
                "desc": "Registry takedown marks package malicious"
            }]
        })))
        .collect();
        let report: cleave::types::CompactReport =
            serde_json::from_value(serde_json::json!({"v": "8", "files": files}))
                .expect("compact parent report");

        let entries: Vec<&cleave::types::CompactFile> = embedded_entries(&report).collect();
        let needs = ctx.raw_needs().union(crate::features::RawNeeds::all());
        let evals: MemberEvals =
            score_embedded_files(&entries, "parent.tgz", needs, &ctx, &model, None)
                .into_iter()
                .map(|member| (member.id, member))
                .collect();

        assert_eq!(evals.len(), usize::try_from(tail_id).unwrap());
        let tail = evals
            .get(&u64::from(tail_id))
            .expect("high-index security-held dependency must be graded");
        assert_eq!(tail.classification, Classification::Hostile);
        assert_eq!(
            worst_member(&evals).expect("worst member").class,
            Classification::Hostile,
            "the high-index dependency must elevate its benign parent"
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
            .find("let parsed = if fetch_edges.is_empty() {")
            .expect("root featurization call");
        let inject = src
            .find("inject_dependency_backref(&mut report_json, backref);")
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
            embedded_files: dep.members,
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
            serde_json::to_value(&dep_env.raw).unwrap(),
            serde_json::to_value(&direct_env.raw).unwrap(),
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

    /// Selection itself is exhaustive and order-independent. This pure test does
    /// not need a model bundle and guards every caller of `embedded_entries`.
    #[test]
    fn embedded_selection_has_no_count_limit() {
        let count = FORMER_EMBEDDED_FILE_LIMIT * 3;
        let members: Vec<serde_json::Value> = (0..count)
            .map(|i| serde_json::json!({"id": i + 1, "path": format!("m{i}.js"), "sha": "m".repeat(64), "type": "javascript", "size": 1, "depth": 1}))
            .collect();
        let report: cleave::types::CompactReport = serde_json::from_value(serde_json::json!({
            "v": "8",
            "files": std::iter::once(serde_json::json!({"id": 0, "path": "d.tgz", "sha": "d".repeat(64), "type": "npm", "size": 1, "depth": 0}))
                .chain(members)
                .collect::<Vec<_>>(),
        }))
        .unwrap();
        let entries: Vec<&cleave::types::CompactFile> = embedded_entries(&report).collect();
        assert_eq!(
            entries.len(),
            count,
            "every embedded node must be selected regardless of its position",
        );
        assert_eq!(
            usize::try_from(entries.last().expect("tail member").id).unwrap(),
            count
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod dep_backref_tests {
    use super::*;

    fn empty_report() -> cleave::AnalysisReport {
        serde_json::from_value(serde_json::json!({"version": "3"})).unwrap()
    }

    /// A capability finding at a chosen criticality — the only shape these
    /// rendering tests need.
    fn finding(id: &str, desc: &str, crit: cleave::Criticality) -> cleave::Finding {
        let mut f = cleave::Finding::new(
            id.to_string(),
            cleave::types::FindingKind::Capability,
            desc.to_string(),
            cleave::Finding::default().conf,
        );
        f.crit = crit;
        f
    }

    fn backref(class: Classification) -> DepBackref {
        DepBackref {
            source_sha: "s".repeat(64),
            source_offset: Some(42),
            source_len: 9,
            locator: "pkg:npm/zaboodle@1.49".to_string(),
            dep_sha: "d".repeat(64),
            dep_type: "javascript".to_string(),
            class,
        }
    }

    /// Compact report: a depth-0 archive containing the declaring manifest and
    /// an unrelated sibling.
    /// Retention decides what ships in the stored report, so this pins the
    /// rubric directly: root kept, top-3-by-risk kept, notable+ kept, and an
    /// ordinary quiet member dropped.
    #[test]
    fn retention_keeps_only_readable_nodes_from_a_real_report() {
        use cleave::types::{Criticality, FindingKind};

        let mut files = Vec::new();
        for (id, depth, risk, crit) in [
            (0u32, 0u32, 0u32, Criticality::Baseline),
            (1, 1, 99, Criticality::Baseline),
            (2, 1, 0, Criticality::Baseline),
            (3, 1, 0, Criticality::Notable),
            (4, 1, 0, Criticality::Baseline),
        ] {
            let mut fa = cleave::FileAnalysis {
                id,
                path: format!("m{id}.py"),
                file_type: "python".to_string(),
                sha256: format!("{id:064}"),
                size: 100,
                depth,
                score: risk,
                ..Default::default()
            };
            let mut f = cleave::types::Finding::new(
                "objectives/execution/shell::sh".to_string(),
                FindingKind::Capability,
                String::new(),
                0.9,
            );
            f.crit = crit;
            fa.findings.push(f);
            files.push(fa);
        }
        let mut report = cleave::types::compact_from_files(&files);
        apply_report_retention(&mut report);

        let kept: Vec<u32> = report.files.iter().map(|f| f.id).collect();
        // 0: root. 1: top-3 by risk. 3: carries a notable finding.
        // 2 and 4 are quiet, low-risk members — nobody reads them again.
        assert_eq!(kept, vec![0, 1, 2, 3], "retention kept the wrong set");
    }

    /// Members with nothing suspicious to say go first; the root and any
    /// suspicious member survive whatever the budget, and a dropped container
    /// takes its nested members with it.
    #[test]
    fn interpret_render_budget_drops_weakest_members_first() {
        use cleave::types::{Criticality, FindingKind};

        let mut files = Vec::new();
        for (id, depth, parent, crit, n) in [
            (0u32, 0u32, None, Criticality::Notable, 1usize),
            (1, 1, Some(0), Criticality::Suspicious, 1),
            (2, 1, Some(0), Criticality::Notable, 3),
            (3, 1, Some(0), Criticality::Notable, 2),
            (4, 1, Some(0), Criticality::Notable, 1),
            (5, 2, Some(4), Criticality::Notable, 4),
        ] {
            let mut fa = cleave::FileAnalysis {
                id,
                path: format!("m{id}.py"),
                file_type: "python".to_string(),
                sha256: format!("{id:064}"),
                size: 100,
                depth,
                parent_id: parent,
                ..Default::default()
            };
            for i in 0..n {
                // One id per finding: the render keeps a trait id once across
                // the whole report, so repeats would empty the members.
                let mut f = cleave::types::Finding::new(
                    format!("objectives/execution/shell::m{id}f{i}"),
                    FindingKind::Capability,
                    format!(
                        "finding {i} on member {id}, padded so the render has weight {}",
                        "x".repeat(64)
                    ),
                    0.9,
                );
                f.crit = crit;
                fa.findings.push(f);
            }
            files.push(fa);
        }
        let mut report = empty_report();
        report.files = files;

        let (full, dropped) = budget_primary_context(&mut report.clone(), 0);
        assert_eq!(dropped, 0, "budget 0 must disable the cap");
        for id in 0..6 {
            assert!(
                full.contains(&format!("m{id}.py")),
                "uncapped render lists m{id}:\n{full}"
            );
        }

        // A budget nobody can meet still keeps the root and the suspicious
        // member: those are never on the table.
        let (tight, dropped) = budget_primary_context(&mut report.clone(), 1);
        assert!(tight.contains("m0.py"), "root survives: {tight}");
        assert!(
            tight.contains("m1.py"),
            "suspicious member survives: {tight}"
        );
        assert_eq!(dropped, 4, "every droppable member went: {tight}");

        // Just over the line: the member with the least to say (one notable
        // finding) goes first, and its nested child goes with it even though
        // the child alone, with four findings, would have outranked m2 and m3.
        let over = full.len() - 1;
        let (capped, dropped) = budget_primary_context(&mut report.clone(), over);
        assert!(capped.len() <= over, "{} > {over}", capped.len());
        assert!(
            !capped.contains("m4.py"),
            "weakest member dropped first: {capped}"
        );
        assert!(
            !capped.contains("m5.py"),
            "nested member follows its container: {capped}"
        );
        assert!(
            capped.contains("m2.py") && capped.contains("m3.py"),
            "notable members kept: {capped}"
        );
        assert_eq!(dropped, 2);
    }

    /// `count_findings` reports the root file's own findings, bucketed by
    /// criticality — with everything below notable folding into `baseline`.
    #[test]
    fn finding_counts_bucket_every_criticality() {
        use cleave::types::{Criticality, FindingKind};
        let mut fa = cleave::FileAnalysis {
            id: 0,
            path: "a.py".to_string(),
            file_type: "python".to_string(),
            sha256: "a".repeat(64),
            size: 10,
            ..Default::default()
        };
        for (i, crit) in [
            Criticality::Hostile,
            Criticality::Suspicious,
            Criticality::Notable,
            Criticality::Baseline,
            Criticality::Component,
        ]
        .into_iter()
        .enumerate()
        {
            let mut f = cleave::types::Finding::new(
                format!("objectives/x/y::t{i}"),
                FindingKind::Capability,
                String::new(),
                0.9,
            );
            f.crit = crit;
            fa.findings.push(f);
        }
        let report = cleave::types::compact_from_files(&[fa]);
        assert_eq!(
            count_findings(&report),
            FindingCounts {
                hostile: 1,
                suspicious: 1,
                notable: 1,
                // Baseline and Component both fall through to `baseline`.
                baseline: 2,
            }
        );
    }

    /// A container, the manifest that declared the dependency, and an unrelated
    /// sibling — the shape `inject_dependency_backref` walks.
    fn compact_fixture() -> cleave::types::CompactReport {
        serde_json::from_value(serde_json::json!({"files": [
            {"id": 0, "size": 1, "type": "npm", "sha": "r".repeat(64), "path": "pkg.tgz",
             "traits": [{"id": "existing/trait", "crit": 1}]},
            {"id": 1, "size": 1, "depth": 1, "type": "json", "sha": "s".repeat(64), "path": "pkg.tgz!!package.json"},
            {"id": 2, "size": 1, "depth": 1, "type": "markdown", "sha": "o".repeat(64), "path": "pkg.tgz!!README.md"},
        ]}))
        .unwrap()
    }

    /// The wire form of a report, which is what the backref assertions are
    /// about: prism and hopper read these keys.
    fn wire(report: &cleave::types::CompactReport) -> serde_json::Value {
        serde_json::to_value(report).unwrap()
    }

    #[test]
    fn retention_rubric_keeps_only_readable_nodes() {
        let mut report: cleave::types::CompactReport = serde_json::from_value(serde_json::json!({"files": [
            // Root: kept (depth 0 / id 0) even with no traits.
            {"id": 0, "path": "p.tgz", "sha": "0", "size": 1, "depth": 0, "type": "npm", "risk": 1},
            // Own notable trait: kept.
            {"id": 1, "path": "p.tgz!!a.js", "sha": "1", "size": 1, "depth": 1, "type": "javascript",
             "traits": [{"id": "a", "crit": 3}]},
            // Cited by a notable composite on another node: kept.
            {"id": 2, "path": "p.tgz!!b.md", "sha": "2", "size": 1, "depth": 1, "type": "markdown",
             "traits": [{"id": "b", "crit": 1}]},
            // The citing node (notable, with from): kept.
            {"id": 3, "path": "p.tgz!!c.js", "sha": "3", "size": 1, "depth": 1, "type": "javascript",
             "traits": [{"id": "c", "crit": 4, "from": [{"file": 2}]}]},
            // Provenance skeleton: kept.
            {"id": 4, "path": "p.tgz!!reg", "sha": "4", "size": 1, "depth": 2, "type": "registry"},
            {"id": 5, "path": "p.tgz!!edge", "sha": "5", "size": 1, "depth": 2, "type": "", "rel": "fetched"},
            // High risk: kept via top-3 (risks 9 > root's 1).
            {"id": 6, "path": "p.tgz!!e.txt", "sha": "6", "size": 1, "depth": 1, "type": "text", "risk": 9},
            // Sub-notable, uncited, low-risk: dropped.
            {"id": 7, "path": "p.tgz!!f.md", "sha": "7", "size": 1, "depth": 1, "type": "markdown",
             "traits": [{"id": "d", "crit": 2}]},
            {"id": 8, "path": "p.tgz!!g.txt", "sha": "8", "size": 1, "depth": 2, "type": "text"},
            // Equally quiet, but carrying identity claims: stripped, not dropped.
            {"id": 9, "path": "p.tgz!!tool.exe", "sha": "9", "size": 1, "depth": 1, "type": "pe",
             "mol": "C2H4", "traits": [{"id": "e", "crit": 2}],
             "ident": {"name": {"value": "tool", "source": "pe.version.product_name", "verified": false},
                       "version": {"value": "1.2.3", "source": "pe.version.file_version", "verified": false},
                       "trust": "unsigned"}},
        ]}))
        .unwrap();
        apply_report_retention(&mut report);
        let ids: Vec<u32> = report.files.iter().map(|f| f.id).collect();
        assert!(
            ids.contains(&0) && ids.contains(&1) && ids.contains(&2) && ids.contains(&3),
            "root, notable, cited contributor, and citing node survive: {ids:?}"
        );
        assert!(
            ids.contains(&4) && ids.contains(&5),
            "registry and fetch-placeholder provenance skeleton survive: {ids:?}"
        );
        assert!(ids.contains(&6), "top-risk node survives: {ids:?}");
        let exe = report
            .files
            .iter()
            .find(|f| f.id == 9)
            .expect("identity-bearing member survives as a listing entry");
        assert!(
            exe.findings.is_empty() && exe.formula.is_none(),
            "stripped member sheds its analysis payload"
        );
        assert!(
            exe.identity.as_ref().is_some_and(|i| !i.is_empty()),
            "stripped member keeps its identity claims"
        );
        assert!(
            !ids.contains(&7) && !ids.contains(&8),
            "sub-notable uncited nodes are dropped outright: {ids:?}"
        );
    }

    /// A hostile archive can carry unlimited quiet members that each claim an
    /// identity, so the one attacker-driven retention rule has to stop.
    #[test]
    fn identity_only_retention_is_capped() {
        let members: Vec<serde_json::Value> = (1..=super::MAX_IDENTITY_ONLY_NODES + 500)
            .map(|i| {
                serde_json::json!({
                    "id": i, "path": format!("p.tgz!!m{i}.exe"), "sha": format!("{i}"),
                    "size": 1, "depth": 1, "type": "pe",
                    "ident": {
                        "name": {"value": "tool", "source": "pe.version.product_name",
                                 "verified": false},
                        "trust": "unsigned"
                    }
                })
            })
            .collect();
        let mut report: cleave::types::CompactReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": std::iter::once(serde_json::json!(
                {"id": 0, "path": "p.tgz", "sha": "0", "size": 1, "depth": 0, "type": "zip"}))
                .chain(members)
                .collect::<Vec<_>>(),
        }))
        .unwrap();

        apply_report_retention(&mut report);

        // Three nodes survive on rules the identity budget never sees: the root
        // (rule 1) and ids 1-2, which win top-3-by-risk because every risk ties
        // and the sort is stable. The other 4096 are the identity budget spent
        // in full — the remaining 500 members are dropped.
        assert_eq!(
            report.files.len(),
            super::MAX_IDENTITY_ONLY_NODES + 3,
            "identity-only survivors stop at the cap"
        );
        assert!(
            report.files.iter().any(|f| f.id == 0),
            "the root is kept by rule 1, not by the identity budget"
        );
        let highest_kept =
            u32::try_from(super::MAX_IDENTITY_ONLY_NODES + 2).expect("cap fits in an id");
        assert!(
            report.files.iter().all(|f| f.id <= highest_kept),
            "the cap keeps a deterministic prefix, not an arbitrary subset"
        );
    }

    #[test]
    fn declarer_gets_span_and_structured_dep() {
        let mut report = compact_fixture();
        inject_dependency_backref(&mut report, &backref(Classification::Hostile));
        let report_json = wire(&report);

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
        assert_eq!(
            t["spans"][0][1], 9,
            "span length survives typed-report drop"
        );
    }

    #[test]
    fn ancestor_carries_dep_without_span_and_siblings_stay_clean() {
        let mut report = compact_fixture();
        inject_dependency_backref(&mut report, &backref(Classification::Hostile));
        let report_json = wire(&report);

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
        let mut report = compact_fixture();
        inject_dependency_backref(&mut report, &backref(Classification::Suspicious));
        let report_json = wire(&report);

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
        let mut report = compact_fixture();
        let mut b = backref(Classification::Hostile);
        b.locator = "http://x.y.z/x.exe".to_string();
        b.dep_type = "pe".to_string();
        inject_dependency_backref(&mut report, &b);
        let report_json = wire(&report);

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
        // The record is projected for the grader: identity plus whichever
        // credibility signals the registry supplied, not the full normalized
        // record (see `project_registry_record`) — and never the provider
        // document behind it.
        assert!(
            ctx.contains(r#""record":{"package":"test/dep@1"}"#),
            "{ctx}"
        );
        assert!(
            !ctx.contains("provider_only") && !ctx.contains(r#""raw""#),
            "provider documents must not reach the grader: {ctx}"
        );
        assert!(ctx.contains(r#""source_path":"root.tgz""#));
        assert!(
            !ctx.contains(":null"),
            "sparse records must omit nulls: {ctx}"
        );

        let terminal = render_terminal_fetch_context(&edges, &deps, &registries, &report)
            .expect("hostile dependency is shown");
        let terminal = crate::deptree::strip_ansi(&terminal);
        let provenance = terminal.find("dependency from this file").unwrap();
        let finding = terminal.find("dependency package finding").unwrap();
        assert!(provenance < finding);
        assert!(
            terminal.contains(
                "\n   └─ dependency from this file · HOSTILE 97%\n      🔗  pkg:test/dep@1"
            )
        );
        assert!(terminal.contains("\n      ●●● dependency package finding"));
        assert!(!terminal.contains("\n\n    ●●●"));
        assert_eq!(terminal.matches("pkg:test/dep@1").count(), 1);
        assert!(
            terminal.contains(&"d".repeat(64)),
            "hash must stay complete"
        );
        assert!(!terminal.contains("📄"));
        assert!(!terminal.contains("transfer"));
        assert!(!terminal.contains("cache:"));
        assert!(terminal.contains("metadata  https://registry.example/dep"));
    }

    /// The grader sees the package's whole registry identity — every signal a
    /// supply-chain judgement leans on — and none of the provider document it
    /// was derived from, however large that document is.
    #[test]
    fn interpret_provenance_carries_identity_and_drops_provider_documents() {
        let root = cleave::FileAnalysis {
            id: 0,
            path: "unload-0.0.1.tgz".to_string(),
            sha256: "r".repeat(64),
            ..cleave::FileAnalysis::default()
        };
        let mut report = empty_report();
        report.files = vec![root];
        let now = scan_now_secs();
        let day = 86_400;
        let record: fletch::Registry = serde_json::from_value(serde_json::json!({
            "ecosystem": "npm",
            "name": "unload",
            "version": "0.0.1",
            "title": "unload",
            "description": "Run a piece of code when the javascript process is about to exit",
            "author": "pubkey",
            "publisher": "zefixx",
            "publisher_email_domain": "outlook.com",
            "publisher_in_maintainers": false,
            "maintainers": 1,
            "homepage": "https://github.com/pubkey/unload#readme",
            "repository": "git+https://github.com/pubkey/unload.git",
            "license": "MIT",
            "downloads_total": null,
            "downloads_recent": 0,
            "published_at": now - 3 * day,
            "first_published_at": now - 3558 * day,
            "previous_published_at": now - 400 * day,
            "release_count": 17,
            "releases_24h": 2,
            "releases_48h": 3,
            "latest_version": "2.4.1",
            "has_install_script": true,
            "security_hold": true,
            "version_removed": false,
            "deprecated": null,
            "unpacked_size": 12345,
            "file_count": 7,
            "vulnerability_count": 0,
        }))
        .expect("registry record");
        // A 60 KB packument: the kind of provider document that was 30% of every
        // prompt before it was dropped.
        let packument = format!(
            r#"{{"name":"unload","versions":{{{}}}}}"#,
            (0..600)
                .map(|i| format!(r#""{i}.0.0":{{"dist":{{"tarball":"https://registry.npmjs.org/unload/-/unload-{i}.0.0.tgz","shasum":"{}"}}}}"#, "0".repeat(40)))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(packument.len() > 60_000);
        let provenance = crate::provenance::RegistryProvenance::from_record_sources(
            record,
            &[fletch::fetch::RecordedSource {
                url: "https://registry.npmjs.org/unload".to_string(),
                status: 200,
                content_type: Some("application/json".to_string()),
                bytes: packument.into_bytes(),
            }],
        );

        let ctx = render_interpret_context(
            "unload-0.0.1.tgz",
            &"r".repeat(64),
            None,
            Some(&provenance),
            &[],
            &[],
            &[],
            &report,
        );
        let line = ctx
            .lines()
            .find(|l| l.starts_with("provenance="))
            .expect("primary provenance line");
        let json: serde_json::Value =
            serde_json::from_str(line.trim_start_matches("provenance=")).expect("valid JSON");
        let rec = &json["registry"]["record"];

        // Identity, in words the grader can read.
        assert_eq!(rec["package"], "npm/unload@0.0.1");
        assert_eq!(rec["title"], "unload");
        assert!(
            rec["description"]
                .as_str()
                .unwrap()
                .starts_with("Run a piece of code")
        );
        assert_eq!(rec["publisher"], "zefixx");
        assert_eq!(rec["author"], "pubkey");
        assert_eq!(rec["publisher_email_domain"], "outlook.com");
        assert_eq!(rec["maintainers"], 1);
        assert_eq!(
            rec["repository"],
            "git+https://github.com/pubkey/unload.git"
        );
        assert_eq!(rec["homepage"], "https://github.com/pubkey/unload#readme");
        assert_eq!(rec["license"], "MIT");
        // Usage: a zero download count is a signal and is kept; a null is not.
        assert_eq!(rec["downloads_recent"], 0);
        assert!(rec.get("downloads_total").is_none());
        // Age and cadence, measured at scan time.
        assert_eq!(rec["package_age_days"], 3558);
        assert_eq!(rec["version_age_days"], 3);
        assert_eq!(rec["previous_release_age_days"], 400);
        assert_eq!(rec["release_count"], 17);
        assert_eq!(rec["releases_24h"], 2);
        assert_eq!(rec["releases_48h"], 3);
        assert_eq!(rec["latest_version"], "2.4.1");
        assert_eq!(rec["is_latest"], false);
        // Registry flags: only the ones that are set.
        assert_eq!(rec["has_install_script"], true);
        assert_eq!(rec["security_hold"], true);
        assert!(
            rec.get("version_removed").is_none(),
            "false flags are omitted"
        );
        assert!(rec.get("deprecated").is_none());
        assert!(rec.get("publisher_in_maintainers").is_none());
        assert_eq!(rec["unpacked_size"], 12345);
        assert_eq!(rec["file_count"], 7);
        assert!(
            rec.get("vulnerability_count").is_none(),
            "zero vulnerabilities is not a signal"
        );

        // The provider document is gone, and the line is small however large
        // the document was.
        assert!(json["registry"].get("raw").is_none(), "{line}");
        assert!(
            !ctx.contains("registry.npmjs.org/unload/-/unload-"),
            "{ctx}"
        );
        assert!(
            line.len() < 1_500,
            "provenance line must stay a few hundred tokens, got {} bytes: {line}",
            line.len()
        );
        // Nothing about how or from where the sample was *collected* reaches
        // the grader: a feed or category name would tell it what it is being
        // asked to find.
        for word in ["collector", "category", "feed", "forager", "hopper"] {
            assert!(!line.contains(word), "{word} in {line}");
        }
    }

    /// Identity survives for artifacts that have no registry at all.
    ///
    /// A PE, a Mach-O or an Office document is never a package: dropping
    /// `registry.raw` from the prompt (2026-09-05) left them with *only* what
    /// the file claims about itself, so the claims cleave lifts out of a
    /// version resource, a code signature or `docProps/core.xml` are the whole
    /// identity signal a supply-chain judgement can use — a signer that
    /// disagrees with the product, an author on a document that arrived from
    /// nowhere. They reach the model through cleave's minimal header
    /// (`identity_headline`) rather than through `provenance=`, which is
    /// exactly why a change to the provenance line could drop them unnoticed.
    #[test]
    fn interpret_render_carries_file_identity_for_unregistered_artifacts() {
        let identity = |json: serde_json::Value| -> Option<filefacts::Identity> {
            Some(serde_json::from_value(json).expect("identity fixture"))
        };
        let claim =
            |value: &str, source: &str| serde_json::json!({"value": value, "source": source});

        let mut root = cleave::FileAnalysis {
            id: 0,
            path: "vendor-bundle.zip".to_string(),
            sha256: "r".repeat(64),
            file_type: "zip".to_string(),
            ..cleave::FileAnalysis::default()
        };
        root.findings.push(finding(
            "root/notable",
            "archive with mixed content",
            cleave::Criticality::Notable,
        ));

        // A signed Windows binary: the version resource says one company, the
        // Authenticode chain says another. Both must reach the model.
        let mut pe = cleave::FileAnalysis {
            id: 1,
            parent_id: Some(0),
            depth: 1,
            path: "vendor-bundle.zip/setup.exe".to_string(),
            sha256: "p".repeat(64),
            file_type: "pe".to_string(),
            size: 1_500_000,
            identity: identity(serde_json::json!({
                "name": claim("setup.exe", "pe.version.original_filename"),
                "project": claim("Contoso Updater", "pe.version.product_name"),
                "version": claim("3.5.1", "pe.version.file_version"),
                "organization": claim("Contoso Ltd", "pe.version.company_name"),
                "signer": {
                    "common_name": "Vanguard Tech Limited",
                    "organization": "Vanguard Tech Limited",
                    "source": "pe.signatures[0]",
                },
                "trust": "ca_signed",
                "build_path": claim(
                    r"C:\Users\dev\.cargo\registry\src\index.crates.io\serde_json-1.0.114\src\de.rs",
                    "strings",
                ),
            })),
            ..cleave::FileAnalysis::default()
        };
        pe.findings.push(finding(
            "pe/notable",
            "imports network APIs",
            cleave::Criticality::Notable,
        ));

        // A macOS dylib: the bundle identifier and the Apple team that signed it.
        let mut macho = cleave::FileAnalysis {
            id: 2,
            parent_id: Some(0),
            depth: 1,
            path: "vendor-bundle.zip/libhelper.dylib".to_string(),
            sha256: "m".repeat(64),
            file_type: "macho".to_string(),
            size: 802_000,
            identity: identity(serde_json::json!({
                "identifier": claim("com.contoso.helper", "macho.bundle_identifier"),
                "version": claim("1.4.0", "macho.bundle_version"),
                "signer": {
                    "common_name": "Developer ID Application: Contoso Ltd (AB12CD34EF)",
                    "organization": "Contoso Ltd",
                    "source": "macho.code_signature",
                },
                "team_id": claim("AB12CD34EF", "macho.code_signature"),
                "trust": "developer_id",
            })),
            ..cleave::FileAnalysis::default()
        };
        macho.findings.push(finding(
            "macho/notable",
            "resolves symbols at runtime",
            cleave::Criticality::Notable,
        ));

        // An Office document: title, the person named as its author, and the
        // application that produced it.
        let mut docx = cleave::FileAnalysis {
            id: 3,
            parent_id: Some(0),
            depth: 1,
            path: "vendor-bundle.zip/invoice.docx".to_string(),
            sha256: "d".repeat(64),
            file_type: "docx".to_string(),
            size: 20_000,
            identity: identity(serde_json::json!({
                "title": claim("Q3 Vendor Invoice", "docprops.core.title"),
                "authors": [{
                    "name": "Aleksandr Petrov",
                    "role": "creator",
                    "source": "docprops.core.creator",
                }],
                "organization": claim("Contoso Ltd", "docprops.app.company"),
                "producer": claim("Microsoft Office Word", "docprops.app.application"),
                "trust": "unsigned",
            })),
            ..cleave::FileAnalysis::default()
        };
        docx.findings.push(finding(
            "docx/notable",
            "document contains an external relationship",
            cleave::Criticality::Notable,
        ));

        let mut report = empty_report();
        report.files = vec![root, pe, macho, docx];

        let ctx = render_interpret_context(
            "vendor-bundle.zip",
            &"r".repeat(64),
            None,
            None,
            &[],
            &[],
            &[],
            &report,
        );

        // Every artifact keeps a header naming what it claims to be.
        for (path, kind) in [
            ("setup.exe", "pe"),
            ("libhelper.dylib", "macho"),
            ("invoice.docx", "docx"),
        ] {
            let header = ctx
                .lines()
                .find(|l| l.contains(path) && l.contains(kind))
                .unwrap_or_else(|| panic!("no header for {path} in:\n{ctx}"));
            assert!(
                header.matches('\t').count() >= 2,
                "{path} header lost its identity field: {header:?}"
            );
        }

        // PE: the signer, and the product it claims to be. A signer that
        // disagrees with the company in the version resource is the finding a
        // reader can only make when both are present.
        assert!(ctx.contains("Vanguard Tech Limited"), "PE signer: {ctx}");
        assert!(ctx.contains("ca-signed"), "PE trust tier: {ctx}");
        // The version resource's file version rides along with the name.
        assert!(ctx.contains("3.5.1"), "PE version: {ctx}");
        // What the binary claims about *itself*. The headline can name only
        // one party and picks the signature, so these reach the reader on
        // cleave's `claims` line — and the disagreement between the claimed
        // "Contoso Ltd" and the signing "Vanguard Tech Limited" is a signal
        // that exists only because both are rendered.
        assert!(
            ctx.contains(r#"product="Contoso Updater""#),
            "PE product name: {ctx}"
        );
        assert!(
            ctx.contains(r#"company="Contoso Ltd""#),
            "PE company: {ctx}"
        );
        // The build path leaks the developer account and is rendered beside it.
        assert!(ctx.contains("serde_json-1.0.114"), "PE build path: {ctx}");

        // Mach-O: bundle identity, the Apple team, and the bundle version —
        // which the headline drops whenever the identifier is the subject.
        assert!(
            ctx.contains("com.contoso.helper"),
            "Mach-O identifier: {ctx}"
        );
        assert!(ctx.contains("developer-id"), "Mach-O trust tier: {ctx}");
        assert!(ctx.contains(r#"team="AB12CD34EF""#), "Mach-O team: {ctx}");
        assert!(
            ctx.contains(r#"version="1.4.0""#),
            "Mach-O bundle version: {ctx}"
        );

        // Office document: title, author, producing application.
        assert!(ctx.contains("Q3 Vendor Invoice"), "docx title: {ctx}");
        assert!(ctx.contains("Aleksandr Petrov"), "docx author: {ctx}");
        assert!(
            ctx.contains("Microsoft Office Word"),
            "docx producer: {ctx}"
        );
        // The company the document claims, which its author outranked in the
        // headline: a document authored outside the company it names is the
        // same shape of tell as the PE above.
        assert!(
            ctx.contains(r#"company="Contoso Ltd""#),
            "docx company: {ctx}"
        );

        // None of this is registry provenance: these artifacts have no package
        // record, and the one `provenance=` line is the artifact's own hash.
        assert_eq!(ctx.matches("provenance=").count(), 1, "{ctx}");
        assert!(!ctx.contains(r#""registry""#), "{ctx}");
    }

    /// Byte windows are evidence, not hinting: the LLM render must keep the
    /// rows around a hex hit. Dropping them (`full_context: false`) was tried
    /// as a size lever on 2026-09-05 and turned a known-bad PE
    /// (`fffmpeg.exe`) from hostile to benign on the shipped prompt, while
    /// every finding description stayed. Size is taken from the metadata
    /// instead (see `slim_provenance_for_interpret`).
    #[test]
    fn interpret_render_keeps_byte_windows_as_evidence() {
        let opts = cleave::output::TinyOpts::tiny();
        assert!(
            opts.full_context,
            "hex hits must render with their surrounding rows"
        );
        assert!(
            opts.context_lines.is_some(),
            "and a bounded window, not the whole capture"
        );
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
        // The record reaches the grader; the provider document behind it does not.
        assert!(ctx.contains(r#""record":{"#), "{ctx}");
        assert!(!ctx.contains("provider_only"), "{ctx}");
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
        artifact.findings.push(finding(
            "package/composite",
            "",
            cleave::Criticality::Notable,
        ));
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
        assert_eq!(terminal_fetch_source(&rec, &report), "this file");
        write_terminal_fetch_redirects(&mut out, &rec);
        assert!(!out.contains("byte 42"));
        assert!(!out.contains(&sha));
        assert!(!out.contains("resolved"));
        assert!(!out.contains("final"));

        rec.final_url = Some("https://cdn.example.test/stage.sh".to_string());
        out.clear();
        write_terminal_fetch_redirects(&mut out, &rec);
        assert!(out.contains("final  https://cdn.example.test/stage.sh"));
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
        &tiny_opts_for(config),
        config.interpret(),
        path,
        config.fetch_policy(),
        config.zip_passwords(),
        OutputNeeds {
            llm_view: matches!(config.format(), OutputFormat::Interpret),
            // The live fetch log renders only on the interactive terminal path;
            // JSON and tiny stay machine-clean (the edges ride along in the
            // JSON `fetched` array regardless).
            fetch_progress: matches!(config.format(), OutputFormat::Terminal),
            render_context: !is_json,
            // `--show=all` with JSON output: list every archive member, even
            // the no-finding ones cleave skipped analyzing.
            list_all_members: config.filter().is_all() && is_json,
            deps_for_upload: config.hopper().is_some(),
        },
        root_registry,
        root_fetch,
        bloom_mark,
        None,
        None, // no admission gate on the CLI path
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
    // Engine-internal attribution (per-phase thread-time, regex store churn)
    // for slow-scan diagnosis; one line per stat at info.
    cleave::log_scan_stats();

    // Retain the raw cleave report for JSON output, and whenever uploading to
    // hopper — the renewed result must carry the full report (so hopper stores it
    // and explodes archive members), and the fetch edges inside it drive the
    // content reconciliation in `record_file_result`. Without this, a terminal
    // `--upload` would post a verdict with an empty report.
    let cleave = if is_json || config.hopper().is_some() {
        Some(cr.report)
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
        pending_llm: cr.pending_llm,
        analysis_cached: cr.analysis_cached,
        dependency_results: cr.dependency_results,
        bloom_mark,
        hopper_route: HopperRoute::Normal,
    })
}

/// Count a report's findings by criticality — the root file's own findings,
/// which are the ones that describe the artifact scan was asked about.
#[must_use]
pub fn count_findings(report: &cleave::types::CompactReport) -> FindingCounts {
    let Some(first) = report.files.first() else {
        return FindingCounts::default();
    };
    let mut counts = FindingCounts::default();
    for f in &first.findings {
        match f.criticality {
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

    cleave::set_compact_member_retention(true); // compact projection only
    let mut cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        ..Default::default()
    };
    add_zip_passwords(&mut cleave_opts, config.zip_passwords());
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

    cleave::set_compact_member_retention(true); // compact projection only
    let mut cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        ..Default::default()
    };
    add_zip_passwords(&mut cleave_opts, config.zip_passwords());
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
pub fn extract_top_findings(
    findings: &[cleave::types::CompactTrait],
    classification: &Classification,
) -> Vec<TopFinding> {
    let min_crit: u8 = match classification {
        Classification::Hostile => 5,
        Classification::Suspicious | Classification::Benign => 4,
    };

    let mut relevant: Vec<TopFinding> = findings
        .iter()
        .filter(|f| f.criticality >= min_crit)
        .map(TopFinding::from)
        .collect();

    // Fall back to suspicious-level findings if no hostile-level findings.
    if relevant.is_empty() && min_crit == 5 {
        relevant = findings
            .iter()
            .filter(|f| f.criticality >= 4)
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
    report: &cleave::types::CompactReport,
    root_prob: f32,
    root_level: Option<i32>,
    embedded_files: &MemberEvals,
) -> Vec<serde_json::Value> {
    let mut out = Vec::with_capacity(report.files.len());
    for entry in &report.files {
        // Listing-only members (added for `--show=all`) were never analyzed, so
        // they carry no ML verdict; their sentinel risk keeps them out of the
        // classified `ml.files` while they remain in the raw `files` manifest.
        if entry.risk == UNANALYZED_MEMBER_RISK {
            continue;
        }
        let id = u64::from(entry.id);
        let evaluation = if entry.depth == 0 {
            Some((root_prob, root_level))
        } else {
            embedded_files.get(&id).map(|ef| (ef.probability, ef.level))
        };

        match evaluation {
            Some((prob, file_level)) => out.push(serde_json::json!({
                "id": id,
                "type": entry.file_type,
                "prob": prob,
                "lvl": file_level,
                "conf": level_confidence(file_level),
            })),
            // No verdict fields: consumers (hopper's forMember) treat a
            // prob-less entry as "not analyzed", which is the truth.
            None => out.push(serde_json::json!({
                "id": id,
                "type": entry.file_type,
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
    pub raw: cleave::types::CompactReport,
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
    pub raw: &'a cleave::types::CompactReport,
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
        let raw = self.cleave.clone().unwrap_or_default();
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
        let raw = self.cleave.unwrap_or_default();
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
    /// path (e.g. the `scan` and [`crate::ps`] `emit_result` paths) — it avoids
    /// cloning the cleave report and all owned string fields. Prefer
    /// [`Self::into_envelope`] when the caller can give up ownership.
    #[must_use]
    pub fn envelope_ref(&self) -> ScanResultEnvelopeRef<'_> {
        static EMPTY_RAW: OnceLock<cleave::types::CompactReport> = OnceLock::new();
        let raw: &cleave::types::CompactReport = match &self.cleave {
            Some(v) => v,
            None => EMPTY_RAW.get_or_init(cleave::types::CompactReport::default),
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
