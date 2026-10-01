//! The scan result and the JSON envelope it serializes to.

use std::sync::OnceLock;

use anyhow::{Context as _, Result};
use cleave::types::{CompactReport, CompactTrait};

use super::{
    ENGINE_VERSION, FindingCounts, FloorDecision, PendingLlm, Reason, SCHEMA_VERSION,
    UNANALYZED_MEMBER_RISK, level_confidence,
};
use crate::interpret::Interpretation;
use crate::model::{Classification, Decision, Level, RouteScore, SkippedRoute};

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
    /// Independent of the deploy `-l`, so the envelope is identical across
    /// levels and cache-shareable — `-l` only moves the cutoffs that turn
    /// `level` into `classification`.
    pub level: Level,
    /// Model version identifier (spec version, ABI version, model hash prefix).
    pub version: String,
    /// UTC timestamp of when this analysis was performed (RFC 3339).
    pub analyzed_at: String,
    /// The compact cleave report as stored: retention applied, dependency
    /// verdicts pinned. `None` when the caller had no use for it.
    pub cleave: Option<CompactReport>,
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
    pub interpretation: Option<Interpretation>,
    /// A second opinion still to run; see [`PendingLlm`]. Never serialized.
    pub pending_llm: Option<PendingLlm>,
    /// Whether cleave replayed this analysis from its on-disk cache instead of
    /// running the pipeline. Not serialized — it describes how this run reached
    /// the verdict, not the verdict — but it is what tells an operator whether a
    /// fast response was cached work or a fast file.
    pub analysis_cached: bool,
    /// Wall-clock the LLM second opinion took, in milliseconds; 0 when none
    /// ran. Not serialized. The server subtracts it from the latency it
    /// reports to the router: the endpoint is shared by every worker, so its
    /// slowness says nothing about which worker to pick.
    pub interpret_ms: u64,
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
    /// What the model alone concluded, across the root and every member,
    /// before the trait floor and before any interpretation.
    ///
    /// [`classification`](Self::classification) is the verdict; this is the
    /// model's share of it. The two differ exactly when the floor fired on
    /// whichever file decided the verdict. Not serialized: the envelope
    /// reports one verdict, and this is for a caller that reports the model
    /// and the floor as separate opinions rather than as one number.
    ///
    /// Recovering it is cheap because the floor fires only on a model-Benign
    /// decision — so on a file it raised, the model's own reading was benign,
    /// and everywhere else the stored decision is already the model's.
    pub model: Decision,
    /// The gravest trait-floor firing anywhere in this artifact, if any.
    ///
    /// `None` means the floor had nothing to add, never that it found the
    /// artifact clean — the floor has no way to say benign. Not serialized,
    /// for the same reason as [`model`](Self::model).
    pub floor: Option<FloorDecision>,
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
    /// Cleave-assigned confidence in `[0.0, 1.0]`.
    pub conf: f32,
    /// Human-readable description of the finding.
    pub desc: String,
}

impl From<&CompactTrait> for TopFinding {
    fn from(f: &CompactTrait) -> Self {
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
    pub level: Level,
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
    /// What the trait floor did to this member, if it fired.
    ///
    /// Not serialized: `ml.files` is a stable wire shape with consumers that
    /// were not compiled against this field. It exists so a caller reporting
    /// the model and the floor as separate opinions can tell which of them
    /// convicted a member — `classification` above is the floored verdict and
    /// no longer says which produced it.
    #[serde(skip)]
    pub floor: Option<FloorDecision>,
}

impl EmbeddedFile {
    /// This member's verdict as a [`Decision`], for outranking comparisons.
    pub(super) fn decision(&self) -> Decision {
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

/// Build the `/api/result` envelope for a fetched dependency: the standalone
/// cleave report scan captured for it as `raw`, and the aggregate verdict it
/// computed as the `ml` section — the same shape a first-hand scan of those bytes
/// would post, so hopper records the dependency exactly as if it had been scanned
/// directly. `version`/`analyzed_at` are the parent run's, identifying the build.
///
/// `Ok(None)` when the dependency carries no verdict (see [`DepResult::verdict`])
/// — there is nothing to post, and scan does not invent one.
///
/// # Errors
/// The dependency's report does not parse. Posting an empty report in its place
/// would read to hopper as a dependency with no files.
pub(crate) fn dep_envelope(
    dep: &DepResult,
    version: &str,
    analyzed_at: &str,
) -> Result<Option<ScanResultEnvelope>> {
    let Some(verdict) = dep.verdict else {
        return Ok(None);
    };
    // The report text is decoded only here, for the short-lived envelope being
    // POSTed — not for the job-long retention window.
    let raw: CompactReport = serde_json::from_str(&dep.raw)
        .with_context(|| format!("report of dependency {}", dep.locator))?;
    let level = verdict.level;
    Ok(Some(ScanResultEnvelope {
        ml: MlSection {
            v: SCHEMA_VERSION,
            probability: verdict.probability,
            level,
            conf: level_confidence(level),
            model_scores: Vec::new(),
            skipped_models: Vec::new(),
            version: version.to_string(),
            eng: ENGINE_VERSION,
            analyzed_at: analyzed_at.to_string(),
            files: ml_files(&raw, verdict.probability, level, &dep.members),
            pids: None,
            deleted: None,
        },
        llm: None,
        raw,
    }))
}

/// One row of `ml.files`: a report node and, when it was evaluated, its own
/// verdict.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub(crate) struct MlFile {
    pub(crate) id: u64,
    #[serde(rename = "type")]
    pub(crate) file_type: String,
    /// Absent, not defaulted, for a node nobody evaluated: hopper reads a
    /// prob-less row as "not analyzed", which is the truth.
    #[serde(flatten)]
    pub(crate) verdict: Option<MlFileVerdict>,
}

/// The verdict fields of an evaluated [`MlFile`] row.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub(crate) struct MlFileVerdict {
    /// Widened to `f64` because these rows were built with `json!`, which
    /// widens; a direct `f32` would print fewer digits and change the bytes.
    pub(crate) prob: f64,
    pub(crate) lvl: Level,
    pub(crate) conf: Option<u8>,
}

/// Build the `ml.files` rows, one per analyzed report node.
///
/// The root (`depth == 0`) carries the envelope's probability and level;
/// members are joined by id and report their *own* — so every row's `lvl` is
/// the level-independent marker for that file. A member with no evaluation gets
/// no verdict fields. It must NEVER inherit the root's verdict: a member can
/// occur in many containers, and hopper mirrors these rows into the member's
/// own sample row (`litmusResultForMember`), so a fabricated value becomes that
/// file's grade everywhere it appears.
fn ml_files(
    report: &CompactReport,
    root_prob: f32,
    root_level: Level,
    members: &MemberEvals,
) -> Vec<MlFile> {
    report
        .files
        .iter()
        // Listing-only members (`--show=all`) were never analyzed; they stay in
        // the raw `files` manifest but not here.
        .filter(|entry| entry.risk != UNANALYZED_MEMBER_RISK)
        .map(|entry| {
            let id = u64::from(entry.id);
            let evaluation = if entry.depth == 0 {
                Some((root_prob, root_level))
            } else {
                members.get(&id).map(|ef| (ef.probability, ef.level))
            };
            MlFile {
                id,
                file_type: entry.file_type.clone(),
                verdict: evaluation.map(|(prob, lvl)| MlFileVerdict {
                    prob: f64::from(prob),
                    lvl,
                    conf: level_confidence(lvl),
                }),
            }
        })
        .collect()
}

/// Top-level JSON envelope: `{"ml": {...}, "llm": {...}, "raw": {...}}`.
#[derive(Debug, serde::Serialize)]
pub struct ScanResultEnvelope {
    /// ML classification section.
    pub ml: MlSection,
    /// LLM interpretation section (`--interpret`); omitted when not run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm: Option<Interpretation>,
    /// Raw cleave analysis report.
    pub raw: CompactReport,
}

/// The envelope with the report and interpretation borrowed, for writing a
/// result without cloning its (possibly multi-MB) report. Serializes to the
/// same bytes as [`ScanResultEnvelope`].
#[derive(Debug, serde::Serialize)]
pub struct ScanResultEnvelopeRef<'a> {
    /// ML classification section.
    pub ml: MlSection,
    /// LLM interpretation section (`--interpret`); omitted when not run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm: Option<&'a Interpretation>,
    /// Raw cleave analysis report.
    pub raw: &'a CompactReport,
}

/// The `ml` section of the response envelope.
#[derive(Debug, serde::Serialize)]
pub struct MlSection {
    pub(crate) v: &'static str,
    #[serde(rename = "prob")]
    pub(crate) probability: f32,
    /// Level-independent verdict marker, always serialized (`null` in
    /// manual-threshold mode). See [`ScanResult::level`].
    #[serde(rename = "lvl")]
    pub(crate) level: Level,
    /// Pessimistic integer confidence percent derived from `level`; `null` when no
    /// level table applies (manual-threshold mode).
    pub(crate) conf: Option<u8>,
    #[serde(rename = "mods", skip_serializing_if = "Vec::is_empty")]
    pub(crate) model_scores: Vec<RouteScore>,
    #[serde(rename = "skip", skip_serializing_if = "Vec::is_empty")]
    pub(crate) skipped_models: Vec<SkippedRoute>,
    pub(crate) version: String,
    /// Scan engine build (`CARGO_PKG_VERSION`) that produced this report.
    pub(crate) eng: &'static str,
    pub(crate) analyzed_at: String,
    pub(crate) files: Vec<MlFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) deleted: Option<bool>,
}

impl ScanResult {
    /// The `ml` section. Clones only the small fields; `files` is built here.
    fn ml_section(&self) -> MlSection {
        MlSection {
            v: self.v,
            probability: self.probability,
            level: self.level,
            conf: level_confidence(self.level),
            model_scores: self.model_scores.clone(),
            skipped_models: self.skipped_models.clone(),
            version: self.version.clone(),
            eng: ENGINE_VERSION,
            analyzed_at: self.analyzed_at.clone(),
            files: self.cleave.as_ref().map_or_else(Vec::new, |raw| {
                ml_files(raw, self.probability, self.level, &self.embedded_files)
            }),
            pids: self.pids.clone(),
            deleted: self.deleted,
        }
    }

    /// Build the envelope, cloning the cleave report. Prefer
    /// [`Self::envelope_ref`] to write it, or [`Self::into_envelope`] when the
    /// result is not needed afterwards.
    #[must_use]
    pub fn to_envelope(&self) -> ScanResultEnvelope {
        ScanResultEnvelope {
            ml: self.ml_section(),
            llm: self.interpretation.clone(),
            raw: self.cleave.clone().unwrap_or_default(),
        }
    }

    /// Build the envelope, consuming the result so the report moves instead of
    /// being cloned.
    #[must_use]
    pub fn into_envelope(self) -> ScanResultEnvelope {
        ScanResultEnvelope {
            ml: self.ml_section(),
            llm: self.interpretation,
            raw: self.cleave.unwrap_or_default(),
        }
    }

    /// The envelope with the report borrowed, for the JSON output path.
    #[must_use]
    pub fn envelope_ref(&self) -> ScanResultEnvelopeRef<'_> {
        static EMPTY_RAW: OnceLock<CompactReport> = OnceLock::new();
        ScanResultEnvelopeRef {
            ml: self.ml_section(),
            llm: self.interpretation.as_ref(),
            raw: self
                .cleave
                .as_ref()
                .unwrap_or_else(|| EMPTY_RAW.get_or_init(CompactReport::default)),
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Decode a wire-shaped fixture into the typed report the pipeline carries,
    /// so these tests exercise the same reader production does.
    fn report(v: serde_json::Value) -> cleave::types::CompactReport {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn ml_files_drops_listing_only_members() {
        // A root file, an analyzed member, and a listing-only member (risk -1).
        let report = report(serde_json::json!({
            "files": [
                {"id": 0, "path": "app.zip", "type": "zip", "sha": "r", "size": 4096, "depth": 0},
                {"id": 1, "path": "app.zip!!evil.sh", "type": "shell", "sha": "a", "size": 200, "depth": 1, "risk": 9},
                {"id": 2, "path": "app.zip!!README.md", "type": "markdown", "sha": "b", "size": 1024, "depth": 1, "risk": -1},
            ]
        }));
        let ml = ml_files(&report, 0.9, Level::At(100), &MemberEvals::new());
        let ids: Vec<u64> = ml.iter().map(|f| f.id).collect();
        assert_eq!(
            ids,
            vec![0, 1],
            "listing-only member is excluded from ml.files"
        );
    }

    #[test]
    fn ml_files_never_stamps_members_with_root_verdict() {
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
        let member = |id: u64, path: &str, prob: f32, level: Level| EmbeddedFile {
            floor: None,
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
        let evaluated = MemberEvals::from([
            (1, member(1, "compatibility", 0.00001, Level::Clean)),
            (2, member(2, "page", 0.00002, Level::Clean)),
            (3, member(3, "page", 0.7, Level::At(3000))),
        ]);
        let ml = serde_json::to_value(ml_files(&report, 0.99, Level::At(0), &evaluated)).unwrap();

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
                level: Level::At(100),
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
            .expect("the report parses")
            .expect("a dependency with a verdict yields an envelope");
        assert_eq!(
            env.ml.level,
            Level::At(100),
            "aggregate verdict rides in ml.lvl"
        );
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
            dep_envelope(&dep, "model-9", "2026-06-28T00:00:00Z")
                .expect("the report parses")
                .is_none(),
            "an unevaluated dependency posts no verdict",
        );
    }

    /// A report that does not parse is an error, not an empty report: posted
    /// empty, it reads to hopper as a dependency with no files.
    #[test]
    fn dep_envelope_rejects_an_unparseable_report() {
        let dep = DepResult {
            raw: "{not json".to_string(),
            ..evaluated_dep()
        };
        assert!(dep_envelope(&dep, "model-9", "2026-06-28T00:00:00Z").is_err());
    }

    /// `ml.files` is a wire format hopper stores: probabilities print widened
    /// to f64, rows keep `id, type, prob, lvl, conf` order, and an unevaluated
    /// row is `{id, type}` alone.
    #[test]
    fn ml_files_bytes_are_pinned() {
        let report = report(serde_json::json!({
            "files": [
                {"id": 0, "path": "a.zip", "type": "zip", "sha": "r", "size": 1, "depth": 0},
                {"id": 1, "path": "a.zip!!x", "type": "js", "sha": "x", "size": 1, "depth": 1},
            ]
        }));
        let rows = ml_files(&report, 0.42, Level::Manual, &MemberEvals::new());
        assert_eq!(
            serde_json::to_string(&rows).unwrap(),
            r#"[{"id":0,"type":"zip","prob":0.41999998688697815,"lvl":null,"conf":null},{"id":1,"type":"js"}]"#
        );
        let rows = ml_files(&report, 0.42, Level::Clean, &MemberEvals::new());
        assert_eq!(
            serde_json::to_string(&rows).unwrap(),
            r#"[{"id":0,"type":"zip","prob":0.41999998688697815,"lvl":-1,"conf":0},{"id":1,"type":"js"}]"#
        );
        let rows = ml_files(&report, 0.42, Level::At(25), &MemberEvals::new());
        assert_eq!(
            serde_json::to_string(&rows).unwrap(),
            r#"[{"id":0,"type":"zip","prob":0.41999998688697815,"lvl":25,"conf":92},{"id":1,"type":"js"}]"#
        );
    }

    pub(crate) fn base_result() -> ScanResult {
        ScanResult {
            v: SCHEMA_VERSION,
            model: Decision {
                class: Classification::Benign,
                probability: 0.10,
                threshold: 0.65,
                level: Level::Clean,
            },
            floor: None,
            analysis_cached: false,
            interpret_ms: 0,
            classification: Classification::Benign,
            probability: 0.10,
            threshold: 0.65,
            // Benign that never fires at any grid level.
            level: Level::Clean,
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
        r.level = Level::Manual;
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
        r.level = Level::At(7);
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
        r.level = Level::At(500);
        let json = serde_json::to_value(r.to_envelope()).expect("serialize");
        assert_eq!(json["ml"]["lvl"].as_i64(), Some(500));
    }

    #[test]
    fn envelope_per_file_level_reflects_each_member() {
        // Each `files[]` row reports its own file's lowest-firing-level: the root
        // carries the envelope `lvl`, members their own (matched by path suffix).
        let mut r = base_result();
        r.level = Level::At(20);
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
        let member = |id: u64, path: &str, level: Level, prob: f32| EmbeddedFile {
            floor: None,
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
            (1, member(1, "evil.sh", Level::At(2), 0.99)),
            (2, member(2, "readme.txt", Level::Clean, 0.01)),
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
