//! The scan result: one analysis, and what each output reads from it.

use cleave::types::{CompactReport, CompactTrait};

use super::{FindingCounts, FloorDecision, PendingLlm, Reason, UploadDoc};
use crate::interpret::Interpretation;
use crate::lookup::{Hit, collect_hits};
use crate::model::{Classification, Decision, Level, RouteScore, SkippedRoute};

/// Classification result for a single analyzed file or executable.
///
/// In terminal mode only a subset of results may be shown, but in JSON mode
/// every scanned item is emitted as a `ScanResult`.
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// Model classification outcome.
    pub classification: Classification,
    /// Dependency attention counts retained even when the report is dropped.
    pub dependency_attention: super::DependencyAttention,
    /// Probability the verdict was decided on.
    pub probability: f32,
    /// Cutoff defining the verdict band — the same value `probability` was
    /// compared against to produce `classification`.
    pub threshold: f32,
    /// Level-independent envelope marker (`ml.lvl`): the lowest false-positive
    /// level (pool-adjusted FP scale) at which this file's hostile decision fires.
    /// Independent of the deploy `-l`, so the envelope is identical across
    /// levels and cache-shareable — `-l` only moves the cutoffs that turn
    /// `level` into `classification`.
    pub level: Level,
    /// Model version identifier (spec version, ABI version, model hash prefix).
    pub version: String,
    /// UTC timestamp of when this analysis was performed (RFC 3339).
    pub analyzed_at: String,
    /// What the result keeps of its cleave report (see [`StoredReport`]).
    /// `None` when the caller had no use for it.
    pub cleave: Option<StoredReport>,
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
    /// Optional LLM interpretation blended with the ML verdict (`--llm`).
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
    /// reports to the router: the endpoint is shared by every server, so its
    /// slowness says nothing about which server to pick.
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
    /// On a file the floor raised, [`FloorDecision::model`] holds the model's
    /// own reading; everywhere else the stored decision is already the model's.
    pub model: Decision,
    /// The gravest trait-floor firing anywhere in this artifact, if any.
    ///
    /// `None` means the floor had nothing to add, never that it found the
    /// artifact clean — the floor has no way to say benign. Not serialized,
    /// for the same reason as [`model`](Self::model).
    pub floor: Option<FloorDecision>,
    /// What a hopper upload of this result says: the version 8 body, short of
    /// the run stamp. `None` unless the result was classified for upload.
    /// Never part of the JSON envelope.
    pub upload: Option<UploadDoc>,
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

/// What a finished result keeps of its cleave report: the worst findings,
/// which the verdict index and the JSON decision report.
///
/// The report itself is not kept. The terminal card is rendered and the
/// upload document built before a result leaves the pipeline, and a result
/// lives for the whole job, so holding the report would cost its full size
/// for nothing.
#[derive(Debug, Clone)]
pub struct StoredReport {
    /// The worst findings, attributed to no PURL; see [`Self::hits`].
    hits: Vec<Hit>,
}

impl StoredReport {
    /// Keep what a finished result needs of `report`.
    #[must_use]
    pub fn new(report: &CompactReport) -> Self {
        Self {
            hits: collect_hits(report),
        }
    }

    /// The worst findings, most critical first. A finding not from a
    /// dependency is attributed to `purl`, the coordinate the artifact was
    /// requested by — which is per request, so it is applied here rather
    /// than when the report was stored.
    #[must_use]
    pub fn hits(&self, purl: Option<&str>) -> Vec<Hit> {
        let mut hits = self.hits.clone();
        if let Some(purl) = purl {
            for hit in hits.iter_mut().filter(|h| h.pkg.is_empty()) {
                purl.clone_into(&mut hit.pkg);
            }
        }
        hits
    }
}

/// A fetched dependency to mirror into hopper as its own sample: the aggregate
/// verdict scan computed for it during the parent analysis, plus its standalone
/// compact cleave report and the registry provenance captured during analysis.
/// Only artifact bytes are recovered lazily from the fetch blob cache at upload
/// time.
#[derive(Debug, Clone)]
pub struct DepResult {
    /// SHA-256 of the dependency's bytes — its identity and `/v1/work/result` key.
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
    /// The version 8 body for the dependency's own result, built from its
    /// standalone report. `None` without a verdict, or when the scan was not
    /// classified for upload.
    pub upload: Option<UploadDoc>,
}

/// A representative cleave finding surfaced alongside a classification.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TopFinding {
    /// Finding identifier (e.g. "objectives/evasion/process::injection").
    pub id: String,
    /// Criticality; serialized as its [`cleave::Criticality::rank`]
    /// (0=filtered .. 5=hostile).
    #[serde(serialize_with = "serialize_rank")]
    pub crit: cleave::Criticality,
    /// Cleave-assigned confidence in `[0.0, 1.0]`.
    pub conf: f32,
    /// Human-readable description of the finding.
    pub desc: String,
}

impl From<&CompactTrait> for TopFinding {
    fn from(f: &CompactTrait) -> Self {
        Self {
            id: f.id.clone(),
            crit: super::compact_crit(f),
            conf: f.confidence,
            desc: f.description.clone(),
        }
    }
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's `serialize_with` passes the field by reference"
)]
fn serialize_rank<S: serde::Serializer>(
    crit: &cleave::Criticality,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_u8(crit.rank())
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
    /// does not say which produced it.
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

    /// The `n` gravest evaluations, worst first by the order verdicts
    /// outrank each other in (class, then level, then probability), for
    /// diagnostics displays. Probability alone would rank across routes whose
    /// scores are not on one scale. A view — the table itself is never sorted
    /// or truncated.
    #[must_use]
    pub(crate) fn top_offenders(evals: &MemberEvals, n: usize) -> Vec<&EmbeddedFile> {
        let mut v: Vec<&EmbeddedFile> = evals.values().collect();
        v.sort_by(|a, b| super::verdict::worst_first(&a.decision(), &b.decision()));
        v.truncate(n);
        v
    }
}

impl ScanResult {
    /// The full result: the version 8 body as it is uploaded, but naming no
    /// worker. With `every`, it names every member the document holds
    /// (`--all-members`). `None` when the result was not classified for upload
    /// or its body cannot be written.
    #[must_use]
    pub fn full_body(&self, every: bool) -> Option<Vec<u8>> {
        let doc = self.upload.as_ref()?;
        let at = super::analyzed_at(&self.analyzed_at);
        let stamp = super::UploadStamp {
            at: &at,
            model: &self.version,
            worker: None,
            ms: None,
        };
        let llm = self.interpretation.as_ref();
        if every {
            doc.encode_every(&self.sha256, llm, &stamp)
        } else {
            doc.encode(&self.sha256, llm, &stamp)
        }
    }

    /// Static findings remain attention-worthy when the model grades benign.
    pub(crate) fn has_severe_findings(&self) -> bool {
        self.finding_counts.hostile > 0
            || self.finding_counts.suspicious > 0
            || self
                .top_findings
                .iter()
                .any(|finding| finding.crit >= cleave::Criticality::Suspicious)
            || self.embedded_files.values().any(|member| {
                member
                    .top_findings
                    .iter()
                    .any(|finding| finding.crit >= cleave::Criticality::Suspicious)
            })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(crate) fn base_result() -> ScanResult {
        ScanResult {
            dependency_attention: super::super::DependencyAttention::default(),
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
            upload: None,
        }
    }
}
