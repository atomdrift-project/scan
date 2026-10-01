//! The classification pipeline: one analyzed report in, one verdict out.
//!
//! `classify_report` runs five stages, each returning what the next reads:
//! `prepare` (finalize, fetch and graft references, compact), `score` (the
//! model on the root and every member), `grade_dependencies`, `interpret` (the
//! LLM's second opinion), and `render`.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use cleave::types::{CompactFile, CompactReport, CompactTrait};
use cleave::{AnalysisReport, Criticality, FileAnalysis};
use fletch::fetch::FetchRecord;

use super::render_cards::{
    CardHead, CardProvenance, render_terminal_context, render_terminal_fetch_context,
};
use super::render_context::{
    Fetched, Primary, ReportIndex, recategorize_annotations, render_dependency_context,
    render_interpret_context,
};
use super::retention::{ArchiveMemberStub, append_unanalyzed_members, apply_report_retention};
use super::verdict::{
    apply_trait_floor, decision_outranks, graver, interpreted_level, softened_level, worst_member,
    worst_member_floor, worst_member_model,
};
use super::{
    DepResult, EmbeddedFile, FindingCounts, FloorDecision, HopperRoute, MemberEvals, Reason,
    SCHEMA_VERSION, ScanResult, TopFinding, Tuning, archive_leaf, compact_crit, is_inside,
    model_version_string, now_rfc3339,
};
use crate::explain::ShapImportance;
use crate::features::{ExtractContext, ParsedReport, RawNeeds};
use crate::interpret::{Admitted, Graded, Interpretation, LlmGrade};
use crate::model::{Classification, Decision, Level, Model, RouteScore, SkippedRoute};

/// What [`classify_report`] needs besides the report itself. Build one with
/// [`ClassifyRequest::new`] and override the optional context with struct
/// update syntax.
pub(crate) struct ClassifyRequest<'a> {
    /// Display label; becomes the result's path.
    pub label: &'a str,
    /// The artifact on disk, for the root imperative-reference hunt. Callers
    /// analyzing bytes pass the label; the hunt then finds nothing to re-read.
    pub root_path: &'a std::path::Path,
    pub model: &'a Model,
    pub shap: Option<&'a ShapImportance>,
    pub cancellation: Option<&'a AtomicBool>,
    pub tiny_opts: cleave::output::TinyOpts,
    pub interpret: Option<&'a crate::interpret::InterpretConfig>,
    pub fetch: crate::fetch::FetchPolicy,
    pub zip_passwords: &'a [String],
    pub needs: OutputNeeds,
    /// The root artifact's own registry metadata (one-shot `pkg:`/`url` path,
    /// or provenance a collector supplied). Grafted as a child `registry`
    /// node and correlated with the artifact, as `fetch::orchestrate` does
    /// for each fetched dependency.
    pub root_registry: Option<&'a crate::provenance::RegistryProvenance>,
    /// How a one-shot `pkg:`/`url` root was acquired. Local scans have none.
    pub root_fetch: Option<&'a FetchRecord>,
    /// Bloom flag (🚩 known-bad / 🏴 conflicted) for the terminal header.
    pub bloom_mark: Option<crate::output::BloomMark>,
    /// Stage tracker for the worker/server census, so dependency fetch and
    /// analysis are not misattributed to featurization.
    pub phase: Option<&'a cleave::PhaseTracker>,
    pub cpu_lease: Option<CpuLease>,
    /// Operator settings from the environment; [`Tuning::get`] by default.
    pub tuning: &'a Tuning,
}

impl<'a> ClassifyRequest<'a> {
    /// A request with every optional input absent: no SHAP, LLM, fetching,
    /// renders, or tracking.
    pub(crate) fn new(label: &'a str, root_path: &'a std::path::Path, model: &'a Model) -> Self {
        Self {
            label,
            root_path,
            model,
            shap: None,
            cancellation: None,
            tiny_opts: cleave::output::TinyOpts::tiny(),
            interpret: None,
            fetch: crate::fetch::FetchPolicy::default(),
            zip_passwords: &[],
            needs: OutputNeeds::default(),
            root_registry: None,
            root_fetch: None,
            bloom_mark: None,
            phase: None,
            cpu_lease: None,
            tuning: Tuning::get(),
        }
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

/// Wall-clock of each post-analysis phase inside `classify_report`. Static
/// analysis (cleave/filefacts) runs *before* this and is not included —
/// subtract `total` from the caller's whole-invocation elapsed to isolate it.
/// Logged per root file on the CLI path so a slow scan is self-diagnosing:
/// usually the LLM `interpret` time when an endpoint is contended, or `fetch`
/// on a wide `--fetch`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PhaseTimings {
    pub(crate) fetch: Duration,
    pub(crate) interpret: Duration,
    pub(crate) render: Duration,
    pub(crate) total: Duration,
}

/// What [`classify_report`] produced: the result, and where its time went.
pub(crate) struct ClassifiedReport {
    pub(crate) phase: PhaseTimings,
    /// The verdict, with `path` set to the request label and the build fields
    /// (`version`, `analyzed_at`) left for [`Self::into_scan_result`].
    pub(crate) result: ScanResult,
}

impl ClassifiedReport {
    /// The wire result for this classification of `path` under `bundle`.
    /// `keep_report` retains the raw cleave report, which JSON output and
    /// hopper renewals need and terminal output does not.
    pub(crate) fn into_scan_result(
        self,
        path: String,
        bundle: &Model,
        keep_report: bool,
    ) -> ScanResult {
        let mut result = self.result;
        result.path = path;
        result.version = model_version_string(bundle.info());
        result.analyzed_at = now_rfc3339();
        if !keep_report {
            result.cleave = None;
        }
        result
    }
}

/// Run the full cleave-finalize + model inference pipeline on a report.
/// This is the single authoritative inference path used by scan, ps, and the server.
pub(crate) fn classify_report(
    report: AnalysisReport,
    mut req: ClassifyRequest<'_>,
) -> Result<ClassifiedReport> {
    let started = Instant::now();
    let cpu_lease = req.cpu_lease.take();
    let needs = req.needs;

    let mut prepared = prepare(report, &req);
    let index = ReportIndex::new(&prepared.typed);
    let scored = score(&prepared, &req)?;
    // After featurization, never before: a backref is a conclusion about
    // *another* artifact, and fed as a feature of this one it would teach the
    // model that declaring dependencies is itself malicious.
    for backref in &scored.backrefs {
        inject_dependency_backref(&mut prepared.compact, backref);
    }
    let top_findings =
        extract_top_findings(root_findings(&prepared.compact), &scored.decision.class);
    let dependency_results = grade_dependencies(
        std::mem::take(&mut prepared.fetch.dependencies),
        &prepared.fetch.registries,
        &req,
    )?;

    let root = prepared.compact.files.first();
    let sha256 = root.map_or_else(String::new, |f| f.sha.clone());
    let fetched = Fetched {
        edges: &prepared.fetch.records,
        deps: &dependency_results,
        registries: &prepared.fetch.registries,
    };
    let mut decision = scored.decision;
    let llm = interpret(
        &prepared.typed,
        &index,
        fetched,
        &sha256,
        decision,
        &req,
        cpu_lease,
    );
    if let Some(Interpretation::Graded(graded)) = &llm.interpretation {
        blend_interpretation(
            req.label,
            req.model,
            graded,
            &mut decision.class,
            &mut decision.probability,
            &mut decision.level,
        );
    }

    let render_started = Instant::now();
    let rendered_context = if needs.render_context {
        let card = CardHead {
            decision: &decision,
            reasons: &scored.reasons,
            interpretation: llm.interpretation.as_ref(),
            sha256: &sha256,
            label: req.label,
            bloom_mark: req.bloom_mark,
            provenance: CardProvenance::default(),
        };
        render(
            &prepared.typed,
            &index,
            fetched,
            card,
            &scored.members,
            llm.context.as_deref(),
            &req,
        )
    } else {
        String::new()
    };
    let render_time = render_started.elapsed();

    let root = prepared.compact.files.first();
    let file_type = root.map_or_else(|| "unknown".to_string(), |f| f.file_type.clone());
    let size_bytes = root.map_or(0, |f| f.size);
    let formula = root.and_then(|f| f.formula.clone()).unwrap_or_default();
    // Retention rubric: everything above — featurization, the model, the
    // embedded pass, renders — consumed the complete report; what remains is
    // what gets stored (hopper bodies, JSON output). Sub-signal member nodes
    // are the bulk of that weight and nobody reads them again, so they are
    // dropped here (see `apply_report_retention`). A `--show=all` manifest
    // request is an explicit ask for the complete listing, so it opts out.
    let mut compact = prepared.compact;
    if !needs.list_all_members && !req.tuning.keep_all_members {
        apply_report_retention(&mut compact);
    }
    // Surface the archive members cleave catalogued but never analyzed, so a
    // `--show=all` JSON manifest lists every file (path/type/size) — not just the
    // ones that produced findings. Appended last, after featurization and the
    // embedded-file pass, so the listing never feeds the model.
    if !prepared.listed_members.is_empty() {
        append_unanalyzed_members(&mut compact, &prepared.listed_members);
    }

    let phase = PhaseTimings {
        fetch: prepared.fetch_time,
        interpret: llm.elapsed,
        render: render_time,
        total: started.elapsed(),
    };
    Ok(ClassifiedReport {
        phase,
        result: ScanResult {
            v: SCHEMA_VERSION,
            classification: decision.class,
            probability: decision.probability,
            threshold: decision.threshold,
            level: decision.level,
            version: String::new(),
            analyzed_at: String::new(),
            cleave: Some(compact),
            pids: None,
            deleted: None,
            path: req.label.to_string(),
            finding_counts: scored.finding_counts,
            formula,
            reasons: scored.reasons,
            top_findings,
            file_type,
            size_bytes,
            sha256,
            embedded_files: scored.members,
            model_scores: scored.model_scores,
            skipped_models: scored.skipped_models,
            rendered_context,
            interpretation: llm.interpretation,
            pending_llm: llm.pending,
            analysis_cached: prepared.analysis_cached,
            interpret_ms: crate::duration_ms(llm.elapsed),
            dependency_results,
            bloom_mark: req.bloom_mark,
            hopper_route: HopperRoute::Normal,
            model: scored.model,
            floor: scored.floor,
        },
    })
}

/// The report after cleave's finalize, reference fetching and grafting: what
/// featurization and every later stage read.
struct Prepared {
    /// The typed report: kept whole only when a later stage renders it.
    typed: AnalysisReport,
    compact: CompactReport,
    /// The sample's own files, captured before fetched content was grafted on.
    own_shas: HashSet<String>,
    fetch: crate::fetch::FetchOutcome,
    /// Per fetch edge, the length of the reference it cites — the span a
    /// dependency backref pins.
    ref_lengths: Vec<u64>,
    /// `--show=all`: every archive member cleave listed, analyzed or not.
    listed_members: Vec<ArchiveMemberStub>,
    fetch_time: Duration,
    /// Whether cleave replayed this analysis from its cache.
    analysis_cached: bool,
}

fn prepare(mut report: AnalysisReport, req: &ClassifyRequest<'_>) -> Prepared {
    let needs = req.needs;
    // Text and LLM renders consume the typed report; plain JSON does not.
    let keep_typed = needs.render_context
        || needs.llm_view
        || req.interpret.is_some()
        || req.tuning.interpret_dump_dir.is_some();
    // Read before the pipeline consumes `report`: whether cleave produced this
    // analysis or replayed it from its cache is the difference between a
    // request that did the work and one that did not.
    let analysis_cached = report.cache_hit;
    // Capture every archive member — including the ones cleave catalogues but
    // never analyzes (docs, data files, images) — before `finalize()` clears
    // `archive_contents`. With `--show=all` JSON output these are surfaced as
    // listing-only entries so the manifest is complete.
    let listed_members: Vec<ArchiveMemberStub> = if needs.list_all_members {
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
    // from these alone: fetched content is external — reached via a reference —
    // and may *escalate* the verdict through the per-file embedded path, but must
    // never dilute the sample's own aggregate (a benign fetched dependency would
    // otherwise mask a hostile manifest).
    let own_shas: HashSet<String> = report.files.iter().map(|f| f.sha256.clone()).collect();
    // Fetch the external references the analysis surfaced and graft each payload
    // into report.files as a uniform node — after finalize() (which populates
    // files[] and the per-file declared references) and before featurization, so
    // fetched content feeds the verdict like any other file. Off unless the
    // policy selects a kind.
    //
    // Standalone per-dependency captures (and their grading) exist for hopper
    // uploads, the LLM query payload, and the rendered dependency appendix. When
    // none of those consumers is active — a plain JSON scan with no upload
    // target — skip the capture up front.
    let capture_deps = needs.deps_for_upload || keep_typed;
    if let Some(p) = req.phase {
        p.set("fetch+graft");
    }
    let fetch_started = Instant::now();
    let fetch = crate::fetch::orchestrate(
        &mut report,
        req.root_path,
        req.fetch,
        needs.fetch_progress,
        capture_deps,
        req.zip_passwords,
    );
    let fetch_time = fetch_started.elapsed();
    if let Some(p) = req.phase {
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
    // the root itself; it keeps its provenance for interpret but gets no
    // duplicate graft.
    if let Some(reg) = req.root_registry
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
    let ref_lengths = reference_lengths(&fetch.records, &report);
    // Release the wide typed graph *during* compact conversion when nothing
    // renders it: the consuming variant drops each `FileAnalysis` as its compact
    // projection is built, so the typed graph and the compact copy never
    // co-reside (on a member-heavy sample the typed graph is the single largest
    // live allocation in the process).
    let mut compact = if keep_typed {
        cleave::types::compact::compact_from_files(&report.files)
    } else {
        let compact =
            cleave::types::compact::compact_from_files_consuming(std::mem::take(&mut report.files));
        report = AnalysisReport::new(report.target.clone());
        compact
    };
    validate_report_references(req.label, &compact);
    // Attach the fetch edge log at report level (`source_sha256 → content_sha256`
    // per reference). Report-level, not per-file: a fetch is a per-event
    // observation, so it never falsely dedups when content is exploded by hash.
    if !fetch.records.is_empty() {
        compact.fetched = fetch
            .records
            .iter()
            .filter_map(|edge| {
                serde_json::to_value(edge)
                    .map_err(|error| {
                        tracing::error!(locator = %edge.locator, %error, "fetch edge not recorded");
                    })
                    .ok()
            })
            .collect();
    }
    Prepared {
        typed: report,
        compact,
        own_shas,
        fetch,
        ref_lengths,
        listed_members,
        fetch_time,
        analysis_cached,
    }
}

/// Per fetch edge, the byte length of the reference it cites — the span a
/// dependency backref pins (1 when unknown). Read before compaction, which may
/// release the typed graph that knows it.
fn reference_lengths(edges: &[FetchRecord], report: &AnalysisReport) -> Vec<u64> {
    if edges.is_empty() {
        return Vec::new();
    }
    let mut by_sha: HashMap<&str, &FileAnalysis> = HashMap::new();
    for file in &report.files {
        by_sha.entry(file.sha256.as_str()).or_insert(file);
    }
    edges
        .iter()
        .map(|edge| {
            edge.source_offset.map_or(1, |offset| {
                edge.source_sha256
                    .as_deref()
                    .and_then(|sha| by_sha.get(sha))
                    .and_then(|file| file.filefacts.as_ref())
                    .and_then(|facts| facts.references.iter().find(|r| r.offset == offset))
                    .map_or(1, |r| u64::try_from(r.evidence.len()).unwrap_or(u64::MAX))
            })
        })
        .collect()
}

/// The model's verdict on the root and on every member, and what follows from
/// them: the elevated decision, and the dependency verdicts to pin back onto
/// the files that declared them.
struct Scored {
    /// The verdict: the root's, floored, raised by its worst member or adopted
    /// dependency verdict.
    decision: Decision,
    /// What the model alone concluded; see [`ScanResult::model`].
    model: Decision,
    /// The gravest trait-floor firing anywhere; see [`ScanResult::floor`].
    floor: Option<FloorDecision>,
    members: MemberEvals,
    reasons: Vec<Reason>,
    model_scores: Vec<RouteScore>,
    skipped_models: Vec<SkippedRoute>,
    finding_counts: FindingCounts,
    backrefs: Vec<DepBackref>,
}

fn score(prepared: &Prepared, req: &ClassifyRequest<'_>) -> Result<Scored> {
    let (label, model) = (req.label, req.model);
    let ctx = model.ctx();
    let compact = &prepared.compact;
    // Parse the report once with every optional raw subtree any specialist may
    // read, then share it across the root and embedded-file scoring passes.
    let needs = ctx.raw_needs().union(RawNeeds::all());
    // The sample's own decision featurizes its own files only. With nothing
    // fetched this is the whole report; otherwise drop the grafted payloads so
    // they can't dilute the aggregate (they still classify individually via the
    // embedded pass over the full report, where a hostile one elevates).
    let fetched = !prepared.fetch.records.is_empty();
    let parsed =
        ParsedReport::from_compact_report(compact, needs, fetched.then_some(&prepared.own_shas));
    let mut raw_features = ctx.extract_from_parsed(&parsed);
    let nonzero = raw_features.iter().filter(|&&v| v != 0.0).count();
    let expected = model.spec().total_features();
    if raw_features.len() != expected {
        bail!(
            "feature vector length mismatch: got {} expected {} — model/feature_spec out of sync",
            raw_features.len(),
            expected,
        );
    }
    // SHAP explanations use raw (unstandardized) values, so explain first, then
    // standardize in place for `predict()`.
    let reasons = req
        .shap
        .map(|s| s.explain(&raw_features))
        .unwrap_or_default();
    model.spec().standardize(&mut raw_features);

    // The cleave file_type drives ensemble routing.
    let root = compact.files.first();
    let file_type = root.map_or("unknown", |f| f.file_type.as_str());
    let (mut decision, model_scores, skipped_models) =
        model.predict_for_report_detailed(file_type, &raw_features, &parsed)?;
    let finding_counts = count_findings(compact);
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
        formula = %root.and_then(|f| f.formula.as_deref()).unwrap_or_default(),
        "classified file",
    );

    // Trait floor on the root's own findings. The pre-floor decision is kept so
    // the result can report the model and the floor as separate opinions.
    let root_model = decision;
    let root_floor = apply_trait_floor(
        &mut decision,
        root_findings(compact),
        model.active_level(),
        model.grid_max(),
        label,
    );

    // Every embedded file (depth > 0) is scored on its own and may elevate the
    // parent. A count cap here would make the result depend on archive ordering
    // and permit a hostile tail member to evade elevation.
    let entries: Vec<&CompactFile> = embedded_entries(compact).collect();
    let mut members = MemberEvals::new();
    for ef in score_embedded_files(&entries, label, needs, ctx, model, req.cancellation)? {
        tracing::debug!(
            parent = %label,
            embedded_path = %ef.path,
            probability = format!("{:.4}", ef.probability),
            classification = ?ef.classification,
            "classified embedded file",
        );
        members.insert(ef.id, ef);
    }

    // A fetched dependency that classifies hostile/suspicious is pinned back
    // onto the file that declared it, at the reference byte. Keyed by the
    // content sha of the retrieved payload.
    let declared: HashMap<&str, Declaration<'_>> = prepared
        .fetch
        .records
        .iter()
        .zip(&prepared.ref_lengths)
        .filter_map(|(edge, &len)| {
            let content = edge.content_sha256.as_deref()?;
            Some((
                content,
                Declaration {
                    source_sha: edge.source_sha256.as_deref(),
                    source_offset: edge.source_offset,
                    source_len: len,
                    locator: &edge.locator,
                },
            ))
        })
        .collect();
    let mut backrefs: Vec<DepBackref> = members
        .values()
        .filter(|ef| ef.classification >= Classification::Suspicious)
        .filter_map(|ef| {
            let declaration = declared.get(ef.sha256.as_str())?;
            Some(declaration.backref(&ef.sha256, &ef.file_type, ef.classification, None))
        })
        .collect();

    // Dependencies whose verdict came from hopper's corpus never reached the
    // embedded pass — nothing was analyzed — so they have no eval to be filtered
    // above. They are still this scan's answer for those bytes (the corpus
    // produced them under the analyzer this build runs), and a hostile one is
    // precisely what the backref exists to surface, so they are pinned the same
    // way and counted in the same elevation.
    let adopted: Vec<(Decision, &String, &crate::corpus_precheck::Verdict)> = prepared
        .fetch
        .adopted
        .iter()
        .filter_map(|(sha, v)| {
            let d = adopted_decision(v.fires_at, model.active_level(), model.grid_max())?;
            Some((d, sha, v))
        })
        .collect();
    if !adopted.is_empty() {
        let mut type_of: HashMap<&str, &str> = HashMap::new();
        for file in &compact.files {
            type_of.entry(file.sha.as_str()).or_insert(&file.file_type);
        }
        backrefs.extend(adopted.iter().filter_map(|(d, sha, v)| {
            if d.class < Classification::Suspicious {
                return None;
            }
            let declaration = declared.get(sha.as_str())?;
            let file_type = type_of.get(sha.as_str()).copied().unwrap_or_default();
            Some(declaration.backref(sha, file_type, d.class, adopted_detail(v)))
        }));
    }

    // The same fold, over the model's own readings: members the floor raised
    // are excluded because on those the model said benign, and the root's
    // pre-floor decision stands in for the root. Adopted corpus verdicts count
    // as model evidence — they are another scan's verdict, not this artifact's
    // traits. Purely observational; the verdict below does not read it.
    let model_decision = worst_member_model(&members)
        .into_iter()
        .chain(adopted.iter().map(|(d, ..)| *d))
        .filter(|candidate| decision_outranks(candidate, &root_model))
        .reduce(graver)
        .unwrap_or(root_model);
    let floor = match (root_floor, worst_member_floor(&members)) {
        (Some(root), Some(member)) => Some(root.worse_of(member)),
        (floor, None) | (None, floor) => floor,
    };

    // Elevate the container by its worst member and by any adopted dependency
    // verdict, so a hostile dependency weighs the same whether this scan
    // computed the verdict or the corpus handed it over.
    let elevated = worst_member(&members)
        .into_iter()
        .chain(adopted.into_iter().map(|(d, ..)| d))
        .filter(|worst| decision_outranks(worst, &decision))
        .reduce(graver);
    if let Some(worst) = elevated {
        tracing::info!(
            path = %label,
            original_probability = format!("{:.4}", decision.probability),
            elevated_probability = format!("{:.4}", worst.probability),
            elevated_classification = ?worst.class,
            elevated_threshold = format!("{:.4}", worst.threshold),
            "elevated archive classification due to embedded file",
        );
        decision = worst;
    }

    Ok(Scored {
        decision,
        model: model_decision,
        floor,
        members,
        reasons,
        model_scores,
        skipped_models,
        finding_counts,
        backrefs,
    })
}

/// Where a fetched payload was declared: the file and byte that named it.
struct Declaration<'a> {
    source_sha: Option<&'a str>,
    source_offset: Option<u64>,
    source_len: u64,
    locator: &'a str,
}

impl Declaration<'_> {
    fn backref(
        &self,
        dep_sha: &str,
        dep_type: &str,
        class: Classification,
        detail: Option<String>,
    ) -> DepBackref {
        DepBackref {
            source_sha: self.source_sha.map(str::to_owned),
            source_offset: self.source_offset,
            source_len: self.source_len,
            locator: self.locator.to_string(),
            dep_sha: dep_sha.to_string(),
            dep_type: dep_type.to_string(),
            class,
            detail,
        }
    }
}

/// Grade each fetched dependency on its own report, for hopper and the renders.
///
/// Each report is parsed once: graded, then trimmed by the retention rubric,
/// then kept as text (see `FetchedDependency::raw` for why text).
fn grade_dependencies(
    dependencies: Vec<crate::fetch::FetchedDependency>,
    registries: &[crate::fetch::DependencyRegistry],
    req: &ClassifyRequest<'_>,
) -> Result<Vec<DepResult>> {
    let provenance_by_locator: HashMap<&str, &crate::provenance::RegistryProvenance> = registries
        .iter()
        .map(|registry| (registry.locator.as_str(), &registry.provenance))
        .collect();
    dependencies
        .into_iter()
        .map(|dep| {
            let (verdict, members, raw) = match serde_json::from_str::<CompactReport>(&dep.raw) {
                Ok(mut report) => {
                    let graded = classify_dependency(&report, &dep.locator, req)?;
                    // Same retention rubric as the root report: grading consumed
                    // the complete report; the stored body keeps only the nodes
                    // someone will read again.
                    if !req.tuning.keep_all_members {
                        apply_report_retention(&mut report);
                    }
                    let raw = serde_json::to_string(&report).unwrap_or(dep.raw);
                    match graded {
                        Some((verdict, members)) => (Some(verdict), members, raw),
                        None => (None, MemberEvals::new(), raw),
                    }
                }
                Err(error) => {
                    // No verdict rather than an invented one; its bytes and
                    // provenance still go to hopper, which can analyze them.
                    tracing::warn!(locator = %dep.locator, %error, "dependency report does not parse; not graded");
                    (None, MemberEvals::new(), dep.raw)
                }
            };
            Ok(DepResult {
                verdict,
                members,
                sha256: dep.content_sha,
                provenance: provenance_by_locator
                    .get(dep.locator.as_str())
                    .map(|provenance| (**provenance).clone()),
                locator: dep.locator,
                url: dep.url,
                size: dep.size,
                raw,
            })
        })
        .collect()
}

/// Grade a fetched dependency on its own standalone report: the container's own
/// verdict, elevated by its worst member. That is exactly how a first-hand scan
/// of the same bytes resolves, which is the point — a dependency is an artifact
/// someone else's manifest happened to name, not a region of the report it was
/// grafted into.
///
/// `Ok(None)` when the feature vector does not match the model or the model
/// fails on it; the caller reports no verdict rather than inventing one.
///
/// # Errors
/// The analysis was cancelled.
fn classify_dependency(
    report: &CompactReport,
    label: &str,
    req: &ClassifyRequest<'_>,
) -> Result<Option<(Decision, MemberEvals)>> {
    let model = req.model;
    let ctx = model.ctx();
    let needs = ctx.raw_needs().union(RawNeeds::all());

    // The container's own verdict. No own_shas filtering: this report is the one
    // cleave produced for the dependency's bytes alone, before anything was
    // grafted onto it, so every file in it is the dependency's own.
    let parsed = ParsedReport::from_compact_report(report, needs, None);
    let mut features = ctx.extract_from_parsed(&parsed);
    if features.len() != model.spec().total_features() {
        tracing::warn!(
            dependency = label,
            got = features.len(),
            expected = model.spec().total_features(),
            "dependency not graded: feature vector length mismatch"
        );
        return Ok(None);
    }
    model.spec().standardize(&mut features);
    let file_type = report
        .files
        .first()
        .map_or("unknown", |f| f.file_type.as_str());
    let mut decision = match model.predict_for_report_detailed(file_type, &features, &parsed) {
        Ok((decision, _, _)) => decision,
        Err(error) => {
            tracing::warn!(
                dependency = label,
                error = format!("{error:#}"),
                "dependency not graded"
            );
            return Ok(None);
        }
    };
    apply_trait_floor(
        &mut decision,
        root_findings(report),
        model.active_level(),
        model.grid_max(),
        label,
    );

    // Every member elevates the dependency as it would in a first-hand scan.
    let entries: Vec<&CompactFile> = embedded_entries(report).collect();
    let members: MemberEvals =
        score_embedded_files(&entries, label, needs, ctx, model, req.cancellation)?
            .into_iter()
            .map(|ef| (ef.id, ef))
            .collect();
    let decision = members
        .values()
        .map(EmbeddedFile::decision)
        .fold(decision, graver);
    Ok(Some((decision, members)))
}

/// Every analyzed embedded node receives an individual model verdict.
///
/// This deliberately has no count limit: selecting only an archive prefix makes
/// malware detection depend on member ordering and lets a hostile tail member,
/// fetched dependency, or registry security sidecar evade container elevation.
fn embedded_entries(report: &CompactReport) -> impl Iterator<Item = &CompactFile> {
    report.files.iter().filter(|file| file.depth > 0)
}

/// Feature-extract and model-score a report's embedded files — the archive
/// members at depth > 0 — returning one evaluation per member the model could
/// score.
///
/// A member the model fails on is logged and left out: a gap in the table, not
/// a verdict. Nothing downstream may read an absent member as clean.
///
/// Per-member work is pure and runs in parallel: reports with thousands of
/// embedded files (nested npm tarballs, fetched dependency trees) previously ran
/// this serially on one rayon worker, and on member-heavy archives that pass —
/// not cleave's analysis — was the scan's wall-clock tail.
///
/// # Errors
/// The analysis was cancelled; the partial table is discarded.
fn score_embedded_files(
    entries: &[&CompactFile],
    label: &str,
    needs: RawNeeds,
    ctx: &ExtractContext,
    model: &Model,
    cancelled: Option<&AtomicBool>,
) -> Result<Vec<EmbeddedFile>> {
    use rayon::prelude::*;
    let is_cancelled = || cancelled.is_some_and(|c| c.load(Ordering::Relaxed));
    let scored = entries
        .par_iter()
        .filter_map(|&ef| {
            if is_cancelled() {
                return None;
            }
            let parsed = ParsedReport::from_compact_file(ef, needs);
            let mut features = ctx.extract_from_parsed(&parsed);
            model.spec().standardize(&mut features);
            let (mut decision, model_scores, skipped_models) =
                match model.predict_for_report_detailed(&ef.file_type, &features, &parsed) {
                    Ok(scored) => scored,
                    Err(error) => {
                        tracing::warn!(
                            container = label,
                            member = %ef.path,
                            error = format!("{error:#}"),
                            "member not scored; left out of the verdict"
                        );
                        return None;
                    }
                };
            // Trait floor on the member's own findings — a sparse, severe
            // dropper (the npm install-hook beacon lives in the embedded
            // package.json) is caught even when the container's findings
            // dilute it; the floored member then elevates the container.
            let floor = apply_trait_floor(
                &mut decision,
                &ef.findings,
                model.active_level(),
                model.grid_max(),
                if ef.path.is_empty() { label } else { &ef.path },
            );
            Some(EmbeddedFile {
                id: u64::from(ef.id),
                sha256: ef.sha.clone(),
                path: archive_leaf(&ef.path).to_string(),
                file_type: ef.file_type.clone(),
                classification: decision.class,
                probability: decision.probability,
                threshold: decision.threshold,
                level: decision.level,
                model_scores,
                skipped_models,
                formula: ef.formula.clone().unwrap_or_default(),
                top_findings: ef
                    .findings
                    .iter()
                    .filter(|f| compact_crit(f) >= Criticality::Suspicious)
                    .take(3)
                    .map(TopFinding::from)
                    .collect(),
                floor,
            })
        })
        .collect();
    if is_cancelled() {
        bail!("analysis cancelled during embedded file processing");
    }
    Ok(scored)
}

/// The LLM's part: the render it reads, and its opinion — run inline, or left
/// for the caller to run once it has posted the ML verdict.
struct Interpreted {
    /// The sanitized render, when anything reads it.
    context: Option<String>,
    interpretation: Option<Interpretation>,
    pending: Option<PendingLlm>,
    elapsed: Duration,
}

/// Build the LLM render and, when `--interpret` is on, get its opinion of the
/// verdict reached so far. Returns the caller's CPU admission before the
/// network round trip.
fn interpret(
    typed: &AnalysisReport,
    index: &ReportIndex<'_>,
    fetched: Fetched<'_>,
    sha256: &str,
    decision: Decision,
    req: &ClassifyRequest<'_>,
    cpu_lease: Option<CpuLease>,
) -> Interpreted {
    let (label, model) = (req.label, req.model);
    // Built once: it feeds the model and, when `SCAN_INTERPRET_DUMP_DIR` is set,
    // is written to `<dir>/<sha256>.render` — the raw render, so the
    // prompt-tuning harness (`hacks/interpret-tune`) can sweep render variants
    // offline from one scan, independent of whether `--interpret` is on.
    let dump_dir = req.tuning.interpret_dump_dir.as_deref();
    let context =
        (req.interpret.is_some() || dump_dir.is_some() || req.needs.llm_view).then(|| {
            let primary = Primary {
                label,
                sha256,
                fetch: req.root_fetch,
                registry: req.root_registry,
            };
            let rendered = render_interpret_context(&primary, fetched, typed, index, req.tuning);
            crate::interpret::sanitize_context(&rendered)
        });
    if let (Some(dir), Some(context)) = (dump_dir, context.as_deref()) {
        let path = dir.join(format!("{sha256}.render"));
        if let Err(error) =
            std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, context))
        {
            tracing::warn!(path = %path.display(), %error, "could not write the interpret render dump");
        }
    }
    // CPU work is done: featurized, scored, dependencies graded. What follows
    // is a network wait (the LLM) and light rendering. Give the admission
    // permit back now so the pool is not idle for the round trip. A caller that
    // gave a lease owns the tail: it posts the ML verdict now and runs the LLM
    // step from `pending` afterwards.
    let owner_runs_llm = cpu_lease.is_some();
    if let Some(release) = cpu_lease {
        release();
    }
    let started = Instant::now();
    let levels = crate::interpret::LevelContext {
        fired: decision.level,
        active: model.active_level(),
        grid_max: model.grid_max(),
    };
    let mut pending = None;
    let mut interpretation = None;
    if let (Some(cfg), Some(ctx)) = (req.interpret, context.as_deref()) {
        // cleave's own verdict, read from the structured report rather than
        // re-parsed out of the render it produced. See `FindingSeverity`.
        let findings = crate::interpret::FindingSeverity::from_report(typed);
        let admitted = crate::interpret::admit(
            cfg,
            decision.class,
            decision.probability,
            levels,
            findings,
            ctx,
            label,
        );
        if owner_runs_llm {
            pending = admitted.map(|admitted| PendingLlm {
                ctx: ctx.to_string(),
                admitted,
            });
        } else if let Some(admitted) = admitted {
            // Census label for the round trip; without it the wait read as
            // "features+model", hiding that the pool was idle for the network.
            if let Some(p) = req.phase {
                p.set("interpret");
            }
            // Inline means a caller is waiting (a serve request or an
            // interactive scan), so it never queues behind background work.
            interpretation = crate::interpret::interpret_admitted(
                cfg,
                &admitted,
                ctx,
                crate::interpret::LlmCaller::Foreground,
            );
            if let Some(Interpretation::Graded(graded)) = &interpretation {
                log_interpretation(label, sha256, graded);
            }
        }
    }
    Interpreted {
        context,
        interpretation,
        pending,
        // The LLM round trip (queue wait + generation) against a shared
        // endpoint: the usual suspect for a slow run.
        elapsed: started.elapsed(),
    }
}

/// What the card's provenance rows say about the root. A fetched root has
/// none: `pkg`/`url` already printed the registry banner before the fetch.
fn card_provenance(req: &ClassifyRequest<'_>, sha256: &str) -> CardProvenance {
    if req.root_fetch.is_some() {
        return CardProvenance::default();
    }
    CardProvenance {
        purl: crate::provenance::collector_purl(req.root_path, sha256),
        registry: req
            .root_registry
            .map(|provenance| crate::fetch::registry_summary(&provenance.record))
            .filter(|parts| !parts.is_empty())
            .map(|parts| parts.join(" \u{00b7} ")),
    }
}

/// The text render the caller's format asks for: the terminal card, the LLM
/// payload (`--format interpret`), or cleave's context (`--format tiny`).
fn render(
    typed: &AnalysisReport,
    index: &ReportIndex<'_>,
    fetched: Fetched<'_>,
    mut card: CardHead<'_>,
    members: &MemberEvals,
    llm_context: Option<&str>,
    req: &ClassifyRequest<'_>,
) -> String {
    if req.needs.llm_view {
        // `--format interpret`: byte-for-byte the user message the live
        // `--interpret` query sends — the sanitized render with its annotations
        // recategorized, just without the system prompt. The recategorization is
        // applied here and in `interpret::interpret_admitted`, never in
        // `render_interpret_context` itself: scan parses that render back for the
        // LLM admission gate and for `Evidence` (`has_elevated_finding`,
        // `has_hostile_finding`, `render_mostly_readable`), all of which key on
        // the `SEV` letter this transform removes. Stripping it upstream silently
        // withdrew samples from the gate — measured as seven true positives
        // dropping from a caught verdict to `Level::Clean`. The dependency appendix
        // is already part of that render.
        return recategorize_annotations(llm_context.unwrap_or_default());
    }
    if req.tiny_opts.header == cleave::output::HeaderStyle::Rich {
        // The terminal card covers the sample's own files; fetched payloads and
        // registry sidecars get their own account below it.
        let registry_ids: HashSet<u32> = fetched.registries.iter().map(|r| r.file_id).collect();
        let own: Vec<&FileAnalysis> = typed
            .files
            .iter()
            .filter(|file| !index.is_fetched(file.id) && !registry_ids.contains(&file.id))
            .collect();
        card.provenance = card_provenance(req, card.sha256);
        let mut rendered = render_terminal_context(&own, &card, members);
        if let Some(account) = render_terminal_fetch_context(fetched, typed, index) {
            rendered.push_str(&account);
        }
        return rendered;
    }
    // The dependency appendix goes to every text format: it is the only place a
    // render states that a verdict was inherited from something the sample
    // merely pointed at — naming each dependency, the locator it came from, its
    // own classification, and its elevated findings.
    let mut rendered = cleave::output::format_context(typed, &req.tiny_opts);
    if let Some(appendix) = render_dependency_context(fetched, typed, index) {
        rendered.push_str(&appendix);
    }
    rendered
}

/// The findings of a report's root file — the ones that describe the artifact
/// scan was asked about, as opposed to its members.
pub(super) fn root_findings(report: &CompactReport) -> &[CompactTrait] {
    report.files.first().map_or(&[], |f| &f.findings)
}

/// Fold an LLM interpretation into the verdict: the three ways a second
/// opinion may move class, probability and level. Shared by the inline
/// path in [`classify_report`] and the deferred one in
/// [`apply_pending_interpretation`], so both land on the same answer.
fn blend_interpretation(
    label: &str,
    model: &Model,
    graded: &Graded,
    class: &mut Classification,
    probability: &mut f32,
    lvl: &mut Level,
) {
    // Adopt the blended verdict as the effective one when the LLM out-read ML
    // (escalating a missed threat, or clearing an ML false positive). The `ml`
    // section reflects litmus's final answer; the LLM's raw grade + rationale
    // stay in the `llm` section. The interpreted level is pinned to the target
    // band's loosest rung (see `interpreted_level`): the active hostile level for
    // an escalation, the suspicious ceiling for a hold/downgrade, clean for benign.
    if graded.outcome != *class {
        // INFO, not WARN: an LLM override of the ML verdict is normal operation,
        // not a fault. (It also kept surfacing as the last stderr line a caller
        // grabbed when a slow run was externally killed, making a benign shift look
        // like a crash cause.)
        tracing::info!(
            path = %label,
            ml = ?*class,
            outcome = ?graded.outcome,
            grade = graded.grade.as_str(),
            conf = format!("{:.4}", graded.blended),
            reason = %graded.interpretation,
            "LLM interpretation shifted the verdict",
        );
        let ml_class = *class;
        let ml_level = *lvl;
        *class = graded.outcome;
        *probability = graded.blended;
        *lvl = if ml_class == Classification::Hostile
            && graded.outcome == Classification::Suspicious
        {
            // A cleared hostile is placed by how deep ML fired, not pinned to the
            // band's edge: the level ML reached is the budget for how far one
            // contrary opinion may move it.
            softened_level(ml_level, model.active_level(), model.grid_max())
        } else {
            interpreted_level(
                model.active_level(),
                model.grid_max(),
                graded.outcome,
                graded.corroborated,
            )
        };
    } else if graded.grade == LlmGrade::Benign
        && *class == Classification::Hostile
        && *lvl == Level::At(0)
    {
        // `may_cross` refused to move a verdict off the grid's tightest budget on
        // one contrary opinion, and that stands — but the disagreement is still
        // evidence, so the verdict gives up the depth it cannot justify and sits
        // on the weakest hostile rung instead. Still blocked, still reviewed.
        let weakened = model.active_level().map_or(Level::Manual, Level::At);
        tracing::info!(
            path = %label,
            from = 0,
            to = %weakened,
            reason = %graded.interpretation,
            "LLM cleared an L0 hostile — held in band, moved to the weakest rung",
        );
        *lvl = weakened;
    } else if graded.grade == LlmGrade::Hostile
        && *class == Classification::Hostile
        && let Level::At(level) = *lvl
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
        *lvl = Level::At(strengthened);
    }
}

/// The INFO line for a graded second opinion, inline or deferred.
fn log_interpretation(file: &str, sha256: &str, graded: &Graded) {
    tracing::info!(
        file = %file,
        sha256 = %sha256,
        grade = graded.grade.as_str(),
        outcome = %graded.outcome,
        conf = format!("{:.4}", graded.blended),
        cached = graded.cached,
        interpretation = %graded.interpretation,
        "LLM interpretation",
    );
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
    /// The gate's verdict, and the ML reading it was made on.
    pub admitted: Admitted,
}

/// Run a deferred second opinion and fold it into `result` exactly as
/// [`classify_report`] would have inline. Returns whether anything changed
/// and therefore needs re-posting: the LLM section itself is new information,
/// so any interpretation counts. `false` when nothing was pending or the
/// render was empty.
pub(crate) fn apply_pending_interpretation(
    result: &mut ScanResult,
    cfg: &crate::interpret::InterpretConfig,
    model: &Model,
) -> bool {
    let Some(pending) = result.pending_llm.take() else {
        return false;
    };
    // Deferred means nobody is on the line: a worker job, or serve's own idle
    // puller. It takes its permit behind foreground callers.
    let Some(interp) = crate::interpret::interpret_admitted(
        cfg,
        &pending.admitted,
        &pending.ctx,
        crate::interpret::LlmCaller::Background,
    ) else {
        return false;
    };
    if let Interpretation::Graded(graded) = &interp {
        log_interpretation(&result.path, &result.sha256, graded);
        blend_interpretation(
            &result.path,
            model,
            graded,
            &mut result.classification,
            &mut result.probability,
            &mut result.level,
        );
    }
    result.interpretation = Some(interp);
    true
}

/// Whether registry metadata is external context that needs grafting beneath
/// the root. A metadata-only package fallback already analyzes the registry
/// document as its root and must not receive a duplicate sidecar node.
pub(super) fn root_needs_registry_graft(report: &cleave::AnalysisReport) -> bool {
    !report
        .files
        .first()
        .is_some_and(|file| file.file_type == "registry")
}

/// One confirmed hostile/suspicious fetched dependency, ready to pin back onto
/// the file that declared it: the declaring file and reference byte, plus the
/// dependency's identity — locator (PURL/URL), content sha, sniffed file type,
/// and class.
struct DepBackref {
    source_sha: Option<String>,
    source_offset: Option<u64>,
    source_len: u64,
    locator: String,
    dep_sha: String,
    dep_type: String,
    class: Classification,
    /// Why, in the corpus's own words, when this verdict was adopted rather than
    /// computed here — the stored reason, else its strongest finding. `None` for
    /// a dependency this scan analyzed itself: its traits are already in the
    /// report, a `!!`-path away, and repeating one in the backref prose would be
    /// the same fact twice.
    detail: Option<String>,
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
fn inject_dependency_backref(report: &mut CompactReport, backref: &DepBackref) {
    let (crit, sev) = match backref.class {
        Classification::Hostile => (5u8, "Malicious"),
        _ => (4u8, "Suspicious"),
    };
    let desc = match &backref.detail {
        Some(detail) => format!(
            "{sev} dependency: {} | {} — {detail}",
            backref.locator, backref.dep_sha
        ),
        None => format!(
            "{sev} dependency: {} | {}",
            backref.locator, backref.dep_sha
        ),
    };
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
        .find(|f| Some(f.sha.as_str()) == backref.source_sha.as_deref())
        .map(|f| f.path.clone());

    for f in &mut report.files {
        let is_decl = Some(f.sha.as_str()) == backref.source_sha.as_deref();
        let is_ancestor = decl_path
            .as_deref()
            .is_some_and(|dp| is_inside(dp, &f.path));
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

/// The decision an adopted corpus verdict carries at this deploy's level.
///
/// `fires_at` is the same measured quantity [`Decision::level`] holds — the
/// tightest false-positive budget at which these bytes grade hostile — so the
/// rule that turns it into a class is the model's own [`verdict_for_level`],
/// never a second implementation of it. There is no probability: no model ran
/// here. `class` is what elevation and the backref read, and `level` is what
/// travels onward, so the missing score costs nothing but a tie-break, which
/// an adopted verdict should lose anyway.
///
/// `None` in manual-threshold mode, where no level table applies and no honest
/// class can be derived — the same answer `decide` gives for that case.
fn adopted_decision(fires_at: Level, active_level: Option<u16>, grid_max: u16) -> Option<Decision> {
    let class = match fires_at {
        Level::Clean => Classification::Benign,
        Level::At(fired) => crate::model::verdict_for_level(fired, active_level?, grid_max),
        Level::Manual => return None,
    };
    Some(Decision {
        class,
        probability: 0.0,
        threshold: 0.0,
        level: fires_at,
    })
}

/// What an adopted verdict says for itself: the corpus's own sentence, else the
/// id of its strongest finding. A dependency skipped on the corpus's word has no
/// traits in this report to point at, so without this the backref would name a
/// malicious package and offer nothing behind the claim.
fn adopted_detail(v: &crate::corpus_precheck::Verdict) -> Option<String> {
    if let Some(reason) = v.reason.as_deref().map(str::trim).filter(|r| !r.is_empty()) {
        return Some(reason.to_string());
    }
    v.findings.iter().max_by_key(|f| f.crit).map(|f| {
        if f.desc.is_empty() {
            f.id.clone()
        } else {
            f.desc.clone()
        }
    })
}

/// Count a report's findings by criticality — the root file's own findings,
/// which are the ones that describe the artifact scan was asked about.
#[must_use]
pub fn count_findings(report: &CompactReport) -> FindingCounts {
    let mut counts = FindingCounts::default();
    for f in root_findings(report) {
        match compact_crit(f) {
            Criticality::Hostile => counts.hostile += 1,
            Criticality::Suspicious => counts.suspicious += 1,
            Criticality::Notable => counts.notable += 1,
            _ => counts.baseline += 1,
        }
    }
    counts
}

/// Extract a small set of human-facing findings relevant to the classification.
#[must_use]
pub fn extract_top_findings(
    findings: &[CompactTrait],
    classification: &Classification,
) -> Vec<TopFinding> {
    let min_crit = match classification {
        Classification::Hostile => Criticality::Hostile,
        Classification::Suspicious | Classification::Benign => Criticality::Suspicious,
    };
    let at_least = |crit: Criticality| -> Vec<TopFinding> {
        findings
            .iter()
            .filter(|f| compact_crit(f) >= crit)
            .map(TopFinding::from)
            .collect()
    };
    let mut relevant = at_least(min_crit);
    // Fall back to suspicious-level findings if no hostile-level findings.
    if relevant.is_empty() && min_crit == Criticality::Hostile {
        relevant = at_least(Criticality::Suspicious);
    }

    // Deduplicate by base ID.
    let mut seen = HashSet::new();
    relevant.retain(|f| {
        let base = f.id.split("::").next().unwrap_or(&f.id);
        seen.insert(base.to_string())
    });

    relevant.sort_by_key(|f| std::cmp::Reverse(f.crit));
    relevant.truncate(5);
    relevant
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
mod tests {
    use super::super::dep_envelope;
    use super::*;
    use cleave::types::Role;

    /// Grade `raw` as `classify_report` grades a fetched dependency.
    fn grade(raw: &str, label: &str, model: &Model) -> Option<(Decision, MemberEvals)> {
        let report: CompactReport = serde_json::from_str(raw).expect("dependency report parses");
        let req = ClassifyRequest::new(label, std::path::Path::new(label), model);
        classify_dependency(&report, label, &req).expect("not cancelled")
    }

    /// The former production cutoff. Tests deliberately put evidence beyond it
    /// so reintroducing a prefix-only grading policy cannot pass unnoticed.
    const FORMER_EMBEDDED_FILE_LIMIT: usize = 100;

    /// Grading needs a real model to score against. Point SCAN_MODELS_DIR at a
    /// bundle — a single model (`model.onnx`) or a routed one (`config.json`) —
    /// to run these; without one there is nothing to exercise, so they skip
    /// rather than fail.
    fn model_bundle() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from(std::env::var_os("SCAN_MODELS_DIR")?);
        (p.join("model.onnx").is_file() || p.join("config.json").is_file()).then_some(p)
    }

    /// One dependency node, plus `members` javascript members beneath it.
    fn dep_report(members: usize, member_traits: &str) -> String {
        let files: Vec<serde_json::Value> = std::iter::once(serde_json::json!({
            "id": 0, "sha": "d".repeat(64), "type": "npm", "depth": 0, "size": 1,
            "path": "evil-1.0.0.tgz"
        }))
        .chain((0..members).map(|i| {
            let mut f = serde_json::json!({
                "id": i + 1, "sha": format!("{:064x}", i), "type": "javascript",
                "depth": 1, "size": 1, "path": format!("evil-1.0.0.tgz!!lib/{i}.js"),
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

        let count = FORMER_EMBEDDED_FILE_LIMIT * 2;
        let raw = dep_report(count, "");
        let (_verdict, members) = grade(&raw, "pkg:npm/evil@1.0.0", &model)
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
            // Enough corroborated evidence to clear the hostile floor arm
            // (two anchors, three severe findings, three families), so the
            // verdict does not hinge on how the model reads a registry file.
            "traits": [{
                "id": "objectives/supply-chain/impersonation/registry/publish::registry-takedown-security-hold",
                "crit": 5, "conf": 0.99,
                "desc": "Registry takedown marks package malicious"
            }, {
                "id": "objectives/command-and-control/backdoor/rat::beacon",
                "crit": 5, "conf": 0.95
            }, {
                "id": "objectives/execution/install-hook/npm::postinstall",
                "crit": 4, "conf": 0.9
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
                .expect("not cancelled")
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
        let dep = crate::fetch::FetchedDependency {
            locator: "pkg:npm/x@1".to_string(),
            url: "https://reg.test/x-1.tgz".to_string(),
            content_sha: "d".repeat(64),
            size: 1,
            raw: "{not json".to_string(),
        };
        let req = ClassifyRequest::new("x", std::path::Path::new("x"), &model);
        let graded = grade_dependencies(vec![dep], &[], &req).expect("not cancelled");
        assert!(
            graded[0].verdict.is_none(),
            "an unparseable report must yield no verdict",
        );
        assert_eq!(
            graded[0].raw, "{not json",
            "its bytes still go to hopper as they came"
        );
    }

    /// A member the model cannot score is a gap, never a benign verdict, and a
    /// cancelled pass is an error, never a table of fabricated members.
    #[test]
    fn cancellation_is_an_error_not_a_clean_member_table() {
        let Some(dir) = model_bundle() else {
            eprintln!("skipping: no model bundle (set SCAN_MODELS_DIR)");
            return;
        };
        let model = Model::load(&dir, None, None).expect("load model bundle");
        let ctx = ExtractContext::new(model.spec());
        let report: CompactReport = serde_json::from_str(&dep_report(3, "")).unwrap();
        let entries: Vec<&CompactFile> = embedded_entries(&report).collect();
        let needs = ctx.raw_needs().union(RawNeeds::all());
        let cancelled = AtomicBool::new(true);
        assert!(
            score_embedded_files(&entries, "d.tgz", needs, &ctx, &model, Some(&cancelled)).is_err()
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

        let (clean, _) =
            grade(&dep_report(3, ""), "pkg:npm/a@1", &model).expect("clean report grades");
        let (severe, _) = grade(
            &dep_report(
                3,
                r#"[{"id":"objectives/command-and-control/backdoor::a","crit":5,"conf":1.0},
                    {"id":"objectives/exfiltration/credentials::b","crit":5,"conf":0.98}]"#,
            ),
            "pkg:npm/b@1",
            &model,
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
    /// Nothing in the types enforces that — it holds only because
    /// `classify_report` injects after the `score` stage returns, and `score`
    /// is where featurization happens. This pins that order so a future edit
    /// that moves either one fails here instead of silently contaminating
    /// training data. Delete this test only by making the distinction explicit
    /// in the trait itself.
    #[test]
    fn dependency_backref_is_injected_after_featurization() {
        // Production code only: the needles below also appear in this test.
        let src = include_str!("pipeline.rs");
        let src = src.split("#[cfg(test)]").next().unwrap_or(src);
        let featurize = src
            .find("let scored = score(&prepared, &req)?;")
            .expect("the featurizing stage");
        let inject = src
            .find("inject_dependency_backref(&mut prepared.compact, backref);")
            .expect("backref injection call");
        assert_eq!(
            src.matches("inject_dependency_backref(").count(),
            2,
            "one definition, one call",
        );
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

        let raw = dep_report(
            3,
            r#"[{"id":"objectives/command-and-control/backdoor::a","crit":5,"conf":1.0}]"#,
        );
        let (verdict, members) =
            grade(&raw, "pkg:npm/evil@1.0.0", &model).expect("dependency grades");

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
            .expect("the report parses")
            .expect("a graded dependency yields an envelope");

        // The same bytes as a first-hand scan would produce them.
        let direct = ScanResult {
            v: SCHEMA_VERSION,
            classification: verdict.class,
            probability: verdict.probability,
            threshold: verdict.threshold,
            level: verdict.level,
            version: "model-9".to_string(),
            analyzed_at: "2026-06-28T00:00:00Z".to_string(),
            cleave: Some(serde_json::from_str(&raw).expect("report parses")),
            embedded_files: dep.members,
            ..super::super::envelope::tests::base_result()
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
            dep_env.ml.files.iter().any(|f| f.verdict.is_some()),
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

    /// A verdict adopted from the corpus is graded by the model's own
    /// level rule — the one line that must never be re-implemented — and carries
    /// the level onward, with no probability to fabricate.
    #[test]
    fn an_adopted_verdict_is_graded_by_the_model_level_rule() {
        let grid_max = 25_000;
        // Fires at a level tighter than this deploy's budget: hostile.
        let d = adopted_decision(Level::At(2), Some(25), grid_max).expect("graded");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.level, Level::At(2));
        assert_eq!(d.probability, 0.0, "no model ran; nothing to report");
        // Fires only far above the budget, inside the suspicious band.
        assert_eq!(
            adopted_decision(Level::At(200), Some(25), grid_max)
                .expect("graded")
                .class,
            Classification::Suspicious
        );
        // Clean is the absence of a level, never the tightest one.
        let clean = adopted_decision(Level::Clean, Some(25), grid_max).expect("graded");
        assert_eq!(clean.class, Classification::Benign);
        assert_eq!(clean.level, Level::Clean);
        // Manual-threshold mode has no level table, so there is no honest class.
        assert!(adopted_decision(Level::At(2), None, grid_max).is_none());
    }

    /// The backref names a package as malicious, so it has to say what for. A
    /// dependency skipped on the corpus's word has no traits in this report to
    /// point at — the reason, else its worst finding, is all the evidence there
    /// is.
    #[test]
    fn an_adopted_backref_carries_its_evidence() {
        use crate::corpus_precheck::{Finding, Verdict};
        let finding = |id: &str, desc: &str, crit: u32| Finding {
            id: id.to_string(),
            desc: desc.to_string(),
            crit,
        };
        let verdict = |reason: Option<&str>, findings: Vec<Finding>| Verdict {
            fires_at: Level::At(2),
            reason: reason.map(String::from),
            findings,
        };
        assert_eq!(
            adopted_detail(&verdict(Some("steals credentials"), Vec::new())).as_deref(),
            Some("steals credentials")
        );
        // No sentence: the strongest finding stands in, preferring its prose.
        assert_eq!(
            adopted_detail(&verdict(
                None,
                vec![
                    finding("feed/osv", "cited by OSV", 4),
                    finding("objectives/exfil::env", "", 5),
                ],
            ))
            .as_deref(),
            Some("objectives/exfil::env")
        );
        // Nothing to say is said as nothing, not as an empty clause.
        assert_eq!(adopted_detail(&verdict(Some("  "), Vec::new())), None);
        assert_eq!(adopted_detail(&verdict(None, Vec::new())), None);
    }

    fn backref(class: Classification) -> DepBackref {
        DepBackref {
            source_sha: Some("s".repeat(64)),
            source_offset: Some(42),
            source_len: 9,
            locator: "pkg:npm/zaboodle@1.49".to_string(),
            dep_sha: "d".repeat(64),
            dep_type: "javascript".to_string(),
            class,
            detail: None,
        }
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
