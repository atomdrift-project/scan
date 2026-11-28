//! The package-aware render the LLM reads (`--llm`,
//! `--format interpret`), and the fetched-subject selection the terminal
//! render shares with it.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use cleave::types::Rel;
use cleave::{AnalysisReport, Criticality, FileAnalysis};
use fletch::fetch::FetchRecord;

use super::{DepResult, is_inside};
use crate::fetch::DependencyRegistry;
use crate::model::{Classification, Decision, Level};
use crate::provenance::RegistryProvenance;

/// Cap on elevated-finding lines rendered per fetched dependency in the LLM
/// context appendix; a dependency with more still shows its worst, and the
/// omission is stated so the model never mistakes the cut for completeness.
const MAX_DEP_FINDING_LINES: usize = 12;

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
/// pool-adjusted false-positive level at which its hostile decision
/// still fires, so a *lower* level is the more confident call and a clean one
/// fires at no level. Probability breaks ties within a level.
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
        // a level of 3000 (one FP plus an estimated additional 2999/100M).
        Some(
            d @ Decision {
                level: Level::At(level),
                ..
            },
        ) => (3, -i32::from(level), prob_key(d)),
        _ if severe_finding => (2, 0, 0),
        Some(d) => (1, 0, prob_key(d)),
        None => (0, 0, 0),
    }
}

/// Upper bound, in bytes, on the assembled LLM user message: the primary
/// subject and every dependency subject together.
///
/// The render is one prompt, and the GPU pays for every token of it. Prompts
/// over 20k tokens are a small share of requests but a large share of all
/// prefill, and the outliers are structural — a jar inside a wheel fanning out
/// to 83 `.class` members, a wheel with 163 Mach-O members each carrying a hex
/// window — not evidence anyone reads. The render tokenizes at ~2 bytes/token
/// on those (hex rows and JSON), so 96 KiB is roughly 32–48k tokens, past the
/// p95 of what the model is asked to read and under the point where one
/// request holds the KV cache against everyone else.
///
/// It is also a hard cap. An endpoint refuses the whole call when the prompt
/// exceeds the model's context (a 2 MB render was rejected at 262081 input
/// tokens against a 262144-token window). Keeping every suspicious member
/// "whatever it costs" produces that refusal, which is zero evidence. The
/// budget keeps the strongest evidence that fits.
const INTERPRET_PROMPT_BUDGET_BYTES: usize = 96 * 1024;

/// Bytes held back from the evidence budget for the one-line notes that say
/// content was cut (`members_omitted`, `deps_omitted`). Those notes are part
/// of the prompt, and the evidence is sized so they still fit.
const BUDGET_NOTE: usize = 96;

/// What [`budget_primary_context`] removed so the render would fit.
struct BudgetedRender {
    text: String,
    members_omitted: usize,
    findings_omitted: usize,
}

/// Render `primary` for the LLM, dropping the least relevant content until the
/// render fits in `budget` bytes.
///
/// Order is fixed, and each step runs only when the previous one left the
/// render over budget:
///
/// 1. Members (never a root of this view) whose findings all sit below
///    suspicious. These are hinting and metadata.
/// 2. The remaining members, weakest first: lowest peak criticality, then
///    fewest findings. A suspicious member is dropped only once nothing weaker
///    remains. An over-budget prompt is refused whole, so that member would
///    not have been read anyway. A dropped member takes its nested members
///    with it, so the render never shows a child under a container it no
///    longer lists.
/// 3. Individual findings on what remains, weakest first, except the single
///    strongest finding. A subject that is one enormous file cannot shed
///    members; its least relevant notes go instead.
///
/// Within one rank, drops are proportional to the overshoot so a large archive
/// converges in a handful of renders, and a batch never crosses into a
/// stronger rank — the hostile member is not spent to pay for a suspicious one.
fn budget_primary_context(primary: &mut AnalysisReport, budget: usize) -> BudgetedRender {
    let mut rendered = crate::interpret::payload::render(primary);
    if rendered.len() <= budget {
        return BudgetedRender {
            text: rendered,
            members_omitted: 0,
            findings_omitted: 0,
        };
    }
    let roots = subject_roots(primary);
    let mut members_omitted = shed_members(primary, &mut rendered, budget, &roots, |f| {
        f.findings.iter().all(|t| t.crit < Criticality::Suspicious)
    });
    if rendered.len() > budget {
        members_omitted += shed_members(primary, &mut rendered, budget, &roots, |_| true);
    }
    let findings_before: usize = primary.files.iter().map(|f| f.findings.len()).sum();
    if rendered.len() > budget {
        shed_findings(primary, &mut rendered, budget);
    }
    let findings_after: usize = primary.files.iter().map(|f| f.findings.len()).sum();
    BudgetedRender {
        text: rendered,
        members_omitted,
        findings_omitted: findings_before.saturating_sub(findings_after),
    }
}

/// File ids at the shallowest depth in `report`: the subject this view is
/// about. A dependency view is a slice of the parent report, so its root is
/// not file 0 and not depth 0.
fn subject_roots(report: &AnalysisReport) -> HashSet<u32> {
    let Some(depth) = report.files.iter().map(|f| f.depth).min() else {
        return HashSet::new();
    };
    report
        .files
        .iter()
        .filter(|f| f.depth == depth)
        .map(|f| f.id)
        .collect()
}

/// Drop members `droppable` names, weakest first, until `rendered` fits.
/// `roots` are never dropped. Returns how many files went.
fn shed_members(
    report: &mut AnalysisReport,
    rendered: &mut String,
    budget: usize,
    roots: &HashSet<u32>,
    mut droppable: impl FnMut(&FileAnalysis) -> bool,
) -> usize {
    let mut order: Vec<((Criticality, usize), u32)> = report
        .files
        .iter()
        .filter(|f| !roots.contains(&f.id) && droppable(f))
        .map(|f| {
            let peak = f.findings.iter().map(|t| t.crit).max().unwrap_or_default();
            ((peak, f.findings.len()), f.id)
        })
        .collect();
    order.sort_by_key(|(key, _)| *key);

    let mut dropped = 0usize;
    while rendered.len() > budget && !order.is_empty() {
        let lead = order[0].0;
        let same = order.iter().take_while(|(key, _)| *key == lead).count();
        let per_file = rendered.len() / report.files.len().max(1);
        let n = ((rendered.len() - budget) / per_file.max(1)).clamp(1, same);
        let mut victims: HashSet<u32> = order.drain(..n).map(|(_, id)| id).collect();
        loop {
            let before = victims.len();
            for f in &report.files {
                if !roots.contains(&f.id) && f.parent_id.is_some_and(|p| victims.contains(&p)) {
                    victims.insert(f.id);
                }
            }
            if victims.len() == before {
                break;
            }
        }
        order.retain(|(_, id)| !victims.contains(id));
        let before = report.files.len();
        report.files.retain(|f| !victims.contains(&f.id));
        dropped += before - report.files.len();
        *rendered = crate::interpret::payload::render(report);
    }
    dropped
}

/// Drop the weakest findings until `rendered` fits, keeping the single
/// strongest so a one-finding subject still has something to grade.
fn shed_findings(report: &mut AnalysisReport, rendered: &mut String, budget: usize) {
    let protected = report
        .files
        .iter()
        .flat_map(|f| f.findings.iter().map(move |finding| (f.id, finding)))
        .max_by(|a, b| {
            a.1.crit
                .cmp(&b.1.crit)
                .then(a.1.conf.total_cmp(&b.1.conf))
                .then(a.0.cmp(&b.0))
                .then(a.1.id.cmp(&b.1.id))
        })
        .map(|(file, finding)| (file, finding.id.clone()));
    while rendered.len() > budget {
        let mut ranked: Vec<(Criticality, u32, u32, usize)> = Vec::new();
        for file in &report.files {
            for (index, finding) in file.findings.iter().enumerate() {
                let keep = protected
                    .as_ref()
                    .is_some_and(|(id, finding_id)| *id == file.id && finding_id == &finding.id);
                if keep {
                    continue;
                }
                ranked.push((
                    finding.crit,
                    finding.conf.max(0.0).to_bits(),
                    file.id,
                    index,
                ));
            }
        }
        if ranked.is_empty() {
            break;
        }
        ranked.sort_by_key(|key| (key.0, key.1));
        let lead = (ranked[0].0, ranked[0].1);
        let same = ranked
            .iter()
            .take_while(|key| (key.0, key.1) == lead)
            .count();
        // A finding note is a short line; size the batch from that, not from
        // the whole render, or one window would take every note with it.
        let n = ((rendered.len() - budget) / 80).clamp(1, same);
        let mut drop_at: HashMap<u32, HashSet<usize>> = HashMap::new();
        for (_, _, file_id, index) in ranked.into_iter().take(n) {
            drop_at.entry(file_id).or_default().insert(index);
        }
        // Kept aside so a batch that does not shrink the render is undone.
        // The prompt is the render; a finding removed only from the report
        // would be announced as omitted while still sitting in the text.
        let snapshot: Vec<(u32, Vec<cleave::Finding>)> = report
            .files
            .iter()
            .map(|file| (file.id, file.findings.clone()))
            .collect();
        for file in &mut report.files {
            if let Some(indexes) = drop_at.get(&file.id) {
                let mut index = 0usize;
                file.findings.retain(|_| {
                    let drop = indexes.contains(&index);
                    index += 1;
                    !drop
                });
            }
        }
        let next = crate::interpret::payload::render(report);
        if next.len() >= rendered.len() {
            for file in &mut report.files {
                if let Some((_, findings)) = snapshot.iter().find(|(id, _)| *id == file.id) {
                    file.findings.clone_from(findings);
                }
            }
            break;
        }
        *rendered = next;
    }
}

/// What the fetch phase grafted into the report, as the renders read it.
#[derive(Clone, Copy)]
pub(super) struct Fetched<'a> {
    pub(super) edges: &'a [FetchRecord],
    pub(super) deps: &'a [DepResult],
    pub(super) registries: &'a [DependencyRegistry],
}

/// Lookups over a typed report that the renders share, built once.
pub(super) struct ReportIndex<'a> {
    pub(super) by_id: HashMap<u32, &'a FileAnalysis>,
    /// The first file carrying each sha256.
    pub(super) by_sha: HashMap<&'a str, &'a FileAnalysis>,
    /// The first fetched-payload root carrying each sha256.
    fetched_by_sha: HashMap<&'a str, &'a FileAnalysis>,
    /// File id → the root id of the fetched payload it belongs to.
    fetched_root: HashMap<u32, u32>,
}

impl<'a> ReportIndex<'a> {
    pub(super) fn new(report: &'a AnalysisReport) -> Self {
        let by_id: HashMap<u32, &FileAnalysis> = report.files.iter().map(|f| (f.id, f)).collect();
        let mut by_sha = HashMap::new();
        let mut fetched_by_sha = HashMap::new();
        let mut fetched_root = HashMap::new();
        for file in &report.files {
            by_sha.entry(file.sha256.as_str()).or_insert(file);
            if file.rel == Rel::Fetched {
                fetched_by_sha.entry(file.sha256.as_str()).or_insert(file);
            }
            if let Some(root) = fetched_root_id(file, &by_id) {
                fetched_root.insert(file.id, root);
            }
        }
        Self {
            by_id,
            by_sha,
            fetched_by_sha,
            fetched_root,
        }
    }

    /// Whether the file belongs to a fetched payload rather than to the sample.
    pub(super) fn is_fetched(&self, id: u32) -> bool {
        self.fetched_root.contains_key(&id)
    }
}

/// A fetched payload grafted into the report, with what scan knows about it.
pub(super) struct FetchedSubject<'a> {
    pub(super) edge: &'a FetchRecord,
    pub(super) root: &'a FileAnalysis,
    /// Every file of the payload, its root included.
    pub(super) files: HashSet<u32>,
    graded: Option<&'a DepResult>,
    pub(super) registry: Option<&'a DependencyRegistry>,
}

impl FetchedSubject<'_> {
    /// The verdict scan computed for the payload, when it graded it.
    pub(super) fn verdict(&self) -> Option<Decision> {
        self.graded.and_then(|d| d.verdict)
    }

    /// Whether a file of the payload carries a finding at `crit` or graver.
    pub(super) fn has_finding(&self, index: &ReportIndex<'_>, crit: Criticality) -> bool {
        self.files
            .iter()
            .filter_map(|id| index.by_id.get(id))
            .flat_map(|file| &file.findings)
            .any(|finding| finding.crit >= crit)
    }
}

/// One subject per fetched payload that landed bytes, in fetch order.
pub(super) fn fetched_subjects<'a>(
    fetched: Fetched<'a>,
    index: &ReportIndex<'a>,
) -> Vec<FetchedSubject<'a>> {
    let mut graded: HashMap<&str, &DepResult> = HashMap::new();
    for dep in fetched.deps {
        graded.entry(dep.sha256.as_str()).or_insert(dep);
    }
    let mut registries: HashMap<&str, &DependencyRegistry> = HashMap::new();
    for registry in fetched.registries {
        registries
            .entry(registry.locator.as_str())
            .or_insert(registry);
    }
    // Grouped once: a scan can fetch hundreds of payloads into a report of
    // tens of thousands of files, and a walk of every file per payload is
    // their product.
    let mut payloads: HashMap<u32, HashSet<u32>> = HashMap::new();
    for (&file, &root) in &index.fetched_root {
        payloads.entry(root).or_default().insert(file);
    }
    // Removing a payload as its subject is built also keeps one subject per
    // root when several edges landed the same bytes.
    fetched
        .edges
        .iter()
        .filter_map(|edge| {
            let content = edge.content_sha256.as_deref()?;
            let root = *index.fetched_by_sha.get(content)?;
            Some(FetchedSubject {
                edge,
                root,
                files: payloads.remove(&root.id)?,
                graded: graded.get(content).copied(),
                registry: registries.get(edge.locator.as_str()).copied(),
            })
        })
        .collect()
}

/// Registries whose package landed no bytes (removed, age-gated, too large):
/// their provenance is the only evidence about it.
pub(super) fn registry_only<'a>(
    fetched: Fetched<'a>,
) -> impl Iterator<Item = &'a DependencyRegistry> {
    let landed: HashSet<&str> = fetched
        .edges
        .iter()
        .filter(|edge| edge.content_sha256.is_some())
        .map(|edge| edge.locator.as_str())
        .collect();
    fetched
        .registries
        .iter()
        .filter(move |registry| !landed.contains(registry.locator.as_str()))
}

/// The report narrowed to the files `keep` selects, for cleave's renderer. It
/// reads the files, the report-level gaps and the target, so only those are
/// copied — never the whole report.
fn report_view(report: &AnalysisReport, keep: impl Fn(&FileAnalysis) -> bool) -> AnalysisReport {
    let mut view = AnalysisReport::new(report.target.clone());
    view.analysis_gaps = report.analysis_gaps.clone();
    view.files = report.files.iter().filter(|f| keep(f)).cloned().collect();
    view
}

/// The artifact a scan was asked about, as the LLM render names it.
pub(super) struct Primary<'a> {
    pub(super) label: &'a str,
    pub(super) sha256: &'a str,
    pub(super) fetch: Option<&'a FetchRecord>,
    pub(super) registry: Option<&'a RegistryProvenance>,
}

/// Render the package-aware user message used by `--llm` and
/// `--format interpret`.
///
/// Cleave's merged report contains the primary artifact, fetched artifacts, and
/// registry sidecars in one flat file list. Rendering that list directly makes
/// fetched packages look like archive members and puts their traits before the
/// appendix that explains where they came from. Build one subject block at a
/// time instead: compact provenance first, then only that package's cleave
/// context. Dependencies are omitted unless suspicious/hostile or a notable+
/// match is tied to their registry provenance (directly or through composite
/// sources), and only the [`MAX_INTERPRET_DEP_SUBJECTS`] riskiest are rendered.
pub(super) fn render_interpret_context(
    primary: &Primary<'_>,
    fetched: Fetched<'_>,
    report: &AnalysisReport,
    index: &ReportIndex<'_>,
) -> String {
    let now_secs = i64::try_from(crate::civil::now_secs()).unwrap_or(i64::MAX);
    let budget = INTERPRET_PROMPT_BUDGET_BYTES;

    // Ranked before the primary is rendered: a hostile dependency is often the
    // whole verdict, and the primary must leave it room inside the one budget.
    enum Subject<'s, 'a> {
        Fetched(&'s FetchedSubject<'a>),
        RegistryOnly(&'a DependencyRegistry),
    }
    let fetched_subjects = fetched_subjects(fetched, index);
    let registry_only: Vec<&DependencyRegistry> = registry_only(fetched).collect();
    let candidates = fetched_subjects.len() + registry_only.len();
    let mut ranked: Vec<((u8, i32, u32), Subject<'_, '_>)> = Vec::new();
    for subject in &fetched_subjects {
        let severe_finding = subject.has_finding(index, Criticality::Suspicious);
        let severe_verdict = subject
            .verdict()
            .is_some_and(|v| v.class >= Classification::Suspicious);
        let provenance_hit = subject.registry.is_some_and(|registry| {
            provenance_has_notable_match(index, &subject.files, registry.file_id)
        });
        if severe_finding || severe_verdict || provenance_hit {
            let risk = dep_subject_risk(subject.verdict(), severe_finding);
            ranked.push((risk, Subject::Fetched(subject)));
        }
    }
    // A removed or age-gated dependency may have no artifact subtree at all.
    // Its atomic provenance finding is still evidence and gets its own subject.
    for registry in registry_only {
        if provenance_has_notable_match(index, &HashSet::new(), registry.file_id) {
            // No artifact was analyzed, so there is no verdict and no member trait.
            ranked.push((
                dep_subject_risk(None, false),
                Subject::RegistryOnly(registry),
            ));
        }
    }
    // Worst first; the sort is stable, so equal risks keep discovery order.
    ranked.sort_by_key(|(risk, _)| Reverse(*risk));
    ranked.truncate(MAX_INTERPRET_DEP_SUBJECTS);
    let dep_reserve = if ranked.is_empty() { 0 } else { budget / 3 };

    let mut out = String::new();
    let registry_ids: HashSet<u32> = fetched.registries.iter().map(|r| r.file_id).collect();
    let mut own = report_view(report, |f| {
        !index.is_fetched(f.id) && !registry_ids.contains(&f.id)
    });
    let shown = interpret_display_name(primary.label);
    let provenance = primary_provenance(shown, primary, now_secs);
    let identity = provenance
        .get("registry")
        .and_then(|registry| registry.get("record"))
        .filter(|record| record.as_object().is_some_and(|r| !r.is_empty()))
        .map(|identity| {
            let line = serde_json::json!({"registry": {"record": identity}});
            format!("\n== SUBJECT IDENTITY (registry claim for PRIMARY) ==\nprovenance={line}\n")
        });
    let mut head = String::new();
    let _ = writeln!(head, "== PRIMARY {shown} ==");
    // A local file's provenance is only its path, already in the header, and
    // a SHA-256 the grader can do nothing with. The line exists to carry a
    // fetch record or a registry identity; without one it is noise.
    let artifact_only = provenance
        .as_object()
        .is_some_and(|fields| fields.keys().all(|k| k == "artifact"));
    if !artifact_only {
        let _ = writeln!(head, "provenance={provenance}");
    }
    let context_budget = budget.saturating_sub(
        dep_reserve + head.len() + identity.as_ref().map_or(0, String::len) + BUDGET_NOTE,
    );
    let primary_cut = budget_primary_context(&mut own, context_budget);
    if primary_cut.members_omitted > 0 || primary_cut.findings_omitted > 0 {
        tracing::info!(
            label = primary.label,
            members_omitted = primary_cut.members_omitted,
            findings_omitted = primary_cut.findings_omitted,
            budget_bytes = context_budget,
            rendered_bytes = primary_cut.text.len(),
            "interpret render over budget; weakest content dropped"
        );
    }
    out.push_str(&head);
    out.push_str(&primary_cut.text);
    write_budget_omission(&mut out, &primary_cut);
    // Restated after the findings. One `provenance=` line at the head of a render
    // that runs to tens of KB is not read: the grader infers identity from
    // filenames and source instead, and the registry's claim never enters the
    // judgment at all. The same bytes at the tail are read: on a mislabelled
    // sample, head-only provenance grades benign ("standard LDAP gem", describing
    // the code and ignoring the claim) while the tail restatement catches it
    // ("registry mismatch").
    // Only the registry record is restated, never the whole block. The claim is
    // what the grader needs a second look at.
    if let Some(identity) = &identity {
        out.push_str(identity);
    }

    let mut shown_deps = 0usize;
    for (_, subject) in &ranked {
        let room = budget.saturating_sub(out.len() + BUDGET_NOTE);
        let block = match subject {
            Subject::Fetched(subject) => {
                render_fetched_subject(subject, report, index, now_secs, room)
            }
            Subject::RegistryOnly(registry) => {
                render_registry_subject(registry, report, now_secs, room)
            }
        };
        if block.is_empty() || block.len() > room {
            continue;
        }
        out.push_str(&block);
        shown_deps += 1;
    }

    let omitted = candidates.saturating_sub(shown_deps);
    if omitted > 0 {
        let _ = writeln!(out, "\ndeps_omitted={omitted}");
    }
    out
}

fn write_budget_omission(out: &mut String, cut: &BudgetedRender) {
    if cut.members_omitted > 0 || cut.findings_omitted > 0 {
        let _ = writeln!(
            out,
            "members_omitted={} findings_omitted={} (prompt budget)",
            cut.members_omitted, cut.findings_omitted
        );
    }
}

/// One fetched dependency's block: header, provenance, then its own context,
/// trimmed to `room` bytes. Empty when nothing relevant fits.
fn render_fetched_subject(
    subject: &FetchedSubject<'_>,
    report: &AnalysisReport,
    index: &ReportIndex<'_>,
    now_secs: i64,
    room: usize,
) -> String {
    let mut block = String::new();
    let class = subject
        .verdict()
        .map_or_else(|| "not-evaluated".to_string(), |v| v.class.to_string());
    let kind = if subject.edge.kind == fletch::RefKind::Dependency {
        "DEP"
    } else {
        "FETCH"
    };
    let _ = writeln!(
        block,
        "\n== {kind} {} class={class} ==",
        subject.edge.locator
    );
    let source_path = subject
        .edge
        .source_sha256
        .as_deref()
        .and_then(|sha| index.by_sha.get(sha))
        .map(|file| file.path.as_str());
    let provenance = dependency_provenance(subject.edge, subject.registry, source_path, now_secs);
    let _ = writeln!(block, "provenance={provenance}");
    if block.len().saturating_add(BUDGET_NOTE) >= room {
        return String::new();
    }
    let mut view = report_view(report, |f| {
        subject.files.contains(&f.id) || subject.registry.is_some_and(|r| r.file_id == f.id)
    });
    let cut = budget_primary_context(&mut view, room - block.len() - BUDGET_NOTE);
    block.push_str(&cut.text);
    write_budget_omission(&mut block, &cut);
    block
}

/// A dependency known only from its registry record, trimmed to `room` bytes.
/// Empty when the record does not fit.
fn render_registry_subject(
    registry: &DependencyRegistry,
    report: &AnalysisReport,
    now_secs: i64,
    room: usize,
) -> String {
    let mut block = String::new();
    let status = registry.artifact_skip.unwrap_or("registry-only");
    let _ = writeln!(block, "\n== DEP {} artifact={status} ==", registry.locator);
    let _ = writeln!(
        block,
        "provenance={}",
        registry_provenance(&registry.provenance, now_secs)
    );
    if block.len().saturating_add(BUDGET_NOTE) >= room {
        return String::new();
    }
    let mut view = report_view(report, |f| f.id == registry.file_id);
    let cut = budget_primary_context(&mut view, room - block.len() - BUDGET_NOTE);
    block.push_str(&cut.text);
    write_budget_omission(&mut block, &cut);
    block
}

/// Rewrite each finding annotation from a graded conclusion into a categorized
/// observation, for the LLM view only.
///
/// cleave announces a finding as `# SEV LOC desc (trait::id)`. Both the severity
/// letter and the prose are the analyzer's *answer*, and handing the answer to a
/// second opinion asked to check it produces agreement, not review: the model
/// summarizes the highest-severity assertion instead of reading the bytes under
/// it. A .NET single-file bundle whose overlay carries the CLR grades hostile
/// under every prompt, provenance and carve-out variant, and benign as soon as
/// the annotation stops asserting "process-hollowing API chain" outright.
///
/// The rewrite drops the grade and names the finding's category. Beside source
/// that is the whole annotation, because the row under it says the rest:
///
/// ```text
/// // N 261:131 axios.get() HTTP GET request (micro-behaviors/communications/http/…)
/// // 261 communications/http
/// ```
///
/// Over raw bytes the description stays, stripped of its authority: on a packed
/// binary the prose is the only readable signal, and a real dropper graded
/// suspicious without it (`docs/interpret-tuning.md`).
///
/// ```text
/// // H Dynamically resolved process-hollowing API chain (objectives/evasion/process/injection/hollowing::…)
/// // evasion/process — Dynamically resolved process-hollowing API chain
/// ```
///
/// The terminal view is untouched — this is the machine/LLM render alone.
fn recategorize_annotations(rendered: &str) -> String {
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

/// Assemble the exact user message after sanitization. The final cap covers
/// a protected finding, header, or source line too large to shed. The budget
/// before it sized the render with its annotations unrewritten, which is the
/// larger of the two, so it errs toward shedding.
pub(crate) fn interpret_user_message(context: &str) -> (String, bool) {
    const TRUNCATED: &str = "\n[Context truncated to fit the prompt budget.]\n";
    let mut user = recategorize_annotations(context);
    let truncated = user.len() > INTERPRET_PROMPT_BUDGET_BYTES;
    if truncated {
        let end = user.floor_char_boundary(INTERPRET_PROMPT_BUDGET_BYTES - TRUNCATED.len());
        user.truncate(end);
        user.push_str(TRUNCATED);
    }
    (user, truncated)
}

/// A cleave finding annotation, split into its parts:
/// `{indent}{marker} {SEV} {rest}`, where `marker` is `//`, `--` or `#` and
/// `SEV` is one letter of `HSNBCF`, followed by a space or the end of the line.
///
/// The one definition of the grammar: everything that asks whether a line is
/// an annotation asks here. Strict on purpose, so it never takes a real source
/// comment like `// Something` or `# S3 bucket` for one.
pub(crate) fn split_annotation(line: &str) -> Option<(&str, &'static str, char, &str)> {
    let indent_len = line.len() - line.trim_start().len();
    let (indent, rest) = line.split_at(indent_len);
    let (marker, rest) = ["//", "--", "#"]
        .into_iter()
        .find_map(|m| Some((m, rest.strip_prefix(m)?.strip_prefix(' ')?)))?;
    // By chars, not byte offsets: a body can open with a multi-byte character.
    let mut head = rest.chars();
    let sev = head.next()?;
    if !matches!(sev, 'H' | 'S' | 'N' | 'B' | 'C' | 'F') {
        return None;
    }
    let rest = match head.next() {
        None => "",
        // `sev` is ASCII and the char after it a space, so this is a boundary.
        Some(' ') => rest.get(sev.len_utf8() + 1..)?,
        Some(_) => return None,
    };
    Some((indent, marker, sev, rest))
}

/// One annotation line, or `None` when the line is not one.
fn recategorize_annotation(line: &str) -> Option<String> {
    let (indent, comment, _grade, rest) = split_annotation(line)?;
    // A third-party signature carries no prose: the id *is* the body, unwrapped —
    // `// H third_party/elastic/Linux_Trojan_Ladvix/linux/trojan/ladvix`. Naming
    // its category is the whole rewrite; there is no description to keep.
    if !rest.contains('(') && !rest.contains(char::is_whitespace) && rest.contains('/') {
        return Some(format!("{indent}{comment} {}", trait_category(rest)));
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
    let category = trait_category(inner);
    // The location stays; it is a pointer, not a verdict.
    let (head, desc) = body.split_once(' ').unwrap_or(("", body));
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    Some(match head.split_once(':') {
        // A `line:col` places the finding beside source, which speaks for
        // itself: a description would only repeat the row below, and with
        // the row in view the line number alone locates it.
        Some((line, col)) if digits(line) && digits(col) => {
            format!("{indent}{comment} {line} {category}")
        }
        _ if head.starts_with('@') => format!("{indent}{comment} {head} {category} — {desc}"),
        _ => format!("{indent}{comment} {category} — {body}"),
    })
}

/// The family a trait belongs to: the two path components below its namespace,
/// e.g. `objectives/evasion/process/injection/hollowing::x` → `evasion/process`.
/// Broad on purpose — it should place the match, not characterize it.
fn trait_category(trait_id: &str) -> String {
    let path = crate::features::trait_path(trait_id);
    let mut parts = path.split('/').filter(|p| !p.is_empty());
    let Some(namespace) = parts.next() else {
        return "pattern".to_string();
    };
    // `well-known/` is the exception: its depth is not a taxonomy of technique,
    // it is an *identity* — `unwanted/newtab-wallpaper-adware/owhit` names the
    // family, and the family is the whole finding. Truncating it to two
    // components throws away the only part that decides the verdict, which is how
    // a Chrome extension cleave had already recognized as newtab wallpaper adware
    // reached the grader as an unremarkable observation and was cleared.
    if namespace == "well-known" {
        let identity = parts.collect::<Vec<_>>().join("/");
        return if identity.is_empty() {
            namespace.to_string()
        } else {
            identity
        };
    }
    match (parts.next(), parts.next()) {
        (Some(a), Some(b)) => format!("{a}/{b}"),
        (Some(a), None) => a.to_string(),
        (None, _) => namespace.to_string(),
    }
}

/// The id of the fetched-payload root at or above `file`, or `None` for the
/// sample's own files. Bounded by the file count, so a parent cycle ends.
fn fetched_root_id(file: &FileAnalysis, by_id: &HashMap<u32, &FileAnalysis>) -> Option<u32> {
    let mut current = file;
    for _ in 0..=by_id.len() {
        if current.rel == Rel::Fetched {
            return Some(current.id);
        }
        current = *by_id.get(&current.parent_id?)?;
    }
    None
}

/// True for a notable+ atomic match on the registry node, or a notable+
/// composite on the dependency artifact whose resolved sources include it.
fn provenance_has_notable_match(
    index: &ReportIndex<'_>,
    artifact_ids: &HashSet<u32>,
    registry_id: u32,
) -> bool {
    let registry_hit = index.by_id.get(&registry_id).is_some_and(|registry| {
        registry
            .findings
            .iter()
            .any(|f| f.crit >= Criticality::Notable)
    });
    registry_hit
        || artifact_ids
            .iter()
            .filter_map(|id| index.by_id.get(id))
            .any(|file| {
                file.findings.iter().any(|finding| {
                    finding.crit >= Criticality::Notable
                        && file
                            .composite_sources
                            .get(finding.id.as_str())
                            .is_some_and(|sources| sources.iter().any(|s| s.file == registry_id))
                })
            })
}

/// The name the grader is shown for a scanned file: its last path component.
///
/// The directories above a file are the operator's filing, not evidence about
/// it. A corpus sorted into `hostile/` or `benign/`, or a download placed under
/// a folder an attacker named, would otherwise hand the model its verdict. A
/// URL is kept whole: where a fetched artifact came from is part of what it is.
fn interpret_display_name(label: &str) -> &str {
    if label.contains("://") {
        return label;
    }
    label
        .rsplit(['/', '\\'])
        .find(|part| !part.is_empty())
        .unwrap_or(label)
}

fn primary_provenance(shown: &str, primary: &Primary<'_>, now_secs: i64) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    match primary.fetch {
        Some(fetch) => out.insert("fetch".to_string(), compact_fetch_record(fetch, None)),
        None => out.insert(
            "artifact".to_string(),
            serde_json::json!({"path": shown, "sha256": primary.sha256}),
        ),
    };
    if let Some(registry) = primary.registry {
        out.insert(
            "registry".to_string(),
            registry_provenance(registry, now_secs),
        );
    }
    serde_json::Value::Object(out)
}

fn dependency_provenance(
    fetch: &FetchRecord,
    registry: Option<&DependencyRegistry>,
    source_path: Option<&str>,
    now_secs: i64,
) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    out.insert(
        "fetch".to_string(),
        compact_fetch_record(fetch, source_path),
    );
    if let Some(registry) = registry {
        out.insert(
            "registry".to_string(),
            registry_provenance(&registry.provenance, now_secs),
        );
    }
    serde_json::Value::Object(out)
}

/// Fetch provenance minus response headers and declaring-file hashes. Those are
/// high-volume and either irrelevant to interpretation or already rendered as
/// source paths; URLs, redirects, timing, cache/pin state, and content identity
/// remain.
fn compact_fetch_record(fetch: &FetchRecord, source_path: Option<&str>) -> serde_json::Value {
    let Ok(mut value) = serde_json::to_value(fetch) else {
        return serde_json::Value::Null;
    };
    if let Some(obj) = value.as_object_mut() {
        obj.remove("headers");
        obj.remove("source_sha256");
        if let Some(source_path) = source_path {
            obj.insert(
                "source_path".to_string(),
                serde_json::Value::String(source_path.to_string()),
            );
        }
    }
    value
}

/// The registry's account of a package, as the grader reads it: the record
/// projected to identity and credibility signals ([`project_registry_record`]).
///
/// The provider documents behind the record (`raw`) are not rendered. They are
/// about a third of every prompt's tokens, and every identity signal they
/// carry is summarised by the record fletch derives from them.
fn registry_provenance(provenance: &RegistryProvenance, now_secs: i64) -> serde_json::Value {
    serde_json::json!({ "record": project_registry_record(&provenance.record, now_secs) })
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
/// at a few dozen tokens.
///
/// Ages are measured at scan time from `first_published_at`/`published_at`,
/// not read from the record's `age_days`: an offline `--registry-map` sidecar
/// freezes that at *collection* time, and a corpus collected within days of
/// publication would tell the grader every long-established package is new.
///
/// Keys are the record's wire names, which every consumer of the envelope
/// already sees. Ecosystems populate different subsets of the record (a gem
/// carries downloads but no release count; a pypi record carries both); absent
/// and empty values are omitted, and a zero download count is kept, because
/// zero is the signal.
fn project_registry_record(record: &fletch::Registry, now_secs: i64) -> serde_json::Value {
    fn text(value: Option<&String>) -> Option<&str> {
        value.map(String::as_str).filter(|s| !s.is_empty())
    }
    fn truncate(s: &str, max: usize) -> String {
        if s.chars().count() <= max {
            s.to_string()
        } else {
            let head: String = s.chars().take(max).collect();
            format!("{head}…")
        }
    }

    let mut out = serde_json::Map::new();
    let mut put = |key: &str, value: serde_json::Value| {
        out.insert(key.to_string(), value);
    };
    let version = record.version.as_str();
    if !record.name.is_empty() {
        let eco = record.ecosystem.as_str();
        let mut id = String::new();
        if !eco.is_empty() {
            id.push_str(eco);
            id.push('/');
        }
        id.push_str(&record.name);
        if !version.is_empty() {
            id.push('@');
            id.push_str(version);
        }
        put("package", serde_json::json!(id));
    }
    for (key, value, max) in [
        ("title", &record.title, 120),
        ("description", &record.description, 300),
        // Who. `publisher` is the account that pushed this version; `author`
        // is whatever the manifest claims. Both matter when they disagree.
        ("publisher", &record.publisher, 60),
        ("author", &record.author, 60),
        ("publisher_email_domain", &record.publisher_email_domain, 80),
    ] {
        if let Some(text) = text(value.as_ref()) {
            put(key, serde_json::json!(truncate(text, max)));
        }
    }
    if let Some(n) = record.maintainers {
        put("maintainers", serde_json::json!(n));
    }
    for (key, flag) in [
        ("publisher_in_maintainers", record.publisher_in_maintainers),
        ("publisher_verified", record.publisher_verified),
    ] {
        if flag == Some(true) {
            put(key, serde_json::json!(true));
        }
    }
    // Where it claims to come from.
    for (key, value) in [
        ("repository", &record.repository),
        ("homepage", &record.homepage),
        ("license", &record.license),
    ] {
        if let Some(text) = text(value.as_ref()) {
            put(key, serde_json::json!(truncate(text, 200)));
        }
    }
    // Whether anyone uses it.
    for (key, value) in [
        ("downloads_total", record.downloads_total),
        ("downloads_recent", record.downloads_recent),
    ] {
        if let Some(n) = value {
            put(key, serde_json::json!(n));
        }
    }
    // Two distinct ages, both measured at scan time. `package_age_days` is how
    // long the *package* has existed and is the credibility signal;
    // `version_age_days` is how long this *version* has, a freshness signal.
    // Folding them into one would report a decade-old gem as days old whenever
    // it had just cut a release — inverting exactly the signal that matters.
    let days_since = |ts: Option<u64>| {
        let ts = i64::try_from(ts?).ok()?;
        (now_secs > ts).then(|| (now_secs - ts) / crate::civil::SECS_PER_DAY)
    };
    for (key, at) in [
        ("package_age_days", record.first_published_at),
        ("version_age_days", record.published_at),
        ("previous_release_age_days", record.previous_published_at),
    ] {
        if let Some(days) = days_since(at) {
            put(key, serde_json::json!(days));
        }
    }
    for (key, value) in [
        ("release_count", record.release_count),
        ("releases_24h", record.releases_24h),
        ("releases_48h", record.releases_48h),
    ] {
        if let Some(n) = value {
            put(key, serde_json::json!(n));
        }
    }
    if let Some(latest) = text(record.latest_version.as_ref()) {
        put("latest_version", serde_json::json!(latest));
        if !version.is_empty() && latest != version {
            put("is_latest", serde_json::json!(false));
        }
    }
    // What the registry itself has flagged. Only when true: a sea of `false`
    // is noise, and an absent flag reads the same as a false one.
    for (key, flag) in [
        ("has_install_script", record.has_install_script),
        ("security_hold", record.security_hold),
        ("version_removed", record.version_removed),
    ] {
        if flag == Some(true) {
            put(key, serde_json::json!(true));
        }
    }
    if let Some(text) = text(record.deprecated.as_ref()) {
        put("deprecated", serde_json::json!(truncate(text, 120)));
    }
    if let Some(n) = record.unpacked_size.filter(|n| *n > 0) {
        put("unpacked_size", serde_json::json!(n));
    }
    if let Some(n) = record.file_count.filter(|n| *n > 0) {
        put("file_count", serde_json::json!(n));
    }
    if let Some(vulns) = record.vulnerability_count.filter(|v| *v > 0) {
        put("vulnerability_count", serde_json::json!(vulns));
    }
    serde_json::Value::Object(out)
}

/// Render the fetched-dependencies appendix for the text renders.
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
pub(super) fn render_dependency_context(
    fetched: Fetched<'_>,
    report: &AnalysisReport,
    index: &ReportIndex<'_>,
) -> Option<String> {
    // Only edges that landed bytes have a grafted node to describe.
    let landed: Vec<(&FetchRecord, &str)> = fetched
        .edges
        .iter()
        .filter_map(|edge| Some((edge, edge.content_sha256.as_deref()?)))
        .collect();
    if landed.is_empty() {
        return None;
    }
    let mut graded: HashMap<&str, &DepResult> = HashMap::new();
    for dep in fetched.deps {
        graded.entry(dep.sha256.as_str()).or_insert(dep);
    }
    let mut out = String::new();
    out.push_str(
        "\n== FETCHED DEPENDENCIES ==\n\
         The scan followed references declared by this sample and retrieved the content below.\n\
         Each payload was analyzed and appears above under its locator — these files are\n\
         EXTERNAL retrieved content, not members of the sample's own archive. A hostile or\n\
         suspicious dependency elevates the sample's verdict (fetch/dependency-verdict).\n",
    );
    for (rec, content_sha) in landed {
        let root = index.by_sha.get(content_sha).copied();
        // Every name below is the sample's or a registry's text, and this
        // appendix is printed by `--format tiny`: escaped like the cards are.
        let tty = crate::output::tty_text;
        let _ = writeln!(out, "\ndependency: {}", tty(&rec.locator));
        if let Some(resolved) = rec
            .resolved_url
            .as_deref()
            .filter(|url| *url != rec.locator)
        {
            let _ = writeln!(out, "  resolved url: {}", tty(resolved));
        }
        let source = rec
            .source_sha256
            .as_deref()
            .and_then(|sha| index.by_sha.get(sha))
            .map_or("<unknown file>", |f| f.path.as_str());
        let _ = write!(
            out,
            "  referenced: {} in {}",
            ref_kind_phrase(rec.kind),
            tty(source)
        );
        if let Some(off) = rec.source_offset {
            let _ = write!(out, " @ byte {off}");
        }
        out.push('\n');
        let _ = writeln!(out, "  content sha256: {content_sha}");
        if let Some(root) = root {
            let _ = writeln!(out, "  analyzed above as: {}", tty(&root.path));
        }
        // The verdict scan computed for this dependency, from the same graded
        // results it uploads, so the render and the record cannot disagree.
        match graded.get(content_sha).and_then(|d| d.verdict) {
            Some(v) => {
                let _ = writeln!(
                    out,
                    "  classification: {} (p={:.2})",
                    v.class, v.probability
                );
            }
            // Say so rather than printing nothing: an ungraded dependency and an
            // unremarkable one are different, and only one of them is a coverage
            // gap worth chasing.
            None => out.push_str("  classification: not evaluated\n"),
        }
        let Some(root) = root else { continue };
        // The dependency's own elevated findings: the root node plus every
        // member below it, deduped by id, worst first.
        let mut elevated: Vec<(&FileAnalysis, &cleave::Finding)> = report
            .files
            .iter()
            .filter(|f| f.path == root.path || is_inside(&f.path, &root.path))
            .flat_map(|f| f.findings.iter().map(move |fd| (f, fd)))
            .filter(|(_, fd)| fd.crit >= Criticality::Suspicious)
            .collect();
        elevated.sort_by_key(|(_, fd)| Reverse(fd.crit));
        let mut seen = HashSet::new();
        elevated.retain(|(_, fd)| seen.insert(fd.id.as_str()));
        if elevated.is_empty() {
            continue;
        }
        out.push_str("  elevated findings on this dependency's files:\n");
        let total = elevated.len();
        for (f, fd) in elevated.into_iter().take(MAX_DEP_FINDING_LINES) {
            let sev = if fd.crit >= Criticality::Hostile {
                'H'
            } else {
                'S'
            };
            let _ = write!(out, "  # {sev} {}", fd.id);
            if !fd.desc.is_empty() {
                let _ = write!(out, " — {}", tty(&fd.desc));
            }
            let _ = writeln!(out, " [{}]", tty(&f.path));
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
fn ref_kind_phrase(kind: fletch::RefKind) -> &'static str {
    match kind {
        fletch::RefKind::Dependency => "declared as a dependency",
        fletch::RefKind::Command => "named by an install command",
        fletch::RefKind::UrlFetch => "fetched from a URL",
        fletch::RefKind::Repository => "named as the source repository",
        fletch::RefKind::Local => "a local reference",
        _ => "referenced",
    }
}

/// Replay an artifact-only interpretation input with the live prompt budget and
/// annotation rewrite. Reports with fetched/registry subjects need the complete
/// saved presentation envelope; failing is preferable to silently dropping them.
pub(super) fn render_saved_interpret(
    loaded: &cleave::report_input::LoadedReport,
) -> anyhow::Result<String> {
    anyhow::ensure!(
        loaded.fetched.is_empty()
            && loaded
                .report
                .files
                .iter()
                .all(|f| f.rel != Rel::Fetched && f.role != cleave::types::Role::Sidecar),
        "package-aware interpretation replay requires saved fetch verdicts and registry provenance; this input is incomplete"
    );
    let report = &loaded.report;
    let root = report
        .files
        .iter()
        .find(|f| f.parent_id.is_none())
        .ok_or_else(|| anyhow::anyhow!("report has no root artifact"))?;
    let primary = Primary {
        label: &root.path,
        sha256: &root.sha256,
        fetch: None,
        registry: None,
    };
    let fetched = Fetched {
        edges: &[],
        deps: &[],
        registries: &[],
    };
    let index = ReportIndex::new(report);
    let rendered = render_interpret_context(&primary, fetched, report, &index);
    // The live path sanitizes before it rewrites; so must a replay, or it is
    // not the payload the model saw.
    Ok(interpret_user_message(&crate::interpret::sanitize_context(&rendered)).0)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod saved_render_tests;
