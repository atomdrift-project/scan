//! The package-aware render the LLM reads (`--interpret`,
//! `--format interpret`), and the fetched-subject selection the terminal
//! render shares with it.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use cleave::output::{TinyOpts, format_context};
use cleave::types::Rel;
use cleave::{AnalysisReport, Criticality, FileAnalysis};
use fletch::fetch::FetchRecord;

use super::{DepResult, Tuning, is_inside};
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
/// false-positive budget (FP per 100M benigns) at which its hostile decision
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
        // a level of 3000 (fires only when 3000 FP per 100M is acceptable).
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
/// `SCAN_INTERPRET_BUDGET_BYTES` overrides it (see [`Tuning`]); `0` disables
/// the cap.
pub(crate) const INTERPRET_PRIMARY_BUDGET_BYTES: usize = 96 * 1024;

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
fn budget_primary_context(primary: &mut AnalysisReport, budget: usize) -> (String, usize) {
    let mut rendered = format_context(primary, &TinyOpts::tiny());
    if budget == 0 || rendered.len() <= budget {
        return (rendered, 0);
    }
    let mut order: Vec<((Criticality, usize), u32)> = primary
        .files
        .iter()
        .filter(|f| f.depth > 0 && f.id != 0)
        .filter(|f| f.findings.iter().all(|t| t.crit < Criticality::Suspicious))
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
        let mut victims: HashSet<u32> = order.drain(..n).map(|(_, id)| id).collect();
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

    /// Every file of the fetched payload rooted at `root`.
    fn payload(&self, root: u32) -> HashSet<u32> {
        self.fetched_root
            .iter()
            .filter_map(|(&file, &payload)| (payload == root).then_some(file))
            .collect()
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
    let mut seen = HashSet::new();
    fetched
        .edges
        .iter()
        .filter_map(|edge| {
            let content = edge.content_sha256.as_deref()?;
            let root = *index.fetched_by_sha.get(content)?;
            seen.insert(root.id).then(|| FetchedSubject {
                edge,
                root,
                files: index.payload(root.id),
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
/// sources), and only the [`MAX_INTERPRET_DEP_SUBJECTS`] riskiest are rendered.
pub(super) fn render_interpret_context(
    primary: &Primary<'_>,
    fetched: Fetched<'_>,
    report: &AnalysisReport,
    index: &ReportIndex<'_>,
    tuning: &Tuning,
) -> String {
    let now_secs = scan_now_secs();
    let budget = tuning.interpret_budget_bytes;
    let mut out = String::new();

    let registry_ids: HashSet<u32> = fetched.registries.iter().map(|r| r.file_id).collect();
    let mut own = report_view(report, |f| {
        !index.is_fetched(f.id) && !registry_ids.contains(&f.id)
    });
    let (primary_context, members_omitted) = budget_primary_context(&mut own, budget);
    if members_omitted > 0 {
        tracing::info!(
            label = primary.label,
            members_omitted,
            budget_bytes = budget,
            "interpret render over budget; weakest members dropped"
        );
    }
    let shown = interpret_display_name(primary.label);
    let _ = writeln!(out, "== PRIMARY {shown} ==");
    let provenance = primary_provenance(shown, primary, now_secs);
    let _ = writeln!(out, "provenance={provenance}");
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
    // Only the registry record is restated, never the whole block. The claim is
    // what the grader needs a second look at.
    let identity = provenance
        .get("registry")
        .and_then(|registry| registry.get("record"))
        .filter(|record| record.as_object().is_some_and(|r| !r.is_empty()));
    if let Some(identity) = identity {
        let line = serde_json::json!({"registry": {"record": identity}});
        let _ = writeln!(
            out,
            "\n== SUBJECT IDENTITY (registry claim for PRIMARY) ==\nprovenance={line}"
        );
    }

    // Ranked before anything is rendered: every subject that clears the gate
    // costs a provenance line and a context block, and only the riskiest few
    // are kept.
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
            provenance_has_notable_match(report, &subject.files, registry.file_id)
        });
        if severe_finding || severe_verdict || provenance_hit {
            let risk = dep_subject_risk(subject.verdict(), severe_finding);
            ranked.push((risk, Subject::Fetched(subject)));
        }
    }
    // A removed or age-gated dependency may have no artifact subtree at all.
    // Its atomic provenance finding is still evidence and gets its own subject.
    for registry in registry_only {
        if provenance_has_notable_match(report, &HashSet::new(), registry.file_id) {
            // No artifact was analyzed, so there is no verdict and no member trait.
            ranked.push((
                dep_subject_risk(None, false),
                Subject::RegistryOnly(registry),
            ));
        }
    }
    // Worst first; the sort is stable, so equal risks keep discovery order. The
    // render is one prompt, and an over-budget prompt is refused whole rather
    // than truncated by the endpoint.
    ranked.sort_by_key(|(risk, _)| Reverse(*risk));
    ranked.truncate(MAX_INTERPRET_DEP_SUBJECTS);
    for (_, subject) in &ranked {
        match subject {
            Subject::Fetched(subject) => {
                render_fetched_subject(&mut out, subject, report, index, now_secs);
            }
            Subject::RegistryOnly(registry) => {
                render_registry_subject(&mut out, registry, report, now_secs);
            }
        }
    }

    let omitted = candidates.saturating_sub(ranked.len());
    if omitted > 0 {
        let _ = writeln!(out, "\ndeps_omitted={omitted}");
    }
    out
}

/// One fetched dependency's block: header, provenance, then its own context.
fn render_fetched_subject(
    out: &mut String,
    subject: &FetchedSubject<'_>,
    report: &AnalysisReport,
    index: &ReportIndex<'_>,
    now_secs: i64,
) {
    let class = subject
        .verdict()
        .map_or_else(|| "not-evaluated".to_string(), |v| v.class.to_string());
    let kind = if subject.edge.kind == fletch::RefKind::Dependency {
        "DEP"
    } else {
        "FETCH"
    };
    let _ = writeln!(out, "\n== {kind} {} class={class} ==", subject.edge.locator);
    let source_path = subject
        .edge
        .source_sha256
        .as_deref()
        .and_then(|sha| index.by_sha.get(sha))
        .map(|file| file.path.as_str());
    let provenance = dependency_provenance(subject.edge, subject.registry, source_path, now_secs);
    let _ = writeln!(out, "provenance={provenance}");
    let view = report_view(report, |f| {
        subject.files.contains(&f.id) || subject.registry.is_some_and(|r| r.file_id == f.id)
    });
    out.push_str(&format_context(&view, &TinyOpts::tiny()));
}

/// A dependency known only from its registry record.
fn render_registry_subject(
    out: &mut String,
    registry: &DependencyRegistry,
    report: &AnalysisReport,
    now_secs: i64,
) {
    let status = registry.artifact_skip.unwrap_or("registry-only");
    let _ = writeln!(out, "\n== DEP {} artifact={status} ==", registry.locator);
    let _ = writeln!(
        out,
        "provenance={}",
        registry_provenance(&registry.provenance, now_secs)
    );
    let view = report_view(report, |f| f.id == registry.file_id);
    out.push_str(&format_context(&view, &TinyOpts::tiny()));
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
    if let Some(body) = recategorize_suppression(rest) {
        return Some(format!("{indent}{comment} {body}"));
    }
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

/// The body of a suppression line, rewritten to name only what did the
/// suppressing; `None` when `rest` (the text after `SEV `) is not one.
///
/// cleave lists the conclusions it matched but withheld or demoted as
/// `withheld <id> by <leg> @spans; <leg>…[, +N more]`. There is no trailing
/// parenthesized id, so the category rewrite never saw these lines, and both
/// the grade and the suppressed conclusion's id reached the grader verbatim —
/// measured on a benign package graded suspicious for "postinstall hook without
/// repository", echoing the withheld `…::no-repo-with-hooks`. The suppressed id
/// is dropped outright; the legs survive, because that a pattern was ruled out,
/// and by what, is benign counterweight the grader should weigh. Byte spans are
/// internal and go too.
fn recategorize_suppression(rest: &str) -> Option<String> {
    let (verb, rest) = rest.split_once(' ')?;
    let outcome = match verb {
        "withheld" | "suppressed" => "ruled out",
        "downgraded" => "weakened",
        _ => return None,
    };
    let (id, legs) = match rest.split_once(' ') {
        Some((id, tail)) => (id, tail.strip_prefix("by ")?),
        None => (rest, ""),
    };
    if !id.contains('/') {
        return None;
    }
    let mut named: Vec<String> = Vec::new();
    for leg in legs.split(';') {
        // A leg is `<id>[ @spans]`; the last may trail `, +N more`.
        let Some(leg_id) = leg.split_whitespace().next() else {
            continue;
        };
        let leg_id = leg_id.trim_end_matches(',');
        if leg_id.is_empty() {
            continue;
        }
        // The leaf is kept only where the namespace is descriptive rather than a
        // conclusion: `metadata/` facts, `micro-behaviors/` observations and
        // recognized tools, apps and libraries are the benign context itself,
        // whereas an `objectives/` leaf would name a verdict all over again. A
        // bare `communications/http` says too little to weigh as context.
        let descriptive = leg_id.starts_with("metadata/")
            || leg_id.starts_with("micro-behaviors/")
            || leg_id
                .strip_prefix("well-known/")
                .and_then(|r| r.split_once('/'))
                .is_some_and(|(kind, _)| matches!(kind, "tool" | "app" | "lib"));
        let category = trait_category(leg_id);
        let label = match leg_id.split_once("::") {
            Some((_, leaf)) if descriptive && !leaf.is_empty() => format!("{category} ({leaf})"),
            _ => category,
        };
        if !named.contains(&label) {
            named.push(label);
        }
    }
    Some(if named.is_empty() {
        format!("Possible benign context — a pattern here was {outcome}")
    } else {
        format!(
            "Possible benign context — {outcome} by: {}",
            named.join(", ")
        )
    })
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
    report: &AnalysisReport,
    artifact_ids: &HashSet<u32>,
    registry_id: u32,
) -> bool {
    report.files.iter().any(|file| {
        if file.id == registry_id && file.findings.iter().any(|f| f.crit >= Criticality::Notable) {
            return true;
        }
        artifact_ids.contains(&file.id)
            && file.findings.iter().any(|finding| {
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
/// The provider documents behind the record (`raw`) are not rendered. They were
/// until 2026-09-05, so provider-only fields could reach the grader; measured
/// over 113 rendered PURLs they were 30% of every prompt token, every identity
/// signal they carried is summarised by the record fletch derives from them,
/// and an A/B on the shipped prompt without them graded the same samples
/// identically (`hacks/interpret-tune/tune.py --templates noraw`).
fn registry_provenance(provenance: &RegistryProvenance, now_secs: i64) -> serde_json::Value {
    let record = serde_json::to_value(&provenance.record).unwrap_or(serde_json::Value::Null);
    serde_json::json!({ "record": project_registry_record(&record, now_secs) })
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
/// Ages are measured at scan time from `first_published_at`/`published_at`,
/// not read from the record's `age_days`: an offline `--registry-map` sidecar
/// freezes that at *collection* time, and a corpus collected within days of
/// publication would tell the grader every long-established package is new.
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
        let _ = writeln!(out, "\ndependency: {}", rec.locator);
        if let Some(resolved) = rec
            .resolved_url
            .as_deref()
            .filter(|url| *url != rec.locator)
        {
            let _ = writeln!(out, "  resolved url: {resolved}");
        }
        let source = rec
            .source_sha256
            .as_deref()
            .and_then(|sha| index.by_sha.get(sha))
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
mod tests {
    use super::*;
    use crate::engine::MemberEvals;
    use crate::engine::pipeline::root_needs_registry_graft;

    /// The LLM render with default tuning, for a local (unfetched) root.
    fn interpret(
        label: &str,
        sha256: &str,
        registry: Option<&RegistryProvenance>,
        edges: &[FetchRecord],
        deps: &[DepResult],
        registries: &[DependencyRegistry],
        report: &AnalysisReport,
    ) -> String {
        let primary = Primary {
            label,
            sha256,
            fetch: None,
            registry,
        };
        let fetched = Fetched {
            edges,
            deps,
            registries,
        };
        render_interpret_context(
            &primary,
            fetched,
            report,
            &ReportIndex::new(report),
            &Tuning::default(),
        )
    }

    /// The dependency appendix, with nothing fetched from a registry.
    fn appendix(
        edges: &[FetchRecord],
        deps: &[DepResult],
        report: &AnalysisReport,
    ) -> Option<String> {
        let fetched = Fetched {
            edges,
            deps,
            registries: &[],
        };
        render_dependency_context(fetched, report, &ReportIndex::new(report))
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
    fn recategorizing_a_suppression_hides_the_grade_and_the_conclusion() {
        // Withheld, `//` form, repeated suppressor with `@N` spans: the grade and
        // the withheld conclusion go, the suppressor stays once, spans are dropped.
        assert_eq!(
            recategorize_annotation(
                "// S withheld objectives/supply-chain/install-hook/package/manifest::no-repo-with-hooks by metadata/package/freeform::pkg-skeleton-small @30; metadata/package/freeform::conventional-version @30; metadata/package/freeform::pkg-skeleton-small @30,41"
            )
            .as_deref(),
            Some(
                "// Possible benign context — ruled out by: package/freeform (pkg-skeleton-small), package/freeform (conventional-version)"
            ),
        );
        // Downgraded reads differently from withheld.
        assert_eq!(
            recategorize_annotation(
                "// S downgraded objectives/supply-chain/install-hook/scripts/declaration::sparse-package-install-hook by metadata/package/freeform::pkg-skeleton-small @30"
            )
            .as_deref(),
            Some("// Possible benign context — weakened by: package/freeform (pkg-skeleton-small)"),
        );
        // `#` form, indented, hex spans, a `+N more` tail, a recognized tool and a
        // micro-behavior (leaves kept) and a bare condition-kind leg.
        assert_eq!(
            recategorize_annotation(
                "  # H withheld well-known/malware/stealer/x::family by well-known/tool/sysadmin/chocolatey::official-profile-provisioner @1a2b; micro-behaviors/communications/http/services/telegram::telegram-api-host @426,558; text, +2 more"
            )
            .as_deref(),
            Some(
                "  # Possible benign context — ruled out by: tool/sysadmin/chocolatey (official-profile-provisioner), communications/http (telegram-api-host), text"
            ),
        );
        // No legs at all: still no grade and no conclusion id.
        assert_eq!(
            recategorize_annotation("-- N withheld objectives/evasion/process/injection::x")
                .as_deref(),
            Some("-- Possible benign context — a pattern here was ruled out"),
        );
        // Multi-byte text around a suppression must not panic or be misparsed.
        assert_eq!(
            recategorize_annotation(
                "  // S withheld objectives/x/y::über by metadata/file/naïve::café @3"
            )
            .as_deref(),
            Some("  // Possible benign context — ruled out by: file/naïve (café)"),
        );
        // Ordinary lines that merely use the verbs are untouched.
        assert_eq!(recategorize_annotation("# N withheld the payment"), None);
        assert_eq!(
            recategorize_annotation("// S withheld ümlaut prose here"),
            None
        );
        assert_eq!(
            recategorize_annotation("  // withheld objectives/x::y by metadata/a::b"),
            None
        );
        assert_eq!(recategorize_annotation("// +3 more suppressed"), None);
        // A whole render keeps its other lines byte for byte.
        let render = "code();\n// B downgraded objectives/a/b::c by metadata/d/e::f\nmore();\n";
        assert_eq!(
            recategorize_annotations(render),
            "code();\n// Possible benign context — weakened by: d/e (f)\nmore();\n"
        );
    }

    fn decision(level: u16, probability: f32) -> Decision {
        Decision {
            class: Classification::Hostile,
            probability,
            threshold: 0.5,
            level: Level::At(level),
        }
    }

    fn clean(probability: f32) -> Decision {
        Decision {
            level: Level::Clean,
            ..decision(0, probability)
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

    /// A clean level never fires, so a graded-clean dependency must rank below
    /// an ungraded one carrying a suspicious-or-worse member trait.
    #[test]
    fn clean_verdict_ranks_below_a_severe_finding() {
        assert!(dep_subject_risk(None, true) > dep_subject_risk(Some(clean(0.01)), false));
    }

    /// Registry-provenance-only subjects have no risk signal at all.
    #[test]
    fn ungraded_and_unremarkable_ranks_last() {
        assert!(dep_subject_risk(Some(clean(0.01)), false) > dep_subject_risk(None, false));
        assert!(dep_subject_risk(None, true) > dep_subject_risk(None, false));
    }

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
            "served": "network",
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
                level: Level::Manual,
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
        let ctx = appendix(&edges, &deps, &report).unwrap();
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
        let ctx = appendix(&edges, &deps, &report).unwrap();
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
        assert!(appendix(&[], &deps, &report).is_none());
        edges[0].content_sha256 = None;
        assert!(appendix(&edges, &deps, &report).is_none());
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
            "served": "cache",
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
                level: Level::Manual,
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
                    size: 31,
                    bytes: Some(br#"{"provider_only":{"kept":true}}"#.to_vec()),
                }],
            ),
            file_id: 2,
            artifact_skip: None,
        }];
        let edges = vec![edge];

        let ctx = interpret(
            "root.tgz",
            &"r".repeat(64),
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

        let fetched = Fetched {
            edges: &edges,
            deps: &deps,
            registries: &registries,
        };
        let terminal = crate::engine::render_cards::render_terminal_fetch_context(
            fetched,
            &report,
            &ReportIndex::new(&report),
        )
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
                size: packument.len() as u64,
                bytes: Some(packument.into_bytes()),
            }],
        );

        let ctx = interpret(
            "unload-0.0.1.tgz",
            &"r".repeat(64),
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

        let ctx = interpret(
            "vendor-bundle.zip",
            &"r".repeat(64),
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
    /// instead (see `registry_provenance`).
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
    fn interpret_shows_the_grader_a_filename_not_its_directories() {
        assert_eq!(
            interpret_display_name("/corpus/hostile/2024/x.whl"),
            "x.whl"
        );
        assert_eq!(
            interpret_display_name(r"C:\\samples\\benign\\setup.exe"),
            "setup.exe"
        );
        assert_eq!(interpret_display_name("dir/sub/"), "sub");
        assert_eq!(interpret_display_name("plain.js"), "plain.js");
        assert_eq!(
            interpret_display_name("https://example.test/a/b.png"),
            "https://example.test/a/b.png"
        );

        let mut report = empty_report();
        report.files = vec![cleave::FileAnalysis {
            id: 0,
            path: "/corpus/samples/benign/x.whl".to_string(),
            file_type: "whl".to_string(),
            sha256: "a".repeat(64),
            ..cleave::FileAnalysis::default()
        }];
        let ctx = interpret(
            "/corpus/samples/benign/x.whl",
            &"a".repeat(64),
            None,
            &[],
            &[],
            &[],
            &report,
        );
        assert!(ctx.contains("== PRIMARY x.whl =="), "{ctx}");
        assert!(ctx.contains(r#""path":"x.whl""#), "{ctx}");
        assert!(!ctx.contains("samples/benign"), "{ctx}");
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
                size: 31,
                bytes: Some(br#"{"provider_only":{"kept":true}}"#.to_vec()),
            }],
        );
        let ctx = interpret(
            "removed@1.registry.json",
            &"a".repeat(64),
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
}
