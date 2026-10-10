//! The terminal result card, the `--format tiny` and `--extra` writers, and
//! the human-facing account of fetched dependencies.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::io;
use std::path::Path;

use cleave::output::terminal_width;
use cleave::{Criticality, FileAnalysis, Finding};
use fletch::fetch::FetchRecord;

use super::render_context::{Fetched, ReportIndex, fetched_subjects, registry_only};
use super::verdict::{decision_outranks, graver, trait_family};
use super::{
    ARCHIVE_DELIMITER, EmbeddedFile, MemberEvals, Reason, ScanResult, archive_leaf, is_inside,
};
use crate::fetch::DependencyRegistry;
use crate::interpret::Interpretation;
use crate::model::{Classification, Decision, Level};
use crate::output::{self, BloomMark, Rgb, TerminalTrait};

/// Append full ML diagnostics (route scores + SHAP reasons) under the rendered
/// context, for `--extra`. Shows which route drove the grade and the top SHAP
/// features behind the top-level featurization. Embedded archive members list
/// their route scores; per-member SHAP reasons are not computed (reasons exist
/// only for the top-level file), so to attribute an embedded hit, scan the
/// extracted member directly — it then becomes the top-level file.
pub(crate) fn write_extra_diagnostics(out: &mut dyn io::Write, r: &ScanResult) -> io::Result<()> {
    // Lead with the level, as `output::print_extra` does. It is the only place
    // the loose tail above the suspicious ceiling (L3000..=L25000) is legible:
    // those files grade benign, so nothing else in this render names the rung
    // they fired on.
    writeln!(out, "  level: {}", r.level)?;
    if !r.model_scores.is_empty() {
        writeln!(
            out,
            "  routes (raw): {}",
            output::format_route_scores(&r.model_scores),
        )?;
    }
    if !r.reasons.is_empty() {
        writeln!(out, "  shap (top features by importance):")?;
        for reason in r.reasons.iter().take(12) {
            writeln!(
                out,
                "    imp={:.4} val={:.4}  {}  [{}]",
                reason.importance, reason.value, reason.feature, reason.description,
            )?;
        }
    }
    // A top-10 view of the evaluation table; the table itself is complete.
    for ef in EmbeddedFile::top_offenders(&r.embedded_files, 10) {
        if !ef.model_scores.is_empty() {
            writeln!(
                out,
                "  embedded {} ({}): routes (raw) {}",
                output::tty_text(&ef.path),
                ef.file_type,
                output::format_route_scores(&ef.model_scores),
            )?;
        }
    }
    Ok(())
}

/// Write litmus's `--format tiny` view (machine/LLM-facing, never colored): one
/// ML-verdict line — the gate, calibrated confidence, matched false-positive
/// level — then cleave's annotated context.
pub(crate) fn write_tiny(out: &mut dyn io::Write, r: &ScanResult) -> io::Result<()> {
    let class = r.classification;
    // The bloom flag rides the machine line as `bloom=known-bad|conflicted` — the
    // machine-readable analog of the terminal 🚩/🏴 (the JSON envelope is untouched).
    let bloom = r
        .bloom_mark
        .map_or_else(String::new, |m| format!(" bloom={}", m.tiny_str()));
    let fp_level = match (r.level, class) {
        // Benign is not always "fired nowhere": everything above the suspicious
        // ceiling (L3000) grades benign while still carrying the rung it fired
        // on, up to the grid max (L25000). Report that rung — dropping it hid
        // the whole loose tail from the LLM-facing render. A benign file that
        // fired at no level has nothing to report and keeps the bare line.
        (Level::At(lvl), _) => format!(" fp-level=L{lvl}"),
        (_, Classification::Benign) => String::new(),
        (Level::Clean, _) => " fp-level=L-1".to_string(),
        (Level::Manual, _) => " fp-level=-".to_string(),
    };
    writeln!(
        out,
        "scan {class} confidence={:.3}{fp_level}{bloom}",
        r.probability
    )?;
    if let Some(llm) = &r.interpretation {
        out.write_all(format_llm_line(llm, false).as_bytes())?;
    }
    out.write_all(r.rendered_context.as_bytes())
}

/// One-line LLM verdict, colored by the blended outcome. Shown under the ML
/// verdict in terminal output when `--interpret` produced a result. `color` is
/// false for `--format tiny` (LLM-facing output is never colored).
pub(crate) fn format_llm_line(llm: &Interpretation, color: bool) -> String {
    use colored::Colorize;
    let graded = match llm {
        Interpretation::Failed(failed) if !color => {
            return format!("llm error  {}\n", failed.error);
        }
        Interpretation::Failed(failed) => {
            return format!(
                "llm {}  {}\n",
                "error".truecolor(255, 175, 0),
                failed.error.bright_black(),
            );
        }
        Interpretation::Graded(graded) => graded,
    };
    let grade = graded.grade.as_str();
    if !color {
        return format!(
            "llm {grade} → {} blended={:.3}  {}\n",
            graded.outcome, graded.blended, graded.interpretation,
        );
    }
    let (r, g, b) = match graded.outcome {
        Classification::Hostile => (215, 95, 95),
        Classification::Suspicious => (255, 175, 0),
        Classification::Benign => (95, 175, 95),
    };
    let outcome = graded.outcome.to_string().truecolor(r, g, b).bold();
    format!(
        "llm {} → {outcome} blended={:.3}  {}\n",
        grade.bright_black(),
        graded.blended,
        graded.interpretation.bright_black(),
    )
}

fn terminal_safe_text(text: &str) -> String {
    crate::deptree::strip_ansi(text)
        .chars()
        .filter(|&c| !output::tty_hostile(c))
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

/// The provenance rows a card can carry above its digest: what the artifact is,
/// and what the registry said about it. Both are recovered from a collector's
/// capture record or a `--registry-map`, and both are empty for a scan of a file
/// nobody collected.
#[derive(Debug, Default)]
pub(super) struct CardProvenance {
    pub(super) purl: Option<String>,
    pub(super) registry: Option<String>,
}

/// What a terminal card says about the artifact besides its findings.
pub(super) struct CardHead<'a> {
    pub(super) decision: &'a Decision,
    pub(super) reasons: &'a [Reason],
    pub(super) interpretation: Option<&'a Interpretation>,
    pub(super) sha256: &'a str,
    pub(super) label: &'a str,
    pub(super) bloom_mark: Option<BloomMark>,
    pub(super) provenance: CardProvenance,
}

/// What the card claims this artifact *is*, in the row under its name.
///
/// The bytes come first: an identity cleave read out of the artifact is a
/// property of the sample itself. A collected sample usually has none — a
/// package tarball carries no self-description cleave admits, and its filename
/// is just a filename — so the coordinate the collector recorded falls in
/// behind it. Without that row a scan of a collected corpus never names the
/// package it is judging, which is the one thing an operator needs to act on a
/// verdict about a package that no longer exists to look up.
fn card_identity(
    root: Option<&FileAnalysis>,
    label: &str,
    collected: Option<&str>,
) -> Option<String> {
    root.and_then(|root| terminal_identity_summary(root, label))
        .or_else(|| collected.map(str::to_string))
}

fn terminal_identity_summary(root: &FileAnalysis, label: &str) -> Option<String> {
    let identity = root.identity.as_ref()?;
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
    let leaf = archive_leaf(&path);
    let name = leaf.rsplit('/').next().unwrap_or(leaf);
    terminal_safe_text(name)
}

fn terminal_note_anchor(file: &FileAnalysis, finding_id: &str) -> Option<String> {
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

/// Where a finding sits, for its card row: a line in the primary file, or a
/// member path with its line when it is known.
fn terminal_finding_location(
    by_id: &HashMap<u32, &FileAnalysis>,
    primary_id: Option<u32>,
    file: &FileAnalysis,
    finding: &Finding,
) -> String {
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
            let _ = write!(location, ":{line}");
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
fn finding_is_native(file: &FileAnalysis, finding: &Finding) -> bool {
    finding.src.is_none() || file.composite_sources.contains_key(finding.id.as_str())
}

/// The (at most three) findings a card shows for `files`, the first of which is
/// the card's primary file: strongest severity first, shallower layers before
/// deeper ones, one per behavioral family before a second from any.
fn terminal_top_traits(files: &[&FileAnalysis]) -> Vec<TerminalTrait> {
    let by_id: HashMap<u32, &FileAnalysis> = files.iter().map(|f| (f.id, *f)).collect();
    let primary_id = files.first().map(|f| f.id);
    let mut deepest: HashMap<&str, u32> = HashMap::new();
    for file in files {
        for finding in file.findings.iter().filter(|f| finding_is_native(file, f)) {
            deepest
                .entry(finding.id.as_str())
                .and_modify(|depth| *depth = (*depth).max(file.depth))
                .or_insert(file.depth);
        }
    }

    let mut ranked: Vec<(&FileAnalysis, &Finding)> = files
        .iter()
        .flat_map(|&file| {
            let deepest = &deepest;
            file.findings.iter().filter_map(move |finding| {
                (finding_is_native(file, finding)
                    && finding.crit >= Criticality::Notable
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
    let mut seen_ids = HashSet::new();
    let mut seen_families = HashSet::new();
    for criticality in [
        Criticality::Hostile,
        Criticality::Suspicious,
        Criticality::Notable,
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
                selected.push(TerminalTrait {
                    criticality: finding.crit,
                    description,
                    location: terminal_finding_location(&by_id, primary_id, file, finding),
                });
                // A hostile conclusion is sufficient for triage. Additional
                // rows at the same or weaker severity only repeat support for
                // a verdict the first row already makes unambiguously.
                if criticality == Criticality::Hostile || selected.len() == 3 {
                    return selected;
                }
            }
        }
    }
    selected
}

/// The card's head, top-down: verdict rule (or a plain badge when output is
/// not colored) → artifact → claimed identity → registry → SHA-256.
fn card_head(files: &[&FileAnalysis], head: &CardHead<'_>, is_container: bool) -> String {
    let root = files.first().copied();
    let file_type = root.map_or("", |f| f.file_type.as_str());
    let size = root.map_or(0, |f| f.size);
    let decision = head.decision;
    // Each card *leads* with its blank separator (setting it off from the
    // banner or the previous card) and ends flush — the footer brings its own
    // spacing — so a clean scan's quiet summary still hugs the banner.
    let mut out = String::from("\n");
    if output::color_enabled() {
        out.push_str(&output::terminal_rule(
            &decision.class,
            decision.probability,
            decision.threshold,
            decision.level,
            terminal_width(),
        ));
        out.push('\n');
    } else {
        let (stamp, _) = output::terminal_badge(
            &decision.class,
            decision.probability,
            decision.threshold,
            decision.level,
        );
        out.push_str(&stamp);
        out.push(' ');
    }
    out.push_str(&output::terminal_artifact_line(
        head.label,
        file_type,
        size,
        is_container,
    ));
    out.push('\n');
    let rows = [
        card_identity(root, head.label, head.provenance.purl.as_deref())
            .as_deref()
            .and_then(output::terminal_identity_line),
        head.provenance
            .registry
            .as_deref()
            .and_then(output::terminal_registry_line),
        output::terminal_hash_line(head.sha256, head.bloom_mark),
    ];
    for row in rows.into_iter().flatten() {
        out.push_str(&row);
        out.push('\n');
    }
    out
}

/// The terminal result card for the artifact whose own files are `files`
/// (the first is its root): the head, the model's or the LLM's one-line
/// reason, and the three strongest traits across the whole artifact. Plain
/// (piped) output keeps the same information as unframed grep-able lines.
pub(super) fn render_terminal_context(
    files: &[&FileAnalysis],
    head: &CardHead<'_>,
    members: &MemberEvals,
) -> String {
    let is_container = files.iter().any(|f| f.depth > 0);

    // A grab-bag archive (several independent packages zipped together) reads
    // far better as a stack of per-package verdict cards than as one inherited
    // verdict over a flat member list. When the archive holds two or more
    // packages that independently scored suspicious+, switch to that layout.
    if is_container && let Some(cards) = render_archive_cards(files, head, members) {
        return cards;
    }

    let mut out = card_head(files, head, is_container);
    if let Some(interp) = output::terminal_interpretation(head.interpretation, 1) {
        out.push_str(&interp);
        out.push('\n');
    } else if let Some(trailer) = output::terminal_trailer(head.reasons) {
        // Without an LLM, keep the model's compact explanation in the same slot.
        out.push_str(&trailer);
        out.push('\n');
    }

    let body = output::terminal_trait_rows(&terminal_top_traits(files), terminal_width());
    if !body.trim().is_empty() {
        out.push_str(&body);
    }
    // Flush ending: the next card (or the footer) supplies the separator.
    while out.ends_with("\n\n") {
        out.pop();
    }
    out
}

/// The id of the depth-1 ancestor of `file` — the top-level package it belongs
/// to inside the root archive — or `None` for the root itself. `parent_id` is
/// unreliable here (it is dropped for non-container members during compaction),
/// so membership is decided by path: a file belongs to the depth-1 package whose
/// path is the longest archive-prefix of its own.
fn top_package_id(file: &FileAnalysis, roots: &[(u32, &str)]) -> Option<u32> {
    if file.depth == 0 {
        return None;
    }
    if file.depth == 1 {
        return Some(file.id);
    }
    roots
        .iter()
        .filter(|(_, root)| is_inside(&file.path, root))
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
fn render_archive_cards(
    files: &[&FileAnalysis],
    head: &CardHead<'_>,
    members: &MemberEvals,
) -> Option<String> {
    let by_id: HashMap<u32, &FileAnalysis> = files.iter().map(|f| (f.id, *f)).collect();

    // Group every file under its top-level package (its depth-1 ancestor).
    let roots: Vec<(u32, &str)> = files
        .iter()
        .filter(|f| f.depth == 1)
        .map(|f| (f.id, f.path.as_str()))
        .collect();
    let mut members_of: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for file in files {
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
        .filter_map(|(pkg, package_members)| {
            let decision = package_members
                .iter()
                .filter_map(|id| members.get(&u64::from(*id)).map(EmbeddedFile::decision))
                .reduce(graver)?;
            Some(Package {
                id: pkg,
                decision,
                members: package_members,
            })
        })
        .collect();

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
        .filter(|p| p.decision.class >= Classification::Suspicious)
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
    let member_native: HashSet<&str> = files
        .iter()
        .filter(|f| f.depth > 0)
        .flat_map(|f| {
            f.findings
                .iter()
                .filter(|x| x.src.is_none())
                .map(|x| x.id.as_str())
        })
        .collect();
    let root = files.first().copied();
    let is_archive_own = |f: &Finding| f.src.is_none() && !member_native.contains(f.id.as_str());
    let archive_card = root.is_some_and(|root| {
        root.findings
            .iter()
            .any(|f| f.crit >= Criticality::Hostile && is_archive_own(f))
    });

    // Fall back to the single classic card unless this is genuinely a grab-bag
    // (two or more independently-notable packages).
    if notable.len() < 2 {
        return None;
    }

    // ── Banner: verdict rule → 📦 name · TYPE · size → hash ──
    let mut out = card_head(files, head, true);
    if let Some(interp) = output::terminal_interpretation(head.interpretation, 1) {
        out.push_str(&interp);
        out.push('\n');
    }

    // Ordinary archive members are files, not independent packages. Keep their
    // strongest findings in one aligned grid; reserve cards for nested
    // containers. The archive-level verdict may be inherited from these same
    // members, so it must not force the package-card layout.
    if notable.iter().all(|pkg| {
        pkg.members.len() == 1
            && by_id
                .get(&pkg.id)
                .is_some_and(|file| decoded_region_display_path(&file.path).is_none())
    }) {
        let mut traits = Vec::new();
        for pkg in notable.iter().take(3) {
            let file = *by_id.get(&pkg.id)?;
            if let Some(mut finding) = terminal_top_traits(&[file]).into_iter().next() {
                let path = terminal_safe_text(&package_display_path(&file.path));
                finding.location = finding
                    .location
                    .strip_prefix("line ")
                    .map_or_else(|| path.clone(), |line| format!("{path}:{line}"));
                traits.push(finding);
            }
        }
        out.push_str(&output::terminal_trait_rows(&traits, terminal_width()));
        out.push('\n');
        let additional = notable.len().saturating_sub(traits.len());
        let clean = packages.len().saturating_sub(notable.len());
        let mut omitted = Vec::new();
        if additional > 0 {
            let noun = if additional == 1 { "file" } else { "files" };
            omitted.push(format!("{additional} additional affected {noun}"));
        }
        if clean > 0 {
            let noun = if clean == 1 { "file" } else { "files" };
            omitted.push(format!("{clean} clean {noun} omitted"));
        }
        if !omitted.is_empty() {
            let _ = write!(out, "\n {}\n", omitted.join(" · "));
        }
        return Some(out);
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
        let ptype = file.map_or("", |f| f.file_type.as_str());
        let psize = file.map_or(0, |f| f.size);
        let member_ids: HashSet<u32> = pkg.members.iter().copied().collect();

        // Rank over every member in the package, then spend exactly three rows
        // on its strongest distinct traits. A large package reads like one
        // artifact instead of a transcript of its member traversal.
        let package: Vec<FileAnalysis> = files
            .iter()
            .filter(|f| member_ids.contains(&f.id))
            .map(|f| {
                let mut f = (*f).clone();
                f.path = collapse_decoded_dup(&f.path);
                f
            })
            .collect();
        let package: Vec<&FileAnalysis> = package.iter().collect();
        let rows = output::terminal_trait_rows(
            &terminal_top_traits(&package),
            terminal_width().saturating_sub(2),
        );
        if embedded_tree {
            output::terminal_embedded_branch(&pkg.decision.class, &name, ptype, psize, &rows, last)
        } else {
            output::terminal_card(&pkg.decision.class, &name, ptype, psize, &rows)
        }
    };

    // The root's own conclusion belongs directly beneath the root metadata in a
    // decoded tree. A second card repeating the root filename would imply a
    // sibling object where none exists.
    if archive_card && let Some(root) = root {
        // The archive's own findings, and the member files their cross-package
        // trails point at — the members must stay in the view so those `↳` legs
        // resolve to real paths, but only the root's findings block is kept.
        let own: HashSet<&str> = root
            .findings
            .iter()
            .filter(|f| is_archive_own(f))
            .map(|f| f.id.as_str())
            .collect();
        let referenced: HashSet<u32> = own
            .iter()
            .filter_map(|id| root.composite_sources.get(*id))
            .flatten()
            .map(|s| s.file)
            .collect();
        let mut own_root = root.clone();
        own_root.findings.retain(|f| own.contains(f.id.as_str()));
        let view: Vec<&FileAnalysis> = files
            .iter()
            .filter(|f| f.id == root.id || referenced.contains(&f.id))
            .map(|&f| if f.id == root.id { &own_root } else { f })
            .collect();
        let rows = output::terminal_trait_rows(
            &terminal_top_traits(&view),
            terminal_width().saturating_sub(2),
        );
        if !rows.trim().is_empty() {
            if embedded_tree {
                out.push_str(&rows);
                out.push('\n');
            } else {
                let file_type = root.file_type.as_str();
                out.push('\n');
                out.push_str(&output::terminal_card(
                    &head.decision.class,
                    head.label,
                    file_type,
                    root.size,
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
        let note = format!("{omitted} clean {plural} not shown");
        let _ = writeln!(out, "\n {}", output::fg(Rgb(100, 100, 100), &note));
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
    let Some((rest, seg)) = head.rsplit_once(ARCHIVE_DELIMITER) else {
        return path.to_string();
    };
    if !seg.is_empty()
        && (rest == seg
            || rest
                .strip_suffix(seg)
                .is_some_and(|before| before.ends_with(ARCHIVE_DELIMITER)))
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
    path.split_once(ARCHIVE_DELIMITER).map_or_else(
        || path.to_string(),
        |(_, m)| m.replace(ARCHIVE_DELIMITER, "/"),
    )
}

/// Human-facing counterpart to the LLM render: the same fetched subjects, with
/// provenance before traits, but only hostile ones, and only the processed
/// acquisition and registry fields.
pub(super) fn render_terminal_fetch_context(
    fetched: Fetched<'_>,
    report: &cleave::AnalysisReport,
    index: &ReportIndex<'_>,
) -> Option<String> {
    struct FetchBranch {
        verdict: Option<(Classification, f32)>,
        subject: &'static str,
        source: String,
        body: String,
    }
    let mut out = String::new();
    let mut branches = Vec::new();

    for subject in fetched_subjects(fetched, index) {
        // Only hostile dependencies earn a provenance block: a benign-but-old or
        // merely-notable dependency is exactly the noise that buried the real
        // ones. Hostility is a hostile verdict on the fetched bytes, or a hostile
        // finding on one of its members.
        let hostile_finding = subject.has_finding(index, Criticality::Hostile);
        let hostile_verdict = subject
            .verdict()
            .is_some_and(|verdict| verdict.class == Classification::Hostile);
        if !hostile_finding && !hostile_verdict {
            continue;
        }
        let rec = subject.edge;
        let mut body = String::new();
        let _ = writeln!(
            body,
            "{}",
            output::terminal_reference_locator_row(&rec.locator)
        );
        write_terminal_fetch_redirects(&mut body, rec);
        let _ = writeln!(
            body,
            "{}",
            output::terminal_reference_hash_row(
                &subject.root.sha256,
                &subject.root.file_type,
                subject.root.size,
            )
        );
        let mut signals = Vec::new();
        if rec.pin_verified == Some(false) {
            signals.push("checksum mismatch");
        }
        if let Some(registry) = subject.registry {
            write_terminal_registry_provenance(&mut body, registry, &mut signals, true);
        } else if !signals.is_empty() {
            let _ = writeln!(body, " \u{00b7}   signals  {}", signals.join(" · "));
        }

        let view: Vec<&FileAnalysis> = report
            .files
            .iter()
            .filter(|file| {
                subject.files.contains(&file.id)
                    || subject.registry.is_some_and(|r| r.file_id == file.id)
            })
            .collect();
        let rows = output::terminal_trait_rows(
            &terminal_top_traits(&view),
            terminal_width().saturating_sub(3),
        );
        for row in rows.lines() {
            let _ = writeln!(body, "{row}");
        }
        branches.push(FetchBranch {
            verdict: subject.verdict().map(|v| (v.class, v.probability)),
            subject: if rec.kind == fletch::RefKind::Dependency {
                "dependency"
            } else {
                "external URL"
            },
            source: terminal_fetch_source(rec, index),
            body,
        });
    }

    for (index, branch) in branches.iter().enumerate() {
        if index == 0 {
            out.push('\n');
        }
        out.push_str(&output::terminal_reference_branch(
            branch
                .verdict
                .as_ref()
                .map(|(classification, probability)| (classification, *probability)),
            branch.subject,
            &branch.source,
            &branch.body,
            index + 1 == branches.len(),
        ));
    }

    for registry in registry_only(fetched) {
        // A registry-only entry (no bytes fetched, so no verdict) is shown only
        // when the registry itself flags it hostile — a pulled version or a
        // security hold. "Older than fetch age limit" and other benign states are
        // not findings; they were the bulk of the old noise.
        let record = &registry.provenance.record;
        if record.version_removed != Some(true) && record.security_hold != Some(true) {
            continue;
        }
        let status = registry
            .artifact_skip
            .unwrap_or("REGISTRY ONLY")
            .to_uppercase();
        let _ = writeln!(
            out,
            "\n{}",
            output::terminal_reference_status_heading(&status, "REGISTRY")
        );
        let _ = writeln!(
            out,
            "{}",
            output::terminal_reference_locator(&registry.locator)
        );
        let mut signals = Vec::new();
        write_terminal_registry_provenance(&mut out, registry, &mut signals, false);
        let view: Vec<&FileAnalysis> = report
            .files
            .iter()
            .filter(|file| file.id == registry.file_id)
            .collect();
        let rows = output::terminal_trait_rows(
            &terminal_top_traits(&view),
            terminal_width().saturating_sub(3),
        );
        for row in rows.lines() {
            let _ = writeln!(out, "   {row}");
        }
    }

    // Nothing is footnoted for the artifacts that stayed quiet. Their count
    // says nothing at the decision point — the streamed fetch summary already
    // reports how many were retrieved — and a section that ends by naming what
    // it declined to show reads as withheld evidence rather than a clean pass.
    (!out.is_empty()).then_some(out)
}

fn terminal_fetch_source(rec: &FetchRecord, index: &ReportIndex<'_>) -> String {
    rec.source_sha256
        .as_deref()
        .and_then(|sha| index.by_sha.get(sha))
        .map_or_else(
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

fn write_terminal_fetch_redirects(out: &mut String, rec: &FetchRecord) {
    let resolved = rec.resolved_url.as_deref();
    if let Some(resolved) = resolved.filter(|url| *url != rec.locator) {
        let _ = writeln!(out, " \u{00b7}   resolved  {resolved}");
    }
    if let Some(final_url) = rec
        .final_url
        .as_deref()
        .filter(|url| Some(*url) != resolved && *url != rec.locator)
    {
        let _ = writeln!(out, " \u{00b7}   final  {final_url}");
    }
}

fn write_terminal_registry_provenance(
    out: &mut String,
    registry: &DependencyRegistry,
    signals: &mut Vec<&'static str>,
    marker_rows: bool,
) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn terminal_report(value: serde_json::Value) -> cleave::AnalysisReport {
        serde_json::from_value(value).expect("valid terminal report fixture")
    }

    /// A collected package tarball describes itself nowhere cleave can read, so
    /// without the capture record the card names a filename and nothing else.
    #[test]
    fn collected_coordinate_names_a_package_the_bytes_do_not() {
        let report = terminal_report(serde_json::json!({
            "version": "3",
            "files": [{
                "id": 0, "path": "blueai-cli-0.7.0.tgz", "depth": 0,
                "file_type": "gzip", "sha256": "root", "size": 10, "findings": []
            }]
        }));
        assert_eq!(
            card_identity(
                report.files.first(),
                "blueai-cli-0.7.0.tgz",
                Some("pkg:npm/blueai-cli@0.7.0")
            )
            .as_deref(),
            Some("pkg:npm/blueai-cli@0.7.0"),
        );
        assert_eq!(
            card_identity(report.files.first(), "blueai-cli-0.7.0.tgz", None),
            None
        );
    }

    /// An identity read out of the bytes outranks one claimed about them.
    #[test]
    fn content_identity_outranks_the_capture_record() {
        let report = terminal_report(serde_json::json!({
            "version": "3",
            "files": [{
                "id": 0, "path": "installer.msi", "depth": 0,
                "file_type": "msi", "sha256": "root", "size": 10, "findings": [],
                "identity": {
                    "title": {"value": "NordPass Installer", "source": "msi.title", "verified": false},
                    "trust": "unsigned"
                }
            }]
        }));
        assert_eq!(
            card_identity(
                report.files.first(),
                "installer.msi",
                Some("pkg:npm/evil@1.0.0")
            )
            .as_deref(),
            Some("\u{201c}NordPass Installer\u{201d}"),
        );
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

        let traits = terminal_top_traits(&report.files.iter().collect::<Vec<_>>());
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

        let traits = terminal_top_traits(&report.files.iter().collect::<Vec<_>>());
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

        let traits = terminal_top_traits(&report.files.iter().collect::<Vec<_>>());
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
            terminal_identity_summary(&report.files[0], "invoice.docx").as_deref(),
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
            assert_eq!(terminal_identity_summary(&report.files[0], label), None);
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
            terminal_identity_summary(&report.files[0], "agent-1.2.3.tgz").as_deref(),
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

    fn empty_report() -> cleave::AnalysisReport {
        serde_json::from_value(serde_json::json!({"version": "3"})).unwrap()
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
            "served": "network",
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
        assert_eq!(
            terminal_fetch_source(&rec, &ReportIndex::new(&report)),
            "this file"
        );
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
