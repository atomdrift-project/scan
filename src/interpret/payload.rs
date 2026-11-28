//! The evidence the LLM grader reads: a cleave report rendered as the code
//! its findings matched in, each finding announced on a comment line above the
//! row it describes.
//!
//! ```text
//! evil.sh<TAB>shell 4KB 142
//! # H 20:1 Decodes and runs a base64 payload (exec/decode-exec-base64)
//! exec(base64.b64decode(p))
//! ```
//!
//! The header is `path<TAB>type size score`, then the identity headline when
//! the file has one. An annotation is `{marker} SEV LINE:COL desc (id)` for
//! source and `{marker} SEV OFFSET desc (id)` for binaries, `SEV` one of
//! `H`/`S`/`N`/`C`/`B`. Only the top-ranked files render, and only as code:
//! a finding with no drawn window is not shown. Source windows keep the rows
//! beside each finding and the file's leading comment, nothing wider.
//!
//! `--format interpret` prints exactly this render (sanitized and
//! recategorized, see [`crate::engine`]), the same bytes the `--llm` query
//! sends.

mod identity;
mod select;
mod windows;

use std::collections::{BTreeSet, HashMap, HashSet};

use cleave::output::{sample_text, sanitize_terminal_output};
use cleave::types::{
    AnalysisGap, AnalysisReport, Criticality, ENCODING_DELIMITER, FileAnalysis, Finding, Istr,
    Note, Rel,
};

use self::select::{
    Planned, archive_container_paths, evidence_legs, indent_block, native_finding_depths,
    relabel_composite_notes, select_ids, tiny_path, window_plan,
};

/// Files that draw context, highest ranked first. Four lost two benign
/// samples on the LLM fixtures once rendered for real (2026-11-23).
const MAX_CONTEXT_FILES: usize = 5;

/// Findings shown per file, ranked by `crit × conf`. Suspicious and hostile
/// findings are shown whatever the cap.
const TOP_N: usize = 250;

/// A context file's selection is topped up with its best low-tier traits until
/// this many show, so a benign binary with nothing notable still reads as
/// *something* — its strongest imports, capabilities and structure.
const LOW_TIER_FILL: usize = 15;

/// Windows drawn per finding id in a file, in file order: a trait's second and
/// third location repeat the evidence its first already showed, and over a
/// 96-sample corpus the repeats were 11% of the render with no grade that
/// depended on them.
const WINDOWS_PER_FINDING: usize = 1;

/// Context windows drawn per file — highest criticality first, counting a
/// composite leg at its withheld composite's, then the most story per byte
/// across trait families.
const MAX_WINDOWS: usize = 64;

/// Bytes of window content drawn per file, strongest windows first. A byte
/// budget rather than a count because windows differ in weight: a flat count
/// of 16 took a 63 MB installer's 100+ windows (30k tokens) down to the
/// sixteen that carried its grade, but also cut a 28 KB minified extension's
/// 68 line windows to 16 and lost its grade. 12 KiB is ~40 byte windows or
/// ~80 source windows.
const WINDOW_BYTES: usize = 12 * 1024;

/// Columns of a source row shown, centred on its match. In a sweep of ten
/// widths from 104 to 200, the widths that held every LLM fixture grade were
/// 120, 168, 184 and 200; each one between lost a sample, and not the same
/// one, so a loss is where the clip happens to land rather than a threshold.
/// 200 is the narrowest width with nothing failing above it, plus a margin of
/// 16 (2026-11-27).
const SOURCE_WIDTH: usize = 200;

/// Source rows kept on each side of a row a finding matched in. None lost
/// three fixture grades and more than one changed none, so one it is; rows
/// before only, or after only, each lost grades too (2026-11-27).
const CONTEXT_ROWS: usize = 1;

/// Hook scripts a package installer runs on its own at install, upgrade or
/// removal (macOS pkg `preinstall`/`postinstall`, Debian maintainer scripts,
/// RPM-style `pre`/`postremove`). What an installer does to the machine is in
/// them, beside the bulk of payload binaries that outscore them on breadth.
const INSTALL_HOOKS: [&str; 12] = [
    "preinstall",
    "postinstall",
    "preupgrade",
    "postupgrade",
    "preflight",
    "postflight",
    "preinst",
    "postinst",
    "prerm",
    "postrm",
    "preremove",
    "postremove",
];

/// Render `report` for the LLM grader.
#[must_use]
pub(crate) fn render(report: &AnalysisReport) -> String {
    // Composites are withheld; this is the evidence owed in their place, per
    // file (see `evidence_legs`).
    let legs = evidence_legs(&report.files);
    let files: Vec<&FileAnalysis> = report
        .files
        .iter()
        .filter(|f| {
            f.findings.iter().any(|g| g.crit != Criticality::Filtered)
                || f.identity.is_some()
                || legs.contains_key(&f.id)
        })
        .collect();
    if files.is_empty() {
        // A report with no files at all keeps its own gap (an archive cleave
        // could not open): the one thing it can say.
        let mut out = String::new();
        if report.files.is_empty() && !report.analysis_gaps.is_empty() {
            let reasons: Vec<&str> = report
                .analysis_gaps
                .iter()
                .map(AnalysisGap::label)
                .collect();
            out = format!(
                "Analysis incomplete for {:?}: {}\n",
                tiny_path(&report.target.path),
                reasons.join(", ")
            );
        }
        return sanitize_terminal_output(&out).into_owned();
    }

    // Archive containers re-evaluate their members' content at container
    // scope, so they carry copies of member findings beside their genuine
    // cross-file composites. A finding native deeper down belongs to that
    // member, not the container above it.
    let container_paths = archive_container_paths(&files);
    let id_max_native_depth = native_finding_depths(&files);
    let id_to_file: HashMap<u32, &FileAnalysis> = files.iter().map(|f| (f.id, *f)).collect();
    let context_file_ids = context_files(&files, &container_paths, &legs);

    // Which artifact each file belongs to: the id of the fetched-subtree root
    // above it, or None for the sample's own files. A fetched payload is a
    // separate artifact that happens to be rendered here.
    let artifact_of: HashMap<u32, Option<u32>> = files
        .iter()
        .map(|f| (f.id, fetched_root_of(f, &id_to_file)))
        .collect();

    // Each distinct finding shows once per artifact: the same product name or
    // "no symbol table" across a hundred members is noise. Root renders first,
    // so its native findings win. Not across an artifact boundary: a fetched
    // dependency firing the same finding is a second, independent occurrence.
    let mut seen_by_artifact: HashMap<Option<u32>, HashSet<&str>> = HashMap::new();
    let root_depth = files.iter().map(|f| f.depth).min().unwrap_or(0);
    // What each shown file draws from: its selection, owed evidence and notes.
    let mut entries: Vec<Entry<'_>> = Vec::new();
    for &file in &files {
        // The view renders its context files and nothing else, with one
        // exception: the subject's root keeps its header and identity lines
        // without a slot. What a package says it is (a wheel's description, a
        // binary's signer) is what behavior is weighed against.
        let identity_only = !context_file_ids.contains(&file.id)
            && file.depth == root_depth
            && file.identity.is_some();
        if !context_file_ids.contains(&file.id) && !identity_only {
            continue;
        }
        // A platform-signed file that raised nothing above notable is told by
        // its header and identity lines alone.
        let mut selected = if identity_only || identity::vouched_and_quiet(file) {
            Vec::new()
        } else {
            select_ids(file)
        };

        // An archive container keeps only what is specific to it: cross-file
        // composites and true matches on its own bytes.
        let is_container = container_paths.contains(file.path.as_str());
        if is_container {
            selected.retain(|id| {
                id_max_native_depth
                    .get(id)
                    .is_none_or(|&deepest| deepest <= file.depth)
            });
        }

        // A decoded layer is exempt from dedup. Its host matches the decoded
        // fragments back onto the encoded bytes and renders first, so the
        // host's copy — annotating an unreadable run of char codes or hex —
        // would win, and the layer, the one place that finding reads as code,
        // would be left with nothing to draw.
        if !file.path.contains(ENCODING_DELIMITER) {
            let seen = seen_by_artifact
                .entry(artifact_of.get(&file.id).copied().flatten())
                .or_default();
            selected.retain(|&id| seen.insert(id));
        }

        // Owed evidence renders here whatever the filters above decided.
        let owed: HashSet<&str> = legs.get(&file.id).into_iter().flatten().copied().collect();
        for &id in legs.get(&file.id).into_iter().flatten() {
            if !selected.contains(&id) {
                selected.push(id);
            }
        }
        if selected.is_empty() && file.identity.is_none() {
            continue;
        }

        // A container's bytes are packed member data, so its windows are
        // reserved for genuine atomic matches on its own bytes, never a
        // cross-file composite that merely landed a stray offset.
        if is_container {
            selected.retain(|id| {
                file.findings
                    .iter()
                    .any(|f| f.id == **id && f.src.is_none() && f.trait_refs.is_empty())
            });
        }

        // Capture's overlap dedup lets a withheld composite's note outrank its
        // own leg's at the same span; hand such notes to the legs.
        let notes = cleave::context::notes(&file.findings, &file.context);
        let notes = relabel_composite_notes(file, &notes, &selected).unwrap_or(notes);
        entries.push(Entry {
            file,
            selected,
            owed,
            notes,
            is_container,
        });
    }

    let plans: Vec<Vec<Planned<'_>>> = entries
        .iter()
        .map(|e| window_plan(e.file, &e.notes, &e.selected))
        .collect();
    let mut out = String::new();
    let mut emitted = false;
    for (entry, plan) in entries.iter().zip(&plans) {
        let file = entry.file;
        let mut body = String::new();
        if let Some(id) = &file.identity {
            identity::render_identity(&mut body, id);
        }
        windows::render_windows(&mut body, file, plan, &entry.owed);
        // A file whose body came out empty is a lone header with nothing
        // beneath it — unless it is a container, whose header anchors the
        // members nested below it, or carries an identity headline.
        if body.trim().is_empty() && file.identity.is_none() && !entry.is_container {
            continue;
        }
        if emitted {
            out.push('\n');
        }
        emitted = true;
        let mut block = String::new();
        identity::header(&mut block, file);
        block.push_str(&body);
        indent_block(&mut out, &block, file.depth);
    }
    if emitted {
        out.push('\n');
    }
    sanitize_terminal_output(&out).into_owned()
}

/// A file the view shows, and what it draws from.
struct Entry<'a> {
    file: &'a FileAnalysis,
    /// Finding ids it may annotate.
    selected: Vec<&'a str>,
    /// Evidence a withheld composite rests on, shown whatever its grade.
    owed: HashSet<&'a str>,
    /// Notes per captured window.
    notes: Vec<Vec<Note>>,
    is_container: bool,
}

/// The files that draw context, best first, up to [`MAX_CONTEXT_FILES`].
///
/// A file past the cap renders nothing at all — a long tail of headers and
/// bare notes is comment without code. Only files that captured context
/// compete for a slot, and never a container: its bytes are packed member
/// data. Ranking, in order:
/// - A decoded layer stands in for its host: the host matches the decoded
///   fragments back onto its encoded bytes and so outscores the layer, but its
///   windows show the encoding and the layer's show the program. The layer
///   ranks at its host's score and the host after every other file.
/// - A file holding the evidence a hostile or suspicious composite rests on,
///   so that evidence is drawn.
/// - An installer's hook scripts: what it does to the machine is there, and
///   payload binaries outscore them on breadth.
/// - Score.
///
/// Byte-identical copies take one slot between them (a package that ships its
/// modules twice), and so do a package's per-architecture builds of one
/// program (`x64/zalocli`, `arm64/zalocli`), which differ in SHA-256 but show
/// the same code.
fn context_files(
    files: &[&FileAnalysis],
    container_paths: &HashSet<&str>,
    legs: &HashMap<u32, BTreeSet<&str>>,
) -> HashSet<u32> {
    let mut ranked: Vec<&FileAnalysis> = files
        .iter()
        .copied()
        .filter(|f| !f.context.is_empty() && !container_paths.contains(f.path.as_str()))
        .collect();
    let host_of = |f: &FileAnalysis| {
        f.path
            .split_once(ENCODING_DELIMITER)
            .map(|(host, _)| host.to_string())
    };
    let host_score: HashMap<&str, u32> = files
        .iter()
        .filter(|f| !f.path.contains(ENCODING_DELIMITER))
        .map(|f| (f.path.as_str(), f.score))
        .collect();
    // Only a layer that can take the slot (it captured windows) displaces its
    // host; otherwise the host's slot would go to some unrelated file.
    let decoded_hosts: HashSet<String> = ranked.iter().filter_map(|f| host_of(f)).collect();
    let rank_score = |f: &FileAnalysis| match host_of(f) {
        Some(host) => f
            .score
            .max(host_score.get(host.as_str()).copied().unwrap_or(0)),
        None => f.score,
    };
    ranked.sort_by_key(|f| {
        std::cmp::Reverse((
            !decoded_hosts.contains(&f.path),
            legs.contains_key(&f.id),
            is_install_hook(&f.path),
            rank_score(f),
        ))
    });
    let mut seen_sha = HashSet::new();
    let mut seen_executable: HashSet<&str> = HashSet::new();
    ranked
        .iter()
        .filter(|f| f.sha256.is_empty() || seen_sha.insert(f.sha256.as_str()))
        .filter(|f| {
            !matches!(f.file_type.as_str(), "macho" | "elf" | "pe")
                || seen_executable.insert(base_name(&f.path))
        })
        .take(MAX_CONTEXT_FILES)
        .map(|f| f.id)
        .collect()
}

/// The id of the fetched-subtree root at or above `file`, or `None` when it
/// belongs to the sample's own tree. A graft root carries `Rel::Fetched`;
/// everything beneath it inherits membership. The walk is bounded by the file
/// count so a crafted report with a parent cycle terminates.
fn fetched_root_of(file: &FileAnalysis, by_id: &HashMap<u32, &FileAnalysis>) -> Option<u32> {
    let mut cur = file;
    for _ in 0..by_id.len() {
        if cur.rel == Rel::Fetched {
            return Some(cur.id);
        }
        cur = by_id.get(&cur.parent_id?)?;
    }
    None
}

/// A path's last component, past both `/` and the archive delimiter.
fn base_name(path: &str) -> &str {
    path.rsplit(['/', '!']).next().unwrap_or(path)
}

/// Whether `path` names an installer hook script ([`INSTALL_HOOKS`]).
fn is_install_hook(path: &str) -> bool {
    let name = base_name(path);
    INSTALL_HOOKS.iter().any(|h| name.eq_ignore_ascii_case(h))
}

/// The atoms `roots`' composites rest on within `file`: an id some finding
/// here composes from `trait_refs` is expanded, every other id is an atom.
/// A `BTreeSet` so every consumer iterates in one stable order.
fn composite_atoms<'a>(
    file: &'a FileAnalysis,
    roots: impl IntoIterator<Item = &'a Finding>,
) -> BTreeSet<&'a str> {
    let refs_of: HashMap<&str, &[Istr]> = file
        .findings
        .iter()
        .filter(|f| !f.trait_refs.is_empty())
        .map(|f| (f.id.as_str(), f.trait_refs.as_slice()))
        .collect();
    let mut stack: Vec<&str> = roots
        .into_iter()
        .flat_map(|f| f.trait_refs.iter().map(Istr::as_str))
        .collect();
    let mut seen = HashSet::new();
    let mut atoms = BTreeSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match refs_of.get(id) {
            Some(refs) => stack.extend(refs.iter().map(Istr::as_str)),
            None => {
                atoms.insert(id);
            }
        }
    }
    atoms
}

/// Whether `finding` is shown at all: a component only when a composite in
/// the file draws on it.
fn tiny_should_show(finding: &Finding, file: &FileAnalysis) -> bool {
    finding.crit != Criticality::Component
        || file
            .findings
            .iter()
            .any(|f| f.trait_refs.iter().any(|id| id == &finding.id))
}

/// A description as displayed. Engine findings put sample text in
/// descriptions (a signer name, an embedded file's name), so controls are
/// made visible.
fn display_description(desc: &str) -> String {
    sample_text(desc).trim().to_string()
}

/// `desc (id)`, or a bare `id` when the description is empty, so the grader
/// can name the exact trait that fired.
fn annotate_desc(desc: &str, id: &str) -> String {
    // Engine ids can carry sample text (an entitlement key, an import name).
    let id = sample_text(id);
    if desc.is_empty() {
        id.into_owned()
    } else {
        format!("{desc} ({id})")
    }
}

/// A header field on one line: runs of whitespace collapse to one space.
fn tiny_field(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A byte size as `512B`, `4KB`, `1.2MB` or `3.4GB`; empty for zero.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    match bytes {
        0 => String::new(),
        b if b >= GB => format!("{:.1}GB", b as f64 / GB as f64),
        b if b >= MB => format!("{:.1}MB", b as f64 / MB as f64),
        b if b >= KB => format!("{}KB", b / KB),
        b => format!("{b}B"),
    }
}

#[cfg(test)]
mod tests;
