use super::*;

fn grade(line: &str) -> Option<char> {
    split_annotation(line).map(|(_, _, sev, _)| sev)
}

#[test]
fn annotation_parser_recognizes_findings_not_source_comments() {
    // Annotations with a location, without one, and at every marker parse to
    // their severity letter…
    assert_eq!(grade("// N 22:21 fetch() API call"), Some('N'));
    assert_eq!(
        grade("// H Anomalous package hides exfiltration"),
        Some('H')
    );
    assert_eq!(grade("  # S 2:2 suspicious thing"), Some('S'));
    assert_eq!(grade("-- B"), Some('B'));
    // …while real source comments that superficially resemble the shape do
    // not (lower-case word, no space after a capital, prose after a colon).
    assert_eq!(grade("// Suspicious behavior handled here"), None);
    assert_eq!(grade("# Note: N is the count"), None);
    assert_eq!(grade("-- hostile is a variable"), None);
    assert_eq!(grade("code();"), None);
}

/// Suppression records share the annotation grammar, so the readability
/// gate skips them as non-source like any other annotation.
#[test]
fn suppression_records_are_annotations() {
    for line in [
        "// H withheld objectives/exfil/x::y by metadata/pkg/test-fixture::z @30",
        "  # H downgraded objectives/a::b",
        "-- S suppressed objectives/a::b by micro-behaviors/c::d; metadata/e::f, +2 more",
    ] {
        assert!(split_annotation(line).is_some(), "{line}");
    }
}

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
    render_interpret_context(&primary, fetched, report, &ReportIndex::new(report))
}

/// The dependency appendix, with nothing fetched from a registry.
fn appendix(edges: &[FetchRecord], deps: &[DepResult], report: &AnalysisReport) -> Option<String> {
    let fetched = Fetched {
        edges,
        deps,
        registries: &[],
    };
    render_dependency_context(fetched, report, &ReportIndex::new(report))
}

#[test]
fn recategorizing_an_annotation_survives_multibyte_descriptions() {
    // A description opening with a multi-byte character must not split the
    // severity off at a computed byte index that lands inside the character.
    let line = "  // H über-loader resolves imports (objectives/anti-static/obfuscation/string::x)";
    assert_eq!(
        recategorize_annotation(line).as_deref(),
        Some("  // Possible anti-static/obfuscation — über-loader resolves imports"),
    );
    // Multi-byte content anywhere else is fine too, and a whole render of it
    // must not panic.
    let render =
        "== PRIMARY x ==\n# S 4:2 naïve café résumé (micro-behaviors/data/encode::y)\n  körper\n";
    assert!(recategorize_annotations(render).contains("Possible data/encode — naïve café résumé"));
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
        recategorize_annotation("// F 9:1 packed section (metadata/binary/packer::upx)").as_deref(),
        // The `line:col` pointer survives ahead of the category — it locates
        // the finding rather than grading it.
        Some("// 9:1 Possible binary/packer — packed section"),
    );
    // A third-party signature has no prose and no parenthesized id: the path
    // *is* the body. Left alone, `// H Detects Quasar RAT (third_party/…)`
    // and its bare cousin were the loudest grades still leaking through.
    assert_eq!(
        recategorize_annotation("// H third_party/elastic/Linux_Trojan_Ladvix/linux/trojan/ladvix")
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
        recategorize_annotation("-- N withheld objectives/evasion/process/injection::x").as_deref(),
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

    let (full, dropped) = budget_primary_context(&mut report.clone(), usize::MAX);
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
    let mk_file = |sha: &str, path: &str, findings: Vec<cleave::Finding>| cleave::FileAnalysis {
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
        report: None,
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
/// The render and the uploaded record share one source for the verdict, so
/// the only way to omit one is for there genuinely not to be one. Reading it
/// from the parent's embedded pass instead, which is bounded by the parent's
/// budget, would let a dependency that pass never reached silently lose its
/// classification line and read as unremarkable.
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

/// A root that fetched one hostile dependency, with its edge, its verdict,
/// and the registry record grafted beside it.
fn provenance_fixture() -> (
    cleave::AnalysisReport,
    Vec<fletch::fetch::FetchRecord>,
    Vec<DepResult>,
    Vec<crate::fetch::DependencyRegistry>,
) {
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
        report: None,
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
    (report, edges, deps, registries)
}

#[test]
fn interpret_context_puts_provenance_before_each_packages_traits() {
    let (report, edges, deps, registries) = provenance_fixture();

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
    let dep_provenance = ctx.get(dep_header..).unwrap().find("provenance=").unwrap() + dep_header;
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
}

#[test]
fn terminal_fetch_context_puts_provenance_before_each_packages_traits() {
    let (report, edges, deps, registries) = provenance_fixture();
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
        terminal
            .contains("\n   └─ dependency from this file · HOSTILE 97%\n      🔗  pkg:test/dep@1")
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

    // A single suspicious finding is visible even when the model grades
    // the dependency benign; notable-only packages remain omitted.
    let mut deps = deps;
    let verdict = deps[0].verdict.as_mut().unwrap();
    verdict.class = Classification::Benign;
    verdict.probability = 0.01;
    let mut report = report;
    report.files[1].findings[0].crit = cleave::Criticality::Suspicious;
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
    .expect("suspicious static dependency finding must be shown");
    assert!(terminal.contains("dependency package finding"));
    assert!(terminal.contains("pkg:test/dep@1"));
    report.files[1].findings[0].crit = cleave::Criticality::Notable;
    assert!(
        crate::engine::render_cards::render_terminal_fetch_context(
            fetched,
            &report,
            &ReportIndex::new(&report),
        )
        .is_none()
    );
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
    let now = i64::try_from(crate::civil::now_secs()).unwrap_or(i64::MAX);
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
/// A PE, a Mach-O or an Office document is never a package: with no
/// `registry.raw` in the prompt they have *only* what
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
    let claim = |value: &str, source: &str| serde_json::json!({"value": value, "source": source});

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
/// rows around a hex hit. Dropping them (`full_context: false`) to save size
/// turns a known-bad PE (`fffmpeg.exe`) from hostile to benign on the shipped
/// prompt, even though every finding description stays. Size is taken from
/// the metadata instead (see `registry_provenance`).

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
