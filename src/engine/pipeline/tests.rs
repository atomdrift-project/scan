use super::super::dep_envelope;
use super::*;
use cleave::types::Role;

/// Grade `raw` as `classify_report` grades a fetched dependency.
fn grade(raw: &str, label: &str, model: &Model) -> Option<(Decision, MemberEvals)> {
    let report: CompactReport = serde_json::from_str(raw).expect("dependency report parses");
    let req = ClassifyRequest::new(label, std::path::Path::new(label), model);
    classify_dependency(&report, label, &req).expect("not cancelled")
}

/// Store `raw` the way `classify_report` stores a report.
fn stored(raw: &str) -> StoredReport {
    StoredReport::new(&serde_json::from_str(raw).expect("report parses")).expect("serializes")
}

/// The former production cutoff. Tests deliberately put evidence beyond it
/// so reintroducing a prefix-only grading policy cannot pass unnoticed.
const FORMER_EMBEDDED_FILE_LIMIT: usize = 100;

/// Grading needs a real model to score against. Point SCAN_MODELS_DIR at a
/// bundle — a single model (`model.onnx`) or a routed one (`config.json`) —
/// to run these; without one there is nothing to exercise, so they skip
/// rather than fail.
fn model_bundle() -> Option<std::path::PathBuf> {
    let p = crate::env::path(crate::env::SCAN_MODELS_DIR)?;
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
    assert!(
        graded[0].report.is_none(),
        "an unparseable report has no stored form"
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

    let (clean, _) = grade(&dep_report(3, ""), "pkg:npm/a@1", &model).expect("clean report grades");
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
    let src = include_str!("../pipeline.rs");
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
    let (verdict, members) = grade(&raw, "pkg:npm/evil@1.0.0", &model).expect("dependency grades");

    let dep = DepResult {
        sha256: "d".repeat(64),
        locator: "pkg:npm/evil@1.0.0".to_string(),
        url: "https://reg.test/evil-1.0.0.tgz".to_string(),
        size: 1234,
        provenance: None,
        verdict: Some(verdict),
        members,
        report: Some(stored(&raw)),
    };
    let dep_env = dep_envelope(&dep, "model-9", "2026-06-28T00:00:00Z")
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
        cleave: Some(stored(&raw)),
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
    use crate::corpus_precheck::Verdict;
    use crate::hopper::Finding;
    let finding = |id: &str, desc: &str, crit: u8| Finding {
        id: id.to_string(),
        desc: Some(desc.to_string()),
        crit,
    };
    let verdict = |reason: Option<&str>, findings: Vec<Finding>| Verdict {
        fires_at: Level::At(2),
        reason: reason.map(String::from),
        findings,
        ..Verdict::default()
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
    let report = cleave::types::compact_from_files(&[fa], cleave::types::Facts::Full);
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
    let report = cleave::types::compact::compact_from_files(
        &[
            file(0, "a.tar", "tar", "s0", None),
            file(1, "a.tar!!x.py", "python", "s1", Some(0)),
            orphan,
        ],
        Facts::Full,
    );
    let integ = validate_report_references("test", &report);
    assert_eq!(integ.orphan_sidecars, 1, "orphan sidecar");
    assert_eq!(integ.dangling_refs, 0, "no dangling refs");
}

#[test]
fn clean_report_has_zero_integrity_counts() {
    // A sidecar that correctly names its parent is fine.
    let mut sidecar = file(1, "reg", "registry", "s1", Some(0));
    sidecar.role = Role::Sidecar;
    let report = cleave::types::compact::compact_from_files(
        &[file(0, "a.tar", "tar", "s0", None), sidecar],
        Facts::Full,
    );
    let integ = validate_report_references("test", &report);
    assert_eq!(integ, super::ReportIntegrity::default());
}
