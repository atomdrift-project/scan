use super::*;
use crate::engine::FloorArm;

fn sha(c: char) -> String {
    c.to_string().repeat(64)
}

fn decision(class: Classification, level: Level) -> Decision {
    Decision {
        class,
        probability: match class {
            Classification::Benign => 0.01,
            Classification::Suspicious => 0.5,
            Classification::Hostile => 0.99,
        },
        threshold: 0.65,
        level,
    }
}

fn benign() -> Decision {
    decision(Classification::Benign, Level::Clean)
}

fn floor(class: Classification, model: Decision) -> FloorDecision {
    FloorDecision {
        class,
        confidence: 0.93,
        level: Level::At(12),
        arm: FloorArm::Crit5,
        hostile: 2,
        suspicious: 1,
        families: 3,
        model,
    }
}

/// A member evaluation: the model's `verdict`, raised by `floor` when given.
fn member(id: u64, verdict: Decision, floor: Option<FloorDecision>) -> EmbeddedFile {
    let raised = floor.map_or(verdict, |f| Decision {
        class: f.class,
        probability: f.confidence,
        threshold: verdict.threshold,
        level: f.level,
    });
    EmbeddedFile {
        id,
        sha256: String::new(),
        path: String::new(),
        file_type: "javascript".to_string(),
        classification: raised.class,
        probability: raised.probability,
        threshold: raised.threshold,
        level: raised.level,
        model_scores: Vec::new(),
        skipped_models: Vec::new(),
        formula: String::new(),
        top_findings: Vec::new(),
        floor,
    }
}

fn report(files: serde_json::Value) -> CompactReport {
    let mut report = serde_json::json!({"rev": "r1"});
    report["files"] = files;
    serde_json::from_value(report).expect("report")
}

/// Build, attach reports from the same report, and encode with no LLM.
fn encode(report: &CompactReport, members: &MemberEvals) -> Vec<serde_json::Value> {
    encode_with(report, members, &[], None)
}

fn encode_with(
    report: &CompactReport,
    members: &MemberEvals,
    records: &[FetchRecord],
    llm: Option<&Interpretation>,
) -> Vec<serde_json::Value> {
    lines(&build(report, members, records), llm)
}

/// The document for `report`, keeping the members retention would.
fn build(report: &CompactReport, members: &MemberEvals, records: &[FetchRecord]) -> UploadDoc {
    let kept = crate::engine::retention::kept_members(report, |id| {
        members.get(&u64::from(id)).map(|m| m.classification)
    });
    build_kept(report, members, &kept, records)
}

fn build_kept(
    report: &CompactReport,
    members: &MemberEvals,
    kept: &HashSet<u32>,
    records: &[FetchRecord],
) -> UploadDoc {
    build_listed(report, members, kept, &[], records)
}

fn build_listed(
    report: &CompactReport,
    members: &MemberEvals,
    kept: &HashSet<u32>,
    listed: &[Listed],
    records: &[FetchRecord],
) -> UploadDoc {
    let mut doc = UploadDoc::new(&Inputs {
        report,
        root: OwnReading {
            verdict: benign(),
            model: benign(),
            floor: None,
        },
        root_mods: &[],
        root_skip: &[],
        members,
        kept,
        every: !listed.is_empty(),
        listed,
        records,
        level: Some(25),
    })
    .expect("a report with files yields a document");
    doc.attach_reports(report);
    doc
}

fn lines(doc: &UploadDoc, llm: Option<&Interpretation>) -> Vec<serde_json::Value> {
    let stamp = Stamp {
        at: "2026-11-12T15:00:00Z",
        model: "v20.20",
        worker: None,
        ms: None,
    };
    let body = doc
        .encode(&doc.files[0].sha256, llm, &stamp)
        .expect("encodes");
    assert_eq!(body.last(), Some(&b'\n'), "every line ends in a newline");
    body.split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_slice(l).expect("each line is JSON"))
        .collect()
}

/// A package holding a script, a nested zip with a hostile script, and a
/// fetched dependency with a registry record beneath the package.
fn package() -> CompactReport {
    report(serde_json::json!([
        {"id": 0, "path": "/tmp/pkg.tgz", "type": "npm", "sha": sha('a'), "size": 100},
        {"id": 1, "pid": 0, "depth": 1, "path": "/tmp/pkg.tgz!!package/index.js", "type": "javascript", "sha": sha('b'), "size": 10},
        {"id": 2, "pid": 0, "depth": 1, "path": "/tmp/pkg.tgz!!package/inner.zip", "type": "zip", "sha": sha('c'), "size": 20},
        {"id": 3, "pid": 2, "depth": 2, "path": "/tmp/pkg.tgz!!package/inner.zip!!evil.js", "type": "javascript", "sha": sha('d'), "size": 5,
         "traits": [{"id": "objectives/c2/backdoor::a", "crit": 5, "conf": 1.0}]},
        {"id": 4, "pid": 1, "depth": 2, "rel": "fetched", "path": "pkg:npm/left-pad@1.3.0", "type": "npm", "sha": sha('e'), "size": 7},
        {"id": 5, "pid": 4, "depth": 3, "path": "pkg:npm/left-pad@1.3.0!!index.js", "type": "javascript", "sha": sha('f'), "size": 3},
        {"id": 6, "pid": 0, "depth": 1, "rel": "registry", "role": "sidecar", "path": "/tmp/pkg.tgz!!registry", "type": "registry", "sha": sha('9'), "size": 2}
    ]))
}

fn package_members() -> MemberEvals {
    let hostile = decision(Classification::Hostile, Level::At(12));
    MemberEvals::from([
        (1, member(1, benign(), None)),
        (2, member(2, benign(), None)),
        (
            3,
            member(3, benign(), Some(floor(Classification::Hostile, benign()))),
        ),
        // The dependency's file is hostile; it must not reach the package.
        (5, member(5, hostile, None)),
    ])
}

#[test]
fn one_line_per_own_file_and_no_dependency_or_registry_record() {
    let lines = encode(&package(), &package_members());
    let header = &lines[0];
    assert_eq!(header["v"], 8);
    assert_eq!(header["files"], 4);
    assert_eq!(header["scan"]["ruleset"], "r1");
    assert_eq!(header["scan"]["level"], 25);
    assert!(header["scan"].get("worker").is_none(), "no worker name");
    let shas: Vec<&str> = lines[1..=4]
        .iter()
        .map(|l| l["sha256"].as_str().unwrap())
        .collect();
    assert_eq!(shas, [sha('a'), sha('b'), sha('c'), sha('d')]);
    for line in &lines[1..=4] {
        let sha = line["sha256"].as_str().unwrap();
        assert!(sha != self::sha('e') && sha != self::sha('f') && sha != self::sha('9'));
    }
}

#[test]
fn contents_lists_immediate_members_by_path() {
    let lines = encode(&package(), &package_members());
    assert_eq!(
        lines[1]["contents"],
        serde_json::json!({"children": [1, 2], "paths": ["package/index.js", "package/inner.zip"]}),
        "paths are relative to the container, never to the scan's own path"
    );
    assert_eq!(
        lines[3]["contents"],
        serde_json::json!({"children": [3], "paths": ["evil.js"]})
    );
    assert!(
        lines[2].get("contents").is_none(),
        "a file that holds nothing says nothing"
    );
    assert!(
        lines[2].get("refs").is_none(),
        "a member line never carries refs"
    );
    assert_eq!(
        lines[1]["refs"],
        serde_json::json!([]),
        "the artifact always does"
    );
}

#[test]
fn every_container_folds_in_what_it_holds() {
    let lines = encode(&package(), &package_members());
    let (pkg, zip, evil) = (&lines[1], &lines[3], &lines[4]);
    assert_eq!(evil["verdict"]["class"], "hostile");
    assert_eq!(evil["verdict"]["by"], "floor");
    assert!(
        evil["verdict"].get("in").is_none(),
        "its own bytes decided it"
    );
    assert_eq!(evil["floor"]["arm"], "crit5");
    for container in [zip, pkg] {
        assert_eq!(container["verdict"]["class"], "hostile");
        assert_eq!(container["verdict"]["fires_at"], 12);
        assert_eq!(container["verdict"]["by"], "floor");
        assert_eq!(
            container["verdict"]["in"],
            sha('d'),
            "the file that carried it"
        );
        assert_eq!(container["floor"]["in"], sha('d'));
        assert_eq!(
            container["ml"]["class"], "benign",
            "the model saw nothing anywhere"
        );
    }
    assert_eq!(lines[2]["verdict"]["class"], "benign");
}

#[test]
fn a_dependency_never_raises_the_artifact() {
    // Only the dependency's file is hostile.
    let mut members = package_members();
    members.insert(3, member(3, benign(), None));
    let lines = encode(&package(), &members);
    assert_eq!(lines[1]["verdict"]["class"], "benign");
    assert_eq!(lines[1]["ml"]["class"], "benign");
}

#[test]
fn a_members_line_is_the_same_whichever_archive_carried_it() {
    let inner_alone = report(serde_json::json!([
        {"id": 0, "path": "/elsewhere/other.zip", "type": "zip", "sha": sha('7'), "size": 1},
        {"id": 1, "pid": 0, "depth": 1, "path": "/elsewhere/other.zip!!lib/inner.zip", "type": "zip", "sha": sha('c'), "size": 20},
        {"id": 2, "pid": 1, "depth": 2, "path": "/elsewhere/other.zip!!lib/inner.zip!!evil.js", "type": "javascript", "sha": sha('d'), "size": 5,
         "traits": [{"id": "objectives/c2/backdoor::a", "crit": 5, "conf": 1.0}]}
    ]));
    let members = MemberEvals::from([
        (1, member(1, benign(), None)),
        (
            2,
            member(2, benign(), Some(floor(Classification::Hostile, benign()))),
        ),
    ]);
    let a = encode(&package(), &package_members());
    let b = encode(&inner_alone, &members);
    let line = |lines: &[serde_json::Value], s: &str| {
        lines
            .iter()
            .find(|l| l["sha256"] == s && l.get("v").is_none())
            .cloned()
            .unwrap()
    };
    let (a_zip, b_zip) = (line(&a, &sha('c')), line(&b, &sha('c')));
    // Child indexes are positions in each body; everything else must match.
    assert_eq!(a_zip["verdict"], b_zip["verdict"]);
    assert_eq!(a_zip["ml"], b_zip["ml"]);
    assert_eq!(a_zip["floor"], b_zip["floor"]);
    assert_eq!(a_zip["contents"]["paths"], b_zip["contents"]["paths"]);
    assert_eq!(line(&a, &sha('d')), line(&b, &sha('d')));
}

#[test]
fn reports_are_placement_free_and_cite_by_digest() {
    let mut pkg = package();
    // The package inherits the member's finding, citing it by node id; and a
    // dependency verdict is pinned on it last.
    pkg.files[0].findings = serde_json::from_value(serde_json::json!([
        {"id": "objectives/c2/backdoor::a", "crit": 5, "conf": 1.0, "from": [{"file": 3, "off": 40}]},
        {"id": "fetch/dependency-verdict", "crit": 5, "conf": 1.0,
         "dep": {"locator": "pkg:npm/left-pad@1.3.0", "sha": sha('e'), "type": "npm"}}
    ]))
    .unwrap();
    let lines = encode(&pkg, &package_members());
    let report_lines = &lines[5..];
    let named: Vec<&serde_json::Value> = lines[1..=4]
        .iter()
        .filter(|l| l.get("report").is_some())
        .collect();
    assert_eq!(
        named.len(),
        report_lines.len(),
        "one report line per file naming one"
    );
    let root_report = &report_lines[0];
    assert_eq!(
        root_report["traits"],
        serde_json::json!([{"conf": 1.0, "crit": 5, "from": [{"off": 40, "sha256": sha('d')}], "id": "objectives/c2/backdoor::a"}]),
        "cited by digest, keys sorted, the pinned dependency verdict gone"
    );
    for key in PLACEMENT_KEYS {
        assert!(
            root_report.get(key).is_none(),
            "{key} says where the file sat"
        );
    }
    let text = serde_json::to_string(root_report).unwrap();
    assert_eq!(
        lines[1]["report"],
        format!("{:x}", Sha256::digest(text.as_bytes())),
        "the file line names its report by the report line's own digest"
    );
    assert_eq!(
        lines[1]["verdict"]["top"],
        serde_json::json!([{"id": "objectives/c2/backdoor::a", "crit": 5}]),
        "a pinned dependency verdict is not one of the artifact's findings"
    );
}

#[test]
fn the_same_findings_give_the_same_report_from_any_path() {
    let at = |root: &str| {
        let doc = build(
            &report(serde_json::json!([
                {"id": 7, "path": root, "type": "javascript", "sha": sha('b'), "size": 10, "risk": 3,
                 "traits": [{"id": "x::y", "crit": 3, "conf": 0.8}]}
            ])),
            &MemberEvals::new(),
            &[],
        );
        doc.files[0].report.clone().unwrap().digest
    };
    assert_eq!(at("/tmp/a/index.js"), at("/var/b/renamed.js"));
}

#[test]
fn only_kept_members_and_their_containers_get_a_line() {
    // The quiet script is not kept; the zip is, as the way to the hostile one.
    let kept = HashSet::from([2, 3]);
    let lines = lines(
        &build_kept(&package(), &package_members(), &kept, &[]),
        None,
    );
    assert_eq!(lines[0]["files"], 3);
    assert!(
        lines.iter().all(|l| l["sha256"] != sha('b')),
        "no line, no row"
    );
    assert_eq!(
        lines[1]["contents"],
        serde_json::json!({"children": [1], "paths": ["package/inner.zip"]}),
        "contents lists the members that have a line"
    );
    let named = lines[1..=3]
        .iter()
        .filter(|l| l.get("report").is_some())
        .count();
    assert_eq!(named, 3, "every file line carries its report");
}

#[test]
fn the_cut_hides_nothing_from_a_verdict() {
    let lines = lines(
        &build_kept(&package(), &package_members(), &HashSet::new(), &[]),
        None,
    );
    assert_eq!(lines[0]["files"], 1);
    assert!(lines[1].get("contents").is_none());
    assert_eq!(lines[1]["verdict"]["class"], "hostile");
    assert_eq!(
        lines[1]["verdict"]["in"],
        sha('d'),
        "names the file that carried it, line or not"
    );
}

fn record(json: serde_json::Value) -> FetchRecord {
    serde_json::from_value(json).expect("fetch record")
}

#[test]
fn refs_carry_the_resolved_version_and_the_request() {
    let mut pkg = package();
    pkg.files[1].refs = serde_json::from_value(serde_json::json!([
        {"to": "pkg:npm/left-pad?version_requirement=%5E1.3.0", "kind": "dependency", "off": 10},
        {"to": "pkg:npm/is-even?vers=vers:npm%2F%3E%3D1.0.0", "kind": "dependency"},
        {"to": "pkg:npm/is-number@latest", "kind": "command"},
        {"to": "pkg:npm/is-odd@3.0.1", "kind": "dependency", "off": 40},
        {"to": "pkg:npm/is-odd@3.0.1", "kind": "dependency", "off": 90},
        {"to": "https://example.com/x.sh", "kind": "url_fetch", "off": 70},
        {"to": "./lib/util.js", "kind": "local"}
    ]))
    .unwrap();
    let records = [
        record(
            serde_json::json!({"source_sha256": sha('b'), "source_offset": 10, "kind": "dependency",
            "locator": "pkg:npm/left-pad@1.3.4", "content_sha256": sha('e'), "outcome": "ok"}),
        ),
        record(
            serde_json::json!({"source_sha256": sha('b'), "source_offset": 70, "kind": "url_fetch",
            "locator": "https://example.com/x.sh", "content_sha256": sha('5'), "outcome": "ok"}),
        ),
    ];
    let lines = encode_with(&package(), &package_members(), &[], None);
    assert_eq!(lines[1]["refs"], serde_json::json!([]));
    let lines = encode_with(&pkg, &package_members(), &records, None);
    assert_eq!(
        lines[1]["refs"],
        serde_json::json!([
            {"kind": "command", "name": "pkg:npm/is-number", "requested": "latest"},
            {"kind": "dependency", "name": "pkg:npm/is-even", "requested": ">=1.0.0"},
            {"kind": "dependency", "name": "pkg:npm/is-odd", "version": "3.0.1"},
            {"kind": "dependency", "name": "pkg:npm/left-pad", "version": "1.3.4", "requested": "^1.3.0"},
            {"kind": "url_fetch", "name": "https://example.com/x.sh", "sha256": sha('5')}
        ]),
        "declared anywhere in the artifact, deduplicated, on the artifact's line"
    );
}

#[test]
fn free_text_is_cut_to_fit_on_a_character_boundary() {
    let reason = "é".repeat(2000);
    let judged = Judged {
        verdict: benign(),
        by: Juror::Ml,
        verdict_in: Some(sha('d')),
        ml: benign(),
        ml_in: None,
        mods: Vec::new(),
        skip: Vec::new(),
        floor: None,
        floor_in: None,
    };
    let top: Vec<Finding> = (0..3)
        .map(|i| Finding {
            id: format!("objectives/{}::{i}", "x".repeat(150)),
            crit: 5,
        })
        .collect();
    let object = verdict_object(&judged, &top, &reason).expect("fits");
    assert!(object.get().len() <= MAX_OBJECT);
    let value: serde_json::Value = serde_json::from_str(object.get()).unwrap();
    let kept = value["reason"].as_str().unwrap_or_default();
    assert!(kept.len() <= MAX_TEXT && kept.chars().all(|c| c == 'é'));
    assert_eq!(bounded("aé", 2), "a", "never splits a character");
}

#[test]
fn an_llm_that_moves_the_verdict_decides_it() {
    let mut doc = build(&package(), &package_members(), &[]);
    doc.blend(|d| {
        d.class = Classification::Suspicious;
        d.level = Level::At(255);
        true
    });
    let graded = crate::interpret::Graded {
        grade: crate::interpret::LlmGrade::Suspicious,
        outcome: Classification::Suspicious,
        corroborated: false,
        may_soften: false,
        blended: 0.7,
        interpretation: "Fetches and runs a remote script.".to_string(),
        model: "m".to_string(),
        analyzer_directed: false,
        cached: false,
        before: crate::interpret::MlVerdict {
            class: Classification::Benign,
            prob: 0.1,
            lvl: Level::Clean,
        },
    };
    let llm = Interpretation::Graded(graded);
    let lines = lines(&doc, Some(&llm));
    let root = &lines[1];
    assert_eq!(root["verdict"]["by"], "llm");
    assert_eq!(root["verdict"]["fires_at"], 255);
    assert!(root["verdict"].get("in").is_none());
    assert_eq!(
        root["verdict"]["reason"],
        "Fetches and runs a remote script."
    );
    assert_eq!(
        root["llm"],
        serde_json::json!({"class": "suspicious", "conf": 0.7, "model": "m", "reason": "Fetches and runs a remote script."})
    );
    assert!(
        lines[2].get("llm").is_none(),
        "only the artifact carries the LLM"
    );
}

#[test]
fn a_file_inside_itself_is_not_its_own_member() {
    let looped = report(serde_json::json!([
        {"id": 0, "path": "q.zip", "type": "zip", "sha": sha('a'), "size": 1},
        {"id": 1, "pid": 0, "depth": 1, "path": "q.zip!!q.zip", "type": "zip", "sha": sha('a'), "size": 1}
    ]));
    let lines = encode(&looped, &MemberEvals::new());
    assert_eq!(lines[0]["files"], 1);
    assert!(lines[1].get("contents").is_none());
}

#[test]
fn every_path_of_a_kept_file_is_listed_up_to_the_cap() {
    let copies = MAX_MEMBERS + 44;
    let files: Vec<serde_json::Value> = std::iter::once(serde_json::json!(
        {"id": 0, "path": "z.zip", "type": "zip", "sha": sha('a'), "size": 1}
    ))
    .chain((1..=copies).map(|i| {
        serde_json::json!({"id": i, "pid": 0, "depth": 1, "path": format!("z.zip!!d{i:04}/x.js"),
                           "type": "javascript", "sha": sha('b'), "size": 1})
    }))
    .collect();
    let report = report(serde_json::Value::Array(files));
    // Only the first copy was ranked in.
    let lines = lines(
        &build_kept(&report, &MemberEvals::new(), &HashSet::from([1]), &[]),
        None,
    );
    assert_eq!(lines[0]["files"], 2, "one file, one line");
    let paths = lines[1]["contents"]["paths"].as_array().unwrap();
    assert_eq!(
        paths.len(),
        MAX_MEMBERS,
        "every copy is a place it sits, up to the cap"
    );
    assert_eq!(paths[0], "d0001/x.js");
}

#[test]
fn identity_rides_on_the_file_line_not_in_the_report() {
    let mut pkg = package();
    pkg.files[1].identity = serde_json::from_value(serde_json::json!({
        "name": {"value": "tool", "source": "pe.version.product_name", "verified": false},
        "trust": "unsigned"
    }))
    .unwrap();
    let lines = encode(&pkg, &package_members());
    assert_eq!(lines[2]["ident"]["name"]["value"], "tool");
    assert!(
        lines[1].get("ident").is_none(),
        "a file that claims nothing says nothing"
    );
    for report in &lines[5..] {
        assert!(
            report.get("ident").is_none(),
            "stored once, on the file line"
        );
    }
}

#[test]
fn identity_is_cleaned_and_shortened_not_dropped() {
    let mut pkg = package();
    let emails: Vec<String> = (0..2000).map(|i| format!("user{i}@example.com")).collect();
    pkg.files[0].identity = serde_json::from_value(serde_json::json!({
        "name": {"value": "to\u{0}ol", "source": "npm.name", "verified": false},
        "emails": emails,
        "trust": "unsigned"
    }))
    .unwrap();
    let lines = encode(&pkg, &package_members());
    let ident = &lines[1]["ident"];
    assert_eq!(
        ident["name"]["value"], "tool",
        "NULs from the file are removed"
    );
    assert!(serde_json::to_string(ident).unwrap().len() <= MAX_IDENT);
    let kept = ident["emails"].as_array().unwrap();
    assert!(
        !kept.is_empty() && kept.len() < 2000,
        "the list is cut, the block kept"
    );
    assert_eq!(kept[0], "user0@example.com", "the first entries stay");
}

#[test]
fn no_object_carries_a_nul() {
    let doc = build(&package(), &package_members(), &[]);
    let graded = crate::interpret::Graded {
        grade: crate::interpret::LlmGrade::Hostile,
        outcome: Classification::Hostile,
        corroborated: false,
        may_soften: false,
        blended: 0.9,
        interpretation: "runs\0 a shell".to_string(),
        model: "m".to_string(),
        analyzer_directed: false,
        cached: false,
        before: crate::interpret::MlVerdict {
            class: Classification::Benign,
            prob: 0.1,
            lvl: Level::Clean,
        },
    };
    let stamp = Stamp {
        at: "2026-11-12T15:00:00Z",
        model: "v20.20",
        worker: None,
        ms: None,
    };
    let body = doc
        .encode(&sha('a'), Some(&Interpretation::Graded(graded)), &stamp)
        .unwrap();
    let text = String::from_utf8(body).unwrap();
    assert!(!text.contains("\\u0000"), "{text}");
    assert!(text.contains("runs a shell"));
}

/// `--all-members`: every member cleave listed gets a line, a quiet one with
/// its identity and nothing else, under the container its path names. The
/// upload is the same with or without it.
#[test]
fn listed_members_get_lines_only_when_every_member_is_asked_for() {
    let listed = |path: &str, file_type: &str, c: char, size| Listed {
        path: path.to_string(),
        file_type: file_type.to_string(),
        sha256: sha(c),
        size,
    };
    let listed = [
        listed("package/README.md", "markdown", '1', 9),
        listed("package/inner.zip", "zip", 'c', 20),
        listed("package/inner.zip!notes.txt", "text", '2', 4),
    ];
    // Only the hostile script and the way to it are kept.
    let kept = HashSet::from([2, 3]);
    let doc = build_listed(&package(), &package_members(), &kept, &listed, &[]);
    let stamp = Stamp {
        at: "2026-11-12T15:00:00Z",
        model: "v20.20",
        worker: None,
        ms: None,
    };
    let body = |every: bool| {
        let bytes = if every {
            doc.encode_every(&sha('a'), None, &stamp)
        } else {
            doc.encode(&sha('a'), None, &stamp)
        };
        bytes
            .unwrap()
            .split(|&b| b == b'\n')
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_slice::<serde_json::Value>(l).unwrap())
            .collect::<Vec<_>>()
    };

    let upload = body(false);
    assert_eq!(
        upload[0]["files"], 3,
        "the upload names the kept files only"
    );
    assert_eq!(
        upload[1]["contents"],
        serde_json::json!({"children": [1], "paths": ["package/inner.zip"]})
    );
    let kept_only = lines(
        &build_kept(&package(), &package_members(), &kept, &[]),
        None,
    );
    assert_eq!(
        upload, kept_only,
        "the same upload with or without --all-members"
    );

    let every = body(true);
    assert_eq!(every[0]["files"], 6);
    let readme = every.iter().find(|l| l["sha256"] == sha('1')).unwrap();
    assert_eq!(
        *readme,
        serde_json::json!({"sha256": sha('1'), "size": 9, "type": "markdown"}),
        "listed and not analyzed: identity and nothing else"
    );
    assert_eq!(
        every[1]["contents"]["paths"],
        serde_json::json!(["package/README.md", "package/index.js", "package/inner.zip"])
    );
    let zip = every.iter().find(|l| l["sha256"] == sha('c')).unwrap();
    assert_eq!(
        zip["contents"]["paths"],
        serde_json::json!(["evil.js", "notes.txt"]),
        "a listed member nests under the container its path names"
    );
}

/// Whoever builds the upload around scan's (postdoc) adds the drift jurors and
/// may decide the verdict; scan's own three are not set from outside.
#[test]
fn drift_jurors_and_a_decided_verdict_reach_the_artifacts_line() {
    let mut doc = build(&package(), &package_members(), &[]);
    let before = doc.verdict().expect("scan judged the artifact");
    assert_eq!(before.class, Classification::Hostile);

    doc.set_juror(
        Juror::DriftHeuristic,
        serde_json::json!({"class": "hostile", "fires_at": 25, "base": sha('7'), "engine": "isomer 0.5.0"}),
    )
    .unwrap();
    doc.set_juror(
        Juror::DriftLlm,
        serde_json::json!({"class": "suspicious", "model": "m", "reason": "adds a post\u{0}install hook"}),
    )
    .unwrap();
    doc.decide(Classification::Hostile, Level::At(1), Juror::DriftHeuristic);

    let lines = lines(&doc, None);
    let root = &lines[1];
    assert_eq!(root["drift_heuristic"]["base"], sha('7'));
    assert_eq!(
        root["drift_llm"]["reason"], "adds a postinstall hook",
        "NULs removed"
    );
    assert_eq!(root["verdict"]["by"], "drift_heuristic");
    assert_eq!(root["verdict"]["fires_at"], 1);
    assert!(
        root["verdict"].get("in").is_none(),
        "the drift juror read the release itself"
    );
    for member in &lines[2..=4] {
        assert!(member.get("drift_heuristic").is_none() && member.get("drift_llm").is_none());
    }

    assert_eq!(
        doc.set_juror(Juror::Ml, serde_json::json!({"class": "benign"})),
        Err(OpinionError::NotSettable(Juror::Ml))
    );
    assert_eq!(
        doc.set_juror(
            Juror::DriftHeuristic,
            serde_json::json!({"class": "malicious"})
        ),
        Err(OpinionError::Class)
    );
    assert_eq!(
        doc.set_juror(
            Juror::DriftHeuristic,
            serde_json::json!({"outcome": "error", "class": "benign"})
        ),
        Err(OpinionError::Class)
    );
    assert!(
        doc.set_juror(
            Juror::DriftLlm,
            serde_json::json!({"outcome": "error", "error": "timed out"})
        )
        .is_ok()
    );
    assert!(matches!(
        doc.set_juror(
            Juror::DriftHeuristic,
            serde_json::json!({"class": "benign", "reason": "x".repeat(1000)})
        ),
        Err(OpinionError::TooLarge(_))
    ));
    assert_eq!(
        doc.set_juror(
            Juror::DriftHeuristic,
            serde_json::json!(["not", "an", "object"])
        ),
        Err(OpinionError::NotAnObject)
    );
}

/// Third-party scanners report on the artifact's line under `scanners`,
/// keyed by name, and never move the verdict.
#[test]
fn scanners_reach_the_artifacts_line_and_leave_the_verdict_alone() {
    let mut doc = build(&package(), &package_members(), &[]);
    let before = lines(&doc, None);
    assert!(
        before[1].get("scanners").is_none(),
        "no scanner ran, so none is named"
    );

    doc.set_scanner(
        Scanner::Clamav,
        serde_json::json!({"class": "hostile", "engine": "1.4.1", "db": "28044", "signature": "Js.Trojan.Agent-1"}),
    )
    .unwrap();
    doc.set_scanner(
        Scanner::Malcontent,
        serde_json::json!({"outcome": "timeout", "error": "no answer in 600s", "engine": "1.28.1"}),
    )
    .unwrap();

    let after = lines(&doc, None);
    let root = &after[1];
    assert_eq!(root["scanners"]["clamav"]["signature"], "Js.Trojan.Agent-1");
    assert_eq!(root["scanners"]["malcontent"]["outcome"], "timeout");
    assert_eq!(
        root["verdict"], before[1]["verdict"],
        "a scanner never feeds the verdict"
    );
    for member in &after[2..=4] {
        assert!(member.get("scanners").is_none());
    }

    assert_eq!(
        doc.set_scanner(
            Scanner::Malcontent,
            serde_json::json!({"class": "hostile", "report": "ab"})
        ),
        Err(OpinionError::Reserved("report"))
    );
    assert_eq!(
        doc.set_scanner(Scanner::Clamav, serde_json::json!({"outcome": "skipped"})),
        Err(OpinionError::Outcome)
    );
}
