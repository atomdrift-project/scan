//! Imported modules against declared dependencies: Python and npm.

use super::*;

#[test]
fn python_imports_keep_nearest_declared_constraints_and_shared_contexts() {
    let mut report = report_of(&serde_json::json!([
        {"id":0,"path":"bundle.zip","depth":0,"file_type":"zip","sha256":"root","size":100},
        {"id":1,"path":"bundle.zip!!a/requirements.txt","depth":1,"file_type":"requirements","sha256":"manifest-a","size":20},
        {"id":2,"path":"bundle.zip!!a/app.py","depth":1,"file_type":"python","sha256":"code-a","size":20},
        {"id":3,"path":"bundle.zip!!b/requirements.txt","depth":1,"file_type":"requirements","sha256":"manifest-b","size":20},
        {"id":4,"path":"bundle.zip!!b/app.py","depth":1,"file_type":"python","sha256":"code-b","size":20}
    ]));
    let declaration = |purl: &str| {
        Reference::new(
            RefLocator::Purl(purl.into()),
            RefKind::Dependency,
            "requirements.txt",
            "some_pkg",
        )
    };
    let import = || {
        Reference::new(
            RefLocator::Purl("pkg:pypi/some_pkg".into()),
            RefKind::Command,
            "ast-python-import-candidate",
            "some_pkg.api",
        )
    };
    let original = vec![
        (
            "manifest-a".into(),
            vec![declaration("pkg:pypi/Some_Pkg?version_requirement=%3C2.0")],
        ),
        ("code-a".into(), vec![import()]),
        (
            "manifest-b".into(),
            vec![declaration("pkg:pypi/some-pkg@8.0")],
        ),
        ("code-b".into(), vec![import()]),
    ];
    let mut references_only = original.clone();
    assert!(reconcile_python_imports(&report, &mut references_only, false).is_empty());
    assert_eq!(
        locator(&references_only[1].1[0]),
        "pkg:pypi/Some_Pkg?version_requirement=%3C2.0"
    );
    assert_eq!(locator(&references_only[3].1[0]), "pkg:pypi/some-pkg@8.0");
    let mut all = original.clone();
    assert_eq!(reconcile_python_imports(&report, &mut all, true).len(), 2);
    assert!(all[1].1.is_empty() && all[3].1.is_empty());

    // The same source bytes can belong to both environments. Follow their
    // two declared possibilities rather than selecting unconstrained latest.
    report.files[4].sha256 = "code-a".into();
    let mut shared = original.clone();
    shared[3].0 = "code-a".into();
    reconcile_python_imports(&report, &mut shared, false);
    assert_eq!(shared[1].1.len(), 2);
    assert!(
        !shared[1]
            .1
            .iter()
            .any(|r| locator(r) == "pkg:pypi/some_pkg")
    );

    // A nested package boundary does not borrow its parent's requirement.
    report.files[2].path = "bundle.zip!!a/nested/app.py".into();
    report.files[4].sha256 = "code-b".into();
    report.files.push(serde_json::from_value(serde_json::json!({"id":5,"path":"bundle.zip!!a/nested/pyproject.toml","depth":1,"file_type":"toml","sha256":"empty-manifest","size":2})).unwrap());
    let mut nested = original;
    reconcile_python_imports(&report, &mut nested, false);
    assert_eq!(locator(&nested[1].1[0]), "pkg:pypi/some_pkg");
}

#[test]
fn python_wheel_imports_use_declared_scope_and_never_a_sibling_pin() {
    let report = report_of(&serde_json::json!([
        {"id":0,"path":"owner.whl","depth":0,"file_type":"whl","sha256":"wheel","size":100},
        {"id":1,"path":"owner.whl!!pkg/app.py","depth":1,"file_type":"python","sha256":"code","size":20},
        {"id":2,"path":"sibling.whl","depth":0,"file_type":"whl","sha256":"sibling","size":100}
    ]));
    let import = || {
        Reference::new(
            RefLocator::Purl("pkg:pypi/provider".into()),
            RefKind::Command,
            "ast-python-import-candidate",
            "provider.api",
        )
    };
    let mut declaration = Reference::new(
        RefLocator::Purl("pkg:pypi/provider@2.0".into()),
        RefKind::Dependency,
        "wheel",
        "provider",
    );
    declaration.context = Some(filefacts::DependencyContext {
        scope: filefacts::DependencyScope::Development,
        optional: false,
        has_install_script: false,
        installed_path: None,
    });
    let original = vec![
        ("wheel".into(), vec![declaration]),
        ("code".into(), vec![import()]),
        (
            "sibling".into(),
            vec![Reference::new(
                RefLocator::Purl("pkg:pypi/provider@9.0".into()),
                RefKind::Dependency,
                "wheel",
                "provider",
            )],
        ),
    ];
    let mut references_only = original.clone();
    reconcile_python_imports(&report, &mut references_only, false);
    assert_eq!(locator(&references_only[1].1[0]), "pkg:pypi/provider@2.0");
    assert_eq!(
        references_only[1].1[0].context.as_ref().unwrap().scope,
        filefacts::DependencyScope::Development
    );
    let mut all = original;
    assert_eq!(reconcile_python_imports(&report, &mut all, true).len(), 1);
    assert!(
        all[1].1.is_empty(),
        "excluded development dependencies must not be reintroduced as unconstrained runtime imports"
    );

    let mut unrelated = vec![
        ("code".into(), vec![import()]),
        references_only.pop().unwrap(),
    ];
    reconcile_python_imports(&report, &mut unrelated, false);
    assert_eq!(locator(&unrelated[0].1[0]), "pkg:pypi/provider");
}

#[test]
fn ruby_process_downloads_are_discovered_from_retained_and_fetched_member_facts() {
    let source = b"if File.exist?('/tmp/marker')\n system \"curl -fsSL https://ruby.fixturetests.net/b/endpoint >/dev/null 2>&1\"\nend\n";
    let parsed = filefacts::OpenOptions::new()
        .path(Path::new("formula.rb"))
        .open(source);
    let sub = report_of(&serde_json::json!([
        {"id":0,"path":"bundle.zip","depth":0,"file_type":"zip","sha256":"c".repeat(64),"size":300},
        {"id":1,"parent_id":0,"path":"bundle.zip!!formula.rb","depth":1,"file_type":"ruby","sha256":"b".repeat(64),"size":source.len(),"filefacts":{"symbols":parsed.symbols()}}
    ]));
    let retained: AnalysisReport =
        serde_json::from_slice(&serde_json::to_vec(&sub).unwrap()).unwrap();
    let refs =
        collect_references_from_input(&retained, Path::new("not-on-disk"), None, CiRefs::Skip);
    let found: Vec<_> = refs
        .iter()
        .flat_map(|(_, refs)| refs)
        .filter(|r| r.source == "ast-fetch-call")
        .collect();
    assert_eq!(found.len(), 1);
    assert_eq!(
        locator(found[0]),
        "https://ruby.fixturetests.net/b/endpoint"
    );

    let mut importer = report_of(&serde_json::json!([
        {"id":0,"path":"entry.sh","depth":0,"file_type":"shell","sha256":"a".repeat(64),"size":30}
    ]));
    let mut record = FetchRecord::terminal(
        "https://ruby.fixturetests.net/archive.zip".into(),
        Outcome::Ok,
    );
    record.source_sha256 = Some("a".repeat(64));
    record.content_sha256 = Some("c".repeat(64));
    let mut graft = Graft::new(&importer);
    let next = merge_payload(
        &mut importer,
        &mut graft,
        &record,
        Analyzed {
            sub: Some(retained),
            content_sha: "c".repeat(64),
            next_from_bytes: Vec::new(),
            corpus: None,
        },
    );
    let found: Vec<_> = next
        .iter()
        .flat_map(|(sha, refs)| {
            refs.iter()
                .filter(|r| r.source == "ast-fetch-call")
                .map(move |r| (sha, r))
        })
        .collect();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].0, &"b".repeat(64));
    assert_eq!(
        locator(found[0].1),
        "https://ruby.fixturetests.net/b/endpoint"
    );
}

#[test]
fn ruby_dynamic_downloader_arguments_produce_a_visible_gap_without_a_guessed_url() {
    let source = b"system('curl', options, 'https://ruby.fixturetests.net/after-unknown')\n";
    let parsed = filefacts::OpenOptions::new()
        .path(Path::new("runtime.rb"))
        .open(source);
    let report = report_of(&serde_json::json!([
        {"id":0,"path":"runtime.rb","depth":0,"file_type":"ruby","sha256":"a".repeat(64),"size":source.len(),"filefacts":{"symbols":parsed.symbols()}}
    ]));
    let groups =
        collect_references_from_input(&report, Path::new("not-on-disk"), None, CiRefs::Skip);
    assert!(
        groups
            .iter()
            .all(|(_, refs)| refs.iter().all(|r| r.source != "ast-fetch-call"))
    );
    let records =
        unresolved_downloader_records(&report, &FetchPolicy::parse_follow("references").unwrap());
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].source_sha256.as_deref(),
        Some("a".repeat(64).as_str())
    );
    assert!(records[0].content_sha256.is_none());
    assert!(terminal_fetch_row_visible(&records[0]));
    assert_eq!(fetch_row(&records[0]).label, "gap");
    assert!(!records[0].locator.contains("after-unknown"));
}

#[test]
fn python_stubs_are_not_followed_from_retained_facts() {
    let source = b"from scapy.tools.UTscapy import main\n";
    let parsed = filefacts::OpenOptions::new()
        .path(Path::new("types.py"))
        .open(source);
    let report = report_of(&serde_json::json!([
        {"id":0,"path":"bundle.zip","depth":0,"file_type":"zip","sha256":"root","size":100},
        {"id":1,"path":"bundle.zip!!types.pyi","depth":1,"file_type":"python","sha256":"stub","size":source.len(),"filefacts":{"symbols":parsed.symbols()}}
    ]));
    let groups =
        collect_references_from_input(&report, Path::new("not-on-disk"), None, CiRefs::Skip);
    assert!(
        !groups
            .iter()
            .flat_map(|(_, refs)| refs)
            .any(is_python_import_candidate)
    );
}

#[test]
fn python_static_imports_reach_the_worklist_without_original_bytes() {
    let source =
        b"from scapy.tools.UTscapy import main as run\nfrom .helpers import client\nimport sys\n";
    let parsed = filefacts::OpenOptions::new()
        .path(Path::new("entry.py"))
        .open(source);
    let report = report_of(&serde_json::json!([
        {"id":0,"path":"bundle.zip","depth":0,"file_type":"zip","sha256":"a".repeat(64),"size":500},
        {"id":1,"path":"bundle.zip!!entry.py","depth":1,"file_type":"python","sha256":"b".repeat(64),"size":source.len(),"filefacts":{"symbols":parsed.symbols()}}
    ]));
    let groups =
        collect_references_from_input(&report, Path::new("not-on-disk"), None, CiRefs::Skip);
    let refs: Vec<_> = groups
        .iter()
        .flat_map(|(_, refs)| refs)
        .filter(|r| is_python_import_candidate(r))
        .collect();
    assert_eq!(refs.len(), 1);
    assert_eq!(locator(refs[0]), "pkg:pypi/scapy");
    assert_eq!(refs[0].evidence, "scapy.tools.UTscapy");
}

#[test]
fn supplied_python_namespaces_respect_archive_and_runtime_boundaries() {
    let report = report_of(&serde_json::json!([
        {"id":0,"path":"bundle.zip","depth":0,"file_type":"zip","sha256":"root","size":100},
        {"id":1,"path":"bundle.zip!!scapy/__init__.py","depth":1,"file_type":"python","sha256":"local","size":20},
        {"id":2,"path":"bundle.zip!!nested.whl!!requests/__init__.py","depth":2,"file_type":"python","sha256":"nested","size":20},
        {"id":3,"path":"bundle.zip!!only_types/__init__.pyi","depth":1,"file_type":"python","sha256":"stub","size":20},
        {"id":4,"path":"bundle.zip!!native_module.abi3.so","depth":1,"file_type":"elf","sha256":"native","size":20}
    ]));
    let local = PythonNamespaces::from_report(&report);
    assert!(local.resolves("bundle.zip!!entry.py", "scapy.tools.UTscapy"));
    assert!(local.resolves("bundle.zip!!entry.py", "native_module"));
    assert!(!local.resolves("bundle.zip!!entry.py", "requests"));
    assert!(!local.resolves("bundle.zip!!entry.py", "only_types"));
    assert!(local.resolves("bundle.zip!!nested.whl!!entry.py", "requests"));
}

#[test]
fn python_candidate_provider_requires_runtime_namespace_and_keeps_uncertainty() {
    let reference = Reference::new(
        RefLocator::Purl("pkg:pypi/scapy".into()),
        RefKind::Command,
        "ast-python-import-candidate",
        "scapy.tools.UTscapy",
    );
    let mut report = report_of(&serde_json::json!([
        {"id":0,"path":"candidate.whl","depth":0,"file_type":"whl","sha256":"root","size":100},
        {"id":1,"path":"candidate.whl!!other/__init__.py","depth":1,"file_type":"python","sha256":"wrong","size":20}
    ]));
    let mut record = FetchRecord::terminal("pkg:pypi/scapy@1.0".into(), Outcome::Ok);
    record.content_sha256 = Some("candidate-hash".into());
    assert!(!annotate_python_import_candidate(
        &mut record,
        &reference,
        Some(&report),
        None
    ));
    assert!(analysis_unavailable(&record));
    assert!(terminal_fetch_row_visible(&record));
    assert_eq!(fetch_row(&record).label, "gap");
    assert_eq!(
        record.outcome,
        Outcome::Ok,
        "retrieval success is distinct from provider verification"
    );
    report.files[1].path = "candidate.whl!!scapy/__init__.pyi".into();
    assert!(!python_candidate_provides_namespace(&reference, &report));
    report.files[1].path = "candidate.whl!!scapy/__init__.py".into();
    record.coverage = vec![Coverage::Incomplete];
    assert!(annotate_python_import_candidate(
        &mut record,
        &reference,
        Some(&report),
        None
    ));
    assert_eq!(
        record.coverage,
        [
            Coverage::Incomplete,
            Coverage::Candidate {
                source: CandidateSource::PythonImport
            }
        ]
    );
    assert!(coverage_text(&record).contains("installed distribution/version unknown"));
    report.files[0].path = "candidate.tar.gz".into();
    report.files[0].file_type = "tar".into();
    assert!(
        !python_candidate_provides_namespace(&reference, &report),
        "sdist contents do not establish an installed module layout"
    );
}

#[test]
fn npm_self_and_framework_imports_use_nearest_manifest() {
    let mut contexts = HashMap::new();
    contexts.insert(
        "bundle!!package".to_string(),
        ("ember-velcro".to_string(), true),
    );
    contexts.insert(
        "bundle!!package/nested".to_string(),
        ("nested".to_string(), false),
    );
    let path = "bundle!!package/dist/index.js";
    assert!(npm_import_is_package_or_framework_provided(
        path,
        "ember-velcro",
        &contexts
    ));
    assert!(npm_import_is_package_or_framework_provided(
        path,
        "@ember/debug",
        &contexts
    ));
    assert!(npm_import_is_package_or_framework_provided(
        path,
        "@glimmer/tracking",
        &contexts
    ));
    assert!(!npm_import_is_package_or_framework_provided(
        path,
        "@glimmer/component",
        &contexts
    ));
    assert!(!npm_import_is_package_or_framework_provided(
        "bundle!!package/nested/index.js",
        "@ember/debug",
        &contexts
    ));
    assert!(!npm_import_is_package_or_framework_provided(
        "other!!package/index.js",
        "@ember/debug",
        &contexts
    ));
}

#[test]
fn imports_use_only_nearest_manifest_declarations() {
    let sha = |c: char| c.to_string().repeat(64);
    let report: AnalysisReport = serde_json::from_value(serde_json::json!({"version":"3","files":[
            {"id":0,"path":"root.tgz","depth":0,"file_type":"npm","sha256":sha('a'),"size":1},
            {"id":1,"path":"root.tgz!!package/package.json","depth":1,"file_type":"package.json","sha256":sha('b'),"size":1},
            {"id":2,"path":"root.tgz!!package/index.js","depth":1,"file_type":"javascript","sha256":sha('c'),"size":1},
            {"id":3,"path":"root.tgz!!package/nested/package.json","depth":1,"file_type":"package.json","sha256":sha('d'),"size":1},
            {"id":4,"path":"root.tgz!!package/nested/index.js","depth":1,"file_type":"javascript","sha256":sha('e'),"size":1}
        ]})).unwrap();
    let dep = Reference::new(
        RefLocator::Purl("pkg:npm/normalize-url?version_requirement=%5E3".into()),
        RefKind::Dependency,
        "package.json",
        "normalize-url",
    );
    let groups = vec![(sha('b'), vec![dep])];
    let result = declared_npm_imports_in_hop(&report, &groups);
    assert!(result[&sha('c')].contains_key("normalize-url"));
    assert!(result[&sha('e')].is_empty());
    let mut ambiguous = report.clone();
    ambiguous.files[4].sha256 = sha('c');
    assert!(declared_npm_imports_in_hop(&ambiguous, &groups)[&sha('c')].is_empty());
}

#[test]
fn static_es_module_references_keep_locations_and_skip_builtins() {
    let values = serde_json::json!({"source":{"js":{"runtime_imports":[
        {"specifier":"axios", "offset":12},
        {"specifier":"@scope/dep/subpath", "offset":42},
        {"specifier":"node:fs", "offset":70},
        {"specifier":"./local.js", "offset":90}
    ]}}});
    let refs = static_es_module_refs("root!!package/dist/index.js", "javascript", &values);
    assert_eq!(
        refs.iter().map(locator).collect::<Vec<_>>(),
        ["pkg:npm/axios", "pkg:npm/%40scope/dep"]
    );
    assert_eq!(refs[0].offset, Some(12));
    assert!(refs.iter().all(|r| r.source == "ast-import"));
    assert!(static_es_module_refs("root!!package/index.d.ts", "typescript", &values).is_empty());
}

#[test]
fn npm_undeclared_imports_stay_inside_the_package_and_ignore_self() {
    let sha = |c: char| c.to_string().repeat(64);
    let report: AnalysisReport = serde_json::from_value(serde_json::json!({"version":"3","files":[
            {"id":0,"path":"root.tgz","depth":0,"file_type":"npm","sha256":sha('a'),"size":100},
            {"id":1,"path":"root.tgz!!package/package.json","depth":1,"file_type":"package.json","sha256":sha('b'),"size":100,
             "identity":{"name":{"value":"my-package","source":"npm.name","verified":false},"trust":"unsigned"}},
            {"id":2,"path":"root.tgz!!package/dist/index.js","depth":1,"file_type":"javascript","sha256":sha('c'),"size":100},
            {"id":3,"path":"root.tgz!!package/test/index.js","depth":1,"file_type":"javascript","sha256":sha('d'),"size":100},
            {"id":4,"path":"root.tgz!!package/node_modules/other/index.js","depth":1,"file_type":"javascript","sha256":sha('e'),"size":100}
        ]})).unwrap();
    let reference = |kind, purl: &str, source: &str| {
        Reference::new(RefLocator::Purl(purl.into()), kind, source, purl)
    };
    let groups = vec![
        (
            sha('b'),
            vec![
                reference(
                    RefKind::Dependency,
                    "pkg:npm/declared@1.0.0",
                    "package.json",
                ),
                reference(
                    RefKind::Dependency,
                    "pkg:npm/peer@1.0.0",
                    "package.json:peerDependencies",
                ),
            ],
        ),
        (
            sha('c'),
            vec![
                reference(RefKind::Command, "pkg:npm/axios", "ast-call"),
                reference(RefKind::Command, "pkg:npm/declared", "ast-call"),
                reference(RefKind::Command, "pkg:npm/peer", "ast-call"),
                reference(RefKind::Command, "pkg:npm/my-package", "ast-call"),
            ],
        ),
        (
            sha('d'),
            vec![reference(RefKind::Command, "pkg:npm/test-only", "ast-call")],
        ),
        (
            sha('e'),
            vec![reference(
                RefKind::Command,
                "pkg:npm/nested-only",
                "ast-call",
            )],
        ),
    ];
    let found = undeclared_npm_imports(&report, &groups);
    assert_eq!(found.len(), 1);
    assert_eq!(
        found[&sha('c')].iter().map(locator).collect::<Vec<_>>(),
        ["pkg:npm/axios"]
    );
}

#[test]
fn member_require_of_undeclared_package_is_flagged() {
    // A package.json declaring `mobx`, and an archive member index.js whose
    // retained AST symbols `require("mobx")` (declared) and
    // `require("db-dx-connector")` (covert). The facts-only import hunt runs
    // on the member without its bytes; the diff flags only the undeclared one.
    let report: AnalysisReport = serde_json::from_value(serde_json::json!({
        "version": "3",
        "files": [
            { "id": 0, "path": "package/package.json", "depth": 1,
              "file_type": "package_json", "sha256": "cd".repeat(32), "size": 120u64,
              "filefacts": { "references": [{
                  "locator": {"purl": "pkg:npm/mobx@^6.0.0"}, "kind": "dependency",
                  "source": "package.json", "evidence": "mobx", "offset": 0 }] } },
            { "id": 1, "path": "package/dist/index.js", "depth": 1,
              "file_type": "javascript", "sha256": "ef".repeat(32), "size": 300u64,
              "filefacts": { "symbols": [
                  {"kind": "call", "target": "require",
                   "args": [{"shape": "string", "value": "mobx"}]},
                  {"kind": "call", "target": "require",
                   "args": [{"shape": "string", "value": "db-dx-connector"}]}
              ] } }
        ]
    }))
    .expect("report deserializes");

    // No on-disk root text hunt — a missing path just skips it.
    let groups = collect_references(&report, std::path::Path::new("/nonexistent"), CiRefs::Skip);
    let all: Vec<Reference> = groups.iter().flat_map(|(_, r)| r.iter().cloned()).collect();
    let undeclared: Vec<String> = find::undeclared_packages(&all)
        .iter()
        .map(|r| locator(r).to_owned())
        .collect();
    assert!(
        undeclared.contains(&"pkg:npm/db-dx-connector".to_string()),
        "covert member require should be flagged undeclared: {undeclared:?}"
    );
    assert!(
        !undeclared.iter().any(|u| u.contains("mobx")),
        "declared dep must not be flagged: {undeclared:?}"
    );
}

#[test]
fn member_import_does_not_refetch_a_locally_resolvable_npm_package() {
    let report: AnalysisReport = serde_json::from_value(serde_json::json!({
        "version": "3",
        "files": [
            { "id": 0, "path": "bundle/node_modules/express/package.json", "depth": 1,
              "file_type": "package.json", "sha256": "a".repeat(64), "size": 80u64 },
            { "id": 1, "path": "bundle/node_modules/@scope/tool/package.json", "depth": 1,
              "file_type": "package.json", "sha256": "b".repeat(64), "size": 80u64 },
            { "id": 2, "path": "bundle/lib/index.js", "depth": 1,
              "file_type": "javascript", "sha256": "c".repeat(64), "size": 200u64,
              "filefacts": { "symbols": [
                  {"kind": "call", "target": "require",
                   "args": [{"shape": "string", "value": "express"}]},
                  {"kind": "call", "target": "require",
                   "args": [{"shape": "string", "value": "@scope/tool/subpath"}]},
                  {"kind": "call", "target": "require",
                   "args": [{"shape": "string", "value": "not-vendored"}]}
              ] } }
        ]
    }))
    .expect("report deserializes");

    let groups = collect_references(&report, std::path::Path::new("/nonexistent"), CiRefs::Skip);
    let locs: Vec<String> = groups
        .iter()
        .flat_map(|(_, refs)| refs.iter().map(|r| locator(r).to_owned()))
        .collect();
    assert_eq!(
        locs,
        vec!["pkg:npm/not-vendored"],
        "only an import absent from the ancestor node_modules is external"
    );
}

#[test]
fn sibling_node_modules_does_not_suppress_an_external_import() {
    let report: AnalysisReport = serde_json::from_value(serde_json::json!({
        "version": "3",
        "files": [
            { "id": 0, "path": "one/node_modules/express/package.json", "depth": 1,
              "file_type": "package.json", "sha256": "a".repeat(64), "size": 80u64 },
            { "id": 1, "path": "two/index.js", "depth": 1,
              "file_type": "javascript", "sha256": "b".repeat(64), "size": 200u64,
              "filefacts": { "symbols": [
                  {"kind": "call", "target": "require",
                   "args": [{"shape": "string", "value": "express"}]}
              ] } }
        ]
    }))
    .expect("report deserializes");

    let groups = collect_references(&report, std::path::Path::new("/nonexistent"), CiRefs::Skip);
    let locs: Vec<String> = groups
        .iter()
        .flat_map(|(_, refs)| refs.iter().map(|r| locator(r).to_owned()))
        .collect();
    assert_eq!(locs, vec!["pkg:npm/express"]);
}

#[test]
fn vendored_member_does_not_download_a_missing_import() {
    let report: AnalysisReport = serde_json::from_value(serde_json::json!({
        "version": "3",
        "files": [
            { "id": 0, "path": "bundle/node_modules/qs/test/parse.js", "depth": 1,
              "file_type": "javascript", "sha256": "a".repeat(64), "size": 200u64,
              "filefacts": { "symbols": [
                  {"kind": "call", "target": "require",
                   "args": [{"shape": "string", "value": "test-only-package"}]}
              ] } }
        ]
    }))
    .expect("report deserializes");

    let groups = collect_references(&report, std::path::Path::new("/nonexistent"), CiRefs::Skip);
    assert!(
        groups.is_empty(),
        "an absent import inside the captured install tree is not a network fetch"
    );
}
