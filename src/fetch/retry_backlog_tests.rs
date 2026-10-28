use super::*;
#[test]
fn budget_transient_and_partial_work_remain_pending_but_permanent_refusals_do_not() {
    static RESOURCES: OnceLock<Resources> = OnceLock::new();
    let resources = RESOURCES.get_or_init(|| Resources {
        net: Box::new(super::tests_offline::Offline),
        cache: super::tests_offline::cache(),
    });
    for (outcome, expected) in [
        (Outcome::BudgetExceeded, true),
        (Outcome::Failed(FetchError::Timeout), true),
        (Outcome::Failed(FetchError::Transport("DNS".into())), true),
        (Outcome::Failed(FetchError::Status(503)), true),
        (
            Outcome::Failed(FetchError::Refused("private".into())),
            false,
        ),
        (Outcome::Failed(FetchError::Status(404)), false),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(pending::Store::open(&dir.path().join("queue")).unwrap());
        let mut report: AnalysisReport = serde_json::from_value(serde_json::json!({"version":"3","files":[{"id":0,"path":"sample","file_type":"javascript","sha256":"root","size":1,"depth":0}]})).unwrap();
        let r = Reference::new(
            RefLocator::Purl("pkg:npm/pkg@1.0.0".into()),
            RefKind::Dependency,
            "lock",
            "pkg",
        );
        let entry = pending::Entry {
            root_sha: "root".into(),
            source_sha: "source".into(),
            hop: 2,
            reference: r.clone(),
            reason: "in progress".into(),
            redirect_credit: 0,
        };
        store.update(&[entry], &[]).unwrap();
        let mut session = FetchSession::new(
            &report,
            &[],
            resources,
            Follow::quiet(FetchPolicy {
                deps: true,
                ..FetchPolicy::default()
            }),
        );
        session.pending = Some(store.clone());
        session.current_hop = 2;
        let mut record = FetchRecord::terminal("pkg:npm/pkg@1.0.0".into(), outcome);
        record.source_sha256 = Some("source".into());
        session.merge_group(
            &mut report,
            GroupFetch {
                source_sha: "source".into(),
                registries: vec![],
                landed: vec![Landed {
                    reference: r,
                    record,
                    standing: Standing::Analyze,
                }],
            },
            GroupAnalysis {
                registries: vec![],
                payloads: vec![None],
            },
            &mut vec![],
        );
        assert_eq!(!store.entries("root").unwrap().is_empty(), expected);
    }
}
#[test]
fn all_versions_audit_keeps_an_older_pinned_release() {
    static RESOURCES: OnceLock<Resources> = OnceLock::new();
    let resources = RESOURCES.get_or_init(|| Resources {
        net: Box::new(super::tests_offline::Offline),
        cache: super::tests_offline::cache(),
    });
    let report: AnalysisReport =
        serde_json::from_value(serde_json::json!({"version":"3","files":[]})).unwrap();
    let r = Reference::new(
        RefLocator::Purl("pkg:npm/pkg@1.0.0".into()),
        RefKind::Dependency,
        "lock",
        "pkg",
    );
    let mut session = FetchSession::new(
        &report,
        &[],
        resources,
        Follow::quiet(FetchPolicy {
            deps: true,
            ..FetchPolicy::default()
        }),
    );
    session.newest.insert("pkg:npm/pkg".into(), "2.0.0".into());
    let mut skipped = 0;
    assert!(!session.newest_version(&r, &mut skipped));
    assert_eq!(skipped, 1);
    session.policy.all_versions = true;
    assert!(session.newest_version(&r, &mut skipped));
    assert_eq!(skipped, 1);
}
