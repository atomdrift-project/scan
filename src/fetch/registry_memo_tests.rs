use super::*;

fn memoized(error: FetchError, purl: &str) -> Option<Option<Registry>> {
    let res = Resources {
        net: Box::new(super::tests_offline::Failing(error)),
        cache: super::tests_offline::cache(),
    };
    let reference = Reference::new(
        RefLocator::Purl(purl.into()),
        RefKind::Dependency,
        "package.json",
        "dependencies",
    );
    lookup_registries(std::slice::from_ref(&reference), &res, 0);
    registry_memo()
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .peek(locator(&reference))
        .map(|memoized| memoized.record.clone())
}

/// A registry that could not be reached may answer next time; one that
/// said "no such package" will not.
#[test]
fn only_a_final_answer_is_memoized() {
    assert_eq!(
        memoized(
            FetchError::Transport("connection reset".into()),
            "pkg:npm/memo-test-outage@1.0.0"
        ),
        None,
        "an outage must not be remembered"
    );
    assert_eq!(
        memoized(FetchError::Status(404), "pkg:npm/memo-test-missing@1.0.0"),
        Some(None),
        "a missing package is remembered as missing"
    );
}

/// A memo entry stands in for the registry only for its TTL; past it the
/// lookup runs again, so a long-lived server sees a version yanked or held
/// after it first looked.
#[test]
fn a_memo_entry_expires() {
    let purl = "pkg:npm/memo-test-expiry@1.0.0";
    let res = Resources {
        net: Box::new(super::tests_offline::Failing(FetchError::Status(404))),
        cache: super::tests_offline::cache(),
    };
    let reference = Reference::new(
        RefLocator::Purl(purl.into()),
        RefKind::Dependency,
        "package.json",
        "dependencies",
    );
    let looked_up_at = || {
        registry_memo()
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .peek(locator(&reference))
            .map(|memoized| memoized.at)
    };
    lookup_registries(std::slice::from_ref(&reference), &res, 1);
    assert_eq!(looked_up_at(), Some(1));
    // Within the TTL the memo answers; nothing is looked up again.
    lookup_registries(
        std::slice::from_ref(&reference),
        &res,
        1 + REGISTRY_MEMO_TTL_SECS,
    );
    assert_eq!(looked_up_at(), Some(1));
    // Past it, the lookup runs again and the entry is renewed.
    let later = 2 + REGISTRY_MEMO_TTL_SECS;
    lookup_registries(std::slice::from_ref(&reference), &res, later);
    assert_eq!(looked_up_at(), Some(later));
}
