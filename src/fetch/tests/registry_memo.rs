use super::*;

/// What the memo holds for `purl` after one lookup that failed with `error`:
/// the record slot and whether it is the brief "unavailable" kind.
fn memoized(error: FetchError, purl: &str) -> Option<(Option<Registry>, bool)> {
    let res = Resources {
        net: Box::new(super::offline::Failing(error)),
        cache: super::offline::cache(),
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
        .map(|memoized| (memoized.record.clone(), memoized.unavailable))
}

/// A registry that could not be reached may answer soon, so its miss is held
/// only briefly; one that said "no such package" is a final answer.
#[test]
fn an_outage_is_held_briefly_and_a_final_answer_for_the_ttl() {
    assert_eq!(
        memoized(
            FetchError::Transport("connection reset".into()),
            "pkg:npm/memo-test-outage@1.0.0"
        ),
        Some((None, true)),
        "an outage is remembered as unavailable"
    );
    assert_eq!(
        memoized(FetchError::Status(429), "pkg:npm/memo-test-limited@1.0.0"),
        Some((None, true)),
        "a rate limit is remembered as unavailable"
    );
    assert_eq!(
        memoized(FetchError::Status(404), "pkg:npm/memo-test-missing@1.0.0"),
        Some((None, false)),
        "a missing package is remembered as missing"
    );
}

/// A rate-limited registry is not asked again by every file naming the
/// dependency, but is asked again once the short hold passes.
#[test]
fn an_unavailable_miss_expires_quickly() {
    let purl = "pkg:npm/memo-test-unavailable-expiry@1.0.0";
    let res = Resources {
        net: Box::new(super::offline::Failing(FetchError::Status(403))),
        cache: super::offline::cache(),
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
    // Within the hold, the memo answers "unavailable" without asking.
    let found = lookup_registries(
        std::slice::from_ref(&reference),
        &res,
        1 + UNAVAILABLE_MEMO_TTL_SECS,
    );
    assert!(found[0].is_none());
    assert_eq!(looked_up_at(), Some(1));
    // Past it — long before the final-answer TTL — the registry is asked again.
    let later = 2 + UNAVAILABLE_MEMO_TTL_SECS;
    assert!(later < REGISTRY_MEMO_TTL_SECS);
    lookup_registries(std::slice::from_ref(&reference), &res, later);
    assert_eq!(looked_up_at(), Some(later));
}

/// A memo entry stands in for the registry only for its TTL; past it the
/// lookup runs again, so a long-lived server sees a version yanked or held
/// after it first looked.
#[test]
fn a_memo_entry_expires() {
    let purl = "pkg:npm/memo-test-expiry@1.0.0";
    let res = Resources {
        net: Box::new(super::offline::Failing(FetchError::Status(404))),
        cache: super::offline::cache(),
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
