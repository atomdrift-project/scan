use super::*;

struct Answers(FetchError);

impl Fetch for Answers {
    fn send(&self, _request: &Request<'_>) -> Result<Fetched, FetchError> {
        Err(self.0.clone())
    }
}

fn memoized(error: FetchError, purl: &str) -> Option<Option<Registry>> {
    let res = Resources {
        net: Box::new(Answers(error)),
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
        .cloned()
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
