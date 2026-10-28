use super::*;

/// Only an answer that the release does not exist lets `scan purl` judge
/// a package on its registry record alone. A size cap, a refused address
/// or an outage says nothing about the package.
#[test]
fn only_a_missing_release_is_gone() {
    let gone = |outcome| {
        Unretrievable {
            target: "https://registry.test/x".into(),
            outcome,
        }
        .gone()
    };
    assert!(gone(Outcome::Failed(FetchError::Status(404))));
    assert!(gone(Outcome::Failed(FetchError::Status(410))));
    assert!(gone(Outcome::Unresolved(
        fletch::fetch::Unresolved::NoRelease
    )));
    assert!(!gone(Outcome::Failed(FetchError::TooLarge)));
    assert!(!gone(Outcome::Failed(FetchError::Refused(
        "private address".into()
    ))));
    assert!(!gone(Outcome::Failed(FetchError::Timeout)));
    assert!(!gone(Outcome::Failed(FetchError::Status(503))));
    assert!(!gone(Outcome::BudgetExceeded));
}

struct FailingMetadata(FetchError);

impl Fetch for FailingMetadata {
    fn send(&self, _request: &Request<'_>) -> Result<Fetched, FetchError> {
        Err(self.0.clone())
    }
}

#[test]
fn metadata_outages_survive_resolution_as_infrastructure_failures() {
    let reference = Reference::new(
        RefLocator::Purl("pkg:pypi/dns-regression@1.0.0".into()),
        RefKind::Dependency,
        "test",
        "",
    );
    for error in [
        FetchError::Transport("lookup pypi.org: no such host".into()),
        FetchError::Timeout,
        FetchError::Status(429),
        FetchError::Status(503),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let cache = BlobCache::with_dir(dir.path());
        let record = fetch_one_record(&reference, &FailingMetadata(error.clone()), &cache);
        assert_eq!(record.outcome, Outcome::Failed(error));
    }
}

#[test]
fn a_metadata_404_remains_a_missing_release() {
    let dir = tempfile::tempdir().unwrap();
    let reference = Reference::new(
        RefLocator::Purl("pkg:pypi/missing-regression@1.0.0".into()),
        RefKind::Dependency,
        "test",
        "",
    );
    let record = fetch_one_record(
        &reference,
        &FailingMetadata(FetchError::Status(404)),
        &BlobCache::with_dir(dir.path()),
    );
    assert!(matches!(record.outcome, Outcome::Unresolved(_)));
}
