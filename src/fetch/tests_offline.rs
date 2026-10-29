use super::{BlobCache, Fetch, FetchError, Fetched, Request};

/// Refuses every request, so a test that would fetch says so instead of
/// depending on what the internet answers today.
pub(super) struct Offline;

impl Fetch for Offline {
    fn send(&self, request: &Request<'_>) -> Result<Fetched, FetchError> {
        Err(FetchError::Refused(format!(
            "offline test: {}",
            request.url
        )))
    }
}

/// Answers every request with the same error.
pub(super) struct Failing(pub(super) FetchError);

impl Fetch for Failing {
    fn send(&self, _request: &Request<'_>) -> Result<Fetched, FetchError> {
        Err(self.0.clone())
    }
}

/// A blob cache in a directory that lives as long as the test process.
pub(super) fn cache() -> BlobCache {
    let dir = tempfile::tempdir().expect("tempdir").keep();
    BlobCache::with_dir(dir)
}
