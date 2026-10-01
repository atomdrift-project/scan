//! The hosted release manifest behind the update notice.
//!
//! Each tool publishes a small static TOML file (e.g. `litmus.toml`) under a
//! well-known base URL naming its newest release:
//!
//! ```toml
//! latest = "2.0.1"
//! url    = "https://atomdrift.org/litmus"
//! ```
//!
//! This module is pure: it parses text and answers questions about it, with no
//! I/O. Fetching and caching live in [`crate::update_check`].

use anyhow::{Context, Result};
use semver::Version;
use serde::Deserialize;

/// A hosted update manifest, one per tool.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Manifest {
    /// Newest released version, compared against the installed version for the
    /// update notice.
    pub latest: String,
    /// Optional URL shown in the notice (download page or release notes).
    #[serde(default)]
    pub url: Option<String>,
}

impl Manifest {
    /// Parse a manifest from TOML text.
    ///
    /// # Errors
    /// Returns an error if `text` is not a valid manifest.
    pub(crate) fn parse(text: &str) -> Result<Self> {
        toml::from_str(text).context("parsing update manifest")
    }
}

/// True when `latest` is a strictly newer semver than `installed`.
///
/// Returns `false` (the fail-safe, keep-quiet answer) when either version
/// string is not valid semver — a malformed manifest must never produce a
/// spurious "update available" notice.
#[must_use]
pub(crate) fn is_newer(latest: &str, installed: &str) -> bool {
    match (Version::parse(latest), Version::parse(installed)) {
        (Ok(latest), Ok(installed)) => latest > installed,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_fields() {
        // `schema` and `[[rule]]` are what manifests from the git-ref era still
        // carry; they must keep parsing.
        let m = Manifest::parse(
            "schema = 1\nlatest = \"2.0.1\"\nurl = \"https://atomdrift.org/litmus\"\n[[rule]]\nmatch = '^2'\nref = \"2.0\"\n",
        )
        .unwrap();
        assert_eq!(m.latest, "2.0.1");
        assert_eq!(m.url.as_deref(), Some("https://atomdrift.org/litmus"));
    }

    #[test]
    fn url_is_optional() {
        let m = Manifest::parse("latest = \"9.9.9\"\n").unwrap();
        assert!(m.url.is_none());
    }

    #[test]
    fn malformed_toml_errors() {
        assert!(Manifest::parse("this is = = not toml").is_err());
        assert!(Manifest::parse("url = \"no latest\"").is_err());
    }

    #[test]
    fn is_newer_handles_prerelease_ordering() {
        // A final release outranks its own release candidate.
        assert!(is_newer("2.0.0", "2.0.0-rc.3"));
        // rc.3 is newer than rc.1.
        assert!(is_newer("2.0.0-rc.3", "2.0.0-rc.1"));
        // Same version is not newer.
        assert!(!is_newer("2.0.0", "2.0.0"));
        // Older is not newer.
        assert!(!is_newer("1.9.0", "2.0.0"));
        // Unparseable input stays quiet.
        assert!(!is_newer("not-a-version", "2.0.0"));
    }
}
