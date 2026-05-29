//! Parsing and resolution for the hosted update manifests.
//!
//! Each tool publishes a small static TOML file (e.g. `litmus.toml`,
//! `cleave.toml`) under a well-known base URL. A manifest names the newest
//! released version — used for a once-a-day update notice — and carries an
//! ordered list of rules mapping an installed version line to the git ref its
//! data repository (models or traits) should track.
//!
//! ```toml
//! schema = 1
//! latest = "2.0.1"
//! url    = "https://atomdrift.org/litmus"
//!
//! [[rule]]
//! match = '^2\.0\.'   # regex tested against the installed version
//! ref   = "2.0"       # git ref (branch or commit) for this release line
//! ```
//!
//! This module is pure: it parses text and answers questions about it, with no
//! I/O. Fetching and caching live in [`crate::update_check`].

use anyhow::{Context, Result};
use semver::Version;
use serde::Deserialize;

/// A hosted update manifest, one per tool.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    /// Schema version; bumped on incompatible layout changes.
    pub schema: u32,
    /// Newest released version, compared against the installed version for the
    /// update notice.
    pub latest: String,
    /// Optional URL shown in the notice (download page or release notes).
    #[serde(default)]
    pub url: Option<String>,
    /// Ref-selection rules, evaluated in order; the first match wins.
    #[serde(default)]
    pub rule: Vec<Rule>,
}

/// One ref-selection rule within a [`Manifest`].
#[derive(Debug, Clone, Deserialize)]
pub struct Rule {
    /// Regular expression tested against the installed version string.
    #[serde(rename = "match")]
    pub pattern: String,
    /// Git ref (branch or commit) the data repo should track for this line.
    #[serde(rename = "ref")]
    pub git_ref: String,
    /// Marks this release line end-of-life (informational).
    #[serde(default)]
    pub eol: bool,
}

impl Manifest {
    /// Parse a manifest from TOML text.
    pub fn parse(text: &str) -> Result<Self> {
        toml::from_str(text).context("parsing update manifest")
    }

    /// Return the first rule whose `match` regex matches `installed`.
    ///
    /// Rules with an invalid regex are skipped (and logged), so one malformed
    /// entry never breaks resolution of the others.
    #[must_use]
    pub fn resolve(&self, installed: &str) -> Option<&Rule> {
        self.rule.iter().find(|rule| match regex::Regex::new(&rule.pattern) {
            Ok(re) => re.is_match(installed),
            Err(e) => {
                tracing::debug!(pattern = %rule.pattern, error = %e, "skipping invalid manifest rule");
                false
            }
        })
    }
}

/// True when `latest` is a strictly newer semver than `installed`.
///
/// Returns `false` (the fail-safe, keep-quiet answer) when either version
/// string is not valid semver — a malformed manifest must never produce a
/// spurious "update available" notice.
#[must_use]
pub fn is_newer(latest: &str, installed: &str) -> bool {
    match (Version::parse(latest), Version::parse(installed)) {
        (Ok(latest), Ok(installed)) => latest > installed,
        _ => false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
schema = 1
latest = "2.0.1"
url = "https://atomdrift.org/litmus"

[[rule]]
match = '^2\.0\.'
ref = "2.0"

[[rule]]
match = '^1\.'
ref = "v1"
eol = true
"#;

    #[test]
    fn parses_all_fields() {
        let m = Manifest::parse(SAMPLE).unwrap();
        assert_eq!(m.schema, 1);
        assert_eq!(m.latest, "2.0.1");
        assert_eq!(m.url.as_deref(), Some("https://atomdrift.org/litmus"));
        assert_eq!(m.rule.len(), 2);
        assert!(m.rule[1].eol);
    }

    #[test]
    fn resolve_first_match_wins() {
        let m = Manifest::parse(SAMPLE).unwrap();
        assert_eq!(m.resolve("2.0.0-rc.3").unwrap().git_ref, "2.0");
        assert_eq!(m.resolve("1.4.2").unwrap().git_ref, "v1");
        assert!(m.resolve("3.0.0").is_none());
    }

    #[test]
    fn invalid_regex_is_skipped_not_fatal() {
        let m = Manifest::parse(
            "schema = 1\nlatest = \"1.0.0\"\n[[rule]]\nmatch = \"(\"\nref = \"bad\"\n[[rule]]\nmatch = \"^1\\\\.\"\nref = \"good\"\n",
        )
        .unwrap();
        assert_eq!(m.resolve("1.2.3").unwrap().git_ref, "good");
    }

    #[test]
    fn rules_default_to_empty() {
        let m = Manifest::parse("schema = 1\nlatest = \"9.9.9\"\n").unwrap();
        assert!(m.rule.is_empty());
        assert!(m.url.is_none());
    }

    #[test]
    fn malformed_toml_errors() {
        assert!(Manifest::parse("this is = = not toml").is_err());
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
