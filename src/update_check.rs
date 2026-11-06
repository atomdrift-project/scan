//! Release-update notifier.
//!
//! On first run and at most once every 24 hours, scan fetches a small static
//! TOML manifest (`scan.toml`) and prints a one-line notice when a newer
//! release exists. The fetch is a plain GET of a static file; the only thing it
//! says about this install is its User-Agent, `scan/<version>`. A fresh result
//! is cached locally so subsequent runs print the notice without touching the
//! network at all.
//!
//! Set `SCAN_NO_UPDATE_CHECK=1` to disable the notice.
//!
//! The manifest names the newest release:
//!
//! ```toml
//! latest = "2.0.1"
//! url    = "https://atomdrift.org/scan"
//! ```

use anyhow::{Context, Result};
use semver::Version;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// The hosted manifest the notice reads.
const MANIFEST_URL: &str = "https://atomdrift.org/updates/scan.toml";
/// Minimum interval between network checks, for the notice and for the rules
/// refresh in [`crate::auto_update`].
pub(crate) const CHECK_INTERVAL_SECS: u64 = 24 * 60 * 60;
/// Whole-request budget for a manifest fetch. Short on purpose: the check must
/// never noticeably delay a command, and an offline host should fail fast.
const FETCH_TIMEOUT: Duration = Duration::from_secs(3);

/// The hosted release manifest.
#[derive(Debug, Clone, Deserialize)]
struct Manifest {
    /// Newest released version, compared against the installed one.
    latest: String,
    /// Optional URL shown in the notice (download page or release notes).
    #[serde(default)]
    url: Option<String>,
}

impl Manifest {
    fn parse(text: &str) -> Result<Self> {
        toml::from_str(text).context("parsing update manifest")
    }
}

/// True when `latest` is a strictly newer semver than `installed`.
///
/// `false` — the keep-quiet answer — when either is not valid semver: a
/// malformed manifest must never produce a spurious notice.
fn is_newer(latest: &str, installed: &str) -> bool {
    match (Version::parse(latest), Version::parse(installed)) {
        (Ok(latest), Ok(installed)) => latest > installed,
        _ => false,
    }
}

/// Fetch and parse a manifest over HTTPS.
fn fetch(url: &str) -> Result<Manifest> {
    let client = reqwest::blocking::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .connect_timeout(FETCH_TIMEOUT)
        .https_only(true)
        .user_agent(concat!("scan/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("building update-check HTTP client")?;
    // Size-capped, like every other manifest read.
    Manifest::parse(&crate::bundle::manifest(
        &crate::bundle::Http::once(client),
        url,
    )?)
}

/// On-disk record of the last notice check (successful or failed).
#[derive(Debug, Default, Serialize, Deserialize)]
struct Cache {
    /// Unix time the manifest was last fetched.
    checked_unix: u64,
    /// `latest` field from that fetch.
    latest: String,
    /// `url` field from that fetch, if any.
    #[serde(default)]
    url: Option<String>,
}

/// Path to the notice cache, in [`crate::cache_dir`].
fn cache_path() -> Option<PathBuf> {
    Some(crate::cache_dir()?.join("update-check.toml"))
}

/// Persist the notice cache (best-effort; failures are non-fatal).
fn write_cache(cache: &Cache) {
    let Some(path) = cache_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(text) = toml::to_string(cache) {
        let _ = std::fs::write(&path, text);
    }
}

/// Print the one-line notice, but only when `latest` is genuinely newer.
fn notify(latest: &str, url: Option<&str>, installed: &str) {
    if !is_newer(latest, installed) {
        return;
    }
    // Both strings come off the network: a tampered manifest must not drive
    // the terminal.
    let latest = crate::output::tty_text(latest);
    match url.map(crate::output::tty_text) {
        Some(url) => {
            eprintln!("update available: Atomdrift Scan {latest} (you have {installed}) — {url}");
        }
        None => eprintln!("update available: Atomdrift Scan {latest} (you have {installed})"),
    }
}

/// Run the update notice for an interactive command.
///
/// A fresh (<24h) cache is used without any network access. Otherwise the
/// manifest is fetched once (bounded by `FETCH_TIMEOUT`), the cache is
/// refreshed, and a notice is printed if a newer release exists. Any network or
/// parse failure is logged at debug level and otherwise ignored — the check is
/// strictly best-effort and never blocks or fails a command.
pub fn maybe_notify() {
    if crate::env::flag(crate::env::SCAN_NO_UPDATE_CHECK) {
        return;
    }
    let installed = env!("CARGO_PKG_VERSION");
    // Any read or parse error is equivalent to a missing cache.
    let cached = cache_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| toml::from_str::<Cache>(&text).ok());

    // A fresh stamp (a recent success *or* a recent failed attempt) is used
    // without any network access.
    if let Some(cache) = &cached
        && crate::civil::now_secs().saturating_sub(cache.checked_unix) < CHECK_INTERVAL_SECS
    {
        notify(&cache.latest, cache.url.as_deref(), installed);
        return;
    }

    let cache = match fetch(MANIFEST_URL) {
        Ok(manifest) => Cache {
            checked_unix: crate::civil::now_secs(),
            latest: manifest.latest,
            url: manifest.url,
        },
        Err(e) => {
            tracing::debug!(error = %format!("{e:#}"), "update check skipped");
            // Stamp the failed attempt so an unreachable manifest doesn't
            // trigger a network call on every run. Preserve any previously
            // known `latest` so a transient failure still shows the last notice.
            Cache {
                checked_unix: crate::civil::now_secs(),
                ..cached.unwrap_or_default()
            }
        }
    };
    write_cache(&cache);
    notify(&cache.latest, cache.url.as_deref(), installed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_fields() {
        // `schema` and `[[rule]]` are what manifests from the git-ref era still
        // carry; they must keep parsing.
        let m = Manifest::parse(
            "schema = 1\nlatest = \"2.0.1\"\nurl = \"https://atomdrift.org/scan\"\n[[rule]]\nmatch = '^2'\nref = \"2.0\"\n",
        )
        .unwrap();
        assert_eq!(m.latest, "2.0.1");
        assert_eq!(m.url.as_deref(), Some("https://atomdrift.org/scan"));
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
