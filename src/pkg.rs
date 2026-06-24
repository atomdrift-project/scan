//! `ascan url <url>` and `ascan pkg <purl>`: fetch one external artifact and run
//! the full scan pipeline on it, exactly as if it were a local file.
//!
//! `url` takes a raw URL (`https://host/path`); `pkg` takes a package URL
//! (`pkg:npm/left-pad@1.3.0`, `pkg:pypi/requests`, …), which fletch resolves to
//! its download URL. Both share one fetch+scan path, differing only in how the
//! locator is built. The fetch is cached and SSRF-guarded like every other fetch.

use anyhow::Result;
use fletch::RefLocator;

use crate::engine::ScanConfig;
use crate::{OutputFormat, ScanSummary};

/// Fetch and scan a raw URL.
///
/// # Errors
/// Returns an error if the URL cannot be fetched or cleave analysis fails.
pub fn run_url(url: &str, config: &ScanConfig) -> Result<ScanSummary> {
    run(RefLocator::Url(url.to_string()), config)
}

/// Fetch and scan a package URL (PURL, e.g. `pkg:npm/left-pad@1.3.0`).
///
/// # Errors
/// Returns an error if the PURL cannot be resolved/fetched or cleave analysis
/// fails.
pub fn run_pkg(purl: &str, config: &ScanConfig) -> Result<ScanSummary> {
    run(RefLocator::Purl(purl.to_string()), config)
}

fn run(locator: RefLocator, config: &ScanConfig) -> Result<ScanSummary> {
    let progress = matches!(config.format(), OutputFormat::Terminal);
    // For a known package system, surface the registry's metadata (age, author,
    // popularity, deprecation) and scan the normalized record itself, so traits
    // over `registry.*` facts fire — the same metadata the dependency age-gate
    // consults, looked up before the fetch.
    if let Some(reg) = crate::fetch::registry(&locator) {
        crate::fetch::report_registry(&reg, progress);
        if let Some((name, bytes)) = crate::fetch::registry_document(&reg) {
            crate::engine::run_bytes(&name, &name, bytes, config)?;
        }
    }
    let (bytes, name, rec) = crate::fetch::fetch_one(locator, progress)?;
    // Display under the resolved URL when there is one (a PURL resolves to its
    // download URL); fall back to the locator itself.
    let label = if rec.resolved_url.is_empty() {
        rec.locator.clone()
    } else {
        rec.resolved_url.clone()
    };
    crate::engine::run_bytes(&label, &name, bytes, config)
}
