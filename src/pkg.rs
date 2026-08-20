//! `scan url <url>` and `scan purl <purl>`: fetch one external artifact and run
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
/// The `pkg:` scheme is optional on the command line: a bare
/// `npm/left-pad@1.3.0` is accepted and normalized to a full PURL, since the
/// scheme is the same for every package and only adds typing. The whole
/// argument is then normalized (see [`fletch::purl::normalize`]) so every
/// spelling of a package — legacy types, the AUR's variants, a misordered
/// qualifier tail — enters the pipeline as the one canonical form the registry
/// lookup, bloom filters, fetcher, and provenance all key on.
///
/// # Errors
/// Returns an error if the argument isn't a usable PURL, or the PURL cannot be
/// resolved/fetched, or cleave analysis fails.
pub fn run_pkg(purl: &str, config: &ScanConfig) -> Result<ScanSummary> {
    let raw = purl.trim();
    let prefixed = if raw.starts_with("pkg:") {
        raw.to_string()
    } else {
        format!("pkg:{raw}")
    };
    let Some(purl) = fletch::purl::normalize(&prefixed) else {
        anyhow::bail!("not a package URL: {raw} (expected pkg:<type>/<name>[@version])");
    };
    run(RefLocator::Purl(purl), config)
}

fn run(locator: RefLocator, config: &ScanConfig) -> Result<ScanSummary> {
    let progress = matches!(config.format(), OutputFormat::Terminal);
    // A raw URL usually still names a package: a registry tarball URL maps back
    // onto its coordinate. Recover it so `scan url
    // https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz` gets the same
    // registry record, provenance, and package-scope composites as the
    // equivalent `scan purl` — it is the same artifact either way, and which
    // spelling the operator reached for should not change what we know about
    // it, or what hopper stores it as.
    let derived_purl = match &locator {
        RefLocator::Url(url) => fletch::purl::url_to_purl(url).map(RefLocator::Purl),
        _ => None,
    };
    // Used for the registry lookup only. Everything downstream — the fetch, the
    // bloom gate, the unpublished-version shortcut — stays keyed on what was
    // actually asked for: a URL must still be fetched from that URL, even when
    // the package it belongs to has since been pulled from the registry.
    let registry_locator = derived_purl.as_ref().unwrap_or(&locator);
    // For a known package system, look up the registry's account of the package
    // (age, author, popularity, deprecation) and print the one-line summary. The
    // record itself is grafted into the artifact's report below as a `registry`
    // node — a layer of the analyzed package, trait-matched and trained on like
    // any other file — rather than scanned as a disconnected side report, so a
    // `scope: package` composite can correlate the registry's account with the
    // artifact's own behavior.
    let (registry, registry_sources) = crate::fetch::registry_with_sources(registry_locator);
    let registry_provenance = registry.clone().map(|record| {
        crate::provenance::RegistryProvenance::from_record_sources(record, &registry_sources)
    });
    if let Some(reg) = &registry {
        crate::fetch::report_registry(reg, progress);
        // The registry lookup we just did (the packument we need for the record
        // anyway) already tells us whether this version was pulled — so when it
        // was, skip the doomed tarball fetch entirely and scan the registry
        // record directly. No extra lookup, no wasted 404 round-trip.
        if reg.version_removed == Some(true)
            && matches!(locator, RefLocator::Purl(_))
            && let Some((name, bytes)) = crate::fetch::registry_document(reg)
        {
            if progress {
                eprintln!(
                    "\n  \x1b[38;2;230;180;80m\u{26a0}\x1b[0m  \x1b[38;2;160;160;160mversion unpublished — scanning registry metadata only\x1b[0m"
                );
            }
            return crate::engine::run_bytes(
                &name,
                &name,
                bytes,
                config,
                registry_provenance.as_ref(),
                None,
            );
        }
    }
    // Known-good/known-bad by package coordinate, consulted *after* the registry
    // lookup so a known-good vouch can be overridden when the version was pulled
    // or freshly published (its trust may predate the current bytes). A known-bad
    // coordinate returns `None` here and falls through to a full fetch+scan — the
    // content-digest gate in `run_bytes` flags it inline. Removed versions are
    // already handled above (metadata-only scan).
    if let RefLocator::Purl(purl) = &locator
        && let Some(lookup) = config.bloom()
    {
        let rescan = registry
            .as_ref()
            .is_some_and(|reg| crate::fetch::must_rescan(reg, crate::fetch::unix_now()));
        if !rescan
            && let Some(summary) = crate::engine::bloom_gate(config, purl, lookup.decide_purl(purl))
        {
            return Ok(summary);
        }
    }
    match crate::fetch::fetch_one(locator, progress) {
        Ok((bytes, name, rec)) => {
            // Display under the resolved URL when there is one (a PURL resolves to
            // its download URL); fall back to the locator itself.
            let label = if rec.resolved_url.is_empty() {
                rec.locator.clone()
            } else {
                rec.resolved_url.clone()
            };
            crate::engine::run_bytes(
                &label,
                &name,
                bytes,
                config,
                registry_provenance.as_ref(),
                Some(&rec),
            )
        }
        // The artifact couldn't be fetched — most often because the version was
        // unpublished/removed (the metadata still resolves). Rather than fail the
        // whole scan, fall back to scanning the registry record so its signals —
        // including `registry.version_removed` and the freshness/custody anomalies
        // — still produce a verdict. A removed days-old package is itself a strong
        // supply-chain tell.
        Err(e) => match registry.as_ref().and_then(crate::fetch::registry_document) {
            Some((name, bytes)) => {
                if progress {
                    eprintln!(
                        "\n  \x1b[38;2;230;180;80m\u{26a0}\x1b[0m  \x1b[38;2;160;160;160martifact unavailable — scanning registry metadata only\x1b[0m"
                    );
                }
                crate::engine::run_bytes(
                    &name,
                    &name,
                    bytes,
                    config,
                    registry_provenance.as_ref(),
                    None,
                )
            }
            None => Err(e),
        },
    }
}
