//! Download and install of the known-good / known-bad bloom filters.
//!
//! Installed through `crate::bundle`: fetch `bloom.toml`, download each
//! filter at `<base>/bloom/v<N>/<file>`, verify its sha256, validate it loads
//! (which fails closed on any layout but this build's) and matches its
//! declared identity, then swap the whole set into place. Validation happens
//! on the staged copy *before* the swap, so a broken or partial download never
//! replaces the live filters.
//!
//! `<N>` is [`burton::FORMAT_VERSION`]. A format bump moves the whole prefix,
//! so old and new clients never read each other's artifacts: each fetches the
//! one format it reads, and a publisher serves every format still in use side
//! by side.

use std::path::Path;

use anyhow::{Context, Result, bail};

use burton::{MANIFEST_FILE, Manifest};

use crate::bundle::{self, Bundle};

/// True when every filter the remote offers is already installed with the same
/// sha256, and the installed set holds no more than the remote. Content-based,
/// so a same-day rebuild that changes bytes is still detected — unlike comparing
/// the `built` date, which is coarse to the day and so blind to hourly rebuilds.
fn is_current(installed: &Manifest, remote: &Manifest) -> bool {
    remote.filter.len() == installed.filter.len()
        && remote.filter.iter().all(|(stem, entry)| {
            installed
                .filter
                .get(stem)
                .is_some_and(|have| have.sha256 == entry.sha256)
        })
}

/// Whether `remote` was built before `installed`. Only well-formed
/// `YYYY-MM-DD` dates compare, and a same-day rebuild is never a rollback.
fn rolls_back(installed: &Manifest, remote: &Manifest) -> bool {
    let day = |d: &str| crate::output::parse_ymd(d);
    matches!((day(&installed.built), day(&remote.built)), (Some(have), Some(next)) if next < have)
}

/// Install or refresh the bloom filters. Skips the download when every
/// installed filter's sha256 already matches the manifest, unless `force`.
/// Returns `true` when filters were installed.
///
/// # Errors
/// Returns an error if the manifest or any filter cannot be fetched, a sha256
/// mismatches, a filter fails to load, or the install fails.
pub fn update(dir: &Path, force: bool, quiet: bool) -> Result<bool> {
    bundle::update::<Bloom>(dir, force, quiet)
}

/// The bloom filters, as [`bundle::update`] installs them.
struct Bloom;

/// Where this build's bundle is published, manifest and filters alike.
fn prefix() -> String {
    format!("bloom/v{}", burton::FORMAT_VERSION)
}

impl Bundle for Bloom {
    const NAME: &'static str = "Bloom filters";
    type Installed = Manifest;
    type Remote = Manifest;

    /// The manifest the last install left beside the filters, so their
    /// sha256s can be compared against the remote one on refresh.
    fn installed(dir: &Path) -> Option<Manifest> {
        burton::read_manifest(dir).ok().flatten()
    }

    fn fetch(http: &bundle::Http) -> Result<Manifest> {
        let url = bundle::url(&format!("{}/{MANIFEST_FILE}", prefix()));
        let text = bundle::manifest(http, &url)?;
        toml::from_str(&text).with_context(|| format!("parsing bloom manifest {url}"))
    }

    fn is_current(_dir: &Path, have: &Manifest, remote: &Manifest) -> bool {
        is_current(have, remote)
    }

    fn rollback(have: &Manifest, remote: &Manifest) -> Option<String> {
        rolls_back(have, remote).then(|| {
            format!(
                "the published filters (built {}) are older than the installed ones (built {})",
                remote.built, have.built
            )
        })
    }

    fn stage(staging: &Path, http: &bundle::Http, remote: &Manifest) -> Result<()> {
        stage(staging, http, remote)
    }

    fn describe(remote: &Manifest) -> String {
        remote.built.clone()
    }
}

/// Download, verify, and validate every filter into `staging`, then write the
/// manifest beside them as the sidecar.
fn stage(staging: &Path, http: &bundle::Http, manifest: &Manifest) -> Result<()> {
    // The build date is what the rollback guard orders bundles by; a manifest
    // without one could replace newer filters with older ones unnoticed.
    if crate::output::parse_ymd(&manifest.built).is_none() {
        bail!(
            "bloom manifest has no valid build date ({:?}); refusing to install",
            manifest.built
        );
    }
    // Each filter is named for the stem its own header gives it, which
    // `stage_filters` checks against the manifest's key.
    if let Some((stem, entry)) = manifest
        .filter
        .iter()
        .find(|(stem, entry)| entry.file != format!("{stem}.adbl"))
    {
        bail!(
            "bloom filter {stem} is named {:?}; refusing to install",
            entry.file
        );
    }
    bundle::stage_filters(staging, http, &bundle::url(&prefix()), manifest)?;
    // The sidecar is the manifest itself, which `Bloom::installed` reads back.
    let rendered = toml::to_string(manifest).context("rendering bloom sidecar")?;
    std::fs::write(staging.join(MANIFEST_FILE), rendered).context("writing bloom sidecar")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn built(date: &str) -> Manifest {
        Manifest {
            schema: burton::MANIFEST_SCHEMA,
            built: date.to_owned(),
            key_scheme: crate::bloom_repo::KEY_SCHEME.to_owned(),
            provenance: toml::Table::new(),
            filter: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn an_older_bundle_is_a_rollback_and_a_same_day_rebuild_is_not() {
        let installed = built("2026-10-09");
        assert!(rolls_back(&installed, &built("2026-09-30")));
        assert!(!rolls_back(&installed, &built("2026-10-09")));
        assert!(!rolls_back(&installed, &built("2026-10-10")));
        // The guard only orders well-formed dates; `stage` refuses a remote
        // manifest without one, so only a damaged install reads as unknown.
        assert!(!rolls_back(&installed, &built("")));
        assert!(!rolls_back(&built("unknown"), &built("2020-01-01")));
    }

    #[test]
    fn a_manifest_without_a_build_date_is_refused() {
        let staging = tempfile::tempdir().expect("tempdir");
        let http = bundle::Http::once(reqwest::blocking::Client::new());
        for date in ["", "unknown", "2026-13-40"] {
            let err = stage(staging.path(), &http, &built(date)).expect_err(date);
            assert!(err.to_string().contains("no valid build date"), "{err}");
        }
    }
}
