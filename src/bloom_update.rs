//! R2-backed download and install of the known-good / known-bad bloom filters.
//!
//! Installed through `crate::bundle`: fetch `bloom.toml`, download each
//! filter at `<base>/bloom/v<N>/<file>`, verify its sha256, validate it loads
//! (the version gate fails closed on a layout outside
//! [`burton::SUPPORTED_VERSIONS`]) and matches its declared identity, then
//! swap the whole set into place. Validation happens on the staged copy
//! *before* the swap, so a broken or partial download never replaces the live
//! filters.
//!
//! `<N>` is resolved rather than assumed: the newest published prefix this build
//! can read wins, so one bucket can serve a bundle per format version and a
//! client takes the best one it understands. See `fetch_manifest`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;

use burton::{Filter, Manifest, SUPPORTED_VERSIONS};

use crate::bundle::{self, Bundle};

/// Sidecar holding the installed manifest, so the installed filters' sha256s can
/// be compared against the remote manifest on refresh.
const SIDECAR: &str = "bloom.toml";

/// The installed manifest (sidecar), if present and parseable.
fn installed_manifest(dir: &Path) -> Option<Manifest> {
    let text = std::fs::read_to_string(dir.join(SIDECAR)).ok()?;
    toml::from_str(&text).ok()
}

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

/// Report what would be installed without changing anything.
///
/// # Errors
/// Returns an error if the manifest cannot be fetched or parsed.
pub fn check(dir: &Path) -> Result<()> {
    bundle::check::<Bloom>(dir)
}

/// The bloom filters, as [`bundle::update`] installs them.
struct Bloom;

/// A manifest and the prefix it was published under, which its filters must
/// be fetched from too.
struct Published {
    manifest: Manifest,
    prefix: String,
}

impl Bundle for Bloom {
    const NAME: &'static str = "Bloom filters";
    type Installed = Manifest;
    type Remote = Published;

    fn installed(dir: &Path) -> Option<Manifest> {
        installed_manifest(dir)
    }

    fn fetch(client: &Client) -> Result<Published> {
        fetch_manifest(client)
    }

    fn is_current(_dir: &Path, have: &Manifest, remote: &Published) -> bool {
        is_current(have, &remote.manifest)
    }

    fn rollback(have: &Manifest, remote: &Published) -> Option<String> {
        rolls_back(have, &remote.manifest).then(|| {
            format!(
                "the published filters (built {}) are older than the installed ones (built {})",
                remote.manifest.built, have.built
            )
        })
    }

    fn stage(staging: &Path, client: &Client, remote: &Published) -> Result<()> {
        stage(staging, client, remote)
    }

    fn describe(remote: &Published) -> String {
        remote.manifest.built.clone()
    }

    fn describe_installed(have: &Manifest) -> String {
        have.built.clone()
    }
}

/// The newest bundle the bucket actually carries, with the prefix it came from.
///
/// Tries each of [`SUPPORTED_VERSIONS`] newest-first, so a build that speaks v2
/// keeps working against a bucket that has not been dual-published yet, and
/// against the v1 prefix during a rollback. The prefix travels with the manifest
/// because the filters must be fetched from the SAME one: deriving it again
/// later would download v2 files against a v1 manifest the moment the two
/// disagree.
///
/// Only a 404 advances to the next prefix. Every other failure — a timeout, a
/// 5xx, DNS — is a real failure and is returned as one; treating it as "not
/// published here" would let one slow request silently downgrade a client to an
/// older format and leave it there.
fn fetch_manifest(client: &Client) -> Result<Published> {
    let mut tried: Vec<String> = Vec::new();
    for version in SUPPORTED_VERSIONS {
        // Namespaced under `bloom/` and selecting the on-wire format this build
        // speaks, e.g. `bloom/v1`. A format bump moves the whole prefix, so old
        // and new clients never read each other's artifacts.
        let prefix = format!("bloom/v{version}");
        let url = bundle::url(&format!("{prefix}/{SIDECAR}"));
        match bundle::manifest(client, &url) {
            Ok(text) => {
                let manifest: Manifest = toml::from_str(&text)
                    .with_context(|| format!("parsing bloom manifest {url}"))?;
                if !tried.is_empty() {
                    tracing::debug!("no bloom bundle at {}; using {prefix}", tried.join(", "));
                }
                return Ok(Published { manifest, prefix });
            }
            Err(err) if is_not_found(&err) => tried.push(prefix),
            Err(err) => return Err(err),
        }
    }
    bail!(
        "no bloom manifest published at any prefix this build understands ({})",
        tried.join(", ")
    )
}

fn is_not_found(err: &anyhow::Error) -> bool {
    err.chain()
        .filter_map(|cause| cause.downcast_ref::<reqwest::Error>())
        .any(|e| e.status() == Some(reqwest::StatusCode::NOT_FOUND))
}

/// Download, verify, and validate every filter into `staging`, then write the
/// manifest beside them as the sidecar.
fn stage(staging: &Path, client: &Client, remote: &Published) -> Result<()> {
    let Published { manifest, prefix } = remote;
    for (stem, entry) in &manifest.filter {
        // A version this build cannot read is a real stop, not something to
        // work around: the layout would have to be guessed at. Fail closed and
        // keep the installed filters, which are at least a layout we understand.
        if !SUPPORTED_VERSIONS.contains(&entry.format_version) {
            bail!(
                "bloom filter {stem} needs format v{}, this build reads {:?}; upgrade scan",
                entry.format_version,
                SUPPORTED_VERSIONS
            );
        }
        // The name must be the one the filter's own header gives it, checked
        // below; until then it must at least stay inside `staging`.
        if Path::new(&entry.file).file_name() != Some(std::ffi::OsStr::new(&entry.file)) {
            bail!(
                "bloom filter {stem} has an unsafe file name {:?}",
                entry.file
            );
        }
        let path = staging.join(&entry.file);
        let url = bundle::url(&format!("{prefix}/{}", entry.file));
        bundle::download_to(client, &url, &entry.sha256, &path)?;

        // It must load (FORMAT_VERSION gate) and its header identity must match
        // the file name the manifest gave it.
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", entry.file))?;
        let filter = Filter::load(bytes).with_context(|| format!("validating {}", entry.file))?;
        let want = format!("{}.adbl", filter.stem());
        if want != entry.file {
            bail!(
                "bloom filter {} identifies as {want}; refusing to install",
                entry.file
            );
        }
    }

    // The sidecar is the manifest itself, which `installed_manifest` reads back.
    let rendered = toml::to_string(manifest).context("rendering bloom sidecar")?;
    std::fs::write(staging.join(SIDECAR), rendered).context("writing bloom sidecar")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn built(date: &str) -> Manifest {
        Manifest {
            schema: 1,
            built: date.to_owned(),
            key_scheme: None,
            filter: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn an_older_bundle_is_a_rollback_and_a_same_day_rebuild_is_not() {
        let installed = built("2026-10-09");
        assert!(rolls_back(&installed, &built("2026-09-30")));
        assert!(!rolls_back(&installed, &built("2026-10-09")));
        assert!(!rolls_back(&installed, &built("2026-10-10")));
        // Malformed dates never block an update: the guard only reads what it
        // can order.
        assert!(!rolls_back(&installed, &built("")));
        assert!(!rolls_back(&built("unknown"), &built("2020-01-01")));
    }
}
