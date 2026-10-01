//! R2-backed download and install of the known-good / known-bad bloom filters.
//!
//! Shares [`crate::model_update`]'s plumbing: fetch `bloom.toml`, download each
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
//!
//! No signing: the filters carry only a versioned layout and per-file sha256,
//! not an authenticity claim — trust is HTTPS to our own bucket. The base URL is
//! overridable with `SCAN_BLOOM_URL` (to point at a local server).

use std::path::Path;

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;

use burton::{Filter, Manifest, SUPPORTED_VERSIONS};

use crate::model_update::{Installer, client, get, get_verified, hands_off};

/// Sidecar holding the installed manifest, so the installed filters' sha256s can
/// be compared against the remote manifest on refresh.
const SIDECAR: &str = "bloom.toml";

/// Path prefix for this build's bloom artifacts: namespaced under `bloom/` and
/// selecting the on-wire format it speaks, e.g. `bloom/v1`. A format bump moves
/// the whole prefix, so old and new clients never read each other's artifacts.
fn bloom_prefix_for(version: u16) -> String {
    format!("bloom/v{version}")
}

fn base_url() -> String {
    std::env::var("SCAN_BLOOM_URL").unwrap_or_else(|_| crate::model_update::BASE_URL.to_owned())
}

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

/// Install or refresh the bloom filters. Skips the download when every
/// installed filter's sha256 already matches the manifest, unless `force`.
/// Returns `true` when filters were installed.
///
/// # Errors
/// Returns an error if the manifest or any filter cannot be fetched, a sha256
/// mismatches, a filter fails to load, or the install fails.
pub fn update(dir: &Path, force: bool, quiet: bool) -> Result<bool> {
    if let Some(why) = hands_off("Bloom filters", dir, "SCAN_BLOOM_DIR") {
        if !quiet {
            eprintln!("{why}");
        }
        return Ok(false);
    }
    let base = base_url();
    let client = client(quiet && installed_manifest(dir).is_some())?;
    let (manifest, prefix) = fetch_manifest(&client, &base)?;

    let current =
        || !force && installed_manifest(dir).is_some_and(|have| is_current(&have, &manifest));
    if current() {
        if !quiet {
            eprintln!("Bloom filters already up to date: {}", manifest.built);
        }
        return Ok(false);
    }
    let installer = Installer::lock(dir)?;
    if current() {
        return Ok(false); // another updater installed it while we waited
    }
    let prefix_url = format!("{base}/{prefix}");
    installer.install(|staging| stage(staging, &client, &prefix_url, &manifest))?;
    if !quiet {
        eprintln!(
            "Bloom filters updated to {} at {}",
            manifest.built,
            dir.display()
        );
    }
    Ok(true)
}

/// Report what would be installed without changing anything.
///
/// # Errors
/// Returns an error if the manifest cannot be fetched or parsed.
pub fn check(dir: &Path) -> Result<()> {
    if let Some(why) = hands_off("Bloom filters", dir, "SCAN_BLOOM_DIR") {
        eprintln!("{why}");
        return Ok(());
    }
    let (manifest, _prefix) = fetch_manifest(&client(false)?, &base_url())?;
    match installed_manifest(dir) {
        Some(installed) if is_current(&installed, &manifest) => {
            eprintln!("Bloom filters up to date: {}", installed.built);
        }
        Some(installed) => {
            eprintln!(
                "Bloom filter update available: {} — currently {}",
                manifest.built, installed.built
            );
        }
        None => eprintln!("Bloom filters not installed; available: {}", manifest.built),
    }
    Ok(())
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
fn fetch_manifest(client: &Client, base: &str) -> Result<(Manifest, String)> {
    let mut tried: Vec<String> = Vec::new();
    for version in SUPPORTED_VERSIONS {
        let prefix = bloom_prefix_for(*version);
        let url = format!("{base}/{prefix}/bloom.toml");
        match get_optional(client, &url)? {
            Some(bytes) => {
                let text = String::from_utf8(bytes).context("bloom manifest is not valid UTF-8")?;
                let manifest: Manifest = toml::from_str(&text)
                    .with_context(|| format!("parsing bloom manifest {url}"))?;
                if !tried.is_empty() {
                    tracing::debug!("no bloom bundle at {}; using {prefix}", tried.join(", "));
                }
                return Ok((manifest, prefix));
            }
            None => tried.push(prefix),
        }
    }
    bail!(
        "no bloom manifest published at any prefix this build understands ({})",
        tried.join(", ")
    )
}

/// `Ok(None)` when the object is absent, `Err` when the fetch itself failed.
fn get_optional(client: &Client, url: &str) -> Result<Option<Vec<u8>>> {
    match get(client, url) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(err) => {
            let missing = err
                .chain()
                .filter_map(|cause| cause.downcast_ref::<reqwest::Error>())
                .any(|e| e.status() == Some(reqwest::StatusCode::NOT_FOUND));
            if missing { Ok(None) } else { Err(err) }
        }
    }
}

/// Download (from `prefix_url`), verify, and validate every filter into
/// `staging`, then write the manifest beside them as the sidecar.
fn stage(staging: &Path, client: &Client, prefix_url: &str, manifest: &Manifest) -> Result<()> {
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
        let bytes = get_verified(
            client,
            &format!("{prefix_url}/{}", entry.file),
            &entry.sha256,
        )?;

        // It must load (FORMAT_VERSION gate) and its header identity must match
        // the file name the manifest gave it.
        let filter = Filter::load(bytes).with_context(|| format!("validating {}", entry.file))?;
        let want = format!("{}.adbl", filter.stem());
        if want != entry.file {
            bail!(
                "bloom filter {} identifies as {want}; refusing to install",
                entry.file
            );
        }

        // Written from the validated filter, so the bytes that land in staging
        // are exactly the bytes that passed the checks above.
        std::fs::write(staging.join(&entry.file), filter.as_bytes())
            .with_context(|| format!("writing {}", entry.file))?;
    }

    // The sidecar is the manifest itself, which `installed_manifest` reads back.
    let rendered = toml::to_string(manifest).context("rendering bloom sidecar")?;
    std::fs::write(staging.join(SIDECAR), rendered).context("writing bloom sidecar")
}
