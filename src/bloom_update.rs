//! R2-backed download and install of the known-good / known-bad bloom filters.
//!
//! Mirrors [`crate::model_update`]: fetch `bloom.toml`, download each filter at
//! `<base>/v<FORMAT_VERSION>/<file>`, verify its sha256, validate it loads (the
//! [`FORMAT_VERSION`] gate fails closed on an unknown layout) and matches its
//! declared identity, then atomically swap the whole set into place. Validation
//! happens on the staged copy *before* the swap, so a broken or partial download
//! never replaces the live filters.
//!
//! No signing: the filters carry only a versioned layout and per-file sha256,
//! not an authenticity claim — trust is HTTPS to our own bucket. The base URL is
//! overridable with `SCAN_BLOOM_URL` (used to point at a local server in tests).

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::bloom::{FORMAT_VERSION, Filter};
use crate::bloom_build::Manifest;

/// Public update bucket base. Bloom artifacts live under a format-versioned
/// prefix beneath `bloom/` (`<base>/bloom/v<FORMAT_VERSION>/bloom.toml`,
/// `<base>/bloom/v<FORMAT_VERSION>/<file>`), so an incompatible format change
/// publishes to a new prefix without disturbing the one older clients still read.
const DEFAULT_BASE_URL: &str = "https://updates.atomdrift.org/litmus";

/// Whole-request budget; bloom filters are small, but the SHA-256 one can be tens of MB.
const TIMEOUT: Duration = Duration::from_secs(120);

/// Sidecar holding the installed manifest, so the installed filters' sha256s can
/// be compared against the remote manifest on refresh.
const SIDECAR: &str = "bloom.toml";

/// Path prefix for this build's bloom artifacts: namespaced under `bloom/` and
/// selecting the on-wire format it speaks, e.g. `bloom/v1`. A format bump moves
/// the whole prefix, so old and new clients never read each other's artifacts.
fn bloom_prefix() -> String {
    format!("bloom/v{FORMAT_VERSION}")
}

fn base_url() -> String {
    std::env::var("SCAN_BLOOM_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned())
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

/// Install or refresh the bloom filters. Skips the download when the installed
/// `built` already matches the manifest, unless `force`.
///
/// # Errors
/// Returns an error if the manifest or any filter cannot be fetched, a sha256
/// mismatches, a filter fails to load, or the atomic install fails.
pub fn update(dir: &Path, force: bool) -> Result<()> {
    let manifest = fetch_manifest()?;
    if !force && installed_manifest(dir).is_some_and(|installed| is_current(&installed, &manifest)) {
        eprintln!("Bloom filters already up to date: {}", manifest.built);
        return Ok(());
    }
    install(dir, &manifest)?;
    eprintln!(
        "Bloom filters updated to {} at {}",
        manifest.built,
        dir.display()
    );
    Ok(())
}

/// Report what would be installed without changing anything.
///
/// # Errors
/// Returns an error if the manifest cannot be fetched or parsed.
pub fn check(dir: &Path) -> Result<()> {
    let manifest = fetch_manifest()?;
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

fn fetch_manifest() -> Result<Manifest> {
    let url = format!("{}/{}/bloom.toml", base_url(), bloom_prefix());
    let text = http_get(&url)?;
    let text = String::from_utf8(text).context("bloom manifest is not valid UTF-8")?;
    toml::from_str(&text).with_context(|| format!("parsing bloom manifest {url}"))
}

/// Download, verify, and validate every filter into a staging dir, then swap it
/// in atomically (old aside, staging in, drop old; restore on failure).
fn install(dir: &Path, manifest: &Manifest) -> Result<()> {
    let base = base_url();
    let prefix = bloom_prefix();
    let parent = dir.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;

    let staging = parent.join(".scan-bloom-staging");
    let backup = parent.join(".scan-bloom-backup");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).context("creating staging dir")?;

    for (stem, entry) in &manifest.filter {
        if entry.format_version != FORMAT_VERSION {
            bail!(
                "bloom filter {stem} needs format v{}, this build supports v{FORMAT_VERSION}; upgrade scan",
                entry.format_version
            );
        }
        let url = format!("{base}/{prefix}/{}", entry.file);
        let bytes = http_get(&url)?;

        let got = format!("{:x}", Sha256::digest(&bytes));
        if got != entry.sha256 {
            bail!(
                "sha256 mismatch for {}: got {got}, manifest says {}",
                entry.file,
                entry.sha256
            );
        }

        // Validate before staging: it must load (FORMAT_VERSION gate) and its
        // header identity must match the file name the manifest gave it.
        let filter = Filter::load(&bytes).with_context(|| format!("validating {}", entry.file))?;
        let want = format!("{}.adbl", filter.artifact_stem());
        if want != entry.file {
            bail!(
                "bloom filter {} identifies as {want}; refusing to install",
                entry.file
            );
        }

        std::fs::write(staging.join(&entry.file), &bytes)
            .with_context(|| format!("writing {}", entry.file))?;
    }

    // Sidecar = the manifest itself, so `installed_built` reads it back.
    let rendered = toml::to_string(manifest).context("rendering bloom sidecar")?;
    std::fs::write(staging.join(SIDECAR), rendered).context("writing bloom sidecar")?;

    let _ = std::fs::remove_dir_all(&backup);
    if dir.exists() {
        std::fs::rename(dir, &backup).context("backing up current bloom filters")?;
    }
    if let Err(e) = std::fs::rename(&staging, dir) {
        if backup.exists() {
            let _ = std::fs::rename(&backup, dir);
        }
        return Err(e).context("installing bloom filters");
    }
    let _ = std::fs::remove_dir_all(&backup);
    Ok(())
}

fn http_get(url: &str) -> Result<Vec<u8>> {
    eprintln!("fetching {url}");
    let client = reqwest::blocking::Client::builder()
        .timeout(TIMEOUT)
        .build()
        .context("building http client")?;
    let resp = client
        .get(url)
        .send()
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("GET {url}"))?;
    Ok(resp
        .bytes()
        .with_context(|| format!("reading {url}"))?
        .to_vec())
}
