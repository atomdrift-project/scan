//! R2-backed download and install of the known-good / known-bad bloom filters.
//!
//! Mirrors [`crate::model_update`]: fetch `bloom.toml`, download each filter at
//! `<base>/bloom/<built>/<file>`, verify its sha256, validate it loads (the
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

/// Public update bucket base (`<base>/bloom.toml`, `<base>/bloom/<date>/...`).
const DEFAULT_BASE_URL: &str = "https://updates.atomdrift.org/litmus";

/// Whole-request budget; bloom filters are small, but the SHA-256 one can be tens of MB.
const TIMEOUT: Duration = Duration::from_secs(120);

/// Sidecar holding the installed manifest, so `built` can be compared on refresh.
const SIDECAR: &str = "bloom.toml";

fn base_url() -> String {
    std::env::var("SCAN_BLOOM_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned())
}

/// The `built` date of the installed set, read from the sidecar, if any.
fn installed_built(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join(SIDECAR)).ok()?;
    toml::from_str::<Manifest>(&text).ok().map(|m| m.built)
}

/// Install or refresh the bloom filters. Skips the download when the installed
/// `built` already matches the manifest, unless `force`.
///
/// # Errors
/// Returns an error if the manifest or any filter cannot be fetched, a sha256
/// mismatches, a filter fails to load, or the atomic install fails.
pub fn update(dir: &Path, force: bool) -> Result<()> {
    let manifest = fetch_manifest()?;
    if !force && installed_built(dir).as_deref() == Some(manifest.built.as_str()) {
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
    match installed_built(dir) {
        Some(built) if built == manifest.built => {
            eprintln!("Bloom filters up to date: {built}");
        }
        Some(built) => {
            eprintln!(
                "Bloom filter update available: {} — currently {built}",
                manifest.built
            );
        }
        None => eprintln!("Bloom filters not installed; available: {}", manifest.built),
    }
    Ok(())
}

fn fetch_manifest() -> Result<Manifest> {
    let url = format!("{}/bloom.toml", base_url());
    let text = http_get(&url)?;
    let text = String::from_utf8(text).context("bloom manifest is not valid UTF-8")?;
    toml::from_str(&text).with_context(|| format!("parsing bloom manifest {url}"))
}

/// Download, verify, and validate every filter into a staging dir, then swap it
/// in atomically (old aside, staging in, drop old; restore on failure).
fn install(dir: &Path, manifest: &Manifest) -> Result<()> {
    let base = base_url();
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
        let url = format!("{base}/bloom/{}/{}", manifest.built, entry.file);
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
