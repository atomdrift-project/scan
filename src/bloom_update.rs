//! R2-backed download and install of the known-good / known-bad bloom filters.
//!
//! Mirrors [`crate::model_update`]: fetch `bloom.toml`, download each filter at
//! `<base>/bloom/v<N>/<file>`, verify its sha256, validate it loads (the
//! version gate fails closed on a layout outside
//! [`burton::SUPPORTED_VERSIONS`]) and matches its declared identity, then
//! atomically swap the whole set into place. Validation happens on the staged
//! copy *before* the swap, so a broken or partial download never replaces the
//! live filters.
//!
//! `<N>` is resolved rather than assumed: the newest published prefix this build
//! can read wins, so one bucket can serve a bundle per format version and a
//! client takes the best one it understands. See [`fetch_manifest`].
//!
//! No signing: the filters carry only a versioned layout and per-file sha256,
//! not an authenticity claim — trust is HTTPS to our own bucket. The base URL is
//! overridable with `SCAN_BLOOM_URL` (used to point at a local server in tests).

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use burton::{Filter, Manifest, SUPPORTED_VERSIONS};

/// Public update bucket base. Bloom artifacts live under a format-versioned
/// prefix beneath `bloom/` (`<base>/bloom/v<FORMAT_VERSION>/bloom.toml`,
/// `<base>/bloom/v<FORMAT_VERSION>/<file>`), so an incompatible format change
/// publishes to a new prefix without disturbing the one older clients still read.
const DEFAULT_BASE_URL: &str = "https://updates.atomdrift.org/litmus";

/// Whole-request budget; bloom filters are small, but the SHA-256 one can be tens of MB.
const TIMEOUT: Duration = Duration::from_secs(120);
/// Connect budget — fail fast when the bucket is unreachable so the default
/// auto-update can't stall a scan on an offline host (the whole-request
/// [`TIMEOUT`] still covers a slow but reachable download).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(4);

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
pub fn update(dir: &Path, force: bool, quiet: bool) -> Result<bool> {
    // A quiet refresh of an already-present install is the auto-update path:
    // fail fast if the bucket is unreachable. A first-ever download — or any
    // explicit, non-quiet update — stays patient so a slow first fetch still
    // completes.
    let connect = (quiet && installed_manifest(dir).is_some()).then_some(CONNECT_TIMEOUT);
    let (manifest, prefix) = fetch_manifest(connect)?;
    if !force && installed_manifest(dir).is_some_and(|installed| is_current(&installed, &manifest))
    {
        if !quiet {
            eprintln!("Bloom filters already up to date: {}", manifest.built);
        }
        return Ok(false);
    }
    install(dir, &manifest, &prefix)?;
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
    let (manifest, _prefix) = fetch_manifest(None)?;
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
fn fetch_manifest(connect: Option<Duration>) -> Result<(Manifest, String)> {
    let mut tried: Vec<String> = Vec::new();
    for version in SUPPORTED_VERSIONS {
        let prefix = bloom_prefix_for(*version);
        let url = format!("{}/{prefix}/bloom.toml", base_url());
        match http_get_optional(&url, connect)? {
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
fn http_get_optional(url: &str, connect: Option<Duration>) -> Result<Option<Vec<u8>>> {
    match http_get(url, connect) {
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

/// Download, verify, and validate every filter into a staging dir, then swap it
/// in atomically (old aside, staging in, drop old; restore on failure).
fn install(dir: &Path, manifest: &Manifest, prefix: &str) -> Result<()> {
    let base = base_url();
    let parent = dir.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;

    let staging = parent.join(".scan-bloom-staging");
    let backup = parent.join(".scan-bloom-backup");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).context("creating staging dir")?;

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
        let url = format!("{base}/{prefix}/{}", entry.file);
        // Patient: the manifest fetch already reached the bucket, so the bundle
        // download isn't where an offline host stalls.
        let bytes = http_get(&url, None)?;

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

fn http_get(url: &str, connect: Option<Duration>) -> Result<Vec<u8>> {
    tracing::debug!("fetching {url}");
    let mut builder = reqwest::blocking::Client::builder().timeout(TIMEOUT);
    if let Some(connect) = connect {
        builder = builder.connect_timeout(connect);
    }
    let client = builder.build().context("building http client")?;
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
