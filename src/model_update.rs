//! R2-backed model updates.
//!
//! `ascan update-rules` fetches a manifest from the update bucket, resolves the
//! model bundle compatible with *this* litmus release, downloads it, verifies its
//! sha256, validates it by loading, and atomically installs it.
//!
//! Mirrors cleave's trait updater. Signature verification is deferred for v1:
//! trust is HTTPS to our own bucket plus the manifest's per-artifact sha256. The
//! manifest is cosign-signed upstream, so authenticity checking can be added
//! later without changing the publish side.
//!
//! Unlike a plain swap, the freshly extracted bundle is validated with
//! [`crate::model::Model::load`] *before* it replaces the live models, so a
//! broken bundle never goes live (no swap-then-rollback window).

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Base URL for the public update bucket (`<base>/versions.toml`, `<base>/models/...`).
const BASE_URL: &str = "https://updates.atomdrift.org/litmus";

/// Whole-request budget — models are larger than trait bundles.
const TIMEOUT: Duration = Duration::from_secs(120);

/// Sidecar recording what the R2 backend installed (no `.git` in the tree).
const SIDECAR: &str = ".litmus-models.toml";

/// This build's version, matched against the manifest's release keys (e.g. `2.0.0-rc.4`).
fn our_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    latest: String,
    #[serde(default)]
    artifacts: BTreeMap<String, Artifact>,
    #[serde(default)]
    stable: BTreeMap<String, String>,
    #[serde(default)]
    upgrade: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct Artifact {
    file: String,
    sha256: String,
    commit: String,
    date: String,
}

/// Sidecar contents — what the R2 backend last installed.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Installed {
    /// Content id of the installed model bundle (manifest artifact key).
    pub commit: String,
    /// Build date of the bundle (`YYYY-MM-DD`).
    pub date: String,
    /// `stable:<version>` or `latest` — how the pointer was resolved.
    pub source: String,
    /// The litmus version that performed the install.
    pub version: String,
}

/// Read the install sidecar, if the models dir was populated by the R2 backend.
#[must_use]
pub fn installed(dir: &Path) -> Option<Installed> {
    let text = std::fs::read_to_string(dir.join(SIDECAR)).ok()?;
    toml::from_str(&text).ok()
}

/// True if the dir is a git checkout (or symlinked to one) — a dev tree we must
/// never overwrite. `join(".git").exists()` follows the symlink, so it catches
/// both the symlinked-to-a-checkout and the direct-checkout cases.
fn is_git_managed(dir: &Path) -> bool {
    dir.join(".git").exists()
}

/// Install or refresh the model bundle compatible with this litmus release.
pub fn update(dir: &Path, force: bool) -> Result<()> {
    if is_git_managed(dir) {
        eprintln!(
            "Models at {} are git-managed (a checkout or symlink to one); leaving them untouched.\nUse 'git pull' there to update, or remove the directory to switch to bundle updates.",
            dir.display()
        );
        return Ok(());
    }
    let manifest = fetch_manifest()?;
    let (key, source) = resolve(&manifest)?;
    let artifact = artifact_for(&manifest, &key)?;

    if !force && installed(dir).is_some_and(|i| i.commit.starts_with(&key)) {
        eprintln!("Models already up to date: {} ({})", key, artifact.date);
        warn_if_behind(&manifest);
        return Ok(());
    }

    eprintln!(
        "Installing models {} ({}) for Atomdrift Scan {} [{}]...",
        key,
        artifact.date,
        our_version(),
        source
    );
    install(dir, artifact, &source)?;
    eprintln!(
        "Models updated to {} ({}) at {}",
        key,
        artifact.date,
        dir.display()
    );
    warn_if_behind(&manifest);
    Ok(())
}

/// Report what would be installed without changing anything.
pub fn check(dir: &Path) -> Result<()> {
    if is_git_managed(dir) {
        eprintln!(
            "Models at {} are git-managed; use 'git pull' there to update.",
            dir.display()
        );
        return Ok(());
    }
    let manifest = fetch_manifest()?;
    let (key, source) = resolve(&manifest)?;
    let artifact = artifact_for(&manifest, &key)?;

    match installed(dir) {
        Some(i) if i.commit.starts_with(&key) => {
            eprintln!("Models up to date: {} ({})", key, artifact.date);
        }
        Some(i) => {
            let current: String = i.commit.chars().take(12).collect();
            eprintln!(
                "Model update available: {} ({}) — currently {} [{}]",
                key, artifact.date, current, source
            );
        }
        None => eprintln!(
            "Models not installed; available: {} ({}) [{}]",
            key, artifact.date, source
        ),
    }
    warn_if_behind(&manifest);
    Ok(())
}

// --- internals --------------------------------------------------------------

/// Resolve this build's model pointer: its own `[stable]` entry, else `latest`.
fn resolve(m: &Manifest) -> Result<(String, String)> {
    if let Some(key) = m.stable.get(our_version()) {
        return Ok((key.clone(), format!("stable:{}", our_version())));
    }
    if !m.latest.is_empty() {
        return Ok((m.latest.clone(), "latest".to_string()));
    }
    bail!("manifest has no pointer for this version and no `latest`")
}

fn artifact_for<'a>(m: &'a Manifest, key: &str) -> Result<&'a Artifact> {
    m.artifacts
        .get(key)
        .with_context(|| format!("manifest references {key} but has no [artifacts.{key}] entry"))
}

/// Warn (but don't fail) if a newer *release* supports models this build can't.
/// The manifest's `[upgrade]` table already excludes HEAD-only-ahead cases, so a
/// dev/unlisted build is never warned.
fn warn_if_behind(m: &Manifest) {
    if let Some(target) = m.upgrade.get(our_version()) {
        eprintln!(
            "Note: Atomdrift Scan {} cannot use the newest models. Upgrade to Atomdrift Scan {} for the latest detections.",
            our_version(),
            target
        );
    }
}

fn fetch_manifest() -> Result<Manifest> {
    let url = format!("{BASE_URL}/versions.toml");
    let text = http_get(&url)?;
    let text = String::from_utf8(text).context("manifest is not valid UTF-8")?;
    toml::from_str(&text).with_context(|| format!("parsing manifest {url}"))
}

/// Download a bundle, verify its sha256, validate it loads, and atomically swap
/// it into `dir`. Validation happens on the staged copy *before* the swap, so a
/// broken bundle never replaces the live models.
fn install(dir: &Path, artifact: &Artifact, source: &str) -> Result<()> {
    let url = format!("{BASE_URL}/{}", artifact.file);
    let bytes = http_get(&url)?;

    let got = format!("{:x}", Sha256::digest(&bytes));
    if got != artifact.sha256 {
        bail!(
            "sha256 mismatch for {}: got {got}, manifest says {}",
            artifact.file,
            artifact.sha256
        );
    }

    let parent = dir.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;

    let staging = parent.join(".litmus-models-staging");
    let backup = parent.join(".litmus-models-backup");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).context("creating staging dir")?;

    // .tar.zst (litmus already depends on zstd; preferred over xz for models).
    let decoder = zstd::Decoder::new(Cursor::new(&bytes)).context("opening zstd stream")?;
    tar::Archive::new(decoder)
        .unpack(&staging)
        .with_context(|| format!("extracting {}", artifact.file))?;

    // Validate before going live: load the staged bundle and force every
    // specialist route to construct its ONNX graph at least once.
    let staged_model = crate::model::Model::load(&staging, None, None).with_context(|| {
        format!(
            "staged bundle {} failed to load; not installing",
            artifact.file
        )
    })?;
    staged_model.validate_all_routes().with_context(|| {
        format!(
            "staged bundle {} has invalid specialist routes",
            artifact.file
        )
    })?;

    let meta = Installed {
        commit: artifact.commit.clone(),
        date: artifact.date.clone(),
        source: source.to_string(),
        version: our_version().to_string(),
    };
    let rendered = toml::to_string(&meta).context("rendering sidecar")?;
    std::fs::write(staging.join(SIDECAR), rendered).context("writing sidecar")?;

    // Atomic swap: move old aside, move staging in, drop old. Restore on failure.
    let _ = std::fs::remove_dir_all(&backup);
    if dir.exists() {
        std::fs::rename(dir, &backup).context("backing up current models")?;
    }
    if let Err(e) = std::fs::rename(&staging, dir) {
        if backup.exists() {
            let _ = std::fs::rename(&backup, dir);
        }
        return Err(e).context("installing models");
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
