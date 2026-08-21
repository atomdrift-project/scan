//! R2-backed model updates.
//!
//! `scan update-rules` fetches a manifest from the update bucket, resolves the
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
/// Connect budget — fail fast when the bucket is unreachable so the default
/// auto-update can't stall a scan on an offline host (the whole-request
/// [`TIMEOUT`] still covers a slow but reachable download).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(4);

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

/// Same directory, comparing symlink-resolved paths when both exist.
fn same_dir(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// True when `dir` is the bundle a `SCAN_MODELS_DIR` override points at.
///
/// An explicit override means "load exactly this bundle" — a deploy validating
/// a candidate, a test pinning a fixture, a developer on a working tree — so
/// the updater treats it as read-only. [`is_git_managed`] cannot stand in for
/// this: collimator's deploy stages its candidate into a `mktemp -d`, which has
/// no `.git`, and a swap there silently replaces the candidate with the
/// currently published bundle. Every gate downstream then validates the *old*
/// models and the deploy mirrors them straight back (2026-08-21: a whole
/// nightly retrain shipped nothing but this module's sidecar).
fn is_pinned_override(dir: &Path) -> bool {
    std::env::var_os("SCAN_MODELS_DIR").is_some_and(|pinned| same_dir(Path::new(&pinned), dir))
}

/// Install or refresh the model bundle compatible with this litmus release.
pub fn update(dir: &Path, force: bool, quiet: bool) -> Result<()> {
    if is_pinned_override(dir) {
        if !quiet {
            eprintln!(
                "Models at {} come from SCAN_MODELS_DIR; leaving them untouched.\nUnset SCAN_MODELS_DIR to update the installed bundle.",
                dir.display()
            );
        }
        return Ok(());
    }
    if is_git_managed(dir) {
        if !quiet {
            eprintln!(
                "Models at {} are git-managed (a checkout or symlink to one); leaving them untouched.\nUse 'git pull' there to update, or remove the directory to switch to bundle updates.",
                dir.display()
            );
        }
        return Ok(());
    }
    // A quiet refresh of an already-present bundle is the auto-update path: fail
    // fast if the bucket is unreachable. A first-ever download — or any explicit,
    // non-quiet update — stays patient so a slow first fetch still completes.
    let connect = (quiet && installed(dir).is_some()).then_some(CONNECT_TIMEOUT);
    let manifest = fetch_manifest(connect)?;
    let (key, source) = resolve(&manifest)?;
    let artifact = artifact_for(&manifest, &key)?;

    if !force && installed(dir).is_some_and(|i| i.commit.starts_with(&key)) {
        if !quiet {
            eprintln!("Models already up to date: {} ({})", key, artifact.date);
            warn_if_behind(&manifest);
        }
        return Ok(());
    }

    if !quiet {
        eprintln!(
            "Installing models {} ({}) for Atomdrift Scan {} [{}]...",
            key,
            artifact.date,
            our_version(),
            source
        );
    }
    install(dir, artifact, &source)?;
    if !quiet {
        eprintln!(
            "Models updated to {} ({}) at {}",
            key,
            artifact.date,
            dir.display()
        );
        warn_if_behind(&manifest);
    }
    Ok(())
}

/// Report what would be installed without changing anything.
pub fn check(dir: &Path) -> Result<()> {
    if is_pinned_override(dir) {
        eprintln!(
            "Models at {} come from SCAN_MODELS_DIR; unset it to check the installed bundle.",
            dir.display()
        );
        return Ok(());
    }
    if is_git_managed(dir) {
        eprintln!(
            "Models at {} are git-managed; use 'git pull' there to update.",
            dir.display()
        );
        return Ok(());
    }
    let manifest = fetch_manifest(None)?;
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

fn fetch_manifest(connect: Option<Duration>) -> Result<Manifest> {
    let url = format!("{BASE_URL}/versions.toml");
    let text = http_get(&url, connect)?;
    let text = String::from_utf8(text).context("manifest is not valid UTF-8")?;
    toml::from_str(&text).with_context(|| format!("parsing manifest {url}"))
}

/// Download a bundle, verify its sha256, validate it loads, and atomically swap
/// it into `dir`. Validation happens on the staged copy *before* the swap, so a
/// broken bundle never replaces the live models.
fn install(dir: &Path, artifact: &Artifact, source: &str) -> Result<()> {
    let url = format!("{BASE_URL}/{}", artifact.file);
    // Patient: the manifest fetch already reached the bucket.
    let bytes = http_get(&url, None)?;

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::same_dir;
    use std::path::Path;

    #[test]
    fn same_dir_matches_identical_paths() {
        assert!(same_dir(Path::new("/tmp/bundle"), Path::new("/tmp/bundle")));
    }

    #[test]
    fn same_dir_rejects_unrelated_paths() {
        assert!(!same_dir(Path::new("/tmp/bundle"), Path::new("/tmp/other")));
    }

    #[test]
    #[cfg(unix)] // std::os::unix symlink; Windows symlink creation also needs privileges
    fn same_dir_resolves_symlinks() {
        // The deploy dir is a symlink to a checkout; a pin naming either spelling
        // must resolve to the same bundle.
        let root = tempfile::tempdir().expect("tempdir");
        let real = root.path().join("real");
        std::fs::create_dir(&real).expect("mkdir");
        let link = root.path().join("link");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        assert!(same_dir(&link, &real));
    }

    #[test]
    fn same_dir_is_lexical_when_paths_do_not_exist() {
        // Nonexistent paths can't be canonicalized; equal spellings still match,
        // unequal ones stay distinct rather than erroring.
        let a = Path::new("/nonexistent/a");
        assert!(same_dir(a, Path::new("/nonexistent/a")));
        assert!(!same_dir(a, Path::new("/nonexistent/b")));
    }
}
