//! R2-backed model updates, and the download-and-install plumbing the bloom
//! updater ([`crate::bloom_update`]) shares.
//!
//! `scan update-rules` fetches a manifest from the update bucket, resolves the
//! model bundle compatible with *this* litmus release, downloads it, verifies its
//! sha256, validates it by loading, and installs it.
//!
//! Nothing here is signed. The manifest's per-artifact sha256 catches a corrupt
//! or truncated download, not a compromised bucket: trust is HTTPS to our own
//! bucket.
//!
//! The extracted bundle is validated with [`crate::model::Model::load`]
//! *before* it replaces the live models, so a broken bundle never goes live.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Cursor};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Base URL for the public update bucket (`<base>/versions.toml`,
/// `<base>/models/...`, `<base>/bloom/v<N>/...`).
pub(crate) const BASE_URL: &str = "https://updates.atomdrift.org/litmus";

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
    let text = fs::read_to_string(dir.join(SIDECAR)).ok()?;
    toml::from_str(&text).ok()
}

/// Install or refresh the model bundle compatible with this litmus release.
/// Returns `true` when a bundle was installed.
///
/// # Errors
/// Returns an error if the manifest or bundle cannot be fetched, the sha256
/// mismatches, the staged bundle fails to load, or the install fails.
pub fn update(dir: &Path, force: bool, quiet: bool) -> Result<bool> {
    if let Some(why) = hands_off("Models", dir, "SCAN_MODELS_DIR") {
        if !quiet {
            eprintln!("{why}");
        }
        return Ok(false);
    }
    let client = client(quiet && installed(dir).is_some())?;
    let manifest = fetch_manifest(&client)?;
    let (key, source) = resolve(&manifest)?;
    let artifact = artifact_for(&manifest, &key)?;

    let current = || !force && installed(dir).is_some_and(|i| i.commit.starts_with(&key));
    if current() {
        if !quiet {
            eprintln!("Models already up to date: {} ({})", key, artifact.date);
            warn_if_behind(&manifest);
        }
        return Ok(false);
    }
    let installer = Installer::lock(dir)?;
    if current() {
        return Ok(false); // another updater installed it while we waited
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
    let url = format!("{BASE_URL}/{}", artifact.file);
    let bytes = get_verified(&client, &url, &artifact.sha256)?;
    installer.install(|staging| stage(staging, &bytes, artifact, &source))?;
    if !quiet {
        eprintln!(
            "Models updated to {} ({}) at {}",
            key,
            artifact.date,
            dir.display()
        );
        warn_if_behind(&manifest);
    }
    Ok(true)
}

/// Report what would be installed without changing anything.
///
/// # Errors
/// Returns an error if the manifest cannot be fetched or resolved.
pub fn check(dir: &Path) -> Result<()> {
    if let Some(why) = hands_off("Models", dir, "SCAN_MODELS_DIR") {
        eprintln!("{why}");
        return Ok(());
    }
    let manifest = fetch_manifest(&client(false)?)?;
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

fn fetch_manifest(client: &Client) -> Result<Manifest> {
    let url = format!("{BASE_URL}/versions.toml");
    let text = String::from_utf8(get(client, &url)?).context("manifest is not valid UTF-8")?;
    toml::from_str(&text).with_context(|| format!("parsing manifest {url}"))
}

/// Unpack the verified `.tar.zst` into `staging`, prove it loads, and record
/// the sidecar. Validation runs here, before the swap, so a broken bundle
/// never replaces the live models.
fn stage(staging: &Path, bytes: &[u8], artifact: &Artifact, source: &str) -> Result<()> {
    let decoder = zstd::Decoder::new(Cursor::new(bytes)).context("opening zstd stream")?;
    tar::Archive::new(decoder)
        .unpack(staging)
        .with_context(|| format!("extracting {}", artifact.file))?;

    // Force every specialist route to construct its ONNX graph at least once.
    let staged_model = crate::model::Model::load(staging, None, None).with_context(|| {
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
    fs::write(staging.join(SIDECAR), rendered).context("writing sidecar")
}

// --- plumbing shared with crate::bloom_update --------------------------------

/// Whole-request budget: model bundles and the SHA-256 bloom filter run to tens
/// of MB.
const TIMEOUT: Duration = Duration::from_secs(120);
/// Connect budget for a quiet refresh of an installed bundle, so the default
/// auto-update can't stall a scan on an offline host.
const QUICK_CONNECT_TIMEOUT: Duration = Duration::from_secs(4);

/// The HTTP client for one update run, shared by its manifest and bundle
/// fetches. `quick` fails fast when the bucket is unreachable; a first install
/// or an explicit update stays patient so a slow first fetch still completes.
pub(crate) fn client(quick: bool) -> Result<Client> {
    let mut builder = Client::builder().timeout(TIMEOUT);
    if quick {
        builder = builder.connect_timeout(QUICK_CONNECT_TIMEOUT);
    }
    builder.build().context("building update http client")
}

/// GET `url`'s body. An HTTP error status is an error whose chain carries the
/// [`reqwest::Error`], so a caller can test for a 404.
pub(crate) fn get(client: &Client, url: &str) -> Result<Vec<u8>> {
    tracing::debug!("fetching {url}");
    let resp = client
        .get(url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .with_context(|| format!("GET {url}"))?;
    Ok(resp
        .bytes()
        .with_context(|| format!("reading {url}"))?
        .into())
}

/// GET `url` and check its body against `sha256` (lowercase hex). That catches
/// a corrupt or truncated download, not a hostile bucket: the manifest the
/// digest comes from is unsigned.
pub(crate) fn get_verified(client: &Client, url: &str, sha256: &str) -> Result<Vec<u8>> {
    let bytes = get(client, url)?;
    let got = format!("{:x}", Sha256::digest(&bytes));
    if got != sha256 {
        bail!("sha256 mismatch for {url}: got {got}, manifest says {sha256}");
    }
    Ok(bytes)
}

/// Why the updater must leave `dir` alone, as a line for the user; `None` when
/// it may install there. `what` names the bundle ("Models"); `pin_var` is the
/// env var that selects it.
///
/// An explicit `pin_var` means "use exactly this bundle" — a deploy validating
/// a candidate, a test pinning a fixture, a developer's hand-built set — so the
/// updater treats it as read-only. A git check cannot stand in for this:
/// collimator's deploy stages its candidate into a `mktemp -d`, which has no
/// `.git`, and a swap there silently replaced the candidate with the published
/// bundle (2026-08-21: a whole nightly retrain shipped nothing but the
/// sidecar). A git checkout, or a symlink into one, is a dev tree that git
/// updates.
pub(crate) fn hands_off(what: &str, dir: &Path, pin_var: &str) -> Option<String> {
    if std::env::var_os(pin_var).is_some_and(|pin| same_dir(Path::new(&pin), dir)) {
        return Some(format!(
            "{what} at {} come from {pin_var}; leaving them untouched.\nUnset {pin_var} to update the installed copy.",
            dir.display()
        ));
    }
    // `exists` follows a symlinked `dir`, so a link into a checkout counts.
    if dir.join(".git").exists() {
        return Some(format!(
            "{what} at {} are git-managed (a checkout or symlink to one); leaving them untouched.\nUse 'git pull' there to update, or remove the directory to switch to bundle updates.",
            dir.display()
        ));
    }
    None
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

/// The right to replace one install directory, held across processes.
///
/// Several updaters can target one directory at once (a scan's auto-update,
/// `update-rules`, a server's `/_/update`, a worker's renewal). The lock, a
/// `.<name>.lock` file beside the target, serializes their staging and swap.
/// Readers don't take it; see [`settle`].
#[derive(Debug)]
pub(crate) struct Installer {
    target: PathBuf,
    /// `None` where the platform has no file locking; updaters then race as
    /// they did before this lock existed.
    lock: Option<File>,
}

impl Installer {
    /// Wait for, then take, `target`'s install lock.
    pub(crate) fn lock(target: &Path) -> Result<Self> {
        let (parent, _) = split(target)?;
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        let path = lock_path(target)?;
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let lock = match file.lock() {
            Ok(()) => Some(file),
            Err(e) if e.kind() == io::ErrorKind::Unsupported => {
                tracing::debug!(path = %path.display(), "no file locking here; installing unlocked");
                None
            }
            Err(e) => return Err(e).with_context(|| format!("locking {}", path.display())),
        };
        Ok(Self {
            target: target.to_owned(),
            lock,
        })
    }

    /// Replace the target with the bundle `fill` writes into an empty staging
    /// directory beside it: fill, flush to disk, swap. On any failure the
    /// target keeps its old contents.
    ///
    /// The swap is two renames (old out, new in), so for an instant the target
    /// does not exist. Moving to a `current` symlink would close that, but the
    /// readers in other modules open the directory by path, and symlinks need
    /// privileges on Windows. Readers that find nothing call [`settle`].
    pub(crate) fn install(self, fill: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
        let (parent, name) = split(&self.target)?;
        let prefix = format!(".{name}.staging-");
        if self.lock.is_some() {
            // Holding the lock, no other updater is staging: anything with our
            // prefix was left by one that died mid-install (SIGKILL, Ctrl-C).
            for entry in fs::read_dir(parent).into_iter().flatten().flatten() {
                if entry.file_name().to_string_lossy().starts_with(&prefix) {
                    let _ = fs::remove_dir_all(entry.path());
                }
            }
        }
        let tempdir = || {
            tempfile::Builder::new()
                .prefix(&prefix)
                .tempdir_in(parent)
                .with_context(|| format!("creating a staging dir in {}", parent.display()))
        };

        let staging = tempdir()?;
        fill(staging.path())?;
        sync_tree(staging.path()).context("flushing the staged bundle to disk")?;

        // The old bundle moves into `trash`, which deletes it on drop.
        let trash = tempdir()?;
        let old = trash.path().join(name);
        let had_old = match fs::rename(&self.target, &old) {
            Ok(()) => true,
            Err(e) if e.kind() == io::ErrorKind::NotFound => false,
            Err(e) => {
                return Err(e).with_context(|| format!("moving {} aside", self.target.display()));
            }
        };
        if let Err(e) = fs::rename(staging.path(), &self.target) {
            if had_old {
                let _ = fs::rename(&old, &self.target);
            }
            return Err(e).with_context(|| format!("installing {}", self.target.display()));
        }
        let _ = staging.keep(); // it is the target now
        fsync(parent).with_context(|| format!("flushing {}", parent.display()))
    }
}

/// Wait out any install in progress at `dir`; `false` when no updater has
/// ever run there. A reader that finds `dir` missing calls this and looks
/// again, so the instant between [`Installer::install`]'s two renames never
/// reads as "not installed".
pub(crate) fn settle(dir: &Path) -> bool {
    let Ok(path) = lock_path(dir) else {
        return false;
    };
    File::open(path).is_ok_and(|file| file.lock_shared().is_ok())
}

/// `target`'s parent directory and file name.
fn split(target: &Path) -> Result<(&Path, String)> {
    let name = target
        .file_name()
        .with_context(|| format!("install target {} has no name", target.display()))?;
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok((parent, name.to_string_lossy().into_owned()))
}

fn lock_path(target: &Path) -> Result<PathBuf> {
    let (parent, name) = split(target)?;
    Ok(parent.join(format!(".{name}.lock")))
}

/// Flush every file and directory under `dir` to disk, so a crash after the
/// swap can't leave an installed bundle of empty or torn files.
fn sync_tree(dir: &Path) -> Result<()> {
    for entry in walkdir::WalkDir::new(dir) {
        let entry = entry?;
        if entry.file_type().is_file() || entry.file_type().is_dir() {
            fsync(entry.path())?;
        }
    }
    Ok(())
}

/// fsync a file or directory. Unix only: on Windows, flushing needs a writable
/// handle, which a read-only extracted file refuses, and a directory can't be
/// opened as a file.
fn fsync(path: &Path) -> io::Result<()> {
    if cfg!(unix) {
        File::open(path)?.sync_all()
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        fs::create_dir(&real).expect("mkdir");
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

    #[test]
    fn a_pinned_dir_is_left_alone() {
        // The pin is whatever directory the named variable holds. Cargo sets
        // CARGO_MANIFEST_DIR for test runs, so it stands in for SCAN_BLOOM_DIR.
        let pinned = Path::new(env!("CARGO_MANIFEST_DIR"));
        let why = hands_off("Bloom filters", pinned, "CARGO_MANIFEST_DIR").unwrap();
        assert!(why.contains("come from CARGO_MANIFEST_DIR"), "{why}");
        let elsewhere = tempfile::tempdir().unwrap();
        assert_eq!(
            hands_off("Bloom filters", elsewhere.path(), "CARGO_MANIFEST_DIR"),
            None
        );
    }

    #[test]
    fn a_git_checkout_is_left_alone() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("bundle");
        fs::create_dir_all(dir.join(".git")).unwrap();
        let why = hands_off("Models", &dir, "SCAN_TEST_PIN_THAT_IS_NEVER_SET").unwrap();
        assert!(why.contains("git-managed"), "{why}");
        fs::remove_dir(dir.join(".git")).unwrap();
        assert_eq!(
            hands_off("Models", &dir, "SCAN_TEST_PIN_THAT_IS_NEVER_SET"),
            None
        );
    }

    fn files(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn install_replaces_the_target_and_leaves_only_the_lock_behind() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("bundle");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("old.bin"), b"old").unwrap();
        // A staging dir abandoned by an updater that was killed mid-install.
        fs::create_dir(root.path().join(".bundle.staging-dead")).unwrap();

        Installer::lock(&target)
            .unwrap()
            .install(|staging| Ok(fs::write(staging.join("new.bin"), b"new")?))
            .unwrap();

        assert_eq!(files(&target), ["new.bin"]);
        assert_eq!(files(root.path()), [".bundle.lock", "bundle"]);
        assert!(settle(&target), "an updater ran here");
    }

    #[test]
    fn a_failed_fill_keeps_the_old_bundle() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("bundle");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("old.bin"), b"old").unwrap();

        let err = Installer::lock(&target)
            .unwrap()
            .install(|staging| {
                fs::write(staging.join("half.bin"), b"half")?;
                bail!("validation failed")
            })
            .unwrap_err();

        assert!(format!("{err:#}").contains("validation failed"));
        assert_eq!(files(&target), ["old.bin"]);
        assert_eq!(files(root.path()), [".bundle.lock", "bundle"]);
    }

    #[test]
    fn a_first_install_creates_the_target() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("nested").join("bundle");
        assert!(!settle(&target), "no updater has run here yet");
        Installer::lock(&target)
            .unwrap()
            .install(|staging| Ok(fs::write(staging.join("f"), b"x")?))
            .unwrap();
        assert_eq!(files(&target), ["f"]);
    }
}
