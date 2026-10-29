//! Model bundle updates.
//!
//! The manifest (`versions.toml`) names a bundle per release; this build takes
//! its own `[stable]` pointer, else `latest`. The bundle is downloaded,
//! checked against its sha256, and loaded route by route *before* it replaces
//! the live models, so a broken bundle never goes live. See `crate::bundle`.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::bundle::{self, Bundle};

/// Sidecar recording what the R2 backend installed (no `.git` in the tree).
const SIDECAR: &str = ".litmus-models.toml";

/// This build's version, matched against the manifest's release keys (e.g. `2.0.0-rc.4`).
const OUR_VERSION: &str = env!("CARGO_PKG_VERSION");

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
    /// The scan version that performed the install.
    pub version: String,
}

/// Read the install sidecar, if the models dir was populated by the R2 backend.
#[must_use]
pub fn installed(dir: &Path) -> Option<Installed> {
    let text = fs::read_to_string(dir.join(SIDECAR)).ok()?;
    toml::from_str(&text).ok()
}

/// Install or refresh the model bundle compatible with this scan release.
/// Returns `true` when a bundle was installed.
///
/// # Errors
/// Returns an error if the manifest or bundle cannot be fetched, the sha256
/// mismatches, the staged bundle fails to load, or the install fails.
pub fn update(dir: &Path, force: bool, quiet: bool) -> Result<bool> {
    bundle::update::<Models>(dir, force, quiet)
}

/// Report what would be installed without changing anything.
///
/// # Errors
/// Returns an error if the manifest cannot be fetched or resolved.
pub fn check(dir: &Path) -> Result<()> {
    bundle::check::<Models>(dir)
}

/// The model bundle, as [`bundle::update`] installs it.
/// Most bytes a model bundle may unpack to. The download is capped, but zstd
/// is not: a small tarball can claim terabytes.
const MAX_UNPACKED_BYTES: u64 = 16 << 30;

/// Extract a model tarball into `staging`, accepting only what a model bundle
/// is: regular files and directories, each named once, by a relative path of
/// plain components. Links are refused outright — one named like the sidecar
/// would turn the metadata write after it into a write anywhere — and so is a
/// `.git`, which would mark the tree git-managed and stop every later update.
fn unpack(reader: impl Read, staging: &Path) -> Result<()> {
    use std::path::Component;
    let mut archive = tar::Archive::new(reader);
    // Never group- or world-writable, whatever modes the tarball carries.
    archive.set_mask(0o022);
    let mut seen = std::collections::HashSet::new();
    let mut total: u64 = 0;
    for entry in archive.entries().context("reading tar entries")? {
        let mut entry = entry.context("reading tar entry")?;
        let kind = entry.header().entry_type();
        if kind.is_pax_global_extensions() {
            std::io::copy(&mut entry, &mut std::io::sink()).context("reading a PAX header")?;
            continue;
        }
        let path = entry
            .path()
            .context("reading a tar entry path")?
            .into_owned();
        if !kind.is_file() && !kind.is_dir() {
            bail!("unsupported tar entry {kind:?} at {}", path.display());
        }
        // `./name` is how `tar -C dir .` spells a member; anything else that is
        // not a plain name (`..`, a root, a prefix) is refused.
        if !path
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
        {
            bail!("tar entry {} is not a plain relative path", path.display());
        }
        let path: std::path::PathBuf = path
            .components()
            .filter(|c| matches!(c, Component::Normal(_)))
            .collect();
        if path.as_os_str().is_empty() && kind.is_dir() {
            continue; // the archive's own `./`
        }
        if path.as_os_str().is_empty() || path.starts_with(".git") || path == Path::new(SIDECAR) {
            bail!("tar entry {} is reserved", path.display());
        }
        if !seen.insert(path.clone()) {
            bail!("tar entry {} appears twice", path.display());
        }
        total = total.saturating_add(entry.size());
        if total > MAX_UNPACKED_BYTES {
            bail!("bundle unpacks to more than {MAX_UNPACKED_BYTES} bytes");
        }
        entry
            .unpack_in(staging)
            .with_context(|| format!("extracting {}", path.display()))?;
    }
    Ok(())
}

struct Models;

/// The bundle this build should run, resolved from the manifest.
struct Release {
    /// Manifest artifact key, a content id.
    key: String,
    /// How the pointer was resolved; see [`Installed::source`].
    source: String,
    artifact: Artifact,
    /// A newer release that can run models this build cannot.
    upgrade: Option<String>,
}

impl Bundle for Models {
    const NAME: &'static str = "Models";
    type Installed = Installed;
    type Remote = Release;

    fn installed(dir: &Path) -> Option<Installed> {
        installed(dir)
    }

    fn fetch(client: &Client) -> Result<Release> {
        let url = bundle::url("versions.toml");
        let text = bundle::manifest(client, &url)?;
        let mut manifest: Manifest =
            toml::from_str(&text).with_context(|| format!("parsing manifest {url}"))?;
        // This build's own `[stable]` pointer, else `latest`.
        let (key, source) = match manifest.stable.remove(OUR_VERSION) {
            Some(key) => (key, format!("stable:{}", OUR_VERSION)),
            None if !manifest.latest.is_empty() => (manifest.latest, "latest".to_owned()),
            None => bail!("manifest has no pointer for this version and no `latest`"),
        };
        let artifact = manifest.artifacts.remove(&key).with_context(|| {
            format!("manifest references {key} but has no [artifacts.{key}] entry")
        })?;
        Ok(Release {
            upgrade: manifest.upgrade.remove(OUR_VERSION),
            key,
            source,
            artifact,
        })
    }

    fn is_current(_dir: &Path, have: &Installed, remote: &Release) -> bool {
        have.commit.starts_with(&remote.key)
    }

    /// Never: the pointer follows binary compatibility, not time. A build
    /// pinned to its `[stable]` bundle legitimately installs an older one
    /// than a newer build left behind.
    fn rollback(_have: &Installed, _remote: &Release) -> Option<String> {
        None
    }

    /// Unpack the verified `.tar.zst`, prove every route loads, and record
    /// the sidecar.
    fn stage(staging: &Path, client: &Client, remote: &Release) -> Result<()> {
        let artifact = &remote.artifact;
        let mut tarball = tempfile::tempfile().context("creating a download file")?;
        bundle::download(
            client,
            &bundle::url(&artifact.file),
            &artifact.sha256,
            &mut tarball,
        )?;
        tarball.rewind().context("rewinding the download")?;
        let decoder = zstd::Decoder::new(tarball).context("opening zstd stream")?;
        unpack(decoder, staging).with_context(|| format!("extracting {}", artifact.file))?;

        // Force every specialist route to construct its ONNX graph at least once.
        crate::model::Model::load(staging, None, None)
            .with_context(|| format!("staged bundle {} failed to load", artifact.file))?
            .validate_all_routes()
            .with_context(|| {
                format!(
                    "staged bundle {} has invalid specialist routes",
                    artifact.file
                )
            })?;

        let meta = Installed {
            commit: artifact.commit.clone(),
            date: artifact.date.clone(),
            source: remote.source.clone(),
            version: OUR_VERSION.to_owned(),
        };
        let rendered = toml::to_string(&meta).context("rendering sidecar")?;
        // `create_new`: `unpack` refuses a bundle entry by this name, and this
        // must never write through whatever is there instead.
        let mut sidecar = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(staging.join(SIDECAR))
            .context("creating sidecar")?;
        sidecar
            .write_all(rendered.as_bytes())
            .context("writing sidecar")
    }

    fn describe(remote: &Release) -> String {
        format!(
            "{} ({}) [{}]",
            remote.key, remote.artifact.date, remote.source
        )
    }

    fn describe_installed(have: &Installed) -> String {
        let commit: String = have.commit.chars().take(12).collect();
        format!("{commit} ({})", have.date)
    }

    /// A newer *release* runs models this build can't. The manifest's
    /// `[upgrade]` table already leaves out dev and unlisted builds.
    fn note(remote: &Release) -> Option<String> {
        remote.upgrade.as_ref().map(|target| {
            format!(
                "Note: Atomdrift Scan {} cannot use the newest models. Upgrade to Atomdrift Scan {target} for the latest detections.",
                OUR_VERSION
            )
        })
    }
}
