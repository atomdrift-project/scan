//! Independently versioned NSRL Modern allowlist, published by Pulsar.
//!
//! Uses Burton's ADBL encoding in its own directory and key scheme. Scan's
//! measured bad and outside-claim filters always veto this source. The bundle
//! installs and refreshes with the other rules. An unchanged manifest costs
//! only a small HTTP request.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burton::{Filter, Manifest};
use reqwest::blocking::Client;

use crate::bundle::{self, Bundle};

/// NSRL's digest-only identity; deliberately separate from Scan's PURL bundle.
pub const KEY_SCHEME: &str = "nsrl/sha256/v1";
const PREFIX: &str = "nsrl/modern/v1";
/// The manifest, published and installed beside the filters.
const SIDECAR: &str = "bloom.toml";

/// Where the separately installed NSRL bundle lives.
///
/// # Errors
/// When there is no override and no data directory; see [`crate::data_dir`].
pub fn install_dir() -> anyhow::Result<PathBuf> {
    Ok(crate::data_dir()?.join("nsrl-modern"))
}

/// The installed NSRL identity, without loading its large bit array.
#[must_use]
pub fn installed_manifest() -> Option<Manifest> {
    let manifest = burton::build::read_manifest(&install_dir().ok()?)?;
    validate_manifest(&manifest).ok()?;
    Some(manifest)
}

/// Upstream NSRL release version, independent of Scan and the ADBL layout.
#[must_use]
pub fn installed_release() -> Option<String> {
    let text = std::fs::read_to_string(install_dir().ok()?.join(SIDECAR)).ok()?;
    source_release(&text)
}

fn source_release(text: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Version {
        release: String,
    }
    let version = toml::from_str::<Version>(text).ok()?.release;
    release_key(&version)?;
    Some(version)
}

fn release_key(version: &str) -> Option<[u32; 3]> {
    let parts: Vec<_> = version
        .split('.')
        .map(str::parse::<u32>)
        .collect::<Result<_, _>>()
        .ok()?;
    let parts: [u32; 3] = parts.try_into().ok()?;
    (parts[0] >= 2021 && (1..=12).contains(&parts[1]) && parts[2] > 0).then_some(parts)
}

/// Open an NSRL bundle, checking its explicit dataset identity.
pub(crate) fn open(dir: &Path) -> Result<burton::Lookup> {
    let manifest = burton::build::read_manifest(dir).context("reading NSRL manifest")?;
    validate_manifest(&manifest)?;
    burton::Lookup::open(dir, KEY_SCHEME).context("opening NSRL bundle")
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.schema != 1
        || manifest.key_scheme.as_deref() != Some(KEY_SCHEME)
        || manifest.filter.len() != 2
        || crate::output::parse_ymd(&manifest.built).is_none()
    {
        bail!("unsupported NSRL manifest identity or schema");
    }
    for (stem, must_be_empty) in [("sha256-good", false), ("sha256-bad", true)] {
        let entry = manifest.filter.get(stem).context("missing NSRL filter")?;
        if entry.format_version != 1
            || burton::parse_sha256_hex(&entry.sha256).is_none()
            || entry.file != format!("{stem}-{}.adbl", entry.sha256)
            || (entry.n == 0) != must_be_empty
        {
            bail!("invalid NSRL filter {stem}");
        }
    }
    Ok(())
}

/// Install or refresh NSRL independently of the daily Scan Bloom bundle.
/// Returns `true` when filters were installed.
///
/// # Errors
/// Returns download, integrity, format, or installation failures.
pub fn update(dir: &Path, force: bool, quiet: bool) -> Result<bool> {
    bundle::update::<Nsrl>(dir, force, quiet)
}

/// Check availability without downloading the filter bit arrays.
///
/// # Errors
/// Returns manifest download or validation errors.
pub fn check(dir: &Path) -> Result<()> {
    bundle::check::<Nsrl>(dir)
}

/// The NSRL filters, as [`bundle::update`] installs them.
struct Nsrl;

/// A validated manifest, with its text: the sidecar keeps the release and
/// provenance fields beyond Burton's shared filter schema.
struct Release {
    manifest: Manifest,
    text: String,
    release: [u32; 3],
}

impl Release {
    fn parse(text: String) -> Result<Self> {
        let manifest: Manifest = toml::from_str(&text).context("parsing NSRL manifest")?;
        validate_manifest(&manifest)?;
        let release = source_release(&text)
            .and_then(|version| release_key(&version))
            .context("missing or invalid NSRL release version")?;
        Ok(Self {
            manifest,
            text,
            release,
        })
    }
}

impl Bundle for Nsrl {
    const NAME: &'static str = "NSRL filters";
    type Installed = Release;
    type Remote = Release;

    fn installed(dir: &Path) -> Option<Release> {
        Release::parse(std::fs::read_to_string(dir.join(SIDECAR)).ok()?).ok()
    }

    fn fetch(client: &Client) -> Result<Release> {
        Release::parse(bundle::manifest(
            client,
            &bundle::url(&format!("{PREFIX}/{SIDECAR}")),
        )?)
    }

    fn is_current(dir: &Path, have: &Release, remote: &Release) -> bool {
        have.manifest.filter == remote.manifest.filter
            && have
                .manifest
                .filter
                .values()
                .all(|entry| dir.join(&entry.file).is_file())
    }

    /// Both orders must hold: the build date, and the upstream release number,
    /// which catches a same-month patch rollback the date alone would not.
    fn rollback(have: &Release, remote: &Release) -> Option<String> {
        let day = |m: &Manifest| crate::output::parse_ymd(&m.built);
        if day(&remote.manifest) < day(&have.manifest) {
            return Some(format!(
                "the published filters (built {}) are older than the installed ones (built {})",
                remote.manifest.built, have.manifest.built
            ));
        }
        (remote.release < have.release).then(|| {
            format!(
                "the published release {} is older than the installed {}",
                release_name(remote.release),
                release_name(have.release)
            )
        })
    }

    fn stage(staging: &Path, client: &Client, remote: &Release) -> Result<()> {
        stage(staging, client, &bundle::url(PREFIX), remote)
    }

    fn describe(remote: &Release) -> String {
        format!(
            "{} (built {})",
            release_name(remote.release),
            remote.manifest.built
        )
    }

    fn describe_installed(have: &Release) -> String {
        Self::describe(have)
    }
}

fn release_name([year, month, patch]: [u32; 3]) -> String {
    format!("{year}.{month:02}.{patch}")
}

/// Download, verify, and validate both filters into `staging` from `base`,
/// then write the manifest beside them, and prove the whole bundle opens.
fn stage(staging: &Path, client: &Client, base: &str, remote: &Release) -> Result<()> {
    for (stem, entry) in &remote.manifest.filter {
        // `validate_manifest` pinned the name to `<stem>-<sha256>.adbl`.
        let path = staging.join(&entry.file);
        let url = format!("{base}/{}", entry.file);
        bundle::download_to(client, &url, &entry.sha256, &path)?;
        let filter = Filter::load(std::fs::read(&path)?).context("invalid NSRL filter")?;
        if filter.stem() != *stem
            || filter.version() != entry.format_version
            || filter.len() != entry.n
        {
            bail!("NSRL filter header disagrees with manifest");
        }
    }
    std::fs::write(staging.join(SIDECAR), &remote.text).context("writing NSRL sidecar")?;
    open(staging)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_order_is_numeric_and_independent_of_layout_and_build_date() {
        assert!(release_key("2026.09.10") > release_key("2026.09.2"));
        assert!(release_key("2027.03.1") > release_key("2026.12.1"));
        assert!(release_key("2026.13.1").is_none());
        assert!(release_key("2026.09").is_none());
        assert_eq!(
            source_release("release = \"2026.09.1\"\n"),
            Some("2026.09.1".to_owned())
        );
    }

    fn release(built: &str, version: &str) -> Release {
        let entry = |stem: &str, n: u64| burton::build::Entry {
            file: format!("{stem}-{}.adbl", "a".repeat(64)),
            sha256: "a".repeat(64),
            format_version: 1,
            n,
        };
        let manifest = Manifest {
            schema: 1,
            built: built.to_owned(),
            key_scheme: Some(KEY_SCHEME.to_owned()),
            filter: [
                ("sha256-good".to_owned(), entry("sha256-good", 100)),
                ("sha256-bad".to_owned(), entry("sha256-bad", 0)),
            ]
            .into(),
        };
        let text = format!(
            "release = \"{version}\"\n{}",
            toml::to_string(&manifest).unwrap()
        );
        Release::parse(text).unwrap()
    }

    #[test]
    fn a_corrupt_filter_download_fails_staging() {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(&mut connection);
            let mut request = String::new();
            while !request.ends_with("\r\n\r\n") {
                assert!(reader.read_line(&mut request).unwrap() > 0);
            }
            connection
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nbad")
                .unwrap();
        });
        let staging = tempfile::tempdir().unwrap();
        let err = stage(
            staging.path(),
            &Client::new(),
            &base,
            &release("2026-09-01", "2026.09.1"),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("sha256 mismatch"), "{err:#}");
        server.join().unwrap();
    }

    #[test]
    fn an_older_build_or_release_is_a_rollback() {
        let have = release("2026-09-10", "2026.09.2");
        assert_eq!(
            Nsrl::rollback(&have, &release("2026-09-10", "2026.09.2")),
            None
        );
        assert_eq!(
            Nsrl::rollback(&have, &release("2026-10-01", "2026.10.1")),
            None
        );
        assert!(Nsrl::rollback(&have, &release("2026-09-01", "2026.09.2")).is_some());
        // A same-month patch rollback, rebuilt later.
        assert!(Nsrl::rollback(&have, &release("2026-09-20", "2026.09.1")).is_some());
    }

    #[test]
    fn reads_the_pulsar_go_builders_adbl_wire_vector() {
        // Same independent fixture as pulsar/cmd/nsrl-bloom's wire test.
        use sha2::{Digest, Sha256};
        let dir = tempfile::tempdir().unwrap();
        let good = vec![
            b'A', b'D', b'B', b'L', 1, 0, 3, 1, 10, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x20, 0x49, 0x92, 0x24, 0x01, 0, 0, 0,
        ];
        let mut bad = good.clone();
        bad[7] = 2;
        bad[20] = 0;
        bad[36..].fill(0);
        let mut manifest = Manifest {
            schema: 1,
            built: "2026-09-01".to_owned(),
            key_scheme: Some(KEY_SCHEME.to_owned()),
            filter: Default::default(),
        };
        for (stem, bytes, n) in [("sha256-good", good, 1), ("sha256-bad", bad, 0)] {
            let sha256 = format!("{:x}", Sha256::digest(&bytes));
            let file = format!("{stem}-{sha256}.adbl");
            std::fs::write(dir.path().join(&file), bytes).unwrap();
            manifest.filter.insert(
                stem.to_owned(),
                burton::build::Entry {
                    file,
                    sha256,
                    n,
                    format_version: 1,
                },
            );
        }
        std::fs::write(
            dir.path().join("bloom.toml"),
            toml::to_string(&manifest).unwrap(),
        )
        .unwrap();
        let lookup = open(dir.path()).unwrap();
        let mut digest = [0; 32];
        digest[0] = 5;
        digest[8] = 2;
        assert_eq!(
            lookup.decide(&burton::Artifact::sha256(&digest)),
            burton::Decision::Skip
        );
        digest[0] = 6;
        assert_eq!(
            lookup.decide(&burton::Artifact::sha256(&digest)),
            burton::Decision::Unknown
        );
    }

    #[test]
    fn accepts_only_content_addressed_digest_filters_from_nsrl() {
        let entry = |stem: &str, n: u64| burton::build::Entry {
            file: format!("{stem}-{}.adbl", "a".repeat(64)),
            sha256: "a".repeat(64),
            format_version: 1,
            n,
        };
        let mut manifest = Manifest {
            schema: 1,
            built: "2026-09-01".to_owned(),
            key_scheme: Some(KEY_SCHEME.to_owned()),
            filter: [
                ("sha256-good".to_owned(), entry("sha256-good", 100)),
                ("sha256-bad".to_owned(), entry("sha256-bad", 0)),
            ]
            .into(),
        };
        assert!(validate_manifest(&manifest).is_ok());
        manifest.key_scheme = Some(crate::bloom_repo::KEY_SCHEME.to_owned());
        assert!(validate_manifest(&manifest).is_err());
        manifest.key_scheme = Some(KEY_SCHEME.to_owned());
        manifest.filter.get_mut("sha256-good").unwrap().file = "../../bad.adbl".to_owned();
        assert!(validate_manifest(&manifest).is_err());
    }
}
