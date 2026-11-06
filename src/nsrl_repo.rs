//! Independently versioned NSRL Modern allowlist, published by Pulsar.
//!
//! Uses Burton's ADBL encoding in its own directory and key scheme. Scan's
//! measured bad and outside-claim filters always veto this source. The bundle
//! installs and refreshes with the other rules. An unchanged manifest costs
//! only a small HTTP request.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burton::{Kind, MANIFEST_FILE, Manifest, Tier};

use crate::bundle::{self, Bundle};

/// NSRL's digest-only identity; deliberately separate from Scan's PURL bundle.
pub(crate) const KEY_SCHEME: &str = "nsrl/sha256/v1";
/// Moves with the bundle format, so a build never fetches a bundle it cannot
/// read: `v1` carries the format-1 bundles pre-release builds read.
const PREFIX: &str = "nsrl/modern/v2";

/// Where the separately installed NSRL bundle lives.
///
/// # Errors
/// When there is no override and no data directory; see [`crate::data_dir`].
pub(crate) fn install_dir() -> anyhow::Result<PathBuf> {
    Ok(crate::data_dir()?.join("nsrl-modern"))
}

/// The installed NSRL identity, without loading its large bit array.
#[must_use]
pub fn installed_manifest() -> Option<Manifest> {
    let manifest = burton::read_manifest(&install_dir().ok()?)
        .inspect_err(|error| tracing::warn!(%error, "NSRL manifest unreadable"))
        .ok()??;
    validate_manifest(&manifest).ok()?;
    Some(manifest)
}

/// Upstream NSRL release version, independent of Scan and the ADBL layout.
#[must_use]
pub fn installed_release() -> Option<String> {
    let manifest = burton::read_manifest(install_dir().ok()?).ok()??;
    source_release(&manifest).map(str::to_owned)
}

/// The NIST release Pulsar recorded in the manifest's provenance, if valid.
fn source_release(manifest: &Manifest) -> Option<&str> {
    let version = manifest.provenance.get("release")?.as_str()?;
    release_key(version)?;
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
    let manifest = burton::read_manifest(dir)
        .context("reading NSRL manifest")?
        .context("no NSRL manifest")?;
    validate_manifest(&manifest)?;
    burton::Lookup::open(dir, KEY_SCHEME).context("opening NSRL bundle")
}

/// NSRL vouches for digests and nothing else: every slot of the bundle is
/// present, as burton requires, but only `sha256-good` holds keys. Each file
/// is named for its content, so a client mid-update never mixes releases.
fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.schema != burton::MANIFEST_SCHEMA
        || manifest.key_scheme != KEY_SCHEME
        || manifest.filter.len() != Kind::ALL.len() * Tier::ALL.len()
        || crate::output::parse_ymd(&manifest.built).is_none()
    {
        bail!("unsupported NSRL manifest identity or schema");
    }
    for kind in Kind::ALL {
        for tier in Tier::ALL {
            let stem = format!("{kind}-{tier}");
            let entry = manifest.filter.get(&stem).context("missing NSRL filter")?;
            let must_be_empty = (kind, tier) != (Kind::Sha256, Tier::Good);
            if burton::parse_sha256_hex(&entry.sha256).is_none()
                || entry.file != format!("{stem}-{}.adbl", entry.sha256)
                || (entry.n == 0) != must_be_empty
            {
                bail!("invalid NSRL filter {stem}");
            }
        }
    }
    Ok(())
}

/// Install or refresh NSRL independently of the daily Scan Bloom bundle.
/// Returns `true` when filters were installed.
///
/// # Errors
/// Returns download, integrity, format, or installation failures.
pub(crate) fn update(dir: &Path, force: bool, quiet: bool) -> Result<bool> {
    bundle::update::<Nsrl>(dir, force, quiet)
}

/// The NSRL filters, as [`bundle::update`] installs them.
struct Nsrl;

/// A validated manifest, with its text as published, written back verbatim
/// as the sidecar.
struct Release {
    manifest: Manifest,
    text: String,
    release: [u32; 3],
}

impl Release {
    fn parse(text: String) -> Result<Self> {
        let manifest: Manifest = toml::from_str(&text).context("parsing NSRL manifest")?;
        validate_manifest(&manifest)?;
        let release = source_release(&manifest)
            .and_then(release_key)
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
        Release::parse(std::fs::read_to_string(dir.join(MANIFEST_FILE)).ok()?).ok()
    }

    fn fetch(http: &bundle::Http) -> Result<Release> {
        Release::parse(bundle::manifest(
            http,
            &bundle::url(&format!("{PREFIX}/{MANIFEST_FILE}")),
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

    fn stage(staging: &Path, http: &bundle::Http, remote: &Release) -> Result<()> {
        stage(staging, http, &bundle::url(PREFIX), remote)
    }

    fn describe(remote: &Release) -> String {
        format!(
            "{} (built {})",
            release_name(remote.release),
            remote.manifest.built
        )
    }
}

fn release_name([year, month, patch]: [u32; 3]) -> String {
    format!("{year}.{month:02}.{patch}")
}

/// Download, verify, and validate both filters into `staging` from `base`,
/// then write the manifest beside them, and prove the whole bundle opens.
fn stage(staging: &Path, http: &bundle::Http, base: &str, remote: &Release) -> Result<()> {
    bundle::stage_filters(staging, http, base, &remote.manifest)?;
    std::fs::write(staging.join(MANIFEST_FILE), &remote.text).context("writing NSRL sidecar")?;
    open(staging)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every slot of an NSRL manifest, content-addressed, with `sha256-good`
    /// holding `good` keys and the rest empty.
    fn manifest(built: &str, version: &str, good: u64) -> Manifest {
        let mut filter = std::collections::BTreeMap::new();
        for kind in Kind::ALL {
            for tier in Tier::ALL {
                let stem = format!("{kind}-{tier}");
                let n = if (kind, tier) == (Kind::Sha256, Tier::Good) {
                    good
                } else {
                    0
                };
                let sha256 = "a".repeat(64);
                let file = format!("{stem}-{sha256}.adbl");
                filter.insert(stem, burton::Entry { file, sha256, n });
            }
        }
        Manifest {
            schema: burton::MANIFEST_SCHEMA,
            built: built.to_owned(),
            key_scheme: KEY_SCHEME.to_owned(),
            provenance: toml::from_str(&format!("release = \"{version}\"")).unwrap(),
            filter,
        }
    }

    fn release(built: &str, version: &str) -> Release {
        Release::parse(toml::to_string(&manifest(built, version, 100)).unwrap()).unwrap()
    }

    #[test]
    fn release_order_is_numeric_and_independent_of_layout_and_build_date() {
        assert!(release_key("2026.09.10") > release_key("2026.09.2"));
        assert!(release_key("2027.03.1") > release_key("2026.12.1"));
        assert!(release_key("2026.13.1").is_none());
        assert!(release_key("2026.09").is_none());
        let m = manifest("2026-09-01", "2026.09.1", 1);
        assert_eq!(source_release(&m), Some("2026.09.1"));
        let mut unreleased = m;
        unreleased.provenance.clear();
        assert_eq!(source_release(&unreleased), None);
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
            &bundle::Http::once(reqwest::blocking::Client::new()),
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
        // The vector pulsar/cmd/nsrl-bloom's wire test and burton's
        // matches_the_shared_wire_vector share, computed from burton's
        // FORMAT.md by a separate implementation: format 3, a one-block
        // `sha256-good` holding the key with lanes 5, 2, 0, 0, and every other
        // slot empty.
        use sha2::{Digest, Sha256};
        let dir = tempfile::tempdir().unwrap();
        let mut good = vec![0u8; 2 * 4096];
        good[..21].copy_from_slice(&[
            b'A', b'D', b'B', b'L', 3, 0, 3, 1, 10, 0, 0, 0, 0, 0x80, 0, 0, 0, 0, 0, 0, 1,
        ]);
        for (at, byte) in [
            (280, 4),
            (2153, 8),
            (2317, 32),
            (2432, 1),
            (3018, 4),
            (3290, 64),
            (3741, 32),
            (3755, 1),
            (3942, 4),
            (4057, 128),
        ] {
            good[4096 + at] = byte;
        }
        let mut manifest = manifest("2026-09-01", "2026.09.1", 1);
        for (kind_byte, kind) in [(1u8, Kind::Purl), (3, Kind::Sha256)] {
            for (tier_byte, tier) in [
                (1u8, Tier::Good),
                (2, Tier::Bad),
                (4, Tier::SightedHostile),
                (5, Tier::SightedSuspicious),
            ] {
                let mut bytes = good.clone();
                if (kind, tier) != (Kind::Sha256, Tier::Good) {
                    bytes[6] = kind_byte;
                    bytes[7] = tier_byte;
                    bytes[20] = 0;
                    bytes[4096..].fill(0);
                }
                let stem = format!("{kind}-{tier}");
                let sha256 = format!("{:x}", Sha256::digest(&bytes));
                let file = format!("{stem}-{sha256}.adbl");
                std::fs::write(dir.path().join(&file), bytes).unwrap();
                let entry = manifest.filter.get_mut(&stem).unwrap();
                entry.file = file;
                entry.sha256 = sha256;
            }
        }
        std::fs::write(
            dir.path().join(MANIFEST_FILE),
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
        let mut manifest = manifest("2026-09-01", "2026.09.1", 100);
        assert!(validate_manifest(&manifest).is_ok());
        manifest.key_scheme = crate::bloom_repo::KEY_SCHEME.to_owned();
        assert!(validate_manifest(&manifest).is_err());
        manifest.key_scheme = KEY_SCHEME.to_owned();
        manifest.filter.get_mut("sha256-good").unwrap().file = "../../bad.adbl".to_owned();
        assert!(validate_manifest(&manifest).is_err());
    }

    /// NSRL holds digests only: keys in any other slot, or a missing slot,
    /// mean the bundle is not the one Pulsar publishes.
    #[test]
    fn an_nsrl_bundle_keys_only_sha256_good_and_has_every_slot() {
        let mut keyed = manifest("2026-09-01", "2026.09.1", 100);
        keyed.filter.get_mut("purl-good").unwrap().n = 3;
        assert!(validate_manifest(&keyed).is_err());
        let mut partial = manifest("2026-09-01", "2026.09.1", 100);
        partial.filter.remove("sha256-sighted-suspicious");
        assert!(validate_manifest(&partial).is_err());
        assert!(validate_manifest(&manifest("2026-09-01", "2026.09.1", 0)).is_err());
    }
}
