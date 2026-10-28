//! Independently versioned NSRL Modern allowlist, published by Pulsar.
//!
//! Uses Burton's ADBL encoding in its own directory and key scheme. Scan's
//! measured bad and outside-claim filters always veto this source. The bundle
//! installs and refreshes with the other rules. An unchanged manifest costs
//! only a small HTTP request. SCAN_NSRL=0 disables this source.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use burton::{Filter, Manifest};
use reqwest::blocking::Client;

use crate::model_update::{Installer, hands_off};

/// NSRL's digest-only identity; deliberately separate from Scan's PURL bundle.
pub const KEY_SCHEME: &str = "nsrl/sha256/v1";
const PREFIX: &str = "nsrl/modern/v1";

/// Where the separately installed NSRL bundle lives.
#[must_use]
pub fn install_dir() -> PathBuf {
    std::env::var_os("SCAN_NSRL_DIR").map_or_else(
        || {
            dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("atomdrift/scan/nsrl-modern")
        },
        PathBuf::from,
    )
}

/// The installed NSRL identity, without loading its large bit array.
#[must_use]
pub fn installed_manifest() -> Option<Manifest> {
    let manifest = burton::build::read_manifest(&install_dir())?;
    validate_manifest(&manifest).ok()?;
    Some(manifest)
}

/// Upstream NSRL release version, independent of Scan and the ADBL layout.
#[must_use]
pub fn installed_release() -> Option<String> {
    let text = std::fs::read_to_string(install_dir().join("bloom.toml")).ok()?;
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

/// Whether installed NSRL coverage participates in decisions.
#[must_use]
pub fn enabled() -> bool {
    std::env::var("SCAN_NSRL").as_deref() != Ok("0")
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

fn same_filters(a: &Manifest, b: &Manifest) -> bool {
    a.key_scheme == b.key_scheme && a.filter == b.filter
}

fn client(quiet: bool) -> Result<Client> {
    let mut client = Client::builder().timeout(Duration::from_secs(900));
    if quiet {
        client = client.connect_timeout(Duration::from_secs(4));
    }
    client.build().context("creating NSRL HTTP client")
}

fn base_url() -> String {
    let base =
        std::env::var("SCAN_NSRL_URL").unwrap_or_else(|_| crate::model_update::BASE_URL.to_owned());
    format!("{}/{PREFIX}", base.trim_end_matches('/'))
}

fn remote_manifest(client: &Client, base: &str) -> Result<(Manifest, String)> {
    let bytes = crate::model_update::get(client, &format!("{base}/bloom.toml"))?;
    let text = String::from_utf8(bytes)?;
    let manifest = toml::from_str(&text)?;
    validate_manifest(&manifest)?;
    source_release(&text).context("missing or invalid NSRL release version")?;
    Ok((manifest, text))
}

/// Install or refresh NSRL independently of the daily Scan Bloom bundle.
///
/// A complete staged bundle is validated before replacing the installed copy.
/// The normal rules updater includes first install as well as refreshes.
///
/// # Errors
/// Returns download, integrity, format, or installation failures.
pub fn update(dir: &Path, quiet: bool) -> Result<bool> {
    if let Some(why) = hands_off("NSRL filters", dir, "SCAN_NSRL_DIR") {
        if !quiet {
            eprintln!("{why}");
        }
        return Ok(false);
    }
    let client = client(quiet)?;
    let base = base_url();
    let (manifest, text) = remote_manifest(&client, &base)?;
    let current = || {
        burton::build::read_manifest(dir).is_some_and(|have| {
            same_filters(&have, &manifest)
                && have
                    .filter
                    .values()
                    .all(|entry| dir.join(&entry.file).is_file())
        })
    };
    if current() {
        if !quiet {
            eprintln!("NSRL already up to date: {}", manifest.built);
        }
        return Ok(false);
    }
    let installer = Installer::lock(dir)?;
    if current() {
        return Ok(false);
    }
    if let Some(have) = burton::build::read_manifest(dir)
        && have.built > manifest.built
    {
        bail!(
            "refusing NSRL rollback from {} to {}",
            have.built,
            manifest.built
        );
    }
    if let Some(previous) = std::fs::read_to_string(dir.join("bloom.toml"))
        .ok()
        .and_then(|text| source_release(&text))
        .and_then(|version| release_key(&version))
        && let Some(next) = source_release(&text).and_then(|version| release_key(&version))
        && next < previous
    {
        bail!("refusing an older NSRL release, including same-month patch rollbacks");
    }
    installer.install(|staging| stage(staging, &client, &base, &manifest, &text))?;
    if !quiet {
        eprintln!("NSRL updated to {} at {}", manifest.built, dir.display());
    }
    Ok(true)
}

fn stage(
    staging: &Path,
    client: &Client,
    base: &str,
    manifest: &Manifest,
    text: &str,
) -> Result<()> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Write};

    validate_manifest(manifest)?;
    for (stem, entry) in &manifest.filter {
        // Stream to disk while hashing, avoiding a second 512 MiB HTTP buffer.
        let path = staging.join(&entry.file);
        let mut response = client
            .get(format!("{base}/{}", entry.file))
            .send()?
            .error_for_status()?;
        let mut output = std::fs::File::create(&path)?;
        let mut digest = Sha256::new();
        let mut buffer = vec![0; 1 << 20];
        loop {
            let n = response.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            output.write_all(&buffer[..n])?;
            digest.update(&buffer[..n]);
        }
        drop(output);
        if format!("{:x}", digest.finalize()) != entry.sha256 {
            bail!("NSRL checksum mismatch for {}", entry.file);
        }
        let filter = Filter::load(std::fs::read(path)?).context("invalid NSRL filter")?;
        if filter.stem() != *stem
            || filter.version() != entry.format_version
            || filter.len() != entry.n
        {
            bail!("NSRL filter header disagrees with manifest");
        }
    }
    // Preserve release/provenance fields beyond Burton's shared filter schema.
    std::fs::write(staging.join("bloom.toml"), text)?;
    open(staging)?;
    Ok(())
}

/// Check availability without downloading the filter bit arrays.
///
/// # Errors
/// Returns manifest download or validation errors.
pub fn check(dir: &Path) -> Result<()> {
    if let Some(why) = hands_off("NSRL filters", dir, "SCAN_NSRL_DIR") {
        eprintln!("{why}");
        return Ok(());
    }
    let (remote, _) = remote_manifest(&client(false)?, &base_url())?;
    let current =
        burton::build::read_manifest(dir).is_some_and(|have| same_filters(&have, &remote));
    eprintln!(
        "NSRL {}: {}",
        if current { "up to date" } else { "available" },
        remote.built
    );
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

    #[test]
    fn failed_filter_download_keeps_the_previous_installation() {
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
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("installed");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("previous"), "keep me").unwrap();
        let entry = |stem: &str, n: u64| burton::build::Entry {
            file: format!("{stem}-{}.adbl", "a".repeat(64)),
            sha256: "a".repeat(64),
            format_version: 1,
            n,
        };
        let manifest = Manifest {
            schema: 1,
            built: "2026-09-01".to_owned(),
            key_scheme: Some(KEY_SCHEME.to_owned()),
            filter: [
                ("sha256-good".to_owned(), entry("sha256-good", 100)),
                ("sha256-bad".to_owned(), entry("sha256-bad", 0)),
            ]
            .into(),
        };
        let text = toml::to_string(&manifest).unwrap();
        let result = Installer::lock(&target)
            .unwrap()
            .install(|staging| stage(staging, &client(false).unwrap(), &base, &manifest, &text));
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("checksum mismatch")
        );
        server.join().unwrap();
        assert_eq!(
            std::fs::read_to_string(target.join("previous")).unwrap(),
            "keep me"
        );
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
