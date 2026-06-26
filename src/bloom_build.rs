//! Producer side: turn a labelled good/bad pool into a publishable bloom bundle.
//!
//! The input is NDJSON, one pool record per line — the decoupled contract a
//! hopper export fills (`{ "purl": "pkg:npm/x@1", "sha256": "<hex>", "label":
//! "good" }`). Either `purl` or `sha256` may be absent; `label` is `good` or
//! `bad` and is taken verbatim. The *policy* (good = explicitly labelled good
//! with no hostile findings) lives in whatever produces the export, not here —
//! this module only does the set algebra and serialization.
//!
//! [`read_pool`] parses the stream; [`crate::bloom::generate`] builds the four
//! filters (with `good − bad` subtraction); [`write_bundle`] writes each
//! `<stem>.adbl` plus a `bloom.toml` manifest naming every artifact, its sha256,
//! its layout version, and its element count — the input the download flow
//! verifies against, mirroring the model `versions.toml`.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
// Manifest derives both Serialize (producer writes bloom.toml) and Deserialize
// (the download flow and the installed sidecar read it back).

use crate::bloom::{FORMAT_VERSION, Filter, KeySets, Label, Record, parse_sha256_hex};

/// `bloom.toml` schema version; bump on an incompatible manifest layout change.
pub const MANIFEST_SCHEMA: u32 = 1;

/// One line of the NDJSON pool export.
#[derive(Debug, Deserialize)]
struct PoolRecord {
    #[serde(default)]
    purl: Option<String>,
    #[serde(default)]
    sha256: Option<String>,
    label: String,
}

/// Ingest diagnostics for a pool read; the keys themselves live in [`KeySets`].
#[derive(Debug, Default)]
pub struct PoolStats {
    /// Total non-blank lines read.
    pub records: u64,
    /// Lines that were not valid JSON.
    pub malformed: u64,
    /// Records whose `sha256` was present but not a 64-char hex digest.
    pub bad_sha: u64,
    /// Records with a label other than `good`/`bad` (ignored).
    pub other_label: u64,
}

/// Stream an NDJSON pool into a [`KeySets`] accumulator, returning it with
/// ingest diagnostics. Records are inserted as they are parsed — never buffered
/// — so peak memory is the deduplicated key sets alone, independent of row
/// count. Malformed lines are counted and skipped, never fatal: one bad row must
/// not abort a 25M-row build.
///
/// # Errors
/// Propagates only underlying read errors from `reader`.
pub fn read_pool(reader: impl BufRead) -> Result<(KeySets, PoolStats)> {
    let mut sets = KeySets::default();
    let mut stats = PoolStats::default();
    for line in reader.lines() {
        let line = line.context("reading pool line")?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        stats.records += 1;
        let Ok(rec) = serde_json::from_str::<PoolRecord>(line) else {
            stats.malformed += 1;
            continue;
        };
        let sha = match rec.sha256.as_deref().map(parse_sha256_hex) {
            Some(Some(digest)) => Some(digest),
            Some(None) => {
                stats.bad_sha += 1;
                None
            }
            None => None,
        };
        if rec.purl.is_none() && sha.is_none() {
            continue; // nothing to key on
        }
        let label = match rec.label.as_str() {
            "good" => Label::Good,
            "bad" => Label::Bad,
            _ => {
                stats.other_label += 1;
                continue;
            }
        };
        sets.insert(
            label,
            Record {
                purl: rec.purl,
                sha256: sha,
            },
        );
    }
    Ok((sets, stats))
}

/// One filter artifact's entry in `bloom.toml`.
#[derive(Debug, Serialize, Deserialize)]
pub struct Entry {
    /// File name relative to the bundle, e.g. `purl-good.adbl`.
    pub file: String,
    /// Lowercase hex SHA-256 of the file, verified on download.
    pub sha256: String,
    /// On-disk [`FORMAT_VERSION`] the file was written with.
    pub format_version: u16,
    /// Element count (informational).
    pub n: u64,
}

/// The `bloom.toml` manifest: the contract between publish and download.
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    /// Manifest layout version ([`MANIFEST_SCHEMA`]).
    pub schema: u32,
    /// Build date (`YYYY-MM-DD`), stamped by the caller for reproducibility.
    pub built: String,
    /// One entry per filter, keyed by artifact stem (`purl-good`, …).
    pub filter: BTreeMap<String, Entry>,
}

/// Write each filter to `<dir>/<stem>.adbl` and a `bloom.toml` manifest naming
/// them, returning the manifest. `built` is the build date the caller supplies.
///
/// # Errors
/// Returns an error if `dir` cannot be written or the manifest cannot be
/// serialized.
pub fn write_bundle(dir: &Path, filters: &[Filter], built: &str) -> Result<Manifest> {
    let mut entries = BTreeMap::new();
    for filter in filters {
        let stem = filter.artifact_stem();
        let file = format!("{stem}.adbl");
        let bytes = filter.to_bytes();
        std::fs::write(dir.join(&file), &bytes).with_context(|| format!("writing {file}"))?;
        entries.insert(
            stem,
            Entry {
                file,
                sha256: hex_sha256(&bytes),
                format_version: FORMAT_VERSION,
                n: filter.len(),
            },
        );
    }
    let manifest = Manifest {
        schema: MANIFEST_SCHEMA,
        built: built.to_owned(),
        filter: entries,
    };
    let text = toml::to_string(&manifest).context("serializing bloom.toml")?;
    std::fs::write(dir.join("bloom.toml"), text).context("writing bloom.toml")?;
    Ok(manifest)
}

/// Lowercase hex SHA-256 of `bytes`.
fn hex_sha256(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn read_pool_partitions_and_counts() {
        let input = concat!(
            r#"{"purl":"pkg:npm/good@1","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","label":"good"}"#,
            "\n",
            r#"{"purl":"pkg:npm/evil@1","label":"bad"}"#,
            "\n",
            "\n",                  // blank line ignored
            r#"{"label":"good"}"#, // no purl/sha → contributes nothing
            "\n",
            r#"{"sha256":"nothex","label":"bad"}"#, // bad sha, no purl → nothing, counted
            "\n",
            r#"{"purl":"pkg:npm/x@1","label":"review"}"#, // other label
            "\n",
            "not json at all",
            "\n",
        );
        let (sets, stats) = read_pool(input.as_bytes()).unwrap();
        assert_eq!(sets.good_counts(), (1, 1), "one good purl + one good sha");
        assert_eq!(sets.bad_counts(), (1, 0), "one bad purl, no bad sha");
        assert_eq!(stats.records, 6, "six non-blank lines");
        assert_eq!(stats.malformed, 1);
        assert_eq!(stats.bad_sha, 1);
        assert_eq!(stats.other_label, 1);
    }

    #[test]
    fn write_bundle_round_trips_through_manifest_and_loader() {
        let tmp = tempfile::tempdir().unwrap();
        let (sets, _) = read_pool(
            concat!(
                r#"{"purl":"pkg:npm/good@1","label":"good"}"#,
                "\n",
                r#"{"purl":"pkg:npm/evil@1","label":"bad"}"#,
                "\n",
            )
            .as_bytes(),
        )
        .unwrap();
        let filters = sets.into_filters(1e-9);
        let manifest = write_bundle(tmp.path(), &filters, "2026-06-26").unwrap();

        assert_eq!(manifest.schema, MANIFEST_SCHEMA);
        assert_eq!(manifest.built, "2026-06-26");
        // Every entry's recorded sha256 matches the bytes actually on disk.
        for (stem, entry) in &manifest.filter {
            let bytes = std::fs::read(tmp.path().join(&entry.file)).unwrap();
            assert_eq!(entry.sha256, hex_sha256(&bytes), "{stem} sha256 mismatch");
            assert_eq!(entry.format_version, FORMAT_VERSION);
        }
        // bloom.toml exists and the bundle loads back through the consumer path.
        assert!(tmp.path().join("bloom.toml").is_file());
        let lk = crate::bloom_repo::Lookup::load_from(tmp.path());
        assert_eq!(
            lk.decide_purl("pkg:npm/good@1"),
            crate::bloom_repo::Decision::Skip
        );
        assert_eq!(
            lk.decide_purl("pkg:npm/evil@1"),
            crate::bloom_repo::Decision::KnownBad
        );
    }
}
