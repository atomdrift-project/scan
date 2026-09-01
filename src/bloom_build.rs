//! Reading the labelled pool export that a bloom bundle is built from.
//!
//! The input is NDJSON, one record per line — the decoupled contract a hopper
//! export fills:
//!
//! ```text
//! {"purl": "pkg:npm/x@1", "sha256": "<64 hex>", "label": "good"}
//! ```
//!
//! Either `purl` or `sha256` may be absent. `label` is one of `good`, `bad`,
//! `sighted-hostile`, `sighted-suspicious`, and is taken verbatim: the policy
//! deciding which rows earn which label lives in whatever produces the export.
//!
//! Everything downstream of here — set algebra, sizing, serialization, and the
//! pre-publish safety checks — is [`burton`].

use std::io::BufRead;

use anyhow::{Context, Result};
use burton::{KeySets, Record, Tier, parse_sha256_hex};
use serde::Deserialize;

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
    /// Records whose `purl` was present but would not canonicalize, and so
    /// contributed no key. A degenerate string must never become a matchable
    /// filter bit.
    pub bad_purl: u64,
    /// Records carrying a label this build does not know: counted and DROPPED.
    /// Non-zero means the producer is ahead of the builder.
    pub other_label: u64,
}

/// Stream an NDJSON pool into a [`KeySets`] accumulator.
///
/// Records are inserted as they are parsed and never buffered, so peak memory
/// is the deduplicated key sets alone, independent of row count. Malformed
/// lines are counted and skipped rather than fatal: one bad row must not abort
/// a 25M-row build.
///
/// PURLs are canonicalized here, by the same function the scanner uses, so the
/// producer and the consumer cannot disagree about what a key looks like. The
/// scheme is recorded in the bundle under that name; see
/// [`crate::bloom_repo::KEY_SCHEME`].
///
/// # Errors
/// Propagates only underlying read errors from `reader`.
pub fn read_pool(reader: impl BufRead) -> Result<(KeySets, PoolStats)> {
    let mut sets = KeySets::new();
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

        let sha256 = match rec.sha256.as_deref().map(parse_sha256_hex) {
            Some(Some(digest)) => Some(digest),
            Some(None) => {
                stats.bad_sha += 1;
                None
            }
            None => None,
        };
        let purl = match rec.purl.as_deref().map(crate::bloom_repo::purl_key) {
            Some(Some(key)) => Some(key),
            Some(None) => {
                stats.bad_purl += 1;
                None
            }
            None => None,
        };
        if purl.is_none() && sha256.is_none() {
            continue; // nothing to key on
        }

        // Label strings are the producer's contract. An unrecognized one is
        // counted and dropped, which is silent data loss if a producer starts
        // emitting a tier this build predates — hence the count, and hence the
        // rule that the builder change lands first.
        let tier = match rec.label.as_str() {
            "good" => Tier::Good,
            "bad" => Tier::Bad,
            "sighted-hostile" => Tier::SightedHostile,
            "sighted-suspicious" => Tier::SightedSuspicious,
            _ => {
                stats.other_label += 1;
                continue;
            }
        };
        sets.insert(tier, Record { purl, sha256 });
    }
    Ok((sets, stats))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const CLEAN: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    fn read(lines: &[&str]) -> (KeySets, PoolStats) {
        read_pool(lines.join("\n").as_bytes()).unwrap()
    }

    #[test]
    fn partitions_by_label_and_counts_what_it_dropped() {
        let (sets, stats) = read(
            &[
                r#"{"purl":"pkg:npm/good@1","sha256":"$C","label":"good"}"#,
                r#"{"purl":"pkg:npm/bad@1","label":"bad"}"#,
                r#"{"purl":"pkg:npm/seen@1","label":"sighted-hostile"}"#,
                r#"{"purl":"pkg:npm/maybe@1","label":"sighted-suspicious"}"#,
                r#"{"purl":"pkg:npm/future@1","label":"tier-from-the-future"}"#,
                r#"{"purl":"pkg:npm/shortsha@1","sha256":"abc","label":"good"}"#,
                r#"{"purl":"not a purl","label":"good"}"#,
                "not json at all",
                "",
            ]
            .iter()
            .map(|l| l.replace("$C", CLEAN))
            .collect::<Vec<_>>()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        );

        assert_eq!(stats.records, 8);
        assert_eq!(stats.malformed, 1);
        assert_eq!(stats.bad_sha, 1);
        assert_eq!(stats.bad_purl, 1);
        assert_eq!(stats.other_label, 1);

        assert_eq!(sets.counts(Tier::Good), (2, 1), "good purls / shas");
        assert_eq!(sets.counts(Tier::Bad), (1, 0));
        assert_eq!(sets.counts(Tier::SightedHostile), (1, 0));
        assert_eq!(sets.counts(Tier::SightedSuspicious), (1, 0));
    }

    #[test]
    fn a_row_with_nothing_to_key_on_is_skipped() {
        let (sets, stats) = read(&[r#"{"label":"good"}"#]);
        assert_eq!(stats.records, 1);
        assert_eq!(sets.counts(Tier::Good), (0, 0));
    }

    #[test]
    fn keys_land_in_the_form_the_scanner_looks_them_up_by() {
        let (sets, _) = read(&[r#"{"purl":"pkg:NPM/left-pad@1.3.0?arch=x86_64","label":"good"}"#]);
        let filters = sets.into_filters(1e-9);
        let dir = tempfile::tempdir().unwrap();
        burton::build::write_bundle(
            dir.path(),
            &filters,
            "2026-08-31",
            crate::bloom_repo::KEY_SCHEME,
        )
        .unwrap();

        let lk = crate::bloom_repo::Lookup::load_from(dir.path());
        // The type is lowercased and artifact-selection qualifiers are dropped,
        // so an SBOM-stamped spelling and the bare coordinate are one key.
        assert_eq!(
            lk.decide_purl("pkg:npm/left-pad@1.3.0"),
            crate::bloom_repo::Decision::Skip
        );
    }
}
