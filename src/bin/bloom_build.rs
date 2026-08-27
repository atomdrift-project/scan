//! `scan-bloom-build` — the bloom filter producer.
//!
//! Reads a labelled good/bad pool as NDJSON (stdin or `--in`), builds the
//! known-good/known-bad filters for PURLs and SHA-256s, and writes them plus a
//! `bloom.toml` manifest to `--out`. The `make publish-bloom` target feeds it a
//! hopper pool export and uploads the result to R2, mirroring `publish-models`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use sha2::{Digest, Sha256};

use scan::bloom_build::{read_pool, read_prev_manifest, verify_build, write_bundle};

#[derive(Debug, Parser)]
#[command(
    name = "scan-bloom-build",
    about = "Build known-good/known-bad bloom filters from a pool"
)]
struct Args {
    /// NDJSON pool file (`{purl?, sha256?, label}` per line); reads stdin if unset.
    #[arg(long = "in", value_name = "FILE")]
    input: Option<PathBuf>,

    /// Output directory for the `.adbl` filters and `bloom.toml` manifest.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,

    /// Build date stamped into the manifest (`YYYY-MM-DD`).
    #[arg(long, value_name = "YYYY-MM-DD")]
    date: String,

    /// Target false-positive rate. Sized for accidental collisions, not grind
    /// resistance (which filter size cannot buy). 1e-9 (1-in-a-billion) keeps a
    /// good-set FP — which skips the full scan — vanishingly rare; at the current
    /// ~1.8M-key corpus each SHA filter lands in the 16 MiB power-of-two bucket.
    #[arg(long, default_value_t = 1e-9, value_name = "P")]
    fp: f64,

    /// On-disk layout version to emit. Defaults to this build's
    /// [`scan::bloom::FORMAT_VERSION`].
    ///
    /// Publishing an older version alongside the current one is how clients that
    /// predate a tier keep receiving fresh data: the bundle is projected down
    /// (see `KeySets::into_filters_for`) rather than withheld. Run once per
    /// version over the same pool file — the pool is on disk anyway, and a
    /// second read is far cheaper than holding a second copy of the key sets.
    #[arg(long = "format-version", default_value_t = scan::bloom::FORMAT_VERSION,
          value_name = "N")]
    format_version: u16,

    /// Accept a filter whose row count shrank or grew past the safety ratios.
    ///
    /// For a deliberate tier-rule change, where the large step IS the change and
    /// is indistinguishable from the corrupt export the ratios exist to catch.
    /// Narrow by construction: canaries, caller-vouched digests, and the
    /// collapse-to-zero guard all stay fatal, and every accepted breach is
    /// printed. Prefer setting it for the one cutover run rather than leaving it
    /// on in the timer.
    #[arg(
        long = "accept-unusual-growth",
        env = "SCAN_BLOOM_ACCEPT_UNUSUAL_GROWTH"
    )]
    accept_unusual_growth: bool,
}

/// Ubiquitous system binaries that must never be catalogued bad. Hashed at build
/// time — so the check sees the host's *actual* binaries rather than a list of
/// digests that rot across distros — and asserted absent from `sha256-bad`.
/// Missing ones are skipped; we can only vouch for what's present.
const HOST_GOOD_BINS: &[&str] = &["/bin/ls", "/bin/sh", "/bin/cat", "/usr/bin/env"];

/// `(path, sha256)` for each [`HOST_GOOD_BINS`] entry that exists on the host.
fn host_good_digests() -> Vec<(String, [u8; 32])> {
    HOST_GOOD_BINS
        .iter()
        .filter_map(|path| {
            let bytes = std::fs::read(path).ok()?;
            let digest: [u8; 32] = Sha256::digest(&bytes).into();
            Some(((*path).to_owned(), digest))
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();

    let reader: Box<dyn BufRead> = match &args.input {
        Some(path) => Box::new(BufReader::new(
            File::open(path).with_context(|| format!("opening {}", path.display()))?,
        )),
        None => Box::new(BufReader::new(std::io::stdin().lock())),
    };

    let (sets, stats) = read_pool(reader)?;
    let (good_purl, good_sha) = sets.good_counts();
    let (bad_purl, bad_sha) = sets.bad_counts();
    eprintln!(
        "pool: {} records — good {good_purl} purl / {good_sha} sha, bad {bad_purl} purl / {bad_sha} sha \
         ({} malformed, {} bad-sha, {} other-label skipped)",
        stats.records, stats.malformed, stats.bad_sha, stats.other_label,
    );

    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;
    // Size the new build against the previous one (read before we overwrite it).
    let prev = read_prev_manifest(&args.out);
    let filters = sets.into_filters_for(args.format_version, args.fp);
    if filters.is_empty() {
        anyhow::bail!(
            "format version {} produced no filters — is it a version this build knows? \
             (supported: {:?})",
            args.format_version,
            scan::bloom::SUPPORTED_VERSIONS,
        );
    }

    // Fail loud before writing, so a poisoned or hollow build never lands in the
    // output dir (nor reaches `make publish-bloom`).
    verify_build(
        &filters,
        prev.as_ref(),
        &host_good_digests(),
        args.accept_unusual_growth,
    )
    .context("bloom safety check failed")?;

    let manifest = write_bundle(&args.out, &filters, &args.date)?;

    for (stem, entry) in &manifest.filter {
        eprintln!(
            "  {stem}: {} elements → {} ({})",
            entry.n, entry.file, entry.sha256
        );
    }
    eprintln!(
        "wrote {} filters + bloom.toml (format v{}) to {}",
        manifest.filter.len(),
        args.format_version,
        args.out.display()
    );
    Ok(())
}
