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

use scan::bloom_build::{read_pool, write_bundle};

#[derive(Debug, Parser)]
#[command(name = "scan-bloom-build", about = "Build known-good/known-bad bloom filters from a pool")]
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
    /// resistance (which filter size cannot buy); ~1e-7 is plenty.
    #[arg(long, default_value_t = 1e-7, value_name = "P")]
    fp: f64,
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
    let filters = sets.into_filters(args.fp);
    let manifest = write_bundle(&args.out, &filters, &args.date)?;

    for (stem, entry) in &manifest.filter {
        eprintln!("  {stem}: {} elements → {} ({})", entry.n, entry.file, entry.sha256);
    }
    eprintln!("wrote {} filters + bloom.toml to {}", manifest.filter.len(), args.out.display());
    Ok(())
}
