use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod classifier;
mod diff;
mod extract;
mod features;
mod hybrid_model;
mod multi_model;
mod output;
mod scan;

#[derive(Parser)]
#[command(name = "litmus")]
#[command(author, version, about = "Malware classification using ML analysis")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output format
    #[arg(short, long, global = true, default_value = "terminal")]
    format: OutputFormat,

    /// Custom hostile threshold (0.0-1.0)
    #[arg(long, global = true)]
    threshold: Option<f32>,
}

#[derive(Clone, Copy, Default, clap::ValueEnum)]
enum OutputFormat {
    #[default]
    Terminal,
    Json,
}

#[derive(Subcommand)]
enum Commands {
    /// Classify a file or directory as benign/suspicious/hostile
    Scan {
        /// Path to file or directory to scan
        path: PathBuf,

        /// Show SHAP explanations (requires Python)
        #[arg(long)]
        explain: bool,

        /// Exit with error if highest criticality matches these levels (comma-separated: hostile,suspicious,notable,inert)
        #[arg(long, value_delimiter = ',')]
        error_if: Vec<String>,
    },

    /// Compare two versions for supply chain attack detection
    Diff {
        /// Path to old/baseline version
        old: PathBuf,

        /// Path to new/target version
        new: PathBuf,
    },

    /// Extract features for training (fast batch mode)
    Extract {
        /// Directory containing samples, or file with list of paths
        input: PathBuf,

        /// Output file (JSON format)
        #[arg(short, long)]
        output: PathBuf,

        /// Label for all samples (0=benign, 1=malicious)
        #[arg(short, long)]
        label: u8,

        /// Number of parallel workers
        #[arg(short, long, default_value = "8")]
        workers: usize,

        /// Limit number of samples to process
        #[arg(long)]
        limit: Option<usize>,

        /// Only include samples with suspicious/hostile traits (for malware training)
        /// Also filters out files with "clean" in the path
        #[arg(long)]
        require_suspicious: bool,

        /// Only include files with this extension (e.g., "js", "py", "exe")
        #[arg(short = 'x', long)]
        extension: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set custom threshold if provided
    let threshold = cli.threshold.unwrap_or(0.80);

    match cli.command {
        Commands::Scan { path, explain, error_if } => {
            scan::run(&path, explain, threshold, cli.format, &error_if)?;
        }
        Commands::Diff { old, new } => {
            diff::run(&old, &new, threshold, cli.format)?;
        }
        Commands::Extract {
            input,
            output,
            label,
            workers,
            limit,
            require_suspicious,
            extension,
        } => {
            extract::run(&input, &output, label, workers, limit, require_suspicious, extension.as_deref())?;
        }
    }

    Ok(())
}
