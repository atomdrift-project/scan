use anyhow::Result;
use clap::{Parser, Subcommand};
use litmus::scan::DisplayFilter;
use litmus::OutputFormat;
use std::path::PathBuf;
use std::process;

/// Classification values accepted by `--show`.
#[derive(Clone, clap::ValueEnum)]
enum Show {
    Hostile,
    Sus,
    Benign,
    All,
}

#[derive(Parser)]
#[command(name = "litmus")]
#[command(version)]
#[command(about = "Malware classification powered by ML + static analysis")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan files or directories for hostile/suspicious content
    Scan {
        /// Path to file or directory to scan
        path: PathBuf,

        /// Directory containing model files (model.onnx, feature_spec.json, etc.)
        #[arg(long, default_value = "../collimator/out")]
        model_dir: PathBuf,

        /// Output format
        #[arg(short, long, default_value = "terminal")]
        format: OutputFormat,

        /// Probability threshold for suspicious classification (0.0-1.0)
        #[arg(long, default_value = "0.75")]
        threshold_suspicious: f32,

        /// Probability threshold for hostile classification (0.0-1.0)
        #[arg(long, default_value = "0.85")]
        threshold_hostile: f32,

        /// Classifications to display: hostile, sus, benign, all (comma-separated)
        #[arg(long, value_delimiter = ',', default_values = ["hostile", "sus"])]
        show: Vec<Show>,

        /// Show cleave finding counts (h/s/n/b) and raw model score per file
        #[arg(long, short)]
        verbose: bool,
    },

    /// Scan executables of all running processes
    Ps {
        /// Directory containing model files (model.onnx, feature_spec.json, etc.)
        #[arg(long, default_value = "../collimator/out")]
        model_dir: PathBuf,

        /// Output format
        #[arg(short, long, default_value = "terminal")]
        format: OutputFormat,

        /// Probability threshold for suspicious classification (0.0-1.0)
        #[arg(long, default_value = "0.75")]
        threshold_suspicious: f32,

        /// Probability threshold for hostile classification (0.0-1.0)
        #[arg(long, default_value = "0.85")]
        threshold_hostile: f32,

        /// Classifications to display: hostile, sus, benign, all (comma-separated)
        #[arg(long, value_delimiter = ',', default_values = ["hostile", "sus"])]
        show: Vec<Show>,

        /// Show cleave finding counts (h/s/n/b) and raw model score per file
        #[arg(long, short)]
        verbose: bool,
    },

    /// Print version information
    Version,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    #[cfg(debug_assertions)]
    log::warn!("DEBUG binary — litmus will be very slow; use `make release` for production builds");

    let cli = Cli::parse();

    match cli.command {
        Commands::Scan {
            path,
            model_dir,
            format,
            threshold_suspicious,
            threshold_hostile,
            show,
            verbose,
        } => {
            let all = show.iter().any(|s| matches!(s, Show::All));
            let config = litmus::ScanConfig {
                model_dir,
                format,
                threshold_suspicious,
                threshold_hostile,
                filter: DisplayFilter {
                    hostile: all || show.iter().any(|s| matches!(s, Show::Hostile)),
                    suspicious: all || show.iter().any(|s| matches!(s, Show::Sus)),
                    benign: all || show.iter().any(|s| matches!(s, Show::Benign)),
                },
                verbose,
            };
            let result = litmus::scan::run(&path, &config)?;

            if result.hostile > 0 {
                process::exit(1);
            }
            if result.suspicious > 0 {
                process::exit(2);
            }
        }
        Commands::Ps {
            model_dir,
            format,
            threshold_suspicious,
            threshold_hostile,
            show,
            verbose,
        } => {
            let all = show.iter().any(|s| matches!(s, Show::All));
            let config = litmus::ps::PsConfig {
                model_dir,
                format,
                threshold_suspicious,
                threshold_hostile,
                filter: DisplayFilter {
                    hostile: all || show.iter().any(|s| matches!(s, Show::Hostile)),
                    suspicious: all || show.iter().any(|s| matches!(s, Show::Sus)),
                    benign: all || show.iter().any(|s| matches!(s, Show::Benign)),
                },
                verbose,
            };
            let result = litmus::ps::run(&config)?;

            if result.hostile > 0 {
                process::exit(1);
            }
            if result.suspicious > 0 {
                process::exit(2);
            }
        }
        Commands::Version => {
            println!("litmus {}", env!("CARGO_PKG_VERSION"));
        }
    }

    Ok(())
}
