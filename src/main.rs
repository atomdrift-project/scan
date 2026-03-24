//! litmus — ML-powered malware classification CLI.

#[cfg(all(unix, not(any(target_os = "freebsd", target_os = "dragonfly"))))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use litmus::scan::DisplayFilter;
use litmus::OutputFormat;
use std::net::SocketAddr;
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
    /// Enable debug logging for litmus and cleave
    #[arg(long)]
    verbose: bool,

    /// Update models and traits before running (failures are non-fatal)
    #[arg(short = 'u', long)]
    update: bool,

    /// Force light-background color theme
    #[arg(long, conflicts_with = "dark")]
    light: bool,

    /// Force dark-background color theme
    #[arg(long, conflicts_with = "light")]
    dark: bool,

    /// Override model directory (default: auto-resolved from models repo)
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Output format
    #[arg(short, long, default_value = "terminal")]
    format: OutputFormat,

    /// Probability threshold for suspicious classification (0.0-1.0)
    #[arg(long, default_value_t = litmus::model::Thresholds::DEFAULT_SUSPICIOUS)]
    threshold_suspicious: f32,

    /// Probability threshold for hostile classification (0.0-1.0)
    #[arg(long, default_value_t = litmus::model::Thresholds::DEFAULT_HOSTILE)]
    threshold_hostile: f32,

    /// Classifications to display: hostile, sus, benign, all (comma-separated)
    #[arg(long, value_delimiter = ',', default_values = ["hostile", "sus"])]
    show: Vec<Show>,

    /// Warn when a single rule takes longer than this many milliseconds
    #[arg(long, default_value = "4000")]
    slow_rule_ms: u64,

    /// Show raw probability and SHAP feature values in terminal output
    #[arg(long, hide = true)]
    extra: bool,

    /// Paths to files or directories to scan (shorthand for `litmus scan <paths...>`)
    paths: Vec<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Scan files or directories for hostile/suspicious content
    Scan {
        /// Paths to files or directories to scan
        #[arg(required = true, num_args = 1..)]
        paths: Vec<PathBuf>,
    },

    /// Scan executables of all running processes
    Ps,

    /// Update models (and optionally cleave traits)
    UpdateRules {
        /// Only update models; skip cleave traits update
        #[arg(long)]
        models_only: bool,

        /// Check for updates without applying them
        #[arg(long)]
        check: bool,
    },

    /// Run as an HTTP classification server
    Serve {
        /// Address to listen on
        #[arg(long, default_value = "127.0.0.1:8081")]
        bind: SocketAddr,

        /// Per-request analysis timeout in seconds
        #[arg(long, default_value = "120")]
        timeout_secs: u64,

        /// Maximum upload size in megabytes
        #[arg(long, default_value = "100")]
        max_size_mb: usize,

        /// Maximum RSS in gigabytes before rejecting requests (Linux only)
        #[arg(long, default_value = "8")]
        max_rss_gb: u64,
    },

    /// Print version information
    Version,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Default to Scan when bare paths are given without a subcommand.
    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            if !cli.paths.is_empty() {
                Commands::Scan {
                    paths: cli.paths.clone(),
                }
            } else {
                Cli::parse_from(["litmus", "--help"]);
                std::process::exit(0);
            }
        }
    };

    let is_serve = matches!(command, Commands::Serve { .. });
    let filter = if cli.verbose {
        tracing_subscriber::EnvFilter::new("litmus=debug,cleave=debug")
    } else if is_serve {
        tracing_subscriber::EnvFilter::new("litmus=info,cleave=warn")
    } else {
        tracing_subscriber::EnvFilter::new("litmus=warn,cleave=error")
    };
    let fmt = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr);
    if is_serve {
        fmt.init();
    } else {
        fmt.without_time().init();
    }

    // Initialize terminal theme before any output.
    if cli.light {
        litmus::output::set_theme(litmus::output::Theme::Light);
    } else if cli.dark {
        litmus::output::set_theme(litmus::output::Theme::Dark);
    } else {
        litmus::output::detect_theme();
    }

    #[cfg(debug_assertions)]
    tracing::warn!(
        "DEBUG binary — litmus will be very slow; use `make release` for production builds"
    );

    if cli.update {
        std::thread::scope(|s| {
            s.spawn(|| {
                if let Err(e) = litmus::models_repo::update() {
                    eprintln!("Warning: model update failed: {e}");
                }
            });
            s.spawn(|| {
                if let Err(e) = cleave::traits_repo::update(false) {
                    eprintln!("Warning: traits update failed: {e}");
                }
            });
        });
    }

    let model_dir = match cli.model_dir {
        Some(model_dir) => model_dir,
        None => litmus::models_repo::model_dir()
            .map_err(|e| anyhow::anyhow!("failed to resolve model directory: {e}"))?,
    };
    let all = cli.show.iter().any(|s| matches!(s, Show::All));
    let filter = DisplayFilter::new(
        all || cli.show.iter().any(|s| matches!(s, Show::Hostile)),
        all || cli.show.iter().any(|s| matches!(s, Show::Sus)),
        all || cli.show.iter().any(|s| matches!(s, Show::Benign)),
    );
    let thresholds = litmus::model::Thresholds {
        suspicious: cli.threshold_suspicious,
        hostile: cli.threshold_hostile,
    };

    match command {
        Commands::Scan { paths } => {
            let config =
                litmus::ScanConfig::new(model_dir, cli.format, thresholds, filter, cli.slow_rule_ms, cli.extra)?;
            let result = run_scan_paths(&paths, &config)?;

            if result.hostile > 0 {
                process::exit(1);
            }
            if result.suspicious > 0 {
                process::exit(2);
            }
        }
        Commands::Ps => {
            let config =
                litmus::ScanConfig::new(model_dir, cli.format, thresholds, filter, cli.slow_rule_ms, cli.extra)?;
            let result = litmus::ps::run(&config)?;

            if result.hostile > 0 {
                process::exit(1);
            }
            if result.suspicious > 0 {
                process::exit(2);
            }
        }
        Commands::Serve {
            bind,
            timeout_secs,
            max_size_mb,
            max_rss_gb,
        } => {
            let config = litmus::server::ServerConfig::new(
                bind,
                timeout_secs,
                max_size_mb.saturating_mul(1024 * 1024),
                max_rss_gb.saturating_mul(1024 * 1024 * 1024),
                model_dir,
                thresholds,
                cli.slow_rule_ms,
            )?;
            eprintln!("Starting litmus server on http://{} ...", bind);
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(litmus::server::run(config))?;
        }

        Commands::UpdateRules { models_only, check } => {
            if check {
                if let Err(e) = litmus::models_repo::check_updates() {
                    eprintln!("Error checking model updates: {e}");
                    process::exit(1);
                }
                if !models_only {
                    if let Err(e) = cleave::traits_repo::check_updates() {
                        eprintln!("Error checking traits updates: {e}");
                        process::exit(1);
                    }
                }
            } else {
                if let Err(e) = litmus::models_repo::update() {
                    eprintln!("Error updating models: {e}");
                    process::exit(1);
                }
                if !models_only {
                    if let Err(e) = cleave::traits_repo::update(false) {
                        eprintln!("Error updating traits: {e}");
                        process::exit(1);
                    }
                }
            }
        }
        Commands::Version => {
            println!("litmus {}", env!("CARGO_PKG_VERSION"));
            if let Some(v) = litmus::models_repo::version() {
                println!("  models: {v}");
            }
            if let Some(v) = cleave::traits_repo::version() {
                println!("  traits: {v}");
            }
        }
    }

    Ok(())
}

fn run_scan_paths(paths: &[PathBuf], config: &litmus::ScanConfig) -> Result<litmus::ScanSummary> {
    let started = std::time::Instant::now();
    let mut summary = litmus::ScanSummary {
        total_files: 0,
        hostile: 0,
        suspicious: 0,
        benign: 0,
        errors: 0,
        duration_ms: 0,
    };

    for path in paths {
        let result = litmus::scan::run(path, config)?;
        summary.total_files += result.total_files;
        summary.hostile += result.hostile;
        summary.suspicious += result.suspicious;
        summary.benign += result.benign;
        summary.errors += result.errors;
    }

    summary.duration_ms = started.elapsed().as_millis() as u64;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use super::{Cli, Commands};
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn bare_paths_default_to_scan_shorthand() -> Result<()> {
        let cli = Cli::try_parse_from(["litmus", "/tmp/a", "/tmp/b"])
            .context("parse should work")?;
        assert_eq!(
            cli.paths,
            vec![PathBuf::from("/tmp/a"), PathBuf::from("/tmp/b")]
        );
        assert!(cli.command.is_none());
        Ok(())
    }

    #[test]
    fn scan_subcommand_accepts_multiple_paths() -> Result<()> {
        let cli = Cli::try_parse_from(["litmus", "scan", "/tmp/a", "/tmp/b"])
            .context("parse should work")?;
        match cli.command.context("scan subcommand expected")? {
            Commands::Scan { paths } => {
                assert_eq!(
                    paths,
                    vec![PathBuf::from("/tmp/a"), PathBuf::from("/tmp/b")]
                );
            }
            other => anyhow::bail!("unexpected command: {other:?}"),
        }
        Ok(())
    }
}
