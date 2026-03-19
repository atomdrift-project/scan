//! litmus — ML-powered malware classification CLI.

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

    /// Path to file or directory to scan (shorthand for `litmus scan <path>`)
    path: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan files or directories for hostile/suspicious content
    Scan {
        /// Path to file or directory to scan
        path: PathBuf,
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

    // Default to Scan when a bare path is given without a subcommand.
    let command = match cli.command {
        Some(cmd) => cmd,
        None => match cli.path {
            Some(ref path) => Commands::Scan { path: path.clone() },
            None => {
                Cli::parse_from(["litmus", "--help"]);
                std::process::exit(0);
            }
        },
    };

    let filter = if cli.verbose {
        tracing_subscriber::EnvFilter::new("litmus=debug,cleave=debug")
    } else {
        tracing_subscriber::EnvFilter::new("litmus=warn,cleave=error")
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .without_time()
        .init();

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
    let filter = DisplayFilter {
        hostile: all || cli.show.iter().any(|s| matches!(s, Show::Hostile)),
        suspicious: all || cli.show.iter().any(|s| matches!(s, Show::Sus)),
        benign: all || cli.show.iter().any(|s| matches!(s, Show::Benign)),
    };

    match command {
        Commands::Scan { path } => {
            let config = litmus::ScanConfig {
                model_dir,
                format: cli.format,
                threshold_suspicious: cli.threshold_suspicious,
                threshold_hostile: cli.threshold_hostile,
                filter,
                slow_rule_ms: cli.slow_rule_ms,
            };
            let result = litmus::scan::run(&path, &config)?;

            if result.hostile > 0 {
                process::exit(1);
            }
            if result.suspicious > 0 {
                process::exit(2);
            }
        }
        Commands::Ps => {
            let config = litmus::ScanConfig {
                model_dir,
                format: cli.format,
                threshold_suspicious: cli.threshold_suspicious,
                threshold_hostile: cli.threshold_hostile,
                filter,
                slow_rule_ms: cli.slow_rule_ms,
            };
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
            let config = litmus::server::ServerConfig {
                bind,
                timeout_secs,
                max_body_size: max_size_mb.saturating_mul(1024 * 1024),
                max_rss_bytes: max_rss_gb.saturating_mul(1024 * 1024 * 1024),
                model_dir,
                threshold_suspicious: cli.threshold_suspicious,
                threshold_hostile: cli.threshold_hostile,
                slow_rule_ms: cli.slow_rule_ms,
            };
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
