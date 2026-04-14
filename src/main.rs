//! litmus — ML-powered malware classification CLI.

#[cfg(all(unix, not(any(target_os = "freebsd", target_os = "dragonfly", target_os = "openbsd"))))]
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
    #[value(name = "sus", alias = "suspicious")]
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

    /// Override suspicious threshold (0.0-1.0); omit to use model's recommendation
    #[arg(long)]
    threshold_suspicious: Option<f32>,

    /// Override hostile threshold (0.0-1.0); omit to use model's recommendation
    #[arg(long)]
    threshold_hostile: Option<f32>,

    /// Classifications to display: hostile, suspicious, sus, benign, all (comma-separated)
    #[arg(long, value_delimiter = ',', default_values = ["hostile", "sus"])]
    show: Vec<Show>,

    /// Warn when a single rule takes longer than this many milliseconds
    #[arg(long, default_value = "4000")]
    slow_rule_ms: u64,

    /// Number of parallel cleave scan workers for directory/archive-heavy scans
    #[arg(long)]
    scan_threads: Option<usize>,

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
        #[arg(long, default_value = "127.0.0.1:49999")]
        bind: SocketAddr,

        /// Maximum upload size in megabytes
        #[arg(long, default_value = "100")]
        max_size_mb: usize,

        /// Maximum RSS in gigabytes before rejecting requests.
        /// Defaults to 0, which means auto: min(50% RAM, 32 GiB).
        #[arg(long, default_value = "0")]
        max_rss_gb: u64,

        /// Comma-separated directories allowed for /analyze-path requests
        #[arg(long)]
        allowed_dirs: Option<String>,

        /// Directory for extracting archive members (passed to cleave)
        #[arg(long)]
        extract_dir: Option<String>,

        /// Maximum concurrent analyses (defaults to max(1, num_cpus / 2))
        #[arg(long)]
        workers: Option<usize>,

        /// Comma-separated CIDR networks (in addition to loopback) allowed to
        /// reach the server. /analyze-path is always restricted to loopback
        /// regardless of this list. Pair with --bind 0.0.0.0:PORT to actually
        /// accept remote connections.
        #[arg(long)]
        allow_cidr: Option<String>,
    },

    /// Run as a pull-based worker, polling a hopper instance for analysis jobs
    Worker {
        /// Hopper API base URL (e.g. http://hopper-host:8081)
        #[arg(long)]
        url: String,

        /// Worker name (defaults to hostname)
        #[arg(long)]
        name: Option<String>,

        /// Number of concurrent analysis slots
        #[arg(long)]
        workers: Option<usize>,

        /// Poll interval in seconds when no work is available
        #[arg(long, default_value = "2")]
        poll_secs: u64,

        /// Per-request analysis timeout in seconds (0 = no timeout)
        #[arg(long, default_value = "0")]
        timeout_secs: u64,

        /// Maximum RSS in gigabytes before pausing claims (0 = auto)
        #[arg(long, default_value = "0")]
        max_rss_gb: u64,

        /// Rules/model update interval in minutes (0 = disabled)
        #[arg(long, default_value = "60")]
        update_interval_mins: u64,

        /// Local data directory. Hopper returns relative paths; the worker
        /// joins them with this root to find files locally instead of
        /// downloading. SHA256 is verified before using a local file.
        #[arg(long)]
        data_dir: Option<PathBuf>,
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

    let is_serve = matches!(command, Commands::Serve { .. } | Commands::Worker { .. });
    let filter = if cli.verbose {
        tracing_subscriber::EnvFilter::new("litmus=debug,cleave=debug")
    } else if is_serve {
        tracing_subscriber::EnvFilter::new("litmus=info,cleave=warn")
    } else {
        tracing_subscriber::EnvFilter::new("litmus=warn,cleave=error")
    };
    let fmt = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_thread_names(true)
        .with_writer(std::io::stderr);
    if is_serve {
        fmt.init();
    } else {
        fmt.without_time().init();
    }

    // Configure the global rayon pool BEFORE any cleave work runs. Cleave's
    // archive + YARA-X + composite-rule evaluation paths recurse deeply enough
    // to exhaust rayon's default 2 MB worker stack, producing an unnamed
    // `thread '<unknown>' has overflowed its stack` abort. Cleave's CLI does
    // this in `cli_bootstrap`, but litmus uses cleave as a library and never
    // calls that, so we have to install the global pool ourselves.
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .stack_size(16 * 1024 * 1024)
        .thread_name(|i| format!("rayon-{i}"))
        .build_global()
    {
        tracing::warn!(error = %e, "failed to install global rayon pool; using default");
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

    // Resolve model directory lazily — update-rules and version don't need it,
    // and eagerly resolving triggers auto-clone before those commands can run.
    let resolve_model_dir = || -> Result<PathBuf> {
        match &cli.model_dir {
            Some(d) => Ok(d.clone()),
            None => litmus::models_repo::model_dir()
                .map_err(|e| anyhow::anyhow!("failed to resolve model directory: {e}")),
        }
    };
    let all = cli.show.iter().any(|s| matches!(s, Show::All));
    let filter = DisplayFilter::new(
        all || cli.show.iter().any(|s| matches!(s, Show::Hostile)),
        all || cli.show.iter().any(|s| matches!(s, Show::Sus)),
        all || cli.show.iter().any(|s| matches!(s, Show::Benign)),
    );
    // Only construct explicit thresholds if at least one CLI flag was provided.
    // When both are omitted, pass None so Model::load uses evaluation.json.
    let thresholds = match (cli.threshold_suspicious, cli.threshold_hostile) {
        (None, None) => None,
        (sus, hos) => Some(litmus::model::Thresholds {
            suspicious: sus.unwrap_or(litmus::model::Thresholds::FALLBACK_SUSPICIOUS),
            hostile: hos.unwrap_or(litmus::model::Thresholds::FALLBACK_HOSTILE),
        }),
    };

    match command {
        Commands::Scan { paths } => {
            let config = litmus::ScanConfig::new(
                resolve_model_dir()?,
                cli.format,
                thresholds,
                filter,
                cli.slow_rule_ms,
                cli.scan_threads,
                cli.extra,
            )?;
            let result = run_scan_paths(&paths, &config)?;

            if result.hostile > 0 {
                process::exit(1);
            }
            if result.suspicious > 0 {
                process::exit(2);
            }
            if result.errors > 0 {
                process::exit(3);
            }
        }
        Commands::Ps => {
            let config = litmus::ScanConfig::new(
                resolve_model_dir()?,
                cli.format,
                thresholds,
                filter,
                cli.slow_rule_ms,
                cli.scan_threads,
                cli.extra,
            )?;
            let result = litmus::ps::run(&config)?;

            if result.hostile > 0 {
                process::exit(1);
            }
            if result.suspicious > 0 {
                process::exit(2);
            }
            if result.errors > 0 {
                process::exit(3);
            }
        }
        Commands::Serve {
            bind,
            max_size_mb,
            max_rss_gb,
            allowed_dirs,
            extract_dir,
            workers,
            allow_cidr,
        } => {
            let dirs: Vec<std::path::PathBuf> = allowed_dirs
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let p = std::path::PathBuf::from(s);
                    // Canonicalize allowed dirs at startup so symlink-resolved
                    // request paths match correctly in starts_with checks.
                    p.canonicalize().unwrap_or(p)
                })
                .collect();
            let workers = workers.unwrap_or_else(|| {
                let cores = std::thread::available_parallelism()
                    .map(std::num::NonZero::get)
                    .unwrap_or(4);
                std::cmp::max(2, cores / 2)
            });
            let allow_cidrs = match allow_cidr {
                Some(s) => litmus::server::parse_cidr_list(&s)
                    .map_err(|e| anyhow::anyhow!("--allow-cidr: {e}"))?,
                None => Vec::new(),
            };
            let max_rss_bytes = if max_rss_gb == 0 {
                cleave::memory_tracker::memory_limit()
            } else {
                max_rss_gb.saturating_mul(1024 * 1024 * 1024)
            };
            let config = litmus::server::ServerConfig::new(
                bind,
                max_size_mb.saturating_mul(1024 * 1024),
                max_rss_bytes,
                resolve_model_dir()?,
                thresholds,
                cli.slow_rule_ms,
                dirs,
                extract_dir.map(std::path::PathBuf::from),
                workers,
                allow_cidrs,
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
        Commands::Worker {
            url,
            name,
            workers,
            poll_secs,
            timeout_secs,
            max_rss_gb,
            update_interval_mins,
            data_dir,
        } => {
            let model_dir = resolve_model_dir()?;
            let workers = workers.unwrap_or_else(|| {
                let cores = std::thread::available_parallelism()
                    .map(std::num::NonZero::get)
                    .unwrap_or(4);
                std::cmp::max(2, cores / 2)
            });
            let name = name.unwrap_or_else(|| {
                hostname::get()
                    .ok()
                    .and_then(|h| h.into_string().ok())
                    .unwrap_or_else(|| "unknown".to_string())
            });
            let config = litmus::worker::WorkerConfig {
                hopper_url: url,
                name,
                workers,
                poll_secs,
                timeout_secs,
                max_rss_gb,
                update_interval_mins,
                data_dir,
                model_dir,
                thresholds,
                slow_rule_ms: cli.slow_rule_ms,
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(litmus::worker::run(config))?;
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
    use super::{Cli, Commands};
    use anyhow::{Context, Result};
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn bare_paths_default_to_scan_shorthand() -> Result<()> {
        let cli =
            Cli::try_parse_from(["litmus", "/tmp/a", "/tmp/b"]).context("parse should work")?;
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
