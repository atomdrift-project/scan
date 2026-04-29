//! litmus — ML-powered malware classification CLI.

#[cfg(all(
    unix,
    not(any(target_os = "freebsd", target_os = "dragonfly", target_os = "openbsd"))
))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use litmus::scan::DisplayFilter;
use litmus::OutputFormat;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

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
#[command(group(
    clap::ArgGroup::new("severity_level")
        .args([
            "level_1", "level_2", "level_3", "level_4", "level_5",
            "level_6", "level_7", "level_8", "level_9",
        ])
        .multiple(false)
        .conflicts_with_all(["threshold_suspicious", "threshold_hostile"])
))]
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

    /// Use severity level 1: zero-FP targets; least noisy
    #[arg(short = '1', long = "loose", global = true, action = clap::ArgAction::SetTrue)]
    level_1: bool,

    /// Use severity level 2
    #[arg(short = '2', global = true, action = clap::ArgAction::SetTrue)]
    level_2: bool,

    /// Use severity level 3
    #[arg(short = '3', global = true, action = clap::ArgAction::SetTrue)]
    level_3: bool,

    /// Use severity level 4
    #[arg(short = '4', global = true, action = clap::ArgAction::SetTrue)]
    level_4: bool,

    /// Use severity level 5: default operating point
    #[arg(short = '5', global = true, action = clap::ArgAction::SetTrue)]
    level_5: bool,

    /// Use severity level 6
    #[arg(short = '6', global = true, action = clap::ArgAction::SetTrue)]
    level_6: bool,

    /// Use severity level 7
    #[arg(short = '7', global = true, action = clap::ArgAction::SetTrue)]
    level_7: bool,

    /// Use severity level 8
    #[arg(short = '8', global = true, action = clap::ArgAction::SetTrue)]
    level_8: bool,

    /// Use severity level 9: most sensitive
    #[arg(short = '9', long = "paranoid", global = true, action = clap::ArgAction::SetTrue)]
    level_9: bool,

    /// Classifications to display: hostile, suspicious, sus, benign, all (comma-separated)
    #[arg(long, value_delimiter = ',', default_values = ["hostile", "sus"])]
    show: Vec<Show>,

    /// Warn when a single rule takes longer than this many milliseconds
    #[arg(long, default_value = "4000")]
    slow_rule_ms: u64,

    /// Show raw probability and SHAP feature values in terminal output
    #[arg(long, hide = true)]
    extra: bool,

    /// Apply built-in heuristics that upgrade ML classifications when cleave
    /// findings clearly disagree. Disable with --upgrade-heuristic=false to see
    /// raw model output.
    #[arg(
        long,
        hide = true,
        action = clap::ArgAction::Set,
        default_value_t = true,
        require_equals = true,
    )]
    upgrade_heuristic: bool,

    /// Paths to files or directories to scan (shorthand for `litmus scan <paths...>`)
    paths: Vec<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

impl Cli {
    fn selected_severity_level(&self) -> Option<u8> {
        [
            (self.level_1, 1),
            (self.level_2, 2),
            (self.level_3, 3),
            (self.level_4, 4),
            (self.level_5, 5),
            (self.level_6, 6),
            (self.level_7, 7),
            (self.level_8, 8),
            (self.level_9, 9),
        ]
        .into_iter()
        .find_map(|(selected, level)| selected.then_some(level))
    }
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

    /// Validate cleave traits, load the model, and ensure benign samples stay benign
    Validate,

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

        /// Path to a writable cleave traits directory (overrides CLEAVE_TRAITS_DIR).
        /// Use when running as a restricted user whose $HOME is not writable
        /// (e.g. macOS system accounts where $HOME=/var/empty). Traits are
        /// cloned automatically if the directory does not yet exist.
        #[arg(long)]
        traits_dir: Option<PathBuf>,
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

        /// Maximum RSS in gigabytes before pausing claims.
        /// 0 (default) auto-resolves to 85% of total system RAM; -1 disables
        /// in-process throttling entirely (use when an external supervisor
        /// like systemd `MemoryMax=` already enforces a hard cap).
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        max_rss_gb: i64,

        /// Local data directory. Hopper returns relative paths; the worker
        /// joins them with this root to find files locally instead of
        /// downloading. SHA256 is verified before using a local file.
        #[arg(long)]
        data_dir: Option<PathBuf>,

        /// Exit after this many jobs have been analyzed (default: run forever)
        #[arg(long)]
        max_jobs: Option<u64>,

        /// Path to a writable cleave traits directory (overrides CLEAVE_TRAITS_DIR).
        /// Use when running as a restricted user whose $HOME is not writable
        /// (e.g. macOS system accounts where $HOME=/var/empty). Traits are
        /// cloned automatically if the directory does not yet exist.
        #[arg(long)]
        traits_dir: Option<PathBuf>,

        /// Nice value applied to the worker process at startup. Default 10
        /// keeps analysis bursts from starving other work on the host. Pass 0
        /// to leave priority unchanged (e.g. when profiling). Unprivileged
        /// processes can only raise the nice value.
        #[arg(long, default_value = "10", allow_hyphen_values = true)]
        nice: i32,
    },

    /// Print version information
    Version,
}

fn threshold_overrides_for_model(
    model_dir: &Path,
    severity_level: Option<u8>,
    threshold_suspicious: Option<f32>,
    threshold_hostile: Option<f32>,
) -> Result<Option<litmus::model::Thresholds>> {
    if let Some(level) = severity_level {
        let thresholds =
            litmus::model::load_severity_thresholds(model_dir, level)?.with_context(|| {
                format!(
                    "model bundle does not contain severity level {level} thresholds; \
                     run 'litmus update-rules' to refresh the installed models"
                )
            })?;
        return Ok(Some(thresholds));
    }

    // Only construct explicit thresholds if at least one CLI flag was provided.
    // When both are omitted, pass None so Model::load uses model metadata.
    Ok(match (threshold_suspicious, threshold_hostile) {
        (None, None) => None,
        (sus, hos) => Some(litmus::model::Thresholds {
            suspicious: sus.unwrap_or(litmus::model::Thresholds::FALLBACK_SUSPICIOUS),
            hostile: hos.unwrap_or(litmus::model::Thresholds::FALLBACK_HOSTILE),
        }),
    })
}

fn main() -> Result<()> {
    // Block SIGUSR1 process-wide before spawning any threads so they all inherit
    // the blocked mask; the dedicated sigusr1 thread below consumes it via sigwait.
    #[cfg(unix)]
    unsafe {
        let mut mask: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut mask);
        libc::sigaddset(&mut mask, libc::SIGUSR1);
        libc::pthread_sigmask(libc::SIG_BLOCK, &mask, std::ptr::null_mut());
    }
    // Allow a forked debugger to ptrace us under yama.ptrace_scope=1.
    #[cfg(target_os = "linux")]
    unsafe {
        libc::prctl(libc::PR_SET_PTRACER, libc::PR_SET_PTRACER_ANY, 0, 0, 0);
    }

    let cli = Cli::parse();
    let selected_severity_level = cli.selected_severity_level();
    let threshold_suspicious = cli.threshold_suspicious;
    let threshold_hostile = cli.threshold_hostile;

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

    // Stabilize cleave trait discovery before any cleave shared resources are
    // initialized. This avoids clone-into-existing-directory failures when the
    // default traits checkout was installed by cleave or another litmus run.
    litmus::traits_repo::prepare_runtime_env();

    let rayon_threads = std::env::var("CLEAVE_RAYON_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(4)
        });
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_threads)
        .stack_size(64 * 1024 * 1024)
        .thread_name(|i| format!("rayon-{i}"))
        .build_global()
    {
        tracing::warn!(error = %e, "failed to install global rayon pool; using default");
    }
    tracing::info!(
        threads = rayon::current_num_threads(),
        stack_mb = 64,
        "rayon pool ready"
    );

    // Terminal theme detection is only needed for scan/ps with terminal output.
    // The OSC color-scheme query blocks on a TTY response and hangs in any
    // environment that doesn't reply (SSH, some tmux configs, worker daemons).
    let needs_terminal_theme = cli.format == litmus::OutputFormat::Terminal
        && matches!(command, Commands::Scan { .. } | Commands::Ps);
    if cli.light {
        litmus::output::set_theme(litmus::output::Theme::Light);
    } else if cli.dark {
        litmus::output::set_theme(litmus::output::Theme::Dark);
    } else if needs_terminal_theme {
        litmus::output::detect_theme();
    }

    // Dump all thread backtraces on SIGUSR1 (Linux equivalent of BSD SIGINFO / Ctrl-T).
    // Attaches lldb/gdb to ourselves so every thread is reported with symbols.
    // Best-effort: if spawning fails we simply don't get backtraces on signal.
    #[cfg(unix)]
    let _sigusr1_thread = std::thread::Builder::new()
        .name("sigusr1".into())
        .spawn(|| {
            use std::io::Write;
            use std::process::{Command, Stdio};
            let mut mask: libc::sigset_t = unsafe { std::mem::zeroed() };
            unsafe {
                libc::sigemptyset(&mut mask);
                libc::sigaddset(&mut mask, libc::SIGUSR1);
            }
            loop {
                let mut sig: libc::c_int = 0;
                if unsafe { libc::sigwait(&mask, &mut sig) } != 0 {
                    continue;
                }
                let pid = std::process::id().to_string();
                let _ = writeln!(
                    std::io::stderr(),
                    "\n--- SIGUSR1 all-thread backtrace (pid {pid}) ---"
                );
                let lldb = Command::new("lldb")
                    .args([
                        "--batch",
                        "-p",
                        &pid,
                        "-o",
                        "thread backtrace all",
                        "-o",
                        "detach",
                        "-o",
                        "quit",
                    ])
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .status();
                if !matches!(lldb, Ok(s) if s.success()) {
                    let _ = Command::new("gdb")
                        .args([
                            "-batch",
                            "-nx",
                            "-p",
                            &pid,
                            "-ex",
                            "thread apply all bt",
                            "-ex",
                            "detach",
                            "-ex",
                            "quit",
                        ])
                        .stdout(Stdio::inherit())
                        .stderr(Stdio::inherit())
                        .status();
                }
                let _ = writeln!(std::io::stderr(), "--- end backtrace ---\n");
            }
        });

    #[cfg(debug_assertions)]
    tracing::warn!(
        "DEBUG binary — litmus will be very slow; use `make release` for production builds"
    );

    // Warn about missing analysis tools for commands that will run cleave.
    if matches!(
        command,
        Commands::Scan { .. }
            | Commands::Ps
            | Commands::Validate
            | Commands::Serve { .. }
            | Commands::Worker { .. }
    ) {
        litmus::tools::warn_missing();
    }

    if cli.update {
        std::thread::scope(|s| {
            s.spawn(|| {
                if let Err(e) = litmus::models_repo::update() {
                    eprintln!("Warning: model update failed: {e}");
                }
            });
            s.spawn(|| {
                if let Err(e) = litmus::traits_repo::update(false) {
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
    let threshold_overrides = |model_dir: &Path| -> Result<Option<litmus::model::Thresholds>> {
        threshold_overrides_for_model(
            model_dir,
            selected_severity_level,
            threshold_suspicious,
            threshold_hostile,
        )
    };
    let all = cli.show.iter().any(|s| matches!(s, Show::All));
    let filter = DisplayFilter::new(
        all || cli.show.iter().any(|s| matches!(s, Show::Hostile)),
        all || cli.show.iter().any(|s| matches!(s, Show::Sus)),
        all || cli.show.iter().any(|s| matches!(s, Show::Benign)),
    );

    match command {
        Commands::Scan { paths } => {
            let model_dir = resolve_model_dir()?;
            let thresholds = threshold_overrides(&model_dir)?;
            let config = litmus::ScanConfig::new(
                model_dir,
                cli.format,
                thresholds,
                filter,
                cli.slow_rule_ms,
                cli.extra,
            )?
            .with_upgrade_heuristic(cli.upgrade_heuristic);
            exit_for_summary(&run_scan_paths(&paths, &config)?);
        }
        Commands::Ps => {
            let model_dir = resolve_model_dir()?;
            let thresholds = threshold_overrides(&model_dir)?;
            let config = litmus::ScanConfig::new(
                model_dir,
                cli.format,
                thresholds,
                filter,
                cli.slow_rule_ms,
                cli.extra,
            )?
            .with_upgrade_heuristic(cli.upgrade_heuristic);
            exit_for_summary(&litmus::ps::run(&config)?);
        }
        Commands::Serve {
            bind,
            max_size_mb,
            max_rss_gb,
            allowed_dirs,
            extract_dir,
            workers,
            allow_cidr,
            traits_dir,
        } => {
            if let Some(ref p) = traits_dir {
                std::env::set_var("CLEAVE_TRAITS_DIR", p);
            }
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
            let workers = workers.unwrap_or_else(default_workers);
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
            let model_dir = resolve_model_dir()?;
            let thresholds = threshold_overrides(&model_dir)?;
            let config = litmus::server::ServerConfig::new(
                bind,
                max_size_mb.saturating_mul(1024 * 1024),
                max_rss_bytes,
                model_dir,
                thresholds,
                cli.slow_rule_ms,
                dirs,
                extract_dir.map(std::path::PathBuf::from),
                workers,
                allow_cidrs,
            )?
            .with_upgrade_heuristic(cli.upgrade_heuristic);
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
                    if let Err(e) = litmus::traits_repo::check_updates() {
                        eprintln!("Error checking traits updates: {e}");
                        process::exit(1);
                    }
                }
            } else {
                // Pull models, then validate the new bundle by attempting to load it.
                // If the load fails we roll the on-disk repo back to the prior commit
                // so future invocations (and litmus workers that pull on startup)
                // see the last-known-good state rather than the broken one.
                let prev_models_head = match litmus::models_repo::update() {
                    Ok(prev) => prev,
                    Err(e) => {
                        eprintln!("Error updating models: {e}");
                        process::exit(1);
                    }
                };
                if let Some(prev) = prev_models_head.as_deref() {
                    match litmus::models_repo::model_dir() {
                        Ok(dir) => {
                            if let Err(load_err) = litmus::model::Model::load(&dir, None) {
                                eprintln!("Pulled models failed to load: {load_err}");
                                eprintln!("Rolling back to {prev}...");
                                if let Err(rb) = litmus::models_repo::rollback(prev) {
                                    eprintln!("Rollback to {prev} failed: {rb}");
                                }
                                process::exit(1);
                            }
                        }
                        Err(e) => {
                            eprintln!("Error resolving model directory for validation: {e}");
                            process::exit(1);
                        }
                    }
                }
                if !models_only {
                    if let Err(e) = litmus::traits_repo::update(false) {
                        eprintln!("Error updating traits: {e}");
                        process::exit(1);
                    }
                }
            }
        }
        Commands::Validate => {
            let model_dir = resolve_model_dir()?;
            let thresholds = threshold_overrides(&model_dir)?;
            let config = litmus::ScanConfig::new(
                model_dir,
                litmus::OutputFormat::Terminal,
                thresholds,
                DisplayFilter::alerts_only(),
                cli.slow_rule_ms,
                cli.extra,
            )?
            .with_upgrade_heuristic(cli.upgrade_heuristic);
            litmus::validate::run(&config)?;
        }
        Commands::Worker {
            url,
            name,
            workers,
            poll_secs,
            max_rss_gb,
            data_dir,
            max_jobs,
            traits_dir,
            nice,
        } => {
            if let Some(ref p) = traits_dir {
                std::env::set_var("CLEAVE_TRAITS_DIR", p);
            }
            let workers = workers.unwrap_or_else(default_workers);
            let name = name.unwrap_or_else(|| {
                hostname::get()
                    .ok()
                    .and_then(|h| h.into_string().ok())
                    .unwrap_or_else(|| "unknown".to_string())
            });
            let raw_max_rss_gb = max_rss_gb;
            let max_rss_gb: u64 = match raw_max_rss_gb {
                n if n < 0 => 0,
                0 => resolve_worker_max_rss_gb(),
                n => n as u64,
            };
            log_worker_startup_diagnostics(WorkerStartupDiagnostics {
                argv: std::env::args().collect(),
                hopper_url: &url,
                name: &name,
                workers,
                poll_secs,
                raw_max_rss_gb,
                resolved_max_rss_gb: max_rss_gb,
                data_dir: data_dir.as_deref(),
                max_jobs,
                traits_dir: traits_dir.as_deref(),
                nice,
            });
            // Refresh models and traits at startup so long-lived workers pick up
            // new rules on each restart. Failures are non-fatal — a worker in a
            // disconnected environment must still start with whatever is on disk.
            std::thread::scope(|s| {
                s.spawn(|| {
                    if let Err(e) = litmus::models_repo::update() {
                        eprintln!("Warning: model update failed: {e}");
                    }
                });
                s.spawn(|| {
                    if let Err(e) = litmus::traits_repo::update(false) {
                        eprintln!("Warning: traits update failed: {e}");
                    }
                });
            });
            let model_dir = resolve_model_dir()?;
            let thresholds = threshold_overrides(&model_dir)?;
            let validate_config = litmus::ScanConfig::new(
                model_dir.clone(),
                litmus::OutputFormat::Terminal,
                thresholds,
                DisplayFilter::alerts_only(),
                cli.slow_rule_ms,
                cli.extra,
            )?
            .with_upgrade_heuristic(cli.upgrade_heuristic);
            if let Err(e) = litmus::validate::run(&validate_config) {
                eprintln!("Worker startup validation failed: {e:#}");
                process::exit(1);
            }
            match raw_max_rss_gb {
                n if n < 0 => {
                    tracing::info!(
                        "in-process RSS throttling disabled (--max-rss-gb=-1); \
                         relying on external supervisor for OOM enforcement",
                    );
                }
                0 => {
                    tracing::info!(
                        max_rss_gb,
                        "auto-resolved worker RSS ceiling (set --max-rss-gb to override, -1 to disable)",
                    );
                }
                _ => {}
            }
            let config = litmus::worker::WorkerConfig {
                hopper_url: url,
                name,
                workers,
                poll_secs,
                max_rss_gb,
                data_dir,
                max_jobs,
                model_dir,
                thresholds,
                slow_rule_ms: cli.slow_rule_ms,
                upgrade_heuristic: cli.upgrade_heuristic,
                nice,
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(litmus::worker::run(config))?;
        }
        Commands::Version => {
            if cli.format == litmus::OutputFormat::Json {
                let version = serde_json::json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "models": litmus::models_repo::version(),
                    "traits": cleave::traits_repo::version(),
                });
                println!("{version}");
            } else {
                println!("litmus {}", env!("CARGO_PKG_VERSION"));
                if let Some(v) = litmus::models_repo::version() {
                    println!("  models: {v}");
                }
                if let Some(v) = cleave::traits_repo::version() {
                    println!("  traits: {v}");
                }
            }
        }
    }

    Ok(())
}

/// Default worker count: at least 2, and half the available CPU cores.
fn default_workers() -> usize {
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4);
    std::cmp::max(2, cores / 2)
}

struct WorkerStartupDiagnostics<'a> {
    argv: Vec<String>,
    hopper_url: &'a str,
    name: &'a str,
    workers: usize,
    poll_secs: u64,
    raw_max_rss_gb: i64,
    resolved_max_rss_gb: u64,
    data_dir: Option<&'a Path>,
    max_jobs: Option<u64>,
    traits_dir: Option<&'a Path>,
    nice: i32,
}

fn log_worker_startup_diagnostics(d: WorkerStartupDiagnostics<'_>) {
    let total_memory_mb = cleave::memory_tracker::total_memory().map(|b| b / MIB);
    let memory_limit_mb = cleave::memory_tracker::memory_limit() / MIB;
    let current_rss_mb = cleave::memory_tracker::current_rss().map(|b| b / MIB);
    let proc_memtotal = proc_memtotal_mb();
    let (proc_memtotal_mb, proc_memtotal_error) = match proc_memtotal {
        Ok(mb) => (Some(mb), None),
        Err(e) => (None, Some(e)),
    };
    let cgroup = cgroup_memory_diagnostics();
    let memory_basis = worker_memory_basis();

    if proc_memtotal_error.is_some() {
        if let Some(limit) = cgroup.effective_limit_bytes() {
            tracing::warn!(
                proc_memtotal_error = ?proc_memtotal_error,
                cgroup_memory_high = ?cgroup.memory_high,
                cgroup_memory_max = ?cgroup.memory_max,
                cgroup_effective_limit_mb = limit / MIB,
                auto_memory_basis_source = memory_basis.source,
                auto_memory_basis_mb = memory_basis.bytes / MIB,
                "proc meminfo unavailable; using shared memory detector for worker RSS auto-resolution",
            );
        } else if memory_basis.source == "fallback_16g" {
            tracing::warn!(
                proc_memtotal_error = ?proc_memtotal_error,
                "physical memory and cgroup memory limit unavailable; using 16 GiB fallback for worker RSS auto-resolution",
            );
        }
    }

    tracing::info!(
        argv = ?d.argv,
        hopper_url = d.hopper_url,
        worker_name = d.name,
        workers = d.workers,
        poll_secs = d.poll_secs,
        raw_max_rss_gb = d.raw_max_rss_gb,
        resolved_max_rss_gb = d.resolved_max_rss_gb,
        resolved_max_rss_mb = d.resolved_max_rss_gb.saturating_mul(1024),
        rss_throttling_enabled = d.resolved_max_rss_gb > 0,
        data_dir = ?d.data_dir,
        max_jobs = ?d.max_jobs,
        traits_dir = ?d.traits_dir,
        nice = d.nice,
        total_memory_mb = ?total_memory_mb,
        cleave_memory_limit_mb = memory_limit_mb,
        current_rss_mb = ?current_rss_mb,
        proc_memtotal_mb = ?proc_memtotal_mb,
        proc_memtotal_error = ?proc_memtotal_error,
        auto_memory_basis_source = memory_basis.source,
        auto_memory_basis_mb = memory_basis.bytes / MIB,
        cgroup_path = ?cgroup.path,
        cgroup_memory_current = ?cgroup.memory_current,
        cgroup_memory_current_mb = ?cgroup.memory_current_mb,
        cgroup_memory_high = ?cgroup.memory_high,
        cgroup_memory_high_mb = ?cgroup.memory_high_mb,
        cgroup_memory_max = ?cgroup.memory_max,
        cgroup_memory_max_mb = ?cgroup.memory_max_mb,
        "worker startup diagnostics",
    );
}

fn proc_memtotal_mb() -> Result<u64, String> {
    let meminfo =
        std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("read /proc/meminfo: {e}"))?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let raw = rest
                .split_whitespace()
                .next()
                .ok_or_else(|| "parse /proc/meminfo MemTotal: missing value".to_string())?;
            let kb: u64 = raw
                .parse()
                .map_err(|e| format!("parse /proc/meminfo MemTotal value {raw:?}: {e}"))?;
            return Ok(kb / 1024);
        }
    }
    Err("parse /proc/meminfo: MemTotal not found".to_string())
}

#[derive(Default)]
struct CgroupMemoryDiagnostics {
    path: Option<String>,
    memory_current: Option<String>,
    memory_current_mb: Option<u64>,
    memory_high: Option<String>,
    memory_high_mb: Option<u64>,
    memory_max: Option<String>,
    memory_max_mb: Option<u64>,
}

impl CgroupMemoryDiagnostics {
    fn effective_limit_bytes(&self) -> Option<u64> {
        [self.memory_high.as_deref(), self.memory_max.as_deref()]
            .into_iter()
            .flatten()
            .filter_map(|v| memory_value_bytes(Some(v)))
            .min()
    }
}

#[cfg(target_os = "linux")]
fn cgroup_memory_diagnostics() -> CgroupMemoryDiagnostics {
    let Some(path) = cgroup_v2_path() else {
        return CgroupMemoryDiagnostics::default();
    };
    let memory_current = read_trimmed(path.join("memory.current"));
    let memory_high = read_trimmed(path.join("memory.high"));
    let memory_max = read_trimmed(path.join("memory.max"));
    CgroupMemoryDiagnostics {
        path: Some(path.display().to_string()),
        memory_current_mb: memory_value_mb(memory_current.as_deref()),
        memory_high_mb: memory_value_mb(memory_high.as_deref()),
        memory_max_mb: memory_value_mb(memory_max.as_deref()),
        memory_current,
        memory_high,
        memory_max,
    }
}

#[cfg(not(target_os = "linux"))]
fn cgroup_memory_diagnostics() -> CgroupMemoryDiagnostics {
    CgroupMemoryDiagnostics::default()
}

#[cfg(target_os = "linux")]
fn cgroup_v2_path() -> Option<PathBuf> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    for line in cgroup.lines() {
        let mut parts = line.splitn(3, ':');
        let hierarchy = parts.next()?;
        let controllers = parts.next()?;
        let rel = parts.next()?;
        if hierarchy == "0" && controllers.is_empty() {
            let rel = rel.trim_start_matches('/');
            return Some(if rel.is_empty() {
                PathBuf::from("/sys/fs/cgroup")
            } else {
                PathBuf::from("/sys/fs/cgroup").join(rel)
            });
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn read_trimmed(path: PathBuf) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(target_os = "linux")]
fn memory_value_mb(value: Option<&str>) -> Option<u64> {
    memory_value_bytes(value).map(|b| b / MIB)
}

fn memory_value_bytes(value: Option<&str>) -> Option<u64> {
    let value = value?;
    if value == "max" {
        return None;
    }
    value.parse::<u64>().ok()
}

struct WorkerMemoryBasis {
    bytes: u64,
    source: &'static str,
}

fn worker_memory_basis() -> WorkerMemoryBasis {
    if let Some(bytes) = cleave::memory_tracker::total_memory() {
        return WorkerMemoryBasis {
            bytes,
            source: "cleave_total_memory",
        };
    }
    WorkerMemoryBasis {
        bytes: 16 * GIB,
        source: "fallback_16g",
    }
}

/// Auto-resolve the worker RSS ceiling when the user hasn't set `--max-rss-gb`.
///
/// Pick 85% of cleave's shared memory detector, which is cgroup-aware on
/// Linux and falls back to 16 GiB only when no memory signal is available.
fn resolve_worker_max_rss_gb() -> u64 {
    let total_bytes = worker_memory_basis().bytes;
    std::cmp::max(1, (total_bytes * 85 / 100) / GIB)
}

/// Exit with the appropriate code based on scan summary counters.
fn exit_for_summary(summary: &litmus::ScanSummary) {
    if summary.hostile > 0 {
        process::exit(1);
    }
    if summary.suspicious > 0 {
        process::exit(2);
    }
    if summary.errors > 0 {
        process::exit(3);
    }
}

fn run_scan_paths(paths: &[PathBuf], config: &litmus::ScanConfig) -> Result<litmus::ScanSummary> {
    // Warm YARA + capability mapper off the rayon pool before any analysis
    // spawns rayon work. Directory scans run on a dedicated rayon pool; if
    // any of those workers is the first to hit `yara_engine()`, the init's
    // internal par_iter deadlocks against its peers parked on the OnceLock.
    // Prefetching from main (non-rayon) fills the OnceLock safely.
    cleave::prefetch_shared_resources(true);

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

    #[allow(clippy::cast_possible_truncation)]
    {
        summary.duration_ms = started.elapsed().as_millis() as u64;
    }
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

    #[test]
    fn upgrade_heuristic_defaults_to_true() -> Result<()> {
        let cli = Cli::try_parse_from(["litmus", "/tmp/a"]).context("parse should work")?;
        assert!(cli.upgrade_heuristic);
        Ok(())
    }

    #[test]
    fn upgrade_heuristic_accepts_false() -> Result<()> {
        let cli = Cli::try_parse_from(["litmus", "--upgrade-heuristic=false", "/tmp/a"])
            .context("parse should work")?;
        assert!(!cli.upgrade_heuristic);
        Ok(())
    }

    #[test]
    fn upgrade_heuristic_accepts_true_explicitly() -> Result<()> {
        let cli = Cli::try_parse_from(["litmus", "--upgrade-heuristic=true", "/tmp/a"])
            .context("parse should work")?;
        assert!(cli.upgrade_heuristic);
        Ok(())
    }

    #[test]
    fn upgrade_heuristic_rejects_bare_form() {
        // require_equals means --upgrade-heuristic without =value is a parse error.
        // This keeps parsing unambiguous when positional paths follow.
        let result = Cli::try_parse_from(["litmus", "--upgrade-heuristic", "/tmp/a"]);
        assert!(result.is_err());
    }

    #[test]
    fn severity_level_flags_parse() -> Result<()> {
        let cli = Cli::try_parse_from(["litmus", "-1", "/tmp/a"]).context("-1 should parse")?;
        assert_eq!(cli.selected_severity_level(), Some(1));

        let cli =
            Cli::try_parse_from(["litmus", "--loose", "/tmp/a"]).context("--loose should parse")?;
        assert_eq!(cli.selected_severity_level(), Some(1));

        let cli = Cli::try_parse_from(["litmus", "--paranoid", "/tmp/a"])
            .context("--paranoid should parse")?;
        assert_eq!(cli.selected_severity_level(), Some(9));

        let cli = Cli::try_parse_from(["litmus", "scan", "--paranoid", "/tmp/a"])
            .context("global --paranoid should parse after scan")?;
        assert_eq!(cli.selected_severity_level(), Some(9));
        Ok(())
    }

    #[test]
    fn gzip_long_aliases_are_not_accepted() {
        assert!(Cli::try_parse_from(["litmus", "--fast", "/tmp/a"]).is_err());
        assert!(Cli::try_parse_from(["litmus", "--best", "/tmp/a"]).is_err());
    }

    #[test]
    fn severity_level_flags_conflict_with_each_other_and_manual_thresholds() {
        assert!(Cli::try_parse_from(["litmus", "-1", "-9", "/tmp/a"]).is_err());
        assert!(
            Cli::try_parse_from(["litmus", "-9", "--threshold-hostile", "0.90", "/tmp/a"]).is_err()
        );
    }

    #[test]
    fn upgrade_heuristic_rejects_non_bool_value() {
        let result = Cli::try_parse_from(["litmus", "--upgrade-heuristic=yes", "/tmp/a"]);
        assert!(result.is_err());
    }
}
