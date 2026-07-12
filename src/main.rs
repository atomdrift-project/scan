//! Atomdrift Scan (`atomscan`) — ML-powered malware classification CLI.

#[cfg(all(
    unix,
    not(any(
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "openbsd",
        target_os = "illumos",
        target_os = "solaris",
    ))
))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Compile-time jemalloc tuning, read once at allocator initialization.
///
/// `narenas:16` caps the arena count (jemalloc's default is 4× CPUs — 64
/// arenas on a 16-core box) at roughly one per core: measured 30–60 MB lower
/// peak RSS on the reference archive scan at equal-or-better wall time
/// (narenas:8 saved ~100 MB but cost ~4% wall to arena contention — wall is
/// the priority). Runtime configuration still wins: `_RJEM_MALLOC_CONF` (and
/// `/etc/malloc.conf`) override this string, so deployments can retune
/// without rebuilding. Guarded by the same cfg as the allocator itself — on
/// platforms where scan uses the system allocator this symbol is never read.
#[cfg(all(
    unix,
    not(any(
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "openbsd",
        target_os = "illumos",
        target_os = "solaris",
    ))
))]
mod jemalloc_conf {
    /// A `Sync` wrapper so a raw `*const c_char` can live in a static;
    /// jemalloc reads the pointer, it is never written after link time.
    #[repr(transparent)]
    struct SyncPtr(*const std::os::raw::c_char);
    // SAFETY: the pointer targets a NUL-terminated 'static literal and is
    // never mutated, so sharing it across threads is sound.
    unsafe impl Sync for SyncPtr {}

    #[allow(non_upper_case_globals)]
    #[unsafe(no_mangle)]
    static _rjem_malloc_conf: SyncPtr = SyncPtr(c"narenas:16".as_ptr());
}

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use scan::OutputFormat;
use scan::engine::DisplayFilter;
use std::net::SocketAddr;
use std::num::NonZeroU64;
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

/// Warn threshold for a single slow cleave rule (ms); was the `--slow-rule-ms`
/// flag, now a fixed advisory default.
const DEFAULT_SLOW_RULE_MS: u64 = 4000;

#[derive(Parser)]
#[command(name = "atomscan")]
#[command(version)]
#[command(about = "Atomdrift Scan — context-free malware detection (ML + static analysis)")]
#[command(group(
    clap::ArgGroup::new("severity_level")
        .args(["level"])
        .conflicts_with_all(["threshold_suspicious", "threshold_hostile"])
))]
struct Cli {
    /// Enable debug logging for Atomdrift Scan and cleave
    #[arg(long)]
    verbose: bool,

    /// Update models and traits before running (failures are non-fatal)
    #[arg(short = 'u', long)]
    update: bool,

    /// Disable the automatic rules/models refresh (on by default when the local
    /// ruleset is over 24h stale). Also settable via `SCAN_NO_UPDATE`.
    #[arg(long)]
    no_update: bool,

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
    #[arg(short, long, env = "SCAN_FORMAT", default_value = "terminal")]
    format: OutputFormat,

    /// Scan mode: `fast` (bloom matching only), `balanced` (bloom short-circuits,
    /// then full scan), or `slow` (no bloom; always full scan). Workers are
    /// always slow.
    #[arg(long, global = true, default_value = "balanced")]
    mode: scan::Mode,

    /// Override suspicious threshold (0.0-1.0); omit to use model's recommendation
    #[arg(long)]
    threshold_suspicious: Option<f32>,

    /// Override hostile threshold (0.0-1.0); omit to use model's recommendation
    #[arg(long)]
    threshold_hostile: Option<f32>,

    /// Tune thresholds for false-positive level N (0-25000, FP per 100M benigns): higher = more sensitive, noisier. Bundle decides which levels are calibrated.
    #[arg(short = 'l', long, value_name = "N", value_parser = clap::value_parser!(u16).range(0..=25000), global = true)]
    level: Option<u16>,

    /// Classifications to display in the terminal view: hostile, suspicious,
    /// sus, benign, all (comma-separated). The machine formats (json, tiny,
    /// interpret) emit every scanned file regardless.
    #[arg(long, value_delimiter = ',', default_values = ["hostile", "sus"])]
    show: Vec<Show>,

    /// Show raw probability and SHAP feature values in terminal output
    #[arg(long, hide = true)]
    extra: bool,

    /// Send non-trivial samples to a local LLM for a second opinion, blended
    /// with the ML verdict (stored in the `llm` JSON section and shown inline).
    /// Enable via env with `SCAN_INTERPRET=1` (endpoint from `SCAN_LLM`).
    #[arg(long, global = true, env = "SCAN_INTERPRET")]
    interpret: bool,

    /// LLM endpoint or provider (env: SCAN_LLM)
    #[arg(long, global = true, value_name = "LLM")]
    llm: Option<String>,

    /// LLM model name (env: SCAN_LLM_MODEL)
    #[arg(long, global = true, value_name = "NAME")]
    llm_model: Option<String>,

    /// LLM bearer token (env: SCAN_LLM_KEY); omit for local endpoints
    #[arg(long, global = true, value_name = "KEY")]
    llm_key: Option<String>,

    /// Minimum ML probability for a sample to be sent to the LLM
    #[arg(long, global = true, default_value_t = scan::interpret::DEFAULT_MIN_PROB, value_name = "P")]
    interpret_min_prob: f32,

    /// Per-request LLM timeout, in seconds
    #[arg(long, global = true, default_value_t = scan::interpret::DEFAULT_TIMEOUT_SECS, value_name = "SECS")]
    llm_timeout: u64,

    /// [EXPERIMENTAL] Fetch the external references discovered in analyzed
    /// files, re-analyze each payload, and fold it into the verdict. A
    /// comma-separated selection of what to fetch, by how strongly the reference
    /// is bound to the artifact:
    ///   `deps`     — strict dependencies declared in a manifest/lockfile
    ///                (package.json, Cargo.lock, .SRCINFO depends);
    ///   `packages` — packages merely mentioned by an install command
    ///                (`npm install foo`, `pip install bar`), typically injected
    ///                by a build/lifecycle script rather than pinned;
    ///   `urls`     — raw URLs with no package identity (curl/wget targets,
    ///                staged downloads), the largest exposure.
    /// `all` (or a bare `--fetch`) selects every kind. When omitted, an
    /// interactive scan defaults to `deps,packages` (the references bound to the
    /// artifact, not raw URLs); a worker defaults to `all`. Also settable per
    /// deployment via the `SCAN_FETCH` env var (e.g. `SCAN_FETCH=urls,packages`
    /// or `SCAN_FETCH=all`), which overrides the default in both modes. An online
    /// step performed after the offline analysis.
    #[arg(
        long,
        global = true,
        value_name = "KINDS",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "all",
        env = "SCAN_FETCH"
    )]
    fetch: Option<scan::fetch::FetchPolicy>,

    /// [EXPERIMENTAL] How many hops of references to follow when `--fetch` is on:
    /// `1` fetches only what the scanned files reference, `2` also follows
    /// references found inside those payloads (reaching a stage-3 `curl | bash`
    /// dropper), and so on. Also settable via `SCAN_FETCH_DEPTH`.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = scan::fetch::DEFAULT_FETCH_DEPTH,
        env = "SCAN_FETCH_DEPTH"
    )]
    fetch_depth: u8,

    /// [EXPERIMENTAL] Skip fetching a declared dependency whose registry publish
    /// date is older than this many days — the cheap provenance lookup runs
    /// first, and only recent (freshest-risk) releases are pulled and scanned.
    /// Applies to declared dependencies only; URLs are never age-gated. `0`
    /// disables the gate (fetch every resolvable dependency). A dependency whose
    /// age can't be determined is always fetched. Also settable via
    /// `SCAN_FETCH_MAX_AGE`.
    #[arg(
        long,
        global = true,
        value_name = "DAYS",
        default_value_t = scan::fetch::DEFAULT_MAX_DEP_AGE_DAYS,
        env = "SCAN_FETCH_MAX_AGE"
    )]
    fetch_max_age: u32,

    /// [EXPERIMENTAL] Fetch and scan native-binary dependencies for every
    /// platform, not just the host's. By default an `<os>-<arch>` package
    /// (biome, esbuild, swc, rollup, sharp…) is pulled only for the host
    /// platform — the linux/windows variants never run here, so scanning them
    /// multiplies the expensive binary analysis with no added coverage for this
    /// host. Pass this to audit the binaries that ship for *other* platforms (a
    /// CI image, a published release). Also settable via
    /// `SCAN_FETCH_ALL_PLATFORMS`.
    #[arg(long, global = true, env = "SCAN_FETCH_ALL_PLATFORMS")]
    fetch_all_platforms: bool,

    /// [EXPERIMENTAL] How long to trust cached *mutable* registry metadata before
    /// revalidating. Accepts a unit suffix (`90s`, `30m`, `4h`, `2d`) — a bare
    /// number is seconds; `never` caches indefinitely (offline/air-gapped). This
    /// bounds the two mutable tiers: a pinned version's packument (whose yank
    /// status can change after publish) and a `latest`/versionless lookup.
    /// Unset keeps the defaults — 4h pinned, 1h unpinned. A released version's
    /// immutable file list is never re-checked regardless. Also settable via
    /// `SCAN_REGISTRY_TTL`.
    #[arg(
        long,
        global = true,
        value_name = "DUR",
        value_parser = scan::fetch::parse_duration,
        env = "SCAN_REGISTRY_TTL"
    )]
    registry_ttl: Option<std::time::Duration>,

    /// [EXPERIMENTAL] Size cap for a single downloaded artifact. Accepts a unit
    /// suffix (`256M`, `2G`, `512K`); a bare number is bytes. A response larger
    /// than this is abandoned, so one artifact can't dominate a run. Also
    /// settable via `SCAN_FETCH_MAX_SIZE`.
    #[arg(
        long,
        global = true,
        value_name = "SIZE",
        default_value = "256M",
        value_parser = scan::fetch::parse_bytes,
        env = "SCAN_FETCH_MAX_SIZE"
    )]
    fetch_max_size: u64,

    /// [EXPERIMENTAL] Maximum number of *live* fetches triggered by a single
    /// scanned file. Cache hits are always served and never counted, so a warm
    /// re-run is never throttled. References past the cap are recorded as
    /// budget-exceeded, never silently dropped. Also settable via
    /// `SCAN_FETCH_MAX_FILE_FETCHES`.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = scan::fetch::DEFAULT_MAX_FILE_FETCHES,
        env = "SCAN_FETCH_MAX_FILE_FETCHES"
    )]
    fetch_max_file_fetches: usize,

    /// [EXPERIMENTAL] Maximum total bytes fetched on behalf of a single scanned
    /// file. Accepts a unit suffix (`2G`); a bare number is bytes. Also settable
    /// via `SCAN_FETCH_MAX_FILE_SIZE`.
    #[arg(
        long,
        global = true,
        value_name = "SIZE",
        default_value = "2G",
        value_parser = scan::fetch::parse_bytes,
        env = "SCAN_FETCH_MAX_FILE_SIZE"
    )]
    fetch_max_file_size: u64,

    /// [EXPERIMENTAL] Maximum number of *live* fetches across the whole
    /// execution — a hard ceiling over every scanned file combined. Lifted in
    /// long-lived server modes (`serve`/`worker`), where the per-file caps bound
    /// each job instead. Also settable via `SCAN_FETCH_MAX_TOTAL_FETCHES`.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = scan::fetch::DEFAULT_MAX_TOTAL_FETCHES,
        env = "SCAN_FETCH_MAX_TOTAL_FETCHES"
    )]
    fetch_max_total_fetches: usize,

    /// [EXPERIMENTAL] Maximum total bytes fetched across the whole execution.
    /// Accepts a unit suffix (`10G`); a bare number is bytes. Lifted in
    /// long-lived server modes (`serve`/`worker`). Also settable via
    /// `SCAN_FETCH_MAX_TOTAL_SIZE`.
    #[arg(
        long,
        global = true,
        value_name = "SIZE",
        default_value = "10G",
        value_parser = scan::fetch::parse_bytes,
        env = "SCAN_FETCH_MAX_TOTAL_SIZE"
    )]
    fetch_max_total_size: u64,

    /// Paths to files or directories to scan (shorthand for `scan path <paths...>`)
    paths: Vec<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

impl Cli {
    /// Build the LLM interpretation config from `--interpret` and the `--llm*`
    /// flags, falling back to env vars. `None` when `--interpret` is not set.
    fn interpret_config(&self) -> Option<scan::interpret::InterpretConfig> {
        if !self.interpret {
            return None;
        }
        use scan::interpret::{DEFAULT_BASE_URL, DEFAULT_MAX_CONCURRENCY, DEFAULT_MODEL};
        let from_env = |flag: &Option<String>, key: &str| -> Option<String> {
            flag.clone()
                .or_else(|| std::env::var(key).ok())
                .filter(|s| !s.is_empty())
        };
        Some(scan::interpret::InterpretConfig {
            base_url: from_env(&self.llm, "SCAN_LLM")
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            model: from_env(&self.llm_model, "SCAN_LLM_MODEL")
                .unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            api_key: from_env(&self.llm_key, "SCAN_LLM_KEY"),
            min_prob: self.interpret_min_prob,
            timeout: std::time::Duration::from_secs(self.llm_timeout),
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
        })
    }
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Scan files or directories for hostile/suspicious content
    #[command(aliases = ["fs", "scan"])]
    Path {
        /// Paths to files or directories to scan
        #[arg(required = true, num_args = 1..)]
        paths: Vec<PathBuf>,

        /// Renew each result on a hopper instance by POSTing its envelope to
        /// `<URL>/api/result`. Assumes the SHA256 was already ingested by
        /// hopper; this refreshes the stored verdict with the local build's
        /// traits and model. Upload failures are logged, never fatal.
        #[arg(long, visible_alias = "upload", value_name = "URL")]
        hopper: Option<String>,

        /// Apply pre-collected registry metadata per scanned file, so an offline
        /// scan reasons over the same registry facts a live `pkg`/`url` scan would
        /// — without refetching, and even when the package has since been pulled.
        /// Takes a JSON object mapping each file's sha256 to its registry record
        /// (the `registry.record` slot of the sidecar hopper stores; the rest of
        /// the provenance is ignored). A file absent from the map scans normally.
        /// Also settable via the `SCAN_REGISTRY_MAP` env var.
        #[arg(long, value_name = "FILE", env = "SCAN_REGISTRY_MAP")]
        registry_map: Option<PathBuf>,
    },

    /// Scan executables of all running processes
    Ps,

    /// Triage this host: running-process executables plus common persistence
    /// and temp locations where malware stages
    #[command(alias = "host")]
    Sys,

    /// Fetch a URL and scan the retrieved bytes
    Url {
        /// URL to fetch and scan (e.g. https://host/path/file)
        url: String,
    },

    /// Fetch a package by PURL and scan it (e.g. npm/left-pad@1.3.0)
    #[command(aliases = ["pkg", "package", "pkgs"])]
    Purl {
        /// Package URL to resolve, fetch, and scan. The `pkg:` scheme is
        /// optional (`npm/foo` == `pkg:npm/foo`). A versionless PURL resolves
        /// to the registry's current release.
        purl: String,
    },

    /// Update models (and optionally cleave traits)
    UpdateRules {
        /// Only update models; skip cleave traits update
        #[arg(long)]
        models_only: bool,

        /// Check for updates without applying them
        #[arg(long)]
        check: bool,
    },

    /// Validate the model bundle and benign fixture corpus
    Validate {
        /// Skip trait-dependent fixture inference; validate model layout only
        #[arg(long)]
        skip_traits: bool,
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
        /// 0 (default) auto-resolves to the process memory limit; -1 disables
        /// in-process throttling entirely (use when an external supervisor
        /// like systemd `MemoryMax=` already enforces a hard cap).
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        max_rss_gb: i64,

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

        /// Renew every analyzed result (parent and members) on a hopper instance
        /// by POSTing to <URL>/api/result as analyses complete — the warm-server
        /// equivalent of `scan path --hopper`. Upload failures are logged, never
        /// fatal. Reads $HOPPER_UPLOAD_TOKEN for authentication.
        #[arg(long, visible_alias = "upload", value_name = "URL")]
        hopper: Option<String>,
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
        #[arg(short = 'j', long)]
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

        /// Nice value applied to the worker process at startup. Default 18
        /// keeps analysis bursts from starving other work on the host. Pass 0
        /// to leave priority unchanged (e.g. when profiling). Unprivileged
        /// processes can only raise the nice value.
        #[arg(long, default_value = "18", allow_hyphen_values = true)]
        nice: i32,

        /// Skip the startup model/traits refresh (git pull), running against
        /// whatever rules are already on disk. Use when the local traits/models
        /// are intentionally ahead of (or diverged from) the remote, e.g. local
        /// edits that would block the pull.
        #[arg(long)]
        no_update: bool,

        /// Skip the strict startup trait-validation gate. Use for benchmarking
        /// or dev runs against locally-edited (possibly not-yet-valid) traits;
        /// the analysis path tolerates lint-level issues the pre-flight rejects.
        #[arg(long)]
        no_validate: bool,

        /// Exit cleanly once the hopper reports no further work and the prefetch
        /// queue drains. For benchmarks / batch runs over a finite dataset;
        /// unlike `--max-jobs` it needs no job count and can't wedge on a
        /// blocked claim.
        #[arg(long)]
        exit_if_empty: bool,
    },

    /// Print version information
    Version,
}

fn threshold_overrides_for_model(
    threshold_suspicious: Option<f32>,
    threshold_hostile: Option<f32>,
) -> Option<scan::model::Thresholds> {
    // Level mode resolves the verdict from the model's level grid — the
    // per-file level sweep plus the active level's hostile/suspicious cutoffs —
    // so no explicit thresholds are loaded here; `Model::load` keeps its
    // level-independent defaults and the active level is passed separately.
    //
    // Only manual `--threshold-*` overrides produce a Thresholds. We do NOT
    // derive a suspicious cutoff: the level-space lookup needs a level table and
    // a known level, neither of which applies when the operator picks
    // `--threshold-hostile` directly. Collapsing suspicious to hostile means
    // `classify` never returns Suspicious — the operator gets hostile-vs-benign
    // verdicts only, which is what we want for manual mode.
    match (threshold_suspicious, threshold_hostile) {
        (None, None) => None,
        (sus, hos) => {
            let hostile = hos.unwrap_or(scan::model::Thresholds::FALLBACK_HOSTILE);
            let suspicious = sus.unwrap_or(hostile);
            Some(scan::model::Thresholds {
                suspicious,
                hostile,
            })
        }
    }
}

/// `SCAN_NO_ANALYSIS_CACHE=1` — one switch that disables every cache of our
/// own analysis across the stack: filefacts file metadata, stng extracted
/// strings, cleave analysis results, and scan's analysis envelope + LLM
/// verdicts. Download caches (fletch registry metadata) and rule-compilation
/// caches (YARA, trait mapper) stay on — they hold inputs, not analysis.
///
/// Implemented by filling in each layer's own env var, so per-layer semantics
/// stay defined in one place and child processes inherit the policy. Only
/// unset vars are filled in: a per-layer var the operator set explicitly
/// always wins over the umbrella.
fn propagate_no_analysis_cache() {
    let on = std::env::var("SCAN_NO_ANALYSIS_CACHE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    if !on {
        return;
    }
    // CLEAVE_SKIP_CACHE=1 also drags cleave's YARA rule-compilation cache with
    // it (legacy behavior); pin that cache back on — recompiling rules costs
    // 4-18s per process and compiles rules, not sample analysis.
    let defaults = [
        ("FILEFACTS_CACHE", "0"),
        ("STNG_STRING_CACHE", "0"),
        ("CLEAVE_SKIP_CACHE", "1"),
        ("CLEAVE_SKIP_YARA_CACHE", "0"),
        ("SCAN_ANALYSIS_CACHE", "0"),
    ];
    for (key, value) in defaults {
        if std::env::var_os(key).is_none() {
            // SAFETY: called from the top of main before any thread is
            // spawned, so no concurrent environment access can race this.
            unsafe { std::env::set_var(key, value) };
        }
    }
}

fn main() -> Result<()> {
    propagate_no_analysis_cache();
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
    let selected_severity_level = cli.level;
    let threshold_suspicious = cli.threshold_suspicious;
    let threshold_hostile = cli.threshold_hostile;
    // Resolve before `cli.command` is moved out below; only one arm uses it.
    let interpret_cfg = cli.interpret_config();
    // The kind selection comes from `--fetch`; the hop count from `--fetch-depth`,
    // the dependency age ceiling from `--fetch-max-age`, and the per-file ceilings
    // from `--fetch-max-file-*` (each its own flag/env). When `--fetch`/`SCAN_FETCH`
    // is unset the default depends on run mode: an interactive scan fetches
    // declared dependencies and install-command packages (the references bound to
    // the artifact) but not raw URLs, while a worker fetches every kind. An
    // explicit selection is honored verbatim in both; the knobs always apply.
    let with_knobs = |mut policy: scan::fetch::FetchPolicy| {
        policy.depth = cli.fetch_depth;
        policy.max_dep_age_days = cli.fetch_max_age;
        policy.max_file_fetches = cli.fetch_max_file_fetches;
        policy.max_file_bytes = cli.fetch_max_file_size;
        policy.host_platform_only = !cli.fetch_all_platforms;
        policy
    };
    // The per-fetch size ceiling is enforced in the HTTP layer, so it's a
    // process-global rather than a per-policy field — set it once here and every
    // mode (interactive scan and worker alike) honors `--fetch-max-size`.
    scan::fetch::set_max_fetch_bytes(cli.fetch_max_size);
    // Registry-metadata staleness bound: a process-global for the same reason,
    // consulted by every registry lookup. `None` keeps the tiered defaults.
    scan::fetch::set_registry_ttl(cli.registry_ttl);
    let scan_default = scan::fetch::FetchPolicy {
        deps: true,
        packages: true,
        ..scan::fetch::FetchPolicy::default()
    };
    let worker_default = scan::fetch::FetchPolicy {
        urls: true,
        packages: true,
        deps: true,
        ..scan::fetch::FetchPolicy::default()
    };
    let fetch_policy = with_knobs(cli.fetch.unwrap_or(scan_default));
    let worker_fetch_policy = with_knobs(cli.fetch.unwrap_or(worker_default));
    if let Some(cfg) = &interpret_cfg {
        tracing::info!(
            endpoint = %cfg.base_url,
            model = %cfg.model,
            min_prob = format!("{:.4}", cfg.min_prob),
            "LLM interpretation enabled",
        );
    }

    // Default to a file scan when bare paths are given without a subcommand.
    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            if !cli.paths.is_empty() {
                Commands::Path {
                    paths: cli.paths.clone(),
                    hopper: None,
                    // Bare `scan <path>` bypasses clap's per-subcommand parsing, so
                    // read the registry-map env var here too — this is the form
                    // cyclotron's LLM agents run.
                    registry_map: std::env::var_os("SCAN_REGISTRY_MAP").map(PathBuf::from),
                }
            } else {
                Cli::parse_from(["scan", "--help"]);
                std::process::exit(0);
            }
        }
    };

    let is_serve = matches!(command, Commands::Serve { .. } | Commands::Worker { .. });
    // Reclaim stale/oversized caches (stng strings+r2, scan analysis+interpret,
    // fletch blobs). One detached, self-gated sweep for a CLI run; a recurring
    // loop for the never-exiting serve/worker daemons. Non-blocking either way.
    scan::cache_cleanup::start(is_serve);
    // Long-lived server modes never short-circuit on bloom filters: every job is
    // analyzed on its own merits. Force slow mode there regardless of `--mode`.
    let effective_mode = if is_serve { scan::Mode::Slow } else { cli.mode };
    // Per-execution fetch ceiling: a hard cap across the whole invocation. The
    // long-lived server modes (`serve`/`worker`) scan unboundedly many jobs over
    // their lifetime, so they're exempt — each job is bounded by `--fetch-max-file-*`
    // instead. A one-shot scan gets the full `--fetch-max-total-*` (the budget
    // defaults to unlimited, so leaving it unset in server mode lifts it).
    if !is_serve {
        scan::fetch::set_total_budget(cli.fetch_max_total_fetches, cli.fetch_max_total_size);
    }
    // RUST_LOG (when set) wins over the mode-derived defaults, so profiling
    // runs can surface targeted modules (e.g. `cleave::mem_profile=info`)
    // without paying for full `--verbose` debug output.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        if cli.verbose {
            tracing_subscriber::EnvFilter::new("scan=debug,cleave=debug")
        } else if is_serve {
            tracing_subscriber::EnvFilter::new("scan=info,cleave=warn")
        } else {
            tracing_subscriber::EnvFilter::new("scan=warn,cleave=error")
        }
    });
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
    scan::traits_repo::prepare_runtime_env();

    // The trait-validation gate run by `validate` and at worker startup is a
    // runtime deploy gate, not a trait-authoring lint: a bundle must be rejected
    // only when it won't load or silently loses detections, never for authoring
    // hygiene (taxonomy, size, dedup, regex style, precision). Request cleave's
    // soft validation via env so it applies to the in-process
    // `cleave::commands::validate::run` call without threading a flag through
    // (and is silently ignored by engines predating soft support).
    //
    // SAFETY: as above — still single-threaded here; rayon and tokio pools are
    // constructed below.
    if matches!(command, Commands::Validate { .. } | Commands::Worker { .. }) {
        unsafe { std::env::set_var("CLEAVE_VALIDATE_SOFT", "1") };
    }

    const RAYON_FALLBACK_THREADS: usize = 4;
    let detected_cores = detect_cpu_count();
    let rayon_threads = std::env::var("CLEAVE_RAYON_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            detected_cores.unwrap_or_else(|| {
                tracing::warn!(
                    fallback = RAYON_FALLBACK_THREADS,
                    "CPU count detection failed; rayon pool DOWNGRADED to \
                     {RAYON_FALLBACK_THREADS} threads. On a many-core host this throttles \
                     throughput and oversubscribes workers — set CLEAVE_RAYON_THREADS to the \
                     core count.",
                );
                RAYON_FALLBACK_THREADS
            })
        });
    // 256 MB stacks: cleave's archive analysis is nested-parallel, and a rayon
    // worker blocked in an inner join steals other pending tasks — including
    // other in-flight analyses' tasks on this shared pool — and runs them on
    // top of its current stack. Frames from independent deep analyses stack
    // up, so the headroom must cover several, not one (64 MB overflowed in
    // production with 4 large archives in flight). Stacks are virtual memory;
    // only pages actually touched are committed, so the cost of the extra
    // headroom is address space, not RSS.
    const RAYON_STACK_MB: usize = 256;
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_threads)
        .stack_size(RAYON_STACK_MB * 1024 * 1024)
        .thread_name(|i| format!("rayon-{i}"))
        // Register each pool worker for the SIGUSR1 in-process thread dump, so a
        // wedge can be backtraced without a debugger (lldb/gdb can't attach in
        // the production jails).
        .start_handler(|_| scan::thread_dump::register_self())
        .build_global()
    {
        tracing::warn!(error = %e, "failed to install global rayon pool; using default");
    }
    let active_threads = rayon::current_num_threads();
    // A pool smaller than the detected core count is a resource downgrade that
    // oversubscribes the worker slots — surface it loudly so it is diagnosable
    // from a single log line.
    if let Some(cores) = detected_cores
        && active_threads < cores
    {
        tracing::warn!(
            rayon_threads = active_threads,
            detected_cores = cores,
            "rayon pool smaller than detected cores; worker slots may oversubscribe the CPU",
        );
    }
    tracing::info!(
        threads = active_threads,
        detected_cores,
        stack_mb = RAYON_STACK_MB,
        "rayon pool ready"
    );

    // Terminal theme detection is only needed for scan/ps with terminal output.
    // The OSC color-scheme query blocks on a TTY response and hangs in any
    // environment that doesn't reply (SSH, some tmux configs, worker daemons).
    let needs_terminal_theme = cli.format == scan::OutputFormat::Terminal
        && matches!(
            command,
            Commands::Path { .. }
                | Commands::Ps
                | Commands::Sys
                | Commands::Url { .. }
                | Commands::Purl { .. }
        );
    if cli.light {
        scan::output::set_theme(scan::output::Theme::Light);
    } else if cli.dark {
        scan::output::set_theme(scan::output::Theme::Dark);
    } else if needs_terminal_theme {
        scan::output::detect_theme();
    }

    // Name the in-flight analyses if the process aborts (e.g. a stack overflow
    // deep in cleave's analysis). Adds a suspect list to the abort log after the
    // runtime's own overflow message. Best-effort and async-signal-safe.
    scan::crash_dump::install();

    // Install the in-process backtrace capture handler so SIGUSR1 can dump every
    // analysis thread's stack without a debugger (lldb/gdb can't attach in the
    // production jails). Must precede the SIGUSR1 thread below.
    scan::thread_dump::install();

    // Dump all analysis-thread backtraces on SIGUSR1 (Linux equivalent of BSD
    // SIGINFO / Ctrl-T), captured in-process — works in jails where ptrace is
    // unavailable. Best-effort: if spawning the handler thread fails we simply
    // don't get backtraces on signal.
    #[cfg(unix)]
    let _sigusr1_thread = std::thread::Builder::new()
        .name("sigusr1".into())
        .spawn(|| {
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
                // In-process capture: no debugger, works in jails. See
                // `scan::thread_dump`.
                scan::thread_dump::dump_all_threads();
            }
        });

    #[cfg(debug_assertions)]
    tracing::warn!(
        "DEBUG binary — scan will be very slow; use `make release` for production builds"
    );

    // Warn about missing analysis tools for commands that will run cleave.
    if matches!(
        command,
        Commands::Path { .. }
            | Commands::Ps
            | Commands::Sys
            | Commands::Url { .. }
            | Commands::Purl { .. }
            | Commands::Validate { .. }
            | Commands::Serve { .. }
            | Commands::Worker { .. }
    ) {
        scan::tools::warn_missing();
    }

    // Interactive commands get a once-a-day, zero-telemetry update notice. The
    // long-running daemons (serve/worker) are excluded — they refresh on
    // restart and shouldn't print transient notices into their logs.
    if matches!(
        command,
        Commands::Path { .. }
            | Commands::Ps
            | Commands::Sys
            | Commands::Url { .. }
            | Commands::Purl { .. }
            | Commands::Version
            | Commands::UpdateRules { .. }
    ) {
        scan::update_check::maybe_notify(false);
    }

    // Default-on refresh: bring rules + models (+ bloom) current when the local
    // ruleset is over 24h stale — or immediately with `-u/--update` — unless
    // `--no-update`/`SCAN_NO_UPDATE` disables it. Scanning commands only: the
    // daemons refresh on restart, and `version`/`update-rules` manage updates
    // themselves.
    if matches!(
        command,
        Commands::Path { .. }
            | Commands::Ps
            | Commands::Sys
            | Commands::Url { .. }
            | Commands::Purl { .. }
    ) {
        scan::auto_update::refresh_if_stale(
            cli.update,
            cli.no_update,
            effective_mode,
            cli.format == scan::OutputFormat::Terminal,
        );
    }

    // Resolve model directory lazily — update-rules and version don't need it,
    // and eagerly resolving triggers auto-clone before those commands can run.
    let resolve_model_dir = || -> Result<PathBuf> {
        match &cli.model_dir {
            Some(d) => Ok(d.clone()),
            None => scan::models_repo::model_dir().context("failed to resolve model directory"),
        }
    };
    let threshold_overrides =
        || threshold_overrides_for_model(threshold_suspicious, threshold_hostile);
    // The envelope's `ml.lvl` encodes the FPR severity that produced the
    // resolved thresholds (or the `-1` benign sentinel — see engine::level_confidence).
    // Manual `--threshold-*` overrides bypass the levels table entirely, so we
    // pass `None` here; benign verdicts will still surface as `-1` regardless.
    let manual_thresholds = threshold_suspicious.is_some() || threshold_hostile.is_some();
    // Operating-point resolution, in priority order:
    //   1. explicit CLI level (`-l`/`--level-*`)
    //   2. the model-prescribed default baked into the bundle's config.json
    //      (`default_severity_level`) — so a bundle calibrated at L50 deploys at
    //      L50 with no litmus rebuild
    //   3. the `DEFAULT_SEVERITY_LEVEL` const fallback (older configless bundles)
    // Manual `--threshold-*` overrides bypass the levels table, so the envelope
    // level is None then. Resolved per model_dir because (2) lives in the bundle.
    let resolve_envelope_level = |model_dir: &Path| -> Option<u16> {
        if manual_thresholds {
            None
        } else {
            Some(
                selected_severity_level
                    .or_else(|| scan::model::model_default_level(model_dir))
                    .unwrap_or(scan::model::DEFAULT_SEVERITY_LEVEL),
            )
        }
    };
    let all = cli.show.iter().any(|s| matches!(s, Show::All));
    let filter = DisplayFilter::new(
        all || cli.show.iter().any(|s| matches!(s, Show::Hostile)),
        all || cli.show.iter().any(|s| matches!(s, Show::Sus)),
        all || cli.show.iter().any(|s| matches!(s, Show::Benign)),
    );

    match command {
        Commands::Path {
            paths,
            hopper,
            registry_map,
        } => {
            let model_dir = resolve_model_dir()?;
            let envelope_level = resolve_envelope_level(&model_dir);
            let thresholds = threshold_overrides();
            let mut config = scan::ScanConfig::new(
                model_dir,
                cli.format,
                thresholds,
                filter,
                DEFAULT_SLOW_RULE_MS,
                cli.extra,
            )?
            .with_level(envelope_level)
            .with_interpret(interpret_cfg.clone())
            .with_fetch(fetch_policy)
            .with_hopper(hopper);
            // Per-file SHA-256 known-good/known-bad short-circuit (explicit file
            // args only; directories are walked by cleave). Slow mode / workers skip it.
            if effective_mode != scan::Mode::Slow {
                config = config.with_bloom(effective_mode, scan::bloom_repo::Lookup::load());
            }
            // `--registry-map <file>`: a JSON object {sha256: registry record}.
            // Each scanned file is enriched with its own registry provenance,
            // matched by content sha. Entries that don't parse to a record are
            // skipped — registry provenance is best-effort, never required.
            let registry_map = match &registry_map {
                Some(path) => {
                    let bytes = std::fs::read(path)
                        .with_context(|| format!("reading registry map {}", path.display()))?;
                    let raw: std::collections::HashMap<String, serde_json::Value> =
                        serde_json::from_slice(&bytes)
                            .with_context(|| format!("parsing registry map {}", path.display()))?;
                    let mut map = std::collections::HashMap::with_capacity(raw.len());
                    for (sha, value) in raw {
                        let value_bytes = serde_json::to_vec(&value).unwrap_or_default();
                        if let Some(record) = scan::provenance::registry_record(&value_bytes) {
                            map.insert(sha, record);
                        }
                    }
                    Some(map)
                }
                None => None,
            };
            exit_for_summary(&run_scan_paths(&paths, &config, registry_map.as_ref())?);
        }
        Commands::Sys => {
            let model_dir = resolve_model_dir()?;
            let envelope_level = resolve_envelope_level(&model_dir);
            let thresholds = threshold_overrides();
            let config = scan::ScanConfig::new(
                model_dir,
                cli.format,
                thresholds,
                filter,
                DEFAULT_SLOW_RULE_MS,
                cli.extra,
            )?
            .with_level(envelope_level)
            .with_interpret(interpret_cfg.clone())
            .with_fetch(fetch_policy);
            exit_for_summary(&scan::sys::run(&config)?);
        }
        Commands::Ps => {
            let model_dir = resolve_model_dir()?;
            let envelope_level = resolve_envelope_level(&model_dir);
            let thresholds = threshold_overrides();
            let mut config = scan::ScanConfig::new(
                model_dir,
                cli.format,
                thresholds,
                filter,
                DEFAULT_SLOW_RULE_MS,
                cli.extra,
            )?
            .with_level(envelope_level)
            .with_interpret(interpret_cfg.clone())
            .with_fetch(fetch_policy);
            // Per-binary known-good/known-bad short-circuit (by executable sha256).
            if effective_mode != scan::Mode::Slow {
                config = config.with_bloom(effective_mode, scan::bloom_repo::Lookup::load());
            }
            exit_for_summary(&scan::ps::run(&config)?);
        }
        Commands::Url { url } => {
            let model_dir = resolve_model_dir()?;
            let envelope_level = resolve_envelope_level(&model_dir);
            let thresholds = threshold_overrides();
            let config = scan::ScanConfig::new(
                model_dir,
                cli.format,
                thresholds,
                filter,
                DEFAULT_SLOW_RULE_MS,
                cli.extra,
            )?
            .with_level(envelope_level)
            .with_interpret(interpret_cfg.clone())
            .with_fetch(fetch_policy);
            exit_for_summary(&scan::pkg::run_url(&url, &config)?);
        }
        Commands::Purl { purl } => {
            let model_dir = resolve_model_dir()?;
            let envelope_level = resolve_envelope_level(&model_dir);
            let thresholds = threshold_overrides();
            let mut config = scan::ScanConfig::new(
                model_dir,
                cli.format,
                thresholds,
                filter,
                DEFAULT_SLOW_RULE_MS,
                cli.extra,
            )?
            .with_level(envelope_level)
            .with_interpret(interpret_cfg.clone())
            .with_fetch(fetch_policy);
            if effective_mode != scan::Mode::Slow {
                config = config.with_bloom(effective_mode, scan::bloom_repo::Lookup::load());
            }
            exit_for_summary(&scan::pkg::run_pkg(&purl, &config)?);
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
            hopper,
        } => {
            if let Some(p) = traits_dir.as_ref() {
                cleave::traits_repo::set_override_dir(Some(p.into()));
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
                Some(s) => scan::server::parse_cidr_list(&s)
                    .map_err(|e| anyhow::anyhow!("--allow-cidr: {e}"))?,
                None => Vec::new(),
            };
            let max_rss_bytes = resolve_process_max_rss_bytes(max_rss_gb);
            log_max_rss_resolution("server", MaxRssPolicy::from_cli(max_rss_gb), max_rss_bytes);
            let model_dir = resolve_model_dir()?;
            let envelope_level = resolve_envelope_level(&model_dir);
            let thresholds = threshold_overrides();
            let config = scan::server::ServerConfig::new(
                bind,
                max_size_mb.saturating_mul(1024 * 1024),
                max_rss_bytes,
                model_dir,
                thresholds,
                DEFAULT_SLOW_RULE_MS,
                dirs,
                extract_dir.map(std::path::PathBuf::from),
                workers,
                allow_cidrs,
            )?
            .with_level(envelope_level)
            .with_interpret(interpret_cfg.clone())
            .with_fetch(fetch_policy)
            .with_hopper(hopper.clone());
            if let Some(url) = hopper.as_deref() {
                eprintln!("Renewing results on hopper at {url}");
            }
            eprintln!("Starting Atomdrift Scan server on http://{bind} ...");
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(scan::server::run(config))?;
        }

        Commands::UpdateRules { models_only, check } => {
            // R2-backed: model_update validates the staged bundle (Model::load)
            // before swapping it in, so a broken bundle never goes live.
            let dir = scan::models_repo::install_target();
            if check {
                if let Err(e) = scan::model_update::check(&dir) {
                    eprintln!("Error checking model updates: {e}");
                    process::exit(1);
                }
                if !models_only && let Err(e) = scan::traits_repo::check_updates() {
                    eprintln!("Error checking traits updates: {e}");
                    process::exit(1);
                }
                // Bloom filters are an optional fast path; a check failure is non-fatal.
                if !models_only
                    && let Err(e) = scan::bloom_update::check(&scan::bloom_repo::install_dir())
                {
                    eprintln!("Warning: bloom filter check failed: {e}");
                }
            } else {
                if let Err(e) = scan::model_update::update(&dir, false, false) {
                    eprintln!("Error updating models: {e}");
                    process::exit(1);
                }
                if !models_only && let Err(e) = scan::traits_repo::update(false, false) {
                    eprintln!("Error updating traits: {e}");
                    process::exit(1);
                }
                // Bloom filters are an optional fast path; a sync failure is non-fatal
                // — the scan simply runs without the known-good/known-bad short-circuit.
                if !models_only
                    && let Err(e) =
                        scan::bloom_update::update(&scan::bloom_repo::install_dir(), false, false)
                {
                    eprintln!("Warning: bloom filter update failed (non-fatal): {e}");
                }
            }
        }
        Commands::Validate { skip_traits } => {
            let model_dir = resolve_model_dir()?;
            let envelope_level = resolve_envelope_level(&model_dir);
            let thresholds = threshold_overrides();
            let config = scan::ScanConfig::new(
                model_dir,
                scan::OutputFormat::Terminal,
                thresholds,
                DisplayFilter::alerts_only(),
                DEFAULT_SLOW_RULE_MS,
                cli.extra,
            )?
            .with_level(envelope_level);
            scan::validate::run(&config, skip_traits)?;
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
            no_update,
            no_validate,
            exit_if_empty,
        } => {
            if let Some(p) = traits_dir.as_ref() {
                cleave::traits_repo::set_override_dir(Some(p.into()));
            }
            let workers = workers.unwrap_or_else(default_workers);
            let name = name.unwrap_or_else(|| {
                hostname::get()
                    .ok()
                    .and_then(|h| h.into_string().ok())
                    .unwrap_or_else(|| "unknown".to_string())
            });
            let raw_max_rss_gb = max_rss_gb;
            let max_rss_gb = resolve_worker_max_rss_gb(raw_max_rss_gb);
            log_worker_startup_diagnostics(&WorkerStartupDiagnostics {
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
            // `--no-update` skips the refresh; `--no-validate` skips the strict
            // pre-flight (benchmark / local-dev against on-disk rules as-is).
            if no_update {
                tracing::warn!("--no-update: skipping startup model/traits refresh");
            } else {
                std::thread::scope(|s| {
                    s.spawn(|| {
                        let dir = scan::models_repo::install_target();
                        if let Err(e) = scan::model_update::update(&dir, false, false) {
                            eprintln!("Warning: model update failed: {e}");
                        }
                    });
                    s.spawn(|| {
                        if let Err(e) = scan::traits_repo::update(false, false) {
                            eprintln!("Warning: traits update failed: {e}");
                        }
                    });
                });
            }
            let model_dir = resolve_model_dir()?;
            let envelope_level = resolve_envelope_level(&model_dir);
            let thresholds = threshold_overrides();
            if no_validate {
                tracing::warn!(
                    "--no-validate: skipping the trait-validation gate; running \
                     against on-disk rules as-is",
                );
            } else {
                let validate_config = scan::ScanConfig::new(
                    model_dir.clone(),
                    scan::OutputFormat::Terminal,
                    thresholds,
                    DisplayFilter::alerts_only(),
                    DEFAULT_SLOW_RULE_MS,
                    cli.extra,
                )?
                .with_level(envelope_level);
                if let Err(e) = scan::validate::run(&validate_config, false) {
                    eprintln!("Worker startup validation failed: {e:#}");
                    process::exit(1);
                }
            }
            log_max_rss_resolution(
                "worker",
                MaxRssPolicy::from_cli(raw_max_rss_gb),
                max_rss_gb.saturating_mul(GIB),
            );
            let config = scan::worker::WorkerConfig {
                hopper_url: url,
                name,
                workers,
                poll_secs,
                max_rss_gb,
                data_dir,
                max_jobs,
                model_dir,
                thresholds,
                slow_rule_ms: DEFAULT_SLOW_RULE_MS,
                level: envelope_level,
                nice,
                exit_if_empty,
                interpret: interpret_cfg.clone(),
                fetch: worker_fetch_policy,
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(scan::worker::run(config))?;
        }
        Commands::Version => {
            let bloom = scan::bloom_repo::installed_manifest();
            if cli.format == scan::OutputFormat::Json {
                let bloom_json = bloom.as_ref().map(|m| {
                    let filters: serde_json::Map<String, serde_json::Value> = m
                        .filter
                        .iter()
                        .map(|(stem, e)| {
                            (
                                stem.clone(),
                                serde_json::json!({
                                    "n": e.n,
                                    "format_version": e.format_version,
                                    "sha256": e.sha256,
                                }),
                            )
                        })
                        .collect();
                    serde_json::json!({
                        "built": m.built,
                        "schema": m.schema,
                        "filters": filters,
                    })
                });
                let version = serde_json::json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "models": scan::models_repo::version(),
                    "traits": cleave::traits_repo::version(),
                    "bloom": bloom_json,
                });
                println!("{version}");
            } else {
                use scan::output::SourceRow;

                // Counting traits/composites/YARA initialises the rule engines.
                let info = cleave::version_info();

                // Atomics, composites, and third-party YARA all ship in the
                // traits repo, so they share one identity and build date. Prefer
                // the installed bundle's sidecar (commit + committer date); for a
                // dev checkout with no sidecar, read git HEAD directly.
                let traits_dir = cleave::cache::traits_path();
                let traits_installed = cleave::rule_update::installed(&traits_dir);
                let traits_git = traits_installed
                    .is_none()
                    .then(|| git_head(&traits_dir))
                    .flatten();
                let traits_version = traits_installed
                    .as_ref()
                    .map(|i| i.commit.chars().take(9).collect::<String>())
                    .or_else(|| traits_git.as_ref().map(|(c, _)| c.clone()))
                    .unwrap_or_default();
                let traits_epoch = traits_installed
                    .as_ref()
                    .and_then(|i| scan::output::parse_ymd(&i.date))
                    .or_else(|| traits_git.as_ref().map(|&(_, e)| e));

                // Bloom: total element count and the manifest's build date.
                let bloom_count: u64 = bloom
                    .as_ref()
                    .map_or(0, |m| m.filter.values().map(|e| e.n).sum());
                let bloom_version = bloom
                    .as_ref()
                    .and_then(|m| m.filter.values().next())
                    .map(|e| format!("v{}", e.format_version))
                    .unwrap_or_default();
                let bloom_epoch = bloom
                    .as_ref()
                    .and_then(|m| scan::output::parse_ymd(&m.built));

                let total = info.trait_count as u64
                    + info.composite_count as u64
                    + info.yara_rules as u64
                    + bloom_count;

                let mut rows = Vec::new();
                if bloom.is_some() {
                    rows.push(SourceRow {
                        label: "blooms",
                        count: bloom_count,
                        metric: None,
                        version: bloom_version,
                        epoch: bloom_epoch,
                    });
                }
                rows.push(SourceRow {
                    label: "atomics",
                    count: info.trait_count as u64,
                    metric: None,
                    version: traits_version.clone(),
                    epoch: traits_epoch,
                });
                rows.push(SourceRow {
                    label: "composites",
                    count: info.composite_count as u64,
                    metric: None,
                    version: traits_version.clone(),
                    epoch: traits_epoch,
                });
                rows.push(SourceRow {
                    label: "third-party YARA",
                    count: info.yara_rules as u64,
                    metric: None,
                    version: traits_version,
                    epoch: traits_epoch,
                });

                // ML models: route-model count by feature dimensionality
                // (e.g. `64 x 229`), plus the bundle's commit and build date.
                if let Some(count) = scan::models_repo::model_count() {
                    let model_installed =
                        scan::model_update::installed(&scan::models_repo::install_target());
                    let metric = scan::models_repo::feature_dimension()
                        .map(|dim| format!("{count} x {dim}"));
                    rows.push(SourceRow {
                        label: "ML models",
                        count: count as u64,
                        metric,
                        // Truncate to 9 chars to match the traits commit width.
                        version: scan::models_repo::version()
                            .map(|c| c.chars().take(9).collect())
                            .unwrap_or_default(),
                        epoch: model_installed
                            .as_ref()
                            .and_then(|i| scan::output::parse_ymd(&i.date)),
                    });
                }

                scan::output::print_version(env!("CARGO_PKG_VERSION"), total, &rows);
            }
        }
    }

    Ok(())
}

/// The abbreviated commit hash and committer date (Unix seconds) of `HEAD` in
/// `dir`, when `dir` is a git checkout. Returns `None` if git is unavailable or
/// `dir` is not a repository, in which case `scan version` omits that source's
/// identity. This recovers the authentic upstream identity and date for a traits
/// checkout that has no install sidecar.
fn git_head(dir: &Path) -> Option<(String, i64)> {
    let out = process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["show", "-s", "--format=%H %ct", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8(out.stdout).ok()?;
    let mut fields = stdout.split_whitespace();
    let commit: String = fields.next()?.chars().take(9).collect();
    let epoch: i64 = fields.next()?.parse().ok()?;
    Some((commit, epoch))
}

/// CPU count available to this process.
///
/// `std::thread::available_parallelism()` returns `Err(Unsupported)` on
/// illumos/Solaris, where it would otherwise silently collapse the rayon pool
/// to a small fallback on a many-core host. Probe `sysconf` directly there.
/// Returns `None` only when no source works, so callers can log a downgrade.
///
/// Kept local rather than using `cleave::memory_tracker::cpu_count` because the
/// `cleave` dependency is pinned to a published git revision; this fix must take
/// effect in the worker binary without waiting for that pin to advance.
fn detect_cpu_count() -> Option<usize> {
    #[cfg(any(target_os = "illumos", target_os = "solaris"))]
    {
        // SAFETY: sysconf is a pure C function; `_SC_NPROCESSORS_ONLN` is a
        // well-defined POSIX selector and a non-positive return is rejected.
        let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        if n > 0 {
            return Some(n as usize);
        }
    }
    std::thread::available_parallelism()
        .ok()
        .map(std::num::NonZero::get)
}

/// Default worker count: at least 2, and half the available CPU cores.
fn default_workers() -> usize {
    let cores = detect_cpu_count().unwrap_or_else(|| {
        tracing::warn!(
            fallback = 4,
            "CPU count detection failed; defaulting worker basis to 4 cores",
        );
        4
    });
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

fn log_worker_startup_diagnostics(d: &WorkerStartupDiagnostics<'_>) {
    let total_memory_mb = cleave::memory_tracker::total_memory().map(|b| b / MIB);
    let memory_limit_mb = cleave::memory_tracker::memory_limit() / MIB;
    let current_rss_mb = cleave::memory_tracker::current_rss().map(|b| b / MIB);
    let (proc_memtotal_mb, proc_memtotal_error) = match proc_memtotal_mb() {
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

/// User-supplied resolution policy for `--max-rss-gb`.
///
/// The CLI accepts an `i64` so a negative value can opt out, but the three
/// possible behaviours are encoded in the type system from this point on so
/// that downstream code cannot accidentally treat "disabled" as "ceiling = 0".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaxRssPolicy {
    /// `--max-rss-gb=-1`: disable in-process RSS throttling entirely. Use when
    /// an external supervisor (systemd `MemoryMax=`, jail rctl, etc.) already
    /// enforces a hard memory cap.
    Disabled,
    /// `--max-rss-gb=0`: auto-resolve the ceiling from the platform's memory
    /// signal (cgroup limits, /proc/meminfo, or a conservative fallback).
    Auto,
    /// `--max-rss-gb=N` (N > 0): explicit ceiling in gigabytes.
    Explicit(NonZeroU64),
}

impl MaxRssPolicy {
    fn from_cli(raw: i64) -> Self {
        match raw {
            n if n < 0 => Self::Disabled,
            0 => Self::Auto,
            // The previous arms exclude n <= 0, so `cast_unsigned` is
            // value-preserving and `NonZeroU64::new` always returns `Some`.
            // `unwrap_or(MIN)` documents that the fallback is unreachable.
            n => Self::Explicit(NonZeroU64::new(n.cast_unsigned()).unwrap_or(NonZeroU64::MIN)),
        }
    }
}

/// Auto-resolve the worker RSS ceiling: 85% of cleave's shared memory detector,
/// which is cgroup-aware on Linux and falls back to 16 GiB only when no memory
/// signal is available. The fraction scales with the host, so it makes sense
/// from a 16 GB Mac mini up to a 1 TB workstation; slot count scales with cores
/// in parallel (`default_workers`), so the larger ceiling on a big host is what
/// actually lets those extra slots run concurrently.
fn auto_worker_max_rss_gb() -> u64 {
    let total_bytes = worker_memory_basis().bytes;
    std::cmp::max(1, (total_bytes * 85 / 100) / GIB)
}

fn resolve_worker_max_rss_gb(raw_max_rss_gb: i64) -> u64 {
    match MaxRssPolicy::from_cli(raw_max_rss_gb) {
        MaxRssPolicy::Disabled => 0,
        MaxRssPolicy::Auto => auto_worker_max_rss_gb(),
        MaxRssPolicy::Explicit(gb) => gb.get(),
    }
}

fn resolve_process_max_rss_bytes(raw_max_rss_gb: i64) -> u64 {
    match MaxRssPolicy::from_cli(raw_max_rss_gb) {
        MaxRssPolicy::Disabled => 0,
        MaxRssPolicy::Auto => cleave::memory_tracker::memory_limit(),
        MaxRssPolicy::Explicit(gb) => gb.get().saturating_mul(GIB),
    }
}

/// Emit a startup log line describing how `--max-rss-gb` was resolved. The
/// explicit case is intentionally silent: the user picked the number, so
/// echoing it back adds no information.
fn log_max_rss_resolution(role: &'static str, policy: MaxRssPolicy, resolved_bytes: u64) {
    match policy {
        MaxRssPolicy::Disabled => tracing::info!(
            role,
            "in-process RSS throttling disabled (--max-rss-gb=-1); \
             relying on external supervisor for OOM enforcement",
        ),
        MaxRssPolicy::Auto => tracing::info!(
            role,
            resolved_max_rss_mb = resolved_bytes / MIB,
            "auto-resolved RSS ceiling (set --max-rss-gb to override, -1 to disable)",
        ),
        MaxRssPolicy::Explicit(_) => {}
    }
}

/// Exit with the appropriate code based on scan summary counters.
fn exit_for_summary(summary: &scan::ScanSummary) {
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

fn run_scan_paths(
    paths: &[PathBuf],
    config: &scan::ScanConfig,
    registry_map: Option<&std::collections::HashMap<String, fletch::Registry>>,
) -> Result<scan::ScanSummary> {
    // Warm YARA + capability mapper off the rayon pool before any analysis
    // spawns rayon work. Directory scans run on a dedicated rayon pool; if
    // any of those workers is the first to hit `yara_engine()`, the init's
    // internal par_iter deadlocks against its peers parked on the OnceLock.
    // Prefetching from main (non-rayon) fills the OnceLock safely.
    cleave::prefetch_shared_resources(true);

    // Explicit files are analyzed as one parallel batch and each directory is
    // streamed; run_paths shares one model load and verdict tally across all.
    scan::engine::run_paths(paths, config, registry_map)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{
        Cli, Commands, GIB, MaxRssPolicy, resolve_process_max_rss_bytes, resolve_worker_max_rss_gb,
    };
    use anyhow::{Context, Result};
    use clap::Parser;
    use std::net::SocketAddr;
    use std::num::NonZeroU64;
    use std::path::PathBuf;

    #[test]
    fn bare_paths_default_to_scan_shorthand() -> Result<()> {
        let cli = Cli::try_parse_from(["scan", "/tmp/a", "/tmp/b"]).context("parse should work")?;
        assert_eq!(
            cli.paths,
            vec![PathBuf::from("/tmp/a"), PathBuf::from("/tmp/b")]
        );
        assert!(cli.command.is_none());
        Ok(())
    }

    #[test]
    fn fs_subcommand_accepts_multiple_paths() -> Result<()> {
        let cli =
            Cli::try_parse_from(["scan", "fs", "/tmp/a", "/tmp/b"]).context("parse should work")?;
        match cli.command.context("fs subcommand expected")? {
            Commands::Path { paths, .. } => {
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
    fn fs_hopper_flag_and_upload_alias_parse() -> Result<()> {
        for flag in ["--hopper", "--upload"] {
            let cli = Cli::try_parse_from(["scan", "fs", flag, "http://hopper:8081", "/tmp/a"])
                .context("parse should work")?;
            match cli.command.context("fs subcommand expected")? {
                Commands::Path { paths, hopper, .. } => {
                    assert_eq!(paths, vec![PathBuf::from("/tmp/a")]);
                    assert_eq!(hopper.as_deref(), Some("http://hopper:8081"));
                }
                other => anyhow::bail!("unexpected command: {other:?}"),
            }
        }
        Ok(())
    }

    #[test]
    fn fs_without_hopper_defaults_to_none() -> Result<()> {
        let cli = Cli::try_parse_from(["scan", "fs", "/tmp/a"]).context("parse should work")?;
        match cli.command.context("fs subcommand expected")? {
            Commands::Path { hopper, .. } => assert!(hopper.is_none()),
            other => anyhow::bail!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn path_subcommand_and_its_aliases_resolve() -> Result<()> {
        for name in ["path", "fs", "scan"] {
            let cli = Cli::try_parse_from(["scan", name, "/tmp/a"])
                .with_context(|| format!("parse of `{name}` should work"))?;
            match cli
                .command
                .with_context(|| format!("path expected via `{name}`"))?
            {
                Commands::Path { paths, .. } => {
                    assert_eq!(paths, vec![PathBuf::from("/tmp/a")])
                }
                other => anyhow::bail!("unexpected command for `{name}`: {other:?}"),
            }
        }
        Ok(())
    }

    #[test]
    fn purl_subcommand_and_its_aliases_resolve() -> Result<()> {
        for name in ["purl", "pkg", "package", "pkgs"] {
            let cli = Cli::try_parse_from(["scan", name, "pkg:npm/left-pad@1.3.0"])
                .with_context(|| format!("parse of `{name}` should work"))?;
            match cli
                .command
                .with_context(|| format!("purl expected via `{name}`"))?
            {
                Commands::Purl { purl } => assert_eq!(purl, "pkg:npm/left-pad@1.3.0"),
                other => anyhow::bail!("unexpected command for `{name}`: {other:?}"),
            }
        }
        Ok(())
    }

    #[test]
    fn serve_and_worker_accept_negative_max_rss_disable() -> Result<()> {
        let cli = Cli::try_parse_from([
            "scan",
            "serve",
            "--bind",
            "127.0.0.1:49999",
            "--max-rss-gb",
            "-1",
        ])
        .context("serve -1 should parse")?;
        match cli.command.context("serve subcommand expected")? {
            Commands::Serve {
                bind, max_rss_gb, ..
            } => {
                assert_eq!(bind, "127.0.0.1:49999".parse::<SocketAddr>()?);
                assert_eq!(max_rss_gb, -1);
            }
            other => anyhow::bail!("unexpected command: {other:?}"),
        }

        let cli = Cli::try_parse_from([
            "scan",
            "worker",
            "--url",
            "http://127.0.0.1:8081",
            "--max-rss-gb",
            "-1",
        ])
        .context("worker -1 should parse")?;
        match cli.command.context("worker subcommand expected")? {
            Commands::Worker { max_rss_gb, .. } => assert_eq!(max_rss_gb, -1),
            other => anyhow::bail!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn max_rss_semantics_match_for_disabled_and_explicit_values() {
        assert_eq!(resolve_process_max_rss_bytes(-1), 0);
        assert_eq!(resolve_worker_max_rss_gb(-1), 0);

        assert_eq!(resolve_process_max_rss_bytes(3), 3 * GIB);
        assert_eq!(resolve_worker_max_rss_gb(3), 3);

        assert!(resolve_process_max_rss_bytes(0) > 0);
        assert!(resolve_worker_max_rss_gb(0) > 0);
    }

    #[test]
    fn max_rss_policy_classifies_cli_inputs() {
        assert_eq!(MaxRssPolicy::from_cli(-1), MaxRssPolicy::Disabled);
        assert_eq!(MaxRssPolicy::from_cli(i64::MIN), MaxRssPolicy::Disabled);
        assert_eq!(MaxRssPolicy::from_cli(0), MaxRssPolicy::Auto);
        assert_eq!(
            MaxRssPolicy::from_cli(7),
            MaxRssPolicy::Explicit(NonZeroU64::new(7).expect("non-zero"))
        );
    }

    #[test]
    fn server_config_treats_zero_max_rss_as_disabled() {
        let cfg = scan::server::ServerConfig::new(
            "127.0.0.1:0".parse().expect("addr"),
            1024 * 1024,
            0,
            "/tmp/models",
            None,
            0,
            vec![],
            None,
            1,
            vec![],
        )
        .expect("valid config");
        assert!(cfg.max_rss_bytes().is_none(), "0 must mean disabled");

        let cfg = scan::server::ServerConfig::new(
            "127.0.0.1:0".parse().expect("addr"),
            1024 * 1024,
            8 * GIB,
            "/tmp/models",
            None,
            0,
            vec![],
            None,
            1,
            vec![],
        )
        .expect("valid config");
        assert_eq!(
            cfg.max_rss_bytes().map(NonZeroU64::get),
            Some(8 * GIB),
            "explicit limit must round-trip through accessor"
        );
    }

    #[test]
    fn level_flag_parses_and_shortcuts_removed() -> Result<()> {
        let cli =
            Cli::try_parse_from(["scan", "-l", "100", "/tmp/a"]).context("-l 100 should parse")?;
        assert_eq!(cli.level, Some(100));

        let cli = Cli::try_parse_from(["scan", "--level", "12", "/tmp/a"])
            .context("--level 12 should parse")?;
        assert_eq!(cli.level, Some(12));

        // Out-of-range rejected (0..=25000, per-100M since the per-million migration).
        assert!(Cli::try_parse_from(["scan", "-l", "25001", "/tmp/a"]).is_err());
        // --level still conflicts with explicit thresholds.
        assert!(
            Cli::try_parse_from(["scan", "-l", "10", "--threshold-hostile", "0.5", "/tmp/a"])
                .is_err()
        );
        // The numeric shortcuts and their aliases were removed.
        assert!(Cli::try_parse_from(["scan", "-5", "/tmp/a"]).is_err());
        assert!(Cli::try_parse_from(["scan", "--loose", "/tmp/a"]).is_err());
        assert!(Cli::try_parse_from(["scan", "--paranoid", "/tmp/a"]).is_err());
        Ok(())
    }

    #[test]
    fn gzip_long_aliases_are_not_accepted() {
        assert!(Cli::try_parse_from(["scan", "--fast", "/tmp/a"]).is_err());
        assert!(Cli::try_parse_from(["scan", "--best", "/tmp/a"]).is_err());
    }

    #[test]
    fn severity_level_flags_conflict_with_each_other_and_manual_thresholds() {
        assert!(Cli::try_parse_from(["scan", "-1", "-9", "/tmp/a"]).is_err());
        assert!(
            Cli::try_parse_from(["scan", "-9", "--threshold-hostile", "0.90", "/tmp/a"]).is_err()
        );
    }
}
