//! Atomdrift Scan (`atomscan`) — ML-powered malware classification CLI.

/// Abort-on-exhaustion `malloc`/`calloc`/`realloc` for tree-sitter over one
/// allocator family, and `route_tree_sitter`, which installs them.
///
/// tree-sitter's C core otherwise allocates through the system malloc, outside
/// the process allocator's budgets, decay policy and heap profiles; on macOS
/// the default small-object zone also keeps freed pages resident. The shims go
/// through [`tree_sitter::set_allocator`] rather than raw `ts_set_allocator`:
/// the crate keeps its own free function for C strings it releases, and
/// bypassing the wrapper would free these pointers with libc `free`.
///
/// tree-sitter trusts a custom allocator — its array growth writes through a
/// `realloc` result unchecked — while the allocators here return null on
/// exhaustion. Passing that through would turn memory pressure while parsing a
/// hostile file into a wild write, so the shims abort, as tree-sitter's own
/// defaults do.
macro_rules! tree_sitter_allocator {
    ($malloc:path, $calloc:path, $realloc:path, $free:path) => {
        /// Route tree-sitter's C-core allocations through this allocator.
        ///
        /// # Safety
        ///
        /// Inherits [`tree_sitter::set_allocator`]'s contract: no tree-sitter
        /// API may have been called yet, no tree-sitter object may be live, and
        /// no other thread may be in tree-sitter. Call once, first thing in
        /// `main`.
        pub(super) unsafe fn route_tree_sitter() {
            // SAFETY: the four functions are one allocator family and satisfy
            // libc malloc alignment; the ordering and thread-exclusivity
            // clauses are this function's documented precondition.
            unsafe {
                tree_sitter::set_allocator(Some(tree_sitter::Allocator {
                    malloc: ts_malloc,
                    calloc: ts_calloc,
                    realloc: ts_realloc,
                    free: $free,
                }));
            }
        }

        /// # Safety
        ///
        /// The C `malloc` contract.
        unsafe extern "C" fn ts_malloc(size: usize) -> *mut std::ffi::c_void {
            // SAFETY: the allocator's malloc accepts any size.
            let ptr = unsafe { $malloc(size) };
            if ptr.is_null() && size != 0 {
                std::process::abort();
            }
            ptr
        }

        /// # Safety
        ///
        /// The C `calloc` contract.
        unsafe extern "C" fn ts_calloc(count: usize, size: usize) -> *mut std::ffi::c_void {
            // SAFETY: the allocator's calloc accepts any count and size, and
            // returns null on overflow.
            let ptr = unsafe { $calloc(count, size) };
            if ptr.is_null() && count != 0 && size != 0 {
                std::process::abort();
            }
            ptr
        }

        /// # Safety
        ///
        /// The C `realloc` contract: `ptr` is null or came from this allocator.
        unsafe extern "C" fn ts_realloc(
            ptr: *mut std::ffi::c_void,
            size: usize,
        ) -> *mut std::ffi::c_void {
            // SAFETY: forwarded under the caller's contract.
            let grown = unsafe { $realloc(ptr, size) };
            if grown.is_null() && size != 0 {
                std::process::abort();
            }
            grown
        }
    };
}

/// jemalloc, plus the compile-time tuning it reads at initialization.
///
/// Allocator and configuration share one `cfg` so they cannot drift apart: a
/// build that uses the system allocator must not leave a `_rjem_malloc_conf`
/// symbol behind, and one that uses jemalloc must never be left unconfigured.
/// The excluded targets ship a native allocator (see Cargo.toml); FreeBSD's
/// *is* jemalloc, but it reads the unprefixed `MALLOC_CONF` and
/// `/etc/malloc.conf`, so this symbol would not reach it anyway.
///
/// The string lives in cleave (`cleave::JEMALLOC_CONF`) so every binary running
/// cleave's analysis gets the same allocator behaviour; that constant documents
/// each option.
///
/// Runtime configuration still wins: jemalloc applies its compiled-in string
/// first, then `/etc/malloc.conf`, then the environment, later sources
/// overriding earlier ones per key. The variable is `_RJEM_MALLOC_CONF` —
/// `tikv-jemallocator` builds jemalloc with the `_rjem_` prefix, so plain
/// `MALLOC_CONF` is ignored.
#[cfg(all(
    unix,
    not(any(
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "illumos",
        target_os = "solaris",
    ))
))]
mod allocator {
    #[global_allocator]
    static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

    /// A `Sync` wrapper so a raw `*const c_char` can live in a static;
    /// jemalloc reads the pointer, it is never written after link time.
    #[repr(transparent)]
    struct SyncPtr(*const std::os::raw::c_char);
    // SAFETY: the pointer targets a NUL-terminated 'static CStr and is never
    // mutated, so sharing it across threads is sound.
    unsafe impl Sync for SyncPtr {}

    #[unsafe(no_mangle)]
    static _rjem_malloc_conf: SyncPtr = SyncPtr(cleave::JEMALLOC_CONF.as_ptr());

    // `tikv-jemalloc-sys` is the crate `tikv-jemallocator` binds, so these
    // signatures and the symbol prefix track the jemalloc actually linked.
    tree_sitter_allocator!(
        tikv_jemalloc_sys::malloc,
        tikv_jemalloc_sys::calloc,
        tikv_jemalloc_sys::realloc,
        tikv_jemalloc_sys::free
    );
}

/// mimalloc, the Windows counterpart of the jemalloc module: the CRT heap
/// convoys under many-way archive-member analysis.
#[cfg(windows)]
mod allocator {
    #[global_allocator]
    static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

    tree_sitter_allocator!(
        libmimalloc_sys::mi_malloc,
        libmimalloc_sys::mi_calloc,
        libmimalloc_sys::mi_realloc,
        libmimalloc_sys::mi_free
    );
}

/// The platform's own allocator, which tree-sitter already uses.
#[cfg(not(any(
    windows,
    all(
        unix,
        not(any(
            target_os = "freebsd",
            target_os = "dragonfly",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "illumos",
            target_os = "solaris",
        ))
    )
)))]
mod allocator {
    /// Nothing to route.
    ///
    /// # Safety
    ///
    /// None; `unsafe` only to match the other allocators' signature.
    pub(super) unsafe fn route_tree_sitter() {}
}

#[path = "main/version.rs"]
mod version;

use anyhow::{Context, Result};
use clap::builder::TypedValueParser as _;
use clap::{CommandFactory, Parser, Subcommand};
use scan::cli::GlobalArgs;
use scan::interpret::InterpretConfig;
use scan::memory::{MaxRssPolicy, log_max_rss_resolution};
use std::ffi::OsString;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;
use tracing_subscriber::prelude::*;

/// How long a daemon's runtime waits, once it returns, for blocking work —
/// cancelled analyses unwinding, their temp files coming off disk — before the
/// process exits anyway. Dropping the runtime instead waits forever, and one
/// wedged analysis would then hold the process past every drain deadline.
const RUNTIME_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

#[derive(Parser)]
#[command(name = "atomscan")]
#[command(version)]
#[command(about = "Atomdrift Scan — context-free malware detection (ML + static analysis)")]
#[command(
    after_help = "Bare paths run `path`: `atomscan ~/Downloads` is `atomscan path ~/Downloads`, \
and takes every flag that subcommand accepts."
)]
struct Cli {
    #[command(flatten)]
    global: GlobalArgs,

    #[command(subcommand)]
    command: Option<Commands>,
}

/// The binary's online default: everything the artifact itself reaches, but not
/// CI — GitHub Actions run only on a runner and never land in an installed
/// artifact, so auditing them is an explicit
/// `--follow=ci-actions`/`--follow=all` opt-in.
/// Keep `FetchPolicy::default` offline for the library API.
///
/// Must agree with the `default_missing_value` on `Cli::follow`, so a bare
/// `--follow` and an absent one select the same targets.
fn default_cli_follow_policy() -> scan::fetch::FetchPolicy {
    scan::fetch::FetchPolicy {
        urls: true,
        packages: true,
        deps: true,
        ..scan::fetch::FetchPolicy::default()
    }
}

/// Publish the known-good/known-bad filters process-wide, so
/// `scan::fetch::age_gate` can skip a dependency whose coordinate is already
/// vouched. This is deliberately separate from `ScanConfig::with_bloom`, which
/// additionally lets the bloom short-circuit the *scan target* itself: every
/// mode wants the dependency skip, but only a bulk walk wants its own input
/// answered from a bless.
///
/// `--mode slow` means "consult no filters", so it publishes nothing. Note this
/// reads the operator's `--mode` rather than the effective mode: `serve` forces
/// itself slow so a submitted job is always analyzed on its own merits, and
/// that internal choice must not also switch off its dependency skip — only an
/// explicit `--mode slow` does.
///
/// Returns the loaded filters so `fs`/`ps` can hand the same copy to
/// `ScanConfig::with_bloom` instead of loading a second one.
fn publish_bloom_filters(mode: scan::Mode) -> Option<std::sync::Arc<scan::bloom_repo::Lookup>> {
    (mode != scan::Mode::Slow).then(|| {
        let lookup = std::sync::Arc::new(scan::bloom_repo::Lookup::load());
        scan::bloom_repo::set_global(std::sync::Arc::clone(&lookup));
        lookup
    })
}

/// `config` with the per-file sha256 known-good/known-bad short-circuit, for
/// scans that discover their own files (a walk, the process table). Slow mode
/// consults no filters.
fn with_bloom_filters(config: scan::ScanConfig, mode: scan::Mode) -> scan::ScanConfig {
    match publish_bloom_filters(mode) {
        Some(lookup) => config.with_bloom(mode, lookup),
        None => config,
    }
}

/// The subcommand a bare `atomscan <path>` stands for.
const DEFAULT_SUBCOMMAND: &str = "fs";

/// Rewrite a command line so bare paths run the file scanner with every flag
/// `fs` accepts.
///
/// clap resolves a flag against the command it is typed under, so `--upload`,
/// which belongs to `fs` rather than to the top-level command, was rejected
/// outright in the shorthand form — and a second, hand-built `Commands::Path`
/// had to guess the rest from the environment. Inserting the subcommand the
/// shorthand stands for makes it a true alias: one parse, one set of flags, no
/// second construction path to keep in sync.
///
/// Declined when the line already names a subcommand, asks for help or version,
/// or carries nothing that could be a path; each of those means the top-level
/// parser, which still owns the globals, should see the line as typed.
fn with_default_subcommand<I, T>(args: I) -> Vec<OsString>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let args: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let Some((program, rest)) = args.split_first() else {
        return args;
    };
    let command = Cli::command();
    // The flags a shorthand line may carry: the globals, and `fs`'s own.
    let flags: Vec<&clap::Arg> = command
        .get_arguments()
        .chain(
            command
                .find_subcommand(DEFAULT_SUBCOMMAND)
                .into_iter()
                .flat_map(clap::Command::get_arguments),
        )
        .collect();
    let mut tokens = rest.iter();
    let mut operand = None;
    // Everything after `--` is a path by definition, never a subcommand.
    let mut escaped_path = false;
    while let Some(arg) = tokens.next() {
        if arg == "--" {
            escaped_path = tokens.next().is_some();
            break;
        }
        let Some(text) = arg.to_str() else {
            // Only a path can be other than UTF-8.
            operand = operand.or(Some(arg));
            continue;
        };
        if matches!(text, "-h" | "--help" | "-V" | "--version") {
            return args;
        }
        if text.len() > 1 && text.starts_with('-') {
            // A flag's value is not an operand, even when it looks like one.
            let next = tokens.as_slice().first().and_then(|next| next.to_str());
            if takes_next_value(&flags, text, next) {
                tokens.next();
            }
        } else if operand.is_none() {
            operand = Some(arg);
        }
    }
    // The first operand is where clap looks for a subcommand. With none there
    // is nothing to scan: the top-level parser answers, which is what prints
    // the help for a bare `atomscan`.
    let is_subcommand = |name: &str| {
        name == "help"
            || command
                .get_subcommands()
                .any(|sub| sub.get_name() == name || sub.get_all_aliases().any(|a| a == name))
    };
    match operand {
        Some(first) if first.to_str().is_some_and(is_subcommand) => return args,
        None if !escaped_path => return args,
        _ => {}
    }
    let mut rewritten = Vec::with_capacity(args.len() + 1);
    rewritten.push(program.clone());
    rewritten.push(OsString::from(DEFAULT_SUBCOMMAND));
    rewritten.extend(rest.iter().cloned());
    rewritten
}

/// Whether clap reads `next`, the token after `flag`, as its value: a long flag
/// that takes one, unless given as `--flag=value` or only with `=`, or a short
/// cluster ending in one with nothing attached (`-f json`, not `-fjson`). A
/// flag whose value is optional (`--llm`) leaves a following flag alone, as
/// clap does.
fn takes_next_value(flags: &[&clap::Arg], flag: &str, next: Option<&str>) -> bool {
    let next_is_flag = next.is_some_and(|next| next.len() > 1 && next.starts_with('-'));
    let takes = |arg: &&clap::Arg| {
        arg.get_action().takes_values()
            && !arg.is_require_equals_set()
            && !(next_is_flag && arg.get_num_args().is_some_and(|n| n.min_values() == 0))
    };
    if let Some(long) = flag.strip_prefix("--") {
        return !long.contains('=')
            && flags
                .iter()
                .any(|arg| arg.get_long() == Some(long) && takes(arg));
    }
    let shorts = flag.strip_prefix('-').unwrap_or(flag);
    for (i, short) in shorts.char_indices() {
        if flags
            .iter()
            .any(|arg| arg.get_short() == Some(short) && takes(arg))
        {
            return i + short.len_utf8() == shorts.len();
        }
    }
    false
}

/// Resolve the hopper destination consistently for every scan mode. Clap fills
/// a subcommand's `--hopper` from `SCAN_HOPPER`; this fallback covers the scan
/// modes that have no such flag of their own.
fn resolve_hopper(hopper: Option<String>) -> Option<String> {
    let hopper = resolve_hopper_value(hopper, scan::env::text(scan::env::SCAN_HOPPER));
    if let Some(raw) = hopper.as_deref() {
        scan::hopper::warn_if_cleartext(raw);
    }
    hopper
}

fn resolve_hopper_value(hopper: Option<String>, env: Option<String>) -> Option<String> {
    hopper.or(env).filter(|url| !url.trim().is_empty())
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Experimental: render saved JSON locally, without models, updates, or fetching. With
    /// `--llm`, print the LLM's grade of the interpret render as `{grade, model, reason}` JSON.
    Render {
        /// Saved report input.
        #[command(flatten)]
        args: cleave::report_input::RenderArgs,
    },

    /// Scan files or directories for hostile/suspicious content
    #[command(aliases = ["fs", "scan"])]
    Path {
        /// Paths to files or directories to scan
        #[arg(required = true, num_args = 1..)]
        paths: Vec<PathBuf>,

        /// Renew each result on a hopper instance by POSTing its envelope to
        /// `<URL>/api/result`. A sample hopper has not seen is uploaded first,
        /// bytes and provenance, so it lands as its own record. Upload failures
        /// are reported as errors, but never make the scan fatal. Also settable
        /// via the `SCAN_HOPPER` env var. Authenticates with `~/.tok/hopper`;
        /// hopper rejects an unauthenticated request with 401.
        #[arg(
            long,
            visible_alias = "upload",
            value_name = "URL",
            env = scan::env::SCAN_HOPPER.name
        )]
        hopper: Option<String>,

        /// Apply pre-collected registry metadata per scanned file, so an offline
        /// scan reasons over the same registry facts a live `pkg`/`url` scan would
        /// — without refetching, and even when the package has since been pulled.
        /// Takes a JSON object mapping each file's sha256 to its hopper sidecar,
        /// a `{record,sources}` envelope, or a bare registry record. A file
        /// absent from the map scans normally. Also settable via the
        /// `SCAN_REGISTRY_MAP` env var.
        #[arg(long, value_name = "FILE", env = scan::env::SCAN_REGISTRY_MAP.name)]
        registry_map: Option<PathBuf>,

        /// Treat collector capture records (`<file>.forage.json` beside
        /// `<file>`) found while walking a directory as bookkeeping: still
        /// analyzed, but shown only when suspicious or hostile, and uploaded
        /// to hopper only when hostile. Without it they scan like any file.
        #[arg(long)]
        enable_sidecar: bool,
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

        /// Push the fetched artifact to a hopper instance: bytes and provenance
        /// when hopper does not already have the SHA, then the verdict. Also
        /// settable via the `SCAN_HOPPER` env var. Authenticates with
        /// `~/.tok/hopper`; hopper rejects an unauthenticated request with 401.
        #[arg(
            long,
            visible_alias = "upload",
            value_name = "URL",
            env = scan::env::SCAN_HOPPER.name
        )]
        hopper: Option<String>,
    },

    /// Fetch a package by PURL and scan it (e.g. npm/left-pad@1.3.0)
    #[command(aliases = ["pkg", "package", "pkgs"])]
    Purl {
        /// Package URL to resolve, fetch, and scan. The `pkg:` scheme is
        /// optional (`npm/foo` == `pkg:npm/foo`). A versionless PURL resolves
        /// to the registry's current release.
        purl: String,

        /// Push the fetched package to a hopper instance: bytes and provenance
        /// (including the resolved registry record) when hopper does not
        /// already have the SHA, then the verdict. Also settable via the
        /// `SCAN_HOPPER` env var. Authenticates with `~/.tok/hopper`; hopper
        /// rejects an unauthenticated request with 401.
        #[arg(
            long,
            visible_alias = "upload",
            value_name = "URL",
            env = scan::env::SCAN_HOPPER.name
        )]
        hopper: Option<String>,
    },

    /// Update models, cleave traits, and the known-file filters
    UpdateRules,

    /// Validate the model bundle and benign fixture corpus
    Validate,

    /// Run as an HTTP classification server
    Serve(ServeArgs),

    /// Print version information
    Version,
}

// `serve`'s flags. No doc comment: clap would promote it onto the subcommand,
// replacing the variant's own description in `--help`.
#[derive(Debug, clap::Args)]
struct ServeArgs {
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
    #[arg(
        long,
        default_value = "0",
        allow_hyphen_values = true,
        value_parser = clap::value_parser!(i64).map(MaxRssPolicy::from_cli)
    )]
    max_rss_gb: MaxRssPolicy,

    /// Let `POST /v1/analyze?path=` scan files under DIR. Repeatable. Refused
    /// unless bound to 127.0.0.1, or both --allow-cidr and --token-file are set.
    #[arg(long, value_name = "DIR")]
    allow_local_paths: Vec<PathBuf>,

    /// Maximum concurrent analyses (default: three per physical core, at
    /// least 2). Past it a request is refused with 429, not queued.
    #[arg(long)]
    workers: Option<NonZeroUsize>,

    /// Comma-separated CIDR networks (in addition to loopback) allowed to
    /// reach the server. Pair with --bind 0.0.0.0:PORT to actually accept
    /// remote connections.
    #[arg(long)]
    allow_cidr: Option<String>,

    /// Require `Authorization: Bearer <token>` on every route except
    /// `/_/health`, reading the token from the first non-empty line of
    /// PATH. Loopback is not exempt: behind a tunnel or reverse proxy every
    /// request arrives from loopback. A missing, empty, or unreadable
    /// file is a startup error — never a silent drop to unauthenticated.
    #[arg(long, value_name = "PATH")]
    token_file: Option<PathBuf>,

    /// The hopper this server reads from and files to.
    ///
    /// Every analyzed result (parent and members) is renewed by POSTing to
    /// `<URL>/api/result` as analyses complete, and a lookup this server's
    /// own index cannot answer is deferred to the same place.
    ///
    /// Several addresses may be given, comma-separated, in preference order:
    /// put a read replica first and the primary behind it. Reads try them in
    /// that order, and a retried write walks down the list, so a replica that
    /// stops answering costs one attempt rather than a lost verdict.
    ///
    /// `--hopper https://hopper-replica.example.com,https://hopper.example.com`
    ///
    /// Upload failures are reported as errors, but never make the scan
    /// fatal. Also settable via the `SCAN_HOPPER` env var. Authenticates
    /// with `~/.tok/hopper`; hopper rejects an unauthenticated request with
    /// 401.
    #[arg(
        long,
        visible_alias = "upload",
        value_name = "URL",
        env = scan::env::SCAN_HOPPER.name
    )]
    hopper: Option<String>,

    /// Per-request analysis timeout in seconds; 0 disables. Raise when
    /// `--follow` is on and dependency analysis can exceed the default.
    #[arg(
        long,
        default_value_t = scan::server::DEFAULT_ANALYSIS_TIMEOUT_SECS,
        value_name = "SECS"
    )]
    analysis_timeout: u64,
}

impl Commands {
    /// An interactive scan: one target or set of targets, then exit.
    fn is_scan(&self) -> bool {
        matches!(
            self,
            Self::Path { .. } | Self::Ps | Self::Sys | Self::Url { .. } | Self::Purl { .. }
        )
    }

    /// A long-lived role that scans on behalf of other hosts.
    fn is_daemon(&self) -> bool {
        matches!(self, Self::Serve(_))
    }
}

fn main() -> ExitCode {
    // SAFETY: the first statement of `main` — no tree-sitter API has run, no
    // tree-sitter object is live, and no threads have been spawned yet.
    unsafe { allocator::route_tree_sitter() };
    // SAFETY: still the first statements of `main`. No thread has been
    // spawned and nothing has read the environment yet.
    unsafe { scan::runtime::install() };

    let cli = Cli::parse_from(with_default_subcommand(std::env::args_os()));
    match run(cli) {
        Ok(code) => code,
        // The reader went away (`atomscan … | head`) before anything was
        // flagged: nothing is listening for an error, so stop quietly. (A
        // flagged scan returns its summary instead, and exits on its verdict.)
        Err(e) if caused_by(&e, std::io::ErrorKind::BrokenPipe) => ExitCode::SUCCESS,
        // Ctrl-C stopped the scan: the conventional 128 + SIGINT, so a script
        // never mistakes a partial scan for a clean one.
        Err(e) if caused_by(&e, std::io::ErrorKind::Interrupted) => {
            eprintln!("Error: {e:#}");
            ExitCode::from(130)
        }
        // What `main() -> Result` printed, so scripts see the same text.
        Err(e) => {
            eprintln!("Error: {e:?}");
            ExitCode::FAILURE
        }
    }
}

/// Whether an I/O error of `kind` appears anywhere in `e`'s cause chain.
fn caused_by(e: &anyhow::Error, kind: std::io::ErrorKind) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == kind)
    })
}

/// Set up the process for `command`, then run it.
fn run(cli: Cli) -> Result<ExitCode> {
    scan::env::check()?;
    let Cli { global, command } = cli;
    // Bare paths already became `fs` in `with_default_subcommand`, so a missing
    // subcommand here means an empty command line: print the help and stop.
    let Some(command) = command else {
        Cli::command().print_long_help()?;
        return Ok(ExitCode::SUCCESS);
    };
    if let Commands::Render { args } = &command {
        return render(&global, args);
    }
    let interpret = bootstrap(&global, &command)?;
    let fetch = fetch_policy(&global, &command);
    match command {
        Commands::Render { .. } => anyhow::bail!("render must run before analysis bootstrap"),
        Commands::Path {
            paths,
            hopper,
            registry_map,
            enable_sidecar,
        } => {
            let config =
                scan_config(&global, interpret, fetch, hopper)?.with_sidecars(enable_sidecar);
            scan_paths(&global, config, &paths, registry_map.as_deref())
        }
        Commands::Ps => {
            let config =
                with_bloom_filters(scan_config(&global, interpret, fetch, None)?, global.mode);
            Ok(finish_scan(&scan::ps::run(&config)?))
        }
        // `sys` is `ps` plus a walk of staging directories; both phases want
        // the filters a plain `ps` or `fs` gets.
        Commands::Sys => {
            let config =
                with_bloom_filters(scan_config(&global, interpret, fetch, None)?, global.mode);
            Ok(finish_scan(&scan::sys::run(&config)?))
        }
        // `url` and `purl` name one artifact, so neither gets `with_bloom`: the
        // thing the operator asked about is always fetched and scanned, never
        // answered from a bless. The filters are still published process-wide,
        // which is what `fetch::age_gate` reads to skip the *dependencies* the
        // scan discovers — the bulk case a bloom is actually for. `--mode slow`
        // opts out of consulting them at all.
        Commands::Url { url, hopper } => {
            let config = scan_config(&global, interpret, fetch, hopper)?;
            publish_bloom_filters(global.mode);
            Ok(finish_scan(&scan::pkg::run_url(&url, &config)?))
        }
        Commands::Purl { purl, hopper } => {
            let config = scan_config(&global, interpret, fetch, hopper)?;
            publish_bloom_filters(global.mode);
            Ok(finish_scan(&scan::pkg::run_pkg(&purl, &config)?))
        }
        Commands::Serve(args) => serve(global, args, interpret, fetch),
        Commands::UpdateRules => Ok(update_rules()),
        Commands::Validate => {
            let model_dir = scan::cli::resolve_model_dir(None)?;
            let level = scan::cli::operating_level(global.level, false, &model_dir);
            let config = scan::ScanConfig::new(model_dir, scan::OutputFormat::Terminal, None)?
                .with_extra(global.extra)
                .with_level(level)
                .with_zip_passwords(global.zip_passwords);
            scan::validate::run(&config)?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Version => {
            version::print(global.format)?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// `atomscan render`: a saved report, rendered locally with no models,
/// updates or fetching. `--llm` grades what a live query sends, which is the
/// interpret render.
fn render(global: &GlobalArgs, args: &cleave::report_input::RenderArgs) -> Result<ExitCode> {
    use std::io::Write as _;
    let llm = global.interpret_config()?;
    let format = if llm.is_some() {
        scan::OutputFormat::Interpret
    } else {
        global.format
    };
    let rendered = cleave::report_input::run(args, |loaded| {
        scan::engine::render_saved_report(loaded, format)
    })?;
    let out = match &llm {
        Some(cfg) => format!("{}\n", scan::interpret::grade_render(cfg, &rendered)?),
        None => rendered,
    };
    std::io::stdout().lock().write_all(out.as_bytes())?;
    Ok(ExitCode::SUCCESS)
}

/// Everything a command that analyzes needs before it starts: settings and
/// logging, the rules channel, the allocator check, the Rayon pool and
/// diagnostics, and — for interactive commands — the update notice and the
/// stale-rules refresh. Returns the LLM configuration, if any.
fn bootstrap(global: &GlobalArgs, command: &Commands) -> Result<Option<InterpretConfig>> {
    configure_process(global, command)?;
    init_logging(global, command.is_daemon());

    // Resolved after logging is up: with no hardcoded default, the model comes
    // from the endpoint's own listing, so which one was picked has to be
    // visible rather than swallowed by an uninitialized subscriber.
    let interpret = global.interpret_config()?;
    if let Some(cfg) = &interpret {
        tracing::info!(
            endpoint = %cfg.base_url,
            model = %cfg.model,
            "LLM interpretation enabled",
        );
    }

    // A held token picks the Uplink rules tree, `--uplink` insists on it, and
    // `ATOMDRIFT_UPLINK=0` opts out: decided before anything resolves one.
    cleave::enriched::resolve(global.uplink)?;
    // Stabilize cleave trait discovery before any cleave shared resources are
    // initialized. This avoids clone-into-existing-directory failures when the
    // default traits checkout was installed by cleave or another scan run.
    scan::traits_repo::prepare_runtime_env();

    scan::freebsd_malloc::check_settings();
    scan::runtime::install_rayon_pool();
    scan::runtime::install_diagnostics();

    #[cfg(debug_assertions)]
    tracing::warn!(
        "DEBUG binary — scan will be very slow; use `make release` for production builds"
    );

    // Warn about missing analysis tools for commands that will run cleave.
    if command.is_scan() || command.is_daemon() || matches!(command, Commands::Validate) {
        scan::tools::warn_missing();
    }
    // Interactive commands get a once-a-day, zero-telemetry update notice. The
    // long-running daemons are excluded — they refresh on restart and shouldn't
    // print transient notices into their logs — and so is `validate`.
    if !command.is_daemon() && !matches!(command, Commands::Validate) {
        scan::update_check::maybe_notify();
    }
    // Default-on refresh: bring rules + models (+ bloom) current when the local
    // ruleset is over 24h stale — or immediately with `-u/--update` — unless
    // `--no-update` disables it. Scanning commands only: the daemons refresh on
    // restart, and `version`/`update-rules` manage updates themselves.
    if command.is_scan() {
        // `--uplink` needs a token before the refresh below would ask for
        // one mid-download; asked here, once.
        if cleave::enriched::required() {
            cleave::rule_update::sign_in()?;
        }
        scan::auto_update::refresh_if_stale(
            global.refresh(),
            global.mode,
            global.format == scan::OutputFormat::Terminal,
        )?;
        // Where artifacts are judged, settled once the rules are: an uplink
        // scan that needed a token has been asked for it by now, and one whose
        // token was refused has fallen back to open source, and asks nothing.
        scan::remote::resolve(global.analyze, cleave::enriched::enabled())?;
    }
    Ok(interpret)
}

/// Settle the process's settings from its flags, and start the cache sweeper.
fn configure_process(global: &GlobalArgs, command: &Commands) -> Result<()> {
    scan::settings::configure(scan::settings::Settings {
        // A hard cap across one invocation. The daemons scan unboundedly many
        // jobs over their lifetime, so each job is bounded by the fetch
        // policy's per-file ceilings instead.
        fetch_total: (!command.is_daemon()).then_some(scan::settings::FetchTotal {
            fetches: scan::fetch::DEFAULT_MAX_TOTAL_FETCHES,
            bytes: scan::fetch::DEFAULT_MAX_TOTAL_SIZE,
        }),
        theme: theme(global, command),
        ..Default::default()
    })?;
    // Reclaim stale/oversized caches (stng strings, scan analysis+interpret,
    // fletch blobs). One detached, self-gated sweep for a CLI run; a recurring
    // loop for the never-exiting daemons. Non-blocking either way.
    scan::cache_cleanup::start(command.is_daemon());
    Ok(())
}

/// Install the tracing subscriber and quiet the `log` bridge.
fn init_logging(global: &GlobalArgs, daemon: bool) {
    // RUST_LOG (when set) wins over the mode-derived defaults, so profiling
    // runs can surface targeted modules (e.g. `cleave::mem_profile=info`)
    // without paying for full `--verbose` debug output.
    // `atomscan` is this binary's own target: startup diagnostics logged from
    // main.rs (rule refresh, authentication, LLM configuration) are not in the
    // `scan` library and were being filtered out of the daemons' logs.
    let explicit = global.verbose || scan::env::text(scan::env::RUST_LOG).is_some();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        if global.verbose {
            tracing_subscriber::EnvFilter::new("atomscan=debug,scan=debug,cleave=debug")
        } else if daemon {
            tracing_subscriber::EnvFilter::new("atomscan=info,scan=info,cleave=warn")
        } else {
            tracing_subscriber::EnvFilter::new("atomscan=warn,scan=warn,cleave=error")
        }
    });
    let layer = tracing_subscriber::fmt::layer()
        .with_thread_names(true)
        .with_writer(std::io::stderr);
    if daemon {
        tracing_subscriber::registry()
            .with(filter)
            .with(layer)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(layer.without_time())
            .init();
    }

    // Silence the `log` -> `tracing` bridge that `init()` just installed.
    //
    // Cargo.toml asks for `tracing-subscriber` without default features exactly
    // so this bridge is never built, but cleave and stng both take it *with*
    // defaults and feature unification wins, so `SubscriberInitExt::init()`
    // installs a `LogTracer` regardless. Every `log!()` in every dependency then
    // pays a tracing dispatcher lookup plus an `EnvFilter` directive walk.
    //
    // That is not a rounding error on hostile input. goblin's permissive PE
    // import walker emits one `warn!` per bad-RVA lookup-table entry, and a
    // forged import directory drives that loop into the millions, until the
    // analysis spends a large share of its time in the bridge rather than the
    // parse. `set_max_level(Off)` makes those call sites no-ops at the macro,
    // which is what the dependency comment intends.
    //
    // An operator who asked for logs still gets them: `--verbose` or an explicit
    // `RUST_LOG` leaves the bridge at its default level, so a dependency's `log`
    // output remains reachable when it is actually wanted.
    if !explicit {
        log::set_max_level(log::LevelFilter::Off);
    }
}

/// The terminal color theme. Detection is only for an interactive scan
/// drawing to a terminal: the OSC color-scheme query blocks on a TTY response
/// and hangs in any environment that doesn't reply (SSH, some tmux configs,
/// daemons).
fn theme(global: &GlobalArgs, command: &Commands) -> Option<scan::output::Theme> {
    if global.light {
        Some(scan::output::Theme::Light)
    } else if global.dark {
        Some(scan::output::Theme::Dark)
    } else if global.format == scan::OutputFormat::Terminal && command.is_scan() {
        Some(scan::output::detect_theme())
    } else {
        None
    }
}

/// The follow/fetch policy for `command`.
///
/// When `--follow`/`SCAN_FOLLOW` is unset, an interactive scan follows what an
/// installed artifact can reach (CI actions remain opt-in), while `serve`
/// follows everything (see [`scan::fetch::default_service_follow_policy`]).
/// An explicit selection is honored verbatim; the knobs always apply.
fn fetch_policy(global: &GlobalArgs, command: &Commands) -> scan::fetch::FetchPolicy {
    let daemon = command.is_daemon();
    let default_follow = if daemon {
        scan::fetch::default_service_follow_policy
    } else {
        default_cli_follow_policy
    };
    global.fetch_policy(
        global.follow.unwrap_or_else(default_follow),
        scan::fetch::DEFAULT_MAX_DEP_AGE_DAYS,
        daemon,
    )
}

/// The configuration every interactive scan runs with.
fn scan_config(
    global: &GlobalArgs,
    interpret: Option<InterpretConfig>,
    fetch: scan::fetch::FetchPolicy,
    hopper: Option<String>,
) -> Result<scan::ScanConfig> {
    // Resolved only here: `update-rules` and `version` don't need it, and
    // resolving triggers the auto-clone.
    let model_dir = scan::cli::resolve_model_dir(None)?;
    let level = scan::cli::operating_level(global.level, false, &model_dir);
    Ok(scan::ScanConfig::new(model_dir, global.format, None)?
        .with_filter(global.display_filter())
        .with_extra(global.extra)
        .with_level(level)
        .with_interpret(interpret)
        .with_fetch(fetch)
        .with_zip_passwords(global.zip_passwords.clone())
        .with_hopper(resolve_hopper(hopper)))
}

/// `atomscan path`: scan files and directories.
fn scan_paths(
    global: &GlobalArgs,
    config: scan::ScanConfig,
    paths: &[PathBuf],
    registry_map: Option<&Path>,
) -> Result<ExitCode> {
    // Applies to the files found by walking a directory the operator named; a
    // path named directly on the command line is always analyzed — see
    // `engine::named_target_opts`.
    let config = with_bloom_filters(config, global.mode);
    // `--registry-map <file>`: a JSON object {sha256: provenance}. Preserve each
    // complete value and derive the normalized record beside it, so map-backed
    // scans have the same provider data as live and hopper-backed scans. Entries
    // without a record are skipped — registry provenance is best-effort, never
    // required.
    let registry_map = match registry_map {
        Some(path) => {
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading registry map {}", path.display()))?;
            Some(
                scan::provenance::registry_map(&bytes)
                    .with_context(|| format!("parsing registry map {}", path.display()))?,
            )
        }
        None => None,
    };
    // Warm YARA + capability mapper off the rayon pool before any analysis
    // spawns rayon work. Directory scans run on a dedicated rayon pool; if any
    // of those workers is the first to hit `yara_engine()`, the init's internal
    // par_iter deadlocks against its peers parked on the OnceLock. Prefetching
    // from main (non-rayon) fills the OnceLock safely.
    scan::engine::prefetch_cleave_resources();
    // Explicit files are analyzed as one parallel batch and each directory is
    // streamed; run_paths shares one model load and verdict tally across all.
    let summary = scan::engine::run_paths(paths, &config, registry_map.as_ref())?;
    Ok(finish_scan(&summary))
}

/// Report a degraded YARA engine, then pick the exit status from the scan's
/// worst outcome: 1 hostile, 4 degraded, 2 suspicious, 3 errors, else 0.
///
/// A degraded engine (panic breaker tripped, or rule sources that failed to
/// compile at runtime) outranks everything but a hostile verdict: the verdicts
/// were made with fewer rules than the trait set defines, so the run must not
/// look like a clean scan. A hostile verdict stands even degraded, since rules
/// only add detections.
fn finish_scan(summary: &scan::ScanSummary) -> ExitCode {
    let degradation = cleave::yara_engine::yara_degradation();
    if let Some(msg) = &degradation {
        eprintln!("\n❌ YARA engine degraded during this run: {msg}");
        eprintln!(
            "   Verdicts above were produced WITHOUT the full rule set; treat them as incomplete."
        );
    }
    let code = if summary.hostile > 0 {
        1
    } else if degradation.is_some() {
        4
    } else if summary.suspicious > 0 {
        2
    } else if summary.errors > 0 {
        3
    } else {
        0
    };
    ExitCode::from(code)
}

/// `atomscan serve`.
fn serve(
    global: GlobalArgs,
    args: ServeArgs,
    interpret: Option<InterpretConfig>,
    fetch: scan::fetch::FetchPolicy,
) -> Result<ExitCode> {
    let hopper = resolve_hopper(args.hopper);
    let max_rss = args.max_rss_gb;
    log_max_rss_resolution(
        "server",
        max_rss,
        max_rss
            .process_ceiling()
            .map_or(0, std::num::NonZeroU64::get),
    );
    // `resolve` also refreshes rules on restart. That is the step that
    // installs the traits on a fresh deploy — without it the server starts,
    // reports healthy, and fails every analysis on a tree that never got
    // created.
    let config = scan::server::Startup {
        bind: args.bind,
        max_size_mb: args.max_size_mb,
        max_rss,
        allow_local_paths: args.allow_local_paths,
        workers: args.workers,
        allow_cidr: args.allow_cidr,
        token_file: args.token_file,
        hopper,
        analysis_timeout_secs: args.analysis_timeout,
        rules: scan::rules::RulesStartup {
            model_dir: None,
            level: global.level,
            thresholds: None,
            refresh: global.refresh(),
            slow_rule_ms: scan::cli::DEFAULT_SLOW_RULE_MS,
            interpret,
            fetch,
            zip_passwords: global.zip_passwords.into(),
        },
    }
    .resolve()?;
    if let Some(url) = config.hopper.as_deref() {
        tracing::info!(
            hopper = url,
            "renewing results on hopper and deferring unknown lookups to it"
        );
    }
    // Serve never bloom-skips an /analyze job (Mode::Slow), but the membership
    // endpoint and the --follow dependency gate read the process-wide handle.
    // Missing files fail closed (no skip).
    publish_bloom_filters(global.mode);
    tracing::info!(bind = %args.bind, "starting Atomdrift Scan server");
    // The runtime's blocking threads run small analyses inline, so they get
    // the stack every analysis pool has ([`scan::RAYON_STACK_MB`]): a hostile
    // sample that recurses fine on a pool must not overflow tokio's 2 MiB
    // default and abort the whole daemon.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(scan::RAYON_STACK_MB * 1024 * 1024)
        .build()?;
    let served = runtime.block_on(scan::server::run(config));
    runtime.shutdown_timeout(RUNTIME_SHUTDOWN_GRACE);
    served?;
    Ok(ExitCode::SUCCESS)
}

/// `atomscan update-rules`: bring every rule source up to date. Traits and
/// models failing is an error; the bloom and NSRL filters only let a scan skip
/// work, so their failure is a warning.
fn update_rules() -> ExitCode {
    let mut code = ExitCode::SUCCESS;
    for (source, e) in
        scan::rules::refresh(&scan::rules::Source::ALL, None, false, false).failures()
    {
        if source.optional() {
            eprintln!(
                "Warning: {} update failed (non-fatal): {e:#}",
                source.name()
            );
        } else {
            eprintln!("Error updating {}: {e:#}", source.name());
            code = ExitCode::FAILURE;
        }
    }
    code
}

#[cfg(test)]
#[path = "main/tests.rs"]
mod tests;
