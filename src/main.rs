//! Atomdrift Scan (`atomscan`) — ML-powered malware classification CLI.

// Doc comments on `clap` structs are user-facing `--help` text, so they carry
// `[EXPERIMENTAL]` tags, bare URLs, and `<URL>` placeholders that rustdoc would
// otherwise read as broken links or markup.
#![allow(
    rustdoc::broken_intra_doc_links,
    rustdoc::bare_urls,
    rustdoc::invalid_html_tags
)]

/// jemalloc, plus the compile-time tuning it reads at initialization.
///
/// Allocator and configuration share one `cfg` so they cannot drift apart: a
/// build that swaps in the system allocator must not leave a `_rjem_malloc_conf`
/// symbol behind, and one that uses jemalloc must never be left unconfigured.
/// On the excluded targets this crate uses the system allocator; on FreeBSD that
/// *is* jemalloc, but it reads the unprefixed `MALLOC_CONF` / `/etc/malloc.conf`,
/// so this symbol would not reach it anyway.
///
/// The string lives in cleave (`cleave::JEMALLOC_CONF`) so every binary running
/// cleave's analysis gets the same allocator behaviour; see that constant for
/// what each option buys and the measurements behind them.
///
/// Runtime configuration still wins: jemalloc applies its compiled-in string
/// first, then `/etc/malloc.conf`, then the environment, with later sources
/// overriding earlier ones per key. The environment variable is
/// `_RJEM_MALLOC_CONF` — `tikv-jemallocator` builds jemalloc with the `_rjem_`
/// prefix, so plain `MALLOC_CONF` is silently ignored.
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
mod jemalloc {
    #[global_allocator]
    static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

    /// A `Sync` wrapper so a raw `*const c_char` can live in a static;
    /// jemalloc reads the pointer, it is never written after link time.
    #[repr(transparent)]
    struct SyncPtr(*const std::os::raw::c_char);
    // SAFETY: the pointer targets a NUL-terminated 'static CStr and is never
    // mutated, so sharing it across threads is sound.
    unsafe impl Sync for SyncPtr {}

    #[allow(non_upper_case_globals)]
    #[unsafe(no_mangle)]
    static _rjem_malloc_conf: SyncPtr = SyncPtr(cleave::JEMALLOC_CONF.as_ptr());

    /// Route tree-sitter's C-core allocations through jemalloc. Its parse
    /// trees otherwise go to the system malloc — outside every jemalloc
    /// budget, decay policy, and heap profile this codebase relies on, and
    /// on macOS the default small-object zone retains freed pages
    /// (measured: ~0.8 GB of empty retained MALLOC_SMALL at the gauntlet
    /// peak).
    ///
    /// Installs via the tree-sitter crate's [`tree_sitter::set_allocator`]
    /// rather than raw `ts_set_allocator`: the crate keeps an internal free-fn
    /// for C strings it releases (query errors, etc.), and bypassing the
    /// wrapper would free jemalloc pointers with libc `free`. The four entry
    /// points come from `tikv-jemalloc-sys`, the same crate `tikv-jemallocator`
    /// binds, so their signatures and symbol prefix track the jemalloc actually
    /// linked in rather than a hand-copied guess.
    ///
    /// # Safety
    ///
    /// Inherits [`tree_sitter::set_allocator`]'s contract, whose unmet clauses
    /// are the caller's to discharge: no tree-sitter API may have been called
    /// yet, no tree-sitter object may be live, and no other thread may be in
    /// tree-sitter concurrently. In practice: call once, first thing in `main`.
    pub(super) unsafe fn route_tree_sitter_through_jemalloc() {
        // SAFETY: jemalloc's malloc/calloc/realloc/free are one allocator
        // family, never return null for non-zero sizes, and satisfy libc
        // malloc alignment. The ordering and thread-exclusivity clauses are
        // this function's own documented precondition.
        unsafe {
            tree_sitter::set_allocator(Some(tree_sitter::Allocator {
                malloc: tikv_jemalloc_sys::malloc,
                calloc: tikv_jemalloc_sys::calloc,
                realloc: tikv_jemalloc_sys::realloc,
                free: tikv_jemalloc_sys::free,
            }));
        }
    }
}

/// Windows counterpart of [`jemalloc`]: CRT heap was 12% exclusive on the
/// two-Go WPR profile, and 16 rayon workers convoy on the process heap.
/// Route tree-sitter's C allocator through the same arena so parse trees
/// are not a second heap.
#[cfg(all(windows, not(feature = "crt-heap")))]
mod mimalloc_alloc {
    #[global_allocator]
    static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

    /// # Safety
    ///
    /// Same contract as [`super::jemalloc::route_tree_sitter_through_jemalloc`]:
    /// call once, first thing in `main`, before any tree-sitter API.
    unsafe extern "C" fn calloc_compat(count: usize, size: usize) -> *mut std::ffi::c_void {
        let Some(bytes) = count.checked_mul(size) else {
            return std::ptr::null_mut();
        };
        let ptr = unsafe { libmimalloc_sys::mi_malloc(bytes) };
        if !ptr.is_null() && bytes > 0 {
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes) };
        }
        ptr
    }

    pub(super) unsafe fn route_tree_sitter_through_mimalloc() {
        unsafe {
            tree_sitter::set_allocator(Some(tree_sitter::Allocator {
                malloc: libmimalloc_sys::mi_malloc,
                calloc: calloc_compat,
                realloc: libmimalloc_sys::mi_realloc,
                free: libmimalloc_sys::mi_free,
            }));
        }
    }
}

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use scan::engine::DisplayFilter;
use scan::worker::{default_worker_name, default_workers};
use std::ffi::OsString;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process;
use tracing_subscriber::prelude::*;

const EXPECTED_YARA_CACHE_MISMATCH: &str =
    "compiled YARA rules do not match the rule sources on disk; ignoring them and recompiling";

fn is_expected_yara_cache_mismatch(target: &str, message: &str) -> bool {
    target == "cleave::yara_engine" && message.contains(EXPECTED_YARA_CACHE_MISMATCH)
}

/// Hide one expected cache invalidation without muting real YARA errors from
/// the same tracing target. The upstream event remains available whenever the
/// operator explicitly asks for diagnostic logs.
#[derive(Debug, Clone, Copy)]
struct ExpectedYaraCacheFilter {
    hide: bool,
}

#[derive(Default)]
struct EventMessage {
    text: String,
}

impl tracing::field::Visit for EventMessage {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.text = format!("{value:?}");
        }
    }
}

impl<S> tracing_subscriber::layer::Filter<S> for ExpectedYaraCacheFilter
where
    S: tracing::Subscriber,
{
    fn enabled(
        &self,
        _metadata: &tracing::Metadata<'_>,
        _context: &tracing_subscriber::layer::Context<'_, S>,
    ) -> bool {
        true
    }

    fn event_enabled(
        &self,
        event: &tracing::Event<'_>,
        _context: &tracing_subscriber::layer::Context<'_, S>,
    ) -> bool {
        if !self.hide || event.metadata().target() != "cleave::yara_engine" {
            return true;
        }
        let mut message = EventMessage::default();
        event.record(&mut message);
        !is_expected_yara_cache_mismatch(event.metadata().target(), &message.text)
    }
}

use scan::memory::{
    MaxRssPolicy, cgroup_memory_diagnostics, log_max_rss_resolution, proc_memtotal_mb,
    resolve_process_max_rss_bytes, resolve_worker_max_rss_gb, worker_memory_basis,
};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

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
    global: scan::cli::GlobalArgs,

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
/// reads the operator's `cli.global.mode` rather than the effective mode: `serve` and
/// `worker` force themselves slow so a submitted job is always analyzed on its
/// own merits, and that internal choice must not also switch off their
/// dependency skip — only an explicit `--mode slow` does.
fn publish_bloom_filters(mode: scan::Mode) {
    if mode != scan::Mode::Slow {
        scan::bloom_repo::set_global(std::sync::Arc::new(scan::bloom_repo::Lookup::load()));
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
    // Everything after `--` is a value by definition, so a subcommand name can
    // only appear before it.
    let named = rest
        .iter()
        .take_while(|arg| *arg != "--")
        .filter_map(|arg| arg.to_str());
    let mut subcommands = Cli::command();
    let subcommands: Vec<&str> = subcommands
        .get_subcommands_mut()
        .flat_map(|sub| {
            std::iter::once(sub.get_name()).chain(sub.get_all_aliases().collect::<Vec<_>>())
        })
        .collect();
    for arg in named {
        if subcommands.contains(&arg)
            || matches!(arg, "help" | "-h" | "--help" | "-V" | "--version")
        {
            return args;
        }
    }
    // No operand means nothing to scan: let the top-level parser answer, which
    // is what prints the help for a bare `atomscan`.
    let mut operands = rest
        .iter()
        .skip_while(|arg| *arg != "--")
        .skip(1)
        .peekable();
    let has_operand = operands.peek().is_some()
        || rest
            .iter()
            .take_while(|arg| *arg != "--")
            .any(|arg| !arg.as_encoded_bytes().starts_with(b"-"));
    if !has_operand {
        return args;
    }
    let mut rewritten = Vec::with_capacity(args.len() + 1);
    rewritten.push(program.clone());
    rewritten.push(OsString::from(DEFAULT_SUBCOMMAND));
    rewritten.extend(rest.iter().cloned());
    rewritten
}

/// Resolve the hopper destination consistently for every scan mode. Most
/// subcommands let clap populate their local `hopper` field from the
/// environment, but bare-path shorthand bypasses subcommand parsing.
fn resolve_hopper(hopper: Option<String>) -> Option<String> {
    resolve_hopper_value(hopper, std::env::var("SCAN_HOPPER").ok())
}

fn resolve_hopper_value(hopper: Option<String>, env: Option<String>) -> Option<String> {
    hopper.or(env).filter(|url| !url.trim().is_empty())
}

fn command_scans_for_other_hosts(command: Option<&Commands>) -> bool {
    matches!(
        command,
        Some(Commands::Serve { .. } | Commands::Worker { .. })
    )
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
        /// `<URL>/api/result`. A SHA hopper has not ingested is negotiated over
        /// `/api/known` and uploaded bytes-and-provenance first, so a
        /// never-before-seen sample lands as its own row instead of being
        /// dropped as an unknown-SHA no-op. Upload failures are reported as
        /// errors, but never make the scan fatal. Also settable via the
        /// `SCAN_HOPPER` env var.
        /// Authenticates with `~/.tok/hopper` (or `$HOPPER_TOKEN_FILE` /
        /// `$HOPPER_TOKEN`); hopper rejects an unauthenticated request with 401.
        #[arg(
            long,
            visible_alias = "upload",
            value_name = "URL",
            env = "SCAN_HOPPER"
        )]
        hopper: Option<String>,

        /// Apply pre-collected registry metadata per scanned file, so an offline
        /// scan reasons over the same registry facts a live `pkg`/`url` scan would
        /// — without refetching, and even when the package has since been pulled.
        /// Takes a JSON object mapping each file's sha256 to its complete hopper
        /// sidecar, a `{record,sources}` fletch envelope, or a legacy bare
        /// registry record. Complete inputs retain their raw provider data.
        /// A file absent from the map scans normally.
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

        /// Push the fetched artifact to a hopper instance: bytes and provenance
        /// when hopper does not already have the SHA, then the verdict. A
        /// fetched URL is exactly the never-before-seen case, so this is the
        /// flag that gets a freshly-discovered sample into the corpus.
        /// Also settable via the `SCAN_HOPPER` env var.
        /// Authenticates with `~/.tok/hopper` (or `$HOPPER_TOKEN_FILE` /
        /// `$HOPPER_TOKEN`); hopper rejects an unauthenticated request with 401.
        #[arg(
            long,
            visible_alias = "upload",
            value_name = "URL",
            env = "SCAN_HOPPER"
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
        /// when hopper does not already have the SHA, then the verdict. The
        /// registry record resolved for the package rides along as the
        /// sidecar's `registry` node, so the uploaded sample carries the same
        /// provenance a forager-collected one would.
        /// Also settable via the `SCAN_HOPPER` env var.
        /// Authenticates with `~/.tok/hopper` (or `$HOPPER_TOKEN_FILE` /
        /// `$HOPPER_TOKEN`); hopper rejects an unauthenticated request with 401.
        #[arg(
            long,
            visible_alias = "upload",
            value_name = "URL",
            env = "SCAN_HOPPER"
        )]
        hopper: Option<String>,
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

        /// Maximum concurrent analyses (defaults to the physical
        /// performance-core count, min 2)
        #[arg(long)]
        workers: Option<NonZeroUsize>,

        /// Comma-separated CIDR networks (in addition to loopback) allowed to
        /// reach the server. /analyze-path is always restricted to loopback
        /// regardless of this list. Pair with --bind 0.0.0.0:PORT to actually
        /// accept remote connections.
        #[arg(long)]
        allow_cidr: Option<String>,

        /// Require `Authorization: Bearer <token>` on every route except
        /// /_/health, reading the token from the first non-empty line of
        /// PATH. Loopback is not exempt: behind a Cloudflare tunnel every
        /// request arrives from loopback. A missing, empty, or unreadable
        /// file is a startup error — never a silent drop to unauthenticated.
        #[arg(long, value_name = "PATH")]
        token_file: Option<PathBuf>,

        /// Path to a writable cleave traits directory (overrides CLEAVE_TRAITS_DIR).
        /// Use when running as a restricted user whose $HOME is not writable
        /// (e.g. macOS system accounts where $HOME=/var/empty). Traits are
        /// cloned automatically if the directory does not yet exist.
        #[arg(long)]
        traits_dir: Option<PathBuf>,

        /// The hopper this server reads from and files to.
        ///
        /// Every analyzed result (parent and members) is renewed by POSTing to
        /// <URL>/api/result as analyses complete — the warm-server equivalent of
        /// `scan path --hopper` — and a lookup this server's own index cannot
        /// answer is deferred to the same place.
        ///
        /// Several addresses may be given, comma-separated, in preference
        /// order: put the replica first and the primary behind it. Reads try
        /// them in that order, and a retry on a write walks down the list, so a
        /// replica that stops answering costs one attempt rather than a lost
        /// verdict. Reads and writes deliberately take the same list — routing
        /// them apart is a topology this server would have to know, and
        /// hopper's write relay exists so that it does not: a replica answers
        /// lookups locally and forwards the renewals.
        ///
        ///   --hopper https://hops-ro.example,http://hopper.internal:8081
        ///
        /// Upload failures are reported as errors, but never make the scan
        /// fatal. Also settable via the `SCAN_HOPPER` env var. Authenticates
        /// with `~/.tok/hopper` (or
        /// `$HOPPER_TOKEN_FILE` / `$HOPPER_TOKEN`); hopper rejects an
        /// unauthenticated request with 401.
        #[arg(
            long,
            visible_alias = "upload",
            value_name = "URL",
            env = "SCAN_HOPPER"
        )]
        hopper: Option<String>,

        /// Fill idle capacity with queue work from `--hopper`, pausing the
        /// moment an analysis request arrives.
        ///
        /// A serve process spends most of its life waiting while hopper holds a
        /// backlog, so the spare capacity is otherwise wasted. Analysis
        /// requests always win: the worker stops claiming while any is in
        /// flight and remains paused for 7 seconds after the latest analysis
        /// request, so a new burst does not compete with background work.
        ///
        /// Defaults to half of `--workers` (rounded down), leaving the other
        /// half reserved for requests. The requested value is capped at that
        /// same half-slot limit. 0 disables it.
        /// Requires `--hopper`.
        ///
        /// When `--hopper` names several addresses this claims from the
        /// primary — the last of them — and only from it. A replica refuses
        /// worker routes with a 403 even with its relay enabled, so unlike
        /// lookups and renewals there is no second address to fall back to.
        #[arg(long, value_name = "N", env = "SCAN_IDLE_WORKER_SLOTS")]
        idle_worker_slots: Option<usize>,

        /// Per-request analysis timeout in seconds; 0 disables. Raise when
        /// `--fetch` is on and dependency analysis can exceed the default.
        #[arg(
            long,
            default_value_t = scan::server::DEFAULT_ANALYSIS_TIMEOUT_SECS,
            value_name = "SECS",
            env = "SCAN_ANALYSIS_TIMEOUT"
        )]
        analysis_timeout: u64,
    },

    /// Run as a pull-based worker, polling a hopper instance for analysis jobs
    Worker {
        /// Hopper API base URL (e.g. http://hopper-host:8081). Every call
        /// authenticates with `~/.tok/hopper` (or `$HOPPER_TOKEN_FILE` /
        /// `$HOPPER_TOKEN`); without it hopper rejects the poll with 401.
        ///
        /// Accepts the comma list `serve --hopper` takes, so one deploy
        /// variable can feed both, but a worker uses only the primary — the
        /// last address. A replica refuses worker routes outright, so the
        /// earlier ones are not a fallback here.
        /// Also settable via `SCAN_HOPPER`.
        #[arg(long, env = "SCAN_HOPPER")]
        url: String,

        /// Worker name (defaults to hostname)
        #[arg(long)]
        name: Option<String>,

        /// Number of concurrent analysis slots
        #[arg(short = 'j', long)]
        workers: Option<NonZeroUsize>,

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

/// Warn when `MALLOC_CONF` asks FreeBSD's in-libc jemalloc for a background
/// purge thread, which permanently breaks allocation.
///
/// FreeBSD builds libc's jemalloc without `JEMALLOC_BACKGROUND_THREAD` (libc
/// cannot depend on libthr), so `background_thread_boot0()` fails. It is called
/// from `malloc_init_hard()` *after* `malloc_init_state` has been set to
/// `malloc_init_recursible`, so init returns early and never reaches
/// `malloc_init_initialized`. `malloc_initialized()` is then false for the life
/// of the process: every allocation re-enters `malloc_init_hard()` and
/// serializes on the global `init_lock`, collapsing a many-core box to roughly
/// one allocating thread. jemalloc treats this as unsupported-not-invalid, so
/// `abort_conf:true` does not catch it and nothing is written to stderr — the
/// only symptom is throughput death, which is why this check exists.
///
/// Warn rather than fail: the setting is harmless (merely ignored) on the
/// platforms that link the bundled jemalloc, and an operator override should
/// never be able to refuse to boot a worker.
fn warn_on_broken_freebsd_malloc_conf() {
    if !cfg!(target_os = "freebsd") {
        return;
    }
    let conf = std::env::var("MALLOC_CONF").unwrap_or_default();
    // Only `background_thread:true` breaks init; an explicit `:false` is fine.
    if !conf.contains("background_thread:true") {
        return;
    }
    tracing::warn!(
        malloc_conf = %conf,
        "MALLOC_CONF sets background_thread:true, which FreeBSD's in-libc jemalloc does not \
         support: malloc initialization aborts partway and every allocation then serializes on \
         jemalloc's global init_lock. Expect near-total throughput collapse and analyses that \
         never finish. Remove background_thread from MALLOC_CONF.",
    );
}

/// Refresh models and traits before a long-lived daemon starts serving.
///
/// `force` is `-u/--update`: re-fetch even when the local copy looks current.
///
/// Both daemons re-read rules only at startup, so this is what a restart is
/// for. Failures are warnings, never fatal: a disconnected host must still come
/// up on whatever is already on disk. It runs *after* `--traits-dir` has been
/// applied, so a pinned traits directory is the one that gets installed into —
/// which is also how that directory comes to exist on a fresh deploy.
fn main() -> Result<()> {
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
    // SAFETY: the first statement of `main` — no tree-sitter API has run, no
    // tree-sitter object is live, and no threads have been spawned yet.
    unsafe {
        jemalloc::route_tree_sitter_through_jemalloc()
    };
    #[cfg(all(windows, not(feature = "crt-heap")))]
    // SAFETY: first statements of `main` — no tree-sitter API has run.
    unsafe {
        mimalloc_alloc::route_tree_sitter_through_mimalloc()
    };
    // SAFETY: still the first statements of `main`. No thread has been
    // spawned and nothing has read the environment yet.
    unsafe { scan::runtime::install() };

    let mut cli = Cli::parse_from(with_default_subcommand(std::env::args_os()));
    // Install the process-wide Rizin budget before any analysis or Rayon worker
    // can start. Filefacts owns the subprocess lifecycle; Atomscan only selects
    // the deadline through this single global configuration point.
    filefacts::rizin::set_timeout_secs(cli.global.rizin_timeout_secs);
    let selected_severity_level = cli.global.level;
    let threshold_suspicious = cli.global.threshold_suspicious;
    let threshold_hostile = cli.global.threshold_hostile;
    // The selection comes from `--follow`; the hop count from `--follow-depth`,
    // the dependency age ceiling from `--fetch-max-age`, and the per-file ceilings
    // from `--fetch-max-file-*` (each its own flag/env). When `--follow`/`SCAN_FOLLOW`
    // is unset, the binary defaults to dependencies and executable references
    // in every mode (CI actions remain opt-in). An explicit selection is honored
    // verbatim in both; the knobs always apply. `GlobalArgs::fetch_policy` owns
    // that mapping so every front-end applies the flags identically.
    // The per-fetch size ceiling is enforced in the HTTP layer, so it's a
    // process-global rather than a per-policy field — set it once here and every
    // mode (interactive scan and worker alike) honors `--fetch-max-size`.
    scan::fetch::set_max_fetch_bytes(cli.global.fetch_max_size);
    // Registry-metadata staleness bound: a process-global for the same reason,
    // consulted by every registry lookup. `None` keeps the tiered defaults.
    scan::fetch::set_registry_ttl(cli.global.registry_ttl);
    let scans_for_other_hosts = command_scans_for_other_hosts(cli.command.as_ref());
    // An unflagged `serve` follows everything; an unflagged interactive scan
    // follows only what an installed artifact can reach. See
    // [`scan::fetch::default_service_follow_policy`].
    let default_follow = if scans_for_other_hosts {
        scan::fetch::default_service_follow_policy
    } else {
        default_cli_follow_policy
    };
    let fetch_policy = cli.global.fetch_policy(
        cli.global.follow.unwrap_or_else(default_follow),
        scan::fetch::DEFAULT_MAX_DEP_AGE_DAYS,
        scans_for_other_hosts,
    );
    // A worker exists to populate the shared corpus, not to answer one question
    // quickly: every resolvable dependency it pulls becomes a hopper sample with a
    // package coordinate, which is what later grows known-good bloom coverage. The
    // fresh-risk window that keeps an interactive scan fast is exactly the wrong
    // default there — it discards the long tail the cache most wants.
    let worker_fetch_policy = cli.global.fetch_policy(
        cli.global
            .follow
            .unwrap_or_else(scan::fetch::default_service_follow_policy),
        scan::fetch::WORKER_MAX_DEP_AGE_DAYS,
        true,
    );

    // Taken, not moved, so `cli` stays whole for `interpret_config` below.
    // Bare paths already became `fs` in `with_default_subcommand`, so a missing
    // subcommand here means an empty command line: print the help and stop.
    let Some(command) = cli.command.take() else {
        Cli::parse_from(["scan", "--help"]);
        std::process::exit(0);
    };

    let is_serve = matches!(command, Commands::Serve { .. } | Commands::Worker { .. });
    // Reclaim stale/oversized caches (stng strings+r2, scan analysis+interpret,
    // fletch blobs). One detached, self-gated sweep for a CLI run; a recurring
    // loop for the never-exiting serve/worker daemons. Non-blocking either way.
    scan::cache_cleanup::start(is_serve);
    // Long-lived server modes never short-circuit on bloom filters: every job is
    // analyzed on its own merits. Force slow mode there regardless of `--mode`.
    let effective_mode = if is_serve {
        scan::Mode::Slow
    } else {
        cli.global.mode
    };
    // Per-execution fetch ceiling: a hard cap across the whole invocation. The
    // long-lived server modes (`serve`/`worker`) scan unboundedly many jobs over
    // their lifetime, so they're exempt — each job is bounded by `--fetch-max-file-*`
    // instead. A one-shot scan gets the full `--fetch-max-total-*` (the budget
    // defaults to unlimited, so leaving it unset in server mode lifts it).
    if !is_serve {
        scan::fetch::set_total_budget(
            cli.global.fetch_max_total_fetches,
            cli.global.fetch_max_total_size,
        );
    }
    // RUST_LOG (when set) wins over the mode-derived defaults, so profiling
    // runs can surface targeted modules (e.g. `cleave::mem_profile=info`)
    // without paying for full `--verbose` debug output.
    // `atomscan` is this binary's own target: startup diagnostics logged from
    // main.rs (rule refresh, authentication, LLM configuration) are not in the
    // `scan` library and were being filtered out of the daemons' logs.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        if cli.global.verbose {
            tracing_subscriber::EnvFilter::new("atomscan=debug,scan=debug,cleave=debug")
        } else if is_serve {
            tracing_subscriber::EnvFilter::new("atomscan=info,scan=info,cleave=warn")
        } else {
            tracing_subscriber::EnvFilter::new("atomscan=warn,scan=warn,cleave=error")
        }
    });
    // Cleave rejects a stale precompiled YARA cache safely and recompiles from
    // source. That expected maintenance path is not an operator-facing error;
    // filter only its exact event. Explicit diagnostics still show it.
    let quiet_expected_yara_cache = ExpectedYaraCacheFilter {
        hide: !cli.global.verbose && std::env::var_os("RUST_LOG").is_none(),
    };
    if is_serve {
        tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_thread_names(true)
                    .with_writer(std::io::stderr)
                    .with_filter(quiet_expected_yara_cache),
            )
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .without_time()
                    .with_thread_names(true)
                    .with_writer(std::io::stderr)
                    .with_filter(quiet_expected_yara_cache),
            )
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
    // forged import directory drives that loop into the millions: on the worker
    // wedged 2026-09-04, 40% of the only running Rayon thread's samples were in
    // the bridge rather than in the parse. `set_max_level(Off)` makes those call
    // sites no-ops at the macro, which is what the dependency comment intends.
    //
    // An operator who asked for logs still gets them: `--verbose` or an explicit
    // `RUST_LOG` leaves the bridge at its default level, so a dependency's `log`
    // output remains reachable when it is actually wanted.
    if !cli.global.verbose && std::env::var_os("RUST_LOG").is_none() {
        log::set_max_level(log::LevelFilter::Off);
    }

    // Resolved after logging is up: with no hardcoded default, the model comes
    // from the endpoint's own listing, so which one was picked has to be
    // visible rather than swallowed by an uninitialized subscriber.
    let interpret_cfg = cli.global.interpret_config()?;
    if let Some(cfg) = &interpret_cfg {
        tracing::info!(
            endpoint = %cfg.base_url,
            model = %cfg.model,
            min_level = cfg
                .min_level
                .map_or_else(
                    || scan::interpret::DEFAULT_MIN_LEVEL_LABEL.to_string(),
                    |n| n.to_string()
                ),
            "LLM interpretation enabled",
        );
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

    warn_on_broken_freebsd_malloc_conf();
    scan::heap_profile::report_debug_allocator();

    const RAYON_FALLBACK_THREADS: usize = 4;
    // Pool width depends on the host's shape. On a wide host, physical cores
    // win: logical SMT siblings (32 on a 16-core host) oversubscribe
    // archive-member analysis, and S2 dropped 56.6 s → 51.8 s at 16 threads.
    // On a small SMT host the opposite holds: the pool is the only
    // parallelism there is, and it idles whenever the one admitted archive is
    // in a serial phase (tar/gzip read). Measured 2026-08-30 on a 4c/8t AMD
    // Ryzen 3400G over the 121-job poppy worker benchmark: 4 threads 34:13,
    // 5 → 32:47, 6 → 31:15, 8 → 27:43 (−19%), peak RSS flat (4.3–4.6 GB),
    // outputs identical. The wider pool is taken only where that class of
    // sibling exists: x86 SMT from AMD or Intel, below SMALL_HOST_CORES
    // physical cores. Apple silicon never qualifies — its extra "logical"
    // capacity is efficiency cores, certified not to help this workload —
    // and any other vendor or a host with no SMT keeps physical cores.
    const SMALL_HOST_CORES: usize = 8;
    let physical = cleave::memory_tracker::physical_cpu_count();
    let logical = cleave::memory_tracker::cpu_count();
    let detected_cores = match (physical, logical) {
        (Some(p), Some(l)) if p < SMALL_HOST_CORES && l > p && smt_pool_vendor() => Some(l),
        (Some(p), _) => Some(p),
        (None, l) => l,
    };
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
    const RAYON_STACK_MB: usize = scan::RAYON_STACK_MB;
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_threads)
        .stack_size(RAYON_STACK_MB * 1024 * 1024)
        .thread_name(|i| format!("rayon-{i}"))
        // Register each pool worker for the SIGUSR1 in-process thread dump, so a
        // wedge can be backtraced without a debugger (lldb/gdb can't attach in
        // the production jails). Each worker also steps its own scheduling
        // priority one notch below normal (see `lower_pool_thread_priority`).
        .start_handler(|_| {
            scan::thread_dump::register_self();
            lower_pool_thread_priority();
        })
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
    let needs_terminal_theme = cli.global.format == scan::OutputFormat::Terminal
        && matches!(
            command,
            Commands::Path { .. }
                | Commands::Ps
                | Commands::Sys
                | Commands::Url { .. }
                | Commands::Purl { .. }
        );
    if cli.global.light {
        scan::output::set_theme(scan::output::Theme::Light);
    } else if cli.global.dark {
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
    scan::heap_profile::install();

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
                scan::heap_profile::dump_on_signal();
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
            cli.global.update,
            cli.global.no_update,
            effective_mode,
            cli.global.format == scan::OutputFormat::Terminal,
        );
    }

    // Resolve model directory lazily — update-rules and version don't need it,
    // and eagerly resolving triggers auto-clone before those commands can run.
    let resolve_model_dir = || -> Result<PathBuf> {
        match &cli.global.model_dir {
            Some(d) => Ok(d.clone()),
            None => scan::models_repo::model_dir().context("failed to resolve model directory"),
        }
    };
    let threshold_overrides = || cli.global.thresholds();
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
    let filter = cli.global.display_filter();
    let new_scan_config = |hopper| -> Result<scan::ScanConfig> {
        let model_dir = resolve_model_dir()?;
        let envelope_level = resolve_envelope_level(&model_dir);
        Ok(scan::ScanConfig::new(
            model_dir,
            cli.global.format,
            threshold_overrides(),
            filter,
            scan::cli::DEFAULT_SLOW_RULE_MS,
            cli.global.extra,
        )?
        .with_level(envelope_level)
        .with_interpret(interpret_cfg.clone())
        .with_fetch(fetch_policy)
        .with_zip_passwords(cli.global.zip_passwords.clone())
        .with_hopper(resolve_hopper(hopper)))
    };

    match command {
        Commands::Path {
            paths,
            hopper,
            registry_map,
        } => {
            let mut config = new_scan_config(hopper)?;
            // Per-file SHA-256 known-good/known-bad short-circuit, for the files
            // found by walking a directory the operator named. A path named
            // directly on the command line is always analyzed — see
            // `engine::named_target_opts`. Slow mode / workers skip it entirely.
            if effective_mode != scan::Mode::Slow {
                config = config.with_bloom(effective_mode, scan::bloom_repo::Lookup::load());
            }
            // `--registry-map <file>`: a JSON object {sha256: provenance}.
            // Preserve each complete value and derive the normalized record
            // beside it, so map-backed scans have the same provider data as
            // live and worker scans. Entries without a record are skipped —
            // registry provenance is best-effort, never required.
            let registry_map = match &registry_map {
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
            exit_for_summary(&run_scan_paths(&paths, &config, registry_map.as_ref())?);
        }
        Commands::Sys => {
            let config = new_scan_config(None)?;
            exit_for_summary(&scan::sys::run(&config)?);
        }
        Commands::Ps => {
            let mut config = new_scan_config(None)?;
            // Per-binary known-good/known-bad short-circuit (by executable sha256).
            if effective_mode != scan::Mode::Slow {
                config = config.with_bloom(effective_mode, scan::bloom_repo::Lookup::load());
            }
            exit_for_summary(&scan::ps::run(&config)?);
        }
        // `url` and `purl` name one artifact, so neither gets `with_bloom`: the
        // thing the operator asked about is always fetched and scanned, never
        // answered from a bless. The filters are still published process-wide,
        // which is what `fetch::age_gate` reads to skip the *dependencies* the
        // scan discovers — the bulk case a bloom is actually for. `--mode slow`
        // opts out of consulting them at all.
        Commands::Url { url, hopper } => {
            let config = new_scan_config(hopper)?;
            publish_bloom_filters(cli.global.mode);
            exit_for_summary(&scan::pkg::run_url(&url, &config)?);
        }
        Commands::Purl { purl, hopper } => {
            let config = new_scan_config(hopper)?;
            publish_bloom_filters(cli.global.mode);
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
            token_file,
            traits_dir,
            hopper,
            idle_worker_slots,
            analysis_timeout,
        } => {
            let hopper = resolve_hopper(hopper);
            // Same contract as the worker: refresh on restart. This is also the
            // step that populates a `--traits-dir` pointing at an empty state
            // directory on a fresh deploy — without it the server starts, reports
            // healthy, and fails every analysis on a traits path that never
            // got created.
            log_max_rss_resolution(
                "server",
                MaxRssPolicy::from_cli(max_rss_gb),
                resolve_process_max_rss_bytes(max_rss_gb),
            );
            let config = scan::server::Startup {
                bind,
                max_size_mb,
                max_rss_gb,
                allowed_dirs,
                extract_dir: extract_dir.map(std::path::PathBuf::from),
                workers,
                allow_cidr,
                token_file,
                traits_dir,
                update: cli.global.update,
                no_update: cli.global.no_update,
                hopper: hopper.clone(),
                idle_worker_slots,
                analysis_timeout_secs: analysis_timeout,
                model_dir: cli.global.model_dir.clone(),
                level: selected_severity_level,
                thresholds: cli.global.thresholds(),
                slow_rule_ms: scan::cli::DEFAULT_SLOW_RULE_MS,
                interpret: interpret_cfg.clone(),
                fetch: fetch_policy,
                zip_passwords: cli.global.zip_passwords.clone().into(),
            }
            .resolve()?;
            if let Some(url) = hopper.as_deref() {
                eprintln!("Renewing results on hopper at {url}");
            }
            if let Some(url) = hopper.as_deref() {
                eprintln!("Deferring unknown lookups to the corpus at {url}");
            }
            // Serve never bloom-skips an /analyze job (Mode::Slow), but the
            // membership endpoint and the --fetch dependency gate read the
            // process-wide handle. Missing files fail closed (no skip).
            publish_bloom_filters(cli.global.mode);
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
                scan::cli::DEFAULT_SLOW_RULE_MS,
                cli.global.extra,
            )?
            .with_level(envelope_level)
            .with_zip_passwords(cli.global.zip_passwords.clone());
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
            no_validate,
            exit_if_empty,
        } => {
            // Claimed jobs are always analyzed in full (worker mode forces
            // `Mode::Slow`, and `ScanConfig` never gets `with_bloom`), but the
            // *dependency* gate in `fetch::age_gate` consults the process-wide
            // bloom handle: a fetched dep whose resolved coordinate is vouched
            // known-good (and not vetoed by the known-bad channel) skips its
            // artifact fetch+scan, keeping only the registry-metadata node.
            // Publishing the filters here enables that skip without touching
            // the job-level always-scan guarantee.
            publish_bloom_filters(cli.global.mode);
            // Accept the comma list `serve --hopper` takes, so one deploy
            // variable can feed both, but keep only the primary: a replica
            // refuses worker routes with a 403 whether or not its relay is on,
            // so the later addresses are not a fallback for this loop. Passing
            // the whole string through reaches a URL parser, which reads the
            // commas as part of one very strange hostname.
            let Some(url) = scan::upload::worker_endpoint(&url) else {
                anyhow::bail!("--url names no hopper address");
            };
            let raw_max_rss_gb = max_rss_gb;
            // Resolved twice over: once here for the startup log, and again
            // inside `Startup::resolve`, which is where the value that reaches
            // the worker is settled. Both call the same function.
            let resolved_max_rss_gb = resolve_worker_max_rss_gb(raw_max_rss_gb);
            let workers = workers.unwrap_or_else(default_workers);
            let name = name.unwrap_or_else(default_worker_name);
            log_worker_startup_diagnostics(&WorkerStartupDiagnostics {
                argv: redact_zip_passwords(std::env::args()),
                hopper_url: &url,
                name: &name,
                workers: workers.get(),
                poll_secs,
                raw_max_rss_gb,
                resolved_max_rss_gb,
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
            log_max_rss_resolution(
                "worker",
                MaxRssPolicy::from_cli(raw_max_rss_gb),
                resolved_max_rss_gb.saturating_mul(GIB),
            );
            // Every default this worker runs on is settled in one place, so a
            // second binary starting one cannot drift from this one.
            let config = scan::worker::Startup {
                // Standalone: owns its signals, its nice value, and its models.
                hopper_url: url,
                name: Some(name),
                workers: Some(workers),
                poll_secs,
                max_rss_gb: raw_max_rss_gb,
                nice,
                data_dir,
                max_jobs,
                exit_if_empty,
                traits_dir,
                update: cli.global.update,
                model_dir: cli.global.model_dir.clone(),
                level: selected_severity_level,
                thresholds: cli.global.thresholds(),
                no_update: cli.global.no_update,
                no_validate,
                slow_rule_ms: scan::cli::DEFAULT_SLOW_RULE_MS,
                interpret: interpret_cfg.clone(),
                fetch: worker_fetch_policy,
                zip_passwords: cli.global.zip_passwords.clone().into(),
            }
            .resolve()
            .inspect_err(|e| eprintln!("Worker startup failed: {e:#}"))?;
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(scan::worker::run(config))?;
        }
        Commands::Version => {
            let bloom = scan::bloom_repo::installed_manifest();
            if cli.global.format == scan::OutputFormat::Json {
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

                let rules =
                    info.trait_count as u64 + info.composite_count as u64 + info.yara_rules as u64;

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

                scan::output::print_version(env!("CARGO_PKG_VERSION"), bloom_count, rules, &rows);
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

/// Run the calling rayon worker one scheduling notch below normal.
///
/// The pool saturates every core, and the analysis it drives also spawns
/// external CPU-bound helpers — rizin's `aaa` on a stripped ELF runs 10–30 s
/// single-threaded — whose caller sits parked in the pool until they finish.
/// Those helpers are on the critical path of the archive that owns them, yet
/// they share cores as equals with sixteen workers that could just as well
/// run later. Measured on the npm corpus: a 3 MB arm64 `.so` whose rizin pass
/// takes 26 s alone stretched to ~50 s inside a saturated scan. Demoting the
/// workers lets the scheduler hand such helpers a whole core the moment they
/// become runnable; total pool throughput is unchanged because the demoted
/// threads still own every idle cycle. Windows and Linux support a per-thread
/// priority; elsewhere this is a no-op (FreeBSD's `setpriority` acts on the
/// whole process, which the worker already nices).
fn lower_pool_thread_priority() {
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut core::ffi::c_void;
            fn SetThreadPriority(thread: *mut core::ffi::c_void, priority: i32) -> i32;
        }
        const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
        // SAFETY: both calls take the pseudo-handle of the calling thread and
        // touch no memory we own.
        unsafe {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        }
    }
    #[cfg(target_os = "linux")]
    {
        // SAFETY: setpriority on the calling thread's tid (PRIO_PROCESS with a
        // tid is Linux's per-thread nice); no memory effects. Raising the nice
        // value never needs privilege, and a failure just leaves the thread at
        // its inherited priority.
        unsafe {
            // A tid always fits in id_t; the fallible conversion only keeps
            // the cast lint honest, and on failure the thread stays as it was.
            let Ok(tid) = libc::id_t::try_from(libc::syscall(libc::SYS_gettid)) else {
                return;
            };
            let current = libc::getpriority(libc::PRIO_PROCESS, tid);
            libc::setpriority(libc::PRIO_PROCESS, tid, (current + 1).min(19));
        }
    }
}

/// Default worker count: at least 2, and the physical core count from
/// `sysmem` (SMT siblings don't add analysis throughput; on Apple silicon
/// this is the performance-core count via `hw.perflevel0.physicalcpu`).
/// Platforms where `sysmem` has no physical-core signal fall back to half
/// the logical CPUs — identical to the physical count on 2-way SMT.
/// Whether this host's SMT siblings are the kind the small-host pool policy
/// was measured on: x86 hyperthreads from AMD (measured) or Intel (same
/// design). Apple silicon and unknown vendors return false.
fn smt_pool_vendor() -> bool {
    if cfg!(target_vendor = "apple") || !cfg!(target_arch = "x86_64") {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .is_ok_and(|info| info.contains("AuthenticAMD") || info.contains("GenuineIntel"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        // x86_64 FreeBSD/Windows/illumos hosts: AMD/Intel is the only x86 SMT
        // there is; the arch check above already excludes everything else.
        true
    }
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

fn redact_zip_passwords(args: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut args = args.into_iter();
    let mut redacted = Vec::new();
    while let Some(arg) = args.next() {
        if arg == "--zip-password" {
            redacted.push(arg);
            if args.next().is_some() {
                redacted.push("<redacted>".to_string());
            }
        } else if arg.starts_with("--zip-password=") {
            redacted.push("--zip-password=<redacted>".to_string());
        } else {
            redacted.push(arg);
        }
    }
    redacted
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

/// Exit with the appropriate code based on scan summary counters.
///
/// A degraded YARA engine (panic breaker tripped, or rule sources that failed
/// to compile at runtime) outranks everything: the verdicts above were made
/// with fewer rules than the trait set defines, so the run must not look like
/// a clean scan.
fn exit_for_summary(summary: &scan::ScanSummary) {
    let degradation = cleave::yara_engine::yara_degradation();
    if let Some(msg) = &degradation {
        eprintln!("\n❌ YARA engine degraded during this run: {msg}");
        eprintln!(
            "   Verdicts above were produced WITHOUT the full rule set; treat them as incomplete."
        );
    }
    // A hostile verdict stands even degraded (rules only add detections), but a
    // run that would otherwise look clean or merely suspicious must fail loudly.
    if summary.hostile > 0 {
        process::exit(1);
    }
    if degradation.is_some() {
        process::exit(4);
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
    registry_map: Option<&std::collections::HashMap<String, scan::provenance::RegistryProvenance>>,
) -> Result<scan::ScanSummary> {
    // Warm YARA + capability mapper off the rayon pool before any analysis
    // spawns rayon work. Directory scans run on a dedicated rayon pool; if
    // any of those workers is the first to hit `yara_engine()`, the init's
    // internal par_iter deadlocks against its peers parked on the OnceLock.
    // Prefetching from main (non-rayon) fills the OnceLock safely.
    //
    // The regex prewarm is for first-file latency on a long-lived process;
    // a directory scan amortizes lazy compilation over thousands of members
    // and only builds the patterns its file types reach. Skipping it there
    // took a 12.5k-member module zip from 4.6 GB to 3.1 GB peak at equal
    // wall. Explicit file lists keep it: a single small file otherwise spends
    // most of its wall in first-use compiles.
    // `paths` may already be the expanded file list of a directory the
    // operator named, so a directory is not the only bulk signal.
    let bulk = paths.len() >= 8 || paths.iter().any(|p| p.is_dir());
    cleave::prefetch_shared_resources_with(true, !bulk);

    // Explicit files are analyzed as one parallel batch and each directory is
    // streamed; run_paths shares one model load and verdict tally across all.
    scan::engine::run_paths(paths, config, registry_map)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{
        Cli, Commands, GIB, MaxRssPolicy, command_scans_for_other_hosts, default_cli_follow_policy,
        is_expected_yara_cache_mismatch, redact_zip_passwords, resolve_hopper_value,
        with_default_subcommand,
    };
    use anyhow::{Context, Result};
    use clap::Parser;
    use scan::OutputFormat;
    use scan::cli::DEFAULT_RIZIN_TIMEOUT_SECS;
    use std::ffi::OsString;
    use std::net::SocketAddr;
    use std::num::NonZeroU64;
    use std::path::PathBuf;

    #[test]
    fn only_expected_yara_cache_mismatch_is_quietable() {
        let expected = "compiled YARA rules do not match the rule sources on disk; \
                        ignoring them and recompiling";
        assert!(is_expected_yara_cache_mismatch(
            "cleave::yara_engine",
            expected
        ));
        assert!(!is_expected_yara_cache_mismatch(
            "cleave::yara_engine",
            "YARA bucket scan failed, skipping bucket"
        ));
        assert!(!is_expected_yara_cache_mismatch("scan::engine", expected));
    }

    #[test]
    fn follow_defaults_to_references_and_dependencies() -> Result<()> {
        let cli = Cli::try_parse_from(with_default_subcommand(["scan", "/tmp/a"]))
            .context("parse should work")?;
        assert!(
            cli.global.follow.is_none(),
            "absence of --follow is resolved at startup"
        );

        let policy = default_cli_follow_policy();
        assert!(policy.urls && policy.packages && policy.deps);
        assert!(
            !policy.ci,
            "CI actions never reach an installed artifact; auditing them is opt-in"
        );
        // A bare `--follow` must select exactly what an absent one resolves to.
        let bare = Cli::try_parse_from(with_default_subcommand(["scan", "--follow", "/tmp/a"]))
            .context("bare --follow should parse")?
            .global
            .follow
            .context("bare --follow has a default_missing_value")?;
        assert_eq!(
            (bare.urls, bare.packages, bare.deps, bare.ci),
            (policy.urls, policy.packages, policy.deps, policy.ci),
        );
        assert_eq!(
            policy.max_file_fetches,
            scan::fetch::DEFAULT_MAX_FILE_FETCHES
        );
        assert_eq!(policy.max_file_fetches, 100);
        assert_eq!(policy.max_url_fetches, scan::fetch::DEFAULT_MAX_URL_FETCHES);
        assert_eq!(policy.max_url_fetches, 4);
        Ok(())
    }

    /// A service follows everything it can reach, because the corpus it fills
    /// is asked about artifacts nobody has looked at yet: a category left
    /// unfollowed is one nobody ever learns about.
    #[test]
    fn serve_and_worker_follow_everything_by_default() {
        let service = scan::fetch::default_service_follow_policy();
        assert!(
            service.urls && service.packages && service.deps && service.ci,
            "a cache-population role must follow every category"
        );

        // The interactive default stays narrow: CI actions never reach an
        // installed artifact, so one person's scan should not pay for them.
        let interactive = default_cli_follow_policy();
        assert!(
            !interactive.ci,
            "the interactive default must not widen with the service one"
        );
        assert_ne!(
            (
                interactive.urls,
                interactive.packages,
                interactive.deps,
                interactive.ci
            ),
            (service.urls, service.packages, service.deps, service.ci),
            "the two defaults are deliberately different; if they converge, say so on purpose"
        );
    }

    /// The roles that get the wide default are exactly the ones that scan on
    /// behalf of other hosts. Keeping the two in step is what stops a new
    /// service subcommand from quietly populating the corpus with a narrow
    /// verdict.
    #[test]
    fn the_wide_default_covers_every_service_role() {
        for args in [
            vec!["scan", "serve"],
            vec!["scan", "worker", "--url", "http://hopper.invalid"],
        ] {
            let cli = Cli::try_parse_from(&args).expect("service command should parse");
            assert!(
                command_scans_for_other_hosts(cli.command.as_ref()),
                "{args:?} scans for other hosts and must take the wide follow default"
            );
        }
        let interactive = Cli::try_parse_from(with_default_subcommand(["scan", "/tmp/a"]))
            .expect("path scan should parse");
        assert!(
            !command_scans_for_other_hosts(interactive.command.as_ref()),
            "an interactive path scan keeps the narrow default"
        );
    }

    #[test]
    fn old_fetch_flag_and_target_names_remain_aliases() -> Result<()> {
        let old = Cli::try_parse_from(with_default_subcommand([
            "scan",
            "--fetch=deps,packages,urls,ci",
            "/tmp/a",
        ]))
        .context("legacy --fetch vocabulary should parse")?
        .global
        .follow
        .context("legacy --fetch should select a policy")?;
        let new = Cli::try_parse_from(with_default_subcommand([
            "scan",
            "--follow=dependencies,references,ci-actions",
            "/tmp/a",
        ]))
        .context("canonical --follow vocabulary should parse")?
        .global
        .follow
        .context("canonical --follow should select a policy")?;
        assert_eq!(old, new);
        Ok(())
    }

    #[test]
    fn rizin_timeout_defaults_to_ten_minutes_and_is_overridable() -> Result<()> {
        let default = Cli::try_parse_from(with_default_subcommand(["scan", "/tmp/a"]))
            .context("default timeout should parse")?;
        assert_eq!(
            default.global.rizin_timeout_secs,
            DEFAULT_RIZIN_TIMEOUT_SECS
        );

        let overridden = Cli::try_parse_from(with_default_subcommand([
            "scan",
            "--rizin-timeout-secs",
            "42",
            "/tmp/a",
        ]))
        .context("timeout override should parse")?;
        assert_eq!(overridden.global.rizin_timeout_secs, 42);
        assert!(
            Cli::try_parse_from(with_default_subcommand([
                "scan",
                "--rizin-timeout-secs",
                "0",
                "/tmp/a"
            ]))
            .is_err(),
            "zero would disable the hard deadline and must be rejected"
        );
        Ok(())
    }

    #[test]
    fn dependency_platform_scope_follows_scanner_role() -> Result<()> {
        let default = Cli::try_parse_from(with_default_subcommand(["scan", "/tmp/a"]))
            .context("default should parse")?;
        assert!(
            default
                .global
                .host_platform_only(super::command_scans_for_other_hosts(
                    default.command.as_ref()
                )),
            "interactive scans optimize for the current host"
        );
        assert!(
            scan::fetch::FetchPolicy::default().host_platform_only,
            "library and CLI defaults must agree"
        );
        let compatible = Cli::try_parse_from(with_default_subcommand([
            "scan",
            "--fetch-all-platforms",
            "/tmp/a",
        ]))
        .context("all-platforms opt-in should parse")?;
        assert!(!compatible.global.host_platform_only(false));

        let host_only = Cli::try_parse_from(with_default_subcommand([
            "scan",
            "--fetch-host-platform-only",
            "/tmp/a",
        ]))
        .context("host-only opt-out should parse")?;
        assert!(host_only.global.host_platform_only(true));

        for daemon in [
            Cli::try_parse_from(["scan", "serve"]).context("serve should parse")?,
            Cli::try_parse_from(["scan", "worker", "--url", "http://hopper.test"])
                .context("worker should parse")?,
        ] {
            let role = super::command_scans_for_other_hosts(daemon.command.as_ref());
            assert!(role);
            assert!(
                !daemon.global.host_platform_only(role),
                "serve/worker scan on behalf of other hosts"
            );
        }
        Ok(())
    }

    #[test]
    fn bare_paths_default_to_scan_shorthand() -> Result<()> {
        let cli = Cli::try_parse_from(with_default_subcommand(["scan", "/tmp/a", "/tmp/b"]))
            .context("parse should work")?;
        match cli.command.context("fs subcommand expected")? {
            Commands::Path { paths, .. } => assert_eq!(
                paths,
                vec![PathBuf::from("/tmp/a"), PathBuf::from("/tmp/b")]
            ),
            other => anyhow::bail!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    /// The shorthand is the form operators and agents actually type, and every
    /// flag `fs` accepts has to survive it — `--upload` was rejected outright
    /// before the line was rewritten into a real subcommand.
    #[test]
    fn shorthand_accepts_every_fs_flag() -> Result<()> {
        let cli = Cli::try_parse_from(with_default_subcommand([
            "scan",
            "--upload",
            "http://hopper:8081",
            "--registry-map",
            "/tmp/map.json",
            "--format",
            "json",
            "/tmp/a",
        ]))
        .context("parse should work")?;
        assert_eq!(cli.global.format, OutputFormat::Json);
        match cli.command.context("fs subcommand expected")? {
            Commands::Path {
                paths,
                hopper,
                registry_map,
            } => {
                assert_eq!(paths, vec![PathBuf::from("/tmp/a")]);
                assert_eq!(hopper.as_deref(), Some("http://hopper:8081"));
                assert_eq!(registry_map, Some(PathBuf::from("/tmp/map.json")));
            }
            other => anyhow::bail!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn default_subcommand_declines_lines_that_name_their_own() {
        for line in [
            vec!["scan", "fs", "/tmp/a"],
            vec!["scan", "purl", "pkg:npm/left-pad@1.3.0"],
            vec!["scan", "serve"],
            vec!["scan", "help"],
            vec!["scan", "--help"],
            vec!["scan", "--version"],
            vec!["scan"],
            vec!["scan", "--verbose"],
        ] {
            assert_eq!(
                with_default_subcommand(line.clone()),
                line.iter().map(OsString::from).collect::<Vec<_>>(),
                "{line:?} should reach the top-level parser as typed",
            );
        }
    }

    #[test]
    fn default_subcommand_inserts_before_operands() {
        assert_eq!(
            with_default_subcommand(["scan", "-f", "json", "/tmp/a"]),
            ["scan", "fs", "-f", "json", "/tmp/a"]
                .iter()
                .map(OsString::from)
                .collect::<Vec<_>>(),
        );
        // `--` marks the operand, so a path spelled like a flag still scans.
        assert_eq!(
            with_default_subcommand(["scan", "--", "-weird-name"]),
            ["scan", "fs", "--", "-weird-name"]
                .iter()
                .map(OsString::from)
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn hopper_environment_fills_missing_scan_destination() {
        assert_eq!(
            resolve_hopper_value(None, Some("http://hopper:8081/".to_string())).as_deref(),
            Some("http://hopper:8081/")
        );
        assert_eq!(
            resolve_hopper_value(
                Some("http://flag:8081".to_string()),
                Some("http://env".to_string())
            )
            .as_deref(),
            Some("http://flag:8081")
        );
        assert!(resolve_hopper_value(None, Some("  ".to_string())).is_none());
    }

    #[test]
    fn fs_subcommand_accepts_multiple_paths() -> Result<()> {
        let cli = Cli::try_parse_from(with_default_subcommand(["scan", "fs", "/tmp/a", "/tmp/b"]))
            .context("parse should work")?;
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
            let cli = Cli::try_parse_from(with_default_subcommand([
                "scan",
                "fs",
                flag,
                "http://hopper:8081",
                "/tmp/a",
            ]))
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
    fn url_and_purl_accept_hopper_flag() -> Result<()> {
        let cli = Cli::try_parse_from([
            "scan",
            "url",
            "https://h/f.tgz",
            "--hopper",
            "http://x:8081",
        ])
        .context("parse should work")?;
        match cli.command.context("url subcommand expected")? {
            Commands::Url { url, hopper } => {
                assert_eq!(url, "https://h/f.tgz");
                assert_eq!(hopper.as_deref(), Some("http://x:8081"));
            }
            other => anyhow::bail!("unexpected command: {other:?}"),
        }
        let cli =
            Cli::try_parse_from(["scan", "purl", "npm/left-pad", "--upload", "http://x:8081"])
                .context("parse should work")?;
        match cli.command.context("purl subcommand expected")? {
            Commands::Purl { hopper, .. } => {
                assert_eq!(hopper.as_deref(), Some("http://x:8081"));
            }
            other => anyhow::bail!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn fs_without_hopper_defaults_to_none() -> Result<()> {
        let cli = Cli::try_parse_from(with_default_subcommand(["scan", "fs", "/tmp/a"]))
            .context("parse should work")?;
        match cli.command.context("fs subcommand expected")? {
            Commands::Path { hopper, .. } => assert!(hopper.is_none()),
            other => anyhow::bail!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn path_subcommand_and_its_aliases_resolve() -> Result<()> {
        for name in ["path", "fs", "scan"] {
            let cli = Cli::try_parse_from(with_default_subcommand(["scan", name, "/tmp/a"]))
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
                Commands::Purl { purl, .. } => assert_eq!(purl, "pkg:npm/left-pad@1.3.0"),
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
        let cli = Cli::try_parse_from(with_default_subcommand(["scan", "-l", "100", "/tmp/a"]))
            .context("-l 100 should parse")?;
        assert_eq!(cli.global.level, Some(100));

        let cli = Cli::try_parse_from(with_default_subcommand(["scan", "--level", "12", "/tmp/a"]))
            .context("--level 12 should parse")?;
        assert_eq!(cli.global.level, Some(12));

        // Out-of-range rejected (0..=25000, per-100M since the per-million migration).
        assert!(
            Cli::try_parse_from(with_default_subcommand(["scan", "-l", "25001", "/tmp/a"]))
                .is_err()
        );
        // --level still conflicts with explicit thresholds.
        assert!(
            Cli::try_parse_from(with_default_subcommand([
                "scan",
                "-l",
                "10",
                "--threshold-hostile",
                "0.5",
                "/tmp/a"
            ]))
            .is_err()
        );
        // The numeric shortcuts and their aliases were removed.
        assert!(Cli::try_parse_from(with_default_subcommand(["scan", "-5", "/tmp/a"])).is_err());
        assert!(
            Cli::try_parse_from(with_default_subcommand(["scan", "--loose", "/tmp/a"])).is_err()
        );
        assert!(
            Cli::try_parse_from(with_default_subcommand(["scan", "--paranoid", "/tmp/a"])).is_err()
        );
        Ok(())
    }

    #[test]
    fn gzip_long_aliases_are_not_accepted() {
        assert!(
            Cli::try_parse_from(with_default_subcommand(["scan", "--fast", "/tmp/a"])).is_err()
        );
        assert!(
            Cli::try_parse_from(with_default_subcommand(["scan", "--best", "/tmp/a"])).is_err()
        );
    }

    #[test]
    fn worker_diagnostics_redact_archive_passwords() {
        let args = [
            "atomscan",
            "worker",
            "--zip-password",
            "secret one",
            "--zip-password=secret-two",
            "--verbose",
        ]
        .map(str::to_string);

        assert_eq!(
            redact_zip_passwords(args),
            [
                "atomscan",
                "worker",
                "--zip-password",
                "<redacted>",
                "--zip-password=<redacted>",
                "--verbose",
            ]
        );
    }

    #[test]
    fn archive_password_flag_is_repeatable_and_global() -> Result<()> {
        let cli = Cli::try_parse_from(with_default_subcommand([
            "atomscan",
            "path",
            "/tmp/a",
            "--zip-password",
            "one",
            "--zip-password=two",
        ]))?;

        assert_eq!(cli.global.zip_passwords, ["one", "two"]);
        Ok(())
    }

    #[test]
    fn concurrency_flags_reject_zero() {
        assert!(Cli::try_parse_from(["atomscan", "serve", "--workers", "0"]).is_err());
        assert!(
            Cli::try_parse_from([
                "atomscan",
                "worker",
                "--url",
                "http://hopper",
                "--workers",
                "0",
            ])
            .is_err()
        );
    }

    #[test]
    fn severity_level_flags_conflict_with_each_other_and_manual_thresholds() {
        assert!(
            Cli::try_parse_from(with_default_subcommand(["scan", "-1", "-9", "/tmp/a"])).is_err()
        );
        assert!(
            Cli::try_parse_from(with_default_subcommand([
                "scan",
                "-9",
                "--threshold-hostile",
                "0.90",
                "/tmp/a"
            ]))
            .is_err()
        );
    }

    fn with_isolated_llm_env<T>(home: Option<&std::path::Path>, f: impl FnOnce() -> T) -> T {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let saved = [
            ("HOME", std::env::var_os("HOME")),
            ("SCAN_LLM", std::env::var_os("SCAN_LLM")),
            ("SCAN_LLM_MODEL", std::env::var_os("SCAN_LLM_MODEL")),
            ("SCAN_LLM_KEY", std::env::var_os("SCAN_LLM_KEY")),
        ];
        unsafe {
            std::env::remove_var("SCAN_LLM");
            std::env::remove_var("SCAN_LLM_MODEL");
            std::env::remove_var("SCAN_LLM_KEY");
            if let Some(h) = home {
                std::env::set_var("HOME", h);
            }
        }
        let out = f();
        unsafe {
            for (k, v) in saved {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
        out
    }

    #[test]
    fn openrouter_alias_requires_key_and_defaults_model() -> Result<()> {
        let empty_home = tempfile::tempdir()?;
        with_isolated_llm_env(Some(empty_home.path()), || {
            let missing_key = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "openrouter",
                "--llm-model",
                "qwen/qwen3.8-27b",
                "/tmp/a",
            ]))?;
            let err = missing_key
                .global
                .interpret_config()
                .expect_err("openrouter without a key must fail");
            assert!(
                err.to_string().contains("~/.tok/openrouter"),
                "unexpected error: {err}"
            );

            let missing_model = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "openrouter",
                "--llm-key",
                "sk-test",
                "/tmp/a",
            ]))?;
            let cfg = missing_model
                .global
                .interpret_config()?
                .context("openrouter without a pinned model should default to auto")?;
            assert_eq!(cfg.model, scan::interpret::OPENROUTER_DEFAULT_MODEL);

            let cli = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "openrouter",
                "--llm-model",
                "qwen/qwen3.8-27b",
                "--llm-key",
                "sk-test",
                "/tmp/a",
            ]))?;
            let cfg = cli
                .global
                .interpret_config()?
                .context("openrouter with model+key should enable interpret")?;
            assert_eq!(cfg.base_url, scan::interpret::OPENROUTER_BASE_URL);
            assert_eq!(cfg.model, "qwen/qwen3.8-27b");
            assert_eq!(cfg.api_key.as_deref(), Some("sk-test"));
            Ok(())
        })
    }

    #[test]
    fn llm_flags_are_global_on_serve_and_worker() -> Result<()> {
        with_isolated_llm_env(None, || {
            let serve = Cli::try_parse_from([
                "atomscan",
                "--llm",
                "openrouter",
                "--llm-model",
                "qwen/qwen3.8-27b",
                "--llm-key",
                "sk-test",
                "serve",
            ])?;
            assert_eq!(serve.global.llm.as_deref(), Some("openrouter"));
            let cfg = serve
                .global
                .interpret_config()?
                .context("serve inherits --llm")?;
            assert_eq!(cfg.base_url, scan::interpret::OPENROUTER_BASE_URL);

            let worker = Cli::try_parse_from([
                "atomscan",
                "worker",
                "--url",
                "http://hopper.test",
                "--llm",
                "openrouter",
                "--llm-model",
                "qwen/qwen3.8-27b",
                "--llm-key",
                "sk-test",
            ])?;
            assert_eq!(worker.global.llm.as_deref(), Some("openrouter"));
            Ok(())
        })
    }

    /// `~/.tok/llm` authenticates *our* endpoint. Handing it to OpenRouter
    /// would send a live credential to a third party, and arrive there as an
    /// ordinary 401 that says nothing about what leaked.
    #[test]
    fn the_vllm_token_is_never_sent_to_openrouter() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let tok = dir.path().join(".tok");
        std::fs::create_dir(&tok)?;
        std::fs::write(tok.join("llm"), "sk-vllm\n")?;
        with_isolated_llm_env(Some(dir.path()), || {
            // Alone, OpenRouter without its own key is the same hard error it
            // has always been — not a silent borrow of the vLLM token.
            let cli = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "openrouter",
                "--llm-model",
                "qwen/qwen3.8-27b",
                "/tmp/a",
            ]))?;
            let err = cli
                .global
                .interpret_config()
                .expect_err("~/.tok/llm must not satisfy OpenRouter");
            assert!(err.to_string().contains("~/.tok/openrouter"), "{err}");
            Ok(())
        })
    }

    #[test]
    fn llm_failover_chain_resolves_each_endpoint_separately() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let tok = dir.path().join(".tok");
        std::fs::create_dir(&tok)?;
        std::fs::write(tok.join("llm"), "sk-vllm\n")?;
        std::fs::write(tok.join("openrouter"), "sk-openrouter\n")?;
        with_isolated_llm_env(Some(dir.path()), || {
            // The shape we deploy by default. Both models pinned so the test
            // resolves without touching the network.
            let cli = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "https://llm.isotope13.ai/v1,openrouter",
                "--llm-model",
                "Qwen/Qwen3.8-27B,qwen/qwen3.8-27b",
                "/tmp/a",
            ]))?;
            let cfg = cli
                .global
                .interpret_config()?
                .context("chain should resolve")?;
            assert_eq!(cfg.base_url, "https://llm.isotope13.ai/v1");
            assert_eq!(cfg.model, "Qwen/Qwen3.8-27B");
            // Each endpoint takes its own token file: the vLLM key must not be
            // sent to OpenRouter, nor the reverse.
            assert_eq!(cfg.api_key.as_deref(), Some("sk-vllm"));
            assert_eq!(cfg.fallbacks.len(), 1);
            assert_eq!(
                cfg.fallbacks[0].base_url,
                scan::interpret::OPENROUTER_BASE_URL
            );
            assert_eq!(cfg.fallbacks[0].model, "qwen/qwen3.8-27b");
            assert_eq!(cfg.fallbacks[0].api_key.as_deref(), Some("sk-openrouter"));
            Ok(())
        })
    }

    /// A fallback that cannot be used is dropped, not fatal: the primary is
    /// what the scan actually needs.
    #[test]
    fn an_unusable_fallback_is_dropped_but_the_primary_stands() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let tok = dir.path().join(".tok");
        std::fs::create_dir(&tok)?;
        std::fs::write(tok.join("llm"), "sk-vllm\n")?;
        with_isolated_llm_env(Some(dir.path()), || {
            // No ~/.tok/openrouter, so the OpenRouter slot cannot be called
            // (its model would default to `openrouter/auto`, but the missing
            // key still drops it); the same config with OpenRouter *alone* is
            // a hard error (see openrouter_alias_requires_key_and_defaults_model).
            let cli = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "https://llm.isotope13.ai/v1,openrouter",
                "--llm-model",
                "Qwen/Qwen3.8-27B",
                "/tmp/a",
            ]))?;
            let cfg = cli
                .global
                .interpret_config()?
                .context("primary should stand")?;
            assert_eq!(cfg.base_url, "https://llm.isotope13.ai/v1");
            assert!(
                cfg.fallbacks.is_empty(),
                "an OpenRouter slot with no key must not be kept: {:?}",
                cfg.fallbacks
            );
            Ok(())
        })
    }

    /// ...but losing every endpoint is fatal, and says why for each.
    #[test]
    fn a_chain_with_no_usable_endpoint_is_an_error() -> Result<()> {
        let empty_home = tempfile::tempdir()?;
        with_isolated_llm_env(Some(empty_home.path()), || {
            // Port 1 refuses immediately, so discovery fails without a wait;
            // the OpenRouter slot has neither key nor model.
            let cli = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "http://127.0.0.1:1/v1,openrouter",
                "/tmp/a",
            ]))?;
            let err = cli
                .global
                .interpret_config()
                .expect_err("no endpoint is usable here");
            let text = err.to_string();
            assert!(text.contains("no usable LLM endpoint"), "{text}");
            assert!(text.contains("127.0.0.1:1"), "{text}");
            assert!(text.contains("OpenRouter"), "{text}");
            Ok(())
        })
    }

    #[test]
    fn llm_key_falls_back_to_tok_llm_file() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let tok = dir.path().join(".tok");
        std::fs::create_dir(&tok)?;
        std::fs::write(tok.join("llm"), "sk-vllm-file\n")?;
        with_isolated_llm_env(Some(dir.path()), || {
            // A pinned model keeps this off the network: an unpinned one would
            // probe the endpoint for its catalog.
            let cli = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "http://interpret:8000/v1",
                "--llm-model",
                "Qwen/Qwen3.8-27B",
                "/tmp/a",
            ]))?;
            let cfg = cli
                .global
                .interpret_config()?
                .context("~/.tok/llm should supply the key")?;
            assert_eq!(cfg.api_key.as_deref(), Some("sk-vllm-file"));

            // An explicit key still wins over the file.
            let cli = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "http://interpret:8000/v1",
                "--llm-model",
                "Qwen/Qwen3.8-27B",
                "--llm-key",
                "sk-flag",
                "/tmp/a",
            ]))?;
            let cfg = cli.global.interpret_config()?.context("explicit key")?;
            assert_eq!(cfg.api_key.as_deref(), Some("sk-flag"));
            Ok(())
        })
    }

    #[test]
    fn openrouter_prefers_its_own_tok_file_over_tok_llm() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let tok = dir.path().join(".tok");
        std::fs::create_dir(&tok)?;
        std::fs::write(tok.join("llm"), "sk-vllm-file\n")?;
        std::fs::write(tok.join("openrouter"), "sk-openrouter-file\n")?;
        with_isolated_llm_env(Some(dir.path()), || {
            let cli = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "openrouter",
                "--llm-model",
                "qwen/qwen3.8-27b",
                "/tmp/a",
            ]))?;
            let cfg = cli.global.interpret_config()?.context("openrouter key")?;
            assert_eq!(cfg.api_key.as_deref(), Some("sk-openrouter-file"));
            Ok(())
        })
    }

    #[test]
    fn no_tok_llm_file_leaves_the_endpoint_unauthenticated() -> Result<()> {
        let empty_home = tempfile::tempdir()?;
        with_isolated_llm_env(Some(empty_home.path()), || {
            let cli = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "http://interpret:8000/v1",
                "--llm-model",
                "Qwen/Qwen3.8-27B",
                "/tmp/a",
            ]))?;
            let cfg = cli.global.interpret_config()?.context("local target")?;
            assert_eq!(cfg.api_key, None);
            Ok(())
        })
    }

    #[test]
    fn openrouter_key_falls_back_to_tok_file() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let tok = dir.path().join(".tok");
        std::fs::create_dir(&tok)?;
        std::fs::write(tok.join("openrouter"), "sk-from-file\n")?;
        with_isolated_llm_env(Some(dir.path()), || {
            let cli = Cli::try_parse_from(with_default_subcommand([
                "atomscan",
                "--llm",
                "openrouter",
                "--llm-model",
                "qwen/qwen3.8-27b",
                "/tmp/a",
            ]))?;
            let cfg = cli
                .global
                .interpret_config()?
                .context("tok file should supply key")?;
            assert_eq!(cfg.api_key.as_deref(), Some("sk-from-file"));
            Ok(())
        })
    }
}
