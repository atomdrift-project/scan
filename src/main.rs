//! Atomdrift Scan (`atomscan`) — ML-powered malware classification CLI.

// Doc comments on `clap` structs are user-facing `--help` text, so they carry
// `[EXPERIMENTAL]` tags, bare URLs, and `<URL>` placeholders that rustdoc would
// otherwise read as broken links or markup.
#![allow(
    rustdoc::broken_intra_doc_links,
    rustdoc::bare_urls,
    rustdoc::invalid_html_tags,
    reason = "these doc comments are `--help` text, not rustdoc"
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
        target_arch = "arm",
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

    #[unsafe(no_mangle)]
    static _rjem_malloc_conf: SyncPtr = SyncPtr(cleave::JEMALLOC_CONF.as_ptr());

    /// Route tree-sitter's C-core allocations through jemalloc. Its parse
    /// trees otherwise go to the system malloc — outside every jemalloc
    /// budget, decay policy, and heap profile this codebase relies on, and
    /// on macOS the default small-object zone retains freed pages (on the
    /// order of a gigabyte of empty MALLOC_SMALL at a large batch's peak).
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
        // family and satisfy libc malloc alignment; the shims below restore
        // tree-sitter's abort-on-null contract. The ordering and
        // thread-exclusivity clauses are this function's own documented
        // precondition.
        unsafe {
            tree_sitter::set_allocator(Some(tree_sitter::Allocator {
                malloc: ts_malloc,
                calloc: ts_calloc,
                realloc: ts_realloc,
                free: tikv_jemalloc_sys::free,
            }));
        }
    }

    // tree-sitter's own allocators abort on null; a custom one is trusted, and
    // its array growth (`array.h`) writes through a `realloc` result unchecked.
    // jemalloc does return null on exhaustion (ENOMEM is real under
    // `vm.max_map_count`), so passing it straight through would turn memory
    // pressure while parsing a hostile source file into a wild write. Abort
    // instead, exactly as tree-sitter's defaults do.

    /// # Safety
    ///
    /// The C `malloc` contract.
    unsafe extern "C" fn ts_malloc(size: usize) -> *mut std::ffi::c_void {
        // SAFETY: jemalloc's malloc accepts any size.
        let ptr = unsafe { tikv_jemalloc_sys::malloc(size) };
        if ptr.is_null() && size != 0 {
            std::process::abort();
        }
        ptr
    }

    /// # Safety
    ///
    /// The C `calloc` contract.
    unsafe extern "C" fn ts_calloc(count: usize, size: usize) -> *mut std::ffi::c_void {
        // SAFETY: jemalloc's calloc accepts any count and size, and returns
        // null on overflow.
        let ptr = unsafe { tikv_jemalloc_sys::calloc(count, size) };
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
        let grown = unsafe { tikv_jemalloc_sys::realloc(ptr, size) };
        if grown.is_null() && size != 0 {
            std::process::abort();
        }
        grown
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

    /// `calloc` for tree-sitter: a zeroed `count * size` allocation from
    /// mimalloc. Aborts on overflow or exhaustion, as tree-sitter's own
    /// allocators do: a custom allocator is trusted, and its array growth
    /// writes through the result unchecked.
    ///
    /// # Safety
    ///
    /// The C allocator contract: the result is freed only through `mi_free`.
    unsafe extern "C" fn calloc_compat(count: usize, size: usize) -> *mut std::ffi::c_void {
        let Some(bytes) = count.checked_mul(size) else {
            std::process::abort();
        };
        // SAFETY: `mi_malloc` accepts any size and returns null on failure.
        let ptr = unsafe { libmimalloc_sys::mi_malloc(bytes) };
        if ptr.is_null() && bytes > 0 {
            std::process::abort();
        }
        if bytes > 0 {
            // SAFETY: `ptr` is a fresh, non-null allocation of `bytes` bytes.
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes) };
        }
        ptr
    }

    /// # Safety
    ///
    /// Same contract as [`super::jemalloc::route_tree_sitter_through_jemalloc`]:
    /// call once, first thing in `main`, before any tree-sitter API.
    pub(super) unsafe fn route_tree_sitter_through_mimalloc() {
        // SAFETY: mimalloc's malloc/calloc/realloc/free are one allocator
        // family; the ordering and thread-exclusivity clauses are this
        // function's own documented precondition.
        unsafe {
            tree_sitter::set_allocator(Some(tree_sitter::Allocator {
                malloc: malloc_compat,
                calloc: calloc_compat,
                realloc: realloc_compat,
                free: libmimalloc_sys::mi_free,
            }));
        }
    }

    /// `malloc` for tree-sitter, aborting on exhaustion (see [`calloc_compat`]).
    ///
    /// # Safety
    ///
    /// The C allocator contract: the result is freed only through `mi_free`.
    unsafe extern "C" fn malloc_compat(size: usize) -> *mut std::ffi::c_void {
        // SAFETY: `mi_malloc` accepts any size and returns null on failure.
        let ptr = unsafe { libmimalloc_sys::mi_malloc(size) };
        if ptr.is_null() && size != 0 {
            std::process::abort();
        }
        ptr
    }

    /// `realloc` for tree-sitter, aborting on exhaustion (see [`calloc_compat`]).
    ///
    /// # Safety
    ///
    /// `ptr` is null or came from mimalloc.
    unsafe extern "C" fn realloc_compat(
        ptr: *mut std::ffi::c_void,
        size: usize,
    ) -> *mut std::ffi::c_void {
        // SAFETY: forwarded under the caller's contract.
        let grown = unsafe { libmimalloc_sys::mi_realloc(ptr, size) };
        if grown.is_null() && size != 0 {
            std::process::abort();
        }
        grown
    }
}

use anyhow::{Context, Result};
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

const EXPECTED_YARA_CACHE_MISMATCH: &str =
    "compiled YARA rules do not match the rule sources on disk; ignoring them and recompiling";

fn is_expected_yara_cache_mismatch(target: &str, message: &str) -> bool {
    target == "cleave::yara_engine" && message.contains(EXPECTED_YARA_CACHE_MISMATCH)
}

/// Hide one expected cache invalidation without muting real YARA errors from
/// the same tracing target. The upstream event remains available whenever the
/// operator explicitly asks for diagnostic logs.
///
/// cleave emits it as prose with no structured field to key on, so the
/// message is matched — but only for an ERROR from that one target, so no
/// other event pays for formatting it.
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
        let metadata = event.metadata();
        if !self.hide
            || *metadata.level() != tracing::Level::ERROR
            || metadata.target() != "cleave::yara_engine"
        {
            return true;
        }
        let mut message = EventMessage::default();
        event.record(&mut message);
        !is_expected_yara_cache_mismatch(metadata.target(), &message.text)
    }
}

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
        /// `<URL>/api/result`. A SHA hopper has not ingested is negotiated over
        /// `/api/known` and uploaded bytes-and-provenance first, so a
        /// never-before-seen sample lands as its own row instead of being
        /// dropped as an unknown-SHA no-op. Upload failures are reported as
        /// errors, but never make the scan fatal. Also settable via the
        /// `SCAN_HOPPER` env var.
        /// Authenticates with `~/.tok/hopper`; hopper rejects an
        /// unauthenticated request with 401.
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
        /// when hopper does not already have the SHA, then the verdict. A
        /// fetched URL is exactly the never-before-seen case, so this is the
        /// flag that gets a freshly-discovered sample into the corpus.
        /// Also settable via the `SCAN_HOPPER` env var.
        /// Authenticates with `~/.tok/hopper`; hopper rejects an
        /// unauthenticated request with 401.
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
        /// Authenticates with `~/.tok/hopper`; hopper rejects an
        /// unauthenticated request with 401.
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
    #[arg(long, default_value = "0", allow_hyphen_values = true)]
    max_rss_gb: i64,

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
    /// with `~/.tok/hopper`; hopper rejects an unauthenticated request with
    /// 401.
    #[arg(
        long,
        visible_alias = "upload",
        value_name = "URL",
        env = "SCAN_HOPPER"
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
/// never be able to refuse to boot a server.
fn warn_on_broken_freebsd_malloc_conf() {
    if !cfg!(target_os = "freebsd") {
        return;
    }
    let conf = scan::env::text(scan::env::MALLOC_CONF).unwrap_or_default();
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

fn main() -> ExitCode {
    #[cfg(all(
        unix,
        not(any(
            target_os = "freebsd",
            target_os = "dragonfly",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "illumos",
            target_os = "solaris",
            target_arch = "arm",
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

    let cli = Cli::parse_from(with_default_subcommand(std::env::args_os()));
    match run(cli) {
        Ok(code) => code,
        // The reader went away (`atomscan … | head`) before anything was
        // flagged: nothing is listening for an error, so stop quietly. (A
        // flagged scan returns its summary instead, and exits on its verdict.)
        Err(e) if is_broken_pipe(&e) => ExitCode::SUCCESS,
        // Ctrl-C stopped the scan: the conventional 128 + SIGINT, so a script
        // never mistakes a partial scan for a clean one.
        Err(e) if is_interrupted(&e) => {
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

/// Whether `e` was caused by writing to a closed pipe.
fn is_broken_pipe(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::BrokenPipe)
    })
}

/// Whether `e` reports a scan stopped by Ctrl-C.
fn is_interrupted(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::Interrupted)
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
        // `--llm` grades what a live query sends, which is the interpret render.
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
        use std::io::Write as _;
        std::io::stdout().lock().write_all(out.as_bytes())?;
        return Ok(ExitCode::SUCCESS);
    }

    configure_process(&global, &command)?;
    init_logging(&global, command.is_daemon());

    // Resolved after logging is up: with no hardcoded default, the model comes
    // from the endpoint's own listing, so which one was picked has to be
    // visible rather than swallowed by an uninitialized subscriber.
    let interpret = global.interpret_config()?;
    if let Some(cfg) = &interpret {
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

    // A held token picks the Enriched rules tree and `--enriched` insists on
    // it: decided before anything resolves one.
    cleave::enriched::resolve(global.enriched)?;
    // Stabilize cleave trait discovery before any cleave shared resources are
    // initialized. This avoids clone-into-existing-directory failures when the
    // default traits checkout was installed by cleave or another scan run.
    scan::traits_repo::prepare_runtime_env();

    warn_on_broken_freebsd_malloc_conf();
    scan::heap_profile::report_debug_allocator();
    scan::runtime::install_rayon_pool();
    scan::runtime::install_diagnostics();

    #[cfg(debug_assertions)]
    tracing::warn!(
        "DEBUG binary — scan will be very slow; use `make release` for production builds"
    );

    // Warn about missing analysis tools for commands that will run cleave.
    if command.is_scan() || command.is_daemon() || matches!(command, Commands::Validate { .. }) {
        scan::tools::warn_missing();
    }
    // Interactive commands get a once-a-day, zero-telemetry update notice. The
    // long-running daemons are excluded — they refresh on restart and shouldn't
    // print transient notices into their logs — and so is `validate`.
    if !command.is_daemon() && !matches!(command, Commands::Validate { .. }) {
        scan::update_check::maybe_notify(false);
    }
    // Default-on refresh: bring rules + models (+ bloom) current when the local
    // ruleset is over 24h stale — or immediately with `-u/--update` — unless
    // `--no-update` disables it. Scanning commands only: the
    // daemons refresh on restart, and `version`/`update-rules` manage updates
    // themselves.
    if command.is_scan() {
        // `--enriched` needs a token before the refresh below would ask for
        // one mid-download; asked here, once.
        if cleave::enriched::required() {
            cleave::rule_update::sign_in()?;
        }
        scan::auto_update::refresh_if_stale(
            global.refresh(),
            global.mode,
            global.format == scan::OutputFormat::Terminal,
        )?;
        // Where artifacts are judged, settled once the rules are: an enriched
        // scan that needed a token has been asked for it by now, and one whose
        // token was refused has fallen back to open source, and asks nothing.
        scan::remote::resolve(global.analyze, cleave::enriched::enabled())?;
    }

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
            let mut config = scan_config(&global, interpret, fetch, None)?;
            // Per-binary known-good/known-bad short-circuit (by executable sha256).
            if let Some(lookup) = publish_bloom_filters(global.mode) {
                config = config.with_bloom(global.mode, lookup);
            }
            Ok(finish_scan(&scan::ps::run(&config)?))
        }
        Commands::Sys => {
            let config = scan_config(&global, interpret, fetch, None)?;
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
        Commands::Serve(args) => serve(&global, args, interpret, fetch),
        Commands::UpdateRules { models_only, check } => Ok(update_rules(
            models_only,
            check,
            global.model_dir.as_deref(),
        )),
        Commands::Validate { skip_traits } => {
            let model_dir = scan::cli::resolve_model_dir(global.model_dir.clone())?;
            let level = operating_level(&global, &model_dir);
            let config = scan::ScanConfig::new(
                model_dir,
                scan::OutputFormat::Terminal,
                global.thresholds(),
            )?
            .with_extra(global.extra)
            .with_level(level)
            .with_zip_passwords(global.zip_passwords);
            scan::validate::run(&config, skip_traits)?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Version => {
            print_version(global.format)?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Settle the process's settings from its flags, and start the cache sweeper.
fn configure_process(global: &GlobalArgs, command: &Commands) -> Result<()> {
    scan::settings::configure(scan::settings::Settings {
        rizin_timeout: Some(Duration::from_secs(global.rizin_timeout_secs)),
        rizin_retry_timeout: global.rizin_retry_timeout_secs.map(Duration::from_secs),
        // Every mode, interactive scan and server alike, honors
        // `--fetch-max-size` and `--registry-ttl`.
        fetch: scan::fetch::Settings {
            max_fetch_bytes: global.fetch_max_size,
            registry_ttl: global.registry_ttl,
        },
        // A hard cap across one invocation. The daemons scan unboundedly many
        // jobs over their lifetime, so each job is bounded by
        // `--fetch-max-file-*` instead.
        fetch_total: (!command.is_daemon()).then_some(scan::settings::FetchTotal {
            fetches: global.fetch_max_total_fetches,
            bytes: global.fetch_max_total_size,
        }),
        fetch_backlog: global.fetch_pending.clone(),
        cache_dir: None,
        theme: theme(global, command),
    })?;
    // Reclaim stale/oversized caches (stng strings+r2, scan analysis+interpret,
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
    // Cleave rejects a stale precompiled YARA cache safely and recompiles from
    // source. That expected maintenance path is not an operator-facing error;
    // filter only its exact event. Explicit diagnostics still show it.
    let quiet_expected_yara_cache = ExpectedYaraCacheFilter { hide: !explicit };
    let layer = tracing_subscriber::fmt::layer()
        .with_thread_names(true)
        .with_writer(std::io::stderr);
    if daemon {
        tracing_subscriber::registry()
            .with(filter)
            .with(layer.with_filter(quiet_expected_yara_cache))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(layer.without_time().with_filter(quiet_expected_yara_cache))
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

/// The operating point for a run against `model_dir`; see
/// [`scan::cli::operating_level`].
fn operating_level(global: &GlobalArgs, model_dir: &Path) -> Option<u16> {
    let manual = global.threshold_suspicious.is_some() || global.threshold_hostile.is_some();
    scan::cli::operating_level(global.level, manual, model_dir)
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
    let model_dir = scan::cli::resolve_model_dir(global.model_dir.clone())?;
    let level = operating_level(global, &model_dir);
    Ok(
        scan::ScanConfig::new(model_dir, global.format, global.thresholds())?
            .with_filter(global.display_filter())
            .with_extra(global.extra)
            .with_level(level)
            .with_interpret(interpret)
            .with_fetch(fetch)
            .with_zip_passwords(global.zip_passwords.clone())
            .with_hopper(resolve_hopper(hopper)),
    )
}

/// `atomscan path`: scan files and directories.
fn scan_paths(
    global: &GlobalArgs,
    mut config: scan::ScanConfig,
    paths: &[PathBuf],
    registry_map: Option<&Path>,
) -> Result<ExitCode> {
    // Per-file SHA-256 known-good/known-bad short-circuit, for the files found
    // by walking a directory the operator named. A path named directly on the
    // command line is always analyzed — see `engine::named_target_opts`. Slow
    // mode skips it entirely.
    if let Some(lookup) = publish_bloom_filters(global.mode) {
        config = config.with_bloom(global.mode, lookup);
    }
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

/// A multi-threaded runtime for a daemon.
///
/// Its blocking threads run small analyses inline (`classify_bytes` without a
/// whale lane), so they get the same stack as every analysis pool
/// ([`scan::RAYON_STACK_MB`]): a hostile sample that recurses fine on a pool
/// must not overflow tokio's 2 MiB default and abort the whole daemon.
fn daemon_runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(scan::RAYON_STACK_MB * 1024 * 1024)
        .build()?)
}

/// `atomscan serve`.
fn serve(
    global: &GlobalArgs,
    args: ServeArgs,
    interpret: Option<InterpretConfig>,
    fetch: scan::fetch::FetchPolicy,
) -> Result<ExitCode> {
    let hopper = resolve_hopper(args.hopper);
    let max_rss = MaxRssPolicy::from_cli(args.max_rss_gb);
    log_max_rss_resolution(
        "server",
        max_rss,
        max_rss
            .process_ceiling()
            .map_or(0, std::num::NonZeroU64::get),
    );
    // `resolve` also refreshes rules on restart. That is the step that
    // populates a `--traits-dir` pointing at an empty state directory on a
    // fresh deploy — without it the server starts, reports healthy, and fails
    // every analysis on a traits path that never got created.
    let config = scan::server::Startup {
        bind: args.bind,
        max_size_mb: args.max_size_mb,
        max_rss,
        allow_local_paths: args.allow_local_paths,
        workers: args.workers,
        allow_cidr: args.allow_cidr,
        token_file: args.token_file,
        hopper: hopper.clone(),
        analysis_timeout_secs: args.analysis_timeout,
        rules: scan::rules::RulesStartup {
            model_dir: global.model_dir.clone(),
            level: global.level,
            thresholds: global.thresholds(),
            traits_dir: args.traits_dir,
            refresh: global.refresh(),
            slow_rule_ms: scan::cli::DEFAULT_SLOW_RULE_MS,
            interpret,
            fetch,
            zip_passwords: global.zip_passwords.clone().into(),
        },
    }
    .resolve()?;
    if let Some(url) = hopper.as_deref() {
        eprintln!("Renewing results on hopper at {url}");
        eprintln!("Deferring unknown lookups to the corpus at {url}");
    }
    // Serve never bloom-skips an /analyze job (Mode::Slow), but the membership
    // endpoint and the --follow dependency gate read the process-wide handle.
    // Missing files fail closed (no skip).
    publish_bloom_filters(global.mode);
    eprintln!("Starting Atomdrift Scan server on http://{} ...", args.bind);
    let runtime = daemon_runtime()?;
    let served = runtime.block_on(scan::server::run(config));
    // Runtime shutdown may abandon blocking analyses. Close child admission
    // and kill live tool groups while their supervisors still exist.
    cleave::kill_all_rizin_groups();
    runtime.shutdown_timeout(RUNTIME_SHUTDOWN_GRACE);
    served?;
    Ok(ExitCode::SUCCESS)
}

/// `atomscan update-rules`: check for, or apply, rule updates. Traits and
/// models failing is an error; the bloom and NSRL filters only let a scan skip
/// work, so their failure is a warning.
fn update_rules(models_only: bool, check: bool, model_dir: Option<&Path>) -> ExitCode {
    use scan::rules::Source;
    let sources: &[Source] = if models_only {
        &[Source::Models]
    } else {
        &Source::ALL
    };
    if check {
        return check_rules(sources, model_dir);
    }
    let mut code = ExitCode::SUCCESS;
    for (source, e) in scan::rules::refresh(sources, model_dir, false, false).failures() {
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

/// `atomscan update-rules --check`: report what each source would install.
fn check_rules(sources: &[scan::rules::Source], model_dir: Option<&Path>) -> ExitCode {
    use scan::rules::Source;
    let mut code = ExitCode::SUCCESS;
    for &source in sources {
        let checked = match source {
            Source::Traits => scan::traits_repo::check_updates(),
            Source::Models => model_dir
                .map_or_else(scan::models_repo::install_target, |dir| {
                    Ok(dir.to_path_buf())
                })
                .and_then(|dir| scan::model_update::check(&dir)),
            Source::Bloom => {
                scan::bloom_repo::install_dir().and_then(|dir| scan::bloom_update::check(&dir))
            }
            Source::Nsrl => {
                scan::nsrl_repo::install_dir().and_then(|dir| scan::nsrl_repo::check(&dir))
            }
        };
        match checked {
            Ok(()) => {}
            Err(e) if source.optional() => {
                eprintln!("Warning: {} check failed: {e:#}", source.name());
            }
            Err(e) => {
                eprintln!("Error checking {} updates: {e:#}", source.name());
                code = ExitCode::FAILURE;
            }
        }
    }
    code
}

/// `atomscan version`: this build, and the identity and size of every rule
/// source it would scan with. A closed pipe is an error, not a panic.
fn print_version(format: scan::OutputFormat) -> std::io::Result<()> {
    use std::io::Write as _;
    let bloom = scan::bloom_repo::installed_manifest();
    let nsrl = scan::nsrl_repo::installed_manifest();
    if format == scan::OutputFormat::Json {
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
        // Dates beside the commits, so a benchmark comparing two rules channels
        // can say how fresh each one was, not only which it was.
        let traits_date =
            cleave::rule_update::installed(&cleave::traits_repo::install_target()).map(|i| i.date);
        let models_date = scan::models_repo::install_target()
            .ok()
            .and_then(|dir| scan::model_update::installed(&dir))
            .map(|i| i.date);
        let version = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "models": scan::models_repo::version(),
            "models_date": models_date,
            "traits": cleave::traits_repo::version(),
            "traits_date": traits_date,
            "traits_channel": cleave::traits_repo::Edition::current().as_str(),
            "bloom": bloom_json,
            "nsrl": nsrl,
            "nsrl_release": scan::nsrl_repo::installed_release(),
        });
        return writeln!(std::io::stdout(), "{version}");
    }

    use scan::output::SourceRow;

    // Counting traits/composites/YARA initialises the rule engines.
    let info = cleave::version_info();

    // Atomics, composites, and third-party YARA all ship in the traits repo, so
    // they share one identity and build date. Prefer the installed bundle's
    // sidecar (commit + committer date); for a dev checkout with no sidecar,
    // read git HEAD directly.
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

    let rules = info.trait_count as u64 + info.composite_count as u64 + info.yara_rules as u64;

    let mut rows = Vec::new();
    if let Some(nsrl) = nsrl {
        rows.push(SourceRow {
            label: "nsrl",
            count: nsrl.filter.values().map(|e| e.n).sum(),
            metric: None,
            version: scan::nsrl_repo::installed_release().unwrap_or_else(|| "v1".to_owned()),
            epoch: scan::output::parse_ymd(&nsrl.built),
        });
    }
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

    // ML models: route-model count by feature dimensionality (e.g. `64 x 229`),
    // plus the bundle's commit and build date.
    if let Some(count) = scan::models_repo::model_count() {
        let model_installed = scan::models_repo::install_target()
            .ok()
            .and_then(|dir| scan::model_update::installed(&dir));
        let metric = scan::models_repo::feature_dimension().map(|dim| format!("{count} x {dim}"));
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

    scan::output::print_version(
        env!("CARGO_PKG_VERSION"),
        &info.traits_edition.styled(),
        bloom_count,
        rules,
        &rows,
    )
}

/// The abbreviated commit hash and committer date (Unix seconds) of `HEAD` in
/// `dir`, when `dir` is a git checkout. Returns `None` if git is unavailable or
/// `dir` is not a repository, in which case `scan version` omits that source's
/// identity. This recovers the authentic upstream identity and date for a traits
/// checkout that has no install sidecar.
fn git_head(dir: &Path) -> Option<(String, i64)> {
    let out = std::process::Command::new("git")
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

#[cfg(test)]
#[path = "main/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "main/dependency_audit_cli_tests.rs"]
mod dependency_audit_cli_tests;
