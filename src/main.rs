//! Atomdrift Scan (`atomscan`) — ML-powered malware classification CLI.

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
use clap::{Parser, Subcommand};
use scan::OutputFormat;
use scan::engine::DisplayFilter;
use std::net::SocketAddr;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::process;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const DEFAULT_RIZIN_TIMEOUT_SECS: u64 = 10 * 60;

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

/// Dependency age ceiling for `worker` mode: `0` — no gate, fetch every
/// resolvable dependency.
///
/// An interactive scan gates at [`scan::fetch::DEFAULT_MAX_DEP_AGE_DAYS`] because
/// a fresh release is where a supply-chain compromise shows up and the operator is
/// waiting. A worker is the opposite trade: it runs unattended to populate the
/// shared corpus, and every dependency it resolves lands in hopper carrying a
/// package coordinate — the raw material known-good bloom coverage is built from.
/// Gating those out means the cache never learns the long tail that real scans
/// keep re-resolving.
const WORKER_MAX_DEP_AGE_DAYS: u32 = 0;

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

    /// Hard wall-clock limit for each Rizin subprocess, in seconds. On expiry
    /// Atomscan kills and reaps Rizin before releasing the analysis worker; on
    /// Unix it also kills the complete process group. Also settable via
    /// `SCAN_RIZIN_TIMEOUT_SECS`.
    #[arg(
        long,
        global = true,
        value_name = "SECS",
        env = "SCAN_RIZIN_TIMEOUT_SECS",
        default_value_t = DEFAULT_RIZIN_TIMEOUT_SECS,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    rizin_timeout_secs: u64,

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

    /// [deprecated] Legacy on-switch for LLM interpretation; superseded by
    /// `--llm`. Kept for compatibility (env: SCAN_INTERPRET).
    #[arg(long, global = true, env = "SCAN_INTERPRET", hide = true)]
    interpret: bool,

    /// [optional] Enable additional LLM interpretation of analyzed samples: a
    /// second opinion blended with the ML verdict (stored in the `llm` JSON
    /// section and shown inline). Given with no value, uses a local model (an
    /// OpenAI-compatible endpoint at http://localhost:8000/v1). TARGET may be
    /// `local`, `openrouter` (https://openrouter.ai/api/v1; key from `--llm-key`,
    /// `SCAN_LLM_KEY`, or `~/.tok/openrouter`; `--llm-model` is required), or an
    /// explicit OpenAI-compatible base URL. (env: SCAN_LLM)
    #[arg(
        long,
        global = true,
        value_name = "TARGET",
        num_args = 0..=1,
        default_missing_value = "local",
    )]
    llm: Option<String>,

    /// LLM model name, e.g. Qwen/Qwen3.8-27B. Defaults to the largest model the
    /// endpoint itself reports serving; nothing is hardcoded (env:
    /// SCAN_LLM_MODEL)
    #[arg(long, global = true, value_name = "NAME")]
    llm_model: Option<String>,

    /// LLM bearer token (env: SCAN_LLM_KEY); omit for local endpoints
    #[arg(long, global = true, value_name = "KEY")]
    llm_key: Option<String>,

    /// Loosest FP level (0-25000, per 100M benigns) at which ML alone sends a
    /// sample to the LLM; higher = more samples. Defaults to the model's grid
    /// ceiling — anything ML flagged at any level. Files cleave flagged
    /// suspicious/hostile are sent regardless of this cutoff.
    #[arg(
        long,
        global = true,
        alias = "interpret-min-level",
        value_parser = clap::value_parser!(u16).range(0..=25000),
        value_name = "N",
    )]
    llm_min_level: Option<u16>,

    /// Per-request LLM timeout, in seconds
    #[arg(long, global = true, default_value_t = scan::interpret::DEFAULT_TIMEOUT_SECS, value_name = "SECS")]
    llm_timeout: u64,

    /// Additional passwords to try for encrypted ZIP/7z archives. Repeat the
    /// option to provide more than one; cleave's common defaults remain active.
    #[arg(long = "zip-password", value_name = "PASSWORD", global = true)]
    zip_passwords: Vec<String>,

    /// [EXPERIMENTAL] Follow references discovered inside the requested
    /// artifact, analyze their payloads, and fold them into the verdict:
    ///   `dependencies` — manifest and lockfile dependencies;
    ///   `references`   — packages and URLs named by install/download commands;
    ///   `ci-actions`   — third-party actions referenced by CI configuration.
    /// `all` selects every category; `none` analyzes only the requested artifact.
    /// A bare `--follow` and the default follow dependencies and references but
    /// not CI actions. The old `--fetch` flag and `deps`, `packages`, `urls`, and
    /// `ci` values remain accepted as aliases. Also settable via `SCAN_FOLLOW`;
    /// `SCAN_FETCH` remains a compatibility alias.
    #[arg(
        long = "follow",
        visible_alias = "fetch",
        global = true,
        value_name = "TARGETS",
        num_args = 0..=1,
        require_equals = true,
        // A bare `--follow` selects the artifact's own reachable code, not CI:
        // `--follow=all` (or `--follow=ci-actions`) is the explicit opt-in for GitHub
        // Actions, which run only in CI and never reach an installed artifact.
        // Keep in lockstep with `default_cli_follow_policy`, which resolves an
        // absent `--follow` to the same targets.
        default_missing_value = "references,dependencies",
        env = "SCAN_FOLLOW"
    )]
    follow: Option<scan::fetch::FetchPolicy>,

    /// [EXPERIMENTAL] How many hops of references to follow when `--follow` is on:
    /// `1` fetches only what the scanned files reference, `2` also follows
    /// references found inside those payloads (reaching a stage-3 `curl | bash`
    /// dropper), and so on. Also settable via `SCAN_FOLLOW_DEPTH`; the old
    /// `--fetch-depth` and `SCAN_FETCH_DEPTH` names remain aliases.
    #[arg(
        long = "follow-depth",
        visible_alias = "fetch-depth",
        global = true,
        value_name = "N",
        default_value_t = scan::fetch::DEFAULT_FETCH_DEPTH,
        env = "SCAN_FOLLOW_DEPTH"
    )]
    fetch_depth: u8,

    /// [EXPERIMENTAL] Skip fetching a declared dependency whose registry publish
    /// date is older than this many days — the cheap provenance lookup runs
    /// first, and only recent (freshest-risk) releases are pulled and scanned.
    /// Applies to declared dependencies only; URLs are never age-gated. `0`
    /// disables the gate (fetch every resolvable dependency). A dependency whose
    /// age can't be determined is always fetched. Also settable via
    /// `SCAN_FETCH_MAX_AGE`.
    ///
    /// Unset, the ceiling depends on the mode: an interactive scan wants a
    /// fresh-risk window ([`scan::fetch::DEFAULT_MAX_DEP_AGE_DAYS`]), while a
    /// worker is a cache-population role and takes every resolvable dependency
    /// ([`WORKER_MAX_DEP_AGE_DAYS`]). Optional rather than defaulted so an
    /// explicit `--fetch-max-age` still wins in both.
    #[arg(long, global = true, value_name = "DAYS", env = "SCAN_FETCH_MAX_AGE")]
    fetch_max_age: Option<u32>,

    /// [EXPERIMENTAL] Fetch and scan native-binary dependencies for every
    /// platform, not just the host's. This is automatic in `serve` and
    /// `worker`, which scan on behalf of other machines; interactive scans
    /// stay host-only for latency unless this flag is passed. Also settable via
    /// `SCAN_FETCH_ALL_PLATFORMS`.
    #[arg(
        long,
        global = true,
        env = "SCAN_FETCH_ALL_PLATFORMS",
        conflicts_with = "fetch_host_platform_only"
    )]
    fetch_all_platforms: bool,

    /// [EXPERIMENTAL] Fetch only native-binary dependencies matching this
    /// host's OS and architecture. This is the interactive default and an
    /// explicit completeness opt-out for `serve` / `worker`. Also settable
    /// via `SCAN_FETCH_HOST_PLATFORM_ONLY`.
    #[arg(
        long,
        global = true,
        env = "SCAN_FETCH_HOST_PLATFORM_ONLY",
        conflicts_with = "fetch_all_platforms"
    )]
    fetch_host_platform_only: bool,

    /// [EXPERIMENTAL] Follow declared dependencies past the first hop — the
    /// dependencies of a fetched dependency, out to `--fetch-depth`. This is
    /// automatic in `serve` and `worker`, which populate the shared corpus;
    /// an interactive scan stops declared dependencies at the first hop, since
    /// the transitive tail costs a registry lookup each and is almost entirely
    /// old releases the age gate then discards. URLs and command-mentioned
    /// packages — the dropper chain — are followed at every hop either way.
    /// Also settable via `SCAN_FETCH_TRANSITIVE_DEPS`.
    #[arg(
        long,
        global = true,
        env = "SCAN_FETCH_TRANSITIVE_DEPS",
        conflicts_with = "fetch_direct_deps_only"
    )]
    fetch_transitive_deps: bool,

    /// [EXPERIMENTAL] Follow declared dependencies for one hop only. This is the
    /// interactive default and an explicit completeness opt-out for `serve` /
    /// `worker`. Also settable via `SCAN_FETCH_DIRECT_DEPS_ONLY`.
    #[arg(
        long,
        global = true,
        env = "SCAN_FETCH_DIRECT_DEPS_ONLY",
        conflicts_with = "fetch_transitive_deps"
    )]
    fetch_direct_deps_only: bool,

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

    /// [EXPERIMENTAL] Maximum number of *live* dependency/package fetches
    /// triggered by a single scanned file. This is 100 by default. Cache hits
    /// are always served and never counted, so a warm re-run is never throttled.
    /// References past the cap are recorded as budget-exceeded, never silently
    /// dropped. Also settable via `SCAN_FETCH_MAX_FILE_FETCHES`.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = scan::fetch::DEFAULT_MAX_FILE_FETCHES,
        env = "SCAN_FETCH_MAX_FILE_FETCHES"
    )]
    fetch_max_file_fetches: usize,

    /// [EXPERIMENTAL] Maximum number of *live* opportunistic raw-URL fetches
    /// triggered by a single scanned file. This is 4 by default. URL references
    /// declared as dependencies or command-mentioned packages use the larger
    /// `--fetch-max-file-fetches` cap instead. Also settable via
    /// `SCAN_FETCH_MAX_URLS`.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = scan::fetch::DEFAULT_MAX_URL_FETCHES,
        env = "SCAN_FETCH_MAX_URLS"
    )]
    fetch_max_urls: usize,

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
    /// Build the LLM interpretation config from `--llm` (or the legacy
    /// `--interpret`) and the `--llm-*` flags, falling back to env vars. `None`
    /// when interpretation is not requested.
    fn interpret_config(&self) -> Result<Option<scan::interpret::InterpretConfig>> {
        use scan::interpret::{
            DEFAULT_BASE_URL, DEFAULT_MAX_CONCURRENCY, is_openrouter_endpoint, llm_base_url,
            openrouter_key_from_home,
        };
        let from_env = |flag: &Option<String>, key: &str| -> Option<String> {
            flag.clone()
                .or_else(|| std::env::var(key).ok())
                .filter(|s| !s.is_empty())
        };
        // `--llm [TARGET]` / SCAN_LLM (the bare flag defaults TARGET to `local`)
        // or the legacy `--interpret` flag turns the pass on.
        let target = from_env(&self.llm, "SCAN_LLM");
        if target.is_none() && !self.interpret {
            return Ok(None);
        }
        // Resolve the target to a base URL: `local` (also the bare-flag default)
        // maps to the local endpoint; `openrouter` is the public API; anything
        // else is an OpenAI-compatible base URL.
        let base_url = match target.as_deref() {
            None | Some("local") => DEFAULT_BASE_URL.to_string(),
            Some(name) => llm_base_url(name),
        };
        let mut api_key = from_env(&self.llm_key, "SCAN_LLM_KEY");
        if api_key.is_none() && is_openrouter_endpoint(&base_url) {
            api_key = openrouter_key_from_home();
        }
        let openrouter = is_openrouter_endpoint(&base_url);
        if openrouter && api_key.is_none() {
            anyhow::bail!(
                "OpenRouter requires a key: --llm-key, SCAN_LLM_KEY, or ~/.tok/openrouter"
            );
        }
        // A pinned model wins; otherwise take what the endpoint says it serves.
        // OpenRouter's catalog is large and billed — never auto-pick. Nothing
        // else is hardcoded: if a local endpoint lists no model there is
        // nothing sensible to send, and a guessed name would surface as an
        // opaque server-side error mid-scan instead of here.
        let model = from_env(&self.llm_model, "SCAN_LLM_MODEL");
        let model = if openrouter {
            model.ok_or_else(|| {
                anyhow::anyhow!(
                    "OpenRouter requires --llm-model (env: SCAN_LLM_MODEL); \
                     the catalog is not auto-selected"
                )
            })?
        } else {
            model
                .or_else(|| scan::interpret::discover_model(&base_url, api_key.as_deref()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no LLM model available: {base_url}/models listed none (or was \
                         unreachable). Start the endpoint, or name a model with \
                         --llm-model (env: SCAN_LLM_MODEL)"
                    )
                })?
        };
        Ok(Some(scan::interpret::InterpretConfig {
            base_url,
            model,
            api_key,
            min_level: self.llm_min_level,
            timeout: std::time::Duration::from_secs(self.llm_timeout),
            max_concurrency: NonZeroUsize::new(DEFAULT_MAX_CONCURRENCY)
                .unwrap_or(NonZeroUsize::MIN),
        }))
    }
}

/// The binary's online default: everything the artifact itself reaches, but not
/// CI — GitHub Actions run only on a runner and never land in an installed
/// artifact, so auditing them is an explicit
/// `--follow=ci-actions`/`--follow=all` opt-in.
/// Keep [`FetchPolicy::default`] offline for the library API.
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
/// [`scan::fetch::age_gate`] can skip a dependency whose coordinate is already
/// vouched. This is deliberately separate from `ScanConfig::with_bloom`, which
/// additionally lets the bloom short-circuit the *scan target* itself: every
/// mode wants the dependency skip, but only a bulk walk wants its own input
/// answered from a bless.
///
/// `--mode slow` means "consult no filters", so it publishes nothing. Note this
/// reads the operator's `cli.mode` rather than the effective mode: `serve` and
/// `worker` force themselves slow so a submitted job is always analyzed on its
/// own merits, and that internal choice must not also switch off their
/// dependency skip — only an explicit `--mode slow` does.
fn publish_bloom_filters(mode: scan::Mode) {
    if mode != scan::Mode::Slow {
        scan::bloom_repo::set_global(std::sync::Arc::new(scan::bloom_repo::Lookup::load()));
    }
}

fn cli_host_platform_only(cli: &Cli, scans_for_other_hosts: bool) -> bool {
    cli.fetch_host_platform_only || (!cli.fetch_all_platforms && !scans_for_other_hosts)
}

/// Whether declared dependencies are followed past the first hop. Corpus-facing
/// modes take the transitive tail by default; an interactive scan stops at the
/// artifact's own declared dependencies. Either default is overridable, and the
/// two flags conflict, so at most one arm can fire.
fn cli_transitive_deps(cli: &Cli, scans_for_other_hosts: bool) -> bool {
    cli.fetch_transitive_deps || (!cli.fetch_direct_deps_only && scans_for_other_hosts)
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
        /// moment a request arrives.
        ///
        /// A serve process spends most of its life waiting while hopper holds a
        /// backlog, so the spare capacity is otherwise wasted. Requests always
        /// win: the worker stops claiming while any is in flight, and the slots
        /// it is *not* given are the interactive reserve, so an arriving
        /// request never queues behind background work.
        ///
        /// Defaults to 1, leaving every other slot reserved for requests, and
        /// the worker holds exactly one claim at a time. 0 disables it.
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

/// Refresh models and traits before a long-lived daemon starts serving.
///
/// `force` is `-u/--update`: re-fetch even when the local copy looks current.
///
/// Both daemons re-read rules only at startup, so this is what a restart is
/// for. Failures are warnings, never fatal: a disconnected host must still come
/// up on whatever is already on disk. It runs *after* `--traits-dir` has been
/// applied, so a pinned traits directory is the one that gets installed into —
/// which is also how that directory comes to exist on a fresh deploy.
fn refresh_rules_at_startup(force: bool, no_update: bool) {
    if no_update {
        tracing::warn!("--no-update: skipping startup model/traits refresh");
        return;
    }
    std::thread::scope(|s| {
        s.spawn(|| {
            let dir = scan::models_repo::install_target();
            if let Err(e) = scan::model_update::update(&dir, force, false) {
                tracing::warn!(dir = %dir.display(), error = %e, "model update failed");
            }
        });
        s.spawn(|| {
            if let Err(e) = scan::traits_repo::update(force, false) {
                tracing::warn!(
                    dir = %cleave::traits_repo::install_target().display(),
                    error = format!("{e:#}"),
                    "traits update failed",
                );
            }
        });
    });
}

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

    // Preserve the old deployment variable while teaching new configurations
    // the same vocabulary as the HTTP API. The canonical variable wins when
    // both are present. This runs before clap or any worker thread starts.
    if std::env::var_os("SCAN_FOLLOW").is_none()
        && let Some(value) = std::env::var_os("SCAN_FETCH")
    {
        // SAFETY: argument parsing happens before this process starts threads.
        unsafe { std::env::set_var("SCAN_FOLLOW", value) };
    }
    if std::env::var_os("SCAN_FOLLOW_DEPTH").is_none()
        && let Some(value) = std::env::var_os("SCAN_FETCH_DEPTH")
    {
        // SAFETY: argument parsing happens before this process starts threads.
        unsafe { std::env::set_var("SCAN_FOLLOW_DEPTH", value) };
    }
    let mut cli = Cli::parse();
    // Install the process-wide Rizin budget before any analysis or Rayon worker
    // can start. Filefacts owns the subprocess lifecycle; Atomscan only selects
    // the deadline through this single global configuration point.
    filefacts::rizin::set_timeout_secs(cli.rizin_timeout_secs);
    let selected_severity_level = cli.level;
    let threshold_suspicious = cli.threshold_suspicious;
    let threshold_hostile = cli.threshold_hostile;
    // The selection comes from `--follow`; the hop count from `--follow-depth`,
    // the dependency age ceiling from `--fetch-max-age`, and the per-file ceilings
    // from `--fetch-max-file-*` (each its own flag/env). When `--follow`/`SCAN_FOLLOW`
    // is unset, the binary defaults to dependencies and executable references
    // in every mode (CI actions remain opt-in). An explicit selection is honored
    // verbatim in both; the knobs always apply.
    let with_knobs = |mut policy: scan::fetch::FetchPolicy,
                      default_max_age: u32,
                      scans_for_other_hosts: bool| {
        policy.depth = cli.fetch_depth;
        policy.max_dep_age_days = cli.fetch_max_age.unwrap_or(default_max_age);
        policy.max_file_fetches = cli.fetch_max_file_fetches;
        policy.max_url_fetches = cli.fetch_max_urls;
        policy.max_file_bytes = cli.fetch_max_file_size;
        policy.host_platform_only = cli_host_platform_only(&cli, scans_for_other_hosts);
        policy.transitive_deps = cli_transitive_deps(&cli, scans_for_other_hosts);
        policy
    };
    // The per-fetch size ceiling is enforced in the HTTP layer, so it's a
    // process-global rather than a per-policy field — set it once here and every
    // mode (interactive scan and worker alike) honors `--fetch-max-size`.
    scan::fetch::set_max_fetch_bytes(cli.fetch_max_size);
    // Registry-metadata staleness bound: a process-global for the same reason,
    // consulted by every registry lookup. `None` keeps the tiered defaults.
    scan::fetch::set_registry_ttl(cli.registry_ttl);
    let scans_for_other_hosts = command_scans_for_other_hosts(cli.command.as_ref());
    let fetch_policy = with_knobs(
        cli.follow.unwrap_or_else(default_cli_follow_policy),
        scan::fetch::DEFAULT_MAX_DEP_AGE_DAYS,
        scans_for_other_hosts,
    );
    // A worker exists to populate the shared corpus, not to answer one question
    // quickly: every resolvable dependency it pulls becomes a hopper sample with a
    // package coordinate, which is what later grows known-good bloom coverage. The
    // fresh-risk window that keeps an interactive scan fast is exactly the wrong
    // default there — it discards the long tail the cache most wants.
    let worker_fetch_policy = with_knobs(
        cli.follow.unwrap_or_else(default_cli_follow_policy),
        WORKER_MAX_DEP_AGE_DAYS,
        true,
    );

    // Default to a file scan when bare paths are given without a subcommand.
    // Taken, not moved, so `cli` stays whole for `interpret_config` below.
    let command = match cli.command.take() {
        Some(cmd) => cmd,
        None => {
            if !cli.paths.is_empty() {
                Commands::Path {
                    paths: cli.paths.clone(),
                    hopper: resolve_hopper(None),
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
    // `atomscan` is this binary's own target: startup diagnostics logged from
    // main.rs (rule refresh, authentication, LLM configuration) are not in the
    // `scan` library and were being filtered out of the daemons' logs.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        if cli.verbose {
            tracing_subscriber::EnvFilter::new("atomscan=debug,scan=debug,cleave=debug")
        } else if is_serve {
            tracing_subscriber::EnvFilter::new("atomscan=info,scan=info,cleave=warn")
        } else {
            tracing_subscriber::EnvFilter::new("atomscan=warn,scan=warn,cleave=error")
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

    // Resolved after logging is up: with no hardcoded default, the model comes
    // from the endpoint's own listing, so which one was picked has to be
    // visible rather than swallowed by an uninitialized subscriber.
    let interpret_cfg = cli.interpret_config()?;
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

    const RAYON_FALLBACK_THREADS: usize = 4;
    // Physical cores, matching cleave's own pool. Logical SMT siblings
    // (32 on this 16-core host) oversubscribe archive-member analysis:
    // S2 dropped 56.6 s → 51.8 s at 16 threads.
    let detected_cores =
        cleave::memory_tracker::physical_cpu_count().or_else(cleave::memory_tracker::cpu_count);
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
    let new_scan_config = |hopper| -> Result<scan::ScanConfig> {
        let model_dir = resolve_model_dir()?;
        let envelope_level = resolve_envelope_level(&model_dir);
        Ok(scan::ScanConfig::new(
            model_dir,
            cli.format,
            threshold_overrides(),
            filter,
            DEFAULT_SLOW_RULE_MS,
            cli.extra,
        )?
        .with_level(envelope_level)
        .with_interpret(interpret_cfg.clone())
        .with_fetch(fetch_policy)
        .with_zip_passwords(cli.zip_passwords.clone())
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
            publish_bloom_filters(cli.mode);
            exit_for_summary(&scan::pkg::run_url(&url, &config)?);
        }
        Commands::Purl { purl, hopper } => {
            let config = new_scan_config(hopper)?;
            publish_bloom_filters(cli.mode);
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
            if let Some(p) = traits_dir.as_ref() {
                cleave::traits_repo::set_override_dir(Some(p.into()));
            }
            // Same contract as the worker: refresh on restart. This is also the
            // step that populates a `--traits-dir` pointing at an empty state
            // directory on a fresh deploy — without it the server starts, reports
            // healthy, and fails every analysis on a traits path that never
            // got created.
            refresh_rules_at_startup(cli.update, cli.no_update);
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
            let workers = workers.unwrap_or_else(default_workers).get();
            let allow_cidrs = match allow_cidr {
                Some(s) => scan::server::parse_cidr_list(&s)
                    .map_err(|e| anyhow::anyhow!("--allow-cidr: {e}"))?,
                None => Vec::new(),
            };
            // Fail closed: an operator who asked for authentication must never
            // get an open server because the token file went missing.
            let auth_digest = match token_file {
                Some(ref path) => {
                    let token = scan::interpret::read_token_file(path).ok_or_else(|| {
                        anyhow::anyhow!(
                            "--token-file {}: missing, empty, or unreadable",
                            path.display()
                        )
                    })?;
                    let digest = scan::server::TokenDigest::new(&token)
                        .map_err(|e| anyhow::anyhow!("--token-file {}: {e}", path.display()))?;
                    // Name the file the running process actually read: after a
                    // rotation the token in a file and the token in memory can
                    // differ, and the 401 that follows is otherwise unreadable.
                    tracing::info!(
                        token_file = %path.display(),
                        "bearer authentication enabled (token is read once, at startup)",
                    );
                    Some(digest)
                }
                None => None,
            };
            let max_rss_bytes = resolve_process_max_rss_bytes(max_rss_gb);
            log_max_rss_resolution("server", MaxRssPolicy::from_cli(max_rss_gb), max_rss_bytes);
            let model_dir = resolve_model_dir()?;
            let envelope_level = resolve_envelope_level(&model_dir);
            let thresholds = threshold_overrides();
            // One slot by default: every other slot stays reserved for
            // requests. Pausing stops new claims but does not abandon a running
            // job, so the reserve — not the pause — is what keeps an arriving
            // request from waiting. One slot also bounds the worst case to a
            // single background analysis in flight, which is the whole point of
            // filling *idle* capacity rather than competing for it.
            //
            // Disabled without --hopper: there would be nothing to claim from.
            let idle_slots = match (hopper.as_deref(), idle_worker_slots) {
                (None, _) => 0,
                (Some(_), Some(n)) => n,
                (Some(_), None) => 1,
            };

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
            .with_auth_token(auth_digest)
            .with_interpret(interpret_cfg.clone())
            .with_fetch(fetch_policy)
            .with_zip_passwords(cli.zip_passwords.clone())
            .with_hopper(hopper.clone())
            .with_idle_worker_slots(idle_slots)
            .with_analysis_timeout(analysis_timeout);
            if let Some(url) = hopper.as_deref() {
                eprintln!("Renewing results on hopper at {url}");
            }
            if let Some(url) = hopper.as_deref() {
                eprintln!("Deferring unknown lookups to the corpus at {url}");
            }
            // Serve never bloom-skips an /analyze job (Mode::Slow), but the
            // membership endpoint and the --fetch dependency gate read the
            // process-wide handle. Missing files fail closed (no skip).
            publish_bloom_filters(cli.mode);
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
            .with_level(envelope_level)
            .with_zip_passwords(cli.zip_passwords.clone());
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
            // Claimed jobs are always analyzed in full (worker mode forces
            // `Mode::Slow`, and `ScanConfig` never gets `with_bloom`), but the
            // *dependency* gate in `fetch::age_gate` consults the process-wide
            // bloom handle: a fetched dep whose resolved coordinate is vouched
            // known-good (and not vetoed by the known-bad channel) skips its
            // artifact fetch+scan, keeping only the registry-metadata node.
            // Publishing the filters here enables that skip without touching
            // the job-level always-scan guarantee.
            publish_bloom_filters(cli.mode);
            // Accept the comma list `serve --hopper` takes, so one deploy
            // variable can feed both, but keep only the primary: a replica
            // refuses worker routes with a 403 whether or not its relay is on,
            // so the later addresses are not a fallback for this loop. Passing
            // the whole string through reaches a URL parser, which reads the
            // commas as part of one very strange hostname.
            let Some(url) = scan::upload::worker_endpoint(&url) else {
                anyhow::bail!("--url names no hopper address");
            };
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
                argv: redact_zip_passwords(std::env::args()),
                hopper_url: &url,
                name: &name,
                workers: workers.get(),
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
            refresh_rules_at_startup(cli.update, no_update);
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
                .with_level(envelope_level)
                .with_zip_passwords(cli.zip_passwords.clone());
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
                // Standalone: owns its signals, its nice value, and its models.
                embedded: None,
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
                zip_passwords: cli.zip_passwords.clone().into(),
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

/// Default worker count: at least 2, and the physical core count from
/// `sysmem` (SMT siblings don't add analysis throughput; on Apple silicon
/// this is the performance-core count via `hw.perflevel0.physicalcpu`).
/// Platforms where `sysmem` has no physical-core signal fall back to half
/// the logical CPUs — identical to the physical count on 2-way SMT.
fn default_workers() -> NonZeroUsize {
    if let Some(cores) = cleave::memory_tracker::physical_cpu_count() {
        return NonZeroUsize::new(std::cmp::max(2, cores)).unwrap_or(NonZeroUsize::MIN);
    }
    let cores = cleave::memory_tracker::cpu_count().unwrap_or_else(|| {
        tracing::warn!(
            fallback = 4,
            "CPU count detection failed; defaulting worker basis to 4 cores",
        );
        4
    });
    NonZeroUsize::new(std::cmp::max(2, cores / 2)).unwrap_or(NonZeroUsize::MIN)
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

fn resolve_worker_max_rss_gb(raw_max_rss_gb: i64) -> u64 {
    match MaxRssPolicy::from_cli(raw_max_rss_gb) {
        MaxRssPolicy::Disabled => 0,
        // 85% of the cgroup-aware memory basis, with a one-GiB floor. Slot
        // count scales with cores, so larger hosts need a proportionate ceiling.
        MaxRssPolicy::Auto => std::cmp::max(1, (worker_memory_basis().bytes * 85 / 100) / GIB),
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
    cleave::prefetch_shared_resources(true);

    // Explicit files are analyzed as one parallel batch and each directory is
    // streamed; run_paths shares one model load and verdict tally across all.
    scan::engine::run_paths(paths, config, registry_map)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{
        Cli, Commands, DEFAULT_RIZIN_TIMEOUT_SECS, GIB, MaxRssPolicy, default_cli_follow_policy,
        redact_zip_passwords, resolve_hopper_value, resolve_process_max_rss_bytes,
        resolve_worker_max_rss_gb,
    };
    use anyhow::{Context, Result};
    use clap::Parser;
    use std::net::SocketAddr;
    use std::num::NonZeroU64;
    use std::path::PathBuf;

    #[test]
    fn follow_defaults_to_references_and_dependencies() -> Result<()> {
        let cli = Cli::try_parse_from(["scan", "/tmp/a"]).context("parse should work")?;
        assert!(
            cli.follow.is_none(),
            "absence of --follow is resolved at startup"
        );

        let policy = default_cli_follow_policy();
        assert!(policy.urls && policy.packages && policy.deps);
        assert!(
            !policy.ci,
            "CI actions never reach an installed artifact; auditing them is opt-in"
        );
        // A bare `--follow` must select exactly what an absent one resolves to.
        let bare = Cli::try_parse_from(["scan", "--follow", "/tmp/a"])
            .context("bare --follow should parse")?
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

    #[test]
    fn old_fetch_flag_and_target_names_remain_aliases() -> Result<()> {
        let old = Cli::try_parse_from(["scan", "--fetch=deps,packages,urls,ci", "/tmp/a"])
            .context("legacy --fetch vocabulary should parse")?
            .follow
            .context("legacy --fetch should select a policy")?;
        let new = Cli::try_parse_from([
            "scan",
            "--follow=dependencies,references,ci-actions",
            "/tmp/a",
        ])
        .context("canonical --follow vocabulary should parse")?
        .follow
        .context("canonical --follow should select a policy")?;
        assert_eq!(old, new);
        Ok(())
    }

    #[test]
    fn rizin_timeout_defaults_to_ten_minutes_and_is_overridable() -> Result<()> {
        let default =
            Cli::try_parse_from(["scan", "/tmp/a"]).context("default timeout should parse")?;
        assert_eq!(default.rizin_timeout_secs, DEFAULT_RIZIN_TIMEOUT_SECS);

        let overridden = Cli::try_parse_from(["scan", "--rizin-timeout-secs", "42", "/tmp/a"])
            .context("timeout override should parse")?;
        assert_eq!(overridden.rizin_timeout_secs, 42);
        assert!(
            Cli::try_parse_from(["scan", "--rizin-timeout-secs", "0", "/tmp/a"]).is_err(),
            "zero would disable the hard deadline and must be rejected"
        );
        Ok(())
    }

    #[test]
    fn dependency_platform_scope_follows_scanner_role() -> Result<()> {
        let default = Cli::try_parse_from(["scan", "/tmp/a"]).context("default should parse")?;
        assert!(
            super::cli_host_platform_only(
                &default,
                super::command_scans_for_other_hosts(default.command.as_ref())
            ),
            "interactive scans optimize for the current host"
        );
        assert!(
            scan::fetch::FetchPolicy::default().host_platform_only,
            "library and CLI defaults must agree"
        );
        let compatible = Cli::try_parse_from(["scan", "--fetch-all-platforms", "/tmp/a"])
            .context("all-platforms opt-in should parse")?;
        assert!(!super::cli_host_platform_only(&compatible, false));

        let host_only = Cli::try_parse_from(["scan", "--fetch-host-platform-only", "/tmp/a"])
            .context("host-only opt-out should parse")?;
        assert!(super::cli_host_platform_only(&host_only, true));

        for daemon in [
            Cli::try_parse_from(["scan", "serve"]).context("serve should parse")?,
            Cli::try_parse_from(["scan", "worker", "--url", "http://hopper.test"])
                .context("worker should parse")?,
        ] {
            let role = super::command_scans_for_other_hosts(daemon.command.as_ref());
            assert!(role);
            assert!(
                !super::cli_host_platform_only(&daemon, role),
                "serve/worker scan on behalf of other hosts"
            );
        }
        Ok(())
    }

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
        let cli = Cli::try_parse_from([
            "atomscan",
            "path",
            "/tmp/a",
            "--zip-password",
            "one",
            "--zip-password=two",
        ])?;

        assert_eq!(cli.zip_passwords, ["one", "two"]);
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
        assert!(Cli::try_parse_from(["scan", "-1", "-9", "/tmp/a"]).is_err());
        assert!(
            Cli::try_parse_from(["scan", "-9", "--threshold-hostile", "0.90", "/tmp/a"]).is_err()
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
    fn openrouter_alias_requires_model_and_key() -> Result<()> {
        let empty_home = tempfile::tempdir()?;
        with_isolated_llm_env(Some(empty_home.path()), || {
            let missing_key = Cli::try_parse_from([
                "atomscan",
                "--llm",
                "openrouter",
                "--llm-model",
                "qwen/qwen3.8-27b",
                "/tmp/a",
            ])?;
            let err = missing_key
                .interpret_config()
                .expect_err("openrouter without a key must fail");
            assert!(
                err.to_string().contains("~/.tok/openrouter"),
                "unexpected error: {err}"
            );

            let missing_model = Cli::try_parse_from([
                "atomscan",
                "--llm",
                "openrouter",
                "--llm-key",
                "sk-test",
                "/tmp/a",
            ])?;
            let err = missing_model
                .interpret_config()
                .expect_err("openrouter without a model must fail");
            assert!(
                err.to_string().contains("--llm-model"),
                "unexpected error: {err}"
            );

            let cli = Cli::try_parse_from([
                "atomscan",
                "--llm",
                "openrouter",
                "--llm-model",
                "qwen/qwen3.8-27b",
                "--llm-key",
                "sk-test",
                "/tmp/a",
            ])?;
            let cfg = cli
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
            assert_eq!(serve.llm.as_deref(), Some("openrouter"));
            let cfg = serve.interpret_config()?.context("serve inherits --llm")?;
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
            assert_eq!(worker.llm.as_deref(), Some("openrouter"));
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
            let cli = Cli::try_parse_from([
                "atomscan",
                "--llm",
                "openrouter",
                "--llm-model",
                "qwen/qwen3.8-27b",
                "/tmp/a",
            ])?;
            let cfg = cli
                .interpret_config()?
                .context("tok file should supply key")?;
            assert_eq!(cfg.api_key.as_deref(), Some("sk-from-file"));
            Ok(())
        })
    }
}
