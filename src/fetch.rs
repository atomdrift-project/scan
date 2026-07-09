//! Scan-side orchestration of [`fletch`]: discover the external references in
//! an analysis report, fetch them, and graft each retrieved payload back into
//! the report as a uniform file node — so the ML verdict and every downstream
//! consumer (hopper, prism) treat a fetched payload exactly like any other
//! analyzed file.
//!
//! cleave stays a pure offline analyzer; fletch is a pure find/fetch mechanism.
//! This module is the only place the two meet. For every file (root and archive
//! member) it runs fletch's *facts-based* discovery over the references, values,
//! and symbols cleave retained per file (`FilefactsView`) — declared
//! dependencies, value-driven hunts like npm lifecycle hooks, and module-load
//! calls (`require`/`import`/`__import__`) recovered from the retained AST
//! symbols. For the root sample, where the raw bytes are on disk, it
//! additionally runs the *text-based* hunt (`curl|sh`, `npm install` in a
//! `RUN`). It then retrieves the lot through the SSRF-guarded client and
//! re-analyzes what came back with cleave.
//!
//! The remaining gap: an archive member that is itself a shell script or
//! Dockerfile gets declared/value/symbol-based references only, not the
//! text-based command-stream hunt — that needs the member's bytes, which a
//! prior analysis extracted and discarded. The facts-only import hunt above
//! already covers the module-load vector (a member's `require("undeclared")`)
//! without those bytes.
//!
//! The fetch *edges* (`source_sha256 → content_sha256`) are returned as
//! [`FetchRecord`]s rather than embedded in any file: a fetch is a per-event
//! observation, not an intrinsic property of either file's bytes, so it never
//! falsely dedups when content is exploded by hash in hopper. The caller injects
//! them at report level.
//!
//! Off by default. Enabled, it is an online step performed after the offline
//! analysis; failures degrade gracefully to "no fetches".

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{OnceLock, PoisonError, RwLock};
use std::time::Duration;

use cleave::{AnalysisOptions, AnalysisReport, Finding};
use fletch::fetch::{
    BlobCache, FetchBudget, FetchRecord, HttpFetch, Outcome, fetch_ref, fetch_references_with,
};
use fletch::{RefKind, RefLocator, Reference, Registry, find};

use crate::analysis_cache::AnalysisCache;
use crate::deptree::{DepState, DepTree};

/// Default fetch recursion depth — the number of hops followed from the root.
/// `2` reaches a stage-3 payload (root → stage-2 → stage-3), since multi-stage
/// `curl | bash` droppers are the common case.
pub const DEFAULT_FETCH_DEPTH: u8 = 2;

/// Default age ceiling for fetching a declared dependency, in days. A version
/// older than this has had a long window for community discovery, so only its
/// registry metadata is looked up (a PURL lookup) and the expensive
/// fetch-and-scan is skipped; recent releases — where a supply-chain compromise
/// is freshest and least-vetted — are still pulled and fully scanned. Set to a
/// week: past that, a malicious release has almost always been caught and
/// yanked, and the byte scan's cost isn't worth it. `0` disables the gate.
pub const DEFAULT_MAX_DEP_AGE_DAYS: u32 = 7;

/// 1024-based size units, the basis for every `--fetch-max-*-size` ceiling.
const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// Default size ceiling for a single fetched artifact (`--fetch-max-size`) —
/// mirrors fletch's [`fletch::fetch::DEFAULT_MAX_FETCH_BYTES`] (256 MiB). A
/// response larger than this is abandoned, so one artifact can't dominate a run.
pub const DEFAULT_MAX_FETCH_SIZE: u64 = 256 * MIB;

/// Default ceiling on *live* fetches triggered by a single scanned file
/// (`--fetch-max-file-fetches`). Cache hits don't count, so a warm re-run is
/// never throttled; this bounds the fan-out one crafted file can trigger.
pub const DEFAULT_MAX_FILE_FETCHES: usize = 100;

/// Default ceiling on total bytes fetched on behalf of a single scanned file
/// (`--fetch-max-file-size`).
pub const DEFAULT_MAX_FILE_SIZE: u64 = 2 * GIB;

/// Default ceiling on *live* fetches across one whole execution
/// (`--fetch-max-total-fetches`). Lifted in long-lived server modes, where each
/// job is bounded by the per-file caps instead.
pub const DEFAULT_MAX_TOTAL_FETCHES: usize = 1000;

/// Default ceiling on total bytes fetched across one whole execution
/// (`--fetch-max-total-size`). Lifted in long-lived server modes.
pub const DEFAULT_MAX_TOTAL_SIZE: u64 = 10 * GIB;

/// Set the process-wide per-fetch byte ceiling (re-exported from [`fletch`] so
/// the binary configures it through `scan::fetch`). See
/// [`fletch::fetch::set_max_fetch_bytes`].
pub use fletch::fetch::set_max_fetch_bytes;

/// Override the process-wide mutable registry-metadata TTLs (re-exported from
/// [`fletch`]). `None` keeps the tiered defaults (4h for a pinned version's
/// packument, 1h for a `latest` lookup); a value collapses both to that
/// lifetime. The immutable tier (a released version's file list) is never
/// re-checked regardless. See [`fletch::fetch::set_registry_ttl`].
pub use fletch::fetch::set_registry_ttl;

/// Per-execution fetch budget — a running total shared by every [`orchestrate`]
/// call in the process. Scans run concurrently, so it's atomic. Set once at
/// startup via [`set_total_budget`] for one-shot runs; left at the unlimited
/// default in long-lived server modes, where each job is bounded by the per-file
/// caps instead. Each per-file [`FetchBudget`] is clamped to what remains here.
static TOTAL_FETCH_COUNT: AtomicUsize = AtomicUsize::new(usize::MAX);
static TOTAL_FETCH_BYTES: AtomicU64 = AtomicU64::new(u64::MAX);

/// Set the process-wide per-execution fetch ceiling (`--fetch-max-total-*`).
/// Called once at startup for one-shot scans; server modes leave it unlimited.
pub fn set_total_budget(max_fetches: usize, max_bytes: u64) {
    TOTAL_FETCH_COUNT.store(max_fetches, Ordering::Relaxed);
    TOTAL_FETCH_BYTES.store(max_bytes, Ordering::Relaxed);
}

/// Charge the per-execution budget for one file's live fetches, saturating at
/// zero so a charge never wraps. Cache hits and budget-skipped edges cost
/// nothing (the caller filters them out before charging).
fn charge_total_budget(fetches: usize, bytes: u64) {
    let _ = TOTAL_FETCH_COUNT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
        Some(n.saturating_sub(fetches))
    });
    let _ = TOTAL_FETCH_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
        Some(n.saturating_sub(bytes))
    });
}

/// Parse a byte size with an optional 1024-based unit suffix — `K`, `M`, `G`, or
/// `T`, case-insensitive, with an optional trailing `B` (`40M`, `40MB`, and
/// `40m` are equal). A bare number is bytes (`10240`). Powers every
/// `--fetch-max-*-size` flag, so an operator writes the natural unit and the
/// conversion happens once.
///
/// # Errors
/// Returns a human-readable message when the number is missing, unparseable, or
/// the result overflows `u64`.
pub fn parse_bytes(s: &str) -> Result<u64, String> {
    let lowered = s.trim().to_ascii_lowercase();
    // Drop an optional trailing `b` so `40mb` == `40m` == `40`, then peel a unit
    // suffix off the number — `strip_suffix` keeps us off byte indexing.
    let body = lowered.strip_suffix('b').unwrap_or(&lowered);
    let units = [('k', 1024_u64), ('m', MIB), ('g', GIB), ('t', 1024 * GIB)];
    let (number, mult) = units
        .iter()
        .find_map(|&(suffix, mult)| body.strip_suffix(suffix).map(|n| (n, mult)))
        .unwrap_or((body, 1));
    let n: u64 = number
        .trim()
        .parse()
        .map_err(|e| format!("invalid size {s:?}: {e} (examples: 40M, 2G, 10240)"))?;
    n.checked_mul(mult)
        .ok_or_else(|| format!("size {s:?} is too large"))
}

/// Parse a duration with an optional unit suffix — `s`, `m`, `h`, or `d`
/// (case-insensitive); a bare number is seconds (`90` == `90s`). The words
/// `never`/`inf`/`forever` mean "cache indefinitely" ([`Duration::MAX`]), for an
/// offline/air-gapped run that must not revalidate. Powers `--registry-ttl`.
///
/// # Errors
/// Returns a human-readable message when the number is missing, unparseable, or
/// the result overflows.
pub fn parse_duration(s: &str) -> Result<Duration, String> {
    let lowered = s.trim().to_ascii_lowercase();
    if matches!(lowered.as_str(), "never" | "inf" | "infinite" | "forever") {
        return Ok(Duration::MAX);
    }
    let units = [('s', 1_u64), ('m', 60), ('h', 3600), ('d', 86_400)];
    let (number, mult) = units
        .iter()
        .find_map(|&(suffix, mult)| lowered.strip_suffix(suffix).map(|n| (n, mult)))
        .unwrap_or((lowered.as_str(), 1));
    let n: u64 = number
        .trim()
        .parse()
        .map_err(|e| format!("invalid duration {s:?}: {e} (examples: 90s, 30m, 4h, 2d, never)"))?;
    n.checked_mul(mult)
        .map(Duration::from_secs)
        .ok_or_else(|| format!("duration {s:?} is too large"))
}

/// What to fetch — a selection of reference *kinds*, parsed from the
/// comma-separated `--fetch=KINDS` flag (`urls`, `packages`, `deps`) — plus how
/// many hops to follow. An empty kind selection (the default) disables fetching.
///
/// The three kinds map onto [`fletch`]'s [`RefKind`] taxonomy, so the selection
/// distinguishes how strongly a reference is bound to the artifact:
///
/// - `deps`     → [`RefKind::Dependency`]: a **strict dependency** declared in a
///   manifest or lockfile (`package.json`, `Cargo.lock`, `.SRCINFO depends`).
/// - `packages` → [`RefKind::Command`]: a **package merely mentioned** by an
///   install-command invocation (`npm install foo`, `pip install bar`,
///   `cargo install baz`) — typically injected by a build/lifecycle script
///   rather than pinned in a manifest.
/// - `urls`     → [`RefKind::UrlFetch`]: a **raw URL** with no package identity
///   (a `curl`/`wget` target, a staged download) — the largest exposure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FetchPolicy {
    /// Fetch raw `http(s)` URLs ([`RefKind::UrlFetch`]).
    pub urls: bool,
    /// Fetch packages named by an install command ([`RefKind::Command`]) — e.g.
    /// `npm install foo`. These are *mentioned*, not declared.
    pub packages: bool,
    /// Fetch strict declared dependencies ([`RefKind::Dependency`]) — manifest
    /// and lockfile entries.
    pub deps: bool,
    /// How many hops to follow: `1` fetches the references found in the root,
    /// `2` also follows references found *inside* those payloads, and so on.
    pub depth: u8,
    /// Skip fetching a declared dependency older than this many days, judged by
    /// the registry's publish date (looked up cheaply before the artifact is
    /// pulled). `0` disables the gate. Applies to declared dependencies only —
    /// URLs and command-mentioned packages are never age-gated, since their risk
    /// isn't tied to a registry release date.
    pub max_dep_age_days: u32,
    /// Ceiling on *live* fetches triggered by a single scanned file
    /// (`--fetch-max-file-fetches`). Cache hits are always served and never
    /// counted, so this caps only the cold-cache network fan-out, never a warm
    /// re-run. `0` disables fetching entirely.
    pub max_file_fetches: usize,
    /// Ceiling on total bytes fetched on behalf of a single scanned file
    /// (`--fetch-max-file-size`). The sweep stops once retrieved bytes cross it.
    pub max_file_bytes: u64,
    /// Skip fetching a *dependency* whose name pins it to a platform other than
    /// the host — the `@scope/pkg-<os>-<arch>` native-binary packages (biome,
    /// esbuild, swc, rollup, sharp…) that ship one prebuilt per platform. On a
    /// darwin-arm64 host only the darwin-arm64 variant is pulled; the linux and
    /// windows ones never run here, so scanning all of them multiplies the
    /// expensive binary analysis with no added coverage for this host. `false`
    /// pulls every platform (`--fetch-all-platforms`) — needed to audit the
    /// binaries that will run on *other* hosts (a CI image, a shipped release).
    /// Applies to fetched dependencies only; a directly-scanned artifact is
    /// always analyzed.
    pub host_platform_only: bool,
}

impl Default for FetchPolicy {
    fn default() -> Self {
        Self {
            urls: false,
            packages: false,
            deps: false,
            depth: DEFAULT_FETCH_DEPTH,
            max_dep_age_days: DEFAULT_MAX_DEP_AGE_DAYS,
            max_file_fetches: DEFAULT_MAX_FILE_FETCHES,
            max_file_bytes: DEFAULT_MAX_FILE_SIZE,
            host_platform_only: true,
        }
    }
}

impl FetchPolicy {
    /// True when at least one kind is selected — the master switch.
    #[must_use]
    pub(crate) const fn enabled(&self) -> bool {
        self.urls || self.packages || self.deps
    }

    /// Whether `kind` is selected by this policy. References whose kind is
    /// neither a URL, a command-mentioned package, nor a declared dependency
    /// (e.g. [`RefKind::Repository`] identity) are never fetched.
    #[must_use]
    fn wants(&self, kind: RefKind) -> bool {
        match kind {
            RefKind::UrlFetch => self.urls,
            RefKind::Command => self.packages,
            RefKind::Dependency => self.deps,
            _ => false,
        }
    }
}

impl std::str::FromStr for FetchPolicy {
    type Err = String;

    /// Parse a comma-separated kind list (`urls`, `packages`, `deps`, or `all`).
    /// `all` selects every kind. Empty entries are ignored; an unknown kind or an
    /// empty selection is an error so a typo is never silently a no-op.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        const VALID: &str = "valid: all, urls, packages, deps";
        let mut policy = Self::default();
        for kind in s.split(',') {
            match kind.trim() {
                "" => {}
                "all" => {
                    policy.urls = true;
                    policy.packages = true;
                    policy.deps = true;
                }
                "urls" => policy.urls = true,
                "packages" => policy.packages = true,
                "deps" => policy.deps = true,
                other => {
                    return Err(format!("unknown fetch kind {other:?} ({VALID})"));
                }
            }
        }
        if !policy.enabled() {
            return Err(format!("empty fetch selection ({VALID})"));
        }
        Ok(policy)
    }
}

/// Recognised npm platform tokens (`process.platform` / `process.arch`), so a
/// match keys on a genuine `<os>-<arch>` native-binary name rather than an
/// incidental word. Kept to the pairs that ship prebuilt per-platform binaries.
const NPM_OS: &[&str] = &[
    "darwin", "linux", "win32", "freebsd", "openbsd", "netbsd", "sunos", "android", "aix",
];
const NPM_ARCH: &[&str] = &[
    "x64", "arm64", "ia32", "arm", "ppc64", "s390x", "riscv64", "loong64", "mips64el",
];

/// The host's npm-style `(os, arch)` tokens, mapped from Rust's target
/// constants. An unmapped target yields an empty token, which disables that
/// half of the platform match — fail open, so a dependency is never skipped on a
/// host we can't confidently name.
fn host_platform() -> (&'static str, &'static str) {
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        "solaris" | "illumos" => "sunos",
        os @ ("linux" | "freebsd" | "openbsd" | "netbsd" | "android") => os,
        _ => "",
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        "x86" => "ia32",
        arch @ ("arm" | "ppc64" | "s390x" | "riscv64") => arch,
        _ => "",
    };
    (os, arch)
}

/// Whether a dependency reference is a native-binary package pinned to a
/// platform other than `host`. Matches the well-known `<os>-<arch>` (or
/// `<arch>-<os>`) adjacent-segment convention native packages use —
/// `cli-darwin-arm64`, `rollup-linux-x64-gnu`, `@img/sharp-win32-x64` — so a
/// package that names no such pair is treated as portable and kept. Returns
/// `false` when the host platform can't be named (fail open) or for non-PURL
/// locators (a raw URL carries no package identity to place).
fn off_host_platform(r: &Reference, host: (&str, &str)) -> bool {
    let (host_os, host_arch) = host;
    if host_os.is_empty() || host_arch.is_empty() {
        return false;
    }
    let RefLocator::Purl(purl) = &r.locator else {
        return false;
    };
    // Package name = the PURL body without its trailing `@version`; split into
    // lowercase alphanumeric segments (`cli-darwin-arm64` → [cli, darwin, arm64],
    // `%40biomejs` → [40, biomejs]).
    let name = purl.rsplit_once('@').map_or(purl.as_str(), |(n, _)| n);
    let segs: Vec<String> = name
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase)
        .collect();
    // An adjacent os+arch pair (either order) marks a platform-specific package;
    // skip it when either token disagrees with the host.
    for w in segs.windows(2) {
        let (os, arch) = if NPM_OS.contains(&w[0].as_str()) && NPM_ARCH.contains(&w[1].as_str()) {
            (&w[0], &w[1])
        } else if NPM_ARCH.contains(&w[0].as_str()) && NPM_OS.contains(&w[1].as_str()) {
            (&w[1], &w[0])
        } else {
            continue;
        };
        return os != host_os || arch != host_arch;
    }
    false
}

/// The root sample's imperative hunt re-reads it from disk and re-parses it.
/// Skip that for large roots — the win is scripts/manifests/Dockerfiles, which
/// are small; a multi-megabyte binary root has no imperative install commands to
/// find and would just pay a wasted parse. Declared references are read from the
/// report regardless, so nothing is lost for large roots.
const ROOT_HUNT_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// The HTTP client and blob cache, built once per process and shared across
/// every analyzed file (and rayon worker). Fetch is opt-in, so this is
/// initialized lazily on the first fetching analysis. `None` means the client
/// or cache couldn't be created — fetching degrades to a no-op.
struct Resources {
    net: HttpFetch,
    cache: BlobCache,
}

fn shared_resources() -> Option<&'static Resources> {
    static RESOURCES: OnceLock<Option<Resources>> = OnceLock::new();
    RESOURCES
        .get_or_init(|| match (HttpFetch::new(), BlobCache::open()) {
            (Ok(net), Ok(cache)) => Some(Resources { net, cache }),
            (Err(e), _) => {
                tracing::warn!("fetch disabled: http client unavailable: {e:#}");
                None
            }
            (_, Err(e)) => {
                tracing::warn!("fetch disabled: blob cache unavailable: {e:#}");
                None
            }
        })
        .as_ref()
}

/// A fetched dependency captured for upload to hopper as its own sample. Carries
/// the standalone analysis report cleave produced for the dependency's bytes (the
/// same report a first-hand `pkg:`/`url` scan yields, stripped and compacted),
/// plus the shas of every file in it — so the caller can harvest the dependency's
/// aggregate verdict from the embedded-classification pass it already runs over
/// the merged report, without re-running the model.
pub(crate) struct FetchedDependency {
    /// The reference locator (PURL or URL) the bytes were fetched from.
    pub locator: String,
    /// The URL the locator resolved to — drives the stored filename/type sniff.
    pub url: String,
    /// SHA-256 of the fetched bytes — the dependency's identity in hopper.
    pub content_sha: String,
    /// Size of the fetched bytes, recorded in the provenance sidecar.
    pub size: u64,
    /// The dependency's own compact cleave report as **JSON text** (`raw` for
    /// its `/api/result`). Text form on purpose: a `serde_json::Value` tree
    /// costs 3-6x the text size, and up to eight jobs' dependencies are
    /// co-resident in a worker from graft until their result POST — measured
    /// ~800 MB of retained `Value`s on the realworld worker benchmark. The
    /// envelope build parses it back transiently.
    pub raw: String,
    /// Every file sha in the report, so the caller can attribute the embedded
    /// pass's per-node decisions back to this dependency.
    pub member_shas: Vec<String>,
}

/// Discover, fetch, and graft, following references up to `policy.depth` hops.
/// Mutates `report.files` in place with one node per fetched payload (and any
/// extracted members) and returns the fetch edge log plus the standalone report
/// captured for each fetched dependency. A disabled policy, an unavailable
/// cache/client, or zero references all yield empty logs.
pub(crate) fn orchestrate(
    report: &mut AnalysisReport,
    root_path: &Path,
    policy: FetchPolicy,
    progress: bool,
) -> (Vec<FetchRecord>, Vec<FetchedDependency>) {
    if !policy.enabled() {
        return (Vec::new(), Vec::new());
    }
    let Some(res) = shared_resources() else {
        return (Vec::new(), Vec::new());
    };

    // Analyze fetched payloads with the same bloom short-circuit the top-level
    // scan uses, so a trusted binary shipped inside a dependency isn't needlessly
    // re-disassembled; and memoize the whole analysis by content sha, so a warm
    // re-run reuses it rather than repeating a minutes-long pass.
    let opts = AnalysisOptions {
        skip_predicate: dep_skip_predicate(),
        ..AnalysisOptions::default()
    };
    // Opened lazily on the first payload actually analyzed. Opening it derives
    // the ruleset-version namespace, which calls `cleave::version_info` — and that
    // spins up the YARA engine just to count rules. A scan that fetches nothing
    // (every reference age-gated or none present, the common `pkg:` case) must
    // not pay that: `None` here means "not yet opened".
    let mut acache: Option<Option<AnalysisCache>> = None;
    let mut records = Vec::new();
    // Standalone reports for each fetched dependency, captured before the payload
    // is grafted into the merged report. Uploaded to hopper as their own samples.
    let mut dependencies: Vec<FetchedDependency> = Vec::new();
    // Two budget tiers. Per scanned file: each file's references get a fresh
    // ceiling (`--fetch-max-file-fetches` live fetches, `--fetch-max-file-size`
    // bytes), so one file can't starve the rest. Per execution: a process-wide
    // running total (`--fetch-max-total-*`, lifted in server modes) shared across
    // every file scanned, so a crafted corpus can't multiply the per-file cap
    // into a fetch storm. Each per-file budget below is clamped to what the total
    // budget still allows, and every live fetch is charged against it. Cache hits
    // are free and uncounted (a warm re-run is never throttled); refs past a cap
    // become `BudgetExceeded`, never silently dropped.
    // Loop guard: a locator is fetched at most once per run, so a chain that
    // points back at an earlier stage can't cycle.
    let mut seen: HashSet<String> = HashSet::new();

    // Where the fetch phase's progress goes: the live in-place dependency tree on
    // an interactive single-artifact scan, the streamed log above any active
    // scan bar otherwise, or nothing for machine output. Created once and shared
    // across every hop, so transitive dependencies join the same view.
    let reporter = Reporter::new(progress);

    // Hop 0's work-list: declared references from every file in the report plus
    // fletch's imperative discovery over the root sample's bytes. Each later hop
    // works from the references found *inside* the previous hop's payloads.
    let mut worklist = collect_references(report, root_path);
    // Phantom-dependency signal: a package imperatively installed or loaded
    // somewhere in this artifact but absent from its manifest's declared deps —
    // a covertly-installed companion or a dependency-confusion target. Computed
    // across the whole work-list so a member's `require("x")` is diffed against
    // the root manifest's declarations.
    let all_refs: Vec<Reference> = worklist
        .iter()
        .flat_map(|(_, refs)| refs.iter().cloned())
        .collect();
    // Surfaced at debug, not warn: this is only meaningful when a manifest is
    // present to diff against. Scanning loose files (no manifest) flags every
    // imperative import as "undeclared", so emitting it by default is noise.
    // `--verbose` (scan=debug) still exposes it for investigation.
    for u in find::undeclared_packages(&all_refs) {
        tracing::debug!(
            package = %locator_key(u),
            source = %u.source,
            "undeclared dependency: imperatively acquired but not declared in manifest"
        );
    }
    // One wall-clock reading for the whole run, so every dependency's age is
    // judged against the same instant.
    let now = unix_now();
    // The host platform, for filtering off-host native-binary dependencies.
    // Sampled once — it can't change mid-run.
    let host = host_platform();
    for _hop in 0..policy.depth {
        if worklist.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for (source_sha, refs) in std::mem::take(&mut worklist) {
            // Keep only the kinds this policy selected — by RefKind, so a
            // command-mentioned package (`packages`) is distinct from a declared
            // dependency (`deps`) even though both are PURLs — and that haven't
            // been fetched yet this run.
            let selected: Vec<Reference> = refs
                .into_iter()
                .filter(|r| policy.wants(r.kind))
                // Drop native-binary dependencies built for another platform
                // before they're ever fetched — the host variant is scanned, its
                // linux/windows siblings never run here. Off unless the policy
                // asks for it (`--fetch-all-platforms` audits every platform).
                .filter(|r| {
                    let off_host = policy.host_platform_only && off_host_platform(r, host);
                    if off_host {
                        tracing::debug!(
                            package = %locator_key(r),
                            host_os = host.0,
                            host_arch = host.1,
                            "dependency pinned to another platform; skipped (--fetch-all-platforms to include)"
                        );
                    }
                    !off_host
                })
                .filter(|r| seen.insert(locator_key(r)))
                .collect();
            if selected.is_empty() {
                continue;
            }
            // Look up each declared dependency's registry metadata first. Every
            // resolved record is materialized as a `*.registry.json` node so its
            // facts are trait-matched, and releases older than the age ceiling
            // are dropped before the expensive fetch+scan of their bytes. Skips
            // are reported, never silent.
            let (selected, registries) = age_gate(selected, &policy, res, now);
            // Reveal the kept (to-fetch) set as pending, so the tree shows the
            // dependencies it will actually scan up front. Aged-out deps are
            // deliberately never announced — for a large npm graph they are the
            // overwhelming majority and only a registry-metadata lookup runs on
            // them, so listing them would bury the handful of live scans.
            reporter.announce(&selected);
            // Registry findings keyed by locator, captured as each record is
            // materialized so the package pass below can pair an artifact with
            // its own registry metadata (see `apply_package_composites`).
            let mut registry_findings: HashMap<String, Vec<Finding>> = HashMap::new();
            for (r, reg, skip) in &registries {
                if let Some(sub) = registry_node(reg, &opts) {
                    registry_findings
                        .entry(locator_key(r))
                        .or_default()
                        .extend(sub_findings(&sub));
                    merge_registry(report, &source_sha, sub);
                }
                // The record is materialized either way; only the artifact fetch
                // is skipped. `None` = kept for fetch+scan.
                let Some(reason) = skip else {
                    continue;
                };
                let common = tracing::field::display(locator_key(r));
                // A known-good skip is counted into the bloom tally so the summary
                // reflects it alongside skipped known-good files.
                if *reason == SkipReason::KnownGood {
                    crate::bloom_repo::record(crate::bloom_repo::Decision::Skip, false);
                }
                let log_reason = match reason {
                    SkipReason::Removed => "version removed from registry",
                    SkipReason::AgedOut => "older than --max-dep-age",
                    SkipReason::KnownGood => "known-good (bloom); trust not stale",
                };
                // Age-outs are the common, expected case; they stay at debug so
                // `--verbose` can still see them, while removals and known-good
                // skips — the interesting decisions — are surfaced at info.
                match reason {
                    SkipReason::AgedOut => tracing::debug!(
                        package = %common,
                        ecosystem = %reg.ecosystem,
                        version = %reg.version,
                        age_days = reg.age_days.unwrap_or(0),
                        downloads = reg.downloads_recent.or(reg.downloads_total),
                        reason = log_reason,
                        "registry record materialized; artifact fetch skipped"
                    ),
                    SkipReason::Removed | SkipReason::KnownGood => tracing::info!(
                        package = %common,
                        ecosystem = %reg.ecosystem,
                        version = %reg.version,
                        age_days = reg.age_days.unwrap_or(0),
                        downloads = reg.downloads_recent.or(reg.downloads_total),
                        reason = log_reason,
                        "registry record materialized; artifact fetch skipped"
                    ),
                }
                // Settle the skipped row. The tree shows every reason (so no row
                // is left hanging as pending); the stream keeps flooding-averse
                // behaviour, printing only the surfaced removals/known-good skips.
                reporter.skipped(r, reg, now, *reason);
            }
            if selected.is_empty() {
                continue;
            }
            // This file's ceiling, clamped to what the per-execution budget still
            // allows. The total budget is a process global, so concurrent scans
            // share one running total.
            let budget = FetchBudget {
                max_count: policy
                    .max_file_fetches
                    .min(TOTAL_FETCH_COUNT.load(Ordering::Relaxed)),
                max_bytes: policy
                    .max_file_bytes
                    .min(TOTAL_FETCH_BYTES.load(Ordering::Relaxed)),
            };
            // Mark the to-fetch set in flight, then fetch. The callback fires as
            // each download lands (from a pool worker, so it's `Sync`), flipping
            // that row to "analyzing" the moment its bytes arrive rather than when
            // the whole concurrent batch returns — keyed on the original
            // reference, since a versionless locator may be refined during fetch.
            reporter.fetching(&selected);
            let on_fetched = |r: &Reference, rec: &FetchRecord| reporter.landed(r, rec);
            let fetched = fetch_references_with(
                &selected,
                &source_sha,
                policy.urls,
                &res.net,
                &res.cache,
                budget,
                &on_fetched,
            );
            // Charge the per-execution budget for what this file fetched live.
            // Only live fetches count (cache hits are free); a `BudgetExceeded`
            // edge consumes nothing.
            let spent = fetched
                .iter()
                .filter(|r| fletch::fetch::counts_against_budget(r))
                .count();
            let bytes: u64 = fetched.iter().filter_map(|r| r.size).sum();
            charge_total_budget(spent, bytes);

            // Authoritative pass over every returned edge: settle each row (the
            // tree finalizes any budget-clipped edge the live callback never saw;
            // re-settling a callback-landed row is idempotent) and print the
            // streamed line. `selected` and `fetched` align one-to-one and in
            // order — every selected reference is a fetch target, so fletch emits
            // exactly one record per reference in declaration order.
            for (r, rec) in selected.iter().zip(&fetched) {
                reporter.landed(r, rec);
                reporter.report(rec);
            }

            // Analyze the payloads concurrently (bounded), then merge each into
            // the report serially in fetch order so file ids stay deterministic.
            // The analysis — a full cleave pass per payload — is the real cost
            // here; the fetch is near-free on a cache hit. The callback settles
            // each row from "analyzing" to its final glyph as its scan finishes.
            let on_analyzed = |i: usize| {
                if let (Some(r), Some(rec)) = (selected.get(i), fetched.get(i)) {
                    reporter.analyzed(r, rec);
                }
            };
            let acache_ref = acache
                .get_or_insert_with(crate::analysis_cache::AnalysisCache::open)
                .as_ref();
            let analyzed = analyze_payloads(&fetched, &res.cache, &opts, acache_ref, &on_analyzed);
            for (rec, payload) in fetched.iter().zip(analyzed) {
                if let Some(payload) = payload {
                    // Capture the artifact's findings and identity before
                    // `merge_payload` consumes the sub-report, then graft any
                    // package-scoped composite that correlates them with this
                    // package's registry metadata.
                    let artifact = payload.sub.as_ref().map(sub_findings).unwrap_or_default();
                    let artifact_sha = payload.content_sha.clone();
                    // Capture the dependency's standalone report before merge_payload
                    // consumes the sub-report into the merged tree.
                    if let Some(dep) = capture_dependency(rec, &payload) {
                        dependencies.push(dep);
                    }
                    next.extend(merge_payload(report, rec, payload));
                    // `rec.locator` is the original filefacts locator (the PURL),
                    // the same key the registry findings were captured under.
                    if let Some(reg) = registry_findings.get(rec.locator.as_str()) {
                        apply_package_composites(report, &artifact_sha, &artifact, reg, &opts);
                    }
                }
            }
            records.extend(fetched);
        }
        worklist = next;
    }
    reporter.finish(&records);
    (records, dependencies)
}

/// Where the fetch phase's progress is surfaced.
///
/// `Off` — machine output (JSON/tiny/server): nothing is printed; the edges ride
/// the report. `Stream` — the append-only fetch log, printed above any active
/// scan progress bar (a multi-file scan, a pipeline); reveals each reference
/// lazily as it completes. `Tree` — the live, in-place dependency tree that
/// takes over stderr for an interactive single-artifact scan (see
/// [`crate::deptree`]), listing the whole known set up front and animating each
/// row through its lifecycle.
///
/// The methods take `&self` (the stream's header latch is atomic) so the fetch
/// completion callback — invoked concurrently from fletch's pool — can share one
/// reporter with the sequential orchestration.
enum Reporter {
    Off,
    Stream { header: AtomicBool },
    Tree(DepTree),
}

impl Reporter {
    /// Choose a channel: the live tree when it can own the terminal, else the
    /// stream when progress is requested, else off.
    fn new(progress: bool) -> Self {
        if !progress {
            return Self::Off;
        }
        DepTree::activate().map_or_else(
            || Self::Stream {
                header: AtomicBool::new(false),
            },
            Self::Tree,
        )
    }

    /// Reveal a hop's references as pending (tree only) so the whole known set is
    /// visible before any network work begins.
    fn announce(&self, refs: &[Reference]) {
        if let Self::Tree(tree) = self {
            for r in refs {
                tree.add(&locator_key(r), &dep_display_name(r));
            }
        }
    }

    /// Mark the to-fetch set in flight (tree only).
    fn fetching(&self, refs: &[Reference]) {
        if let Self::Tree(tree) = self {
            for r in refs {
                tree.set(&locator_key(r), DepState::Fetching);
            }
        }
    }

    /// A fetch landed: move its row to "analyzing" (bytes in hand, scan pending)
    /// or settle it (skipped/failed/budget). Tree only — keyed on the original
    /// reference, so a locator refined during fetch still matches the row. Called
    /// live per completion and again authoritatively after the batch; both are
    /// idempotent.
    fn landed(&self, r: &Reference, rec: &FetchRecord) {
        if let Self::Tree(tree) = self {
            tree.set(&locator_key(r), landed_state(rec));
        }
    }

    /// Print the streamed fetch line (stream only); the tree already moved this
    /// row in [`Reporter::landed`].
    fn report(&self, rec: &FetchRecord) {
        if let Self::Stream { header } = self {
            crate::engine::print_above_bar(|| {
                fetch_header(header);
                report_fetch(rec);
            });
        }
    }

    /// A payload finished analysis: settle its row to the final fetch glyph
    /// (tree only).
    fn analyzed(&self, r: &Reference, rec: &FetchRecord) {
        if let Self::Tree(tree) = self {
            tree.set(&locator_key(r), done_state(rec));
        }
    }

    /// A dependency was skipped at the age gate. Only the *meaningful* skips —
    /// a withdrawn version, or a known-good coordinate — are surfaced; an aged-out
    /// dep (the common case, only a metadata lookup ran) is dropped entirely, in
    /// both the stream and the tree. The tree wasn't told about aged-outs
    /// (`announce` sees only the kept set), so it adds a row here for the skips it
    /// does surface.
    fn skipped(&self, r: &Reference, reg: &Registry, now: u64, reason: SkipReason) {
        if matches!(reason, SkipReason::AgedOut) {
            return;
        }
        match self {
            Self::Off => {}
            Self::Stream { header } => crate::engine::print_above_bar(|| {
                fetch_header(header);
                report_skip(r, reg, now, reason);
            }),
            Self::Tree(tree) => {
                tree.add(&locator_key(r), &dep_display_name(r));
                tree.set(&locator_key(r), skip_state(reg, now, reason));
            }
        }
    }

    /// Close out the phase: the stream prints its one-line tally (only if it ever
    /// printed a row); the tree settles and prints the same tally beneath the
    /// dependency rows.
    fn finish(&self, records: &[FetchRecord]) {
        match self {
            Self::Off => {}
            Self::Stream { header } => {
                if header.load(Ordering::Relaxed) {
                    report_summary(records);
                }
            }
            Self::Tree(tree) => tree.finish(&summary_line(records)),
        }
    }
}

/// Emit the streamed log's lazy header once, before its first row.
fn fetch_header(header: &AtomicBool) {
    if !header.swap(true, Ordering::Relaxed) {
        eprintln!(
            "\n  \x1b[38;2;100;180;255m\u{2b07}\x1b[0m  \x1b[38;2;160;160;160mfetching external references\x1b[0m"
        );
    }
}

/// A compact, human display name for a reference: `name version` for a PURL
/// (scope preserved, e.g. `@biomejs/cli-darwin-arm64 2.5.0`), or the URL with
/// its scheme trimmed. This is what the tree shows in place of the full registry
/// URL the streamed log prints.
fn dep_display_name(r: &Reference) -> String {
    match &r.locator {
        RefLocator::Purl(p) => purl_display(p),
        RefLocator::Url(u) | RefLocator::Path(u) => u
            .strip_prefix("https://")
            .or_else(|| u.strip_prefix("http://"))
            .unwrap_or(u)
            .to_string(),
    }
}

/// Render a PURL as `name version`. `pkg:npm/%40scope/pkg@1.2.3` becomes
/// `@scope/pkg 1.2.3`; a versionless coordinate shows just the name. Falls back
/// to the raw PURL for anything that doesn't parse.
fn purl_display(purl: &str) -> String {
    let body = purl.strip_prefix("pkg:").unwrap_or(purl);
    let Some((_ecosystem, rest)) = body.split_once('/') else {
        return purl.to_string();
    };
    // The version follows the last '@'; a scope's '@' is `%40`-encoded, so a
    // literal '@' only ever separates the version.
    let (name, version) = rest.rsplit_once('@').unwrap_or((rest, ""));
    let name = name.replace("%40", "@");
    let version = version.split(['?', '#']).next().unwrap_or(version);
    if version.is_empty() {
        name
    } else {
        format!("{name} {version}")
    }
}

/// The tree state for a fetch the moment it lands: "analyzing" when bytes are in
/// hand and a scan will follow (an `Ok` or a pin mismatch — the mismatch settles
/// to its own glyph once analyzed), else the settled fetch glyph.
fn landed_state(rec: &FetchRecord) -> DepState {
    if matches!(rec.outcome, Outcome::Ok | Outcome::PinMismatch) {
        DepState::Analyzing
    } else {
        done_state(rec)
    }
}

/// The settled tree state for a fetch: the shared [`fetch_row`] glyph/colour,
/// with the detail column (a size, or a failure note) as its trailing text.
fn done_state(rec: &FetchRecord) -> DepState {
    let (glyph, _label, r, g, b, detail) = fetch_row(rec);
    DepState::Done {
        glyph,
        color: (r, g, b),
        detail: detail.unwrap_or_else(|| rec.size.map_or(String::new(), human_bytes)),
    }
}

/// The settled tree state for an age-gate skip, mirroring [`report_skip`]'s
/// glyph and colour with a concise reason as the detail.
fn skip_state(reg: &Registry, now: u64, reason: SkipReason) -> DepState {
    let age_days = reg.age_secs(now).unwrap_or(0) / 86_400;
    let (glyph, color, detail) = match reason {
        SkipReason::KnownGood => ('\u{2713}', (80, 200, 80), "known-good".to_string()),
        SkipReason::Removed => ('\u{00b7}', (120, 120, 120), "removed".to_string()),
        SkipReason::AgedOut => ('\u{00b7}', (120, 120, 120), format!("{age_days}d old")),
    };
    DepState::Done {
        glyph,
        color,
        detail,
    }
}

/// Capture a fetched payload's standalone report for upload as its own hopper
/// sample. The report is the pristine one cleave produced for the dependency's
/// own bytes (container at depth 0, correct member structure) — so it needs no
/// rerooting, unlike reconstructing a subtree out of the merged parent report.
/// Compacted from a borrow: no clone of the report, and no strip pass (the raw is
/// never fed to a model here, and a single dependency never nears the body
/// limit). Returns `None` when there is nothing to upload.
fn capture_dependency(rec: &FetchRecord, analyzed: &Analyzed) -> Option<FetchedDependency> {
    let sub = analyzed.sub.as_ref()?;
    if analyzed.content_sha.is_empty() {
        return None;
    }
    let member_shas: Vec<String> = sub.files.iter().map(|f| f.sha256.clone()).collect();
    let compact = cleave::types::compact::compact_from_files(&sub.files);
    let raw = serde_json::to_string(&compact).ok()?;
    let url = rec
        .final_url
        .clone()
        .unwrap_or_else(|| rec.resolved_url.clone());
    Some(FetchedDependency {
        locator: rec.locator.clone(),
        url,
        content_sha: analyzed.content_sha.clone(),
        size: rec.size.unwrap_or(0),
        raw,
        member_shas,
    })
}

/// Fetch a single external reference — a `pkg:` PURL or a URL — and return its
/// bytes, a filename for cleave's type detection, and the fetch record. Powers
/// the `pkg`/`url` subcommands: one artifact, pulled and handed to the scanner.
/// On a terminal (`progress`), logs the live/cache outcome and resolved URL,
/// matching `--fetch`. Errors if the client/cache is unavailable or nothing was
/// retrieved (unresolved, failed, skipped).
pub fn fetch_one(
    locator: RefLocator,
    progress: bool,
) -> anyhow::Result<(Vec<u8>, String, FetchRecord)> {
    let Some(res) = shared_resources() else {
        anyhow::bail!("fetch unavailable: HTTP client or blob cache could not be initialized");
    };
    let kind = match &locator {
        RefLocator::Purl(_) => RefKind::Dependency,
        RefLocator::Url(_) => RefKind::UrlFetch,
        RefLocator::Path(_) => RefKind::Local,
    };
    let reference = Reference {
        locator,
        kind,
        source: "cli".to_string(),
        evidence: String::new(),
        offset: 0,
        pinned_hash: None,
        content_sha256: None,
    };
    let rec = fetch_ref(&reference, &res.net, &res.cache);
    if progress {
        eprintln!(
            "\n  \x1b[38;2;100;180;255m\u{2b07}\x1b[0m  \x1b[38;2;160;160;160mfetching\x1b[0m"
        );
        report_fetch(&rec);
    }
    if !matches!(rec.outcome, Outcome::Ok | Outcome::PinMismatch) {
        let target = if rec.resolved_url.is_empty() {
            rec.locator.as_str()
        } else {
            rec.resolved_url.as_str()
        };
        anyhow::bail!("fetch retrieved nothing for {target}: {:?}", rec.outcome);
    }
    let bytes = res
        .cache
        .load(&rec.locator)
        .ok_or_else(|| anyhow::anyhow!("fetched content for {} not in cache", rec.locator))?;
    let name = payload_name(&rec);
    Ok((bytes, name, rec))
}

/// Look up the normalized registry metadata for a one-shot `pkg`/`url` target,
/// using the shared fetch resources, with its relative age stamped. `None` for a
/// raw URL, an unsupported ecosystem, or an unreachable registry — the metadata
/// document is cached, so a later fetch of the same package pays nothing for it.
#[must_use]
pub fn registry(locator: &RefLocator) -> Option<Registry> {
    let res = shared_resources()?;
    fletch::registry(locator, &res.net, &res.cache).map(|reg| reg.with_age(unix_now()))
}

/// Serialize a registry record to its `*.registry.json` document — its synthetic
/// name and bytes — so the one-shot `pkg:`/`url` path can scan the registry
/// metadata directly when the artifact itself can't be fetched (e.g. the version
/// was unpublished). `None` if it can't be serialized.
#[must_use]
pub fn registry_document(reg: &Registry) -> Option<(String, Vec<u8>)> {
    Some((registry_doc_name(reg), serde_json::to_vec(reg).ok()?))
}

/// Like [`registry`], but also returns the raw provider documents the lookup
/// read — recovered from the cache the scan already populated, so it costs no
/// extra fetch. Used by `--upload` to archive the raw registry snapshot in hopper
/// alongside the normalized record, the same re-parsing backup forager stores.
#[must_use]
pub fn registry_with_sources(
    locator: &RefLocator,
) -> (Option<Registry>, Vec<fletch::fetch::RecordedSource>) {
    let Some(res) = shared_resources() else {
        return (None, Vec::new());
    };
    let (record, sources) = fletch::registry_with_sources(locator, &res.net, &res.cache);
    (record.map(|reg| reg.with_age(unix_now())), sources)
}

/// One-shot `pkg:`/`url`: graft the root artifact's own registry metadata into
/// its finalized report as a child node of the root, then run the package pass.
///
/// The registry record is materialized as a `*.registry.json` node — detected as
/// the `registry` filetype, carrying `registry.*` facts — and merged under the
/// root, exactly as the `--fetch` path grafts a dependency's registry beside the
/// dependency. So the registry metadata becomes a real layer of the analyzed
/// package: it is trait-matched, featurized, and trained on like any other file,
/// rather than living in a disconnected side report. With both halves now in one
/// tree, [`apply_package_composites`] correlates the artifact's behavior with the
/// registry's account of it. A no-op if the report has no root or the record
/// can't be analyzed.
pub(crate) fn graft_root_registry(report: &mut AnalysisReport, reg: &Registry) {
    let Some(root_sha) = report.files.first().map(|f| f.sha256.clone()) else {
        return;
    };
    let opts = AnalysisOptions::default();
    let Some(sub) = registry_node(reg, &opts) else {
        return;
    };
    // The artifact's own findings (before the registry node joins the tree) and
    // the registry node's findings — the two halves of the package pass.
    let artifact = sub_findings(report);
    let registry = sub_findings(&sub);
    merge_registry(report, &root_sha, sub);
    apply_package_composites(report, &root_sha, &artifact, &registry, &opts);
}

/// Print and log a package's normalized registry metadata for the one-shot
/// `pkg`/`url` scan path, so the operator sees the registry's own account of an
/// artifact (age, author, popularity, deprecation) beside the scan of its bytes.
pub fn report_registry(reg: &Registry, progress: bool) {
    tracing::info!(
        ecosystem = %reg.ecosystem,
        package = %reg.name,
        version = %reg.version,
        age_days = reg.age_days,
        author = reg.author.as_deref(),
        downloads = reg.downloads_recent.or(reg.downloads_total),
        deprecated = reg.deprecated.as_deref(),
        "package registry metadata"
    );
    if !progress {
        return;
    }
    let mut parts: Vec<String> = Vec::new();
    if !reg.version.is_empty() {
        parts.push(format!("v{}", reg.version.trim_start_matches('v')));
    }
    if let Some(d) = reg.age_days {
        parts.push(format!("{d}d old"));
    }
    if let Some(a) = &reg.author {
        parts.push(format!("by {a}"));
    }
    if let Some(d) = reg.downloads_recent.or(reg.downloads_total) {
        parts.push(format!("{d} dl"));
    }
    if let Some(r) = reg.rating {
        parts.push(format!("\u{2605}{r:.1}"));
    }
    if let Some(l) = &reg.license {
        parts.push(l.clone());
    }
    if let Some(dep) = &reg.deprecated {
        parts.push(format!("\u{26a0} {dep}"));
    }
    eprintln!(
        "\n  \x1b[38;2;180;160;255m\u{24d8}\x1b[0m  \x1b[38;2;160;160;160mregistry\x1b[0m \x1b[2m{}\x1b[0m",
        reg.ecosystem
    );
    if !parts.is_empty() {
        eprintln!(
            "    \x1b[38;2;130;130;130m{}\x1b[0m",
            parts.join("  \u{00b7}  ")
        );
    }
}

/// Render one fetched reference to stderr, distinguishing a live network fetch
/// from a cache hit and naming the actual URL that was (or would be) retrieved.
/// Only the interactive terminal path passes `progress`; JSON/server callers
/// stay silent. Colors mirror the scan progress bar's truecolor palette.
fn report_fetch(rec: &FetchRecord) {
    let (glyph, label, r, g, b, detail) = fetch_row(rec);

    // The actual URL fetched; fall back to the bare locator (PURL) when the
    // reference never resolved to one.
    let url = if rec.resolved_url.is_empty() {
        rec.locator.as_str()
    } else {
        rec.resolved_url.as_str()
    };
    // A redirect lands the payload elsewhere — name the host, dimmed. Only the
    // host: a release-asset redirect carries a time-limited SAS token and JWT in
    // its query, which are noise on the line and a credential better kept out of
    // terminals and logs.
    let redirect = match &rec.final_url {
        Some(f) if f != &rec.resolved_url => {
            format!("  \x1b[2m\u{2192} {}\x1b[0m", url_host(f))
        }
        _ => String::new(),
    };
    let column = detail.unwrap_or_else(|| rec.size.map_or(String::new(), human_bytes));

    eprintln!(
        "    \x1b[38;2;{r};{g};{b}m{glyph} {label:<6}\x1b[0m \x1b[38;2;130;130;130m{column:>10}\x1b[0m  {url}{redirect}"
    );
}

/// The host of a URL — scheme, path, and query stripped — for a compact redirect
/// note (`\u{2192} cdn.example.com`). Anything after the authority is dropped, so
/// a signed CDN URL's SAS token / JWT never reaches the line.
fn url_host(url: &str) -> &str {
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    // Keep `host[:port]`, dropping any `userinfo@` prefix.
    authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host)
}

/// One `report_fetch` display row: `(glyph, label, r, g, b, detail)`, where
/// `detail` replaces the size column when set.
type FetchRow = (char, &'static str, u8, u8, u8, Option<String>);

/// The display row for a fetch outcome — glyph, label, truecolor, and the
/// optional detail that replaces the size column when a fetch delivered no
/// bytes. Shared by the streamed log ([`report_fetch`]) and the live tree, so a
/// dependency reads the same either way: a *fetched* dep the local bloom filters
/// vouch for (or flag) is relabeled `known` instead of `live`/`cache` (green ✓
/// known-good, red ✗ known-bad); `skip`/`fail` rows are left as-is.
fn fetch_row(rec: &FetchRecord) -> FetchRow {
    match &rec.outcome {
        Outcome::PinMismatch => (
            '\u{2716}',
            "pin!",
            255,
            90,
            90,
            Some("hash mismatch".to_string()),
        ),
        Outcome::Ok if rec.stale => ('\u{25cf}', "stale", 230, 180, 80, None),
        Outcome::Ok => {
            if let Some(verdict) = bloom_fetch_verdict(rec) {
                verdict
            } else if rec.cached {
                ('\u{25cf}', "cache", 120, 200, 140, None)
            } else {
                ('\u{2b07}', "live", 100, 180, 255, None)
            }
        }
        Outcome::BudgetExceeded => (
            '\u{25cb}',
            "budget",
            230,
            180,
            80,
            Some("over fetch budget".to_string()),
        ),
        Outcome::Unresolved => (
            '\u{00b7}',
            "skip",
            120,
            120,
            120,
            Some("unresolved".to_string()),
        ),
        Outcome::Skipped => (
            '\u{00b7}',
            "skip",
            120,
            120,
            120,
            Some("not a target".to_string()),
        ),
        Outcome::Failed(why) => (
            '\u{2716}',
            "fail",
            255,
            90,
            90,
            Some(failure_detail(rec, why)),
        ),
    }
}

/// A bloom verdict for a fetched artifact, as a `report_fetch` row override:
/// every known state renders as the `known` label, distinguished by glyph —
/// 🚩 known-bad, 🏴 conflicted (both still scanned; the flag also rides the
/// result header), green ✓ known-good (fetched here only because a pulled/fresh
/// exception forced a re-scan). `None` when bloom is disabled or the artifact is
/// in neither set. Checks the fetched content's sha256 and, for a dependency, its
/// PURL; a conflict/bad hit on either wins over a good hit.
fn bloom_fetch_verdict(rec: &FetchRecord) -> Option<FetchRow> {
    use crate::bloom_repo::Decision;
    let lookup = crate::bloom_repo::global()?;

    let sha = rec
        .content_sha256
        .as_deref()
        .and_then(crate::bloom::parse_sha256_hex)
        .map(|d| lookup.decide_sha256(&d));
    let purl = rec
        .locator
        .starts_with("pkg:")
        .then(|| lookup.decide_purl(&rec.locator));

    let decisions = [sha, purl];
    let has = |want: Decision| decisions.iter().flatten().any(|d| *d == want);
    if has(Decision::Conflicted) {
        return Some(('\u{1f3f4}', "known", 230, 180, 80, None)); // 🏴
    }
    if has(Decision::KnownBad) {
        return Some(('\u{1f6a9}', "known", 235, 120, 120, None)); // 🚩
    }
    if has(Decision::Skip) {
        return Some(('\u{2713}', "known", 80, 200, 80, None));
    }
    None
}

/// The compact failure note for a failed fetch — the HTTP status when one was
/// seen (the common, informative case), else the transport reason trimmed.
fn failure_detail(rec: &FetchRecord, why: &str) -> String {
    rec.status.map_or_else(
        || {
            why.split(['\n', ':'])
                .next()
                .unwrap_or(why)
                .trim()
                .to_string()
        },
        |s| format!("HTTP {s}"),
    )
}

/// Why a dependency's artifact fetch was skipped. The registry record is
/// materialized (and trait-matched) in every case; only the byte fetch+scan is
/// skipped. Drives how the skip is surfaced in the fetch progress block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkipReason {
    /// Older than `--max-dep-age`. The common, expected case — reported at debug
    /// only, no progress line.
    AgedOut,
    /// The version was unpublished/yanked from the registry — no artifact to
    /// fetch. Rare and worth surfacing.
    Removed,
    /// Matched the known-good bloom set by PURL — a trusted coordinate, so its
    /// bytes are not re-fetched or re-scanned (see [`must_rescan`] for the
    /// pulled/fresh exceptions that override this).
    KnownGood,
}

/// Whether a dependency version has been withdrawn from its registry — an npm
/// unpublish/yank (`version_removed`), a pypi/crates yank (recorded as a
/// `deprecated` reason, which never sets `version_removed`), or an npm security
/// takedown (`security_hold`). A withdrawn version's known-good vouch is suspect
/// — it is often *removed because* it was found malicious — so it is re-scanned.
fn dep_pulled(reg: &Registry) -> bool {
    reg.version_removed == Some(true)
        || reg.security_hold == Some(true)
        || reg.deprecated.as_deref().is_some_and(|d| {
            let d = d.to_ascii_lowercase();
            d.contains("yank") || d.contains("withdrawn") || d.contains("unpublish")
        })
}

/// Number of seconds in the freshness window: a version published this recently
/// is re-scanned rather than trusted on a known-good vouch. Mirrors the local
/// file recency window in [`crate::engine`].
pub(crate) const FRESH_WINDOW_SECS: u64 = 48 * 3_600;

/// Whether this version was published within the last 48h. A known-good bloom
/// vouch is built ahead of time; for a freshly minted release the vouch may
/// predate the bytes now being served, so re-scan rather than trust it.
fn fresh_48h(reg: &Registry, now: u64) -> bool {
    reg.age_secs(now)
        .is_some_and(|age| age <= FRESH_WINDOW_SECS)
}

/// A known-good dependency is normally skipped; re-scan it anyway when its trust
/// may be stale — the version was pulled/yanked, or published in the last 48h.
pub(crate) fn must_rescan(reg: &Registry, now: u64) -> bool {
    dep_pulled(reg) || fresh_48h(reg, now)
}

/// Whether a reference is a known-good package coordinate per the loaded bloom
/// filters. Purl-keyed, so it vouches for the coordinate (not the exact bytes);
/// callers pair it with [`must_rescan`] before trusting it enough to skip.
fn bloom_known_good_purl(r: &Reference) -> bool {
    let RefLocator::Purl(purl) = &r.locator else {
        return false;
    };
    crate::bloom_repo::global()
        .is_some_and(|lk| lk.decide_purl(purl) == crate::bloom_repo::Decision::Skip)
}

/// Skip predicate for fetched-dependency analysis: skip any member cleave is
/// about to analyze whose sha256 the installed bloom filters vouch known-good —
/// the same short-circuit the top-level scan applies, so a prebuilt native tool
/// shipped inside a dependency isn't needlessly re-disassembled. Unlike the
/// top-level predicate it applies no local-file freshness guard: a dependency's
/// bytes are content-addressed (fetched by locator, sha-verified) and extracted
/// to fresh temp files, so an mtime check would spuriously force analysis every
/// run. Known-bad, conflicted, and unknown members are always analyzed. `None`
/// when no bloom set is installed, leaving analysis unfiltered.
fn dep_skip_predicate() -> Option<cleave::SkipPredicate> {
    let lookup = crate::bloom_repo::global()?;
    Some(cleave::SkipPredicate(std::sync::Arc::new(
        move |sha_hex: &str, _path: &Path| {
            crate::bloom::parse_sha256_hex(sha_hex)
                .is_some_and(|d| lookup.decide_sha256(&d) == crate::bloom_repo::Decision::Skip)
        },
    )))
}

/// One dependency's age-gate outcome: the registry record (materialized as facts
/// either way) paired with why its byte fetch was skipped — `None` when the
/// dependency is kept for a full fetch+scan.
type GatedDep = (Reference, Registry, Option<SkipReason>);

/// Look up each declared dependency's registry metadata, stamp its relative
/// age, and decide which to fetch. A dependency older than the policy's age
/// ceiling — or one whose coordinate is known-good and whose trust isn't stale —
/// is dropped before the expensive fetch+scan of its bytes; one whose age is
/// unknown or under the ceiling is kept — fail open, so a registry hiccup or an
/// unsupported ecosystem never silently hides a dependency from the scan. URLs
/// and command-mentioned packages aren't gated: their risk isn't a function of a
/// registry release date. Returns the refs to fetch plus, for *every* dependency
/// that resolved a registry record, its [`GatedDep`] — the record is materialized
/// whether or not its bytes are fetched, and the reason drives the skip report.
fn age_gate(
    selected: Vec<Reference>,
    policy: &FetchPolicy,
    res: &Resources,
    now: u64,
) -> (Vec<Reference>, Vec<GatedDep>) {
    // `None` ceiling (the `--max-dep-age 0` opt-out) gates nothing, but registry
    // records are still looked up and materialized.
    let max_age =
        (policy.max_dep_age_days > 0).then(|| u64::from(policy.max_dep_age_days) * 86_400);
    // The network round-trips run concurrently up front; the gate decision below
    // is then pure, so it stays deterministic in `selected` order.
    let lookups = lookup_registries(&selected, res, now);
    let mut keep = Vec::with_capacity(selected.len());
    let mut registries = Vec::new();
    for (r, lookup) in selected.into_iter().zip(lookups) {
        match lookup {
            // A resolved record: gate on age, but materialize it either way. A
            // version the registry has already removed has no fetchable artifact,
            // so skip the doomed fetch too. A known-good coordinate is skipped
            // unless its trust may be stale (pulled/yanked or <48h old). In every
            // skip case the materialized record's signals still surface.
            Some(reg) => {
                let reason = if reg.version_removed == Some(true) {
                    Some(SkipReason::Removed)
                } else if max_age.is_some_and(|max| reg.age_secs(now).is_some_and(|age| age >= max))
                {
                    Some(SkipReason::AgedOut)
                } else if bloom_known_good_purl(&r) && !must_rescan(&reg, now) {
                    Some(SkipReason::KnownGood)
                } else {
                    None
                };
                if reason.is_none() {
                    keep.push(r.clone());
                }
                registries.push((r, reg, reason));
            }
            // A non-dependency, or a dependency whose record didn't resolve —
            // fetch it (fail open).
            None => keep.push(r),
        }
    }
    (keep, registries)
}

/// Process-wide memo of registry lookups, keyed by locator string. A package
/// named across many scanned files resolves once: the first lookup fills this,
/// and every later file reads the record straight from memory — no repeated disk
/// read, JSON parse, and ecosystem mapping. The *un-aged* record is stored (age
/// is relative to each scan's clock, so [`Registry::with_age`] is applied per
/// read); `None` is memoized too, so an unsupported ecosystem or an unresolved
/// package isn't re-attempted for every file that names it. Lives for the
/// process, fronting the on-disk blob cache.
fn registry_memo() -> &'static RwLock<HashMap<String, Option<Registry>>> {
    static MEMO: OnceLock<RwLock<HashMap<String, Option<Registry>>>> = OnceLock::new();
    MEMO.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Look up each declared dependency's registry record, returning one slot per
/// input ref in `selected` order. A non-dependency ref, or one whose record
/// can't be resolved, yields `None`. Memoized hits ([`registry_memo`]) are served
/// from memory; the remaining misses are resolved concurrently, bounded by
/// [`REGISTRY_LOOKUP_CONCURRENCY`] on plain OS threads, then folded back into the
/// memo. Each lookup is keyed by a distinct locator, so the shared cache sees no
/// write contention.
fn lookup_registries(selected: &[Reference], res: &Resources, now: u64) -> Vec<Option<Registry>> {
    let mut out: Vec<Option<Registry>> = selected.iter().map(|_| None).collect();

    // Split dependency refs into memo hits — served from memory, no disk or
    // network — and misses that still need a lookup.
    let mut misses: Vec<usize> = Vec::new();
    {
        let memo = registry_memo()
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        for i in (0..selected.len()).filter(|&i| selected[i].kind == RefKind::Dependency) {
            match memo.get(&locator_key(&selected[i])) {
                // Stored un-aged; stamp the age signals from this scan's clock.
                Some(hit) => out[i] = hit.clone().map(|reg| reg.with_age(now)),
                None => misses.push(i),
            }
        }
    }
    if misses.is_empty() {
        return out;
    }

    let cursor = AtomicUsize::new(0);
    let workers = REGISTRY_LOOKUP_CONCURRENCY.min(misses.len());
    let collected: Vec<Vec<(usize, Option<Registry>)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let t = cursor.fetch_add(1, Ordering::Relaxed);
                        let Some(&i) = misses.get(t) else {
                            break;
                        };
                        // The raw, un-aged record (or `None` for an unresolved or
                        // unsupported package) — both worth memoizing so the
                        // lookup isn't re-attempted for every file that names it.
                        local.push((
                            i,
                            fletch::registry(&selected[i].locator, &res.net, &res.cache),
                        ));
                    }
                    local
                })
            })
            .collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });

    // Stamp the aged copy for this scan into each result slot, keeping the raw
    // record to memoize.
    let mut writes: Vec<(String, Option<Registry>)> = Vec::with_capacity(misses.len());
    for (i, reg) in collected.into_iter().flatten() {
        out[i] = reg.clone().map(|reg| reg.with_age(now));
        writes.push((locator_key(&selected[i]), reg));
    }
    // One short critical section: nothing but the batch insert runs under the lock.
    registry_memo()
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .extend(writes);
    out
}

/// Serialize a registry record to its `*.registry.json` document and analyze it
/// with cleave, so filefacts parses the `registry.*` facts and the trait engine
/// runs over them. The document is named by the package so detection routes it
/// to `FileType::Registry`. `None` if it can't be serialized or analyzed.
fn registry_node(reg: &Registry, opts: &AnalysisOptions) -> Option<AnalysisReport> {
    let bytes = serde_json::to_vec(reg).ok()?;
    match cleave::analyze_bytes_owned(bytes, &registry_doc_name(reg), opts) {
        Ok(mut sub) => {
            sub.finalize();
            Some(sub)
        }
        Err(e) => {
            tracing::warn!(package = %reg.name, "registry metadata analysis failed: {e:#}");
            None
        }
    }
}

/// Graft a materialized registry sub-report under the file that declared the
/// dependency (its sha256), mirroring [`merge_payload`]'s id/depth re-basing.
/// The node carries only facts — a registry document references nothing to
/// fetch — so no next-hop work-list is produced.
fn merge_registry(report: &mut AnalysisReport, parent_sha: &str, sub: AnalysisReport) {
    let (parent_id, parent_depth) = report
        .files
        .iter()
        .find(|f| f.sha256 == parent_sha)
        .map_or((0, 0), |f| (f.id, f.depth));
    let id_base = report.files.iter().map(|f| f.id).max().map_or(0, |m| m + 1);
    for mut file in sub.files {
        // The registry document itself (the sub-report's root) is a sidecar:
        // metadata about its parent package, analyzed from its own canonical
        // JSON bytes so its findings feed ML, but not standalone content.
        if file.parent_id.is_none() {
            file.rel = cleave::types::Rel::Registry;
            file.role = cleave::types::Role::Sidecar;
        }
        file.id += id_base;
        file.parent_id = Some(file.parent_id.map_or(parent_id, |p| p + id_base));
        file.depth += parent_depth + 1;
        report.files.push(file);
    }
}

/// The synthetic filename for a registry document: `<name>@<version>.registry
/// .json`, with path-unsafe characters folded to `_` so a scoped or `vendor/pkg`
/// name can't escape into a directory. The `.registry.json` suffix is what
/// filefacts detects.
fn registry_doc_name(reg: &Registry) -> String {
    let stem = if reg.version.is_empty() {
        reg.name.clone()
    } else {
        format!("{}@{}", reg.name, reg.version)
    };
    let base: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '@') {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{base}.registry.json")
}

/// Render one skipped dependency to stderr in the fetch progress block: the
/// package, its age in days, and the strongest reputation signal the registry
/// gave (downloads, else votes/rating). A known-good skip reads green with a ✓
/// (trusted, not re-scanned); a removed version reads muted (no artifact to
/// fetch). Aged-out deps never reach here — they stay at debug.
fn report_skip(r: &Reference, reg: &Registry, now: u64, reason: SkipReason) {
    let age_days = reg.age_secs(now).unwrap_or(0) / 86_400;
    let signal = reg
        .downloads_recent
        .or(reg.downloads_total)
        .map(|d| format!("{d} dl"))
        .or_else(|| reg.rating_count.map(|v| format!("{v} votes")))
        .unwrap_or_default();
    let column = format!("{age_days}d old");
    let detail = if signal.is_empty() {
        String::new()
    } else {
        format!("  \x1b[2m{signal}\x1b[0m")
    };
    // (glyph, label, r, g, b) — green ✓ for a trusted known-good skip, muted for
    // an unpublished/removed version.
    let (glyph, label, cr, cg, cb) = match reason {
        SkipReason::KnownGood => ('\u{2713}', "known-good", 80, 200, 80),
        SkipReason::Removed => ('\u{00b7}', "removed", 120, 120, 120),
        SkipReason::AgedOut => ('\u{00b7}', "skip", 120, 120, 120),
    };
    eprintln!(
        "    \x1b[38;2;{cr};{cg};{cb}m{glyph} {label:<10}\x1b[0m \x1b[38;2;130;130;130m{column:>10}\x1b[0m  {}{detail}",
        locator_key(r)
    );
}

/// Wall-clock now as Unix seconds, saturating to `0` before the epoch.
pub(crate) fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Print the run's fetch summary to stderr (the streamed log's closing line).
fn report_summary(records: &[FetchRecord]) {
    eprintln!("{}", summary_line(records));
}

/// Tally the run's fetches into a one-line summary mirroring the progress bar's
/// completion line: how many came live off the network vs. served from cache,
/// how many failed, and the total bytes pulled. Shared by the streamed log and
/// the live tree, which prints it beneath the settled dependency rows.
fn summary_line(records: &[FetchRecord]) -> String {
    let mut live = 0u32;
    let mut cached = 0u32;
    let mut failed = 0u32;
    let mut bytes = 0u64;
    for rec in records {
        bytes += rec.size.unwrap_or(0);
        match &rec.outcome {
            Outcome::Ok | Outcome::PinMismatch if rec.cached => cached += 1,
            Outcome::Ok | Outcome::PinMismatch => live += 1,
            Outcome::Failed(_) => failed += 1,
            Outcome::BudgetExceeded | Outcome::Unresolved | Outcome::Skipped => {}
        }
    }
    // Only the counts that actually happened, so a warm run reads
    // `2 cached  ·  160.5 MB` instead of padding a `0 live` nobody asked about.
    // Bytes always show — the total pulled is the headline the tally exists for.
    let mut parts = Vec::new();
    if live > 0 {
        parts.push(format!("{live} live"));
    }
    if cached > 0 {
        parts.push(format!("{cached} cached"));
    }
    if failed > 0 {
        parts.push(format!("{failed} failed"));
    }
    parts.push(human_bytes(bytes));
    format!(
        "  \x1b[38;2;80;220;80m\u{2713}\x1b[0m  \x1b[38;2;160;160;160m{}\x1b[0m",
        parts.join("  \u{b7}  ")
    )
}

/// Bytes in a compact human-readable form (`45.2 KB`), for the fetch log.
fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    #[allow(clippy::cast_precision_loss)] // display only; magnitude, not exact bytes
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// References to fetch, grouped by the sha256 of the file that declared them.
fn collect_references(report: &AnalysisReport, root_path: &Path) -> Vec<(String, Vec<Reference>)> {
    let mut groups: Vec<(String, Vec<Reference>)> = Vec::new();
    for file in &report.files {
        let Some(view) = &file.filefacts else {
            continue;
        };
        // Declared references plus the value-driven hunt (npm lifecycle hooks),
        // both from facts the report already carries — so every archive member,
        // not just the root, contributes its references without re-extraction.
        let mut refs = find::references_from_facts(&view.values, &view.references);
        // Module-load calls from the member's retained AST symbols — the
        // facts-only import vector, so `require("undeclared-pkg")` inside an
        // archive member is hunted without re-extracting its discarded bytes.
        refs.extend(find::import_calls(&view.symbols));
        if refs.is_empty() {
            continue;
        }
        groups.push((file.sha256.clone(), refs));
    }

    // The root sample's imperative hunt (curl|sh, `npm install` in a RUN, a URL
    // in a shell variable) needs its raw text, which the report doesn't carry —
    // read it back from disk for small text-ish roots and merge, deduping
    // against the declared references already collected for the root.
    if let Some(root) = report.files.first()
        && root.size <= ROOT_HUNT_MAX_BYTES
        && let Ok(bytes) = std::fs::read(root_path)
    {
        let name = root_path
            .file_name()
            .map_or_else(|| root_path.to_string_lossy(), |n| n.to_string_lossy());
        let hunted = find::references_in_bytes(&bytes, &name);
        if !hunted.is_empty() {
            merge_into_root(&mut groups, &root.sha256, hunted);
        }
    }
    groups
}

/// Merge the root's hunted references into its group (creating it if the root
/// declared none), skipping any locator already present.
fn merge_into_root(
    groups: &mut Vec<(String, Vec<Reference>)>,
    root_sha: &str,
    hunted: Vec<Reference>,
) {
    if !groups.iter().any(|(sha, _)| sha == root_sha) {
        groups.push((root_sha.to_string(), Vec::new()));
    }
    let Some((_, group)) = groups.iter_mut().find(|(sha, _)| sha == root_sha) else {
        return; // unreachable: just ensured the group exists
    };
    let mut seen: HashSet<String> = group.iter().map(locator_key).collect();
    for r in hunted {
        if seen.insert(locator_key(&r)) {
            group.push(r);
        }
    }
}

/// A reference's locator as a stable string for dedup and cross-record pairing
/// (it matches `FetchRecord::locator`, the original filefacts locator).
fn locator_key(r: &Reference) -> String {
    match &r.locator {
        RefLocator::Purl(s) | RefLocator::Url(s) | RefLocator::Path(s) => s.clone(),
    }
}

/// Every finding a finalized sub-report carries, flattened across its file
/// nodes — the seed half the package pass contributes from one side (artifact
/// or registry metadata).
fn sub_findings(sub: &AnalysisReport) -> Vec<Finding> {
    sub.files
        .iter()
        .flat_map(|f| f.findings.iter().cloned())
        .collect()
}

/// How many registry-metadata lookups run at once in [`age_gate`]. Unlike the
/// CPU-bound payload analysis, these are I/O-bound (small, cached HTTP GETs), so
/// a higher fan-out turns a manifest's worth of serial round-trips into a few
/// parallel batches without taxing the CPU — while staying polite to registries.
const REGISTRY_LOOKUP_CONCURRENCY: usize = 8;

/// The product of analyzing one fetched payload: the finalized sub-report to
/// graft (absent if the payload couldn't be analyzed) and the next-hop
/// references found in its own bytes. Produced off the report so the expensive
/// analysis can run concurrently; [`merge_payload`] folds it in serially.
struct Analyzed {
    sub: Option<AnalysisReport>,
    content_sha: String,
    next_from_bytes: Vec<(String, Vec<Reference>)>,
}

/// Analyze the bytes of fetched payloads on cleave's shared rayon pool, one slot
/// per input record in order (`None` where there was nothing to analyze).
///
/// The payloads fan out as rayon tasks, so the whole pool works the batch: a
/// dependency carrying a large native binary — a minutes-long, single-threaded
/// disassembly that no amount of threads can split — runs *alongside* its
/// siblings instead of being metered a couple at a time on a private pair of OS
/// threads. Nesting is safe and is the point: each payload's own analysis is
/// itself rayon-parallel, and a task that blocks awaiting its children steals
/// and runs other pending work, so the machine stays saturated rather than
/// idling behind one slow binary. Called from a plain thread the caller simply
/// blocks on the pool; called from a worker it nests — either way `on_analyzed`
/// fires as each payload settles, and the indexed collect preserves input order.
///
/// Concurrency is bounded by the pool width (work-stealing runs ~one payload per
/// worker at a time), so at most that many payloads' bytes are resident at once
/// — the batch size itself never dictates peak memory.
fn analyze_payloads(
    fetched: &[FetchRecord],
    cache: &BlobCache,
    opts: &AnalysisOptions,
    acache: Option<&AnalysisCache>,
    on_analyzed: &(dyn Fn(usize) + Sync),
) -> Vec<Option<Analyzed>> {
    use rayon::prelude::*;
    fetched
        .par_iter()
        .enumerate()
        .map(|(i, rec)| {
            let a = analyze_payload(rec, cache, opts, acache);
            on_analyzed(i);
            a
        })
        .collect()
}

/// Analyze a fetched payload's bytes (the expensive, report-independent half of
/// grafting): hunt its own bytes for next-hop references and run cleave over it.
/// Returns `None` when there is nothing to analyze (a skipped/unresolved/failed
/// fetch, or bytes that vanished from cache). Pure with respect to the report,
/// so it is safe to run concurrently; [`merge_payload`] does the report mutation.
fn analyze_payload(
    rec: &FetchRecord,
    cache: &BlobCache,
    opts: &AnalysisOptions,
    acache: Option<&AnalysisCache>,
) -> Option<Analyzed> {
    // Scan whatever bytes we hold: a clean fetch or a pin mismatch (a mismatch
    // is exactly the case worth analyzing). Skipped/unresolved/failed have none.
    if !matches!(rec.outcome, Outcome::Ok | Outcome::PinMismatch) {
        return None;
    }
    let content_sha = rec.content_sha256.clone().unwrap_or_default();

    // Warm-cache hit: reuse the prior analysis of these exact bytes, skipping the
    // re-analysis (a minutes-long disassembly for a big native binary). Keyed by
    // content sha under a ruleset-version namespace, so an entry is only ever one
    // the current detector produced — a rules/engine change misses and re-scans.
    if let Some(ac) = acache
        && !content_sha.is_empty()
        && let Some(hit) = ac.get(&content_sha)
    {
        tracing::debug!(
            locator = %rec.locator,
            content_sha = %content_sha,
            "analysis cache hit; reusing prior result"
        );
        return Some(Analyzed {
            sub: hit.sub,
            content_sha,
            next_from_bytes: hit.next,
        });
    }

    let bytes = cache.load(&rec.locator)?;
    let name = payload_name(rec);

    // Next-hop references discovered in the payload's own bytes — the full hunt,
    // so a stage-2 script's `curl | bash` (or an encoded URL) is followed.
    let mut next_from_bytes = Vec::new();
    let payload_refs = find::references_in_bytes(&bytes, &name);
    if !content_sha.is_empty() && !payload_refs.is_empty() {
        next_from_bytes.push((content_sha.clone(), payload_refs));
    }

    let sub = match cleave::analyze_bytes_owned(bytes, &name, opts) {
        Ok(mut sub) => {
            // finalize() collapses the sub-analysis into its files[]; without it
            // the payload's data stays in top-level fields and files[] is empty.
            sub.finalize();
            Some(sub)
        }
        Err(e) => {
            tracing::warn!("analysis of fetched {} failed: {e:#}", rec.locator);
            None
        }
    };

    // Memoize for the next run's warm hit (best-effort; borrowed, so no clone of
    // the report). Only cache a definite result — an analysis error might be a
    // transient (a cache-evicted byte, an OOM), so leave it to re-run.
    if let Some(ac) = acache
        && !content_sha.is_empty()
        && sub.is_some()
    {
        ac.put(&content_sha, &sub, &next_from_bytes);
    }

    Some(Analyzed {
        sub,
        content_sha,
        next_from_bytes,
    })
}

/// Fold an [`Analyzed`] payload into the report: append its file nodes nested
/// under the file that declared the reference, and return the references the
/// payload yields for the next hop. The fetch edge (`source_sha256 →
/// content_sha256`) is the authoritative link; ids and depth are renumbered so
/// the grafted nodes are a well-formed subtree that never collides with the main
/// report's. Must run serially — it reads and extends `report.files`.
fn merge_payload(
    report: &mut AnalysisReport,
    rec: &FetchRecord,
    analyzed: Analyzed,
) -> Vec<(String, Vec<Reference>)> {
    let mut next = analyzed.next_from_bytes;
    let Some(sub) = analyzed.sub else {
        return next;
    };

    // Attach under the file that declared the reference (its sha256 is the
    // edge's source endpoint); fall back to the root file.
    let (parent_id, parent_depth) = report
        .files
        .iter()
        .find(|f| f.sha256 == rec.source_sha256)
        .map_or((0, 0), |f| (f.id, f.depth));
    let id_base = report.files.iter().map(|f| f.id).max().map_or(0, |m| m + 1);
    // The resolved download URL (falling back to the bare locator/PURL) this
    // subtree came from, recorded on the graft root as `via`.
    let via_str = if rec.resolved_url.is_empty() {
        rec.locator.as_str()
    } else {
        rec.resolved_url.as_str()
    };
    let via = (!via_str.is_empty()).then(|| via_str.to_string());
    let first_new = report.files.len();
    for mut file in sub.files {
        // The payload's own top node (the sub-report's root) is a fetched edge:
        // pulled from `via`, not contained in its parent. Its exploded members
        // stay ordinary members.
        let is_sub_root = file.parent_id.is_none();
        file.id += id_base;
        file.parent_id = Some(file.parent_id.map_or(parent_id, |p| p + id_base));
        file.depth += parent_depth + 1;
        if is_sub_root {
            file.rel = cleave::types::Rel::Fetched;
            file.via = via.clone();
        }
        report.files.push(file);
    }

    // If the payload was an archive, its members' facts (declared deps, npm
    // hooks) are the next hop too — the bytes hunt above only saw the container.
    // The payload's own node is skipped; the bytes hunt already covered it.
    for file in &report.files[first_new..] {
        if file.sha256 == analyzed.content_sha {
            continue;
        }
        if let Some(view) = &file.filefacts {
            let refs = find::references_from_facts(&view.values, &view.references);
            if !refs.is_empty() {
                next.push((file.sha256.clone(), refs));
            }
        }
    }
    next
}

/// Run the package-scoped composite pass for one fetched artifact, grafting any
/// composite that correlates its bytes with its registry metadata onto the
/// artifact node. A `scope: package` (or `scope: outer`) rule can thus fire on,
/// say, "deprecated on the registry **and** ships a native addon" even though
/// the artifact and the registry document were analyzed as separate reports and
/// never share an archive. The grafted composite carries its members in
/// `trait_refs`, so the later `strip_unmatched_traits` keeps the registry
/// building-block traits it fired on. A no-op when either side is empty.
fn apply_package_composites(
    report: &mut AnalysisReport,
    artifact_sha: &str,
    artifact_findings: &[Finding],
    registry_findings: &[Finding],
    opts: &AnalysisOptions,
) {
    match cleave::graft_package_composites(
        report,
        artifact_sha,
        artifact_findings,
        registry_findings,
        opts,
    ) {
        Ok(0) => {}
        Ok(n) => tracing::info!(
            grafted = n,
            "package-scoped composites fired across artifact and registry metadata"
        ),
        Err(e) => tracing::warn!("package composite pass failed: {e:#}"),
    }
}

/// A filename for a fetched payload: the final URL's basename, else the
/// content hash. Drives cleave's extension-based type detection.
fn payload_name(rec: &FetchRecord) -> String {
    let url = rec.final_url.as_deref().unwrap_or(&rec.resolved_url);
    url.rsplit('/')
        .next()
        .and_then(|s| s.split(['?', '#']).next())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| rec.content_sha256.clone())
        .unwrap_or_else(|| "fetched".to_string())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use fletch::RefKind;

    #[test]
    fn dep_pulled_covers_removed_yank_and_hold() {
        let base = Registry::default();
        assert!(!dep_pulled(&base));
        assert!(dep_pulled(&Registry {
            version_removed: Some(true),
            ..Registry::default()
        }));
        assert!(dep_pulled(&Registry {
            security_hold: Some(true),
            ..Registry::default()
        }));
        // pypi/crates record a yank as a `deprecated` reason, never `version_removed`.
        assert!(dep_pulled(&Registry {
            deprecated: Some("Yanked: critical CVE".to_string()),
            ..Registry::default()
        }));
        // An ordinary deprecation notice is not a withdrawal.
        assert!(!dep_pulled(&Registry {
            deprecated: Some("use v2 instead".to_string()),
            ..Registry::default()
        }));
    }

    #[test]
    fn url_host_strips_scheme_path_and_query() {
        // A signed release-asset redirect: only the host survives, the SAS
        // token / JWT query never reaches the line.
        assert_eq!(
            url_host("https://release-assets.githubusercontent.com/x/y?sig=abc&jwt=xyz"),
            "release-assets.githubusercontent.com"
        );
        assert_eq!(url_host("https://example.com"), "example.com");
        // `userinfo@` is dropped; `host:port` is kept.
        assert_eq!(
            url_host("http://user:pass@host.test:8080/x"),
            "host.test:8080"
        );
        // Not a URL: returned as-is.
        assert_eq!(url_host("bareword"), "bareword");
    }

    #[test]
    fn summary_line_omits_zero_counts() {
        let record = |outcome: Outcome, cached: bool, size: Option<u64>| FetchRecord {
            source_sha256: String::new(),
            source_offset: None,
            locator: "pkg:npm/x".to_string(),
            resolved_url: String::new(),
            final_url: None,
            redirects: Vec::new(),
            status: None,
            headers: Vec::new(),
            fetched_at: 0,
            content_sha256: None,
            size,
            cached,
            stale: false,
            pin_verified: None,
            outcome,
        };
        // Two cache hits, nothing live: the `0 live` is dropped, bytes stay.
        let warm = vec![
            record(Outcome::Ok, true, Some(512)),
            record(Outcome::Ok, true, Some(512)),
        ];
        let line = summary_line(&warm);
        assert!(line.contains("2 cached"), "{line}");
        assert!(
            !line.contains("live"),
            "zero `live` must be omitted: {line}"
        );
        // A mixed run keeps both non-zero counts.
        let mixed = vec![
            record(Outcome::Ok, false, Some(0)),
            record(Outcome::Ok, true, Some(0)),
        ];
        let line = summary_line(&mixed);
        assert!(line.contains("1 live"), "{line}");
        assert!(line.contains("1 cached"), "{line}");
    }

    #[test]
    fn fresh_48h_and_must_rescan_track_publish_age() {
        let now = 1_000_000_u64;
        let fresh = Registry {
            published_at: Some(now - 3_600), // 1h ago
            ..Registry::default()
        };
        // 30h ago: inside the 48h window, but would be outside a 24h one.
        let day_and_a_half = Registry {
            published_at: Some(now - 30 * 3_600),
            ..Registry::default()
        };
        let stale = Registry {
            published_at: Some(now - 300_000), // ~3.5d ago
            ..Registry::default()
        };
        assert!(fresh_48h(&fresh, now));
        assert!(fresh_48h(&day_and_a_half, now));
        assert!(!fresh_48h(&stale, now));
        // A stale, unwithdrawn version needs no re-scan; a fresh one does, and a
        // withdrawn one always does regardless of age.
        assert!(!must_rescan(&stale, now));
        assert!(must_rescan(&fresh, now));
        assert!(must_rescan(
            &Registry {
                published_at: Some(now - 300_000),
                version_removed: Some(true),
                ..Registry::default()
            },
            now
        ));
    }

    fn url_ref(url: &str) -> Reference {
        Reference {
            locator: RefLocator::Url(url.to_string()),
            kind: RefKind::UrlFetch,
            source: "test".to_string(),
            evidence: url.to_string(),
            offset: 0,
            pinned_hash: None,
            content_sha256: None,
        }
    }

    fn purl_ref(purl: &str) -> Reference {
        Reference {
            locator: RefLocator::Purl(purl.to_string()),
            kind: RefKind::Dependency,
            source: "test".to_string(),
            evidence: purl.to_string(),
            offset: 0,
            pinned_hash: None,
            content_sha256: None,
        }
    }

    #[test]
    fn off_host_platform_matches_native_binary_naming() {
        let host = ("darwin", "arm64");
        // The host variant is kept; other-platform siblings are skipped.
        assert!(!off_host_platform(
            &purl_ref("pkg:npm/%40biomejs/cli-darwin-arm64@2.5.0"),
            host
        ));
        assert!(off_host_platform(
            &purl_ref("pkg:npm/%40biomejs/cli-linux-x64-musl@2.5.0"),
            host
        ));
        assert!(off_host_platform(
            &purl_ref("pkg:npm/%40biomejs/cli-win32-arm64@2.5.0"),
            host
        ));
        // Same OS, wrong arch is still off-host.
        assert!(off_host_platform(
            &purl_ref("pkg:npm/%40biomejs/cli-darwin-x64@2.5.0"),
            host
        ));
        // `<arch>-<os>` order (esbuild-style) and other scopes.
        assert!(off_host_platform(
            &purl_ref("pkg:npm/%40esbuild/linux-x64@0.21.0"),
            host
        ));
        // A portable package (no os+arch pair) is never platform-skipped.
        assert!(!off_host_platform(
            &purl_ref("pkg:npm/left-pad@1.3.0"),
            host
        ));
        assert!(!off_host_platform(&purl_ref("pkg:npm/semver@7.5.0"), host));
        // A raw URL carries no package identity to place.
        assert!(!off_host_platform(
            &url_ref("https://example.com/x.tgz"),
            host
        ));
        // Fail open when the host platform can't be named.
        assert!(!off_host_platform(
            &purl_ref("pkg:npm/%40biomejs/cli-linux-x64@2.5.0"),
            ("", "")
        ));
    }

    #[test]
    fn fetch_policy_parses_kinds_and_rejects_garbage() {
        assert_eq!(
            "deps".parse(),
            Ok(FetchPolicy {
                deps: true,
                ..FetchPolicy::default()
            })
        );
        assert_eq!(
            "packages".parse(),
            Ok(FetchPolicy {
                packages: true,
                ..FetchPolicy::default()
            })
        );
        assert_eq!(
            " urls , packages , deps ".parse(),
            Ok(FetchPolicy {
                urls: true,
                packages: true,
                deps: true,
                ..FetchPolicy::default()
            })
        );
        // `all` is shorthand for every kind.
        assert_eq!(
            "all".parse(),
            Ok(FetchPolicy {
                urls: true,
                packages: true,
                deps: true,
                ..FetchPolicy::default()
            })
        );
        assert_eq!("all".parse::<FetchPolicy>(), "urls,packages,deps".parse());
        // Parsing leaves depth at its default — the CLI sets it separately.
        assert_eq!(
            "deps".parse::<FetchPolicy>().unwrap().depth,
            DEFAULT_FETCH_DEPTH
        );
        assert!("".parse::<FetchPolicy>().is_err());
        assert!("sigs".parse::<FetchPolicy>().is_err());
        // The retired vocabulary is now a hard error, not a silent no-op.
        assert!("refs".parse::<FetchPolicy>().is_err());
        assert!("deps,bogus".parse::<FetchPolicy>().is_err());
        assert!(!FetchPolicy::default().enabled());

        // Selection is by kind: `packages` fetches command-mentioned packages
        // but not declared deps, and vice versa.
        let pkgs: FetchPolicy = "packages".parse().unwrap();
        assert!(pkgs.wants(RefKind::Command));
        assert!(!pkgs.wants(RefKind::Dependency));
        assert!(!pkgs.wants(RefKind::UrlFetch));
        let deps: FetchPolicy = "deps".parse().unwrap();
        assert!(deps.wants(RefKind::Dependency));
        assert!(!deps.wants(RefKind::Command));
        // Repository identity is never a fetch target.
        assert!(
            !"urls,packages,deps"
                .parse::<FetchPolicy>()
                .unwrap()
                .wants(RefKind::Repository)
        );
    }

    #[test]
    fn collect_references_unions_declared_facts_and_root_hunt() {
        // Root Dockerfile on disk: its RUN curls a URL (imperative hunt) and it
        // also declares a package dependency (a filefacts fact in the report).
        let tmp = tempfile::tempdir().expect("tempdir");
        let df = tmp.path().join("Dockerfile");
        std::fs::write(
            &df,
            b"FROM alpine\nRUN curl -fsSL https://stage.test/x.sh | sh\n",
        )
        .expect("write dockerfile");
        let sha = "ab".repeat(32);
        // Minimal one-file report (FileAnalysis has no public constructor, so
        // build it by deserialization) declaring one package dependency.
        let report: AnalysisReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [{
                "id": 0, "path": "root", "depth": 0, "file_type": "dockerfile",
                "sha256": sha, "size": 64u64,
                "filefacts": { "references": [{
                    "locator": {"purl": "pkg:npm/declared-dep@1.0.0"},
                    "kind": "dependency", "source": "test", "evidence": "declared", "offset": 0
                }]}
            }]
        }))
        .expect("minimal report deserializes");

        let groups = collect_references(&report, &df);
        assert_eq!(groups.len(), 1);
        let (gsha, refs) = &groups[0];
        assert_eq!(gsha, &sha);
        let locs: Vec<String> = refs.iter().map(locator_key).collect();
        assert!(
            locs.iter().any(|l| l == "pkg:npm/declared-dep@1.0.0"),
            "declared dep retained: {locs:?}"
        );
        assert!(
            locs.iter().any(|l| l == "https://stage.test/x.sh"),
            "hunted RUN url merged in: {locs:?}"
        );
    }

    #[test]
    fn member_require_of_undeclared_package_is_flagged() {
        // A package.json declaring `mobx`, and an archive member index.js whose
        // retained AST symbols `require("mobx")` (declared) and
        // `require("db-dx-connector")` (covert). The facts-only import hunt runs
        // on the member without its bytes; the diff flags only the undeclared one.
        let report: AnalysisReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [
                { "id": 0, "path": "package/package.json", "depth": 1,
                  "file_type": "package_json", "sha256": "cd".repeat(32), "size": 120u64,
                  "filefacts": { "references": [{
                      "locator": {"purl": "pkg:npm/mobx@^6.0.0"}, "kind": "dependency",
                      "source": "package.json", "evidence": "mobx", "offset": 0 }] } },
                { "id": 1, "path": "package/dist/index.js", "depth": 1,
                  "file_type": "javascript", "sha256": "ef".repeat(32), "size": 300u64,
                  "filefacts": { "symbols": [
                      {"kind": "call", "target": "require",
                       "args": [{"shape": "string", "value": "mobx"}]},
                      {"kind": "call", "target": "require",
                       "args": [{"shape": "string", "value": "db-dx-connector"}]}
                  ] } }
            ]
        }))
        .expect("report deserializes");

        // No on-disk root text hunt — a missing path just skips it.
        let groups = collect_references(&report, std::path::Path::new("/nonexistent"));
        let all: Vec<Reference> = groups.iter().flat_map(|(_, r)| r.iter().cloned()).collect();
        let undeclared: Vec<String> = find::undeclared_packages(&all)
            .iter()
            .map(|r| locator_key(r))
            .collect();
        assert!(
            undeclared.contains(&"pkg:npm/db-dx-connector".to_string()),
            "covert member require should be flagged undeclared: {undeclared:?}"
        );
        assert!(
            !undeclared.iter().any(|u| u.contains("mobx")),
            "declared dep must not be flagged: {undeclared:?}"
        );
    }

    #[test]
    fn merge_dedups_against_declared_and_creates_group_for_undeclared_root() {
        // Root declared one ref; the hunt finds that same one plus a new one.
        let mut groups = vec![("rootsha".to_string(), vec![url_ref("https://a.test/x")])];
        merge_into_root(
            &mut groups,
            "rootsha",
            vec![url_ref("https://a.test/x"), url_ref("https://b.test/y")],
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].1.len(), 2, "duplicate locator must not be added");

        // A root that declared nothing still receives a group from the hunt.
        let mut empty = Vec::new();
        merge_into_root(&mut empty, "rootsha", vec![url_ref("https://c.test/z")]);
        assert_eq!(
            empty,
            vec![("rootsha".to_string(), vec![url_ref("https://c.test/z")])]
        );
    }

    #[test]
    fn payload_name_prefers_url_basename_then_falls_back_to_hash() {
        let mut rec = FetchRecord {
            source_sha256: String::new(),
            source_offset: None,
            locator: "pkg:npm/x".to_string(),
            resolved_url: "https://reg.test/x/-/x-1.0.0.tgz".to_string(),
            final_url: None,
            redirects: Vec::new(),
            status: None,
            headers: Vec::new(),
            fetched_at: 0,
            content_sha256: Some("abc123".to_string()),
            size: None,
            cached: false,
            stale: false,
            pin_verified: None,
            outcome: Outcome::Ok,
        };
        assert_eq!(payload_name(&rec), "x-1.0.0.tgz");

        // Query string is stripped.
        rec.resolved_url = "https://reg.test/dl?file=stage2.sh".to_string();
        // basename before '?' is "dl" (path component), so query strip applies to it.
        assert_eq!(payload_name(&rec), "dl");

        // No usable basename → content hash.
        rec.resolved_url = "https://reg.test/".to_string();
        assert_eq!(payload_name(&rec), "abc123");
    }

    #[test]
    fn parse_bytes_reads_units_and_matches_cli_defaults() {
        // Unit suffixes are 1024-based, case-insensitive, with an optional `B`.
        assert_eq!(parse_bytes("40M"), Ok(40 * MIB));
        assert_eq!(parse_bytes("40m"), Ok(40 * MIB));
        assert_eq!(parse_bytes("40MB"), Ok(40 * MIB));
        assert_eq!(parse_bytes("40mb"), Ok(40 * MIB));
        assert_eq!(parse_bytes("2G"), Ok(2 * GIB));
        assert_eq!(parse_bytes(" 1k "), Ok(1024));
        assert_eq!(parse_bytes("512K"), Ok(512 * 1024));
        // A bare number is bytes; a trailing `B` alone is bytes too.
        assert_eq!(parse_bytes("10240"), Ok(10240));
        assert_eq!(parse_bytes("4096B"), Ok(4096));
        // Garbage, an empty number, or a lone unit are all errors.
        assert!(parse_bytes("").is_err());
        assert!(parse_bytes("abc").is_err());
        assert!(parse_bytes("M").is_err());
        assert!(parse_bytes("1.5G").is_err());

        // The CLI default strings must parse to the matching constants, so the
        // help text and the policy never drift apart.
        assert_eq!(parse_bytes("256M"), Ok(DEFAULT_MAX_FETCH_SIZE));
        assert_eq!(parse_bytes("2G"), Ok(DEFAULT_MAX_FILE_SIZE));
        assert_eq!(parse_bytes("10G"), Ok(DEFAULT_MAX_TOTAL_SIZE));
    }
}
