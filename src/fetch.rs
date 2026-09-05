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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{OnceLock, PoisonError, RwLock};
use std::time::Duration;

use cleave::{AnalysisOptions, AnalysisReport, Finding};
use fletch::fetch::{
    BlobCache, FetchBudget, FetchRecord, HttpFetch, Outcome, fetch_ref, fetch_references_with,
};
use fletch::{RefKind, RefLocator, Reference, Registry, find};

use crate::analysis_cache::AnalysisCache;
use crate::deptree::{DepState, DepTree};
use crate::hosts;

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

/// Default ceiling on *live* non-URL fetches triggered by a single scanned file
/// (`--fetch-max-file-fetches`). Cache hits don't count, so a warm re-run is
/// never throttled; this bounds the dependency/package fan-out one crafted file
/// can trigger.
pub const DEFAULT_MAX_FILE_FETCHES: usize = 100;

/// Default ceiling on *live* opportunistic URL fetches triggered by a single
/// scanned file (`--fetch-max-urls`). Cache hits don't count, so a warm re-run
/// is never throttled.
pub const DEFAULT_MAX_URL_FETCHES: usize = 4;

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

/// Describe the count budget that can clip one fetch class. The per-file cap
/// is normally the limiting value; a smaller process-wide remainder takes
/// precedence so the notice names the budget that actually stopped the work.
fn fetch_count_budget_notice(
    per_file_flag: &str,
    per_file_limit: usize,
    total_remaining: usize,
) -> String {
    if total_remaining < per_file_limit {
        format!(
            "Skipping remaining fetches, hit fetch budget (--fetch-max-total-fetches={total_remaining})"
        )
    } else {
        format!("Skipping remaining fetches, hit fetch budget ({per_file_flag}={per_file_limit})")
    }
}

/// Count the live network work represented by a batch. Cache hits and
/// budget-clipped edges are intentionally free of both process-wide budgets.
fn live_fetch_usage(records: &[FetchRecord]) -> (usize, u64) {
    records
        .iter()
        .filter(|record| fletch::fetch::counts_against_budget(record))
        .fold((0, 0), |(fetches, bytes), record| {
            (fetches + 1, bytes.saturating_add(record.size.unwrap_or(0)))
        })
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

/// Which discovered references to follow, plus how many hops to traverse.
///
/// The public vocabulary groups implementation-level reference kinds by what a
/// caller means: `dependencies` are manifest/lockfile declarations,
/// `references` are packages or URLs named by executable commands, and
/// `ci-actions` are third-party CI actions. The older `deps`, `packages`,
/// `urls`, and `ci` spellings remain CLI aliases.
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
    /// Fetch third-party code declared in a **CI** context — GitHub Actions
    /// `uses:` steps. These are `Dependency`-kind references like any other, but
    /// they run only in CI and never reach an installed artifact, so they are
    /// off by default (a routine `deps` fetch skips them) and enabled only when
    /// auditing CI itself: `--fetch=all`, `--fetch=ci`, or isomer `ci`.
    pub ci: bool,
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
    /// counted, so this caps only the cold-cache dependency/package fan-out,
    /// never a warm re-run. `0` disables these fetches entirely.
    pub max_file_fetches: usize,
    /// Ceiling on *live* opportunistic raw-URL fetches triggered by a single
    /// scanned file (`--fetch-max-urls`). Cache hits are always served and
    /// never counted. `0` disables opportunistic URL fetching while leaving
    /// declared dependencies and command-mentioned packages unaffected.
    pub max_url_fetches: usize,
    /// Ceiling on total bytes fetched on behalf of a single scanned file
    /// (`--fetch-max-file-size`). The sweep stops once retrieved bytes cross it.
    pub max_file_bytes: u64,
    /// Follow declared dependencies past the **first** hop — the dependencies
    /// of a fetched dependency, and so on to `depth`.
    ///
    /// Off for an interactive scan. Hop 1 is the artifact's own declared supply
    /// chain, which is the thing being judged; hop 2+ is a transitive closure
    /// that multiplies per hop and is dominated by the long tail of ordinary,
    /// old releases. Each of those costs a registry round trip *before* the age
    /// gate can rule it out, because the publish date is what the lookup is for:
    /// one 2.2 KB manifest sidecar pointing at a Go module drew 398 lookups at
    /// hop 2, of which 398 aged out and none was fetched.
    ///
    /// The dropper chain `--fetch-depth 2` exists for runs through URLs and
    /// install-command packages, and those are followed at every hop regardless.
    /// `serve`/`worker` set this because they are cache-population roles, where
    /// the transitive tail is the point rather than an overhead.
    pub transitive_deps: bool,
    /// Skip fetching a *dependency* whose name pins it to a platform other than
    /// the host — the `@scope/pkg-<os>-<arch>` native-binary packages (biome,
    /// esbuild, swc, rollup, sharp…) that ship one prebuilt per platform. On a
    /// darwin-arm64 host only the darwin-arm64 variant is pulled; the linux and
    /// windows siblings cannot run locally. `false` audits every platform,
    /// which service/corpus roles enable because they scan on behalf of others.
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
            ci: false,
            depth: DEFAULT_FETCH_DEPTH,
            max_dep_age_days: DEFAULT_MAX_DEP_AGE_DAYS,
            max_file_fetches: DEFAULT_MAX_FILE_FETCHES,
            max_url_fetches: DEFAULT_MAX_URL_FETCHES,
            max_file_bytes: DEFAULT_MAX_FILE_SIZE,
            transitive_deps: false,
            host_platform_only: true,
        }
    }
}

impl FetchPolicy {
    /// True when at least one kind is selected — the master switch.
    #[must_use]
    pub(crate) const fn enabled(&self) -> bool {
        self.urls || self.packages || self.deps || self.ci
    }

    /// Parse the customer-facing `follow` vocabulary. Unlike [`FromStr`], this
    /// deliberately rejects the old CLI aliases so the HTTP contract has one
    /// clear spelling for each concept.
    pub(crate) fn parse_follow(value: &str) -> Result<Self, String> {
        Self::parse_selection(value, false)
    }

    /// Copy only the selected reference kinds onto an operator-configured
    /// policy, preserving depth, age, byte, platform, and fan-out ceilings.
    #[must_use]
    pub(crate) const fn with_selection(mut self, selected: Self) -> Self {
        self.urls = selected.urls;
        self.packages = selected.packages;
        self.deps = selected.deps;
        self.ci = selected.ci;
        self
    }

    /// Compact identity for single-flight keys and structured logs.
    #[must_use]
    pub(crate) const fn selection_bits(&self) -> u8 {
        (self.urls as u8)
            | ((self.packages as u8) << 1)
            | ((self.deps as u8) << 2)
            | ((self.ci as u8) << 3)
    }

    /// This selection in the customer-facing `follow` vocabulary, when it has
    /// a name there.
    ///
    /// The inverse of [`Self::parse_follow`], and it exists so a caller never
    /// has to guess which policy produced an answer: whoever files the verdict
    /// files it under the name returned here, and a name that disagreed with
    /// the analysis would file it under the wrong question.
    ///
    /// `None` for a selection the vocabulary cannot spell. `references` moves
    /// `urls` and `packages` together, so the legacy `--follow=urls` alias can
    /// set one without the other and leave a policy with no customer word for
    /// it. Saying nothing is right there: an approximate name is worse than an
    /// absent one, because the absent one falls back to the caller's own
    /// resolution while the approximate one silently misfiles.
    #[must_use]
    pub(crate) fn follow_name(&self) -> Option<String> {
        if self.urls != self.packages {
            return None;
        }
        let references = self.urls;
        if !references && !self.deps && !self.ci {
            return Some("none".to_owned());
        }
        if references && self.deps && self.ci {
            return Some("all".to_owned());
        }
        // Spelled in the order the customer vocabulary lists them, so one
        // policy has exactly one name and a cache keyed by that name does not
        // split on word order.
        let mut parts = Vec::with_capacity(3);
        if self.deps {
            parts.push("dependencies");
        }
        if references {
            parts.push("references");
        }
        if self.ci {
            parts.push("ci-actions");
        }
        Some(parts.join(","))
    }

    fn parse_selection(value: &str, legacy_aliases: bool) -> Result<Self, String> {
        const VALID: &str = "valid: all, dependencies, references, ci-actions, none";
        let mut policy = Self::default();
        let mut saw_kind = false;
        let mut saw_none = false;

        for raw in value.split(',') {
            let kind = raw.trim();
            if kind.is_empty() {
                continue;
            }
            saw_kind = true;
            match kind {
                "none" => saw_none = true,
                "all" => {
                    policy.urls = true;
                    policy.packages = true;
                    policy.deps = true;
                    policy.ci = true;
                }
                "dependencies" => policy.deps = true,
                "references" => {
                    policy.urls = true;
                    policy.packages = true;
                }
                // A CI action is represented as a dependency with CI context,
                // so selecting actions necessarily enables dependency traversal.
                "ci-actions" => {
                    policy.deps = true;
                    policy.ci = true;
                }
                "deps" if legacy_aliases => policy.deps = true,
                "packages" if legacy_aliases => policy.packages = true,
                "urls" if legacy_aliases => policy.urls = true,
                "ci" if legacy_aliases => {
                    policy.deps = true;
                    policy.ci = true;
                }
                other => return Err(format!("unknown follow target {other:?} ({VALID})")),
            }
        }

        if !saw_kind {
            return Err(format!("empty follow selection ({VALID})"));
        }
        if saw_none && policy.enabled() {
            return Err("none cannot be combined with another follow target".to_string());
        }
        if saw_none {
            return Ok(Self::default());
        }
        if !policy.enabled() {
            return Err(format!("empty follow selection ({VALID})"));
        }
        Ok(policy)
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

    /// Whether `kind` is selected on hop `hop` (0-based). Identical to
    /// [`Self::wants`] except that declared dependencies stop at the first hop
    /// unless [`Self::transitive_deps`] is set — see that field for why.
    #[must_use]
    fn wants_at(&self, kind: RefKind, hop: u8) -> bool {
        self.wants(kind) && (self.transitive_deps || hop == 0 || kind != RefKind::Dependency)
    }
}

impl std::str::FromStr for FetchPolicy {
    type Err = String;

    /// Parse the canonical `follow` vocabulary and the legacy CLI aliases.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_selection(s, true)
    }
}

/// Canonicalize an OS name segment to its npm token (`process.platform`).
/// Accepts the npm spelling plus the Rust/Go target spellings that appear in
/// cargo platform crates (`windows_x86_64_gnu`) and Go module paths, so a
/// match keys on a genuine `<os>-<arch>` native-binary name rather than an
/// incidental word. `None` for anything that names no OS.
fn canonical_os(seg: &str) -> Option<&'static str> {
    Some(match seg {
        "darwin" | "macos" => "darwin",
        "win32" | "windows" => "win32",
        "sunos" | "solaris" | "illumos" => "sunos",
        "linux" => "linux",
        // musl is its own platform: sharp/libvips ship separate `linuxmusl`
        // prebuilts, and neither libc's binaries load on the other's host.
        "linuxmusl" | "musllinux" => "linuxmusl",
        // StackBlitz-style wasm sandbox builds; never a scan host.
        "webcontainers" | "wasi" => "webcontainers",
        "freebsd" => "freebsd",
        "openbsd" => "openbsd",
        "netbsd" => "netbsd",
        "android" => "android",
        "aix" => "aix",
        _ => return None,
    })
}

/// Canonicalize a CPU-architecture segment to its npm token (`process.arch`).
/// Same vocabulary rule as [`canonical_os`].
fn canonical_arch(seg: &str) -> Option<&'static str> {
    Some(match seg {
        "x64" | "x8664" | "amd64" => "x64",
        "arm64" | "aarch64" => "arm64",
        "ia32" | "i686" | "i386" | "x86" => "ia32",
        "arm" => "arm",
        "ppc64" => "ppc64",
        "s390x" => "s390x",
        "riscv64" => "riscv64",
        "loong64" => "loong64",
        "mips64el" => "mips64el",
        // Wasm sandbox builds pair with an os token (`freebsd-wasm32`,
        // `webcontainers-wasm32`) and never match a real host arch.
        "wasm32" | "wasm64" => "wasm32",
        _ => return None,
    })
}

/// The host's npm-style `(os, arch)` tokens, mapped from Rust's target
/// constants. An unmapped target yields an empty token, which disables that
/// half of the platform match — fail open, so a dependency is never skipped on a
/// host we can't confidently name.
fn host_platform() -> (&'static str, &'static str) {
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        "solaris" | "illumos" => "sunos",
        // A musl build (Alpine workers) is its own platform: glibc prebuilts
        // don't load there and musl prebuilts don't load on glibc hosts, and
        // native packages ship separate `linuxmusl` variants (sharp/libvips).
        "linux" if cfg!(target_env = "musl") => "linuxmusl",
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
    let mut segs: Vec<String> = name
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase)
        .collect();
    // `x86_64` splits at its underscore; re-join the pair so cargo/Go names
    // (`windows_x86_64_gnu`, `linux_x86_64`) carry one arch token.
    let mut i = 0;
    while i + 1 < segs.len() {
        if segs[i] == "x86" && segs[i + 1] == "64" {
            segs[i] = "x8664".to_string();
            segs.remove(i + 1);
        }
        i += 1;
    }
    // An adjacent os+arch pair (either order) marks a platform-specific package;
    // skip it when either token disagrees with the host.
    for w in segs.windows(2) {
        let (os, arch) = match (canonical_os(&w[0]), canonical_arch(&w[1])) {
            (Some(os), Some(arch)) => (os, arch),
            _ => match (canonical_arch(&w[0]), canonical_os(&w[1])) {
                (Some(arch), Some(os)) => (os, arch),
                _ => continue,
            },
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

/// URL suffixes that are normally pages, API responses, or other site
/// resources rather than a payload a dropper would retrieve.
const NON_PAYLOAD_URL_EXTENSIONS: &[&str] = &[
    "asp",
    "aspx",
    "atom",
    "avif",
    "bmp",
    "cfm",
    "cgi",
    "css",
    "csv",
    "dtd",
    "gif",
    "htm",
    "html",
    "ico",
    "jpeg",
    "jpg",
    "json",
    "log",
    "md",
    "pdf",
    "php",
    "png",
    "rss",
    "svg",
    "txt",
    "webmanifest",
    "webp",
    "xml",
    "xhtml",
    "yaml",
    "yml",
];

/// Path components that make a URL explicitly file/download-shaped. These
/// allow extensionless payload names and file routes on otherwise API-shaped
/// hosts, while a bare `/download` still fails the basename check.
const DOWNLOAD_URL_PATH_COMPONENTS: &[&str] = &[
    "archive",
    "archives",
    "attachment",
    "attachments",
    "blob",
    "download",
    "downloads",
    "file",
    "files",
    "raw",
    "release",
    "releases",
    "resolve",
];

/// Path components that are strong signs of an API or service endpoint.
const API_URL_PATH_COMPONENTS: &[&str] = &[
    "api", "graphql", "health", "lookup", "metrics", "oauth", "query", "rpc", "search", "status",
    "token",
];

/// Whether a discovered URL looks enough like a dropper download to spend a
/// network request on it.
///
/// This is deliberately a shape check, not a content or reputation check:
/// direct scans still fetch exactly what the operator names, and a URL with a
/// plausible payload basename remains eligible even when its host is unknown.
/// Whether a path component reads as a version rather than a filename: every
/// dot-separated segment is digits, with at least one dot and an optional
/// leading `v` (`0.40.0`, `v2.1`, `10.0.1`). A bare `v1` or a plain number has
/// no dot and keeps whatever the surrounding rules decide.
///
/// Deliberately strict about the tail. Recognizing a pre-release suffix as part
/// of the version means splitting at `-`, which throws away everything after —
/// including a real extension. A Go module's
/// `v0.0.0-20260823143148-1fb3b878e2fb.zip` then reads as version `0.0.0` and
/// the artifact stops being fetched. Requiring every segment to be numeric can
/// only ever miss a version, never swallow a file: anything ending in an
/// alphabetic extension fails the test by construction.
fn is_version_shaped(component: &str) -> bool {
    let core = component.strip_prefix(['v', 'V']).unwrap_or(component);
    core.contains('.')
        && core
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

/// Whether a discovered URL has a real network host. URL extraction also sees
/// relative paths, malformed authority strings, and single-label local names
/// such as `wpad`; none can identify a public download host. IP literals are
/// valid, while DNS names must have at least two labels.
fn valid_discovered_url_host(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    if !matches!(parsed.scheme(), "http" | "https") {
        return false;
    }
    let Some(host) = parsed.host_str() else {
        return false;
    };
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return public_ip(ip);
    }

    let host = host.strip_suffix('.').unwrap_or(host);
    host.len() <= 253
        && host.contains('.')
        && host.split('.').all(|label| {
            let bytes = label.as_bytes();
            !bytes.is_empty()
                && bytes.len() <= 63
                && bytes[0].is_ascii_alphanumeric()
                && bytes[bytes.len() - 1].is_ascii_alphanumeric()
                && bytes
                    .iter()
                    .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'-')
        })
}

/// Whether a discovered URL still contains a source-template placeholder.
/// These commonly appear in documentation or client code as repository and
/// release examples (`$REPO`, `${this.repositoryId}`, `$VERSION`); fetching
/// them can only produce an avoidable 4xx response.
fn has_unexpanded_url_placeholder(url: &str) -> bool {
    let contains_placeholder = |part: &str| {
        let bytes = part.as_bytes();
        bytes.windows(2).any(|window| {
            window[0] == b'$'
                && (window[1] == b'{' || window[1].is_ascii_alphabetic() || window[1] == b'_')
        })
    };
    if contains_placeholder(url) {
        return true;
    }
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    [Some(parsed.path()), parsed.query()]
        .into_iter()
        .flatten()
        .any(contains_placeholder)
}

/// Whether an IP literal is publicly routable enough to justify a discovered
/// fetch. This excludes RFC1918/private space and the other special-use ranges
/// that describe the scanner's host, a lab network, or documentation rather
/// than an external payload service.
fn public_ip(ip: std::net::IpAddr) -> bool {
    if let std::net::IpAddr::V6(ipv6) = ip
        && let Some(ipv4) = ipv6.to_ipv4()
    {
        return public_ip(std::net::IpAddr::V4(ipv4));
    }
    match ip {
        std::net::IpAddr::V4(ip) => {
            let octets = ip.octets();
            !ip.is_private()
                && !ip.is_loopback()
                && !ip.is_link_local()
                && !ip.is_unspecified()
                && !ip.is_broadcast()
                && !ip.is_multicast()
                && !(octets[0] == 100 && (64..=127).contains(&octets[1]))
                && !(octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
                && !(octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
                && !(octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
                && !(octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
                && !(octets[0] == 198 && (18..=19).contains(&octets[1]))
                && !(octets[0] == 192 && octets[1] == 88 && octets[2] == 99)
                && octets[0] < 224
        }
        std::net::IpAddr::V6(ip) => {
            let segments = ip.segments();
            !ip.is_loopback()
                && !ip.is_unspecified()
                && !ip.is_multicast()
                && (segments[0] & 0xfe00) != 0xfc00
                && (segments[0] & 0xffc0) != 0xfe80
                && (segments[0] & 0xffc0) != 0xfec0
                && !(segments[0] == 0x2001 && segments[1] == 0x0db8)
        }
    }
}

fn looks_like_dropper_download_url(url: &str) -> bool {
    let Some((scheme, rest)) = url.split_once("://") else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return false;
    }

    // Stop at the first path/query/fragment delimiter. A URL with no path is
    // a site or API root, not a downloadable file.
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    if authority_end == 0 {
        return false;
    }
    let Some(path_and_suffix) = rest.get(authority_end..) else {
        return false;
    };
    let path = path_and_suffix.split(['?', '#']).next().unwrap_or_default();
    // A path ending in `/` names a directory, not a file: whatever a server
    // returns for it is an index or a landing page, never the download itself.
    // `https://pypi.org/project/diffusers/0.40.0/` was being fetched as a
    // payload because the trailing component parsed as a filename.
    if path.ends_with('/') {
        return false;
    }
    let components: Vec<&str> = path
        .split('/')
        .filter(|component| !component.is_empty())
        .collect();
    let Some(filename) = components.last().copied() else {
        return false;
    };
    if filename == "." || filename == ".." {
        return false;
    }
    // A version is not a file. `0.40.0`, `v2.1`, `1.2.3-rc1` all end in what
    // looks like an extension, so the dotted-basename test reads them as
    // downloads and pulls project pages, release-tag pages, and API version
    // roots. Nothing named this way is an artifact.
    if is_version_shaped(filename) {
        return false;
    }

    let filename_lower = filename.to_ascii_lowercase();
    let has_dot = filename_lower.contains('.');
    let extension = filename_lower
        .rsplit_once('.')
        .and_then(|(stem, extension)| {
            (!stem.is_empty() && !extension.is_empty()).then_some(extension)
        });
    let has_payload_extension = extension.is_some_and(|extension| {
        extension.len() <= 12
            && extension.bytes().all(|byte| byte.is_ascii_alphanumeric())
            && !NON_PAYLOAD_URL_EXTENSIONS.contains(&extension)
    });
    let explicit_download_path = components.iter().any(|component| {
        DOWNLOAD_URL_PATH_COMPONENTS.contains(&component.to_ascii_lowercase().as_str())
    });
    let extensionless_download = !has_dot && explicit_download_path && components.len() >= 2;

    // A basename with a plausible extension is enough; an extensionless name
    // needs an explicit file/download route and at least one component before
    // the basename (`/download` itself is still just an endpoint).
    if !has_payload_extension && !extensionless_download {
        return false;
    }

    // An API host or endpoint with a file-shaped response is still allowed
    // when the URL says it is fetching a file. This keeps routes such as
    // `/releases/download/...`, `/raw/...`, and Telegram-style `/file/...`
    // eligible while dropping `/api/v1/models`, `/graphql`, and similar
    // service calls (which already fail the basename test in most cases).
    let api_host = hosts::host_of(url)
        .split('.')
        .next()
        .is_some_and(|label| label.eq_ignore_ascii_case("api"));
    let api_path = components.iter().any(|component| {
        API_URL_PATH_COMPONENTS.contains(&component.to_ascii_lowercase().as_str())
    });
    !(api_host || api_path) || explicit_download_path
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
}

/// Registry provenance materialized while walking a dependency graph.
///
/// This is separate from [`FetchedDependency`] because a dependency can have
/// notable registry findings even when its artifact is age-gated or removed and
/// therefore never fetched. `file_id` ties the record to the exact sidecar node
/// cleave analyzed, including composite-source attribution.
#[derive(Debug, Clone)]
pub(crate) struct DependencyRegistry {
    pub locator: String,
    pub provenance: crate::provenance::RegistryProvenance,
    pub file_id: u32,
    pub artifact_skip: Option<&'static str>,
}

/// Discover, fetch, and graft, following references up to `policy.depth` hops.
/// Mutates `report.files` in place with one node per fetched payload (and any
/// extracted members) and returns the fetch edge log plus the standalone report
/// captured for each fetched dependency. A disabled policy, an unavailable
/// cache/client, or zero references all yield empty logs.
///
/// `capture_deps` controls whether each fetched dependency's standalone report
/// is serialized and returned as a [`FetchedDependency`]. Those captures exist
/// only for hopper uploads and the dependency appendix of text/LLM renders — a
/// plain JSON scan with no upload target drops them unread, so skipping the
/// capture (and the downstream re-parse + per-dep model pass it feeds) is pure
/// saved work. Grafting, verdicts, and fetch edges are unaffected.
pub(crate) fn orchestrate(
    report: &mut AnalysisReport,
    root_path: &Path,
    policy: FetchPolicy,
    progress: bool,
    capture_deps: bool,
    zip_passwords: &[String],
) -> (
    Vec<FetchRecord>,
    Vec<FetchedDependency>,
    Vec<DependencyRegistry>,
) {
    if !policy.enabled() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let Some(res) = shared_resources() else {
        return (Vec::new(), Vec::new(), Vec::new());
    };

    // Analyze fetched payloads with the same bloom short-circuit the top-level
    // scan uses, so a trusted binary shipped inside a dependency isn't needlessly
    // re-disassembled; and memoize the whole analysis by content sha, so a warm
    // re-run reuses it rather than repeating a minutes-long pass.
    cleave::set_compact_member_retention(true); // compact projection only
    let mut opts = AnalysisOptions {
        skip_predicate: dep_skip_predicate(),
        ..AnalysisOptions::default()
    };
    crate::engine::add_zip_passwords(&mut opts, zip_passwords);
    // Opened lazily on the first payload actually analyzed. Opening it derives
    // the ruleset-version namespace, which calls `cleave::version_info` — and that
    // spins up the YARA engine just to count rules. A scan that fetches nothing
    // (every reference age-gated or none present, the common `pkg:` case) must
    // not pay that: `None` here means "not yet opened".
    let mut acache: Option<Option<AnalysisCache>> = None;
    let mut records = Vec::new();
    // (declaring file sha) -> (registry records materialized, of which security-held)
    let mut registry_outcomes: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    // Standalone reports for each fetched dependency, captured before the payload
    // is grafted into the merged report. Uploaded to hopper as their own samples.
    let mut dependencies: Vec<FetchedDependency> = Vec::new();
    let mut dependency_registries: Vec<DependencyRegistry> = Vec::new();
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
    let mut worklist = collect_references(
        report,
        root_path,
        if policy.ci {
            CiRefs::Include
        } else {
            CiRefs::Skip
        },
    );
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
    // Newest-version gate: only the most recent version of a package in the
    // dependency tree is fetched and analyzed. Deep ungated trees pin dozens
    // of releases of the same packages (syn ×27, libc ×21 on the mx crate
    // benchmark — 48% of its tree was older duplicates); analyzing every
    // pinned release repeats near-identical work without detection value the
    // newest release doesn't provide. Versionless references are exempt (they
    // already resolve to the latest release), and the kept version is monotone
    // across hops: once a release is scanned, an older sibling discovered in a
    // later hop never resurrects the package.
    let mut newest_seen: HashMap<String, String> = HashMap::new();
    // Coordinates (`pkg:eco/name`) for which a version-pinned reference exists
    // anywhere in the tree. A manifest range (`"puppeteer": "^10.4.0"`) reaches
    // us version-stripped as a bare `pkg:npm/puppeteer` and would resolve to
    // `dist-tags/latest` — a version the project never installs. When the
    // co-located lockfile also pins the coordinate (`pkg:npm/puppeteer@10.4.2`),
    // that pin is ground truth and must win, so the bare sibling is dropped
    // below. Monotone across hops, mirroring `newest_seen`.
    let mut pinned_coords: HashSet<String> = HashSet::new();
    // source content-sha → the declaring manifest's path (relative to the root
    // artifact), so each dependency row can name the file it came from. Built
    // once from the report's own files; a source discovered only inside a fetched
    // payload (a deeper hop) simply isn't found and stays unnamed.
    let source_manifests: HashMap<String, String> = report
        .files
        .iter()
        .map(|f| (f.sha256.clone(), manifest_relpath(&f.path)))
        .collect();
    for hop in 0..policy.depth {
        if worklist.is_empty() {
            break;
        }
        // Pre-scan the whole hop so the winner is hop-wide, not
        // first-group-wins: every fetchable versioned reference bids, and the
        // running cross-hop maximum only rises.
        for (_, refs) in &worklist {
            for r in refs {
                if !policy.wants_at(r.kind, hop) {
                    continue;
                }
                let locator = locator_key(r);
                let Some((key, version)) = versioned_purl(&locator) else {
                    continue;
                };
                // This coordinate is pinned somewhere; its bare sibling loses.
                pinned_coords.insert(key.to_string());
                match newest_seen.get(key) {
                    Some(best)
                        if lenient_version_cmp(version, best) != std::cmp::Ordering::Greater => {}
                    _ => {
                        newest_seen.insert(key.to_string(), version.to_string());
                    }
                }
            }
        }
        let dropped_old_versions = std::cell::Cell::new(0usize);
        let mut next = Vec::new();
        // The declaring-file groups in a hop are independent until their serial
        // merge, but a deep tree yields many small groups — one per previous-hop
        // payload — and analyzing one group at a time strands most of a large
        // machine on the tail. Groups are therefore processed in batches:
        // selection, gating, and fetching stay serial in group order (`seen`
        // dedup and budget charges keep their exact order), registry-record and
        // payload analysis fan out across the whole batch, and merging replays
        // serially in group order — report ids, and therefore output, are
        // identical to the one-group-at-a-time code this replaces.
        struct GroupWork {
            source_sha: String,
            selected: Vec<Reference>,
            registries: Vec<GatedDep>,
            fetched: Vec<FetchRecord>,
        }
        // Caps the payloads held un-merged at once: enough to keep every core
        // busy across many small groups without holding a whole hop's analyzed
        // reports in memory.
        const BATCH_PAYLOAD_TARGET: usize = 256;
        // A registry record is canonical JSON we serialized ourselves; its
        // signal is entirely `registry.*` value facts and no YARA rule targets
        // it. Disabling YARA here removed ~1400s of system time per scan — the
        // engine's per-analysis setup, paid hundreds of times. Built once per
        // hop: cloning `AnalysisOptions` per record cost more user time than
        // the YARA saving returned.
        cleave::set_compact_member_retention(true); // compact projection only
        let registry_opts = AnalysisOptions {
            disable_yara: true,
            ..opts.clone()
        };
        let mut groups = std::mem::take(&mut worklist).into_iter().peekable();
        while groups.peek().is_some() {
            let mut batch: Vec<GroupWork> = Vec::new();
            let mut batch_payloads = 0usize;
            while batch_payloads < BATCH_PAYLOAD_TARGET {
                let Some((source_sha, refs)) = groups.next() else {
                    break;
                };
                // Keep only the kinds this policy selected — by RefKind, so a
                // command-mentioned package (`packages`) is distinct from a declared
                // dependency (`deps`) even though both are PURLs — and that haven't
                // been fetched yet this run.
                let selected: Vec<Reference> = refs
                    .into_iter()
                    .filter(|r| {
                        let wanted = policy.wants_at(r.kind, hop);
                        if !wanted && policy.wants(r.kind) {
                            tracing::debug!(
                                package = %locator_key(r),
                                hop = hop + 1,
                                "transitive dependency; registry lookup and fetch both skipped"
                            );
                        }
                        wanted
                    })
                    // Drop publisher-controlled URLs, obvious site/API
                    // endpoints, and the exact documentation/update URLs
                    // observed in stock /bin binaries. They cost a round trip
                    // each and are unlikely to yield a dropper payload.
                    // Applied per hop so a payload's own boilerplate is
                    // filtered too, and only to *discovered* references:
                    // `scan url <url>` fetches whatever the operator names
                    // (see `crate::hosts`).
                    .filter(|r| {
                        let RefLocator::Url(url) = &r.locator else {
                            return true;
                        };
                        if !valid_discovered_url_host(url) {
                            tracing::debug!(
                                url = %url,
                                source = %r.source,
                                "invalid or local URL host; fetch skipped"
                            );
                            return false;
                        }
                        if has_unexpanded_url_placeholder(url) {
                            tracing::debug!(
                                url = %url,
                                source = %r.source,
                                "unexpanded URL template; fetch skipped"
                            );
                            return false;
                        }
                        let boilerplate = hosts::publisher_controlled(url)
                            || hosts::discovery_exception(url);
                        if boilerplate {
                            tracing::debug!(
                                url = %url,
                                source = %r.source,
                                "known boilerplate URL; fetch skipped"
                            );
                            return false;
                        }
                        if r.kind == RefKind::UrlFetch && !looks_like_dropper_download_url(url) {
                            tracing::debug!(
                                url = %url,
                                source = %r.source,
                                "URL does not look like a dropper download; fetch skipped"
                            );
                            return false;
                        }
                        true
                    })
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
                    // Older-version duplicates are skipped entirely — no
                    // registry-record materialization, no fetch (operator
                    // policy 2, 2026-07-30). Never silent: each skip logs at
                    // debug, the hop logs one count at info.
                    .filter(|r| {
                        let locator = locator_key(r);
                        let Some((key, version)) = versioned_purl(&locator) else {
                            return true;
                        };
                        let newest = !matches!(
                            newest_seen.get(key),
                            Some(best) if lenient_version_cmp(version, best)
                                == std::cmp::Ordering::Less
                        );
                        if !newest {
                            dropped_old_versions.set(dropped_old_versions.get() + 1);
                            tracing::debug!(
                                package = %locator,
                                newest = %newest_seen[key],
                                "older version of an already-kept package; skipped (newest-version policy)"
                            );
                        }
                        newest
                    })
                    // A lockfile pin supersedes the manifest's versionless
                    // sibling: drop a bare `pkg:eco/name` when the same
                    // coordinate is pinned elsewhere in the tree, so the exact
                    // installed version is scanned instead of `dist-tags/latest`.
                    .filter(|r| {
                        if superseded_by_pin(r, &pinned_coords) {
                            tracing::debug!(
                                package = %locator_key(r),
                                "versionless dependency superseded by a lockfile-pinned sibling; skipped"
                            );
                            return false;
                        }
                        true
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
                reporter.announce(
                    &selected,
                    source_manifests.get(&source_sha).map_or("", String::as_str),
                );
                if selected.is_empty() {
                    // Nothing to fetch; the group still merges its registry
                    // records below.
                    batch.push(GroupWork {
                        source_sha,
                        selected,
                        registries,
                        fetched: Vec::new(),
                    });
                    continue;
                }
                // URL fetches have a deliberately smaller independent fan-out cap
                // than declared dependencies and command-mentioned packages. Split
                // the groups so fletch's per-call count budget can enforce both
                // ceilings without making one category consume the other's slots.
                let (url_selected, dep_selected): (Vec<Reference>, Vec<Reference>) = selected
                    .iter()
                    .cloned()
                    .partition(|r| r.kind == RefKind::UrlFetch);

                let dep_budget_notice = fetch_count_budget_notice(
                    "--fetch-max-file-fetches",
                    policy.max_file_fetches,
                    TOTAL_FETCH_COUNT.load(Ordering::Relaxed),
                );

                // Mark the to-fetch set in flight, then fetch. The callback fires as
                // each download lands (from a pool worker, so it's `Sync`), flipping
                // that row to "analyzing" the moment its bytes arrive rather than when
                // the whole concurrent batch returns — keyed on the original
                // reference, since a versionless locator may be refined during fetch.
                reporter.fetching(&selected);
                let on_fetched = |r: &Reference, rec: &FetchRecord| reporter.landed(r, rec);
                let mut file_bytes_remaining = policy.max_file_bytes;

                // Dependencies/packages get the larger cap. Charge this group before
                // starting URLs so the process-wide budget is also respected between
                // the two independent fletch calls.
                let dep_budget = FetchBudget {
                    max_count: policy
                        .max_file_fetches
                        .min(TOTAL_FETCH_COUNT.load(Ordering::Relaxed)),
                    max_bytes: file_bytes_remaining.min(TOTAL_FETCH_BYTES.load(Ordering::Relaxed)),
                };
                // Pre-fetch PURL negotiation: hopper may hold a standing
                // verdict (same rules as the per-sha corpus precheck) for a
                // registry dependency whose content sha we would otherwise
                // only learn by downloading it. One batched lookup up front
                // skips the download, the analysis, and the re-upload for
                // every such PURL; the answer's sha still records the fetch
                // edge. Anything unanswered fetches exactly as before.
                let purl_candidates: Vec<String> = dep_selected
                    .iter()
                    .filter(|r| {
                        matches!(r.locator, RefLocator::Purl(_)) && r.content_sha256.is_none()
                    })
                    .map(|r| locator_str(&r.locator))
                    .collect();
                let corpus_hits = crate::corpus_precheck::precheck_purls(&purl_candidates);
                let dep_to_fetch: Vec<Reference> = dep_selected
                    .iter()
                    .filter(|r| !corpus_hits.contains_key(&locator_str(&r.locator)))
                    .cloned()
                    .collect();
                let dep_fetched_real = fetch_references_with(
                    &dep_to_fetch,
                    &source_sha,
                    false,
                    &res.net,
                    &res.cache,
                    dep_budget,
                    &on_fetched,
                );
                // Reassemble in dep_selected order: downstream zips records
                // against the selected refs positionally.
                let mut real_iter = dep_fetched_real.into_iter();
                let dep_fetched: Vec<FetchRecord> = dep_selected
                    .iter()
                    .filter_map(|r| match corpus_hits.get(&locator_str(&r.locator)) {
                        Some(sha) => Some(corpus_hit_record(r, &source_sha, sha)),
                        // fletch emits one record per reference it selects as a
                        // fetch target; anything it declines has no record and
                        // drops out here, exactly as in the regrouping below.
                        None => real_iter.next(),
                    })
                    .collect();
                if !corpus_hits.is_empty() {
                    tracing::info!(
                        skipped = corpus_hits.len(),
                        asked = purl_candidates.len(),
                        "purl precheck: hopper verdicts stand; skipped fetch+analysis+upload"
                    );
                }
                let (dep_spent, dep_bytes) = live_fetch_usage(&dep_fetched);
                charge_total_budget(dep_spent, dep_bytes);
                file_bytes_remaining = file_bytes_remaining.saturating_sub(dep_bytes);

                // Opportunistic raw URLs get their own, smaller cap. A URL that is
                // expressed as a declared dependency or command-mentioned package
                // was kept in `dep_selected` and therefore follows the 100 cap.
                let url_budget = FetchBudget {
                    max_count: policy
                        .max_url_fetches
                        .min(TOTAL_FETCH_COUNT.load(Ordering::Relaxed)),
                    max_bytes: file_bytes_remaining.min(TOTAL_FETCH_BYTES.load(Ordering::Relaxed)),
                };
                let url_budget_notice = fetch_count_budget_notice(
                    "--fetch-max-urls",
                    policy.max_url_fetches,
                    TOTAL_FETCH_COUNT.load(Ordering::Relaxed),
                );
                let url_fetched = fetch_references_with(
                    &url_selected,
                    &source_sha,
                    true,
                    &res.net,
                    &res.cache,
                    url_budget,
                    &on_fetched,
                );
                let (url_spent, url_bytes) = live_fetch_usage(&url_fetched);
                charge_total_budget(url_spent, url_bytes);

                // Reassemble in the original declaration order. Each fletch call
                // preserves order within its category, and the two iterators restore
                // the order expected by the analysis/grafting pass below.
                let mut dep_iter = dep_fetched.into_iter();
                let mut url_iter = url_fetched.into_iter();
                let mut fetched = Vec::with_capacity(selected.len());
                for r in &selected {
                    if r.kind == RefKind::UrlFetch {
                        if let Some(record) = url_iter.next() {
                            fetched.push(record);
                        }
                    } else if let Some(record) = dep_iter.next() {
                        fetched.push(record);
                    }
                }
                debug_assert_eq!(
                    fetched.len(),
                    selected.len(),
                    "grouped fetch results must align with selected refs"
                );

                // Authoritative pass over every returned edge: settle each row (the
                // tree finalizes any budget-clipped edge the live callback never saw;
                // re-settling a callback-landed row is idempotent) and print the
                // streamed line. `selected` and `fetched` align one-to-one and in
                // order — every selected reference is a fetch target, so fletch emits
                // exactly one record per reference in declaration order.
                for (r, rec) in selected.iter().zip(&fetched) {
                    reporter.landed(r, rec);
                    let budget_notice = if r.kind == RefKind::UrlFetch {
                        &url_budget_notice
                    } else {
                        &dep_budget_notice
                    };
                    reporter.report(rec, Some(budget_notice));
                }

                // Benchmark escape hatch: stop after the network phase, before the
                // (far more expensive) analysis of what was pulled. Fetch tuning —
                // depth, kind selection, age gating, concurrency — is about what we
                // retrieve, and re-analyzing every payload to measure that turns a
                // sub-minute experiment into a long one. Reports what was fetched so
                // a run is still comparable, then exits non-analyzing. Earlier
                // groups in this batch contribute their edges too; their registry
                // nodes — exactly the analysis this mode skips — do not merge.
                if std::env::var("SCAN_FETCH_ONLY").as_deref() == Ok("1") {
                    let bytes: u64 = fetched.iter().filter_map(|r| r.size).sum();
                    tracing::info!(
                        refs_selected = selected.len(),
                        payloads_fetched = fetched.iter().filter(|r| r.size.is_some()).count(),
                        fetched_bytes = bytes,
                        "SCAN_FETCH_ONLY: stopping before payload analysis"
                    );
                    println!(
                        "fetch-only: selected={} fetched={} bytes={}",
                        selected.len(),
                        fetched.iter().filter(|r| r.size.is_some()).count(),
                        bytes
                    );
                    for g in batch {
                        records.extend(g.fetched);
                    }
                    records.extend(fetched);
                    reporter.finish(&records);
                    return (records, dependencies, dependency_registries);
                }
                batch_payloads += fetched.iter().filter(|r| delivered_bytes(r)).count();
                batch.push(GroupWork {
                    source_sha,
                    selected,
                    registries,
                    fetched,
                });
            }

            // Analyze the batch. Registry records and fetched payloads are both
            // report-independent (`registry_node` and `analyze_payload` are pure
            // with respect to the report), so they fan out together across every
            // group in the batch. Each `registry_node` is an independent cleave
            // analysis of one small JSON document at ~23 ms; a payload is a full
            // cleave pass. The callback settles each payload row from
            // "analyzing" to its final glyph as its scan finishes.
            let acache_ref = acache
                .get_or_insert_with(crate::analysis_cache::AnalysisCache::open)
                .as_ref();
            // Per-group results, aligned with `batch`.
            type BatchRegistrySubs = Vec<Vec<Option<AnalysisReport>>>;
            type BatchAnalyzed = Vec<Vec<Option<Analyzed>>>;
            // Both halves dispatch full cleave analyses, so they fan out only
            // while the pool has headroom (see `payload_fanout_allowed`).
            // Saturated, the batch runs inline on this blocking thread — the
            // shape cleave's own nesting throttle is written for.
            let registries_of = |g: &GroupWork| -> Vec<Option<AnalysisReport>> {
                g.registries
                    .iter()
                    .map(|(_, provenance, _)| registry_node(&provenance.record, &registry_opts))
                    .collect()
            };
            let payloads_of = |g: &GroupWork| -> Vec<Option<Analyzed>> {
                let on_analyzed = |i: usize| {
                    if let (Some(r), Some(rec)) = (g.selected.get(i), g.fetched.get(i)) {
                        reporter.analyzed(r, rec);
                    }
                };
                analyze_payloads(&g.fetched, &res.cache, &opts, acache_ref, &on_analyzed)
            };
            let (registry_subs, analyzed): (BatchRegistrySubs, BatchAnalyzed) =
                if payload_fanout_allowed() {
                    use rayon::prelude::*;
                    rayon::join(
                        || batch.par_iter().map(&registries_of).collect(),
                        || batch.par_iter().map(&payloads_of).collect(),
                    )
                } else {
                    (
                        batch.iter().map(&registries_of).collect(),
                        batch.iter().map(&payloads_of).collect(),
                    )
                };

            // Merge serially — groups in order, registry records before payloads
            // within a group, both in materialization order — because
            // `merge_registry`/`merge_payload` assign report ids from a running
            // max; parallelizing the merge would make ids (and therefore output)
            // depend on completion order.
            for (g, (subs, payloads)) in batch
                .into_iter()
                .zip(registry_subs.into_iter().zip(analyzed))
            {
                // Registry findings keyed by locator, captured as each record is
                // merged so the package pass below can pair an artifact with
                // its own registry metadata (see `apply_package_composites`).
                let mut registry_findings: HashMap<String, Vec<Finding>> = HashMap::new();
                for ((r, provenance, skip), sub) in g.registries.iter().zip(subs) {
                    let reg = &provenance.record;
                    if let Some(sub) = sub {
                        let findings = sub_findings(&sub);
                        // Every registry record we materialized for this file is a
                        // reference whose outcome the declarer should carry. Tallied
                        // here rather than from `g.fetched` because the two travel
                        // separately: a dependency resolved without a live download
                        // still yields a registry document, so attributing only from
                        // fetch records left the commonest case with no outcome at
                        // all.
                        let tally = registry_outcomes
                            .entry(g.source_sha.clone())
                            .or_insert((0u64, 0u64));
                        tally.0 += 1;
                        if findings
                            .iter()
                            .any(|f| f.id.contains("registry-security-hold-record"))
                        {
                            tally.1 += 1;
                        }
                        // A skipped dependency has no artifact upload and only
                        // appears in provenance output when its registry node is
                        // notable. Drop its raw provider document immediately when
                        // neither condition applies; lockfiles can contain hundreds
                        // of ordinary aged-out records.
                        let retain_provenance = skip.is_none()
                            || findings
                                .iter()
                                .any(|finding| finding.crit >= cleave::Criticality::Notable);
                        registry_findings
                            .entry(locator_key(r))
                            .or_default()
                            .extend(findings);
                        if let Some(file_id) = merge_registry(report, &g.source_sha, sub)
                            && retain_provenance
                        {
                            let artifact_skip = skip.map(|reason| match reason {
                                SkipReason::Removed => "version removed",
                                SkipReason::AgedOut => "older than fetch age limit",
                                SkipReason::KnownGood => "known-good coordinate",
                            });
                            dependency_registries.push(DependencyRegistry {
                                locator: locator_key(r),
                                provenance: provenance.clone(),
                                file_id,
                                artifact_skip,
                            });
                        }
                    }
                    // The record is materialized either way; only the artifact fetch
                    // is skipped. `None` = kept for fetch+scan.
                    let Some(reason) = skip else {
                        continue;
                    };
                    let common = tracing::field::display(locator_key(r));
                    if *reason == SkipReason::KnownGood {
                        crate::bloom_repo::record(crate::bloom_repo::Decision::Skip, false);
                    }
                    let log_reason = match reason {
                        SkipReason::Removed => "version removed from registry",
                        SkipReason::AgedOut => "older than --max-dep-age",
                        SkipReason::KnownGood => "known-good (bloom, resolved version)",
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
                for (selected_ref, (rec, payload)) in
                    g.selected.iter().zip(g.fetched.iter().zip(payloads))
                {
                    if let Some(mut payload) = payload {
                        // Run registry-aware package composites on the dependency's
                        // standalone report before either consumer takes it. This
                        // lets the dependency grader see the same finding that the
                        // merged parent's embedded-file pass sees, so a suspicious
                        // or hostile dependency can be pinned back to its declaring
                        // manifest. Running this after `merge_payload` left the
                        // standalone capture blind to registry/package composites.
                        prepare_dependency_report(
                            &mut payload,
                            registry_findings_for_reference(&registry_findings, selected_ref),
                            &opts,
                        );
                        // Capture the dependency's standalone report before merge_payload
                        // consumes the sub-report into the merged tree — only when a
                        // consumer (hopper upload, dependency appendix) will read it.
                        if capture_deps && let Some(dep) = capture_dependency(rec, &payload) {
                            dependencies.push(dep);
                        }
                        next.extend(merge_payload(report, rec, payload));
                    }
                }
                records.extend(g.fetched);
            }
        }
        if dropped_old_versions.get() > 0 {
            tracing::info!(
                skipped = dropped_old_versions.get(),
                "older package versions skipped this hop (newest-version policy)"
            );
        }
        worklist = next;
    }
    reporter.finish(&records);
    attribute_reference_outcomes(report, &records, &registry_outcomes);
    (records, dependencies, dependency_registries)
}

/// Record, on each file that declared a reference, what became of the
/// references it declared.
///
/// A resolved payload needs nothing here: `merge_payload` already grafts it
/// into the tree as a child of its declaring file, so its findings are reachable
/// from the declarer. An **unresolved** reference produces no payload and
/// therefore no node — which left the most interesting outcome of a follow the
/// one thing no trait could see.
///
/// It is worth seeing. A manifest naming a dependency the registry no longer
/// serves is pointing at something that was withdrawn, and packages get
/// withdrawn for reasons: the VS Code marketplace pulls extensions for malware,
/// npm unpublishes for the same. The declaring package is often still installed
/// everywhere, still pointing at it.
///
/// Attributed by `source_sha256`, the edge's declaring endpoint, so the facts
/// land on the manifest that made the claim rather than on the archive root.
/// Emitted as ordinary `references.*` metrics and values, so an ordinary
/// file-scoped trait reads them — no new composite scope required.
fn attribute_reference_outcomes(
    report: &mut AnalysisReport,
    records: &[FetchRecord],
    registry_outcomes: &BTreeMap<String, (u64, u64)>,
) {
    #[derive(Default)]
    struct Tally {
        declared: u64,
        unresolved: Vec<String>,
    }

    let mut touched: Vec<String> = Vec::new();
    let mut by_source: BTreeMap<&str, Tally> = BTreeMap::new();
    for rec in records {
        if rec.source_sha256.is_empty() {
            continue;
        }
        let tally = by_source.entry(rec.source_sha256.as_str()).or_default();
        tally.declared += 1;
        if matches!(rec.outcome, Outcome::Unresolved) {
            tally.unresolved.push(rec.locator.clone());
        }
    }

    // One pass, two sources. A reference can leave a fetch record, a registry
    // document, or both, and the declarer should carry its outcome either way --
    // reading only the fetch records meant a dependency resolved without a live
    // download was attributed nothing at all.
    for file in &mut report.files {
        let fetched = by_source.get(file.sha256.as_str());
        let registry = registry_outcomes.get(file.sha256.as_str());
        if fetched.is_none() && registry.is_none() {
            continue;
        }
        let declared = fetched
            .map_or(0, |t| t.declared)
            .max(registry.map_or(0, |r| r.0));
        let unresolved = fetched.map_or(0, |t| t.unresolved.len() as u64);
        let held = registry.map_or(0, |r| r.1);
        let metrics = file.filefacts_metrics.get_or_insert_with(Default::default);
        metrics.insert("references.declared_count".to_string(), declared as f64);
        metrics.insert("references.unresolved_count".to_string(), unresolved as f64);
        metrics.insert("references.security_hold_count".to_string(), held as f64);
        // Editor-marketplace removals get their own count, because they do not
        // mean what a registry 404 means. npm serves a 404 for a private name,
        // a typo, or a package that moved; the VS Code and Open VSX galleries
        // *remove* extensions, and removal is what they do to malware. A rule
        // convicting on the first would be noise and on the second is not.
        //
        // Two fixed keys rather than one per ecosystem: the metric catalog
        // checks exact names, so a key built from whatever PURL type happened
        // to appear could never be declared, and an undeclared key validates
        // against nothing. An archive member also carries no values tree for a
        // `type: value` list to read, so a metric is the only surface that
        // survives member retention.
        let extension_unresolved = fetched.map_or(0, |t| {
            t.unresolved
                .iter()
                .filter(|l| l.starts_with("pkg:vscode/") || l.starts_with("pkg:openvsx/"))
                .count()
        });
        metrics.insert(
            "references.unresolved_extension_count".to_string(),
            extension_unresolved as f64,
        );
        touched.push(file.sha256.clone());
    }

    // Trait evaluation already ran, before the follow phase that produced these
    // facts. Re-run it for just the files whose facts changed, so the rules that
    // read `references.*` get their pass.
    for sha in touched {
        match cleave::graft_reference_outcome_traits(report, &sha, &AnalysisOptions::default()) {
            Ok(0) => {}
            Err(e) => tracing::warn!(sha = %sha, "reference-outcome pass failed: {e:#}"),
            Ok(n) => tracing::debug!(
                grafted = n,
                sha = %sha,
                "reference-outcome traits fired on a declaring file"
            ),
        }
    }
}

/// Where the fetch phase's progress is surfaced.
///
/// `Off` — machine output (JSON/tiny/server): nothing is printed; the edges ride
/// the report. `Stream` — the fetch work is folded into the active scan bar; only
/// actionable per-reference outcomes are logged above it. `Tree` — the live,
/// in-place dependency tree that takes over stderr for an interactive
/// single-artifact scan (see
/// [`crate::deptree`]), listing the whole known set up front and animating each
/// row through its lifecycle.
///
/// The methods take `&self` so the fetch completion callback — invoked
/// concurrently from fletch's pool — can share one reporter with the sequential
/// orchestration.
enum Reporter {
    Off,
    Stream {
        external_dependencies: AtomicU32,
        external_urls: AtomicU32,
        budget_notice: AtomicBool,
    },
    Tree {
        tree: DepTree,
        budget_notice: std::sync::Mutex<Option<String>>,
    },
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
                external_dependencies: AtomicU32::new(0),
                external_urls: AtomicU32::new(0),
                budget_notice: AtomicBool::new(false),
            },
            |tree| Self::Tree {
                tree,
                budget_notice: std::sync::Mutex::new(None),
            },
        )
    }

    /// Reveal a hop's references as pending (tree only) so the whole known set is
    /// visible before any network work begins. `source` is the manifest they were
    /// declared in (package-relative path), shown per row so each dependency can
    /// be traced back to its declaring file.
    fn announce(&self, refs: &[Reference], source: &str) {
        if let Self::Tree { tree, .. } = self {
            for r in refs {
                tree.add(&locator_key(r), &dep_display_name(r), source);
            }
        }
    }

    /// Mark the to-fetch set in flight. The stream folds its counts into the
    /// active scan bar; the tree animates each row in place.
    fn fetching(&self, refs: &[Reference]) {
        match self {
            Self::Stream {
                external_dependencies,
                external_urls,
                ..
            } => {
                let urls = refs.iter().filter(|r| r.kind == RefKind::UrlFetch).count();
                let dependencies = refs.len().saturating_sub(urls);
                let dependencies_u32 = u32::try_from(dependencies).unwrap_or(u32::MAX);
                let urls_u32 = u32::try_from(urls).unwrap_or(u32::MAX);
                external_dependencies.fetch_add(dependencies_u32, Ordering::Relaxed);
                external_urls.fetch_add(urls_u32, Ordering::Relaxed);
                crate::engine::external_fetch_started(dependencies, urls);
            }
            Self::Tree { tree, .. } => {
                for r in refs {
                    tree.set(&locator_key(r), DepState::Fetching);
                }
            }
            Self::Off => {}
        }
    }

    /// A fetch landed: move its row to "analyzing" (bytes in hand, scan pending)
    /// or settle it (skipped/failed/budget). Tree only — keyed on the original
    /// reference, so a locator refined during fetch still matches the row. Called
    /// live per completion and again authoritatively after the batch; both are
    /// idempotent.
    fn landed(&self, r: &Reference, rec: &FetchRecord) {
        if let Self::Tree { tree, .. } = self {
            if matches!(rec.outcome, Outcome::BudgetExceeded) || !terminal_fetch_row_visible(rec) {
                tree.set(&locator_key(r), DepState::Hidden);
            } else {
                tree.set(&locator_key(r), landed_state(rec));
            }
        }
    }

    /// Print an actionable streamed fetch line (stream only); successful fetches
    /// are represented by the aggregate header and final summary. The tree
    /// already moved this row in [`Reporter::landed`].
    ///
    /// Successful rows are intentionally omitted; failures, skips, and pin
    /// mismatches remain visible because they need attention.
    fn report(&self, rec: &FetchRecord, budget_notice: Option<&str>) {
        if matches!(rec.outcome, Outcome::BudgetExceeded) {
            match self {
                Self::Off => {}
                Self::Stream {
                    budget_notice: emitted,
                    ..
                } => {
                    if !emitted.swap(true, Ordering::Relaxed) {
                        tracing::debug!(
                            message = budget_notice
                                .unwrap_or("Skipping remaining fetches, hit fetch budget"),
                            "fetch budget exceeded; remaining references skipped"
                        );
                    }
                }
                Self::Tree {
                    budget_notice: stored,
                    ..
                } => {
                    let mut guard = stored
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if guard.is_none() {
                        *guard = budget_notice.map(str::to_owned);
                    }
                }
            }
            return;
        }
        if !terminal_fetch_row_visible(rec) {
            return;
        }
        if let Self::Stream { .. } = self {
            if matches!(rec.outcome, Outcome::Ok) {
                return;
            }
            crate::engine::print_above_bar(|| report_fetch(rec));
        }
    }

    /// A payload finished analysis: settle its row to the final fetch glyph
    /// (tree only).
    fn analyzed(&self, r: &Reference, rec: &FetchRecord) {
        if let Self::Tree { tree, .. } = self {
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
            Self::Stream { .. } => {
                crate::engine::print_above_bar(|| report_skip(r, reg, now, reason))
            }
            Self::Tree { tree, .. } => {
                tree.add(&locator_key(r), &dep_display_name(r), "");
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
            Self::Stream {
                external_dependencies,
                external_urls,
                ..
            } => {
                let dependencies = external_dependencies.swap(0, Ordering::Relaxed);
                let urls = external_urls.swap(0, Ordering::Relaxed);
                crate::engine::external_fetch_finished(dependencies as usize, urls as usize);
            }
            Self::Tree {
                tree,
                budget_notice,
            } => {
                tree.finish(&summary_line(records));
                let message = budget_notice
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                if let Some(message) = message {
                    eprintln!("    {message}");
                }
            }
        }
    }
}

/// A compact, human display name for a reference: `name version` for a PURL
/// (scope preserved, e.g. `@biomejs/cli-darwin-arm64 2.5.0`), or the URL with
/// its scheme trimmed. This is what the tree shows in place of the full registry
/// URL the streamed log prints.
/// The source manifest's path as the dep tree shows it, led by the scanned
/// artifact so a nested manifest reads plainly as a file *inside* it:
/// `demo.zip!!vexium-1.0.tgz!!package/package.json` →
/// `demo.zip/vexium-1.0.tgz/package/package.json`. The root is reduced to its
/// basename (`/tmp/demo.zip` → `demo.zip`) and deeper archive boundaries become
/// `/`. A bare path with no archive nesting (a plain manifest scanned directly)
/// shows just its basename.
fn manifest_relpath(path: &str) -> String {
    match path.split_once("!!") {
        Some((root, rest)) => {
            let root_base = root.rsplit(['/', '\\']).next().unwrap_or(root);
            format!("{root_base}/{}", rest.replace("!!", "/"))
        }
        None => path.rsplit(['/', '\\']).next().unwrap_or(path).to_string(),
    }
}

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
    // literal '@' only ever separates the version. Qualifiers are dropped from
    // both halves: spec order puts them after the version, and the non-spec
    // ordering older exports emitted glues them onto the name.
    let (name, version) = rest.rsplit_once('@').unwrap_or((rest, ""));
    let name = name.split(['?', '#']).next().unwrap_or(name);
    let name = name.replace("%40", "@");
    let version = version.split(['?', '#']).next().unwrap_or(version);
    if version.is_empty() {
        name
    } else {
        format!("{name} {version}")
    }
}

/// Whether a fetch put bytes in our hands: a clean fetch, bytes whose hash
/// contradicted the declared pin, or bytes carrying a pin Fletch cannot verify.
/// All three are worth scanning — the two pin outcomes most of all — while an
/// unresolved, skipped, budget-capped, or failed fetch has nothing to scan.
fn delivered_bytes(rec: &FetchRecord) -> bool {
    matches!(
        rec.outcome,
        Outcome::Ok | Outcome::PinMismatch | Outcome::UnverifiablePin
    )
}

/// The tree state for a fetch the moment it lands: "analyzing" when bytes are in
/// hand and a scan will follow (an `Ok`, a pin mismatch, or an unverifiable pin —
/// each pin outcome settles to its own glyph once analyzed), else the settled
/// fetch glyph.
fn landed_state(rec: &FetchRecord) -> DepState {
    if delivered_bytes(rec) {
        DepState::Analyzing
    } else {
        done_state(rec)
    }
}

/// The settled tree state for a fetch: the shared [`fetch_row`] glyph/colour,
/// with the detail column (a size, or a failure note) as its trailing text.
fn done_state(rec: &FetchRecord) -> DepState {
    if !terminal_fetch_row_visible(rec) {
        return DepState::Hidden;
    }
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
/// The record a corpus-satisfied dependency gets instead of a download: no
/// bytes, no budget charge, hopper's sha as `content_sha256` so the fetch
/// edge (`source → content`) is still recorded. `Outcome::Skipped` with a
/// content sha never occurs naturally (a real skip never learned one), and
/// [`analyze_payload`] keys on exactly that pair to produce the same
/// "verdict stands in hopper" result as the per-sha precheck — without the
/// second lookup roundtrip.
/// The string a [`RefLocator`] is keyed by everywhere a record carries it.
fn locator_str(locator: &RefLocator) -> String {
    match locator {
        RefLocator::Purl(p) | RefLocator::Url(p) | RefLocator::Path(p) => p.clone(),
    }
}

fn corpus_hit_record(r: &Reference, source_sha: &str, sha: &str) -> FetchRecord {
    FetchRecord {
        source_sha256: source_sha.to_string(),
        source_offset: Some(r.offset),
        kind: r.kind,
        locator: locator_str(&r.locator),
        resolved_url: String::new(),
        final_url: None,
        redirects: Vec::new(),
        status: None,
        headers: Vec::new(),
        fetched_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        content_sha256: Some(sha.to_string()),
        size: None,
        cached: true,
        stale: false,
        pin_verified: None,
        outcome: Outcome::Skipped,
    }
}

fn capture_dependency(rec: &FetchRecord, analyzed: &Analyzed) -> Option<FetchedDependency> {
    let sub = analyzed.sub.as_ref()?;
    if analyzed.content_sha.is_empty() {
        return None;
    }
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
    })
}

/// The artifact could not be retrieved: the registry refused it, the locator
/// resolved to no URL, or the budget ran out before it was reached.
///
/// A distinct type rather than a message, because the difference between "this
/// artifact is not available" and "this server is broken" decides an HTTP
/// status, and a status decides what every caller upstream does next. Beamline
/// reads a 5xx as a sick worker: it opens that worker's circuit breaker, moves
/// the request to the next one, and reports `unavailable` once the fleet is
/// exhausted — so a package nobody can download used to eject healthy workers
/// from the pool and arrive at poppy as an outage rather than a download
/// failure. Classifying that on a substring of a `Debug`-formatted outcome is
/// how it went unnoticed; the type cannot be reworded by accident.
#[derive(Debug)]
pub struct Unretrievable {
    /// What was asked for: the resolved URL, or the locator when there was
    /// none to resolve.
    pub target: String,
    /// How the fetch ended. Carried so a caller can tell a refusal apart from
    /// a budget stop without re-parsing the message.
    pub outcome: Outcome,
}

impl std::fmt::Display for Unretrievable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The wording predates the type and is what the logs and the `scan`
        // CLI have always shown for this failure. Kept verbatim.
        write!(
            f,
            "fetch retrieved nothing for {}: {:?}",
            self.target, self.outcome
        )
    }
}

impl std::error::Error for Unretrievable {}

/// Fetch a single external reference — a `pkg:` PURL or a URL — and return its
/// bytes, a filename for cleave's type detection, and the fetch record. Powers
/// the `pkg`/`url` subcommands: one artifact, pulled and handed to the scanner.
/// On a terminal (`progress`), logs the live/cache outcome and resolved URL,
/// matching `--fetch`. Errors if the client/cache is unavailable
/// ([`anyhow::Error`]) or nothing was retrieved ([`Unretrievable`]).
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
    if !delivered_bytes(&rec) {
        let target = if rec.resolved_url.is_empty() {
            rec.locator.as_str()
        } else {
            rec.resolved_url.as_str()
        };
        return Err(Unretrievable {
            target: target.to_string(),
            outcome: rec.outcome.clone(),
        }
        .into());
    }
    let bytes = res
        .cache
        .load(&rec.locator)
        .ok_or_else(|| anyhow::anyhow!("fetched content for {} not in cache", rec.locator))?;
    let name = payload_name(&rec);
    Ok((bytes, name, rec))
}

/// Serialize a registry record to its `*.registry.json` document — its synthetic
/// name and bytes — so the one-shot `pkg:`/`url` path can scan the registry
/// metadata directly when the artifact itself can't be fetched (e.g. the version
/// was unpublished). `None` if it can't be serialized.
#[must_use]
pub fn registry_document(reg: &Registry) -> Option<(String, Vec<u8>)> {
    Some((registry_doc_name(reg), serde_json::to_vec(reg).ok()?))
}

/// Look up normalized registry metadata plus the provider documents it came
/// from, with relative age stamped. Used by one-shot packages and memo-hit
/// dependency scans; fletch's blob cache prevents a network refetch.
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
    let _ = merge_registry(report, &root_sha, sub);
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
    if !terminal_fetch_row_visible(rec) {
        return;
    }
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
            format!("  \x1b[2m\u{2192} {}\x1b[0m", hosts::host_of(f))
        }
        _ => String::new(),
    };
    let column = detail.unwrap_or_else(|| rec.size.map_or(String::new(), human_bytes));

    eprintln!(
        "    \x1b[38;2;{r};{g};{b}m{glyph} {label:<6}\x1b[0m \x1b[38;2;130;130;130m{column:>10}\x1b[0m  {url}{redirect}"
    );
}

/// Whether one fetch outcome deserves a terminal row. An unresolved locator
/// produced no bytes and carries no actionable failure detail; retain its
/// [`FetchRecord`] for machine output and diagnostics, but keep the default
/// human view quiet.
fn terminal_fetch_row_visible(rec: &FetchRecord) -> bool {
    !matches!(rec.outcome, Outcome::Unresolved)
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
        Outcome::UnverifiablePin => (
            '\u{25cb}',
            "pin?",
            230,
            180,
            80,
            Some("pin unverifiable".to_string()),
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
/// 🚩 known-bad, 🏴 conflicted, 👁 sighted by somebody else (all still scanned;
/// the flag also rides the result header), green ✓ known-good (fetched here
/// only because a pulled/fresh exception forced a re-scan). `None` when bloom
/// is disabled or the artifact is in neither set.
///
/// The digest and the PURL both name this one artifact, so both are handed to
/// the filters together and `burton` combines them: the worst claim against
/// either wins, and the green tick requires every key to agree. Deciding here
/// by hand is what once let this row show a blessed coordinate as clean while
/// its digest was cited by threat intelligence.
fn bloom_fetch_verdict(rec: &FetchRecord) -> Option<FetchRow> {
    use crate::bloom_repo::Decision;
    let lookup = crate::bloom_repo::global()?;

    let digest = rec
        .content_sha256
        .as_deref()
        .and_then(burton::parse_sha256_hex);
    let purl = rec
        .locator
        .starts_with("pkg:")
        .then_some(rec.locator.as_str());
    if digest.is_none() && purl.is_none() {
        return None;
    }

    match lookup.decide_any(purl, digest.as_ref()) {
        Decision::Conflicted => Some(('\u{1f3f4}', "known", 230, 180, 80, None)), // 🏴
        Decision::KnownBad => Some(('\u{1f6a9}', "known", 235, 120, 120, None)),  // 🚩
        Decision::SightedHostile | Decision::SightedSuspicious => {
            Some(('\u{1f441}', "known", 235, 170, 120, None)) // 👁
        }
        Decision::Skip => Some(('\u{2713}', "known", 80, 200, 80, None)),
        Decision::Unknown => None,
    }
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
    /// The *resolved* coordinate is vouched by the known-good bloom, and its
    /// trust is not stale. Reached only by references whose declared locator
    /// carried no version (a manifest range), which the pre-lookup probe in
    /// [`age_gate`] cannot match; pinned coordinates are filtered before the
    /// lookup instead.
    KnownGood,
}

/// Whether a dependency version has been withdrawn from its registry — an npm
/// unpublish (`version_removed`), a pypi/crates yank (recorded as a
/// `deprecated` reason, which never sets `version_removed`), or an npm security
/// takedown (`security_hold`). A withdrawn version's known-good vouch is suspect
/// — it is often *removed because* it was found malicious — so it is re-scanned.
///
/// Withdrawn is not the same as unfetchable, which is what makes re-scanning
/// worth doing. Only an npm unpublish actually removes the bytes; a yank on
/// crates.io or PyPI leaves the artifact downloadable forever (it only stops
/// *new* resolution, so pinned builds keep working), and npm's security hold
/// replaces the release with a placeholder that is served like any other. Those
/// are the cases worth a second look, and their bytes are still there to look at.
///
/// The `version_removed` arm is consequently unreachable from [`age_gate`],
/// which tests it first and settles those as [`SkipReason::Removed`] before
/// consulting [`must_rescan`] at all. It is kept because this predicate is about
/// withdrawal, not about that one caller's ordering.
fn dep_pulled(reg: &Registry) -> bool {
    reg.version_removed == Some(true)
        || reg.security_hold == Some(true)
        || reg.deprecated.as_deref().is_some_and(|d| {
            let d = d.to_ascii_lowercase();
            d.contains("yank") || d.contains("withdrawn") || d.contains("unpublish")
        })
}

/// Number of seconds in the freshness window: a version published this recently
/// is re-scanned rather than trusted on a known-good vouch.
///
/// A published registry version is immutable — npm, crates.io and PyPI all
/// refuse to re-publish `name@version` with different bytes — so a vouch for a
/// pinned coordinate cannot go stale the way a mutable one can, and age is
/// otherwise no reason to distrust it. What this window buys is narrower: cover
/// for the hours right after a release, where a compromise is freshest and
/// least-vetted, and insurance against the bloom itself being wrong about a
/// brand-new package. Hours, not days, is the right size for that.
pub(crate) const FRESH_WINDOW_SECS: u64 = 4 * 3_600;

/// Whether this version was published inside [`FRESH_WINDOW_SECS`].
fn freshly_published(reg: &Registry, now: u64) -> bool {
    reg.age_secs(now)
        .is_some_and(|age| age <= FRESH_WINDOW_SECS)
}

/// A known-good dependency is normally skipped; re-scan it anyway when its trust
/// may be stale — the version was pulled/yanked, or published very recently.
pub(crate) fn must_rescan(reg: &Registry, now: u64) -> bool {
    dep_pulled(reg) || freshly_published(reg, now)
}

/// The publish timestamp a Go pseudo-version carries in its own version string,
/// as seconds since the epoch.
///
/// The module proxy mints a pseudo-version for a commit with no semver tag:
/// `v0.0.0-20260528132821-f66b8cdce5b3`, or `v1.2.3-0.20260528132821-abc…` when
/// it follows a release. The middle field is a UTC `yyyymmddhhmmss` stamp of the
/// commit — the same date the registry would report, already in hand. Parsing it
/// locally lets the age gate reject an old module with no round-trip at all; a
/// `go.sum` alone can declare hundreds.
///
/// `None` for a tagged version (`v1.9.0`), a malformed stamp, or an
/// out-of-range field — anything not confidently datable falls through to the
/// registry lookup rather than being guessed at, so this can only ever save
/// work, never invent an age.
fn go_pseudo_version_published(purl: &str) -> Option<u64> {
    let (_, tail) = purl.strip_prefix("pkg:golang/")?.split_once('@')?;
    // Split on both separators: the bare form joins the stamp with dashes
    // (`v0.0.0-<stamp>-<hash>`), while the post-release form reaches it through
    // a dotted pre-release segment (`v1.2.3-0.<stamp>-<hash>`). The version's own
    // numeric fields are far too short to be mistaken for a 14-digit stamp, and
    // a Go commit hash is 12 hex chars.
    let stamp = tail
        .split(['-', '.'])
        .find(|f| f.len() == 14 && f.bytes().all(|b| b.is_ascii_digit()))?;
    let n = |a: usize, b: usize| stamp.get(a..b)?.parse::<u32>().ok();
    let (y, mo, d) = (n(0, 4)?, n(4, 6)?, n(6, 8)?);
    let (h, mi, s) = (n(8, 10)?, n(10, 12)?, n(12, 14)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || s > 60 {
        return None;
    }
    // Days from the civil epoch — Howard Hinnant's `days_from_civil`, which is
    // exact for the proleptic Gregorian calendar and needs no date crate.
    let y = i64::from(y) - i64::from(mo <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = i64::from(mo) + if mo > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    u64::try_from(days * 86_400 + i64::from(h) * 3_600 + i64::from(mi) * 60 + i64::from(s)).ok()
}

/// Whether a reference is a known-good package coordinate per the loaded bloom
/// filters. Purl-keyed, so it vouches for the coordinate, not the exact bytes.
///
/// Used by [`age_gate`] as a pre-lookup filter: a vouched coordinate skips both
/// the registry round-trip and the artifact fetch. That means the yank check
/// [`must_rescan`] performs is *not* applied there — it needs a registry record
/// this path deliberately never fetches. The trade is intentional: a vouched
/// coordinate is not worth a network round-trip to re-confirm.
/// The reference's coordinate with the registry's resolved version attached, or
/// `None` when it isn't a PURL or nothing resolved.
///
/// A manifest declares a *range* (`"axios": "^1.6.0"`), so an npm reference's
/// locator usually carries no version at all — and the bloom is keyed on exact
/// `name@version`, so probing the declared form can only ever miss. Measured on a
/// 55-package corpus: 22 of 25 npm references arrived version-less, i.e. the
/// known-good check was structurally dead for npm. Lockfile ecosystems (cargo,
/// golang) pin exact versions and are unaffected.
///
/// The version is appended to the declared locator rather than rebuilt from
/// `reg.ecosystem`, so the PURL type and namespace stay exactly as the reference
/// resolver produced them. A scoped npm name percent-encodes its `@`
/// (`pkg:npm/%40scope/name`), so a bare `@` in the body can only be a version.
fn resolved_purl(r: &Reference, reg: &Registry) -> Option<String> {
    let RefLocator::Purl(purl) = &r.locator else {
        return None;
    };
    if reg.version.is_empty() {
        return None;
    }
    let body = purl.strip_prefix("pkg:")?;
    if body.contains('@') {
        return Some(purl.clone());
    }
    Some(format!("{purl}@{}", reg.version))
}

/// Whether the *resolved* coordinate is vouched known-good. The post-lookup
/// counterpart to [`bloom_known_good_purl`]: it costs nothing extra (the record
/// is already in hand) and catches the range-declared references the pre-lookup
/// probe cannot. Pairs with [`must_rescan`], which the pre-lookup path has to
/// forgo — here the record exists, so a yanked version is still caught.
fn bloom_known_good_resolved(r: &Reference, reg: &Registry) -> bool {
    resolved_purl(r, reg).is_some_and(|purl| {
        crate::bloom_repo::global().is_some_and(|lk| lk.decide_purl(&purl).may_skip())
    })
}

fn bloom_known_good_purl(r: &Reference) -> bool {
    let RefLocator::Purl(purl) = &r.locator else {
        return false;
    };
    crate::bloom_repo::global().is_some_and(|lk| lk.decide_purl(purl).may_skip())
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
            burton::parse_sha256_hex(sha_hex)
                .is_some_and(|d| lookup.may_skip(&burton::Artifact::sha256(&d)))
        },
    )))
}

/// One dependency's age-gate outcome: its normalized record plus the raw
/// provider snapshot it came from, paired with why its byte fetch was skipped —
/// `None` when the dependency is kept for a full fetch+scan.
type GatedDep = (
    Reference,
    crate::provenance::RegistryProvenance,
    Option<SkipReason>,
);
type RegistryLookup = (Registry, Vec<fletch::fetch::RecordedSource>);
type IndexedRegistryLookup = (usize, Option<RegistryLookup>);

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
/// Split a locator into `(package key, version)` when it carries an explicit
/// version. The key is everything before the last `@`, so npm scoped names
/// (`pkg:npm/@scope/name@1.2.3`) keep their leading `@`. A candidate version
/// must start with an ASCII digit — git refs, tags, and a scoped name with no
/// version at all (`pkg:npm/@scope/name`) are not versions and exempt the
/// reference from the newest-version gate.
fn versioned_purl(locator: &str) -> Option<(&str, &str)> {
    let (key, version) = locator.rsplit_once('@')?;
    if key.is_empty() || !version.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    Some((key, version))
}

/// True when `r` is a bare (versionless) PURL whose coordinate is pinned by a
/// version-carrying sibling elsewhere in the tree (`pinned` holds those
/// coordinates). A manifest range reaches us version-stripped — `pkg:npm/foo` —
/// and would resolve to `dist-tags/latest`; when a lockfile pins the same
/// coordinate (`pkg:npm/foo@1.2.3`), that pin is ground truth and the bare
/// sibling is redundant, so it is dropped. A versionless PURL's whole string is
/// its coordinate, so an exact set membership is the supersede; a git/tag pin
/// (`pkg:npm/foo@dev`) keeps its `@ref`, is not versionless, and never matches.
fn superseded_by_pin(r: &Reference, pinned: &HashSet<String>) -> bool {
    let RefLocator::Purl(p) = &r.locator else {
        return false;
    };
    versioned_purl(p).is_none() && pinned.contains(p.as_str())
}

/// Lenient, numeric-aware version ordering: the string splits into components
/// at `.`, `-`, `_`, and `+`; each component compares by its leading integer
/// first (`3` > `rc1`? — a fully numeric component outranks a text-led one, so
/// `1.2.3` > `1.2.rc1`), then by remaining text. More components with an equal
/// prefix is newer (`1.2.1` > `1.2`). This orders real registry releases of
/// one package — semver, pep440-ish, and date-like schemes — without
/// validating any of them.
fn lenient_version_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    fn component_cmp(x: &str, y: &str) -> Ordering {
        let digits = |s: &str| s.chars().take_while(char::is_ascii_digit).count();
        let (nx, rx) = x.split_at(digits(x));
        let (ny, ry) = y.split_at(digits(y));
        match (nx.is_empty(), ny.is_empty()) {
            (false, true) => return Ordering::Greater,
            (true, false) => return Ordering::Less,
            (true, true) => return x.cmp(y),
            (false, false) => {}
        }
        let num = match (nx.parse::<u64>(), ny.parse::<u64>()) {
            (Ok(vx), Ok(vy)) => vx.cmp(&vy),
            // A digit run longer than u64 still orders consistently as text.
            _ => nx.cmp(ny),
        };
        num.then_with(|| rx.cmp(ry))
    }
    let (ca, cb): (Vec<&str>, Vec<&str>) = (
        a.split(['.', '-', '_', '+']).collect(),
        b.split(['.', '-', '_', '+']).collect(),
    );
    for (x, y) in ca.iter().zip(cb.iter()) {
        let ord = component_cmp(x, y);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    ca.len().cmp(&cb.len())
}

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
    // Vouched coordinates never reach the network. The bloom probe is a local
    // filter test on the PURL, so asking it *before* `lookup_registries` turns a
    // known-good dependency from "one registry round-trip, then skip" into "no
    // I/O at all" — the single largest saving available on a lockfile-heavy scan,
    // where a `Cargo.lock` alone can declare 700 coordinates.
    //
    // Two things are given up, deliberately. The artifact is not re-scanned even
    // if the version was later yanked (`must_rescan` needs the record we no
    // longer fetch) — an accepted trade: a known-good coordinate is not worth
    // re-fetching. And no `*.registry.json` node is materialized for it, so its
    // registry metadata contributes no findings. Both apply *only* to
    // bloom-vouched coordinates; everything else keeps the full path below.
    let (selected, vouched): (Vec<Reference>, Vec<Reference>) = selected
        .into_iter()
        .partition(|r| !bloom_known_good_purl(r));
    // Second local filter: a Go pseudo-version states its own commit date, so an
    // old module can be aged out without asking the proxy. Only applies when a
    // ceiling is set — with `--fetch-max-age 0` (worker mode) nothing ages out,
    // so there is nothing to short-circuit.
    let (selected, self_dated): (Vec<Reference>, Vec<Reference>) =
        selected.into_iter().partition(|r| {
            let RefLocator::Purl(purl) = &r.locator else {
                return true;
            };
            !max_age.is_some_and(|max| {
                go_pseudo_version_published(purl)
                    .is_some_and(|published| now.saturating_sub(published) >= max)
            })
        });
    for r in &self_dated {
        tracing::debug!(
            package = %locator_key(r),
            "go pseudo-version dates itself past --max-dep-age; registry lookup skipped"
        );
    }
    for r in &vouched {
        crate::bloom_repo::record(crate::bloom_repo::Decision::Skip, false);
        tracing::debug!(
            package = %locator_key(r),
            "known-good coordinate (bloom); registry lookup and fetch both skipped"
        );
    }
    // The network round-trips run concurrently up front; the gate decision below
    // is then pure, so it stays deterministic in `selected` order.
    let lookups = lookup_registries(&selected, res, now);
    let mut keep = Vec::with_capacity(selected.len());
    let mut registries = Vec::new();
    for (r, lookup) in selected.into_iter().zip(lookups) {
        match lookup {
            // A resolved record: gate on age, but materialize it either way. A
            // version the registry has already removed has no fetchable artifact,
            // so skip the doomed fetch too. In every skip case the materialized
            // record's signals still surface. (Known-good coordinates never get
            // here — they were filtered out above, before the lookup.)
            Some(provenance) => {
                let reg = &provenance.record;
                let reason = if reg.version_removed == Some(true) {
                    Some(SkipReason::Removed)
                } else if max_age.is_some_and(|max| reg.age_secs(now).is_some_and(|age| age >= max))
                {
                    Some(SkipReason::AgedOut)
                } else if bloom_known_good_resolved(&r, reg) && !must_rescan(reg, now) {
                    Some(SkipReason::KnownGood)
                } else {
                    None
                };
                if reason.is_none() {
                    keep.push(r.clone());
                }
                registries.push((r, provenance, reason));
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
fn registry_memo() -> &'static RwLock<lru::LruCache<String, Option<Registry>>> {
    static MEMO: OnceLock<RwLock<lru::LruCache<String, Option<Registry>>>> = OnceLock::new();
    MEMO.get_or_init(|| RwLock::new(lru::LruCache::new(REGISTRY_MEMO_CAPACITY)))
}

/// Entries the registry memo keeps. It was an unbounded `HashMap` keyed by
/// raw PURL/URL — one entry per distinct dependency locator a worker ever
/// resolved, forever — which on a days-old fetching worker is hundreds of MB
/// of `Registry` records (a big packument's `release_times` alone is
/// 40-80 KB). Dependency sets cluster in time, so an LRU of a few thousand
/// serves the same hit rate; a miss falls through to fletch's blob cache.
const REGISTRY_MEMO_CAPACITY: std::num::NonZeroUsize = match std::num::NonZeroUsize::new(4096) {
    Some(n) => n,
    None => std::num::NonZeroUsize::MIN,
};

/// Release-cadence window `Registry::with_age` looks back over (48 h).
const RELEASE_CADENCE_WINDOW_SECS: u64 = 172_800;

/// Look up each declared dependency's registry record and raw provider snapshot,
/// returning one slot per input ref in `selected` order. A non-dependency ref,
/// or one whose record can't be resolved, yields `None`.
///
/// Fresh misses use `registry_with_sources` once, so the normalized record and
/// provenance come from one lookup. The process memo deliberately retains only
/// the small record; memo hits recover raw bytes from fletch's blob cache rather
/// than pinning every packument in a long-lived daemon's heap.
fn lookup_registries(
    selected: &[Reference],
    res: &Resources,
    now: u64,
) -> Vec<Option<crate::provenance::RegistryProvenance>> {
    let mut records: Vec<Option<Registry>> = selected.iter().map(|_| None).collect();

    // Split dependency refs into memo hits — served from memory, no disk or
    // network — and misses that still need a lookup.
    let mut misses: Vec<usize> = Vec::new();
    {
        // `peek`, not `get`: a read lock cannot bump LRU order, and a memo
        // hit is cheap enough that recency-on-read is not worth a write lock.
        let memo = registry_memo()
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        for i in (0..selected.len()).filter(|&i| selected[i].kind == RefKind::Dependency) {
            match memo.peek(&locator_key(&selected[i])) {
                // Stored un-aged; stamp the age signals from this scan's clock.
                Some(hit) => records[i] = hit.clone().map(|reg| reg.with_age(now)),
                None => misses.push(i),
            }
        }
    }

    // Run the lookups on the existing rayon pool rather than spawning a fresh
    // batch of OS threads.
    //
    // `std::thread::scope` here created `REGISTRY_LOOKUP_CONCURRENCY` threads per
    // batch, and a directory scan reaches this once per scanned root (times each
    // `--fetch-depth` hop). Every spawn and exit takes the address space's
    // `mmap_lock` to map and unmap a stack, and with the whole rayon pool already
    // resident that serializes into `native_queued_spin_lock_slowpath` — measured
    // at 65% of all samples, with the fetch path burning ~1350s of system time
    // against ~13s for the same scan offline. Rayon's workers already exist, so
    // this is the same fan-out with no thread churn.
    //
    // The lookups are also not the I/O-bound work the old fan-out assumed: nearly
    // all are blob-cache hits that parse JSON and map an ecosystem, i.e. CPU. A
    // separate experiment raising the old constant 8 -> 64 made the scan *slower*
    // (40s vs 33s) for exactly that reason.
    let collected: Vec<IndexedRegistryLookup> = {
        use rayon::prelude::*;
        misses
            .par_iter()
            .map(|&i| {
                // The raw, un-aged record (or `None` for an unresolved or
                // unsupported package) — both worth memoizing so the lookup isn't
                // re-attempted for every file that names it.
                (i, {
                    let (record, sources) =
                        fletch::registry_with_sources(&selected[i].locator, &res.net, &res.cache);
                    record.map(|record| (record, sources))
                })
            })
            .collect()
    };

    // Stamp the aged copy for this scan into each result slot, keeping the raw
    // record to memoize.
    let mut writes: Vec<(String, Option<Registry>)> = Vec::with_capacity(misses.len());
    let mut fresh_sources = HashMap::with_capacity(collected.len());
    for (i, lookup) in collected {
        match lookup {
            Some((record, sources)) => {
                records[i] = Some(record.clone().with_age(now));
                fresh_sources.insert(i, sources);
                // Keep only the release times `with_age` can still count from
                // any later clock: a release older than the cadence window at
                // memo time can never fall inside it again. Bounds the one
                // unbounded field a memoized record carries.
                let mut record = record;
                record
                    .release_times
                    .retain(|&t| now.saturating_sub(t) <= RELEASE_CADENCE_WINDOW_SECS);
                writes.push((locator_key(&selected[i]), Some(record)));
            }
            None => writes.push((locator_key(&selected[i]), None)),
        }
    }
    // One short critical section: nothing but the batch insert runs under the lock.
    {
        let mut memo = registry_memo()
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        for (key, value) in writes {
            memo.put(key, value);
        }
    }

    records
        .into_iter()
        .enumerate()
        .map(|(i, record)| {
            let record = record?;
            let sources = fresh_sources.remove(&i).unwrap_or_else(|| {
                // Memo hit: the provider documents remain in fletch's bounded
                // blob cache, not in the unbounded process memo.
                let (_, sources) =
                    fletch::registry_with_sources(&selected[i].locator, &res.net, &res.cache);
                sources
            });
            Some(crate::provenance::RegistryProvenance::from_record_sources(
                record, &sources,
            ))
        })
        .collect()
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
fn merge_registry(
    report: &mut AnalysisReport,
    parent_sha: &str,
    sub: AnalysisReport,
) -> Option<u32> {
    let (parent_id, parent_depth) = report
        .files
        .iter()
        .find(|f| f.sha256 == parent_sha)
        .map_or((0, 0), |f| (f.id, f.depth));
    let id_base = report.files.iter().map(|f| f.id).max().map_or(0, |m| m + 1);
    let mut root_id = None;
    for mut file in sub.files {
        // The registry document itself (the sub-report's root) is a sidecar:
        // metadata about its parent package, analyzed from its own canonical
        // JSON bytes so its findings feed ML, but not standalone content.
        if file.parent_id.is_none() {
            file.rel = cleave::types::Rel::Registry;
            file.role = cleave::types::Role::Sidecar;
            root_id = Some(file.id + id_base);
        }
        file.id += id_base;
        file.parent_id = Some(file.parent_id.map_or(parent_id, |p| p + id_base));
        file.depth += parent_depth + 1;
        report.files.push(file);
    }
    root_id
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

/// Tally the run's fetches into a one-line summary mirroring the progress bar's
/// completion line: how many came live off the network vs. served from cache,
/// how many failed, and the total bytes pulled. Used by the live tree, which
/// prints it beneath the settled dependency rows.
fn summary_line(records: &[FetchRecord]) -> String {
    let mut live = 0u32;
    let mut cached = 0u32;
    let mut failed = 0u32;
    let mut bytes = 0u64;
    for rec in records {
        bytes += rec.size.unwrap_or(0);
        match &rec.outcome {
            Outcome::Ok | Outcome::PinMismatch | Outcome::UnverifiablePin if rec.cached => {
                cached += 1;
            }
            Outcome::Ok | Outcome::PinMismatch | Outcome::UnverifiablePin => live += 1,
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

/// A file whose references execute only in CI, never in an installed artifact
/// — a GitHub Actions workflow or composite `action.yml`. Its `uses:` actions
/// and `run:` fetches are third-party code, but they run on the CI runner, so a
/// routine dependency fetch skips them; `--fetch=all`/`--fetch=ci` opts in.
fn is_ci_context(file_type: &str) -> bool {
    file_type == "github_actions"
}

/// Whether [`collect_references`] follows references that only ever execute in
/// CI. Auditing an artifact and auditing the pipeline that built it are
/// different questions, and the caller always knows which one it is asking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CiRefs {
    Skip,
    Include,
}

/// References to fetch, grouped by the sha256 of the file that declared them.
fn collect_references(
    report: &AnalysisReport,
    root_path: &Path,
    ci: CiRefs,
) -> Vec<(String, Vec<Reference>)> {
    // Members under a vendored node_modules tree have already been analyzed.
    // Fetching each of their `require("x")` targets again grafted a newer
    // registry release onto the report and repeated work over bytes that came
    // with the sample. Keep hunting absent imports, but resolve local packages
    // first from the package.json members already present in this report.
    let local_npm = LocalNpmPackages::from_report(report);
    let mut local_imports_skipped = 0usize;
    let mut vendored_imports_skipped = 0usize;
    let mut groups: Vec<(String, Vec<Reference>)> = Vec::new();
    for file in &report.files {
        // A GitHub Actions workflow is a CI-only context: its `uses:` actions
        // execute in CI, never in an installed artifact. Skip the whole member
        // unless CI auditing was requested (`--fetch=all`, `--fetch=ci`). The
        // root of a single-file workflow scan is a member here like any other,
        // so this one gate covers it too.
        if ci == CiRefs::Skip && is_ci_context(&file.file_type) {
            continue;
        }
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
        refs.extend(
            find::import_calls(&file.file_type, &view.symbols)
                .into_iter()
                .filter(|reference| {
                    let Some(package) = npm_import_name(reference) else {
                        return true;
                    };
                    // `node_modules` is the installed dependency graph captured
                    // in the artifact. If one of those files imports a package
                    // absent from that graph, Node fails or takes the module's
                    // own fallback path; it does not download today's registry
                    // release. Declared package.json dependencies are collected
                    // separately, so suppress only inferred import-call hunts.
                    if is_vendored_node_module(&file.path) {
                        vendored_imports_skipped += 1;
                        tracing::debug!(
                            source = %file.path,
                            package,
                            "import originates in vendored node_modules; external fetch avoided"
                        );
                        return false;
                    }
                    let present = local_npm.resolves(&file.path, &package);
                    if present {
                        local_imports_skipped += 1;
                        tracing::debug!(
                            source = %file.path,
                            package,
                            "import resolves to vendored node_modules; skipping external fetch"
                        );
                    }
                    !present
                }),
        );
        if refs.is_empty() {
            continue;
        }
        groups.push((file.sha256.clone(), refs));
    }
    if local_imports_skipped > 0 {
        tracing::info!(
            local_imports_skipped,
            "vendored imports already present in report; external refetch avoided"
        );
    }
    if vendored_imports_skipped > 0 {
        tracing::info!(
            vendored_imports_skipped,
            "imports from captured node_modules kept inside artifact boundary"
        );
    }

    // The root sample's imperative hunt (curl|sh, `npm install` in a RUN, a URL
    // in a shell variable) needs its raw text, which the report doesn't carry —
    // read it back from disk for small text-ish roots and merge, deduping
    // against the declared references already collected for the root. The hunt
    // reads bytes the loop above never opened, so it repeats that loop's CI
    // gate — a workflow's raw text re-discovers the `uses:` actions just
    // skipped.
    if let Some(root) = report.files.first()
        && (ci == CiRefs::Include || !is_ci_context(&root.file_type))
        && root.size <= ROOT_HUNT_MAX_BYTES
        && let Ok(bytes) = std::fs::read(root_path)
    {
        let name = root_path
            .file_name()
            .map_or_else(|| root_path.to_string_lossy(), |n| n.to_string_lossy());
        // A provenance document names one artifact and catalogues many. The
        // string hunt cannot tell those apart, so it is replaced by the subject
        // this document is *about* — see `provenance_subject`.
        let hunted = if is_provenance_document(&root.file_type, &name) {
            provenance_subject(&bytes).into_iter().collect()
        } else {
            find::references_in_bytes(&bytes, &name)
        };
        if !hunted.is_empty() {
            merge_into_root(&mut groups, &root.sha256, hunted);
        }
    }
    groups
}

/// Whether this root is one of our own provenance records rather than a
/// collected artifact: hopper's `*.forage.json` collection sidecar, or the
/// normalized `*.registry.json` a fetch materializes.
fn is_provenance_document(file_type: &str, name: &str) -> bool {
    file_type == "registry"
        || name
            .rsplit_once(".forage.")
            .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case("json"))
}

/// The single artifact a provenance document is *about*, as a pinned reference.
///
/// These documents embed the provider's verbatim response, and for PyPI that is
/// the project's whole release catalogue: one 179 KB `diffusers` sidecar carries
/// 191 `files.pythonhosted.org` URLs covering every version ever published. The
/// string hunt has no way to tell the subject from the catalogue — it recovered
/// all 199 URLs and started pulling `diffusers` releases from 0.0.1 upward until
/// the URL budget stopped it. Mining our own cached metadata for dropper
/// candidates is the bug; the document already states its subject, so read that
/// instead of guessing from `strings`.
///
/// Parsed here rather than from filefacts' `values` because a large sidecar
/// exceeds cleave's JSON parse limit (76 KB) while still being small enough to
/// hunt, so the facts view cannot be relied on for exactly the documents that
/// carry the biggest catalogues.
fn provenance_subject(bytes: &[u8]) -> Option<Reference> {
    let doc: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let url = doc
        .pointer("/fetch/url")
        .or_else(|| doc.pointer("/registry/url"))
        .and_then(serde_json::Value::as_str)
        .filter(|u| !u.is_empty())?;
    // The recorded digest pins the fetch: this document exists because those
    // exact bytes were collected, so a mismatch is a substitution worth failing.
    let pinned_hash = doc
        .pointer("/artifact/sha256")
        .and_then(serde_json::Value::as_str)
        .filter(|d| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(|d| fletch::PinnedHash {
            algo: fletch::HashAlgo::Sha256,
            value: d.to_ascii_lowercase(),
        });
    let content_sha256 = pinned_hash.as_ref().map(|p| p.value.clone());
    Some(Reference {
        locator: RefLocator::Url(url.to_string()),
        kind: RefKind::UrlFetch,
        source: "forage.fetch.url".to_string(),
        evidence: url.to_string(),
        offset: 0,
        pinned_hash,
        content_sha256,
    })
}

fn is_vendored_node_module(path: &str) -> bool {
    path.split(['/', '\\'])
        .any(|component| component == "node_modules")
}

/// Package roots physically present under `node_modules`, keyed by their full
/// virtual path. Only roots with an analyzed package.json enter the index, so
/// an incidental path component named node_modules is not enough.
#[derive(Default)]
struct LocalNpmPackages {
    roots: HashSet<String>,
}

impl LocalNpmPackages {
    fn from_report(report: &AnalysisReport) -> Self {
        let roots = report
            .files
            .iter()
            .filter_map(|file| npm_package_root(&file.path))
            .collect();
        Self { roots }
    }

    /// Mirror Node's package-level lookup: walk upward from the importing
    /// file, looking for `node_modules/<package>`. Exports and subpaths do not
    /// matter here because fletch retrieves whole packages.
    fn resolves(&self, source_path: &str, package: &str) -> bool {
        let normalized = source_path.replace('\\', "/");
        let Some((mut dir, _)) = normalized.rsplit_once('/') else {
            return false;
        };
        loop {
            if self
                .roots
                .contains(&format!("{dir}/node_modules/{package}"))
            {
                return true;
            }
            let Some((parent, _)) = dir.rsplit_once('/') else {
                return false;
            };
            dir = parent;
        }
    }
}

/// Return the virtual package root for an analyzed node_modules/package.json.
fn npm_package_root(path: &str) -> Option<String> {
    let normalized = path.replace('\\', "/");
    let root = normalized.strip_suffix("/package.json")?;
    let (_, package) = root.rsplit_once("/node_modules/")?;
    let mut parts = package.split('/');
    let first = parts.next()?;
    let valid = if let Some(scope) = first.strip_prefix('@') {
        let name = parts.next().unwrap_or_default();
        !scope.is_empty() && !name.is_empty() && parts.next().is_none()
    } else {
        !first.is_empty() && parts.next().is_none()
    };
    valid.then(|| root.to_string())
}

/// Extract the npm package name from a reference produced by
/// `find::import_calls`. Such PURLs are normally versionless; tolerating a
/// version keeps this correct if fletch later learns one from the symbol.
fn npm_import_name(reference: &Reference) -> Option<String> {
    let RefLocator::Purl(purl) = &reference.locator else {
        return None;
    };
    let coordinate = purl.strip_prefix("pkg:npm/")?.split('?').next()?;
    let encoded_name = coordinate
        .rsplit_once('@')
        .map_or(coordinate, |(name, _)| name);
    Some(
        encoded_name
            .replace("%40", "@")
            .replace("%2F", "/")
            .replace("%2f", "/"),
    )
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

/// Whether this batch may fan its payload analyses across the Rayon pool.
///
/// Each fetched payload is a full cleave analysis, and cleave bounds how many
/// analyses fan out at once on the assumption that the throttled ones make
/// serial progress on their own blocking threads (see
/// [`cleave::pool_has_headroom`]). Dispatching them from `par_iter` breaks
/// that: a throttled payload analysis occupies a Rayon worker instead of
/// freeing one, and the dispatcher sits blocked-and-stealing on top. Measured
/// on a wedged worker 2026-09-04, that left every pool thread carrying 15-29
/// nested blocked joins with frames from unrelated analyses interleaved — one
/// runaway leaf then pinned the whole pool rather than one thread.
///
/// So fan out only when the pool has headroom — a lone analysis, or a scan
/// draining its queue, which are exactly the cases where fanning out is what
/// keeps the pool busy. Under saturation the payloads run inline on the
/// blocking thread that owns this batch, which is both the shape cleave's
/// throttle expects and no loss of machine utilization: the sibling analyses
/// already have every core.
///
/// `SCAN_PAYLOAD_FANOUT` overrides: `always` restores the unconditional
/// fan-out, `never` forces inline.
fn payload_fanout_allowed() -> bool {
    match std::env::var("SCAN_PAYLOAD_FANOUT").ok().as_deref() {
        Some("always") => true,
        Some("never") => false,
        _ => cleave::pool_has_headroom(),
    }
}

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
    let one = |(i, rec): (usize, &FetchRecord)| {
        let a = analyze_payload(rec, cache, opts, acache);
        on_analyzed(i);
        a
    };
    if payload_fanout_allowed() {
        use rayon::prelude::*;
        fetched.par_iter().enumerate().map(one).collect()
    } else {
        fetched.iter().enumerate().map(one).collect()
    }
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
    // Scan whatever bytes we hold: a clean fetch, a pin mismatch, or a pin we
    // could not verify (the pin outcomes are exactly the cases worth analyzing).
    // Skipped/unresolved/failed have no bytes.
    if !delivered_bytes(rec) {
        return None;
    }
    // Corpus-satisfied PURL (see `corpus_hit_record`): the verdict already
    // stands in hopper; produce the same skip the per-sha precheck would,
    // without re-asking.
    if matches!(rec.outcome, Outcome::Skipped) {
        return rec.content_sha256.clone().map(|content_sha| Analyzed {
            sub: None,
            content_sha,
            next_from_bytes: Vec::new(),
        });
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

    // Fleet-shared skip: the corpus already holds a benign verdict for these
    // exact bytes, analyzed within the freshness window. The local cache above
    // is better when it hits (it returns the full sub-report to graft); this
    // covers the fleet-wide case it cannot — another worker analyzed the same
    // dependency, or a release just invalidated every local cache at once. A
    // skipped payload merges like a benign analysis that found nothing: no
    // sub-report to graft, no next-hop references, and — because it never
    // enters the envelope — no member fan-out or renewal on hopper's side.
    if !content_sha.is_empty() && crate::corpus_precheck::skip_reanalysis(&content_sha) {
        tracing::debug!(
            locator = %rec.locator,
            content_sha = %content_sha,
            "corpus precheck: verdict stands in hopper; skipping re-analysis"
        );
        return Some(Analyzed {
            sub: None,
            content_sha,
            next_from_bytes: Vec::new(),
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
    // Name the subtree for what it is. cleave named these from payload_name — the
    // URL's basename, which it needs for extension type detection but which says
    // nothing about origin. In the merged report that left a fetched dependency
    // indistinguishable from an archive member, and put two dependencies whose
    // URLs end in the same basename (index.js, package.tgz, download) under one
    // path, so anything keyed on path merged them.
    //
    // The locator is unique per dependency and is what a reader recognizes.
    // Rewritten across the whole subtree, not just its root, so members stay
    // attached to it — the appendix and every other per-path lookup walk a
    // "<root>!!" prefix. Merged report only: the standalone report captured for
    // hopper was taken before this and keeps cleave's own naming.
    let old_root = sub
        .files
        .iter()
        .find(|f| f.parent_id.is_none())
        .map(|f| f.path.clone());
    let rename = old_root
        .filter(|old| !old.is_empty() && !rec.locator.is_empty())
        .map(|old| (format!("{old}!!"), old, rec.locator.clone()));

    let first_new = report.files.len();
    for mut file in sub.files {
        // The payload's own top node (the sub-report's root) is a fetched edge:
        // pulled from `via`, not contained in its parent. Its exploded members
        // stay ordinary members.
        let is_sub_root = file.parent_id.is_none();
        file.id += id_base;
        file.parent_id = Some(file.parent_id.map_or(parent_id, |p| p + id_base));
        file.depth += parent_depth + 1;
        if let Some((old_prefix, old, locator)) = &rename {
            if file.path == *old {
                file.path.clone_from(locator);
            } else if let Some(rest) = file.path.strip_prefix(old_prefix.as_str()) {
                file.path = format!("{locator}!!{rest}");
            }
        }
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
        // Per-package and routine on any registry-metadata scan: one line per
        // artifact drowns a worker's log. The grafted composites themselves
        // show up in the report, which is where they matter.
        Ok(n) => tracing::debug!(
            grafted = n,
            "package-scoped composites fired across artifact and registry metadata"
        ),
        Err(e) => tracing::warn!("package composite pass failed: {e:#}"),
    }
}

/// Enrich a fetched dependency's standalone report with package-scoped
/// registry composites before it is captured or merged into its parent.
fn prepare_dependency_report(
    payload: &mut Analyzed,
    registry_findings: Option<&[Finding]>,
    opts: &AnalysisOptions,
) {
    prepare_dependency_report_with(payload, registry_findings, opts, apply_package_composites);
}

/// Return the registry findings paired with the original declared reference.
/// A fetch may canonicalize a versionless/ranged PURL to the concrete version
/// it downloaded, so joining on `FetchRecord::locator` loses exactly the
/// registry transitions (including security holders) this pass must correlate.
fn registry_findings_for_reference<'a>(
    findings: &'a HashMap<String, Vec<Finding>>,
    reference: &Reference,
) -> Option<&'a [Finding]> {
    findings.get(&locator_key(reference)).map(Vec::as_slice)
}

/// Injectable core of [`prepare_dependency_report`]. Keeping the mutation in a
/// small function makes the ordering contract testable without network access
/// or depending on the machine's installed trait bundle.
fn prepare_dependency_report_with(
    payload: &mut Analyzed,
    registry_findings: Option<&[Finding]>,
    opts: &AnalysisOptions,
    graft: impl FnOnce(&mut AnalysisReport, &str, &[Finding], &[Finding], &AnalysisOptions),
) {
    let Some(registry) = registry_findings else {
        return;
    };
    let artifact = payload.sub.as_ref().map(sub_findings).unwrap_or_default();
    let artifact_sha = payload.content_sha.clone();
    let Some(sub) = payload.sub.as_mut() else {
        return;
    };
    graft(sub, &artifact_sha, &artifact, registry, opts);
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
    fn manifest_relpath_leads_with_the_scanned_artifact() {
        // A plain manifest scanned directly shows just its basename.
        assert_eq!(manifest_relpath("/home/u/package.json"), "package.json");
        // A nested manifest reads as a file inside the scanned archive.
        assert_eq!(
            manifest_relpath("/tmp/demo.zip!!vexium-1.0.tgz!!package/package.json"),
            "demo.zip/vexium-1.0.tgz/package/package.json"
        );
        assert_eq!(
            manifest_relpath("demo.zip!!requirements.txt"),
            "demo.zip/requirements.txt"
        );
    }

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

    fn test_finding(id: &str, crit: cleave::Criticality) -> Finding {
        let mut finding = Finding::new(
            id.to_string(),
            cleave::types::FindingKind::Capability,
            id.to_string(),
            1.0,
        );
        finding.crit = crit;
        finding
    }

    fn fetched_payload_with_finding(id: &str) -> Analyzed {
        let mut report: AnalysisReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [{
                "id": 0,
                "path": "holder.tgz",
                "depth": 0,
                "file_type": "npm",
                "sha256": "d".repeat(64),
                "size": 399u64
            }]
        }))
        .expect("dependency report");
        report.files[0]
            .findings
            .push(test_finding(id, cleave::Criticality::Notable));
        Analyzed {
            sub: Some(report),
            content_sha: "d".repeat(64),
            next_from_bytes: Vec::new(),
        }
    }

    fn fetched_record() -> FetchRecord {
        FetchRecord {
            source_sha256: "s".repeat(64),
            source_offset: Some(17),
            kind: RefKind::Dependency,
            locator: "pkg:npm/held@0.0.1-security".to_string(),
            resolved_url: "https://registry.test/held-0.0.1-security.tgz".to_string(),
            final_url: None,
            redirects: Vec::new(),
            status: Some(200),
            headers: Vec::new(),
            fetched_at: 0,
            content_sha256: Some("d".repeat(64)),
            size: Some(399),
            cached: true,
            stale: false,
            pin_verified: None,
            outcome: Outcome::Ok,
        }
    }

    /// Regression for the fetched-package ordering bug: registry/package
    /// composites used to be grafted only after `capture_dependency` consumed
    /// its snapshot, so Hopper graded the dependency as clean and the parent
    /// never received a dependency-verdict back-reference.
    #[test]
    fn registry_composite_reaches_standalone_capture_and_merged_dependency() {
        let mut payload = fetched_payload_with_finding("artifact/seed");
        let registry = vec![test_finding(
            "metadata/registry::registry-security-hold-record",
            cleave::Criticality::Suspicious,
        )];
        let expected_composite =
            "objectives/supply-chain::registry-security-withdrawn-package-coordinate";

        prepare_dependency_report_with(
            &mut payload,
            Some(&registry),
            &AnalysisOptions::default(),
            |report, artifact_sha, artifact, registry, _opts| {
                assert_eq!(artifact_sha, "d".repeat(64));
                assert!(artifact.iter().any(|f| f.id == "artifact/seed"));
                assert!(
                    registry
                        .iter()
                        .any(|f| { f.id == "metadata/registry::registry-security-hold-record" })
                );
                report
                    .files
                    .iter_mut()
                    .find(|file| file.sha256 == artifact_sha)
                    .expect("artifact node")
                    .findings
                    .push(test_finding(
                        expected_composite,
                        cleave::Criticality::Hostile,
                    ));
            },
        );

        let rec = fetched_record();
        let captured = capture_dependency(&rec, &payload).expect("standalone capture");
        let captured: cleave::types::CompactReport =
            serde_json::from_str(&captured.raw).expect("captured report parses");
        assert!(
            captured.files[0]
                .findings
                .iter()
                .any(|f| f.id == expected_composite && f.criticality == 5),
            "the standalone report graded for Hopper must contain the hostile registry composite"
        );

        let mut parent: AnalysisReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [{
                "id": 0,
                "path": "package.json",
                "depth": 0,
                "file_type": "package_json",
                "sha256": "s".repeat(64),
                "size": 100u64
            }]
        }))
        .expect("parent report");
        merge_payload(&mut parent, &rec, payload);
        let fetched = parent
            .files
            .iter()
            .find(|file| file.sha256 == "d".repeat(64))
            .expect("merged dependency root");
        assert_eq!(fetched.rel, cleave::types::Rel::Fetched);
        assert!(
            fetched
                .findings
                .iter()
                .any(|f| f.id == expected_composite && f.crit == cleave::Criticality::Hostile),
            "the embedded-file grader must see the same hostile registry composite"
        );
    }

    /// Registry correlation is opt-in per fetched edge. A dependency without a
    /// registry record must be captured unchanged and must not invoke the
    /// package-composite pass with unrelated metadata.
    #[test]
    fn missing_registry_does_not_mutate_fetched_dependency() {
        let mut payload = fetched_payload_with_finding("artifact/seed");
        prepare_dependency_report_with(
            &mut payload,
            None,
            &AnalysisOptions::default(),
            |_report, _sha, _artifact, _registry, _opts| {
                panic!("package composite pass must not run without matching registry metadata")
            },
        );

        let captured = capture_dependency(&fetched_record(), &payload).expect("capture");
        let captured: cleave::types::CompactReport =
            serde_json::from_str(&captured.raw).expect("captured report parses");
        assert_eq!(captured.files[0].findings.len(), 1);
        assert_eq!(captured.files[0].findings[0].id, "artifact/seed");
    }

    /// npm resolves a versionless or ranged declaration to a concrete holder
    /// release. Registry metadata remains keyed by the declaration; the fetch
    /// record carries the resolved coordinate. The package pass must join on
    /// the former or a security-holder transition disappears.
    #[test]
    fn registry_join_survives_versionless_purl_resolution() {
        let declared = Reference {
            locator: RefLocator::Purl("pkg:npm/held".to_string()),
            kind: RefKind::Dependency,
            source: "package.json".to_string(),
            evidence: "held".to_string(),
            offset: 17,
            pinned_hash: None,
            content_sha256: None,
        };
        let fetched = fetched_record();
        assert_eq!(fetched.locator, "pkg:npm/held@0.0.1-security");
        assert_ne!(locator_key(&declared), fetched.locator);

        let mut by_declared_locator = HashMap::new();
        by_declared_locator.insert(
            locator_key(&declared),
            vec![test_finding(
                "metadata/registry::registry-security-hold-record",
                cleave::Criticality::Suspicious,
            )],
        );
        let paired = registry_findings_for_reference(&by_declared_locator, &declared)
            .expect("versionless declaration retains its registry sidecar");
        assert_eq!(paired.len(), 1);
        assert_eq!(
            paired[0].id,
            "metadata/registry::registry-security-hold-record"
        );
        assert!(
            !by_declared_locator.contains_key(&fetched.locator),
            "control: joining on the resolved fetch locator would drop the sidecar"
        );
    }

    #[test]
    fn summary_line_omits_zero_counts() {
        let record = |outcome: Outcome, cached: bool, size: Option<u64>| FetchRecord {
            source_sha256: String::new(),
            source_offset: None,
            kind: RefKind::Dependency,
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
    fn unresolved_fetches_stay_out_of_the_terminal_view() {
        let mut rec = fetched_record();
        rec.outcome = Outcome::Unresolved;

        assert!(!terminal_fetch_row_visible(&rec));
        assert!(matches!(landed_state(&rec), DepState::Hidden));
        assert!(matches!(done_state(&rec), DepState::Hidden));

        rec.outcome = Outcome::Failed("transport".to_string());
        assert!(terminal_fetch_row_visible(&rec));
        assert!(matches!(done_state(&rec), DepState::Done { .. }));
    }

    /// Regression for the silent-skip bug: `UnverifiablePin` delivers bytes just
    /// as `Ok` and `PinMismatch` do. The "did we get bytes" gates are `matches!`,
    /// which the compiler does not check for exhaustiveness, so a new Outcome can
    /// slip through them and drop a fetched payload out of analysis unnoticed —
    /// precisely the payload whose pin could not be verified.
    #[test]
    fn an_unverifiable_pin_is_analyzed_and_tallied_like_any_delivered_bytes() {
        let mut rec = fetched_record();
        rec.outcome = Outcome::UnverifiablePin;

        assert!(delivered_bytes(&rec));
        assert!(matches!(landed_state(&rec), DepState::Analyzing));
        assert!(terminal_fetch_row_visible(&rec));

        // It must reach the run summary rather than vanish from the counts.
        let line = summary_line(std::slice::from_ref(&rec));
        assert!(line.contains("1 cached"), "{line}");

        // And read as its own row, distinct from a verified fetch and from the
        // harder `pin!` mismatch.
        let (_, label, ..) = fetch_row(&rec);
        assert_eq!(label, "pin?");
    }

    #[test]
    fn budget_notice_names_the_limiting_count_flag() {
        assert_eq!(
            fetch_count_budget_notice("--fetch-max-urls", 4, usize::MAX),
            "Skipping remaining fetches, hit fetch budget (--fetch-max-urls=4)"
        );
        assert_eq!(
            fetch_count_budget_notice("--fetch-max-file-fetches", 100, 7),
            "Skipping remaining fetches, hit fetch budget (--fetch-max-total-fetches=7)"
        );
    }

    #[test]
    fn go_pseudo_versions_date_themselves_without_a_lookup() {
        // Fixed points spanning the civil-days arithmetic: the epoch itself, a
        // century/leap-rule boundary, a leap day, and both year edges. These
        // catch an off-by-one era or an early month roll, which self-consistent
        // round-tripping would not.
        for (stamp, want) in [
            ("19700101000000", 0),
            ("20000101000000", 946_684_800), // 2000-01-01, leap-century
            ("20240229120000", 1_709_208_000), // leap day
            ("20251231235959", 1_767_225_599), // last second of a year
            ("20260101000000", 1_767_225_600), // first second of the next
        ] {
            assert_eq!(
                go_pseudo_version_published(&format!("pkg:golang/x/y@v0.0.0-{stamp}-abc")),
                Some(want),
                "{stamp}"
            );
        }

        // Real coordinates from the benchmark corpus, in both spellings the Go
        // proxy mints: the bare `v0.0.0-` form and the `vX.Y.Z-0.` form that
        // follows a release tag.
        assert_eq!(
            go_pseudo_version_published(
                "pkg:golang/github.com/deckhouse/deckhouse@v0.0.0-20260528132821-f66b8cdce5b3"
            ),
            Some(1_779_974_901),
        );
        assert_eq!(
            go_pseudo_version_published(
                "pkg:golang/cloud.google.com/go@v0.20.1-0.20260528200609-1134b3699ee5"
            ),
            Some(1_779_998_769),
        );
        // A module path containing digits and dots must not confuse the stamp
        // hunt, and neither must a 14-digit-looking commit hash prefix.
        assert_eq!(
            go_pseudo_version_published(
                "pkg:golang/gopkg.in/yaml.v3@v0.0.0-20260720151329-12345678901234"
            ),
            Some(1_784_560_409),
        );

        // Anything not confidently datable falls through to the registry rather
        // than being guessed at — a wrong age here silently skips a fetch.
        for undatable in [
            "pkg:golang/github.com/stretchr/testify@v1.9.0", // tagged release
            "pkg:golang/x/y@v0.0.0-2026052813282-abc",       // 13 digits
            "pkg:golang/x/y@v0.0.0-202605281328210-abc",     // 15 digits
            "pkg:golang/x/y@v0.0.0-20261328132821-abc",      // month 13
            "pkg:golang/x/y@v0.0.0-20260028132821-abc",      // month 0
            "pkg:golang/x/y@v0.0.0-20260532132821-abc",      // day 32
            "pkg:golang/x/y@v0.0.0-20260500132821-abc",      // day 0
            "pkg:golang/x/y@v0.0.0-20260528243821-abc",      // hour 24
            "pkg:golang/x/y@v0.0.0-20260528136021-abc",      // minute 60
            "pkg:golang/x/y",                                // no version at all
            "pkg:cargo/serde@1.0.219",                       // wrong ecosystem
            "pkg:npm/axios@1.6.8",                           // wrong ecosystem
            "not-a-purl",
        ] {
            assert_eq!(
                go_pseudo_version_published(undatable),
                None,
                "{undatable} must not be dated locally"
            );
        }
    }

    #[test]
    fn go_pseudo_version_age_decides_the_gate_the_same_way_a_registry_would() {
        // The gate compares `now - published >= max_age`. Pin that the local date
        // drives the same decision the registry record would, on both sides of
        // the boundary — this is the behaviour, not just the parse.
        let published = go_pseudo_version_published(
            "pkg:golang/github.com/deckhouse/deckhouse@v0.0.0-20260528132821-f66b8cdce5b3",
        )
        .expect("datable");
        let week = 7 * 86_400;

        // A month later: comfortably past a 7-day ceiling.
        let now = published + 30 * 86_400;
        assert!(now.saturating_sub(published) >= week);
        // A day later: inside the window, so it must still be fetched.
        let now = published + 86_400;
        assert!(now.saturating_sub(published) < week);
        // Exactly at the boundary counts as aged out, matching `age_secs >= max`.
        let now = published + week;
        assert!(now.saturating_sub(published) >= week);
        // A clock behind the stamp must not underflow into "ancient".
        let now = published - 86_400;
        assert_eq!(now.saturating_sub(published), 0);
    }

    #[test]
    fn resolved_purl_pairs_registry_version_onto_range_declarations() {
        let dep = |locator: &str| Reference {
            locator: RefLocator::Purl(locator.to_string()),
            kind: RefKind::Dependency,
            source: String::new(),
            evidence: String::new(),
            offset: 0,
            pinned_hash: None,
            content_sha256: None,
        };
        let at = |v: &str| Registry {
            version: v.to_string(),
            ..Registry::default()
        };

        // The npm case this exists for: a manifest range leaves the locator
        // version-less, so the bloom (keyed `name@version`) could never match it.
        assert_eq!(
            resolved_purl(&dep("pkg:npm/axios"), &at("1.6.8")).as_deref(),
            Some("pkg:npm/axios@1.6.8")
        );
        // A scoped npm name percent-encodes its `@`, so the scope must not be
        // mistaken for a version already being present.
        assert_eq!(
            resolved_purl(&dep("pkg:npm/%40scope/pkg"), &at("2.0.0")).as_deref(),
            Some("pkg:npm/%40scope/pkg@2.0.0")
        );
        // Lockfile ecosystems already pin a version — returned untouched, and
        // notably NOT re-suffixed with the registry's version.
        assert_eq!(
            resolved_purl(&dep("pkg:cargo/serde@1.0.219"), &at("1.0.220")).as_deref(),
            Some("pkg:cargo/serde@1.0.219")
        );
        assert_eq!(
            resolved_purl(
                &dep("pkg:golang/github.com/stretchr/testify@v1.9.0"),
                &at("v1.9.1")
            )
            .as_deref(),
            Some("pkg:golang/github.com/stretchr/testify@v1.9.0")
        );
        // Nothing resolved, or not a PURL: no coordinate to probe.
        assert_eq!(resolved_purl(&dep("pkg:npm/axios"), &at("")), None);
        assert_eq!(
            resolved_purl(
                &Reference {
                    locator: RefLocator::Url("https://example.test/x.tgz".into()),
                    kind: RefKind::UrlFetch,
                    source: String::new(),
                    evidence: String::new(),
                    offset: 0,
                    pinned_hash: None,
                    content_sha256: None,
                },
                &at("1.0.0")
            ),
            None
        );
    }

    #[test]
    fn freshly_published_and_must_rescan_track_publish_age() {
        let now = 1_000_000_u64;
        let fresh = Registry {
            published_at: Some(now - 3_600), // 1h ago
            ..Registry::default()
        };
        // 30h ago: outside the 4h window, and the case that separates it from
        // the local-file window in `engine`, which is still measured in days.
        let day_and_a_half = Registry {
            published_at: Some(now - 30 * 3_600),
            ..Registry::default()
        };
        let stale = Registry {
            published_at: Some(now - 300_000), // ~3.5d ago
            ..Registry::default()
        };
        assert!(freshly_published(&fresh, now));
        assert!(!freshly_published(&day_and_a_half, now));
        assert!(!freshly_published(&stale, now));
        // A settled, unwithdrawn version needs no re-scan; a just-published one
        // does, and a withdrawn one always does regardless of age.
        assert!(!must_rescan(&stale, now));
        assert!(!must_rescan(&day_and_a_half, now));
        assert!(must_rescan(&fresh, now));
        // A yank leaves the artifact downloadable, so re-scanning it is not a
        // doomed fetch — this is the arm that earns `dep_pulled` its keep.
        assert!(must_rescan(
            &Registry {
                published_at: Some(now - 300_000),
                deprecated: Some("yanked".to_string()),
                ..Registry::default()
            },
            now
        ));
        assert!(must_rescan(
            &Registry {
                published_at: Some(now - 300_000),
                security_hold: Some(true),
                ..Registry::default()
            },
            now
        ));
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
    fn discovered_url_filter_keeps_download_shapes_and_drops_sites_and_endpoints() {
        for url in [
            "https://downloads.example.test/stage-2.sh",
            "https://cdn.example.test/releases/download/v1/payload",
            "https://api.telegram.org/file/bot123/documents/payload.exe",
            "https://huggingface.co/o/m/resolve/main/payload.bin",
            "https://example.test/archive/payload.tar.gz?sig=abc",
            "https://example.test/payload.js",
        ] {
            assert!(
                looks_like_dropper_download_url(url),
                "download-shaped URL was rejected: {url}"
            );
        }

        for url in [
            "https://example.test",
            "https://example.test/",
            "https://example.test/docs/",
            "https://example.test/download",
            "https://example.test/download?file=stage.sh",
            "https://api.example.test/v1/models",
            "https://example.test/api/v1/query",
            "https://example.test/graphql",
            "https://example.test/index.html",
            "https://example.test/result.json",
            "https://example.test/submit.php?id=1",
            "https://example.test/download/index.html",
            "https://example.test/file%2Ezip",
            "ftp://example.test/stage-2.sh",
            // A trailing `/` names a directory; the response is an index or a
            // landing page. This is the shape that pulled PyPI project pages.
            "https://pypi.org/project/diffusers/0.40.0/",
            "https://example.test/releases/v1.2.3/",
            // A version is not a filename, with or without the trailing slash.
            "https://pypi.org/project/diffusers/0.40.0",
            "https://github.com/foo/bar/releases/tag/v1.2.3",
            "https://api.example.test/v2.1",
            "https://example.test/lib/1.2.3/",
            "https://example.test/pkg/2.0.0",
        ] {
            assert!(
                !looks_like_dropper_download_url(url),
                "site/API-shaped URL was kept: {url}"
            );
        }
    }

    #[test]
    fn discovered_urls_need_a_domain_or_ip_host() {
        for url in ["https://example.com/stage.sh", "http://8.8.8.8/payload.bin"] {
            assert!(
                valid_discovered_url_host(url),
                "valid host was rejected: {url}"
            );
        }
        for url in [
            "http://wpad/wpad.dat",
            "http://localhost/payload.bin",
            "http://%@:%u/rfc2585/%@.crl",
            "http://10.0.0.1/payload.bin",
            "http://100.64.0.1/payload.bin",
            "http://172.16.0.1/payload.bin",
            "http://192.168.1.1/payload.bin",
            "http://192.0.2.1/payload.bin",
            "http://127.0.0.1/payload.bin",
            "http://[::1]/payload.bin",
            "http://[fd00::1]/payload.bin",
            "http://[fe80::1]/payload.bin",
            "/relative/payload.bin",
            "relative/payload.bin",
        ] {
            assert!(
                !valid_discovered_url_host(url),
                "invalid or local host was accepted: {url}"
            );
        }
    }

    #[test]
    fn discovered_urls_with_unexpanded_templates_are_skipped() {
        for url in [
            "https://github.com/$REPO/releases/latest",
            "https://api.github.com/repos/$REPO/releases/latest",
            "https://github.com/$REPO/releases/download/v$VERSION",
            "https://github.com/$REPO.git",
            "https://bitbucket.org/${this.repositoryId}/raw/${r}/${e}",
            "https://gitlab.com/${this.repositoryId}/raw/${r}/${e}",
        ] {
            assert!(
                has_unexpanded_url_placeholder(url),
                "unexpanded template was not recognized: {url}"
            );
        }
        assert!(!has_unexpanded_url_placeholder(
            "https://github.com/atomdrift-project/scan/releases/download/v2.8.0/atomscan"
        ));
    }

    #[test]
    fn version_shape_does_not_swallow_real_filenames() {
        // Versions — rejected as download targets.
        for v in ["0.40.0", "v2.1", "V10.0.1", "2.1"] {
            assert!(is_version_shaped(v), "version not recognized: {v}");
        }
        // Not versions: a real artifact whose name merely contains digits and
        // dots must still be fetchable. A Go module's pseudo-version filename
        // is the case that matters — mistaking it for a version stops the
        // module being fetched at all.
        for v in [
            "v0.0.0-20260823143148-1fb3b878e2fb.zip",
            "diffusers-0.0.1.tar.gz",
            "payload.exe",
            "stage-2.sh",
            "v1",
            "1",
            "lib.so.6",
            ".2.3",
            "1.2.",
            "",
        ] {
            assert!(!is_version_shaped(v), "filename misread as version: {v}");
        }
        // The versioned artifacts themselves stay fetchable end to end.
        for url in [
            "https://files.pythonhosted.org/packages/a0/05/x/diffusers-0.0.1.tar.gz",
            "https://example.test/releases/v1.2.3/payload.bin",
            "https://proxy.golang.org/github.com/o/r/@v/v0.0.0-20260823143148-1fb3b878e2fb.zip",
        ] {
            assert!(
                looks_like_dropper_download_url(url),
                "versioned artifact was rejected: {url}"
            );
        }
    }

    #[test]
    fn provenance_documents_yield_their_subject_not_their_catalogue() {
        // A forage sidecar: one artifact named by `fetch.url`, wrapped around a
        // provider response that lists every release of the project.
        let doc = serde_json::json!({
            "fetch": {"url": "https://files.example.test/pkg-1.0.tar.gz"},
            "artifact": {"sha256": "A".repeat(64), "filename": "pkg-1.0.tar.gz"},
            "registry": {
                "url": "https://files.example.test/pkg-1.0.tar.gz",
                "raw": [
                    {"url": "https://files.example.test/pkg-0.0.1.tar.gz"},
                    {"url": "https://files.example.test/pkg-0.0.2.tar.gz"}
                ]
            }
        });
        let bytes = serde_json::to_vec(&doc).expect("serialize");
        let subject = provenance_subject(&bytes).expect("subject reference");
        assert_eq!(
            subject.locator,
            RefLocator::Url("https://files.example.test/pkg-1.0.tar.gz".into()),
            "the catalogue must not supply the reference"
        );
        // The recorded digest pins the fetch.
        assert_eq!(
            subject.content_sha256.as_deref(),
            Some("a".repeat(64).as_str())
        );
        assert_eq!(subject.kind, RefKind::UrlFetch);

        // Both provenance shapes are recognized; an ordinary JSON root is not.
        assert!(is_provenance_document("json", "pkg-1.0.tar.gz.forage.json"));
        assert!(is_provenance_document(
            "registry",
            "left-pad@1.3.0.registry.json"
        ));
        assert!(!is_provenance_document("json", "package.json"));
        assert!(!is_provenance_document("json", "forage.json.txt"));

        // A document with no subject yields nothing rather than falling back to
        // the catalogue.
        let empty =
            serde_json::to_vec(&serde_json::json!({"registry": {"raw": []}})).expect("serialize");
        assert!(provenance_subject(&empty).is_none());
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
    fn off_host_platform_matches_cargo_target_naming() {
        // Rust target spellings: `windows`/`i686`/`x86_64`/`aarch64`. The
        // windows-rs platform crates ship multi-megabyte import libraries that
        // no Linux host will ever link; they are the cargo analogue of npm's
        // `cli-win32-x64`.
        let host = ("linux", "x64");
        assert!(off_host_platform(
            &purl_ref("pkg:cargo/windows_i686_gnu@0.52.0"),
            host
        ));
        assert!(off_host_platform(
            &purl_ref("pkg:cargo/windows_x86_64_gnu@0.53.0"),
            host
        ));
        assert!(off_host_platform(
            &purl_ref("pkg:cargo/windows_x86_64_gnullvm@0.52.4"),
            host
        ));
        assert!(off_host_platform(
            &purl_ref("pkg:cargo/windows_aarch64_msvc@0.48.5"),
            host
        ));
        // A same-OS other-arch name still ages out on arch alone.
        assert!(off_host_platform(
            &purl_ref("pkg:npm/app-linux-aarch64@1.0.0"),
            host
        ));
        // The host's own spelling variants are kept.
        assert!(!off_host_platform(
            &purl_ref("pkg:npm/app-linux-x86_64@1.0.0"),
            host
        ));
        // musl variants are a different platform from a glibc host, wasm
        // sandbox builds never match a real host, and both directions of the
        // sharp/libvips naming are recognized.
        assert!(off_host_platform(
            &purl_ref("pkg:npm/@img/sharp-libvips-linuxmusl-x64@1.3.2"),
            host
        ));
        assert!(off_host_platform(
            &purl_ref("pkg:npm/@img/sharp-linuxmusl-arm64@0.35.3"),
            host
        ));
        assert!(off_host_platform(
            &purl_ref("pkg:npm/@img/sharp-freebsd-wasm32@0.35.3"),
            host
        ));
        assert!(off_host_platform(
            &purl_ref("pkg:npm/@img/sharp-webcontainers-wasm32@0.35.3"),
            host
        ));
        // ...while the host's real variant stays fetchable.
        assert!(!off_host_platform(
            &purl_ref("pkg:npm/@img/sharp-libvips-linux-x64@1.3.2"),
            host
        ));
        // A musl host keeps its own variants and skips the glibc one.
        let musl_host = ("linuxmusl", "x64");
        assert!(!off_host_platform(
            &purl_ref("pkg:npm/@img/sharp-linuxmusl-x64@0.35.3"),
            musl_host
        ));
        assert!(off_host_platform(
            &purl_ref("pkg:npm/@img/sharp-linux-x64@0.35.3"),
            musl_host
        ));
        // Portable cargo crates carry no os+arch pair.
        assert!(!off_host_platform(
            &purl_ref("pkg:cargo/windows@0.52.0"),
            host
        ));
        assert!(!off_host_platform(
            &purl_ref("pkg:cargo/windows-sys@0.52.0"),
            host
        ));
        assert!(!off_host_platform(&purl_ref("pkg:cargo/serde@1.0.0"), host));
    }

    #[test]
    fn versioned_purl_splits_and_exempts() {
        assert_eq!(
            versioned_purl("pkg:cargo/syn@2.0.104"),
            Some(("pkg:cargo/syn", "2.0.104"))
        );
        // npm scoped names keep their leading @ in the key.
        assert_eq!(
            versioned_purl("pkg:npm/@babel/core@7.24.0"),
            Some(("pkg:npm/@babel/core", "7.24.0"))
        );
        // No version, or a non-numeric ref, exempts the reference.
        assert_eq!(versioned_purl("pkg:npm/@scope/name"), None);
        assert_eq!(versioned_purl("pkg:cargo/serde"), None);
        assert_eq!(versioned_purl("pkg:generic/x@deadbeef"), None);
    }

    #[test]
    fn versionless_dep_superseded_only_when_coordinate_is_pinned() {
        // The lockfile pin for `puppeteer` is present in the tree.
        let pinned: HashSet<String> = ["pkg:npm/puppeteer".to_string()].into_iter().collect();

        // The manifest's version-stripped `pkg:npm/puppeteer` loses to the pin.
        assert!(superseded_by_pin(&purl_ref("pkg:npm/puppeteer"), &pinned));
        // The pin itself is kept — it is versioned, not a bare coordinate.
        assert!(!superseded_by_pin(
            &purl_ref("pkg:npm/puppeteer@10.4.2"),
            &pinned
        ));
        // A different, unpinned coordinate keeps its versionless fallback.
        assert!(!superseded_by_pin(&purl_ref("pkg:npm/left-pad"), &pinned));
        // A git/tag ref is not versionless and never equals a bare coordinate.
        assert!(!superseded_by_pin(
            &purl_ref("pkg:npm/puppeteer@dev"),
            &pinned
        ));
        // Non-PURL locators are out of scope.
        assert!(!superseded_by_pin(
            &url_ref("https://example.test/x.tgz"),
            &pinned
        ));

        // Scoped npm names: the bare scoped coordinate loses to its scoped pin.
        let scoped: HashSet<String> = ["pkg:npm/@puppeteer/browsers".to_string()]
            .into_iter()
            .collect();
        assert!(superseded_by_pin(
            &purl_ref("pkg:npm/@puppeteer/browsers"),
            &scoped
        ));
        assert!(!superseded_by_pin(
            &purl_ref("pkg:npm/@puppeteer/browsers@3.2.0"),
            &scoped
        ));
    }

    #[test]
    fn lenient_version_cmp_orders_release_schemes() {
        use std::cmp::Ordering::*;
        let cmp = lenient_version_cmp;
        assert_eq!(cmp("2.0.104", "2.0.9"), Greater);
        assert_eq!(cmp("1.2", "1.2.1"), Less);
        assert_eq!(cmp("0.52.0", "0.48.5"), Greater);
        assert_eq!(cmp("1.0.0", "1.0.0"), Equal);
        // pep440-ish and date-like schemes still order sensibly.
        assert_eq!(cmp("0.1.5rc1", "0.1.4"), Greater);
        assert_eq!(cmp("20260528.18.2", "20260101.1.1"), Greater);
        // Numeric outranks a text suffix at the same position.
        assert_eq!(cmp("1.2.3", "1.2.rc1"), Greater);
    }

    #[test]
    fn fetch_policy_parses_kinds_and_rejects_garbage() {
        assert_eq!(
            FetchPolicy::parse_follow("dependencies,references"),
            Ok(FetchPolicy {
                urls: true,
                packages: true,
                deps: true,
                ..FetchPolicy::default()
            })
        );
        let actions = FetchPolicy::parse_follow("ci-actions").unwrap();
        assert!(actions.ci && actions.deps);
        assert_eq!(
            FetchPolicy::parse_follow("none"),
            Ok(FetchPolicy::default())
        );
        assert!(FetchPolicy::parse_follow("deps").is_err());
        assert!(FetchPolicy::parse_follow("none,references").is_err());
    }

    /// `follow_name` inverts `parse_follow`. This is the property the header
    /// rests on: a caller files an answer under the name we return, so a name
    /// that does not parse back to the policy that produced it files the
    /// verdict under a question nobody asked.
    #[test]
    fn follow_name_round_trips_through_parse_follow() {
        for spelling in [
            "none",
            "dependencies",
            "references",
            "dependencies,references",
            "all",
        ] {
            let policy = FetchPolicy::parse_follow(spelling).unwrap();
            let name = policy
                .follow_name()
                .expect("vocabulary spelling has a name");
            assert_eq!(name, spelling, "{spelling} did not round-trip");
            assert_eq!(
                FetchPolicy::parse_follow(&name).unwrap().selection_bits(),
                policy.selection_bits(),
                "{spelling} reparsed to a different selection",
            );
        }

        // `ci-actions` implies `dependencies`, so its canonical name says so
        // rather than echoing the shorthand back.
        let actions = FetchPolicy::parse_follow("ci-actions").unwrap();
        assert_eq!(
            actions.follow_name().as_deref(),
            Some("dependencies,ci-actions")
        );
        assert_eq!(
            FetchPolicy::parse_follow("dependencies,ci-actions")
                .unwrap()
                .selection_bits(),
            actions.selection_bits(),
        );

        // The full set is spelled `all`, not enumerated, so one policy has one
        // name.
        let every = FetchPolicy::parse_follow("dependencies,references,ci-actions").unwrap();
        assert_eq!(every.follow_name().as_deref(), Some("all"));

        // A legacy alias can set half of `references`, which the customer
        // vocabulary cannot spell. Unnameable is reported, never approximated.
        let half = FetchPolicy {
            urls: true,
            packages: false,
            ..FetchPolicy::default()
        };
        assert_eq!(half.follow_name(), None);

        // Legacy CLI values remain aliases for existing scripts.
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
        // `all` is shorthand for every kind, CI included.
        assert_eq!(
            "all".parse(),
            Ok(FetchPolicy {
                urls: true,
                packages: true,
                deps: true,
                ci: true,
                ..FetchPolicy::default()
            })
        );
        assert_eq!(
            "all".parse::<FetchPolicy>(),
            "urls,packages,deps,ci".parse()
        );
        // A routine `deps` fetch leaves CI off — GitHub Actions run only in CI
        // and never reach an installed artifact; `ci` (or `all`) opts in.
        assert!(!"deps".parse::<FetchPolicy>().unwrap().ci);
        assert!(!"urls,packages,deps".parse::<FetchPolicy>().unwrap().ci);
        // `ci` turns on the CI context *and* dependency fetching, since a CI
        // action is a declared dependency.
        let ci = "ci".parse::<FetchPolicy>().unwrap();
        assert!(ci.ci && ci.deps);
        // Parsing leaves depth at its default — the CLI sets it separately.
        assert_eq!(
            "deps".parse::<FetchPolicy>().unwrap().depth,
            DEFAULT_FETCH_DEPTH
        );
        assert_eq!(
            FetchPolicy::default().max_file_fetches,
            DEFAULT_MAX_FILE_FETCHES
        );
        assert_eq!(
            FetchPolicy::default().max_url_fetches,
            DEFAULT_MAX_URL_FETCHES
        );
        assert!("".parse::<FetchPolicy>().is_err());
        assert!("sigs".parse::<FetchPolicy>().is_err());
        // A truly retired vocabulary is a hard error, not a silent no-op.
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
    fn declared_deps_stop_at_the_first_hop_unless_transitive() {
        // Hop 0 is the artifact's own declared supply chain and is always
        // followed. Past it, declared dependencies are the transitive tail —
        // a registry lookup each, almost all of it aged out — so an interactive
        // policy drops them while the dropper kinds keep going.
        let mut policy: FetchPolicy = "all".parse().unwrap();
        assert!(!policy.transitive_deps, "interactive default");
        assert!(policy.wants_at(RefKind::Dependency, 0));
        assert!(!policy.wants_at(RefKind::Dependency, 1));
        for hop in 0..3 {
            assert!(policy.wants_at(RefKind::UrlFetch, hop), "hop {hop}");
            assert!(policy.wants_at(RefKind::Command, hop), "hop {hop}");
        }
        // A corpus-facing role takes the whole closure.
        policy.transitive_deps = true;
        assert!(policy.wants_at(RefKind::Dependency, 3));
        // The hop rule never *adds* a kind the selection left out.
        let urls: FetchPolicy = "urls".parse().unwrap();
        assert!(!urls.wants_at(RefKind::Dependency, 0));
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

        let groups = collect_references(&report, &df, CiRefs::Skip);
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
        let groups =
            collect_references(&report, std::path::Path::new("/nonexistent"), CiRefs::Skip);
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
    fn member_import_does_not_refetch_a_locally_resolvable_npm_package() {
        let report: AnalysisReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [
                { "id": 0, "path": "bundle/node_modules/express/package.json", "depth": 1,
                  "file_type": "package.json", "sha256": "a".repeat(64), "size": 80u64 },
                { "id": 1, "path": "bundle/node_modules/@scope/tool/package.json", "depth": 1,
                  "file_type": "package.json", "sha256": "b".repeat(64), "size": 80u64 },
                { "id": 2, "path": "bundle/lib/index.js", "depth": 1,
                  "file_type": "javascript", "sha256": "c".repeat(64), "size": 200u64,
                  "filefacts": { "symbols": [
                      {"kind": "call", "target": "require",
                       "args": [{"shape": "string", "value": "express"}]},
                      {"kind": "call", "target": "require",
                       "args": [{"shape": "string", "value": "@scope/tool/subpath"}]},
                      {"kind": "call", "target": "require",
                       "args": [{"shape": "string", "value": "not-vendored"}]}
                  ] } }
            ]
        }))
        .expect("report deserializes");

        let groups =
            collect_references(&report, std::path::Path::new("/nonexistent"), CiRefs::Skip);
        let locs: Vec<String> = groups
            .iter()
            .flat_map(|(_, refs)| refs.iter().map(locator_key))
            .collect();
        assert_eq!(
            locs,
            vec!["pkg:npm/not-vendored"],
            "only an import absent from the ancestor node_modules is external"
        );
    }

    #[test]
    fn sibling_node_modules_does_not_suppress_an_external_import() {
        let report: AnalysisReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [
                { "id": 0, "path": "one/node_modules/express/package.json", "depth": 1,
                  "file_type": "package.json", "sha256": "a".repeat(64), "size": 80u64 },
                { "id": 1, "path": "two/index.js", "depth": 1,
                  "file_type": "javascript", "sha256": "b".repeat(64), "size": 200u64,
                  "filefacts": { "symbols": [
                      {"kind": "call", "target": "require",
                       "args": [{"shape": "string", "value": "express"}]}
                  ] } }
            ]
        }))
        .expect("report deserializes");

        let groups =
            collect_references(&report, std::path::Path::new("/nonexistent"), CiRefs::Skip);
        let locs: Vec<String> = groups
            .iter()
            .flat_map(|(_, refs)| refs.iter().map(locator_key))
            .collect();
        assert_eq!(locs, vec!["pkg:npm/express"]);
    }

    #[test]
    fn vendored_member_does_not_download_a_missing_import() {
        let report: AnalysisReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [
                { "id": 0, "path": "bundle/node_modules/qs/test/parse.js", "depth": 1,
                  "file_type": "javascript", "sha256": "a".repeat(64), "size": 200u64,
                  "filefacts": { "symbols": [
                      {"kind": "call", "target": "require",
                       "args": [{"shape": "string", "value": "test-only-package"}]}
                  ] } }
            ]
        }))
        .expect("report deserializes");

        let groups =
            collect_references(&report, std::path::Path::new("/nonexistent"), CiRefs::Skip);
        assert!(
            groups.is_empty(),
            "an absent import inside the captured install tree is not a network fetch"
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

    /// A fetched subtree is renamed to its locator — root and members alike — so
    /// the merged report says where the bytes came from, and so two dependencies
    /// whose URLs end in the same basename stay distinct. Anything keyed on path
    /// (the dependency appendix walks a "<root>!!" prefix) merged them before.
    #[test]
    fn merge_payload_renames_the_subtree_to_its_locator() {
        let sub_report = |name: &str| -> AnalysisReport {
            serde_json::from_value(serde_json::json!({
                "version": "3",
                "files": [
                    {"id": 0, "path": name, "depth": 0, "file_type": "npm",
                     "sha256": "d".repeat(64), "size": 64u64},
                    {"id": 1, "parent_id": 0, "path": format!("{name}!!lib/a.js"), "depth": 1,
                     "file_type": "javascript", "sha256": "e".repeat(64), "size": 32u64},
                ],
            }))
            .expect("sub report")
        };
        let rec_for = |locator: &str, url: &str| FetchRecord {
            source_sha256: String::new(),
            source_offset: None,
            kind: RefKind::Dependency,
            locator: locator.to_string(),
            resolved_url: url.to_string(),
            final_url: None,
            redirects: Vec::new(),
            status: None,
            headers: Vec::new(),
            fetched_at: 0,
            content_sha256: Some("d".repeat(64)),
            size: None,
            cached: false,
            stale: false,
            pin_verified: None,
            outcome: Outcome::Ok,
        };

        let mut report: AnalysisReport =
            serde_json::from_value(serde_json::json!({"version": "3", "files": []}))
                .expect("root report");

        // Two dependencies whose URLs share a basename — the collision case.
        for (locator, url) in [
            ("pkg:npm/alpha@1.0.0", "https://a.test/index.js"),
            ("pkg:npm/beta@2.0.0", "https://b.test/index.js"),
        ] {
            merge_payload(
                &mut report,
                &rec_for(locator, url),
                Analyzed {
                    sub: Some(sub_report("index.js")),
                    content_sha: "d".repeat(64),
                    next_from_bytes: Vec::new(),
                },
            );
        }

        let paths: Vec<&str> = report.files.iter().map(|f| f.path.as_str()).collect();
        assert!(
            paths.contains(&"pkg:npm/alpha@1.0.0") && paths.contains(&"pkg:npm/beta@2.0.0"),
            "each dependency root is named by its locator: {paths:?}",
        );
        assert!(
            paths.contains(&"pkg:npm/alpha@1.0.0!!lib/a.js")
                && paths.contains(&"pkg:npm/beta@2.0.0!!lib/a.js"),
            "members follow their root, so prefix lookups still reach them: {paths:?}",
        );
        assert!(
            !paths.iter().any(|p| p.starts_with("index.js")),
            "no node keeps the URL basename that made the two collide: {paths:?}",
        );
    }

    #[test]
    fn payload_name_prefers_url_basename_then_falls_back_to_hash() {
        let mut rec = FetchRecord {
            source_sha256: String::new(),
            source_offset: None,
            kind: RefKind::Dependency,
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
