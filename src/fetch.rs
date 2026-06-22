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

use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use cleave::{AnalysisOptions, AnalysisReport};
use fletch::fetch::{
    BlobCache, FetchBudget, FetchRecord, HttpFetch, Outcome, fetch_ref, fetch_references,
};
use fletch::{ExternalRef, RefKind, RefLocator, find};

/// Default fetch recursion depth — the number of hops followed from the root.
/// `2` reaches a stage-3 payload (root → stage-2 → stage-3), since multi-stage
/// `curl | bash` droppers are the common case.
pub const DEFAULT_FETCH_DEPTH: u8 = 2;

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
}

impl Default for FetchPolicy {
    fn default() -> Self {
        Self {
            urls: false,
            packages: false,
            deps: false,
            depth: DEFAULT_FETCH_DEPTH,
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

/// Discover, fetch, and graft, following references up to `policy.depth` hops.
/// Mutates `report.files` in place with one node per fetched payload (and any
/// extracted members) and returns the fetch edge log. A disabled policy, an
/// unavailable cache/client, or zero references all yield an empty log.
pub(crate) fn orchestrate(
    report: &mut AnalysisReport,
    root_path: &Path,
    policy: FetchPolicy,
    progress: bool,
) -> Vec<FetchRecord> {
    if !policy.enabled() {
        return Vec::new();
    }
    let Some(res) = shared_resources() else {
        return Vec::new();
    };

    let opts = AnalysisOptions::default();
    let mut records = Vec::new();
    // One budget across the whole run (every hop, every file) so a crafted chain
    // can't multiply the per-file cap into a fetch storm. Refs past the cap are
    // recorded as `BudgetExceeded`, never silently dropped.
    let mut budget = FetchBudget::default();
    // Loop guard: a locator is fetched at most once per run, so a chain that
    // points back at an earlier stage can't cycle.
    let mut seen: HashSet<String> = HashSet::new();

    // Hop 0's work-list: declared references from every file in the report plus
    // fletch's imperative discovery over the root sample's bytes. Each later hop
    // works from the references found *inside* the previous hop's payloads.
    // Whether the "fetching external references" header has been emitted — it is
    // printed lazily before the first fetched record so a run that fetches
    // nothing (everything filtered out) stays silent.
    let mut header_printed = false;
    let mut worklist = collect_references(report, root_path);
    // Phantom-dependency signal: a package imperatively installed or loaded
    // somewhere in this artifact but absent from its manifest's declared deps —
    // a covertly-installed companion or a dependency-confusion target. Computed
    // across the whole work-list so a member's `require("x")` is diffed against
    // the root manifest's declarations.
    let all_refs: Vec<ExternalRef> = worklist
        .iter()
        .flat_map(|(_, refs)| refs.iter().cloned())
        .collect();
    for u in find::undeclared_packages(&all_refs) {
        tracing::warn!(
            package = %locator_key(u),
            source = %u.source,
            "undeclared dependency: imperatively acquired but not declared in manifest"
        );
    }
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
            let selected: Vec<ExternalRef> = refs
                .into_iter()
                .filter(|r| policy.wants(r.kind) && seen.insert(locator_key(r)))
                .collect();
            if selected.is_empty() {
                continue;
            }
            let fetched = fetch_references(
                &selected,
                &source_sha,
                policy.urls,
                &res.net,
                &res.cache,
                budget,
            );
            // Charge what this group consumed so the next sees the remainder (a
            // `BudgetExceeded` record consumes nothing).
            let spent = fetched
                .iter()
                .filter(|r| !matches!(r.outcome, Outcome::BudgetExceeded))
                .count();
            let bytes: u64 = fetched.iter().filter_map(|r| r.size).sum();
            budget.max_count = budget.max_count.saturating_sub(spent);
            budget.max_bytes = budget.max_bytes.saturating_sub(bytes);

            // Report the fetch outcomes up front — they're known the moment the
            // (cached or live) fetch returns, so the operator sees the full
            // edge list immediately rather than drip-fed behind each analysis.
            if progress {
                for rec in &fetched {
                    if !header_printed {
                        eprintln!(
                            "\n  \x1b[38;2;100;180;255m\u{2b07}\x1b[0m  \x1b[38;2;160;160;160mfetching external references\x1b[0m"
                        );
                        header_printed = true;
                    }
                    report_fetch(rec);
                }
            }

            // Analyze the payloads concurrently (bounded), then merge each into
            // the report serially in fetch order so file ids stay deterministic.
            // The analysis — a full cleave pass per payload — is the real cost
            // here; the fetch is near-free on a cache hit.
            let analyzed = analyze_payloads(&fetched, &res.cache, &opts);
            for (rec, payload) in fetched.iter().zip(analyzed) {
                if let Some(payload) = payload {
                    next.extend(merge_payload(report, rec, payload));
                }
            }
            records.extend(fetched);
        }
        worklist = next;
    }
    if header_printed {
        report_summary(&records);
    }
    records
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
    };
    let reference = ExternalRef {
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

/// Render one fetched reference to stderr, distinguishing a live network fetch
/// from a cache hit and naming the actual URL that was (or would be) retrieved.
/// Only the interactive terminal path passes `progress`; JSON/server callers
/// stay silent. Colors mirror the scan progress bar's truecolor palette.
fn report_fetch(rec: &FetchRecord) {
    // (glyph, label, r, g, b, detail) — detail replaces the size column when a
    // fetch never delivered bytes (a failure or a skip).
    let (glyph, label, r, g, b, detail) = match &rec.outcome {
        Outcome::PinMismatch => (
            '\u{2716}',
            "pin!",
            255,
            90,
            90,
            Some("hash mismatch".to_string()),
        ),
        Outcome::Ok if rec.stale => ('\u{25cf}', "stale", 230, 180, 80, None),
        Outcome::Ok if rec.cached => ('\u{25cf}', "cache", 120, 200, 140, None),
        Outcome::Ok => ('\u{2b07}', "live", 100, 180, 255, None),
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
    };

    // The actual URL fetched; fall back to the bare locator (PURL) when the
    // reference never resolved to one.
    let url = if rec.resolved_url.is_empty() {
        rec.locator.as_str()
    } else {
        rec.resolved_url.as_str()
    };
    // A redirect lands the payload elsewhere — show where, dimmed.
    let redirect = match &rec.final_url {
        Some(f) if f != &rec.resolved_url => format!("  \x1b[2m\u{2192} {f}\x1b[0m"),
        _ => String::new(),
    };
    let column = detail.unwrap_or_else(|| rec.size.map_or(String::new(), human_bytes));

    eprintln!(
        "    \x1b[38;2;{r};{g};{b}m{glyph} {label:<6}\x1b[0m \x1b[38;2;130;130;130m{column:>10}\x1b[0m  {url}{redirect}"
    );
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

/// Tally the run's fetches into a one-line summary mirroring the progress bar's
/// completion line: how many came live off the network vs. served from cache,
/// how many failed, and the total bytes pulled.
fn report_summary(records: &[FetchRecord]) {
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
    let mut parts = vec![format!("{live} live"), format!("{cached} cached")];
    if failed > 0 {
        parts.push(format!("{failed} failed"));
    }
    parts.push(human_bytes(bytes));
    eprintln!(
        "  \x1b[38;2;80;220;80m\u{2713}\x1b[0m  \x1b[38;2;160;160;160m{}\x1b[0m",
        parts.join("  \u{b7}  ")
    );
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
fn collect_references(
    report: &AnalysisReport,
    root_path: &Path,
) -> Vec<(String, Vec<ExternalRef>)> {
    let mut groups: Vec<(String, Vec<ExternalRef>)> = Vec::new();
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
    groups: &mut Vec<(String, Vec<ExternalRef>)>,
    root_sha: &str,
    hunted: Vec<ExternalRef>,
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

/// A reference's locator as a stable string for dedup.
fn locator_key(r: &ExternalRef) -> String {
    match &r.locator {
        RefLocator::Purl(p) => p.clone(),
        RefLocator::Url(u) => u.clone(),
    }
}

/// Concurrent analyses of fetched payloads per group. Kept deliberately low:
/// analyzing a single archive is *already* rayon-parallel internally (one job
/// fans its members across the shared pool), so running many payload analyses
/// at once would oversubscribe that pool and thrash rather than speed up. Two
/// overlaps the serial portions (I/O, setup, the container parse) of one
/// analysis with another's member fan-out without contending for the whole CPU.
/// A worker-style global admission scheme could safely go higher, but that is
/// more machinery than this online side-channel warrants.
const ANALYSIS_CONCURRENCY: usize = 2;

/// The product of analyzing one fetched payload: the finalized sub-report to
/// graft (absent if the payload couldn't be analyzed) and the next-hop
/// references found in its own bytes. Produced off the report so the expensive
/// analysis can run concurrently; [`merge_payload`] folds it in serially.
struct Analyzed {
    sub: Option<AnalysisReport>,
    content_sha: String,
    next_from_bytes: Vec<(String, Vec<ExternalRef>)>,
}

/// Analyze the bytes of fetched payloads concurrently, returning one slot per
/// input record in the same order (`None` where there was nothing to analyze).
/// Bounded by [`ANALYSIS_CONCURRENCY`] and run on plain OS threads — not the
/// shared rayon pool each analysis itself uses — so a fetching run never starves
/// the pool. Output order is the input order regardless of completion order, so
/// the serial merge that follows assigns file ids deterministically.
fn analyze_payloads(
    fetched: &[FetchRecord],
    cache: &BlobCache,
    opts: &AnalysisOptions,
) -> Vec<Option<Analyzed>> {
    let n = fetched.len();
    if n == 0 {
        return Vec::new();
    }
    let cursor = AtomicUsize::new(0);
    let workers = ANALYSIS_CONCURRENCY.min(n);
    let collected: Vec<Vec<(usize, Analyzed)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        if let Some(a) = analyze_payload(&fetched[i], cache, opts) {
                            local.push((i, a));
                        }
                    }
                    local
                })
            })
            .collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let mut slots: Vec<Option<Analyzed>> = (0..n).map(|_| None).collect();
    for chunk in collected {
        for (i, a) in chunk {
            slots[i] = Some(a);
        }
    }
    slots
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
) -> Option<Analyzed> {
    // Scan whatever bytes we hold: a clean fetch or a pin mismatch (a mismatch
    // is exactly the case worth analyzing). Skipped/unresolved/failed have none.
    if !matches!(rec.outcome, Outcome::Ok | Outcome::PinMismatch) {
        return None;
    }
    let bytes = cache.load(&rec.locator)?;
    let name = payload_name(rec);
    let content_sha = rec.content_sha256.clone().unwrap_or_default();

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
) -> Vec<(String, Vec<ExternalRef>)> {
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
    let first_new = report.files.len();
    for mut file in sub.files {
        file.id += id_base;
        file.parent_id = Some(file.parent_id.map_or(parent_id, |p| p + id_base));
        file.depth += parent_depth + 1;
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

    fn url_ref(url: &str) -> ExternalRef {
        ExternalRef {
            locator: RefLocator::Url(url.to_string()),
            kind: RefKind::UrlFetch,
            source: "test".to_string(),
            evidence: url.to_string(),
            offset: 0,
            pinned_hash: None,
            content_sha256: None,
        }
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
        let all: Vec<ExternalRef> = groups.iter().flat_map(|(_, r)| r.iter().cloned()).collect();
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
}
