//! Scan-side orchestration of [`fletch`]: discover the external references in
//! an analysis report, fetch them, and graft each retrieved payload back into
//! the report as a uniform file node — so the ML verdict and every downstream
//! consumer (hopper, prism) treat a fetched payload exactly like any other
//! analyzed file.
//!
//! cleave stays a pure offline analyzer; fletch is a pure find/fetch mechanism.
//! This module is the only place the two meet. For every file (root and archive
//! member) it runs fletch's *facts-based* discovery over the references and
//! values cleave retained per file (`FilefactsView`) — declared dependencies
//! plus value-driven hunts like npm lifecycle hooks. For the root sample, where
//! the raw bytes are on disk, it additionally runs the *text-based* hunt
//! (`curl|sh`, `npm install` in a `RUN`). It then retrieves the lot through the
//! SSRF-guarded client and re-analyzes what came back with cleave.
//!
//! The remaining gap: an archive member that is itself a shell script or
//! Dockerfile gets declared/value-based references only, not the text-based
//! command-stream hunt — that needs the member's bytes, which a prior analysis
//! extracted and discarded.
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

use cleave::{AnalysisOptions, AnalysisReport};
use fletch::fetch::{BlobCache, FetchBudget, FetchRecord, HttpFetch, Outcome, fetch_references};
use fletch::{ExternalRef, RefLocator, find};

/// Default fetch recursion depth — the number of hops followed from the root.
/// `2` reaches a stage-3 payload (root → stage-2 → stage-3), since multi-stage
/// `curl | bash` droppers are the common case.
pub const DEFAULT_FETCH_DEPTH: u8 = 2;

/// What to fetch — a selection of reference kinds, parsed from the
/// comma-separated `--fetch=KINDS` flag (`deps`, `refs`) — plus how many hops to
/// follow. An empty kind selection (the default) disables fetching. Designed to
/// grow new kinds (e.g. `sigs`) without changing the flag's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FetchPolicy {
    /// Fetch declared/installed **dependencies** — registry packages (PURLs).
    pub deps: bool,
    /// Fetch bare/arbitrary **references** — `http(s)` URLs (the larger
    /// exposure: curl/wget targets, staged downloads).
    pub refs: bool,
    /// How many hops to follow: `1` fetches the references found in the root,
    /// `2` also follows references found *inside* those payloads, and so on.
    pub depth: u8,
}

impl Default for FetchPolicy {
    fn default() -> Self {
        Self {
            deps: false,
            refs: false,
            depth: DEFAULT_FETCH_DEPTH,
        }
    }
}

impl FetchPolicy {
    /// True when at least one kind is selected — the master switch.
    #[must_use]
    pub(crate) const fn enabled(&self) -> bool {
        self.deps || self.refs
    }
}

impl std::str::FromStr for FetchPolicy {
    type Err = String;

    /// Parse a comma-separated kind list (`deps`, `refs`). Empty entries are
    /// ignored; an unknown kind or an empty selection is an error so a typo is
    /// never silently a no-op.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut policy = Self::default();
        for kind in s.split(',') {
            match kind.trim() {
                "" => {}
                "deps" => policy.deps = true,
                "refs" => policy.refs = true,
                other => {
                    return Err(format!("unknown fetch kind {other:?} (valid: deps, refs)"));
                }
            }
        }
        if !policy.enabled() {
            return Err("empty fetch selection (valid: deps, refs)".to_string());
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
    let mut worklist = collect_references(report, root_path);
    for _hop in 0..policy.depth {
        if worklist.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for (source_sha, refs) in std::mem::take(&mut worklist) {
            // Keep only the kinds this policy selected (packages need `deps`,
            // URLs need `refs`) and that haven't been fetched yet this run.
            let selected: Vec<ExternalRef> = refs
                .into_iter()
                .filter(|r| {
                    let wanted = match r.locator {
                        RefLocator::Purl(_) => policy.deps,
                        RefLocator::Url(_) => policy.refs,
                    };
                    wanted && seen.insert(locator_key(r))
                })
                .collect();
            if selected.is_empty() {
                continue;
            }
            let fetched = fetch_references(
                &selected,
                &source_sha,
                policy.refs,
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
            for rec in &fetched {
                next.extend(graft(report, rec, &res.cache, &opts));
            }
            records.extend(fetched);
        }
        worklist = next;
    }
    records
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
        let refs = find::references_from_facts(&view.values, &view.references);
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

/// Analyze a fetched payload (when one was retrieved), append its file nodes to
/// the report nested under the file that declared the reference, and return the
/// references found *inside* the payload — the next hop's work. The fetch edge
/// (`source_sha256 → content_sha256`) is the authoritative link; ids and depth
/// are renumbered so the grafted nodes are a well-formed subtree that never
/// collides with the main report's.
fn graft(
    report: &mut AnalysisReport,
    rec: &FetchRecord,
    cache: &BlobCache,
    opts: &AnalysisOptions,
) -> Vec<(String, Vec<ExternalRef>)> {
    // Scan whatever bytes we hold: a clean fetch or a pin mismatch (a mismatch
    // is exactly the case worth analyzing). Skipped/unresolved/failed have none.
    if !matches!(rec.outcome, Outcome::Ok | Outcome::PinMismatch) {
        return Vec::new();
    }
    let Some(bytes) = cache.load(&rec.locator) else {
        return Vec::new();
    };
    let name = payload_name(rec);
    let content_sha = rec.content_sha256.clone().unwrap_or_default();

    // Next-hop references discovered in the payload's own bytes — the full hunt,
    // so a stage-2 script's `curl | bash` (or an encoded URL) is followed.
    let mut next = Vec::new();
    let payload_refs = find::references_in_bytes(&bytes, &name);
    if !content_sha.is_empty() && !payload_refs.is_empty() {
        next.push((content_sha.clone(), payload_refs));
    }

    let mut sub = match cleave::analyze_bytes_owned(bytes, &name, opts) {
        Ok(sub) => sub,
        Err(e) => {
            tracing::warn!("analysis of fetched {} failed: {e:#}", rec.locator);
            return next;
        }
    };
    // finalize() collapses the sub-analysis into its files[]; without it the
    // payload's data stays in top-level fields and files[] is empty.
    sub.finalize();

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
        if file.sha256 == content_sha {
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
            "refs".parse(),
            Ok(FetchPolicy {
                refs: true,
                ..FetchPolicy::default()
            })
        );
        assert_eq!(
            " deps , refs ".parse(),
            Ok(FetchPolicy {
                deps: true,
                refs: true,
                ..FetchPolicy::default()
            })
        );
        // Parsing leaves depth at its default — the CLI sets it separately.
        assert_eq!(
            "deps".parse::<FetchPolicy>().unwrap().depth,
            DEFAULT_FETCH_DEPTH
        );
        assert!("".parse::<FetchPolicy>().is_err());
        assert!("sigs".parse::<FetchPolicy>().is_err());
        assert!("deps,bogus".parse::<FetchPolicy>().is_err());
        assert!(!FetchPolicy::default().enabled());
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
