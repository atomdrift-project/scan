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
use std::sync::{Arc, Mutex, OnceLock, PoisonError, RwLock};
use std::time::{Duration, Instant};

use cleave::{AnalysisOptions, AnalysisReport, Finding};
use fletch::fetch::{
    BlobCache, CandidateSource, Coverage, Fetch, FetchBudget, FetchError, FetchRecord, Fetched,
    HttpFetch, Method, Outcome, RecordedSource, Request, Served, UrlFetches, fetch_ref,
    fetch_references_with,
};
use fletch::{RefKind, RefLocator, Reference, Registry, find};
use reqwest::Url;

use crate::analysis_cache::AnalysisCache;
use crate::corpus_precheck::{Precheck, Standing, Verdict};
use crate::deptree::{DepState, DepTree};
use crate::hosts::{self, UrlHost};
use crate::output::{Rgb, fg};
use crate::provenance::RegistryProvenance;

mod age_gate;
mod budget;
mod discover;
mod imports;
mod merge;
mod pending;
mod policy;
mod registry;
mod render;
mod session;
mod urls;

pub use self::{age_gate::*, budget::*, merge::*, policy::*, registry::*, render::*};
#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use self::{discover::*, imports::*, session::*, urls::*};

/// Unfinished fetch work kept for a later run; see
/// [`crate::settings::Settings::fetch_backlog`].
pub(crate) use pending::Store as Backlog;

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

fn settings() -> Settings {
    crate::settings::get().fetch
}

/// fletch's blob cache under the configured [`Settings`]. Everything that reads
/// fetched bytes back opens it here, so it can read whatever a fetch admitted.
///
/// # Errors
/// When there is no OS cache directory.
pub fn open_blob_cache() -> std::io::Result<BlobCache> {
    let settings = settings();
    Ok(BlobCache::open()?
        .with_max_bytes(settings.max_fetch_bytes)
        .with_registry_ttl(settings.registry_ttl))
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

/// Whether a PURL names one immutable release: a concrete version
/// (`1.2.3`, `v0.0.0-2024…`) or a full commit hash. A version-less PURL
/// (`npx foo` → `pkg:npm/foo`), a dist-tag (`@latest`), a range (`^1.0`) or a
/// branch moves under the caller: a corpus answer for it may describe a
/// different, older release than the one a fetch would get, so it must never
/// stand in for that fetch.
fn pins_exact_version(purl: &str) -> bool {
    let Some(version) = Coordinate::of(purl).and_then(|c| c.version) else {
        return false;
    };
    let numeric = version.strip_prefix('v').unwrap_or(version);
    let commit = matches!(version.len(), 40 | 64) && version.bytes().all(|b| b.is_ascii_hexdigit());
    (numeric.starts_with(|c: char| c.is_ascii_digit()) || commit)
        && !version.contains(['*', '^', '~', '<', '>', '=', '|', ',', ' '])
        && !version
            .split('.')
            .any(|part| part.eq_ignore_ascii_case("x"))
}

/// How long after its first file a PyPI release stops accepting new ones:
/// PyPI's 14 days (July 2026), plus a day because PyPI does not say where it
/// counts from and our clock is not theirs.
const PYPI_RELEASE_CLOSED_SECS: u64 = 15 * 86_400;

/// Whether a PURL names bytes that can never change: a registry that refuses
/// to replace a published version (npm, crates.io, the Go checksum database,
/// NuGet, Maven Central releases), a PyPI release past
/// [`PYPI_RELEASE_CLOSED_SECS`] (`age_secs` is the release's age, `None` when
/// unknown), or a VCS coordinate whose version is itself a commit hash. Only
/// there is a hex version a commit: a 40-digit PyPI version is an ordinary
/// (still open) release, and a 64-hex container tag is just a tag — a real
/// digest is spelled `sha256:…`. Everywhere else the
/// same coordinate can name new bytes, so nothing stored against it may stand
/// in for a fetch: git tags, gems, container tags, distro packages, a Maven
/// `-SNAPSHOT`, a young PyPI release, anything served from a non-default
/// repository.
pub(crate) fn immutable_coordinate(purl: &str, age_secs: Option<u64>) -> bool {
    let Some(parsed) = fletch::purl::Purl::parse(purl).ok() else {
        return false;
    };
    let Some(version) = parsed.version() else {
        return false;
    };
    let pinned = default_source(&parsed);
    let commit = matches!(parsed.typ(), "github" | "gitlab" | "bitbucket")
        && matches!(version.len(), 40 | 64)
        && version.bytes().all(|b| b.is_ascii_hexdigit());
    let fixed_registry = match parsed.typ() {
        "npm" | "cargo" | "golang" | "nuget" => true,
        "maven" => !version.ends_with("-SNAPSHOT"),
        "pypi" => age_secs.is_some_and(|age| age > PYPI_RELEASE_CLOSED_SECS),
        _ => false,
    };
    pinned && pins_exact_version(purl) && (commit || fixed_registry)
}

/// Whether `purl` names one file whose bytes can never change: an
/// [`immutable_coordinate`], or a single PyPI file (`file_name`) of an exact
/// version. PyPI never lets a filename be replaced or reused, even after
/// deletion, though a young release can still gain *other* files.
///
/// Only for a cache keyed by that exact file, such as a URL's alias. The
/// corpus, the bloom filters and isotope13 key a PURL by its release and drop
/// `file_name`, so there a young release must stay mutable: one file's verdict
/// must not vouch for a different file added to it later.
pub(crate) fn immutable_file(purl: &str) -> bool {
    if immutable_coordinate(purl, None) {
        return true;
    }
    let Some(parsed) = fletch::purl::Purl::parse(purl).ok() else {
        return false;
    };
    parsed.typ() == "pypi"
        && parsed
            .qualifier("file_name")
            .is_some_and(|name| !name.is_empty())
        && default_source(&parsed)
        && pins_exact_version(purl)
}

/// Whether `parsed` is fetched from its type's default registry: no qualifier
/// points it at other bytes or another index.
fn default_source(parsed: &fletch::purl::Purl) -> bool {
    ["download_url", "vcs_url", "repository_url"]
        .iter()
        .all(|q| parsed.qualifier(q).is_none())
}

/// The sha256 a registry lists for the exact file a fetch of `purl` would
/// download — the [`fletch::fetch::ArtifactMatrix::preferred`] candidate the
/// fetch itself picks. `None` when the registry lists no sha256 for it.
fn registry_digest(purl: &str, net: &dyn Fetch, cache: &BlobCache) -> Option<String> {
    let matrix = fletch::fetch::resolve_artifacts(&RefLocator::Purl(purl.to_owned()), net, cache)?;
    let sha = matrix.preferred()?.checksums.get("sha256")?;
    (sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| sha.to_ascii_lowercase())
}

/// The root sample's imperative hunt re-reads it from disk and re-parses it.
/// Skip that for large roots — the win is scripts/manifests/Dockerfiles, which
/// are small; a multi-megabyte binary root has no imperative install commands to
/// find and would just pay a wasted parse. Declared references are read from the
/// report regardless, so nothing is lost for large roots.
const ROOT_HUNT_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// The transport and blob cache every fetch goes through. A test supplies its
/// own transport; everything else shares [`shared_resources`].
struct Resources {
    net: Box<dyn Fetch + Send>,
    cache: BlobCache,
}

/// The process's HTTP transport and blob cache, built on the first fetching
/// analysis and shared by every file and rayon worker after it. `None` when
/// either cannot be built — that scan fetches nothing, and the next one tries
/// again rather than inheriting the failure.
fn shared_resources() -> Option<&'static Resources> {
    static RESOURCES: OnceLock<Resources> = OnceLock::new();
    if let Some(res) = RESOURCES.get() {
        return Some(res);
    }
    let net = HttpFetch::new()
        .map_err(|e| tracing::warn!("fetch disabled: http client unavailable: {e:#}"))
        .ok();
    let cache = open_blob_cache()
        .map_err(|e| tracing::warn!("fetch disabled: blob cache unavailable: {e:#}"))
        .ok();
    let (net, cache) = net.zip(cache)?;
    Some(RESOURCES.get_or_init(|| Resources {
        net: Box::new(net.with_max_bytes(settings().max_fetch_bytes)),
        cache,
    }))
}

/// A fetched dependency captured for upload to hopper as its own sample. Carries
/// the standalone analysis report cleave produced for the dependency's bytes (the
/// same report a first-hand `pkg:`/`url` scan yields, stripped and compacted),
/// plus the shas of every file in it — so the caller can harvest the dependency's
/// aggregate verdict from the embedded-classification pass it already runs over
/// the merged report, without re-running the model.
#[derive(Debug)]
pub struct FetchedDependency {
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
    /// costs 3-6x the text size, and several jobs' dependencies can be
    /// co-resident in one process from graft until their result POST, where
    /// retained `Value` trees reach hundreds of megabytes. The envelope build
    /// parses it back transiently.
    pub raw: String,
}

/// Registry provenance materialized while walking a dependency graph.
///
/// This is separate from [`FetchedDependency`] because a dependency can have
/// notable registry findings even when its artifact is age-gated or removed and
/// therefore never fetched. `file_id` ties the record to the exact sidecar node
/// cleave analyzed, including composite-source attribution.
#[derive(Debug, Clone)]
pub struct DependencyRegistry {
    /// Package or URL locator associated with this registry result.
    pub locator: String,
    /// Normalized provenance collected from the registry.
    pub provenance: crate::provenance::RegistryProvenance,
    /// ID of the cleave report node associated with this result.
    pub file_id: u32,
    /// Why the artifact was not fetched, if it was skipped.
    pub artifact_skip: Option<&'static str>,
}

/// What one fetch phase produced: the edges it recorded, the dependencies it
/// captured for upload, the registry documents it materialized, and the corpus
/// verdicts it adopted in place of analyzing bytes (keyed by content sha).
#[derive(Debug, Default)]
pub struct FetchOutcome {
    /// Fetch edges observed while processing the dependency graph.
    pub records: Vec<FetchRecord>,
    /// Downloaded dependencies to include in the result upload.
    pub dependencies: Vec<FetchedDependency>,
    /// Registry provenance records collected during graph traversal.
    pub registries: Vec<DependencyRegistry>,
    /// Existing corpus verdicts adopted instead of analyzing these bytes.
    pub adopted: HashMap<String, crate::corpus_precheck::Verdict>,
}

/// References to fetch, grouped by the sha256 of the file that declared them.
type Group = (String, Vec<Reference>);

/// Payloads a batch gathers before analyzing: enough to keep every core busy
/// across many small groups without holding a whole hop's analyzed reports in
/// memory.
const BATCH_PAYLOAD_TARGET: usize = 256;

/// How [`orchestrate`] follows one artifact's references.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Follow<'a> {
    /// Which references to follow, how far, and within what limits.
    pub policy: FetchPolicy,
    /// Draw fetch progress on the terminal.
    pub progress: bool,
    /// Keep each fetched dependency's standalone report as a
    /// [`FetchedDependency`]. Only hopper uploads, the LLM payload and the
    /// rendered dependency appendix read them; a plain JSON scan skips the
    /// capture and the per-dependency model pass it feeds. Grafting, verdicts
    /// and fetch edges are the same either way.
    pub capture_deps: bool,
    /// Passwords to try on encrypted archives among the payloads.
    pub zip_passwords: &'a [String],
}

#[cfg(test)]
impl Follow<'_> {
    /// `policy`, with no progress, captures or passwords.
    const fn quiet(policy: FetchPolicy) -> Self {
        Self {
            policy,
            progress: false,
            capture_deps: false,
            zip_passwords: &[],
        }
    }
}

/// Discover, fetch, and graft, following references up to `policy.depth` hops.
/// Mutates `report.files` in place with one node per fetched payload (and any
/// extracted members) and returns the fetch edge log plus the standalone report
/// captured for each fetched dependency. A disabled policy, an unavailable
/// cache/client, or zero references all yield empty logs.
pub(crate) fn orchestrate(
    report: &mut AnalysisReport,
    root_path: &Path,
    root_bytes: Option<&[u8]>,
    follow: Follow<'_>,
) -> FetchOutcome {
    let policy = follow.policy;
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
    let worklist = collect_references_from_input(report, root_path, root_bytes, ci);
    attribute_undeclared_npm_imports(report, &worklist);
    let mut session = FetchSession::new(report, &worklist, res, follow);
    session.record_unresolved_fetch_calls(report);
    session.run(report, worklist);
    session.finish(report)
}

#[cfg(test)]
mod retrieval_failure_tests;

/// Whether `fact` says a reference's analysis is missing or has gaps: what a
/// later run retries, and what a summary counts as incomplete.
fn is_analysis_gap(fact: &Coverage) -> bool {
    matches!(fact, Coverage::Unavailable { .. } | Coverage::Incomplete)
}

/// `rec`'s coverage facts as one line for people.
fn coverage_text(rec: &FetchRecord) -> String {
    let facts: Vec<String> = rec.coverage.iter().map(ToString::to_string).collect();
    facts.join("; ")
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
#[must_use]
pub fn locator_str(locator: &RefLocator) -> &str {
    match locator {
        RefLocator::Purl(s) | RefLocator::Url(s) | RefLocator::Path(s) => s,
        _ => "",
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod dependency_gap_tests;

#[cfg(test)]
mod retry_backlog_tests;

#[cfg(test)]
mod scope_inheritance_tests;

/// A transport for tests that must not reach the network, and a cache no other
/// test shares.
#[cfg(test)]
mod tests_offline;

#[cfg(test)]
mod registry_memo_tests;

#[cfg(test)]
mod qualifier_tests;
