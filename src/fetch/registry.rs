//! Fetching one artifact, and what its registry says about it: lookups,
//! the lookup memo, and the registry record grafted beside a package.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// The artifact could not be retrieved: the registry refused it, the locator
/// resolved to no URL, the transport failed, or the budget ran out. The typed
/// outcome distinguishes package refusals from retryable infrastructure faults.
///
/// A distinct type rather than a message, because the difference between "this
/// artifact is not available" and "this server is broken" decides an HTTP
/// status, and a status decides what every caller upstream does next. Beamline
/// reads a 5xx as a sick server: it opens that server's circuit breaker, moves
/// the request to the next one, and reports `unavailable` once the fleet is
/// exhausted — so a package nobody can download, misreported as a 5xx, ejects
/// healthy servers from the pool and arrives at poppy as an outage rather than
/// a download failure. A substring of a `Debug`-formatted outcome can be
/// reworded by accident; the type cannot.
#[derive(Debug)]
pub struct Unretrievable {
    /// What was asked for: the resolved URL, or the locator when there was
    /// none to resolve.
    pub target: String,
    /// How the fetch ended. Carried so a caller can tell a refusal apart from
    /// a budget stop without re-parsing the message.
    pub outcome: Outcome,
}

impl Unretrievable {
    /// Whether the artifact is gone: its source answered that there is no
    /// such release, or there is nothing to fetch it from. Only then is the
    /// registry's record all there is to scan.
    pub(crate) fn gone(&self) -> bool {
        matches!(
            &self.outcome,
            Outcome::Failed(FetchError::Status(404 | 410))
                | Outcome::Unresolved(
                    fletch::fetch::Unresolved::NoRelease | fletch::fetch::Unresolved::Unsupported
                )
        )
    }

    /// Whether a healthier server may retrieve this artifact. Infrastructure
    /// failures and local budget stops say nothing about the package.
    pub(crate) fn infrastructure_failure(&self) -> bool {
        match &self.outcome {
            Outcome::Failed(error) => fetch_infrastructure_failure(error),
            Outcome::BudgetExceeded => true,
            _ => false,
        }
    }
}

pub(super) fn fetch_infrastructure_failure(error: &FetchError) -> bool {
    matches!(
        error,
        FetchError::Transport(_)
            | FetchError::Timeout
            | FetchError::Internal(_)
            | FetchError::Status(408 | 429 | 500..=599)
    )
}

// Registry resolution can turn a failed metadata request into NoRelease.
// Keep its typed failure scoped to this fetch, without changing Fletch's
// optional metadata API or mistaking a DNS outage for a missing package.
pub(super) struct FetchAttempt<'a> {
    pub(super) net: &'a dyn Fetch,
    pub(super) failure: Mutex<Option<FetchError>>,
}

impl Fetch for FetchAttempt<'_> {
    fn send(&self, request: &Request<'_>) -> Result<Fetched, FetchError> {
        let result = self.net.send(request);
        if let Err(error) = &result
            && fetch_infrastructure_failure(error)
        {
            *self.failure.lock().unwrap_or_else(PoisonError::into_inner) = Some(error.clone());
        }
        result
    }

    fn allows_oci(&self) -> bool {
        self.net.allows_oci()
    }
}

pub(super) fn fetch_one_record(
    reference: &Reference,
    net: &dyn Fetch,
    cache: &BlobCache,
) -> FetchRecord {
    let attempt = FetchAttempt {
        net,
        failure: Mutex::new(None),
    };
    let mut record = fetch_ref(reference, &attempt, cache);
    if matches!(record.outcome, Outcome::Unresolved(_))
        && let Some(error) = attempt
            .failure
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
    {
        record.outcome = Outcome::Failed(error);
    }
    record
}

impl std::fmt::Display for Unretrievable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let why = match &self.outcome {
            Outcome::Failed(error) if fetch_infrastructure_failure(error) => error.to_string(),
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
/// matching `--follow`. Errors if the client/cache is unavailable
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
    let rec = fetch_one_record(&reference, &*res.net, &res.cache);
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
    let (record, sources) = fletch::registry::lookup_recorded(locator, &*res.net, &res.cache);
    (
        record
            .ok()
            .map(|reg| reg.with_age(crate::civil::now_secs())),
        sources,
    )
}

/// One-shot `pkg:`/`url`: graft the root artifact's own registry metadata into
/// its finalized report as a child node of the root, then run the package pass.
///
/// The registry record is materialized as a `*.registry.json` node — detected as
/// the `registry` filetype, carrying `registry.*` facts — and merged under the
/// root, exactly as the `--follow` path grafts a dependency's registry beside the
/// dependency. So the registry metadata becomes a real layer of the analyzed
/// package: it is trait-matched, featurized, and trained on like any other file,
/// rather than living in a disconnected side report. With both halves now in one
/// tree, `apply_package_composites` correlates the artifact's behavior with the
/// registry's account of it. A no-op if the report has no root or the record
/// can't be analyzed.
pub fn graft_root_registry(report: &mut AnalysisReport, reg: &Registry) {
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
#[must_use]
pub fn registry_summary(reg: &Registry) -> Vec<String> {
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
    // Registry text is publisher-controlled and lands on a terminal: an author
    // or deprecation notice carrying escapes must not rewrite the verdict.
    parts
        .into_iter()
        .map(|part| crate::output::tty_text(&part).into_owned())
        .collect()
}

/// Most registry requests of each kind (range resolution, record lookup) one
/// manifest may cost. Generous for real lockfiles — a large `Cargo.lock` is
/// ~700 coordinates, a monorepo `package-lock.json` a few thousand — and a hard
/// stop for a crafted one.
pub(super) const MAX_REGISTRY_LOOKUPS: usize = 5_000;

/// Process-wide memo of registry lookups, keyed by locator string. A package
/// named across many scanned files resolves once: the first lookup fills this,
/// and every later file reads the record straight from memory — no repeated disk
/// read, JSON parse, and ecosystem mapping. The *un-aged* record is stored (age
/// is relative to each scan's clock, so [`Registry::with_age`] is applied per
/// read); `None` is memoized too, so an unsupported ecosystem or an unresolved
/// package isn't re-attempted for every file that names it. Lives for the
/// process, fronting the on-disk blob cache, but each entry only for
/// [`REGISTRY_MEMO_TTL_SECS`].
pub(super) fn registry_memo() -> &'static RwLock<lru::LruCache<String, Memoized>> {
    static MEMO: OnceLock<RwLock<lru::LruCache<String, Memoized>>> = OnceLock::new();
    MEMO.get_or_init(|| RwLock::new(lru::LruCache::new(REGISTRY_MEMO_CAPACITY)))
}

/// One registry memo entry: the un-aged record (or the final "none"), and the
/// scan clock when it was looked up.
#[derive(Clone)]
pub(super) struct Memoized {
    pub(super) at: u64,
    pub(super) record: Option<Registry>,
}

/// How long a memo entry stands in for the blob cache. The memo exists to
/// spare a burst of scans naming the same dependencies the disk read and the
/// JSON parse; held for the life of a server, it would also outlive
/// `--registry-ttl` and keep serving a record whose version has since been
/// yanked or security-held. Past this the blob cache answers again, under its
/// own TTL.
pub(super) const REGISTRY_MEMO_TTL_SECS: u64 = 30 * 60;

/// Entries the registry memo keeps. Unbounded, one entry per distinct
/// dependency locator a long-lived process ever resolved would grow to
/// hundreds of MB of `Registry` records (a big packument's `release_times`
/// alone is 40-80 KB). Dependency sets cluster in time, so an LRU of a few
/// thousand serves the same hit rate; a miss falls through to fletch's blob
/// cache.
pub(super) const REGISTRY_MEMO_CAPACITY: std::num::NonZeroUsize =
    match std::num::NonZeroUsize::new(4096) {
        Some(n) => n,
        None => std::num::NonZeroUsize::MIN,
    };

/// Release-cadence window `Registry::with_age` looks back over (48 h).
pub(super) const RELEASE_CADENCE_WINDOW_SECS: u64 = 172_800;

/// Look up each declared dependency's registry record, returning one slot per
/// input ref in `selected` order. A non-dependency ref, or one whose record
/// can't be resolved, yields `None`.
///
/// Fresh misses use `registry_with_sources` once, so the normalized record and
/// provenance come from one lookup. The process memo deliberately retains only
/// the small record; a memo hit leaves the provider documents in fletch's blob
/// cache, read again only for a record that is kept (see [`Gated::sources`])
/// rather than pinned in a long-lived daemon's heap or re-read for every hit.
pub(super) fn lookup_registries(
    selected: &[Reference],
    res: &Resources,
    now: u64,
) -> Vec<Option<Lookup>> {
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
            match memo
                .peek(locator(r))
                .filter(|hit| now.saturating_sub(hit.at) <= REGISTRY_MEMO_TTL_SECS)
            {
                // Stored un-aged; stamp the age signals from this scan's clock.
                Some(hit) => {
                    found[i] = hit.record.clone().map(|record| Lookup {
                        record: record.with_age(now),
                        sources: None,
                    });
                }
                // Absent, or expired: the blob cache decides, under its TTL.
                None => misses.push(i),
            }
        }
    }

    // On the rayon pool, not fresh OS threads: nearly every lookup is a
    // blob-cache hit that parses JSON, i.e. CPU, and a thread spawned per lookup
    // per scanned root serializes on the address space's mmap lock.
    let fresh: Vec<(usize, Result<Lookup, fletch::RegistryError>)> = {
        use rayon::prelude::*;
        misses
            .par_iter()
            .map(|&i| {
                let (record, sources) =
                    fletch::registry::lookup_recorded(&selected[i].locator, &*res.net, &res.cache);
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
        let Lookup { record, sources } = match lookup {
            Ok(lookup) => lookup,
            // A registry that could not be reached may answer next time, so
            // the miss is not remembered; every other answer is final.
            Err(fletch::RegistryError::Unavailable(error)) => {
                tracing::warn!(locator = %key, %error, "registry unavailable; not caching the miss");
                continue;
            }
            Err(error) => {
                tracing::debug!(locator = %key, %error, "no registry record");
                writes.push((key, None));
                continue;
            }
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
        for (key, record) in writes {
            memo.put(key, Memoized { at: now, record });
        }
    }
    found
}

/// Serialize a registry record to its `*.registry.json` document and analyze it
/// with cleave, so filefacts parses the `registry.*` facts and the trait engine
/// runs over them. The document is named by the package so detection routes it
/// to `FileType::Registry`. `None` if it can't be serialized or analyzed.
pub(super) fn registry_node(reg: &Registry, opts: &AnalysisOptions) -> Option<AnalysisReport> {
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
pub(super) fn registry_doc_name(reg: &Registry) -> String {
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

/// A manifest peer range is a candidate, not evidence of an installed version.
pub(super) fn mark_peer_candidate(record: &mut FetchRecord, source: &str) {
    let source = if source == "package.json:peerDependencies" {
        CandidateSource::PeerDependency
    } else if source.starts_with("arch-pkginfo:") {
        CandidateSource::ArchRepository
    } else {
        return;
    };
    record.coverage.push(Coverage::Candidate { source });
}
