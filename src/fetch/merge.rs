//! Analyzing what was fetched and grafting it into the root's report.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// Every finding a finalized sub-report carries, flattened across its file
/// nodes — the seed half the package pass contributes from one side (artifact
/// or registry metadata).
pub(super) fn sub_findings(sub: &AnalysisReport) -> Vec<Finding> {
    sub.files
        .iter()
        .flat_map(|f| f.findings.iter().cloned())
        .collect()
}

/// The product of analyzing one fetched payload: the finalized sub-report to
/// graft (absent if the payload couldn't be analyzed) and the next-hop
/// references found in its own bytes. Produced off the report so the expensive
/// analysis can run concurrently; [`merge_payload`] folds it in serially.
pub(super) struct Analyzed {
    pub(super) content_sha: String,
    pub(super) sub: Option<AnalysisReport>,
    pub(super) next_from_bytes: Vec<Group>,
    /// The corpus verdict adopted in place of analyzing these bytes — set only
    /// when hopper's stored verdict came from the analyzer this build is
    /// running (see [`Standing`]).
    pub(super) corpus: Option<Verdict>,
}

/// Everything analyzing a fetched payload reads: the blob cache its bytes sit
/// in, the analysis options, the warm analysis cache, and hopper's corpus.
pub(super) struct Payloads<'a> {
    pub(super) cache: &'a BlobCache,
    pub(super) opts: &'a AnalysisOptions,
    pub(super) acache: Option<&'a AnalysisCache>,
    pub(super) precheck: Option<&'a Precheck>,
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
    pub(super) fn analyze_all(
        &self,
        landed: &[Landed],
        fanout: bool,
        on_analyzed: &(dyn Fn(usize) + Sync),
    ) -> Vec<Option<Analyzed>> {
        let one = |(i, l): (usize, &Landed)| {
            let analyzed = self.analyze(l);
            PAYLOADS_ANALYZED_TOTAL.fetch_add(1, Ordering::Relaxed);
            on_analyzed(i);
            analyzed
        };
        if fanout {
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
    pub(super) fn analyze(&self, landed: &Landed) -> Option<Analyzed> {
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
        let name = payload_name(rec);
        let cache_key = crate::analysis_cache::entry_key(&content_sha, &name);

        // Warm-cache hit: reuse the prior analysis of these exact bytes under
        // this name, skipping the re-analysis (a minutes-long disassembly for a
        // big native binary). Keyed by content sha and name under a
        // ruleset-version namespace, so an entry is only ever one the current
        // detector produced for this input — a rules/engine change misses and
        // re-scans.
        if let Some(hit) = self.acache.and_then(|ac| ac.get(&cache_key)) {
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
        // fleet-wide case it cannot — another host analyzed the same
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
            ac.put(&cache_key, &sub, &next_from_bytes);
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
pub(super) fn corpus_hit_record(r: &Reference, source_sha: &str, sha: &str) -> FetchRecord {
    FetchRecord {
        source_sha256: (!source_sha.is_empty()).then(|| source_sha.to_owned()),
        context: r.context.clone(),
        coverage: Vec::new(),
        source_offset: r.offset,
        kind: r.kind,
        locator: locator(r).to_owned(),
        resolved_url: None,
        final_url: None,
        redirects: Vec::new(),
        status: None,
        headers: Vec::new(),
        fetched_at: Some(crate::civil::now_secs()),
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
pub(super) fn capture_dependency(
    rec: &FetchRecord,
    analyzed: &Analyzed,
) -> Option<FetchedDependency> {
    let sub = analyzed.sub.as_ref()?;
    let compact =
        cleave::types::compact::compact_from_files(&sub.files, cleave::types::Facts::Full);
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
pub(super) struct Graft {
    pub(super) next_id: u32,
    pub(super) first: HashMap<String, (u32, u32)>,
}

impl Graft {
    pub(super) fn new(report: &AnalysisReport) -> Self {
        let mut graft = Self::default();
        for file in &report.files {
            graft.note(file);
        }
        graft
    }

    pub(super) fn note(&mut self, file: &cleave::types::FileAnalysis) {
        self.next_id = self.next_id.max(file.id.saturating_add(1));
        self.first
            .entry(file.sha256.clone())
            .or_insert((file.id, file.depth));
    }

    /// The `(id, depth)` of the file `sha` names, falling back to the root.
    pub(super) fn parent(&self, sha: Option<&str>) -> (u32, u32) {
        sha.and_then(|sha| self.first.get(sha))
            .copied()
            .unwrap_or((0, 0))
    }

    /// Append one node, keeping the index current.
    pub(super) fn push(&mut self, report: &mut AnalysisReport, file: cleave::types::FileAnalysis) {
        self.note(&file);
        report.files.push(file);
    }
}

/// Graft a materialized registry sub-report under the file that declared the
/// dependency (its sha256), mirroring [`merge_payload`]'s id/depth re-basing.
/// The node carries only facts — a registry document references nothing to
/// fetch — so no next-hop work-list is produced. Returns the registry node's id.
pub(super) fn merge_registry(
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
pub(super) fn append_adopted_node(
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

/// Published npm packages' ordinary lockfiles describe the publisher's own
/// development install. npm uses npm-shrinkwrap.json in consumed packages;
/// retain it. Directly scanned workspaces retain their lockfiles as before.
pub(super) fn is_published_npm_development_lock(locator: &str, path: &str) -> bool {
    if !locator.starts_with("pkg:npm/") || !path.contains("!!") {
        return false;
    }
    matches!(
        path.rsplit('/').next(),
        Some("package-lock.json" | "yarn.lock" | "pnpm-lock.yaml")
    )
}

pub(super) fn mark_published_npm_development_refs(
    locator: &str,
    path: &str,
    refs: &mut [Reference],
) {
    if is_published_npm_development_lock(locator, path) {
        mark_scope(refs, filefacts::DependencyScope::Development);
    }
}

/// Fold an [`Analyzed`] payload into the report: append its file nodes nested
/// under the file that declared the reference, and return the references the
/// payload yields for the next hop. The fetch edge (`source_sha256 →
/// content_sha256`) is the authoritative link; ids and depth are renumbered so
/// the grafted nodes are a well-formed subtree that never collides with the main
/// report's. Must run serially — it reads and extends `report.files`.
pub(super) fn merge_payload(
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

    let local = LocalModules::of(&sub);
    let first_new = report.files.len();
    for mut file in sub.files {
        // Preserve the payload's original paths for local package resolution.
        if !is_vendored_node_module(&file.path)
            && let Some(view) = &file.filefacts
        {
            let refs: Vec<_> = static_es_module_refs(&file.path, &file.file_type, &view.values)
                .into_iter()
                .chain(find::python_imports(&file.file_type, &view.symbols))
                .chain(
                    find::import_calls(&file.file_type, &view.symbols)
                        .into_iter()
                        .filter(is_python_import_candidate),
                )
                .filter(|r| {
                    !local.provides_python(&file, r)
                        && npm_import_name(r)
                            .is_none_or(|name| !local.provides_npm(&file.path, &name))
                })
                .collect();
            if !refs.is_empty() {
                next.push((file.sha256.clone(), refs));
            }
        }
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
            refs.extend(find::fetch_calls(&file.file_type, &view.symbols));
            refs.extend(find::discover_batch_programs(&file.file_type, &view.values).references);
            mark_published_npm_development_refs(&rec.locator, &file.path, &mut refs);
            if is_ci_context(file) {
                mark_scope(&mut refs, filefacts::DependencyScope::Ci);
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
pub(super) fn apply_package_composites(
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
        // artifact drowns a daemon's log. The grafted composites themselves

        // show up in the report, which is where they matter.
        Ok(n) => tracing::debug!(
            grafted = n,
            "package-scoped composites fired across artifact and registry metadata"
        ),
        Err(e) => tracing::warn!("package composite pass failed: {e:#}"),
    }
}

/// Return the registry findings paired with the original declared reference.
/// A fetch may canonicalize a versionless/ranged PURL to the concrete version
/// it downloaded, so joining on `FetchRecord::locator` loses exactly the
/// registry transitions (including security holders) this pass must correlate.
pub(super) fn registry_findings_for_reference<'a>(
    findings: &'a HashMap<String, Vec<Finding>>,
    reference: &Reference,
) -> Option<&'a [Finding]> {
    findings.get(locator(reference)).map(Vec::as_slice)
}

/// Enrich a fetched dependency's standalone report with package-scoped
/// registry composites (`graft`, [`apply_package_composites`] in production)
/// before it is captured or merged into its parent. Taking `graft` keeps the
/// ordering contract testable without network access or depending on the
/// machine's installed trait bundle.
pub(super) fn prepare_dependency_report_with(
    payload: &mut Analyzed,
    registry_findings: Option<&[Finding]>,
    opts: &AnalysisOptions,
    graft: impl FnOnce(&mut AnalysisReport, &str, &[Finding], &[Finding], &AnalysisOptions),
) {
    let (Some(registry), Some(sub)) = (registry_findings, payload.sub.as_mut()) else {
        return;
    };
    let artifact = sub_findings(sub);
    graft(sub, &payload.content_sha, &artifact, registry, opts);
}

/// The URL a fetch's bytes came from: where the redirects ended, else where
/// the locator resolved. `None` when it never reached one.
#[must_use]
pub fn fetched_url(rec: &FetchRecord) -> Option<&str> {
    rec.final_url.as_deref().or(rec.resolved_url.as_deref())
}

/// A filename for a fetched payload: the final URL's basename, else the
/// content hash. Drives cleave's extension-based type detection.
pub(super) fn payload_name(rec: &FetchRecord) -> String {
    fetched_url(rec)
        .and_then(|url| url.rsplit('/').next())
        .and_then(|s| s.split(['?', '#']).next())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| rec.content_sha256.clone())
        .unwrap_or_else(|| "fetched".to_string())
}
