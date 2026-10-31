//! One fetch session: the hop-by-hop loop that selects references, fetches
//! them, analyzes what lands, and grafts it into the root's report.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

fn is_registry_security_hold_finding(id: &str) -> bool {
    id.strip_prefix("metadata/registry::")
        .is_some_and(|name| name == TRAIT_SECURITY_HOLD_RECORD || name == "registry-security-hold")
}

/// One group's references after selection and the age gate: the ones to
/// fetch, and the registry records materialized for its dependencies.
pub(super) struct GroupPlan {
    pub(super) source_sha: String,
    pub(super) selected: Vec<Reference>,
    pub(super) registries: Vec<Gated>,
}

/// One group after the network phase.
pub(super) struct GroupFetch {
    pub(super) source_sha: String,
    pub(super) registries: Vec<Gated>,
    pub(super) landed: Vec<Landed>,
}

/// One group's analyses, aligned with its [`GroupFetch`]: a sub-report per
/// registry record and an [`Analyzed`] per landed reference, `None` where
/// there was nothing to graft.
pub(super) struct GroupAnalysis {
    pub(super) registries: Vec<Option<AnalysisReport>>,
    pub(super) payloads: Vec<Option<Analyzed>>,
}

/// One selected reference after the network phase: the edge recorded for it,
/// and what the corpus already lets us skip — [`Standing::Analyze`] for
/// anything that went to the network.
pub(super) struct Landed {
    pub(super) reference: Reference,
    pub(super) record: FetchRecord,
    pub(super) standing: Standing,
}

/// Analysis gaps are per observed call, not guessed URL references. The same
/// retained facts are available for roots and extracted/fetched members.
pub(super) fn unresolved_downloader_records(
    report: &AnalysisReport,
    policy: &FetchPolicy,
) -> Vec<FetchRecord> {
    if !policy.wants(RefKind::UrlFetch) {
        return Vec::new();
    }
    let mut records = Vec::new();
    for file in &report.files {
        if !policy.ci && is_ci_context(file) {
            continue;
        }
        let Some(view) = &file.filefacts else {
            continue;
        };
        let file_type = report_file_type(&file.file_type);
        for call in
            find::references_from_facts(file_type, &view.values, &view.symbols, &[]).unresolved
        {
            let location = call
                .offset
                .map_or_else(String::new, |offset| format!(" at byte {offset}"));
            let mut record = FetchRecord::terminal(
                format!(
                    "{} (unresolved downloader arguments){location} in {}",
                    call.command, file.sha256
                ),
                Outcome::Unresolved(fletch::fetch::Unresolved::Unsupported),
            );
            record.source_sha256 = Some(file.sha256.clone());
            record.source_offset = call.offset;
            record.kind = RefKind::UrlFetch;
            record.coverage.push(Coverage::Unavailable {
                reason: Some(
                    "downloader arguments require runtime evaluation; no URL inferred".into(),
                ),
            });
            records.push(record);
        }
    }
    records
}

/// POM declarations that need an effective Maven model are coverage gaps,
/// never silently substituted with a latest version or Maven Central artifact.
pub(super) fn unresolved_maven_records(
    report: &AnalysisReport,
    policy: &FetchPolicy,
) -> Vec<FetchRecord> {
    if !policy.wants(RefKind::Dependency) {
        return Vec::new();
    }
    let mut records = Vec::new();
    for file in &report.files {
        let Some(view) = &file.filefacts else {
            continue;
        };
        let Some(gaps) = view
            .values
            .pointer("/jar/pom/dependency_gaps")
            .and_then(serde_json::Value::as_array)
        else {
            continue;
        };
        for gap in gaps {
            let (Some(member), Some(reason)) = (
                gap.get("member").and_then(serde_json::Value::as_str),
                gap.get("reason").and_then(serde_json::Value::as_str),
            ) else {
                continue;
            };
            let mut record = FetchRecord::terminal(
                format!("Maven POM {member} in {}: {reason}", file.sha256),
                Outcome::Unresolved(fletch::fetch::Unresolved::Unsupported),
            );
            record.source_sha256 = Some(file.sha256.clone());
            record.kind = RefKind::Dependency;
            record.coverage.push(Coverage::Unavailable {
                reason: Some(format!("Maven POM dependency: {reason}")),
            });
            records.push(record);
        }
    }
    records
}

/// One fetch phase: everything `orchestrate` carries from hop to hop.
pub(super) struct FetchSession {
    pub(super) policy: FetchPolicy,
    pub(super) res: &'static Resources,
    /// hopper's corpus, when this process has one to ask.
    pub(super) precheck: Option<Arc<Precheck>>,
    /// How fetched payloads are analyzed: with the same bloom short-circuit
    /// the top-level scan uses, so a trusted binary shipped inside a dependency
    /// isn't needlessly re-disassembled.
    pub(super) opts: AnalysisOptions,
    /// A registry record is canonical JSON we serialized ourselves; its signal
    /// is entirely `registry.*` value facts and no YARA rule targets it.
    /// Disabling YARA here removed ~1400s of system time per scan — the
    /// engine's per-analysis setup, paid hundreds of times. Built once:
    /// cloning `AnalysisOptions` per record cost more user time than the YARA
    /// saving returned.
    pub(super) registry_opts: AnalysisOptions,
    /// Opened on the first payload actually analyzed, so a scan that analyzes
    /// nothing (every reference age-gated or none present, the common `pkg:`
    /// case) never creates or prunes the cache directory.
    pub(super) acache: OnceLock<Option<AnalysisCache>>,
    /// Where the fetch phase's progress goes: the live in-place dependency tree
    /// on an interactive single-artifact scan, the streamed log above any active
    /// scan bar otherwise, or nothing for machine output. Shared across every
    /// hop, so transitive dependencies join the same view.
    pub(super) reporter: Reporter,
    pub(super) capture_deps: bool,
    /// One wall-clock reading for the whole run, so every dependency's age is
    /// judged against the same instant.
    pub(super) now: u64,
    /// The host platform, for filtering off-host native-binary dependencies.
    pub(super) host: (&'static str, &'static str),
    pub(super) deadline: Deadline,
    pub(super) budget: RootBudget,
    /// Loop guard: a locator is fetched at most once per run, so a chain that
    /// points back at an earlier stage can't cycle.
    pub(super) seen: HashSet<String>,
    /// Newest-version gate: only the most recent version of a package in the
    /// dependency tree is fetched and analyzed. Deep ungated trees pin dozens
    /// of releases of the same packages (syn ×27, libc ×21 on the mx crate
    /// benchmark — 48% of its tree was older duplicates); analyzing every
    /// pinned release repeats near-identical work without detection value the
    /// newest release doesn't provide. Versionless references are exempt (they
    /// already resolve to the latest release), and the kept version is monotone
    /// across hops: once a release is scanned, an older sibling discovered in a
    /// later hop never resurrects the package.
    pub(super) newest: HashMap<String, String>,
    /// Coordinates (`pkg:eco/name`) for which a version-pinned reference exists
    /// anywhere in the tree. A manifest range (`"puppeteer": "^10.4.0"`) reaches
    /// us version-stripped as a bare `pkg:npm/puppeteer` and would resolve to
    /// `dist-tags/latest` — a version the project never installs. When the
    /// co-located lockfile also pins the coordinate (`pkg:npm/puppeteer@10.4.2`),
    /// that pin is ground truth and must win, so the bare sibling is dropped.
    /// Monotone across hops, mirroring `newest`.
    pub(super) pinned: HashSet<String>,
    /// A redirect page is not a stage. A shortener's safety or preview page
    /// stands where an HTTP 3xx would, naming the payload instead of being it,
    /// so a payload reached through one credits its group a hop and the chain
    /// behind the redirect keeps its full `policy.depth`. Keyed by the payload
    /// content sha that `merge_payload` groups next-hop references under.
    pub(super) redirect_credit: HashMap<String, u8>,
    /// Image URLs whose declaring file shows a stego loader: exempt from the
    /// page-asset extension filter. Hop 0 only — later hops hunt payload bytes
    /// before any analysis, so there are no findings to vouch for one.
    pub(super) carriers: HashSet<String>,
    /// source content-sha → the declaring manifest's path (relative to the root
    /// artifact), so each dependency row can name the file it came from. A
    /// source discovered only inside a fetched payload (a deeper hop) simply
    /// isn't found and stays unnamed.
    pub(super) manifests: HashMap<String, String>,
    pub(super) graft: Graft,
    pub(super) out: FetchOutcome,
    /// (declaring file sha) -> (registry records materialized, of which security-held)
    pub(super) registry_outcomes: BTreeMap<String, (u64, u64)>,
    pub(super) root_sha: String,
    pub(super) current_hop: u8,
    pub(super) pending_error: Option<String>,
    pub(super) pending: Option<Arc<pending::Store>>,
    /// Locally supplied copies: source sha → locator → path.
    pub(super) local_coverage: HashMap<String, HashMap<String, String>>,
    pub(super) unresolved_fetch_calls: HashSet<(String, Option<u64>, String)>,
}

impl FetchSession {
    pub(super) fn new(
        report: &AnalysisReport,
        worklist: &[Group],
        res: &'static Resources,
        follow: Follow<'_>,
    ) -> Self {
        let Follow {
            policy,
            progress,
            capture_deps,
            zip_passwords,
        } = follow;
        let mut opts = AnalysisOptions {
            compact_members: true, // compact projection only
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
            precheck: crate::corpus_precheck::armed(),
            registry_opts: AnalysisOptions {
                yara: false,
                ..opts.clone()
            },
            opts,
            acache: OnceLock::new(),
            reporter: Reporter::new(progress),
            capture_deps,
            now: crate::civil::now_secs(),
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
            root_sha: report
                .files
                .first()
                .map(|f| f.sha256.clone())
                .unwrap_or_default(),
            current_hop: 0,
            pending_error: None,
            pending: crate::settings::fetch_backlog().cloned(),
            local_coverage: HashMap::new(),
            unresolved_fetch_calls: HashSet::new(),
        }
    }

    /// Keep unresolved downloader calls and Maven declarations visible without
    /// inventing URLs, artifact coordinates, or effective-model defaults.
    pub(super) fn record_unresolved_fetch_calls(&mut self, report: &AnalysisReport) {
        for record in unresolved_downloader_records(report, &self.policy)
            .into_iter()
            .chain(unresolved_maven_records(report, &self.policy))
        {
            let key = (
                record.source_sha256.clone().unwrap_or_default(),
                record.source_offset,
                record.locator.clone(),
            );
            if self.unresolved_fetch_calls.insert(key) {
                self.reporter.report(&record, "");
                if let Reporter::Tree { tree, .. } = &self.reporter {
                    tree.add(&record.locator, &record.locator, "");
                    tree.set(&record.locator, landed_state(&record));
                }
                self.out.records.push(record);
            }
        }
    }

    /// Walk the hops: each works from the references found inside the
    /// previous hop's payloads.
    pub(super) fn run(&mut self, report: &mut AnalysisReport, mut worklist: Vec<Group>) {
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
            // Reconcile imports before policy selection. Declared imports
            // must not evade a transitive-dependency limit by resolving latest
            // as a command. References-only scans retain the declaration's
            // constraint while following the observed import.
            let declared_imports = declared_npm_imports_in_hop(report, &worklist);
            let mut covered = Vec::new();
            for (source, reference) in
                reconcile_python_imports(report, &mut worklist, self.policy.deps)
            {
                self.out.records.push(skipped_record(
                    &source,
                    &reference,
                    Coverage::CoveredByDeclaration,
                ));
                covered.push(self.pending_entry(
                    &source,
                    &reference,
                    hop,
                    "declared Python requirement",
                ));
            }
            for (source, refs) in &mut worklist {
                refs.retain_mut(|reference| {
                    if !self.policy.wants_at(RefKind::Command, hop)
                        || reference.kind != RefKind::Command
                        || !matches!(reference.source.as_str(), "ast-call" | "ast-import")
                    {
                        return true;
                    }
                    let Some(name) = npm_import_name(reference) else {
                        return true;
                    };
                    let Some(declaration) = declared_imports
                        .get(source)
                        .and_then(|names| names.get(&name))
                    else {
                        return true;
                    };
                    if !self.policy.deps {
                        reference.locator = declaration.locator.clone();
                        reference.pinned_hash = declaration.pinned_hash.clone();
                        reference.context = declaration.context.clone();
                        return true;
                    }
                    let coverage = if self.policy.wants_at(RefKind::Dependency, hop) {
                        Coverage::CoveredByDeclaration
                    } else {
                        Coverage::TransitiveNotFollowed
                    };
                    self.out
                        .records
                        .push(skipped_record(source, reference, coverage));
                    covered.push(self.pending_entry(source, reference, hop, "manifest dependency"));
                    false
                });
            }
            let local = LocalNpmPackages::from_report(report);
            for (source, refs) in &mut worklist {
                refs.retain(|reference| {
                    if !self.policy.wants_at(reference.kind, hop) {
                        return true;
                    }
                    let Some(path) = local.declared_coverage(report, source, reference) else {
                        return true;
                    };
                    if reference.pinned_hash.is_some() {
                        self.local_coverage
                            .entry(source.clone())
                            .or_default()
                            .insert(locator(reference).to_owned(), path);
                        return true;
                    }
                    self.out.records.push(skipped_record(
                        source,
                        reference,
                        Coverage::Supplied {
                            path,
                            pin_verified: None,
                        },
                    ));
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
        }
        self.defer_groups(&worklist, end, "depth limit");
        for (hop, groups) in resumed {
            self.defer_groups(&groups, hop, "depth limit or fetch timeout");
        }
    }

    pub(super) fn pending_entry(
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
    pub(super) fn checkpoint(
        &mut self,
        additions: &[pending::Entry],
        completed: &[pending::Entry],
    ) {
        if let Some(store) = self.pending.as_ref()
            && let Err(error) = store.update(additions, completed)
        {
            tracing::error!(%error, "fetch backlog checkpoint failed");
            self.pending_error = Some(error.to_string());
        }
    }
    pub(super) fn defer_groups(&mut self, groups: &[Group], hop: u8, reason: &str) {
        let entries: Vec<_> = groups
            .iter()
            .flat_map(|(sha, refs)| {
                refs.iter()
                    .filter(|r| self.policy.wants(r.kind))
                    .map(|r| self.pending_entry(sha, r, hop, reason))
            })
            .collect();
        // Every skip is reported, but only what `select` would take at `hop`
        // is checkpointed: anything else is never fetched, so it would never
        // be completed and would sit in the backlog forever.
        let resumable: Vec<_> = entries
            .iter()
            .filter(|e| self.wanted(&e.reference, hop) && !self.dev_excluded(&e.reference))
            .cloned()
            .collect();
        self.checkpoint(&resumable, &[]);
        for entry in entries {
            self.out.records.push(skipped_record(
                &entry.source_sha,
                &entry.reference,
                Coverage::Pending {
                    reason: reason.to_owned(),
                    hop,
                },
            ));
        }
    }

    /// Pre-scan a whole hop so the newest version is hop-wide, not
    /// first-group-wins: every fetchable versioned reference bids, and the
    /// running cross-hop maximum only rises.
    pub(super) fn bid_versions(&mut self, worklist: &[Group], hop: u8) {
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
    pub(super) fn hop(
        &mut self,
        report: &mut AnalysisReport,
        worklist: Vec<Group>,
        hop: u8,
    ) -> Vec<Group> {
        let target = BATCH_PAYLOAD_TARGET;
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
            let mut batch: Vec<GroupFetch> = plans
                .into_iter()
                .map(|plan| self.fetch_group(plan, &corpus))
                .collect();
            let analyzed = self.analyze_batch(&mut batch);
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
    pub(super) fn select(
        &mut self,
        source_sha: &str,
        refs: Vec<Reference>,
        hop: u8,
        dropped_old: &mut usize,
    ) -> Vec<Reference> {
        let mut selected = Vec::new();
        for r in refs {
            if self.policy.wants(r.kind) && !self.policy.wants_at(r.kind, hop) {
                self.out.records.push(skipped_record(
                    source_sha,
                    &r,
                    Coverage::TransitiveNotFollowed,
                ));
                continue;
            }
            if self.dev_excluded(&r) {
                self.out.records.push(skipped_record(
                    source_sha,
                    &r,
                    Coverage::DevelopmentExcluded,
                ));
                continue;
            }
            if !self.wanted(&r, hop) {
                continue;
            }
            if let Some(reason) = self.fetch_skip_reason(&r) {
                // A downloader the code calls is evidence, so its exclusion is
                // recorded; any other unfetchable reference only logs.
                if r.source == "ast-fetch-call" {
                    self.out.records.push(skipped_record(
                        source_sha,
                        &r,
                        Coverage::Unavailable {
                            reason: Some(format!("downloader URL excluded by policy ({reason})")),
                        },
                    ));
                } else {
                    tracing::debug!(locator = %locator(&r), source = %r.source, "{reason}; fetch skipped");
                }
                continue;
            }
            if !self.off_host(&r)
                && self.newest_version(&r, dropped_old)
                && !self.superseded(&r)
                && self.seen.insert(fetch_work_key(&r))
            {
                selected.push(r);
            }
        }
        selected
    }

    /// Whether `r` is a development dependency this policy leaves out.
    fn dev_excluded(&self, r: &Reference) -> bool {
        !self.policy.include_dev_dependencies
            && r.context
                .as_ref()
                .is_some_and(|c| c.scope == filefacts::DependencyScope::Development)
    }

    /// Whether this hop follows `r`, CI references only under `--ci`. Called
    /// from [`Self::select`] after it has recorded and dropped transitive and
    /// development references.
    fn wanted(&self, r: &Reference, hop: u8) -> bool {
        (self.policy.ci
            || !r
                .context
                .as_ref()
                .is_some_and(|c| c.scope == filefacts::DependencyScope::Ci))
            && self.policy.wants_at(r.kind, hop)
    }

    /// Whether a reference can be fetched at all, and — for a discovered URL —
    /// whether it is worth a round trip. Publisher-controlled URLs, obvious
    /// site/API endpoints, and the exact documentation/update URLs observed in
    /// stock /bin binaries cost a round trip each and are unlikely to yield a
    /// dropper payload. Applied per hop so a payload's own boilerplate is
    /// filtered too, and only to *discovered* references: `scan url <url>`
    /// fetches whatever the operator names (see `crate::hosts`).
    pub(super) fn fetchable(&self, r: &Reference) -> bool {
        if let Some(reason) = self.fetch_skip_reason(r) {
            tracing::debug!(locator = %locator(r), source = %r.source, "{reason}; fetch skipped");
            return false;
        }
        true
    }

    pub(super) fn fetch_skip_reason(&self, r: &Reference) -> Option<&'static str> {
        let raw = match &r.locator {
            RefLocator::Purl(_) => return None,
            // Resolved against the artifact's own files, never fetched.
            RefLocator::Path(_) => return Some("intra-artifact path"),
            RefLocator::Url(raw) => raw,
            // A locator kind this scan predates names nothing it can fetch.
            _ => return Some("unsupported locator"),
        };
        let Ok(url) = Url::parse(raw) else {
            return Some("invalid or local URL host");
        };
        if !valid_discovered_url_host(&url) {
            return Some("invalid or local URL host");
        }
        if has_unexpanded_url_placeholder(raw) {
            return Some("unexpanded URL template");
        }
        if hosts::publisher_controlled(&url) || hosts::discovery_exception(raw, &url) {
            return Some("publisher-controlled or boilerplate URL");
        }
        if r.kind == RefKind::UrlFetch
            // An AST-observed downloader call is already a fetch operation;
            // its endpoint need not have a script/archive filename suffix.
            && r.source != "ast-fetch-call"
            && !looks_like_dropper_download_url(&url)
            && !is_eval_pipeline_url(r)
            && !is_download_to_file_url(r)
            && !is_redirect_destination(r)
            && !self.carriers.contains(raw)
        {
            return Some("URL does not look like a dropper download or eval pipeline");
        }
        None
    }

    /// Native-binary dependencies built for another platform are dropped before
    /// they're ever fetched — the host variant is scanned, its linux/windows
    /// siblings never run here. Off unless the policy asks for it
    /// (`serve` audits every platform).
    pub(super) fn off_host(&self, r: &Reference) -> bool {
        let off_host = self.policy.host_platform_only && off_host_platform(r, self.host);
        if off_host {
            tracing::debug!(
                package = %locator(r),
                host_os = self.host.0,
                host_arch = self.host.1,
                "dependency pinned to another platform; skipped"
            );
        }
        off_host
    }

    /// Older-version duplicates are skipped entirely — no registry-record
    /// materialization, no fetch (operator policy). Never silent:
    /// each skip logs at debug, the hop logs one count at info.
    pub(super) fn newest_version(&self, r: &Reference, dropped_old: &mut usize) -> bool {
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
    pub(super) fn superseded(&self, r: &Reference) -> bool {
        // Python imports use their own nearest declaration, not a pin in an
        // unrelated package that merely shares the distribution spelling.
        if is_python_import_candidate(r) {
            return false;
        }
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
    pub(super) fn plan(&self, source_sha: String, selected: Vec<Reference>) -> GroupPlan {
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
    pub(super) fn precheck_purls(
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
                    && pins_exact_version(locator(r))
            })
            .map(|r| locator(r).to_owned())
            .collect();
        // A coordinate whose bytes can change is asked about by the digest its
        // registry lists today, never by name; see `precheck_resolved`. A
        // release's age, where the age gate looked it up, can settle that.
        let ages: HashMap<&str, u64> = plans
            .iter()
            .flat_map(|plan| &plan.registries)
            .filter_map(|g| Some((locator(&g.reference), g.record.age_secs(self.now)?)))
            .collect();
        let (candidates, mutable): (Vec<String>, Vec<String>) = candidates
            .into_iter()
            .partition(|purl| immutable_coordinate(purl, ages.get(purl.as_str()).copied()));
        // Hopper is ours; isotope13 is told only public-registry coordinates.
        // The rest are fetched and asked about by digest.
        let candidates = if precheck.is_isotope13() {
            let (public, withheld): (Vec<String>, Vec<String>) = candidates
                .into_iter()
                .partition(|purl| crate::remote::public_purl(purl));
            if !withheld.is_empty() {
                tracing::debug!(
                    event = "remote.withheld",
                    kind = "purl",
                    reason = "not_public",
                    count = withheld.len()
                );
            }
            public
        } else {
            candidates
        };
        let mut hits = if candidates.is_empty() {
            HashMap::new()
        } else {
            precheck.purls(&candidates)
        };
        let by_digest = self.precheck_resolved(precheck, &mutable);
        let resolved = by_digest.len();
        hits.extend(by_digest);
        if !hits.is_empty() {
            tracing::info!(
                skipped = hits.len(),
                asked = candidates.len() + mutable.len(),
                by_digest = resolved,
                "purl precheck: hopper verdicts stand; skipped fetch+analysis+upload"
            );
        }
        hits
    }

    /// The standing verdicts for mutable coordinates, asked about by the sha256
    /// their registry lists for the exact file a fetch would download — the
    /// one identity of theirs that cannot be repointed. Resolving reads the
    /// registry document the age gate's lookup usually just cached, so it seldom
    /// costs a request, and one it does cost a fetch would have paid anyway. A
    /// coordinate with no listed digest (a git tag, a container tag) is fetched.
    fn precheck_resolved(
        &self,
        precheck: &crate::corpus_precheck::Precheck,
        purls: &[String],
    ) -> HashMap<String, crate::corpus_precheck::PurlHit> {
        use rayon::prelude::*;
        if purls.is_empty() {
            return HashMap::new();
        }
        let resolved: Vec<(&String, String)> = purls
            .par_iter()
            .filter_map(|purl| {
                let sha = registry_digest(purl, &*self.res.net, &self.res.cache);
                tracing::debug!(
                    package = %purl,
                    sha256 = sha.as_deref().unwrap_or("none listed; fetching"),
                    "purl precheck: mutable coordinate resolved to its registry digest"
                );
                Some((purl, sha?))
            })
            .collect();
        let shas: Vec<String> = resolved.iter().map(|(_, sha)| sha.clone()).collect();
        let standings = precheck.shas(&shas);
        resolved
            .into_iter()
            .filter_map(|(purl, sha)| {
                let standing = standings.get(&sha)?.clone();
                standing.skips_analysis().then(|| {
                    (
                        purl.clone(),
                        crate::corpus_precheck::PurlHit { sha, standing },
                    )
                })
            })
            .collect()
    }

    /// Fetch one group's selected references: dependencies and
    /// command-mentioned packages under one cap, opportunistic URLs under their
    /// own smaller one, and none at all for a dependency the corpus answered.
    pub(super) fn fetch_group(
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
                .get(&source_sha)
                .and_then(|by_locator| by_locator.get(locator(&l.reference)))
            {
                l.record.coverage.push(Coverage::Supplied {
                    path: path.clone(),
                    pin_verified: pin_verified(&l.reference, &l.record),
                });
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
    pub(super) fn fetch_class(
        &mut self,
        class: FetchClass,
        refs: &[Reference],
        source_sha: &str,
    ) -> (Vec<FetchRecord>, String) {
        if refs.is_empty() {
            return (Vec::new(), String::new());
        }
        let (kind, limit) = match class {
            FetchClass::Deps => ("dependency", self.policy.max_file_fetches),
            FetchClass::Urls => ("URL", self.policy.max_url_fetches),
        };
        let want = self.budget.want(class);
        let grant = TOTAL_BUDGET.reserve(want);
        let total_limited = if grant.fetches < want.fetches {
            grant.fetches
        } else {
            usize::MAX
        };
        let notice = fetch_count_budget_notice(kind, limit, total_limited);
        // The callback fires as each download lands (from a pool worker, so
        // it's `Sync`), flipping that row to "analyzing" the moment its bytes
        // arrive rather than when the whole concurrent batch returns.
        let reporter = &self.reporter;
        let on_fetched = |r: &Reference, rec: &FetchRecord| reporter.landed(r, rec);
        let budget = FetchBudget {
            max_count: grant.fetches,
            max_bytes: grant.bytes,
            ..FetchBudget::default()
        };
        let mut records = match class {
            FetchClass::Deps => fetch_references(
                refs,
                Some(source_sha),
                UrlFetches::Skip,
                &*self.res.net,
                &self.res.cache,
                budget,
                &on_fetched,
            ),
            FetchClass::Urls => {
                let net = AsClient {
                    net: &*self.res.net,
                    agents: refs
                        .iter()
                        .filter_map(|r| Some((locator(r).to_owned(), client_user_agent(r)?)))
                        .collect(),
                };
                fetch_references(
                    refs,
                    Some(source_sha),
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
                record.coverage.push(Coverage::BudgetExceeded {
                    notice: notice.to_string(),
                    fetches: u64::try_from(grant.fetches).unwrap_or(u64::MAX),
                    bytes: grant.bytes,
                });
            }
        }
        let spent = live_fetch_usage(&records);
        TOTAL_BUDGET.settle(grant, spent);
        self.budget.spend(class, spent);
        (records, notice)
    }

    /// Analyze a batch. Registry records and fetched payloads are both
    /// report-independent, so they fan out together across every group in the
    /// batch — when the pool has headroom; saturated, the batch runs inline on
    /// this blocking thread. A payload analysis dispatched from `par_iter` on a
    /// saturated pool occupies a rayon worker that cleave's nesting throttle
    /// assumes is free (see [`cleave::pool_has_headroom`]); inline, it runs
    /// where the throttle expects. Each `registry_node` is an independent cleave
    /// analysis of one small JSON document, far cheaper than a payload's full
    /// cleave pass. Results align with `batch`.
    pub(super) fn analyze_batch(&self, batch: &mut [GroupFetch]) -> Vec<GroupAnalysis> {
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
        let fanout = cleave::pool_has_headroom();
        let registries_of = |registries: &mut [Gated]| -> Vec<Option<AnalysisReport>> {
            registries
                .iter_mut()
                .map(|gated| {
                    let sub = registry_node(&gated.record, &self.registry_opts);
                    gated.finish_registry_analysis(sub.as_ref());
                    sub
                })
                .collect()
        };
        let payloads_of = |landed: &[Landed]| -> Vec<Option<Analyzed>> {
            // Settles each payload row from "analyzing" to its final glyph as
            // its scan finishes.
            let on_analyzed = |i: usize| {
                if let Some(l) = landed.get(i) {
                    self.reporter.analyzed(&l.reference, &l.record);
                }
            };
            payloads.analyze_all(landed, fanout, &on_analyzed)
        };
        // The two branches touch disjoint fields. Registry analysis can now
        // release provider buffers without waiting for the slowest payload;
        // all report grafting still happens later in the original group order.
        let (registries, landed): (Vec<_>, Vec<_>) = batch
            .iter_mut()
            .map(|group| (group.registries.as_mut_slice(), group.landed.as_slice()))
            .unzip();
        let (subs, payloads): (Vec<_>, Vec<_>) = if fanout {
            use rayon::prelude::*;
            rayon::join(
                || registries.into_par_iter().map(&registries_of).collect(),
                || landed.into_par_iter().map(&payloads_of).collect(),
            )
        } else {
            (
                registries.into_iter().map(&registries_of).collect(),
                landed.into_iter().map(&payloads_of).collect(),
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
    pub(super) fn merge_group(
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
        let mut unverified_registry_issues: HashMap<String, std::collections::BTreeSet<String>> =
            HashMap::new();
        let verified_python: HashSet<&str> = landed
            .iter()
            .zip(&payloads)
            .filter(|(landed, payload)| {
                is_python_import_candidate(&landed.reference)
                    && delivered_bytes(&landed.record)
                    && payload
                        .as_ref()
                        .and_then(|payload| payload.sub.as_ref())
                        .is_some_and(|sub| {
                            python_candidate_provides_namespace(&landed.reference, sub)
                        })
            })
            .map(|(landed, _)| landed.reference.evidence.as_str())
            .collect();
        for (gated, sub) in registries.into_iter().zip(subs) {
            // Registry claims against an unverified name must not convict a
            // Python importer whose distribution identity is still unknown.
            if is_python_import_candidate(&gated.reference)
                && !verified_python.contains(gated.reference.evidence.as_str())
            {
                if let Some(sub) = sub {
                    unverified_registry_issues
                        .entry(gated.reference.evidence)
                        .or_default()
                        .extend(candidate_static_issues(&sub));
                }
                continue;
            }
            self.merge_registry_record(report, &source_sha, gated, sub, &mut registry_findings);
        }
        // `next` spans the whole hop; only what this group adds is new here.
        let discovered_from = next.len();
        for (l, mut payload) in landed.iter_mut().zip(payloads) {
            if payload.is_none()
                && delivered_bytes(&l.record)
                && matches!(l.standing, Standing::Analyze)
            {
                l.record
                    .coverage
                    .push(Coverage::Unavailable { reason: None });
            }
            if payload
                .as_ref()
                .and_then(|p| p.sub.as_ref())
                .is_some_and(|sub| {
                    !sub.analysis_gaps.is_empty()
                        || sub.files.iter().any(|f| !f.analysis_gaps.is_empty())
                })
            {
                l.record.coverage.push(Coverage::Incomplete);
            }
            mark_peer_candidate(&mut l.record, &l.reference.source);
            if is_python_import_candidate(&l.reference) {
                if !annotate_python_import_candidate(
                    &mut l.record,
                    &l.reference,
                    payload.as_ref().and_then(|p| p.sub.as_ref()),
                    unverified_registry_issues.get(&l.reference.evidence),
                ) {
                    payload = None;
                }
                self.reporter.analyzed(&l.reference, &l.record);
                self.reporter.report(&l.record, "");
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
                || l.record.coverage.iter().any(is_analysis_gap)
            {
                pending.push(entry);
            } else {
                completed.push(entry);
            }
        }
        self.checkpoint(&pending, &completed);
        // Only what `select` would admit on the next hop: a reference it drops
        // is never fetched, so it would never be completed and would sit in
        // the backlog forever.
        let next_hop = self.current_hop.saturating_add(1);
        let discovered: Vec<_> = next[discovered_from..]
            .iter()
            .flat_map(|(sha, refs)| {
                refs.iter()
                    .filter(|r| {
                        self.wanted(r, next_hop) && !self.dev_excluded(r) && self.fetchable(r)
                    })
                    .map(|r| self.pending_entry(sha, r, next_hop, "discovered"))
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
    pub(super) fn merge_registry_record(
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
            provenance,
            skip,
        } = gated;
        if let Some(sub) = sub {
            let retain_provenance = keep_registry_provenance(skip, &sub);
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
                .any(|f| is_registry_security_hold_finding(&f.id))
            {
                tally.1 += 1;
            }
            // A skipped dependency has no artifact upload and only appears in
            // provenance output when its registry node is notable. Drop its raw
            // provider document when neither applies; lockfiles can contain
            // hundreds of ordinary aged-out records.
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
                let provenance = provenance.unwrap_or_else(|| {
                    let sources = sources.unwrap_or_else(|| {
                        fletch::registry::lookup_recorded(
                            &reference.locator,
                            &*self.res.net,
                            &self.res.cache,
                        )
                        .1
                    });
                    RegistryProvenance::from_record_sources(record.clone(), &sources)
                });
                self.out.registries.push(DependencyRegistry {
                    locator: locator(&reference).to_owned(),
                    provenance,
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
    pub(super) fn merge_landed(
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
        prepare_dependency_report_with(
            &mut payload,
            registry_findings_for_reference(registry_findings, &l.reference),
            &self.opts,
            apply_package_composites,
        );
        if let Some(sub) = &payload.sub {
            self.record_unresolved_fetch_calls(sub);
        }
        // Check each fetched npm package in its own manifest context before
        // it is captured or merged into its parent's report. Only with its
        // bytes in hand: the root's path is a download name, not a file, and
        // read from disk it would resolve against the working directory.
        if let Some(sub) = &mut payload.sub
            && let Some(root_path) = sub
                .files
                .first()
                .filter(|f| f.file_type == "npm")
                .map(|f| f.path.clone())
            && let Some(bytes) = self.res.cache.load(&l.record.locator)
        {
            let ci = if self.policy.ci {
                CiRefs::Include
            } else {
                CiRefs::Skip
            };
            let refs = collect_references_from_input(sub, Path::new(&root_path), Some(&bytes), ci);
            attribute_undeclared_npm_imports(sub, &refs);
        }
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

    /// Close out the phase: settle the progress view and record, on each file
    /// that declared references, what became of them.
    pub(super) fn finish(mut self, report: &mut AnalysisReport) -> FetchOutcome {
        if let Some(error) = self.pending_error {
            let mut record = FetchRecord::terminal(
                "fetch backlog".into(),
                Outcome::Failed(FetchError::Internal(error)),
            );
            record.coverage.push(Coverage::BacklogUnsaved);
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
pub(super) fn keyed(r: &Reference, index: usize) -> Reference {
    let mut keyed = r.clone();
    keyed.offset = Some(index as u64);
    keyed
}

/// Put each record fletch returned in the slot of the reference it was fetched
/// for — the one whose index [`keyed`] stamped as its offset — and restore the
/// reference's real offset on the record.
pub(super) fn pair_records(
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
/// therefore no node, so without this record the most interesting outcome of a
/// follow would be the one thing no trait could see.
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
/// file-scoped trait reads them — no composite scope required.
pub(super) fn attribute_reference_outcomes(
    report: &mut AnalysisReport,
    records: &[FetchRecord],
    registry_outcomes: &BTreeMap<String, (u64, u64)>,
) {
    #[derive(Default)]
    struct Tally<'a> {
        declared: u64,
        unresolved: Vec<&'a str>,
        unavailable_commands: u64,
        unavailable_command_spans: Vec<filefacts::Span>,
    }

    let mut touched: Vec<String> = Vec::new();
    let mut by_source: BTreeMap<&str, Tally<'_>> = BTreeMap::new();
    for rec in records {
        let Some(source) = rec.source_sha256.as_deref() else {
            continue;
        };
        let tally = by_source.entry(source).or_default();
        tally.declared += 1;
        // Only an actual command edge with an explicit absent-artifact HTTP
        // response counts. Unknown resolution, 403, timeouts and docs do not.
        if rec.kind == RefKind::Command
            && matches!(rec.outcome, Outcome::Failed(FetchError::Status(404 | 410)))
            && artifact_absent(rec)
        {
            tally.unavailable_commands += 1;
            if let Some(offset) = rec.source_offset {
                tally
                    .unavailable_command_spans
                    .push(filefacts::Span::new(offset, 1));
            }
        }
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
        metrics.insert(
            "references.unavailable_command_count".to_string(),
            fetched.map_or(0, |t| t.unavailable_commands) as f64,
        );
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
        // The pass below covers every file with this sha at once.
        if !touched.contains(&file.sha256) {
            touched.push(file.sha256.clone());
        }
    }

    // Trait evaluation already ran, before the follow phase that produced these
    // facts. Re-run it for just the files whose facts changed, so the rules that
    // read `references.*` get their pass.
    let sources: Vec<_> = touched
        .into_iter()
        .map(|sha| {
            let spans = by_source
                .get(sha.as_str())
                .map(|t| {
                    BTreeMap::from([(
                        "references.unavailable_command_count".to_string(),
                        t.unavailable_command_spans.clone(),
                    )])
                })
                .unwrap_or_default();
            (sha, spans)
        })
        .collect();
    let sources: Vec<_> = sources
        .iter()
        .map(|(sha, spans)| (sha.as_str(), spans))
        .collect();
    match cleave::graft_reference_outcomes_batch(report, &sources, &AnalysisOptions::default()) {
        Ok(0) => {}
        Err(e) => tracing::warn!("reference-outcome pass failed: {e:#}"),
        Ok(n) => tracing::debug!(
            grafted = n,
            sources = sources.len(),
            "reference-outcome traits fired"
        ),
    }
}

#[cfg(test)]
mod security_hold_finding_tests {
    use super::is_registry_security_hold_finding;

    #[test]
    fn recognizes_only_registry_hold_evidence() {
        assert!(is_registry_security_hold_finding(
            "metadata/registry::registry-security-hold"
        ));
        assert!(is_registry_security_hold_finding(
            "metadata/registry::registry-security-hold-record"
        ));
        assert!(!is_registry_security_hold_finding(
            "metadata/registry::registry-security-hold-record-impostor"
        ));
        assert!(!is_registry_security_hold_finding(
            "other/registry::registry-security-hold-record"
        ));
        assert!(!is_registry_security_hold_finding(
            "metadata/registry::deprecated"
        ));
    }
}
