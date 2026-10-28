//! What a fetch says on the terminal: live progress, one row per
//! dependency, and the summary line.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

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
pub(super) enum Reporter {
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
    pub(super) fn new(progress: bool) -> Self {
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
    pub(super) fn announce(&self, refs: &[Reference], source: &str) {
        if let Self::Tree { tree, .. } = self {
            for r in refs {
                tree.add(locator(r), &dep_display_name(r), source);
            }
        }
    }

    /// Mark the to-fetch set in flight. The stream folds its counts into the
    /// active scan bar; the tree animates each row in place.
    pub(super) fn fetching(&self, refs: &[Reference]) {
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
    pub(super) fn landed(&self, r: &Reference, rec: &FetchRecord) {
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
    pub(super) fn report(&self, rec: &FetchRecord, budget_notice: &str) {
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
            if matches!(rec.outcome, Outcome::Ok) && !analysis_unavailable(rec) {
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
    pub(super) fn claim_row(&self, rec: &FetchRecord) -> bool {
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
    pub(super) fn analyzed(&self, r: &Reference, rec: &FetchRecord) {
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
    pub(super) fn skipped(&self, r: &Reference, reg: &Registry, now: u64, reason: SkipReason) {
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
    pub(super) fn finish(&self, records: &[FetchRecord]) {
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
pub(super) const LIVE: Rgb = Rgb(100, 180, 255);
pub(super) const CACHED: Rgb = Rgb(120, 200, 140);
pub(super) const CAUTION: Rgb = Rgb(230, 180, 80);
pub(super) const FAILED: Rgb = Rgb(255, 90, 90);
pub(super) const GOOD: Rgb = Rgb(80, 200, 80);
pub(super) const MUTED: Rgb = Rgb(120, 120, 120);
pub(super) const DETAIL: Rgb = Rgb(130, 130, 130);
pub(super) const LABEL: Rgb = Rgb(160, 160, 160);

/// The source manifest's path as the dep tree shows it, led by the scanned
/// artifact so a nested manifest reads plainly as a file *inside* it:
/// `demo.zip!!vexium-1.0.tgz!!package/package.json` →
/// `demo.zip/vexium-1.0.tgz/package/package.json`. The root is reduced to its
/// basename (`/tmp/demo.zip` → `demo.zip`) and deeper archive boundaries become
/// `/`. A bare path with no archive nesting (a plain manifest scanned directly)
/// shows just its basename.
pub(super) fn manifest_relpath(path: &str) -> String {
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
pub(super) fn dep_display_name(r: &Reference) -> String {
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
pub(super) fn purl_display(purl: &str) -> String {
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
pub(super) fn delivered_bytes(rec: &FetchRecord) -> bool {
    matches!(
        rec.outcome,
        Outcome::Ok | Outcome::PinMismatch | Outcome::UnverifiablePin
    )
}

/// What a fetch is shown and filed under: the URL it resolved to, or the bare
/// locator when it never resolved to one.
#[must_use]
pub fn fetch_target(rec: &FetchRecord) -> &str {
    rec.resolved_url.as_deref().unwrap_or(&rec.locator)
}

/// The tree state for a fetch the moment it lands: "analyzing" when bytes are in
/// hand and a scan will follow (an `Ok`, a pin mismatch, or an unverifiable pin —
/// each pin outcome settles to its own glyph once analyzed), else the settled
/// fetch glyph.
pub(super) fn landed_state(rec: &FetchRecord) -> DepState {
    if delivered_bytes(rec) {
        DepState::Analyzing
    } else {
        done_state(rec)
    }
}

/// The settled tree state for a fetch: the shared [`fetch_row`] glyph/colour,
/// with the detail column (a size, or a failure note) as its trailing text.
pub(super) fn done_state(rec: &FetchRecord) -> DepState {
    if !terminal_fetch_row_visible(rec) {
        return DepState::Hidden;
    }
    let row = fetch_row(rec);
    DepState::Done {
        glyph: row.glyph,
        color: row.color,
        detail: row
            .detail
            .unwrap_or_else(|| rec.size.map_or(String::new(), crate::output::human_size)),
    }
}

/// The settled tree state for an age-gate skip, mirroring [`report_skip`]'s
/// glyph and colour with a concise reason as the detail.
pub(super) fn skip_state(reg: &Registry, now: u64, reason: SkipReason) -> DepState {
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
pub(super) fn report_fetch(rec: &FetchRecord) {
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
        .unwrap_or_else(|| rec.size.map_or(String::new(), crate::output::human_size));
    eprintln!(
        "    {} {}  {}{redirect}",
        fg(row.color, &format!("{} {:<6}", row.glyph, row.label)),
        fg(DETAIL, &format!("{column:>10}")),
        crate::output::tty_text(fetch_target(rec))
    );
    if analysis_unavailable(rec) {
        eprintln!("      {}", crate::output::tty_text(&coverage_text(rec)));
    }
}

/// Whether one fetch outcome deserves a terminal row. An unresolved locator
/// is quiet unless it explicitly records an analysis gap. A non-target
/// reference (a source repo, an unclassified string) was never going to be
/// fetched at all — a row for either says nothing about the artifact being
/// scanned. Retain their [`FetchRecord`]s for machine output and diagnostics,
/// but keep the default human view quiet.
///
/// A corpus hit (see [`corpus_hit_record`]) is quiet for the same reason: the
/// fleet already judged those bytes, so there is nothing here to look at.
pub(super) fn terminal_fetch_row_visible(rec: &FetchRecord) -> bool {
    if analysis_unavailable(rec) {
        return true;
    }
    match rec.outcome {
        Outcome::Unresolved(_) | Outcome::Skipped => false,
        Outcome::Failed(_) => !artifact_absent(rec),
        _ => true,
    }
}

pub(super) fn analysis_unavailable(rec: &FetchRecord) -> bool {
    rec.coverage
        .iter()
        .any(|fact| matches!(fact, Coverage::Unavailable { .. }))
}

/// Attention counts survive dropping the full report for terminal output.
/// Candidate issues concern the retrieved candidate, not the importing file.
pub(crate) fn dependency_attention(records: &[FetchRecord]) -> crate::engine::DependencyAttention {
    let mut attention = crate::engine::DependencyAttention::default();
    for record in records {
        if record
            .coverage
            .iter()
            .any(|fact| matches!(fact, Coverage::CandidateFlagged { .. }))
        {
            attention.flagged_candidates += 1;
        }
        if record.coverage.iter().any(Coverage::is_gap)
            || matches!(
                record.outcome,
                Outcome::Failed(_)
                    | Outcome::BudgetExceeded
                    | Outcome::PinMismatch
                    | Outcome::UnverifiablePin
            )
            || (matches!(record.outcome, Outcome::Unresolved(_))
                && matches!(
                    record.kind,
                    RefKind::Dependency | RefKind::Command | RefKind::UrlFetch
                ))
        {
            attention.gaps += 1;
        }
    }
    attention
}

/// Whether a failed fetch failed because the artifact is simply not there:
/// `404` (never published, or a name the registry doesn't carry) or `410`
/// (withdrawn). Neither says anything about the artifact being scanned — a
/// manifest naming a package no registry ever had is a fact about the manifest
/// — and a large lockfile produces them by the dozen, so they are recorded and
/// logged at debug rather than printed. Every other failure (transport,
/// timeout, `5xx`, a `403` that may be an authenticated mirror) is a fetch that
/// *should* have worked, and stays visible.
pub(super) fn artifact_absent(rec: &FetchRecord) -> bool {
    matches!(rec.status, Some(404 | 410))
}

/// How one fetch outcome reads in the terminal: a glyph and short label in one
/// colour, and the detail that replaces the size column when the fetch
/// delivered no bytes.
pub(super) struct FetchRow {
    pub(super) glyph: char,
    pub(super) label: &'static str,
    pub(super) color: Rgb,
    pub(super) detail: Option<String>,
}

impl FetchRow {
    pub(super) const fn new(glyph: char, label: &'static str, color: Rgb) -> Self {
        Self {
            glyph,
            label,
            color,
            detail: None,
        }
    }

    pub(super) fn detail(mut self, detail: String) -> Self {
        self.detail = Some(detail);
        self
    }
}

/// The display row for a fetch outcome. Shared by the streamed log
/// ([`report_fetch`]) and the live tree, so a dependency reads the same either
/// way: a *fetched* dep the local bloom filters vouch for (or flag) is relabeled
/// `known` instead of `live`/`cache`; `skip`/`fail` rows are left as-is.
pub(super) fn fetch_row(rec: &FetchRecord) -> FetchRow {
    if analysis_unavailable(rec)
        && matches!(
            rec.outcome,
            Outcome::Ok | Outcome::Skipped | Outcome::Unresolved(_)
        )
    {
        let flagged = rec
            .coverage
            .iter()
            .any(|fact| matches!(fact, Coverage::CandidateFlagged { .. }));
        return FetchRow::new('\u{26a0}', "gap", CAUTION).detail(
            if flagged {
                "candidate flagged"
            } else {
                "not verified"
            }
            .into(),
        );
    }
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
        Outcome::Unresolved(_) if terminal_fetch_row_visible(rec) => {
            FetchRow::new('\u{26a0}', "gap", CAUTION).detail("not analyzed".to_string())
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
pub(super) fn bloom_fetch_verdict(rec: &FetchRecord) -> Option<FetchRow> {
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
pub(super) fn failure_detail(why: &FetchError) -> String {
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

/// Render one skipped dependency to stderr in the fetch progress block: the
/// package, its age in days, and the strongest reputation signal the registry
/// gave (downloads, else votes/rating). A known-good skip reads green with a ✓
/// (trusted, not re-scanned); a removed version reads muted (no artifact to
/// fetch). Aged-out deps never reach here — they stay at debug.
pub(super) fn report_skip(r: &Reference, reg: &Registry, now: u64, reason: SkipReason) {
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
        crate::output::tty_text(locator(r))
    );
}

pub(super) fn summary_line(records: &[FetchRecord]) -> String {
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
        let has = |pick: fn(&Coverage) -> bool| rec.coverage.iter().any(pick);
        if matches!(
            rec.outcome,
            Outcome::BudgetExceeded | Outcome::Unresolved(_)
        ) || has(|fact| matches!(fact, Coverage::Pending { .. }))
        {
            pending += 1;
        }
        if has(|fact| matches!(fact, Coverage::Supplied { .. })) {
            local += 1;
        }
        if has(|fact| matches!(fact, Coverage::DevelopmentExcluded)) {
            dev += 1;
        }
        if has(is_analysis_gap) {
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
    parts.push(crate::output::human_size(bytes));
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
