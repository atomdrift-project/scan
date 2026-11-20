//! What a fetch says on the terminal: live progress, one row per
//! dependency, and the summary line.

use std::collections::HashSet;
use std::sync::PoisonError;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use fletch::fetch::{Coverage, FetchError, FetchRecord, Outcome, Served, Unresolved};
use fletch::{RefKind, RefLocator, Reference, Registry};
use reqwest::Url;

use crate::deptree::{DepState, DepTree};
use crate::hosts;
use crate::output::{Ink, Rgb};

use super::age_gate::SkipReason;
use super::registry::registry_summary;
use super::{Coordinate, SECS_PER_DAY, coverage_text, is_analysis_gap, locator};

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
        /// many manifests is one fact, not one line per manifest.
        printed: std::sync::Mutex<HashSet<String>>,
        /// Every row printed, in order: what [`Reporter::finish`] returns.
        shown: std::sync::Mutex<String>,
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
                shown: std::sync::Mutex::default(),
            },
            |tree| Self::Tree {
                tree,
                budget_notice: std::sync::Mutex::new(None),
            },
        )
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
            // and pin trouble earn a line. A coverage gap is neither — a
            // package with hundreds of optional imports has hundreds of them,
            // and a line above a bar shared by every file being scanned names
            // no owner — so the file's own card summarizes them (see
            // [`coverage_gaps`]). A flagged candidate is the exception: that
            // one is a finding about something this artifact may load.
            let flagged = rec
                .coverage
                .iter()
                .any(|fact| matches!(fact, Coverage::CandidateFlagged { .. }));
            if coverage_gap(rec) {
                if !flagged {
                    return;
                }
            } else if matches!(rec.outcome, Outcome::Ok) {
                return;
            }
            if !self.claim_row(rec) {
                return;
            }
            self.say(&fetch_rows(rec));
        }
    }

    /// Print `rows` (whole lines) above the scan bar and remember them
    /// (stream only).
    fn say(&self, rows: &str) {
        if let Self::Stream { shown, .. } = self {
            crate::engine::print_above_bar(|| eprint!("{rows}"));
            shown
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push_str(rows);
        }
    }

    /// Whether this row is the first of its kind, claiming it if so. A row is
    /// its outcome over its target, because that pair is the whole of what the
    /// line says: one dependency named by many manifests fails identically
    /// once per manifest, and a monorepo would spend most of its output
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
                    age_days = reg.age_secs(now).unwrap_or(0) / SECS_PER_DAY,
                    "fetch: known-good dependency skipped"
                );
            }
            // Aged-out returned above and known-good is the arm before, so
            // only a withdrawn version reaches the stream.
            Self::Stream { .. } => self.say(&removed_row(r, reg, now)),
            Self::Tree { tree, .. } => {
                tree.add(locator(r), &dep_display_name(r), "");
                tree.set(locator(r), skip_state(reg, now, reason));
            }
        }
    }

    /// Close out the phase: the stream prints its one-line tally (only if it ever
    /// printed a row); the tree settles and prints the same tally beneath the
    /// dependency rows. Returns what the phase leaves on stderr, so a replay
    /// of the scan can show it again.
    pub(super) fn finish(&self, records: &[FetchRecord]) -> String {
        match self {
            Self::Off => String::new(),
            Self::Stream {
                external_dependencies,
                external_urls,
                shown,
                ..
            } => {
                let dependencies = external_dependencies.swap(0, Ordering::Relaxed);
                let urls = external_urls.swap(0, Ordering::Relaxed);
                crate::engine::external_fetch_finished(dependencies as usize, urls as usize);
                std::mem::take(&mut *shown.lock().unwrap_or_else(PoisonError::into_inner))
            }
            Self::Tree {
                tree,
                budget_notice,
            } => {
                let mut left = tree.finish(&summary_line(records));
                let message = budget_notice
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take();
                if let Some(message) = message {
                    let line = format!("    {message}\n");
                    eprint!("{line}");
                    left.push_str(&line);
                }
                left
            }
        }
    }
}

/// The fetch log's palette, matching the scan progress bar's truecolor one.
pub(super) const LIVE: Rgb = Rgb(100, 180, 255);
const CACHED: Rgb = Rgb(120, 200, 140);
const CAUTION: Rgb = Rgb(230, 180, 80);
const FAILED: Rgb = Rgb(255, 90, 90);
const GOOD: Rgb = Rgb(80, 200, 80);
const MUTED: Rgb = Rgb(120, 120, 120);
const DETAIL: Rgb = Rgb(130, 130, 130);
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
    let name = coordinate
        .decoded_path()
        .unwrap_or_else(|| coordinate.path.to_owned());
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

/// The settled tree state for an age-gate skip, mirroring [`removed_row`]'s
/// glyph and colour with a concise reason as the detail.
fn skip_state(reg: &Registry, now: u64, reason: SkipReason) -> DepState {
    let age_days = reg.age_secs(now).unwrap_or(0) / SECS_PER_DAY;
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
        Ink::stderr().fg(Rgb(180, 160, 255), "\u{24d8}"),
        Ink::stderr().fg(LABEL, "registry"),
        Ink::stderr().fg(MUTED, reg.ecosystem.as_str())
    );
    let parts = registry_summary(reg);
    if !parts.is_empty() {
        eprintln!(
            "    {}",
            Ink::stderr().fg(DETAIL, &parts.join("  \u{00b7}  "))
        );
    }
}

/// One fetched reference as its stderr rows (whole lines; empty when the row
/// is not shown), distinguishing a live network fetch from a cache hit and
/// naming the actual URL that was (or would be) retrieved. Only the
/// interactive terminal path prints them; JSON/server callers stay silent.
pub(super) fn fetch_rows(rec: &FetchRecord) -> String {
    if !terminal_fetch_row_visible(rec) {
        return String::new();
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
                Ink::stderr().fg(MUTED, &format!("\u{2192} {}", hosts::authority(&f)))
            )
        })
        .unwrap_or_default();
    let column = row
        .detail
        .unwrap_or_else(|| rec.size.map_or(String::new(), crate::output::human_size));
    let mut rows = format!(
        "    {} {}  {}{redirect}\n",
        Ink::stderr().fg(row.color, &format!("{} {:<6}", row.glyph, row.label)),
        Ink::stderr().fg(DETAIL, &format!("{column:>10}")),
        crate::output::tty_text(fetch_target(rec))
    );
    if analysis_unavailable(rec) {
        rows.push_str("      ");
        rows.push_str(&crate::output::tty_text(&coverage_text(rec)));
        rows.push('\n');
    }
    rows
}

/// Whether one fetch outcome deserves a terminal row. An unresolved locator
/// is quiet unless it explicitly records an analysis gap. A non-target
/// reference (a source repo, an unclassified string) was never going to be
/// fetched at all — a row for either says nothing about the artifact being
/// scanned. Retain their [`FetchRecord`]s for machine output and diagnostics,
/// but keep the default human view quiet.
///
/// A corpus hit (see [`corpus_hit_record`](super::merge::corpus_hit_record)) is quiet for the same reason: the
/// corpus already judged those bytes, so there is nothing here to look at.
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

/// Whether `rec` is a coverage gap rather than a fetch that went wrong: the
/// reference was retrieved, or deliberately left alone, and what is missing
/// is an analysis of it.
fn coverage_gap(rec: &FetchRecord) -> bool {
    analysis_unavailable(rec)
        && matches!(
            rec.outcome,
            Outcome::Ok | Outcome::Skipped | Outcome::Unresolved(_)
        )
}

/// The coverage gaps among `records`, grouped by reason in first-seen order,
/// each with the distinct references it covers: what a result card shows in
/// place of one row per reference. `source_path` names the file a declaring
/// sha belongs to, so a gap reads as a place in the artifact, not a digest.
pub(crate) fn coverage_gaps(
    records: &[FetchRecord],
    source_path: impl Fn(&str) -> Option<String>,
) -> Vec<(&str, Vec<String>)> {
    let mut groups: Vec<(&str, Vec<String>)> = Vec::new();
    let mut seen = HashSet::new();
    for rec in records.iter().filter(|rec| coverage_gap(rec)) {
        let reason = rec
            .coverage
            .iter()
            .find_map(|fact| match fact {
                Coverage::Unavailable { reason } => reason.as_deref(),
                _ => None,
            })
            .unwrap_or("not analyzed");
        let target = fetch_target(rec);
        let label = if rec.locator.starts_with("pkg:") {
            purl_display(&rec.locator)
        } else if let Some(sha) = rec.source_sha256.as_deref()
            && let Some(path) = source_path(sha)
        {
            target.replace(sha, &path)
        } else {
            target.to_owned()
        };
        if !seen.insert((reason, label.clone())) {
            continue;
        }
        match groups.iter_mut().find(|(known, _)| *known == reason) {
            Some((_, labels)) => labels.push(label),
            None => groups.push((reason, vec![label])),
        }
    }
    groups
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

/// Whether every fetch reached an answer that asking again would not change.
/// A transport failure, a registry reply that could not be read, an exhausted
/// budget and a reference the run stopped short of are all accidents of this
/// run: a result built on one must not stand in for the next run's.
pub(crate) fn settled(records: &[FetchRecord]) -> bool {
    let unsettled = |record: &&FetchRecord| {
        match &record.outcome {
        Outcome::BudgetExceeded | Outcome::Unresolved(Unresolved::InvalidResponse) => true,
        Outcome::Failed(_) => !artifact_absent(record),
        _ => record.coverage.iter().any(|fact| {
            matches!(fact, Coverage::Pending { reason, .. } if reason == super::session::FETCH_TIMEOUT)
        }),
    }
    };
    let mut all = true;
    for record in records.iter().filter(unsettled) {
        tracing::debug!(
            locator = %record.locator,
            outcome = ?record.outcome,
            status = ?record.status,
            coverage = ?record.coverage,
            "fetch not settled: a later run may answer differently"
        );
        all = false;
    }
    all
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
    rec.status.is_some_and(gone_status)
}

/// Whether an HTTP status says the artifact is not there: `404` (never
/// published, or a name the source does not carry) or `410` (withdrawn).
pub(super) const fn gone_status(status: u16) -> bool {
    matches!(status, 404 | 410)
}

/// How one fetch outcome reads in the terminal: a glyph and short label in one
/// colour, and the detail that replaces the size column when the fetch
/// delivered no bytes.
pub(super) struct FetchRow {
    glyph: char,
    pub(super) label: &'static str,
    color: Rgb,
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
/// ([`fetch_rows`]) and the live tree, so a dependency reads the same either
/// way: a *fetched* dep the local bloom filters vouch for (or flag) is relabeled
/// `known` instead of `live`/`cache`; `skip`/`fail` rows are left as-is.
pub(super) fn fetch_row(rec: &FetchRecord) -> FetchRow {
    if coverage_gap(rec) {
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
        // An outcome newer than this build: say so rather than guess.
        _ => FetchRow::new('\u{00b7}', "?", MUTED).detail("unrecognized outcome".to_string()),
    }
}

/// A bloom verdict for a fetched artifact, as a [`fetch_rows`] row override:
/// every known state renders as the `known` label, distinguished by glyph —
/// 🚩 known-bad, 🏴 conflicted, 👁 sighted by somebody else (all still scanned;
/// the flag also rides the result header), green ✓ known-good (fetched here
/// only because a pulled/fresh exception forced a re-scan). `None` when bloom
/// is disabled or the artifact is in neither set.
///
/// The digest and the PURL both name this one artifact, so both are handed to
/// the filters together and `burton` combines them: the worst claim against
/// either wins, and the green tick requires every key to agree, so a blessed
/// coordinate never shows clean while its digest is cited by threat
/// intelligence.
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
        Decision::Unknown | Decision::Unavailable => return None,
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

/// Render one withdrawn dependency (no artifact to fetch) to stderr in the
/// fetch progress block: the package, its age in days, and the strongest
/// reputation signal the registry gave (downloads, else votes/rating). The
/// other skips never reach the stream: aged-out and known-good stay at debug.
fn removed_row(r: &Reference, reg: &Registry, now: u64) -> String {
    let age_days = reg.age_secs(now).unwrap_or(0) / SECS_PER_DAY;
    let signal = reg
        .downloads_recent
        .or(reg.downloads_total)
        .map(|d| format!("{d} dl"))
        .or_else(|| reg.rating_count.map(|v| format!("{v} votes")));
    let detail = signal.map_or_else(String::new, |signal| {
        format!("  {}", Ink::stderr().fg(MUTED, &signal))
    });
    format!(
        "    {} {}  {}{detail}\n",
        Ink::stderr().fg(MUTED, &format!("\u{00b7} {:<10}", "removed")),
        Ink::stderr().fg(DETAIL, &format!("{:>10}", format!("{age_days}d old"))),
        crate::output::tty_text(locator(r))
    )
}

pub(super) fn summary_line(records: &[FetchRecord]) -> String {
    let mut live = 0u32;
    let mut cached = 0u32;
    let mut failed = 0u32;
    let mut bytes = 0u64;
    let mut pending = 0u32;
    let mut local = 0u32;
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
        if has(is_analysis_gap) {
            incomplete += 1;
        }
        match &rec.outcome {
            Outcome::Ok | Outcome::PinMismatch | Outcome::UnverifiablePin if rec.is_cached() => {
                cached += 1;
            }
            Outcome::Ok | Outcome::PinMismatch | Outcome::UnverifiablePin => live += 1,
            Outcome::Failed(_) => failed += 1,
            // Not a byte transfer, or an outcome newer than this build.
            _ => {}
        }
    }
    // Only the counts that actually happened, so a warm run reads
    // `2 cached  ·  160.5 MB` instead of padding a `0 live` nobody asked about.
    // Bytes always show — the total pulled is the headline the tally exists for.
    let mut parts = Vec::new();
    for (count, label) in [
        (live, "live"),
        (cached, "cached"),
        (failed, "failed"),
        (pending, "pending"),
        (local, "supplied locally"),
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
            Ink::stderr().fg(Rgb(220, 160, 80), "!")
        } else {
            Ink::stderr().fg(Rgb(80, 220, 80), "\u{2713}")
        },
        Ink::stderr().fg(LABEL, &parts.join("  \u{b7}  "))
    )
}
