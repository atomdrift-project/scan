//! Live, in-place dependency progress tree for the interactive fetch phase.
//!
//! When a single artifact is scanned on a terminal, the external references it
//! pulls (declared dependencies, install-command packages, fetched URLs) are
//! shown as a tree that updates in place: every known dependency is listed the
//! moment its hop is discovered, a spinner rides the ones being fetched or
//! analyzed, and each settles to a final glyph — fetched, cached, skipped, or
//! failed — as it completes. Transitive dependencies surfaced by a later hop
//! append to the list, so the tree grows to the full graph as it is walked.
//!
//! This is the counterpart to `crate::engine::Progress`: that bar owns the
//! terminal for a multi-file scan (one line, a file-count denominator), so the
//! tree steps aside there and the fetch log streams above the bar instead (see
//! [`crate::fetch`]). The tree takes over only when no bar is live — a lone
//! artifact — and stderr is an unclobbered terminal.
//!
//! Rendering is a bounded, cursor-relative redraw: the region is capped to the
//! viewport height (a long graph windows around the active frontier), so the
//! cursor moves never walk off-screen and the tree never scrolls itself apart.
//! State changes only mark the tree dirty; one heartbeat thread does all the
//! drawing, at most once a tick, so network threads never wait on the terminal.

use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::engine::{PROGRESS_TICK, SPINNER, bar_active, term_dims};
use crate::output::{Rgb, fg};

/// The lifecycle state a caller drives a dependency through. `Fetching` and
/// `Analyzing` render with a live spinner; `Done` is terminal and carries its
/// own glyph, colour, and trailing detail (a size, a status, a skip reason).
pub(crate) enum DepState {
    Fetching,
    Analyzing,
    /// The fetch has no useful per-reference terminal result (for example, a
    /// budget-clipped or unresolved locator). Keep the key settled internally
    /// while omitting its row.
    Hidden,
    Done {
        glyph: char,
        color: Rgb,
        detail: String,
    },
}

/// Minimum terminal height (rows) worth an in-place tree. A shorter terminal
/// can't spare enough lines around the region for the cursor math to stay on
/// screen, so the caller falls back to the streamed log.
const MIN_TERM_ROWS: usize = 8;

/// Hard ceiling on the rendered region. Kept well under a typical viewport so
/// the in-place redraw never approaches the screen's bottom edge — a region that
/// fills the viewport can't be cursor-moved back to its top after the terminal
/// scrolls, which doubles the header. The focused view (below) shows the active
/// frontier, so a compact panel loses nothing on a large graph.
const MAX_REGION_ROWS: usize = 16;

/// Rows kept clear below the region, so printing it never scrolls the screen out
/// from under the cursor and the summary line has somewhere to land on finish.
const RESERVE_ROWS: usize = 3;

/// Cap on the name column, so one long scoped package can't push rows past the
/// terminal width (a wrapped row breaks the in-place redraw).
const NAME_MAX: usize = 34;

/// Erase the current line and return to its start.
const CLEAR_LINE: &str = "\r\x1b[2K";

const DIM: Rgb = Rgb(110, 110, 110);
const DETAIL: Rgb = Rgb(130, 130, 130);
const SOURCE: Rgb = Rgb(90, 90, 90);
const LABEL: Rgb = Rgb(160, 160, 160);
const IN_FLIGHT: Rgb = Rgb(100, 180, 255);
const SETTLED: Rgb = Rgb(80, 200, 80);

/// One dependency's row: its display name, current status, and a monotonic
/// sequence stamped on each change so the focused view can pick the most
/// recently active rows.
struct Entry {
    name: String,
    status: Status,
    seq: u64,
    /// The manifest this dependency was declared in, shown dim at the end of the
    /// row so a reader can trace each dependency back to its source file. Empty
    /// for a reference discovered imperatively (no declaring manifest).
    source: String,
}

/// A dependency's rendered status. `Active` holds the transient label shown next
/// to the spinner; `Done` holds the settled presentation.
enum Status {
    Pending,
    Active(&'static str),
    Hidden,
    Done {
        glyph: char,
        color: Rgb,
        detail: String,
    },
}

impl Status {
    /// Whether this status is terminal (contributes to the `done/total` count).
    fn is_done(&self) -> bool {
        matches!(self, Status::Done { .. } | Status::Hidden)
    }
}

/// Mutable tree state, guarded by a single mutex.
struct State {
    /// Entries in discovery order; the render window is computed over this.
    entries: Vec<Entry>,
    /// Locator key → index into `entries`, for O(1) status transitions.
    index: HashMap<String, usize>,
    /// Monotonic change counter; each `add`/`set` stamps the entry's `seq` from
    /// it, so the focused view can rank rows by how recently they moved.
    clock: u64,
    /// Whether anything changed since the last redraw.
    dirty: bool,
    /// Spinner frame, advanced once per heartbeat.
    tick: u32,
    /// Lines the region currently occupies on screen, so the next redraw knows
    /// how far up to move the cursor.
    drawn: usize,
    cols: usize,
    rows: usize,
}

impl State {
    /// Next monotonic sequence value; marks the tree for the next redraw.
    fn stamp(&mut self) -> u64 {
        self.clock += 1;
        self.dirty = true;
        self.clock
    }

    fn add(&mut self, key: &str, name: &str, source: &str) {
        if self.index.contains_key(key) {
            return;
        }
        let idx = self.entries.len();
        let seq = self.stamp();
        self.entries.push(Entry {
            name: name.to_string(),
            status: Status::Pending,
            seq,
            source: source.to_string(),
        });
        self.index.insert(key.to_string(), idx);
    }

    fn set(&mut self, key: &str, status: Status) {
        let Some(&idx) = self.index.get(key) else {
            return;
        };
        let seq = self.stamp();
        let entry = &mut self.entries[idx];
        entry.seq = seq;
        entry.status = status;
    }

    fn any_active(&self) -> bool {
        self.entries
            .iter()
            .any(|e| matches!(e.status, Status::Active(_)))
    }
}

/// Shared between the handle the fetch loop drives and the heartbeat thread.
struct Inner {
    state: Mutex<State>,
    /// Set on finish: the heartbeat exits and no further redraw runs, so nothing
    /// clobbers the final tree or the summary printed beneath it.
    stopped: AtomicBool,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Handle the fetch loop drives. Finishing — or dropping — it stops and joins
/// the heartbeat.
pub(crate) struct DepTree {
    inner: Arc<Inner>,
    /// Taken by whichever of `finish` and `drop` stops it first.
    heartbeat: Mutex<Option<JoinHandle<()>>>,
}

impl DepTree {
    /// Activate a live tree, or `None` to fall back to the streamed fetch log.
    ///
    /// Declines when stderr isn't a terminal (piped/redirected output), when a
    /// scan progress bar already owns the terminal (a multi-file scan), when the
    /// terminal is too short for the region to stay on screen, when info-level
    /// logging is on (`--verbose`/`RUST_LOG`) — stray log lines would desync the
    /// in-place redraw, and the append-only stream coexists with them cleanly —
    /// or when the heartbeat that draws it cannot start.
    pub(crate) fn activate() -> Option<DepTree> {
        if !std::io::stderr().is_terminal() || bar_active() {
            return None;
        }
        if tracing::enabled!(target: "scan", tracing::Level::INFO) {
            return None;
        }
        let (cols, rows) = term_dims();
        if rows < MIN_TERM_ROWS {
            return None;
        }
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                entries: Vec::new(),
                index: HashMap::new(),
                clock: 0,
                dirty: false,
                tick: 0,
                drawn: 0,
                cols,
                rows,
            }),
            stopped: AtomicBool::new(false),
        });
        let beat = Arc::clone(&inner);
        let heartbeat = std::thread::Builder::new()
            .name("dep-tree".into())
            .spawn(move || {
                loop {
                    // Unparked early by `finish`, so stopping costs no tick.
                    std::thread::park_timeout(PROGRESS_TICK);
                    if beat.stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    let mut state = beat.lock();
                    state.tick = state.tick.wrapping_add(1);
                    // Spinners animate; a settled tree redraws only on change.
                    if state.dirty || state.any_active() {
                        render(&mut state, false);
                    }
                }
            })
            .map_err(|e| tracing::debug!(error = %e, "dependency tree heartbeat did not start; streaming instead"))
            .ok()?;
        Some(DepTree {
            inner,
            heartbeat: Mutex::new(Some(heartbeat)),
        })
    }

    /// Register a dependency as pending, keyed by its locator, noting the
    /// manifest `source` it was declared in. A key already present is ignored, so
    /// re-announcing a hop's references is idempotent.
    pub(crate) fn add(&self, key: &str, name: &str, source: &str) {
        self.inner.lock().add(key, name, source);
    }

    /// Transition a known dependency's status. A key never announced is ignored
    /// (the tree only shows what it was told about).
    pub(crate) fn set(&self, key: &str, state: DepState) {
        let status = match state {
            DepState::Fetching => Status::Active("fetching"),
            DepState::Analyzing => Status::Active("analyzing"),
            DepState::Hidden => Status::Hidden,
            DepState::Done {
                glyph,
                color,
                detail,
            } => Status::Done {
                glyph,
                color,
                detail,
            },
        };
        self.inner.lock().set(key, status);
    }

    /// Stop the heartbeat, render the settled tree one last time, and print
    /// `summary` on the line beneath it. After this the tree is inert scrollback.
    /// A tree that never saw a dependency (nothing was fetched) prints nothing —
    /// the phase stays silent, matching the streamed log.
    pub(crate) fn finish(&self, summary: &str) {
        self.stop();
        let mut state = self.inner.lock();
        if state.entries.is_empty() {
            return;
        }
        render(&mut state, true);
        drop(state);
        if !summary.is_empty() {
            eprintln!("{summary}");
        }
    }

    /// Stop and join the heartbeat, so no redraw can follow.
    fn stop(&self) {
        self.inner.stopped.store(true, Ordering::Relaxed);
        let heartbeat = self
            .heartbeat
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(heartbeat) = heartbeat {
            heartbeat.thread().unpark();
            let _ = heartbeat.join();
        }
    }
}

impl Drop for DepTree {
    fn drop(&mut self) {
        // Safety net: stop the heartbeat even if `finish` was skipped.
        self.stop();
    }
}

/// Paint the region in place from the top down, then leave the cursor on the
/// blank line just beneath it.
fn render(state: &mut State, finished: bool) {
    let lines = compose(state, finished);
    let mut out = String::new();
    // Move up to the top of the previously drawn region.
    if state.drawn > 0 {
        out.push_str(&format!("\x1b[{}A", state.drawn));
    }
    // Repaint each line, clearing whatever it overwrites.
    for line in &lines {
        out.push_str(CLEAR_LINE);
        out.push_str(line);
        out.push('\n');
    }
    // Clear any lines a taller previous region left below, then step back up so
    // the cursor rests just beneath the new content.
    let leftover = state.drawn.saturating_sub(lines.len());
    for _ in 0..leftover {
        out.push_str(CLEAR_LINE);
        out.push('\n');
    }
    if leftover > 0 {
        out.push_str(&format!("\x1b[{leftover}A"));
    }
    state.drawn = lines.len();
    state.dirty = false;
    eprint!("{out}");
    let _ = std::io::stderr().flush();
}

/// Build the region's lines: a header, then either the whole (small) graph or —
/// once it outgrows the region — a view focused on what's *being worked on right
/// now*, with the rest folded into a one-line footer count. The focus is the
/// point: on a large graph the answer to "what is it doing" is the handful of
/// in-flight rows, not a scroll of already-settled ones.
fn compose(state: &State, finished: bool) -> Vec<String> {
    let visible: Vec<usize> = state
        .entries
        .iter()
        .enumerate()
        .filter_map(|(i, entry)| (!matches!(entry.status, Status::Hidden)).then_some(i))
        .collect();
    let total = visible.len();
    // Nothing announced yet (or a scan that fetches nothing): draw no region, so
    // the header never flashes before the first dependency and a no-fetch scan
    // stays silent.
    if total == 0 {
        return Vec::new();
    }
    let done = visible
        .iter()
        .filter(|&&i| state.entries[i].status.is_done())
        .count();
    let active = state
        .entries
        .iter()
        .filter(|e| matches!(e.status, Status::Active(_)))
        .count();
    let cap = state
        .rows
        .saturating_sub(RESERVE_ROWS)
        .clamp(2, MAX_REGION_ROWS);
    // Stable name column: the widest name, capped so a long scoped package can't
    // wrap a row (which would break the redraw) or crowd out the detail.
    let name_cap = state.cols.saturating_sub(20).clamp(8, NAME_MAX);
    let widest = visible
        .iter()
        .map(|&i| state.entries[i].name.width())
        .max()
        .unwrap_or(0);
    let namew = widest.clamp(8, name_cap);
    let draw = |i: usize| row(&state.entries[i], state.tick, namew, state.cols);

    let mut lines = Vec::with_capacity(cap);
    lines.push(header(done, total, active, finished, state.cols));
    let body = cap - 1;

    if total <= body {
        lines.extend(visible.iter().map(|&i| draw(i)));
        return lines;
    }

    // Focused view. Choose which rows to show, most-relevant first: everything in
    // flight, then the most recently settled (a sense of motion), then upcoming
    // pending — reserving one line for the footer that counts the remainder.
    let slots = body.saturating_sub(1);
    let mut chosen: Vec<usize> = visible
        .iter()
        .copied()
        .filter(|&i| matches!(state.entries[i].status, Status::Active(_)))
        .collect();
    let mut recent: Vec<usize> = visible
        .iter()
        .copied()
        .filter(|&i| state.entries[i].status.is_done())
        .collect();
    recent.sort_unstable_by_key(|&i| std::cmp::Reverse(state.entries[i].seq));
    for i in recent {
        if chosen.len() >= slots {
            break;
        }
        chosen.push(i);
    }
    for &i in &visible {
        if chosen.len() >= slots {
            break;
        }
        if matches!(state.entries[i].status, Status::Pending) {
            chosen.push(i);
        }
    }
    // Render in discovery order so rows hold a stable position frame to frame.
    chosen.sort_unstable();
    lines.extend(chosen.iter().map(|&i| draw(i)));
    let hidden = total - chosen.len();
    if hidden > 0 {
        lines.push(footer(hidden, state.cols));
    }
    lines
}

/// The region's header: an arrow (or a check, once finished), the running
/// `done/total` count, and how many references are in flight right now.
fn header(done: usize, total: usize, active: usize, finished: bool, cols: usize) -> String {
    let (glyph, color) = if finished {
        ('\u{2713}', SETTLED)
    } else {
        ('\u{2b07}', IN_FLIGHT)
    };
    let mut text = format!("dependencies  {done}/{total}");
    if active > 0 {
        text.push_str(&format!("  \u{b7}  {active} in flight"));
    }
    let line = format!(
        "  {}  {}",
        fg(color, glyph.encode_utf8(&mut [0; 4])),
        fg(LABEL, &text)
    );
    clip(&line, 4 + char_width(glyph) + text.width(), cols)
}

/// The focused view's footer: how many rows aren't shown (the settled and
/// not-yet-started remainder). Dim, so the eye stays on the active rows above.
fn footer(hidden: usize, cols: usize) -> String {
    let text = format!("\u{2026} {hidden} more");
    clip(&format!("    {}", fg(DIM, &text)), 4 + text.width(), cols)
}

/// One dependency row: a status glyph (spinner while in flight), the name padded
/// to the shared column, and a dim trailing detail.
fn row(entry: &Entry, tick: u32, namew: usize, cols: usize) -> String {
    let (glyph, color, detail) = match &entry.status {
        Status::Pending => ('\u{00b7}', DIM, "pending".to_string()),
        Status::Active(label) => (spinner(tick), IN_FLIGHT, format!("{label}\u{2026}")),
        Status::Hidden => (' ', Rgb(0, 0, 0), String::new()),
        Status::Done {
            glyph,
            color,
            detail,
        } => (*glyph, *color, detail.clone()),
    };
    let name = elide_middle(&entry.name, namew);
    let namecol = namew.max(name.width());
    let pad = " ".repeat(namecol - name.width());
    let detail_col = if detail.is_empty() {
        String::new()
    } else {
        format!("  {}", fg(DETAIL, &detail))
    };
    // Trailing dim source: `from <manifest>`, so a reader can trace each
    // dependency back to the file that declared it. Sized to the columns left
    // after the name and result, and middle-elided only when it won't fit — a
    // long package-relative path keeps its telling head and its filename tail
    // (`github.com-…/package-lock.json`) rather than being chopped to a stub.
    const SOURCE_MIN: usize = 20;
    let glyph_width = char_width(glyph);
    let detail_width = if detail.is_empty() {
        0
    } else {
        2 + detail.width()
    };
    let used = 4 + glyph_width + 1 + namecol + detail_width;
    let avail = cols.saturating_sub(used + 2 + 5); // 2 gap + "from "
    let source = if entry.source.is_empty() || avail < SOURCE_MIN {
        String::new()
    } else {
        elide_middle(&entry.source, avail)
    };
    let (source_col, source_width) = if source.is_empty() {
        (String::new(), 0)
    } else {
        (
            format!("  {}", fg(SOURCE, &format!("from {source}"))),
            2 + 5 + source.width(),
        )
    };
    clip(
        &format!(
            "    {} {name}{pad}{detail_col}{source_col}",
            fg(color, glyph.encode_utf8(&mut [0; 4]))
        ),
        used + source_width,
        cols,
    )
}

/// The spinner frame for this heartbeat tick.
fn spinner(tick: u32) -> char {
    SPINNER[tick as usize % SPINNER.len()]
}

/// Display columns a glyph takes: two for an emoji flag, one otherwise.
fn char_width(c: char) -> usize {
    c.width().unwrap_or(1)
}

/// Truncate `text` to `width` display columns, marking a cut with an ellipsis.
fn truncate(text: &str, width: usize) -> String {
    if text.width() <= width {
        return text.to_string();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in text.chars() {
        let w = char_width(c);
        if used + w + 1 > width {
            break;
        }
        out.push(c);
        used += w;
    }
    out.push('\u{2026}');
    out
}

/// Shorten a name to `width` display columns by eliding the *middle*, keeping a
/// balanced head and tail around a central ellipsis. For a URL or a scoped
/// package the tail (a filename, a version) is as telling as the head (a host, a
/// scope), so a middle cut reads more than a trailing one:
/// `github.com/be5inv…aCurlySlab-34.7.0.zip` says "a versioned zip from github"
/// where a tail cut (`github.com/be5invis/Iosevka/relea…`) hides both.
fn elide_middle(name: &str, width: usize) -> String {
    if name.width() <= width {
        return name.to_string();
    }
    if width <= 1 {
        return "\u{2026}".to_string();
    }
    let keep = width - 1;
    let head_budget = keep.div_ceil(2);
    let tail_budget = keep - head_budget;
    let mut head = String::new();
    let mut used = 0;
    for c in name.chars() {
        let w = char_width(c);
        if used + w > head_budget {
            break;
        }
        head.push(c);
        used += w;
    }
    let mut tail: Vec<char> = Vec::new();
    let mut used = 0;
    for c in name.chars().rev() {
        let w = char_width(c);
        if used + w > tail_budget {
            break;
        }
        tail.push(c);
        used += w;
    }
    head.push('\u{2026}');
    head.extend(tail.into_iter().rev());
    head
}

/// Clip a coloured line whose *visible* width is `visible` so it never exceeds
/// `cols` and wraps (a wrapped line breaks the in-place redraw). When it fits,
/// it's returned untouched; when it doesn't, the ANSI-bearing tail is dropped
/// wholesale and the plain text is re-truncated with an ellipsis — details are
/// short, so this only bites pathologically narrow terminals.
fn clip(line: &str, visible: usize, cols: usize) -> String {
    if visible <= cols {
        return line.to_string();
    }
    // Fall back to a plain, hard-truncated form; colour is sacrificed for
    // correctness of the region geometry.
    truncate(&strip_ansi(line), cols)
}

/// Drop ANSI SGR escapes from a line, leaving its visible text.
pub(crate) fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // Skip up to and including the terminating 'm' of the SGR sequence.
            for e in chars.by_ref() {
                if e == 'm' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(entries: Vec<Entry>, rows: usize) -> State {
        let index = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name.clone(), i))
            .collect();
        let clock = entries.len() as u64;
        State {
            entries,
            index,
            clock,
            dirty: false,
            tick: 0,
            drawn: 0,
            cols: 100,
            rows,
        }
    }

    fn entry(name: &str, status: Status, seq: u64) -> Entry {
        Entry {
            name: name.to_string(),
            status,
            seq,
            source: String::new(),
        }
    }

    fn pending(name: &str) -> Entry {
        entry(name, Status::Pending, 0)
    }

    fn done(name: &str) -> Entry {
        entry(
            name,
            Status::Done {
                glyph: '\u{2713}',
                color: SETTLED,
                detail: "1 KB".into(),
            },
            0,
        )
    }

    fn active(name: &str, seq: u64) -> Entry {
        entry(name, Status::Active("fetching"), seq)
    }

    #[test]
    fn small_graph_renders_every_entry_plus_header() {
        let s = state(vec![done("a"), pending("b"), pending("c")], 40);
        let lines = compose(&s, false);
        // header + 3 entries, no footer.
        assert_eq!(lines.len(), 4);
        assert!(strip_ansi(&lines[0]).contains("1/3"));
    }

    #[test]
    fn hidden_entries_are_omitted_from_the_tree_count() {
        let s = state(
            vec![done("a"), entry("b", Status::Hidden, 1), pending("c")],
            40,
        );
        let body = strip_ansi(&compose(&s, false).join("\n"));
        assert!(body.contains("1/2"));
        assert!(!body.contains("b"));
        assert!(body.contains("a") && body.contains("c"));
    }

    #[test]
    fn row_shows_the_source_manifest() {
        let mut e = pending("react-dropzone");
        e.source = "vexium-1.0.tgz/package/package.json".to_string();
        let s = state(vec![e], 40);
        let body = strip_ansi(&compose(&s, false).join("\n"));
        assert!(body.contains("react-dropzone"));
        assert!(body.contains("package.json"), "source manifest not shown");
    }

    #[test]
    fn tall_graph_focuses_on_active_and_stays_within_the_cap() {
        // 100 entries, short terminal. Most are settled; a couple are in flight
        // deep in the list — the focused view must surface them, not the tail.
        let mut entries: Vec<Entry> = (0..100).map(|i| done(&format!("d{i}"))).collect();
        entries[50] = active("d50", 500);
        entries[51] = active("d51", 501);
        let rows = 12;
        let s = state(entries, rows);
        let lines = compose(&s, false);
        assert!(
            lines.len() <= rows - RESERVE_ROWS,
            "region exceeds viewport"
        );
        let body = strip_ansi(&lines.join("\n"));
        assert!(body.contains("d50"), "active row d50 not surfaced");
        assert!(body.contains("d51"), "active row d51 not surfaced");
        assert!(body.contains("more"), "hidden remainder not counted");
        assert!(strip_ansi(&lines[0]).contains("2 in flight"));
    }

    #[test]
    fn done_and_total_track_status() {
        let s = state(vec![done("a"), done("b"), pending("c")], 40);
        let lines = compose(&s, false);
        assert!(strip_ansi(&lines[0]).contains("2/3"));
    }

    /// Wide names and glyphs are measured in display columns, so a row never
    /// outgrows the terminal and wraps the in-place region apart.
    #[test]
    fn rows_fit_the_terminal_in_display_columns() {
        let mut s = state(
            vec![
                entry(
                    "日本語のパッケージ名がとても長い場合のテスト",
                    Status::Done {
                        glyph: '\u{1f6a9}',
                        color: SETTLED,
                        detail: "1 KB".into(),
                    },
                    0,
                ),
                pending("short"),
            ],
            40,
        );
        s.cols = 40;
        for line in compose(&s, false) {
            let width = strip_ansi(&line).width();
            assert!(width <= 40, "{width} columns: {line}");
        }
    }

    /// Settled state changes are drawn by the heartbeat, never by the caller;
    /// `add`/`set` only mark the tree dirty.
    #[test]
    fn a_change_marks_the_tree_dirty() {
        let mut s = state(Vec::new(), 40);
        assert!(!s.dirty);
        s.stamp();
        assert!(s.dirty);
    }

    #[test]
    fn strip_ansi_removes_color_codes() {
        assert_eq!(strip_ansi("\x1b[38;2;1;2;3mhi\x1b[0m"), "hi");
    }

    #[test]
    fn elide_middle_keeps_head_and_tail() {
        // Fits within the width: returned untouched.
        assert_eq!(elide_middle("short", 10), "short");
        // A long URL keeps its host prefix and its filename tail around the
        // ellipsis, so the version and extension survive the cut.
        let url = "github.com/be5invis/Iosevka/releases/download/v34.7.0/PkgTTF-IosevkaCurlySlab-34.7.0.zip";
        let out = elide_middle(url, 34);
        assert_eq!(
            out.chars().count(),
            34,
            "elided name must fill exactly the column"
        );
        assert!(out.starts_with("github.com/"), "host prefix lost: {out}");
        assert!(out.ends_with(".zip"), "filename tail lost: {out}");
        assert!(
            out.contains('\u{2026}'),
            "no ellipsis marking the cut: {out}"
        );
        // Wide characters count double.
        assert!(elide_middle("ああああああああああ", 9).width() <= 9);
    }
}
