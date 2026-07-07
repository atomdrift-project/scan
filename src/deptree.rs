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
//! This is the counterpart to [`crate::engine::Progress`]: that bar owns the
//! terminal for a multi-file scan (one line, a file-count denominator), so the
//! tree steps aside there and the fetch log streams above the bar instead (see
//! [`crate::fetch`]). The tree takes over only when no bar is live — a lone
//! artifact — and stderr is an unclobbered terminal.
//!
//! Rendering is a bounded, cursor-relative redraw: the region is capped to the
//! viewport height (a long graph windows around the active frontier), so the
//! `\x1b[{n}A` cursor moves never walk off-screen and the tree never scrolls
//! itself apart.

use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use crate::engine::{PROGRESS_TICK, SPINNER, bar_active, term_dims};

/// The lifecycle state a caller drives a dependency through. `Fetching` and
/// `Analyzing` render with a live spinner; `Done` is terminal and carries its
/// own glyph, colour, and trailing detail (a size, a status, a skip reason).
pub(crate) enum DepState {
    Fetching,
    Analyzing,
    Done {
        glyph: char,
        color: (u8, u8, u8),
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

/// One dependency's row: its display name, current status, and a monotonic
/// sequence stamped on each change so the focused view can pick the most
/// recently active rows.
struct Entry {
    name: String,
    status: Status,
    seq: u64,
}

/// A dependency's rendered status. `Active` holds the transient label shown next
/// to the spinner; `Done` holds the settled presentation.
enum Status {
    Pending,
    Active(&'static str),
    Done {
        glyph: char,
        color: (u8, u8, u8),
        detail: String,
    },
}

impl Status {
    /// Whether this status is terminal (contributes to the `done/total` count).
    fn is_done(&self) -> bool {
        matches!(self, Status::Done { .. })
    }
}

/// Mutable tree state, guarded by a single mutex so state changes and the
/// stderr redraw they trigger never interleave with the heartbeat thread's.
struct State {
    /// Entries in discovery order; the render window is computed over this.
    entries: Vec<Entry>,
    /// Locator key → index into `entries`, for O(1) status transitions.
    index: HashMap<String, usize>,
    /// Monotonic change counter; each `add`/`set` stamps the entry's `seq` from
    /// it, so the focused view can rank rows by how recently they moved.
    clock: u64,
    /// Lines the region currently occupies on screen, so the next redraw knows
    /// how far up to move the cursor.
    drawn: usize,
    cols: usize,
    rows: usize,
}

impl State {
    /// Next monotonic sequence value.
    fn stamp(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }
}

/// Shared tree behind an `Arc` so the heartbeat thread holds a clone independent
/// of the handle the fetch loop drives.
struct Inner {
    state: Mutex<State>,
    /// Spinner frame, advanced once per heartbeat so in-flight rows animate even
    /// while no dependency changes state.
    tick: AtomicU32,
    /// Set on finish: the heartbeat exits and no further redraw runs, so nothing
    /// clobbers the final tree or the summary printed beneath it.
    stopped: AtomicBool,
}

/// Handle the fetch loop drives. Dropping it stops the heartbeat.
pub(crate) struct DepTree {
    inner: Arc<Inner>,
}

impl DepTree {
    /// Activate a live tree, or `None` to fall back to the streamed fetch log.
    ///
    /// Declines when stderr isn't a terminal (piped/redirected output), when a
    /// scan progress bar already owns the terminal (a multi-file scan), when the
    /// terminal is too short for the region to stay on screen, or when info-level
    /// logging is on (`--verbose`/`RUST_LOG`) — stray log lines would desync the
    /// in-place redraw, and the append-only stream coexists with them cleanly.
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
                drawn: 0,
                cols,
                rows,
            }),
            tick: AtomicU32::new(0),
            stopped: AtomicBool::new(false),
        });
        // Detached heartbeat: it observes `stopped` and exits within one tick of
        // finish/drop. A spawn failure just means no animation — state changes
        // still redraw the tree.
        let beat = Arc::clone(&inner);
        let _ = std::thread::Builder::new()
            .name("dep-tree".into())
            .spawn(move || {
                while !beat.stopped.load(Ordering::Relaxed) {
                    std::thread::sleep(PROGRESS_TICK);
                    if beat.stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    beat.tick.fetch_add(1, Ordering::Relaxed);
                    beat.render();
                }
            });
        Some(DepTree { inner })
    }

    /// Register a dependency as pending, keyed by its locator. A key already
    /// present is ignored, so re-announcing a hop's references is idempotent.
    pub(crate) fn add(&self, key: &str, name: &str) {
        let mut state = self.lock();
        if state.index.contains_key(key) {
            return;
        }
        let idx = state.entries.len();
        let seq = state.stamp();
        state.entries.push(Entry {
            name: name.to_string(),
            status: Status::Pending,
            seq,
        });
        state.index.insert(key.to_string(), idx);
        drop_render(&self.inner, state);
    }

    /// Transition a known dependency's status. A key never announced is ignored
    /// (the tree only shows what it was told about).
    pub(crate) fn set(&self, key: &str, state: DepState) {
        let mut guard = self.lock();
        let Some(&idx) = guard.index.get(key) else {
            return;
        };
        let seq = guard.stamp();
        let entry = &mut guard.entries[idx];
        entry.seq = seq;
        entry.status = match state {
            DepState::Fetching => Status::Active("fetching"),
            DepState::Analyzing => Status::Active("analyzing"),
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
        drop_render(&self.inner, guard);
    }

    /// Stop the heartbeat, render the settled tree one last time, and print
    /// `summary` on the line beneath it. After this the tree is inert scrollback.
    /// A tree that never saw a dependency (nothing was fetched) prints nothing —
    /// the phase stays silent, matching the streamed log.
    pub(crate) fn finish(&self, summary: &str) {
        self.inner.stopped.store(true, Ordering::Relaxed);
        // Heartbeat is stopped, so this crate's fetch loop is the only accessor —
        // the re-lock for the final render is uncontended.
        if self.lock().entries.is_empty() {
            return;
        }
        let tick = self.inner.tick.load(Ordering::Relaxed);
        render_locked(self.lock(), tick, true);
        if !summary.is_empty() {
            eprintln!("{summary}");
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Drop for DepTree {
    fn drop(&mut self) {
        // Safety net: stop the heartbeat even if `finish` was skipped.
        self.inner.stopped.store(true, Ordering::Relaxed);
    }
}

impl Inner {
    /// Heartbeat/state-change redraw at the current spinner frame, unless
    /// finished (a late tick must never repaint over the summary line).
    fn render(&self) {
        if self.stopped.load(Ordering::Relaxed) {
            return;
        }
        let guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tick = self.tick.load(Ordering::Relaxed);
        render_locked(guard, tick, false);
    }
}

/// Redraw after a state change, holding the lock through the write so the
/// heartbeat can't repaint between the mutation and its redraw.
fn drop_render(inner: &Arc<Inner>, guard: std::sync::MutexGuard<'_, State>) {
    let tick = inner.tick.load(Ordering::Relaxed);
    render_locked(guard, tick, false);
}

/// Paint the region in place from the top down, then leave the cursor on the
/// blank line just beneath it. The caller holds the state lock.
fn render_locked(mut state: std::sync::MutexGuard<'_, State>, tick: u32, finished: bool) {
    let lines = compose(&state, tick, finished);
    let mut out = String::new();
    // Move up to the top of the previously drawn region.
    if state.drawn > 0 {
        out.push_str(&format!("\x1b[{}A", state.drawn));
    }
    // Repaint each line, clearing whatever it overwrites.
    for line in &lines {
        out.push_str("\r\x1b[2K");
        out.push_str(line);
        out.push('\n');
    }
    // Clear any lines a taller previous region left below, then step back up so
    // the cursor rests just beneath the new content.
    let leftover = state.drawn.saturating_sub(lines.len());
    for _ in 0..leftover {
        out.push_str("\r\x1b[2K\n");
    }
    if leftover > 0 {
        out.push_str(&format!("\x1b[{leftover}A"));
    }
    state.drawn = lines.len();
    eprint!("{out}");
    let _ = std::io::stderr().flush();
}

/// Build the region's lines: a header, then either the whole (small) graph or —
/// once it outgrows the region — a view focused on what's *being worked on right
/// now*, with the rest folded into a one-line footer count. The focus is the
/// point: on a large graph the answer to "what is it doing" is the handful of
/// in-flight rows, not a scroll of already-settled ones.
fn compose(state: &State, tick: u32, finished: bool) -> Vec<String> {
    let total = state.entries.len();
    // Nothing announced yet (or a scan that fetches nothing): draw no region, so
    // the header never flashes before the first dependency and a no-fetch scan
    // stays silent.
    if total == 0 {
        return Vec::new();
    }
    let done = state.entries.iter().filter(|e| e.status.is_done()).count();
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
    let widest = state
        .entries
        .iter()
        .map(|e| e.name.chars().count())
        .max()
        .unwrap_or(0);
    let namew = widest.clamp(8, name_cap);

    let mut lines = Vec::with_capacity(cap);
    lines.push(header(done, total, active, finished, state.cols));
    let body = cap - 1;

    if total <= body {
        for e in &state.entries {
            lines.push(row(e, tick, namew, state.cols));
        }
        return lines;
    }

    // Focused view. Choose which rows to show, most-relevant first: everything in
    // flight, then the most recently settled (a sense of motion), then upcoming
    // pending — reserving one line for the footer that counts the remainder.
    let slots = body.saturating_sub(1);
    let mut chosen: Vec<usize> = (0..total)
        .filter(|&i| matches!(state.entries[i].status, Status::Active(_)))
        .collect();
    let mut recent: Vec<usize> = (0..total)
        .filter(|&i| state.entries[i].status.is_done())
        .collect();
    recent.sort_unstable_by_key(|&i| std::cmp::Reverse(state.entries[i].seq));
    for i in recent {
        if chosen.len() >= slots {
            break;
        }
        chosen.push(i);
    }
    if chosen.len() < slots {
        for i in 0..total {
            if chosen.len() >= slots {
                break;
            }
            if matches!(state.entries[i].status, Status::Pending) {
                chosen.push(i);
            }
        }
    }
    // Render in discovery order so rows hold a stable position frame to frame.
    chosen.sort_unstable();
    for &i in &chosen {
        lines.push(row(&state.entries[i], tick, namew, state.cols));
    }
    let hidden = total - chosen.len();
    if hidden > 0 {
        lines.push(footer(hidden, state.cols));
    }
    lines
}

/// The region's header: an arrow (or a check, once finished), the running
/// `done/total` count, and how many references are in flight right now.
fn header(done: usize, total: usize, active: usize, finished: bool, cols: usize) -> String {
    let (glyph, r, g, b) = if finished {
        ('\u{2713}', 80, 200, 80)
    } else {
        ('\u{2b07}', 100, 180, 255)
    };
    let mut text = format!("dependencies  {done}/{total}");
    if active > 0 {
        text.push_str(&format!("  \u{b7}  {active} in flight"));
    }
    clip(
        &format!("  \x1b[38;2;{r};{g};{b}m{glyph}\x1b[0m  \x1b[38;2;160;160;160m{text}\x1b[0m"),
        // Visible width: 2 + 1 (glyph) + 2 + the text.
        5 + text.chars().count(),
        cols,
    )
}

/// The focused view's footer: how many rows aren't shown (the settled and
/// not-yet-started remainder). Dim, so the eye stays on the active rows above.
fn footer(hidden: usize, cols: usize) -> String {
    let text = format!("\u{2026} {hidden} more");
    clip(
        &format!("    \x1b[38;2;110;110;110m{text}\x1b[0m"),
        4 + text.chars().count(),
        cols,
    )
}

/// One dependency row: a status glyph (spinner while in flight), the name padded
/// to the shared column, and a dim trailing detail.
fn row(entry: &Entry, tick: u32, namew: usize, cols: usize) -> String {
    let (glyph, color, detail) = match &entry.status {
        Status::Pending => ('\u{00b7}', (110, 110, 110), "pending".to_string()),
        Status::Active(label) => (spinner(tick), (100, 180, 255), format!("{label}\u{2026}")),
        Status::Done {
            glyph,
            color,
            detail,
        } => (*glyph, *color, detail.clone()),
    };
    let name = elide_middle(&entry.name, namew);
    let (r, g, b) = color;
    let detail_col = if detail.is_empty() {
        String::new()
    } else {
        format!("  \x1b[38;2;130;130;130m{detail}\x1b[0m")
    };
    // Visible width for clipping: 4 indent + glyph + space + padded name + 2 +
    // detail. The name is already truncated to `namew`.
    let visible = 4 + 1 + 1 + namew.max(name.chars().count()) + 2 + detail.chars().count();
    clip(
        &format!("    \x1b[38;2;{r};{g};{b}m{glyph}\x1b[0m {name:<namew$}{detail_col}"),
        visible,
        cols,
    )
}

/// The spinner frame for this heartbeat tick.
fn spinner(tick: u32) -> char {
    SPINNER[tick as usize % SPINNER.len()]
}

/// Truncate a name to `width` display columns, marking a cut with an ellipsis.
fn truncate(name: &str, width: usize) -> String {
    if name.chars().count() <= width {
        return name.to_string();
    }
    let mut out: String = name.chars().take(width.saturating_sub(1)).collect();
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
    let total = name.chars().count();
    if total <= width {
        return name.to_string();
    }
    if width <= 1 {
        return "\u{2026}".to_string();
    }
    let keep = width - 1;
    let head = keep.div_ceil(2);
    let tail = keep - head;
    let chars: Vec<char> = name.chars().collect();
    let mut out: String = chars[..head].iter().collect();
    out.push('\u{2026}');
    out.extend(&chars[total - tail..]);
    out
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
    let plain: String = strip_ansi(line);
    truncate(&plain, cols)
}

/// Drop ANSI SGR escapes from a line, leaving its visible text.
fn strip_ansi(line: &str) -> String {
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
#[allow(clippy::unwrap_used)]
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
                color: (80, 200, 80),
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
        let lines = compose(&s, 0, false);
        // header + 3 entries, no footer.
        assert_eq!(lines.len(), 4);
        assert!(strip_ansi(&lines[0]).contains("1/3"));
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
        let lines = compose(&s, 0, false);
        assert!(lines.len() <= rows - RESERVE_ROWS, "region exceeds viewport");
        let body = strip_ansi(&lines.join("\n"));
        assert!(body.contains("d50"), "active row d50 not surfaced");
        assert!(body.contains("d51"), "active row d51 not surfaced");
        assert!(body.contains("more"), "hidden remainder not counted");
        assert!(strip_ansi(&lines[0]).contains("2 in flight"));
    }

    #[test]
    fn done_and_total_track_status() {
        let s = state(vec![done("a"), done("b"), pending("c")], 40);
        let lines = compose(&s, 0, false);
        assert!(strip_ansi(&lines[0]).contains("2/3"));
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
        assert_eq!(out.chars().count(), 34, "elided name must fill exactly the column");
        assert!(out.starts_with("github.com/"), "host prefix lost: {out}");
        assert!(out.ends_with(".zip"), "filename tail lost: {out}");
        assert!(out.contains('\u{2026}'), "no ellipsis marking the cut: {out}");
    }
}
