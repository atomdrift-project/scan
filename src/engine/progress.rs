//! Terminal progress: the multi-file bar, the single-artifact spinner, and
//! the hook other stderr writers use to print above them.

use std::collections::HashSet;
use std::io::{IsTerminal as _, Write as _};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::output::{Rgb, fg};

pub(crate) const SPINNER: &[char] = &[
    '\u{2800}', '\u{2801}', '\u{2809}', '\u{2819}', '\u{281B}', '\u{281E}', '\u{2816}', '\u{2812}',
    '\u{2810}', '\u{2800}',
];

/// Heartbeat redraw cadence. A background thread redraws on this interval so
/// the spinner keeps animating — and the long-tail notice can appear — even
/// while no file completes (a single slow rizin analysis can stall for minutes).
/// Shared with [`crate::deptree`], whose live tree animates on the same cadence.
pub(crate) const PROGRESS_TICK: Duration = Duration::from_millis(125);

/// Files-remaining threshold below which a stall counts as the "long tail".
const TAIL_FILES: u32 = 4;

/// How long the tail must sit without advancing before we reassure the user
/// that the scan is grinding, not hung.
const TAIL_STALL: Duration = Duration::from_millis(150);

/// The long-tail reassurance text (plain; colorized at render time).
const TAIL_MESSAGE: &str =
    "\u{2014} on the final long tail of difficult reverse engineering; please be patient\u{2026}";

/// Return to column 0 and erase the line: how the bar and the spinner clear
/// themselves.
const ERASE_LINE: &str = "\r\x1b[2K";
/// Erase to the end of the line — clears a wider previous frame (a longer ETA,
/// or the notice once it's gone) without padding.
const ERASE_TO_EOL: &str = "\x1b[K";

const ACCENT: Rgb = Rgb(100, 180, 255);
const BAR: Rgb = Rgb(80, 160, 220);
const TRACK: Rgb = Rgb(50, 50, 50);
const STATS: Rgb = Rgb(160, 160, 160);
const NOTE: Rgb = Rgb(120, 120, 120);
const FAINT: Rgb = Rgb(80, 80, 80);

/// Write a redraw to stderr. Progress is cosmetic: a failed redraw must not
/// stop the scan, so its write errors are ignored — here, and nowhere else.
fn draw_stderr(text: &str) {
    let mut stderr = std::io::stderr().lock();
    let _ = stderr
        .write_all(text.as_bytes())
        .and_then(|()| stderr.flush());
}

/// Lock a progress mutex. Both guard drawing state only, so a holder that
/// panicked leaves nothing to repair: carry on with the inner value.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Build the dim, space-prefixed notice that fits within `budget` visible
/// columns, truncating with an ellipsis. Empty when there isn't room for a
/// meaningful slice (a very narrow terminal) — the bar alone still renders.
fn fit_notice(msg: &str, budget: usize) -> String {
    const MIN: usize = 12;
    if budget < MIN {
        return String::new();
    }
    let text = if msg.chars().count() <= budget {
        msg.to_string()
    } else {
        let mut t: String = msg.chars().take(budget - 1).collect();
        t.push('\u{2026}');
        t
    };
    format!(" {}", fg(NOTE, &text))
}

/// Terminal size as `(cols, rows)`, for capping the progress line and bounding
/// the live dependency tree. Falls back to `(80, 24)` when the size can't be
/// queried (not a tty, or the ioctl fails).
pub(crate) fn term_dims() -> (usize, usize) {
    #[cfg(unix)]
    {
        // SAFETY: TIOCGWINSZ fills a zero-initialised `winsize`; we trust the
        // result only when the ioctl reports success and a non-zero width.
        unsafe {
            let mut ws: libc::winsize = std::mem::zeroed();
            if libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
                let rows = if ws.ws_row > 0 {
                    ws.ws_row as usize
                } else {
                    24
                };
                return (ws.ws_col as usize, rows);
            }
        }
    }
    (80, 24)
}

/// Shared progress state. Lives behind an `Arc` so the heartbeat thread can hold
/// a clone independent of the `Progress` handle the scan loop borrows.
struct Inner {
    analyzed: AtomicU32,
    external_dependencies: AtomicU32,
    external_urls: AtomicU32,
    total: u32,
    start: Instant,
    /// Elapsed millis at the most recent `increment`; lets `render` measure how
    /// long the count has been frozen.
    last_advance_ms: AtomicU64,
    /// Spinner frame counter, advanced once per render so the spinner animates
    /// on the heartbeat regardless of completions.
    tick: AtomicU32,
    /// Terminal width (columns), sampled once at construction. Caps the
    /// long-tail notice so the bar line never wraps — a wrapped line couldn't be
    /// erased in place by the redraw/clear.
    term_cols: usize,
    /// Set when the scan is done: the heartbeat thread exits and no further
    /// render runs, so a late redraw can never clobber the final summary line.
    stopped: AtomicBool,
    /// Serialises renders from the rayon workers and the heartbeat thread; their
    /// `\r`-prefixed writes must never interleave.
    draw_lock: Mutex<()>,
}

/// The active terminal progress bar, published so incidental stderr writers —
/// chiefly actionable fetch failures/skips — can print *above* the bar instead
/// of grafting onto or racing its `\r`-parked line. Only the single-process
/// terminal CLI ever installs a bar; server/JSON modes run none and leave this
/// `None`. A `Weak` so a finished bar is never kept alive.
static ACTIVE_BAR: Mutex<Option<Weak<Inner>>> = Mutex::new(None);

/// The live bar, if any. The registry lock is released on return, so a caller
/// can take the bar's `draw_lock` without holding both (no ordering cycle).
fn active_bar() -> Option<Arc<Inner>> {
    lock(&ACTIVE_BAR).as_ref().and_then(Weak::upgrade)
}

/// Whether an interactive scan progress bar currently owns the terminal. The
/// live dependency tree ([`crate::deptree`]) defers to it: a multi-file scan's
/// bar owns external-fetch status, so only a single-artifact scan — where no bar
/// is live — takes over stderr with an in-place tree.
pub(crate) fn bar_active() -> bool {
    active_bar().is_some()
}

/// Add one file's active external-reference work to the main scan bar. The
/// counters are process-wide only while that bar is alive; concurrent files
/// contribute to the same compact status note.
pub(crate) fn external_fetch_started(dependencies: usize, urls: usize) {
    adjust_external(dependencies, urls, u32::saturating_add);
}

/// Remove one file's external-reference work from the main scan bar.
pub(crate) fn external_fetch_finished(dependencies: usize, urls: usize) {
    adjust_external(dependencies, urls, u32::saturating_sub);
}

fn adjust_external(dependencies: usize, urls: usize, op: fn(u32, u32) -> u32) {
    let Some(inner) = active_bar() else { return };
    for (counter, n) in [
        (&inner.external_dependencies, dependencies),
        (&inner.external_urls, urls),
    ] {
        let n = u32::try_from(n).unwrap_or(u32::MAX);
        counter
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                Some(op(count, n))
            })
            .ok();
    }
}

/// Run `print` (which writes one or more *complete* newline-terminated lines to
/// stderr) so its output lands cleanly above the progress bar rather than
/// interleaving with it. When a live bar is active, the bar line is erased and
/// `print` runs under the bar's `draw_lock`, so neither the heartbeat thread nor
/// a parallel worker can repaint the bar between the erase and the write; the
/// bar redraws itself on the next `increment`/heartbeat tick, beneath the lines
/// just printed. With no active bar (server/JSON modes, or after the scan ends)
/// the closure simply runs. Callers must not already hold `draw_lock`.
pub(crate) fn print_above_bar(print: impl FnOnce()) {
    let Some(inner) = active_bar() else {
        print();
        return;
    };
    let _guard = lock(&inner.draw_lock);
    // Erase the bar only if it's still live; once stopped, `finish` already
    // cleared the line and the cursor is at a fresh column 0.
    if !inner.stopped.load(Ordering::Relaxed) {
        draw_stderr(ERASE_LINE);
    }
    print();
}

impl Inner {
    fn draw(&self) {
        let _guard = lock(&self.draw_lock);
        if self.stopped.load(Ordering::Relaxed) {
            return;
        }
        self.render();
    }

    fn render(&self) {
        let done = self.analyzed.load(Ordering::Relaxed);
        let external_dependencies = self.external_dependencies.load(Ordering::Relaxed);
        let external_urls = self.external_urls.load(Ordering::Relaxed);
        let elapsed = self.start.elapsed();

        let frame = SPINNER[self.tick.fetch_add(1, Ordering::Relaxed) as usize % SPINNER.len()];
        let bar_w = 20;
        let filled = (done as usize * bar_w / self.total.max(1) as usize).min(bar_w);
        let bar: String = (0..bar_w)
            .map(|i| {
                if i < filled {
                    '\u{2501}' // ━
                } else if i == filled {
                    '\u{2578}' // ╸
                } else {
                    '\u{2500}' // ─
                }
            })
            .collect();

        let filled_str: String = bar.chars().take(filled + 1).collect();
        let dim_str: String = bar.chars().skip(filled + 1).collect();

        let stats = progress_stats(done, self.total, elapsed);

        // Long-tail reassurance: when only a few files remain and the count has
        // not moved for a while, the scan is almost certainly deep in a slow
        // reverse-engineering pass, not hung. Appended to the bar line (not a
        // separate line) so the bar's own clear erases it — the note shows while
        // the tail is stalled and vanishes the instant a file completes or the
        // scan ends. Capped to the terminal width so it can never wrap.
        let left = self.total.saturating_sub(done);
        let stalled = elapsed.saturating_sub(Duration::from_millis(
            self.last_advance_ms.load(Ordering::Relaxed),
        ));
        let note_text = external_fetch_note(external_dependencies, external_urls).or_else(|| {
            ((1..=TAIL_FILES).contains(&left) && stalled >= TAIL_STALL)
                .then(|| TAIL_MESSAGE.to_string())
        });
        let used = 25 + stats.chars().count();
        let note = note_text.map_or_else(String::new, |text| {
            fit_notice(&text, self.term_cols.saturating_sub(used + 1))
        });

        draw_stderr(&format!(
            "\r {} {}{}  {}{note}{ERASE_TO_EOL}",
            fg(ACCENT, frame.encode_utf8(&mut [0; 4])),
            fg(BAR, &filled_str),
            fg(TRACK, &dim_str),
            fg(STATS, &stats),
        ));
    }

    /// Halt the heartbeat and wait out any in-flight render, so the caller can
    /// write a final line the ticker can no longer overwrite.
    fn quiesce(&self) {
        self.stopped.store(true, Ordering::Relaxed);
        drop(lock(&self.draw_lock));
    }
}

/// Stable right-hand progress text. There is no rate before the first file
/// completes, so keep that otherwise-nonsensical initial ETA qualitative.
fn progress_stats(done: u32, total: u32, elapsed: Duration) -> String {
    if done == 0 {
        return format!("{done}/{total}  Estimating\u{2026}");
    }

    let rate = f64::from(done) / elapsed.as_secs_f64().max(0.001);
    let eta = f64::from(total.saturating_sub(done)) / rate.max(0.001);
    format!("{done}/{total}  {rate:.0}/s  {}", format_eta(eta))
}

/// Compact status text for external work currently shared by the scan's
/// concurrent workers. Keep it short enough to coexist with the file counter.
fn external_fetch_note(dependencies: u32, urls: u32) -> Option<String> {
    fn noun(count: u32, singular: &str, plural: &str) -> String {
        format!("{count} {}", if count == 1 { singular } else { plural })
    }

    match (dependencies, urls) {
        (0, 0) => None,
        (0, urls) => Some(format!("fetching {}", noun(urls, "URL", "URLs"))),
        (dependencies, 0) => Some(format!(
            "fetching {}",
            noun(dependencies, "dependency", "dependencies")
        )),
        (dependencies, urls) => Some(format!(
            "fetching {} · {}",
            noun(dependencies, "dependency", "dependencies"),
            noun(urls, "URL", "URLs")
        )),
    }
}

/// A single-artifact scan has no file-count denominator to fill a progress bar,
/// but the analysis of one archive can still take many seconds (extraction,
/// disassembly, per-member scoring) with nothing on screen. This is a minimal
/// animated spinner for that gap — `⠙ scanning demo.zip · 12s` redrawn in place
/// on stderr — so the scan visibly *works* rather than appearing to hang. It is
/// deliberately not a [`Progress`] bar and never registers in `ACTIVE_BAR`, so
/// the live dependency tree still takes over stderr during the fetch phase.
pub(crate) struct Spinner {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Spinner {
    /// Start spinning next to `label`, or `None` when stderr isn't a terminal
    /// (piped/redirected output draws nothing, matching the progress bar).
    pub(crate) fn start(label: String) -> Option<Self> {
        if !std::io::stderr().is_terminal() {
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let started = Instant::now();
        let thread = std::thread::Builder::new()
            .name("scan-spinner".into())
            .spawn(move || {
                let mut tick = 0usize;
                // The distinct archive members seen entering analysis. cleave
                // fans members across the rayon pool and names each on a
                // per-thread breadcrumb; sampling those each tick lets us show a
                // live count and the member currently in hand — real progress
                // through a deeply nested archive, where one file expands to
                // thousands of members with nothing else to count. It undercounts
                // members that begin and finish between two samples, so the count
                // is prefixed `~`.
                let mut seen: HashSet<String> = HashSet::new();
                let mut current = String::new();
                while !flag.load(Ordering::Relaxed) {
                    let frame = SPINNER[tick % SPINNER.len()];
                    let secs = started.elapsed().as_secs();
                    // One sample of the per-thread breadcrumbs: grow the distinct
                    // member set (the live count) and show the oldest in-flight
                    // member (most likely the slow one holding up the scan).
                    let members = cleave::breadcrumb::snapshot();
                    for crumb in &members {
                        if crumb.analyzer == "member" {
                            seen.insert(crumb.target.clone());
                        }
                    }
                    if let Some(oldest) = members.iter().find(|c| c.analyzer == "member") {
                        current.clone_from(&oldest.target);
                    }
                    let detail = if seen.is_empty() {
                        String::new()
                    } else {
                        format!(
                            "  {}  {}",
                            fg(NOTE, &format!("~{} members", seen.len())),
                            fg(FAINT, &spinner_tail(&current, 48)),
                        )
                    };
                    draw_stderr(&format!(
                        "{ERASE_LINE} {}  {}{detail}  {}",
                        fg(ACCENT, frame.encode_utf8(&mut [0; 4])),
                        fg(STATS, &format!("scanning {label}")),
                        fg(FAINT, &format!("{secs}s")),
                    ));
                    tick += 1;
                    // `drop` unparks this thread, so stopping costs no tick.
                    std::thread::park_timeout(PROGRESS_TICK);
                }
            })
            .ok()?;
        Some(Self {
            stop,
            thread: Some(thread),
        })
    }
}

/// Keep the last `width` characters of a member path — its filename and nearest
/// directories, the telling part — eliding the head with a leading `…`. A short
/// path is returned unchanged.
fn spinner_tail(path: &str, width: usize) -> String {
    let count = path.chars().count();
    if count <= width {
        return path.to_string();
    }
    let tail: String = path.chars().skip(count - width + 1).collect();
    format!("\u{2026}{tail}")
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            // A spinner that panicked has nothing left to draw, and the scan
            // must not fail over its animation.
            let _ = thread.join();
        }
        // Erase the spinner line so the next output starts clean.
        draw_stderr(ERASE_LINE);
    }
}

/// Progress bar handle held by the scan loop. Dropping it (or calling `finish`
/// / `quiesce`) stops the background heartbeat thread; dropping it joins it.
pub(crate) struct Progress {
    inner: Arc<Inner>,
    heartbeat: Option<JoinHandle<()>>,
}

impl Progress {
    pub(crate) fn new(total: u32) -> Self {
        let inner = Arc::new(Inner {
            analyzed: AtomicU32::new(0),
            external_dependencies: AtomicU32::new(0),
            external_urls: AtomicU32::new(0),
            total,
            start: Instant::now(),
            last_advance_ms: AtomicU64::new(0),
            tick: AtomicU32::new(0),
            term_cols: term_dims().0,
            stopped: AtomicBool::new(false),
            draw_lock: Mutex::new(()),
        });
        // Publish this bar so fetch status can join it and actionable failures
        // can print above it (see `print_above_bar`). The terminal CLI runs one
        // bar at a time; this one unpublishes itself on drop.
        *lock(&ACTIVE_BAR) = Some(Arc::downgrade(&inner));
        // A spawn failure costs only the heartbeat: the per-file redraws still
        // drive the bar.
        let beat = Arc::clone(&inner);
        let heartbeat = std::thread::Builder::new()
            .name("scan-progress".into())
            .spawn(move || {
                loop {
                    std::thread::park_timeout(PROGRESS_TICK);
                    if beat.stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    beat.draw();
                }
            })
            .ok();
        Self { inner, heartbeat }
    }

    pub(crate) fn increment(&self) {
        self.inner.analyzed.fetch_add(1, Ordering::Relaxed);
        let ms = u64::try_from(self.inner.start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.inner.last_advance_ms.store(ms, Ordering::Relaxed);
        self.inner.draw();
    }

    /// Erase the bar, run `print` (which writes a file's result to stdout), then
    /// redraw the bar beneath it — all while holding the draw lock so the
    /// heartbeat thread cannot repaint the bar between the erase and the write.
    /// Without the leading erase the result is grafted onto the bar line the
    /// last render left on screen (cursor parked at its end, no newline).
    pub(crate) fn around_result<T>(&self, print: impl FnOnce() -> T) -> T {
        let inner = &self.inner;
        let _guard = lock(&inner.draw_lock);
        draw_stderr(ERASE_LINE);
        let printed = print();
        if !inner.stopped.load(Ordering::Relaxed) {
            inner.render();
        }
        printed
    }

    /// Stop the heartbeat without printing anything, for callers that render
    /// their own final line in the bar's place (e.g. `ps`).
    pub(crate) fn quiesce(&self) {
        self.inner.quiesce();
        if let Some(heartbeat) = &self.heartbeat {
            heartbeat.thread().unpark();
        }
    }

    /// Stop the heartbeat and erase the progress bar, leaving the cursor at the
    /// start of its now-blank line so the caller can write the closing summary in
    /// its place. The bar deliberately prints no completion line of its own — the
    /// summary is the single end statement (one verdict, one duration), so a
    /// separate "N files in Xs" line here would only repeat it.
    pub(crate) fn finish(&self) {
        self.quiesce();
        draw_stderr(ERASE_LINE);
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        // Safety net: stop the heartbeat even on an early return or error that
        // skipped `finish`/`quiesce`.
        self.inner.stopped.store(true, Ordering::Relaxed);
        if let Some(heartbeat) = self.heartbeat.take() {
            heartbeat.thread().unpark();
            // A heartbeat that panicked only stopped redrawing; nothing to recover.
            let _ = heartbeat.join();
        }
        // Unpublish, so a later `print_above_bar` doesn't erase a bar that no
        // longer owns the terminal — but only if the published bar is this one.
        let mut active = lock(&ACTIVE_BAR);
        if active
            .as_ref()
            .is_some_and(|bar| bar.as_ptr() == Arc::as_ptr(&self.inner))
        {
            *active = None;
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "`secs` is a positive, finite ETA; minutes fit a u32 for any real scan"
)]
fn format_eta(secs: f64) -> String {
    if secs < 1.0 {
        "<1s".to_string()
    } else if secs < 60.0 {
        format!("~{:.0}s", secs)
    } else {
        format!("~{}m{:.0}s", (secs / 60.0) as u32, secs % 60.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_waits_for_a_real_sample() {
        assert_eq!(
            progress_stats(0, 100, Duration::from_secs(30)),
            "0/100  Estimating…",
            "elapsed time alone cannot produce a rate"
        );
    }

    #[test]
    fn eta_appears_after_the_first_completed_file() {
        let stats = progress_stats(1, 100, Duration::from_millis(500));
        assert!(!stats.contains("Estimating"));
        assert!(stats.contains("/s"));
    }

    /// Dropping an older bar must not unpublish the newer one.
    #[test]
    fn a_dropped_bar_unpublishes_only_itself() {
        let first = Progress::new(2);
        let second = Progress::new(2);
        drop(first);
        assert!(bar_active(), "the newer bar is still the live one");
        drop(second);
        assert!(!bar_active());
    }

    #[test]
    fn spinner_tail_keeps_the_filename_end() {
        assert_eq!(spinner_tail("short.py", 48), "short.py");
        let long = "animica/stratum_pool/_data/aicf_rag/chunks/deeply/nested/file.json";
        let tail = spinner_tail(long, 20);
        assert!(tail.starts_with('\u{2026}'));
        assert!(tail.ends_with("file.json"));
        assert_eq!(tail.chars().count(), 20);
    }

    #[test]
    fn external_fetch_note_stays_compact_and_grammatical() {
        assert_eq!(external_fetch_note(0, 0), None);
        assert_eq!(
            external_fetch_note(1, 1).as_deref(),
            Some("fetching 1 dependency · 1 URL")
        );
        assert_eq!(
            external_fetch_note(0, 2).as_deref(),
            Some("fetching 2 URLs")
        );
    }
}
