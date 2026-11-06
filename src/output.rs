//! Terminal and JSON output formatting: the colour palette, the card and
//! summary renderers, and the helpers every terminal writer shares. Supports
//! dark and light terminal backgrounds.

use std::fmt::Write as _;
use std::io::{IsTerminal, Write as _};
use std::sync::{LazyLock, OnceLock};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::OutputFormat;
pub use crate::civil::parse_ymd;
use crate::civil::{SECS_PER_DAY, SECS_PER_HOUR, SECS_PER_MINUTE};
use crate::engine::{ScanResult, ScanSummary, level_confidence};
use crate::model::{Classification, Level, RouteScore};

impl OutputFormat {
    /// Whether every scanned file is written, whatever `--show` says. JSON,
    /// tiny and interpret are payloads for programs and LLMs, which need the
    /// complete stream; `--show` filters only what a person reads.
    #[must_use]
    pub(crate) const fn emits_every_result(self) -> bool {
        !matches!(self, Self::Terminal)
    }
}

/// Return to column 0 and erase the line: how the progress bar, the spinner
/// and the dependency tree clear what they drew.
pub(crate) const ERASE_LINE: &str = "\r\x1b[2K";

/// Whether `c` can rewrite or disguise what a terminal shows: a C0/C1
/// control (ESC opens CSI/OSC sequences that move the cursor, erase lines,
/// write the clipboard or plant links) or a bidi override/isolate, which
/// reorders the text around it.
pub(crate) fn tty_hostile(c: char) -> bool {
    c.is_control() || matches!(c, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
}

/// `text` safe to print to a terminal: every [`tty_hostile`] character is
/// shown as its `\u{…}` escape instead of being interpreted. File names,
/// paths, package names and URLs come from hostile samples; a member named
/// `a\x1b[1A\x1b[2K✓ clean.js` must not erase the verdict printed above it.
pub(crate) fn tty_text(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.chars().any(tty_hostile) {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        if tty_hostile(c) {
            out.extend(c.escape_unicode());
        } else {
            out.push(c);
        }
    }
    std::borrow::Cow::Owned(out)
}

/// Which part of a too-wide text [`elide`] keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Keep {
    /// The start: prose, where the opening words carry the meaning.
    Head,
    /// Both ends around a central cut: a URL or package path, whose host and
    /// file name are both telling.
    Ends,
    /// The end: a member path, whose file name is the telling part.
    Tail,
}

/// `text` fitted to `width` terminal columns, an ellipsis marking the cut.
/// Measured in display columns, so wide characters count double and a fitted
/// line never wraps. Returned whole when it already fits.
pub(crate) fn elide(text: &str, width: usize, keep: Keep) -> String {
    if text.width() <= width {
        return text.to_owned();
    }
    let Some(budget) = width.checked_sub(1) else {
        return String::new();
    };
    let (head, tail) = match keep {
        Keep::Head => (budget, 0),
        Keep::Tail => (0, budget),
        Keep::Ends => (budget / 2, budget - budget / 2),
    };
    /// The leading chars of `chars` that fit in `columns`.
    fn fit(chars: impl Iterator<Item = char>, columns: usize) -> Vec<char> {
        let mut used = 0;
        chars
            .take_while(|c| {
                used += c.width().unwrap_or(0);
                used <= columns
            })
            .collect()
    }
    let mut out: String = fit(text.chars(), head).into_iter().collect();
    out.push('\u{2026}');
    out.extend(fit(text.chars().rev(), tail).into_iter().rev());
    out
}

/// Remove ANSI CSI escape sequences (`ESC [ … <final>`) and any other lone
/// `ESC`. Only the sequence itself goes: text after a non-SGR sequence
/// (`ESC [2J`) is kept, not eaten up to the next `m`.
pub(crate) fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        // Skip a CSI sequence: ESC '[' params... final byte in '@'..='~'.
        if chars.clone().next() == Some('[') {
            chars.next();
            for n in chars.by_ref() {
                if ('@'..='~').contains(&n) {
                    break;
                }
            }
        }
    }
    out
}

const TERMINAL_MARKER_WIDTH: usize = 3;

/// Prefix a card row with one fixed-width marker rail and one-cell gutter.
/// Emoji are generally two terminal cells while severity dots are one cell
/// each; measuring display width keeps their following text on one column.
fn terminal_marker_prefix(indent: usize, marker: &str) -> String {
    terminal_marker_prefix_with_width(indent, marker, UnicodeWidthStr::width(marker))
}

fn terminal_marker_prefix_with_width(indent: usize, marker: &str, marker_width: usize) -> String {
    format!(
        "{}{marker}{} ",
        " ".repeat(indent),
        " ".repeat(TERMINAL_MARKER_WIDTH.saturating_sub(marker_width)),
    )
}

/// RGB color tuple.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Rgb(pub(crate) u8, pub(crate) u8, pub(crate) u8);

/// Terminal background theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    /// Dark terminal background (default).
    Dark,
    /// Light/white terminal background.
    Light,
}

/// Color palette tuned for a specific background theme. Every coloured
/// terminal line takes its colours from here, so a theme is one table.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Palette {
    // Classification colors — high-vis on both backgrounds.
    pub(crate) hostile: Rgb,
    pub(crate) suspicious: Rgb,
    pub(crate) benign: Rgb,

    // Live activity: the progress bar, spinners, in-flight fetches.
    pub(crate) live: Rgb,
    pub(crate) bar: Rgb,

    // UI chrome.
    pub(crate) dot_sep: Rgb,
    pub(crate) label: Rgb,
    pub(crate) dim: Rgb,
    pub(crate) very_dim: Rgb,
    pub(crate) path_name: Rgb,
    pub(crate) header_path: Rgb,
    pub(crate) summary_line: Rgb,
    pub(crate) warning: Rgb,
    pub(crate) reason: Rgb,
}

impl Palette {
    const fn dark() -> Self {
        Self {
            hostile: Rgb(255, 70, 70),
            suspicious: Rgb(255, 175, 55),
            benign: Rgb(80, 200, 80),

            live: Rgb(100, 180, 255),
            bar: Rgb(80, 160, 220),

            dot_sep: Rgb(50, 50, 50),
            label: Rgb(160, 160, 160),
            dim: Rgb(100, 100, 100),
            very_dim: Rgb(80, 80, 80),
            path_name: Rgb(230, 230, 230),
            header_path: Rgb(180, 180, 180),
            summary_line: Rgb(50, 50, 50),
            warning: Rgb(180, 180, 60),
            reason: Rgb(100, 100, 100),
        }
    }

    const fn light() -> Self {
        Self {
            hostile: Rgb(200, 30, 30),
            suspicious: Rgb(180, 120, 0),
            benign: Rgb(30, 140, 30),

            live: Rgb(0, 110, 200),
            bar: Rgb(30, 120, 190),

            dot_sep: Rgb(180, 180, 180),
            label: Rgb(90, 90, 90),
            dim: Rgb(120, 120, 120),
            very_dim: Rgb(150, 150, 150),
            path_name: Rgb(30, 30, 30),
            header_path: Rgb(80, 80, 80),
            summary_line: Rgb(190, 190, 190),
            warning: Rgb(140, 140, 0),
            reason: Rgb(100, 100, 100),
        }
    }
}

/// The terminal's own theme, asked once and remembered: its answer, else
/// dark. Only ask on a terminal that answers; the query blocks on one that
/// does not.
pub fn detect_theme() -> Theme {
    static DETECTED: OnceLock<Theme> = OnceLock::new();
    *DETECTED.get_or_init(|| {
        // Terminal queries must only be attempted on a real TTY. Without one,
        // the read blocks indefinitely or the kernel sends SIGTTIN.
        if std::io::stderr().is_terminal()
            && let Ok(terminal_colorsaurus::ColorScheme::Light) =
                terminal_colorsaurus::color_scheme(terminal_colorsaurus::QueryOptions::default())
        {
            Theme::Light
        } else {
            Theme::Dark
        }
    })
}

/// The active theme's palette; see [`crate::settings::Settings::theme`].
pub(crate) fn palette() -> &'static Palette {
    static DARK: Palette = Palette::dark();
    static LIGHT: Palette = Palette::light();

    match crate::settings::get().theme.unwrap_or(Theme::Dark) {
        Theme::Dark => &DARK,
        Theme::Light => &LIGHT,
    }
}

#[cfg(test)]
thread_local! {
    /// Set by [`PlainOutput`]: this thread renders without colour.
    static PLAIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Renders on this thread without colour while held, whatever the
/// environment says, so a test asserts on the plain text under test even on a
/// host that sets `CLICOLOR_FORCE`. Per thread, so parallel tests are
/// untouched; dropping it restores the environment's decision.
#[cfg(test)]
struct PlainOutput(());

#[cfg(test)]
impl PlainOutput {
    pub(crate) fn new() -> Self {
        PLAIN.set(true);
        Self(())
    }
}

#[cfg(test)]
impl Drop for PlainOutput {
    fn drop(&mut self) {
        PLAIN.set(false);
    }
}

/// Whether stdout, where results go, may carry ANSI colour. Decided as the
/// `colored` crate decides it, which is also how cleave paints the context it
/// renders: `CLICOLOR_FORCE` forces colour on, `NO_COLOR` forces it off, and
/// otherwise stdout must be a terminal and `CLICOLOR` not `0`.
pub(crate) fn color_enabled() -> bool {
    #[cfg(test)]
    if PLAIN.get() {
        return false;
    }
    colored::control::SHOULD_COLORIZE.should_colorize()
}

/// Paints text for one output stream: truecolor when that stream takes
/// colour, plain otherwise.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ink {
    color: bool,
}

impl Ink {
    /// For stdout, where result cards go.
    pub(crate) fn stdout() -> Self {
        Self {
            color: color_enabled(),
        }
    }

    /// For stderr, where progress, banners, warnings and summaries go: colour
    /// only when [`color_enabled`] allows it at all — `NO_COLOR` and
    /// `CLICOLOR=0` hold for both streams — and stderr is itself a terminal,
    /// so `atomscan … 2> log` writes no escapes into the log.
    pub(crate) fn stderr() -> Self {
        static TERMINAL: LazyLock<bool> = LazyLock::new(|| std::io::stderr().is_terminal());
        Self {
            color: color_enabled() && *TERMINAL,
        }
    }

    /// Never coloured: text for programs and LLMs.
    pub(crate) const fn plain() -> Self {
        Self { color: false }
    }

    /// Whether this stream takes colour.
    pub(crate) const fn on(self) -> bool {
        self.color
    }

    /// `text` in truecolor foreground, or plain.
    pub(crate) fn fg(self, Rgb(r, g, b): Rgb, text: &str) -> String {
        if self.color {
            format!("\x1b[38;2;{r};{g};{b}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    }

    /// `text` in bold truecolor foreground, or plain.
    pub(crate) fn fg_bold(self, Rgb(r, g, b): Rgb, text: &str) -> String {
        if self.color {
            format!("\x1b[1;38;2;{r};{g};{b}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    }
}

/// `text` in truecolor foreground for stdout, or plain when stdout takes no
/// colour.
pub(crate) fn fg(rgb: Rgb, text: &str) -> String {
    Ink::stdout().fg(rgb, text)
}

/// `text` in bold truecolor foreground for stdout, or plain when stdout takes
/// no colour.
pub(crate) fn fg_bold(rgb: Rgb, text: &str) -> String {
    Ink::stdout().fg_bold(rgb, text)
}

/// `text` as a stamp: bold white on a solid `background`. Colour mode only.
fn stamp(Rgb(r, g, b): Rgb, text: &str) -> String {
    format!("\x1b[1;38;2;255;255;255;48;2;{r};{g};{b}m{text}\x1b[0m")
}

/// Rescale raw model probability for display.
///
/// `threshold` is the deciding cutoff (the value `raw` was compared against).
/// Below the cutoff is mapped into the bottom 50%, above it into the top 50%.
fn display_probability(raw: f32, threshold: f32) -> f32 {
    if raw <= threshold {
        // [0, threshold] → [0%, 50%]
        (raw / threshold.max(f32::EPSILON)) * 0.5
    } else {
        // [threshold, 1.0] → [50%, 100%]
        0.5 + ((raw - threshold) / (1.0 - threshold).max(f32::EPSILON)) * 0.5
    }
}

/// Level-derived confidence for DISPLAY, or `None` when the level carries no
/// confidence worth showing.
///
/// [`level_confidence`] maps [`Level::Clean`] — "cleared no calibrated level
/// at all" — onto `Some(0)`. That is a defensible wire value, but rendering it
/// prints ` BENIGN    0% `, which reads as *maximum* confidence in benign. The
/// model never said that. It said the sample cleared no calibrated level,
/// which is the same state as carrying no level marker: unknown, not certain.
/// A file just under a strict cutoff would otherwise read `BENIGN[0%]` above
/// cleave's own hostile finding for it. Returning `None` routes it through the
/// no-level path — the threshold-relative percentage — so the badge shows how
/// near the cutoff the sample sat.
///
/// Display-only by construction: the JSON `conf` field keeps calling
/// [`level_confidence`], so the machine-readable shape is unchanged.
fn display_confidence(level: Level) -> Option<u8> {
    match level {
        Level::Clean => None,
        other => level_confidence(other),
    }
}

/// Fixed per-class anchor color: red hostile, yellow suspicious, green benign.
/// The verdict stamp and card rules use this — the word names the verdict, so
/// the hue must match it exactly; confidence nuance lives in the percentage.
pub(crate) fn class_color(classification: Classification) -> Rgb {
    let p = palette();
    match classification {
        Classification::Hostile => p.hostile,
        Classification::Suspicious => p.suspicious,
        Classification::Benign => p.benign,
    }
}

/// The verdict word stamped on cards and badges.
const fn verdict_word(classification: Classification) -> &'static str {
    match classification {
        Classification::Hostile => "HOSTILE",
        Classification::Suspicious => "SUSPECT",
        Classification::Benign => "BENIGN",
    }
}

/// The card's marker rail — `marker` in its fixed-width column — when color
/// is on; `plain` (grep-able text, no glyph alignment) when it is off.
fn rail(indent: usize, marker: &str, plain: &str) -> String {
    if color_enabled() {
        terminal_marker_prefix(indent, marker)
    } else {
        plain.to_string()
    }
}

/// The scan's verdict stamp that leads cleave's headline: the verdict word and
/// band percentage on a filled, classification-colored background —
/// ` HOSTILE 100% ` (red), ` SUSPECT  62% ` (orange), ` BENIGN    3% `
/// (green) — so the verdict reads in both the color and the word. Every stamp
/// is the same [`BADGE_WIDTH`] columns (7-char word slot, right-aligned
/// percentage), so verdicts line up down a scan listing. Plain
/// `HOSTILE[100%]` when color is disabled (piped). Returns the stamp and its
/// visible width (for aligning lines beneath it).
pub(crate) fn terminal_badge(
    classification: Classification,
    probability: f32,
    threshold: f32,
    level: Level,
) -> (String, usize) {
    // ` ` + 7-char word + ` ` + 4-char percentage + ` `.
    const BADGE_WIDTH: usize = 14;
    let pct = display_confidence(level).map_or_else(
        || {
            format!(
                "{:.0}%",
                display_probability(probability, threshold) * 100.0
            )
        },
        |conf| format!("{conf}%"),
    );
    let word = verdict_word(classification);
    if !color_enabled() {
        let stamp = format!("{word}[{pct}]");
        let width = stamp.chars().count();
        return (stamp, width);
    }
    // Bold white on the class anchor color (fixed red/yellow/green — the word
    // names the verdict, so the hue must agree with it; the percentage carries
    // the confidence), fixed-width so every stamp is the same size block.
    (
        stamp(
            class_color(classification),
            &format!(" {word:<7} {pct:>4} "),
        ),
        BADGE_WIDTH,
    )
}

/// One aligned model-summary row: ` ·   <text>`.
/// built from the SHAP reasons. `None` when there are none. When an LLM
/// interpretation exists it supersedes this entirely — rendered prominent on
/// its own line by [`terminal_interpretation`] instead of dim on the headline.
pub(crate) fn terminal_trailer(reasons: &[crate::explain::Reason]) -> Option<String> {
    if reasons.is_empty() {
        return None;
    }
    let body = reasons
        .iter()
        .take(3)
        .map(|r| r.description.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let p = palette();
    let lead = if color_enabled() {
        terminal_marker_prefix_with_width(1, &fg(p.dim, "\u{00b7}"), 1)
    } else {
        "\u{00b7} ".to_string()
    };
    Some(format!("{lead}{}", fg(p.reason, &body)))
}

/// The LLM interpretation as its own headline-grade line, sitting directly
/// above the SHA-256: `✨ <one-line interpretation>` in bold foreground (white
/// on dark terminals), the sparkle marking it as model-generated. A failed
/// interpretation renders dim instead — an error is not a finding. `None` when
/// no interpretation ran.
pub(crate) fn terminal_interpretation(
    interpretation: Option<&crate::interpret::Interpretation>,
) -> Option<String> {
    let llm = interpretation?;
    let lead = rail(1, "\u{2728}", " \u{2728} ");
    let p = palette();
    Some(match llm {
        crate::interpret::Interpretation::Failed(_) => {
            format!("{lead}{}", fg(p.very_dim, "interpretation unavailable"))
        }
        crate::interpret::Interpretation::Graded(graded) => {
            format!("{lead}{}", fg_bold(p.path_name, &graded.interpretation))
        }
    })
}

/// One concise identity claim beneath the artifact name. The ID-card glyph
/// gives this optional line the same visual gutter as artifact and hash lines.
pub(crate) fn terminal_identity_line(identity: &str) -> Option<String> {
    let identity = identity.trim();
    if identity.is_empty() {
        return None;
    }
    Some(format!(
        "{}{}",
        rail(1, "\u{1faaA}", "   "), // 🪪
        fg(palette().header_path, identity)
    ))
}

/// The registry's account of the package, beneath its identity. Carries the
/// same ⓘ glyph and the same fields the one-shot `pkg`/`url` banner prints, so
/// a package reads the same way whether it was fetched by coordinate or scanned
/// from a collected copy.
pub(crate) fn terminal_registry_line(summary: &str) -> Option<String> {
    let summary = summary.trim();
    if summary.is_empty() {
        return None;
    }
    Some(format!(
        "{}{}",
        rail(1, "\u{24d8}", "   "), // ⓘ
        fg(palette().dim, summary)
    ))
}

/// One finding in the artifact-level terminal summary.
pub(crate) struct TerminalTrait {
    pub(crate) criticality: cleave::Criticality,
    pub(crate) description: String,
    pub(crate) location: String,
}

fn severity_dots(criticality: cleave::Criticality) -> (String, usize) {
    let dots = match criticality {
        cleave::Criticality::Hostile => "●●●",
        cleave::Criticality::Suspicious => "●●",
        cleave::Criticality::Notable => "●",
        _ => "·",
    };
    let width = UnicodeWidthStr::width(dots);
    if !color_enabled() {
        return (dots.to_string(), width);
    }
    let rendered = match criticality {
        cleave::Criticality::Hostile => cleave::theme::paint_hostile(dots).to_string(),
        cleave::Criticality::Suspicious => cleave::theme::paint_suspicious(dots).to_string(),
        cleave::Criticality::Notable => cleave::theme::paint_notable(dots).to_string(),
        cleave::Criticality::Baseline => cleave::theme::paint_baseline(dots).to_string(),
        _ => cleave::theme::paint_component(dots).to_string(),
    };
    (rendered, width)
}

/// Render at most three globally-curated traits as a compact aligned grid.
/// Severity is encoded twice (dot count and color), while the evidence anchor
/// sits in a quiet right column for fast triage and copy/paste.
pub(crate) fn terminal_trait_rows(traits: &[TerminalTrait], terminal_width: usize) -> String {
    if traits.is_empty() {
        return String::new();
    }

    /// Widest a row grows: keeps the reading measure composed on very wide
    /// terminals, while a narrow one still caps it so a row never wraps.
    const MAX_ROW_WIDTH: usize = 160;
    /// Leading space + 3-cell marker rail + 1-cell gutter.
    const ROW_PREFIX_WIDTH: usize = 5;
    /// Narrowest content area that still gets a location column.
    const MIN_WIDTH_FOR_LOCATION: usize = 28;
    /// Columns the description keeps when a location column is shown.
    const MIN_DESCRIPTION_WIDTH: usize = 20;
    let width = terminal_width.min(MAX_ROW_WIDTH);
    let available = width.saturating_sub(ROW_PREFIX_WIDTH);
    let max_location = traits
        .iter()
        .map(|t| UnicodeWidthStr::width(t.location.as_str()))
        .max()
        .unwrap_or(0);
    let location_width = if available >= MIN_WIDTH_FOR_LOCATION && max_location > 0 {
        max_location
            .min(available / 2)
            .min(available.saturating_sub(MIN_DESCRIPTION_WIDTH))
    } else {
        0
    };
    let location_gap = usize::from(location_width > 0) * 2;
    let description_width = available.saturating_sub(location_width + location_gap);
    let p = palette();

    traits
        .iter()
        .take(3)
        .map(|t| {
            let description = elide(&t.description, description_width, Keep::Head);
            let (dots, dots_width) = severity_dots(t.criticality);
            let prefix = terminal_marker_prefix_with_width(1, &dots, dots_width);
            if location_width == 0 {
                return format!("{prefix}{description}");
            }
            let location = elide(&t.location, location_width, Keep::Ends);
            let padding = " ".repeat(
                description_width.saturating_sub(UnicodeWidthStr::width(description.as_str())),
            );
            let location_padding = " "
                .repeat(location_width.saturating_sub(UnicodeWidthStr::width(location.as_str())));
            let location = fg(p.dim, &location);
            format!("{prefix}{description}{padding}  {location_padding}{location}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The card's identity line: a glyph, then the full SHA-256 (copy-able for
/// lookups) in the dimmest ink on screen. The glyph carries the bloom verdict —
/// 🧬 ordinarily, 🚩 known-bad, 📡 sighted, 🏴 conflicted — because the SHA-256 is what the
/// filters matched on; a known-good-but-rescanned file keeps 🧬 with a green ✓
/// trailer. Plain mode prints a grep-able text marker instead. `None` when the
/// hash is absent.
pub(crate) fn terminal_hash_line(sha256: &str, bloom: Option<BloomMark>) -> Option<String> {
    if sha256.is_empty() {
        return None;
    }
    if !color_enabled() {
        // Plain output keeps the grep-able text marker instead of the glyph.
        let mark = bloom.map_or_else(String::new, BloomMark::marker);
        return Some(format!("{sha256}{mark}"));
    }
    let p = palette();
    let (glyph, trailer) = match bloom {
        Some(BloomMark::KnownBad) => ("\u{1f6a9}", String::new()), // 🚩
        Some(BloomMark::Sighted) => ("\u{1f4e1}", String::new()),  // 📡
        Some(BloomMark::Conflicted) => ("\u{1f3f4}", String::new()), // 🏴
        Some(BloomMark::KnownGood) => ("\u{1f9ec}", format!("  {}", fg(p.benign, "\u{2713}"))),
        None => ("\u{1f9ec}", String::new()), // 🧬
    };
    Some(format!(
        "{}{}{trailer}",
        terminal_marker_prefix(1, glyph),
        fg(p.very_dim, sha256)
    ))
}

/// Human-compact byte size for headers: `352B`, `5.4KB`, `1.2MB`. Empty for 0
/// so callers can skip the segment entirely.
pub(crate) fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    match bytes {
        0 => String::new(),
        b if b >= GB => format!("{:.1}GB", b as f64 / GB as f64),
        b if b >= MB => format!("{:.1}MB", b as f64 / MB as f64),
        b if b >= KB => format!("{:.1}KB", b as f64 / KB as f64),
        b => format!("{b}B"),
    }
}

/// The card-opening rule: ` ━━ <stamp> ━━━…`, heavy strokes in the band color
/// running the terminal width, with the fixed-width verdict stamp embedded near
/// the left edge. Color-mode only — plain output frames nothing.
pub(crate) fn terminal_rule(
    classification: Classification,
    probability: f32,
    threshold: f32,
    level: Level,
    width: usize,
) -> String {
    let (stamp, stamp_w) = terminal_badge(classification, probability, threshold, level);
    let accent = class_color(classification);
    // ` ` + `━━` + ` ` + stamp + ` ` + fill, with a 1-col right margin.
    let fill = width.saturating_sub(stamp_w + 7).max(2);
    format!(
        " {} {stamp} {}",
        fg(accent, "\u{2501}\u{2501}"),
        fg(accent, &"\u{2501}".repeat(fill)),
    )
}

/// The card's artifact line: ` 📦 name · TYPE · size` (📄 for a standalone,
/// member-less file) — bright name, dim metadata. Plain mode drops the glyph
/// so piped output stays grep-able text.
pub(crate) fn terminal_artifact_line(
    label: &str,
    file_type: &str,
    size: u64,
    container: bool,
) -> String {
    let label = tty_text(label);
    let glyph = if container {
        "\u{1f4e6}" // 📦
    } else {
        "\u{1f4c4}" // 📄
    };
    let meta = type_and_size(file_type, size);
    let p = palette();
    format!(
        "{}{} {}",
        rail(1, glyph, ""),
        fg_bold(p.path_name, &label),
        fg(p.dim, &format!("\u{00b7} {meta}")),
    )
}

/// Render one fetched artifact as a branch beneath the file that referenced it.
/// Provenance belongs in the relationship (`dependency from package.json`),
/// while URL, hash, and traits occupy one aligned child column.
pub(crate) fn terminal_reference_branch(
    verdict: Option<(Classification, f32)>,
    subject: &str,
    source: &str,
    body: &str,
    last: bool,
) -> String {
    let relationship = format!("{} from {}", tty_text(subject), tty_text(source));
    let fork = if last {
        "\u{2514}\u{2500}"
    } else {
        "\u{251c}\u{2500}"
    };
    let stem = if last { "  " } else { "\u{2502} " };
    let indent = "   ";
    let p = palette();
    let status = match verdict {
        Some((classification, probability)) => format!(
            "{} {}",
            fg_bold(class_color(classification), verdict_word(classification)),
            fg(p.dim, &format!("{:.0}%", probability * 100.0)),
        ),
        None => fg_bold(p.warning, "NOT EVALUATED"),
    };
    let mut out = format!(
        "{indent}{} {} {} {status}\n",
        fg(p.very_dim, fork),
        fg_bold(p.path_name, &relationship),
        fg(p.very_dim, "\u{00b7}"),
    );
    let stem = fg(p.very_dim, stem);
    for line in body.lines() {
        out.push_str(indent);
        out.push_str(&stem);
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Non-graded counterpart for a fetched/registry branch.
pub(crate) fn terminal_reference_status_heading(status: &str, subject: &str) -> String {
    let p = palette();
    format!(
        "  ↳ {} {} {}",
        fg_bold(p.warning, status),
        fg(p.very_dim, "·"),
        fg(p.dim, &tty_text(subject)),
    )
}

/// The fetched locator as a secondary identity line in the nested branch.
pub(crate) fn terminal_reference_locator(locator: &str) -> String {
    format!("    {}", fg_bold(palette().path_name, &tty_text(locator)))
}

/// Locator row used inside a compact fetched-reference branch.
pub(crate) fn terminal_reference_locator_row(locator: &str) -> String {
    format!(
        "{}{}",
        terminal_marker_prefix(1, "\u{1f517}"), // 🔗
        fg_bold(palette().path_name, &tty_text(locator))
    )
}

/// Hash/type/size row used inside a compact fetched-reference branch.
pub(crate) fn terminal_reference_hash_row(sha256: &str, file_type: &str, size: u64) -> String {
    let meta = type_and_size(file_type, size);
    let hash = format!(
        "{}{}",
        terminal_marker_prefix(1, "\u{1f9ec}"), // 🧬
        fg(palette().very_dim, sha256)
    );
    if meta.is_empty() {
        return hash;
    }
    format!("{hash}  {}", fg(palette().dim, &format!("\u{00b7} {meta}")))
}

/// Frame one independently-scanned package nested inside an archive as a
/// bordered card: a verdict word + name header opening a left rule that runs
/// down the package's own findings (`body`, cleave's rendered context), so a
/// grab-bag archive reads as a stack of distinct verdicts rather than one flat
/// member list. Plain (piped) mode drops the border and indents two spaces,
/// keeping every line grep-able.
pub(crate) fn terminal_card(
    classification: Classification,
    name: &str,
    file_type: &str,
    size: u64,
    body: &str,
) -> String {
    // A card leads with the class *word*, not a percentage: every package in a
    // hostile archive shares the same confidence band, so eight identical `92%`
    // stamps read as a stuck gauge. The one confidence number lives on the
    // archive banner; the cards rank by severity (worst first) and by class.
    let word = verdict_word(classification);
    let name = tty_text(name);
    let meta = type_and_size(file_type, size);
    let mut out = String::new();
    // Plain (piped) output drops the frame and indents instead, so every line
    // stays grep-able.
    if color_enabled() {
        let p = palette();
        let accent = class_color(classification);
        let bar = fg(accent, "\u{2502}"); // │
        let stamp = stamp(accent, &format!(" {word} "));
        let _ = writeln!(
            out,
            "{} {stamp} {} {}",
            fg(accent, "\u{256d}\u{2500}"), // ╭─
            fg_bold(p.path_name, &name),
            fg(p.dim, &format!("\u{00b7} {meta}")),
        );
        for line in body.lines() {
            if line.is_empty() {
                out.push_str(&bar);
                out.push('\n');
            } else {
                let _ = writeln!(out, "{bar} {line}");
            }
        }
        out.push_str(&fg(accent, "\u{2570}\u{2500}")); // ╰─
        out.push('\n');
    } else {
        let _ = writeln!(out, "{word} {name} \u{00b7} {meta}");
        for line in body.lines() {
            if line.is_empty() {
                out.push('\n');
            } else {
                let _ = writeln!(out, "  {line}");
            }
        }
    }
    out
}

/// Render one decoded child as a compact branch beneath its parent artifact.
/// Unlike a package card, a two-line embedded result does not need a closing
/// border: the branch itself communicates containment and keeps sibling
/// payloads visually connected.
pub(crate) fn terminal_embedded_branch(
    classification: Classification,
    name: &str,
    file_type: &str,
    size: u64,
    body: &str,
    last: bool,
) -> String {
    let name = tty_text(name);
    let meta = type_and_size(file_type, size);

    let fork = if last {
        "\u{2514}\u{2500}"
    } else {
        "\u{251c}\u{2500}"
    }; // └─ / ├─
    let stem = if last { "  " } else { "\u{2502} " }; //    / │
    let indent = "   ";
    let p = palette();
    let mut out = format!(
        "{indent}{} {} {}{}\n",
        fg(p.very_dim, fork),
        fg_bold(p.path_name, &name),
        fg(p.dim, &format!("\u{00b7} {meta} \u{00b7} ")),
        fg_bold(class_color(classification), verdict_word(classification)),
    );
    let stem = fg(p.very_dim, stem);
    for line in body.lines() {
        out.push_str(indent);
        out.push_str(&stem);
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// The line a process scan adds beneath a running binary's card: the PIDs
/// running it, the first few when there are many, and whether the binary has
/// since been deleted from disk.
pub(crate) fn terminal_pids_line(pids: &[u32], deleted: bool) -> String {
    /// More PIDs than this are summarised as a count.
    const MAX_LISTED: usize = 5;
    let listed = if pids.len() <= MAX_LISTED {
        pids.iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        let shown = MAX_LISTED - 1;
        let first: Vec<String> = pids[..shown].iter().map(u32::to_string).collect();
        format!("{} +{} more", first.join(", "), pids.len() - shown)
    };
    let p = palette();
    let deleted = if deleted {
        format!("  {}", fg_bold(p.hostile, "(deleted)"))
    } else {
        String::new()
    };
    format!(
        "{}{}{deleted}",
        rail(1, "\u{2699}\u{fe0f}", "   "), // ⚙️
        fg(p.dim, &format!("pids: {listed}"))
    )
}

/// Format a route-score list as `route=raw` tokens, using each route's raw
/// model output rather than the probability the verdict reports. Per-route
/// verdicts are intentionally omitted: blend-driven routes have no
/// individually-attributable classification (see `policy_route_class`).
fn format_route_scores(scores: &[RouteScore]) -> String {
    scores
        .iter()
        .map(|s| format!("{}={:.6}", s.model, s.raw))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Write the ML diagnostics behind a grade (`--extra`) beneath its card: the
/// level, each route's raw score and the routes skipped, the SHAP features
/// behind the top-level featurization, and the routing of the worst archive
/// members. Leads with the level: it is the only place the rung a benign
/// file fired on is legible.
///
/// Members carry route scores but no SHAP reasons (reasons exist only for the
/// top-level file); to attribute a member's hit, scan the extracted member
/// directly so it becomes the top-level file.
///
/// # Errors
/// Writing to `out` failed.
pub(crate) fn write_extra(out: &mut dyn std::io::Write, r: &ScanResult) -> std::io::Result<()> {
    /// SHAP features listed, by importance.
    const MAX_REASONS: usize = 12;
    /// Archive members listed, worst first.
    const MAX_MEMBERS: usize = 10;
    writeln!(out, "  level: {}", r.level)?;
    if !r.model_scores.is_empty() {
        writeln!(
            out,
            "  routes (raw): {}",
            format_route_scores(&r.model_scores)
        )?;
    }
    if !r.skipped_models.is_empty() {
        let skipped: Vec<String> = r
            .skipped_models
            .iter()
            .map(|s| format!("{}:{}", s.model, s.reason))
            .collect();
        writeln!(out, "  skipped: {}", skipped.join(" "))?;
    }
    if !r.reasons.is_empty() {
        writeln!(out, "  shap (top features by importance):")?;
        for reason in r.reasons.iter().take(MAX_REASONS) {
            writeln!(
                out,
                "    imp={:.4} val={:.4}  {}  [{}]",
                reason.importance, reason.value, reason.feature, reason.description,
            )?;
        }
    }
    // A view of the evaluation table; the table itself is complete.
    for ef in crate::engine::EmbeddedFile::top_offenders(&r.embedded_files, MAX_MEMBERS) {
        if !ef.model_scores.is_empty() {
            writeln!(
                out,
                "  embedded {} ({} {}): routes (raw) {}",
                tty_text(&ef.path),
                ef.file_type,
                ef.level,
                format_route_scores(&ef.model_scores),
            )?;
        }
    }
    Ok(())
}

/// A bloom-filter verdict that stands instead of analysis. Known-bad and
/// conflicted are flags, not short-circuits: they are scanned, and ride the
/// result as a [`BloomMark`].
#[derive(Debug, Clone, Copy)]
pub(crate) enum BloomVerdict {
    /// Matched the known-good set — the scan was skipped.
    KnownGood,
    /// In neither set; left unscanned because the mode is `fast`.
    Unscanned,
}

/// The inline bloom provenance marker for a *scanned* file — the glyph plus its
/// label, anchored to the field that produced the match (the SHA-256 line for a
/// content match, the coordinate line for a PURL match) so the "why" is legible.
/// A file matched known-good yet still scanned (a fresh vouch inside the rescan
/// window) carries the ✓ marker; unknown files carry none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BloomMark {
    /// Matched the known-good set but was scanned anyway (a fresh vouch, inside
    /// the rescan window) — ✓.
    KnownGood,
    /// Matched the known-bad set — 🚩.
    KnownBad,
    /// Matched an outside threat-intelligence claim rather than a measurement
    /// of ours — 📡. Marked apart from [`Self::KnownBad`] because the two say
    /// different things: one is what we found, the other is what somebody else
    /// reported. Still scanned, exactly as known-bad is.
    Sighted,
    /// In both the good and bad sets — 🏴; treated as known-bad, still scanned.
    Conflicted,
}

impl BloomMark {
    /// The marker from a raw bloom decision. `None` for unknown — an unremarkable
    /// file that matched neither set carries no mark. Known-good maps to the ✓
    /// marker; callers that short-circuit a non-fresh known-good simply never
    /// reach this with `Skip`.
    #[must_use]
    pub(crate) const fn from_decision(decision: crate::bloom_repo::Decision) -> Option<Self> {
        match decision {
            crate::bloom_repo::Decision::Skip => Some(Self::KnownGood),
            crate::bloom_repo::Decision::KnownBad => Some(Self::KnownBad),
            crate::bloom_repo::Decision::SightedHostile
            | crate::bloom_repo::Decision::SightedSuspicious => Some(Self::Sighted),
            crate::bloom_repo::Decision::Conflicted => Some(Self::Conflicted),
            crate::bloom_repo::Decision::Unknown | crate::bloom_repo::Decision::Unavailable => None,
        }
    }

    /// Whether this mark must bypass the terminal's ordinary classification
    /// filter. A known-good file that was rescanned and remained benign is still
    /// quiet; bad, sighted, and conflicted claims must be surfaced even when the
    /// model itself rates the bytes benign.
    pub(crate) const fn forces_terminal_display(self) -> bool {
        !matches!(self, Self::KnownGood)
    }

    /// The glyph identifying this status: ✓ known-good, 🚩 known-bad,
    /// 📡 sighted by outside intelligence, 🏴 conflicted.
    const fn glyph(self) -> &'static str {
        match self {
            Self::KnownGood => "\u{2713}",   // ✓
            Self::KnownBad => "\u{1f6a9}",   // 🚩
            Self::Sighted => "\u{1f4e1}",    // 📡
            Self::Conflicted => "\u{1f3f4}", // 🏴
        }
    }

    /// The trust color for this marker's glyph and label.
    fn color(self) -> Rgb {
        let p = palette();
        match self {
            Self::KnownGood => p.benign,
            Self::KnownBad => p.hostile,
            // Warning, not hostile: an outside citation is somebody's claim,
            // and this scan has not confirmed it.
            Self::Sighted | Self::Conflicted => p.warning,
        }
    }

    /// The provenance suffix appended to the SHA-256 (or PURL) line: two spaces,
    /// then the glyph and label colored by trust. Plain when color is disabled.
    fn marker(self) -> String {
        let body = format!("{} {}", self.glyph(), self.tiny_str());
        format!("  {}", fg(self.color(), &body))
    }

    /// The machine token for the `--format tiny` verdict line.
    pub(crate) const fn tiny_str(self) -> &'static str {
        match self {
            Self::KnownGood => "known-good",
            Self::KnownBad => "known-bad",
            Self::Sighted => "sighted",
            Self::Conflicted => "conflicted",
        }
    }
}

/// Report a bloom decision as a real result, not a debug line: a JSON record on
/// the output stream in `--format json`, a themed one-liner on stderr otherwise.
pub(crate) fn print_bloom_verdict(
    label: &str,
    verdict: BloomVerdict,
    format: OutputFormat,
) -> std::io::Result<()> {
    if matches!(format, OutputFormat::Json) {
        return writeln!(
            std::io::stdout().lock(),
            "{}",
            serde_json::json!({
                "path": label,
                "source": "bloom",
                "verdict": match verdict {
                    BloomVerdict::KnownGood => "known-good",
                    BloomVerdict::Unscanned => "unscanned",
                },
                "scanned": false,
            })
        );
    }

    let p = palette();
    let ink = Ink::stderr();
    let (icon, name, reason) = match verdict {
        BloomVerdict::KnownGood => (
            ink.fg(p.benign, "\u{2713}"),
            ink.fg(p.benign, "known-good"),
            "skipping scan",
        ),
        BloomVerdict::Unscanned => (
            ink.fg(p.warning, "!"),
            ink.fg(p.very_dim, "unscanned"),
            "not in known sets (fast mode)",
        ),
    };
    writeln!(
        std::io::stderr().lock(),
        " {icon}  {name} {}  {}",
        ink.fg(p.very_dim, reason),
        ink.fg(p.header_path, &tty_text(label)),
    )
}

/// Write one artifact Atomdrift Uplink judged (`--uplink`), in place of a local
/// result.
///
/// JSON gets a record shaped like the bloom one — `"source": "remote"` says
/// whose verdict it is — carrying what the service said: the level, the digest
/// it describes, when and by what it was analyzed, and every finding it
/// reported. Every other format gets one line, naming its first finding.
///
/// # Errors
/// Writing to `out` failed.
pub(crate) fn write_remote_verdict(
    out: &mut impl std::io::Write,
    label: &str,
    sha256: &str,
    purl: Option<&str>,
    verdict: &crate::corpus_precheck::Verdict,
    class: Classification,
    format: OutputFormat,
) -> std::io::Result<()> {
    let name = class.to_string();
    if matches!(format, OutputFormat::Json) {
        let findings: Vec<_> = verdict
            .findings
            .iter()
            .map(|f| {
                let desc = f.desc.as_deref().unwrap_or_default();
                serde_json::json!({ "id": f.id, "crit": f.crit, "desc": desc })
            })
            .collect();
        let mut record = serde_json::json!({
            "path": label,
            "source": "remote",
            "verdict": name,
            "sha256": sha256,
            "fires_at": verdict.fires_at,
            "analyzed_at": verdict.analyzed_at,
            "engine_version": verdict.engine_version,
            "findings": findings,
        });
        if let Some(purl) = purl {
            record["purl"] = serde_json::Value::from(purl);
        }
        serde_json::to_writer(&mut *out, &record)?;
        return out.write_all(b"\n");
    }

    let p = palette();
    let (icon, color) = match class {
        Classification::Hostile => ("\u{2717}", p.hostile),
        Classification::Suspicious => ("!", p.warning),
        Classification::Benign => ("\u{2713}", p.benign),
    };
    let when = verdict
        .analyzed_at
        .as_deref()
        .and_then(|at| at.get(..10))
        .map(|day| format!(" \u{b7} analyzed {}", tty_text(day)))
        .unwrap_or_default();
    let why = verdict
        .findings
        .first()
        .map(|f| format!(" \u{b7} {}", tty_text(&f.id)))
        .unwrap_or_default();
    writeln!(
        out,
        " {}  {} {}  {}",
        fg(color, icon),
        fg(color, &name),
        fg(p.very_dim, &format!("Uplink{when}{why}")),
        fg(p.header_path, &tty_text(label)),
    )
}

/// Print scan summary. A flagged scan closes its card sequence with a heavy
/// rule in the worst verdict's color, the counts embedded in the rule (bloom
/// hits folded in). A clean scan keeps the quiet thin rule and ✓ line — the
/// frame inherits the color of what it contains, and a clean run contains no
/// color.
pub(crate) fn print_summary(summary: &ScanSummary) {
    let p = palette();
    let ink = Ink::stderr();
    let flagged = summary.hostile > 0 || summary.suspicious > 0 || summary.static_flags > 0;
    let bloom = crate::bloom_repo::counts();

    if flagged && ink.on() {
        // Cards end flush; the footer leads with its own separating blank.
        eprintln!();
        let accent = if summary.hostile > 0 {
            p.hostile
        } else {
            p.suspicious
        };
        let mut parts = summary_counts(summary, p, ink);
        if bloom.flagged > 0 {
            let s = format!("{} known-bad", bloom.flagged);
            parts.push((ink.fg(p.hostile, &s), s));
        }
        if bloom.conflicted > 0 {
            let s = format!("{} conflicted", bloom.conflicted);
            parts.push((ink.fg(p.warning, &s), s));
        }
        {
            let s = format_elapsed(summary.duration_ms);
            parts.push((ink.fg(p.very_dim, &s), s));
        }
        let sep_colored = format!(" {} ", ink.fg(p.dot_sep, "\u{00b7}"));
        let colored_line: Vec<&str> = parts.iter().map(|(c, _)| c.as_str()).collect();
        let plain_len: usize =
            parts.iter().map(|(_, s)| s.chars().count()).sum::<usize>() + (parts.len() - 1) * 3;
        // ` ` + `━━` + ` ` + content + ` ` + fill, 1-col right margin.
        let fill = cleave::output::terminal_width()
            .saturating_sub(plain_len + 6)
            .max(2);
        eprintln!(
            " {} {} {}",
            ink.fg(accent, "\u{2501}\u{2501}"),
            colored_line.join(&sep_colored),
            ink.fg(accent, &"\u{2501}".repeat(fill)),
        );
        eprintln!();
        return;
    }

    let line = ink.fg(p.summary_line, &"\u{2500}".repeat(52));
    eprintln!(" {line}");

    eprintln!(" {}", summary_status_line(summary));
    eprintln!();
}

/// The one-line scan verdict — icon plus counts plus elapsed — without the
/// leading indent, surrounding separator, or trailing blank that
/// [`print_summary`] adds. Shared by the full filesystem-scan summary and
/// `ps`'s compact finish, which overwrites its progress bar with this line.
#[must_use]
pub(crate) fn summary_status_line(summary: &ScanSummary) -> String {
    let p = palette();
    let ink = Ink::stderr();

    if summary.total_files == 0 {
        let (icon, msg) = if summary.errors > 0 {
            (
                ink.fg(p.warning, "!"),
                format!(
                    "no files scanned, {}",
                    ink.fg(p.header_path, &format!("{} errors", summary.errors)),
                ),
            )
        } else {
            (
                ink.fg(p.warning, "!"),
                "no scannable files found".to_string(),
            )
        };
        return format!(
            "{}  {}  {}",
            icon,
            msg,
            ink.fg(p.very_dim, &format_elapsed(summary.duration_ms)),
        );
    }

    if summary.hostile == 0
        && summary.suspicious == 0
        && summary.errors == 0
        && summary.dependency_attention.gaps == 0
        && summary.dependency_attention.flagged_candidates == 0
        && summary.static_flags == 0
    {
        return format!(
            "{}  {} files scanned, all graded benign  {}",
            ink.fg(p.benign, "\u{2713}"),
            summary.total_files,
            ink.fg(p.very_dim, &format_elapsed(summary.duration_ms)),
        );
    }

    let mut parts: Vec<String> = summary_counts(summary, p, ink)
        .into_iter()
        .map(|(colored, _)| colored)
        .collect();
    parts.push(ink.fg(p.very_dim, &format_elapsed(summary.duration_ms)));

    let sep = format!("  {}  ", ink.fg(p.dot_sep, "\u{00b7}"));
    parts.join(&sep)
}

/// The count segments both summaries share, in order, each as (colored,
/// plain) so the flagged rule can size its fill from visible width.
fn summary_counts(summary: &ScanSummary, p: &Palette, ink: Ink) -> Vec<(String, String)> {
    let plural = |n: u32| if n == 1 { "" } else { "s" };
    let ScanSummary {
        total_files,
        hostile,
        suspicious,
        benign,
        errors,
        static_flags,
        dependency_attention:
            crate::engine::DependencyAttention {
                gaps,
                flagged_candidates,
            },
        ..
    } = *summary;
    let files = format!("{total_files} file{}", plural(total_files));
    let mut parts = vec![(files.clone(), files)];
    for (n, color, text) in [
        (hostile, p.hostile, format!("{hostile} hostile")),
        (suspicious, p.suspicious, format!("{suspicious} suspicious")),
        (errors, p.header_path, format!("{errors} errors")),
        (
            gaps,
            p.warning,
            format!("{gaps} dependency gap{}", plural(gaps)),
        ),
        (
            static_flags,
            p.warning,
            format!("{static_flags} with severe static findings"),
        ),
        (
            flagged_candidates,
            p.warning,
            format!(
                "{flagged_candidates} flagged candidate{}",
                plural(flagged_candidates)
            ),
        ),
    ] {
        if n > 0 {
            parts.push((ink.fg(color, &text), text));
        }
    }
    let benign = format!("{benign} graded benign");
    parts.push((ink.fg(p.very_dim, &benign), benign));
    parts
}

/// `TYPE · size`, the size omitted when it renders empty.
fn type_and_size(file_type: &str, size: u64) -> String {
    let size = human_size(size);
    if size.is_empty() {
        file_type.to_uppercase()
    } else {
        format!("{} \u{00b7} {size}", file_type.to_uppercase())
    }
}

/// Warn that a process scan skipped `count` processes it may not inspect.
pub(crate) fn print_permission_warning(count: u32) {
    let p = palette();
    let ink = Ink::stderr();
    eprintln!(
        " {}  {}\n",
        ink.fg(p.suspicious, "\u{26A0}"),
        ink.fg(
            p.header_path,
            &format!(
                "Skipping {count} processes due to insufficient permissions — re-run as root for full coverage"
            ),
        ),
    );
}

/// Print the scan banner shared by every scanning subcommand (`ps`, directory,
/// multi-path): the atom mark, version, and the two detection inventories already
/// resident in memory. SHA-256/PURL Bloom entries stay distinct from executable
/// trait/composite/YARA rules instead of inflating a misleading "rules" total.
pub(crate) fn print_banner(counts: crate::engine::DetectionCounts) {
    let p = palette();
    let ink = Ink::stderr();
    eprintln!();
    eprintln!(
        " \u{269b}\u{fe0f} {} {}",
        ink.fg_bold(
            p.path_name,
            &format!("Atomdrift Scan v{}", env!("CARGO_PKG_VERSION"))
        ),
        ink.fg(
            p.very_dim,
            &detection_inventory_text(counts.hashes_and_purls, counts.rules)
        ),
    );
}

pub(crate) fn detection_inventory_text(hashes_and_purls: u64, rules: u64) -> String {
    format!(
        "\u{00b7} {} hashes \u{00b7} {} rules",
        with_commas(hashes_and_purls),
        with_commas(rules),
    )
}

/// Format an integer with `,` thousands separators (e.g. `1093027` → `1,093,027`).
fn with_commas(n: u64) -> String {
    let digits = n.to_string();
    let len = digits.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, c) in digits.char_indices() {
        if i != 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn format_elapsed(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}

/// One detection source in `scan version`: a named rule family with its headline
/// metric, the content version that identifies it, and the upstream build time.
#[derive(Debug)]
pub struct SourceRow {
    /// Family name shown in the first column (`blooms`, `atomics`, …).
    pub label: &'static str,
    /// Headline tally — a rule count, formatted with thousands separators when
    /// [`Self::metric`] is `None`.
    pub count: u64,
    /// Pre-formatted metric that overrides `count` when set (e.g. ML models
    /// shown as `"64 x 229"` — model count by feature dimensionality).
    pub metric: Option<String>,
    /// Short content identity — `v1`, a commit, or empty when none applies.
    pub version: String,
    /// Upstream build time (Unix seconds), if known: when the source was
    /// *published*, not when it was fetched locally.
    pub epoch: Option<i64>,
}

/// Render `scan version`: which rules edition is in use and the split
/// signature/rule inventory, then one aligned row per detection source with
/// its count, version, and upstream age.
///
/// `edition` is already styled for the terminal (see
/// `cleave::traits_repo::Edition::styled`).
pub fn print_version(
    scan_version: &str,
    edition: &str,
    hashes_and_purls: u64,
    rules: u64,
    rows: &[SourceRow],
) -> std::io::Result<()> {
    let mut out = std::io::stdout().lock();
    let p = palette();
    let now = i64::try_from(crate::civil::now_secs()).unwrap_or(i64::MAX);

    writeln!(out)?;
    writeln!(
        out,
        " \u{269b}\u{fe0f} {} {} {edition} {}",
        fg_bold(p.path_name, &format!("Atomdrift Scan v{scan_version}")),
        fg(p.very_dim, "\u{00b7}"),
        fg(
            p.very_dim,
            &detection_inventory_text(hashes_and_purls, rules)
        ),
    )?;
    writeln!(out)?;

    let metrics: Vec<String> = rows
        .iter()
        .map(|r| r.metric.clone().unwrap_or_else(|| with_commas(r.count)))
        .collect();
    let label_w = rows.iter().map(|r| r.label.len()).max().unwrap_or(0);
    let metric_w = metrics.iter().map(String::len).max().unwrap_or(0);
    let version_w = rows.iter().map(|r| r.version.len()).max().unwrap_or(0);

    for (r, metric) in rows.iter().zip(&metrics) {
        let (date, age, freshness) = match r.epoch {
            Some(e) => (fmt_date(e), fmt_age(now - e), freshness_color(now - e)),
            None => ("\u{2014}".to_string(), String::new(), p.very_dim),
        };
        writeln!(
            out,
            "   {}  {}  {}  {}  {}",
            fg(p.dim, &format!("{:<label_w$}", r.label)),
            fg(p.path_name, &format!("{metric:>metric_w$}")),
            fg(p.very_dim, &format!("{:<version_w$}", r.version)),
            fg(freshness, &format!("{date:<10}")),
            fg(freshness, &age),
        )?;
    }
    writeln!(out)
}

/// Format an upstream Unix timestamp as a UTC `YYYY-MM-DD` date.
fn fmt_date(epoch: i64) -> String {
    let (y, m, d) = crate::civil::from_days(epoch.div_euclid(SECS_PER_DAY));
    format!("{y:04}-{m:02}-{d:02}")
}

/// Render an age in seconds as a compact "just now / Nm / Nh / Nd ago" string.
fn fmt_age(secs: i64) -> String {
    let s = secs.max(0);
    if s < SECS_PER_MINUTE {
        "just now".to_string()
    } else if s < SECS_PER_HOUR {
        format!("{}m ago", s / SECS_PER_MINUTE)
    } else if s < SECS_PER_DAY {
        format!("{}h ago", s / SECS_PER_HOUR)
    } else {
        format!("{}d ago", s / SECS_PER_DAY)
    }
}

/// Staleness color for a source's age: green within 48h, yellow within a week,
/// red beyond. Reuses the classification palette so freshness reads the same way
/// a verdict does.
fn freshness_color(secs: i64) -> Rgb {
    const FRESH: i64 = 2 * SECS_PER_DAY;
    const CURRENT: i64 = 7 * SECS_PER_DAY;
    let p = palette();
    match secs.max(0) {
        age if age <= FRESH => p.benign,
        age if age <= CURRENT => p.suspicious,
        _ => p.hostile,
    }
}

#[cfg(test)]
mod tests;
