//! Terminal and JSON output formatting with litmus-colored confidence indicators.
//! Supports both dark and light terminal backgrounds via auto-detection.

use std::io::IsTerminal;
use std::sync::{LazyLock, RwLock};
use unicode_width::UnicodeWidthStr;

use crate::OutputFormat;
use crate::engine::{ScanResult, ScanSummary, level_confidence};
use crate::model::{Classification, RouteScore};

const BLOCK: &str = "\u{2588}";
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
struct Rgb(u8, u8, u8);

/// Terminal background theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    /// Dark terminal background (default).
    Dark,
    /// Light/white terminal background.
    Light,
}

/// Color palette tuned for a specific background theme.
#[derive(Debug, Clone, Copy)]
struct Palette {
    // Classification colors — high-vis on both backgrounds.
    hostile: Rgb,
    hostile_finding: Rgb,
    suspicious: Rgb,
    suspicious_finding: Rgb,
    benign: Rgb,

    // Semantic accent colors.
    filetype: Rgb,
    formula: Rgb,

    // UI chrome.
    dot_sep: Rgb,
    dim: Rgb,
    very_dim: Rgb,
    path_name: Rgb,
    header_path: Rgb,
    summary_line: Rgb,
    warning: Rgb,
    arrow: Rgb,
    reason: Rgb,
}

impl Palette {
    const fn dark() -> Self {
        Self {
            hostile: Rgb(255, 70, 70),
            hostile_finding: Rgb(255, 100, 100),
            suspicious: Rgb(255, 175, 55),
            suspicious_finding: Rgb(255, 190, 90),
            benign: Rgb(80, 200, 80),

            filetype: Rgb(110, 110, 110),
            formula: Rgb(110, 110, 110),

            dot_sep: Rgb(50, 50, 50),
            dim: Rgb(100, 100, 100),
            very_dim: Rgb(80, 80, 80),
            path_name: Rgb(230, 230, 230),
            header_path: Rgb(180, 180, 180),
            summary_line: Rgb(50, 50, 50),
            warning: Rgb(180, 180, 60),
            arrow: Rgb(70, 70, 70),
            reason: Rgb(100, 100, 100),
        }
    }

    const fn light() -> Self {
        Self {
            hostile: Rgb(200, 30, 30),
            hostile_finding: Rgb(180, 40, 40),
            suspicious: Rgb(180, 120, 0),
            suspicious_finding: Rgb(160, 100, 0),
            benign: Rgb(30, 140, 30),

            filetype: Rgb(130, 130, 130),
            formula: Rgb(130, 130, 130),

            dot_sep: Rgb(180, 180, 180),
            dim: Rgb(120, 120, 120),
            very_dim: Rgb(150, 150, 150),
            path_name: Rgb(30, 30, 30),
            header_path: Rgb(80, 80, 80),
            summary_line: Rgb(190, 190, 190),
            warning: Rgb(140, 140, 0),
            arrow: Rgb(160, 160, 160),
            reason: Rgb(100, 100, 100),
        }
    }
}

/// Process-global theme selection. Starts unset, then caches either an explicit
/// override or the first successful auto-detection result.
static THEME: LazyLock<RwLock<Option<Theme>>> = LazyLock::new(|| RwLock::new(None));

/// Detect the terminal theme, with env var override.
///
/// Priority: `SCAN_THEME` env var > terminal query > default (dark).
pub fn detect_theme() -> Theme {
    if let Ok(theme) = THEME.read()
        && let Some(theme) = *theme
    {
        return theme;
    }

    let detected = if let Ok(val) = std::env::var("SCAN_THEME") {
        match val.to_ascii_lowercase().as_str() {
            "light" | "white" => Theme::Light,
            _ => Theme::Dark,
        }
    } else if
    // Terminal queries must only be attempted on a real TTY. Without one,
    // the read blocks indefinitely or the kernel sends SIGTTIN.
    !std::io::stderr().is_terminal() {
        Theme::Dark
    } else {
        match terminal_colorsaurus::color_scheme(terminal_colorsaurus::QueryOptions::default()) {
            Ok(scheme) => match scheme {
                terminal_colorsaurus::ColorScheme::Dark => Theme::Dark,
                terminal_colorsaurus::ColorScheme::Light => Theme::Light,
            },
            Err(_) => Theme::Dark,
        }
    };

    if let Ok(mut theme) = THEME.write() {
        *theme = Some(detected);
    }

    detected
}

/// Override the theme (called from CLI flags before any output).
pub fn set_theme(theme: Theme) {
    if let Ok(mut current) = THEME.write() {
        *current = Some(theme);
    }
}

/// The active theme, defaulting to dark when unset or the lock is poisoned.
fn current_theme() -> Theme {
    THEME
        .read()
        .ok()
        .and_then(|theme| *theme)
        .unwrap_or(Theme::Dark)
}

fn palette() -> &'static Palette {
    static DARK: Palette = Palette::dark();
    static LIGHT: Palette = Palette::light();

    match current_theme() {
        Theme::Dark => &DARK,
        Theme::Light => &LIGHT,
    }
}

/// Set truecolor foreground on text.
fn fg(Rgb(r, g, b): Rgb, text: &str) -> String {
    format!("\x1b[38;2;{r};{g};{b}m{text}\x1b[0m")
}

/// Set truecolor foreground + bold on text.
fn fg_bold(Rgb(r, g, b): Rgb, text: &str) -> String {
    format!("\x1b[1;38;2;{r};{g};{b}m{text}\x1b[0m")
}

/// Linear interpolation between two RGB colors.
// The .clamp(0.0, 255.0) bounds the value before the u8 cast; sign-loss is
// impossible because clamp ensures non-negative output.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn mix_rgb(a: Rgb, b: Rgb, t: f32) -> Rgb {
    let ch = |x: u8, y: u8| -> u8 {
        (x as f32 + (y as f32 - x as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Rgb(ch(a.0, b.0), ch(a.1, b.1), ch(a.2, b.2))
}

/// Normalized progress within the current classification band.
///
/// `threshold` is the cutoff that defined `classification` (the value `prob`
/// was compared against). The envelope no longer carries both suspicious and
/// hostile cutoffs, so the suspicious band uses the same "distance to 1.0"
/// formula as the hostile band — an acceptable approximation for the
/// band-internal color ramp.
fn band_progress(probability: f32, classification: &Classification, threshold: f32) -> f32 {
    match classification {
        Classification::Hostile | Classification::Suspicious => {
            ((probability - threshold) / (1.0 - threshold).max(f32::EPSILON)).clamp(0.0, 1.0)
        }
        Classification::Benign => (probability / threshold.max(f32::EPSILON)).clamp(0.0, 1.0),
    }
}

/// Litmus-style color progression within each classification band.
fn indicator_colors(
    probability: f32,
    classification: &Classification,
    threshold: f32,
) -> (Rgb, Rgb, Rgb) {
    let t = band_progress(probability, classification, threshold);
    let theme = current_theme();

    match (theme, classification) {
        // Benign shifts from strong green toward yellow-green near the suspicious boundary.
        (Theme::Dark, Classification::Benign) => (
            mix_rgb(Rgb(25, 170, 120), Rgb(120, 190, 40), t),
            mix_rgb(Rgb(70, 215, 135), Rgb(195, 210, 60), t),
            mix_rgb(Rgb(45, 195, 105), Rgb(160, 205, 50), t),
        ),
        (Theme::Light, Classification::Benign) => (
            mix_rgb(Rgb(20, 130, 95), Rgb(120, 135, 20), t),
            mix_rgb(Rgb(35, 155, 100), Rgb(150, 150, 25), t),
            mix_rgb(Rgb(25, 145, 95), Rgb(135, 145, 20), t),
        ),

        // Suspicious starts at greenish-yellow, moves through yellow, and ends orange.
        (Theme::Dark, Classification::Suspicious) => (
            mix_rgb(Rgb(170, 190, 45), Rgb(255, 180, 40), t),
            mix_rgb(Rgb(235, 220, 65), Rgb(255, 125, 20), t),
            mix_rgb(Rgb(210, 205, 55), Rgb(255, 150, 30), t),
        ),
        (Theme::Light, Classification::Suspicious) => (
            mix_rgb(Rgb(125, 145, 20), Rgb(205, 135, 0), t),
            mix_rgb(Rgb(165, 150, 20), Rgb(175, 100, 0), t),
            mix_rgb(Rgb(145, 145, 15), Rgb(185, 115, 0), t),
        ),

        // Hostile starts orange-red at the boundary and deepens to saturated red.
        (Theme::Dark, Classification::Hostile) => (
            mix_rgb(Rgb(255, 135, 40), Rgb(255, 50, 65), t),
            mix_rgb(Rgb(255, 95, 35), Rgb(255, 35, 35), t),
            mix_rgb(Rgb(255, 120, 40), Rgb(255, 45, 50), t),
        ),
        (Theme::Light, Classification::Hostile) => (
            mix_rgb(Rgb(190, 85, 10), Rgb(200, 30, 30), t),
            mix_rgb(Rgb(170, 65, 10), Rgb(175, 25, 25), t),
            mix_rgb(Rgb(185, 75, 10), Rgb(190, 30, 30), t),
        ),
    }
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
/// [`level_confidence`] maps a negative level — "cleared no calibrated level
/// at all" — onto `Some(0)`. That is a defensible wire value, but rendering it
/// prints ` BENIGN    0% `, which reads as *maximum* confidence in benign. The
/// model never said that. It said the sample cleared no calibrated level,
/// which is the same state as carrying no level marker: unknown, not certain.
///
/// The failure this fixes: an npm package with a raw probability of 0.998
/// against a higher cutoff rendered as `BENIGN[0%]` directly above cleave's
/// own hostile finding for the same file. Returning `None` routes those
/// through the existing no-level path — the threshold-relative percentage —
/// so the badge shows how near the cutoff the sample actually sat.
///
/// Display-only by construction: the JSON `conf` field keeps calling
/// [`level_confidence`], so the machine-readable shape is unchanged.
fn display_confidence(level: Option<i32>) -> Option<u8> {
    match level {
        Some(n) if n < 0 => None,
        other => level_confidence(other),
    }
}

/// Color percentage text to match the classification band.
fn colored_pct(probability: f32, classification: &Classification, threshold: f32) -> String {
    let pct = format!(
        "{:>4}",
        format!(
            "{:.0}%",
            display_probability(probability, threshold) * 100.0
        )
    );
    let (_, _, accent) = indicator_colors(probability, classification, threshold);
    fg(accent, &pct)
}

/// Header severity token: the level-derived confidence percent when the level
/// marker exists, otherwise the rescaled threshold-relative percentage.
fn colored_conf_or_pct(
    probability: f32,
    classification: &Classification,
    threshold: f32,
    level: Option<i32>,
) -> String {
    match display_confidence(level) {
        Some(conf) => {
            let (_, _, accent) = indicator_colors(probability, classification, threshold);
            fg(accent, &format!("{:>4}", format!("{conf}%")))
        }
        None => colored_pct(probability, classification, threshold),
    }
}

/// Fixed per-class anchor color: red hostile, yellow suspicious, green benign.
/// The verdict stamp and card rules use this — the word names the verdict, so
/// the hue must match it exactly; confidence nuance lives in the percentage.
/// (The band-interior litmus gradient survives only in `ps`'s wordless
/// two-block indicator, where hue is the sole carrier.)
fn class_color(classification: &Classification) -> Rgb {
    let p = palette();
    match classification {
        Classification::Hostile => p.hostile,
        Classification::Suspicious => p.suspicious,
        Classification::Benign => p.benign,
    }
}

/// Classification label colored to match the band.
fn colored_label(classification: &Classification, p: &Palette) -> String {
    match classification {
        Classification::Hostile => fg(p.hostile, "hostile"),
        Classification::Suspicious => fg(p.suspicious, "suspicious"),
        Classification::Benign => fg(p.benign, "benign"),
    }
}

/// litmus's verdict stamp that leads cleave's headline: the verdict word and
/// band percentage on a filled, classification-colored background —
/// ` HOSTILE 100% ` (red), ` SUSPECT  62% ` (orange), ` BENIGN    3% `
/// (green) — so the verdict reads in both the color and the word. Every stamp
/// is the same [`BADGE_WIDTH`] columns (7-char word slot, right-aligned
/// percentage), so verdicts line up down a scan listing. Plain
/// `HOSTILE[100%]` when color is disabled (piped). Returns the stamp and its
/// visible width (for aligning lines beneath it).
pub(crate) fn terminal_badge(
    classification: &Classification,
    probability: f32,
    threshold: f32,
    level: Option<i32>,
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
    let word = match classification {
        Classification::Hostile => "HOSTILE",
        Classification::Suspicious => "SUSPECT",
        Classification::Benign => "BENIGN",
    };
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        let stamp = format!("{word}[{pct}]");
        let width = stamp.chars().count();
        return (stamp, width);
    }
    // Bold white on the class anchor color (fixed red/yellow/green — the word
    // names the verdict, so the hue must agree with it; the percentage carries
    // the confidence), fixed-width so every stamp is the same size block.
    let Rgb(r, g, b) = class_color(classification);
    (
        format!("\x1b[1;38;2;255;255;255;48;2;{r};{g};{b}m {word:<7} {pct:>4} \x1b[0m"),
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
    if colored::control::SHOULD_COLORIZE.should_colorize() {
        let p = palette();
        Some(format!(
            "{}{}",
            terminal_marker_prefix_with_width(1, &fg(p.dim, "\u{00b7}"), 1),
            fg(p.reason, &body)
        ))
    } else {
        Some(format!("\u{00b7} {body}"))
    }
}

/// The LLM interpretation as its own headline-grade line, sitting directly
/// above the SHA-256: `✨ <one-line interpretation>` in bold foreground (white
/// on dark terminals), the sparkle marking it as model-generated. A failed
/// interpretation renders dim instead — an error is not a finding. `None` when
/// no interpretation ran.
pub(crate) fn terminal_interpretation(
    interpretation: Option<&crate::interpret::Interpretation>,
    indent: usize,
) -> Option<String> {
    let llm = interpretation?;
    let pad = " ".repeat(indent);
    let color = colored::control::SHOULD_COLORIZE.should_colorize();
    if llm.error.is_some() {
        return Some(if color {
            format!(
                "{}{}",
                terminal_marker_prefix(indent, "\u{2728}"),
                fg(palette().very_dim, "interpretation unavailable")
            )
        } else {
            format!("{pad}\u{2728} interpretation unavailable")
        });
    }
    Some(if color {
        format!(
            "{}{}",
            terminal_marker_prefix(indent, "\u{2728}"),
            fg_bold(palette().path_name, &llm.interpretation)
        )
    } else {
        format!("{pad}\u{2728} {}", llm.interpretation)
    })
}

/// One concise identity claim beneath the artifact name. The ID-card glyph
/// gives this optional line the same visual gutter as artifact and hash lines.
pub(crate) fn terminal_identity_line(identity: &str) -> Option<String> {
    let identity = identity.trim();
    if identity.is_empty() {
        return None;
    }
    Some(if colored::control::SHOULD_COLORIZE.should_colorize() {
        format!(
            "{}{}",
            terminal_marker_prefix(1, "\u{1faaA}"), // 🪪
            fg(palette().header_path, identity)
        )
    } else {
        format!("   {identity}")
    })
}

/// One finding in the artifact-level terminal summary.
pub(crate) struct TerminalTrait {
    pub(crate) criticality: cleave::Criticality,
    pub(crate) description: String,
    pub(crate) location: String,
}

fn truncate_terminal_text(text: &str, width: usize) -> String {
    let len = text.chars().count();
    if len <= width {
        return text.to_string();
    }
    if width <= 1 {
        return "…".chars().take(width).collect();
    }
    text.chars()
        .take(width - 1)
        .chain(std::iter::once('…'))
        .collect()
}

fn truncate_terminal_location(text: &str, width: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= width {
        return text.to_string();
    }
    if width <= 1 {
        return "…".chars().take(width).collect();
    }
    let head = (width - 1) / 2;
    let tail = width - 1 - head;
    chars[..head]
        .iter()
        .copied()
        .chain(std::iter::once('…'))
        .chain(chars[chars.len() - tail..].iter().copied())
        .collect()
}

fn severity_dots(criticality: cleave::Criticality) -> (String, usize) {
    let dots = match criticality {
        cleave::Criticality::Hostile => "●●●",
        cleave::Criticality::Suspicious => "●●",
        cleave::Criticality::Notable => "●",
        _ => "·",
    };
    let width = UnicodeWidthStr::width(dots);
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
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

    // Keep the reading measure composed on very wide terminals, but respect a
    // narrow terminal so these rows never become an accidental second line.
    let width = terminal_width.min(100);
    let fixed = 5usize; // leading space + 3-cell marker rail + 1-cell gutter
    let available = width.saturating_sub(fixed);
    let max_location = traits
        .iter()
        .map(|t| t.location.chars().count())
        .max()
        .unwrap_or(0);
    let location_width = if available >= 28 && max_location > 0 {
        max_location.min(36).min(available.saturating_sub(20))
    } else {
        0
    };
    let location_gap = usize::from(location_width > 0) * 2;
    let description_width = available.saturating_sub(location_width + location_gap);
    let color = colored::control::SHOULD_COLORIZE.should_colorize();
    let p = palette();

    traits
        .iter()
        .take(3)
        .map(|t| {
            let description = truncate_terminal_text(&t.description, description_width);
            let (dots, dots_width) = severity_dots(t.criticality);
            let prefix = terminal_marker_prefix_with_width(1, &dots, dots_width);
            if location_width == 0 {
                return format!("{prefix}{description}");
            }
            let location = truncate_terminal_location(&t.location, location_width);
            let padding = " ".repeat(description_width.saturating_sub(description.chars().count()));
            let location = if color {
                fg(p.dim, &location)
            } else {
                location
            };
            format!("{prefix}{description}{padding}  {location}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The card's identity line: a glyph, then the full SHA-256 (copy-able for
/// lookups) in the dimmest ink on screen. The glyph carries the bloom verdict —
/// 🧬 ordinarily, 🚩 known-bad, 🏴 conflicted — because the SHA-256 is what the
/// filters matched on; a known-good-but-rescanned file keeps 🧬 with a green ✓
/// trailer. Plain mode keeps the old text marker (grep-able). `None` when the
/// hash is absent.
pub(crate) fn terminal_hash_line(sha256: &str, bloom: Option<BloomMark>) -> Option<String> {
    if sha256.is_empty() {
        return None;
    }
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
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
    classification: &Classification,
    probability: f32,
    threshold: f32,
    level: Option<i32>,
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
    let glyph = if container {
        "\u{1f4e6}" // 📦
    } else {
        "\u{1f4c4}" // 📄
    };
    let mut meta = file_type.to_uppercase();
    let size = human_size(size);
    if !size.is_empty() {
        meta.push_str(&format!(" \u{00b7} {size}"));
    }
    if colored::control::SHOULD_COLORIZE.should_colorize() {
        let p = palette();
        format!(
            "{}{} {}",
            terminal_marker_prefix(1, glyph),
            fg_bold(p.path_name, label),
            fg(p.dim, &format!("\u{00b7} {meta}")),
        )
    } else {
        format!("{label} \u{00b7} {meta}")
    }
}

/// Render one fetched artifact as a branch beneath the file that referenced it.
/// Provenance belongs in the relationship (`dependency from package.json`),
/// while URL, hash, and traits occupy one aligned child column.
pub(crate) fn terminal_reference_branch(
    verdict: Option<(&Classification, f32)>,
    subject: &str,
    source: &str,
    body: &str,
    last: bool,
) -> String {
    let status = verdict.map(|(classification, probability)| {
        let word = match classification {
            Classification::Hostile => "HOSTILE",
            Classification::Suspicious => "SUSPECT",
            Classification::Benign => "BENIGN",
        };
        (word, format!("{:.0}%", probability * 100.0))
    });
    let relationship = format!("{subject} from {source}");
    let fork = if last {
        "\u{2514}\u{2500}"
    } else {
        "\u{251c}\u{2500}"
    };
    let stem = if last { "  " } else { "\u{2502} " };
    let indent = "   ";
    let mut out = String::new();
    if colored::control::SHOULD_COLORIZE.should_colorize() {
        let p = palette();
        out.push_str(indent);
        out.push_str(&fg(p.very_dim, fork));
        out.push(' ');
        out.push_str(&fg_bold(p.path_name, &relationship));
        out.push(' ');
        out.push_str(&fg(p.very_dim, "·"));
        out.push(' ');
        if let (Some((classification, _)), Some((word, pct))) = (verdict, status.as_ref()) {
            out.push_str(&fg_bold(class_color(classification), word));
            out.push(' ');
            out.push_str(&fg(p.dim, pct));
        } else {
            out.push_str(&fg_bold(p.warning, "NOT EVALUATED"));
        }
        out.push('\n');
        for line in body.lines() {
            out.push_str(indent);
            out.push_str(&fg(p.very_dim, stem));
            out.push_str(line);
            out.push('\n');
        }
    } else {
        let status = status.map_or_else(
            || "NOT EVALUATED".to_string(),
            |(word, pct)| format!("{word} {pct}"),
        );
        out.push_str(&format!(
            "{indent}{fork} {relationship} \u{00b7} {status}\n"
        ));
        for line in body.lines() {
            out.push_str(indent);
            out.push_str(stem);
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Non-graded counterpart for a fetched/registry branch.
pub(crate) fn terminal_reference_status_heading(status: &str, subject: &str) -> String {
    if colored::control::SHOULD_COLORIZE.should_colorize() {
        let p = palette();
        format!(
            "  ↳ {} {} {}",
            fg_bold(p.warning, status),
            fg(p.very_dim, "·"),
            fg(p.dim, subject),
        )
    } else {
        format!("  ↳ {status} · {subject}")
    }
}

/// The fetched locator as a secondary identity line in the nested branch.
pub(crate) fn terminal_reference_locator(locator: &str) -> String {
    if colored::control::SHOULD_COLORIZE.should_colorize() {
        format!("    {}", fg_bold(palette().path_name, locator))
    } else {
        format!("    {locator}")
    }
}

/// Locator row used inside a compact fetched-reference branch.
pub(crate) fn terminal_reference_locator_row(locator: &str) -> String {
    if colored::control::SHOULD_COLORIZE.should_colorize() {
        format!(
            "{}{}",
            terminal_marker_prefix(1, "\u{1f517}"), // 🔗
            fg_bold(palette().path_name, locator)
        )
    } else {
        format!(" \u{1f517}  {locator}")
    }
}

/// Hash/type/size row used inside a compact fetched-reference branch.
pub(crate) fn terminal_reference_hash_row(sha256: &str, file_type: &str, size: u64) -> String {
    let mut meta = file_type.to_uppercase();
    let size = human_size(size);
    if !size.is_empty() {
        meta.push_str(&format!(" \u{00b7} {size}"));
    }
    let hash = if colored::control::SHOULD_COLORIZE.should_colorize() {
        format!(
            "{}{}",
            terminal_marker_prefix(1, "\u{1f9ec}"), // 🧬
            fg(palette().very_dim, sha256)
        )
    } else {
        format!(" \u{1f9ec}  {sha256}")
    };
    if meta.is_empty() {
        return hash;
    }
    if colored::control::SHOULD_COLORIZE.should_colorize() {
        format!("{hash}  {}", fg(palette().dim, &format!("\u{00b7} {meta}")))
    } else {
        format!("{hash}  \u{00b7} {meta}")
    }
}

/// Frame one independently-scanned package nested inside an archive as a
/// bordered card: a verdict word + name header opening a left rule that runs
/// down the package's own findings (`body`, cleave's rendered context), so a
/// grab-bag archive reads as a stack of distinct verdicts rather than one flat
/// member list. Plain (piped) mode drops the border and indents two spaces,
/// keeping every line grep-able.
pub(crate) fn terminal_card(
    classification: &Classification,
    name: &str,
    file_type: &str,
    size: u64,
    body: &str,
) -> String {
    // A card leads with the class *word*, not a percentage: every package in a
    // hostile archive shares the same confidence band, so eight identical `92%`
    // stamps read as a stuck gauge. The one calibrated number lives on the
    // archive banner; the cards rank by severity (worst first) and by class.
    let word = match classification {
        Classification::Hostile => "HOSTILE",
        Classification::Suspicious => "SUSPECT",
        Classification::Benign => "BENIGN",
    };
    let mut meta = file_type.to_uppercase();
    let size = human_size(size);
    if !size.is_empty() {
        meta.push_str(&format!(" \u{00b7} {size}"));
    }
    let mut out = String::new();
    if colored::control::SHOULD_COLORIZE.should_colorize() {
        let p = palette();
        let accent = class_color(classification);
        let Rgb(r, g, b) = accent;
        let bar = fg(accent, "\u{2502}"); // │
        let stamp = format!("\x1b[1;38;2;255;255;255;48;2;{r};{g};{b}m {word} \x1b[0m");
        out.push_str(&format!(
            "{} {stamp} {} {}\n",
            fg(accent, "\u{256d}\u{2500}"), // ╭─
            fg_bold(p.path_name, name),
            fg(p.dim, &format!("\u{00b7} {meta}")),
        ));
        for line in body.lines() {
            if line.is_empty() {
                out.push_str(&bar);
                out.push('\n');
            } else {
                out.push_str(&format!("{bar} {line}\n"));
            }
        }
        out.push_str(&fg(accent, "\u{2570}\u{2500}")); // ╰─
        out.push('\n');
    } else {
        out.push_str(&format!("{word} {name} \u{00b7} {meta}\n"));
        for line in body.lines() {
            if line.is_empty() {
                out.push('\n');
            } else {
                out.push_str(&format!("  {line}\n"));
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
    classification: &Classification,
    name: &str,
    file_type: &str,
    size: u64,
    body: &str,
    last: bool,
) -> String {
    let word = match classification {
        Classification::Hostile => "HOSTILE",
        Classification::Suspicious => "SUSPECT",
        Classification::Benign => "BENIGN",
    };
    let mut meta = file_type.to_uppercase();
    let size = human_size(size);
    if !size.is_empty() {
        meta.push_str(&format!(" \u{00b7} {size}"));
    }

    let fork = if last {
        "\u{2514}\u{2500}"
    } else {
        "\u{251c}\u{2500}"
    }; // └─ / ├─
    let stem = if last { "  " } else { "\u{2502} " }; //    / │
    let indent = "   ";
    let mut out = String::new();
    if colored::control::SHOULD_COLORIZE.should_colorize() {
        let p = palette();
        let accent = class_color(classification);
        out.push_str(indent);
        out.push_str(&fg(p.very_dim, fork));
        out.push(' ');
        out.push_str(&fg_bold(p.path_name, name));
        out.push(' ');
        out.push_str(&fg(p.dim, &format!("\u{00b7} {meta} \u{00b7} ")));
        out.push_str(&fg_bold(accent, word));
        out.push('\n');
        for line in body.lines() {
            out.push_str(indent);
            out.push_str(&fg(p.very_dim, stem));
            out.push_str(line);
            out.push('\n');
        }
    } else {
        out.push_str(&format!(
            "{indent}{fork} {name} \u{00b7} {meta} \u{00b7} {word}\n"
        ));
        for line in body.lines() {
            out.push_str(indent);
            out.push_str(stem);
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Print a process scan result with PID annotations.
pub fn print_ps_result(
    result: &ScanResult,
    pids: &[u32],
    deleted: bool,
    has_progress: bool,
    extra: bool,
) {
    if has_progress {
        eprint!("\r\x1b[2K");
    }
    let p = palette();
    // Shift both blocks across the verdict band so threshold-adjacent results
    // remain distinct from high-confidence results in the same class.
    let (left, right, _) =
        indicator_colors(result.probability, &result.classification, result.threshold);
    let blocks = format!("{}{}", fg(left, BLOCK), fg(right, BLOCK));
    let pct = colored_conf_or_pct(
        result.probability,
        &result.classification,
        result.threshold,
        result.level,
    );

    // Format PID list.
    let pid_str = if pids.len() <= 5 {
        pids.iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        let first: Vec<String> = pids[..4].iter().map(u32::to_string).collect();
        format!("{} +{} more", first.join(", "), pids.len() - 4)
    };
    let pid_display = fg(p.dim, &format!("[pids: {pid_str}]"));
    let label = colored_label(&result.classification, p);

    let deleted_marker = if deleted {
        format!(" {}", fg(p.hostile_finding, "(deleted)"))
    } else {
        String::new()
    };

    eprintln!(
        " {blocks} {pct} {label}  {}{deleted_marker}  {pid_display}",
        fg_bold(p.path_name, &result.path),
    );

    if let Some(llm) = &result.interpretation {
        eprint!("   {}", crate::engine::format_llm_line(llm, true));
    }
    print_detail_lines(result, p);
    print_reasons(result, p);
    if extra {
        print_extra(result, p);
    }
    eprintln!();
}

/// Print metadata line (filetype + formula) and findings line.
fn print_detail_lines(result: &ScanResult, p: &Palette) {
    let dot = fg(p.dot_sep, "\u{00b7}");

    // Line 2: filetype · formula. The headline carries the confidence; the raw
    // level remains available in --extra output for debugging.
    let mut meta: Vec<String> = Vec::new();
    meta.push(fg(p.filetype, &result.file_type));
    if !result.formula.is_empty() {
        meta.push(fg(p.formula, &result.formula));
    }
    eprintln!("          {}", meta.join(&format!(" {dot} ")));

    // Line 3: findings (classification-colored)
    if !result.top_findings.is_empty() {
        let findings: Vec<String> = result
            .top_findings
            .iter()
            .take(4)
            .map(|f| {
                let base = f.id.split("::").next().unwrap_or(&f.id);
                let name = base
                    .strip_prefix("objectives/")
                    .or_else(|| base.strip_prefix("micro-behaviors/"))
                    .or_else(|| base.strip_prefix("well-known/"))
                    .or_else(|| base.strip_prefix("metadata/"))
                    .unwrap_or(base);
                match f.crit {
                    5 => fg(p.hostile_finding, name),
                    4 => fg(p.suspicious_finding, name),
                    _ => name.to_string(),
                }
            })
            .collect();
        eprintln!("          {}", findings.join(&format!(" {dot} ")));
    }
}

/// Print SHAP reason line if present.
fn print_reasons(result: &ScanResult, p: &Palette) {
    if !result.reasons.is_empty() {
        let reason_strs: Vec<&str> = result
            .reasons
            .iter()
            .take(3)
            .map(|r| r.description.as_str())
            .collect();
        eprintln!(
            "          {} {}",
            fg(p.arrow, "\u{2191}"),
            fg(p.reason, &reason_strs.join(", ")),
        );
    }
}

/// Format a route-score list as `route=raw` tokens, using the *raw*
/// (pre-isotonic) model score. The calibrated probability saturates the upper
/// tail to 1.0, so it hides which route actually drove the grade and how
/// confident the model really was — the raw score preserves that resolution.
/// Per-route verdicts are intentionally omitted: blend-driven routes have no
/// individually-attributable classification (see `policy_route_class`).
pub(crate) fn format_route_scores(scores: &[RouteScore]) -> String {
    scores
        .iter()
        .map(|s| format!("{}={:.6}", s.model, s.raw))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render a matched operating-point level for display: `L0`, `L50`, or `—`
/// when the file fired at no calibrated level (benign past the loosest row).
pub(crate) fn format_level(level: Option<i32>) -> String {
    level.map_or_else(|| "\u{2014}".to_string(), |n| format!("L{n}"))
}

/// Print the matched level, raw route scores, and SHAP feature values
/// (hidden --extra mode). Leads with the level — the operating point a file
/// maps to is the actionable signal; the calibrated probability is not shown
/// because isotonic saturation collapses it to 1.0 across the whole upper tail.
fn print_extra(result: &ScanResult, p: &Palette) {
    eprintln!(
        "          {} {}",
        fg(p.dim, "level:"),
        fg(p.dim, &format_level(result.level)),
    );
    if !result.model_scores.is_empty() {
        eprintln!(
            "          {} {}",
            fg(p.dim, "models:"),
            fg(p.dim, &format_route_scores(&result.model_scores)),
        );
    }
    // Surface the routing of archive members. A container often inherits a
    // member's verdict (see `elevated archive classification`), so when the
    // container's own models disagree with the final grade, the responsible
    // route lives here — not above. A top-10 view of the evaluation table.
    for ef in crate::engine::EmbeddedFile::top_offenders(&result.embedded_files, 10) {
        if ef.model_scores.is_empty() {
            continue;
        }
        eprintln!(
            "          {} {} {} {} {}",
            fg(p.dim, "embedded:"),
            fg(p.dim, &ef.path),
            fg(
                p.dim,
                &format!("[{} {}]", ef.file_type, format_level(ef.level)),
            ),
            fg(p.dim, &format_route_scores(&ef.model_scores)),
            fg(p.very_dim, "(raw scores)"),
        );
    }
    if !result.skipped_models.is_empty() {
        let skipped: Vec<String> = result
            .skipped_models
            .iter()
            .map(|s| format!("{}:{}", s.model, s.reason))
            .collect();
        eprintln!(
            "          {} {}",
            fg(p.dim, "skipped:"),
            fg(p.dim, &skipped.join(" "))
        );
    }
    if !result.reasons.is_empty() {
        eprintln!("          {}", fg(p.dim, "shap:"));
        for r in &result.reasons {
            eprintln!(
                "            {} {} {}",
                fg(p.dim, &format!("{:>8.4}", r.importance)),
                fg(p.dim, &format!("val={:<8.2}", r.value)),
                fg(p.very_dim, &r.feature),
            );
        }
    }
}

/// A bloom-filter verdict, surfaced before (or instead of) expensive analysis.
#[derive(Debug, Clone, Copy)]
pub enum BloomVerdict {
    /// Matched the known-good set — the scan was skipped.
    KnownGood,
    /// Matched the known-bad set — flagged, and still scanned to confirm.
    KnownBad,
    /// In both the good and bad sets — contradictory; flagged and scanned.
    Conflicted,
    /// In neither set; left unscanned because the mode is `fast`.
    Unscanned,
}

impl BloomVerdict {
    /// Whether the artifact still goes through full analysis after this verdict.
    /// Known-bad and conflicted are flags, not short-circuits; good and unscanned
    /// stop here.
    const fn scanned(self) -> bool {
        matches!(self, Self::KnownBad | Self::Conflicted)
    }

    const fn json_str(self) -> &'static str {
        match self {
            Self::KnownGood => "known-good",
            Self::KnownBad => "known-bad",
            Self::Conflicted => "conflicted",
            Self::Unscanned => "unscanned",
        }
    }
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
            crate::bloom_repo::Decision::Unknown => None,
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
        if colored::control::SHOULD_COLORIZE.should_colorize() {
            format!("  {}", fg(self.color(), &body))
        } else {
            format!("  {body}")
        }
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
pub fn print_bloom_verdict(label: &str, verdict: BloomVerdict, format: OutputFormat) {
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "path": label,
                "source": "bloom",
                "verdict": verdict.json_str(),
                "scanned": verdict.scanned(),
            })
        );
        return;
    }

    let p = palette();
    let (icon, name, reason) = match verdict {
        BloomVerdict::KnownGood => (
            fg(p.benign, "\u{2713}"),
            fg(p.benign, "known-good"),
            "skipping scan",
        ),
        BloomVerdict::KnownBad => (
            fg(p.hostile, "\u{2717}"),
            fg(p.hostile, "known-bad"),
            "flagged — scanning",
        ),
        BloomVerdict::Conflicted => (
            fg(p.warning, "\u{26a0}"),
            fg(p.warning, "conflicted"),
            "in good and bad — scanning",
        ),
        BloomVerdict::Unscanned => (
            fg(p.warning, "!"),
            fg(p.very_dim, "unscanned"),
            "not in known sets (fast mode)",
        ),
    };
    eprintln!(
        " {icon}  {name} {}  {}",
        fg(p.very_dim, reason),
        fg(p.header_path, label),
    );
}

/// Print scan summary. A flagged scan closes its card sequence with a heavy
/// rule in the worst verdict's color, the counts embedded in the rule (bloom
/// hits folded in). A clean scan keeps the quiet thin rule and ✓ line — the
/// frame inherits the color of what it contains, and a clean run contains no
/// color.
pub fn print_summary(summary: &ScanSummary) {
    let p = palette();
    let flagged = summary.hostile > 0 || summary.suspicious > 0;
    let color = colored::control::SHOULD_COLORIZE.should_colorize();
    let bloom = crate::bloom_repo::counts();

    if flagged && color {
        // Cards end flush; the footer leads with its own separating blank.
        eprintln!();
        let accent = if summary.hostile > 0 {
            p.hostile
        } else {
            p.suspicious
        };
        // Colored segments paired with their plain text, so the rule fill can
        // be sized from real visible width.
        let mut parts: Vec<(String, String)> = Vec::new();
        let plain = |s: &str| (s.to_string(), s.to_string());
        parts.push(plain(&format!(
            "{} file{}",
            summary.total_files,
            if summary.total_files == 1 { "" } else { "s" }
        )));
        if summary.hostile > 0 {
            let s = format!("{} hostile", summary.hostile);
            parts.push((fg(p.hostile, &s), s));
        }
        if summary.suspicious > 0 {
            let s = format!("{} suspicious", summary.suspicious);
            parts.push((fg(p.suspicious, &s), s));
        }
        if summary.errors > 0 {
            let s = format!("{} errors", summary.errors);
            parts.push((fg(p.header_path, &s), s));
        }
        {
            let s = format!("{} clean", summary.benign);
            parts.push((fg(p.very_dim, &s), s));
        }
        if bloom.flagged > 0 {
            let s = format!("{} known-bad", bloom.flagged);
            parts.push((fg(p.hostile, &s), s));
        }
        if bloom.conflicted > 0 {
            let s = format!("{} conflicted", bloom.conflicted);
            parts.push((fg(p.warning, &s), s));
        }
        {
            let s = format_elapsed(summary.duration_ms);
            parts.push((fg(p.very_dim, &s), s));
        }
        let sep_colored = format!(" {} ", fg(p.dot_sep, "\u{00b7}"));
        let colored_line: Vec<String> = parts.iter().map(|(c, _)| c.clone()).collect();
        let plain_len: usize =
            parts.iter().map(|(_, s)| s.chars().count()).sum::<usize>() + (parts.len() - 1) * 3;
        // ` ` + `━━` + ` ` + content + ` ` + fill, 1-col right margin.
        let fill = cleave::output::terminal_width()
            .saturating_sub(plain_len + 6)
            .max(2);
        eprintln!(
            " {} {} {}",
            fg(accent, "\u{2501}\u{2501}"),
            colored_line.join(&sep_colored),
            fg(accent, &"\u{2501}".repeat(fill)),
        );
        eprintln!();
        return;
    }

    let line = fg(p.summary_line, &"\u{2500}".repeat(52));
    eprintln!(" {line}");

    eprintln!(" {}", summary_status_line(summary));
    eprintln!();
}

/// The one-line scan verdict — icon plus counts plus elapsed — without the
/// leading indent, surrounding separator, or trailing blank that
/// [`print_summary`] adds. Shared by the full filesystem-scan summary and
/// `ps`'s compact finish, which overwrites its progress bar with this line.
#[must_use]
pub fn summary_status_line(summary: &ScanSummary) -> String {
    let p = palette();

    if summary.total_files == 0 {
        let (icon, msg) = if summary.errors > 0 {
            (
                fg(p.warning, "!"),
                format!(
                    "no files scanned, {}",
                    fg(p.header_path, &format!("{} errors", summary.errors)),
                ),
            )
        } else {
            (fg(p.warning, "!"), "no scannable files found".to_string())
        };
        return format!(
            "{}  {}  {}",
            icon,
            msg,
            fg(p.very_dim, &format_elapsed(summary.duration_ms)),
        );
    }

    if summary.hostile == 0 && summary.suspicious == 0 && summary.errors == 0 {
        return format!(
            "{}  {} files scanned, all clean  {}",
            fg(p.benign, "\u{2713}"),
            summary.total_files,
            fg(p.very_dim, &format_elapsed(summary.duration_ms)),
        );
    }

    let mut parts = vec![format!(
        "{} file{}",
        summary.total_files,
        if summary.total_files == 1 { "" } else { "s" }
    )];
    if summary.hostile > 0 {
        parts.push(fg(p.hostile, &format!("{} hostile", summary.hostile)));
    }
    if summary.suspicious > 0 {
        parts.push(fg(
            p.suspicious,
            &format!("{} suspicious", summary.suspicious),
        ));
    }
    if summary.errors > 0 {
        parts.push(fg(p.header_path, &format!("{} errors", summary.errors)));
    }
    parts.push(fg(p.very_dim, &format!("{} clean", summary.benign)));
    parts.push(fg(p.very_dim, &format_elapsed(summary.duration_ms)));

    let sep = format!("  {}  ", fg(p.dot_sep, "\u{00b7}"));
    parts.join(&sep)
}

/// Print the scan banner shared by every scanning subcommand (`ps`, directory,
/// multi-path): the atom mark, version, and the two detection inventories already
/// resident in memory. SHA-256/PURL Bloom entries stay distinct from executable
/// trait/composite/YARA rules instead of inflating a misleading "rules" total.
pub(crate) fn print_banner(counts: crate::engine::DetectionCounts) {
    let p = palette();
    eprintln!();
    eprintln!(
        " \u{269b}\u{fe0f} {} {}",
        fg_bold(
            p.path_name,
            &format!("Atomdrift Scan v{}", env!("CARGO_PKG_VERSION"))
        ),
        fg(
            p.very_dim,
            &detection_inventory_text(counts.hashes_and_purls, counts.rules)
        ),
    );
}

fn detection_inventory_text(hashes_and_purls: u64, rules: u64) -> String {
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

/// Render `scan version`: the split signature/rule inventory, then one aligned
/// row per detection source with its count, version, and upstream age.
pub fn print_version(scan_version: &str, hashes_and_purls: u64, rules: u64, rows: &[SourceRow]) {
    let p = palette();
    let now = now_unix();

    println!();
    println!(
        " \u{269b}\u{fe0f} {} {}",
        fg_bold(p.path_name, &format!("Atomdrift Scan v{scan_version}")),
        fg(
            p.very_dim,
            &detection_inventory_text(hashes_and_purls, rules)
        ),
    );
    println!();

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
        println!(
            "   {}  {}  {}  {}  {}",
            fg(p.dim, &format!("{:<label_w$}", r.label)),
            fg(p.path_name, &format!("{metric:>metric_w$}")),
            fg(p.very_dim, &format!("{:<version_w$}", r.version)),
            fg(freshness, &format!("{date:<10}")),
            fg(freshness, &age),
        );
    }
    println!();
}

/// Current Unix time in whole seconds (UTC), or 0 if the clock predates the epoch.
fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_secs()).ok())
        .unwrap_or(0)
}

/// Format an upstream Unix timestamp as a UTC `YYYY-MM-DD` date.
fn fmt_date(epoch: i64) -> String {
    let (y, m, d) = civil_from_days(epoch.div_euclid(86_400));
    format!("{y:04}-{m:02}-{d:02}")
}

/// Render an age in seconds as a compact "just now / Nm / Nh / Nd ago" string.
fn fmt_age(secs: i64) -> String {
    let s = secs.max(0);
    if s < 60 {
        "just now".to_string()
    } else if s < 3600 {
        format!("{}m ago", s / 60)
    } else if s < 86_400 {
        format!("{}h ago", s / 3600)
    } else {
        format!("{}d ago", s / 86_400)
    }
}

/// Staleness color for a source's age: green within 48h, yellow within a week,
/// red beyond. Reuses the classification palette so freshness reads the same way
/// a verdict does.
fn freshness_color(secs: i64) -> Rgb {
    let p = palette();
    match secs.max(0) {
        ..=172_800 => p.benign,
        172_801..=604_800 => p.suspicious,
        _ => p.hostile,
    }
}

/// Convert days since the Unix epoch to `(year, month, day)` in the proleptic
/// Gregorian calendar (UTC). Howard Hinnant's `civil_from_days`.
// Casts are bounded by the algorithm: `d` ∈ [1,31] and `m` ∈ [1,12].
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (y + i64::from(m <= 2), m, d)
}

/// Parse a `YYYY-MM-DD` date as a UTC-midnight Unix timestamp (seconds).
/// Returns `None` for any malformed field. Inverse of `civil_from_days`
/// (Howard Hinnant's `days_from_civil`).
#[must_use]
pub fn parse_ymd(s: &str) -> Option<i64> {
    let mut parts = s.splitn(3, '-');
    let y: i64 = parts.next()?.trim().parse().ok()?;
    let m: i64 = parts.next()?.trim().parse().ok()?;
    let d: i64 = parts.next()?.trim().parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let y = y - i64::from(m <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * if m > 2 { m - 3 } else { m + 9 } + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    Some((era * 146_097 + doe - 719_468) * 86_400)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn bloom_mark_from_decision() {
        use crate::bloom_repo::Decision;
        // Skip (known-good) now marks a scanned-anyway file ✓; Unknown is
        // unremarkable (matched neither set) and carries no mark.
        assert_eq!(
            BloomMark::from_decision(Decision::Skip),
            Some(BloomMark::KnownGood)
        );
        assert_eq!(BloomMark::from_decision(Decision::Unknown), None);
        assert_eq!(
            BloomMark::from_decision(Decision::KnownBad),
            Some(BloomMark::KnownBad)
        );
        assert_eq!(
            BloomMark::from_decision(Decision::Conflicted),
            Some(BloomMark::Conflicted)
        );
        assert_eq!(BloomMark::KnownGood.tiny_str(), "known-good");
        assert_eq!(BloomMark::KnownBad.tiny_str(), "known-bad");
        assert_eq!(BloomMark::Conflicted.tiny_str(), "conflicted");
        // Each state is visually distinct.
        assert_ne!(BloomMark::KnownGood.glyph(), BloomMark::KnownBad.glyph());
        assert_ne!(BloomMark::KnownBad.glyph(), BloomMark::Conflicted.glyph());
        assert!(!BloomMark::KnownGood.forces_terminal_display());
        assert!(BloomMark::KnownBad.forces_terminal_display());
        assert!(BloomMark::Sighted.forces_terminal_display());
        assert!(BloomMark::Conflicted.forces_terminal_display());
    }

    #[test]
    fn hash_line_carries_the_bloom_glyph() {
        // The bloom verdict rides the hash line's glyph slot — 🚩 supplants 🧬
        // for a known-bad file — never the verdict badge. (These tests run
        // uncolored, so the line uses the plain text-marker form.)
        let plain = terminal_hash_line("deadbeef", None).expect("hash line");
        let flagged = terminal_hash_line("deadbeef", Some(BloomMark::KnownBad)).expect("hash line");
        assert!(plain.contains("deadbeef"));
        assert!(!plain.contains(BloomMark::KnownBad.glyph()));
        assert!(flagged.contains("deadbeef"));
        assert!(flagged.contains(BloomMark::KnownBad.glyph()));
        // An absent hash renders nothing.
        assert!(terminal_hash_line("", Some(BloomMark::KnownBad)).is_none());
        // The badge itself carries no bloom glyph.
        let (badge, _) = terminal_badge(&Classification::Hostile, 0.99, 0.65, Some(0));
        assert!(!badge.contains(BloomMark::KnownBad.glyph()));
    }

    #[test]
    fn human_size_compact_units() {
        assert_eq!(human_size(0), "");
        assert_eq!(human_size(352), "352B");
        assert_eq!(human_size(5 * 1024 + 400), "5.4KB");
        assert_eq!(human_size(44 * 1024 * 1024 / 10 * 10), "44.0MB");
    }

    #[test]
    fn card_markers_share_one_content_column() {
        for marker in [
            "📦",
            "📄",
            "🧬",
            "🚩",
            "📡",
            "🏴",
            "🪪",
            "✨",
            "●",
            "●●",
            "●●●",
        ] {
            let prefix = terminal_marker_prefix(1, marker);
            assert_eq!(
                UnicodeWidthStr::width(prefix.as_str()),
                5,
                "marker {marker:?} produced prefix {prefix:?}"
            );
        }
    }

    #[test]
    fn terminal_trait_grid_is_three_rows_and_keeps_location_tail() {
        let traits = [
            TerminalTrait {
                criticality: cleave::Criticality::Hostile,
                description: "Remote terminal agent".to_string(),
                location: "very-long-hash-named-member-that-needs-truncation.py:114".to_string(),
            },
            TerminalTrait {
                criticality: cleave::Criticality::Suspicious,
                description: "Hidden execution".to_string(),
                location: "runner.py:9".to_string(),
            },
            TerminalTrait {
                criticality: cleave::Criticality::Notable,
                description: "Third trait".to_string(),
                location: "manifest.json".to_string(),
            },
            TerminalTrait {
                criticality: cleave::Criticality::Notable,
                description: "Must not render".to_string(),
                location: "fourth.txt".to_string(),
            },
        ];
        let rendered = terminal_trait_rows(&traits, 72);
        assert_eq!(rendered.lines().count(), 3);
        assert!(rendered.lines().next().unwrap().contains("…"));
        assert!(rendered.lines().next().unwrap().ends_with("py:114"));
        assert!(!rendered.contains("Must not render"));
    }

    #[test]
    fn embedded_branch_uses_two_lines_without_a_closing_rule() {
        let body = " \u{25cf}\u{25cf}\u{25cf} Payload behavior";
        let rendered = terminal_embedded_branch(
            &Classification::Hostile,
            "embedded base64 @ 11096",
            "shell",
            11_766,
            body,
            true,
        );
        assert_eq!(rendered.lines().count(), 2);
        assert!(rendered.contains("   \u{2514}\u{2500} embedded base64 @ 11096"));
        assert!(rendered.contains("      \u{25cf}\u{25cf}\u{25cf} Payload behavior"));
        assert!(!rendered.contains('\u{2570}'));
    }

    #[test]
    fn commas_group_by_threes() {
        assert_eq!(with_commas(0), "0");
        assert_eq!(with_commas(42), "42");
        assert_eq!(with_commas(999), "999");
        assert_eq!(with_commas(1_000), "1,000");
        assert_eq!(with_commas(12_345), "12,345");
        assert_eq!(with_commas(123_456), "123,456");
        assert_eq!(with_commas(1_093_027), "1,093,027");
    }

    #[test]
    fn detection_inventory_keeps_signatures_separate_from_rules() {
        assert_eq!(
            detection_inventory_text(9_099_016, 120_680),
            "\u{00b7} 9,099,016 hashes \u{00b7} 120,680 rules"
        );
    }

    #[test]
    fn display_probability_at_threshold() {
        let thr = 0.65_f32;
        // At zero → 0%
        assert!((display_probability(0.0, thr) - 0.0).abs() < 1e-6);
        // At the deciding threshold → exactly 50%
        assert!((display_probability(thr, thr) - 0.5).abs() < 1e-6);
        // At 1.0 → 100%
        assert!((display_probability(1.0, thr) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn display_probability_monotonic() {
        let thr = 0.65_f32;
        let steps: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
        for pair in steps.windows(2) {
            assert!(
                display_probability(pair[1], thr) > display_probability(pair[0], thr),
                "not monotonic at {} -> {}",
                pair[0],
                pair[1],
            );
        }
    }

    #[test]
    fn display_probability_below_threshold_is_under_50() {
        let thr = 0.65_f32;
        assert!(display_probability(thr - 0.01, thr) < 0.5);
        assert!(display_probability(0.5, thr) < 0.5);
    }

    #[test]
    fn display_probability_uses_custom_threshold() {
        let thr = 0.80_f32;
        assert!((display_probability(thr, thr) - 0.5).abs() < 1e-6);
        assert!(display_probability(0.79, thr) < 0.5);
    }

    #[test]
    fn display_confidence_is_absent_for_unflagged_levels() {
        // Every level that is actually on the grid keeps its confidence: 0 is
        // the tightest and most certain, higher levels are looser and less so.
        assert_eq!(display_confidence(Some(0)), Some(100));
        assert_eq!(display_confidence(Some(5)), Some(95));
        // No level marker at all → no level-derived confidence. Unchanged.
        assert_eq!(display_confidence(None), None);
        // A negative level means the sample cleared no calibrated level. The
        // wire value stays 0 (consumers of the JSON `conf` field see no change)
        // but the display must not reuse it: 0% renders as certainty, and the
        // model expressed absence.
        assert_eq!(level_confidence(Some(-1)), Some(0));
        assert_eq!(display_confidence(Some(-1)), None);
        assert_eq!(display_confidence(Some(-7)), None);
    }

    #[test]
    fn unflagged_badge_reports_distance_to_cutoff_not_zero() {
        // Regression: an npm package whose raw probability was 0.998 against a
        // higher cutoff rendered as `BENIGN[0%]`, printed directly above
        // cleave's own hostile finding for the same file. Zero asserted a
        // certainty the model never expressed. The honest number is how near
        // the deciding cutoff the sample actually sat — here, right on it.
        //
        // Asserted with `contains` so the test holds in both the plain and the
        // colored badge form.
        let (unflagged, _) = terminal_badge(&Classification::Benign, 0.998, 0.999, Some(-1));
        assert!(
            unflagged.contains("BENIGN") && unflagged.contains("50%"),
            "unflagged badge should show the threshold-relative percentage, got {unflagged:?}"
        );

        // A level that IS on the grid still reports its calibrated confidence,
        // so the fix does not disturb the flagged path.
        let (flagged, _) = terminal_badge(&Classification::Hostile, 0.99, 0.65, Some(0));
        assert!(
            flagged.contains("HOSTILE") && flagged.contains("100%"),
            "flagged badge should keep the level-derived confidence, got {flagged:?}"
        );
    }

    #[test]
    fn parse_ymd_known_dates() {
        assert_eq!(parse_ymd("1970-01-01"), Some(0));
        assert_eq!(parse_ymd("2026-06-29"), Some(1_782_691_200));
        // Malformed input yields no timestamp rather than a wrong guess.
        assert_eq!(parse_ymd("2026-13-01"), None);
        assert_eq!(parse_ymd("not-a-date"), None);
        assert_eq!(parse_ymd("2026-06"), None);
    }

    #[test]
    fn date_round_trips_through_epoch() {
        for date in ["1970-01-01", "2000-02-29", "2026-06-29", "2099-12-31"] {
            let epoch = parse_ymd(date).expect("valid date");
            assert_eq!(fmt_date(epoch), date);
        }
    }

    #[test]
    fn freshness_bands_map_to_verdict_colors() {
        let p = palette();
        let same = |a: Rgb, b: Rgb| a.0 == b.0 && a.1 == b.1 && a.2 == b.2;
        assert!(same(freshness_color(0), p.benign));
        assert!(same(freshness_color(172_800), p.benign)); // exactly 48h → still green
        assert!(same(freshness_color(172_801), p.suspicious)); // just past 48h → yellow
        assert!(same(freshness_color(604_800), p.suspicious)); // exactly a week → yellow
        assert!(same(freshness_color(604_801), p.hostile)); // past a week → red
    }

    #[test]
    fn age_formats_by_unit() {
        assert_eq!(fmt_age(-5), "just now");
        assert_eq!(fmt_age(30), "just now");
        assert_eq!(fmt_age(90), "1m ago");
        assert_eq!(fmt_age(3 * 3600), "3h ago");
        assert_eq!(fmt_age(2 * 86_400 + 5 * 3600), "2d ago");
    }
}
