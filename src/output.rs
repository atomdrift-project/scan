//! Terminal and JSON output formatting with litmus-colored confidence indicators.
//! Supports both dark and light terminal backgrounds via auto-detection.

use std::sync::{LazyLock, RwLock};

use crate::OutputFormat;
use crate::engine::{ScanResult, ScanSummary, level_confidence};
use crate::model::{Classification, RouteScore};

const BLOCK: &str = "\u{2588}";

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
    header_icon: Rgb,
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
            header_icon: Rgb(160, 160, 160),
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
            header_icon: Rgb(110, 110, 110),
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
    // SAFETY: isatty() is a trivial syscall with no preconditions.
    unsafe { libc::isatty(libc::STDERR_FILENO) != 1 } {
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

/// Two-block litmus confidence indicator.
///
/// The colors shift across the band itself, so threshold-adjacent results
/// stay visually distinct from high-confidence results in the same class.
fn confidence_blocks(probability: f32, classification: &Classification, threshold: f32) -> String {
    let (left, right, _) = indicator_colors(probability, classification, threshold);
    format!("{}{}", fg(left, BLOCK), fg(right, BLOCK))
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
    match level_confidence(level) {
        Some(conf) => {
            let (_, _, accent) = indicator_colors(probability, classification, threshold);
            fg(accent, &format!("{:>4}", format!("{conf}%")))
        }
        None => colored_pct(probability, classification, threshold),
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

/// litmus's verdict stamp that leads cleave's headline: the band percentage on
/// a filled, classification-colored background — ` 100% ` (red hostile, orange
/// suspicious, green benign) — so the color alone carries the verdict and the
/// word is unnecessary. Plain `100%` when color is disabled (piped). Returns the
/// stamp and its visible width (for aligning lines beneath it).
pub(crate) fn terminal_badge(
    classification: &Classification,
    probability: f32,
    threshold: f32,
    level: Option<i32>,
) -> (String, usize) {
    let pct = level_confidence(level).map_or_else(
        || {
            format!(
                "{:.0}%",
                display_probability(probability, threshold) * 100.0
            )
        },
        |conf| format!("{conf}%"),
    );
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        let width = pct.chars().count();
        return (pct, width);
    }
    // Bold white on the band color, padded with a space either side — so the
    // visible width is the percentage plus the two padding spaces.
    let Rgb(r, g, b) = indicator_colors(probability, classification, threshold).2;
    let width = pct.chars().count() + 2;
    (
        format!("\x1b[1;38;2;255;255;255;48;2;{r};{g};{b}m {pct} \x1b[0m"),
        width,
    )
}

/// The one-line summary that trails the file type on the headline: `· <text>`.
/// `--interpret` data supersedes the model's own reasoning when present;
/// otherwise the SHAP reasons. `None` when neither exists.
pub(crate) fn terminal_trailer(
    reasons: &[crate::explain::Reason],
    interpretation: Option<&crate::interpret::Interpretation>,
) -> Option<String> {
    let color = colored::control::SHOULD_COLORIZE.should_colorize();
    let (body, review) = if let Some(llm) = interpretation {
        let body = llm.error.as_deref().map_or_else(
            || llm.interpretation.clone(),
            |_| "interpretation unavailable".into(),
        );
        (body, llm.review)
    } else if reasons.is_empty() {
        return None;
    } else {
        let joined = reasons
            .iter()
            .take(3)
            .map(|r| r.description.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        (joined, false)
    };
    let flag = if review { "  \u{26a0} review" } else { "" };
    if color {
        let p = palette();
        Some(format!(
            "{} {}{}",
            fg(p.dim, "\u{00b7}"),
            fg(p.reason, &body),
            flag
        ))
    } else {
        Some(format!("\u{00b7} {body}{flag}"))
    }
}

/// The line beneath the headline: the full SHA-256 (copy-able for lookups), no
/// label, indented to align under the filename. `None` when absent.
pub(crate) fn terminal_subtitle(sha256: &str, indent: usize) -> Option<String> {
    if sha256.is_empty() {
        return None;
    }
    let hash = if colored::control::SHOULD_COLORIZE.should_colorize() {
        fg(palette().dim, sha256)
    } else {
        sha256.to_string()
    };
    Some(format!("{}{hash}", " ".repeat(indent)))
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
    let blocks = confidence_blocks(result.probability, &result.classification, result.threshold);
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
        "  {blocks} {pct} {label}  {}{deleted_marker}  {pid_display}",
        fg_bold(p.path_name, &result.path),
    );

    if let Some(llm) = &result.interpretation {
        eprint!("    {}", crate::engine::format_llm_line(llm, true));
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
    eprintln!("           {}", meta.join(&format!(" {dot} ")));

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
        eprintln!("           {}", findings.join(&format!(" {dot} ")));
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
            "           {} {}",
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
fn format_level(level: Option<i32>) -> String {
    level.map_or_else(|| "\u{2014}".to_string(), |n| format!("L{n}"))
}

/// Print the matched level, raw route scores, and SHAP feature values
/// (hidden --extra mode). Leads with the level — the operating point a file
/// maps to is the actionable signal; the calibrated probability is not shown
/// because isotonic saturation collapses it to 1.0 across the whole upper tail.
fn print_extra(result: &ScanResult, p: &Palette) {
    eprintln!(
        "           {} {}",
        fg(p.dim, "level:"),
        fg(p.dim, &format_level(result.level)),
    );
    if !result.model_scores.is_empty() {
        eprintln!(
            "           {} {}",
            fg(p.dim, "models:"),
            fg(p.dim, &format_route_scores(&result.model_scores)),
        );
    }
    // Surface the routing of archive members. A container often inherits a
    // member's verdict (see `elevated archive classification`), so when the
    // container's own models disagree with the final grade, the responsible
    // route lives here — not above.
    for ef in &result.embedded_files {
        if ef.model_scores.is_empty() {
            continue;
        }
        eprintln!(
            "           {} {} {} {} {}",
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
            "           {} {}",
            fg(p.dim, "skipped:"),
            fg(p.dim, &skipped.join(" "))
        );
    }
    if !result.reasons.is_empty() {
        eprintln!("           {}", fg(p.dim, "shap:"));
        for r in &result.reasons {
            eprintln!(
                "             {} {} {}",
                fg(p.dim, &format!("{:>8.4}", r.importance)),
                fg(p.dim, &format!("val={:<8.2}", r.value)),
                fg(p.very_dim, &r.feature),
            );
        }
    }
}

/// Print the scan header.
pub fn print_header(path: &std::path::Path, count: usize) {
    let p = palette();
    eprintln!();
    eprintln!(
        "  {}  {} files in {}",
        fg(p.header_icon, "\u{25c6}"),
        count,
        fg(p.header_path, &path.display().to_string()),
    );
    eprintln!();
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
        BloomVerdict::KnownGood => {
            (fg(p.benign, "\u{2713}"), fg(p.benign, "known-good"), "skipping scan")
        }
        BloomVerdict::KnownBad => {
            (fg(p.hostile, "\u{2717}"), fg(p.hostile, "known-bad"), "flagged — scanning")
        }
        BloomVerdict::Conflicted => (
            fg(p.warning, "\u{26a0}"),
            fg(p.warning, "conflicted"),
            "in good and bad — scanning",
        ),
        BloomVerdict::Unscanned => {
            (fg(p.warning, "!"), fg(p.very_dim, "unscanned"), "not in known sets (fast mode)")
        }
    };
    eprintln!(
        "  {icon}  {name} {}  {}",
        fg(p.very_dim, reason),
        fg(p.header_path, label),
    );
}

/// Print scan summary.
pub fn print_summary(summary: &ScanSummary) {
    let p = palette();
    let line = fg(p.summary_line, &"\u{2500}".repeat(52));
    eprintln!("  {line}");

    // Bloom decision tally (only when filters were consulted this run).
    let bloom = crate::bloom_repo::counts();
    if !bloom.is_empty() {
        let mut parts = Vec::new();
        if bloom.skipped > 0 {
            parts.push(fg(p.benign, &format!("{} known-good", bloom.skipped)));
        }
        if bloom.flagged > 0 {
            parts.push(fg(p.hostile, &format!("{} known-bad", bloom.flagged)));
        }
        if bloom.conflicted > 0 {
            parts.push(fg(p.warning, &format!("{} conflicted", bloom.conflicted)));
        }
        if bloom.unscanned > 0 {
            parts.push(fg(p.very_dim, &format!("{} unscanned", bloom.unscanned)));
        }
        let sep = format!("  {}  ", fg(p.dot_sep, "\u{00b7}"));
        eprintln!("  {}  bloom  {}", fg(p.header_icon, "\u{25c6}"), parts.join(&sep));
    }

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
        eprintln!(
            "  {}  {}  {}",
            icon,
            msg,
            fg(p.very_dim, &format_elapsed(summary.duration_ms)),
        );
        eprintln!();
        return;
    }

    if summary.hostile == 0 && summary.suspicious == 0 && summary.errors == 0 {
        eprintln!(
            "  {}  {} files scanned, all clean  {}",
            fg(p.benign, "\u{2713}"),
            summary.total_files,
            fg(p.very_dim, &format_elapsed(summary.duration_ms)),
        );
        eprintln!();
        return;
    }

    let mut parts = vec![format!("{} files", summary.total_files)];
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
    eprintln!("  {}", parts.join(&sep));
    eprintln!();
}

fn format_elapsed(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
