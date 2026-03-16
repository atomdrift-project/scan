//! Terminal and JSON output formatting with litmus-colored confidence indicators.
//! Supports both dark and light terminal backgrounds via auto-detection.

use std::sync::OnceLock;

use colored::Colorize;

use crate::model::{Classification, Thresholds};
use crate::scan::{ScanResult, ScanSummary};

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

            filetype: Rgb(120, 160, 220),
            formula: Rgb(80, 160, 140),

            dot_sep: Rgb(50, 50, 50),
            dim: Rgb(100, 100, 100),
            very_dim: Rgb(80, 80, 80),
            header_icon: Rgb(120, 180, 255),
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

            filetype: Rgb(40, 90, 160),
            formula: Rgb(30, 120, 100),

            dot_sep: Rgb(180, 180, 180),
            dim: Rgb(120, 120, 120),
            very_dim: Rgb(150, 150, 150),
            header_icon: Rgb(40, 100, 200),
            header_path: Rgb(80, 80, 80),
            summary_line: Rgb(190, 190, 190),
            warning: Rgb(140, 140, 0),
            arrow: Rgb(160, 160, 160),
            reason: Rgb(100, 100, 100),
        }
    }
}

/// Globally cached theme detection result.
static THEME: OnceLock<Theme> = OnceLock::new();

/// Detect the terminal theme, with env var override.
///
/// Priority: `LITMUS_THEME` env var > terminal query > default (dark).
pub fn detect_theme() -> Theme {
    *THEME.get_or_init(|| {
        // Check env override first.
        if let Ok(val) = std::env::var("LITMUS_THEME") {
            return match val.to_ascii_lowercase().as_str() {
                "light" | "white" => Theme::Light,
                _ => Theme::Dark,
            };
        }

        // Query terminal via OSC 11 + COLORFGBG fallback.
        match terminal_colorsaurus::color_scheme(terminal_colorsaurus::QueryOptions::default()) {
            Ok(scheme) => match scheme {
                terminal_colorsaurus::ColorScheme::Dark => Theme::Dark,
                terminal_colorsaurus::ColorScheme::Light => Theme::Light,
            },
            Err(_) => Theme::Dark,
        }
    })
}

/// Override the theme (called from CLI flags before any output).
pub fn set_theme(theme: Theme) {
    let _ = THEME.set(theme);
}

fn palette() -> &'static Palette {
    static DARK: Palette = Palette::dark();
    static LIGHT: Palette = Palette::light();

    match THEME.get().copied().unwrap_or(Theme::Dark) {
        Theme::Dark => &DARK,
        Theme::Light => &LIGHT,
    }
}

/// Set truecolor foreground on text.
fn fg(Rgb(r, g, b): Rgb, text: &str) -> String {
    format!("\x1b[38;2;{r};{g};{b}m{text}\x1b[0m")
}

/// Linear interpolation between two u8 values.
fn lerp(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round().clamp(0.0, 255.0) as u8
}

/// Two-block litmus confidence indicator.
///
/// At maximum confidence both blocks are identical (pure saturated color).
/// As confidence decreases within a classification band, the left block
/// darkens and the right block lightens/desaturates, creating a visual
/// "spread" that intuitively conveys uncertainty.
fn confidence_blocks(probability: f32, classification: &Classification) -> String {
    let p = palette();
    let thresh = Thresholds::default();
    let hostile_range = 1.0 - thresh.hostile;
    let suspicious_range = thresh.hostile - thresh.suspicious;
    match classification {
        Classification::Hostile => {
            let t = ((probability - thresh.hostile) / hostile_range).clamp(0.0, 1.0);
            let base = p.hostile;
            let l = fg(Rgb(lerp(base.0 / 2, base.0, t), 0, 0), BLOCK);
            let r = fg(
                Rgb(base.0, lerp(150, 0, t), lerp(150, 0, t)),
                BLOCK,
            );
            format!("{l}{r}")
        }
        Classification::Suspicious => {
            let t = ((probability - thresh.suspicious) / suspicious_range).clamp(0.0, 1.0);
            let base = p.suspicious;
            let l = fg(
                Rgb(lerp((base.0 as u16 * 3 / 4) as u8, base.0, t), lerp(base.1 / 2, base.1, t), 0),
                BLOCK,
            );
            let r = fg(
                Rgb(base.0, lerp(base.1 + 30, base.1, t), lerp(80, 0, t)),
                BLOCK,
            );
            format!("{l}{r}")
        }
        Classification::Benign => {
            let t = (probability / thresh.suspicious).clamp(0.0, 1.0);
            let base = p.benign;
            let l = fg(Rgb(0, lerp(base.1 / 2, base.1, t), 0), BLOCK);
            let r = fg(Rgb(0, lerp(base.1 / 3, (base.1 as u16 * 4 / 5) as u8, t), 0), BLOCK);
            format!("{l}{r}")
        }
    }
}

/// Color percentage text to match the classification band.
fn colored_pct(probability: f32, classification: &Classification) -> String {
    let p = palette();
    let pct = format!("{:>4}", format!("{:.0}%", probability * 100.0));
    match classification {
        Classification::Hostile => fg(p.hostile, &pct),
        Classification::Suspicious => fg(p.suspicious, &pct),
        Classification::Benign => fg(p.benign, &pct),
    }
}

/// Print a single file result immediately, clearing progress line if active.
pub fn print_file_result_streaming(result: &ScanResult, has_progress: bool) {
    if has_progress {
        eprint!("\r\x1b[2K");
    }

    let p = palette();
    let blocks = confidence_blocks(result.probability, &result.classification);
    let pct = colored_pct(result.probability, &result.classification);

    eprintln!("  {blocks} {pct}  {}", result.path.bold());
    print_detail_lines(result, p);
    print_reasons(result, p);
    eprintln!();
}

/// Print a process scan result with PID annotations.
pub fn print_ps_result(result: &ScanResult, pids: &[u32], deleted: bool) {
    let p = palette();
    let blocks = confidence_blocks(result.probability, &result.classification);
    let pct = colored_pct(result.probability, &result.classification);

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

    let deleted_marker = if deleted {
        format!(" {}", fg(p.hostile_finding, "(deleted)"))
    } else {
        String::new()
    };

    eprintln!(
        "  {blocks} {pct}  {}{deleted_marker}  {pid_display}",
        result.path.bold(),
    );

    print_detail_lines(result, p);
    print_reasons(result, p);
    eprintln!();
}

/// Print metadata line (filetype + formula) and findings line.
fn print_detail_lines(result: &ScanResult, p: &Palette) {
    let dot = fg(p.dot_sep, "\u{00b7}");

    // Line 2: filetype · formula
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
                let name = short_finding_id(&f.id);
                match result.classification {
                    Classification::Hostile => fg(p.hostile_finding, &name),
                    Classification::Suspicious => fg(p.suspicious_finding, &name),
                    Classification::Benign => name,
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

/// Print scan summary.
pub fn print_summary(summary: &ScanSummary) {
    let p = palette();
    let line = fg(p.summary_line, &"\u{2500}".repeat(52));
    eprintln!("  {line}");

    if summary.total_files == 0 {
        eprintln!(
            "  {}  no scannable files found  {}",
            fg(p.warning, "!"),
            fg(p.very_dim, &format_elapsed(summary.duration_ms)),
        );
        eprintln!();
        return;
    }

    if summary.hostile == 0 && summary.suspicious == 0 {
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

/// Shorten finding IDs by stripping namespace prefixes and variants.
fn short_finding_id(id: &str) -> String {
    let base = id.split("::").next().unwrap_or(id);
    base.strip_prefix("objectives/")
        .or_else(|| base.strip_prefix("micro-behaviors/"))
        .or_else(|| base.strip_prefix("well-known/"))
        .or_else(|| base.strip_prefix("metadata/"))
        .unwrap_or(base)
        .to_string()
}

fn format_elapsed(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}
