//! Terminal and JSON output formatting with litmus-colored confidence indicators.

use colored::Colorize;

use crate::model::Classification;
use crate::scan::{ScanResult, ScanSummary};

const BLOCK: &str = "\u{2588}";

/// Set truecolor foreground on text.
fn fg(r: u8, g: u8, b: u8, text: &str) -> String {
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
    match classification {
        Classification::Hostile => {
            // Band: 0.85 .. 1.0 -> t: 0.0 .. 1.0
            let t = ((probability - 0.85) / 0.15).clamp(0.0, 1.0);
            // Left: dark red (#640000) -> bright red (#FF0000)
            let l = fg(lerp(100, 255, t), 0, 0, BLOCK);
            // Right: pink (#FF9696) -> bright red (#FF0000)
            let r = fg(255, lerp(150, 0, t), lerp(150, 0, t), BLOCK);
            format!("{l}{r}")
        }
        Classification::Suspicious => {
            // Band: 0.75 .. 0.85 -> t: 0.0 .. 1.0
            let t = ((probability - 0.75) / 0.10).clamp(0.0, 1.0);
            // Left: muted amber (#B46400) -> bright amber (#FFA000)
            let l = fg(lerp(180, 255, t), lerp(100, 160, t), 0, BLOCK);
            // Right: pale orange (#FFB450) -> bright amber (#FF8C00)
            let r = fg(255, lerp(180, 140, t), lerp(80, 0, t), BLOCK);
            format!("{l}{r}")
        }
        Classification::Benign => {
            let t = (probability / 0.75).clamp(0.0, 1.0);
            let l = fg(0, lerp(100, 180, t), 0, BLOCK);
            let r = fg(0, lerp(80, 160, t), 0, BLOCK);
            format!("{l}{r}")
        }
    }
}

/// Color percentage text to match the classification band.
fn colored_pct(probability: f32, classification: &Classification) -> String {
    let pct = format!("{:>4}", format!("{:.0}%", probability * 100.0));
    match classification {
        Classification::Hostile => fg(255, 70, 70, &pct),
        Classification::Suspicious => fg(255, 175, 55, &pct),
        Classification::Benign => fg(80, 200, 80, &pct),
    }
}

/// Print a single file result immediately, clearing progress line if active.
pub fn print_file_result_streaming(result: &ScanResult, has_progress: bool, verbose: bool) {
    if has_progress {
        eprint!("\r\x1b[2K");
    }

    let blocks = confidence_blocks(result.probability, &result.classification);
    let pct = colored_pct(result.probability, &result.classification);
    let vt_link = format!(
        "\x1b]8;;https://www.virustotal.com/gui/file/{sha}\x1b\\{sha}\x1b]8;;\x1b\\",
        sha = result.sha256,
    );
    let mut meta = format!(
        "{} \u{00b7} {} \u{00b7} {}",
        result.file_type,
        format_size(result.size_bytes),
        vt_link,
    );
    if !result.formula.is_empty() {
        meta.push_str(&format!(" \u{00b7} {}", result.formula));
    }

    eprintln!("  {blocks} {pct}  {}  {}", result.path.bold(), meta.bright_black());

    if verbose {
        let fc = &result.finding_counts;
        let h = if fc.hostile > 0 {
            fg(255, 80, 80, &format!("h:{}", fc.hostile))
        } else {
            fg(60, 60, 60, &format!("h:{}", fc.hostile))
        };
        let s = if fc.suspicious > 0 {
            fg(255, 175, 55, &format!("s:{}", fc.suspicious))
        } else {
            fg(60, 60, 60, &format!("s:{}", fc.suspicious))
        };
        let n = if fc.notable > 0 {
            fg(180, 180, 100, &format!("n:{}", fc.notable))
        } else {
            fg(60, 60, 60, &format!("n:{}", fc.notable))
        };
        let b = fg(60, 60, 60, &format!("b:{}", fc.baseline));
        let dot = fg(50, 50, 50, "\u{00b7}");
        let p = fg(100, 100, 100, &format!("p={:.4}", result.probability));
        eprintln!("           {h}  {s}  {n}  {b}  {dot}  {p}");
    }

    if !result.top_findings.is_empty() {
        let findings: Vec<String> = result
            .top_findings
            .iter()
            .take(4)
            .map(|f| short_finding_id(&f.id))
            .collect();
        let line = findings.join(&format!(" {} ", fg(80, 80, 80, "\u{00b7}")));
        let colored = match result.classification {
            Classification::Hostile => fg(255, 100, 100, &line),
            Classification::Suspicious => fg(255, 190, 90, &line),
            Classification::Benign => line,
        };
        eprintln!("           {colored}");
    }

    if !result.reasons.is_empty() {
        let reason_strs: Vec<&str> = result
            .reasons
            .iter()
            .take(3)
            .map(|r| r.description.as_str())
            .collect();
        eprintln!(
            "           {} {}",
            fg(70, 70, 70, "\u{2191}"),
            fg(100, 100, 100, &reason_strs.join(", ")),
        );
    }

    eprintln!();
}

/// Print a process scan result with PID annotations.
pub fn print_ps_result(result: &ScanResult, pids: &[u32], deleted: bool, verbose: bool) {
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
    let pid_display = fg(100, 100, 100, &format!("[pids: {pid_str}]"));

    let deleted_marker = if deleted {
        format!(" {}", fg(255, 100, 100, "(deleted)"))
    } else {
        String::new()
    };

    let vt_link = format!(
        "\x1b]8;;https://www.virustotal.com/gui/file/{sha}\x1b\\{sha}\x1b]8;;\x1b\\",
        sha = result.sha256,
    );
    let mut meta = format!(
        "{} \u{00b7} {} \u{00b7} {}",
        result.file_type,
        format_size(result.size_bytes),
        vt_link,
    );
    if !result.formula.is_empty() {
        meta.push_str(&format!(" \u{00b7} {}", result.formula));
    }

    eprintln!(
        "  {blocks} {pct}  {}{deleted_marker}  {pid_display}  {}",
        result.path.bold(),
        meta.bright_black(),
    );

    if verbose {
        let fc = &result.finding_counts;
        let h = if fc.hostile > 0 {
            fg(255, 80, 80, &format!("h:{}", fc.hostile))
        } else {
            fg(60, 60, 60, &format!("h:{}", fc.hostile))
        };
        let s = if fc.suspicious > 0 {
            fg(255, 175, 55, &format!("s:{}", fc.suspicious))
        } else {
            fg(60, 60, 60, &format!("s:{}", fc.suspicious))
        };
        let n = if fc.notable > 0 {
            fg(180, 180, 100, &format!("n:{}", fc.notable))
        } else {
            fg(60, 60, 60, &format!("n:{}", fc.notable))
        };
        let b = fg(60, 60, 60, &format!("b:{}", fc.baseline));
        let dot = fg(50, 50, 50, "\u{00b7}");
        let p = fg(100, 100, 100, &format!("p={:.4}", result.probability));
        eprintln!("           {h}  {s}  {n}  {b}  {dot}  {p}");
    }

    if !result.top_findings.is_empty() {
        let findings: Vec<String> = result
            .top_findings
            .iter()
            .take(4)
            .map(|f| short_finding_id(&f.id))
            .collect();
        let line = findings.join(&format!(" {} ", fg(80, 80, 80, "\u{00b7}")));
        let colored = match result.classification {
            Classification::Hostile => fg(255, 100, 100, &line),
            Classification::Suspicious => fg(255, 190, 90, &line),
            Classification::Benign => line,
        };
        eprintln!("           {colored}");
    }

    if !result.reasons.is_empty() {
        let reason_strs: Vec<&str> = result
            .reasons
            .iter()
            .take(3)
            .map(|r| r.description.as_str())
            .collect();
        eprintln!(
            "           {} {}",
            fg(70, 70, 70, "\u{2191}"),
            fg(100, 100, 100, &reason_strs.join(", ")),
        );
    }

    eprintln!();
}

/// Print the scan header.
pub fn print_header(path: &std::path::Path, count: usize) {
    eprintln!();
    eprintln!(
        "  {}  {} files in {}",
        fg(120, 180, 255, "\u{25c6}"),
        count,
        fg(180, 180, 180, &path.display().to_string()),
    );
    eprintln!();
}

/// Print scan summary.
pub fn print_summary(summary: &ScanSummary) {
    if summary.total_files == 0 {
        return;
    }

    let line = fg(50, 50, 50, &"\u{2500}".repeat(52));
    eprintln!("  {line}");

    if summary.hostile == 0 && summary.suspicious == 0 {
        eprintln!(
            "  {}  {} files scanned, all clean  {}",
            fg(80, 220, 80, "\u{2713}"),
            summary.total_files,
            fg(80, 80, 80, &format_elapsed(summary.duration_ms)),
        );
        eprintln!();
        return;
    }

    let mut parts = vec![format!("{} files", summary.total_files)];
    if summary.hostile > 0 {
        parts.push(fg(255, 70, 70, &format!("{} hostile", summary.hostile)));
    }
    if summary.suspicious > 0 {
        parts.push(fg(255, 175, 55, &format!("{} suspicious", summary.suspicious)));
    }
    if summary.errors > 0 {
        parts.push(fg(180, 180, 180, &format!("{} errors", summary.errors)));
    }
    parts.push(fg(80, 80, 80, &format!("{} clean", summary.benign)));
    parts.push(fg(80, 80, 80, &format_elapsed(summary.duration_ms)));

    let sep = format!("  {}  ", fg(50, 50, 50, "\u{00b7}"));
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

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn format_elapsed(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}
