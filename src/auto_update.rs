//! Default-on, quiet refresh of detection rules and models when they go stale.
//!
//! On an interactive scan, if the local ruleset hasn't been checked in 24 hours
//! (and `--no-update` / `SCAN_NO_UPDATE` is not set), scan refreshes the cleave
//! trait bundle and the ML model bundle in the background. The check is gated by
//! a tiny on-disk stamp, so the common case touches neither the network nor the
//! terminal: most runs return in microseconds having done nothing.
//!
//! When a refresh *does* run, the update calls are silenced (their own progress
//! lines would be noise) and, on a terminal, a transient "updating rules…" line
//! is shown. If new rules were actually installed it resolves into a one-line
//! checkmark with the current rule count and the newest ruleset date; if nothing
//! changed the line simply vanishes — the mechanism stays invisible unless it
//! had something to report.

use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::Mode;

/// Refresh rules + models once per day. Matches the network-check cadence of the
/// version notice ([`crate::update_check`]).
const CHECK_INTERVAL_SECS: u64 = 24 * 60 * 60;

/// On-disk stamp of the last refresh attempt (success *or* failure), so an
/// unreachable bucket doesn't trigger a network call on every run.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Stamp {
    /// Unix time the refresh was last attempted.
    checked_unix: u64,
}

/// Path to the stamp (`<cache_dir>/atomdrift/scan/rules-update.toml`).
fn stamp_path() -> Option<PathBuf> {
    Some(
        dirs::cache_dir()?
            .join("atomdrift")
            .join("scan")
            .join("rules-update.toml"),
    )
}

/// Current Unix time in seconds (0 if the clock predates the epoch).
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Unix time of the last refresh attempt (0 when never stamped or unreadable).
fn last_checked() -> u64 {
    let Some(path) = stamp_path() else { return 0 };
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| toml::from_str::<Stamp>(&text).ok())
        .map_or(0, |stamp| stamp.checked_unix)
}

/// Stamp the current time (best-effort; failures are non-fatal).
fn write_stamp() {
    let Some(path) = stamp_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(text) = toml::to_string(&Stamp {
        checked_unix: now_unix(),
    }) {
        let _ = std::fs::write(&path, text);
    }
}

/// Refresh the trait and model bundles when they are stale, by default.
///
/// `force` (from `-u/--update`) bypasses the 24h gate. `disabled` (from
/// `--no-update`) — like `SCAN_NO_UPDATE` — turns the whole mechanism off.
/// `mode` matches the run's bloom usage so the reported rule count equals the
/// banner's. `terminal` gates the transient/checkmark UI: a non-terminal run
/// still refreshes, silently.
///
/// Always best-effort: network or update failures are logged at debug level and
/// never block or fail the scan. Stamps every attempt so a stale or offline
/// bucket can't cause a refresh storm.
pub fn refresh_if_stale(force: bool, disabled: bool, mode: Mode, terminal: bool) {
    if disabled || std::env::var_os("SCAN_NO_UPDATE").is_some() {
        return;
    }
    if !force && now_unix().saturating_sub(last_checked()) < CHECK_INTERVAL_SECS {
        return;
    }

    if terminal {
        eprint!(" \x1b[38;2;120;120;120m\u{27f3} updating rules\u{2026}\x1b[0m");
        let _ = std::io::stderr().flush();
    }

    let outcome = refresh();
    // Always stamp the attempt — success *or* failure — so an offline host
    // records that it tried and won't re-check (and re-time-out) every run; the
    // next attempt is a day away, or immediate with `-u`/`update-rules`.
    write_stamp();

    if !terminal {
        return;
    }
    // Resolve the transient line:
    //   - a real update earns a checkmark;
    //   - a reachable, already-current ruleset leaves no trace (invisible);
    //   - an unreachable bucket gets one actionable line, at most once a day.
    if outcome.changed {
        print_checkmark(mode);
    } else if outcome.failed {
        eprint!(
            "\r\x1b[2K \x1b[38;2;255;175;55m\u{26a0}\x1b[0m \x1b[38;2;160;160;160mauto-update timed out \u{2014} run `atomscan update-rules` to update\x1b[0m\n",
        );
        let _ = std::io::stderr().flush();
    } else {
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }
}

/// Whether a refresh installed anything new, and whether any of the three
/// bundles failed to reach the server.
struct Outcome {
    changed: bool,
    failed: bool,
}

/// Refresh the three bundle types — traits, models, bloom filters — concurrently
/// and quietly. Each runs on its own thread so the slow path is one network
/// round-trip, not three; a failure of one (offline, mirror error) is recorded
/// and never blocks the others. This is the same per-type `update()` machinery
/// that first-run auto-install uses — staleness, rather than absence, is the only
/// difference in trigger.
fn refresh() -> Outcome {
    let models_dir = crate::models_repo::install_target();
    let bloom_dir = crate::bloom_repo::install_dir();
    let model_before = crate::model_update::installed(&models_dir).map(|i| i.commit);

    let mut changed = false;
    let mut failed = false;
    std::thread::scope(|s| {
        let traits = s.spawn(|| crate::traits_repo::update(false, true));
        let models = s.spawn(|| crate::model_update::update(&models_dir, false, true));
        let bloom = s.spawn(|| crate::bloom_update::update(&bloom_dir, false, true));

        match traits.join() {
            Ok(Ok(c)) => changed |= c,
            Ok(Err(e)) => {
                failed = true;
                tracing::debug!(error = %format!("{e:#}"), "auto traits update failed");
            }
            Err(_) => {
                failed = true;
                tracing::debug!("auto traits update panicked");
            }
        }
        match models.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                failed = true;
                tracing::debug!(error = %format!("{e:#}"), "auto model update failed");
            }
            Err(_) => {
                failed = true;
                tracing::debug!("auto model update panicked");
            }
        }
        match bloom.join() {
            Ok(Ok(c)) => changed |= c,
            Ok(Err(e)) => {
                failed = true;
                tracing::debug!(error = %format!("{e:#}"), "auto bloom update failed");
            }
            Err(_) => {
                failed = true;
                tracing::debug!("auto bloom update panicked");
            }
        }
    });

    let model_after = crate::model_update::installed(&models_dir).map(|i| i.commit);
    changed |= model_before != model_after;
    Outcome { changed, failed }
}

/// Erase the transient line and print the update checkmark: the current split
/// signature/rule inventory and newest trait/model bundle date.
fn print_checkmark(mode: Mode) {
    let hashes_and_purls = if mode == Mode::Slow {
        0
    } else {
        crate::bloom_repo::Lookup::load().rule_count()
    };
    let counts = crate::engine::detection_counts_from(hashes_and_purls);

    let traits_dir = cleave::traits_repo::install_target();
    let models_dir = crate::models_repo::install_target();
    let newest = [
        cleave::rule_update::installed(&traits_dir).map(|i| i.date),
        crate::model_update::installed(&models_dir).map(|i| i.date),
    ]
    .into_iter()
    .flatten()
    .max();

    // The date is omitted only if neither bundle reports one, which a successful
    // install should never do.
    let tail = match &newest {
        Some(date) => format!(" \u{b7} newest {date}"),
        None => String::new(),
    };
    eprint!(
        "\r\x1b[2K \x1b[38;2;80;220;80m\u{2713}\x1b[0m \x1b[38;2;160;160;160mupdated \u{b7} {} hashes \u{b7} {} rules{tail}\x1b[0m\n",
        commas(counts.hashes_and_purls),
        commas(counts.rules),
    );
    let _ = std::io::stderr().flush();
}

/// Format an integer with `,` thousands separators (`1093027` → `1,093,027`),
/// matching the banner's rule-count rendering.
fn commas(n: u64) -> String {
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
