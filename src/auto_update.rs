//! Default-on, quiet refresh of detection rules and models when they go stale.
//!
//! On an interactive scan, if the local ruleset hasn't been checked in 24 hours
//! (and `--no-update` is not given), scan refreshes the cleave
//! trait bundle, the ML model bundle, and the bloom filters before it starts.
//! The check is gated by a tiny on-disk stamp, so the common case touches
//! neither the network nor the terminal: most runs return in microseconds
//! having done nothing.
//!
//! When a refresh *does* run, the update calls are silenced (their own progress
//! lines would be noise) and, on a terminal, a transient "updating rules…" line
//! is shown. If new rules were actually installed it resolves into a one-line
//! checkmark with the current rule count and the newest ruleset date; if nothing
//! changed the line simply vanishes — the mechanism stays invisible unless it
//! had something to report.

use std::io::Write;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::Mode;
use crate::cli::Refresh;

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

/// Path to the stamp (`<cache_dir>/atomdrift/scan/rules-update.toml`), one per
/// rules channel: the enriched tree goes stale on its own clock, and a host
/// that switches to `--enriched` must not inherit the open-source tree's
/// "checked an hour ago".
fn stamp_path() -> Option<PathBuf> {
    let name = if cleave::enriched::enabled() {
        "rules-update-enriched.toml"
    } else {
        "rules-update.toml"
    };
    Some(dirs::cache_dir()?.join("atomdrift").join("scan").join(name))
}

/// Stamp the current time (best-effort; failures are non-fatal).
fn write_stamp() {
    let Some(path) = stamp_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(text) = toml::to_string(&Stamp {
        checked_unix: crate::civil::now_secs(),
    }) {
        let _ = std::fs::write(&path, text);
    }
}

/// Refresh the trait and model bundles when they are stale, by default.
///
/// [`Refresh::Force`] (`-u/--update`) bypasses the 24h gate;
/// [`Refresh::Skip`] (`--no-update`) turns the whole mechanism off.
/// `mode` matches the run's bloom usage so the reported rule count equals the
/// banner's. `terminal` gates the transient/checkmark UI: a non-terminal run
/// still refreshes, silently.
///
/// Best-effort: network or update failures are logged at debug level and
/// never block or fail the scan. Stamps every attempt so a stale or offline
/// bucket can't cause a refresh storm.
///
/// # Errors
/// The one failure that is not best-effort: `--enriched` was asked for and the
/// token is missing or refused. Scanning on with other rules would answer a
/// question nobody asked.
pub fn refresh_if_stale(refresh: Refresh, mode: Mode, terminal: bool) -> anyhow::Result<()> {
    if refresh == Refresh::Skip {
        return Ok(());
    }
    // Unix time of the last refresh attempt (0 when never stamped or unreadable).
    let last_checked = stamp_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| toml::from_str::<Stamp>(&text).ok())
        .map_or(0, |stamp| stamp.checked_unix);
    if refresh == Refresh::IfStale
        && crate::civil::now_secs().saturating_sub(last_checked) < CHECK_INTERVAL_SECS
    {
        return Ok(());
    }

    if terminal {
        eprint!(" \x1b[38;2;120;120;120m\u{27f3} updating rules\u{2026}\x1b[0m");
        let _ = std::io::stderr().flush();
    }

    let outcome = self::refresh();
    // Always stamp the attempt — success *or* failure — so an offline host
    // records that it tried and won't re-check (and re-time-out) every run; the
    // next attempt is a day away, or immediate with `-u`/`update-rules`.
    if let Some(refused) = outcome.refused {
        if terminal {
            eprint!("\r\x1b[2K");
        }
        // Not stamped: the next run must try again, not wait a day.
        return Err(anyhow::anyhow!(refused));
    }
    write_stamp();

    if !terminal {
        return Ok(());
    }
    // Resolve the transient line:
    //   - a real update earns a checkmark;
    //   - a reachable, already-current ruleset leaves no trace (invisible);
    //   - a failure (offline, mirror error) gets one actionable line, at most
    //     once a day.
    if outcome.changed {
        print_checkmark(mode);
    } else if outcome.failed {
        eprint!(
            "\r\x1b[2K \x1b[38;2;255;175;55m\u{26a0}\x1b[0m \x1b[38;2;160;160;160mauto-update failed \u{2014} run `atomscan update-rules` to update\x1b[0m\n",
        );
        let _ = std::io::stderr().flush();
    } else {
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }
    Ok(())
}

/// Whether a refresh installed anything new, and whether any source failed
/// to update.
struct Outcome {
    changed: bool,
    failed: bool,
    /// Why `--enriched` cannot be honoured, when it cannot.
    refused: Option<String>,
}

/// Refresh every rule source, quietly: the same refresh `update-rules` and a
/// server's startup run. Staleness, rather than absence, is the only
/// difference in trigger.
fn refresh() -> Outcome {
    let refreshed = crate::rules::refresh(&crate::rules::Source::ALL, None, false, true);
    for (source, e) in refreshed.failures() {
        tracing::debug!(
            source = source.name(),
            error = format!("{e:#}"),
            "auto update failed"
        );
    }
    Outcome {
        changed: refreshed.changed(),
        failed: refreshed.failures().next().is_some(),
        refused: refreshed.enriched_refusal().map(str::to_owned),
    }
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
    let models_dir = crate::models_repo::install_target().ok();
    let newest = [
        cleave::rule_update::installed(&traits_dir).map(|i| i.date),
        models_dir
            .and_then(|dir| crate::model_update::installed(&dir))
            .map(|i| i.date),
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
        "\r\x1b[2K \x1b[38;2;80;220;80m\u{2713}\x1b[0m \x1b[38;2;160;160;160mupdated {}{tail}\x1b[0m\n",
        crate::output::detection_inventory_text(counts.hashes_and_purls, counts.rules),
    );
    let _ = std::io::stderr().flush();
}
