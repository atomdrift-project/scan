//! Default-on, quiet refresh of detection rules and models when they go stale.
//!
//! On an interactive scan, if the local ruleset hasn't been checked in a day
//! (and `--no-update` is not given), scan refreshes the cleave
//! trait bundle, the ML model bundle, and the bloom filters before it starts.
//! The check is gated by a tiny on-disk stamp, so the common case touches
//! neither the network nor the terminal: most runs return in microseconds
//! having done nothing.
//!
//! When a refresh *does* run, the update calls are silenced (their own progress
//! lines would be noise) and, on a terminal, one live status line is shown:
//! "checking for rule updates…" while the manifests are read, then each bundle
//! downloading with a bar, size, rate and — when more than two seconds remain —
//! an ETA, then "installing…" while it is verified. If new rules were actually
//! installed it resolves into a one-line checkmark with the current rule count,
//! the newest ruleset date and what was fetched; if nothing changed the line
//! simply vanishes — the mechanism stays invisible unless it had something to
//! report.

use std::io::{IsTerminal as _, Write};
use std::path::PathBuf;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::Mode;
use crate::cli::Refresh;

/// On-disk stamp of the last refresh attempt (success *or* failure), so an
/// unreachable bucket doesn't trigger a network call on every run.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Stamp {
    /// Unix time the refresh was last attempted.
    checked_unix: u64,
}

/// Path to the stamp in [`crate::cache_dir`], one per rule feed *and* per
/// build. Each feed's tree goes stale on its own clock, and a host that
/// switches feeds must not inherit another tree's "checked an hour ago".
/// Likewise an upgraded (or downgraded, or side-by-side) build must not
/// inherit another version's "checked an hour ago": the data that version was
/// content with may be in a format this one cannot read, so each version gets
/// its own clock and refreshes once on first use.
fn stamp_path() -> Option<PathBuf> {
    Some(crate::cache_dir()?.join(stamp_file_name(cleave::feed::current())))
}

fn stamp_file_name(feed: cleave::feed::Feed) -> String {
    format!(
        "rules-update-{}-{}.toml",
        feed.slug(),
        env!("CARGO_PKG_VERSION")
    )
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
/// The one failure that is not best-effort: a feed named with `--feed` cannot
/// be had — no token, or one refused for it. Scanning on with other rules
/// would answer a question nobody asked.
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
        && crate::civil::now_secs().saturating_sub(last_checked)
            < crate::update_check::CHECK_INTERVAL_SECS
    {
        return Ok(());
    }

    // A live line only where it can be redrawn in place; a redirected stderr
    // still gets the one-line outcome below.
    let live = terminal && std::io::stderr().is_terminal();
    crate::transfer::reset();
    let started = Instant::now();
    let outcome = if live {
        with_status_line(self::refresh)
    } else {
        self::refresh()
    };
    let fetched: u64 = crate::transfer::snapshot()
        .iter()
        .map(|(_, tally)| tally.done)
        .sum();
    tracing::debug!(
        fetched_bytes = fetched,
        elapsed_ms = started.elapsed().as_millis(),
        changed = outcome.changed,
        failed = outcome.failed,
        "rules refresh finished"
    );
    // Always stamp the attempt — success *or* failure — so an offline host
    // records that it tried and won't re-check (and re-time-out) every run; the
    // next attempt is a day away, or immediate with `-u`/`update-rules`.
    if let Some(refused) = outcome.refused {
        if live {
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
        let fetched = (fetched > 0).then(|| {
            format!(
                "{} in {}",
                crate::output::human_size(fetched),
                elapsed_text(started.elapsed())
            )
        });
        print_checkmark(mode, fetched.as_deref());
    } else if outcome.failed {
        eprint!(
            "\r\x1b[2K \x1b[38;2;255;175;55m\u{26a0}\x1b[0m \x1b[38;2;160;160;160mrules update failed \u{2014} scanning with the installed rules; the next run tries again\x1b[0m\n",
        );
        let _ = std::io::stderr().flush();
    } else if live {
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }
    Ok(())
}

/// Run `work` under a live status line, redrawn every tick until it returns.
/// The line is left for the caller to erase or resolve.
fn with_status_line<T>(work: impl FnOnce() -> T) -> T {
    let (stop, stopped) = mpsc::channel::<()>();
    std::thread::scope(|s| {
        s.spawn(move || {
            let cols = cleave::output::terminal_width();
            // When the first byte arrived: the rate is averaged from there,
            // not from the manifest checks before it.
            let mut first_byte: Option<Instant> = None;
            for tick in 0.. {
                let tallies = crate::transfer::snapshot();
                if first_byte.is_none() && tallies.iter().any(|(_, t)| t.done > 0) {
                    first_byte = Some(Instant::now());
                }
                let line = status_line(&tallies, first_byte.map(|t| t.elapsed()), tick, cols);
                let mut stderr = std::io::stderr().lock();
                // Cosmetic: a failed redraw must not stop the refresh.
                let _ = write!(stderr, "\r{line}\x1b[K").and_then(|()| stderr.flush());
                drop(stderr);
                if !matches!(
                    stopped.recv_timeout(crate::engine::PROGRESS_TICK),
                    Err(RecvTimeoutError::Timeout)
                ) {
                    break;
                }
            }
        });
        let result = work();
        drop(stop);
        result
    })
}

/// Columns of the download bar.
const BAR_WIDTH: usize = 20;
/// How long bytes must flow before a rate is worth showing.
const RATE_AFTER: Duration = Duration::from_millis(500);
/// The shortest remaining time worth an ETA; below it the bar says enough.
const ETA_ABOVE_SECS: f64 = 2.0;

/// One frame of the status line, fitted to `cols`:
///
/// ```text
///  ⠙ Checking for rule updates…
///  ⠙ Downloading Bloom filters  ━━━━━━━╸────────────  142.1MB / 334.0MB · 11.2MB/s · ~17s left
///  ⠙ Installing Bloom filters…
/// ```
///
/// `downloading_for` is how long since the first byte arrived. The bar, size
/// and rate cover every download of this refresh, finished ones included, so
/// the bar never moves backwards as a bundle completes.
fn status_line(
    tallies: &[(&str, crate::transfer::Tally)],
    downloading_for: Option<Duration>,
    tick: usize,
    cols: usize,
) -> String {
    use crate::output::{Ink, Keep, elide, human_size, palette};

    let (ink, p) = (Ink::stderr(), palette());
    let frame = crate::engine::SPINNER[tick % crate::engine::SPINNER.len()];
    let spinner = ink.fg(p.live, frame.encode_utf8(&mut [0; 4]));
    let names = |open: bool| {
        tallies
            .iter()
            .filter(|(_, tally)| (tally.open > 0) == open)
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(", ")
    };
    if tallies.is_empty() {
        return format!(
            " {spinner} {}",
            ink.fg(p.label, "Checking for rule updates\u{2026}")
        );
    }
    let downloading = names(true);
    if downloading.is_empty() {
        let line = format!("Installing {}\u{2026}", names(false));
        return format!(
            " {spinner} {}",
            ink.fg(p.label, &elide(&line, cols.saturating_sub(4), Keep::Head))
        );
    }

    let done: u64 = tallies.iter().map(|(_, tally)| tally.done).sum();
    // Held back until bytes have flowed a moment: files announce their sizes
    // as their responses arrive, and a bar drawn from the first few would
    // jump backwards as the rest land.
    let settled = downloading_for.filter(|&elapsed| elapsed >= RATE_AFTER);
    let total: Option<u64> = tallies
        .iter()
        .try_fold(0, |sum, (_, tally)| Some(sum + tally.total?))
        .filter(|&total| total > 0 && settled.is_some());
    let size = |bytes: u64| match bytes {
        0 => "0B".to_owned(),
        bytes => human_size(bytes),
    };

    let mut stats = match total {
        Some(total) => format!("{} / {}", size(done), size(total)),
        None => size(done),
    };
    if let Some(elapsed) = settled {
        // Bytes per second, averaged since the first byte.
        let millis = u64::try_from(elapsed.as_millis())
            .unwrap_or(u64::MAX)
            .max(1);
        let rate = done.saturating_mul(1000) / millis;
        stats.push_str(&format!(" \u{b7} {}/s", size(rate)));
        if let Some(total) = total {
            #[expect(
                clippy::cast_precision_loss,
                reason = "byte counts far below 2^52; display only"
            )]
            let eta = total.saturating_sub(done) as f64 / rate.max(1) as f64;
            if eta > ETA_ABOVE_SECS {
                stats.push_str(&format!(" \u{b7} {} left", crate::engine::format_eta(eta)));
            }
        }
    }

    // ` ⠙ ` + `Downloading ` + names + bar + `  ` + stats, one column spare so
    // it never wraps. A narrow terminal gives up the bar before the names.
    let room = cols.saturating_sub(3 + "Downloading ".len() + 2 + stats.chars().count() + 1);
    let show_bar = total.is_some() && room >= BAR_WIDTH + 2 + downloading.chars().count().min(16);
    let names = elide(
        &downloading,
        room.saturating_sub(if show_bar { BAR_WIDTH + 2 } else { 0 })
            .max(1),
        Keep::Head,
    );
    // ━━━╸ then ─── for the remainder, as the scan's own bar draws it.
    let bar = match total {
        Some(total) if show_bar => {
            let filled = usize::try_from(done.saturating_mul(BAR_WIDTH as u64) / total)
                .unwrap_or(BAR_WIDTH)
                .min(BAR_WIDTH);
            let tip = if filled < BAR_WIDTH { "\u{2578}" } else { "" };
            format!(
                "  {}{}",
                ink.fg(p.bar, &format!("{}{tip}", "\u{2501}".repeat(filled))),
                ink.fg(
                    p.dot_sep,
                    &"\u{2500}".repeat(BAR_WIDTH.saturating_sub(filled + 1))
                ),
            )
        }
        _ => String::new(),
    };
    format!(
        " {spinner} {}{bar}  {}",
        ink.fg(p.label, &format!("Downloading {names}")),
        ink.fg(p.label, &stats),
    )
}

/// A whole-second duration for the checkmark: `840ms`, `29s`, `2m05s`.
fn elapsed_text(elapsed: Duration) -> String {
    let secs = elapsed.as_secs();
    match secs {
        0 => format!("{}ms", elapsed.as_millis()),
        1..60 => format!("{secs}s"),
        _ => format!("{}m{:02}s", secs / 60, secs % 60),
    }
}

/// Whether a refresh installed anything new, and whether any source failed
/// to update.
struct Outcome {
    changed: bool,
    failed: bool,
    /// Why the feed named with `--feed` cannot be had, when it cannot.
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
        refused: refreshed.feed_refusal().map(str::to_owned),
    }
}

/// Erase the transient line and print the update checkmark: the current split
/// signature/rule inventory, newest trait/model bundle date, and what this
/// refresh downloaded.
fn print_checkmark(mode: Mode, fetched: Option<&str>) {
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
    let mut tail = match &newest {
        Some(date) => format!(" \u{b7} newest {date}"),
        None => String::new(),
    };
    if let Some(fetched) = fetched {
        tail.push_str(&format!(" \u{b7} fetched {fetched}"));
    }
    eprint!(
        "\r\x1b[2K \x1b[38;2;80;220;80m\u{2713}\x1b[0m \x1b[38;2;160;160;160mupdated {}{tail}\x1b[0m\n",
        crate::output::detection_inventory_text(counts.hashes_and_purls, counts.rules),
    );
    let _ = std::io::stderr().flush();
}

#[cfg(test)]
mod tests {
    use super::{elapsed_text, stamp_file_name, status_line};
    use crate::transfer::Tally;
    use std::time::Duration;

    const MB: u64 = 1024 * 1024;

    fn bloom(done: u64, total: Option<u64>, open: u32) -> Vec<(&'static str, Tally)> {
        vec![("Bloom filters", Tally { done, total, open })]
    }

    #[test]
    fn status_line_checks_then_downloads_then_installs() {
        let line = status_line(&[], None, 0, 100);
        assert!(line.contains("Checking for rule updates\u{2026}"), "{line}");

        let line = status_line(&bloom(0, Some(300 * MB), 1), None, 0, 100);
        assert!(line.contains("Downloading Bloom filters  0B"), "{line}");
        assert!(
            !line.contains("/s") && !line.contains("300.0MB"),
            "no rate or total before bytes flow: {line}"
        );

        let line = status_line(
            &bloom(100 * MB, Some(300 * MB), 1),
            Some(Duration::from_secs(10)),
            0,
            100,
        );
        assert!(line.contains("100.0MB / 300.0MB"), "{line}");
        assert!(line.contains("10.0MB/s"), "{line}");
        assert!(line.contains("~20s left"), "{line}");
        assert!(
            line.contains("\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2578}"),
            "{line}"
        );

        let line = status_line(&bloom(300 * MB, Some(300 * MB), 0), None, 0, 100);
        assert!(line.contains("Installing Bloom filters\u{2026}"), "{line}");
    }

    #[test]
    fn eta_shows_only_beyond_two_seconds() {
        let line = status_line(
            &bloom(290 * MB, Some(300 * MB), 1),
            Some(Duration::from_secs(10)),
            0,
            100,
        );
        assert!(line.contains("/s"), "{line}");
        assert!(!line.contains("left"), "{line}");
    }

    #[test]
    fn an_unsized_download_shows_bytes_without_bar_or_eta() {
        let line = status_line(
            &bloom(5 * MB, None, 1),
            Some(Duration::from_secs(5)),
            0,
            100,
        );
        assert!(line.contains("5.0MB \u{b7} 1.0MB/s"), "{line}");
        assert!(
            !line.contains('\u{2500}') && !line.contains("left"),
            "{line}"
        );
    }

    #[test]
    fn status_line_names_only_open_bundles_and_never_wraps() {
        let tallies = [
            (
                "Models",
                Tally {
                    done: 20 * MB,
                    total: Some(20 * MB),
                    open: 0,
                },
            ),
            (
                "Bloom filters",
                Tally {
                    done: MB,
                    total: Some(300 * MB),
                    open: 1,
                },
            ),
            (
                "NSRL filters",
                Tally {
                    done: MB,
                    total: Some(80 * MB),
                    open: 1,
                },
            ),
        ];
        let line = status_line(&tallies, Some(Duration::from_secs(1)), 0, 200);
        assert!(
            line.contains("Downloading Bloom filters, NSRL filters"),
            "{line}"
        );
        assert!(line.contains("22.0MB / 400.0MB"), "{line}");
        let narrow = status_line(&tallies, Some(Duration::from_secs(1)), 0, 80);
        assert!(narrow.chars().count() <= 80, "{narrow}");
    }

    #[test]
    fn elapsed_text_reads_naturally() {
        assert_eq!(elapsed_text(Duration::from_millis(840)), "840ms");
        assert_eq!(elapsed_text(Duration::from_secs(29)), "29s");
        assert_eq!(elapsed_text(Duration::from_secs(125)), "2m05s");
    }

    #[test]
    fn stamp_is_keyed_by_build_version_and_feed() {
        use cleave::feed::Feed;
        let names: Vec<_> = Feed::ALL.into_iter().map(stamp_file_name).collect();
        for name in &names {
            assert!(name.contains(env!("CARGO_PKG_VERSION")), "{name}");
        }
        assert!(names[0] != names[1] && names[1] != names[2] && names[0] != names[2]);
        // A stamp from an older layout must never be read back.
        for old in ["rules-update.toml", "rules-update-enriched.toml"] {
            assert!(!names.iter().any(|name| name == old), "{old}");
        }
        let pre_feed = format!("rules-update-{}.toml", env!("CARGO_PKG_VERSION"));
        assert!(!names.contains(&pre_feed), "{pre_feed}");
    }
}
