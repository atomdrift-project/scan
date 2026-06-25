//! Host triage (`ascan sys`): scan running-process executables together with the
//! directories where malware commonly stages and persists, then merge the two
//! tallies into a single verdict.
//!
//! This is the deliberately simple composition: a process scan (`ps`) followed
//! by a file scan over a curated, OS-aware set of temp and persistence
//! locations. The locations are a static list today.
//!
//! Future work (path following): resolve the directory containing each running
//! process's binary, and walk persistence configs — cron entries, systemd
//! units, LaunchAgent plists — to extract the files they reference, feeding
//! those back in as scan targets. Tracked for a later release.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

use crate::OutputFormat;
use crate::engine::{ScanConfig, ScanSummary};

/// Run a host triage: process executables plus curated persistence/temp dirs.
///
/// The returned summary merges the process and file phases so the caller's exit
/// code reflects the worst verdict found anywhere on the host.
pub fn run(config: &ScanConfig) -> Result<ScanSummary> {
    let start = Instant::now();
    let is_terminal = matches!(config.format(), OutputFormat::Terminal);

    if is_terminal {
        eprintln!("\n\x1b[1m\u{2592} processes\x1b[0m");
    }
    let ps_summary = crate::ps::run(config)?;

    let targets = curated_targets();
    if is_terminal {
        eprintln!(
            "\n\x1b[1m\u{2592} persistence & temp locations\x1b[0m  ({} paths)",
            targets.len(),
        );
    }
    let fs_summary = if targets.is_empty() {
        zero_summary()
    } else {
        crate::engine::run_paths(&targets, config, None)?
    };

    Ok(merge(&ps_summary, &fs_summary, start))
}

/// Curated, OS-aware set of locations where malware commonly stages or persists.
///
/// Only paths that currently exist are returned, so the file scan never inflates
/// the error count with absent directories. Both files (e.g. `/etc/crontab`) and
/// directories are valid targets — `run_paths` walks each accordingly.
fn curated_targets() -> Vec<PathBuf> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut candidates: Vec<PathBuf> = Vec::new();

    // Common across unix: staging/temp plus a frequent drop target on PATH.
    for p in ["/tmp", "/var/tmp", "/usr/local/bin"] {
        candidates.push(PathBuf::from(p));
    }

    #[cfg(target_os = "linux")]
    {
        for p in [
            "/dev/shm",
            "/etc/cron.d",
            "/etc/cron.hourly",
            "/etc/cron.daily",
            "/etc/cron.weekly",
            "/etc/cron.monthly",
            "/etc/crontab",
            "/etc/systemd/system",
            "/etc/init.d",
            "/etc/rc.local",
        ] {
            candidates.push(PathBuf::from(p));
        }
        if let Some(home) = &home {
            candidates.push(home.join(".config/autostart"));
            candidates.push(home.join(".config/systemd/user"));
        }
    }

    #[cfg(target_os = "macos")]
    {
        for p in [
            "/Library/LaunchAgents",
            "/Library/LaunchDaemons",
            "/Library/StartupItems",
            "/etc/periodic",
        ] {
            candidates.push(PathBuf::from(p));
        }
        if let Some(tmpdir) = std::env::var_os("TMPDIR") {
            candidates.push(PathBuf::from(tmpdir));
        }
        if let Some(home) = &home {
            candidates.push(home.join("Library/LaunchAgents"));
        }
    }

    candidates.retain(|p| p.exists());
    candidates.sort();
    candidates.dedup();
    candidates
}

/// Sum the count fields of two phase summaries; duration is the triage wall time.
fn merge(a: &ScanSummary, b: &ScanSummary, start: Instant) -> ScanSummary {
    ScanSummary {
        total_files: a.total_files.saturating_add(b.total_files),
        hostile: a.hostile.saturating_add(b.hostile),
        suspicious: a.suspicious.saturating_add(b.suspicious),
        benign: a.benign.saturating_add(b.benign),
        errors: a.errors.saturating_add(b.errors),
        duration_ms: crate::duration_ms(start.elapsed()),
    }
}

/// A zero-count summary for when no curated targets exist on this host.
fn zero_summary() -> ScanSummary {
    ScanSummary {
        total_files: 0,
        hostile: 0,
        suspicious: 0,
        benign: 0,
        errors: 0,
        duration_ms: 0,
    }
}
