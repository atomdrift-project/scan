//! Process scanning: enumerate running processes, deduplicate by SHA256, and
//! scan each unique executable through the model.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::OutputFormat;
use crate::engine::{Progress, ScanConfig, ScanResult, ScanSummary};
use crate::explain::ShapImportance;
use crate::model::{Classification, Model};
use crate::output;

/// A group of processes sharing the same executable binary (by SHA256).
struct ProcessGroup {
    /// Canonical path to the executable.
    path: PathBuf,
    /// All PIDs running this binary.
    pids: Vec<u32>,
    /// Whether the binary was deleted from disk.
    deleted: bool,
    /// Precomputed SHA256 hex digest.
    sha256: String,
    /// Where its bytes were read for hashing, and are read again to scan:
    /// `path`, or the running image under `/proc` for a deleted binary.
    scan_path: PathBuf,
}

/// One group per distinct running binary: processes sharing an executable
/// path are merged, then paths sharing a sha256. Also returns how many
/// binaries could not be hashed, which are left out.
fn group_by_binary(entries: &[proclist::ProcessEntry]) -> (Vec<ProcessGroup>, u32) {
    // Group by exe_path → (pids, deleted).
    let mut by_path: HashMap<PathBuf, (Vec<u32>, bool)> = HashMap::new();
    for entry in entries {
        let (pids, deleted) = by_path.entry(entry.exe_path.clone()).or_default();
        pids.push(entry.pid);
        // If any entry says deleted, mark it.
        *deleted |= entry.deleted;
    }

    // Deduplicate by SHA256.
    let mut by_sha: HashMap<String, ProcessGroup> = HashMap::new();
    let mut hash_errors = 0u32;

    for (path, (pids, deleted)) in by_path {
        // On Linux, the running image itself whenever it is ours to read: the
        // path may since name other bytes (a decoy renamed over it) or not a
        // file at all. Otherwise the path — which for a deleted binary is
        // only good while something still exists there.
        let image = PathBuf::from(format!("/proc/{}/exe", pids[0]));
        let source = if cfg!(target_os = "linux") && (deleted || std::fs::metadata(&image).is_ok())
        {
            image
        } else if !deleted || path.exists() {
            path.clone()
        } else {
            tracing::warn!(path = %path.display(), "cannot hash a deleted binary; not scanned");
            hash_errors += 1;
            continue;
        };
        let hash = match crate::digest::sha256_file(&source) {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!(path = %path.display(), pid = pids[0], error = %e, "cannot hash a running binary; not scanned");
                hash_errors += 1;
                continue;
            }
        };

        let group = by_sha.entry(hash.clone()).or_insert_with(|| ProcessGroup {
            path,
            pids: Vec::new(),
            deleted: false,
            sha256: hash,
            scan_path: source,
        });
        group.pids.extend(pids);
        group.deleted |= deleted;
    }

    (by_sha.into_values().collect(), hash_errors)
}

/// Counters a process scan's rayon workers share.
#[derive(Default)]
struct Tally {
    hostile: AtomicU32,
    suspicious: AtomicU32,
    benign: AtomicU32,
    errors: AtomicU32,
    dependency_gaps: AtomicU32,
    static_flags: AtomicU32,
    flagged_candidates: AtomicU32,
}

impl Tally {
    fn add(counter: &AtomicU32, n: u32) {
        counter.fetch_add(n, Ordering::Relaxed);
    }

    /// Account for one classified binary.
    fn count(&self, result: &ScanResult) {
        let class = match result.classification {
            Classification::Hostile => &self.hostile,
            Classification::Suspicious => &self.suspicious,
            Classification::Benign => &self.benign,
        };
        Self::add(class, 1);
        Self::add(&self.dependency_gaps, result.dependency_attention.gaps);
        Self::add(&self.static_flags, u32::from(result.has_severe_findings()));
        Self::add(
            &self.flagged_candidates,
            result.dependency_attention.flagged_candidates,
        );
    }

    fn summary(&self, total_files: u32, started: Instant) -> ScanSummary {
        let load = |counter: &AtomicU32| counter.load(Ordering::Relaxed);
        ScanSummary {
            total_files,
            hostile: load(&self.hostile),
            suspicious: load(&self.suspicious),
            benign: load(&self.benign),
            errors: load(&self.errors),
            dependency_attention: crate::engine::DependencyAttention {
                gaps: load(&self.dependency_gaps),
                flagged_candidates: load(&self.flagged_candidates),
            },
            static_flags: load(&self.static_flags),
            duration_ms: crate::duration_ms(started.elapsed()),
        }
    }
}

/// Run a process scan: enumerate, deduplicate, classify.
///
/// # Errors
/// The model could not be loaded, processes could not be enumerated, cleave
/// could not start, or stdout closed before anything was flagged.
pub fn run(config: &ScanConfig) -> Result<ScanSummary> {
    // Warm cleave's YARA engine + capability mapper off the rayon pool before
    // any scan fires rayon work, so no worker races the one-time build.
    crate::engine::prefetch_cleave_resources();
    // Load the model before hashing every running executable, so a broken
    // bundle is reported at once rather than after the slowest step.
    let model = Model::load(config.model_dir(), config.thresholds(), config.level())?;
    let shap = ShapImportance::load_or_warn(config.model_dir());

    let scan_start = Instant::now();
    let is_terminal = matches!(config.format(), OutputFormat::Terminal);

    let proc_result = proclist::enumerate().map_err(|e| anyhow::anyhow!("{e}"))?;
    let perm_denied = proc_result.stats.permission_denied;
    if perm_denied > 0 && !is_root() {
        output::print_permission_warning(perm_denied);
    }

    if proc_result.entries.is_empty() {
        let summary = Tally::default().summary(0, scan_start);
        if is_terminal {
            output::print_summary(&summary);
        }
        return Ok(summary);
    }

    let (groups, hash_errors) = group_by_binary(&proc_result.entries);
    let total = u32::try_from(groups.len()).unwrap_or(u32::MAX);

    if is_terminal {
        // Detection rules already resident in memory — no extra loading.
        output::print_banner(crate::engine::detection_counts(config));
    }

    let cancellation = crate::interrupt::arm("Interrupted — finishing current process…");
    let mut cleave_opts = cleave::AnalysisOptions {
        compact_members: true, // compact projection only
        slow_rule_ms: config.slow_rule_ms(),
        cancellation: Some(Arc::clone(&cancellation)),
        ..Default::default()
    };
    crate::engine::add_zip_passwords(&mut cleave_opts, config.zip_passwords());

    let progress = (is_terminal && total > 1).then(|| Progress::new(total));
    let tally = Tally {
        errors: AtomicU32::new(hash_errors),
        ..Tally::default()
    };
    // Set when stdout closes under the scan (`atomscan ps | head`): nobody can
    // be told the remaining verdicts, so the scan stops.
    let broken_pipe = AtomicBool::new(false);

    let by_scan_path = unskipped(&groups, config, &tally, progress.as_ref());
    let scan_paths: Vec<PathBuf> = by_scan_path.keys().cloned().collect();

    // Analyze the unique binaries through cleave's batch API: it loads the
    // CapabilityMapper + YARA engine once and fans the files out across a single
    // rayon pass. A hand-rolled `par_iter` over the single-file `analyze_file`
    // would nest rayon inside rayon — each file's inner parallelism multiplied
    // by the outer fan-out oversubscribes the pool and runs slower than serial
    // at several times the CPU. Verdict order in the output stream is
    // completion order, which JSON/terminal consumers tolerate.
    let analyze = |scan_path: &Path, analyzed: Result<cleave::AnalysisReport>| {
        let Some(group) = by_scan_path.get(scan_path) else {
            return;
        };
        if let Some(p) = &progress {
            p.increment();
        }
        let result = analyzed.and_then(|report| {
            build_result(
                report,
                group,
                &model,
                shap.as_ref(),
                config,
                Some(&cancellation),
            )
        });
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                tracing::warn!(path = %group.path.display(), error = %format!("{error:#}"), "process binary analysis failed");
                Tally::add(&tally.errors, 1);
                return;
            }
        };
        tally.count(&result);
        if !(config.format().emits_every_result() || config.filter().shows(result.classification)) {
            return;
        }
        let print = || match crate::engine::emit_result(&result, config) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {
                broken_pipe.store(true, Ordering::Relaxed);
                cancellation.store(true, Ordering::Relaxed);
            }
            Err(e) => {
                tracing::error!(path = %result.path, error = %e, "could not write a scan result");
            }
        };
        match &progress {
            Some(p) => p.around_result(print),
            None => print(),
        }
    };

    cleave::scan_files(&scan_paths, &cleave_opts, |event| {
        if let cleave::ScanEvent::File { path, result } = event {
            analyze(&path, *result);
        }
    })?;

    let summary = tally.summary(total, scan_start);

    // As a path scan does: a verdict already reached still decides the exit
    // status, so under `pipefail` a hostile process does not read as success
    // because the reader stopped reading.
    if broken_pipe.load(Ordering::Relaxed) {
        if summary.hostile > 0 || summary.suspicious > 0 {
            return Ok(summary);
        }
        return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe))
            .context("writing scan results to stdout");
    }

    if is_terminal {
        // Collapse the progress bar straight into the verdict: when a bar is
        // live it sits on its own line below the banner, so clear it and write
        // the one-line summary in its place. The finished scan then shows just
        // the banner and this line — no intermediate "files in" line, no
        // separator, no trailing blank.
        let line = output::summary_status_line(&summary);
        if let Some(p) = &progress {
            p.quiesce(); // stop the heartbeat before we claim the bar's line
            eprintln!("{} {line}", output::ERASE_LINE);
            let _ = std::io::stderr().flush();
        } else {
            eprintln!(" {line}");
        }
    }

    Ok(summary)
}

/// The groups left to analyze, by the path cleave reads each from. A
/// known-good binary is counted benign here and left out, unless it was
/// created, changed or modified within the rescan window, when it is scanned
/// on its own merits; known-bad and conflicted are flagged but still analyzed.
fn unskipped<'g>(
    groups: &'g [ProcessGroup],
    config: &ScanConfig,
    tally: &Tally,
    progress: Option<&Progress>,
) -> HashMap<PathBuf, &'g ProcessGroup> {
    let mut by_scan_path = HashMap::with_capacity(groups.len());
    for group in groups {
        if let Some(lookup) = config.bloom()
            && let Some(digest) = burton::parse_sha256_hex(&group.sha256)
            && let Some(benign) = crate::engine::bloom_gate_fresh(
                config,
                &group.path,
                lookup.decide_sha256(&digest),
                |label, verdict| {
                    // As with every other `ps` line, a closed stream is not an error.
                    let _ = output::print_bloom_verdict(label, verdict, config.format());
                },
            )
        {
            Tally::add(&tally.benign, benign);
            if let Some(p) = progress {
                p.increment();
            }
            continue;
        }
        by_scan_path.insert(group.scan_path.clone(), group);
    }
    by_scan_path
}

/// Build a ScanResult from a cleave report, injecting pids and deleted state.
fn build_result(
    report: cleave::AnalysisReport,
    group: &ProcessGroup,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
    cancellation: Option<&Arc<AtomicBool>>,
) -> Result<ScanResult> {
    let display_path = group.path.as_path();
    // The bloom flag for a known-bad/conflicted process image, re-derived from the
    // executable's sha256 (known-good is short-circuited before we ever scan, so it
    // never reaches here). Drives the inline 🚩/🏴 in the terminal header.
    let bloom_mark = config
        .bloom()
        .and_then(|lk| burton::parse_sha256_hex(&group.sha256).map(|d| lk.decide_sha256(&d)))
        .and_then(output::BloomMark::from_decision);
    let label = display_path.display().to_string();
    let cr = crate::engine::classify_report(
        report,
        crate::engine::ClassifyRequest {
            shap,
            cancellation,
            tiny_opts: crate::engine::tiny_opts_for(config),
            interpret: config.interpret(),
            fetch: config.fetch_policy(),
            zip_passwords: config.zip_passwords(),
            needs: config.output_needs(),
            bloom_mark,
            ..crate::engine::ClassifyRequest::new(&label, display_path, model)
        },
    )?;
    let mut result = cr.into_scan_result(label, model, config.keeps_report());
    result.pids = Some(group.pids.clone());
    result.deleted = group.deleted.then_some(true);
    result.sha256.clone_from(&group.sha256);
    result.bloom_mark = bloom_mark;
    Ok(result)
}

/// Check if running as root/admin.
fn is_root() -> bool {
    #[cfg(unix)]
    {
        // SAFETY: getuid() is a trivial syscall with no preconditions.
        unsafe { libc::getuid() == 0 }
    }
    #[cfg(not(unix))]
    {
        false // Conservative: always show the warning on non-unix
    }
}
