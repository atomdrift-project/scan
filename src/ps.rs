//! Process scanning: enumerate running processes, deduplicate by SHA256, and
//! scan each unique executable through the litmus model.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

use crate::OutputFormat;
use crate::engine::{Progress, ScanConfig, ScanResult, ScanSummary};
use crate::explain::ShapImportance;
use crate::features::ExtractContext;
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
}

/// Compute SHA256 of a file, reading from the given path.
fn sha256_file(path: &std::path::Path) -> Result<String> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Run a process scan: enumerate, deduplicate, classify.
pub fn run(config: &ScanConfig) -> Result<ScanSummary> {
    // Warm cleave's YARA engine + capability mapper off the rayon pool before
    // any scan fires rayon work. See `run_scan_paths` for why this matters.
    cleave::prefetch_shared_resources(true);

    // Tune rizin for live-process scanning. Unlike a filesystem scan (where we
    // want every architecture and tolerate minutes of deep analysis), `ps` is
    // interactive and dominated by a few giant signed apps (Electron/Bun
    // binaries can be 100–215 MB). Three caps keep it responsive without
    // changing verdicts materially:
    //   - native-arch-only: a universal binary's non-host slice never runs here,
    //     so don't pay full `aaa` on it (roughly halves fat-binary cost).
    //   - 60s timeout: a process binary needing more rizin than that is
    //     pathological; goblin's typed views still classify it.
    //   - 100 MB size gate: skip rizin on the giants entirely — disassembling a
    //     signed 200 MB app is never worth blocking the scan on.
    filefacts::rizin::set_native_arch_only(true);
    filefacts::rizin::set_timeout_secs(60);
    filefacts::rizin::set_max_bytes(100 * 1024 * 1024);

    let scan_start = Instant::now();
    let is_terminal = matches!(config.format(), OutputFormat::Terminal);

    // Enumerate processes.
    let proc_result = proclist::enumerate().map_err(|e| anyhow::anyhow!("{e}"))?;

    let perm_denied = proc_result.stats.permission_denied;

    // Warn about permission-denied processes.
    if perm_denied > 0 && !is_root() {
        eprintln!(
            "  \x1b[38;2;255;175;55m\u{26A0}\x1b[0m  \x1b[38;2;180;180;180mSkipping {} processes due to insufficient permissions — re-run as root for full coverage\x1b[0m\n",
            perm_denied,
        );
    }

    if proc_result.entries.is_empty() {
        let summary = ScanSummary {
            total_files: 0,
            hostile: 0,
            suspicious: 0,
            benign: 0,
            errors: 0,
            duration_ms: crate::duration_ms(scan_start.elapsed()),
        };
        if is_terminal {
            output::print_summary(&summary);
        }
        return Ok(summary);
    }

    // Group by exe_path → (pids, deleted).
    let mut by_path: HashMap<PathBuf, (Vec<u32>, bool)> = HashMap::new();
    for entry in &proc_result.entries {
        let (pids, deleted) = by_path
            .entry(entry.exe_path.clone())
            .or_insert_with(|| (Vec::new(), entry.deleted));
        pids.push(entry.pid);
        // If any entry says deleted, mark it.
        if entry.deleted {
            *deleted = true;
        }
    }

    // Deduplicate by SHA256.
    let mut by_sha: HashMap<String, ProcessGroup> = HashMap::new();
    let mut hash_errors = 0u32;

    for (path, (pids, deleted)) in by_path {
        let hash = if deleted && !path.exists() {
            // Try reading via /proc on Linux.
            #[cfg(target_os = "linux")]
            {
                match sha256_file(std::path::Path::new(&format!("/proc/{}/exe", pids[0]))) {
                    Ok(h) => h,
                    Err(e) => {
                        tracing::warn!(
                            "cannot hash deleted binary {} (pid {}): {e}",
                            path.display(),
                            pids[0],
                        );
                        hash_errors += 1;
                        continue;
                    }
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                tracing::warn!("cannot hash deleted binary: {}", path.display());
                hash_errors += 1;
                continue;
            }
        } else {
            match sha256_file(&path) {
                Ok(h) => h,
                Err(e) => {
                    tracing::warn!("cannot hash {}: {e}", path.display());
                    hash_errors += 1;
                    continue;
                }
            }
        };

        let group = by_sha.entry(hash.clone()).or_insert_with(|| ProcessGroup {
            path,
            pids: Vec::new(),
            deleted: false,
            sha256: hash,
        });
        group.pids.extend(pids);
        group.deleted |= deleted;
    }

    let mut groups: Vec<ProcessGroup> = by_sha.into_values().collect();
    // Optional cap for fast iteration / benchmarking: `SCAN_PS_LIMIT=N` scans
    // only the first N unique binaries. Sorted by path first so the chosen
    // subset is stable across runs (modulo which processes are alive at the
    // time), which keeps before/after timings comparable. Unset = scan all.
    if let Some(limit) = std::env::var("SCAN_PS_LIMIT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        groups.sort_by(|a, b| a.path.cmp(&b.path));
        groups.truncate(limit);
        tracing::info!(
            limit,
            kept = groups.len(),
            "SCAN_PS_LIMIT active — scanning first {limit} unique binaries (path-sorted)"
        );
    }
    let groups = groups;
    let total = u32::try_from(groups.len()).unwrap_or(u32::MAX);

    if is_terminal {
        eprintln!(
            "\n  \x1b[38;2;120;180;255m\u{25c6}\x1b[0m  {} unique binaries across {} processes\n",
            total,
            proc_result.entries.len(),
        );
    }

    // Load model and scan each unique binary.
    let model = Model::load(config.model_dir(), config.thresholds(), config.level())?;
    let shap = ShapImportance::load(config.model_dir()).ok();
    let ctx = ExtractContext::new(model.spec());
    let cancellation = Arc::new(AtomicBool::new(false));
    let ctrlc_flag = Arc::clone(&cancellation);
    let _ = ctrlc::set_handler(move || {
        if ctrlc_flag.load(Ordering::Relaxed) {
            // Second ctrl-c: reap rizin workers, then hard exit. Cleave runs
            // each rizin in its own process group, so SIGINT on the terminal
            // never reaches them — without an explicit SIGKILL here, every
            // in-flight child would outlive us as an orphan.
            cleave::kill_all_rizin_groups();
            std::process::exit(130);
        }
        eprintln!("\nInterrupted — finishing current process…");
        ctrlc_flag.store(true, Ordering::Relaxed);
    });
    let cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        cancellation: Some(Arc::clone(&cancellation)),
        ..Default::default()
    };
    let stdout = Mutex::new(std::io::stdout());

    let progress = if is_terminal && total > 1 {
        Some(Progress::new(total))
    } else {
        None
    };

    let hostile = AtomicU32::new(0);
    let suspicious = AtomicU32::new(0);
    let benign = AtomicU32::new(0);
    let errors = AtomicU32::new(hash_errors);

    // Map each scan path back to its process group for PID annotation. Deleted
    // binaries that aren't scannable are counted as errors here and omitted.
    let mut scan_paths: Vec<PathBuf> = Vec::with_capacity(groups.len());
    let mut by_scan_path: HashMap<PathBuf, &ProcessGroup> = HashMap::with_capacity(groups.len());
    for group in &groups {
        // Known-good/known-bad short-circuit by the executable's sha256: a
        // known-good binary is skipped here (and counted benign); known-bad and
        // conflicted are flagged but still analyzed below.
        if let Some(lookup) = config.bloom()
            && let Some(digest) = crate::bloom::parse_sha256_hex(&group.sha256)
            && let Some(summary) = crate::engine::bloom_gate(
                config,
                &group.path.display().to_string(),
                lookup.decide_sha256(&digest),
            )
        {
            benign.fetch_add(summary.benign, Ordering::Relaxed);
            if let Some(p) = &progress {
                p.increment();
            }
            continue;
        }

        let scan_path = if group.deleted && !group.path.exists() {
            // On Linux, try scanning via /proc/pid/exe.
            #[cfg(target_os = "linux")]
            {
                PathBuf::from(format!("/proc/{}/exe", group.pids[0]))
            }
            #[cfg(not(target_os = "linux"))]
            {
                tracing::warn!(
                    "deleted binary not scannable: {} (pids: {:?})",
                    group.path.display(),
                    group.pids
                );
                errors.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        } else {
            group.path.clone()
        };
        by_scan_path.insert(scan_path.clone(), group);
        scan_paths.push(scan_path);
    }

    // Analyze the unique binaries through cleave's batch API: it loads the
    // CapabilityMapper + YARA engine once and fans the files out across a single
    // rayon pass. A hand-rolled `par_iter` over the single-file `analyze_file`
    // instead nests rayon inside rayon — each file's inner parallelism multiplied
    // by the outer fan-out oversubscribed the pool and ran *slower* than serial
    // (measured: 2.8× util, 3× the CPU). Verdict order in the output stream is
    // now completion order, which JSON/terminal consumers already tolerate.
    let analyze = |scan_path: &std::path::Path, result: Result<cleave::AnalysisReport>| {
        let Some(group) = by_scan_path.get(scan_path) else {
            return;
        };
        let report = match result {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("error analyzing {}: {e}", group.path.display());
                errors.fetch_add(1, Ordering::Relaxed);
                if let Some(p) = &progress {
                    p.increment();
                }
                return;
            }
        };

        if let Some(p) = &progress {
            p.increment();
        }

        match build_result(
            &group.path,
            report,
            &ctx,
            &model,
            shap.as_ref(),
            config,
            group,
            Some(&cancellation),
        ) {
            Ok(result) => {
                match result.classification {
                    Classification::Hostile => hostile.fetch_add(1, Ordering::Relaxed),
                    Classification::Suspicious => suspicious.fetch_add(1, Ordering::Relaxed),
                    Classification::Benign => benign.fetch_add(1, Ordering::Relaxed),
                };
                if config.format() == OutputFormat::Json
                    || config.filter().shows(&result.classification)
                {
                    emit_result(
                        &result,
                        config,
                        &group.pids,
                        group.deleted,
                        progress.is_some(),
                        &stdout,
                    );
                    if let Some(p) = &progress {
                        p.redraw();
                    }
                }
            }
            Err(e) => {
                tracing::warn!("error processing {}: {e}", group.path.display());
                errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    };

    cleave::scan_files(&scan_paths, &cleave_opts, |event| {
        if let cleave::ScanEvent::File { path, result } = event {
            analyze(&path, *result);
        }
    })?;

    let hostile = hostile.load(Ordering::Relaxed);
    let suspicious = suspicious.load(Ordering::Relaxed);
    let benign = benign.load(Ordering::Relaxed);
    let errors = errors.load(Ordering::Relaxed);

    if let Some(p) = &progress {
        p.finish();
    }

    let summary = ScanSummary {
        total_files: total,
        hostile,
        suspicious,
        benign,
        errors,
        duration_ms: crate::duration_ms(scan_start.elapsed()),
    };

    if is_terminal {
        output::print_summary(&summary);
    }

    Ok(summary)
}

/// Build a ScanResult from a cleave report, injecting pids and deleted state.
#[allow(clippy::too_many_arguments)]
fn build_result(
    display_path: &std::path::Path,
    report: cleave::AnalysisReport,
    ctx: &ExtractContext,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
    group: &ProcessGroup,
    cancellation: Option<&Arc<AtomicBool>>,
) -> Result<ScanResult> {
    let cr = crate::engine::classify_report(
        &display_path.display().to_string(),
        report,
        ctx,
        model,
        shap,
        cancellation,
        Some(100),
        &crate::engine::tiny_opts_for(config),
        config.interpret(),
        display_path,
        config.fetch_policy(),
        false, // ps emits machine-readable output; no interactive fetch log
        matches!(config.format(), OutputFormat::Tiny),
        // `--show=all` with JSON: list every member of an archive-backed image.
        config.filter().is_all() && matches!(config.format(), OutputFormat::Json),
        None, // process images carry no fetched-package registry metadata
    )?;
    let is_json = matches!(config.format(), OutputFormat::Json);

    let cleave = if is_json { Some(cr.report_json) } else { None };

    Ok(ScanResult {
        v: "7",
        classification: cr.classification,
        probability: cr.probability,
        threshold: cr.threshold,
        level: cr.level,
        version: crate::engine::model_version_string(model.info()),
        analyzed_at: crate::engine::now_rfc3339(),
        cleave,
        pids: Some(group.pids.clone()),
        deleted: group.deleted.then_some(true),
        path: display_path.display().to_string(),
        finding_counts: cr.finding_counts,
        formula: cr.formula,
        reasons: cr.reasons,
        top_findings: cr.top_findings,
        model_scores: cr.model_scores,
        skipped_models: cr.skipped_models,
        file_type: cr.file_type,
        size_bytes: cr.size_bytes,
        sha256: group.sha256.clone(),
        embedded_files: cr.embedded_files,
        rendered_context: cr.rendered_context,
        interpretation: cr.interpretation,
    })
}

/// Emit a single result with PID annotations.
fn emit_result(
    r: &ScanResult,
    config: &ScanConfig,
    pids: &[u32],
    deleted: bool,
    has_progress: bool,
    stdout: &Mutex<std::io::Stdout>,
) {
    match config.format() {
        OutputFormat::Terminal => {
            output::print_ps_result(r, pids, deleted, has_progress, config.extra());
        }
        OutputFormat::Json => {
            let envelope = r.envelope_ref();
            let Ok(mut out) = stdout.lock() else {
                return;
            };
            if let Err(e) = serde_json::to_writer(&mut *out, &envelope) {
                tracing::error!(path = %r.path, "failed to serialize scan result: {e}");
                return;
            }
            let _ = out.write_all(b"\n");
        }
        OutputFormat::Tiny => {
            let Ok(mut out) = stdout.lock() else {
                return;
            };
            crate::engine::write_tiny(&mut *out, r);
        }
    }
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
