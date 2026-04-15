//! Process scanning: enumerate running processes, deduplicate by SHA256, and
//! scan each unique executable through the litmus model.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Classification, Model};
use crate::output;
use crate::scan::{Progress, ScanConfig, ScanResult, ScanSummary};
use crate::OutputFormat;

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
            duration_ms: scan_start.elapsed().as_millis() as u64,
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

    let groups: Vec<ProcessGroup> = by_sha.into_values().collect();
    let total = groups.len() as u32;

    if is_terminal {
        eprintln!(
            "\n  \x1b[38;2;120;180;255m\u{25c6}\x1b[0m  {} unique binaries across {} processes\n",
            total,
            proc_result.entries.len(),
        );
    }

    // Load model and scan each unique binary.
    let model = Model::load(config.model_dir(), config.thresholds())?;
    let shap = ShapImportance::load(config.model_dir()).ok();
    let ctx = ExtractContext::new(model.spec());
    let cancellation = Arc::new(AtomicBool::new(false));
    let ctrlc_flag = Arc::clone(&cancellation);
    let _ = ctrlc::set_handler(move || {
        if ctrlc_flag.load(Ordering::Relaxed) {
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

    let mut hostile = 0u32;
    let mut suspicious = 0u32;
    let mut benign = 0u32;
    let mut errors = hash_errors;

    for group in &groups {
        if cancellation.load(Ordering::Relaxed) {
            break;
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
                errors += 1;
                continue;
            }
        } else {
            group.path.clone()
        };

        let report = match cleave::analyze_file(&scan_path, &cleave_opts) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("error analyzing {}: {e}", group.path.display());
                errors += 1;
                if let Some(p) = &progress {
                    p.increment();
                }
                continue;
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
                    Classification::Hostile => hostile += 1,
                    Classification::Suspicious => suspicious += 1,
                    Classification::Benign => benign += 1,
                }
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
                errors += 1;
            }
        }
    }

    if let Some(p) = &progress {
        p.finish();
    }

    let summary = ScanSummary {
        total_files: total,
        hostile,
        suspicious,
        benign,
        errors,
        duration_ms: scan_start.elapsed().as_millis() as u64,
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
    let cr = crate::scan::classify_report(
        &display_path.display().to_string(),
        report,
        ctx,
        model,
        shap,
        cancellation,
    )?;
    let is_json = matches!(config.format(), OutputFormat::Json);

    let cleave = if is_json { Some(cr.report_json) } else { None };
    let thresholds = model.thresholds();

    Ok(ScanResult {
        v: "4",
        classification: cr.classification,
        probability: cr.probability,
        thresholds,
        version: crate::scan::model_version_string(model.info()),
        analyzed_at: crate::scan::now_rfc3339(),
        cleave,
        pids: Some(group.pids.clone()),
        deleted: group.deleted.then_some(true),
        path: display_path.display().to_string(),
        finding_counts: cr.finding_counts,
        formula: cr.formula,
        reasons: cr.reasons,
        top_findings: cr.top_findings,
        file_type: cr.file_type,
        size_bytes: cr.size_bytes,
        sha256: group.sha256.clone(),
        embedded_files: cr.embedded_files,
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
        OutputFormat::Json => match serde_json::to_string(&r.to_envelope()) {
            Ok(line) => {
                if let Ok(mut out) = stdout.lock() {
                    let _ = writeln!(out, "{line}");
                }
            }
            Err(e) => {
                tracing::error!(path = %r.path, "failed to serialize scan result: {e}");
            }
        },
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
