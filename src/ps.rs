//! Process scanning: enumerate running processes, deduplicate by SHA256, and
//! scan each unique executable through the litmus model.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Classification, Model};
use crate::output;
use crate::scan::{DisplayFilter, ScanResult, ScanSummary};
use crate::OutputFormat;

/// Configuration for process scanning (mirrors ScanConfig fields).
#[derive(Debug)]
pub struct PsConfig {
    /// Directory containing model.onnx and feature_spec.json.
    pub model_dir: PathBuf,
    /// Output format.
    pub format: OutputFormat,
    /// Minimum probability to classify as suspicious.
    pub threshold_suspicious: f32,
    /// Minimum probability to classify as hostile.
    pub threshold_hostile: f32,
    /// Which classifications to display.
    pub filter: DisplayFilter,
    /// Warn when a single rule takes longer than this many milliseconds (default: 4000).
    pub slow_rule_ms: u64,
}

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
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// SHA256 a file via /proc/pid/exe (Linux only — for deleted binaries).
#[cfg(target_os = "linux")]
fn sha256_proc_exe(pid: u32) -> Result<String> {
    let proc_path = format!("/proc/{pid}/exe");
    sha256_file(std::path::Path::new(&proc_path))
}

/// Run a process scan: enumerate, deduplicate, classify.
pub fn run(config: &PsConfig) -> Result<ScanSummary> {
    let scan_start = Instant::now();
    let is_terminal = matches!(config.format, OutputFormat::Terminal);

    // Enumerate processes.
    let proc_result = proclist::enumerate()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

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
                let first_pid = pids[0];
                match sha256_proc_exe(first_pid) {
                    Ok(h) => h,
                    Err(e) => {
                        tracing::warn!("cannot hash deleted binary {} (pid {}): {e}", path.display(), first_pid);
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

        match by_sha.get_mut(&hash) {
            Some(group) => {
                group.pids.extend(pids);
                if deleted {
                    group.deleted = true;
                }
            }
            None => {
                by_sha.insert(
                    hash.clone(),
                    ProcessGroup {
                        path,
                        pids,
                        deleted,
                        sha256: hash,
                    },
                );
            }
        }
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
    let model = Model::load(
        &config.model_dir,
        crate::model::Thresholds {
            suspicious: config.threshold_suspicious,
            hostile: config.threshold_hostile,
        },
    )?;
    let shap = ShapImportance::load(&config.model_dir).ok();
    let ctx = ExtractContext::new(&model.spec);
    let cleave_opts = cleave::AnalysisOptions { slow_rule_ms: config.slow_rule_ms, ..Default::default() };
    let stdout = Mutex::new(std::io::stdout());

    let mut hostile = 0u32;
    let mut suspicious = 0u32;
    let mut benign = 0u32;
    let mut errors = hash_errors;

    for group in &groups {
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
                continue;
            }
        };

        match build_result(&group.path, report, &ctx, &model, shap.as_ref(), config, group) {
            Ok(result) => {
                match result.classification {
                    Classification::Hostile => hostile += 1,
                    Classification::Suspicious => suspicious += 1,
                    Classification::Benign => benign += 1,
                }
                if config.format == OutputFormat::Json || config.filter.shows(&result.classification) {
                    emit_result(&result, config, &group.pids, group.deleted, &stdout);
                }
            }
            Err(e) => {
                tracing::warn!("error processing {}: {e}", group.path.display());
                errors += 1;
            }
        }
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
fn build_result(
    display_path: &std::path::Path,
    mut report: cleave::AnalysisReport,
    ctx: &ExtractContext,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &PsConfig,
    group: &ProcessGroup,
) -> Result<ScanResult> {
    let formula = cleave::formula_from_report(&report);
    report.finalize();

    let report_json = serde_json::to_value(&report).context("serializing cleave report")?;
    let mut features = ctx.extract(&report_json);
    model.spec.standardize(&mut features);
    let (probability, classification) = model.predict(&features)?;

    let finding_counts = crate::scan::count_findings_from_json(&report_json);

    let (reasons, top_findings) = if classification != Classification::Benign {
        let r = shap
            .map(|s| s.explain(&features, &model.spec.feature_names, 5))
            .unwrap_or_default();
        let f = crate::scan::extract_top_findings_from_json(&report_json, &classification);
        (r, f)
    } else {
        (vec![], vec![])
    };

    let pf = report_json["files"]
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or(&report_json);
    let file_type = pf["file_type"].as_str().unwrap_or("unknown").to_string();
    let size_bytes = pf["size"].as_u64().unwrap_or(0);

    let is_json = matches!(config.format, OutputFormat::Json);

    Ok(ScanResult {
        path: display_path.display().to_string(),
        classification,
        probability,
        thresholds: crate::model::Thresholds {
            suspicious: config.threshold_suspicious,
            hostile: config.threshold_hostile,
        },
        finding_counts,
        formula,
        reasons,
        top_findings,
        file_type,
        size_bytes,
        sha256: group.sha256.clone(),
        model: if is_json { Some(model.info.clone()) } else { None },
        cleave: if is_json { Some(report_json) } else { None },
        pids: Some(group.pids.clone()),
        deleted: if group.deleted { Some(true) } else { None },
    })
}

/// Emit a single result with PID annotations.
fn emit_result(
    r: &ScanResult,
    config: &PsConfig,
    pids: &[u32],
    deleted: bool,
    stdout: &Mutex<std::io::Stdout>,
) {
    match config.format {
        OutputFormat::Terminal => {
            output::print_ps_result(r, pids, deleted);
        }
        OutputFormat::Json => {
            if let Ok(line) = serde_json::to_string(r) {
                if let Ok(mut out) = stdout.lock() {
                    let _ = writeln!(out, "{line}");
                }
            }
        }
    }
}

/// Check if running as root/admin.
fn is_root() -> bool {
    #[cfg(unix)]
    {
        unsafe { libc::getuid() == 0 }
    }
    #[cfg(not(unix))]
    {
        false // Conservative: always show the warning on non-unix
    }
}
