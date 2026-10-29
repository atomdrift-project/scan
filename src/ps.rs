//! Process scanning: enumerate running processes, deduplicate by SHA256, and
//! scan each unique executable through the model.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;

use anyhow::Result;

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

/// Rizin's size gate for a live-process scan.
///
/// Unlike a filesystem scan (where we want every architecture and tolerate
/// minutes of deep analysis), `ps` is interactive and dominated by a few giant
/// signed apps (Electron/Bun binaries can be 100–215 MB). Two caps keep it
/// responsive without changing verdicts materially: this gate skips Rizin on
/// the giants entirely, and native-arch-only analysis never pays full `aaa` on
/// a universal binary's non-host slice, which never runs here.
const PS_RIZIN_MAX_BYTES: usize = 100 * 1024 * 1024;

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
            tracing::warn!("cannot hash deleted binary: {}", path.display());
            hash_errors += 1;
            continue;
        };
        let hash = match crate::digest::sha256_file(&source) {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!("cannot hash {} (pid {}): {e}", path.display(), pids[0]);
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

/// Run a process scan: enumerate, deduplicate, classify.
pub fn run(config: &ScanConfig) -> Result<ScanSummary> {
    // Warm cleave's YARA engine + capability mapper off the rayon pool before
    // any scan fires rayon work. See `run_scan_paths` for why this matters.
    crate::engine::prefetch_cleave_resources();
    // Load the model before hashing every running executable, so a broken
    // bundle is reported at once rather than after the slowest step.
    let model = Model::load(config.model_dir(), config.thresholds(), config.level())?;
    let shap = ShapImportance::load(config.model_dir())?;

    let scan_start = Instant::now();
    let is_terminal = matches!(config.format(), OutputFormat::Terminal);

    // Enumerate processes.
    let proc_result = proclist::enumerate().map_err(|e| anyhow::anyhow!("{e}"))?;

    let perm_denied = proc_result.stats.permission_denied;

    // Warn about permission-denied processes.
    if perm_denied > 0 && !is_root() {
        crate::output::print_permission_warning(perm_denied);
    }

    if proc_result.entries.is_empty() {
        let summary = ScanSummary {
            duration_ms: crate::duration_ms(scan_start.elapsed()),
            ..ScanSummary::default()
        };
        if is_terminal {
            output::print_summary(&summary);
        }
        return Ok(summary);
    }

    let (groups, hash_errors) = group_by_binary(&proc_result.entries);
    let total = u32::try_from(groups.len()).unwrap_or(u32::MAX);

    if is_terminal {
        // Detection rules already resident in memory — no extra loading.
        // `prefetch_cleave_resources` above has fully built the YARA engine and
        // capability mapper; the bloom signatures were loaded with the config.
        output::print_banner(crate::engine::detection_counts(config));
    }

    // Scan each unique binary.
    let cancellation = crate::interrupt::arm("Interrupted — finishing current process…");
    cleave::set_compact_member_retention(true); // compact projection only
    let mut cleave_opts = cleave::AnalysisOptions {
        slow_rule_ms: config.slow_rule_ms(),
        cancellation: Some(Arc::clone(&cancellation)),
        // The ps-only caps (see `PS_RIZIN_MAX_BYTES`) are per analysis, so the
        // file scan `scan sys` runs next keeps the defaults.
        rizin_timeout: crate::engine::rizin_timeout(),
        rizin_max_bytes: Some(PS_RIZIN_MAX_BYTES),
        rizin_native_arch_only: true,
        ..Default::default()
    };
    crate::engine::add_zip_passwords(&mut cleave_opts, config.zip_passwords());

    let progress = if is_terminal && total > 1 {
        Some(Progress::new(total))
    } else {
        None
    };

    let hostile = AtomicU32::new(0);
    let suspicious = AtomicU32::new(0);
    let benign = AtomicU32::new(0);
    let errors = AtomicU32::new(hash_errors);
    let dependency_gaps = AtomicU32::new(0);
    let static_flags = AtomicU32::new(0);
    let flagged_candidates = AtomicU32::new(0);

    // Map each scan path back to its process group for PID annotation.
    let mut scan_paths: Vec<PathBuf> = Vec::with_capacity(groups.len());
    let mut by_scan_path: HashMap<PathBuf, &ProcessGroup> = HashMap::with_capacity(groups.len());
    for group in &groups {
        // Known-good/known-bad short-circuit by the executable's sha256: a
        // known-good binary is skipped here (and counted benign) unless it was
        // created/changed/modified within the last 48h, in which case it is
        // scanned on its own merits; known-bad and conflicted are flagged but
        // still analyzed below.
        if let Some(lookup) = config.bloom()
            && let Some(digest) = burton::parse_sha256_hex(&group.sha256)
            && let Some(n) =
                crate::engine::bloom_gate_fresh(config, &group.path, lookup.decide_sha256(&digest))
        {
            benign.fetch_add(n, Ordering::Relaxed);
            if let Some(p) = &progress {
                p.increment();
            }
            continue;
        }

        by_scan_path.insert(group.scan_path.clone(), group);
        scan_paths.push(group.scan_path.clone());
    }

    // Analyze the unique binaries through cleave's batch API: it loads the
    // CapabilityMapper + YARA engine once and fans the files out across a single
    // rayon pass. A hand-rolled `par_iter` over the single-file `analyze_file`
    // would nest rayon inside rayon — each file's inner parallelism multiplied
    // by the outer fan-out oversubscribes the pool and runs *slower* than
    // serial, at about three times the CPU. Verdict order in the output stream
    // is completion order, which JSON/terminal consumers tolerate.

    let analyze = |scan_path: &Path, result: Result<cleave::AnalysisReport>| {
        let Some(group) = by_scan_path.get(scan_path) else {
            return;
        };
        if let Some(p) = &progress {
            p.increment();
        }
        let report = match result {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("error analyzing {}: {e}", group.path.display());
                errors.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        match build_result(
            report,
            group,
            &model,
            shap.as_ref(),
            config,
            Some(cancellation.as_ref()),
        ) {
            Ok(result) => {
                match result.classification {
                    Classification::Hostile => hostile.fetch_add(1, Ordering::Relaxed),
                    Classification::Suspicious => suspicious.fetch_add(1, Ordering::Relaxed),
                    Classification::Benign => benign.fetch_add(1, Ordering::Relaxed),
                };
                // JSON, tiny, and interpret are machine/LLM payload formats:
                // they emit every scanned process — `--show` gates only the
                // terminal view.
                if matches!(
                    config.format(),
                    OutputFormat::Json | OutputFormat::Tiny | OutputFormat::Interpret
                ) || config.filter().shows(result.classification)
                {
                    let print = || {
                        emit_result(&result, config, progress.is_some());
                    };
                    if let Some(p) = &progress {
                        p.around_result(print);
                    } else {
                        print();
                    }
                }
                dependency_gaps.fetch_add(result.dependency_attention.gaps, Ordering::Relaxed);
                static_flags.fetch_add(u32::from(result.has_severe_findings()), Ordering::Relaxed);
                flagged_candidates.fetch_add(
                    result.dependency_attention.flagged_candidates,
                    Ordering::Relaxed,
                );
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

    let summary = ScanSummary {
        total_files: total,
        hostile,
        suspicious,
        benign,
        errors,
        dependency_attention: crate::engine::DependencyAttention {
            gaps: dependency_gaps.load(Ordering::Relaxed),
            flagged_candidates: flagged_candidates.load(Ordering::Relaxed),
        },
        static_flags: static_flags.load(Ordering::Relaxed),
        duration_ms: crate::duration_ms(scan_start.elapsed()),
    };

    if is_terminal {
        // Collapse the progress bar straight into the verdict: when a bar is
        // live it sits on its own line below the banner, so clear it and write
        // the one-line summary in its place. The finished scan then shows just
        // the banner and this line — no intermediate "files in" line, no
        // separator, no trailing blank.
        let line = output::summary_status_line(&summary);
        if let Some(p) = &progress {
            p.quiesce(); // stop the heartbeat before we claim the bar's line
            eprint!("\r\x1b[2K {line}\n");
            let _ = std::io::stderr().flush();
        } else {
            eprintln!(" {line}");
        }
    }

    Ok(summary)
}

/// Build a ScanResult from a cleave report, injecting pids and deleted state.
fn build_result(
    report: cleave::AnalysisReport,
    group: &ProcessGroup,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
    cancellation: Option<&AtomicBool>,
) -> Result<ScanResult> {
    let display_path = group.path.as_path();
    // The bloom flag for a known-bad/conflicted process image, re-derived from the
    // executable's sha256 (known-good is short-circuited before we ever scan, so it
    // never reaches here). Drives the inline 🚩/🏴 in the terminal header.
    let bloom_mark = config
        .bloom()
        .and_then(|lk| burton::parse_sha256_hex(&group.sha256).map(|d| lk.decide_sha256(&d)))
        .and_then(crate::output::BloomMark::from_decision);
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
            needs: crate::engine::OutputNeeds {
                // Matching scan's `--format interpret`: the live LLM query's user
                // message, byte-for-byte, without the system prompt.
                llm_view: matches!(config.format(), OutputFormat::Interpret),
                fetch_progress: false, // machine-readable output; no live fetch log
                render_context: matches!(
                    config.format(),
                    OutputFormat::Tiny | OutputFormat::Interpret
                ),
                // `--show=all` with JSON: list every member of an archive-backed image.
                list_all_members: config.filter().is_all()
                    && matches!(config.format(), OutputFormat::Json),
                deps_for_upload: false, // ps has no hopper renewal
            },
            bloom_mark,
            ..crate::engine::ClassifyRequest::new(&label, display_path, model)
        },
    )?;
    let is_json = matches!(config.format(), OutputFormat::Json);

    let mut result = cr.into_scan_result(label, model, is_json);
    result.pids = Some(group.pids.clone());
    result.deleted = group.deleted.then_some(true);
    result.sha256.clone_from(&group.sha256);
    result.bloom_mark = bloom_mark;
    Ok(result)
}

/// Emit a single result with PID annotations.
fn emit_result(r: &ScanResult, config: &ScanConfig, has_progress: bool) {
    match config.format() {
        OutputFormat::Terminal => {
            output::print_ps_result(r, has_progress, config.extra());
        }
        // Serialized before stdout is locked, so a slow serialization never
        // holds up another thread's line.
        OutputFormat::Json => match serde_json::to_vec(&r.envelope_ref()) {
            Ok(mut bytes) => {
                bytes.push(b'\n');
                let _ = std::io::stdout().lock().write_all(&bytes);
            }
            Err(e) => tracing::error!(path = %r.path, "failed to serialize scan result: {e}"),
        },
        OutputFormat::Tiny => {
            if let Err(e) = crate::engine::write_tiny(&mut std::io::stdout().lock(), r) {
                tracing::error!(path = %r.path, "failed to write scan result: {e}");
            }
        }
        // The LLM payload verbatim — no verdict line, no PID annotations.
        OutputFormat::Interpret => {
            let _ = std::io::stdout()
                .lock()
                .write_all(r.rendered_context.as_bytes());
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
