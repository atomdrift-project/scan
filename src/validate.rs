//! Validation command support.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::features::ExtractContext;
use crate::model::{Classification, Model, Thresholds};
use crate::scan::{self, ClassifiedReport, EmbeddedFile, ScanConfig};

/// Run full validation: cleave trait validation, model loading, and benign-corpus inference.
///
/// The analyzed corpus mirrors `cleave validate`: common platform utilities plus
/// every file in the cleave traits `testdata/does-nothing` tree.
pub fn run(config: &ScanConfig) -> Result<()> {
    let targets = collect_targets()?;

    let cleave_output = cleave::commands::validate::run(&cleave::cli::OutputFormat::Terminal, None)
        .context("cleave validate")?;
    print!("{cleave_output}");

    // Keep model validation aligned with `cleave validate`: no YARA/radare2/UPX,
    // one mapper shared by all target analyses, and the same benign corpus.
    cleave::cache::set_skip_cache_override(Some(true));
    let options = cleave::AnalysisOptions {
        disable_yara: true,
        disable_radare2: true,
        disable_upx: true,
        slow_rule_ms: config.slow_rule_ms(),
        ..Default::default()
    };
    let mapper = Arc::new(cleave::CapabilityMapper::try_new_with_load_options(
        cleave::CapabilityMapper::DEFAULT_MIN_HOSTILE_PRECISION,
        cleave::CapabilityMapper::DEFAULT_MIN_SUSPICIOUS_PRECISION,
        true,
        false,
    )?);

    let model = Model::load(config.model_dir(), config.thresholds())?;
    let thresholds = model.thresholds();
    let ctx = ExtractContext::new(model.spec());

    let results: Vec<(PathBuf, Result<ClassifiedReport>)> = targets
        .into_par_iter()
        .map(|path| {
            let result = cleave::analyze_file_with_mapper(&path, &options, &mapper)
                .with_context(|| format!("cleave analysis of {}", path.display()))
                .and_then(|report| {
                    scan::classify_report(
                        &path.display().to_string(),
                        report,
                        &ctx,
                        &model,
                        None,
                        None,
                        config.upgrade_heuristic(),
                        None,
                    )
                });
            (path, result)
        })
        .collect();

    let (passed, total, warnings) = evaluate(results, thresholds)?;

    let models_ver = crate::models_repo::version()
        .map(|v| format!("  models: {v}"))
        .unwrap_or_default();
    eprintln!(
        "validate ok:{models_ver}  benign corpus {}/{}  warnings={}",
        passed, total, warnings,
    );
    Ok(())
}

fn collect_targets() -> Result<Vec<PathBuf>> {
    let mut targets = Vec::new();

    for path in [
        "/bin/ls",
        "/bin/cp",
        "/bin/sh",
        "/usr/bin/curl",
        "/bin/capsh",
        "/bin/sulogin",
        "/bin/gpgconf",
        "/usr/lib/systemd/system/arptables.service",
    ] {
        let p = PathBuf::from(path);
        if p.exists() {
            targets.push(p);
        }
    }

    if let Ok(traits_dir) = cleave::traits_repo::try_resolve() {
        let dn_dir = traits_dir.join("testdata").join("does-nothing");
        if dn_dir.is_dir() {
            walk_files(&dn_dir, &mut targets)?;
        }
    }

    Ok(targets)
}

fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("reading directory {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            if entry.file_name().to_string_lossy().starts_with(".git") {
                continue;
            }
            walk_files(&path, out)?;
        } else if file_type.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn evaluate(
    results: Vec<(PathBuf, Result<ClassifiedReport>)>,
    thresholds: Thresholds,
) -> Result<(usize, usize, usize)> {
    let mut passed = 0usize;
    let mut total = 0usize;
    let mut analysis_failed = 0usize;
    let mut warnings = 0usize;

    for (path, result) in results {
        total += 1;
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                analysis_failed += 1;
                eprintln!("FAILED {}: analysis failed: {error:#}", path.display());
                continue;
            }
        };

        let mut file_warned = false;
        if result.classification != Classification::Benign {
            file_warned = true;
            eprintln!(
                "WARN {}: grade={} probability={:.4} thresholds suspicious={:.4} hostile={:.4}",
                path.display(),
                result.classification,
                result.probability,
                thresholds.suspicious,
                thresholds.hostile,
            );
            print_top_findings(&result.top_findings, "  ");
        }
        for embedded in &result.embedded_files {
            if embedded.classification != Classification::Benign {
                file_warned = true;
                eprintln!(
                    "WARN {}!!{}: grade={} probability={:.4} thresholds suspicious={:.4} hostile={:.4}",
                    path.display(),
                    embedded.path,
                    embedded.classification,
                    embedded.probability,
                    thresholds.suspicious,
                    thresholds.hostile,
                );
                print_embedded_findings(embedded, "  ");
            }
        }

        if file_warned {
            warnings += 1;
        } else {
            passed += 1;
        }
    }

    if analysis_failed > 0 {
        anyhow::bail!(
            "{analysis_failed} validation check(s) failed during analysis ({passed}/{total} targets benign, {warnings} warning(s))"
        );
    }
    Ok((passed, total, warnings))
}

fn print_top_findings(findings: &[scan::TopFinding], indent: &str) {
    for finding in findings {
        eprintln!("{indent}l{} {}  {}", finding.crit, finding.id, finding.desc);
    }
}

fn print_embedded_findings(file: &EmbeddedFile, indent: &str) {
    for finding in &file.top_findings {
        eprintln!("{indent}l{} {}  {}", finding.crit, finding.id, finding.desc);
    }
}
