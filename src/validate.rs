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
///
/// When `skip_traits` is set, both the explicit cleave trait validation AND the
/// benign-corpus inference are skipped — the latter necessarily, because it loads
/// the trait corpus to extract features. Only the trait-independent model check
/// (feature-layout) runs. Use it to validate a model bundle independently of
/// trait-corpus churn; traits are versioned separately from the deployed model.
pub fn run(config: &ScanConfig, skip_traits: bool) -> Result<()> {
    // Feature-layout validation is trait-independent: load the model and reject a
    // structurally incompatible bundle deterministically, before any trait corpus
    // is touched or any file is analyzed. The benign corpus below can't be relied
    // on to exercise every offset-written family (the unsigned-bigram overflow
    // only triggers on packed/unsigned samples), so anchor-failure must fail here.
    let model = Model::load(config.model_dir(), config.thresholds(), config.level())?;
    let thresholds = model.thresholds();
    let ctx = ExtractContext::new(model.spec());
    ctx.validate_layout().context("feature layout validation")?;

    if skip_traits {
        // The benign-corpus inference below requires loading the cleave trait
        // corpus (the CapabilityMapper extracts features through it), which would
        // re-run trait validation. --skip-traits validates the model structurally
        // (feature layout) and skips the trait-dependent benign-corpus pass —
        // traits are versioned separately from the deployed model.
        eprintln!(
            "validate ok (--skip-traits): model feature layout valid; \
             benign-corpus inference skipped (it requires the cleave trait corpus)"
        );
        return Ok(());
    }

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
                        None,
                        &cleave::output::TinyOpts::tiny(),
                    )
                });
            (path, result)
        })
        .collect();

    let (passed, total, warnings) = evaluate(results, thresholds)?;

    // The corpus pass above sets this if any extraction saw feature-layout drift
    // (total_features > cursor: feature_names slots no writer fills, extracting
    // to zero). Fail before reporting ok — a drifted bundle must not deploy.
    if ctx.had_layout_drift() {
        anyhow::bail!(
            "feature-layout drift: the spec declares more features than the extractor \
             writes, so some feature_names slots extract to zero (see WARN above); \
             resync features.rs layout constants with collimator before deploying"
        );
    }

    // Every target is a known-benign file (platform utilities + cleave's
    // does-nothing corpus), so any non-benign grade is a false positive — a
    // quality regression that must block the deploy, not merely print a warning.
    if warnings > 0 {
        anyhow::bail!(
            "{warnings} benign-corpus sample(s) graded non-benign (false positives); \
             see the WARN lines above"
        );
    }

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
                "WARN {}: grade={} level={} probability={:.4} decision_threshold={:.4} active_thresholds suspicious={:.4} hostile={:.4}",
                path.display(),
                result.classification,
                format_level(result.level),
                result.probability,
                result.threshold,
                thresholds.suspicious,
                thresholds.hostile,
            );
            print_top_findings(&result.top_findings, "  ");
        }
        for embedded in &result.embedded_files {
            if embedded.classification != Classification::Benign {
                file_warned = true;
                eprintln!(
                    "WARN {}!!{}: grade={} level={} probability={:.4} decision_threshold={:.4} active_thresholds suspicious={:.4} hostile={:.4}",
                    path.display(),
                    embedded.path,
                    embedded.classification,
                    format_level(embedded.level),
                    embedded.probability,
                    embedded.threshold,
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

fn format_level(level: Option<i32>) -> String {
    match level {
        Some(-1) => "clean".to_string(),
        Some(n) => format!("L{n}"),
        None => "manual".to_string(),
    }
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
