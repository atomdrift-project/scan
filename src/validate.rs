//! Validation command support.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::engine::{self, ClassifiedReport, ScanConfig};
use crate::features::ExtractContext;
use crate::model::{Classification, Model, Thresholds};

const PROGRESS_EVERY: usize = 10;
const SLOW_FIXTURE_THRESHOLD: Duration = Duration::from_secs(15);

/// Run validation: model loading, feature-layout checks, and benign fixture inference.
///
/// The analyzed corpus mirrors the Atomdrift model false-positive gate: common
/// platform utilities plus every file in the cleave traits `testdata/does-nothing`
/// tree. Full cleave trait-rule validation remains the job of `cleave validate`;
/// running it here as a second uncached pass made this command too slow for a
/// local deploy/pre-commit gate.
pub fn run(config: &ScanConfig, skip_traits: bool) -> Result<()> {
    // Feature-layout validation is trait-independent: load the model and reject a
    // structurally incompatible bundle deterministically, before any trait corpus
    // is touched or any file is analyzed. The benign smoke pass below can't be relied
    // on to exercise every offset-written family (the unsigned-bigram overflow
    // only triggers on packed/unsigned samples), so anchor-failure must fail here.
    let model = Model::load(config.model_dir(), config.thresholds(), config.level())?;
    model
        .validate_all_routes()
        .context("validating specialist model routes")?;
    let thresholds = model.thresholds();
    let ctx = ExtractContext::new(model.spec());
    ctx.validate_layout().context("feature layout validation")?;

    // A model whose spec declares features this build's extractor cannot
    // produce is a hard failure in validate mode, not a warning: those slots
    // extract to zero, so the deployed model is silently degraded relative to
    // its training. A normal scan only WARNs here (it degrades gracefully), but
    // a deploy gate must not let a degraded model through. Absent *optional*
    // features — feature groups disabled at training, the normal subset case —
    // never appear in this list, so they stay non-fatal.
    let degraded = model.spec().degraded_feature_names();
    if !degraded.is_empty() {
        let preview: Vec<&str> = degraded.iter().map(String::as_str).take(10).collect();
        anyhow::bail!(
            "model degraded: feature_spec.json declares {} feature(s) this litmus build's \
             extractor cannot produce, so they extract as zeros (e.g. {preview:?}); the model \
             is out of sync with collimator — rebuild litmus with matching feature extraction \
             before deploying",
            degraded.len(),
        );
    }

    if skip_traits {
        let models_ver = crate::models_repo::version()
            .map(|v| format!("  models: {v}"))
            .unwrap_or_default();
        eprintln!(
            "validate ok:{models_ver}  model feature layout valid; \
             benign fixture inference skipped"
        );
        return Ok(());
    }

    let targets = collect_targets()?;
    if targets.is_empty() {
        anyhow::bail!("no benign fixture targets found");
    }

    // Keep model fixture validation cheap and deterministic: no YARA/radare2/UPX,
    // one mapper shared by all target analyses, and analysis caching enabled.
    // The cache key includes the traits revision, so current trait edits still
    // invalidate stale reports without forcing every pre-commit run to rescan
    // the whole cleave fixture tree.
    cleave::cache::set_skip_cache_override(Some(false));
    cleave::set_compact_member_retention(true); // compact projection only
    let mut options = cleave::AnalysisOptions {
        disable_yara: true,
        disable_radare2: true,
        disable_upx: true,
        slow_rule_ms: config.slow_rule_ms(),
        ..Default::default()
    };
    crate::engine::add_zip_passwords(&mut options, config.zip_passwords());
    let mapper = Arc::new(cleave::CapabilityMapper::try_new_with_load_options(
        cleave::CapabilityMapper::DEFAULT_MIN_HOSTILE_PRECISION,
        cleave::CapabilityMapper::DEFAULT_MIN_SUSPICIOUS_PRECISION,
        false,
        false,
    )?);

    let total_targets = targets.len();
    eprintln!("validate fixtures: scanning {total_targets} benign targets...");
    let completed = AtomicUsize::new(0);
    let slow_fixtures = Mutex::new(Vec::new());

    let results: Vec<(PathBuf, Result<ClassifiedReport>)> = targets
        .into_par_iter()
        .map(|path| {
            let started = Instant::now();
            let analysis_started = Instant::now();
            let result = cleave::analyze_file_with_mapper(&path, &options, &mapper)
                .with_context(|| format!("cleave analysis of {}", path.display()))
                .and_then(|report| {
                    let analysis_elapsed = analysis_started.elapsed();
                    let classify_started = Instant::now();
                    let classified = engine::classify_report(
                        &path.display().to_string(),
                        report,
                        &ctx,
                        &model,
                        None,
                        None,
                        &cleave::output::TinyOpts::tiny(),
                        None, // validation corpus never calls the LLM
                        &path,
                        crate::fetch::FetchPolicy::default(),
                        config.zip_passwords(),
                        // Validation consumes ML verdicts only — no renders,
                        // no manifest listing, no dependency uploads.
                        engine::OutputNeeds::default(),
                        None, // validation fixtures are local files, not fetched packages
                        None, // local validation fixtures have no acquisition fetch record
                        None, // validation consumes ML verdicts only; no bloom flag
                        None,
                        None, // no admission gate
                    );
                    let classify_elapsed = classify_started.elapsed();
                    if analysis_elapsed > SLOW_FIXTURE_THRESHOLD
                        || classify_elapsed > SLOW_FIXTURE_THRESHOLD
                    {
                        eprintln!(
                            "SLOW fixture stages {}: analysis={:.1}s classify={:.1}s",
                            path.display(),
                            analysis_elapsed.as_secs_f64(),
                            classify_elapsed.as_secs_f64()
                        );
                    }
                    classified
                });
            let elapsed = started.elapsed();
            if elapsed > SLOW_FIXTURE_THRESHOLD {
                eprintln!(
                    "SLOW fixture {}: {:.1}s",
                    path.display(),
                    elapsed.as_secs_f64()
                );
                if let Ok(mut slow) = slow_fixtures.lock() {
                    slow.push((path.clone(), elapsed));
                }
            }
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done == total_targets || done.is_multiple_of(PROGRESS_EVERY) {
                eprintln!("validate fixtures: {done}/{total_targets} complete");
            }
            (path, result)
        })
        .collect();

    let slow_count = slow_fixtures.lock().map_or(0, |slow| slow.len());
    if slow_count > 0 {
        eprintln!("validate fixtures: {slow_count} fixture(s) exceeded 15s");
    }

    let (passed, total, hostile_fps, suspicious_fps) = evaluate(results, thresholds)?;

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
    // does-nothing corpus). A HOSTILE grade on any of them is a hard false
    // positive that must block the deploy. A merely Suspicious grade is a
    // softer signal the operator tolerates on benign input: report it, but do
    // not fail the gate (suspicious is allowed to be suspicious).
    if hostile_fps > 0 {
        anyhow::bail!(
            "{hostile_fps} benign fixture sample(s) graded HOSTILE (false positives); \
             see the WARN lines above. ({suspicious_fps} graded suspicious — tolerated.)"
        );
    }

    let models_ver = crate::models_repo::version()
        .map(|v| format!("  models: {v}"))
        .unwrap_or_default();
    if suspicious_fps > 0 {
        eprintln!(
            "validate ok (with {suspicious_fps} suspicious — tolerated):{models_ver}  \
             benign fixtures {passed}/{total} clean, {suspicious_fps} suspicious, 0 hostile"
        );
    } else {
        eprintln!(
            "validate ok:{models_ver}  benign fixtures {passed}/{total}  0 suspicious  0 hostile"
        );
    }
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
        if !p.exists() {
            continue;
        }
        targets.push(p);
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
) -> Result<(usize, usize, usize, usize)> {
    let mut passed = 0usize;
    let mut total = 0usize;
    let mut analysis_failed = 0usize;
    let mut hostile_fps = 0usize;
    let mut suspicious_fps = 0usize;

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

        // Worst grade across the sample and any embedded artifact decides the
        // outcome: a benign fixture file is a HARD failure only if something in
        // it grades Hostile. A merely Suspicious top-grade is reported but
        // tolerated — the operator accepts suspicious on benign input.
        let mut any_nonbenign = false;
        let mut any_hostile = false;
        if result.classification != Classification::Benign {
            any_nonbenign = true;
            any_hostile |= result.classification == Classification::Hostile;
            eprintln!(
                "WARN {}: grade={} level={} probability={} decision_threshold={} margin_logit={:+.3} ensemble_thresholds suspicious={} hostile={}",
                path.display(),
                result.classification,
                format_level(result.level),
                result.probability,
                result.threshold,
                logit_margin(result.probability, result.threshold),
                thresholds.suspicious,
                thresholds.hostile,
            );
            for finding in &result.top_findings {
                eprintln!("  l{} {}  {}", finding.crit, finding.id, finding.desc);
            }
        }
        for embedded in result.embedded_files.values() {
            if embedded.classification != Classification::Benign {
                any_nonbenign = true;
                any_hostile |= embedded.classification == Classification::Hostile;
                eprintln!(
                    "WARN {}!!{}: grade={} level={} probability={} decision_threshold={} margin_logit={:+.3} ensemble_thresholds suspicious={} hostile={}",
                    path.display(),
                    embedded.path,
                    embedded.classification,
                    format_level(embedded.level),
                    embedded.probability,
                    embedded.threshold,
                    logit_margin(embedded.probability, embedded.threshold),
                    thresholds.suspicious,
                    thresholds.hostile,
                );
                for finding in &embedded.top_findings {
                    eprintln!("  l{} {}  {}", finding.crit, finding.id, finding.desc);
                }
            }
        }

        if any_hostile {
            hostile_fps += 1;
        } else if any_nonbenign {
            suspicious_fps += 1;
        } else {
            passed += 1;
        }
    }

    if analysis_failed > 0 {
        anyhow::bail!(
            "{analysis_failed} validation check(s) failed during analysis ({passed}/{total} targets benign, {hostile_fps} hostile FP, {suspicious_fps} suspicious)"
        );
    }
    Ok((passed, total, hostile_fps, suspicious_fps))
}

/// Decision margin in log-odds: `logit(probability) - logit(threshold)`.
/// Positive means the file crossed its threshold and fired.
///
/// Fixed-decimal probabilities cannot express these decisions. Both ends of the
/// range are degenerate under `{:.4}`: a malformed bundle once decided an
/// OpenDocument file at probability 1.031e-05 against a threshold of
/// 8.072e-06, printing `probability=0.0000 decision_threshold=0.0000`, and the
/// far commoner case of a threshold at 0.99954 against a score of 0.99955
/// prints both as `0.9995`. Either way the WARN line shows a verdict with no
/// visible cause, on the one line an operator has to work from.
///
/// Log-odds is the scale the thresholds are actually built in — collimator
/// fits its whole per-level threshold curve in logit space — so equal steps
/// here are equal steps of evidence, and one signed number says both which way
/// the decision went and by how much. The raw probability and threshold are
/// still printed alongside, at full precision.
fn logit_margin(probability: f32, threshold: f32) -> f64 {
    // f32 probabilities saturate at both ends; clamp inside the representable
    // open interval so a saturated score yields a large finite margin rather
    // than an infinity.
    fn logit(p: f64) -> f64 {
        let p = p.clamp(1e-45, 1.0 - f64::from(f32::EPSILON));
        (p / (1.0 - p)).ln()
    }
    logit(f64::from(probability)) - logit(f64::from(threshold))
}

fn format_level(level: Option<i32>) -> String {
    match level {
        Some(-1) => "clean".to_string(),
        Some(n) => format!("L{n}"),
        None => "manual".to_string(),
    }
}

#[cfg(test)]
mod logit_margin_tests {
    use super::logit_margin;

    /// The 2026-08-04 OpenDocument misgrade. Under `{:.4}` the probability and
    /// its threshold both printed as `0.0000`; the margin says it fired, and
    /// by how little.
    #[test]
    fn separates_a_decision_at_the_bottom_of_the_range() {
        let margin = logit_margin(1.031_160_4e-5, 8.072_087e-6);
        assert!(margin > 0.0, "file crossed its threshold: {margin}");
        assert!((margin - 0.245).abs() < 0.01, "{margin}");
    }

    /// The commoner case, and the one plain `{:.4}` also loses: a threshold at
    /// 0.99954 against a score just under it, both printing as `0.9995`.
    #[test]
    fn separates_a_decision_at_the_top_of_the_range() {
        let margin = logit_margin(0.999_541_5, 0.999_545_6);
        assert!(margin < 0.0, "file stayed under its threshold: {margin}");
        assert!(
            margin.abs() < 0.05,
            "a near-miss is a small margin: {margin}"
        );
    }

    /// A saturated f32 score must not produce an infinity.
    #[test]
    fn saturated_scores_stay_finite() {
        assert!(logit_margin(1.0, 0.5).is_finite());
        assert!(logit_margin(0.0, 0.5).is_finite());
        assert!(logit_margin(1.0, 1.0).is_finite());
    }
}
