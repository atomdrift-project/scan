//! Validation command support.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::engine::{self, ClassifiedReport, ScanConfig};
use crate::model::{Classification, Decision, Model, Thresholds};

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
    // Model validation is trait-independent: load the model and every specialist
    // and reject a structurally incompatible bundle deterministically, before any
    // trait corpus is touched or any file is analyzed.
    let model = Model::load(config.model_dir(), config.thresholds(), config.level())?;
    model
        .validate_all_routes()
        .context("validating specialist model routes")?;
    let thresholds = model.thresholds();

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

    // The LLM admission gate names trait families and ids directly, so a
    // taxonomy move breaks it silently: the prefixes still compile, match
    // nothing, and quietly stop escalating the samples they were carrying.
    // Check them against the installed tree whenever one resolves. Skipped
    // along with the rest of the trait work under `--skip-traits`, and a no-op
    // when no traits are installed — there is nothing to compare against.
    if !skip_traits && let Ok(traits_dir) = cleave::traits_repo::try_resolve() {
        let problems = crate::interpret::validate_gate_prefixes(&traits_dir);
        if !problems.is_empty() {
            for problem in &problems {
                eprintln!("WARNING {problem}");
            }
            // A warning, not a failure. A stale prefix costs the coverage of
            // that one prefix; refusing to start costs the whole node, and
            // this check compares against whichever traits revision happens to
            // be installed — which the traits auto-update can change under a
            // running fleet, with no scan release involved. Blocking on it
            // meant a traits publish could stop every worker from booting,
            // which is a far worse failure than the one being guarded against.
            // This command is the only place the installed tree is checked:
            // the test suite deliberately asserts nothing about it, since trait
            // definitions are validated in the traits repo, not here.
            eprintln!(
                "WARNING {} LLM gate prefix(es) match nothing in {}; \
                 the gate still admits on criticality, probability and level, \
                 so this narrows coverage rather than disabling it",
                problems.len(),
                traits_dir.display(),
            );
        }
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
    // one engine shared by all target analyses, and analysis caching enabled.
    // The cache key includes the traits revision, so current trait edits still
    // invalidate stale reports without forcing every pre-commit run to rescan
    // the whole cleave fixture tree. The cache override is put back when this
    // returns; compact member retention is the mode every scan entry point
    // sets for the process, so it stays.
    let _cache = CacheOverride::force_enabled();
    cleave::set_compact_member_retention(true); // compact projection only
    let mut options = cleave::AnalysisOptions {
        disable_yara: true,
        disable_radare2: true,
        disable_upx: true,
        slow_rule_ms: config.slow_rule_ms(),
        ..Default::default()
    };
    crate::engine::add_zip_passwords(&mut options, config.zip_passwords());
    // The engine carries the rules and the compact member folding production
    // scans use, so each fixture's members are folded under the rules it is
    // evaluated with. Folding used to consult a process-global mapper that
    // validation never loads, and member features production keeps went
    // missing here.
    let engine = cleave::Engine::for_options(&options)?.with_compact_members(true);

    let total_targets = targets.len();
    eprintln!("validate fixtures: scanning {total_targets} benign targets...");
    let completed = AtomicUsize::new(0);
    let slow_fixtures = Mutex::new(Vec::new());

    let results: Vec<(PathBuf, Result<ClassifiedReport>)> = targets
        .into_par_iter()
        .map(|path| {
            let started = Instant::now();
            let analysis_started = Instant::now();
            let result = engine
                .analyze_file(&path, &options)
                .with_context(|| format!("cleave analysis of {}", path.display()))
                .and_then(|report| {
                    let analysis_elapsed = analysis_started.elapsed();
                    let classify_started = Instant::now();
                    let label = path.display().to_string();
                    // Validation consumes ML verdicts only: no LLM, renders,
                    // manifest listing, or dependency uploads.
                    let classified = engine::classify_report(
                        report,
                        engine::ClassifyRequest {
                            zip_passwords: config.zip_passwords(),
                            ..engine::ClassifyRequest::new(&label, &path, &model)
                        },
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

    let FixtureTally {
        passed,
        total,
        hostile_fps,
        suspicious_fps,
    } = evaluate(results, thresholds)?;

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

/// What the benign fixture pass found.
#[derive(Debug, Default, PartialEq, Eq)]
struct FixtureTally {
    /// Targets graded benign throughout.
    passed: usize,
    total: usize,
    /// Targets in which something graded hostile: hard false positives.
    hostile_fps: usize,
    /// Targets whose worst grade was suspicious: tolerated.
    suspicious_fps: usize,
}

fn evaluate(
    results: Vec<(PathBuf, Result<ClassifiedReport>)>,
    thresholds: Thresholds,
) -> Result<FixtureTally> {
    let mut tally = FixtureTally::default();
    let mut analysis_failed = 0usize;

    for (path, result) in results {
        tally.total += 1;
        let result = match result {
            Ok(classified) => classified.result,
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
        let label = path.display().to_string();
        let root = Decision {
            class: result.classification,
            probability: result.probability,
            threshold: result.threshold,
            level: result.level,
        };
        let embedded = result.embedded_files.values().map(|e| {
            let decision = Decision {
                class: e.classification,
                probability: e.probability,
                threshold: e.threshold,
                level: e.level,
            };
            (
                format!("{label}!!{}", e.path),
                decision,
                e.top_findings.as_slice(),
            )
        });
        let mut worst = Classification::Benign;
        for (label, decision, findings) in
            std::iter::once((label.clone(), root, result.top_findings.as_slice())).chain(embedded)
        {
            if decision.class == Classification::Benign {
                continue;
            }
            worst = worst.max(decision.class);
            warn_nonbenign(&label, &decision, thresholds, findings);
        }

        match worst {
            Classification::Hostile => tally.hostile_fps += 1,
            Classification::Benign => tally.passed += 1,
            _ => tally.suspicious_fps += 1,
        }
    }

    if analysis_failed > 0 {
        anyhow::bail!(
            "{analysis_failed} validation check(s) failed during analysis ({}/{} targets benign, {} hostile FP, {} suspicious)",
            tally.passed,
            tally.total,
            tally.hostile_fps,
            tally.suspicious_fps,
        );
    }
    Ok(tally)
}

/// The WARN block for one non-benign grade on a benign fixture.
fn warn_nonbenign(
    label: &str,
    decision: &Decision,
    thresholds: Thresholds,
    findings: &[engine::TopFinding],
) {
    eprintln!(
        "WARN {label}: grade={} level={} probability={} decision_threshold={} margin_logit={:+.3} ensemble_thresholds suspicious={} hostile={}",
        decision.class,
        decision.level,
        decision.probability,
        decision.threshold,
        logit_margin(decision.probability, decision.threshold),
        thresholds.suspicious,
        thresholds.hostile,
    );
    for finding in findings {
        eprintln!("  l{} {}  {}", finding.crit, finding.id, finding.desc);
    }
}

/// Forces cleave's analysis cache on for the life of the guard, then restores
/// the setting that was in effect before.
struct CacheOverride {
    previous: bool,
}

impl CacheOverride {
    fn force_enabled() -> Self {
        let previous = cleave::cache::skip_cache();
        cleave::cache::set_skip_cache_override(Some(false));
        Self { previous }
    }
}

impl Drop for CacheOverride {
    fn drop(&mut self) {
        cleave::cache::set_skip_cache_override(Some(self.previous));
    }
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
