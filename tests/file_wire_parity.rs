//! Opt-in codec validation against every feature spec in a shipping bundle.
use anyhow::{Result, ensure};
use scan::features::{ExtractContext, FeatureSpec};
use std::path::{Path, PathBuf};

fn specs(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            specs(&entry.path(), out)?;
        } else if entry.file_name() == "feature_spec.json" {
            out.push(entry.path());
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires SCAN_MODELS_DIR and SCAN_FILE_WIRE_CASES with v8.json/trimmed.json cases"]
fn shipping_features_match_file_wire_roundtrip() -> Result<()> {
    let models = PathBuf::from(std::env::var("SCAN_MODELS_DIR")?);
    let cases = PathBuf::from(std::env::var("SCAN_FILE_WIRE_CASES")?);
    let mut paths = Vec::new();
    specs(&models, &mut paths)?;
    paths.sort();
    ensure!(!paths.is_empty(), "no shipping feature specs");
    let contexts: Vec<_> = paths
        .iter()
        .map(|path| Ok((path.clone(), ExtractContext::new(&FeatureSpec::load(path)?))))
        .collect::<Result<_>>()?;
    let mut count = 0;
    for entry in std::fs::read_dir(cases)? {
        let folder = entry?.path();
        if !folder.join("v8.json").exists() {
            continue;
        }
        let before: cleave::types::CompactReport =
            serde_json::from_slice(&std::fs::read(folder.join("v8.json"))?)?;
        let expected_vectors: Vec<_> = contexts
            .iter()
            .map(|(_, ctx)| ctx.extract(&before))
            .collect();
        let variants = ["trimmed.json", "shared.json"];
        for variant in variants {
            let bytes = std::fs::read(folder.join(variant))?;
            let loaded = cleave::report_input::decode(&bytes)?;
            let after = cleave::types::compact_from_files(&loaded.report.files);
            let direct = cleave::file_wire::features::FeatureReport::from_slice(&bytes)?;
            for ((path, ctx), expected) in contexts.iter().zip(&expected_vectors) {
                for (mode, actual) in [
                    ("compatibility", ctx.extract(&after)),
                    ("direct", ctx.extract_wire(&direct)?),
                ] {
                    ensure!(expected.len() == actual.len(), "feature count mismatch");
                    let different = expected
                        .iter()
                        .zip(&actual)
                        .position(|(a, b)| a.to_bits() != b.to_bits());
                    ensure!(
                        different.is_none(),
                        "feature mismatch: {} / {variant} / {mode} / {} at {different:?}",
                        folder.display(),
                        path.display()
                    );
                }
            }
        }
        count += 1;
    }
    ensure!(count > 0, "no codec cases");
    eprintln!(
        "wire feature parity (flat/shared and direct/compatibility): {count} reports × {} shipping route specs",
        contexts.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires shipping SCAN_MODELS_DIR and complete SCAN_FILE_WIRE_CASES"]
fn shipping_routed_scores_match_direct_wire() -> Result<()> {
    let model = scan::model::Model::load(
        &PathBuf::from(std::env::var("SCAN_MODELS_DIR")?),
        None,
        None,
    )?;
    let root = PathBuf::from(std::env::var("SCAN_FILE_WIRE_CASES")?);
    let mut count = 0;
    for entry in std::fs::read_dir(root)? {
        let folder = entry?.path();
        if !folder.join("v8.json").exists() {
            continue;
        }
        let before: cleave::types::CompactReport =
            serde_json::from_slice(&std::fs::read(folder.join("v8.json"))?)?;
        let (expected, expected_scores, expected_skips) = model.predict_report(&before)?;
        for variant in ["trimmed.json", "shared.json"] {
            let direct = cleave::file_wire::features::FeatureReport::from_slice(&std::fs::read(
                folder.join(variant),
            )?)?;
            let (actual, scores, skips) = model.predict_wire(&direct)?;
            ensure!(
                actual.class == expected.class
                    && actual.level == expected.level
                    && actual.probability.to_bits() == expected.probability.to_bits()
                    && actual.threshold.to_bits() == expected.threshold.to_bits(),
                "decision mismatch {} / {variant}",
                folder.display()
            );
            ensure!(
                serde_json::to_value(&scores)? == serde_json::to_value(&expected_scores)?,
                "route score mismatch {} / {variant}",
                folder.display()
            );
            ensure!(
                serde_json::to_value(&skips)? == serde_json::to_value(&expected_skips)?,
                "skipped route mismatch"
            );
        }
        count += 1;
        eprintln!("{} routed decisions equal", folder.display());
    }
    ensure!(count > 0, "no routed score cases");
    eprintln!("direct wire score parity: {count} reports, both formats");
    Ok(())
}
