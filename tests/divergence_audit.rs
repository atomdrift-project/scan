//! Diagnostic — count parity divergences per feature family, per route.
//! Doesn't fail; prints a table. Run with `--nocapture`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use litmus::features::{ExtractContext, FeatureSpec};
use std::collections::BTreeMap;

#[derive(serde::Deserialize)]
struct Fixture {
    n_features: usize,
    #[serde(default)]
    feature_names: Vec<String>,
    samples: Vec<Sample>,
}

#[derive(serde::Deserialize)]
struct Sample {
    report: serde_json::Value,
    expected_features: Vec<f32>,
}

fn family_of(name: &str) -> &'static str {
    for fam in [
        "bigrams:",
        "trigram:",
        "tierbi:",
        "tiertri:",
        "kv:",
        "symbol_bi:",
        "symbol_tri:",
        "symbol:",
        "textenc:",
        "metrics:derived_",
        "metrics:",
        "agg:",
        "struct:",
        "elements:",
        "rare:",
        "crit:",
        "critbi:",
        "crittri:",
        "atkbi:",
        "atktri:",
        "mbcbi:",
        "mbctri:",
        "present:",
        "maxcrit:",
        "format:",
        "inter:",
        "filetype:",
        "skeleton:",
        "ghost:",
        "score:",
        "formula:",
        "missing:",
        "gap:",
        "intent_gap:",
        "unsigned_bigram:",
        "ext:",
    ] {
        if name.starts_with(fam) {
            return fam.trim_end_matches(':').trim_end_matches('_');
        }
    }
    "(other)"
}

#[test]
fn audit_per_route_divergences() {
    let dir = std::env::var("LITMUS_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from("/home/t/collimator/out/models/azoth")
        });
    let mut found = 0usize;
    for parent in ["filetypes", "filegroups"] {
        let parent_dir = dir.join(parent);
        let Ok(entries) = std::fs::read_dir(&parent_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let route_dir = entry.path();
            let spec_path = route_dir.join("feature_spec.json");
            let fixture_path = route_dir.join("extraction_fixture.json");
            if !spec_path.exists() || !fixture_path.exists() {
                continue;
            }
            found += 1;
            let spec = FeatureSpec::load(&spec_path).unwrap();
            let fixture_data = std::fs::read_to_string(&fixture_path).unwrap();
            let fixture: Fixture = serde_json::from_str(&fixture_data).unwrap();
            assert_eq!(spec.total_features(), fixture.n_features);
            let ctx = ExtractContext::new(&spec);

            let mut by_family_div: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
            let mut by_family_nonzero: BTreeMap<&'static str, usize> = BTreeMap::new();
            let mut detailed_metrics: Vec<(String, f32, f32)> = Vec::new();
            for sample in &fixture.samples {
                let rust_vec = ctx.extract(&sample.report);
                for (j, (r, p)) in rust_vec
                    .iter()
                    .zip(sample.expected_features.iter())
                    .enumerate()
                {
                    let name = spec.feature_names()[j].as_str();
                    let fam = family_of(name);
                    if p.abs() > 1e-5 || r.abs() > 1e-5 {
                        *by_family_nonzero.entry(fam).or_default() += 1;
                    }
                    let diff = (r - p).abs();
                    if diff > 1e-5 {
                        by_family_div.entry(fam).or_default().0 += 1;
                        if fam == "metrics" && detailed_metrics.len() < 10 {
                            detailed_metrics.push((name.to_string(), *r, *p));
                        }
                    }
                }
            }
            for fname in spec.feature_names() {
                let fam = family_of(fname);
                by_family_div.entry(fam).or_default().1 += 1;
            }
            let route_label = route_dir
                .strip_prefix(&dir)
                .unwrap_or(&route_dir)
                .display()
                .to_string();
            eprintln!("=== {route_label} ({} samples) ===", fixture.samples.len());
            eprintln!("  {:<16} {:>9}   {:>9}   {}", "family", "diverge", "non-zero", "slots");
            for (fam, (div, total)) in &by_family_div {
                let nz = by_family_nonzero.get(fam).copied().unwrap_or(0);
                if *div > 0 || nz > 0 {
                    eprintln!("  {fam:<16} {div:>9}   {nz:>9}   {total}");
                }
            }
            if !detailed_metrics.is_empty() {
                eprintln!("  metrics divergences (first 10):");
                for (name, r, p) in &detailed_metrics {
                    eprintln!("    {name} rust={r:.6} python={p:.6}");
                }
            }
            // In-scope spot check: enumerate diverging slots whose names fall
            // inside the v17 family port. These should always be zero — if
            // not, my port has a bug.
            let mut in_scope_div: Vec<(String, f32, f32)> = Vec::new();
            for sample in &fixture.samples {
                let rust_vec = ctx.extract(&sample.report);
                for (j, (r, p)) in rust_vec
                    .iter()
                    .zip(sample.expected_features.iter())
                    .enumerate()
                {
                    let diff = (r - p).abs();
                    if diff > 1e-5 {
                        let name = spec.feature_names()[j].as_str();
                        let in_scope = name.starts_with("kv:")
                            || name.starts_with("symbol:")
                            || name.starts_with("symbol_bi:")
                            || name.starts_with("symbol_tri:")
                            || name.starts_with("textenc:")
                            || name.starts_with("metrics:derived_")
                            || name == "struct:silent_packer_signal"
                            || name == "agg:suspicious_bigram_count"
                            || name == "agg:suspicious_trigram_count";
                        if in_scope && in_scope_div.len() < 20 {
                            in_scope_div.push((name.to_string(), *r, *p));
                        }
                    }
                }
            }
            if !in_scope_div.is_empty() {
                eprintln!("  IN-SCOPE divergences (first 20):");
                for (name, r, p) in &in_scope_div {
                    eprintln!("    {name} rust={r:.6} python={p:.6}");
                }
            }
        }
    }
    eprintln!("\n(checked {found} routes)");
}
