use super::{FileSummary, RawNeeds};
use cleave::types::{CompactReport, compact_from_files};

/// Build a report with the shapes that actually vary: findings at several
/// criticalities and confidences, imports, metrics, a nested member.
fn sample_report() -> CompactReport {
    use cleave::types::{Criticality, FindingKind};
    let mut files = Vec::new();
    for (id, path, crit, conf) in [
        (0u32, "pkg.zip", Criticality::Notable, 0.9_f32),
        (1, "pkg.zip!!lib/net.py", Criticality::Suspicious, 0.7),
        (2, "pkg.zip!!lib/quiet.py", Criticality::Baseline, 0.5),
    ] {
        let mut fa = cleave::FileAnalysis {
            id,
            path: path.to_string(),
            file_type: "python".to_string(),
            sha256: format!("{id:064}"),
            size: 2048 * u64::from(id + 1),
            ..Default::default()
        };
        let mut f = cleave::types::Finding::new(
            "objectives/execution/shell::sh".to_string(),
            FindingKind::Capability,
            "spawns a shell".to_string(),
            conf,
        );
        f.crit = crit;
        f.attack = Some("T1059".into());
        fa.findings.push(f);
        if id > 0 {
            fa.depth = 1;
            fa.parent_id = Some(0);
        }
        files.push(fa);
    }
    compact_from_files(&files, cleave::types::Facts::Full)
}

/// The typed reader's per-file output, pinned. These values feed the model
/// directly, so drift here moves ML output silently rather than failing —
/// which is why they are asserted rather than reviewed.
#[test]
fn typed_reader_summarizes_every_file() {
    let report = sample_report();
    for needs in [RawNeeds::default(), RawNeeds::all()] {
        let summaries: Vec<FileSummary> = report
            .files
            .iter()
            .map(|f| FileSummary::from_compact(f, needs))
            .collect();
        assert_eq!(summaries.len(), report.files.len());
        for (summary, file) in summaries.iter().zip(&report.files) {
            assert_eq!(
                summary.raw_findings.len(),
                file.findings.len(),
                "every finding on {} must reach the summary",
                file.path
            );
            for (raw, finding) in summary.raw_findings.iter().zip(&file.findings) {
                assert_eq!(raw.id, finding.id);
                assert_eq!(raw.crit, u32::from(finding.criticality));
                // The encoder omits `conf` at 0.5/0.0 and readers restore a
                // missing `conf` as 1.0 — preserved deliberately, since it
                // is what every deployed model was calibrated against.
                assert_eq!(raw.conf, f64::from(finding.confidence));
            }
        }
    }
}

/// Imports live in the hand-encoded `facts` block, so the tokens derived
/// from them are the likeliest thing to drift.
#[test]
fn typed_reader_derives_import_tokens() {
    let mut fa = cleave::FileAnalysis {
        id: 0,
        path: "app.exe".to_string(),
        file_type: "pe".to_string(),
        sha256: "d".repeat(64),
        size: 65536,
        ..Default::default()
    };
    fa.imports = vec![cleave::types::Import {
        symbol: "CreateProcessW".to_string(),
        library: Some("kernel32.dll".to_string()),
        ..Default::default()
    }];
    let report = compact_from_files(&[fa], cleave::types::Facts::Full);
    let summary = FileSummary::from_compact(&report.files[0], RawNeeds::all());
    // Both forms: LOGIC_GAPS matches the bare name, the vocab the qualified.
    assert!(summary.imports.contains("CreateProcessW"));
    assert!(summary.imports.contains("kernel32.dll!CreateProcessW"));
}
