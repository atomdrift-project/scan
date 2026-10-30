use super::*;

#[test]
fn saved_interpret_keeps_bytes_and_rewrites_annotation_authority() {
    let mut report = AnalysisReport::new(cleave::TargetInfo::default());
    report.files.push(FileAnalysis {
        id: 0,
        path: "/does/not/exist/payload.js".into(),
        file_type: "javascript".into(),
        sha256: "a".repeat(64),
        size: 12,
        findings: vec![cleave::Finding::new(
            "micro-behaviors/process/interpreter::eval",
            cleave::FindingKind::Indicator,
            "Executes a variable",
            0.95,
        )],
        ..FileAnalysis::default()
    });
    report.files[0].findings[0].crit = Criticality::Notable;
    // The match the renderer annotates: `eval`, the first four bytes.
    report.files[0].findings[0].evidence = vec![cleave::Evidence {
        method: "ast".into(),
        value: "eval".into(),
        offsets: vec![0],
        ..Default::default()
    }];
    // A window as schema 8 wrote it (Z85 `b`), which readers still accept.
    report.files[0].context = serde_json::from_value(serde_json::json!([{
        "ln":0,"line":1,"col":1,
        "b":"wP?G#c$D+dvjsnB",
    }]))
    .unwrap();
    let loaded = cleave::report_input::decode(&serde_json::to_vec(&report).unwrap()).unwrap();
    let rendered = render_saved_interpret(&loaded).unwrap();
    assert!(rendered.starts_with("== PRIMARY payload.js ==\nprovenance="));
    assert!(rendered.contains("Possible process/interpreter"));
    assert!(!rendered.contains("// N "));
    assert!(rendered.contains("eval(data)"));
}

#[test]
fn saved_interpret_rejects_missing_dependency_presentation_state() {
    let mut report = AnalysisReport::new(cleave::TargetInfo::default());
    report.files.push(FileAnalysis {
        id: 0,
        path: "payload.js".into(),
        file_type: "javascript".into(),
        sha256: "a".repeat(64),
        ..FileAnalysis::default()
    });
    let mut loaded = cleave::report_input::decode(&serde_json::to_vec(&report).unwrap()).unwrap();
    loaded
        .fetched
        .push(serde_json::json!({"source_sha256":"a"}));
    assert!(render_saved_interpret(&loaded).is_err());
}

/// A replay is the live payload, byte for byte, so it goes through the live
/// sanitizer: whatever `sanitize_context` rewrites (ANSI, `!!`) must come out of
/// a replay rewritten the same way.
#[test]
fn saved_interpret_sanitizes_like_the_live_path() {
    let mut report = AnalysisReport::new(cleave::TargetInfo::default());
    report.files.push(FileAnalysis {
        id: 0,
        path: "payload.js".into(),
        file_type: "javascript".into(),
        sha256: "a".repeat(64),
        size: 12,
        findings: vec![cleave::Finding::new(
            "micro-behaviors/process/interpreter::eval",
            cleave::FindingKind::Indicator,
            "Executes a variable",
            0.95,
        )],
        ..FileAnalysis::default()
    });
    report.files[0].findings[0].crit = Criticality::Notable;
    report.files[0].findings[0].evidence = vec![cleave::Evidence {
        method: "ast".into(),
        value: "eval".into(),
        offsets: vec![0],
        ..Default::default()
    }];
    report.files[0].context = serde_json::from_value(serde_json::json!([{
        "ln":0,"line":1,"col":1,"s":"eval(!!flag)",
    }]))
    .unwrap();
    let loaded = cleave::report_input::decode(&serde_json::to_vec(&report).unwrap()).unwrap();
    let rendered = render_saved_interpret(&loaded).unwrap();
    assert!(!rendered.contains("!!"), "{rendered}");
}
