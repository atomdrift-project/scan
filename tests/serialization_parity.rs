//! Byte-for-byte parity between the two JSON envelope serialization paths.
//!
//! `scan` serializes the analysis envelope (`{"ml": {...}, "llm": ..., "raw":
//! {...}}`) through two parallel builders on [`ScanResult`]:
//!
//!   - **CLI `--format=json`** streams the borrowed [`ScanResult::envelope_ref`]
//!     (`ScanResultEnvelopeRef` / `MlSectionRef`) straight to stdout.
//!   - **Server `/analyze` and the hopper worker upload** serialize the owned
//!     [`ScanResult::into_envelope`] (`ScanResultEnvelope` / `MlSection`); the
//!     worker flattens that owned envelope into its `/api/result` body via
//!     `#[serde(flatten)]`, so the bytes hopper stores are exactly these
//!     envelope bytes plus the transport wrapper (`sha256`/`worker`/…).
//!
//! These are hand-written mirror types with independently-declared serde
//! attributes (e.g. the borrowed `mods`/`skip` fields skip on
//! `route_scores_empty`/`skipped_routes_empty` while the owned ones skip on
//! `Vec::is_empty`). If they ever drift, the JSON a human sees via the CLI would
//! differ from what every server and worker sends to hopper for the same file.
//! This test pins them together: the same `ScanResult`, serialized every way,
//! must produce identical bytes.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use scan::engine::{EmbeddedFile, FindingCounts, ScanResult};
use scan::interpret::Interpretation;
use scan::model::{Classification, RouteScore, SkippedRoute};

/// Serialize a `ScanResult` through all three envelope builders and assert the
/// resulting JSON is byte-for-byte identical. `to_writer`/`to_vec` share
/// serde_json's compact formatter (as does axum's `Json` responder), so equal
/// bytes here means the CLI, server, and worker emit the same envelope.
fn assert_envelopes_identical(result: ScanResult, case: &str) {
    // CLI path: exactly what `emit_result` writes (`serde_json::to_writer` over
    // the borrowed view). Borrow first so the owned conversions can consume.
    let cli = serde_json::to_vec(&result.envelope_ref()).unwrap();
    // Clone-based owned builder (used where the caller keeps the result).
    let owned_clone = serde_json::to_vec(&result.to_envelope()).unwrap();
    // Move-based owned builder: the server `/analyze` response and the hopper
    // worker upload both serialize this exact value. Consumes `result`, so last.
    let owned_move = serde_json::to_vec(&result.into_envelope()).unwrap();

    let cli = String::from_utf8(cli).unwrap();
    let owned_clone = String::from_utf8(owned_clone).unwrap();
    let owned_move = String::from_utf8(owned_move).unwrap();

    assert_eq!(
        cli, owned_clone,
        "[{case}] CLI envelope_ref() diverged from to_envelope() (clone path)"
    );
    assert_eq!(
        cli, owned_move,
        "[{case}] CLI envelope_ref() diverged from into_envelope() (server/hopper path)"
    );
}

/// A fully-populated result: every optional field present and every
/// `skip_serializing_if` collection non-empty, so the *present* branch of each
/// envelope field is exercised in both serializers.
#[test]
fn full_result_serializes_identically() {
    let result = ScanResult {
        v: "8",
        classification: Classification::Hostile,
        probability: 0.873_21,
        threshold: 0.5,
        level: Some(50),
        version: "model-abc123".to_string(),
        analyzed_at: "2026-06-17T15:01:22Z".to_string(),
        cleave: Some(serde_json::json!({
            "v": 8,
            "files": [
                { "id": 0, "dp": 0, "type": "whl", "path": "pkg.whl", "risk": 3 },
                { "id": 1, "dp": 1, "type": "python", "path": "pkg.whl!!a/__init__.py", "risk": 1 },
                { "id": 2, "dp": 1, "type": "python", "path": "pkg.whl!!a/_utilities.py", "risk": 3 }
            ]
        })),
        pids: Some(vec![123, 456]),
        deleted: Some(true),
        path: "pkg.whl".to_string(),
        finding_counts: FindingCounts::default(),
        formula: "H\u{2082}(Db\u{2082}Os)".to_string(),
        reasons: vec![],
        top_findings: vec![],
        file_type: "whl".to_string(),
        size_bytes: 13_189,
        sha256: "c7159256a21402fd4c650fbf906ad13c56f1f3ae818df3ba7385ae6f51db2585".to_string(),
        embedded_files: scan::engine::MemberEvals::from([(
            1,
            EmbeddedFile {
                id: 1,
                sha256: "a".repeat(64),
                path: "a/_utilities.py".to_string(),
                file_type: "python".to_string(),
                classification: Classification::Suspicious,
                probability: 0.42,
                threshold: 0.5,
                level: Some(100),
                model_scores: vec![],
                skipped_models: vec![],
                formula: String::new(),
                top_findings: vec![],
            },
        )]),
        model_scores: vec![RouteScore {
            model: "az".to_string(),
            probability: 0.873_21,
            raw: 0.91,
            classification: Classification::Hostile,
        }],
        skipped_models: vec![SkippedRoute {
            model: "az/elf".to_string(),
            reason: "type not applicable",
        }],
        rendered_context: "1  import base64".to_string(),
        interpretation: Some(Interpretation {
            grade: None,
            outcome: Classification::Hostile,
            blended: 0.873_21,
            interpretation: "imports base64 and writes to sys.modules".to_string(),
            model: "claude-test".to_string(),
            error: None,
            analyzer_directed: true,
        }),
        dependency_results: vec![],
        bloom_mark: None,
    };
    assert_envelopes_identical(result, "full");
}

/// A minimal result: every optional field absent and every
/// `skip_serializing_if` collection empty, so the *skipped* branch of each
/// envelope field is exercised. This is what catches a divergence in the skip
/// predicates themselves (`route_scores_empty` vs `Vec::is_empty`, etc.).
#[test]
fn minimal_result_serializes_identically() {
    let result = ScanResult {
        v: "8",
        classification: Classification::Benign,
        probability: 0.01,
        threshold: 0.5,
        level: Some(-1),
        version: "model-abc123".to_string(),
        analyzed_at: "2026-06-17T15:01:22Z".to_string(),
        cleave: None,
        pids: None,
        deleted: None,
        path: "empty.bin".to_string(),
        finding_counts: FindingCounts::default(),
        formula: String::new(),
        reasons: vec![],
        top_findings: vec![],
        file_type: "data".to_string(),
        size_bytes: 0,
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
        embedded_files: scan::engine::MemberEvals::new(),
        model_scores: vec![],
        skipped_models: vec![],
        rendered_context: String::new(),
        interpretation: None,
        dependency_results: vec![],
        bloom_mark: None,
    };
    assert_envelopes_identical(result, "minimal");
}
