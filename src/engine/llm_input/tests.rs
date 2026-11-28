use std::fmt::Write as _;

use super::*;
use crate::engine::{EmbeddedFile, FloorDecision};
use crate::interpret::{FindingSeverity, InterpretConfig, LevelContext, admit};
use crate::model::{RouteScore, SkippedRoute};
use crate::upload::encode_result_body;

fn sha(c: char) -> String {
    c.to_string().repeat(64)
}

fn report(files: serde_json::Value) -> cleave::types::CompactReport {
    let mut report = serde_json::json!({"rev": "r1"});
    report["files"] = files;
    serde_json::from_value(report).expect("report")
}

fn decision(class: Classification, probability: f32, level: Level) -> Decision {
    Decision {
        class,
        probability,
        threshold: 0.65,
        level,
    }
}

fn routes(class: Classification, probability: f32) -> (Vec<RouteScore>, Vec<SkippedRoute>) {
    (
        vec![RouteScore {
            model: "az/js".to_string(),
            probability,
            raw: probability,
            classification: class,
        }],
        vec![SkippedRoute {
            model: "az/native".to_string(),
            reason: SkippedRoute::UNCALIBRATED,
        }],
    )
}

/// Member `id`'s evaluation: hostile every seventh, suspicious every third,
/// benign otherwise, and a trait-floor firing on each hostile one.
fn member(id: u32) -> EmbeddedFile {
    let (class, probability, level) = match id {
        _ if id.is_multiple_of(7) => (Classification::Hostile, 0.98, Level::At(9)),
        _ if id.is_multiple_of(3) => (Classification::Suspicious, 0.71, Level::At(60)),
        _ => (Classification::Benign, 0.04, Level::Clean),
    };
    let (model_scores, skipped_models) = routes(class, probability);
    let model = decision(class, probability, level);
    EmbeddedFile {
        id: u64::from(id),
        sha256: String::new(),
        path: String::new(),
        file_type: "javascript".to_string(),
        classification: class,
        probability,
        threshold: 0.65,
        level,
        model_scores,
        skipped_models,
        formula: String::new(),
        top_findings: Vec::new(),
        floor: (class == Classification::Hostile).then_some(FloorDecision {
            class,
            confidence: 0.93,
            level: Level::At(12),
            arm: crate::engine::FloorArm::Crit5,
            hostile: 2,
            suspicious: 1,
            families: 3,
            model,
        }),
    }
}

/// A hostile npm package of `members` scripts, each carrying `traits`
/// findings, with references to a dependency and a URL.
fn package(members: u32, traits: u32) -> cleave::types::CompactReport {
    let families = [
        "anti-static/obfuscation/strings/hex",
        "execution/process/spawn/shell",
        "net/http/request/fetch",
        "fs/file/read/home",
        "data/encode/base64/decode",
        "objectives/exfil/credentials/npmrc",
        "discovery/system/hostname",
        "persistence/install-script/postinstall",
    ];
    let mut files = vec![serde_json::json!({
        "id": 0, "path": "/tmp/pkg.tgz", "type": "npm", "sha": sha('a'), "size": 48_000_000,
        "refs": [
            {"to": "pkg:npm/left-pad@1.3.0", "kind": "dependency"},
            {"to": "https://203.0.113.7/stage2.sh", "kind": "url_fetch"}
        ]
    })];
    for id in 1..=members {
        let findings: Vec<_> = (0..traits)
            .map(|n| {
                let family = families[((id + n) % 8) as usize];
                serde_json::json!({
                    "id": format!("{family}::{n}"),
                    "crit": 2 + (id + n) % 4,
                    "conf": 0.9,
                    "desc": format!("{family} pattern {n} in module {id}"),
                    "ev": [[u64::from(n) * 120, 48]],
                })
            })
            .collect();
        files.push(serde_json::json!({
            "id": id, "pid": 0, "depth": 1,
            "path": format!("/tmp/pkg.tgz!!package/lib/module{id:04}/index.js"),
            "type": "javascript",
            "sha": format!("{:064x}", u64::from(id) * 0x9e37_79b9),
            "size": 20_000 + id,
            "traits": findings,
        }));
    }
    report(serde_json::Value::Array(files))
}

/// A hostile verdict on `report` the gate admitted, its upload document
/// built as the pipeline builds one, with its LLM pass pending.
fn pending_result(report: &cleave::types::CompactReport) -> ScanResult {
    let members: MemberEvals = report
        .files
        .iter()
        .skip(1)
        .map(|f| (u64::from(f.id), member(f.id)))
        .collect();
    let kept = crate::engine::retention::kept_members(report, |id| {
        members.get(&u64::from(id)).map(|m| m.classification)
    });
    let verdict = decision(Classification::Hostile, 0.97, Level::At(12));
    let (root_mods, root_skip) = routes(Classification::Hostile, 0.97);
    let mut doc = UploadDoc::new(&crate::engine::v8::Inputs {
        report,
        root: crate::engine::v8::OwnReading {
            verdict,
            model: verdict,
            floor: None,
        },
        root_mods: &root_mods,
        root_skip: &root_skip,
        members: &members,
        kept: &kept,
        every: false,
        listed: &[],
        records: &[],
        root_purl: None,
        level: Some(25),
    })
    .expect("a report with files yields a document");
    doc.attach_reports(report);

    let mut result = crate::engine::envelope::tests::base_result();
    result.sha256 = sha('a');
    result.classification = Classification::Hostile;
    result.probability = 0.97;
    result.level = Level::At(12);
    result.path = "pkg/evil-1.0.0.tgz".to_string();
    result.file_type = "npm".to_string();
    result.size_bytes = 48_000_000;
    result.upload = Some(doc);
    let levels = LevelContext {
        fired: Level::At(12),
        active: Some(25),
        grid_max: 100,
    };
    let findings = FindingSeverity {
        hostile: true,
        suspicious: true,
        ..FindingSeverity::default()
    };
    let admitted = admit(
        &InterpretConfig::default(),
        Classification::Hostile,
        0.97,
        levels,
        findings,
        "test",
    )
    .expect("a hostile verdict is admitted");
    result.pending_llm = Some(PendingLlm {
        ctx: "FILE pkg/evil.js\nH exfiltrates ~/.npmrc to https://203.0.113.7/x\n".to_string(),
        admitted,
    });
    result
}

fn small_report() -> cleave::types::CompactReport {
    package(8, 3)
}

fn encode(result: &ScanResult) -> Vec<u8> {
    let pending = result.pending_llm.as_ref().expect("pending");
    LlmInput::encode(result, pending, 1234).expect("encodes")
}

/// The version 8 body an amendment of `result` posts, decompressed.
fn body(result: &ScanResult) -> Vec<u8> {
    let (body, encoding) = encode_result_body(result, &result.sha256, "w", Some(7)).expect("body");
    assert_eq!(encoding, Some("zstd"));
    zstd::decode_all(body.as_slice()).expect("zstd")
}

#[test]
fn round_trip_keeps_what_the_pass_reads_and_the_amendment_posts() {
    let original = pending_result(&small_report());
    let blob = encode(&original);
    assert!(
        blob.starts_with(b"ADLI\x01"),
        "magic and format lead the blob"
    );

    let input = LlmInput::decode(&blob).expect("decodes");
    assert_eq!(input.sha256(), sha('a'));
    assert_eq!(input.analysis_ms(), 1234);
    assert_eq!(input.tier(), LlmAdmission::Required);
    let resumed = input.into_result();

    assert_eq!(resumed.sha256, original.sha256);
    assert_eq!(resumed.path, original.path);
    assert_eq!(resumed.file_type, original.file_type);
    assert_eq!(resumed.size_bytes, original.size_bytes);
    assert_eq!(resumed.version, original.version);
    assert_eq!(resumed.analyzed_at, original.analyzed_at);
    assert_eq!(resumed.classification, original.classification);
    assert_eq!(
        resumed.probability.to_bits(),
        original.probability.to_bits()
    );
    assert_eq!(resumed.threshold.to_bits(), original.threshold.to_bits());
    assert_eq!(resumed.level, original.level);
    assert!(resumed.interpretation.is_none());
    let (was, now) = (
        original.pending_llm.as_ref().expect("pending"),
        resumed.pending_llm.as_ref().expect("still pending"),
    );
    assert_eq!(now.ctx, was.ctx);
    assert_eq!(now.admitted, was.admitted);
    // The amendment re-posts the same body the original would have.
    assert_eq!(body(&resumed), body(&original));
}

#[test]
fn round_trip_keeps_floor_routes_jurors_and_scanners() {
    let mut original = pending_result(&small_report());
    let doc = original.upload.as_mut().expect("doc");
    doc.set_scanner(
        crate::engine::Scanner::Clamav,
        serde_json::json!({"outcome": "ok", "class": "hostile", "sig": "Js.Stealer"}),
    )
    .expect("scanner");
    doc.set_juror(
        crate::engine::Juror::DriftHeuristic,
        serde_json::json!({"outcome": "ok", "class": "suspicious"}),
    )
    .expect("juror");
    let resumed = LlmInput::decode(&encode(&original))
        .expect("decodes")
        .into_result();
    let posted = body(&resumed);
    assert_eq!(posted, body(&original));
    let text = String::from_utf8_lossy(&posted);
    for carried in [
        "\"floor\"",
        "url_fetch",
        "clamav",
        "drift_heuristic",
        "uncalibrated",
        "az/js",
    ] {
        assert!(text.contains(carried), "{carried} missing from {text}");
    }
}

#[test]
fn a_foreign_or_damaged_blob_is_absent() {
    let blob = encode(&pending_result(&small_report()));
    let mut newer = blob.clone();
    newer[4] = LLM_INPUT_FORMAT + 1;
    assert!(LlmInput::decode(&newer).is_none(), "another format");
    let mut older = blob.clone();
    older[4] = 0;
    assert!(LlmInput::decode(&older).is_none(), "an older format");
    assert!(
        LlmInput::decode(&blob[..blob.len() / 2]).is_none(),
        "truncated"
    );
    assert!(LlmInput::decode(b"ADLI").is_none(), "no format byte");
    assert!(LlmInput::decode(b"").is_none(), "empty");
    let mut magic = blob.clone();
    magic[0] = b'X';
    assert!(LlmInput::decode(&magic).is_none(), "not an input");
    // A well-formed frame of something else.
    let mut other = b"ADLI\x01".to_vec();
    other.extend(zstd::encode_all(&b"{\"sha256\": 1}"[..], 3).expect("zstd"));
    assert!(LlmInput::decode(&other).is_none(), "malformed JSON");
    let mut flipped = blob;
    let last = flipped.len() - 8;
    flipped[last] ^= 0xff;
    assert!(LlmInput::decode(&flipped).is_none(), "corrupt frame");
}

#[test]
fn a_result_without_an_upload_body_is_not_encoded() {
    let mut result = pending_result(&small_report());
    result.upload = None;
    let pending = result.pending_llm.clone().expect("pending");
    assert!(matches!(
        LlmInput::encode(&result, &pending, 0),
        Err(LlmInputError::NotForUpload)
    ));
}

/// A large npm package: 1,500 scripts with a dozen findings each, the most
/// retention keeps judged and reported, and a render at the prompt cap. The
/// encoded input stays far under the upload cap.
#[test]
fn a_large_result_encodes_well_under_the_cap() {
    let mut result = pending_result(&package(1500, 12));
    let mut ctx = String::new();
    for line in 0..4000 {
        let _ = writeln!(
            ctx,
            "N net/http/request/fetch https://cdn{line}.example.net/{line}.js"
        );
    }
    if let Some(pending) = &mut result.pending_llm {
        pending.ctx = ctx;
    }
    let blob = encode(&result);
    let json = zstd::decode_all(&blob[5..]).expect("zstd");
    let posted = body(&result).len();
    eprintln!(
        "large LLM input: {} bytes encoded, {} bytes of JSON; the result body it re-posts is {posted} bytes",
        blob.len(),
        json.len(),
    );
    assert!(
        blob.len() < LLM_INPUT_MAX_BYTES / 4,
        "{} bytes is too close to the cap",
        blob.len()
    );
    let resumed = LlmInput::decode(&blob).expect("decodes").into_result();
    assert_eq!(body(&resumed), body(&result));
}
