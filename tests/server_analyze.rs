//! Integration tests for the analyze routes against a real model bundle.
//!
//! Each needs `SCAN_MODELS_DIR`, so each is ignored by default; run them with
//! `cargo test --test server_analyze -- --ignored`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use anyhow::{Context, Result};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::{init_tracing, loopback, ready_app, ready_config, status_and_json};
use scan::server::ServerConfig;
use tower::ServiceExt;

fn multipart_body(file_bytes: &[u8], filename: &str) -> (String, Vec<u8>) {
    let boundary = "----litmus-test-boundary";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(file_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

fn test_archive() -> Result<Vec<u8>> {
    let testdata = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/encrypted.zip");
    std::fs::read(&testdata).context("testdata/encrypted.zip not found — copy a test sample there")
}

fn analyze_request(file_bytes: &[u8]) -> Result<Request<Body>> {
    let (content_type, body) = multipart_body(file_bytes, "encrypted.zip");
    Ok(loopback(
        Request::builder()
            .method("POST")
            .uri("/analyze")
            .header("content-type", content_type)
            .body(Body::from(body))?,
    ))
}

/// Submit an encrypted zip via /analyze and verify JSON response structure.
#[tokio::test]
#[ignore = "needs a model bundle: set SCAN_MODELS_DIR and run with --ignored"]
async fn analyze_encrypted_zip_returns_json() -> Result<()> {
    init_tracing();
    let file_bytes = test_archive()?;
    let app = ready_app(&ready_config()?).await?;

    let response = app.oneshot(analyze_request(&file_bytes)?).await?;
    let (status, json) = status_and_json(response).await?;
    assert_eq!(
        status,
        StatusCode::OK,
        "expected 200 but got {status}: {json}"
    );

    // Every response must have the v7 envelope fields, regardless of classification.
    let ml = json["ml"].as_object().context("missing ml section")?;
    assert_eq!(ml["v"].as_str(), Some("7"), "envelope version must be v7");
    assert!(ml["prob"].is_number(), "missing probability");
    assert!(ml.contains_key("lvl"), "missing lvl field");
    assert!(ml["version"].is_string(), "missing model version");
    assert!(ml["files"].is_array(), "missing per-file ML results");
    assert!(json["raw"].is_object(), "missing raw cleave report");
    assert!(json["raw"]["files"].is_array(), "missing cleave files");

    // v7 drops legacy verdict fields from the envelope; consumers derive the
    // verdict from `lvl` instead (-1 = benign; anything else = hostile).
    for dropped in ["class", "threshold", "level", "l", "fs", "models"] {
        assert!(
            !ml.contains_key(dropped),
            "v7 envelope must not emit `{dropped}`"
        );
    }

    if let Some(l) = ml["lvl"].as_i64() {
        assert!(l == -1 || (0..=100).contains(&l), "unexpected l value: {l}");
    } // null is also valid (manual thresholds on a hostile verdict)
    Ok(())
}

/// Concurrent identical uploads share one analysis.
///
/// The coordination itself is unit-tested in `server::flight`; what this covers
/// is the part those tests cannot reach — that a follower, which never ran an
/// analysis of its own, renders the leader's real report correctly and gets
/// byte-for-byte the same answer.
#[tokio::test]
#[ignore = "needs a model bundle: set SCAN_MODELS_DIR and run with --ignored"]
async fn concurrent_identical_uploads_share_one_analysis() -> Result<()> {
    init_tracing();
    let file_bytes = test_archive()?;
    // One slot: without sharing, the duplicates would 429 instead of riding
    // along with the analysis already running.
    let config = ServerConfig {
        workers: 1,
        ..ready_config()?
    };
    let app = ready_app(&config).await?;

    let mut requests = Vec::new();
    for _ in 0..4 {
        let request = analyze_request(&file_bytes)?;
        let app = app.clone();
        requests.push(tokio::spawn(async move {
            let response = app.oneshot(request).await.expect("analyze request failed");
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 10 * 1024 * 1024)
                .await
                .expect("read response body");
            (status, bytes)
        }));
    }

    let mut answers = Vec::new();
    for request in requests {
        answers.push(request.await.context("joining request task")?);
    }

    let (first_status, first_body) = &answers[0];
    assert_eq!(
        *first_status,
        StatusCode::OK,
        "expected 200 but got {first_status}: {}",
        String::from_utf8_lossy(first_body),
    );
    for (status, body) in &answers[1..] {
        assert_eq!(status, first_status, "every sharer gets the same status");
        assert_eq!(body, first_body, "every sharer gets the same report");
    }
    Ok(())
}

/// `/analyze-path` reports its completions to `/_/stats` like every other
/// analyze route. It used to skip the shared completion step, so
/// `jobs_completed` never moved and `jobs_unfinished` climbed with every
/// request it served — the shape of a sick server, on a healthy one.
#[tokio::test]
#[ignore = "needs a model bundle: set SCAN_MODELS_DIR and run with --ignored"]
async fn analyze_path_counts_as_completed() -> Result<()> {
    init_tracing();
    let dir = tempfile::tempdir()?;
    let sample = dir.path().join("encrypted.zip");
    std::fs::write(&sample, test_archive()?)?;
    let config = ServerConfig {
        allowed_dirs: vec![dir.path().canonicalize()?],
        ..ready_config()?
    };
    let app = ready_app(&config).await?;

    let body = serde_json::json!({ "path": sample }).to_string();
    let response = app
        .clone()
        .oneshot(loopback(
            Request::builder()
                .method("POST")
                .uri("/analyze-path")
                .header("content-type", "application/json")
                .body(Body::from(body))?,
        ))
        .await?;
    let (status, json) = status_and_json(response).await?;
    assert_eq!(status, StatusCode::OK, "{json}");

    let response = app
        .oneshot(loopback(
            Request::builder().uri("/_/stats").body(Body::empty())?,
        ))
        .await?;
    let (_, stats) = status_and_json(response).await?;
    assert_eq!(stats["jobs_started"], 1, "{stats}");
    assert_eq!(stats["jobs_completed"], 1, "{stats}");
    assert_eq!(stats["jobs_unfinished"], 0, "{stats}");
    Ok(())
}
