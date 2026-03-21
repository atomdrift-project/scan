//! Integration tests for the /analyze endpoint.

use anyhow::{Context, Result};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use litmus::server::{build_app, ServerConfig};
use tower::ServiceExt;

fn model_dir() -> Result<std::path::PathBuf> {
    if let Ok(d) = std::env::var("LITMUS_MODELS_DIR") {
        return Ok(std::path::PathBuf::from(d));
    }
    litmus::models_repo::model_dir().context("failed to resolve model directory")
}

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

/// Submit an encrypted zip via /analyze and verify JSON response structure.
#[tokio::test]
async fn analyze_encrypted_zip_returns_json() -> Result<()> {
    let testdata = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/encrypted.zip");
    assert!(
        testdata.exists(),
        "testdata/encrypted.zip not found — copy a test sample there"
    );
    let file_bytes = std::fs::read(&testdata).context("failed to read test archive")?;

    // Debug builds are ~10x slower than release; use a generous timeout so
    // YARA warmup + encrypted-zip analysis can finish without a 504.
    let timeout_secs = if cfg!(debug_assertions) { 600 } else { 120 };

    let config = ServerConfig::new(
        std::net::SocketAddr::from(([127, 0, 0, 1], 8081)),
        timeout_secs,
        100 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
        model_dir()?,
        litmus::model::Thresholds::default(),
        4000,
    )?;
    let app = build_app(&config).await.context("failed to build app")?;

    // Wait for background resource loading to complete before sending requests.
    eprintln!("waiting for server readiness...");
    for _ in 0..100 {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/_/health")
                    .body(Body::empty())
                    .context("failed to build health request")?,
            )
            .await
            .context("health request failed")?;
        if resp.status() == StatusCode::OK {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    let (content_type, body) = multipart_body(&file_bytes, "encrypted.zip");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header("content-type", content_type)
                .body(Body::from(body))
                .context("failed to build analyze request")?,
        )
        .await
        .context("analyze request failed")?;

    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(response.into_body(), 10 * 1024 * 1024)
        .await
        .context("failed to read response body")?;
    let json: serde_json::Value =
        serde_json::from_slice(&body_bytes).context("response must be valid JSON")?;

    // Every response must have these fields, regardless of classification.
    assert!(json["classification"].is_string(), "missing classification");
    assert!(json["probability"].is_number(), "missing probability");
    assert!(json["thresholds"].is_object(), "missing thresholds");
    assert!(json["formula"].is_string(), "missing formula");
    assert!(json["model"].is_object(), "missing model metadata");
    assert!(json["reasons"].is_array(), "missing reasons array");
    assert!(
        json["top_findings"].is_array(),
        "missing top_findings array"
    );
    assert!(json["file_type"].is_string(), "missing file_type");
    assert!(json["sha256"].is_string(), "missing sha256");
    assert!(json["cleave"].is_object(), "missing cleave report");

    let classification = json["classification"]
        .as_str()
        .context("classification must be a string")?;
    assert!(
        ["benign", "suspicious", "hostile"].contains(&classification),
        "unexpected classification: {classification}"
    );
    Ok(())
}
