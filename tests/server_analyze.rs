//! Integration tests for the /analyze endpoint.

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use litmus::server::{build_app, ServerConfig};
use std::net::SocketAddr;
use tower::ServiceExt;
use tracing_subscriber::EnvFilter;

/// Inject a loopback ConnectInfo on a request so the ACL middleware sees a
/// peer address. axum's `into_make_service_with_connect_info` runs only when
/// the server is started via `axum::serve`; tests using `oneshot` must add
/// it manually or every request 403s.
fn loopback<B>(mut req: Request<B>) -> Request<B> {
    req.extensions_mut()
        .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 0))));
    req
}

/// Install a tracing subscriber so server logs are visible on test failure.
/// Silently ignored if another test in the process already installed one.
fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_test_writer()
        .try_init();
}

fn model_dir() -> Result<std::path::PathBuf> {
    std::env::var("LITMUS_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .context("set LITMUS_MODELS_DIR to run integration tests against real model artifacts")
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
    init_tracing();
    if std::env::var_os("LITMUS_MODELS_DIR").is_none() {
        eprintln!("skipping: LITMUS_MODELS_DIR is not set");
        return Ok(());
    }

    let testdata = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/encrypted.zip");
    assert!(
        testdata.exists(),
        "testdata/encrypted.zip not found — copy a test sample there"
    );
    let file_bytes = std::fs::read(&testdata).context("failed to read test archive")?;

    let config = ServerConfig::new(
        SocketAddr::from(([127, 0, 0, 1], 8081)),
        100 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
        model_dir()?,
        None,
        4000,
        vec![],
        None,
        2,
        vec![],
    )?;
    let app = build_app(&config).await.context("failed to build app")?;

    // Wait for background resource loading to complete before sending requests.
    // YARA warmup can take ~15s in release and longer in debug builds.
    let max_health_polls: u32 = if cfg!(debug_assertions) { 1800 } else { 600 };
    eprintln!(
        "waiting for server readiness (up to {}s)...",
        max_health_polls / 10
    );
    let mut ready = false;
    for _ in 0..max_health_polls {
        let resp = app
            .clone()
            .oneshot(loopback(
                Request::builder()
                    .uri("/_/health")
                    .body(Body::empty())
                    .context("failed to build health request")?,
            ))
            .await
            .context("health request failed")?;
        if resp.status() == StatusCode::OK {
            ready = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(
        ready,
        "server did not become ready within {}s",
        max_health_polls / 10
    );

    let (content_type, body) = multipart_body(&file_bytes, "encrypted.zip");

    let response = app
        .oneshot(loopback(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header("content-type", content_type)
                .body(Body::from(body))
                .context("failed to build analyze request")?,
        ))
        .await
        .context("analyze request failed")?;

    let status = response.status();
    let body_bytes = axum::body::to_bytes(response.into_body(), 10 * 1024 * 1024)
        .await
        .context("failed to read response body")?;

    assert_eq!(
        status,
        StatusCode::OK,
        "expected 200 but got {status}: {}",
        String::from_utf8_lossy(&body_bytes),
    );

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
