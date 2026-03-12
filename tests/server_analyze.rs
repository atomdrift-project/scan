//! Integration tests for the /analyze endpoint.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use litmus::server::{ServerConfig, build_app};
use tower::ServiceExt;

fn model_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LITMUS_MODELS_DIR") {
        return std::path::PathBuf::from(d);
    }
    litmus::models_repo::model_dir()
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
    (
        format!("multipart/form-data; boundary={boundary}"),
        body,
    )
}

/// Submit an encrypted zip via /analyze and verify JSON response structure.
#[tokio::test]
async fn analyze_encrypted_zip_returns_json() {
    let testdata = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/encrypted.zip");
    if !testdata.exists() {
        panic!("testdata/encrypted.zip not found — copy a test sample there");
    }
    let file_bytes = std::fs::read(&testdata).unwrap();

    let config = ServerConfig {
        model_dir: model_dir(),
        ..Default::default()
    };
    let app = build_app(&config).await.expect("failed to build app");

    let (content_type, body) = multipart_body(&file_bytes, "encrypted.zip");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header("content-type", content_type)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(response.into_body(), 10 * 1024 * 1024)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body_bytes)
        .expect("response must be valid JSON");

    // Every response must have these fields, regardless of classification.
    assert!(json["classification"].is_string(), "missing classification");
    assert!(json["probability"].is_number(), "missing probability");
    assert!(json["thresholds"].is_object(), "missing thresholds");
    assert!(json["finding_counts"].is_object(), "missing finding_counts");
    assert!(json["formula"].is_string(), "missing formula");
    assert!(json["reasons"].is_array(), "missing reasons array");
    assert!(json["top_findings"].is_array(), "missing top_findings array");
    assert!(json["file_type"].is_string(), "missing file_type");
    assert!(json["sha256"].is_string(), "missing sha256");
    assert!(json["cleave"].is_object(), "missing cleave report");

    let classification = json["classification"].as_str().unwrap();
    assert!(
        ["benign", "suspicious", "hostile"].contains(&classification),
        "unexpected classification: {classification}"
    );
}
