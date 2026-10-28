//! Integration tests for `POST /v1/analyze?path=`, which scans a file already
//! on the server's host.
//!
//! Like the other v1 tests these run against an empty model directory, so no
//! request reaches a real analysis. What they check is the gate in front of it:
//! the feature is off unless configured, refusals do not reveal which paths
//! exist, and a server that could be reached remotely refuses to start with it.

mod common;

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::{encoded_purl, loopback, unready_config};
use scan::server::{Cidr, ServerConfig, TokenDigest, build_app};
use tower::ServiceExt;

async fn post(app: Router, uri: &str, body: &str) -> Result<(StatusCode, Vec<u8>)> {
    let req = loopback(
        Request::builder()
            .method("POST")
            .uri(uri)
            .body(Body::from(body.to_owned()))?,
    );
    let res = app.oneshot(req).await?;
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 256 * 1024).await?;
    Ok((status, bytes.to_vec()))
}

fn code(body: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v["error"]["code"].as_str().map(str::to_owned))
        .unwrap_or_default()
}

fn uri(path: &std::path::Path) -> String {
    format!("/v1/analyze?path={}", encoded_purl(&path.to_string_lossy()))
}

/// A server allowed to read files under `dir`.
async fn serving(dir: &tempfile::TempDir) -> Result<Router> {
    build_app(&ServerConfig {
        local_paths: vec![dir.path().canonicalize()?],
        ..unready_config()
    })
    .await
}

#[tokio::test]
async fn the_feature_is_off_unless_configured() -> Result<()> {
    let (status, body) = post(common::app().await?, "/v1/analyze?path=%2Fetc%2Fpasswd", "").await?;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(code(&body), "local_paths_disabled");
    Ok(())
}

/// The route must not be an oracle for which paths exist on the host.
#[tokio::test]
async fn a_missing_file_and_a_forbidden_one_read_the_same() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let outside = tempfile::tempdir()?;
    let secret = outside.path().join("secret");
    std::fs::write(&secret, b"secret")?;

    let missing = post(serving(&dir).await?, &uri(&dir.path().join("missing")), "").await?;
    let forbidden = post(serving(&dir).await?, &uri(&secret), "").await?;
    assert_eq!(missing.0, StatusCode::FORBIDDEN);
    assert_eq!(code(&missing.1), "path_not_allowed");
    assert_eq!(missing, forbidden);
    Ok(())
}

#[tokio::test]
async fn an_allowed_file_reaches_the_analysis() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("left-pad-1.3.0.tgz");
    std::fs::write(&file, b"not really a tarball")?;
    let (status, body) = post(serving(&dir).await?, &uri(&file), "").await?;
    // No model is loaded, so the analysis itself cannot run; what matters is
    // that the request got past every check in front of it.
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "{}",
        String::from_utf8_lossy(&body)
    );
    Ok(())
}

#[tokio::test]
async fn malformed_path_requests_are_refused_by_name() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("a.bin");
    std::fs::write(&file, b"a")?;

    let (status, body) = post(
        serving(&dir).await?,
        "/v1/analyze?path=relative%2Fa.bin",
        "",
    )
    .await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code(&body), "invalid_path");

    let (status, body) = post(serving(&dir).await?, &uri(&file), "bytes").await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code(&body), "path_with_body");

    let with_purl = format!("{}&purl=npm%2Fa%401.0.0", uri(&file));
    let (status, body) = post(serving(&dir).await?, &with_purl, "").await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code(&body), "multiple_locators");
    Ok(())
}

/// Local paths are for this host's callers, or for remote callers the operator
/// has both named and credentialed.
#[tokio::test]
async fn a_server_reachable_remotely_refuses_to_start_without_both_gates() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let local_paths: Vec<PathBuf> = vec![dir.path().canonicalize()?];
    let config = |bind: [u8; 4], cidrs: bool, token: bool| -> Result<ServerConfig> {
        Ok(ServerConfig {
            bind: SocketAddr::from((bind, 0)),
            local_paths: local_paths.clone(),
            allow_cidrs: if cidrs {
                vec![Cidr::parse("203.0.113.0/24").map_err(anyhow::Error::msg)?]
            } else {
                Vec::new()
            },
            auth_digest: if token {
                Some(
                    TokenDigest::new("0123456789abcdef0123456789abcdef")
                        .map_err(anyhow::Error::msg)?,
                )
            } else {
                None
            },
            ..unready_config()
        })
    };
    assert!(
        build_app(&config([127, 0, 0, 1], false, false)?)
            .await
            .is_ok()
    );
    assert!(build_app(&config([0, 0, 0, 0], true, true)?).await.is_ok());
    for (cidrs, token) in [(false, false), (true, false), (false, true)] {
        assert!(
            build_app(&config([0, 0, 0, 0], cidrs, token)?)
                .await
                .is_err(),
            "0.0.0.0 with cidrs={cidrs} token={token} must be refused"
        );
    }
    // Loopback in name only: any other 127/8 address is still not 127.0.0.1.
    assert!(
        build_app(&config([127, 0, 0, 2], false, false)?)
            .await
            .is_err()
    );
    Ok(())
}
