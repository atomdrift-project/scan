//! Helpers shared by the `server_*` integration tests.

// Each test binary compiles this module and uses its own subset of it; an
// `expect` would be unfulfilled in the binaries that happen to use them all.
#![allow(
    dead_code,
    reason = "each test binary uses its own subset of these helpers"
)]

use std::net::SocketAddr;

use anyhow::{Context, Result};
use axum::Router;
use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use scan::server::{ServerConfig, build_app};
use tower::ServiceExt;

/// Inject a peer address, as `into_make_service_with_connect_info` does in
/// production. Without it the ACL fails closed and every request 403s.
pub(crate) fn with_peer<B>(mut req: Request<B>, ip: [u8; 4]) -> Request<B> {
    req.extensions_mut()
        .insert(ConnectInfo(SocketAddr::from((ip, 0))));
    req
}

/// A request from a loopback peer.
pub(crate) fn loopback<B>(req: Request<B>) -> Request<B> {
    with_peer(req, [127, 0, 0, 1])
}

/// A server pointed at an empty model directory. Background loading fails and
/// it never becomes ready, which is the point for every route that must answer
/// without a model: lookups, refusals, and the ACL.
pub(crate) fn unready_config() -> ServerConfig {
    ServerConfig {
        bind: SocketAddr::from(([127, 0, 0, 1], 0)),
        max_body_size: 1024 * 1024,
        model_dir: std::env::temp_dir(),
        workers: 2,
        ..ServerConfig::default()
    }
}

/// The app for [`unready_config`].
pub(crate) async fn app() -> Result<Router> {
    build_app(&unready_config()).await
}

/// Percent-encode a PURL for a query string.
pub(crate) fn encoded_purl(purl: &str) -> String {
    purl.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '.' | '_' | '~' => c.to_string(),
            other => format!("%{:02X}", other as u32),
        })
        .collect()
}

/// The model bundle under `SCAN_MODELS_DIR`. A test that calls this is
/// ignored by default and run on purpose, so a missing bundle fails it rather
/// than letting it pass without testing anything.
pub(crate) fn model_dir() -> Result<std::path::PathBuf> {
    std::env::var("SCAN_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .context("set SCAN_MODELS_DIR to run integration tests against real model artifacts")
}

/// A config over the real model bundle.
pub(crate) fn ready_config() -> Result<ServerConfig> {
    Ok(ServerConfig {
        model_dir: model_dir()?,
        ..unready_config()
    })
}

/// Install a tracing subscriber so server logs are visible on test failure.
/// Silently ignored if another test in the process already installed one.
pub(crate) fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_test_writer()
        .try_init();
}

/// Build the app for `config` and wait until it reports ready. YARA warmup
/// can take ~15s in release and longer in debug builds.
pub(crate) async fn ready_app(config: &ServerConfig) -> Result<Router> {
    let app = build_app(config).await.context("failed to build app")?;
    let polls: u32 = if cfg!(debug_assertions) { 1800 } else { 600 };
    for _ in 0..polls {
        let response = app
            .clone()
            .oneshot(loopback(
                Request::builder().uri("/_/health").body(Body::empty())?,
            ))
            .await
            .context("health request failed")?;
        let (status, body) = status_and_json(response).await?;
        if status == StatusCode::OK {
            return Ok(app);
        }
        // A startup that failed will not recover by waiting; the reason is in
        // the server log above.
        if body["status"] == "failed" {
            anyhow::bail!("server failed to start: {body}");
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    anyhow::bail!("server did not become ready within {}s", polls / 10)
}

/// The status and JSON body of a response. A body that is not one JSON
/// document — an NDJSON stream, an empty body — reads as `Null`.
pub(crate) async fn status_and_json(
    response: axum::response::Response,
) -> Result<(StatusCode, serde_json::Value)> {
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 16 * 1024 * 1024).await?;
    Ok((
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    ))
}
