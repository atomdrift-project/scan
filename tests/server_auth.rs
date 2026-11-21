//! Integration tests for bearer-token authentication on the HTTP API.
//!
//! These drive the assembled router through `oneshot`, so they exercise the
//! real ACL middleware without binding a socket. None needs model artifacts:
//! the middleware decides long before a handler needs a model.
//!
//! A request the ACL lets through is shown by `GET /v1/analyze` answering 405:
//! the router refused the method, so the request got past the middleware
//! without running an analysis.

mod common;

use anyhow::{Context, Result};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::{loopback, with_peer};
use scan::server::{Cidr, ServerConfig, TokenDigest, build_app};
use tower::ServiceExt;

const TOKEN: &str = "0123456789abcdef0123456789abcdef";

fn get(uri: &str, authorization: Option<&str>) -> Result<Request<Body>> {
    let mut builder = Request::builder().uri(uri);
    if let Some(value) = authorization {
        builder = builder.header("authorization", value);
    }
    builder.body(Body::empty()).context("build request")
}

/// A config pointing at an empty model directory. Background loading fails and
/// the server never becomes ready, which is irrelevant to the ACL: the
/// middleware runs ahead of every handler.
fn config(authenticated: bool) -> Result<ServerConfig> {
    let auth_digest = if authenticated {
        Some(TokenDigest::new(TOKEN).map_err(anyhow::Error::msg)?)
    } else {
        None
    };
    Ok(ServerConfig {
        auth_digest,
        ..common::unready_config()
    })
}

/// The case this feature exists for: behind a Cloudflare tunnel, `cloudflared`
/// dials the service over loopback, so a loopback peer is *not* evidence of a
/// local caller. A loopback request without a token must be rejected.
#[tokio::test]
async fn loopback_is_not_exempt_from_the_token() -> Result<()> {
    let app = build_app(&config(true)?).await?;

    let response = app
        .oneshot(loopback(get("/v1/analyze", None)?))
        .await
        .context("request failed")?;

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response
            .headers()
            .get("www-authenticate")
            .and_then(|v| v.to_str().ok()),
        Some("Bearer"),
    );
    Ok(())
}

#[tokio::test]
async fn rejects_wrong_and_malformed_credentials() -> Result<()> {
    let app = build_app(&config(true)?).await?;

    let truncated = TOKEN.get(..TOKEN.len() - 1).unwrap_or_default();
    for header in [
        None,
        Some("Bearer wrong-token-wrong-token".to_string()),
        // A correct token under the wrong scheme is still no credential.
        Some(format!("Basic {TOKEN}")),
        Some(TOKEN.to_string()),
        Some(format!("Bearer{TOKEN}")),
        Some("Bearer ".to_string()),
        Some("Bearer".to_string()),
        // Truncations and extensions of a valid token.
        Some(format!("Bearer {truncated}")),
        Some(format!("Bearer {TOKEN}x")),
        Some(format!("Bearer {}", TOKEN.to_uppercase())),
    ] {
        let response = app
            .clone()
            .oneshot(loopback(get("/v1/lookup", header.as_deref())?))
            .await
            .context("request failed")?;
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "expected 401 for {header:?}",
        );
    }
    Ok(())
}

#[tokio::test]
async fn accepts_a_valid_token() -> Result<()> {
    let app = build_app(&config(true)?).await?;

    for header in [
        format!("Bearer {TOKEN}"),
        // RFC 9110 §11.1: the scheme is case-insensitive.
        format!("bearer {TOKEN}"),
    ] {
        let response = app
            .clone()
            .oneshot(loopback(get("/v1/analyze", Some(&header))?))
            .await
            .context("request failed")?;
        assert_eq!(
            response.status(),
            StatusCode::METHOD_NOT_ALLOWED,
            "valid token rejected for {header:?}",
        );
    }
    Ok(())
}

/// No path is exempt from the token — including the retired `/_/health`
/// probe path, which once was — and an unknown path is refused by the ACL
/// before the router can say it does not exist.
#[tokio::test]
async fn no_path_is_exempt_from_the_token() -> Result<()> {
    let app = build_app(&config(true)?).await?;

    for uri in ["/_/health", "/_/info", "/healthz", "/", "/v1/nope"] {
        let response = app
            .clone()
            .oneshot(loopback(get(uri, None)?))
            .await
            .context("request failed")?;
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "expected 401 for {uri}",
        );
    }
    Ok(())
}

/// The peer-IP gate runs first and independently: a valid token does not buy
/// access from a peer outside `--allow-cidr`.
#[tokio::test]
async fn a_valid_token_does_not_bypass_the_ip_acl() -> Result<()> {
    let app = build_app(&config(true)?).await?;
    let authorization = format!("Bearer {TOKEN}");

    let response = app
        .oneshot(with_peer(
            get("/v1/analyze", Some(&authorization))?,
            [203, 0, 113, 7],
        ))
        .await
        .context("request failed")?;

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    Ok(())
}

/// Without `--token-file` the API behaves exactly as it did before tokens
/// existed, so an upgrade does not lock out an existing deployment.
#[tokio::test]
async fn unauthenticated_server_is_unchanged() -> Result<()> {
    let app = build_app(&config(false)?).await?;

    let response = app
        .oneshot(loopback(get("/v1/analyze", None)?))
        .await
        .context("request failed")?;

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    Ok(())
}

/// Only the `/v1` API is served. The retired diagnostic and admin routes are
/// gone rather than hidden, and `--allow-cidr` grants a peer the API alone.
#[tokio::test]
async fn retired_routes_are_gone() -> Result<()> {
    let allowed = [203, 0, 113, 7];
    let app = build_app(&ServerConfig {
        allow_cidrs: vec![Cidr::parse("203.0.113.0/24").map_err(anyhow::Error::msg)?],
        ..config(false)?
    })
    .await?;
    for route in [
        "/_/health",
        "/_/info",
        "/_/stats",
        "/_/memory",
        "/_/requests",
        "/_/threads",
        "/_/reload",
        "/_/update",
    ] {
        for (method, peer) in [("GET", allowed), ("POST", [127, 0, 0, 1])] {
            let request = Request::builder()
                .method(method)
                .uri(route)
                .body(Body::empty())?;
            let response = app.clone().oneshot(with_peer(request, peer)).await?;
            assert_eq!(
                response.status(),
                StatusCode::NOT_FOUND,
                "{method} {route} is still served",
            );
        }
    }
    let response = app
        .oneshot(with_peer(get("/v1/analyze", None)?, allowed))
        .await?;
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    Ok(())
}
