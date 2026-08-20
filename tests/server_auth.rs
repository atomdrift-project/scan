//! Integration tests for bearer-token authentication on the HTTP API.
//!
//! These drive the assembled router through `oneshot`, so they exercise the
//! real ACL middleware without binding a socket. Everything except the health
//! body-detail test runs without model artifacts: the middleware rejects a
//! request long before a handler needs a model.

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use scan::server::{ServerConfig, TokenDigest, build_app};
use std::net::SocketAddr;
use tower::ServiceExt;

const TOKEN: &str = "0123456789abcdef0123456789abcdef";

/// Inject a peer address, as `into_make_service_with_connect_info` does in
/// production. Without it the ACL fails closed and every request 403s.
fn with_peer<B>(mut req: Request<B>, ip: [u8; 4]) -> Request<B> {
    req.extensions_mut()
        .insert(ConnectInfo(SocketAddr::from((ip, 0))));
    req
}

fn loopback<B>(req: Request<B>) -> Request<B> {
    with_peer(req, [127, 0, 0, 1])
}

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
    let config = ServerConfig::new(
        SocketAddr::from(([127, 0, 0, 1], 0)),
        1024 * 1024,
        0,
        std::env::temp_dir(),
        None,
        4000,
        vec![],
        None,
        2,
        vec![],
    )?;
    if !authenticated {
        return Ok(config);
    }
    let digest = TokenDigest::new(TOKEN).map_err(anyhow::Error::msg)?;
    Ok(config.with_auth_token(Some(digest)))
}

/// The case this feature exists for: behind a Cloudflare tunnel, `cloudflared`
/// dials the service over loopback, so a loopback peer is *not* evidence of a
/// local caller. A loopback request without a token must be rejected.
#[tokio::test]
async fn loopback_is_not_exempt_from_the_token() -> Result<()> {
    let app = build_app(&config(true)?).await?;

    let response = app
        .oneshot(loopback(get("/analyze", None)?))
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
            .oneshot(loopback(get("/_/info", header.as_deref())?))
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
            .oneshot(loopback(get("/_/info", Some(&header))?))
            .await
            .context("request failed")?;
        assert_ne!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "valid token rejected for {header:?}",
        );
    }
    Ok(())
}

/// Health is the one route reachable without a credential, so tunnel and load
/// balancer probes work without holding a secret. An invalid token there is
/// ignored rather than rejected — a stale credential must not take monitoring
/// down.
#[tokio::test]
async fn health_never_requires_a_token() -> Result<()> {
    let app = build_app(&config(true)?).await?;

    for header in [None, Some("Bearer nonsense-nonsense"), Some("garbage")] {
        let response = app
            .clone()
            .oneshot(loopback(get("/_/health", header)?))
            .await
            .context("request failed")?;
        assert_ne!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "health rejected for {header:?}",
        );
    }
    Ok(())
}

/// The exemption is an exact path match. A prefix or suffix match would open
/// every route whose path starts with `/_/health`.
#[tokio::test]
async fn health_exemption_does_not_extend_to_similar_paths() -> Result<()> {
    let app = build_app(&config(true)?).await?;

    for uri in ["/_/healthz", "/_/health/", "/_/health/x", "/_/", "/"] {
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
            get("/_/info", Some(&authorization))?,
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
        .oneshot(loopback(get("/_/info", None)?))
        .await
        .context("request failed")?;

    assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    Ok(())
}

/// `/_/health` is public, so its body must not name the samples being
/// analysed. The diagnostic keys appear only for a request that authenticated.
///
/// Needs a ready server — the privileged keys live in the ready-state body —
/// so this one is gated on `SCAN_MODELS_DIR` like the analyze tests.
#[tokio::test]
async fn health_detail_requires_authentication() -> Result<()> {
    let Ok(models) = std::env::var("SCAN_MODELS_DIR") else {
        eprintln!("skipping: SCAN_MODELS_DIR is not set");
        return Ok(());
    };

    let digest = TokenDigest::new(TOKEN).map_err(anyhow::Error::msg)?;
    let config = ServerConfig::new(
        SocketAddr::from(([127, 0, 0, 1], 0)),
        1024 * 1024,
        0,
        models,
        None,
        4000,
        vec![],
        None,
        2,
        vec![],
    )?
    .with_auth_token(Some(digest));
    let app = build_app(&config).await?;

    let body_of = async |authorization: Option<&str>| -> Result<serde_json::Value> {
        let response = app
            .clone()
            .oneshot(loopback(get("/_/health", authorization)?))
            .await
            .context("health request failed")?;
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await?;
        serde_json::from_slice(&bytes).context("health body is not JSON")
    };

    // Readiness can take ~15s in release and longer in debug.
    let max_polls: u32 = if cfg!(debug_assertions) { 1800 } else { 600 };
    let mut ready = false;
    for _ in 0..max_polls {
        if body_of(None).await?["status"] == "ok" {
            ready = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(ready, "server did not become ready");

    let public = body_of(None).await?;
    for key in ["status", "rss_mb", "active_tasks", "uptime_secs", "load"] {
        assert!(public.get(key).is_some(), "monitors need {key}");
    }
    for key in [
        "long_running_tasks",
        "oldest_task",
        "stuck_orphans",
        "rayon_threads",
    ] {
        assert!(
            public.get(key).is_none(),
            "unauthenticated health leaked {key}: {public}",
        );
    }

    let private = body_of(Some(&format!("Bearer {TOKEN}"))).await?;
    for key in ["long_running_tasks", "stuck_orphans", "rayon_threads"] {
        assert!(
            private.get(key).is_some(),
            "authenticated health is missing {key}: {private}",
        );
    }
    Ok(())
}
