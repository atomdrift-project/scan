//! Integration tests for bearer-token authentication on the HTTP API.
//!
//! These drive the assembled router through `oneshot`, so they exercise the
//! real ACL middleware without binding a socket. Everything except the health
//! body-detail test runs without model artifacts: the middleware rejects a
//! request long before a handler needs a model.

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

/// The admin routes reload what the server runs, so `--allow-cidr` — which
/// grants analysis — must not grant them too. Without a token they answer
/// loopback alone; with one, the token is their credential like any route's.
///
/// Asked with GET: a peer the ACL lets through reaches the router and gets 405,
/// which tells the two outcomes apart without running a reload.
#[tokio::test]
async fn admin_routes_need_loopback_or_a_token() -> Result<()> {
    let allowed = [203, 0, 113, 7];
    let open = build_app(&ServerConfig {
        allow_cidrs: vec![Cidr::parse("203.0.113.0/24").map_err(anyhow::Error::msg)?],
        ..config(false)?
    })
    .await?;
    for route in ["/_/reload", "/_/update"] {
        let response = open
            .clone()
            .oneshot(with_peer(get(route, None)?, allowed))
            .await?;
        assert_eq!(
            response.status(),
            StatusCode::FORBIDDEN,
            "an allowed peer reached {route} on a server with no token",
        );
        let response = open.clone().oneshot(loopback(get(route, None)?)).await?;
        assert_eq!(
            response.status(),
            StatusCode::METHOD_NOT_ALLOWED,
            "loopback must still reach {route}",
        );
        // Analysis stays open to the allowed peer.
        let response = open
            .clone()
            .oneshot(with_peer(get("/analyze", None)?, allowed))
            .await?;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    let authenticated = build_app(&ServerConfig {
        allow_cidrs: vec![Cidr::parse("203.0.113.0/24").map_err(anyhow::Error::msg)?],
        ..config(true)?
    })
    .await?;
    let authorization = format!("Bearer {TOKEN}");
    let response = authenticated
        .oneshot(with_peer(get("/_/reload", Some(&authorization))?, allowed))
        .await?;
    assert_eq!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "with a token configured, the token is the admin credential",
    );
    Ok(())
}

/// `/_/health` is public, so its body must not name the samples being
/// analysed. The diagnostic keys appear only for a request that authenticated.
///
/// Needs a ready server — the privileged keys live in the ready-state body.
#[tokio::test]
#[ignore = "needs a model bundle: set SCAN_MODELS_DIR and run with --ignored"]
async fn health_detail_requires_authentication() -> Result<()> {
    let digest = TokenDigest::new(TOKEN).map_err(anyhow::Error::msg)?;
    let config = ServerConfig {
        auth_digest: Some(digest),
        ..common::ready_config()?
    };
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
