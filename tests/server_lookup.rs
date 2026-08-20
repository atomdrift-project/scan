//! Integration tests for the lookup routes (`GET /sha256/{sha}`, `GET /purl`).
//!
//! Every test here runs against an empty model directory, so the server never
//! becomes ready — which is the point. A lookup reads stored knowledge and the
//! bloom filters, neither of which needs the model, so it must answer while a
//! restarted server is still loading. If these ever start failing on readiness,
//! the routes have picked up a dependency they should not have.

use anyhow::{Context, Result};
use axum::Router;
use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use scan::server::{ServerConfig, build_app};
use std::net::SocketAddr;
use tower::ServiceExt;

/// Inject a peer address, as `into_make_service_with_connect_info` does in
/// production. Without it the ACL fails closed and every request 403s.
fn loopback<B>(mut req: Request<B>) -> Request<B> {
    req.extensions_mut()
        .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 0))));
    req
}

async fn app() -> Result<Router> {
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
    build_app(&config).await
}

/// Status and parsed JSON body of one GET.
async fn get(uri: &str) -> Result<(StatusCode, serde_json::Value, Option<String>)> {
    let app = app().await?;
    let req = loopback(Request::builder().uri(uri).body(Body::empty())?);
    let res = app.oneshot(req).await?;
    let status = res.status();
    let cache_control = res
        .headers()
        .get(axum::http::header::CACHE_CONTROL)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let bytes = axum::body::to_bytes(res.into_body(), 64 * 1024).await?;
    let body = serde_json::from_slice(&bytes).context("lookup body is JSON")?;
    Ok((status, body, cache_control))
}

/// Nothing stored is `404 unknown sample` — not an empty 200, and not a 500.
/// The bloom decision rides along, so one round trip answers both "do we have
/// an analysis" and "does a filter already vouch for this".
#[tokio::test]
async fn unknown_sha_is_a_404_carrying_the_bloom_decision() -> Result<()> {
    let sha = "a".repeat(64);
    let (status, body, cache) = get(&format!("/sha256/{sha}")).await?;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"], "unknown sample");
    assert_eq!(body["bloom"], "unknown", "no filters installed fails closed");
    assert_eq!(
        cache.as_deref(),
        Some("no-store"),
        "a miss becomes a hit as soon as anything analyzes the artifact",
    );
    Ok(())
}

#[tokio::test]
async fn unknown_purl_is_a_404_carrying_the_bloom_decision() -> Result<()> {
    let (status, body, _) = get("/purl?purl=pkg%3Anpm%2Fleft-pad%401.3.0").await?;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"], "unknown sample");
    assert_eq!(body["bloom"], "unknown");
    Ok(())
}

/// A PURL's own grammar carries `/`, `?` and `#`, so it travels as a query
/// parameter. Qualifiers must survive the trip — they are part of the identity
/// the filters and the index key on.
#[tokio::test]
async fn purl_qualifiers_survive_the_query_encoding() -> Result<()> {
    let (status, body, _) =
        get("/purl?purl=pkg%3Ageneric%2Fx%401.0%3Fdownload_url%3Dhttps%3A%2F%2Fe.test%2Fx.tgz")
            .await?;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "a well-formed PURL with qualifiers is a miss, not a 400",
    );
    assert_eq!(body["error"], "unknown sample");
    Ok(())
}

#[tokio::test]
async fn malformed_keys_are_rejected() -> Result<()> {
    let (status, body, _) = get("/sha256/not-a-sha").await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "invalid sha256");

    let (status, body, _) = get(&format!("/sha256/{}", "g".repeat(64))).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST, "64 chars but not hex");
    assert_eq!(body["error"], "invalid sha256");

    let (status, body, _) = get("/purl").await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "missing purl");

    let (status, body, _) = get("/purl?purl=left-pad").await?;
    assert_eq!(status, StatusCode::BAD_REQUEST, "no PURL scheme or type");
    assert_eq!(body["error"], "invalid purl");
    Ok(())
}

/// Case and surrounding whitespace name the same artifact.
#[tokio::test]
async fn digests_are_matched_case_insensitively() -> Result<()> {
    let (status, _, _) = get(&format!("/sha256/{}", "A".repeat(64))).await?;
    assert_eq!(status, StatusCode::NOT_FOUND, "uppercase hex is still hex");
    Ok(())
}
