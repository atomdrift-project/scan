//! Integration tests for `GET /status?sha256=… | ?purl=…`.
//!
//! The endpoint exists for one caller: the one whose connection did not survive
//! the analysis. A proxy that gives up at its own ceiling leaves the run going
//! here, and from the outside that is indistinguishable from a run that never
//! started — both are `404 unknown sample` on /lookup. Getting that wrong costs
//! a whole second analysis of a package that is already minutes in.
//!
//! Like the lookup tests, these run against an empty model directory: status
//! reads the flight registry and the verdict index, so it must answer while a
//! restarted server is still loading.

use anyhow::{Context, Result};
use axum::Router;
use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode};
use scan::server::{ServerConfig, build_app};
use std::net::SocketAddr;
use tower::ServiceExt;

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

async fn get(uri: &str) -> Result<(StatusCode, serde_json::Value)> {
    let app = app().await?;
    let req = loopback(Request::builder().uri(uri).body(Body::empty())?);
    let res = app.oneshot(req).await?;
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 64 * 1024).await?;
    let body = serde_json::from_slice(&bytes).context("status body is JSON")?;
    Ok((status, body))
}

/// A package nothing will ever have analyzed. The verdict index is a real
/// directory shared by every scan on this machine, so "not stored" has to be a
/// key that cannot be stored.
const UNKNOWN: &str = "pkg:npm/scan-status-fixture-never-analyzed@0.0.0";

/// The same package spelled without `pkg:`, which every route accepts and
/// canonicalizes. Written out rather than derived, so the test needs no
/// fallible step of its own to set up.
const UNKNOWN_BARE: &str = "npm/scan-status-fixture-never-analyzed@0.0.0";

fn encoded_purl(purl: &str) -> String {
    purl.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '.' | '_' | '~' => c.to_string(),
            other => format!("%{:02X}", other as u32),
        })
        .collect()
}

/// Nothing running and nothing stored. The caller reads this as "lost" when it
/// follows a dispatch of their own, which is why the state is reported plainly
/// rather than as an error: there is nothing wrong, there is just no run.
#[tokio::test]
async fn a_package_nobody_is_analyzing_is_unknown() -> Result<()> {
    let (status, body) = get(&format!("/status?purl={}", encoded_purl(UNKNOWN))).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["state"], "unknown");
    assert_eq!(body["purl"], UNKNOWN);
    Ok(())
}

/// A flight is keyed by the canonical PURL, so an uncanonical spelling has to
/// resolve to the same run — otherwise a caller who reconnects with `pkg:`
/// omitted would miss the analysis they started and pay for a second one.
#[tokio::test]
async fn the_canonical_form_is_what_a_run_is_keyed_by() -> Result<()> {
    let (status, body) = get(&format!("/status?purl={}", encoded_purl(UNKNOWN_BARE))).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body["purl"], UNKNOWN,
        "an uncanonical spelling named a different artifact",
    );
    Ok(())
}

#[tokio::test]
async fn a_status_call_naming_nothing_is_rejected() -> Result<()> {
    let (status, _) = get("/status").await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    Ok(())
}

#[tokio::test]
async fn an_unparseable_purl_is_rejected() -> Result<()> {
    let (status, _) = get(&format!("/status?purl={}", encoded_purl("not a purl"))).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    Ok(())
}

/// A digest names an artifact just as a PURL does, so status answers for it —
/// beamline knows the digest for an upload and the PURL for a package, and it
/// must be able to ask about whichever it holds.
#[tokio::test]
async fn a_digest_can_be_asked_about_too() -> Result<()> {
    let sha = "b".repeat(64);
    let (status, body) = get(&format!("/status?sha256={sha}")).await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["state"], "unknown");
    assert_eq!(body["sha256"], sha);
    Ok(())
}
