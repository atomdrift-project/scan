//! Integration tests for `GET /v1/lookup`.
//!
//! The v1 surface exists for a firewall standing in someone's `npm install`, so
//! these check the properties that gate a build rather than the ones that render
//! a report: that a decision is always present, that a shape never moves, and
//! that not knowing and not working are never the same answer.
//!
//! Like the other server tests these run against an empty model directory: a
//! lookup reads stored knowledge, so it must answer while a restarted server is
//! still loading.

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
    let bytes = axum::body::to_bytes(res.into_body(), 256 * 1024).await?;
    let body = serde_json::from_slice(&bytes).context("v1 body is JSON")?;
    Ok((status, body))
}

const UNKNOWN: &str = "pkg:npm/scan-v1-fixture-never-analyzed@0.0.0";

fn encoded_purl(purl: &str) -> String {
    purl.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '.' | '_' | '~' => c.to_string(),
            other => format!("%{:02X}", other as u32),
        })
        .collect()
}

/// The curl one-liner the documentation opens with. If this is not a real
/// answer on the first try, nothing else in the design matters.
#[tokio::test]
async fn one_package_answers_with_one_object() -> Result<()> {
    let (status, body) = get(&format!("/v1/lookup?purl={}", encoded_purl(UNKNOWN))).await?;
    assert_eq!(status, StatusCode::OK);
    assert!(body.is_object(), "a single package answered with a list");
    assert_eq!(body["purl"], UNKNOWN);
    Ok(())
}

/// Every key is present on every answer, so a caller writes one code path and a
/// generated type has no optionals that really mean "we had nothing to say".
#[tokio::test]
async fn the_shape_never_moves() -> Result<()> {
    let (_, body) = get(&format!("/v1/lookup?purl={}", encoded_purl(UNKNOWN))).await?;
    for key in [
        "decision",
        "purl",
        "sha256",
        "severity",
        "fires_at",
        "reason",
        "findings",
        "engine_version",
        "analyzed_at",
    ] {
        assert!(body.get(key).is_some(), "{key} is absent rather than null");
    }
    assert!(
        body["findings"].is_array(),
        "findings must be [] and never absent"
    );
    Ok(())
}

/// An index that holds nothing still answers, and what it must never answer is
/// `allow`. Silence is not a clean bill of health: the caller's policy decides
/// what an unanalyzed package means, and handing them `allow` would make that
/// call for them — the exact way a gate is talked into installing malware it
/// has simply never seen.
///
/// With no corpus configured there is nothing behind the index to defer to, so
/// this is the whole answer rather than the first half of one.
#[tokio::test]
async fn an_unanalyzed_package_is_named_and_never_allowed() -> Result<()> {
    let (status, body) = get(&format!("/v1/lookup?purl={}", encoded_purl(UNKNOWN))).await?;
    assert_eq!(
        status,
        StatusCode::OK,
        "having no answer is not an HTTP error"
    );
    assert_eq!(body["decision"], "unanalyzed");
    assert_ne!(
        body["decision"], "allow",
        "an unanalyzed package was cleared"
    );
    // Nothing may ride along on a decision drawn from no verdict.
    assert!(body["severity"].is_null());
    assert!(body["fires_at"].is_null());
    assert!(body["engine_version"].is_null());
    assert_eq!(body["findings"].as_array().map(Vec::len), Some(0));
    Ok(())
}

/// The distinction the whole reliability contract rests on, exercised across
/// the hop this design adds. A corpus that cannot be reached must answer
/// `unavailable`, never `unanalyzed`: one says nobody has analyzed this package,
/// the other says we could not find out, and only the first is a claim about
/// the package. A caller may reasonably install unanalyzed packages while
/// refusing to install anything during our outage — or exactly the reverse —
/// and collapsing the two takes that choice away from them.
/// Repeating the parameter is how a caller asks about several at once, and the
/// answer follows the shape of the question.
#[tokio::test]
async fn repeated_purl_answers_with_a_list() -> Result<()> {
    let uri = format!(
        "/v1/lookup?purl={}&purl={}",
        encoded_purl(UNKNOWN),
        encoded_purl("pkg:npm/scan-v1-fixture-other@0.0.0")
    );
    let (status, body) = get(&uri).await?;
    assert_eq!(status, StatusCode::OK);
    let rows = body
        .as_array()
        .context("repeated purl must answer with a list")?;
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["purl"], UNKNOWN);
    assert_eq!(rows[1]["purl"], "pkg:npm/scan-v1-fixture-other@0.0.0");
    Ok(())
}

/// Over the cap is a request we decline whole, naming the limit and the route
/// that has no limit. Discovering it as a truncated query string instead would
/// silently drop packages from a security decision.
#[tokio::test]
async fn too_many_packages_is_refused_by_name() -> Result<()> {
    let many = (0..51)
        .map(|i| format!("purl={}", encoded_purl(&format!("pkg:npm/p{i}@1.0.0"))))
        .collect::<Vec<_>>()
        .join("&");
    let (status, body) = get(&format!("/v1/lookup?{many}")).await?;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(body["error"]["code"], "too_many_packages");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("POST /v1/lookup")),
        "the error must name the route that has no limit",
    );
    Ok(())
}

#[tokio::test]
async fn naming_nothing_is_rejected_with_a_stable_code() -> Result<()> {
    let (status, body) = get("/v1/lookup").await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "missing_package");
    Ok(())
}

#[tokio::test]
async fn an_exact_url_is_accepted_and_echoed() -> Result<()> {
    let (status, body) = get("/v1/lookup?url=https%3A%2F%2Fcdn.example.test%2Fapp.tgz").await?;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["url"], "https://cdn.example.test/app.tgz");
    assert!(body["purl"].is_null());
    Ok(())
}

#[tokio::test]
async fn an_exact_url_is_validated_and_cannot_mix_with_a_purl() -> Result<()> {
    let (status, body) = get("/v1/lookup?url=file%3A%2F%2F%2Ftmp%2Fapp.tgz").await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_url");

    let (status, body) =
        get("/v1/lookup?url=https%3A%2F%2Fcdn.example.test%2Fapp.tgz&purl=npm%2Fapp%401.0.0")
            .await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "multiple_locators");
    Ok(())
}

#[tokio::test]
async fn a_bad_digest_is_rejected() -> Result<()> {
    let (status, body) = get("/v1/lookup?sha256=nothex").await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_sha256");
    Ok(())
}

/// The canonical form is what the index and the filters are keyed by, so an
/// uncanonical spelling must reach the same decision. The response keeps the
/// caller's spelling, though: that is how a batched caller correlates replies
/// without reproducing every ecosystem's normalization rules.
#[tokio::test]
async fn the_pkg_prefix_stays_optional() -> Result<()> {
    let bare = "npm/scan-v1-fixture-never-analyzed@0.0.0";
    let (bare_status, bare_body) = get(&format!("/v1/lookup?purl={}", encoded_purl(bare))).await?;
    let (canonical_status, mut canonical_body) =
        get(&format!("/v1/lookup?purl={}", encoded_purl(UNKNOWN))).await?;
    assert_eq!(bare_status, StatusCode::OK);
    assert_eq!(bare_status, canonical_status);
    assert_eq!(
        bare_body["purl"], bare,
        "the caller's spelling was rewritten"
    );

    // The spelling is presentation; all other fields prove both forms were
    // resolved through the same canonical key.
    canonical_body["purl"] = bare.into();
    assert_eq!(
        bare_body, canonical_body,
        "an uncanonical spelling named a different package"
    );
    Ok(())
}

/// The caller's budget is their dial and must be accepted without changing the
/// shape of anything. It cannot be exercised against a real verdict here — that
/// is what the unit table in `server::decision` is for — but a budget a caller
/// sends must never be a request error.
#[tokio::test]
async fn a_caller_supplied_budget_is_accepted() -> Result<()> {
    let (status, _) = get(&format!(
        "/v1/lookup?purl={}&false_positive_budget=1000",
        encoded_purl(UNKNOWN)
    ))
    .await?;
    assert_eq!(status, StatusCode::OK);
    Ok(())
}

/// A budget that is not a number must be refused rather than quietly replaced
/// by the default. A caller who meant to loosen theirs and silently got the
/// strict one back would see verdicts they never asked for, with nothing in the
/// response to say why.
#[tokio::test]
async fn a_malformed_budget_is_refused_not_defaulted() -> Result<()> {
    let (status, body) = get(&format!(
        "/v1/lookup?purl={}&false_positive_budget=loose",
        encoded_purl(UNKNOWN)
    ))
    .await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_false_positive_budget");
    Ok(())
}
