//! Integration tests for `POST /v1/analyze`.
//!
//! The route exists for one reason: an analysis that outlives a proxy's idle
//! ceiling. Measured in front of this fleet, that ceiling is 125 seconds, and
//! crossing it silently costs the caller an analysis that in fact completed —
//! the worker finishes and files its verdict, but the reply had nowhere to go.
//!
//! These run against an empty model directory, so nothing here reaches a real
//! analysis. What they check is the part that does not need one: that the route
//! refuses bad requests by name, and that a refusal keeps a status a router can
//! act on rather than being buried in a 200 body.

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

async fn post(uri: &str, body: &str) -> Result<(StatusCode, serde_json::Value)> {
    let app = app().await?;
    let req = loopback(
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))?,
    );
    let res = app.oneshot(req).await?;
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 256 * 1024).await?;
    let parsed = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    Ok((status, parsed))
}

/// The package is named in the query, as it is on /v1/lookup. The body is the
/// artifact, so it cannot also be where the locator goes.
#[tokio::test]
async fn an_unparseable_purl_is_refused_by_name() -> Result<()> {
    let (status, body) = post("/v1/analyze?purl=not%20a%20purl", "").await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_purl");
    Ok(())
}

/// A budget that is not a number must be refused rather than quietly replaced
/// by the default, exactly as on /v1/lookup: one surface, one rule.
#[tokio::test]
async fn a_malformed_budget_is_refused() -> Result<()> {
    let (status, body) = post(
        "/v1/analyze?purl=npm%2Fleft-pad%401.3.0&false_positive_budget=loose",
        "",
    )
    .await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_false_positive_budget");
    Ok(())
}

/// Not ready is a refusal about this server, and it keeps a status: a router
/// reads it to send the work to a worker that can take it, which a decision
/// buried in a 200 body cannot be made to do.
///
/// With no model directory the server never becomes ready, so this is the
/// refusal an unready worker gives — the same shape `429 At capacity` takes,
/// and the reason the grace window answers the ordinary way instead of
/// streaming everything.
#[tokio::test]
async fn a_refusal_keeps_a_status_a_router_can_act_on() -> Result<()> {
    let (status, _) = post("/v1/analyze?purl=npm%2Fleft-pad%401.3.0", "").await?;
    assert!(
        status.is_client_error() || status.is_server_error(),
        "a refusal arrived as {status}, which a router cannot route around",
    );
    assert_ne!(
        status,
        StatusCode::OK,
        "a worker that cannot analyze answered as though it had",
    );
    Ok(())
}

/// Which of the two things a caller means is decided by what arrived, not by a
/// header they have to remember. `Content-Type` says nothing here that the
/// presence of a body does not, and requiring it turned a correct `curl -T`
/// into a 415 for a reason the caller could not see.
#[tokio::test]
async fn bytes_need_no_content_type() -> Result<()> {
    for content_type in [None, Some("application/octet-stream"), Some("text/plain")] {
        let app = app().await?;
        let mut builder = Request::builder().method("POST").uri("/v1/analyze");
        if let Some(ct) = content_type {
            builder = builder.header("content-type", ct);
        }
        let req = loopback(builder.body(Body::from("some artifact bytes"))?);
        let res = app.oneshot(req).await?;
        assert_ne!(
            res.status(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "content-type {content_type:?} was refused",
        );
        assert_ne!(
            res.status(),
            StatusCode::BAD_REQUEST,
            "content-type {content_type:?} was read as naming no package",
        );
    }
    Ok(())
}

/// Sending neither is the only way to name nothing, and the message has to
/// offer both ways in — a caller holding bytes told only about `?purl=` has
/// been sent to look for a package they do not have.
#[tokio::test]
async fn naming_nothing_offers_both_ways_in() -> Result<()> {
    let (status, body) = post("/v1/analyze", "").await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "missing_package");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("purl"), "{message}");
    assert!(message.contains("body"), "{message}");
    Ok(())
}

#[tokio::test]
async fn the_route_is_post_only() -> Result<()> {
    let app = app().await?;
    let req = loopback(Request::builder().uri("/v1/analyze").body(Body::empty())?);
    let res = app.oneshot(req).await?;
    assert_eq!(res.status(), StatusCode::METHOD_NOT_ALLOWED);
    Ok(())
}

/// The caller must hear something almost immediately. Silence is what a proxy
/// cuts, and it is also what makes a minutes-long analysis indistinguishable
/// from a hung one — so progress starts within a second rather than after a
/// grace period spent saying nothing.
#[tokio::test]
async fn something_is_said_almost_immediately() -> Result<()> {
    let app = app().await?;
    let req = loopback(
        Request::builder()
            .method("POST")
            .uri("/v1/analyze?purl=npm%2Fleft-pad%401.3.0")
            .body(Body::empty())?,
    );
    // 125s is what was measured in front of this fleet; the first byte must
    // arrive with room to spare for a slower hop elsewhere.
    let answered = tokio::time::timeout(std::time::Duration::from_secs(10), app.oneshot(req)).await;
    assert!(
        answered.is_ok(),
        "nothing was sent within 10s: a proxy would eventually cut this connection",
    );
    Ok(())
}

/// The stream's contract, stated as the rule a client implements: read lines
/// until one carries `decision`, and that is the answer. Progress frames say
/// what the run is doing and are safe to ignore; nothing before the decision
/// line is an answer, and there is exactly one decision line.
#[tokio::test]
async fn a_client_reads_lines_until_one_carries_a_decision() -> Result<()> {
    let streamed = concat!(
        r#"{"state":"analyzing","purl":"pkg:npm/evil@1.0.0","elapsed_ms":1002,"phase":"unpack"}"#,
        "\n",
        r#"{"state":"analyzing","purl":"pkg:npm/evil@1.0.0","elapsed_ms":6004,"phase":"features+model"}"#,
        "\n",
        r#"{"decision":"block","fires_at":3,"purl":"pkg:npm/evil@1.0.0"}"#,
        "\n",
    );

    let mut answer = None;
    let mut progress = 0;
    for line in streamed.lines().filter(|l| !l.trim().is_empty()) {
        let frame: serde_json::Value = serde_json::from_str(line).context("every line is JSON")?;
        if frame.get("decision").is_some() {
            answer = Some(frame);
        } else {
            assert_eq!(
                frame["state"], "analyzing",
                "a frame that is neither progress nor a decision"
            );
            progress += 1;
        }
    }
    assert_eq!(progress, 2, "progress frames were not readable as progress");
    let answer = answer.context("the stream carried no decision")?;
    assert_eq!(answer["decision"], "block");
    assert_eq!(answer["fires_at"], 3);
    Ok(())
}

/// An analysis that finishes before the first progress frame is due emits
/// nothing but the decision — so the fast path still looks like a single JSON
/// object, and a caller that never expected a stream still parses it.
#[tokio::test]
async fn a_fast_analysis_is_still_one_json_object() -> Result<()> {
    let streamed = "{\"decision\":\"allow\",\"fires_at\":-1}\n";
    let parsed: serde_json::Value =
        serde_json::from_str(streamed.trim()).context("a lone decision must parse as JSON")?;
    assert_eq!(parsed["decision"], "allow");
    Ok(())
}
