//! `POST /v1/analyze` asks before it works.
//!
//! The route used to analyze unconditionally. Measured against production, three
//! consecutive analyses of `pkg:cargo/tokio@1.40.0` ran 291s, 161s and 116s while
//! `GET /v1/lookup` answered the same question from the corpus in one hop — so
//! every one of them re-derived a verdict the fleet already held.
//!
//! These run against an empty model directory, which is what makes the assertions
//! sharp: an analysis that actually runs cannot succeed and answers non-200 (see
//! `server_v1_analyze::a_refusal_keeps_a_status_a_router_can_act_on`). So a 200
//! carrying a decision can only have come from the precheck, and a non-200 can
//! only mean the route went on to analyze. Nothing here needs a working model to
//! tell those two apart.
//!
//! The corpus is a stub standing in for hopper, so the test covers the real
//! chain — index miss, then corpus — rather than a seam invented for it.

mod common;

use anyhow::Result;
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use scan::server::{ServerConfig, build_app};
use sha2::{Digest, Sha256};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Mutex;
use tower::ServiceExt;

/// What the stub corpus answers with.
#[derive(Clone, Copy)]
enum Holds {
    /// A verdict, tagged so the answer's provenance is unambiguous.
    Verdict,
    /// Reachable, and holding nothing. Not a verdict.
    Nothing,
}

/// The queries the stub was asked, and how many times.
#[derive(Default)]
struct Asked {
    count: AtomicUsize,
    queries: Mutex<Vec<String>>,
}

/// A stand-in for hopper's `/v1/lookup`.
///
/// Returns its base URL and the record of what it was asked. Everything other
/// than the lookup 404s, including `/_/replica` — a corpus that cannot report
/// its lag is simply not known to be stale.
async fn stub_corpus(holds: Holds) -> Result<(String, Arc<Asked>)> {
    let asked = Arc::new(Asked::default());
    let seen = Arc::clone(&asked);
    let app = Router::new().route(
        "/v1/lookup",
        axum::routing::get(move |uri: axum::http::Uri| {
            let seen = Arc::clone(&seen);
            async move {
                seen.count.fetch_add(1, Ordering::SeqCst);
                seen.queries
                    .lock()
                    .await
                    .push(uri.query().unwrap_or_default().to_owned());
                match holds {
                    Holds::Verdict => (
                        StatusCode::OK,
                        [("content-type", "application/json")],
                        r#"{"sha256":null,"purl":null,"fires_at":3,
                            "engine_version":"corpus-probe",
                            "analyzed_at":"2026-08-01T00:00:00Z",
                            "reason":null,"findings":[]}"#,
                    ),
                    Holds::Nothing => (
                        StatusCode::NOT_FOUND,
                        [("content-type", "application/json")],
                        r#"{"error":"unknown"}"#,
                    ),
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).await?;
    let base = format!("http://{}", listener.local_addr()?);
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok((base, asked))
}

async fn app_with_corpus(base: Option<&str>) -> Result<Router> {
    let config = ServerConfig {
        hopper: base.map(str::to_owned),
        ..common::unready_config()
    };
    build_app(&config).await
}

async fn analyze(app: Router, uri: &str, body: &str) -> Result<(StatusCode, serde_json::Value)> {
    let req = common::loopback(
        Request::builder()
            .method("POST")
            .uri(uri)
            .body(Body::from(body.to_owned()))?,
    );
    let res = app.oneshot(req).await?;
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 256 * 1024).await?;
    // The short-circuit answers one JSON object. A run that reaches the streaming
    // path answers NDJSON, which will not parse whole — and that is itself the
    // signal, so a parse failure is Null rather than an error.
    let parsed = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    Ok((status, parsed))
}

/// The whole point: a verdict the corpus already holds is answered without
/// spending an analysis slot.
#[tokio::test(flavor = "multi_thread")]
async fn a_held_verdict_answers_without_analyzing() -> Result<()> {
    let (base, asked) = stub_corpus(Holds::Verdict).await?;
    let app = app_with_corpus(Some(&base)).await?;
    let (status, body) = analyze(app, "/v1/analyze?purl=npm%2Fleft-pad%401.3.0", "").await?;

    assert_eq!(
        status,
        StatusCode::OK,
        "a held verdict was not answered with"
    );
    assert_eq!(
        body["engine_version"], "corpus-probe",
        "the answer did not come from the corpus: {body}",
    );
    assert!(!body["decision"].is_null(), "no decision in {body}");
    assert_eq!(
        asked.count.load(Ordering::SeqCst),
        1,
        "the corpus was not asked exactly once"
    );
    let queries = asked.queries.lock().await;
    assert!(
        queries[0].contains("purl="),
        "a named package must be asked about by PURL, got {:?}",
        queries[0],
    );
    drop(queries);
    Ok(())
}

/// A policy the operator did not configure is still a question the corpus can
/// answer.
///
/// This guards the rule that cost the fleet its corpus. Sharing used to require
/// the request's policy to equal this server's own default — and beamline
/// resolves `references` for a PURL while an unflagged server defaults to
/// `dependencies,references`, so no ordinary request ever matched. Every
/// verdict was treated as private to its caller: not read from the index, not
/// offered to hopper, not answerable from what the corpus already held. The
/// analysis was correct and nothing was kept.
///
/// `follow=none` differs from any default this server can be started with, so a
/// 200 here can only mean the precheck ran for a policy that is not the
/// configured one. Under the old rule this analyzed instead, and answered
/// non-200 against the empty model directory.
#[tokio::test(flavor = "multi_thread")]
async fn a_policy_the_operator_did_not_configure_still_reads_the_corpus() -> Result<()> {
    let (base, asked) = stub_corpus(Holds::Verdict).await?;
    let app = app_with_corpus(Some(&base)).await?;
    let (status, body) = analyze(
        app,
        "/v1/analyze?purl=npm%2Fleft-pad%401.3.0&follow=none",
        "",
    )
    .await?;

    assert_eq!(
        status,
        StatusCode::OK,
        "a non-default policy did not consult the corpus, so it analyzed: {body}",
    );
    assert_eq!(
        body["engine_version"], "corpus-probe",
        "the answer did not come from the corpus: {body}",
    );
    assert_eq!(
        asked.count.load(Ordering::SeqCst),
        1,
        "the corpus was not asked exactly once"
    );
    Ok(())
}

/// An upload is asked about by the digest of the bytes in hand — the identity —
/// so re-sending an artifact the fleet has already seen costs nothing either.
#[tokio::test(flavor = "multi_thread")]
async fn an_upload_is_answered_from_a_verdict_held_for_its_digest() -> Result<()> {
    let (base, asked) = stub_corpus(Holds::Verdict).await?;
    let app = app_with_corpus(Some(&base)).await?;
    let artifact = "these exact bytes have been seen before";
    let sha = format!("{:x}", Sha256::digest(artifact.as_bytes()));

    let (status, body) = analyze(app, "/v1/analyze", artifact).await?;

    assert_eq!(
        status,
        StatusCode::OK,
        "a held verdict for these bytes was not answered with"
    );
    assert_eq!(
        body["engine_version"], "corpus-probe",
        "not from the corpus: {body}"
    );
    let queries = asked.queries.lock().await;
    assert!(
        queries[0].contains(&format!("sha256={sha}")),
        "an upload must be asked about by its digest, got {:?}",
        queries[0],
    );
    drop(queries);
    Ok(())
}

/// An upload's PURL must never reach the resolver. It is provenance, not
/// identity, and the corpus answers a `?sha256=…&purl=…` query on *either* key.
///
/// Shipped once and caught in production: 25 bytes of text sent as
/// `?purl=pkg:npm/chalk@5.3.0` came back `allow`, carrying chalk's digest, with
/// the bytes never looked at. Any artifact could be laundered through a
/// reputable coordinate — the exact failure this route exists to catch. The
/// index path was already safe (`pick_verdict` accepts a PURL's verdict only
/// when it describes the same bytes); the corpus path had no such guard.
#[tokio::test(flavor = "multi_thread")]
async fn an_uploads_purl_never_reaches_the_resolver() -> Result<()> {
    let (base, asked) = stub_corpus(Holds::Verdict).await?;
    let app = app_with_corpus(Some(&base)).await?;
    let artifact = "these bytes are not that package";
    let sha = format!("{:x}", Sha256::digest(artifact.as_bytes()));

    let (_status, body) = analyze(app, "/v1/analyze?purl=npm%2Fchalk%405.3.0", artifact).await?;

    let queries = asked.queries.lock().await;
    if let Some(query) = queries.first() {
        assert!(
            !query.contains("purl="),
            "the upload's PURL was sent to the corpus, which answers on it: {query:?}",
        );
        assert!(
            query.contains(&format!("sha256={sha}")),
            "the upload was not asked about by its own digest: {query:?}",
        );
    }
    // Whatever comes back, it must not be a verdict about some other artifact.
    if let Some(reported) = body["sha256"].as_str() {
        assert_eq!(
            reported, sha,
            "answered about a different artifact than the bytes uploaded",
        );
    }
    drop(queries);
    Ok(())
}

/// `unknown` is not a verdict. Answering with it would tell a caller nobody has
/// analyzed the artifact — which is precisely what they asked us to change.
#[tokio::test(flavor = "multi_thread")]
async fn nothing_held_still_analyzes() -> Result<()> {
    let (base, asked) = stub_corpus(Holds::Nothing).await?;
    let app = app_with_corpus(Some(&base)).await?;
    let (status, body) = analyze(app, "/v1/analyze?purl=npm%2Fleft-pad%401.3.0", "").await?;

    assert_eq!(
        asked.count.load(Ordering::SeqCst),
        1,
        "the corpus was not consulted"
    );
    assert_ne!(
        status,
        StatusCode::OK,
        "an empty corpus was answered with as though it were a verdict: {body}",
    );
    Ok(())
}

/// An unreachable corpus says nothing about the artifact. Turning that into an
/// answer would make an outage look like a refusal to work — the failure this
/// keeps `unavailable` distinct from `unknown` to avoid.
#[tokio::test(flavor = "multi_thread")]
async fn an_unreachable_corpus_still_analyzes() -> Result<()> {
    // Port 1 on loopback: nothing listens, and connection is refused promptly.
    let app = app_with_corpus(Some("http://127.0.0.1:1")).await?;
    let (status, body) = analyze(app, "/v1/analyze?purl=npm%2Fleft-pad%401.3.0", "").await?;
    assert_ne!(
        status,
        StatusCode::OK,
        "an unreachable corpus was answered with as though it were a verdict: {body}",
    );
    Ok(())
}

/// No corpus configured at all is the same claim as one that cannot be reached:
/// not evidence about the artifact, so the analysis still runs.
#[tokio::test(flavor = "multi_thread")]
async fn no_corpus_configured_still_analyzes() -> Result<()> {
    let app = app_with_corpus(None).await?;
    let (status, body) = analyze(app, "/v1/analyze?purl=npm%2Fleft-pad%401.3.0", "").await?;
    assert_ne!(
        status,
        StatusCode::OK,
        "answered without a corpus or a model: {body}"
    );
    Ok(())
}

/// `force=1` is the way to re-analyze something already known — after an engine
/// upgrade, say. It has to beat a verdict the corpus is holding, or it is not a
/// force at all.
#[tokio::test(flavor = "multi_thread")]
async fn force_analyzes_even_when_a_verdict_is_held() -> Result<()> {
    let (base, asked) = stub_corpus(Holds::Verdict).await?;
    let app = app_with_corpus(Some(&base)).await?;
    let (status, body) =
        analyze(app, "/v1/analyze?purl=npm%2Fleft-pad%401.3.0&force=1", "").await?;

    assert_eq!(
        asked.count.load(Ordering::SeqCst),
        0,
        "force still consulted the corpus, which is work it asked us to skip",
    );
    assert_ne!(
        status,
        StatusCode::OK,
        "force was answered from the held verdict it asked us to ignore: {body}",
    );
    Ok(())
}
