//! Error responses, rendered in the shape of the route family that returns them.
//!
//! The legacy routes answer `{"error": "<message>"}`; the `/v1` routes answer
//! `{"error": {"code": "<code>", "message": "<message>"}}`. Both carry the same
//! routing hints beside the error — `lane`, `retry_after_secs`,
//! `timeout_secs` — and a 429 adds the `Retry-After` header, so a router reads
//! a refusal the same way whichever route produced it.
//!
//! A message never carries internal detail: no error chain, no filesystem
//! path, no internal URL. That goes to the log, under the request's id.

use std::borrow::Cow;

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Json, Response};
use serde::Serialize;

/// One error answer. `code` is stable and machine-readable; `message` is for
/// humans and may be reworded freely.
#[derive(Clone, Debug)]
pub(super) struct ApiError {
    pub(super) status: StatusCode,
    pub(super) code: &'static str,
    pub(super) message: Cow<'static, str>,
    /// Which admission lane refused the request, on a 429.
    pub(super) lane: Option<&'static str>,
    /// When to try again, on a 429; also sent as `Retry-After`.
    pub(super) retry_after_secs: Option<u32>,
    /// The budget an analysis ran out of, on a 504.
    pub(super) timeout_secs: Option<u64>,
}

impl ApiError {
    pub(super) fn new(
        status: StatusCode,
        code: &'static str,
        message: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            lane: None,
            retry_after_secs: None,
            timeout_secs: None,
        }
    }

    pub(super) fn bad_request(code: &'static str, message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, code, message)
    }

    /// A fault of ours. The cause is logged where it happened.
    pub(super) fn internal() -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "Internal error",
        )
    }

    /// The model bundle is still loading.
    pub(super) fn starting() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "starting",
            "Server starting up",
        )
    }

    /// Startup failed. The reason is in the log, never in the answer.
    pub(super) fn init_failed() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "starting",
            "Server failed to initialize",
        )
    }

    pub(super) fn overloaded() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded",
            "Server overloaded (memory)",
        )
    }

    pub(super) fn timeout(secs: u64) -> Self {
        Self {
            timeout_secs: Some(secs),
            ..Self::new(
                StatusCode::GATEWAY_TIMEOUT,
                "analysis_timeout",
                "analysis timeout",
            )
        }
    }

    /// A refusal for want of capacity: a router answers it by trying another
    /// worker, which is why it is a fast 429 rather than a queue.
    pub(super) fn at_capacity(
        message: impl Into<Cow<'static, str>>,
        lane: Option<&'static str>,
        retry_after_secs: Option<u32>,
    ) -> Self {
        Self {
            lane,
            retry_after_secs,
            ..Self::new(StatusCode::TOO_MANY_REQUESTS, "at_capacity", message)
        }
    }

    /// The answer an analysis failure becomes.
    ///
    /// Typed errors are recognized by type; cleave's are not typed, so the rest
    /// fall to [`classify_analysis_error`]. A failure about the artifact keeps
    /// its root cause as the message, because that is what the caller can act
    /// on; one about us is a bare 500.
    pub(super) fn from_analysis(error: &anyhow::Error) -> Self {
        if let Some(busy) = error.downcast_ref::<crate::analysis::WhaleSlotBusy>() {
            // The same shape admission refuses with before the stream starts,
            // so a router treats both as "full, try the next worker".
            return Self::at_capacity(busy.to_string(), Some("whale"), Some(30));
        }
        let cause = || error.root_cause().to_string();
        // An artifact that could not be retrieved is answered from the type
        // rather than from any wording, because that verdict is the one this
        // fleet cannot afford to get wrong: see [`crate::fetch::Unretrievable`].
        if error
            .downcast_ref::<crate::fetch::Unretrievable>()
            .is_some()
        {
            return Self::new(StatusCode::UNPROCESSABLE_ENTITY, "unretrievable", cause());
        }
        match classify_analysis_error(&format!("{error:#}")) {
            StatusCode::UNSUPPORTED_MEDIA_TYPE => Self::new(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_artifact",
                cause(),
            ),
            StatusCode::UNPROCESSABLE_ENTITY => Self::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_artifact",
                cause(),
            ),
            _ => Self::internal(),
        }
    }

    /// Render in the `/v1` shape.
    pub(super) fn v1(&self) -> Response {
        self.render(V1Detail {
            code: self.code,
            message: &self.message,
        })
    }

    /// The `/v1` body alone, for a stream frame that carries an error.
    pub(super) fn v1_body(&self) -> serde_json::Value {
        serde_json::to_value(self.body(V1Detail {
            code: self.code,
            message: &self.message,
        }))
        .unwrap_or(serde_json::Value::Null)
    }

    fn body<E: Serialize>(&self, error: E) -> Body<'_, E> {
        Body {
            error,
            lane: self.lane,
            retry_after_secs: self.retry_after_secs,
            timeout_secs: self.timeout_secs,
        }
    }

    fn render<E: Serialize>(&self, error: E) -> Response {
        let mut response = (self.status, Json(self.body(error))).into_response();
        if let Some(secs) = self.retry_after_secs {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(secs));
        }
        response
    }
}

/// The legacy shape.
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        self.render(&*self.message)
    }
}

#[derive(Serialize)]
struct Body<'a, E> {
    error: E,
    #[serde(skip_serializing_if = "Option::is_none")]
    lane: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    retry_after_secs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout_secs: Option<u64>,
}

#[derive(Serialize)]
struct V1Detail<'a> {
    code: &'a str,
    message: &'a str,
}

/// The status an untyped analysis failure gets, read off its whole error chain.
///
/// An upstream limitation, kept in this one place: cleave reports every
/// failure as an `anyhow` string, so the only way to tell a malformed upload
/// (the caller's problem, 4xx) from a fault of ours (5xx) is the wording. The
/// whole chain is read, not the root cause, because "unsupported file type" is
/// often a middle link wrapped around an io error. Remove this when cleave
/// exposes typed errors.
pub(super) fn classify_analysis_error(chain: &str) -> StatusCode {
    let normalized = chain.to_ascii_lowercase();

    if normalized.contains("unsupported file type")
        || normalized.contains("unsupported archive type")
        || normalized.contains("unsupported compression")
    {
        StatusCode::UNSUPPORTED_MEDIA_TYPE
    } else if normalized.contains("archive is encrypted but no passwords configured")
        || normalized.contains("invalid ")
        || normalized.contains("not a valid ")
        || normalized.contains("truncated")
        || normalized.contains("corrupt")
        || normalized.contains("unexpected end of")
        || normalized.contains("too small")
        || normalized.contains("out of bounds")
        || normalized.contains("empty package.json")
        || normalized.contains("maximum archive depth")
        || normalized.contains("maximum decode depth")
        || normalized.contains("exceeded maximum")
        || normalized.contains("file count limit exceeded")
        || normalized.contains("file name too long")
    {
        StatusCode::UNPROCESSABLE_ENTITY
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn body_of(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("read body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    #[test]
    fn classify_unsupported_file_type_as_415() {
        assert_eq!(
            classify_analysis_error("Unsupported file type: Unknown"),
            StatusCode::UNSUPPORTED_MEDIA_TYPE
        );
    }

    #[test]
    fn classify_invalid_archive_as_422() {
        assert_eq!(
            classify_analysis_error("Archive is encrypted but no passwords configured"),
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    #[test]
    fn classify_unexpected_failure_as_500() {
        assert_eq!(
            classify_analysis_error("model evaluation failed"),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    /// A malformed upload is the caller's problem, not a server fault: cleave
    /// reports it from deep in the archive reader, so it arrives as a 422 only
    /// because the whole chain is classified.
    #[test]
    fn classify_corrupt_archive_as_422() {
        assert_eq!(
            classify_analysis_error(
                "cleave analysis of bad.tgz: Failed to read tar entry: corrupt deflate stream"
            ),
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    /// The wrapping context is part of what gets classified: a "truncated"
    /// cause buried under `cleave analysis of x.tgz` is still a 422, not a 500.
    #[test]
    fn classify_reads_the_whole_error_chain() {
        assert_eq!(
            classify_analysis_error("cleave analysis of x.tgz: truncated gzip stream"),
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    /// An artifact the registry would not serve is a 422, not a 500. Beamline
    /// reads a 5xx as a sick worker — it opens the breaker and retries the
    /// fleet — so answering a plain 404 from a package host with a server
    /// fault ejected healthy workers and reached poppy as an outage instead of
    /// a download failure.
    #[test]
    fn unretrievable_artifact_is_422_not_500() {
        let error = anyhow::Error::new(crate::fetch::Unretrievable {
            target: "https://proxy.golang.org/gitlab.com/!nebulous!labs/!sia/@v/v1.5.5-rc2.zip"
                .to_string(),
            outcome: fletch::fetch::Outcome::Failed(fletch::fetch::FetchError::Status(404)),
        });
        assert_eq!(
            ApiError::from_analysis(&error).status,
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    /// Recognized by type, through whatever context the analysis wrapped it in
    /// — the message alone says nothing a substring rule would catch.
    #[test]
    fn unretrievable_survives_added_context() {
        use anyhow::Context as _;

        let error = Err::<(), _>(anyhow::Error::new(crate::fetch::Unretrievable {
            target: "https://proxy.golang.org/example.com/m/@v/v1.0.0.zip".to_string(),
            outcome: fletch::fetch::Outcome::Unresolved(fletch::fetch::Unresolved::NoRelease),
        }))
        .context("analyzing pkg:golang/example.com/m@v1.0.0")
        .unwrap_err();
        assert_eq!(
            ApiError::from_analysis(&error).status,
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    /// The rest of the chain still classifies as it did: a fault this server
    /// is responsible for stays a 500, so a real outage is not quietly
    /// downgraded into "that package is unavailable".
    #[test]
    fn a_server_fault_is_still_500() {
        let error = anyhow::anyhow!("fetch unavailable: HTTP client could not be initialized");
        assert_eq!(
            ApiError::from_analysis(&error).status,
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    /// Internal detail stays in the log. A 500 says only that it was ours, and
    /// no answer carries the error chain, which names temp paths and context
    /// only an operator needs.
    #[tokio::test]
    async fn an_analysis_failure_carries_no_internal_detail() {
        use anyhow::Context as _;

        let fault = Err::<(), _>(anyhow::anyhow!("open /tmp/scan-x1y2/a.zip: denied"))
            .context("cleave analysis of a.zip")
            .unwrap_err();
        let body = body_of(ApiError::from_analysis(&fault).into_response()).await;
        assert_eq!(body, serde_json::json!({ "error": "Internal error" }));

        // A failure about the artifact keeps its root cause, and only that.
        let corrupt = Err::<(), _>(anyhow::anyhow!("corrupt deflate stream"))
            .context("cleave analysis of /tmp/scan-x1y2/a.tgz")
            .unwrap_err();
        let body = body_of(ApiError::from_analysis(&corrupt).into_response()).await;
        assert_eq!(
            body,
            serde_json::json!({ "error": "corrupt deflate stream" })
        );
    }

    /// One error, two shapes: the legacy routes keep their string, `/v1` its
    /// object, and the routing hints sit beside either one.
    #[tokio::test]
    async fn each_route_family_keeps_its_shape() {
        let refusal = ApiError::at_capacity("whale lane at capacity", Some("whale"), Some(30));

        let legacy = refusal.clone().into_response();
        assert_eq!(legacy.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(legacy.headers()[header::RETRY_AFTER], "30");
        assert_eq!(
            body_of(legacy).await,
            serde_json::json!({
                "error": "whale lane at capacity",
                "lane": "whale",
                "retry_after_secs": 30,
            })
        );

        let v1 = refusal.v1();
        assert_eq!(v1.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(v1.headers()[header::RETRY_AFTER], "30");
        assert_eq!(
            body_of(v1).await,
            serde_json::json!({
                "error": { "code": "at_capacity", "message": "whale lane at capacity" },
                "lane": "whale",
                "retry_after_secs": 30,
            })
        );

        let timeout = body_of(ApiError::timeout(60).into_response()).await;
        assert_eq!(
            timeout,
            serde_json::json!({ "error": "analysis timeout", "timeout_secs": 60 })
        );
    }
}
