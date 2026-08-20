//! One structured log line per HTTP request, emitted after the response is
//! produced.
//!
//! The line is the server's audit and traffic record, and is deliberately the
//! *only* place a request's edge facts are written: who connected, whether they
//! authenticated, what they asked for, what they got, and how long it took.
//! [`acl`](super::acl) tags every response with its [`Auth`] decision — including
//! the rejections it produces itself — so a denied request is one line here
//! rather than a warning there and a mystery here.
//!
//! Fields, in order: `id`, `status`, `dur_ms`, `peer`, `fwd`, `auth`,
//! `req_bytes`, `shared`, `sha256`, `purl`, `path`, `cred_len`, `cred_fp`,
//! `trace`, `ua`. Everything that reaches the line
//! is either generated here or parsed/bounded before it is printed, so a
//! hostile header cannot shape the log. A field with nothing to say is left
//! off the line entirely rather than printed empty.
//!
//! The `id` is allocated here and handed down to the handlers as a request
//! extension, so the access line and every analysis line about the same request
//! share one identifier.

use std::fmt;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, Request, header};
use axum::middleware::Next;
use axum::response::Response;

use super::AppState;

/// The server-assigned identifier for one request, shared by the access line
/// and every log line the handlers write while serving it.
#[derive(Clone, Copy, Debug)]
pub(super) struct RequestId(pub(super) u64);

impl RequestId {
    /// The numeric id, as every log line and diagnostic endpoint spells it.
    pub(super) fn get(self) -> u64 {
        self.0
    }
}

/// Marks a response that did not perform its own analysis: it attached to a
/// run already in flight for the same bytes (or the same PURL) and replayed
/// that result.
///
/// Without it a follower's access line is indistinguishable from a leader's,
/// which makes a burst of duplicate submissions look like a burst of work.
#[derive(Clone, Copy, Debug)]
pub(super) struct Shared;

/// Longest identifier echoed onto the access line. A PURL carrying a scope, a
/// version and qualifiers fits well inside this; past it the excerpt is still
/// enough to recognise what a caller sent.
const MAX_SUBJECT_LEN: usize = 200;

/// Which artifact a request was about.
///
/// The line carries the method and route, but a route is not an artifact:
/// `GET /lookup` and `POST /analyze` are the whole path for every caller, and
/// the identifier travels in a query string or a body that is deliberately
/// never logged raw. Each handler attaches the key it actually parsed, so one
/// line says both what was asked and what it was asked about — and a request
/// can be found by the artifact it concerned without joining `id` across two
/// lines.
///
/// Exactly one field is the caller's question; any others are what it
/// resolved to.
#[derive(Clone, Debug, Default)]
pub(super) struct Subject {
    sha256: Option<String>,
    purl: Option<String>,
    path: Option<String>,
}

impl Subject {
    /// A request about an artifact named by content digest — a `/lookup` by
    /// sha, or the digest of the bytes uploaded to `/analyze`.
    pub(super) fn sha256(value: &str) -> Self {
        Self {
            sha256: subject_field(value),
            ..Self::default()
        }
    }

    /// A request about an artifact named by package locator. `sha256` is the
    /// digest that locator resolved to, when it resolved to one — for a lookup
    /// that missed, or a locator that would not parse, there is no digest and
    /// the field is simply absent.
    pub(super) fn purl(value: &str, sha256: Option<&str>) -> Self {
        Self {
            sha256: sha256.and_then(subject_field),
            purl: subject_field(value),
            ..Self::default()
        }
    }

    /// A request about a file already on this host — `/analyze-path`. The path
    /// as the caller wrote it, which is what they will search the log for; the
    /// canonicalized form is on the handler's own line when they differ.
    pub(super) fn path(value: &str) -> Self {
        Self {
            path: subject_field(value),
            ..Self::default()
        }
    }
}

/// Bound and strip caller-supplied text so it cannot forge or flood a log
/// line: no control characters (a newline would fabricate a second line) and
/// no unbounded length. Empty in, nothing out.
fn subject_field(value: &str) -> Option<String> {
    let clean = sanitize(value, MAX_SUBJECT_LEN);
    (!clean.is_empty()).then_some(clean)
}

/// Attach a [`Subject`] to a response, so the access line for this request
/// names what it was about.
pub(super) fn with_subject(mut response: Response, subject: Subject) -> Response {
    response.extensions_mut().insert(subject);
    response
}

/// How a request fared at the access-control edge.
///
/// Carried on the response so the access line can report the outcome — and,
/// for a rejection, the specific gate that produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Auth {
    /// Presented a valid bearer token.
    Token,
    /// Trusted because no `--token-file` is configured: the API is open.
    Open,
    /// No valid credential, on the one route that does not require one.
    Anonymous,
    /// Refused: peer address unavailable. A wiring bug, not a client error.
    NoPeerInfo,
    /// Refused: peer IP matched no `--allow-cidr` network.
    PeerDenied,
    /// Refused: a non-loopback peer asked for a loopback-only route.
    LoopbackOnly,
    /// Refused: no `Authorization` header at all.
    NoCredential,
    /// Refused: an `Authorization` header that is not a usable `Bearer`
    /// credential — wrong scheme, missing separator, or empty.
    MalformedCredential,
    /// Refused: a well-formed bearer credential that is not this server's
    /// token. Carries the credential's length and [`Fingerprint`], which is
    /// what distinguishes a stale token from a truncated or mangled one.
    BadToken { len: usize, fp: Fingerprint },
}

impl Auth {
    fn as_str(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::Open => "open",
            Self::Anonymous => "anon",
            Self::NoPeerInfo => "no-peer-info",
            Self::PeerDenied => "peer-denied",
            Self::LoopbackOnly => "loopback-only",
            Self::NoCredential => "no-credential",
            Self::MalformedCredential => "malformed-credential",
            Self::BadToken { .. } => "bad-token",
        }
    }

    /// Whether this outcome refused the request. Denials are the security
    /// record, so they are logged a level above ordinary traffic.
    fn denied(self) -> bool {
        !matches!(self, Self::Token | Self::Open | Self::Anonymous)
    }
}

/// The first four bytes of a credential's SHA-256, hex-encoded on demand.
///
/// A rejected token cannot be logged — but "which token was it" is the only
/// question a 401 raises, and without an answer the operator is left guessing
/// between a stale credential, a truncated one, and a client pointed at the
/// wrong server. The fingerprint answers it without recording a secret: 32
/// bits of a preimage-resistant digest identify a token to an operator who
/// already holds it, and are useless to anyone who does not. Compare against
/// whichever token file is believed current:
///
/// ```sh
/// printf %s "$(cat ~/.tok/scan)" | shasum -a 256 | cut -c1-8
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Fingerprint(pub(super) [u8; 4]);

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// A header value trusted enough to log: printable ASCII only, length-capped.
///
/// Header values cannot contain newlines (the HTTP parser rejects them), but
/// they can carry control characters and arbitrary length. Both are stripped
/// here rather than in the log formatter, so nothing unbounded or unprintable
/// reaches the record.
fn loggable(headers: &HeaderMap, name: header::HeaderName, max_len: usize) -> Option<String> {
    let raw = headers.get(name)?.to_str().ok()?;
    let clean = sanitize(raw, max_len);
    (!clean.is_empty()).then_some(clean)
}

/// Drop control characters and bound the length. The one gate every piece of
/// caller-supplied text passes through before it reaches a log line.
fn sanitize(raw: &str, max_len: usize) -> String {
    raw.chars()
        .filter(|c| !c.is_ascii_control())
        .take(max_len)
        .collect()
}

/// The originating client address a fronting proxy reported, if any.
///
/// Behind a Cloudflare tunnel every peer is loopback, so the socket address
/// says nothing about who is calling. This is advisory — it is never used for
/// access control — and it is parsed as an [`IpAddr`] before it is logged, so
/// a forged header can only ever produce a well-formed address.
fn forwarded_for(headers: &HeaderMap) -> Option<IpAddr> {
    const HEADERS: [&str; 3] = ["cf-connecting-ip", "x-forwarded-for", "x-real-ip"];
    HEADERS.iter().find_map(|name| {
        let value = headers.get(*name)?.to_str().ok()?;
        // X-Forwarded-For is a client-to-proxy chain; the first entry is the
        // originating client.
        value.split(',').next()?.trim().parse().ok()
    })
}

/// Request Content-Length as a number, for the bytes-in field.
fn content_length(headers: &HeaderMap) -> Option<u64> {
    headers
        .get(header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

/// Middleware emitting the per-request access line.
///
/// Installed outermost in [`super::build_app`] so it observes every request,
/// including the ones the ACL rejects.
pub(super) async fn access_log(
    State(state): State<Arc<AppState>>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    let started = Instant::now();
    let id = state.next_request_id();
    req.extensions_mut().insert(RequestId(id));

    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let peer = req
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|c| c.0.ip());
    let fwd = forwarded_for(req.headers());
    let req_bytes = content_length(req.headers());
    // Callers propagating their own correlation id (beamline does) get it
    // echoed into the line, which is what joins the two services' logs.
    let trace = loggable(
        req.headers(),
        header::HeaderName::from_static("x-request-id"),
        64,
    );
    let ua = loggable(req.headers(), header::USER_AGENT, 120);

    let response = next.run(req).await;

    let status = response.status();
    let auth = response.extensions().get::<Auth>().copied();
    let dur_ms = crate::duration_ms(started.elapsed());

    // A healthy liveness probe every few seconds is not a record worth keeping;
    // anything else about /_/health is. Server faults are warnings — they are
    // the lines an operator greps for first — and a missing peer address is an
    // error because it means the ConnectInfo wiring is broken, not the client.
    let level = if auth == Some(Auth::NoPeerInfo) {
        tracing::Level::ERROR
    } else if status.is_server_error() || auth.is_some_and(Auth::denied) {
        tracing::Level::WARN
    } else if path == super::acl::HEALTH_ROUTE {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    // Only a rejected credential has these, and only then are they worth a
    // reader's attention: they are what turns "401" into "the client is holding
    // a different token than the one this process loaded".
    let shared = response.extensions().get::<Shared>().map(|_| true);
    let subject = response
        .extensions()
        .get::<Subject>()
        .cloned()
        .unwrap_or_default();
    let (cred_len, cred_fp) = match auth {
        Some(Auth::BadToken { len, fp }) => (Some(len), Some(fp.to_string())),
        _ => (None, None),
    };

    // `tracing` fixes a level per call site, so the branches are spelled out.
    // The field set is identical across them by construction: one macro body,
    // one place to change it.
    macro_rules! emit {
        ($level:path) => {
            tracing::event!(
                $level,
                id,
                status = status.as_u16(),
                dur_ms,
                peer = peer.map(tracing::field::display),
                fwd = fwd.map(tracing::field::display),
                auth = auth.map(Auth::as_str),
                req_bytes,
                shared,
                sha256 = subject.sha256.as_deref(),
                purl = subject.purl.as_deref(),
                path = subject.path.as_deref(),
                cred_len,
                cred_fp = cred_fp.as_deref(),
                trace = trace.as_deref(),
                ua = ua.as_deref(),
                "{method} {path}",
            )
        };
    }
    match level {
        tracing::Level::ERROR => emit!(tracing::Level::ERROR),
        tracing::Level::WARN => emit!(tracing::Level::WARN),
        tracing::Level::DEBUG => emit!(tracing::Level::DEBUG),
        _ => emit!(tracing::Level::INFO),
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    /// A lookup key reaches the line as the handler parsed it.
    #[test]
    fn subject_carries_both_keys() {
        let by_sha = Subject::sha256("ab".repeat(32).as_str());
        assert_eq!(by_sha.sha256.as_deref(), Some("ab".repeat(32).as_str()));
        assert_eq!(by_sha.purl, None);

        let by_purl = Subject::purl("pkg:npm/left-pad@1.3.0", Some("beef"));
        assert_eq!(by_purl.purl.as_deref(), Some("pkg:npm/left-pad@1.3.0"));
        assert_eq!(by_purl.sha256.as_deref(), Some("beef"));
    }

    /// A local-file request is named by the path the caller asked for.
    #[test]
    fn subject_carries_a_path() {
        let subject = Subject::path("/srv/samples/a.bin");
        assert_eq!(subject.path.as_deref(), Some("/srv/samples/a.bin"));
        assert_eq!(subject.sha256, None);
        assert_eq!(subject.purl, None);
    }

    /// A PURL lookup that missed resolves to no digest, and an absent field is
    /// left off the line rather than printed empty.
    #[test]
    fn subject_omits_unresolved_digest() {
        assert_eq!(
            Subject::purl("pkg:npm/left-pad@1.3.0", Some("")).sha256,
            None
        );
        assert_eq!(Subject::purl("pkg:npm/left-pad@1.3.0", None).sha256, None);
    }

    /// The query string is caller-controlled: a newline in it must not be able
    /// to fabricate a second log line, and length must not be able to flood
    /// one. This is why the raw query is never logged directly.
    #[test]
    fn subject_cannot_forge_or_flood_a_log_line() {
        let forged = Subject::purl("pkg:npm/a\n2026-01-01 INFO forged line", None);
        // The newline is gone, so what would have been a second line is now
        // just text inside the first one's `purl` field.
        assert_eq!(
            forged.purl.as_deref(),
            Some("pkg:npm/a2026-01-01 INFO forged line"),
        );

        let flood = Subject::sha256(&"a".repeat(MAX_SUBJECT_LEN * 4));
        assert_eq!(flood.sha256.map(|s| s.len()), Some(MAX_SUBJECT_LEN));
    }

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (name, value) in pairs {
            if let Ok(v) = HeaderValue::from_str(value) {
                h.insert(header::HeaderName::from_static(name), v);
            }
        }
        h
    }

    #[test]
    fn forwarded_prefers_the_cloudflare_header_and_the_first_chain_entry() {
        let h = headers(&[
            ("x-forwarded-for", "203.0.113.9, 70.41.3.18"),
            ("cf-connecting-ip", "198.51.100.7"),
        ]);
        assert_eq!(
            forwarded_for(&h).map(|ip| ip.to_string()).as_deref(),
            Some("198.51.100.7")
        );

        let h = headers(&[("x-forwarded-for", "203.0.113.9, 70.41.3.18")]);
        assert_eq!(
            forwarded_for(&h).map(|ip| ip.to_string()).as_deref(),
            Some("203.0.113.9")
        );
    }

    /// A forged forwarding header can only ever reach the log as an address.
    #[test]
    fn forwarded_rejects_anything_that_is_not_an_address() {
        let h = headers(&[("x-real-ip", "not-an-ip")]);
        assert_eq!(forwarded_for(&h), None);
    }

    /// A user agent is bounded and stripped of control characters before it is
    /// logged, so a hostile client cannot shape the record.
    #[test]
    fn user_agent_is_bounded_and_stripped() {
        let h = headers(&[("user-agent", "beam\tline/0.3")]);
        assert_eq!(
            loggable(&h, header::USER_AGENT, 120).as_deref(),
            Some("beamline/0.3")
        );

        let long = "x".repeat(500);
        let h = headers(&[("user-agent", &long)]);
        assert_eq!(
            loggable(&h, header::USER_AGENT, 120).map(|s| s.len()),
            Some(120)
        );
    }
}
