//! Per-request access control for the litmus HTTP API: a peer-IP ACL, and
//! bearer-token authentication when `--token-file` is set.
//!
//! Two gates run in order on every request:
//!
//! 1. **Peer IP.** Loopback connections pass. Every other peer must match one
//!    of the configured `--allow-cidr` networks, and may never reach a
//!    [`LOOPBACK_ONLY_ROUTES`] entry.
//! 2. **Bearer token.** When a token is configured, every route except
//!    [`HEALTH_ROUTE`] requires `Authorization: Bearer <token>`. Loopback is
//!    deliberately *not* exempt: the server is meant to sit behind a
//!    Cloudflare tunnel, where `cloudflared` runs on the host and dials the
//!    service over loopback, so every remote request arrives with a loopback
//!    peer address. Exempting loopback would exempt the entire internet.
//!
//! That same property makes [`LOOPBACK_ONLY_ROUTES`] defence in depth rather
//! than a guarantee: behind a tunnel, "loopback" means "local *or* tunnelled".
//! `/analyze-path` stays on that list, but the real protection is an empty
//! `--allowed-dirs`, which makes it reject every request.
//!
//! Requests clearing both gates carry a [`Trusted`] marker, which handlers use
//! to decide whether a response may include privileged diagnostic detail.
//!
//! Every response — pass or reject — is tagged with the [`Auth`] decision that
//! produced it, which [`access_log`](super::access::access_log) turns into the
//! `auth=` field of that request's one access line. This module therefore logs
//! nothing itself: a rejection is a normal, expected outcome, and it belongs in
//! the traffic record rather than in a warning of its own.
//!
//! CIDR matching is hand-rolled (no extra crate dependency) and supports
//! both IPv4 and IPv6.

use std::fmt;
use std::net::IpAddr;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, Request, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Json, Response};
use sha2::{Digest, Sha256};

use super::AppState;
use super::access::Auth;

/// A single CIDR network parsed from `--allow-cidr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cidr {
    network: IpAddr,
    prefix_len: u8,
}

impl Cidr {
    /// Parse `addr/prefix`. Both v4 and v6 are supported.
    pub fn parse(s: &str) -> Result<Self, String> {
        let (addr, prefix) = s
            .split_once('/')
            .ok_or_else(|| format!("missing /prefix in {s:?}"))?;
        let network: IpAddr = addr
            .parse()
            .map_err(|e| format!("invalid address {addr:?}: {e}"))?;
        let prefix_len: u8 = prefix
            .parse()
            .map_err(|e| format!("invalid prefix {prefix:?}: {e}"))?;
        let max = if network.is_ipv4() { 32 } else { 128 };
        if prefix_len > max {
            return Err(format!(
                "prefix length {prefix_len} exceeds max {max} for {addr}"
            ));
        }
        Ok(Self {
            network,
            prefix_len,
        })
    }

    /// Returns true if `ip` lies inside this network.
    #[must_use]
    pub fn contains(&self, ip: IpAddr) -> bool {
        match (self.network, ip) {
            (IpAddr::V4(net), IpAddr::V4(ip)) => {
                if self.prefix_len == 0 {
                    return true;
                }
                let mask: u32 = u32::MAX << (32 - self.prefix_len);
                (u32::from(net) & mask) == (u32::from(ip) & mask)
            }
            (IpAddr::V6(net), IpAddr::V6(ip)) => {
                if self.prefix_len == 0 {
                    return true;
                }
                let mask: u128 = u128::MAX << (128 - self.prefix_len);
                (u128::from(net) & mask) == (u128::from(ip) & mask)
            }
            _ => false,
        }
    }
}

/// Parse a comma-separated list of CIDRs. Whitespace is trimmed; empty
/// entries are skipped. Returns the first parse error encountered.
pub fn parse_cidr_list(s: &str) -> Result<Vec<Cidr>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(Cidr::parse)
        .collect()
}

/// Normalize an IPv4-mapped IPv6 address (`::ffff:127.0.0.1`) to its
/// underlying IPv4 form so loopback / CIDR checks behave consistently
/// regardless of the listening socket's address family.
fn normalize(ip: IpAddr) -> IpAddr {
    if let IpAddr::V6(v6) = ip
        && let Some(v4) = v6.to_ipv4_mapped()
    {
        return IpAddr::V4(v4);
    }
    ip
}

/// Shortest token accepted from `--token-file`. A short secret in front of a
/// public tunnel is brute-forceable; the deploy scripts generate 64 hex
/// characters.
const MIN_TOKEN_LEN: usize = 16;

/// SHA-256 of the bearer token the server requires.
///
/// The plaintext token is hashed at construction and never stored, so it
/// cannot surface in a log line, a `Debug` dump, or a core file. `Debug`
/// redacts the digest as well: it is preimage-resistant, but there is no
/// reason to print it either.
#[derive(Clone, Copy)]
pub struct TokenDigest([u8; 32]);

impl TokenDigest {
    /// Hash a token read from `--token-file`.
    ///
    /// # Errors
    /// Returns an error if the token is shorter than [`MIN_TOKEN_LEN`] bytes.
    pub fn new(token: &str) -> Result<Self, String> {
        let len = token.len();
        if len < MIN_TOKEN_LEN {
            return Err(format!(
                "token is {len} bytes; at least {MIN_TOKEN_LEN} required"
            ));
        }
        Ok(Self(Sha256::digest(token.as_bytes()).into()))
    }

    /// Whether `presented` is the token this digest was built from.
    ///
    /// Both sides are hashed and the digests compared without an early exit,
    /// so neither the expected token's length nor a shared prefix is
    /// observable in the response time.
    fn matches(&self, presented: &[u8]) -> bool {
        let presented: [u8; 32] = Sha256::digest(presented).into();
        let mut diff = 0u8;
        for (expected, actual) in self.0.iter().zip(presented.iter()) {
            diff |= expected ^ actual;
        }
        diff == 0
    }
}

impl fmt::Debug for TokenDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TokenDigest(<redacted>)")
    }
}

/// The credential from an `Authorization: Bearer <token>` header.
///
/// The scheme is matched case-insensitively (RFC 9110 §11.1); the credential
/// itself is returned verbatim and compared byte-exactly. Exactly one space
/// may separate the two, as the grammar requires.
fn bearer_credential(headers: &HeaderMap) -> Option<&[u8]> {
    const SCHEME: &[u8] = b"Bearer";

    let value = headers.get(header::AUTHORIZATION)?.as_bytes();
    let (scheme, rest) = value.split_at_checked(SCHEME.len())?;
    if !scheme.eq_ignore_ascii_case(SCHEME) {
        return None;
    }
    let credential = rest.strip_prefix(b" ")?;
    (!credential.is_empty()).then_some(credential)
}

/// Marker attached to requests allowed to see privileged diagnostic detail:
/// in-flight sample names, thread counts, orphan counts. Present when the
/// request carried a valid bearer token — or when authentication is disabled,
/// which leaves those responses exactly as they were before tokens existed.
#[derive(Clone, Copy, Debug)]
pub(super) struct Trusted;

/// Routes that may only be reached from loopback regardless of
/// `--allow-cidr`. `/analyze-path` accepts a server-side path and is gated
/// against `--allowed-dirs`, but exposing it to the network would still hand
/// any allowed peer a remote-read primitive against those directories. Keep
/// it loopback-only.
const LOOPBACK_ONLY_ROUTES: &[&str] = &["/analyze-path"];

/// The one route reachable without a bearer token, so that load balancers,
/// tunnel health checks, and monitoring can probe liveness without holding a
/// credential. A valid token still upgrades the response to the full
/// diagnostic body; see [`Trusted`].
pub(super) const HEALTH_ROUTE: &str = "/_/health";

fn forbidden(message: &str, auth: Auth) -> Response {
    tag(
        (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "error": message })),
        )
            .into_response(),
        auth,
    )
}

/// Record the access decision on the response for the access log to report.
fn tag(mut response: Response, auth: Auth) -> Response {
    response.extensions_mut().insert(auth);
    response
}

/// 401 for a missing *or* invalid token. The two are deliberately
/// indistinguishable, so the endpoint cannot be used as an oracle for whether
/// a guessed token exists.
fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        Json(serde_json::json!({ "error": "missing or invalid bearer token" })),
    )
        .into_response()
}

/// Per-request access-control middleware: peer-IP ACL, then bearer token.
/// Installed via [`axum::middleware::from_fn_with_state`] in
/// [`super::build_app`], outside the body limit, so a rejected request never
/// gets to upload bytes.
pub(super) async fn acl(
    State(state): State<Arc<AppState>>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    let path = req.uri().path();
    let loopback_only = LOOPBACK_ONLY_ROUTES.contains(&path);
    let auth_exempt = path == HEALTH_ROUTE;

    let peer = req
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|c| normalize(c.0.ip()));

    // ConnectInfo is installed by `into_make_service_with_connect_info` in
    // `run`. Tests inject it manually. If it's missing here in production
    // that's a wiring bug — fail closed rather than allow.
    let Some(ip) = peer else {
        return forbidden("peer address unavailable", Auth::NoPeerInfo);
    };

    // Gate 1: peer IP. Loopback passes; anything else needs an --allow-cidr
    // match and can never reach a loopback-only route.
    if !ip.is_loopback() {
        if loopback_only {
            return forbidden(
                "this route requires a loopback connection",
                Auth::LoopbackOnly,
            );
        }
        if !state.allow_cidrs.iter().any(|c| c.contains(ip)) {
            return forbidden("peer address not in any allow-cidr", Auth::PeerDenied);
        }
    }

    // Gate 2: bearer token. Loopback is not exempt — see the module docs.
    let auth = match state.auth_digest {
        None => Auth::Open,
        Some(digest) if bearer_credential(req.headers()).is_some_and(|c| digest.matches(c)) => {
            Auth::Token
        }
        Some(_) if auth_exempt => Auth::Anonymous,
        Some(_) => return tag(unauthorized(), Auth::BadToken),
    };
    if auth != Auth::Anonymous {
        req.extensions_mut().insert(Trusted);
    }

    tag(next.run(req).await, auth)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // tests intentionally unwrap parse results to assert success
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn parses_v4_cidr() {
        let c = Cidr::parse("10.0.0.0/8").unwrap();
        assert!(c.contains(ip("10.0.0.1")));
        assert!(c.contains(ip("10.255.255.255")));
        assert!(!c.contains(ip("11.0.0.1")));
        assert!(!c.contains(ip("127.0.0.1")));
    }

    #[test]
    fn parses_v4_host() {
        let c = Cidr::parse("192.168.1.5/32").unwrap();
        assert!(c.contains(ip("192.168.1.5")));
        assert!(!c.contains(ip("192.168.1.6")));
    }

    #[test]
    fn parses_zero_prefix() {
        let c = Cidr::parse("0.0.0.0/0").unwrap();
        assert!(c.contains(ip("1.2.3.4")));
        assert!(c.contains(ip("255.255.255.255")));
    }

    #[test]
    fn rejects_bad_prefix() {
        assert!(Cidr::parse("10.0.0.0/33").is_err());
        assert!(Cidr::parse("10.0.0.0").is_err());
        assert!(Cidr::parse("nope/8").is_err());
        assert!(Cidr::parse("::1/129").is_err());
    }

    #[test]
    fn parses_v6() {
        let c = Cidr::parse("fd00::/8").unwrap();
        assert!(c.contains(ip("fd00::1")));
        assert!(c.contains(ip("fdff::ffff")));
        assert!(!c.contains(ip("fe00::1")));
    }

    #[test]
    fn cross_family_does_not_match() {
        let v4 = Cidr::parse("10.0.0.0/8").unwrap();
        assert!(!v4.contains(ip("::1")));
        let v6 = Cidr::parse("fd00::/8").unwrap();
        assert!(!v6.contains(ip("10.0.0.1")));
    }

    #[test]
    fn parses_list() {
        let list = parse_cidr_list("10.0.0.0/8, 192.168.0.0/16").unwrap();
        assert_eq!(list.len(), 2);
        assert!(list[0].contains(ip("10.1.2.3")));
        assert!(list[1].contains(ip("192.168.5.5")));
    }

    #[test]
    fn parses_empty_list() {
        let list = parse_cidr_list("").unwrap();
        assert!(list.is_empty());
        let list = parse_cidr_list("  ").unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn list_propagates_error() {
        assert!(parse_cidr_list("10.0.0.0/8,bogus").is_err());
    }

    fn auth_header(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, value.parse().unwrap());
        headers
    }

    const TOKEN: &str = "0123456789abcdef0123";

    #[test]
    fn token_digest_matches_only_the_exact_token() {
        let digest = TokenDigest::new(TOKEN).unwrap();
        assert!(digest.matches(TOKEN.as_bytes()));
        assert!(!digest.matches(b"0123456789abcdef012"), "prefix");
        assert!(!digest.matches(b"0123456789abcdef01234"), "extension");
        assert!(!digest.matches(b"0123456789ABCDEF0123"), "case differs");
        assert!(!digest.matches(b""));
    }

    #[test]
    fn token_digest_rejects_short_tokens() {
        assert!(TokenDigest::new("").is_err());
        assert!(TokenDigest::new(&"a".repeat(MIN_TOKEN_LEN - 1)).is_err());
        assert!(TokenDigest::new(&"a".repeat(MIN_TOKEN_LEN)).is_ok());
    }

    /// The token must not be recoverable from a log line or a crash dump.
    #[test]
    fn token_digest_debug_redacts() {
        let rendered = format!("{:?}", TokenDigest::new(TOKEN).unwrap());
        assert_eq!(rendered, "TokenDigest(<redacted>)");
        assert!(!rendered.contains(TOKEN));
    }

    #[test]
    fn parses_bearer_credential() {
        assert_eq!(
            bearer_credential(&auth_header("Bearer abc123")),
            Some(&b"abc123"[..])
        );
        // RFC 9110 §11.1: the scheme is case-insensitive.
        assert_eq!(
            bearer_credential(&auth_header("bEaReR abc123")),
            Some(&b"abc123"[..])
        );
    }

    #[test]
    fn rejects_malformed_authorization() {
        assert!(bearer_credential(&HeaderMap::new()).is_none(), "absent");
        assert!(bearer_credential(&auth_header("Bearer")).is_none(), "bare");
        assert!(
            bearer_credential(&auth_header("Bearer ")).is_none(),
            "empty"
        );
        assert!(bearer_credential(&auth_header("Basic abc123")).is_none());
        assert!(bearer_credential(&auth_header("Bearerabc123")).is_none());
        assert!(bearer_credential(&auth_header("abc123")).is_none());
        // A second space belongs to the credential, not the separator, and so
        // will not match any token the deploy scripts generate.
        assert_eq!(
            bearer_credential(&auth_header("Bearer  abc123")),
            Some(&b" abc123"[..])
        );
    }

    #[test]
    fn normalize_ipv4_mapped() {
        let mapped = ip("::ffff:127.0.0.1");
        assert!(normalize(mapped).is_loopback());
        assert!(normalize(ip("::1")).is_loopback());
        assert!(normalize(ip("127.0.0.1")).is_loopback());
    }
}
