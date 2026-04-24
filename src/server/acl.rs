//! Per-route IP ACL for the litmus HTTP API.
//!
//! Loopback connections are allowed unconditionally. The `/analyze-path`
//! endpoint can read any file under `--allowed-dirs` and is therefore
//! restricted to loopback at the transport layer regardless of `--allow-cidr`
//! — exposing it to the network would let any allowed peer turn the server
//! into a remote file-read primitive. All other routes accept loopback OR
//! any peer matching one of the configured `--allow-cidr` networks.
//!
//! CIDR matching is hand-rolled (no extra crate dependency) and supports
//! both IPv4 and IPv6.

use std::net::IpAddr;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Json, Response};

use super::AppState;

/// A single CIDR network parsed from `--allow-cidr`.
#[derive(Debug, Clone, Copy)]
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
    if let IpAddr::V6(v6) = ip {
        if let Some(v4) = v6.to_ipv4_mapped() {
            return IpAddr::V4(v4);
        }
    }
    ip
}

/// Routes that may only be reached from loopback regardless of
/// `--allow-cidr`. `/analyze-path` accepts a server-side path and is gated
/// against `--allowed-dirs`, but exposing it to the network would still hand
/// any allowed peer a remote-read primitive against those directories. Keep
/// it loopback-only.
const LOOPBACK_ONLY_ROUTES: &[&str] = &["/analyze-path"];

fn forbidden(message: &str) -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(serde_json::json!({ "error": message })),
    )
        .into_response()
}

/// Per-request peer-IP ACL middleware. Installed via
/// [`axum::middleware::from_fn_with_state`] in [`super::build_app`].
pub(super) async fn acl(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let path = req.uri().path();
    let loopback_only = LOOPBACK_ONLY_ROUTES.contains(&path);

    let peer = req
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|c| normalize(c.0.ip()));

    // ConnectInfo is installed by `into_make_service_with_connect_info` in
    // `run`. Tests inject it manually. If it's missing here in production
    // that's a wiring bug — fail closed rather than allow.
    let Some(ip) = peer else {
        tracing::error!(path, "ACL: peer ConnectInfo missing — refusing");
        return forbidden("peer address unavailable");
    };

    if ip.is_loopback() {
        return next.run(req).await;
    }

    if loopback_only {
        tracing::warn!(%ip, path, "ACL: rejected non-loopback access to loopback-only route");
        return forbidden("this route requires a loopback connection");
    }

    if state.allow_cidrs.iter().any(|c| c.contains(ip)) {
        return next.run(req).await;
    }

    tracing::warn!(%ip, path, "ACL: rejected unauthorized peer");
    forbidden("peer address not in any allow-cidr")
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

    #[test]
    fn normalize_ipv4_mapped() {
        let mapped = ip("::ffff:127.0.0.1");
        assert!(normalize(mapped).is_loopback());
        assert!(normalize(ip("::1")).is_loopback());
        assert!(normalize(ip("127.0.0.1")).is_loopback());
    }
}
