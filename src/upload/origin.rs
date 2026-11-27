//! Who an upload is on behalf of.
//!
//! Hopper files every `/v1/intake/upload` under the source `upload`, and
//! records who asked for it as `upload:<origin>`. A sample usually reaches
//! hopper through relays — poppy asks beamline, beamline asks a scan worker,
//! the worker uploads — so hopper's own view of the caller is only the last
//! hop's User-Agent. The first atomdrift service a request reaches names the
//! caller in [`ORIGIN_HEADER`]; each relay passes it on unchanged, and the
//! uploader sends it with every artifact the request causes it to file.

use serde::{Deserialize, Serialize};

/// The request header carrying an [`Origin`], on the inbound analyze request
/// and on the outbound `/v1/intake/upload`.
pub const ORIGIN_HEADER: &str = "Atomdrift-Origin";

/// The longest origin, in characters.
const MAX_LEN: usize = 32;

/// A client name: 1 to 32 characters of `[a-z0-9-]`, which also makes it a
/// valid header value. Built only through [`Origin::new`], so a value read back
/// from the upload spool holds the same invariant as one built from a request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Origin(String);

impl Origin {
    /// `name` with everything outside `[A-Za-z0-9-]` dropped, lowercased, and
    /// cut to 32 characters. `None` when nothing is left.
    #[must_use]
    pub fn new(name: &str) -> Option<Self> {
        let name: String = name
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
            .map(|c| c.to_ascii_lowercase())
            .take(MAX_LEN)
            .collect();
        (!name.is_empty()).then_some(Self(name))
    }

    /// The origin of a request carrying `header` (its [`ORIGIN_HEADER`]) and
    /// `user_agent`. A named origin is believed, sanitized: it comes from a
    /// relay that already decided. Otherwise the caller is the first hop, and
    /// is named by its User-Agent's product token — the text before the first
    /// `/` or space — except that every browser is `website`. `None` when
    /// neither names anything.
    ///
    /// Beamline applies the same rule (`originOf` in beamline.js) before the
    /// request reaches a worker; the two must agree.
    #[must_use]
    pub fn from_request(header: Option<&str>, user_agent: Option<&str>) -> Option<Self> {
        header
            .and_then(Self::new)
            .or_else(|| user_agent.and_then(Self::from_user_agent))
    }

    /// [`Self::from_request`] for a request arriving with `headers`. A value
    /// that is not visible ASCII names nothing.
    #[must_use]
    pub fn from_headers(headers: &reqwest::header::HeaderMap) -> Option<Self> {
        let header = |name| headers.get(name).and_then(|v| v.to_str().ok());
        Self::from_request(
            header(ORIGIN_HEADER),
            header(reqwest::header::USER_AGENT.as_str()),
        )
    }

    fn from_user_agent(user_agent: &str) -> Option<Self> {
        let user_agent = user_agent.trim();
        // Every browser opens with this token, whichever it is.
        if user_agent
            .get(..8)
            .is_some_and(|token| token.eq_ignore_ascii_case("mozilla/"))
        {
            return Some(Self("website".to_owned()));
        }
        let product = user_agent
            .split(|c: char| c == '/' || c.is_whitespace())
            .next()
            .unwrap_or_default();
        Self::new(product)
    }

    /// The name, as sent and as hopper records it.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Origin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for Origin {
    type Error = &'static str;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        Self::new(&name).ok_or("an origin names no client")
    }
}

impl From<Origin> for String {
    fn from(origin: Origin) -> Self {
        origin.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(header: Option<&str>, user_agent: Option<&str>) -> Option<String> {
        Origin::from_request(header, user_agent).map(String::from)
    }

    #[test]
    fn a_first_hop_is_named_by_its_user_agent() {
        let poppy = Some("poppy/1 (+https://github.com/atomdrift-project/poppy)");
        assert_eq!(named(None, poppy).as_deref(), Some("poppy"));
        assert_eq!(named(None, Some("Poppy/0.3.1")).as_deref(), Some("poppy"));
        assert_eq!(named(None, Some("beamline/1")).as_deref(), Some("beamline"));
        assert_eq!(named(None, Some("postdoc")).as_deref(), Some("postdoc"));
        assert_eq!(
            named(None, Some("my-tool (+https://example.test)")).as_deref(),
            Some("my-tool"),
            "a comment is not part of the product"
        );
    }

    #[test]
    fn every_browser_is_the_website() {
        let safari = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15";
        assert_eq!(named(None, Some(safari)).as_deref(), Some("website"));
        assert_eq!(named(None, Some("mozilla/4.0")).as_deref(), Some("website"));
    }

    #[test]
    fn a_relayed_origin_is_passed_through() {
        assert_eq!(
            named(Some("poppy"), Some("beamline/1")).as_deref(),
            Some("poppy")
        );
        assert_eq!(
            named(Some("Website"), Some("beamline/1")).as_deref(),
            Some("website")
        );
    }

    #[test]
    fn junk_is_sanitized_and_bounded() {
        assert_eq!(named(Some("Po_ppy!é"), None).as_deref(), Some("poppy"));
        assert_eq!(
            named(Some(&"a".repeat(40)), None),
            Some("a".repeat(MAX_LEN))
        );
        assert_eq!(
            named(None, Some(&format!("{}/1", "b".repeat(40)))),
            Some("b".repeat(MAX_LEN))
        );
        assert_eq!(
            named(Some("!!!"), Some("curl/8.4.0")).as_deref(),
            Some("curl"),
            "a header naming nothing is no header"
        );
    }

    #[test]
    fn nothing_to_name_is_no_origin() {
        assert_eq!(named(None, None), None);
        assert_eq!(named(Some(""), Some("")), None);
        assert_eq!(named(Some("  "), Some("/1.0")), None);
        assert_eq!(named(None, Some("___/1")), None);
    }

    /// Header names are case-insensitive on the wire, and a value that is not
    /// text is not a name.
    #[test]
    fn a_request_is_read_off_its_headers() {
        use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
        let mut headers = HeaderMap::new();
        assert_eq!(Origin::from_headers(&headers), None);
        headers.insert(USER_AGENT, HeaderValue::from_static("beamline/1"));
        assert_eq!(
            Origin::from_headers(&headers).map(String::from).as_deref(),
            Some("beamline")
        );
        headers.insert("atomdrift-origin", HeaderValue::from_static("Poppy"));
        assert_eq!(
            Origin::from_headers(&headers).map(String::from).as_deref(),
            Some("poppy")
        );
        let opaque = HeaderValue::from_bytes(b"\xffpoppy").expect("opaque value");
        headers.insert("atomdrift-origin", opaque);
        assert_eq!(
            Origin::from_headers(&headers).map(String::from).as_deref(),
            Some("beamline"),
            "an unreadable header falls back to the User-Agent"
        );
    }

    /// The spool holds an origin as its bare name, and reading one back holds
    /// it to the same rule as a header.
    #[test]
    fn an_origin_serializes_as_its_name() {
        let origin = Origin::new("poppy").expect("valid");
        assert_eq!(
            serde_json::to_string(&origin).expect("serialize"),
            r#""poppy""#
        );
        assert_eq!(
            serde_json::from_str::<Origin>(r#""poppy""#).expect("deserialize"),
            origin
        );
        assert!(serde_json::from_str::<Origin>(r#""!!""#).is_err());
    }
}
