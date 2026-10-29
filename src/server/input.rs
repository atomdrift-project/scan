//! Validation of the names, package URLs and URLs a request supplies.

/// The name an upload is staged and reported under.
///
/// Every character outside `[A-Za-z0-9_.-]` becomes `_`, `..` collapses to
/// `__`, and the result keeps its last 63 bytes so the extension survives —
/// cleave detects file type from it, and the name is also the `path` label in
/// the report and in every log line about this request.
///
/// The filter is deliberately ASCII-only rather than Unicode-aware. Two
/// reasons, both of them the client's choice to make otherwise: a
/// `char::is_alphanumeric` filter keeps multi-byte characters, and the
/// right-truncation below would then slice mid-character and panic the
/// request; and a name that reaches logs and a filesystem path should not
/// carry homoglyphs, combining marks, or bidi-shaped text.
pub(super) fn sanitize_upload_filename(raw: &str) -> String {
    let sanitized: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .replace("..", "__");
    // Every retained character is one ASCII byte, so this index is always a
    // character boundary.
    #[expect(
        clippy::string_slice,
        reason = "every retained character is one ASCII byte"
    )]
    if sanitized.len() > 63 {
        sanitized[sanitized.len() - 63..].to_string()
    } else {
        sanitized
    }
}

/// Canonical `pkg:…` form, or a 400 message. Same prefixing rule as `atomscan purl`.
pub(super) fn normalize_pkg_purl(raw: &str) -> Result<String, &'static str> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("missing purl");
    }
    let prefixed = if raw.starts_with("pkg:") {
        raw.to_string()
    } else {
        format!("pkg:{raw}")
    };
    fletch::purl::normalize(&prefixed).ok_or("not a package URL")
}

/// Whether `raw` is an absolute http(s) URL. Callers keep and log `raw`
/// itself, and `Url::parse` silently drops tab, CR and LF, so a raw control
/// character (a forged log line, a terminal escape) is refused outright; a
/// real URL percent-encodes them.
///
/// A URL carrying credentials (`https://user:pass@host/`) is refused too: it
/// would be logged, listed in `/_/requests` and stored in fetch provenance.
pub(super) fn valid_http_url(raw: &str) -> bool {
    !raw.chars().any(char::is_control)
        && reqwest::Url::parse(raw).is_ok_and(|url| {
            matches!(url.scheme(), "http" | "https")
                && url.username().is_empty()
                && url.password().is_none()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// logged: a raw control character must be refused, not normalized away.
    #[test]
    fn submitted_urls_refuse_raw_control_characters() {
        assert!(valid_http_url("https://a.example/x?q=%0A"));
        for raw in [
            "https://a.example/x\nINFO forged",
            "https://a.example/x\r\n",
            "https://a.example/\tx",
            "https://a.example/\u{1b}[2J",
            "file:///etc/passwd",
        ] {
            assert!(!valid_http_url(raw), "{raw:?}");
        }
    }

    /// A hostile upload filename must not be able to panic the request. A
    /// Unicode-aware filter kept multi-byte characters, and the tail
    /// truncation then sliced one in half.
    #[test]
    fn sanitize_upload_filename_survives_multibyte_names() {
        let raw = "\u{3041}".repeat(80) + ".zip";
        let name = sanitize_upload_filename(&raw);
        assert_eq!(name.len(), 63);
        assert!(name.is_ascii(), "{name}");
        assert!(name.ends_with(".zip"), "the extension must survive: {name}");
    }

    #[test]
    fn sanitize_upload_filename_defuses_paths_and_control_characters() {
        assert_eq!(
            sanitize_upload_filename("../../etc/shadow"),
            "______etc_shadow"
        );
        assert_eq!(
            sanitize_upload_filename("a\nb\r\u{202e}gpj.exe"),
            "a_b__gpj.exe"
        );
        // A name already inside the alphabet is left exactly as it is.
        assert_eq!(
            sanitize_upload_filename("left-pad-1.3.0.tgz"),
            "left-pad-1.3.0.tgz"
        );
    }

    #[test]
    fn normalize_pkg_purl_accepts_bare_and_full() {
        assert_eq!(
            normalize_pkg_purl("pkg:npm/left-pad@1.3.0").unwrap(),
            "pkg:npm/left-pad@1.3.0"
        );
        assert_eq!(
            normalize_pkg_purl("npm/left-pad@1.3.0").unwrap(),
            "pkg:npm/left-pad@1.3.0"
        );
        assert!(normalize_pkg_purl("").is_err());
        assert!(normalize_pkg_purl("not a purl").is_err());
    }
}
