//! Which discovered URLs are payloads worth fetching, how to ask for them as
//! the client that would have, and where an HTML redirect page points.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// URL suffixes that are normally pages, API responses, or other site
/// resources rather than a payload a dropper would retrieve.
pub(super) const NON_PAYLOAD_URL_EXTENSIONS: &[&str] = &[
    "asp",
    "aspx",
    "atom",
    "avif",
    "bmp",
    "cfm",
    "cgi",
    "css",
    "csv",
    "dtd",
    "gif",
    "htm",
    "html",
    "ico",
    "jpeg",
    "jpg",
    "json",
    "log",
    "md",
    "pdf",
    "php",
    "png",
    "rss",
    "svg",
    "txt",
    "webmanifest",
    "webp",
    "xml",
    "xhtml",
    "yaml",
    "yml",
];

/// Path components that make a URL explicitly file/download-shaped. These
/// allow extensionless payload names and file routes on otherwise API-shaped
/// hosts, while a bare `/download` still fails the basename check.
pub(super) const DOWNLOAD_URL_PATH_COMPONENTS: &[&str] = &[
    "archive",
    "archives",
    "attachment",
    "attachments",
    "blob",
    "download",
    "downloads",
    "file",
    "files",
    "raw",
    "release",
    "releases",
    "resolve",
];

/// Path components that are strong signs of an API or service endpoint.
pub(super) const API_URL_PATH_COMPONENTS: &[&str] = &[
    "api", "graphql", "health", "lookup", "metrics", "oauth", "query", "rpc", "search", "status",
    "token",
];

/// Whether a path component reads as a version rather than a filename: every
/// dot-separated segment is digits, with at least one dot and an optional
/// leading `v` (`0.40.0`, `v2.1`, `10.0.1`). A bare `v1` or a plain number has
/// no dot and keeps whatever the surrounding rules decide.
///
/// Deliberately strict about the tail. Recognizing a pre-release suffix as part
/// of the version means splitting at `-`, which throws away everything after —
/// including a real extension. A Go module's
/// `v0.0.0-20260823143148-1fb3b878e2fb.zip` then reads as version `0.0.0` and
/// the artifact stops being fetched. Requiring every segment to be numeric can
/// only ever miss a version, never swallow a file: anything ending in an
/// alphabetic extension fails the test by construction.
pub(super) fn is_version_shaped(component: &str) -> bool {
    let core = component.strip_prefix(['v', 'V']).unwrap_or(component);
    core.contains('.')
        && core
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

/// Whether a discovered URL has a real network host. URL extraction also sees
/// relative paths, malformed authority strings, and single-label local names
/// such as `wpad`; none can identify a public download host. Public IP literals
/// of either family are valid, while DNS names must have at least two labels.
pub(super) fn valid_discovered_url_host(url: &Url) -> bool {
    if !matches!(url.scheme(), "http" | "https") {
        return false;
    }
    let host = match hosts::url_host(url) {
        Some(UrlHost::Ip(ip)) => return public_ip(ip),
        Some(UrlHost::Domain(host)) => host,
        None => return false,
    };
    let host = host.strip_suffix('.').unwrap_or(host);
    host.len() <= 253
        && host.contains('.')
        && host.split('.').all(|label| {
            let bytes = label.as_bytes();
            !bytes.is_empty()
                && bytes.len() <= 63
                && bytes[0].is_ascii_alphanumeric()
                && bytes[bytes.len() - 1].is_ascii_alphanumeric()
                && bytes
                    .iter()
                    .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'-')
        })
}

/// Whether a discovered URL still contains a source-template placeholder.
/// These appear in documentation and client code as repository and release
/// examples; fetching one can only produce an avoidable 4xx.
///
/// Three spellings, and the argument for the last two is the same: RFC 3986
/// admits neither a bare brace nor a `%` outside a two-hex-digit escape, so an
/// unencoded one is never a literal a server could serve — it is a slot some
/// renderer was supposed to fill. Recognizing only the shell `$` form let a
/// build script's `.../download/v{version}/{}` through to five certain 404s.
pub(super) fn has_unexpanded_url_placeholder(url: &str) -> bool {
    // `$VERSION`, `${this.repositoryId}`
    let shell = url.as_bytes().windows(2).any(|pair| {
        pair[0] == b'$' && (pair[1] == b'{' || pair[1].is_ascii_alphabetic() || pair[1] == b'_')
    });
    // `{version}`, `{}`
    let brace = url.contains(['{', '}']);
    // `%s`, `%d`, Python's `%(version)s`
    let printf = url.split('%').skip(1).any(|after| {
        !after
            .as_bytes()
            .get(..2)
            .is_some_and(|escape| escape.iter().all(u8::is_ascii_hexdigit))
    });
    shell || brace || printf
}

/// Whether an IP literal is publicly routable enough to justify a discovered
/// fetch. This excludes RFC1918/private space and the other special-use ranges
/// that describe the scanner's host, a lab network, or documentation rather
/// than an external payload service.
pub(super) fn public_ip(ip: std::net::IpAddr) -> bool {
    // Only an IPv4-*mapped* address is an IPv4 host in disguise; the
    // deprecated IPv4-compatible form would turn `::1` into `0.0.0.1`.
    if let std::net::IpAddr::V6(ipv6) = ip
        && let Some(ipv4) = ipv6.to_ipv4_mapped()
    {
        return public_ip(std::net::IpAddr::V4(ipv4));
    }
    match ip {
        std::net::IpAddr::V4(ip) => {
            let octets = ip.octets();
            !ip.is_private()
                && !ip.is_loopback()
                && !ip.is_link_local()
                && !ip.is_unspecified()
                && !ip.is_broadcast()
                && !ip.is_multicast()
                && !(octets[0] == 100 && (64..=127).contains(&octets[1]))
                && !(octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
                && !(octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
                && !(octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
                && !(octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
                && !(octets[0] == 198 && (18..=19).contains(&octets[1]))
                && !(octets[0] == 192 && octets[1] == 88 && octets[2] == 99)
                && octets[0] < 224
        }
        std::net::IpAddr::V6(ip) => {
            let segments = ip.segments();
            !ip.is_loopback()
                && !ip.is_unspecified()
                && !ip.is_multicast()
                && (segments[0] & 0xfe00) != 0xfc00
                && (segments[0] & 0xffc0) != 0xfe80
                && (segments[0] & 0xffc0) != 0xfec0
                && !(segments[0] == 0x2001 && segments[1] == 0x0db8)
        }
    }
}

/// Whether a discovered URL looks enough like a dropper download to spend a
/// network request on it.
///
/// This is deliberately a shape check, not a content or reputation check:
/// direct scans still fetch exactly what the operator names, and a URL with a
/// plausible payload basename remains eligible even when its host is unknown.
pub(super) fn looks_like_dropper_download_url(url: &Url) -> bool {
    if !matches!(url.scheme(), "http" | "https") {
        return false;
    }
    let path = url.path();
    // A path ending in `/` names a directory, not a file: whatever a server
    // returns for it is an index or a landing page, never the download itself.
    // `https://pypi.org/project/diffusers/0.40.0/` was being fetched as a
    // payload because the trailing component parsed as a filename.
    if path.ends_with('/') {
        return false;
    }
    let components: Vec<&str> = path
        .split('/')
        .filter(|component| !component.is_empty())
        .collect();
    let Some(filename) = components.last().copied() else {
        return false;
    };
    if filename == "." || filename == ".." {
        return false;
    }
    // A version is not a file. `0.40.0`, `v2.1`, `1.2.3-rc1` all end in what
    // looks like an extension, so the dotted-basename test reads them as
    // downloads and pulls project pages, release-tag pages, and API version
    // roots. Nothing named this way is an artifact.
    if is_version_shaped(filename) {
        return false;
    }

    let filename_lower = filename.to_ascii_lowercase();
    let has_dot = filename_lower.contains('.');
    let extension = filename_lower
        .rsplit_once('.')
        .and_then(|(stem, extension)| {
            (!stem.is_empty() && !extension.is_empty()).then_some(extension)
        });
    let has_payload_extension = extension.is_some_and(|extension| {
        extension.len() <= 12
            && extension.bytes().all(|byte| byte.is_ascii_alphanumeric())
            && !NON_PAYLOAD_URL_EXTENSIONS.contains(&extension)
    });
    // A download route names the file that follows it, so the route word
    // cannot also be the basename: `/repos/o/r/releases` is the listing API,
    // not the asset, and unauthenticated it only ever answers 403.
    if DOWNLOAD_URL_PATH_COMPONENTS.contains(&filename_lower.as_str()) {
        return false;
    }
    let explicit_download_path = components.iter().any(|component| {
        DOWNLOAD_URL_PATH_COMPONENTS
            .iter()
            .any(|route| component.eq_ignore_ascii_case(route))
    });
    let extensionless_download = !has_dot && explicit_download_path && components.len() >= 2;

    // A basename with a plausible extension is enough; an extensionless name
    // needs an explicit file/download route and at least one component before
    // the basename (`/download` itself is still just an endpoint).
    if !has_payload_extension && !extensionless_download {
        return false;
    }

    // An API host or endpoint with a file-shaped response is still allowed
    // when the URL says it is fetching a file. This keeps routes such as
    // `/releases/download/...`, `/raw/...`, and Telegram-style `/file/...`
    // eligible while dropping `/api/v1/models`, `/graphql`, and similar
    // service calls (which already fail the basename test in most cases).
    let api_host = url
        .host_str()
        .and_then(|host| host.split('.').next())
        .is_some_and(|label| label.eq_ignore_ascii_case("api"));
    let api_path = components.iter().any(|component| {
        API_URL_PATH_COMPONENTS
            .iter()
            .any(|route| component.eq_ignore_ascii_case(route))
    });
    !(api_host || api_path) || explicit_download_path
}

/// A URL fetched directly into `IEX`/`Invoke-Expression` is a dropper edge even
/// when its final path component is opaque (`/abc123`) or extensionless. The
/// response is executable by construction, and these one-line fetch/evaluate
/// forms are uncommon enough that following them is more useful than applying
/// the ordinary download-shape filter. The evidence is the recognizer's full
/// command line, so an unrelated URL elsewhere in a file does not qualify.
pub(super) fn is_eval_pipeline_url(reference: &Reference) -> bool {
    if reference.kind != RefKind::UrlFetch {
        return false;
    }
    if !reference.evidence.contains('|') {
        return false;
    }

    let command_matches = |candidates: &[&str]| {
        reference.evidence.split_whitespace().any(|word| {
            let word = word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-');
            let lowered = word.to_ascii_lowercase();
            let normalized = lowered
                .strip_suffix(".exe")
                .unwrap_or(&lowered)
                .replace('-', "");
            candidates.iter().any(|candidate| normalized == *candidate)
        })
    };

    let fetch = command_matches(&[
        "irm",
        "iwr",
        "curl",
        "wget",
        "invokerestmethod",
        "invokewebrequest",
    ]);
    let execution_sink = command_matches(&[
        "iex",
        "invokeexpression",
        "sh",
        "bash",
        "zsh",
        "ash",
        "dash",
        "cmd",
        "powershell",
        "pwsh",
    ]);
    fetch && execution_sink
}

/// A URL a fetch command saves to a named file (`curl -o f URL`, `wget -O f
/// URL`, `iwr URL -OutFile f`) is a staged-download edge even when its path is
/// an opaque API route: the command states the response is an artifact to keep,
/// which is exactly the download-then-execute dropper shape. `wget -qO-` and
/// `-o -` write to stdout and do not qualify; a pipe into a shell is
/// [`is_eval_pipeline_url`]'s case. Flags are only read between a fetch command
/// and the end of its pipeline stage, so `unzip -o` after `&&` does not count.
pub(super) fn is_download_to_file_url(reference: &Reference) -> bool {
    if reference.kind != RefKind::UrlFetch {
        return false;
    }
    let mut words = reference
        .evidence
        .split_whitespace()
        .map(|word| word.trim_matches(['"', '\'', '^']))
        .peekable();
    let mut in_fetch = false;
    while let Some(word) = words.next() {
        let lowered = word.to_ascii_lowercase();
        let command = lowered.strip_suffix(".exe").unwrap_or(&lowered);
        if matches!(
            command,
            "curl" | "wget" | "iwr" | "invoke-webrequest" | "invoke-restmethod" | "irm"
        ) {
            in_fetch = true;
            continue;
        }
        if matches!(word, "|" | "||" | "&&" | "&") || word.ends_with(';') {
            in_fetch = false;
            continue;
        }
        if !in_fetch {
            continue;
        }
        // The value an output flag names: attached (`-ofile`, `--output=f`,
        // `-qO-`) or the following word.
        let value = if let Some(long) = lowered.strip_prefix("--") {
            match long.split_once('=') {
                Some(("output" | "output-document", value)) => Some(value.to_string()),
                None if matches!(long, "output" | "output-document") => None,
                _ => continue,
            }
        } else if lowered == "-outfile" {
            None
        } else if let Some(cluster) = word.strip_prefix('-') {
            // A short-flag cluster such as `-sLo` or `-qO-`: find the output
            // flag among leading letters; what follows it is its value.
            let Some((_, flag_and_value)) = cluster
                .find(['o', 'O'])
                .and_then(|at| cluster.split_at_checked(at))
                .filter(|(flags, _)| flags.bytes().all(|b| b.is_ascii_alphabetic()))
            else {
                continue;
            };
            let attached = flag_and_value.get(1..).unwrap_or_default();
            (!attached.is_empty()).then(|| attached.to_string())
        } else {
            continue;
        };
        let value = value.or_else(|| words.peek().map(|next| (*next).to_string()));
        if value.is_some_and(|value| !value.is_empty() && value != "-") {
            return true;
        }
    }
    false
}

/// User-Agents of the command-line clients a staged download names. Pinned to
/// current releases; a staging server gates on the product token, not the
/// exact version.
pub(super) const CURL_USER_AGENT: &str = "curl/8.7.1";
pub(super) const WGET_USER_AGENT: &str = "Wget/1.21.4";
pub(super) const POWERSHELL_USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT; Windows NT 10.0; en-US) WindowsPowerShell/5.1.19041.4648";

/// The User-Agent the command that names a URL reference would send.
///
/// Staging servers routinely answer only the client their one-liner uses: a
/// C2 behind `curl -L … | sh` returns 403 to a browser or a scanner that
/// announces itself, so fetching as `fletch` retrieves nothing. A redirect
/// destination continues a download chain and is fetched as curl, the client
/// such chains overwhelmingly use. `None` (no fetch command in the evidence)
/// keeps fletch's own agent.
pub(super) fn client_user_agent(reference: &Reference) -> Option<&'static str> {
    if reference.kind != RefKind::UrlFetch {
        return None;
    }
    if is_redirect_destination(reference) {
        return Some(CURL_USER_AGENT);
    }
    reference.evidence.split_whitespace().find_map(|word| {
        let word = word
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-')
            .to_ascii_lowercase();
        match word.strip_suffix(".exe").unwrap_or(&word) {
            "curl" => Some(CURL_USER_AGENT),
            "wget" => Some(WGET_USER_AGENT),
            "iwr" | "irm" | "invoke-webrequest" | "invoke-restmethod" => {
                Some(POWERSHELL_USER_AGENT)
            }
            _ => None,
        }
    })
}

/// A fetch backend that requests each URL with the User-Agent of the client
/// that names it (see [`client_user_agent`]), keyed by locator. Only a plain
/// GET is dressed so; every other request — one carrying headers of its own,
/// one asking for any status, a POST — passes through unchanged.
pub(super) struct AsClient<'a> {
    pub(super) net: &'a dyn Fetch,
    pub(super) agents: HashMap<String, &'static str>,
}

impl Fetch for AsClient<'_> {
    fn send(&self, request: &Request<'_>) -> Result<Fetched, FetchError> {
        let plain =
            request.method == Method::Get && request.headers.is_empty() && !request.any_status;
        match self.agents.get(request.url) {
            Some(&agent) if plain => {
                let headers = [("User-Agent", agent)];
                let mut request = *request;
                request.headers = &headers;
                self.net.send(&request)
            }
            _ => self.net.send(request),
        }
    }

    fn allows_oci(&self) -> bool {
        self.net.allows_oci()
    }
}

/// [`Reference::source`] for a URL that a fetched redirect page names as its
/// destination (see [`redirect_destinations`]). A tag in a text field, because
/// next-hop references are fletch's type and ride the analysis cache as such;
/// [`is_redirect_destination`] is the one place that reads it.
pub(super) const REDIRECT_DESTINATION_SOURCE: &str = "redirect-destination";

/// Whether a fetched redirect page named this reference as its destination.
pub(super) fn is_redirect_destination(r: &Reference) -> bool {
    r.source == REDIRECT_DESTINATION_SOURCE
}

/// Redirect pages larger than this are real sites, not an interstitial.
pub(super) const REDIRECT_PAGE_MAX_BYTES: usize = 1024 * 1024;

/// Most redirect pages a chain may pass through before depth accounting
/// resumes; bounds the extra hops [`orchestrate`]'s redirect credit can grant.
pub(super) const MAX_REDIRECT_HOPS: u8 = 3;

/// Destinations an HTML redirect page sends its visitor to: a
/// `<meta http-equiv=refresh content="0; url=…">`, a URL-shortener
/// interstitial's hidden destination field (`<input type=hidden id=long_url
/// value=…>`), or a script's `location = "…"` / `location.replace("…")`.
///
/// A dropper URL behind a shortener (`curl -L https://short.link/x | sh`)
/// answers with such a page whenever the shortener shows a preview or safety
/// warning instead of a 3xx. The page is not the payload; the destination is.
/// Only absolute http(s) URLs are returned.
pub(super) fn redirect_destinations(bytes: &[u8]) -> Vec<String> {
    if bytes.len() > REDIRECT_PAGE_MAX_BYTES {
        return Vec::new();
    }
    let html = String::from_utf8_lossy(bytes);
    let lower = html.to_ascii_lowercase();
    // An HTML document opens with markup; a script that merely mentions a
    // `<meta` tag in a comment or heredoc is not a redirect page.
    let head = lower
        .get(..lower.floor_char_boundary(1024))
        .unwrap_or_default();
    if !head
        .trim_start_matches(['\u{feff}', ' ', '\t', '\r', '\n'])
        .starts_with('<')
        || !["<!doctype html", "<html", "<head", "<meta"]
            .iter()
            .any(|marker| head.contains(marker))
    {
        return Vec::new();
    }
    let (html, lower): (&str, &str) = (&html, &lower);
    // A match inside an earlier same-name tag's body is attribute text, not a
    // tag, and once no `>` remains no later tag can close. Skipping both keeps the scan
    // linear: a hostile page of `<input <input <input …` with one `>` at the
    // end would otherwise re-scan and copy the whole tail per match.
    let tags = |name: &'static str| {
        let mut consumed = 0;
        lower.match_indices(name).filter_map(move |(at, _)| {
            if at < consumed {
                return None;
            }
            let Some(close) = html.get(at..).and_then(|rest| rest.find('>')) else {
                consumed = html.len();
                return None;
            };
            let end = at + close;
            consumed = end;
            html.get(at + name.len()..end)
                .filter(|body| body.starts_with(|c: char| c.is_ascii_whitespace()))
        })
    };

    let mut found = Vec::new();
    for tag in tags("<meta") {
        if html_attr(tag, "http-equiv").is_some_and(|v| v.eq_ignore_ascii_case("refresh"))
            && let Some(content) = html_attr(tag, "content")
            && let Some(at) = content.to_ascii_lowercase().find("url=")
            && let Some(url) = content.get(at + "url=".len()..)
        {
            found.push(url.trim_matches(|c: char| c.is_whitespace() || c == '\''));
        }
    }
    for tag in tags("<input") {
        let names_url = ["id", "name"].iter().any(|attr| {
            html_attr(tag, attr).is_some_and(|v| v.to_ascii_lowercase().contains("url"))
        });
        if names_url
            && html_attr(tag, "type").is_some_and(|v| v.eq_ignore_ascii_case("hidden"))
            && let Some(value) = html_attr(tag, "value")
        {
            found.push(value);
        }
    }
    for (at, keyword) in html.match_indices("location") {
        let Some(rest) = html.get(at + keyword.len()..) else {
            continue;
        };
        let rest = rest.strip_prefix(".href").unwrap_or(rest).trim_start();
        let rest = match rest.strip_prefix('=') {
            Some(assigned) if !assigned.starts_with('=') => assigned,
            Some(_) => continue,
            None => match rest
                .strip_prefix(".replace(")
                .or_else(|| rest.strip_prefix(".assign("))
            {
                Some(argument) => argument,
                None => continue,
            },
        }
        .trim_start();
        let Some(quote @ ('"' | '\'' | '`')) = rest.chars().next() else {
            continue;
        };
        // `rest` opens with the quote, so the literal is the second piece.
        if let Some(value) = rest.split(quote).nth(1) {
            found.push(value);
        }
    }

    // First-seen order, deduplicated through a set: a hostile page can name
    // thousands of destinations, and a linear `contains` makes that quadratic.
    let mut seen = std::collections::HashSet::new();
    let mut urls: Vec<String> = Vec::new();
    for url in found {
        let url = url.replace("\\/", "/").replace("&amp;", "&");
        let lowered = url.to_ascii_lowercase();
        if (lowered.starts_with("https://") || lowered.starts_with("http://"))
            && seen.insert(url.clone())
        {
            urls.push(url);
        }
    }
    urls
}

/// The value of attribute `name` in the body of one HTML start tag, quoted or
/// bare. Attribute names match case-insensitively.
pub(super) fn html_attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    // ASCII lowercasing keeps every byte offset, so `lower` indexes `tag`.
    let lower = tag.to_ascii_lowercase();
    let mut from = 0;
    while let Some(found) = lower.get(from..)?.find(name) {
        let start = from + found;
        from = start + name.len();
        if !lower
            .get(..start)
            .is_some_and(|before| before.ends_with(|c: char| c.is_ascii_whitespace()))
        {
            continue;
        }
        let Some(value) = tag.get(from..)?.trim_start().strip_prefix('=') else {
            continue;
        };
        let value = value.trim_start();
        return match value.chars().next() {
            Some(quote @ ('"' | '\'')) => value.split(quote).nth(1),
            _ => value.split(|c: char| c.is_ascii_whitespace()).next(),
        };
    }
    None
}

/// Mark the references a fetched redirect page names as its destination, so
/// the fetch gate follows them as the continuation of the edge that reached
/// the page. A destination the byte hunt missed (a script-escaped `https:\/\/`)
/// is added.
pub(super) fn mark_redirect_destinations(bytes: &[u8], refs: &mut Vec<Reference>) {
    let destinations = redirect_destinations(bytes);
    if destinations.is_empty() {
        return;
    }
    // Indexed once: a hostile page can name thousands of destinations, and a
    // search of `refs` per destination would be quadratic in them.
    let mut by_url: HashMap<String, usize> = HashMap::new();
    for (i, r) in refs.iter().enumerate() {
        if let RefLocator::Url(u) = &r.locator {
            by_url.entry(u.clone()).or_insert(i);
        }
    }
    tracing::info!(
        destinations = destinations.len(),
        "fetched page is a redirect; following its destination"
    );
    for url in destinations {
        tracing::debug!(url = ?url, "redirect destination");
        match by_url.get(&url) {
            Some(&i) => {
                let reference = &mut refs[i];
                reference.kind = RefKind::UrlFetch;
                REDIRECT_DESTINATION_SOURCE.clone_into(&mut reference.source);
            }
            None => refs.push(Reference::new(
                RefLocator::Url(url.clone()),
                RefKind::UrlFetch,
                REDIRECT_DESTINATION_SOURCE,
                url,
            )),
        }
    }
}
