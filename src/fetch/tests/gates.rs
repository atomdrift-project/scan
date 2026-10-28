//! The follow policy, the host-platform filter, and the age gate.

use super::*;

#[test]
fn go_pseudo_versions_date_themselves_without_a_lookup() {
    // Fixed points spanning the civil-days arithmetic: the epoch itself, a
    // century/leap-rule boundary, a leap day, and both year edges. These
    // catch an off-by-one era or an early month roll, which self-consistent
    // round-tripping would not.
    for (stamp, want) in [
        ("19700101000000", 0),
        ("20000101000000", 946_684_800),   // 2000-01-01, leap-century
        ("20240229120000", 1_709_208_000), // leap day
        ("20251231235959", 1_767_225_599), // last second of a year
        ("20260101000000", 1_767_225_600), // first second of the next
    ] {
        assert_eq!(
            go_pseudo_version_published(&format!("pkg:golang/x/y@v0.0.0-{stamp}-abc")),
            Some(want),
            "{stamp}"
        );
    }

    // Real coordinates from the benchmark corpus, in both spellings the Go
    // proxy mints: the bare `v0.0.0-` form and the `vX.Y.Z-0.` form that
    // follows a release tag.
    assert_eq!(
        go_pseudo_version_published(
            "pkg:golang/github.com/deckhouse/deckhouse@v0.0.0-20260528132821-f66b8cdce5b3"
        ),
        Some(1_779_974_901),
    );
    assert_eq!(
        go_pseudo_version_published(
            "pkg:golang/cloud.google.com/go@v0.20.1-0.20260528200609-1134b3699ee5"
        ),
        Some(1_779_998_769),
    );
    // A module path containing digits and dots must not confuse the stamp
    // hunt, and neither must a 14-digit-looking commit hash prefix.
    assert_eq!(
        go_pseudo_version_published(
            "pkg:golang/gopkg.in/yaml.v3@v0.0.0-20260720151329-12345678901234"
        ),
        Some(1_784_560_409),
    );

    // Anything not confidently datable falls through to the registry rather
    // than being guessed at — a wrong age here silently skips a fetch.
    for undatable in [
        "pkg:golang/github.com/stretchr/testify@v1.9.0", // tagged release
        "pkg:golang/x/y@v0.0.0-2026052813282-abc",       // 13 digits
        "pkg:golang/x/y@v0.0.0-202605281328210-abc",     // 15 digits
        "pkg:golang/x/y@v0.0.0-20261328132821-abc",      // month 13
        "pkg:golang/x/y@v0.0.0-20260028132821-abc",      // month 0
        "pkg:golang/x/y@v0.0.0-20260532132821-abc",      // day 32
        "pkg:golang/x/y@v0.0.0-20260500132821-abc",      // day 0
        "pkg:golang/x/y@v0.0.0-20260528243821-abc",      // hour 24
        "pkg:golang/x/y@v0.0.0-20260528136021-abc",      // minute 60
        "pkg:golang/x/y",                                // no version at all
        "pkg:cargo/serde@1.0.219",                       // wrong ecosystem
        "pkg:npm/axios@1.6.8",                           // wrong ecosystem
        "not-a-purl",
    ] {
        assert_eq!(
            go_pseudo_version_published(undatable),
            None,
            "{undatable} must not be dated locally"
        );
    }
}

#[test]
fn go_pseudo_version_age_decides_the_gate_the_same_way_a_registry_would() {
    // The gate compares `now - published >= max_age`. Pin that the local date
    // drives the same decision the registry record would, on both sides of
    // the boundary — this is the behaviour, not just the parse.
    let published = go_pseudo_version_published(
        "pkg:golang/github.com/deckhouse/deckhouse@v0.0.0-20260528132821-f66b8cdce5b3",
    )
    .expect("datable");
    let week = 7 * 86_400;

    // A month later: comfortably past a 7-day ceiling.
    let now = published + 30 * 86_400;
    assert!(now.saturating_sub(published) >= week);
    // A day later: inside the window, so it must still be fetched.
    let now = published + 86_400;
    assert!(now.saturating_sub(published) < week);
    // Exactly at the boundary counts as aged out, matching `age_secs >= max`.
    let now = published + week;
    assert!(now.saturating_sub(published) >= week);
    // A clock behind the stamp must not underflow into "ancient".
    let now = published - 86_400;
    assert_eq!(now.saturating_sub(published), 0);
}

#[test]
fn resolved_purl_pairs_registry_version_onto_range_declarations() {
    let dep = |locator: &str| {
        Reference::new(
            RefLocator::Purl(locator.to_string()),
            RefKind::Dependency,
            "",
            "",
        )
    };
    let at = |v: &str| Registry {
        version: v.to_string(),
        ..Registry::default()
    };

    // The npm case this exists for: a manifest range leaves the locator
    // version-less, so the bloom (keyed `name@version`) could never match it.
    assert_eq!(
        resolved_purl(&dep("pkg:npm/axios"), &at("1.6.8")).as_deref(),
        Some("pkg:npm/axios@1.6.8")
    );
    // A scoped npm name percent-encodes its `@`, so the scope must not be
    // mistaken for a version already being present.
    assert_eq!(
        resolved_purl(&dep("pkg:npm/%40scope/pkg"), &at("2.0.0")).as_deref(),
        Some("pkg:npm/%40scope/pkg@2.0.0")
    );
    // Lockfile ecosystems already pin a version — returned untouched, and
    // notably NOT re-suffixed with the registry's version.
    assert_eq!(
        resolved_purl(&dep("pkg:cargo/serde@1.0.219"), &at("1.0.220")).as_deref(),
        Some("pkg:cargo/serde@1.0.219")
    );
    assert_eq!(
        resolved_purl(
            &dep("pkg:golang/github.com/stretchr/testify@v1.9.0"),
            &at("v1.9.1")
        )
        .as_deref(),
        Some("pkg:golang/github.com/stretchr/testify@v1.9.0")
    );
    // Nothing resolved, or not a PURL: no coordinate to probe.
    assert_eq!(resolved_purl(&dep("pkg:npm/axios"), &at("")), None);
    assert_eq!(
        resolved_purl(
            &Reference::new(
                RefLocator::Url("https://example.test/x.tgz".into()),
                RefKind::UrlFetch,
                "",
                "",
            ),
            &at("1.0.0")
        ),
        None
    );
}

#[test]
fn freshly_published_and_must_rescan_track_publish_age() {
    let now = 1_000_000_u64;
    let fresh = Registry {
        published_at: Some(now - 3_600), // 1h ago
        ..Registry::default()
    };
    // 30h ago: outside the 4h window, and the case that separates it from
    // the local-file window in `engine`, which is still measured in days.
    let day_and_a_half = Registry {
        published_at: Some(now - 30 * 3_600),
        ..Registry::default()
    };
    let stale = Registry {
        published_at: Some(now - 300_000), // ~3.5d ago
        ..Registry::default()
    };
    assert!(freshly_published(&fresh, now));
    assert!(!freshly_published(&day_and_a_half, now));
    assert!(!freshly_published(&stale, now));
    // A settled, unwithdrawn version needs no re-scan; a just-published one
    // does, and a withdrawn one always does regardless of age.
    assert!(!must_rescan(&stale, now));
    assert!(!must_rescan(&day_and_a_half, now));
    assert!(must_rescan(&fresh, now));
    // A yank leaves the artifact downloadable, so re-scanning it is not a
    // doomed fetch — this is the arm that earns `dep_pulled` its keep.
    assert!(must_rescan(
        &Registry {
            published_at: Some(now - 300_000),
            deprecated: Some("yanked".to_string()),
            ..Registry::default()
        },
        now
    ));
    assert!(must_rescan(
        &Registry {
            published_at: Some(now - 300_000),
            security_hold: Some(true),
            ..Registry::default()
        },
        now
    ));
    assert!(must_rescan(
        &Registry {
            published_at: Some(now - 300_000),
            version_removed: Some(true),
            ..Registry::default()
        },
        now
    ));
}

#[test]
fn off_host_platform_matches_native_binary_naming() {
    let host = ("darwin", "arm64");
    // The host variant is kept; other-platform siblings are skipped.
    assert!(!off_host_platform(
        &purl_ref("pkg:npm/%40biomejs/cli-darwin-arm64@2.5.0"),
        host
    ));
    assert!(off_host_platform(
        &purl_ref("pkg:npm/%40biomejs/cli-linux-x64-musl@2.5.0"),
        host
    ));
    assert!(off_host_platform(
        &purl_ref("pkg:npm/%40biomejs/cli-win32-arm64@2.5.0"),
        host
    ));
    // Same OS, wrong arch is still off-host.
    assert!(off_host_platform(
        &purl_ref("pkg:npm/%40biomejs/cli-darwin-x64@2.5.0"),
        host
    ));
    // `<arch>-<os>` order (esbuild-style) and other scopes.
    assert!(off_host_platform(
        &purl_ref("pkg:npm/%40esbuild/linux-x64@0.21.0"),
        host
    ));
    // A portable package (no os+arch pair) is never platform-skipped.
    assert!(!off_host_platform(
        &purl_ref("pkg:npm/left-pad@1.3.0"),
        host
    ));
    assert!(!off_host_platform(&purl_ref("pkg:npm/semver@7.5.0"), host));
    // A raw URL carries no package identity to place.
    assert!(!off_host_platform(
        &url_ref("https://example.com/x.tgz"),
        host
    ));
    // Fail open when the host platform can't be named.
    assert!(!off_host_platform(
        &purl_ref("pkg:npm/%40biomejs/cli-linux-x64@2.5.0"),
        ("", "")
    ));
}

#[test]
fn off_host_platform_matches_cargo_target_naming() {
    // Rust target spellings: `windows`/`i686`/`x86_64`/`aarch64`. The
    // windows-rs platform crates ship multi-megabyte import libraries that
    // no Linux host will ever link; they are the cargo analogue of npm's
    // `cli-win32-x64`.
    let host = ("linux", "x64");
    assert!(off_host_platform(
        &purl_ref("pkg:cargo/windows_i686_gnu@0.52.0"),
        host
    ));
    assert!(off_host_platform(
        &purl_ref("pkg:cargo/windows_x86_64_gnu@0.53.0"),
        host
    ));
    assert!(off_host_platform(
        &purl_ref("pkg:cargo/windows_x86_64_gnullvm@0.52.4"),
        host
    ));
    assert!(off_host_platform(
        &purl_ref("pkg:cargo/windows_aarch64_msvc@0.48.5"),
        host
    ));
    // A same-OS other-arch name still ages out on arch alone.
    assert!(off_host_platform(
        &purl_ref("pkg:npm/app-linux-aarch64@1.0.0"),
        host
    ));
    // The host's own spelling variants are kept.
    assert!(!off_host_platform(
        &purl_ref("pkg:npm/app-linux-x86_64@1.0.0"),
        host
    ));
    // musl variants are a different platform from a glibc host, wasm
    // sandbox builds never match a real host, and both directions of the
    // sharp/libvips naming are recognized.
    assert!(off_host_platform(
        &purl_ref("pkg:npm/@img/sharp-libvips-linuxmusl-x64@1.3.2"),
        host
    ));
    assert!(off_host_platform(
        &purl_ref("pkg:npm/@img/sharp-linuxmusl-arm64@0.35.3"),
        host
    ));
    assert!(off_host_platform(
        &purl_ref("pkg:npm/@img/sharp-freebsd-wasm32@0.35.3"),
        host
    ));
    assert!(off_host_platform(
        &purl_ref("pkg:npm/@img/sharp-webcontainers-wasm32@0.35.3"),
        host
    ));
    // ...while the host's real variant stays fetchable.
    assert!(!off_host_platform(
        &purl_ref("pkg:npm/@img/sharp-libvips-linux-x64@1.3.2"),
        host
    ));
    // A musl host keeps its own variants and skips the glibc one.
    let musl_host = ("linuxmusl", "x64");
    assert!(!off_host_platform(
        &purl_ref("pkg:npm/@img/sharp-linuxmusl-x64@0.35.3"),
        musl_host
    ));
    assert!(off_host_platform(
        &purl_ref("pkg:npm/@img/sharp-linux-x64@0.35.3"),
        musl_host
    ));
    // Portable cargo crates carry no os+arch pair.
    assert!(!off_host_platform(
        &purl_ref("pkg:cargo/windows@0.52.0"),
        host
    ));
    assert!(!off_host_platform(
        &purl_ref("pkg:cargo/windows-sys@0.52.0"),
        host
    ));
    assert!(!off_host_platform(&purl_ref("pkg:cargo/serde@1.0.0"), host));
}

#[test]
fn versioned_purl_splits_and_exempts() {
    assert_eq!(
        versioned_purl("pkg:cargo/syn@2.0.104"),
        Some(("pkg:cargo/syn", "2.0.104"))
    );
    // npm scoped names keep their leading @ in the key.
    assert_eq!(
        versioned_purl("pkg:npm/@babel/core@7.24.0"),
        Some(("pkg:npm/@babel/core", "7.24.0"))
    );
    // No version, or a non-numeric ref, exempts the reference.
    assert_eq!(versioned_purl("pkg:npm/@scope/name"), None);
    assert_eq!(versioned_purl("pkg:cargo/serde"), None);
    assert_eq!(versioned_purl("pkg:generic/x@deadbeef"), None);
}

#[test]
fn versionless_dep_superseded_only_when_coordinate_is_pinned() {
    // The lockfile pin for `puppeteer` is present in the tree.
    let pinned: HashSet<String> = ["pkg:npm/puppeteer".to_string()].into_iter().collect();

    // The manifest's version-stripped `pkg:npm/puppeteer` loses to the pin.
    assert!(superseded_by_pin(&purl_ref("pkg:npm/puppeteer"), &pinned));
    // The pin itself is kept — it is versioned, not a bare coordinate.
    assert!(!superseded_by_pin(
        &purl_ref("pkg:npm/puppeteer@10.4.2"),
        &pinned
    ));
    // A different, unpinned coordinate keeps its versionless fallback.
    assert!(!superseded_by_pin(&purl_ref("pkg:npm/left-pad"), &pinned));
    // A git/tag ref is not versionless and never equals a bare coordinate.
    assert!(!superseded_by_pin(
        &purl_ref("pkg:npm/puppeteer@dev"),
        &pinned
    ));
    // Non-PURL locators are out of scope.
    assert!(!superseded_by_pin(
        &url_ref("https://example.test/x.tgz"),
        &pinned
    ));

    // Scoped npm names: the bare scoped coordinate loses to its scoped pin.
    let scoped: HashSet<String> = ["pkg:npm/@puppeteer/browsers".to_string()]
        .into_iter()
        .collect();
    assert!(superseded_by_pin(
        &purl_ref("pkg:npm/@puppeteer/browsers"),
        &scoped
    ));
    assert!(!superseded_by_pin(
        &purl_ref("pkg:npm/@puppeteer/browsers@3.2.0"),
        &scoped
    ));
}

#[test]
fn lenient_version_cmp_orders_release_schemes() {
    use std::cmp::Ordering::*;
    let cmp = lenient_version_cmp;
    assert_eq!(cmp("2.0.104", "2.0.9"), Greater);
    assert_eq!(cmp("1.2", "1.2.1"), Less);
    assert_eq!(cmp("0.52.0", "0.48.5"), Greater);
    assert_eq!(cmp("1.0.0", "1.0.0"), Equal);
    // pep440-ish and date-like schemes still order sensibly.
    assert_eq!(cmp("0.1.5rc1", "0.1.4"), Greater);
    assert_eq!(cmp("20260528.18.2", "20260101.1.1"), Greater);
    // Numeric outranks a text suffix at the same position.
    assert_eq!(cmp("1.2.3", "1.2.rc1"), Greater);
}

/// The fetch phase carries a wall-clock cap by default, and the two
/// spellings that turn it off both reach the "no deadline" branch that
/// `orchestrate` tests for.
#[test]
fn fetch_timeout_defaults_on_and_has_two_off_switches() {
    assert_eq!(FetchPolicy::default().max_duration, DEFAULT_FETCH_TIMEOUT);
    assert_eq!(DEFAULT_FETCH_TIMEOUT, Duration::from_secs(300));
    // A selection never disturbs the ceilings around it.
    assert_eq!(
        FetchPolicy::parse_follow("all").unwrap().max_duration,
        DEFAULT_FETCH_TIMEOUT
    );
    assert!(parse_duration("0").unwrap().is_zero());
    assert!(
        Instant::now()
            .checked_add(parse_duration("never").unwrap())
            .is_none()
    );
    assert_eq!(parse_duration("5m").unwrap(), DEFAULT_FETCH_TIMEOUT);
}

#[test]
fn fetch_policy_parses_kinds_and_rejects_garbage() {
    assert_eq!(
        FetchPolicy::parse_follow("dependencies,references"),
        Ok(FetchPolicy {
            urls: true,
            packages: true,
            deps: true,
            ..FetchPolicy::default()
        })
    );
    let actions = FetchPolicy::parse_follow("ci-actions").unwrap();
    assert!(actions.ci && actions.deps);
    assert_eq!(
        FetchPolicy::parse_follow("none"),
        Ok(FetchPolicy::default())
    );
    assert!(FetchPolicy::parse_follow("deps").is_err());
    assert!(FetchPolicy::parse_follow("none,references").is_err());
}

/// `follow_name` inverts `parse_follow`. This is the property the header
/// rests on: a caller files an answer under the name we return, so a name
/// that does not parse back to the policy that produced it files the
/// verdict under a question nobody asked.
#[test]
fn follow_name_round_trips_through_parse_follow() {
    for spelling in [
        "none",
        "dependencies",
        "references",
        "dependencies,references",
        "all",
    ] {
        let policy = FetchPolicy::parse_follow(spelling).unwrap();
        let name = policy
            .follow_name()
            .expect("vocabulary spelling has a name");
        assert_eq!(name, spelling, "{spelling} did not round-trip");
        assert_eq!(
            FetchPolicy::parse_follow(&name).unwrap().selection_bits(),
            policy.selection_bits(),
            "{spelling} reparsed to a different selection",
        );
    }

    // `ci-actions` implies `dependencies`, so its canonical name says so
    // rather than echoing the shorthand back.
    let actions = FetchPolicy::parse_follow("ci-actions").unwrap();
    assert_eq!(
        actions.follow_name().as_deref(),
        Some("dependencies,ci-actions")
    );
    assert_eq!(
        FetchPolicy::parse_follow("dependencies,ci-actions")
            .unwrap()
            .selection_bits(),
        actions.selection_bits(),
    );

    // The full set is spelled `all`, not enumerated, so one policy has one
    // name.
    let every = FetchPolicy::parse_follow("dependencies,references,ci-actions").unwrap();
    assert_eq!(every.follow_name().as_deref(), Some("all"));

    // A legacy alias can set half of `references`, which the customer
    // vocabulary cannot spell. Unnameable is reported, never approximated.
    let half = FetchPolicy {
        urls: true,
        packages: false,
        ..FetchPolicy::default()
    };
    assert_eq!(half.follow_name(), None);

    // The retired CLI spellings are errors now, never silent aliases.
    for retired in ["deps", "packages", "urls", "ci", "urls,packages,deps"] {
        assert!(retired.parse::<FetchPolicy>().is_err(), "{retired}");
    }
    // `all` is shorthand for every kind, CI included.
    assert_eq!(
        "all".parse(),
        Ok(FetchPolicy {
            urls: true,
            packages: true,
            deps: true,
            ci: true,
            ..FetchPolicy::default()
        })
    );
    assert_eq!(
        "all".parse::<FetchPolicy>(),
        " dependencies , references , ci-actions ".parse()
    );
    // A routine dependency fetch leaves CI off — GitHub Actions run only in
    // CI and never reach an installed artifact; `ci-actions` (or `all`) opts in.
    assert!(!"dependencies".parse::<FetchPolicy>().unwrap().ci);
    assert!(!"references,dependencies".parse::<FetchPolicy>().unwrap().ci);
    // `ci-actions` turns on the CI context *and* dependency fetching, since a
    // CI action is a declared dependency.
    let ci = "ci-actions".parse::<FetchPolicy>().unwrap();
    assert!(ci.ci && ci.deps);
    // Parsing leaves depth at its default — the CLI sets it separately.
    assert_eq!(
        "dependencies".parse::<FetchPolicy>().unwrap().depth,
        DEFAULT_FETCH_DEPTH
    );
    assert_eq!(
        FetchPolicy::default().max_file_fetches,
        DEFAULT_MAX_FILE_FETCHES
    );
    assert_eq!(
        FetchPolicy::default().max_url_fetches,
        DEFAULT_MAX_URL_FETCHES
    );
    assert!("".parse::<FetchPolicy>().is_err());
    assert!("sigs".parse::<FetchPolicy>().is_err());
    assert!("refs".parse::<FetchPolicy>().is_err());
    assert!("dependencies,bogus".parse::<FetchPolicy>().is_err());
    assert!(!FetchPolicy::default().enabled());

    // Selection is by kind: `references` fetches command-mentioned packages
    // and URLs but not declared deps, and vice versa.
    let refs: FetchPolicy = "references".parse().unwrap();
    assert!(refs.wants(RefKind::Command));
    assert!(refs.wants(RefKind::UrlFetch));
    assert!(!refs.wants(RefKind::Dependency));
    let deps: FetchPolicy = "dependencies".parse().unwrap();
    assert!(deps.wants(RefKind::Dependency));
    assert!(!deps.wants(RefKind::Command));
    // Repository identity is never a fetch target.
    assert!(
        !"all"
            .parse::<FetchPolicy>()
            .unwrap()
            .wants(RefKind::Repository)
    );
}

#[test]
fn declared_deps_stop_at_the_first_hop_unless_transitive() {
    // Hop 0 is the artifact's own declared supply chain and is always
    // followed. Past it, declared dependencies are the transitive tail —
    // a registry lookup each, almost all of it aged out — so an interactive
    // policy drops them while the dropper kinds keep going.
    let mut policy: FetchPolicy = "all".parse().unwrap();
    assert!(!policy.transitive_deps, "interactive default");
    assert!(policy.wants_at(RefKind::Dependency, 0));
    assert!(!policy.wants_at(RefKind::Dependency, 1));
    for hop in 0..3 {
        assert!(policy.wants_at(RefKind::UrlFetch, hop), "hop {hop}");
        assert!(policy.wants_at(RefKind::Command, hop), "hop {hop}");
    }
    // A corpus-facing role takes the whole closure.
    policy.transitive_deps = true;
    assert!(policy.wants_at(RefKind::Dependency, 3));
    // The hop rule never *adds* a kind the selection left out.
    let refs: FetchPolicy = "references".parse().unwrap();
    assert!(!refs.wants_at(RefKind::Dependency, 0));
}
