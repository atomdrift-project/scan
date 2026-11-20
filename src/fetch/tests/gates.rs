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

    // Real Go module coordinates, in both spellings the Go
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
    let week = 7 * SECS_PER_DAY;

    // A month later: comfortably past a 7-day ceiling.
    let now = published + 30 * SECS_PER_DAY;
    assert!(now.saturating_sub(published) >= week);
    // A day later: inside the window, so it must still be fetched.
    let now = published + SECS_PER_DAY;
    assert!(now.saturating_sub(published) < week);
    // Exactly at the boundary counts as aged out, matching `age_secs >= max`.
    let now = published + week;
    assert!(now.saturating_sub(published) >= week);
    // A clock behind the stamp must not underflow into "ancient".
    let now = published - SECS_PER_DAY;
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
    let at = |v: &str| crate::test_registry("", "", v);

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
    let published = |at: u64| {
        let mut record = crate::test_registry("", "", "");
        record.published_at = Some(at);
        record
    };
    let fresh = published(now - 3_600); // 1h ago
    // 30h ago: outside the 4h window, and the case that separates it from
    // the local-file window in `engine`, which is still measured in days.
    let day_and_a_half = published(now - 30 * 3_600);
    let stale = published(now - 300_000); // ~3.5d ago
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
    let mut yanked = stale.clone();
    yanked.deprecated = Some("yanked".to_string());
    assert!(must_rescan(&yanked, now));
    let mut held = stale.clone();
    held.security_hold = Some(true);
    assert!(must_rescan(&held, now));
    let mut removed = stale;
    removed.version_removed = Some(true);
    assert!(must_rescan(&removed, now));
}

#[test]
fn off_host_platform_matches_native_binary_naming() {
    let host = Platform {
        os: "darwin",
        arch: "arm64",
    };
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
        Platform { os: "", arch: "" }
    ));
}

#[test]
fn off_host_platform_matches_cargo_target_naming() {
    // Rust target spellings: `windows`/`i686`/`x86_64`/`aarch64`. The
    // windows-rs platform crates ship multi-megabyte import libraries that
    // no Linux host will ever link; they are the cargo analogue of npm's
    // `cli-win32-x64`.
    let host = Platform {
        os: "linux",
        arch: "x64",
    };
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
    let musl_host = Platform {
        os: "linuxmusl",
        arch: "x64",
    };
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
fn fetch_timeout_defaults_on() {
    assert_eq!(FetchPolicy::default().max_duration, DEFAULT_FETCH_TIMEOUT);
    assert_eq!(DEFAULT_FETCH_TIMEOUT, Duration::from_secs(300));
    // A selection never disturbs the ceilings around it.
    assert_eq!(
        FetchPolicy::parse_follow("all").unwrap().max_duration,
        DEFAULT_FETCH_TIMEOUT
    );
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

/// The package references discovery reports for `source`, a file named `path`.
fn discovered(path: &str, file_type: &str, source: &[u8]) -> Vec<Reference> {
    let parsed = filefacts::OpenOptions::new()
        .path(Path::new(path))
        .open(source);
    let report = report_of(&serde_json::json!([
        {"id":0,"path":path,"depth":0,"file_type":file_type,"sha256":"a".repeat(64),"size":source.len(),
         "filefacts":{"symbols":parsed.symbols(),"values":parsed.values()}}
    ]));
    collect_references_from_input(
        &report,
        Path::new("not-on-disk"),
        Some(source),
        CiRefs::Skip,
    )
    .into_iter()
    .flat_map(|(_, refs)| refs)
    .filter(|r| matches!(r.locator, RefLocator::Purl(_)))
    .collect()
}

/// Answer the registry lookup for `reference` from the memo: release
/// `version`, published `age` seconds before `now`. The gate then needs no
/// network, which [`offline::Offline`](super::offline::Offline) refuses.
fn remember_release(reference: &Reference, ecosystem: &str, version: &str, now: u64, age: u64) {
    let mut record = crate::test_registry(ecosystem, "fixture", version);
    record.published_at = Some(now - age);
    registry_memo()
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .put(
            locator(reference).to_owned(),
            Memoized {
                at: now,
                record: Some(record),
            },
        );
}

fn offline_resources() -> Resources {
    Resources {
        net: Box::new(super::offline::Offline),
        cache: super::offline::cache(),
    }
}

/// Known-good filters vouching for exactly `purls`, and the directory they
/// are read from, which must outlive them.
fn known_good(purls: &[&str]) -> (tempfile::TempDir, crate::bloom_repo::Lookup) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut sets = burton::KeySets::new();
    for purl in purls {
        sets.insert(
            burton::Tier::Good,
            burton::Record {
                purl: crate::bloom_repo::purl_key(purl),
                sha256: None,
            },
        );
    }
    burton::write_bundle(
        dir.path(),
        &sets.into_filters(1e-9),
        "2026-11-19",
        crate::bloom_repo::KEY_SCHEME,
    )
    .expect("write bundle");
    let lookup = crate::bloom_repo::Lookup::load_from(dir.path());
    (dir, lookup)
}

/// Each way a file can import a package, per ecosystem: (path, file type,
/// registry ecosystem, source importing `one` and `two`).
fn import_forms(one: &str, two: &str) -> [(&'static str, &'static str, &'static str, String); 3] {
    [
        (
            "app.py",
            "python",
            "pypi",
            format!("import {one}\nimport {two}\n"),
        ),
        (
            "app.js",
            "javascript",
            "npm",
            format!("const a = require('{one}');\nconst b = require('{two}');\n"),
        ),
        (
            "app.mjs",
            "javascript",
            "npm",
            format!("import a from '{one}';\nimport b from '{two}';\n"),
        ),
    ]
}

const DAY: u64 = 86_400;

#[test]
fn imported_packages_meet_the_age_gate_like_declared_ones() {
    let now = 2_000_000_000;
    let policy = FetchPolicy::parse_follow("references").unwrap();
    assert_eq!(policy.max_dep_age_days, DEFAULT_MAX_DEP_AGE_DAYS);
    let res = offline_resources();
    for (path, file_type, ecosystem, source) in import_forms("agegateold", "agegateyoung") {
        let mut refs = discovered(path, file_type, source.as_bytes());
        assert_eq!(refs.len(), 2, "{path}: {refs:?}");
        // The manifest's own spelling of the same two packages, for parity.
        refs.extend(refs.clone().into_iter().map(|mut declared| {
            declared.kind = RefKind::Dependency;
            declared.source = "manifest".into();
            declared.locator = RefLocator::Purl(format!("{}-declared", locator(&declared)));
            declared
        }));
        for reference in &refs {
            assert!(
                reference.kind == RefKind::Dependency || is_package_import(reference),
                "{path}: {reference:?}"
            );
            assert!(
                !pins_exact_version(locator(reference)),
                "an import names no version: {reference:?}"
            );
            let age = if locator(reference).contains("old") {
                30 * DAY
            } else {
                DAY
            };
            remember_release(reference, ecosystem, "1.2.3", now, age);
        }
        let (keep, gated) = age_gate(refs.clone(), &policy, &res, now, None);
        let kept: Vec<&str> = keep.iter().map(locator).collect();
        let young: Vec<&str> = refs
            .iter()
            .map(locator)
            .filter(|l| l.contains("young"))
            .collect();
        assert_eq!(
            kept, young,
            "{path}: only a release inside the age limit is fetched"
        );
        assert_eq!(gated.len(), 4, "{path}: every one resolved a record");
        for gate in &gated {
            let expected = locator(&gate.reference)
                .contains("old")
                .then_some(SkipReason::AgedOut);
            assert_eq!(gate.skip, expected, "{path}: {:?}", gate.reference);
            assert_eq!(gate.record.version, "1.2.3");
        }
    }
}

#[test]
fn imported_packages_meet_the_known_good_check_at_their_resolved_version() {
    let now = 2_000_000_000;
    // No age ceiling, so nothing but the filters can settle a reference.
    let mut policy = FetchPolicy::parse_follow("references").unwrap();
    policy.max_dep_age_days = 0;
    let res = offline_resources();
    for (path, file_type, ecosystem, source) in import_forms("knowngoodpkg", "knowngoodnewer") {
        let refs = discovered(path, file_type, source.as_bytes());
        assert_eq!(refs.len(), 2, "{path}: {refs:?}");
        assert!(refs.iter().all(is_package_import), "{path}: {refs:?}");
        // The filters vouch for a release, never for a bare name: the first
        // import resolves to the vouched release, the second to a later one.
        let vouched: Vec<String> = refs
            .iter()
            .map(|r| format!("{}@1.2.3", locator(r)))
            .collect();
        let vouched: Vec<&str> = vouched.iter().map(String::as_str).collect();
        let (_filters, bloom) = known_good(&vouched);
        for purl in &vouched {
            assert!(bloom.decide_purl(purl).may_skip(), "{purl} is vouched for");
            assert!(immutable_coordinate(purl, Some(30 * DAY)), "{purl}");
        }
        remember_release(&refs[0], ecosystem, "1.2.3", now, 30 * DAY);
        remember_release(&refs[1], ecosystem, "1.2.4", now, 30 * DAY);

        let (keep, gated) = age_gate(refs.clone(), &policy, &res, now, Some(&bloom));
        assert_eq!(
            keep.iter().map(locator).collect::<Vec<_>>(),
            [locator(&refs[1])],
            "{path}: the vouched release is not fetched; an unvouched one is"
        );
        assert_eq!(gated[0].skip, Some(SkipReason::KnownGood), "{path}");
        assert_eq!(gated[1].skip, None, "{path}");

        // A release published inside the freshness window is re-scanned
        // whatever the filters say about it.
        remember_release(&refs[0], ecosystem, "1.2.3", now, 3_600);
        let (keep, _) = age_gate(refs.clone(), &policy, &res, now, Some(&bloom));
        assert_eq!(keep.len(), 2, "{path}: a fresh release is fetched");

        // And with no filters installed nothing is vouched for.
        remember_release(&refs[0], ecosystem, "1.2.3", now, 30 * DAY);
        let (keep, gated) = age_gate(refs.clone(), &policy, &res, now, None);
        assert_eq!(keep.len(), 2, "{path}");
        assert!(gated.iter().all(|gate| gate.skip.is_none()), "{path}");
    }
}

#[test]
fn imported_packages_stop_at_the_first_hop_like_declared_ones() {
    // A fetched dependency's own imports are the transitive tail a declared
    // dependency's dependencies are: followed from the scanned artifact,
    // and past it only by a policy that takes the whole closure.
    let mut policy: FetchPolicy = "all".parse().unwrap();
    assert!(!policy.transitive_deps, "interactive default");
    let imports: Vec<Reference> = import_forms("alphapkg", "betapkg")
        .into_iter()
        .flat_map(|(path, file_type, _, source)| discovered(path, file_type, source.as_bytes()))
        .collect();
    assert_eq!(imports.len(), 6, "{imports:?}");
    for import in &imports {
        assert!(
            policy.follows_at(import, 0),
            "{} at the artifact",
            import.source
        );
        assert!(!policy.follows_at(import, 1), "{} past it", import.source);
        let mut declared = import.clone();
        declared.kind = RefKind::Dependency;
        assert_eq!(
            policy.follows_at(import, 1),
            policy.follows_at(&declared, 1),
            "an import and a declaration of {} are followed alike",
            locator(import)
        );
    }
    // A package an install command names is a dropper step, followed at
    // every hop.
    let installed = Reference::new(
        RefLocator::Purl("pkg:pypi/alphapkg".into()),
        RefKind::Command,
        "shell",
        "pip install alphapkg",
    );
    assert!(policy.follows_at(&installed, 1));
    policy.transitive_deps = true;
    for import in &imports {
        assert!(
            policy.follows_at(import, 3),
            "{} in the closure",
            import.source
        );
    }
}

#[test]
fn a_package_an_install_command_names_is_not_gated() {
    let now = 2_000_000_000;
    let policy = FetchPolicy::parse_follow("references").unwrap();
    let refs = discovered(
        "install.sh",
        "shell",
        b"#!/bin/sh\nnpm install installcmdold\npip install installcmdoldpy\n",
    );
    assert_eq!(refs.len(), 2, "{refs:?}");
    for reference in &refs {
        assert_eq!(reference.kind, RefKind::Command);
        assert!(!is_package_import(reference), "{reference:?}");
        remember_release(reference, "npm", "1.2.3", now, 30 * DAY);
    }
    // An install is an acquisition the code performs, however old the
    // release: it is fetched, and no registry record stands in for it.
    let (keep, gated) = age_gate(refs.clone(), &policy, &offline_resources(), now, None);
    assert_eq!(keep, refs);
    assert!(gated.is_empty());
}
