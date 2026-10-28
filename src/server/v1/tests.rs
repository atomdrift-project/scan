use super::*;
use crate::model::Level;
use crate::server::decision::V1Finding;

#[tokio::test]
async fn versioned_purl_uses_corpus_even_with_a_clean_local_alias() -> anyhow::Result<()> {
    use super::super::ServerConfig;
    use crate::lookup::Verdict;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let purl = "pkg:pypi/critical-release-choice@1.0.0";
    let clean_sha = "c1".repeat(32);
    let critical_sha = "d2".repeat(32);
    let index = crate::lookup::global().expect("verdict index enabled");
    index.replace(&Verdict {
        sha256: clean_sha.clone(),
        lvl: Level::Clean,
        eng: "local".into(),
        at: "2026-01-02T00:00:00Z".into(),
        purl: Some(purl.into()),
        why: None,
        hits: Vec::new(),
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let payload = serde_json::json!({
        "sha256": critical_sha, "purl": purl, "fires_at": 0,
        "engine_version": "corpus", "analyzed_at": "2026-01-01T00:00:00Z", "findings": []
    });
    let corpus = axum::Router::new().route(
        "/v1/lookup",
        axum::routing::get(move || {
            counted.fetch_add(1, Ordering::SeqCst);
            let payload = payload.clone();
            async move { axum::Json(payload) }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let task = tokio::spawn(async move { axum::serve(listener, corpus).await });
    let (_router, state) = super::super::assemble(ServerConfig {
        hopper: Some(format!("http://{address}")),
        ..ServerConfig::default()
    })?;
    let key = Key::parse(None, Some(purl), None).expect("valid purl");
    assert!(
        held(index, &key, true).is_none(),
        "one payload cannot stand for its release"
    );
    assert!(
        held(index, &key, false).is_some(),
        "standalone workers retain local lookups"
    );
    let (single, _) = decide(&state, &key, 25, false).await;
    assert_eq!(single.fires_at, Level::At(0));
    assert_eq!(single.sha256.as_deref(), Some(critical_sha.as_str()));
    let batch = decide_many(&state, vec![key.clone(), key], 25, false).await;
    for (decision, _) in batch {
        assert_eq!(decision.fires_at, Level::At(0));
        assert_eq!(decision.sha256.as_deref(), Some(critical_sha.as_str()));
    }
    let before = calls.load(Ordering::SeqCst);
    assert!(
        before >= 3,
        "single and batch lookups must reach the corpus"
    );
    let exact = Key::parse(Some(&clean_sha), Some(purl), None).expect("valid keys");
    let (decision, _) = decide(&state, &exact, 25, false).await;
    assert_eq!(decision.fires_at, Level::Clean);
    assert_eq!(decision.sha256.as_deref(), Some(clean_sha.as_str()));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        before,
        "digest hits stay artifact-specific"
    );
    task.abort();
    Ok(())
}

#[test]
fn only_versioned_purl_only_keys_require_corpus_selection() {
    for (sha, purl, expected) in [
        (None, Some("pkg:pypi/litellm@1.107.0"), true),
        (None, Some("pkg:npm/%40scope/name@1.0"), true),
        (None, Some("pkg:pypi/litellm"), false),
        (None, Some("pkg:npm/%40scope/name"), false),
        (
            Some("a".repeat(64)),
            Some("pkg:pypi/litellm@1.107.0"),
            false,
        ),
    ] {
        let key = Key::parse(sha.as_deref(), purl, None).expect("valid keys");
        assert_eq!(key.needs_corpus(true), expected, "{key:?}");
        assert!(!key.needs_corpus(false));
    }
}

/// A pending analysis must send progress before it has an outcome. Use a
/// paused clock and an unpublished flight so neither a fast refusal nor
/// an installed verdict can stand in for the streaming path.
#[tokio::test]
async fn a_pending_analysis_streams_progress_before_its_decision() -> anyhow::Result<()> {
    use anyhow::Context as _;
    use futures_core::Stream as _;
    use std::future::poll_fn;
    use std::pin::Pin;

    let purl = "pkg:npm/stream-fixture@1.0.0";
    let flights = Arc::new(super::super::flight::Flights::default());
    let attachment = flights.join(FlightKey::from(Artifact::Purl(purl.into())));
    let publisher = flights.publisher(attachment.flight());
    let request = V1Analyze {
        request_id: 1,
        started: Instant::now(),
        q: V1Query::parse(None),
        budget: 25,
        follow: RequestFollow {
            policy: crate::fetch::FetchPolicy::default(),
            refresh: false,
        },
        purl: Some((purl.into(), purl.into())),
        url: None,
        path: None,
    };
    let named = Named {
        key: Some(purl.into()),
        asked: Some(purl.into()),
        subject: purl.into(),
        is_url: false,
    };
    let response =
        tokio::time::timeout(Duration::from_secs(1), answer(&request, attachment, named))
            .await
            .context("a pending analysis did not send headers within one second")?;
    assert_eq!(response.headers()["content-type"], "application/x-ndjson");
    assert_eq!(response.headers()["X-Scan-Follow"], "none");

    let mut body = response.into_body().into_data_stream();
    let progress = tokio::time::timeout(
        Duration::from_secs(5),
        poll_fn(|cx| Pin::new(&mut body).poll_next(cx)),
    )
    .await
    .context("a pending analysis did not send its first progress frame")?
    .context("stream ended before its first progress frame")??;
    let progress: serde_json::Value = serde_json::from_slice(&progress)?;
    assert_eq!(progress["state"], "analyzing");
    assert_eq!(progress["purl"], purl);
    assert!(progress.get("decision").is_none());

    publisher.publish(Outcome::Failed(ApiError::init_failed()));
    let decision = tokio::time::timeout(
        Duration::from_secs(1),
        poll_fn(|cx| Pin::new(&mut body).poll_next(cx)),
    )
    .await
    .context("stream did not send the completed decision")?
    .context("stream ended without a decision")??;
    let decision: serde_json::Value = serde_json::from_slice(&decision)?;
    assert_eq!(decision["decision"], "unavailable");
    assert!(
        tokio::time::timeout(
            Duration::from_secs(1),
            poll_fn(|cx| Pin::new(&mut body).poll_next(cx)),
        )
        .await
        .context("stream did not end after its decision")?
        .is_none()
    );
    Ok(())
}

/// A finding says only what is known about it.
///
/// The decision object around it keeps every key at all times; a finding
/// does not, because how much is known about one varies by where it came
/// from. A corpus finding knows the trait and its criticality and nothing
/// else — sending four nulls to say so is most of the object.
#[test]
fn a_finding_omits_what_it_does_not_know() {
    use crate::hopper::Finding;
    let corpus = V1Finding::from_corpus(&Finding {
        id: "intel/feed/malicious".into(),
        crit: 5,
        desc: Some("Cited as malicious by 3 independent sources.".into()),
    });
    let json = serde_json::to_value(&corpus).expect("serialize");
    let obj = json.as_object().expect("object");
    assert!(obj.contains_key("id"), "id is the finding's identity");
    assert!(obj.contains_key("crit"), "crit is always known");
    assert!(obj.contains_key("desc"), "a desc that exists must be sent");
    for absent in ["file", "pkg", "off", "line"] {
        assert!(
            !obj.contains_key(absent),
            "{absent} is unknown here and must be omitted, not null"
        );
    }
    // Never null: absent is how "nothing to say" is spelled in a finding.
    assert!(
        obj.values().all(|v| !v.is_null()),
        "a null survived into a finding: {json}"
    );
}

/// What may answer `/v1/analyze` without spending a slot.
///
/// The rule is "anything that actually answers the question", which is
/// wider than "a measurement of ours" — an operator's call, made because a
/// fast answer from what we already know beats rediscovering it. What stays
/// excluded is what was always excluded: `unanalyzed`, which is the very thing
/// the route exists to fix, and `unavailable`, which is a statement about us
/// rather than about the artifact.
#[test]
fn only_a_real_answer_may_replace_an_analysis() {
    use super::decision::Decision;
    let purl = Some("pkg:npm/left-pad@1.3.0");
    assert!(!V1Decision::unanalyzed(None, purl).is_answerable());
    assert!(!V1Decision::unavailable(None, purl).is_answerable());

    // A decision with no level answers nothing, whatever it is labelled.
    assert!(!V1Decision::empty(Decision::Block, None, purl).is_answerable());

    let measured = |d| {
        let mut v = V1Decision::empty(d, None, purl);
        v.engine_version = Some("2.8.0".into());
        v.fires_at = Level::At(10);
        v
    };
    assert!(measured(Decision::Allow).is_answerable());
    assert!(measured(Decision::Block).is_answerable());
}

/// A derived answer may answer, but must never claim to be a measurement:
/// the absent engine is what stops it being cached as one downstream, and
/// what `?fresh=1` exists to get past.
#[test]
fn a_derived_answer_answers_without_claiming_an_engine() {
    let purl = Some("pkg:npm/left-pad@1.3.0");
    for d in [
        crate::bloom_repo::Decision::Skip,
        crate::bloom_repo::Decision::SightedHostile,
        crate::bloom_repo::Decision::SightedSuspicious,
        crate::bloom_repo::Decision::KnownBad,
    ] {
        let derived = V1Decision::bloom(d, None, purl, 25).expect("answerable");
        assert!(derived.is_answerable(), "{d:?}");
        assert!(!derived.is_verdict(), "{d:?} is not a measurement");
        assert!(derived.engine_version.is_none(), "{d:?}");
        assert!(derived.analyzed_at.is_none(), "{d:?}");
    }
    // No filter had an opinion: nothing to answer with.
    assert!(V1Decision::bloom(crate::bloom_repo::Decision::Unknown, None, purl, 25).is_none());
}

/// `fresh` is opt-in on the same affirmative-only terms as `force`, and is
/// a separate question from it: `force` spends a slot, `fresh` chooses
/// which layer may answer. A caller can want either without the other.
#[test]
fn only_an_affirmative_fresh_bypasses_the_filters() {
    let q = |raw| V1Query::parse(Some(raw));
    assert!(q("purl=pkg:npm/left-pad@1.3.0&fresh=1").fresh);
    assert!(q("purl=pkg:npm/left-pad@1.3.0&fresh=true").fresh);
    assert!(q("purl=pkg:npm/left-pad@1.3.0&fresh=yes").fresh);
    assert!(!q("purl=pkg:npm/left-pad@1.3.0&fresh=0").fresh);
    assert!(!q("purl=pkg:npm/left-pad@1.3.0&fresh=").fresh);
    assert!(!q("purl=pkg:npm/left-pad@1.3.0").fresh);

    // Independent of `force`, in both directions.
    let both = q("purl=pkg:npm/left-pad@1.3.0&fresh=1&force=0");
    assert!(both.fresh && !both.force);
    let other = q("purl=pkg:npm/left-pad@1.3.0&fresh=0&force=1");
    assert!(!other.fresh && other.force);
}

/// The header spelling is an alias, not an override: either opts in, and a
/// proxy that adds the header cannot be defeated by a stale `fresh=0`
/// further down the chain.
#[test]
fn the_fresh_header_is_an_alias_that_only_opts_in() {
    let headers = |value: Option<&str>| {
        let mut h = HeaderMap::new();
        if let Some(v) = value {
            h.insert(
                "x-hopper-fresh",
                HeaderValue::from_str(v).expect("header value"),
            );
        }
        h
    };
    let q = |raw, header| {
        V1Query::parse(Some(raw))
            .with_fresh_header(&headers(header))
            .fresh
    };
    let base = "purl=pkg:npm/left-pad@1.3.0";
    assert!(q(base, Some("1")), "the header alone opts in");
    assert!(q(base, Some("true")));
    assert!(!q(base, Some("0")), "a negative header is not an opt-in");
    assert!(!q(base, None));
    // Neither spelling can opt back out of the other.
    assert!(q("purl=pkg:npm/left-pad@1.3.0&fresh=1", Some("0")));
    assert!(q("purl=pkg:npm/left-pad@1.3.0&fresh=0", Some("1")));
}

/// Forcing a fresh run is opt-in, and only an affirmative spelling opts in.
/// The expensive reading of an ambiguous value is the one that burns an
/// analysis slot, so anything else leaves the cheap path in place.
#[test]
fn only_an_affirmative_force_spends_a_slot() {
    let q = |raw| V1Query::parse(Some(raw)).force;
    assert!(q("purl=pkg:npm/left-pad@1.3.0&force=1"));
    assert!(q("purl=pkg:npm/left-pad@1.3.0&force=true"));
    assert!(q("purl=pkg:npm/left-pad@1.3.0&force=yes"));
    assert!(!q("purl=pkg:npm/left-pad@1.3.0&force=0"));
    assert!(!q("purl=pkg:npm/left-pad@1.3.0&force=false"));
    assert!(!q("purl=pkg:npm/left-pad@1.3.0&force="));
    assert!(!q("purl=pkg:npm/left-pad@1.3.0"));
}

#[test]
fn refresh_has_one_unambiguous_wire_spelling() {
    let q = |raw: &str| -> bool { V1Query::parse(Some(raw)).refresh };
    assert!(q(
        "sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa&refresh=1"
    ));
    for value in ["", "0", "true", "yes", "2"] {
        let raw = format!(
            "sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa&refresh={value}"
        );
        assert!(!q(&raw));
    }
}

#[test]
fn refresh_reuses_only_an_explicit_matching_traits_version() {
    assert!(matching_traits_version(Some("abc12"), Some("abc12")));
    assert!(!matching_traits_version(Some("old00"), Some("abc12")));
    assert!(!matching_traits_version(None, Some("abc12")));
    assert!(!matching_traits_version(Some("abc12"), None));
    assert!(!matching_traits_version(None, None));
}

#[test]
fn follow_policy_repeats_union_and_only_narrows_the_server_default() {
    use crate::fetch::FetchPolicy;

    let configured: FetchPolicy = "all".parse().unwrap();
    let query = V1Query::parse(Some(
        "purl=pkg:npm/app@1.0.0&follow=references&follow=ci-actions",
    ));
    assert_eq!(query.follow, ["references", "ci-actions"]);
    let effective = v1_follow_policy(&query, configured).expect("valid selection");
    assert!(effective.urls && effective.packages && effective.deps && effective.ci);

    // A server configured not to follow references must not be talked into
    // it, and must say so rather than answer a shallower question.
    let dependencies_only: FetchPolicy = "dependencies".parse().unwrap();
    let widen = V1Query::parse(Some("follow=references,dependencies"));
    let refused = v1_follow_policy(&widen, dependencies_only).expect_err("widening refused");
    assert_eq!(refused.status, StatusCode::BAD_REQUEST);
    assert_eq!(refused.code, "follow_not_allowed");
    let within = V1Query::parse(Some("follow=dependencies"));
    let effective = v1_follow_policy(&within, dependencies_only).expect("same policy");
    assert!(effective.deps && !effective.urls && !effective.packages && !effective.ci);

    let narrow = V1Query::parse(Some("follow=none"));
    let effective = v1_follow_policy(&narrow, configured).expect("valid selection");
    assert!(!effective.urls && !effective.packages && !effective.deps && !effective.ci);

    let legacy = V1Query::parse(Some("follow=deps"));
    assert!(v1_follow_policy(&legacy, configured).is_err());
}

/// One shape, whichever route answered and whatever it found.
///
/// A caller writes one parser against nine keys and reads `decision` to
/// know what happened. That only holds if every way of producing a decision
/// produces the same keys — a field present on a lookup and absent on an
/// analysis is a field nobody can rely on, and the difference would show up
/// as an intermittent null rather than as an error.
#[test]
fn every_decision_has_the_same_shape() {
    use crate::hopper::{Finding, Record};
    use crate::lookup::{Hit, Verdict};

    let stored = Verdict {
        sha256: "a".repeat(64),
        lvl: Level::At(3),
        eng: "2.8.0".into(),
        at: "2026-08-01T00:00:00Z".into(),
        purl: Some("pkg:npm/evil@1.0.0".into()),
        why: Some("Reverse shell in postinstall.".into()),
        hits: vec![Hit {
            id: "objectives/c2/backdoor".into(),
            crit: 5,
            file: "lib/install.js".into(),
            pkg: String::new(),
            desc: "Spawns bash".into(),
            off: Some(109),
            line: Some(12),
        }],
    };
    let from_corpus = Record {
        sha256: Some("a".repeat(64)),
        purl: Some("pkg:npm/evil@1.0.0".into()),
        fires_at: Level::At(3),
        engine_version: Some("2.8.0".into()),
        traits_version: None,
        analyzed_at: Some("2026-08-01T00:00:00Z".into()),
        reason: Some("Reverse shell in postinstall.".into()),
        findings: vec![Finding {
            id: "objectives/c2/backdoor".into(),
            crit: 5,
            desc: None,
        }],
        ..Record::default()
    };

    let shapes = [
        // What /v1/lookup and /v1/analyze both answer with on a hit.
        V1Decision::stored(&stored, Some("pkg:npm/evil@1.0.0"), 25),
        // What a lookup answers with when the corpus knew instead.
        V1Decision::corpus(&from_corpus, None, Some("pkg:npm/evil@1.0.0"), 25),
        V1Decision::unanalyzed(None, Some("pkg:npm/evil@1.0.0")),
        V1Decision::unavailable(None, Some("pkg:npm/evil@1.0.0")),
    ];

    let keys = |d: &V1Decision| -> Vec<String> {
        let v = serde_json::to_value(d).expect("serializes");
        let mut k: Vec<String> = v
            .as_object()
            .expect("an object")
            .keys()
            .map(String::clone)
            .collect();
        k.sort();
        k
    };
    let expected = keys(&shapes[0]);
    assert_eq!(expected.len(), 9, "the shape changed: {expected:?}");
    for shape in &shapes[1..] {
        assert_eq!(
            keys(shape),
            expected,
            "a decision answered with different keys"
        );
    }

    // A finding, by contrast, does NOT keep one shape across the two
    // sources, and that is deliberate. The decision object answers a fixed
    // set of questions, so its keys never move; a finding's content depends
    // on where it came from, and the corpus holds no file or offset at all.
    // Sending four nulls to say so is most of the object, so absence is how
    // "nothing to say" is spelled here.
    let finding_keys = |d: &V1Decision| -> Vec<String> {
        let v = serde_json::to_value(d).expect("serializes");
        let mut k: Vec<String> = v["findings"][0]
            .as_object()
            .expect("a finding")
            .keys()
            .map(String::clone)
            .collect();
        k.sort();
        k
    };
    let stored_finding = finding_keys(&shapes[0]);
    let corpus_finding = finding_keys(&shapes[1]);
    assert_eq!(
        corpus_finding,
        ["crit", "id"],
        "a corpus finding must carry only what it knows",
    );
    // Whatever a finding does carry, it is never a null.
    for shape in &shapes[..2] {
        let v = serde_json::to_value(shape).expect("serializes");
        assert!(
            v["findings"][0]
                .as_object()
                .expect("a finding")
                .values()
                .all(|x| !x.is_null()),
            "a null survived into a finding: {}",
            v["findings"][0]
        );
    }
    // The identity and severity are the two a caller may always rely on,
    // whichever side of the index answered.
    for id_or_crit in ["id", "crit"] {
        assert!(
            stored_finding.iter().any(|k| k == id_or_crit)
                && corpus_finding.iter().any(|k| k == id_or_crit),
            "{id_or_crit} must be present on every finding",
        );
    }
}

/// A caller gets an answer about the package they named, spelled the way
/// they named it.
///
/// Found in production: `pkg:pypi/info.gianlucacosta.eos.core@2.0.2` came
/// back answered about `pkg:pypi/info-gianlucacosta-eos-core@2.0.2`. Both
/// name the same project — PEP 503 folds `.` and `_` to `-` — but a caller
/// cannot know that without implementing PEP 503, and a lookup that names
/// fifty packages is matched to its request by this field. Rewriting it
/// breaks that correlation silently, and only for names with a `.` or `_`
/// in them.
#[test]
fn a_decision_is_spelled_the_way_the_caller_asked() {
    use crate::lookup::Verdict;

    let asked = "pkg:pypi/info.gianlucacosta.eos.core@2.0.2";
    let normalized = "pkg:pypi/info-gianlucacosta-eos-core@2.0.2";
    assert_eq!(
        normalize_pkg_purl(asked).as_deref(),
        Ok(normalized),
        "the premise: these two spellings are one package",
    );

    // PEP 503 lowercases as well as folding separators, and that half was
    // sighted separately in production: `pkg:pypi/ImportanceScore@1.2` came
    // back answered about `pkg:pypi/importancescore@1.2`. Same cause, and a
    // caller whose package name has no `.` or `_` in it at all.
    let mixed = "pkg:pypi/ImportanceScore@1.2";
    assert_eq!(
        normalize_pkg_purl(mixed).as_deref(),
        Ok("pkg:pypi/importancescore@1.2"),
        "the premise: case folds too",
    );
    let cased =
        V1Decision::unanalyzed(None, Some("pkg:pypi/importancescore@1.2")).asked_about(Some(mixed));
    assert_eq!(
        serde_json::to_value(&cased).expect("serializes")["purl"],
        mixed,
        "the caller's capitalization was rewritten",
    );

    let stored = Verdict {
        sha256: "a".repeat(64),
        lvl: Level::Clean,
        eng: "2.8.0".into(),
        at: "2026-08-01T00:00:00Z".into(),
        // What the index holds, which is always the normalized key.
        purl: Some(normalized.to_string()),
        why: None,
        hits: Vec::new(),
    };

    // Every kind of decision, since a caller correlating a list of fifty
    // gets whichever kind we happen to have.
    let decisions = [
        V1Decision::stored(&stored, Some(normalized), 25).asked_about(Some(asked)),
        V1Decision::unanalyzed(None, Some(normalized)).asked_about(Some(asked)),
        V1Decision::unavailable(None, Some(normalized)).asked_about(Some(asked)),
    ];
    for d in &decisions {
        let v = serde_json::to_value(d).expect("serializes");
        assert_eq!(
            v["purl"], asked,
            "answered about a different spelling than was asked about",
        );
    }

    // The digest still names the artifact, and is what proves two
    // spellings are one thing.
    let v = serde_json::to_value(&decisions[0]).expect("serializes");
    assert_eq!(v["sha256"], "a".repeat(64));

    // A lookup by digest alone has no spelling to echo, so the stored one
    // stands rather than becoming null.
    let by_sha = V1Decision::stored(&stored, None, 25).asked_about(None);
    let v = serde_json::to_value(&by_sha).expect("serializes");
    assert_eq!(v["purl"], normalized);
}

/// Findings are evidence for the decision, not a dump of everything the
/// scanner noticed. A benign crate matched ten "Rust test marker" traits at
/// `crit: 3`; answering an `allow` with all ten invites a caller to read
/// significance into noise, and it disagreed with the same artifact looked
/// up from the corpus, where the trigger had already cut them.
#[test]
fn only_the_strongest_few_findings_reach_the_wire() {
    use crate::hopper::{Finding, Record};
    use crate::lookup::{Hit, Verdict};

    let hit = |id: &str, crit: u8| Hit {
        id: id.into(),
        crit,
        file: "lib/install.js".into(),
        pkg: String::new(),
        desc: String::new(),
        off: None,
        line: None,
    };
    let ids = |d: &V1Decision| -> Vec<String> { d.findings.iter().map(|f| f.id.clone()).collect() };

    let benign = Verdict {
        sha256: "a".repeat(64),
        lvl: Level::Clean,
        eng: "2.8.0".into(),
        at: "2026-08-01T00:00:00Z".into(),
        purl: Some("pkg:cargo/tokio@1.40.0".into()),
        why: None,
        hits: (0..10)
            .map(|i| hit(&format!("testing/harness::{i}"), 3))
            .collect(),
    };
    let d = V1Decision::stored(&benign, None, 25);
    assert_eq!(d.decision, super::decision::Decision::Allow);
    assert!(
        ids(&d).is_empty(),
        "sub-threshold traits were reported as evidence: {:?}",
        ids(&d),
    );

    // Worst first, capped at three, and equal criticalities keep the order
    // their source listed them in.
    let noisy = Verdict {
        hits: vec![
            hit("weak", 4),
            hit("worst", 6),
            hit("dropped", 3),
            hit("strong-a", 5),
            hit("strong-b", 5),
            hit("cut", 4),
        ],
        ..benign
    };
    assert_eq!(
        ids(&V1Decision::stored(&noisy, None, 25)),
        ["worst", "strong-a", "strong-b"],
    );

    // The corpus applies the same rule in a trigger, so passing it through
    // here changes nothing — which is the point: one artifact reads the
    // same whichever side of the index answered.
    let record = Record {
        sha256: Some("a".repeat(64)),
        purl: Some("pkg:cargo/tokio@1.40.0".into()),
        fires_at: Level::Clean,
        engine_version: Some("2.8.0".into()),
        traits_version: None,
        analyzed_at: Some("2026-08-01T00:00:00Z".into()),
        reason: None,
        findings: (0..10)
            .map(|i| Finding {
                id: format!("testing/harness::{i}"),
                crit: 3,
                desc: None,
            })
            .collect(),
        ..Record::default()
    };
    assert!(
        ids(&V1Decision::corpus(&record, None, None, 25)).is_empty(),
        "a corpus record reported findings a stored verdict would have cut",
    );
}

/// `unavailable` is a statement about us, not about the artifact, so nothing
/// about the artifact may ride along on one. A caller that could read a
/// severity or a budget out of a failed lookup would eventually branch on
/// it, and would then be treating our outage as evidence.
#[test]
fn an_unavailable_decision_carries_nothing_about_the_package() {
    let d = V1Decision::unavailable(Some("a"), Some("pkg:npm/x@1.0.0"));
    let v = serde_json::to_value(&d).expect("serializes");
    assert_eq!(v["decision"], "unavailable");
    assert_eq!(v["purl"], "pkg:npm/x@1.0.0");
    for empty in [
        "severity",
        "fires_at",
        "reason",
        "engine_version",
        "analyzed_at",
    ] {
        assert!(
            v[empty].is_null(),
            "{empty} leaked into an unavailable decision"
        );
    }
    assert_eq!(v["findings"].as_array().map(Vec::len), Some(0));
}

/// The line this draws is the one [`ApiError::from_analysis`] already
/// draws for unstreamed callers: an artifact nobody can download is the
/// package's failure, and everything else is ours. Poppy only ever reads
/// the stream, so without it a dead package — a deleted Go repo, say —
/// would reach it as an outage and count against the fleet error rate.
#[test]
fn only_the_artifacts_own_failures_are_named() {
    assert_eq!(
        unretrievable_reason(StatusCode::UNPROCESSABLE_ENTITY),
        Some("unretrievable"),
    );
    assert_eq!(
        unretrievable_reason(StatusCode::PAYLOAD_TOO_LARGE),
        Some("too_large"),
    );

    // Ours. These stay an unqualified outage, because retrying them against
    // a healthier server is exactly the right thing for a caller to do.
    for status in [
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::GATEWAY_TIMEOUT,
        StatusCode::TOO_MANY_REQUESTS,
        StatusCode::SERVICE_UNAVAILABLE,
        StatusCode::NOT_FOUND,
    ] {
        assert_eq!(unretrievable_reason(status), None, "status {status}");
    }
}

/// `because` qualifies an unavailable without promoting it: the decision is
/// still that we did not answer, so nothing may read an assessment from it.
#[test]
fn a_reason_qualifies_but_does_not_become_a_verdict() {
    let bare = V1Decision::unavailable(None, Some("pkg:golang/example.com/x@v1.0.0"));
    assert_eq!(bare.reason, None);

    let named = V1Decision::unavailable(None, Some("pkg:golang/example.com/x@v1.0.0"))
        .because(unretrievable_reason(StatusCode::UNPROCESSABLE_ENTITY));
    assert_eq!(named.reason.as_deref(), Some("unretrievable"));
    assert_eq!(named.decision, super::decision::Decision::Unavailable);
    assert_eq!(named.severity, None);
    assert_eq!(named.fires_at, Level::Manual);
    assert!(named.findings.is_empty());

    // No reason leaves the decision byte-for-byte what it was.
    let untouched = V1Decision::unavailable(None, None).because(None);
    assert_eq!(untouched.reason, None);
}

/// A streamed upload's progress names the phase of the run it rides. The
/// upload's `/_/requests` entry is named by its filename while its subject
/// is the digest, so a search by name would report `phase: null` for
/// every streamed upload.
#[test]
fn a_streamed_uploads_progress_carries_its_phase() {
    let flights = Arc::new(super::super::flight::Flights::default());
    let sha = "c".repeat(64);
    let leader = flights.join(FlightKey::from(Artifact::Sha(sha.clone())));
    let follower = flights.join(FlightKey::from(Artifact::Sha(sha.clone())));
    let phase = crate::analysis::RequestPhase::with_label("req#3 upload-3");
    leader.flight().set_phase(phase.clone());
    phase.set("cleave:analyze");

    let named = Named {
        key: None,
        asked: None,
        subject: sha.clone(),
        is_url: false,
    };
    let frame = progress_frame(&follower, &named, Instant::now());
    assert_eq!(frame["state"], "analyzing");
    assert_eq!(frame["phase"], "cleave:analyze");
    assert_eq!(frame["sha256"], sha, "an upload is named by its digest");
}
