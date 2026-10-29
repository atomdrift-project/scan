use super::*;
use std::io::Write;

#[test]
fn ranked_tokens_keep_strongest_with_stable_ties() {
    let mut summary = FindingSummary::default();
    for i in (0..600).rev() {
        let path = format!("path{i:04}");
        summary.sample_paths.insert(path.clone(), 3);
        summary.path_scores.insert(path, 3.0);
    }
    summary.sample_paths.insert("strong".into(), 5);
    summary.path_scores.insert("strong".into(), 4.5);
    for (order, limit) in [(2, 512), (3, 128), (4, 64)] {
        let tokens = ranked_tiered_path_tokens(&summary, 3, &HashMap::new(), order);
        let mut expected: Vec<_> = (0..limit - 1).map(|i| format!("n:path{i:04}")).collect();
        expected.push("h:strong".into());
        expected.sort();
        assert_eq!(tokens, expected);
    }
}

#[test]
fn ranked_score_uses_paired_severity_and_confidence() {
    let summary = summarize_findings(&[
        RawFinding {
            id: "a/b::one".into(),
            crit: 5,
            conf: 0.7,
            ..RawFinding::default()
        },
        RawFinding {
            id: "a/b::two".into(),
            crit: 3,
            conf: 1.0,
            ..RawFinding::default()
        },
        RawFinding {
            id: "a/b::low".into(),
            crit: 5,
            conf: 0.1,
            ..RawFinding::default()
        },
    ]);
    assert_eq!(summary.path_scores["a/b"], 5.0 * f64::from(0.7_f32));
    assert_eq!(summary.path_scores["a"], 5.0 * f64::from(0.7_f32));
}

#[test]
fn ranked_base_paths_select_before_vocabulary_filtering() {
    let findings: Vec<_> = (0..600)
        .rev()
        .map(|i| RawFinding {
            id: format!("path{i:04}::trait"),
            conf: 1.0,
            crit: 3,
            ..RawFinding::default()
        })
        .collect();
    assert_eq!(
        ranked_ngram_paths(&findings, 128),
        (0..128).map(|i| format!("path{i:04}")).collect::<Vec<_>>()
    );
}

fn paths(id: &str) -> Vec<&str> {
    finding_paths(id, default_trait_path_depth()).collect()
}

/// An ATT&CK ID whose first four bytes straddle a codepoint must not take
/// the analysis task down with it. `atk` is copied straight from a finding
/// and nothing validates it, so hostile or mojibake input reaches the
/// tactic-prefix slice directly; in production this panicked 38 analyses
/// with "end byte index 4 is not a char boundary".
#[test]
fn attack_tactic_prefix_tolerates_non_ascii_technique_id() {
    // T(0) ü(1..3) ─(3..6) — byte 4 lands inside the box char.
    let codes = ["T\u{fc}\u{2500}x".to_string(), "T1059.001".to_string()];
    let lookup: HashMap<String, usize> = [("agg:attack_tactic_count".to_string(), 0)]
        .into_iter()
        .collect();
    let mut vec = vec![0.0f32; 1];
    write_code_features(
        &FindingSummary::default(),
        &codes,
        &[],
        &mut FeatureWriter {
            vec: &mut vec,
            lookup: &lookup,
        },
    );
    // The unusable ID is skipped; the well-formed one still counts.
    assert_eq!(
        vec[0], 1.0,
        "expected only the ASCII technique to yield a tactic prefix"
    );
}

/// The ordinals the writers compare against are cleave's criticality ranks.
#[test]
fn criticality_ordinals_match_cleave_ranks() {
    use cleave::Criticality;
    assert_eq!(u32::from(Criticality::Baseline.rank()), CRIT_BASELINE);
    assert_eq!(u32::from(Criticality::Notable.rank()), CRIT_NOTABLE);
    assert_eq!(u32::from(Criticality::Suspicious.rank()), CRIT_SUSPICIOUS);
    assert_eq!(u32::from(Criticality::Hostile.rank()), CRIT_HOSTILE);
}

/// `kv:` tokens come from boolean and string metrics only, as in
/// collimator with shape mode off.
#[test]
fn kv_tokens_cover_booleans_and_strings_only() {
    let metrics = serde_json::json!({
        "pe": {"signed": true, "subsystem": "  windows   gui ", "sections": 4},
        "text": "not a group",
    });
    let mut tokens = kv_tokens(Some(&metrics));
    tokens.sort();
    assert_eq!(tokens, ["kv:pe.signed=true", "kv:pe.subsystem=windows gui"]);
    assert!(kv_tokens(None).is_empty());
}

#[test]
fn trait_depth_spec_defaults_and_reads_explicit_value() {
    let old: FeatureSpec = serde_json::from_str("{}").expect("legacy defaults");
    assert_eq!(old.trait_path_depth, 3);
    let new: FeatureSpec = serde_json::from_str(r#"{"trait_path_depth":5}"#).expect("depth field");
    assert_eq!(new.trait_path_depth, 5);
}

#[test]
fn trait_depth_prefixes_are_configurable() {
    let id = "objectives/evasion/process/injection::method";
    assert_eq!(finding_paths(id, 1).collect::<Vec<_>>(), ["objectives"]);
    assert_eq!(
        finding_paths(id, 2).collect::<Vec<_>>(),
        ["objectives", "objectives/evasion"]
    );
    assert_eq!(
        finding_paths(id, 4).collect::<Vec<_>>(),
        [
            "objectives",
            "objectives/evasion",
            "objectives/evasion/process",
            "objectives/evasion/process/injection",
        ]
    );
    assert_eq!(
        finding_paths("métadata/format::x", 12).collect::<Vec<_>>(),
        ["métadata", "métadata/format"]
    );
}

#[test]
fn renamed_metrics_resolve_for_numbers_and_categorical_tokens() {
    for &(old_group, old_key, new_group, new_key) in METRIC_RENAMES {
        for value in [
            serde_json::json!(5.27),
            serde_json::json!(true),
            serde_json::json!("signed"),
        ] {
            let mut metrics = serde_json::json!({});
            metrics[new_group] = serde_json::json!({new_key: value});
            let original = metrics.clone();
            let fields: Vec<_> = metric_fields(Some(&metrics)).collect();
            assert!(fields.contains(&(old_group, old_key, &value)));
            if value.is_number() || value.is_boolean() {
                assert_eq!(
                    numeric_metrics(Some(&metrics))[old_group][old_key],
                    if value.is_boolean() { 1.0 } else { 5.27 }
                );
            } else {
                assert!(
                    kv_tokens(Some(&metrics)).contains(&format!("kv:{old_group}.{old_key}=signed"))
                );
            }
            assert!(
                kv_tokens_with_options(Some(&metrics), true, false)
                    .contains(&format!("kv:{old_group}.{old_key}:exists"))
            );
            assert_eq!(metrics, original, "aliasing must not mutate the report");
        }
    }
}

#[test]
fn historical_metrics_fill_v9_names_and_sibling_aliases() {
    for &(old_group, old_key, new_group, new_key) in METRIC_RENAMES {
        for value in [
            serde_json::json!(0),
            serde_json::Value::Null,
            serde_json::json!(false),
            serde_json::json!(5.27),
            serde_json::json!("signed"),
        ] {
            let metrics = serde_json::json!({old_group: {old_key: value}});
            let original = metrics.clone();
            let fields: Vec<_> = metric_fields(Some(&metrics)).collect();
            assert!(fields.contains(&(new_group, new_key, &value)));
            assert_eq!(metrics, original);
        }
    }
    let metrics = serde_json::json!({"ast": {"max_array_len": 0, "max_array_length": 12}});
    let fields: Vec<_> = metric_fields(Some(&metrics)).collect();
    assert!(fields.contains(&("ast", "array_literal_max_length", &serde_json::json!(12))));
    assert!(fields.contains(&("ast", "max_array_len", &serde_json::json!(0))));
}

#[test]
fn metric_aliases_preserve_explicit_values_and_malformed_groups() {
    for value in [
        serde_json::json!(0),
        serde_json::Value::Null,
        serde_json::json!(false),
        serde_json::json!("old"),
    ] {
        let metrics =
            serde_json::json!({"file": {"entropy": 5.27}, "binary": {"overall_entropy": value}});
        let fields: Vec<_> = metric_fields(Some(&metrics)).collect();
        let old: Vec<_> = fields
            .iter()
            .filter(|(g, k, _)| *g == "binary" && *k == "overall_entropy")
            .collect();
        assert_eq!(old.len(), 1);
        assert_eq!(old[0].2, &value);
    }
    for group in [
        serde_json::Value::Null,
        serde_json::json!(3),
        serde_json::json!([]),
    ] {
        let metrics = serde_json::json!({"file": {"entropy": 5.27}, "binary": group});
        assert!(
            !metric_fields(Some(&metrics)).any(|(g, k, _)| g == "binary" && k == "overall_entropy")
        );
    }
    assert_eq!(metric_fields(None).count(), 0);
}

#[test]
fn member_metadata_is_independent_of_archive_depth() {
    let names = [
        "present:objectives",
        "maxcrit:objectives",
        "score:hopper_score",
        "score:density",
        "inter:javascript*score",
        "formula:skeleton_len",
        "formula:unique_elements",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let spec = anchor_spec(vec!["objectives".into()], vec![], vec![], names);
    let ctx = ExtractContext::new(&spec);
    for score in [0, 124] {
        for formula in [None, Some("O₂H₄")] {
            let mut file: cleave::types::CompactFile = serde_json::from_value(serde_json::json!({
                "id": 7, "sha": "member", "path": "pkg.vsix!!main.js",
                "type": "javascript", "size": 2048, "risk": score,
                "mol": formula,
                "traits": [{"id": "objectives/execution::shell", "conf": 1.0, "crit": 4}]
            }))
            .expect("member fixture");
            let expected = ctx
                .extract_from_parsed(&ParsedReport::from_compact_files(&[&file], ctx.raw_needs()));
            for depth in [0, 1, 3] {
                file.depth = depth;
                let parsed = ParsedReport::from_compact_file(&file, ctx.raw_needs());
                assert_eq!(parsed.sample_score, score);
                assert_eq!(parsed.formula_str, formula.unwrap_or(""));
                assert_eq!(
                    parsed.elements_str,
                    if formula.is_some() { "OH" } else { "" }
                );
                assert_eq!(ctx.extract_from_parsed(&parsed), expected, "depth {depth}");
            }
        }
    }
}

#[test]
fn archive_depths_are_relative_order_independent_and_cycle_safe() {
    let root = FileSummary {
        id: Some(10),
        depth: 2,
        ..Default::default()
    };
    let child = FileSummary {
        id: Some(20),
        parent: Some(10),
        depth: 3,
        ..Default::default()
    };
    let nested = FileSummary {
        id: Some(30),
        parent: Some(20),
        ..Default::default()
    };
    assert_eq!(
        relative_depths(&[root.clone(), child.clone(), nested.clone()]),
        vec![0, 1, 2]
    );
    assert_eq!(
        relative_depths(&[nested, child.clone(), root]),
        vec![2, 1, 0]
    );
    assert_eq!(relative_depths(&[child]), vec![0]);
    let cycle = [
        FileSummary {
            id: Some(1),
            parent: Some(2),
            ..Default::default()
        },
        FileSummary {
            id: Some(2),
            parent: Some(1),
            ..Default::default()
        },
    ];
    assert_eq!(relative_depths(&cycle), vec![0, 0]);
    let chain: Vec<_> = (0..5000u32)
        .rev()
        .map(|id| FileSummary {
            id: Some(id),
            parent: id.checked_sub(1),
            ..Default::default()
        })
        .collect();
    assert_eq!(relative_depths(&chain)[0], 4999);
}

#[test]
fn explicit_metric_transform_overrides_legacy_defaults() -> Result<()> {
    for (transform, expected) in [("identity", 9.0_f32), ("log1p_abs", 10.0_f32.ln())] {
        let mut file = tempfile::NamedTempFile::new()?;
        let value = serde_json::json!({
            "version":21,"abi_version":21,"presence_vocab":["metadata"],
            "feature_names":["metrics:text_max_inline_whitespace_run"],"total_features":1,
            "metric_transforms":{"text_max_inline_whitespace_run":transform},"standardized":false
        });
        serde_json::to_writer(&mut file, &value)?;
        let spec = FeatureSpec::load(file.path())?;
        let report = serde_json::from_value(serde_json::json!({"v":"8","files":[{
            "id":0,"path":"test.go","type":"go","sha":"fixture","size":100,
            "facts":{"metrics":{"text":{"max_inline_whitespace_run":9}}}
        }]}))?;
        assert!((ExtractContext::new(&spec).extract(&report)[0] - expected).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn new_metric_contract_is_required_and_legacy_abi_defaults_to_version() -> Result<()> {
    for (version, transforms, succeeds) in [
        (20, None, true),
        (21, None, false),
        (21, Some("identity"), true),
        (21, Some("log10"), false),
    ] {
        let mut file = tempfile::NamedTempFile::new()?;
        let mut value = serde_json::json!({"version":version,"presence_vocab":["metadata"],
                "feature_names":["metrics:text_max_inline_whitespace_run"],"total_features":1,"standardized":false});
        if let Some(t) = transforms {
            value["metric_transforms"] = serde_json::json!({"text_max_inline_whitespace_run":t});
        }
        serde_json::to_writer(&mut file, &value)?;
        let result = FeatureSpec::load(file.path());
        assert_eq!(result.is_ok(), succeeds, "{value}");
        if let Ok(spec) = result {
            assert_eq!(spec.abi_version, version);
        }
    }
    Ok(())
}

#[test]
fn aggregate_metadata_still_comes_only_from_root() {
    let report: cleave::types::CompactReport = serde_json::from_value(
            serde_json::json!({"v": "8", "files": [
                {"id": 1, "sha": "member", "path": "pkg.vsix!!main.js", "type": "javascript", "size": 2048,
                 "depth": 1, "risk": 124, "mol": "O₂"},
                {"id": 0, "sha": "root", "path": "pkg.vsix", "type": "vsix", "size": 4096,
                 "depth": 0, "risk": 10, "mol": "H₄"}
            ]}),
        ).expect("aggregate fixture");
    let parsed = ParsedReport::from_compact_report(&report, RawNeeds::all(), None);
    assert_eq!(parsed.sample_score, 10);
    assert_eq!(parsed.formula_str, "H₄");
    assert_eq!(parsed.summaries.len(), 2);
    let keep = ["member".to_owned()].into_iter().collect();
    let filtered = ParsedReport::from_compact_report(&report, RawNeeds::all(), Some(&keep));
    assert_eq!(filtered.sample_score, 0);
    assert!(filtered.formula_str.is_empty());
    assert_eq!(filtered.summaries.len(), 1);
}

/// Unlike report parity, this exercises the entry point used to grade an
/// embedded file. Expected vectors must come from its training DB row.
#[test]
#[ignore = "needs SCAN_MODELS_DIR with a member training extraction fixture"]
fn member_features_match_training_fixture() -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Sample {
        report: cleave::types::CompactReport,
        expected_features: Vec<f32>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        samples: Vec<Sample>,
    }
    let dir = crate::env::path(crate::env::SCAN_MODELS_DIR).context("SCAN_MODELS_DIR")?;
    let spec = FeatureSpec::load(&dir.join("feature_spec.json"))?;
    let fixture: Fixture =
        serde_json::from_slice(&std::fs::read(dir.join("extraction_fixture.json"))?)?;
    assert!(!fixture.samples.is_empty());
    let ctx = ExtractContext::new(&spec);
    for sample in fixture.samples {
        assert_eq!(
            sample.report.files.len(),
            1,
            "fixture must contain only the graded member"
        );
        let parsed = ParsedReport::from_compact_file(&sample.report.files[0], ctx.raw_needs());
        let actual = ctx.extract_from_parsed(&parsed);
        assert_eq!(actual.len(), sample.expected_features.len());
        for (i, (actual, expected)) in actual.iter().zip(&sample.expected_features).enumerate() {
            assert!(
                actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= 1e-5,
                "{}: serving={actual}, training={expected}",
                spec.feature_names[i]
            );
        }
    }
    Ok(())
}

#[test]
fn trait_depth_is_per_route_for_shared_report() {
    let file: cleave::types::CompactFile = serde_json::from_value(serde_json::json!({
        "id": 0, "sha": "fixture", "path": "fixture", "type": "go", "size": 1,
        "traits": [{"id": "objectives/evasion/process/injection::method", "conf": 1.0, "crit": 4}]
    }))
    .expect("compact fixture");
    let names = vec![
        "present:objectives/evasion/process/injection".to_string(),
        "maxcrit:objectives/evasion/process/injection".to_string(),
        "tierbi:s:objectives + s:objectives/evasion/process/injection".to_string(),
        "tiertri:s:objectives + s:objectives/evasion + s:objectives/evasion/process/injection"
            .to_string(),
    ];
    let mut spec = anchor_spec(
        vec!["objectives/evasion/process/injection".to_string()],
        vec![],
        vec![],
        names,
    );
    let shared = ParsedReport::from_compact_file(&file, RawNeeds::all());
    spec.trait_path_depth = 4;
    let deep = ExtractContext::new(&spec);
    assert_eq!(deep.extract_from_parsed(&shared), [1.0, 4.0, 1.0, 1.0]);
    assert!(std::sync::Arc::ptr_eq(
        &shared.at_trait_path_depth(4),
        &shared.at_trait_path_depth(4)
    ));
    spec.trait_path_depth = 3;
    let default = ExtractContext::new(&spec);
    assert_eq!(default.extract_from_parsed(&shared), [0.0; 4]);
    assert_eq!(deep.extract_from_parsed(&shared), [1.0, 4.0, 1.0, 1.0]);
}

#[test]
fn test_finding_paths_deep() {
    assert_eq!(
        paths("objectives/evasion/process/injection::technique-x"),
        vec![
            "objectives",
            "objectives/evasion",
            "objectives/evasion/process"
        ]
    );
}

#[test]
fn test_finding_paths_two_levels() {
    assert_eq!(
        paths("metadata/format::no-functions"),
        vec!["metadata", "metadata/format"]
    );
}

#[test]
fn test_finding_paths_one_level() {
    assert_eq!(paths("standalone"), vec!["standalone"]);
}

/// `file_symbols` reads imports/exports/functions PLUS the
/// filefacts AST symbol kinds (`tgt` = call targets, `mbr` = member
/// chains). Tokens shorter than 2 chars are dropped to match collimator's
/// `_file_symbols` threshold.
#[test]
fn file_symbols_picks_up_targets_and_members() {
    let file: cleave::types::CompactFile = serde_json::from_value(serde_json::json!({
        "id": 0,
        "path": "lib.rs",
        "type": "rust",
        "sha": "a".repeat(64),
        "size": 1024,
        "facts": {
            "imp": [["libc", "open"], ["libc", "x"]],   // "x" too short, skipped
            "tgt": ["client.get", "attempt.url().origin", "a"],
            "mbr": ["window.localStorage", "process.env.PATH"],
            "exp": [["exported_fn"]],
            "funcs": [["render"], ["x"]]                // "x" too short, skipped
        }
    }))
    .expect("fixture is a valid compact file");
    let syms = file_symbols(&file.facts);
    assert!(
        syms.is_sorted(),
        "symbols are sorted for the n-gram writers"
    );
    assert_eq!(
        FileSummary::from_compact(&file, RawNeeds::all()).symbols,
        syms
    );

    for expected in [
        "open",                 // import
        "libc!open",            // composite import
        "exported_fn",          // export
        "render",               // function
        "client.get",           // target
        "attempt.url().origin", // target
        "window.localStorage",  // member chain
        "process.env.PATH",     // member chain
    ] {
        assert!(
            syms.iter().any(|s| s == expected),
            "expected {expected:?} in symbol set, got {syms:?}"
        );
    }
    assert!(
        syms.iter().all(|s| s != "a" && s != "x"),
        "1-char symbols should be filtered out"
    );
}

#[test]
fn test_standardize() {
    let spec = FeatureSpec {
        version: 16,
        abi_version: 16,
        presence_vocab: vec![],
        filetype_vocab: vec![],
        element_vocab: vec![],
        bigram_vocab: vec![],
        ghost_vocab: vec![],
        skeleton_vocab: vec![],
        rare_element_vocab: vec![],
        trigram_vocab: vec![],
        metric_vocab: vec![],
        metric_transforms: None,
        metric_getters: Vec::new(),
        metric_reference_year: 0,
        metric_correlations: HashMap::new(),
        crit_unigram_vocab: vec![],
        crit_bigram_vocab: vec![],
        crit_trigram_vocab: vec![],
        attack_bigram_vocab: vec![],
        attack_trigram_vocab: vec![],
        mbc_bigram_vocab: vec![],
        mbc_trigram_vocab: vec![],
        tiered_bigram_vocab: vec![],
        tiered_trigram_vocab: vec![],
        tiered_quadgram_vocab: vec![],
        kv_vocab: vec![],
        symbol_vocab: vec![],
        symbol_bigram_vocab: vec![],
        symbol_trigram_vocab: vec![],
        feature_names: vec![],
        total_features: 3,
        feature_means: Some(vec![0.0, 1.0, 2.0]),
        feature_stds: Some(vec![1.0, 2.0, 0.5]),
        standardized: true,
        trait_path_depth: default_trait_path_depth(),
        tiered_min_crit: default_tiered_min_crit(),
        tiered_branch_min_crit: HashMap::new(),
    };
    let mut features = vec![1.0, 3.0, 3.0];
    spec.standardize(&mut features);
    assert_eq!(features[0], 0.0);
    assert!((features[1] - 1.0).abs() < 1e-6);
    assert!((features[2] - 2.0).abs() < 1e-6);
}

#[test]
fn test_load_rejects_missing_feature_names() -> Result<()> {
    let mut file = tempfile::NamedTempFile::new()?;
    writeln!(
        file,
        "{{\"version\":16,\"presence_vocab\":[\"objectives\"],\"filetype_vocab\":[\"sh\"],\"total_features\":51,\"standardized\":false}}"
    )?;
    let Err(err) = FeatureSpec::load(file.path()) else {
        anyhow::bail!("missing feature_names should be rejected");
    };
    assert!(err.to_string().contains("feature_names length"));
    Ok(())
}

/// Minimal spec carrying only the fields the offset-family anchoring reads.
fn anchor_spec(
    presence_vocab: Vec<String>,
    bigram_vocab: Vec<String>,
    trigram_vocab: Vec<String>,
    feature_names: Vec<String>,
) -> FeatureSpec {
    FeatureSpec {
        version: 17,
        abi_version: 17,
        presence_vocab,
        bigram_vocab,
        trigram_vocab,
        total_features: feature_names.len(),
        feature_names,
        filetype_vocab: vec![],
        element_vocab: vec![],
        ghost_vocab: vec![],
        skeleton_vocab: vec![],
        rare_element_vocab: vec![],
        metric_vocab: vec![],
        metric_transforms: None,
        metric_getters: Vec::new(),
        metric_reference_year: 0,
        metric_correlations: HashMap::new(),
        crit_unigram_vocab: vec![],
        crit_bigram_vocab: vec![],
        crit_trigram_vocab: vec![],
        attack_bigram_vocab: vec![],
        attack_trigram_vocab: vec![],
        mbc_bigram_vocab: vec![],
        mbc_trigram_vocab: vec![],
        tiered_bigram_vocab: vec![],
        tiered_trigram_vocab: vec![],
        tiered_quadgram_vocab: vec![],
        kv_vocab: vec![],
        symbol_vocab: vec![],
        symbol_bigram_vocab: vec![],
        symbol_trigram_vocab: vec![],
        feature_means: None,
        feature_stds: None,
        standardized: false,
        trait_path_depth: default_trait_path_depth(),
        tiered_min_crit: default_tiered_min_crit(),
        tiered_branch_min_crit: HashMap::new(),
    }
}

#[test]
fn offset_families_anchor_to_real_spec_positions() {
    // feature_names deliberately places every offset family somewhere a
    // hand-maintained running cursor would NOT land. Anchoring must follow
    // the spec, not the cursor — this is the exact drift that ran the
    // unsigned-bigram block off the end of the vector in production.
    let feature_names = vec![
        "filler:0".to_string(),              // 0
        "present:objectives".to_string(),    // 1
        "maxcrit:objectives".to_string(),    // 2
        "filler:1".to_string(),              // 3
        "bigrams:a + b".to_string(),         // 4
        "bigrams:a + c".to_string(),         // 5
        "trigram:a + b + c".to_string(),     // 6
        "unsigned_bigram:a + b".to_string(), // 7
        "unsigned_bigram:a + c".to_string(), // 8
    ];
    let spec = anchor_spec(
        vec!["objectives".to_string()],
        vec!["a + b".to_string(), "a + c".to_string()],
        vec!["a + b + c".to_string()],
        feature_names,
    );
    let ctx = ExtractContext::new(&spec);
    // Each family member resolves to its real feature_names slot by name,
    // wherever a running cursor would have landed.
    assert_eq!(ctx.present_slots, vec![Some(1)]);
    assert_eq!(ctx.maxcrit_slots, vec![Some(2)]);
    assert_eq!(ctx.bigram_slots, vec![Some(4), Some(5)]);
    assert_eq!(ctx.trigram_slots, vec![Some(6)]);
    assert_eq!(ctx.unsigned_bigram_slots, vec![Some(7), Some(8)]);
}

#[test]
fn non_contiguous_offset_family_resolves_each_member_to_its_slot() {
    // The bigram family is laid out REVERSED vs vocab order. The old
    // `base + idx` scheme couldn't represent this and rejected the bundle;
    // the per-member slot map resolves each vocab entry to its true slot
    // (vocab idx 0 -> slot 1, idx 1 -> slot 0), so the bundle is accepted and
    // writes land correctly.
    let feature_names = vec![
        "bigrams:a + c".to_string(), // 0 — vocab idx 1
        "bigrams:a + b".to_string(), // 1 — vocab idx 0
    ];
    let spec = anchor_spec(
        vec![],
        vec!["a + b".to_string(), "a + c".to_string()],
        vec![],
        feature_names,
    );
    let ctx = ExtractContext::new(&spec);
    assert_eq!(ctx.bigram_slots, vec![Some(1), Some(0)]);
}

#[test]
fn shared_vocab_subset_family_maps_only_present_members() {
    // present: and maxcrit: share presence_vocab, but an allowlist prune
    // dropped a maxcrit: name while keeping its present: counterpart. The
    // dropped entry maps to None (skipped); the kept ones resolve to their
    // true slots. Indexing by `base + idx` would misalign them and reject the
    // bundle; this real-world shape is why the slot map exists.
    let feature_names = vec![
        "present:a".to_string(), // 0
        "present:b".to_string(), // 1
        "maxcrit:b".to_string(), // 2  (maxcrit:a was pruned)
    ];
    let spec = anchor_spec(
        vec!["a".to_string(), "b".to_string()],
        vec![],
        vec![],
        feature_names,
    );
    let ctx = ExtractContext::new(&spec);
    assert_eq!(ctx.present_slots, vec![Some(0), Some(1)]);
    assert_eq!(ctx.maxcrit_slots, vec![None, Some(2)]);
}

#[test]
fn fully_absent_family_is_skipped_not_rejected() {
    // bigram_vocab is non-empty but feature_names has no "bigrams:"/
    // "unsigned_bigram:" entries: the family was pruned/disabled in the spec.
    // Every entry maps to None and the extractor skips it — a legitimately
    // smaller model, not an incompatibility, so validation passes.
    let spec = anchor_spec(
        vec![],
        vec!["a + b".to_string()],
        vec![],
        vec!["filler:0".to_string()],
    );
    let ctx = ExtractContext::new(&spec);
    assert_eq!(ctx.bigram_slots, vec![None]);
}
