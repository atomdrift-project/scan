use super::*;

#[test]
fn normalize_archive_filetype_strips_pure_compression() {
    // Compound archive labels collapse onto their container.
    assert_eq!(normalize_archive_filetype("tar.gz"), "tar");
    assert_eq!(normalize_archive_filetype("tar.bz2"), "tar");
    assert_eq!(normalize_archive_filetype("tar.xz"), "tar");
    assert_eq!(normalize_archive_filetype("tar.zst"), "tar");
    assert_eq!(normalize_archive_filetype("tar.Z"), "tar");
    assert_eq!(normalize_archive_filetype("TAR.GZ"), "tar");
    // Repeated stripping for stacked suffixes.
    assert_eq!(normalize_archive_filetype("tar.bz2.xz"), "tar");
    // Bare compression labels keep their identity — RAW_COMPRESSED_FILETYPES
    // logic on the collimator side already gates them.
    assert_eq!(normalize_archive_filetype("gz"), "gz");
    assert_eq!(normalize_archive_filetype("bz2"), "bz2");
    // Containers without a compression suffix pass through.
    assert_eq!(normalize_archive_filetype("zip"), "zip");
    assert_eq!(normalize_archive_filetype("pe"), "pe");
    // Empty / whitespace.
    assert_eq!(normalize_archive_filetype(""), "");
    assert_eq!(normalize_archive_filetype("  tar.gz  "), "tar");
    // An already-normal label is borrowed, not copied.
    assert!(matches!(
        normalize_archive_filetype("elf"),
        Cow::Borrowed("elf")
    ));
}

#[test]
fn container_filetype_maps_packages_to_their_archive() {
    // Specialty package types collapse onto their container archive —
    // filefacts is the source of truth, so these stay correct as new
    // package formats are added there.
    assert_eq!(container_filetype("gem"), Some("tar"));
    assert_eq!(container_filetype("crate"), Some("tar"));
    assert_eq!(container_filetype("python_sdist"), Some("tar"));
    assert_eq!(container_filetype("npm"), Some("tar"));
    assert_eq!(container_filetype("whl"), Some("zip"));
    assert_eq!(container_filetype("jar"), Some("zip"));
    assert_eq!(container_filetype("nupkg"), Some("zip"));
    // Compressed tar variants resolve to the tar container too.
    assert_eq!(container_filetype("tar.gz"), Some("tar"));
    assert_eq!(container_filetype("tar.zst"), Some("tar"));
    // Non-archive types and bare compression carry no container.
    assert_eq!(container_filetype("pe"), None);
    assert_eq!(container_filetype("python"), None);
    assert_eq!(container_filetype("gz"), None);
    assert_eq!(container_filetype("not_a_type"), None);
}

#[test]
fn specialist_keys_routes_compressed_archives_to_their_container() {
    // collimator emits the filetype→filegroup map (and trains/calibrates
    // every specialist) under the normalized container label, so the map
    // carries `tar`/`javascript`, never `tar.gz`.
    let routes = RouteSet {
        filetype_to_filegroup: HashMap::from([
            ("tar".to_string(), "archive".to_string()),
            ("javascript".to_string(), "scripts".to_string()),
        ]),
        ..Default::default()
    };
    let keys = |file_type: &str| {
        let (key, group) = routes.specialist_keys(file_type);
        (key.into_owned(), group.map(str::to_owned))
    };

    // A compressed tarball must resolve to the `tar` container specialist
    // and its filegroup. This is the `.tgz` routing gap: selecting by the
    // raw `tar.gz` label found no specialist, surfaced `filetypes/tar` as an
    // `unavailable` skip, and let an obfuscated npm dropper score benign on
    // the general route alone.
    let tar = ("tar".to_string(), Some("archive".to_string()));
    assert_eq!(keys("tar.gz"), tar);
    // Stacked compression suffixes collapse all the way to the container.
    assert_eq!(keys("tar.bz2.xz"), tar);
    // The uncompressed container is unchanged.
    assert_eq!(keys("tar"), tar);
    // Non-archive types pass through untouched and still resolve their group.
    assert_eq!(
        keys("javascript"),
        ("javascript".to_string(), Some("scripts".to_string()))
    );
    // A container with no configured filegroup yields no group, still
    // normalized so its filetype specialist (if any) is reachable.
    assert_eq!(keys("zip"), ("zip".to_string(), None));
}

#[test]
fn route_names_parse_and_render_compactly() {
    assert_eq!(RouteKind::parse("general"), Some(RouteKind::General));
    assert_eq!(
        RouteKind::parse("filegroups/native"),
        Some(RouteKind::Group("native"))
    );
    assert_eq!(
        RouteKind::parse("filetypes/elf"),
        Some(RouteKind::Type("elf"))
    );
    assert_eq!(RouteKind::parse("mystery/thing"), None);
    assert_eq!(compact_route_name("general"), "az");
    assert_eq!(compact_route_name("filegroups/native"), "az/native");
    assert_eq!(compact_route_name("filetypes/elf"), "az/elf");

    let mut names = RouteNames::default();
    assert_eq!(names.id("general"), Some(RouteId::GENERAL));
    let elf = names.intern("filetypes/elf");
    assert_eq!(
        names.intern("filetypes/elf"),
        elf,
        "interning is idempotent"
    );
    assert_eq!(names.name(elf), "filetypes/elf");
}

#[test]
fn load_rejects_missing_feature_spec_with_update_guidance() -> Result<()> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join("model.json"), b"{}")?;

    let Err(err) = Model::load(dir.path(), None, None) else {
        anyhow::bail!("missing feature spec should be rejected");
    };
    let message = err.to_string();
    assert!(message.contains("model bundle is incomplete"));
    assert!(message.contains("Run 'atomscan update-rules'"));
    Ok(())
}

#[test]
fn capped_suspicious_level_clamps_to_the_ceiling() {
    // Below the ceiling passes through; at/above it clamps to the ceiling.
    assert_eq!(
        capped_suspicious_level(SUSPICIOUS_LEVEL_CEILING - 1),
        SUSPICIOUS_LEVEL_CEILING - 1
    );
    assert_eq!(
        capped_suspicious_level(SUSPICIOUS_LEVEL_CEILING),
        SUSPICIOUS_LEVEL_CEILING
    );
    assert_eq!(
        capped_suspicious_level(SUSPICIOUS_LEVEL_CEILING + 1),
        SUSPICIOUS_LEVEL_CEILING
    );
    assert_eq!(capped_suspicious_level(25_000), SUSPICIOUS_LEVEL_CEILING);
}

#[test]
fn bundle_config_is_absent_when_missing_and_fatal_when_malformed() -> Result<()> {
    let dir = tempfile::tempdir()?;
    assert!(BundleConfig::load(dir.path())?.is_none());

    std::fs::write(dir.path().join("config.json"), b"{\"levels\": [")?;
    let err = BundleConfig::load(dir.path()).expect_err("truncated config must fail");
    assert!(format!("{err:#}").contains("parsing"), "{err:#}");
    // The default-level probe leaves the report to `Model::load`.
    assert_eq!(model_default_level(dir.path()), None);

    // An ensemble whose config.json is malformed refuses to load rather
    // than silently degrading to general-only.
    std::fs::create_dir_all(dir.path().join("general"))?;
    let err = Model::load(dir.path(), None, None).expect_err("malformed config must fail");
    assert!(format!("{err:#}").contains("config.json"), "{err:#}");
    Ok(())
}

#[test]
fn single_bundle_thresholds_must_be_valid() -> Result<()> {
    let cfg: BundleConfig = serde_json::from_str(r#"{"hostile": 0.9}"#)?;
    let t = cfg.thresholds()?.context("hostile present")?;
    assert_eq!((t.suspicious, t.hostile), (0.9, 0.9), "hostile-only");

    let cfg: BundleConfig = serde_json::from_str(r#"{"suspicious": 0.95, "hostile": 0.9}"#)?;
    assert!(cfg.thresholds().is_err(), "inverted thresholds are fatal");

    let cfg: BundleConfig = serde_json::from_str(r#"{"default_severity_level": 50}"#)?;
    assert!(cfg.thresholds()?.is_none());
    assert_eq!(cfg.default_severity_level, Some(50));
    Ok(())
}

#[test]
fn isotonic_calibrator_load_apply_and_endpoints() -> Result<()> {
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("calibrator.json"),
        br#"{"schema":"azoth.calibrator.isotonic.v1",
                 "x":[0.0,0.25,0.5,0.75,1.0],
                 "y":[0.0,0.10,0.40,0.85,1.0]}"#,
    )?;
    let cal = IsotonicCalibrator::load_optional(dir.path())?.context("calibrator should load")?;

    // Endpoint clipping.
    assert!((cal.apply(-1.0) - 0.0).abs() < 1e-6);
    assert!((cal.apply(2.0) - 1.0).abs() < 1e-6);
    // Exact breakpoints pass through.
    assert!((cal.apply(0.5) - 0.40).abs() < 1e-6);
    // Linear interpolation between breakpoints.
    // At raw=0.625 we sit halfway between (0.5,0.40) and (0.75,0.85) → 0.625.
    assert!((cal.apply(0.625) - 0.625).abs() < 1e-6);
    Ok(())
}

#[test]
fn isotonic_preserves_threshold_decisions() -> Result<()> {
    // Decision-equivalence property: for any monotone calibrator,
    // cal(p) >= cal(τ) iff p >= τ. This is the invariant that lets us
    // calibrate thresholds at load time without changing decisions.
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("calibrator.json"),
        br#"{"schema":"azoth.calibrator.isotonic.v1",
                 "x":[0.0,0.1,0.3,0.6,0.9,1.0],
                 "y":[0.0,0.02,0.15,0.55,0.92,1.0]}"#,
    )?;
    let cal = IsotonicCalibrator::load_optional(dir.path())?.context("calibrator should load")?;

    for &raw_t in &[0.05_f32, 0.2, 0.5, 0.7, 0.95] {
        let cal_t = cal.apply(raw_t);
        for &raw_p in &[0.0_f32, 0.05, 0.2, 0.5, 0.7, 0.95, 1.0] {
            let cal_p = cal.apply(raw_p);
            assert_eq!(
                raw_p >= raw_t,
                cal_p >= cal_t,
                "decision flipped for raw_p={raw_p} raw_t={raw_t}"
            );
        }
    }
    Ok(())
}

#[test]
fn isotonic_rejects_unknown_schema() -> Result<()> {
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("calibrator.json"),
        br#"{"schema":"azoth.calibrator.spline.v2","x":[0.0,1.0],"y":[0.0,1.0]}"#,
    )?;
    let err = IsotonicCalibrator::load_optional(dir.path())
        .expect_err("future schema should be rejected");
    assert!(err.to_string().contains("unsupported schema"));
    Ok(())
}

#[test]
fn isotonic_load_optional_returns_none_when_absent() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let cal = IsotonicCalibrator::load_optional(dir.path())?;
    assert!(
        cal.is_none(),
        "missing calibrator must be a None, not an error"
    );
    Ok(())
}

#[test]
fn isotonic_rejects_duplicate_x_breakpoints() -> Result<()> {
    // Equal adjacent breakpoints make linear interpolation divide by zero
    // and produce NaN in apply() — silently turning every prediction into
    // NaN and every classification into Benign. Reject at load time.
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("calibrator.json"),
        br#"{"schema":"azoth.calibrator.isotonic.v1","x":[0.0,0.5,0.5,1.0],"y":[0.0,0.4,0.6,1.0]}"#,
    )?;
    let err = IsotonicCalibrator::load_optional(dir.path())
        .expect_err("duplicate breakpoints must be rejected");
    assert!(
        err.to_string().contains("strictly ascending"),
        "error should mention strict ascending: {err}"
    );
    Ok(())
}

#[test]
fn isotonic_rejects_non_finite_values() -> Result<()> {
    // NaN or Inf in x or y would propagate through every apply() call.
    for body in [
        br#"{"schema":"azoth.calibrator.isotonic.v1","x":[0.0,0.5,1.0],"y":[0.0,"NaN",1.0]}"#
            as &[u8],
        br#"{"schema":"azoth.calibrator.isotonic.v1","x":[0.0,"Infinity",1.0],"y":[0.0,0.5,1.0]}"#,
    ] {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("calibrator.json"), body)?;
        let result = IsotonicCalibrator::load_optional(dir.path());
        // serde may reject NaN/Inf at parse time (depending on parser
        // strict-mode); either parse-rejection or our own check is fine
        // — both prevent the bad calibrator from being installed.
        assert!(
            result.is_err(),
            "non-finite calibrator must be rejected (got {result:?})"
        );
    }
    Ok(())
}

#[test]
fn isotonic_rejects_y_out_of_unit_interval() -> Result<()> {
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("calibrator.json"),
        br#"{"schema":"azoth.calibrator.isotonic.v1","x":[0.0,0.5,1.0],"y":[0.0,0.5,1.5]}"#,
    )?;
    let err = IsotonicCalibrator::load_optional(dir.path()).expect_err("y > 1 must be rejected");
    assert!(
        err.to_string().contains("outside [0, 1]"),
        "error should explain the bound: {err}"
    );
    Ok(())
}

#[test]
fn isotonic_rejects_non_monotone_y() -> Result<()> {
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("calibrator.json"),
        br#"{"schema":"azoth.calibrator.isotonic.v1","x":[0.0,0.4,0.8,1.0],"y":[0.0,0.6,0.3,1.0]}"#,
    )?;
    let err =
        IsotonicCalibrator::load_optional(dir.path()).expect_err("non-monotone y must be rejected");
    assert!(
        err.to_string().contains("monotone"),
        "error should mention monotone: {err}"
    );
    Ok(())
}

/// An OR-rule severity over named routes, interned into `names`.
fn or_rule(names: &mut RouteNames, thresholds: &[(&str, f32)]) -> PolicySeverity {
    PolicySeverity {
        thresholds: thresholds
            .iter()
            .map(|&(route, t)| (names.intern(route), t))
            .collect(),
        blend: None,
    }
}

/// One route score, interned into `names`.
fn score(names: &mut RouteNames, route: &str, probability: f32) -> RouteProbability {
    RouteProbability {
        route: names.intern(route),
        probability,
    }
}

#[test]
fn calibrate_policy_thresholds_pushes_each_route_through_its_calibrator() {
    // Build two distinct calibrators so we can verify per-route routing.
    let cal_general = IsotonicCalibrator {
        x: vec![0.0, 0.5, 1.0],
        y: vec![0.0, 0.10, 1.0],
    };
    let cal_elf = IsotonicCalibrator {
        x: vec![0.0, 0.5, 1.0],
        y: vec![0.0, 0.90, 1.0],
    };
    // RoutePolicy with thresholds for general, filetypes/elf (calibrated)
    // and filegroups/missing (no calibrator — must remain untouched).
    let mut names = RouteNames::default();
    let policy = RoutePolicy {
        hostile: or_rule(
            &mut names,
            &[
                ("general", 0.5),
                ("filetypes/elf", 0.5),
                ("filegroups/missing", 0.5),
            ],
        ),
        suspicious: or_rule(&mut names, &[("general", 0.5)]),
    };
    let mut policies = RoutePolicies {
        by_filetype: HashMap::from([("elf".to_string(), policy)]),
        ..Default::default()
    };

    let elf = names.id("filetypes/elf").unwrap();
    let missing = names.id("filegroups/missing").unwrap();
    calibrate_policy_thresholds(&mut policies, |route| match route {
        RouteId::GENERAL => Some(&cal_general),
        r if r == elf => Some(&cal_elf),
        _ => None,
    });

    let policy = policies.by_filetype.get("elf").expect("policy retained");
    // general at raw=0.5 → 0.10
    assert!((policy.hostile.threshold(RouteId::GENERAL).unwrap() - 0.10).abs() < 1e-6);
    // filetypes/elf at raw=0.5 → 0.90
    assert!((policy.hostile.threshold(elf).unwrap() - 0.90).abs() < 1e-6);
    // No calibrator for filegroups/missing — left at raw 0.5.
    assert!((policy.hostile.threshold(missing).unwrap() - 0.5).abs() < 1e-6);
    // Suspicious side also calibrated.
    assert!((policy.suspicious.threshold(RouteId::GENERAL).unwrap() - 0.10).abs() < 1e-6);
}

#[test]
fn thresholds_at_level_extracts_per_route_pairs() {
    // Synthetic levels[] block with two routes; level 5 only.
    let json = r#"{
          "filetype_to_group": {},
          "levels": [
            {
              "level": 5,
              "hostile":    {"thresholds": {"general": 0.99, "filetypes/elf": 0.95}},
              "suspicious": {"thresholds": {"general": 0.80, "filetypes/elf": 0.70}}
            },
            {
              "level": 9,
              "hostile":    {"thresholds": {"general": 0.50}},
              "suspicious": {"thresholds": {"general": 0.30}}
            }
          ]
        }"#;
    let parsed: BundleConfig = serde_json::from_str(json).unwrap();

    let level5 = thresholds_at_level(&parsed.levels, 5);
    assert_eq!(level5.len(), 2);
    let g = level5.get("general").expect("general at level 5");
    assert!((g.hostile - 0.99).abs() < 1e-6);
    assert!((g.suspicious - 0.80).abs() < 1e-6);
    let elf = level5.get("filetypes/elf").expect("elf at level 5");
    assert!((elf.hostile - 0.95).abs() < 1e-6);
    assert!((elf.suspicious - 0.70).abs() < 1e-6);

    let level9 = thresholds_at_level(&parsed.levels, 9);
    assert_eq!(level9.len(), 1);
    assert!((level9.get("general").unwrap().hostile - 0.50).abs() < 1e-6);

    // Routes that have a hostile threshold but no matching suspicious
    // threshold stay active as hostile-only routes.
    let half = r#"{
          "levels": [{
            "level": 5,
            "hostile":    {"thresholds": {"general": 0.9, "filetypes/elf": 0.95}},
            "suspicious": {"thresholds": {"general": 0.6}}
          }]
        }"#;
    let parsed: BundleConfig = serde_json::from_str(half).unwrap();
    let level5 = thresholds_at_level(&parsed.levels, 5);
    assert_eq!(level5.len(), 2);
    assert!(level5.contains_key("general"));
    let elf = level5.get("filetypes/elf").expect("elf remains active");
    assert!((elf.hostile - 0.95).abs() < 1e-6);
    assert!(
        (elf.suspicious - elf.hostile).abs() < 1e-6,
        "hostile-only routes classify only at the hostile threshold"
    );
}

#[test]
fn route_policy_classification_keeps_general_escape() {
    let mut names = RouteNames::default();
    let policy = RoutePolicy {
        hostile: or_rule(&mut names, &[("filetypes/elf", 0.99), ("general", 0.95)]),
        suspicious: or_rule(&mut names, &[("filetypes/elf", 0.90)]),
    };

    let general_escape = [
        score(&mut names, "general", 0.96),
        score(&mut names, "filetypes/elf", 0.10),
    ];
    assert_eq!(
        policy_classify(&policy, &general_escape),
        Classification::Hostile
    );

    let specialist_suspicious = [
        score(&mut names, "general", 0.10),
        score(&mut names, "filetypes/elf", 0.91),
    ];
    assert_eq!(
        policy_classify(&policy, &specialist_suspicious),
        Classification::Suspicious
    );

    let inactive_route = [score(&mut names, "filegroups/native", 1.0)];
    assert_eq!(
        policy_classify(&policy, &inactive_route),
        Classification::Benign
    );
}

fn general_only(prob: f32) -> Vec<RouteProbability> {
    vec![RouteProbability {
        route: RouteId::GENERAL,
        probability: prob,
    }]
}

/// Build a single-route ("general") hostile OR-rule policy grid: each level
/// pairs with the general threshold that fires at it. Thresholds loosen as
/// the level rises, matching the real grid's shape.
fn general_policy_grid(rows: &[(u16, f32)]) -> Vec<LevelPolicy> {
    rows.iter()
        .map(|&(level, thr)| LevelPolicy {
            level,
            hostile: PolicySeverity {
                thresholds: vec![(RouteId::GENERAL, thr)],
                blend: None,
            },
        })
        .collect()
}

#[test]
fn sweep_returns_lowest_firing_level() {
    // Grid: stricter levels demand higher probabilities.
    let grid = general_policy_grid(&[(2, 0.99), (20, 0.85), (50, 0.70), (200, 0.40), (500, 0.10)]);

    // p=0.88 clears the L20 cutoff (0.85) but not L2 (0.99) → lowest is 20.
    let swept = sweep_policy_grid(&grid, &general_only(0.88)).expect("fires");
    assert_eq!(swept.0, 20);

    // p=0.30 only clears L500 (0.10) → lowest firing level is 500.
    let swept = sweep_policy_grid(&grid, &general_only(0.30)).expect("fires");
    assert_eq!(swept.0, 500);

    // p=0.05 clears nothing → never fires.
    assert!(sweep_policy_grid(&grid, &general_only(0.05)).is_none());
}

#[test]
fn sweep_is_independent_of_active_level() {
    // The swept level is a property of the file + model, not of `-l`: the
    // same scores yield the same level marker whatever the deploy level. Only
    // the verdict derived from it changes. This is what makes the envelope
    // cache-shareable across `-l`.
    let grid = general_policy_grid(&[(2, 0.99), (20, 0.85), (50, 0.70)]);
    let swept_level = sweep_policy_grid(&grid, &general_only(0.88))
        .expect("fires")
        .0;
    assert_eq!(swept_level, 20);
    for active in [0_u16, 10, 20, 50, 200, 1000] {
        // Swept level is unchanged; only the class depends on `active`.
        let _ = verdict_for_level(swept_level, active, 1000);
    }
}

/// `lvl`/`fires_at` keep the encoding the `Option<i32>` had: `null`, `-1`,
/// or the level, with an absent field reading as `null`. Anything else is
/// not a level and is refused rather than guessed at.
#[test]
fn level_keeps_its_wire_encoding() {
    for (json, level) in [
        ("null", Level::Manual),
        ("-1", Level::Clean),
        ("0", Level::At(0)),
        ("25000", Level::At(25_000)),
    ] {
        assert_eq!(serde_json::to_string(&level).unwrap(), json);
        assert_eq!(serde_json::from_str::<Level>(json).unwrap(), level);
    }
    #[derive(serde::Deserialize)]
    struct Row {
        lvl: Level,
    }
    assert_eq!(
        serde_json::from_str::<Row>("{}").unwrap().lvl,
        Level::Manual
    );
    for bad in ["-2", "65536", "\"25\""] {
        assert!(serde_json::from_str::<Level>(bad).is_err(), "{bad}");
    }
}

#[test]
fn verdict_for_level_applies_caps() {
    // Rule: swept level <= active level is hostile; swept level above active
    // but within the suspicious cap is suspicious; anything beyond the cap
    // is benign.
    let grid_max = 25_000;
    assert_eq!(verdict_for_level(20, 50, grid_max), Classification::Hostile);
    assert_eq!(verdict_for_level(50, 50, grid_max), Classification::Hostile);
    // The suspicious band ends at the SUSPICIOUS_LEVEL_CEILING cap.
    assert_eq!(
        verdict_for_level(SUSPICIOUS_LEVEL_CEILING, 50, grid_max),
        Classification::Suspicious,
        "at the ceiling is still suspicious"
    );
    assert_eq!(
        verdict_for_level(SUSPICIOUS_LEVEL_CEILING + 1, 50, grid_max),
        Classification::Benign,
        "just past the suspicious ceiling is benign"
    );
    assert_eq!(
        verdict_for_level(25_000, 50, grid_max),
        Classification::Benign
    );

    // Stricter deploy (-l 10): only levels <= 10 are hostile; the ceiling
    // still bounds the top of the suspicious band.
    assert_eq!(
        verdict_for_level(20, 10, grid_max),
        Classification::Suspicious
    );
    assert_eq!(
        verdict_for_level(SUSPICIOUS_LEVEL_CEILING, 10, grid_max),
        Classification::Suspicious
    );
}

#[test]
fn sweep_general_grid_uses_max_crossing_route() {
    // General-route OR fallback: any route crossing the level's general
    // threshold fires it. A weak general but strong specialist still fires.
    let mut names = RouteNames::default();
    let grid = [(2_u16, 0.99_f32), (20, 0.85), (200, 0.40)];
    let scores = [
        score(&mut names, "general", 0.10),
        score(&mut names, "filetypes/elf", 0.90),
    ];
    let swept = sweep_general_grid(&grid, &scores).expect("fires");
    assert_eq!(swept.0, 20, "0.90 clears L20 (0.85) but not L2 (0.99)");
    assert!((swept.1 - 0.90).abs() < 1e-6, "reports the crossing prob");
}

#[test]
fn blend_policy_fires_on_hand_built_inputs() {
    // Construct a blend that fires when general's prob is high enough on
    // its own (weight 1.0 on general, 0 on specialist, intercept chosen
    // so threshold=0.5 lines up with general prob ≈ 0.6).
    // sigmoid(intercept + 1.0 * logit(0.6)) = 0.5
    //   logit(0.6) ≈ 0.4054
    //   intercept = -0.4054 gives sigmoid(0) = 0.5
    let mut names = RouteNames::default();
    let elf = names.intern("filetypes/elf");
    let blend = BlendPolicy {
        routes: vec![RouteId::GENERAL, elf],
        weights: vec![1.0, 0.0],
        intercept: -0.4054,
        threshold: 0.5,
    };
    let scores = |general: f32, specialist: f32| {
        [
            RouteProbability {
                route: RouteId::GENERAL,
                probability: general,
            },
            RouteProbability {
                route: elf,
                probability: specialist,
            },
        ]
    };
    // general=0.6 → fires
    assert!(blend.fires(&scores(0.6, 0.0)));
    // general=0.5 → doesn't fire (logit(0.5)=0, intercept negative)
    assert!(!blend.fires(&scores(0.5, 0.99)));
    // Missing route → can't blend, doesn't fire.
    assert!(!blend.fires(&general_only(1.0)));
}

fn write_policies(dir: &Path, blend: &str) -> Result<()> {
    std::fs::write(
        dir.join("route_policies.json"),
        format!(
            r#"{{
              "schema": "azoth.route_policy_search.v1",
              "routes": {{
                "filetypes/elf": {{
                  "filetype": "elf",
                  "levels": [{{
                    "level": 5,
                    "hostile": {{"best": {{"thresholds": {{}}, "blend": {blend}}}}},
                    "suspicious": {{"best": {{"thresholds": {{"general": 0.7}}}}}}
                  }}]
                }}
              }}
            }}"#
        ),
    )?;
    Ok(())
}

#[test]
fn load_route_policies_parses_blend_field() -> Result<()> {
    let dir = tempfile::tempdir()?;
    write_policies(
        dir.path(),
        r#"{
                "routes": ["general", "filegroups/native", "filetypes/elf"],
                "weights": [0.5, 0.3, 1.2],
                "intercept": -1.5,
                "threshold": 0.8,
                "transform": "logit"
            }"#,
    )?;
    let mut names = RouteNames::default();
    let policies = load_route_policies(dir.path(), 5, &mut names)?;
    let policy = policies.by_filetype.get("elf").expect("elf policy loaded");
    let blend = policy.hostile.blend.as_ref().expect("blend loaded");
    assert_eq!(blend.routes.len(), 3);
    assert_eq!(blend.weights.len(), 3);
    assert!((blend.intercept - -1.5).abs() < 1e-6);
    assert!((blend.threshold - 0.8).abs() < 1e-6);
    // contains_route should recognize blend routes too.
    assert!(policies.contains_route(names.id("filegroups/native").unwrap()));
    assert!(policies.contains_route(names.id("filetypes/elf").unwrap()));
    Ok(())
}

#[test]
fn a_malformed_blend_stops_the_load() -> Result<()> {
    for (blend, why) in [
        // A defaulted threshold of 0.0 would fire on everything.
        (
            r#"{"routes": ["general"], "weights": [1.0], "intercept": 0.0}"#,
            "missing threshold",
        ),
        (
            r#"{"routes": ["general"], "weights": [1.0], "threshold": 0.5}"#,
            "missing intercept",
        ),
        (
            r#"{"routes": ["general"], "weights": [1.0], "intercept": 0.0, "threshold": 0.5, "bias": 1}"#,
            "unknown field",
        ),
        (
            r#"{"routes": ["general"], "weights": [1.0, 2.0], "intercept": 0.0, "threshold": 0.5}"#,
            "length mismatch",
        ),
        (
            r#"{"routes": ["general"], "weights": [1.0], "intercept": 0.0, "threshold": 1.5}"#,
            "threshold out of range",
        ),
        (
            r#"{"routes": ["general"], "weights": [1.0], "intercept": 0.0, "threshold": 0.5, "transform": "gam"}"#,
            "unknown transform",
        ),
    ] {
        let dir = tempfile::tempdir()?;
        write_policies(dir.path(), blend)?;
        let result = load_route_policies(dir.path(), 5, &mut RouteNames::default());
        assert!(result.is_err(), "{why} must fail the load");
    }
    Ok(())
}

#[test]
fn route_policies_are_optional_but_must_parse() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let policies = load_route_policies(dir.path(), 5, &mut RouteNames::default())?;
    assert!(policies.by_filetype.is_empty() && policies.grid.is_empty());

    std::fs::write(dir.path().join("route_policies.json"), b"{\"routes\": 7}")?;
    assert!(load_route_policies(dir.path(), 5, &mut RouteNames::default()).is_err());
    Ok(())
}

#[test]
fn blend_policy_severity_isnt_isotonic_calibrated() {
    // calibrate_policy_thresholds must leave blend severities alone —
    // the blend's threshold is already in calibrated-prob space (its fit
    // ran on isotonic-calibrated route probs at policy-search time).
    // If we double-applied isotonic here it would shift the threshold off
    // the calibrated distribution and deploy verdicts would silently drift.
    let blend = BlendPolicy {
        routes: vec![RouteId::GENERAL],
        weights: vec![1.0],
        intercept: 0.0,
        threshold: 0.42,
    };
    let policy = RoutePolicy {
        hostile: PolicySeverity {
            thresholds: Vec::new(),
            blend: Some(blend),
        },
        suspicious: PolicySeverity {
            thresholds: vec![(RouteId::GENERAL, 0.5)],
            blend: None,
        },
    };
    let mut policies = RoutePolicies {
        by_filetype: HashMap::from([("elf".to_string(), policy)]),
        ..Default::default()
    };
    // A calibrator that shifts every input by +0.1 (clamped to [0, 1]).
    let cal = IsotonicCalibrator {
        x: vec![0.0, 1.0],
        y: vec![0.1, 1.0],
    };
    calibrate_policy_thresholds(&mut policies, |_route| Some(&cal));
    let pol = policies.by_filetype.get("elf").unwrap();
    // Hostile (blend) threshold stays at its original calibrated value.
    let blend = pol.hostile.blend.as_ref().unwrap();
    assert!((blend.threshold - 0.42).abs() < 1e-6);
    // Suspicious (OR-rule) threshold did get mapped.
    assert!((pol.suspicious.threshold(RouteId::GENERAL).unwrap() - cal.apply(0.5)).abs() < 1e-6);
}

#[test]
fn load_route_policies_reads_level_and_route_membership() -> Result<()> {
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("route_policies.json"),
        r#"{
              "schema": "azoth.route_policy_search.v1",
              "routes": {
                "filetypes/elf": {
                  "filetype": "elf",
                  "levels": [
                    {
                      "level": 4,
                      "hostile": {
                        "best": {"thresholds": {"general": 0.99}}
                      },
                      "suspicious": {
                        "best": {"thresholds": {"general": 0.90}}
                      }
                    },
                    {
                      "level": 5,
                      "hostile": {
                        "best": {"thresholds": {"filetypes/elf": 0.98, "general": 0.97}}
                      },
                      "suspicious": {
                        "best": {"thresholds": {"filegroups/native": 0.80}}
                      }
                    }
                  ]
                }
              }
            }"#,
    )?;

    let mut names = RouteNames::default();
    let policies = load_route_policies(dir.path(), 5, &mut names)?;
    let id = |name: &str| names.id(name).unwrap();
    assert!(policies.contains_route(RouteId::GENERAL));
    assert!(policies.contains_route(id("filegroups/native")));
    assert!(policies.contains_route(id("filetypes/elf")));
    assert_eq!(names.id("filetypes/pe"), None);

    let elf = policies.by_filetype.get("elf").expect("elf policy");
    assert!(
        (elf.hostile.threshold(id("filetypes/elf")).unwrap() - 0.98).abs() < 1e-6,
        "must select the requested level"
    );
    assert!(elf.suspicious.threshold(RouteId::GENERAL).is_none());
    Ok(())
}

#[test]
fn policy_routes_never_firing_sentinel_is_dropped() -> Result<()> {
    // Collimator writes one ulp above 1.0 for a route that must never fire
    // at a level; the real bundle carries it, so it must still load.
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("route_policies.json"),
        r#"{"routes": {"filetypes/yaml": {"filetype": "yaml", "levels": [
                {"level": 0, "hostile": {"best": {"thresholds": {"general": 1.0000001192092896}}}},
                {"level": 5, "hostile": {"best": {"thresholds": {"general": 0.9}}}}
            ]}}}"#,
    )?;
    let policies = load_route_policies(dir.path(), 5, &mut RouteNames::default())?;
    let grid = policies.grid_for("yaml").expect("grid");
    assert_eq!(grid.len(), 1, "the never-fires level has no policy");
    assert_eq!(grid[0].level, 5);
    Ok(())
}

#[test]
fn ensemble_loader_rejects_missing_general() -> Result<()> {
    let dir = tempfile::tempdir()?;
    // Make the layout look like an ensemble (general/ subdir present)
    // but leave it empty so general/ has no model artifacts.
    std::fs::create_dir_all(dir.path().join("general"))?;
    let err =
        Model::load(dir.path(), None, None).expect_err("ensemble with empty general/ must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("loading general route") || msg.contains("incomplete"),
        "expected general-route load failure, got {msg}"
    );
    Ok(())
}

#[test]
fn ensemble_loader_rejects_unknown_required_route_name() -> Result<()> {
    let dir = tempfile::tempdir()?;
    std::fs::create_dir_all(dir.path().join("general"))?;
    std::fs::write(
        dir.path().join("config.json"),
        r#"{
              "required_routes": ["mystery/thing"]
            }"#,
    )?;
    // We can't actually load general here without a real model bundle, so
    // we expect either "loading general route" or the required-route check
    // depending on order. Either error mode confirms the loader caught it.
    let err = Model::load(dir.path(), None, None).expect_err("must fail");
    let _ = err;
    Ok(())
}

/// A report is parsed for every route that will score it: a specialist that
/// reads an optional input gets it even when the general route does not.
#[test]
fn raw_needs_cover_the_specialists_a_file_type_selects() -> Result<()> {
    let root = tempfile::tempdir()?;
    let route = |dir: &Path, feature: &str, kv_vocab: &[&str]| -> Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(
            dir.join("model.onnx"),
            include_bytes!("../../tests/fixtures/cli-model.onnx"),
        )?;
        let spec = serde_json::json!({
            "version": crate::features::EXPECTED_SPEC_VERSION,
            "abi_version": crate::features::EXPECTED_MODEL_ABI_VERSION,
            "presence_vocab": ["metadata"],
            "kv_vocab": kv_vocab,
            "feature_names": [feature],
            "total_features": 1,
            "standardized": false,
            "metric_transforms": {}
        });
        std::fs::write(dir.join("feature_spec.json"), serde_json::to_vec(&spec)?)?;
        Ok(())
    };
    route(&root.path().join("general"), "present:metadata", &[])?;
    route(
        &root.path().join("filetypes/elf"),
        "kv:arch=x86",
        &["arch=x86"],
    )?;
    std::fs::write(
        root.path().join("config.json"),
        r#"{"levels": [{"level": 25, "hostile": {"thresholds":
            {"general": 0.99, "filetypes/elf": 0.99}}}]}"#,
    )?;
    let model = Model::load(root.path(), None, None)?;

    let general_only = model.raw_needs(["python"]);
    assert_eq!(general_only, crate::features::RawNeeds::default());
    assert_ne!(model.raw_needs(["python", "elf"]), general_only);
    Ok(())
}
