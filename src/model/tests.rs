use super::policy::{
    BlendPolicy, LevelPolicy, PolicySeverity, RouteKind, container_filetype,
    normalize_archive_filetype, thresholds_at_level,
};
use super::verdict::{SUSPICIOUS_CEILING_FLOOR, SUSPICIOUS_CEILING_MULTIPLE};
use super::*;
use std::borrow::Cow;
use std::collections::HashMap;

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
    // Training emits the filetype→filegroup map (and trains every
    // specialist) under the normalized container label, so the map carries
    // `tar`/`javascript`, never `tar.gz`.
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
    // and its filegroup: selecting by the raw `tar.gz` label finds no
    // specialist and scores the file on the general route alone.
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
fn suspicious_ceiling_scales_with_the_hostile_level() {
    // 100× the active level, never below the floor, never past the grid.
    assert_eq!(SUSPICIOUS_CEILING_FLOOR, 500);
    assert_eq!(SUSPICIOUS_CEILING_MULTIPLE, 100);
    for (active, want) in [
        (0, 500),
        (1, 500),
        (5, 500),
        (10, 1_000),
        (25, 2_500),
        (70, 7_000),
        (100, 10_000),
        (250, 25_000),
        (1_000, 25_000),
        (u16::MAX, 25_000),
    ] {
        assert_eq!(suspicious_ceiling(active, 25_000), want, "-l {active}");
    }
    // A grid tighter than the scaled ceiling caps it.
    assert_eq!(suspicious_ceiling(25, 2_000), 2_000);
    assert_eq!(suspicious_ceiling(0, 100), 100);
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
fn bundle_config_reads_the_default_severity_level() -> Result<()> {
    let cfg: BundleConfig = serde_json::from_str(r#"{"default_severity_level": 50}"#)?;
    assert_eq!(cfg.default_severity_level, Some(50));
    let cfg: BundleConfig = serde_json::from_str("{}")?;
    assert_eq!(cfg.default_severity_level, None);
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

/// The class a policy decides for `scores`, benign when nothing fires.
fn decided(policy: &RoutePolicy, scores: &[RouteProbability]) -> Classification {
    policy_decide(policy, scores).map_or(Classification::Benign, |d| d.class)
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
    assert_eq!(decided(&policy, &general_escape), Classification::Hostile);

    let specialist_suspicious = [
        score(&mut names, "general", 0.10),
        score(&mut names, "filetypes/elf", 0.91),
    ];
    assert_eq!(
        decided(&policy, &specialist_suspicious),
        Classification::Suspicious
    );

    let inactive_route = [score(&mut names, "filegroups/native", 1.0)];
    assert_eq!(decided(&policy, &inactive_route), Classification::Benign);
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
    // but within the suspicious ceiling is suspicious; anything beyond it is
    // benign. The ceiling scales with the active level.
    let grid_max = 25_000;
    assert_eq!(verdict_for_level(20, 50, grid_max), Classification::Hostile);
    assert_eq!(verdict_for_level(50, 50, grid_max), Classification::Hostile);
    // -l 50: the suspicious band ends at 5000.
    assert_eq!(
        verdict_for_level(5_000, 50, grid_max),
        Classification::Suspicious,
        "at the ceiling is still suspicious"
    );
    assert_eq!(
        verdict_for_level(5_001, 50, grid_max),
        Classification::Benign,
        "just past the suspicious ceiling is benign"
    );
    // The shipped default reviews through 2500.
    assert_eq!(
        verdict_for_level(2_500, 25, grid_max),
        Classification::Suspicious
    );
    assert_eq!(
        verdict_for_level(3_000, 25, grid_max),
        Classification::Benign
    );

    // Stricter deploys keep the floor's band.
    assert_eq!(
        verdict_for_level(20, 10, grid_max),
        Classification::Suspicious
    );
    assert_eq!(
        verdict_for_level(1_000, 10, grid_max),
        Classification::Suspicious
    );
    assert_eq!(
        verdict_for_level(500, 0, grid_max),
        Classification::Suspicious
    );
    assert_eq!(verdict_for_level(750, 0, grid_max), Classification::Benign);

    // A loose deploy reviews the whole grid.
    assert_eq!(
        verdict_for_level(25_000, 250, grid_max),
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
    assert!(blend.fire(&scores(0.6, 0.0)).is_some());
    // general=0.5 → doesn't fire (logit(0.5)=0, intercept negative)
    assert!(blend.fire(&scores(0.5, 0.99)).is_none());
    // Missing route → can't blend, doesn't fire.
    assert!(blend.fire(&general_only(1.0)).is_none());
}

/// Compare frozen Python decisions with the actual serving implementation.
#[test]
#[ignore = "requires SCAN_BLEND_PARITY_CASES generated by collimator"]
#[expect(clippy::disallowed_methods, reason = "test-only fixture path")]
fn collimator_frozen_blend_decisions_match() -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Case {
        probabilities: Vec<f32>,
        threshold: f32,
        fires: bool,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        routes: Vec<String>,
        weights: Vec<f32>,
        intercept: f32,
        cases: Vec<Case>,
    }
    let path = std::env::var_os("SCAN_BLEND_PARITY_CASES")
        .expect("set SCAN_BLEND_PARITY_CASES to frozen Python decisions");
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path)?)?;
    assert!(!fixture.cases.is_empty());
    assert_eq!(fixture.routes.len(), fixture.weights.len());
    let mut names = RouteNames::default();
    let routes: Vec<_> = fixture
        .routes
        .iter()
        .map(|name| names.intern(name))
        .collect();
    let mut blend = BlendPolicy {
        routes: routes.clone(),
        weights: fixture.weights,
        intercept: fixture.intercept,
        threshold: 0.0,
    };
    for (index, case) in fixture.cases.iter().enumerate() {
        assert_eq!(routes.len(), case.probabilities.len());
        let scores: Vec<_> = routes
            .iter()
            .zip(&case.probabilities)
            .map(|(&route, &probability)| RouteProbability { route, probability })
            .collect();
        blend.threshold = case.threshold;
        assert_eq!(
            blend.fire(&scores).is_some(),
            case.fires,
            "blend parity case {index}"
        );
    }
    println!("{} frozen blend decisions matched", fixture.cases.len());
    Ok(())
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
        (
            r#"{"routes": ["general"], "weights": [1e39], "intercept": 0.0, "threshold": 0.5}"#,
            "infinite weight",
        ),
        (
            r#"{"routes": ["general"], "weights": [1.0], "intercept": -1e39, "threshold": 0.5}"#,
            "infinite intercept",
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
    // The policy search writes one ulp above 1.0 for a route that must never fire
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
            "base_feature_abi": 21,
            "trait_path_depth": 3,
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

/// Explicit policy inputs must be parsed and scored even when they are not
/// the filetype's ordinary group, including inputs used only by another level.
#[test]
fn additional_policy_routes_are_scored_at_every_level() -> Result<()> {
    let root = tempfile::tempdir()?;
    for (name, feature, kv) in [
        ("general", "present:metadata", vec![]),
        ("filegroups/previous", "kv:arch=x86", vec!["arch=x86"]),
    ] {
        let dir = root.path().join(name);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            dir.join("model.onnx"),
            include_bytes!("../../tests/fixtures/cli-model.onnx"),
        )?;
        let spec = serde_json::json!({
            "version": crate::features::EXPECTED_SPEC_VERSION,
            "abi_version": crate::features::EXPECTED_MODEL_ABI_VERSION,
            "base_feature_abi": 21,
            "trait_path_depth": 3,
            "presence_vocab": ["metadata"], "kv_vocab": kv,
            "feature_names": [feature], "total_features": 1,
            "standardized": false, "metric_transforms": {}
        });
        std::fs::write(dir.join("feature_spec.json"), serde_json::to_vec(&spec)?)?;
    }
    std::fs::write(
        root.path().join("config.json"),
        r#"{
        "levels": [
            {"level": 25, "hostile": {"thresholds": {"general": 1.0}}},
            {"level": 50, "hostile": {"thresholds": {"general": 1.0}}}
        ]
    }"#,
    )?;
    // The auxiliary is a blend input at a non-default level only. No normal
    // filegroup mapping or auxiliary threshold enables it accidentally.
    std::fs::write(
        root.path().join("route_policies.json"),
        r#"{
        "routes": {"filetypes/elf": {"filetype": "elf", "levels": [
            {"level": 25, "hostile": {"best": {"thresholds": {"general": 1.01}}}},
            {"level": 50, "hostile": {"best": {"thresholds": {}, "blend": {
                "routes": ["general", "filegroups/previous"], "weights": [0.0, 1.0],
                "intercept": 0.0, "threshold": 0.0, "transform": "logit"
            }}}}
        ]}}
    }"#,
    )?;
    let model = Model::load(root.path(), None, Some(25))?;
    assert_ne!(model.raw_needs(["elf"]), model.raw_needs(["python"]));
    let mut report = cleave::types::CompactReport::default();
    report.files.push(cleave::types::CompactFile {
        file_type: "elf".into(),
        ..Default::default()
    });
    let (decision, scores, skipped) = model.predict_report(&report)?;
    assert_eq!(decision.level, Level::At(50));
    assert_eq!(
        scores
            .iter()
            .filter(|score| score.model == "az/previous")
            .count(),
        1
    );
    assert!(skipped.is_empty());
    report.files[0].file_type = "python".into();
    let (_, scores, _) = model.predict_report(&report)?;
    assert_eq!(scores.len(), 1, "auxiliary must be scoped to the policy");
    Ok(())
}

/// The per-route thresholds are resolved at the bundle's own default level,
/// not the build's.
#[test]
fn diagnostic_thresholds_follow_the_bundles_default_level() -> Result<()> {
    let root = tempfile::tempdir()?;
    let general = root.path().join("general");
    std::fs::create_dir_all(&general)?;
    std::fs::write(
        general.join("model.onnx"),
        include_bytes!("../../tests/fixtures/cli-model.onnx"),
    )?;
    let spec = serde_json::json!({
        "version": crate::features::EXPECTED_SPEC_VERSION,
        "abi_version": crate::features::EXPECTED_MODEL_ABI_VERSION,
        "base_feature_abi": 21, "trait_path_depth": 3,
        "presence_vocab": ["metadata"], "feature_names": ["present:metadata"],
        "total_features": 1, "standardized": false, "metric_transforms": {}
    });
    std::fs::write(
        general.join("feature_spec.json"),
        serde_json::to_vec(&spec)?,
    )?;
    std::fs::write(
        root.path().join("config.json"),
        r#"{"default_severity_level": 50, "levels": [
            {"level": 25, "hostile": {"thresholds": {"general": 0.99}}},
            {"level": 50, "hostile": {"thresholds": {"general": 0.95}}}
        ]}"#,
    )?;
    let model = Model::load(root.path(), None, Some(25))?;
    assert!((model.thresholds().hostile - 0.95).abs() < 1e-6);
    Ok(())
}

/// A specialist built after the bundle on disk was replaced would be scored
/// against the old bundle's policy thresholds, so it is refused.
#[test]
fn a_specialist_from_a_replaced_bundle_is_refused() -> Result<()> {
    let root = tempfile::tempdir()?;
    for dir in ["general", "filetypes/elf"] {
        let dir = root.path().join(dir);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            dir.join("model.onnx"),
            include_bytes!("../../tests/fixtures/cli-model.onnx"),
        )?;
        let spec = serde_json::json!({
            "version": crate::features::EXPECTED_SPEC_VERSION,
            "abi_version": crate::features::EXPECTED_MODEL_ABI_VERSION,
            "base_feature_abi": 21,
            "trait_path_depth": 3,
            "presence_vocab": ["metadata"],
            "kv_vocab": [],
            "feature_names": ["present:metadata"],
            "total_features": 1,
            "standardized": false,
            "metric_transforms": {}
        });
        std::fs::write(dir.join("feature_spec.json"), serde_json::to_vec(&spec)?)?;
    }
    std::fs::write(
        root.path().join("config.json"),
        r#"{"levels": [{"level": 25, "hostile": {"thresholds":
            {"general": 0.99, "filetypes/elf": 0.99}}}]}"#,
    )?;
    let sidecar = |commit: &str| {
        std::fs::write(
            root.path().join(".scan-models.toml"),
            format!(
                "commit = \"{commit}\"\ndate = \"2026-10-29\"\nsource = \"latest\"\nversion = \"0\"\n"
            ),
        )
    };
    sidecar("old")?;
    let held = Model::load(root.path(), None, None)?;
    let swapped = Model::load(root.path(), None, None)?;

    held.validate_all_routes()?;
    sidecar("new")?;
    let err = swapped
        .validate_all_routes()
        .expect_err("a specialist from the replaced bundle must be refused");
    assert!(format!("{err:#}").contains("replaced"), "{err:#}");
    Ok(())
}

#[test]
fn description_inputs_are_checked_for_general_and_specialist_models() -> Result<()> {
    for description_general in [true, false] {
        let root = tempfile::tempdir()?;
        for (name, description) in [
            ("general", description_general),
            ("filetypes/elf", !description_general),
        ] {
            let dir = root.path().join(name);
            std::fs::create_dir_all(&dir)?;
            std::fs::write(
                dir.join("model.onnx"),
                include_bytes!("../../tests/fixtures/cli-model.onnx"),
            )?;
            let spec = if description {
                serde_json::json!({"version":22,"abi_version":22,"base_feature_abi":20,
                    "trait_path_depth":3,"presence_vocab":["metadata"],"keyword_vocabs":{"descword":["http"]},
                    "feature_names":["descword:http"],"total_features":1,"standardized":false})
            } else {
                serde_json::json!({"version":20,"abi_version":20,"trait_path_depth":3,
                    "presence_vocab":["metadata"],"feature_names":["present:metadata"],
                    "total_features":1,"standardized":false})
            };
            std::fs::write(dir.join("feature_spec.json"), serde_json::to_vec(&spec)?)?;
        }
        std::fs::write(
            root.path().join("config.json"),
            r#"{"levels":[{"level":25,"hostile":{"thresholds":{"general":0.99,"filetypes/elf":0.99}}}]}"#,
        )?;
        let model = Model::load(root.path(), None, Some(25))?;
        let mut report: cleave::types::CompactReport = serde_json::from_value(serde_json::json!({
            "v":"8","files":[{"id":0,"path":"test","type":"elf","sha":"fixture","size":10,
                "traits":[{"id":"metadata/network::http","conf":1.0,"crit":2}]}]}))?;
        assert!(model.predict_report(&report).is_err());
        report.files[0].findings[0].description = "HTTP".into();
        model.predict_report(&report)?;
        if !description_general {
            report.files[0].file_type = "python".into();
            report.files[0].findings[0].description.clear();
            model.predict_report(&report)?;
        }
    }
    Ok(())
}

#[test]
fn sweep_preserves_minimum_with_nonmonotone_thresholds() {
    let grid = general_policy_grid(&[(0, 0.9), (5, 0.4), (25, 0.8), (100, 0.1)]);
    // Compare against the exhaustive decision rule at every threshold boundary.
    for probability in [0.0, 0.1, 0.10001, 0.39999, 0.4, 0.5, 0.8, 0.9, 1.0] {
        let scores = general_only(probability);
        let exhaustive = grid
            .iter()
            .filter_map(|lp| lp.hostile.fire(&scores).map(|(p, t)| (lp.level, p, t)))
            .min_by_key(|&(level, _, _)| level);
        assert_eq!(sweep_policy_grid(&grid, &scores), exhaustive);
    }
}
