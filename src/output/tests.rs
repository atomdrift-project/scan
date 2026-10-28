use super::*;

#[test]
fn unresolved_dependencies_preclude_a_clean_summary() {
    let mut summary = ScanSummary {
        total_files: 1,
        hostile: 0,
        suspicious: 0,
        benign: 1,
        errors: 0,
        duration_ms: 1,
        static_flags: 0,
        dependency_attention: crate::engine::DependencyAttention {
            gaps: 2,
            flagged_candidates: 1,
        },
    };
    let line = summary_status_line(&summary);
    assert!(line.contains("2 dependency gaps") && line.contains("1 flagged candidate"));
    assert!(!line.contains("all graded benign") && !line.contains('\u{2713}'));
    summary.dependency_attention = Default::default();
    assert!(summary_status_line(&summary).contains("all graded benign"));
    summary.static_flags = 1;
    let line = summary_status_line(&summary);
    assert!(line.contains("1 with severe static findings"));
    assert!(!line.contains("all graded benign"));
}

#[test]
fn bloom_mark_from_decision() {
    use crate::bloom_repo::Decision;
    // Skip (known-good) now marks a scanned-anyway file ✓; Unknown is
    // unremarkable (matched neither set) and carries no mark.
    assert_eq!(
        BloomMark::from_decision(Decision::Skip),
        Some(BloomMark::KnownGood)
    );
    assert_eq!(BloomMark::from_decision(Decision::Unknown), None);
    assert_eq!(
        BloomMark::from_decision(Decision::KnownBad),
        Some(BloomMark::KnownBad)
    );
    assert_eq!(
        BloomMark::from_decision(Decision::Conflicted),
        Some(BloomMark::Conflicted)
    );
    assert_eq!(BloomMark::KnownGood.tiny_str(), "known-good");
    assert_eq!(BloomMark::KnownBad.tiny_str(), "known-bad");
    assert_eq!(BloomMark::Conflicted.tiny_str(), "conflicted");
    // Each state is visually distinct.
    assert_ne!(BloomMark::KnownGood.glyph(), BloomMark::KnownBad.glyph());
    assert_ne!(BloomMark::KnownBad.glyph(), BloomMark::Conflicted.glyph());
    assert!(!BloomMark::KnownGood.forces_terminal_display());
    assert!(BloomMark::KnownBad.forces_terminal_display());
    assert!(BloomMark::Sighted.forces_terminal_display());
    assert!(BloomMark::Conflicted.forces_terminal_display());
}

#[test]
fn hash_line_carries_the_bloom_glyph() {
    // The bloom verdict rides the hash line's glyph slot — 🚩 supplants 🧬
    // for a known-bad file — never the verdict badge. (These tests run
    // uncolored, so the line uses the plain text-marker form.)
    let plain = terminal_hash_line("deadbeef", None).expect("hash line");
    let flagged = terminal_hash_line("deadbeef", Some(BloomMark::KnownBad)).expect("hash line");
    assert!(plain.contains("deadbeef"));
    assert!(!plain.contains(BloomMark::KnownBad.glyph()));
    assert!(flagged.contains("deadbeef"));
    assert!(flagged.contains(BloomMark::KnownBad.glyph()));
    // An absent hash renders nothing.
    assert!(terminal_hash_line("", Some(BloomMark::KnownBad)).is_none());
    // The badge itself carries no bloom glyph.
    let (badge, _) = terminal_badge(&Classification::Hostile, 0.99, 0.65, Level::At(0));
    assert!(!badge.contains(BloomMark::KnownBad.glyph()));
}

#[test]
fn human_size_compact_units() {
    assert_eq!(human_size(0), "");
    assert_eq!(human_size(352), "352B");
    assert_eq!(human_size(5 * 1024 + 400), "5.4KB");
    assert_eq!(human_size(44 * 1024 * 1024 / 10 * 10), "44.0MB");
}

#[test]
fn card_markers_share_one_content_column() {
    for marker in [
        "📦",
        "📄",
        "🧬",
        "🚩",
        "📡",
        "🏴",
        "🪪",
        "✨",
        "●",
        "●●",
        "●●●",
    ] {
        let prefix = terminal_marker_prefix(1, marker);
        assert_eq!(
            UnicodeWidthStr::width(prefix.as_str()),
            5,
            "marker {marker:?} produced prefix {prefix:?}"
        );
    }
}

#[test]
fn terminal_trait_grid_is_three_rows_and_keeps_location_tail() {
    let traits = [
        TerminalTrait {
            criticality: cleave::Criticality::Hostile,
            description: "Remote terminal agent".to_string(),
            location: "very-long-hash-named-member-that-needs-truncation.py:114".to_string(),
        },
        TerminalTrait {
            criticality: cleave::Criticality::Suspicious,
            description: "Hidden execution".to_string(),
            location: "runner.py:9".to_string(),
        },
        TerminalTrait {
            criticality: cleave::Criticality::Notable,
            description: "Third trait".to_string(),
            location: "manifest.json".to_string(),
        },
        TerminalTrait {
            criticality: cleave::Criticality::Notable,
            description: "Must not render".to_string(),
            location: "fourth.txt".to_string(),
        },
    ];
    // Assert on the plain rendering, whatever the environment thinks about
    // colour: `CLICOLOR_FORCE` outranks the tty check, so on a host that
    // sets it these rows arrive wrapped in escapes and every prefix and
    // suffix assertion below reads a different string than the one under
    // test.
    colored::control::set_override(false);
    let rendered = terminal_trait_rows(&traits, 72);
    assert_eq!(rendered.lines().count(), 3);
    assert!(rendered.lines().next().unwrap().contains("…"));
    assert!(rendered.lines().next().unwrap().ends_with("py:114"));
    assert!(!rendered.contains("Must not render"));
}

#[test]
fn embedded_branch_uses_two_lines_without_a_closing_rule() {
    // See `terminal_trait_grid_is_three_rows_and_keeps_location_tail`.
    colored::control::set_override(false);
    let body = " \u{25cf}\u{25cf}\u{25cf} Payload behavior";
    let rendered = terminal_embedded_branch(
        &Classification::Hostile,
        "embedded base64 @ 11096",
        "shell",
        11_766,
        body,
        true,
    );
    assert_eq!(rendered.lines().count(), 2);
    assert!(rendered.contains("   \u{2514}\u{2500} embedded base64 @ 11096"));
    assert!(rendered.contains("      \u{25cf}\u{25cf}\u{25cf} Payload behavior"));
    assert!(!rendered.contains('\u{2570}'));
}

/// Piped output: each renderer's plain form, byte for byte, and no
/// escape anywhere.
#[test]
fn plain_renderers_emit_grep_able_text() {
    // See `terminal_trait_grid_is_three_rows_and_keeps_location_tail`.
    colored::control::set_override(false);
    let lines = [
        terminal_artifact_line("pkg.tgz", "npm", 2048, true),
        terminal_identity_line("lodash 4.17.21").unwrap(),
        terminal_registry_line("published 3d ago").unwrap(),
        terminal_reference_status_heading("NOT FETCHED", "left-pad"),
        terminal_reference_locator_row("https://example.com/x"),
        terminal_reference_hash_row("ab12", "pe", 0),
        terminal_trailer(&[crate::explain::Reason {
            feature: "agg:max_crit".to_string(),
            importance: 1.0,
            value: 5.0,
            description: "aggregate: max crit".to_string(),
        }])
        .unwrap(),
        terminal_reference_branch(
            Some((&Classification::Hostile, 0.97)),
            "dependency",
            "package.json",
            "body",
            true,
        ),
    ];
    assert_eq!(
        lines,
        [
            "pkg.tgz \u{00b7} NPM \u{00b7} 2.0KB",
            "   lodash 4.17.21",
            "   published 3d ago",
            "  ↳ NOT FETCHED · left-pad",
            " \u{1f517}  https://example.com/x",
            " \u{1f9ec}  ab12  \u{00b7} PE",
            "\u{00b7} aggregate: max crit",
            "   \u{2514}\u{2500} dependency from package.json \u{00b7} HOSTILE 97%\n     body\n",
        ]
    );
    assert!(lines.iter().all(|l| !l.contains('\x1b')));
}

#[test]
fn a_level_names_both_levelless_states() {
    assert_eq!(Level::At(0).to_string(), "L0");
    assert_eq!(Level::At(250).to_string(), "L250");
    assert_eq!(Level::Clean.to_string(), "clean");
    assert_eq!(Level::Manual.to_string(), "manual");
}

#[test]
fn commas_group_by_threes() {
    assert_eq!(with_commas(0), "0");
    assert_eq!(with_commas(42), "42");
    assert_eq!(with_commas(999), "999");
    assert_eq!(with_commas(1_000), "1,000");
    assert_eq!(with_commas(12_345), "12,345");
    assert_eq!(with_commas(123_456), "123,456");
    assert_eq!(with_commas(1_093_027), "1,093,027");
}

#[test]
fn detection_inventory_keeps_signatures_separate_from_rules() {
    assert_eq!(
        detection_inventory_text(9_099_016, 120_680),
        "\u{00b7} 9,099,016 hashes \u{00b7} 120,680 rules"
    );
}

#[test]
fn display_probability_at_threshold() {
    let thr = 0.65_f32;
    // At zero → 0%
    assert!((display_probability(0.0, thr) - 0.0).abs() < 1e-6);
    // At the deciding threshold → exactly 50%
    assert!((display_probability(thr, thr) - 0.5).abs() < 1e-6);
    // At 1.0 → 100%
    assert!((display_probability(1.0, thr) - 1.0).abs() < 1e-6);
}

#[test]
fn display_probability_monotonic() {
    let thr = 0.65_f32;
    let steps: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
    for pair in steps.windows(2) {
        assert!(
            display_probability(pair[1], thr) > display_probability(pair[0], thr),
            "not monotonic at {} -> {}",
            pair[0],
            pair[1],
        );
    }
}

#[test]
fn display_probability_below_threshold_is_under_50() {
    let thr = 0.65_f32;
    assert!(display_probability(thr - 0.01, thr) < 0.5);
    assert!(display_probability(0.5, thr) < 0.5);
}

#[test]
fn display_probability_uses_custom_threshold() {
    let thr = 0.80_f32;
    assert!((display_probability(thr, thr) - 0.5).abs() < 1e-6);
    assert!(display_probability(0.79, thr) < 0.5);
}

#[test]
fn display_confidence_is_absent_for_unflagged_levels() {
    // Every level that is actually on the grid keeps its confidence: 0 is
    // the tightest and most certain, higher levels are looser and less so.
    assert_eq!(display_confidence(Level::At(0)), Some(100));
    assert_eq!(display_confidence(Level::At(5)), Some(95));
    // No level marker at all → no level-derived confidence. Unchanged.
    assert_eq!(display_confidence(Level::Manual), None);
    // A clean level means the sample cleared no calibrated level. The
    // wire value stays 0 (consumers of the JSON `conf` field see no change)
    // but the display must not reuse it: 0% renders as certainty, and the
    // model expressed absence.
    assert_eq!(level_confidence(Level::Clean), Some(0));
    assert_eq!(display_confidence(Level::Clean), None);
}

#[test]
fn unflagged_badge_reports_distance_to_cutoff_not_zero() {
    // Regression: an npm package whose raw probability was 0.998 against a
    // higher cutoff rendered as `BENIGN[0%]`, printed directly above
    // cleave's own hostile finding for the same file. Zero asserted a
    // certainty the model never expressed. The honest number is how near
    // the deciding cutoff the sample actually sat — here, right on it.
    //
    // Asserted with `contains` so the test holds in both the plain and the
    // colored badge form.
    let (unflagged, _) = terminal_badge(&Classification::Benign, 0.998, 0.999, Level::Clean);
    assert!(
        unflagged.contains("BENIGN") && unflagged.contains("50%"),
        "unflagged badge should show the threshold-relative percentage, got {unflagged:?}"
    );

    // A level that IS on the grid still reports its calibrated confidence,
    // so the fix does not disturb the flagged path.
    let (flagged, _) = terminal_badge(&Classification::Hostile, 0.99, 0.65, Level::At(0));
    assert!(
        flagged.contains("HOSTILE") && flagged.contains("100%"),
        "flagged badge should keep the level-derived confidence, got {flagged:?}"
    );
}

#[test]
fn parse_ymd_known_dates() {
    assert_eq!(parse_ymd("1970-01-01"), Some(0));
    assert_eq!(parse_ymd("2026-06-29"), Some(1_782_691_200));
    // Malformed input yields no timestamp rather than a wrong guess.
    assert_eq!(parse_ymd("2026-13-01"), None);
    assert_eq!(parse_ymd("not-a-date"), None);
    assert_eq!(parse_ymd("2026-06"), None);
}

#[test]
fn date_round_trips_through_epoch() {
    for date in ["1970-01-01", "2000-02-29", "2026-06-29", "2099-12-31"] {
        let epoch = parse_ymd(date).expect("valid date");
        assert_eq!(fmt_date(epoch), date);
    }
}

#[test]
fn freshness_bands_map_to_verdict_colors() {
    let p = palette();
    let same = |a: Rgb, b: Rgb| a.0 == b.0 && a.1 == b.1 && a.2 == b.2;
    assert!(same(freshness_color(0), p.benign));
    assert!(same(freshness_color(172_800), p.benign)); // exactly 48h → still green
    assert!(same(freshness_color(172_801), p.suspicious)); // just past 48h → yellow
    assert!(same(freshness_color(604_800), p.suspicious)); // exactly a week → yellow
    assert!(same(freshness_color(604_801), p.hostile)); // past a week → red
}

#[test]
fn age_formats_by_unit() {
    assert_eq!(fmt_age(-5), "just now");
    assert_eq!(fmt_age(30), "just now");
    assert_eq!(fmt_age(90), "1m ago");
    assert_eq!(fmt_age(3 * 3600), "3h ago");
    assert_eq!(fmt_age(2 * 86_400 + 5 * 3600), "2d ago");
}
