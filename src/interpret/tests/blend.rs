//! How an LLM grade may move the model's verdict, and never further.

use super::*;

/// A deploy level of `-l 25` over a full grid, so the two band boundaries sit
/// at L25 (suspicious|hostile, conf 92) and L3000 (benign|suspicious, conf 54).
fn levels(fired: Level) -> LevelContext {
    LevelContext {
        fired,
        active: Some(25),
        grid_max: 25_000,
    }
}

/// Evidence with ML placed at `fired` and a trusted, uncorroborated read —
/// the shape for exercising the proximity gate.
fn ev_at(fired: Level) -> Evidence {
    Evidence {
        readable: true,
        analyzer_directed: false,
        hostile_finding: false,
        levels: levels(fired),
    }
}

/// Evidence in manual-threshold mode: with no level table the proximity gate
/// abstains, isolating the *steering* rule from the *crossing* rule so each
/// can be tested on its own.
fn ev_open(readable: bool, analyzer_directed: bool) -> Evidence {
    Evidence {
        readable,
        analyzer_directed,
        hostile_finding: false,
        levels: LevelContext {
            fired: Level::Manual,
            active: None,
            grid_max: 0,
        },
    }
}

/// Every (class, grade, evidence, score) combination — both guard flags, the
/// corroboration hatch, manual mode, and ML positions on either side of each
/// boundary. Used by the invariant tests below.
fn all_blend_inputs() -> impl Iterator<Item = (Classification, LlmGrade, Evidence, f32)> {
    const CLASSES: [Classification; 3] = [
        Classification::Benign,
        Classification::Suspicious,
        Classification::Hostile,
    ];
    const GRADES: [LlmGrade; 3] = [LlmGrade::Benign, LlmGrade::Suspicious, LlmGrade::Hostile];
    const SCORES: [f32; 7] = [0.0, 0.001, 0.024, 0.5, 0.9, 0.999, 1.0];
    // Manual mode, never fires, and placements straddling both boundaries.
    const FIRED: [Level; 6] = [
        Level::Manual,
        Level::Clean,
        Level::At(0),
        Level::At(30),
        Level::At(3000),
        Level::At(25_000),
    ];
    const ACTIVE: [Option<u16>; 2] = [None, Some(25)];
    let evidence = FIRED.into_iter().flat_map(|fired| {
        ACTIVE.into_iter().flat_map(move |active| {
            [true, false].into_iter().flat_map(move |readable| {
                [true, false].into_iter().flat_map(move |directed| {
                    [true, false]
                        .into_iter()
                        .map(move |hostile_finding| Evidence {
                            readable,
                            analyzer_directed: directed,
                            hostile_finding,
                            levels: LevelContext {
                                fired,
                                active,
                                grid_max: 25_000,
                            },
                        })
                })
            })
        })
    });
    let evidence: Vec<Evidence> = evidence.collect();
    CLASSES.into_iter().flat_map(move |ml| {
        let evidence = evidence.clone();
        GRADES.into_iter().flat_map(move |g| {
            let evidence = evidence.clone();
            SCORES
                .into_iter()
                .flat_map(move |p| evidence.clone().into_iter().map(move |ev| (ml, g, ev, p)))
        })
    })
}

#[test]
fn ml_admits_defaults_to_the_whole_grid() {
    // `levels()` carries grid_max 25000, so `None` means "fired at all".
    for fired in [
        Level::At(0),
        Level::At(3000),
        Level::At(10_000),
        Level::At(25_000),
    ] {
        assert!(levels(fired).ml_admits(None), "{fired} fires on the grid");
    }
    // Never fires: ML saw nothing, which is not an admission.
    assert!(!levels(Level::Clean).ml_admits(None));
    // Past the grid ceiling (the trait floor's former off-grid markers) is
    // not an ML admission.
    assert!(!levels(Level::At(25_001)).ml_admits(None));
    // Manual-threshold mode: no calibrated axis, so ML abstains.
    assert!(
        !LevelContext {
            fired: Level::Manual,
            active: None,
            grid_max: 0,
        }
        .ml_admits(None)
    );
}

#[test]
fn an_explicit_cutoff_tightens_the_admission() {
    assert!(levels(Level::At(10_000)).ml_admits(Some(10_000)));
    assert!(!levels(Level::At(10_001)).ml_admits(Some(10_000)));
    assert!(!levels(Level::At(25_000)).ml_admits(Some(10_000)));
    // Still no admission for a file that never fired.
    assert!(!levels(Level::Clean).ml_admits(Some(25_000)));
}

#[test]
fn steer_moves_a_bounded_fraction_and_never_crosses() {
    for p in [0.0_f32, 0.001, 0.25, 0.5, 0.9, 1.0] {
        let up = steer(p, true);
        let down = steer(p, false);
        // Exactly MAX_STEER of the distance to the bound it moves toward.
        assert!((up - (p + (1.0 - p) * MAX_STEER)).abs() < 1e-6);
        assert!((down - p * (1.0 - MAX_STEER)).abs() < 1e-6);
        // Directional, in range, and never past the bound.
        assert!(up >= p && up <= 1.0, "up {up} from {p}");
        assert!(down <= p && down >= 0.0, "down {down} from {p}");
    }
    // The bounds are fixed points: a certain score cannot be steered past 1,
    // and a zero score cannot be steered below 0.
    assert!((steer(1.0, true) - 1.0).abs() < 1e-6);
    assert!(steer(0.0, false).abs() < 1e-6);
    // Monotonic in p, so ML's ranking survives a uniform LLM opinion — the
    // property a flat escalation constant would destroy.
    let mut prev_up = f32::MIN;
    let mut prev_down = f32::MIN;
    for i in 0_i16..=100 {
        let p = f32::from(i) / 100.0;
        let (up, down) = (steer(p, true), steer(p, false));
        assert!(up > prev_up && down > prev_down, "monotonic at {p}");
        prev_up = up;
        prev_down = down;
    }
}

#[test]
fn one_step_toward_moves_exactly_one_rung() {
    use Classification::{Benign, Hostile, Suspicious};
    // Up.
    assert_eq!(one_step_toward(Benign, Suspicious), Suspicious);
    assert_eq!(one_step_toward(Benign, Hostile), Suspicious, "capped");
    assert_eq!(one_step_toward(Suspicious, Hostile), Hostile);
    // Down.
    assert_eq!(one_step_toward(Hostile, Suspicious), Suspicious);
    assert_eq!(one_step_toward(Hostile, Benign), Suspicious, "capped");
    assert_eq!(one_step_toward(Suspicious, Benign), Benign);
    // Already there.
    for c in [Benign, Suspicious, Hostile] {
        assert_eq!(one_step_toward(c, c), c);
    }
}

#[test]
fn blend_agreement_firms_up_the_verdict_it_confirms() {
    // A corroborated hostile moves toward 1.0…
    let (out, conf) = blend(
        Classification::Hostile,
        0.8,
        LlmGrade::Hostile,
        ev_open(true, false),
    );
    assert_eq!(out, Classification::Hostile);
    assert!((conf - (0.8 + 0.2 * MAX_STEER)).abs() < 1e-6);
    // …and a corroborated clean file moves toward 0.0: agreement on benign
    // must never raise the malice probability.

    let (out2, conf2) = blend(
        Classification::Benign,
        0.2,
        LlmGrade::Benign,
        ev_open(true, false),
    );
    assert_eq!(out2, Classification::Benign);
    assert!((conf2 - 0.2 * (1.0 - MAX_STEER)).abs() < 1e-6);
}

#[test]
fn blend_escalation_is_capped_at_one_step() {
    // ML benign, LLM suspicious → suspicious, steered up by at most MAX_STEER.
    let (out, conf) = blend(
        Classification::Benign,
        0.0,
        LlmGrade::Suspicious,
        ev_open(true, false),
    );
    assert_eq!(out, Classification::Suspicious);
    assert!((conf - MAX_STEER).abs() < 1e-6);
    // ML suspicious, LLM hostile → hostile (one step).
    let (out2, _) = blend(
        Classification::Suspicious,
        0.6,
        LlmGrade::Hostile,
        ev_open(true, false),
    );
    assert_eq!(out2, Classification::Hostile);
    // ML benign, LLM hostile → SUSPICIOUS, not hostile: two steps on one
    // model's word over attacker-controlled input is a review flag, not a block.
    let (out3, conf3) = blend(
        Classification::Benign,
        0.024,
        LlmGrade::Hostile,
        ev_open(true, false),
    );
    assert_eq!(out3, Classification::Suspicious);
    assert!(conf3 < 0.35, "steer stays bounded: {conf3}");
}

#[test]
fn blend_de_escalation_is_capped_and_guarded() {
    // ML suspicious (a false positive) on readable source, LLM benign →
    // cleared one step to benign, score steered down by MAX_STEER.
    let (out, conf) = blend(
        Classification::Suspicious,
        0.7,
        LlmGrade::Benign,
        ev_open(true, false),
    );
    assert_eq!(out, Classification::Benign);
    assert!((conf - 0.7 * (1.0 - MAX_STEER)).abs() < 1e-6);
    // A confident ML hostile the LLM calls benign drops one step at most, even
    // on readable source — never straight to benign.
    let (out2, _) = blend(
        Classification::Hostile,
        0.9,
        LlmGrade::Benign,
        ev_open(true, false),
    );
    assert_eq!(out2, Classification::Suspicious);
    // Opaque render → the clear is discarded entirely; ML stands untouched.
    let (out3, conf3) = blend(
        Classification::Hostile,
        0.9,
        LlmGrade::Benign,
        ev_open(false, false),
    );
    assert_eq!(out3, Classification::Hostile);
    assert!((conf3 - 0.9).abs() < 1e-6);
    // Analyzer-directed text → same, even though the render reads fine. This
    // is the injected-sample case: a clear is exactly what it was fishing for.
    let (out4, conf4) = blend(
        Classification::Hostile,
        0.9,
        LlmGrade::Benign,
        ev_open(true, true),
    );
    assert_eq!(out4, Classification::Hostile);
    assert!((conf4 - 0.9).abs() < 1e-6);
    // …but injection never blocks an escalation — talking your own sample up
    // gains an attacker nothing, so that path stays ungated.
    let (out5, _) = blend(
        Classification::Benign,
        0.1,
        LlmGrade::Suspicious,
        ev_open(true, true),
    );
    assert_eq!(out5, Classification::Suspicious);
}

#[test]
fn blend_never_moves_more_than_one_step_or_a_third_of_the_range() {
    for (ml, g, ev, p) in all_blend_inputs() {
        let (out, conf) = blend(ml, p, g, ev);
        let ctx = format!("{ml:?} + {g:?} @ {p} {ev:?}");
        assert!(
            (0.0..=1.0).contains(&conf),
            "{ctx}: conf out of range {conf}"
        );
        // Class moves at most one rung — except a corroborated escalation to
        // hostile, which two independent detectors earn (see `blend`).
        let step = i16::from(out as u8) - i16::from(ml as u8);
        let corroborated_escalation = out == Classification::Hostile
            && g.classification() == Classification::Hostile
            && ev.hostile_finding;
        let max_up = if corroborated_escalation { 2 } else { 1 };
        assert!(step <= max_up, "{ctx}: moved up to {out:?}");
        // Softening is capped at one rung, always and without exception: the
        // relaxation above is escalation-only, so nothing a sample carries can
        // buy it a two-rung clearing.
        assert!(step >= -1, "{ctx}: moved down to {out:?}");
        // Confidence moves at most MAX_STEER of the distance to the bound it
        // moved toward — the defensibility claim, stated as an assertion.
        let moved = conf - p;
        let room = if moved >= 0.0 { 1.0 - p } else { p };
        assert!(
            moved.abs() <= room * MAX_STEER + 1e-6,
            "{ctx}: moved {moved} of available {room}",
        );
        // The blend never contradicts itself: the score never moves *against*
        // the class step. It may not move at all — a score already at 0.0 or
        // 1.0 is pinned there while the class still steps.
        let opposed = (step > 0 && moved < -1e-6) || (step < 0 && moved > 1e-6);
        assert!(
            !opposed,
            "{ctx}: class step {step} contradicted by score move {moved}",
        );
    }
}

#[test]
fn ml_benign_plus_llm_hostile_always_reaches_suspicious() {
    // The case `--interpret` exists for: ML missed it entirely. An ML false
    // negative sits at `Level::Clean` by definition, so a proximity gate on the
    // suspicious boundary would make it unreachable — the escalation must not
    // depend on where ML happened to place a file it got wrong.
    for fired in [
        Level::Clean,
        Level::At(25_000),
        Level::At(10_000),
        Level::At(3000),
    ] {
        let (out, conf) = blend(
            Classification::Benign,
            0.024,
            LlmGrade::Hostile,
            ev_at(fired),
        );
        assert_eq!(out, Classification::Suspicious, "{fired} must reach review");
        assert!((conf - steer(0.024, true)).abs() < 1e-6);
    }
    // Same for the one-step case, and on an opaque render — an unreadable
    // sample is not a reason to *withhold* a review flag.
    let opaque = Evidence {
        readable: false,
        ..ev_at(Level::Clean)
    };
    let (out, _) = blend(Classification::Benign, 0.0, LlmGrade::Suspicious, opaque);
    assert_eq!(out, Classification::Suspicious);
}

#[test]
fn the_suspicious_boundary_is_a_routing_decision_not_a_budget() {
    use Classification::{Benign, Hostile, Suspicious};
    // Crossings that do not touch the hostile band are ungated at every ML
    // position, in both directions — no FP budget is being spent.
    for fired in [
        Level::Clean,
        Level::At(0),
        Level::At(30),
        Level::At(3000),
        Level::At(25_000),
    ] {
        let ev = ev_at(fired);
        assert!(ev.may_cross(Benign, Suspicious), "{fired} up");
        assert!(ev.may_cross(Suspicious, Benign), "{fired} down");
    }
    // Crossings that touch it are gated: near the line yes, far from it no.
    assert!(ev_at(Level::At(30)).may_cross(Suspicious, Hostile));
    assert!(!ev_at(Level::At(500)).may_cross(Suspicious, Hostile));
}

#[test]
fn the_proximity_gate_is_asymmetric_by_design() {
    use Classification::{Hostile, Suspicious};
    // Escalating *into* hostile spends the deploy level's FP budget, so it
    // stays proximity-gated: near the line yes, far from it no.
    assert!(ev_at(Level::At(30)).may_cross(Suspicious, Hostile));
    assert!(!ev_at(Level::At(500)).may_cross(Suspicious, Hostile));
    // Leaving hostile returns budget rather than spending it, and lands on
    // suspicious, so it is ungated at any depth…
    for fired in [
        Level::Clean,
        Level::At(1),
        Level::At(5),
        Level::At(20),
        Level::At(25),
        Level::At(30),
        Level::At(500),
        Level::At(3000),
        Level::At(25_000),
    ] {
        assert!(
            ev_at(fired).may_cross(Hostile, Suspicious),
            "{fired} should be free to step down",
        );
    }
    // …except from the strictest rung there is.
    assert!(!ev_at(Level::At(0)).may_cross(Hostile, Suspicious));
}

#[test]
fn band_crossing_requires_ml_to_be_within_one_steer() {
    // Suspicious → hostile, boundary L25 (conf 92). A file firing at L30 sits
    // just outside the hostile budget, close enough for the LLM to tip it…
    let (out, _) = blend(
        Classification::Suspicious,
        0.6,
        LlmGrade::Hostile,
        ev_at(Level::At(30)),
    );
    assert_eq!(out, Classification::Hostile);
    // …while L500 (conf 78) is far outside it. The LLM's opinion is recorded
    // and the score still firms up, but the band holds — ML's own evidence
    // was nowhere near the line, so the crossing was not earned.
    let (held, conf) = blend(
        Classification::Suspicious,
        0.6,
        LlmGrade::Hostile,
        ev_at(Level::At(500)),
    );
    assert_eq!(held, Classification::Suspicious, "band held");
    assert!(
        (conf - steer(0.6, true)).abs() < 1e-6,
        "score still steered"
    );
}

#[test]
fn readability_is_judged_per_member_not_per_line() {
    // A wheel of readable source that also ships one binary: the binary's hex
    // rows outnumber every source line put together, which is exactly how
    // `PyAutoIt` — a legitimate AutoIt wrapper — read as opaque and had its
    // clear discarded.
    let mut render = String::from("pkg.whl\twhl 900KB 3\n");
    for i in 0..4 {
        render.push_str(&format!("  pkg.whl/mod{i}.py\tpython 2KB 1\n"));
        render.push_str("  def handler(request):\n      return process(request)\n");
    }
    render.push_str("  pkg.whl/lib/native.dll\tpe 400KB 9\n");
    for _ in 0..80 {
        render.push_str(concat!(
            r"  0000: MZ\x90\x00\x03\x00\x00\x00\x04\x00\xff\xff",
            "\n"
        ));
    }
    assert!(
        render_mostly_readable(&render),
        "four readable members beside one binary is a readable archive",
    );

    // A sample that really is one packed binary still has nothing to read.
    let mut packed = String::from("dropper.exe\tpe 400KB 9\n");
    for _ in 0..80 {
        packed.push_str(concat!(
            r"  0000: MZ\x90\x00\x03\x00\x00\x00\x04\x00\xff\xff",
            "\n"
        ));
    }
    assert!(!render_mostly_readable(&packed));

    // Header detection: a tab-indented source line is not a member header.
    assert!(is_member_header("  pkg.whl/lib/native.dll\tpe 400KB 9"));
    assert!(!is_member_header("\tif x:\treturn 1"));
    assert!(!is_member_header("no tabs here at all"));
}

#[test]
fn only_the_strictest_rung_cannot_be_talked_down() {
    // A hostile verdict the LLM clears steps to *suspicious* — never benign —
    // wherever inside the band ML fired, so the sample is routed for review
    // rather than released. Depth decides where in the suspicious band it
    // lands (`engine::softened_level`), not whether it may move at all.
    for fired in [Level::At(20), Level::At(5), Level::At(1)] {
        let (out, conf) = blend(
            Classification::Hostile,
            0.99,
            LlmGrade::Benign,
            ev_at(fired),
        );
        assert_eq!(out, Classification::Suspicious, "{fired} should step down");
        assert!((conf - steer(0.99, false)).abs() < 1e-6);
    }
    // L0 is the exception: the tightest budget the grid has does not move on
    // one fallible opinion, however confident its prose.
    let (held, conf) = blend(
        Classification::Hostile,
        0.99,
        LlmGrade::Benign,
        ev_at(Level::At(0)),
    );
    assert_eq!(held, Classification::Hostile);
    assert!((conf - steer(0.99, false)).abs() < 1e-6);
}

#[test]
fn corroboration_earns_a_hostile_crossing_ml_is_far_from() {
    // A packed dropper ML scored as merely suspicious, firing way out at
    // L5000 — far below the hostile boundary, so the LLM alone cannot take it
    // there.
    let alone = Evidence {
        readable: false,
        ..ev_at(Level::At(5000))
    };
    let (out, _) = blend(Classification::Suspicious, 0.3, LlmGrade::Hostile, alone);
    assert_eq!(out, Classification::Suspicious, "LLM alone cannot cross");
    // cleave independently flagging a hostile trait is the second witness that
    // earns it — two detectors agreeing with ML as the outlier.
    let corroborated = Evidence {
        hostile_finding: true,
        ..alone
    };
    let (rescued, _) = blend(
        Classification::Suspicious,
        0.3,
        LlmGrade::Hostile,
        corroborated,
    );
    assert_eq!(rescued, Classification::Hostile);
}

#[test]
fn corroboration_never_licenses_a_downgrade() {
    // The hatch is escalation-only: a cleave hostile finding must never help
    // the LLM talk a verdict *down*, or a sample could earn its own clearing.
    let ev = Evidence {
        hostile_finding: true,
        ..ev_at(Level::At(0))
    };
    let (out, _) = blend(Classification::Hostile, 0.99, LlmGrade::Benign, ev);
    assert_eq!(out, Classification::Hostile);
}

/// A prompt injection the phrase list misses still cannot clear a sample
/// cleave independently flagged hostile, on any rung: the class and the
/// score both hold.
#[test]
fn a_hostile_finding_vetoes_every_softening() {
    for (ml, fired) in [
        (Classification::Hostile, Level::At(1)),
        (Classification::Suspicious, Level::At(1000)),
        (Classification::Benign, Level::At(5000)),
    ] {
        let ev = Evidence {
            hostile_finding: true,
            ..ev_at(fired)
        };
        let (out, conf) = blend(ml, 0.6, LlmGrade::Benign, ev);
        assert_eq!(out, ml, "{ml:?} at {fired:?}");
        assert!((conf - 0.6).abs() < 1e-6, "{ml:?} at {fired:?}: {conf}");
    }
}

#[test]
fn manual_threshold_mode_abstains_from_the_proximity_gate() {
    // With no level table there is no calibrated axis to measure against, so
    // the gate has nothing to say and the other guards carry the decision.
    let (out, _) = blend(
        Classification::Suspicious,
        0.6,
        LlmGrade::Hostile,
        ev_open(true, false),
    );
    assert_eq!(out, Classification::Hostile);
}

/// Only the structured report admits a sample. The render carries the
/// sample's own source, so a line its author wrote to look like a cleave
/// annotation (`// H …`) must count for nothing; the gate never sees it.
#[test]
fn only_structured_findings_admit() {
    let levels = LevelContext {
        fired: Level::Clean,
        active: None,
        grid_max: 0,
    };
    let hostile = FindingSeverity {
        elevated: true,
        hostile: true,
        ..FindingSeverity::default()
    };
    let gate = |findings: FindingSeverity| {
        admit(
            &InterpretConfig::default(),
            Classification::Benign,
            0.0,
            levels,
            findings,
            "t",
        )
        .map(|a| a.tier)
    };
    assert_eq!(gate(FindingSeverity::default()), None);
    assert_eq!(gate(hostile), Some(LlmAdmission::Required));
}

#[test]
fn every_class_move_was_permitted_by_the_gate() {
    // The crossing rule stated as an invariant: no band changes without the
    // gate having allowed that exact move.
    for (ml, g, ev, p) in all_blend_inputs() {
        let (out, _) = blend(ml, p, g, ev);
        assert!(
            out == ml || ev.may_cross(ml, out),
            "{ml:?} + {g:?} @ {p} {ev:?} → {out:?} without permission",
        );
    }
}

#[test]
fn untrusted_read_never_softens_the_verdict() {
    // The security invariant, over every input: when the render is opaque or
    // carries analyzer-directed text, the LLM cannot lower the class OR the
    // score — the two things an injected sample is fishing for. Escalation
    // stays available, since talking your own sample up gains an attacker
    // nothing.
    for (ml, g, ev, p) in all_blend_inputs() {
        if ev.readable && !ev.analyzer_directed {
            continue; // trusted read; softening is allowed by design
        }
        let (out, conf) = blend(ml, p, g, ev);
        let ctx = format!("{ml:?} + {g:?} @ {p} {ev:?}");
        assert!(out >= ml, "{ctx}: class softened to {out:?}",);
        assert!(conf >= p - 1e-6, "{ctx}: score softened to {conf}");
    }
    // Specifically: agreement on benign normally steers the score down, but
    // not when the sample is talking to the grader.
    let (_, clean) = blend(
        Classification::Benign,
        0.4,
        LlmGrade::Benign,
        ev_open(true, false),
    );
    assert!((clean - 0.4 * (1.0 - MAX_STEER)).abs() < 1e-6);
    let (_, injected) = blend(
        Classification::Benign,
        0.4,
        LlmGrade::Benign,
        ev_open(true, true),
    );
    assert!((injected - 0.4).abs() < 1e-6, "held at ML's score");
}
