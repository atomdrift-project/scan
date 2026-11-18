//! The interpret prompt `atomscan render` prints for a saved report.
//!
//! `render` reads one JSON report and writes the prompt, through
//! [`cleave::report_input::decode`] and [`scan::engine::render_saved_report`].
//! The fixture is a primary archive with a member whose description alone is
//! larger than the 96 KiB prompt budget. The expected prompt keeps the hostile
//! member and the short suspicious member, and records that the other two
//! were dropped.

#![allow(clippy::unwrap_used)]

/// Bytes the interpret prompt is held to. Matches
/// `INTERPRET_PROMPT_BUDGET_BYTES` in the render.
const PROMPT_BUDGET_BYTES: usize = 96 * 1024;

#[test]
fn render_keeps_the_prompt_inside_the_budget() {
    let input = include_bytes!("data/interpret-budget/input.json");
    let expected = include_str!("data/interpret-budget/expected.txt");
    let loaded = cleave::report_input::decode(input).unwrap();
    let rendered =
        scan::engine::render_saved_report(&loaded, scan::OutputFormat::Interpret).unwrap();

    assert!(
        input.len() > PROMPT_BUDGET_BYTES,
        "the fixture has to exceed the budget or it does not exercise the cut"
    );
    assert!(
        rendered.len() <= PROMPT_BUDGET_BYTES,
        "prompt is {} bytes",
        rendered.len()
    );
    assert_eq!(rendered, expected);
}

#[test]
fn render_caps_an_oversized_protected_finding() {
    let input = include_bytes!("data/interpret-budget/input.json");
    let mut loaded = cleave::report_input::decode(input).unwrap();
    loaded.report.files.truncate(1);
    let finding = &mut loaded.report.files[0].findings[0];
    finding.crit = cleave::Criticality::Hostile;
    finding.desc = format!("Oversized evidence {}", "é".repeat(PROMPT_BUDGET_BYTES)).into();
    let rendered =
        scan::engine::render_saved_report(&loaded, scan::OutputFormat::Interpret).unwrap();

    assert!(rendered.len() <= PROMPT_BUDGET_BYTES);
    assert!(rendered.contains("Oversized evidence"));
    assert!(rendered.ends_with("[Context truncated to fit the prompt budget.]\n"));
}
