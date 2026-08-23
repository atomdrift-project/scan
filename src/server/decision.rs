//! The v1 decision layer: turning a verdict into an answer a firewall can act on.
//!
//! Everything downstream of us is a gate in someone's `npm install`. It does not
//! want a verdict object, it wants to know whether to let the package through —
//! so this module owns the one mapping that answers that, and the wire names for
//! it.
//!
//! The mapping is deliberately a pure function of the stored verdict and the
//! caller's threshold. A verdict's `lvl` is a property of the file and the model,
//! not of anyone's policy (see [`crate::model::verdict_for_level`]), which is
//! what lets one cached envelope serve every caller whatever threshold they set.
//! Nothing here may consult server state, or that stops being true.

use serde::Serialize;

/// The budget applied when a caller names none: this server's own operating
/// point.
///
/// Not a number of its own. `--level` already resolves through CLI flag, then
/// the model bundle's baked-in `default_severity_level`, then
/// [`crate::model::DEFAULT_SEVERITY_LEVEL`] — and a second constant beside that
/// chain would be a fourth opinion that drifts from the three. A caller who
/// sends no threshold gets the one this deploy was tuned for, which is the only
/// default that stays true when the model is retuned.
///
/// `None` is manual-threshold mode, where no level table applies at all; the
/// fallback const stands in so the parameter still has a meaning, though
/// [`decide`] answers `Unknown` for such verdicts regardless.
pub(crate) fn default_budget(server_level: Option<u16>) -> u16 {
    server_level.unwrap_or(crate::model::DEFAULT_SEVERITY_LEVEL)
}

/// Levels above this are never worse than suspicious, whatever the grid.
///
/// Mirrors `capped_suspicious_level(grid_max)`, which is
/// `min(grid_max, SUSPICIOUS_LEVEL_CEILING)`. A stored verdict does not carry
/// the grid it came from, and production grids run to 25000, so the ceiling is
/// the constant. `agrees_with_the_model_grid` pins the two together.
const SUSPICIOUS_CEILING: i32 = 3000;

/// What a caller should do about this package.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Decision {
    /// Analyzed, and not hostile at the caller's threshold.
    Allow,
    /// Analyzed, and hostile at the caller's threshold.
    Block,
    /// Nobody has analyzed this. Nothing is wrong; there is simply no answer.
    Unknown,
    /// We failed. This says nothing about the package.
    ///
    /// Kept rigorously distinct from [`Self::Unknown`]: a caller may reasonably
    /// install unanalyzed packages while refusing to install anything during an
    /// outage, or exactly the reverse, and collapsing the two takes that choice
    /// away from them.
    Unavailable,
}

/// How bad the artifact is, independent of anyone's threshold.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Severity {
    /// Fires at no level, or only above the suspicious ceiling.
    Benign,
    /// Fires, but only under a budget looser than the ceiling.
    Suspicious,
    /// Fires at or under the caller's budget.
    ///
    /// The same word [`crate::model::Classification`] uses, deliberately. This
    /// enum exists only because that one serializes as an ordinal and the wire
    /// wants a name — it is not a second opinion about what the grades are, and
    /// `agrees_with_the_model_grid` holds the two together.
    Hostile,
}

/// The decision and severity for one artifact at one caller's budget.
///
/// The two arguments share a scale and are different things, which is the
/// distinction this whole layer turns on and the one the wire names carry:
/// `fires_at` is measured — the tightest budget at which *this artifact* grades
/// hostile — while `budget` is chosen, being how many false positives per 100
/// million the caller will tolerate. Mirrors `verdict_for_level(fired_level,
/// level, ..)`, whose two parameters are that same pair.
///
/// `None` is manual-threshold mode and answers [`Decision::Unknown`] rather
/// than guessing: no level table applies to such a verdict, so no budget can be
/// evaluated against it, and saying so is honest where allowing it would not.
pub(crate) fn decide(fires_at: Option<i32>, budget: u16) -> (Decision, Severity) {
    let Some(lvl) = fires_at else {
        return (Decision::Unknown, Severity::Benign);
    };
    // The sentinel is not a level: it is the absence of one, and it is negative
    // precisely so it cannot be compared as though it were the tightest budget
    // of all. Ordering it against `budget` numerically would invert the scale
    // and block every clean artifact we have.
    if lvl < 0 {
        return (Decision::Allow, Severity::Benign);
    }
    if lvl <= i32::from(budget) {
        (Decision::Block, Severity::Hostile)
    } else if lvl <= SUSPICIOUS_CEILING {
        (Decision::Allow, Severity::Suspicious)
    } else {
        (Decision::Allow, Severity::Benign)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// The whole point of the layer, and the one line that must never invert.
    #[test]
    fn lower_levels_are_worse() {
        assert_eq!(
            decide(Some(2), 25).0,
            Decision::Block,
            "a tight firing level is hostile"
        );
        assert_eq!(
            decide(Some(25), 25).0,
            Decision::Block,
            "the budget is inclusive"
        );
        assert_eq!(
            decide(Some(26), 25).0,
            Decision::Allow,
            "one past the budget is not"
        );
        assert_eq!(decide(Some(20000), 25).0, Decision::Allow);
    }

    /// `-1` means "fires at no level". Compared numerically it is lower than
    /// every budget, so a naive `lvl <= budget` blocks every clean package in
    /// the corpus. This is the bug the sentinel invites and the reason `decide`
    /// tests the sign before the threshold.
    #[test]
    fn the_clean_sentinel_is_not_the_tightest_level() {
        for budget in [0, 1, 25, 25_000] {
            assert_eq!(
                decide(Some(-1), budget),
                (Decision::Allow, Severity::Benign),
                "the clean sentinel was read as a firing level at budget={budget}",
            );
        }
    }

    /// A verdict with no level table cannot be judged against a budget. Unknown
    /// hands the choice to the caller's policy; allow would make it for them.
    #[test]
    fn manual_threshold_mode_is_unknown_not_allowed() {
        assert_eq!(decide(None, 25).0, Decision::Unknown);
        assert_eq!(decide(None, 25_000).0, Decision::Unknown);
    }

    #[test]
    fn severity_is_independent_of_the_budget() {
        // Same artifact, two callers: the decision moves, the severity does not.
        assert_eq!(
            decide(Some(500), 25),
            (Decision::Allow, Severity::Suspicious)
        );
        assert_eq!(
            decide(Some(500), 1000),
            (Decision::Block, Severity::Hostile)
        );
        assert_eq!(decide(Some(3001), 25), (Decision::Allow, Severity::Benign));
        assert_eq!(
            decide(Some(3000), 25),
            (Decision::Allow, Severity::Suspicious)
        );
    }

    /// The default follows the deploy rather than restating it. If someone
    /// retunes this server to a looser level, a caller who sends no threshold
    /// must move with it — otherwise the API quietly enforces a policy the
    /// operator stopped running.
    #[test]
    fn the_default_budget_follows_the_server() {
        assert_eq!(default_budget(Some(500)), 500);
        assert_eq!(default_budget(Some(0)), 0);
        assert_eq!(
            default_budget(None),
            crate::model::DEFAULT_SEVERITY_LEVEL,
            "manual-threshold mode falls back to the shipped default",
        );
        // The shipped default is strict on purpose: this number decides whether
        // a build breaks, and a firewall that cries wolf gets switched off.
        assert_eq!(crate::model::DEFAULT_SEVERITY_LEVEL, 25);
        assert_eq!(decide(Some(50), default_budget(None)).0, Decision::Allow);
        assert_eq!(decide(Some(25), default_budget(None)).0, Decision::Block);
    }

    /// This layer restates a rule that already exists in the model, so it can
    /// drift from it. Pin them together across the grid rather than trusting
    /// two copies of one sentence to stay equal.
    #[test]
    fn agrees_with_the_model_grid() {
        use crate::model::{Classification, verdict_for_level};
        const GRID_MAX: u16 = 25_000;
        for level in [0_u16, 1, 2, 25, 26, 500, 2999, 3000, 3001, 20_000, 25_000] {
            for budget in [0_u16, 25, 500, 3000] {
                let (decision, severity) = decide(Some(i32::from(level)), budget);
                let expected = verdict_for_level(level, budget, GRID_MAX);
                let blocked = decision == Decision::Block;
                assert_eq!(
                    blocked,
                    expected == Classification::Hostile,
                    "level={level} budget={budget}: decision and model disagree",
                );
                let expected_severity = match expected {
                    Classification::Hostile => Severity::Hostile,
                    Classification::Suspicious => Severity::Suspicious,
                    Classification::Benign => Severity::Benign,
                };
                assert_eq!(severity, expected_severity, "level={level} budget={budget}");
            }
        }
    }
}
