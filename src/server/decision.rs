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

use crate::model::Level;

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
/// [`decide`] answers `Unanalyzed` for such verdicts regardless.
pub(crate) fn default_budget(server_level: Option<u16>) -> u16 {
    server_level.unwrap_or(crate::model::DEFAULT_SEVERITY_LEVEL)
}

/// Levels above this are never worse than suspicious, whatever the grid.
///
/// Mirrors `capped_suspicious_level(grid_max)`, which is
/// `min(grid_max, SUSPICIOUS_LEVEL_CEILING)`. A stored verdict does not carry
/// the grid it came from, and production grids run to 25000, so the ceiling is
/// the constant. `agrees_with_the_model_grid` pins the two together.
const SUSPICIOUS_CEILING: u16 = 3000;

/// What a caller should do about this package.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Decision {
    /// Analyzed, and not hostile at the caller's threshold.
    Allow,
    /// Analyzed, and hostile at the caller's threshold.
    Block,
    /// Nobody has analyzed this. Nothing is wrong; there is simply no answer.
    Unanalyzed,
    /// We failed. This says nothing about the package.
    ///
    /// Kept rigorously distinct from [`Self::Unanalyzed`]: a caller may reasonably
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
/// [`Level::Manual`] answers [`Decision::Unanalyzed`] rather than guessing: no
/// level table applies to such a verdict, so no budget can be evaluated against
/// it, and saying so is honest where allowing it would not.
pub(crate) fn decide(fires_at: Level, budget: u16) -> (Decision, Severity) {
    match fires_at {
        Level::Manual => (Decision::Unanalyzed, Severity::Benign),
        Level::At(lvl) if lvl <= budget => (Decision::Block, Severity::Hostile),
        Level::At(lvl) if lvl <= SUSPICIOUS_CEILING => (Decision::Allow, Severity::Suspicious),
        Level::Clean | Level::At(_) => (Decision::Allow, Severity::Benign),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the layer, and the one line that must never invert.
    #[test]
    fn lower_levels_are_worse() {
        assert_eq!(
            decide(Level::At(2), 25).0,
            Decision::Block,
            "a tight firing level is hostile"
        );
        assert_eq!(
            decide(Level::At(25), 25).0,
            Decision::Block,
            "the budget is inclusive"
        );
        assert_eq!(
            decide(Level::At(26), 25).0,
            Decision::Allow,
            "one past the budget is not"
        );
        assert_eq!(decide(Level::At(20000), 25).0, Decision::Allow);
    }

    /// Clean means "fires at no level". Its wire form, `-1`, compared
    /// numerically is lower than every budget, so a naive `lvl <= budget`
    /// blocks every clean package in the corpus.
    #[test]
    fn the_clean_sentinel_is_not_the_tightest_level() {
        for budget in [0, 1, 25, 25_000] {
            assert_eq!(
                decide(Level::Clean, budget),
                (Decision::Allow, Severity::Benign),
                "the clean sentinel was read as a firing level at budget={budget}",
            );
        }
    }

    /// A verdict with no level table cannot be judged against a budget. Unanalyzed
    /// hands the choice to the caller's policy; allow would make it for them.
    #[test]
    fn manual_threshold_mode_is_unanalyzed_not_allowed() {
        assert_eq!(decide(Level::Manual, 25).0, Decision::Unanalyzed);
        assert_eq!(decide(Level::Manual, 25_000).0, Decision::Unanalyzed);
    }

    #[test]
    fn severity_is_independent_of_the_budget() {
        // Same artifact, two callers: the decision moves, the severity does not.
        assert_eq!(
            decide(Level::At(500), 25),
            (Decision::Allow, Severity::Suspicious)
        );
        assert_eq!(
            decide(Level::At(500), 1000),
            (Decision::Block, Severity::Hostile)
        );
        assert_eq!(
            decide(Level::At(3001), 25),
            (Decision::Allow, Severity::Benign)
        );
        assert_eq!(
            decide(Level::At(3000), 25),
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
        assert_eq!(
            decide(Level::At(50), default_budget(None)).0,
            Decision::Allow
        );
        assert_eq!(
            decide(Level::At(25), default_budget(None)).0,
            Decision::Block
        );
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
                let (decision, severity) = decide(Level::At(level), budget);
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

/// One decision, as it goes on the wire.
///
/// Every field but `url` is always present. A key that is unknown is `null`
/// and a list that is empty is `[]`, never absent — a caller writes one code path against
/// a shape that does not move, and a generated type has no optionals to unwrap
/// that are really just "we had nothing to say".
#[derive(serde::Serialize)]
pub(super) struct V1Decision {
    pub(super) decision: Decision,
    pub(super) purl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) url: Option<String>,
    pub(super) sha256: Option<String>,
    pub(super) severity: Option<Severity>,
    /// The tightest false-positive budget per 100 million benign files at which
    /// this artifact grades hostile — lower being worse, and `-1` meaning it
    /// fires at none. A property of the file and the model: measured, never
    /// chosen, which is what separates it from the caller's own
    /// `false_positive_budget` that it is compared against. Present so a caller
    /// can tune that budget against real numbers; `null` when there is none.
    pub(super) fires_at: Level,
    pub(super) reason: Option<String>,
    pub(super) findings: Vec<V1Finding>,
    pub(super) engine_version: Option<String>,
    pub(super) analyzed_at: Option<String>,
}

impl V1Decision {
    /// Nobody has analyzed this artifact. Nothing is wrong; there is simply no
    /// answer, and what a caller does about that is their policy to set.
    pub(super) fn unanalyzed(sha: Option<&str>, purl: Option<&str>) -> Self {
        Self::empty(Decision::Unanalyzed, sha, purl)
    }

    /// We could not answer. Deliberately carries no severity, no level and no
    /// findings: this decision is about us, not about the artifact, and a
    /// caller must not be able to read anything into it.
    pub(super) fn unavailable(sha: Option<&str>, purl: Option<&str>) -> Self {
        Self::empty(Decision::Unavailable, sha, purl)
    }

    /// What the filters alone justify, for an artifact no stored verdict and no
    /// corpus record covers. `None` when the filters had no opinion.
    ///
    /// `engine_version` and `analyzed_at` stay `None`, which is the whole
    /// contract: an engine is what separates a measurement from a citation, so
    /// a caller (and beamline's cache) can tell this from a scan we ran, and
    /// `/v1/analyze` is free to replace it with a real one.
    ///
    /// Levels come from [`crate::lookup::bloom_claim`] — the loosest each tier
    /// can justify, since a filter carries membership and not a measurement.
    pub(super) fn bloom(
        d: crate::bloom_repo::Decision,
        sha: Option<&str>,
        purl: Option<&str>,
        budget: u16,
    ) -> Option<Self> {
        let claim = crate::lookup::bloom_claim(d)?;
        let fires_at = claim.lvl;
        let (decided, severity) = decide(fires_at, budget);
        Some(Self {
            decision: decided,
            purl: purl.map(str::to_owned),
            url: None,
            sha256: sha.map(str::to_owned),
            severity: Some(severity),
            fires_at,
            reason: claim.finding.as_ref().map(|f| f.desc.to_owned()),
            findings: claim
                .finding
                .as_ref()
                .map(|f| V1Finding::from_bloom(f, purl))
                .into_iter()
                .collect(),
            engine_version: None,
            analyzed_at: None,
        })
    }

    pub(super) fn empty(decided: Decision, sha: Option<&str>, purl: Option<&str>) -> Self {
        Self {
            decision: decided,
            purl: purl.map(str::to_owned),
            url: None,
            sha256: sha.map(str::to_owned),
            severity: None,
            fires_at: Level::Manual,
            reason: None,
            findings: Vec::new(),
            engine_version: None,
            analyzed_at: None,
        }
    }

    /// Whether this says something about the artifact rather than about us,
    /// and says it because an engine of ours measured it.
    ///
    /// `unanalyzed` reports that nobody has analyzed it, which is precisely what
    /// `/v1/analyze` exists to fix, and `unavailable` reports that we could not
    /// find out. Neither may stand in for a run.
    ///
    /// Nor may a level derived from threat-feed citations, and that one is the
    /// easy miss: it carries a real `decision`, so it reads as a verdict at
    /// every glance. Standing in for the run would mean an artifact nobody has
    /// analyzed never gets analyzed — the caller is told `block`, the corpus
    /// learns nothing, and the gap the derived level papers over stays open for
    /// good. An engine is exactly what separates a measurement from a citation,
    /// so an engine is what is asked for.
    pub(super) fn is_verdict(&self) -> bool {
        !matches!(self.decision, Decision::Unanalyzed | Decision::Unavailable)
            && self.engine_version.is_some()
    }

    /// Anything that answers the caller's question, measured or not.
    ///
    /// Wider than [`Self::is_verdict`] on purpose, and the difference is a
    /// policy choice rather than an oversight. `is_verdict` remains the strict
    /// question — is this a measurement of ours — and downstream still asks it
    /// by looking for an engine. This one governs whether `/v1/analyze` may
    /// answer at all, where the operator's judgement is that a fast answer from
    /// what we already know beats spending a slot to rediscover it.
    ///
    /// The cost is real and worth naming: for an artifact nobody has analyzed
    /// and a feed has cited, this answers from the citation and the analysis
    /// never happens, so the corpus does not learn. `?fresh=1` is the escape
    /// hatch for a caller who needs the measurement, and the derived answer
    /// still carries no `engine_version`, so nothing downstream mistakes it for
    /// one.
    ///
    /// `unanalyzed` and `unavailable` are excluded exactly as before: the first is
    /// what `/v1/analyze` exists to fix, the second is a statement about us.
    pub(super) fn is_answerable(&self) -> bool {
        !matches!(self.decision, Decision::Unanalyzed | Decision::Unavailable)
            && !self.fires_at.is_manual()
    }

    /// A verdict this server holds in its own index.
    ///
    /// Takes no digest: a stored verdict always carries its own, and it is the
    /// artifact's identity rather than whatever the caller happened to type.
    pub(super) fn stored(v: &crate::lookup::Verdict, purl: Option<&str>, budget: u16) -> Self {
        let (decided, severity) = decide(v.lvl, budget);
        Self {
            decision: decided,
            // The verdict names the artifact it is about; the caller's spelling
            // only fills in what it could not.
            purl: v.purl.clone().or_else(|| purl.map(str::to_owned)),
            url: None,
            sha256: Some(v.sha256.clone()),
            severity: Some(severity),
            fires_at: v.lvl,
            reason: v.why.clone(),
            findings: V1Finding::worth_reporting(v.hits.iter().map(V1Finding::from_hit)),
            engine_version: Some(v.eng.clone()),
            analyzed_at: Some(v.at.clone()),
        }
    }

    /// A record the corpus holds. Decided here rather than there: hopper stores
    /// what an artifact is, and turning that into allow or block is policy this
    /// server owns, so the same budget produces the same answer whichever side
    /// of the index the record came from.
    pub(super) fn corpus(
        r: &crate::hopper::Record,
        sha: Option<&str>,
        purl: Option<&str>,
        budget: u16,
    ) -> Self {
        let (decided, severity) = decide(r.fires_at, budget);
        Self {
            decision: decided,
            purl: r.purl.clone().or_else(|| purl.map(str::to_owned)),
            url: None,
            // Empty is absent. A record standing on threat-feed citations for a
            // package nobody has analyzed names no bytes, and the corpus sends
            // the field as "" rather than omitting it — which would put an
            // empty string where the wire contract says string|null, and where
            // a caller comparing digests to prove two spellings are one thing
            // would find them equal.
            sha256: r
                .sha256
                .clone()
                .filter(|s| !s.is_empty())
                .or_else(|| sha.map(str::to_owned)),
            severity: Some(severity),
            fires_at: r.fires_at,
            reason: r.reason.clone(),
            findings: V1Finding::worth_reporting(r.findings.iter().map(V1Finding::from_corpus)),
            engine_version: r.engine_version.clone(),
            analyzed_at: r.analyzed_at.clone(),
        }
    }

    /// Attach the reason an `unavailable` came about. `None` leaves the
    /// decision exactly as it was, so the common outage keeps its bare shape.
    pub(super) fn because(mut self, reason: Option<&'static str>) -> Self {
        if let Some(reason) = reason {
            self.reason = Some(reason.to_owned());
        }
        self
    }

    /// Answer about the coordinate the caller named, in the spelling they
    /// named it.
    ///
    /// Every key is normalized before it is looked up — PyPI folds `.` and `_`
    /// to `-` per PEP 503, npm scopes are unwrapped, a bare `npm/left-pad` gets
    /// its `pkg:` — and echoing the normalized form back is how
    /// `pkg:pypi/info.gianlucacosta.eos.core@2.0.2` came home answered about
    /// `pkg:pypi/info-gianlucacosta-eos-core@2.0.2`. The same package, and a
    /// caller has no way to know that without implementing PEP 503 themselves.
    ///
    /// That matters most where it is least visible: a lookup may name fifty
    /// packages and the reply is a list, so `purl` is what a caller matches
    /// response to request by. Rewriting the spelling breaks that silently, and
    /// only for the names that happen to contain a `.` or a `_`.
    ///
    /// So the field answers "the package you asked about" and the caller's
    /// bytes are returned unaltered. `sha256` remains the identity, and it is
    /// the field to compare when two spellings must be proven to be one thing.
    pub(super) fn asked_about(mut self, asked: Option<&str>) -> Self {
        if let Some(asked) = asked {
            self.purl = Some(asked.to_owned());
        }
        self
    }

    pub(super) fn asked_about_url(mut self, asked: Option<&str>) -> Self {
        if let Some(asked) = asked {
            self.url = Some(asked.to_owned());
        }
        self
    }

    pub(super) fn with_url(mut self, url: Option<&str>) -> Self {
        if let Some(url) = url {
            self.url = Some(url.to_owned());
        }
        self
    }
}

/// One finding on the wire.
///
/// Fed from this server's index or from the corpus, which know different
/// amounts about the same thing: a stored hit carries the file and offset it
/// fired on, while the corpus keeps only the trait and its criticality — those
/// details live in the one column a lookup must not read.
///
/// `id` and `crit` are therefore the only fields always present, and the rest
/// are omitted when there is nothing to say rather than sent as null. That is
/// the opposite of the rule the enclosing decision object follows, and the two
/// differ because the questions do. A decision has a FIXED set of things it
/// answers, so a caller writes one code path against nine keys that never move
/// and `"engine_version": null` is itself the answer to "which engine". A
/// finding has no such set: how much is known about one varies by where it came
/// from, four nulls per corpus finding is most of the object, and a reader
/// checking `desc` has to handle absence anyway.
#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct V1Finding {
    pub(super) id: String,
    pub(super) crit: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) pkg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) desc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) off: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) line: Option<u64>,
}

impl V1Finding {
    pub(super) fn from_hit(h: &crate::lookup::Hit) -> Self {
        let some = |s: &String| (!s.is_empty()).then(|| s.clone());
        Self {
            id: h.id.clone(),
            crit: h.crit,
            file: some(&h.file),
            pkg: some(&h.pkg),
            desc: some(&h.desc),
            off: h.off,
            line: h.line,
        }
    }

    /// A finding synthesized from a filter hit. Carries no `file`, `off` or
    /// `line` — a filter knows membership and nothing about where anything
    /// fired — which is the same shape the corpus sends for a citation.
    pub(super) fn from_bloom(c: &crate::lookup::BloomFinding, purl: Option<&str>) -> Self {
        Self {
            id: c.id.to_owned(),
            crit: c.crit,
            file: None,
            pkg: purl.map(str::to_owned),
            desc: Some(c.desc.to_owned()),
            off: None,
            line: None,
        }
    }

    pub(super) fn from_corpus(f: &crate::hopper::Finding) -> Self {
        Self {
            id: f.id.clone(),
            crit: f.crit,
            file: None,
            pkg: None,
            desc: f.desc.clone(),
            off: None,
            line: None,
        }
    }

    /// The findings worth putting on the wire: the strongest few, worst first.
    ///
    /// The corpus already decides this in a trigger — `crit >= 4`, ordered by
    /// criticality, at most three — and a decision answered from this server's
    /// own index has to land on the same set, or one artifact reads differently
    /// depending on which side of the index happened to answer. Applying it
    /// here rather than trusting each source keeps the two in step by
    /// construction; on corpus records it is a no-op.
    ///
    /// A benign artifact clears the bar with nothing, and that is the intended
    /// answer rather than a gap to fill: ten "Rust test marker" hits explain
    /// nothing about an allow, and listing them invites a caller to read
    /// significance into noise.
    pub(super) fn worth_reporting(all: impl Iterator<Item = Self>) -> Vec<Self> {
        let mut kept: Vec<Self> = all.filter(|f| f.crit >= REPORT_MIN_CRIT).collect();
        // Stable, so equal criticalities keep the order their source listed
        // them in — the same tiebreak as the trigger's `ORDER BY crit DESC,
        // ord`.
        kept.sort_by_key(|f| std::cmp::Reverse(f.crit));
        kept.truncate(REPORT_LIMIT);
        kept
    }
}

/// Suspicious and above. Below this a trait is an observation, not a reason.
const REPORT_MIN_CRIT: u8 = 4;

/// Enough to show why, few enough to read. Matches the corpus's `LIMIT 3`.
const REPORT_LIMIT: usize = 3;
