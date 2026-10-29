//! How an LLM grade moves the model's verdict: at most one step, and only
//! as far as the evidence allows.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// The LLM's trinary verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmGrade {
    /// No malicious intent.
    Benign,
    /// Warrants human review.
    Suspicious,
    /// Almost certainly malicious.
    Hostile,
}

impl LlmGrade {
    /// Parse a model-emitted grade, case-insensitively. Unknown → `None`.
    pub(super) fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "benign" => Some(Self::Benign),
            "suspicious" => Some(Self::Suspicious),
            "hostile" => Some(Self::Hostile),
            _ => None,
        }
    }

    /// Lowercase label.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Benign => "benign",
            Self::Suspicious => "suspicious",
            Self::Hostile => "hostile",
        }
    }

    /// The equivalent ML [`Classification`].
    pub(super) fn classification(self) -> Classification {
        match self {
            Self::Benign => Classification::Benign,
            Self::Suspicious => Classification::Suspicious,
            Self::Hostile => Classification::Hostile,
        }
    }
}

/// The result of an interpretation pass, serialized as the response `llm` object.
/// Present whenever the pass was *attempted* (the sample was [`admit`]ted).
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Interpretation {
    /// The LLM answered, and its grade was blended into the verdict.
    Graded(Graded),
    /// The call produced no verdict; the ML verdict stands.
    Failed(Failed),
}

/// An interpretation pass that produced a grade. Serialized fields are in wire
/// order.
#[derive(Debug, Clone, Serialize)]
pub struct Graded {
    /// The LLM's raw trinary verdict.
    pub grade: LlmGrade,
    /// The blended verdict.
    #[serde(serialize_with = "label")]
    pub outcome: Classification,
    /// Whether cleave independently surfaced a hostile finding on this sample.
    /// Carried out of the blend because it is what earns a corroborated
    /// escalation, and what places an interpreted-suspicious verdict nearer the
    /// hostile boundary than the middle of the band.
    #[serde(skip)]
    pub corroborated: bool,
    /// Blended confidence in `[0, 1]`.
    #[serde(rename = "conf")]
    pub blended: f32,
    /// One-sentence rationale from the model.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub interpretation: String,
    /// Model name targeted by this pass.
    pub model: String,
    /// Whether the render carried text addressed to the grader rather than to a
    /// human reading the program (see `addresses_the_analyzer`). Surfaced as
    /// `inject: true` so an operator can see that a clearing verdict was
    /// distrusted — and that the sample tried.
    #[serde(rename = "inject", skip_serializing_if = "is_false")]
    pub analyzer_directed: bool,
    /// Whether the grade was replayed from the verdict cache instead of queried.
    ///
    /// A cached pass is the difference between a request that took a minute and
    /// one that took a tenth of a second, so it is logged rather than left to be
    /// inferred from the timing. Not serialized: it describes how this run got
    /// its answer, not the answer.
    #[serde(skip)]
    pub cached: bool,
    /// The ML verdict as it stood before this interpretation was folded in.
    ///
    /// The blend overwrites `class`, `probability` and `lvl` in place, and the
    /// `ml` section of the report reports the scan's *final* answer — so without
    /// this the number the model actually produced survives only in an INFO log
    /// line. Recorded here, in the `llm` section, because it is a fact about the
    /// second opinion rather than a second ML reading.
    pub before: MlVerdict,
}

/// An interpretation pass that was attempted and failed. It repeats the ML
/// verdict as `outcome`/`conf` so the `llm` section is still self-contained.
#[derive(Debug, Clone, Serialize)]
pub struct Failed {
    /// The ML class, unchanged.
    #[serde(rename = "outcome", serialize_with = "label")]
    pub class: Classification,
    /// The raw ML probability.
    #[serde(rename = "conf")]
    pub prob: f32,
    /// Model name targeted by this pass.
    pub model: String,
    /// See [`Graded::analyzer_directed`].
    #[serde(rename = "inject", skip_serializing_if = "is_false")]
    pub analyzer_directed: bool,
    /// Why the call produced no verdict (connection, timeout, unparseable
    /// reply, …).
    pub error: String,
}

/// Where ML had the sample before the LLM was consulted. Serialized as the
/// `llm.before` object.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct MlVerdict {
    /// The class ML decided on its own.
    #[serde(serialize_with = "label")]
    pub class: Classification,
    /// The raw model probability, before any steer.
    pub prob: f32,
    /// ML's fired level, omitted in manual-threshold mode where no level axis
    /// applies.
    #[serde(skip_serializing_if = "Level::is_manual")]
    pub lvl: Level,
}

/// A class as its word: the `llm` section spells classes out, where `ml`
/// carries the ordinal.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's `serialize_with` passes the field by reference"
)]
pub(super) fn label<S: serde::Serializer>(
    class: &Classification,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_str(class)
}

pub(super) const fn is_false(value: &bool) -> bool {
    !*value
}

/// The most of the remaining confidence range the LLM may move the score, in
/// either direction. The LLM is one fallible opinion formed over an
/// attacker-controlled render; it argues a *direction*, it does not get to set
/// the number. Capping the move as a fraction of the distance to the bound keeps
/// the steer proportionate at every starting score and keeps ML's ranking intact
/// — two files the LLM grades identically stay ordered by what ML thought of
/// them, which a flat replacement constant destroyed.
pub(super) const MAX_STEER: f32 = 0.33;

/// Move `p` at most [`MAX_STEER`] of the way toward 1.0 (`toward_severe`) or 0.0.
/// Monotonic and range-preserving: the result stays in `[0, 1]` and never crosses
/// the bound it moves toward.
pub(super) fn steer(p: f32, toward_severe: bool) -> f32 {
    if toward_severe {
        p + (1.0 - p) * MAX_STEER
    } else {
        p * (1.0 - MAX_STEER)
    }
}

/// Where the ML verdict sits on the model's calibrated false-positive axis, and
/// where the band boundaries are. This is the *only* coordinate the two decision
/// paths share: `Model::decide_swept` classifies by level, and for route-policy
/// filetypes the reported `probability` may be a per-route score or a learned
/// blend's output, so probability space is not comparable across files. Level
/// space is, which is why proximity is measured here.
#[derive(Debug, Clone, Copy)]
pub struct LevelContext {
    /// ML's fired level (`ml.lvl`): the lowest FP level (per 100M benigns) at
    /// which this file's hostile decision fires.
    pub fired: Level,
    /// Active deploy level (`-l`) — the suspicious|hostile boundary. `None` in
    /// manual-threshold mode.
    pub active: Option<u16>,
    /// Highest level on the model's grid, which caps the suspicious ceiling.
    pub grid_max: u16,
}

impl LevelContext {
    /// Whether the calibration placed this file at a real level at all.
    ///
    /// Broader than [`Self::ml_admits`], which additionally requires the fired
    /// level to sit at or below a cutoff. Being placed anywhere is itself the
    /// signal: across 1072 measured samples only 4 of 655 benign are placed,
    /// against 226 of 417 malicious, which is a better exchange rate than any
    /// probability floor on that population.
    ///
    /// [`Level::Clean`] is ML seeing nothing and is not a placement; counting
    /// it would make this admission unconditional in level mode.
    /// Manual-threshold mode has no axis to read and abstains.
    pub(super) fn ml_placed(self) -> bool {
        matches!(self.fired, Level::At(_))
    }

    /// Whether ML's own position admits this file to the LLM: it fires somewhere
    /// on the grid, no looser than the cutoff. [`Level::Clean`] (never fires) is
    /// ML seeing nothing and is not an admission.
    ///
    /// `None` resolves to [`Self::grid_max`], which makes the default admission
    /// "ML fired at all" — the most inclusive line that still means something,
    /// and the one that cannot drift: a literal ceiling stops meaning *the whole
    /// grid* the moment the calibrated grid is re-cut (`level_confidence` already
    /// reserves rungs for an L50000 grid). It also costs almost nothing over a
    /// tighter literal, because the benign quantile runs out of tail resolution
    /// long before the ceiling: in a typical bundle more than half the routes
    /// have the same threshold at L10000 and L25000, and most of the rest differ
    /// by a few hundredths.
    ///
    /// Manual-threshold mode has no calibrated axis to read, so ML abstains and
    /// the caller's remaining admissions — an elevated cleave finding, a
    /// non-benign class — carry the decision. Nothing is lost: with
    /// operator-set thresholds, a score above them already lands as non-benign.
    pub(super) fn ml_admits(self, min_level: Option<u16>) -> bool {
        let cutoff = min_level.unwrap_or(self.grid_max);
        matches!(self.fired, Level::At(fired) if fired <= cutoff)
    }

    /// Whether ML placed this file within a single [`MAX_STEER`] of the hostile
    /// boundary — the active deploy level — measured in confidence space
    /// (`level_confidence` is the calibrated, monotone projection of the level
    /// axis, and the same table that produces `ml.conf`).
    ///
    /// Symmetric by construction: `min`/`max` means the same expression covers a
    /// crossing in either direction, because "the gap between ML's position and
    /// the boundary is no wider than one steer" does not care which side ML is on.
    ///
    /// Abstains (returns `true`) in manual-threshold mode: with no level table
    /// there is no calibrated axis to measure against, so the gate has nothing to
    /// say and the remaining guards carry the decision.
    pub(super) fn within_one_steer_of_hostile_boundary(self) -> bool {
        let (Level::Clean | Level::At(_), Some(active)) = (self.fired, self.active) else {
            return true;
        };
        let (a, b) = (
            band_confidence(self.fired),
            band_confidence(Level::At(active)),
        );
        steer(a.min(b), true) >= a.max(b)
    }
}

/// A level's confidence as a `0.0..=1.0` fraction. `level_confidence` is the
/// same pessimistic table that produces `ml.conf`, so proximity is measured on
/// exactly the axis an operator reads.
pub(super) fn band_confidence(level: Level) -> f32 {
    f32::from(crate::engine::level_confidence(level).unwrap_or(0)) / 100.0
}

/// What the render and the ML verdict jointly permit the LLM to do. Bundled so
/// [`blend`] keeps a readable signature as the guards accumulate.
#[derive(Debug, Clone, Copy)]
pub(super) struct Evidence {
    /// Render is mostly readable source rather than escaped bytes.
    pub(super) readable: bool,
    /// Render carries text addressed to the grader (see [`addresses_the_analyzer`]).
    pub(super) analyzer_directed: bool,
    /// cleave independently surfaced a hostile (`H`) finding.
    pub(super) hostile_finding: bool,
    /// Where ML placed the file on the calibrated FP axis.
    pub(super) levels: LevelContext,
}

impl Evidence {
    /// Whether the LLM may move the verdict from band `from` to band `to`.
    ///
    /// Only the **hostile** boundary has to be earned. The two boundaries are not
    /// the same kind of claim:
    ///
    /// - Crossing into (or out of) hostile spends the deploy level's
    ///   false-positive budget — `-l` *is* an FP-per-100M budget. Asserting a file
    ///   belongs in that budget when ML placed it nowhere near is a calibration
    ///   claim one fallible opinion cannot back, so ML must already sit within a
    ///   steer of the line.
    /// - The suspicious boundary is a routing decision — "should a human look?" —
    ///   with no budget attached. Two detectors disagreeing is precisely the
    ///   signal that one should, so gating it would defeat the purpose: an ML
    ///   false negative sits at [`Level::Clean`] by definition, which no bounded steer
    ///   can lift. That is the case `--interpret` exists to catch.
    ///
    /// The one relaxation on the hostile side is an escalation corroborated by a
    /// cleave hostile finding — two independent detectors agreeing with ML as the
    /// outlier. Escalation-only, so a sample can never corroborate its own clearing.
    pub(super) fn may_cross(self, from: Classification, to: Classification) -> bool {
        if from == to {
            return true;
        }
        if from != Classification::Hostile && to != Classification::Hostile {
            return true;
        }
        if to > from {
            return self.hostile_finding || self.levels.within_one_steer_of_hostile_boundary();
        }
        // Leaving hostile is permitted except from the strictest rung there is.
        //
        // It is not gated on the same proximity test as an escalation: that
        // would make the LLM's ability to correct a false positive shrink as ML
        // grows more confidently wrong — backwards for the case `--interpret`
        // exists to catch. Benign tools such as yt-dlp, gallery-dl, androguard,
        // crawl4ai and pyarmor can sit at L1 with a correct exoneration in the
        // same record.

        //
        // The step lands on *suspicious*, never benign, so a real threat the LLM
        // wrongly clears is routed for review rather than released, and how far
        // into the suspicious band it lands scales with how deep ML fired (see
        // `crate::engine::softened_level`). `L0` is the exception: nothing one
        // fallible opinion says moves a file off the tightest budget the grid has.
        self.levels.fired != Level::At(0)
    }
}

/// The class one severity step from `ml` in the direction of `target`, or `ml`
/// when it is already there. Caps how far a single LLM opinion can move the
/// verdict: benign and hostile are two steps apart, so neither can reach the
/// other on the model's word alone.
pub(super) fn one_step_toward(ml: Classification, target: Classification) -> Classification {
    use std::cmp::Ordering;
    match target.cmp(&ml) {
        Ordering::Greater => match ml {
            Classification::Benign => Classification::Suspicious,
            _ => Classification::Hostile,
        },
        Ordering::Less => match ml {
            Classification::Hostile => Classification::Suspicious,
            _ => Classification::Benign,
        },
        Ordering::Equal => ml,
    }
}

/// Blend the ML verdict with the LLM grade into `(outcome, confidence)`.
///
/// Two bounds, symmetric in both directions:
///
/// 1. **The LLM may move the verdict at most one severity step, and the score at
///    most [`MAX_STEER`] toward the bound it argues for.**
/// 2. **A band crossing must be earned**: ML must already have placed the file
///    within one steer of that boundary on the calibrated FP axis (see
///    [`Evidence::may_cross`]), or cleave must independently corroborate it.
///    Otherwise the score steers but the band holds.
///
/// ML decides; the LLM steers. The second bound is what makes the first
/// load-bearing — the verdict travels downstream as `ml.lvl`, so bounding only
/// the score would leave the number that actually carries the class unguarded.
///
/// - **LLM more severe** → step one rung up and steer the score up. It may have
///   seen badness ML missed, so an ML-benign file the LLM calls hostile becomes
///   *suspicious* (review), not hostile — a two-step jump on one model's word,
///   over input the author controls, is not evidence enough to block.
/// - **They agree** → steer toward the pole the verdict already sits at:
///   corroborated malice raises the score, a corroborated clean file lowers it.
/// - **LLM less severe** → step one rung down and steer the score down, but only
///   when its read is trustworthy. Two things make it untrustworthy, and either
///   discards the clear outright, leaving the ML verdict exactly as it was (a
///   third, a cleave hostile finding, is below):
///   an **opaque render** (packed/escaped bytes — a text model cannot clear what
///   it cannot read) or **analyzer-directed text** in the sample (see
///   [`addresses_the_analyzer`]; a clear is precisely what an injected sample is
///   fishing for).
///
/// The trust requirement covers *every* softening, not just a class drop — an
/// agreed-benign score also moves down, so it is gated the same way. The
/// resulting invariant is the one worth remembering:
///
/// A third discards it too: a **cleave hostile finding**, which the LLM's prose
/// cannot overrule.
///
/// > When the LLM's read is untrusted, the blend never lowers the class and never
/// > lowers the score.
///
/// Escalation is deliberately ungated: an attacker gains nothing by talking their
/// own sample up, so only the softening path is worth attacking.
pub(super) fn blend(
    ml: Classification,
    ml_prob: f32,
    llm: LlmGrade,
    ev: Evidence,
) -> (Classification, f32) {
    use std::cmp::Ordering;
    let p = ml_prob.clamp(0.0, 1.0);
    let target = llm.classification();
    // The LLM is trusted to make a sample look *worse* unconditionally, but to
    // make it look *better* only when its read is credible: the render must be
    // readable (a text model cannot honestly clear escaped bytes) and free of
    // text aimed at the grader. Either failing discards the softening entirely,
    // leaving ML's class and score exactly as they were. So does an independent
    // cleave hostile finding: the LLM reads text the author wrote, so a prompt
    // injection the phrase list misses must still not argue a sample past a
    // detector that never read its prose.
    let may_soften = ev.readable && !ev.analyzer_directed && !ev.hostile_finding;
    // A class move is capped at one rung *and* has to clear the proximity gate;
    // when it does not, the score still steers but the band holds.
    let stepped = |toward_severe: bool| {
        let one = one_step_toward(ml, target);
        let class = if ev.may_cross(ml, one) { one } else { ml };
        (class, steer(p, toward_severe))
    };
    match target.cmp(&ml) {
        // Corroborated escalation to hostile crosses both rungs. The one-rung cap
        // exists because "a two-step jump on *one model's word*, over input the
        // author controls, is not evidence enough to block" — and here it is not
        // one model's word: cleave independently surfaced a hostile finding and
        // the LLM independently graded the sample hostile, with ML the lone
        // outlier. Escalation-only, so a sample still cannot corroborate its own
        // clearing, and `may_cross` still has to admit the crossing.
        Ordering::Greater if target == Classification::Hostile && ev.hostile_finding => {
            let class = if ev.may_cross(ml, target) {
                target
            } else {
                one_step_toward(ml, target)
            };
            (class, steer(p, true))
        }
        Ordering::Greater => stepped(true),
        // Agreement is corroboration, so it firms up the verdict already held:
        // toward 1.0 for a flagged file…
        Ordering::Equal if ml != Classification::Benign => (ml, steer(p, true)),
        // …and toward 0.0 for a clean one, which is a softening like any other.
        Ordering::Equal => (ml, if may_soften { steer(p, false) } else { p }),
        Ordering::Less if may_soften => stepped(false),
        Ordering::Less => (ml, p),
    }
}

/// Assemble a successful [`Interpretation`] from a grade + reason, applying the
/// agreement-adjusted blend against the current ML verdict.
pub(super) fn blended(
    model: &str,
    ml_class: Classification,
    ml_prob: f32,
    grade: LlmGrade,
    reason: String,
    ev: Evidence,
    cached: bool,
) -> Interpretation {
    let (outcome, conf) = blend(ml_class, ml_prob, grade, ev);
    Interpretation::Graded(Graded {
        grade,
        outcome,
        blended: conf,
        interpretation: reason,
        model: model.to_string(),
        analyzer_directed: ev.analyzer_directed,
        before: MlVerdict {
            class: ml_class,
            prob: ml_prob,
            lvl: ev.levels.fired,
        },
        corroborated: ev.hostile_finding,
        cached,
    })
}
