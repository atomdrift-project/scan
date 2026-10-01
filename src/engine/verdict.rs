//! Verdict arithmetic: how decisions outrank each other, where a synthesized
//! verdict sits on the level axis, and the trait floor.

use std::cmp::Ordering;
use std::collections::HashSet;

use cleave::Criticality;
use cleave::types::CompactTrait;

use super::{EmbeddedFile, MemberEvals, compact_crit};
use crate::model::{Classification, Decision, Level, Model};

/// Map a level marker (`ml.lvl`) to a pessimistic human-facing confidence percent.
///
/// This is a display/export confidence, not the model probability (`ml.prob`) and
/// not a posterior probability. The table is intentionally integer-valued and
/// strictly separated across the calibrated deploy grid. `25001`/`25002` (and
/// `50001`/`50002` for an L50000 grid) are the trait floor's former off-grid
/// markers (`grid_max + 1/2`); nothing places a verdict there now, but they
/// stay ranked so a stored level reads as it always did.
#[must_use]
pub const fn level_confidence(level: Level) -> Option<u8> {
    match level {
        Level::Manual => None,
        Level::Clean => Some(0),
        Level::At(0) => Some(100),
        Level::At(1) => Some(99),
        Level::At(2) => Some(98),
        Level::At(3) => Some(97),
        Level::At(4) => Some(96),
        Level::At(5) => Some(95),
        Level::At(10) => Some(94),
        Level::At(20) => Some(93),
        Level::At(30) => Some(92),
        Level::At(40) => Some(91),
        Level::At(50) => Some(90),
        Level::At(60) => Some(89),
        Level::At(70) => Some(88),
        Level::At(80) => Some(87),
        Level::At(90) => Some(86),
        Level::At(100) => Some(85),
        Level::At(200) => Some(82),
        Level::At(300) => Some(80),
        Level::At(500) => Some(78),
        Level::At(1000) => Some(75),
        Level::At(2000) => Some(66),
        Level::At(5000) => Some(54),
        Level::At(7500) => Some(49),
        Level::At(10000) => Some(45),
        Level::At(15000) => Some(38),
        Level::At(20000) => Some(33),
        Level::At(25000) => Some(29),
        Level::At(25001) => Some(28),
        Level::At(25002) => Some(27),
        Level::At(50000) => Some(17),
        Level::At(50001) => Some(16),
        Level::At(50002) => Some(15),
        Level::At(n) if n > 50002 => Some(15),
        Level::At(n) if n > 25002 => Some(26),
        Level::At(n) if n > 25000 => Some(28),
        Level::At(n) if n > 20000 => Some(29),
        Level::At(n) if n > 15000 => Some(33),
        Level::At(n) if n > 10000 => Some(38),
        Level::At(n) if n > 7500 => Some(45),
        Level::At(n) if n > 5000 => Some(49),
        Level::At(n) if n > 2000 => Some(54),
        Level::At(n) if n > 1000 => Some(66),
        Level::At(n) if n > 500 => Some(75),
        Level::At(n) if n > 300 => Some(78),
        Level::At(n) if n > 200 => Some(80),
        Level::At(n) if n > 100 => Some(82),
        Level::At(n) if n > 90 => Some(85),
        Level::At(n) if n > 80 => Some(86),
        Level::At(n) if n > 70 => Some(87),
        Level::At(n) if n > 60 => Some(88),
        Level::At(n) if n > 50 => Some(89),
        Level::At(n) if n > 40 => Some(90),
        Level::At(n) if n > 30 => Some(91),
        Level::At(n) if n > 20 => Some(92),
        Level::At(n) if n > 10 => Some(93),
        Level::At(n) if n > 5 => Some(94),
        Level::At(_) => Some(95),
    }
}

/// Where a hostile verdict lands when the LLM clears it.
///
/// Positioned by how deep ML fired rather than pinned: the level ML reached is
/// the budget for how far one contrary opinion may move it. A file that fired at
/// the loosest hostile rung barely survived the boundary, so a clear pushes it
/// most of the way across the suspicious band; a file that fired near the
/// strictest rung is moved barely past it.
///
/// Geometric, for the same reason as [`interpreted_level`] — the axis is. At the
/// shipped `-l 25` an ML `L1` lands at `L31`, an `L12` at `L248`.
///
/// `L0` never reaches here: `interpret::Evidence::may_cross` refuses the
/// crossing outright.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a positive value clamped to `ceiling`, which is at most `grid_max`"
)]
pub(super) fn softened_level(ml_level: Level, active_level: Option<u16>, grid_max: u16) -> Level {
    let Some(active) = active_level else {
        return Level::Manual;
    };
    let ceiling = crate::model::capped_suspicious_level(grid_max);
    let floor = active.saturating_add(1);
    if floor >= ceiling {
        return Level::At(ceiling);
    }
    // No level to scale by (manual-threshold mode) falls back to the midpoint.
    let fraction = match ml_level {
        Level::At(lvl) if lvl > 0 && active > 0 => {
            (f64::from(lvl) / f64::from(active)).clamp(0.0, 1.0)
        }
        _ => 0.5,
    };
    let placed = f64::from(floor) * (f64::from(ceiling) / f64::from(floor)).powf(fraction);
    Level::At((placed.round() as u16).clamp(floor, ceiling))
}

/// Place a synthesized verdict inside its band, the way an interpreted one is
/// placed.
///
/// A verdict that did not come from the model has no measured firing level, but
/// a consumer that reads only the number still has to land in the right band.
/// This produces the level that decodes back to `outcome` under the same rules
/// the model's own verdicts obey — the deploy level for hostile, and a point
/// inside the suspicious band for suspicious, lower when something else
/// corroborates it.
///
/// [`Level::Manual`] in manual-threshold mode, where no calibrated level applies.
#[must_use]
pub fn synthesized_level(model: &Model, outcome: Classification, corroborated: bool) -> Level {
    interpreted_level(
        model.active_level(),
        model.grid_max(),
        outcome,
        corroborated,
    )
}

/// The level a verdict that did not come from the model is given, so a
/// level-only consumer decodes it back to `outcome` under the model's own rules
/// ([`crate::model::verdict_for_level`]):
/// - **Hostile** → the active deploy level (`-l`), the weakest rung still inside
///   the hostile budget.
/// - **Suspicious** → a point inside the suspicious band, nearer the hostile
///   boundary when something else corroborates it.
/// - **Benign** → [`Level::Clean`].
///
/// `active_level` is `None` in manual-threshold mode (no grid); hostile and
/// suspicious then return [`Level::Manual`], as a genuine ML verdict does there.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a positive value clamped to `ceiling`, which is at most `grid_max`"
)]
pub(super) fn interpreted_level(
    active_level: Option<u16>,
    grid_max: u16,
    outcome: Classification,
    corroborated: bool,
) -> Level {
    match (outcome, active_level) {
        (Classification::Benign, _) => Level::Clean,
        (_, None) => Level::Manual,
        (Classification::Hostile, Some(active)) => Level::At(active),
        (Classification::Suspicious, Some(active)) => {
            let ceiling = crate::model::capped_suspicious_level(grid_max);
            let floor = active.saturating_add(1);
            if floor >= ceiling {
                return Level::At(ceiling);
            }
            // Placed *within* the band rather than at its weakest rung.
            //
            // Pinning every interpreted-suspicious verdict to the ceiling put all
            // of them on one number, which threw away the ordering the level axis
            // exists to carry: a sample cleave independently flagged and a sample
            // resting on the LLM's word alone both read as 3000.
            //
            // Geometric, because the axis is: `level_confidence` compresses
            // 200→82, 1000→75, 2000→66, 5000→54, so a *linear* midpoint of
            // 26..3000 sits at 1513 and reads as barely-suspicious. The geometric
            // one lands near 279, which is where the middle of the band actually
            // is in confidence terms.
            //
            // Corroboration moves it a quarter of the way in instead of half —
            // nearer the hostile boundary, because two detectors agreeing is a
            // stronger claim than one.
            let fraction = if corroborated { 0.25 } else { 0.5 };
            let placed = f64::from(floor) * (f64::from(ceiling) / f64::from(floor)).powf(fraction);
            Level::At((placed.round() as u16).clamp(floor, ceiling))
        }
    }
}

/// The worst (highest-outranking) member evaluation, or `None` when no member
/// was evaluated. Iteration is id order == report entry order, so ties keep
/// the earliest member exactly as the old in-loop fold did.
pub(super) fn worst_member(evals: &MemberEvals) -> Option<Decision> {
    evals.values().map(EmbeddedFile::decision).reduce(graver)
}

/// The worst verdict the *model alone* reached across an archive's members.
///
/// Members the trait floor raised contribute nothing here, and correctly so:
/// the floor fires only on a model-Benign decision, so on those members the
/// model's own reading was benign. Everywhere else the stored verdict is the
/// model's, untouched. `None` means no member was scored at all, which is not
/// the same as every member being clean.
pub(super) fn worst_member_model(evals: &MemberEvals) -> Option<Decision> {
    evals
        .values()
        .filter(|member| member.floor.is_none())
        .map(EmbeddedFile::decision)
        .reduce(graver)
}

/// The gravest trait-floor firing anywhere in an archive.
pub(super) fn worst_member_floor(evals: &MemberEvals) -> Option<FloorDecision> {
    evals
        .values()
        .filter_map(|member| member.floor)
        .reduce(FloorDecision::worse_of)
}

/// Confident crit-5 findings the hostile arm needs — the anchor the rest of the
/// arm corroborates.
///
/// Was `1`, on the reasoning that one suffices *because* it is corroborated:
/// real malware routinely carries a single hostile trait beside supporting
/// crit-4s (`rex-powershell`, `openclaude`, and a rhadamanthys coinminer all
/// have exactly one), and an earlier attempt at two lost all three.
///
/// Raised to `2` (2026-08-27) after the single-anchor arm was observed marking
/// model-clean legitimate packages hostile on one trait that describes the
/// package's own function: `ansible-core` on a PowerShell base64-exec trait —
/// which is how its Windows connection plugin works — with `confident_hostile=1,
/// severe=3, families=3`, and Cherry Studio (`v2.0.9.tar.gz`) at `1/4/4`. Both
/// fired the arm at its exact minimum. A single crit-5 is one witness however
/// many crit-4s sit beside it, and the crit-4s in both cases were describing the
/// same legitimate behavior from other angles.
///
/// The named regressions above are the known cost; re-measure them alongside the
/// gauntlet missed-sample pool before relaxing this again.
const TRAIT_FLOOR_HOSTILE_CRIT5: u32 = 2;

/// Confident severe findings (crit-5 *or* crit-4) the hostile arm needs in
/// total. Rejects the thin pair that family diversity alone admits: a
/// ScreenConnect RMM signature plus `pe-large-without-material-section` is two
/// findings in two families, but a generic PE-layout anomaly is not evidence of
/// malice, and that pair was marking a stock `libwebp.dll` hostile.
const TRAIT_FLOOR_HOSTILE_SEVERE: u32 = 3;

/// Distinct trait families the hostile arm's severe findings must span.
/// Counting alone treats one behavior described three ways as three independent
/// witnesses: the static-keys false positive presented as
/// `rust-inline-hook-hijack`, `rust-hook-byte-copy`, and
/// `rust-mprotect-hook-patch` — three findings, one directory, overlapping
/// regexes over the same two tokens. Corroboration has to come from somewhere
/// else in the tree to be corroboration at all.
///
/// Counted over both severe tiers, so crit-5 and crit-4 families together have
/// to reach it — the same set the arm's `severe()` total draws from.
///
/// Was two. The argument for two was that every observed false positive was a
/// *single*-family cluster, already rejected, while three would have discarded
/// `darkglitch` — a Python RAT whose hostile traits span only `backdoor/rat` and
/// `backdoor/tasking`. That argument was made against a four-deep family, and
/// [`TRAIT_FLOOR_FAMILY_DEPTH`] is now two: `darkglitch`'s pair collapses to one
/// family at this depth regardless, so three no longer costs what it did.
///
/// Raised to three (2026-08-27) because two families is a materially weaker claim
/// once a family is a *kind* of behavior rather than a technique. `PyAutoIt` — a
/// legitimate AutoIt wrapper — reached the hostile arm at `confident_hostile=4,
/// severe=5, families=2`, on the input-synthesis and window-manipulation traits
/// that are AutoIt's entire purpose. It also makes the two arms consistent: the
/// suspicious arm has always required three (see
/// [`TRAIT_FLOOR_SUSPICIOUS_FAMILIES`]), and the hostile arm demanding *less*
/// diversity than the suspicious one had no principle behind it.
const TRAIT_FLOOR_HOSTILE_FAMILIES: usize = 3;

/// Distinct trait families the suspicious arm's crit-4 findings must span.
/// Subsumes a count — n families need n findings — so this is the arm's only
/// threshold.
///
/// Set above [`TRAIT_FLOOR_HOSTILE_FAMILIES`] deliberately. The hostile arm has
/// a confident crit-5 anchoring it; this arm has nothing but the breadth of its
/// own evidence, so it has to be broader to make the same claim.
const TRAIT_FLOOR_SUSPICIOUS_FAMILIES: usize = 3;

/// Trait-hierarchy depth that defines a family: `objectives/anti-static` rather
/// than a leaf path or the finer `objectives/anti-static/obfuscation/string`.
/// One definition, used by both arms.
///
/// Was four (`objectives/evasion/process/hook`), chosen as the shallowest depth
/// that gave nothing up: at three, the static-keys cluster correctly collapsed to
/// one family, but so did `darkglitch` — a Python RAT whose three hostile traits
/// are genuinely distinct capabilities under `command-and-control/backdoor`
/// (`rat/multi` and `tasking/filesystem`) — and it lost its verdict entirely.
///
/// Shallowed to two (2026-08-27). Four proved too fine to be corroboration on the
/// gauntlet false-positive pool: `ansible-core` reached the hostile arm with
/// three "independent" families that are all one idea — a PowerShell base64
/// decode-and-execute, which is how its Windows connection plugin works, counted
/// once per subdirectory it was spelled in. Two makes a family a *kind* of
/// behavior rather than a technique, so restating one behavior at different
/// depths can no longer corroborate itself.
///
/// The darkglitch class of verdict is the known cost, and it is the thing to
/// re-measure first if the missed-sample pool regresses.
const TRAIT_FLOOR_FAMILY_DEPTH: usize = 2;

/// The family a finding belongs to: the first [`TRAIT_FLOOR_FAMILY_DEPTH`]
/// segments of its hierarchy path. A trait id is `path::leaf`
/// (`objectives/evasion/process/hook/inline::rust-inline-hook-hijack`); an id
/// carrying no path, or a shorter one, is its own family rather than joining a
/// catch-all bucket that would let unrelated traits corroborate each other.
pub(super) fn trait_family(id: &str) -> &str {
    let path = id.split("::").next().unwrap_or(id);
    path.match_indices('/')
        .nth(TRAIT_FLOOR_FAMILY_DEPTH - 1)
        // `cut` indexes an ASCII '/', so it is a char boundary by construction;
        // `get` keeps that fact from resting on a panic.
        .and_then(|(cut, _)| path.get(..cut))
        .unwrap_or(path)
}

/// Minimum cleave confidence (`c`) for a finding to count toward the trait floor.
/// Low-confidence high-crit findings are exactly the incidental ones that fire on
/// busy benign binaries (e.g. a couple of speculative crit-4s among hundreds of
/// findings), so the floor only acts on evidence cleave is sure about. Measured:
/// at 0.76 the dropper keeps all 4 crit-4s and the PE keeps all 3 crit-5s, while
/// no /usr/bin benign trips the crit-5 arm.
const TRAIT_FLOOR_MIN_CONFIDENCE: f32 = 0.76;

/// Confidence-filtered crit-5/crit-4 tallies, with the trait families each tier
/// drew from. Only findings scoring `>= TRAIT_FLOOR_MIN_CONFIDENCE` are counted
/// at all, so an unscored or hedged trait contributes to neither arm.
#[derive(Default)]
struct TraitFloorCounts<'a> {
    hostile: u32,
    suspicious: u32,
    hostile_confidence: f32,
    suspicious_confidence: f32,
    /// Families across both severe tiers — the hostile arm's diversity test.
    severe_families: HashSet<&'a str>,
    /// Families among the crit-4s alone — the suspicious arm's.
    suspicious_families: HashSet<&'a str>,
}

impl TraitFloorCounts<'_> {
    /// Confident severe findings across both tiers.
    const fn severe(&self) -> u32 {
        self.hostile + self.suspicious
    }
}

/// `well-known/` subtrees that positively identify a sample as a *named piece of
/// software* rather than as a threat. `malware/` and `unwanted/` are pointedly
/// absent: those recognizers identify a sample too, but identifying it as
/// malware is not a reason to hold back.
const KNOWN_SOFTWARE_PREFIXES: [&str; 5] = [
    "well-known/app/",
    "well-known/lib/",
    "well-known/tool/",
    "well-known/game/",
    "well-known/dual-use/",
];

/// Whether cleave confidently recognized this sample as a named application,
/// library, tool or known dual-use utility.
///
/// Used to raise — never to waive — the hostile arm's anchor requirement. A
/// sample cleave can name is one whose hostile traits are far more likely to be
/// describing the program's own advertised function: `ansible-core`'s PowerShell
/// base64-exec *is* its Windows connection plugin, and eight
/// `well-known/app/infrastructure` Ansible recognizers fire on the same render
/// while the floor promotes it to hostile on that one trait.
fn identified_as_known_software(findings: &[CompactTrait]) -> bool {
    findings.iter().any(|f| {
        f.confidence >= TRAIT_FLOOR_MIN_CONFIDENCE
            && KNOWN_SOFTWARE_PREFIXES
                .iter()
                .any(|prefix| f.id.starts_with(prefix))
    })
}

fn trait_floor_counts(findings: &[CompactTrait]) -> TraitFloorCounts<'_> {
    let mut out = TraitFloorCounts::default();
    for f in findings {
        let conf = f.confidence;
        if conf < TRAIT_FLOOR_MIN_CONFIDENCE {
            continue;
        }
        let family = trait_family(&f.id);
        match compact_crit(f) {
            Criticality::Hostile => {
                out.hostile += 1;
                out.hostile_confidence = out.hostile_confidence.max(conf);
                out.severe_families.insert(family);
            }
            Criticality::Suspicious => {
                out.suspicious += 1;
                out.suspicious_confidence = out.suspicious_confidence.max(conf);
                out.severe_families.insert(family);
                out.suspicious_families.insert(family);
            }
            _ => {}
        }
    }
    out
}

/// Which arm of the trait floor fired.
///
/// The two arms answer different questions — "this is malicious" versus "a
/// human should look" — and they are reached by different evidence, so a
/// consumer that reports or meters the floor wants to tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloorArm {
    /// A confident hostile (crit-5) anchor, corroborated across families.
    Crit5,
    /// Confident suspicious (crit-4) traits spanning enough families.
    Crit4,
}

impl FloorArm {
    /// Stable identifier for logs and metrics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Crit5 => "crit5",
            Self::Crit4 => "crit4",
        }
    }
}

/// What the trait floor concluded, on the occasions it concludes anything.
///
/// Returned rather than applied, so a caller can record that the floor — not
/// the model — is what convicted an artifact. `apply_trait_floor` is the
/// in-place form scan's own paths use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloorDecision {
    /// The class the floor raises the verdict to.
    pub class: Classification,
    /// Confidence of the trait evidence that carried it, in `[0, 1]`. This is
    /// cleave's confidence in its own match, not a model probability.
    pub confidence: f32,
    /// Synthetic level placing the override inside its band, or
    /// [`Level::Manual`] under manual thresholds.
    pub level: Level,
    /// Which arm fired.
    pub arm: FloorArm,
    /// Confident crit-5 findings counted.
    pub hostile: u32,
    /// Confident crit-4 findings counted.
    pub suspicious: u32,
    /// Trait families the firing arm measured its evidence across.
    pub families: usize,
}

impl FloorDecision {
    /// Confident severe findings across both tiers.
    #[must_use]
    pub const fn severe(&self) -> u32 {
        self.hostile + self.suspicious
    }

    /// The worse of two firings, for summarizing an archive by its members.
    ///
    /// The graver class wins; between two of the same class, the better
    /// corroborated one does.
    #[must_use]
    pub fn worse_of(self, other: Self) -> Self {
        match self.class.cmp(&other.class) {
            Ordering::Greater => self,
            Ordering::Less => other,
            Ordering::Equal => {
                if other.confidence > self.confidence {
                    other
                } else {
                    self
                }
            }
        }
    }
}

/// Trait floor: override a model-**Benign** verdict when cleave surfaced
/// high-criticality evidence the model did not act on:
///   - a hostile (crit-5) trait, corroborated by enough further severe findings
///     to reach `TRAIT_FLOOR_HOSTILE_SEVERE` across
///     `TRAIT_FLOOR_HOSTILE_FAMILIES` trait families → **Hostile**
///   - suspicious (crit-4) traits spanning
///     `TRAIT_FLOOR_SUSPICIOUS_FAMILIES` families → **Suspicious**
///
/// Both arms count only confident findings (`c >= TRAIT_FLOOR_MIN_CONFIDENCE`),
/// and both measure evidence by the families it comes from, so one behavior
/// spelled several ways cannot corroborate itself.
///
/// This is a backstop for model misses, not a second opinion: it fires only on
/// a model-Benign verdict, and every threshold above is deliberately set where
/// a lone mislabeled trait — or a cluster of near-duplicate ones — cannot reach
/// it alone.
///
/// Never lowers a model verdict, and `None` here means "the floor had nothing
/// to add", never "benign". Override levels are pinned to the same band
/// boundaries used by ordinary and interpreted verdicts, so a level-only
/// downstream consumer cannot reinterpret the override as another class.
#[must_use]
pub fn trait_floor(
    findings: &[CompactTrait],
    model_class: Classification,
    active_level: Option<u16>,
    grid_max: u16,
) -> Option<FloorDecision> {
    if model_class != Classification::Benign {
        return None;
    }
    let counts = trait_floor_counts(findings);
    // Recognized software has to clear a higher anchor before the floor will
    // *block* it. Not an exemption — one extra confident crit-5, which a genuine
    // compromise adding its own hostile behavior still reaches, while a lone
    // trait describing the program's own advertised function no longer promotes
    // a model-clean sample straight past review. Falling short here drops
    // through to the suspicious arm below, which is where "a human should look"
    // belongs.
    let required_crit5 = if identified_as_known_software(findings) {
        TRAIT_FLOOR_HOSTILE_CRIT5.saturating_add(1)
    } else {
        TRAIT_FLOOR_HOSTILE_CRIT5
    };
    if counts.hostile >= required_crit5
        && counts.severe() >= TRAIT_FLOOR_HOSTILE_SEVERE
        && counts.severe_families.len() >= TRAIT_FLOOR_HOSTILE_FAMILIES
    {
        return Some(FloorDecision {
            class: Classification::Hostile,
            confidence: counts.hostile_confidence,
            level: interpreted_level(active_level, grid_max, Classification::Hostile, true),
            arm: FloorArm::Crit5,
            hostile: counts.hostile,
            suspicious: counts.suspicious,
            // The hostile arm's diversity test spans both severe tiers.
            families: counts.severe_families.len(),
        });
    }
    if counts.suspicious_families.len() >= TRAIT_FLOOR_SUSPICIOUS_FAMILIES {
        return Some(FloorDecision {
            class: Classification::Suspicious,
            confidence: counts.suspicious_confidence,
            // The crit-4 arm by definition lacked the crit-5 anchor, but a lone
            // confident hostile trait may still be present and is corroboration.
            level: interpreted_level(
                active_level,
                grid_max,
                Classification::Suspicious,
                counts.hostile > 0,
            ),
            arm: FloorArm::Crit4,
            hostile: counts.hostile,
            suspicious: counts.suspicious,
            // The suspicious arm measures diversity among the crit-4s alone.
            families: counts.suspicious_families.len(),
        });
    }
    None
}

/// Apply [`trait_floor`] to a decision in place, and say so in the log.
///
/// The model graded these benign yet cleave is confident they carry severe
/// traits — a model gap worth investigating. INFO keeps it visible in
/// serve/worker mode (`scan=info`) without spamming default CLI runs
/// (`scan=warn`).
///
/// Returns what the floor did, so a caller reporting the model and the floor
/// as separate opinions can tell which of them convicted a file. Callers that
/// only want the verdict ignore it. Because the floor fires only on a
/// model-Benign decision, "the floor fired here" is also the whole record of
/// what the model said: benign.
pub(super) fn apply_trait_floor(
    decision: &mut Decision,
    findings: &[CompactTrait],
    active_level: Option<u16>,
    grid_max: u16,
    label: &str,
) -> Option<FloorDecision> {
    let floor = trait_floor(findings, decision.class, active_level, grid_max)?;
    decision.class = floor.class;
    decision.probability = floor.confidence;
    decision.level = floor.level;
    // One static message with the arm as a field, so an aggregator groups the
    // floor's firings together and facets them, rather than needing to know
    // both arms' wordings. Every count either arm logged before is here.
    tracing::info!(
        path = %label,
        arm = floor.arm.as_str(),
        escalated_to = %floor.class,
        confident_hostile = floor.hostile,
        confident_suspicious = floor.suspicious,
        confident_severe = floor.severe(),
        families = floor.families,
        trait_confidence = format!("{:.3}", floor.confidence),
        level = ?floor.level,
        "TRAIT FLOOR: model said benign but cleave found corroborated severe traits",
    );
    Some(floor)
}

/// Whether `candidate` should replace `current` as the dominant decision: the
/// graver class wins, and a higher probability breaks a tie.
pub(super) fn decision_outranks(candidate: &Decision, current: &Decision) -> bool {
    match candidate.class.cmp(&current.class) {
        Ordering::Greater => true,
        Ordering::Equal => candidate.probability > current.probability,
        Ordering::Less => false,
    }
}

/// The graver of two decisions, for folding: a tie keeps `current`, so the
/// earliest of equals wins.
pub(super) fn graver(current: Decision, candidate: Decision) -> Decision {
    if decision_outranks(&candidate, &current) {
        candidate
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::pipeline::root_findings;

    fn benign() -> Decision {
        Decision {
            class: Classification::Benign,
            probability: 0.1,
            threshold: 0.65,
            level: Level::Clean,
        }
    }

    /// One finding, in the family `objectives/<area>/<n>` — distinct `area`
    /// values put findings in distinct families.
    fn finding(area: &str, crit: u8, conf: f32) -> cleave::types::CompactTrait {
        cleave::types::CompactTrait {
            id: format!("objectives/{area}/sub/leaf::trait-{crit}-{conf}"),
            criticality: crit,
            confidence: conf,
            ..cleave::types::CompactTrait::default()
        }
    }

    /// `n` findings at `crit`/`conf`, each in its own family.
    fn spread(crit: u8, conf: f32, n: usize) -> Vec<cleave::types::CompactTrait> {
        (0..n)
            .map(|i| finding(&format!("area{i}"), crit, conf))
            .collect()
    }

    /// `n` findings at `crit`/`conf`, all in one family — the shape a single
    /// behavior described several ways produces.
    fn clustered(crit: u8, conf: f32, n: usize) -> Vec<cleave::types::CompactTrait> {
        (0..n)
            .map(|i| cleave::types::CompactTrait {
                id: format!("objectives/evasion/process/hook/inline::variant-{i}"),
                criticality: crit,
                confidence: conf,
                ..cleave::types::CompactTrait::default()
            })
            .collect()
    }

    #[test]
    fn family_is_the_first_two_hierarchy_segments() {
        assert_eq!(
            trait_family("objectives/evasion/process/hook/inline::rust-inline-hook-hijack"),
            "objectives/evasion"
        );
        // One behavior spelled several ways shares a family...
        assert_eq!(
            trait_family("objectives/evasion/process/hook/inline::rust-inline-hook-hijack"),
            trait_family("objectives/evasion/process/hook/inline::rust-hook-byte-copy"),
        );
        // ...and so does one behavior restated under sibling techniques, which
        // is what a four-deep family missed: `ansible-core`'s base64
        // decode-and-execute counted once per subdirectory it was spelled in,
        // and three such counts cleared a diversity test meant to require three
        // independent witnesses.
        assert_eq!(
            trait_family("objectives/anti-static/obfuscation/string/reconstruct::a"),
            trait_family("objectives/anti-static/obfuscation/eval/scripting::b"),
        );
        // Distinct objectives stay distinct.
        assert_ne!(
            trait_family("objectives/evasion/process/hook/inline::a"),
            trait_family("objectives/supply-chain/install-hook/npm::b"),
        );
        // The accepted cost: `darkglitch`'s two genuinely distinct backdoor
        // capabilities now share a family, so a verdict resting on that pair
        // alone no longer clears the diversity test. See
        // [`TRAIT_FLOOR_FAMILY_DEPTH`].
        assert_eq!(
            trait_family("objectives/command-and-control/backdoor/rat/multi::a"),
            trait_family("objectives/command-and-control/backdoor/tasking/filesystem::b"),
        );
        // A short or path-less id is its own family, never a shared bucket.
        assert_eq!(
            trait_family("micro-behaviors/mem::x"),
            "micro-behaviors/mem"
        );
        assert_eq!(trait_family("bare-id"), "bare-id");
        // The leaf trait id is never part of the family, at any path length —
        // otherwise every trait would be its own family and the diversity test
        // would pass on any cluster of near-duplicates.
        for id in [
            "objectives/evasion/process/hook/inline::rust-inline-hook-hijack",
            "objectives/a/b::leaf",
            "micro-behaviors/mem::x",
            "a::b",
        ] {
            assert!(
                !trait_family(id).contains("::"),
                "family kept the trait id: {id} -> {}",
                trait_family(id)
            );
        }
        // Two traits differing only in their leaf are one family.
        assert_eq!(
            trait_family("objectives/a/b/c/d::one"),
            trait_family("objectives/a/b/c/d::two"),
        );
    }

    #[test]
    fn two_crit5_with_a_third_severe_escalates_to_hostile_band() {
        let mut d = benign();
        let mut ts = spread(5, 0.8, 2);
        ts.push(finding("supply-chain", 4, 0.9));
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        // The probability is the strongest *crit-5*, not the strongest finding.
        assert_eq!(d.probability, 0.8);
        assert_eq!(d.level, Level::At(50));
    }

    #[test]
    fn a_lone_crit5_stays_benign() {
        let mut d = benign();
        // One hostile trait cannot carry a verdict, however confident: nothing
        // corroborates it. This is the shape that graded WannaCry hostile off a
        // single finding — and static-keys benign code with it.
        apply_trait_floor(&mut d, &spread(5, 0.98, 1), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn a_crit5_with_one_thin_corroborator_stays_benign() {
        let mut d = benign();
        // Two findings in two families still falls short of the severe count.
        // This is the real pair that marked a stock `libwebp.dll` hostile: an
        // RMM signature plus a generic PE-layout anomaly.
        let ts = vec![
            finding("command-and-control", 5, 0.9),
            finding("binary-anomaly", 4, 0.97),
        ];
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn two_crit5_corroborated_across_families_escalates_to_hostile_band() {
        let mut d = benign();
        // Two anchors plus a further severe finding.
        let mut ts = spread(5, 0.98, 2);
        ts.extend([finding("execution", 4, 0.94)]);
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.probability, 0.98);
        assert_eq!(d.level, Level::At(50));
    }

    #[test]
    fn two_families_no_longer_reach_the_hostile_arm() {
        // Enough anchors and enough severe findings, but drawn from only two
        // kinds of behavior — the `PyAutoIt` shape, where input synthesis and
        // window manipulation are the package's whole purpose. Both arms now ask
        // for three (see [`TRAIT_FLOOR_HOSTILE_FAMILIES`]).
        let mut d = benign();
        let mut ts = spread(5, 0.98, 2);
        ts.extend([finding("area0", 4, 0.94), finding("area1", 4, 0.92)]);
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_ne!(d.class, Classification::Hostile);

        // A third family is what earns it.
        let mut wider = benign();
        let mut ts = spread(5, 0.98, 2);
        ts.extend([finding("execution", 4, 0.94)]);
        apply_trait_floor(&mut wider, &ts, Some(50), 100, "test");
        assert_eq!(wider.class, Classification::Hostile);
    }

    #[test]
    fn a_lone_crit5_beside_crit4s_no_longer_reaches_the_hostile_arm() {
        // The `rex-powershell` shape, and the one `ansible-core` fired on: a
        // single crit-5 anchor with two crit-4s beside it. An accepted
        // regression — see [`TRAIT_FLOOR_HOSTILE_CRIT5`]. It does not go
        // unremarked, only unblocked: the suspicious arm still has it.
        let mut d = benign();
        let mut ts = spread(5, 0.98, 1);
        ts.extend([finding("evasion", 4, 0.8), finding("execution", 4, 0.94)]);
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_ne!(d.class, Classification::Hostile);
    }

    #[test]
    fn recognized_software_needs_one_more_anchor_before_the_hostile_arm() {
        let evidence = || {
            let mut ts = spread(5, 0.98, 2);
            ts.extend([finding("execution", 4, 0.94)]);
            ts
        };
        // Two anchors clear the arm for a sample cleave cannot name...
        let mut unknown = benign();
        apply_trait_floor(&mut unknown, &evidence(), Some(50), 100, "test");
        assert_eq!(unknown.class, Classification::Hostile);

        // ...and the same evidence does not, once it can. Blocking a named
        // application on evidence this thin is what promoted `ansible-core`.
        let mut known = benign();
        let mut with_id = evidence();
        with_id.push(cleave::types::CompactTrait {
            id: "well-known/app/infrastructure/ansible::module-utils-path".to_string(),
            criticality: 1,
            confidence: 0.95,
            ..cleave::types::CompactTrait::default()
        });
        apply_trait_floor(&mut known, &with_id, Some(50), 100, "test");
        assert_ne!(known.class, Classification::Hostile);

        // Recognizing a sample as *malware* is not recognition that holds back.
        let mut named_malware = benign();
        let mut with_malware_id = evidence();
        with_malware_id.push(cleave::types::CompactTrait {
            id: "well-known/malware/rat/darkglitch::tasking".to_string(),
            criticality: 1,
            confidence: 0.95,
            ..cleave::types::CompactTrait::default()
        });
        apply_trait_floor(&mut named_malware, &with_malware_id, Some(50), 100, "test");
        assert_eq!(named_malware.class, Classification::Hostile);
    }

    #[test]
    fn three_crit4_in_two_families_stays_benign() {
        let mut d = benign();
        // Count alone would clear the suspicious arm; breadth does not.
        let ts = vec![
            finding("evasion", 4, 0.9),
            finding("evasion", 4, 0.92),
            finding("execution", 4, 0.88),
        ];
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn two_crit5_without_further_corroboration_stays_benign() {
        let mut d = benign();
        apply_trait_floor(&mut d, &spread(5, 0.9, 2), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn a_single_family_cluster_cannot_corroborate_itself() {
        let mut d = benign();
        // Three confident crit-5s, all from `objectives/evasion/process` —
        // counts clear, diversity does not. This is the static-keys shape.
        apply_trait_floor(&mut d, &clustered(5, 0.98, 3), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
        // Same for the suspicious arm.
        let mut d = benign();
        apply_trait_floor(&mut d, &clustered(4, 0.93, 4), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn low_confidence_crit5_is_ignored() {
        let mut d = benign();
        // c < 0.76 → not counted, stays benign.
        apply_trait_floor(&mut d, &spread(5, 0.5, 3), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn three_confident_crit4_across_families_escalates_to_suspicious_band() {
        let mut d = benign();
        apply_trait_floor(&mut d, &spread(4, 0.9, 3), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Suspicious);
        assert_eq!(d.probability, 0.9);
        // Placed inside the band (51..=100) rather than pinned to its weakest
        // rung — see `interpreted_level`. No crit-5 here, so it sits at the
        // uncorroborated midpoint.
        let Level::At(lvl) = d.level else {
            panic!("a floored verdict carries a level, got {:?}", d.level);
        };
        assert!(lvl > 50 && lvl < 100, "expected inside the band, got {lvl}");
    }

    #[test]
    fn two_confident_crit4_stays_benign() {
        let mut d = benign();
        apply_trait_floor(&mut d, &spread(4, 0.9, 2), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn a_busy_file_is_not_diluted_out_of_an_escalation() {
        let mut d = benign();
        // Three confident crit-4s in distinct families, among 200 baseline
        // findings. The fraction gate this replaced scored ~0.015 here and
        // stayed benign; activity elsewhere in the file is not evidence about
        // these three.
        let mut ts = spread(4, 0.9, 3);
        ts.extend((0..200).map(|i| finding(&format!("noise{i}"), 0, 0.9)));
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Suspicious);
    }

    #[test]
    fn low_confidence_crit4_does_not_count_toward_the_trio() {
        let mut d = benign();
        // Only one confident crit-4; the other two are below threshold.
        let ts = vec![
            finding("a", 4, 0.9),
            finding("b", 4, 0.5),
            finding("c", 4, 0.6),
        ];
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    /// cleave now always writes `conf`, but reports from builds that omitted it
    /// still decode — as 0.0, "no confidence recorded", which is below the 0.76
    /// gate. So unscored crit-5s in an old report never trip the floor.
    #[test]
    fn confidence_omitted_by_an_older_build_decodes_below_threshold() {
        let mut d = benign();
        let ts: Vec<cleave::types::CompactTrait> = serde_json::from_value(serde_json::json!([
            {"id": "objectives/a/b/c::x", "crit": 5},
            {"id": "objectives/d/e/f::y", "crit": 5},
            {"id": "objectives/g/h/i::z", "crit": 4},
        ]))
        .unwrap();
        assert_eq!(
            ts[0].confidence, 0.0,
            "an omitted conf records no confidence"
        );
        apply_trait_floor(&mut d, &ts, Some(50), 100, "test");
        assert_eq!(d.class, Classification::Benign);
    }

    #[test]
    fn never_lowers_a_non_benign_verdict() {
        let mut d = benign();
        d.class = Classification::Hostile;
        d.level = Level::At(50);
        apply_trait_floor(&mut d, &spread(4, 0.9, 5), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.level, Level::At(50));
    }

    #[test]
    fn the_reported_floor_names_the_arm_and_its_evidence() {
        // Two anchors plus a further severe finding, across three families.
        let mut ts = spread(5, 0.98, 2);
        ts.extend([finding("execution", 4, 0.94)]);
        let floor = trait_floor(&ts, Classification::Benign, Some(50), 100)
            .expect("corroborated crit-5 evidence must reach the hostile arm");

        assert_eq!(floor.arm, FloorArm::Crit5);
        assert_eq!(floor.class, Classification::Hostile);
        assert_eq!(floor.confidence, 0.98);
        assert_eq!(floor.level, Level::At(50));
        assert_eq!(floor.hostile, 2);
        assert_eq!(floor.suspicious, 1);
        assert_eq!(floor.severe(), 3);
        // The hostile arm's diversity test spans both severe tiers.
        assert_eq!(floor.families, 3);
    }

    #[test]
    fn the_reported_floor_counts_the_suspicious_arms_own_families() {
        let floor = trait_floor(&spread(4, 0.9, 3), Classification::Benign, Some(50), 100)
            .expect("three confident crit-4 families must reach the suspicious arm");

        assert_eq!(floor.arm, FloorArm::Crit4);
        assert_eq!(floor.class, Classification::Suspicious);
        assert_eq!(floor.hostile, 0);
        assert_eq!(floor.suspicious, 3);
        assert_eq!(floor.families, 3);
    }

    #[test]
    fn nothing_to_add_is_reported_as_nothing_not_as_benign() {
        // A caller reading the floor separately from the model must be able to
        // tell "the floor was silent" from "the floor said benign" — it never
        // says benign, and a silent floor must not be read as a clean bill.
        assert!(trait_floor(&spread(4, 0.9, 2), Classification::Benign, Some(50), 100).is_none());
        assert!(trait_floor(&[], Classification::Benign, Some(50), 100).is_none());
    }

    #[test]
    fn the_floor_declines_a_verdict_the_model_already_reached() {
        for already in [Classification::Suspicious, Classification::Hostile] {
            assert!(
                trait_floor(&spread(4, 0.9, 5), already, Some(50), 100).is_none(),
                "the floor is a backstop for model misses, not a second opinion"
            );
        }
    }

    #[test]
    fn applying_the_floor_agrees_with_reporting_it() {
        // `apply_trait_floor` is the in-place form of `trait_floor`, and scan's
        // own paths take the first while consumers take the second. A drift
        // between them would give two callers two different verdicts on one
        // artifact, so it is pinned across every shape the arms distinguish.
        let mut shapes = vec![
            Vec::new(),
            spread(4, 0.9, 2),
            spread(4, 0.9, 3),
            spread(4, 0.9, 5),
            spread(5, 0.98, 1),
            spread(5, 0.98, 2),
            spread(5, 0.5, 3),
            clustered(5, 0.98, 3),
            clustered(4, 0.93, 4),
        ];
        let mut corroborated = spread(5, 0.98, 2);
        corroborated.extend([finding("execution", 4, 0.94)]);
        shapes.push(corroborated);

        for findings in shapes {
            let mut applied = benign();
            apply_trait_floor(&mut applied, &findings, Some(50), 100, "test");
            let reported = trait_floor(&findings, Classification::Benign, Some(50), 100);

            match reported {
                Some(floor) => {
                    assert_eq!(applied.class, floor.class);
                    assert_eq!(applied.probability, floor.confidence);
                    assert_eq!(applied.level, floor.level);
                }
                None => {
                    let untouched = benign();
                    assert_eq!(
                        applied.class, untouched.class,
                        "a silent floor changes nothing"
                    );
                    assert_eq!(applied.probability, untouched.probability);
                    assert_eq!(applied.level, untouched.level);
                }
            }
        }
    }

    #[test]
    fn floors_on_the_root_files_own_findings() {
        // `root_findings` is what the classify path passes: a report's findings
        // live on files[0], never at report level.
        let mut d = benign();
        let report: cleave::types::CompactReport = serde_json::from_value(serde_json::json!({
            "files": [{"id": 0, "path": "x", "type": "elf", "sha": "s", "size": 1,
                       "traits": [
                           {"id": "objectives/command-and-control/backdoor/a::t1", "crit": 5, "conf": 0.98},
                           {"id": "objectives/persistence/service/b::t2", "crit": 5, "conf": 0.9},
                           {"id": "objectives/discovery/env-vars/c::t3", "crit": 4, "conf": 0.8}
                       ]}]
        }))
        .unwrap();
        apply_trait_floor(&mut d, root_findings(&report), Some(50), 100, "test");
        assert_eq!(d.class, Classification::Hostile);
        assert_eq!(d.probability, 0.98);
        assert_eq!(d.level, Level::At(50));
    }

    #[test]
    fn level_confidence_maps_known_grid_and_override_markers() {
        let cases = [
            (Level::Manual, None),
            (Level::Clean, Some(0)),
            (Level::At(0), Some(100)),
            (Level::At(1), Some(99)),
            (Level::At(2), Some(98)),
            (Level::At(5), Some(95)),
            (Level::At(50), Some(90)),
            (Level::At(25000), Some(29)),
            (Level::At(25001), Some(28)),
            (Level::At(25002), Some(27)),
            (Level::At(50000), Some(17)),
            (Level::At(50001), Some(16)),
            (Level::At(50002), Some(15)),
        ];
        for (level, want) in cases {
            assert_eq!(level_confidence(level), want, "level {level:?}");
        }
    }

    #[test]
    fn interpreted_level_places_a_verdict_inside_its_band() {
        use crate::model::{capped_suspicious_level, verdict_for_level};
        use Classification::{Benign, Hostile, Suspicious};

        let grid_max = 30_000;
        let ceiling = capped_suspicious_level(grid_max);
        let at = |level: Level| match level {
            Level::At(n) => n,
            other => panic!("expected a placed level, got {other:?}"),
        };
        for deploy in [4_u16, 5, 25, 50] {
            // Escalation to hostile lands on the active deploy level: the loosest
            // rung still inside the hostile budget.
            assert_eq!(
                interpreted_level(Some(deploy), grid_max, Hostile, true),
                Level::At(deploy)
            );

            // Suspicious lands *within* the band, not on its weakest rung, so two
            // interpreted verdicts of differing strength no longer collapse onto
            // one number.
            let alone = at(interpreted_level(Some(deploy), grid_max, Suspicious, false));
            let corroborated = at(interpreted_level(Some(deploy), grid_max, Suspicious, true));
            assert!(
                deploy < corroborated && corroborated < alone && alone < ceiling,
                "deploy {deploy}: expected deploy < {corroborated} < {alone} < {ceiling}",
            );

            // The round trip is the load-bearing part: whatever level we synthesize,
            // the model's own classifier must read it back as the class we lifted
            // the sample to, or a `lvl`-only consumer (hopper) sees a different
            // verdict than we published.
            for lvl in [deploy, alone, corroborated] {
                assert_ne!(
                    verdict_for_level(lvl, deploy, grid_max),
                    Benign,
                    "deploy {deploy}: level {lvl} read back as benign",
                );
            }
            assert_eq!(verdict_for_level(alone, deploy, grid_max), Suspicious);
            assert_eq!(
                verdict_for_level(corroborated, deploy, grid_max),
                Suspicious
            );
        }
        // A grid tighter than the ceiling keeps the placement inside it.
        let tight = at(interpreted_level(Some(25), 2_000, Suspicious, false));
        assert!(tight <= capped_suspicious_level(2_000) && tight > 25);
        // A deploy level at or above the ceiling leaves no band to place within.
        assert_eq!(
            interpreted_level(Some(3_000), grid_max, Suspicious, false),
            Level::At(ceiling)
        );
        // Benign is the clean marker regardless of grid (even in manual mode).
        assert_eq!(
            interpreted_level(Some(25), grid_max, Benign, false),
            Level::Clean
        );
        assert_eq!(interpreted_level(None, 0, Benign, false), Level::Clean);
        // Manual-threshold mode (no grid): no synthetic hostile/suspicious level.
        assert_eq!(interpreted_level(None, 0, Hostile, true), Level::Manual);
        assert_eq!(interpreted_level(None, 0, Suspicious, false), Level::Manual);
    }

    fn floor(class: Classification, confidence: f32) -> FloorDecision {
        FloorDecision {
            class,
            confidence,
            level: Level::At(50),
            arm: FloorArm::Crit5,
            hostile: 2,
            suspicious: 1,
            families: 3,
        }
    }

    fn member(
        id: u64,
        class: Classification,
        prob: f32,
        fired: Option<FloorDecision>,
    ) -> EmbeddedFile {
        EmbeddedFile {
            id,
            sha256: String::new(),
            path: format!("member-{id}"),
            file_type: "js".to_string(),
            classification: class,
            probability: prob,
            threshold: 0.5,
            level: Level::At(25),
            model_scores: Vec::new(),
            skipped_models: Vec::new(),
            formula: String::new(),
            top_findings: Vec::new(),
            floor: fired,
        }
    }

    fn evals(members: Vec<EmbeddedFile>) -> MemberEvals {
        members.into_iter().map(|m| (m.id, m)).collect()
    }

    #[test]
    fn a_floored_member_contributes_nothing_to_the_models_own_reading() {
        // The member's stored verdict is hostile, but the floor put it there —
        // so the model's reading of that member was benign, and a caller
        // reporting the model separately must not be handed the floor's work
        // as if the model had done it.
        let table = evals(vec![member(
            1,
            Classification::Hostile,
            0.98,
            Some(floor(Classification::Hostile, 0.98)),
        )]);
        assert_eq!(
            worst_member(&table).map(|d| d.class),
            Some(Classification::Hostile)
        );
        assert!(
            worst_member_model(&table).is_none(),
            "every member was floored, so the model convicted nobody"
        );
    }

    #[test]
    fn an_unfloored_member_is_the_models_own_reading() {
        let table = evals(vec![member(1, Classification::Hostile, 0.98, None)]);
        assert_eq!(
            worst_member_model(&table).map(|d| d.class),
            Some(Classification::Hostile)
        );
    }

    #[test]
    fn the_model_is_read_across_members_the_floor_left_alone() {
        let table = evals(vec![
            member(
                1,
                Classification::Hostile,
                0.99,
                Some(floor(Classification::Hostile, 0.99)),
            ),
            member(2, Classification::Suspicious, 0.70, None),
            member(3, Classification::Benign, 0.01, None),
        ]);
        // The verdict comes from the floored member; the model's own worst
        // reading is the suspicious one it reached unaided.
        assert_eq!(
            worst_member(&table).map(|d| d.class),
            Some(Classification::Hostile)
        );
        assert_eq!(
            worst_member_model(&table).map(|d| d.class),
            Some(Classification::Suspicious)
        );
    }

    #[test]
    fn the_gravest_firing_wins_and_corroboration_breaks_a_tie() {
        let table = evals(vec![
            member(
                1,
                Classification::Suspicious,
                0.80,
                Some(floor(Classification::Suspicious, 0.80)),
            ),
            member(
                2,
                Classification::Hostile,
                0.90,
                Some(floor(Classification::Hostile, 0.90)),
            ),
            member(
                3,
                Classification::Hostile,
                0.95,
                Some(floor(Classification::Hostile, 0.95)),
            ),
        ]);
        let worst = worst_member_floor(&table).expect("three members fired");
        assert_eq!(worst.class, Classification::Hostile);
        assert_eq!(worst.confidence, 0.95);
    }

    #[test]
    fn no_firing_anywhere_reports_nothing_rather_than_benign() {
        // The floor has no way to say benign, so its absence must stay absent.
        let table = evals(vec![member(1, Classification::Benign, 0.01, None)]);
        assert!(worst_member_floor(&table).is_none());
    }

    #[test]
    fn worse_of_is_commutative_over_firings() {
        let cases = [
            floor(Classification::Suspicious, 0.80),
            floor(Classification::Suspicious, 0.95),
            floor(Classification::Hostile, 0.80),
            floor(Classification::Hostile, 0.95),
        ];
        for a in cases {
            for b in cases {
                assert_eq!(
                    a.worse_of(b),
                    b.worse_of(a),
                    "a summary must not depend on member ordering"
                );
            }
        }
    }
}
