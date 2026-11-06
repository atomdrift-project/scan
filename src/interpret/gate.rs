//! Which samples get a second opinion: the trait prefixes that admit one, and
//! the admission decision itself.

use std::collections::HashSet;
use std::path::Path;

use super::{InterpretConfig, LevelContext};
use crate::model::Classification;

/// Default raw-probability floor for sending a sample to the LLM.
///
/// One of several independent admissions (see [`admit`]); an **OR**, so it
/// can only ever send more, never block. The floor sits at the elbow of the
/// admission curve on a labeled corpus: below it malicious admission is flat
/// while benign admission keeps climbing, so every step lower buys volume and
/// no coverage; well above it, each benign sample saved costs a malicious one.
///
/// A floor this high is affordable only because this test does not carry the
/// gate alone: the gate prefixes cover the low-probability tail at a far
/// better exchange rate. A lone floor low enough to match the whole gate's
/// coverage would send more than twice the benign volume and still reach
/// fewer malicious samples. `docs/interpret-tuning.md` describes how to
/// re-measure it.
pub const DEFAULT_LLM_MIN_PROB: f32 = 0.55;

/// Trait families and ids that admit a sample to the LLM on their own, at
/// notable criticality or above.
///
/// These exist because the criticality and probability tests are blind in the
/// same place: a package whose only signal is one notable finding. Each entry
/// earned its place by leave-one-out measurement on a labeled corpus: it must
/// still admit malicious samples that nothing else, the other entries
/// included, admits, at little benign cost.
///
/// An entry is either a family (matching every id at or below it) or a
/// `family::leaf` id, where the leaf may use `*` to match a run of characters
/// (`family::curl-*`). Prefer a family or a glob to an exact leaf: leaf names
/// are renamed in passing, and an exact one that stops matching costs its
/// coverage silently. Keep entries narrow, though: wide families fail the
/// measurement. `micro-behaviors/communications/http/url` as a family admits
/// most benign packages, because nearly every package builds a URL, so only
/// its `scheme` and `query` leaves are listed. Measure a candidate rather than
/// adding one that reads plausible.
///
/// Every id here is declared in the cleave traits tree, which is what lets
/// [`validate_gate_prefixes`] prove the taxonomy has not moved underneath them.
pub(super) const LLM_GATE_PREFIXES: &[&str] = &[
    // Encrypted archive members: almost never benign. Named by leaf so the
    // rest of the archive metadata family does not enter the gate.
    "metadata/file/archive::archive-encrypted-member",
    // Agent/editor config paths — where prompt-injection payloads in agent
    // skills and editor extensions live.
    "micro-behaviors/fs/path/config/app",
    "micro-behaviors/communications/email/send",
    // Any curl upload leaf, not one named one. The family's curl traits all
    // describe the same act at notable-or-above — a curl invocation moving a
    // local file or a raw body to a remote host — so they are equally worth a
    // reader's attention, and a glob keeps admitting them through the leaf
    // renames that an exact id stops matching silently, and that an installed
    // traits revision older or newer than this list may not share.
    "micro-behaviors/communications/http/upload::*curl*",
    // A marketplace/registry publisher that is not the owner of the repository
    // the package points at — the shape of a squatted or impersonated
    // extension, which is otherwise invisible: the code is usually lifted
    // wholesale from the project it impersonates, so nothing in it reads as
    // hostile and no other rule fires.
    //
    // An exact leaf, deliberately. The sibling `npm-name-repo-mismatch` in this
    // family is the opposite trade — monorepos, forks and scoped renames all
    // disagree with their repository — so a `*mismatch*` glob would admit
    // mostly benign packages. Publisher-is-not-repo-owner is the specific
    // claim worth reading.
    "metadata/package/repository::vsix-publisher-repo-owner-mismatch",
    "micro-behaviors/communications/http/url/scheme::cleartext-http-url",
    "micro-behaviors/communications/http/url/query::query-serializer-call",
    "micro-behaviors/communications/http/cookie-store::document-cookie-split-iterate",
];

/// Gate prefixes whose ids cleave synthesizes at analysis time rather than
/// reading from the traits tree, so there is no directory to check them against.
///
/// Kept apart from [`LLM_GATE_PREFIXES`] for exactly that reason: the split is
/// what tells [`validate_gate_prefixes`] which entries it can prove and which it
/// must leave alone, and a tag on a single list would have said the same thing
/// less plainly. Both lists admit identically: [`admit`] does not
/// distinguish them.
///
/// `metadata/encoded-payload/*` is built by cleave's encoded-payload pass. Only
/// its `url` arm is listed: the rest of the family admitted benign samples and
/// no malicious one the gate did not already reach.
pub(super) const LLM_GATE_SYNTHESIZED_PREFIXES: &[&str] = &["metadata/encoded-payload/url"];

/// Notable-or-above findings above which an ML-benign sample with no hostile
/// finding is *not* sent to the LLM, whatever else admitted it. `0` disables
/// the veto.
///
/// The admissions above all fire on volume: a large legitimate package
/// accumulates notable findings until it crosses the probability floor or
/// carries one of the gate prefixes — and the LLM then reads a long prompt to
/// move nothing. Scored by whether each call moved
/// the outcome in the right direction, the calls that correct a verdict sit on
/// small samples and the wasted benign calls on large ones. This cap keeps
/// nearly every correction and drops most of the wasted calls; a lower one
/// starts losing corrections, a higher one keeps benign calls that correct
/// nothing.
///
/// A count of findings rather than prompt tokens: the token proxy lost more
/// corrections at the same benign volume. Level-placed samples are never
/// vetoed — an interpretation is wanted for everything on the grid.
pub const DEFAULT_LLM_BENIGN_NOTABLE_CAP: usize = 300;

/// Gate prefixes that override the size veto ([`DEFAULT_LLM_BENIGN_NOTABLE_CAP`]).
///
/// The ordinary gate prefixes were chosen for coverage and several of them
/// (`*curl*` uploads, encoded-payload URLs) sit on exactly the big benign
/// packages the veto exists to skip, so the veto has to outrank them. These
/// are the prefixes with, as near as a labeled corpus can show, no benign
/// carrier at all, so a sample carrying one is worth a read however large it
/// is. Some are here for corrections the bare veto lost: a large package that
/// decodes a payload to a file from a shell command, and an agent skill that
/// points the agent at an off-vendor API base URL.
///
/// Prefer families and narrow globs where their whole match set is useful.
/// The encrypted-member trait moved into a broad family, so its exact id keeps
/// unrelated archive findings out of this override. Off-vendor URL assignments
/// are atomic traits under a shared family; the narrow glob covers their
/// provider-specific ids without admitting other URL configuration findings.
/// [`validate_gate_prefixes`] checks these entries against installed traits.
pub(super) const LLM_GATE_STRONG_PREFIXES: &[&str] = &[
    // Password-protected payload droppers.
    "metadata/file/archive::archive-encrypted-member",
    // Publisher is not the owner of the repository it points at.
    "metadata/package/repository::vsix-publisher-repo-owner-mismatch",
    // Agent instruction files (CLAUDE.md, AGENTS.md, rules).
    "micro-behaviors/fs/path/agent-instructions",
    // Editor/agent *project* settings and tasks — the per-repository files an
    // extension or skill drops to run on open.
    "micro-behaviors/fs/path/config/app::*project-*",
    // An off-vendor LLM API base URL assignment can redirect agent traffic;
    // the glob covers every provider's setting of that shape.
    "micro-behaviors/os/env/provider::*base-url-off-vendor",
    // Decoding a payload to a file from a shell command.
    "micro-behaviors/data/decode/command::*to-file*",
];

/// Whether `pattern` matches trait leaf `id`, where `*` stands for any run of
/// characters. A pattern without `*` is an exact name, as before.
///
/// Leaf ids are the most volatile part of the taxonomy — a family is a
/// directory somebody has to move deliberately, while a leaf gets renamed in
/// passing (`curl-upload` → `curl-upload-file`) — so a gate entry that has to
/// name one leaf can name a shape instead: `curl-*` keeps admitting the trait
/// through a rename that an exact id would silently stop matching.
#[must_use]
pub(super) fn leaf_matches(pattern: &str, id: &str) -> bool {
    let Some((first, rest)) = pattern.split_once('*') else {
        return pattern == id;
    };
    let Some(mut cursor) = id.strip_prefix(first) else {
        return false;
    };
    let mut parts = rest.split('*').peekable();
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            // The trailing literal must land at the very end, unless the
            // pattern ended with `*` (in which case `part` is empty).
            return cursor.ends_with(part);
        }
        if part.is_empty() {
            continue;
        }
        // `find` returns a char boundary and `part.len()` is the matched
        // slice's own byte length, so the tail always exists; `get` says that
        // without asking the reader to take it on trust.
        match cursor
            .find(part)
            .and_then(|at| cursor.get(at + part.len()..))
        {
            Some(tail) => cursor = tail,
            None => return false,
        }
    }
    true
}

/// Whether any of `prefixes` admits trait `id`.
///
/// A prefix carrying `::` names one trait (its leaf possibly a glob, see
/// [`leaf_matches`]) and matches only within its own family. A prefix without
/// it names a family and matches every id at or below it, compared against the
/// id's head so that `.../archive/member` admits
/// `.../archive/member::archive-encrypted-member` without also admitting a
/// sibling family whose name merely starts with the same letters
/// (`.../archive/members-of-interest`). A synthesized id has no `::` at all, so
/// it is its own head and the same comparison places it.
pub(super) fn prefix_admits<'a>(prefixes: impl IntoIterator<Item = &'a str>, id: &str) -> bool {
    let head = crate::features::trait_path(id);
    prefixes.into_iter().any(|prefix| {
        if let Some((family, leaf)) = prefix.split_once("::") {
            head == family && id.split("::").nth(1).is_some_and(|l| leaf_matches(leaf, l))
        } else {
            head == prefix
                || head
                    .strip_prefix(prefix)
                    .is_some_and(|rest| rest.starts_with('/'))
        }
    })
}

/// Check every gate prefix entry against an installed cleave traits
/// tree, returning one message per broken prefix. An empty vector means the gate
/// still matches what the matcher will emit.
///
/// This exists because a stale prefix fails silently and expensively. Trait ids
/// are `<directory path>::<leaf id>`, so a taxonomy move — a family renamed, a
/// leaf re-homed under a sibling — leaves the gate syntactically fine and
/// matching nothing. Nobody sees an error; the samples that prefix was carrying
/// simply stop reaching the LLM, and the loss only shows up as a coverage
/// regression on a corpus nobody reruns. Cheap to check, so check it.
///
/// A family prefix must still be a directory. An exact `family::leaf` id must
/// additionally be declared by some YAML directly in that directory.
/// Synthesized prefixes have no directory to point at and are not checked
/// here.
#[must_use]
pub fn validate_gate_prefixes(traits_dir: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    let mut checked = HashSet::new();
    for prefix in LLM_GATE_PREFIXES.iter().chain(LLM_GATE_STRONG_PREFIXES) {
        if !checked.insert(prefix) {
            continue;
        }
        let (family, leaf) = match prefix.split_once("::") {
            Some((family, leaf)) => (family, Some(leaf)),
            None => (*prefix, None),
        };
        let dir = traits_dir.join(family);
        let fault = if !dir.is_dir() {
            format!("names no directory ({})", dir.display())
        } else if let Some(leaf) = leaf
            && !trait_declared(&dir, leaf)
        {
            format!("matches no trait in {}", dir.display())
        } else {
            continue;
        };
        problems.push(format!("`{prefix}` {fault}"));
    }
    problems
}

/// Whether any YAML directly in `dir` declares a trait with this `id`.
///
/// Line-oriented on purpose: a trait declaration is `- id: <leaf>` at any
/// indentation, and recognizing that costs nothing next to loading a parser to
/// answer one yes/no question. False negatives are the safe direction — they
/// fail the check loudly rather than passing a broken prefix.
fn trait_declared(dir: &Path, id: &str) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "yaml") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let declared = text.lines().any(|line| {
            let rest = line.trim_start().trim_start_matches("- ").trim_start();
            rest.strip_prefix("id:")
                .is_some_and(|value| leaf_matches(id, value.trim().trim_matches(['"', '\''])))
        });
        if declared {
            return true;
        }
    }
    false
}

/// What cleave's own findings say about a sample, read from the structured
/// report.
///
/// These two questions are not answered by scanning the *render* for `H`/`S`
/// annotation letters. The render is a prompt, not a database: it is truncated
/// to the highest-scoring members, deduplicated across the archive, and it
/// drops composites evaluated at container scope entirely, listing only the
/// atomic legs they were built from. So a package whose single suspicious
/// finding is a cross-file composite renders with no `S` at all and would be
/// silently withdrawn from the gate. Reading the letters also couples the gate
/// to the annotation format, which an upstream change can alter at any time.
///
/// The render is still the right input for `render_mostly_readable` and
/// `addresses_the_analyzer`, which ask about the bytes the model will see.
#[derive(Debug, Clone, Copy, Default)]
pub struct FindingSeverity {
    /// cleave surfaced a hostile-criticality finding.
    pub hostile: bool,
    /// cleave surfaced a notable-or-above finding under a gate prefix.
    ///
    /// Separate from `hostile` because it answers a different question: not
    /// "how bad did this look" but "is this one of the specific behaviours a
    /// reader catches and criticality does not". It is the admission that
    /// carries the low-probability tail — packages whose only signal is a
    /// single notable finding — and it is why the probability floor
    /// ([`DEFAULT_LLM_MIN_PROB`]) can sit as high as it does.
    pub gate_trait: bool,
    /// cleave surfaced a notable-or-above finding under a strong gate prefix:
    /// admitted past the size veto.
    pub strong_trait: bool,
    /// Notable-or-above findings across the whole report — every member file,
    /// duplicates included. The size the veto reads.
    pub notable: usize,
}

impl FindingSeverity {
    /// Read whether cleave reached hostile criticality anywhere in the report,
    /// and whether any finding sits under a gate prefix.
    #[must_use]
    pub fn from_report(report: &cleave::AnalysisReport) -> Self {
        Self::from_findings(report.files.iter().flat_map(|file| file.findings.iter()))
    }

    /// As [`Self::from_report`], over a bare finding sequence. The report is
    /// only ever a container for these, and taking the findings directly is
    /// what lets the gate be exercised against hand-built fixtures.
    ///
    /// Deliberately not an early return on hostile: `gate_trait` has to see
    /// every finding, and a hostile one says nothing about whether a gate
    /// prefix also fired. The loop is over an already-materialized report.
    #[must_use]
    pub fn from_findings<'a>(findings: impl IntoIterator<Item = &'a cleave::Finding>) -> Self {
        let mut out = Self::default();
        for finding in findings {
            if finding.crit >= cleave::Criticality::Hostile {
                out.hostile = true;
            }
            // Notable is the floor: below it sit baseline noise and the
            // component legs of composites, which fire on ordinary packages and
            // would turn every prefix into a volume tax.
            if finding.crit >= cleave::Criticality::Notable {
                out.notable += 1;
                let gate = LLM_GATE_PREFIXES
                    .iter()
                    .chain(LLM_GATE_SYNTHESIZED_PREFIXES);
                if prefix_admits(gate.copied(), &finding.id) {
                    out.gate_trait = true;
                }
                if prefix_admits(LLM_GATE_STRONG_PREFIXES.iter().copied(), &finding.id) {
                    out.strong_trait = true;
                }
            }
        }
        out
    }
}

/// Why a sample is admitted to the LLM, for a caller that runs the second
/// opinion off the analysis path and must ration a saturated endpoint.
///
/// `Required` is every admission that can change a verdict: a hostile cleave
/// finding, a finding under a gate prefix, a non-benign ML class, or a
/// raw score above the probability floor — unless the size veto
/// ([`DEFAULT_LLM_BENIGN_NOTABLE_CAP`]) withholds it. `Optional` is the routine
/// case — ML placed the file on its grid but nothing else says so — where the
/// second opinion is context for the reader rather than a correction. Under
/// load such a caller serves `Required` first and skips `Optional`; unloaded,
/// both run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmAdmission {
    /// Can change the verdict: never skipped.
    Required,
    /// Context for the reader: skipped when the endpoint is saturated.
    Optional,
}

/// A sample the LLM gate admitted, carrying the ML verdict and the cleave
/// evidence it was admitted on. [`interpret_admitted`](super::interpret_admitted)
/// consumes it, so a caller
/// that defers the LLM call does not run the gate a second time.
#[derive(Debug, Clone, Copy)]
pub struct Admitted {
    /// Whether the endpoint may skip this sample under load.
    pub tier: LlmAdmission,
    pub(super) ml_class: Classification,
    pub(super) ml_prob: f32,
    pub(super) levels: LevelContext,
    /// cleave's report carries a hostile finding.
    pub(super) hostile_finding: bool,
}

/// The LLM gate: whether a sample gets a second opinion, answered without
/// running the model. Returns the token
/// [`interpret_admitted`](super::interpret_admitted) takes, so a caller that
/// defers the call does not run the gate twice; `None` when the sample is not
/// admitted at all.
///
/// Five independent admissions, all `OR`, because they fail on different
/// files: a raw score at or above the configured floor
/// ([`DEFAULT_LLM_MIN_PROB`] by default); a hostile cleave finding; a
/// notable-or-above finding under a gate prefix; one under a strong gate
/// prefix; and a non-benign verdict class, which catches a container carrying
/// a member's hostile class on its own near-zero score. Any of them makes the
/// sample [`LlmAdmission::Required`] unless the size veto
/// ([`DEFAULT_LLM_BENIGN_NOTABLE_CAP`]) withholds it; a level placement alone
/// makes it [`LlmAdmission::Optional`].
#[must_use]
pub fn admit(
    cfg: &InterpretConfig,
    ml_class: Classification,
    ml_prob: f32,
    levels: LevelContext,
    findings: FindingSeverity,
    subject: &str,
) -> Option<Admitted> {
    // Gate: interpret when ML fired at or below the cutoff level, OR when cleave
    // surfaced a hostile finding that ML scored below it —
    // that disagreement is exactly where a second opinion pays off, and it is how
    // an ML-blind packed binary (prob ≈ 0 yet flagged by cleave) still reaches the
    // LLM. Truly-clean files (no hostile finding, no calibrated ML signal) skip.
    //
    // The verdict class is a third admission on its own: a container whose
    // hostile call came from a member elevation carries the member's class but
    // its own raw score, which can sit far below the cutoff (class hostile at
    // level 0 with a raw score near zero, and no hostile finding left in the
    // render). Gating that out publishes a hostile verdict with no
    // interpretation and no error trace; anything the scan itself calls
    // non-benign must reach the LLM.
    //
    // `findings` comes from the structured report, never the render; see
    // [`FindingSeverity`] for why.
    //
    // A raw-probability floor is a fourth admission, and it exists because the
    // level grid can be silent on a file the model is not actually comfortable
    // with. `ml_admits` needs a *fired* level, so a sample the calibration never
    // placed (`Level::Clean`) is inadmissible at every cutoff — including a sample
    // scoring 0.19, which is two orders of magnitude above the benign mass and
    // plainly worth a second opinion. That is the exact shape of an editor
    // extension whose payload is a small, unobfuscated recon-and-eval chain: too
    // little mass for the grid, more than enough for a reader.
    //
    // Deliberately a floor on the raw score rather than another level knob: the
    // point is to cover the case where there is no level to reason about.
    // Two independent ML admissions, because they fail on different files. A
    // placed level is the precise one, but it is silent on every file the grid
    // never reached. The probability floor covers those at a known volume
    // cost. See [`DEFAULT_LLM_MIN_PROB`].
    //
    // Every admission is evaluated, not short-circuited, so the log line can
    // say which ones fired — the question a reader asks of a skipped sample is
    // "what would have had to be true", and a partial answer is no answer.
    let hostile = findings.hostile;
    let benign_class = matches!(ml_class, Classification::Benign);
    let mut reasons: Vec<&str> = Vec::new();
    if ml_prob >= cfg.min_prob {
        reasons.push("prob");
    }
    // Hostile, not suspicious: suspicious-only findings admit benign samples
    // without ever moving a verdict.
    if hostile {
        reasons.push("hostile-finding");
    }
    if findings.gate_trait {
        reasons.push("gate-trait");
    }
    if findings.strong_trait {
        reasons.push("strong-trait");
    }
    if !benign_class {
        reasons.push("class");
    }
    let placed = levels.ml_admits(cfg.min_level);
    // The size veto: see `DEFAULT_LLM_BENIGN_NOTABLE_CAP`. Only the Required
    // tier is subject to it; a level placement below is never withheld.
    let vetoed = cfg.benign_notable_cap > 0
        && benign_class
        && !hostile
        && !findings.strong_trait
        && findings.notable > cfg.benign_notable_cap;
    let decision = if !reasons.is_empty() && !vetoed {
        Some(LlmAdmission::Required)
    } else if placed {
        Some(LlmAdmission::Optional)
    } else {
        None
    };
    let why = match (decision, vetoed) {
        (Some(LlmAdmission::Required), _) => "admitted",
        (Some(LlmAdmission::Optional), true) => "vetoed by size, kept for its level placement",
        (Some(LlmAdmission::Optional), false) => "level placement only",
        (None, true) => "vetoed by size",
        (None, false) => "no admission fired",
    };
    tracing::info!(
        subject,
        decision = decision.map_or("skipped", |d| match d {
            LlmAdmission::Required => "required",
            LlmAdmission::Optional => "optional",
        }),
        why,
        admissions = %reasons.join(","),
        ml_class = %ml_class,
        ml_prob,
        ml_level = %levels.fired,
        min_prob = cfg.min_prob,
        hostile_finding = hostile,
        gate_trait = findings.gate_trait,
        strong_trait = findings.strong_trait,
        notable_findings = findings.notable,
        notable_cap = cfg.benign_notable_cap,
        "LLM gate"
    );
    decision.map(|tier| Admitted {
        tier,
        ml_class,
        ml_prob,
        levels,
        hostile_finding: hostile,
    })
}
