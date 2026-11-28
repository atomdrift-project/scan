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

const ENCRYPTED_ARCHIVE_MEMBER: &[&str] = &[
    "metadata/archive/**::archive-encrypted-member",
    "metadata/file/archive/**::archive-encrypted-member",
];
const VSIX_PUBLISHER_MISMATCH: &[&str] =
    &["metadata/package/repository::vsix-publisher-repo-owner-mismatch"];

/// Trait families and ids that admit a sample to the LLM on their own, at
/// notable criticality or above.
///
/// These exist because the criticality and probability tests are blind in the
/// same place: a package whose only signal is one notable finding. Each entry
/// earned its place by leave-one-out measurement on a labeled corpus: it must
/// still admit malicious samples that nothing else, the other entries
/// included, admits, at little benign cost.
///
/// Each group describes one signal and lists equivalent selectors from
/// different taxonomy layouts. One matching selector is enough for admission
/// and for [`validate_gate_prefixes`]. A family selects its whole subtree,
/// `family::leaf` selects a leaf in one directory, and `family/**::leaf`
/// searches for the leaf throughout that subtree, including the root. The leaf
/// can use `*` as a wildcard. Keep selectors narrow and measure new signals:
/// a broad family like URL construction would admit many benign packages.
pub(super) const LLM_GATE_PREFIXES: &[&[&str]] = &[
    ENCRYPTED_ARCHIVE_MEMBER,
    // Agent/editor config paths — where prompt-injection payloads in agent
    // skills and editor extensions live.
    &["micro-behaviors/fs/path/config/app"],
    &["micro-behaviors/communications/email/send"],
    // Most curl upload traits moved into upload subfamilies. The at-source
    // body trait kept its leaf but moved to request/body.
    &[
        "micro-behaviors/communications/http/upload/**::*curl*",
        "micro-behaviors/communications/http/request/body/**::curl-at-source-body-command",
    ],
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
    VSIX_PUBLISHER_MISMATCH,
    &[
        "micro-behaviors/communications/url/scheme/**::http-url-shaped-text",
        "micro-behaviors/communications/http/url/scheme/**::http-url-shaped-text",
        "micro-behaviors/communications/http/url/scheme/**::cleartext-http-url",
    ],
    &[
        "micro-behaviors/communications/url/**::query-serializer-call",
        "micro-behaviors/communications/http/url/**::query-serializer-call",
    ],
    &["micro-behaviors/communications/http/cookie-store::document-cookie-split-iterate"],
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

/// Gate prefixes with, as near as a labeled corpus can show, no benign carrier
/// at all: a notable finding under one admits a sample to the LLM on its own
/// (`strong-trait`), however large the sample is. Among them: a package that
/// decodes a payload to a file from a shell command, and an agent skill that
/// points the agent at an off-vendor API base URL.
///
/// Prefer families and narrow globs where their whole match set is useful.
/// The encrypted-member trait moved into a broad family, so its exact leaf keeps
/// unrelated archive findings out of this list. Off-vendor URL assignments
/// are atomic traits under a shared family; the narrow glob covers their
/// provider-specific ids without admitting other URL configuration findings.
/// [`validate_gate_prefixes`] checks these entries against installed traits.
pub(super) const LLM_GATE_STRONG_PREFIXES: &[&[&str]] = &[
    // Password-protected payload droppers.
    ENCRYPTED_ARCHIVE_MEMBER,
    // Publisher is not the owner of the repository it points at.
    VSIX_PUBLISHER_MISMATCH,
    // Agent instruction files (CLAUDE.md, AGENTS.md, rules).
    &["micro-behaviors/fs/path/agent-instructions"],
    // Editor/agent *project* settings and tasks — the per-repository files an
    // extension or skill drops to run on open.
    &["micro-behaviors/fs/path/config/app::*project-*"],
    // An off-vendor LLM API base URL assignment can redirect agent traffic;
    // the glob covers every provider's setting of that shape.
    &["micro-behaviors/os/env/provider::*base-url-off-vendor"],
    // Decoding a payload to a file from a shell command.
    &["micro-behaviors/data/decode/command::*to-file*"],
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
/// Matching descendants requires a path boundary. Synthesized IDs have no
/// leaf, so only a family selector can admit them.
pub(super) fn prefix_admits<'a>(prefixes: impl IntoIterator<Item = &'a str>, id: &str) -> bool {
    let head = crate::features::trait_path(id);
    let leaf = id.split("::").nth(1);
    prefixes.into_iter().any(|prefix| {
        let (family, pattern, recursive) = prefix_parts(prefix);
        let in_family = head == family
            || (recursive
                && head
                    .strip_prefix(family)
                    .is_some_and(|rest| rest.starts_with('/')));
        in_family && pattern.is_none_or(|pattern| leaf.is_some_and(|id| leaf_matches(pattern, id)))
    })
}

fn prefix_parts(prefix: &str) -> (&str, Option<&str>, bool) {
    let Some((family, leaf)) = prefix.split_once("::") else {
        return (prefix, None, true);
    };
    match family.strip_suffix("/**") {
        Some(root) => (root, Some(leaf), true),
        None => (family, Some(leaf), false),
    }
}

/// Check each traits-backed signal against the installed Cleave tree. Report
/// the signal once when none of its selectors matches a declaration.
#[must_use]
pub fn validate_gate_prefixes(traits_dir: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    let mut checked = HashSet::new();
    for &selectors in LLM_GATE_PREFIXES.iter().chain(LLM_GATE_STRONG_PREFIXES) {
        if !checked.insert(selectors) {
            continue;
        }
        // Collecting into Option stops at the first selector that resolves (None).
        let faults: Option<Vec<_>> = selectors
            .iter()
            .map(|prefix| validate_prefix(traits_dir, prefix))
            .collect();
        if let Some(faults) = faults {
            problems.push(faults.join(" or "));
        }
    }
    problems
}

fn validate_prefix(traits_dir: &Path, prefix: &str) -> Option<String> {
    let (family, leaf, recursive) = prefix_parts(prefix);
    let dir = traits_dir.join(family);
    if !dir.is_dir() {
        return Some(format!("`{prefix}` names no directory ({})", dir.display()));
    }
    if trait_declared(&dir, leaf.unwrap_or("*"), recursive) {
        return None;
    }
    Some(format!("`{prefix}` matches no trait in {}", dir.display()))
}

#[derive(serde::Deserialize)]
struct TraitDeclarations {
    #[serde(default)]
    traits: Vec<TraitDeclaration>,
    #[serde(default)]
    composite_rules: Vec<TraitDeclaration>,
}

#[derive(serde::Deserialize)]
struct TraitDeclaration {
    id: String,
}

fn trait_declared(dir: &Path, pattern: &str, recursive: bool) -> bool {
    let mut files = walkdir::WalkDir::new(dir).min_depth(1);
    if !recursive {
        files = files.max_depth(1);
    }
    let options = serde_saphyr::options! { strict_booleans: true };
    for file in files.into_iter().flatten() {
        let path = file.path();
        if path
            .extension()
            .is_none_or(|ext| ext != "yaml" && ext != "yml")
        {
            continue;
        }
        let Ok(contents) = std::fs::read_to_string(path) else {
            continue;
        };
        let Ok(declarations) =
            serde_saphyr::from_str_with_options::<TraitDeclarations>(&contents, options.clone())
        else {
            continue;
        };
        if declarations
            .traits
            .iter()
            .chain(&declarations.composite_rules)
            .any(|rule| leaf_matches(pattern, &rule.id))
        {
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
    /// cleave surfaced a suspicious-or-above finding. Not an admission (see
    /// [`admit`]); it decides where an L0 verdict the LLM clears lands.
    pub suspicious: bool,
    /// cleave surfaced a notable-or-above finding under a gate prefix.
    ///
    /// Separate from `hostile` because it answers a different question: not
    /// "how bad did this look" but "is this one of the specific behaviours a
    /// reader catches and criticality does not". It is the admission that
    /// carries the low-probability tail — packages whose only signal is a
    /// single notable finding — and it is why the probability floor
    /// ([`DEFAULT_LLM_MIN_PROB`]) can sit as high as it does.
    pub gate_trait: bool,
    /// cleave surfaced a notable-or-above finding under a strong gate prefix.
    pub strong_trait: bool,
    /// Notable-or-above findings across the whole report — every member file,
    /// duplicates included. Logged with each gate decision.
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
            if finding.crit >= cleave::Criticality::Suspicious {
                out.suspicious = true;
            }
            // Notable is the floor: below it sit baseline noise and the
            // component legs of composites, which fire on ordinary packages and
            // would turn every prefix into a volume tax.
            if finding.crit >= cleave::Criticality::Notable {
                out.notable += 1;
                let gate = LLM_GATE_PREFIXES
                    .iter()
                    .flat_map(|selectors| selectors.iter().copied())
                    .chain(LLM_GATE_SYNTHESIZED_PREFIXES.iter().copied());
                if prefix_admits(gate, &finding.id) {
                    out.gate_trait = true;
                }
                let strong = LLM_GATE_STRONG_PREFIXES
                    .iter()
                    .flat_map(|selectors| selectors.iter().copied());
                if prefix_admits(strong, &finding.id) {
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
/// raw score above the probability floor. `Optional` is the routine
/// case — ML placed the file on its grid but nothing else says so — where the
/// second opinion is context for the reader rather than a correction. Under
/// load such a caller serves `Required` first and skips `Optional`; unloaded,
/// both run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
///
/// Serializable, so a caller that cannot run the call itself can hand it to
/// one that can ([`LlmInput`](crate::engine::LlmInput)).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Admitted {
    /// Whether the endpoint may skip this sample under load.
    pub tier: LlmAdmission,
    pub(super) ml_class: Classification,
    pub(super) ml_prob: f32,
    pub(super) levels: LevelContext,
    /// cleave's report carries a hostile finding.
    pub(super) hostile_finding: bool,
    /// cleave's report carries a suspicious-or-above finding.
    pub(super) suspicious_finding: bool,
}

impl Admitted {
    /// Where ML placed the sample, and the deploy level and grid it was
    /// placed on: the model facts a grade is blended against.
    #[must_use]
    pub const fn levels(&self) -> LevelContext {
        self.levels
    }
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
/// sample [`LlmAdmission::Required`], however large it is; a level placement
/// alone makes it [`LlmAdmission::Optional`].
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
    // No size veto: a large sample that admitted itself is read like any
    // other. The veto (an ML-benign sample with no hostile finding and over
    // 300 notable findings was not sent) was removed 2026-11-27; it rejected
    // exactly the big packages a compromised release hides in.
    let decision = if !reasons.is_empty() {
        Some(LlmAdmission::Required)
    } else if placed {
        Some(LlmAdmission::Optional)
    } else {
        None
    };
    let why = match decision {
        Some(LlmAdmission::Required) => "admitted",
        Some(LlmAdmission::Optional) => "level placement only",
        None => "no admission fired",
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
        "LLM gate"
    );
    decision.map(|tier| Admitted {
        tier,
        ml_class,
        ml_prob,
        levels,
        hostile_finding: hostile,
        suspicious_finding: findings.suspicious,
    })
}
