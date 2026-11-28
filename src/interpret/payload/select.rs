//! Which findings each file shows, and which of its context windows are
//! drawn.

use std::collections::{BTreeSet, HashMap, HashSet};

use cleave::types::{
    ARCHIVE_DELIMITER, Criticality, FileAnalysis, Finding, Istr, Note, finding_spans,
};

use super::{
    LOW_TIER_FILL, MAX_WINDOWS, TOP_N, WINDOW_BYTES, WINDOWS_PER_FINDING, composite_atoms,
    tiny_should_show,
};

/// Per file id, the evidence the LLM/tiny view must render in that file: each
/// suspicious/hostile atom the file matched, and every atom a
/// suspicious/hostile composite anywhere in the report transitively rests on,
/// placed in the member where it matched.
///
/// The view withholds composites (see `select_ids`), so these atoms are all
/// that is left of a withheld conclusion; losing one to a criticality floor,
/// the one-note-per-row rule, dedup or a window cap silently weakened the
/// evidence a grader weighs. Resolution uses what analysis recorded — no rule
/// is re-evaluated, and a leg absent from the report (never matched, or
/// suppressed) resolves to nothing:
///   - a container's leg that also matched natively in members below it
///     belongs to the deepest such members, narrowed to the composite's
///     recorded `composite_sources` members when any of them carry it;
///   - a native composite leg expands into its own `trait_refs` in place;
///   - an inherited copy (`src` set) resolves in the member it came from.
pub(super) fn evidence_legs(files: &[FileAnalysis]) -> HashMap<u32, BTreeSet<&str>> {
    let by_id: HashMap<u32, &FileAnalysis> = files.iter().map(|f| (f.id, f)).collect();
    // Per id, the files it matched natively in.
    let mut native: HashMap<&str, Vec<&FileAnalysis>> = HashMap::new();
    for file in files {
        for f in file.findings.iter().filter(|f| f.src.is_none()) {
            let homes = native.entry(f.id.as_str()).or_default();
            if homes.last().is_none_or(|h| h.id != file.id) {
                homes.push(file);
            }
        }
    }
    let below = |d: &FileAnalysis, f: &FileAnalysis| {
        d.path
            .strip_prefix(f.path.as_str())
            .and_then(|rest| rest.strip_prefix(ARCHIVE_DELIMITER))
            .is_some_and(|member| !member.is_empty())
    };

    let mut out: HashMap<u32, BTreeSet<&str>> = HashMap::new();
    for file in files {
        for c in &file.findings {
            let owned = c.src.is_none() || file.composite_sources.contains_key(c.id.as_str());
            if c.crit < Criticality::Suspicious || !owned {
                continue;
            }
            let sources: HashSet<u32> = file
                .composite_sources
                .get(c.id.as_str())
                .into_iter()
                .flatten()
                .map(|s| s.file)
                .collect();
            // An elevated atom resolves like a leg: a container's copy of a
            // member's match belongs to that member.
            let mut stack: Vec<(&FileAnalysis, &str)> = if c.trait_refs.is_empty() {
                vec![(file, c.id.as_str())]
            } else {
                c.trait_refs.iter().map(|r| (file, r.as_str())).collect()
            };
            let mut seen: HashSet<(u32, &str)> = HashSet::new();
            while let Some((f, id)) = stack.pop() {
                if !seen.insert((f.id, id)) {
                    continue;
                }
                let homes = native.get(id).map_or(&[][..], Vec::as_slice);
                let inner: Vec<&FileAnalysis> =
                    homes.iter().copied().filter(|d| below(d, f)).collect();
                // Only the deepest: a nested container re-carries its members'.
                let mut deeper: Vec<&FileAnalysis> = inner
                    .iter()
                    .copied()
                    .filter(|d| !inner.iter().any(|e| below(e, d)))
                    .collect();
                if deeper.iter().any(|d| sources.contains(&d.id)) {
                    deeper.retain(|d| sources.contains(&d.id));
                }
                if !deeper.is_empty() {
                    stack.extend(deeper.into_iter().map(|d| (d, id)));
                } else if let Some(g) = f
                    .findings
                    .iter()
                    .find(|g| g.src.is_none() && g.id.as_str() == id)
                {
                    if g.trait_refs.is_empty() {
                        out.entry(f.id).or_default().insert(g.id.as_str());
                    } else {
                        stack.extend(g.trait_refs.iter().map(|r| (f, r.as_str())));
                    }
                } else if let Some(m) = f
                    .findings
                    .iter()
                    .find_map(|g| (g.id.as_str() == id).then_some(g.src).flatten())
                    .and_then(|m| by_id.get(&m))
                {
                    stack.push((m, id));
                }
            }
        }
    }
    out
}

/// Paths of files that are archive containers — i.e. some other file is an
/// archive member of theirs (`<container>!!<member>`). Synthetic embedded-binary
/// extractions (`…!!embedded:…`) are sub-views of a member, not members, so they
/// don't make their carrier a container.
pub(super) fn archive_container_paths<'a>(files: &[&'a FileAnalysis]) -> HashSet<&'a str> {
    let mut paths = HashSet::new();
    for f in files {
        if let Some((container, leaf)) = f.path.rsplit_once(ARCHIVE_DELIMITER)
            && !leaf.starts_with("embedded:")
        {
            paths.insert(container);
        }
    }
    paths
}

/// The deepest nesting level at which each finding id appears as a *native*
/// finding (one located in that file, `src` unset — not inherited from a child).
/// A container drops any finding also native deeper down, since that member is
/// where it was actually located.
pub(super) fn native_finding_depths<'a>(files: &[&'a FileAnalysis]) -> HashMap<&'a str, u32> {
    let mut depths: HashMap<&str, u32> = HashMap::new();
    for f in files {
        for finding in f.findings.iter().filter(|fd| fd.src.is_none()) {
            depths
                .entry(finding.id.as_str())
                .and_modify(|d| *d = (*d).max(f.depth))
                .or_insert(f.depth);
        }
    }
    depths
}

/// Append `block` to `out`, indenting every non-empty line by `depth` levels
/// (two spaces each) so archive members and other nested files nest visually
/// under their container. Blank lines stay blank.
pub(super) fn indent_block(out: &mut String, block: &str, depth: u32) {
    if depth == 0 {
        out.push_str(block);
        return;
    }
    let indent = "  ".repeat(depth as usize);
    for seg in block.split_inclusive('\n') {
        let (line, newline) = match seg.strip_suffix('\n') {
            Some(line) => (line, "\n"),
            None => (seg, ""),
        };
        if !line.is_empty() {
            out.push_str(&indent);
            out.push_str(line);
        }
        out.push_str(newline);
    }
}

/// Which context windows a file draws and which selected findings each one
/// annotates: `None` for a window not drawn, else the note ids it renders.
///
/// Applies [`WINDOWS_PER_FINDING`] in file order (a finding's first windows
/// claim it). When the file's windows then exceed [`MAX_WINDOWS`] or
/// [`WINDOW_BYTES`], [`pick_windows`] chooses which to keep. A selected
/// finding in no drawn window is not rendered: the view shows annotations only
/// beside code.
pub(super) fn window_plan<'a>(
    file: &FileAnalysis,
    notes: &'a [Vec<Note>],
    selected: &[&'a str],
) -> Vec<Planned<'a>> {
    let sel: HashSet<&str> = selected.iter().copied().collect();
    let mut drawn: HashMap<&str, usize> = HashMap::new();
    let mut plan: Vec<Planned<'a>> = Vec::with_capacity(notes.len());
    for window_notes in notes {
        let mut ids: HashSet<&'a str> = HashSet::new();
        for n in window_notes {
            let id = n.id.as_str();
            if !sel.contains(id) || ids.contains(id) {
                continue;
            }
            let count = drawn.entry(id).or_insert(0);
            if *count >= WINDOWS_PER_FINDING {
                continue;
            }
            *count += 1;
            ids.insert(id);
        }
        plan.push(Planned {
            notes: window_notes,
            ids: (!ids.is_empty()).then_some(ids),
        });
    }
    // A text file's leading comment (`context::capture` takes it alongside
    // the finding windows) carries no note, but it is what the file says it
    // is for: it is drawn, and no window budget sheds it.
    if let (Some(first), Some(line)) = (plan.first_mut(), file.context.first())
        && first.ids.is_none()
        && line.loc == 0
        && first.notes.is_empty()
        && line.line.is_some()
    {
        first.ids = Some(HashSet::new());
    }
    let doc_window = plan
        .first()
        .is_some_and(|w| w.ids.as_ref().is_some_and(HashSet::is_empty));
    let windows = plan.iter().filter(|w| w.ids.is_some()).count();
    let bytes: usize = plan
        .iter()
        .zip(&file.context)
        .filter(|(w, _)| w.ids.is_some())
        .map(|(_, line)| line.data.len())
        .sum();
    if windows > MAX_WINDOWS || bytes > WINDOW_BYTES {
        let kept = pick_windows(file, &plan, &sel);
        for (i, (w, keep)) in plan.iter_mut().zip(kept).enumerate() {
            if !keep && !(doc_window && i == 0) {
                w.ids = None;
            }
        }
    }
    plan
}

/// One context window in the render plan: the notes derived for it and, when
/// the plan draws it, the finding ids it annotates.
pub(super) struct Planned<'a> {
    pub(super) notes: &'a [Note],
    pub(super) ids: Option<HashSet<&'a str>>,
}

/// One annotated note's say in which windows a budgeted file draws.
pub(super) struct StoryNote<'a> {
    pub(super) conf: f32,
    /// The trait's family two and three path segments below its namespace
    /// (`objectives/command-and-control/infrastructure[/paste]`).
    pub(super) family2: &'a str,
    pub(super) family3: &'a str,
    /// `1 + 0.25` per withheld composite drawing on the note, up to four.
    pub(super) support: f32,
}

/// A window [`pick_windows`] may draw.
pub(super) struct Candidate<'a> {
    pub(super) line: usize,
    /// The window's effective criticality rank: the highest of its notes' own
    /// criticality and that of any withheld composite drawing on them.
    pub(super) tier: u8,
    pub(super) bytes: usize,
    pub(super) notes: Vec<StoryNote<'a>>,
}

impl Candidate<'_> {
    /// What drawing this window adds to the story told so far: each note's
    /// confidence, discounted for families already drawn and raised for
    /// composite support, over the square root of the window's size so a
    /// compact window showing several things beats a sprawling one.
    pub(super) fn gain(&self, drawn2: &HashMap<&str, f32>, drawn3: &HashMap<&str, f32>) -> f32 {
        let value: f32 = self
            .notes
            .iter()
            .map(|n| {
                let decay = drawn2.get(n.family2).copied().unwrap_or(1.0)
                    * drawn3.get(n.family3).copied().unwrap_or(1.0);
                n.conf * decay * n.support
            })
            .sum();
        value / (self.bytes.max(1) as f32).sqrt()
    }
}

/// Which of a file's planned windows fit its window budget, indexed like
/// `plan`, chosen to tell the most story in the fewest bytes.
///
/// Criticality wins: windows are drawn in descending [`Candidate::tier`], so a
/// suspicious-backed window beats any number of notable ones. Within a tier the
/// next window is the one with the best [`Candidate::gain`], recomputed after
/// every pick: each pick halves its notes' three-segment families and halves
/// their two-segment families again, so the budget spreads across what the
/// program does instead of repeating one behavior. Ties fall to file order.
/// A window that would overspend the byte budget is skipped, and smaller ones
/// may still fit.
///
/// A composite leg ranks by the withheld composite it supports, but its window
/// is still annotated with the leg's own grade and id: the composite's
/// conclusion never reaches the grader. A `metadata/` note that no withheld
/// composite draws on (an import, say) earns no window here — a file's import
/// block would otherwise outbid its behavior — and still lists as a
/// location-less note.
pub(super) fn pick_windows<'a>(
    file: &FileAnalysis,
    plan: &[Planned<'a>],
    selected: &HashSet<&str>,
) -> Vec<bool> {
    let candidates = candidates(file, plan, selected);
    let mut kept = vec![false; plan.len()];
    for line in allot(candidates, MAX_WINDOWS, WINDOW_BYTES) {
        kept[line] = true;
    }
    kept
}

/// The windows of `plan` that carry a note worth a window.
pub(super) fn candidates<'a>(
    file: &FileAnalysis,
    plan: &[Planned<'a>],
    selected: &HashSet<&str>,
) -> Vec<Candidate<'a>> {
    let support = withheld_composite_support(file, selected);
    let mut candidates: Vec<Candidate<'a>> = Vec::new();
    for (i, w) in plan.iter().enumerate() {
        let Some(ids) = &w.ids else { continue };
        let mut seen: HashSet<&str> = HashSet::new();
        let mut tier = 0u8;
        let mut notes = Vec::new();
        for n in w.notes {
            let id = n.id.as_str();
            if !ids.contains(id) || !seen.insert(id) {
                continue;
            }
            let backing = support.get(id);
            if backing.is_none() && id.starts_with("metadata/") {
                continue;
            }
            let (rank, composites) = backing.map_or((0, 0), |b| (b.rank, b.composites));
            tier = tier.max(n.crit.rank()).max(rank);
            notes.push(StoryNote {
                conf: n.conf,
                family2: trait_family(id, 2),
                family3: trait_family(id, 3),
                support: 1.0 + 0.25 * composites.min(4) as f32,
            });
        }
        if !notes.is_empty() {
            candidates.push(Candidate {
                line: i,
                tier,
                bytes: file.context[i].data.len(),
                notes,
            });
        }
    }
    candidates
}

/// Greedily draw the candidates that tell the most story within `budget`
/// bytes and `max_windows` windows; returns the window index of each pick.
pub(super) fn allot(
    mut candidates: Vec<Candidate<'_>>,
    max_windows: usize,
    budget: usize,
) -> Vec<usize> {
    let mut picks: Vec<usize> = Vec::new();
    let mut drawn2: HashMap<&str, f32> = HashMap::new();
    let mut drawn3: HashMap<&str, f32> = HashMap::new();
    let mut spent = 0usize;
    while picks.len() < max_windows {
        let best = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| spent + c.bytes <= budget)
            .map(|(k, c)| (k, c.tier, c.gain(&drawn2, &drawn3)))
            .max_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2)).then(b.0.cmp(&a.0)));
        let Some((k, _, _)) = best else {
            break;
        };
        let pick = candidates.remove(k);
        picks.push(pick.line);
        spent += pick.bytes;
        let families2: HashSet<&str> = pick.notes.iter().map(|n| n.family2).collect();
        let families3: HashSet<&str> = pick.notes.iter().map(|n| n.family3).collect();
        for f in families2 {
            *drawn2.entry(f).or_insert(1.0) *= 0.5;
        }
        for f in families3 {
            *drawn3.entry(f).or_insert(1.0) *= 0.5;
        }
    }
    picks
}

/// A trait id's family: its namespace plus the next `depth` path segments,
/// e.g. `objectives/command-and-control/infrastructure` at depth 2. A shorter
/// path is its own family.
pub(crate) fn trait_family(id: &str, depth: usize) -> &str {
    let path = id.split_once("::").map_or(id, |(path, _)| path);
    path.match_indices('/')
        .nth(depth)
        .and_then(|(i, _)| path.get(..i))
        .unwrap_or(path)
}

/// How the withheld composites in a file back one trait.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CompositeSupport {
    /// Highest criticality rank among them.
    pub(super) rank: u8,
    /// How many distinct composites draw on the trait.
    pub(super) composites: usize,
}

/// Per id, the native composites in `file` that are not in `selected` and
/// reference it — directly, or through nested composites.
///
/// The tiny view withholds composites and shows their legs, but a leg's note
/// carries only the leg's own grade, so evidence behind a hostile conclusion
/// ranked like any notable atom and lost the window budget to a file's import
/// block. [`pick_windows`] ranks the leg by this backing instead, without
/// putting the composite's id or grade in front of the grader.
pub(super) fn withheld_composite_support<'a>(
    file: &'a FileAnalysis,
    selected: &HashSet<&str>,
) -> HashMap<&'a str, CompositeSupport> {
    let composites: HashMap<&str, &Finding> = file
        .findings
        .iter()
        .filter(|f| f.src.is_none() && !f.trait_refs.is_empty())
        .map(|f| (f.id.as_str(), f))
        .collect();
    let mut support: HashMap<&str, CompositeSupport> = HashMap::new();
    for f in composites
        .values()
        .filter(|f| !selected.contains(f.id.as_str()))
    {
        let rank = f.crit.rank();
        let mut seen: HashSet<&str> = HashSet::from([f.id.as_str()]);
        let mut stack: Vec<&str> = f.trait_refs.iter().map(Istr::as_str).collect();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let s = support.entry(id).or_insert(CompositeSupport {
                rank,
                composites: 0,
            });
            s.rank = s.rank.max(rank);
            s.composites += 1;
            if let Some(g) = composites.get(id) {
                stack.extend(g.trait_refs.iter().map(Istr::as_str));
            }
        }
    }
    support
}

/// A file's window notes with every note for a withheld composite reassigned
/// to the composite's own legs, or `None` when nothing changes.
///
/// The LLM/tiny view drops composites and shows their legs (`select_ids`), but
/// placement keeps only the strongest note on an overlapping span, and
/// a hostile composite anchored on its leg's evidence beats that notable leg
/// every time. The leg then owned no note at all: the window with the matched
/// line was drawn only if some other selected trait happened to sit on it, and
/// its annotation was never the leg's. A note whose id is a composite (native
/// or inherited: notes ignore provenance, see [`cleave::context::notes`])
/// becomes one note per leg that is selected here, has no located note of its
/// own, and has a span of its own evidence overlapping the note's, at the
/// composite's span with the leg's identity. A composite's note covers only
/// the one leg it was anchored on, so the overlap test keeps its other legs
/// from being labeled onto a line that is not their evidence; they stay
/// location-less. A composite with no such leg keeps its (unselected,
/// unrendered) note, and a composite that is itself selected (a leg of a
/// wider one) keeps its own.
pub(super) fn relabel_composite_notes(
    file: &FileAnalysis,
    notes: &[Vec<Note>],
    selected: &[&str],
) -> Option<Vec<Vec<Note>>> {
    let sel: HashSet<&str> = selected.iter().copied().collect();
    let located: HashSet<&str> = notes.iter().flatten().map(|n| n.id.as_str()).collect();
    // Per withheld composite, each leg it can hand a note to and that leg's
    // own evidence spans.
    type Legs<'a> = Vec<(&'a Finding, Vec<[u64; 2]>)>;
    let mut legs: HashMap<&str, Legs<'_>> = HashMap::new();
    for f in &file.findings {
        // A composite that is itself selected — a leg of a wider composite,
        // guaranteed by `select_ids` — keeps its own note: it is shown, so
        // its evidence stays with it.
        if f.trait_refs.is_empty() || sel.contains(f.id.as_str()) {
            continue;
        }
        let mine: Legs<'_> = composite_atoms(file, [f])
            .into_iter()
            .filter(|r| sel.contains(r) && !located.contains(r))
            .filter_map(|r| {
                file.findings
                    .iter()
                    .find(|g| g.src.is_none() && g.id.as_str() == r)
            })
            .map(|g| (g, finding_spans(g)))
            .collect();
        if !mine.is_empty() {
            legs.entry(f.id.as_str()).or_default().extend(mine);
        }
    }
    if legs.is_empty() {
        return None;
    }
    let mut changed = false;
    let relabeled = notes
        .iter()
        .map(|window| {
            let mut out: Vec<Note> = Vec::with_capacity(window.len());
            for n in window {
                let end = n.off + u64::from(n.len.max(1));
                let covered: Vec<&Finding> = legs
                    .get(n.id.as_str())
                    .into_iter()
                    .flatten()
                    .filter(|(_, spans)| {
                        spans
                            .iter()
                            .any(|&[off, len]| off < end && n.off < off + len.max(1))
                    })
                    .map(|&(leg, _)| leg)
                    .collect();
                if covered.is_empty() {
                    out.push(n.clone());
                    continue;
                }
                changed = true;
                for leg in covered {
                    if out.iter().any(|m| m.id == leg.id && m.off == n.off) {
                        continue;
                    }
                    out.push(Note {
                        crit: leg.crit,
                        id: leg.id.clone(),
                        desc: leg.desc.clone(),
                        off: n.off,
                        len: n.len,
                        conf: leg.conf,
                    });
                }
            }
            out
        })
        .collect();
    changed.then_some(relabeled)
}

/// The finding ids a file shows: up to [`TOP_N`], ranked by `crit × conf`
/// (highest first) and floored at component, every suspicious or hostile one
/// whatever the cap.
///
/// The view shows evidence, not conclusions. A composite *is* cleave's own
/// verdict, and putting it in front of the grader invites the model to agree
/// with it instead of reasoning from what was observed; the legs carry the
/// same information without the conclusion attached. So composites are
/// dropped and their legs — resolved through nested composites to the atoms
/// beneath — guaranteed in their place.
///
/// [`LOW_TIER_FILL`] then tops the selection up with the best remaining
/// low-tier traits — components and baselines alike, ranked by the same
/// `crit × conf` — until that many show. This is what keeps an undetected
/// file legible: with nothing notable to show, its strongest low-tier traits
/// still convey what the program *does*, so a file wrongly graded hostile can
/// be talked back down.
pub(super) fn select_ids(file: &FileAnalysis) -> Vec<&str> {
    let composite_legs: BTreeSet<&str> = composite_atoms(file, file.findings.iter());

    // Per id, keep the best score and the highest criticality seen — the
    // latter decides whether the id bypasses the cap.
    let mut scored: HashMap<&str, (f32, Criticality)> = HashMap::new();
    for f in &file.findings {
        // An inherited copy (`src` set, native to a member below) is rendered by
        // its origin member, not here. A *cross-file composite* also carries
        // source provenance — `composite_sources` records the members it drew
        // from — but is native to no member at all: it exists only on this
        // container.
        let inherited_copy = f.src.is_some() && !file.composite_sources.contains_key(f.id.as_str());
        // Baseline (universal low-signal noise) is dropped from the primary
        // selection; referenced component traits stay. (Both tiers re-enter
        // below as fill candidates.)
        if inherited_copy
            || !tiny_should_show(f, file)
            || !f.trait_refs.is_empty()
            || f.crit < Criticality::Component
            || f.crit == Criticality::Baseline
        {
            continue;
        }
        let score = f.conf * f32::from(f.crit.rank());
        scored
            .entry(f.id.as_str())
            .and_modify(|(s, c)| {
                *s = s.max(score);
                *c = (*c).max(f.crit);
            })
            .or_insert((score, f.crit));
    }
    let mut ranked: Vec<(&str, f32, Criticality)> =
        scored.into_iter().map(|(id, (s, c))| (id, s, c)).collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    let (elevated, filler): (Vec<_>, Vec<_>) = ranked
        .into_iter()
        .partition(|&(_, _, crit)| crit >= Criticality::Suspicious);
    let mut out: Vec<&str> = elevated.into_iter().map(|(id, _, _)| id).collect();
    let remaining = TOP_N.saturating_sub(out.len());
    out.extend(filler.into_iter().take(remaining).map(|(id, _, _)| id));

    // Guarantee the legs of every omitted composite. They are usually
    // `component`, so the cap or the ranking could otherwise drop the only
    // evidence left for a finding whose conclusion was just withheld. Native
    // only: a leg carried by another member is rendered there.
    let mut have: HashSet<&str> = out.iter().copied().collect();
    for f in &file.findings {
        let id = f.id.as_str();
        if f.src.is_none() && composite_legs.contains(id) && have.insert(id) {
            out.push(id);
        }
    }

    // Top up with the best low-tier traits the primary pass excluded
    // (unreferenced components, baselines) — additive only.
    if out.len() < LOW_TIER_FILL {
        let mut low: HashMap<&str, f32> = HashMap::new();
        for f in &file.findings {
            if f.src.is_some()
                || !matches!(f.crit, Criticality::Component | Criticality::Baseline)
                || have.contains(f.id.as_str())
            {
                continue;
            }
            let score = f.conf * f32::from(f.crit.rank());
            low.entry(f.id.as_str())
                .and_modify(|s| *s = s.max(score))
                .or_insert(score);
        }
        let mut low_ranked: Vec<(&str, f32)> = low.into_iter().collect();
        low_ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        out.extend(
            low_ranked
                .into_iter()
                .take(LOW_TIER_FILL - out.len())
                .map(|(id, _)| id),
        );
    }
    out
}

/// Display path for a header. An archive member is shown archive-relative — the
/// container's on-disk directory is noise and the internal `!!` delimiter reads
/// better as `/`: `/tmp/x.zip!!lib/a.so` → `x.zip/lib/a.so`. A standalone/root
/// path is reduced to its basename: the directory is noise and can leak a
/// corpus label.
pub(super) fn tiny_path(path: &str) -> String {
    fn basename(p: &str) -> &str {
        p.rsplit(['/', '\\']).next().unwrap_or(p)
    }
    let path = match path.split_once(ARCHIVE_DELIMITER) {
        Some((root, members)) => {
            format!(
                "{}/{}",
                basename(root),
                members.replace(ARCHIVE_DELIMITER, "/")
            )
        }
        None => basename(path).to_string(),
    };
    cleave::output::sample_text(&path).into_owned()
}
