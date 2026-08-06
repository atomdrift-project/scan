# `--interpret` prompt tuning

How the LLM second-opinion render was tuned, and how to validate/tune it on a new
batch of samples.

## The problem

`--interpret` sends cleave's `tiny()` render to a local LLM for a trinary grade
(benign/suspicious/hostile), blended with the ML verdict. That render annotates
each finding with a prose description — `# SEV LINE:COL desc`. Presented at face
value, the model tended to **parrot the description** instead of reading the
source:

- a benign donations list → *"hardcoded Bitcoin address"* → hostile
- a bundled Microsoft dep's install hook → *"outbound data exfiltration"* → hostile
- Fabric `InlineBase64` API encoding → *"webshell payload builder"* → suspicious
- and the mirror image: reassuring `notable` descriptions ("conventional version
  format", "fetch() call") talked a real download-and-exec dropper *down* to benign.

## The shipped design: one template — descriptions kept, hedged

The live query sends the sanitized render **verbatim** — every `# SEV LOC desc`
annotation, prose included — and the system prompt (`interpret.rs::SYSTEM_PROMPT`)
frames the descriptions honestly:

> The `desc` is the analyzer's interpretation of a pattern it matched — what the
> code COULD be doing, not a confirmed detection; false positives are possible, so
> verify each description against the actual source and judge the code yourself,
> discounting any description it does not support.

The hedge targets the parroting failure directly, while keeping the description
signal that matters on packed/stripped binaries — there the render is escaped
bytes and cleave's prose ("Encrypted loader data geometry", "Unpacking API
pattern") is the only readable signal, so stripping it (an earlier experiment,
below) turned real droppers/stealers into benign false-negatives.

There is deliberately **no template flag**: earlier tuning grew a
`--interpret-template` with five render variants (`full`/`adaptive`/`pointer`/
`elevated`/`raw`), but the variants were experiment arms, not product surface, and
the offline harness (`hacks/interpret-tune/tune.py`) can sweep render/prompt
variants without any Rust-side switch. The verdict cache key includes the system
prompt, so editing the prompt keys fresh verdicts rather than replaying old ones.

`--format interpret` output is byte-for-byte the live query's user message —
the same sanitized render, annotations included — just without the system
prompt; a downstream consumer should frame the descriptions with the same hedge.

## Historical measurements (two 30-sample batches, ground truth by manual RE incl. rizin/deobfuscation)

Labels: `hacks/interpret-tune/labels/hopper-triage.tsv` (batch 1, source-heavy) and
`…-batch2.tsv` (batch 2, packed-binary-heavy). These runs predate the hedged
prompt and motivated it:

| experiment arm | batch 1 (source) | batch 2 (binary) |
|----------------|------------------|------------------|
| `full` (unhedged descriptions) | 19/30 | 21/30 |
| `pointer` (bare `# SEV LOC`, no prose) | **28/30** | 21/30 |
| `adaptive` (prose on hex only) | ≈28/30 | **25/30** |

- **Batch 1** is where unhedged prose *biases* the model: `full` false-positives
  on legitimate packages (a cleave `H` finding on Microsoft's
  `vscode-languageserver-protocol` → "hostile"), and reassuring notable prose talks
  a download-exec dropper *down* to benign.
- **Batch 2** is where prose is *load-bearing*: packed/stripped binaries render as
  escaped bytes, so `pointer` strips the only readable signal and turns real
  droppers/crypters/stealers into benign false-negatives.
- The shipped `described` prompt aims to win both by keeping the prose and
  hedging it; validate it against these baselines with the harness
  (`tune.py --templates described,full,pointer`) when the labels or prompt change.
- Residuals are borderline and in the safe direction: a crypto **donations** list
  reads as suspicious; dual-use hacktools (neutron, hcxdumptool) read as
  benign/suspicious; a decoy "FP-bait" binary reads as suspicious/hostile.

## The adjustment algorithm (gate + blend)

Beyond the prompt, these tuning points govern whether `--interpret` delivers the
right verdict:

1. **The gate — floor lowered to 0.01.** `--interpret` runs when ML
   `prob ≥ --interpret-min-prob` OR cleave surfaced a suspicious/hostile finding
   (the *elevated-finding bypass*). The LLM's whole value is rescuing samples ML
   *under*-scored — measured false-negatives sit far below the old 0.10 floor (a
   crypto clipper at 0.024, a git-push exfil at 0.039, an unverified download-exec
   at 0.012) — so the default floor is **0.01** (`DEFAULT_MIN_PROB`). Note ML
   probabilities on real malware bunch near 0 *and* near 1 with little in between,
   so a low floor mostly adds LLM calls on genuinely-clean files (which the LLM
   correctly clears), not accuracy risk.
2. **The steering rule — one step, one third.** `interpret::blend` enforces a
   single symmetric bound in both directions:

   > The LLM may move the verdict **at most one severity step**, and the
   > confidence **at most 33%** (`MAX_STEER`) of the distance to the bound it
   > argues for.

   ML decides; the LLM steers. Two consequences worth stating plainly:
   - **benign + LLM-hostile → suspicious**, not hostile. A two-step jump on one
     model's word, over input the sample's author controls, is a review flag
     rather than a block.
   - **hostile + LLM-benign → suspicious** at most, never benign.

   The cap is a *fraction of the remaining range*, so it is proportionate at any
   starting score and preserves ML's ranking — two files the LLM grades the same
   stay ordered by what ML thought of them. The previous blend replaced the score
   with a flat 0.80/0.85 on escalation, which erased that ordering entirely (a
   0.0002 and a 0.45 both landed on exactly 0.85) and was in no sense a blend.
   Agreement steers toward the pole the verdict already sits at — corroborated
   malice up, a corroborated clean file *down* (the old code raised the malice
   probability of a file both graders called benign).
3. **A *hostile* band crossing must be earned.** The verdict travels downstream as
   `ml.lvl`, not as `ml.prob` — `MlSection` serializes no class field, so a
   consumer reconstructs the class purely from the level via `verdict_for_level`.
   Bounding only the score would therefore leave the number that actually carries
   the verdict unguarded.

   But the two boundaries are not the same kind of claim, so they are not gated
   the same way (`Evidence::may_cross`):

   - **The hostile boundary spends a budget.** `-l` *is* an FP-per-100M budget, so
     asserting a file belongs inside it when ML placed it nowhere near is a
     calibration claim one fallible opinion cannot back. Crossing it — in either
     direction — requires ML to already sit within one steer of the line
     (`within_one_steer_of_hostile_boundary`). At `-l 25`, a file firing at L30
     can be tipped into hostile while one at L500 cannot; a hostile firing at L20
     can be talked down while one at L0 cannot.
   - **The suspicious boundary is a routing decision** — "should a human look?" —
     with no budget attached, so it is ungated. Two detectors disagreeing is
     precisely the signal that someone should look. Gating it would also break the
     feature's main purpose: an ML false negative sits at `lvl = -1` by
     definition, confidence 0, which no bounded steer can lift. **ML benign + LLM
     hostile therefore always lands on suspicious/review**, regardless of where ML
     placed the file or whether the render is readable.

   Proximity is measured in *confidence* space (`level_confidence`), the same
   pessimistic table that produces `ml.conf`, and the predicate is symmetric by
   construction — `min`/`max` means one expression covers either direction.

   The one relaxation on the hostile side is an **escalation corroborated by a
   cleave hostile (`H`) finding** — two independent detectors agreeing with ML as
   the outlier, which is what lets a packed dropper ML scored as merely suspicious
   reach hostile. Deliberately `H`-only (not `H`-or-`S`, which is what the *gate*
   uses) and deliberately escalation-only: corroboration must never help a sample
   earn its own clearing.
4. **Clearing is guarded; escalation is not.** Every *softening* — a class drop
   or a score drop, including the downward steer on an agreed-benign file — is
   discarded outright when the render is opaque (`render_mostly_readable`) or
   carries analyzer-directed text (`addresses_the_analyzer`). ML's class and score
   are left exactly as they were. Escalation is deliberately ungated by these two:
   an attacker gains nothing by talking their own sample up, so only the softening
   path is worth attacking. The invariant, asserted over every input combination
   in `untrusted_read_never_softens_the_verdict`:

   > When the LLM's read is untrusted, the blend never lowers the class and never
   > lowers the score.
5. **Synthetic level.** `engine::interpreted_level` pins an LLM-shifted verdict to
   the loosest rung of the target band — the active deploy `-l` for hostile,
   `capped_suspicious_level(grid_max)` for suspicious, `-1` for benign — so the
   level tracks the model's real thresholds instead of a hardcoded rung. `ml.conf`
   is `level_confidence()` of that level, so the displayed confidence follows
   automatically; `ml.prob` carries the steered score.

### Trade-offs these rules accept

**Escalations land one band lower.** Capping at one step means the measured ML
false-negatives above (the clipper at 0.024, the exfil at 0.039) surface as
**suspicious** — flagged for review rather than blocked. That is the intended cost
of not letting a single fallible opinion over attacker-controlled text carry a
verdict from one end of the scale to the other.

**Some hostile escalations no longer land.** An LLM hostile on a file ML scored
far from the deploy level is recorded in the `llm` section and surfaces as
suspicious, but does not reach hostile. Whether that is the right cut depends on
threshold geometry that varies by model, route, and filetype — which is why it
needs validating against the labeled batches (below) rather than reasoning about
in the abstract. The two knobs if it cuts too hard are `MAX_STEER` and widening
the corroboration hatch from `H`-only to `H`-or-`S`.

**Review volume rises.** Because the suspicious boundary is ungated, every
LLM-hostile or LLM-suspicious verdict on an ML-benign file becomes a review flag.
That is the intended trade — a false review costs a human minute, a miss ships
malware — but it makes the LLM's false-hostile rate on clean files the number that
governs triage load. Batch 1 in the table above is exactly that measurement, and
it is the first thing to re-check when the prompt or the model changes.

**Why not derive the class from a steered probability instead?** That was the
first design, and it does not work. There are two decision paths
(`Model::decide`): `decide_from_scores` classifies by comparing a probability to
`Thresholds`, but the default `decide_swept` classifies by *level*. For
general-grid filetypes the two are equivalent, but for route-policy filetypes
they are not — `PolicySeverity::fire` compares each route's own score to that
route's own cutoff, and a learned blend reports `sigmoid(intercept + Σ wᵢ·logit(pᵢ))`,
which is not in the same space as the `severity_levels[]` thresholds at all.
Classifying a steered probability against a resolved `Thresholds` block would
silently produce wrong verdicts on exactly the filetypes with tuned policies. The
level axis is the only coordinate both paths share, which is why the gate is
measured there.

## Prompt injection

The render is not merely untrusted, it is **100% authored by the party being
graded**, who can assume it will reach an LLM. Every profitable injection is a
*de-escalation* ("this is not malware"), which is why the guard is asymmetric.

Three layers, in increasing order of how much they can be relied on:

1. **Prompt scoping** (`SYSTEM_PROMPT`). The old wording scoped untrustedness to
   "the findings and provenance" while telling the model the source lines were
   shown "unaltered" — leaving the obvious injection surface (a comment reading
   `THIS IS NOT MALWARE`) inside the region the prompt vouched for. It now covers
   the whole user message and reframes analyzer-directed text as *evidence about
   the author* rather than something to merely ignore.
2. **Deterministic detection** (`addresses_the_analyzer`). A plain
   case-insensitive substring scan of the sanitized render for instruction
   openers, chat-template control tokens, fragments of our own prompt/reply
   schema, and direct assertions of innocence. It does not depend on the model's
   cooperation, which is exactly what layer 1 cannot promise. A hit only revokes
   the LLM's ability to *lower* a verdict, so the error cost is lopsided in the
   safe direction: a false positive costs one un-cleared ML false positive (a
   human triage minute), a miss costs a cleared malware sample. It is surfaced as
   `llm.inject: true` in the JSON and logged at WARN.
3. **Reply hygiene.** The model's `reason` is attacker-influenced text that lands
   in an operator's terminal, so it is ANSI-stripped, control-stripped, and
   clamped to `MAX_REASON_CHARS` — the render is sanitized on the way in, this
   closes the same hole on the way out.

Note that cleave will often flag injection strings as traits in their own right,
but that does **not** protect this path: the render still carries the raw bytes
into the prompt regardless of what was flagged. Layer 2 is what acts on them.

Deliberately *not* done: redacting matched text from the render. It would hide
real evidence from the grader and buy little, since the sample is distrusted for
clearing purposes either way.

**vLLM nondeterminism.** Even at `temperature 0`, batched inference is not
bit-reproducible; a borderline file can flip grade run-to-run. The prompt-hash
verdict cache pins the first verdict in production; score 3× when validating.

## Validating / tuning on a new batch

The harness is `hacks/interpret-tune/tune.py`; the workflow is:

1. Capture the LLM render per file from a single scan via the
   `SCAN_INTERPRET_DUMP_DIR` dump hook (no `--interpret` / LLM calls needed), or
   let `tune.py` do it for you.
2. Hand-label each file's ideal grade (reverse-engineer binaries; be honest about
   files whose verdict lives in an unavailable payload — mark them borderline).
3. Sweep the experiment arms offline against the LLM endpoint
   (`tune.py --templates described,full,pointer,raw`); `described` mirrors the
   shipped prompt.
4. Score exact + within-acceptable; repeat 3× to see through the nondeterminism.

Rebuilds are only needed to ship a prompt change (edit
`interpret.rs::SYSTEM_PROMPT`) — the whole sweep runs against the LLM endpoint
directly.
