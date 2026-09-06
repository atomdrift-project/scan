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

## Removed: the archive-member escalation rule (2026-08-27)

The prompt used to close its first paragraph with an absolute:

> A malicious embedded archive member (a path nested under an archive, e.g.
> `app.zip/evil.sh`) makes the whole sample hostile.

It was removed. It duplicated — badly — a rule the engine already applies, and it
only ever cost false positives:

- **The engine's rule is the real one.** `worst_member` (`engine.rs`) reduces over
  member `Decision`s with `decision_outranks` (class first, probability on ties)
  and elevates the container to the worst of them, logging *"elevated archive
  classification due to embedded file"*. That is a calibrated comparison of ML
  decisions.
- **The prompt could not do that.** The model sees only the rendered `# SEV`
  annotations, never a member's decision, so "malicious member" degraded in
  practice to "nested path carrying a finding" — a single `S` satisfied it.
- **It could not add a true positive.** The elevation runs *before*
  `interpret::interpret`, which is handed the already-elevated
  `final_decision.{class,probability,level}`. A genuinely hostile member has
  already moved the ML verdict by the time the LLM is asked.
- **It outranked the hedge.** An unconditional "makes the whole sample hostile"
  overrides the paragraph-2 instruction to verify each description against the
  source, so the model stopped verifying and echoed the rule back as its reason.

The failure that surfaced it: `github.com/bradfitz/latlong`, a Go module whose
generated `z_gen_tables.go` embeds timezone tables as base64'd gzip. cleave
decodes that region and renders it as a synthetic member,
`z_gen_tables.go##base64@0xaaa`, with one `S` finding — and every package format
(gem, whl, tgz, zip, conda) is an archive, so the "nested path" precondition is
universally true in this corpus.

Ablation, prompt otherwise unchanged (Qwen3.8-27B, `temperature 0`):

| sample | rule kept | rule removed |
|--------|-----------|--------------|
| `latlong` (FP) | suspicious — *"Embedded base64 gzip payload"* | **benign** — *"legitimate Go library"* |
| `pyarmor` (FP) | benign | benign |
| `omniauth-ldap` (clean) | benign | benign |
| `gitversion-5.1.4` (FP) | benign | benign |
| `iamhungryrn` (**bad**) | suspicious | suspicious |
| `darkglitch` (**bad**) | hostile — *"DarkGlitch RAT with C2 and file exfil"* | hostile — *identical reason* |

No true positive moved; on `darkglitch` the reason came back byte-for-byte
identical, so the rule was doing no work there at all.

## The adjustment algorithm (gate + blend)

Beyond the prompt, these tuning points govern whether `--interpret` delivers the
right verdict:

1. **The gate — the calibrated level axis, not a probability.** `--interpret` runs
   when ML fires at or below `--llm-min-level` OR cleave surfaced a
   suspicious/hostile finding (the *elevated-finding bypass*) OR the scan's own
   class is non-benign. The cutoff defaults to the model's **grid ceiling**
   (`LevelContext::ml_admits` resolves `None` to `grid_max`), so ML admits any file
   it placed anywhere on the calibrated grid; only `ml.lvl = -1` — ML saw nothing —
   is an ML non-admission.

   It replaced a flat `prob ≥ 0.01` floor, which could not express "ML saw
   something": the score that means *something* is per-route. On the 2026-08-21
   azoth bundle, L10000 lands at 0.036 for `clojure`, 0.99 for `python`, 0.9996
   for `pe` — so one scalar is simultaneously far too loose for one file type and
   far too tight for another, and 0.01 sat below all of them. ML probabilities on
   real malware bunch near 0 *and* near 1 with little in between, so the old floor
   mostly bought LLM calls on genuinely-clean files.

   **Why the ceiling and not a literal rung.** L10000 and L25000 are nearly the
   same cutoff in practice — on that bundle 58 of 104 comparable routes carry an
   *identical* threshold at both, and most of the rest differ by 0.037 (the shared
   `general` row); only `scala` (0.946 → 0.393) and `markdown` (0.920 → 0.820)
   move meaningfully. The benign quantile runs out of tail resolution well before
   the ceiling — what azoth's own search notes as levels "clustering at the 1-FP
   ceiling below resolution". So a literal buys nothing over the ceiling but can
   drift: `level_confidence` already reserves rungs for an L50000 grid, and a
   hardcoded `25000` stops meaning *the whole grid* the day the grid is re-cut.

   Note the admission sits far past the suspicious ceiling (L3000, see
   `SUSPICIOUS_LEVEL_CEILING`): a file admitted by ML alone out here is *not*
   suspicious under the deployed policy, just the weak tail. Measured ML
   false-negatives (a crypto clipper at 0.024, a git-push exfil at 0.039, an
   unverified download-exec at 0.012) sit under *every* calibrated cutoff — they
   are carried by the elevated-finding bypass, not by the ML admission, so the
   level knob cannot chase them; lowering it only buys clean-file calls.

   Off-grid trait-floor markers (`grid_max + 1/2`) fall outside the ceiling and are
   not an ML admission — they are floored *because* a trait fired, so the class
   and finding bypasses already carry them.

   In manual-threshold mode there is no level axis; ML abstains from the gate and
   the bypasses carry it (a score over an operator-set threshold is already
   non-benign).
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

## Prompt size: where the tokens went (2026-09-05)

Measured with `hacks/prompt-study.py` over the 113 renderable of the 128 most
recent PURLs beamline asked the fleet for (`atomscan -f interpret purl …`,
tokenized by the fleet's own vLLM). Median payload 2,652 tokens, p90 6,742,
max 54,000.

| share of all prompt tokens | |
|---|---|
| provenance JSON | 40% |
| — of which `registry.raw` provider documents | **30%** |
| hex byte windows inside members | 13% |
| findings, source context, headers | ~47% |

- `raw` was a packument projection (`time`, `dist`, `versions`, `_rev`,
  `maintainers`, …) repeated per subject: 53–60% of an npm/gem/cargo payload,
  24% of a pypi one. Every identity signal in it is summarised by the record
  fletch derives from it, so `slim_provenance_for_interpret` now drops `raw`
  and `project_registry_record` keeps the *whole* identity in words —
  title, description, publisher/author/email domain, maintainers, repository,
  both download counts, package and version age, release cadence,
  `latest_version`, and the registry's own flags (install scripts, security
  holds, removed/deprecated versions). ~90–220 tokens where `raw` was
  ~1,000–1,400. A dependency-confusion placeholder now reads
  `"description":"security holding package","maintainers":0,"downloads_recent":10`.
- The outliers are structural: a 39 MB jar inside a wheel fanning out to 83
  `.class` members (17.7k tokens), and a bundled OpenSSL dylib whose 14
  findings each carry a hex window (16.5k). Those are cleave-side (nested
  archive fan-out, `TinyOpts::tiny()` inheriting `full_context: true`).
- **Byte windows stay.** Removing them (`tune.py --templates nohex`) cut ~19%
  of tokens and turned a known-bad PE (`fffmpeg.exe`) hostile→benign on the
  shipped prompt while every finding line remained: the rows are the evidence
  the model grades from, the `// N …` descriptions are hints. The point of the
  pass is to catch what the rules miss, so cut hinting and metadata, never
  evidence.

### File identity: every claim, not the headline's three

A PE, a Mach-O or an Office document is never a package, so dropping `raw`
left them with only what the file says about itself. That arrives through
cleave's minimal header (`identity_headline`) rather than `provenance=`, and
the headline names *one* subject, *one* responsible party and a trust tier —
so the other half of every pair was being discarded, and the other half is
routinely the interesting one. cleave `3c5bd9d2` now renders the remainder as
labelled pairs, because the disagreements are the point:

- a signed PE claiming `company="Contoso Ltd"` in its version resource while
  its Authenticode chain says `Vanguard Tech Limited`, with the chain's own
  `subject`/`issuer`/`at` beside it (`fffmpeg.exe`: SSL.com intermediate,
  London org, signed 2024-03-12);
- an Office document whose `company` lost to its author, plus every party
  that touched it rather than the first;
- a Mach-O's `team` and bundle `version`, both dropped whenever the bundle
  identifier was the subject;
- a VS Code extension that previously rendered *no* identity at all and now
  leads with `sfra-toolkit — SFRA-FAKA` plus `claims file="SFRA Toolkit"`;
- the producing tool on anything signed, which the trust word displaced.

Measured cost over the eight-sample A/B corpus: **+243 tokens (+0.9%)**, and
nothing at all on files that carry no identity. Guarded by
`engine.rs::interpret_render_carries_file_identity_for_unregistered_artifacts`
(scan side, PE + Mach-O + docx through `render_interpret_context`) and five
`output.rs` tests in cleave.

The A/B for the `raw` drop ran on two samples from each of `~/data/benchmark`'s
`compendium-clean`, `compendium-dirty`, `scan-purls` and `mspd` pools (highest
ML probability plus one borderline per pool, identity via `--registry-map`),
majority of 3 runs on the fleet's Qwen3.8-27B:

| arm | exact | mean tokens |
|---|---|---|
| `described` (shipped, `raw` kept) | 6/8 | 4,041 |
| `noraw` | **7/8** | 3,388 |
| `pointer` (bare markers, less hinting) | 6/8 | 3,181 |
| `raw` (no annotations) | 6/8 | 2,951 |

The sample `noraw` fixes is `node-ipc@11.1.0` — the clean release of the
protestware package — which grades *hostile, flapping* with the packument's
version history in front of the model and benign without it; the low-hint arms
call it hostile too. The one miss every arm shares, a VS Code extension with no
registry record and only notable-level metadata findings, is a baseline blind
spot rather than a render effect. The same corpus was then scanned for real
(`atomscan --interpret`, old binary vs new, 3 runs each, no analysis cache) as
the authoritative check: **7/8 → 7/8, every grade identical across all six
runs, no flapping**, with the same single miss — at 14% fewer prompt tokens
over the corpus (up to 49% on a small package, where `raw` was most of the
payload; binaries and samples without a registry record are unchanged). Two traps that harness run exposed, now guarded by
`engine.rs` tests (`interpret_provenance_carries_identity_and_drops_provider_documents`,
`interpret_render_keeps_byte_windows_as_evidence`): the render's first line and
`provenance=` carry the sample *path*, so a corpus laid out by pool name tells
the model which pool it is grading (lay samples out as `s1/…sN/<basename>`);
and capture runs `SCAN_FOLLOW=none`, which also skips the `*.forage.json`
sidecars — identity comes from `--registry-map`, built with
`fletch registry <purl>` keyed by sha256.

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

The endpoint requires a bearer token. `tune.py` reads it from `~/.tok/llm`
(`SCAN_LLM_KEY` overrides) and warns when it finds neither, which matters here:
a refused call scores as a miss, so an unauthenticated sweep reads as a model
that got the whole corpus wrong rather than as a run that never happened.

## Prompt sweep on FP8 (2026-09-06)

Measured on the four labelled sets (mspd 226, compendium-dirty 191,
compendium-clean 153, scan-purls 249; renders and harness under
`hacks/vllm-tune/`, results in `~/data/vllm-tune/`), grading the same
rendered prompts with each candidate system prompt against the fleet's vLLM
(Qwen3.8-27B-FP8, temperature 0). "Right side" is hostile+suspicious on the
bad sets and benign on the clean ones. BF16-vs-BF16 repeat agreement is
99.3–100% per set, so a one-sample move is noise.

| prompt | tokens | mspd | comp-dirty | comp-clean | scan-purls |
|---|---|---|---|---|---|
| previous (2026-09-05) | 599 | 76.1 | 75.9 | 96.1 | 98.8 |
| previous + signed-binary identity paragraph | 669 | 76.5 | 77.0 | 96.1 | 98.8 |
| tightened rewrite + identity | 520 | 75.2 | 75.9 | 96.1 | 99.2 |
| **hedge sentences verbatim, exposition cut, + identity (shipped)** | **508** | **78.8** | **77.0** | 95.4 | 98.8 |
| minimal + identity | 348 | 83.6 | 78.0 | 94.1 | 97.6 |
| ultra + identity | 278 | 85.8 | — | 92.8 | 97.2 |

Two findings drove the choice:

- **The signed-binary identity paragraph is what flips `Delfino.exe`.** A
  CA-signed PE whose version resource claims DreamSecurity while the
  Authenticode subject is another company, with cleave's stolen-certificate
  rule firing, graded *suspicious* on FP8 under the previous prompt and the
  blend de-escalated it from hostile L12 to suspicious L3000. Telling the
  model that a signer/claim mismatch or a certificate reported stolen is
  impersonation returns it to hostile, and costs nothing on the 402 clean
  samples (compendium-clean includes signed proprietary software).
- **The hedge sentences carry the clean sets; the exposition does not.**
  Every shorter prompt that paraphrased "a category alone is never evidence
  of malice … verify each description against the actual source" bought its
  extra detection with false positives: +6 clean samples at 348 tokens, +7 at
  278. Keeping those sentences word for word and cutting only the format
  description (LINE:COL/@OFFSET semantics, the xxd gutter, blank-line
  windows) kept the clean sets whole at 508 tokens.

The one stable clean-set change under the shipped prompt is
`@tiledesk/tiledesk-server 2.18.5` reading *suspicious* ("Obfuscated Stripe
module in package") in both repeats — a review flag on a package that does
ship an obfuscated module, which is what the grade means.

Format decision made the same day, same harness: FP8 (`Qwen/Qwen3.8-27B-FP8`)
agrees with BF16 at 99.0–100% per set at ~1.5× throughput; NVFP4
(`Inferact/Qwen3.8-27B-NVFP4`) runs 2× but de-escalates 20–26 samples per
bad set and returns unparseable replies, so it was rejected.
