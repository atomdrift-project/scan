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

Beyond the prompt, three tuning points govern whether `--interpret` delivers the
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
2. **Content-gated safety valve.** When ML says hostile and the LLM says benign,
   the blend clears to benign **only if the render is readable source** (the LLM
   read the actual code — clears ML false positives like a signed tool ML
   mislabeled). If the render is opaque/packed bytes, it holds at suspicious +
   review, since a text model can be fooled by obfuscation. See
   `interpret::blend` / `render_mostly_readable`.
3. **Synthetic level.** `engine::interpreted_level` lands an LLM escalation at L4
   (hostile) / L99 (suspicious) — inside the current L25 / L3000 bands at the
   default `-l 25`, and low enough that an LLM "hostile" stays hostile at stricter
   deploy levels.

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
