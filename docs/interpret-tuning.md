# `--interpret` template tuning

How the LLM second-opinion render was tuned, and how to validate/tune it on a new
batch of samples.

## The problem

`--interpret` sends cleave's `tiny()` render to a local LLM for a trinary grade
(benign/suspicious/hostile), blended with the ML verdict. That render annotates
each finding with a prose description — `# SEV LINE:COL desc`. The model tended to
**parrot the description** instead of reading the source:

- a benign donations list → *"hardcoded Bitcoin address"* → hostile
- a bundled Microsoft dep's install hook → *"outbound data exfiltration"* → hostile
- Fabric `InlineBase64` API encoding → *"webshell payload builder"* → suspicious
- and the mirror image: reassuring `notable` descriptions ("conventional version
  format", "fetch() call") talked a real download-and-exec dropper *down* to benign.

## The fix: `--interpret-template`

`--interpret-template` (`interpret.rs::InterpretTemplate`):

| mode | what the model sees |
|------|---------------------|
| `full` | cleave's `# SEV LOC desc` everywhere (pre-tuning behavior) |
| **`adaptive`** (default) | per finding: keep the prose where it anchors hex/binary bytes, drop to `# SEV LOC` where it anchors readable source |
| `pointer` | `# SEV LOC` everywhere — no prose |
| `elevated` | `# SEV LOC` for hostile/suspicious only; notable/baseline dropped |
| `raw` | no annotations; the model reasons from source alone |

The system prompt is swapped to match (the pointer-style prompt tells the model
the marker is only a hint — "decide the grade yourself from the source"). The
verdict cache key includes the system prompt, so switching templates never
replays a cross-template verdict.

### Why adaptive, not plain pointer

`pointer` won a **source-heavy** corpus (batch 1: 28/30 vs full's 19/30) because
there the prose *biases* the model — it parrots "hardcoded Bitcoin address" over a
donations list. But a **binary-heavy** corpus (batch 2: packed/stripped Windows +
ELF malware) *reversed* it: those samples render as unreadable escaped bytes, so
cleave's prose ("Encrypted loader data geometry", "Win32 clipboard access",
"Unpacking API pattern") is the **only** signal — stripping it turned real
droppers/stealers into benign false-negatives.

`adaptive` decides per finding: for a finding whose following context is escaped
binary (`is_binary_render` — `\xNN`/`\0` escapes or a low printable ratio) it keeps
the full prose; otherwise it pointerizes. Result on batch 2: **adaptive 25/30 vs
full 21/30 vs pointer 21/30** — it keeps pointer's zero false positives on the legit
`good/`/`new/` set *and* recovers the binary false-negatives, without regressing
batch 1's source wins.

## Results (two 30-sample batches, ground truth by manual RE incl. rizin/deobfuscation)

Labels: `hacks/interpret-tune/labels/hopper-triage.tsv` (batch 1, source-heavy) and
`…-batch2.tsv` (batch 2, packed-binary-heavy).

| template | batch 1 (source) | batch 2 (binary) |
|----------|------------------|------------------|
| `full` (old default) | 19/30 | 21/30 |
| `pointer` | **28/30** | 21/30 |
| **`adaptive`** (default) | ≈28/30 | **25/30** |

- **Batch 1** is where prose *biases* the model: `full` false-positives on
  legitimate packages (a cleave `H` finding on Microsoft's
  `vscode-languageserver-protocol` → "hostile"), and reassuring notable prose talks
  a download-exec dropper *down* to benign. Dropping prose fixes both — no
  bloom-suppression needed.
- **Batch 2** is where prose is *load-bearing*: packed/stripped binaries render as
  escaped bytes, so `pointer` strips the only readable signal and turns real
  droppers/crypters/stealers into benign false-negatives. `full`/`adaptive` keep it
  and catch them.
- `adaptive` is the only template that wins both — per-finding it keeps prose on
  hex context and drops it on source.
- Residuals are borderline and in the safe direction: a crypto **donations** list
  reads as suspicious; dual-use hacktools (neutron, hcxdumptool) read as
  benign/suspicious; a decoy "FP-bait" binary reads as suspicious/hostile.

## Two things the template does NOT change (set them per run)

1. **The gate.** `--interpret` only runs when ML `prob ≥ --interpret-min-prob`
   (default 0.10). ML false-negatives sit *below* that (e.g. an unverified
   download-exec at 0.01), so **triage/validation runs must pass
   `--interpret-min-prob 0`** or the LLM never sees them.
2. **vLLM nondeterminism.** Even at `temperature 0`, batched inference is not
   bit-reproducible; a borderline file (neutron) can flip grade run-to-run. The
   prompt-hash verdict cache pins the first verdict in production.

## Validating / tuning on a new batch

The tuning harness lives in the session scratchpad, but the workflow is:

1. Capture the LLM render per file: `atomscan --interpret --interpret-min-prob 0
   --verbose <file>` and extract the block between `--- user ---` and ` model=`.
2. Hand-label each file's ideal grade (reverse-engineer binaries; be honest about
   files whose verdict lives in an unavailable payload — mark them borderline).
3. Transform the render per template (drop/trim the `{#|//|--} SEV [LOC] desc`
   annotation lines) and query the LLM directly with the matching system prompt.
4. Score exact + within-acceptable; repeat 3× to see through the nondeterminism.

Rebuilds are only needed to ship a change — the whole sweep runs against the LLM
endpoint directly.
