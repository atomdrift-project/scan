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

Strip the prose, keep the bare pointer. Four modes (`interpret.rs::InterpretTemplate`):

| mode | what the model sees | 
|------|---------------------|
| `full` | cleave's `# SEV LOC desc` (pre-tuning behavior) |
| **`pointer`** (default) | `# SEV LOC` — severity + location, **no prose** |
| `elevated` | `# SEV LOC` for hostile/suspicious only; notable/baseline dropped |
| `raw` | no annotations; the model reasons from source alone |

The system prompt is swapped to match (`pointer`/`elevated` tell the model the
marker is only a pointer — "decide the grade yourself from the source"). The
verdict cache key includes the system prompt, so switching templates never
replays a cross-template verdict.

## Result (dataset: `/var/tmp/hopper-triage`, 30 samples, ground-truth by manual
RE incl. rizin/deobfuscation)

| template | agreement with expert triage |
|----------|------------------------------|
| `full` (old default) | 19/30 (63%) |
| **`pointer`** | **28/30 (93%)** |

- **Zero regression on the 6 confirmed-malware samples** — they stay hostile even
  in `raw`; the raw bytes (C2 domains, flood templates, clipper symbols, Telegram
  token) convict them without cleave's prose.
- `pointer` alone fixes the bundled-dependency false positives (a cleave `H`
  finding on Microsoft's `vscode-languageserver-protocol`) — **no bloom-suppression
  needed** once the no-location prose is stripped too.
- Residuals are borderline and in the safe direction: a crypto **donations** list
  (gemini-coder) reads as suspicious; a dual-use vuln-scanner (neutron) reads as
  suspicious/hostile.

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
