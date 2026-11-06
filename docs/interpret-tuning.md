# `--llm`: design and tuning

How the LLM second opinion decides what to send, what it sends, and how its
answer moves a verdict — and how to check a change to any of that before
shipping it. The code is in `src/interpret/` (gate, prompt, blend, client);
the system prompt is `interpret::SYSTEM_PROMPT`.

## What is sent: the render, descriptions kept and hedged

The user message is cleave's sanitized context render: every finding as
`# SEV LOC desc`, the source lines or byte windows around it, and the file's
identity and registry provenance. `atomscan -f interpret` prints exactly this
message, without the system prompt, and contacts no LLM.

Taken at face value, the finding descriptions get parroted: a donations list
becomes "hardcoded Bitcoin address", an install hook "data exfiltration", and
reassuring descriptions talk a real dropper down. Removing them is worse: on
packed or stripped binaries the render is escaped bytes, and cleave's
description is the only readable signal. So the descriptions stay, and the
system prompt frames them as the analyzer's interpretation of a pattern — what
the code *could* be doing — to verify against the source and discount when it
does not hold up.

Some deliberate choices in the render:

- **Byte windows stay.** They are the evidence the model grades from; the
  descriptions are hints. Cut hinting and metadata, never evidence
  (`interpret_render_keeps_byte_windows_as_evidence`).
- **Provider documents go; identity stays.** Raw registry documents are mostly
  version history and cost the most tokens. The record derived from them —
  publisher, maintainers, repository, downloads, package and version age,
  release cadence, the registry's own flags — is kept in words
  (`interpret_provenance_carries_identity_and_drops_provider_documents`).
- **Every identity claim, not just the headline.** A signed binary's version
  resource and its Authenticode signer are shown side by side, because the
  disagreement is the signal
  (`interpret_render_carries_file_identity_for_unregistered_artifacts`).
- **No archive rule in the prompt.** The engine already elevates a container
  to its worst member (`worst_member`, `decision_outranks`) before the LLM is
  asked. A prompt rule saying "a malicious member makes the sample hostile"
  could only see rendered findings, fired on any nested finding, and added
  false positives without adding a detection.

There is no template flag. Render and prompt variants are experiments, run
offline (below), not product surface.

## Who is asked: the gate

`interpret::gate::admit` evaluates every admission and logs which fired. A
sample is sent when any of these holds:

- ML placed it anywhere on the calibrated level grid (`ml.lvl` ≥ 0);
- its raw probability is at least `DEFAULT_LLM_MIN_PROB`, which covers files
  the grid never placed;
- cleave reported a hostile finding;
- a finding matches one of `LLM_GATE_PREFIXES` (trait families with a good
  malicious-to-benign exchange rate on their own);
- the scan's own class is non-benign — a container elevated by a member must
  reach the LLM even when its own score is low.

A size veto drops benign samples with more than
`DEFAULT_LLM_BENIGN_NOTABLE_CAP` notable findings, which are almost always
large legitimate packages, unless a hostile finding or one of
`LLM_GATE_STRONG_PREFIXES` is present. A sample placed on the grid is never
withheld by the veto.

`atomscan validate` checks every gate prefix against the installed traits and
warns about any that match nothing.

## How the answer moves the verdict: the blend

ML decides; the LLM steers (`interpret::blend`):

1. **One step, one third.** The LLM may move the class at most one severity
   step, and the score at most `MAX_STEER` of the distance to the bound it
   argues for. Benign + LLM-hostile becomes suspicious; hostile + LLM-benign
   becomes suspicious at most. The cap is a fraction of the remaining range,
   so ML's ranking is preserved.
2. **The hostile boundary is earned.** Crossing it in either direction
   requires ML to already sit within one steer of it
   (`within_one_steer_of_hostile_boundary`), measured in the same confidence
   space as `ml.conf`. The one relaxation is an escalation corroborated by a
   cleave hostile finding. The suspicious boundary is ungated: it routes a file
   to a human rather than spending a false-positive budget, so an ML false
   negative with an LLM-hostile read always lands on suspicious.
3. **Clearing is guarded; escalation is not.** A render that is mostly
   unreadable (`render_mostly_readable`) or that addresses the analyzer
   (`addresses_the_analyzer`) may not lower the class or the score. An author
   gains nothing by talking their own sample up, so only softening is guarded
   (`untrusted_read_never_softens_the_verdict`).
4. **The level follows the class.** The verdict travels as `ml.lvl`, so an
   LLM-shifted verdict gets a synthetic level in its new band
   (`engine::interpreted_level`) and `ml.cls` beside it; `ml.conf` follows from
   the level. Classifying a steered probability against thresholds would be
   wrong for routes with their own cutoffs; the level axis is the coordinate
   every decision path shares.

The cost of these rules is accepted on purpose: escalations land one band
lower, some LLM-hostile reads surface only as suspicious, and every
LLM-hostile read on an ML-benign file becomes a review flag. That makes the
LLM's false-hostile rate on clean files the number that governs triage load.

## Prompt injection

The render is written entirely by the party being graded, and every profitable
injection is a de-escalation. Three layers:

1. **The prompt** treats the whole user message as untrusted, and text aimed
   at the analyzer as evidence about the author.
2. **Detection** (`addresses_the_analyzer`): a deterministic scan for
   instruction openers, chat-template tokens, fragments of the prompt's own
   schema and assertions of innocence. A hit revokes only the LLM's ability to
   lower the verdict, and is reported as `llm.inject: true`.
3. **Reply hygiene:** the model's `reason` is stripped of ANSI and control
   characters and clamped to `MAX_REASON_CHARS` before it reaches a terminal.

Matched text is not redacted from the render: that would hide evidence from
the grader for no gain.

## Validating a change

Run this for any change to the system prompt, the render, the gate or the
blend, and when the serving model changes.

1. **Assemble labelled sets.** At least one set of known-bad samples and one
   of known-clean ones, including clean files that look alarming (signed
   proprietary binaries, large packages, obfuscated but legitimate modules).
   Label each file's ideal grade by hand; mark files whose verdict depends on
   a payload you do not have as borderline.
2. **Capture the renders** with `atomscan -f interpret` on the build being
   tested. No LLM is called. Lay samples out as `s1/<basename>` … `sN/…`: the
   render carries the sample path, and a directory named for the set tells
   the model which set it is grading. Supply registry identity with
   `--registry-map` rather than live fetches, so every run sees the same input.
3. **Grade offline** with the harness, `scripts/hacks/interpret-tune/tune.py`,
   which sends each render with each candidate prompt to the endpoint and
   scores exact matches and acceptable ones. It reads the endpoint token from
   `~/.tok/llm` (`SCAN_LLM_KEY` overrides) and warns without one: a refused
   call scores as a miss, so an unauthenticated sweep looks like a model that
   got everything wrong.
4. **Repeat each arm three times.** Batched inference is not bit-reproducible
   even at temperature 0, and a borderline file can flip between runs. Treat a
   one-sample move as noise.
5. **Accept on the clean sets first.** A change that buys detections with
   false positives on the clean sets is a regression, since every LLM-hostile
   read on a clean file is a review flag. Keep the hedge sentences of the
   system prompt word for word unless the clean sets say otherwise; shorter
   paraphrases have consistently bought detection with false positives.
6. **Confirm end to end** with `atomscan --llm` on the same corpus, old build
   against new, with `SCAN_NO_ANALYSIS_CACHE=1`.

Verdicts are cached by system prompt, so a prompt change never replays old
verdicts, and `SCAN_LLM_SYSTEM_PROMPT_FILE` runs a candidate prompt without a
rebuild. Shipping it means editing `interpret::SYSTEM_PROMPT`.

To re-measure `DEFAULT_LLM_MIN_PROB` or `DEFAULT_LLM_BENIGN_NOTABLE_CAP`, use
the same sets and the `LLM gate` log line, which names every admission that
fired for each sample: pick the floor where malicious admissions stop rising
while benign ones keep climbing, and the cap that keeps the calls that changed
a verdict while dropping the benign ones that changed nothing.
