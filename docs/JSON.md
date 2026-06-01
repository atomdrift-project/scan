# Litmus JSON Report

Every analysis — whether returned by [the server](SERVER_API.md), posted
by [a worker](WORKERS.md), or written to disk by the CLI — uses the
same envelope:

    {
      "ml":  { ... curated classification ... },
      "raw": { ... full cleave report ... }
    }

`ml` is litmus. `raw` is the cleave [AnalysisReport](../../cleave/docs/JSON.md);
the cleave repo owns its schema.

## Why the field names are short

Field names inside `ml` are abbreviated on purpose. The envelope is
designed to be stored — in object stores, in row-oriented databases,
in append-only log streams — and replayed later. The cost of every
field name is paid once per record, and there are a lot of records.

`prob` instead of `probability` saves seven bytes per record. `l`
instead of spelling out the level/verdict is the same trade. `m`,
`why`, `skip`, `fs` follow the same rule. Over a billion analyses the
storage difference is not academic, and the names map cleanly to
obvious things. The tables below are the key.

## `ml` — `MlSection`

`src/scan.rs`.

| JSON          | Rust                      | Type                   | Meaning                                              |
| ------------- | ------------------------- | ---------------------- | ---------------------------------------------------- |
| `v`           | `v`                       | string                 | Envelope schema version. Currently `"6"`.            |
| `prob`        | `probability`             | f32 in `[0, 1]`        | Probability the verdict was decided on.              |
| `l`           | `l`                       | i32 or null            | Lowest-false-positive-level marker. See **The verdict encoding** below. |
| `models`      | `model_scores`            | array of `RouteScore`  | Per-route ensemble scores. Omitted if empty.         |
| `skip`        | `skipped_models`          | array of `SkippedRoute`| Routes that were applicable but unused.              |
| `version`     | `version`                 | string                 | Model version: spec, ABI, hash prefix.               |
| `analyzed_at` | `analyzed_at`             | string (RFC 3339, UTC) | Completion timestamp.                                |
| `fs`          | `fs`                      | array of `EmbeddedFile`| Top findings for display; archive members.           |
| `pids`        | `pids`                    | array of u32           | Running PIDs. Present only on process scans.         |
| `deleted`     | `deleted`                 | bool                   | Whether the on-disk binary was deleted (process scan). |

## The verdict encoding

The envelope carries no `class` or `threshold`. Instead it reports a
single number — `l` — that is a **property of the file and the model,
not of your deploy setting**:

> `l` is the lowest false-positive budget — in false positives per 100
> million benign files — at which the model flags this file as hostile.

A consumer reads it as:

- `l` in `0..=1000` → the lowest level at which the file fires. Lower
  means more obviously hostile: `l=2` fires even under an extremely
  strict 2-FP-per-100M budget, while `l=500` is only caught once you
  tolerate 500 (`l=50` ≡ 0.5 FP/M, `l=1000` ≡ 10 FP/M).
- `l == -1` (sentinel) → the file fires at **no** grid level. Nothing
  short of disabling the model would flag it — it is clean.
- `l == null` → manual `--threshold-hostile` / `--threshold-suspicious`
  were supplied, so no level table applies.

**`l` does not depend on `-l`.** It is computed by sweeping the full
level grid regardless of the deploy level, so the entire `ml` envelope
(including `prob` and `models[]`) is byte-identical no matter what
`-l` the caller used. That is deliberate: a result can be **cached once
and shared across every deploy level**.

### Deriving the verdict

The hostile/suspicious/benign label is *not* stored — the consumer
derives it from `l` and the active level `N` (default `50`):

- **hostile** when `l <= N` (default: `l <= 50`),
- **suspicious** when `l <= min(1000, 4 × N)` (default: `l <= 200`) —
  the L×4 rule gives the suspicious band a 4× wider FP budget that
  catches more "maybe-bad" files,
- **benign** otherwise. Note a file with, say, `l = 500` is benign
  under the default caps yet still reports `l = 500`; raising `-l` is
  what turns the same envelope into a suspicious or hostile verdict.

The litmus CLI/server applies these caps internally to pick exit codes
and terminal output; downstream consumers reading stored envelopes
apply whichever caps they prefer.

`prob` is the value the firing decision was made on — the firing
route's probability for OR-rule policies, the blend's sigmoid output
for learned-blend policies, or the elevating embedded file's
probability when an archive member outranked its parent. It is raw
model confidence, not a verdict; for a confidence-style figure, map
`l` onto a scale (e.g. 100% at `l=0` sliding to 10% at `l=1000`).

Each `ml.fs[]` entry carries its **own** `prob` and `l`: the root file
(`dp=0`) repeats the envelope's, and every archive member reports the
lowest firing level for that specific member.

## `RouteScore`

`src/model.rs`.

| JSON    | Rust             | Type   | Meaning                                  |
| ------- | ---------------- | ------ | ---------------------------------------- |
| `m`     | `model`          | string | Route name, e.g. `az`, `az/native`, `az/elf`. |
| `prob`  | `probability`    | f32    | This route's probability.                |
| `class` | `classification` | u8     | This route's classification.             |

## `SkippedRoute`

`src/model.rs`.

| JSON  | Rust     | Type   | Meaning                              |
| ----- | -------- | ------ | ------------------------------------ |
| `m`   | `model`  | string | Route name.                          |
| `why` | `reason` | string | Why this route was not scored.       |

## `EmbeddedFile`

`src/scan.rs`. One entry per top-level finding for display, plus one
per archive member when the input is an archive.

| JSON              | Type                    | Meaning                                          |
| ----------------- | ----------------------- | ------------------------------------------------ |
| `path`            | string                  | Relative path inside the archive, after `!!`.    |
| `file_type`       | string                  | Detected type.                                   |
| `classification`  | u8                      | Per-member classification.                       |
| `probability`     | f32                     | Per-member probability.                          |
| `threshold`       | f32                     | Per-member deciding cutoff.                      |
| `l`               | i32 or null             | Per-member lowest-firing-level marker (same encoding as `ml.l`). |
| `model_scores`    | array of `RouteScore`   | Omitted if empty.                                |
| `skipped_models`  | array of `SkippedRoute` | Omitted if empty.                                |
| `formula`         | string                  | Molecular formula. Omitted if empty.             |
| `top_findings`    | array of `TopFinding`   | Behaviours driving the classification.           |

## `TopFinding`

`src/scan.rs`.

| JSON   | Type   | Meaning                                                 |
| ------ | ------ | ------------------------------------------------------- |
| `id`   | string | Trait id, e.g. `objectives/evasion/process::injection`. |
| `crit` | u32    | Criticality ordinal. 0 = filtered, 5 = hostile.         |
| `desc` | string | Human-readable description.                             |

## Errors

Errors share a single shape regardless of status code:

    { "error": "string", "detail": "optional chain" }

The status code carries the category; see
[SERVER_API.md#status-codes](SERVER_API.md#status-codes).

## A complete example

A hostile verdict produced at level 3:

    {
      "ml": {
        "v": "6",
        "prob": 0.998,
        "l": 3,
        "models": [
          { "m": "az/native", "prob": 0.998, "class": 2 },
          { "m": "az",        "prob": 0.71,  "class": 1 }
        ],
        "skip": [
          { "m": "az/elf", "why": "wrong-format" }
        ],
        "version": "spec=4 abi=1 hash=8f3a91",
        "analyzed_at": "2026-05-14T18:22:01Z",
        "fs": [
          { "id": 0, "prob": 0.998, "l": 3 }
        ]
      },
      "raw": { "...": "full cleave AnalysisReport" }
    }

A benign verdict (sentinel `l = -1`):

    {
      "ml": {
        "v": "6",
        "prob": 0.04,
        "l": -1,
        "version": "spec=4 abi=1 hash=8f3a91",
        "analyzed_at": "2026-05-14T18:22:01Z",
        "fs": [
          { "id": 0, "prob": 0.04, "l": -1 }
        ]
      },
      "raw": { "...": "full cleave AnalysisReport" }
    }

The hostile envelope above fired because `prob` (0.998) crossed the
level-3 hostile cutoff. The `az/native` specialist route drove the
decision; the general `az` route alone would have been suspicious.

## Migration from v=5

`v=5` carried `class` (0/1/2), `threshold` (the firing cutoff), and a
separate `level` (the per-100M-benigns level or `null`). The
invariant — `class = bucket(prob, threshold)` — meant the same fact
was on the wire three times.

`v=6` collapses all three into `l`:

- benign is `l = -1`,
- hostile-with-known-level is `l = 0..=1000`,
- hostile-with-manual-threshold is `l = null`.

The deciding cutoff itself is no longer transmitted. Consumers that
want to derive a Suspicious band do so against their own threshold;
the envelope does not commit to one. Members of `ml.fs[]` follow the
same shape — `{id, prob, l}` — instead of the prior
`{id, class, prob, threshold}`.

## Migration from v=4

`v=4` carried `thresholds: [suspicious, hostile]` (always the model's
global pair) plus optional `oclass`/`oprob` from a finding-based
upgrade heuristic. Both were misleading: the global thresholds did not
necessarily produce `class` (per-filetype policies and learned blends
have their own cutoffs), and the upgrade heuristic mutated the model's
output without surfacing which cutoff applied. `v=5` removed the pair
and the heuristic; `v=6` further collapses `class`/`threshold`/`level`
into `l` as described above.
