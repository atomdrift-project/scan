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

`prob` instead of `probability` saves seven bytes per record. `lvl`
instead of spelling out the level/verdict is the same trade. `rte`,
`why`, `skip`, `files` follow the same rule. Over a billion analyses the
storage difference is not academic, and the names map cleanly to
obvious things. The tables below are the key.

## `ml` — `MlSection`

`src/scan.rs`.

| JSON          | Rust                      | Type                   | Meaning                                              |
| ------------- | ------------------------- | ---------------------- | ---------------------------------------------------- |
| `v`           | `v`                       | string                 | Envelope schema version. Currently `"7"`.            |
| `prob`        | `probability`             | f32 in `[0, 1]`        | Probability the verdict was decided on.              |
| `lvl`         | `l`                       | i32 or null            | Lowest-false-positive-level marker. See **The verdict encoding** below. |
| `conf`        | `conf`                    | u8 percent or null     | Pessimistic display/export confidence derived from `lvl`; `null` when no level table applies. |
| `mods`        | `model_scores`            | array of `RouteScore`  | Per-route ensemble scores. Omitted if empty.         |
| `skip`        | `skipped_models`          | array of `SkippedRoute`| Routes that were applicable but unused.              |
| `version`     | `version`                 | string                 | Model version: spec, ABI, hash prefix.               |
| `analyzed_at` | `analyzed_at`             | string (RFC 3339, UTC) | Completion timestamp.                                |
| `files`       | `fs`                      | array of `EmbeddedFile`| Per-file model summaries, including archive members. |
| `pids`        | `pids`                    | array of u32           | Running PIDs. Present only on process scans.         |
| `deleted`     | `deleted`                 | bool                   | Whether the on-disk binary was deleted (process scan). |

## The verdict encoding

The envelope carries no `class` or `threshold`. Instead it reports a
single number — `lvl` — that is a **property of the file and the model,
not of your deploy setting**:

> `lvl` is the lowest false-positive budget — in false positives per 100
> million benign files — at which the model flags this file as hostile.

A consumer reads it as:

- `lvl` in the calibrated grid (`0..=25000` today; consumers should
  tolerate `50000`) → the lowest level at which the file fires. Lower
  means more obviously hostile: `l=2` fires even under an extremely
  strict 2-FP-per-100M budget, while `lvl=500` is only caught once you
  tolerate 500 (`lvl=50` == 0.5 FP/M, `lvl=1000` == 10 FP/M).
- `lvl == 25001` or `lvl == 25002` → off-grid trait-floor override
  markers (`grid_max + 1/2`) where litmus manually raised a model-clean
  result to suspicious because cleave found confident severe traits.
- `lvl == -1` (sentinel) → the file fires at **no** grid level. Nothing
  short of disabling the model would flag it — it is clean.
- `lvl == null` → manual `--threshold-hostile` / `--threshold-suspicious`
  were supplied, so no level table applies.

`conf` is the same `lvl` rendered as a deliberately pessimistic integer
percentage for humans and APIs that need a confidence-style field. It
is not a posterior probability and it does not replace `prob`. The
current anchors are:

| `lvl` | `conf` |
| --- | ------ |
| `0` | `100` |
| `1` | `99` |
| `2` | `98` |
| `3` | `97` |
| `4` | `96` |
| `5` | `95` |
| `10` | `94` |
| `20` | `93` |
| `30` | `92` |
| `40` | `91` |
| `50` | `90` |
| `60` | `89` |
| `70` | `88` |
| `80` | `87` |
| `90` | `86` |
| `100` | `85` |
| `200` | `82` |
| `300` | `80` |
| `500` | `78` |
| `1000` | `75` |
| `2000` | `66` |
| `5000` | `54` |
| `7500` | `49` |
| `10000` | `45` |
| `15000` | `38` |
| `20000` | `33` |
| `25000` | `29` |
| `25001` | `28` |
| `25002` | `27` |
| `50000` | `17` |

Future `50001`/`50002` override markers are reserved as `16`/`15`.
Intermediate non-grid levels round down to the next-lower confidence
bucket.

**`lvl` does not depend on `-l`.** It is computed by sweeping the full
level grid regardless of the deploy level, so the entire `ml` envelope
(including `prob` and `mods[]`) is byte-identical no matter what
`-l` the caller used. That is deliberate: a result can be **cached once
and shared across every deploy level**.

### Deriving the verdict

The hostile/suspicious/benign label is *not* stored — the consumer
derives it from `lvl` and the active level `N` (default `50`):

- **hostile** when `lvl <= N` (default: `lvl <= 50`),
- **suspicious** when `lvl <= min(grid_max, 4 × N)` (default: `lvl <= 200`) —
  the L×4 rule gives the suspicious band a 4× wider FP budget that
  catches more "maybe-bad" files,
- **benign** otherwise. Note a file with, say, `lvl = 500` is benign
  under the default caps yet still reports `lvl = 500`; raising `-l` is
  what turns the same envelope into a suspicious or hostile verdict.

The litmus CLI/server applies these caps internally to pick exit codes
and terminal output; downstream consumers reading stored envelopes
apply whichever caps they prefer.

`prob` is the value the firing decision was made on — the firing
route's probability for OR-rule policies, the blend's sigmoid output
for learned-blend policies, or the elevating embedded file's
probability when an archive member outranked its parent. It is raw
model confidence, not a verdict; use `conf` for the level-derived
confidence-style figure.

Each `ml.files[]` entry carries its **own** `prob`, `lvl`, and `conf`: the
root file (`dp=0`) repeats the envelope's, and every archive member
reports the lowest firing level and derived confidence for that
specific member.

## `RouteScore`

`src/model.rs`.

| JSON    | Rust             | Type   | Meaning                                  |
| ------- | ---------------- | ------ | ---------------------------------------- |
| `rte`   | `model`          | string | Route name, e.g. `az`, `az/native`, `az/elf`. |
| `prob`  | `probability`    | f32    | This route's probability.                |
| `cls`   | `classification` | u8     | This route's classification.             |

## `SkippedRoute`

`src/model.rs`.

| JSON  | Rust     | Type   | Meaning                              |
| ----- | -------- | ------ | ------------------------------------ |
| `rte` | `model`  | string | Route name.                          |
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
| `lvl`             | i32 or null             | Per-member lowest-firing-level marker (same encoding as `ml.lvl`). |
| `conf`            | u8 percent or null      | Per-member confidence derived from `lvl`.        |
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
        "v": "7",
        "prob": 0.998,
        "lvl": 3,
        "conf": 97,
        "mods": [
          { "rte": "az/native", "prob": 0.998, "cls": 2 },
          { "rte": "az",        "prob": 0.71,  "cls": 1 }
        ],
        "skip": [
          { "rte": "az/elf", "why": "wrong-format" }
        ],
        "version": "spec=4 abi=1 hash=8f3a91",
        "analyzed_at": "2026-05-14T18:22:01Z",
        "files": [
          { "id": 0, "prob": 0.998, "lvl": 3, "conf": 97 }
        ]
      },
      "raw": { "...": "full cleave AnalysisReport" }
    }

A benign verdict (sentinel `lvl = -1`):

    {
      "ml": {
        "v": "7",
        "prob": 0.04,
        "lvl": -1,
        "conf": 0,
        "version": "spec=4 abi=1 hash=8f3a91",
        "analyzed_at": "2026-05-14T18:22:01Z",
        "files": [
          { "id": 0, "prob": 0.04, "lvl": -1, "conf": 0 }
        ]
      },
      "raw": { "...": "full cleave AnalysisReport" }
    }

The hostile envelope above fired because `prob` (0.998) crossed the
level-3 hostile cutoff. The `az/native` specialist route drove the
decision; the general `az` route alone would have been suspicious.

## Migration from v=6

`v=6` introduced the level/confidence encoding but still used several
one-letter field names. `v=7` keeps the same semantics and renames the
wire keys for readability:

- `ml.l` → `ml.lvl`
- `ml.models` → `ml.mods`
- `ml.fs` → `ml.files`
- `RouteScore.m` / `SkippedRoute.m` → `rte`
- `RouteScore.class` → `cls`

Consumers should accept both v6 and v7 during migration. Producers emit
v7.

## Migration from v=5

`v=5` carried `class` (0/1/2), `threshold` (the firing cutoff), and a
separate `level` (the per-100M-benigns level or `null`). The
invariant — `class = bucket(prob, threshold)` — meant the same fact
was on the wire three times.

`v=6` collapsed all three into `l`; `v=7` uses the clearer name `lvl`:

- benign is `lvl = -1`,
- hostile-with-known-level is `lvl = 0..=grid_max`,
- hostile-with-manual-threshold is `lvl = null`.

The deciding cutoff itself is no longer transmitted. Consumers that
want to derive a Suspicious band do so against their own threshold;
the envelope does not commit to one. Members of `ml.fs[]` follow the
same shape in v6, and `ml.files[]` follows the v7 shape
`{id, prob, lvl, conf}` — instead of the prior
`{id, class, prob, threshold}`.

## Migration from v=4

`v=4` carried `thresholds: [suspicious, hostile]` (always the model's
global pair) plus optional `oclass`/`oprob` from a finding-based
upgrade heuristic. Both were misleading: the global thresholds did not
necessarily produce `class` (per-filetype policies and learned blends
have their own cutoffs), and the upgrade heuristic mutated the model's
output without surfacing which cutoff applied. `v=5` removed the pair
and the heuristic; `v=6` further collapses `class`/`threshold`/`level`
into `l`, renamed to `lvl` in v7 as described above.
