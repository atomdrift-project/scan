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
| `l`           | `l`                       | i32 or null            | Verdict-and-level marker. See **The verdict encoding** below. |
| `models`      | `model_scores`            | array of `RouteScore`  | Per-route ensemble scores. Omitted if empty.         |
| `skip`        | `skipped_models`          | array of `SkippedRoute`| Routes that were applicable but unused.              |
| `version`     | `version`                 | string                 | Model version: spec, ABI, hash prefix.               |
| `analyzed_at` | `analyzed_at`             | string (RFC 3339, UTC) | Completion timestamp.                                |
| `fs`          | `fs`                      | array of `EmbeddedFile`| Top findings for display; archive members.           |
| `pids`        | `pids`                    | array of u32           | Running PIDs. Present only on process scans.         |
| `deleted`     | `deleted`                 | bool                   | Whether the on-disk binary was deleted (process scan). |

## The verdict encoding

The envelope no longer carries `class` or `threshold`. Both are
collapsed into a single field — `l` — which a consumer reads as:

- `l == -1` (sentinel) → the file is **benign**, regardless of how
  thresholds were resolved.
- `l` in `0..=1000` → the file is **hostile**, and the integer is the
  per-100M-benigns level that selected the firing threshold
  (so `l=50` ≡ 0.5 FP/M, `l=1000` ≡ 10 FP/M).
- `l == null` → the file is **hostile**, but manual
  `--threshold-hostile` / `--threshold-suspicious` were supplied, so
  no level table applies.

In short: `l == -1` iff benign; anything else (including `null`) iff
hostile.

`prob` is the value the decision was made on — the firing route's
probability for OR-rule policies, the blend's sigmoid output for
learned-blend policies, or the elevating embedded file's probability
when an archive member outranked its parent.

Suspicious is derived consumer-side as a **level-table lookup**: for
an active hostile level `N`, litmus reads the hostile threshold at
level `min(1000, 4 × N)` from the same `levels[]` table and uses it
as the suspicious cutoff (so a deploy at L50 hostile uses L200's
hostile threshold for its suspicious band — a 4× wider FP budget that
catches more "maybe-bad" files). The envelope itself does not surface
the suspicious band: a Suspicious result is encoded the same way as
Hostile — `l` is the resolved level (or `null`), never `-1`. Manual
`--threshold-hostile` skips the derivation entirely; only
hostile/benign verdicts are possible in that mode.

Each `ml.fs[]` entry carries its own `prob` and `l`, following the
same rules.

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
