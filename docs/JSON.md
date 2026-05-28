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

`prob` instead of `probability` saves seven bytes per record. `class`
instead of `classification` saves nine. `m`, `why`, `skip`, `fs` are
the same trade. Over a billion analyses the storage difference is not
academic, and the names map cleanly to obvious things. The tables
below are the key.

## `ml` — `MlSection`

`src/scan.rs`.

| JSON          | Rust                      | Type                   | Meaning                                              |
| ------------- | ------------------------- | ---------------------- | ---------------------------------------------------- |
| `v`           | `v`                       | string                 | Envelope schema version. Currently `"5"`.            |
| `class`       | `classification`          | u8 (0/1/2)             | Final classification.                                |
| `prob`        | `probability`             | f32 in `[0, 1]`        | Probability the verdict was decided on.              |
| `threshold`   | `threshold`               | f32 in `[0, 1]`        | Cutoff defining the verdict band. See **The invariant** below. |
| `level`       | `level`                   | u8 (1..=9) or null     | Severity level that selected the threshold, or `null` when manual thresholds were used. |
| `models`      | `model_scores`            | array of `RouteScore`  | Per-route ensemble scores. Omitted if empty.         |
| `skip`        | `skipped_models`          | array of `SkippedRoute`| Routes that were applicable but unused.              |
| `version`     | `version`                 | string                 | Model version: spec, ABI, hash prefix.               |
| `analyzed_at` | `analyzed_at`             | string (RFC 3339, UTC) | Completion timestamp.                                |
| `fs`          | `fs`                      | array of `EmbeddedFile`| Top findings for display; archive members.           |
| `pids`        | `pids`                    | array of u32           | Running PIDs. Present only on process scans.         |
| `deleted`     | `deleted`                 | bool                   | Whether the on-disk binary was deleted (process scan). |

Classification ints: `0 = benign`, `1 = suspicious`, `2 = hostile`.

## The invariant

`class = bucket(prob, threshold)`:

- `class == Hostile` (2) iff `prob >= threshold` and `threshold` is the hostile cutoff that fired.
- `class == Suspicious` (1) iff `prob >= threshold` and `threshold` is the suspicious cutoff that fired.
- `class == Benign` (0) iff `prob < threshold`; `threshold` is the suspicious cutoff that `prob` did not reach.

`prob` is the value the decision was made on — the firing route's
probability for OR-rule policies, the blend's sigmoid output for
learned-blend policies, or the elevating embedded file's probability
when an archive member outranked its parent. `threshold` is whichever
cutoff that value was compared against. The two are always
self-consistent for the verdict.

`level` answers "what false-positive level produced this threshold?"
For level-driven thresholds (the default, or `-1`..`-9` / `--loose` /
`--paranoid`), `level` is `1..=9`. When `--threshold-suspicious` or
`--threshold-hostile` was supplied, `level` is `null` — the threshold
came from the operator, not from the model's level table.

Each `ml.fs[]` entry carries its own `prob`/`class`/`threshold`, so the
invariant holds per-row too.

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

    {
      "ml": {
        "v": "5",
        "class": 2,
        "prob": 0.998,
        "threshold": 0.95,
        "level": 3,
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
          {
            "path": "payload.bin",
            "file_type": "elf",
            "classification": 2,
            "probability": 0.998,
            "threshold": 0.95,
            "top_findings": [
              { "id": "objectives/evasion/process::injection",
                "crit": 5,
                "desc": "writes to another process's address space" }
            ]
          }
        ]
      },
      "raw": { "...": "full cleave AnalysisReport" }
    }

The verdict is hostile because `prob` (0.998) crossed the hostile
cutoff (0.95) at severity level 3. The `az/native` specialist route
drove the decision; the general `az` route alone would have been
suspicious.

## Migration from v=4

`v=4` carried `thresholds: [suspicious, hostile]` (always the model's
global pair) plus optional `oclass`/`oprob` from a finding-based
upgrade heuristic. Both were misleading: the global thresholds did not
necessarily produce `class` (per-filetype policies and learned blends
have their own cutoffs), and the upgrade heuristic mutated the model's
output without surfacing which cutoff applied.

`v=5` reports the actual deciding `(prob, threshold)` pair plus the
`level` that selected it. The upgrade heuristic is gone; `oclass` and
`oprob` no longer appear. The `thresholds` pair is replaced by a
single `threshold` that is consistent with `class` and `prob` by
construction.
