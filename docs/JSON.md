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
instead of `classification` saves nine. `oclass`, `oprob`, `m`, `why`,
`skip`, `fs` are the same trade. Over a billion analyses the storage
difference is not academic, and the names map cleanly to obvious
things. The tables below are the key.

## `ml` — `MlSection`

`src/scan.rs:1438`.

| JSON          | Rust                      | Type                   | Meaning                                              |
| ------------- | ------------------------- | ---------------------- | ---------------------------------------------------- |
| `v`           | `v`                       | string                 | Envelope schema version. Currently `"4"`.            |
| `class`       | `classification`          | u8 (0/1/2)             | Final classification.                                |
| `prob`        | `probability`             | f32 in `[0, 1]`        | Final malware probability.                           |
| `oclass`      | `original_classification` | u8                     | Pre-upgrade class. Omitted if the verdict was not bumped. |
| `oprob`       | `original_probability`    | f32                    | Pre-upgrade probability. Omitted if not bumped.      |
| `thresholds`  | `thresholds`              | `[suspicious, hostile]`| Active thresholds for this request.                  |
| `models`      | `model_scores`            | array of `RouteScore`  | Per-route ensemble scores. Omitted if empty.         |
| `skip`        | `skipped_models`          | array of `SkippedRoute`| Routes that were applicable but unused.              |
| `version`     | `version`                 | string                 | Model version: spec, ABI, hash prefix.               |
| `analyzed_at` | `analyzed_at`             | string (RFC 3339, UTC) | Completion timestamp.                                |
| `fs`          | `fs`                      | array of `EmbeddedFile`| Top findings for display; archive members.           |
| `pids`        | `pids`                    | array of u32           | Running PIDs. Present only on process scans.         |
| `deleted`     | `deleted`                 | bool                   | Whether the on-disk binary was deleted (process scan). |

Classification ints: `0 = benign`, `1 = suspicious`, `2 = hostile`.

## `RouteScore`

`src/model.rs:358`.

| JSON    | Rust             | Type   | Meaning                                  |
| ------- | ---------------- | ------ | ---------------------------------------- |
| `m`     | `model`          | string | Route name, e.g. `az`, `az/native`, `az/elf`. |
| `prob`  | `probability`    | f32    | This route's probability.                |
| `class` | `classification` | u8     | This route's classification.             |

## `SkippedRoute`

`src/model.rs:372`.

| JSON  | Rust     | Type   | Meaning                              |
| ----- | -------- | ------ | ------------------------------------ |
| `m`   | `model`  | string | Route name.                          |
| `why` | `reason` | string | Why this route was not scored.       |

## `EmbeddedFile`

`src/scan.rs:578`. One entry per top-level finding for display, plus
one per archive member when the input is an archive.

| JSON              | Type                    | Meaning                                          |
| ----------------- | ----------------------- | ------------------------------------------------ |
| `path`            | string                  | Relative path inside the archive, after `!!`.    |
| `file_type`       | string                  | Detected type.                                   |
| `classification`  | u8                      | Per-member classification.                       |
| `probability`     | f32                     | Per-member probability.                          |
| `model_scores`    | array of `RouteScore`   | Omitted if empty.                                |
| `skipped_models`  | array of `SkippedRoute` | Omitted if empty.                                |
| `formula`         | string                  | Molecular formula. Omitted if empty.             |
| `top_findings`    | array of `TopFinding`   | Behaviours driving the classification.           |

## `TopFinding`

`src/scan.rs:557`.

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
        "v": "4",
        "class": 2,
        "prob": 0.97,
        "oclass": 1,
        "oprob": 0.71,
        "thresholds": [0.65, 0.90],
        "models": [
          { "m": "az/native", "prob": 0.97, "class": 2 },
          { "m": "az",        "prob": 0.71, "class": 1 }
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
            "probability": 0.97,
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

The verdict was upgraded from suspicious (0.71) to hostile (0.97)
because cleave found a process-injection trait. `oclass` and `oprob`
preserve the model's original opinion so downstream systems can audit
the heuristic.
