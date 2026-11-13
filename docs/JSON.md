# Atomdrift Scan JSON

The CLI writes JSON in two shapes, one record per scanned file:

| Format | What each record is |
| --- | --- |
| `--format json` | a decision: the object `/v1/analyze` answers with |
| `--format json-full` | the full result: the version 8 body scan uploads to hopper |

The server answers the same two: `/v1/analyze` returns the decision, and
`/v1/analyze?full=1` returns the full result.

## `--format json`: the decision

One JSON object per line, per file, whatever the verdict:

```json
{"decision":"block","purl":null,"path":"./pkg/evil.tgz","sha256":"ab12…","severity":"hostile",
 "fires_at":12,"reason":null,"findings":[{"id":"objectives/c2/backdoor::a","crit":5,"file":"package/install.js"}],
 "engine_version":"3.0.0","analyzed_at":"2026-11-12T15:00:00Z"}
```

| Field | Meaning |
| --- | --- |
| `decision` | `allow`, `block`, `unanalyzed` or `unavailable` |
| `severity` | `benign`, `suspicious` or `hostile`; `null` when there is no verdict |
| `fires_at` | the tightest false-positive level at which the file grades hostile (see below) |
| `reason` | the LLM's one-sentence rationale, when it gave one |
| `findings` | at most three findings at crit 4 or 5, worst first |
| `sha256`, `purl` | what the file is |
| `engine_version`, `analyzed_at` | which scan build produced it, and when |

The fields are the server's ([SERVER_API.md](SERVER_API.md)), with those the
CLI adds: `path`, the file it was asked about; `source`, what answered for a
file the CLI did not analyze (`bloom` or `uplink`); and for `atomscan ps`,
`pids` and `deleted`.

The CLI decides by the class it graded the file at, which follows its own
`-l` exactly as the terminal does, so the JSON and the terminal agree. The
server reaches the same answer from `fires_at` and the caller's
`false_positive_budget`.

Files the CLI did not analyze still get a line:

| Case | `decision` | `source` | `reason` |
| --- | --- | --- | --- |
| known good to the bloom filters | `allow`, severity `benign` | `bloom` | `known good` |
| in neither filter set, fast mode | `unanalyzed` | `bloom` | why it was not analyzed |
| taken from the corpus (`--uplink`) | the corpus verdict at the CLI's level | `uplink` | the corpus's |
| analysis failed | `unavailable` | | the error |

A failed file still makes the process exit 3.

### `fires_at`

`fires_at` is a property of the file and the model, not of anyone's
setting: the lowest pool-adjusted false-positive level at which the model
flags the file as hostile. Levels are defined in
[INTEGRATION.md](INTEGRATION.md#set-the-false-positive-level).

- `fires_at` in the calibrated grid (`0..=25000` today; consumers should
  tolerate `50000`) → the lowest level at which the file fires. Lower
  means it fires under a stricter calibrated allowance: `lvl=2` fires at
  one FP plus an estimated additional 1/100M; `lvl=500` fires at one FP
  plus an estimated additional 499/100M. These are targets, not guarantees
  of future observed rates.
- `lvl == 25001` or `lvl == 25002` → off-grid trait-floor override
  markers (`grid_max + 1/2`) where Atomdrift Scan manually raised a model-clean
  result to suspicious because cleave found confident severe traits.
- `lvl == -1` (sentinel) → the file fires at **no** grid level. Nothing
  short of disabling the model would flag it — it is clean.
- `lvl == null` → a library caller supplied manual probability cutoffs, so
  no level table applies.

A file is **hostile** at level `N` when `fires_at <= N`, **suspicious** when
`fires_at <= min(grid_max, max(500, 100 × N))`, and **benign** otherwise.
`N` defaults to the model bundle's `default_severity_level` (`25` on current
bundles), so the default suspicious ceiling is L2500.

## `--format json-full`: the full result

The body scan uploads to hopper, written to stdout: JSON lines, a header,
one line per file, then one line per report. Several files scanned in one
run write one body each, each starting with its header. The format is
specified in `JSON_PLAN.md` at the repository root; in short:

- **Header:** `{"v":8,"scan":{"engine","ruleset","level","at"},"files":N}`.
  An upload also names its `worker`, which hopper requires, and postdoc's
  adds `ms`, how long the analysis took.
- **File lines:** the scanned artifact first, then up to 256 of its members,
  the gravest first, and the containers on the way to them. Each carries
  `sha256`, `size`, `type`, its `verdict`, one object per juror (`ml`,
  `floor`, and on the artifact `llm`), `ident` when the file claims an
  identity, `contents` for a container, `refs` on the artifact, and `report`,
  the sha256 of its report line.
- **Report lines:** each file's cleave findings and facts, with nothing that
  says where the file sat; a source in another file is cited by its sha256.

A file with no analysis has no full result: a bloom, Uplink or failed file
gets its decision line instead, as in `--format json`. A reader tells the two
apart by the header's `"v":8` against a decision's `decision` key.

`--all-members` (hidden, for coverage tools) gives every member cleave listed
a line, analyzed or not, rather than the 256 an upload keeps: a member with no
`verdict` was listed and not analyzed. It needs `--format json-full`, and it
changes only what is written to stdout: an upload with `--hopper` is the same
with or without it.

## Render saved reports locally

**Experimental:** the `render` command and its candidate JSON formats may change.

`atomscan render report.json --format interpret` reads a saved cleave report
(or the `raw` report of a JSON line saved by an older scan) and uses the live primary-context budget and
annotation rewrite. It does not load models, analyze the original files, fetch
references, or update rules. `--format tiny` and `terminal` render cleave's
corresponding context views; they do not synthesize an ML verdict. Use `-` as
the input path to read one JSON document from stdin.

With `--llm` (and the usual `--llm-model`/`SCAN_LLM*` settings),
`render` sends the interpret render to the LLM with the live grading prompt and
prints `{"grade","model","reason"}` instead: the grade alone, with no ML
admission gate or blend, for checking that a saved report grades as its live
scan did. It queues at background priority.

For a byte-for-byte comparison, use:

```sh
atomscan render before.json --format interpret > before.txt
atomscan render after.json --format interpret > after.txt
diff -u before.txt after.txt
```

Rendered output goes to stdout, and warnings go to stderr. Run both inputs with the
same renderer build.

The reader accepts compact v9 or v8 reports and cleave's raw v3 snapshots
(`cleave --format snapshot SAMPLE -o before.json`); unknown schemas fail.
Reports store no context annotations: the renderer derives them from each
file's findings exactly as a live analysis does, so a saved report renders as
its analysis did.

The initial interpretation replay supports artifacts and their archive members.
Fetched subjects and registry sidecars additionally require saved dependency
verdicts and registry provenance, which the compact report does not fully carry;
those inputs are rejected rather than rendered with missing subjects.
