# Integrating with Atomdrift Scan

Atomdrift Scan reads a file and calls it **benign**, **suspicious**, or
**hostile**. It works offline and gives the same answer every time. It makes
no network calls unless you ask it to.

This guide is for wiring it into a pipeline: a registry gating uploads, a CI
check, an ingestion worker.

## Start in one minute

```bash
# install (Linux/macOS via Homebrew)
brew install atomdrift-project/tap/scan

# or build from source — needs Rust 1.96+
git clone https://github.com/atomdrift-project/scan.git && cd scan && make install

# scan a file, a directory, or an archive (archives unpack automatically)
atomscan suspicious_package.tar.gz
atomscan ./incoming/
```

A clean run prints a one-line summary and exits `0`. A finding is printed
inline and changes the exit code. For most integrations, that exit code is
the whole interface.

To refresh the models before a scan, add `-u` (a failed update is
non-fatal); the models also auto-refresh when the local ruleset is over
24h stale. Disable that automatic refresh with `--no-update` (or the
`SCAN_NO_UPDATE` environment variable), and silence the once-a-day update
notice with `SCAN_NO_UPDATE_CHECK`.

## Choose a path

All four paths return the **same JSON envelope** ([JSON.md](JSON.md)). Start
with the CLI; move to a server or worker when volume demands it. Nothing
about how you parse results changes.

| Path         | Use when                                                  | Cost                                            | Reference                          |
| ------------ | -------------------------------------------------------- | ----------------------------------------------- | ---------------------------------- |
| **CLI**      | One-shot or batch, up to ~5 scans/min.                   | Reloads the model on every run.                 | `atomscan --help`                 |
| **HTTP server** | Sustained traffic past ~5 scans/min.                 | You run a long-lived process.                   | [SERVER_API.md](SERVER_API.md)     |
| **Worker**   | Distributed ingestion: workers pull jobs from a [hopper](https://github.com/atomdrift-project/hopper) queue. | You run a hopper for them to poll. | [WORKERS.md](WORKERS.md)           |
| **Rust library** | You are already in Rust and want in-process calls. | More setup, weaker API stability. See [Library](#library). | source — `classify_file`, `classify_bytes` |

A registry gating uploads on ingest should start with the **CLI** and grow
into the **Worker**.

## Gate on the exit code

The CLI turns a verdict into an exit code, so a gate is one line:

```bash
atomscan ./package/ || echo "flagged — block this upload"
```

| Code | Meaning                                          |
| ---- | ------------------------------------------------ |
| `0`  | All clean.                                        |
| `1`  | At least one **hostile** file.                    |
| `2`  | At least one **suspicious** file, none hostile.   |
| `3`  | An error occurred, nothing hostile or suspicious. |
| `4`  | The rule set was incomplete — see below.          |

The most severe verdict wins: one hostile file in a tree returns `1` no
matter what else is in it. A typical policy blocks on `1`, sends `2` to a
human, and pages on `3`.

`4` means the YARA engine ran degraded: rule sources failed to compile, or
scanning was disabled mid-run after an engine panic. The scan finished, but
with fewer rules than the trait set defines, so a clean or merely suspicious
result proves nothing — the detection that would have fired may simply not
have been loaded. It outranks `2` and `3` for that reason, and `1` outranks
it in turn: a hostile verdict still stands, because missing rules can only
cost you detections, never invent one.

Treat `4` as "run it again", not as a verdict. Gate on it the way you gate on
a failed build rather than a flagged artifact, and do not let it fall into a
`>= 2` bucket that a policy reads as "suspicious, needs a human" — nobody can
review a scan that did not fully happen.

## Set the false-positive level

This is the dial you tune the gate with. You do not set "sensitivity" in the
abstract — you set a **false-positive level**: how many benign files you will
tolerate being flagged, per 100 million, calibrated for each file type.

```bash
atomscan -l 0 ./pkg/       # zero false positives — strictest, fewest alerts
atomscan ./pkg/            # default: level 50 (~0.5 flagged per million benign)
atomscan -l 5000 ./pkg/    # any calibrated point on the 0–25000 grid
atomscan -l 25000 ./pkg/   # most sensitive — catches more, cries wolf more
```

Choose by where a mistake hurts. A hard upload gate wants a strict level
(`-l 0`, `-l 1`) so it almost never blocks a legitimate package. A triage
queue feeding human reviewers can run a high level like `-l 25000` and let
people sort it out.

If you would rather set the raw probability cutoffs yourself,
`--threshold-hostile <0.0–1.0>` and `--threshold-suspicious <0.0–1.0>` do
that directly. They replace `-l`; you use one approach or the other, not
both.

Whichever you pick is fixed when the process starts. A running server picks
up a new setting with `POST /_/reload` — no restart.

## Output formats

`-f` is a global flag, so it comes before the path:

```bash
atomscan -f json ./pkg/    # NDJSON, one envelope per file, including archive members
atomscan -f tiny ./pkg/    # compact text, built to feed a local LLM
atomscan ./pkg/            # default: human-readable terminal output
```

JSON gives you one line per file whatever the verdict. Read `ml.lvl` for the
result (`-1` means not flagged) and `ml.prob` for the score. The field names
are short on purpose; [JSON.md](JSON.md) is the key. The envelope is stable
within a major version.

## A second opinion: `--interpret`

`--interpret` sends non-trivial samples to a **local** LLM and blends its
read into the verdict (kept in the `llm` JSON section). It talks to an
OpenAI-compatible endpoint at `http://localhost:8000/v1` by default. Nothing
leaves your network.

```bash
atomscan --interpret ./pkg/
```

No model name is baked in. Unless you pin one with `--llm-model`, atomscan
asks the endpoint what it serves (`GET /v1/models`) and takes the largest
model listed — so whatever you started the server with is what gets used. If
the endpoint lists nothing, the scan stops with an error rather than guessing.

We recommend `Qwen/Qwen3.8-27B` under vLLM; a Qwen3-class model works well
generally, and ~9B is enough if that is what fits.

Point it elsewhere with `--llm`, `--llm-model`, `--llm-key`. `--llm openrouter`
uses `https://openrouter.ai/api/v1`; the key is `--llm-key`, `SCAN_LLM_KEY`, or
`~/.tok/openrouter`, and `--llm-model` is required. Control which samples
qualify with `--llm-min-level`: a sample is sent when ML fires at or below that
FP level — the model's own per-route cutoff — or when cleave surfaced a
suspicious/hostile finding, whichever comes first. It defaults to the model's
grid ceiling, so ML admits anything it flagged at any level; pass a lower `N` to
tighten.

## Following references: `--follow` (experimental)

`--follow` retrieves references discovered inside the requested artifact,
analyzes them, and folds their results back into its verdict. Choose declared
manifest/lockfile dependencies (`dependencies`), packages and URLs named by
install/download commands (`references`), or third-party CI actions
(`ci-actions`, which also implies dependencies). A bare `--follow` and the
default select `dependencies,references`; `all` also includes CI actions. With
`--follow-depth` it follows hops: a script that pulls a loader that runs a
`curl | bash` dropper is caught as one chain.

```bash
atomscan --follow=dependencies ./pkg/             # manifests and lockfiles only
atomscan --follow --follow-depth 3 ./pkg/          # dependencies and references
atomscan --follow=all --follow-depth 3 ./pkg/      # include CI actions too
```

This is the only analysis phase that makes additional network requests. Use
`--follow=none` for air-gapped or transparent-proxy gates that already know
which artifacts will be requested. Set the default per
deployment with `SCAN_FOLLOW` and `SCAN_FOLLOW_DEPTH` instead of flags if that
suits you better. Existing `--fetch` and `SCAN_FETCH` configurations continue
to work, as do the old `deps`, `packages`, `urls`, and `ci` value names.

## Library

The in-process Rust API is the most direct path and the least stable. Until
1.0, minor releases may break it, so **pin a commit** of
`github.com/atomdrift-project/scan`. Two things it asks of you: give `main` an
8 MB rayon stack, and call `kill_all_rizin_groups()` at shutdown so
disassembler subprocesses do not leak. It returns the same JSON envelope as
every other path.
