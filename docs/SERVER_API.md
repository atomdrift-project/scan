# Atomdrift Scan Server API

`atomscan serve` is an HTTP daemon that takes a file and returns a
classification. It binds to loopback. It has no authentication. Treat
it as a local service and put a reverse proxy in front of it if
anything else needs to reach it.

For the pull-based worker, see [WORKERS.md](WORKERS.md).
For the response schema, see [JSON.md](JSON.md).

## Running the server

    atomscan serve

The defaults are deliberate. Override them only when you have a reason.

| Flag             | Default            | Meaning                                                    |
| ---------------- | ------------------ | ---------------------------------------------------------- |
| `--bind`         | `127.0.0.1:49999`  | Listen address.                                            |
| `--workers`      | `max(1, ncpu / 2)` | Hard cap on concurrent analyses. Excess requests get 503.  |
| `--max-size-mb`  | `100`              | Per-request upload limit.                                  |
| `--max-rss-gb`   | `0` (auto)         | RSS ceiling. `0` reads the cgroup signal. `-1` disables.   |
| `--allowed-dirs` | none               | Comma-separated roots permitted by `/analyze-path`.        |
| `--extract-dir`  | none               | Where cleave unpacks archive members.                      |
| `--allow-cidr`   | none               | Extra CIDR networks allowed beyond loopback.               |
| `--traits-dir`   | none               | Writable cleave traits directory (sets env on launch).     |

Environment variables read at startup:

| Variable               | Effect                                                       |
| ---------------------- | ------------------------------------------------------------ |
| `CLEAVE_TRAITS_DIR`    | Traits directory. `--traits-dir` overrides.                  |
| `CLEAVE_RAYON_THREADS` | Override rayon pool size. Default is system parallelism.     |
| `SCAN_MODELS_REPO`   | Model repository URL.                                        |

The listener binds before the model is loaded. While loading, every
route returns 503 with `{"error":"server starting"}`. Poll `/_/health`
until the status flips to `ok`.

### Deploy

`make deploy` (alias of `make deploy-server`) installs a long-lived
`atomscan serve`:

- **FreeBSD.** Bastille build jail + run jail, rc.d service
  (`scripts/server/rollout-bastille.sh`).
- **Linux (systemd).** Native host install, unit `scan.service`
  (`scripts/server/server-linux.sh`). Same shape as
  `make deploy-worker` on Linux: unprivileged `scan` user, `MemoryMax=`,
  traits under `/var/lib/atomdrift/scan`.

Linux overrides (passed through the environment): `BIND=` (default
`0.0.0.0:49999`, matching the FreeBSD jail), `ALLOW_CIDR=` (default
`10.0.0.0/8`; set empty to omit), `LLM=`, `WORKERS=`, `MEMORY_MAX=`.
`make uninstall-server` tears the unit down.

## Endpoints

### `POST /analyze`

Multipart upload. One part named `file`. The filename is sanitised
(`[A-Za-z0-9_.-]`, `..` collapsed, truncated to 63 bytes) and copied to
a private temp directory so cleave sees a plausible extension.

    curl -s -F file=@/bin/ls http://127.0.0.1:49999/analyze | jq .ml

Returns 200 with the [response envelope](JSON.md). 413 if the
body exceeds `--max-size-mb`, 415 for unsupported types, 422 for
truncated or malformed input, 503 if the server is starting or
saturated, 504 if analysis exceeds the watchdog deadline.

### `POST /analyze-path`

JSON body: `{"path": "/absolute/path"}`. Loopback only, always. The
path is canonicalised before it is compared against `--allowed-dirs`,
so symlinks cannot escape. Without `--allowed-dirs`, every request
returns 403.

    curl -s -H 'content-type: application/json' \
      -d '{"path":"/usr/bin/ls"}' \
      http://127.0.0.1:49999/analyze-path | jq .ml

Same response envelope as `/analyze`.

### `POST /analyze-purl`

JSON body: `{"purl": "pkg:npm/left-pad@1.3.0"}`. The `pkg:` scheme is
optional (`npm/left-pad@1.3.0` is accepted). Scan resolves the package,
looks up registry provenance itself, and returns the same envelope as
`/analyze`. Takes an analyze slot. 400 if the argument is not a PURL.

    curl -s -H 'content-type: application/json' \
      -d '{"purl":"pkg:npm/left-pad@1.3.0"}' \
      http://127.0.0.1:49999/analyze-purl | jq .ml

Beamline backends should start the server with `--fetch --interpret
--analysis-timeout 1800` so dependency follow and LLM interpretation
match a live `atomscan purl` run.

### `GET /_/bloom`

Known-good / known-bad membership. Does **not** take an analyze slot
and answers while models are still loading. Provide exactly one key:

    GET /_/bloom?sha256=<64hex>
    GET /_/bloom?purl=<url-encoded>

    { "decision": "skip" | "known-bad" | "conflicted" | "unknown" }

`skip` is known-good and not revoked by the bad filter. Missing filters
fail closed (`unknown`). The server memos the last 4096 SHA-256 and 4096
PURL decisions in process (mutex + LRU). `Cache-Control: public,
max-age=3600`. 400 if both keys, neither key, or a malformed sha256.

### `GET /_/health`

Liveness and load. 200 when ready, 503 while loading or failed.

    {
      "status": "ok",
      "rss_mb": 312,
      "max_rss_mb": 16384,
      "active_tasks": 1,
      "stuck_orphans": 0,
      "long_running_tasks": [],
      "max_concurrent_tasks": 4,
      "load": 0.25,
      "load_avg": 0.42,
      "uptime_secs": 91,
      "rayon_threads": 8
    }

`status` is one of `ok`, `starting`, `failed`, `degraded`, `saturated`.

### `GET /_/info`

Static facts: version, slot count, CPU count, upload and RSS limits,
total memory, the model and traits commit hashes.

### `POST /_/reload`

Reread the model bundle from disk and hot-swap it atomically. Returns
elapsed time. Use this after editing `evaluation.json` to change
thresholds without restarting.

### `POST /_/update`

`git pull` the models and traits repositories, then reload. Returns
which side changed and the new commits.

### `GET /_/memory`, `GET /_/requests`, `GET /_/threads`

Diagnostics. `/_/memory` exposes jemalloc counters and rayon pool
size. `/_/requests` lists in-flight analyses with elapsed time and
phase. `/_/threads` reports per-thread state (Linux: `wchan`, context
switch counts; FreeBSD: rayon thread count; other platforms: error).

## Status codes

| Code | Cause                                                                   |
| ---- | ----------------------------------------------------------------------- |
| 400  | Malformed request body.                                                 |
| 403  | `/analyze-path` outside `--allowed-dirs`, or non-loopback peer.         |
| 413  | Body exceeds `--max-size-mb`.                                           |
| 415  | Unsupported file, archive, or compression format.                       |
| 422  | Truncated, encrypted-without-password, depth or count limit exceeded.   |
| 500  | Internal error.                                                         |
| 503  | Starting, failed, overloaded, or at capacity.                           |
| 504  | Analysis exceeded the per-request watchdog.                             |

`/analyze`, `/analyze-purl`, and `/analyze-path` also set `X-Total-Ms` on the response.

Errors share a single shape:

    { "error": "string", "detail": "optional chain" }

## Thresholds

The v7 envelope no longer wire-encodes a verdict class. `ml.lvl` is the
lowest false-positive budget (FP per 100M benigns) at which the model
flags the file as hostile — a property of the file and model, not of
the deploy level:

    lvl in grid        -> lowest level at which the file fires (lower = more hostile)
    lvl == -1          -> fires at no grid level (clean)
    lvl == null        -> manual --threshold-* mode (no level table)

The calibrated grid currently tops out at L25000, and consumers should
tolerate a future L50000. Atomdrift Scan also reserves off-grid `grid_max + 1`
and `grid_max + 2` markers for trait-floor overrides where the model was
clean but confident severe cleave traits manually raised the result to
suspicious; with today's grid those are `25001` and `25002`.

`ml.conf` is the same level rendered as a pessimistic integer confidence
percent for display/export. It is `null` when `ml.lvl` is `null`, `0` for
the benign `-1` sentinel, `100` at L0, `99` at L1, `98` at L2, `95` at
L5, `90` at L50, `29` at L25000, `28`/`27` for `25001`/`25002`, and
`17` at L50000. `ml.prob` remains the raw model score used for the
decision.

Because `lvl` is swept over the full grid independent of `-l`, the whole
`ml` envelope is identical across deploy levels, so a result can be
cached once and shared. The consumer derives the verdict from `lvl` and
the active level `N` (default 50):

    hostile     when lvl <= N                      (default lvl <= 50)
    suspicious  when lvl <= min(grid_max, 4 × N)    (default lvl <= 200)
    benign      otherwise

The L×4 rule gives suspicious a 4× wider FP budget that catches more
"maybe-bad" files while keeping hostile crisp. A file with `lvl = 500` is
benign under the defaults yet still reports `lvl = 500`; raising `-l` is
what reclassifies the same envelope. When `--threshold-*` is supplied
(manual mode), `lvl` is `null` and only hostile/benign verdicts apply.

The verdict mode is resolved as follows:

1. **Level mode (default).** The model's per-level grid
   (`route_policies.json`, falling back to `config.json` `levels[]`)
   drives the per-file `lvl` sweep. The active level `N` sets the verdict
   caps and comes from `-l <N>` / `--level <N>` for any integer in
   `0`-`25000` (`src/main.rs`). The default deploy level is L50
   (= 50 FP/100M = 0.5 FP/M). Higher `N` is more sensitive. Crucially, `N` only
   moves the caps — it does **not** change `lvl` or the serialized envelope.
2. **Manual mode.** `--threshold-hostile` / `--threshold-suspicious`
   bypass the level grid entirely: the verdict comes from those raw
   cutoffs, `ml.lvl` is `null`, and only hostile/benign verdicts are
   possible (suspicious is not derived).
3. If the bundle carries no level grid (e.g. a single-bundle dev model),
   the verdict falls back to the bundle's recommended/fallback thresholds
   and `ml.lvl` is `null`.

The level/grid is baked in at server start. To change models: stop the
server, or edit the bundle and `POST /_/reload`, or push new models and
`POST /_/update`. The active level is fixed per server process; there is
no per-request override.

See [JSON.md](JSON.md) for the full `ml.lvl` encoding and how consumers
derive hostile/suspicious/benign from it.

## Security

The server is built for trusted networks. The defaults reflect that.

- **Bind is loopback.** Do not change `--bind` without thinking about
  what else can now reach it.
- **No authentication. No TLS.** If the server is reachable from
  anywhere but localhost, put a reverse proxy in front of it that does
  both.
- **`/analyze-path` is loopback-only, always.** `--allow-cidr` does
  not widen it. The path is `canonicalize()`d before the
  `--allowed-dirs` prefix check, so symlinks cannot point outside the
  allowed roots.
- **Filenames are sanitised.** Alnum plus `_.-` only. `..` is
  collapsed. The result is truncated to 63 bytes. The original name is
  never used on disk.
- **`--allow-cidr` is a footgun if `--bind` is loopback.** The CIDR
  list cannot match a loopback peer. The server logs a warning at
  startup; read it.
- **Body size is capped** by `--max-size-mb` and enforced during
  streaming, not after.
- **RSS is capped.** New requests get 503 once the process exceeds
  the limit. The auto value reads the cgroup signal; the fallback is
  16 GiB; `-1` disables the check (use only behind `MemoryMax=` or
  equivalent).
- **Concurrency is a hard semaphore.** When `--workers` slots are
  full, new requests get 503 immediately. There is no queue.
- **Static analysis only.** No untrusted code is executed. Cleave
  runs rizin in an isolated process group; if it gets stuck, the
  watchdog kills the group, not just the parent.

## Example session

    $ atomscan serve --bind 127.0.0.1:49999 --workers 4
    $ curl -s http://127.0.0.1:49999/_/health | jq -r .status
    ok
    $ curl -s -F file=@/bin/ls http://127.0.0.1:49999/analyze \
        | jq '.ml | {lvl, prob, version}'
    {
      "lvl": -1,
      "prob": 0.01,
      "version": "spec=4 abi=1 hash=8f3a91"
    }
