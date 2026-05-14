# Litmus Server API

`litmus serve` is an HTTP daemon that takes a file and returns a
classification. It binds to loopback. It has no authentication. Treat
it as a local service and put a reverse proxy in front of it if
anything else needs to reach it.

For the pull-based worker, see [WORKERS.md](WORKERS.md).
For the response schema, see [JSON.md](JSON.md).

## Running the server

    litmus serve

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
| `LITMUS_MODELS_REPO`   | Model repository URL.                                        |
| `LITMUS_MODELS_REF`    | Branch or commit to pull.                                    |

The listener binds before the model is loaded. While loading, every
route returns 503 with `{"error":"server starting"}`. Poll `/_/health`
until the status flips to `ok`.

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

`/analyze` and `/analyze-path` also set `X-Total-Ms` on the response.

Errors share a single shape:

    { "error": "string", "detail": "optional chain" }

## Thresholds

Two numbers per model. Classification is:

    prob >= hostile     -> 2 (hostile)
    prob >= suspicious  -> 1 (suspicious)
    otherwise           -> 0 (benign)

The numbers are resolved in this order, each step overriding the
previous one:

1. `evaluation.json` in the model bundle (`src/model.rs:270`).
2. Severity level flags `-1` … `-9` pick a row from the bundle
   (`src/main.rs:160`). Higher levels are stricter.
3. `--threshold-suspicious` and `--threshold-hostile` override
   everything (`src/main.rs:301`).
4. If the bundle carries no thresholds and no flag is passed, the
   fallbacks are `suspicious = 0.65`, `hostile = 0.90`
   (`src/model.rs:425`).

Thresholds are baked in at server start. To change them: stop the
server, or edit the bundle and `POST /_/reload`, or push new models
and `POST /_/update`. There is no per-request override.

The upgrade heuristic can bump a verdict after the threshold compare
when cleave findings clearly disagree with the model. When it bumps,
the pre-bump values appear as `oclass` and `oprob` in the response.
The active thresholds for a request are always reported in the
`thresholds` field, so the response is self-describing.

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

    $ litmus serve --bind 127.0.0.1:49999 --workers 4
    $ curl -s http://127.0.0.1:49999/_/health | jq -r .status
    ok
    $ curl -s -F file=@/bin/ls http://127.0.0.1:49999/analyze \
        | jq '.ml | {class, prob, version}'
    {
      "class": 0,
      "prob": 0.01,
      "version": "spec=4 abi=1 hash=8f3a91"
    }
