# Atomdrift Scan Server API

`atomscan serve` is an HTTP daemon that takes a file and returns a
classification. It binds to loopback. Pass `--token-file` to require a
bearer token on every route but `/_/health`; `make deploy` always does.

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
| `--token-file`   | none               | File holding the required bearer token. See below.         |
| `--traits-dir`   | none               | Writable cleave traits directory (sets env on launch).     |
| `--hopper`       | none               | Hopper base URL. Every analyzed result is renewed on its `/api/result`. Needs a hopper token; see below. |

Environment variables read at startup:

| Variable               | Effect                                                       |
| ---------------------- | ------------------------------------------------------------ |
| `CLEAVE_TRAITS_DIR`    | Traits directory. `--traits-dir` overrides.                  |
| `CLEAVE_RAYON_THREADS` | Override rayon pool size. Default is system parallelism.     |
| `SCAN_MODELS_REPO`   | Model repository URL.                                        |

The listener binds before the model is loaded. While loading, every
route returns 503 with `{"error":"server starting"}`. Poll `/_/health`
until the status flips to `ok`.

Models and traits are refreshed once at startup — that is what a restart is
for, and with `--traits-dir` it is also what installs traits into a directory
that does not exist yet. `-u` forces the refresh even when the local copy looks
current; `--no-update` (before the subcommand) skips it. If traits still cannot
be resolved afterwards the server never reports ready: `/_/health` returns 503
with `{"status":"failed","reason":"initialization_failed"}` and the log names
the path, rather than reporting healthy and failing every analysis.

## Authentication

`--token-file PATH` reads a token from the first non-empty line of `PATH`,
stripped of surrounding whitespace — a trailing newline is not part of the
secret — and requires it on every route except `/_/health`:

    curl -H "Authorization: Bearer $(cat ~/.tok/scan)" ...

The examples further down abbreviate that as
`AUTH="Authorization: Bearer $(cat ~/.tok/scan)"`.

The scheme is case-insensitive; the token is compared byte-exactly. A
missing or invalid token gets `401` with `WWW-Authenticate: Bearer` and an
identical body either way, so the endpoint is not an oracle for guesses.

The token itself must be at least 16 bytes and drawn from the character set a
bearer credential is allowed to carry (RFC 6750 `token68`: `A-Z a-z 0-9 - . _
~ + /`, plus trailing `=` padding). Hex, base64, and URL-safe base64 all pass.
This is a sanity check, not a strength policy: a token containing anything
else — a space, a quote, a stray `Bearer ` prefix pasted into the file — cannot
be sent in a header at all, so the server refuses to start and names the
offending character rather than 401ing every request for the rest of its life.

Four properties are deliberate:

- **Loopback is not exempt.** A Cloudflare tunnel runs `cloudflared` on the
  host and dials the service over loopback, so every remote request arrives
  with a loopback peer address. Exempting loopback would exempt the internet.
  For the same reason `--allow-cidr` cannot filter tunnelled traffic, and
  `/analyze-path`'s loopback-only restriction stops meaning "local" — leave
  `--allowed-dirs` empty on a tunnelled host, which makes that route reject
  everything.
- **The token is a file, never an argument or an environment variable.**
  `argv` is world-readable through `ps`, and systemd unit files are
  world-readable in `/etc/systemd/system`. Only the SHA-256 digest is kept in
  memory, so the token cannot surface in a log line or a core file.
- **Missing means fatal.** If `--token-file` is set and the file is missing,
  empty, or unreadable, the server refuses to start. It never falls back to
  serving unauthenticated.
- **Rotation needs a restart.** The token is read once at startup;
  `/_/reload` does not re-read it. A rotated-but-not-restarted server is the
  usual cause of a 401 against a token file that looks correct — the access
  log's `cred_fp` field distinguishes that from a wrong token, see
  [Logging](#logging).

`/_/health` stays open so tunnel and load-balancer probes work without a
credential — but a valid token there upgrades the response, see below.

### Deploy

`make deploy` (alias of `make deploy-server`) installs a long-lived
`atomscan serve`:

- **FreeBSD.** Bastille build jail + run jail, rc.d service
  (`scripts/server/rollout-bastille.sh`).
- **Linux (systemd).** Native host install, unit `scan.service`
  (`scripts/server/server-linux.sh`). Same shape as
  `make deploy-worker` on Linux: unprivileged `scan` user, `MemoryMax=`,
  traits under the deployed state directory (by default
  `/var/lib/atomdrift/scan`; the systemd installer resolves symlinked mounts
  such as `/var/lib/atomdrift` → `/data/atomdrift`).

Both paths install an API token. It is read from `~/.tok/scan` on the
deploying host — generated there on first deploy if absent — and copied into
the service account's own `~/.tok/scan`, which the unit passes as
`--token-file`. Rotate by editing `~/.tok/scan` and redeploying — a changed
token restarts the service, since it is read only at startup. Hand clients
`$(cat ~/.tok/scan)`.

#### Uploading to hopper

`HOPPER=` is **required**. The deploy refuses to install a server without it:

    make deploy HOPPER=https://hopper-host

A server with no `--hopper` answers every analysis and files none of them. The
caller caches the verdict, so the same artifact is never asked for again, and
hopper never receives it. Nothing fails at deploy time and nothing fails at
request time — the loss only surfaces later, as a sample hopper should hold and
does not. Pass `HOPPER=none` to opt out deliberately (a laptop, a CI box); it
is the same shape as `TOKEN_SRC=` for a deliberately unauthenticated server.

That adds `--hopper <url>` to the service. The credential it needs is a second,
unrelated token: `~/.tok/scan` authenticates clients *to this server*,
`~/.tok/hopper` authenticates *this server to hopper*. Without it, hopper
rejects every result renewal with 401 — it requires a bearer token on every
route and does not exempt loopback. See
[WORKERS.md](WORKERS.md#authenticating-to-hopper).

Every `make deploy` copies the deploying user's `~/.tok/hopper` into the
service account's own `~/.tok/hopper` (`HOPPER_TOKEN_FILE=` overrides the
source), whether or not `HOPPER=` is set — so turning renewal on later needs
nothing else in place. The file is inert while `--hopper` is off.

On FreeBSD the URL lands in the jail's `rc.conf` as `scan_hopper`, so it can
also be changed in place — `bastille sysrc <jail> scan_hopper=<url>` plus a
service restart — without a redeploy. Dropping `HOPPER=` from a later deploy is
now refused rather than silently clearing it; `HOPPER=none` clears it
explicitly, so renewal stops on purpose rather than by omission.

Linux overrides (passed through the environment): `BIND=` (default
`127.0.0.1:49999`, on the assumption that a Cloudflare tunnel or another local
proxy provides the ingress; set `0.0.0.0:49999` to listen on every interface),
`TOKEN_SRC=` (default `~/.tok/scan`; set empty to deploy without
authentication), `ALLOW_CIDR=` (default `10.0.0.0/8`; set empty to omit),
`LLM=` / `LLM_URL=` (`local`, `openrouter`, or a base URL), `LLM_MODEL=`
(required for OpenRouter), `WORKERS=`, `MEMORY_MAX=`. `make uninstall-server`
tears the unit down.

`ALLOW_CIDR=` and `TOKEN_SRC=` treat *empty* as a deliberate choice — no CIDR
allow-list, no authentication — so unlike the others they are not declared in
the Makefile, where they would be exported empty on every deploy. Pass them on
the command line when you mean them.

The FreeBSD jail keeps `--bind 0.0.0.0:49999` with `--allow-cidr 10.0.0.0/8`,
since it is reached over the network rather than through a tunnel, and always
requires a token. `HOPPER=` / `HOPPER_TOKEN_FILE=` apply there too; the other
overrides above are Linux-only.

## Endpoints

### `POST /analyze`

Multipart upload. One part named `file`. The filename is sanitised
(`[A-Za-z0-9_.-]`, `..` collapsed, truncated to 63 bytes) and copied to
a private temp directory so cleave sees a plausible extension.

    curl -s -H "$AUTH" -F file=@/bin/ls http://127.0.0.1:49999/analyze | jq .ml

Returns 200 with the [response envelope](JSON.md). 413 if the
body exceeds `--max-size-mb`, 415 for unsupported types, 422 for
truncated or malformed input, 503 if the server is starting or
saturated, 504 if analysis exceeds the watchdog deadline.

### `POST /analyze-path`

JSON body: `{"path": "/absolute/path"}`. Loopback only, always — but see the
tunnel caveat under [Authentication](#authentication): behind a tunnel,
"loopback" includes every remote caller. The path is canonicalised before it
is compared against `--allowed-dirs`, so symlinks cannot escape. Without
`--allowed-dirs`, every request returns 403.

    curl -s -H "$AUTH" -H 'content-type: application/json' \
      -d '{"path":"/usr/bin/ls"}' \
      http://127.0.0.1:49999/analyze-path | jq .ml

Same response envelope as `/analyze`.

### `POST /analyze-purl`

JSON body: `{"purl": "pkg:npm/left-pad@1.3.0"}`. The `pkg:` scheme is
optional (`npm/left-pad@1.3.0` is accepted). Scan resolves the package,
looks up registry provenance itself, and returns the same envelope as
`/analyze`. Takes an analyze slot. 400 if the argument is not a PURL.

    curl -s -H "$AUTH" -H 'content-type: application/json' \
      -d '{"purl":"pkg:npm/left-pad@1.3.0"}' \
      http://127.0.0.1:49999/analyze-purl | jq .ml

Beamline backends should start the server with `--fetch --interpret
--analysis-timeout 1800` so dependency follow and LLM interpretation
match a live `atomscan purl` run.

### `GET /lookup`

What scan already knows about an artifact or a package. Reads stored
verdicts and the bloom filters; **never analyzes**. Takes no analyze
slot and answers while the model is still loading, so a restarting
server keeps serving lookups.

    GET /lookup?sha256=<64hex>
    GET /lookup?purl=<url-encoded>

Exactly one key; both or neither is a 400. On `purl`, the `pkg:` prefix
is optional (`npm/left-pad@1.3.0` works), and the value is canonicalized
the way `/analyze-purl` and `atomscan purl` canonicalize it.

Both identifiers travel as query parameters. A PURL's own grammar
carries `/`, `?` and `#`, so `pkg:npm/x@1?arch=x64` in a path segment
would have everything from the `?` parsed as the URL's query and a
`#subpath` dropped by the client — silently keying on a different
package. Qualifiers are part of the identity the filters key on.

A stored verdict is a 200:

    {
      "sha": "2cf24dba…",
      "purl": "pkg:npm/evil@1.0.0",
      "lvl": 3,
      "eng": "2.8.0-beta.1",
      "why": "Postinstall launches a reverse shell.",
      "hits": [ { "id": "objectives/…", "crit": 5, "file": "lib/install.js",
                  "pkg": "pkg:npm/evil@1.0.0", "desc": "…",
                  "off": 109, "line": 12 } ],
      "bloom": "unknown"
    }

`lvl` is the tightest false-positive budget per 100M benigns at which the
artifact grades hostile; `-1` never fires. Gate on it. `why` is the
interpreter's sentence when `--interpret` ran. Empty fields are omitted.

`hits` carries at most three findings of criticality 3 or worse, worst
first. `off` is the byte offset of the match within `file` and `line` its
1-based source line, read from the context window that recorded the
match; either may be absent, since a binary has no line structure and a
report whose context was trimmed keeps only a coarse evidence span.

Only findings *native* to the file they are reported on become hits. An
archive repeats its members' findings on itself, carrying no path or
offset of its own; those copies are dropped in favour of the member's,
and a cross-file composite — which has no single place to point at — is
dropped with them.

Holding nothing is a 404 — with the filter's opinion still attached, so
one round trip answers both questions:

    { "error": "unknown sample", "bloom": "skip" }

`bloom` is `skip` (known-good, not revoked), `known-bad`, `conflicted`
(in both sets), or `unknown`. It rides on the 200 as well. A filter hit
is not an analysis: it says a published set vouches for this key, not
what the artifact is, so the two never collapse into one field. Missing
filters fail closed (`unknown`).

Verdicts are stored per ruleset (scan release, traits commit, model
bundle), so a rules or model update reads as a miss rather than serving
a verdict the current detector would no longer give. `SCAN_ANALYSIS_CACHE=0`
disables the store, which degrades every lookup to `unknown sample`.

Headers: `X-SHA256`, `X-Scan-Source: index`, and `Cache-Control` —
`max-age=86400` on a verdict (immutable for the ruleset that produced
it), `no-store` on a miss (it becomes a hit the moment anything analyzes
the artifact). The scope is `private` when a token is configured.

400 for a malformed digest, for a string that is not a PURL, or for
anything other than exactly one key.

### `GET /_/health`

Liveness and load. 200 when ready, 503 while loading or failed. The only
route that does not require a bearer token.

    {
      "status": "ok",
      "rss_mb": 312,
      "max_rss_mb": 16384,
      "active_tasks": 1,
      "max_concurrent_tasks": 4,
      "load": 0.25,
      "load_avg": 0.42,
      "uptime_secs": 91
    }

`status` is one of `ok`, `starting`, `failed`, `degraded`, `saturated`.

Because the route is public, that body carries no operational detail. A
request that *does* present a valid token — or any request to a server with
no `--token-file`, which behaves exactly as it did before tokens existed —
additionally gets:

    "stuck_orphans": 0,
    "long_running_tasks": [ ... ],
    "oldest_task": { "name": "...", "elapsed_secs": 310 },
    "rayon_threads": 8

`long_running_tasks` and `oldest_task` name the samples currently being
analysed, which is why they are not public.

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
| 422  | Corrupt, truncated, encrypted-without-password, depth or count limit.   |
| 500  | Internal error.                                                         |
| 503  | Starting, failed, overloaded, or at capacity.                           |
| 504  | Analysis exceeded the per-request watchdog.                             |

`/analyze`, `/analyze-purl`, and `/analyze-path` also set `X-Total-Ms` on the response.

Errors share a single shape:

    { "error": "string", "detail": "optional chain" }

## Logging

The server writes to stderr (journald under systemd). Every request produces
exactly one access line when its response is ready:

    INFO scan::server::access: POST /analyze-purl id=9 status=200 dur_ms=10351
      peer=127.0.0.1 fwd=203.0.113.9 auth="token" req_bytes=33
      purl="pkg:npm/left-pad@1.3.0" trace="bl-9f2c" ua="beamline/0.3"

A field with nothing to say is left off the line rather than printed empty.

| Field       | Meaning                                                                |
| ----------- | ---------------------------------------------------------------------- |
| `id`        | Server-assigned request id; every other line about this request repeats it. |
| `status`    | HTTP status returned.                                                  |
| `dur_ms`    | Wall time from arrival to response.                                    |
| `peer`      | Socket peer address. Behind a tunnel this is always loopback.          |
| `fwd`       | Client address a proxy reported (`CF-Connecting-IP`, `X-Forwarded-For`, `X-Real-IP`). Advisory: logged, never used for access control. |
| `auth`      | `token`, `open` (no `--token-file`), `anon` (unauthenticated `/_/health`), or a rejection reason: `no-credential`, `malformed-credential`, `bad-token`, `peer-denied`, `loopback-only`, `no-peer-info`. |
| `req_bytes` | Request `Content-Length`, when the client sent one.                    |
| `shared`    | `true` when this request attached to an analysis already in flight for the same bytes or PURL and replayed its result instead of doing the work. Absent otherwise. |
| `sha256`    | The artifact the request was about, by digest: the `/lookup?sha256=` key, or the digest of the bytes uploaded to `/analyze`. On a `/lookup?purl=` that hit, the digest that PURL resolved to. |
| `purl`      | The artifact the request was about, by package locator, in canonical form — `/lookup?purl=` or `/analyze-purl`. A locator that failed to parse is echoed as sent, so a 400 names the typo. |
| `path`      | The file `/analyze-path` was asked for, as the caller wrote it. Where the canonicalized form differs, a rejection line carries both. |
| `cred_len`  | Length of the rejected bearer credential (`bad-token` only).           |
| `cred_fp`   | First four bytes of its SHA-256, hex (`bad-token` only). See below.    |
| `trace`     | The caller's `X-Request-Id`, for correlating with the calling service. |
| `ua`        | `User-Agent`, control-stripped and truncated to 120 characters.        |

### Following a result to hopper

An analysis that succeeds writes a completion line before its access line, and
that line ends with where the verdict goes next:

    INFO scan::server::handlers: <-- 200 OK id=45 key=purl:pkg:npm/left-pad@1.3.0
      elapsed_ms=11563 classification=benign analysis="fresh" hopper="queued"

`hopper="queued"` means the verdict was handed to the background uploader;
`hopper="disabled"` means the server was started without `--hopper`, so the
answer lives only in this process's verdict index. A server with no `--hopper`
also says so once, at startup.

The uploader reports the outcome on its own thread, after the response has
already gone back to the caller, naming the artifact by digest and — when the
request named one — by PURL:

    INFO scan::upload: upload: result renewed on hopper
      sha256=870c0fe… purl="pkg:npm/left-pad@1.3.0" attempt=0

    WARN scan::upload: upload: hopper unreachable, giving up after retries
      sha256=870c0fe… purl="pkg:npm/left-pad@1.3.0" attempts=4

A rejected upload logs `upload: rejected by hopper; not retrying` with the
status and body — a 401 there means this server's `~/.tok/hopper` is not the
token hopper loaded. See
[WORKERS.md](WORKERS.md#authenticating-to-hopper).

An upload failure never fails the request: the caller already has its answer,
and hopper renewal is best-effort by design.

The identifier is never taken from the raw query string or request body: each
handler attaches the key it actually parsed and validated, control-stripped and
bounded to 200 characters. A newline in a caller-supplied locator therefore
cannot fabricate a second log line.

A 401 never says which token was presented — but `cred_fp` identifies it to
anyone already holding the token, without recording a secret. Compare it with
whichever token file you believe is current:

    printf %s "$(cat ~/.tok/scan)" | shasum -a 256 | cut -c1-8

A match means the client is using the right token and the *server* is not —
the token is read once at startup, so a rotation without a restart looks
exactly like this. A mismatch means the client is holding a different token.
The startup log names the file the running process actually read.

Levels: 5xx logs at WARN, every access-control rejection at WARN, a missing
peer address at ERROR, a successful `/_/health` probe at DEBUG (so liveness checks do not fill the log), and
everything else at INFO. Rejected requests are logged only here — the ACL does
not log a second line of its own.

Analyses add `--> POST …` when they start and `<-- 200 OK` / `<-- analysis
failed` when they finish, both carrying the same `id`. A successful analysis
carries two fields naming where its answer came from, so a fast response is
never a mystery:

| Field      | Values                          | Meaning                          |
| ---------- | ------------------------------- | -------------------------------- |
| `analysis` | `fresh` / `cached`              | Whether cleave ran the pipeline or replayed the whole report from its on-disk cache (SQLite, keyed by content digest, options, and traits revision). Survives restarts. |
| `llm`      | `queried` / `cached` / `failed` | Where the `--interpret` verdict came from. Absent when no pass ran. |

Together with `shared=true` on the access line, those cover every way a request
avoids work: riding another request's in-flight run (`shared`), replaying a
stored report (`analysis=cached`), and skipping the LLM call (`llm=cached`). A failure line carries
the whole error chain, which is the same text the response body returns as
`detail`:

    WARN scan::server::handlers: <-- analysis failed id=5 status=422
      error=cleave analysis of bad.tgz: Failed to read tar entry: corrupt deflate stream

`RUST_LOG` overrides the defaults (`scan=info,cleave=warn` in server mode) —
`RUST_LOG=scan=debug` turns on health-probe lines and per-request diagnostics.

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

    $ atomscan serve --bind 127.0.0.1:49999 --workers 4 --token-file ~/.tok/scan
    $ AUTH="Authorization: Bearer $(cat ~/.tok/scan)"
    $ curl -s http://127.0.0.1:49999/_/health | jq -r .status
    ok
    $ curl -s -H "$AUTH" -F file=@/bin/ls http://127.0.0.1:49999/analyze \
        | jq '.ml | {lvl, prob, version}'
    {
      "lvl": -1,
      "prob": 0.01,
      "version": "spec=4 abi=1 hash=8f3a91"
    }
