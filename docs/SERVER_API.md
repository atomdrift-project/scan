# Atomdrift Scan Server API

`atomscan serve` is an HTTP daemon that takes an artifact and answers whether
to let it through. It binds to loopback by default. Pass `--token-file` to
require a bearer token on every route but `/_/health`.

## Running the server

    atomscan serve --token-file ~/.tok/scan

| Flag                  | Default           | Meaning |
| --------------------- | ----------------- | ------- |
| `--bind`              | `127.0.0.1:49999` | Listen address. |
| `--workers`           | 3 per physical core, at least 2 | Concurrent analyses. Past it a request gets `429`, never a queue. |
| `--max-size-mb`       | `100`             | Largest artifact accepted, uploaded or read from disk. |
| `--max-rss-gb`        | `0` (auto)        | RSS ceiling. `0` reads the cgroup or system limit; `-1` disables the check. |
| `--allow-cidr`        | none              | Comma-separated networks, besides loopback, allowed to connect. |
| `--token-file`        | none              | File holding the bearer token. See [Authentication](#authentication). |
| `--allow-local-paths` | none              | Directory `?path=` may read from. Repeatable. See [Local files](#local-files). |
| `--hopper`            | none              | Hopper base URL(s). Results are filed there, and lookups this server cannot answer are deferred to it. |
| `--traits-dir`        | none              | Writable cleave traits directory. |
| `--analysis-timeout`  | `2040`            | Per-request analysis budget in seconds; `0` disables. |

Global flags (`--follow`, `--llm`, `--level`, …) apply as they do to a scan;
`atomscan --help` lists them. Environment variables are listed in
[ENVIRONMENT.md](ENVIRONMENT.md).

The listener binds before the model loads. Until it has, analysis routes
answer `503` with code `starting`; poll `/_/health` until `status` is `ok`.
Models and traits are refreshed once at startup (`-u` forces it, `--no-update`
skips it). If the traits cannot be resolved the server never reports ready:
`/_/health` answers `503` with `{"status":"failed","reason":"initialization_failed"}`
and the log names the path.

## Authentication

`--token-file PATH` reads the token from the first non-empty line of `PATH`
and requires it on every route except `/_/health`:

    curl -H "Authorization: Bearer $(cat ~/.tok/scan)" …

The examples below abbreviate that header as `$AUTH`.

- **Loopback is not exempt.** A Cloudflare tunnel's connector dials the server
  over loopback, so every remote request arrives with a loopback address.
  Exempting loopback would exempt the internet.
- **A token is at least 16 bytes** of the RFC 6750 `token68` alphabet. Anything
  else could not be sent in a header, so the server refuses to start rather
  than answer `401` forever.
- **Missing means fatal.** If `--token-file` is set and the file is missing,
  empty or unreadable, the server does not start.
- **Only a digest is kept**, so the token cannot surface in a log or a core.
- **Rotation needs a restart.** The token is read once.

A missing and a wrong token get the same `401` with `WWW-Authenticate: Bearer`.
Without a token, `/_/reload` and `/_/update` answer loopback only, so
`--allow-cidr` grants analysis and never administration.

## Endpoints

Every error has one shape, with a stable `code`:

    {"error": {"code": "invalid_purl", "message": "not a package URL"}}

### `GET /v1/lookup`

What is known about an artifact, as a decision. Never analyzes and never
takes a slot.

    GET /v1/lookup?purl=<url-encoded>
    GET /v1/lookup?sha256=<64 hex>
    GET /v1/lookup?url=<url-encoded>
    GET /v1/lookup?purl=a&purl=b                  up to 50 per request
    GET /v1/lookup?purl=…&false_positive_budget=25

Answers from this server's own index first, then from `--hopper`. With no
hopper it answers from the index alone.

    {
      "decision": "block",
      "purl": "pkg:npm/evil@1.0.0",
      "sha256": "2cf24dba…",
      "severity": "hostile",
      "fires_at": 3,
      "reason": "Postinstall launches a reverse shell.",
      "findings": [{"id": "objectives/execution/shell/bash", "crit": 5}],
      "engine_version": "2.13.0",
      "analyzed_at": "2026-08-01T00:00:00Z"
    }

One artifact answers with one object; repeated keys answer with a list, in
the order asked.

A digest names its bytes forever. A name does only when it cannot come to
mean other bytes: a published npm, crates.io, Go, NuGet or Maven release, or a
commit. Every other name — a gem, a PyPI release, a git tag, a version-less
PURL — is answered from this server's index for 30 minutes after the analysis
that last resolved it, then resolved again. A `?url=` is judged by the PURL it
parses into when it is a registry download (`registry.npmjs.org`,
`static.crates.io`, …), so a published tarball's URL answers like its PURL.
A PyPI file's URL (`files.pythonhosted.org`) is immutable too, because PyPI
never replaces or reuses a filename, though its release by PURL is not. Any
other URL is a mutable name. `POST /v1/analyze?url=` answers from the same
index before it fetches anything, as `?purl=` does; `force=1` fetches anyway.

`X-Scan-Source` names the layer that answered: `scan:index`, `scan:bloom`,
`scan:replica`, `scan:primary`, or `none` when nothing could. A list reports
the most expensive layer any row reached, in that order.

| `decision`    | Meaning |
| ------------- | ------- |
| `allow`       | Analyzed; not hostile at the caller's budget. |
| `block`       | Analyzed; hostile at the caller's budget. |
| `unanalyzed`  | Nobody has analyzed it. |
| `unavailable` | We could not answer. Says nothing about the artifact. |

`unanalyzed` is a claim about the artifact and `unavailable` a claim about
us; a caller's policy is entitled to treat them differently.

`fires_at` is the tightest false-positive budget, per 100 million benign
files, at which the artifact grades hostile: `-1` when it fires at none,
`null` for a verdict reached with manual thresholds. It is measured.
`false_positive_budget` is chosen: what the caller tolerates. It defaults to
this server's `--level`. `decision` is `block` when `fires_at` is at or below
the budget. `severity` is `benign`, `suspicious` or `hostile`, independent of
the budget.

### `POST /v1/analyze`

Analyze an artifact and answer with the same decision.

    POST /v1/analyze?purl=pkg:npm/left-pad@1.3.0
    POST /v1/analyze?url=https://example.com/app.tgz
    POST /v1/analyze                         <the artifact as the body>
    POST /v1/analyze?path=/srv/incoming/app.tgz
    POST /v1/analyze?sha256=<64 hex>&refresh=1

Name exactly one artifact. A body is the artifact itself, analyzed under the
digest of its bytes; `X-Filename` names it so cleave can type it by
extension. A `?purl=` sent with a body is provenance, not the artifact.

A verdict already held answers immediately and takes no slot. `force=1`
analyzes regardless. `fresh=1` (or `X-Hopper-Fresh: 1`) skips the bloom
filters. `refresh=1` reconciles one digest with hopper, analyzing it here
when hopper's verdict is missing or from other traits.

The requested artifact is always analyzed. `follow` chooses which references
found inside it are also fetched and analyzed: `dependencies`, `references`,
`ci-actions` (implies `dependencies`), `all` or `none`, comma-separated or
repeated. It may narrow the server's own policy (`--follow`); naming a
category the server does not follow is a `400 follow_not_allowed`. A request
under any policy may be answered from a verdict already held,
and every analysis is filed whatever policy produced it, so the index and
hopper keep one verdict per artifact; pass `force=1` to analyze under exactly
the requested policy. Requests under different policies never share an
in-flight analysis. `X-Scan-Follow` names the policy an analysis ran under.

Two requests for the same artifact share one analysis. The reply is
newline-delimited JSON: progress frames while the run goes on, then the
decision. Read lines until one carries `decision`.

    {"state":"analyzing","elapsed_ms":1002,"phase":"fetch","purl":"…"}
    {"decision":"block","fires_at":3,"purl":"pkg:npm/evil@1.0.0",…}

Nothing is sent for the first 250 ms, so an outcome that needed no work keeps
an ordinary status: `429` at capacity is a real `429` a router can act on.
Once streaming starts the status is committed, and a failure arrives as
`decision: unavailable` with a `reason`. A stream that ends without a
decision was cut short; the analysis continues, and a retry joins it.

Headers: `X-Total-Ms`, `X-Scan-Source` (`scan:index`, `scan:cached`,
`scan:analysis`, …), `X-Scan-Follow`, and `X-Scan-Purl` with the normalized
coordinate.

#### Local files

`?path=` analyzes a file already on the server's host. It is off unless the
server was started with `--allow-local-paths <DIR>`, and the server refuses to
start with it unless it is bound to `127.0.0.1`, or both `--allow-cidr` and
`--token-file` are set. Bound to `127.0.0.1` without a token, it warns at
startup: a tunnel makes any caller a loopback caller.

The path must be absolute. It is resolved, checked against the allowed
directories, and opened once without following a final symlink; the
analysis runs on the bytes read from that handle, so what is analyzed is
what was checked. A file that does not exist, lies outside the allowed
directories, or is not a regular file gets the same `403 path_not_allowed`,
so the route does not reveal which paths exist. The answer is the decision an
upload of the same bytes would get.

### `GET /_/health`

Liveness and load: `200` when ready, `503` while starting or failed. The only
route that needs no token. `status` is `ok`, `starting`, `failed`,
`degraded` or `saturated`. A request with a valid token (or any request to a
server without one) also gets the names of the samples being analyzed.

### `GET /_/info`, `GET /_/stats`

`/_/info` is static: version, slots, CPUs, limits, model and traits commits.
`/_/stats` is what a router reads to choose a server: free slots, CPU busy,
recent latency by type and size, upload backlog.

### `POST /_/reload`, `POST /_/update`

`/_/reload` reloads the model bundle and traits from disk and swaps them in;
on failure the previous model stays. `/_/update` first pulls the latest
traits, models, bloom and NSRL filters; its `update_errors` names each source
that failed to update, and is empty when none did. Why a source or a reload
failed is logged, never returned: those errors carry install paths. One at a time; a
concurrent call gets `409`.

### `GET /_/memory`, `GET /_/requests`, `GET /_/threads`

Diagnostics: allocator counters, in-flight analyses with their phase, and
per-thread state where the platform reports it.

## Status codes

| Code | Cause |
| ---- | ----- |
| 400  | Malformed request: a bad locator or budget, a `follow` wider than the server's, or more than one artifact named. |
| 401  | Missing or wrong bearer token. |
| 403  | Peer not allowed, admin route without a token, or a `?path=` refused. |
| 409  | A reload or update is already running. |
| 413  | Artifact larger than `--max-size-mb`. |
| 415  | Unsupported file, archive or compression format. |
| 422  | The artifact is corrupt, truncated, encrypted, or past a depth or count limit, or could not be retrieved. |
| 429  | At capacity. `Retry-After` says when to try again. |
| 500  | Internal error. |
| 503  | Starting, failed to start, or over the memory ceiling. |
| 504  | The analysis exceeded `--analysis-timeout`. |

## Logging

The server logs to stderr. Every request produces one access line when its
response is ready, and every other line about it repeats its `id`:

    INFO scan::server::access: POST /v1/analyze id=9 status=200 dur_ms=10351
      peer=127.0.0.1 auth="token" purl="pkg:npm/left-pad@1.3.0" trace="bl-9f2c"

| Field       | Meaning |
| ----------- | ------- |
| `id`        | Server-assigned request id. |
| `status`    | HTTP status returned. |
| `dur_ms`    | Wall time from arrival to response. |
| `peer`      | Socket peer. Behind a tunnel, always loopback. |
| `fwd`       | Client address a proxy reported. Logged, never trusted. |
| `auth`      | `token`, `open`, `anon`, or the reason a request was refused. |
| `shared`    | `true` when the request rode an analysis already in flight. |
| `streamed`  | `true` when the answer was streamed. |
| `sha256`, `purl`, `url`, `path` | The artifact, as validated. Caller text is control-stripped and bounded, so it cannot forge a line. |
| `cred_len`, `cred_fp` | On a `bad-token` rejection: the length and first four SHA-256 bytes of what was sent, to tell a wrong token from a server that has not been restarted since rotation. |
| `trace`     | The caller's `X-Request-Id`. |
| `ua`        | `User-Agent`, truncated. |

`RUST_LOG` overrides the default levels (`scan=info,cleave=warn`).

## Security

- **Bind is loopback.** Changing `--bind` changes who can reach every route.
- **No TLS.** Put a proxy that terminates TLS in front of a server reachable
  from anywhere else.
- **Local files are opt-in, per directory**, and refused at startup on a
  server reachable from the network without both an address allow-list and a
  token.
- **Upload names are sanitized** to `[A-Za-z0-9_.-]`, `..` collapsed, at most
  63 bytes; the caller's name is never used on disk.
- **Sizes are capped while streaming**, not after.
- **Capacity is a hard limit.** Past `--workers`, `429`; past the RSS ceiling,
  `503`. There is no queue.
- **Static analysis only.** No artifact is executed.

## Example session

    $ atomscan serve --token-file ~/.tok/scan &
    $ AUTH="Authorization: Bearer $(cat ~/.tok/scan)"
    $ curl -s http://127.0.0.1:49999/_/health | jq -r .status
    ok
    $ curl -s -H "$AUTH" -X POST \
        'http://127.0.0.1:49999/v1/analyze?purl=pkg:npm/left-pad@1.3.0' | jq .decision
    "allow"
