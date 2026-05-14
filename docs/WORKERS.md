# Litmus Workers

`litmus worker` is the pull-based counterpart to `litmus serve`. It
exposes no HTTP. It polls a [hopper](https://codeberg.org/atomdrift/hopper)
instance for jobs, analyses each file, and posts the result back.

The classifier, thresholds, and response shape are identical to the
server. See [SERVER_API.md](SERVER_API.md) for the server, and
[JSON_REPORT.md](JSON_REPORT.md) for the JSON envelope every worker
result carries.

## Running a worker

    litmus worker --url http://hopper-host:8081

`--url` is required. Everything else has a sensible default.

| Flag             | Default          | Meaning                                                                  |
| ---------------- | ---------------- | ------------------------------------------------------------------------ |
| `--url`          | (required)       | Hopper base URL.                                                         |
| `--name`         | hostname         | Worker identity reported to hopper.                                      |
| `--workers`      | auto             | Concurrent analysis slots.                                               |
| `--poll-secs`    | `2`              | Sleep between polls when the queue is empty.                             |
| `--max-rss-gb`   | `0` (auto)       | RSS ceiling. Auto = 85 % of total system RAM. `-1` disables.             |
| `--data-dir`     | none             | Local file root. When set, the worker reads files locally instead of fetching them; SHA-256 is verified before use. |
| `--max-jobs`     | unlimited        | Exit after this many jobs. Useful for cron-style batches.                |
| `--traits-dir`   | none             | Writable cleave traits directory. Cloned on first run if missing.        |
| `--nice`         | `10`             | `nice(2)` value applied at startup. `0` leaves priority unchanged.       |

The same environment variables apply as for the server:
`CLEAVE_TRAITS_DIR`, `CLEAVE_RAYON_THREADS`, `LITMUS_MODELS_REPO`,
`LITMUS_MODELS_REF`.

## Job lifecycle

1. **Claim.** The worker calls hopper to claim a job. The job carries
   the file path (relative to `--data-dir`, if set) or a download URL,
   plus the expected SHA-256.
2. **Fetch.** If `--data-dir` is set and the file exists there, the
   worker hashes it and uses it directly. Otherwise it downloads.
   Mismatched SHA-256 fails the claim.
3. **Analyse.** Same `classify_file` / `classify_bytes` path used by
   the server.
4. **Report.** The worker posts the [response envelope](JSON_REPORT.md)
   back to hopper. The shape is the same as the server's `/analyze`
   response.
5. **Sleep.** If the queue was empty on the last claim, the worker
   sleeps `--poll-secs` before trying again.

## Resource discipline

The worker is designed to share a host with other work.

- **Nice.** The default `--nice 10` keeps analysis bursts from
  starving interactive processes. Unprivileged callers can only raise
  the nice value, never lower it; pass `0` when profiling.
- **RSS ceiling.** When current RSS exceeds the limit, the worker
  *pauses claims* — it does not abort jobs in flight. It will resume
  once memory drops. Auto-resolves to 85 % of total system RAM. Set
  `-1` only behind an external supervisor (`MemoryMax=`, jail
  `memoryuse`, cgroup, etc.) that already enforces a hard cap.
- **Concurrency.** `--workers` is the hard limit on simultaneous
  analyses. There is no queue beyond hopper itself.
- **Rayon stack.** Worker startup installs an 8 MB rayon stack;
  cleave's archive recursion exhausts the default 2 MB stack on
  pathological inputs.

## Operations

- **Idempotent deploys.** The worker tolerates repeated launches and
  trait/model directories that already exist; it does not re-clone if
  the working tree is current.
- **Bastille jails.** Before upgrading the binary inside a jail,
  force-kill the worker with `kill -9`. Graceful shutdown can stall
  during rizin teardown.
- **Profiling.** `profile_worker.sh` at the repo root runs a worker
  under `samply` with the right stack and env settings.

## Security

The worker is a client of hopper. It does not listen on any port.

- **Outbound only.** The trust boundary is the hopper URL; treat it
  the way you treat the rest of your service mesh.
- **File integrity.** Both download and local-file paths verify
  SHA-256 before analysis. A worker will not analyse a file whose
  hash does not match the job claim.
- **No code execution.** Same static-analysis-only guarantees as the
  server. See [SERVER_API.md#security](SERVER_API.md#security).
- **Worker identity.** `--name` is purely identifying; hopper trusts
  any worker that can reach it. Lock down hopper's network, not the
  worker.
