# Integrating with Litmus

Three paths in. Pick by volume.

| Path        | Pick when                                              | Reference                                    |
| ----------- | ------------------------------------------------------ | -------------------------------------------- |
| CLI         | One-shot or batch use up to ~5 scans / minute.         | `litmus --help`                              |
| HTTP server | Sustained traffic past ~5 scans / minute.              | [SERVER_API.md](SERVER_API.md)               |
| Worker      | Pull-based from a hopper queue.                        | [WORKERS.md](WORKERS.md)                     |
| Rust library| You are already in Rust and need direct access.        | source — `classify_file`, `classify_bytes`   |

All four emit the same response envelope. Schema and field names:
[JSON.md](JSON.md).

## Notes

**Thresholds** are resolved from the model bundle, severity flags, or
`--threshold-*` overrides, in that order. They are fixed at process
start; the server reloads via `POST /_/reload`. See
[SERVER_API.md#thresholds](SERVER_API.md#thresholds).

**Exit codes** (CLI only): `0` clean, `1` hostile, `2` suspicious,
`3` error. The server returns the same classifications in the
`class` field; map yourself if you want exit-code semantics.

**Library stability** is weaker than the CLI and HTTP surfaces.
Breaking changes between minor releases until 1.0. Pin a commit. The
JSON envelope is stable within a major version regardless of path.

**Operational requirements** for the library: install an 8 MB rayon
stack in `main`, call `kill_all_rizin_groups()` at shutdown.

