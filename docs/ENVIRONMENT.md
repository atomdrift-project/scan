# Environment variables

Configuration is on the command line (`atomscan --help`). These are the
variables scan reads, and the only ones: credentials, settings it shares with
sibling tools, and switches set once for a whole host. A flag or number that
does not parse is a startup error, not a silent default.

## Flags with an environment fallback

| Variable | Flag |
| --- | --- |
| `SCAN_HOPPER` | `--hopper` |
| `SCAN_FOLLOW` | `--follow` |
| `ATOMDRIFT_UPLINK` | `--uplink` (shared with cleave and isomer; legacy alias: `ATOMDRIFT_ENRICHED`) |

## LLM second opinion

| Variable | Meaning |
| --- | --- |
| `SCAN_LLM` | Endpoint list, when `--llm` is not given. |
| `SCAN_LLM_MODEL` | Model name, when `--llm-model` is not given. |
| `SCAN_LLM_KEY` | Bearer token, in place of `~/.tok/llm`. |
| `SCAN_LLM_CONCURRENCY` | LLM calls in flight per process. Default: physical cores, 4–16. |
| `SCAN_LLM_BACKGROUND_CONCURRENCY` | Of those, how many work nobody waits on may hold. Default: a quarter, at least one. |
| `SCAN_LLM_SYSTEM_PROMPT_FILE` | Replace the built-in grading prompt with this file. Cached verdicts key on the prompt, so an override never replays built-in ones. |

## Host settings

| Variable | Meaning |
| --- | --- |
| `SCAN_MODELS_DIR` | Use the model bundle in this directory, and never update it. |
| `SCAN_NO_ANALYSIS_CACHE` | `1` disables every analysis cache in the stack (scan, cleave, filefacts, stng) while keeping compiled rules, for honest benchmarks. |
| `SCAN_NO_UPDATE_CHECK` | `1` skips the once-a-day check for a newer release. |
| `SCAN_NO_UPLOAD` | `1` disables CLI result and artifact uploads; Hopper corpus lookups continue. |
| `CLEAVE_RAYON_THREADS` | Width of the analysis thread pool. Default: the physical core count. |
| `CLEAVE_TRAITS_DIR` | Read traits from this directory instead of the installed ones (shared with cleave). |
| `RUST_LOG` | Log filter, e.g. `scan=debug`. |
| `MALLOC_CONF` | Not read for configuration; on FreeBSD, scan warns about an option known to break allocation. |
