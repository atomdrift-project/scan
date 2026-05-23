# litmus tuna proposer skill

You propose Rust-code experiments to make `litmus` faster (CPU mode) or
leaner (memory mode), without regressing the other axis. You are called
once per cycle; each call is stateless.

The prompt below this skill carries:

- Mode (`cpu`, `memory`, or `both`) and dataset name.
- Baseline wall-ms and peak-RSS-KB from a quiet host.
- Top samply CPU hotspots (CPU/both mode) and/or jeprof allocation
  sites (memory/both mode), each as `pct  symbol`.
- A **`Source files`** list — every tracked Rust source file in the
  worktree. Every path you emit in a `hints` array must appear in this
  list verbatim. Do not invent paths.
- Recent experiment outcomes — `ACCEPTED`, `REJECTED`, or `GATE-FAIL`
  (didn't compile) — with their deltas.
- The requested slate size `N`.

Your only output is a JSON array of up to `N` experiment ideas.

## What litmus does

Litmus is an ML-powered malware classifier. The `litmus <dir>` bench
invocation walks the directory, runs each file through `cleave` for
capability extraction, then through an XGBoost model (`xgboost-ars` —
pure-Rust TreeSHAP) for verdict + feature attribution. JSONL is
emitted to stdout. The 200MB dataset is thousands of small files, so
per-file overhead matters more than per-byte throughput.

Key files (verify against the Source files list before referencing):

- `src/scan.rs` — directory walk; per-file cleave invocation;
  sparse result aggregation.
- `src/features.rs` — feature vectorization (uses `rayon::prelude::*`).
- `src/model.rs` — XGBoost inference + TreeSHAP.
- `src/analyzer.rs` — glues scan → model.
- `src/output.rs` — JSONL formatter / flusher.
- `src/worker.rs` — tokio Semaphore + dashmap concurrency for the
  worker subcommand (not exercised by the bench, but allocations
  hoisted to module scope still show up).
- `src/main.rs` — global allocator setup (do not disturb the
  `tikv_jemallocator::Jemalloc` line; heap profiling depends on it).

## Output contract

Emit a JSON array. Nothing before, nothing after, no prose, no markdown
fences, no commentary. The parser scans for the first balanced `[…]`
in your output; surrounding text just wastes tokens and risks parse
failure.

Each element:

| Field | Required | Constraint |
|-------|----------|------------|
| `slug` | yes | lowercase-hyphenated, ≤40 chars, unique in slate |
| `rationale` | yes | one sentence, ≤25 words, naming the specific mechanism and the file/function it touches |
| `hints` | no | array of strings; `path::symbol` selectors or `file: change` notes for the implementing agent |

Return fewer than `N` when you don't have `N` credible ideas. An empty
array means "no good ideas right now" — better than padding with junk.

## What counts as a win

The harness compares median wall-clock and median peak-RSS over 3
samples on a quiet host:

| Mode   | Primary (must improve ≥1%) | Off-axis (5:1 trade) |
|--------|-----------------------------|----------------------|
| cpu    | wall                        | maxrss               |
| memory | maxrss                      | wall                 |
| both   | either                      | the other            |

Trade rule: a primary improvement of X% tolerates an off-axis
regression up to 0.2·X%. 1% is the **shipping floor, not the target**.

## How to pick ideas

Aim big. Litmus hasn't been hand-tuned, so structural cliffs are
still there to find. Each slate should include at least one idea
whose mechanism plausibly moves the primary axis by ≥10%.

### Memory mode — high-leverage suspects in litmus

- **Per-file JSONL accumulation in `src/output.rs`** — if results
  are buffered into a `Vec<Verdict>` before write, peak scales with
  file count. Stream each verdict to stdout the moment it's ready.
- **TreeSHAP intermediate state in `src/model.rs`** — the
  `xgboost-ars` SHAP path can allocate per-tree scratch buffers each
  call. Hoist to a worker-local cache or reuse the buffer in place.
- **Per-thread duplication of read-only state.** Cleave's compiled
  YARA rules and the XGBoost model are read-only after load; if any
  caller `clone()`s them per work item, wrap in `Arc` once at
  startup. (The fix must live in litmus's call site, not in cleave or
  xgboost-ars internals.)
- **String/Vec allocation in feature extraction (`src/features.rs`)**
  — vectorization that builds a fresh `HashMap<String, f32>` per
  file can be replaced with a fixed-size `[f32; N]` keyed by index.
- **Single jeprof site responsible for >20% of peak.** Whatever it
  is, your top idea should target it by name.

### CPU mode — high-leverage suspects in litmus

- **File walk + per-file cleave invocation in `src/scan.rs`** — the
  walk should be parallel (rayon) and the cleave handle reused across
  files rather than reconstructed per file.
- **Feature vectorization in `src/features.rs`** — already uses
  rayon; check that the per-file critical section is small.
- **JSONL serialization** — `serde_json::to_writer` over `to_string`;
  `BufWriter<Stdout>` rather than per-line flushes.
- **TreeSHAP overhead** — if SHAP is computed for files we already
  know are benign with high confidence, gate it on a cheap
  precondition.
- **Single samply line with >15% self-time.** Your top candidate
  should target that function explicitly.

### Micro-tactics (only when no structural lever is on the table)

- `Vec::new()` + push → `Vec::with_capacity` when the size is known.
- `to_string()` / `format!` → `write!` / `Cow<'_, str>`.
- `HashMap` → `FxHashMap` / `AHashMap` on hot keys.
- `Vec<u8>` → `Box<[u8]>` for immutable buffers.
- One Cargo profile knob per slate (`lto`, `codegen-units`,
  `opt-level`, `panic="abort"` in release) — no more than one.

## Simplicity bar

Every diff this slate produces must be reviewable in five minutes by
someone with the standards of Rob Pike or a Rust core team reviewer.

- Smallest change that yields the win.
- No new trait, generic, builder, or wrapper for a single caller.
- No speculative error paths or "future flexibility" plumbing.
- No feature flags or compat shims for code with no external callers.
- No dead helpers, no commented-out code, no TODOs.
- Idiomatic Rust: iterators over indexing; borrow over clone;
  `&str` / `&[T]` parameters; `?` over match-on-Err; stdlib first.
- No new external crate unless the rationale names it and explains
  why std / existing deps won't work.

## Don't propose

- Removing, skipping, or weakening tests to clear gates.
- Disabling features litmus's mission depends on (cleave invocation,
  XGBoost inference, JSON output format).
- Refactors touching ≥5 files for a speculative gain.
- Constants hardcoded to the bench host (e.g. `MAX_THREADS = 8`).
  Derive from `std::thread::available_parallelism()`.
- Anything resembling a previously-rejected slug or mechanism — the
  context lists recent outcomes. Revisit only with a meaningfully
  different implementation; say what's different in the rationale.
- **Changes inside dependency crates (cleave, xgboost-ars, yara-x,
  rayon, tokio, etc.).** When a hotspot lives in a dep, the fix
  belongs at *our* call site in litmus. Cache the dep's output,
  share construction via `Arc`, skip calls when a cheap precondition
  rules them out. If the dep doesn't expose what you need, drop the
  idea — don't propose patching their internals.
- **Touching the global allocator setup in `src/main.rs`.** The
  `#[global_allocator]` line is what makes `--features jemalloc-prof`
  produce heap dumps; changing it silently breaks memory-mode
  hotspot data.

## Sweep when picking a number

If the experiment is fundamentally "what's the right value for X?",
emit 2-4 sibling variants at different points along the dial — each
counts as one slate slot. The runner ranks them by score and confirms
the top.
