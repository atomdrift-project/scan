# Edit allowlist for implementing agents (litmus)

The proposer hands ideas to a coding-agent (gemini by default) which
edits files inside the worktree. The agent has wide latitude in *how*
to realize an idea, but the following boundaries are enforced.

## May edit

- `src/**/*.rs`
- `Cargo.toml`
- `Cargo.lock` (let cargo regenerate after dep changes)
- Anywhere under a Rust source tree the proposer named explicitly via `hints`.

## Must not edit

- `tests/**` — never weaken test coverage to make a perf change pass.
- `.github/**` — CI changes are out of scope.
- `Makefile` — bench targets are the contract; changing them invalidates the measurement.
- `benches/**` if added later — same reasoning as tests.
- `vendor/**` — vendored sources are locked.
- `packaging/**` — Wolfi/melange builds are out of scope.
- `scripts/**` — deployment + rollout scripts; platform-specific behaviors are not perf-relevant.
- `testdata/**` — fixture files; changes invalidate the bench corpus.

## Trigger an auto-revert

`cleave-tuna` reverts the experiment without benchmarking if:

- `cargo check` fails.
- `cargo test --lib` fails.
- The agent produced no changes after its run.
- Diff touches any path in the "must not edit" list.

The third one matters: if you can't realize the idea, return early.
Better to leave the slate slot empty than to commit a no-op.

## Litmus-specific guardrails

- The `#[global_allocator]` declaration in `src/main.rs` is load-
  bearing for heap profiling. Tuna's memory-mode hotspot data depends
  on `tikv_jemallocator::Jemalloc` being the active allocator with the
  `jemalloc-prof` feature available. Don't replace it.
- Litmus depends on `cleave` and `xgboost-ars` as Cargo deps. Edits
  inside those crates' sources are outside this worktree and won't
  apply cleanly anyway. Realize ideas at the litmus call sites.
