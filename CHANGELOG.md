# Changelog

All notable changes to Atomdrift Scan (atomscan, formerly litmus). Versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## 3.0.0-beta.0 (unreleased)

### Security

- Stop vouching for git tags, snapshots and young PyPI releases by name; check the digest they serve now.
- Refuse submitted URLs carrying credentials, which were logged, listed in requests and stored in provenance.
- Scan `.github/`, `.gitlab-ci.yml`, `.gitmodules` and other `.git`-prefixed names in directory scans; only a `.git` directory is skipped.
- Fail a directory scan (exit 3) when part of the tree cannot be read, instead of reporting it clean.
- Ignore a hopper or Uplink answer that names bytes other than the ones asked about or uploaded.
- Reuse a hopper verdict on `/v1` refresh only when it was classified, matching the dependency precheck.

### Features

- Connect to Atomdrift Uplink with `--uplink` for zero-hour rules, remote verdicts and a hosted LLM endpoint.
- Choose where artifacts are judged with `--analyze`: auto, local or remote.
- Install the NSRL Modern allowlist, which skips known files only when no adverse match exists.
- Support v23 azoth models, adding keyword, ID and decoded-context features.
- Follow browser PAC configuration URLs found in archives as code dependencies.
- Show dependency attention counts and unfetched URLs in triage output, even when reports are dropped.
- Keep static suspicious and hostile findings visible when the model grades a file benign.
- Grade a saved interpret render with an LLM through `render --llm`.
- Show collector capture records only when suspicious or hostile with `--enable-sidecar`.
- Exit with code 130 when interrupted before every file was analyzed.
- Resolve dependencies on members of explicit Yarn workspaces to their local source instead of a same-named registry package.
- Apply the known-good/known-bad filters to `sys` scans, as `ps` and path scans already did.
- Breaking: remove the worker subcommand; pull-based workers now live in postdoc.
- Breaking: read burton filter format 3 and manifest schema 2, fetched from `bloom/v3/` and `nsrl/modern/v2/`; earlier builds keep reading `bloom/v2/` and `nsrl/modern/v1/`.
- Breaking: remove the legacy analyze, analyze-path, analyze-purl, lookup and status endpoints; use the v1 API.
- Breaking: replace --allowed-dirs with a repeatable `--allow-local-paths`, refused off loopback without CIDR and token.
- Breaking: drop the hopper token environment variables; hopper tokens come from `~/.tok/hopper`, LLM keys from `SCAN_LLM_KEY` or `~/.tok`.
- Breaking: remove flags `--model-dir`, `--traits-dir`, `--threshold-hostile`, `--threshold-suspicious`, `--interpret`, `--llm-key`, `--llm-min-level` and `--llm-min-prob` (and their `--interpret-min-*` aliases), `--llm-benign-notable-cap`, `--llm-timeout`, `--rizin-timeout-secs`, `--rizin-retry-timeout-secs`, `--registry-ttl`, `--fetch` and `--fetch-depth` (use `--follow`/`--follow-depth`), every other `--fetch-*` flag, `serve --idle-worker-slots`, `update-rules --check`/`--models-only` and `validate --skip-traits`.
- Breaking: remove the rizin integration; binary reverse engineering uses binform.
- Breaking: load only routed model bundles at feature ABI 20 or newer; older single-file bundles, calibrators and `evaluation.json` are no longer read, and a bundle without thresholds fails to load instead of falling back to fixed ones.
- Breaking: fetch models, bloom and NSRL filters from `updates.atomdrift.org/scan/` and the release notice from `atomdrift.org/updates/scan.toml`; installed models are recorded in `.scan-models.toml`, so they download once more after upgrading.
- Breaking: refuse analyses past the concurrency limit with 429 instead of queuing them.

### Fixes

- Time out request headers after 30 seconds and cap the server at 1,024 connections.
- Treat DNS and transport failures as retryable outages, not as verdicts about the package.
- Pair `uv.lock` and `pdm.lock` with a co-located `pyproject.toml`, as `poetry.lock` already was.
- Refuse to follow discovered URLs whose host is a NAT64, 6to4 or `0.0.0.0/8` form of a private address.
- Decode every percent escape in npm package names.
- Keep the package type of fetched downloads in model input, even when cached by digest.
- Score every route a file type's policy names, matching the reference model scores.
- Invalidate cached verdicts when cleave's compact report schema changes.
- Warn and disable explanations when a bundle's SHAP file is unusable, instead of failing the scan or server startup.
- Read an out-of-range criticality in a stored report as no signal instead of hostile.
- Let corroborated hostile traits raise a model-suspicious verdict to hostile, as they already raised a model-benign one.
- Analyze large direct uploads on the dedicated whale lanes, answering 429 when every slot is busy, as URL and PURL fetches already did.

### Optimizations

- Keep registry documents compressed, and compact them once registry analysis finishes.
- Spool upload overflow to temporary files instead of pausing analysis or holding reports in memory.
- Extract keyword features without per-word allocations.
- Build CI release binaries with profile-guided optimization from a shared training script.
- Size the analysis pool at 1.5 threads per physical core on large SMT hosts.

### Packaging

- Ship `LICENSE` and `THIRD_PARTY_LICENSES` in every release archive.
- Breaking: stop publishing 32-bit ARM (`arm`/`armv7-unknown-linux-musleabihf`) and Solaris binaries. Neither builds today: stng requires a 64-bit target, and wasmtime does not build on Solaris.

## 2.13.0 (2026-10-17)

### Security

- Downloaded dependencies require current-rule analysis unless a local report or matching-analyzer corpus verdict is available.
- Local and untracked rule changes now invalidate cached verdicts, including rules installed without Git metadata.
- Untrusted upload rejection bodies are truncated and flattened before logging.

### Features

- Added experimental `render` command for saved reports, with terminal, tiny, and interpretation output.
- Added unstable daemon-support library interfaces for Postdoc integration.
- Updated Cleave to 3.0.0-beta.2.

### Fixes

- `update-rules` now honors `--model-dir`.
- FreeBSD worker freeze and resume operations now report an error when the worker has exited.
- Removed server `--extract-dir` support.
- Disabled OpenBSD and NetBSD release builds pending compatible Rust toolchains.

### Optimizations

- Cache versioning reuses Cleave's memoized rule-content and analyzer-build fingerprint.
- Experimental wire-report inference avoids rendering and compact-report copies; shared analyses are summarized once.

## 2.12.1 (2026-10-10)

### Fixes

- Escape control and bidi characters in terminal output, preventing ANSI verdict rewriting.
- Reject submitted URLs containing raw control characters.
- Time out uploads that stall for 30 seconds, and cap `/analyze-purl` bodies at 64 KiB.
- Fall back to local corpus lookups instead of a panic-prone client without timeouts.
- Give daemon blocking threads the analysis-pool stack size to prevent stack-overflow aborts.
- Abort on null allocations in tree-sitter allocator shims.
- Redact `--llm-key` in logs, and warn about hopper tokens sent over plaintext HTTP.
- Refuse bloom filters older than the installed one unless `--force` is given.
- Key the dependency cache by content SHA plus filename.
- Skip the corpus precheck for unpinned, ranged or dist-tag PURLs, which are now fetched.
- Cap registry resolution at 5,000 per manifest and backlog at 10,000 per root.
- Make HTML redirect tag scanning linear, fixing a quadratic DoS.
- Stop LLM interpretation from lowering scores of files cleave flagged hostile.
- Size-check re-extracted bytes in archive member hunts, defeating duplicate-name smuggling.
- Fetch Debian, Red Hat, openSUSE, GNU and Fedora references instead of dropping them.

### Features

- Add the `references.unavailable_command_count` metric for commands whose artifact returns 404/410.
- Archive member hunts now trigger on autoinstall and runner-bridge findings, with a 16 MiB limit.

### Optimizations

- Drop oci-client, jsonwebtoken and their transitive crates from the build, via the fletch update.

## 2.12.0 (2026-10-09)

### Features

- Fetches remote URLs found inside archive members, including images alongside stego-loader evidence.
- Follows `irm|iex`, `curl|sh` and `wget -O` droppers, sending the matching User-Agent.
- Output is plain and grep-friendly when piped or when `NO_COLOR` is set.

### Fixes

- Workers size their memory from cgroup-v2 limits instead of total host memory.
- The open-file limit is raised at startup, fixing EMFILE errors under systemd.
- The first Ctrl-C during `scan sys` now actually cancels the scan.
- Broken pipes on stdout now exit cleanly.
- Busy cores are counted as logical cores, so hyperthreaded hosts stay in routing.
- Fixed a grandchild-process race when suspending workers, and restored companion-worker mode.
- `--interpret` no longer reveals grader paths or withheld conclusion IDs.
- Streamed `/v1` failures now give a reason: `unretrievable` or `too_large`.

### Optimizations

- Removed the eager regex prewarm from scan startup.
- Model scoring caches its logistic kernel and avoids allocating on each prediction.

## 2.11.0 (2026-09-14)

### Features

- Go dependency edges reconcile through filefacts' module/workspace context.
- Cargo/Python manifests refined by co-located lockfile and registry resolution.
- Fetch worklist ordered by execution priority: build deps and proc-macros first, dev deps last.
- Quiet members with analysis gaps are retained in compact output.
- See https://github.com/atomdrift-project/cleave/releases/tag/v2.11.0 for inherited changes.

### Fixes

- References carrying `version_requirement` are no longer superseded by another package's pin.
- LLM gate prefixes updated for renamed trait hierarchies.

### Optimizations

- wasmtime/cranelift at `opt-level = 2` and `test = false` on binaries.

## 2.10.0 (2026-09-10)

### API Features

- `?full=1` on the analyze endpoints returns the complete report instead of the verdict object.
- `?refresh=1` replaces a standing local verdict and re-indexes/re-uploads the fresh result.
- Analysis artifacts are staged and uploaded durably to hopper when an uploader is configured.

### CLI features

- Terminal output for nested hostile archives renders each package's top trait findings inline.
- Dependency progress streams state each distinct row once instead of repeating.

### Optimizations

- Dependency fetches consult the corpus first.
- The scan→hopper upload protocol was simplified.

## 2.9.0 (2026-09-08)

### End-user impacting

- Default severity level is now 25, down from 50, and comes from the model bundle.
- The `--fetch` flag is now `--follow`, with `--follow-depth` controlling chained hops (compatibility alias preserved).
- The suspicious ceiling is a flat 3000 instead of 4 × level, so raising `-l` moves only the hostile line.

### Features

- New `/v1/lookup` takes batched queries and a caller-supplied FP budget, returning allow, block, unanalyzed, or unavailable.
- New `/v1/analyze` streams NDJSON progress and accepts `force=1` and a per-request follow policy.
- `GET /lookup` and `GET /status` answer without analyzing, even while the model is still loading.
- New `POST /analyze-purl` resolves the package and its registry provenance server-side.
- `--llm` takes a failover chain so you grade locally and fall back to a billed endpoint only on refusal.
- Added `--llm openrouter` with `openrouter/auto` routing, and no model is hardcoded anywhere.
- Bearer tokens are read from files under `~/.tok/`, never from argv or the environment.
- `--token-file` requires a bearer token on every route but `/_/health`, and loopback is not exempt.
- Added Cloudflare tunnel support plus `make deploy` for systemd, FreeBSD, and Bastille jails.
- Every request emits one structured access-log line with per-phase timings.
- Dependencies are skipped against hopper's shared corpus and not rescanned for 30 days.
- Windows gains mimalloc, physical-core sizing, safe extraction, and symlink handling.
- Warns when the host malloc is a debug build.

### Fixes

- Fixed two panics on multi-byte UTF-8 that killed the whole analysis.
- `/analyze` can now be forced past a cached verdict after an engine upgrade.
- Fixed OpenRouter auto-routes, orphaned LLM permits, and request gating.
- The interpret circuit breaker now probes with one caller after cooldown instead of all of them.
- Unretrievable artifacts are now distinct from server errors.
- Bytes carrying a mismatched or unverifiable pin are scanned instead of dropped.
- Fixed the default bind identity, hopper registry uploads, the upload API, and replica failover.
- Uploads no longer queue before a verdict exists, which had raced hopper's claim queue.

### Optimizations

- PGO in `make release` cuts 8% of wall time and 11% of CPU with identical output.
- Each payload gets a private thread pool sized by load, taking a 128-PURL sweep from 200s to 53s and p90 from 5.3s to 1.8s.
- Large archives take a bounded slot instead of oversubscribing the host.
- Idle pull-worker threads run at lower priority so background work never costs request latency.
- LLM admission adds a size veto and separate background concurrency, halving benign calls with no verdict changes.
- Worker latency is now a decayed percentile rather than a mean, so old incidents stop steering routing.
- Concurrent identical analyses collapse into a single run.
- Bloom filters moved to the burton crate with an LRU cache on the lookup path.

## 2.7.2 (2026-08-14)

### LLM interpretation

- No model name is hardcoded anymore. Qwen/Qwen3.6-27B was baked in as a last-resort default.
- If the endpoint lists nothing (or is unreachable) and no `--llm-model` is pinned, the scan fails up front with a clear message.
- Docs (`README.md`, `docs/INTEGRATION.md`) now describe discovery and recommend Qwen/Qwen3.8-27B rather than implying it's required.

### Release engineering

- Release matrix split into fast (native, Windows, cross — ~30 min) and slow (emulated/VM platforms — up to 75 min) waves.
- Publish logic extracted into a reusable publish-assets composite action shared by both waves.
- Windows ARM64 dropped from the default matrix.
- 32-bit ARM musl targets link `-lgcc` for the jemalloc intrinsics musl doesn't carry.

### Dependencies

- Picks up cleave 2.7.2 — including the Windows build fix and the content-hashed YARA rule fingerprint.
- Updated filefacts, fletch, stng 1.8.0 → 1.9.0, libredox 0.1.19 → 0.1.20.

## 2.7.1 (2026-08-14)

- Updated cleave engine and other dependencies.
- New exit code 4: the YARA rule set is incomplete or corrupt.

## 2.7.0 (2026-08-14)

### Better verdicts

- Every embedded file now gets graded — the old 100-member cap silently skipped deep archive entries, so a hostile file at index 347 never elevated its parent. It does now.
- Registry findings (e.g. security-withdrawn packages) are applied to fetched dependencies before grading, so a pulled npm package flagged by the registry shows that finding in its own report — not just the parent's.
- Duplicate dependency fetches reduced: a versionless purl is skipped when a lockfile-pinned version of the same package is present.

### Encrypted archives

- New repeatable `--zip-password` flag, plumbed through CLI, server, and workers. Passwords are redacted from logs and recorded command lines.

### LLM interpretation (`--interpret`)

- Any non-benign verdict now reaches the LLM — previously a container elevated to hostile by a member could ship uninterpreted if its raw probability was low.
- Prompts capped at the 2 riskiest dependency subjects, worst-first. Fixes 1.2 MB prompts (Kibana: 257 subjects) being rejected outright with a 400.
- 4xx error bodies are preserved, so a bad model name or over-budget prompt is diagnosable instead of a bare "400 Bad Request".

### Terminal output

- Containers with multiple notable packages render per-package cards, ranked worst-first, with an "N hostile · N suspicious · N clean inside" summary. Cards show the top 6 members plus anything referenced by a finding; the rest collapse to "+N quieter files".
- Dependency-tree rows show which manifest declared them (archive-relative path, e.g. `demo.zip/…/package.json`).
- Single-file scans get a spinner with elapsed time and the member currently being analyzed.

### Under the hood

- GitHub Actions workflows now tracked as CI dependencies.
- jemalloc allocator and its config unified in one place (config shared from cleave).

## 2.6.0 (2026-08-10)

### Fetch

- New `ci` kind: `--fetch ci` follows GitHub Actions dependencies (and implies `deps`).
- Workflow-declared references are skipped unless `ci` is on — they used to ride along with `deps`.
- Fetched artifact filenames are sanitized; a redirect picks the final URL segment, so `%2f`, RTL overrides, and control characters were attacker-reachable.
- A manifest declaring a hostile dependency carries `fetch/dependency-verdict`, rolled up to every containing archive.
- Listed-but-unanalyzed members report `risk: -1`, so silence isn't read as clean.

### Interpret

- ML decides, the LLM steers: at most one severity step, and at most 33% of the distance to the bound it argues for.
- Crossing into hostile also requires ML to already sit within one steer — that boundary spends the deploy level's FP budget. Suspicious stays ungated.
- A clearing verdict is distrusted when the render addresses the grader rather than a human, and marked `inject: true`.

### Correctness

- The cache key omitted the model bundle, so a model update reused the old namespace and every seen file kept the previous bundle's verdict.
- Admission budgets were `usize`, so a 10 GiB archive estimate saturated at 4 GiB on 32-bit targets.
- `validate` reports a log-odds margin; fixed-decimal probabilities can't express decisions at either end.

### Faster

- Featurizing reads the typed compact report directly, skipping a JSON DOM that ran multi-GB for a few-hundred-MB report.
- tree-sitter allocates through jemalloc — it was retaining ~0.8 GB invisible to every budget.

### Output

- Verdict stamp leads the headline — HOSTILE 100%, fixed-width so verdicts line up down a listing.
- LLM interpretation gets its own ✨ line instead of a dim trailer.
- Identity line carries the full SHA-256 and a bloom glyph: 🧬, 🚩 known-bad, 🏴 conflicted.

### Supply chain

- Releases ship a Sigstore bundle beside `SHA256SUMS` — keyless GitHub OIDC, verifiable with plain cosign.
- macOS binaries are Developer ID signed and notarized.

### Platforms

- New targets: armv7 and arm musleabihf, loongarch64-musl.
- Compatibility probes against published binaries: OpenIndiana, Tribblix, GhostBSD, openEuler, BlissOS.
- Haiku and GNU/Hurd build probes, opt-in — both Tier 3, no rustup toolchain.
- FreeBSD/arm64 shelved: 8h+ under TCG, needs hardware acceleration.
- `brew install atomdrift-project/tap/scan`; installer moved to its own repo, Wolfi packaging removed.

## 2.5.0 (2026-08-04)

### Fetch targeting

- Discovered URLs are filtered before the network phase.
- Only the newest version of a package in a dependency tree is fetched.
- Go pseudo-versions are dated from the version string, so the age gate rejects old modules with no round trip.
- Worker mode drops the age gate entirely.

### Provenance

- Registry metadata travels with the sample, so a hopper-sourced scan reasons over the same facts a live fetch would — no refetch.
- `--interpret` is organized per package, provenance before traits.
- `--hopper` works on url/pkg (or `SCAN_HOPPER`).

### Faster

- Registry records skip YARA — nothing targets canonical JSON we serialized ourselves.
- Registry lookups and dependency hops now run concurrently.
- Plain JSON scans skip the per-dependency model pass and other surfaces nothing reads.
- Worker startup indexing is threaded and logs progress.

### Output

- Stored reports keep only member nodes someone will read again: ~77% fewer nodes, ~56% fewer bytes.
- A file that fails to scan emits an error line instead of silently producing no record.

### Platforms

- FreeBSD: `MALLOC_CONF=background_thread:true` permanently breaks libc jemalloc init.
- New targets: DragonFlyBSD, NetBSD, illumos/OmniOS, Solaris 11.4.

## 2.4.0 (2026-07-25)

### Dependency scanning

- Fetching on by default (`--fetch=all`); every fetched dependency gets its own verdict, not a slice of its parent's.
- Worker mode scans all resolvable dependencies; interactive scans stay gated to fresh releases — now including correctly-dated Go pseudo-versions.

### Correctness

- Never posts a verdict it didn't compute — an unreached dependency no longer gets an invented "benign" that would bless a bad package.

### Scale & speed

- Files up to 16 GiB (streamed — a 16 GiB sample scans on an 8 GiB host).
- Lockfile-heavy scans faster: registry JSON no longer YARA-scanned (~20s saved on 851 records).
- Worker count back to physical (non-E) cores — better throughput on hybrid CPUs.

### LLM (optional)

- Legacy `--interpret` flag was renamed to `--llm`.
- Now auto-picks the largest model served.

### Project

- Rust 1.94, moved to GitHub, dependency updates.

## 2.3.0 (2026-07-11)

### Features

- `--format=interpret`: print the LLM prompt locally; no LLM call. Works with `scan` and `ps`.
- `SCAN_FORMAT` env var sets default output format.
- `SCAN_NO_ANALYSIS_CACHE=1` disables all analysis caches.
- Reports record provenance: registry sidecars, fetch URLs, embedded file id/sha256.
- Machine formats now emit all files; `--show` only filters the terminal view.

### Fixes

- Trait floor read the wrong JSON key and never fired on current cleave reports.
- Crit-5 traits now escalate a benign model verdict to hostile, not suspicious.
- Benign archive members no longer inherit their container's hostile verdict.
- Hostile transitive dependencies now surface at the archive root, not just the manifest.
- Unparseable PURLs no longer produce bloom-filter keys.
- All PURL spellings normalize to one canonical form.
- FreeBSD worker stop no longer hangs or orphans the scanning child.

### Optimizations

- 30-60 MB lower peak RSS via jemalloc arena cap.
- 800 MB less retained worker memory: dependency reports kept as text, not parsed trees.
- Capped uploader sha set; prune stale cache versions.
- Worker shutdown drain: 60s to 15s.

## 2.2.1 (2026-07-07)

- Performance improvements: fixed intermittent purl/url hangs.
- New `--registry-map FILE` flag.

## 2.2.0 (2026-07-05)

### Features

- Bloom-filter matching — scans consult bloom filters before doing work: known-good & known-bad, known-good samples are skipped by default. Tune with `--mode`: fast (bloom only), balanced (default), or slow (always full scan).
- Dependency fetching by default — interactive scans fetch and scan a sample's declared dependencies and install-command packages.
- Renamed CLI to atomscan (avoids clashing with avast's scan); `fs`→`path`, `pkg`→`purl` (old names aliased).
- New `purl` subcommand: `purl npm/left-pad@1.3.0` scans a package straight from its PURL.
- New `url` subcommand to scan a remote URL.
- Adaptive `--interpret` templates.
- Auto-refreshes rules/models when stale (`--no-update` to disable); `--hopper` upload in server mode; `--show=all` keeps all archive members.
- Handles registry records, provenance, and local references, including offline registry maps.

### Fixes

- Recalibrated thresholds (hostile L25, suspicious L3000); confident-but-opaque ML "hostile" held at suspicious, not cleared.
- Fixed bloom bucket loading, progress-bar glitches, FreeBSD build, and tar.gz extraction.
- Removed obsolete flags (`--slow-rule-ms`, `--models-repo`, `--no-update-check`).

### Performance

- Caches fetched-dependency analysis by content hash, skipping repeat disassembly across scans.
- Parallel dependency fetching and scanning.

## 2.1.2 (2026-06-22)

- Sync with cleave 2.1.2, which allows building under Rust 1.94 again.
- interpret: use `/` instead of `!!` in paths to avoid confusing the LLM.

## 2.1.0 (2026-06-22)

### Highlights for operators

- Experimental `--fetch` feature to fetch and scan remote dependencies.
- New `pkg` and `url` subcommands to scan remote samples directly.
- Hopper upload — `--hopper`/`--upload` pushes refreshed scan data upstream.
- Faster scans — dropped tract (CPU hotspot) for a leaner model engine; parallelized `ps` and specialist startup; dynamic admission sizing.
- v8 model migration — more reliable model releases, models pulled via git, per-file typing.
- Leaner output — slimmer JSON payloads from workers.
- Broader worker support — added SUSE, Void, and Fedora (on top of illumos, FreeBSD, Arch, etc).
- cleave v2.1.0, which has its own changelog: https://codeberg.org/atomdrift/cleave/releases/tag/v2.1.0

## 2.0.0 (2026-06-13)

### Since the last official release — v1.2.1

- Now generally available, running ONNX-only inference (pure Rust, no XGBoost/LightGBM) on the new azoth model.
- Rethought severity — scored as false positives per 100M files instead of a 1–9 scale; "suspicious" is now a bounded middle ground.
- LLM interpretation and tiny output, on top of a new analysis engine (cleave 2.0 / filefacts).
- Signed model and trait updates delivered as compressed bundles, replacing the old git-based updater.
- Steadier under load — memory-aware scheduling, smallest-job-first ordering, and larger thread stacks that fix archive crashes.
- Much deeper diagnostics — crash dumps, worker census, thread dumps, breadcrumbs.
- Rebrand — litmus is now Atomdrift Scan.

### Since 2.0.0-rc.5

- LLM second opinion — a new `--interpret` mode sends tricky samples to a local LLM for a verdict and one-line reason, then blends it with the ML score. Fails silently if the model is unreachable.
- Smarter job ordering — the worker now runs the smallest job first instead of first-in-first-out, so quick scans stop waiting behind slow archives.
- Better diagnostics — worker census, thread dumps, crash dumps, and tracing breadcrumbs for debugging stuck or aborted scans.
- New output — `--format=tiny` for LLM context; cleaner, more readable v7 JSON.
- Internal cleanup — core scan engine refactored and admission control simplified.

## 2.0.0-rc.5 (2026-06-08)

### Features

- Added R2-backed model updates with sha256 verification, staged load validation, atomic install, and update/check support.
- Added signed model publishing via compat-tested `.tar.zst` bundles for azoth.
- Updated to cleave 2.0.0-rc.5, filefacts 0.9.5, and stng 1.6.0 for newer package, PE/CLR, archive, and string signals.
- Added crash-time in-flight analysis dumps to identify likely files behind stack-overflow aborts.
- Added trait-floor escalation: confident high-severity cleave findings can raise a model-benign verdict to suspicious.
- Added severity-fraction aggregate features for sparse high-risk finding clusters.

### Performance

- Replaced git-based model and trait updates with compressed R2 bundles.
- Removed old git updater code and rollback paths; model bundles are validated before going live.
- Model publishing now reuses cleave's manifest generator with cached compatibility validation.

### Bugfixes

- Suspicious verdicts are now capped at level 20000, preventing very loose grid hits from becoming suspicious.
- Raised max CLI severity level to 25000 while keeping suspicious bounded.
- Preserves rare anti-analysis metrics like AST depth capping for model features.
- Validation output now reports level and decision threshold, making model failures easier to triage.
- Removed a temporary benign-sample validation waiver so validation reflects current model behavior.

## 2.0.0-rc.4 (2026-06-07)

### Severity levels & verdicts

- Severity levels are now expressed as false positives per 100M benign files (default L50); the old 1–9 scale is gone.
- Each model bundle ships its own default level; `-l` still overrides.
- Every file reports `l`: the strictest level at which it fires.
- "Suspicious" now means a file fires above the hostile budget but within the calibrated grid — no separate calibration.
- Output shows the matched level and raw route scores; calibrated percentages were dropped (they saturate to 1.0).

### Models

- ONNX is now the only model format; LightGBM and XGBoost loaders removed.
- `validate` fails bundles with feature-layout drift instead of silently scoring zeros; features are matched by name, so reordered specs can't misalign.
- Archive routing treats tar.gz as tar, matching training labels.
- New `validate --skip-traits` checks a model independently of trait updates.

### Worker

- Memory-aware admission: large archives run one at a time instead of co-residing and exhausting RAM.
- Analyses share one thread pool with larger stacks, fixing overflows under concurrent deep archives.
- Oversized results post the verdict without the raw report rather than losing the analysis.
- New heartbeat API reports liveness, load, queue depth, and throughput even while busy.
- Native FreeBSD deploy support; correct CPU detection on illumos/Solaris.
- `bench_hopper`: a local mock server for benchmarking the worker end to end.

### Other

- Daily zero-telemetry update check; a hosted manifest also pins the matching models ref.
- `scan` accepts files and directories together in one run.
- Dependency updates.

## 2.0.0-rc.3 (2026-05-28)

### Features

- The big change is feature extraction. litmus now computes the full set of fields the latest azoth model wants.
- The ML JSON output is now unambiguous. Schema v=5 replaces the old `oclass`/`oprob` and threshold pair with a single self-consistent decision.

### Fixes

- Repaired the json→model plumbing: reading the `ff`/`m`/`ms` fields out of filefacts had broken.
- `scan.rs` lost ~350 lines in the process; same behavior, less machinery.

### Operational

- The `WORKERS` knob is honored across all worker scripts (alpine, bastille, debian, macos, omnios, openbsd), with better update logging.
- `scan-no-deadlock` now emits logs so a hang is diagnosable.

## 2.0.0-rc.2 (2026-05-26)

### Behavior changes

- Traits-dir override uses the new cleave API instead of the `CLEAVE_TRAITS_DIR` env var.

### Build & deps

- Default release profile = thin LTO + `codegen-units = 16`.
- Edition bumped to Rust 2024, version → 2.0.0-rc.2.
- Removed local-dev `[patch]` overrides that pointed cleave and filefacts at sibling worktrees.
- Updated deps.

### Internal

- Bump Rust API to 2024.

## 2.0.0-rc.1 (2026-05-25)

### Breaking

- General availability: beta warning removed.
- Default model bundle moved from litmus-models to azoth.
- Worker `--nice` default 10 → 18.
- Linux worker `MEMORY_MAX` default 32G → 80%.

### Models & inference

- ONNX backend via tract-onnx (pure-Rust, no FFI). When present, `.onnx` is preferred over `.txt`/`.json`: ~50% smaller, 3–5× faster load, 5–11× faster inference.
- Blended-route policies: `route_policies.json` gains a blend severity (`sigmoid(intercept + Σ wᵢ · logit(pᵢ)) ≥ threshold`) alongside per-route OR rules.
- Model-load anomalies now log at error level so deploy verification catches them.

### Public API (Rust library)

- New `litmus::Analyzer`: reusable bundle of model + features + SHAP + config, with `scan_bytes` / `scan_path` and chainable threshold/slow-rule/heuristic setters.
- New free function `scan::scan_bytes` for in-memory payloads.
- New `ScanResult::top_traits(n)`, ranked by criticality × confidence with family dedup.
- `TopFinding` gained a `conf` field.

### Build & ops

- Release profile: `lto = "fat"`, `codegen-units = 1`.
- Git network timeout unified at 10 min (server outer: 21 min); fixes spurious timeouts on large pulls.
- illumos worker uses disk-backed `/var/litmus/tmp` instead of swap-backed `/tmp`.

### Packaging

- Wolfi/apko OCI pipeline under `packaging/wolfi/` (melange + apko + cosign; Lima VM on macOS). `make wolfi`, `make docker-publish`.
- cleave-tuna integration: `make bench-build`, `sampled-benchmark`, `heap-build`, `heap-benchmark`, `tuna`, `tuna-once`.

### Docs

- New `docs/INTEGRATION.md`, `docs/JSON.md`, `docs/SERVER_API.md`, `docs/WORKERS.md`.
- README rewritten around GA framing and azoth.

## 1.2.1 (2026-05-13)

### Features

- Workers now report litmus version, traits version, RSS, load, and available tools when polling Hopper.
- Worker resources refresh every 10 minutes instead of hourly.
- Model discovery now accepts multi-seed bundles that only provide `models/seed_*.{txt,json}`.
- OmniOS worker install improvements.

### Bugfixes

- Model repo mismatch or incomplete model bundles now fail with explicit instructions instead of deleting/recloning after a delay.
- Multi-seed ensemble layout detection no longer requires a legacy top-level model file.
- OmniOS setup only installs IPS/pkgsrc build prerequisites when a source build is actually needed.

## 1.2.0 (2026-05-10)

### Features

- Support for Azoth ensemble bundles — `general/` plus optional `filegroups/<name>/` and `filetypes/<name>/` specialists, routed per file from a top-level `config.json`.
- Multi-seed model averaging — `models/seed_NN.{txt,json}` collapse to a mean prediction; K=1 stays byte-identical to the old path.
- Per-route isotonic calibrators map raw probabilities to calibrated ones at load.
- LightGBM backend alongside XGBoost, picked by file extension.
- Tiered severity-prefixed n-gram vocabularies (`tierbi:`, `tiertri:`).
- Disable RSS throttling using `--max-rss-gb=-1` (for systemd-supervised workers).
- OmniOS/illumos worker — SMF manifest, `/opt` install, gmake in jails, rizin via `LD_LIBRARY_PATH`.
- Cleave 1.2 — Kotlin tree-sitter; validate runs with YARA/radare2/UPX off.

### Bugfixes

- Calibrator hardening — reject NaN/Inf, require strictly-ascending x, monotone y in [0, 1], re-validate the calibrated thresholds.
- Score mismatch fixed — feature vector ordering now matches training.
- Worker hints "peer did not speak TLS; try http://" on `InvalidContentType` instead of dumping a TLS error.
- Poll failures log URL, error chain, elapsed, backoff, consecutive errors, body excerpt.
- Git failures fall through stderr → stdout → exit code instead of going silent.
- `validate` miscategorizations are warnings, not fatal; output shows the active traits dir.
- Specialist load failures log the full error chain, not just the top message.

## 1.1.0 (2026-04-28)

### Features

- New worker mode — pulls jobs from a hopper queue for distributed scanning.
- New `validate` subcommand — checks rules + model and re-scans known-benign samples.
- New `tools` subcommand and `--extract-dir` for writing archive members to disk.
- Gzip-style sensitivity flags `-1` through `-9` (default `-5`).
- Criticality upgrade-heuristic auto-promotes ML verdicts when cleave findings disagree (`--upgrade-heuristic`).
- Server: new `/analyze-path` and `/_/info` endpoints; `--allow-cidr` ACLs; per-request `--timeout-secs`.
- v16 feature plumbing: name-based feature specs, metric/crit/ATT&CK/MBC n-gram vocabularies.
- `version --format=json`.

### Performance

- Prefetch cleave shared resources at startup to avoid rayon-init YARA deadlock.
- Stack size raised to 32–64 MB to prevent archive-extraction panics.
- jemalloc tuned to aggressively return freed pages.
- Default nice 10 to play well with co-tenant processes.
- Simpler more scalable rayon pooling; `MIN_THREADS_PER_SLOT=4`.

### Bugfixes

- Cooperative 10-minute cancellation that signals cleave to abort slow YARA rules.
- Safe TTY/theme detection that doesn't block on unattended stderr.
- Git pulls bounded by 60s inner / 150s outer timeouts; auto-revert on failed rule update.
- Aggressive rizin subprocess kill on exit.
- v16 feature-parity fixes: missing aggregate features, `score:density` computation, boolean metric handling.
- Resign macOS binary to avoid SIGKILL after install.
- Linux `make install` works around "text file busy".

## 1.0.0 (2026-03-26)

### Features

- Upgraded to the v15 model/feature pipeline, with stronger aggregate risk signals and a new stealth-oriented structural signal.
- Thresholds now come from the model bundle by default, with config and CLI overrides still supported.

### Bugfixes

- Fixed threshold handling across CLI, process scans, and server mode so classification uses model settings consistently.
- Added stricter model compatibility checks and clearer failures for incomplete or mismatched model bundles.

### Performance

- Improved server startup by warming model, SHAP, and YARA resources in parallel.
- Added bounded analysis concurrency and reduced first-request latency.

## 0.5.0 (2026-03-23)

### New Features

- Multi-path scanning — `litmus scan` accepts multiple paths in one invocation.
- Archive unpacking — Embedded files in archives are classified individually; the parent's score is elevated if an inner file scores higher.
- Feature spec v14 — 12 new finding-density signals, multi-file aggregation, and multi-hot file type encoding.
- jemalloc — Default allocator on Unix (except FreeBSD/DragonFly) for better performance.
- Progress bar for `ps` scans — Same spinner/ETA as file scans.
- `make install` — Auto-detects the best bin directory.

### Bug Fixes

- Server no longer exits on sustained memory pressure — keeps running, rejects requests until RSS recovers.
- Model repo was checking for `model.onnx` instead of `model.json`.
- Added `sync_all()` on uploaded temp files before analysis, preventing async write races.
- `has_yara` feature narrowed to `third_party/yara*` only, not all `third_party/*`.
- Dead features (zero std) are now zeroed to prevent NaN/inf propagation.
- SHAP sorting uses `total_cmp` instead of `partial_cmp`, fixing NaN ordering.

### Optimizations

- SHA256 hashing uses `std::io::copy` instead of manual read loops.
- Single `classify_report()` path replaces three separate inference copies across scan, ps, and server.
- `PsConfig` removed — process scanning shares `ScanConfig`.

### Breaking Changes

- Default thresholds lowered: suspicious 0.97→0.65, hostile 0.99→0.887.
- `Thresholds` validates at construction (range, ordering) with a proper error type.
- `FeatureSpec`, `Model`, `ScanConfig`, `ServerConfig`, `DisplayFilter` fields are private — use accessors.

## 0.2.0 (2026-03-16)

### New Features

- Server mode — Run litmus as a persistent HTTP server with readiness checks and lame duck mode for faster startup.
- Monitoring endpoints — New diagnostic/health endpoints.
- `-u` update flag — Update rules inline from CLI.
- Model data in JSON output — Include model metadata in JSON responses.
- v13 model upgrade — Updated ML model to v13.
- Verbose logging — Added `--verbose` flag with structured logging support.

### Performance

- Improved memory limits — Better memory management under load.
- Faster startup via lame duck mode during initialization.
- XGBoost native upgrade — Updated xgboost-native dependency.

### Bug Fixes

- Ensure output is always provided (no silent empty results).
- Clean up sus/hostile detection thresholds.

### Infrastructure

- Lint rules added — Enforced clippy/formatting standards.
- Tests run before rollout — CI now gates on tests passing.
- Bastille port changed to 8880.
- ORT runtime — Resolved ONNX runtime linking (forked ort, settled on load-dynamic).
