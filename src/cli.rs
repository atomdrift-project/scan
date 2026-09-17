//! The command-line flags every Atomdrift Scan front-end shares.
//!
//! [`GlobalArgs`] holds the `global = true` options of the `atomscan` binary —
//! the LLM, follow/fetch, threshold, display and model-refresh knobs that apply
//! to every subcommand. It lives in the library, not in `src/main.rs`, so a
//! second binary that re-exposes `serve`/`worker` can `#[command(flatten)]` the
//! same declarations instead of restating them. Restating them is how two
//! front-ends silently acquire different defaults; there is exactly one copy of
//! each flag, here.
//!
//! Adding a flag here adds it to every binary that flattens this struct, and
//! renaming one is a user-visible change to all of them.

// Doc comments on `clap` structs are user-facing `--help` text, so they carry
// `[EXPERIMENTAL]` tags, bare URLs, and `<URL>` placeholders that rustdoc would
// otherwise read as broken links or markup.
#![allow(
    rustdoc::broken_intra_doc_links,
    rustdoc::bare_urls,
    rustdoc::invalid_html_tags
)]

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::Result;

use crate::OutputFormat;

/// Warn threshold for a single slow cleave rule (ms); was the `--slow-rule-ms`
/// flag, now a fixed advisory default. Shared, so every binary running this
/// analysis stack warns at the same point rather than each picking a number.
pub const DEFAULT_SLOW_RULE_MS: u64 = 4000;

/// Default hard wall-clock limit for each Rizin subprocess, in seconds.
pub const DEFAULT_RIZIN_TIMEOUT_SECS: u64 = 10 * 60;

/// Classification values accepted by `--show`.
//
// The variants are deliberately left undocumented: clap's `ValueEnum` derive
// turns a variant doc comment into that value's help text, which would switch
// `--show`'s `[possible values: ...]` line into a multi-line list. That is
// user-visible help output, so the lint yields to it here.
#[allow(missing_docs)]
#[derive(Debug, Clone, clap::ValueEnum)]
pub enum Show {
    Hostile,
    #[value(name = "sus", alias = "suspicious")]
    Sus,
    Benign,
    All,
}

/// Every `global = true` flag of the `atomscan` command line.
///
/// Flatten it into a `clap::Parser` with `#[command(flatten)]` to give a binary
/// the identical set of global options:
///
/// ```no_run
/// #[derive(clap::Parser)]
/// struct Cli {
///     #[command(flatten)]
///     global: scan::cli::GlobalArgs,
/// }
/// ```
//
// `about`/`long_about` are reset because `clap`'s `Args` derive would otherwise
// promote this struct's doc comment onto whichever command flattens it,
// replacing that command's own description in `--help`.
#[derive(Debug, clap::Args)]
#[command(about = None, long_about = None)]
pub struct GlobalArgs {
    /// Enable debug logging for Atomdrift Scan and cleave
    #[arg(long, global = true)]
    pub verbose: bool,

    /// Update models and traits before running (failures are non-fatal)
    #[arg(short = 'u', long, global = true)]
    pub update: bool,

    /// Disable the automatic rules/models refresh (on by default when the local
    /// ruleset is over 24h stale). Use when the local traits/models are
    /// intentionally ahead of (or diverged from) the remote, e.g. local edits
    /// that would block the pull. Also settable via `SCAN_NO_UPDATE`.
    #[arg(long, global = true)]
    pub no_update: bool,

    /// Force light-background color theme
    #[arg(long, global = true, conflicts_with = "dark")]
    pub light: bool,

    /// Force dark-background color theme
    #[arg(long, global = true, conflicts_with = "light")]
    pub dark: bool,

    /// Override model directory (default: auto-resolved from models repo)
    #[arg(long, global = true)]
    pub model_dir: Option<PathBuf>,

    /// Output format
    #[arg(
        short,
        long,
        global = true,
        env = "SCAN_FORMAT",
        default_value = "terminal"
    )]
    pub format: OutputFormat,

    /// Scan mode: `fast` (bloom matching only), `balanced` (bloom short-circuits,
    /// then full scan), or `slow` (no bloom; always full scan). Workers are
    /// always slow.
    #[arg(long, global = true, default_value = "balanced")]
    pub mode: crate::Mode,

    /// Hard wall-clock limit for each Rizin subprocess, in seconds. On expiry
    /// Atomscan kills and reaps Rizin before releasing the analysis worker; on
    /// Unix it also kills the complete process group. Also settable via
    /// `SCAN_RIZIN_TIMEOUT_SECS`.
    #[arg(
        long,
        global = true,
        value_name = "SECS",
        env = "SCAN_RIZIN_TIMEOUT_SECS",
        default_value_t = DEFAULT_RIZIN_TIMEOUT_SECS,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    pub rizin_timeout_secs: u64,

    /// Override suspicious threshold (0.0-1.0); omit to use model's recommendation
    #[arg(long, global = true)]
    pub threshold_suspicious: Option<f32>,

    /// Override hostile threshold (0.0-1.0); omit to use model's recommendation
    #[arg(long, global = true)]
    pub threshold_hostile: Option<f32>,

    /// Tune thresholds for false-positive level N (0-25000, FP per 100M benigns): higher = more sensitive, noisier. Bundle decides which levels are calibrated.
    #[arg(
        short = 'l',
        long,
        value_name = "N",
        value_parser = clap::value_parser!(u16).range(0..=25000),
        global = true,
        conflicts_with_all = ["threshold_suspicious", "threshold_hostile"],
    )]
    pub level: Option<u16>,

    /// Classifications to display in the terminal view: hostile, suspicious,
    /// sus, benign, all (comma-separated). The machine formats (json, tiny,
    /// interpret) emit every scanned file regardless.
    #[arg(long, global = true, value_delimiter = ',', default_values = ["hostile", "sus"])]
    pub show: Vec<Show>,

    /// Show raw probability and SHAP feature values in terminal output
    #[arg(long, global = true, hide = true)]
    pub extra: bool,

    /// [deprecated] Legacy on-switch for LLM interpretation; superseded by
    /// `--llm`. Kept for compatibility (env: SCAN_INTERPRET).
    #[arg(long, global = true, env = "SCAN_INTERPRET", hide = true)]
    pub interpret: bool,

    /// [optional] Enable additional LLM interpretation of analyzed samples: a
    /// second opinion blended with the ML verdict (stored in the `llm` JSON
    /// section and shown inline). Given with no value, uses a local model (an
    /// OpenAI-compatible endpoint at http://localhost:8000/v1). TARGET may be
    /// `local`, `openrouter` (https://openrouter.ai/api/v1; key from `--llm-key`,
    /// `SCAN_LLM_KEY`, or `~/.tok/openrouter`; defaults to `openrouter/auto`
    /// unless `--llm-model` names one), or an explicit OpenAI-compatible base
    /// URL. Endpoints that require a bearer
    /// token take it from `~/.tok/llm` when that file exists. Comma-separate
    /// several to fail over in order, e.g.
    /// `https://llm.isotope13.ai/v1,openrouter`. (env: SCAN_LLM)
    #[arg(
        long,
        global = true,
        value_name = "TARGET",
        num_args = 0..=1,
        default_missing_value = "local",
    )]
    pub llm: Option<String>,

    /// LLM model name, e.g. Qwen/Qwen3.8-27B. Defaults to the largest model the
    /// endpoint itself reports serving, or `openrouter/auto` for an OpenRouter
    /// endpoint; nothing else is hardcoded. With a comma-separated `--llm`
    /// chain, comma-separate one name per endpoint in the same order (a blank
    /// slot discovers/defaults, a single name applies to all)
    /// (env: SCAN_LLM_MODEL)
    #[arg(long, global = true, value_name = "NAME")]
    pub llm_model: Option<String>,

    /// LLM bearer token (env: SCAN_LLM_KEY). Defaults to `~/.tok/llm` when that
    /// file exists (`~/.tok/openrouter` for OpenRouter); omit for an endpoint
    /// that needs no key
    #[arg(long, global = true, value_name = "KEY")]
    pub llm_key: Option<String>,

    /// Loosest FP level (0-25000, per 100M benigns) at which ML alone sends a
    /// sample to the LLM; higher = more samples. Defaults to the model's grid
    /// ceiling — anything ML flagged at any level. Files cleave flagged
    /// suspicious/hostile are sent regardless of this cutoff.
    #[arg(
        long,
        global = true,
        alias = "interpret-min-level",
        value_parser = clap::value_parser!(u16).range(0..=25000),
        value_name = "N",
    )]
    pub llm_min_level: Option<u16>,

    /// Raw ML probability at or above which a sample reaches the LLM, whatever
    /// the calibrated level grid and cleave findings say.
    ///
    /// One of several independent admissions, so it can only send more, never
    /// block. The default sits on a measured elbow: malicious admission is flat
    /// across 0.46..0.57 while benign admission keeps falling through it.
    /// Lowering it buys the overlap region at roughly 50 benign samples per
    /// malicious one; raising it sheds about four benign per malicious.
    #[arg(
        long,
        global = true,
        alias = "interpret-min-prob",
        value_name = "P",
        default_value_t = crate::interpret::DEFAULT_LLM_MIN_PROB,
    )]
    pub llm_min_prob: f32,

    /// Size veto for the LLM gate: an ML-benign sample with no hostile finding
    /// and more than this many notable-or-above findings is not sent to the
    /// LLM, unless it carries a strong gate trait or ML placed it on the level
    /// grid. Big legitimate packages accumulate notable findings until some
    /// admission fires, and the reader then moves nothing; on the measured
    /// corpus the veto at 300 halved benign calls and kept every correct
    /// shift. `0` disables it.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = crate::interpret::DEFAULT_LLM_BENIGN_NOTABLE_CAP,
    )]
    pub llm_benign_notable_cap: usize,

    /// Per-request LLM timeout, in seconds. Once it elapses the endpoint is
    /// treated as a refusal and the next one in the `--llm` chain is tried.
    ///
    /// Applies to every mode and every hop in the chain; an OpenRouter hop
    /// always gets at least 30.
    #[arg(
        long,
        global = true,
        value_name = "SECS",
        default_value_t = crate::interpret::DEFAULT_TIMEOUT_SECS
    )]
    pub llm_timeout: u64,

    /// Additional passwords to try for encrypted ZIP/7z archives. Repeat the
    /// option to provide more than one; cleave's common defaults remain active.
    #[arg(long = "zip-password", value_name = "PASSWORD", global = true)]
    pub zip_passwords: Vec<String>,

    /// [EXPERIMENTAL] Follow references discovered inside the requested
    /// artifact, analyze their payloads, and fold them into the verdict:
    ///   `dependencies` — manifest and lockfile dependencies;
    ///   `references`   — packages and URLs named by install/download commands;
    ///   `ci-actions`   — third-party actions referenced by CI configuration.
    /// `all` selects every category; `none` analyzes only the requested artifact.
    /// A bare `--follow` follows dependencies and references but not CI actions,
    /// and so does an absent one for an interactive scan. `serve` and `worker`
    /// default to `all` instead: they populate the shared corpus, where a
    /// category nobody followed is one nobody ever learns. The old `--fetch` flag and `deps`, `packages`, `urls`, and
    /// `ci` values remain accepted as aliases. Also settable via `SCAN_FOLLOW`;
    /// `SCAN_FETCH` remains a compatibility alias.
    #[arg(
        long = "follow",
        visible_alias = "fetch",
        global = true,
        value_name = "TARGETS",
        num_args = 0..=1,
        require_equals = true,
        // A bare `--follow` selects the artifact's own reachable code, not CI:
        // `--follow=all` (or `--follow=ci-actions`) is the explicit opt-in for GitHub
        // Actions, which run only in CI and never reach an installed artifact.
        // Keep in lockstep with `default_cli_follow_policy`, which resolves an
        // absent `--follow` to the same targets.
        default_missing_value = "references,dependencies",
        env = "SCAN_FOLLOW"
    )]
    pub follow: Option<crate::fetch::FetchPolicy>,

    /// [EXPERIMENTAL] How many hops of references to follow when `--follow` is on:
    /// `1` fetches only what the scanned files reference, `2` also follows
    /// references found inside those payloads (reaching a stage-3 `curl | bash`
    /// dropper), and so on. Also settable via `SCAN_FOLLOW_DEPTH`; the old
    /// `--fetch-depth` and `SCAN_FETCH_DEPTH` names remain aliases.
    #[arg(
        long = "follow-depth",
        visible_alias = "fetch-depth",
        global = true,
        value_name = "N",
        default_value_t = crate::fetch::DEFAULT_FETCH_DEPTH,
        env = "SCAN_FOLLOW_DEPTH"
    )]
    pub fetch_depth: u8,

    /// [EXPERIMENTAL] Skip fetching a declared dependency whose registry publish
    /// date is older than this many days — the cheap provenance lookup runs
    /// first, and only recent (freshest-risk) releases are pulled and scanned.
    /// Applies to declared dependencies only; URLs are never age-gated. `0`
    /// disables the gate (fetch every resolvable dependency). A dependency whose
    /// age can't be determined is always fetched. Also settable via
    /// `SCAN_FETCH_MAX_AGE`.
    ///
    /// Unset, the ceiling depends on the mode: an interactive scan wants a
    /// fresh-risk window ([`scan::fetch::DEFAULT_MAX_DEP_AGE_DAYS`]), while a
    /// worker is a cache-population role and takes every resolvable dependency
    /// ([`WORKER_MAX_DEP_AGE_DAYS`]). Optional rather than defaulted so an
    /// explicit `--fetch-max-age` still wins in both.
    #[arg(long, global = true, value_name = "DAYS", env = "SCAN_FETCH_MAX_AGE")]
    pub fetch_max_age: Option<u32>,

    /// [EXPERIMENTAL] Fetch and scan native-binary dependencies for every
    /// platform, not just the host's. This is automatic in `serve` and
    /// `worker`, which scan on behalf of other machines; interactive scans
    /// stay host-only for latency unless this flag is passed. Also settable via
    /// `SCAN_FETCH_ALL_PLATFORMS`.
    #[arg(
        long,
        global = true,
        env = "SCAN_FETCH_ALL_PLATFORMS",
        conflicts_with = "fetch_host_platform_only"
    )]
    pub fetch_all_platforms: bool,

    /// [EXPERIMENTAL] Fetch only native-binary dependencies matching this
    /// host's OS and architecture. This is the interactive default and an
    /// explicit completeness opt-out for `serve` / `worker`. Also settable
    /// via `SCAN_FETCH_HOST_PLATFORM_ONLY`.
    #[arg(
        long,
        global = true,
        env = "SCAN_FETCH_HOST_PLATFORM_ONLY",
        conflicts_with = "fetch_all_platforms"
    )]
    pub fetch_host_platform_only: bool,

    /// [EXPERIMENTAL] Follow declared dependencies past the first hop — the
    /// dependencies of a fetched dependency, out to `--fetch-depth`. This is
    /// automatic in `serve` and `worker`, which populate the shared corpus;
    /// an interactive scan stops declared dependencies at the first hop, since
    /// the transitive tail costs a registry lookup each and is almost entirely
    /// old releases the age gate then discards. URLs and command-mentioned
    /// packages — the dropper chain — are followed at every hop either way.
    /// Also settable via `SCAN_FETCH_TRANSITIVE_DEPS`.
    #[arg(
        long,
        global = true,
        env = "SCAN_FETCH_TRANSITIVE_DEPS",
        conflicts_with = "fetch_direct_deps_only"
    )]
    pub fetch_transitive_deps: bool,

    /// [EXPERIMENTAL] Follow declared dependencies for one hop only. This is the
    /// interactive default and an explicit completeness opt-out for `serve` /
    /// `worker`. Also settable via `SCAN_FETCH_DIRECT_DEPS_ONLY`.
    #[arg(
        long,
        global = true,
        env = "SCAN_FETCH_DIRECT_DEPS_ONLY",
        conflicts_with = "fetch_transitive_deps"
    )]
    pub fetch_direct_deps_only: bool,

    /// [EXPERIMENTAL] How long to trust cached *mutable* registry metadata before
    /// revalidating. Accepts a unit suffix (`90s`, `30m`, `4h`, `2d`) — a bare
    /// number is seconds; `never` caches indefinitely (offline/air-gapped). This
    /// bounds the two mutable tiers: a pinned version's packument (whose yank
    /// status can change after publish) and a `latest`/versionless lookup.
    /// Unset keeps the defaults — 4h pinned, 1h unpinned. A released version's
    /// immutable file list is never re-checked regardless. Also settable via
    /// `SCAN_REGISTRY_TTL`.
    #[arg(
        long,
        global = true,
        value_name = "DUR",
        value_parser = crate::fetch::parse_duration,
        env = "SCAN_REGISTRY_TTL"
    )]
    pub registry_ttl: Option<std::time::Duration>,

    /// [EXPERIMENTAL] Size cap for a single downloaded artifact. Accepts a unit
    /// suffix (`256M`, `2G`, `512K`); a bare number is bytes. A response larger
    /// than this is abandoned, so one artifact can't dominate a run. Also
    /// settable via `SCAN_FETCH_MAX_SIZE`.
    #[arg(
        long,
        global = true,
        value_name = "SIZE",
        default_value = "256M",
        value_parser = crate::fetch::parse_bytes,
        env = "SCAN_FETCH_MAX_SIZE"
    )]
    pub fetch_max_size: u64,

    /// [EXPERIMENTAL] Maximum number of *live* dependency/package fetches
    /// triggered by a single scanned file. This is 100 by default. Cache hits
    /// are always served and never counted, so a warm re-run is never throttled.
    /// References past the cap are recorded as budget-exceeded, never silently
    /// dropped. Also settable via `SCAN_FETCH_MAX_FILE_FETCHES`.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = crate::fetch::DEFAULT_MAX_FILE_FETCHES,
        env = "SCAN_FETCH_MAX_FILE_FETCHES"
    )]
    pub fetch_max_file_fetches: usize,

    /// [EXPERIMENTAL] Maximum number of *live* opportunistic raw-URL fetches
    /// triggered by a single scanned file. This is 4 by default. URL references
    /// declared as dependencies or command-mentioned packages use the larger
    /// `--fetch-max-file-fetches` cap instead. Also settable via
    /// `SCAN_FETCH_MAX_URLS`.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = crate::fetch::DEFAULT_MAX_URL_FETCHES,
        env = "SCAN_FETCH_MAX_URLS"
    )]
    pub fetch_max_urls: usize,

    /// [EXPERIMENTAL] Maximum total bytes fetched on behalf of a single scanned
    /// file. Accepts a unit suffix (`2G`); a bare number is bytes. Also settable
    /// via `SCAN_FETCH_MAX_FILE_SIZE`.
    #[arg(
        long,
        global = true,
        value_name = "SIZE",
        default_value = "2G",
        value_parser = crate::fetch::parse_bytes,
        env = "SCAN_FETCH_MAX_FILE_SIZE"
    )]
    pub fetch_max_file_size: u64,

    /// [EXPERIMENTAL] Wall-clock ceiling on the fetch phase for a single
    /// scanned artifact. Accepts a unit suffix (`90s`, `5m`, `1h`); a bare
    /// number is seconds; `0` or `never` disables the cap. This is 5 minutes by
    /// default. The count and size budgets bound how much a scan fetches, not
    /// how long fetching takes — a wide tree of slow registries can hold a scan
    /// open with every count budget still unspent. References not reached
    /// before the cap are left unfollowed; whatever was already fetched is
    /// analyzed and graded as usual. Also settable via `SCAN_FETCH_TIMEOUT`.
    #[arg(
        long,
        global = true,
        value_name = "DUR",
        default_value = "5m",
        value_parser = crate::fetch::parse_duration,
        env = "SCAN_FETCH_TIMEOUT"
    )]
    pub fetch_timeout: std::time::Duration,

    /// [EXPERIMENTAL] Maximum number of *live* fetches across the whole
    /// execution — a hard ceiling over every scanned file combined. Lifted in
    /// long-lived server modes (`serve`/`worker`), where the per-file caps bound
    /// each job instead. Also settable via `SCAN_FETCH_MAX_TOTAL_FETCHES`.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = crate::fetch::DEFAULT_MAX_TOTAL_FETCHES,
        env = "SCAN_FETCH_MAX_TOTAL_FETCHES"
    )]
    pub fetch_max_total_fetches: usize,

    /// [EXPERIMENTAL] Maximum total bytes fetched across the whole execution.
    /// Accepts a unit suffix (`10G`); a bare number is bytes. Lifted in
    /// long-lived server modes (`serve`/`worker`). Also settable via
    /// `SCAN_FETCH_MAX_TOTAL_SIZE`.
    #[arg(
        long,
        global = true,
        value_name = "SIZE",
        default_value = "10G",
        value_parser = crate::fetch::parse_bytes,
        env = "SCAN_FETCH_MAX_TOTAL_SIZE"
    )]
    pub fetch_max_total_size: u64,
}

impl GlobalArgs {
    /// Build the LLM interpretation config from `--llm` (or the legacy
    /// `--interpret`) and the `--llm-*` flags, falling back to env vars. `None`
    /// when interpretation is not requested.
    ///
    /// # Errors
    ///
    /// Returns an error when `--llm` names no endpoint, when the only
    /// endpoint requested is unusable (an OpenRouter target with no key, or
    /// one whose model can be neither pinned nor discovered), or when every
    /// endpoint in a failover chain was dropped for those reasons.
    pub fn interpret_config(&self) -> Result<Option<crate::interpret::InterpretConfig>> {
        use crate::interpret::{
            DEFAULT_BASE_URL, LlmEndpoint, is_openrouter_endpoint, llm_key_from_home, llm_models,
            llm_targets, openrouter_key_from_home,
        };
        let from_env = |flag: &Option<String>, key: &str| -> Option<String> {
            flag.clone()
                .or_else(|| std::env::var(key).ok())
                .filter(|s| !s.is_empty())
        };
        // `--llm [TARGET]` / SCAN_LLM (the bare flag defaults TARGET to `local`)
        // or the legacy `--interpret` flag turns the pass on.
        let target = from_env(&self.llm, "SCAN_LLM");
        if target.is_none() && !self.interpret {
            return Ok(None);
        }
        // Resolve the target to base URLs: `local` (also the bare-flag default)
        // maps to the local endpoint; `openrouter` is the public API; anything
        // else is an OpenAI-compatible base URL. A comma-separated target is a
        // failover chain, tried in order.
        let targets = match target.as_deref() {
            None => vec![DEFAULT_BASE_URL.to_string()],
            Some(raw) => llm_targets(raw),
        };
        if targets.is_empty() {
            anyhow::bail!("--llm (env: SCAN_LLM) names no endpoint");
        }
        let pinned = llm_models(
            from_env(&self.llm_model, "SCAN_LLM_MODEL").as_deref(),
            targets.len(),
        );
        // An explicit key wins, and applies to every endpoint in the chain —
        // it is the operator naming one credential. Otherwise each endpoint
        // resolves its own, and only its own: `~/.tok/openrouter` for
        // OpenRouter, `~/.tok/llm` for everything else — our own vLLM requires
        // one, and a host that has the file authenticates without any flag.
        // Absent a file, the request goes out unauthenticated, which is still
        // right for an endpoint that wants no key.
        let explicit_key = from_env(&self.llm_key, "SCAN_LLM_KEY");

        // One endpoint must work; the rest are a cushion. So a config problem
        // is fatal when it is the only endpoint (a misconfigured `--llm` must
        // not be silent), and a warning when others remain — an OpenRouter
        // fallback with no model pinned, or a primary that is down at startup,
        // should cost that entry, not the scan.
        let single = targets.len() == 1;
        let mut resolved: Vec<LlmEndpoint> = Vec::with_capacity(targets.len());
        let mut skipped: Vec<String> = Vec::new();
        for (base_url, pinned) in targets.into_iter().zip(pinned) {
            let openrouter = is_openrouter_endpoint(&base_url);
            // `~/.tok/llm` is *our* endpoint's token and must never travel to
            // a third party, so OpenRouter takes its own file or nothing —
            // sending the vLLM key there would hand a working credential to an
            // unrelated host and read as a plain 401 when it did.
            let api_key = explicit_key.clone().or_else(|| {
                if openrouter {
                    openrouter_key_from_home()
                } else {
                    llm_key_from_home()
                }
            });
            if openrouter && api_key.is_none() {
                let why =
                    "OpenRouter requires a key: --llm-key, SCAN_LLM_KEY, or ~/.tok/openrouter";
                if single {
                    anyhow::bail!("{why}");
                }
                skipped.push(format!("{base_url}: {why}"));
                continue;
            }
            // A pinned model wins; otherwise take what the endpoint says it
            // serves. OpenRouter's catalog is large and billed, so nothing
            // from it is guessed — but OpenRouter itself ships a stable
            // `openrouter/auto` alias that picks a suitable model per
            // request, so that's the default rather than a hard error.
            // Nothing else is hardcoded: if a non-OpenRouter endpoint lists no
            // model there is nothing sensible to send, and a guessed name
            // would surface as an opaque server-side error mid-scan instead of
            // here.
            let model = if openrouter {
                pinned.unwrap_or_else(|| crate::interpret::OPENROUTER_DEFAULT_MODEL.to_string())
            } else {
                match pinned {
                    Some(m) => m,
                    None => match crate::interpret::discover_model(&base_url, api_key.as_deref()) {
                        Ok(m) => m,
                        // Say which of the several ways discovery can fail this
                        // was — an unreachable host, a 404 from a base URL
                        // missing its /v1, a rejected key and an endpoint
                        // serving nothing all need different fixes, and only
                        // pinning a model is common to all of them.
                        Err(e) => {
                            let why = format!(
                                "no LLM model available from {base_url}: {e}. Fix the endpoint, \
                                 or name a model with --llm-model (env: SCAN_LLM_MODEL)"
                            );
                            if single {
                                anyhow::bail!("{why}");
                            }
                            skipped.push(why);
                            continue;
                        }
                    },
                }
            };
            resolved.push(LlmEndpoint {
                base_url,
                model,
                api_key,
            });
        }
        for why in &skipped {
            tracing::warn!("LLM endpoint unusable, dropped from the failover chain: {why}");
        }
        let mut resolved = resolved.into_iter();
        let primary = resolved.next().ok_or_else(|| {
            anyhow::anyhow!("no usable LLM endpoint:\n  {}", skipped.join("\n  "))
        })?;
        Ok(Some(crate::interpret::InterpretConfig {
            base_url: primary.base_url,
            model: primary.model,
            api_key: primary.api_key,
            min_level: self.llm_min_level,
            min_prob: self.llm_min_prob,
            benign_notable_cap: self.llm_benign_notable_cap,
            // One budget per hop for every mode: a wedged endpoint is caught
            // by the connect timeout and the breaker, not by this.
            timeout: std::time::Duration::from_secs(self.llm_timeout),
            // `SCAN_LLM_CONCURRENCY` overrides the in-flight cap; the default
            // scales with the box (see `interpret::default_max_concurrency`).
            max_concurrency: std::env::var("SCAN_LLM_CONCURRENCY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .and_then(NonZeroUsize::new)
                .unwrap_or_else(crate::interpret::default_max_concurrency),
            fallbacks: resolved.collect(),
        }))
    }

    /// Whether native-binary dependencies are fetched for this host only.
    ///
    /// `scans_for_other_hosts` is true for the corpus-facing roles (`serve`,
    /// `worker`), which scan on behalf of other machines and so take every
    /// platform by default. The two flags conflict, so at most one arm can fire.
    #[must_use]
    pub fn host_platform_only(&self, scans_for_other_hosts: bool) -> bool {
        self.fetch_host_platform_only || (!self.fetch_all_platforms && !scans_for_other_hosts)
    }

    /// Whether declared dependencies are followed past the first hop. Corpus-facing
    /// modes take the transitive tail by default; an interactive scan stops at the
    /// artifact's own declared dependencies. Either default is overridable, and the
    /// two flags conflict, so at most one arm can fire.
    #[must_use]
    pub fn transitive_deps(&self, scans_for_other_hosts: bool) -> bool {
        self.fetch_transitive_deps || (!self.fetch_direct_deps_only && scans_for_other_hosts)
    }

    /// Apply the follow/fetch knobs to `policy`.
    ///
    /// The selection comes from `--follow`; the hop count from `--follow-depth`,
    /// the dependency age ceiling from `--fetch-max-age`, and the per-file ceilings
    /// from `--fetch-max-file-*` (each its own flag/env). `default_max_age` is the
    /// ceiling used when `--fetch-max-age` is unset — an interactive scan wants a
    /// fresh-risk window, a worker takes every resolvable dependency. An explicit
    /// selection is honored verbatim; the knobs always apply.
    #[must_use]
    pub fn fetch_policy(
        &self,
        mut policy: crate::fetch::FetchPolicy,
        default_max_age: u32,
        scans_for_other_hosts: bool,
    ) -> crate::fetch::FetchPolicy {
        policy.depth = self.fetch_depth;
        policy.max_dep_age_days = self.fetch_max_age.unwrap_or(default_max_age);
        policy.max_file_fetches = self.fetch_max_file_fetches;
        policy.max_url_fetches = self.fetch_max_urls;
        policy.max_file_bytes = self.fetch_max_file_size;
        policy.max_duration = self.fetch_timeout;
        policy.host_platform_only = self.host_platform_only(scans_for_other_hosts);
        policy.transitive_deps = self.transitive_deps(scans_for_other_hosts);
        policy
    }

    /// Manual probability cutoffs, when the operator named either.
    ///
    /// `None` is the ordinary path: the verdict comes from the model's level
    /// grid — the per-file level sweep plus the active level's cutoffs — so no
    /// explicit thresholds are loaded and `Model::load` keeps its
    /// level-independent defaults.
    ///
    /// Given only `--threshold-hostile`, the suspicious cutoff collapses onto
    /// it rather than being derived. The level-space lookup a suspicious band
    /// needs wants a level table and a known level, and neither applies once an
    /// operator picks a probability directly — so manual mode answers
    /// hostile-versus-benign, which is what it is for.
    #[must_use]
    pub fn thresholds(&self) -> Option<crate::model::Thresholds> {
        match (self.threshold_suspicious, self.threshold_hostile) {
            (None, None) => None,
            (suspicious, hostile) => {
                let hostile = hostile.unwrap_or(crate::model::Thresholds::FALLBACK_HOSTILE);
                Some(crate::model::Thresholds {
                    suspicious: suspicious.unwrap_or(hostile),
                    hostile,
                })
            }
        }
    }

    /// The display filter selected by `--show`.
    #[must_use]
    pub fn display_filter(&self) -> crate::DisplayFilter {
        let all = self.show.iter().any(|s| matches!(s, Show::All));
        crate::DisplayFilter::new(
            all || self.show.iter().any(|s| matches!(s, Show::Hostile)),
            all || self.show.iter().any(|s| matches!(s, Show::Sus)),
            all || self.show.iter().any(|s| matches!(s, Show::Benign)),
        )
    }
}
