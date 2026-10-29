//! The command-line flags every Atomdrift Scan front-end shares.
//!
//! [`GlobalArgs`] holds the `global = true` options of the `atomscan` binary —
//! the LLM, follow/fetch, threshold, display and model-refresh knobs that apply
//! to every subcommand. It lives in the library, not in `src/main.rs`, so a
//! second binary that re-exposes `serve` can `#[command(flatten)]` the
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
    rustdoc::invalid_html_tags,
    reason = "these doc comments are `--help` text, not rustdoc"
)]

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};

use crate::OutputFormat;

/// Warn threshold for a single slow cleave rule (ms); was the `--slow-rule-ms`
/// flag, now a fixed advisory default. Shared, so every binary running this
/// analysis stack warns at the same point rather than each picking a number.
pub const DEFAULT_SLOW_RULE_MS: u64 = 4000;

/// Default hard wall-clock limit for each Rizin subprocess, in seconds.
pub const DEFAULT_RIZIN_TIMEOUT_SECS: u64 = 10 * 60;

/// What a run does about its rules and models before it starts: the
/// `-u`/`--update` and `--no-update` flags, settled into one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refresh {
    /// `--no-update`: run on whatever is on disk. Wins over `--update`.
    Skip,
    /// The default: refresh when the local copy is out of date.
    IfStale,
    /// `-u`/`--update`: re-fetch even when the local copy looks current.
    Force,
}

/// What `--llm` resolution reads from outside the command line: the
/// `SCAN_LLM*` variables and the `~/.tok` token files.
///
/// Gathered in one place so resolution itself never touches the process
/// environment — tests hand it their own.
#[derive(Debug, Default, Clone)]
pub struct LlmEnv {
    /// `SCAN_LLM`: the endpoint list, when `--llm` is absent.
    pub target: Option<String>,
    /// `SCAN_LLM_MODEL`.
    pub model: Option<String>,
    /// `SCAN_LLM_KEY`.
    pub key: Option<String>,
    /// `SCAN_LLM_CONCURRENCY`: in-flight cap; `None` scales with the host.
    pub concurrency: Option<NonZeroUsize>,
    /// `SCAN_LLM_BACKGROUND_CONCURRENCY`: in-flight calls background work may
    /// hold; `None` takes a quarter of the cap.
    pub background_concurrency: Option<NonZeroUsize>,
    /// `SCAN_LLM_SYSTEM_PROMPT_FILE`: a grading prompt to use in place of the
    /// built-in one, for a prompt-tuning A/B.
    pub system_prompt_file: Option<PathBuf>,
    /// `~/.tok/llm`: our own endpoint's bearer token.
    pub llm_token_file: Option<PathBuf>,
    /// `~/.tok/openrouter`.
    pub openrouter_token_file: Option<PathBuf>,
}

impl LlmEnv {
    /// Read the process environment.
    ///
    /// # Errors
    ///
    /// Returns an error when a concurrency variable is set but not a number.
    pub fn from_process() -> Result<Self> {
        use crate::env;
        // 0 keeps its old meaning: the default.
        let count = |var: env::Var| -> Result<Option<NonZeroUsize>> {
            env::text(var)
                .map(|raw| {
                    raw.trim()
                        .parse()
                        .with_context(|| format!("{}={raw:?}", var.name))
                })
                .transpose()
                .map(|n| n.and_then(NonZeroUsize::new))
        };
        Ok(Self {
            target: env::text(env::SCAN_LLM),
            model: env::text(env::SCAN_LLM_MODEL),
            key: env::text(env::SCAN_LLM_KEY),
            concurrency: count(env::SCAN_LLM_CONCURRENCY)?,
            background_concurrency: count(env::SCAN_LLM_BACKGROUND_CONCURRENCY)?,
            system_prompt_file: env::path(env::SCAN_LLM_SYSTEM_PROMPT_FILE),
            llm_token_file: crate::interpret::llm_token_path(),
            openrouter_token_file: crate::interpret::openrouter_token_path(),
        })
    }
}

/// The model bundle a run analyzes with: `dir` when the operator named one,
/// otherwise the installed bundle (fetched on first use).
///
/// # Errors
///
/// Returns an error when no bundle is named and none can be installed.
pub fn resolve_model_dir(dir: Option<PathBuf>) -> Result<PathBuf> {
    match dir {
        Some(dir) => Ok(dir),
        None => crate::models_repo::ensure_model_dir().context("failed to resolve model directory"),
    }
}

/// The operating point a run classifies at. (The envelope's `ml.lvl` is not
/// this: it is the lowest level at which the sample fires, whatever this is.)
///
/// In priority order: the explicit `-l/--level`, the bundle's own
/// `default_severity_level` (so a bundle calibrated at L50 deploys at L50 with
/// no rebuild), then [`crate::model::DEFAULT_SEVERITY_LEVEL`]. Manual
/// `--threshold-*` cutoffs bypass the level grid entirely, so there is no
/// level to report then and the answer is `None`.
#[must_use]
pub fn operating_level(
    level: Option<u16>,
    manual_thresholds: bool,
    model_dir: &Path,
) -> Option<u16> {
    if manual_thresholds {
        return None;
    }
    Some(
        level
            .or_else(|| crate::model::model_default_level(model_dir))
            .unwrap_or(crate::model::DEFAULT_SEVERITY_LEVEL),
    )
}

/// Classification values accepted by `--show`.
//
// clap's `ValueEnum` derive turns a variant doc comment into that value's help
// text, which would switch `--show`'s `[possible values: ...]` line into a
// multi-line list.
#[expect(
    missing_docs,
    reason = "variant docs would change `--show` help output"
)]
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
    /// that would block the pull.
    #[arg(long, global = true)]
    pub no_update: bool,

    /// Insist on isotope13 Enriched: the latest rules, the moment they clear
    /// QA, and remote verdicts (see `--analyze`). A computer that holds a
    /// token (`ISOTOPE13_TOKEN`, or the one saved when it was connected) uses
    /// them without this flag. With it, a computer that holds none is
    /// connected — a terminal shows a code to approve at dash.isotope13.io —
    /// and a refused token stops the scan instead of falling back to the
    /// open-source rules. Also settable via `ATOMDRIFT_ENRICHED=1`; `=0`
    /// keeps the open-source rules.
    #[arg(long, global = true, env = "ATOMDRIFT_ENRICHED", value_parser = clap::builder::FalseyValueParser::new())]
    pub enriched: bool,

    /// Where artifacts are judged. `auto` (the default) asks isotope13 for a
    /// verdict it already holds — sending only the SHA-256 of local files of
    /// 1 MiB or more, and the PURLs of packages on mainstream public registries — and analyzes
    /// locally whatever it does not know; it needs isotope13 Enriched (a
    /// token, or `--enriched`), and without it is the same as `local`.
    /// `local` never asks, and is the reproducible mode. `remote` sends
    /// everything to isotope13 for analysis, uploading local files' contents,
    /// and needs isotope13 Enriched.
    #[arg(long, global = true, value_enum, default_value = "auto")]
    pub analyze: crate::remote::Analyze,

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
    #[arg(short, long, global = true, default_value = "terminal")]
    pub format: OutputFormat,

    /// Scan mode: `fast` (bloom matching only), `balanced` (bloom short-circuits,
    /// then full scan), or `slow` (no bloom; always full scan). Workers are
    /// always slow.
    #[arg(long, global = true, default_value = "balanced")]
    pub mode: crate::Mode,

    /// Hard wall-clock limit for each Rizin subprocess, in seconds. On expiry
    /// Atomscan kills and reaps Rizin before releasing the analysis worker; on
    /// Unix it also kills the complete process group.
    #[arg(
        long,
        global = true,
        value_name = "SECS",
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

    /// Verdicts or severe static findings to display in the terminal view:
    /// hostile, suspicious, sus, benign, all (comma-separated). Suspicious and
    /// hostile findings remain visible even with a benign model verdict.
    /// The machine formats (json, tiny, interpret) emit every scanned file
    /// regardless.
    #[arg(long, global = true, value_delimiter = ',', default_values = ["hostile", "sus"])]
    pub show: Vec<Show>,

    /// Show raw probability and SHAP feature values in terminal output
    #[arg(long, global = true, hide = true)]
    pub extra: bool,

    /// [deprecated] Legacy on-switch for LLM interpretation; superseded by
    /// `--llm`.
    #[arg(long, global = true, hide = true)]
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
    /// block. The default sits on an elbow: malicious admission is flat
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
    /// admission fires, and the reader then moves nothing; on the benchmark
    /// corpus a veto at 300 halves benign calls and keeps every correct
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
    /// and so does an absent one for an interactive scan. `serve`
    /// defaults to `all` instead: it answers for other hosts, where a category
    /// nobody followed is one nobody ever learns. Also settable via
    /// `SCAN_FOLLOW`.
    #[arg(
        long = "follow",
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
    /// dropper), and so on.
    #[arg(
        long = "follow-depth",
        global = true,
        value_name = "N",
        default_value_t = crate::fetch::DEFAULT_FETCH_DEPTH
    )]
    pub follow_depth: u8,

    /// [EXPERIMENTAL] Skip fetching a declared dependency whose registry publish
    /// date is older than this many days — the cheap provenance lookup runs
    /// first, and only recent (freshest-risk) releases are pulled and scanned.
    /// Applies to declared dependencies only; URLs are never age-gated. `0`
    /// disables the gate (fetch every resolvable dependency). A dependency whose
    /// age can't be determined is always fetched.
    ///
    /// Unset, the ceiling is a fresh-risk window
    /// ([`scan::fetch::DEFAULT_MAX_DEP_AGE_DAYS`]). Optional rather than
    /// defaulted so each front-end supplies its own fallback.
    #[arg(long, global = true, value_name = "DAYS")]
    pub fetch_max_age: Option<u32>,

    /// [EXPERIMENTAL] Fetch and scan native-binary dependencies for every
    /// platform, not just the host's. This is automatic in `serve`, which
    /// scans on behalf of other machines; interactive scans
    /// stay host-only for latency unless this flag is passed.
    #[arg(long, global = true, conflicts_with = "fetch_host_platform_only")]
    pub fetch_all_platforms: bool,

    /// [EXPERIMENTAL] Fetch only native-binary dependencies matching this
    /// host's OS and architecture. This is the interactive default and an
    /// explicit completeness opt-out for `serve`.
    #[arg(long, global = true, conflicts_with = "fetch_all_platforms")]
    pub fetch_host_platform_only: bool,

    /// [EXPERIMENTAL] Follow declared dependencies past the first hop — the
    /// dependencies of a fetched dependency, out to `--follow-depth`. This is
    /// automatic in `serve`, which answers for other hosts;
    /// an interactive scan stops declared dependencies at the first hop, since
    /// the transitive tail costs a registry lookup each and is almost entirely
    /// old releases the age gate then discards. URLs and command-mentioned
    /// packages — the dropper chain — are followed at every hop either way.
    ///
    #[arg(long, global = true, conflicts_with = "fetch_direct_deps_only")]
    pub fetch_transitive_deps: bool,

    /// [EXPERIMENTAL] Follow declared dependencies for one hop only. This is the
    /// interactive default and an explicit completeness opt-out for `serve`.
    #[arg(long, global = true, conflicts_with = "fetch_transitive_deps")]
    pub fetch_direct_deps_only: bool,

    /// [EXPERIMENTAL] How long to trust cached *mutable* registry metadata before
    /// revalidating. Accepts a unit suffix (`90s`, `30m`, `4h`, `2d`) — a bare
    /// number is seconds; `never` caches indefinitely (offline/air-gapped). This
    /// bounds the two mutable tiers: a pinned version's packument (whose yank
    /// status can change after publish) and a `latest`/versionless lookup.
    /// Unset keeps the defaults — 4h pinned, 1h unpinned. A released version's
    /// immutable file list is never re-checked regardless.
    #[arg(
        long,
        global = true,
        value_name = "DUR",
        value_parser = crate::fetch::parse_duration
    )]
    pub registry_ttl: Option<std::time::Duration>,

    /// [EXPERIMENTAL] Size cap for a single downloaded artifact. Accepts a unit
    /// suffix (`256M`, `2G`, `512K`); a bare number is bytes. A response larger
    /// than this is abandoned, so one artifact can't dominate a run.
    #[arg(
        long,
        global = true,
        value_name = "SIZE",
        default_value = "256M",
        value_parser = crate::fetch::parse_bytes
    )]
    pub fetch_max_size: u64,

    /// Persist unfinished fetches and resume them when scanning the same content.
    /// Increasing --follow-depth resumes saved deeper references.
    #[arg(long, global = true, value_name = "PATH")]
    pub fetch_pending: Option<std::path::PathBuf>,

    /// Include lockfile dependencies used exclusively for development or tests.
    #[arg(long, global = true, action = clap::ArgAction::Set, default_value_t = true)]
    pub fetch_dev_deps: bool,

    /// Analyze all pinned dependency versions, including older releases.
    #[arg(long, global = true)]
    pub fetch_all_versions: bool,

    /// Retry only native binaries whose disassembler timed out, once, with this
    /// larger per-file deadline. Unset disables the extra attempt.
    #[arg(long, global = true, value_name = "SECS", value_parser = clap::value_parser!(u64).range(1..))]
    pub rizin_retry_timeout_secs: Option<u64>,

    /// [EXPERIMENTAL] Maximum number of *live* dependency/package fetches
    /// triggered by a single scanned file. This is 100 by default. Cache hits
    /// are always served and never counted, so a warm re-run is never throttled.
    /// References past the cap are recorded as budget-exceeded, never silently
    /// dropped.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = crate::fetch::DEFAULT_MAX_FILE_FETCHES
    )]
    pub fetch_max_file_fetches: usize,

    /// [EXPERIMENTAL] Maximum number of *live* opportunistic raw-URL fetches
    /// triggered by a single scanned file. This is 4 by default. URL references
    /// declared as dependencies or command-mentioned packages use the larger
    /// `--fetch-max-file-fetches` cap instead.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = crate::fetch::DEFAULT_MAX_URL_FETCHES
    )]
    pub fetch_max_urls: usize,

    /// [EXPERIMENTAL] Maximum total bytes fetched on behalf of a single scanned
    /// file. Accepts a unit suffix (`2G`); a bare number is bytes.
    #[arg(
        long,
        global = true,
        value_name = "SIZE",
        default_value = "2G",
        value_parser = crate::fetch::parse_bytes
    )]
    pub fetch_max_file_size: u64,

    /// [EXPERIMENTAL] Wall-clock ceiling on the fetch phase for a single
    /// scanned artifact. Accepts a unit suffix (`90s`, `5m`, `1h`); a bare
    /// number is seconds; `0` or `never` disables the cap. This is 5 minutes by
    /// default. The count and size budgets bound how much a scan fetches, not
    /// how long fetching takes — a wide tree of slow registries can hold a scan
    /// open with every count budget still unspent. References not reached
    /// before the cap are left unfollowed; whatever was already fetched is
    /// analyzed and graded as usual.
    #[arg(
        long,
        global = true,
        value_name = "DUR",
        default_value = "5m",
        value_parser = crate::fetch::parse_duration
    )]
    pub fetch_timeout: std::time::Duration,

    /// [EXPERIMENTAL] Maximum number of *live* fetches across the whole
    /// execution — a hard ceiling over every scanned file combined. Lifted in
    /// the long-lived `serve` mode, where the per-file caps bound
    /// each job instead.
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = crate::fetch::DEFAULT_MAX_TOTAL_FETCHES
    )]
    pub fetch_max_total_fetches: usize,

    /// [EXPERIMENTAL] Maximum total bytes fetched across the whole execution.
    /// Accepts a unit suffix (`10G`); a bare number is bytes. Lifted in
    /// the long-lived `serve` mode.
    #[arg(
        long,
        global = true,
        value_name = "SIZE",
        default_value = "10G",
        value_parser = crate::fetch::parse_bytes
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
    /// Returns an error when `SCAN_LLM_CONCURRENCY` does not parse, or for any
    /// reason [`GlobalArgs::interpret_config_with`] gives.
    pub fn interpret_config(&self) -> Result<Option<crate::interpret::InterpretConfig>> {
        self.interpret_config_with(&LlmEnv::from_process()?)
    }

    /// [`GlobalArgs::interpret_config`] against an explicit environment.
    ///
    /// # Errors
    ///
    /// Returns an error when `--llm` names no endpoint, when the only
    /// endpoint requested is unusable (an OpenRouter target with no key, or
    /// one whose model can be neither pinned nor discovered), or when every
    /// endpoint in a failover chain was dropped for those reasons.
    pub fn interpret_config_with(
        &self,
        env: &LlmEnv,
    ) -> Result<Option<crate::interpret::InterpretConfig>> {
        use crate::interpret::{
            DEFAULT_BASE_URL, LlmEndpoint, is_openrouter_endpoint, llm_models, llm_targets,
        };
        let flag_or = |flag: &Option<String>, env: &Option<String>| -> Option<String> {
            flag.clone()
                .or_else(|| env.clone())
                .filter(|s| !s.is_empty())
        };
        // `--llm [TARGET]` / SCAN_LLM (the bare flag defaults TARGET to `local`)
        // or the legacy `--interpret` flag turns the pass on.
        let target = flag_or(&self.llm, &env.target);
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
            flag_or(&self.llm_model, &env.model).as_deref(),
            targets.len(),
        );
        // An explicit key wins, and applies to every endpoint in the chain —
        // it is the operator naming one credential. Otherwise each endpoint
        // resolves its own, and only its own: `~/.tok/openrouter` for
        // OpenRouter, `~/.tok/llm` for everything else — our own vLLM requires
        // one, and a host that has the file authenticates without any flag.
        // Absent a file, the request goes out unauthenticated, which is still
        // right for an endpoint that wants no key.
        let explicit_key = flag_or(&self.llm_key, &env.key);

        // One endpoint must work; the rest are a cushion. So a config problem
        // is fatal when it is the only endpoint (a misconfigured `--llm` must
        // not be silent), and a warning when others remain — an OpenRouter
        // fallback with no model pinned, or a primary that is down at startup,
        // should cost that entry, not the scan.
        let single = targets.len() == 1;
        // A key given outright is for the kind of endpoint named first: an
        // OpenRouter key never goes to our own vLLM, and our vLLM key never
        // to OpenRouter, whichever the chain falls back to.
        let explicit_for_openrouter = targets
            .first()
            .is_some_and(|base_url| is_openrouter_endpoint(base_url));
        let mut resolved: Vec<LlmEndpoint> = Vec::with_capacity(targets.len());
        let mut skipped: Vec<String> = Vec::new();
        for (base_url, pinned) in targets.into_iter().zip(pinned) {
            let openrouter = is_openrouter_endpoint(&base_url);
            // `~/.tok/llm` is *our* endpoint's token and must never travel to
            // a third party, so OpenRouter takes its own file or nothing —
            // sending the vLLM key there would hand a working credential to an
            // unrelated host and read as a plain 401 when it did.
            let api_key = explicit_key
                .clone()
                .filter(|_| openrouter == explicit_for_openrouter)
                .or_else(|| {
                    let file = if openrouter {
                        &env.openrouter_token_file
                    } else {
                        &env.llm_token_file
                    };
                    crate::interpret::read_token_file(file.as_deref()?)
                });
            if api_key.is_some()
                && let Ok(url) = reqwest::Url::parse(&base_url)
                && url.scheme() == "http"
                && !url.host_str().is_some_and(|host| {
                    let host = host.trim_start_matches('[').trim_end_matches(']');
                    host.eq_ignore_ascii_case("localhost")
                        || host
                            .parse::<std::net::IpAddr>()
                            .is_ok_and(|ip| ip.is_loopback())
                })
            {
                tracing::warn!(
                    llm = %base_url,
                    "the LLM key is sent to this address in cleartext; use https unless the network itself is encrypted"
                );
            }
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
            let model = match pinned {
                Some(m) => m,
                None if openrouter => crate::interpret::OPENROUTER_DEFAULT_MODEL.to_string(),
                None => match crate::interpret::discover_model(&base_url, api_key.as_deref()) {
                    Ok(m) => m,
                    // Say which of the several ways discovery can fail this
                    // was — an unreachable host, a 404 from a base URL missing
                    // its /v1, a rejected key and an endpoint serving nothing
                    // all need different fixes, and only pinning a model is
                    // common to all of them.
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
            max_concurrency: env
                .concurrency
                .unwrap_or_else(crate::interpret::default_max_concurrency),
            background_concurrency: env.background_concurrency,
            system_prompt: match &env.system_prompt_file {
                Some(path) => crate::interpret::read_system_prompt(path)?,
                None => crate::interpret::SYSTEM_PROMPT.into(),
            },
            fallbacks: resolved.collect(),
        }))
    }

    /// The `-u`/`--update` and `--no-update` flags as one value.
    #[must_use]
    pub fn refresh(&self) -> Refresh {
        if self.no_update {
            Refresh::Skip
        } else if self.update {
            Refresh::Force
        } else {
            Refresh::IfStale
        }
    }

    /// Whether native-binary dependencies are fetched for this host only.
    ///
    /// `scans_for_other_hosts` is true for the corpus-facing role (`serve`),
    /// which scans on behalf of other machines and so takes every
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
    /// ceiling used when `--fetch-max-age` is unset. An explicit
    /// selection is honored verbatim; the knobs always apply.
    #[must_use]
    pub fn fetch_policy(
        &self,
        mut policy: crate::fetch::FetchPolicy,
        default_max_age: u32,
        scans_for_other_hosts: bool,
    ) -> crate::fetch::FetchPolicy {
        policy.depth = self.follow_depth;
        policy.include_dev_dependencies = self.fetch_dev_deps;
        policy.all_versions = self.fetch_all_versions;
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
        crate::DisplayFilter {
            hostile: all || self.show.iter().any(|s| matches!(s, Show::Hostile)),
            suspicious: all || self.show.iter().any(|s| matches!(s, Show::Sus)),
            benign: all || self.show.iter().any(|s| matches!(s, Show::Benign)),
        }
    }
}
