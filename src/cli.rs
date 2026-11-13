//! The command-line flags every Atomdrift Scan front-end shares.
//!
//! [`GlobalArgs`] holds the `global = true` options of the `atomscan` binary —
//! the LLM, follow, display and model-refresh knobs that apply
//! to every subcommand. It lives in the library, not in `src/main.rs`, so a
//! second binary that re-exposes `serve` can `#[command(flatten)]` the
//! same declarations instead of restating them. Restating them is how two
//! front-ends silently acquire different defaults; there is exactly one copy of
//! each flag, here.
//!
//! Adding a flag here adds it to every binary that flattens this struct, and
//! renaming one is a user-visible change to all of them.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};

use crate::OutputFormat;

/// Warn threshold for a single slow cleave rule (ms); was the `--slow-rule-ms`
/// flag, now a fixed advisory default. Shared, so every binary running this
/// analysis stack warns at the same point rather than each picking a number.
pub const DEFAULT_SLOW_RULE_MS: u64 = 4000;

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
    /// `~/.tok/llm`: the bearer token for any endpoint but OpenRouter.
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
/// probability cutoffs (a library caller's [`crate::model::Thresholds`])
/// bypass the level grid entirely, so there is no level to report then and
/// the answer is `None`.
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

/// One link of an `--llm` chain, with its key and model, or why it cannot be
/// used.
///
/// `explicit_key` is `SCAN_LLM_KEY`, with whether it belongs to OpenRouter;
/// it applies only to endpoints of that kind. Otherwise each endpoint takes
/// its own token file, and only its own: `~/.tok/openrouter` for OpenRouter,
/// `~/.tok/llm` for everything else, so a host's credential never travels to
/// a third party. Absent a file the request goes out unauthenticated, which is
/// right for an endpoint that wants no key.
///
/// A pinned model wins. OpenRouter's catalog is large and billed, so nothing
/// is guessed from it: it gets its own `openrouter/auto` router. Any other
/// endpoint is asked what it serves — the one network call in resolving the
/// flags — since a guessed name would surface as an opaque server error
/// mid-scan instead of here.
fn resolve_endpoint(
    base_url: String,
    pinned: Option<String>,
    explicit_key: Option<(&str, bool)>,
    env: &LlmEnv,
) -> std::result::Result<crate::interpret::LlmEndpoint, String> {
    let openrouter = crate::interpret::is_openrouter_endpoint(&base_url);
    let api_key = explicit_key
        .filter(|&(_, for_openrouter)| for_openrouter == openrouter)
        .map(|(key, _)| key.to_owned())
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
        return Err(format!(
            "{base_url}: OpenRouter requires a key: SCAN_LLM_KEY or ~/.tok/openrouter"
        ));
    }
    let model = match pinned {
        Some(model) => model,
        None if openrouter => crate::interpret::OPENROUTER_DEFAULT_MODEL.to_string(),
        // Say which of the several ways discovery can fail this was — an
        // unreachable host, a 404 from a base URL missing its /v1, a rejected
        // key and an endpoint serving nothing all need different fixes, and
        // only pinning a model is common to all of them.
        None => crate::interpret::discover_model(&base_url, api_key.as_deref()).map_err(|e| {
            format!(
                "no LLM model available from {base_url}: {e}. Fix the endpoint, \
                 or name a model with --llm-model (env: SCAN_LLM_MODEL)"
            )
        })?,
    };
    Ok(crate::interpret::LlmEndpoint {
        base_url,
        model,
        api_key,
    })
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

    /// Insist on Atomdrift Uplink: zero-hour rules the moment they clear QA,
    /// remote dependency and hash verdicts (see `--analyze`), and a default
    /// hosted LLM endpoint (raw files stay local unless `--analyze=remote`).
    /// A computer that holds a token (`ISOTOPE13_TOKEN`, or the one saved when
    /// it was connected) uses Uplink without this flag. With it, a computer
    /// that holds none is connected — a terminal shows a code to approve at
    /// dash.isotope13.io — and a refused token stops the scan instead of
    /// falling back to the open-source rules. Also settable via
    /// `ATOMDRIFT_UPLINK=1`; `=0` keeps the open-source rules.
    #[arg(long, global = true, env = "ATOMDRIFT_UPLINK", value_parser = clap::builder::FalseyValueParser::new())]
    pub uplink: bool,

    /// Where artifacts are judged. `auto` (the default) asks Atomdrift Uplink
    /// for a verdict it already holds — sending only the SHA-256 of local
    /// files of 1 MiB or more, and the PURLs of packages on mainstream public
    /// registries — and analyzes locally whatever it does not know; it needs
    /// Atomdrift Uplink (a token, or `--uplink`), and without it is the same
    /// as `local`. `local` never asks, and is the reproducible mode. `remote`
    /// sends everything to Uplink for analysis, uploading local files'
    /// contents, and needs Atomdrift Uplink.
    #[arg(long, global = true, value_enum, default_value = "auto")]
    pub analyze: crate::remote::Analyze,

    /// Force light-background color theme
    #[arg(long, global = true, conflicts_with = "dark")]
    pub light: bool,

    /// Force dark-background color theme
    #[arg(long, global = true, conflicts_with = "light")]
    pub dark: bool,

    /// Output format
    #[arg(short, long, global = true, default_value = "terminal")]
    pub format: OutputFormat,

    /// Scan mode: `fast` (bloom matching only), `balanced` (bloom short-circuits,
    /// then full scan), or `slow` (no bloom; always full scan). `serve` always
    /// analyzes a submitted file in full.
    #[arg(long, global = true, default_value = "balanced")]
    pub mode: crate::Mode,

    /// Choose FP level N (0-25000): L0 allows zero observed calibration FPs;
    /// positive L allows one FP plus an estimated (L-1) per 100M benigns.
    /// Higher = more sensitive. The bundle supplies calibrated thresholds.
    #[arg(
        short = 'l',
        long,
        value_name = "N",
        value_parser = clap::value_parser!(u16).range(0..=25000),
        global = true,
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

    /// With `--format json-full`: give every archive member a line, analyzed
    /// or not, rather than only the members an upload keeps. For tools that
    /// measure coverage; what is sent to hopper does not change.
    #[arg(long, global = true, hide = true)]
    pub all_members: bool,

    /// Enable additional LLM interpretation of analyzed samples: a second
    /// opinion blended with the ML verdict (stored in the `llm` JSON section
    /// and shown inline). Given with no value, uses a local model (an
    /// OpenAI-compatible endpoint at `http://localhost:8000/v1`). TARGET may be
    /// `local`, `openrouter` (`https://openrouter.ai/api/v1`; key from
    /// `SCAN_LLM_KEY` or `~/.tok/openrouter`; defaults to `openrouter/auto`
    /// unless `--llm-model` names one), or an explicit OpenAI-compatible base
    /// URL. Endpoints that require a bearer token take it from `~/.tok/llm`
    /// when that file exists. Comma-separate several to fail over in order,
    /// e.g. `https://llm.example.com/v1,openrouter`. (env: SCAN_LLM)
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

    /// Additional passwords to try for encrypted ZIP/7z archives. Repeat the
    /// option to provide more than one; cleave's common defaults remain active.
    #[arg(long = "zip-password", value_name = "PASSWORD", global = true)]
    pub zip_passwords: Vec<String>,

    /// Experimental: follow references discovered inside the requested
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
        env = crate::env::SCAN_FOLLOW.name
    )]
    pub follow: Option<crate::fetch::FetchPolicy>,

    /// Experimental: how many hops of references to follow when `--follow` is on:
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
}

impl GlobalArgs {
    /// Build the LLM interpretation config from `--llm` and `--llm-model`,
    /// falling back to env vars. `None` when interpretation is not requested.
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
    /// Each endpoint in a failover chain resolves on its own, which may mean
    /// asking it what model it serves. One
    /// endpoint must work; the rest are a cushion. So a problem is fatal when
    /// it is the only endpoint — a misconfigured `--llm` must not be silent —
    /// and a warning when others remain.
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
        use crate::interpret::{is_openrouter_endpoint, llm_models, llm_targets};
        // The flag wins over the environment, and an empty value means unset.
        fn flag_or<'a>(flag: Option<&'a str>, env: Option<&'a str>) -> Option<&'a str> {
            flag.or(env).filter(|s| !s.is_empty())
        }
        // `--llm [TARGET]` / SCAN_LLM (the bare flag defaults TARGET to `local`)
        // turns the pass on. `local` maps to the local endpoint, `openrouter`
        // to the public API, anything else is an OpenAI-compatible base URL;
        // a comma-separated target is a failover chain, tried in order.
        let Some(target) = flag_or(self.llm.as_deref(), env.target.as_deref()) else {
            return Ok(None);
        };
        let targets = llm_targets(target);
        if targets.is_empty() {
            anyhow::bail!("--llm (env: SCAN_LLM) names no endpoint");
        }
        let pinned = llm_models(
            flag_or(self.llm_model.as_deref(), env.model.as_deref()),
            targets.len(),
        );
        // `SCAN_LLM_KEY` is the operator naming one credential, for the kind
        // of endpoint named first: an OpenRouter key never goes to another
        // host, and another host's key never to OpenRouter, whichever the
        // chain falls back to.
        let explicit_key = env.key.as_deref().filter(|s| !s.is_empty()).map(|key| {
            let for_openrouter = targets
                .first()
                .is_some_and(|base_url| is_openrouter_endpoint(base_url));
            (key, for_openrouter)
        });
        let single = targets.len() == 1;
        let mut resolved = Vec::with_capacity(targets.len());
        let mut skipped: Vec<String> = Vec::new();
        for (base_url, pinned) in targets.into_iter().zip(pinned) {
            match resolve_endpoint(base_url, pinned, explicit_key, env) {
                Ok(endpoint) => resolved.push(endpoint),
                Err(why) if single => anyhow::bail!("{why}"),
                Err(why) => {
                    tracing::warn!(reason = %why, "LLM endpoint unusable, dropped from the failover chain");
                    skipped.push(why);
                }
            }
        }
        let mut resolved = resolved.into_iter();
        let primary = resolved.next().ok_or_else(|| {
            anyhow::anyhow!("no usable LLM endpoint:\n  {}", skipped.join("\n  "))
        })?;
        Ok(Some(crate::interpret::InterpretConfig {
            base_url: primary.base_url,
            model: primary.model,
            api_key: primary.api_key,
            // The admission gate is not a flag: ML admits anything it placed
            // on the level grid, and the probability floor and size veto sit
            // at their measured defaults.
            min_level: None,
            min_prob: crate::interpret::DEFAULT_LLM_MIN_PROB,
            benign_notable_cap: crate::interpret::DEFAULT_LLM_BENIGN_NOTABLE_CAP,
            // One budget per hop for every mode: a wedged endpoint is caught
            // by the connect timeout and the breaker, not by this.
            timeout: std::time::Duration::from_secs(crate::interpret::DEFAULT_TIMEOUT_SECS),
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

    /// Apply the follow knobs to `policy`: the hop count from `--follow-depth`,
    /// the dependency age ceiling, and the scanner's role. `serve` scans on
    /// behalf of other hosts, so it takes every platform's native-binary
    /// dependencies and the transitive tail; an interactive scan stays
    /// host-only and stops declared dependencies at the first hop, since the
    /// transitive tail costs a registry lookup each and is almost entirely old
    /// releases the age gate then discards. An explicit selection is honored
    /// verbatim, and the per-file ceilings are `policy`'s own.
    #[must_use]
    pub fn fetch_policy(
        &self,
        mut policy: crate::fetch::FetchPolicy,
        max_dep_age_days: u32,
        scans_for_other_hosts: bool,
    ) -> crate::fetch::FetchPolicy {
        policy.depth = self.follow_depth;
        policy.max_dep_age_days = max_dep_age_days;
        policy.host_platform_only = !scans_for_other_hosts;
        policy.transitive_deps = scans_for_other_hosts;
        policy
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
