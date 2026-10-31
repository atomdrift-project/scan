use super::{
    Cli, Commands, MaxRssPolicy, default_cli_follow_policy, fetch_policy,
    is_expected_yara_cache_mismatch, resolve_hopper_value, with_default_subcommand,
};
use anyhow::{Context, Result};
use clap::Parser;
use scan::OutputFormat;
use scan::cli::LlmEnv;
use std::ffi::OsString;
use std::net::SocketAddr;
use std::num::NonZeroU64;
use std::path::PathBuf;

#[test]
fn update_rules_takes_no_flags_of_its_own() {
    let cli = Cli::try_parse_from(["atomscan", "update-rules"]).unwrap();
    assert!(matches!(cli.command, Some(Commands::UpdateRules)));
    for flag in ["--nsrl", "--check", "--models-only"] {
        assert!(Cli::try_parse_from(["atomscan", "update-rules", flag]).is_err());
    }
}

#[test]
fn a_closed_pipe_anywhere_in_the_chain_exits_quietly() {
    let pipe = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::BrokenPipe))
        .context("writing a result");
    assert!(super::is_broken_pipe(&pipe));
    let other = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::NotFound));
    assert!(!super::is_broken_pipe(&other));
}

#[test]
fn only_expected_yara_cache_mismatch_is_quietable() {
    let expected = "compiled YARA rules do not match the rule sources on disk; \
                        ignoring them and recompiling";
    assert!(is_expected_yara_cache_mismatch(
        "cleave::yara_engine",
        expected
    ));
    assert!(!is_expected_yara_cache_mismatch(
        "cleave::yara_engine",
        "YARA bucket scan failed, skipping bucket"
    ));
    assert!(!is_expected_yara_cache_mismatch("scan::engine", expected));
}

#[test]
fn follow_defaults_to_references_and_dependencies() -> Result<()> {
    let cli = Cli::try_parse_from(with_default_subcommand(["scan", "/tmp/a"]))
        .context("parse should work")?;
    assert!(
        cli.global.follow.is_none(),
        "absence of --follow is resolved at startup"
    );

    let policy = default_cli_follow_policy();
    assert!(policy.urls && policy.packages && policy.deps);
    assert!(
        !policy.ci,
        "CI actions never reach an installed artifact; auditing them is opt-in"
    );
    // A bare `--follow` must select exactly what an absent one resolves to.
    let bare = Cli::try_parse_from(with_default_subcommand(["scan", "--follow", "/tmp/a"]))
        .context("bare --follow should parse")?
        .global
        .follow
        .context("bare --follow has a default_missing_value")?;
    assert_eq!(
        (bare.urls, bare.packages, bare.deps, bare.ci),
        (policy.urls, policy.packages, policy.deps, policy.ci),
    );
    assert_eq!(
        policy.max_file_fetches,
        scan::fetch::DEFAULT_MAX_FILE_FETCHES
    );
    assert_eq!(policy.max_file_fetches, 100);
    assert_eq!(policy.max_url_fetches, scan::fetch::DEFAULT_MAX_URL_FETCHES);
    assert_eq!(policy.max_url_fetches, 4);
    Ok(())
}

/// A service follows everything it can reach, because the corpus it fills
/// is asked about artifacts nobody has looked at yet: a category left
/// unfollowed is one nobody ever learns about.
#[test]
fn serve_follows_everything_by_default() {
    let service = scan::fetch::default_service_follow_policy();
    assert!(
        service.urls && service.packages && service.deps && service.ci,
        "a cache-population role must follow every category"
    );

    // The interactive default stays narrow: CI actions never reach an
    // installed artifact, so one person's scan should not pay for them.
    let interactive = default_cli_follow_policy();
    assert!(
        !interactive.ci,
        "the interactive default must not widen with the service one"
    );
    assert_ne!(
        (
            interactive.urls,
            interactive.packages,
            interactive.deps,
            interactive.ci
        ),
        (service.urls, service.packages, service.deps, service.ci),
        "the two defaults are deliberately different; if they converge, say so on purpose"
    );
}

/// The roles that get the wide default are exactly the ones that scan on
/// behalf of other hosts. Keeping the two in step is what stops a new
/// service subcommand from quietly populating the corpus with a narrow
/// verdict.
#[test]
fn the_wide_default_covers_every_service_role() {
    let cli = Cli::try_parse_from(["scan", "serve"]).expect("serve should parse");
    assert!(
        cli.command.as_ref().is_some_and(Commands::is_daemon),
        "serve scans for other hosts and must take the wide follow default"
    );
    let interactive = Cli::try_parse_from(with_default_subcommand(["scan", "/tmp/a"]))
        .expect("path scan should parse");
    assert!(
        !interactive
            .command
            .as_ref()
            .is_some_and(Commands::is_daemon),
        "an interactive path scan keeps the narrow default"
    );
}

/// `--llm` takes an optional value, so clap leaves a following flag alone;
/// the shorthand scan must too, or it misreads that flag's value as a path and
/// rewrites a line that names a subcommand.
#[test]
fn an_optional_value_flag_does_not_swallow_the_next_flag() {
    let args = ["scan", "--llm", "-f", "json", "purl", "pkg:npm/x"];
    let rewritten = with_default_subcommand(args);
    assert_eq!(rewritten, args.map(std::ffi::OsString::from));
    let with_value = with_default_subcommand(["scan", "--llm", "local", "/tmp/a"]);
    assert_eq!(with_value[1], "fs", "{with_value:?}");
}

#[test]
fn retired_follow_spellings_are_refused() {
    let parse = |args: &[&str]| Cli::try_parse_from(with_default_subcommand(args.iter().copied()));
    for args in [
        ["scan", "--fetch=all", "/tmp/a"],
        ["scan", "--fetch-depth=2", "/tmp/a"],
        ["scan", "--follow=deps,packages,urls,ci", "/tmp/a"],
    ] {
        assert!(parse(&args).is_err(), "{args:?} should be refused");
    }
    assert!(
        parse(&[
            "scan",
            "--follow=dependencies,references,ci-actions",
            "/tmp/a"
        ])
        .is_ok()
    );
    assert!(parse(&["scan", "--follow-depth=2", "/tmp/a"]).is_ok());
}

#[test]
fn dependency_scope_follows_scanner_role() -> Result<()> {
    let cli = Cli::try_parse_from(with_default_subcommand(["scan", "/tmp/a"]))
        .context("default should parse")?;
    let interactive = fetch_policy(&cli.global, cli.command.as_ref().context("a command")?);
    assert!(
        interactive.host_platform_only,
        "interactive scans optimize for the current host"
    );
    assert!(
        !interactive.transitive_deps,
        "interactive scans stop declared dependencies at the first hop"
    );
    assert!(
        scan::fetch::FetchPolicy::default().host_platform_only,
        "library and CLI defaults must agree"
    );

    let daemon = Cli::try_parse_from(["scan", "serve"]).context("serve should parse")?;
    let served = fetch_policy(
        &daemon.global,
        daemon.command.as_ref().context("a command")?,
    );
    assert!(
        !served.host_platform_only,
        "serve scans on behalf of other hosts"
    );
    assert!(served.transitive_deps, "serve populates the corpus");
    Ok(())
}

#[test]
fn bare_paths_default_to_scan_shorthand() -> Result<()> {
    let cli = Cli::try_parse_from(with_default_subcommand(["scan", "/tmp/a", "/tmp/b"]))
        .context("parse should work")?;
    match cli.command.context("fs subcommand expected")? {
        Commands::Path { paths, .. } => assert_eq!(
            paths,
            vec![PathBuf::from("/tmp/a"), PathBuf::from("/tmp/b")]
        ),
        other => anyhow::bail!("unexpected command: {other:?}"),
    }
    Ok(())
}

/// The shorthand is the form operators and agents actually type, and every
/// flag `fs` accepts has to survive it — `--upload` was rejected outright
/// before the line was rewritten into a real subcommand.
#[test]
fn shorthand_accepts_every_fs_flag() -> Result<()> {
    let cli = Cli::try_parse_from(with_default_subcommand([
        "scan",
        "--upload",
        "http://hopper:8081",
        "--registry-map",
        "/tmp/map.json",
        "--format",
        "json",
        "/tmp/a",
    ]))
    .context("parse should work")?;
    assert_eq!(cli.global.format, OutputFormat::Json);
    match cli.command.context("fs subcommand expected")? {
        Commands::Path {
            paths,
            hopper,
            registry_map,
            enable_sidecar,
        } => {
            assert_eq!(paths, vec![PathBuf::from("/tmp/a")]);
            assert_eq!(hopper.as_deref(), Some("http://hopper:8081"));
            assert_eq!(registry_map, Some(PathBuf::from("/tmp/map.json")));
            assert!(!enable_sidecar, "sidecar handling is opt-in");
        }
        other => anyhow::bail!("unexpected command: {other:?}"),
    }
    Ok(())
}

/// `--enable-sidecar` is a flag of the path scan, and the bare-path
/// shorthand carries it like any other.
#[test]
fn enable_sidecar_is_a_path_scan_flag() -> Result<()> {
    let cli = Cli::try_parse_from(with_default_subcommand([
        "scan",
        "--enable-sidecar",
        "/tmp/a",
    ]))
    .context("parse should work")?;
    match cli.command.context("fs subcommand expected")? {
        Commands::Path {
            paths,
            enable_sidecar,
            ..
        } => {
            assert_eq!(paths, vec![PathBuf::from("/tmp/a")]);
            assert!(enable_sidecar);
        }
        other => anyhow::bail!("unexpected command: {other:?}"),
    }
    Ok(())
}

#[test]
fn default_subcommand_declines_lines_that_name_their_own() {
    for line in [
        vec!["scan", "fs", "/tmp/a"],
        vec!["scan", "purl", "pkg:npm/left-pad@1.3.0"],
        vec!["scan", "serve"],
        vec!["scan", "help"],
        vec!["scan", "--help"],
        vec!["scan", "--version"],
        vec!["scan"],
        vec!["scan", "--verbose"],
    ] {
        assert_eq!(
            with_default_subcommand(line.clone()),
            line.iter().map(OsString::from).collect::<Vec<_>>(),
            "{line:?} should reach the top-level parser as typed",
        );
    }
}

#[test]
fn default_subcommand_inserts_before_operands() {
    assert_eq!(
        with_default_subcommand(["scan", "-f", "json", "/tmp/a"]),
        ["scan", "fs", "-f", "json", "/tmp/a"]
            .iter()
            .map(OsString::from)
            .collect::<Vec<_>>(),
    );
    // `--` marks the operand, so a path spelled like a flag still scans.
    assert_eq!(
        with_default_subcommand(["scan", "--", "-weird-name"]),
        ["scan", "fs", "--", "-weird-name"]
            .iter()
            .map(OsString::from)
            .collect::<Vec<_>>(),
    );
}

#[test]
fn hopper_environment_fills_missing_scan_destination() {
    assert_eq!(
        resolve_hopper_value(None, Some("http://hopper:8081/".to_string())).as_deref(),
        Some("http://hopper:8081/")
    );
    assert_eq!(
        resolve_hopper_value(
            Some("http://flag:8081".to_string()),
            Some("http://env".to_string())
        )
        .as_deref(),
        Some("http://flag:8081")
    );
    assert!(resolve_hopper_value(None, Some("  ".to_string())).is_none());
}

#[test]
fn fs_subcommand_accepts_multiple_paths() -> Result<()> {
    let cli = Cli::try_parse_from(with_default_subcommand(["scan", "fs", "/tmp/a", "/tmp/b"]))
        .context("parse should work")?;
    match cli.command.context("fs subcommand expected")? {
        Commands::Path { paths, .. } => {
            assert_eq!(
                paths,
                vec![PathBuf::from("/tmp/a"), PathBuf::from("/tmp/b")]
            );
        }
        other => anyhow::bail!("unexpected command: {other:?}"),
    }
    Ok(())
}

#[test]
fn fs_hopper_flag_and_upload_alias_parse() -> Result<()> {
    for flag in ["--hopper", "--upload"] {
        let cli = Cli::try_parse_from(with_default_subcommand([
            "scan",
            "fs",
            flag,
            "http://hopper:8081",
            "/tmp/a",
        ]))
        .context("parse should work")?;
        match cli.command.context("fs subcommand expected")? {
            Commands::Path { paths, hopper, .. } => {
                assert_eq!(paths, vec![PathBuf::from("/tmp/a")]);
                assert_eq!(hopper.as_deref(), Some("http://hopper:8081"));
            }
            other => anyhow::bail!("unexpected command: {other:?}"),
        }
    }
    Ok(())
}

#[test]
fn url_and_purl_accept_hopper_flag() -> Result<()> {
    let cli = Cli::try_parse_from([
        "scan",
        "url",
        "https://h/f.tgz",
        "--hopper",
        "http://x:8081",
    ])
    .context("parse should work")?;
    match cli.command.context("url subcommand expected")? {
        Commands::Url { url, hopper } => {
            assert_eq!(url, "https://h/f.tgz");
            assert_eq!(hopper.as_deref(), Some("http://x:8081"));
        }
        other => anyhow::bail!("unexpected command: {other:?}"),
    }
    let cli = Cli::try_parse_from(["scan", "purl", "npm/left-pad", "--upload", "http://x:8081"])
        .context("parse should work")?;
    match cli.command.context("purl subcommand expected")? {
        Commands::Purl { hopper, .. } => {
            assert_eq!(hopper.as_deref(), Some("http://x:8081"));
        }
        other => anyhow::bail!("unexpected command: {other:?}"),
    }
    Ok(())
}

#[test]
fn fs_without_hopper_defaults_to_none() -> Result<()> {
    let cli = Cli::try_parse_from(with_default_subcommand(["scan", "fs", "/tmp/a"]))
        .context("parse should work")?;
    match cli.command.context("fs subcommand expected")? {
        Commands::Path { hopper, .. } => assert!(hopper.is_none()),
        other => anyhow::bail!("unexpected command: {other:?}"),
    }
    Ok(())
}

#[test]
fn path_subcommand_and_its_aliases_resolve() -> Result<()> {
    for name in ["path", "fs", "scan"] {
        let cli = Cli::try_parse_from(with_default_subcommand(["scan", name, "/tmp/a"]))
            .with_context(|| format!("parse of `{name}` should work"))?;
        match cli
            .command
            .with_context(|| format!("path expected via `{name}`"))?
        {
            Commands::Path { paths, .. } => {
                assert_eq!(paths, vec![PathBuf::from("/tmp/a")])
            }
            other => anyhow::bail!("unexpected command for `{name}`: {other:?}"),
        }
    }
    Ok(())
}

#[test]
fn purl_subcommand_and_its_aliases_resolve() -> Result<()> {
    for name in ["purl", "pkg", "package", "pkgs"] {
        let cli = Cli::try_parse_from(["scan", name, "pkg:npm/left-pad@1.3.0"])
            .with_context(|| format!("parse of `{name}` should work"))?;
        match cli
            .command
            .with_context(|| format!("purl expected via `{name}`"))?
        {
            Commands::Purl { purl, .. } => assert_eq!(purl, "pkg:npm/left-pad@1.3.0"),
            other => anyhow::bail!("unexpected command for `{name}`: {other:?}"),
        }
    }
    Ok(())
}

#[test]
fn serve_accepts_negative_max_rss_disable() -> Result<()> {
    let cli = Cli::try_parse_from([
        "scan",
        "serve",
        "--bind",
        "127.0.0.1:49999",
        "--max-rss-gb",
        "-1",
    ])
    .context("serve -1 should parse")?;
    match cli.command.context("serve subcommand expected")? {
        Commands::Serve(args) => {
            assert_eq!(args.bind, "127.0.0.1:49999".parse::<SocketAddr>()?);
            assert_eq!(args.max_rss_gb, -1);
        }
        other => anyhow::bail!("unexpected command: {other:?}"),
    }
    Ok(())
}

#[test]
fn max_rss_policy_classifies_cli_inputs() {
    assert_eq!(MaxRssPolicy::from_cli(-1), MaxRssPolicy::Disabled);
    assert_eq!(MaxRssPolicy::from_cli(i64::MIN), MaxRssPolicy::Disabled);
    assert_eq!(MaxRssPolicy::from_cli(0), MaxRssPolicy::Auto);
    assert_eq!(
        MaxRssPolicy::from_cli(7),
        MaxRssPolicy::Explicit(NonZeroU64::new(7).expect("non-zero"))
    );
}

#[test]
fn level_flag_parses_and_shortcuts_removed() -> Result<()> {
    let cli = Cli::try_parse_from(with_default_subcommand(["scan", "-l", "100", "/tmp/a"]))
        .context("-l 100 should parse")?;
    assert_eq!(cli.global.level, Some(100));

    let cli = Cli::try_parse_from(with_default_subcommand(["scan", "--level", "12", "/tmp/a"]))
        .context("--level 12 should parse")?;
    assert_eq!(cli.global.level, Some(12));

    // Out-of-range rejected (0..=25000, per-100M since the per-million migration).
    assert!(
        Cli::try_parse_from(with_default_subcommand(["scan", "-l", "25001", "/tmp/a"])).is_err()
    );
    // The numeric shortcuts and their aliases were removed.
    assert!(Cli::try_parse_from(with_default_subcommand(["scan", "-5", "/tmp/a"])).is_err());
    assert!(Cli::try_parse_from(with_default_subcommand(["scan", "--loose", "/tmp/a"])).is_err());
    assert!(
        Cli::try_parse_from(with_default_subcommand(["scan", "--paranoid", "/tmp/a"])).is_err()
    );
    Ok(())
}

#[test]
fn gzip_long_aliases_are_not_accepted() {
    assert!(Cli::try_parse_from(with_default_subcommand(["scan", "--fast", "/tmp/a"])).is_err());
    assert!(Cli::try_parse_from(with_default_subcommand(["scan", "--best", "/tmp/a"])).is_err());
}

#[test]
fn archive_password_flag_is_repeatable_and_global() -> Result<()> {
    let cli = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "path",
        "/tmp/a",
        "--zip-password",
        "one",
        "--zip-password=two",
    ]))?;

    assert_eq!(cli.global.zip_passwords, ["one", "two"]);
    Ok(())
}

#[test]
fn concurrency_flags_reject_zero() {
    assert!(Cli::try_parse_from(["atomscan", "serve", "--workers", "0"]).is_err());
}

/// The LLM environment as a test wants it: token files under `home`'s
/// `.tok`, or none, and no `SCAN_LLM*` variables — whatever the process
/// running the tests has set.
fn llm_env(home: Option<&std::path::Path>) -> LlmEnv {
    LlmEnv {
        llm_token_file: home.map(|h| h.join(".tok").join("llm")),
        openrouter_token_file: home.map(|h| h.join(".tok").join("openrouter")),
        ..LlmEnv::default()
    }
}

#[test]
fn openrouter_alias_requires_key_and_defaults_model() -> Result<()> {
    let empty_home = tempfile::tempdir()?;
    let env = llm_env(Some(empty_home.path()));
    let missing_key = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--llm",
        "openrouter",
        "--llm-model",
        "qwen/qwen3.8-27b",
        "/tmp/a",
    ]))?;
    let err = missing_key
        .global
        .interpret_config_with(&env)
        .expect_err("openrouter without a key must fail");
    assert!(
        err.to_string().contains("~/.tok/openrouter"),
        "unexpected error: {err}"
    );

    let keyed = LlmEnv {
        key: Some("sk-test".into()),
        ..env
    };
    let missing_model = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--llm",
        "openrouter",
        "/tmp/a",
    ]))?;
    let cfg = missing_model
        .global
        .interpret_config_with(&keyed)?
        .context("openrouter without a pinned model should default to auto")?;
    assert_eq!(cfg.model, scan::interpret::OPENROUTER_DEFAULT_MODEL);

    let cli = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--llm",
        "openrouter",
        "--llm-model",
        "qwen/qwen3.8-27b",
        "/tmp/a",
    ]))?;
    let cfg = cli
        .global
        .interpret_config_with(&keyed)?
        .context("openrouter with model+key should enable interpret")?;
    assert_eq!(cfg.base_url, scan::interpret::OPENROUTER_BASE_URL);
    assert_eq!(cfg.model, "qwen/qwen3.8-27b");
    assert_eq!(cfg.api_key.as_deref(), Some("sk-test"));
    Ok(())
}

#[test]
fn llm_flags_are_global_on_serve() -> Result<()> {
    let env = LlmEnv {
        key: Some("sk-test".into()),
        ..llm_env(None)
    };
    let serve = Cli::try_parse_from([
        "atomscan",
        "--llm",
        "openrouter",
        "--llm-model",
        "qwen/qwen3.8-27b",
        "serve",
    ])?;
    assert_eq!(serve.global.llm.as_deref(), Some("openrouter"));
    let cfg = serve
        .global
        .interpret_config_with(&env)?
        .context("serve inherits --llm")?;
    assert_eq!(cfg.base_url, scan::interpret::OPENROUTER_BASE_URL);
    Ok(())
}

/// `~/.tok/llm` authenticates *our* endpoint. Handing it to OpenRouter
/// would send a live credential to a third party, and arrive there as an
/// ordinary 401 that says nothing about what leaked.
#[test]
fn the_vllm_token_is_never_sent_to_openrouter() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let tok = dir.path().join(".tok");
    std::fs::create_dir(&tok)?;
    std::fs::write(tok.join("llm"), "sk-vllm\n")?;
    let env = llm_env(Some(dir.path()));
    // Alone, OpenRouter without its own key is the same hard error it
    // has always been — not a silent borrow of the vLLM token.
    let cli = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--llm",
        "openrouter",
        "--llm-model",
        "qwen/qwen3.8-27b",
        "/tmp/a",
    ]))?;
    let err = cli
        .global
        .interpret_config_with(&env)
        .expect_err("~/.tok/llm must not satisfy OpenRouter");
    assert!(err.to_string().contains("~/.tok/openrouter"), "{err}");
    Ok(())
}

#[test]
fn llm_failover_chain_resolves_each_endpoint_separately() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let tok = dir.path().join(".tok");
    std::fs::create_dir(&tok)?;
    std::fs::write(tok.join("llm"), "sk-vllm\n")?;
    std::fs::write(tok.join("openrouter"), "sk-openrouter\n")?;
    let env = llm_env(Some(dir.path()));
    // The shape we deploy by default. Both models pinned so the test
    // resolves without touching the network.
    let cli = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--llm",
        "https://llm.isotope13.ai/v1,openrouter",
        "--llm-model",
        "Qwen/Qwen3.8-27B,qwen/qwen3.8-27b",
        "/tmp/a",
    ]))?;
    let cfg = cli
        .global
        .interpret_config_with(&env)?
        .context("chain should resolve")?;
    assert_eq!(cfg.base_url, "https://llm.isotope13.ai/v1");
    assert_eq!(cfg.model, "Qwen/Qwen3.8-27B");
    // Each endpoint takes its own token file: the vLLM key must not be
    // sent to OpenRouter, nor the reverse.
    assert_eq!(cfg.api_key.as_deref(), Some("sk-vllm"));
    assert_eq!(cfg.fallbacks.len(), 1);
    assert_eq!(
        cfg.fallbacks[0].base_url,
        scan::interpret::OPENROUTER_BASE_URL
    );
    assert_eq!(cfg.fallbacks[0].model, "qwen/qwen3.8-27b");
    assert_eq!(cfg.fallbacks[0].api_key.as_deref(), Some("sk-openrouter"));
    Ok(())
}

/// A fallback that cannot be used is dropped, not fatal: the primary is
/// what the scan actually needs.
#[test]
fn an_unusable_fallback_is_dropped_but_the_primary_stands() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let tok = dir.path().join(".tok");
    std::fs::create_dir(&tok)?;
    std::fs::write(tok.join("llm"), "sk-vllm\n")?;
    let env = llm_env(Some(dir.path()));
    // No ~/.tok/openrouter, so the OpenRouter slot cannot be called
    // (its model would default to `openrouter/auto`, but the missing
    // key still drops it); the same config with OpenRouter *alone* is
    // a hard error (see openrouter_alias_requires_key_and_defaults_model).
    let cli = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--llm",
        "https://llm.isotope13.ai/v1,openrouter",
        "--llm-model",
        "Qwen/Qwen3.8-27B",
        "/tmp/a",
    ]))?;
    let cfg = cli
        .global
        .interpret_config_with(&env)?
        .context("primary should stand")?;
    assert_eq!(cfg.base_url, "https://llm.isotope13.ai/v1");
    assert!(
        cfg.fallbacks.is_empty(),
        "an OpenRouter slot with no key must not be kept: {:?}",
        cfg.fallbacks
    );
    Ok(())
}

/// ...but losing every endpoint is fatal, and says why for each.
#[test]
fn a_chain_with_no_usable_endpoint_is_an_error() -> Result<()> {
    let empty_home = tempfile::tempdir()?;
    let env = llm_env(Some(empty_home.path()));
    // Port 1 refuses immediately, so discovery fails without a wait;
    // the OpenRouter slot has neither key nor model.
    let cli = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--llm",
        "http://127.0.0.1:1/v1,openrouter",
        "/tmp/a",
    ]))?;
    let err = cli
        .global
        .interpret_config_with(&env)
        .expect_err("no endpoint is usable here");
    let text = err.to_string();
    assert!(text.contains("no usable LLM endpoint"), "{text}");
    assert!(text.contains("127.0.0.1:1"), "{text}");
    assert!(text.contains("OpenRouter"), "{text}");
    Ok(())
}

#[test]
fn llm_key_falls_back_to_tok_llm_file() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let tok = dir.path().join(".tok");
    std::fs::create_dir(&tok)?;
    std::fs::write(tok.join("llm"), "sk-vllm-file\n")?;
    let env = llm_env(Some(dir.path()));
    // A pinned model keeps this off the network: an unpinned one would
    // probe the endpoint for its catalog.
    let cli = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--llm",
        "http://interpret:8000/v1",
        "--llm-model",
        "Qwen/Qwen3.8-27B",
        "/tmp/a",
    ]))?;
    let cfg = cli
        .global
        .interpret_config_with(&env)?
        .context("~/.tok/llm should supply the key")?;
    assert_eq!(cfg.api_key.as_deref(), Some("sk-vllm-file"));

    // `SCAN_LLM_KEY` still wins over the file.
    let keyed = LlmEnv {
        key: Some("sk-env".into()),
        ..env
    };
    let cfg = cli
        .global
        .interpret_config_with(&keyed)?
        .context("explicit key")?;
    assert_eq!(cfg.api_key.as_deref(), Some("sk-env"));
    Ok(())
}

#[test]
fn openrouter_prefers_its_own_tok_file_over_tok_llm() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let tok = dir.path().join(".tok");
    std::fs::create_dir(&tok)?;
    std::fs::write(tok.join("llm"), "sk-vllm-file\n")?;
    std::fs::write(tok.join("openrouter"), "sk-openrouter-file\n")?;
    let env = llm_env(Some(dir.path()));
    let cli = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--llm",
        "openrouter",
        "--llm-model",
        "qwen/qwen3.8-27b",
        "/tmp/a",
    ]))?;
    let cfg = cli
        .global
        .interpret_config_with(&env)?
        .context("openrouter key")?;
    assert_eq!(cfg.api_key.as_deref(), Some("sk-openrouter-file"));
    Ok(())
}

#[test]
fn no_tok_llm_file_leaves_the_endpoint_unauthenticated() -> Result<()> {
    let empty_home = tempfile::tempdir()?;
    let env = llm_env(Some(empty_home.path()));
    let cli = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--llm",
        "http://interpret:8000/v1",
        "--llm-model",
        "Qwen/Qwen3.8-27B",
        "/tmp/a",
    ]))?;
    let cfg = cli
        .global
        .interpret_config_with(&env)?
        .context("local target")?;
    assert_eq!(cfg.api_key, None);
    Ok(())
}

#[test]
fn openrouter_key_falls_back_to_tok_file() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let tok = dir.path().join(".tok");
    std::fs::create_dir(&tok)?;
    std::fs::write(tok.join("openrouter"), "sk-from-file\n")?;
    let env = llm_env(Some(dir.path()));
    let cli = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--llm",
        "openrouter",
        "--llm-model",
        "qwen/qwen3.8-27b",
        "/tmp/a",
    ]))?;
    let cfg = cli
        .global
        .interpret_config_with(&env)?
        .context("tok file should supply key")?;
    assert_eq!(cfg.api_key.as_deref(), Some("sk-from-file"));
    Ok(())
}

/// A flag's value is not a path: the shorthand inserts `fs` only when an
/// operand clap would actually see is present, and leaves a named subcommand
/// alone even when a value-taking flag comes first.
#[test]
fn the_shorthand_skips_flag_values() -> Result<()> {
    let rewrites = |args: &[&str]| {
        let rewritten = with_default_subcommand(args.iter().copied());
        rewritten.get(1).is_some_and(|arg| arg == "fs") && rewritten.len() == args.len() + 1
    };
    assert!(!rewrites(&["scan", "--format", "json"]), "no path to scan");
    assert!(
        !rewrites(&["scan", "-l", "25"]),
        "a short flag's value is no path"
    );
    assert!(!rewrites(&["scan", "--format", "json", "serve"]));
    assert!(rewrites(&["scan", "--format=json", "/tmp/a"]));
    assert!(rewrites(&["scan", "-l25", "/tmp/a"]));
    assert!(
        rewrites(&["scan", "--", "serve"]),
        "after `--` it is a path"
    );
    assert!(!rewrites(&["scan", "--"]));

    let cli = Cli::try_parse_from(with_default_subcommand([
        "scan", "--format", "json", "/tmp/a",
    ]))
    .context("a path after a flag value should parse")?;
    assert_eq!(cli.global.format, OutputFormat::Json);
    match cli.command.context("a subcommand")? {
        Commands::Path { paths, .. } => assert_eq!(paths, vec![PathBuf::from("/tmp/a")]),
        other => anyhow::bail!("unexpected command: {other:?}"),
    }
    Ok(())
}
