use super::*;
#[test]
fn resume_scope_and_native_retry_flags_parse_and_reject_invalid_values() {
    let cli = Cli::try_parse_from(with_default_subcommand([
        "atomscan",
        "--follow=all",
        "--fetch-pending=/tmp/backlog.json",
        "--fetch-dev-deps=false",
        "--fetch-all-versions",
        "--rizin-timeout-secs=60",
        "--rizin-retry-timeout-secs=120",
        "/tmp/sample",
    ]))
    .unwrap();
    assert_eq!(
        cli.global.fetch_pending.as_deref(),
        Some(std::path::Path::new("/tmp/backlog.json"))
    );
    assert_eq!(cli.global.rizin_retry_timeout_secs, Some(120));
    let policy = cli
        .global
        .fetch_policy(cli.global.follow.unwrap(), 0, false);
    assert!(!policy.include_dev_dependencies);
    assert!(policy.all_versions);
    for flag in ["--rizin-retry-timeout-secs=0", "--fetch-dev-deps=maybe"] {
        assert!(
            Cli::try_parse_from(with_default_subcommand(["atomscan", flag, "/tmp/sample"]))
                .is_err()
        );
    }
}
#[test]
fn audit_defaults_preserve_development_coverage_and_disable_extra_native_attempt() {
    let cli = Cli::try_parse_from(with_default_subcommand(["atomscan", "/tmp/sample"])).unwrap();
    assert!(cli.global.fetch_dev_deps);
    assert!(!cli.global.fetch_all_versions);
    assert!(cli.global.fetch_pending.is_none());
    assert!(cli.global.rizin_retry_timeout_secs.is_none());
}
