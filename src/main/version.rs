//! `atomscan version`: this build, and the identity and size of every rule
//! source it would scan with.

use std::io::Write as _;
use std::path::Path;

/// Print the version report in `format`. A closed pipe is an error, not a
/// panic.
pub(super) fn print(format: scan::OutputFormat) -> std::io::Result<()> {
    let bloom = scan::bloom_repo::installed_manifest();
    let nsrl = scan::nsrl_repo::installed_manifest();
    if matches!(
        format,
        scan::OutputFormat::Json | scan::OutputFormat::JsonFull
    ) {
        let version = json(bloom.as_ref(), nsrl.as_ref());
        return writeln!(std::io::stdout(), "{version}");
    }
    // Counting traits/composites/YARA initialises the rule engines.
    let info = cleave::version_info();
    let bloom_count: u64 = bloom
        .as_ref()
        .map_or(0, |m| m.filter.values().map(|e| e.n).sum());
    let rules = info.trait_count as u64 + info.composite_count as u64 + info.yara_rules as u64;
    scan::output::print_version(
        env!("CARGO_PKG_VERSION"),
        &info.traits_edition.styled(),
        bloom_count,
        rules,
        &rows(&info, bloom.as_ref(), nsrl.as_ref()),
    )?;
    if let Some(hint) = scan::models_repo::outdated_spec_hint() {
        writeln!(std::io::stdout(), "{hint}")?;
    }
    Ok(())
}

/// `atomscan -f json version`: the build and each rule source's identity.
fn json(bloom: Option<&burton::Manifest>, nsrl: Option<&burton::Manifest>) -> serde_json::Value {
    let bloom_json = bloom.map(|m| {
        let filters: serde_json::Map<String, serde_json::Value> = m
            .filter
            .iter()
            .map(|(stem, e)| {
                (
                    stem.clone(),
                    serde_json::json!({
                        "n": e.n,
                        // One format per bundle now; kept per filter so the
                        // JSON keeps its shape.
                        "format_version": burton::FORMAT_VERSION,
                        "sha256": e.sha256,
                    }),
                )
            })
            .collect();
        serde_json::json!({
            "built": m.built,
            "schema": m.schema,
            "filters": filters,
        })
    });
    // Dates beside the commits, so a benchmark comparing two rules channels
    // can say how fresh each one was, not only which it was.
    let traits_date =
        cleave::rule_update::installed(&cleave::traits_repo::install_target()).map(|i| i.date);
    let models_date = scan::models_repo::install_target()
        .ok()
        .and_then(|dir| scan::model_update::installed(&dir))
        .map(|i| i.date);
    serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "models": scan::models_repo::version(),
        "models_date": models_date,
        "traits": cleave::traits_repo::version(),
        "traits_date": traits_date,
        "traits_channel": cleave::traits_repo::Edition::current().as_str(),
        "bloom": bloom_json,
        "nsrl": nsrl,
        "nsrl_release": scan::nsrl_repo::installed_release(),
    })
}

/// The terminal `version` table: one row per rule source.
fn rows(
    info: &cleave::VersionInfo,
    bloom: Option<&burton::Manifest>,
    nsrl: Option<&burton::Manifest>,
) -> Vec<scan::output::SourceRow> {
    use scan::output::SourceRow;

    // Atomics, composites, and third-party YARA all ship in the traits repo, so
    // they share one identity and build date. Prefer the installed bundle's
    // sidecar (commit + committer date); for a dev checkout with no sidecar,
    // read git HEAD directly.
    let traits_dir = cleave::cache::traits_path();
    let (traits_version, traits_epoch) = match cleave::rule_update::installed(&traits_dir) {
        Some(installed) => (
            installed.commit.chars().take(9).collect(),
            scan::output::parse_ymd(&installed.date),
        ),
        None => git_head(&traits_dir)
            .map_or_else(Default::default, |(commit, epoch)| (commit, Some(epoch))),
    };

    let mut rows = Vec::new();
    if let Some(nsrl) = nsrl {
        rows.push(SourceRow {
            label: "nsrl",
            count: nsrl.filter.values().map(|e| e.n).sum(),
            metric: None,
            version: scan::nsrl_repo::installed_release()
                .unwrap_or_else(|| format!("v{}", burton::FORMAT_VERSION)),
            epoch: scan::output::parse_ymd(&nsrl.built),
        });
    }
    if let Some(bloom) = bloom {
        rows.push(SourceRow {
            label: "blooms",
            count: bloom.filter.values().map(|e| e.n).sum(),
            metric: None,
            version: format!("v{}", burton::FORMAT_VERSION),
            epoch: scan::output::parse_ymd(&bloom.built),
        });
    }
    for (label, count) in [
        ("atomics", info.trait_count),
        ("composites", info.composite_count),
        ("third-party YARA", info.yara_rules),
    ] {
        rows.push(SourceRow {
            label,
            count: count as u64,
            metric: None,
            version: traits_version.clone(),
            epoch: traits_epoch,
        });
    }

    // ML models: route-model count by feature dimensionality (e.g. `64 x 229`),
    // plus the bundle's commit and build date.
    if let Some(count) = scan::models_repo::model_count() {
        let model_installed = scan::models_repo::install_target()
            .ok()
            .and_then(|dir| scan::model_update::installed(&dir));
        let metric = scan::models_repo::feature_dimension().map(|dim| format!("{count} x {dim}"));
        rows.push(SourceRow {
            label: "ML models",
            count: count as u64,
            metric,
            // Truncate to 9 chars to match the traits commit width.
            version: scan::models_repo::version()
                .map(|c| c.chars().take(9).collect())
                .unwrap_or_default(),
            epoch: model_installed
                .as_ref()
                .and_then(|i| scan::output::parse_ymd(&i.date)),
        });
    }
    rows
}

/// The abbreviated commit hash and committer date (Unix seconds) of `HEAD` in
/// `dir`, when `dir` is a git checkout. Returns `None` if git is unavailable or
/// `dir` is not a repository, in which case `scan version` omits that source's
/// identity. This recovers the authentic upstream identity and date for a traits
/// checkout that has no install sidecar.
fn git_head(dir: &Path) -> Option<(String, i64)> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["show", "-s", "--format=%H %ct", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8(out.stdout).ok()?;
    let mut fields = stdout.split_whitespace();
    let commit: String = fields.next()?.chars().take(9).collect();
    let epoch: i64 = fields.next()?.parse().ok()?;
    Some((commit, epoch))
}
