//! Local, on-disk cache of fetched-dependency analysis results, keyed by content
//! sha256 and the name the bytes were analyzed under.
//!
//! Fetching already caches a dependency's *bytes* (fletch's blob cache), so a
//! warm re-run doesn't re-download — but the *analysis* of those bytes was
//! recomputed every time, and for a dependency that ships a large native binary
//! that is a minutes-long, single-threaded disassembly repeated on every scan.
//! This memoizes the analysis: the finalized sub-report and the next-hop
//! references a payload yields are serialized under its content sha256 and
//! name ([`entry_key`]), so a later scan of the same bytes reuses them instead
//! of re-running cleave.
//!
//! Correctness over speed: the cache is namespaced by a *ruleset version* token
//! (scan release, Cleave's rule-content/build fingerprint, the bloom set the
//! skip-predicate consults, and the installed model bundle, whose output the
//! cached verdict is). Any change to what the detector would find lands in a
//! different namespace, so a stale result can never mask a detection a newer
//! ruleset adds — a version bump simply misses and re-analyzes.
//! A hit is only ever a result the *current* detector already produced. Set
//! `SCAN_NO_ANALYSIS_CACHE=1` to disable it entirely.

use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use cleave::AnalysisReport;
use fletch::Reference;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A payload's cached analysis: the finalized sub-report to graft (absent when
/// the bytes couldn't be analyzed) and the next-hop references found in them.
#[derive(Deserialize)]
pub(crate) struct Cached {
    pub(crate) sub: Option<AnalysisReport>,
    pub(crate) next: Vec<(String, Vec<Reference>)>,
}

/// The same payload, borrowed for serialization — so storing never clones the
/// (potentially large) report.
#[derive(Serialize)]
struct StoreRef<'a> {
    sub: Option<&'a AnalysisReport>,
    next: &'a [(String, Vec<Reference>)],
}

/// Prefix of every namespace directory, ahead of [`ruleset_version`]. Bumped
/// when reference discovery changes what a cached next-hop list means, which
/// the trait set alone does not capture: v4 requires Python import candidates
/// to pass namespace verification.
const NAMESPACE_PREFIX: &str = "dependency-v4-";

/// zstd level for cached entries: the default, cheap next to the analysis.
const ZSTD_LEVEL: i32 = 3;

/// Handle to the cache directory for the current ruleset version.
pub(crate) struct AnalysisCache {
    dir: PathBuf,
}

/// Root of the analysis cache (`…/atomdrift/scan/analysis`), above the
/// per-ruleset-version subdirectory. `None` when the OS has no cache directory.
/// Used by [`crate::cache_cleanup`] to reclaim the store; entries live at
/// `analysis/<version>/<key>.zst` (two levels below this root).
pub(crate) fn cache_base() -> Option<PathBuf> {
    Some(crate::cache_dir()?.join("analysis"))
}

impl AnalysisCache {
    /// Open (creating on first use) the cache directory for the active ruleset
    /// version. `None` when disabled via `SCAN_NO_ANALYSIS_CACHE`, when the OS
    /// has no cache directory, or when the directory can't be created — every
    /// such case degrades to "always analyze", never an error.
    pub(crate) fn open() -> Option<Self> {
        if crate::env::flag(crate::env::SCAN_NO_ANALYSIS_CACHE) {
            return None;
        }
        let base = cache_base()?;
        let version = format!("{NAMESPACE_PREFIX}{}", ruleset_version());
        prune_stale_versions(&base, &version);
        let dir = base.join(version);
        if let Err(error) = std::fs::create_dir_all(&dir) {
            tracing::debug!(dir = %dir.display(), %error, "analysis cache: cannot create namespace; caching disabled");
            return None;
        }
        Some(Self { dir })
    }

    /// Reuse a prior analysis under `key` (see [`entry_key`]), or `None` on a miss (no
    /// entry, or an unreadable/garbled one — treated as a miss so a corrupt file
    /// self-heals on the next write).
    pub(crate) fn get(&self, key: &str) -> Option<Cached> {
        let bytes = match std::fs::read(self.path(key)) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
            Err(error) => {
                tracing::debug!(key, %error, "analysis cache: unreadable entry; analyzing");
                return None;
            }
        };
        let cached: Cached = match zstd::decode_all(&bytes[..])
            .map_err(|e| e.to_string())
            .and_then(|json| serde_json::from_slice(&json).map_err(|e| e.to_string()))
        {
            Ok(cached) => cached,
            Err(error) => {
                tracing::debug!(key, %error, "analysis cache: garbled entry; analyzing");
                return None;
            }
        };
        (!cached.sub.as_ref().is_some_and(incomplete)).then_some(cached)
    }

    /// Store an analysis under `key` (see [`entry_key`]). Best-effort: any failure
    /// (serialize, compress, write) leaves the cache untouched and is logged at
    /// debug — the result is already in hand, the cache is only an
    /// optimization. An [`incomplete`] analysis is not stored.
    pub(crate) fn put(
        &self,
        key: &str,
        sub: Option<&AnalysisReport>,
        next: &[(String, Vec<Reference>)],
    ) {
        if sub.is_some_and(incomplete) {
            return;
        }
        let compressed = match serde_json::to_vec(&StoreRef { sub, next }) {
            Ok(json) => zstd::encode_all(&json[..], ZSTD_LEVEL),
            Err(error) => {
                tracing::debug!(key, %error, "analysis cache: cannot serialize entry; not stored");
                return;
            }
        };
        // Write to a unique temp file, then rename into place — a reader never
        // sees a half-written entry, and concurrent writers (in any process)
        // don't collide. Another process may have pruned this namespace as
        // idle (`prune_stale_versions`); recreate it rather than stop caching.
        let stored = compressed.and_then(|compressed| {
            let mut tmp = tempfile::NamedTempFile::new_in(&self.dir).or_else(|_| {
                std::fs::create_dir_all(&self.dir)?;
                tempfile::NamedTempFile::new_in(&self.dir)
            })?;
            tmp.write_all(&compressed)?;
            tmp.persist(self.path(key)).map_err(|e| e.error)?;
            Ok(())
        });
        if let Err(error) = stored {
            tracing::debug!(key, %error, "analysis cache: cannot write entry; not stored");
        }
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.zst"))
    }
}

/// The cache key for `content_sha` analyzed under `name`. The name is part of
/// the analysis: cleave types a file by its name before its content (a `.env`
/// is text), and the reference hunt reads it too. Keyed by sha alone, bytes
/// first fetched under a misleading name would serve that shallow result to
/// every later fetch of the same bytes under their real name.
pub(crate) fn entry_key(content_sha: &str, name: &str) -> String {
    let digest = Sha256::digest(name.as_bytes());
    let mut key = format!("{content_sha}-");
    for b in &digest[..8] {
        let _ = write!(key, "{b:02x}");
    }
    key
}

/// Whether `report` or any file in it is missing results a later run may
/// produce: rule evaluation ran out of time (findings depend on how loaded the
/// machine was) or a source parse was cut short.
/// Caching it would serve the shortfall to every later scan of the same bytes.
/// cleave's own cache makes the same call.
fn incomplete(report: &AnalysisReport) -> bool {
    use cleave::types::{AnalysisGap, AnalysisGaps};
    let incomplete = |gaps: &AnalysisGaps| {
        [
            AnalysisGap::EvaluationDeadline,
            AnalysisGap::SourceParseIncomplete,
        ]
        .into_iter()
        .any(|gap| gaps.contains(gap))
    };
    incomplete(&report.analysis_gaps) || report.files.iter().any(|f| incomplete(&f.analysis_gaps))
}

/// A token identifying the analysis-producing detector, so a rules, model, or
/// engine update invalidates cached results. Folds in the scan release, the
/// rule contents and analyzer build, the installed bloom and NSRL sets (which
/// the dependency skip-predicate consults), and the installed model bundle.
///
/// Reuse Cleave's content fingerprint, shared with rule loading and memoized
/// until reload. It does not compile rules or read their sources a second
/// time. A Git commit alone misses uncommitted edits; directories without
/// Git metadata would all share a single namespace.
///
/// cleave's compact schema version belongs here too: a cached entry holds a
/// report in that format, and a format change must not replay entries written
/// in the previous one.
///
/// The model belongs here for the same reason the rules do: a cached entry
/// holds the *verdict*, and the verdict is the model's output. Without it a
/// model auto-update would reuse the previous bundle's namespace, and every
/// already-seen file would keep the verdict that bundle gave it, so a defect
/// in one bundle would outlive the corrected bundle that fixes it.
pub(crate) fn ruleset_version() -> String {
    let bloom = crate::bloom_repo::installed_manifest()
        .map_or_else(|| "nobloom".to_string(), |m| bloom_token(&m));
    let nsrl = crate::nsrl_repo::installed_manifest().map_or_else(
        || "nonsrl".to_string(),
        |m| format!("nsrl-{}", bloom_token(&m)),
    );
    let model = model_token(
        crate::models_repo::version().as_deref(),
        crate::models_repo::install_target().ok().as_deref(),
    );
    ruleset_token(
        cleave::cache::traits_revision_fingerprint(),
        &bloom,
        &nsrl,
        &model,
    )
}

/// [`ruleset_version`] from its inputs: every one of them, so none can be
/// dropped from the key without a compile error.
fn ruleset_token(traits: i64, bloom: &str, nsrl: &str, model: &str) -> String {
    format!(
        "{}-c{}-r{traits:016x}-{bloom}-{nsrl}-m{model}",
        env!("CARGO_PKG_VERSION"),
        cleave::types::CompactReport::default().version,
    )
}

/// Identity of the installed bloom set, for [`ruleset_version`]: a short hash
/// of every filter's sha256. Not the manifest's `built` date, which is coarse
/// to the day and so blind to a same-day rebuild.
fn bloom_token(manifest: &burton::Manifest) -> String {
    let mut hasher = Sha256::new();
    for (stem, entry) in &manifest.filter {
        hasher.update(stem.as_bytes());
        hasher.update(b"=");
        hasher.update(entry.sha256.as_bytes());
        hasher.update(b"\n");
    }
    let mut token = format!("b{:x}", hasher.finalize());
    token.truncate(13); // "b" and 12 hex digits
    token
}

/// Identity of the installed model bundle, for [`ruleset_version`].
///
/// Prefers the updater's sidecar commit. A bundle installed by hand — a
/// development build symlinked into place — has no sidecar, and that is
/// exactly the setup where bundles change most often, so fall back to a stamp of
/// `config.json`'s size and mtime. Either way a redeployed bundle moves the
/// token. `nomodel` when nothing is installed.
fn model_token(commit: Option<&str>, install_dir: Option<&Path>) -> String {
    if let Some(commit) = commit {
        return sanitize(commit);
    }
    let Some(meta) = install_dir.and_then(|dir| std::fs::metadata(dir.join("config.json")).ok())
    else {
        return "nomodel".to_string();
    };
    let stamp = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    format!("{}.{stamp}", meta.len())
}

/// How long a namespace must go unwritten before another version prunes it.
/// The cache sweep ([`crate::cache_cleanup`]) already drops entries this old
/// by default, so an idle namespace holds nothing it would keep.
const IDLE_NAMESPACE_AGE: Duration =
    Duration::from_secs(30 * crate::civil::SECS_PER_DAY.unsigned_abs());

/// Delete namespace directories under `base`, other than `current`, that no
/// process has written to in [`IDLE_NAMESPACE_AGE`]. Every rules, engine, or
/// bloom update starts a fresh namespace, and the superseded ones would
/// otherwise pile up. Not "everything but mine": a long-running server on the
/// previous ruleset still writes to its own namespace. Best-effort, like the
/// cache itself.
pub(crate) fn prune_stale_versions(base: &Path, current: &str) {
    prune_idle(base, current, SystemTime::now());
}

fn prune_idle(base: &Path, current: &str, now: SystemTime) {
    let Ok(entries) = std::fs::read_dir(base) else {
        return;
    };
    for entry in entries.flatten() {
        let idle = entry.metadata().is_ok_and(|m| {
            m.is_dir()
                && m.modified()
                    .is_ok_and(|t| now.duration_since(t).unwrap_or_default() > IDLE_NAMESPACE_AGE)
        });
        if idle
            && entry.file_name().to_str() != Some(current)
            && let Err(error) = std::fs::remove_dir_all(entry.path())
        {
            tracing::debug!(dir = %entry.path().display(), %error, "analysis cache: cannot prune idle namespace");
        }
    }
}

/// Fold anything but `[A-Za-z0-9._-]` to `-`, so a token is a safe path segment.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run in a child process: changing Cleave's process-wide traits source
    /// must not affect other tests running concurrently.
    #[test]
    fn local_trait_edits_invalidate_cached_results() {
        const PROBE: &str = "SCAN_TEST_LOCAL_TRAITS_CACHE";
        #[expect(
            clippy::disallowed_methods,
            reason = "this test's own child-process marker"
        )]
        let child = std::env::var_os(PROBE).is_some();
        if !child {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "analysis_cache::tests::local_trait_edits_invalidate_cached_results",
                    "--nocapture",
                ])
                .env(PROBE, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            return;
        }

        let root = tempfile::tempdir().unwrap();
        let traits = root.path().join("traits");
        let yaml = traits.join("objectives/probe/local.yaml");
        std::fs::create_dir_all(yaml.parent().unwrap()).unwrap();
        std::fs::create_dir(traits.join(".git")).unwrap();
        let head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n";
        std::fs::write(traits.join(".git/HEAD"), head).unwrap();
        let original = "traits:\n  - id: local\n    if: {type: text, exact: cache_aaa}\n";
        std::fs::write(&yaml, original).unwrap();
        let mtime = std::fs::metadata(&yaml).unwrap().modified().unwrap();
        let version = || {
            // A fresh CLI invocation or an explicit reload re-reads the tree.
            cleave::traits_repo::set_override_dir(Some(traits.clone()));
            ruleset_version()
        };
        let cache = |version: &str| {
            let dir = root.path().join(format!("{NAMESPACE_PREFIX}{version}"));
            std::fs::create_dir_all(&dir).unwrap();
            AnalysisCache { dir }
        };
        let before = version();
        let warm = cache(&before);
        let report = AnalysisReport::new(cleave::TargetInfo {
            path: "sample.txt".into(),
            file_type: "text".into(),
            size_bytes: 1,
            sha256: "sample".into(),
            architectures: None,
        });
        warm.put("sample", Some(&report), &[]);
        assert!(
            cache(&version()).get("sample").is_some(),
            "unchanged rules hit"
        );

        // Same commit, same path, same size, same mtime; different rule bytes.
        std::fs::write(&yaml, original.replace("cache_aaa", "cache_bbb")).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&yaml)
            .unwrap()
            .set_modified(mtime)
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(traits.join(".git/HEAD")).unwrap(),
            head
        );
        let edited = version();
        assert_ne!(before, edited, "local edits must move the cache namespace");
        assert!(
            cache(&edited).get("sample").is_none(),
            "old findings must miss"
        );

        // Newly added rules count even though Git does not track them yet.
        let untracked = traits.join("objectives/probe/new.yar");
        std::fs::write(&untracked, "rule added { condition: true }\n").unwrap();
        assert_ne!(edited, version(), "untracked rules must invalidate");
        std::fs::remove_file(&untracked).unwrap();
        assert_eq!(edited, version(), "removing the rule restores the key");

        // Git metadata is not an input: archive snapshots use the same key.
        std::fs::remove_dir_all(traits.join(".git")).unwrap();
        assert_eq!(edited, version(), "snapshots must use rule contents");
        std::fs::write(&yaml, original).unwrap();
        assert_eq!(before, version(), "reverting rules restores the warm cache");
        assert!(cache(&version()).get("sample").is_some());
    }

    #[test]
    fn pruning_spares_live_namespaces_of_other_versions() {
        let base = tempfile::tempdir().unwrap();
        for ns in ["mine", "theirs"] {
            std::fs::create_dir(base.path().join(ns)).unwrap();
        }
        // Just written: another process's live namespace survives.
        prune_idle(base.path(), "mine", SystemTime::now());
        assert!(base.path().join("theirs").is_dir());

        // A month and a day later, nobody has written to either; only the
        // current version's namespace survives.
        let later = SystemTime::now()
            + IDLE_NAMESPACE_AGE
            + Duration::from_secs(crate::civil::SECS_PER_DAY.unsigned_abs());
        prune_idle(base.path(), "mine", later);
        assert!(base.path().join("mine").is_dir());
        assert!(!base.path().join("theirs").exists());
    }

    #[test]
    fn a_same_day_bloom_rebuild_moves_the_token() {
        let manifest = |sha: &str| -> burton::Manifest {
            toml::from_str(&format!(
                "schema = 2\nbuilt = \"2026-10-01\"\nkey_scheme = \"test/v1\"\n[filter.purl-good]\nfile = \"purl-good.adbl\"\nsha256 = \"{sha}\"\nn = 1\n"
            ))
            .unwrap()
        };
        let (a, b) = (manifest("aa"), manifest("bb"));
        assert_eq!(a.built, b.built);
        assert_ne!(bloom_token(&a), bloom_token(&b));
        assert_eq!(bloom_token(&a), bloom_token(&manifest("aa")));
    }

    #[test]
    fn entry_key_separates_names_of_the_same_bytes() {
        let sha = "ab".repeat(32);
        assert_eq!(entry_key(&sha, "setup.py"), entry_key(&sha, "setup.py"));
        assert_ne!(entry_key(&sha, "setup.py"), entry_key(&sha, ".env"));
        assert!(
            entry_key(&sha, "../x/y")
                .chars()
                .all(|c| c.is_ascii_hexdigit() || c == '-')
        );
    }

    #[test]
    fn put_then_get_round_trips_and_survives_a_pruned_namespace() {
        let base = tempfile::tempdir().unwrap();
        let cache = AnalysisCache {
            dir: base.path().join("ns"),
        };
        // The namespace was pruned by another process after this one opened it.
        cache.put("abc", None, &[("k".to_string(), Vec::new())]);
        let hit = cache.get("abc").unwrap();
        assert!(hit.sub.is_none());
        assert_eq!(hit.next.len(), 1);
        assert!(cache.get("missing").is_none());
    }

    /// A report cut short by the rule deadline or an
    /// interrupted source parse lacks results a later scan would have, so it
    /// must not be served to one.
    #[test]
    fn incomplete_analyses_are_not_stored() {
        use cleave::types::AnalysisGap::{EvaluationDeadline, SourceParseIncomplete};
        let base = tempfile::tempdir().unwrap();
        let cache = AnalysisCache {
            dir: base.path().to_path_buf(),
        };
        let report = || {
            AnalysisReport::new(cleave::TargetInfo {
                path: "x.so".into(),
                file_type: "elf".into(),
                size_bytes: 1,
                sha256: "abc".into(),
                architectures: None,
            })
        };

        cache.put("complete", Some(&report()), &[]);
        assert!(cache.get("complete").is_some());
        for gap in [EvaluationDeadline, SourceParseIncomplete] {
            let cut_short = report();
            cut_short.analysis_gaps.record(gap);
            cache.put(gap.label(), Some(&cut_short), &[]);
            assert!(cache.get(gap.label()).is_none(), "{gap:?}");
        }
    }

    /// The cache key holds verdicts, and verdicts are the model's output — a
    /// bundle swap must land them in a fresh namespace. Guards against the
    /// model token being dropped from the key again.
    #[test]
    fn ruleset_version_carries_the_model_token() {
        let token = model_token(Some("abc123"), None);
        let version = ruleset_token(7, "nobloom", "nonsrl", &token);
        assert!(
            version.ends_with(&format!("-m{token}")),
            "ruleset_version {version} must carry model token {token}",
        );
        assert_ne!(
            version,
            ruleset_token(7, "nobloom", "nonsrl", &model_token(Some("def456"), None)),
            "a model swap must move the namespace"
        );
    }

    #[test]
    fn model_version_is_stable_and_path_safe() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        let path_safe = |token: &str| {
            token
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        };
        for (commit, install) in [
            (Some("v1/../ab c"), None),
            (None, Some(dir.path())),
            (None, None),
        ] {
            let token = model_token(commit, install);
            assert_eq!(token, model_token(commit, install));
            assert!(
                path_safe(&token),
                "model token must be a safe path segment: {token}"
            );
        }
        assert_eq!(model_token(None, None), "nomodel");
    }

    #[test]
    fn historical_partial_cache_entry_is_rejected_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let cache = AnalysisCache {
            dir: dir.path().to_owned(),
        };
        let report = AnalysisReport::new(cleave::TargetInfo {
            path: "binary".into(),
            file_type: "elf".into(),
            size_bytes: 1,
            sha256: "content".into(),
            architectures: None,
        });
        report
            .analysis_gaps
            .record(cleave::types::AnalysisGap::EvaluationDeadline);
        let sub = Some(report);
        let next = vec![];
        let bytes = serde_json::to_vec(&StoreRef {
            sub: sub.as_ref(),
            next: &next,
        })
        .unwrap();
        let historical = String::from_utf8(bytes)
            .unwrap()
            .replace("evaluation-deadline", "disassembly-incomplete");
        std::fs::write(
            cache.path("content"),
            zstd::encode_all(historical.as_bytes(), 3).unwrap(),
        )
        .unwrap();
        assert!(cache.get("content").is_none());
    }
}
