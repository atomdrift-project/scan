//! Local, on-disk cache of fetched-dependency analysis results, keyed by content
//! sha256.
//!
//! Fetching already caches a dependency's *bytes* (fletch's blob cache), so a
//! warm re-run doesn't re-download — but the *analysis* of those bytes was
//! recomputed every time, and for a dependency that ships a large native binary
//! that is a minutes-long, single-threaded disassembly repeated on every scan.
//! This memoizes the analysis: the finalized sub-report and the next-hop
//! references a payload yields are serialized under its content sha256, so a
//! later scan of the same bytes reuses them instead of re-running cleave.
//!
//! Correctness over speed: the cache is namespaced by a *ruleset version* token
//! (scan release, installed traits commit, trait/composite/YARA counts, the
//! content of the bloom set the skip-predicate consults, and the installed
//! model bundle, whose output the cached verdict is). Any change to what the
//! detector would find lands in a different namespace, so a stale result can
//! never mask a detection a newer ruleset adds — a version bump simply misses
//! and re-analyzes.
//! A hit is only ever a result the *current* detector already produced. Set
//! `SCAN_ANALYSIS_CACHE=0` to disable it entirely.

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
    pub sub: Option<AnalysisReport>,
    pub next: Vec<(String, Vec<Reference>)>,
}

/// The same payload, borrowed for serialization — so storing never clones the
/// (potentially large) report.
#[derive(Serialize)]
struct StoreRef<'a> {
    sub: &'a Option<AnalysisReport>,
    next: &'a [(String, Vec<Reference>)],
}

/// Handle to the cache directory for the current ruleset version.
pub(crate) struct AnalysisCache {
    dir: PathBuf,
}

/// Root of the analysis cache (`…/atomdrift/scan/analysis`), above the
/// per-ruleset-version subdirectory. `None` when the OS has no cache directory.
/// Used by [`crate::cache_cleanup`] to reclaim the store; entries live at
/// `analysis/<version>/<sha>.zst` (two levels below this root).
pub(crate) fn cache_base() -> Option<PathBuf> {
    Some(
        dirs::cache_dir()?
            .join("atomdrift")
            .join("scan")
            .join("analysis"),
    )
}

impl AnalysisCache {
    /// Open (creating on first use) the cache directory for the active ruleset
    /// version. `None` when disabled via `SCAN_ANALYSIS_CACHE=0`, when the OS
    /// has no cache directory, or when the directory can't be created — every
    /// such case degrades to "always analyze", never an error.
    pub(crate) fn open() -> Option<Self> {
        if std::env::var("SCAN_ANALYSIS_CACHE").is_ok_and(|v| v == "0" || v == "false") {
            return None;
        }
        let base = cache_base()?;
        let version = ruleset_version();
        prune_stale_versions(&base, &version);
        let dir = base.join(version);
        std::fs::create_dir_all(&dir).ok()?;
        Some(Self { dir })
    }

    /// Reuse a prior analysis of these exact bytes, or `None` on a miss (no
    /// entry, or an unreadable/garbled one — treated as a miss so a corrupt file
    /// self-heals on the next write).
    pub(crate) fn get(&self, content_sha: &str) -> Option<Cached> {
        let bytes = std::fs::read(self.path(content_sha)).ok()?;
        let json = zstd::decode_all(&bytes[..]).ok()?;
        serde_json::from_slice(&json).ok()
    }

    /// Store an analysis under its content sha. Best-effort: any failure
    /// (serialize, compress, write) leaves the cache untouched and is silent —
    /// the result is already in hand, the cache is only an optimization. An
    /// [`incomplete`] analysis is not stored.
    pub(crate) fn put(
        &self,
        content_sha: &str,
        sub: &Option<AnalysisReport>,
        next: &[(String, Vec<Reference>)],
    ) {
        if sub.as_ref().is_some_and(incomplete) {
            return;
        }
        let Ok(json) = serde_json::to_vec(&StoreRef { sub, next }) else {
            return;
        };
        let Ok(compressed) = zstd::encode_all(&json[..], 3) else {
            return;
        };
        // Write to a unique temp file, then rename into place — a reader never
        // sees a half-written entry, and concurrent writers (in any process)
        // don't collide. Another process may have pruned this namespace as
        // idle (`prune_stale_versions`); recreate it rather than stop caching.
        let tmp = tempfile::NamedTempFile::new_in(&self.dir).or_else(|_| {
            std::fs::create_dir_all(&self.dir)?;
            tempfile::NamedTempFile::new_in(&self.dir)
        });
        if let Ok(mut tmp) = tmp
            && tmp.write_all(&compressed).is_ok()
        {
            let _ = tmp.persist(self.path(content_sha));
        }
    }

    fn path(&self, content_sha: &str) -> PathBuf {
        self.dir.join(format!("{content_sha}.zst"))
    }
}

/// Whether `report` or any file in it is missing results a later run may
/// produce: rule evaluation ran out of time (findings depend on how loaded the
/// machine was), Rizin did not finish, or a source parse was cut short.
/// Caching it would serve the shortfall to every later scan of the same bytes.
/// cleave's own cache makes the same call.
fn incomplete(report: &AnalysisReport) -> bool {
    use cleave::types::{AnalysisGap, AnalysisGaps};
    let incomplete = |gaps: &AnalysisGaps| {
        [
            AnalysisGap::EvaluationDeadline,
            AnalysisGap::DisassemblyIncomplete,
            AnalysisGap::SourceParseIncomplete,
        ]
        .into_iter()
        .any(|gap| gaps.contains(gap))
    };
    incomplete(&report.analysis_gaps) || report.files.iter().any(|f| incomplete(&f.analysis_gaps))
}

/// A token identifying the analysis-producing detector, so a rules, model, or
/// engine update invalidates cached results. Folds in the scan release, the
/// installed traits commit, cleave's trait/composite/YARA counts, the content of
/// the installed bloom set (which the dependency skip-predicate consults), and
/// the installed model bundle — any of these changing the analysis lands cached
/// results in a fresh namespace.
///
/// The model belongs here for the same reason the rules do: a cached entry
/// holds the *verdict*, and the verdict is the model's output. Until
/// 2026-08-07 it was absent, so a model auto-update reused the previous
/// bundle's namespace and every already-seen file kept the verdict the old
/// bundle gave it. That is how the 2026-08-04 route-policy defect (benign
/// OpenDocument files graded hostile at every deploy level) would have
/// outlived the corrected bundle that fixed it.
pub(crate) fn ruleset_version() -> String {
    let vi = cleave::version_info();
    let commit = cleave::rule_update::installed(&cleave::traits_repo::install_target())
        .map_or_else(
            || "none".to_string(),
            |i| i.commit.chars().take(12).collect(),
        );
    let bloom = crate::bloom_repo::installed_manifest()
        .map_or_else(|| "nobloom".to_string(), |m| bloom_token(&m));
    format!(
        "{}-{commit}-t{}-c{}-y{}-{bloom}-m{}",
        env!("CARGO_PKG_VERSION"),
        vi.trait_count,
        vi.composite_count,
        vi.yara_rules,
        model_version(),
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
/// Prefers the updater's sidecar commit. A dev tree — the `~/azoth` symlink a
/// local `make azoth-deploy` writes — has no sidecar, and that is exactly the
/// setup where bundles change most often, so fall back to a stamp of
/// `config.json`'s size and mtime. Either way a redeployed bundle moves the
/// token. `nomodel` when nothing is installed.
fn model_version() -> String {
    if let Some(commit) = crate::models_repo::version() {
        return sanitize(&commit);
    }
    let config = crate::models_repo::install_target().join("config.json");
    let Ok(meta) = std::fs::metadata(&config) else {
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
const IDLE_NAMESPACE_AGE: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// Delete namespace directories under `base`, other than `current`, that no
/// process has written to in [`IDLE_NAMESPACE_AGE`]. Every rules, engine, or
/// bloom update starts a fresh namespace, and the superseded ones would
/// otherwise pile up. Not "everything but mine": a long-running server on the
/// previous ruleset still writes to its own namespace. Best-effort and silent,
/// like the cache itself.
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
        if idle && entry.file_name().to_str() != Some(current) {
            let _ = std::fs::remove_dir_all(entry.path());
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
        let later = SystemTime::now() + IDLE_NAMESPACE_AGE + Duration::from_secs(86_400);
        prune_idle(base.path(), "mine", later);
        assert!(base.path().join("mine").is_dir());
        assert!(!base.path().join("theirs").exists());
    }

    #[test]
    fn a_same_day_bloom_rebuild_moves_the_token() {
        let manifest = |sha: &str| -> burton::Manifest {
            toml::from_str(&format!(
                "schema = 1\nbuilt = \"2026-10-01\"\n[filter.purl-good]\nfile = \"purl-good.adbl\"\nsha256 = \"{sha}\"\nformat_version = 1\nn = 1\n"
            ))
            .unwrap()
        };
        let (a, b) = (manifest("aa"), manifest("bb"));
        assert_eq!(a.built, b.built);
        assert_ne!(bloom_token(&a), bloom_token(&b));
        assert_eq!(bloom_token(&a), bloom_token(&manifest("aa")));
    }

    #[test]
    fn put_then_get_round_trips_and_survives_a_pruned_namespace() {
        let base = tempfile::tempdir().unwrap();
        let cache = AnalysisCache {
            dir: base.path().join("ns"),
        };
        // The namespace was pruned by another process after this one opened it.
        cache.put("abc", &None, &[("k".to_string(), Vec::new())]);
        let hit = cache.get("abc").unwrap();
        assert!(hit.sub.is_none());
        assert_eq!(hit.next.len(), 1);
        assert!(cache.get("missing").is_none());
    }

    /// A report cut short by the rule deadline, an unfinished Rizin run or an
    /// interrupted source parse lacks results a later scan would have, so it
    /// must not be served to one.
    #[test]
    fn incomplete_analyses_are_not_stored() {
        use cleave::types::AnalysisGap::{
            DisassemblyIncomplete, EvaluationDeadline, SourceParseIncomplete,
        };
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

        cache.put("complete", &Some(report()), &[]);
        assert!(cache.get("complete").is_some());
        for gap in [
            EvaluationDeadline,
            DisassemblyIncomplete,
            SourceParseIncomplete,
        ] {
            let cut_short = report();
            cut_short.analysis_gaps.record(gap);
            cache.put(gap.label(), &Some(cut_short), &[]);
            assert!(cache.get(gap.label()).is_none(), "{gap:?}");
        }
    }

    /// The cache key holds verdicts, and verdicts are the model's output — a
    /// bundle swap must land them in a fresh namespace. Guards against the
    /// model token being dropped from the key again.
    #[test]
    fn ruleset_version_carries_the_model_token() {
        let version = ruleset_version();
        let token = model_version();
        assert!(
            version.ends_with(&format!("-m{token}")),
            "ruleset_version {version} must carry model token {token}",
        );
    }

    #[test]
    fn model_version_is_stable_and_path_safe() {
        assert_eq!(model_version(), model_version());
        assert!(
            model_version()
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')),
            "model token must be a safe path segment: {}",
            model_version(),
        );
    }
}
