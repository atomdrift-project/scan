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
//! (scan release, installed traits commit, trait/composite/YARA counts, and the
//! bloom set the skip-predicate consults). Any change to what the detector would
//! find lands in a different namespace, so a stale result can never mask a
//! detection a newer ruleset adds — a version bump simply misses and re-analyzes.
//! A hit is only ever a result the *current* detector already produced. Set
//! `SCAN_ANALYSIS_CACHE=0` to disable it entirely.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use cleave::AnalysisReport;
use fletch::Reference;
use serde::{Deserialize, Serialize};

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

/// Distinguishes concurrent temp files so two workers caching the same content
/// (or different content) never write the same scratch path.
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

impl AnalysisCache {
    /// Open (creating on first use) the cache directory for the active ruleset
    /// version. `None` when disabled via `SCAN_ANALYSIS_CACHE=0`, when the OS
    /// has no cache directory, or when the directory can't be created — every
    /// such case degrades to "always analyze", never an error.
    pub(crate) fn open() -> Option<Self> {
        if std::env::var("SCAN_ANALYSIS_CACHE").is_ok_and(|v| v == "0" || v == "false") {
            return None;
        }
        let dir = dirs::cache_dir()?
            .join("atomdrift")
            .join("scan")
            .join("analysis")
            .join(ruleset_version());
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
    /// the result is already in hand, the cache is only an optimization.
    pub(crate) fn put(
        &self,
        content_sha: &str,
        sub: &Option<AnalysisReport>,
        next: &[(String, Vec<Reference>)],
    ) {
        let Ok(json) = serde_json::to_vec(&StoreRef { sub, next }) else {
            return;
        };
        let Ok(compressed) = zstd::encode_all(&json[..], 3) else {
            return;
        };
        // Write to a unique temp path, then rename into place — a reader never
        // sees a half-written entry, and concurrent writers don't collide.
        let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = self.dir.join(format!("{content_sha}.{seq}.tmp"));
        if std::fs::write(&tmp, &compressed).is_ok()
            && std::fs::rename(&tmp, self.path(content_sha)).is_err()
        {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    fn path(&self, content_sha: &str) -> PathBuf {
        self.dir.join(format!("{content_sha}.zst"))
    }
}

/// A token identifying the analysis-producing detector, so a rules or engine
/// update invalidates cached results. Folds in the scan release, the installed
/// traits commit, cleave's trait/composite/YARA counts, and the installed bloom
/// set (which the dependency skip-predicate consults) — any of these changing
/// the analysis lands cached results in a fresh namespace.
fn ruleset_version() -> String {
    let vi = cleave::version_info();
    let commit = cleave::rule_update::installed(&cleave::traits_repo::install_target())
        .map_or_else(
            || "none".to_string(),
            |i| i.commit.chars().take(12).collect(),
        );
    let bloom = crate::bloom_repo::installed_manifest()
        .map_or_else(|| "nobloom".to_string(), |m| sanitize(&m.built));
    format!(
        "{}-{commit}-t{}-c{}-y{}-{bloom}",
        env!("CARGO_PKG_VERSION"),
        vi.trait_count,
        vi.composite_count,
        vi.yara_rules,
    )
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
