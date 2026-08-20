//! Verdict index: what we already know about a sha256 or a PURL, in the compact
//! form the lookup routes answer with.
//!
//! The expensive artifact of an analysis is the cleave report, and that is
//! already cached twice over: cleave memoizes the top-level report by content
//! sha (`toplevel_report_cache`), and [`crate::analysis_cache`] memoizes the
//! grafted sub-report of every fetched dependency. Neither can answer "what is
//! the verdict for this sha" on its own — cleave's key folds in the file type
//! and an options hash a caller holding only a digest cannot reconstruct, and
//! rebuilding `lvl` from a report costs feature extraction plus model
//! inference, which a lookup must not wait on.
//!
//! So this stores the *answer*, not the analysis: a kilobyte of verdict per
//! artifact, written through as each analysis finishes and served without the
//! model, cleave, or an analyze slot. A miss is a normal answer — the routes
//! report "unknown sample" and the caller asks for a real analysis.
//!
//! Namespaced by the same ruleset-version token the analysis cache uses, so a
//! rules, model, or engine change lands in a fresh namespace rather than
//! serving a verdict the current detector would no longer give. Disabled by
//! `SCAN_ANALYSIS_CACHE=0` (or the `SCAN_NO_ANALYSIS_CACHE=1` umbrella), which
//! degrades every lookup to "unknown" rather than to an error.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

/// Findings kept per verdict. The wire view serves the worst few; the rest are
/// headroom so the projection can widen without re-analyzing everything.
const MAX_STORED_HITS: usize = 10;

/// Criticality floor for a stored finding: 3 notable, 4 suspicious, 5 hostile.
/// Anything below is baseline noise that no consumer gates on.
const MIN_HIT_CRIT: u8 = 3;

/// Verdicts memoized in process, ahead of the on-disk read.
const MEMO_CAPACITY: NonZeroUsize = match NonZeroUsize::new(1024) {
    Some(n) => n,
    None => NonZeroUsize::MIN,
};

/// Distinguishes concurrent temp files so two writers never collide.
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// One finding, flattened to what a consumer gates on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Hit {
    /// Stable trait identifier (`objectives/execution/shell/bash`).
    pub id: String,
    /// Criticality ordinal: 3 notable, 4 suspicious, 5 hostile.
    pub crit: u8,
    /// Path inside the artifact the finding fired on.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub file: String,
    /// The component it is about — a dependency's locator when the finding
    /// came from one, else the artifact's own PURL.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub pkg: String,
    /// One-line description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub desc: String,
}

/// The stored answer for one artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Verdict {
    /// Content digest of the artifact this verdict is about.
    pub sha256: String,
    /// `ml.lvl`: the tightest false-positive budget per 100M benigns at which
    /// this artifact grades hostile. `-1` never fires; `None` is
    /// manual-threshold mode. Callers gate on this.
    pub lvl: Option<i32>,
    /// Scan build that produced the verdict.
    pub eng: String,
    /// RFC 3339 timestamp of the analysis.
    pub at: String,
    /// The PURL this artifact was fetched as, when it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purl: Option<String>,
    /// One-sentence rationale from the interpreter, when `--interpret` ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
    /// Worst findings, most critical first, at most [`MAX_STORED_HITS`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hits: Vec<Hit>,
}

impl Verdict {
    /// Project a finished scan into the stored answer. `purl` is the locator
    /// the artifact was requested by, when the request named one.
    pub(crate) fn from_scan(result: &crate::engine::ScanResult, purl: Option<&str>) -> Self {
        let why = result
            .interpretation
            .as_ref()
            .map(|llm| llm.interpretation.trim())
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        Self {
            sha256: result.sha256.clone(),
            lvl: result.level,
            eng: crate::engine::ENGINE_VERSION.to_string(),
            at: result.analyzed_at.clone(),
            purl: purl.map(str::to_owned),
            why,
            hits: result
                .cleave
                .as_ref()
                .map(|report| collect_hits(report, purl))
                .unwrap_or_default(),
        }
    }
}

/// The worst findings across every file in the report, most critical first.
///
/// Mirrors what a consumer would pick out of `raw` itself, so a served verdict
/// and a freshly rendered envelope agree on which findings matter.
fn collect_hits(report: &cleave::types::CompactReport, purl: Option<&str>) -> Vec<Hit> {
    let mut hits: Vec<Hit> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for file in &report.files {
        for finding in &file.findings {
            if finding.criticality < MIN_HIT_CRIT {
                continue;
            }
            let pkg = finding
                .dep
                .as_ref()
                .map(|d| d.locator.clone())
                .or_else(|| purl.map(str::to_owned))
                .unwrap_or_default();
            // One row per (trait, file, component): the same trait firing on a
            // dozen members of one archive is one fact, not a dozen.
            if !seen.insert((finding.id.clone(), file.path.clone(), pkg.clone())) {
                continue;
            }
            hits.push(Hit {
                id: finding.id.clone(),
                crit: finding.criticality,
                file: file.path.clone(),
                pkg,
                desc: finding.description.clone(),
            });
        }
    }
    // Stable within a criticality so repeated analyses of the same bytes store
    // the same rows in the same order.
    hits.sort_by_key(|h| std::cmp::Reverse(h.crit));
    hits.truncate(MAX_STORED_HITS);
    hits
}

/// Findings served on a lookup. The store keeps more (see
/// [`MAX_STORED_HITS`]); a consumer acts on the worst few.
pub(crate) const SERVED_HITS: usize = 3;

/// The wire form of a lookup answer: the verdict plus the filter's opinion of
/// the same key. `bloom` is present whether or not we hold a verdict — it is a
/// separate, cheaper kind of knowledge, and collapsing the two would let a
/// probabilistic filter hit read as an analysis we never ran.
#[derive(Debug, Serialize)]
pub(crate) struct View<'a> {
    pub sha: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub purl: Option<&'a str>,
    pub lvl: Option<i32>,
    pub eng: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<&'a str>,
    #[serde(skip_serializing_if = "<[Hit]>::is_empty")]
    pub hits: &'a [Hit],
    pub bloom: &'a str,
}

impl Verdict {
    /// Project the stored verdict onto the wire, worst findings first.
    /// `requested_purl` is what the caller asked by, which names the artifact
    /// more precisely than a locator recorded on some earlier analysis.
    pub(crate) fn view<'a>(&'a self, bloom: &'a str, requested_purl: Option<&'a str>) -> View<'a> {
        View {
            sha: &self.sha256,
            purl: requested_purl.or(self.purl.as_deref()),
            lvl: self.lvl,
            eng: &self.eng,
            why: self.why.as_deref(),
            hits: &self.hits[..self.hits.len().min(SERVED_HITS)],
            bloom,
        }
    }
}

/// Handle to the index directory for the current ruleset version.
pub(crate) struct Index {
    dir: PathBuf,
    /// Recently served verdicts, keyed by content sha. A PURL hit resolves to a
    /// sha first, so both key kinds share one memo.
    memo: Mutex<lru::LruCache<String, Option<Verdict>>>,
}

/// Root of the verdict index (`…/atomdrift/scan/lookup`), above the
/// per-ruleset-version subdirectory. Entries live at `lookup/<version>/…`, two
/// levels below this root — the shape [`crate::cache_cleanup`] reclaims.
pub(crate) fn index_base() -> Option<PathBuf> {
    Some(
        dirs::cache_dir()?
            .join("atomdrift")
            .join("scan")
            .join("lookup"),
    )
}

/// The process-wide index, opened once. `None` when disabled or unusable, in
/// which case every lookup answers "unknown".
pub(crate) fn global() -> Option<&'static Index> {
    static GLOBAL: OnceLock<Option<Index>> = OnceLock::new();
    GLOBAL.get_or_init(Index::open).as_ref()
}

impl Index {
    /// Open (creating on first use) the index for the active ruleset version.
    /// Every failure degrades to `None` — "we know nothing" — never an error.
    fn open() -> Option<Self> {
        if std::env::var("SCAN_ANALYSIS_CACHE").is_ok_and(|v| v == "0" || v == "false") {
            return None;
        }
        let base = index_base()?;
        let version = crate::analysis_cache::ruleset_version();
        crate::analysis_cache::prune_stale_versions(&base, &version);
        let dir = base.join(version);
        std::fs::create_dir_all(&dir).ok()?;
        Some(Self {
            dir,
            memo: Mutex::new(lru::LruCache::new(MEMO_CAPACITY)),
        })
    }

    /// The verdict for a content digest, or `None` when we hold none.
    pub(crate) fn get_sha(&self, sha256: &str) -> Option<Verdict> {
        let sha = normalize_sha(sha256)?;
        if let Ok(mut memo) = self.memo.lock()
            && let Some(hit) = memo.get(&sha)
        {
            return hit.clone();
        }
        let found = std::fs::read(self.verdict_path(&sha))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Verdict>(&bytes).ok());
        if let Ok(mut memo) = self.memo.lock() {
            memo.put(sha, found.clone());
        }
        found
    }

    /// The verdict for a PURL, resolved through the alias index. `None` when
    /// the PURL is unparseable, unseen, or its artifact has since been pruned.
    pub(crate) fn get_purl(&self, purl: &str) -> Option<Verdict> {
        let key = purl_key(purl)?;
        let sha = std::fs::read_to_string(self.alias_path(&key)).ok()?;
        // An alias outliving its verdict (evicted by the cache sweeper) reads
        // as a miss, which is the honest answer.
        self.get_sha(sha.trim())
    }

    /// Store a verdict, plus a PURL alias when the artifact was fetched as one.
    ///
    /// First write wins: the bytes are the identity, so a second analysis of
    /// them within one ruleset namespace has nothing new to say. Best-effort —
    /// the caller already holds the answer, so any failure here is silent.
    pub(crate) fn put(&self, verdict: &Verdict) {
        let Some(sha) = normalize_sha(&verdict.sha256) else {
            return;
        };
        if self.get_sha(&sha).is_some() {
            return;
        }
        let Ok(json) = serde_json::to_vec(verdict) else {
            return;
        };
        if !self.write_atomic(&self.verdict_path(&sha), &json) {
            return;
        }
        if let Ok(mut memo) = self.memo.lock() {
            memo.put(sha.clone(), Some(verdict.clone()));
        }
        // The alias is written after the verdict it points at, so a PURL never
        // resolves to a sha whose record is not yet readable.
        if let Some(key) = verdict.purl.as_deref().and_then(purl_key) {
            self.write_atomic(&self.alias_path(&key), sha.as_bytes());
        }
    }

    fn verdict_path(&self, sha256: &str) -> PathBuf {
        self.dir.join(format!("{sha256}.json"))
    }

    /// PURL aliases sit beside the verdicts rather than in a subdirectory, so
    /// every entry in the namespace is one file at one depth — the shape the
    /// cache sweeper reclaims (see [`crate::cache_cleanup`]).
    fn alias_path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.purl"))
    }

    /// Write via a unique temp path and rename, so a reader never sees a
    /// half-written entry and concurrent writers do not collide.
    fn write_atomic(&self, path: &std::path::Path, bytes: &[u8]) -> bool {
        let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = self.dir.join(format!("w{seq}.tmp"));
        if std::fs::write(&tmp, bytes).is_err() {
            return false;
        }
        if std::fs::rename(&tmp, path).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return false;
        }
        true
    }
}

/// Lowercased digest, or `None` when the input is not 64 hex characters.
fn normalize_sha(raw: &str) -> Option<String> {
    let sha = raw.trim().to_ascii_lowercase();
    (sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit())).then_some(sha)
}

/// Filename for a PURL's alias: the digest of its identity form, so scoped
/// names, qualifiers, and case folding cannot produce a path component.
/// Keyed through `fletch::purl::identity` — the same canonicalization the
/// bloom filters use, so a PURL that hits one hits the other.
fn purl_key(purl: &str) -> Option<String> {
    use sha2::{Digest, Sha256};
    let identity = fletch::purl::identity(purl.trim())?;
    Some(format!("{:x}", Sha256::digest(identity.as_bytes())))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn verdict(sha: &str) -> Verdict {
        Verdict {
            sha256: sha.to_string(),
            lvl: Some(-1),
            eng: "test".to_string(),
            at: "2026-08-20T00:00:00Z".to_string(),
            purl: None,
            why: None,
            hits: Vec::new(),
        }
    }

    fn temp_index() -> Index {
        let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("scan-lookup-test-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create index dir");
        Index {
            dir,
            memo: Mutex::new(lru::LruCache::new(MEMO_CAPACITY)),
        }
    }

    #[test]
    fn round_trips_by_sha() {
        let idx = temp_index();
        let sha = "a".repeat(64);
        assert!(idx.get_sha(&sha).is_none(), "empty index is a miss");
        idx.put(&verdict(&sha));
        assert_eq!(idx.get_sha(&sha).expect("stored").lvl, Some(-1));
        // Uppercase and surrounding whitespace name the same artifact.
        assert!(idx.get_sha(&format!(" {} ", sha.to_uppercase())).is_some());
        std::fs::remove_dir_all(&idx.dir).ok();
    }

    #[test]
    fn resolves_purl_through_alias() {
        let idx = temp_index();
        let sha = "b".repeat(64);
        let mut v = verdict(&sha);
        v.purl = Some("pkg:npm/left-pad@1.3.0".to_string());
        idx.put(&v);
        let hit = idx.get_purl("pkg:npm/left-pad@1.3.0").expect("purl hit");
        assert_eq!(hit.sha256, sha);
        // Identity folding is the bloom filters' own, so the `pkg:` scheme and
        // case variations resolve to the same artifact.
        assert!(idx.get_purl("PKG:NPM/left-pad@1.3.0").is_some());
        assert!(idx.get_purl("pkg:npm/right-pad@1.3.0").is_none());
        std::fs::remove_dir_all(&idx.dir).ok();
    }

    #[test]
    fn the_first_verdict_for_an_artifact_stands() {
        let idx = temp_index();
        let sha = "c".repeat(64);
        let mut rich = verdict(&sha);
        rich.why = Some("Postinstall launches a reverse shell.".to_string());
        rich.purl = Some("pkg:npm/evil@1.0.0".to_string());
        rich.hits = vec![Hit {
            id: "objectives/execution/shell/bash".to_string(),
            crit: 5,
            file: "lib/install.js".to_string(),
            pkg: "pkg:npm/evil@1.0.0".to_string(),
            desc: "Spawns bash from a npm postinstall hook".to_string(),
        }];
        idx.put(&rich);

        // The same bytes seen again, this time with no interpretation and no
        // findings — a later write never displaces the stored answer.
        idx.put(&verdict(&sha));

        let kept = idx.get_sha(&sha).expect("stored");
        assert_eq!(kept.hits.len(), 1, "findings are not overwritten");
        assert!(kept.why.is_some(), "interpretation is not overwritten");
        assert_eq!(kept.purl.as_deref(), Some("pkg:npm/evil@1.0.0"));
        std::fs::remove_dir_all(&idx.dir).ok();
    }

    #[test]
    fn rejects_keys_that_are_not_digests() {
        let idx = temp_index();
        assert!(idx.get_sha("not-a-sha").is_none());
        assert!(idx.get_sha(&"g".repeat(64)).is_none(), "not hex");
        // A verdict with a malformed digest is dropped rather than written to
        // an attacker-chosen path.
        idx.put(&verdict("../../etc/passwd"));
        assert!(!idx.dir.join("../../etc/passwd.json").exists());
        std::fs::remove_dir_all(&idx.dir).ok();
    }
}
