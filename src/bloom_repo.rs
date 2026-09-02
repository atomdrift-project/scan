//! Where scan keeps its [`burton`] bundle, and the process-wide state around it.
//!
//! The decision rule itself lives in `burton`. This module owns only the parts
//! that are scan's: where the bundle is installed, the handle decoupled
//! subsystems consult, the end-of-scan tally, and a small memo for the lookup
//! routes.
//!
//! Filters live as `<kind>-<tier>.adbl` files, with a `bloom.toml` manifest, in
//! a directory resolved as:
//! 1. `SCAN_BLOOM_DIR`, if set;
//! 2. a `bloom/` directory in the working tree (dev convenience);
//! 3. `<data_dir>/atomdrift/scan/bloom` (what the updater fills).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

mod decision_cache;
use decision_cache::DecisionCache;

pub use burton::{Artifact, Decision};

/// The PURL canonicalization scan's keys are in.
///
/// Recorded in every bundle scan publishes and required of every bundle it
/// opens. If this and [`purl_key`] ever drift apart from the producer's, the
/// bundle stops opening — which is the point. A silent total miss is worse.
pub const KEY_SCHEME: &str = "fletch/purl-identity/v1";

/// The lookup key for a PURL, or `None` when the string cannot normalize.
///
/// The single place scan turns a PURL into a bloom key. A string that will not
/// normalize contributes no key at build time, so probing with one could only
/// ever produce a false positive.
#[must_use]
pub fn purl_key(purl: &str) -> Option<String> {
    fletch::purl::identity(purl)
}

/// Process-wide handle to the loaded bundle, published by the scan config so
/// decoupled subsystems (the dependency-fetch reporter) can consult the same
/// verdicts without threading config through. Set once per run.
static GLOBAL: OnceLock<Arc<Lookup>> = OnceLock::new();

/// Publish the loaded bundle process-wide (from [`crate::engine::ScanConfig::with_bloom`]).
pub fn set_global(lookup: Arc<Lookup>) {
    let _ = GLOBAL.set(lookup);
}

/// The process-wide bundle, if bloom was enabled this run.
#[must_use]
pub fn global() -> Option<Arc<Lookup>> {
    GLOBAL.get().cloned()
}

/// Scan's view of a [`burton::Lookup`]: the bundle, plus a bounded memo for the
/// lookup routes.
///
/// [`Self::memo_sha256`] / [`Self::memo_purl`] keep a 4096-entry LRU of recent
/// decisions for `GET /sha256/{sha}` and `GET /purl`. Scan-time probes go
/// straight through, so a unique-file crawl cannot evict the lookup working set.
#[derive(Debug)]
pub struct Lookup {
    inner: burton::Lookup,
    sha256_memo: DecisionCache<[u8; 32]>,
    purl_memo: DecisionCache<String>,
}

impl Default for Lookup {
    fn default() -> Self {
        Self::wrap(burton::Lookup::empty())
    }
}

impl Lookup {
    fn wrap(inner: burton::Lookup) -> Self {
        Self {
            inner,
            sha256_memo: DecisionCache::new(decision_cache::CAP),
            purl_memo: DecisionCache::new(decision_cache::CAP),
        }
    }

    /// Open the installed bundle, or carry on without one.
    ///
    /// A bundle that will not open in full is not partly used: `burton` refuses
    /// it, and scan runs with no fast path, which is always the safe answer.
    /// That is worth a warning rather than a debug line — it means every file
    /// is about to be analyzed that need not have been.
    #[must_use]
    pub fn load() -> Self {
        Self::load_from(&bloom_dir())
    }

    /// Open a bundle from a specific directory.
    #[must_use]
    pub fn load_from(dir: &Path) -> Self {
        match burton::Lookup::open(dir, KEY_SCHEME) {
            Ok(inner) => {
                tracing::debug!(
                    dir = %dir.display(),
                    keys = inner.keys(),
                    "opened bloom bundle"
                );
                Self::wrap(inner)
            }
            Err(burton::OpenError::Manifest(_, e)) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::debug!(dir = %dir.display(), "no bloom bundle installed");
                Self::default()
            }
            Err(e) => {
                tracing::warn!(
                    dir = %dir.display(),
                    error = %e,
                    "bloom bundle unusable; scanning everything"
                );
                Self::default()
            }
        }
    }

    /// The verdict for one artifact. Supply every key you have: a blessing
    /// needs all of them to agree.
    pub fn decide(&self, artifact: &Artifact<'_>) -> Decision {
        self.inner.decide(artifact)
    }

    /// Whether this artifact may be skipped.
    #[must_use]
    pub fn may_skip(&self, artifact: &Artifact<'_>) -> bool {
        self.inner.may_skip(artifact)
    }

    /// The verdict for a raw PURL, canonicalized on the way in.
    ///
    /// A string that will not normalize is [`Decision::Unknown`].
    pub fn decide_purl(&self, purl: &str) -> Decision {
        purl_key(purl).map_or(Decision::Unknown, |key| {
            self.inner.decide(&Artifact::purl(&key))
        })
    }

    /// The verdict for an artifact digest.
    pub fn decide_sha256(&self, digest: &[u8; 32]) -> Decision {
        self.inner.decide(&Artifact::sha256(digest))
    }

    /// The verdict for an artifact named by a digest, a PURL, or both.
    ///
    /// Naming both is the strongest form of the question and the one that
    /// resists a ground digest: `burton` blesses only when every key given is
    /// blessed. A PURL that will not canonicalize contributes no key, exactly
    /// as it contributed none at build time.
    pub fn decide_any(&self, purl: Option<&str>, digest: Option<&[u8; 32]>) -> Decision {
        let key = purl.and_then(purl_key);
        let mut artifact = Artifact::default();
        if let Some(key) = key.as_deref() {
            artifact = artifact.and_purl(key);
        }
        if let Some(digest) = digest {
            artifact = artifact.and_sha256(digest);
        }
        self.inner.decide(&artifact)
    }

    /// The verdict for an artifact known by both keys.
    pub fn decide_both(&self, purl: &str, digest: &[u8; 32]) -> Decision {
        self.decide_any(Some(purl), Some(digest))
    }

    /// Memoized [`Self::decide_sha256`], for `GET /sha256/{sha}`.
    pub fn memo_sha256(&self, digest: &[u8; 32]) -> Decision {
        self.sha256_memo
            .get_or_insert(digest, || self.decide_sha256(digest))
    }

    /// Memoized [`Self::decide_purl`], for `GET /purl`. Keyed on the request
    /// string, so a hit skips canonicalization as well as the filter probe.
    pub fn memo_purl(&self, purl: &str) -> Decision {
        self.purl_memo
            .get_or_insert(purl, || self.decide_purl(purl))
    }

    /// True when a bundle is loaded.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    /// Total keys across every loaded filter. Summed into the `ps` banner's
    /// rule tally.
    #[must_use]
    pub fn rule_count(&self) -> u64 {
        self.inner.keys()
    }
}

/// Process-wide tally of bloom decisions, for end-of-scan observability.
#[derive(Debug)]
struct BloomStats {
    skipped: AtomicU32,
    flagged: AtomicU32,
    sighted: AtomicU32,
    conflicted: AtomicU32,
    unscanned: AtomicU32,
}

static STATS: BloomStats = BloomStats {
    skipped: AtomicU32::new(0),
    flagged: AtomicU32::new(0),
    sighted: AtomicU32::new(0),
    conflicted: AtomicU32::new(0),
    unscanned: AtomicU32::new(0),
};

/// A point-in-time snapshot of the bloom decision tally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BloomCounts {
    /// Known-good artifacts skipped without analysis.
    pub skipped: u32,
    /// Known-bad artifacts flagged (and still analyzed).
    pub flagged: u32,
    /// Artifacts matched only by an outside claim (and still analyzed).
    /// Counted apart from [`Self::flagged`] because the two answer different
    /// questions: how often our own catalogue fired, versus somebody else's.
    pub sighted: u32,
    /// Artifacts blessed and claimed at once (should never happen).
    pub conflicted: u32,
    /// Artifacts left unscanned in fast mode (in neither set).
    pub unscanned: u32,
}

impl BloomCounts {
    /// True when no bloom decisions were recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.skipped == 0
            && self.flagged == 0
            && self.sighted == 0
            && self.conflicted == 0
            && self.unscanned == 0
    }
}

/// Record one decision against the process-wide tally. Called from the gate so
/// every entry point (file, package, process) is captured uniformly.
/// `unscanned` distinguishes a fast-mode unknown, left unscanned, from one that
/// was analyzed.
pub(crate) fn record(decision: Decision, unscanned: bool) {
    let counter = match decision {
        Decision::Skip => &STATS.skipped,
        Decision::KnownBad => &STATS.flagged,
        Decision::SightedHostile | Decision::SightedSuspicious => &STATS.sighted,
        Decision::Conflicted => &STATS.conflicted,
        Decision::Unknown if unscanned => &STATS.unscanned,
        Decision::Unknown => return,
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Snapshot the bloom decision tally for reporting.
#[must_use]
pub fn counts() -> BloomCounts {
    BloomCounts {
        skipped: STATS.skipped.load(Ordering::Relaxed),
        flagged: STATS.flagged.load(Ordering::Relaxed),
        sighted: STATS.sighted.load(Ordering::Relaxed),
        conflicted: STATS.conflicted.load(Ordering::Relaxed),
        unscanned: STATS.unscanned.load(Ordering::Relaxed),
    }
}

/// The installed manifest — its `built` date, schema, and per-filter key
/// counts — read from the directory filters resolve from. `None` when no
/// bundle is installed. Used by `scan version`.
#[must_use]
pub fn installed_manifest() -> Option<burton::Manifest> {
    burton::build::read_manifest(&bloom_dir())
}

/// Resolve the directory the loader reads filters from (see module docs).
fn bloom_dir() -> PathBuf {
    if let Ok(explicit) = std::env::var("SCAN_BLOOM_DIR") {
        return PathBuf::from(explicit);
    }
    let local = PathBuf::from("bloom");
    if local.is_dir() {
        return local;
    }
    default_install_dir()
}

/// The directory the updater installs into: `SCAN_BLOOM_DIR` if set, else the
/// canonical data path. `bloom_dir` resolves to the same place, its `bloom/`
/// dev fallback aside, so installed filters are found.
#[must_use]
pub fn install_dir() -> PathBuf {
    std::env::var("SCAN_BLOOM_DIR").map_or_else(|_| default_install_dir(), PathBuf::from)
}

fn default_install_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("atomdrift")
        .join("scan")
        .join("bloom")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use burton::{KeySets, Record, Tier};

    fn digest(tag: u8) -> [u8; 32] {
        let mut d = [0u8; 32];
        d[0] = tag;
        d
    }

    fn publish(dir: &Path, rows: &[(Tier, Record)]) {
        let mut sets = KeySets::new();
        for (tier, r) in rows {
            sets.insert(*tier, r.clone());
        }
        burton::build::write_bundle(dir, &sets.into_filters(1e-9), "2026-08-31", KEY_SCHEME)
            .expect("write bundle");
    }

    fn rec(purl: Option<&str>, sha: Option<[u8; 32]>) -> Record {
        Record {
            purl: purl.and_then(purl_key),
            sha256: sha,
        }
    }

    #[test]
    fn no_bundle_means_scan_everything() {
        let tmp = tempfile::tempdir().unwrap();
        let lk = Lookup::load_from(tmp.path());
        assert!(!lk.is_active());
        assert_eq!(lk.rule_count(), 0);
        assert_eq!(lk.decide_sha256(&digest(1)), Decision::Unknown);
        assert_eq!(lk.decide_purl("pkg:npm/left-pad@1.3.0"), Decision::Unknown);
    }

    #[test]
    fn purls_are_canonicalized_on_the_way_in() {
        let tmp = tempfile::tempdir().unwrap();
        publish(
            tmp.path(),
            &[
                (Tier::Good, rec(Some("pkg:npm/left-pad@1.3.0"), None)),
                (Tier::Bad, rec(Some("pkg:npm/evil@6.6.6"), None)),
            ],
        );
        let lk = Lookup::load_from(tmp.path());

        assert_eq!(lk.decide_purl("pkg:npm/left-pad@1.3.0"), Decision::Skip);
        // Artifact-selection qualifiers are dropped by the identity form, so an
        // SBOM-stamped spelling collides with the bare coordinate.
        assert_eq!(
            lk.decide_purl("pkg:npm/left-pad@1.3.0?arch=x86_64"),
            Decision::Skip
        );
        assert_eq!(lk.decide_purl("pkg:npm/evil@6.6.6"), Decision::KnownBad);
        // A string that cannot normalize is never probed.
        assert_eq!(lk.decide_purl("not a purl"), Decision::Unknown);
    }

    #[test]
    fn a_bundle_from_another_key_scheme_is_not_opened() {
        let tmp = tempfile::tempdir().unwrap();
        let mut sets = KeySets::new();
        sets.insert(Tier::Good, rec(None, Some(digest(2))));
        burton::build::write_bundle(
            tmp.path(),
            &sets.into_filters(1e-9),
            "2026-08-31",
            "someone-elses-scheme/v1",
        )
        .unwrap();

        let lk = Lookup::load_from(tmp.path());
        assert!(!lk.is_active(), "a mismatched bundle must not be used");
        assert_eq!(lk.decide_sha256(&digest(2)), Decision::Unknown);
    }

    #[test]
    fn both_keys_must_agree_before_anything_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        publish(
            tmp.path(),
            &[
                (
                    Tier::Good,
                    rec(Some("pkg:npm/left-pad@1.3.0"), Some(digest(3))),
                ),
                (Tier::Bad, rec(None, Some(digest(4)))),
            ],
        );
        let lk = Lookup::load_from(tmp.path());

        assert_eq!(
            lk.decide_both("pkg:npm/left-pad@1.3.0", &digest(3)),
            Decision::Skip
        );
        // A blessed digest under a coordinate nobody blessed: the grinding case.
        assert_eq!(
            lk.decide_both("pkg:npm/attacker@1.0.0", &digest(3)),
            Decision::Unknown
        );
        // Keys that disagree outright.
        assert_eq!(
            lk.decide_both("pkg:npm/left-pad@1.3.0", &digest(4)),
            Decision::Conflicted
        );
    }

    #[test]
    fn the_memo_returns_what_the_direct_probe_would() {
        let tmp = tempfile::tempdir().unwrap();
        publish(
            tmp.path(),
            &[
                (Tier::Good, rec(Some("pkg:npm/ok@1.0.0"), Some(digest(5)))),
                (Tier::SightedHostile, rec(None, Some(digest(6)))),
            ],
        );
        let lk = Lookup::load_from(tmp.path());

        for _ in 0..2 {
            assert_eq!(lk.memo_purl("pkg:npm/ok@1.0.0"), Decision::Skip);
            assert_eq!(lk.memo_sha256(&digest(5)), Decision::Skip);
            assert_eq!(lk.memo_sha256(&digest(6)), Decision::SightedHostile);
            assert_eq!(lk.memo_sha256(&digest(7)), Decision::Unknown);
        }
    }
}
