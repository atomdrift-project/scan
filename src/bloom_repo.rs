//! Local-first resolution and loading of the known-good / known-bad bloom
//! filters, mirroring how [`crate::models_repo`] and [`crate::traits_repo`]
//! locate their data on disk.
//!
//! Filters live as `<kind>-<tier>.adbl` files (e.g. `purl-good.adbl`,
//! `sha256-bad.adbl`) in a directory resolved as:
//! 1. `SCAN_BLOOM_DIR` env var, if set;
//! 2. a `bloom/` directory in the working tree (dev convenience);
//! 3. `<data_dir>/atomdrift/scan/bloom` (the install target the updater fills).
//!
//! Loading is entirely fault-tolerant: a missing, truncated, or wrong-version
//! file simply yields no filter for that slot. No filter means no fast path —
//! the scan runs in full, which is always the safe answer.
//!
//! The one rule that is *not* lenient: a known-good **skip** is only offered
//! when the matching known-bad filter is also loaded. The bad filter is the
//! fast revocation channel; without it a stale bless cannot be vetoed, so the
//! skip is withheld (fail closed). See [`Decision`].

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use crate::bloom::{Filter, Kind, Tier};

mod decision_cache;
use decision_cache::DecisionCache;

/// Process-wide handle to the loaded filters, published by the scan config so
/// decoupled subsystems (the dependency-fetch reporter) can consult the same
/// verdicts without threading config through. Set once per run when bloom is
/// enabled; later sets are ignored.
static GLOBAL: OnceLock<Arc<Lookup>> = OnceLock::new();

/// Publish the loaded filters process-wide (called from [`crate::engine::ScanConfig::with_bloom`]).
pub fn set_global(lookup: Arc<Lookup>) {
    let _ = GLOBAL.set(lookup);
}

/// The process-wide filters, if bloom was enabled this run.
#[must_use]
pub fn global() -> Option<Arc<Lookup>> {
    GLOBAL.get().cloned()
}

/// What to do with a key before doing expensive work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Known-good and not vetoed by the bad channel — skip the full scan.
    Skip,
    /// Matched the known-bad set — surface the match and still scan it.
    KnownBad,
    /// In *both* the good and bad sets (e.g. filter version skew) — the verdict
    /// is contradictory, so trust neither: surface the conflict and scan.
    Conflicted,
    /// Not in either set, or skip withheld for safety — scan normally.
    Unknown,
}

impl Decision {
    /// Wire form for `GET /_/bloom`: lowercase, hyphenated.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Skip => "skip",
            Self::KnownBad => "known-bad",
            Self::Conflicted => "conflicted",
            Self::Unknown => "unknown",
        }
    }
}

/// The loaded filters for every (kind, tier), queried per scan.
///
/// Cheap to hold; load once at startup like the model. An absent slot is
/// `None` and contributes no fast path.
///
/// [`Self::memo_sha256`] / [`Self::memo_purl`] keep a 4096-entry LRU of recent
/// `/_/bloom` decisions. Scan-time [`Self::decide_sha256`] / [`Self::decide_purl`]
/// bypass the memo so a unique-file crawl cannot evict the lookup working set.
pub struct Lookup {
    purl_good: Option<Filter>,
    purl_bad: Option<Filter>,
    sha256_good: Option<Filter>,
    sha256_bad: Option<Filter>,
    sha256_memo: DecisionCache<[u8; 32]>,
    purl_memo: DecisionCache<String>,
}

impl Default for Lookup {
    fn default() -> Self {
        Self {
            purl_good: None,
            purl_bad: None,
            sha256_good: None,
            sha256_bad: None,
            sha256_memo: DecisionCache::new(decision_cache::CAP),
            purl_memo: DecisionCache::new(decision_cache::CAP),
        }
    }
}

impl fmt::Debug for Lookup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lookup")
            .field("purl_good", &self.purl_good)
            .field("purl_bad", &self.purl_bad)
            .field("sha256_good", &self.sha256_good)
            .field("sha256_bad", &self.sha256_bad)
            .finish_non_exhaustive()
    }
}

impl Lookup {
    /// Load filters from the resolved local directory (see module docs).
    #[must_use]
    pub fn load() -> Self {
        Self::load_from(&bloom_dir())
    }

    /// Load filters from a specific directory. Missing or invalid files are
    /// skipped; the result is always a valid (possibly empty) `Lookup`.
    #[must_use]
    pub fn load_from(dir: &Path) -> Self {
        let me = Self {
            purl_good: load_one(dir, Kind::Purl, Tier::Good),
            purl_bad: load_one(dir, Kind::Purl, Tier::Bad),
            sha256_good: load_one(dir, Kind::Sha256, Tier::Good),
            sha256_bad: load_one(dir, Kind::Sha256, Tier::Bad),
            ..Self::default()
        };
        tracing::debug!(
            dir = %dir.display(),
            purl_good = me.purl_good.is_some(),
            purl_bad = me.purl_bad.is_some(),
            sha256_good = me.sha256_good.is_some(),
            sha256_bad = me.sha256_bad.is_some(),
            "loaded bloom filters"
        );
        me
    }

    /// Total number of signatures across every loaded filter — the known-good
    /// and known-bad PURLs and SHA-256s currently resident in memory. Summed
    /// into the `ps` banner's rule tally.
    #[must_use]
    pub fn rule_count(&self) -> u64 {
        [
            &self.purl_good,
            &self.purl_bad,
            &self.sha256_good,
            &self.sha256_bad,
        ]
        .into_iter()
        .filter_map(Option::as_ref)
        .map(Filter::len)
        .sum()
    }

    /// True when at least one filter is loaded — lets a caller skip the lookup
    /// entirely when nothing is synced.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.purl_good.is_some()
            || self.purl_bad.is_some()
            || self.sha256_good.is_some()
            || self.sha256_bad.is_some()
    }

    /// Decide a package by its PURL, before fetching it. Keys use the identity
    /// form (see [`fletch::purl::identity`]), matching the producer: a string
    /// that can't normalize is [`Decision::Unknown`] — the producer never
    /// inserted such a key, so probing the filters with a degenerate string
    /// could only ever false-positive — and artifact-selection qualifiers
    /// (`arch`, `distro`, `kind`, …) are dropped before the probe.
    #[must_use]
    pub fn decide_purl(&self, purl: &str) -> Decision {
        let Some(key) = fletch::purl::identity(purl) else {
            return Decision::Unknown;
        };
        let bytes = key.as_bytes();
        resolve(
            self.purl_bad
                .as_ref()
                .is_some_and(|f| f.contains_key(bytes)),
            self.purl_bad.is_some(),
            self.purl_good
                .as_ref()
                .is_some_and(|f| f.contains_key(bytes)),
        )
    }

    /// Decide an artifact by its SHA-256 digest, after fetching but before the
    /// expensive analysis.
    #[must_use]
    pub fn decide_sha256(&self, digest: &[u8; 32]) -> Decision {
        resolve(
            self.sha256_bad
                .as_ref()
                .is_some_and(|f| f.contains_digest(digest)),
            self.sha256_bad.is_some(),
            self.sha256_good
                .as_ref()
                .is_some_and(|f| f.contains_digest(digest)),
        )
    }

    /// SHA-256 membership for `GET /_/bloom`: 4096-entry LRU.
    #[must_use]
    pub fn memo_sha256(&self, digest: &[u8; 32]) -> Decision {
        self.sha256_memo
            .get_or_insert(digest, || self.decide_sha256(digest))
    }

    /// PURL membership for `GET /_/bloom`: 4096-entry LRU.
    /// Keyed on the request string so a hit skips `identity()` as well as the
    /// filter probe.
    #[must_use]
    pub fn memo_purl(&self, purl: &str) -> Decision {
        self.purl_memo
            .get_or_insert(purl, || self.decide_purl(purl))
    }
}

/// The decision rule, shared by both kinds:
/// - in both sets → [`Decision::Conflicted`] (contradictory; trust neither, scan);
/// - bad only → [`Decision::KnownBad`] (flag and scan);
/// - good only, *and* the bad channel is loaded to veto it → [`Decision::Skip`];
/// - otherwise (incl. a good hit with no bad channel) → [`Decision::Unknown`].
const fn resolve(is_bad: bool, bad_loaded: bool, is_good: bool) -> Decision {
    if is_bad && is_good {
        Decision::Conflicted
    } else if is_bad {
        Decision::KnownBad
    } else if is_good && bad_loaded {
        Decision::Skip
    } else {
        Decision::Unknown
    }
}

/// Process-wide tally of bloom decisions, for end-of-scan observability. A scan
/// records into this as it goes; the summary reads a [`BloomCounts`] snapshot.
#[derive(Debug)]
struct BloomStats {
    skipped: AtomicU32,
    flagged: AtomicU32,
    conflicted: AtomicU32,
    unscanned: AtomicU32,
}

static STATS: BloomStats = BloomStats {
    skipped: AtomicU32::new(0),
    flagged: AtomicU32::new(0),
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
    /// Artifacts found in both good and bad (should never happen).
    pub conflicted: u32,
    /// Artifacts left unscanned in fast mode (in neither set).
    pub unscanned: u32,
}

impl BloomCounts {
    /// True when no bloom decisions were recorded (nothing to report).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.skipped == 0 && self.flagged == 0 && self.conflicted == 0 && self.unscanned == 0
    }
}

/// Record one decision against the process-wide tally. Called from the gate so
/// every entry point (file, package, process) is captured uniformly. `unscanned`
/// distinguishes a fast-mode unknown (left unscanned) from one that was analyzed.
pub(crate) fn record(decision: Decision, unscanned: bool) {
    let counter = match decision {
        Decision::Skip => &STATS.skipped,
        Decision::KnownBad => &STATS.flagged,
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
        conflicted: STATS.conflicted.load(Ordering::Relaxed),
        unscanned: STATS.unscanned.load(Ordering::Relaxed),
    }
}

/// The installed bloom manifest — its `built` date, schema, and per-filter
/// element counts/sha256 — read from the directory the loader resolves filters
/// from (see [`bloom_dir`]). `None` when no bloom set is installed or the
/// manifest is unreadable. Used by `scan version` to report the active filters.
#[must_use]
pub fn installed_manifest() -> Option<crate::bloom_build::Manifest> {
    let text = std::fs::read_to_string(bloom_dir().join("bloom.toml")).ok()?;
    toml::from_str(&text).ok()
}

/// Read and validate one `<kind>-<tier>.adbl` file. Returns `None` (with a log
/// line) for anything that is missing, unreadable, malformed, or whose header
/// does not match the expected kind/tier — never a guess.
fn load_one(dir: &Path, kind: Kind, tier: Tier) -> Option<Filter> {
    let path = dir.join(format!("{}-{}.adbl", kind.as_str(), tier.as_str()));
    let bytes = std::fs::read(&path).ok()?;
    match Filter::load(&bytes) {
        Ok(f) if f.kind() == kind && f.tier() == tier => Some(f),
        Ok(f) => {
            tracing::warn!(
                path = %path.display(),
                got_kind = f.kind().as_str(),
                got_tier = f.tier().as_str(),
                "bloom filter header does not match its file name; ignoring"
            );
            None
        }
        Err(e) => {
            tracing::debug!(path = %path.display(), error = %e, "no usable bloom filter");
            None
        }
    }
}

/// Resolve the directory the loader reads filters from (see module docs):
/// `SCAN_BLOOM_DIR`, then a `bloom/` dev tree, then the canonical install dir.
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

/// The directory the updater installs filters into: `SCAN_BLOOM_DIR` if set,
/// else the canonical data path. The loader's [`bloom_dir`] resolves to the same
/// place (its `bloom/` dev fallback aside), so installed filters are found.
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
    use crate::bloom::{Record, generate};

    /// Build the four filters from a small pool and write them to `dir` exactly
    /// as the producer would, so loading exercises the real on-disk path.
    fn publish(dir: &Path, good: Vec<Record>, bad: Vec<Record>) {
        for f in generate(good, bad, 1e-9) {
            let path = dir.join(format!("{}.adbl", f.artifact_stem()));
            std::fs::write(path, f.to_bytes()).expect("write filter");
        }
    }

    /// A distinct, deterministic digest per tag — enough to tell keys apart.
    fn sha(tag: u8) -> [u8; 32] {
        let mut d = [0u8; 32];
        d[0] = tag;
        d
    }

    #[test]
    fn loads_and_decides_round_trip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let good_sha = sha(1);
        let bad_sha = sha(2);
        publish(
            tmp.path(),
            vec![
                Record {
                    purl: Some("pkg:npm/good@1".into()),
                    sha256: Some(good_sha),
                },
                // also-bad: must be subtracted out of good by `generate`
                Record {
                    purl: Some("pkg:npm/evil@1".into()),
                    sha256: Some(bad_sha),
                },
            ],
            vec![Record {
                purl: Some("pkg:npm/evil@1".into()),
                sha256: Some(bad_sha),
            }],
        );

        let lk = Lookup::load_from(tmp.path());
        assert!(lk.is_active());

        // Known-good, not revoked → skip.
        assert_eq!(lk.decide_purl("pkg:npm/good@1"), Decision::Skip);
        assert_eq!(lk.decide_sha256(&good_sha), Decision::Skip);
        // Known-bad → scan and surface, for both key kinds.
        assert_eq!(lk.decide_purl("pkg:npm/evil@1"), Decision::KnownBad);
        assert_eq!(lk.decide_sha256(&bad_sha), Decision::KnownBad);
        // Never seen → scan normally.
        assert_eq!(lk.decide_purl("pkg:npm/unheard-of@9"), Decision::Unknown);
        assert_eq!(lk.decide_sha256(&sha(3)), Decision::Unknown);

        // The /_/bloom memo must agree with the uncached probe, including on
        // a second call (the LRU hit path).
        assert_eq!(lk.memo_purl("pkg:npm/good@1"), Decision::Skip);
        assert_eq!(lk.memo_purl("pkg:npm/good@1"), Decision::Skip);
        assert_eq!(lk.memo_sha256(&good_sha), Decision::Skip);
        assert_eq!(lk.memo_sha256(&good_sha), Decision::Skip);
        assert_eq!(lk.memo_purl("pkg:npm/evil@1"), Decision::KnownBad);
        assert_eq!(lk.memo_sha256(&bad_sha), Decision::KnownBad);
    }

    #[test]
    fn skip_withheld_without_bad_channel() {
        // Only the good filter is present; the veto channel is missing.
        let tmp = tempfile::tempdir().expect("tempdir");
        let good_sha = sha(1);
        for f in generate(
            vec![Record {
                purl: Some("pkg:npm/good@1".into()),
                sha256: Some(good_sha),
            }],
            Vec::new(),
            1e-9,
        ) {
            if f.tier() == Tier::Good {
                let path = tmp.path().join(format!("{}.adbl", f.artifact_stem()));
                std::fs::write(path, f.to_bytes()).expect("write");
            }
        }

        let lk = Lookup::load_from(tmp.path());
        // Good hit, but no bad filter to veto with → fail closed, scan it.
        assert_eq!(lk.decide_purl("pkg:npm/good@1"), Decision::Unknown);
        assert_eq!(lk.decide_sha256(&good_sha), Decision::Unknown);
    }

    #[test]
    fn empty_dir_is_inactive_and_scans_everything() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let lk = Lookup::load_from(tmp.path());
        assert!(!lk.is_active());
        assert_eq!(lk.decide_purl("pkg:npm/anything@1"), Decision::Unknown);
        assert_eq!(lk.decide_sha256(&[7u8; 32]), Decision::Unknown);
    }

    #[test]
    fn wrong_header_for_filename_is_ignored() {
        // A good filter written under the bad file name must be rejected, not
        // trusted by its name.
        let tmp = tempfile::tempdir().expect("tempdir");
        let good = generate(
            vec![Record {
                purl: Some("pkg:npm/x@1".into()),
                sha256: None,
            }],
            Vec::new(),
            1e-9,
        );
        let purl_good = good
            .iter()
            .find(|f| f.tier() == Tier::Good)
            .expect("good filter");
        std::fs::write(tmp.path().join("purl-bad.adbl"), purl_good.to_bytes()).expect("write");

        let lk = Lookup::load_from(tmp.path());
        // The mislabeled file is dropped, so there is no bad channel and no
        // good filter either → nothing active.
        assert!(!lk.is_active());
    }

    #[test]
    fn key_in_both_filters_is_conflicted() {
        use crate::bloom::{Builder, Kind, Tier};
        // `generate` subtracts bad from good, so a real bundle never overlaps.
        // Hand-build overlapping filters to simulate filter version skew (an old
        // good filter still holding a key the newer bad filter now revokes).
        let tmp = tempfile::tempdir().expect("tempdir");
        let k = sha(9);
        let mut good = Builder::sized_for(Kind::Sha256, Tier::Good, 1, 1e-9, 0);
        good.insert_digest(&k);
        let mut bad = Builder::sized_for(Kind::Sha256, Tier::Bad, 1, 1e-9, 0);
        bad.insert_digest(&k);
        std::fs::write(tmp.path().join("sha256-good.adbl"), good.build().to_bytes()).expect("w");
        std::fs::write(tmp.path().join("sha256-bad.adbl"), bad.build().to_bytes()).expect("w");

        let lk = Lookup::load_from(tmp.path());
        // In both sets → conflicted (not Skip, not KnownBad) → caller scans it.
        assert_eq!(lk.decide_sha256(&k), Decision::Conflicted);
    }
}
