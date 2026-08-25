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
    /// Byte offset of the match inside `file`, when the finding recorded one.
    /// Taken from the first evidence span cleave attached, or from the single
    /// contributing member of an inherited finding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub off: Option<u64>,
    /// 1-based source line of the match, when known. Only text-shaped matches
    /// carry one; a byte-oriented finding has `off` and no `line`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u64>,
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
            // Native matches only. A finding with `from` is the same match
            // reported again on an enclosing archive; the member's own copy
            // carries the real path and byte offset, and we walk every file,
            // so it is already in hand — or is a cross-file composite, which
            // has no single place to point at.
            if !finding.from.is_empty() {
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
            let (off, line) = locate(file, finding);
            hits.push(Hit {
                id: finding.id.clone(),
                crit: finding.criticality,
                file: file.path.clone(),
                pkg,
                desc: finding.description.clone(),
                off,
                line,
            });
        }
    }
    // Worst first, then by id — the same order beamline ranks its own hits in,
    // so one artifact reads the same whichever layer answered.
    hits.sort_by(|a, b| b.crit.cmp(&a.crit).then_with(|| a.id.cmp(&b.id)));
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

/// The finding id on a bloom-derived answer, mirroring hopper's `feedTraitID`.
///
/// Namespaced apart from the analyzer's taxonomy (`objectives/…`, `metadata/…`)
/// for the reason hopper gives: those ids say what an artifact DOES, and this
/// one says where a claim about it CAME FROM. A consumer resolving trait ids to
/// documentation would otherwise look this up and find nothing.
pub(crate) const FEED_TRAIT_ID: &str = "intel/feed/malicious";

/// The finding id when our own catalogue is the source rather than an outside
/// feed. Same namespace, different claim: the corpus convicted this artifact at
/// some earlier point, and the filter remembers only that it did.
pub(crate) const CORPUS_TRAIT_ID: &str = "intel/corpus/known-bad";

/// Synthesize the answer a bloom match justifies on its own, for an artifact no
/// stored verdict covers.
///
/// This is scan's mirror of hopper's `fromLedger`: rather than answering
/// "unknown" about a digest several operators call malware, answer with what
/// they say, marked as what it is. It lets the server answer without a round
/// trip to hopper, and without holding a verdict of its own.
///
/// # The levels
///
/// A filter is membership, not a measurement — the tier survives, the level does
/// not. So each tier reports the LOOSEST level its band can justify, because
/// claiming a tighter one would assert evidence we cannot see:
///
/// * `sighted-hostile` covers hopper's `Conclusive`/`Corroborated`/`Strong`
///   (floors 1/10/25). We cannot tell which, so we report 25 — enough to
///   convict at the default budget, and no more.
/// * `sighted-suspicious` covers `Moderate`/`Weak` (floors 50/100), reported as
///   100. Above the default budget on purpose: a lone unadjudicated citation
///   allows, and blocks only for a caller who has widened past it.
/// * `known-bad` is our own prior conviction, reported at the operating level.
///
/// Never 0, and never below the default budget, for hopper's reason: a derived
/// level should be able to convict without outranking measurement.
///
/// # The empty engine
///
/// `eng` is deliberately empty. Downstream branches on it to tell a measured
/// verdict from a citation — beamline's `customerView` omits an empty `eng`
/// entirely, and its cache treats an engine-less body as short-lived and refuses
/// to let it stand in for an analysis. Stamping a build here would make a filter
/// hit read as a scan we never ran, which is the one thing this must not do.
pub(crate) struct BloomClaim {
    /// The level this claim justifies on its own.
    pub lvl: i32,
    /// Criticality of the synthesized finding: 5 hostile, 4 suspicious.
    pub crit: u8,
    /// Which taxonomy made the claim.
    pub id: &'static str,
    /// One sentence a person reads.
    pub desc: &'static str,
}

/// What a bloom decision alone justifies saying, or `None` when it justifies
/// nothing.
///
/// `Skip` maps to a benign claim carrying no finding: the filters say we looked
/// at this and found nothing, which is an answer even though it names no
/// evidence. `Unknown` maps to `None` — no filter had an opinion, so there is
/// nothing to report and the caller's own policy decides.
///
/// One definition, read by both the native lookup route and the `/v1` decision,
/// so the two cannot drift into answering the same filter hit differently.
pub(crate) fn bloom_claim(decision: crate::bloom_repo::Decision) -> Option<Option<BloomClaim>> {
    use crate::bloom_repo::Decision;
    let default_level = i32::from(crate::model::DEFAULT_SEVERITY_LEVEL);
    Some(Some(match decision {
        // Nothing was claimed: benign, and no finding to name.
        Decision::Skip => return Some(None),
        // Conflicted is a bless standing beside a conviction. Worst wins, so it
        // answers as the conviction — but says so, because the disagreement is
        // the operator's problem to see.
        Decision::KnownBad | Decision::Conflicted => BloomClaim {
            lvl: default_level,
            crit: 5,
            id: CORPUS_TRAIT_ID,
            desc: "Catalogued as malicious by a previous analysis in our corpus.",
        },
        Decision::SightedHostile => BloomClaim {
            lvl: default_level,
            crit: 5,
            id: FEED_TRAIT_ID,
            desc: "Cited as malicious by corroborated threat intelligence.",
        },
        Decision::SightedSuspicious => BloomClaim {
            lvl: SIGHTED_SUSPICIOUS_LEVEL,
            crit: 4,
            id: FEED_TRAIT_ID,
            desc: "Cited as malicious by one unadjudicated threat intelligence source.",
        },
        // No filter had an opinion.
        Decision::Unknown => return None,
    }))
}

/// The native-route projection of [`bloom_claim`]. `hits` is scratch the caller
/// owns so the view can borrow from it.
pub(crate) fn bloom_derived_view<'a>(
    decision: crate::bloom_repo::Decision,
    bloom: &'a str,
    sha256: &'a str,
    purl: Option<&'a str>,
    hits: &'a mut Vec<Hit>,
) -> Option<View<'a>> {
    let claim = bloom_claim(decision)?;
    let lvl = claim.as_ref().map_or(BENIGN_LEVEL, |c| c.lvl);
    if let Some(c) = claim {
        hits.push(Hit {
            id: c.id.to_owned(),
            crit: c.crit,
            desc: c.desc.to_owned(),
            file: String::new(),
            pkg: purl.unwrap_or_default().to_owned(),
            off: None,
            line: None,
        });
    }
    Some(View {
        sha: sha256,
        purl,
        lvl: Some(lvl),
        eng: "",
        why: None,
        hits: &hits[..],
        bloom,
    })
}

/// The level meaning "fires at no budget at all" — a benign answer.
pub(crate) const BENIGN_LEVEL: i32 = -1;

/// The level a lone, unadjudicated outside citation justifies — hopper's
/// `Floor(Weak)`. Above the default budget deliberately, so it does not convict
/// by itself. Mirrors hopper's corroboration.go; keep the pair in sync.
const SIGHTED_SUSPICIOUS_LEVEL: i32 = 100;

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

/// Where a finding fired: `(byte offset, 1-based line)`.
///
/// The context windows carry a note per match, holding the exact offset. The
/// window's own `line` labels its first byte, so the match's line is that plus
/// the newlines between the window start and the match — the derivation the
/// format documents. Binary windows have no line structure and report only the
/// offset. A file whose context was trimmed falls back to the finding's first
/// evidence span, which locates it without naming a line.
fn locate(
    file: &cleave::types::CompactFile,
    finding: &cleave::types::CompactTrait,
) -> (Option<u64>, Option<u64>) {
    for window in &file.context {
        let Some(note) = window.notes.iter().find(|n| *n.id == *finding.id) else {
            continue;
        };
        let line = window.line.map(|start| {
            // A 32-bit target cannot hold a window longer than usize::MAX
            // anyway, so an offset that does not fit is past its end.
            let offset_in_window =
                usize::try_from(note.off.saturating_sub(window.loc)).unwrap_or(usize::MAX);
            let scanned = &window.data[..offset_in_window.min(window.data.len())];
            start + scanned.iter().filter(|&&b| b == b'\n').count() as u64
        });
        return (Some(note.off), line);
    }
    (finding.ev.first().map(|[off, _len]| *off), None)
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
    use crate::bloom_repo::Decision;

    /// The invariant everything downstream depends on: a derived answer carries
    /// no engine. beamline omits an empty `eng` entirely, and branches on its
    /// absence to keep a citation from standing in for an analysis — so if this
    /// ever stamps a build, a filter hit starts reading as a scan we never ran
    /// and `/v1/analyze` stops running.
    #[test]
    fn a_derived_view_never_claims_an_engine() {
        for d in [
            Decision::KnownBad,
            Decision::Conflicted,
            Decision::SightedHostile,
            Decision::SightedSuspicious,
        ] {
            let mut hits = Vec::new();
            let view = bloom_derived_view(d, d.as_str(), "abc", None, &mut hits)
                .expect("an adverse decision is answerable");
            assert_eq!(view.eng, "", "{d:?} must not claim an engine");
            assert_eq!(view.hits.len(), 1, "{d:?}");
        }
    }

    /// No filter had an opinion, so there is nothing to say — this must stay a
    /// miss rather than becoming a fabricated `allow`.
    #[test]
    fn an_unknown_decision_synthesizes_nothing() {
        let mut hits = Vec::new();
        assert!(
            bloom_derived_view(Decision::Unknown, "unknown", "abc", None, &mut hits).is_none()
        );
    }

    /// A bless answers benign and names no evidence: the filters say we looked
    /// and found nothing, which is an answer, but not a finding.
    #[test]
    fn a_bless_answers_benign_with_no_findings() {
        let mut hits = Vec::new();
        let view = bloom_derived_view(Decision::Skip, "skip", "abc", None, &mut hits)
            .expect("a bless is answerable");
        assert_eq!(view.lvl, Some(BENIGN_LEVEL));
        assert!(view.hits.is_empty(), "a bless names no evidence");
        assert_eq!(view.eng, "", "still not a measurement of ours");
    }

    /// Levels follow hopper's floors: the loosest its band can justify, never
    /// tighter than the evidence a filter can carry.
    #[test]
    fn derived_levels_are_the_loosest_of_their_band() {
        let default_level = i32::from(crate::model::DEFAULT_SEVERITY_LEVEL);
        let level = |d: Decision| {
            let mut hits = Vec::new();
            bloom_derived_view(d, d.as_str(), "abc", None, &mut hits)
                .and_then(|v| v.lvl)
                .expect("a level")
        };
        assert_eq!(level(Decision::SightedHostile), default_level);
        assert_eq!(level(Decision::KnownBad), default_level);
        assert_eq!(
            level(Decision::SightedSuspicious),
            SIGHTED_SUSPICIOUS_LEVEL,
            "a lone unadjudicated citation must not convict at the default budget"
        );
        assert!(
            SIGHTED_SUSPICIOUS_LEVEL > default_level,
            "suspicious must sit above the default budget, or it blocks by itself"
        );
    }

    /// Hostile cites the feed taxonomy, our own catalogue does not — the point
    /// of the split is that they are different claims.
    #[test]
    fn derived_findings_name_their_source() {
        let id = |d: Decision| {
            let mut hits = Vec::new();
            let v = bloom_derived_view(d, d.as_str(), "abc", None, &mut hits).expect("a view");
            v.hits[0].id.clone()
        };
        assert_eq!(id(Decision::SightedHostile), FEED_TRAIT_ID);
        assert_eq!(id(Decision::SightedSuspicious), FEED_TRAIT_ID);
        assert_eq!(id(Decision::KnownBad), CORPUS_TRAIT_ID);
    }

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
        let dir =
            std::env::temp_dir().join(format!("scan-lookup-test-{}-{seq}", std::process::id()));
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
            off: Some(2048),
            line: Some(42),
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

    /// A match reports the exact byte its note recorded, and the line that
    /// byte falls on — the window's own line advanced by the newlines before
    /// it. A window without line structure (a binary) reports only the offset.
    #[test]
    fn a_finding_reports_the_byte_and_line_it_fired_on() {
        use cleave::types::{ContextLine, Criticality, Istr, Note};

        let id = "objectives/execution/shell/bash";
        let finding = cleave::types::CompactTrait {
            id: id.to_string(),
            criticality: 5,
            description: String::new(),
            confidence: 1.0,
            mbc: None,
            attack: None,
            from: Vec::new(),
            ev: vec![[2048, 16]],
            dep: None,
        };
        let note = |off: u64| Note {
            crit: Criticality::Hostile,
            id: Istr::from(id),
            desc: Istr::from(""),
            off,
            len: 4,
            conf: 1.0,
        };
        let mut file = cleave::types::CompactFile {
            path: "lib/install.js".to_string(),
            findings: vec![finding.clone()],
            ..Default::default()
        };

        // Two lines of context before the match, so it sits on line 12.
        file.context = vec![ContextLine {
            loc: 100,
            line: Some(10),
            col: Some(1),
            data: b"one\ntwo\nspawn('bash')".to_vec(),
            notes: vec![note(109)],
        }];
        assert_eq!(locate(&file, &finding), (Some(109), Some(12)));

        // A binary window carries no line structure: the offset stands alone.
        file.context[0].line = None;
        assert_eq!(locate(&file, &finding), (Some(109), None));

        // Context trimmed away: fall back to the finding's own evidence span,
        // which locates it without claiming a line.
        file.context.clear();
        assert_eq!(locate(&file, &finding), (Some(2048), None));
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
