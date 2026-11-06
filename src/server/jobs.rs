//! Completion counters: what this server has done and how long it took, by
//! size and PURL type, as `/_/stats` publishes them for a router.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{diag, latency};

/// Upper bounds, in bytes, of each size bucket; the last is open-ended.
/// Chosen around where behaviour actually changes: a source tarball, a typical
/// package, a large archive, and the multi-hundred-megabyte inputs whose member
/// expansion dominates everything else.
pub(super) const SIZE_BUCKETS: [u64; 4] = [1 << 20, 16 << 20, 128 << 20, u64::MAX];

/// Human labels for [`SIZE_BUCKETS`], used as JSON keys on `/_/stats`.
pub(super) const SIZE_BUCKET_NAMES: [&str; 4] = ["le_1mb", "le_16mb", "le_128mb", "gt_128mb"];

/// Completion totals for one class of work.
///
/// Keeps two views because they answer different questions. The running totals
/// are cumulative-with-aging and say what this server has done; the windowed
/// [`latency::Latency`] says what it is doing *now*, and is what a router
/// reads. See that module for why a percentile over a time window beats a mean
/// over a sample count.
#[derive(Debug, Default)]
pub(super) struct JobBucket {
    pub(super) count: AtomicU64,
    pub(super) micros: AtomicU64,
    pub(super) recent: latency::Latency,
}

/// How many completions the cumulative totals remember before they start
/// forgetting. The windowed view has its own, time-based expiry.
const JOB_BUCKET_MEMORY: u64 = 256;

impl JobBucket {
    /// Record one completion in both views.
    pub(super) fn record(&self, micros: u64) {
        self.recent.record(micros);
        let n = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        self.micros.fetch_add(micros, Ordering::Relaxed);
        if n >= JOB_BUCKET_MEMORY {
            // Racy by construction: two threads crossing the line together may
            // both halve. That costs a little extra forgetting and nothing else,
            // which is a better trade here than a lock on the hot path.
            self.count.fetch_sub(n / 2, Ordering::Relaxed);
            let m = self.micros.load(Ordering::Relaxed);
            self.micros.fetch_sub(m / 2, Ordering::Relaxed);
        }
    }

    /// Samples in the aged totals, and their mean in milliseconds.
    pub(super) fn mean_ms(&self) -> (u64, Option<u64>) {
        let n = self.count.load(Ordering::Relaxed);
        let ms = (n > 0).then(|| self.micros.load(Ordering::Relaxed) / n / 1_000);
        (n, ms)
    }

    /// The windowed view, in milliseconds, for `/_/stats`.
    pub(super) fn recent(&self) -> diag::Recent {
        let s = self.recent.summary();
        diag::Recent {
            samples: s.samples,
            p80_ms: s.p80_micros.map(|us| us / 1_000),
            mean_ms: s.mean_micros.map(|us| us / 1_000),
        }
    }
}

/// PURL types tracked separately on `/_/stats`, plus a catch-all.
///
/// A PURL carries no size, so a router choosing a server for one has nothing to
/// look up in [`SIZE_BUCKET_NAMES`] until the artifact has already been fetched
/// — by which point the choice is made. The type is the next best predictor and
/// is known up front: a golang pseudo-version resolves to a repository clone, an
/// npm package to a small tarball, and the two are not comparable work.
pub(super) const PURL_TYPE_NAMES: [&str; 5] = ["cargo", "golang", "npm", "pypi", "other"];

/// The bucket index for `purl`, matching on the type between `pkg:` and `/`.
pub(super) fn purl_type_bucket(purl: &str) -> usize {
    let rest = purl.strip_prefix("pkg:").unwrap_or(purl);
    let ty = rest.split('/').next().unwrap_or("");
    // Only the type is case-insensitive per the PURL spec; the rest is not
    // touched here because nothing downstream of this counter reads it.
    PURL_TYPE_NAMES
        .iter()
        .position(|n| ty.eq_ignore_ascii_case(n))
        .unwrap_or(PURL_TYPE_NAMES.len() - 1)
}

/// The bucket index for an artifact of `size` bytes.
pub(super) fn size_bucket(size: u64) -> usize {
    SIZE_BUCKETS
        .iter()
        .position(|&bound| size <= bound)
        .unwrap_or(SIZE_BUCKETS.len() - 1)
}

/// Analyses this server has begun and completed, and the figures a router
/// reads to choose it.
///
/// Counted rather than sampled: a router wants "how big and how slow are this
/// server's jobs, typically", and totals divided at read time answer that
/// without keeping a window. `started` minus `completed` is also the honest
/// count of work that went in and never came out.
#[derive(Debug, Default)]
pub(super) struct Jobs {
    pub(super) started: AtomicU64,
    pub(super) completed: AtomicU64,
    pub(super) bytes_total: AtomicU64,
    pub(super) micros_total: AtomicU64,
    /// Per-size-bucket completion totals, for the size-aware half of routing.
    ///
    /// One scalar average is not enough to choose a server. A wide spread
    /// between two scanners on the same artifact is typically a large
    /// archive's member analysis, not a constant handicap — a single number
    /// would brand a box "slow" when it is only slow at big inputs, and send
    /// every small package somewhere worse. A caller usually knows the size
    /// before it dispatches, so the useful answer is per bucket.
    pub(super) by_size: [JobBucket; SIZE_BUCKETS.len()],
    pub(super) by_type: [JobBucket; PURL_TYPE_NAMES.len()],
    /// The blended average, aged like the others. Separate from `completed`,
    /// which stays a true lifetime count for reporting: one answers "how fast
    /// is this server now", the other "how much has it done".
    ///
    /// Fresh analyses only — see `cached`. So are `by_size` and `by_type`.
    pub(super) overall: JobBucket,
    /// Analyses answered from cleave's analysis cache.
    ///
    /// Kept apart from the fresh numbers because mixing them makes every
    /// average bimodal and therefore useless for prediction: the same artifact
    /// is milliseconds on a hit and minutes on a miss. A router choosing a
    /// server for work it has not done wants the fresh figure; blending in
    /// cache hits only tells it how lucky this server has been.
    pub(super) cached: JobBucket,
    /// `/lookup` service time. Near-constant — an index probe, not an analysis
    /// — and so the honest input for ordering the cheap-source race, where the
    /// analysis averages would be wrong by three orders of magnitude.
    pub(super) lookups: JobBucket,
}

impl Jobs {
    /// Count one finished analysis. Every analyze route reports here, so
    /// `started` minus `completed` stays the work that never came out.
    pub(super) fn finished(
        &self,
        result: &crate::engine::ScanResult,
        elapsed_ms: u64,
        purl: Option<&str>,
    ) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.bytes_total
            .fetch_add(result.size_bytes, Ordering::Relaxed);
        self.micros_total
            .fetch_add(elapsed_ms.saturating_mul(1_000), Ordering::Relaxed);
        // What the router averages is this server's own service time: the LLM
        // phase is left out, because the endpoint is shared by every server
        // and a contended one makes each that asks it look slow. A freshly
        // restarted server whose first samples all waited on the endpoint
        // would otherwise read as the slowest of them and be starved of work.
        let micros = elapsed_ms
            .saturating_sub(result.interpret_ms)
            .saturating_mul(1_000);
        // Routing predicts the cost of work this server has *not* done, so
        // only fresh analyses feed the figures a router reads. A cache hit is
        // real and worth reporting, but it predicts nothing about the next
        // unseen artifact.
        if result.analysis_cached {
            self.cached.record(micros);
            return;
        }
        self.overall.record(micros);
        self.by_size[size_bucket(result.size_bytes)].record(micros);
        // By PURL type too, when the job was named by one: the only cost signal
        // a router has before dispatch for `?purl=` work.
        if let Some(purl) = purl {
            self.by_type[purl_type_bucket(purl)].record(micros);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every bucket has a label, or `/_/stats` would silently drop one.
    #[test]
    fn every_bucket_is_named() {
        assert_eq!(SIZE_BUCKETS.len(), SIZE_BUCKET_NAMES.len());
    }

    /// The boundaries are inclusive upper bounds and the last is open-ended, so
    /// no size — including zero and u64::MAX — can fall outside.
    #[test]
    fn every_size_lands_in_a_bucket() {
        for size in [0, 1, 1 << 20, (1 << 20) + 1, 16 << 20, 128 << 20, u64::MAX] {
            let i = size_bucket(size);
            assert!(
                i < SIZE_BUCKETS.len(),
                "size {size} fell outside the buckets"
            );
        }
    }

    /// Boundaries are inclusive: an artifact of exactly 1 MiB is a small one,
    /// not the first of the next class up.
    #[test]
    fn boundaries_are_inclusive_and_ordered() {
        assert_eq!(size_bucket(0), 0);
        assert_eq!(size_bucket(1 << 20), 0);
        assert_eq!(size_bucket((1 << 20) + 1), 1);
        assert_eq!(size_bucket(16 << 20), 1);
        assert_eq!(size_bucket((16 << 20) + 1), 2);
        assert_eq!(size_bucket(128 << 20), 2);
        assert_eq!(size_bucket((128 << 20) + 1), 3);
        assert_eq!(size_bucket(u64::MAX), 3);
        // Monotonic: a bigger artifact never lands in an earlier bucket.
        let mut last = 0;
        for size in [0_u64, 1 << 10, 1 << 20, 1 << 24, 1 << 27, 1 << 30, u64::MAX] {
            let i = size_bucket(size);
            assert!(i >= last, "bucket went backwards at {size}");
            last = i;
        }
    }

    #[test]
    fn known_types_get_their_own_bucket() {
        for (i, name) in PURL_TYPE_NAMES.iter().enumerate().take(4) {
            assert_eq!(purl_type_bucket(&format!("pkg:{name}/thing@1.0")), i);
        }
    }

    #[test]
    fn the_type_is_case_insensitive_and_pkg_is_optional() {
        assert_eq!(purl_type_bucket("pkg:PyPI/requests@2.0"), 3);
        assert_eq!(purl_type_bucket("npm/left-pad@1.0"), 2);
    }

    #[test]
    fn unknown_and_malformed_fall_into_other() {
        let other = PURL_TYPE_NAMES.len() - 1;
        assert_eq!(purl_type_bucket("pkg:maven/g/a@1"), other);
        assert_eq!(purl_type_bucket(""), other);
        // "other" is a bucket name, not a type: a PURL literally spelled that
        // way must not be mistaken for a real match on it.
        assert_eq!(purl_type_bucket("pkg:other/x@1"), other);
    }

    #[test]
    fn a_golang_module_path_keeps_its_slashes_out_of_the_type() {
        assert_eq!(
            purl_type_bucket("pkg:golang/github.com/spf13/cobra@v1.10.2"),
            1
        );
    }

    fn mean(b: &JobBucket) -> u64 {
        let n = b.count.load(std::sync::atomic::Ordering::Relaxed);
        b.micros.load(std::sync::atomic::Ordering::Relaxed) / n.max(1)
    }

    #[test]
    fn aging_preserves_the_mean_of_a_steady_stream() {
        let b = JobBucket::default();
        for _ in 0..JOB_BUCKET_MEMORY * 4 {
            b.record(1_000);
        }
        assert_eq!(mean(&b), 1_000, "halving must not shift a constant mean");
        assert!(
            b.count.load(std::sync::atomic::Ordering::Relaxed) <= JOB_BUCKET_MEMORY,
            "memory is unbounded",
        );
    }

    #[test]
    fn an_incident_is_forgotten_once_normal_work_resumes() {
        let b = JobBucket::default();
        // Normal, then an outage's worth of multi-minute jobs, then normal again.
        for _ in 0..200 {
            b.record(5_000_000); // 5s
        }
        for _ in 0..30 {
            b.record(3_300_000_000); // 55 min
        }
        let poisoned = mean(&b);
        assert!(
            poisoned > 100_000_000,
            "test setup failed to poison the mean"
        );
        for _ in 0..JOB_BUCKET_MEMORY * 6 {
            b.record(5_000_000);
        }
        let recovered = mean(&b);
        assert!(
            recovered < 6_000_000,
            "still poisoned after recovery: {recovered}us (was {poisoned}us)",
        );
    }

    fn recent(b: &JobBucket) -> serde_json::Value {
        serde_json::to_value(b.recent()).expect("serializes")
    }

    // `recent` is what ships on /_/stats, and routers index it by these
    // exact names. Asserting the shape here is what stops a rename from
    // silently demoting the router back to lifetime means — a failure that
    // looks like nothing at all from the outside.
    #[test]
    fn recent_publishes_the_keys_routers_read() {
        let b = JobBucket::default();
        b.record(9_000_000); // 9s
        let v = recent(&b);
        assert_eq!(v["samples"], 1);
        assert!(
            v["p80_ms"].is_number(),
            "p80_ms missing or not a number: {v}"
        );
        assert!(
            v["mean_ms"].is_number(),
            "mean_ms missing or not a number: {v}"
        );
    }

    #[test]
    fn recent_reports_an_untouched_bucket_as_empty_not_zero() {
        let v = recent(&JobBucket::default());
        assert_eq!(v["samples"], 0);
        assert!(
            v["p80_ms"].is_null(),
            "an unsampled class must not claim 0ms"
        );
    }

    // The cumulative and windowed views answer different questions and must
    // both advance: routing reads one, operators read the other.
    #[test]
    fn record_feeds_both_the_lifetime_and_the_windowed_view() {
        let b = JobBucket::default();
        for _ in 0..5 {
            b.record(2_000_000);
        }
        assert_eq!(b.count.load(std::sync::atomic::Ordering::Relaxed), 5);
        assert_eq!(recent(&b)["samples"], 5);
    }
}
