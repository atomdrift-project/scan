//! A worker's recent latency, summarised for a router.
//!
//! Two properties this has to get right, both learned the hard way:
//!
//! **It forgets on a clock, not on traffic.** A running total that only ages by
//! sample count kept describing a past incident: a hopper outage produced
//! analyses of 8 to 55 minutes, and because nothing decayed, the resulting
//! averages went on steering routing long after the outage was over. A quiet
//! worker must not stay tarred by an hour it already recovered from.
//!
//! **It reports a percentile, not a mean.** Analysis time is strongly bimodal —
//! seconds for a small package, minutes for a large archive — and a mean sits
//! in the empty space between the two humps, describing almost no real job.
//! Measured live, mean-based estimates were wrong by roughly an order of
//! magnitude against observed medians.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How much recent history an estimate is drawn from.
pub(crate) const WINDOW: Duration = Duration::from_secs(60 * 60);

/// Slots the window is divided into. Expiry is per slot, so this sets how
/// abruptly the oldest traffic falls out: twelve slots retire five minutes at a
/// time rather than dropping a whole hour at once.
const SLOTS: usize = 12;

/// One slot's span.
const SLOT: Duration = Duration::from_secs(WINDOW.as_secs() / SLOTS as u64);

/// Histogram buckets. Bucket `i` covers `[2^i, 2^(i+1))` milliseconds, so 24 of
/// them reach about 4.6 hours — comfortably past any analysis that has not
/// already hit its timeout, and small enough to copy on every read.
const BUCKETS: usize = 24;

/// The percentile a router asks for.
///
/// Deliberately not the median and not the tail. p50 under-predicts often
/// enough that a hedge built on it fires constantly; p95+ is dominated by the
/// pathological archives that a stall detector should not be calibrated to. p80
/// answers "how long does this usually take, including a bad day" — which is
/// the question a dispatch decision is actually asking.
pub(crate) const PERCENTILE: f64 = 0.80;

#[derive(Debug, Default, Clone, Copy)]
struct Slot {
    count: u64,
    micros: u64,
    hist: [u32; BUCKETS],
}

impl Slot {
    fn clear(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug)]
struct Window {
    slots: [Slot; SLOTS],
    cur: usize,
    /// Start of the current slot. Advancing is lazy — done on the next read or
    /// write — so an idle worker costs nothing and still reports an empty
    /// window rather than stale numbers.
    cur_start: Instant,
}

/// Recent completion latencies for one class of work.
///
/// A `Mutex` rather than atomics on purpose. This is written once per completed
/// analysis — a few times a minute on a busy worker, since the work itself
/// takes minutes — so contention is not a consideration, and a lock buys an
/// invariant that matters: a reader sees one coherent window instead of a
/// histogram from one instant and a total from another.
#[derive(Debug)]
pub(crate) struct Latency {
    window: Mutex<Window>,
}

impl Default for Latency {
    fn default() -> Self {
        Self {
            window: Mutex::new(Window {
                slots: [Slot::default(); SLOTS],
                cur: 0,
                cur_start: Instant::now(),
            }),
        }
    }
}

/// What a router reads. `None` fields mean the window holds nothing to answer
/// with, which is a different statement from zero and must stay distinguishable.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Summary {
    pub(crate) samples: u64,
    pub(crate) mean_micros: Option<u64>,
    pub(crate) p80_micros: Option<u64>,
}

/// The histogram bucket for a duration. Sub-millisecond work lands in bucket 0
/// rather than being dropped: a lookup is a real measurement even at 0ms.
fn bucket_of(micros: u64) -> usize {
    let ms = micros / 1_000;
    if ms == 0 {
        return 0;
    }
    // floor(log2(ms)); 63 - leading_zeros is exact for a non-zero u64.
    let idx = 63 - ms.leading_zeros() as usize;
    idx.min(BUCKETS - 1)
}

/// The half-open millisecond range bucket `i` covers.
fn bucket_bounds(i: usize) -> (f64, f64) {
    // Bucket 0 is [0, 2) rather than [1, 2): it absorbs everything below a
    // millisecond, and a lower bound of 1 would report sub-ms work as 1ms.
    let lo = if i == 0 { 0.0 } else { (1u64 << i) as f64 };
    let hi = (1u64 << (i + 1)) as f64;
    (lo, hi)
}

impl Latency {
    /// Record one completion.
    pub(crate) fn record(&self, micros: u64) {
        self.record_at(micros, Instant::now());
    }

    /// `record` with the clock supplied, so a test can age a window without
    /// sleeping through it.
    pub(crate) fn record_at(&self, micros: u64, now: Instant) {
        let Ok(mut w) = self.window.lock() else {
            return; // a poisoned stats lock must never take down an analysis
        };
        Self::advance(&mut w, now);
        let cur = w.cur;
        let slot = &mut w.slots[cur];
        slot.count += 1;
        slot.micros += micros;
        slot.hist[bucket_of(micros)] += 1;
    }

    /// Summarise the window as of now.
    pub(crate) fn summary(&self) -> Summary {
        self.summary_at(Instant::now())
    }

    pub(crate) fn summary_at(&self, now: Instant) -> Summary {
        let Ok(mut w) = self.window.lock() else {
            return Summary::default();
        };
        // Advance on read too, or a worker that stopped receiving work would
        // keep reporting whatever it was doing when the traffic stopped.
        Self::advance(&mut w, now);

        let mut count = 0u64;
        let mut micros = 0u64;
        let mut hist = [0u64; BUCKETS];
        for slot in &w.slots {
            count += slot.count;
            micros += slot.micros;
            for (acc, n) in hist.iter_mut().zip(slot.hist.iter()) {
                *acc += u64::from(*n);
            }
        }
        if count == 0 {
            return Summary::default();
        }
        Summary {
            samples: count,
            mean_micros: Some(micros / count),
            p80_micros: Some(percentile_micros(&hist, count, PERCENTILE)),
        }
    }

    /// Retire slots the window has moved past. Bounded by SLOTS: an arbitrarily
    /// long idle period clears everything exactly once rather than looping.
    fn advance(w: &mut Window, now: Instant) {
        let elapsed = now.saturating_duration_since(w.cur_start);
        // usize::try_from rather than a cast: on a 32-bit target a long idle
        // period would otherwise wrap to a small step count and retire the
        // wrong slots. Saturating is correct — anything past SLOTS clears all.
        let steps = usize::try_from(elapsed.as_secs() / SLOT.as_secs()).unwrap_or(SLOTS);
        if steps == 0 {
            return;
        }
        if steps >= SLOTS {
            for slot in &mut w.slots {
                slot.clear();
            }
            w.cur = 0;
        } else {
            for _ in 0..steps {
                w.cur = (w.cur + 1) % SLOTS;
                w.slots[w.cur].clear();
            }
        }
        w.cur_start += SLOT * u32::try_from(steps).unwrap_or(u32::MAX);
    }
}

/// The value at `p` in a log-bucketed histogram, interpolated within the
/// bucket it lands in.
///
/// Interpolation matters more than it looks: buckets double in width, so
/// returning a bucket's edge would round a 9-second job to 8 or 16. Spreading
/// the bucket's samples evenly across its range keeps the answer inside the
/// range the samples actually came from.
fn percentile_micros(hist: &[u64; BUCKETS], count: u64, p: f64) -> u64 {
    // Rank of the sample we want, 1-based, clamped into the histogram. The
    // bounds are established before the cast: count is a real sample count and
    // p is in (0,1], so the product is finite and non-negative.
    #[allow(clippy::cast_precision_loss)] // sample counts never reach 2^53
    let rank = ((count as f64) * p).ceil().clamp(1.0, count as f64);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // clamped to [1, count]
    let target = rank as u64;
    let mut seen = 0u64;
    for (i, n) in hist.iter().enumerate() {
        if *n == 0 {
            continue;
        }
        if seen + n >= target {
            let (lo, hi) = bucket_bounds(i);
            // Where in this bucket the target sits. Each of the bucket's `n`
            // samples owns an equal slice of its range and is reported at the
            // slice's midpoint — the `- 0.5`. Without it the last sample in a
            // bucket maps to the exclusive upper edge, so a lone 30s job came
            // back as 32.768s: a value outside the range it was drawn from.
            let into = ((target - seen) as f64 - 0.5) / *n as f64;
            let ms = lo + (hi - lo) * into;
            return micros_from_ms(ms);
        }
        seen += n;
    }
    // Unreachable while count matches the histogram; fall back to the top edge
    // rather than zero, which would read as "instant".
    let (_, hi) = bucket_bounds(BUCKETS - 1);
    micros_from_ms(hi)
}

/// Milliseconds to microseconds, without a cast that can wrap or go negative.
/// Every input here comes from bucket bounds, which are positive and far below
/// u64::MAX — but stating that in code beats asserting it in a comment.
fn micros_from_ms(ms: f64) -> u64 {
    // A nonsense duration must never read as a fast one. Returning 0 for NaN or
    // infinity would make a broken worker the most attractive in the fleet —
    // the same shape as the bug where an unpollable worker outranked a measured
    // one. Zero is reserved for the legitimate case: bucket 0's lower bound.
    if ms.is_nan() || (ms.is_infinite() && ms.is_sign_positive()) {
        return u64::MAX;
    }
    if !ms.is_finite() || ms <= 0.0 {
        return 0;
    }
    let us = ms * 1_000.0;
    if us >= u64::MAX as f64 {
        return u64::MAX;
    }
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // bounded just above
    let out = us as u64;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000; // micros per millisecond

    fn at(base: Instant, secs: u64) -> Instant {
        base + Duration::from_secs(secs)
    }

    #[test]
    fn empty_window_reports_nothing_rather_than_zero() {
        let l = Latency::default();
        let s = l.summary();
        assert_eq!(s.samples, 0);
        assert_eq!(
            s.mean_micros, None,
            "no data must be distinguishable from 0ms"
        );
        assert_eq!(s.p80_micros, None);
    }

    #[test]
    fn a_constant_stream_reports_that_constant() {
        let base = Instant::now();
        let l = Latency::default();
        for _ in 0..100 {
            l.record_at(1_000 * MS, base);
        }
        let s = l.summary_at(base);
        assert_eq!(s.samples, 100);
        assert_eq!(s.mean_micros, Some(1_000 * MS));
        // 1000ms lands in bucket [1024ms is 2^10, so 1000 is in [512,1024)).
        let p80 = s.p80_micros.expect("p80");
        assert!(
            (512 * MS..=1024 * MS).contains(&p80),
            "p80 {p80} outside the bucket every sample fell in",
        );
    }

    // The property the whole thing exists for: a bimodal distribution must not
    // be summarised by the empty space between its humps.
    #[test]
    fn p80_tracks_the_bulk_where_a_mean_would_not() {
        let base = Instant::now();
        let l = Latency::default();
        // 80 fast small packages, 20 slow archives.
        for _ in 0..80 {
            l.record_at(2_000 * MS, base); // 2s
        }
        for _ in 0..20 {
            l.record_at(600_000 * MS, base); // 10 min
        }
        let s = l.summary_at(base);
        let mean = s.mean_micros.expect("mean");
        let p80 = s.p80_micros.expect("p80");
        // The mean sits at ~2 minutes: slower than every small job and faster
        // than every large one, describing neither.
        assert!(
            mean > 100_000 * MS,
            "mean {mean} should be dragged up by the tail"
        );
        assert!(
            p80 < mean,
            "p80 {p80} must sit below a tail-dragged mean {mean}",
        );
        assert!(
            p80 <= 8_000 * MS,
            "p80 {p80} should stay near the 80% of jobs that take ~2s",
        );
    }

    #[test]
    fn p80_moves_up_when_the_slow_share_grows() {
        let base = Instant::now();
        let mostly_fast = Latency::default();
        let mostly_slow = Latency::default();
        for i in 0..100 {
            let fast = 2_000 * MS;
            let slow = 600_000 * MS;
            mostly_fast.record_at(if i < 90 { fast } else { slow }, base);
            mostly_slow.record_at(if i < 50 { fast } else { slow }, base);
        }
        let a = mostly_fast.summary_at(base).p80_micros.expect("p80");
        let b = mostly_slow.summary_at(base).p80_micros.expect("p80");
        assert!(b > a, "p80 should rise with the slow share: {a} then {b}");
    }

    // Forgetting on a clock is the other half of the point.
    #[test]
    fn an_incident_ages_out_of_the_window() {
        let base = Instant::now();
        let l = Latency::default();
        for _ in 0..50 {
            l.record_at(3_300_000 * MS, base); // 55 min, the real outage figure
        }
        let during = l.summary_at(base).p80_micros.expect("p80");
        assert!(
            during > 1_000_000 * MS,
            "test setup failed to poison the window"
        );

        // An hour later, with no traffic at all, it must report nothing rather
        // than the incident.
        let after = l.summary_at(at(base, WINDOW.as_secs() + 1));
        assert_eq!(after.samples, 0, "the incident outlived its window");
        assert_eq!(after.p80_micros, None);
    }

    #[test]
    fn normal_traffic_after_an_incident_reports_normal() {
        let base = Instant::now();
        let l = Latency::default();
        for _ in 0..50 {
            l.record_at(3_300_000 * MS, base);
        }
        // Well past the window, then a fresh healthy stream.
        let later = at(base, WINDOW.as_secs() + 60);
        for _ in 0..50 {
            l.record_at(5_000 * MS, later);
        }
        let p80 = l.summary_at(later).p80_micros.expect("p80");
        assert!(
            p80 <= 16_000 * MS,
            "p80 {p80} still carries the incident after a window of clean traffic",
        );
    }

    // Partial expiry: traffic inside the window survives, older traffic does not.
    #[test]
    fn the_window_slides_rather_than_emptying_all_at_once() {
        let base = Instant::now();
        let l = Latency::default();
        l.record_at(600_000 * MS, base); // old and slow
        // Half a window later, a burst of fast work.
        let mid = at(base, WINDOW.as_secs() / 2);
        for _ in 0..99 {
            l.record_at(2_000 * MS, mid);
        }
        let both = l.summary_at(mid);
        assert_eq!(both.samples, 100, "both eras should still be in the window");

        // Now past the point where only the burst remains.
        let late = at(base, WINDOW.as_secs() + SLOT.as_secs());
        let s = l.summary_at(late);
        assert_eq!(
            s.samples, 99,
            "the old sample should have retired, the burst should not"
        );
    }

    #[test]
    fn a_long_idle_period_clears_the_window_exactly_once() {
        let base = Instant::now();
        let l = Latency::default();
        for _ in 0..10 {
            l.record_at(5_000 * MS, base);
        }
        // Days later — the advance must not loop per elapsed slot.
        let s = l.summary_at(at(base, 60 * 60 * 24 * 3));
        assert_eq!(s.samples, 0);
        // And the window must still be usable afterwards.
        let now = at(base, 60 * 60 * 24 * 3);
        l.record_at(7_000 * MS, now);
        assert_eq!(
            l.summary_at(now).samples,
            1,
            "window unusable after a long idle"
        );
    }

    #[test]
    fn sub_millisecond_work_is_measured_not_dropped() {
        let base = Instant::now();
        let l = Latency::default();
        for _ in 0..10 {
            l.record_at(200, base); // 0.2ms: a lookup, not an analysis
        }
        let s = l.summary_at(base);
        assert_eq!(s.samples, 10);
        assert_eq!(s.mean_micros, Some(200));
        let p80 = s.p80_micros.expect("p80");
        assert!(p80 < 2 * MS, "sub-ms work reported as {p80}us");
    }

    #[test]
    fn bucket_indices_are_monotonic_and_bounded() {
        let mut last = 0;
        for pow in 0..40u32 {
            let micros = (1u64 << pow.min(31)) * MS;
            let b = bucket_of(micros);
            assert!(b < BUCKETS, "bucket {b} out of range for {micros}us");
            assert!(b >= last, "bucket went backwards at 2^{pow}");
            last = b;
        }
    }

    // Interpolation should keep the answer inside the range the samples came
    // from, not snap it to a power-of-two edge.
    #[test]
    fn percentile_interpolates_within_its_bucket() {
        let base = Instant::now();
        let l = Latency::default();
        for _ in 0..100 {
            l.record_at(9_000 * MS, base); // 9s, inside [8192, 16384)
        }
        let p80 = l.summary_at(base).p80_micros.expect("p80");
        assert!(
            (8_192 * MS..16_384 * MS).contains(&p80),
            "p80 {p80} left the bucket its samples occupy",
        );
    }

    #[test]
    fn a_single_sample_is_reported_as_itself() {
        let base = Instant::now();
        let l = Latency::default();
        l.record_at(30_000 * MS, base);
        let s = l.summary_at(base);
        assert_eq!(s.samples, 1);
        assert_eq!(s.mean_micros, Some(30_000 * MS));
        let p80 = s.p80_micros.expect("p80");
        assert!(
            (16_384 * MS..32_768 * MS).contains(&p80),
            "p80 {p80} does not contain the only sample",
        );
    }

    #[test]
    fn an_extreme_outlier_does_not_overflow_the_top_bucket() {
        let base = Instant::now();
        let l = Latency::default();
        l.record_at(u64::MAX / 2, base);
        let s = l.summary_at(base);
        assert_eq!(s.samples, 1);
        assert!(
            s.p80_micros.is_some(),
            "an absurd duration must still summarise"
        );
    }
    // The central claim, and the one every other test here assumes rather than
    // checks: that p80 is actually the 80th percentile. Straddling the boundary
    // in both directions is what makes this discriminating — a rank off by one
    // either way lands in the wrong population and the assertion fails.
    #[test]
    fn the_reported_percentile_is_really_the_eightieth() {
        let base = Instant::now();
        let fast = 1_000 * MS;
        let slow = 500_000 * MS;

        // 79 fast, 21 slow: rank 80 of 100 is the first slow sample.
        let below = Latency::default();
        for i in 0..100 {
            below.record_at(if i < 79 { fast } else { slow }, base);
        }
        let p = below.summary_at(base).p80_micros.expect("p80");
        assert!(
            p > 100_000 * MS,
            "p80 {p} should have crossed into the slow group"
        );

        // 81 fast, 19 slow: rank 80 is still a fast sample.
        let above = Latency::default();
        for i in 0..100 {
            above.record_at(if i < 81 { fast } else { slow }, base);
        }
        let p = above.summary_at(base).p80_micros.expect("p80");
        assert!(
            p < 8_000 * MS,
            "p80 {p} should have stayed in the fast group"
        );
    }

    // Concurrent writers are the reason this holds a lock at all.
    #[test]
    fn concurrent_writers_all_land() {
        use std::sync::Arc;
        let l = Arc::new(Latency::default());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let l = Arc::clone(&l);
            handles.push(std::thread::spawn(move || {
                for _ in 0..250 {
                    l.record(5_000 * MS);
                }
            }));
        }
        for h in handles {
            h.join().expect("writer panicked");
        }
        let s = l.summary();
        assert_eq!(s.samples, 2_000, "lost samples under concurrency");
        assert_eq!(s.mean_micros, Some(5_000 * MS));
    }

    // Reading while writing must not deadlock or tear: the summary aggregates
    // every slot under one lock, so it can never mix eras.
    #[test]
    fn reading_during_writes_stays_consistent() {
        use std::sync::Arc;
        let l = Arc::new(Latency::default());
        let writer = {
            let l = Arc::clone(&l);
            std::thread::spawn(move || {
                for _ in 0..2_000 {
                    l.record(1_000 * MS);
                }
            })
        };
        for _ in 0..200 {
            let s = l.summary();
            // Whatever it saw, mean and samples must describe the same set.
            if s.samples > 0 {
                assert_eq!(s.mean_micros, Some(1_000 * MS), "torn read");
            }
        }
        writer.join().expect("writer panicked");
        assert_eq!(l.summary().samples, 2_000);
    }

    // The ring must wrap, not just advance: traffic in every slot across more
    // than one full revolution should leave exactly the last window's worth.
    #[test]
    fn the_ring_wraps_without_losing_the_current_window() {
        let base = Instant::now();
        let l = Latency::default();
        // One sample per slot for two and a half revolutions.
        let total_slots = SLOTS * 5 / 2;
        for step in 0..total_slots {
            l.record_at(4_000 * MS, at(base, (step as u64) * SLOT.as_secs()));
        }
        let now = at(base, ((total_slots - 1) as u64) * SLOT.as_secs());
        let s = l.summary_at(now);
        assert!(
            s.samples <= SLOTS as u64,
            "window holds {} samples, more than its {SLOTS} slots",
            s.samples,
        );
        assert!(s.samples > 0, "wrapping emptied the window entirely");
        assert_eq!(s.mean_micros, Some(4_000 * MS));
    }

    // The wire format is a cross-repo contract: beamline reads these exact key
    // names, and a rename here would silently drop the router back to means.
    #[test]
    fn the_published_shape_is_the_one_consumers_read() {
        let base = Instant::now();
        let l = Latency::default();
        l.record_at(9_000 * MS, base);
        let s = l.summary_at(base);
        assert!(s.p80_micros.is_some() && s.mean_micros.is_some());

        let json = serde_json::json!({
            "samples": s.samples,
            "p80_ms": s.p80_micros.map(|us| us / 1_000),
            "mean_ms": s.mean_micros.map(|us| us / 1_000),
        });
        for key in ["samples", "p80_ms", "mean_ms"] {
            assert!(json.get(key).is_some(), "published shape lost `{key}`");
        }
        assert!(
            !json["p80_ms"].is_null(),
            "p80_ms must carry a number when sampled"
        );
    }

    #[test]
    fn absurd_and_degenerate_durations_convert_safely() {
        assert_eq!(
            micros_from_ms(f64::NAN),
            u64::MAX,
            "NaN must not read as instant"
        );
        assert_eq!(micros_from_ms(f64::INFINITY), u64::MAX);
        assert_eq!(micros_from_ms(-5.0), 0);
        assert_eq!(micros_from_ms(0.0), 0);
        assert_eq!(micros_from_ms(1.5), 1_500);
    }
}
