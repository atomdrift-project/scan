//! Machine-wide cores busy, windowed between reads of `/_/stats`.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::lock;

/// Cores busy across the whole machine, averaged between two reads of
/// `/_/stats`.
///
/// The router polls stats every few seconds, so each poll sees the mean over
/// the interval since the last one — the window that matters for deciding
/// where the next analysis goes. Derived from the kernel's cumulative CPU
/// counters rather than the load average because the load average is not the
/// same number on every platform: Linux counts threads blocked on disk, FreeBSD
/// does not, and a scan host does a great deal of disk. `None` until two
/// reads exist, and on platforms with no counters; the caller then falls back
/// to `load1`.
#[derive(Default)]
pub(super) struct CpuBusy {
    last: Mutex<Option<CpuSample>>,
}

struct CpuSample {
    at: Instant,
    counters: cleave::memory_tracker::CpuTime,
    busy: Option<f64>,
}

/// Reads closer together than this share one window. Without it a second
/// poller — another router, an operator's curl — reset the window for the
/// first and handed it the few milliseconds in between.
const CPU_BUSY_MIN_WINDOW: Duration = Duration::from_secs(1);

impl CpuBusy {
    /// Logical cores busy since the previous window closed, the standing
    /// answer inside a window or when the counters have not moved, or `None`
    /// with nothing to compare yet.
    pub(super) fn sample(&self) -> Option<f64> {
        let mut last = lock(&self.last);
        if let Some(prev) = last.as_ref()
            && prev.at.elapsed() < CPU_BUSY_MIN_WINDOW
        {
            return prev.busy;
        }
        let now = cleave::memory_tracker::cpu_time()?;
        let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        let busy = last
            .as_ref()
            .and_then(|prev| cores_busy(prev.counters, now, cpus).or(prev.busy));
        *last = Some(CpuSample {
            at: Instant::now(),
            counters: now,
            busy,
        });
        busy
    }
}

/// `Δbusy / (Δbusy + Δidle)` of the machine, times its logical CPUs. `None`
/// when the counters have not advanced (two reads inside one tick) or ran
/// backwards (a counter reset), so the caller keeps its previous answer.
fn cores_busy(
    prev: cleave::memory_tracker::CpuTime,
    now: cleave::memory_tracker::CpuTime,
    cpus: usize,
) -> Option<f64> {
    let busy = now.busy.checked_sub(prev.busy)?;
    let idle = now.idle.checked_sub(prev.idle)?;
    let total = busy.checked_add(idle)?;
    (total > 0).then(|| cpus as f64 * busy as f64 / total as f64)
}

#[cfg(test)]
mod tests {
    use super::cores_busy;
    use cleave::memory_tracker::CpuTime;

    #[test]
    fn cores_busy_is_the_busy_share_of_the_machine() {
        let a = CpuTime {
            busy: 1000,
            idle: 3000,
        };
        let b = CpuTime {
            busy: 1300,
            idle: 3100,
        };
        // 300 busy of 400 elapsed ticks on 16 CPUs: twelve cores' worth.
        assert_eq!(cores_busy(a, b, 16), Some(12.0));
        assert_eq!(cores_busy(a, a, 16), None, "no elapsed ticks, no answer");
        assert_eq!(
            cores_busy(b, a, 16),
            None,
            "a counter that ran backwards is not a reading"
        );
    }

    /// A second reader inside the window gets the standing answer instead of
    /// resetting the window to the few milliseconds since the first.
    #[test]
    fn a_second_reader_does_not_shrink_the_window() {
        let busy = super::CpuBusy::default();
        let Some(first) = cleave::memory_tracker::cpu_time() else {
            return; // No counters on this platform: nothing to window.
        };
        let at = std::time::Instant::now();
        *super::lock(&busy.last) = Some(super::CpuSample {
            at,
            counters: first,
            busy: Some(3.5),
        });
        assert_eq!(busy.sample(), Some(3.5));
        let kept = super::lock(&busy.last).as_ref().map(|s| s.at);
        assert_eq!(kept, Some(at), "an early read moved the window");
    }
}
