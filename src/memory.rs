//! How `--max-rss-gb` is interpreted.
//!
//! This lives in the library, not in a binary, because every process that drives
//! analysis needs the same answer. The meanings of `0` collide: an
//! already-resolved ceiling uses `0` for *unlimited*, while every CLI spells `0`
//! as *resolve one for me*, so a raw value passed through would disable memory
//! admission by default. Callers hand over a [`MaxRssPolicy`] instead, and it is
//! resolved here.

use std::num::NonZeroU64;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// User-supplied resolution policy for `--max-rss-gb`.
///
/// The CLI accepts an `i64` so a negative value can opt out, but the three
/// possible behaviours are encoded in the type system from this point on so
/// that downstream code cannot accidentally treat "disabled" as "ceiling = 0".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxRssPolicy {
    /// `--max-rss-gb=-1`: disable in-process RSS throttling entirely. Use when
    /// an external supervisor (systemd `MemoryMax=`, jail rctl, etc.) already
    /// enforces a hard memory cap.
    Disabled,
    /// `--max-rss-gb=0` (the default): resolve a ceiling from the host, clamped
    /// to any cgroup limit that applies.
    Auto,
    /// An explicit ceiling in GiB.
    Explicit(NonZeroU64),
}

impl MaxRssPolicy {
    /// The whole-process ceiling in bytes, resolved against this host's
    /// memory and cgroup limit; `None` when throttling is disabled.
    #[must_use]
    pub fn process_ceiling(self) -> Option<NonZeroU64> {
        match self {
            Self::Disabled => None,
            Self::Auto => NonZeroU64::new(cleave::memory_tracker::memory_limit()),
            Self::Explicit(gb) => NonZeroU64::new(gb.get().saturating_mul(GIB)),
        }
    }

    /// Interpret a raw `--max-rss-gb` value: negative disables, `0` resolves,
    /// positive is an explicit GiB ceiling.
    #[must_use]
    pub fn from_cli(raw: i64) -> Self {
        match raw {
            n if n < 0 => Self::Disabled,
            0 => Self::Auto,
            // The previous arms exclude n <= 0, so `cast_unsigned` is
            // value-preserving and `NonZeroU64::new` always returns `Some`.
            // `unwrap_or(MIN)` documents that the fallback is unreachable.
            n => Self::Explicit(NonZeroU64::new(n.cast_unsigned()).unwrap_or(NonZeroU64::MIN)),
        }
    }
}

/// Emit a startup log line describing how `--max-rss-gb` was resolved. The
/// explicit case is intentionally silent: the user picked the number, so
/// echoing it back adds no information.
pub fn log_max_rss_resolution(role: &'static str, policy: MaxRssPolicy, resolved_bytes: u64) {
    match policy {
        MaxRssPolicy::Disabled => tracing::info!(
            role,
            "in-process RSS throttling disabled (--max-rss-gb=-1); \
             relying on external supervisor for OOM enforcement",
        ),
        MaxRssPolicy::Auto => tracing::info!(
            role,
            resolved_max_rss_mb = resolved_bytes / MIB,
            "auto-resolved RSS ceiling (set --max-rss-gb to override, -1 to disable)",
        ),
        MaxRssPolicy::Explicit(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_resolves_a_real_ceiling_rather_than_meaning_unlimited() {
        // A front end passing raw 0 into its startup config would disable
        // admission. Resolved, 0 must yield a positive ceiling.
        assert!(MaxRssPolicy::from_cli(0).process_ceiling().is_some());
    }

    #[test]
    fn negative_disables_and_explicit_is_verbatim() {
        assert_eq!(MaxRssPolicy::from_cli(-1).process_ceiling(), None);
        assert_eq!(
            MaxRssPolicy::from_cli(3)
                .process_ceiling()
                .map(NonZeroU64::get),
            Some(3 * GIB)
        );
    }
}
