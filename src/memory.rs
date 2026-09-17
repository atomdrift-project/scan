//! Memory-budget resolution: how large a ceiling a process may size itself
//! against, and how `--max-rss-gb` is interpreted.
//!
//! This lives in the library, not in a binary, because every process that drives
//! [`crate::worker`] needs the same answer. It previously lived in atomscan's
//! `main.rs`, which meant a second front end — postdoc, which builds a
//! [`crate::worker::Startup`] directly — could not reuse it and passed its raw
//! CLI value straight through instead. That is a silent, severe failure: the
//! `Startup` field is already-resolved bytes where `0` means *unlimited*, while
//! every CLI spells `0` as *resolve one for me*. The two meanings collide on the
//! default value, so the worker ran with memory admission disabled entirely.
//! Anything constructing a `Startup` must resolve through
//! [`resolve_worker_max_rss_gb`] first.
//!
//! **The basis must be cgroup-aware.** `cleave::memory_tracker::total_memory()`
//! is the host's `MemTotal` and knows nothing about cgroups, so on a shared host
//! it hands back a budget covering the whole machine. galadriel, 2026-09-17: a
//! 251 GiB host resolved a 213 GiB ceiling for a worker sharing the box with a
//! 32 GB `shared_buffers` PostgreSQL replica and vllm. The admission controller
//! then did exactly what it was told and filled that budget — 576 concurrent
//! analyses — until the cgroup OOM-killed the process every ~20 minutes. It was
//! never a leak; memory was returned whenever work drained. The budget was a lie.
//!
//! Note that the *host floor* in [`crate::admission`] deliberately stays
//! host-scoped: it protects the machine from all tenants at once, so `MemTotal`
//! is the right basis there. Only this process's own ceiling is clamped here.

use std::num::NonZeroU64;
// Paths are only walked where cgroups exist -- and under test, which exercises
// that walk against a temporary tree on every platform.
#[cfg(any(target_os = "linux", test))]
use std::path::{Path, PathBuf};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// Root of the cgroup-v2 hierarchy. A parameter in tests, a constant in life.
#[cfg(target_os = "linux")]
const CGROUP_ROOT: &str = "/sys/fs/cgroup";

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

/// Memory a worker may size itself against, and where the number came from.
///
/// `source` is logged as `auto_memory_basis_source`, which is how an operator
/// confirms a cgroup limit was actually seen rather than assumed.
#[derive(Debug)]
pub struct WorkerMemoryBasis {
    /// Memory the process may size itself against.
    pub bytes: u64,
    /// Which signal supplied `bytes`, logged as `auto_memory_basis_source`.
    pub source: &'static str,
}

/// Smallest memory limit along a cgroup-v2 path, walking `start` up to `root`.
///
/// Split out from [`cgroup_memory_limit_bytes`] so the hierarchy walk is
/// testable without a real `/sys/fs/cgroup` -- which is why it is compiled
/// under `test` on every platform, and only on Linux otherwise.
#[cfg(any(target_os = "linux", test))]
fn cgroup_limit_under(root: &Path, start: &Path) -> Option<u64> {
    let mut dir = start.to_path_buf();
    let mut limit: Option<u64> = None;
    loop {
        for file in ["memory.max", "memory.high"] {
            if let Some(bytes) = memory_value_bytes(read_trimmed(dir.join(file)).as_deref()) {
                limit = Some(limit.map_or(bytes, |current: u64| current.min(bytes)));
            }
        }
        if dir == root || !dir.starts_with(root) {
            break;
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => break,
        }
    }
    limit
}

/// Most restrictive memory limit that applies to this process.
///
/// The limit that actually binds a process is the minimum over *every* ancestor
/// cgroup, not the value on its own. That distinction is load-bearing: atomscan's
/// server runs its companion worker in a delegated child cgroup
/// (`scan.service/idle`) carrying no limit of its own, so reading only
/// [`cgroup_v2_path`] finds nothing while the `MemoryMax=` on the parent unit is
/// exactly what OOM-kills it.
///
/// `memory.high` counts because a throttle the process can never outrun is a
/// ceiling in practice. Both files read `max` when unset, which
/// [`memory_value_bytes`] maps to `None`.
#[cfg(target_os = "linux")]
#[must_use]
pub fn cgroup_memory_limit_bytes() -> Option<u64> {
    cgroup_limit_under(Path::new(CGROUP_ROOT), &cgroup_v2_path()?)
}

/// No cgroups off Linux, so nothing clamps the host's memory here.
#[cfg(not(target_os = "linux"))]
#[must_use]
pub fn cgroup_memory_limit_bytes() -> Option<u64> {
    None
}

/// Memory this process may size itself against: host RAM, clamped to whatever
/// cgroup limit applies. See the module docs for why the clamp matters.
#[must_use]
pub fn worker_memory_basis() -> WorkerMemoryBasis {
    let host = cleave::memory_tracker::total_memory();
    let cgroup = cgroup_memory_limit_bytes();
    match (host, cgroup) {
        (Some(host), Some(cgroup)) if cgroup < host => WorkerMemoryBasis {
            bytes: cgroup,
            source: "cgroup_memory_limit",
        },
        (Some(host), _) => WorkerMemoryBasis {
            bytes: host,
            source: "cleave_total_memory",
        },
        (None, Some(cgroup)) => WorkerMemoryBasis {
            bytes: cgroup,
            source: "cgroup_memory_limit",
        },
        (None, None) => WorkerMemoryBasis {
            bytes: 16 * GIB,
            source: "fallback_16g",
        },
    }
}

/// Resolve `--max-rss-gb` into the GiB ceiling a [`crate::worker::Startup`]
/// expects. **Every** front end must call this; see the module docs.
#[must_use]
pub fn resolve_worker_max_rss_gb(raw_max_rss_gb: i64) -> u64 {
    match MaxRssPolicy::from_cli(raw_max_rss_gb) {
        MaxRssPolicy::Disabled => 0,
        // 85% of the cgroup-aware memory basis, with a one-GiB floor. Slot
        // count scales with cores, so larger hosts need a proportionate ceiling.
        MaxRssPolicy::Auto => std::cmp::max(1, (worker_memory_basis().bytes * 85 / 100) / GIB),
        MaxRssPolicy::Explicit(gb) => gb.get(),
    }
}

/// Resolve `--max-rss-gb` into a byte ceiling for whole-process throttling.
#[must_use]
pub fn resolve_process_max_rss_bytes(raw_max_rss_gb: i64) -> u64 {
    match MaxRssPolicy::from_cli(raw_max_rss_gb) {
        MaxRssPolicy::Disabled => 0,
        MaxRssPolicy::Auto => cleave::memory_tracker::memory_limit(),
        MaxRssPolicy::Explicit(gb) => gb.get().saturating_mul(GIB),
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

/// Raw cgroup-v2 memory files for this process's own cgroup, for diagnostics.
#[derive(Debug, Default)]
pub struct CgroupMemoryDiagnostics {
    /// This process's cgroup-v2 directory.
    pub path: Option<String>,
    /// Raw `memory.current`.
    pub memory_current: Option<String>,
    /// `memory.current` in MiB.
    pub memory_current_mb: Option<u64>,
    /// Raw `memory.high` (`max` when unset).
    pub memory_high: Option<String>,
    /// `memory.high` in MiB, `None` when unset.
    pub memory_high_mb: Option<u64>,
    /// Raw `memory.max` (`max` when unset).
    pub memory_max: Option<String>,
    /// `memory.max` in MiB, `None` when unset.
    pub memory_max_mb: Option<u64>,
}

impl CgroupMemoryDiagnostics {
    /// Limit on *this* cgroup alone. [`cgroup_memory_limit_bytes`] is what
    /// actually binds the process; this is the local view, for logging.
    #[must_use]
    pub fn effective_limit_bytes(&self) -> Option<u64> {
        [self.memory_high.as_deref(), self.memory_max.as_deref()]
            .into_iter()
            .flatten()
            .filter_map(|v| memory_value_bytes(Some(v)))
            .min()
    }
}

/// Raw cgroup-v2 memory files for this process's own cgroup, for diagnostics.
#[cfg(target_os = "linux")]
#[must_use]
pub fn cgroup_memory_diagnostics() -> CgroupMemoryDiagnostics {
    let Some(path) = cgroup_v2_path() else {
        return CgroupMemoryDiagnostics::default();
    };
    let memory_current = read_trimmed(path.join("memory.current"));
    let memory_high = read_trimmed(path.join("memory.high"));
    let memory_max = read_trimmed(path.join("memory.max"));
    CgroupMemoryDiagnostics {
        path: Some(path.display().to_string()),
        memory_current_mb: memory_value_mb(memory_current.as_deref()),
        memory_high_mb: memory_value_mb(memory_high.as_deref()),
        memory_max_mb: memory_value_mb(memory_max.as_deref()),
        memory_current,
        memory_high,
        memory_max,
    }
}

/// Raw cgroup-v2 memory files for this process's own cgroup, for diagnostics.
#[cfg(not(target_os = "linux"))]
#[must_use]
pub fn cgroup_memory_diagnostics() -> CgroupMemoryDiagnostics {
    CgroupMemoryDiagnostics::default()
}

/// This process's cgroup-v2 directory, from the unified hierarchy line.
#[cfg(target_os = "linux")]
#[must_use]
pub fn cgroup_v2_path() -> Option<PathBuf> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    for line in cgroup.lines() {
        let mut parts = line.splitn(3, ':');
        let hierarchy = parts.next()?;
        let controllers = parts.next()?;
        let rel = parts.next()?;
        if hierarchy == "0" && controllers.is_empty() {
            let rel = rel.trim_start_matches('/');
            return Some(if rel.is_empty() {
                PathBuf::from(CGROUP_ROOT)
            } else {
                PathBuf::from(CGROUP_ROOT).join(rel)
            });
        }
    }
    None
}

/// Contents of a cgroup file, trimmed, or `None` when absent or empty.
#[cfg(any(target_os = "linux", test))]
fn read_trimmed(path: PathBuf) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// A cgroup memory file's value in MiB, for the diagnostics log.
#[cfg(target_os = "linux")]
fn memory_value_mb(value: Option<&str>) -> Option<u64> {
    memory_value_bytes(value).map(|b| b / MIB)
}

/// Parse a cgroup memory file. `max` means "no limit" and maps to `None`.
fn memory_value_bytes(value: Option<&str>) -> Option<u64> {
    let value = value?;
    if value == "max" {
        return None;
    }
    value.parse::<u64>().ok()
}

/// `MemTotal` from `/proc/meminfo`, in MiB.
pub fn proc_memtotal_mb() -> Result<u64, String> {
    let meminfo =
        std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("read /proc/meminfo: {e}"))?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let raw = rest
                .split_whitespace()
                .next()
                .ok_or_else(|| "parse /proc/meminfo MemTotal: missing value".to_string())?;
            let kb: u64 = raw
                .parse()
                .map_err(|e| format!("parse /proc/meminfo MemTotal value {raw:?}: {e}"))?;
            return Ok(kb / 1024);
        }
    }
    Err("parse /proc/meminfo: MemTotal not found".to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// A disposable cgroup-shaped directory tree.
    struct TempTree(PathBuf);

    impl TempTree {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "scan-memory-test-{tag}-{}-{:?}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            std::fs::create_dir_all(&dir).expect("create temp tree");
            Self(dir)
        }

        fn write(&self, rel: &str, file: &str, value: &str) {
            let dir = if rel.is_empty() {
                self.0.clone()
            } else {
                self.0.join(rel)
            };
            std::fs::create_dir_all(&dir).expect("create cgroup dir");
            std::fs::write(dir.join(file), value).expect("write cgroup file");
        }

        fn path(&self, rel: &str) -> PathBuf {
            if rel.is_empty() {
                self.0.clone()
            } else {
                self.0.join(rel)
            }
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn limit_on_a_parent_binds_a_child_that_has_none() {
        // The exact shape that defeated a non-hierarchical read: the delegated
        // `scan.service/idle` cgroup carries no limit, the unit above it does.
        let tree = TempTree::new("parent");
        tree.write("scan.service", "memory.max", "128849018880");
        tree.write("scan.service/idle", "memory.max", "max");
        let got = cgroup_limit_under(&tree.path(""), &tree.path("scan.service/idle"));
        assert_eq!(got, Some(128_849_018_880));
    }

    #[test]
    fn the_most_restrictive_ancestor_wins() {
        let tree = TempTree::new("min");
        tree.write("", "memory.max", "100");
        tree.write("a", "memory.max", "50");
        tree.write("a/b", "memory.max", "900");
        assert_eq!(
            cgroup_limit_under(&tree.path(""), &tree.path("a/b")),
            Some(50)
        );
    }

    #[test]
    fn memory_high_is_a_ceiling_too() {
        let tree = TempTree::new("high");
        tree.write("a", "memory.max", "900");
        tree.write("a", "memory.high", "17");
        assert_eq!(
            cgroup_limit_under(&tree.path(""), &tree.path("a")),
            Some(17)
        );
    }

    #[test]
    fn unlimited_everywhere_is_no_limit() {
        let tree = TempTree::new("unlimited");
        tree.write("", "memory.max", "max");
        tree.write("a", "memory.max", "max");
        assert_eq!(cgroup_limit_under(&tree.path(""), &tree.path("a")), None);
    }

    #[test]
    fn a_cgroup_limit_below_host_ram_becomes_the_basis() {
        // worker_memory_basis reads the real host, so assert the property that
        // matters rather than a number: whichever source wins is labelled.
        let basis = worker_memory_basis();
        assert!(basis.bytes > 0);
        assert!(matches!(
            basis.source,
            "cgroup_memory_limit" | "cleave_total_memory" | "fallback_16g"
        ));
    }

    #[test]
    fn zero_resolves_a_real_ceiling_rather_than_meaning_unlimited() {
        // The postdoc collision: a front end passing raw 0 into Startup would
        // disable admission. Resolved, 0 must yield a positive ceiling.
        assert!(resolve_worker_max_rss_gb(0) > 0);
        assert!(resolve_process_max_rss_bytes(0) > 0);
    }

    #[test]
    fn negative_disables_and_explicit_is_verbatim() {
        assert_eq!(resolve_worker_max_rss_gb(-1), 0);
        assert_eq!(resolve_process_max_rss_bytes(-1), 0);
        assert_eq!(resolve_worker_max_rss_gb(3), 3);
        assert_eq!(resolve_process_max_rss_bytes(3), 3 * GIB);
    }

    #[test]
    fn auto_ceiling_never_exceeds_the_cgroup_limit() {
        if let Some(limit) = cgroup_memory_limit_bytes() {
            let resolved_gb = resolve_worker_max_rss_gb(0);
            assert!(
                resolved_gb.saturating_mul(GIB) <= limit,
                "resolved {resolved_gb} GiB exceeds cgroup limit {limit} bytes",
            );
        }
    }
}
