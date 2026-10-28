//! Atomdrift Scan classifies files as benign, suspicious, or hostile using
//! `cleave` static analysis plus an ensemble of ONNX models (run via `tract`).
//!
//! Library users want [`Analyzer`], or [`ScanConfig`] with [`engine::run`]:
//! - [`Analyzer`] scans bytes or a file and returns a [`ScanResult`].
//! - [`engine::run`] scans files and directories under a [`ScanConfig`].
//! - [`Classification`] and [`Thresholds`] interpret the model's output.
//!
//! The other public modules exist for the `atomscan` binary and the
//! integration tests, and change with them.
//!
//! # Example
//! ```no_run
//! use scan::{DisplayFilter, OutputFormat, ScanConfig, Thresholds};
//!
//! let config = ScanConfig::new("/path/to/models", OutputFormat::Json, Some(Thresholds::default()))?
//!     .with_filter(DisplayFilter::all());
//!
//! let summary = scan::engine::run(std::path::Path::new("/tmp/sample.exe"), &config)?;
//! println!("scanned {} file(s)", summary.total_files);
//! # Ok::<(), anyhow::Error>(())
//! ```

pub(crate) mod analysis_cache;
pub(crate) mod analyzer;
pub mod auto_update;
pub mod bloom_build;
pub mod bloom_repo;
pub mod bloom_update;
pub(crate) mod bundle;
pub mod cache_cleanup;
mod civil;
pub mod cli;
pub(crate) mod deptree;
mod digest;
pub mod engine;
pub mod env;
pub mod features;
pub mod fetch;
pub mod heap_profile;
pub mod hopper;
pub(crate) mod hosts;
pub mod interpret;
mod interrupt;
pub mod memory;
pub mod model;
pub mod model_update;
pub mod models_repo;
pub mod nsrl_repo;
pub mod output;
pub mod pkg;
pub mod provenance;
pub mod ps;
pub mod remote;
pub mod rules;
pub mod runtime;
pub mod server;
pub mod sys;
pub mod tools;
pub mod traits_repo;
pub mod update_check;
pub(crate) mod update_manifest;
pub mod upload;
pub mod validate;

// Plumbing for the fleet daemon built on this crate. Public so it can link
// against it, hidden because it is not a supported API and changes with it.
#[doc(hidden)]
pub mod analysis;
#[doc(hidden)]
pub mod corpus_precheck;
#[doc(hidden)]
pub mod crash_dump;
#[doc(hidden)]
pub mod explain;
#[doc(hidden)]
pub mod lookup;
#[doc(hidden)]
pub mod thread_dump;
#[doc(hidden)]
pub mod thread_priority;

pub use analyzer::Analyzer;
pub use engine::{DisplayFilter, ScanConfig, ScanResult, ScanSummary};
pub use model::Classification;
pub use model::Thresholds;
pub use rules::refresh_rules_at_startup;

/// Archive passwords kept unique and redacted from debug output.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct ArchivePasswords(std::sync::Arc<[String]>);

impl ArchivePasswords {
    /// Borrow the passwords in command-line order.
    #[must_use]
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }
}

impl From<Vec<String>> for ArchivePasswords {
    fn from(passwords: Vec<String>) -> Self {
        let mut unique = Vec::with_capacity(passwords.len());
        for password in passwords {
            if !unique.contains(&password) {
                unique.push(password);
            }
        }
        Self(unique.into())
    }
}

impl AsRef<[String]> for ArchivePasswords {
    fn as_ref(&self) -> &[String] {
        self.as_slice()
    }
}

impl std::fmt::Debug for ArchivePasswords {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArchivePasswords")
            .field("count", &self.0.len())
            .finish()
    }
}

/// Convert a [`std::time::Duration`] to milliseconds as `u64`, saturating at [`u64::MAX`].
///
/// Avoids the `u128 as u64` truncating cast that `as_millis()` requires.
#[doc(hidden)]
#[must_use]
pub fn duration_ms(d: std::time::Duration) -> u64 {
    d.as_secs()
        .saturating_mul(1_000)
        .saturating_add(u64::from(d.subsec_millis()))
}

/// The 1-minute system load average, or `None` on unsupported platforms.
#[doc(hidden)]
#[must_use]
pub fn system_load_avg() -> Option<f64> {
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "openbsd"
    ))]
    {
        let mut avg: [libc::c_double; 1] = [0.0];
        // SAFETY: `avg` is a live buffer of exactly the one element requested.
        let ret = unsafe { libc::getloadavg(avg.as_mut_ptr(), 1) };
        if ret == 1 { Some(avg[0]) } else { None }
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "openbsd"
    )))]
    {
        None
    }
}

/// Output format for scan results.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable terminal output.
    #[default]
    Terminal,
    /// Newline-delimited JSON, one object per file.
    Json,
    /// Compact, context-centric text for feeding a local LLM: a scan verdict
    /// line (gate, confidence, matched FP level) followed by cleave's annotated
    /// context. See `crate::engine::write_tiny`.
    Tiny,
    /// The LLM payload: byte-for-byte the user message a live `--interpret`
    /// query sends — cleave's sanitized tiny render, finding annotations
    /// included — without the system prompt (which hedges the descriptions as
    /// fallible interpretations; frame them likewise downstream). Binary
    /// windows carry xxd-style offset gutters for addressing. Emits every
    /// scanned file regardless of `--show`. Rendered locally — no LLM is
    /// contacted, and no verdict line is added.
    Interpret,
}

/// How aggressively a scan consults the local known-good / known-bad bloom
/// filters before doing expensive work. See [`crate::bloom_repo`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    /// Bloom matching only: known-good is skipped, known-bad is flagged, and
    /// anything in neither set is left unscanned. Fastest, least thorough.
    Fast,
    /// Bloom filters short-circuit known-good (skip) and known-bad (flag);
    /// everything else gets a full scan. The default.
    #[default]
    Balanced,
    /// No bloom lookups — every artifact is fully scanned. Always used by
    /// `serve`, where each request must be analyzed on its own merits.
    Slow,
}

/// `<platform data directory>/atomdrift/scan`, where the models and filters
/// are installed.
///
/// # Errors
/// When the platform reports no data directory: a user with no home. The
/// working directory is no substitute — a scan run inside an untrusted
/// checkout would load whatever models and filters that checkout supplied.
pub fn data_dir() -> anyhow::Result<std::path::PathBuf> {
    use anyhow::Context as _;
    dirs::data_dir()
        .map(|dir| dir.join("atomdrift").join("scan"))
        .context("this user has no data directory (is HOME set?)")
}

/// The root every scan cache lives under, settled once per process.
static CACHE_DIR: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();

/// Where scan's verdict index and analysis caches live:
/// `<platform cache directory>/atomdrift/scan` unless [`set_cache_dir`] chose
/// otherwise. `None` when the platform has none, and every cache then stays
/// off.
///
/// A unit-test build gets a private temporary directory instead: its tests
/// write verdicts, and must never write them into a developer's real index.
#[must_use]
pub fn cache_dir() -> Option<std::path::PathBuf> {
    CACHE_DIR
        .get_or_init(|| {
            #[cfg(test)]
            {
                Some(tempfile::tempdir().expect("tempdir").keep())
            }
            #[cfg(not(test))]
            {
                Some(dirs::cache_dir()?.join("atomdrift").join("scan"))
            }
        })
        .clone()
}

/// Keep every scan cache under `dir` for the rest of the process. Opening the
/// verdict index prunes the versions it does not use, so a test harness calls
/// this before anything else: a test run must never touch a real index.
///
/// # Errors
/// When the root is already settled, by an earlier call or an opened cache.
pub fn set_cache_dir(dir: std::path::PathBuf) -> anyhow::Result<()> {
    CACHE_DIR.set(Some(dir)).map_err(|rejected| {
        anyhow::anyhow!("the cache directory is already settled; not {rejected:?}")
    })
}

/// Stack size for every analysis thread pool, in MiB. cleave's archive
/// analysis is nested-parallel and deep; 256 MB is what the global rayon pool
/// has always used, and every analysis pool must match it or a member that
/// recurses fine on one pool overflows on another.
pub const RAYON_STACK_MB: usize = 256;

#[cfg(test)]
mod archive_password_tests {
    use super::ArchivePasswords;

    #[test]
    fn passwords_are_unique_and_debug_is_redacted() {
        let passwords = ArchivePasswords::from(vec![
            "secret".to_string(),
            "second".to_string(),
            "secret".to_string(),
        ]);

        assert_eq!(passwords.as_slice(), ["secret", "second"]);
        let debug = format!("{passwords:?}");
        assert!(debug.contains("count: 2"));
        assert!(!debug.contains("secret"));
        assert!(!debug.contains("second"));
    }
}
