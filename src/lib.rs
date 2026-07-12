//! `litmus` classifies files as benign, suspicious, or hostile using
//! `cleave` static analysis plus a gradient-boosted tree model (loaded as
//! ONNX via `tract`).
//!
//! The crate exposes a small public API centered around:
//! - [`ScanConfig`] for validated scan settings
//! - [`engine::run`] for recursive file and directory scans
//! - [`ps::run`] for process scans
//! - [`Classification`] and [`Thresholds`] for interpreting model output
//!
//! # Example
//! ```no_run
//! use scan::{DisplayFilter, OutputFormat, ScanConfig, Thresholds};
//!
//! let config = ScanConfig::new(
//!     "/path/to/models",
//!     OutputFormat::Json,
//!     Some(Thresholds::default()),
//!     DisplayFilter::all(),
//!     4_000,
//!     false,
//! )?;
//!
//! let summary = scan::engine::run(std::path::Path::new("/tmp/sample.exe"), &config)?;
//! println!("scanned {} file(s)", summary.total_files);
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod admission;
pub mod analysis_cache;
pub mod analyzer;
pub mod auto_update;
pub mod bench_hopper;
pub mod bloom;
pub mod bloom_build;
pub mod bloom_repo;
pub mod bloom_update;
pub mod cache_cleanup;
pub mod crash_dump;
pub mod deptree;
pub mod engine;
pub mod explain;
pub mod features;
pub mod fetch;
pub mod inflight;
pub mod interpret;
pub mod model;
pub mod model_update;
pub mod models_repo;
pub mod output;
pub mod pkg;
pub mod provenance;
pub mod ps;
pub mod server;
pub mod sys;
pub mod thread_dump;
pub mod tools;
pub mod traits_repo;
pub mod update_check;
pub mod update_manifest;
pub mod upload;
pub mod validate;
pub mod worker;

pub use analyzer::Analyzer;
pub use engine::{DisplayFilter, ScanConfig, ScanResult, ScanSummary};
pub use model::Classification;
pub use model::Thresholds;

/// Convert a [`std::time::Duration`] to milliseconds as `u64`, saturating at [`u64::MAX`].
///
/// Avoids the `u128 as u64` truncating cast that `as_millis()` requires.
/// In practice elapsed durations are always tiny (seconds to minutes), so
/// saturation never fires — this is purely for lint-cleanliness.
pub(crate) fn duration_ms(d: std::time::Duration) -> u64 {
    d.as_secs()
        .saturating_mul(1_000)
        .saturating_add(u64::from(d.subsec_millis()))
}

/// The 1-minute system load average, or `None` on unsupported platforms.
pub(crate) fn system_load_avg() -> Option<f64> {
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "openbsd"
    ))]
    {
        let mut avg: [libc::c_double; 1] = [0.0];
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

/// First JSON value present under any of `names`, in order. Used throughout
/// report parsing to accept both cleave's verbose and compact field spellings.
pub(crate) fn json_alias<'a>(
    value: &'a serde_json::Value,
    names: &[&str],
) -> Option<&'a serde_json::Value> {
    names.iter().find_map(|name| value.get(*name))
}

/// [`json_alias`] resolved to an array.
pub(crate) fn json_alias_array<'a>(
    value: &'a serde_json::Value,
    names: &[&str],
) -> Option<&'a Vec<serde_json::Value>> {
    json_alias(value, names).and_then(serde_json::Value::as_array)
}

/// [`json_alias`] resolved to a string.
pub(crate) fn json_alias_str<'a>(value: &'a serde_json::Value, names: &[&str]) -> Option<&'a str> {
    json_alias(value, names).and_then(serde_json::Value::as_str)
}

/// Output format for scan results.
#[derive(Debug, Clone, Copy, Default, PartialEq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable terminal output.
    #[default]
    Terminal,
    /// Newline-delimited JSON, one object per file.
    Json,
    /// Compact, context-centric text for feeding a local LLM: a litmus verdict
    /// line (gate, confidence, matched FP level) followed by cleave's annotated
    /// context. See [`crate::output::format_tiny`].
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
    /// long-lived workers, where each job must be analyzed on its own merits.
    Slow,
}
