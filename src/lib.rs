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
pub mod analyzer;
pub mod bench_hopper;
pub mod crash_dump;
pub mod explain;
pub mod features;
pub mod inflight;
pub mod interpret;
pub mod model;
pub mod model_update;
pub mod models_repo;
pub mod output;
pub mod ps;
pub mod engine;
pub mod server;
pub mod sys;
pub mod thread_dump;
pub mod tools;
pub mod traits_repo;
pub mod update_check;
pub mod update_manifest;
pub mod validate;
pub mod worker;

pub use analyzer::Analyzer;
pub use model::Classification;
pub use model::Thresholds;
pub use engine::{DisplayFilter, ScanConfig, ScanResult, ScanSummary};

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
}
