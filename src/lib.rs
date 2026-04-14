//! `litmus` classifies files as benign, suspicious, or hostile using
//! `cleave` static analysis plus an XGBoost model.
//!
//! The crate exposes a small public API centered around:
//! - [`ScanConfig`] for validated scan settings
//! - [`scan::run`] for recursive file and directory scans
//! - [`ps::run`] for process scans
//! - [`Classification`] and [`Thresholds`] for interpreting model output
//!
//! # Example
//! ```no_run
//! use litmus::{DisplayFilter, OutputFormat, ScanConfig, Thresholds};
//!
//! let config = ScanConfig::new(
//!     "/path/to/models",
//!     OutputFormat::Json,
//!     Some(Thresholds::default()),
//!     DisplayFilter::all(),
//!     4_000,
//!     None,
//!     false,
//! )?;
//!
//! let summary = litmus::scan::run(std::path::Path::new("/tmp/sample.exe"), &config)?;
//! println!("scanned {} file(s)", summary.total_files);
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod explain;
pub mod features;
pub mod model;
pub mod models_repo;
pub mod output;
pub mod ps;
pub mod scan;
pub mod server;
pub mod tools;
pub mod worker;

pub use model::Classification;
pub use model::Thresholds;
pub use scan::{DisplayFilter, ScanConfig, ScanResult, ScanSummary};

/// Output format for scan results.
#[derive(Debug, Clone, Copy, Default, PartialEq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable terminal output.
    #[default]
    Terminal,
    /// Newline-delimited JSON, one object per file.
    Json,
}
