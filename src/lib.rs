//! litmus - ML-powered malware classification using cleave static analysis.

pub mod explain;
pub mod features;
pub mod model;
pub mod models_repo;
pub mod output;
pub mod ps;
pub mod scan;
pub mod server;

pub use model::Classification;
pub use scan::{ScanConfig, ScanResult, ScanSummary};

/// Output format for scan results.
#[derive(Debug, Clone, Copy, Default, PartialEq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable terminal output.
    #[default]
    Terminal,
    /// Newline-delimited JSON, one object per file.
    Json,
}
