//! litmus - ML-powered malware classification using cleave static analysis.

pub mod explain;
pub mod features;
pub mod model;
pub mod output;
pub mod ps;
pub mod scan;

pub use model::Classification;
pub use scan::{ScanConfig, ScanResult, ScanSummary};

/// Output format for scan results.
#[derive(Clone, Copy, Default, clap::ValueEnum)]
pub enum OutputFormat {
    #[default]
    Terminal,
    Json,
}
