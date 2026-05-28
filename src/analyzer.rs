//! High-level scanning API that bundles the model, feature extractor,
//! SHAP table, and runtime config so library consumers don't have to wire
//! them together by hand.
//!
//! This is the recommended entry point for embedding litmus in other
//! processes (proxies, daemons, fuzz harnesses). The lower-level
//! [`Model`](crate::model::Model), [`ExtractContext`](crate::features::ExtractContext),
//! [`ShapImportance`](crate::explain::ShapImportance), and
//! [`ScanConfig`](crate::ScanConfig) types remain available for callers
//! that need finer control or want to share components across multiple
//! analyzers.
//!
//! # Example
//! ```no_run
//! use litmus::Analyzer;
//!
//! let analyzer = Analyzer::load("/path/to/models")?;
//! let result = analyzer.scan_bytes(b"#!/bin/sh\necho hi".to_vec(), "install.sh")?;
//! println!("{:?} prob={:.3}", result.classification, result.probability);
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::explain::ShapImportance;
use crate::features::ExtractContext;
use crate::model::{Model, Thresholds};
use crate::scan::{self, ScanResult, ScanSummary};
use crate::{DisplayFilter, OutputFormat, ScanConfig};

/// Bundled analyzer holding everything needed to classify a payload.
///
/// Construct once with [`Analyzer::load`] (which loads model artifacts and
/// warms cleave's shared resources), then reuse across many `scan_*` calls.
/// The struct is `Send + Sync` and cheap to share via `Arc`.
#[derive(Debug)]
pub struct Analyzer {
    model_dir: PathBuf,
    model: Model,
    ctx: ExtractContext,
    shap: Option<ShapImportance>,
    config: ScanConfig,
}

impl Analyzer {
    /// Load model + feature spec + SHAP table from `model_dir` and prepare
    /// for repeated scans.
    ///
    /// Also runs [`cleave::prefetch_shared_resources`] before returning —
    /// this satisfies cleave's "warm me on a non-rayon thread first"
    /// requirement so downstream callers don't have to think about it. The
    /// call is idempotent; future re-invocations short-circuit.
    ///
    /// # Errors
    /// Propagates `Model::load` failures (missing or malformed `model.json`
    /// / `feature_spec.json`). SHAP is optional — missing `shap.json` is
    /// not an error.
    pub fn load(model_dir: impl Into<PathBuf>) -> Result<Self> {
        let model_dir = model_dir.into();
        let model = Model::load(&model_dir, None)?;
        let ctx = ExtractContext::new(model.spec());
        let shap = ShapImportance::load(&model_dir).ok();
        // Library callers always get the full cleave report attached to the
        // ScanResult (`OutputFormat::Json`); humans calling the CLI go
        // through `scan::run` directly, which builds its own config.
        let config = ScanConfig::new(
            model_dir.clone(),
            OutputFormat::Json,
            None,
            DisplayFilter::alerts_only(),
            4_000,
            false,
        )?;
        // Warm cleave's globals from this (non-rayon) thread so the first
        // analysis cannot race into a deadlock against rayon workers.
        cleave::prefetch_shared_resources(true);
        Ok(Self {
            model_dir,
            model,
            ctx,
            shap,
            config,
        })
    }

    /// Override the suspicious/hostile decision thresholds.
    ///
    /// # Errors
    /// Returns an error if the supplied thresholds are invalid (suspicious
    /// must be ≤ hostile, both in `[0, 1]`).
    pub fn with_thresholds(mut self, thresholds: Thresholds) -> Result<Self> {
        self.config = self.rebuild_config(Some(thresholds), self.config.slow_rule_ms())?;
        Ok(self)
    }

    /// Adjust the slow-rule advisory threshold in milliseconds.
    ///
    /// # Errors
    /// Returns an error only if reconstructing the internal `ScanConfig`
    /// rejects the existing thresholds (it does not, in practice — kept
    /// for forward compatibility).
    pub fn with_slow_rule_ms(mut self, ms: u64) -> Result<Self> {
        self.config = self.rebuild_config(self.config.thresholds(), ms)?;
        Ok(self)
    }

    /// Attach the FPR severity level (1..=9) that produced the resolved
    /// thresholds. Surfaces as `ml.level` in the JSON envelope. Pass `None` to
    /// indicate manual thresholds with no level metadata.
    #[must_use]
    pub fn with_level(mut self, level: Option<u8>) -> Self {
        self.config = self.config.with_level(level);
        self
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
        scan::scan_bytes(
            data,
            filename,
            &self.model,
            &self.ctx,
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
        scan::run(path, &self.config)
    }

    fn rebuild_config(
        &self,
        thresholds: Option<Thresholds>,
        slow_rule_ms: u64,
    ) -> Result<ScanConfig> {
        Ok(ScanConfig::new(
            self.model_dir.clone(),
            self.config.format(),
            thresholds,
            self.config.filter(),
            slow_rule_ms,
            self.config.extra(),
        )?
        .with_level(self.config.level()))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// `Analyzer::load` needs real model artifacts on disk, so this test is
    /// `#[ignore]` by default — run with `LITMUS_MODELS_DIR=... cargo test
    /// -- --ignored analyzer_loads_and_scans` to exercise it.
    #[test]
    #[ignore]
    fn analyzer_loads_and_scans() {
        let dir = std::env::var("LITMUS_MODELS_DIR")
            .expect("set LITMUS_MODELS_DIR to a populated model directory");
        let analyzer = Analyzer::load(dir).expect("load");
        let r = analyzer
            .scan_bytes(b"#!/bin/sh\necho hi\n".to_vec(), "install.sh")
            .expect("scan");
        // We don't assert a specific verdict — model-correctness is litmus's
        // concern, not the analyzer scaffolding's.
        assert!(r.probability.is_finite());
    }
}
