//! Running one ONNX graph: tract, or the fast tree evaluator when the graph
//! is a plain tree ensemble; and isotonic calibration of its scores.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// One trained route member: a serialized ONNX graph scored single-sample,
/// returning the positive-class probability. Collimator's `.onnx` route
/// artifacts are the only model format scan loads.
///
/// The graph is `onnxmltools.convert_lightgbm`'s standard output:
/// inputs `("input", float32, [N, n_features])`, outputs
/// `[label (int64, [N]), probabilities (float32, [N, 2])]`. We
/// ignore the label output and use column 1 of probabilities.
///
/// A plain tree ensemble is walked directly ([`FastTreeBackend`]); any other
/// graph falls back to tract.
#[derive(Debug)]
pub(super) enum OnnxModel {
    Fast(FastTreeBackend),
    Tract(TractOnnxBackend),
}

pub(super) struct TractOnnxBackend {
    /// Pre-optimized tract plan. Stored as a runnable model so
    /// `.run` calls don't re-optimize each prediction.
    pub(super) plan: tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>,
    pub(super) n_features: usize,
}

impl std::fmt::Debug for TractOnnxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TractOnnxBackend")
            .field("n_features", &self.n_features)
            .finish_non_exhaustive()
    }
}

impl OnnxModel {
    /// Load one `.onnx` member. Only ONNX is supported — the native LightGBM
    /// (`.txt`) and XGBoost (`.json`) loaders were retired.
    pub(super) fn load(path: &Path) -> Result<Self> {
        if path.extension().and_then(|e| e.to_str()) != Some("onnx") {
            anyhow::bail!(
                "unsupported model file {}; only .onnx is supported \
                 (the LightGBM/XGBoost loaders were removed)",
                path.display(),
            );
        }
        match FastTreeBackend::load(path) {
            Ok(fast) => {
                tracing::debug!(
                    path = %path.display(),
                    features = fast.n_features,
                    trees = fast.trees.len(),
                    "loaded fast ONNX tree ensemble"
                );
                return Ok(Self::Fast(fast));
            }
            Err(error) => {
                tracing::debug!(
                    path = %path.display(),
                    error = %error,
                    "fast ONNX tree loader did not accept model; falling back to tract"
                );
            }
        }
        Ok(Self::Tract(TractOnnxBackend::load(path)?))
    }

    pub(super) fn predict(&self, features: &[f32]) -> Result<f32> {
        match self {
            Self::Fast(fast) => fast.predict(features),
            Self::Tract(tract) => tract.predict(features),
        }
    }

    pub(super) const fn n_features(&self) -> usize {
        match self {
            Self::Fast(fast) => fast.n_features,
            Self::Tract(tract) => tract.n_features,
        }
    }
}

impl TractOnnxBackend {
    pub(super) fn load(path: &Path) -> Result<Self> {
        use tract_onnx::prelude::*;
        // Parse the ONNX graph first to recover n_features from the
        // (dynamic-batch) input fact.
        let mut model = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("parsing ONNX model {}", path.display()))?;
        let input_fact_dyn = model
            .input_fact(0)
            .with_context(|| format!("reading input fact from {}", path.display()))?
            .clone();
        // input fact's shape is a ShapeFactoid with per-dim factoids.
        // batch dim is unknown (the converter set it dynamic); the
        // feature dim is concrete (the converter pinned n_features
        // explicitly). Resolve the concrete feature dim via the
        // Factoid::concretize trait method, then to_usize.
        use tract_onnx::tract_hir::infer::Factoid;
        use tract_onnx::tract_hir::internal::DimLike;
        let n_features = input_fact_dyn
            .shape
            .dim(1)
            .ok_or_else(|| {
                anyhow::anyhow!("ONNX input has no feature dimension in {}", path.display())
            })?
            .concretize()
            .ok_or_else(|| anyhow::anyhow!("non-concrete feature dim in {}", path.display()))?
            .to_usize()
            .with_context(|| format!("feature dim not usize in {}", path.display()))?;
        // Pin batch dim to 1 so tract can optimize the graph.
        // Scan infers one file at a time; multi-seed
        // ensembling averages across seed members, not across a
        // batch dimension. If we ever want batched scoring, swap
        // this for a symbolic dim via tract_pulse.
        let pinned_fact = InferenceFact::dt_shape(f32::datum_type(), tvec!(1, n_features));
        model = model
            .with_input_fact(0, pinned_fact)
            .with_context(|| format!("pinning batch dim for {}", path.display()))?;
        let plan = model
            .into_optimized()
            .with_context(|| format!("optimizing ONNX model {}", path.display()))?
            .into_runnable()
            .with_context(|| format!("making ONNX model runnable {}", path.display()))?;
        Ok(Self { plan, n_features })
    }

    pub(super) fn predict(&self, features: &[f32]) -> Result<f32> {
        use tract_onnx::prelude::*;
        if features.len() != self.n_features {
            anyhow::bail!(
                "feature vector length {} != ONNX expected {}",
                features.len(),
                self.n_features
            );
        }
        // Build a (1, n_features) f32 tensor. We copy because tract
        // takes owned input. For single-sample inference at scan time
        // this is a single allocation per call; perfectly cheap.
        let input: Tensor =
            tract_ndarray::Array2::from_shape_vec((1, self.n_features), features.to_vec())
                .context("building ONNX input tensor")?
                .into();
        let outputs = self
            .plan
            .run(tvec!(input.into()))
            .context("ONNX inference run failed")?;
        // onnxmltools' LightGBM classifier exports outputs in
        // [label, probabilities] order. probabilities is (1, 2) where
        // column 1 is the positive-class probability — matches the
        // contract the rest of model.rs expects from .predict().
        let probs = outputs
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("ONNX missing probabilities output"))?;
        let view = probs
            .to_array_view::<f32>()
            .context("ONNX probabilities output not f32")?;
        let probs_slice = view
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("ONNX probabilities not contiguous"))?;
        if probs_slice.len() != 2 {
            anyhow::bail!(
                "ONNX probabilities length {} != 2 (expected binary [neg, pos])",
                probs_slice.len()
            );
        }
        Ok(probs_slice[1])
    }
}

/// Inference backend powering a loaded [`Model`].
///
/// Holds one or more trained models for a route. Single-model bundles (the
/// single-bundle layout, `model.onnx` directly under the bundle dir) load
/// `models = [one_model]`. Multi-seed bundles store `models/seed_NN.onnx` and
/// load all of them; `predict` returns the arithmetic mean of every member's
/// score, which is the variance-reducing equivalent of training K seeds and
/// ensembling them at inference time.
///
/// All members of `models` are required to have the same `num_features` —
/// mismatches are rejected at load time.
#[derive(Debug)]
pub(super) struct Backend {
    pub(super) first: OnnxModel,
    pub(super) rest: Vec<OnnxModel>,
}

impl Backend {
    pub(super) const fn num_features(&self) -> usize {
        self.first.n_features()
    }

    pub(super) fn predict(&self, features: &[f32]) -> Result<f32> {
        if features.len() != self.num_features() {
            anyhow::bail!(
                "feature vector length {} != ONNX expected {}",
                features.len(),
                self.num_features()
            );
        }
        // Average member predictions in f64 to keep the rounding noise-floor
        // below f32 precision even for K up to a few dozen seeds. The K=1
        // path is mathematically identical to a direct `models[0].predict`.
        //
        // Summed serially, in member order. A rayon `try_reduce` over the
        // members is the wrong shape for the work: a bundle carries 3 seeds, each seed's ONNX call is
        // microseconds, and this runs once per route per *embedded member* —
        // tens of thousands of times in a directory scan, nested inside the
        // per-member `par_iter` which is itself nested inside the per-path
        // pool. Fanning 3 microsecond tasks out at that depth costs more in
        // job/latch/epoch traffic than the calls themselves, and the churn is
        // charged to every rayon worker in the pool. Serial order is also
        // deterministic, which a parallel reduce's association order is not.
        let mut sum = f64::from(self.first.predict(features)?);
        for m in &self.rest {
            sum += f64::from(m.predict(features)?);
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "members emit f32; f64 only keeps the ensemble sum stable"
        )]
        let avg = (sum / self.n_members() as f64) as f32;
        Ok(avg)
    }

    /// Number of trained members (1 = legacy single-model bundle, ≥2 =
    /// multi-seed). Exposed mainly so the load-time `tracing::info!` line
    /// can advertise it.
    pub(super) fn n_members(&self) -> usize {
        1 + self.rest.len()
    }
}

/// Per-route isotonic calibrator persisted by collimator's
/// `azoth_calibrate_ensemble.py` as `calibrator.json` in the route directory.
///
/// Maps a route's raw model probability to a calibrated probability on
/// `[0, 1]`. The calibrator is fit on the deployment-time calibration corpus
/// so the deployed score matches the empirical malware fraction at any score
/// quantile.
///
/// Storage format (`azoth.calibrator.isotonic.v1`):
/// - `x`: ascending raw-probability breakpoints
/// - `y`: monotone-non-decreasing calibrated probabilities at each breakpoint
/// - `out_of_bounds`: `"clip"` — values outside `[x[0], x[-1]]` clamp to the
///   closest endpoint's `y`.
///
/// Apply: linear interpolation between adjacent breakpoints. Linear interp on
/// monotone breakpoints preserves the ranking, so AUC is unchanged — only the
/// probability *scale* shifts to match the empirical observation.
///
/// Backward compat: when `calibrator.json` is absent, raw scores pass through
/// unchanged. Bundles built before this change continue to load.
#[derive(Debug, Clone)]
pub(super) struct IsotonicCalibrator {
    /// Sorted-ascending raw probability breakpoints.
    pub(super) x: Vec<f32>,
    /// Monotone-non-decreasing calibrated probabilities; `y[i]` is the value
    /// at `x[i]`.
    pub(super) y: Vec<f32>,
}

impl IsotonicCalibrator {
    /// Read `<bundle_dir>/calibrator.json` if it exists. Returns Ok(None) if
    /// the file is missing (intentional — pre-calibrator bundles are still
    /// loadable). Errors when the file exists but is unreadable or invalid.
    pub(super) fn load_optional(bundle_dir: &Path) -> Result<Option<Self>> {
        #[derive(serde::Deserialize)]
        struct Raw {
            schema: Option<String>,
            x: Vec<f32>,
            y: Vec<f32>,
        }
        let path = bundle_dir.join("calibrator.json");
        let Some(raw) = read_json::<Raw>(&path)? else {
            return Ok(None);
        };
        if let Some(schema) = raw.schema.as_deref() {
            // Hard-fail on unrecognized future versions to avoid silently
            // applying a calibrator we don't understand.
            if schema != "azoth.calibrator.isotonic.v1" {
                anyhow::bail!(
                    "calibrator at {} has unsupported schema {schema}; this Atomdrift Scan build only supports azoth.calibrator.isotonic.v1",
                    path.display()
                );
            }
        }
        if raw.x.len() != raw.y.len() || raw.x.is_empty() {
            anyhow::bail!(
                "calibrator at {} is malformed: x.len={}, y.len={}",
                path.display(),
                raw.x.len(),
                raw.y.len()
            );
        }
        // Reject non-finite values up front: a single NaN or Inf in x or y
        // would propagate through every `apply()` call (NaN poisoning) — and
        // because thresholds are also calibrated at load time, every file
        // scored against this route would silently classify as Benign.
        for v in raw.x.iter().chain(raw.y.iter()) {
            if !v.is_finite() {
                anyhow::bail!(
                    "calibrator at {} contains a non-finite value (NaN/Inf)",
                    path.display(),
                );
            }
        }
        // Strict-ascending x: equal adjacent breakpoints would make the
        // linear-interpolation denominator zero and produce NaN in apply().
        for w in raw.x.windows(2) {
            if w[1] <= w[0] {
                anyhow::bail!(
                    "calibrator at {} is malformed: x is not strictly ascending ({} >= {})",
                    path.display(),
                    w[0],
                    w[1],
                );
            }
        }
        // y must be monotone-non-decreasing (the contract the calibrator
        // claims) AND in [0, 1] (a probability). If either fails, threshold
        // calibration could swap suspicious > hostile or land outside
        // valid-probability range — both are silent classifier corruption.
        for w in raw.y.windows(2) {
            if w[1] < w[0] {
                anyhow::bail!(
                    "calibrator at {} is malformed: y is not monotone non-decreasing ({} > {})",
                    path.display(),
                    w[0],
                    w[1],
                );
            }
        }
        for &v in &raw.y {
            if !(0.0..=1.0).contains(&v) {
                anyhow::bail!(
                    "calibrator at {} has y value {} outside [0, 1]",
                    path.display(),
                    v,
                );
            }
        }
        Ok(Some(Self { x: raw.x, y: raw.y }))
    }

    /// Apply the calibrator to a single raw probability. Linear interpolation
    /// between adjacent breakpoints; clip to endpoint `y` outside `[x[0], x[-1]]`.
    pub(super) fn apply(&self, raw: f32) -> f32 {
        if raw.is_nan() {
            return raw;
        }
        // Out-of-bounds clipping (matches sklearn IsotonicRegression(out_of_bounds="clip")).
        if raw <= self.x[0] {
            return self.y[0];
        }
        if raw >= self.x[self.x.len() - 1] {
            return self.y[self.y.len() - 1];
        }
        // Binary search for the interval containing `raw`.
        let i = match self
            .x
            .binary_search_by(|p| p.partial_cmp(&raw).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(idx) => return self.y[idx], // exact hit on a breakpoint
            Err(idx) => idx,
        };
        // raw is in (x[i-1], x[i]); linear interpolation.
        let x0 = self.x[i - 1];
        let x1 = self.x[i];
        let y0 = self.y[i - 1];
        let y1 = self.y[i];
        let t = (raw - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }
}
