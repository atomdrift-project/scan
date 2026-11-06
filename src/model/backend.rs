//! Running one ONNX graph: tract, or the fast tree evaluator when the graph
//! is a plain tree ensemble.

use std::path::Path;

use anyhow::{Context, Result};

use super::fast_tree::FastTreeBackend;

/// One trained route member: a serialized ONNX graph scored single-sample,
/// returning the positive-class probability. ONNX is the only model format
/// scan loads.
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
    /// Load one `.onnx` member.
    pub(super) fn load(path: &Path) -> Result<Self> {
        if path.extension().and_then(|e| e.to_str()) != Some("onnx") {
            anyhow::bail!(
                "unsupported model file {}; only .onnx is supported",
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
        // Read from an open file rather than by path: given a path, tract
        // resolves a tensor's `external_data` location against the model's
        // directory with no containment, so a downloaded model could read any
        // local file into its weights. Without a directory, it refuses.
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("opening ONNX model {}", path.display()))?;
        let mut model = tract_onnx::onnx()
            .model_for_read(&mut file)
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

    /// Score one vector of `n_features` (checked by [`Backend::predict`]).
    pub(super) fn predict(&self, features: &[f32]) -> Result<f32> {
        use tract_onnx::prelude::*;
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
/// Holds one or more trained models for a route: a single `model.onnx` is
/// `first` with no `rest`; a multi-seed route (`models/seed_NN.onnx`) loads
/// every seed, and `predict` returns the arithmetic mean of their scores.
///
/// Every member has the same `num_features`; a mismatch is rejected at load.
#[derive(Debug)]
pub(super) struct Backend {
    pub(super) first: OnnxModel,
    pub(super) rest: Vec<OnnxModel>,
}

impl Backend {
    pub(super) const fn num_features(&self) -> usize {
        self.first.n_features()
    }

    /// The members' mean score. The one length check on the scoring path:
    /// a caller's vector of the wrong length is an error here, not a panic
    /// in a member.
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
        // Summed serially, in member order: each member's call is
        // microseconds and runs once per route per embedded member, already
        // inside the per-file parallelism, so fanning it out costs more than
        // the calls. Serial order is also deterministic, which a parallel
        // reduce's association order is not.
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

    /// Number of trained members (1 for a single `model.onnx`), for the
    /// load-time log.
    pub(super) fn n_members(&self) -> usize {
        1 + self.rest.len()
    }
}
