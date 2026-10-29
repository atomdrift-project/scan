//! Model loading, thresholding, and inference.
//!
//! ## Model file format
//!
//! Only **`.onnx`** is supported — a portable inference graph emitted by
//! collimator's training pipeline (LightGBM via `onnxmltools.convert_lightgbm`)
//! or any other framework that exports to ONNX. Loaded via `tract` (pure-Rust,
//! no FFI). The native LightGBM (`.txt`) and XGBoost (`.json`) loaders were
//! retired now that collimator deploys ONNX-only bundles; any non-ONNX model
//! artifact is rejected at load time.
//!
//! ## Bundle layouts
//!
//! Two on-disk shapes are supported:
//!
//! ### Single-bundle (legacy / dev)
//!
//! ```text
//! <model_dir>/
//!   model.onnx
//!   feature_spec.json
//!   config.json
//!   evaluation.json
//! ```
//!
//! ### Ensemble (azoth) — see `~/azoth/DESIGN.md`
//!
//! ```text
//! <model_dir>/
//!   config.json                   ensemble-level config: route map + thresholds
//!   route_policies.json           optional per-filetype decision policies
//!   general/                      always required
//!     model.onnx | models/seed_*.onnx
//!     feature_spec.json
//!   filegroups/<group>/           optional, a filefacts group: native, scripts, archives, …
//!     model.onnx | models/seed_*.onnx
//!     feature_spec.json           may be absent → uses general's spec
//!   filetypes/<type>/             optional, e.g. elf, pe
//!     model.onnx | models/seed_*.onnx
//!     feature_spec.json           may be absent → uses general's spec
//! ```
//!
//! Detection is by presence of `general/` immediately under `<model_dir>`.
//!
//! ## Ensemble `config.json` schema (`azoth.routed_ensemble.v1`)
//!
//! Emitted by collimator's calibration pipeline. The top-level config is the
//! coarse compatibility source of thresholds for every route; specialist
//! subdirectories do not carry their own `config.json`. When
//! `route_policies.json` is present, it is the primary runtime decision
//! artifact.
//!
//! ```text
//! {
//!   "schema": "azoth.routed_ensemble.v1",
//!   "filetype_to_group": { "elf": "native", "pe": "native", "py": "scripts", … },
//!   "required_routes": ["general"],                    // optional
//!   "models": [
//!     {"route": "general",        "kind": "general",   "rows": …},
//!     {"route": "filegroups/native", "kind": "filegroup", "rows": …},
//!     {"route": "filetypes/elf",  "kind": "filetype",  "rows": …}
//!   ],
//!   "levels": [
//!     {
//!       "level": 50,
//!       "hostile": {
//!         "target_per_100M": 50,
//!         "hostile_per_million": 0.5,
//!         "budget": 8,
//!         "thresholds": { "general": 0.997, "filetypes/elf": 0.951, … },
//!         "tp": …, "fp": …, "recall": …
//!       }
//!     },
//!     …
//!   ],
//!   "calibration_snapshot_id": …,
//!   "score_table_hash": "…",
//!   "model_set_hash":   "…"
//! }
//! ```
//!
//! Route names use slash-separated paths matching the on-disk layout:
//! `"general"`, `"filegroups/<name>"`, `"filetypes/<name>"`.
//!
//! ## Ensemble `route_policies.json` schema (`azoth.route_policy_search.v1`)
//!
//! This optional artifact records the calibrated decision policy per filetype
//! and level. Each severity has a `best.thresholds` map:
//!
//! ```text
//! {
//!   "routes": {
//!     "filetypes/elf": {
//!       "filetype": "elf",
//!       "levels": [{
//!         "level": 50,
//!         "hostile": {
//!           "best": {
//!             "policy": "specialist_primary_with_escape",
//!             "thresholds": {
//!               "filetypes/elf": 0.995,
//!               "general": 0.968
//!             }
//!           }
//!         }
//!       }]
//!     }
//!   }
//! }
//! ```
//!
//! At runtime scan scores the applicable loaded routes, then applies only
//! the route thresholds named by the filetype policy. A route absent from the
//! policy does not participate in that severity decision. If
//! `route_policies.json` is absent or has no policy for a filetype, scan
//! falls back to an OR over `config.json` route thresholds.
//!
//! ## Specialist feature-spec rule
//!
//! Specialists may carry their own `feature_spec.json`. It may differ from
//! `general/feature_spec.json`, but it must have the ABI version this scan
//! binary understands and it must match the specialist model's feature count.
//! At runtime each route extracts its own feature vector from the same cleave
//! report before scoring.
//!
//! ## ABI mismatch
//!
//! - `general/` ABI mismatch: fatal. Refuse to start.
//! - Specialist ABI mismatch: warn, drop that specialist, continue.
//!
//! ## Optional `suspicious` in collimator JSON
//!
//! Collimator does not emit a `suspicious` field in any of the JSONs it
//! writes (`config.json`, `evaluation.json`, `route_policies.json`). Scan
//! derives it consumer-side as a **level-table lookup**: the suspicious
//! threshold is the hostile threshold at level
//! `min(max_grid_level, SUSPICIOUS_LEVEL_CEILING)` (the looser-budget row in
//! the same `levels[]` table). Manual `--threshold-hostile <val>` (no `-l`)
//! skips the derivation: the `Thresholds` struct carries
//! `suspicious == hostile`, so `classify` only ever returns Benign or Hostile.
//!
//! ## Malformed metadata
//!
//! A bundle file that is absent is simply not used. One that is present but
//! unreadable, unparseable, or internally inconsistent stops the load with an
//! error: a silently-ignored threshold table degrades every verdict.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::features::{
    EXPECTED_MODEL_ABI_VERSION, ExtractContext, FeatureSpec, ParsedReport, RawNeeds,
};

/// FALLBACK default severity level (per-100M-benigns scale; 50 = 50 FP/100M ≡
/// 0.5 FP/M). The operating point is resolved as: explicit CLI level → the
/// model-prescribed default from the bundle's config.json
/// (`model_default_level`, baked by collimator from its `DEFAULT_SEVERITY_LEVEL`)
/// → THIS const. So the tuning goal normally travels embedded in the model and
/// this is only used for older bundles whose config.json lacks
/// `default_severity_level`. Keep it equal to collimator's current default so
/// the fallback is sane; `make deploy` (collimator's azoth-deploy-final) fails
/// if the staged bundle's level, collimator's const, and this const disagree.
pub const DEFAULT_SEVERITY_LEVEL: u16 = 25;

/// The deploy tuning goal prescribed by THIS model's `config.json`
/// (`default_severity_level`), or `None` if the bundle predates the field.
///
/// Resolution order for the operating point is: explicit CLI level → this
/// model-prescribed default → the `DEFAULT_SEVERITY_LEVEL` const fallback. So a
/// bundle calibrated at L50 operates at L50 without any scan rebuild, and a
/// caller can still pin a different level explicitly. A malformed config.json
/// reads as `None` here; `Model::load` reports it.
#[must_use]
pub fn model_default_level(model_dir: &Path) -> Option<u16> {
    BundleConfig::load(model_dir).ok()??.default_severity_level
}

/// Loaded model plus the feature spec and thresholds used for inference.
///
/// For an ensemble bundle, `inner`/`spec`/`thresholds` are the *general*
/// route, and `routes` carries the optional specialists. For a single-bundle
/// deployment, `routes` is empty and the model behaves exactly like before.
#[derive(Debug)]
pub struct Model {
    inner: Backend,
    spec: FeatureSpec,
    /// Extraction tables for `spec`, built once at load.
    ctx: ExtractContext,
    thresholds: Thresholds,
    info: ModelInfo,
    routes: RouteSet,
    /// Optional isotonic calibrator for the general model. Applied after
    /// `inner.predict` everywhere we score with the general route. Mirrors
    /// the per-route calibrator on Route; backward compat = None.
    general_calibrator: Option<IsotonicCalibrator>,
    /// Active deploy level (FP per 100M benigns) that drives the verdict and
    /// the envelope level marker. `None` in manual-threshold mode (`--threshold-*`)
    /// and on single-bundle deployments with no level grid; the sweep is then
    /// bypassed and the level serializes as `null`.
    active_level: Option<u16>,
    /// General route's hostile threshold per level (ascending), for the
    /// no-policy verdict sweep. Empty on single-bundle / pre-grid deployments.
    general_grid: Vec<(u16, f32)>,
    /// Largest grid level; suspicious cap = `capped_suspicious_level(grid_max)`.
    grid_max: u16,
}

/// One route's score for a file, before it is split into the decision input
/// and the wire record.
struct Scored {
    route: RouteId,
    raw: f32,
    probability: f32,
    class: Classification,
}

impl Model {
    /// Load model artifacts from a directory containing `feature_spec.json`
    /// and `model.onnx` (or `models/seed_*.onnx`).
    ///
    /// The same directory may also contain optional metadata such as
    /// `shap_importance.json` and git history, but those are not required here.
    ///
    /// # Errors
    /// Returns an error if thresholds are invalid, required model artifacts are
    /// missing, any present metadata file is malformed, or the loaded feature
    /// spec does not match this build.
    ///
    /// Threshold resolution order:
    /// 1. Explicit `thresholds` argument (from CLI flags)
    /// 2. `config.json` in the model directory
    /// 3. Recommended thresholds from `evaluation.json`
    /// 4. Conservative fallback constants
    ///
    /// If explicit thresholds are provided *and* `evaluation.json` contains
    /// recommendations, a warning is emitted when they diverge significantly.
    pub fn load(
        model_dir: &Path,
        thresholds: Option<Thresholds>,
        active_level: Option<u16>,
    ) -> Result<Self> {
        let started = std::time::Instant::now();
        // Detect ensemble layout by the presence of `general/` immediately
        // under model_dir. Otherwise treat model_dir itself as a single bundle.
        let model = if model_dir.join("general").is_dir() {
            Self::load_ensemble(model_dir, thresholds, active_level)
        } else {
            Self::load_single_bundle(model_dir, thresholds, active_level)
        }?;
        tracing::info!(
            path = %model_dir.display(),
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            "model load complete",
        );
        Ok(model)
    }

    /// Load a single-bundle layout (legacy / dev). The model artifacts
    /// (`model.onnx`, `feature_spec.json`, `config.json`,
    /// `evaluation.json`) sit directly in `model_dir`. No specialists.
    fn load_single_bundle(
        model_dir: &Path,
        thresholds: Option<Thresholds>,
        active_level: Option<u16>,
    ) -> Result<Self> {
        let bundle = load_bundle(model_dir, thresholds, /* is_general = */ true)?;
        tracing::info!(
            n_members = bundle.backend.n_members(),
            features = bundle.spec.total_features(),
            model_abi_version = bundle.spec.abi_version(),
            threshold_suspicious = bundle.thresholds.suspicious,
            threshold_hostile = bundle.thresholds.hostile,
            threshold_source = bundle.threshold_source,
            spec_version = bundle.spec.version(),
            layout = "single-bundle",
            "model loaded",
        );
        // Single-bundle deployments carry no level grid, so the verdict sweep
        // has nothing to sweep: `decide` falls back to the threshold path and
        // the level serializes as `null`.
        Ok(Self::new(
            bundle,
            RouteSet::default(),
            active_level,
            Vec::new(),
            0,
        ))
    }

    /// Load an ensemble layout (`general/` + `filegroups/*` + `filetypes/*`).
    /// See module-level docs for the on-disk shape and config schema.
    fn load_ensemble(
        model_dir: &Path,
        thresholds: Option<Thresholds>,
        active_level: Option<u16>,
    ) -> Result<Self> {
        // Ensemble-level config.json carries the routing map and the per-route
        // thresholds for every level. We resolve thresholds from it before
        // loading any individual route bundle so each bundle gets the right
        // pre-resolved thresholds and skips its own (nonexistent) config.json.
        //
        // The diagnostic `by_filetype` policy and per-route thresholds are
        // pinned to DEFAULT_SEVERITY_LEVEL (not the active level) so the
        // `models[]` array and emitted `prob` are identical regardless of the
        // caller's `-l` — the JSON envelope stays cacheable across levels. The
        // active level enters only the final verdict derivation, via the
        // level-independent sweep over `policies.grid` / `general_grid`.
        // Pinned before anything is read: a bundle swapped in mid-load then
        // fails the pin, rather than its specialists being scored against
        // the thresholds read from the bundle it replaced.
        let pin = Arc::new(BundlePin::new(model_dir));
        let cfg = EnsembleConfig::new(
            BundleConfig::load(model_dir)?.unwrap_or_default(),
            DEFAULT_SEVERITY_LEVEL,
        );
        let mut names = RouteNames::default();
        for route in cfg.route_thresholds.keys() {
            names.intern(route);
        }
        let general_dir = model_dir.join("general");
        let general_thresholds =
            thresholds.or_else(|| cfg.route_thresholds.get("general").copied());

        // The policy search JSON includes large diagnostic tables. Parse it
        // while decoding the general model: neither operation depends on the
        // other. Specialist registration needs the policy route names, so it
        // follows both loads; it only reads small calibrator files.
        let specialist_start = std::time::Instant::now();
        let (policies, general) = rayon::join(
            || load_route_policies(model_dir, DEFAULT_SEVERITY_LEVEL, &mut names),
            || {
                load_bundle(
                    &general_dir,
                    general_thresholds,
                    /* is_general = */ true,
                )
                .with_context(|| format!("loading general route from {}", general_dir.display()))
            },
        );
        let mut policies = policies?;
        let general = general?;
        let (filegroups, filetypes) = rayon::join(
            || {
                load_specialists(
                    &model_dir.join("filegroups"),
                    &pin,
                    &cfg.route_thresholds,
                    &policies,
                    &names,
                    "filegroups",
                )
            },
            || {
                load_specialists(
                    &model_dir.join("filetypes"),
                    &pin,
                    &cfg.route_thresholds,
                    &policies,
                    &names,
                    "filetypes",
                )
            },
        );
        tracing::info!(
            filegroups = filegroups.len(),
            filetypes = filetypes.len(),
            elapsed_ms = specialist_start.elapsed().as_millis(),
            "prepared specialist routes",
        );

        let routes = RouteSet {
            filegroups,
            filetypes,
            filetype_to_filegroup: cfg.filetype_to_filegroup,
            policies: RoutePolicies::default(),
            names,
        };

        // Required routes from config: any listed name that didn't load is
        // fatal. "general" is implicitly required and was already loaded above.
        for required in &cfg.required_routes {
            match RouteKind::parse(required) {
                Some(RouteKind::General) => {}
                Some(RouteKind::Group(name)) => {
                    routes.filegroups.validate_route(name).with_context(|| {
                        format!(
                            "ensemble config marks filegroup {name:?} as required but it failed to load"
                        )
                    })?;
                }
                Some(RouteKind::Type(name)) => {
                    routes.filetypes.validate_route(name).with_context(|| {
                        format!(
                            "ensemble config marks filetype {name:?} as required but it failed to load"
                        )
                    })?;
                }
                None => anyhow::bail!(
                    "unknown required-route name {required:?} in ensemble config; \
                     expected `general`, `filegroups/<name>`, or `filetypes/<name>`"
                ),
            }
        }

        // The route_policies.json thresholds for the OR-of-routes decision
        // path were loaded in raw-score space, but the route scorers emit
        // calibrated probabilities (see `load_bundle`). Push each per-route
        // threshold through that route's calibrator so the policy compares
        // like with like. Isotonic monotonicity preserves the decision.
        calibrate_policy_thresholds(&mut policies, |route| {
            routes.calibrator(general.calibrator.as_ref(), route)
        });
        let routes = RouteSet { policies, ..routes };
        // The general grid comes from the same raw `levels[]` table, and the
        // fallback sweep compares it with calibrated probabilities too.
        let mut general_grid = cfg.general_grid;
        if let Some(cal) = general.calibrator.as_ref() {
            for (_, threshold) in &mut general_grid {
                *threshold = cal.apply(*threshold);
            }
        }

        tracing::info!(
            n_members = general.backend.n_members(),
            features = general.spec.total_features(),
            model_abi_version = general.spec.abi_version(),
            threshold_suspicious = general.thresholds.suspicious,
            threshold_hostile = general.thresholds.hostile,
            threshold_source = general.threshold_source,
            spec_version = general.spec.version(),
            filegroups = routes.filegroups.len(),
            filetypes = routes.filetypes.len(),
            layout = "ensemble",
            "model loaded",
        );

        Ok(Self::new(
            general,
            routes,
            active_level,
            general_grid,
            cfg.grid_max,
        ))
    }

    fn new(
        general: LoadedBundle,
        routes: RouteSet,
        active_level: Option<u16>,
        general_grid: Vec<(u16, f32)>,
        grid_max: u16,
    ) -> Self {
        let info = ModelInfo {
            version: general.spec.version(),
            abi_version: general.spec.abi_version(),
        };
        let ctx = ExtractContext::new(&general.spec);
        Self {
            inner: general.backend,
            spec: general.spec,
            ctx,
            thresholds: general.thresholds,
            info,
            routes,
            general_calibrator: general.calibrator,
            active_level,
            general_grid,
            grid_max,
        }
    }

    /// Force every calibrated specialist route to load and validate.
    ///
    /// Normal scanning initializes specialist ONNX graphs on demand from the
    /// observed file type. Validation and install gates call this explicitly so
    /// malformed specialist artifacts are caught even if the current fixture set
    /// never routes to them.
    ///
    /// # Errors
    /// Returns the first specialist that fails to load.
    pub fn validate_all_routes(&self) -> Result<()> {
        if self.routes.is_empty() {
            return Ok(());
        }
        let started = std::time::Instant::now();
        self.routes.validate_all()?;
        tracing::info!(
            filegroups = self.routes.filegroups.len(),
            filetypes = self.routes.filetypes.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "validated specialist routes",
        );
        Ok(())
    }

    /// Classify a cleave report the way a scan does: featurize every file,
    /// standardize, score each route the report's primary file type reaches,
    /// and decide.
    ///
    /// # Errors
    /// Returns an error if a consulted route's backend fails.
    pub fn predict_report(
        &self,
        report: &cleave::types::CompactReport,
    ) -> Result<(Decision, Vec<RouteScore>, Vec<SkippedRoute>)> {
        let needs = self.raw_needs(report.files.iter().map(|f| f.file_type.as_str()));
        let parsed = ParsedReport::from_compact_report(report, needs, None);
        let mut features = self.ctx.extract_from_parsed(&parsed);
        self.spec.standardize(&mut features);
        let file_type = report
            .files
            .first()
            .map_or("unknown", |f| f.file_type.as_str());
        self.predict_for_report_detailed(file_type, &features, &parsed)
    }

    /// Routed prediction from a parsed cleave report (a whole sample or one
    /// archive member), with per-route scores retained for JSON and `--extra`
    /// output.
    ///
    /// This is the production ensemble path. General is scored from the
    /// caller-provided standardized general feature vector; each specialist
    /// scores its own route-specific vector from the shared [`ParsedReport`].
    pub(crate) fn predict_for_report_detailed(
        &self,
        file_type: &str,
        general_features: &[f32],
        parsed: &ParsedReport,
    ) -> Result<(Decision, Vec<RouteScore>, Vec<SkippedRoute>)> {
        let policy = self.routes.policies.policy_for(file_type);
        let (route_probs, scores, mut skipped) =
            self.score_routes(file_type, policy, general_features, parsed)?;
        let decision = self.decide(file_type, policy, &route_probs, &mut skipped);
        Ok((decision, scores, skipped))
    }

    /// Score every applicable route for `file_type` and return the
    /// `(probabilities, RouteScore list, SkippedRoute list)` triple. The
    /// per-route `RouteScore.classification` uses the policy's per-route
    /// classifier when available, else the route's own thresholds.
    fn score_routes(
        &self,
        file_type: &str,
        policy: Option<&RoutePolicy>,
        general_features: &[f32],
        parsed: &ParsedReport,
    ) -> Result<(Vec<RouteProbability>, Vec<RouteScore>, Vec<SkippedRoute>)> {
        // Select specialists by ARCHIVE format, not compression: a `.tgz`
        // resolves to the `tar` container specialist instead of routing
        // general-only and surfacing `filetypes/tar` as an `unavailable` skip.
        let (type_key, group_key) = self.routes.specialist_keys(file_type);
        let group = group_key.and_then(|name| self.routes.filegroups.get(name));
        let filetype = self.routes.filetypes.get(&type_key);

        let classify = |route: RouteId, probability: f32, own: &Thresholds| {
            policy.map_or_else(
                || own.classify(probability),
                |policy| policy_route_class(policy, route, probability),
            )
        };
        let score_general = || -> Result<Scored> {
            let raw = self.inner.predict(general_features)?;
            let probability = self
                .general_calibrator
                .as_ref()
                .map_or(raw, |cal| cal.apply(raw));
            Ok(Scored {
                route: RouteId::GENERAL,
                raw,
                probability,
                class: classify(RouteId::GENERAL, probability, &self.thresholds),
            })
        };
        let score_specialist = |entry: Option<&(RouteId, Arc<Route>)>| -> Result<Option<Scored>> {
            let Some((id, route)) = entry else {
                return Ok(None);
            };
            let mut features = route.ctx.extract_from_parsed(parsed);
            route.spec.standardize(&mut features);
            let raw = route.backend.predict(&features)?;
            let probability = route.calibrator.as_ref().map_or(raw, |cal| cal.apply(raw));
            Ok(Some(Scored {
                route: *id,
                raw,
                probability,
                class: classify(*id, probability, &route.thresholds),
            }))
        };
        // Files (including embedded members) already score in parallel. These
        // three short route calls do not need another nested join: a rayon worker
        // waiting there can steal a long, unrelated archive task and delay the
        // otherwise completed file. Keep the same result and error ordering.
        let general = score_general();
        let group_score = score_specialist(group.as_ref());
        let filetype_score = score_specialist(filetype.as_ref());

        let mut route_probs = Vec::with_capacity(3);
        let mut scores = Vec::with_capacity(3);
        for scored in [Some(general?), group_score?, filetype_score?]
            .into_iter()
            .flatten()
        {
            route_probs.push(RouteProbability {
                route: scored.route,
                probability: scored.probability,
            });
            scores.push(RouteScore {
                model: compact_route_name(self.routes.names.name(scored.route)),
                probability: scored.probability,
                raw: scored.raw,
                classification: scored.class,
            });
        }

        let mut skipped = Vec::new();
        if group.is_none()
            && let Some(name) = group_key
            && self.routes.filegroups.skipped.contains(name)
        {
            skipped.push(SkippedRoute {
                model: format!("az/{name}"),
                reason: "uncalibrated",
            });
        }
        if filetype.is_none() && self.routes.filetypes.skipped.contains(type_key.as_ref()) {
            skipped.push(SkippedRoute {
                model: format!("az/{type_key}"),
                reason: "uncalibrated",
            });
        }

        Ok((route_probs, scores, skipped))
    }

    /// Pick the strongest [`Decision`] across the route scores using only the
    /// model's general thresholds (no per-filetype policy). Picks the highest
    /// class; on ties, the highest probability. Benign fallback uses the
    /// general route's prob against the suspicious cutoff.
    fn decide_from_scores(&self, scores: &[RouteProbability]) -> Decision {
        // General route is always first when scoring through predict_*_detailed.
        let general_prob = scores
            .iter()
            .find(|s| s.route == RouteId::GENERAL)
            .map_or(0.0, |s| s.probability);
        let mut best = self.thresholds.decide(general_prob);
        for score in scores {
            let candidate = self.thresholds.decide(score.probability);
            let better = match candidate.class.cmp(&best.class) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Equal => candidate.probability > best.probability,
                std::cmp::Ordering::Less => false,
            };
            if better {
                best = candidate;
            }
        }
        best
    }

    /// Pick the final [`Decision`] from per-route scores.
    ///
    /// In level mode the verdict comes from the level-independent sweep (see
    /// [`Self::decide_swept`]). In manual-threshold mode (`active_level` is
    /// `None`) it keeps the pre-level behaviour: the per-filetype policy at the
    /// default level when one exists, else the OR over the model's thresholds —
    /// with no level marker, since no level table applies.
    fn decide(
        &self,
        file_type: &str,
        policy: Option<&RoutePolicy>,
        route_probs: &[RouteProbability],
        skipped: &mut Vec<SkippedRoute>,
    ) -> Decision {
        // Diagnostic: note routes the default-level policy references but that
        // produced no score this run.
        if let Some(policy) = policy {
            for &(route, _) in policy
                .hostile
                .thresholds
                .iter()
                .chain(&policy.suspicious.thresholds)
            {
                if route_probs.iter().any(|score| score.route == route) {
                    continue;
                }
                // Once per route: both severities usually name the same ones,
                // and scoring may already have reported it skipped.
                let model = compact_route_name(self.routes.names.name(route));
                if !skipped.iter().any(|s| s.model == model) {
                    skipped.push(SkippedRoute {
                        model,
                        reason: "unavailable",
                    });
                }
            }
        }

        let Some(level) = self.active_level else {
            // Manual-threshold mode: pre-level semantics, level is null.
            return match policy {
                Some(policy) => policy_decide(policy, route_probs)
                    .unwrap_or_else(|| self.benign_fallback(route_probs, Level::Manual)),
                None => self.decide_from_scores(route_probs),
            };
        };

        self.decide_swept(file_type, route_probs, level)
    }

    /// Benign decision reporting the general route's probability against the
    /// model's suspicious cutoff (the band the score didn't reach), tagged with
    /// the given level marker.
    fn benign_fallback(&self, route_probs: &[RouteProbability], level_marker: Level) -> Decision {
        let general_prob = route_probs
            .iter()
            .find(|s| s.route == RouteId::GENERAL)
            .map_or(0.0, |s| s.probability);
        Decision {
            class: Classification::Benign,
            probability: general_prob,
            threshold: self.thresholds.suspicious,
            level: level_marker,
        }
    }

    /// Derive the verdict from the level-independent lowest-firing-level sweep.
    ///
    /// The swept level is computed over the full grid with no reference to the
    /// active deploy level, so it — and the entire serialized envelope — is
    /// identical regardless of the deploy `-l`, keeping the result
    /// cache-shareable. The active deploy level only positions the cutoffs:
    /// hostile when the swept level is <= the deploy level, suspicious when it
    /// is <= `capped_suspicious_level(grid_max)`, else benign. A file that
    /// fires at no grid level is [`Level::Clean`].
    fn decide_swept(
        &self,
        file_type: &str,
        route_probs: &[RouteProbability],
        level: u16,
    ) -> Decision {
        let swept = if let Some(grid) = self.routes.policies.grid_for(file_type) {
            sweep_policy_grid(grid, route_probs)
        } else if self.general_grid.is_empty() {
            // No grid to sweep (general-only bundle without a levels[] table):
            // fall back to the threshold path; the level stays null.
            return self.decide_from_scores(route_probs);
        } else {
            sweep_general_grid(&self.general_grid, route_probs)
        };

        let Some((fired_level, probability, threshold)) = swept else {
            return self.benign_fallback(route_probs, Level::Clean);
        };

        Decision {
            class: verdict_for_level(fired_level, level, self.grid_max),
            probability,
            threshold,
            level: Level::At(fired_level),
        }
    }

    /// Feature specification used to build input vectors.
    #[must_use]
    pub const fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    /// Extraction tables for [`Self::spec`], built once at load.
    #[must_use]
    pub const fn ctx(&self) -> &ExtractContext {
        &self.ctx
    }

    /// The optional raw inputs a report must have parsed for every route that
    /// will score it: the general route, and the specialists each of
    /// `file_types` selects. Specialists load here if they have not yet;
    /// scoring the same types would load them next anyway.
    pub(crate) fn raw_needs<'a>(&self, file_types: impl IntoIterator<Item = &'a str>) -> RawNeeds {
        let mut needs = self.ctx.raw_needs();
        let mut seen = HashSet::new();
        for file_type in file_types {
            if !seen.insert(file_type) {
                continue;
            }
            let (type_key, group_key) = self.routes.specialist_keys(file_type);
            let group = group_key.and_then(|name| self.routes.filegroups.get(name));
            let filetype = self.routes.filetypes.get(&type_key);
            for (_, route) in group.iter().chain(filetype.iter()) {
                needs = needs.union(route.ctx.raw_needs());
            }
        }
        needs
    }

    /// Largest calibrated grid level, which caps the suspicious ceiling (see
    /// `capped_suspicious_level`).
    #[must_use]
    pub(crate) const fn grid_max(&self) -> u16 {
        self.grid_max
    }

    /// Active deploy level (`-l`, FP per 100M benigns): the single hostile
    /// threshold — a file is hostile iff its firing level is `<=` this. `None`
    /// in manual-threshold mode and on single-bundle deployments with no grid.
    #[must_use]
    pub(crate) const fn active_level(&self) -> Option<u16> {
        self.active_level
    }

    /// Classification thresholds carried by this loaded model.
    #[must_use]
    pub const fn thresholds(&self) -> Thresholds {
        self.thresholds
    }

    /// Stable metadata describing the loaded model artifacts.
    #[must_use]
    pub const fn info(&self) -> &ModelInfo {
        &self.info
    }
}

mod backend;
mod fast_tree;
mod load;
mod policy;
mod verdict;

pub(crate) use self::policy::verdict_for_level;
pub use self::verdict::*;
#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use self::{backend::*, fast_tree::*, load::*, policy::*};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod fast_tree_tests;
