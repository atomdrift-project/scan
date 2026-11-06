//! Model loading, thresholding, and inference.
//!
//! ## Model file format
//!
//! Only **`.onnx`** is supported: the portable inference graph the training
//! pipeline exports (LightGBM through `onnxmltools.convert_lightgbm`). It is
//! loaded with `tract` (pure Rust, no FFI), or walked directly when it is a
//! plain tree ensemble. Any other model artifact is rejected at load.
//!
//! ## Bundle layout
//!
//! A bundle is a routed ensemble:
//!
//! ```text
//! <model_dir>/
//!   config.json                   route map + per-level threshold grid
//!   route_policies.json           optional per-filetype decision policies
//!   general/                      always required
//!     model.onnx | models/seed_*.onnx
//!     feature_spec.json
//!   filegroups/<group>/           optional, a filefacts group: native, scripts, archives, …
//!     model.onnx | models/seed_*.onnx
//!     feature_spec.json
//!   filetypes/<type>/             optional, e.g. elf, pe
//!     model.onnx | models/seed_*.onnx
//!     feature_spec.json
//! ```
//!
//! ## `config.json` schema (`azoth.routed_ensemble.v1`)
//!
//! The top-level config is the source of thresholds for every route;
//! specialist subdirectories carry no `config.json` of their own. When
//! `route_policies.json` is present, it is the primary runtime decision
//! artifact.
//!
//! ```text
//! {
//!   "schema": "azoth.routed_ensemble.v1",
//!   "filetype_to_group": { "elf": "native", "pe": "native", "py": "scripts", … },
//!   "required_routes": ["general"],                    // optional
//!   "default_severity_level": 25,                      // optional
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
//! `"general"`, `"filegroups/<name>"`, `"filetypes/<name>"`. The general
//! route must have a threshold at the bundle's default level, or explicit
//! thresholds must be given: a bundle with neither cannot classify and does
//! not load.
//!
//! ## `route_policies.json` schema (`azoth.route_policy_search.v1`)
//!
//! This optional artifact records the decision policy per filetype and
//! level. Each severity has a `best.thresholds` map or a learned `best.blend`:
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
//! ## Specialist feature specs
//!
//! Each route carries its own `feature_spec.json`, which may differ from
//! general's but must have an ABI this build reads and match its model's
//! feature count. Every route extracts its own feature vector from the same
//! parsed cleave report.
//!
//! A general route this build cannot load stops the load; a specialist that
//! fails is dropped with a warning and its files fall back to the other
//! routes.
//!
//! ## The suspicious band
//!
//! The JSON artifacts carry hostile thresholds only. The suspicious
//! threshold is the hostile threshold at level `suspicious_ceiling(level,
//! max_grid_level)` — `min(max_grid_level, max(500, 100 × level))` — or at
//! the loosest row below it when that is not a row of its own. Explicit
//! thresholds skip the derivation and are used as given.
//!
//! ## Malformed metadata
//!
//! An optional bundle file that is absent is simply not used. One that is
//! present but unreadable, unparseable, or internally inconsistent stops the
//! load with an error: a silently ignored threshold table degrades every
//! verdict.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::features::{ExtractContext, FeatureSpec, ParsedReport, RawNeeds};

use self::backend::Backend;
use self::load::{BundleConfig, BundlePin, LoadedBundle, Route, RouteSet, load_bundle};
use self::policy::{
    EnsembleConfig, RouteId, RouteNames, RoutePolicy, RouteProbability, compact_route_name,
    load_route_policies, policy_decide, policy_route_class, sweep_general_grid, sweep_policy_grid,
};

/// Default severity level (per-100M-benigns scale; 25 = 25 FP/100M).
///
/// The operating point is resolved as: an explicit level → the bundle's own
/// `default_severity_level` (see [`model_default_level`]) → this constant.
/// Keep it equal to the level the deployed model is calibrated for, so a
/// bundle that omits the field still operates where it was tuned.
pub const DEFAULT_SEVERITY_LEVEL: u16 = 25;

/// The default level prescribed by the bundle's `config.json`
/// (`default_severity_level`), or `None` when the bundle names none.
///
/// A bundle calibrated at L50 thus operates at L50 without a scan rebuild,
/// and a caller can still pin another level explicitly. An unreadable
/// config.json is logged and reads as `None` here; `Model::load` refuses it.
#[must_use]
pub fn model_default_level(model_dir: &Path) -> Option<u16> {
    match BundleConfig::load(model_dir) {
        Ok(config) => config?.default_severity_level,
        Err(error) => {
            tracing::warn!(
                path = %model_dir.display(),
                error = %format!("{error:#}"),
                "cannot read the model's default severity level",
            );
            None
        }
    }
}

/// A loaded model bundle: the general route, its feature spec and
/// thresholds, and the specialist routes.
#[derive(Debug)]
pub struct Model {
    inner: Backend,
    spec: FeatureSpec,
    /// Extraction tables for `spec`, built once at load.
    ctx: ExtractContext,
    thresholds: Thresholds,
    info: ModelInfo,
    routes: RouteSet,
    /// Active deploy level (pool-adjusted FP scale) that drives the verdict and
    /// the envelope level marker. `None` in manual-threshold mode
    /// (`--threshold-*`): the sweep is then bypassed and the level serializes
    /// as `null`.
    active_level: Option<u16>,
    /// General route's hostile threshold per level (ascending), for the
    /// no-policy verdict sweep.
    general_grid: Vec<(u16, f32)>,
    /// Largest grid level; caps `suspicious_ceiling(active, grid_max)`.
    grid_max: u16,
}

/// One route's score for a file, before it is split into the decision input
/// and the wire record.
struct Scored {
    route: RouteId,
    probability: f32,
    class: Classification,
}

impl Model {
    /// Load a routed model bundle (see the module docs for its layout).
    ///
    /// `thresholds` overrides the general route's thresholds from
    /// `config.json`; `active_level` is the deploy level, `None` for manual
    /// thresholds.
    ///
    /// # Errors
    /// Returns an error if thresholds are invalid or absent, required model
    /// artifacts are missing, any present metadata file is malformed, or a
    /// feature spec does not match this build.
    pub fn load(
        model_dir: &Path,
        thresholds: Option<Thresholds>,
        active_level: Option<u16>,
    ) -> Result<Self> {
        let started = Instant::now();
        anyhow::ensure!(
            model_dir.join("general").is_dir(),
            "model bundle is incomplete: no general/ route in {}. Run 'atomscan update-rules' to refresh the installed models.",
            model_dir.display(),
        );
        let model = Self::load_ensemble(model_dir, thresholds, active_level)?;
        tracing::info!(
            path = %model_dir.display(),
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            "model load complete",
        );
        Ok(model)
    }

    fn load_ensemble(
        model_dir: &Path,
        explicit: Option<Thresholds>,
        active_level: Option<u16>,
    ) -> Result<Self> {
        // Pinned before anything is read: a bundle swapped in mid-load then
        // fails the pin, rather than its specialists being scored against
        // the thresholds read from the bundle it replaced.
        let pin = Arc::new(BundlePin::new(model_dir));
        let config = BundleConfig::load(model_dir)?.unwrap_or_default();
        // The diagnostic `by_filetype` policy and per-route thresholds are
        // resolved at the bundle's own default level, never the caller's
        // `-l`, so the `models[]` array and emitted `prob` are identical
        // across levels and the JSON envelope stays cacheable. The active
        // level enters only the final verdict, via the level-independent
        // sweep over the policy grid or `general_grid`.
        let diagnostic_level = config
            .default_severity_level
            .unwrap_or(DEFAULT_SEVERITY_LEVEL);
        let cfg = EnsembleConfig::new(config, diagnostic_level);
        let mut names = RouteNames::default();
        for route in cfg.route_thresholds.keys() {
            names.intern(route);
        }

        // The policy JSON carries large diagnostic tables. Parse it while
        // decoding the general model: neither depends on the other.
        let general_dir = model_dir.join("general");
        let (policies, general) = rayon::join(
            || load_route_policies(model_dir, diagnostic_level, &mut names),
            || {
                load_bundle(&general_dir).with_context(|| {
                    format!("loading general route from {}", general_dir.display())
                })
            },
        );
        let policies = policies?;
        let general = general?;
        let (thresholds, threshold_source) = match explicit {
            Some(t) => {
                t.validate().context("invalid thresholds")?;
                (t, "explicit")
            }
            None => {
                let t = cfg.route_thresholds.get("general").copied().with_context(|| {
                    format!(
                        "{} has no general-route thresholds at level {diagnostic_level}, so the model cannot classify; reinstall the bundle with 'atomscan update-rules'",
                        model_dir.join("config.json").display(),
                    )
                })?;
                (t, "config.json")
            }
        };

        let routes = RouteSet::load(
            model_dir,
            &pin,
            &cfg.route_thresholds,
            cfg.filetype_to_filegroup,
            policies,
            names,
        );
        routes.check_required(&cfg.required_routes)?;

        tracing::info!(
            n_members = general.backend.n_members(),
            features = general.spec.total_features(),
            model_abi_version = general.spec.abi_version(),
            threshold_suspicious = thresholds.suspicious,
            threshold_hostile = thresholds.hostile,
            threshold_source,
            diagnostic_level,
            spec_version = general.spec.version(),
            filegroups = routes.filegroups.len(),
            filetypes = routes.filetypes.len(),
            "model loaded",
        );
        Ok(Self::new(
            general,
            thresholds,
            routes,
            active_level,
            cfg.general_grid,
            cfg.grid_max,
        ))
    }

    fn new(
        general: LoadedBundle,
        thresholds: Thresholds,
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
            thresholds,
            info,
            routes,
            active_level,
            general_grid,
            grid_max,
        }
    }

    /// Force every registered specialist route to load and validate.
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
        let started = Instant::now();
        self.routes.validate_all()?;
        tracing::info!(
            filegroups = self.routes.filegroups.len(),
            filetypes = self.routes.filetypes.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "validated specialist routes",
        );
        Ok(())
    }

    /// Classify a whole cleave report: featurize every file, standardize,
    /// score each route the report's primary file type reaches, and decide.
    ///
    /// A scan reaches the same decision through
    /// `Self::predict_for_report_detailed`, sharing one parse between the
    /// verdict, archive members and explanations.
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

        // A route's own class comes from the filetype's policy, else its
        // thresholds at the diagnostic level; a route with neither is benign
        // on its own and still feeds the decision.
        let classify = |route: RouteId, probability: f32, own: Option<&Thresholds>| match policy {
            Some(policy) => policy_route_class(policy, route, probability),
            None => own.map_or(Classification::Benign, |t| t.classify(probability)),
        };
        let score_general = || -> Result<Scored> {
            self.ctx.validate_keyword_inputs(parsed)?;
            let probability = self.inner.predict(general_features)?;
            Ok(Scored {
                route: RouteId::GENERAL,
                probability,
                class: classify(RouteId::GENERAL, probability, Some(&self.thresholds)),
            })
        };
        let score_specialist = |entry: Option<&(RouteId, Arc<Route>)>| -> Result<Option<Scored>> {
            let Some((id, route)) = entry else {
                return Ok(None);
            };
            route.ctx.validate_keyword_inputs(parsed)?;
            let mut features = route.ctx.extract_from_parsed(parsed);
            route.spec.standardize(&mut features);
            let probability = route.backend.predict(&features)?;
            Ok(Some(Scored {
                route: *id,
                probability,
                class: classify(*id, probability, route.thresholds.as_ref()),
            }))
        };
        // Files (including embedded members) already score in parallel. These
        // three short route calls do not need another nested join: a rayon worker
        // waiting there can steal a long, unrelated archive task and delay the
        // otherwise completed file. Keep the same result and error ordering.
        let general = score_general();
        let group_score = score_specialist(group.as_ref());
        let filetype_score = score_specialist(filetype.as_ref());
        let additional = self
            .routes
            .additional_policy_routes(file_type, &type_key, group_key);
        let additional_scores = additional
            .iter()
            .map(|entry| score_specialist(Some(entry)))
            .collect::<Result<Vec<_>>>()?;

        let mut route_probs = Vec::with_capacity(3);
        let mut scores = Vec::with_capacity(3);
        for scored in [Some(general?), group_score?, filetype_score?]
            .into_iter()
            .chain(additional_scores)
            .flatten()
        {
            route_probs.push(RouteProbability {
                route: scored.route,
                probability: scored.probability,
            });
            scores.push(RouteScore {
                model: compact_route_name(self.routes.names.name(scored.route)),
                probability: scored.probability,
                raw: scored.probability,
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
    /// is <= `suspicious_ceiling(level, grid_max)`, else benign. A file that
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
            // No grid to sweep (explicit thresholds on a bundle without a
            // levels[] table): fall back to the threshold path; the level
            // stays null.
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
            let additional = self
                .routes
                .additional_policy_routes(file_type, &type_key, group_key);
            for (_, route) in group.iter().chain(filetype.iter()).chain(additional.iter()) {
                needs = needs.union(route.ctx.raw_needs());
            }
        }
        needs
    }

    /// Largest calibrated grid level, which caps the suspicious ceiling (see
    /// `suspicious_ceiling`).
    #[must_use]
    pub(crate) const fn grid_max(&self) -> u16 {
        self.grid_max
    }

    /// Active deploy level (`-l`, pool-adjusted FP scale): the single hostile
    /// threshold — a file is hostile iff its firing level is `<=` this. `None`
    /// in manual-threshold mode.
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
pub(crate) use self::verdict::suspicious_ceiling;
pub use self::verdict::{
    Classification, Decision, Level, ModelInfo, RouteScore, SkippedRoute, ThresholdValidationError,
    Thresholds,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod fast_tree_tests;
