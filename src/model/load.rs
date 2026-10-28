//! Loading a model bundle: its config, backends, specialists, and the
//! routes that load lazily on first use.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// Parse `path` as JSON. `Ok(None)` when the file does not exist; a read or
/// parse failure is an error.
pub(super) fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    let data = match std::fs::read(path) {
        Ok(data) => data,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    serde_json::from_slice(&data)
        .map(Some)
        .with_context(|| format!("parsing {}", path.display()))
}

/// A bundle's `config.json`, parsed once. The single-bundle layout carries
/// top-level `suspicious`/`hostile`; an ensemble carries the routing map and
/// the per-level threshold grid. Keys this loader does not read are ignored.
#[derive(Debug, Default, serde::Deserialize)]
pub(super) struct BundleConfig {
    pub(super) suspicious: Option<f32>,
    pub(super) hostile: Option<f32>,
    /// `cleave file_type → filegroup name`.
    #[serde(default)]
    pub(super) filetype_to_group: HashMap<String, String>,
    /// Routes whose absence is fatal at startup.
    #[serde(default)]
    pub(super) required_routes: Vec<String>,
    #[serde(default)]
    pub(super) levels: Vec<LevelEntryJson>,
    /// Deploy tuning goal prescribed by the model (collimator bakes its
    /// `DEFAULT_SEVERITY_LEVEL` here). Absent on older bundles.
    pub(super) default_severity_level: Option<u16>,
}

impl BundleConfig {
    /// `<dir>/config.json`, or `None` when the bundle has none.
    pub(super) fn load(dir: &Path) -> Result<Option<Self>> {
        read_json(&dir.join("config.json"))
    }

    /// The single-bundle top-level thresholds. `suspicious` defaults to
    /// `hostile` (hostile-only): the level-space derivation needs a level
    /// table, which this block does not have.
    pub(super) fn thresholds(&self) -> Result<Option<Thresholds>> {
        let Some(hostile) = self.hostile else {
            return Ok(None);
        };
        let t = Thresholds {
            suspicious: self.suspicious.unwrap_or(hostile),
            hostile,
        };
        t.validate().context("config.json thresholds are invalid")?;
        Ok(Some(t))
    }
}

/// `evaluation.json`: the legacy home of recommended thresholds.
#[derive(Debug, serde::Deserialize)]
pub(super) struct EvaluationJson {
    #[serde(default = "default_model_abi_version")]
    pub(super) model_abi_version: u32,
    pub(super) recommended_thresholds: Option<EvaluationThresholds>,
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct EvaluationThresholds {
    pub(super) suspicious: Option<f32>,
    pub(super) hostile: Option<f32>,
}

pub(super) const fn default_model_abi_version() -> u32 {
    EXPECTED_MODEL_ABI_VERSION
}

/// Recommended thresholds from `<dir>/evaluation.json`, or `None` when the
/// file or its recommendation is absent. As with `config.json`, a missing
/// `suspicious` collapses to hostile-only.
pub(super) fn load_evaluation_thresholds(dir: &Path) -> Result<Option<Thresholds>> {
    let path = dir.join("evaluation.json");
    let Some(eval) = read_json::<EvaluationJson>(&path)? else {
        return Ok(None);
    };
    if !(18..=EXPECTED_MODEL_ABI_VERSION).contains(&eval.model_abi_version) {
        anyhow::bail!(
            "{} has model_abi_version {} but this build expects {EXPECTED_MODEL_ABI_VERSION}",
            path.display(),
            eval.model_abi_version,
        );
    }
    let Some(hostile) = eval.recommended_thresholds.as_ref().and_then(|r| r.hostile) else {
        return Ok(None);
    };
    let t = Thresholds {
        suspicious: eval
            .recommended_thresholds
            .and_then(|r| r.suspicious)
            .unwrap_or(hostile),
        hostile,
    };
    t.validate()
        .with_context(|| format!("{} recommended thresholds are invalid", path.display()))?;
    Ok(Some(t))
}

/// Output of the per-bundle loader: backend, spec, resolved thresholds, plus
/// where the thresholds came from for the load-time log line.
pub(super) struct LoadedBundle {
    pub(super) backend: Backend,
    pub(super) spec: FeatureSpec,
    pub(super) thresholds: Thresholds,
    pub(super) threshold_source: &'static str,
    /// Per-route isotonic calibrator from `calibrator.json`. Optional —
    /// pre-calibrator bundles return `None` here and skip calibration.
    pub(super) calibrator: Option<IsotonicCalibrator>,
}

/// Load the inference backend for a bundle directory. Multi-seed bundles
/// store every member at `models/seed_NN.onnx` (one file per seed); the
/// single-bundle layout has a single `model.onnx` directly under the bundle
/// dir. Both layouts are accepted; the multi-seed layout is preferred when the
/// `models/` subdirectory is present and non-empty.
///
/// Mixing the single-bundle and multi-seed layouts in the same bundle is
/// rejected.
pub(super) fn load_backend(bundle_dir: &Path) -> Result<Backend> {
    let multi_dir = bundle_dir.join("models");
    let multi = if multi_dir.is_dir() {
        collect_multi_seed_paths(&multi_dir)?
    } else {
        Vec::new()
    };

    // Single-model layout: model.onnx directly under bundle_dir. The native
    // model.txt (LightGBM) / model.json (XGBoost) layouts were retired — the
    // deploy ships ONNX-only.
    let single_onnx = bundle_dir.join("model.onnx");

    if !multi.is_empty() && single_onnx.is_file() {
        anyhow::bail!(
            "model bundle is ambiguous: both `models/` (multi-seed) and a top-level model.onnx \
             exist in {}; remove one to disambiguate the layout",
            bundle_dir.display(),
        );
    }

    let model_paths: Vec<PathBuf> = if !multi.is_empty() {
        multi
    } else if single_onnx.is_file() {
        vec![single_onnx]
    } else {
        anyhow::bail!(
            "model bundle is incomplete: no model.onnx and no models/ directory in {}",
            bundle_dir.display(),
        )
    };

    // Load seeds serially rather than with `into_par_iter`. This runs inside a
    // specialist route's `OnceLock` initializer (`RouteStore::get`), which is
    // itself reached from the rayon parallel classify. A nested rayon call here
    // lets the initializing worker thread steal another classify task that needs
    // the *same* route, re-enter the in-progress `OnceLock`, and deadlock forever
    // on its wait semaphore, hanging the scan intermittently. A bundle is at
    // most a few small ONNX seeds, so serial loading costs
    // nothing measurable and removes the re-entrancy entirely.
    let mut paths = model_paths.into_iter();
    let Some(first_path) = paths.next() else {
        anyhow::bail!("model bundle contains no loadable model")
    };
    let first = OnnxModel::load(&first_path)?;
    let n_features = first.n_features();
    let mut rest = Vec::with_capacity(paths.len());
    for path in paths {
        let member = OnnxModel::load(&path)?;
        // Refuse to mix feature counts — averaging across heterogeneous
        // feature spaces would silently produce nonsense scores.
        let n = member.n_features();
        if n_features != n {
            anyhow::bail!(
                "model bundle has mismatched feature counts in {}: member {} expects {n} \
                 features but earlier members expected {n_features}",
                bundle_dir.display(),
                path.display(),
            );
        }
        rest.push(member);
    }

    Ok(Backend { first, rest })
}

/// Pick up every `seed_*.onnx` under a `models/` directory. Native
/// `.txt`/`.json` seeds are ignored — only ONNX is loadable. Output is
/// sorted (so seed_42 lands before seed_43) and deterministic across runs.
pub(super) fn collect_multi_seed_paths(multi_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(multi_dir)
        .with_context(|| format!("reading multi-seed models dir {}", multi_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("reading entry under {}", multi_dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Tolerate bystander files (READMEs, hashes) and retired native dumps
        // so they don't break loading; only ONNX seeds are members.
        if name.starts_with("seed_") && path.extension().and_then(|e| e.to_str()) == Some("onnx") {
            paths.push(path);
        }
    }
    // Sorted for a deterministic load order of the averaged ensemble.
    paths.sort();
    Ok(paths)
}

/// Load one model bundle (model file + feature spec + thresholds).
///
/// `is_general` controls how missing artifacts are reported: for the general
/// route they're fatal load errors; for specialists callers handle the error
/// non-fatally (drop with a warning).
///
/// Threshold resolution: `explicit` → the bundle's `config.json` →
/// `evaluation.json` recommendations → [`Thresholds::default`].
pub(super) fn load_bundle(
    bundle_dir: &Path,
    explicit: Option<Thresholds>,
    is_general: bool,
) -> Result<LoadedBundle> {
    let spec_path = bundle_dir.join("feature_spec.json");
    if !spec_path.is_file() {
        if is_general {
            anyhow::bail!(
                "model bundle is incomplete: missing {}. Run 'atomscan update-rules' to refresh the installed models.",
                spec_path.display(),
            );
        }
        anyhow::bail!("missing {}", spec_path.display());
    }

    let backend = load_backend(bundle_dir)?;

    tracing::debug!(path = %spec_path.display(), "loading feature spec");
    let spec = FeatureSpec::load(&spec_path)
        .with_context(|| format!("loading feature spec from {}", spec_path.display()))?;

    if spec.total_features() != backend.num_features() {
        anyhow::bail!(
            "feature count mismatch: feature_spec.json has {} features but the model expects {} — \
             these artifacts are from different training runs",
            spec.total_features(),
            backend.num_features(),
        );
    }

    let (thresholds, threshold_source) = if let Some(explicit) = explicit {
        explicit
            .validate()
            .map_err(|error| anyhow::anyhow!("invalid thresholds: {error}"))?;
        if let Some(recommended) = load_evaluation_thresholds(bundle_dir)? {
            explicit.warn_if_divergent(&recommended);
        }
        (explicit, "explicit")
    } else if let Some(t) = BundleConfig::load(bundle_dir)?
        .map(|cfg| cfg.thresholds())
        .transpose()?
        .flatten()
    {
        (t, "config.json")
    } else if let Some(t) = load_evaluation_thresholds(bundle_dir)? {
        (t, "evaluation.json")
    } else {
        tracing::error!(
            bundle = %bundle_dir.display(),
            "no config.json or evaluation.json thresholds found — using conservative \
             fallback (suspicious={}, hostile={})",
            Thresholds::FALLBACK_SUSPICIOUS,
            Thresholds::FALLBACK_HOSTILE,
        );
        (Thresholds::default(), "fallback")
    };

    let calibrator = IsotonicCalibrator::load_optional(bundle_dir)
        .with_context(|| format!("loading optional calibrator from {}", bundle_dir.display()))?;
    // When a calibrator is present, push the bundle-level thresholds through
    // it so threshold comparisons happen in the same (calibrated) probability
    // space as `predict_calibrated`'s output.  Isotonic is monotone, so this
    // is decision-equivalent to comparing raw_score >= raw_threshold — but
    // every emitted probability is now on a meaningful [0,1] scale matching
    // the empirical malware fraction we observed at training time.
    //
    // Per-route policy thresholds (in `route_policies.json`, used by the
    // OR-of-routes deployment policies) live in a different on-disk file and
    // can't be calibrated here — the per-route calibrators aren't all loaded
    // yet at this point. They get a second pass in `Model::load_ensemble`
    // via `calibrate_policy_thresholds`, after every specialist has been
    // loaded and its calibrator is available for lookup.
    let thresholds = if let Some(cal) = calibrator.as_ref() {
        let calibrated = Thresholds {
            suspicious: cal.apply(thresholds.suspicious),
            hostile: cal.apply(thresholds.hostile),
        };
        // Re-validate: a calibrator could in principle push hostile below
        // suspicious (it shouldn't given monotonicity, but defense in depth)
        // or out of [0,1]. We'd rather refuse to load than ship a bundle
        // whose decision boundaries are nonsensical.
        calibrated.validate().with_context(|| {
            format!(
                "calibrated thresholds for {} are invalid (suspicious={}, hostile={}); \
             check the calibrator at {}/calibrator.json",
                bundle_dir.display(),
                calibrated.suspicious,
                calibrated.hostile,
                bundle_dir.display(),
            )
        })?;
        tracing::debug!(
            bundle = %bundle_dir.display(),
            breakpoints = cal.x.len(),
            "loaded isotonic calibrator (thresholds calibrated to match)"
        );
        calibrated
    } else {
        thresholds
    };

    Ok(LoadedBundle {
        backend,
        spec,
        thresholds,
        threshold_source,
        calibrator,
    })
}

/// Walk a `filegroups/` or `filetypes/` directory and register each
/// subdirectory as a lazily loaded specialist. A directory the deployment
/// config does not mention, or whose calibrator is invalid, is recorded as
/// skipped rather than failing the whole load.
///
/// `category` is the path prefix used in the ensemble config's route names —
/// either `"filegroups"` or `"filetypes"` — so specialist thresholds can be
/// looked up under e.g. `"filegroups/native"`.
pub(super) fn load_specialists(
    parent: &Path,
    route_thresholds: &HashMap<String, Thresholds>,
    route_policies: &RoutePolicies,
    names: &RouteNames,
    category: &'static str,
) -> RouteStore {
    let mut out = RouteStore::default();
    let entries = match std::fs::read_dir(parent) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return out,
        Err(e) => {
            tracing::warn!(parent = %parent.display(), error = %e, "cannot read specialist directory");
            return out;
        }
    };

    // Building each specialist's runnable tract plan is the expensive part,
    // so scans keep lazy descriptors after cheap calibrator validation.
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let name = name.to_owned();
        let route_name = format!("{category}/{name}");
        // Skip on-disk subdirectories that the deployment config doesn't list.
        // These are common as artifacts of experimentation; loading them with
        // fallback thresholds would put uncalibrated routes in the OR. Every
        // route the config or the policies name was interned before this runs.
        let route_t = route_thresholds.get(&route_name).copied();
        let Some(id) = names
            .id(&route_name)
            .filter(|&id| route_t.is_some() || route_policies.contains_route(id))
        else {
            tracing::debug!(
                category = %category,
                name = %name,
                "skipping specialist directory: no thresholds in ensemble config"
            );
            out.skipped.insert(name);
            continue;
        };
        let calibrator = match IsotonicCalibrator::load_optional(&path) {
            Ok(calibrator) => calibrator,
            Err(error) => {
                tracing::warn!(
                    category = %category,
                    name = %name,
                    path = %path.display(),
                    error = ?error,
                    "dropping specialist with invalid calibrator",
                );
                out.skipped.insert(name);
                continue;
            }
        };
        out.lazy.insert(
            name.clone(),
            LazyRoute {
                bundle_dir: path,
                name,
                category,
                id,
                thresholds: route_t.unwrap_or_default(),
                calibrator,
                loaded: OnceLock::new(),
            },
        );
    }
    out
}

/// Load and validate one specialist bundle, applying the ABI-version rule.
/// Errors here are recoverable — the specialist is dropped, not fatal.
pub(super) fn load_specialist(
    bundle_dir: &Path,
    name: &str,
    thresholds: Thresholds,
) -> Result<Route> {
    let bundle = load_bundle(bundle_dir, Some(thresholds), /* is_general = */ false)
        .with_context(|| format!("specialist {name}"))?;

    if !(18..=EXPECTED_MODEL_ABI_VERSION).contains(&bundle.spec.abi_version()) {
        anyhow::bail!(
            "ABI mismatch: specialist abi_version={} (build expects {})",
            bundle.spec.abi_version(),
            EXPECTED_MODEL_ABI_VERSION,
        );
    }

    let ctx = ExtractContext::new(&bundle.spec);

    Ok(Route {
        backend: bundle.backend,
        spec: bundle.spec,
        ctx,
        thresholds: bundle.thresholds,
        calibrator: bundle.calibrator,
    })
}

/// One inference path (general, a filegroup specialist, or a filetype specialist).
///
/// Specialists may have their own feature space. Route scoring extracts and
/// standardizes features with the route's own spec before calling its backend.
#[derive(Debug)]
pub(super) struct Route {
    pub(super) backend: Backend,
    pub(super) spec: FeatureSpec,
    pub(super) ctx: ExtractContext,
    pub(super) thresholds: Thresholds,
    /// Optional isotonic calibrator. When present, the route's raw
    /// probability is mapped through this before any downstream consumer
    /// (threshold check, OR aggregation, JSON output) sees it. Older
    /// bundles without `calibrator.json` carry `None` and behave like
    /// before — backward compatible.
    pub(super) calibrator: Option<IsotonicCalibrator>,
}

impl Route {
    /// Score a feature vector and return `(raw, calibrated)`. The raw value is
    /// the backend's pre-isotonic probability; the calibrated value is what
    /// every decision path consumes.
    pub(super) fn predict_raw_calibrated(&self, features: &[f32]) -> Result<(f32, f32)> {
        let raw = self.backend.predict(features)?;
        let calibrated = self.calibrator.as_ref().map_or(raw, |cal| cal.apply(raw));
        Ok((raw, calibrated))
    }
}

/// A specialist registered at load and built on first use.
#[derive(Debug)]
pub(super) struct LazyRoute {
    pub(super) bundle_dir: PathBuf,
    pub(super) name: String,
    pub(super) category: &'static str,
    pub(super) id: RouteId,
    pub(super) thresholds: Thresholds,
    pub(super) calibrator: Option<IsotonicCalibrator>,
    pub(super) loaded: OnceLock<RouteLoad>,
}

#[derive(Debug)]
pub(super) enum RouteLoad {
    Loaded(Arc<Route>),
    Failed(String),
}

impl LazyRoute {
    pub(super) fn load_once(&self) -> &RouteLoad {
        self.loaded.get_or_init(|| {
            match load_specialist(&self.bundle_dir, &self.name, self.thresholds) {
                Ok(route) => RouteLoad::Loaded(Arc::new(route)),
                Err(error) => {
                    let message = format!("{error:#}");
                    tracing::warn!(
                        category = %self.category,
                        name = %self.name,
                        path = %self.bundle_dir.display(),
                        error = %message,
                        "dropping lazy specialist; route will degrade to general or filegroup",
                    );
                    RouteLoad::Failed(message)
                }
            }
        })
    }

    pub(super) fn get(&self) -> Option<Arc<Route>> {
        match self.load_once() {
            RouteLoad::Loaded(route) => Some(Arc::clone(route)),
            RouteLoad::Failed(_) => None,
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        match self.load_once() {
            RouteLoad::Loaded(_) => Ok(()),
            RouteLoad::Failed(message) => {
                anyhow::bail!(
                    "specialist {}/{} at {} failed to load: {}",
                    self.category,
                    self.name,
                    self.bundle_dir.display(),
                    message
                )
            }
        }
    }
}

/// The specialists of one category (`filegroups` or `filetypes`), by name.
#[derive(Debug, Default)]
pub(super) struct RouteStore {
    pub(super) lazy: HashMap<String, LazyRoute>,
    /// On-disk specialists that were not registered (uncalibrated or invalid).
    pub(super) skipped: HashSet<String>,
}

impl RouteStore {
    /// The named specialist, loading it on first use; `None` when absent or
    /// when it failed to load.
    pub(super) fn get(&self, name: &str) -> Option<(RouteId, Arc<Route>)> {
        let lazy = self.lazy.get(name)?;
        Some((lazy.id, lazy.get()?))
    }

    pub(super) fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    pub(super) fn calibrator(&self, name: &str) -> Option<&IsotonicCalibrator> {
        self.lazy.get(name)?.calibrator.as_ref()
    }

    pub(super) fn len(&self) -> usize {
        self.lazy.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.lazy.is_empty()
    }

    pub(super) fn validate_all(&self) -> Result<()> {
        self.lazy
            .par_iter()
            .try_for_each(|(_, route)| route.validate())
    }

    pub(super) fn validate_route(&self, name: &str) -> Result<()> {
        match self.lazy.get(name) {
            Some(route) => route.validate(),
            None => anyhow::bail!("route {name:?} is not available"),
        }
    }
}

/// Routing decision: which route to consult for a file of type `T` in group `G`.
///
/// DESIGN.md says: filetype if present → filegroup if present → general always.
/// Each present route contributes to the OR over thresholds.
#[derive(Debug, Default)]
pub(super) struct RouteSet {
    /// Filegroup name → specialist model or lazy descriptor.
    pub(super) filegroups: RouteStore,
    /// Filetype name → specialist model or lazy descriptor.
    pub(super) filetypes: RouteStore,
    /// Filetype → filegroup mapping from `config.json`. Used to translate a
    /// scanned file's `type` into the applicable filegroup specialist.
    pub(super) filetype_to_filegroup: HashMap<String, String>,
    /// Optional searched policies keyed by cleave file type.
    pub(super) policies: RoutePolicies,
    /// Every route name the config, the policies and the specialists use.
    pub(super) names: RouteNames,
}

impl RouteSet {
    /// True when there are no specialist routes available; routed prediction
    /// then degrades to the general route alone.
    pub(super) fn is_empty(&self) -> bool {
        self.filegroups.is_empty() && self.filetypes.is_empty()
    }

    /// Specialist lookup keys for a scanned `file_type`, selected by ARCHIVE
    /// format rather than compression. A compressed label collapses onto its
    /// container (`tar.gz` → `tar`, `tar.bz2.xz` → `tar`) so the compressed
    /// variant reaches the same filetype/filegroup specialists — and the same
    /// per-route policy thresholds — that collimator trained and calibrated
    /// under the normalized label. Returns `(filetype key, filegroup)`.
    /// Agrees with [`lookup_filetype`], so route selection and the policy's
    /// per-route threshold keys agree.
    pub(super) fn specialist_keys<'a>(
        &'a self,
        file_type: &'a str,
    ) -> (Cow<'a, str>, Option<&'a str>) {
        let normalized = normalize_archive_filetype(file_type);
        // Specialty-else-container: prefer a specialist trained for the type
        // itself (e.g. `gem`, `whl`, `python_sdist`); when none exists, fall
        // back to its container archive (`tar`, `zip`) as filefacts defines
        // it. Compression is already collapsed by `normalize_archive_filetype`
        // (`tar.gz` → `tar`), so this only adds the package → container hop.
        let key = if self.filetypes.contains(&normalized) {
            normalized
        } else if let Some(container) =
            container_filetype(file_type).filter(|c| self.filetypes.contains(c))
        {
            Cow::Borrowed(container)
        } else {
            normalized
        };
        let group = self
            .filetype_to_filegroup
            .get(key.as_ref())
            .map(String::as_str);
        (key, group)
    }

    /// The isotonic calibrator of a route, if it has one.
    pub(super) fn calibrator<'a>(
        &'a self,
        general: Option<&'a IsotonicCalibrator>,
        route: RouteId,
    ) -> Option<&'a IsotonicCalibrator> {
        match RouteKind::parse(self.names.name(route))? {
            RouteKind::General => general,
            RouteKind::Group(name) => self.filegroups.calibrator(name),
            RouteKind::Type(name) => self.filetypes.calibrator(name),
        }
    }

    pub(super) fn validate_all(&self) -> Result<()> {
        let (filegroups, filetypes) = rayon::join(
            || self.filegroups.validate_all(),
            || self.filetypes.validate_all(),
        );
        filegroups.context("validating filegroup specialist routes")?;
        filetypes.context("validating filetype specialist routes")?;
        Ok(())
    }
}
