//! Loading a model bundle: its config, backends, specialists, and the
//! routes that load lazily on first use.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result};
use rayon::prelude::*;

use super::backend::{Backend, OnnxModel};
use super::policy::{
    LevelEntryJson, RouteId, RouteKind, RouteNames, RoutePolicies, container_filetype,
    normalize_archive_filetype,
};
use super::verdict::Thresholds;
use crate::features::{ExtractContext, FeatureSpec};

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

/// A bundle's `config.json`, parsed once: the routing map and the per-level
/// threshold grid. Keys this loader does not read are ignored.
#[derive(Debug, Default, serde::Deserialize)]
pub(super) struct BundleConfig {
    /// `cleave file_type → filegroup name`.
    #[serde(default)]
    pub(super) filetype_to_group: HashMap<String, String>,
    /// Routes whose absence is fatal at startup.
    #[serde(default)]
    pub(super) required_routes: Vec<String>,
    #[serde(default)]
    pub(super) levels: Vec<LevelEntryJson>,
    /// The level the model was calibrated to operate at.
    pub(super) default_severity_level: Option<u16>,
}

impl BundleConfig {
    /// `<dir>/config.json`, or `None` when the bundle has none.
    pub(super) fn load(dir: &Path) -> Result<Option<Self>> {
        read_json(&dir.join("config.json"))
    }
}

/// One route's model and the feature spec it was trained with.
pub(super) struct LoadedBundle {
    pub(super) backend: Backend,
    pub(super) spec: FeatureSpec,
}

/// Load the inference backend for a route directory: every member at
/// `models/seed_NN.onnx`, or a single `model.onnx`. A route holding both is
/// ambiguous and rejected.
fn load_backend(bundle_dir: &Path) -> Result<Backend> {
    let multi_dir = bundle_dir.join("models");
    let multi = if multi_dir.is_dir() {
        collect_multi_seed_paths(&multi_dir)?
    } else {
        Vec::new()
    };

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

    // Serially: a route is a few small seeds, so parallel loading buys
    // nothing measurable.
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

/// Every `seed_*.onnx` under a `models/` directory, sorted (seed_42 before
/// seed_43) so the averaged ensemble loads in a deterministic order.
fn collect_multi_seed_paths(multi_dir: &Path) -> Result<Vec<PathBuf>> {
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
        // Bystander files (READMEs, hashes) are not members.
        if name.starts_with("seed_") && path.extension().and_then(|e| e.to_str()) == Some("onnx") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// Load one route: its model and feature spec, which must agree on the
/// feature count. This is the one place that agreement is checked: every
/// vector a route scores is extracted at its spec's length.
pub(super) fn load_bundle(bundle_dir: &Path) -> Result<LoadedBundle> {
    let spec_path = bundle_dir.join("feature_spec.json");
    anyhow::ensure!(
        spec_path.is_file(),
        "model bundle is incomplete: missing {}. Run 'atomscan update-rules' to refresh the installed models.",
        spec_path.display(),
    );
    let backend = load_backend(bundle_dir)?;
    tracing::debug!(path = %spec_path.display(), "loading feature spec");
    let spec = FeatureSpec::load(&spec_path)
        .with_context(|| format!("loading feature spec from {}", spec_path.display()))?;
    anyhow::ensure!(
        spec.total_features() == backend.num_features(),
        "feature count mismatch: feature_spec.json has {} features but the model expects {} — \
         these artifacts are from different training runs",
        spec.total_features(),
        backend.num_features(),
    );
    Ok(LoadedBundle { backend, spec })
}

/// Walk a `filegroups/` or `filetypes/` directory and register each
/// subdirectory as a lazily built specialist. A directory the deployment
/// config and policies do not mention is recorded as skipped.
///
/// `category` is the path prefix used in the ensemble config's route names —
/// either `"filegroups"` or `"filetypes"` — so specialist thresholds can be
/// looked up under e.g. `"filegroups/native"`.
fn load_specialists(
    parent: &Path,
    pin: &Arc<BundlePin>,
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

    // Building a specialist's runnable plan is the expensive part, so
    // routes are registered here and built on first use.
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
        // A directory the deployment config doesn't list was never
        // calibrated, and must not join the OR. Every route the config or
        // the policies name was interned before this runs.
        let thresholds = route_thresholds.get(&route_name).copied();
        let Some(id) = names
            .id(&route_name)
            .filter(|&id| thresholds.is_some() || route_policies.contains_route(id))
        else {
            tracing::debug!(
                category,
                name = %name,
                "skipping specialist directory: no thresholds in ensemble config"
            );
            out.skipped.insert(name);
            continue;
        };
        out.lazy.insert(
            name.clone(),
            LazyRoute {
                bundle_dir: path,
                name,
                category,
                id,
                thresholds,
                pin: Arc::clone(pin),
                loaded: OnceLock::new(),
            },
        );
    }
    out
}

/// One specialist route: its model, spec and extraction tables.
///
/// Specialists may have their own feature space. Route scoring extracts and
/// standardizes features with the route's own spec before calling its backend.
#[derive(Debug)]
pub(super) struct Route {
    pub(super) backend: Backend,
    pub(super) spec: FeatureSpec,
    pub(super) ctx: ExtractContext,
    /// The route's thresholds at the diagnostic level, when the config has
    /// them; a route named only by policies has none.
    pub(super) thresholds: Option<Thresholds>,
}

/// The installed bundle a model was loaded from.
///
/// A specialist is built on first use, which may be days after load, and an
/// update swaps the whole bundle directory meanwhile. A specialist read from
/// the new bundle would be scored against the old one's policy thresholds,
/// so it is refused instead until the next reload. A
/// working tree without the install sidecar reads `None` throughout.
#[derive(Debug)]
pub(super) struct BundlePin {
    dir: PathBuf,
    commit: Option<String>,
}

impl BundlePin {
    pub(super) fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_owned(),
            commit: Self::commit(dir),
        }
    }

    fn commit(dir: &Path) -> Option<String> {
        crate::model_update::installed(dir).map(|installed| installed.commit)
    }

    /// Whether the directory still holds the bundle that was loaded.
    fn holds(&self) -> bool {
        Self::commit(&self.dir) == self.commit
    }
}

/// A specialist registered at load and built on first use.
#[derive(Debug)]
struct LazyRoute {
    bundle_dir: PathBuf,
    name: String,
    category: &'static str,
    id: RouteId,
    thresholds: Option<Thresholds>,
    pin: Arc<BundlePin>,
    loaded: OnceLock<RouteLoad>,
}

#[derive(Debug)]
enum RouteLoad {
    Loaded(Arc<Route>),
    Failed(String),
}

impl LazyRoute {
    fn load_once(&self) -> &RouteLoad {
        self.loaded.get_or_init(|| {
            // Built on a thread of its own. The first use is reached from a
            // rayon worker; were the build to run there, any rayon call inside
            // it would let that worker steal another file's task that needs
            // this same route, re-enter the `OnceLock` it is initializing, and
            // deadlock. The worker instead blocks on the join without stealing.
            let built = std::thread::scope(|scope| {
                std::thread::Builder::new()
                    .name(format!("route-{}", self.name))
                    .stack_size(crate::RAYON_STACK_MB * 1024 * 1024)
                    .spawn_scoped(scope, || self.build())
                    .context("spawning the specialist build thread")?
                    .join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("specialist build panicked")))
            });
            match built {
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

    fn build(&self) -> Result<Route> {
        let bundle =
            load_bundle(&self.bundle_dir).with_context(|| format!("specialist {}", self.name))?;
        // Checked after the read, so a swap that lands mid-read is caught.
        anyhow::ensure!(
            self.pin.holds(),
            "the model bundle was replaced on disk after it was loaded; reload to use the new one"
        );
        Ok(Route {
            ctx: ExtractContext::new(&bundle.spec),
            backend: bundle.backend,
            spec: bundle.spec,
            thresholds: self.thresholds,
        })
    }

    fn get(&self) -> Option<Arc<Route>> {
        match self.load_once() {
            RouteLoad::Loaded(route) => Some(Arc::clone(route)),
            RouteLoad::Failed(_) => None,
        }
    }

    fn validate(&self) -> Result<()> {
        match self.load_once() {
            RouteLoad::Loaded(_) => Ok(()),
            RouteLoad::Failed(message) => {
                anyhow::bail!(
                    "specialist {}/{} at {} failed to load: {message}",
                    self.category,
                    self.name,
                    self.bundle_dir.display(),
                )
            }
        }
    }
}

/// The specialists of one category (`filegroups` or `filetypes`), by name.
#[derive(Debug, Default)]
pub(super) struct RouteStore {
    lazy: HashMap<String, LazyRoute>,
    /// On-disk specialists the deployment config does not register.
    pub(super) skipped: HashSet<String>,
}

impl RouteStore {
    /// The named specialist, loading it on first use; `None` when absent or
    /// when it failed to load.
    pub(super) fn get(&self, name: &str) -> Option<(RouteId, Arc<Route>)> {
        let lazy = self.lazy.get(name)?;
        Some((lazy.id, lazy.get()?))
    }

    /// Whether the specialist is registered. Never builds it: route
    /// selection asks this for every file, and a build belongs to scoring.
    pub(super) fn contains(&self, name: &str) -> bool {
        self.lazy.contains_key(name)
    }

    pub(super) fn len(&self) -> usize {
        self.lazy.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.lazy.is_empty()
    }

    fn validate_all(&self) -> Result<()> {
        self.lazy
            .par_iter()
            .try_for_each(|(_, route)| route.validate())
    }
}

/// Every route of a bundle beyond general. A file of type `T` in group `G`
/// consults general always, plus the `G` filegroup and `T` filetype
/// specialists that exist, and any other route its policy names.
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
    /// Register the bundle's specialists and resolve its policies' routes.
    pub(super) fn load(
        model_dir: &Path,
        pin: &Arc<BundlePin>,
        route_thresholds: &HashMap<String, Thresholds>,
        filetype_to_filegroup: HashMap<String, String>,
        policies: RoutePolicies,
        names: RouteNames,
    ) -> Self {
        let started = Instant::now();
        let register = |category| {
            load_specialists(
                &model_dir.join(category),
                pin,
                route_thresholds,
                &policies,
                &names,
                category,
            )
        };
        let (filegroups, filetypes) =
            rayon::join(|| register("filegroups"), || register("filetypes"));
        tracing::info!(
            filegroups = filegroups.len(),
            filetypes = filetypes.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "prepared specialist routes",
        );
        Self {
            filegroups,
            filetypes,
            filetype_to_filegroup,
            policies,
            names,
        }
    }

    /// Fail unless every route the config marks required is present and
    /// builds. `general` was loaded already.
    pub(super) fn check_required(&self, required: &[String]) -> Result<()> {
        for name in required {
            let (store, kind, specialist) = match RouteKind::parse(name) {
                Some(RouteKind::General) => continue,
                Some(RouteKind::Group(specialist)) => (&self.filegroups, "filegroup", specialist),
                Some(RouteKind::Type(specialist)) => (&self.filetypes, "filetype", specialist),
                None => anyhow::bail!(
                    "unknown required-route name {name:?} in ensemble config; \
                     expected `general`, `filegroups/<name>`, or `filetypes/<name>`"
                ),
            };
            match store.lazy.get(specialist) {
                Some(route) => route.validate(),
                None => Err(anyhow::anyhow!("route {specialist:?} is not available")),
            }
            .with_context(|| {
                format!(
                    "ensemble config marks {kind} {specialist:?} as required but it failed to load"
                )
            })?;
        }
        Ok(())
    }

    /// True when there are no specialist routes available; routed prediction
    /// then degrades to the general route alone.
    pub(super) fn is_empty(&self) -> bool {
        self.filegroups.is_empty() && self.filetypes.is_empty()
    }

    /// Specialist lookup keys for a scanned `file_type`, selected by ARCHIVE
    /// format rather than compression. A compressed label collapses onto its
    /// container (`tar.gz` → `tar`, `tar.bz2.xz` → `tar`) so the compressed
    /// variant reaches the same filetype/filegroup specialists — and the same
    /// per-route policy thresholds — that training calibrated under the
    /// normalized label. Returns `(filetype key, filegroup)`.
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

    /// Policy inputs beyond the standard filetype/filegroup specialists.
    pub(super) fn additional_policy_routes(&self, file_type: &str) -> Vec<(RouteId, Arc<Route>)> {
        let (type_key, group_key) = self.specialist_keys(file_type);
        self.policies
            .route_ids_for(file_type)
            .into_iter()
            .filter_map(|id| match RouteKind::parse(self.names.name(id))? {
                RouteKind::Group(name) if Some(name) != group_key => self.filegroups.get(name),
                RouteKind::Type(name) if name != type_key => self.filetypes.get(name),
                _ => None,
            })
            .collect()
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
