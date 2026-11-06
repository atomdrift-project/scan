//! Route policies: which specialist answers for a file type, and how its
//! scores become a verdict at each level.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use super::load::{BundleConfig, read_json};
use super::verdict::{Classification, Decision, Level, Thresholds, suspicious_ceiling};

/// The ensemble's top-level `config.json`, resolved at one level.
#[derive(Debug, Default)]
pub(super) struct EnsembleConfig {
    /// `cleave file_type → filegroup name` map. A file whose `file_type` is
    /// not a key here consults no filegroup specialist.
    pub(super) filetype_to_filegroup: HashMap<String, String>,
    /// Routes whose absence is fatal at startup. Names: `general`,
    /// `filegroups/<name>`, `filetypes/<name>` (matching the route paths in
    /// `config.json`'s `models[]` array).
    pub(super) required_routes: Vec<String>,
    /// Per-route thresholds at the resolved level. Keyed by route name as
    /// emitted in `levels[].hostile.thresholds`: `"general"`,
    /// `"filegroups/<name>"`, `"filetypes/<name>"`.
    pub(super) route_thresholds: HashMap<String, Thresholds>,
    /// General route's hostile threshold at every level, ascending by level.
    /// Drives the verdict sweep for filetypes with no route policy.
    pub(super) general_grid: Vec<(u16, f32)>,
    /// Largest level present in the `levels[]` grid. Caps
    /// `suspicious_ceiling(level, grid_max)`.
    pub(super) grid_max: u16,
}

impl EnsembleConfig {
    pub(super) fn new(cfg: BundleConfig, level: u16) -> Self {
        let route_thresholds = thresholds_at_level(&cfg.levels, level);
        // General route's hostile threshold per level, ascending. Used by the
        // verdict sweep for filetypes that have no route policy of their own.
        let mut general_grid: Vec<(u16, f32)> = cfg
            .levels
            .iter()
            .filter_map(|entry| {
                let threshold = *entry.hostile.thresholds.get("general")?;
                // Outside [0, 1] is the "never fires" sentinel (or garbage):
                // drop it, as the route-policy loader does.
                (0.0..=1.0)
                    .contains(&threshold)
                    .then_some((entry.level, threshold))
            })
            .collect();
        general_grid.sort_by_key(|&(level, _)| level);
        let grid_max = cfg.levels.iter().map(|e| e.level).max().unwrap_or(0);
        Self {
            filetype_to_filegroup: cfg.filetype_to_group,
            required_routes: cfg.required_routes,
            route_thresholds,
            general_grid,
            grid_max,
        }
    }
}

/// A route's index in [`RouteNames`]. Assigned at load, so the per-file
/// decision compares integers rather than hashing route names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct RouteId(usize);

impl RouteId {
    /// The general route: always interned first.
    pub(super) const GENERAL: Self = Self(0);
}

/// What a route name names. Route names are the on-disk layout paths:
/// `general`, `filegroups/<name>`, `filetypes/<name>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RouteKind<'a> {
    General,
    Group(&'a str),
    Type(&'a str),
}

impl<'a> RouteKind<'a> {
    pub(super) fn parse(name: &'a str) -> Option<Self> {
        if name == "general" {
            return Some(Self::General);
        }
        if let Some(group) = name.strip_prefix("filegroups/") {
            return Some(Self::Group(group));
        }
        name.strip_prefix("filetypes/").map(Self::Type)
    }
}

/// The wire name of a route: `az` for general, `az/<name>` for a specialist.
pub(super) fn compact_route_name(route: &str) -> String {
    match RouteKind::parse(route) {
        Some(RouteKind::General) => "az".to_string(),
        Some(RouteKind::Group(name) | RouteKind::Type(name)) => format!("az/{name}"),
        None => format!("az/{route}"),
    }
}

/// Every route name the bundle refers to — in `config.json`, in
/// `route_policies.json`, or on disk — interned at load. Index 0 is `general`.
#[derive(Debug)]
pub(super) struct RouteNames {
    pub(super) names: Vec<String>,
    pub(super) ids: HashMap<String, RouteId>,
}

impl Default for RouteNames {
    fn default() -> Self {
        let mut names = Self {
            names: Vec::new(),
            ids: HashMap::new(),
        };
        names.intern("general");
        names
    }
}

impl RouteNames {
    pub(super) fn intern(&mut self, name: &str) -> RouteId {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = RouteId(self.names.len());
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        id
    }

    pub(super) fn id(&self, name: &str) -> Option<RouteId> {
        self.ids.get(name).copied()
    }

    pub(super) fn name(&self, id: RouteId) -> &str {
        &self.names[id.0]
    }
}

/// Per-filetype route policy loaded from `route_policies.json`.
///
/// `by_filetype` holds the policy resolved at the default level and drives the
/// diagnostic per-route classification (`models[]`). `grid` retains the hostile
/// policy at *every* level so the verdict path can sweep for the lowest
/// false-positive level at which a file fires — that swept level is the envelope
/// level marker (see `sweep_policy_grid` and `Model::decide_swept`).
#[derive(Debug, Default)]
pub(super) struct RoutePolicies {
    pub(super) by_filetype: HashMap<String, RoutePolicy>,
    pub(super) grid: HashMap<String, Vec<LevelPolicy>>,
}

impl RoutePolicies {
    /// True when `route` is referenced by any policy at any level. Used to
    /// decide whether an on-disk specialist participates in the ensemble — a
    /// route referenced only at non-default levels must still load so the
    /// sweep can evaluate it.
    pub(super) fn contains_route(&self, route: RouteId) -> bool {
        self.grid
            .values()
            .flatten()
            .any(|lp| lp.hostile.references_route(route))
            || self.by_filetype.values().any(|policy| {
                policy.hostile.references_route(route) || policy.suspicious.references_route(route)
            })
    }

    /// The default-level policy for a file type (see [`lookup_filetype`]).
    pub(super) fn policy_for(&self, file_type: &str) -> Option<&RoutePolicy> {
        lookup_filetype(&self.by_filetype, file_type)
    }

    /// The per-level policy grid for a file type (see [`lookup_filetype`]).
    /// Used by the envelope-level sweep.
    pub(super) fn grid_for(&self, file_type: &str) -> Option<&[LevelPolicy]> {
        lookup_filetype(&self.grid, file_type).map(Vec::as_slice)
    }

    /// All inputs needed by this file type's default policy and level sweep.
    /// A policy can name routes beyond its ordinary filetype/filegroup pair.
    pub(super) fn route_ids_for(&self, file_type: &str) -> Vec<RouteId> {
        let severities = self
            .policy_for(file_type)
            .into_iter()
            .flat_map(|policy| [&policy.hostile, &policy.suspicious])
            .chain(
                self.grid_for(file_type)
                    .into_iter()
                    .flatten()
                    .map(|lp| &lp.hostile),
            );
        let mut routes: Vec<_> = severities
            .flat_map(|severity| {
                severity.thresholds.iter().map(|&(route, _)| route).chain(
                    severity
                        .blend
                        .iter()
                        .flat_map(|blend| blend.routes.iter().copied()),
                )
            })
            .collect();
        routes.sort_unstable();
        routes.dedup();
        routes
    }
}

/// Look a cleave file type up in a per-filetype table: as given, then with
/// pure-compression suffixes stripped (`tar.gz` → `tar`; the original
/// spelling goes first so a bundle whose keys still carry the suffix keeps
/// working), then as its container archive (`gem` → `tar`, `whl` → `zip`).
/// Agrees with training's label normalization, and with the specialist
/// fallback in `RouteSet::specialist_keys`, so the resolved policy and the
/// specialists scored agree.
pub(super) fn lookup_filetype<'a, V>(
    table: &'a HashMap<String, V>,
    file_type: &str,
) -> Option<&'a V> {
    if let Some(v) = table.get(file_type) {
        return Some(v);
    }
    let normalized = normalize_archive_filetype(file_type);
    if normalized != file_type
        && let Some(v) = table.get(normalized.as_ref())
    {
        return Some(v);
    }
    let container = container_filetype(file_type)?;
    if container != file_type && container != normalized {
        return table.get(container);
    }
    None
}

/// Pure-compression suffixes — formats with no multi-file container of
/// their own. Stripped from the tail of compound filetype strings (e.g.
/// `tar.gz` → `tar`). Training labels files by the same list, so the two
/// must agree: a suffix one strips and the other keeps routes a file to a
/// specialist that never saw its label.
const PURE_COMPRESSION_SUFFIXES: &[&str] = &["gz", "bz2", "xz", "zst", "z", "lzma"];

/// Strip pure-compression suffixes from a cleave-reported filetype label.
/// `tar.gz` → `tar`, `tar.bz2.xz` → `tar`. Bare compression labels
/// (`gz`, `bz2`, …) are returned unchanged — scan decompresses and
/// re-routes those at extraction time; the wrapper has no route of its
/// own. Borrows when the label is already normalized, the common case.
pub(super) fn normalize_archive_filetype(file_type: &str) -> Cow<'_, str> {
    let trimmed = file_type.trim();
    let mut normalized: Cow<'_, str> = if trimmed.bytes().any(|b| b.is_ascii_uppercase()) {
        Cow::Owned(trimmed.to_ascii_lowercase())
    } else {
        Cow::Borrowed(trimmed)
    };
    while let Some((head, tail)) = normalized.rsplit_once('.') {
        if head.is_empty() || !PURE_COMPRESSION_SUFFIXES.contains(&tail) {
            break;
        }
        let head_len = head.len();
        match &mut normalized {
            Cow::Borrowed(s) => *s = s.get(..head_len).unwrap_or(s),
            Cow::Owned(s) => s.truncate(head_len),
        }
    }
    normalized
}

/// The container archive label for a scanned file type, as filefacts defines
/// it: `gem`/`crate`/`python_sdist` → `tar`, `whl`/`jar`/`nupkg` → `zip`, the
/// compressed `tar.*` variants → `tar`. `None` when the type is not
/// archive-backed (source, binaries, bare single-file compression like `gz`).
///
/// This is the specialty-else-container half of routing: a packaged type with
/// no specialist of its own falls back to its container's. filefacts owns the
/// archive→container knowledge — no table is duplicated here. Unlike
/// [`normalize_archive_filetype`] (which only strips compression suffixes, as
/// training does), this also collapses specialty package types onto their
/// container, so it is consulted *after* the exact and compression-stripped
/// lookups, never instead of them.
pub(super) fn container_filetype(file_type: &str) -> Option<&'static str> {
    filefacts::FileType::from_label(file_type.trim())
        .and_then(filefacts::FileType::archive_format)
        .map(filefacts::ArchiveFormat::label)
}

/// Hostile decision policy for one filetype at one severity level. A filetype's
/// grid is kept in ascending `level` order at load time so the sweep can take
/// the first (lowest) firing level.
#[derive(Debug, Clone)]
pub(super) struct LevelPolicy {
    pub(super) level: u16,
    pub(super) hostile: PolicySeverity,
}

#[derive(Debug, Clone)]
pub(super) struct RoutePolicy {
    pub(super) suspicious: PolicySeverity,
    pub(super) hostile: PolicySeverity,
}

#[derive(Debug, Clone)]
pub(super) struct PolicySeverity {
    /// Route → threshold, ordered by route name. Routes absent
    /// from this list do not participate in that severity decision.
    pub(super) thresholds: Vec<(RouteId, f32)>,
    /// Learned-blend policy. When set, `thresholds` is empty and this
    /// severity is evaluated as `sigmoid(intercept + sum(w_i * logit(p_i))) >=
    /// threshold` over the named routes' scores.
    pub(super) blend: Option<BlendPolicy>,
}

impl PolicySeverity {
    /// The OR-rule threshold for `route`, if it participates.
    pub(super) fn threshold(&self, route: RouteId) -> Option<f32> {
        self.thresholds
            .iter()
            .find(|&&(r, _)| r == route)
            .map(|&(_, t)| t)
    }

    /// True iff this severity could fire on a contribution from `route`.
    /// For OR-rule policies that's "route has a threshold"; for blend policies
    /// it's "route is one of the blend inputs."
    fn references_route(&self, route: RouteId) -> bool {
        self.threshold(route).is_some()
            || self
                .blend
                .as_ref()
                .is_some_and(|blend| blend.routes.contains(&route))
    }

    /// If this severity fires, return the deciding `(probability, threshold)`
    /// pair so callers can build a [`Decision`] whose `probability` and
    /// `threshold` satisfy `probability >= threshold`.
    ///
    /// For OR-rule policies the deciding route is the one with the largest
    /// margin (`probability - threshold`) so the published `prob`/`threshold`
    /// pair best characterises why the verdict fired.
    pub(super) fn fire(&self, scores: &[RouteProbability]) -> Option<(f32, f32)> {
        if let Some(blend) = &self.blend {
            return blend.fire(scores);
        }
        let mut best: Option<(f32, f32)> = None;
        for score in scores {
            if let Some(t) = self.threshold(score.route)
                && score.probability >= t
            {
                let margin = score.probability - t;
                let best_margin = best.map_or(f32::NEG_INFINITY, |(p, bt)| p - bt);
                if margin > best_margin {
                    best = Some((score.probability, t));
                }
            }
        }
        best
    }
}

#[derive(Debug, Clone)]
pub(super) struct BlendPolicy {
    /// Routes consumed by the blend, in the order their weights are listed.
    /// `weights` pairs with them by index, so they never reorder after load.
    pub(super) routes: Vec<RouteId>,
    pub(super) weights: Vec<f32>,
    pub(super) intercept: f32,
    /// Threshold on the sigmoid output.
    pub(super) threshold: f32,
}

impl BlendPolicy {
    /// Evaluate `sigmoid(intercept + sum(w_i * logit(clip(p_i)))) >= threshold`
    /// and, if it fires, return `(sigmoid_output, threshold)`. A missing route
    /// (no specialist for this filetype, or one that failed to load) means the
    /// blend cannot be honestly evaluated, so it does not fire.
    pub(super) fn fire(&self, scores: &[RouteProbability]) -> Option<(f32, f32)> {
        // f64 math throughout — logit blows up near 0/1, and the cumulative
        // weighted sum can drift if we stay in f32. The final compare against
        // `threshold` is in f32, as the policy file stores it.
        const EPS: f64 = 1e-6;
        let mut z: f64 = f64::from(self.intercept);
        for (route, &weight) in self.routes.iter().zip(&self.weights) {
            let score = scores.iter().find(|s| s.route == *route)?;
            let p = (f64::from(score.probability)).clamp(EPS, 1.0 - EPS);
            let logit = (p / (1.0 - p)).ln();
            z += f64::from(weight) * logit;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the blend threshold is an f32; the f64 only steadies the sum"
        )]
        let sigmoid_z = (1.0 / (1.0 + (-z).exp())) as f32;
        (sigmoid_z >= self.threshold).then_some((sigmoid_z, self.threshold))
    }
}

#[derive(Debug, serde::Deserialize)]
struct RoutePoliciesJson {
    #[serde(default)]
    pub(super) routes: HashMap<String, RoutePolicyRouteJson>,
}

#[derive(Debug, serde::Deserialize)]
struct RoutePolicyRouteJson {
    pub(super) filetype: String,
    #[serde(default)]
    pub(super) levels: Vec<RoutePolicyLevelJson>,
}

#[derive(Debug, serde::Deserialize)]
struct RoutePolicyLevelJson {
    pub(super) level: u16,
    pub(super) hostile: RoutePolicySeverityJson,
    // `suspicious` may be omitted; the loader derives it from `hostile`.
    pub(super) suspicious: Option<RoutePolicySeverityJson>,
}

#[derive(Debug, serde::Deserialize)]
struct RoutePolicySeverityJson {
    pub(super) best: Option<RoutePolicyBestJson>,
}

#[derive(Debug, serde::Deserialize)]
struct RoutePolicyBestJson {
    #[serde(default)]
    pub(super) thresholds: HashMap<String, f32>,
    /// Optional learned-blend variant. When set, `thresholds` is typically
    /// empty and the severity classifies via the blend's combined score.
    pub(super) blend: Option<BlendPolicyJson>,
}

/// Every field a blend is evaluated with is required: a defaulted threshold
/// of 0.0 would pass validation and fire on everything.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BlendPolicyJson {
    pub(super) routes: Vec<String>,
    pub(super) weights: Vec<f32>,
    pub(super) intercept: f32,
    pub(super) threshold: f32,
    /// Only `"logit"` is supported; absent means logit. Another transform is
    /// refused at load rather than misapplied.
    pub(super) transform: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct LevelEntryJson {
    pub(super) level: u16,
    pub(super) hostile: SeverityEntryJson,
    // `suspicious` may be omitted; per-route derivation happens in
    // `thresholds_at_level`. An absent block is treated as an empty map,
    // which yields hostile-only routes (the same shape used when only some
    // routes have a suspicious threshold of their own).
    pub(super) suspicious: Option<SeverityEntryJson>,
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct SeverityEntryJson {
    /// Route-name → threshold map for this severity. Route names match the
    /// `models[].route` field: `"general"`, `"filegroups/<name>"`, `"filetypes/<name>"`.
    #[serde(default)]
    pub(super) thresholds: HashMap<String, f32>,
}

/// Load the searched per-filetype decision policies, interning every route
/// they name. Optional: an ensemble without `route_policies.json` keeps the
/// original OR semantics.
pub(super) fn load_route_policies(
    model_dir: &Path,
    level: u16,
    names: &mut RouteNames,
) -> Result<RoutePolicies> {
    let path = model_dir.join("route_policies.json");
    let Some(json) = read_json::<RoutePoliciesJson>(&path)? else {
        return Ok(RoutePolicies::default());
    };

    let mut by_filetype = HashMap::new();
    let mut grid: HashMap<String, Vec<LevelPolicy>> = HashMap::new();
    for route in json.routes.into_values() {
        let context = || format!("{}: filetype {}", path.display(), route.filetype);
        // Retain the hostile policy at every level (ascending) for the verdict
        // sweep. A level with no usable policy (the search found none) is
        // dropped from the grid — the sweep simply can't fire there.
        let mut levels = Vec::new();
        for entry in &route.levels {
            if let Some(hostile) =
                policy_severity_from_json(&entry.hostile, names).with_context(context)?
            {
                levels.push(LevelPolicy {
                    level: entry.level,
                    hostile,
                });
            }
        }
        levels.sort_by_key(|lp| lp.level);

        // The default-level policy drives the diagnostic per-route
        // classification in the `models[]` array. Its suspicious band is
        // derived in level-space via the suspicious-ceiling lookup: the hostile
        // policy at the loosest row within `suspicious_ceiling(level, max)`.
        // An explicit `suspicious` block, when the file has one, wins; the
        // level's own hostile is the final fallback.
        let active = match route.levels.iter().find(|entry| entry.level == level) {
            Some(entry) => {
                match policy_severity_from_json(&entry.hostile, names).with_context(context)? {
                    Some(hostile) => {
                        let explicit = match &entry.suspicious {
                            Some(json) => {
                                policy_severity_from_json(json, names).with_context(context)?
                            }
                            None => None,
                        };
                        let suspicious = explicit.unwrap_or_else(|| {
                            let max = route.levels.iter().map(|e| e.level).max().unwrap_or(level);
                            let ceiling = suspicious_ceiling(level, max);
                            levels
                                .iter()
                                .rev()
                                .find(|lp| lp.level > level && lp.level <= ceiling)
                                .map_or_else(|| hostile.clone(), |lp| lp.hostile.clone())
                        });
                        Some(RoutePolicy {
                            suspicious,
                            hostile,
                        })
                    }
                    None => None,
                }
            }
            None => None,
        };
        if !levels.is_empty() {
            grid.insert(route.filetype.clone(), levels);
        }
        if let Some(policy) = active {
            by_filetype.insert(route.filetype, policy);
        }
    }

    tracing::info!(
        path = %path.display(),
        level = level,
        routes = by_filetype.len(),
        grid_routes = grid.len(),
        "loaded route policies",
    );
    Ok(RoutePolicies { by_filetype, grid })
}

/// One severity's policy, or `None` when the search found none at this level
/// (no `best`, or neither thresholds nor a blend).
///
/// Thresholds outside `[0, 1]` are dropped: the policy search writes one ulp above
/// 1.0 for a route that must never fire at a level, and a probability can
/// never reach it, so dropping it decides the same way.
fn policy_severity_from_json(
    json: &RoutePolicySeverityJson,
    names: &mut RouteNames,
) -> Result<Option<PolicySeverity>> {
    let Some(best) = json.best.as_ref() else {
        return Ok(None);
    };
    let blend = best
        .blend
        .as_ref()
        .map(|blend| blend_policy_from_json(blend, names))
        .transpose()?;
    // Ordered by name so diagnostics that list a policy's routes are stable.
    let mut named: Vec<(&String, f32)> = best
        .thresholds
        .iter()
        .filter(|&(_, t)| (0.0..=1.0).contains(t))
        .map(|(route, &t)| (route, t))
        .collect();
    named.sort_by(|a, b| a.0.cmp(b.0));
    let thresholds: Vec<(RouteId, f32)> = named
        .into_iter()
        .map(|(route, t)| (names.intern(route), t))
        .collect();
    // A severity is loadable if either the OR-rule has at least one
    // threshold OR the blend is well-formed. The two coexist only in
    // pathological writer output — current code emits one or the other.
    if thresholds.is_empty() && blend.is_none() {
        return Ok(None);
    }
    Ok(Some(PolicySeverity { thresholds, blend }))
}

fn blend_policy_from_json(json: &BlendPolicyJson, names: &mut RouteNames) -> Result<BlendPolicy> {
    // Only the logit transform is supported. An unknown transform means the
    // writer is ahead of this deploy binary — refuse rather than misapply.
    if let Some(other) = json.transform.as_deref().filter(|t| *t != "logit") {
        anyhow::bail!("unknown blend transform {other:?}");
    }
    if json.routes.is_empty() || json.routes.len() != json.weights.len() {
        anyhow::bail!(
            "blend needs one weight per route (routes={}, weights={})",
            json.routes.len(),
            json.weights.len(),
        );
    }
    if !(0.0..=1.0).contains(&json.threshold) {
        anyhow::bail!("blend threshold {} is outside [0, 1]", json.threshold);
    }
    // JSON has no NaN, but a weight past f32's range parses as infinite and
    // would turn every blend into a constant.
    if let Some(bad) = json
        .weights
        .iter()
        .chain([&json.intercept])
        .find(|v| !v.is_finite())
    {
        anyhow::bail!("blend weights and intercept must be finite (got {bad})");
    }
    Ok(BlendPolicy {
        routes: json.routes.iter().map(|r| names.intern(r)).collect(),
        weights: json.weights.clone(),
        intercept: json.intercept,
        threshold: json.threshold,
    })
}

/// Pull per-route thresholds from `levels[]` at the requested level. Pairs up
/// hostile and suspicious thresholds for each route. When the JSON omits a
/// `suspicious` block, suspicious is derived in level-space by reading the
/// hostile threshold for the same route at the loosest row within
/// `suspicious_ceiling(level, max_grid_level)`.
pub(super) fn thresholds_at_level(
    levels: &[LevelEntryJson],
    level: u16,
) -> HashMap<String, Thresholds> {
    let Some(entry) = levels.iter().find(|e| e.level == level) else {
        if !levels.is_empty() {
            tracing::warn!(
                level = level,
                available = ?levels.iter().map(|e| e.level).collect::<Vec<_>>(),
                "ensemble config has no thresholds for this severity level"
            );
        }
        return HashMap::new();
    };

    let max = levels.iter().map(|e| e.level).max().unwrap_or(level);
    let ceiling = suspicious_ceiling(level, max);
    let loose_hostile = levels
        .iter()
        .filter(|e| e.level > level && e.level <= ceiling)
        .max_by_key(|e| e.level)
        .map(|e| &e.hostile.thresholds);

    let mut out: HashMap<String, Thresholds> = HashMap::new();
    let suspicious_block = entry.suspicious.as_ref();
    for (route, &hostile) in &entry.hostile.thresholds {
        // Hostile policy is primary. When the JSON carries an explicit
        // `suspicious` block but a specific route is absent from it, that
        // route stays active as hostile-only (suspicious == hostile). When
        // the JSON omits the `suspicious` block entirely, derive the
        // suspicious cutoff from the ceiling row's hostile threshold for the
        // same route (level-space lookup, falls back to hostile-only when
        // the route is missing from the looser row).
        let suspicious = match suspicious_block {
            Some(block) => block.thresholds.get(route).copied().unwrap_or(hostile),
            None => loose_hostile
                .and_then(|m| m.get(route))
                .map_or(hostile, |&s| s.min(hostile)),
        };
        let t = Thresholds {
            suspicious,
            hostile,
        };
        if t.validate().is_ok() {
            out.insert(route.clone(), t);
        } else {
            // Inverted hostile/suspicious thresholds for a single route.
            // Scan drops the route from this level entirely — meaningful
            // recall loss that should not pass deploy verification silently.
            tracing::error!(
                route = %route, level = level,
                suspicious = t.suspicious, hostile = t.hostile,
                "ignoring invalid thresholds in ensemble config"
            );
        }
    }
    if let Some(block) = suspicious_block {
        for (route, &suspicious) in &block.thresholds {
            if out.contains_key(route) {
                continue;
            }
            let t = Thresholds {
                suspicious,
                hostile: 1.0,
            };
            if t.validate().is_ok() {
                out.insert(route.clone(), t);
            } else {
                tracing::error!(
                    route = %route, level = level,
                    suspicious = t.suspicious, hostile = t.hostile,
                    "ignoring invalid suspicious-only thresholds in ensemble config"
                );
            }
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RouteProbability {
    pub(super) route: RouteId,
    pub(super) probability: f32,
}

/// Run the policy against the route scores and return the deciding [`Decision`]
/// for hostile/suspicious fires. Returns `None` when neither severity fires —
/// the caller picks the Benign fallback (typically general's prob against the
/// model's suspicious cutoff).
pub(super) fn policy_decide(policy: &RoutePolicy, scores: &[RouteProbability]) -> Option<Decision> {
    if let Some((p, t)) = policy.hostile.fire(scores) {
        return Some(Decision {
            class: Classification::Hostile,
            probability: p,
            threshold: t,
            level: Level::Manual,
        });
    }
    if let Some((p, t)) = policy.suspicious.fire(scores) {
        return Some(Decision {
            class: Classification::Suspicious,
            probability: p,
            threshold: t,
            level: Level::Manual,
        });
    }
    None
}

/// Lowest level (pool-adjusted FP scale) at which a filetype's per-level hostile
/// policy fires, with the deciding `(probability, threshold)` at that level.
///
/// `grid` is ascending by level. Hostile thresholds loosen as the level rises,
/// so once a file fires it keeps firing — the minimum firing level is the
/// strictest budget that still flags it. The first firing entry is the minimum
/// even if thresholds are not monotone, since the loader sorts by level.
/// Returns `None` when the file fires at no level (clean).
pub(super) fn sweep_policy_grid(
    grid: &[LevelPolicy],
    scores: &[RouteProbability],
) -> Option<(u16, f32, f32)> {
    // The loader sorts by level. The first firing entry is therefore the
    // minimum even when thresholds are not monotone. Stop before evaluating
    // the remaining policies (including potentially expensive learned blends).
    grid.iter()
        .find_map(|lp| lp.hostile.fire(scores).map(|(p, t)| (lp.level, p, t)))
}

/// Map a file's lowest firing level to a verdict at the active deploy level.
///
/// `fired_level` is the level-independent marker from the grid sweep; `level` is
/// the active `-l`. A file is **hostile** when it fires within the hostile
/// budget (`fired_level <= level`) and **suspicious** when it fires above that
/// budget but within the derived suspicious ceiling. Beyond that ceiling is
/// benign.
pub(crate) fn verdict_for_level(fired_level: u16, level: u16, grid_max: u16) -> Classification {
    if fired_level <= level {
        Classification::Hostile
    } else if fired_level <= suspicious_ceiling(level, grid_max) {
        Classification::Suspicious
    } else {
        Classification::Benign
    }
}

/// Lowest level at which the general/OR fallback fires, for filetypes with no
/// route policy. Mirrors `Model::decide_from_scores`: at each level a file
/// fires if any route's probability crosses that level's general hostile
/// threshold. The highest crossing probability is reported for that level.
pub(super) fn sweep_general_grid(
    grid: &[(u16, f32)],
    scores: &[RouteProbability],
) -> Option<(u16, f32, f32)> {
    let mut best: Option<(u16, f32, f32)> = None;
    for &(level, thr) in grid {
        if let Some(p) = scores
            .iter()
            .map(|s| s.probability)
            .filter(|&p| p >= thr)
            .reduce(f32::max)
            && best.is_none_or(|(bl, _, _)| level < bl)
        {
            best = Some((level, p, thr));
        }
    }
    best
}

pub(super) fn policy_route_class(
    policy: &RoutePolicy,
    route: RouteId,
    probability: f32,
) -> Classification {
    // Per-route classification is well-defined only for OR-rule severities —
    // a learned blend's verdict is a function of all its inputs jointly, so
    // no single route can be labeled Hostile/Suspicious on its own. For
    // blend severities we return Benign here; the final verdict still comes
    // from the policy decision, which evaluates the blend over the full score
    // vector. The diagnostic per-route classification just won't get an
    // individual contribution for blend-driven filetypes.
    let fires = |severity: &PolicySeverity| {
        severity.blend.is_none() && severity.threshold(route).is_some_and(|t| probability >= t)
    };
    if fires(&policy.hostile) {
        Classification::Hostile
    } else if fires(&policy.suspicious) {
        Classification::Suspicious
    } else {
        Classification::Benign
    }
}
