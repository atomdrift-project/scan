//! Route policies: which specialist answers for a file type, and how its
//! scores become a verdict at each level.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// The ensemble's top-level `config.json`, resolved at one level.
#[derive(Debug, Default)]
pub(super) struct EnsembleConfig {
    /// `cleave file_type → filegroup name` map. Files whose `file_type` is
    /// not a key here route to general only (per DESIGN.md §Runtime Decision).
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
    /// Drives the verdict sweep for filetypes with no route policy. Calibrated
    /// space, matching the general route's emitted probability.
    pub(super) general_grid: Vec<(u16, f32)>,
    /// Largest level present in the `levels[]` grid. The suspicious cap is
    /// `capped_suspicious_level(grid_max)`.
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
                Some((entry.level, threshold))
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

    /// Every policy severity, for load-time passes over the thresholds.
    pub(super) fn severities_mut(&mut self) -> impl Iterator<Item = &mut PolicySeverity> {
        self.by_filetype
            .values_mut()
            .flat_map(|policy| [&mut policy.hostile, &mut policy.suspicious])
            .chain(self.grid.values_mut().flatten().map(|lp| &mut lp.hostile))
    }
}

/// Look a cleave file type up in a per-filetype table: as given, then with
/// pure-compression suffixes stripped (`tar.gz` → `tar`; the original
/// spelling goes first so a bundle whose keys still carry the suffix keeps
/// working), then as its container archive (`gem` → `tar`, `whl` → `zip`).
/// Mirrors `collimator.data.normalize_archive_filetype`, and the specialist
/// fallback in [`RouteSet::specialist_keys`], so the resolved policy and the
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
/// `tar.gz` → `tar`). Source of truth in
/// `collimator/src/collimator/data.py::_PURE_COMPRESSION_SUFFIXES`; keep
/// the two lists in sync — collimator emits training labels using this
/// rule and scan routes scan-time files using it.
pub(super) const PURE_COMPRESSION_SUFFIXES: &[&str] = &["gz", "bz2", "xz", "zst", "z", "lzma"];

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
/// [`normalize_archive_filetype`] (which only strips compression suffixes for
/// collimator parity), this also collapses specialty package types onto their
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
    /// Route → calibrated threshold, ordered by route name. Routes absent
    /// from this list do not participate in that severity decision.
    pub(super) thresholds: Vec<(RouteId, f32)>,
    /// Learned-blend policy. When set, ``thresholds`` is empty and this
    /// severity is evaluated as ``sigmoid(intercept + sum(w_i * logit(p_i))) >=
    /// threshold`` over the named routes. Mirrors what
    /// ``azoth_route_policy_search._make_learned_blend_candidate_at_fp`` fits
    /// — the weights live in the same calibrated probability space that
    /// ``predict_calibrated`` emits, so no isotonic mapping is applied at
    /// load time.
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

    /// True iff this severity could fire on a contribution from ``route``.
    /// For OR-rule policies that's "route has a threshold"; for blend policies
    /// it's "route is one of the blend inputs."
    pub(super) fn references_route(&self, route: RouteId) -> bool {
        self.threshold(route).is_some()
            || self
                .blend
                .as_ref()
                .is_some_and(|blend| blend.routes.contains(&route))
    }

    /// True iff this severity fires given the per-route scores. Dispatches
    /// to blend evaluation when ``blend`` is set, OR-rule otherwise.
    #[cfg(test)]
    pub(super) fn fires(&self, scores: &[RouteProbability]) -> bool {
        self.fire(scores).is_some()
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
    /// Both ``weights`` and the score lookup happen by index, so reordering
    /// after load is forbidden.
    pub(super) routes: Vec<RouteId>,
    pub(super) weights: Vec<f32>,
    pub(super) intercept: f32,
    /// Calibrated-space threshold on the sigmoid output. Already in the
    /// space the blend was fit in (post-isotonic, matching ``predict_calibrated``),
    /// so the per-route isotonic mapping that ``calibrate_policy_thresholds``
    /// applies to OR-rule thresholds is intentionally NOT applied here.
    pub(super) threshold: f32,
}

impl BlendPolicy {
    /// Evaluate ``sigmoid(intercept + sum(w_i * logit(clip(p_i)))) >= threshold``
    /// over the configured routes. Missing routes (specialist not loaded for
    /// this filetype, scoring failure, etc.) fall back to "doesn't fire" —
    /// the blend can't be honestly evaluated with incomplete inputs and
    /// firing on a partial blend would be a calibration mismatch.
    #[cfg(test)]
    pub(super) fn fires(&self, scores: &[RouteProbability]) -> bool {
        self.fire(scores).is_some()
    }

    /// Compute the blend's sigmoid output and, if it crosses the threshold,
    /// return `(sigmoid_output, threshold)`. Missing routes mean the blend
    /// cannot be honestly evaluated; treat as "doesn't fire."
    pub(super) fn fire(&self, scores: &[RouteProbability]) -> Option<(f32, f32)> {
        // f64 math throughout — logit blows up near 0/1, and the cumulative
        // weighted sum can drift if we stay in f32. The final compare against
        // ``threshold`` is in f32 to match how the calibration step writes it.
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
pub(super) struct RoutePoliciesJson {
    #[serde(default)]
    pub(super) routes: HashMap<String, RoutePolicyRouteJson>,
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct RoutePolicyRouteJson {
    pub(super) filetype: String,
    #[serde(default)]
    pub(super) levels: Vec<RoutePolicyLevelJson>,
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct RoutePolicyLevelJson {
    pub(super) level: u16,
    pub(super) hostile: RoutePolicySeverityJson,
    // Collimator may omit `suspicious`; the loader derives it from `hostile`.
    pub(super) suspicious: Option<RoutePolicySeverityJson>,
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct RoutePolicySeverityJson {
    pub(super) best: Option<RoutePolicyBestJson>,
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct RoutePolicyBestJson {
    #[serde(default)]
    pub(super) thresholds: HashMap<String, f32>,
    /// Optional learned-blend variant. When set, ``thresholds`` is typically
    /// empty and the severity classifies via the blend's combined score.
    /// Mirrors azoth_route_policy_search._make_learned_blend_candidate_at_fp.
    pub(super) blend: Option<BlendPolicyJson>,
}

/// Every field a blend is evaluated with is required: a defaulted threshold
/// of 0.0 would pass validation and fire on everything.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BlendPolicyJson {
    pub(super) routes: Vec<String>,
    pub(super) weights: Vec<f32>,
    pub(super) intercept: f32,
    pub(super) threshold: f32,
    /// Currently only ``"logit"`` is supported; absent means logit. Future
    /// blend variants (e.g., raw-prob linear, monotonic GAM) would advertise
    /// themselves here so deploy refuses unknown shapes loudly rather than
    /// silently misapplying the wrong transform.
    pub(super) transform: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct LevelEntryJson {
    pub(super) level: u16,
    pub(super) hostile: SeverityEntryJson,
    // Collimator may omit `suspicious`; per-route derivation happens in
    // `thresholds_at_level`. An absent block is treated as an empty map,
    // which yields hostile-only routes (the same shape used when only some
    // routes have a calibrated suspicious threshold).
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
        // sweep. A level with no usable policy (collimator's `no_policy`) is
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
        // policy at `capped_suspicious_level(max)`. An explicit `suspicious`
        // block, when collimator emits one, wins; the level's own hostile is
        // the final fallback.
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
                            let suspicious_level = capped_suspicious_level(max);
                            levels
                                .iter()
                                .find(|lp| lp.level == suspicious_level)
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

/// One severity's policy, or `None` when collimator found none at this level
/// (no `best`, or neither thresholds nor a blend).
///
/// Thresholds outside `[0, 1]` are dropped: collimator writes one ulp above
/// 1.0 for a route that must never fire at a level, and a probability can
/// never reach it, so dropping it decides the same way.
pub(super) fn policy_severity_from_json(
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

pub(super) fn blend_policy_from_json(
    json: &BlendPolicyJson,
    names: &mut RouteNames,
) -> Result<BlendPolicy> {
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
/// hostile threshold for the same route at level
/// `capped_suspicious_level(max_grid_level)`.
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
    let suspicious_level = capped_suspicious_level(max);
    let loose_hostile = (suspicious_level != level)
        .then(|| levels.iter().find(|e| e.level == suspicious_level))
        .flatten()
        .map(|e| &e.hostile.thresholds);

    let mut out: HashMap<String, Thresholds> = HashMap::new();
    let suspicious_block = entry.suspicious.as_ref();
    for (route, &hostile) in &entry.hostile.thresholds {
        // Hostile policy is primary. When the JSON carries an explicit
        // `suspicious` block but a specific route is absent from it, that
        // route stays active as hostile-only (suspicious == hostile). When
        // collimator omits the `suspicious` block entirely, derive the
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

#[cfg(test)]
pub(super) fn policy_classify(policy: &RoutePolicy, scores: &[RouteProbability]) -> Classification {
    // Each severity decides via its own rule: OR-rule per-route thresholds or
    // learned blend. Hostile takes precedence over suspicious, same as before.
    if policy.hostile.fires(scores) {
        Classification::Hostile
    } else if policy.suspicious.fires(scores) {
        Classification::Suspicious
    } else {
        Classification::Benign
    }
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

/// Lowest level (FP per 100M benigns) at which a filetype's per-level hostile
/// policy fires, with the deciding `(probability, threshold)` at that level.
///
/// `grid` is ascending by level. Hostile thresholds loosen as the level rises,
/// so once a file fires it keeps firing — the minimum firing level is the
/// strictest budget that still flags it. Scanning every level and keeping the
/// minimum stays correct even if a bundle's grid isn't perfectly monotone.
/// Returns `None` when the file fires at no level (clean).
pub(super) fn sweep_policy_grid(
    grid: &[LevelPolicy],
    scores: &[RouteProbability],
) -> Option<(u16, f32, f32)> {
    let mut best: Option<(u16, f32, f32)> = None;
    for lp in grid {
        if let Some((p, t)) = lp.hostile.fire(scores)
            && best.is_none_or(|(bl, _, _)| lp.level < bl)
        {
            best = Some((lp.level, p, t));
        }
    }
    best
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
    } else if fired_level <= capped_suspicious_level(grid_max) {
        Classification::Suspicious
    } else {
        Classification::Benign
    }
}

/// Lowest level at which the general/OR fallback fires, for filetypes with no
/// route policy. Mirrors [`Model::decide_from_scores`]: at each level a file
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

/// Push each per-route threshold in `route_policies.json` through that route's
/// isotonic calibrator. The scoring path emits calibrated probabilities, so
/// the policy comparison must happen in calibrated space too. Routes without
/// a calibrator (pre-calibrator bundles) are left untouched. Routes referenced
/// by a policy but not loaded as a specialist are also left untouched —
/// they'll never produce a `RouteProbability` entry, so the comparison never
/// fires.
///
/// Blend severities are calibrated at fit time (the policy writer applies
/// isotonic to the per-route inputs before fitting LR), so their threshold
/// and weights are already in calibrated-prob space. Skipping the per-route
/// mapping is intentional — the logistic combination doesn't decompose
/// through isotonic the way per-route OR-rule thresholds do.
pub(super) fn calibrate_policy_thresholds<'a>(
    policies: &mut RoutePolicies,
    lookup: impl Fn(RouteId) -> Option<&'a IsotonicCalibrator>,
) {
    let mut adjusted = 0_usize;
    for severity in policies.severities_mut() {
        if severity.blend.is_some() {
            continue;
        }
        for (route, threshold) in &mut severity.thresholds {
            if let Some(cal) = lookup(*route) {
                *threshold = cal.apply(*threshold);
                adjusted += 1;
            }
        }
    }
    if adjusted > 0 {
        tracing::debug!(adjusted, "calibrated route_policies thresholds");
    }
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
