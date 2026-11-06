//! Which declared dependencies to skip before fetching: too old, superseded
//! by a pin, pulled, or already known good.

use std::collections::HashSet;
use std::path::Path;

use cleave::AnalysisReport;
use fletch::fetch::RecordedSource;
use fletch::{RefKind, RefLocator, Reference, Registry};
use rayon::prelude::*;

use crate::provenance::RegistryProvenance;

use super::policy::FetchPolicy;
use super::registry::{MAX_REGISTRY_LOOKUPS, lookup_registries};
use super::{
    Coordinate, Resources, SECS_PER_DAY, immutable_coordinate, locator, pins_exact_version,
};

/// Why a dependency's artifact fetch was skipped. The registry record is
/// materialized (and trait-matched) in every case; only the byte fetch+scan is
/// skipped. Drives how the skip is surfaced in the fetch progress block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SkipReason {
    /// Older than [`FetchPolicy::max_dep_age_days`]. The common, expected case
    /// — reported at debug only, no progress line.
    AgedOut,
    /// The version was unpublished/yanked from the registry — no artifact to
    /// fetch. Rare and worth surfacing.
    Removed,
    /// The *resolved* coordinate is vouched by the known-good bloom, and its
    /// trust is not stale. Reached only by references whose declared locator
    /// carried no version (a manifest range), which the pre-lookup probe in
    /// [`age_gate`] cannot match; pinned coordinates are filtered before the
    /// lookup instead.
    KnownGood,
}

impl SkipReason {
    /// The reason a skip log line gives.
    pub(super) const fn log_reason(self) -> &'static str {
        match self {
            Self::Removed => "version removed from registry",
            Self::AgedOut => "older than the dependency age limit",
            Self::KnownGood => "known-good (bloom, resolved version)",
        }
    }
}

/// Whether a dependency version has been withdrawn from its registry — an npm
/// unpublish (`version_removed`), a pypi/crates yank (recorded as a
/// `deprecated` reason, which never sets `version_removed`), or an npm security
/// takedown (`security_hold`). A withdrawn version's known-good vouch is suspect
/// — it is often *removed because* it was found malicious — so it is re-scanned.
///
/// Withdrawn is not the same as unfetchable, which is what makes re-scanning
/// worth doing. Only an npm unpublish actually removes the bytes; a yank on
/// crates.io or PyPI leaves the artifact downloadable forever (it only stops
/// *new* resolution, so pinned builds keep working), and npm's security hold
/// replaces the release with a placeholder that is served like any other. Those
/// are the cases worth a second look, and their bytes are still there to look at.
///
/// The `version_removed` arm is consequently unreachable from [`age_gate`],
/// which tests it first and settles those as [`SkipReason::Removed`] before
/// consulting [`must_rescan`] at all. It is kept because this predicate is about
/// withdrawal, not about that one caller's ordering.
pub(super) fn dep_pulled(reg: &Registry) -> bool {
    reg.version_removed == Some(true)
        || reg.security_hold == Some(true)
        || reg.deprecated.as_deref().is_some_and(|d| {
            let d = d.to_ascii_lowercase();
            d.contains("yank") || d.contains("withdrawn") || d.contains("unpublish")
        })
}

/// Number of seconds in the freshness window: a version published this recently
/// is re-scanned rather than trusted on a known-good vouch.
///
/// A published registry version is immutable — npm, crates.io and PyPI all
/// refuse to re-publish `name@version` with different bytes — so a vouch for a
/// pinned coordinate cannot go stale the way a mutable one can, and age is
/// otherwise no reason to distrust it. What this window buys is narrower: cover
/// for the hours right after a release, where a compromise is freshest and
/// least-vetted, and insurance against the bloom itself being wrong about a
/// brand-new package. Hours, not days, is the right size for that.
pub(super) const FRESH_WINDOW_SECS: u64 = 4 * 3_600;

/// Whether this version was published inside [`FRESH_WINDOW_SECS`].
pub(super) fn freshly_published(reg: &Registry, now: u64) -> bool {
    reg.age_secs(now)
        .is_some_and(|age| age <= FRESH_WINDOW_SECS)
}

/// A known-good dependency is normally skipped; re-scan it anyway when its trust
/// may be stale — the version was pulled/yanked, or published very recently.
pub(super) fn must_rescan(reg: &Registry, now: u64) -> bool {
    dep_pulled(reg) || freshly_published(reg, now)
}

/// The publish timestamp a Go pseudo-version carries in its own version string,
/// as seconds since the epoch.
///
/// The module proxy mints a pseudo-version for a commit with no semver tag:
/// `v0.0.0-20260528132821-f66b8cdce5b3`, or `v1.2.3-0.20260528132821-abc…` when
/// it follows a release. The middle field is a UTC `yyyymmddhhmmss` stamp of the
/// commit — the same date the registry would report, already in hand. Parsing it
/// locally lets the age gate reject an old module with no round-trip at all; a
/// `go.sum` alone can declare hundreds.
///
/// `None` for a tagged version (`v1.9.0`), a malformed stamp, or an
/// out-of-range field — anything not confidently datable falls through to the
/// registry lookup rather than being guessed at, so this can only ever save
/// work, never invent an age.
pub(super) fn go_pseudo_version_published(purl: &str) -> Option<u64> {
    let coordinate = Coordinate::of(purl).filter(|c| c.typ == "golang")?;
    // Split on both separators: the bare form joins the stamp with dashes
    // (`v0.0.0-<stamp>-<hash>`), while the post-release form reaches it through
    // a dotted pre-release segment (`v1.2.3-0.<stamp>-<hash>`). The version's own
    // numeric fields are far too short to be mistaken for a 14-digit stamp, and
    // a Go commit hash is 12 hex chars.
    let stamp = coordinate
        .version?
        .split(['-', '.'])
        .find(|f| f.len() == 14 && f.bytes().all(|b| b.is_ascii_digit()))?;
    let n = |a: usize, b: usize| stamp.get(a..b)?.parse::<u32>().ok();
    crate::civil::utc_epoch(
        n(0, 4)?,
        n(4, 6)?,
        n(6, 8)?,
        n(8, 10)?,
        n(10, 12)?,
        n(12, 14)?,
    )
}

/// The reference's coordinate with the registry's resolved version attached, or
/// `None` when it isn't a PURL or nothing resolved.
///
/// A manifest declares a *range* (`"axios": "^1.6.0"`), so an npm reference's
/// locator usually carries no version at all — and the bloom is keyed on exact
/// `name@version`, so probing the declared form can only ever miss. Most npm
/// references arrive version-less, so without this the known-good check would
/// be structurally dead for npm. Lockfile ecosystems (cargo, golang) pin exact
/// versions and are unaffected.
///
/// The version is appended to the declared locator rather than rebuilt from
/// `reg.ecosystem`, so the PURL type and namespace stay exactly as the reference
/// resolver produced them.
pub(super) fn resolved_purl(r: &Reference, reg: &Registry) -> Option<String> {
    let RefLocator::Purl(purl) = &r.locator else {
        return None;
    };
    if reg.version.is_empty() {
        return None;
    }
    if Coordinate::of(purl)?.version.is_some() {
        return Some(purl.clone());
    }
    Some(format!("{purl}@{}", reg.version))
}

/// Skip predicate for fetched-dependency analysis: skip any member cleave is
/// about to analyze whose sha256 the installed bloom filters vouch known-good —
/// the same short-circuit the top-level scan applies, so a prebuilt native tool
/// shipped inside a dependency isn't needlessly re-disassembled. Unlike the
/// top-level predicate it applies no local-file freshness guard: a dependency's
/// bytes are content-addressed (fetched by locator, sha-verified) and extracted
/// to fresh temp files, so an mtime check would spuriously force analysis every
/// run. Known-bad, conflicted, and unknown members are always analyzed. `None`
/// when no bloom set is installed, leaving analysis unfiltered.
pub(super) fn dep_skip_predicate() -> Option<cleave::SkipPredicate> {
    let lookup = crate::bloom_repo::global()?;
    Some(cleave::SkipPredicate(std::sync::Arc::new(
        move |sha_hex: &str, _path: &Path| {
            burton::parse_sha256_hex(sha_hex)
                .is_some_and(|d| lookup.may_skip(&burton::Artifact::sha256(&d)))
        },
    )))
}

/// A dependency whose registry record resolved. The record is materialized
/// whether or not its bytes are fetched; `skip` says why they were not.
pub(super) struct Gated {
    pub(super) reference: Reference,
    pub(super) record: Registry,
    /// The provider documents a fresh lookup read. `None` after a memo hit,
    /// whose documents stay in fletch's blob cache until a kept record needs
    /// them. Also cleared once registry analysis releases or compacts them.
    pub(super) sources: Option<Vec<RecordedSource>>,
    /// A kept fresh lookup, compacted as soon as registry analysis completes.
    pub(super) provenance: Option<RegistryProvenance>,
    pub(super) skip: Option<SkipReason>,
}

impl Gated {
    pub(super) fn finish_registry_analysis(&mut self, sub: Option<&AnalysisReport>) {
        let Some(sources) = self.sources.take() else {
            return; // A memo hit still recovers sources only if merging needs them.
        };
        if sub.is_some_and(|sub| {
            sub.files.iter().any(|file| file.parent_id.is_none())
                && keep_registry_provenance(self.skip, sub)
        }) {
            self.provenance = Some(RegistryProvenance::from_record_sources(
                self.record.clone(),
                &sources,
            ));
        }
        // Even a batch with a very slow payload does not pin raw documents
        // that registry analysis has finished using. Kept documents retain all
        // their original evidence in the compact provenance representation.
    }
}

pub(super) fn keep_registry_provenance(skip: Option<SkipReason>, sub: &AnalysisReport) -> bool {
    skip.is_none()
        || sub.files.iter().any(|file| {
            file.findings
                .iter()
                .any(|finding| finding.crit >= cleave::Criticality::Notable)
        })
}

/// One registry lookup's result, as [`Gated`] carries it.
pub(super) struct Lookup {
    pub(super) record: Registry,
    pub(super) sources: Option<Vec<RecordedSource>>,
}

/// Split a locator into `(package key, version)` when it is a PURL carrying an
/// explicit version. The key keeps npm scoped names whole
/// (`pkg:npm/@scope/name@1.2.3` → `pkg:npm/@scope/name`). A candidate version
/// must start with an ASCII digit — git refs, tags, and a scoped name with no
/// version at all (`pkg:npm/@scope/name`) are not versions and exempt the
/// reference from the newest-version gate.
pub(super) fn versioned_purl(locator: &str) -> Option<(&str, &str)> {
    let coordinate = Coordinate::of(locator)?;
    let version = coordinate
        .version
        .filter(|v| v.starts_with(|c: char| c.is_ascii_digit()))?;
    Some((coordinate.key, version))
}

/// Whether `purl` carries the qualifier `key`, whatever its value.
pub(super) fn has_qualifier(purl: &str, key: &str) -> bool {
    purl.split_once('?')
        .map(|(_, rest)| rest.split('#').next().unwrap_or(rest))
        .is_some_and(|qualifiers| {
            qualifiers
                .split('&')
                .any(|pair| pair.split_once('=').map_or(pair, |(k, _)| k) == key)
        })
}

/// True when `r` is a bare (versionless) PURL whose coordinate is pinned by a
/// version-carrying sibling elsewhere in the tree (`pinned` holds those
/// coordinates). A manifest range reaches us version-stripped — `pkg:npm/foo` —
/// and would resolve to `dist-tags/latest`; when a lockfile pins the same
/// coordinate (`pkg:npm/foo@1.2.3`), that pin is ground truth and the bare
/// sibling is redundant, so it is dropped. A versionless PURL's whole string is
/// its coordinate, so an exact set membership is the supersede; a git/tag pin
/// (`pkg:npm/foo@dev`) keeps its `@ref`, is not versionless, and never matches.
pub(super) fn superseded_by_pin(r: &Reference, pinned: &HashSet<String>) -> bool {
    let RefLocator::Purl(p) = &r.locator else {
        return false;
    };
    // Constrained Cargo/Python manifests are reconciled with their own lock;
    // an unrelated package's pin cannot satisfy their requirement.
    !has_qualifier(p, "version_requirement")
        && versioned_purl(p).is_none()
        && pinned.contains(p.as_str())
}

/// When a dependency's code runs, which orders a hop's fetches: what runs
/// during the install itself first, development-only tooling last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Priority {
    /// Install scripts, build dependencies and proc-macros: run on install.
    Build,
    /// A package an install command names.
    Command,
    /// An ordinary runtime dependency.
    Runtime,
    /// A development-only dependency.
    Development,
}

/// [`Priority`] of `reference`.
pub(super) fn dependency_execution_priority(reference: &Reference) -> Priority {
    if reference
        .context
        .as_ref()
        .is_some_and(|c| c.has_install_script || c.scope == filefacts::DependencyScope::Build)
        || reference.source.starts_with("npm.scripts.")
        || reference.source.contains("build-dependencies")
        || reference.source.contains("build-system.requires")
        || reference.source.contains("proc-macro")
    {
        Priority::Build
    } else if reference.kind == RefKind::Command {
        Priority::Command
    } else if reference
        .context
        .as_ref()
        .is_some_and(|c| c.scope == filefacts::DependencyScope::Development)
        || reference.source.contains("dev-dependencies")
    {
        Priority::Development
    } else {
        Priority::Runtime
    }
}

/// Lenient, numeric-aware version ordering: the string splits into components
/// at `.`, `-`, `_`, and `+`; each component compares by its leading integer
/// first (`3` > `rc1`? — a fully numeric component outranks a text-led one, so
/// `1.2.3` > `1.2.rc1`), then by remaining text. More components with an equal
/// prefix is newer (`1.2.1` > `1.2`). This orders real registry releases of
/// one package — semver, pep440-ish, and date-like schemes — without
/// validating any of them.
pub(super) fn lenient_version_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    fn component_cmp(x: &str, y: &str) -> Ordering {
        let digits = |s: &str| s.chars().take_while(char::is_ascii_digit).count();
        let (nx, rx) = x.split_at(digits(x));
        let (ny, ry) = y.split_at(digits(y));
        match (nx.is_empty(), ny.is_empty()) {
            (false, true) => return Ordering::Greater,
            (true, false) => return Ordering::Less,
            (true, true) => return x.cmp(y),
            (false, false) => {}
        }
        let num = match (nx.parse::<u64>(), ny.parse::<u64>()) {
            (Ok(vx), Ok(vy)) => vx.cmp(&vy),
            // A digit run longer than u64 still orders consistently as text.
            _ => nx.cmp(ny),
        };
        num.then_with(|| rx.cmp(ry))
    }
    const SEPARATORS: [char; 4] = ['.', '-', '_', '+'];
    let (mut ca, mut cb) = (a.split(SEPARATORS), b.split(SEPARATORS));
    loop {
        match (ca.next(), cb.next()) {
            (Some(x), Some(y)) => match component_cmp(x, y) {
                Ordering::Equal => {}
                ord => return ord,
            },
            (Some(_), None) => return Ordering::Greater,
            (None, Some(_)) => return Ordering::Less,
            (None, None) => return Ordering::Equal,
        }
    }
}

/// Look up each declared dependency's registry metadata, stamp its relative
/// age, and decide which to fetch. A dependency older than the policy's age
/// ceiling — or one whose coordinate is known-good and whose trust isn't stale —
/// is dropped before the expensive fetch+scan of its bytes; one whose age is
/// unknown or under the ceiling is kept — fail open, so a registry hiccup or an
/// unsupported ecosystem never silently hides a dependency from the scan. URLs
/// and command-mentioned packages aren't gated: their risk isn't a function of a
/// registry release date. Returns the refs to fetch plus, for *every* dependency
/// that resolved a registry record, its [`Gated`] — the record is materialized
/// whether or not its bytes are fetched, and the reason drives the skip report.
///
/// Both network steps — resolving a range, and the registry record — are
/// capped per manifest at [`MAX_REGISTRY_LOOKUPS`]: a crafted lockfile
/// declaring 100k names must not become 100k registry requests from this host.
/// A reference past the cap is kept unresolved (fail open), so it still meets
/// the fetch budget and stays visible in the outcome.
pub(super) fn age_gate(
    selected: Vec<Reference>,
    policy: &FetchPolicy,
    res: &Resources,
    now: u64,
) -> (Vec<Reference>, Vec<Gated>) {
    let (selected, mut unresolved) = resolve_ranges(selected, res);
    // `None` ceiling (a `max_dep_age_days` of 0) gates nothing, but registry
    // records are still looked up and materialized.
    let max_age =
        (policy.max_dep_age_days > 0).then(|| u64::from(policy.max_dep_age_days) * SECS_PER_DAY);
    let mut selected = settle_locally(selected, max_age, now);
    // The network round-trips run concurrently up front; the gate decision below
    // is then pure, so it stays deterministic in `selected` order.
    if selected.len() > MAX_REGISTRY_LOOKUPS {
        tracing::warn!(
            declared = selected.len(),
            cap = MAX_REGISTRY_LOOKUPS,
            "manifest declares more dependencies than the registry lookup cap; the rest keep no registry record"
        );
        unresolved.extend(selected.split_off(MAX_REGISTRY_LOOKUPS));
    }
    let lookups = lookup_registries(&selected, res, now);
    let mut keep = Vec::with_capacity(selected.len());
    let mut registries = Vec::new();
    for (r, lookup) in selected.into_iter().zip(lookups) {
        // A non-dependency, or a dependency whose record didn't resolve, is
        // fetched (fail open).
        let Some(Lookup { record, sources }) = lookup else {
            keep.push(r);
            continue;
        };
        let skip = skip_reason(&r, &record, max_age, now);
        if skip.is_none() {
            keep.push(r.clone());
        }
        registries.push(Gated {
            reference: r,
            record,
            sources,
            provenance: None,
            skip,
        });
    }
    keep.extend(unresolved);
    (keep, registries)
}

/// Pin each declared range to the exact release it resolves to, so a
/// manifest range is never age-gated by today's unrelated latest release.
/// Resolution may ask a registry, so the references resolve concurrently, in
/// order. Returns the resolved references and the ones left as declared: past
/// the lookup cap, or not resolvable — those bypass registry gating and keep
/// the normal unresolved fetch outcome, so incomplete coverage stays visible.
fn resolve_ranges(selected: Vec<Reference>, res: &Resources) -> (Vec<Reference>, Vec<Reference>) {
    let mut unresolved = Vec::new();
    let mut ranges = 0usize;
    let selected: Vec<Reference> = selected
        .into_iter()
        .filter_map(|r| {
            if let RefLocator::Purl(purl) = &r.locator
                && !pins_exact_version(purl)
            {
                ranges += 1;
                if ranges > MAX_REGISTRY_LOOKUPS {
                    unresolved.push(r);
                    return None;
                }
            }
            Some(r)
        })
        .collect();
    if ranges > MAX_REGISTRY_LOOKUPS {
        tracing::warn!(
            declared = ranges,
            cap = MAX_REGISTRY_LOOKUPS,
            "manifest declares more unpinned dependencies than the lookup cap; the rest are not resolved"
        );
    }
    let resolved: Vec<(Reference, Option<Reference>)> = selected
        .into_par_iter()
        .map(|r| {
            let exact = fletch::fetch::resolve_declared_reference(&r, &*res.net, &res.cache)
                .inspect_err(|error| match error {
                    fletch::fetch::ResolveError::Failed(_) => {
                        tracing::warn!(locator = locator(&r), %error, "registry unavailable; dependency range left unresolved");
                    }
                    _ => tracing::debug!(locator = locator(&r), %error, "dependency range not resolved"),
                })
                .ok();
            (r, exact)
        })
        .collect();
    let mut exact = Vec::with_capacity(resolved.len());
    for (r, resolved) in resolved {
        match resolved {
            Some(resolved) => exact.push(resolved),
            None => unresolved.push(r),
        }
    }
    (exact, unresolved)
}

/// Drop the dependencies settled without a registry round trip: those the
/// known-good bloom vouches for, and Go pseudo-versions whose own commit date
/// is past `max_age`.
///
/// The bloom probe is a local filter test on the PURL, so asking it *before*
/// [`lookup_registries`] turns a known-good dependency from "one registry
/// round-trip, then skip" into "no I/O at all" — the largest saving available
/// on a lockfile-heavy scan. Two things are given up, deliberately: the
/// artifact is not re-scanned even if the version was later yanked
/// ([`must_rescan`] needs the record this skips), and no `*.registry.json`
/// node is materialized for it, so its registry metadata contributes no
/// findings. Both apply *only* to bloom-vouched coordinates.
fn settle_locally(selected: Vec<Reference>, max_age: Option<u64>, now: u64) -> Vec<Reference> {
    let mut kept = Vec::with_capacity(selected.len());
    for r in selected {
        // Purl-keyed, so the bloom vouches for the coordinate, not the exact
        // bytes; a mutable coordinate is never vouched for by name.
        let vouched = matches!(&r.locator, RefLocator::Purl(purl)
            if immutable_coordinate(purl, None)
                && crate::bloom_repo::global().is_some_and(|lk| lk.decide_purl(purl).may_skip()));
        if vouched {
            crate::bloom_repo::record(crate::bloom_repo::Decision::Skip, false);
            tracing::debug!(
                package = %locator(&r),
                "known-good coordinate (bloom); registry lookup and fetch both skipped"
            );
            continue;
        }
        let self_dated = match (&r.locator, max_age) {
            (RefLocator::Purl(purl), Some(max)) => go_pseudo_version_published(purl)
                .is_some_and(|published| now.saturating_sub(published) >= max),
            _ => false,
        };
        if self_dated {
            tracing::debug!(
                package = %locator(&r),
                "go pseudo-version dates itself past the dependency age limit; registry lookup skipped"
            );
            continue;
        }
        kept.push(r);
    }
    kept
}

/// Why a dependency with a registry record is not fetched, if it is not. A
/// version the registry already removed has no fetchable artifact, so the
/// doomed fetch is skipped too; in every skip case the materialized record's
/// signals still surface.
///
/// The known-good check here is the post-lookup counterpart to
/// [`settle_locally`]'s bloom probe: it costs nothing extra (the record is in
/// hand) and catches the range-declared references the pre-lookup probe
/// cannot, and with the record in hand [`must_rescan`] still catches a yanked
/// version. Only for a coordinate whose bytes cannot change
/// ([`immutable_coordinate`]): a vouch for a git tag or a young PyPI release
/// says nothing about what it names now.
fn skip_reason(
    r: &Reference,
    record: &Registry,
    max_age: Option<u64>,
    now: u64,
) -> Option<SkipReason> {
    let vouched = || {
        resolved_purl(r, record)
            .filter(|purl| immutable_coordinate(purl, record.age_secs(now)))
            .is_some_and(|purl| {
                crate::bloom_repo::global().is_some_and(|lk| lk.decide_purl(&purl).may_skip())
            })
    };
    if record.version_removed == Some(true) {
        Some(SkipReason::Removed)
    } else if max_age.is_some_and(|max| record.age_secs(now).is_some_and(|age| age >= max)) {
        Some(SkipReason::AgedOut)
    } else if vouched() && !must_rescan(record, now) {
        Some(SkipReason::KnownGood)
    } else {
        None
    }
}
