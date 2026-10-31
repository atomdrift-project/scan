//! What to follow, and the limits that bound it: the follow vocabulary, the
//! per-file and per-run defaults, and the host platform a native dependency
//! must match.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// Default fetch recursion depth — the number of hops followed from the root.
/// `2` reaches a stage-3 payload (root → stage-2 → stage-3), since multi-stage
/// `curl | bash` droppers are the common case.
pub const DEFAULT_FETCH_DEPTH: u8 = 2;

/// Default age ceiling for fetching a declared dependency, in days. A version
/// older than this has had a long window for community discovery, so only its
/// registry metadata is looked up (a PURL lookup) and the expensive
/// fetch-and-scan is skipped; recent releases — where a supply-chain compromise
/// is freshest and least-vetted — are still pulled and fully scanned. Set to a
/// week: past that, a malicious release has almost always been caught and
/// yanked, and the byte scan's cost isn't worth it. `0` disables the gate.
pub const DEFAULT_MAX_DEP_AGE_DAYS: u32 = 7;

/// Dependency age ceiling for a cache-population role such as postdoc's
/// worker: `0` — no gate, fetch every resolvable dependency.
///
/// An interactive scan gates at [`DEFAULT_MAX_DEP_AGE_DAYS`] because
/// a fresh release is where a supply-chain compromise shows up and the operator is
/// waiting. A cache populator is the opposite trade: it runs unattended to fill the
/// shared corpus, and every dependency it resolves lands in hopper carrying a
/// package coordinate — the raw material known-good bloom coverage is built from.
/// Gating those out means the cache never learns the long tail that real scans
/// keep re-resolving.
pub const WORKER_MAX_DEP_AGE_DAYS: u32 = 0;

/// The follow selection a `serve` process uses when the operator named none:
/// everything, CI actions included.
///
/// This is a cache-population role. It scans on behalf of everybody, and the
/// corpus it fills is asked about artifacts nobody has looked at yet — so a
/// category left unfollowed is one the corpus never learns about, for every
/// consumer, until somebody notices and restarts the fleet with a wider flag.
/// The narrower interactive default exists to keep one person's scan fast,
/// which is not what a service is for.
///
/// Widest-by-default also settles which verdict wins. Hopper holds one verdict
/// per artifact and the last writer takes the row, so a fleet whose members
/// follow different amounts lets a narrow answer overwrite a wide one. When
/// every server follows everything, there is no narrower answer to lose to.
#[must_use]
pub fn default_service_follow_policy() -> FetchPolicy {
    FetchPolicy {
        urls: true,
        packages: true,
        deps: true,
        ci: true,
        ..FetchPolicy::default()
    }
}

/// 1024-based size units, the basis for every byte ceiling.
pub(super) const MIB: u64 = 1024 * 1024;
pub(super) const GIB: u64 = 1024 * MIB;

/// Default size ceiling for a single fetched artifact —
/// mirrors fletch's [`fletch::fetch::DEFAULT_MAX_FETCH_BYTES`] (256 MiB). A
/// response larger than this is abandoned, so one artifact can't dominate a run.
pub const DEFAULT_MAX_FETCH_SIZE: u64 = 256 * MIB;

/// Default ceiling on *live* non-URL fetches triggered by a single scanned
/// file. Cache hits don't count, so a warm re-run is
/// never throttled; this bounds the dependency/package fan-out one crafted file
/// can trigger.
pub const DEFAULT_MAX_FILE_FETCHES: usize = 100;

/// Default ceiling on *live* opportunistic URL fetches triggered by a single
/// scanned file. Cache hits don't count, so a warm re-run is never throttled.
pub const DEFAULT_MAX_URL_FETCHES: usize = 4;

/// Default ceiling on total bytes fetched on behalf of a single scanned file.
pub const DEFAULT_MAX_FILE_SIZE: u64 = 2 * GIB;

/// Default wall-clock ceiling on one artifact's whole fetch phase. The count
/// and byte budgets bound how *much* a scan
/// pulls but not how *long* pulling takes: a wide tree of slow registries can
/// hold a scan open for as long as it likes while every count budget still has
/// room. Five minutes is well past what a healthy tree needs and short enough
/// that a wedged one still returns a verdict. `0` disables the cap.
pub const DEFAULT_FETCH_TIMEOUT: Duration = Duration::from_secs(300);

/// Default ceiling on *live* fetches across one whole execution. Lifted in
/// long-lived server modes, where each job is bounded by the per-file caps
/// instead.
pub const DEFAULT_MAX_TOTAL_FETCHES: usize = 1000;

/// Default ceiling on total bytes fetched across one whole execution. Lifted
/// in long-lived server modes.
pub const DEFAULT_MAX_TOTAL_SIZE: u64 = 10 * GIB;

/// Which discovered references to follow, plus how many hops to traverse.
///
/// The public vocabulary groups implementation-level reference kinds by what a
/// caller means: `dependencies` are manifest/lockfile declarations,
/// `references` are packages or URLs named by executable commands, and
/// `ci-actions` are third-party CI actions. The older `deps`, `packages`,
/// `urls`, and `ci` spellings remain CLI aliases.
///
/// The three kinds map onto [`fletch`]'s [`RefKind`] taxonomy, so the selection
/// distinguishes how strongly a reference is bound to the artifact:
///
/// - `deps`     → [`RefKind::Dependency`]: a **strict dependency** declared in a
///   manifest or lockfile (`package.json`, `Cargo.lock`, `.SRCINFO depends`).
/// - `packages` → [`RefKind::Command`]: a **package merely mentioned** by an
///   install-command invocation (`npm install foo`, `pip install bar`,
///   `cargo install baz`) — typically injected by a build/lifecycle script
///   rather than pinned in a manifest.
/// - `urls`     → [`RefKind::UrlFetch`]: a **raw URL** with no package identity
///   (a `curl`/`wget` target, a staged download) — the largest exposure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FetchPolicy {
    /// Fetch raw `http(s)` URLs ([`RefKind::UrlFetch`]).
    pub urls: bool,
    /// Fetch packages named by an install command ([`RefKind::Command`]) — e.g.
    /// `npm install foo`. These are *mentioned*, not declared.
    pub packages: bool,
    /// Fetch strict declared dependencies ([`RefKind::Dependency`]) — manifest
    /// and lockfile entries.
    pub deps: bool,
    /// Fetch third-party code declared in a **CI** context — GitHub Actions
    /// `uses:` steps. These are `Dependency`-kind references like any other, but
    /// they run only in CI and never reach an installed artifact, so they are
    /// off by default (a routine `deps` fetch skips them) and enabled only when
    /// auditing CI itself: `--follow=all`, `--follow=ci-actions`, or isomer `ci`.
    pub ci: bool,
    /// How many hops to follow: `1` fetches the references found in the root,
    /// `2` also follows references found *inside* those payloads, and so on.
    pub depth: u8,
    /// Skip fetching a declared dependency older than this many days, judged by
    /// the registry's publish date (looked up cheaply before the artifact is
    /// pulled). `0` disables the gate. Applies to declared dependencies only —
    /// URLs and command-mentioned packages are never age-gated, since their risk
    /// isn't tied to a registry release date.
    pub max_dep_age_days: u32,
    /// Ceiling on *live* fetches triggered by a single scanned file. Cache
    /// hits are always served and never
    /// counted, so this caps only the cold-cache dependency/package fan-out,
    /// never a warm re-run. `0` disables these fetches entirely.
    pub max_file_fetches: usize,
    /// Ceiling on *live* opportunistic raw-URL fetches triggered by a single
    /// scanned file. Cache hits are always served and
    /// never counted. `0` disables opportunistic URL fetching while leaving
    /// declared dependencies and command-mentioned packages unaffected.
    pub max_url_fetches: usize,
    /// Wall-clock ceiling on the whole fetch phase for one scanned artifact.
    /// Checked at group boundaries, so a group already
    /// downloading finishes rather than being torn mid-transfer; references not
    /// reached by then are simply not followed. [`Duration::ZERO`] disables it.
    pub max_duration: Duration,
    /// Ceiling on total bytes fetched on behalf of a single scanned file. The
    /// sweep stops once retrieved bytes cross it.
    pub max_file_bytes: u64,
    /// Follow declared dependencies past the **first** hop — the dependencies
    /// of a fetched dependency, and so on to `depth`.
    ///
    /// Off for an interactive scan. Hop 1 is the artifact's own declared supply
    /// chain, which is the thing being judged; hop 2+ is a transitive closure
    /// that multiplies per hop and is dominated by the long tail of ordinary,
    /// old releases. Each of those costs a registry round trip *before* the age
    /// gate can rule it out, because the publish date is what the lookup is for:
    /// one 2.2 KB manifest sidecar pointing at a Go module drew 398 lookups at
    /// hop 2, of which 398 aged out and none was fetched.
    ///
    /// The dropper chain `--follow-depth 2` exists for runs through URLs and
    /// install-command packages, and those are followed at every hop regardless.
    /// `serve` sets this because it is a cache-population role, where
    /// the transitive tail is the point rather than an overhead.
    pub transitive_deps: bool,
    /// Skip fetching a *dependency* whose name pins it to a platform other than
    /// the host — the `@scope/pkg-<os>-<arch>` native-binary packages (biome,
    /// esbuild, swc, rollup, sharp…) that ship one prebuilt per platform. On a
    /// darwin-arm64 host only the darwin-arm64 variant is pulled; the linux and
    /// windows siblings cannot run locally. `false` audits every platform,
    /// which service/corpus roles enable because they scan on behalf of others.
    /// Applies to fetched dependencies only; a directly-scanned artifact is
    /// always analyzed.
    pub host_platform_only: bool,
    /// Audit development-only locked dependencies; true preserves full coverage.
    pub include_dev_dependencies: bool,
    /// Analyze every pinned release, preserving separate version constraints.
    pub all_versions: bool,
}

impl Default for FetchPolicy {
    fn default() -> Self {
        Self {
            urls: false,
            packages: false,
            deps: false,
            ci: false,
            depth: DEFAULT_FETCH_DEPTH,
            max_dep_age_days: DEFAULT_MAX_DEP_AGE_DAYS,
            max_file_fetches: DEFAULT_MAX_FILE_FETCHES,
            max_url_fetches: DEFAULT_MAX_URL_FETCHES,
            max_file_bytes: DEFAULT_MAX_FILE_SIZE,
            max_duration: DEFAULT_FETCH_TIMEOUT,
            transitive_deps: false,
            host_platform_only: true,
            include_dev_dependencies: true,
            all_versions: false,
        }
    }
}

impl FetchPolicy {
    /// True when at least one kind is selected — the master switch.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.urls || self.packages || self.deps || self.ci
    }

    /// This operator-configured policy, following only the reference kinds
    /// `selected` names, or `None` when `selected` names a kind this policy
    /// does not follow: a request can narrow what the operator allowed, never
    /// widen it. Depth, age, byte, platform and fan-out ceilings stay.
    #[must_use]
    pub const fn restricted_to(self, selected: Self) -> Option<Self> {
        if selected.selection_bits() & !self.selection_bits() != 0 {
            return None;
        }
        Some(Self {
            urls: selected.urls,
            packages: selected.packages,
            deps: selected.deps,
            ci: selected.ci,
            ..self
        })
    }

    /// Compact identity for single-flight keys and structured logs.
    #[must_use]
    pub const fn selection_bits(&self) -> u8 {
        (self.urls as u8)
            | ((self.packages as u8) << 1)
            | ((self.deps as u8) << 2)
            | ((self.ci as u8) << 3)
    }

    /// This selection in the customer-facing `follow` vocabulary, when it has
    /// a name there.
    ///
    /// The inverse of [`Self::parse_follow`], and it exists so a caller never
    /// has to guess which policy produced an answer: whoever files the verdict
    /// files it under the name returned here, and a name that disagreed with
    /// the analysis would file it under the wrong question.
    ///
    /// `None` for a selection the vocabulary cannot spell: `references` moves
    /// `urls` and `packages` together, so a policy built with one and not the
    /// other has no word for it. Saying nothing is right there: an approximate name is worse than an
    /// absent one, because the absent one falls back to the caller's own
    /// resolution while the approximate one silently misfiles.
    #[must_use]
    pub fn follow_name(&self) -> Option<String> {
        if self.urls != self.packages {
            return None;
        }
        let references = self.urls;
        if !references && !self.deps && !self.ci {
            return Some("none".to_owned());
        }
        if references && self.deps && self.ci {
            return Some("all".to_owned());
        }
        // Spelled in the order the customer vocabulary lists them, so one
        // policy has exactly one name and a cache keyed by that name does not
        // split on word order.
        let mut parts = Vec::with_capacity(3);
        if self.deps {
            parts.push("dependencies");
        }
        if references {
            parts.push("references");
        }
        if self.ci {
            parts.push("ci-actions");
        }
        Some(parts.join(","))
    }

    /// Parse the `follow` vocabulary — the one spelling the CLI, `SCAN_FOLLOW`
    /// and the HTTP API share.
    ///
    /// # Errors
    /// When `value` names nothing, an unknown target, or `none` with another.
    pub fn parse_follow(value: &str) -> Result<Self, String> {
        const VALID: &str = "valid: all, dependencies, references, ci-actions, none";
        let mut policy = Self::default();
        let mut saw_kind = false;
        let mut saw_none = false;

        for raw in value.split(',') {
            let kind = raw.trim();
            if kind.is_empty() {
                continue;
            }
            saw_kind = true;
            match kind {
                "none" => saw_none = true,
                "all" => {
                    policy.urls = true;
                    policy.packages = true;
                    policy.deps = true;
                    policy.ci = true;
                }
                "dependencies" => policy.deps = true,
                "references" => {
                    policy.urls = true;
                    policy.packages = true;
                }
                // A CI action is represented as a dependency with CI context,
                // so selecting actions necessarily enables dependency traversal.
                "ci-actions" => {
                    policy.deps = true;
                    policy.ci = true;
                }
                other => return Err(format!("unknown follow target {other:?} ({VALID})")),
            }
        }

        if !saw_kind {
            return Err(format!("empty follow selection ({VALID})"));
        }
        if saw_none && policy.enabled() {
            return Err("none cannot be combined with another follow target".to_string());
        }
        if saw_none {
            return Ok(Self::default());
        }
        if !policy.enabled() {
            return Err(format!("empty follow selection ({VALID})"));
        }
        Ok(policy)
    }

    /// Whether `kind` is selected by this policy. References whose kind is
    /// neither a URL, a command-mentioned package, nor a declared dependency
    /// (e.g. [`RefKind::Repository`] identity) are never fetched.
    #[must_use]
    pub(super) fn wants(&self, kind: RefKind) -> bool {
        match kind {
            RefKind::UrlFetch => self.urls,
            RefKind::Command => self.packages,
            RefKind::Dependency => self.deps,
            _ => false,
        }
    }

    /// Whether `kind` is selected on hop `hop` (0-based). Identical to
    /// [`Self::wants`] except that declared dependencies stop at the first hop
    /// unless [`Self::transitive_deps`] is set — see that field for why.
    #[must_use]
    pub(super) fn wants_at(&self, kind: RefKind, hop: u8) -> bool {
        self.wants(kind) && (self.transitive_deps || hop == 0 || kind != RefKind::Dependency)
    }
}

impl std::str::FromStr for FetchPolicy {
    type Err = String;

    /// See [`FetchPolicy::parse_follow`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_follow(s)
    }
}

/// Canonicalize an OS name segment to its npm token (`process.platform`).
/// Accepts the npm spelling plus the Rust/Go target spellings that appear in
/// cargo platform crates (`windows_x86_64_gnu`) and Go module paths, so a
/// match keys on a genuine `<os>-<arch>` native-binary name rather than an
/// incidental word. `None` for anything that names no OS.
pub(super) fn canonical_os(seg: &str) -> Option<&'static str> {
    Some(match seg {
        "darwin" | "macos" => "darwin",
        "win32" | "windows" => "win32",
        "sunos" | "solaris" | "illumos" => "sunos",
        "linux" => "linux",
        // musl is its own platform: sharp/libvips ship separate `linuxmusl`
        // prebuilts, and neither libc's binaries load on the other's host.
        "linuxmusl" | "musllinux" => "linuxmusl",
        // StackBlitz-style wasm sandbox builds; never a scan host.
        "webcontainers" | "wasi" => "webcontainers",
        "freebsd" => "freebsd",
        "openbsd" => "openbsd",
        "netbsd" => "netbsd",
        "android" => "android",
        "aix" => "aix",
        _ => return None,
    })
}

/// Canonicalize a CPU-architecture segment to its npm token (`process.arch`).
/// Same vocabulary rule as [`canonical_os`].
pub(super) fn canonical_arch(seg: &str) -> Option<&'static str> {
    Some(match seg {
        "x64" | "x8664" | "amd64" => "x64",
        "arm64" | "aarch64" => "arm64",
        "ia32" | "i686" | "i386" | "x86" => "ia32",
        "arm" => "arm",
        "ppc64" => "ppc64",
        "s390x" => "s390x",
        "riscv64" => "riscv64",
        "loong64" => "loong64",
        "mips64el" => "mips64el",
        // Wasm sandbox builds pair with an os token (`freebsd-wasm32`,
        // `webcontainers-wasm32`) and never match a real host arch.
        "wasm32" | "wasm64" => "wasm32",
        _ => return None,
    })
}

/// The host's npm-style `(os, arch)` tokens, mapped from Rust's target
/// constants. An unmapped target yields an empty token, which disables that
/// half of the platform match — fail open, so a dependency is never skipped on a
/// host we can't confidently name.
pub(super) fn host_platform() -> (&'static str, &'static str) {
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        "solaris" | "illumos" => "sunos",
        // A musl build (Alpine workers) is its own platform: glibc prebuilts
        // don't load there and musl prebuilts don't load on glibc hosts, and
        // native packages ship separate `linuxmusl` variants (sharp/libvips).
        "linux" if cfg!(target_env = "musl") => "linuxmusl",
        os @ ("linux" | "freebsd" | "openbsd" | "netbsd" | "android") => os,
        _ => "",
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        "x86" => "ia32",
        arch @ ("arm" | "ppc64" | "s390x" | "riscv64") => arch,
        _ => "",
    };
    (os, arch)
}

/// Whether a dependency reference is a native-binary package pinned to a
/// platform other than `host`. Matches the well-known `<os>-<arch>` (or
/// `<arch>-<os>`) adjacent-segment convention native packages use —
/// `cli-darwin-arm64`, `rollup-linux-x64-gnu`, `@img/sharp-win32-x64` — so a
/// package that names no such pair is treated as portable and kept. Returns
/// `false` when the host platform can't be named (fail open) or for non-PURL
/// locators (a raw URL carries no package identity to place).
pub(super) fn off_host_platform(r: &Reference, host: (&str, &str)) -> bool {
    let (host_os, host_arch) = host;
    if host_os.is_empty() || host_arch.is_empty() {
        return false;
    }
    let RefLocator::Purl(purl) = &r.locator else {
        return false;
    };
    let Some(coordinate) = Coordinate::of(purl) else {
        return false;
    };
    // The package path in lowercase alphanumeric segments
    // (`%40biomejs/cli-darwin-arm64` → [40, biomejs, cli, darwin, arm64]).
    let mut segs: Vec<String> = coordinate
        .path
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase)
        .collect();
    // `x86_64` splits at its underscore; re-join the pair so cargo/Go names
    // (`windows_x86_64_gnu`, `linux_x86_64`) carry one arch token.
    let mut i = 0;
    while i + 1 < segs.len() {
        if segs[i] == "x86" && segs[i + 1] == "64" {
            segs[i] = "x8664".to_string();
            segs.remove(i + 1);
        }
        i += 1;
    }
    // An adjacent os+arch pair (either order) marks a platform-specific package;
    // skip it when either token disagrees with the host.
    for w in segs.windows(2) {
        let (os, arch) = match (canonical_os(&w[0]), canonical_arch(&w[1])) {
            (Some(os), Some(arch)) => (os, arch),
            _ => match (canonical_arch(&w[0]), canonical_os(&w[1])) {
                (Some(arch), Some(os)) => (os, arch),
                _ => continue,
            },
        };
        return os != host_os || arch != host_arch;
    }
    false
}
