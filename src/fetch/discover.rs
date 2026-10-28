//! Finding the references to follow: declared dependencies, downloads named
//! in code, members that carry a download, and the packages vendored
//! alongside.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// A file whose references execute only in CI, never in an installed artifact
/// — a GitHub Actions workflow or composite `action.yml`. Its `uses:` actions
/// and `run:` fetches are third-party code, but they run on the CI runner, so a
/// routine dependency fetch skips them; `--follow=all`/`--follow=ci-actions` opts in.
pub(super) fn is_ci_context(file: &cleave::types::FileAnalysis) -> bool {
    file.file_type == filefacts::FileType::GithubActions.label()
}

/// Whether [`collect_references`] follows references that only ever execute in
/// CI. Auditing an artifact and auditing the pipeline that built it are
/// different questions, and the caller always knows which one it is asking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CiRefs {
    Skip,
    Include,
}

/// References to fetch, grouped by the sha256 of the file that declared them.
#[cfg(test)]
pub(super) fn collect_references(
    report: &AnalysisReport,
    root_path: &Path,
    ci: CiRefs,
) -> Vec<Group> {
    collect_references_from_input(report, root_path, None, ci)
}

/// A reference that was not fetched, and why, as its record.
pub(super) fn skipped_record(
    source_sha: &str,
    reference: &Reference,
    why: Coverage,
) -> FetchRecord {
    let mut record = FetchRecord::terminal(locator(reference).to_owned(), Outcome::Skipped);
    record.source_sha256 = Some(source_sha.to_owned());
    record.source_offset = reference.offset;
    record.kind = reference.kind;
    record.context.clone_from(&reference.context);
    record.coverage.push(why);
    record
}

/// Stamp references as running only in CI: their declaring file is a CI
/// context, so whatever they name never reaches an installed artifact.
pub(super) fn mark_ci_scope(refs: &mut [Reference]) {
    for reference in refs {
        let context = reference
            .context
            .get_or_insert(filefacts::DependencyContext {
                scope: filefacts::DependencyScope::Ci,
                optional: false,
                has_install_script: false,
                installed_path: None,
            });
        context.scope = filefacts::DependencyScope::Ci;
    }
}

/// The packages and Python namespaces a report already holds, so an import
/// that resolves inside the artifact is not fetched from a registry.
pub(super) struct LocalModules {
    pub(super) npm: LocalNpmPackages,
    pub(super) python: PythonNamespaces,
    pub(super) package_contexts: HashMap<String, (String, bool)>,
}

impl LocalModules {
    pub(super) fn of(report: &AnalysisReport) -> Self {
        Self {
            npm: LocalNpmPackages::from_report(report),
            python: PythonNamespaces::from_report(report),
            package_contexts: npm_package_contexts(report),
        }
    }

    /// Whether a Python import in `file` needs no fetch: a type stub imports
    /// nothing at run time, and a namespace the report holds is already here.
    pub(super) fn provides_python(
        &self,
        file: &cleave::types::FileAnalysis,
        reference: &Reference,
    ) -> bool {
        (file.file_type == "python" && file.path.ends_with(".pyi"))
            || (is_python_import_candidate(reference)
                && self.python.resolves(&file.path, &reference.evidence))
    }

    /// Whether npm `package`, imported from `path`, is installed in the report
    /// or provided by the package or framework that contains it.
    pub(super) fn provides_npm(&self, path: &str, package: &str) -> bool {
        self.npm.resolves(path, package)
            || npm_import_is_package_or_framework_provided(path, package, &self.package_contexts)
    }
}

pub(super) fn collect_references_from_input(
    report: &AnalysisReport,
    root_path: &Path,
    root_bytes: Option<&[u8]>,
    ci: CiRefs,
) -> Vec<Group> {
    // Members under a vendored node_modules tree have already been analyzed.
    // Fetching each of their `require("x")` targets again grafted a newer
    // registry release onto the report and repeated work over bytes that came
    // with the sample. Keep hunting absent imports, but resolve local packages
    // first from the package.json members already present in this report.
    let local = LocalModules::of(report);
    // Built once: a lockfile is looked up by path, and a declaring file's
    // nodes by sha, once per file — a scan apiece would be quadratic in the
    // size of an archive.
    let mut by_path: HashMap<&str, &cleave::types::FileAnalysis> = HashMap::new();
    let mut by_sha: HashMap<&str, Vec<&cleave::types::FileAnalysis>> = HashMap::new();
    for file in &report.files {
        by_path.entry(file.path.as_str()).or_insert(file);
        by_sha.entry(file.sha256.as_str()).or_default().push(file);
    }
    let mut local_imports_skipped = 0usize;
    let mut vendored_imports_skipped = 0usize;
    let mut groups: Vec<Group> = Vec::new();
    for file in &report.files {
        // A GitHub Actions workflow is a CI-only context: its `uses:` actions
        // execute in CI, never in an installed artifact. Skip the whole member
        // unless CI auditing was requested (`--follow=all`, `--follow=ci-actions`). The
        // root of a single-file workflow scan is a member here like any other,
        // so this one gate covers it too.
        if ci == CiRefs::Skip && is_ci_context(file) {
            continue;
        }
        let Some(view) = &file.filefacts else {
            continue;
        };
        // Declared references plus the value-driven hunt (npm lifecycle hooks),
        // both from facts the report already carries — so every archive member,
        // not just the root, contributes its references without re-extraction.
        let mut refs = find::references_from_facts(&view.values, &view.references);
        // A co-located lockfile is authoritative; do not borrow pins from
        // other bundled packages. Workspace-only locks remain explicit range
        // resolution until workspace ownership can be established.
        if let Some((directory, _)) = file.path.rsplit_once('/') {
            let lock_name = if file.path.ends_with("/Cargo.toml") {
                Some("Cargo.lock")
            } else if file.path.ends_with("/pyproject.toml") {
                Some("poetry.lock")
            } else {
                None
            };
            if let Some(lock_name) = lock_name {
                let lock_path = format!("{directory}/{lock_name}");
                if let Some(lock) = by_path
                    .get(lock_path.as_str())
                    .and_then(|f| f.filefacts.as_ref())
                {
                    refs = refs
                        .iter()
                        .map(|r| fletch::fetch::prefer_lock_pin(r, &lock.references))
                        .collect();
                }
            }
        }
        // Module-load calls from the member's retained AST symbols — the
        // facts-only import vector, so `require("undeclared-pkg")` inside an
        // archive member is hunted without re-extracting its discarded bytes.
        refs.extend(
            find::import_calls(&file.file_type, &view.symbols)
                .into_iter()
                .chain(find::python_imports(&file.file_type, &view.symbols))
                .chain(static_es_module_refs(
                    &file.path,
                    &file.file_type,
                    &view.values,
                ))
                .filter(|reference| {
                    if local.provides_python(file, reference) {
                        return false;
                    }
                    let Some(package) = npm_import_name(reference) else {
                        return true;
                    };
                    // `node_modules` is the installed dependency graph captured
                    // in the artifact. If one of those files imports a package
                    // absent from that graph, Node fails or takes the module's
                    // own fallback path; it does not download today's registry
                    // release. Declared package.json dependencies are collected
                    // separately, so suppress only inferred import-call hunts.
                    if is_vendored_node_module(&file.path) {
                        vendored_imports_skipped += 1;
                        tracing::debug!(
                            source = %file.path,
                            package,
                            "import originates in vendored node_modules; external fetch avoided"
                        );
                        return false;
                    }
                    let present = local.provides_npm(&file.path, &package);
                    if present {
                        local_imports_skipped += 1;
                        tracing::debug!(
                            source = %file.path,
                            package,
                            "import resolves to vendored node_modules; skipping external fetch"
                        );
                    }
                    !present
                }),
        );
        refs.extend(find::fetch_calls(&file.file_type, &view.symbols));
        refs.extend(find::discover_batch_programs(&file.file_type, &view.values).references);
        if refs.is_empty() {
            continue;
        }
        if is_ci_context(file) {
            mark_ci_scope(&mut refs);
        }
        groups.push((file.sha256.clone(), refs));
    }
    if local_imports_skipped > 0 {
        tracing::info!(
            local_imports_skipped,
            "vendored imports already present in report; external refetch avoided"
        );
    }
    if vendored_imports_skipped > 0 {
        tracing::info!(
            vendored_imports_skipped,
            "imports from captured node_modules kept inside artifact boundary"
        );
    }

    // The root sample's imperative hunt (curl|sh, `npm install` in a RUN, a URL
    // in a shell variable) needs its raw text, which the report doesn't carry —
    // read it back from disk for small text-ish roots and merge, deduping
    // against the declared references already collected for the root. The hunt
    // reads bytes the loop above never opened, so it repeats that loop's CI
    // gate — a workflow's raw text re-discovers the `uses:` actions just
    // skipped.
    if let Some(root) = report.files.first()
        && (ci == CiRefs::Include || !is_ci_context(root))
        && root.size <= ROOT_HUNT_MAX_BYTES
        && let Some(bytes) = root_bytes
            .map(std::borrow::Cow::Borrowed)
            .or_else(|| std::fs::read(root_path).ok().map(std::borrow::Cow::Owned))
    {
        let name_path = if root_bytes.is_some() {
            Path::new(&root.path)
        } else {
            root_path
        };
        let name = name_path
            .file_name()
            .map_or_else(|| name_path.to_string_lossy(), |n| n.to_string_lossy());
        // A provenance document names one artifact and catalogues many. The
        // string hunt cannot tell those apart, so it is replaced by the subject
        // this document is *about* — see `provenance_subject`.
        let mut hunted = if is_provenance_document(&root.file_type, &name) {
            provenance_subject(&bytes).into_iter().collect()
        } else {
            find::references_in_bytes(&bytes, &name)
        };
        if is_ci_context(root) {
            mark_ci_scope(&mut hunted);
        }
        if !hunted.is_empty() {
            merge_into_root(&mut groups, &root.sha256, hunted);
        }
    }
    hunt_download_members(report, root_path, root_bytes, ci, &mut groups);
    // Filefacts owns Go's module/workspace semantics. Include raw root hunts
    // in the inputs so they cannot reintroduce an unreconciled declaration.
    let mut group_refs: HashMap<&str, &[Reference]> = HashMap::new();
    for (sha, refs) in &groups {
        group_refs.entry(sha.as_str()).or_insert(refs.as_slice());
    }
    let go_members: Vec<_> = report
        .files
        .iter()
        .map(|file| filefacts::ReferenceMember {
            path: &file.path,
            references: group_refs.get(file.sha256.as_str()).copied().unwrap_or(&[]),
        })
        .collect();
    let go_context = filefacts::go_dependency_context(&go_members);
    for (sha, refs) in &mut groups {
        let contexts: Vec<_> = by_sha
            .get(sha.as_str())
            .into_iter()
            .flatten()
            .filter_map(|f| go_context.get(&f.path))
            .collect();
        if !contexts.is_empty() {
            // Identical manifest bytes may occur under different workspaces.
            // Keep every contextual edge; never let the last path win.
            refs.clear();
            for resolved in contexts {
                for reference in resolved {
                    if !refs.contains(reference) {
                        refs.push(reference.clone());
                    }
                }
            }
        }
    }
    groups
}

/// Most archive members a single scan re-reads for a text hunt. Each read
/// decompresses the root archive up to that member, so a crafted archive
/// packing many flagged members cannot turn the hunt into a decompression
/// storm; the members past the cap keep their declared references.
pub(super) const MEMBER_HUNT_MAX: usize = 16;
/// Largest member a text hunt reads. Bundled editor extensions commonly exceed
/// [`ROOT_HUNT_MAX_BYTES`]; a member is still only admitted by an
/// execution/download finding. The cap applies to the bytes actually
/// re-extracted, not just the size the report recorded: an archive can carry
/// two entries under one name, and re-extraction by name may return the other.
pub(super) const MEMBER_HUNT_MAX_BYTES: u64 = 16 * 1024 * 1024;

/// cleave trait ids this module reads. cleave owns the taxonomy; naming every
/// id here keeps the module's dependence on it in one place.
pub(super) const TRAIT_DROPPER: &str = "objectives/command-and-control/dropper/";
pub(super) const TRAIT_DROPPER_EXECUTION: &str =
    "objectives/command-and-control/dropper/execution/";
pub(super) const TRAIT_STEGO_LOADER: &str =
    "objectives/command-and-control/dropper/execution/stego-loader";
pub(super) const TRAIT_STEGANOGRAPHY: &str = "objectives/anti-static/obfuscation/steganography/";
pub(super) const TRAIT_SHELL_PIPELINE: &str = "micro-behaviors/process/create/shell/pipeline";
pub(super) const TRAIT_PROCESS_CREATE: &str = "micro-behaviors/process/create/";
pub(super) const TRAIT_AUTOINSTALL: &str = "objectives/execution/autoinstall/";
pub(super) const TRAIT_GITHUB_RUNNER_SHELL: &str =
    "micro-behaviors/process/create/shell/bridge::github-runner-shell-execution";
pub(super) const TRAIT_IMAGE_FILE_URL: &str =
    "micro-behaviors/communications/http/url/path::image-file-url";
pub(super) const TRAIT_BROWSER_PAC_URL: &str =
    "micro-behaviors/os/network/webextension::chrome-pac-script-literal-url";
pub(super) const TRAIT_SECURITY_HOLD_RECORD: &str = "registry-security-hold-record";

/// Trait-id namespaces whose findings show that a file downloads something in
/// order to run it: dropper objectives, a pipeline into a shell or an
/// auto-install / CI-runner shell bridge, a browser PAC configuration URL,
/// and a process spawn of `curl`/`wget`. Ordered most- to least-specific; the first match
/// names the reason a member was hunted.
pub(super) fn download_intent(findings: &[Finding]) -> Option<&str> {
    let rank = |id: &str| {
        if id.starts_with(TRAIT_DROPPER) {
            Some(0)
        } else if id.starts_with(TRAIT_SHELL_PIPELINE)
            || id.starts_with(TRAIT_AUTOINSTALL)
            || id == TRAIT_GITHUB_RUNNER_SHELL
        {
            Some(1)
        } else if id == TRAIT_BROWSER_PAC_URL {
            // A PAC URL identifies executable proxy configuration, including
            // inside extension archives whose source bytes were discarded.
            Some(2)
        } else if id.starts_with(TRAIT_PROCESS_CREATE)
            && (id.contains("curl") || id.contains("wget"))
        {
            Some(2)
        } else {
            None
        }
    };
    findings
        .iter()
        .filter_map(|f| rank(&f.id).map(|r| (r, f.id.as_str())))
        .min_by_key(|(r, _)| *r)
        .map(|(_, id)| id)
}

/// Give archive members that show download intent the raw-text hunt the root
/// gets. A member's bytes are discarded after analysis, so a URL a source file
/// hands to a spawned `curl` — in any language, including ones with no
/// dedicated recognizer — never reached the work list: only the root's bytes
/// are re-read. The member's own findings are the gate, so an ordinary source
/// tree costs nothing; a flagged member is re-extracted from the root archive
/// and hunted like a root. Its references keep their recognizer as `source`;
/// the trait that justified the hunt is logged with it.
pub(super) fn hunt_download_members(
    report: &AnalysisReport,
    root_path: &Path,
    root_bytes: Option<&[u8]>,
    ci: CiRefs,
    groups: &mut Vec<Group>,
) {
    let Some((root, members)) = report.files.split_first() else {
        return;
    };
    let prefix = format!("{}!!", root.path);
    let mut hunted_members = 0usize;
    // Cleave extracts from a path; materialize in-memory input only if a member
    // actually needs a text hunt, and reuse the temporary file for all members.
    let mut staged = None;
    for file in members {
        if ci == CiRefs::Skip && is_ci_context(file) {
            continue;
        }
        let Some(trigger) = download_intent(&file.findings) else {
            continue;
        };
        // Only a direct member of the root archive can be re-read: a nested
        // archive member (`!!`) or a decoded layer (`##`) has no path in it.
        let Some(member) = file
            .path
            .strip_prefix(&prefix)
            .filter(|m| !m.contains("!!") && !m.contains("##"))
        else {
            tracing::debug!(
                member = %file.path,
                trigger,
                "download intent in a nested or decoded member; bytes not re-readable, text hunt skipped"
            );
            continue;
        };
        if file.size > MEMBER_HUNT_MAX_BYTES {
            tracing::debug!(
                member,
                trigger,
                size = file.size,
                "download intent in an oversized member; text hunt skipped"
            );
            continue;
        }
        if hunted_members == MEMBER_HUNT_MAX {
            tracing::info!(
                cap = MEMBER_HUNT_MAX,
                "member text-hunt cap reached; remaining flagged members keep declared references only"
            );
            break;
        }
        hunted_members += 1;
        let archive_path = if let Some(data) = root_bytes {
            if staged.is_none() {
                let stage = (|| -> std::io::Result<tempfile::NamedTempFile> {
                    use std::io::Write;
                    let mut file = tempfile::NamedTempFile::new()?;
                    file.write_all(data)?;
                    Ok(file)
                })();
                match stage {
                    Ok(file) => staged = Some(file),
                    Err(e) => {
                        tracing::warn!(
                            member,
                            trigger,
                            "staging archive for member hunt failed: {e}"
                        );
                        break;
                    }
                }
            }
            let Some(file) = staged.as_ref() else {
                break;
            };
            file.path()
        } else {
            root_path
        };
        let bytes = match cleave::extract_member(archive_path, member) {
            Ok(Some(bytes)) if bytes.len() as u64 <= MEMBER_HUNT_MAX_BYTES => bytes,
            Ok(Some(bytes)) => {
                tracing::warn!(
                    member,
                    trigger,
                    recorded = file.size,
                    extracted = bytes.len(),
                    "re-read member exceeds the hunt ceiling (duplicate entry name?); text hunt skipped"
                );
                continue;
            }
            Ok(None) => {
                tracing::debug!(
                    member,
                    trigger,
                    "flagged member not re-readable from the root archive; text hunt skipped"
                );
                continue;
            }
            Err(e) => {
                tracing::warn!(member, trigger, "re-reading flagged member failed: {e:#}");
                continue;
            }
        };
        let mut hunted = find::references_in_bytes(&bytes, member);
        if trigger == TRAIT_BROWSER_PAC_URL {
            // The lightweight trait only gates re-reading. Require the
            // API/mode/binding check before adding a PAC code dependency;
            // generic URL strings in comments or overwritten settings do
            // not become new archive fetches through this gate.
            hunted.retain(|reference| reference.source == "ast-browser-pac-url");
        }
        tracing::info!(
            member,
            trigger,
            references = hunted.len(),
            "member shows download intent; hunted its text for references"
        );
        if !hunted.is_empty() {
            merge_into_root(groups, &file.sha256, hunted);
        }
    }
}

/// Image extensions a stego loader carves its payload out of.
pub(super) const IMAGE_CARRIER_EXTENSIONS: &[&str] =
    &["avif", "bmp", "gif", "jpeg", "jpg", "png", "webp"];

/// Whether a file's findings show it pulls a payload out of an image: a
/// stego-loader or steganography trait, or an image URL alongside a dropper
/// execution trait. Returns the trait that says so.
pub(super) fn image_carrier_evidence(findings: &[Finding]) -> Option<&str> {
    let ids = || findings.iter().map(|f| f.id.as_str());
    ids()
        .find(|id| id.starts_with(TRAIT_STEGO_LOADER) || id.starts_with(TRAIT_STEGANOGRAPHY))
        .or_else(|| {
            ids()
                .any(|id| id == TRAIT_IMAGE_FILE_URL)
                .then(|| ids().find(|id| id.starts_with(TRAIT_DROPPER_EXECUTION)))
                .flatten()
        })
}

/// Image URLs that are payload carriers rather than page assets. An image URL
/// is normally skipped as a site resource (see [`NON_PAYLOAD_URL_EXTENSIONS`]);
/// it is followed only when the file that names it also carries image-carrier
/// evidence, so a README badge never costs a request but a loader's
/// `screenshot.png` with a PE appended is fetched and analyzed.
pub(super) fn image_carrier_urls(
    report: &AnalysisReport,
    groups: &[(String, Vec<Reference>)],
) -> HashSet<String> {
    let mut evidence_by_sha: HashMap<&str, (&cleave::types::FileAnalysis, &str)> = HashMap::new();
    for file in &report.files {
        if let Some(evidence) = image_carrier_evidence(&file.findings) {
            evidence_by_sha
                .entry(file.sha256.as_str())
                .or_insert((file, evidence));
        }
    }
    let mut carriers = HashSet::new();
    for (sha, refs) in groups {
        let Some(&(file, evidence)) = evidence_by_sha.get(sha.as_str()) else {
            continue;
        };
        for reference in refs {
            let RefLocator::Url(url) = &reference.locator else {
                continue;
            };
            if reference.kind != RefKind::UrlFetch || !has_image_extension(url) {
                continue;
            }
            tracing::info!(
                url = %url,
                source = %file.path,
                evidence,
                "image URL named by a stego loader; following it as a payload carrier"
            );
            carriers.insert(url.clone());
        }
    }
    carriers
}

/// Whether a URL's path ends in an image extension.
pub(super) fn has_image_extension(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    parsed.path().rsplit_once('.').is_some_and(|(_, ext)| {
        IMAGE_CARRIER_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
    })
}

/// Whether this root is one of our own provenance records rather than a
/// collected artifact: hopper's `*.forage.json` collection sidecar, or the
/// normalized `*.registry.json` a fetch materializes.
pub(super) fn is_provenance_document(file_type: &str, name: &str) -> bool {
    file_type == filefacts::FileType::Registry.label()
        || name
            .rsplit_once(".forage.")
            .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case("json"))
}

/// The single artifact a provenance document is *about*, as a pinned reference.
///
/// These documents embed the provider's verbatim response, and for PyPI that is
/// the project's whole release catalogue: one 179 KB `diffusers` sidecar carries
/// 191 `files.pythonhosted.org` URLs covering every version ever published. The
/// string hunt has no way to tell the subject from the catalogue — it recovered
/// all 199 URLs and started pulling `diffusers` releases from 0.0.1 upward until
/// the URL budget stopped it. Mining our own cached metadata for dropper
/// candidates is the bug; the document already states its subject, so read that
/// instead of guessing from `strings`.
///
/// Parsed here rather than from filefacts' `values` because a large sidecar
/// exceeds cleave's JSON parse limit (76 KB) while still being small enough to
/// hunt, so the facts view cannot be relied on for exactly the documents that
/// carry the biggest catalogues.
pub(super) fn provenance_subject(bytes: &[u8]) -> Option<Reference> {
    let doc: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let url = doc
        .pointer("/fetch/url")
        .or_else(|| doc.pointer("/registry/url"))
        .and_then(serde_json::Value::as_str)
        .filter(|u| !u.is_empty())?;
    // The recorded digest pins the fetch: this document exists because those
    // exact bytes were collected, so a mismatch is a substitution worth failing.
    let pinned_hash = doc
        .pointer("/artifact/sha256")
        .and_then(serde_json::Value::as_str)
        .filter(|d| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(|d| fletch::PinnedHash {
            algo: fletch::HashAlgo::Sha256,
            value: d.to_ascii_lowercase(),
        });
    let content_sha256 = pinned_hash.as_ref().map(|p| p.value.clone());
    let mut reference = Reference::new(
        RefLocator::Url(url.to_string()),
        RefKind::UrlFetch,
        "forage.fetch.url",
        url,
    );
    reference.pinned_hash = pinned_hash;
    reference.content_sha256 = content_sha256;
    Some(reference)
}

pub(super) fn is_vendored_node_module(path: &str) -> bool {
    path.split(['/', '\\'])
        .any(|component| component == "node_modules")
}

/// Package roots physically present under `node_modules`, keyed by their full
/// virtual path. Only roots with an analyzed package.json enter the index, so
/// an incidental path component named node_modules is not enough.
#[derive(Default)]
pub(super) struct LocalNpmPackages {
    pub(super) roots: HashSet<String>,
    pub(super) declarations: HashMap<String, (String, String)>,
}

impl LocalNpmPackages {
    pub(super) fn from_report(report: &AnalysisReport) -> Self {
        let roots = report
            .files
            .iter()
            .filter_map(|file| npm_package_root(&file.path))
            .collect();
        let mut declarations = HashMap::new();
        let paths: HashSet<_> = report
            .files
            .iter()
            .filter(|f| !f.analysis_gaps.is_empty())
            .map(|f| f.path.replace('\\', "/"))
            .collect();
        let analyzed: HashSet<_> = report
            .files
            .iter()
            .filter(|f| f.analysis_gaps.is_empty())
            .map(|f| f.path.replace('\\', "/"))
            .collect();
        for file in &report.files {
            let Some(root) = npm_package_root(&file.path) else {
                continue;
            };
            let Some(view) = &file.filefacts else {
                continue;
            };
            let Some(npm) = view.values.get("npm") else {
                continue;
            };
            let Some(name) = npm.get("name").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let Some(version) = npm.get("version").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let targets: Vec<_> = view
                .references
                .iter()
                .filter_map(|r| match &r.locator {
                    RefLocator::Path(path)
                        if r.kind == RefKind::Local && r.source.starts_with("package.json:") =>
                    {
                        Some(path)
                    }
                    _ => None,
                })
                .collect();
            // A manifest/README alone is not supplied executable code. Require
            // a captured entry point and all declared entry points, with no
            // incomplete member under this package. Wildcard-only exports and
            // implicit index.js deliberately remain external until established.
            if targets.is_empty()
                || !file.analysis_gaps.is_empty()
                || paths
                    .iter()
                    .any(|path| path.starts_with(&format!("{root}/")))
            {
                continue;
            }
            let complete = targets.iter().all(|target| {
                let Some(path) = safe_package_path(&root, target) else {
                    return false;
                };
                analyzed.contains(&path)
                    || (std::path::Path::new(&path).extension().is_none()
                        && [".js", ".json", "/index.js"]
                            .iter()
                            .any(|ext| analyzed.contains(&format!("{path}{ext}"))))
            });
            if complete {
                declarations.insert(root, (name.to_owned(), version.to_owned()));
            }
        }
        Self {
            roots,
            declarations,
        }
    }

    pub(super) fn declared_coverage(
        &self,
        report: &AnalysisReport,
        source_sha: &str,
        reference: &Reference,
    ) -> Option<String> {
        if reference.kind != RefKind::Dependency {
            return None;
        }
        let coordinate = Coordinate::of(locator(reference)).filter(|c| c.typ == "npm")?;
        let version = coordinate.version?;
        let package = npm_import_name(reference)?;
        let sources: Vec<_> = report
            .files
            .iter()
            .filter(|f| f.sha256 == source_sha)
            .collect();
        if sources.is_empty() {
            return None;
        }
        let mut covered = Vec::new();
        for source in sources {
            let normalized = source.path.replace('\\', "/");
            let directory = virtual_parent(&normalized)?;
            let root = if let Some(installed) = reference
                .context
                .as_ref()
                .and_then(|c| c.installed_path.as_ref())
            {
                safe_package_path(&directory, installed)?
            } else {
                let mut dir = directory;
                loop {
                    let candidate = safe_package_path(&dir, &format!("node_modules/{package}"))?;
                    // A nearer installation shadows ancestors, even if it is
                    // incomplete or has a different version.
                    if self.roots.contains(&candidate) {
                        break candidate;
                    }
                    dir = virtual_parent(&dir)?;
                }
            };
            let (name, supplied_version) = self.declarations.get(&root)?;
            if name != &package || supplied_version != version {
                return None;
            }
            covered.push(root);
        }
        Some(covered.join(", "))
    }

    /// Mirror Node's package-level lookup: walk upward from the importing
    /// file, looking for `node_modules/<package>`. Exports and subpaths do not
    /// matter here because fletch retrieves whole packages.
    pub(super) fn resolves(&self, source_path: &str, package: &str) -> bool {
        let normalized = source_path.replace('\\', "/");
        let Some((mut dir, _)) = normalized.rsplit_once('/') else {
            return false;
        };
        loop {
            if self
                .roots
                .contains(&format!("{dir}/node_modules/{package}"))
            {
                return true;
            }
            let Some((parent, _)) = dir.rsplit_once('/') else {
                return false;
            };
            dir = parent;
        }
    }
}

/// A virtual member's parent stops at its archive boundary.
pub(super) fn virtual_parent(path: &str) -> Option<String> {
    if let Some((archive, member)) = path.rsplit_once("!!") {
        if member.is_empty() {
            return None;
        }
        return Some(match member.rsplit_once('/') {
            Some((parent, _)) => format!("{archive}!!{parent}"),
            None => format!("{archive}!!"),
        });
    }
    path.rsplit_once('/').map(|(parent, _)| parent.to_owned())
}

pub(super) fn safe_package_path(root: &str, target: &str) -> Option<String> {
    let target = target.replace('\\', "/");
    if target.starts_with('/') || target.contains([':', '!', '*']) {
        return None;
    }
    let mut components = Vec::new();
    for part in target.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                components.pop()?;
            }
            other => components.push(other),
        }
    }
    (!components.is_empty()).then(|| {
        let separator = if root.ends_with("!!") { "" } else { "/" };
        format!("{root}{separator}{}", components.join("/"))
    })
}

/// Return the virtual package root for an analyzed node_modules/package.json.
pub(super) fn npm_package_root(path: &str) -> Option<String> {
    let normalized = path.replace('\\', "/");
    let root = normalized.strip_suffix("/package.json")?;
    let (_, package) = root
        .rsplit_once("/node_modules/")
        .or_else(|| root.rsplit_once("!!node_modules/"))?;
    let mut parts = package.split('/');
    let first = parts.next()?;
    let valid = if let Some(scope) = first.strip_prefix('@') {
        let name = parts.next().unwrap_or_default();
        !scope.is_empty() && !name.is_empty() && parts.next().is_none()
    } else {
        !first.is_empty() && parts.next().is_none()
    };
    valid.then(|| root.to_string())
}

/// Extract the npm package name from a reference produced by
/// `find::import_calls`. Such PURLs are normally versionless; tolerating a
/// version keeps this correct if fletch later learns one from the symbol.
pub(super) fn npm_import_name(reference: &Reference) -> Option<String> {
    let RefLocator::Purl(purl) = &reference.locator else {
        return None;
    };
    let coordinate = Coordinate::of(purl).filter(|c| c.typ == "npm")?;
    Some(
        coordinate
            .path
            .replace("%40", "@")
            .replace("%2F", "/")
            .replace("%2f", "/"),
    )
}

/// Merge the root's hunted references into its group (creating it if the root
/// declared none), skipping any locator already present.
pub(super) fn merge_into_root(groups: &mut Vec<Group>, root_sha: &str, hunted: Vec<Reference>) {
    if !groups.iter().any(|(sha, _)| sha == root_sha) {
        groups.push((root_sha.to_string(), Vec::new()));
    }
    let Some((_, group)) = groups.iter_mut().find(|(sha, _)| sha == root_sha) else {
        return; // unreachable: just ensured the group exists
    };
    let mut seen: HashSet<String> = group.iter().map(|r| locator(r).to_owned()).collect();
    for r in hunted {
        if r.kind == RefKind::Undefined {
            if !group.contains(&r) {
                group.push(r);
            }
            continue;
        }
        if seen.insert(locator(&r).to_owned()) {
            group.push(r);
        }
    }
}

/// Scope follows the edge into a dependency tree: runtime imports of a
/// development tool are still development-only work for the scanned artifact.
pub(super) fn inherit_dependency_context(parent: &Reference, groups: &mut [Group]) {
    let Some(parent) = &parent.context else {
        return;
    };
    for (_, references) in groups {
        for reference in references {
            let context = reference
                .context
                .get_or_insert(filefacts::DependencyContext {
                    scope: filefacts::DependencyScope::Runtime,
                    optional: false,
                    has_install_script: false,
                    installed_path: None,
                });
            use filefacts::DependencyScope;
            if parent.scope == DependencyScope::Ci
                || (parent.scope == DependencyScope::Development
                    && context.scope != DependencyScope::Ci)
                || (parent.scope == DependencyScope::Build
                    && context.scope == DependencyScope::Runtime)
            {
                context.scope = parent.scope;
            }
            context.optional |= parent.optional;
        }
    }
}
