//! Reconciling imported modules with what a package declares: an import
//! nothing declares is a phantom dependency, and one the package itself
//! provides is not a dependency at all.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};

use cleave::{AnalysisOptions, AnalysisReport};
use filefacts::FileType;
use fletch::fetch::{CandidateSource, Coverage, FetchRecord};
use fletch::{RefKind, RefLocator, Reference, find};

use super::discover::{PYPROJECT_LOCKS, ancestors, npm_import_name, virtual_parent};
use super::render::delivered_bytes;
use super::{Group, report_file_type};

/// Turn parser-observed static ES module edges into the same package
/// references as literal dynamic imports. Type-only edges were excluded by
/// filefacts; declaration files never execute even when they omit `type`.
pub(super) fn static_es_module_refs(
    path: &str,
    file_type: filefacts::FileType,
    values: &serde_json::Value,
) -> Vec<Reference> {
    if !matches!(
        file_type,
        filefacts::FileType::JavaScript | filefacts::FileType::TypeScript
    ) || path.ends_with(".d.ts")
        || path.ends_with(".d.cts")
        || path.ends_with(".d.mts")
    {
        return Vec::new();
    }
    let Some(rows) = values
        .pointer("/source/js/runtime_imports")
        .and_then(serde_json::Value::as_array)
    else {
        return Vec::new();
    };
    let symbols: Vec<filefacts::Symbol> = rows
        .iter()
        .filter_map(|row| {
            let specifier = row.get("specifier")?.as_str()?;
            Some(filefacts::Symbol::Call {
                target: Some("import".into()),
                args: vec![filefacts::Arg::String {
                    value: specifier.to_owned(),
                }],
                offset: row.get("offset").and_then(serde_json::Value::as_u64),
            })
        })
        .collect();
    let mut refs = find::import_calls(file_type, &symbols);
    for reference in &mut refs {
        reference.source = "ast-import".into();
    }
    refs
}

/// Module roots visibly supplied at an archive's import root. Keep archive
/// boundaries separate: a fixture or nested wheel cannot satisfy its parent.
/// Presence prevents an unrelated registry refetch; it does not prove a module
/// executes successfully, or that arbitrary sys.path modifications are modeled.
#[derive(Default)]
pub(super) struct PythonNamespaces(HashMap<String, HashSet<String>>);

impl PythonNamespaces {
    pub(super) fn from_report(report: &AnalysisReport) -> Self {
        let mut roots: HashMap<String, HashSet<String>> = HashMap::new();
        for file in &report.files {
            let Some((archive, member)) = file.path.rsplit_once("!!") else {
                continue;
            };
            if member.contains("##") || !python_runtime_module_file(member) {
                continue;
            }
            let first = member.split('/').next().unwrap_or_default();
            let module = if member.contains('/') {
                first
            } else {
                first.split('.').next().unwrap_or_default()
            };
            // A Python identifier: what an `import` could name.
            let mut chars = module.chars();
            let identifier = chars
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
            if identifier {
                roots
                    .entry(archive.to_owned())
                    .or_default()
                    .insert(module.to_owned());
            }
        }
        Self(roots)
    }

    pub(super) fn resolves(&self, path: &str, module: &str) -> bool {
        let Some((archive, _)) = path.rsplit_once("!!") else {
            return false;
        };
        self.0
            .get(archive)
            .is_some_and(|roots| roots.contains(module.split('.').next().unwrap_or_default()))
    }
}

fn python_runtime_module_file(path: &str) -> bool {
    path.ends_with(".py") || path.ends_with(".so") || path.ends_with(".pyd")
}

pub(super) fn is_python_import_candidate(reference: &Reference) -> bool {
    reference.source == "ast-python-import-candidate"
}

/// A wheel's root namespace is evidence for a possible provider, not proof
/// of the imported runtime's distribution, version, or import reachability.
pub(super) fn python_candidate_provides_namespace(
    reference: &Reference,
    report: &AnalysisReport,
) -> bool {
    let Some(root) = report.files.first() else {
        return false;
    };
    if report_file_type(&root.file_type) != FileType::Whl {
        return false;
    }
    let module = reference.evidence.split('.').next().unwrap_or_default();
    let prefix = format!("{}!!", root.path);
    report.files.iter().any(|file| {
        let Some(member) = file.path.strip_prefix(&prefix) else {
            return false;
        };
        if member.contains("!!") || member.contains("##") || !python_runtime_module_file(member) {
            return false;
        }
        member.split('/').next().is_some_and(|first| {
            if member.contains('/') {
                first == module
            } else {
                first.split('.').next() == Some(module)
            }
        })
    })
}

/// Severe observations about one candidate's code or registry metadata.
pub(super) fn candidate_static_issues(
    report: &AnalysisReport,
) -> std::collections::BTreeSet<String> {
    report
        .files
        .iter()
        .flat_map(|file| &file.findings)
        .filter(|finding| finding.crit >= cleave::Criticality::Suspicious)
        .map(|finding| {
            format!(
                "{}:{}",
                if finding.crit == cleave::Criticality::Hostile {
                    "hostile"
                } else {
                    "suspicious"
                },
                finding.id
            )
        })
        .collect()
}

/// Do not graft an unrelated same-named PyPI project onto the importer.
/// A namespace check still leaves the installed distribution/version unknown.
pub(super) fn annotate_python_import_candidate(
    record: &mut FetchRecord,
    reference: &Reference,
    report: Option<&AnalysisReport>,
    registry_issues: Option<&std::collections::BTreeSet<String>>,
) -> bool {
    let verified = delivered_bytes(record)
        && report.is_some_and(|report| python_candidate_provides_namespace(reference, report));
    if verified {
        record.coverage.push(Coverage::Candidate {
            source: CandidateSource::PythonImport,
        });
    } else {
        record.coverage.push(Coverage::Unavailable {
            reason: Some(
                "Python import provider unresolved; no candidate namespace verified".into(),
            ),
        });
        // Keep severe observations about the rejected *candidate* reviewable
        // without turning them into findings about the importing artifact.
        let mut issues = report.map(candidate_static_issues).unwrap_or_default();
        if let Some(registry_issues) = registry_issues {
            issues.extend(registry_issues.iter().cloned());
        }
        if !issues.is_empty() {
            record.coverage.push(Coverage::CandidateFlagged {
                findings: issues.into_iter().collect(),
            });
        }
    }
    verified
}

/// The package a `package.json` directory belongs to, for the imports it
/// satisfies without a registry: its own name, and an Ember app's
/// framework-provided modules.
#[derive(Debug, Clone)]
pub(super) struct PackageContext {
    pub(super) name: String,
    /// Depends on `ember-source`, whose `@ember/*` and `@glimmer/tracking`
    /// modules have no independent registry package.
    pub(super) ember: bool,
}

/// A package can import itself, or framework-provided modules that have no
/// independent registry package. Use only the nearest manifest's context.
pub(super) fn npm_package_contexts(report: &AnalysisReport) -> HashMap<String, PackageContext> {
    let mut result = HashMap::new();
    for file in &report.files {
        // Keyed as `virtual_parent` spells directories: an archive's root is
        // `x.zip!!`, so its manifest is `x.zip!!package.json`.
        let Some(dir) = file.path.strip_suffix("/package.json").or_else(|| {
            file.path
                .strip_suffix("package.json")
                .filter(|dir| dir.ends_with("!!"))
        }) else {
            continue;
        };
        let Some(name) = file
            .identity
            .as_ref()
            .and_then(|i| i.name.as_ref())
            .map(|c| c.value.clone())
        else {
            continue;
        };
        let ember = file.filefacts.as_ref().is_some_and(|view| {
            view.references.iter().any(|r| {
                r.kind == RefKind::Dependency
                    && npm_import_name(r).as_deref() == Some("ember-source")
            })
        });
        result.insert(dir.to_owned(), PackageContext { name, ember });
    }
    result
}

pub(super) fn npm_import_is_package_or_framework_provided(
    path: &str,
    package: &str,
    contexts: &HashMap<String, PackageContext>,
) -> bool {
    ancestors(path)
        .find_map(|dir| contexts.get(&dir))
        .is_some_and(|context| {
            package == context.name
                || (context.ember
                    && (package.starts_with("@ember/") || package == "@glimmer/tracking"))
        })
}

/// Match imports to the nearest manifest in this hop. Shared source hashes
/// must agree on the declaration across every occurrence.
pub(super) fn declared_npm_imports_in_hop(
    report: &AnalysisReport,
    worklist: &[Group],
) -> HashMap<String, HashMap<String, Reference>> {
    let manifests: HashMap<&str, &cleave::types::FileAnalysis> = report
        .files
        .iter()
        .filter(|f| f.path.ends_with("/package.json") || f.path.ends_with("!!package.json"))
        .map(|f| (f.path.as_str(), f))
        .collect();
    let mut declarations: HashMap<&str, HashMap<String, Reference>> = HashMap::new();
    for (sha, refs) in worklist {
        for r in refs.iter().filter(|r| {
            r.kind == RefKind::Dependency
                && !r.evidence.contains("npm:")
                && !r
                    .context
                    .as_ref()
                    .is_some_and(|c| c.scope == filefacts::DependencyScope::Development)
        }) {
            if let Some(name) = npm_import_name(r) {
                declarations
                    .entry(sha.as_str())
                    .or_default()
                    .insert(name, r.clone());
            }
        }
    }
    let mut result: HashMap<String, HashMap<String, Reference>> = HashMap::new();
    for file in &report.files {
        if !matches!(
            report_file_type(&file.file_type),
            FileType::JavaScript | FileType::TypeScript
        ) {
            continue;
        }
        let names = ancestors(&file.path)
            .find_map(|dir| {
                let separator = if dir.ends_with("!!") { "" } else { "/" };
                let manifest = manifests.get(format!("{dir}{separator}package.json").as_str())?;
                Some(declarations.get(manifest.sha256.as_str()))
            })
            .flatten();
        match result.entry(file.sha256.clone()) {
            Entry::Occupied(mut existing) => existing.get_mut().retain(|name, reference| {
                names.and_then(|names| names.get(name)) == Some(reference)
            }),
            Entry::Vacant(slot) => {
                slot.insert(names.cloned().unwrap_or_default());
            }
        }
    }
    result
}

fn python_distribution_name(reference: &Reference) -> Option<String> {
    let RefLocator::Purl(raw) = &reference.locator else {
        return None;
    };
    let purl = fletch::purl::Purl::parse(raw).ok()?;
    if purl.typ() != "pypi" || !purl.namespace().is_empty() {
        return None;
    }
    let mut name = String::new();
    for ch in purl.name().chars() {
        let ch = if matches!(ch, '_' | '.' | '-') {
            '-'
        } else {
            ch.to_ascii_lowercase()
        };
        if ch != '-' || !name.ends_with('-') {
            name.push(ch);
        }
    }
    Some(name)
}

/// Python declarations, by owning package directory and then by normalized
/// distribution name, each list reconciled with the lock pins beside it.
type PythonOwners = HashMap<String, HashMap<String, Vec<Reference>>>;

/// The package contexts one Python source occurs in.
#[derive(Default)]
struct PythonContext {
    /// How many times the source occurs.
    occurrences: usize,
    /// What those occurrences declare, by distribution name.
    names: HashMap<String, NameDeclarations>,
}

/// One distribution name's declarations across a source's occurrences.
#[derive(Default)]
struct NameDeclarations {
    /// How many occurrences declare the name: all of them, or the binding is
    /// incomplete.
    occurrences: usize,
    declarations: Vec<Reference>,
}

/// Python declarations constrain matching distribution-name candidates. This
/// is not a general module-to-distribution mapping. Stop at the nearest package
/// boundary; identical source bytes must agree across all supplied contexts.
pub(super) fn reconcile_python_imports(
    report: &AnalysisReport,
    worklist: &mut [Group],
    dependencies: bool,
) -> Vec<(String, Reference)> {
    let candidate_sources: HashSet<&str> = worklist
        .iter()
        .filter(|(_, refs)| refs.iter().any(is_python_import_candidate))
        .map(|(sha, _)| sha.as_str())
        .collect();
    if candidate_sources.is_empty() {
        return Vec::new();
    }
    let owners = python_owners(report, worklist);
    let contexts = python_contexts(report, &candidate_sources, &owners);
    let mut covered = Vec::new();
    for (source, refs) in worklist {
        let mut updated = Vec::new();
        for reference in std::mem::take(refs) {
            let binding = is_python_import_candidate(&reference)
                .then(|| python_distribution_name(&reference))
                .flatten()
                .and_then(|name| {
                    let context = contexts.get(source)?;
                    let named = context.names.get(&name)?;
                    Some((
                        named.occurrences == context.occurrences,
                        &named.declarations,
                    ))
                });
            let Some((complete, declarations)) = binding.filter(|(_, decls)| !decls.is_empty())
            else {
                updated.push(reference);
                continue;
            };
            if dependencies && complete {
                covered.push((source.clone(), reference));
                continue;
            }
            if !dependencies {
                for declaration in declarations {
                    let mut bound = reference.clone();
                    bound.locator.clone_from(&declaration.locator);
                    bound.pinned_hash.clone_from(&declaration.pinned_hash);
                    bound.context.clone_from(&declaration.context);
                    if !updated.contains(&bound) {
                        updated.push(bound);
                    }
                }
            }
            if !complete {
                updated.push(reference);
            }
        }
        *refs = updated;
    }
    covered
}

/// Every Python package boundary in the report — a manifest's directory, a
/// wheel, an installed `.dist-info` — and the distributions it declares.
fn python_owners(report: &AnalysisReport, worklist: &[Group]) -> PythonOwners {
    let mut source_refs: HashMap<&str, Vec<&Reference>> = HashMap::new();
    for (sha, refs) in worklist {
        source_refs.entry(sha).or_default().extend(refs);
    }
    let mut owners = PythonOwners::new();
    for file in &report.files {
        let basename = file.path.rsplit(['/', '!']).next().unwrap_or_default();
        let mut owner = virtual_parent(&file.path).unwrap_or_else(|| ".".into());
        let manifest = matches!(
            basename,
            "pyproject.toml"
                | "requirements.txt"
                | "setup.py"
                | "setup.cfg"
                | "Pipfile"
                | "Pipfile.lock"
        ) || PYPROJECT_LOCKS.contains(&basename);
        if report_file_type(&file.file_type) == FileType::Whl {
            owner = format!("{}!!", file.path);
        } else if basename == "METADATA" && owner.ends_with(".dist-info") {
            owner = virtual_parent(&owner).unwrap_or_else(|| ".".into());
        } else if !manifest {
            continue;
        }
        let names = owners.entry(owner).or_default();
        for &reference in source_refs.get(file.sha256.as_str()).into_iter().flatten() {
            if reference.kind != RefKind::Dependency {
                continue;
            }
            if let Some(name) = python_distribution_name(reference) {
                names.entry(name).or_default().push(reference.clone());
            }
        }
    }
    for names in owners.values_mut() {
        for refs in names.values_mut() {
            let reconciled: Vec<_> = refs
                .iter()
                .map(|r| fletch::fetch::prefer_lock_pin(r, refs))
                .collect();
            *refs = reconciled;
        }
    }
    owners
}

/// For each candidate source sha, the declarations of the nearest package
/// boundary above each of its occurrences. A content hash can occur in
/// several packages: every applicable constraint is kept, and an occurrence
/// with no declaration for a name leaves that name incomplete.
fn python_contexts(
    report: &AnalysisReport,
    candidate_sources: &HashSet<&str>,
    owners: &PythonOwners,
) -> HashMap<String, PythonContext> {
    let mut contexts: HashMap<String, PythonContext> = HashMap::new();
    for file in &report.files {
        if report_file_type(&file.file_type) != FileType::Python
            || file.path.ends_with(".pyi")
            || !candidate_sources.contains(file.sha256.as_str())
        {
            continue;
        }
        let start = virtual_parent(&file.path).unwrap_or_else(|| ".".into());
        let names = std::iter::successors(Some(start), |dir| virtual_parent(dir))
            .find_map(|dir| owners.get(&dir));
        let context = contexts.entry(file.sha256.clone()).or_default();
        context.occurrences += 1;
        for (name, declarations) in names.into_iter().flatten() {
            let named = context.names.entry(name.clone()).or_default();
            named.occurrences += 1;
            named.declarations.extend(declarations.iter().cloned());
        }
    }
    contexts
}

/// Report literal imports from a published npm package's own code that do
/// not appear in its runtime, optional or peer declarations. This is a
/// resolution concern, not an assertion that the imported package is hostile.
pub(super) fn undeclared_npm_imports(
    report: &AnalysisReport,
    worklist: &[Group],
) -> BTreeMap<String, Vec<Reference>> {
    let Some(root) = report
        .files
        .first()
        .filter(|f| report_file_type(&f.file_type) == FileType::Npm)
    else {
        return BTreeMap::new();
    };
    let prefix = format!("{}!!package/", root.path);
    let manifest_path = format!("{prefix}package.json");
    let Some(manifest) = report.files.iter().find(|f| f.path == manifest_path) else {
        return BTreeMap::new();
    };
    let package_name = manifest
        .identity
        .as_ref()
        .and_then(|i| i.name.as_ref())
        .map(|n| n.value.as_str());
    let declared: Vec<Reference> = worklist
        .iter()
        .filter(|(sha, _)| sha == &manifest.sha256)
        .flat_map(|(_, refs)| refs.iter())
        .filter(|r| r.kind == RefKind::Dependency)
        .cloned()
        .collect();
    let mut result = BTreeMap::new();
    for file in &report.files {
        let Some(member) = file.path.strip_prefix(&prefix) else {
            continue;
        };
        if !matches!(
            report_file_type(&file.file_type),
            FileType::JavaScript | FileType::TypeScript
        ) || member.ends_with(".d.ts")
            || member.ends_with(".d.cts")
            || member.ends_with(".d.mts")
            || member.split('/').any(|p| {
                matches!(
                    p,
                    "test" | "tests" | "__tests__" | "examples" | "fixtures" | "node_modules"
                )
            })
        {
            continue;
        }
        let mut refs = declared.clone();
        refs.extend(
            worklist
                .iter()
                .filter(|(sha, _)| sha == &file.sha256)
                .flat_map(|(_, refs)| refs.iter())
                .filter(|r| {
                    r.kind == RefKind::Command
                        && matches!(r.source.as_str(), "ast-call" | "ast-import")
                })
                .filter(|r| npm_import_name(r).is_some_and(|n| Some(n.as_str()) != package_name))
                .cloned(),
        );
        let candidates: Vec<Reference> = find::undeclared_packages(&refs)
            .into_iter()
            .cloned()
            .collect();
        if !candidates.is_empty() {
            result.entry(file.sha256.clone()).or_insert(candidates);
        }
    }
    result
}

pub(super) fn attribute_undeclared_npm_imports(report: &mut AnalysisReport, worklist: &[Group]) {
    let mut sources = Vec::new();
    for (sha, refs) in undeclared_npm_imports(report, worklist) {
        let Some(file) = report.files.iter_mut().find(|f| f.sha256 == sha) else {
            continue;
        };
        file.filefacts_metrics
            .get_or_insert_with(Default::default)
            .insert(
                "references.undeclared_import_count".into(),
                refs.len() as f64,
            );
        let spans = BTreeMap::from([(
            "references.undeclared_import_count".to_owned(),
            refs.iter()
                .filter_map(|r| r.offset.map(|offset| filefacts::Span::new(offset, 1)))
                .collect(),
        )]);
        sources.push((sha, spans));
    }
    let sources: Vec<_> = sources
        .iter()
        .map(|(sha, spans)| (sha.as_str(), spans))
        .collect();
    if let Err(e) =
        cleave::graft_reference_outcomes_batch(report, &sources, &AnalysisOptions::default())
    {
        tracing::warn!(
            error = format!("{e:#}"),
            "undeclared import trait pass failed"
        );
    }
}
