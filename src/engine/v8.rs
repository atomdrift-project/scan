//! The version 8 result upload (`JSON_PLAN.md`): one zstd stream of JSON
//! lines, a header, one line per distinct file, then one line per report.
//!
//! An [`UploadDoc`] is built in two steps. [`UploadDoc::new`] reads the
//! complete report while it still lists every node: which files the artifact
//! holds and where, what scan concluded about each, and what the artifact
//! points at. [`UploadDoc::attach_reports`] runs after retention and adds a
//! report for each file retention kept. The artifact's verdict can still move
//! after both, when a deferred LLM opinion lands, so lines are written only by
//! [`UploadDoc::encode`].

use std::collections::{BTreeMap, HashMap, HashSet};

use cleave::Criticality;
use cleave::types::{CompactFile, CompactReport, CompactTrait};
use filefacts::RefKind;
use fletch::fetch::FetchRecord;
use serde::Serialize;
use serde_json::Value;
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};

use super::retention::{MAX_KEPT_MEMBERS, own_nodes, parents};
use super::verdict::decision_outranks;
use super::{ARCHIVE_DELIMITER, ENGINE_VERSION, EmbeddedFile, FloorDecision, MemberEvals};
use crate::interpret::Interpretation;
use crate::model::{Classification, Decision, Level, RouteScore, SkippedRoute};

/// The body version this module writes.
const VERSION: u8 = 8;

// Hopper's limits on a body (`../hopper/internal/store/v8.go`). It rejects the
// whole upload when one is exceeded, so each is held here.

/// A verdict or juror object, as serialized.
const MAX_OBJECT: usize = 900;
/// `reason` and `error`, before any further cut an object needs to fit.
const MAX_TEXT: usize = 400;
/// File lines in one body.
const MAX_FILES: usize = 1_000_000;
/// Entries in one container's `contents`: the same number as the files one
/// artifact keeps, since no container can hold more distinct kept files than
/// that. Only one file listed under many names reaches it.
const MAX_MEMBERS: usize = MAX_KEPT_MEMBERS;
/// A path in `contents`.
const MAX_PATH: usize = 4096;
/// An `ident` object, as serialized.
const MAX_IDENT: usize = 8192;
/// A ref's `name`.
const MAX_NAME: usize = 2048;
/// A ref's `version` and `requested`.
const MAX_VERSION: usize = 256;
/// A file's `type`.
const MAX_TYPE: usize = 64;
/// One line, a report included.
const MAX_LINE: usize = 64 << 20;
/// The decompressed body. Hopper accepts more; this is the size scan has
/// always held a result to.
const MAX_BODY: usize = 512 << 20;

/// The keys of a compact file a report never carries: `sha`, `size`, `type`
/// and `ident` are on the file line, and the rest say where the file sat or
/// which run produced it, which would make the same bytes report differently
/// from different archives.
const PLACEMENT_KEYS: [&str; 12] = [
    "id", "pid", "depth", "dp", "path", "rel", "via", "role", "sha", "size", "type", "ident",
];

/// What scan concluded from one file's own reading, before anything it
/// contains is folded in.
#[derive(Debug, Clone, Copy)]
pub(super) struct OwnReading {
    /// The verdict: the model's, raised by the floor when it fired.
    pub verdict: Decision,
    /// The model's reading alone.
    pub model: Decision,
    /// The floor's firing, if it fired.
    pub floor: Option<FloorDecision>,
}

impl OwnReading {
    /// A member's own reading, as the embedded pass recorded it.
    fn of(member: &EmbeddedFile) -> Self {
        let verdict = member.decision();
        Self {
            verdict,
            model: member.floor.map_or(verdict, |floor| floor.model),
            floor: member.floor,
        }
    }
}

/// What [`UploadDoc::new`] reads.
pub(super) struct Inputs<'a> {
    /// The complete report, before retention drops the quiet nodes.
    pub report: &'a CompactReport,
    /// The artifact's own reading.
    pub root: OwnReading,
    /// The artifact's route scores and skipped routes.
    pub root_mods: &'a [RouteScore],
    pub root_skip: &'a [SkippedRoute],
    /// The embedded pass's evaluation of each member, by node id.
    pub members: &'a MemberEvals,
    /// The members that get a line, by node id: [`kept_members`].
    pub kept: &'a HashSet<u32>,
    /// Name every own member too, kept or not (`--all-members`). Such lines
    /// are extras: only [`UploadDoc::encode`] asked for every member writes
    /// them, never an upload.
    pub every: bool,
    /// Members cleave listed, analyzed or not, each to get an extra line
    /// (`--all-members`). Empty otherwise.
    pub listed: &'a [Listed],
    /// The scan's fetch records, which say what a ranged reference resolved
    /// to and what a URL served.
    pub records: &'a [FetchRecord],
    /// The deploy level every class is graded at; `None` under manual
    /// thresholds.
    pub level: Option<u16>,
}

/// A member cleave listed in an archive, analyzed or not.
#[derive(Debug, Clone)]
pub(super) struct Listed {
    /// Archive-relative path, a single `!` between nesting levels, as cleave
    /// records it.
    pub path: String,
    pub file_type: String,
    pub sha256: String,
    pub size: u64,
}

/// Everything a version 8 upload of one artifact says.
///
/// Opaque outside the engine: built by the classify pipeline, written by the
/// uploader.
#[derive(Debug, Clone)]
pub struct UploadDoc {
    level: Option<u16>,
    ruleset: String,
    /// File lines; the artifact is first.
    files: Vec<FileEntry>,
    refs: Vec<Ref>,
    /// The node whose report each line carries: the first node of its
    /// sha256. Consumed by [`Self::attach_reports`].
    report_nodes: HashMap<u32, usize>,
    /// Every node's sha256, for citing a source that has no line. Consumed by
    /// [`Self::attach_reports`].
    shas: HashMap<u32, String>,
    /// The drift jurors' objects, for the artifact's line: set by whoever
    /// builds the upload around scan's ([`Self::set_juror`]).
    drift: Vec<(Juror, Box<RawValue>)>,
    /// Third-party scanners' objects, for the artifact's line
    /// ([`Self::set_scanner`]).
    scanners: BTreeMap<Scanner, Box<RawValue>>,
}

#[derive(Debug, Clone)]
struct FileEntry {
    sha256: String,
    size: u64,
    file_type: String,
    /// `None` when scan did not evaluate the file.
    judged: Option<Judged>,
    /// What the file says it is (name, version, signer), ready for the wire;
    /// `None` when it says nothing.
    ident: Option<Box<RawValue>>,
    top: Vec<Finding>,
    report: Option<Report>,
    /// Where each immediate member sits: (path, line), ordered by path and
    /// then by the member's sha256, before any cut.
    members: Vec<(String, usize)>,
    /// Named only because every member was asked for (`--all-members`); an
    /// upload leaves it out.
    extra: bool,
}

/// A report line and the sha256 its file line names it by.
#[derive(Debug, Clone)]
struct Report {
    digest: String,
    text: String,
}

/// What scan concluded about a file and everything in it, and where each
/// reading came from: the sha256 of the contained file whose own bytes
/// carried it, when that was not this file.
#[derive(Debug, Clone)]
struct Judged {
    verdict: Decision,
    /// The juror that decided the verdict.
    by: Juror,
    verdict_in: Option<String>,
    ml: Decision,
    ml_in: Option<String>,
    /// Route scores and skipped routes of the file the model's reading is of.
    mods: Vec<RouteScore>,
    skip: Vec<SkippedRoute>,
    floor: Option<FloorDecision>,
    floor_in: Option<String>,
}

/// A reading and, when a contained node's own bytes carried it, that node's
/// index.
#[derive(Debug, Clone, Copy)]
struct Sourced<T> {
    value: T,
    from: Option<usize>,
}

impl<T> Sourced<T> {
    const fn own(value: T) -> Self {
        Self { value, from: None }
    }
}

/// A node's readings while they are folded up the tree.
#[derive(Debug, Clone, Copy)]
struct Readings {
    verdict: Sourced<Decision>,
    by: Juror,
    ml: Sourced<Decision>,
    floor: Option<Sourced<FloorDecision>>,
}

impl Readings {
    fn own(reading: &OwnReading) -> Self {
        Self {
            verdict: Sourced::own(reading.verdict),
            by: if reading.floor.is_some() {
                Juror::Floor
            } else {
                Juror::Ml
            },
            ml: Sourced::own(reading.model),
            floor: reading.floor.map(Sourced::own),
        }
    }

    /// The same readings as seen from the container of node `at`: each one
    /// names where its evidence came from.
    fn seen_from(mut self, at: usize) -> Self {
        self.verdict.from = self.verdict.from.or(Some(at));
        self.ml.from = self.ml.from.or(Some(at));
        if let Some(floor) = &mut self.floor {
            floor.from = floor.from.or(Some(at));
        }
        self
    }

    /// Fold in a contained node's readings, already [`seen_from`] here. Each
    /// reading is replaced only by a graver one, so a tie keeps the
    /// container's own and then the earliest member's.
    ///
    /// [`seen_from`]: Self::seen_from
    fn absorb(&mut self, child: &Self) {
        if decision_outranks(&child.verdict.value, &self.verdict.value) {
            self.verdict = child.verdict;
            self.by = child.by;
        }
        if decision_outranks(&child.ml.value, &self.ml.value) {
            self.ml = child.ml;
        }
        if let Some(candidate) = child.floor {
            match &mut self.floor {
                Some(current) if current.value.worse_of(candidate.value) == current.value => {}
                slot => *slot = Some(candidate),
            }
        }
    }
}

/// A juror: one component of atomdrift's verdict that states its own opinion
/// of a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Juror {
    /// The model.
    Ml,
    /// The trait floor.
    Floor,
    /// Scan's per-file interpreter.
    Llm,
    /// isomer's deterministic reading of a release against an earlier one:
    /// its rubric, before any interpreter.
    DriftHeuristic,
    /// isomer's interpreter of that difference.
    DriftLlm,
}

/// A scanner that is not atomdrift's, reporting beside its jurors: it never
/// feeds atomdrift's verdict. Hopper knows each by this name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Scanner {
    /// ClamAV, by its signature database.
    Clamav,
    /// Chainguard's malcontent, by its YARA rules.
    Malcontent,
    /// Datadog's GuardDog, by its package heuristics: npm, PyPI, Go and
    /// RubyGems packages only.
    Guarddog,
}

/// Why a juror's or a scanner's object cannot go on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpinionError {
    /// Only the drift jurors are set from outside: scan writes its own three.
    NotSettable(Juror),
    /// The object is not a JSON object.
    NotAnObject,
    /// Its `outcome` is not `ok`, `unsupported`, `timeout` or `error`.
    Outcome,
    /// It names no class on the shared scale, or a class beside a failed
    /// outcome.
    Class,
    /// It carries a field hopper reserves (a scanner's `report`).
    Reserved(&'static str),
    /// Serialized, it is over hopper's limit for one object.
    TooLarge(usize),
}

impl std::fmt::Display for OpinionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSettable(juror) => write!(f, "{juror:?} is scan's own juror and is not set from outside"),
            Self::NotAnObject => f.write_str("an opinion must be a JSON object"),
            Self::Outcome => f.write_str("an opinion's outcome is ok, unsupported, timeout or error"),
            Self::Class => f.write_str(
                "an opinion needs a class (benign, suspicious or hostile), or a failed outcome and no class",
            ),
            Self::Reserved(field) => write!(f, "an opinion may not carry {field:?}; hopper reserves it"),
            Self::TooLarge(bytes) => write!(f, "an opinion is {bytes} bytes; the limit is {MAX_OBJECT}"),
        }
    }
}

impl std::error::Error for OpinionError {}

/// `object` as hopper takes an opinion: NULs removed, a class on the shared
/// scale exactly when the outcome is `ok` (or absent), and at most
/// [`MAX_OBJECT`] bytes.
fn opinion(mut object: Value) -> Result<Box<RawValue>, OpinionError> {
    strip_nul(&mut object);
    let fields = object.as_object().ok_or(OpinionError::NotAnObject)?;
    let class = fields.get("class");
    let class_ok = match fields.get("outcome").map(Value::as_str) {
        None | Some(Some("ok")) => matches!(
            class.and_then(Value::as_str),
            Some("benign" | "suspicious" | "hostile")
        ),
        Some(Some("unsupported" | "timeout" | "error")) => class.is_none(),
        Some(_) => return Err(OpinionError::Outcome),
    };
    if !class_ok {
        return Err(OpinionError::Class);
    }
    // A JSON value always serializes; the error is unreachable.
    let raw = serde_json::value::to_raw_value(&object)
        .map_err(|_unwritable| OpinionError::NotAnObject)?;
    if raw.get().len() > MAX_OBJECT {
        return Err(OpinionError::TooLarge(raw.get().len()));
    }
    Ok(raw)
}

#[derive(Debug, Clone, Serialize)]
struct Finding {
    id: String,
    crit: u8,
}

#[derive(Debug, Clone, Default, Serialize)]
struct Contents {
    children: Vec<usize>,
    paths: Vec<String>,
}

/// Something the artifact points at.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct Ref {
    kind: &'static str,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    requested: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
}

/// What a body's header says about the run, beyond what the document holds.
#[derive(Debug, Clone, Copy)]
pub struct Stamp<'a> {
    /// When the analysis ran, RFC 3339 UTC.
    pub at: &'a str,
    /// The model bundle version every `ml` object names.
    pub model: &'a str,
    /// The uploading worker's name, which hopper requires of an upload. `None`
    /// for output that is not uploaded.
    pub worker: Option<&'a str>,
    /// How long the analysis took, in milliseconds, when the caller timed it.
    pub ms: Option<u64>,
}

impl UploadDoc {
    /// Read the complete report: fold every own node into its containers,
    /// and give the artifact and each kept member a line. `None` when the
    /// report lists no file at all.
    pub(super) fn new(inputs: &Inputs<'_>) -> Option<Self> {
        let nodes = &inputs.report.files;
        let root = nodes.first()?;
        let parents = parents(nodes);
        let own = own_nodes(nodes, &parents);
        let eval = |i: usize| {
            (i != 0)
                .then(|| inputs.members.get(&u64::from(nodes[i].id)))
                .flatten()
        };
        let folded = fold(nodes, &parents, &own, |i| {
            if i == 0 {
                Some(inputs.root)
            } else {
                eval(i).map(OwnReading::of)
            }
        });
        // A file scan did not evaluate carries no verdict, even when
        // something inside it was.
        let judged = |i: usize| -> Option<Judged> {
            if i != 0 && eval(i).is_none() {
                return None;
            }
            let r = folded[i]?;
            let sha = |from: Option<usize>| from.map(|n| nodes[n].sha.clone());
            let (mods, skip) = match r.ml.from.unwrap_or(i) {
                0 => (inputs.root_mods.to_vec(), inputs.root_skip.to_vec()),
                source => eval(source).map_or_else(Default::default, |m| {
                    (m.model_scores.clone(), m.skipped_models.clone())
                }),
            };
            Some(Judged {
                verdict: r.verdict.value,
                by: r.by,
                verdict_in: sha(r.verdict.from),
                ml: r.ml.value,
                ml_in: sha(r.ml.from),
                mods,
                skip,
                floor: r.floor.map(|f| f.value),
                floor_in: r.floor.and_then(|f| sha(f.from)),
            })
        };

        let mut doc = Self {
            level: inputs.level,
            ruleset: inputs
                .report
                .traits_version
                .clone()
                .unwrap_or_else(crate::analysis_cache::ruleset_version),
            files: Vec::new(),
            refs: refs(nodes, &own, inputs.records),
            report_nodes: HashMap::new(),
            shas: nodes.iter().map(|n| (n.id, n.sha.clone())).collect(),
            drift: Vec::new(),
            scanners: BTreeMap::new(),
        };
        let mut lines: HashMap<&str, usize> = HashMap::new();
        let mut line_of: Vec<Option<usize>> = vec![None; nodes.len()];
        let mut capped = false;
        let named = (0..nodes.len()).filter(|&i| {
            i == 0
                || (own[i]
                    && !nodes[i].sha.is_empty()
                    && (inputs.every || inputs.kept.contains(&nodes[i].id)))
        });
        for i in named {
            let node = &nodes[i];
            let extra = i != 0 && !inputs.kept.contains(&node.id);
            if let Some(&line) = lines.get(node.sha.as_str()) {
                line_of[i] = Some(line);
                let file = &mut doc.files[line];
                file.extra &= extra;
                if file.judged.is_none() {
                    file.judged = judged(i);
                    file.top = top(&node.findings);
                }
                if file.ident.is_none() {
                    file.ident = identity(node);
                }
                continue;
            }
            if doc.files.len() == MAX_FILES {
                capped = true;
                continue;
            }
            let line = doc.files.len();
            doc.files.push(FileEntry {
                sha256: node.sha.clone(),
                size: node.size,
                file_type: bounded(&node.file_type, MAX_TYPE).to_owned(),
                judged: judged(i),
                ident: identity(node),
                top: top(&node.findings),
                report: None,
                members: Vec::new(),
                extra,
            });
            lines.insert(node.sha.as_str(), line);
            line_of[i] = Some(line);
            doc.report_nodes.insert(node.id, line);
        }
        // Every other copy of a kept file is another place it sits, whether or
        // not the ranking reached that copy.
        for i in 1..nodes.len() {
            if line_of[i].is_none() && own[i] {
                line_of[i] = lines.get(nodes[i].sha.as_str()).copied();
            }
        }
        let mut listed_edges = Vec::new();
        if !inputs.listed.is_empty() {
            let paths: HashMap<&str, usize> = (0..nodes.len())
                .filter_map(|i| Some((nodes[i].path.as_str(), line_of[i]?)))
                .collect();
            capped |= doc.add_listed(
                inputs.listed,
                &root.path,
                &mut lines,
                &paths,
                &mut listed_edges,
            );
        }
        if capped {
            tracing::warn!(
                sha256 = %root.sha,
                limit = MAX_FILES,
                "upload: artifact keeps more files than one result may name; the rest are left out",
            );
        }

        let mut edges: Vec<Vec<(String, usize)>> = vec![Vec::new(); doc.files.len()];
        for (parent, path, child) in listed_edges {
            edges[parent].push((path, child));
        }
        for i in 1..nodes.len() {
            let (Some(child), Some(parent)) = (line_of[i], parents[i]) else {
                continue;
            };
            let Some(parent_line) = line_of[parent].filter(|&p| p != child) else {
                continue;
            };
            let path = relative(&nodes[i].path, &nodes[parent].path);
            if !path.is_empty() {
                edges[parent_line].push((path.to_owned(), child));
            }
        }
        for (line, mut list) in edges.into_iter().enumerate() {
            list.sort_by(|(a, x), (b, y)| {
                a.as_bytes()
                    .cmp(b.as_bytes())
                    .then_with(|| doc.files[*x].sha256.cmp(&doc.files[*y].sha256))
            });
            list.dedup();
            doc.files[line].members = list;
        }
        Some(doc)
    }

    /// Give each listed member a line, under the container its path names,
    /// as (container, path, member) edges. Returns whether the file cap left
    /// one out.
    fn add_listed<'a>(
        &mut self,
        listed: &'a [Listed],
        root_path: &str,
        lines: &mut HashMap<&'a str, usize>,
        paths: &HashMap<&str, usize>,
        edges: &mut Vec<(usize, String, usize)>,
    ) -> bool {
        let names: HashSet<&str> = listed.iter().map(|m| m.path.as_str()).collect();
        // The compact form of each listed path, which no node spells.
        let mut listed_paths: HashMap<String, usize> = HashMap::new();
        let mut capped = false;
        for member in listed.iter().filter(|m| !m.sha256.is_empty()) {
            let line = match lines.get(member.sha256.as_str()) {
                Some(&line) => line,
                None if self.files.len() < MAX_FILES => {
                    let line = self.files.len();
                    self.files.push(FileEntry {
                        sha256: member.sha256.clone(),
                        size: member.size,
                        file_type: bounded(&member.file_type, MAX_TYPE).to_owned(),
                        judged: None,
                        ident: None,
                        top: Vec::new(),
                        report: None,
                        members: Vec::new(),
                        extra: true,
                    });
                    lines.insert(&member.sha256, line);
                    line
                }
                None => {
                    capped = true;
                    continue;
                }
            };
            let layers = listing_layers(&member.path, &names);
            let Some((inner, outer)) = layers.split_last() else {
                continue;
            };
            let compact = |layers: &[&str]| {
                std::iter::once(root_path)
                    .chain(layers.iter().copied())
                    .collect::<Vec<_>>()
                    .join(ARCHIVE_DELIMITER)
            };
            let parent = if outer.is_empty() {
                Some(0)
            } else {
                let container = compact(outer);
                paths
                    .get(container.as_str())
                    .or_else(|| listed_paths.get(&container))
                    .copied()
            };
            // A container cleave did not list is unknown; the member is then
            // filed under the artifact at its full archive path.
            let (parent, path) = parent.map_or((0, member.path.as_str()), |p| (p, *inner));
            if parent != line && !path.is_empty() {
                edges.push((parent, path.to_owned(), line));
            }
            listed_paths.entry(compact(&layers)).or_insert(line);
        }
        capped
    }

    /// Add a report line for each judged file `retained` kept. `retained` is
    /// the same report [`Self::new`] read, after retention.
    pub(super) fn attach_reports(&mut self, retained: &CompactReport) {
        let report_nodes = std::mem::take(&mut self.report_nodes);
        let shas = std::mem::take(&mut self.shas);
        let sha_of = |id: u32| shas.get(&id).map(String::as_str);
        let files = &self.files;
        let reports: Vec<(usize, String)> = retained
            .files
            .iter()
            .filter_map(|file| {
                // Hopper reads a report only beside a verdict.
                let &line = report_nodes
                    .get(&file.id)
                    .filter(|&&l| files[l].judged.is_some())?;
                let text = report_line(file, &sha_of)?;
                if text.len() > MAX_LINE {
                    tracing::warn!(
                        sha256 = %file.sha,
                        bytes = text.len(),
                        limit = MAX_LINE,
                        "upload: report longer than one line may be; sent without it",
                    );
                    return None;
                }
                Some((line, text))
            })
            .collect();
        let count = reports.len();
        for (line, text) in reports {
            let digest = format!("{:x}", Sha256::digest(text.as_bytes()));
            self.files[line].report = Some(Report { digest, text });
        }
        tracing::debug!(
            sha256 = %self.files.first().map_or("", |f| f.sha256.as_str()),
            files = self.files.len(),
            reports = count,
            refs = self.refs.len(),
            "upload: version 8 document",
        );
    }

    /// The artifact's verdict as it stands: scan's fold of the artifact and
    /// everything inside it, LLM included, unless [`Self::decide`] replaced it.
    /// `None` when scan did not judge the artifact.
    #[must_use]
    pub fn verdict(&self) -> Option<Decision> {
        self.files.first()?.judged.as_ref().map(|j| j.verdict)
    }

    /// Record `object` as the artifact's opinion from the drift juror
    /// `juror`, replacing any earlier one. It is written as given, with NULs
    /// removed: the juror's own fields beside `class` on the shared scale, or
    /// a failed `outcome` and no class.
    ///
    /// # Errors
    /// `juror` is one of scan's own, or the object is not one hopper accepts.
    pub fn set_juror(&mut self, juror: Juror, object: Value) -> Result<(), OpinionError> {
        if !matches!(juror, Juror::DriftHeuristic | Juror::DriftLlm) {
            return Err(OpinionError::NotSettable(juror));
        }
        let raw = opinion(object)?;
        self.drift.retain(|(j, _)| *j != juror);
        self.drift.push((juror, raw));
        Ok(())
    }

    /// Record `object` as what the third-party `scanner` said of the
    /// artifact, replacing any earlier one: its finding beside `class`, or a
    /// failed `outcome` and no class, and its own stamp (`engine`, and
    /// `ruleset` or `db`). It never changes the verdict. A scanner the body
    /// does not name keeps its stored opinion on hopper, so a caller sets one
    /// for every scanner it ran, failures included, and none for the rest.
    ///
    /// # Errors
    /// The object is not one hopper accepts.
    pub fn set_scanner(&mut self, scanner: Scanner, object: Value) -> Result<(), OpinionError> {
        if object.get("report").is_some() {
            return Err(OpinionError::Reserved("report"));
        }
        self.scanners.insert(scanner, opinion(object)?);
        Ok(())
    }

    /// Make the artifact's verdict `class` at level `fires_at`, decided by
    /// `by`: the final verdict of whoever folds other jurors' opinions into
    /// scan's. Nothing happens when scan did not judge the artifact.
    pub fn decide(&mut self, class: Classification, fires_at: Level, by: Juror) {
        if let Some(judged) = self.files.first_mut().and_then(|f| f.judged.as_mut()) {
            judged.verdict.class = class;
            judged.verdict.level = fires_at;
            judged.by = by;
            judged.verdict_in = None;
        }
    }

    /// Let a later opinion move the artifact's verdict. `apply` returns
    /// whether it moved it; the verdict is then the LLM's.
    pub(super) fn blend(&mut self, apply: impl FnOnce(&mut Decision) -> bool) {
        if let Some(judged) = self.files.first_mut().and_then(|f| f.judged.as_mut())
            && apply(&mut judged.verdict)
        {
            judged.by = Juror::Llm;
            judged.verdict_in = None;
        }
    }

    /// The body, uncompressed: the artifact filed under `sha256`, with `llm`
    /// as its LLM opinion. `None` when it cannot be serialized.
    #[must_use]
    pub fn encode(
        &self,
        sha256: &str,
        llm: Option<&Interpretation>,
        stamp: &Stamp<'_>,
    ) -> Option<Vec<u8>> {
        self.encode_with(sha256, llm, stamp, false)
    }

    /// [`Self::encode`], naming every member the document holds, kept or not
    /// (`--all-members`). For output only; an upload is [`Self::encode`].
    #[must_use]
    pub fn encode_every(
        &self,
        sha256: &str,
        llm: Option<&Interpretation>,
        stamp: &Stamp<'_>,
    ) -> Option<Vec<u8>> {
        self.encode_with(sha256, llm, stamp, true)
    }

    fn encode_with(
        &self,
        sha256: &str,
        llm: Option<&Interpretation>,
        stamp: &Stamp<'_>,
        every: bool,
    ) -> Option<Vec<u8>> {
        let body = self.write(sha256, llm, stamp, every, true)?;
        if body.len() <= MAX_BODY {
            return Some(body);
        }
        tracing::warn!(
            sha256,
            bytes = body.len(),
            limit = MAX_BODY,
            "upload: result body too large; sending it without reports",
        );
        drop(body);
        self.write(sha256, llm, stamp, every, false)
    }

    fn write(
        &self,
        sha256: &str,
        llm: Option<&Interpretation>,
        stamp: &Stamp<'_>,
        every: bool,
        with_reports: bool,
    ) -> Option<Vec<u8>> {
        // Each written file's line number in this body; extras are left out
        // unless every member was asked for.
        let mut at = vec![None; self.files.len()];
        let mut written = 0usize;
        for (i, file) in self.files.iter().enumerate() {
            if every || !file.extra {
                at[i] = Some(written);
                written += 1;
            }
        }
        let mut out = Vec::new();
        let header = Header {
            v: VERSION,
            scan: ScanStamp {
                engine: ENGINE_VERSION,
                ruleset: &self.ruleset,
                level: self.level,
                worker: stamp.worker,
                at: stamp.at,
                ms: stamp.ms,
            },
            files: written,
        };
        push_line(&mut out, &header)?;
        // The report lines follow in the order of the file lines that named
        // one, so each is remembered as its file line is written.
        let mut reports = Vec::new();
        for (i, file) in self.files.iter().enumerate() {
            if at[i].is_none() {
                continue;
            }
            let key = if i == 0 { sha256 } else { file.sha256.as_str() };
            let mut line =
                self.file_line(i, key, llm.filter(|_| i == 0), stamp.model, with_reports);
            line.contents = self.contents(i, &at);
            if line.report.is_some()
                && let Some(report) = &file.report
            {
                reports.push(report.text.as_str());
            }
            push_line(&mut out, &line)?;
        }
        for report in reports {
            out.extend_from_slice(report.as_bytes());
            out.push(b'\n');
        }
        Some(out)
    }

    /// File `i`'s `contents` in a body whose line numbers are `at`: its
    /// members written there, cut to what one result may carry.
    fn contents(&self, i: usize, at: &[Option<usize>]) -> Option<Contents> {
        let file = &self.files[i];
        let mut contents = Contents::default();
        let (mut present, mut long) = (0usize, 0usize);
        for (path, child) in &file.members {
            let Some(line) = at[*child] else { continue };
            present += 1;
            if path.len() > MAX_PATH {
                long += 1;
            } else if contents.children.len() < MAX_MEMBERS {
                contents.children.push(line);
                contents.paths.push(path.clone());
            }
        }
        if contents.children.len() < present {
            tracing::warn!(
                sha256 = %file.sha256,
                members = present,
                kept = contents.children.len(),
                paths_too_long = long,
                limit = MAX_MEMBERS,
                "upload: container's member list cut to what one result may carry",
            );
        }
        (!contents.children.is_empty()).then_some(contents)
    }

    fn file_line<'a>(
        &'a self,
        i: usize,
        sha256: &'a str,
        llm: Option<&'a Interpretation>,
        model: &'a str,
        with_reports: bool,
    ) -> FileLine<'a> {
        let file = &self.files[i];
        let mut line = FileLine {
            sha256,
            size: file.size,
            file_type: &file.file_type,
            verdict: None,
            ml: None,
            floor: None,
            llm: None,
            drift_heuristic: None,
            drift_llm: None,
            scanners: None,
            report: None,
            ident: file.ident.as_deref(),
            contents: None,
            refs: (i == 0).then_some(self.refs.as_slice()),
        };
        let Some(judged) = &file.judged else {
            return line;
        };
        let reason = match llm {
            Some(Interpretation::Graded(graded)) => graded.interpretation.as_str(),
            _ => "",
        };
        let (Some(verdict), Some(ml)) = (
            verdict_object(judged, &file.top, reason),
            ml_object(judged, model),
        ) else {
            tracing::error!(
                sha256,
                "upload: verdict does not fit hopper's limit; file sent without it",
            );
            return line;
        };
        line.verdict = Some(verdict);
        line.ml = Some(ml);
        line.floor = judged
            .floor
            .as_ref()
            .and_then(|floor| floor_object(floor, judged.floor_in.as_deref()));
        line.llm = llm.and_then(llm_object);
        if i == 0 {
            for (juror, object) in &self.drift {
                match juror {
                    Juror::DriftHeuristic => line.drift_heuristic = Some(object),
                    Juror::DriftLlm => line.drift_llm = Some(object),
                    Juror::Ml | Juror::Floor | Juror::Llm => {}
                }
            }
            line.scanners = (!self.scanners.is_empty()).then(|| {
                self.scanners
                    .iter()
                    .map(|(&scanner, object)| (scanner, &**object))
                    .collect()
            });
        }
        if with_reports {
            line.report = file.report.as_ref().map(|r| r.digest.as_str());
        }
        line
    }
}

/// Each own node's readings with everything beneath it folded in, by node
/// index. Deepest first, so a container sees its members' finished readings;
/// members at one depth are folded in path order, so the result does not
/// depend on the order cleave listed them.
fn fold(
    nodes: &[CompactFile],
    parents: &[Option<usize>],
    own: &[bool],
    reading: impl Fn(usize) -> Option<OwnReading>,
) -> Vec<Option<Readings>> {
    let mut folded: Vec<Option<Readings>> = (0..nodes.len())
        .map(|i| {
            own[i]
                .then(|| reading(i))
                .flatten()
                .as_ref()
                .map(Readings::own)
        })
        .collect();
    let mut order: Vec<usize> = (1..nodes.len()).filter(|&i| own[i]).collect();
    order.sort_by(|&a, &b| {
        nodes[b]
            .depth
            .cmp(&nodes[a].depth)
            .then_with(|| nodes[a].path.cmp(&nodes[b].path))
            .then_with(|| nodes[a].sha.cmp(&nodes[b].sha))
    });
    for i in order {
        let (Some(parent), Some(child)) = (parents[i], folded[i]) else {
            continue;
        };
        let seen = child.seen_from(i);
        match &mut folded[parent] {
            Some(current) => current.absorb(&seen),
            slot => *slot = Some(seen),
        }
    }
    folded
}

/// What a node says it is, when it says anything, as its `ident` object.
///
/// Every string in it came out of the file, so NULs are removed, and a block
/// over [`MAX_IDENT`] gives up list entries (authors, emails, urls, unique
/// ids) before anything else. Dropping the block instead would let a file
/// shed its identity by padding it.
fn identity(node: &CompactFile) -> Option<Box<RawValue>> {
    let ident = node.identity.as_ref().filter(|ident| !ident.is_empty())?;
    let mut value = serde_json::to_value(ident)
        .inspect_err(|e| tracing::warn!(sha256 = %node.sha, error = %e, "upload: identity does not serialize"))
        .ok()?;
    strip_nul(&mut value);
    let mut json = serde_json::to_string(&value).ok()?;
    while json.len() > MAX_IDENT {
        if !shorten(&mut value) {
            tracing::warn!(sha256 = %node.sha, bytes = json.len(), "upload: identity cannot be shortened to fit");
            return None;
        }
        json = serde_json::to_string(&value).ok()?;
    }
    RawValue::from_string(json).ok()
}

/// Remove every NUL from every string in `value`, keys included.
fn strip_nul(value: &mut Value) {
    match value {
        Value::String(text) => text.retain(|c| c != '\0'),
        Value::Array(items) => items.iter_mut().for_each(strip_nul),
        Value::Object(map) => {
            let entries: Vec<(String, Value)> = std::mem::take(map).into_iter().collect();
            for (mut key, mut child) in entries {
                key.retain(|c| c != '\0');
                strip_nul(&mut child);
                map.insert(key, child);
            }
        }
        _ => {}
    }
}

/// Make `value` smaller by one step: drop the last entry of its longest list,
/// else the last entry of its largest map below the top level, else halve its
/// longest string. `false` when nothing is left to shorten.
fn shorten(value: &mut Value) -> bool {
    fn longest_array(value: &mut Value) -> Option<&mut Vec<Value>> {
        match value {
            Value::Array(items) if !items.is_empty() => Some(items),
            Value::Object(map) => map
                .values_mut()
                .filter_map(longest_array)
                .max_by_key(|list| list.len()),
            _ => None,
        }
    }
    if let Some(list) = longest_array(value) {
        list.pop();
        return true;
    }
    if let Value::Object(top) = value
        && let Some(map) = top
            .values_mut()
            .filter_map(Value::as_object_mut)
            .filter(|m| !m.is_empty())
            .max_by_key(|m| m.len())
    {
        let last = map.keys().next_back().cloned();
        if let Some(key) = last {
            map.remove(&key);
            return true;
        }
    }
    let mut longest: Option<&mut String> = None;
    let mut stack = vec![value];
    while let Some(next) = stack.pop() {
        match next {
            Value::String(text) if text.len() > longest.as_ref().map_or(1, |l| l.len()) => {
                longest = Some(text);
            }
            Value::Array(items) => stack.extend(items.iter_mut()),
            Value::Object(map) => stack.extend(map.values_mut()),
            _ => {}
        }
    }
    let Some(text) = longest else {
        return false;
    };
    let keep = bounded(text, text.len() / 2).len();
    text.truncate(keep);
    true
}

/// The archive layers of a listed path.
///
/// cleave joins nesting levels there with a single `!` (`inner.tar!lib/a.so`),
/// and a member's own name may contain `!` too. A `!` is a layer boundary only
/// where the text before it is itself a listed entry: cleave lists every nested
/// archive before its contents.
fn listing_layers<'a>(path: &'a str, listed: &HashSet<&str>) -> Vec<&'a str> {
    let mut layers = Vec::new();
    let mut start = 0;
    for (bang, _) in path.match_indices('!') {
        // `bang` indexes an ASCII '!', so both slices fall on char boundaries.
        if let (Some(container), Some(layer)) = (path.get(..bang), path.get(start..bang))
            && listed.contains(container)
        {
            layers.push(layer);
            start = bang + 1;
        }
    }
    layers.push(path.get(start..).unwrap_or(path));
    layers
}

/// Where `child` sits inside `parent`, from their compact paths.
fn relative<'a>(child: &'a str, parent: &str) -> &'a str {
    match child.strip_prefix(parent) {
        Some(rest) => rest.strip_prefix(ARCHIVE_DELIMITER).unwrap_or(rest),
        None => child.rsplit(ARCHIVE_DELIMITER).next().unwrap_or(child),
    }
}

/// At most three of a file's findings at crit 5, then crit 4, in report
/// order. A dependency's verdict pinned onto the file is not one of its own.
fn top(findings: &[CompactTrait]) -> Vec<Finding> {
    let mut top = Vec::new();
    for crit in [Criticality::Hostile, Criticality::Suspicious] {
        for finding in findings
            .iter()
            .filter(|f| f.dep.is_none() && super::compact_crit(f) == crit)
        {
            if top.len() == 3 {
                return top;
            }
            top.push(Finding {
                id: finding.id.clone(),
                crit: crit.rank(),
            });
        }
    }
    top
}

/// The artifact's references: the union of what its own files declare, one
/// per distinct (`kind`, `name`, `version`), in that order.
fn refs(nodes: &[CompactFile], own: &[bool], records: &[FetchRecord]) -> Vec<Ref> {
    let mut at: HashMap<(&str, u64), &FetchRecord> = HashMap::new();
    let mut named: HashMap<(&str, &'static str, String), &FetchRecord> = HashMap::new();
    for record in records {
        let (Some(source), Some(kind)) = (record.source_sha256.as_deref(), kind_name(record.kind))
        else {
            continue;
        };
        if let Some(offset) = record.source_offset {
            at.entry((source, offset)).or_insert(record);
        }
        if let Some(name) = bare_name(&record.locator) {
            named.entry((source, kind, name)).or_insert(record);
        }
    }
    let mut refs: Vec<Ref> = nodes
        .iter()
        .zip(own)
        .filter(|(_, own)| **own)
        .flat_map(|(node, _)| node.refs.iter().map(move |r| (node, r)))
        .filter_map(|(node, declared)| {
            let kind = match declared.kind.as_str() {
                "dependency" => "dependency",
                "command" => "command",
                "url_fetch" => "url_fetch",
                _ => return None,
            };
            let record = declared
                .offset
                .and_then(|off| at.get(&(node.sha.as_str(), off)))
                .or_else(|| named.get(&(node.sha.as_str(), kind, bare_name(&declared.locator)?)))
                .copied();
            reference(kind, &declared.locator, record)
        })
        .collect();
    refs.sort();
    refs.dedup_by(|a, b| (a.kind, &a.name, &a.version) == (b.kind, &b.name, &b.version));
    refs
}

/// One reference, or `None` when it names neither a package nor a URL or
/// does not fit hopper's limits.
fn reference(kind: &'static str, locator: &str, record: Option<&FetchRecord>) -> Option<Ref> {
    let r = if locator.starts_with("https://") || locator.starts_with("http://") {
        Ref {
            kind,
            name: locator.to_owned(),
            version: None,
            requested: None,
            sha256: record
                .filter(|_| kind == "url_fetch")
                .and_then(|r| r.content_sha256.clone()),
        }
    } else {
        let purl = fletch::purl::Purl::parse(locator).ok()?;
        let exact = |p: &fletch::purl::Purl| {
            p.pins_exact_version()
                .then(|| p.version().map(str::to_owned))
                .flatten()
        };
        let declared = exact(&purl);
        Ref {
            kind,
            name: bare(&purl)?,
            // `*`, the `vers` spelling of any version, when the declaration
            // states no constraint at all.
            requested: declared
                .is_none()
                .then(|| purl.requirement().unwrap_or("*").to_owned()),
            version: declared.or_else(|| exact(&fletch::purl::Purl::parse(&record?.locator).ok()?)),
            sha256: None,
        }
    };
    let fits = r.name.len() <= MAX_NAME
        && r.version.as_ref().is_none_or(|v| v.len() <= MAX_VERSION)
        && r.requested.as_ref().is_none_or(|v| v.len() <= MAX_VERSION);
    fits.then_some(r)
}

/// A package URL's name: type, namespace and name, without its version,
/// qualifiers or subpath.
fn bare(purl: &fletch::purl::Purl) -> Option<String> {
    let namespace = purl.namespace().join("/");
    fletch::purl::Purl::build(
        purl.typ(),
        Some(namespace.as_str()),
        purl.name(),
        None,
        [],
        None,
    )
    .ok()
    .map(|p| p.to_string())
}

/// [`bare`] of a locator, for matching a reference to the fetch it led to.
/// A URL is its own name.
fn bare_name(locator: &str) -> Option<String> {
    if locator.starts_with("https://") || locator.starts_with("http://") {
        return Some(locator.to_owned());
    }
    fletch::purl::Purl::parse(locator)
        .ok()
        .and_then(|p| bare(&p))
}

fn kind_name(kind: RefKind) -> Option<&'static str> {
    match kind {
        RefKind::Dependency => Some("dependency"),
        RefKind::Command => Some("command"),
        RefKind::UrlFetch => Some("url_fetch"),
        _ => None,
    }
}

/// A file's report: its findings and facts, with nothing that says where it
/// sits, and every source in another file cited by that file's sha256. Keys
/// are sorted at every level so the same findings always give the same bytes.
fn report_line<'a>(file: &CompactFile, sha_of: &impl Fn(u32) -> Option<&'a str>) -> Option<String> {
    let mut value: Value = serde_json::to_string(file)
        .and_then(|text| serde_json::from_str(&text))
        .inspect_err(|e| tracing::error!(sha256 = %file.sha, error = %e, "upload: report does not serialize"))
        .ok()?;
    let object = value.as_object_mut()?;
    for key in PLACEMENT_KEYS {
        object.remove(key);
    }
    if let Some(Value::Array(traits)) = object.get_mut("traits") {
        // A dependency's verdict pinned onto the file is about another
        // artifact. Pinned last, so dropping it moves no `uses` index.
        traits.retain(|t| t.get("dep").is_none());
        for finding in traits.iter_mut() {
            if let Some(Value::Array(sources)) = finding.get_mut("from") {
                sources.retain_mut(|source| cite(source, &file.sha, sha_of));
            }
        }
        if traits.is_empty() {
            object.remove("traits");
        }
    }
    if let Some(Value::Array(refs)) = object.get_mut("refs") {
        for declared in refs.iter_mut() {
            cite(declared, &file.sha, sha_of);
        }
    }
    sort_keys(&mut value);
    serde_json::to_string(&value).ok()
}

/// Replace the scan-local `file` id in `entry` with the cited file's sha256,
/// or with nothing when the file is the report's own. Returns `false` when
/// the id names no file, so the caller can drop the entry.
fn cite<'a>(entry: &mut Value, own: &str, sha_of: &impl Fn(u32) -> Option<&'a str>) -> bool {
    let Some(object) = entry.as_object_mut() else {
        return true;
    };
    let Some(id) = object.remove("file") else {
        return true;
    };
    let Some(sha) = id
        .as_u64()
        .and_then(|id| u32::try_from(id).ok())
        .and_then(sha_of)
    else {
        return false;
    };
    if sha != own {
        object.insert("sha256".to_owned(), Value::String(sha.to_owned()));
    }
    true
}

fn sort_keys(value: &mut Value) {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = std::mem::take(map).into_iter().collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));
            for (key, mut child) in entries {
                sort_keys(&mut child);
                map.insert(key, child);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(sort_keys),
        _ => {}
    }
}

/// `text` without NULs, which hopper refuses in any verdict or juror object.
fn without_nul(text: &str) -> std::borrow::Cow<'_, str> {
    if text.contains('\0') {
        std::borrow::Cow::Owned(text.replace('\0', ""))
    } else {
        std::borrow::Cow::Borrowed(text)
    }
}

/// The longest prefix of `text` within `max` bytes that ends on a character
/// boundary.
fn bounded(text: &str, max: usize) -> &str {
    let mut end = max.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.get(..end).unwrap_or_default()
}

/// Serialize the object `make` builds around its free text, cutting the text
/// until the object fits hopper's limit. `None` when it does not fit even
/// with no text at all.
fn fit<'t, T: Serialize>(text: &'t str, make: impl Fn(&'t str) -> T) -> Option<Box<RawValue>> {
    let mut text = bounded(text, MAX_TEXT);
    loop {
        let json = serde_json::to_string(&make(text)).ok()?;
        if json.len() <= MAX_OBJECT {
            return RawValue::from_string(json).ok();
        }
        if text.is_empty() {
            return None;
        }
        // Escaping can make the text cost more than its length, so cut by at
        // least the overflow and keep going until it fits.
        let over = json.len() - MAX_OBJECT;
        text = bounded(text, text.len().saturating_sub(over.max(1)));
    }
}

const fn class_word(class: Classification) -> &'static str {
    match class {
        Classification::Benign => "benign",
        Classification::Suspicious => "suspicious",
        Classification::Hostile => "hostile",
    }
}

#[derive(Serialize)]
struct VerdictObject<'a> {
    outcome: &'static str,
    class: &'static str,
    fires_at: Level,
    by: Juror,
    #[serde(rename = "in", skip_serializing_if = "Option::is_none")]
    within: Option<&'a str>,
    #[serde(skip_serializing_if = "str::is_empty")]
    reason: &'a str,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    top: &'a [Finding],
}

fn verdict_object(judged: &Judged, top: &[Finding], reason: &str) -> Option<Box<RawValue>> {
    let reason = without_nul(reason);
    // Shorten the reason first and give up findings only if that is not
    // enough.
    (0..=top.len()).rev().find_map(|keep| {
        fit(&reason, |reason| VerdictObject {
            outcome: "ok",
            class: class_word(judged.verdict.class),
            fires_at: judged.verdict.level,
            by: judged.by,
            within: judged.verdict_in.as_deref(),
            reason,
            top: top.get(..keep).unwrap_or_default(),
        })
    })
}

#[derive(Serialize)]
struct MlObject<'a> {
    class: &'static str,
    fires_at: Level,
    prob: f32,
    model: &'a str,
    #[serde(rename = "in", skip_serializing_if = "Option::is_none")]
    within: Option<&'a str>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    mods: &'a [RouteScore],
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    skip: &'a [SkippedRoute],
}

/// The model's reading, with the route scores of the file it is of. The
/// skipped routes go first, then the scores, if the object is too large.
fn ml_object(judged: &Judged, model: &str) -> Option<Box<RawValue>> {
    let (mods, skip) = (judged.mods.as_slice(), judged.skip.as_slice());
    let (no_mods, no_skip): (&[RouteScore], &[SkippedRoute]) = (&[], &[]);
    [(mods, skip), (mods, no_skip), (no_mods, no_skip)]
        .into_iter()
        .find_map(|(mods, skip)| {
            fit("", |_| MlObject {
                class: class_word(judged.ml.class),
                fires_at: judged.ml.level,
                prob: judged.ml.probability,
                model,
                within: judged.ml_in.as_deref(),
                mods,
                skip,
            })
        })
}

#[derive(Serialize)]
struct FloorObject<'a> {
    class: &'static str,
    fires_at: Level,
    #[serde(rename = "in", skip_serializing_if = "Option::is_none")]
    within: Option<&'a str>,
    arm: &'static str,
    hostile: u32,
    suspicious: u32,
    families: usize,
    conf: f32,
}

fn floor_object(f: &FloorDecision, within: Option<&str>) -> Option<Box<RawValue>> {
    fit("", |_| FloorObject {
        class: class_word(f.class),
        fires_at: f.level,
        within,
        arm: f.arm.as_str(),
        hostile: f.hostile,
        suspicious: f.suspicious,
        families: f.families,
        conf: f.confidence,
    })
}

#[derive(Serialize)]
struct LlmGraded<'a> {
    class: &'static str,
    conf: f32,
    model: &'a str,
    #[serde(skip_serializing_if = "str::is_empty")]
    reason: &'a str,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    inject: bool,
}

#[derive(Serialize)]
struct LlmFailed<'a> {
    outcome: &'static str,
    error: &'a str,
    model: &'a str,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    inject: bool,
}

fn llm_object(llm: &Interpretation) -> Option<Box<RawValue>> {
    match llm {
        Interpretation::Graded(graded) => {
            fit(&without_nul(&graded.interpretation), |reason| LlmGraded {
                class: graded.grade.as_str(),
                conf: graded.blended,
                model: &graded.model,
                reason,
                inject: graded.analyzer_directed,
            })
        }
        Interpretation::Failed(failed) => fit(&without_nul(&failed.error), |error| LlmFailed {
            outcome: "error",
            error,
            model: &failed.model,
            inject: failed.analyzer_directed,
        }),
    }
}

#[derive(Serialize)]
struct Header<'a> {
    v: u8,
    scan: ScanStamp<'a>,
    files: usize,
}

#[derive(Serialize)]
struct ScanStamp<'a> {
    engine: &'a str,
    ruleset: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    level: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    worker: Option<&'a str>,
    at: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ms: Option<u64>,
}

#[derive(Serialize)]
struct FileLine<'a> {
    sha256: &'a str,
    size: u64,
    #[serde(rename = "type")]
    file_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    verdict: Option<Box<RawValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ml: Option<Box<RawValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    floor: Option<Box<RawValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    llm: Option<Box<RawValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    drift_heuristic: Option<&'a RawValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    drift_llm: Option<&'a RawValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scanners: Option<BTreeMap<Scanner, &'a RawValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    report: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ident: Option<&'a RawValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    contents: Option<Contents>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refs: Option<&'a [Ref]>,
}

fn push_line(out: &mut Vec<u8>, line: &impl Serialize) -> Option<()> {
    serde_json::to_writer(&mut *out, line)
        .inspect_err(|e| tracing::error!(error = %e, "upload: result line does not serialize"))
        .ok()?;
    out.push(b'\n');
    Some(())
}

/// A benign result for the artifact `report` describes, with its upload
/// document built, filed under `sha256`. For tests outside the engine that
/// need a result to upload.
#[cfg(test)]
pub(crate) fn fixture_result(sha256: &str, report: &CompactReport) -> super::ScanResult {
    let mut result = super::envelope::tests::base_result();
    let root = OwnReading {
        verdict: result.model,
        model: result.model,
        floor: None,
    };
    let members = MemberEvals::new();
    let kept = super::retention::kept_members(report, |_| None);
    let mut doc = UploadDoc::new(&Inputs {
        report,
        root,
        root_mods: &[],
        root_skip: &[],
        members: &members,
        kept: &kept,
        every: false,
        listed: &[],
        records: &[],
        level: Some(25),
    });
    if let Some(doc) = &mut doc {
        doc.attach_reports(report);
    }
    sha256.clone_into(&mut result.sha256);
    result.upload = doc;
    result
}

#[cfg(test)]
mod tests;
