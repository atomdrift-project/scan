//! The CLI scan drivers: walk paths, analyze, classify, print, tally.

use std::collections::HashMap;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use cleave::{AnalysisOptions, AnalysisReport};

use super::hopper::{Origin, renew};
use super::pipeline::{ClassifyRequest, OutputNeeds, classify_report};
use super::progress::{Progress, Spinner};
use super::render_cards::{write_extra_diagnostics, write_tiny};
use super::{
    ENGINE_VERSION, ScanConfig, ScanResult, ScanSummary, add_zip_passwords,
    prefetch_cleave_resources,
};
use crate::bloom_repo::Decision as BloomDecision;
use crate::explain::ShapImportance;
use crate::model::{Classification, Model};
use crate::output::BloomMark;
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
    if !path.is_file() && !path.is_dir() {
        anyhow::bail!("path does not exist: {}", path.display());
    }
    run_paths(&[path.to_path_buf()], config, None)
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
    prefetch_cleave_resources();
    let mut session = Session::start(config)?;

    // Partition the requested paths: explicit files become one unfiltered batch
    // (a named file is always analyzed), each directory is walked upfront into
    // its file list, and anything else is an error. Walking the directories here
    // — rather than letting cleave stream them — lets us learn the total file
    // count before analysis starts, so the progress bar has a denominator.
    let mut files = Vec::new();
    let mut dir_files = Vec::new();
    for path in paths {
        if path.is_file() {
            files.push(path.clone());
        } else if path.is_dir() {
            dir_files.extend(discover_files(path));
        } else {
            tracing::error!("path does not exist: {}", path.display());
            session.tally.errors.fetch_add(1, Ordering::Relaxed);
        }
    }
    session.show_progress(files.len() + dir_files.len());
    scan_partitioned(&session, &files, dir_files, registry_map)?;
    session.finish()
}

/// Analyze the named files and the files found in named directories, and
/// record each result as it completes.
fn scan_partitioned(
    session: &Session<'_>,
    files: &[PathBuf],
    dir_files: Vec<PathBuf>,
    registry_map: Option<&HashMap<String, RegistryProvenance>>,
) -> Result<()> {
    let record = |path: &Path, analyzed: Result<AnalysisReport>, named: bool| {
        // Per-file provenance: match this artifact's content sha to its registry
        // record in the map, and otherwise to the capture record a collector
        // left beside it. Both are the registry's own account of the package,
        // recovered without a refetch — which for a deleted release is the
        // only way to have it at all — so a collected sample reasons over the
        // same facts a live `purl` scan would. An explicit map still wins; a
        // file with neither simply scans without registry provenance.
        let sha = analyzed
            .as_ref()
            .ok()
            .map(|report| report.target.sha256.as_str());
        let mapped = registry_map.zip(sha).and_then(|(map, sha)| map.get(sha));
        let collected = mapped
            .is_none()
            .then(|| crate::provenance::collector_registry(path, sha?))
            .flatten();
        let origin = Origin {
            path,
            registry: mapped.or(collected.as_ref()),
            fetch: None,
        };
        session.record(&Target::new(origin, named), analyzed);
    };

    // A single artifact has no file-count bar, so its static analysis —
    // extraction, disassembly, per-member scoring — would run with nothing on
    // screen for however many seconds it takes. Analyze it directly with a
    // spinner instead of through the streaming API, dropping the spinner before
    // `record` so its fetch tree and final render own the terminal cleanly.
    if session.config.format() == OutputFormat::Terminal && files.len() == 1 && dir_files.is_empty()
    {
        let path = &files[0];
        let label = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        let spinner = Spinner::start(label);
        let analyzed = cleave::analyze_file(path, &named_target_opts(&session.opts))
            .with_context(|| format!("cleave analysis of {}", path.display()));
        drop(spinner);
        record(path, analyzed, true);
    } else {
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
        if !dir_files.is_empty() {
            cleave::scan_paths(dir_files, &session.opts, |event| {
                if let cleave::ScanEvent::File { path, result } = event {
                    record(&path, *result, false);
                }
            })?;
        }
    }
    Ok(())
}

/// Analyze in-memory bytes — a fetched artifact — exactly as a single local
/// file: classify, render, and tally. `label` is the display path (the URL or
/// PURL); `name` drives cleave's extension-based type detection. Powers the
/// `pkg`/`url` subcommands. Honors `config`'s fetch policy, so the fetched
/// package's own declared references are followed when `--fetch` is set.
///
/// `registry_fallback_purl` is `Some(purl)` when `bytes` is a registry-metadata
/// document standing in for a package that couldn't be fetched — never a real
/// artifact. Its bytes hash differently on every fetch, so its verdict is
/// routed by [`super::HopperRoute`] rather than posted under its own sha256.
///
/// # Errors
/// As [`run_paths`].
pub fn run_bytes(
    label: &str,
    name: &str,
    bytes: Vec<u8>,
    config: &ScanConfig,
    root_registry: Option<&RegistryProvenance>,
    root_fetch: Option<&fletch::fetch::FetchRecord>,
    registry_fallback_purl: Option<&str>,
) -> Result<ScanSummary> {
    // Deliberately no `prefetch_cleave_resources()` here. This one-shot single-
    // artifact path (`pkg:`/`url`) never fans out across rayon, and its analyses
    // usually hit cleave's report cache — so the capability mapper's match
    // indexes (a multi-hundred-ms regex build) are left to build lazily only if
    // an analysis actually misses, and are skipped entirely on a warm scan. The
    // lazy build then runs on this main thread, so there is no rayon re-entrancy
    // to guard against. Directory/worker paths still prefetch (they do fan out).
    let session = Session::start(config)?;
    // Content-digest short-circuit happens inside cleave via the skip predicate,
    // complementing the PURL gate in `pkg.rs`: it catches known *content* even
    // when the package locator was not in the filter. `record` emits the verdict
    // from the minimal report cleave returns on a skip.
    let analyzed = cleave::analyze_bytes_owned(bytes, name, &session.opts)
        .with_context(|| format!("cleave analysis of {label}"));
    let origin = Origin {
        path: Path::new(label),
        registry: root_registry,
        fetch: root_fetch,
    };
    // The url/purl the operator named — the fetched artifact itself, not a
    // dependency of it.
    let target = Target {
        fallback_purl: registry_fallback_purl,
        ..Target::new(origin, true)
    };
    session.record(&target, analyzed);
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
/// # Errors
/// Propagates cleave analysis failures, model inference errors, and feature
/// spec mismatches.
pub fn scan_bytes(
    data: Vec<u8>,
    filename: &str,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
) -> Result<ScanResult> {
    prefetch_cleave_resources();
    let report = cleave::analyze_bytes_owned(data, filename, &analysis_options(config))
        .with_context(|| format!("cleave analysis of {filename}"))?;
    let classifier = Classifier {
        config,
        model,
        shap,
        cancel: None,
    };
    classifier.classify(report, Origin::local(Path::new(filename)), None)
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
/// the payload's real extension.
///
/// # Errors
/// Propagates cleave analysis failures and model inference errors.
pub fn scan_file(
    read_path: &Path,
    filename: &str,
    model: &Model,
    shap: Option<&ShapImportance>,
    config: &ScanConfig,
) -> Result<ScanResult> {
    prefetch_cleave_resources();
    let report = cleave::analyze_file(read_path, &analysis_options(config))
        .with_context(|| format!("cleave analysis of {filename}"))?;
    let classifier = Classifier {
        config,
        model,
        shap,
        cancel: None,
    };
    classifier.classify(report, Origin::local(Path::new(filename)), None)
}

/// The cleave options a config implies, before any per-scan additions.
fn analysis_options(config: &ScanConfig) -> AnalysisOptions {
    // Part of cleave's analysis cache key, not only a retention setting: every
    // analysis must set it the same way.
    cleave::set_compact_member_retention(true);
    let mut opts = AnalysisOptions {
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
    model: Model,
    shap: Option<ShapImportance>,
    /// What every file is analyzed with, the bloom skip predicate and the
    /// Ctrl-C flag included.
    opts: AnalysisOptions,
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
    /// The one setup every CLI entry point shares: load the model, arm Ctrl-C,
    /// and build the analysis options and the hopper uploader.
    fn start(config: &'a ScanConfig) -> Result<Self> {
        let model = Model::load(config.model_dir(), config.thresholds(), config.level())?;
        let shap = ShapImportance::load(config.model_dir())?;
        let cancel = crate::interrupt::arm("Interrupted — finishing current file…");
        let mut opts = analysis_options(config);
        opts.cancellation = Some(Arc::clone(&cancel));
        opts.skip_predicate = bloom_skip_predicate(config);
        let uploader = config
            .hopper()
            .map(|url| crate::upload::Uploader::new(url, crate::upload::default_worker_name()));
        Ok(Self {
            config,
            model,
            shap,
            opts,
            cancel,
            tally: Tally::default(),
            progress: None,
            uploader,
            broken_pipe: AtomicBool::new(false),
            started: Instant::now(),
        })
    }

    /// Print the banner and start the bar, when a terminal scan covers more
    /// than one file.
    fn show_progress(&mut self, files: usize) {
        if self.config.format() == OutputFormat::Terminal && files > 1 {
            crate::output::print_banner(detection_counts(self.config));
            self.progress = Some(Progress::new(u32::try_from(files).unwrap_or(u32::MAX)));
        }
    }

    fn classifier(&self) -> Classifier<'_> {
        Classifier {
            config: self.config,
            model: &self.model,
            shap: self.shap.as_ref(),
            cancel: Some(self.cancel.as_ref()),
        }
    }

    /// Classify one analyzed file and account for it: tally, print, renew.
    /// Called from rayon workers, so every shared input is behind `&`/atomics.
    fn record(&self, target: &Target<'_>, analyzed: Result<AnalysisReport>) {
        // The bloom verdict, re-derived from the sha cleave computed. A file
        // cleave skipped at our request (a stale known-good, or a fast-mode
        // unknown) arrives as a minimal report and is counted here without an
        // ML pass. A known-bad or conflicted file — or a known-good scanned on
        // its own merits — was analyzed, and carries a mark on its SHA-256 line.
        let mut bloom_mark = None;
        if let (Some(lookup), Ok(report)) = (self.config.bloom(), &analyzed)
            && let Some(digest) = burton::parse_sha256_hex(&report.target.sha256)
        {
            let decision = lookup.decide_sha256(&digest);
            // Two known-good files are still scanned on their own merits, and in
            // both the skip predicate already declined to skip, so cleave has
            // produced a full report: one created/changed/modified within
            // KNOWN_GOOD_RESCAN, and one the operator named on the command line.
            // The second is a policy, not an optimization: a bless is a bulk
            // shortcut for dependencies and directory walks, and returning one
            // where a scan was asked for answers a different question. The mark
            // is still emitted, so a named known-bad still says so.
            let path = target.origin.path;
            let scan_anyway = decision == BloomDecision::Skip
                && (target.named
                    || file_touched_within(path, KNOWN_GOOD_RESCAN, SystemTime::now()));
            if !scan_anyway
                && let Some(summary) =
                    bloom_gate(self.config, &path.display().to_string(), decision)
            {
                if summary.benign > 0 {
                    self.tally.count(Classification::Benign);
                }
                self.advance();
                return;
            }
            // Non-fast unknown maps to `None` (unremarkable, matched neither set).
            bloom_mark = BloomMark::from_decision(decision);
        }

        let classified = analyzed.and_then(|report| {
            self.classifier()
                .classify(report, target.origin, bloom_mark)
        });
        self.advance();
        match classified {
            Ok(result) => self.report(target, result),
            Err(error) => self.report_error(target.origin.path, &error),
        }
    }

    fn advance(&self) {
        if let Some(progress) = &self.progress {
            progress.increment();
        }
    }

    fn report(&self, target: &Target<'_>, result: ScanResult) {
        self.tally.count(result.classification);
        // An adversely bloom-marked file is always surfaced — its flag replaces
        // the old unconditional banner, so the benign filter must not swallow a
        // known-bad file the model happens to rate benign. JSON, tiny, and
        // interpret are machine/LLM payload formats: they emit every scanned
        // file — `--show` gates only the terminal view.
        let shown = matches!(
            self.config.format(),
            OutputFormat::Json | OutputFormat::Tiny | OutputFormat::Interpret
        ) || result
            .bloom_mark
            .is_some_and(BloomMark::forces_terminal_display)
            || self.config.filter().shows(&result.classification);
        if shown && !self.broken_pipe.load(Ordering::Relaxed) {
            let written = match &self.progress {
                Some(progress) => progress.around_result(|| emit_result(&result, self.config)),
                None => emit_result(&result, self.config),
            };
            if let Err(error) = written {
                self.output_failed(&error);
            }
        }
        // Renewed after the result is printed, so local output never waits on
        // the handoff; the upload itself runs on the uploader's thread. Only
        // successful results are sent — an error envelope has an empty file
        // type, which hopper treats as a delete.
        if let (Some(uploader), Some(hopper_url)) = (&self.uploader, self.config.hopper()) {
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
        tracing::error!("error analyzing {}: {}", path.display(), msg);
        // A failed file still gets a line on stdout under `--format json`, so a
        // consumer reading the NDJSON sees *that* it failed and why, instead of
        // inferring it from a record that never arrives. This does not make the
        // failure anything other than a failure: the tally below still counts it,
        // the process still exits 3, and nothing is handed to hopper — an error
        // envelope carries an empty file type, which hopper reads as a delete.
        //
        // `raw.files` is deliberately empty, and that is load-bearing for
        // compatibility. Readers that predate this record skip a fileless entry
        // and report the sample as having produced no result — which is exactly
        // right. A record carrying a file entry would instead be scored, and an
        // absent `ml.lvl` decodes to 0 in a JSON reader that defaults its
        // numbers, which reads as *hostile*. Keep the list empty.
        if self.config.format() == OutputFormat::Json
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
    /// Stdout closed under the scan, which therefore stopped early.
    fn finish(self) -> Result<ScanSummary> {
        if let Some(progress) = &self.progress {
            progress.finish();
        }
        drop(self.uploader);
        if self.broken_pipe.load(Ordering::Relaxed) {
            return Err(io::Error::from(io::ErrorKind::BrokenPipe))
                .context("writing scan results to stdout");
        }
        let summary = self.tally.summary(self.started);
        if self.config.format() == OutputFormat::Terminal {
            crate::output::print_summary(&summary);
        }
        Ok(summary)
    }
}

/// One file of a CLI scan, and what the scan knows about it besides its report.
#[derive(Clone, Copy)]
struct Target<'a> {
    origin: Origin<'a>,
    /// The operator named it on the command line, so it is analyzed on its own
    /// merits and never answered from a bloom bless (see `named_target_opts`).
    named: bool,
    /// Set when the bytes are a registry-metadata document standing in for a
    /// package that could not be fetched; see [`super::HopperRoute`].
    fallback_purl: Option<&'a str>,
}

impl<'a> Target<'a> {
    const fn new(origin: Origin<'a>, named: bool) -> Self {
        Self {
            origin,
            named,
            fallback_purl: None,
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
    cancel: Option<&'a AtomicBool>,
}

impl Classifier<'_> {
    /// Classify `report` the way the config asks: the renders its format
    /// needs, its fetch policy and LLM, and the report kept when JSON output or
    /// a hopper renewal will read it. Always returns a result, benign included;
    /// the caller decides whether to display it.
    fn classify(
        &self,
        report: AnalysisReport,
        origin: Origin<'_>,
        bloom_mark: Option<BloomMark>,
    ) -> Result<ScanResult> {
        let config = self.config;
        let label = origin.path.display().to_string();
        let json = config.format() == OutputFormat::Json;
        let classified = classify_report(
            report,
            ClassifyRequest {
                shap: self.shap,
                cancellation: self.cancel,
                tiny_opts: tiny_opts_for(config),
                interpret: config.interpret(),
                fetch: config.fetch_policy(),
                zip_passwords: config.zip_passwords(),
                needs: OutputNeeds {
                    llm_view: config.format() == OutputFormat::Interpret,
                    // The live fetch log renders only on the interactive
                    // terminal path; JSON and tiny stay machine-clean (the edges
                    // ride along in the JSON `fetched` array regardless).
                    fetch_progress: config.format() == OutputFormat::Terminal,
                    render_context: !json,
                    // `--show=all` with JSON output: list every archive member,
                    // even the no-finding ones cleave skipped analyzing.
                    list_all_members: json && config.filter().is_all(),
                    deps_for_upload: config.hopper().is_some(),
                },
                root_registry: origin.registry,
                root_fetch: origin.fetch,
                bloom_mark,
                ..ClassifyRequest::new(&label, origin.path, self.model)
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
        // Engine-internal attribution (per-phase thread-time, regex store churn)
        // for slow-scan diagnosis; one line per stat at info.
        cleave::log_scan_stats();

        // JSON output and hopper renewals carry the report (so hopper stores it
        // and explodes archive members); the terminal never reads it.
        let keep_report = json || config.hopper().is_some();
        Ok(classified.into_scan_result(label, self.model, keep_report))
    }
}

/// Write one result to stdout in the configured format.
///
/// # Errors
/// The write failed; `BrokenPipe` means nobody is reading any more.
fn emit_result(r: &ScanResult, config: &ScanConfig) -> io::Result<()> {
    let mut out = io::stdout().lock();
    match config.format() {
        OutputFormat::Json => {
            serde_json::to_writer(&mut out, &r.envelope_ref())?;
            out.write_all(b"\n")
        }
        // The terminal result card was built in `classify_report`; write it
        // verbatim, then flush before the stderr footer.
        OutputFormat::Terminal => {
            out.write_all(r.rendered_context.as_bytes())?;
            if !r.rendered_context.ends_with('\n') {
                out.write_all(b"\n")?;
            }
            // `--extra`: append the full ML diagnostics that explain the grade —
            // which route drove it and the top SHAP features behind it.
            if config.extra() {
                write_extra_diagnostics(&mut out, r)?;
            }
            // The closing summary is written to stderr. Commit this whole card
            // first so stdio buffering cannot strand its final trait beneath the
            // footer (especially visible for a single-file scan).
            out.flush()
        }
        // `--format tiny` prefixes the machine verdict line, never colored.
        OutputFormat::Tiny => write_tiny(&mut out, r),
        // `--format interpret` is the LLM payload verbatim — no verdict line.
        OutputFormat::Interpret => out.write_all(r.rendered_context.as_bytes()),
    }
}

/// Write a JSON error record for a file that failed analysis.
fn write_error_record(file_path: &Path, msg: &str) -> io::Result<()> {
    let path = file_path.display().to_string();
    let record = ErrorRecord {
        err: ErrSection {
            path: &path,
            msg,
            eng: ENGINE_VERSION,
        },
        raw: EmptyRaw { files: [] },
    };
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, &record)?;
    out.write_all(b"\n")
}

/// Cleave context density for machine/LLM output. The primary terminal artifact
/// and its fetched appendix use Scan's own global three-trait cards.
pub(crate) fn tiny_opts_for(config: &ScanConfig) -> cleave::output::TinyOpts {
    if matches!(
        config.format(),
        OutputFormat::Tiny | OutputFormat::Interpret
    ) {
        cleave::output::TinyOpts::tiny()
    } else {
        cleave::output::TinyOpts {
            // Keep five findings per file in machine-facing compact output.
            top_n: 5,
            always_crit: None,
            // Focus on suspicious+ (plus their composite legs) whenever any
            // fired; a merged capture window renders only selected rows, and a
            // suspicious+ hit keeps one trailing row/line of context — the
            // continuation tends to carry the payoff. Unremarkable dependency
            // files fall back to their notable top-five.
            focus_crit: Some(cleave::Criticality::Suspicious),
            // Card layout keeps the compact render headerless.
            card: true,
            // Only the hit lines/rows — no surrounding context, no `⋯` gap
            // markers, no padding rows in the hex view.
            context_lines: Some(1),
            full_context: false,
            header: cleave::output::HeaderStyle::Rich,
            ..cleave::output::TinyOpts::terminal()
        }
    }
}

/// Apply a bloom-filter decision before any expensive analysis. Emits the
/// verdict and returns `Some(summary)` when the artifact is short-circuited (a
/// known-good skip, or — in fast mode — a bloom-only verdict); `None` to proceed.
pub(crate) fn bloom_gate(
    config: &ScanConfig,
    label: &str,
    decision: BloomDecision,
) -> Option<ScanSummary> {
    use crate::output::BloomVerdict;
    let fast = config.mode() == Mode::Fast;
    let format = config.format();
    // Known-good/unscanned are benign-tier: print the per-file line only when
    // benign output is requested, so a large directory scan isn't spammed. The
    // aggregate tally always reports them in the summary regardless.
    let show_quiet = format == OutputFormat::Json || config.filter().shows(&Classification::Benign);
    // Tally for end-of-scan observability. `unscanned` only matters for Unknown.
    crate::bloom_repo::record(decision, fast && decision == BloomDecision::Unknown);
    match decision {
        BloomDecision::Skip => {
            if show_quiet {
                crate::output::print_bloom_verdict(label, BloomVerdict::KnownGood, format);
            }
            Some(one_file_summary(0, 0, 1))
        }
        // Known-bad and conflicted are flags, not short-circuits: they run full
        // analysis in every mode — the scan is the verdict — and the flag rides
        // the normal result inline (see `BloomMark`), so no separate banner is
        // printed here. The caller derives the mark from this same decision.
        // A sighting is the same shape of answer as known-bad — a flag that
        // runs the full analysis — but says somebody else reported it rather
        // than that we measured it. The distinction rides the inline mark.
        BloomDecision::KnownBad
        | BloomDecision::SightedHostile
        | BloomDecision::SightedSuspicious => None,
        BloomDecision::Conflicted => {
            // Build-time subtraction removes bad keys from good, so a conflict can
            // only mean filter version skew or a producer bug — it should never
            // happen. Always scan, but make the noise loud (in the logs).
            tracing::warn!(
                label,
                "bloom conflict: key is in BOTH the good and bad filters (should never happen) — scanning"
            );
            None
        }
        // Unknown is left unscanned only in bloom-only fast mode; otherwise scanned.
        BloomDecision::Unknown => fast.then(|| {
            if show_quiet {
                crate::output::print_bloom_verdict(label, BloomVerdict::Unscanned, format);
            }
            one_file_summary(0, 0, 0)
        }),
    }
}

/// The bloom short-circuit for a scan target, honoring the local known-good
/// freshness override: a known-good file created, status-changed, or modified
/// within the last 48h is scanned on its own merits rather than skipped on its
/// bloom vouch. Returns `Some(summary)` when the target is short-circuited
/// (counted, not scanned) and `None` when it must be scanned. Used by process
/// scans, which decide before analysis; path scans apply the same override in
/// [`Session::record`], where cleave has already produced a minimal report by
/// the time the decision is known and a fresh known-good is re-analyzed in place.
pub(crate) fn bloom_gate_fresh(
    config: &ScanConfig,
    path: &Path,
    decision: BloomDecision,
) -> Option<ScanSummary> {
    if decision == BloomDecision::Skip
        && file_touched_within(path, KNOWN_GOOD_RESCAN, SystemTime::now())
    {
        return None;
    }
    bloom_gate(config, &path.display().to_string(), decision)
}

/// The analysis options for a target the operator named by path, which is the
/// shared options with the bloom skip predicate removed.
///
/// A named target is always analyzed on its own merits, so cleave must not
/// short-circuit it into a minimal report. The gate in [`Session::record`]
/// makes the same call for the same reason; both are needed, because they act at
/// different points — the predicate decides whether analysis runs at all, the
/// gate decides whether the result is counted without an ML pass.
fn named_target_opts(opts: &AnalysisOptions) -> AnalysisOptions {
    AnalysisOptions {
        skip_predicate: None,
        ..opts.clone()
    }
}

/// Build the cleave skip predicate from the active bloom filters: a file whose
/// sha256 is known-good (or, in fast mode, simply unknown) is skipped before
/// analysis. Known-bad and conflicted return `false` so they are still analyzed;
/// every file's verdict line is emitted later in [`Session::record`], which
/// re-derives the decision from the report's sha. `None` when bloom is disabled.
fn bloom_skip_predicate(config: &ScanConfig) -> Option<cleave::SkipPredicate> {
    let lookup = config.bloom_arc()?;
    let fast = config.mode() == Mode::Fast;
    Some(cleave::SkipPredicate(Arc::new(
        move |sha_hex: &str, path: &Path| {
            let Some(digest) = burton::parse_sha256_hex(sha_hex) else {
                return false;
            };
            let decision = lookup.decide(&burton::Artifact::sha256(&digest));
            // Known-good is trusted and skipped, unless the file was created,
            // status-changed, or modified within the last 48h — a fresh
            // known-good is analyzed on its own merits (recent activity, and a
            // guard against a bloom false positive on a freshly planted file),
            // so cleave analyzes it once here rather than skipping and then
            // re-analyzing in `Session::record`.
            //
            // Every other decision is analyzed, adverse or not: in fast mode an
            // unknown is skipped as well, which is what fast mode means.
            if decision.may_skip() {
                !file_touched_within(path, KNOWN_GOOD_RESCAN, SystemTime::now())
            } else {
                fast && decision == BloomDecision::Unknown
            }
        },
    )))
}

/// A one-file [`ScanSummary`] for a bloom-decided artifact that was not analyzed.
const fn one_file_summary(hostile: u32, suspicious: u32, benign: u32) -> ScanSummary {
    ScanSummary {
        total_files: 1,
        hostile,
        suspicious,
        benign,
        errors: 0,
        duration_ms: 0,
    }
}

/// Enumerate the regular files under `dir` for a directory scan, mirroring the
/// structural filters cleave applies during its own walk: skip `.git*` trees,
/// keep only regular files, never follow symlinks. This reads no file contents —
/// it is a cheap `readdir` pass whose purpose is to learn the file count (and
/// the list) upfront, so the progress bar has a denominator before analysis
/// begins. The list is handed to [`cleave::scan_paths`], which still applies its
/// program-type and size filters per file, so a few of these may be analyzed
/// away without a verdict (the bar tops out just under 100%, then the summary
/// reports the true total).
fn discover_files(dir: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !e.file_name().to_string_lossy().starts_with(".git"))
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.path().to_path_buf())
        .filter(|p| !is_attached_sidecar(p))
        .collect()
}

/// Whether `path` is a collector's provenance sidecar sitting beside the
/// artifact it describes.
///
/// A directory walk skips those. The sidecar is a record *about* the file next
/// to it, not something anyone shipped: analyzing it spends a verdict on our own
/// bookkeeping, and with `--hopper` files that bookkeeping as a sample in its own
/// right — where it competes with the artifact it was written to describe.
/// It is still read as provenance for that artifact (see
/// [`crate::provenance::collector_sidecar`]), and naming one directly on the
/// command line still scans it, which is what a scan of a suspect JSON file
/// should do.
fn is_attached_sidecar(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let Some(artifact) = name.strip_suffix(crate::provenance::COLLECTOR_SIDECAR_SUFFIX) else {
        return false;
    };
    !artifact.is_empty() && path.with_file_name(artifact).is_file()
}

/// The two distinct detection inventories shown in the banner. Bloom entries
/// are SHA-256/PURL signatures; traits, composites, and YARA are actual rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DetectionCounts {
    pub(crate) hashes_and_purls: u64,
    pub(crate) rules: u64,
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
    /// one of the four counters, so their sum is the total file count.
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
            duration_ms: crate::duration_ms(scan_start.elapsed()),
        }
    }
}

/// Freshness window for the local known-good re-scan: a known-good file created,
/// changed, or modified this recently is scanned on its own merits rather than
/// skipped.
///
/// Deliberately its own value rather than the dependency window in
/// [`crate::fetch`]. The two answer different questions: that one asks how long
/// a *registry release* stays too new to trust a vouch for, and is sized in
/// hours because a published version is immutable; this one asks how recently a
/// *local file* was written, and is the guard against a bloom false-positive
/// shielding something a live intrusion just planted. Shrinking it to match the
/// dependency window would narrow that guard for no reason connected to it.
const KNOWN_GOOD_RESCAN: Duration = Duration::from_secs(48 * 3_600);

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
        let ctime = md.ctime();
        if ctime >= 0 && recent(UNIX_EPOCH + Duration::from_secs(ctime.unsigned_abs())) {
            return true;
        }
    }
    false
}

/// The `err` section of an error record: the file that failed, and why.
#[derive(Debug, serde::Serialize)]
struct ErrSection<'a> {
    path: &'a str,
    msg: &'a str,
    /// Scan engine build that produced this record, mirroring [`MlSection::eng`]
    /// so an error line is attributable to a build like a successful one is.
    eng: &'static str,
}

/// An error line: `{"err": {...}, "raw": {"files": []}}`.
///
/// Shaped to sit in the same NDJSON stream as [`ScanResultEnvelopeRef`] without
/// being mistaken for one. See the call site for why `raw.files` must stay empty.
#[derive(Debug, serde::Serialize)]
struct ErrorRecord<'a> {
    err: ErrSection<'a>,
    raw: EmptyRaw,
}

#[derive(Debug, serde::Serialize)]
struct EmptyRaw {
    files: [(); 0],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_touched_within_flags_fresh_and_ignores_old_and_missing() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("x.txt");
        std::fs::write(&f, b"hi").unwrap();
        let now = SystemTime::now();
        // Just-written file is within the window.
        assert!(file_touched_within(&f, KNOWN_GOOD_RESCAN, now));
        // Evaluated from a clock 10 days ahead, every timestamp is well outside
        // the 48h window — the not-recent (skip-eligible) case.
        let later = now + Duration::from_secs(10 * 86_400);
        assert!(!file_touched_within(&f, KNOWN_GOOD_RESCAN, later));
        // Unreadable metadata (e.g. a fetched artifact's synthetic path) is treated
        // as not-recent, so the normal known-good skip still applies.
        assert!(!file_touched_within(
            &dir.path().join("does-not-exist"),
            KNOWN_GOOD_RESCAN,
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

    /// A directory of collected samples is the normal input, and its sidecars
    /// describe the artifacts beside them — they are not themselves samples.
    #[test]
    fn directory_walk_skips_attached_sidecars_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        for name in [
            "evil-1.0.0.tgz",
            "evil-1.0.0.tgz.forage.json",
            "orphan.tgz.forage.json",
            "notes.json",
        ] {
            std::fs::write(dir.path().join(name), b"{}").expect("write");
        }
        let mut found: Vec<String> = super::discover_files(dir.path())
            .iter()
            .filter_map(|p| p.file_name()?.to_str().map(String::from))
            .collect();
        found.sort();
        assert_eq!(
            found,
            vec![
                "evil-1.0.0.tgz".to_string(),
                "notes.json".to_string(),
                // Nothing on disk claims this one, so it is scanned like any
                // other file rather than silently trusted.
                "orphan.tgz.forage.json".to_string(),
            ]
        );
    }
}
