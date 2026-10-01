//! One analysis on behalf of a request: claiming capacity, running it on a
//! blocking thread, and filing what it found.
//!
//! Every analyze route goes through here — the single-flight routes through
//! [`lead`], `/analyze-path` through [`RequestGuard::run`] and [`record`]
//! directly — so admission, the `/_/requests` entry, the timeout, the
//! completion log and the `/_/stats` figures are one code path, not five
//! copies that drift.

use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use tokio::runtime::Handle;

use super::corpus::{Corpus, Reached};
use super::error::ApiError;
use super::flight::{Flight, Outcome};
use super::{AnalysisOutcome, AnalysisPermit, AppState, InFlightRequest, RequestGuard};
use crate::analysis::{Analysis, ModelResources, RequestPhase};
use crate::engine::{HopperRoute, ScanResult};
use crate::upload::Uploader;

/// Request-scoped traversal and storage behavior.
#[derive(Clone, Copy)]
pub(super) struct RequestFollow {
    /// Which references discovered in the artifact are followed.
    pub(super) policy: crate::fetch::FetchPolicy,
    /// Replace a stored verdict rather than keep the first (`refresh=1`).
    pub(super) refresh: bool,
}

/// Bytes a request uploaded, staged on disk and waiting to be analyzed.
#[derive(Debug)]
pub(super) struct Upload {
    /// Owns the directory holding [`Self::path`]: dropping it deletes the file.
    pub(super) _dir: tempfile::TempDir,
    /// The staged file, named so cleave can detect its type from the extension.
    pub(super) path: PathBuf,
    /// Sanitized upload filename, used as the display path.
    pub(super) filename: String,
    pub(super) size_bytes: u64,
}

impl Upload {
    /// A temp directory holding `bytes` under `filename`. Blocking.
    fn stage(filename: &str, bytes: &[u8]) -> std::io::Result<Self> {
        let dir = tempfile::Builder::new().prefix("scan-").tempdir()?;
        let path = dir.path().join(filename);
        std::fs::write(&path, bytes)?;
        Ok(Self {
            _dir: dir,
            path,
            filename: filename.to_owned(),
            size_bytes: bytes.len() as u64,
        })
    }
}

/// What a flight's leader analyzes.
pub(super) enum Job {
    /// Bytes already staged on disk (multipart `/analyze`).
    Upload(Upload),
    /// Bytes in memory, staged on the analysis thread (`/v1/analyze`).
    Bytes {
        filename: String,
        bytes: bytes::Bytes,
    },
    /// A package, fetched with its registry record.
    Purl(String),
    /// An exact URL, fetched verbatim.
    Url(String),
}

impl Job {
    /// The payload size when it is known before the fetch.
    fn size_hint(&self) -> Option<u64> {
        match self {
            Self::Upload(upload) => Some(upload.size_bytes),
            Self::Bytes { bytes, .. } => Some(bytes.len() as u64),
            Self::Purl(_) | Self::Url(_) => None,
        }
    }

    /// How `/_/requests` and the phase tracker name the run.
    fn name(&self) -> &str {
        match self {
            Self::Upload(upload) => &upload.filename,
            Self::Bytes { filename, .. } => filename,
            Self::Purl(name) | Self::Url(name) => name,
        }
    }
}

/// Take the bundle snapshot and the capacity an analysis needs, or refuse.
///
/// Refusing past full, rather than queueing. A queue here would hide the one
/// fact the router most needs: a rejection is information, delivered in
/// milliseconds, and beamline answers it by promoting the next arm onto a
/// worker that has room. A queued request looks identical to a slow one from
/// outside, so the router keeps choosing a saturated worker while an idle one
/// waits — and the queued analysis still runs after somebody else has already
/// answered, which is the duplicate work single-flight exists to prevent.
pub(super) fn admit(
    state: &AppState,
    request_id: u64,
    subject: &dyn Display,
    size_hint: Option<u64>,
) -> Result<(Arc<ModelResources>, AnalysisPermit), ApiError> {
    let resources = state.resources().inspect_err(|_| {
        tracing::debug!(id = request_id, "rejected: resources not yet loaded");
    })?;
    let permit = acquire(state, size_hint).inspect_err(|_| {
        tracing::warn!(id = request_id, key = %subject, "rejecting: at capacity");
    })?;
    Ok((resources, permit))
}

/// One analysis permit under whichever admission scheme is active. With slot
/// lanes on, a full lane is a fast 429 with a Retry-After hint rather than a
/// queue; without them this is the flat try-acquire it always was.
/// `size_hint` `None` (an unfetched PURL or URL) classes as a whale, the safe
/// direction.
fn acquire(state: &AppState, size_hint: Option<u64>) -> Result<AnalysisPermit, ApiError> {
    let slot = match &state.lanes {
        None => Arc::clone(&state.slots)
            .try_acquire_owned()
            .map_err(|_no_permit| {
                let max = state.config.workers;
                ApiError::at_capacity(
                    format!("At capacity ({max}/{max} active analyses)"),
                    None,
                    None,
                )
            })?,
        Some(lanes) => {
            let small = size_hint.is_some_and(|size| size <= lanes.small_max_bytes);
            // A small analysis is seconds, so its hint is "come right back"; a
            // whale's is long enough for a router to prefer an idle server.
            let (lane, label, retry_after_secs) = if small {
                (&lanes.small, "small", 5)
            } else {
                (&lanes.whale, "whale", 30)
            };
            Arc::clone(lane).try_acquire_owned().map_err(|_no_permit| {
                ApiError::at_capacity(
                    format!("{label} lane at capacity"),
                    Some(label),
                    Some(retry_after_secs),
                )
            })?
        }
    };
    // A slot without a core is a queue, and the router was promised a
    // refusal rather than a wait.
    let cpu = Arc::clone(&state.cpu)
        .try_acquire_owned()
        .map_err(|_no_permit| {
            ApiError::at_capacity("At capacity (every core busy)", None, Some(30))
        })?;
    Ok(AnalysisPermit::new(slot, cpu))
}

/// Start the analysis behind a flight this request leads, and publish its
/// outcome to every request attached to it.
///
/// The analysis is server-owned work, not part of the request: the leader
/// hanging up does not abandon its followers, and shutdown drains it.
pub(super) fn lead(
    state: &Arc<AppState>,
    request_id: u64,
    flight: &Arc<Flight>,
    job: Job,
    follow: RequestFollow,
) {
    let publisher = state.flights.publisher(flight);
    let (resources, permit) = match admit(state, request_id, flight.key(), job.size_hint()) {
        Ok(admitted) => admitted,
        Err(refusal) => return publisher.publish(Outcome::Failed(refusal)),
    };
    let leader = Leader {
        state: Arc::clone(state),
        request_id,
        flight: Arc::clone(flight),
        follow,
    };
    state.tasks.spawn(async move {
        let outcome = leader.run(job, resources, permit).await;
        publisher.publish(outcome);
    });
}

/// The analysis a flight's leader runs.
struct Leader {
    state: Arc<AppState>,
    request_id: u64,
    flight: Arc<Flight>,
    follow: RequestFollow,
}

impl Leader {
    async fn run(
        self,
        job: Job,
        resources: Arc<ModelResources>,
        permit: AnalysisPermit,
    ) -> Outcome {
        let phase = RequestPhase::with_label(format!("req#{} {}", self.request_id, job.name()));
        self.flight.set_phase(phase.clone());
        let entry = InFlightRequest::new(
            job.name(),
            job.size_hint().unwrap_or(0),
            self.flight.cancellation(),
            phase.clone(),
        );
        let cancellation = Arc::clone(&entry.cancellation);
        let started = Instant::now();

        let slow_rule_ms = self.state.config.slow_rule_ms;
        let policy = self.follow.policy;
        let uploader = self.state.uploader.clone();
        let corpus = self.state.corpus.clone();
        // Captured here, on the runtime, for the one corpus call the PURL
        // fallback makes from the blocking thread.
        let runtime = Handle::current();
        let tracker = phase.clone();
        let work = move || {
            let analysis = Analysis {
                cancellation: Some(&cancellation),
                phase: Some(&tracker),
                follow: policy,
                ..Analysis::new("", &resources, slow_rule_ms)
            };
            // Fetched dependencies ride the hopper renewal, when there is one.
            let deps_for_upload = uploader.is_some();
            match job {
                Job::Upload(upload) => classify_upload(&upload, analysis, uploader.as_deref()),
                Job::Bytes { filename, bytes } => {
                    let upload = Upload::stage(&filename, &bytes)
                        .map_err(|e| anyhow::anyhow!("staging the upload: {e}"))?;
                    drop(bytes);
                    classify_upload(&upload, analysis, uploader.as_deref())
                }
                Job::Purl(purl) => classify_purl(
                    &purl,
                    Analysis {
                        deps_for_upload,
                        ..analysis
                    },
                    uploader.as_deref(),
                    corpus.as_deref().map(|corpus| (corpus, &runtime)),
                ),
                Job::Url(url) => classify_url(
                    &url,
                    Analysis {
                        deps_for_upload,
                        ..analysis
                    },
                    uploader.as_deref(),
                ),
            }
        };
        let outcome = RequestGuard::new(&self.state, self.request_id, entry, permit)
            .run(work)
            .await;

        let key = self.flight.key();
        let finished = Finished {
            request_id: self.request_id,
            key,
            purl: key.purl(),
            elapsed_ms: crate::duration_ms(started.elapsed()),
            phases: phase.timeline(),
        };
        match record(&self.state, &finished, outcome) {
            Ok(result) => self.file(result).await,
            Err(refusal) => Outcome::Failed(refusal),
        }
    }

    /// Index the verdict and renew it on hopper before followers are answered,
    /// so a lookup that follows the answer finds it. Off the reactor: the index
    /// is files, and a full upload queue blocks.
    async fn file(&self, mut result: Box<ScanResult>) -> Outcome {
        let state = Arc::clone(&self.state);
        let purl = self.flight.key().purl().map(str::to_owned);
        let refresh = self.follow.refresh;
        let filed = tokio::task::spawn_blocking(move || {
            index_verdict(&result, purl.as_deref(), refresh);
            if let Some(uploader) = &state.uploader {
                renew(uploader, &mut result, purl);
            }
            result
        })
        .await;
        match filed {
            Ok(result) => Outcome::Report(result),
            Err(e) => {
                tracing::error!(id = self.request_id, error = %e, "filing the verdict panicked");
                Outcome::Failed(ApiError::internal())
            }
        }
    }
}

/// A finished analysis, for its completion line.
pub(super) struct Finished<'a> {
    pub(super) request_id: u64,
    /// What the request was about, as the log names it.
    pub(super) key: &'a (dyn Display + Sync),
    /// The package it was named by, for the per-type figures.
    pub(super) purl: Option<&'a str>,
    pub(super) elapsed_ms: u64,
    /// Where the wall time went, from [`RequestPhase::timeline`].
    pub(super) phases: String,
}

/// Log a finished analysis and count it in the routing figures; turn a failure
/// into its answer. Every analyze route reports here, so none of them can
/// drift out of `/_/stats`.
pub(super) fn record(
    state: &AppState,
    finished: &Finished<'_>,
    outcome: AnalysisOutcome,
) -> Result<Box<ScanResult>, ApiError> {
    let Finished {
        request_id,
        key,
        purl,
        elapsed_ms,
        ref phases,
    } = *finished;
    match outcome {
        AnalysisOutcome::Ok(Ok(result)) => {
            tracing::info!(
                id = request_id,
                key = %key,
                elapsed_ms,
                phases = %phases,
                classification = %result.classification,
                probability = result.probability,
                analysis = analysis_source(&result),
                llm = llm_source(result.interpretation.as_ref()),
                // Where this verdict goes next. `queued` hands it to the
                // uploader thread, whose own line reports whether it landed;
                // `disabled` means the server was started without --hopper and
                // the answer lives only in this process's index.
                hopper = if state.uploader.is_some() {
                    "queued"
                } else {
                    "disabled"
                },
                "<-- 200 OK",
            );
            state.jobs.finished(&result, elapsed_ms, purl);
            Ok(result)
        }
        AnalysisOutcome::Ok(Err(e)) => {
            let refusal = ApiError::from_analysis(&e);
            // The whole chain: `Display` on an anyhow error prints only the
            // outermost context, which never says why.
            tracing::warn!(id = request_id, key = %key, elapsed_ms, status = refusal.status.as_u16(), error = %format!("{e:#}"), "<-- analysis failed");
            Err(refusal)
        }
        AnalysisOutcome::JoinError(e) => {
            tracing::warn!(id = request_id, key = %key, elapsed_ms, error = %e, "<-- 500 task join error (panic?)");
            Err(ApiError::internal())
        }
        AnalysisOutcome::Timeout(secs) => {
            tracing::warn!(id = request_id, key = %key, elapsed_ms, timeout_secs = secs, "<-- 504 analysis timeout");
            Err(ApiError::timeout(secs))
        }
    }
}

/// Where this result's analysis came from, for the completion log line.
///
/// `cached` means cleave replayed the whole report from its on-disk cache
/// (SQLite, keyed by content digest, options, and traits revision) rather than
/// running the pipeline. It survives restarts, so a fast response is not
/// evidence of a warm process. A request that instead rode *another request's*
/// in-flight run reports `shared=true` on its access line — that path never
/// reaches here, because only the leader logs the completion.
fn analysis_source(result: &ScanResult) -> &'static str {
    if result.analysis_cached {
        "cached"
    } else {
        "fresh"
    }
}

/// Where this result's LLM verdict came from, for the completion log line.
///
/// `--interpret` dominates a request's wall time when it actually queries the
/// endpoint and costs nothing when the verdict is replayed from the prompt
/// cache — a minute versus a tenth of a second on the same sample. Naming the
/// source turns that difference from a timing anomaly into a fact on the line.
/// `None` when no pass ran, which omits the field.
fn llm_source(interpretation: Option<&crate::interpret::Interpretation>) -> Option<&'static str> {
    use crate::interpret::Interpretation;
    Some(match interpretation? {
        Interpretation::Failed(_) => "failed",
        Interpretation::Graded(graded) if graded.cached => "cached",
        Interpretation::Graded(_) => "queried",
    })
}

/// Record what an analysis found, so a later lookup of the same artifact is
/// answerable without re-running it. Blocking: the index is files.
/// Best-effort — a lookup that misses is a normal answer.
pub(super) fn index_verdict(result: &ScanResult, purl: Option<&str>, refresh: bool) {
    let Some(index) = crate::lookup::global() else {
        return;
    };
    let verdict = crate::lookup::Verdict::from_scan(result, purl);
    if refresh {
        index.replace(&verdict);
    } else {
        index.put(&verdict);
    }
}

/// Query the verdict index on a blocking thread: its getters read files.
/// `None` when this process has no index.
pub(super) async fn index_query<T: Send + 'static>(
    query: impl FnOnce(&crate::lookup::Index) -> T + Send + 'static,
) -> Option<T> {
    match tokio::task::spawn_blocking(move || crate::lookup::global().map(query)).await {
        Ok(found) => found,
        Err(e) => {
            tracing::error!(error = %e, "verdict index query panicked");
            None
        }
    }
}

/// Renew a verdict on hopper, so it outlives this process and this request: a
/// caller that hangs up — or a proxy that gives up on a long run — finds the
/// answer on its next lookup. Blocking: the upload queue is bounded.
///
/// Fetched dependencies go first, so each one's row exists before its own
/// verdict lands, then the result — under its own sha256, unless
/// `classify_purl`'s registry fallback redirected or suppressed it (see
/// [`HopperRoute`]).
fn renew(uploader: &Uploader, result: &mut ScanResult, purl: Option<String>) {
    let deps = std::mem::take(&mut result.dependency_results);
    if !deps.is_empty() {
        uploader.submit_dependencies(deps, result.version.clone(), result.analyzed_at.clone());
    }
    let sha = match &result.hopper_route {
        HopperRoute::Suppress => return,
        HopperRoute::Redirect(sha) => sha.clone(),
        HopperRoute::Normal => result.sha256.clone(),
    };
    uploader.submit(sha, purl, result.to_envelope());
}

/// Analyze staged bytes, then offer them to hopper.
///
/// Offered after analysis, not on receipt, so hopper never carries a claimable
/// bytes-with-no-verdict row for longer than it takes the uploader to drain
/// its queue; and durably, because the caller deletes the staged directory,
/// on this same blocking thread, once this returns. A failure to stage the
/// offer is logged: the verdict stands.
fn classify_upload(
    upload: &Upload,
    template: Analysis<'_>,
    uploader: Option<&Uploader>,
) -> anyhow::Result<ScanResult> {
    let result = crate::analysis::classify_file(
        &upload.path,
        None,
        Analysis {
            label: &upload.filename,
            ..template
        },
    )?;
    if let Some(uploader) = uploader
        && let Err(error) =
            uploader.submit_artifacts_durable(crate::engine::collect_upload_artifacts(
                &upload.path,
                &result.sha256,
                result.size_bytes,
                crate::engine::upload_collector(),
                None,
                None,
            ))
    {
        tracing::error!(
            sha256 = %result.sha256,
            file = %upload.filename,
            %error,
            "upload: could not stage artifact for hopper; the verdict stands",
        );
    }
    Ok(result)
}

/// Fetch the PURL's artifact (and its registry record), then classify. Scan
/// looks up provenance itself — beamline does not supply it.
///
/// `corpus` comes with the runtime it is read on: the registry fallback asks
/// hopper what it holds, and this runs on a blocking thread.
fn classify_purl(
    purl: &str,
    template: Analysis<'_>,
    uploader: Option<&Uploader>,
    corpus: Option<(&Corpus, &Handle)>,
) -> anyhow::Result<ScanResult> {
    use fletch::RefLocator;

    let phase = template.phase;
    let mark = |name: &str| {
        if let Some(p) = phase {
            p.set(name);
        }
    };
    // Before cleave sees the initial bytes. Kept distinct from `fetch+graft`,
    // the later dependency-fetch phase inside the report pipeline.
    mark("purl:fetch");
    // The registry lookup and the payload download overlap. Neither needs the
    // other's answer — the record is consulted only once both are back (the
    // removed-version short-circuit, the fallback document, provenance) — yet
    // they ran back to back, and each is one or more registry round-trips: a
    // packument, then a tarball; `.info`, then a module zip. Overlapping them
    // costs one extra request in the one case the record would have skipped
    // the download — a removed version, whose download fails anyway.
    let locator = RefLocator::Purl(purl.to_string());
    let (registry_lookup, fetched) = std::thread::scope(|scope| {
        let payload = purl.to_string();
        let download = std::thread::Builder::new()
            .name("purl-fetch".to_string())
            .spawn_scoped(scope, move || {
                crate::fetch::fetch_one(RefLocator::Purl(payload), false)
            });
        let registry = crate::fetch::registry_with_sources(&locator);
        let fetched = match download {
            Ok(handle) => handle
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("payload fetch thread panicked"))),
            // Could not spawn a thread: fetch inline.
            Err(_) => crate::fetch::fetch_one(RefLocator::Purl(purl.to_string()), false),
        };
        (registry, fetched)
    });
    let (registry, registry_sources) = registry_lookup;
    let registry_provenance = registry.clone().map(|record| {
        crate::provenance::RegistryProvenance::from_record_sources(record, &registry_sources)
    });

    // A removed version, or a download that failed where the registry still
    // has a record: analyze the registry's document instead.
    let fallback = match &fetched {
        Ok(_) => registry
            .as_ref()
            .filter(|reg| reg.version_removed == Some(true))
            .and_then(crate::fetch::registry_document),
        Err(_) => registry.as_ref().and_then(crate::fetch::registry_document),
    };
    if let Some((name, bytes)) = fallback {
        mark("purl:registry-document");
        let hopper_route =
            offer_registry_fallback(corpus, uploader, purl, &name, registry_provenance.as_ref());
        let mut result = crate::analysis::classify_bytes(
            bytes::Bytes::from(bytes),
            Analysis {
                label: &name,
                root_registry: registry_provenance.as_ref(),
                ..template
            },
        )?;
        result.hopper_route = hopper_route;
        return Ok(result);
    }

    mark("purl:payload");
    let (bytes, name, rec) = fetched?;
    let result = crate::analysis::classify_bytes(
        bytes::Bytes::from(bytes),
        Analysis {
            label: &name,
            root_registry: registry_provenance.as_ref(),
            ..template
        },
    )?;

    // Offer the artifact — bytes, registry record, and fetch provenance —
    // before its verdict, exactly as the CLI (`scan purl --hopper`) and the
    // pull worker do. Deliberately after analysis, not on fetch: hopper's
    // upload-tier claim query drains bytes-with-no-verdict rows first and
    // ahead of everything else, so offering them before this process has its
    // own verdict in hand would race the sample onto the worker fleet's claim
    // queue for a redundant analysis. Hopper drops a result for a SHA it
    // never ingested, so the verdict alone lands nowhere either — queuing
    // artifacts then result keeps the row unclaimable for only the width of
    // the queue, not the width of an analysis.
    if let Some(uploader) = uploader {
        uploader.submit_artifacts(crate::engine::collect_upload_artifacts(
            Path::new(&name),
            &result.sha256,
            result.size_bytes,
            crate::engine::upload_collector(),
            registry_provenance.as_ref(),
            Some(&rec),
        ));
    }
    Ok(result)
}

/// Fetch an exact URL verbatim and classify it. The result carries the SHA-256
/// beamline uses to alias the URL to the canonical artifact.
fn classify_url(
    url: &str,
    template: Analysis<'_>,
    uploader: Option<&Uploader>,
) -> anyhow::Result<ScanResult> {
    use fletch::RefLocator;

    if let Some(p) = template.phase {
        p.set("url:payload");
    }
    // Propagated as-is: `anyhow!` on an `anyhow::Error` rebuilds it from its
    // `Display`, which drops the chain and with it the `Unretrievable` a URL
    // fetch failure has to be recognized by.
    let (bytes, name, rec) = crate::fetch::fetch_one(RefLocator::Url(url.to_owned()), false)?;
    let result = crate::analysis::classify_bytes(
        bytes::Bytes::from(bytes),
        Analysis {
            label: &name,
            ..template
        },
    )?;
    // See `classify_purl` on why this waits for the verdict.
    if let Some(uploader) = uploader {
        uploader.submit_artifacts(crate::engine::collect_upload_artifacts(
            Path::new(&name),
            &result.sha256,
            result.size_bytes,
            crate::engine::upload_collector(),
            None,
            Some(&rec),
        ));
    }
    Ok(result)
}

/// Offer a registry-metadata fallback's provenance to hopper without ever
/// posting the fallback's own content as a new sample: that content is the
/// registry's JSON record, not a real artifact, and it hashes differently
/// every time it's built (`with_age()`-derived fields are relative to the
/// call), so treating it as content-addressed mints hopper a fresh,
/// never-deduplicating row on every single fetch — confirmed 2026-08-27
/// against production (`lodash.once@4.1.1` and friends: 8-9 distinct shas for
/// 8-9 fetches of the identical coordinate, in under two hours).
///
/// Looks up whether hopper already holds *real* content for this purl (a
/// prior successful fetch, by this process or another producer). If so,
/// backfills this fresh registry metadata onto that existing sha as
/// provenance-only — no bytes move — and the caller's verdict should redirect
/// onto it too. If hopper has never seen this coordinate under any sha, there
/// is nothing to attach to; the caller's verdict is suppressed rather than
/// minting a placeholder that would just be more of the same churn.
///
/// Runs on a blocking thread, and reads the corpus by blocking on the runtime
/// handle it was given.
fn offer_registry_fallback(
    corpus: Option<(&Corpus, &Handle)>,
    uploader: Option<&Uploader>,
    purl: &str,
    name: &str,
    registry_provenance: Option<&crate::provenance::RegistryProvenance>,
) -> HopperRoute {
    let Some((corpus, runtime)) = corpus else {
        return HopperRoute::Suppress;
    };
    let (reached, _source) = runtime.block_on(corpus.known_with_source(None, Some(purl)));
    let Reached::Record(record) = reached else {
        return HopperRoute::Suppress;
    };
    let Some(real_sha) = record.sha256 else {
        return HopperRoute::Suppress;
    };
    if let Some(uploader) = uploader
        && let Some(artifact) =
            crate::engine::registry_fallback_artifact(name, &real_sha, purl, registry_provenance)
    {
        uploader.submit_artifacts(vec![artifact]);
    }
    HopperRoute::Redirect(real_sha)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serve one canned HTTP response to the next connection, standing in for
    /// hopper's `/v1/lookup`.
    fn one_response(response: String) -> (String, std::thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind mock corpus");
        let addr = listener.local_addr().expect("addr");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept lookup");
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        });
        (format!("http://{addr}"), server)
    }

    async fn fallback_route(base: &str, purl: &'static str) -> HopperRoute {
        let corpus = Corpus::new(Some(base)).expect("corpus configured");
        let runtime = Handle::current();
        tokio::task::spawn_blocking(move || {
            offer_registry_fallback(
                Some((&corpus, &runtime)),
                None,
                purl,
                "fallback.registry.json",
                None,
            )
        })
        .await
        .expect("task")
    }

    /// A registry-metadata fallback must never mint hopper a placeholder row:
    /// with nothing known about the purl (a 404 from `/v1/lookup`), the result
    /// is `Suppress` — nothing gets posted for it. See
    /// [`offer_registry_fallback`]'s doc comment for why (the fallback's own
    /// content hashes differently on every fetch).
    #[tokio::test]
    async fn unknown_purl_suppresses_the_registry_fallback() {
        let (base, server) = one_response(
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into(),
        );
        let route = fallback_route(&base, "pkg:npm/never-seen@0.0.0").await;
        server.join().expect("server thread");
        assert!(
            matches!(route, HopperRoute::Suppress),
            "an unknown coordinate must suppress the post, not mint a row: {route:?}"
        );
    }

    /// A registry-metadata fallback for a purl hopper already holds real
    /// content for redirects onto that sha instead of minting a new one.
    #[tokio::test]
    async fn known_purl_redirects_the_registry_fallback() {
        let body = format!(r#"{{"sha256":"{}"}}"#, "b".repeat(64));
        let (base, server) = one_response(format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len(),
        ));
        let route = fallback_route(&base, "pkg:npm/known-package@1.0.0").await;
        server.join().expect("server thread");
        match route {
            HopperRoute::Redirect(sha) => assert_eq!(sha, "b".repeat(64)),
            other => panic!("expected a redirect onto the known sha, got {other:?}"),
        }
    }

    /// No corpus to ask is nothing to attach to.
    #[test]
    fn no_corpus_suppresses_the_registry_fallback() {
        assert!(matches!(
            offer_registry_fallback(None, None, "pkg:npm/x@1", "x.json", None),
            HopperRoute::Suppress
        ));
    }

    /// The `llm=` field separates a minute-long endpoint query from a replay of
    /// the prompt cache, which are otherwise distinguishable only by timing.
    #[test]
    fn llm_source_names_where_the_verdict_came_from() {
        use crate::interpret::{Failed, Graded, Interpretation, LlmGrade, MlVerdict};

        let graded = |cached| {
            Interpretation::Graded(Graded {
                grade: LlmGrade::Benign,
                outcome: crate::Classification::Benign,
                blended: 0.1,
                interpretation: String::new(),
                model: "m".to_string(),
                analyzer_directed: false,
                before: MlVerdict {
                    class: crate::Classification::Benign,
                    prob: 0.1,
                    lvl: crate::model::Level::Manual,
                },
                corroborated: false,
                cached,
            })
        };
        let failed = Interpretation::Failed(Failed {
            class: crate::Classification::Benign,
            prob: 0.1,
            model: "m".to_string(),
            analyzer_directed: false,
            error: "timeout".to_string(),
        });
        assert_eq!(llm_source(None), None, "no pass ran");
        assert_eq!(llm_source(Some(&graded(false))), Some("queried"));
        assert_eq!(llm_source(Some(&graded(true))), Some("cached"));
        assert_eq!(llm_source(Some(&failed)), Some("failed"));
    }

    /// The collector must read the same whichever path ingested the sample, so
    /// hopper's provenance does not fork by ingest route.
    #[test]
    fn collector_matches_the_cli_form() {
        let got = crate::engine::upload_collector();
        assert!(got.starts_with("scan+"), "collector = {got}");
        assert_eq!(
            got,
            crate::engine::upload_collector(),
            "collector is not stable"
        );
    }

    /// Staging writes the bytes under the caller's name, so cleave types the
    /// file by its extension, and the directory goes when the upload does.
    #[test]
    fn a_staged_upload_keeps_its_name_and_cleans_up() {
        let upload = Upload::stage("left-pad-1.3.0.tgz", b"bytes").expect("stage");
        assert!(upload.path.ends_with("left-pad-1.3.0.tgz"));
        assert_eq!(std::fs::read(&upload.path).expect("read"), b"bytes");
        assert_eq!(upload.size_bytes, 5);
        let dir = upload
            .path
            .parent()
            .expect("staged in a directory")
            .to_path_buf();
        drop(upload);
        assert!(!dir.exists(), "the staged directory outlived its upload");
    }

    /// A fetched root's bytes live in fletch's blob cache, keyed by locator —
    /// not on disk. Building the artifact from the fetch record is what makes
    /// the upload possible at all.
    ///
    /// It also pins a dependency on the caller: the PURL slot hopper projects
    /// into its queryable `purl_base` column is filled only when the locator
    /// carries the `pkg:` prefix. Beamline forwards the canonical form, so this
    /// holds today; if that ever changes, the artifact still uploads but
    /// `/api/sample?purl=` stops finding it, which is exactly the silent gap
    /// this whole path exists to close.
    #[test]
    fn fetched_root_offers_cached_bytes_and_its_purl() {
        let record = |locator: &str| {
            serde_json::from_value::<fletch::fetch::FetchRecord>(serde_json::json!({
                "locator": locator,
                "resolved_url": "https://crates.io/api/v1/crates/libc/0.2.101/download",
                "outcome": "ok",
            }))
            .expect("FetchRecord")
        };
        let build = |rec: &fletch::fetch::FetchRecord| {
            crate::engine::collect_upload_artifacts(
                Path::new("libc-0.2.101.crate"),
                &"a".repeat(64),
                1234,
                "scan+test",
                None,
                Some(rec),
            )
        };

        let arts = build(&record("pkg:cargo/libc@0.2.101"));
        assert_eq!(arts.len(), 1);
        let art = &arts[0];
        assert_eq!(art.size, 1234);
        assert!(
            matches!(art.bytes, crate::upload::ArtifactBytes::Cached { .. }),
            "a fetched root must take its bytes from the blob cache, not a path",
        );
        assert!(
            art.backfill,
            "a PURL-identified artifact is worth backfilling onto an existing sample",
        );
        let sidecar: serde_json::Value =
            serde_json::from_slice(&art.sidecar).expect("sidecar json");
        assert_eq!(
            sidecar["package"]["purl"], "pkg:cargo/libc@0.2.101",
            "the PURL must reach the sidecar slot hopper reads into purl_base",
        );
    }
}
