//! The legacy HTTP routes — `/analyze`, `/analyze-purl`, `/analyze-path`,
//! `/lookup`, `/status` — and the admin routes `/_/reload` and `/_/update`.

use axum::extract::{Extension, Query, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Json, Response};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::access::{RequestId, Subject, with_subject};
use super::analyze::{self, Finished, Job, RequestFollow, Upload, index_query};
use super::error::ApiError;
use super::flight::{FlightKey, Outcome};
use super::{AppState, InFlightRequest, RequestGuard};
use crate::analysis::{Analysis, RequestPhase};
use crate::model::Model;

/// The name an upload is staged and reported under.
///
/// Every character outside `[A-Za-z0-9_.-]` becomes `_`, `..` collapses to
/// `__`, and the result keeps its last 63 bytes so the extension survives —
/// cleave detects file type from it, and the name is also the `path` label in
/// the report and in every log line about this request.
///
/// The filter is deliberately ASCII-only rather than Unicode-aware. Two
/// reasons, both of them the client's choice to make otherwise: a
/// `char::is_alphanumeric` filter keeps multi-byte characters, and the
/// right-truncation below would then slice mid-character and panic the
/// request; and a name that reaches logs and a filesystem path should not
/// carry homoglyphs, combining marks, or bidi-shaped text.
pub(super) fn sanitize_upload_filename(raw: &str) -> String {
    let sanitized: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .replace("..", "__");
    // Every retained character is one ASCII byte, so this index is always a
    // character boundary.
    #[expect(
        clippy::string_slice,
        reason = "every retained character is one ASCII byte"
    )]
    if sanitized.len() > 63 {
        sanitized[sanitized.len() - 63..].to_string()
    } else {
        sanitized
    }
}

/// Canonical `pkg:…` form, or a 400 message. Same prefixing rule as `atomscan purl`.
pub(super) fn normalize_pkg_purl(raw: &str) -> Result<String, &'static str> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("missing purl");
    }
    let prefixed = if raw.starts_with("pkg:") {
        raw.to_string()
    } else {
        format!("pkg:{raw}")
    };
    fletch::purl::normalize(&prefixed).ok_or("not a package URL")
}

pub(super) fn valid_http_url(raw: &str) -> bool {
    reqwest::Url::parse(raw).is_ok_and(|url| matches!(url.scheme(), "http" | "https"))
}

/// Render the shared outcome as this request's response. `elapsed_ms` is the
/// caller's own wall time, so a follower reports how long *it* waited, `shared`
/// marks the response as one that rode another request's analysis, and `key`
/// names the artifact on the access line.
fn flight_response(outcome: &Outcome, elapsed_ms: u64, shared: bool, key: &FlightKey) -> Response {
    let mut resp = match outcome {
        Outcome::Report(result) => {
            let mut resp = Json(result.envelope_ref()).into_response();
            resp.headers_mut().insert("X-Total-Ms", elapsed_ms.into());
            resp
        }
        Outcome::Failed(refusal) => refusal.clone().into_response(),
    };
    if shared {
        resp.extensions_mut().insert(super::access::Shared);
    }
    resp.extensions_mut().insert(Subject::from(key));
    resp
}

/// POST /analyze — accept a multipart file upload, classify, return the full
/// scan envelope.
pub(super) async fn analyze(
    State(state): State<Arc<AppState>>,
    Extension(request_id): Extension<RequestId>,
    multipart: axum::extract::Multipart,
) -> Response {
    let request_id = request_id.get();
    // Freezes the companion worker for the whole handler, upload included.
    let _busy = state.enter_busy();
    let request_start = Instant::now();
    tracing::info!(id = request_id, "--> POST /analyze");

    if let Err(refusal) = state.admit_request(request_id).await {
        return refusal.into_response();
    }
    let (sha, upload) = match receive_upload(&state, request_id, multipart).await {
        Ok(received) => received,
        Err(refusal) => return refusal.into_response(),
    };

    // Share the run with anyone already analyzing these exact bytes.
    let attachment = state.flights.join(FlightKey::Sha(sha.clone()));
    if attachment.leads() {
        tracing::info!(
            id = request_id,
            filename = %upload.filename,
            size_bytes = upload.size_bytes,
            sha256 = %sha,
            upload_ms = crate::duration_ms(request_start.elapsed()),
            "received file, starting analysis",
        );
        let follow = RequestFollow {
            policy: state.config.fetch,
            refresh: false,
        };
        analyze::lead(
            &state,
            request_id,
            attachment.flight(),
            Job::Upload(upload),
            follow,
        );
    } else {
        tracing::info!(
            id = request_id,
            filename = %upload.filename,
            size_bytes = upload.size_bytes,
            sha256 = %sha,
            "received file, joined an analysis already in flight",
        );
        // These bytes are already being analyzed; ours are surplus. Deleting
        // them is file I/O, so it happens off the reactor.
        state.tasks.spawn(async move {
            let _ = tokio::task::spawn_blocking(move || drop(upload)).await;
        });
    }

    let outcome = attachment.flight().wait().await;
    flight_response(
        &outcome,
        crate::duration_ms(request_start.elapsed()),
        !attachment.leads(),
        attachment.flight().key(),
    )
}

/// Stream the first multipart field to a staged file, hashing it as it
/// arrives: the digest is what lets a second request for these bytes join the
/// first one's analysis.
async fn receive_upload(
    state: &AppState,
    request_id: u64,
    mut multipart: axum::extract::Multipart,
) -> Result<(String, Upload), ApiError> {
    let invalid = |message: &'static str| ApiError::bad_request("invalid_upload", message);
    let save_failed = || {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "Failed to save file data",
        )
    };

    let mut field = match multipart.next_field().await {
        Ok(Some(field)) => field,
        Ok(None) => {
            tracing::warn!(id = request_id, "bad request: no file field");
            return Err(invalid("No file field in request"));
        }
        Err(e) => {
            tracing::warn!(id = request_id, error = %e, "bad request: unparseable multipart body");
            return Err(invalid("Invalid multipart data"));
        }
    };

    // The staged name: the temp file's name (so cleave detects the file type
    // from its extension), the `path` label in the report, and the name in
    // every log line about this request.
    let filename = match field.file_name() {
        Some(name) => sanitize_upload_filename(name),
        // Already within the sanitizer's alphabet.
        None => format!("upload-{request_id}"),
    };

    let dir =
        match tokio::task::spawn_blocking(|| tempfile::Builder::new().prefix("scan-").tempdir())
            .await
        {
            Ok(Ok(dir)) => dir,
            Ok(Err(e)) => {
                tracing::warn!(id = request_id, error = %e, "failed to create temp dir");
                return Err(ApiError::internal());
            }
            Err(e) => {
                tracing::warn!(id = request_id, error = %e, "temp dir task join error (panic?)");
                return Err(ApiError::internal());
            }
        };
    let path = dir.path().join(&filename);
    let mut file = tokio::fs::File::create(&path).await.map_err(|e| {
        tracing::warn!(id = request_id, path = %path.display(), error = %e, "failed to open temp file for writing");
        ApiError::internal()
    })?;

    let max_upload = state.config.max_body_size;
    let mut size = 0usize;
    let mut digest = Sha256::new();
    loop {
        match field.chunk().await {
            Ok(Some(chunk)) => {
                size += chunk.len();
                if size > max_upload {
                    tracing::warn!(
                        id = request_id,
                        file_size = size,
                        max_upload,
                        "upload exceeded size limit"
                    );
                    return Err(ApiError::new(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "artifact_too_large",
                        "File too large",
                    ));
                }
                digest.update(&chunk);
                if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await {
                    tracing::warn!(id = request_id, error = %e, "failed to write upload chunk");
                    return Err(save_failed());
                }
            }
            Ok(None) => break,
            Err(e) => {
                tracing::warn!(id = request_id, error = %e, "failed to read multipart chunk");
                return Err(invalid("Error reading upload data"));
            }
        }
    }
    if let Err(e) = tokio::io::AsyncWriteExt::flush(&mut file).await {
        tracing::warn!(id = request_id, error = %e, "failed to flush temp file");
        return Err(save_failed());
    }
    if let Err(e) = file.sync_all().await {
        tracing::warn!(id = request_id, error = %e, "failed to sync temp file");
        return Err(save_failed());
    }
    drop(file);

    if size == 0 {
        tracing::warn!(id = request_id, "bad request: empty file");
        return Err(invalid("Empty file"));
    }
    let upload = Upload {
        _dir: dir,
        path,
        filename,
        size_bytes: size as u64,
    };
    Ok((format!("{:x}", digest.finalize()), upload))
}

/// POST /analyze-purl — fetch a package by PURL and analyze it.
///
/// Scan looks up registry provenance itself (age, custody, downloads) and
/// grafts it into the report, the same path as `atomscan purl`. Beamline
/// calls this when a PURL is not in hopper; it is a full analysis and takes
/// a slot. Dependency fetch and LLM interpretation follow the process-wide
/// `--follow` / `--interpret` flags.
#[derive(serde::Deserialize)]
pub(super) struct AnalyzePurlRequest {
    purl: String,
}

pub(super) async fn analyze_purl(
    State(state): State<Arc<AppState>>,
    Extension(request_id): Extension<RequestId>,
    Json(req): Json<AnalyzePurlRequest>,
) -> Response {
    let request_id = request_id.get();
    let _busy = state.enter_busy();
    let request_start = Instant::now();

    let purl = match normalize_pkg_purl(&req.purl) {
        Ok(p) => p,
        // No flight, so no key to take the subject from: name what the caller
        // sent, bounded, as the lookup route does.
        Err(message) => {
            return with_subject(
                ApiError::bad_request("invalid_purl", message).into_response(),
                Subject::purl(&req.purl, None),
            );
        }
    };
    if let Err(refusal) = state.admit_request(request_id).await {
        return refusal.into_response();
    }

    // Share the run with anyone already analyzing this PURL.
    let attachment = state.flights.join(FlightKey::Purl(purl.clone()));
    if attachment.leads() {
        tracing::info!(id = request_id, purl = %purl, "--> POST /analyze-purl");
        let follow = RequestFollow {
            policy: state.config.fetch,
            refresh: false,
        };
        analyze::lead(
            &state,
            request_id,
            attachment.flight(),
            Job::Purl(purl),
            follow,
        );
    } else {
        tracing::info!(
            id = request_id,
            purl = %purl,
            "--> POST /analyze-purl (joined an analysis already in flight)",
        );
    }

    let outcome = attachment.flight().wait().await;
    flight_response(
        &outcome,
        crate::duration_ms(request_start.elapsed()),
        !attachment.leads(),
        attachment.flight().key(),
    )
}

#[derive(serde::Deserialize)]
pub(super) struct AnalyzePathRequest {
    path: String,
    /// Optional registry provenance for this file, in any shape
    /// [`crate::provenance::registry_provenance`] accepts (a hopper sidecar, a
    /// bare fletch envelope, or a normalized record). This is the server-side
    /// equivalent of the CLI's `--registry-map` entry for the same sha: it lets
    /// a caller that already holds the facts — promoter fetches them from
    /// hopper — hand them over instead of making the scan refetch or go without.
    /// Absent or unparseable means the scan simply runs without registry facts,
    /// exactly as it did before this field existed.
    #[serde(default)]
    registry: Option<Box<serde_json::value::RawValue>>,
}

/// POST /analyze-path — analyze a file by its on-disk path.
///
/// Accepts `{"path": "/full/path/to/file", "registry": {...}}` (registry
/// optional). The path must be under one of the directories specified by
/// `--allowed-dirs`. Returns the same `{"ml": {...}, "raw": {...}}` envelope as
/// `/analyze`.
pub(super) async fn analyze_path(
    State(state): State<Arc<AppState>>,
    Extension(request_id): Extension<RequestId>,
    Json(req): Json<AnalyzePathRequest>,
) -> Response {
    let _busy = state.enter_busy();
    // Attached around the whole handler rather than at each return: this route
    // rejects from several places — not found, not under an allowed dir, under
    // memory pressure — and a rejected path is the one an operator most needs
    // named. The path is as the caller wrote it; where the canonical form
    // differs, the rejection line below carries both.
    let subject = Subject::path(&req.path);
    let response = analyze_path_inner(&state, request_id.get(), req)
        .await
        .unwrap_or_else(IntoResponse::into_response);
    with_subject(response, subject)
}

async fn analyze_path_inner(
    state: &Arc<AppState>,
    request_id: u64,
    req: AnalyzePathRequest,
) -> Result<Response, ApiError> {
    let request_start = Instant::now();
    let not_found = || ApiError::new(StatusCode::NOT_FOUND, "not_found", "File not found");

    // Resolve symlinks and canonicalize BEFORE the allowed-dirs check to
    // prevent symlink-based path traversal (e.g., /allowed/link → /etc/shadow).
    let path = tokio::fs::canonicalize(&req.path)
        .await
        .map_err(|_unresolvable| not_found())?;
    let allowed = &state.config.allowed_dirs;
    if allowed.is_empty() || !allowed.iter().any(|dir| path.starts_with(dir)) {
        tracing::warn!(id = request_id, path = %req.path, canonical = %path.display(), "analyze-path rejected: not under allowed dirs");
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "forbidden",
            "Path not under allowed directories",
        ));
    }
    let file_size = match tokio::fs::metadata(&path).await {
        Ok(meta) if meta.is_file() => meta.len(),
        _ => return Err(not_found()),
    };

    state.admit_request(request_id).await?;

    let key = format!("path:{}", req.path);
    tracing::info!(
        id = request_id,
        path = %req.path,
        size_bytes = file_size,
        "--> POST /analyze-path",
    );
    let (resources, permit) = analyze::admit(state, request_id, &key, Some(file_size))?;

    // Registry provenance the caller supplied for this file, parsed before the
    // analysis thread starts so a malformed document costs nothing downstream.
    // Provenance enriches a scan but is never required, so an unparseable
    // document degrades to a warning and a registry-less scan rather than a 400.
    let root_registry = req.registry.as_ref().and_then(|raw| {
        let provenance = crate::provenance::registry_provenance(raw.get().as_bytes());
        if provenance.is_none() {
            tracing::warn!(
                id = request_id,
                path = %req.path,
                "analyze-path registry provenance carries no record; scanning without it",
            );
        }
        provenance
    });

    let phase = RequestPhase::with_label(format!("req#{request_id} {}", req.path));
    let cancellation = Arc::new(AtomicBool::new(false));
    let entry = InFlightRequest::new(
        &req.path,
        file_size,
        Arc::clone(&cancellation),
        phase.clone(),
    );
    let started = Instant::now();
    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let extract_dir = state.config.extract_dir.clone();
    let slow_rule_ms = state.config.slow_rule_ms;
    // Dependencies ride the hopper renewal below.
    let deps_for_upload = state.uploader.is_some();
    let (analyzed, tracker) = (path.clone(), phase.clone());
    // The guard lives in this handler's future, so a caller who hangs up
    // cancels the run.
    let outcome = RequestGuard::new(state, request_id, entry, permit)
        .run(move || {
            crate::analysis::classify_file(
                &analyzed,
                extract_dir.as_deref(),
                Analysis {
                    cancellation: Some(&cancellation),
                    phase: Some(&tracker),
                    // Caller-supplied, the server-side `--registry-map` equivalent.
                    root_registry: root_registry.as_ref(),
                    deps_for_upload,
                    ..Analysis::new(&filename, &resources, slow_rule_ms)
                },
            )
        })
        .await;
    let finished = Finished {
        request_id,
        key: &key,
        purl: None,
        elapsed_ms: crate::duration_ms(started.elapsed()),
        phases: phase.timeline(),
    };
    let mut result = analyze::record(state, &finished, outcome)?;

    // Off the reactor: the extraction check stats a directory, the index is
    // files, and the envelope can be megabytes.
    let extract_dir = state.config.extract_dir.clone();
    let filed = tokio::task::spawn_blocking(move || {
        note_extracted_path(extract_dir.as_deref(), &mut result);
        analyze::index_verdict(&result, None, false);
        let renewal = (
            result.sha256.clone(),
            result.size_bytes,
            std::mem::take(&mut result.dependency_results),
        );
        let envelope = result.into_envelope();
        let body = serde_json::to_vec(&envelope);
        (body, renewal, envelope)
    })
    .await;
    let Ok((body, (sha256, size, deps), envelope)) = filed else {
        tracing::error!(id = request_id, "filing the verdict panicked");
        return Err(ApiError::internal());
    };
    let body = body.map_err(|e| {
        tracing::error!(id = request_id, error = %e, "could not serialize the envelope");
        ApiError::internal()
    })?;
    // Renew the result on hopper, with the file and its fetched dependencies.
    if let Some(uploader) = &state.uploader {
        let uploader = Arc::clone(uploader);
        state.tasks.spawn(async move {
            let _ = tokio::task::spawn_blocking(move || {
                crate::engine::upload_scan_result(
                    &uploader,
                    crate::engine::Origin::local(&path),
                    sha256,
                    size,
                    deps,
                    envelope,
                );
            })
            .await;
        });
    }
    let mut resp = ([(header::CONTENT_TYPE, "application/json")], body).into_response();
    resp.headers_mut().insert(
        "X-Total-Ms",
        crate::duration_ms(request_start.elapsed()).into(),
    );
    Ok(resp)
}

/// Record where archive members were extracted on disk, so cyclotron can open
/// them.
fn note_extracted_path(extract_dir: Option<&std::path::Path>, result: &mut crate::ScanResult) {
    if let (Some(extract_dir), Some(raw)) = (extract_dir, &mut result.cleave)
        && let Some(first) = raw.files.first().map(|f| f.sha.as_str())
    {
        // SHA hex is ASCII; byte slice is always a valid UTF-8 boundary.
        let short = first.get(..first.len().min(6)).unwrap_or(first);
        let dir = extract_dir.join(short);
        if dir.is_dir() {
            raw.extracted_path = Some(dir.to_string_lossy().into_owned());
        }
    }
}

/// Query string for `GET /lookup` and `GET /status`.
///
/// Both identifiers travel as query parameters. A PURL's own grammar carries
/// `/`, `?` and `#` — `pkg:npm/@scope/name@1.0.0?arch=x64` in a path segment
/// would have everything from the `?` parsed as the *URL's* query and a
/// `#subpath` dropped by the client, silently keying on a different package —
/// and a digest gains nothing from a prettier URL that the other key cannot
/// have too.
#[derive(Debug, Default, serde::Deserialize)]
pub(super) struct LookupQuery {
    sha256: Option<String>,
    purl: Option<String>,
    url: Option<String>,
}

/// GET /lookup?sha256=… | ?purl=… — what we already know about an artifact.
///
/// Never analyzes: no slot, no fetch, and it answers while the model is still
/// loading. A caller that gets `404 unknown sample` asks for a real analysis
/// with `/analyze` or `/analyze-purl`.
pub(super) async fn lookup(
    State(state): State<Arc<AppState>>,
    Query(q): Query<LookupQuery>,
) -> Response {
    let started = Instant::now();
    let response = lookup_inner(&state, &q).await;
    // Timed here rather than inside each arm so every answer counts — a
    // rejection is as much a measure of this endpoint's speed as a hit.
    state
        .jobs
        .lookups
        .record(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
    response
}

fn lookup_error(status: StatusCode, message: &'static str) -> Response {
    ApiError::new(status, "bad_request", message).into_response()
}

async fn lookup_inner(state: &AppState, q: &LookupQuery) -> Response {
    let sha = q.sha256.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let purl = q.purl.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let url = q.url.as_deref().map(str::trim).filter(|s| !s.is_empty());
    // Every arm names its subject, so the request's access line says which
    // artifact was asked about — including the arms that reject, where the key
    // is the only way to tell a caller's bug from a caller's typo.
    match (sha, purl, url) {
        (None, None, None) => lookup_error(
            StatusCode::BAD_REQUEST,
            "provide sha256, purl, url, or both",
        ),
        (_, Some(_), Some(_)) | (Some(_), None, Some(_)) => lookup_error(
            StatusCode::BAD_REQUEST,
            "provide one locator plus an optional sha256",
        ),
        (Some(sha), Some(purl), None) => lookup_by_both(state, sha, purl).await,
        (Some(sha), None, None) => {
            with_subject(lookup_by_sha(state, sha).await, Subject::sha256(sha))
        }
        (None, Some(purl), None) => lookup_by_purl(state, purl).await,
        (None, None, Some(url)) => lookup_by_url(url),
    }
}

fn lookup_by_url(raw: &str) -> Response {
    if !valid_http_url(raw) {
        return with_subject(
            lookup_error(StatusCode::BAD_REQUEST, "invalid url"),
            Subject::url(raw, None),
        );
    }
    // The legacy lookup route never analyzes or fetches. URL resolution is
    // provided by `/v1/analyze?url=...`; this route can only answer a URL once
    // the durable index has a record keyed by its resolved digest.
    let mut response = lookup_error(StatusCode::NOT_FOUND, "unknown sample");
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    with_subject(response, Subject::url(raw, None))
}

async fn lookup_by_sha(state: &AppState, sha256: &str) -> Response {
    let Some(digest) = burton::parse_sha256_hex(sha256) else {
        return lookup_error(StatusCode::BAD_REQUEST, "invalid sha256");
    };
    let sha = sha256.to_ascii_lowercase();
    let decision = crate::bloom_repo::global()
        .as_deref()
        .map_or(crate::bloom_repo::Decision::Unknown, |lk| {
            lk.memo_sha256(&digest)
        });
    let key = sha.clone();
    let verdict = index_query(move |index| index.get_sha(&key))
        .await
        .flatten();
    lookup_response(state, verdict.as_ref(), decision, &sha, None)
}

/// Answer for an artifact the caller can name both ways.
///
/// Both filters are consulted, because they are cheap — four in-memory probes,
/// already memoized — and because a caller who names both is asserting they are
/// one artifact, which makes each filter evidence about it. A key the other
/// missed is a hit neither would have produced alone, and a disagreement
/// between them lands on `Conflicted` instead of on whichever was asked first.
///
/// The digest stays the identity. Its stored verdict wins outright; the PURL's
/// is accepted only when it describes the same bytes, because a release whose
/// digest has moved is answering about a different artifact than the one asked
/// about. That check costs nothing here — the index already returns the digest
/// it resolved to.
async fn lookup_by_both(state: &AppState, sha256: &str, raw: &str) -> Response {
    let Some(digest) = burton::parse_sha256_hex(sha256) else {
        return lookup_error(StatusCode::BAD_REQUEST, "invalid sha256");
    };
    let purl = match normalize_pkg_purl(raw) {
        Ok(purl) => purl,
        Err(message) => {
            return with_subject(
                lookup_error(StatusCode::BAD_REQUEST, message),
                Subject::purl(raw, None),
            );
        }
    };
    let sha = sha256.to_ascii_lowercase();

    let decision = crate::bloom_repo::global()
        .as_deref()
        .map_or(crate::bloom_repo::Decision::Unknown, |lk| {
            lk.decide_any(Some(&purl), Some(&digest))
        });

    let (key_sha, key_purl) = (sha.clone(), purl.clone());
    let verdict = index_query(move |index| {
        pick_verdict(
            index.get_sha(&key_sha),
            || index.get_purl(&key_purl),
            &key_sha,
        )
    })
    .await
    .flatten();

    let response = lookup_response(state, verdict.as_ref(), decision, &sha, Some(&purl));
    with_subject(response, Subject::purl(&purl, Some(&sha)))
}

/// Which stored verdict answers for a caller who named both keys.
///
/// Digest first, because a digest names exact bytes and a verdict filed under
/// it is about those bytes and nothing else. The PURL is consulted only when
/// the digest is unknown — lazily, so an exact hit costs no index lookup at all
/// — and its verdict is accepted only if it resolved to the same digest.
///
/// That last condition is the whole point of holding both keys. A release whose
/// artifact has changed under it still has a perfectly good verdict; it is just
/// a verdict about a different artifact than the caller asked about, and
/// serving it would answer a question nobody posed.
pub(super) fn pick_verdict(
    by_sha: Option<crate::lookup::Verdict>,
    by_purl: impl FnOnce() -> Option<crate::lookup::Verdict>,
    sha: &str,
) -> Option<crate::lookup::Verdict> {
    if by_sha.is_some() {
        return by_sha;
    }
    by_purl().filter(|v| v.sha256.eq_ignore_ascii_case(sha))
}

async fn lookup_by_purl(state: &AppState, raw: &str) -> Response {
    // `pkg:` is optional, as it is on /analyze-purl and `atomscan purl`, and
    // the canonical form is what the filters and the index are keyed by — so
    // `npm/left-pad@1.3.0` and `pkg:npm/left-pad@1.3.0` are one question.
    let purl = match normalize_pkg_purl(raw) {
        Ok(purl) => purl,
        // Unparseable: name what the caller actually sent, not the canonical
        // form there isn't one of.
        Err(message) => {
            return with_subject(
                lookup_error(StatusCode::BAD_REQUEST, message),
                Subject::purl(raw, None),
            );
        }
    };
    let decision = crate::bloom_repo::global()
        .as_deref()
        .map_or(crate::bloom_repo::Decision::Unknown, |lk| {
            lk.memo_purl(&purl)
        });
    let key = purl.clone();
    let verdict = index_query(move |index| index.get_purl(&key))
        .await
        .flatten();
    let sha = verdict
        .as_ref()
        .map_or("", |v| v.sha256.as_str())
        .to_owned();
    let response = lookup_response(state, verdict.as_ref(), decision, &sha, Some(&purl));
    // The canonical PURL, plus the digest it resolved to when it hit — that
    // mapping is knowledge only this handler has.
    with_subject(response, Subject::purl(&purl, Some(&sha)))
}

/// How long a bloom-derived answer may be cached.
///
/// Two hours — deliberately longer than the hourly filter rebuild, so a derived
/// answer can outlive one cycle. That is the safe direction to be stale in: the
/// filters only ever ADD claims, so an answer that lags errs toward flagging an
/// artifact rather than clearing one. The cost is the reverse case — an artifact
/// analyzed and found clean can keep reading as cited until the entry ages out —
/// which is bounded, visible in the `bloom` field, and cheaper than paying a
/// hopper round trip on every repeat ask.
///
/// Still far below the 24h a measured verdict earns: that one is immutable for
/// the ruleset that produced it, and this one is not.
const BLOOM_DERIVED_MAX_AGE: u32 = 7200;

/// Render a lookup answer.
///
/// A stored verdict is a 200; an adverse bloom match with nothing stored is a
/// 200 carrying a *derived* answer (see [`crate::lookup::bloom_derived_view`]);
/// holding neither is a 404. The bloom decision rides on all three. That keeps
/// the kinds of knowledge distinguishable — a filter says who has claimed what,
/// an analysis says what the thing *is* — while still answering in one round
/// trip.
///
/// A derived answer carries no `eng`, which is how a consumer tells it from a
/// measurement, and how `/v1/analyze` knows it must still run: a citation is
/// exactly what that route exists to replace with a measurement, so it must
/// never stand in for one.
fn lookup_response(
    state: &AppState,
    verdict: Option<&crate::lookup::Verdict>,
    decision: crate::bloom_repo::Decision,
    sha256: &str,
    purl: Option<&str>,
) -> Response {
    // A token-protected answer must not be stored by a shared cache: it is
    // knowledge about a specific customer's artifact, not public data.
    let scope = if state.config.auth_digest.is_some() {
        "private"
    } else {
        "public"
    };
    let sha256 = sha256.trim();
    let (mut resp, max_age, source) = match verdict {
        // A verdict is immutable for the ruleset that produced it, and the
        // ruleset is part of the namespace it was read from — a rules or
        // model update lands in a fresh namespace, which reads as a miss
        // rather than as this answer going stale.
        Some(verdict) => (
            Json(verdict.view(decision.as_str(), purl)).into_response(),
            86400,
            "scan:analysis",
        ),
        None => {
            // Nothing measured, but the filters may still have something to
            // say. A bloom match is answerable on its own — see
            // `bloom_derived_view` — so rather than answering "unknown" about
            // a digest several operators call malware, answer with what they
            // say, marked as what it is. Mirrors hopper's fromLedger, and
            // saves the round trip to it.
            let mut synth_hits = Vec::new();
            match crate::lookup::bloom_derived_view(
                decision,
                decision.as_str(),
                sha256,
                purl,
                &mut synth_hits,
            ) {
                // Emphatically NOT the 24h a measured verdict gets. This
                // answer stands on a filter that is rebuilt hourly and on a
                // ledger that moves underneath it, and it must stop being
                // served the moment a real analysis exists.
                Some(view) => (
                    Json(view).into_response(),
                    BLOOM_DERIVED_MAX_AGE,
                    "scan:bloom",
                ),
                None => return lookup_miss(state, decision, sha256, purl),
            }
        }
    };
    let headers = resp.headers_mut();
    if let Ok(value) = HeaderValue::from_str(&format!("{scope}, max-age={max_age}")) {
        headers.insert(header::CACHE_CONTROL, value);
    }
    if let Ok(value) = HeaderValue::from_str(sha256) {
        headers.insert("X-SHA256", value);
    }
    headers.insert("X-Scan-Source", HeaderValue::from_static(source));
    resp
}

/// `404 unknown sample`, saying whether an analysis of it is running.
///
/// Nothing stored does not mean nothing happening: an analysis of this very
/// artifact may be minutes in. Saying so costs nothing — the caller is already
/// asking about this key, and the registry is a map lookup — and it is what
/// lets a caller who reconnects be routed back to the worker already running
/// their analysis instead of starting a second one beside it. `/status`
/// answers the same question on its own, for a caller who has nothing else to
/// ask.
fn lookup_miss(
    state: &AppState,
    decision: crate::bloom_repo::Decision,
    sha256: &str,
    purl: Option<&str>,
) -> Response {
    let running = purl
        .and_then(|p| state.flights.running(&FlightKey::Purl(p.to_string())))
        .or_else(|| {
            state
                .flights
                .running(&FlightKey::Sha(sha256.to_ascii_lowercase()))
        });
    let mut body = serde_json::json!({
        "error": "unknown sample",
        "bloom": decision.as_str(),
    });
    if let Some(run) = running {
        body["analyzing"] = serde_json::json!({
            "elapsed_ms": crate::duration_ms(run.elapsed),
            "attached": run.attached,
        });
    }
    // A miss is not cacheable for any length of time: it becomes a hit the
    // moment anything analyzes this artifact. It is attributed to the
    // bloom/lookup layer, whatever the bloom decision.
    let mut resp = (StatusCode::NOT_FOUND, Json(body)).into_response();
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    resp.headers_mut()
        .insert("X-Scan-Source", HeaderValue::from_static("scan:bloom"));
    resp
}

/// `GET /status` answer.
#[derive(serde::Serialize)]
pub(super) struct StatusBody {
    pub(super) state: &'static str,
    pub(super) purl: Option<String>,
    pub(super) url: Option<String>,
    pub(super) sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) elapsed_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) attached: Option<usize>,
}

/// GET /status?sha256=… | ?purl=… | ?url=… — where an analysis of this artifact stands.
///
/// Exists for the caller whose connection did not survive the analysis. The run
/// keeps going here when a proxy gives up at its own ceiling, but from outside
/// a run in progress and a run that never started are both `404 unknown sample`
/// on /lookup. That ambiguity is the whole problem: it is the difference
/// between waiting a little longer and paying for a twenty-minute analysis
/// twice.
///
/// Running is reported before complete, so a caller is never told to go away
/// while a run it could ride is still live. A run's verdict is indexed before
/// its flight is retired, so there is no window where it reads as neither.
///
/// `lost` is deliberately not a state. It is the caller's own inference: they
/// dispatched, their connection died, and this answers `unknown`. Reporting it
/// here would mean keeping a graveyard of every run that ever ended, to tell a
/// caller something they already know.
pub(super) async fn status(
    State(state): State<Arc<AppState>>,
    Query(q): Query<LookupQuery>,
) -> Response {
    let sha = q.sha256.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let raw = q.purl.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let raw_url = q.url.as_deref().map(str::trim).filter(|s| !s.is_empty());
    if raw.is_some() && raw_url.is_some() {
        return lookup_error(StatusCode::BAD_REQUEST, "provide purl or url, not both");
    }
    if let Some(url) = raw_url
        && !valid_http_url(url)
    {
        return lookup_error(StatusCode::BAD_REQUEST, "invalid url");
    }
    if sha.is_none() && raw.is_none() && raw_url.is_none() {
        return lookup_error(
            StatusCode::BAD_REQUEST,
            "provide sha256, purl, url, or both",
        );
    }
    // The canonical form is what a flight is keyed by, so an uncanonical spelling
    // must not read as a different artifact — the same rule /lookup follows.
    let purl = match raw.map(normalize_pkg_purl) {
        Some(Ok(purl)) => Some(purl),
        Some(Err(message)) => return lookup_error(StatusCode::BAD_REQUEST, message),
        None => None,
    };
    let sha = sha.map(str::to_ascii_lowercase);
    let url = raw_url.map(str::to_owned);

    let running = purl
        .as_deref()
        .and_then(|p| state.flights.running(&FlightKey::Purl(p.to_string())))
        .or_else(|| {
            url.as_deref()
                .and_then(|u| state.flights.running(&FlightKey::Url(u.to_string())))
        })
        .or_else(|| {
            sha.as_deref()
                .and_then(|s| state.flights.running(&FlightKey::Sha(s.to_string())))
        });
    if let Some(run) = running {
        return Json(StatusBody {
            state: "running",
            purl,
            url,
            sha256: sha,
            elapsed_ms: Some(crate::duration_ms(run.elapsed)),
            attached: Some(run.attached),
        })
        .into_response();
    }

    let (key_purl, key_sha) = (purl.clone(), sha.clone());
    let complete = index_query(move |index| {
        key_purl
            .as_deref()
            .is_some_and(|p| index.get_purl(p).is_some())
            || key_sha
                .as_deref()
                .is_some_and(|s| index.get_sha(s).is_some())
    })
    .await
    .unwrap_or(false);
    Json(StatusBody {
        state: if complete { "complete" } else { "unknown" },
        purl,
        url,
        sha256: sha,
        elapsed_ms: None,
        attached: None,
    })
    .into_response()
}

/// How long `/_/reload` waits before answering; the reload itself runs on.
const RELOAD_TIMEOUT: Duration = Duration::from_secs(120);

/// How long `/_/update` waits: a backstop around the sequential models and
/// traits pulls, then the reload.
const UPDATE_TIMEOUT: Duration = Duration::from_secs(21 * 60 + 120);

/// What a reload installed.
struct Reloaded {
    elapsed_ms: u128,
    /// Why the cleave traits did not reload; the previous traits stay.
    traits_reload_error: Option<String>,
}

impl AppState {
    /// Reload the cleave traits and the model bundle from disk and serve them.
    /// Blocking. On failure the previous model stays in service.
    fn reload_bundle(&self) -> anyhow::Result<Reloaded> {
        let start = Instant::now();
        // Traits first, so the new model runs against fresh rules.
        let traits_reload_error = match cleave::reload_capability_mapper() {
            Err(e) => {
                tracing::warn!("cleave trait reload failed (previous traits retained): {e:#}");
                Some(e)
            }
            Ok(_) => {
                tracing::info!("cleave traits reloaded");
                None
            }
        };
        cleave::clear_all_thread_caches();

        let config = &self.config;
        let model = Model::load(&config.model_dir, config.thresholds, config.level)?;
        let shap = super::load_shap(&config.model_dir)?;
        let elapsed_ms = start.elapsed().as_millis();
        let spec_version = model.spec().version();
        let features = model.spec().total_features();
        let shap_loaded = shap.is_some();
        if self.install(self.bundle(model, shap)) {
            tracing::info!(
                elapsed_ms,
                spec_version,
                features,
                shap_loaded,
                "model reloaded"
            );
        } else {
            tracing::info!(
                elapsed_ms,
                spec_version,
                features,
                shap_loaded,
                "model loaded via reload — server now ready"
            );
        }
        Ok(Reloaded {
            elapsed_ms,
            traits_reload_error,
        })
    }
}

fn reload_in_progress() -> Response {
    ApiError::new(
        StatusCode::CONFLICT,
        "reload_in_progress",
        "Reload already in progress",
    )
    .into_response()
}

/// POST /_/reload — reload the model bundle from disk and swap it in.
///
/// One reload at a time; concurrent calls receive 409. The lock travels with
/// the blocking work, so a reload that outlives the response still holds it
/// until it finishes — and still installs what it loaded.
pub(super) async fn reload(State(state): State<Arc<AppState>>) -> Response {
    tracing::info!("POST /_/reload");
    // Each load allocates significant memory; never two at once.
    let Ok(held) = Arc::clone(&state.reload_lock).try_lock_owned() else {
        tracing::warn!("reload rejected: already in progress");
        return reload_in_progress();
    };
    let reloading = Arc::clone(&state);
    let job = tokio::task::spawn_blocking(move || {
        let _held = held;
        reloading.reload_bundle()
    });
    match tokio::time::timeout(RELOAD_TIMEOUT, job).await {
        Ok(Ok(Ok(reloaded))) => {
            let mut body = serde_json::json!({
                "status": "ok",
                "elapsed_ms": reloaded.elapsed_ms,
            });
            if let Some(err) = &reloaded.traits_reload_error {
                body["traits_reload_error"] = serde_json::json!(err);
            }
            Json(body).into_response()
        }
        Ok(Ok(Err(e))) => {
            // Logged here; the caller is not told filesystem paths or model internals.
            tracing::warn!("reload failed (previous model retained): {e:#}");
            ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "reload_failed",
                "Failed to load model",
            )
            .into_response()
        }
        Ok(Err(e)) => {
            tracing::warn!("reload task join error: {e}");
            ApiError::internal().into_response()
        }
        Err(_elapsed) => {
            tracing::warn!(
                "reload still running after {}s; it keeps the lock and installs when done",
                RELOAD_TIMEOUT.as_secs()
            );
            ApiError::new(
                StatusCode::GATEWAY_TIMEOUT,
                "reload_timeout",
                "Reload timed out",
            )
            .into_response()
        }
    }
}

/// Pull the latest models and traits. Both pulls are non-fatal: each failure
/// is reported, and the reload that follows runs against whatever is on disk.
/// `model_update` validates a bundle (`Model::load`) before swapping it in, so
/// a broken bundle never lands and there is nothing to roll back. Blocking.
fn pull_updates() -> (Option<String>, Option<String>) {
    let dir = crate::models_repo::install_target();
    let models_err = crate::model_update::update(&dir, false, false)
        .err()
        .map(|e| {
            tracing::warn!("models update failed: {e:#}");
            e.to_string()
        });
    let traits_err = crate::traits_repo::update(false, false).err().map(|e| {
        tracing::warn!("traits update failed: {e:#}");
        e.to_string()
    });
    (models_err, traits_err)
}

/// POST /_/update — pull the latest models and traits, then reload.
///
/// Shares the reload lock with `/_/reload`, so concurrent calls receive 409,
/// and holds it across the pulls and the reload for as long as they run.
pub(super) async fn update(State(state): State<Arc<AppState>>) -> Response {
    tracing::info!("POST /_/update");
    let Ok(held) = Arc::clone(&state.reload_lock).try_lock_owned() else {
        tracing::warn!("update rejected: reload already in progress");
        return reload_in_progress();
    };
    let updating = Arc::clone(&state);
    let job = tokio::task::spawn_blocking(move || {
        let _held = held;
        let pulled = pull_updates();
        (pulled, updating.reload_bundle())
    });
    let ((models_err, traits_err), reloaded) = match tokio::time::timeout(UPDATE_TIMEOUT, job).await
    {
        Ok(Ok(done)) => done,
        Ok(Err(e)) => {
            tracing::warn!("update task join error: {e}");
            return ApiError::internal().into_response();
        }
        Err(_elapsed) => {
            tracing::warn!(
                "update still running after {}s; it keeps the lock and installs when done",
                UPDATE_TIMEOUT.as_secs()
            );
            return ApiError::new(
                StatusCode::GATEWAY_TIMEOUT,
                "update_timeout",
                "Update timed out",
            )
            .into_response();
        }
    };
    match reloaded {
        Ok(reloaded) => {
            let mut body = serde_json::json!({
                "status": "ok",
                "elapsed_ms": reloaded.elapsed_ms,
                "models_updated": models_err.is_none(),
                "traits_updated": traits_err.is_none(),
                "models_error": models_err,
                "traits_error": traits_err,
                "version": env!("CARGO_PKG_VERSION"),
                "model_commit": crate::models_repo::version(),
                "traits_commit": cleave::traits_repo::version(),
            });
            if let Some(err) = &reloaded.traits_reload_error {
                body["traits_reload_error"] = serde_json::json!(err);
            }
            Json(body).into_response()
        }
        Err(e) => {
            // The in-memory model was not swapped, so requests continue against
            // the previous model; the on-disk bundle was validated before
            // install, so there is nothing to roll back.
            tracing::error!("model reload failed after update: {e:#}");
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({
                    "status": "reload_failed",
                    "error": "Failed to load model",
                    "models_updated": models_err.is_none(),
                    "traits_updated": traits_err.is_none(),
                    "models_error": models_err,
                    "traits_error": traits_err,
                })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A follower renders the leader's failure verbatim: same status, same body,
    /// no second analysis and no second error.
    #[tokio::test]
    async fn a_replayed_failure_keeps_its_status_and_body() {
        let outcome = Outcome::Failed(ApiError::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "unsupported_artifact",
            "Unsupported file type",
        ));
        let key = FlightKey::Sha("f".repeat(64));
        let response = flight_response(&outcome, 42, false, &key);
        assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        // Even a replayed failure names its artifact on the access line.
        assert!(
            response
                .extensions()
                .get::<super::super::access::Subject>()
                .is_some(),
            "flight responses carry their subject",
        );

        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let parsed: serde_json::Value = serde_json::from_slice(&body).expect("parse body");
        assert_eq!(
            parsed,
            serde_json::json!({ "error": "Unsupported file type" })
        );
    }

    /// A hostile upload filename must not be able to panic the request. A
    /// Unicode-aware filter kept multi-byte characters, and the tail
    /// truncation then sliced one in half.
    #[test]
    fn sanitize_upload_filename_survives_multibyte_names() {
        let raw = "\u{3041}".repeat(80) + ".zip";
        let name = sanitize_upload_filename(&raw);
        assert_eq!(name.len(), 63);
        assert!(name.is_ascii(), "{name}");
        assert!(name.ends_with(".zip"), "the extension must survive: {name}");
    }

    #[test]
    fn sanitize_upload_filename_defuses_paths_and_control_characters() {
        assert_eq!(
            sanitize_upload_filename("../../etc/shadow"),
            "______etc_shadow"
        );
        assert_eq!(
            sanitize_upload_filename("a\nb\r\u{202e}gpj.exe"),
            "a_b__gpj.exe"
        );
        // A name already inside the alphabet is left exactly as it is.
        assert_eq!(
            sanitize_upload_filename("left-pad-1.3.0.tgz"),
            "left-pad-1.3.0.tgz"
        );
    }

    /// A request without `registry` still parses — the field is optional, so
    /// callers predating it (and files with no hopper record) are unaffected.
    #[test]
    fn analyze_path_registry_is_optional() {
        let req: AnalyzePathRequest =
            serde_json::from_str(r#"{"path":"/tmp/x/a.tgz"}"#).expect("parses without registry");
        assert_eq!(req.path, "/tmp/x/a.tgz");
        assert!(req.registry.is_none());
    }

    /// A supplied record round-trips through the same provenance parser the
    /// CLI's `--registry-map` entries use, so both scan paths accept the exact
    /// document hopper hands promoter.
    #[test]
    fn analyze_path_registry_parses_as_provenance() {
        let body = r#"{"path":"/tmp/x/a.tgz","registry":{"ecosystem":"npm","name":"left-pad","version":"1.3.0"}}"#;
        let req: AnalyzePathRequest = serde_json::from_str(body).expect("parses");
        let raw = req.registry.expect("registry present");
        let provenance = crate::provenance::registry_provenance(raw.get().as_bytes())
            .expect("a bare normalized record is one of the accepted shapes");
        assert_eq!(provenance.record.name, "left-pad");
    }

    /// Provenance enriches a scan but is never required, so a document that
    /// carries no recoverable record degrades to `None` (scan without registry
    /// facts) rather than failing the request.
    #[test]
    fn analyze_path_unparseable_registry_degrades_to_none() {
        let body = r#"{"path":"/tmp/x/a.tgz","registry":[1,2,3]}"#;
        let req: AnalyzePathRequest = serde_json::from_str(body).expect("parses");
        let raw = req.registry.expect("registry present");
        assert!(crate::provenance::registry_provenance(raw.get().as_bytes()).is_none());
    }

    /// The lookup routes read their decision straight off the filters, so the
    /// fixture coverage lives with the filters rather than with a handler.
    #[test]
    fn filters_answer_skip_known_bad_and_unknown() {
        use crate::bloom_repo::{Decision, KEY_SCHEME, Lookup, purl_key};
        use burton::{KeySets, Record, Tier};

        let tmp = tempfile::tempdir().expect("tempdir");
        let mut good_sha = [0u8; 32];
        good_sha[0] = 1;
        let mut bad_sha = [0u8; 32];
        bad_sha[0] = 2;

        let mut sets = KeySets::new();
        sets.insert(
            Tier::Good,
            Record {
                purl: purl_key("pkg:npm/good@1"),
                sha256: Some(good_sha),
            },
        );
        sets.insert(
            Tier::Bad,
            Record {
                purl: purl_key("pkg:npm/evil@1"),
                sha256: Some(bad_sha),
            },
        );
        burton::build::write_bundle(
            tmp.path(),
            &sets.into_filters(1e-9),
            "2026-08-31",
            KEY_SCHEME,
        )
        .expect("write bundle");

        let lk = Lookup::load_from(tmp.path());
        assert_eq!(lk.memo_purl("pkg:npm/good@1"), Decision::Skip);
        assert_eq!(lk.memo_purl("pkg:npm/evil@1"), Decision::KnownBad);
        assert_eq!(lk.memo_sha256(&good_sha), Decision::Skip);
        let mut unseen = [0u8; 32];
        unseen[0] = 0xab;
        assert_eq!(lk.memo_sha256(&unseen), Decision::Unknown);
    }

    /// Without filters installed every key is `unknown` — fail closed, never
    /// an error, so a lookup still answers.
    #[test]
    fn filters_absent_reads_unknown() {
        use crate::bloom_repo::{Decision, Lookup};
        let lk = Lookup::default();
        assert_eq!(lk.memo_purl("pkg:npm/left-pad@1.3.0"), Decision::Unknown);
        assert_eq!(lk.memo_sha256(&[7u8; 32]), Decision::Unknown);
    }

    #[test]
    fn normalize_pkg_purl_accepts_bare_and_full() {
        assert_eq!(
            normalize_pkg_purl("pkg:npm/left-pad@1.3.0").unwrap(),
            "pkg:npm/left-pad@1.3.0"
        );
        assert_eq!(
            normalize_pkg_purl("npm/left-pad@1.3.0").unwrap(),
            "pkg:npm/left-pad@1.3.0"
        );
        assert!(normalize_pkg_purl("").is_err());
        assert!(normalize_pkg_purl("not a purl").is_err());
    }
}

#[cfg(test)]
mod pick_verdict_tests {
    use super::pick_verdict;
    use crate::lookup::Verdict;

    fn verdict(sha: &str, eng: &str) -> Verdict {
        Verdict {
            sha256: sha.to_owned(),
            lvl: crate::model::Level::Clean,
            eng: eng.to_owned(),
            at: "2026-01-01T00:00:00Z".to_owned(),
            purl: None,
            why: None,
            hits: Vec::new(),
        }
    }

    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OTHER: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[test]
    fn a_digest_hit_wins_and_costs_no_purl_lookup() {
        let mut asked = false;
        let got = pick_verdict(
            Some(verdict(SHA, "by-sha")),
            || {
                asked = true;
                Some(verdict(SHA, "by-purl"))
            },
            SHA,
        );
        assert_eq!(got.expect("verdict").eng, "by-sha");
        assert!(!asked, "an exact hit must not cost a second index lookup");
    }

    // The second chance: the index knows the release even though these exact
    // bytes are new to it, and the digests agree.
    #[test]
    fn the_purl_answers_when_the_digest_is_unknown() {
        let got = pick_verdict(None, || Some(verdict(SHA, "by-purl")), SHA);
        assert_eq!(got.expect("verdict").eng, "by-purl");
    }

    // The guard the pair exists for: the release resolved to other bytes.
    #[test]
    fn a_purl_verdict_for_other_bytes_is_refused() {
        let got = pick_verdict(None, || Some(verdict(OTHER, "different-artifact")), SHA);
        assert!(
            got.is_none(),
            "served a verdict about bytes nobody asked about"
        );
    }

    #[test]
    fn digest_comparison_is_case_insensitive() {
        let got = pick_verdict(None, || Some(verdict(&SHA.to_uppercase(), "by-purl")), SHA);
        assert!(got.is_some(), "hex case must not decide identity");
    }

    #[test]
    fn neither_key_known_is_no_verdict() {
        assert!(pick_verdict(None, || None, SHA).is_none());
    }
}
