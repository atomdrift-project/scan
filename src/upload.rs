//! Best-effort upload of scan results to a hopper instance.
//!
//! Mirrors the pull-based worker's `/api/result` contract — the same
//! [`ResultPayload`] wire shape, the same zstd-compressed envelope — but driven
//! by a local `scan fs` run instead of a poll loop. `scan fs --hopper=<url>`
//! uses it to *renew* a sample hopper has already ingested with this build's
//! traits and model: hopper's `/api/result` is a lease-free `UPDATE ... WHERE
//! sha256 = ?`, so posting a result for an already-scanned SHA replaces its
//! stored cleave/litmus envelope (and an unknown SHA is a harmless no-op).
//!
//! Uploads run on a dedicated thread so blocking network I/O never stalls the
//! analysis pool, and every failure degrades to a logged warning — a scan never
//! fails because an upload did.

use std::thread::JoinHandle;
use std::time::Duration;

use serde::Serialize;

use crate::engine::ScanResultEnvelope;

/// Hopper bounds the decompressed result body at 512 MiB (`maxResultBodyBytes`
/// in hopper's api.go); a larger document is truncated mid-stream and rejected
/// as invalid JSON, so an over-limit report is sent ML-verdict-only.
const HOPPER_MAX_RESULT_BODY_BYTES: usize = 512 << 20;

/// zstd's default level. Cleave reports are large, highly repetitive JSON that
/// zstd shrinks 3-5x on the wire; the compression cost is dwarfed by the
/// analysis that produced the payload.
const ZSTD_RESULT_LEVEL: i32 = 3;

/// Bound on results buffered ahead of the uploader thread. A small queue applies
/// backpressure: a slow hopper throttles the scan rather than letting envelopes
/// (each up to hundreds of KB) accumulate unbounded in memory.
const UPLOAD_QUEUE_DEPTH: usize = 16;

/// Attempts per result before giving up. Short, unlike the worker's 20-minute
/// budget: a renew is idempotent and the operator can simply re-run.
const MAX_ATTEMPTS: u32 = 3;

/// Request timeout per POST. Matches the worker so a wedged hopper can't pin an
/// uploader thread indefinitely.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Hopper's `validWorkerName` cap (`maxWorkerNameLen` in api.go).
const MAX_WORKER_NAME_LEN: usize = 64;

/// The JSON body POSTed to hopper's `/api/result`. The `{ml, llm?, raw}`
/// envelope is flattened onto the payload so the wire form is
/// `{sha256, worker, duration_ms, ml, llm, raw}` — byte-for-byte the shape the
/// pull-based worker sends, so hopper handles both identically.
#[derive(Serialize)]
pub(crate) struct ResultPayload {
    pub sha256: String,
    pub worker: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub duration_ms: i64,
    #[serde(flatten)]
    pub envelope: Option<ScanResultEnvelope>,
}

/// Serialize and zstd-compress a result payload for upload. Returns the body
/// bytes and the `Content-Encoding` to advertise (`Some("zstd")` when
/// compression succeeded, `None` when it degraded to raw JSON). Returns `None`
/// only when serialization is unrecoverable — the result is then dropped.
///
/// If the serialized envelope exceeds hopper's body limit, the raw cleave report
/// is dropped and the ML verdict is sent alone: hopper still records the verdict
/// and only skips archive explosion, which beats losing the whole result.
pub(crate) fn encode_result_body(
    mut payload: ResultPayload,
    sha256: &str,
) -> Option<(Vec<u8>, Option<&'static str>)> {
    let json = serialize(&payload, sha256)?;
    let json = if json.len() > HOPPER_MAX_RESULT_BODY_BYTES {
        tracing::warn!(
            sha256 = %sha256,
            json_bytes = json.len(),
            limit_bytes = HOPPER_MAX_RESULT_BODY_BYTES,
            "upload: result JSON exceeds hopper's body limit; dropping raw report, posting ML verdict only",
        );
        // Empty the cleave report but keep the ml/llm verdict. `{}` (not null)
        // mirrors the envelope litmus emits when there is no cleave report, so
        // the dropped-raw form stays a structurally valid envelope.
        if let Some(envelope) = payload.envelope.as_mut() {
            envelope.raw = serde_json::json!({});
        }
        serialize(&payload, sha256)?
    } else {
        json
    };
    match zstd::encode_all(json.as_slice(), ZSTD_RESULT_LEVEL) {
        Ok(compressed) => Some((compressed, Some("zstd"))),
        Err(e) => {
            tracing::warn!(sha256 = %sha256, error = %e, "upload: zstd compress failed; sending uncompressed");
            Some((json, None))
        }
    }
}

fn serialize(payload: &ResultPayload, sha256: &str) -> Option<Vec<u8>> {
    match serde_json::to_vec(payload) {
        Ok(json) => Some(json),
        Err(e) => {
            tracing::error!(sha256 = %sha256, error = %e, "upload: serialize failed");
            None
        }
    }
}

/// Worker identity tagged on uploaded results. Hopper's `validWorkerName`
/// requires a non-empty, space-free, printable-ASCII name no longer than 64
/// bytes; we derive it from the hostname (sanitized and truncated), falling back
/// to a fixed marker so the name is always valid.
#[must_use]
pub fn default_worker_name() -> String {
    let host = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_default();
    let sanitized: String = host
        .chars()
        .filter(char::is_ascii_graphic)
        .take(MAX_WORKER_NAME_LEN)
        .collect();
    if sanitized.is_empty() {
        "ascan-fs".to_string()
    } else {
        sanitized
    }
}

/// A result handed to the background uploader: the file's digest plus the
/// envelope to renew on hopper.
#[derive(Debug)]
struct Job {
    sha256: String,
    envelope: ScanResultEnvelope,
}

/// Background uploader that POSTs scan results to hopper without blocking the
/// analysis threads. Created per `scan fs --hopper` run; results are handed off
/// via [`Uploader::submit`] and flushed when the uploader is dropped.
#[derive(Debug)]
pub struct Uploader {
    /// `None` once flushed, or when the uploader thread failed to spawn (uploads
    /// then degrade to silent no-ops rather than failing the scan).
    tx: Option<std::sync::mpsc::SyncSender<Job>>,
    worker: Option<JoinHandle<()>>,
}

impl Uploader {
    /// Start a background uploader targeting `hopper_url`, tagging every result
    /// with `worker`. Spawn failure is non-fatal: the returned uploader silently
    /// drops submissions so the scan still completes.
    #[must_use]
    pub fn new(hopper_url: &str, worker: String) -> Self {
        let result_url = format!("{}/api/result", hopper_url.trim_end_matches('/'));
        let client = reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        let (tx, rx) = std::sync::mpsc::sync_channel::<Job>(UPLOAD_QUEUE_DEPTH);
        let handle = std::thread::Builder::new()
            .name("ascan-upload".into())
            .spawn(move || {
                for job in rx {
                    post_one(&client, &result_url, &worker, job);
                }
            });
        match handle {
            Ok(worker) => {
                tracing::info!(hopper = %hopper_url, "upload: renewing results on hopper");
                Self {
                    tx: Some(tx),
                    worker: Some(worker),
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "upload: failed to spawn uploader thread; uploads disabled");
                Self {
                    tx: None,
                    worker: None,
                }
            }
        }
    }

    /// Queue a result for upload. Blocks briefly when the upload queue is full
    /// (backpressure); a closed channel drops the result silently.
    pub fn submit(&self, sha256: String, envelope: ScanResultEnvelope) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Job { sha256, envelope });
        }
    }
}

impl Drop for Uploader {
    /// Stop accepting new results and wait for in-flight uploads to finish, so a
    /// scan's results are fully renewed before the process exits.
    fn drop(&mut self) {
        // Dropping the sender ends the thread's `for job in rx` loop.
        self.tx = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Flatten an error and its `source()` chain into one message. reqwest's
/// top-level `Display` is just "error sending request for url (...)"; the real
/// cause (connection refused, DNS failure, timeout) lives one or more links down
/// the chain, so log the whole chain to make a failed upload diagnosable.
pub(crate) fn error_chain(err: &dyn std::error::Error) -> String {
    use std::fmt::Write;
    let mut out = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        let _ = write!(out, ": {cause}");
        source = cause.source();
    }
    out
}

/// POST one result to hopper, retrying transient failures a few times. A 4xx
/// (other than 408/429) can never succeed on resend, so it stops immediately.
fn post_one(client: &reqwest::blocking::Client, result_url: &str, worker: &str, job: Job) {
    let sha256 = job.sha256;
    let payload = ResultPayload {
        sha256: sha256.clone(),
        worker: worker.to_string(),
        error: None,
        // fs renews don't track per-file analysis time; hopper treats this as
        // cosmetic. 0 keeps the wire shape identical to the worker's.
        duration_ms: 0,
        envelope: Some(job.envelope),
    };
    let Some((body, encoding)) = encode_result_body(payload, &sha256) else {
        return;
    };

    for attempt in 0..MAX_ATTEMPTS {
        if attempt > 0 {
            // 1s, 2s — brief, since a renew is idempotent and re-runnable.
            std::thread::sleep(Duration::from_secs(1 << (attempt - 1)));
        }
        let mut request = client
            .post(result_url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.clone());
        if let Some(enc) = encoding {
            request = request.header(reqwest::header::CONTENT_ENCODING, enc);
        }
        match request.send() {
            Ok(resp) if resp.status().is_success() => {
                tracing::debug!(sha256 = %sha256, "upload: result renewed on hopper");
                return;
            }
            Ok(resp) => {
                let status = resp.status();
                if status.is_client_error()
                    && status != reqwest::StatusCode::REQUEST_TIMEOUT
                    && status != reqwest::StatusCode::TOO_MANY_REQUESTS
                {
                    let body = resp.text().unwrap_or_default();
                    tracing::warn!(sha256 = %sha256, %status, body = %body, "upload: rejected by hopper; not retrying");
                    return;
                }
                tracing::warn!(sha256 = %sha256, %status, attempt, "upload: non-success response");
            }
            Err(e) => {
                tracing::warn!(sha256 = %sha256, error = %error_chain(&e), attempt, "upload: send failed");
            }
        }
    }
    tracing::warn!(sha256 = %sha256, attempts = MAX_ATTEMPTS, "upload: giving up after retries");
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The wire body round-trips through zstd back to the exact JSON serde
    /// produced: this is the same shape hopper's `/api/result` decodes, so an fs
    /// upload is byte-identical to a worker upload of the same payload.
    #[test]
    fn encode_result_body_round_trips_through_zstd() {
        let payload = ResultPayload {
            sha256: "a".repeat(64),
            worker: "ascan-fs".to_string(),
            error: None,
            duration_ms: 0,
            envelope: None,
        };
        let expected = serde_json::to_vec(&payload).unwrap();
        let (body, encoding) = encode_result_body(payload, "test").expect("encodes");
        assert_eq!(encoding, Some("zstd"));
        let decoded = zstd::decode_all(body.as_slice()).expect("valid zstd");
        assert_eq!(decoded, expected);

        // The flattened payload carries the transport fields and omits `error`
        // (skip_serializing_if), matching the worker's wire form.
        let value: serde_json::Value = serde_json::from_slice(&decoded).unwrap();
        assert_eq!(value["worker"], "ascan-fs");
        assert_eq!(value["duration_ms"], 0);
        assert!(value.get("error").is_none());
    }

    #[test]
    fn default_worker_name_is_valid_for_hopper() {
        let name = default_worker_name();
        assert!(!name.is_empty());
        assert!(name.len() <= MAX_WORKER_NAME_LEN);
        // Mirrors hopper's `validWorkerName`: printable ASCII, no spaces.
        assert!(name.chars().all(|c| c.is_ascii_graphic()));
    }
}
