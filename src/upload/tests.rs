use super::*;

mod backpressure;
use std::io::{Read, Write};
use std::net::TcpListener;

/// A hopper with every route at `bases`, no token, and a plain client.
fn hopper_at(bases: &str) -> Hopper {
    Hopper::new(
        reqwest::blocking::Client::new(),
        bases,
        "test".to_string(),
        Arc::default(),
    )
}

/// One request off `listener`, answered with `response`; returns the raw
/// request. Reads the body by its Content-Length, which a sized upload
/// must send.
fn serve_one(listener: TcpListener, response: &'static [u8]) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let request = read_request(&mut stream);
        stream.write_all(response).expect("respond");
        request
    })
}

fn read_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("read timeout");
    let mut request = Vec::new();
    let mut buf = [0u8; 8192];
    let body_start = loop {
        let n = stream.read(&mut buf).expect("read");
        assert!(n > 0, "connection closed before the headers ended");
        request.extend_from_slice(&buf[..n]);
        if let Some(at) = request.windows(4).position(|w| w == b"\r\n\r\n") {
            break at + 4;
        }
    };
    let head = String::from_utf8_lossy(&request[..body_start]).to_ascii_lowercase();
    let length: usize = head
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .map_or(0, |v| v.trim().parse().expect("content-length"));
    while request.len() < body_start + length {
        let n = stream.read(&mut buf).expect("read body");
        assert!(n > 0, "connection closed mid-body");
        request.extend_from_slice(&buf[..n]);
    }
    request
}

const NO_CONTENT: &[u8] =
    b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

#[test]
fn renewal_workers_preserve_order_and_keep_other_lanes_moving() {
    let lane = |sha: &str| {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        sha.hash(&mut hash);
        usize::try_from(hash.finish() % RENEW_WORKERS as u64).unwrap_or_default()
    };
    let slow = "a".repeat(64);
    let fast = (0..100)
        .map(|n| format!("{n:064x}"))
        .find(|sha| lane(sha) != lane(&slow))
        .expect("a different FIFO lane");
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("address"));
    let completed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = Arc::clone(&completed);
    let fast_arrived = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let overlapped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let server_overlap = Arc::clone(&overlapped);
    let slow_sha = slow.clone();
    let server = std::thread::spawn(move || {
        let mut handlers = Vec::new();
        for _ in 0..5 {
            let (mut stream, _) = listener.accept().expect("accept renewal");
            let observed = Arc::clone(&observed);
            let fast_arrived = Arc::clone(&fast_arrived);
            let overlapped = Arc::clone(&server_overlap);
            let slow_sha = slow_sha.clone();
            handlers.push(std::thread::spawn(move || {
                let request = read_request(&mut stream);
                let offset = request.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                let value: serde_json::Value = serde_json::from_slice(&request[offset..]).unwrap();
                let sha = value["sha256"].as_str().unwrap().to_owned();
                let sequence = value["sequence"].as_u64().unwrap();
                let (ready, wake) = &*fast_arrived;
                if sha == slow_sha && sequence == 0 {
                    // Hold the first lane until the other lane reaches HTTP.
                    // A blocking lane queue stalls on this digest's third
                    // update and never submits the independent verdict.
                    let (arrived, _) = wake
                        .wait_timeout_while(ready.lock().unwrap(), Duration::from_secs(5), |v| !*v)
                        .unwrap();
                    overlapped.store(*arrived, Ordering::Relaxed);
                } else if sha != slow_sha {
                    *ready.lock().unwrap() = true;
                    wake.notify_all();
                }
                observed.lock().unwrap().push((sha, sequence));
                stream.write_all(NO_CONTENT).expect("acknowledge");
            }));
        }
        for handler in handlers {
            handler.join().expect("HTTP handler");
        }
    });
    let (tx, jobs) = queue::channel(1);
    for (sha, count) in [(&slow, 3), (&fast, 2)] {
        for sequence in 0..count {
            tx.send(Job::Result {
                sha256: sha.clone(),
                purl: None,
                body: EncodedResult {
                    bytes: serde_json::to_vec(
                        &serde_json::json!({"sha256": sha, "sequence": sequence}),
                    )
                    .unwrap(),
                    zstd: false,
                },
            })
            .expect("enqueue");
        }
    }
    drop(tx);
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let hopper = hopper_at(&base);
        let pending = AtomicUsize::new(5);
        hopper.serve(jobs, &pending);
        done.send((
            pending.load(Ordering::Relaxed),
            hopper.tally.uploaded.load(Ordering::Relaxed),
            hopper.tally.failed.load(Ordering::Relaxed),
        ))
        .unwrap();
    });
    assert_eq!(
        finished
            .recv_timeout(Duration::from_secs(10))
            .expect("drain all workers"),
        (0, 5, 0)
    );
    server.join().expect("HTTP server");
    assert!(
        overlapped.load(Ordering::Relaxed),
        "one stalled digest blocked another lane"
    );
    let completed = std::mem::take(&mut *completed.lock().unwrap());
    for (sha, count) in [(&slow, 3), (&fast, 2)] {
        let sequence: Vec<_> = completed
            .iter()
            .filter(|(s, _)| s == sha)
            .map(|(_, n)| *n)
            .collect();
        assert_eq!(sequence, (0..count).collect::<Vec<_>>());
    }
}

fn artifact(sha: &str, size: u64, bytes: ArtifactBytes) -> UploadArtifact {
    UploadArtifact {
        sha256: sha.to_string(),
        size,
        filename: "x.bin".to_string(),
        bytes,
        sidecar: br#"{"schema_version":"1.0"}"#.to_vec(),
        backfill: false,
    }
}

/// A one-file report with a large finding, for a result to upload.
fn fixture_report(evidence: &str) -> cleave::types::CompactReport {
    serde_json::from_value(serde_json::json!({
        "rev": "r1",
        "files": [{"id": 0, "path": "x.txt", "type": "text", "sha": "a".repeat(64), "size": 1,
                   "traits": [{"id": "x::evidence", "crit": 2, "conf": 1.0, "desc": evidence}]}]
    }))
    .expect("fixture report")
}

/// The decoded lines of a version 8 body.
fn body_lines(body: &[u8]) -> Vec<serde_json::Value> {
    body.split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).expect("each line is JSON"))
        .collect()
}

/// A result's body is compressed whole and decompresses to the exact lines
/// the document wrote, filed under the key the caller gave.
#[test]
fn a_result_body_round_trips_through_zstd() {
    let result = crate::engine::fixture_result(&"a".repeat(64), &fixture_report("e"));
    let doc = result.upload.as_ref().expect("classified for upload");
    let stamp = UploadStamp {
        at: "2026-11-12T00:00:00Z",
        model: "test",
        worker: Some("scan-fs"),
        ms: None,
    };
    let tally = Tally::default();
    let encoded = prepare_result(doc, &"b".repeat(64), None, &stamp, &tally).expect("encodes");
    assert!(encoded.zstd);
    let body = zstd::decode_all(encoded.bytes.as_slice()).expect("valid zstd");
    assert_eq!(
        Some(body.clone()),
        doc.encode(&"b".repeat(64), None, &stamp)
    );
    let lines = body_lines(&body);
    assert_eq!(lines[0]["v"], 8);
    assert_eq!(
        lines[0]["scan"]["worker"], "scan-fs",
        "hopper requires an uploader's name"
    );
    assert_eq!(
        lines[1]["sha256"],
        "b".repeat(64),
        "filed under the caller's key"
    );
    assert_eq!(tally.failed.load(Ordering::Relaxed), 0);
}

/// A caller posting its own results sends exactly the body the uploader
/// would, and nothing for a result classified without one.
#[test]
fn an_exported_body_is_the_uploaders() {
    let mut result = crate::engine::fixture_result(&"a".repeat(64), &fixture_report("e"));
    result.analyzed_at = "2026-11-12T00:00:00Z".to_owned();
    let (bytes, encoding) =
        encode_result_body(&result, &"a".repeat(64), "postdoc-1", Some(412)).expect("encodes");
    assert_eq!(encoding, Some("zstd"));
    let stamp = UploadStamp {
        at: &result.analyzed_at,
        model: &result.version,
        worker: Some("postdoc-1"),
        ms: Some(412),
    };
    let doc = result.upload.as_ref().expect("classified for upload");
    let body = zstd::decode_all(bytes.as_slice()).expect("valid zstd");
    assert_eq!(
        Some(body.clone()),
        doc.encode(&"a".repeat(64), None, &stamp)
    );
    assert_eq!(
        body_lines(&body)[0]["scan"]["ms"],
        412,
        "the caller's timing"
    );
    result.upload = None;
    assert!(encode_result_body(&result, &"a".repeat(64), "postdoc-1", None).is_none());
}

/// A result classified without an upload body cannot be renewed; that is a
/// lost verdict, counted where an operator sees it.
#[test]
fn a_result_without_an_upload_body_is_counted_lost() {
    let (tx, _jobs) = queue::channel::<Job>(UPLOAD_QUEUE_DEPTH);
    let uploader = Uploader {
        tx: Some(tx),
        worker: None,
        worker_name: "test".to_owned(),
        pending: Arc::default(),
        tally: Arc::default(),
    };
    let mut result = crate::engine::fixture_result(&"a".repeat(64), &fixture_report("e"));
    result.upload = None;
    uploader.submit("a".repeat(64), None, &result);
    assert_eq!(uploader.stats().failed, 1);
    assert_eq!(uploader.stats().pending, 0);
}

#[test]
fn default_worker_name_is_valid_for_hopper() {
    let name = default_worker_name();
    assert!(!name.is_empty());
    assert!(name.len() <= MAX_WORKER_NAME_LEN);
    // Mirrors hopper's `validWorkerName`: printable ASCII, no spaces.
    assert!(name.chars().all(|c| c.is_ascii_graphic()));
}

/// The `/v1/intake/held` probe walks the list like every other call.
///
/// Its failure is safe — an unanswered probe makes the uploader treat
/// every artifact as missing, and hopper's upsert is idempotent — but it
/// is not free: that is the probe whose whole job is keeping bytes hopper
/// already holds off the wire. Stopping at a dead replica would spend the
/// outage re-uploading the corpus to a primary that was up the whole time.
#[test]
fn a_dead_replica_does_not_stop_the_known_probe() {
    // Port 1 refuses immediately, so this measures the decision rather
    // than a timeout.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let addr = listener.local_addr().expect("server address");
    let server = serve_one(
        listener,
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
          Content-Length: 16\r\nConnection: close\r\n\r\n{\"known\":[\"aa\"]}",
    );
    let hopper = hopper_at(&format!("http://127.0.0.1:1,http://{addr}"));
    let known = hopper.known(&["aa", "bb"]);
    server.join().expect("server thread");
    assert!(known.contains("aa"), "the primary's answer was discarded");
    assert!(!known.contains("bb"), "hopper did not claim to hold bb");
}

#[test]
fn upload_sends_configured_bearer_token() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let addr = listener.local_addr().expect("server address");
    let server = serve_one(listener, NO_CONTENT);
    let mut hopper = hopper_at(&format!("http://{addr}"));
    hopper.token = Some("test-secret");
    let art = artifact(&"a".repeat(64), 1, ArtifactBytes::File(PathBuf::new()));
    assert!(hopper.upload(&art, None));
    let request = String::from_utf8_lossy(&server.join().expect("server thread")).into_owned();
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer test-secret\r\n"),
        "request headers: {request}"
    );
}

/// A file-backed artifact streams from disk with its length declared up
/// front, behind the provenance part hopper's handler reads first.
#[test]
fn an_artifact_streams_from_disk_after_its_provenance() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("payload");
    std::fs::write(&path, b"the artifact bytes").expect("write");
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let addr = listener.local_addr().expect("server address");
    let server = serve_one(listener, NO_CONTENT);
    let hopper = hopper_at(&format!("http://{addr}"));
    let art = artifact(&"b".repeat(64), 18, ArtifactBytes::File(path.clone()));
    assert!(hopper.upload(&art, Some(&Body::File(path))));
    let request = server.join().expect("server thread");
    let text = String::from_utf8_lossy(&request);
    assert!(
        text.to_ascii_lowercase().contains("content-length:"),
        "{text}"
    );
    let provenance = text.find(r#"name="provenance""#).expect("provenance part");
    let file = text
        .find(r#"name="file"; filename="x.bin""#)
        .expect("file part");
    assert!(provenance < file, "provenance must precede the file");
    assert!(text.contains("the artifact bytes"), "{text}");
}

/// An artifact over `HOPPER_MAX_UPLOAD_ARTIFACT_BYTES` must never reach
/// `/v1/intake/upload` at all — hopper would reject it outright, and the
/// connection-drop that rejection produces looks identical to a real
/// network fault (see the constant's doc comment). `reconcile` is expected
/// to skip straight past it after the `/v1/intake/held` probe reports it
/// missing, rather than attempting and retrying a doomed upload.
#[test]
fn oversized_artifact_skips_the_byte_upload() {
    let known_listener = TcpListener::bind("127.0.0.1:0").expect("bind known server");
    let known_addr = known_listener.local_addr().expect("known address");
    let known_server = serve_one(
        known_listener,
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
          Content-Length: 12\r\nConnection: close\r\n\r\n{\"known\":[]}",
    );

    // Bound but never accepted: a wrongly-attempted byte upload would
    // connect here, which the accept() below catches. Nonblocking so a
    // bug that skips the size check fails the assertion instead of
    // hanging the test for the full upload retry budget.
    let upload_listener = TcpListener::bind("127.0.0.1:0").expect("bind upload server");
    let upload_addr = upload_listener.local_addr().expect("upload address");
    upload_listener
        .set_nonblocking(true)
        .expect("nonblocking upload listener");

    let mut hopper = hopper_at(&format!("http://{known_addr}"));
    hopper.upload = Route::new(&[format!("http://{upload_addr}")], "/v1/intake/upload");
    let art = artifact(
        &"d".repeat(64),
        HOPPER_MAX_UPLOAD_ARTIFACT_BYTES + 1,
        ArtifactBytes::File(PathBuf::from("/nonexistent/huge.bin")),
    );
    hopper.reconcile(None, &mut HashSet::new(), vec![art]);
    known_server.join().expect("known server thread");

    match upload_listener.accept() {
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {} // nothing connected — correct
        Ok(_) => panic!("oversized artifact triggered a byte upload attempt"),
        Err(e) => panic!("unexpected accept error: {e}"),
    }
}

/// A job the queue refuses must leave no phantom behind in `pending`: the
/// counter is what tells "nothing to file" from "filing is stuck".
#[test]
fn a_refused_job_does_not_leak_the_pending_count() {
    let (tx, jobs) = queue::channel::<Job>(UPLOAD_QUEUE_DEPTH);
    drop(jobs);
    let uploader = Uploader {
        tx: Some(tx),
        worker: None,
        worker_name: "test".to_owned(),
        pending: Arc::default(),
        tally: Arc::default(),
    };
    let dep = crate::engine::DepResult {
        sha256: "c".repeat(64),
        locator: "pkg:npm/x@1".to_string(),
        url: "https://example/x-1.tgz".to_string(),
        size: 1,
        provenance: None,
        verdict: None,
        members: crate::engine::MemberEvals::new(),
        upload: None,
    };
    uploader.submit_dependencies(vec![dep], "v", "t");
    uploader.submit_artifacts(vec![artifact(
        &"e".repeat(64),
        1,
        ArtifactBytes::File(PathBuf::new()),
    )]);
    assert_eq!(uploader.stats().pending, 0);
}

/// A re-fetch recovers bytes the blob cache evicted, but a locator is not
/// always a pin — a versionless PURL re-resolves to today's `latest`, and a
/// tag can move. Only bytes that still hash to the analyzed digest may be
/// uploaded under it; anything else would file one artifact's content under
/// another's identity, which is worse than the absence being repaired.
#[test]
fn only_bytes_matching_the_analyzed_digest_are_uploaded() {
    let bytes = b"the exact bytes the verdict was computed over";
    let sha = {
        use sha2::{Digest as _, Sha256};
        format!("{:x}", Sha256::digest(bytes))
    };
    assert!(
        bytes_match_digest(bytes, &sha, "pkg:npm/x@1"),
        "the analyzed bytes must be accepted"
    );
    assert!(
        !bytes_match_digest(b"different bytes at the same locator", &sha, "pkg:npm/x@1"),
        "a moved tag or re-resolved range must not be filed under the old digest"
    );
    assert!(
        !bytes_match_digest(bytes, &"0".repeat(64), "pkg:npm/x@1"),
        "an unrelated digest must not accept these bytes"
    );
    // Empty content hashes to a real, well-known digest rather than to
    // nothing, so a truncated or zero-length re-fetch is a mismatch and not
    // an accidental pass.
    assert!(!bytes_match_digest(b"", &sha, "pkg:npm/x@1"));
}

/// A dependency's upload artifact loads its bytes lazily from the fetch cache
/// (never eagerly), derives its stored filename from the resolved URL, and is
/// marked backfillable so its captured registry provenance lands even when
/// hopper already holds the bytes. Built here from an *unevaluated*
/// dependency: the artifact is independent of the verdict, so bytes and
/// provenance reach hopper even when scan has no verdict to post for them.
/// The unsupported PURL proves artifact construction does not perform a
/// registry lookup.
#[test]
fn dep_artifact_loads_bytes_lazily_and_is_backfillable() {
    let dep = crate::engine::DepResult {
        sha256: "c".repeat(64),
        locator: "pkg:bogus/x@1".to_string(),
        url: "https://example/x-1.tgz".to_string(),
        size: 99,
        provenance: Some(crate::provenance::RegistryProvenance::from_record_sources(
            crate::test_registry("bogus", "x", "1"),
            &[fletch::fetch::RecordedSource {
                url: "https://registry.example/x".to_string(),
                status: 200,
                content_type: Some("application/json".to_string()),
                size: 20,
                bytes: Some(br#"{"provider_only":42}"#.to_vec()),
            }],
        )),
        verdict: None,
        members: crate::engine::MemberEvals::new(),
        upload: None,
    };
    let art = dep_artifact(&dep, "scan+test", "2026-06-28T00:00:00Z").expect("sidecar builds");
    assert_eq!(art.sha256, "c".repeat(64));
    assert_eq!(art.size, 99);
    assert_eq!(
        art.filename, "x-1.tgz",
        "filename derived from the fetch URL"
    );
    assert!(art.backfill, "a dependency's provenance is backfillable");
    let sidecar: serde_json::Value = serde_json::from_slice(&art.sidecar).unwrap();
    assert_eq!(
        sidecar["registry"]["raw"][0]["body"]["provider_only"], 42,
        "upload uses the captured provider snapshot"
    );
    assert_eq!(sidecar["package"]["purl"], "pkg:bogus/x@1");
    assert!(
        matches!(&art.bytes, ArtifactBytes::Cached { locator } if locator == "pkg:bogus/x@1"),
        "bytes load lazily from the cache by locator",
    );
}
