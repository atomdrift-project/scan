//! A stalled Hopper must not park analysis, lose uploads, or reorder a
//! verdict ahead of its artifact. No model bundle or external server required.

use std::collections::HashSet;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::time::Duration;

use crate::upload::{ArtifactBytes, UploadArtifact, Uploader};

const SAMPLES: usize = 40;
const DEADLINE: Duration = Duration::from_secs(20);

/// A benign result for the artifact `sha`, whose report carries a finding
/// large enough that losing or truncating it would show.
fn result(sha: &str) -> crate::engine::ScanResult {
    let report = serde_json::from_value(serde_json::json!({
        "rev": "r1",
        "files": [{"id": 0, "path": "x.txt", "type": "text", "sha": sha, "size": 1,
                   "traits": [{"id": "x::evidence", "crit": 2, "conf": 1.0,
                               "desc": "preserve-me".repeat(4096)}]}]
    }))
    .unwrap();
    crate::engine::fixture_result(sha, &report)
}

fn request(stream: &mut TcpStream) -> (String, Vec<u8>) {
    // On macOS and the BSDs an accepted socket inherits the listener's
    // non-blocking mode; reads here must block, up to the timeout.
    stream.set_nonblocking(false).unwrap();
    stream.set_read_timeout(Some(DEADLINE)).unwrap();
    let mut bytes = Vec::new();
    let mut buf = [0u8; 8192];
    let body_start = loop {
        let n = stream.read(&mut buf).expect("request headers");
        assert!(n > 0, "headers truncated");
        bytes.extend_from_slice(&buf[..n]);
        if let Some(at) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            break at + 4;
        }
    };
    let headers = String::from_utf8_lossy(&bytes[..body_start]).to_ascii_lowercase();
    let length = headers
        .lines()
        .find_map(|line| {
            line.strip_prefix("content-length:")
                .map(|n| n.trim().parse::<usize>().unwrap())
        })
        .expect("sized request");
    while bytes.len() < body_start + length {
        let n = stream.read(&mut buf).expect("request body");
        assert!(n > 0, "body truncated");
        bytes.extend_from_slice(&buf[..n]);
    }
    let body = &bytes[body_start..body_start + length];
    let decoded = if headers.contains("content-encoding: zstd") {
        zstd::decode_all(body).unwrap()
    } else {
        body.to_vec()
    };
    (headers, decoded)
}

#[test]
fn slow_hopper_spools_overflow_then_flushes_every_verdict_in_order() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (blocked_tx, blocked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let server = std::thread::spawn(move || {
        let mut known = HashSet::new();
        let mut stored = HashSet::new();
        let mut renewed = HashSet::new();
        let started = std::time::Instant::now();
        for index in 0..SAMPLES * 3 {
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            started.elapsed() < DEADLINE * 3,
                            "uploader stopped delivering"
                        );
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("accept: {e}"),
                }
            };
            let (headers, body) = request(&mut stream);
            if index == 0 {
                blocked_tx.send(()).unwrap();
                release_rx
                    .recv_timeout(DEADLINE * 2)
                    .expect("release stalled Hopper");
            }
            let reply = if headers.starts_with("post /v1/intake/held ") {
                let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                let sha = body["sha256"][0].as_str().expect("known sha").to_owned();
                assert!(known.insert(sha.clone()), "duplicate artifact probe");
                serde_json::json!({"known": []}).to_string()
            } else if headers.starts_with("post /v1/intake/upload ") {
                let sha = format!("{:064x}", index / 3);
                assert!(known.contains(&sha), "upload preceded probe");
                let body = String::from_utf8(body).unwrap();
                assert!(
                    body.contains(&format!("fixture bytes {}", index / 3)),
                    "lost staged bytes"
                );
                assert!(body.contains("name=\"file\""));
                assert!(stored.insert(sha), "duplicate artifact upload");
                "{}".to_owned()
            } else {
                assert!(headers.starts_with("post /v1/work/result "), "{headers}");
                assert!(headers.contains("content-type: application/x-ndjson"));
                let lines: Vec<serde_json::Value> = body
                    .split(|&b| b == b'\n')
                    .filter(|line| !line.is_empty())
                    .map(|line| serde_json::from_slice(line).unwrap())
                    .collect();
                let sha = lines[1]["sha256"].as_str().expect("result sha").to_owned();
                assert!(stored.contains(&sha), "verdict preceded artifact upload");
                assert!(renewed.insert(sha), "duplicate renewal");
                assert_eq!(lines[0]["scan"]["worker"], "backpressure-test");
                assert_eq!(lines[1]["verdict"]["fires_at"], -1);
                assert_eq!(lines[2]["traits"][0]["desc"], "preserve-me".repeat(4096));
                "{}".to_owned()
            };
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                reply.len(),
                reply
            )
            .unwrap();
        }
        assert_eq!(renewed.len(), SAMPLES);
    });
    let (queued_tx, queued_rx) = mpsc::channel();
    let (flushed_tx, flushed_rx) = mpsc::channel();
    let producer = std::thread::spawn(move || {
        let uploader = Uploader::new(&url, "backpressure-test".into());
        let originals = tempfile::tempdir().unwrap();
        for i in 0..SAMPLES {
            let data = format!("fixture bytes {i}");
            let path = originals.path().join(format!("{i}.txt"));
            std::fs::write(&path, &data).unwrap();
            let sha = format!("{i:064x}");
            uploader
                .submit_artifacts_durable(vec![UploadArtifact {
                    sha256: sha.clone(),
                    size: data.len() as u64,
                    filename: format!("{i}.txt"),
                    bytes: ArtifactBytes::File(path),
                    sidecar: b"{}".to_vec(),
                    backfill: false,
                }])
                .expect("stage and enqueue artifact");
            uploader.submit(sha.clone(), None, &result(&sha));
        }
        // The request-owned originals disappear before Hopper resumes. Both
        // resident and spooled jobs must retain their own staged bytes.
        drop(originals);
        let stats = uploader.stats();
        queued_tx.send((stats.pending, stats.capacity)).unwrap();
        drop(uploader); // Must flush; it cannot return while Hopper is held.
        flushed_tx.send(()).unwrap();
    });
    blocked_rx
        .recv_timeout(DEADLINE)
        .expect("Hopper received first request");
    let queued = queued_rx.recv_timeout(DEADLINE);
    let premature_flush = flushed_rx.try_recv().is_ok();
    // Release even on regression, so the old bounded queue can drain and the
    // test reports a failure instead of leaving a blocked uploader behind.
    release_tx.send(()).unwrap();
    // The server first: if it gave up, the uploader is still retrying toward
    // its renewal budget, and waiting on the producer would hang the test
    // instead of failing it.
    server.join().expect("server");
    producer.join().expect("producer");
    let (pending, capacity) = queued.expect("analysis handoff blocked on Hopper");
    assert!(
        pending > capacity,
        "fixture did not overflow the resident queue"
    );
    assert!(!premature_flush, "drop returned before uploads completed");
    flushed_rx
        .recv_timeout(DEADLINE)
        .expect("all uploads flushed");
}
