//! Integration regression: a wedged hopper `/api/result` must not freeze the
//! worker's other analysis slots.
//!
//! Replays the Aug-18 failure mode end-to-end against a mock hopper:
//! one sample's result POST hangs forever; the other samples must still
//! complete and post within a bound.
//!
//! **Requires `SCAN_MODELS_DIR`** (same convention as `server_analyze.rs`).

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::collections::{HashSet, VecDeque};
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use scan::worker::{WorkerConfig, run};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, Notify};

/// Cold YARA + two tiny analyses should finish well under this; anything near
/// the ceiling means the sibling path is wedged behind the hanging post.
const SIBLING_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Clone)]
struct Sample {
    sha256: String,
    rel_path: String,
    bytes: Vec<u8>,
}

struct HopperState {
    /// Remaining jobs for `/api/next` (hang sample first).
    jobs: Mutex<VecDeque<Sample>>,
    /// Files served from `/data/` and resolved via `--data-dir`.
    by_sha: Vec<Sample>,
    hang_sha: String,
    /// SHAs that successfully posted (never includes `hang_sha`).
    posted: Mutex<HashSet<String>>,
    hang_entered: Notify,
    shutdown: AtomicBool,
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn write_sample(dir: &Path, rel: &str, body: &[u8]) -> Sample {
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create sample parent");
    }
    let mut f = std::fs::File::create(&path).expect("create sample");
    f.write_all(body).expect("write sample");
    Sample {
        sha256: sha256_hex(body),
        rel_path: rel.to_string(),
        bytes: body.to_vec(),
    }
}

async fn read_http_request(stream: &mut TcpStream) -> Option<(String, String, Vec<u8>)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = stream.read(&mut tmp).await.ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(header_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let header_bytes = &buf[..header_end];
            let headers = String::from_utf8_lossy(header_bytes);
            let mut lines = headers.lines();
            let request_line = lines.next()?;
            let mut parts = request_line.split_whitespace();
            let method = parts.next()?.to_string();
            let target = parts.next()?.to_string();
            let mut content_length = 0usize;
            for line in lines {
                let lower = line.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap_or(0);
                }
            }
            let body_start = header_end + 4;
            while buf.len() < body_start + content_length {
                let n = stream.read(&mut tmp).await.ok()?;
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let body = buf
                .get(body_start..body_start + content_length)
                .unwrap_or(&[])
                .to_vec();
            return Some((method, target, body));
        }
        if buf.len() > 64 * 1024 {
            break;
        }
    }
    None
}

async fn respond(stream: &mut TcpStream, status: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes()).await;
    let _ = stream.write_all(body).await;
    let _ = stream.flush().await;
}

fn parse_result_sha(body: &[u8]) -> Option<String> {
    // Worker posts zstd-compressed JSON; fall back to raw JSON if uncompressed.
    let json = zstd::decode_all(body).unwrap_or_else(|_| body.to_vec());
    let value: serde_json::Value = serde_json::from_slice(&json).ok()?;
    value
        .get("sha256")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn parse_count(target: &str) -> usize {
    target
        .split(['?', '&'])
        .find_map(|kv| kv.strip_prefix("count="))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

async fn handle_conn(mut stream: TcpStream, state: Arc<HopperState>) {
    let Some((method, target, body)) = read_http_request(&mut stream).await else {
        return;
    };
    if state.shutdown.load(Ordering::Relaxed) {
        return;
    }

    if method == "GET" && target.starts_with("/api/next") {
        let want = parse_count(&target);
        let mut jobs = state.jobs.lock().await;
        let mut batch = Vec::new();
        for _ in 0..want {
            let Some(sample) = jobs.pop_front() else {
                break;
            };
            batch.push(serde_json::json!({
                "sha256": sample.sha256,
                "path": sample.rel_path,
                "size_bytes": sample.bytes.len(),
                "file_type": "text",
            }));
        }
        drop(jobs);
        if batch.is_empty() {
            respond(&mut stream, "204 No Content", b"").await;
        } else {
            let body = serde_json::json!({ "jobs": batch }).to_string();
            respond(&mut stream, "200 OK", body.as_bytes()).await;
        }
        return;
    }

    if method == "GET" && target.starts_with("/api/heartbeat") {
        respond(&mut stream, "200 OK", b"{}").await;
        return;
    }

    if method == "GET" && target.starts_with("/data/") {
        let encoded = target.trim_start_matches("/data/");
        let path = urlencoding_decode(encoded);
        if let Some(sample) = state.by_sha.iter().find(|s| s.rel_path == path) {
            respond(&mut stream, "200 OK", &sample.bytes).await;
        } else {
            respond(&mut stream, "404 Not Found", b"").await;
        }
        return;
    }

    if method == "POST" && target.starts_with("/api/result") {
        let Some(sha) = parse_result_sha(&body) else {
            respond(&mut stream, "400 Bad Request", b"bad body").await;
            return;
        };
        if sha == state.hang_sha {
            state.hang_entered.notify_one();
            // Wedged hopper: accept the connection and never answer.
            std::future::pending::<()>().await;
        }
        state.posted.lock().await.insert(sha);
        respond(&mut stream, "200 OK", b"{}").await;
        return;
    }

    respond(&mut stream, "404 Not Found", b"").await;
}

/// Minimal percent-decoder for `/data/{path}` targets the worker builds with
/// `urlencoding`-style escapes. Good enough for our ASCII test paths.
fn urlencoding_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h1 = bytes[i + 1];
            let h2 = bytes[i + 2];
            if let (Some(a), Some(b)) = (from_hex(h1), from_hex(h2)) {
                out.push((a << 4) | b);
                i += 3;
                continue;
            }
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn from_hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sibling_jobs_complete_while_one_result_post_hangs() {
    let Ok(models_dir) = std::env::var("SCAN_MODELS_DIR") else {
        eprintln!(
            "skipping: SCAN_MODELS_DIR not set (same convention as \
             server_analyze.rs integration tests)"
        );
        return;
    };
    let models_dir = PathBuf::from(models_dir);

    let data = tempfile::tempdir().expect("temp data dir");
    let hang = write_sample(
        data.path(),
        "samples/hang.txt",
        b"wedge-me-on-result-post\n",
    );
    let ok1 = write_sample(
        data.path(),
        "samples/ok1.txt",
        b"sibling-one-should-finish\n",
    );
    let ok2 = write_sample(
        data.path(),
        "samples/ok2.txt",
        b"sibling-two-should-finish\n",
    );
    let hang_sha = hang.sha256.clone();
    let expect_posted: HashSet<String> = [ok1.sha256.clone(), ok2.sha256.clone()].into();
    let all_samples = vec![hang.clone(), ok1.clone(), ok2.clone()];

    let mut queue = VecDeque::new();
    // Hang first so one worker enters the wedged post early.
    queue.push_back(hang);
    queue.push_back(ok1);
    queue.push_back(ok2);

    let state = Arc::new(HopperState {
        by_sha: all_samples,
        hang_sha: hang_sha.clone(),
        jobs: Mutex::new(queue),
        posted: Mutex::new(HashSet::new()),
        hang_entered: Notify::new(),
        shutdown: AtomicBool::new(false),
    });

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock hopper");
    let addr = listener.local_addr().expect("local addr");
    let hopper_state = Arc::clone(&state);
    let hopper = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            if hopper_state.shutdown.load(Ordering::Relaxed) {
                break;
            }
            let state = Arc::clone(&hopper_state);
            tokio::spawn(handle_conn(stream, state));
        }
    });

    let config = WorkerConfig {
        hopper_url: format!("http://{addr}"),
        name: "post-hang-regression".into(),
        workers: NonZeroUsize::new(2).expect("2 workers"),
        poll_secs: 1,
        max_rss_gb: 0,
        model_dir: models_dir,
        thresholds: None,
        data_dir: Some(data.path().to_path_buf()),
        slow_rule_ms: 4000,
        max_jobs: None,
        // Do not exit_if_empty: the hanging post never finishes, so drain
        // would wait forever. The test aborts the worker after siblings post.
        exit_if_empty: false,
        level: None,
        nice: 0,
        interpret: None,
        fetch: scan::fetch::FetchPolicy::default(),
        zip_passwords: scan::ArchivePasswords::default(),
    };

    // Keep YARA cold-cache / SJF off this hang's critical path.
    // SAFETY: single-threaded test setup before the worker task starts.
    unsafe {
        std::env::set_var("CLEAVE_SKIP_YARA_CACHE", "1");
        std::env::set_var("SCAN_SJF", "0");
    }

    let worker = tokio::spawn(run(config));

    tokio::time::timeout(SIBLING_TIMEOUT, state.hang_entered.notified())
        .await
        .expect("hanging /api/result never started — worker never posted the wedge sample");

    let posted_ok = tokio::time::timeout(SIBLING_TIMEOUT, async {
        loop {
            let posted = state.posted.lock().await.clone();
            if expect_posted.is_subset(&posted) {
                return posted;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect(
        "sibling jobs did not post while one /api/result was hung \
         (analysis slot held across post regression)",
    );

    assert!(
        !posted_ok.contains(&hang_sha),
        "hanging sample must not appear in successful posts"
    );
    assert!(
        expect_posted.is_subset(&posted_ok),
        "expected {:?} posted, got {:?}",
        expect_posted,
        posted_ok
    );

    state.shutdown.store(true, Ordering::Relaxed);
    worker.abort();
    hopper.abort();
    let _ = worker.await;
    let _ = hopper.await;
}
