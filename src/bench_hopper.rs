//! Minimal, dependency-light mock hopper for benchmarking and testing the
//! worker model in isolation — the litmus analogue of cleave's
//! `make benchmark DATASET=...`.
//!
//! It serves a local dataset directory over the same HTTP surface a real
//! hopper exposes to workers, so [`crate::worker`] exercises its real claim →
//! prefetch → dispatch → result-post path with no external services:
//!
//! - `GET /api/next?...&count=N` → up to `N` distinct jobs (one per dataset
//!   file, each with the file's *real* sha256 so the worker's local-file and
//!   download integrity checks pass); `204 No Content` once every file has been
//!   handed out, so a `--max-jobs <count>` worker drains and exits.
//! - `GET /data/{url-encoded path}` → file bytes (streamed from disk, so the
//!   hopper's own RSS stays tiny and never pollutes the worker's measurement).
//! - `POST /api/result` → tallied (and optionally appended to a dump file for
//!   parity diffing between runs).
//!
//! Pure `std` (`std::net`, `std::thread`, `std::fs`) plus `sha2`; no async
//! runtime and no extra dependencies.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

/// One served file, precomputed at build time.
#[derive(Clone, Debug)]
struct Job {
    sha256: String,
    /// Forward-slash relative path under the dataset root (the `/data/{path}`
    /// key and the relative path the worker joins with `--data-dir`).
    path: String,
    size_bytes: u64,
}

/// A built, ready-to-serve mock hopper. Walk the dataset once (hashing every
/// file), then [`serve`](BenchHopper::serve) it on a port.
#[derive(Debug)]
pub struct BenchHopper {
    jobs: Vec<Job>,
    by_path: HashMap<String, PathBuf>,
    cursor: AtomicUsize,
    results: Arc<AtomicUsize>,
    dump: Option<Arc<Mutex<File>>>,
}

/// A running hopper: the bound port plus live counters. Dropping it leaves the
/// accept thread running (detached) — fine for a short-lived benchmark process
/// or a test that ends with process teardown.
#[derive(Debug)]
pub struct Handle {
    /// The port the hopper bound (resolved if `serve(0)` was used).
    pub port: u16,
    /// Total number of jobs (dataset files) the hopper will hand out.
    pub jobs_total: usize,
    results: Arc<AtomicUsize>,
}

impl Handle {
    /// Number of `/api/result` posts received so far.
    #[must_use]
    pub fn results_received(&self) -> usize {
        self.results.load(Ordering::Relaxed)
    }
}

impl BenchHopper {
    /// Walk `dataset` recursively, hashing every file, and build the job set.
    /// `dump`, if set, is a file each posted result body is appended to.
    pub fn build(dataset: &Path, dump: Option<PathBuf>) -> io::Result<Self> {
        let mut files = Vec::new();
        collect_files(dataset, dataset, &mut files)?;
        files.sort_by(|a, b| a.0.cmp(&b.0)); // deterministic job order across runs

        let mut jobs = Vec::with_capacity(files.len());
        let mut by_path = HashMap::with_capacity(files.len());
        for (rel, abs) in files {
            let (sha256, size_bytes) = hash_file(&abs)?;
            by_path.insert(rel.clone(), abs);
            jobs.push(Job {
                sha256,
                path: rel,
                size_bytes,
            });
        }

        let dump = match dump {
            Some(path) => Some(Arc::new(Mutex::new(
                OpenOptions::new().create(true).append(true).open(path)?,
            ))),
            None => None,
        };

        Ok(Self {
            jobs,
            by_path,
            cursor: AtomicUsize::new(0),
            results: Arc::new(AtomicUsize::new(0)),
            dump,
        })
    }

    /// Bind `port` (0 = ephemeral) and serve on a background accept thread.
    pub fn serve(self, port: u16) -> io::Result<Handle> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let bound = listener.local_addr()?.port();
        let jobs_total = self.jobs.len();
        let results = Arc::clone(&self.results);
        let server = Arc::new(self);

        std::thread::Builder::new()
            .name("bench-hopper".into())
            .spawn(move || {
                for stream in listener.incoming() {
                    let Ok(stream) = stream else { continue };
                    let server = Arc::clone(&server);
                    // Thread-per-connection: a handful of worker connections at
                    // a time; simplicity beats a pool here.
                    let _ = std::thread::Builder::new()
                        .name("bench-hopper-conn".into())
                        .spawn(move || {
                            let _ = server.handle(stream);
                        });
                }
            })?;

        Ok(Handle {
            port: bound,
            jobs_total,
            results,
        })
    }

    fn handle(&self, mut stream: TcpStream) -> io::Result<()> {
        let (method, target, body) = read_request(&mut stream)?;
        if method == "GET" && target.starts_with("/api/next") {
            self.serve_next(&mut stream, &target)
        } else if method == "GET" && target.starts_with("/data/") {
            self.serve_data(&mut stream, &target)
        } else if method == "POST" && target.starts_with("/api/result") {
            self.serve_result(&mut stream, &body)
        } else if target.starts_with("/api/heartbeat") {
            // The worker checks in here; accept and ignore.
            respond_body(&mut stream, "200 OK", b"{}")
        } else {
            respond_head(&mut stream, "404 Not Found", 0)
        }
    }

    fn serve_next(&self, stream: &mut TcpStream, target: &str) -> io::Result<()> {
        let count = query_param(target, "count").and_then(|v| v.parse().ok()).unwrap_or(1);
        let start = self.cursor.fetch_add(count, Ordering::SeqCst);
        if start >= self.jobs.len() {
            // Drained: the real hopper returns 204 when no work remains.
            return respond_head(stream, "204 No Content", 0);
        }
        let end = (start + count).min(self.jobs.len());
        let mut body = String::from("{\"jobs\":[");
        for (i, job) in self.jobs[start..end].iter().enumerate() {
            if i > 0 {
                body.push(',');
            }
            // file_type left as "data"; cleave detects the real type from bytes.
            body.push_str(&format!(
                "{{\"sha256\":\"{}\",\"path\":{},\"size_bytes\":{},\"file_type\":\"data\"}}",
                job.sha256,
                json_string(&job.path),
                job.size_bytes,
            ));
        }
        body.push_str("]}");
        respond_body(stream, "200 OK", body.as_bytes())
    }

    fn serve_data(&self, stream: &mut TcpStream, target: &str) -> io::Result<()> {
        // Strip "/data/" and any query, then percent-decode each path segment
        // (the worker percent-encodes segments and joins with '/').
        let raw = target
            .strip_prefix("/data/")
            .unwrap_or(target)
            .split(['?', '#'])
            .next()
            .unwrap_or_default();
        let rel = raw
            .split('/')
            .map(percent_decode)
            .collect::<Vec<_>>()
            .join("/");
        let Some(abs) = self.by_path.get(&rel) else {
            return respond_head(stream, "404 Not Found", 0);
        };
        let file = File::open(abs)?;
        let len = file.metadata()?.len();
        respond_head(stream, "200 OK", len)?;
        // Stream the body so the hopper never holds the whole file resident.
        let mut reader = BufReader::new(file);
        io::copy(&mut reader, stream)?;
        stream.flush()
    }

    fn serve_result(&self, stream: &mut TcpStream, body: &[u8]) -> io::Result<()> {
        self.results.fetch_add(1, Ordering::Relaxed);
        if let Some(dump) = &self.dump {
            // The worker zstd-compresses result bodies (raw JSON on fallback).
            // Record one clean line per result — the sample's top-level sha256 —
            // so completeness = distinct lines, not unparseable binary blobs.
            if let Ok(mut f) = dump.lock() {
                let _ = writeln!(f, "{}", result_sample_sha256(body));
            }
        }
        respond_body(stream, "200 OK", b"{}")
    }
}

/// Extract the result's top-level sample sha256 for the manifest line,
/// decompressing the (zstd) post body first and falling back gracefully.
fn result_sample_sha256(body: &[u8]) -> String {
    let json = zstd::decode_all(body).unwrap_or_else(|_| body.to_vec());
    // The first `"sha256":"<64 hex>"` is the ResultPayload's top-level hash; the
    // embedded cleave report's per-member hashes appear later in the body.
    first_sha256(&json).unwrap_or_else(|| "unknown".to_string())
}

/// Find the first `"sha256":"<64 lowercase hex>"` value in a JSON byte slice.
fn first_sha256(json: &[u8]) -> Option<String> {
    let key = b"\"sha256\"";
    let key_pos = json.windows(key.len()).position(|w| w == key)?;
    let rest = &json[key_pos + key.len()..];
    let quote = rest.iter().position(|&b| b == b'"')?;
    let hex = &rest[quote + 1..];
    if hex.len() >= 64 && hex[..64].iter().all(u8::is_ascii_hexdigit) {
        Some(String::from_utf8_lossy(&hex[..64]).into_owned())
    } else {
        None
    }
}

/// Recursively collect `(forward-slash-relpath, absolute)` for every file under
/// `dir`, relative to `root`.
fn collect_files(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_files(root, &path, out)?;
        } else if file_type.is_file() {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            out.push((rel, path));
        }
    }
    Ok(())
}

/// Stream a file through sha256, returning `(hex_digest, size_bytes)`.
fn hash_file(path: &Path) -> io::Result<(String, u64)> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut size = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        size += n as u64;
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{b:02x}"));
    }
    Ok((hex, size))
}

/// Read an HTTP/1.1 request: returns `(method, target, body)`. Reads headers to
/// `\r\n\r\n`, then `Content-Length` body bytes (already-buffered bytes reused).
fn read_request(stream: &mut TcpStream) -> io::Result<(String, String, Vec<u8>)> {
    let mut buf = Vec::with_capacity(2048);
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "no request"));
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > 1 << 20 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "header too large"));
        }
    };

    let head = String::from_utf8_lossy(&buf[..header_end]);
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or_default().to_string();

    let content_len = lines
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim().eq_ignore_ascii_case("content-length").then(|| v.trim().parse().ok())
        })
        .flatten()
        .unwrap_or(0usize);

    let body_start = header_end + 4;
    let mut body = buf[body_start.min(buf.len())..].to_vec();
    while body.len() < content_len {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(content_len);
    Ok((method, target, body))
}

fn respond_head(stream: &mut TcpStream, status: &str, content_len: u64) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {content_len}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(head.as_bytes())
}

fn respond_body(stream: &mut TcpStream, status: &str, body: &[u8]) -> io::Result<()> {
    respond_head(stream, status, body.len() as u64)?;
    stream.write_all(body)?;
    stream.flush()
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn query_param<'a>(target: &'a str, key: &str) -> Option<&'a str> {
    let query = target.split('?').nth(1)?;
    query
        .split('&')
        .find_map(|kv| kv.split_once('=').filter(|(k, _)| *k == key).map(|(_, v)| v))
}

/// Percent-decode one path segment (reverses the worker's `url_encode_into`).
fn percent_decode(seg: &str) -> String {
    let bytes = seg.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2]))
        {
            out.push(hi << 4 | lo);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Serialize a string as a JSON string literal (escapes `"` and `\`; dataset
/// paths don't contain control characters).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn write_file(dir: &Path, rel: &str, bytes: &[u8]) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        File::create(&path).unwrap().write_all(bytes).unwrap();
    }

    fn get(port: u16, target: &str) -> (String, Vec<u8>) {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(
            stream,
            "GET {target} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        let mut resp = Vec::new();
        stream.read_to_end(&mut resp).unwrap();
        let split = find_subslice(&resp, b"\r\n\r\n").unwrap();
        let status = String::from_utf8_lossy(&resp[..split]).lines().next().unwrap().to_string();
        (status, resp[split + 4..].to_vec())
    }

    #[test]
    fn serves_jobs_then_drains_and_serves_data() {
        let dir = std::env::temp_dir().join(format!("bench_hopper_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_file(&dir, "a.txt", b"hello");
        write_file(&dir, "sub/b dir.txt", b"world!!");

        let hopper = BenchHopper::build(&dir, None).unwrap();
        assert_eq!(hopper.jobs.len(), 2);
        let handle = hopper.serve(0).unwrap();
        let port = handle.port;

        // First /api/next hands out both jobs (deterministic, sorted by path).
        let (status, body) = get(port, "/api/next?count=10");
        assert!(status.contains("200"), "status: {status}");
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("\"a.txt\""), "body: {body}");
        // sha256("hello")
        assert!(
            body.contains("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"),
            "body: {body}"
        );

        // Second poll drains: 204.
        let (status, _) = get(port, "/api/next?count=10");
        assert!(status.contains("204"), "status: {status}");

        // /data/ with a percent-encoded space-containing segment.
        let (status, bytes) = get(port, "/data/sub/b%20dir.txt");
        assert!(status.contains("200"), "status: {status}");
        assert_eq!(bytes, b"world!!");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn first_sha256_takes_top_level_hash() {
        let h = "a".repeat(64);
        let m = "b".repeat(64);
        let body = format!("{{\"sha256\":\"{h}\",\"worker\":\"w\",\"raw\":{{\"sha256\":\"{m}\"}}}}");
        assert_eq!(first_sha256(body.as_bytes()).as_deref(), Some(h.as_str()));
        assert_eq!(first_sha256(b"{\"worker\":\"w\"}"), None);
    }

    #[test]
    fn result_sample_sha256_decompresses_zstd() {
        let h = "c".repeat(64);
        let json = format!("{{\"sha256\":\"{h}\",\"worker\":\"w\"}}");
        let compressed = zstd::encode_all(json.as_bytes(), 3).unwrap();
        assert_eq!(result_sample_sha256(&compressed), h);
        // Raw (uncompressed-fallback) bodies also work.
        assert_eq!(result_sample_sha256(json.as_bytes()), h);
    }

    #[test]
    fn percent_decode_reverses_segment_encoding() {
        // Matches the worker's `url_encode_into`: unreserved chars pass through,
        // everything else is %XX (uppercase hex).
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("c%25d"), "c%d");
        assert_eq!(percent_decode("plain.txt"), "plain.txt");
    }
}
