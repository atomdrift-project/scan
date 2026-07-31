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
//! Beyond serving jobs, the hopper is the benchmark's *measurement point*: it
//! stamps each job at claim and at result, so per-job sojourn (claim → result,
//! covering download + queue wait + admission + analysis + post) is measured
//! from outside the worker. A summary — wall over the job window, throughput,
//! and sojourn percentiles broken out by size class — is rewritten to the
//! `summary` path after every result, so it is always current no matter when
//! the benchmark harness stops the process. Job handout order is controlled by
//! [`Order`]: dataset order, seeded shuffle (reproducible realistic mix), or
//! biggest-first (the small-job-starvation worst case).
//!
//! Pure `std` (`std::net`, `std::thread`, `std::fs`) plus `sha2`; no async
//! runtime and no extra dependencies.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use sha2::{Digest, Sha256};

/// Job handout order. Parsed from `--order`; `shuffle` without a seed uses 42.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Dataset enumeration order (path-sorted) — the historical default.
    Fifo,
    /// Deterministic Fisher–Yates shuffle with the given seed: a reproducible
    /// realistic size mix.
    Shuffle(u64),
    /// Largest files first: reproduces the small-job-starvation worst case.
    BigFirst,
    /// Smallest files first: the other extreme, for bracketing.
    SmallFirst,
    /// Each size class spread evenly across the stream, so every claim window
    /// contains a mix of sizes. Emulates a hopper-side size-aware handout — the
    /// global scheduling lever a worker-local reorder window cannot reach.
    Mixed,
}

impl FromStr for Order {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fifo" => Ok(Self::Fifo),
            "big-first" => Ok(Self::BigFirst),
            "small-first" => Ok(Self::SmallFirst),
            "mixed" => Ok(Self::Mixed),
            "shuffle" => Ok(Self::Shuffle(42)),
            other => match other.strip_prefix("shuffle:") {
                Some(seed) => seed
                    .parse()
                    .map(Self::Shuffle)
                    .map_err(|e| format!("bad shuffle seed {seed:?}: {e}")),
                None => Err(format!(
                    "unknown order {other:?} (fifo|shuffle[:seed]|big-first|small-first|mixed)"
                )),
            },
        }
    }
}

impl std::fmt::Display for Order {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fifo => write!(f, "fifo"),
            Self::Shuffle(seed) => write!(f, "shuffle:{seed}"),
            Self::BigFirst => write!(f, "big-first"),
            Self::SmallFirst => write!(f, "small-first"),
            Self::Mixed => write!(f, "mixed"),
        }
    }
}

/// xorshift64* step — a tiny, deterministic PRNG for the seeded shuffle, so the
/// bench needs no rand dependency.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// One served file, precomputed at build time.
#[derive(Clone, Debug)]
struct Job {
    sha256: String,
    /// Forward-slash relative path under the dataset root (the `/data/{path}`
    /// key and the relative path the worker joins with `--data-dir`).
    path: String,
    size_bytes: u64,
}

/// Per-job claim/result timestamps, indexed in lockstep with `jobs`. One mutex
/// guards both: every access touches claim and result state together, and the
/// post rate (one lock per HTTP request) makes contention irrelevant.
#[derive(Debug, Default)]
struct Timings {
    claimed: Vec<Option<Instant>>,
    resulted: Vec<Option<Instant>>,
}

/// A built, ready-to-serve mock hopper. Walk the dataset once (hashing every
/// file), then [`serve`](BenchHopper::serve) it on a port.
#[derive(Debug)]
pub struct BenchHopper {
    jobs: Vec<Job>,
    by_path: HashMap<String, PathBuf>,
    /// sha256 → job indices (a dataset may contain duplicate content; each
    /// result is attributed to the first unresolved index for its sha).
    by_sha: HashMap<String, Vec<usize>>,
    order: Order,
    cursor: AtomicUsize,
    results: Arc<AtomicUsize>,
    timings: Mutex<Timings>,
    dump: Option<Arc<Mutex<File>>>,
    /// Directory each decompressed result body is written to as `<sha>.json`,
    /// for detection-fingerprint comparison between worker runs. Overwrites are
    /// fine: posts are idempotent, so identical shas carry identical bodies.
    dump_bodies: Option<PathBuf>,
    /// Summary JSON destination, rewritten after every result post so the file
    /// is always current no matter when the harness stops this process.
    summary: Option<PathBuf>,
}

/// A running hopper. Dropping it leaves the accept thread running (detached) —
/// fine for a short-lived benchmark process or a test that ends with process
/// teardown.
#[derive(Debug)]
pub struct Handle {
    /// The port the hopper bound (resolved if `serve(0)` was used).
    pub port: u16,
    /// Total number of jobs (dataset files) the hopper will hand out.
    pub jobs_total: usize,
    server: Arc<BenchHopper>,
}

impl Handle {
    /// Number of `/api/result` posts received so far.
    #[must_use]
    pub fn results_received(&self) -> usize {
        self.server.results.load(Ordering::Relaxed)
    }

    /// Render the current latency/throughput summary as JSON.
    #[must_use]
    pub fn summary_json(&self) -> String {
        self.server.summary_json()
    }
}

impl BenchHopper {
    /// Walk `dataset` recursively, hashing every file, and build the job set in
    /// the given handout `order`. `dump`, if set, is a file each posted result
    /// body is appended to; `summary`, if set, receives the latency summary
    /// JSON (rewritten after every result).
    pub fn build(
        dataset: &Path,
        dump: Option<PathBuf>,
        dump_bodies: Option<PathBuf>,
        order: Order,
        summary: Option<PathBuf>,
    ) -> io::Result<Self> {
        let mut files = Vec::new();
        collect_files(dataset, dataset, &mut files)?;
        files.sort_by(|a, b| a.0.cmp(&b.0)); // deterministic base order across runs

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
        order_jobs(&mut jobs, order);

        let mut by_sha: HashMap<String, Vec<usize>> = HashMap::with_capacity(jobs.len());
        for (i, job) in jobs.iter().enumerate() {
            by_sha.entry(job.sha256.clone()).or_default().push(i);
        }

        let dump = match dump {
            Some(path) => Some(Arc::new(Mutex::new(
                OpenOptions::new().create(true).append(true).open(path)?,
            ))),
            None => None,
        };
        if let Some(dir) = &dump_bodies {
            std::fs::create_dir_all(dir)?;
        }

        let timings = Mutex::new(Timings {
            claimed: vec![None; jobs.len()],
            resulted: vec![None; jobs.len()],
        });

        Ok(Self {
            jobs,
            by_path,
            by_sha,
            order,
            cursor: AtomicUsize::new(0),
            results: Arc::new(AtomicUsize::new(0)),
            timings,
            dump,
            dump_bodies,
            summary,
        })
    }

    /// Bind `port` (0 = ephemeral) and serve on a background accept thread.
    pub fn serve(self, port: u16) -> io::Result<Handle> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let bound = listener.local_addr()?.port();
        let jobs_total = self.jobs.len();
        let server = Arc::new(self);
        let accept_server = Arc::clone(&server);

        std::thread::Builder::new()
            .name("bench-hopper".into())
            .spawn(move || {
                for stream in listener.incoming() {
                    let Ok(stream) = stream else { continue };
                    let server = Arc::clone(&accept_server);
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
            server,
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
        let count = query_param(target, "count")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let start = self.cursor.fetch_add(count, Ordering::SeqCst);
        if start >= self.jobs.len() {
            // Drained: the real hopper returns 204 when no work remains.
            return respond_head(stream, "204 No Content", 0);
        }
        let end = (start + count).min(self.jobs.len());
        {
            let now = Instant::now();
            let mut t = self.timings.lock().unwrap_or_else(PoisonError::into_inner);
            for slot in &mut t.claimed[start..end] {
                *slot = Some(now);
            }
        }
        use std::fmt::Write;
        let mut body = String::from("{\"jobs\":[");
        for (i, job) in self.jobs[start..end].iter().enumerate() {
            if i > 0 {
                body.push(',');
            }
            // file_type left as "data"; cleave detects the real type from bytes.
            let _ = write!(
                body,
                "{{\"sha256\":\"{}\",\"path\":{},\"size_bytes\":{},\"file_type\":\"data\"}}",
                job.sha256,
                json_string(&job.path),
                job.size_bytes,
            );
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
        // The worker zstd-compresses result bodies (raw JSON on fallback).
        let json = zstd::decode_all(body).unwrap_or_else(|_| body.to_vec());
        let sha = first_sha256(&json).unwrap_or_else(|| "unknown".to_string());
        if let Some(dump) = &self.dump {
            // Record one clean line per result — the sample's top-level sha256 —
            // so completeness = distinct lines, not unparseable binary blobs.
            if let Ok(mut f) = dump.lock() {
                let _ = writeln!(f, "{sha}");
            }
        }
        if let Some(dir) = &self.dump_bodies {
            let _ = std::fs::write(dir.join(format!("{sha}.json")), &json);
        }
        self.record_result(&sha);
        if let Some(path) = &self.summary {
            // Rewrite the whole summary after every result: O(n log n) over a
            // few thousand jobs is nothing next to a multi-second analysis, and
            // it guarantees the file is complete whenever the harness reads it.
            let _ = std::fs::write(path, self.summary_json());
        }
        respond_body(stream, "200 OK", b"{}")
    }

    /// Stamp the result time on the first unresolved job index for `sha`.
    /// Duplicate-content datasets resolve one index per post; retries past the
    /// last index (or an unknown sha) are ignored.
    fn record_result(&self, sha: &str) {
        let Some(indices) = self.by_sha.get(sha) else {
            return;
        };
        let now = Instant::now();
        let mut t = self.timings.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(&i) = indices.iter().find(|&&i| t.resulted[i].is_none()) {
            t.resulted[i] = Some(now);
        }
    }

    /// Render the latency/throughput summary as JSON: wall over the job window
    /// (first claim → last result), throughput, and per-size-class sojourn
    /// percentiles. One class object per line, so the raw file reads as a table.
    fn summary_json(&self) -> String {
        // Snapshot under the lock, then format with it released.
        let t = self.timings.lock().unwrap_or_else(PoisonError::into_inner);

        let first_claim = t.claimed.iter().flatten().min().copied();
        let last_result = t.resulted.iter().flatten().max().copied();
        let wall_ms = match (first_claim, last_result) {
            (Some(a), Some(b)) => b.duration_since(a).as_millis(),
            _ => 0,
        };
        let done = t.resulted.iter().flatten().count();

        // Two latencies per finished job, bucketed by size class:
        //  - sojourn: claim → result (worker-internal queueing + analysis);
        //  - flow: first claim of the run → result. All bench jobs "arrive" at
        //    t0, so flow is the classic flow-time objective that size-aware
        //    scheduling optimizes — pre-claim queue time counts here, where the
        //    shallow prefetch depth hides it from sojourn.
        let mut classes: Vec<(&str, Vec<u128>, Vec<u128>)> = SIZE_CLASSES
            .iter()
            .map(|(name, _)| (*name, Vec::new(), Vec::new()))
            .collect();
        let (mut all, mut all_flow) = (Vec::with_capacity(done), Vec::with_capacity(done));
        for (i, job) in self.jobs.iter().enumerate() {
            let (Some(claim), Some(result)) = (t.claimed[i], t.resulted[i]) else {
                continue;
            };
            let ms = result.duration_since(claim).as_millis();
            let flow = first_claim.map_or(ms, |t0| result.duration_since(t0).as_millis());
            let class = &mut classes[size_class(job.size_bytes)];
            class.1.push(ms);
            class.2.push(flow);
            all.push(ms);
            all_flow.push(flow);
        }
        drop(t);
        #[allow(clippy::cast_precision_loss)]
        let throughput_jps = if wall_ms > 0 {
            done as f64 / (wall_ms as f64 / 1000.0)
        } else {
            0.0
        };

        let mut out = String::from("{\n");
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!(
                "  \"order\": \"{}\",\n  \"jobs\": {},\n  \"done\": {},\n  \
                 \"wall_ms\": {},\n  \"throughput_jps\": {:.3},\n  \"classes\": [\n",
                self.order,
                self.jobs.len(),
                done,
                wall_ms,
                throughput_jps,
            ),
        );
        let rows = classes
            .iter()
            .map(|(name, ms, flow)| (*name, ms, flow))
            .chain(std::iter::once(("all", &all, &all_flow)));
        let total = classes.len() + 1;
        for (i, (name, ms, flow)) in rows.enumerate() {
            out.push_str("    ");
            out.push_str(&class_row_json(name, ms, flow));
            out.push_str(if i + 1 < total { ",\n" } else { "\n" });
        }
        out.push_str("  ]\n}\n");
        out
    }
}

/// Sojourn size classes: small jobs are the starvation canary, large archives
/// the head-of-line blockers. Upper bounds in bytes; the last is open-ended.
const SIZE_CLASSES: [(&str, u64); 3] = [
    ("small", 1024 * 1024),       // < 1 MiB
    ("medium", 32 * 1024 * 1024), // 1–32 MiB
    ("large", u64::MAX),          // > 32 MiB
];

fn size_class(size_bytes: u64) -> usize {
    SIZE_CLASSES
        .iter()
        .position(|(_, max)| size_bytes < *max)
        .unwrap_or(SIZE_CLASSES.len() - 1)
}

/// One summary row: count plus sojourn (claim → result) and flow (run start →
/// result) mean/p50/p95/p99/max in milliseconds.
fn class_row_json(name: &str, ms: &[u128], flow: &[u128]) -> String {
    let stats = |vals: &[u128]| -> (u128, u128, u128, u128, u128) {
        let mut sorted = vals.to_vec();
        sorted.sort_unstable();
        let mean = if sorted.is_empty() {
            0
        } else {
            sorted.iter().sum::<u128>() / sorted.len() as u128
        };
        (
            mean,
            percentile(&sorted, 50),
            percentile(&sorted, 95),
            percentile(&sorted, 99),
            sorted.last().copied().unwrap_or(0),
        )
    };
    let (mean, p50, p95, p99, max) = stats(ms);
    let (fmean, fp50, fp95, fp99, fmax) = stats(flow);
    format!(
        "{{\"class\": \"{}\", \"count\": {}, \"mean_ms\": {}, \"p50_ms\": {}, \
         \"p95_ms\": {}, \"p99_ms\": {}, \"max_ms\": {}, \
         \"flow_mean_ms\": {}, \"flow_p50_ms\": {}, \"flow_p95_ms\": {}, \
         \"flow_p99_ms\": {}, \"flow_max_ms\": {}}}",
        name,
        ms.len(),
        mean,
        p50,
        p95,
        p99,
        max,
        fmean,
        fp50,
        fp95,
        fp99,
        fmax,
    )
}

/// Nearest-rank percentile over an already-sorted slice; 0 when empty.
fn percentile(sorted: &[u128], p: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (sorted.len() * p).div_ceil(100).max(1);
    sorted[rank - 1]
}

/// Reorder jobs for handout. `Fifo` keeps the path-sorted base order; the size
/// orders tie-break on path so runs stay deterministic.
fn order_jobs(jobs: &mut [Job], order: Order) {
    match order {
        Order::Fifo => {}
        Order::BigFirst => {
            jobs.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes).then(a.path.cmp(&b.path)));
        }
        Order::SmallFirst => {
            jobs.sort_by(|a, b| a.size_bytes.cmp(&b.size_bytes).then(a.path.cmp(&b.path)));
        }
        Order::Shuffle(seed) => {
            // Fisher–Yates over the path-sorted base, so a seed names exactly
            // one permutation regardless of filesystem enumeration order.
            let mut state = seed.max(1); // xorshift state must be non-zero
            for i in (1..jobs.len()).rev() {
                #[allow(clippy::cast_possible_truncation)]
                let j = (xorshift64(&mut state) % (i as u64 + 1)) as usize;
                jobs.swap(i, j);
            }
        }
        Order::Mixed => {
            // Sort by each job's fractional position within its size class —
            // the k-th of n jobs in a class lands at (2k+1)/2n of the stream —
            // so every class is spread evenly and every claim window sees a
            // mix. Compared as exact rationals (cross-multiplied u128s scaled
            // to a common grid) to keep ordering integer-deterministic;
            // original index tie-breaks.
            let mut counts = [0u128; SIZE_CLASSES.len()];
            for job in jobs.iter() {
                counts[size_class(job.size_bytes)] += 1;
            }
            let mut seen = [0u128; SIZE_CLASSES.len()];
            let mut keyed: Vec<(u128, Job)> = jobs
                .iter()
                .map(|job| {
                    let class = size_class(job.size_bytes);
                    let key = ((2 * seen[class] + 1) << 32) / (2 * counts[class]);
                    seen[class] += 1;
                    (key, job.clone())
                })
                .collect();
            keyed.sort_by_key(|kj| kj.0);
            for (slot, (_, job)) in jobs.iter_mut().zip(keyed) {
                *slot = job;
            }
        }
    }
}

/// Extract the result's top-level sample sha256 for the manifest line,
/// decompressing the (zstd) post body first and falling back gracefully.
/// Find the first `"sha256":"<64 lowercase hex>"` value in a JSON byte slice —
/// the ResultPayload's top-level hash; the embedded cleave report's per-member
/// hashes appear later in the body.
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
    use std::fmt::Write;
    let mut hex = String::with_capacity(64);
    for b in digest {
        let _ = write!(hex, "{b:02x}");
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
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "header too large",
            ));
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
            k.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| v.trim().parse().ok())
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
    let head =
        format!("HTTP/1.1 {status}\r\nContent-Length: {content_len}\r\nConnection: close\r\n\r\n");
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
    query.split('&').find_map(|kv| {
        kv.split_once('=')
            .filter(|(k, _)| *k == key)
            .map(|(_, v)| v)
    })
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
        let status = String::from_utf8_lossy(&resp[..split])
            .lines()
            .next()
            .unwrap()
            .to_string();
        (status, resp[split + 4..].to_vec())
    }

    fn post_result(port: u16, body: &[u8]) {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(
            stream,
            "POST /api/result HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len(),
        )
        .unwrap();
        stream.write_all(body).unwrap();
        let mut resp = Vec::new();
        stream.read_to_end(&mut resp).unwrap();
    }

    #[test]
    fn serves_jobs_then_drains_and_serves_data() {
        let dir = std::env::temp_dir().join(format!("bench_hopper_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_file(&dir, "a.txt", b"hello");
        write_file(&dir, "sub/b dir.txt", b"world!!");

        let hopper = BenchHopper::build(&dir, None, None, Order::Fifo, None).unwrap();
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
        let body =
            format!("{{\"sha256\":\"{h}\",\"worker\":\"w\",\"raw\":{{\"sha256\":\"{m}\"}}}}");
        assert_eq!(first_sha256(body.as_bytes()).as_deref(), Some(h.as_str()));
        assert_eq!(first_sha256(b"{\"worker\":\"w\"}"), None);
    }

    #[test]
    fn result_bodies_decompress_and_dump() {
        let h = "c".repeat(64);
        let json = format!("{{\"sha256\":\"{h}\",\"worker\":\"w\"}}");
        let compressed = zstd::encode_all(json.as_bytes(), 3).unwrap();

        let dir = std::env::temp_dir().join(format!("bench-hopper-bodies-{}", std::process::id()));
        let data = dir.join("data");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("f.txt"), b"x").unwrap();
        let bodies = dir.join("bodies");
        let hopper =
            BenchHopper::build(&data, None, Some(bodies.clone()), Order::Fifo, None).unwrap();
        let port = hopper.serve(0).unwrap().port;

        // Compressed and raw (uncompressed-fallback) bodies both land as
        // decompressed `<sha>.json`.
        post_result(port, &compressed);
        let dumped = std::fs::read(bodies.join(format!("{h}.json"))).unwrap();
        assert_eq!(dumped, json.as_bytes());
        post_result(port, json.as_bytes());
        let dumped = std::fs::read(bodies.join(format!("{h}.json"))).unwrap();
        assert_eq!(dumped, json.as_bytes());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn percent_decode_reverses_segment_encoding() {
        // Matches the worker's `url_encode_into`: unreserved chars pass through,
        // everything else is %XX (uppercase hex).
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("c%25d"), "c%d");
        assert_eq!(percent_decode("plain.txt"), "plain.txt");
    }

    fn job(path: &str, size: u64) -> Job {
        Job {
            sha256: format!("{:064}", size),
            path: path.to_string(),
            size_bytes: size,
        }
    }

    #[test]
    fn order_parses_and_round_trips() {
        assert_eq!("fifo".parse::<Order>().unwrap(), Order::Fifo);
        assert_eq!("shuffle".parse::<Order>().unwrap(), Order::Shuffle(42));
        assert_eq!("shuffle:7".parse::<Order>().unwrap(), Order::Shuffle(7));
        assert_eq!("big-first".parse::<Order>().unwrap(), Order::BigFirst);
        assert!("nope".parse::<Order>().is_err());
        assert!("shuffle:x".parse::<Order>().is_err());
        assert_eq!(Order::Shuffle(7).to_string(), "shuffle:7");
    }

    #[test]
    fn order_jobs_sorts_and_shuffles_deterministically() {
        let base = vec![job("a", 10), job("b", 30), job("c", 20)];

        let mut big = base.clone();
        order_jobs(&mut big, Order::BigFirst);
        assert_eq!(
            big.iter().map(|j| j.size_bytes).collect::<Vec<_>>(),
            [30, 20, 10]
        );

        let mut small = base.clone();
        order_jobs(&mut small, Order::SmallFirst);
        assert_eq!(
            small.iter().map(|j| j.size_bytes).collect::<Vec<_>>(),
            [10, 20, 30]
        );

        // Same seed → same permutation; different seed → (here) a different one.
        let (mut s1, mut s2, mut s3) = (base.clone(), base.clone(), base.clone());
        order_jobs(&mut s1, Order::Shuffle(1));
        order_jobs(&mut s2, Order::Shuffle(1));
        order_jobs(&mut s3, Order::Shuffle(2));
        let paths = |v: &[Job]| v.iter().map(|j| j.path.clone()).collect::<Vec<_>>();
        assert_eq!(paths(&s1), paths(&s2));
        assert_ne!(paths(&s1), paths(&s3));
    }

    #[test]
    fn mixed_spreads_each_size_class_evenly() {
        // 4 smalls + 1 large: the large lands mid-stream (position 1/2 falls
        // between the smalls at 1/8, 3/8 and 5/8, 7/8), not at either end.
        let big = 100 * 1024 * 1024;
        let mut jobs = vec![
            job("s1", 10),
            job("s2", 20),
            job("s3", 30),
            job("s4", 40),
            job("big", big),
        ];
        order_jobs(&mut jobs, Order::Mixed);
        let paths: Vec<&str> = jobs.iter().map(|j| j.path.as_str()).collect();
        assert_eq!(paths, ["s1", "s2", "big", "s3", "s4"]);

        // Round-trip parse for the CLI.
        assert_eq!("mixed".parse::<Order>().unwrap(), Order::Mixed);
        assert_eq!(Order::Mixed.to_string(), "mixed");
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let v: Vec<u128> = (1..=100).collect();
        assert_eq!(percentile(&v, 50), 50);
        assert_eq!(percentile(&v, 95), 95);
        assert_eq!(percentile(&v, 99), 99);
        assert_eq!(percentile(&[7], 99), 7);
        assert_eq!(percentile(&[], 50), 0);
    }

    #[test]
    fn size_class_buckets_by_bounds() {
        assert_eq!(size_class(0), 0); // small
        assert_eq!(size_class(1024 * 1024), 1); // exactly 1 MiB → medium
        assert_eq!(size_class(32 * 1024 * 1024), 2); // exactly 32 MiB → large
        assert_eq!(size_class(u64::MAX), 2);
    }

    #[test]
    fn summary_tracks_claim_to_result_sojourn() {
        let dir = std::env::temp_dir().join(format!("bench_hopper_sum_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_file(&dir, "x.bin", b"abc");
        write_file(&dir, "y.bin", b"defgh");

        let hopper = BenchHopper::build(&dir, None, None, Order::Fifo, None).unwrap();
        let shas: Vec<String> = hopper.jobs.iter().map(|j| j.sha256.clone()).collect();
        let handle = hopper.serve(0).unwrap();
        let port = handle.port;

        // Claim both jobs, then post a result for the first sha only.
        let (status, _) = get(port, "/api/next?count=10");
        assert!(status.contains("200"), "status: {status}");
        let body = format!("{{\"sha256\":\"{}\",\"worker\":\"w\"}}", shas[0]);
        post_result(port, body.as_bytes());

        let summary = handle.summary_json();
        assert!(summary.contains("\"jobs\": 2"), "summary: {summary}");
        assert!(summary.contains("\"done\": 1"), "summary: {summary}");
        // Both files are tiny → the one finished job lands in the small class.
        assert!(
            summary.contains("\"class\": \"small\", \"count\": 1"),
            "summary: {summary}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
