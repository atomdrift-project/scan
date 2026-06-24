//! End-to-end regression test for the YARA-init-on-rayon-worker deadlock.
//!
//! Companion to `cleave/tests/yara_init_no_deadlock.rs`. That test exercises
//! cleave's init contract in isolation; this one verifies that `litmus scan`
//! actually satisfies the contract when invoked as a real binary.
//!
//! The bug in production: litmus's worker/scan/server entry points used to
//! trigger YARA initialization from a rayon worker under cold-cache load,
//! which deadlocked the pool. Scan still prefetches shared cleave resources so
//! cold-start work happens before analysis, and cleave's YARA loader is safe if
//! a new entry point misses that prefetch.
//!
//! **Requires `SCAN_MODELS_DIR` to be set** (same convention as
//! `server_analyze.rs`).

#![allow(clippy::expect_used, clippy::panic)]

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Wall-clock ceiling for `litmus scan`. A healthy run completes in a few
/// seconds. Cold YARA compile adds ~10-30s. Anything approaching this bound
/// means the binary is wedged, not slow.
const SCAN_TIMEOUT: Duration = Duration::from_secs(180);

/// Litmus exit codes: 0=clean, 1=hostile, 2=suspicious, 3=errors. Any of
/// 0/1/2 means the scan ran to completion — we only care about exit 3 or
/// a timeout, both of which indicate the binary failed or hung. We don't
/// assert a specific verdict because that's a model-correctness concern,
/// not a deadlock regression concern.
fn scan_ran_to_completion(code: i32) -> bool {
    (0..=2).contains(&code)
}

#[test]
fn scan_cli_completes_under_cold_yara_cache() {
    let Ok(models_dir) = std::env::var("SCAN_MODELS_DIR") else {
        eprintln!(
            "skipping: SCAN_MODELS_DIR not set (same convention as \
             server_analyze.rs integration tests)"
        );
        return;
    };

    // Trivial scan target — we don't care what cleave decides, only that the
    // binary exits cleanly. Use a named temp file so the path survives the
    // child invocation.
    let mut sample = tempfile::Builder::new()
        .prefix("litmus-scan-deadlock-")
        .suffix(".txt")
        .tempfile()
        .expect("create sample file");
    sample
        .write_all(b"hello world, this is a test file for the deadlock regression test\n")
        .expect("write sample");
    let sample_path = sample.path().to_path_buf();

    let bin = env!("CARGO_BIN_EXE_ascan");

    // Capture the child's stderr to a FILE, not a pipe. An undrained pipe can
    // fill its OS buffer (~64KB) and block the child on write — which would
    // look exactly like the hang we're hunting (a false positive). A file has
    // no such limit, and lets us dump everything the child logged if it wedges
    // or exits badly. `--verbose` raises litmus to `litmus=debug,cleave=debug`
    // (a plain `scan` logs at warn/error and prints almost nothing), so the
    // capture actually shows how far init/analysis got before stalling.
    let stderr_file = tempfile::NamedTempFile::new().expect("create stderr capture");
    let stderr_path = stderr_file.path().to_path_buf();
    let child_stderr = stderr_file.reopen().expect("reopen stderr capture");
    let captured = || std::fs::read_to_string(&stderr_path).unwrap_or_default();

    let mut child = Command::new(bin)
        .arg("--verbose")
        .arg("scan")
        .arg(&sample_path)
        // Force the cold-compile code path that used to deadlock. With the
        // prefetch wired correctly this still completes (compile happens on
        // a non-rayon thread before any rayon analysis fires). Without it,
        // the process hangs.
        .env("CLEAVE_SKIP_YARA_CACHE", "1")
        .env("SCAN_MODELS_DIR", &models_dir)
        .env("RUST_BACKTRACE", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::from(child_stderr))
        .spawn()
        .expect("spawn litmus");

    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if start.elapsed() > SCAN_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!(
                        "litmus scan did not exit within {:?} — process wedged. \
                         Captured child stderr (litmus=debug,cleave=debug) below; \
                         if it ends mid-init the last line shows where it stalled.\
                         \n=== captured child stderr ===\n{}\n=== end captured stderr ===",
                        SCAN_TIMEOUT,
                        captured(),
                    );
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => panic!("try_wait failed: {e}"),
        }
    };

    let code = status.code().unwrap_or(-1);
    assert!(
        scan_ran_to_completion(code),
        "litmus scan exited with code {code} (expected 0/1/2 — any verdict \
         is fine, we only reject error code 3 or termination by signal); \
         elapsed {:?}\n=== captured child stderr ===\n{}\n=== end captured stderr ===",
        start.elapsed(),
        captured(),
    );
}
