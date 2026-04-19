//! End-to-end regression test for the YARA-init-on-rayon-worker deadlock.
//!
//! Companion to `cleave/tests/yara_init_no_deadlock.rs`. That test exercises
//! cleave's init contract in isolation; this one verifies that `litmus scan`
//! actually satisfies the contract when invoked as a real binary.
//!
//! The bug in production: litmus's worker/scan/server entry points used to
//! trigger YARA initialization from a rayon worker under cold-cache load,
//! which deadlocked the pool. The fix is a one-line `prefetch_shared_resources`
//! call at the top of each entry point. If someone removes that call from
//! `run_scan_paths`, the binary will hang under `CLEAVE_SKIP_YARA_CACHE=1`
//! and this test will trip its timeout.
//!
//! **Requires `LITMUS_MODELS_DIR` to be set** (same convention as
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
    let Ok(models_dir) = std::env::var("LITMUS_MODELS_DIR") else {
        eprintln!(
            "skipping: LITMUS_MODELS_DIR not set (same convention as \
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

    let bin = env!("CARGO_BIN_EXE_litmus");

    let mut child = Command::new(bin)
        .arg("scan")
        .arg(&sample_path)
        // Force the cold-compile code path that used to deadlock. With the
        // prefetch wired correctly this still completes (compile happens on
        // a non-rayon thread before any rayon analysis fires). Without it,
        // the process hangs.
        .env("CLEAVE_SKIP_YARA_CACHE", "1")
        .env("LITMUS_MODELS_DIR", &models_dir)
        // Silence scanning output — a regression test cares about exit
        // status and wall-clock, not which verdicts were printed.
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
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
                        "litmus scan did not exit within {:?} — \
                         likely YARA-init deadlock (check that every entry \
                         point calls cleave::prefetch_shared_resources before \
                         any rayon analysis)",
                        SCAN_TIMEOUT,
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
         elapsed {:?}",
        start.elapsed(),
    );
}
