//! Integration regression: a worker with nothing to do must stay up.
//!
//! `run()` spawns the prefetcher and the worker tasks, then waits for shutdown
//! before draining. An earlier refactor dropped the wait, so the drain's
//! `SHUTDOWN_DRAIN_SECS` cap started ticking at startup and every worker
//! announced `drain timeout reached` 15 s in, abandoning whatever was in flight.
//! Nothing here asks the worker to stop, so `run()` must simply not return.
//!
//! **Requires `SCAN_MODELS_DIR`**, so it is ignored by default:
//! `SCAN_MODELS_DIR=... cargo test --test worker_idle_lifetime -- --ignored`.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod worker_support;

use std::time::Duration;

use scan::worker::run;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Must exceed `worker::SHUTDOWN_DRAIN_SECS` (15) by enough that a slow model
/// load can't mask the regression this guards.
const WATCH: Duration = Duration::from_secs(25);

/// Answer every request with "no work" so the worker sits fully idle.
async fn handle_conn(mut stream: TcpStream) {
    let mut buf = [0u8; 4096];
    if stream.read(&mut buf).await.unwrap_or(0) == 0 {
        return;
    }
    let _ = stream
        .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        .await;
    let _ = stream.flush().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a model bundle in SCAN_MODELS_DIR; run with --ignored"]
async fn idle_worker_outlives_the_shutdown_drain_window() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock hopper");
    let addr = listener.local_addr().expect("local addr");
    let hopper = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            tokio::spawn(handle_conn(stream));
        }
    });

    // Long-running mode: only a signal ends this worker, and the test sends none.
    let config =
        worker_support::worker_config("idle-lifetime-regression", format!("http://{addr}"), None);

    let mut worker = tokio::spawn(run(config));
    let outcome = tokio::time::timeout(WATCH, &mut worker).await;
    assert!(
        outcome.is_err(),
        "worker returned after {}s with no shutdown signal (drain window armed at startup?): {:?}",
        WATCH.as_secs(),
        outcome,
    );

    worker.abort();
    hopper.abort();
}
