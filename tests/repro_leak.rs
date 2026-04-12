//! Regression test for active_tasks leak on panic.
#![allow(clippy::panic)] // the whole point of this test is to trigger a panic and assert no leak

use dashmap::DashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

// Mocking enough of AppState to test the logic
struct AppState {
    active_tasks: AtomicUsize,
    in_flight: DashMap<u64, InFlightRequest>,
}

struct InFlightRequest {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    size_bytes: u64,
    #[allow(dead_code)]
    started_at: Instant,
    #[allow(dead_code)]
    timed_out: AtomicBool,
}

struct TaskGuard {
    state: Arc<AppState>,
    request_id: u64,
}

impl TaskGuard {
    fn new(state: Arc<AppState>, request_id: u64, name: String, size_bytes: u64) -> Self {
        state.active_tasks.fetch_add(1, Ordering::SeqCst);
        state.in_flight.insert(
            request_id,
            InFlightRequest {
                name,
                size_bytes,
                started_at: Instant::now(),
                timed_out: AtomicBool::new(false),
            },
        );
        Self { state, request_id }
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        self.state.active_tasks.fetch_sub(1, Ordering::SeqCst);
        self.state.in_flight.remove(&self.request_id);
    }
}

/// Verifies that active_tasks and in_flight entries do not leak if a panic occurs.
#[tokio::test]
async fn test_active_tasks_no_leak_on_panic_with_guard() {
    let state = Arc::new(AppState {
        active_tasks: AtomicUsize::new(0),
        in_flight: DashMap::new(),
    });

    let request_id = 1;
    let guard = TaskGuard::new(Arc::clone(&state), request_id, "test".to_string(), 100);

    let handle = tokio::task::spawn_blocking(move || {
        let _moved_guard = guard;
        // Simulate a panic during analysis
        panic!("intended panic for reproduction");
    });

    let _ = handle.await;

    assert_eq!(
        state.active_tasks.load(Ordering::Relaxed),
        0,
        "active_tasks leaked despite guard!"
    );
    assert_eq!(
        state.in_flight.len(),
        0,
        "in_flight entry leaked despite guard!"
    );
}
