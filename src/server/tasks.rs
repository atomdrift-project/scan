//! Background work the server owns, counted so shutdown can drain it.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Notify;

/// Work the server owns but no request waits on: counted so shutdown can wait
/// for it, where a bare `tokio::spawn` would be dropped mid-write.
#[derive(Default)]
pub(super) struct Tasks {
    live: AtomicUsize,
    idle: Notify,
}

impl Tasks {
    pub(super) fn spawn(self: &Arc<Self>, task: impl Future<Output = ()> + Send + 'static) {
        self.live.fetch_add(1, Ordering::AcqRel);
        let done = TaskDone(Arc::clone(self));
        tokio::spawn(async move {
            // Dropped on completion and on cancellation alike.
            let _done = done;
            task.await;
        });
    }

    pub(super) fn live(&self) -> usize {
        self.live.load(Ordering::Acquire)
    }

    /// Resolves once no task is running.
    pub(super) async fn drained(&self) {
        loop {
            // Registered before the count is read, so a task finishing in
            // between still wakes this waiter.
            let notified = self.idle.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            if self.live() == 0 {
                return;
            }
            notified.await;
        }
    }
}

struct TaskDone(Arc<Tasks>);

impl Drop for TaskDone {
    fn drop(&mut self) {
        if self.0.live.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.idle.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tasks;
    use std::sync::Arc;

    /// Drain waits for every owned task, including one that finishes after
    /// the drain began — the wakeup must not be lost between the count and
    /// the wait.
    #[tokio::test]
    async fn drain_waits_for_owned_work() {
        let tasks = Arc::new(Tasks::default());
        tasks.drained().await;

        let (go, wait) = tokio::sync::oneshot::channel::<()>();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&done);
        tasks.spawn(async move {
            let _ = wait.await;
            flag.store(true, std::sync::atomic::Ordering::Release);
        });
        assert_eq!(tasks.live(), 1);
        let _ = go.send(());
        tasks.drained().await;
        assert!(done.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(tasks.live(), 0);
    }
}
