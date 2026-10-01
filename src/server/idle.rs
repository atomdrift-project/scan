//! The pull worker the server runs beside itself, and when it may run.
//!
//! Spare capacity is worth using and worth nothing at all next to a request, so
//! the worker is an ordinary `atomscan worker` in its own process that the
//! server stops dead whenever it is busy. See [`crate::suspend`] for why a
//! separate process is what makes that a guarantee rather than a hope.
//!
//! Two consequences are worth stating, because they are why this is simpler
//! than running the worker inside the server:
//!
//!   - **No sizing of its own.** A standalone worker already sizes itself
//!     across every host in the fleet — slots at three per physical core, the
//!     rayon pool at the core count, the RSS ceiling auto-resolved from the
//!     cgroup or sysctl basis. It gets the whole box, because freezing returns
//!     the whole box instantly; a standing reserve would buy nothing.
//!   - **No configuration of its own.** The child inherits the server's
//!     environment, which is where the hopper token, the LLM endpoint and the
//!     traits directory already live. Only what must differ is passed.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};

use tokio::sync::watch;

use crate::suspend::Idle;

/// How long the supervisor waits before replacing a worker that exited, and
/// between attempts when spawning keeps failing. One cadence rather than a
/// backoff curve: the failures worth surviving here are a transient OOM kill
/// and a bad deploy, and neither is helped by trying harder or by giving up.
const RESTART_DELAY: std::time::Duration = std::time::Duration::from_secs(10);

/// How far the worker is pushed up the OOM killer's list, so that when the
/// pair reaches the cgroup's `MemoryMax` the kernel takes the worker and never
/// the server. This is the whole memory policy: no threshold to tune, no
/// watcher, and the decision is made by the only party that sees the real
/// numbers.
#[cfg(target_os = "linux")]
const IDLE_OOM_SCORE_ADJ: i32 = 500;

/// Requests in flight, counted so the worker is frozen for exactly as long as
/// the server has something to do.
///
/// It is raised twice per request — once when the handler is entered, once
/// when the analysis itself begins — and the overlap is the point: the handler
/// covers the upload, which can be seconds of streaming before an analysis
/// exists, and the analysis covers the work, which outlives the handler when
/// the client hangs up. Neither alone spans the request.
///
/// A count under a lock rather than an atomic: the freeze or thaw a transition
/// causes is made while the lock is held, so a 1→0 thaw and the 0→1 freeze
/// racing it are applied in the order the count changed, and the worker never
/// runs beside a request.
#[derive(Debug, Default)]
pub(super) struct Busy {
    count: Mutex<usize>,
}

impl Busy {
    fn lock(&self) -> std::sync::MutexGuard<'_, usize> {
        self.count.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Mark the server busy until the returned token is dropped, freezing the
    /// worker on the transition into busy and thawing on the way out.
    ///
    /// A freeze that fails is reported and not retried: the worker is nice'd
    /// well below the server and claims none of its permits, so a server that
    /// cannot freeze is degraded, not broken, and saying so once per failure is
    /// more useful than a loop.
    pub(super) fn enter(self: &Arc<Self>, worker: Option<&Arc<Worker>>) -> BusyToken {
        let mut count = self.lock();
        *count += 1;
        if *count == 1
            && let Some(worker) = worker
            && let Err(e) = worker.freeze()
        {
            tracing::warn!(error = %e, "could not freeze the idle worker; it keeps running beside this request");
        }
        drop(count);
        BusyToken {
            busy: Arc::clone(self),
            worker: worker.map(Arc::clone),
        }
    }

    /// Whether the server is currently busy. Published on `/_/stats`.
    pub(super) fn is_busy(&self) -> bool {
        *self.lock() > 0
    }
}

/// Raised while one request — or one analysis — is outstanding.
#[derive(Debug)]
pub(super) struct BusyToken {
    busy: Arc<Busy>,
    worker: Option<Arc<Worker>>,
}

impl Drop for BusyToken {
    fn drop(&mut self) {
        let mut count = self.busy.lock();
        *count -= 1;
        if *count == 0
            && let Some(worker) = &self.worker
            && let Err(e) = worker.thaw()
        {
            // Worse than a failed freeze: the worker stays stopped and the
            // queue work silently stops. Named at error level so it is visible
            // without reading a throughput graph.
            tracing::error!(error = %e, "could not thaw the idle worker; it stays frozen until the next request");
        }
    }
}

/// Build the command for the companion worker.
///
/// `hopper` is the primary address only. Claiming is a worker route and a
/// replica answers those with 403, so the comma list the server takes for
/// lookups and renewals cannot be passed through here.
///
/// `traits_dir` is the server's own `--traits-dir`, and it has to be passed
/// explicitly: the server applies it as an in-process override
/// (`cleave::traits_repo::set_override_dir`), not an environment variable, so a
/// child inherits nothing. Without it the worker re-resolves traits on its own
/// and reaches the same tree only if its working directory happens to hold a
/// `traits/` — the version it reports on every result then depends on that
/// accident rather than on the deploy.
fn worker_command(
    hopper: &str,
    name: &str,
    cache_dir: &PathBuf,
    traits_dir: Option<&std::path::Path>,
) -> std::io::Result<Command> {
    let mut cmd = Command::new(std::env::current_exe()?);
    cmd.arg("worker")
        .arg("--url")
        .arg(hopper)
        .arg("--name")
        .arg(name);
    if let Some(dir) = traits_dir {
        cmd.arg("--traits-dir").arg(dir);
    }
    cmd
        // The server owns updating. Two processes refreshing the same traits
        // checkout and models directory would race each other over a git tree,
        // and the worker's copy is the server's copy. The embedded worker set
        // this too, for the same reason.
        .arg("--no-update")
        // Skip the startup fixture validation: 143 benign targets scanned
        // before the first claim. Worth paying once for a worker an operator
        // started; wasteful for one a supervisor may restart, and the server
        // beside it has already validated these very models and traits.
        .arg("--no-validate")
        // Its own cache root. cleave's analysis cache is SQLite in WAL mode
        // with a single writer and a five-second busy timeout, and its store
        // runs synchronously before an analysis returns — so a freeze landing
        // while the worker held the write lock would stall the *server's* next
        // store for that timeout, on the response path. Splitting the root is
        // the blunt form of the fix and needs nothing from cleave; the narrow
        // one splits only the database and keeps fletch's download cache
        // shared, which is worth having because the server fetches the same
        // artifacts the worker has already pulled.
        .env("XDG_CACHE_HOME", cache_dir)
        // Everything else — the hopper token, the LLM endpoint, the models
        // directory, the open-file limit — is inherited.
        .stdin(std::process::Stdio::null());

    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: runs between fork and exec. One `write(2)` to a proc file
        // with a stack buffer; no allocation.
        unsafe {
            cmd.pre_exec(|| {
                let mut buf = [0u8; 8];
                let text = {
                    use std::io::Write;
                    let mut slice = &mut buf[..];
                    write!(slice, "{IDLE_OOM_SCORE_ADJ}").map_err(|_| {
                        std::io::Error::new(std::io::ErrorKind::InvalidInput, "oom_score_adj")
                    })?;
                    let used = 8 - slice.len();
                    &buf[..used]
                };
                // Best effort: a kernel that refuses this leaves the worker at
                // the server's score, which is the behaviour before this
                // existed and not worth failing the spawn over.
                if let Ok(mut f) = std::fs::File::create("/proc/self/oom_score_adj") {
                    use std::io::Write;
                    let _ = f.write_all(text);
                }
                Ok(())
            });
        }
    }
    Ok(cmd)
}

/// The worker's private cache root: a sibling of the server's own, so both are
/// swept by the same retention and neither can be mistaken for the other.
fn cache_dir() -> Option<PathBuf> {
    Some(dirs::cache_dir()?.join("atomdrift").join("scan-idle"))
}

/// The companion worker and the supervision that keeps one running.
///
/// The child is replaceable, so it lives behind a lock rather than in the
/// `Arc` the request path holds: a worker that is OOM-killed or crashes must
/// come back without the server restarting, and every `freeze`/`thaw` call
/// site must keep working across that swap without knowing it happened.
#[derive(Debug)]
pub(super) struct Worker {
    hopper: String,
    name: String,
    cache: PathBuf,
    /// `None` only between a worker exiting and its replacement starting.
    current: std::sync::RwLock<Option<Idle>>,
}

impl Worker {
    /// Whether a worker process is alive right now.
    pub(super) fn is_running(&self) -> bool {
        self.current
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    /// Suspend the worker, if one is running right now.
    fn freeze(&self) -> std::io::Result<()> {
        let current = self.current.read().unwrap_or_else(PoisonError::into_inner);
        current.as_ref().map_or(Ok(()), Idle::freeze)
    }

    /// Resume the worker, if one is running right now.
    fn thaw(&self) -> std::io::Result<()> {
        let current = self.current.read().unwrap_or_else(PoisonError::into_inner);
        current.as_ref().map_or(Ok(()), Idle::thaw)
    }

    /// Start a worker if none is running, replacing one that exited. Blocking:
    /// it reaps, kills and spawns processes, so it runs on a blocking thread,
    /// and none of that happens under the lock the request path reads.
    fn ensure_running(&self, busy: &Busy) {
        let exited = {
            let mut current = self.current.write().unwrap_or_else(PoisonError::into_inner);
            match current.as_mut().map(Idle::exited) {
                Some(Ok(None)) => return,
                Some(Ok(Some(status))) => {
                    tracing::warn!(%status, worker = %self.name, "idle worker exited");
                }
                Some(Err(e)) => {
                    tracing::warn!(error = %e, "could not check on the idle worker");
                    return;
                }
                // Nothing running: the first start, or a spawn that failed.
                None => {}
            }
            current.take()
        };
        // Retire the outgoing worker *before* starting its replacement. Both
        // live in the same control group, and the primitives that make this
        // design a guarantee act on membership rather than on a pid: the old
        // worker's `Drop` kills its whole cgroup, so a replacement started
        // first is simply killed along with it. Measured on galadriel
        // (2026-09-15): the supervisor restarted, logged the new pid, and the
        // new worker was dead before its first poll — then looped, forever.
        drop(exited);

        let traits_dir = cleave::traits_repo::override_dir();
        let spawned = worker_command(&self.hopper, &self.name, &self.cache, traits_dir.as_deref())
            .and_then(Idle::spawn);
        let started = match spawned {
            Ok(worker) => {
                tracing::info!(pid = worker.pid(), worker = %self.name, "idle worker started");
                worker
            }
            Err(e) => {
                tracing::error!(error = %e, worker = %self.name, "could not start the idle worker; retrying");
                return;
            }
        };
        // Installed under the busy lock, so no request can enter or leave
        // between the freeze decision and the install.
        let count = busy.lock();
        let frozen = if *count > 0 { started.freeze() } else { Ok(()) };
        *self.current.write().unwrap_or_else(PoisonError::into_inner) = Some(started);
        drop(count);
        if let Err(e) = frozen {
            tracing::warn!(error = %e, "could not freeze a freshly started idle worker");
        }
    }

    /// Keep a worker running until shutdown, checking on it every
    /// [`RESTART_DELAY`].
    ///
    /// Polling rather than waiting on the child: the server has no SIGCHLD
    /// handler of its own to hang this off, and adding one would reach every
    /// subprocess the analysis path spawns.
    fn supervise(self: Arc<Self>, busy: Arc<Busy>, mut stop: watch::Receiver<bool>) {
        tokio::spawn(async move {
            loop {
                let (worker, busy) = (Arc::clone(&self), Arc::clone(&busy));
                if tokio::task::spawn_blocking(move || worker.ensure_running(&busy))
                    .await
                    .is_err()
                {
                    tracing::error!(
                        "idle worker supervisor panicked; the worker is no longer supervised"
                    );
                    return;
                }
                tokio::select! {
                    () = tokio::time::sleep(RESTART_DELAY) => {}
                    // Shutdown raised, or the server state is gone.
                    _ = stop.changed() => return,
                }
            }
        });
    }
}

/// Start the companion worker, or say why not.
///
/// Returns `None` for every deliberate reason not to run one, each named: the
/// three that were previously silent cost an afternoon apiece to diagnose,
/// because a worker that never started looks exactly like one with nothing to
/// do. The worker itself starts on the supervisor's first pass, off the
/// reactor.
pub(super) fn start(
    hopper: Option<&str>,
    name: &str,
    busy: &Arc<Busy>,
    stop: watch::Receiver<bool>,
) -> Option<Arc<Worker>> {
    let Some(hopper) = hopper else {
        tracing::info!("idle worker disabled: no --hopper to claim work from");
        return None;
    };
    let Some(primary) = crate::upload::worker_endpoint(hopper) else {
        tracing::info!("idle worker disabled: --hopper names no address");
        return None;
    };
    let Some(cache) = cache_dir() else {
        tracing::warn!(
            "idle worker disabled: no writable cache directory to give it, and \
             sharing the server's would let a freeze stall a request on cleave's \
             analysis-cache write lock",
        );
        return None;
    };
    if let Err(e) = std::fs::create_dir_all(&cache) {
        tracing::warn!(error = %e, path = %cache.display(), "idle worker disabled: cannot create its cache directory");
        return None;
    }

    let worker = Arc::new(Worker {
        hopper: primary,
        name: format!("{name}-idle"),
        cache,
        current: std::sync::RwLock::new(None),
    });
    tracing::info!(
        hopper = %worker.hopper,
        worker = %worker.name,
        cache = %worker.cache.display(),
        "idle worker: filling spare capacity with hopper queue work, frozen \
         whenever a request is in flight",
    );
    Arc::clone(&worker).supervise(Arc::clone(busy), stop);
    Some(worker)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_freezes_once_and_thaws_once() {
        // No worker attached: the counter is still the thing under test, and it
        // must reach zero exactly when the last holder goes away.
        let busy = Arc::new(Busy::default());
        assert!(!busy.is_busy());

        let first = busy.enter(None);
        assert!(busy.is_busy());
        let second = busy.enter(None);
        assert!(busy.is_busy(), "two holders, still busy");

        drop(first);
        assert!(
            busy.is_busy(),
            "the handler returning must not thaw while the analysis runs",
        );
        drop(second);
        assert!(!busy.is_busy(), "the last holder leaving ends the window");
    }

    #[test]
    fn the_command_passes_only_what_must_differ() {
        let dir = PathBuf::from("/tmp/scan-idle-test");
        let cmd =
            worker_command("http://hopper.example", "host-idle", &dir, None).expect("command");
        let args: Vec<_> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert_eq!(
            args,
            [
                "worker",
                "--url",
                "http://hopper.example",
                "--name",
                "host-idle",
                "--no-update",
                "--no-validate"
            ],
            "sizing and tuning belong to the worker's own defaults",
        );
        let cache = cmd
            .get_envs()
            .find(|(k, _)| *k == std::ffi::OsStr::new("XDG_CACHE_HOME"))
            .and_then(|(_, v)| v)
            .expect("the worker must not share the server's analysis cache");
        assert_eq!(cache, dir.as_os_str());
    }

    /// The server's `--traits-dir` is an in-process override, invisible to a
    /// child, so it must be on the command line. Measured on uruk-hai
    /// (2026-09-25): the server ran `--traits-dir …/traits` and its idle worker
    /// was spawned without it.
    #[test]
    fn the_command_forwards_the_servers_traits_dir() {
        let dir = PathBuf::from("/tmp/scan-idle-test");
        let traits = PathBuf::from("/var/lib/atomdrift/scan/traits");
        let cmd = worker_command("http://hopper.example", "host-idle", &dir, Some(&traits))
            .expect("command");
        let args: Vec<_> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        let at = args
            .iter()
            .position(|a| a == "--traits-dir")
            .expect("the traits dir must be forwarded");
        assert_eq!(
            args.get(at + 1).map(String::as_str),
            Some("/var/lib/atomdrift/scan/traits")
        );
    }
}
