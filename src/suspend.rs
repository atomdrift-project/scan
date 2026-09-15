//! Stopping the idle worker dead, so an arriving request gets the whole machine.
//!
//! The server fills spare capacity by running a pull worker beside itself. That
//! work is worth nothing next to an analyze request, so when a request arrives
//! every core it holds must come back — not eventually, and not politely.
//!
//! Cooperative cancellation cannot do this. Cleave polls a cancellation flag
//! between members, so the flag is only read as often as the shortest member
//! takes; and the expensive work is not cleave's at all but the children it
//! spawns. A stripped ELF's `rizin aaa` pass runs 10-30 s single-threaded (see
//! `lower_pool_thread_priority` in main.rs) inside a blocking `Command::output`
//! that no flag reaches. Asking nicely has a tail measured in tens of seconds,
//! which is the whole latency budget.
//!
//! So the kernel does it. Each platform has one primitive that acts on
//! *membership* rather than on a process group, which is what makes it a
//! guarantee: nothing escapes by calling `setsid`, including a third-party
//! analyzer whose process behaviour is not our contract.
//!
//!   - **Linux**: a cgroup v2 subtree. `cgroup.freeze` suspends every process
//!     in it; `cgroup.kill` kills every process in it.
//!   - **FreeBSD**: the reaper API. The child acquires reaper status for its
//!     own subtree before exec, and `PROC_REAP_KILL` delivers a signal to
//!     every descendant however they have rearranged their process groups.
//!
//! Everything else (macOS, for development) falls back to signalling the
//! process group. That is best-effort and deliberately not the production
//! story; the fleet is Linux and FreeBSD.
//!
//! Freezing is preferred to killing because it costs nothing: `SIGCONT` resumes
//! a half-finished analysis where it stopped, so the latency win does not buy
//! itself with wasted work and a redispatch from hopper. Killing is for a
//! worker that has stopped making progress.

use std::io;
use std::process::{Child, Command};

/// A child process the server can stop dead and start again.
///
/// Dropping it kills the process, reaps it, and releases whatever the platform
/// needed to control it — a server that exits must not leave a full-throttle
/// worker behind on the box.
#[derive(Debug)]
pub struct Idle {
    child: Child,
    control: imp::Control,
}

impl Idle {
    /// Spawn `cmd` under this platform's containment primitive.
    ///
    /// The containment is established *before* exec, so there is no window in
    /// which the child could fork a grandchild outside it.
    pub fn spawn(mut cmd: Command) -> io::Result<Self> {
        let control = imp::prepare(&mut cmd)?;
        match cmd.spawn() {
            Ok(child) => Ok(Self { child, control }),
            Err(e) => {
                imp::release(&control);
                Err(e)
            }
        }
    }

    /// Process id of the worker itself (not of its descendants).
    #[must_use]
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Suspend the worker and every descendant. Returns once the kernel has
    /// accepted the request; no core is running its work after that.
    pub fn freeze(&self) -> io::Result<()> {
        imp::freeze(&self.control, self.child.id())
    }

    /// Resume everything [`Idle::freeze`] suspended.
    pub fn thaw(&self) -> io::Result<()> {
        imp::thaw(&self.control, self.child.id())
    }

    /// Kill the worker and every descendant. For a worker that has stopped
    /// making progress; the request path uses [`Idle::freeze`].
    pub fn kill(&self) -> io::Result<()> {
        imp::kill(&self.control, self.child.id())
    }

    /// The worker's exit status if it has already exited, `None` while it runs.
    ///
    /// Also reaps it, so a worker that died on its own does not linger as a
    /// zombie until the server exits.
    pub fn exited(&mut self) -> io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }
}

impl Drop for Idle {
    fn drop(&mut self) {
        // Kill before thaw-less reaping: a frozen process never exits, so
        // waiting on one without killing it first would block forever. On
        // Linux `cgroup.kill` reaches a frozen cgroup; on FreeBSD SIGKILL is
        // delivered to a stopped process without needing SIGCONT first.
        let _ = imp::kill(&self.control, self.child.id());
        let _ = self.child.wait();
        imp::release(&self.control);
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::io;
    use std::fs;
    use std::os::unix::io::AsRawFd;
    use std::os::unix::process::CommandExt;
    use std::path::PathBuf;
    use std::process::Command;

    /// The cgroup the worker lives in: a child of the server's own.
    #[derive(Debug)]
    pub(super) struct Control {
        dir: PathBuf,
    }

    /// Create the worker's cgroup and arrange for the child to join it before
    /// exec.
    ///
    /// Only core interface files are used — `cgroup.freeze`, `cgroup.kill`,
    /// `cgroup.procs` — and no controller is enabled in the subtree. That is
    /// deliberate: enabling a controller would make the server's own cgroup an
    /// inner node, and cgroup v2 forbids an inner node from holding processes,
    /// so the server would have to migrate itself into a leaf first. Sticking
    /// to core files avoids the whole question. Memory is bounded elsewhere —
    /// by the server's `MemoryMax` (which already covers both processes) plus
    /// a high `oom_score_adj` on the child, so the kernel picks the worker as
    /// its victim and never the server.
    pub(super) fn prepare(cmd: &mut Command) -> io::Result<Control> {
        let dir = own_cgroup()?.join("idle");
        match fs::create_dir(&dir) {
            Ok(()) => {}
            // A previous worker's cgroup, left by a server that did not exit
            // cleanly. Reusing it is correct: it is empty, or holds processes
            // we are about to replace.
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }

        // Opened here, in the parent, and written by the child between fork and
        // exec. Writing it from the parent after `spawn` would leave a window —
        // short, but real — in which the child is outside the cgroup, and a
        // grandchild forked in that window would never be frozen with the rest.
        // "0" means the calling process.
        let procs = fs::OpenOptions::new()
            .write(true)
            .open(dir.join("cgroup.procs"))?;

        // SAFETY: the closure runs between fork and exec, where only
        // async-signal-safe work is permitted. It performs one `write(2)` on a
        // descriptor opened before the fork and allocates nothing.
        unsafe {
            cmd.pre_exec(move || {
                let fd = procs.as_raw_fd();
                let buf = b"0\n";
                if libc::write(fd, buf.as_ptr().cast(), buf.len()) < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        Ok(Control { dir })
    }

    pub(super) fn freeze(control: &Control, _pid: u32) -> io::Result<()> {
        fs::write(control.dir.join("cgroup.freeze"), b"1")
    }

    pub(super) fn thaw(control: &Control, _pid: u32) -> io::Result<()> {
        fs::write(control.dir.join("cgroup.freeze"), b"0")
    }

    /// `cgroup.kill` where the kernel has it (5.14+), else every pid in the
    /// cgroup individually. The fallback is still membership-based, so it keeps
    /// the guarantee that a process group would not.
    pub(super) fn kill(control: &Control, _pid: u32) -> io::Result<()> {
        match fs::write(control.dir.join("cgroup.kill"), b"1") {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => kill_each(control),
            Err(e) => Err(e),
        }
    }

    fn kill_each(control: &Control) -> io::Result<()> {
        // A frozen cgroup will not run the handlers, but SIGKILL does not need
        // it to: the kernel reaps a stopped process on SIGKILL.
        let procs = fs::read_to_string(control.dir.join("cgroup.procs"))?;
        for pid in procs.lines().filter_map(|l| l.trim().parse::<i32>().ok()) {
            // SAFETY: `kill(2)` on a pid read from this cgroup; no memory
            // effects, and a failure means the process already exited.
            unsafe { libc::kill(pid, libc::SIGKILL) };
        }
        Ok(())
    }

    /// Remove the worker's cgroup. Fails harmlessly while it still holds
    /// processes, which is why every caller kills and reaps first.
    pub(super) fn release(control: &Control) {
        let _ = fs::remove_dir(&control.dir);
    }

    /// This process's cgroup v2 directory, from the `0::` line of
    /// `/proc/self/cgroup`.
    ///
    /// Loud on failure rather than silently degrading to signals: a server that
    /// believes it can stop the worker and cannot is worse than one that
    /// refuses to start the worker at all.
    fn own_cgroup() -> io::Result<PathBuf> {
        const MOUNT: &str = "/sys/fs/cgroup";
        let text = fs::read_to_string("/proc/self/cgroup")?;
        let relative = text
            .lines()
            .find_map(|line| line.strip_prefix("0::"))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::Unsupported,
                    "no cgroup v2 membership in /proc/self/cgroup; \
                     the idle worker needs a unified hierarchy \
                     (systemd unit with Delegate=yes)",
                )
            })?;
        // The path is absolute within the hierarchy ("/system.slice/scan.service"),
        // so it is joined by concatenation rather than by `Path::join`, which
        // would discard the mount point.
        Ok(PathBuf::from(format!("{MOUNT}{}", relative.trim())))
    }
}

#[cfg(target_os = "freebsd")]
mod imp {
    use super::io;
    use std::os::unix::process::CommandExt;
    use std::process::Command;

    /// Nothing to hold: the containment lives in the kernel's reaper state for
    /// the child, addressed by its pid.
    #[derive(Debug)]
    pub(super) struct Control;

    /// `REAPER_KILL_SUBTREE` is unset, so the signal goes to every descendant
    /// rather than to one subtree of them. Not in the `libc` crate.
    const REAPER_KILL_ALL: libc::c_uint = 0;

    /// `struct procctl_reaper_kill` from `<sys/procctl.h>`. Not in the `libc`
    /// crate, so it is declared here; the layout is the kernel's ABI.
    #[repr(C)]
    struct ReaperKill {
        rk_sig: libc::c_int,
        rk_flags: libc::c_uint,
        rk_subtree: libc::pid_t,
        rk_killed: libc::c_uint,
        rk_fpid: libc::pid_t,
    }

    /// Have the child become the reaper for its own descendants before exec.
    ///
    /// FreeBSD has no cgroups, and a process group is not containment: a
    /// descendant that calls `setsid` leaves it. Reaper status is what makes
    /// the subtree addressable as a whole, and acquiring it before exec means
    /// there is no descendant that predates it.
    pub(super) fn prepare(cmd: &mut Command) -> io::Result<Control> {
        // Its own process group as well, so the development fallback on other
        // platforms and any operator reaching for `kill(1)` see the same shape.
        cmd.process_group(0);
        // SAFETY: runs between fork and exec; `procctl(2)` is a single syscall
        // that allocates nothing. `id` 0 addresses the calling process.
        unsafe {
            cmd.pre_exec(|| {
                if libc::procctl(
                    libc::P_PID,
                    0,
                    libc::PROC_REAP_ACQUIRE,
                    std::ptr::null_mut(),
                ) < 0
                {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        Ok(Control)
    }

    pub(super) fn freeze(_control: &Control, pid: u32) -> io::Result<()> {
        // The worker first, so it forks nothing new while its descendants are
        // still being stopped; then the descendants, which `PROC_REAP_KILL`
        // reaches and a process-group signal would not.
        signal(pid, libc::SIGSTOP)?;
        reap_signal(pid, libc::SIGSTOP)
    }

    pub(super) fn thaw(_control: &Control, pid: u32) -> io::Result<()> {
        // Descendants first, so the worker never runs against children that are
        // still stopped.
        reap_signal(pid, libc::SIGCONT)?;
        signal(pid, libc::SIGCONT)
    }

    pub(super) fn kill(_control: &Control, pid: u32) -> io::Result<()> {
        reap_signal(pid, libc::SIGKILL)?;
        signal(pid, libc::SIGKILL)
    }

    pub(super) fn release(_control: &Control) {}

    /// Signal the worker itself. `PROC_REAP_KILL` reaches only descendants of
    /// the reaper, never the reaper, so this is not redundant with it.
    fn signal(pid: u32, sig: libc::c_int) -> io::Result<()> {
        let pid = libc::pid_t::try_from(pid).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("worker pid {pid}: {e}"),
            )
        })?;
        // SAFETY: `kill(2)` with a pid this process spawned; no memory effects.
        if unsafe { libc::kill(pid, sig) } < 0 {
            let e = io::Error::last_os_error();
            // Already gone is the outcome we wanted.
            if e.raw_os_error() != Some(libc::ESRCH) {
                return Err(e);
            }
        }
        Ok(())
    }

    /// Signal every descendant of the worker.
    fn reap_signal(pid: u32, sig: libc::c_int) -> io::Result<()> {
        let mut rk = ReaperKill {
            rk_sig: sig,
            rk_flags: REAPER_KILL_ALL,
            rk_subtree: 0,
            rk_killed: 0,
            rk_fpid: 0,
        };
        // SAFETY: `procctl(2)` with a correctly-sized `procctl_reaper_kill` for
        // the PROC_REAP_KILL command, addressed to a pid this process spawned.
        let rc = unsafe {
            libc::procctl(
                libc::P_PID,
                libc::id_t::from(pid),
                libc::PROC_REAP_KILL,
                std::ptr::from_mut(&mut rk).cast(),
            )
        };
        if rc < 0 {
            let e = io::Error::last_os_error();
            // ESRCH: no descendants to signal. SIGCHLD-less quiet is success.
            if e.raw_os_error() != Some(libc::ESRCH) {
                return Err(e);
            }
        }
        Ok(())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
mod imp {
    use super::io;
    use std::os::unix::process::CommandExt;
    use std::process::Command;

    /// Development only. A process group is not containment — a descendant that
    /// calls `setsid` escapes it — so this platform gets the behaviour without
    /// the guarantee. The fleet is Linux and FreeBSD, which have one each.
    #[derive(Debug)]
    pub(super) struct Control;

    // The Result is the shape `Idle::spawn` needs from every platform; this one
    // has nothing that can fail.
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn prepare(cmd: &mut Command) -> io::Result<Control> {
        cmd.process_group(0);
        Ok(Control)
    }

    pub(super) fn freeze(_control: &Control, pid: u32) -> io::Result<()> {
        signal(pid, libc::SIGSTOP)
    }

    pub(super) fn thaw(_control: &Control, pid: u32) -> io::Result<()> {
        signal(pid, libc::SIGCONT)
    }

    pub(super) fn kill(_control: &Control, pid: u32) -> io::Result<()> {
        signal(pid, libc::SIGKILL)
    }

    pub(super) fn release(_control: &Control) {}

    /// The whole process group: the worker made itself its own leader in
    /// `prepare`, so its pid is the group id.
    fn signal(pid: u32, sig: libc::c_int) -> io::Result<()> {
        let pid = libc::pid_t::try_from(pid).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("worker pid {pid}: {e}"),
            )
        })?;
        // SAFETY: `killpg(2)` on a group this process created; no memory effects.
        if unsafe { libc::killpg(pid, sig) } < 0 {
            let e = io::Error::last_os_error();
            if e.raw_os_error() != Some(libc::ESRCH) {
                return Err(e);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::Idle;
    use std::process::Command;
    use std::time::{Duration, Instant};

    /// A child that burns CPU as fast as it can, so "stopped" is observable as
    /// the absence of progress rather than asserted from the API's return code.
    fn spinner() -> Command {
        let mut cmd = Command::new("/bin/sh");
        // Counts into a file: the file's length is a monotone clock that only
        // advances while the process runs.
        cmd.arg("-c")
            .arg("i=0; while :; do i=$((i+1)); printf . >>\"$OUT\"; done");
        cmd
    }

    fn progress(path: &std::path::Path) -> u64 {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    }

    /// Spawn, or report why this environment cannot exercise the real thing.
    ///
    /// On Linux the control primitive is a cgroup the process must be able to
    /// create under its own, which needs a delegated subtree (`Delegate=yes`,
    /// or a root container). A developer checkout usually has neither, and a
    /// test that fails there would be reporting the environment rather than the
    /// code. It says so on the way past instead of passing quietly — the
    /// production path is verified again at deploy time.
    fn spawn_or_skip(cmd: Command) -> Option<Idle> {
        match Idle::spawn(cmd) {
            Ok(idle) => Some(idle),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                eprintln!("SKIPPED: no writable cgroup subtree for the idle worker ({e})");
                None
            }
            Err(e) if e.kind() == std::io::ErrorKind::Unsupported => {
                eprintln!("SKIPPED: {e}");
                None
            }
            Err(e) => panic!("spawn: {e}"),
        }
    }

    #[test]
    fn freeze_stops_progress_and_thaw_resumes_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("ticks");
        let mut cmd = spinner();
        cmd.env("OUT", &out);
        let Some(idle) = spawn_or_skip(cmd) else {
            return;
        };

        // Wait for it to be demonstrably running before freezing, so a slow
        // fork cannot pass for a successful freeze.
        let start = Instant::now();
        while progress(&out) == 0 {
            assert!(start.elapsed() < Duration::from_secs(10), "child never ran");
            std::thread::sleep(Duration::from_millis(10));
        }

        idle.freeze().expect("freeze");
        // One sleep to let anything already in flight land, then the reading
        // that must not move.
        std::thread::sleep(Duration::from_millis(100));
        let frozen_at = progress(&out);
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(
            progress(&out),
            frozen_at,
            "a frozen child kept making progress",
        );

        idle.thaw().expect("thaw");
        let start = Instant::now();
        while progress(&out) == frozen_at {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "a thawed child never resumed",
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn kill_reaches_a_frozen_child() {
        // The Drop path depends on this: a frozen process never exits, so
        // reaping one that was not killed first would block forever.
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("ticks");
        let mut cmd = spinner();
        cmd.env("OUT", &out);
        let Some(mut idle) = spawn_or_skip(cmd) else {
            return;
        };

        idle.freeze().expect("freeze");
        idle.kill().expect("kill");

        let start = Instant::now();
        loop {
            if idle.exited().expect("try_wait").is_some() {
                break;
            }
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "a killed frozen child never exited",
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    fn descendants_stop_with_the_worker() {
        // The property a process group would not give us: a grandchild that
        // left the group is still contained. Only asserted on the platforms
        // that promise it — the fallback used elsewhere signals a process
        // group, which is exactly what this escapes.
        //
        // The two base systems detach a session with different tools: Linux
        // has setsid(1) from util-linux, FreeBSD has daemon(8).
        let detach = if cfg!(target_os = "linux") {
            "setsid"
        } else {
            "daemon -f"
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("ticks");
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c")
            .arg(format!(
                "{detach} sh -c 'while :; do printf . >>\"$OUT\"; done' & wait"
            ))
            .env("OUT", &out);
        let Some(idle) = spawn_or_skip(cmd) else {
            return;
        };

        let start = Instant::now();
        while progress(&out) == 0 {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "grandchild never ran",
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        idle.freeze().expect("freeze");
        std::thread::sleep(Duration::from_millis(100));
        let frozen_at = progress(&out);
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(
            progress(&out),
            frozen_at,
            "a grandchild in its own session kept running",
        );
        idle.thaw().expect("thaw");
    }
}
