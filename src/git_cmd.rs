//! Timeout-bounded `git` subprocess helper.
//!
//! Any git operation that touches the network (fetch, pull, clone) can hang
//! indefinitely: SSH waiting on a FIDO2 PIN, a dead TCP connection, a
//! credential helper prompting on a TTY. `std::process::Command::output()`
//! has no way to abort such a call.
//!
//! This helper mirrors cleave's `radare2::execute_rizin_with_timeout`
//! pattern so every long-running subprocess in litmus has the same shape:
//!
//! 1. Place the child in its own process group via `process_group(0)`.
//! 2. Set non-interactive env vars so the common hang modes fail in seconds.
//! 3. Poll `try_wait()` until the deadline, then `kill(-pgid, SIGKILL)` to
//!    tear down the child and any descendants it spawned (`ssh`,
//!    `ssh-askpass`, credential helpers).
//!
//! The group-kill is the part a plain `child.kill()` cannot do. Without it,
//! killing `git` can leave `ssh` behind still holding the network socket
//! and the controlling terminal.
//!
//! Output is captured via pipes. Git's network commands produce little
//! output so pipe-buffer deadlock is not a concern here — the rizin helper
//! needs background reader threads because rizin can emit 100 MB+; git
//! doesn't.

use anyhow::{Context, Result};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// Run `git` with the given args, killing the child (and its process group)
/// if it exceeds `timeout`. Returns the captured output.
pub(crate) fn run(args: &[&str], timeout: Duration) -> Result<Output> {
    let mut cmd = Command::new("git");
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Prevent any prompt from git or its SSH child:
        //   GIT_TERMINAL_PROMPT=0 — no HTTPS username/password prompt.
        //   BatchMode=yes         — no passphrase or FIDO2 PIN prompt; ssh
        //                           fails immediately when interaction would
        //                           be required.
        //   ConnectTimeout=15     — TCP handshake fails fast on a dead host.
        .env("GIT_TERMINAL_PROMPT", "0")
        .env(
            "GIT_SSH_COMMAND",
            "ssh -o BatchMode=yes -o ConnectTimeout=15",
        );

    #[cfg(unix)]
    cmd.process_group(0);

    let mut child = cmd.spawn().context("failed to spawn git")?;
    let pid = child.id();

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait().context("failed to poll git process")? {
            Some(_) => {
                return child
                    .wait_with_output()
                    .context("failed to read git output")
            }
            None if Instant::now() >= deadline => {
                kill_group(pid);
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!("git timed out after {}s", timeout.as_secs());
            }
            None => std::thread::sleep(Duration::from_millis(100)),
        }
    }
}

/// `git rev-parse --short HEAD` for a local repo. Local-only, no network,
/// no timeout required.
pub(crate) fn short_head(path: &Path) -> Option<String> {
    let output = Command::new("git")
        .args([
            "-C",
            &path.to_string_lossy(),
            "rev-parse",
            "--short",
            "HEAD",
        ])
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

#[cfg(unix)]
fn kill_group(pid: u32) {
    // `process_group(0)` gives the child pgid == pid. Signalling -pid
    // delivers to every process in that group — including any ssh /
    // ssh-askpass / credential helper git spawned.
    //
    // SAFETY: kill(2) is signal-safe and taking a signed pgid is POSIX.
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_group(_pid: u32) {}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn run_returns_output_for_fast_command() {
        let out = run(&["--version"], Duration::from_secs(10)).unwrap();
        assert!(out.status.success());
        assert!(String::from_utf8_lossy(&out.stdout).starts_with("git version"));
    }
}
