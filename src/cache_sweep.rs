//! Best-effort, non-blocking cache reclamation.
//!
//! A cache here is a directory of entries, each carrying an mtime. A *sweep*
//! deletes the oldest entries until the directory is within an age, count and
//! size budget. The whole thing runs on one detached thread that dies with the
//! process — no daemon, no async runtime, no persistent index.
//!
//! The count ceiling is the one that usually binds: entries are small, so a
//! byte budget alone lets a directory reach millions of files, which every
//! filesystem tool handles badly and which makes the sweep meant to fix it
//! progressively more expensive.
//!
//! Two properties keep it cheap and safe:
//! - A `.last-sweep` marker gates the walk, so the common case is a single
//!   `stat`, never a scan of a million-entry directory. The marker records
//!   whether the sweep it names *finished*: a detached thread dies with its
//!   process, and a sweep cut short that way must not book itself as a full
//!   day's work, or a cache too large to sweep in one short run would never be
//!   swept at all.
//! - Every filesystem error is ignored: reclaiming disk must never disturb the
//!   program it runs alongside. These caches are written once and never
//!   rewritten, and on Unix unlinking a file another thread has open leaves that
//!   reader's descriptor valid, so a concurrent reader never sees a torn file.
//!
//! scan owns the caches it sweeps, so it owns this mechanism; filefacts and
//! fletch each keep their own sweep for the caches they own.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

/// Re-walk a cache at most this often; cheaper runs just read the marker.
const SWEEP_INTERVAL: Duration = Duration::from_hours(24);
/// Re-walk this soon after a sweep that never reported finishing. A sweep still
/// running after this long can be joined by a second one, which costs a
/// redundant walk but nothing else: eviction is a delete of an entry each
/// process decided was oldest, and a failed delete is already ignored.
const RETRY_INTERVAL: Duration = Duration::from_hours(1);
/// Marker file, written in the primary root: its mtime records when the last
/// sweep started and its single byte whether that sweep finished.
const MARKER: &str = ".last-sweep";
/// Marker contents: a sweep is in flight, or died with its process.
const MARK_STARTED: &[u8] = b"s";
/// Marker contents: the sweep that wrote it ran to completion.
const MARK_DONE: &[u8] = b"d";

/// A cache directory and how deep its entries live below it.
#[derive(Debug)]
pub(crate) struct Root {
    /// Directory to sweep.
    pub path: PathBuf,
    /// `1` = direct children are entries (files or dirs); `2` = grandchildren
    /// are entries and each child is a container to descend into.
    pub depth: u8,
}

/// One component's caches and the budget they must collectively stay within.
#[derive(Debug)]
pub(crate) struct Budget {
    /// Component name, for a debug log line.
    pub label: &'static str,
    /// One or more directories sharing the `max_bytes` ceiling. The first is
    /// the *primary* root and holds the sweep marker.
    pub roots: Vec<Root>,
    /// Delete entries older than this.
    pub max_age: Duration,
    /// After the age pass, delete oldest entries until the total across all
    /// roots is at or below this.
    pub max_bytes: u64,
    /// Likewise for the entry count across all roots. Usually the binding
    /// ceiling; `max_bytes` bounds the case of a few very large entries.
    pub max_entries: usize,
}

/// One sweep in flight per process; a second [`spawn`] is a no-op until it ends.
static RUNNING: AtomicBool = AtomicBool::new(false);

/// Clears [`RUNNING`] on drop, so a panic inside the sweep can never wedge every
/// future [`spawn`] into a permanent no-op.
struct RunningGuard;

impl Drop for RunningGuard {
    fn drop(&mut self) {
        RUNNING.store(false, Ordering::Release);
    }
}

/// Sweep `budgets` on a detached thread and return immediately. The thread is
/// reaped when the process exits; a sweep interrupted that way leaves the
/// marker unfinished, so the next run retries it within the hour rather than
/// waiting out a full interval.
///
/// For a one-shot CLI, call once at startup — it races the real work. For a
/// long-lived daemon, use [`spawn_periodic`] instead.
pub(crate) fn spawn(mut budgets: Vec<Budget>) {
    budgets.retain(|b| !b.roots.is_empty());
    if budgets.is_empty() {
        return;
    }
    if RUNNING.swap(true, Ordering::AcqRel) {
        return; // a sweep is already running
    }
    // `Builder::spawn` returns an error (rather than panicking) if the OS
    // refuses the thread; reset the guard so a later call can retry.
    let started = std::thread::Builder::new()
        .name("cache-sweep".into())
        .spawn(move || {
            let _guard = RunningGuard;
            run_all(&budgets);
        });
    if started.is_err() {
        RUNNING.store(false, Ordering::Release);
    }
}

/// Sweep `budgets` now and every `interval` thereafter, on one detached thread,
/// for the life of the process. For daemons whose process never exits, so a
/// startup-only sweep would never fire again. The daily marker keeps each wake
/// cheap regardless of `interval`.
pub(crate) fn spawn_periodic(mut budgets: Vec<Budget>, interval: Duration) {
    budgets.retain(|b| !b.roots.is_empty());
    if budgets.is_empty() {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("cache-sweep".into())
        .spawn(move || {
            loop {
                run_all(&budgets);
                std::thread::sleep(interval);
            }
        });
}

fn run_all(budgets: &[Budget]) {
    for b in budgets {
        run(b);
    }
}

/// An entry considered for eviction: its path, whether it is a directory, and
/// the size and mtime read once and reused by both passes.
struct Entry {
    path: PathBuf,
    is_dir: bool,
    modified: SystemTime,
    bytes: u64,
}

fn run(b: &Budget) {
    let Some(primary) = b.roots.first().and_then(|r| trusted_dir(&r.path)) else {
        return;
    };
    if !due(&primary) {
        return;
    }
    // Mark the start (mtime = now) so a racing process sees a fresh marker and
    // skips. Best-effort; if the directory doesn't exist yet, there's nothing
    // to sweep anyway.
    let marker = primary.join(MARKER);
    write_marker(&marker, MARK_STARTED);

    let mut entries = Vec::new();
    for r in &b.roots {
        if let Some(path) = trusted_dir(&r.path) {
            collect(&path, r.depth, &mut entries);
        }
    }

    let now = SystemTime::now();
    let mut freed = 0u64;
    let mut removed = 0usize;

    // Age pass: drop anything past the retention window outright.
    entries.retain(|e| {
        if now.duration_since(e.modified).unwrap_or_default() > b.max_age && remove(e) {
            freed += e.bytes;
            removed += 1;
            false
        } else {
            true
        }
    });

    // Count/size pass: over either ceiling, evict oldest first until under 90%
    // of both. Stopping at 90% rather than exactly at the cap keeps a cache
    // sitting at the ceiling from re-triggering a sweep on the very next write.
    // Divide before multiplying so a caller's `u64::MAX`/`usize::MAX` ceiling
    // can't overflow the target.
    let mut count = entries.len();
    let mut total: u64 = entries.iter().map(|e| e.bytes).sum();
    if total > b.max_bytes || count > b.max_entries {
        entries.sort_by_key(|e| e.modified); // oldest first
        let byte_target = b.max_bytes / 10 * 9;
        let count_target = b.max_entries / 10 * 9;
        for e in &entries {
            if total <= byte_target && count <= count_target {
                break;
            }
            if remove(e) {
                total = total.saturating_sub(e.bytes);
                count -= 1;
                freed += e.bytes;
                removed += 1;
            }
        }
    }

    // Record completion, so the next run waits a full interval rather than
    // retrying. Not reached if this thread dies with its process mid-sweep.
    write_marker(&marker, MARK_DONE);

    if removed > 0 {
        tracing::debug!(
            target: "cache_sweep",
            component = b.label,
            removed,
            freed_bytes = freed,
            "swept cache"
        );
    }
}

/// Whether `path` is a directory itself, not a symlink to one.
fn is_real_dir(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|m| m.is_dir())
}

/// Symlinks followed resolving a root before it is refused, as the kernel's
/// own `ELOOP` limit does.
#[cfg(unix)]
const MAX_LINKS: u32 = 40;

/// `dir`, with any symlinks on the way to it resolved, when the sweep may
/// delete from it. Entries are deleted by path, so whoever can change a
/// directory on that path can swap in a symlink between the walk and the
/// delete and aim it at any file this process may remove. Refused: a
/// relative path, which names different files from each working directory
/// (`HOME=.`); a `dir` that is itself a symlink, pointing somewhere this
/// budget does not own; and on unix, a path through a directory or link that
/// another user can change — a cache under a shared directory (`HOME=/tmp`,
/// or `XDG_CACHE_HOME` pointed there) whose `.cache` someone else created.
pub(crate) fn trusted_dir(dir: &Path) -> Option<PathBuf> {
    if !dir.is_absolute() || !is_real_dir(dir) {
        return None;
    }
    #[cfg(unix)]
    let dir = {
        // SAFETY: neither call has preconditions; each only returns an id.
        #[allow(unsafe_code)]
        let ids = unsafe { (libc::geteuid(), libc::getegid()) };
        let mut resolved = PathBuf::new();
        resolve_trusted(dir, ids, &mut resolved, &mut 0)?;
        resolved
    };
    #[cfg(not(unix))]
    let dir = dir.to_path_buf();
    Some(dir)
}

/// Resolve `path` onto `resolved` one component at a time, following
/// symlinks, and fail on the first step someone other than this user (`ids`
/// is the effective uid and gid) or the superuser could change. A directory
/// passes only if no other user can write it: group-writable just for this
/// user's own group, or world-writable only with the sticky bit (`/tmp`),
/// where no one may rename what they do not own. `resolved` never holds a
/// symlink, so `..` is its lexical parent.
#[cfg(unix)]
fn resolve_trusted(
    path: &Path,
    (uid, gid): (u32, u32),
    resolved: &mut PathBuf,
    links: &mut u32,
) -> Option<()> {
    use std::os::unix::fs::MetadataExt;
    use std::path::Component;
    let trusted = |m: &fs::Metadata| {
        let mode = m.mode();
        (m.uid() == uid || m.uid() == 0)
            && (m.is_symlink()
                || mode & 0o1000 != 0
                || (mode & 0o002 == 0 && (mode & 0o020 == 0 || m.gid() == gid)))
    };
    for component in path.components() {
        match component {
            Component::RootDir => {
                *resolved = PathBuf::from("/");
                if !fs::symlink_metadata(&*resolved).is_ok_and(|m| trusted(&m)) {
                    return None;
                }
            }
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                let next = resolved.join(name);
                let meta = fs::symlink_metadata(&next).ok()?;
                if !trusted(&meta) {
                    return None;
                }
                if meta.is_symlink() {
                    *links += 1;
                    if *links > MAX_LINKS {
                        return None;
                    }
                    resolve_trusted(&fs::read_link(&next).ok()?, (uid, gid), resolved, links)?;
                } else {
                    *resolved = next;
                }
            }
            Component::Prefix(_) => return None,
        }
    }
    Some(())
}

/// Write the sweep marker without following a symlink planted in its place:
/// the write would otherwise truncate whatever the link names. Best-effort.
fn write_marker(path: &Path, contents: &[u8]) {
    use std::io::Write as _;
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(not(unix))]
    if fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
        return;
    }
    if let Ok(mut file) = options.open(path) {
        let _ = file.write_all(contents);
    }
}

/// True unless this root was swept recently: within [`SWEEP_INTERVAL`] for a
/// sweep that finished, or [`RETRY_INTERVAL`] for one that died with its
/// process. A missing or unreadable marker counts as due, so a never-swept
/// cache is handled on the first run.
fn due(root: &Path) -> bool {
    let marker = root.join(MARKER);
    let Ok(started) = fs::symlink_metadata(&marker).and_then(|m| m.modified()) else {
        return true;
    };
    let Ok(elapsed) = started.elapsed() else {
        return true; // marker dated in the future (clock skew): sweep now
    };
    let interval = match fs::read(&marker).as_deref() {
        Ok(MARK_DONE) => SWEEP_INTERVAL,
        _ => RETRY_INTERVAL,
    };
    elapsed >= interval
}

/// Gather the cache entries at `depth` below `root` into `out`. At `depth <= 1`
/// each child is an entry; deeper, each directory child is descended into.
fn collect(root: &Path, depth: u8, out: &mut Vec<Entry>) {
    let Ok(rd) = fs::read_dir(root) else {
        return;
    };
    for e in rd.flatten() {
        let Ok(ft) = e.file_type() else {
            continue;
        };
        let path = e.path();
        if depth <= 1 {
            if path.file_name().is_some_and(|n| n == MARKER) {
                continue; // never evict our own marker
            }
            if let Some(entry) = stat_entry(path, ft.is_dir()) {
                out.push(entry);
            }
        } else if ft.is_dir() {
            collect(&path, depth - 1, out);
        }
    }
}

fn stat_entry(path: PathBuf, is_dir: bool) -> Option<Entry> {
    // The entry itself, not a symlink's target: a link is evicted as a link.
    let meta = fs::symlink_metadata(&path).ok()?;
    let modified = meta.modified().ok()?;
    let bytes = if is_dir { dir_size(&path) } else { meta.len() };
    Some(Entry {
        path,
        is_dir,
        modified,
        bytes,
    })
}

/// Recursive byte size of a directory entry. Only ever called on the small
/// per-item directories stng's rizin cache uses (a handful of files each).
///
/// Recursion is gated on `file_type()`, which does NOT follow symlinks: a
/// symlink is neither descended nor counted. That keeps a stray symlink loop
/// in the cache dir from spinning this into an (uncatchable) stack-overflow
/// abort, and a symlink to a large tree from inflating the byte total.
fn dir_size(dir: &Path) -> u64 {
    let Ok(rd) = fs::read_dir(dir) else {
        return 0;
    };
    let mut total = 0;
    for e in rd.flatten() {
        let Ok(ft) = e.file_type() else {
            continue;
        };
        if ft.is_dir() {
            total += dir_size(&e.path());
        } else if ft.is_file() {
            total += e.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    total
}

fn remove(e: &Entry) -> bool {
    let r = if e.is_dir {
        fs::remove_dir_all(&e.path)
    } else {
        fs::remove_file(&e.path)
    };
    r.is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cache-sweep-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_aged(path: &Path, bytes: usize, age: Duration) {
        let mut f = fs::File::create(path).unwrap();
        f.write_all(&vec![b'x'; bytes]).unwrap();
        f.set_modified(SystemTime::now() - age).unwrap();
    }

    fn day(n: u64) -> Duration {
        Duration::from_secs(n * 24 * 60 * 60)
    }

    #[test]
    fn age_pass_removes_only_old() {
        let dir = scratch("age");
        write_aged(&dir.join("old.json"), 10, day(40));
        write_aged(&dir.join("fresh.json"), 10, day(1));
        let budget = Budget {
            label: "test",
            roots: vec![Root {
                path: dir.clone(),
                depth: 1,
            }],
            max_age: day(30),
            max_bytes: u64::MAX,
            max_entries: usize::MAX,
        };
        run(&budget);
        assert!(!dir.join("old.json").exists(), "40-day entry evicted");
        assert!(dir.join("fresh.json").exists(), "1-day entry kept");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn size_pass_evicts_oldest_first() {
        let dir = scratch("size");
        // Five 400-byte entries, distinct ages newest→oldest; cap 1000 → 90% = 900.
        for (i, age) in [5u64, 4, 3, 2, 1].into_iter().enumerate() {
            write_aged(&dir.join(format!("e{i}.json")), 400, day(age));
        }
        let budget = Budget {
            label: "test",
            roots: vec![Root {
                path: dir.clone(),
                depth: 1,
            }],
            max_age: day(3650), // age pass inert
            max_bytes: 1000,
            max_entries: usize::MAX,
        };
        run(&budget);
        // 2000 bytes → evict oldest until ≤ 900: remove e0(5d), e1(4d), e2(3d).
        assert!(!dir.join("e0.json").exists(), "oldest evicted");
        assert!(!dir.join("e1.json").exists());
        assert!(!dir.join("e2.json").exists());
        assert!(dir.join("e3.json").exists(), "newest kept");
        assert!(dir.join("e4.json").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn depth_two_reaches_grandchildren() {
        let dir = scratch("depth2");
        let ver = dir.join("v1");
        fs::create_dir_all(&ver).unwrap();
        write_aged(&ver.join("a.zst"), 10, day(40));
        write_aged(&ver.join("b.zst"), 10, day(1));
        let budget = Budget {
            label: "test",
            roots: vec![Root {
                path: dir.clone(),
                depth: 2,
            }],
            max_age: day(30),
            max_bytes: u64::MAX,
            max_entries: usize::MAX,
        };
        run(&budget);
        assert!(!ver.join("a.zst").exists(), "aged grandchild evicted");
        assert!(ver.join("b.zst").exists(), "fresh grandchild kept");
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn dir_size_does_not_follow_symlink_loops() {
        let dir = scratch("symloop");
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("a.json"), b"1234567890").unwrap(); // 10 bytes
        // A symlink back to the parent: following it would recurse forever and
        // abort the process on stack overflow.
        std::os::unix::fs::symlink(&dir, sub.join("loop")).unwrap();
        // Must return promptly, counting only the real file.
        assert_eq!(
            dir_size(&dir),
            10,
            "the symlink is neither followed nor counted"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn count_pass_evicts_oldest_to_ninety_percent() {
        let dir = scratch("count");
        // Thirteen tiny entries, oldest first; cap 10 → 90% = 9, so the four
        // oldest go. The byte ceiling is inert, isolating the count pass.
        for i in 0..13u64 {
            write_aged(&dir.join(format!("e{i:02}.json")), 1, day(20 - i));
        }
        run(&Budget {
            label: "test",
            roots: vec![Root {
                path: dir.clone(),
                depth: 1,
            }],
            max_age: day(3650),
            max_bytes: u64::MAX,
            max_entries: 10,
        });
        let kept: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .filter(|n| n != MARKER)
            .collect();
        assert_eq!(kept.len(), 9, "evicted down to 90% of the cap");
        assert!(!dir.join("e00.json").exists(), "oldest evicted");
        assert!(!dir.join("e03.json").exists());
        assert!(dir.join("e04.json").exists(), "newest kept");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unfinished_marker_retries_within_the_hour() {
        let dir = scratch("unfinished");
        let marker = dir.join(MARKER);
        let two_hours = Duration::from_hours(2);

        // A sweep that finished two hours ago holds off for a full day.
        fs::write(&marker, MARK_DONE).unwrap();
        fs::File::options()
            .write(true)
            .open(&marker)
            .unwrap()
            .set_modified(SystemTime::now() - two_hours)
            .unwrap();
        assert!(!due(&dir), "completed sweep gates for SWEEP_INTERVAL");

        // One that died mid-walk is retried instead: without this, a cache too
        // large to sweep inside one short-lived process is never swept at all.
        fs::write(&marker, MARK_STARTED).unwrap();
        fs::File::options()
            .write(true)
            .open(&marker)
            .unwrap()
            .set_modified(SystemTime::now() - two_hours)
            .unwrap();
        assert!(
            due(&dir),
            "unfinished sweep is due again after RETRY_INTERVAL"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_root_is_not_swept() {
        let target = scratch("symroot-target");
        write_aged(&target.join("precious.txt"), 10, day(400));
        let link = scratch("symroot-link").join("cache");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        run(&Budget {
            label: "test",
            roots: vec![Root {
                path: link.clone(),
                depth: 1,
            }],
            max_age: day(30),
            max_bytes: 0,
            max_entries: 0,
        });
        assert!(
            target.join("precious.txt").exists(),
            "nothing behind the link"
        );
        assert!(
            !target.join(MARKER).exists(),
            "no marker written through it"
        );
        let _ = fs::remove_dir_all(link.parent().unwrap());
        let _ = fs::remove_dir_all(&target);
    }

    /// A one-root budget over `path` that evicts everything it may.
    fn evict_all(path: PathBuf) -> Budget {
        Budget {
            label: "test",
            roots: vec![Root { path, depth: 1 }],
            max_age: day(30),
            max_bytes: 0,
            max_entries: 0,
        }
    }

    #[test]
    fn relative_root_is_not_swept() {
        // `HOME=.` makes the stng roots relative: they would name whatever
        // tree the process runs in, such as the samples under analysis.
        let dir = scratch("relative");
        write_aged(&dir.join("precious.txt"), 10, day(400));
        let cwd = std::env::current_dir().unwrap();
        let relative: PathBuf = cwd
            .components()
            .skip(1)
            .map(|_| std::path::Component::ParentDir)
            .chain(dir.components().skip(1))
            .collect();
        assert!(relative.join("precious.txt").exists());
        run(&evict_all(relative));
        assert!(dir.join("precious.txt").exists());
        assert!(!dir.join(MARKER).exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// Another user could rename `shared`'s children, so it could aim a
    /// delete through any root below it at the sweeping user's own files: a
    /// root reached through a symlink planted there is the attack itself.
    #[cfg(unix)]
    #[test]
    fn root_another_user_can_redirect_is_not_swept() {
        use std::os::unix::fs::PermissionsExt;
        let base = scratch("shared");
        let shared = base.join("shared");
        let victim = base.join("victim");
        let cache = shared.join("cache");
        fs::create_dir_all(&cache).unwrap();
        fs::create_dir_all(victim.join("strings")).unwrap();
        write_aged(&cache.join("old.json"), 10, day(400));
        write_aged(&victim.join("strings").join("precious.go"), 10, day(400));
        std::os::unix::fs::symlink(&victim, shared.join("stng")).unwrap();

        let mode = |m| fs::set_permissions(&shared, fs::Permissions::from_mode(m)).unwrap();
        mode(0o777);
        run(&evict_all(cache.clone()));
        run(&evict_all(shared.join("stng").join("strings")));
        assert!(cache.join("old.json").exists(), "world-writable parent");
        assert!(victim.join("strings").join("precious.go").exists());

        // Sticky, like /tmp: others can no longer rename what they don't own.
        mode(0o1777);
        run(&evict_all(cache.clone()));
        assert!(!cache.join("old.json").exists(), "sticky parent is swept");
        let _ = fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn marker_write_does_not_follow_a_planted_symlink() {
        let dir = scratch("marker-link");
        let victim = scratch("marker-victim").join("victim.txt");
        fs::write(&victim, b"keep me").unwrap();
        std::os::unix::fs::symlink(&victim, dir.join(MARKER)).unwrap();
        write_marker(&dir.join(MARKER), MARK_DONE);
        assert_eq!(fs::read(&victim).unwrap(), b"keep me");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(victim.parent().unwrap());
    }

    #[test]
    fn marker_gates_and_survives() {
        let dir = scratch("gate");
        assert!(due(&dir), "no marker ⇒ due");
        let budget = Budget {
            label: "test",
            roots: vec![Root {
                path: dir.clone(),
                depth: 1,
            }],
            max_age: day(30),
            max_bytes: u64::MAX,
            max_entries: usize::MAX,
        };
        run(&budget);
        assert!(dir.join(MARKER).exists(), "sweep writes the marker");
        assert!(!due(&dir), "fresh marker ⇒ not due");
        let _ = fs::remove_dir_all(&dir);
    }
}
