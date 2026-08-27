//! Live jemalloc counters and operator-triggered heap profiles.
//!
//! FreeBSD uses jemalloc in libc, so the bundled `tikv-jemallocator` profiling
//! feature is not present in the production worker.  This module talks to the
//! native allocator through its public `mallctl` interface.  It is deliberately
//! opt-in: setting `SCAN_HEAP_PROFILE_DIR` enables `SIGUSR1` heap dumps, while
//! the lightweight counters are always available on FreeBSD.

use std::path::PathBuf;
use std::sync::OnceLock;

/// A point-in-time view of jemalloc's own byte counters.
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    /// Bytes currently allocated to application objects.
    pub allocated: usize,
    /// Bytes in active pages, including allocator fragmentation.
    pub active: usize,
    /// Bytes resident in jemalloc-managed pages.
    pub resident: usize,
    /// Bytes retained in virtual memory but not currently resident.
    pub retained: usize,
}

static PROFILE_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

#[cfg(target_os = "freebsd")]
fn prepare_profile_dir(dir: &std::path::Path) -> bool {
    if read_ctl::<bool>(b"config.prof\0") != Some(true) {
        tracing::warn!(
            path = %dir.display(),
            "SCAN_HEAP_PROFILE_DIR is ignored: the system jemalloc was built without profiling"
        );
        return false;
    }
    if let Err(error) = std::fs::create_dir_all(dir) {
        tracing::warn!(path = %dir.display(), %error, "cannot create heap profile directory");
        false
    } else {
        true
    }
}

#[cfg(not(target_os = "freebsd"))]
fn prepare_profile_dir(dir: &std::path::Path) -> bool {
    tracing::warn!(
        path = %dir.display(),
        "SCAN_HEAP_PROFILE_DIR is ignored: native mallctl heap dumps are only supported on FreeBSD"
    );
    false
}

/// Initialize the opt-in profile destination.
pub fn install() {
    let configured = std::env::var_os("SCAN_HEAP_PROFILE_DIR").map(PathBuf::from);
    let value = configured.filter(|dir| prepare_profile_dir(dir));
    let enabled = value.is_some();
    let _ = PROFILE_DIR.set(value);
    if enabled {
        tracing::info!("live jemalloc heap dumps enabled; send SIGUSR1 to write a profile");
    }
    if let Some(stats) = stats() {
        tracing::info!(
            allocated_mb = stats.allocated / (1024 * 1024),
            active_mb = stats.active / (1024 * 1024),
            resident_mb = stats.resident / (1024 * 1024),
            retained_mb = stats.retained / (1024 * 1024),
            "native jemalloc counters at startup",
        );
    }
    #[cfg(target_os = "freebsd")]
    tracing::info!(
        narenas = ?read_ctl::<u32>(b"opt.narenas\0"),
        dirty_decay_ms = ?read_ctl::<isize>(b"opt.dirty_decay_ms\0"),
        muzzy_decay_ms = ?read_ctl::<isize>(b"opt.muzzy_decay_ms\0"),
        retain = ?read_ctl::<bool>(b"opt.retain\0"),
        profiling_compiled = ?read_ctl::<bool>(b"config.prof\0"),
        "native jemalloc startup options",
    );
}

/// Dump the current jemalloc heap when the operator requested a signal dump.
///
/// This runs on the dedicated signal-wait thread, not in an async signal
/// handler, so filesystem and allocator calls are safe here.
pub fn dump_on_signal() {
    let Some(dir) = PROFILE_DIR.get().and_then(Option::as_ref) else {
        return;
    };
    let pid = std::process::id();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let path = dir.join(format!("atomscan-{pid}-{stamp}.heap"));

    #[cfg(target_os = "freebsd")]
    {
        let Some(path) = path.to_str() else {
            tracing::warn!(path = %path.display(), "heap profile path is not valid UTF-8");
            return;
        };
        match dump_freebsd(path) {
            Ok(()) => tracing::info!(path = %path, "jemalloc heap profile written"),
            Err(error) => tracing::warn!(%error, path = %path, "jemalloc heap profile failed"),
        }
    }

    #[cfg(not(target_os = "freebsd"))]
    let _ = path;
}

/// Read allocator counters, when the host exposes native jemalloc.
#[must_use]
pub fn stats() -> Option<Stats> {
    #[cfg(target_os = "freebsd")]
    {
        refresh_epoch()?;
        Some(Stats {
            allocated: read_ctl(b"stats.allocated\0")?,
            active: read_ctl(b"stats.active\0")?,
            resident: read_ctl(b"stats.resident\0")?,
            retained: read_ctl(b"stats.retained\0")?,
        })
    }
    #[cfg(not(target_os = "freebsd"))]
    {
        None
    }
}

#[cfg(target_os = "freebsd")]
mod freebsd {
    use std::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        pub(super) fn mallctl(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
}

#[cfg(target_os = "freebsd")]
fn read_ctl<T: Copy>(name: &'static [u8]) -> Option<T> {
    use std::mem::{MaybeUninit, size_of};
    use std::ptr;

    let mut value = MaybeUninit::<T>::uninit();
    let mut length = size_of::<T>();
    // SAFETY: `name` is a static NUL-terminated mallctl key, and `value` is a
    // correctly sized, writable buffer described by `length`.
    let status = unsafe {
        freebsd::mallctl(
            name.as_ptr().cast(),
            value.as_mut_ptr().cast(),
            &mut length,
            ptr::null_mut(),
            0,
        )
    };
    if status != 0 || length != size_of::<T>() {
        return None;
    }
    // SAFETY: mallctl reported success and filled the complete value.
    Some(unsafe { value.assume_init() })
}

#[cfg(target_os = "freebsd")]
fn refresh_epoch() -> Option<()> {
    use std::mem::size_of;
    use std::ptr;

    let mut epoch: usize = 1;
    let length = size_of::<usize>();
    // SAFETY: epoch is a documented writable usize mallctl value.
    let status = unsafe {
        freebsd::mallctl(
            c"epoch".as_ptr().cast(),
            ptr::null_mut(),
            ptr::null_mut(),
            (&mut epoch as *mut usize).cast(),
            length,
        )
    };
    (status == 0).then_some(())
}

#[cfg(target_os = "freebsd")]
fn dump_freebsd(path: &str) -> Result<(), String> {
    use std::ffi::CString;
    use std::mem::size_of;
    use std::os::unix::ffi::OsStrExt;
    use std::ptr;

    let c_path = CString::new(std::path::Path::new(path).as_os_str().as_bytes())
        .map_err(|_nul_error| "heap profile path contains NUL".to_string())?;
    let mut filename = c_path.as_ptr();
    // jemalloc's prof.dump command expects newp to point to a `const char *`.
    // SAFETY: the C string and pointer remain alive for the duration of the
    // mallctl call; the command does not retain the pointer after returning.
    let status = unsafe {
        freebsd::mallctl(
            c"prof.dump".as_ptr().cast(),
            ptr::null_mut(),
            ptr::null_mut(),
            (&mut filename as *mut *const std::ffi::c_char).cast(),
            size_of::<*const std::ffi::c_char>(),
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(format!("mallctl prof.dump returned errno {status}"))
    }
}

/// Warn when the host's in-libc jemalloc taxes every allocation.
///
/// FreeBSD builds libc's jemalloc with `--enable-debug` and `--enable-fill` on
/// -CURRENT, so each allocation runs jemalloc's invariant assertions and both
/// `malloc` and `free` memset the region (`opt.junk`). Neither shows up as
/// application time: profiles blame whatever called `malloc`, so the tax is
/// invisible unless something looks for it. Measured on uruk-hai (128-core
/// arm64, 16.0-CURRENT) over four large nested archives at 96 worker slots with
/// warm YARA caches: 284.7 s stock against 227.7 s with `junk:false`, at an
/// identical result hash. For reference, swapping the whole allocator out for
/// mimalloc scored 231.2 s on the same workload — i.e. `junk:false` alone
/// recovers what a different allocator would, which is why there is no bundled
/// allocator on this platform.
///
/// `opt.junk` is settable at startup; the assertions are compiled in and can
/// only be dropped by a `MALLOC_PRODUCTION` world, so the warning names
/// whichever remedy the host still needs.
pub fn warn_if_debug_allocator() {
    #[cfg(target_os = "freebsd")]
    {
        let junk = junk_setting().filter(|setting| *setting != "false");
        let is_debug_build = read_ctl::<bool>(b"config.debug\0") == Some(true);
        if junk.is_none() && !is_debug_build {
            return;
        }
        // One line, not one per condition: this fires on every start on a
        // -CURRENT host, and the operator's next move is the same either way.
        let remedy = if junk.is_some() {
            "add junk:false to MALLOC_CONF"
        } else {
            "build world with MALLOC_PRODUCTION to drop the assertions too"
        };
        tracing::warn!(
            opt_junk = junk.unwrap_or("false"),
            config_debug = is_debug_build,
            "the system jemalloc is a debugging build, which taxes every allocation and \
             free. Measured on a 128-core arm64 -CURRENT host: junk filling alone cost 20% \
             of wall-clock. To fix: {remedy}.",
        );
    }
}

/// The runtime value of `opt.junk` (`"true"`, `"false"`, `"alloc"`, `"free"`).
#[cfg(target_os = "freebsd")]
fn junk_setting() -> Option<&'static str> {
    let value = read_ctl::<*const std::ffi::c_char>(b"opt.junk\0")?;
    if value.is_null() {
        return None;
    }
    // SAFETY: `opt.junk` yields a pointer to a NUL-terminated string owned by
    // libc for the life of the process; it is never mutated or freed.
    unsafe { std::ffi::CStr::from_ptr(value) }.to_str().ok()
}
