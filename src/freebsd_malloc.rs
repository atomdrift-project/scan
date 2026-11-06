//! FreeBSD's in-libc jemalloc: its live counters, and the settings known to
//! cost this workload.
//!
//! Elsewhere there is no native jemalloc to read or check, and every function
//! here is a no-op.

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

/// Log the allocator's counters and settings at startup.
pub(crate) fn install() {
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

/// Warn about the host allocator settings known to cost this workload.
///
/// Both are warnings, never failures: an operator's override must not be able
/// to stop a server from booting.
///
/// **`MALLOC_CONF` asking for `background_thread:true`** permanently breaks
/// allocation. FreeBSD builds libc's jemalloc without
/// `JEMALLOC_BACKGROUND_THREAD` (libc cannot depend on libthr), so
/// `background_thread_boot0()` fails. It runs from `malloc_init_hard()` after
/// `malloc_init_state` has become `malloc_init_recursible`, so initialization
/// returns early and never reaches `malloc_init_initialized`: every later
/// allocation re-enters `malloc_init_hard()` and serializes on the global
/// `init_lock`, collapsing a many-core host to about one allocating thread.
/// jemalloc treats this as unsupported rather than invalid, so `abort_conf`
/// does not catch it and nothing reaches stderr — throughput death is the only
/// symptom, which is why this check exists.
///
/// **A debug build of libc's jemalloc.** FreeBSD -CURRENT builds it with
/// `--enable-debug` and `--enable-fill`, so every allocation runs jemalloc's
/// invariant assertions and both `malloc` and `free` fill the region
/// (`opt.junk`). Profiles charge that to whatever called `malloc`, so the cost
/// is invisible unless something looks for it, and on allocation-heavy archive
/// scans junk filling alone is a large share of wall time. It is settable at
/// startup — `MALLOC_CONF` for one service, an `/etc/malloc.conf` symlink for
/// the host — so it warns. The assertions are compiled in, cost little beside
/// the filling, and need a `MALLOC_PRODUCTION` world to remove, which no
/// deploy can do; they are reported at info, since a warning repeated on every
/// start with no action behind it teaches people to skip warnings.
pub fn check_settings() {
    #[cfg(target_os = "freebsd")]
    {
        let conf = crate::env::text(crate::env::MALLOC_CONF).unwrap_or_default();
        // Only `background_thread:true` breaks init; an explicit `:false` is fine.
        if conf.contains("background_thread:true") {
            tracing::warn!(
                malloc_conf = %conf,
                "MALLOC_CONF sets background_thread:true, which FreeBSD's in-libc jemalloc does \
                 not support: malloc initialization aborts partway and every allocation then \
                 serializes on jemalloc's global init_lock. Expect near-total throughput \
                 collapse and analyses that never finish. Remove background_thread from \
                 MALLOC_CONF.",
            );
        }

        let is_debug_build = read_ctl::<bool>(b"config.debug\0") == Some(true);
        if let Some(junk) = junk_setting().filter(|setting| *setting != "false") {
            tracing::warn!(
                opt_junk = junk,
                config_debug = is_debug_build,
                "the system jemalloc fills every allocation and free (opt.junk), a cost \
                 charged to whatever called malloc and invisible in any profile. To fix: add \
                 junk:false to this service's MALLOC_CONF, or point an /etc/malloc.conf \
                 symlink at it to cover every process on the host.",
            );
        } else if is_debug_build {
            tracing::info!(
                config_debug = true,
                "the system jemalloc still carries its debug assertions; junk filling, the \
                 costly half, is already off. A MALLOC_PRODUCTION world drops the rest.",
            );
        }
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
