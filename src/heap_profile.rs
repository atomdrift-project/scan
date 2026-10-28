//! Live jemalloc counters.
//!
//! FreeBSD uses jemalloc in libc, so this module talks to the native allocator
//! through its public `mallctl` interface; elsewhere there is nothing to read.

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
pub fn install() {
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

/// Report what the host's in-libc jemalloc is charging for every allocation.
///
/// FreeBSD builds libc's jemalloc with `--enable-debug` and `--enable-fill` on
/// -CURRENT, so each allocation runs jemalloc's invariant assertions and both
/// `malloc` and `free` memset the region (`opt.junk`). Neither shows up as
/// application time: profiles blame whatever called `malloc`, so the tax is
/// invisible unless something looks for it. On a many-core arm64 -CURRENT host
/// scanning large nested archives, junk filling costs about a fifth of wall
/// time, and `junk:false` alone recovers what swapping in a different allocator
/// (mimalloc) would, at an identical result hash. That is why there is no
/// bundled allocator on this platform.
///
/// The two halves are not the same size and do not have the same remedy, so
/// they are not the same log line. Junk filling is the expensive one and is
/// settable at startup, by `MALLOC_CONF` for one service or by an
/// `/etc/malloc.conf` symlink for every process on the host — that one warns.
/// The assertions are compiled in and need a `MALLOC_PRODUCTION` world, which
/// is nothing a deploy can do; they are worth little beside the filling, so
/// they are reported at info and left alone.
pub fn report_debug_allocator() {
    #[cfg(target_os = "freebsd")]
    {
        let is_debug_build = read_ctl::<bool>(b"config.debug\0") == Some(true);
        if let Some(junk) = junk_setting().filter(|setting| *setting != "false") {
            tracing::warn!(
                opt_junk = junk,
                config_debug = is_debug_build,
                "the system jemalloc fills every allocation and free (opt.junk), a cost \
                 charged to whatever called malloc and invisible in any profile. Measured \
                 on a 128-core arm64 -CURRENT host: 20% of wall-clock. To fix: add \
                 junk:false to this service's MALLOC_CONF, or point an /etc/malloc.conf \
                 symlink at it to cover every process on the host.",
            );
        } else if is_debug_build {
            // Deliberately not a warning. Junk filling is off, which is the
            // half that cost 20%; what is left is the assertions, and the
            // measurement in this function's docs bounds them: with them still
            // compiled in, junk:false scored 227.7 s where mimalloc — which
            // carries no jemalloc assertions at all — scored 231.2 s. An
            // operator cannot drop them without rebuilding the host, and a
            // warning that repeats on every start with no action behind it is
            // one that teaches people to skip warnings.
            tracing::info!(
                config_debug = true,
                "the system jemalloc still carries its debug assertions; junk filling, the \
                 costly half, is already off. A MALLOC_PRODUCTION world drops the rest.",
            );
        }
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
