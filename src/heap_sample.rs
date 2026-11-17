//! Sampling live-heap profiler for Windows (feature `heap-sample`).
//!
//! jemalloc's profiler covers the unix builds; mimalloc has none. This wraps
//! the global allocator and samples roughly one allocation per
//! [`INTERVAL`] bytes allocated on each thread, recording its call stack. A
//! sampled block is forgotten when freed, so at any instant the table holds an
//! unbiased sample of the *live* heap by call stack — the question a peak-RSS
//! investigation needs answered, which allocation counters cannot.
//!
//! Set `SCAN_HEAP_SAMPLE=<file>` to append a report of the heaviest stacks
//! every `SCAN_HEAP_SAMPLE_SECS` seconds (default 30). Build the `profiling`
//! profile so frames resolve to names.
//!
//! Not counted: memory that bypasses the Rust allocator — tree-sitter's C
//! allocations, anonymous mappings, wasmtime's code and linear memories.

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::collections::HashMap;
use std::io::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// Mean bytes allocated per thread between samples.
const INTERVAL: usize = 256 * 1024;
/// Frames kept per stack.
const DEPTH: usize = 40;
/// Slots in the "might be sampled" filter consulted on every free.
const FILTER_SLOTS: usize = 1 << 22;

type Stack = [usize; DEPTH];

thread_local! {
    static COUNTDOWN: Cell<isize> = const { Cell::new(INTERVAL as isize) };
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

static FILTER: [AtomicU8; FILTER_SLOTS] = [const { AtomicU8::new(0) }; FILTER_SLOTS];
static TRACKED_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Of [`TRACKED_BYTES`], the part in blocks under 32 KiB (mimalloc's share).
static TRACKED_SMALL: AtomicUsize = AtomicUsize::new(0);
const SMALL: usize = 32 * 1024;

struct Table {
    /// Sampled live block -> (stack index, weight in bytes).
    live: HashMap<usize, (u32, usize, bool)>,
    index: HashMap<Stack, u32>,
    stacks: Vec<Stack>,
    /// Per stack: live weighted bytes, live samples.
    totals: Vec<(usize, usize)>,
}

static TABLE: Mutex<Option<Table>> = Mutex::new(None);

unsafe extern "system" {
    fn RtlCaptureStackBackTrace(
        skip: u32,
        count: u32,
        frames: *mut *mut std::ffi::c_void,
        hash: *mut u32,
    ) -> u16;
}

fn slot(ptr: usize) -> usize {
    (ptr >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (usize::BITS - 22)
}

/// Run `f` with sampling suppressed on this thread, so the profiler's own
/// allocations are neither sampled nor re-enter the table lock.
fn guarded(f: impl FnOnce()) {
    let entered = BUSY.try_with(|busy| !busy.replace(true)).unwrap_or(false);
    if entered {
        f();
        let _ = BUSY.try_with(|busy| busy.set(false));
    }
}

fn record(ptr: usize, size: usize) {
    guarded(|| {
        let mut stack: Stack = [0; DEPTH];
        // SAFETY: `stack` has room for DEPTH pointers, and a null hash
        // out-pointer is allowed.
        unsafe {
            RtlCaptureStackBackTrace(
                2,
                DEPTH as u32,
                stack.as_mut_ptr().cast(),
                std::ptr::null_mut(),
            );
        }
        let weight = size.max(INTERVAL);
        let mut guard = TABLE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let table = guard.get_or_insert_with(|| Table {
            live: HashMap::new(),
            index: HashMap::new(),
            stacks: Vec::new(),
            totals: Vec::new(),
        });
        let next = u32::try_from(table.stacks.len()).unwrap_or(u32::MAX);
        let idx = *table.index.entry(stack).or_insert(next);
        if idx == next {
            table.stacks.push(stack);
            table.totals.push((0, 0));
        }
        if let Some(total) = table.totals.get_mut(idx as usize) {
            total.0 += weight;
            total.1 += 1;
        }
        let small = size < SMALL;
        if small {
            TRACKED_SMALL.fetch_add(weight, Ordering::Relaxed);
        }
        if let Some((old_idx, old_weight, old_small)) = table.live.insert(ptr, (idx, weight, small))
        {
            if old_small {
                TRACKED_SMALL.fetch_sub(old_weight, Ordering::Relaxed);
            }
            // A free we missed (it ran while this thread was inside the
            // profiler): retire the stale entry.
            if let Some(total) = table.totals.get_mut(old_idx as usize) {
                total.0 = total.0.saturating_sub(old_weight);
                total.1 = total.1.saturating_sub(1);
            }
            TRACKED_BYTES.fetch_sub(old_weight, Ordering::Relaxed);
        } else {
            FILTER[slot(ptr)].fetch_add(1, Ordering::Relaxed);
        }
        TRACKED_BYTES.fetch_add(weight, Ordering::Relaxed);
    });
}

/// Stop tracking `ptr`; whether it was a sampled block.
fn forget(ptr: usize) -> bool {
    if FILTER[slot(ptr)].load(Ordering::Relaxed) == 0 {
        return false;
    }
    let mut was_tracked = false;
    guarded(|| {
        let mut guard = TABLE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(table) = guard.as_mut() else { return };
        if let Some((idx, weight, small)) = table.live.remove(&ptr) {
            if small {
                TRACKED_SMALL.fetch_sub(weight, Ordering::Relaxed);
            }
            if let Some(total) = table.totals.get_mut(idx as usize) {
                total.0 = total.0.saturating_sub(weight);
                total.1 = total.1.saturating_sub(1);
            }
            FILTER[slot(ptr)].fetch_sub(1, Ordering::Relaxed);
            TRACKED_BYTES.fetch_sub(weight, Ordering::Relaxed);
            was_tracked = true;
        }
    });
    was_tracked
}

/// Count `charged` bytes against this thread's sampling interval and, when
/// it runs out, sample the `size`-byte block at `ptr`.
fn on_alloc(ptr: *mut u8, size: usize, charged: usize) {
    if ptr.is_null() {
        return;
    }
    let due = COUNTDOWN
        .try_with(|c| {
            let left = c.get() - charged.cast_signed();
            if left <= 0 {
                c.set(INTERVAL.cast_signed());
                true
            } else {
                c.set(left);
                false
            }
        })
        .unwrap_or(false);
    if due {
        record(ptr as usize, size);
    }
}

/// A [`GlobalAlloc`] that forwards to `A` and samples what stays live.
#[derive(Debug)]
pub struct Sampling<A>(pub A);

// SAFETY: every call forwards unchanged to the wrapped allocator; the
// bookkeeping around it touches only this module's own state.
unsafe impl<A: GlobalAlloc> GlobalAlloc for Sampling<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded under the caller's contract.
        let ptr = unsafe { self.0.alloc(layout) };
        on_alloc(ptr, layout.size(), layout.size());
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded under the caller's contract.
        let ptr = unsafe { self.0.alloc_zeroed(layout) };
        on_alloc(ptr, layout.size(), layout.size());
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = forget(ptr as usize);
        // SAFETY: forwarded under the caller's contract.
        unsafe { self.0.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let was_tracked = forget(ptr as usize);
        // SAFETY: forwarded under the caller's contract.
        let new = unsafe { self.0.realloc(ptr, layout, new_size) };
        if was_tracked {
            // A sampled block stays sampled at its new size, so a buffer that
            // doubles its way to a gigabyte is charged the whole gigabyte.
            if !new.is_null() {
                record(new as usize, new_size);
            }
        } else {
            // Only the growth is new allocation; when it trips the interval
            // the whole block is what is live.
            on_alloc(new, new_size, new_size.saturating_sub(layout.size()).max(1));
        }
        new
    }
}

/// `mi_malloc`, sampled: for C code (tree-sitter) that takes an allocator.
///
/// # Safety
/// The C `malloc` contract.
pub unsafe extern "C" fn c_malloc(size: usize) -> *mut std::ffi::c_void {
    // SAFETY: mimalloc's malloc accepts any size.
    let ptr = unsafe { libmimalloc_sys::mi_malloc(size) };
    on_alloc(ptr.cast(), size, size);
    ptr
}

/// `mi_calloc`, sampled.
///
/// # Safety
/// The C `calloc` contract.
pub unsafe extern "C" fn c_calloc(count: usize, size: usize) -> *mut std::ffi::c_void {
    // SAFETY: mimalloc's calloc accepts any count and size.
    let ptr = unsafe { libmimalloc_sys::mi_calloc(count, size) };
    let bytes = count.saturating_mul(size);
    on_alloc(ptr.cast(), bytes, bytes);
    ptr
}

/// `mi_realloc`, sampled.
///
/// # Safety
/// The C `realloc` contract.
pub unsafe extern "C" fn c_realloc(
    ptr: *mut std::ffi::c_void,
    size: usize,
) -> *mut std::ffi::c_void {
    let was_tracked = forget(ptr as usize);
    // SAFETY: forwarded under the caller's contract.
    let new = unsafe { libmimalloc_sys::mi_realloc(ptr, size) };
    if was_tracked {
        if !new.is_null() {
            record(new as usize, size);
        }
    } else {
        // The old size is unknown here; charge half, the mean of a doubling.
        on_alloc(new.cast(), size, (size / 2).max(1));
    }
    new
}

/// `mi_free`, sampled.
///
/// # Safety
/// The C `free` contract.
pub unsafe extern "C" fn c_free(ptr: *mut std::ffi::c_void) {
    let _ = forget(ptr as usize);
    // SAFETY: forwarded under the caller's contract.
    unsafe { libmimalloc_sys::mi_free(ptr) }
}

/// Start the reporting thread when `SCAN_HEAP_SAMPLE` names a file.
#[expect(
    clippy::disallowed_methods,
    reason = "diagnostic switches of a profiling-only feature, not configuration"
)]
pub fn start() {
    let Some(path) = std::env::var_os("SCAN_HEAP_SAMPLE") else {
        return;
    };
    let period = std::env::var("SCAN_HEAP_SAMPLE_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(30)
        .max(1);
    let spawned = std::thread::Builder::new()
        .name("heap-sample".into())
        .spawn(move || {
            let started = std::time::Instant::now();
            let mut names: HashMap<usize, String> = HashMap::new();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(period));
                guarded(|| report(&path, started.elapsed().as_secs(), &mut names));
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "cannot start the heap-sample reporter");
    }
}

fn report(path: &std::ffi::OsStr, at_secs: u64, names: &mut HashMap<usize, String>) {
    const TOP: usize = 6000;
    let (mut top, stacks_seen) = {
        let guard = TABLE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(table) = guard.as_ref() else { return };
        let mut rows: Vec<(usize, usize, Stack)> = table
            .totals
            .iter()
            .zip(&table.stacks)
            .filter(|(total, _)| total.0 > 0)
            .map(|(total, stack)| (total.0, total.1, *stack))
            .collect();
        let seen = rows.len();
        rows.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        rows.truncate(TOP);
        (rows, seen)
    };
    let Ok(mut out) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    else {
        return;
    };
    let total = TRACKED_BYTES.load(Ordering::Relaxed);
    let _ = writeln!(
        out,
        "==== t={at_secs}s sampled_live={} MB live_stacks={stacks_seen} small_live={} MB",
        total / (1024 * 1024),
        TRACKED_SMALL.load(Ordering::Relaxed) / (1024 * 1024)
    );
    mimalloc_stats(&mut out);
    address_space(&mut out);
    for (bytes, samples, stack) in top.drain(..) {
        let _ = writeln!(out, "-- {} KB in {samples} samples", bytes / 1024);
        for &ip in stack.iter().take_while(|&&ip| ip != 0) {
            let name = names.entry(ip).or_insert_with(|| {
                let mut resolved = String::new();
                backtrace::resolve(ip as *mut std::ffi::c_void, |symbol| {
                    if resolved.is_empty()
                        && let Some(name) = symbol.name()
                    {
                        resolved = format!("{name:#}");
                    }
                });
                if resolved.is_empty() {
                    resolved = format!("{ip:#x}");
                }
                resolved
            });
            let _ = writeln!(out, "   {name}");
        }
    }
}

/// Append mimalloc's own view (committed, reserved, peak) to the report: the
/// gap between its committed bytes and the sampled live bytes is allocator
/// retention plus the C allocations that bypass the sampler.
fn mimalloc_stats(out: &mut std::fs::File) {
    unsafe extern "C" fn sink(msg: *const std::ffi::c_char, arg: *mut std::ffi::c_void) {
        if msg.is_null() || arg.is_null() {
            return;
        }
        // SAFETY: `arg` is the `&mut File` passed below, alive for the whole
        // `mi_stats_print_out` call; `msg` is a NUL-terminated C string.
        unsafe {
            let file = &mut *arg.cast::<std::fs::File>();
            let _ = file.write_all(std::ffi::CStr::from_ptr(msg).to_bytes());
        }
    }
    let (mut rss, mut peak_rss, mut commit, mut peak_commit) = (0usize, 0usize, 0usize, 0usize);
    // SAFETY: every out-pointer is either null (allowed) or a live local.
    unsafe {
        libmimalloc_sys::mi_process_info(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &raw mut rss,
            &raw mut peak_rss,
            &raw mut commit,
            &raw mut peak_commit,
            std::ptr::null_mut(),
        );
    }
    const MIB: usize = 1024 * 1024;
    let _ = writeln!(
        out,
        "process: rss={} MB peak_rss={} MB commit={} MB peak_commit={} MB",
        rss / MIB,
        peak_rss / MIB,
        commit / MIB,
        peak_commit / MIB
    );
    // SAFETY: `sink` matches `mi_output_fun`, and `out` outlives the call.
    unsafe {
        libmimalloc_sys::mi_stats_print_out(Some(sink), std::ptr::from_mut(out).cast());
    }
}

#[repr(C)]
struct MemoryBasicInformation {
    base_address: usize,
    allocation_base: usize,
    allocation_protect: u32,
    partition_id: u16,
    region_size: usize,
    state: u32,
    protect: u32,
    kind: u32,
}

unsafe extern "system" {
    fn VirtualQuery(address: usize, info: *mut MemoryBasicInformation, len: usize) -> usize;
    fn GetModuleHandleW(name: *const u16) -> usize;
}

/// Append where the process's committed private memory sits, bucketed by the
/// size of the reservation it belongs to: mimalloc's arenas are the gigabyte
/// reservations, thread stacks the 256 MiB ones, and anonymous file snapshots
/// and wasmtime memories everything between.
fn address_space(out: &mut std::fs::File) {
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_PRIVATE: u32 = 0x20000;
    const MIB: usize = 1024 * 1024;
    // (reservation size, committed private bytes) per allocation base.
    let mut regions: Vec<(usize, usize, usize)> = Vec::new();
    let mut other_commit = 0usize;
    let mut address = 0x10000usize;
    loop {
        let mut info = MemoryBasicInformation {
            base_address: 0,
            allocation_base: 0,
            allocation_protect: 0,
            partition_id: 0,
            region_size: 0,
            state: 0,
            protect: 0,
            kind: 0,
        };
        // SAFETY: `info` is a live local of the size passed.
        let got = unsafe {
            VirtualQuery(
                address,
                &raw mut info,
                std::mem::size_of::<MemoryBasicInformation>(),
            )
        };
        if got == 0 || info.region_size == 0 {
            break;
        }
        if info.allocation_base != 0 {
            let committed = if info.state == MEM_COMMIT {
                info.region_size
            } else {
                0
            };
            if info.kind == MEM_PRIVATE {
                match regions.last_mut() {
                    Some(last) if last.0 == info.allocation_base => {
                        last.1 += info.region_size;
                        last.2 += committed;
                    }
                    _ => regions.push((info.allocation_base, info.region_size, committed)),
                }
            } else {
                other_commit += committed;
            }
        }
        let Some(next) = info.base_address.checked_add(info.region_size) else {
            break;
        };
        address = next;
    }
    // Buckets by reservation size.
    let bounds = [1usize, 4, 16, 64, 255, 257, 1023, usize::MAX / MIB];
    let labels = [
        "<1M",
        "1-4M",
        "4-16M",
        "16-64M",
        "64-255M",
        "256M(stacks)",
        "257M-1G",
        ">=1G(arenas)",
    ];
    let mut buckets = [(0usize, 0usize, 0usize); 8];
    for (_, reserved, committed) in &regions {
        let mb = reserved / MIB;
        let idx = bounds.iter().position(|&b| mb < b).unwrap_or(7);
        buckets[idx].0 += 1;
        buckets[idx].1 += reserved;
        buckets[idx].2 += committed;
    }
    // SAFETY: a null name asks for the main module's base.
    let image = unsafe { GetModuleHandleW(std::ptr::null()) };
    let _ = writeln!(out, "image_base={image:#x}");
    let _ = writeln!(
        out,
        "private commit by reservation size (mapped/image commit: {} MB):",
        other_commit / MIB
    );
    for (label, (count, reserved, committed)) in labels.iter().zip(buckets) {
        let _ = writeln!(
            out,
            "  {label:>14}: regions={count:<6} reserved={:<8} MB committed={} MB",
            reserved / MIB,
            committed / MIB
        );
    }
}
