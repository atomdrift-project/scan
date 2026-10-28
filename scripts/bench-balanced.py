#!/usr/bin/env python3
"""Time a complete balanced scan with analysis caches and fetching disabled."""

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import statistics
import sys
import subprocess
import time


def cpu_list(value):
    cpus = set()
    for part in value.split(","):
        bounds = part.split("-")
        start = int(bounds[0])
        end = int(bounds[-1])
        if start < 0 or end < start or len(bounds) > 2:
            raise ValueError(f"Invalid CPU list: {value}")
        cpus.update(range(start, end + 1))
    return cpus


def configure_rizin(args, env):
    """Bake helper settings into a wrapper: filefacts clears helper env vars."""
    if not args.rizin:
        if args.native_cpus:
            raise ValueError("--native-cpus requires --rizin")
        return
    actual = args.rizin.resolve(strict=True)
    if not os.access(actual, os.X_OK):
        raise ValueError(f"Rizin is not executable: {actual}")
    cores = sorted(cpu_list(args.native_cpus)) if args.native_cpus else []
    if cores:
        available = os.sched_getaffinity(0)
        if not set(cores) <= available:
            raise ValueError("Native CPUs must be within the caller's CPU affinity")
        reserved = set(cores)
        for core in cores:
            topology = Path(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list")
            reserved.update(cpu_list(topology.read_text().strip()))
        scanner_cpus = available - reserved
        if not scanner_cpus:
            raise ValueError("Native CPU reservation leaves no CPUs for the scanner")
        os.sched_setaffinity(0, scanner_cpus)
    wrapper = Path(f"{args.output}.rizin").resolve()
    (wrapper / "bin").mkdir(parents=True, exist_ok=True)
    (wrapper / "counter").write_text("0")
    (wrapper / "starts.jsonl").unlink(missing_ok=True)
    code = f"#!{sys.executable}\n" + "import fcntl,json,os,sys,time\n"
    code += f"cores={cores!r}\nactual={str(actual)!r}\nroot={str(wrapper)!r}\n"
    code += '''
try:
    size = os.stat(sys.argv[-1]).st_size
except OSError:
    size = 0
if cores and size >= 16 * 1024 * 1024:
    with open(root + '/counter', 'a+') as counter:
        fcntl.flock(counter, fcntl.LOCK_EX)
        counter.seek(0)
        position = int(counter.read() or '0')
        core = cores[position % len(cores)]
        counter.seek(0)
        counter.truncate()
        counter.write(str(position + 1))
        counter.flush()
    os.sched_setaffinity(0, {core})
    with open(root + '/starts.jsonl', 'a') as log:
        log.write(json.dumps(dict(pid=os.getpid(), start=time.monotonic(), size=size, core=core)) + '\\n')
os.execv(actual, [actual, *sys.argv[1:]])
'''
    executable = wrapper / "bin/rizin"
    executable.write_text(code)
    executable.chmod(0o755)
    env["PATH"] = str(wrapper / "bin") + os.pathsep + env.get("PATH", os.defpath)


def available_memory_bytes():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def zfs_arc_stats():
    fields = {"size", "c_min", "mru_evictable_data", "mfu_evictable_data"}
    try:
        stats = {}
        for line in Path("/proc/spl/kstat/zfs/arcstats").read_text().splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[0] in fields:
                stats[parts[0]] = int(parts[2])
        return stats or None
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("out/atomscan.bench"))
    parser.add_argument("--output", type=Path, default=Path("out/balanced-benchmark"))
    parser.add_argument("--limit", type=float)
    parser.add_argument("--runs", type=int, default=1, help="Require every run to pass the limit and completeness checks")
    parser.add_argument("--rizin", type=Path, help="Use this Rizin executable or scanner-facing wrapper")
    parser.add_argument("--native-cpus", help="Linux CPUs for large native helpers; exclude their SMT siblings from scanner affinity")
    parser.add_argument("--interrupt-after", type=float, help="Interrupt profiling training after this many seconds; interrupted runs always fail the benchmark")
    args = parser.parse_args()
    if not args.path.exists():
        parser.error(f"corpus does not exist: {args.path}")
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.interrupt_after is not None and not 0 < args.interrupt_after < float('inf'):
        parser.error("--interrupt-after must be finite and positive")
    if args.runs > 1:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        measurements = []
        passed = True
        for run in range(1, args.runs + 1):
            prefix = Path(f"{args.output}.run{run}")
            command = [sys.executable, __file__, str(args.path), "--binary", str(args.binary),
                       "--output", str(prefix)]
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            if args.rizin:
                command.extend(["--rizin", str(args.rizin)])
            if args.native_cpus:
                command.extend(["--native-cpus", args.native_cpus])
            if args.interrupt_after is not None:
                command.extend(["--interrupt-after", str(args.interrupt_after)])
            measurement_path = Path(f"{prefix}.time.json")
            measurement_path.unlink(missing_ok=True)
            result = subprocess.run(command, check=False, stdout=subprocess.DEVNULL)
            passed = passed and result.returncode == 0
            if not measurement_path.exists():
                raise SystemExit(f"Run {run} produced no measurement (exit {result.returncode})")
            measurement = json.loads(measurement_path.read_text())
            measurements.append(measurement)
            print(json.dumps(dict(run=run, wall_s=measurement["wall_s"],
                                  output_records=measurement["output_records"],
                                  error_records=measurement["error_records"],
                                  passed=result.returncode == 0)), flush=True)
        times = [m["wall_s"] for m in measurements]
        summary = dict(runs=args.runs, limit_s=args.limit, all_passed=passed,
                       min_s=min(times), median_s=statistics.median(times), max_s=max(times),
                       measurements=measurements)
        Path(f"{args.output}.series.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k:v for k,v in summary.items() if k != "measurements"}))
        return int(not passed)
    files = [p for p in args.path.rglob("*") if p.is_file()] if args.path.is_dir() else [args.path]
    total_bytes = sum(p.stat().st_size for p in files)
    env = dict(os.environ)
    env.update(
        SCAN_NO_ANALYSIS_CACHE="1",
        SCAN_ANALYSIS_CACHE="0",
        FILEFACTS_CACHE="0",
        STNG_STRING_CACHE="0",
        CLEAVE_SKIP_CACHE="1",
        CLEAVE_SKIP_YARA_CACHE="0",
        SCAN_NO_UPDATE_CHECK="1",
        SCAN_FOLLOW="none",
        SCAN_FOLLOW="none",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        configure_rizin(args, env)
    except (OSError, ValueError, AttributeError) as error:
        parser.error(str(error))
    command = [str(args.binary.resolve()), "--mode", "balanced", "-f", "json", str(args.path)]
    interrupted = False
    with Path(f"{args.output}.jsonl").open("w") as out, Path(f"{args.output}.log").open("w") as err:
        memory_start = available_memory_bytes()
        arc_start = zfs_arc_stats()
        start = time.monotonic()
        with subprocess.Popen(command, env=env, stdout=out, stderr=err) as child:
            try:
                exit_code = child.wait(timeout=args.interrupt_after)
            except subprocess.TimeoutExpired:
                interrupted = True
                # The scanner's first Ctrl-C cancels pending work. Its second
                # uses process::exit, allowing LLVM's atexit writer to flush
                # profiling counters even if a current member takes too long.
                child.send_signal(signal.SIGINT)
                try:
                    exit_code = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.send_signal(signal.SIGINT)
                    try:
                        exit_code = child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        exit_code = child.wait()
        elapsed = time.monotonic() - start
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    # Validate completeness outside the timed interval. A hostile-file exit
    # code can coexist with errors, so the process status alone is insufficient.
    records = errors = skipped = invalid = 0
    with Path(f"{args.output}.jsonl").open() as output:
        for line in output:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            records += 1
            errors += bool(record.get("err"))
            skipped += record.get("scanned") is False
    measurement = dict(
        command=command, files=len(files), bytes=total_bytes,
        output_records=records, error_records=errors, skipped_records=skipped,
        invalid_records=invalid,
        wall_s=elapsed, user_s=usage.ru_utime, system_s=usage.ru_stime,
        max_child_rss_bytes=usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        start_monotonic_s=start,
        mib_per_s=total_bytes / 1048576 / elapsed, exit_code=exit_code,
        interrupted=interrupted,
        analysis_cache=False, follow="none", limit_s=args.limit,
        rayon_threads=env.get("CLEAVE_RAYON_THREADS") or "default",
        scan_lanes=env.get("CLEAVE_SCAN_LANES") or "default",
        inner_parallel_owners=env.get("CLEAVE_INNER_PARALLEL_OWNERS", "default"),
        ast_batch=env.get("CLEAVE_AST_BATCH", "1"),
        ast_batch_guarded=env.get("CLEAVE_AST_BATCH_GUARDED", "0"),
        ast_loop_gate=env.get("CLEAVE_AST_LOOP_GATE", "0"),
        ast_batch_cpu_ms=env.get("CLEAVE_AST_BATCH_CPU_MS", "default"),
        serial_traits="CLEAVE_SERIAL_TRAITS" in env,
        member_window_count=env.get("CLEAVE_MEMBER_WINDOW_COUNT", "default"),
        member_window_mb=env.get("CLEAVE_MEMBER_WINDOW_MB", "default"),
        pipeline_member_windows=env.get("CLEAVE_PIPELINE_MEMBER_WINDOWS", "0"),
        heavy_member_parallelism=env.get("CLEAVE_HEAVY_MEMBER_PARALLEL", "default"),
        priority_heavy_members=env.get("CLEAVE_PRIORITY_HEAVY_MEMBERS", "0"),
        native_facts_threads=env.get("CLEAVE_NATIVE_FACTS_THREADS", "0"),
        native_facts_isolated=env.get("CLEAVE_NATIVE_FACTS_ISOLATED", "0"),
        native_member_threads=env.get("CLEAVE_NATIVE_MEMBER_THREADS", "0"),
        source_early_dispatch=env.get("CLEAVE_SOURCE_EARLY_DISPATCH", "0"),
        source_leaf_threads=env.get("CLEAVE_SOURCE_LEAF_THREADS", "0"),
        leaf_cpus=env.get("CLEAVE_LEAF_CPUS", "inherited"),
        source_leaf_min_kb=env.get("CLEAVE_SOURCE_LEAF_MIN_KB", "4096"),
        source_flight_wait=env.get("CLEAVE_SOURCE_FLIGHT_WAIT", "0"),
        analysis_memo_mb=env.get("CLEAVE_ANALYSIS_MEMO_MB", "64"),
        model_memo_mb=env.get("SCAN_MODEL_MEMO_MB", "0"),
        jemalloc_conf=env.get("_RJEM_MALLOC_CONF", "default"),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        member_pressure_floor_mb=env.get("CLEAVE_MEMBER_PRESSURE_FLOOR_MB", "default"),
        memgate_expansion=env.get("CLEAVE_MEMGATE_EXPANSION", "default"),
        raw_gate_source_max_kb=env.get("CLEAVE_RAW_GATE_SOURCE_MAX_KB", "default"),
        rizin_cores=args.native_cpus,
        rizin_executable=str(args.rizin.resolve()) if args.rizin else None,
        rizin=shutil.which("rizin", path=env.get("PATH")),
        available_memory_start_bytes=memory_start,
        available_memory_end_bytes=available_memory_bytes(),
        zfs_arc_start=arc_start,
        zfs_arc_end=zfs_arc_stats(),
    )
    Path(f"{args.output}.time.json").write_text(json.dumps(measurement, indent=2) + "\n")
    print(json.dumps(measurement, indent=2))
    if interrupted or exit_code not in (0, 1, 2) or records != len(files) or errors or skipped or invalid:
        return 1
    return int(args.limit is not None and elapsed > args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
