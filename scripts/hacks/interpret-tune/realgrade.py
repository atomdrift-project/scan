#!/usr/bin/env python3
"""Grade real `--llm` renders of a labelled corpus under several arms, side by side.

Each arm renders every sample with `atomscan path -f interpret` (its own
environment, so cleave tuning knobs and cache switches apply per arm) and
grades the render N times. All arms are graded in one interleaved batch, so
endpoint drift between runs cannot favor one arm over another — the reason
`shrink.py`'s simulated renders and sequential steps were abandoned.

    realgrade.py --corpus DIR --labels TSV --out DIR -n 4 \\
        'base=' 'four=CLEAVE_TUNE_FILES=4' 'tel=SAME=base;PROMPT=sys.txt'

An arm is `name=KEY=VALUE;KEY=VALUE…`. Keys other than these are environment
variables for its renders:
  PROMPT  system prompt file (default: scan's src/interpret/system_prompt.txt)
  SAME    grade another arm's renders instead of rendering
  EXE     atomscan binary (default: --exe)
Renders are saved under --out/<arm>/; --reuse grades saved renders instead
of rendering again. Transport failures (endpoint down) are retried with a
capped backoff and never count as a grade; a sample still failing after the
retries scores "E" (wrong).
"""
import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune  # noqa: E402  (labels, LLM call)

SCAN = Path(__file__).resolve().parents[3]
EXTRAS = ("PROMPT", "SAME", "EXE")


def parse_arm(spec):
    name, _, rest = spec.partition("=")
    kv = dict(x.split("=", 1) for x in rest.split(";") if x)
    return name, {k: kv.pop(k) for k in EXTRAS if k in kv}, kv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, help="directory for renders and results")
    ap.add_argument("--traits", help="CLEAVE_TRAITS_DIR for every render (pin a frozen copy)")
    ap.add_argument("--exe", default=str(SCAN / "target/release/atomscan"))
    ap.add_argument("--endpoint", default=os.environ.get("SCAN_LLM", "https://llm.isotope13.ai/v1"))
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("-n", type=int, default=4, help="grades per sample per arm")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--render-workers", type=int, default=6)
    ap.add_argument("--reuse", action="store_true", help="grade saved renders where present")
    ap.add_argument("arms", nargs="+")
    args = ap.parse_args()

    arms = [parse_arm(a) for a in args.arms]
    labels = tune.load_labels(args.labels)
    files = sorted(labels)
    out = Path(args.out)
    shipped = (SCAN / "src/interpret/system_prompt.txt").read_text().rstrip()

    def render(extra, env, f):
        e = dict(os.environ, **env)
        if args.traits:
            e["CLEAVE_TRAITS_DIR"] = args.traits
        cmd = [extra.get("EXE", args.exe), "path", "--no-update", "--local",
               "--follow-depth", "0", "-f", "interpret", str(Path(args.corpus) / f)]
        return subprocess.run(cmd, env=e, capture_output=True, text=True).stdout

    renders = {}
    jobs = []
    for name, extra, env in arms:
        if "SAME" in extra:
            continue
        for f in files:
            saved = out / name / f"{f}.txt"
            if args.reuse and saved.exists():
                renders[(name, f)] = saved.read_text()
            else:
                jobs.append((name, extra, env, f))
    with cf.ThreadPoolExecutor(args.render_workers) as ex:
        futs = {ex.submit(render, extra, env, f): (name, f) for name, extra, env, f in jobs}
        for n, fu in enumerate(cf.as_completed(futs), 1):
            renders[futs[fu]] = fu.result()
            if n % 50 == 0 or n == len(futs):
                print(f"rendered {n}/{len(futs)}", file=sys.stderr, flush=True)
    for name, extra, _ in arms:
        if "SAME" in extra:
            for f in files:
                renders[(name, f)] = renders[(extra["SAME"], f)]
        (out / name).mkdir(parents=True, exist_ok=True)
        for f in files:
            (out / name / f"{f}.txt").write_text(renders[(name, f)])
        empty = [f for f in files if not renders[(name, f)].strip()]
        if empty:
            print(f"WARNING {name}: {len(empty)} empty renders: {empty[:5]}", file=sys.stderr, flush=True)

    key = tune.llm_key()
    system = {name: Path(extra["PROMPT"]).read_text().rstrip() if "PROMPT" in extra else shipped
              for name, extra, _ in arms}

    def ask(sysp, user):
        for attempt in range(12):
            try:
                return tune.ask(args.endpoint, args.model, sysp, user, 180, key)
            except Exception:
                time.sleep(min(60, 5 * 2 ** attempt))
        return "E"

    votes = {(name, f): [] for name, _, _ in arms for f in files}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {}
        for i in range(args.n):  # interleave arms and passes so drift hits all alike
            for f in files:
                for name, _, _ in arms:
                    futs[ex.submit(ask, system[name], renders[(name, f)])] = (name, f)
        total, started = len(futs), time.time()
        for n, fu in enumerate(cf.as_completed(futs), 1):
            votes[futs[fu]].append(fu.result())
            if n % 50 == 0 or n == total:
                rate = n / max(1.0, time.time() - started)
                print(f"graded {n}/{total}  {rate * 60:.0f}/min  eta {(total - n) / rate / 60:.0f} min",
                      file=sys.stderr, flush=True)

    lines = []
    for (name, f), v in sorted(votes.items()):
        want = labels[f][0]
        if v.count(want) < len(v):
            lines.append(f"{name:12} {f[:46]:46} want={want} right={v.count(want)}/{len(v)} {''.join(x[0] for x in v)}")
    for name, _, _ in arms:
        exp = sum(votes[(name, f)].count(labels[f][0]) / len(votes[(name, f)]) for f in files)
        maj = sum(votes[(name, f)].count(labels[f][0]) * 2 > len(votes[(name, f)]) for f in files)
        size = sum(len(system[name]) + len(renders[(name, f)]) for f in files)
        lines.append(f"TOTAL {name:12} expected={exp:.2f} majority={maj}/{len(files)} bytes={size}")
    report = "\n".join(lines) + "\n"
    (out / "results.txt").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
