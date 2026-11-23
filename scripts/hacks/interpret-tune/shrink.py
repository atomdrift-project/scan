#!/usr/bin/env python3
"""Find the shortest interpret prompt that grades a labelled corpus no worse.

Greedy descent over render knobs that cleave/scan could implement — context
lines per finding, the severity floor for a window, files shown, code-line
width, annotations, the `...` cut markers — plus the system prompt. It starts
from the current render, prices every one-step reduction in bytes (no LLM
call), tries the largest saving first, and accepts a step only when no file
that grades correctly starts grading wrong. It stops when no step is accepted,
then re-grades the winner to rule out a lucky pass.

Input renders come from `tune.py --capture-format tiny` (the interpret render
with severity letters and trait ids still on, before scan's rewrite); each
candidate is rebuilt into exactly what `--llm` sends: reduced, recategorized,
and headed `== PRIMARY <name> ==`.

    shrink.py --corpus DIR --labels TSV --dump-dir DIR [--system NAME=FILE ...]
"""
import argparse, json, os, re, sys
import concurrent.futures as cf
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune  # noqa: E402  (labels, corpus, recategorize, LLM call)

ANNOT = re.compile(r"^\s*(//|#|--) ([HSNBCF])\b")
RANK = {"F": 0, "C": 1, "B": 2, "N": 3, "S": 4, "H": 5}
HEADER = re.compile(r"^\s*\S.*\t\S+ [\d.]+[KMG]?B (\d+)")
# Identity rows: a keyword, then two spaces (`claims  version=…`).
IDENTITY = re.compile(r"^\s*(claims|signer|about|links|ids|contact|build|signed|cert)  ")

# Each knob's steps, longest render first. A step's value is what `reduce` reads.
KNOBS = {
    "bigctx": ["all", 2, 1, 0],       # ...but only in files over BIG bytes
    "ctx": ["all", 2, 1, 0],          # code lines kept after each finding
    "sev": ["C", "B", "N", "S"],      # lowest finding grade that keeps a window
    "files": [5, 4, 3, 2],            # file blocks kept, highest score first
    "width": [0, 200, 120],           # code-line cut (0 = none)
    "ann": [1, 0],                    # finding annotations drawn
    "wincap": ["all", 12, 8, 6, 4, 3, 2],  # windows kept per file, strongest first
    "dots": [1, 2, 0],                # `...` rows: all kept, bare context dropped, all dropped
}


# A file block this large has its context cut by `bigctx`; smaller ones are
# cheap to show whole, and a short file's whole text is often the evidence.
BIG = 4096


def reduce(tiny, name, cfg):
    """The interpret prompt for `tiny` under knob settings `cfg`."""
    lines = tiny.split("\n")
    if lines and lines[0].startswith("scan "):  # the tiny verdict line
        lines = lines[1:]
    # Split into file blocks: a header line and everything up to the next.
    blocks, cur = [], None
    for ln in lines:
        m = HEADER.match(ln)
        if m and not ANNOT.match(ln):
            cur = {"head": ln, "score": int(m.group(1)), "body": []}
            blocks.append(cur)
        elif cur is not None:
            cur["body"].append(ln)
    floor = RANK[cfg["sev"]]
    out_blocks = []
    for b in blocks:
        body, keep = b["body"], []
        ctx = cfg["ctx"]
        if cfg.get("bigctx", "all") != "all" and sum(len(l) + 1 for l in body) > cfg.get("big", BIG):
            ctx = cfg["bigctx"] if ctx == "all" else min(ctx, cfg["bigctx"])
        ident = [l for l in body if IDENTITY.match(l)]
        # Windows are separated by blank lines. Identity rows stay where
        # cleave put them and are always kept.
        win, wins = [], []
        for l in body:
            if not l.strip():
                if win:
                    wins.append(win)
                win = []
            else:
                win.append(l)
        if win:
            wins.append(win)
        for w in wins:
            anns = [l for l in w if ANNOT.match(l)]
            if anns and max(RANK[ANNOT.match(l).group(2)] for l in anns) < floor:
                w = [l for l in w if IDENTITY.match(l)]
                if not w:
                    continue
            kept, after = [], None  # lines since the last annotation group
            for l in w:
                if IDENTITY.match(l):
                    kept.append(l)
                    continue
                m = ANNOT.match(l)
                if m:
                    if RANK[m.group(2)] >= floor and cfg["ann"]:
                        kept.append(l)
                    after = 0 if RANK[m.group(2)] >= floor else None
                    continue
                # dots=0 drops every mid-line row; dots=2 only bare context
                # rows (a finding's own row, right after its annotation, stays).
                if l.lstrip().startswith("...") and (cfg["dots"] == 0 or (cfg["dots"] == 2 and after != 0)):
                    continue
                if ctx == "all" or (after is not None and after <= ctx):
                    kept.append(l[: cfg["width"]] if cfg["width"] else l)
                if after is not None:
                    after += 1
            if any(not ANNOT.match(l) for l in kept) or (kept and cfg["ann"]):
                keep.append(kept)
        # Keep the file's `wincap` strongest windows (by their top annotation),
        # in file order; identity-only windows are not windows and always stay.
        cap = cfg.get("wincap", "all")
        if cap != "all":
            strength = lambda k: max((RANK[ANNOT.match(l).group(2)] for l in k if ANNOT.match(l)), default=-1)
            code_wins = [i for i, k in enumerate(keep) if any(not IDENTITY.match(l) for l in k)]
            top = set(sorted(code_wins, key=lambda i: (-strength(keep[i]), i))[:cap])
            keep = [k for i, k in enumerate(keep) if i in top or i not in code_wins]
        has_code = any(not IDENTITY.match(l) for k in keep for l in k)
        # A block with no window stays when it says what the file is: identity
        # rows, or an identity field in the header line itself.
        if not keep and not ident and b["head"].count("\t") < 2:
            continue
        text = [b["head"]]
        for k in keep:
            text += k + [""]
        if not keep:
            text.append("")
        out_blocks.append((b["score"], len(out_blocks), "\n".join(text), has_code))
    # Identity-only blocks (the root) always stay; code blocks are capped by
    # score, as cleave caps its context files.
    code = [x for x in out_blocks if x[3]]
    # cleave already chose these (its cap excludes the root); only cut further.
    if cfg["files"] < KNOBS["files"][0] and cfg["files"] < len(code):
        code = sorted(code, key=lambda x: -x[0])[: cfg["files"]]
    chosen = sorted([x for x in out_blocks if not x[3]] + code, key=lambda x: x[1])
    # scan's `sanitize_context` rewrites every `!!` to `/` (the archive
    # delimiter); mirror it so the prompt is byte-for-byte what scan sends.
    body = "\n".join(t for _, _, t, _ in chosen).replace("!!", "/")
    return f"== PRIMARY {name} ==\n" + tune.transform(body, "categorized").rstrip("\n") + "\n"


def grade_all(args, renders, labels, cfg, system, runs, key):
    jobs = [(rel, r) for rel in renders for r in range(runs)]
    prompts = {rel: reduce(renders[rel], rel, cfg) for rel in renders}
    votes = {rel: [] for rel in renders}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(tune.ask, args.endpoint, args.model, system, prompts[rel], args.timeout, key): rel
                for rel, _ in jobs}
        for f in cf.as_completed(futs):
            try:
                votes[futs[f]].append(f.result())
            except Exception:
                votes[futs[f]].append("E")
    correct = {rel for rel, v in votes.items()
               if v.count(labels[rel][0]) * 2 > len(v)}
    size = sum(len(system) + len(p) for p in prompts.values())
    return correct, size, votes, prompts


def try_step(args, renders, labels, cfg, system, must, fragile, key):
    """Whether every file in `must` still grades right under `cfg`, stopping
    at the first that does not.

    Each file is asked once; a wrong answer earns more asks, up to a majority
    of three, so one noisy reply cannot reject a step and one lucky reply
    cannot pass a file that is really lost. Files that failed past steps are
    asked first, so a doomed step usually ends after a handful of calls.
    Returns `(lost, size)`; `lost` is empty when the step holds.
    """
    prompts = {rel: reduce(renders[rel], rel, cfg) for rel in must}
    size = sum(len(system) + len(reduce(renders[r], r, cfg)) for r in renders)
    order = sorted(must, key=lambda r: -fragile.get(r, 0))
    want = {rel: labels[rel][0] for rel in must}
    right = {rel: 0 for rel in must}
    wrong = {rel: 0 for rel in must}
    ex = cf.ThreadPoolExecutor(max_workers=args.workers)
    ask = lambda rel: ex.submit(tune.ask, args.endpoint, args.model, system, prompts[rel], args.timeout, key)
    pending = {ask(rel): rel for rel in order}
    lost = set()
    try:
        while pending and not lost:
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in done:
                rel = pending.pop(f)
                try:
                    g = f.result()
                except Exception:
                    g = "E"
                if g == want[rel]:
                    right[rel] += 1
                else:
                    wrong[rel] += 1
                if wrong[rel] and right[rel] < 2 and wrong[rel] < 2:
                    pending[ask(rel)] = rel  # one bad reply: ask again
                elif wrong[rel] >= 2:
                    lost.add(rel)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return lost, size


SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(`])")


def sentences(prompt):
    """The prompt's sentences, in order."""
    return SENTENCE.split(prompt.strip())


def describe(cfg):
    return ",".join(f"{k}={v}" for k, v in cfg.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dump-dir", required=True, help="tiny renders from tune.py --capture-format tiny")
    ap.add_argument("--endpoint", default=os.environ.get("SCAN_LLM", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("SCAN_LLM_MODEL", "Qwen/Qwen3.6-27B"))
    ap.add_argument("--system", action="append", default=[],
                    help="NAME=FILE system prompt steps, longest first (default: the shipped prompt only)")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--baseline-runs", type=int, default=4,
                    help="grades per file for the stability baseline and the final confirmation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--log", default="shrink.jsonl")
    ap.add_argument("--sentences", action="store_true",
                    help="also try dropping each unprotected sentence of the shipped prompt")
    ap.add_argument("--protect", default="0,12,13,15",
                    help="shipped-prompt sentence indexes never dropped (role, injection guard, reply format)")
    ap.add_argument("--start-from", help="resume from the last confirmed winner in this log")
    args = ap.parse_args()

    labels = tune.load_labels(args.labels)
    renders = {r: t for r, t in tune.load_renders(args.dump_dir, args.corpus).items() if r in labels}
    systems = [("shipped", tune.DESCRIBED_PROMPT)] + [
        (n, Path(f).read_text(encoding="utf-8").rstrip()) for n, f in (s.split("=", 1) for s in args.system)]
    key = tune.llm_key()
    log = open(args.log, "a", encoding="utf-8", buffering=1)
    shipped_sentences = sentences(tune.DESCRIBED_PROMPT)
    protect = {int(i) for i in args.protect.split(",") if i}

    def system_for(c):
        """The system prompt for `c`: its base, minus dropped sentences
        (removed as exact text, so dropping none is the base byte for byte)."""
        text = systems[c["sys"]][1]
        for i in c.get("drop", []):
            text = text.replace(shipped_sentences[i], "", 1)
        return re.sub(r"[ \t]{2,}", " ", text).strip()

    cfg = {k: v[0] for k, v in KNOBS.items()}
    cfg["sys"] = 0
    cfg["drop"] = []
    prior = None
    if args.start_from:
        for line in open(args.start_from, encoding="utf-8"):
            e = json.loads(line)
            if e.get("step") == "confirm":
                prior = e
        if prior is None:
            sys.exit(f"no confirmed winner in {args.start_from}")
        cfg = {**cfg, **prior["cfg"]}
    # Stability first: a file the full prompt grades right only some of the
    # time is a coin flip under any prompt, so it cannot veto a step. Only
    # the files graded right every time must stay right.
    if prior:
        votes = prior["votes"]
        size = sum(len(system_for(cfg)) + len(reduce(renders[r], r, cfg)) for r in renders)
    else:
        _, size, votes, _ = grade_all(args, renders, labels, cfg, system_for(cfg), args.baseline_runs, key)
    rate = {rel: v.count(labels[rel][0]) / len(v) for rel, v in votes.items()}
    best = {rel for rel, r in rate.items() if r == 1.0}
    flappy = sorted(rel for rel, r in rate.items() if 0 < r < 1)
    print(f"start  stable {len(best)}/{len(renders)}  expected {sum(rate.values()):.2f}  {size}B  "
          f"flappy={flappy}  {describe(cfg)}", flush=True)
    log.write(json.dumps({"step": "start", "cfg": cfg, "votes": votes, "bytes": size}) + "\n")
    rejected = set()
    fragile = {}  # file -> steps it has sunk
    while True:
        moves = []
        for k, steps in KNOBS.items():
            i = steps.index(cfg[k])
            if i + 1 < len(steps):
                moves.append((k, steps[i + 1]))
        if cfg["sys"] + 1 < len(systems):
            moves.append(("sys", cfg["sys"] + 1))
        if args.sentences and cfg["sys"] == 0:
            for i in range(len(shipped_sentences)):
                if i not in protect and i not in cfg["drop"]:
                    moves.append(("drop", sorted(cfg["drop"] + [i])))
        priced = []
        for k, v in moves:
            if (k, str(v), describe(cfg)) in rejected:
                continue
            trial = dict(cfg, **{k: v})
            sysp = system_for(trial)
            b = sum(len(sysp) + len(reduce(renders[r], r, trial)) for r in renders)
            priced.append((size - b, k, v, trial))
        priced.sort(reverse=True)
        accepted = False
        for saved, k, v, trial in priced:
            if saved <= 0:
                continue
            lost, tsize = try_step(args, renders, labels, trial, system_for(trial), best, fragile, key)
            ok = not lost
            print(f"{'ACCEPT' if ok else 'reject'} {k}={v}  {tsize}B (-{saved}B)  lost={sorted(lost)[:4]}",
                  flush=True)
            log.write(json.dumps({"step": f"{k}={v}", "cfg": trial, "bytes": tsize,
                                  "lost": sorted(lost), "accepted": ok}) + "\n")
            for rel in lost:
                fragile[rel] = fragile.get(rel, 0) + 1
            if ok:
                cfg, size = trial, tsize
                accepted = True
                break
            rejected.add((k, str(v), describe(cfg)))
        if not accepted:
            break
    print(f"\nwinner {describe(cfg)} system={systems[cfg['sys']][0]}  {len(best)}/{len(renders)}  {size}B", flush=True)
    _, tsize, wvotes, _ = grade_all(args, renders, labels, cfg, system_for(cfg), args.baseline_runs, key)
    wrate = {rel: v.count(labels[rel][0]) / len(v) for rel, v in wvotes.items()}
    worse = sorted(rel for rel in renders if wrate[rel] < rate[rel])
    print(f"confirm: expected {sum(wrate.values()):.2f} vs baseline {sum(rate.values()):.2f}  "
          f"{tsize}B vs {sum(len(systems[0][1]) + len(reduce(renders[r], r, {k: v[0] for k, v in KNOBS.items()})) for r in renders)}B  "
          f"lower-rate={worse}", flush=True)
    log.write(json.dumps({"step": "confirm", "cfg": cfg, "votes": wvotes, "bytes": tsize}) + "\n")
    log.write(json.dumps({"step": "winner", "cfg": cfg}) + "\n")


if __name__ == "__main__":
    main()
