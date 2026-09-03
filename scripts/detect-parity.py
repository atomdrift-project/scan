#!/usr/bin/env python3
"""Detection parity between two `atomscan -f json` outputs, tolerant of the
record shapes the CLI emits: analyzed records (`raw.files`), bloom-skipped
records (`scanned: false`), and error records.

usage: jsondiff2.py A.json B.json [--verbose]
"""
import json, os, sys, collections

def norm(path):
    arch, sep, member = path.partition('!!')
    return os.path.basename(arch.replace('\\', '/')) + sep + member

def load(p):
    out = {}
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        raw = r.get('raw') or {}
        files = raw.get('files') or []
        key = norm(files[0].get('path', '')) if files else norm(r.get('path', '?'))
        ml = r.get('ml') or {}
        traits = {}
        for f in files:
            traits[norm(f['path']) if 'path' in f else f.get('id', '?')] = frozenset(
                t['id'] for t in f.get('traits', []))
        out[key] = {
            'lvl': ml.get('lvl'), 'prob': ml.get('prob'),
            'scanned': r.get('scanned', True), 'verdict': r.get('verdict'),
            'members': len(files), 'traits': traits,
        }
    return out

def main():
    verbose = '--verbose' in sys.argv
    a, b = load(sys.argv[1]), load(sys.argv[2])
    c = collections.Counter()
    diffs = []
    for k in sorted(set(a) | set(b)):
        ra, rb = a.get(k), b.get(k)
        if ra is None or rb is None:
            c['missing_record'] += 1
            diffs.append((k, 'present in only one output'))
            continue
        rec = []
        for field in ('lvl', 'prob', 'scanned', 'verdict', 'members'):
            if ra[field] != rb[field]:
                c[field] += 1
                rec.append(f'{field}: {ra[field]} -> {rb[field]}')
        for m in sorted(set(ra['traits']) | set(rb['traits'])):
            sa, sb = ra['traits'].get(m, frozenset()), rb['traits'].get(m, frozenset())
            if sa != sb:
                c['trait_members'] += 1
                c['traits_lost'] += len(sa - sb)
                c['traits_gained'] += len(sb - sa)
                rec.append(f'  {m}: -{sorted(sa - sb) if verbose else len(sa - sb)} '
                           f'+{sorted(sb - sa) if verbose else len(sb - sa)}')
        if rec:
            diffs.append((k, '; '.join(rec)))
    total = sum(len(s) for r in a.values() for s in r['traits'].values())
    print(f'records A={len(a)} B={len(b)}  traits(A)={total}  differing records={len(diffs)}  {dict(c)}')
    for k, d in diffs[:40]:
        print(f'* {k}: {d}')

if __name__ == '__main__':
    main()
