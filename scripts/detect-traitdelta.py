#!/usr/bin/env python3
"""Which trait ids differ between two `-f json` outputs, and on how many members.

usage: traitdelta.py A.json B.json
"""
import collections, json, os, sys

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
        files = (r.get('raw') or {}).get('files') or []
        key = norm(files[0].get('path', '')) if files else norm(r.get('path', '?'))
        out[key] = {
            (norm(f['path']) if 'path' in f else f.get('id', '?')):
            frozenset(t['id'] for t in f.get('traits', []))
            for f in files
        }
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
gained, lost = collections.Counter(), collections.Counter()
examples = collections.defaultdict(list)
for k in set(a) & set(b):
    for m in set(a[k]) | set(b[k]):
        sa, sb = a[k].get(m, frozenset()), b[k].get(m, frozenset())
        for t in sb - sa:
            gained[t] += 1
            if len(examples['+' + t]) < 2:
                examples['+' + t].append(m[-70:])
        for t in sa - sb:
            lost[t] += 1
            if len(examples['-' + t]) < 3:
                examples['-' + t].append(m[-70:])
print(f'gained trait ids: {len(gained)} ({sum(gained.values())} member-hits)')
for t, v in gained.most_common(15):
    print(f'  +{v:4d}  {t}')
    for e in examples['+' + t]:
        print(f'          {e}')
print(f'lost trait ids: {len(lost)} ({sum(lost.values())} member-hits)')
for t, v in lost.most_common(15):
    print(f'  -{v:4d}  {t}')
    for e in examples['-' + t]:
        print(f'          {e}')
