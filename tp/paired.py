"""Paired exact McNemar tests on per-query outcomes. python paired.py A B [metric]"""
import json, sys
from math import comb
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SUB selects the model folder under results/runs/travelplanner (q35 | qvl | hermes)
R = os.path.join(os.environ.get("TP_RUNS", os.path.join(_REPO, "results", "runs", "travelplanner")), os.environ.get("SUB", "q35").strip("/")) + "/"
def rows(m): return json.load(open(R + m + '.jsonl.score.json'))['rows']
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
a, b = sys.argv[1], sys.argv[2]
for metric in sys.argv[3:] or ['final', 'cs_pass', 'hard_pass']:
    x, y = rows(a), rows(b)
    only_a = sum(r[metric] and not s[metric] for r, s in zip(x, y)); only_b = sum(s[metric] and not r[metric] for r, s in zip(x, y))
    print(f'{metric}: {a}={sum(r[metric] for r in x)} {b}={sum(s[metric] for s in y)} only_{a}={only_a} only_{b}={only_b} p={mcnemar(only_a, only_b):.4f}')
