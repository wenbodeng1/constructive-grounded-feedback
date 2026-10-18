"""Rebuild results/aggregate.json (every number in the paper) from the released runs.

python analysis/aggregate.py > results/aggregate.json
Reads results/runs/{travelplanner,naturalplan,chinatravel}/<model>/... (unpack results/runs.tar.xz first).
Models: q35 = Qwen3.5-9B, qvl = Qwen3-VL-8B-Instruct, hermes = Hermes-3-Llama-3.1-8B. Missing runs are skipped.
"""
import json, os, sys
from math import comb
from pathlib import Path

RUNS = Path(os.environ.get('CGF_RUNS', Path(__file__).resolve().parents[1] / 'results' / 'runs'))


def sign_p(b, c):
    n = b + c; k = min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


# ---------------- NATURAL PLAN
def np_load(model, task, method):
    p = RUNS / 'naturalplan' / model / f'{task}_{method}.jsonl'
    if not p.exists(): return None, None
    p = str(p)
    rows = [json.loads(l) for l in open(p)]
    meta = json.load(open(p + '.meta.json')) if os.path.exists(p + '.meta.json') else {}
    return {r['id']: bool(r['ok']) for r in rows}, meta


# ---------------- TravelPlanner (validation, official evaluator scores already computed)
def tp_load(model, method):
    p = RUNS / 'travelplanner' / model / f'{method}.jsonl.score.json'
    if not p.exists(): return None, None
    p = str(p)
    s = json.load(open(p)); meta_p = p.replace('.score.json', '.meta.json')
    return s, (json.load(open(meta_p)) if os.path.exists(meta_p) else {})


def main():
    out = dict(np={}, tp={}, pooled={}, ct={})
    for model in ('q35', 'qvl', 'hermes'):
        for task in ('calendar', 'meeting', 'trip'):
            for m in ('direct', 'check', 'cgf', 'cgf_unranked', 'staged', 'fgs_cgf'):
                ok, meta = np_load(model, task, m)
                if ok is None: continue
                out['np'][f'{model}/{task}/{m}'] = dict(n=len(ok), acc=round(100 * sum(ok.values()) / len(ok), 2),
                                                        calls=round(meta.get('calls', 0) / max(1, meta.get('n', len(ok))), 2) if meta else None,
                                                        ktok=round((meta.get('prompt', 0) + meta.get('completion', 0)) / max(1, meta.get('n', len(ok))) / 1e3, 1) if meta else None,
                                                        errors=meta.get('errors') if meta else None)
    for model in ('q35', 'qvl', 'hermes'):
        for m in ('direct', 'staged', 'staged_ledger', 'direct_repair', 'staged_ledger_repair', 'staged_verified', 'direct_cgf', 'direct_cgf_unranked', 'staged_verified_cgf'):
            s, meta = tp_load(model, m)
            if s is None: continue
            sm = s['summary']
            out['tp'][f'{model}/{m}'] = dict(final=sm['Final Pass Rate'], cs_macro=sm['Commonsense Constraint Macro Pass Rate'],
                                             hard_macro=sm['Hard Constraint Macro Pass Rate'], cs_micro=sm['Commonsense Constraint Micro Pass Rate'],
                                             hard_micro=sm['Hard Constraint Micro Pass Rate'], delivery=sm['Delivery Rate'],
                                             crash=sm.get('evaluator_crash', 0), fails=sm.get('fail_counts', {}),
                                             calls=round(meta['calls'] / meta['n'], 2) if meta else None,
                                             ktok=round((meta['prompt'] + meta['completion']) / meta['n'] / 1e3) if meta else None)
    # paired CGF vs Check in every completed setting
    settings = []
    for model in ('q35', 'qvl', 'hermes'):
        a, _ = tp_load(model, 'direct_repair'); b, _ = tp_load(model, 'direct_cgf')
        if a and b:
            x = {r['idx']: r['final'] for r in a['rows']}; y = {r['idx']: r['final'] for r in b['rows']}
            settings.append((f'TravelPlanner/{model}', x, y))
        for task in ('calendar', 'meeting', 'trip'):
            x, _ = np_load(model, task, 'check'); y, _ = np_load(model, task, 'cgf')
            if x and y: settings.append((f'{task}/{model}', x, y))
    B = C = 0
    for name, x, y in settings:
        b = sum(x[k] and not y[k] for k in x); c = sum(y[k] and not x[k] for k in x); B += b; C += c
        out['pooled'][name] = dict(n=len(x), check=round(100 * sum(x.values()) / len(x), 2), cgf=round(100 * sum(y.values()) / len(y), 2), b=b, c=c, p=sign_p(b, c))
    out['pooled']['_pooled'] = dict(n=sum(len(x) for _, x, _ in settings), b=B, c=C, p=sign_p(B, C), settings=len(settings),
                                    positive=sum(1 for _, x, y in settings if sum(y.values()) > sum(x.values())))
    # FGS+CGF vs Direct+CGF (same checker and suggester; staging is the only difference)
    out['fgs'] = {}; B = C = 0; nset = 0
    for model in ('q35', 'qvl', 'hermes'):
        a, _ = tp_load(model, 'direct_cgf'); b, _ = tp_load(model, 'staged_verified_cgf')
        pairs = []
        if a and b: pairs.append(('TravelPlanner', {r['idx']: r['final'] for r in a['rows']}, {r['idx']: r['final'] for r in b['rows']}))
        for task in ('meeting', 'trip'):
            x, _ = np_load(model, task, 'cgf'); y, _ = np_load(model, task, 'fgs_cgf')
            if x and y: pairs.append((task, x, y))
        for task, x, y in pairs:
            bb = sum(x[k] and not y[k] for k in x); cc = sum(y[k] and not x[k] for k in x); B += bb; C += cc; nset += 1
            out['fgs'][f'{task}/{model}'] = dict(n=len(x), cgf=round(100 * sum(x.values()) / len(x), 2), fgs=round(100 * sum(y.values()) / len(y), 2), b=bb, c=cc, p=sign_p(bb, cc))
    out['fgs']['_pooled'] = dict(b=B, c=C, p=sign_p(B, C), settings=nset)
    # ChinaTravel
    for model in ('q35', 'qvl', 'hermes'):
        for m in ('direct', 'staged', 'check', 'cgf', 'cgf_unranked', 'fgs_cgf'):
            p = RUNS / 'chinatravel' / model / m / '_meta.json'
            if os.path.exists(p): out['ct'][f'{model}/{m}'] = json.load(open(p))
    # ChinaTravel paired sign tests on per-query violation counts (strict success is 0 everywhere)
    out['ct_paired'] = {}
    def ct_counts(model, m):
        p = RUNS / 'chinatravel' / model / m / '_eval.json'
        if not os.path.exists(p): return None
        pu = json.load(open(p))['per_uid']
        return {u: (len(v.get('commonsense_failures') or []), len(v.get('hard_failures') or [])) for u, v in pu.items()}
    for model in ('q35', 'qvl', 'hermes'):
        for a, b in (('direct', 'staged'), ('check', 'cgf'), ('cgf', 'cgf_unranked'), ('direct', 'cgf'), ('cgf', 'fgs_cgf'), ('direct', 'check')):
            x, y = ct_counts(model, a), ct_counts(model, b)
            if not x or not y: continue
            for j, key in ((0, 'cs'), (1, 'hard')):
                better = sum(y[u][j] < x[u][j] for u in x); worse = sum(y[u][j] > x[u][j] for u in x)
                out['ct_paired'][f'{model}/{a}->{b}/{key}'] = dict(b_better=better, b_worse=worse, p=sign_p(better, worse))
    json.dump(out, sys.stdout, indent=1, default=str)


if __name__ == '__main__':
    main()
