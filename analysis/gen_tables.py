"""Generate LaTeX tables for paper v3 from aggregate.json (python gen_tables.py aggregate.json outdir)."""
import json, sys
from pathlib import Path

d = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
M = ('q35', 'qvl', 'hermes')


def f(x, nd=1):
    return '' if x is None else (f'{x:.{nd}f}')


def tp(model, m, key='final'):
    r = d['tp'].get(f'{model}/{m}'); return None if r is None else r[key]


def np_(model, task, m):
    r = d['np'].get(f'{model}/{task}/{m}'); return None if r is None else r['acc']


def ct(model, m, key='eval_commonsense_micro'):
    r = d['ct'].get(f'{model}/{m}'); return None if r is None else r[key]


ROWS = [('Direct', 'direct', 'direct', 'direct'), ('Staged', 'staged', 'staged', 'staged'),
        ('+Check', 'direct_repair', 'check', 'check'), ('+CGF', 'direct_cgf', 'cgf', 'cgf'),
        ('+CGF, unranked', 'direct_cgf_unranked', 'cgf_unranked', 'cgf_unranked'), ('FGS+CGF', 'staged_verified_cgf', 'fgs_cgf', 'fgs_cgf')]


def main_table():
    cols = []  # list of (getter per row)
    for model in M: cols.append(lambda r, model=model: tp(model, r[1]))
    for model in M: cols.append(lambda r, model=model: ct(model, r[2]))
    for task in ('calendar', 'meeting', 'trip'):
        for model in M:
            cols.append(lambda r, model=model, task=task: ('--' if task == 'calendar' and r[3] in ('staged', 'fgs_cgf') else np_(model, task, r[3])))
    vals = [[c(r) for c in cols] for r in ROWS]
    best = []
    for j in range(len(cols)):
        nums = [v[j] for v in vals if isinstance(v[j], (int, float))]
        best.append(max(nums) if nums else None)
    lines = []
    for i, r in enumerate(ROWS):
        cells = []
        for j, v in enumerate(vals[i]):
            if v == '--': cells.append('--')
            elif v is None: cells.append('')
            else: cells.append(f'\\textbf{{{v:.1f}}}' if best[j] is not None and abs(v - best[j]) < 1e-9 else f'{v:.1f}')
        lines.append(f'{r[0]} & ' + ' & '.join(cells) + ' \\\\')
        if r[0] in ('Staged', '+CGF, unranked'): lines.append('\\midrule')
    (out / 'tab_main_rows.tex').write_text('\n'.join(lines) + '\n')


def pooled_table():
    names = dict(q35='Qwen3.5-9B', qvl='Qwen3-VL-8B', hermes='Hermes-3-8B')
    task_name = dict(TravelPlanner='TravelPl.', calendar='Calendar', meeting='Meeting', trip='Trip')
    lines = []
    for k, v in d['pooled'].items():
        if k == '_pooled': continue
        task, model = k.split('/')
        p = v['p']; ps = f'{p:.2f}' if p >= 0.01 else f'${p:.0e}'.replace('e-0', '{\\times}10^{-').replace('e-', '{\\times}10^{-') + '}$'
        lines.append(f'{task_name[task]}, {names[model]} & {v["n"]} & {v["check"]:.1f} & {v["cgf"]:.1f} & {v["b"]}/{v["c"]} & {ps} \\\\')
    pv = d['pooled']['_pooled']; p = pv['p']
    ps = f'${p:.0e}'.replace('e-0', '{\\times}10^{-').replace('e-', '{\\times}10^{-') + '}$'
    lines.append('\\midrule')
    lines.append(f'Pooled ({pv["settings"]} settings) & {pv["n"]} & & & {pv["b"]}/{pv["c"]} & {ps} \\\\')
    (out / 'tab_pooled_rows.tex').write_text('\n'.join(lines) + '\n')


def tp_detail():
    lab = [('Direct', 'direct'), ('Staged', 'staged'), ('Staged+Ledger', 'staged_ledger'), ('Direct+Check', 'direct_repair'),
           ('Staged+Ledger+Check', 'staged_ledger_repair'), ('FGS', 'staged_verified'), ('Direct+CGF', 'direct_cgf'),
           ('Direct+CGF, unranked', 'direct_cgf_unranked'), ('FGS+CGF', 'staged_verified_cgf')]
    lines = []
    for name, m in lab:
        r = d['tp'].get(f'q35/{m}')
        if r is None: continue
        lines.append(f'{name} & {r["cs_micro"]:.1f} & {r["cs_macro"]:.1f} & {r["hard_micro"]:.1f} & {r["hard_macro"]:.1f} & {r["final"]:.1f} & {r["calls"]:.1f} & {r["ktok"]} \\\\')
        if name in ('Staged+Ledger', 'FGS'): lines.append('\\midrule')
    (out / 'tab_tp_rows.tex').write_text('\n'.join(lines) + '\n')


main_table(); pooled_table(); tp_detail()
print('ok', list(out.iterdir()))
