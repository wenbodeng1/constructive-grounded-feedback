"""Paper figure + table from *.score.json / *.meta.json files.

python make_figs.py <outdir> method1 method2 ...   (method = runs/<method>.jsonl stem)
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
RUNS = Path(os.environ.get('TP_RUNS', Path(__file__).resolve().parents[1] / 'results' / 'runs' / 'travelplanner')) / os.environ.get('SUB', 'q35')
LABEL = dict(direct='Direct', staged='Staged', staged_ledger='Staged+Ledger', direct_repair='Direct+Check',
             staged_ledger_repair='Staged+Ledger+Check', staged_verified='FGS (gated staging)', direct_cgf='Direct+CGF',
             staged_verified_cgf='FGS+CGF', direct_cgf_unranked='Direct+CGF, unranked')
SHORT = {'is_valid_restaurants': 'Diverse rest.', 'is_valid_attractions': 'Diverse attr.', 'is_valid_accommodation': 'Min. nights',
         'is_valid_transportation': 'Transp. conflict', 'is_valid_information_in_sandbox': 'In sandbox', 'is_reasonable_visiting_city': 'City route',
         'is_valid_information_in_current_city': 'In current city', 'valid_cost': 'Budget', 'valid_room_rule': 'Room rule',
         'valid_room_type': 'Room type', 'valid_cuisine': 'Cuisine', 'is_not_absent': 'Complete info', 'valid_transportation': 'Transp. pref.'}


def load(m):
    s = json.load(open(RUNS / f'{m}.jsonl.score.json'))['summary']; meta = json.load(open(RUNS / f'{m}.jsonl.meta.json'))
    return s, meta


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); methods = sys.argv[2:]
    rows = []
    base_tok = None
    for m in methods:
        s, meta = load(m); tok = (meta['prompt'] + meta['completion']) / meta['n'] / 1e3
        base_tok = base_tok or tok
        rows.append((LABEL.get(m, m), s['Commonsense Constraint Micro Pass Rate'], s['Commonsense Constraint Macro Pass Rate'],
                     s['Hard Constraint Micro Pass Rate'], s['Hard Constraint Macro Pass Rate'], s['Final Pass Rate'], meta['calls'] / meta['n'], tok))
    best = [max(r[i] for r in rows) for i in range(1, 6)]
    lines = []
    for r in rows:
        cells = [(f'\\textbf{{{v:.1f}}}' if abs(v - best[i]) < 1e-9 else f'{v:.1f}') for i, v in enumerate(r[1:6])]
        lines.append(f'{r[0]} & ' + ' & '.join(cells) + f' & {r[6]:.1f} & {r[7]:.0f} \\\\')
    for idx in (6, 3):
        if len(lines) > idx: lines.insert(idx, '\\midrule')
    (out / 'table_main.tex').write_text('\n'.join(lines) + '\n')
    # failure-count figure for the delivery study
    keys = ['is_valid_restaurants', 'is_valid_attractions', 'is_reasonable_visiting_city', 'is_valid_accommodation', 'valid_cost', 'is_valid_transportation', 'is_valid_information_in_sandbox']
    study = [m for m in methods if m in ('direct', 'staged', 'staged_ledger')]
    fig, ax = plt.subplots(figsize=(3.5, 1.9), dpi=300)
    w = 0.8 / len(study); colors = ['#4C72B0', '#DD8452', '#55A868']
    for i, m in enumerate(study):
        fc = json.load(open(RUNS / f'{m}.jsonl.score.json'))['summary']['fail_counts']
        ax.bar([k + i * w - 0.4 + w / 2 for k in range(len(keys))], [fc.get(k, 0) for k in keys], w, label=LABEL[m], color=colors[i])
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([SHORT[k] for k in keys], rotation=30, ha='right', fontsize=6)
    ax.set_ylabel('# failed plans (of 180)', fontsize=6.5); ax.tick_params(axis='y', labelsize=6)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc='upper right'); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.2); fig.savefig(out / 'fig_failures.pdf'); print((out / 'table_main.tex').read_text())


if __name__ == '__main__':
    main()
