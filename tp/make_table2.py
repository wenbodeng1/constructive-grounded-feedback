"""Second-model table: python make_table2.py out.tex sub:Label [sub:Label ...]"""
import json, sys
from pathlib import Path
import os
R = Path(os.environ.get('TP_RUNS', Path(__file__).resolve().parents[1] / 'results' / 'runs' / 'travelplanner'))  # sub = q35 | qvl | hermes
M = [('direct', 'Direct'), ('staged', 'Staged'), ('direct_repair', 'Direct+Check'), ('direct_cgf', 'Direct+CGF')]
lines = []
for spec in sys.argv[2:]:
    sub, label = spec.split(':', 1); first = True
    for m, name in M:
        f = R / sub / f'{m}.jsonl.score.json'
        if not f.exists(): continue
        s = json.load(open(f))['summary']; fc = s['fail_counts']
        rep = fc.get('is_valid_restaurants', 0) + 0
        lines.append(('\\multirow{4}{*}{' + label + '}' if first else '') + f' & {name} & {s["Delivery Rate"]:.1f} & {s["Commonsense Constraint Macro Pass Rate"]:.1f} & {s["Hard Constraint Macro Pass Rate"]:.1f} & {s["Final Pass Rate"]:.1f} & {fc.get("is_valid_restaurants", 0)}/{fc.get("is_valid_attractions", 0)} \\\\')
        first = False
    lines.append('\\midrule')
Path(sys.argv[1]).write_text('\n'.join(lines[:-1]) + '\n'); print('\n'.join(lines[:-1]))
