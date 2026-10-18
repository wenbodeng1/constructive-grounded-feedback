"""Score plans with the official TravelPlanner evaluator (validation) + per-constraint breakdown.

python score_tp.py runs/direct.jsonl [runs/staged.jsonl ...]
Must be importable from TravelPlanner/evaluation (database paths are relative to it).
"""
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = os.environ.get('TP_ROOT', os.path.join(_REPO, 'third_party', 'TravelPlanner'))
CWD = os.getcwd()  # plan paths on the command line are relative to where the script was started
os.chdir(TP + '/evaluation'); sys.path.insert(0, TP + '/evaluation'); sys.path.insert(0, TP)
import types  # noqa: E402
_gr = types.ModuleType('gradio'); _gr.Error = type('Error', (Exception,), {}); sys.modules.setdefault('gradio', _gr)
import datasets  # noqa: E402
_orig = datasets.load_dataset
datasets.load_dataset = lambda *a, **k: _orig(*a, **{x: y for x, y in k.items() if x != 'download_mode'})
import eval as tp_eval  # noqa: E402
tp_eval.load_dataset = datasets.load_dataset
from commonsense_constraint import evaluation as cs_eval  # noqa: E402
from hard_constraint import evaluation as hard_eval  # noqa: E402


def per_query(path):
    data = list(datasets.load_dataset('osunlp/TravelPlanner', 'validation')['validation'])
    plans = [json.loads(l) for l in open(path)]
    rows = []
    for q, p in zip(data, plans):
        if isinstance(q['local_constraint'], str): q['local_constraint'] = eval(q['local_constraint'])
        row = dict(idx=p['idx'], level=q['level'], days=q['days'], delivered=bool(p['plan']), cs={}, hard={})
        if p['plan']:
            cs = cs_eval(q, p['plan']); row['cs'] = {k: v[0] for k, v in cs.items()}
            if cs['is_not_absent'][0] and cs['is_valid_information_in_sandbox'][0]:
                row['hard'] = {k: v[0] for k, v in hard_eval(q, p['plan']).items()}
        row['cs_pass'] = row['delivered'] and all(v is not False for v in row['cs'].values())
        row['hard_pass'] = bool(row['hard']) and all(v is not False for v in row['hard'].values())
        row['final'] = row['cs_pass'] and row['hard_pass']
        rows.append(row)
    return rows


if __name__ == '__main__':
    for path in sys.argv[1:]:
        path = os.path.abspath(os.path.join(os.environ.get('OLDPWD_TP', CWD), path))
        # Plans that crash the official evaluator are counted as not delivered (all metrics fail).
        data = list(datasets.load_dataset('osunlp/TravelPlanner', 'validation')['validation'])
        plans = [json.loads(l) for l in open(path)]; crashed = []
        for q, p in zip(data, plans):
            if not p['plan']: continue
            qq = dict(q); qq['local_constraint'] = eval(q['local_constraint']) if isinstance(q['local_constraint'], str) else q['local_constraint']
            try:
                cs = cs_eval(qq, p['plan'])
                if cs['is_not_absent'][0] and cs['is_valid_information_in_sandbox'][0]: hard_eval(qq, p['plan'])
            except Exception:
                crashed.append(p['idx']); p['plan'] = None
        if crashed:
            path = path + '.sanitized.jsonl'
            with open(path, 'w') as f:
                for p in plans: f.write(json.dumps(p) + '\n')
        scores, _ = tp_eval.eval_score('validation', path)
        rows = per_query(path)
        fails = {}
        for r in rows:
            for k, v in {**r['cs'], **r['hard']}.items():
                if v is False: fails[k] = fails.get(k, 0) + 1
        out = dict(file=path, evaluator_crash=len(crashed), **{k: round(v * 100, 2) for k, v in scores.items()}, fail_counts=dict(sorted(fails.items(), key=lambda x: -x[1])))
        json.dump(dict(summary=out, rows=rows), open(path.replace('.sanitized.jsonl', '') + '.score.json', 'w'), indent=1)
        print(json.dumps(out))
