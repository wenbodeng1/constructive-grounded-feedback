"""Chinese results summary (markdown) from aggregate.json: python make_report.py aggregate.json > 实验汇总_v3.md"""
import json, sys

d = json.load(open(sys.argv[1]))
M = ('q35', 'qvl', 'hermes'); MN = dict(q35='Qwen3.5-9B', qvl='Qwen3-VL-8B', hermes='Hermes-3-8B')
ROWS = [('Direct', 'direct', 'direct', 'direct'), ('Staged', 'staged', 'staged', 'staged'),
        ('+Check', 'direct_repair', 'check', 'check'), ('+CGF', 'direct_cgf', 'cgf', 'cgf'),
        ('+CGF unranked', 'direct_cgf_unranked', 'cgf_unranked', 'cgf_unranked'), ('FGS+CGF', 'staged_verified_cgf', 'fgs_cgf', 'fgs_cgf')]


def g(sec, key, field):
    r = d[sec].get(key); return None if r is None else r[field]


def fmt(x): return '—' if x is None else f'{x:.1f}'


def pfmt(p): return f'{p:.2f}' if p >= 0.01 else f'{p:.0e}'


print('## 主表（严格成功率 %；ChinaTravel 为 commonsense micro %，其严格成功率在所有 18 组运行中均为 0）\n')
print('| 方法 | ' + ' | '.join(f'{b} {m}' for b in ('TP', 'CT', 'Calendar', 'Meeting', 'Trip') for m in ('Q', 'V', 'H')) + ' |')
print('|' + '---|' * 16)
for name, tp, ct, np_ in ROWS:
    cells = [fmt(g('tp', f'{m}/{tp}', 'final')) for m in M] + [fmt(g('ct', f'{m}/{ct}', 'eval_commonsense_micro')) for m in M]
    for task in ('calendar', 'meeting', 'trip'):
        cells += ['n/a' if task == 'calendar' and np_ in ('staged', 'fgs_cgf') else fmt(g('np', f'{m}/{task}/{np_}', 'acc')) for m in M]
    print(f'| {name} | ' + ' | '.join(cells) + ' |')

print('\n## CGF vs. 只报违规（+Check），配对精确检验\n')
print('| 设置 | n | Check | CGF | 仅Check解出/仅CGF解出 | p |\n|---|---|---|---|---|---|')
for k, v in d['pooled'].items():
    if k == '_pooled': continue
    t, m = k.split('/'); print(f'| {t} {MN[m]} | {v["n"]} | {v["check"]:.1f} | {v["cgf"]:.1f} | {v["b"]}/{v["c"]} | {pfmt(v["p"])} |')
pv = d['pooled']['_pooled']
print(f'| **合并（{pv["settings"]} 个设置，{pv["positive"]} 个 CGF 更高）** | {pv["n"]} | | | {pv["b"]}/{pv["c"]} | {pfmt(pv["p"])} |')

print('\n## FGS+CGF（每阶段 CGF 把关的分阶段）vs. Direct+CGF\n')
print('| 设置 | n | Direct+CGF | FGS+CGF | 仅Direct+CGF/仅FGS+CGF | p |\n|---|---|---|---|---|---|')
for k, v in d['fgs'].items():
    if k == '_pooled': continue
    t, m = k.split('/'); print(f'| {t} {MN[m]} | {v["n"]} | {v["cgf"]:.1f} | {v["fgs"]:.1f} | {v["b"]}/{v["c"]} | {pfmt(v["p"])} |')
pv = d['fgs']['_pooled']; print(f'| **合并（{pv["settings"]} 个设置）** | | | | {pv["b"]}/{pv["c"]} | {pfmt(pv["p"])} |')

print('\n## ChinaTravel 每条查询违规数的符号检验（后者更好/更差）\n')
print('| 模型 | 比较 | commonsense 更好/更差 (p) | hard 更好/更差 (p) |\n|---|---|---|---|')
for m in M:
    for a, b in (('direct', 'staged'), ('direct', 'check'), ('check', 'cgf'), ('cgf', 'cgf_unranked'), ('cgf', 'fgs_cgf')):
        cs = d['ct_paired'].get(f'{m}/{a}->{b}/cs'); hd = d['ct_paired'].get(f'{m}/{a}->{b}/hard')
        if cs: print(f'| {MN[m]} | {a}→{b} | {cs["b_better"]}/{cs["b_worse"]} ({pfmt(cs["p"])}) | {hd["b_better"]}/{hd["b_worse"]} ({pfmt(hd["p"])}) |')

print('\n## 每任务平均模型调用次数\n')
print('| 方法 | ' + ' | '.join(f'{b} {m}' for b in ('TP', 'Meeting', 'Trip') for m in ('Q', 'V', 'H')) + ' |')
print('|' + '---|' * 10)
for name, tp, ct, np_ in ROWS:
    cells = [fmt(g('tp', f'{m}/{tp}', 'calls')) for m in M]
    for task in ('meeting', 'trip'): cells += [fmt(g('np', f'{m}/{task}/{np_}', 'calls')) for m in M]
    print(f'| {name} | ' + ' | '.join(cells) + ' |')
