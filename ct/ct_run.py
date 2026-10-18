"""ChinaTravel (EN, phase-2 competition_test, 100 queries), tool-free sole-planning:
Direct / Staged / +Check / +CGF / FGS+CGF.

python ct_run.py --method direct|staged|check|cgf|cgf_unranked|fgs_cgf --model <served> --ports p1,p2 --out runs/<model>/<method>
The planner sees only the natural-language request and the reference block R (ct_ref);
inner-city transports are filled by the environment's own goto() (ct_expand), identically
for all methods. The checker below uses only the sandbox tables and the expanded plan; it
never reads hard_logic_py / hard_logic_nl or the official evaluator. Candidates are drawn
only from entities present in R, so CGF adds no information beyond R.
"""
import argparse, concurrent.futures as cf, json, math, os, re, threading, time, urllib.request
from pathlib import Path
import ct_common, ct_data, ct_expand, ct_eval

HERE = Path(__file__).resolve().parent
REFS = Path(os.environ.get('CT_REFS', HERE / 'refs'))  # one reference block R per query, built by ct_ref.py --dump_dir
LOCK = threading.Lock(); USAGE = dict(calls=0, prompt=0, completion=0)

INSTR = """You are a travel planner. Plan the complete trip described in the request, using ONLY the reference data below (copy names and IDs exactly).
Output a JSON itinerary: one entry per day; each activity has type (train, airplane, attraction, breakfast, lunch, dinner, accommodation), name (exact place name, or the train/flight ID), start_time and end_time (HH:MM), and transport_mode (walk, metro or taxi) used to travel from the previous activity's place to this one.
Rules: day 1 starts with an outbound train or flight from the outbound list; the last day ends with a return train or flight from the return list (its start_time/end_time are the listed departure/arrival times). Every night except the last ends with an accommodation activity (end_time 24:00) at a hotel from the list. At most one breakfast, one lunch and one dinner per day; lunch must overlap 11:00-14:00 and dinner 17:00-20:00; a breakfast at your hotel is free. Do not visit any attraction or restaurant twice. Visit places only within their opening hours, and leave enough time to travel between places (travel times are computed from real distances).
"""

PLAN_SCHEMA = ct_expand.HL_JSON_SCHEMA
DAY_SCHEMA = dict(type='object', required=['activities'], properties=dict(activities=PLAN_SCHEMA['properties']['itinerary']['items']['properties']['activities']))
SKEL_SCHEMA = dict(type='object', required=['outbound', 'return', 'hotel'], additionalProperties=False,
                   properties=dict(outbound=dict(type='string'), **{'return': dict(type='string')}, hotel=dict(type='string')))


def chat(messages, schema, port, model, max_tokens=4096):
    body = dict(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens, chat_template_kwargs={'enable_thinking': False},
                response_format=dict(type='json_schema', json_schema=dict(name='out', schema=schema)))
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(3):
        try:
            with opener.open(req, timeout=1800) as h: r = json.load(h); break
        except urllib.error.HTTPError as e:
            if e.code == 400: raise RuntimeError('http400 ' + e.read().decode(errors='replace')[:200])
            if attempt == 2: raise
        except OSError:
            if attempt == 2: raise
            time.sleep(3)
    with LOCK:
        USAGE['calls'] += 1; USAGE['prompt'] += r['usage']['prompt_tokens']; USAGE['completion'] += r['usage']['completion_tokens']
    return json.loads(r['choices'][0]['message']['content'] or '{}')


def tmin(s):
    try: h, m = str(s).split(':')[:2]; return int(h) * 60 + int(m)
    except Exception: return None


def hhmm(m): return f'{m // 60:02d}:{m % 60:02d}'


def window(row):
    o, c = tmin(row['opentime']), tmin(row['endtime'])
    if o is None or c is None: return None
    if c <= o: c += 1440
    return o, c


# ---------------------------------------------------------------- checker and suggester
class Ctx:
    def __init__(self, q, ref):
        self.q, self.ref = q, ref
        t = ct_common.tools(); city = q['target_city']
        self.A = t['attractions'].data[city]; self.Rs = t['restaurants'].data[city]; self.H = t['accommodations'].data[city]
        # entities visible to the planner (present in R)
        self.vis = {n for n in set(self.A['name']) | set(self.Rs['name']) | set(self.H['name']) if isinstance(n, str) and n in ref}
        self.out = ct_common.intercity_tables(q['start_city'], q['target_city']); self.ret = ct_common.intercity_tables(q['target_city'], q['start_city'])

    def row(self, df, name):
        hit = df[df['name'] == name]
        return None if len(hit) == 0 else hit.iloc[0]


def check(cx, plan, constructive, ranked=True):
    """Issues found by looking up the expanded plan in the sandbox; optional candidate lists drawn from R."""
    q = cx.q; issues, hints = [], []
    it = plan.get('itinerary') or []
    if len(it) != int(q['days']): issues.append(f'The trip must have exactly {q["days"]} days, but the plan has {len(it)}.')
    if not it or not it[0].get('activities'): return issues + ['The plan is empty.'], hints
    used_a, used_r = {}, {}
    need_ic = []
    first, last = it[0]['activities'][0], it[-1]['activities'][-1] if it[-1].get('activities') else {}
    for tag, act, tabs in (('outbound', first, cx.out), ('return', last, cx.ret)):
        idk = 'TrainID' if act.get('type') == 'train' else 'FlightID' if act.get('type') == 'airplane' else None
        ok = idk is not None and any(str(r[idk]) == str(act.get(idk)) for df in tabs for _, r in (df.iterrows() if idk in df.columns else []))
        if not ok:
            issues.append(f'The {"first" if tag == "outbound" else "last"} activity must be a {tag} train or flight from the {tag} list; found {act.get("type")} "{act.get("position") or act.get(idk or "TrainID") or ""}".')
            need_ic.append(tag)
    for d, day in enumerate(it, 1):
        meals, prev_end = {}, None
        for a in day.get('activities') or []:
            typ = a.get('type'); s, e = tmin(a.get('start_time')), tmin(a.get('end_time'))
            arr = tmin(a['transports'][-1]['end_time']) if a.get('transports') else None
            label = a.get('position') or a.get('TrainID') or a.get('FlightID') or typ
            if typ not in ('train', 'airplane'):
                if s is not None and e is not None and s >= e: issues.append(f'Day {d} {typ} "{label}": end_time must be after start_time.')
                if arr is not None and s is not None and arr > s:
                    issues.append(f'Day {d} {typ} "{label}": you arrive at {hhmm(arr)} (computed travel time) but it starts at {a.get("start_time")}; start at or after {hhmm(arr)} or travel faster.')
                if prev_end is not None and s is not None and s < prev_end: issues.append(f'Day {d} {typ} "{label}" starts before the previous activity ends.')
            if typ == 'attraction':
                r = cx.row(cx.A, label)
                if r is None: issues.append(f'Day {d}: attraction "{label}" is not in the reference data.'); continue
                w = window(r)
                if w and s is not None and e is not None and not (w[0] <= s and e <= w[1]):
                    issues.append(f'Day {d}: attraction "{label}" is open {r["opentime"]}-{r["endtime"]}, but the visit is {a["start_time"]}-{a["end_time"]}.')
                if label in used_a: issues.append(f'Day {d}: attraction "{label}" was already visited on day {used_a[label]}.')
                used_a.setdefault(label, d)
            elif typ in ('breakfast', 'lunch', 'dinner'):
                if typ in meals: issues.append(f'Day {d} has two {typ}s.')
                meals[typ] = 1
                r = cx.row(cx.Rs, label)
                if r is None:
                    if typ == 'breakfast' and cx.row(cx.H, label) is not None:
                        if s is not None and (s >= 540 or (e or 0) <= 360): issues.append(f'Day {d}: a hotel breakfast must overlap 06:00-09:00.')
                        continue
                    issues.append(f'Day {d}: {typ} place "{label}" is not a restaurant in the reference data.'); continue
                w = window(r)
                if w and s is not None and e is not None and not (w[0] <= s and e <= w[1]):
                    issues.append(f'Day {d}: restaurant "{label}" is open {r["opentime"]}-{r["endtime"]}, but the {typ} is {a["start_time"]}-{a["end_time"]}.')
                if typ == 'lunch' and s is not None and (s >= 840 or (e or 0) <= 660): issues.append(f'Day {d}: lunch must overlap 11:00-14:00.')
                if typ == 'dinner' and s is not None and (s >= 1200 or (e or 0) <= 1020): issues.append(f'Day {d}: dinner must overlap 17:00-20:00.')
                if label in used_r: issues.append(f'Day {d}: restaurant "{label}" was already used on day {used_r[label]}.')
                used_r.setdefault(label, d)
            elif typ == 'accommodation':
                if cx.row(cx.H, label) is None: issues.append(f'Day {d}: hotel "{label}" is not in the reference data.')
            if e is not None and typ not in ('train', 'airplane'): prev_end = e
    if int(q['days']) > 1 and not any(a.get('type') == 'accommodation' for day in it for a in day.get('activities') or []):
        issues.append('A multi-day trip needs a hotel (accommodation activity) for the nights.')
    if constructive and issues:
        hints = suggest(cx, plan, issues, used_a, used_r, need_ic) if ranked else suggest_unranked(cx, issues)
    return issues, hints


def suggest(cx, plan, issues, used_a, used_r, need_ic, k=6):
    text = ' '.join(issues); out = []
    if 'attraction' in text:
        rows = [r for _, r in cx.A.iterrows() if r['name'] in cx.vis and r['name'] not in used_a]
        for m in re.finditer(r'attraction "([^"]+)" is (?:open|not in)[^.]*?(?:visit is (\d\d:\d\d)-(\d\d:\d\d))?\.', text):
            s, e = tmin(m.group(2)) if m.group(2) else None, tmin(m.group(3)) if m.group(3) else None
            ok = [r for r in rows if s is None or ((w := window(r)) and w[0] <= s and e <= w[1])]
            ok.sort(key=lambda r: (r['price'], r['name']))
            if ok: out.append((f'Unvisited attractions open during {m.group(2)}-{m.group(3)}' if s is not None else 'Unvisited attractions') + ' (cheapest first): ' +
                              '; '.join(f'"{r["name"]}" {r["opentime"]}-{r["endtime"]}, {ct_common.fmt_num(r["price"])} CNY' for r in ok[:k]))
        if not any(o.startswith('Unvisited attractions') for o in out):
            rows.sort(key=lambda r: (r['price'], r['name']))
            out.append('Unvisited attractions (cheapest first): ' + '; '.join(f'"{r["name"]}" {r["opentime"]}-{r["endtime"]}, {ct_common.fmt_num(r["price"])} CNY' for r in rows[:k]))
    for meal, (lo, hi) in (('lunch', (660, 840)), ('dinner', (1020, 1200))):
        if meal in text or ('restaurant' in text and meal in text):
            rows = [r for _, r in cx.Rs.iterrows() if r['name'] in cx.vis and r['name'] not in used_r and (w := window(r)) and w[0] <= hi - 30 and w[1] >= lo + 60]
            rows.sort(key=lambda r: (r['price'], r['name']))
            if rows: out.append(f'Unused restaurants open for {meal} ({hhmm(lo)}-{hhmm(hi)}), cheapest first: ' +
                                '; '.join(f'"{r["name"]}" {r["opentime"]}-{r["endtime"]}, {ct_common.fmt_num(r["price"])} CNY/person, {r["cuisine"]}' for r in rows[:k]))
    if 'breakfast' in text:
        rows = [r for _, r in cx.Rs.iterrows() if r['name'] in cx.vis and r['name'] not in used_r and (w := window(r)) and w[0] <= 480]
        rows.sort(key=lambda r: (r['price'], r['name']))
        if rows: out.append('Restaurants open by 08:00 (or eat a free breakfast at your hotel), cheapest first: ' + '; '.join(f'"{r["name"]}" {r["opentime"]}-{r["endtime"]}, {ct_common.fmt_num(r["price"])} CNY' for r in rows[:k]))
    if 'hotel' in text:
        rows = [r for _, r in cx.H.iterrows() if r['name'] in cx.vis]
        ppl = int(cx.q['people_number']); rows.sort(key=lambda r: (r['price'] * math.ceil(ppl / max(1, r['numbed'])), r['name']))
        out.append('Hotels, cheapest per night for the group first: ' + '; '.join(f'"{r["name"]}" {ct_common.fmt_num(r["price"])} CNY/room, {int(r["numbed"])} bed(s)/room' for r in rows[:k]))
    for tag in need_ic:
        tr, fl = cx.out if tag == 'outbound' else cx.ret
        opts = [(r['Cost'], f'{r["TrainID"]} {r["BeginTime"]}-{r["EndTime"]} {ct_common.fmt_num(r["Cost"])}') for _, r in tr.iterrows() if r['Cost'] == r['Cost']]
        opts += [(r['Cost'], f'{r["FlightID"]} {r["BeginTime"]}-{r["EndTime"]} {ct_common.fmt_num(r["Cost"])}') for _, r in fl.iterrows() if r['Cost'] == r['Cost']]
        opts.sort()
        out.append(f'Valid {tag} trains/flights, cheapest first: ' + '; '.join(o for _, o in opts[:k]))
    return out


def suggest_unranked(cx, issues, k=6):
    import random
    rng = random.Random(cx.q['uid']); text = ' '.join(issues); out = []
    def samp(df, n): 
        names = sorted(x for x in df['name'] if x in cx.vis); rng.shuffle(names); return [cx.row(df, x) for x in names[:n]]
    if 'attraction' in text: out.append('Attractions: ' + '; '.join(f'"{r["name"]}" {r["opentime"]}-{r["endtime"]}, {ct_common.fmt_num(r["price"])} CNY' for r in samp(cx.A, k)))
    for meal in ('lunch', 'dinner', 'breakfast'):
        if meal in text: out.append(f'Restaurants: ' + '; '.join(f'"{r["name"]}" {r["opentime"]}-{r["endtime"]}, {ct_common.fmt_num(r["price"])} CNY/person, {r["cuisine"]}' for r in samp(cx.Rs, k)))
    if 'hotel' in text: out.append('Hotels: ' + '; '.join(f'"{r["name"]}" {ct_common.fmt_num(r["price"])} CNY/room, {int(r["numbed"])} bed(s)/room' for r in samp(cx.H, k)))
    for tag, tabs in (('outbound', cx.out), ('return', cx.ret)):
        if f'{tag} train or flight' in text:
            tr, fl = tabs; opts = [f'{r["TrainID"]} {r["BeginTime"]}-{r["EndTime"]} {ct_common.fmt_num(r["Cost"])}' for _, r in tr.iterrows()] + [f'{r["FlightID"]} {r["BeginTime"]}-{r["EndTime"]} {ct_common.fmt_num(r["Cost"])}' for _, r in fl.iterrows()]
            rng.shuffle(opts); out.append(f'{tag.capitalize()} trains/flights: ' + '; '.join(opts[:k]))
    return out


# ---------------------------------------------------------------- methods
def base(cx): return INSTR + '\n' + cx.ref + '\n\n## Request\n' + cx.q['nature_language'] + '\n'


def fb_text(issues, hints, what='plan', ranked=True):
    s = '\nA checker that looked up every item of your plan in the sandbox data found these problems:\n' + '\n'.join('- ' + x for x in issues)
    if hints: s += ('\nCandidates from the reference data (filtered and ranked):\n' if ranked else '\nCandidates from the reference data:\n') + '\n'.join('- ' + h for h in hints)
    return s + f'\nRevise the {what} so that all problems are fixed while still satisfying the request. Output the complete revised {what} as JSON.'


def run_direct(cx, port, model):
    return chat([dict(role='user', content=base(cx) + '\nOutput the itinerary JSON.')], PLAN_SCHEMA, port, model)


def repair(cx, hl, port, model, constructive, rounds=3, ranked=True):
    best_hl = hl; plan = ct_expand.expand(cx.q, hl); bi, _ = check(cx, plan, False); log = [len(bi)]
    issues, hints = check(cx, plan, constructive, ranked)
    cur = hl
    for _ in range(rounds):
        if not issues: break
        msg = base(cx) + '\nYour current itinerary JSON:\n' + json.dumps(cur) + fb_text(issues, hints if constructive else [], 'itinerary', ranked)
        try: cur = chat([dict(role='user', content=msg)], PLAN_SCHEMA, port, model)
        except Exception: break
        plan = ct_expand.expand(cx.q, cur); ci, _ = check(cx, plan, False); log.append(len(ci))
        issues, hints = check(cx, plan, constructive, ranked)
        if len(ci) <= len(bi): best_hl, bi = cur, ci
    return best_hl, log


def run_staged(cx, port, model, gated=False):
    q = cx.q; n = int(q['days']); log = []
    ask0 = base(cx) + '\nStage 1: decide only the outbound train/flight ID, the return train/flight ID and the hotel. Output JSON {"outbound": ID, "return": ID, "hotel": name}.'
    sk = chat([dict(role='user', content=ask0)], SKEL_SCHEMA, port, model)
    if gated:
        for _ in range(2):
            stub = {'itinerary': [{'day': 1, 'activities': [{'type': _ictype(cx, sk.get('outbound'), 'out'), 'name': sk.get('outbound'), 'start_time': '00:00', 'end_time': '00:00'}]}] +
                    ([{'day': i, 'activities': []} for i in range(2, n)]) +
                    [{'day': n, 'activities': ([] if n == 1 else [{'type': 'accommodation', 'name': sk.get('hotel'), 'start_time': '08:00', 'end_time': '09:00', 'transport_mode': 'taxi'}]) +
                      [{'type': _ictype(cx, sk.get('return'), 'ret'), 'name': sk.get('return'), 'start_time': '00:00', 'end_time': '00:00', 'transport_mode': 'taxi'}]}]}
            iss, hints = check(cx, ct_expand.expand(q, stub), True)
            iss = [x for x in iss if 'must be a' in x or 'hotel' in x]
            log.append(len(iss))
            if not iss: break
            sk = chat([dict(role='user', content=ask0 + '\nYour previous choice: ' + json.dumps(sk) + fb_text(iss, [h for h in hints if 'trains/flights' in h or 'Hotels' in h], 'choice'))], SKEL_SCHEMA, port, model)
    days = []
    for k in range(1, n + 1):
        req = []
        if k == 1: req.append(f'start with the outbound {sk.get("outbound")}')
        if k == n: req.append(f'end with the return {sk.get("return")}')
        if k < n: req.append(f'end the day with an accommodation activity at "{sk.get("hotel")}" (end_time 24:00)')
        ask = (base(cx) + '\nFixed decisions: ' + json.dumps(sk) + '\nDays already planned (do not change them):\n' + json.dumps(days) +
               f'\nNow plan only day {k} of {n}: ' + '; '.join(req) + '. Output JSON {"activities": [...]}.')
        acts = chat([dict(role='user', content=ask)], DAY_SCHEMA, port, model).get('activities', [])
        if gated:
            prev = ct_expand.expand(q, {'itinerary': days}) if days else None
            before = check(cx, prev, False)[0] if prev else []
            plan = ct_expand.expand(q, {'itinerary': days + [{'day': k, 'activities': acts}]})
            after, hints = check(cx, plan, True)
            new = [x for x in after if x not in before and (f'Day {k}' in x or 'already' in x)]
            if new:
                acts = chat([dict(role='user', content=ask + '\nYour previous day plan: ' + json.dumps(acts) + fb_text(new, hints, f'day {k} plan'))], DAY_SCHEMA, port, model).get('activities', acts)
        days.append({'day': k, 'activities': acts})
    hl = {'itinerary': days}
    if gated: hl, rl = repair(cx, hl, port, model, True); log += rl
    return hl, log


def _ictype(cx, ID, which):
    tr, fl = cx.out if which == 'out' else cx.ret
    return 'train' if 'TrainID' in tr.columns and str(ID) in set(map(str, tr['TrainID'])) else 'airplane'


def solve(cx, method, port, model):
    if method == 'direct': return run_direct(cx, port, model), []
    if method == 'staged': return run_staged(cx, port, model, gated=False)
    if method == 'fgs_cgf': return run_staged(cx, port, model, gated=True)
    hl = run_direct(cx, port, model)
    return repair(cx, hl, port, model, constructive=(method in ('cgf', 'cgf_unranked')), ranked=(method != 'cgf_unranked'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--method'); ap.add_argument('--model'); ap.add_argument('--ports'); ap.add_argument('--out')
    ap.add_argument('--limit', type=int); ap.add_argument('--workers', type=int, default=50)
    a = ap.parse_args(); ports = [int(x) for x in a.ports.split(',')]
    uids, qs = ct_data.load_queries()
    if a.limit: uids = uids[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    refs = {u: (REFS / f'{u}.txt').read_text() for u in uids}
    ct_common.tools()

    def job(i):
        u = uids[i]
        try:
            cx = Ctx(qs[u], refs[u]); hl, log = solve(cx, a.method, ports[i % len(ports)], a.model)
        except Exception as e:
            hl, log = {}, ['error:' + repr(e)[:200]]
        plan, notes = ct_expand.expand_with_notes(qs[u], hl) if hl else ({}, [])
        (out / f'{u}.json').write_text(json.dumps(plan, ensure_ascii=False))
        (out / f'{u}.hl.json').write_text(json.dumps(dict(hl=hl, log=log, notes=notes), ensure_ascii=False))
        return u, plan, log
    plans, errs = {}, 0
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for u, plan, log in pool.map(job, range(len(uids))):
            plans[u] = plan; errs += any(str(x).startswith('error') for x in log)
    res = ct_eval.evaluate(plans, uids=uids)
    keep = {k: v for k, v in res.items() if not isinstance(v, (dict, list))}
    meta = {**{f'eval_{k}': v for k, v in keep.items()}, **dict(method=a.method, model=a.model, n=len(uids), seconds=round(time.time() - t0), errors=errs), **USAGE}
    (out / '_meta.json').write_text(json.dumps(meta, indent=1)); (out / '_eval.json').write_text(json.dumps(res, default=str, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
