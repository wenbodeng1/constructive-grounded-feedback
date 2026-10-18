"""NATURAL PLAN (calendar / meeting / trip): Direct vs violation-only Check vs CGF.

python np_run.py --task calendar --method cgf --model np-qwen35-9b --ports 9100,...,9105 --out runs/q35/calendar_cgf.jsonl
Official 5-shot prompts; greedy decoding; official evaluation functions for scoring.
Checkers use only the problem statement (prompt text; for meeting, the structured
copy of the same distances/availabilities given in the prompt), never the golden plan.
"""
import argparse, collections, concurrent.futures as cf, datetime, importlib.util, json, re, sys, threading, time, urllib.request
from pathlib import Path

NP = Path(__import__('os').environ.get('NP_ROOT', Path(__file__).resolve().parents[1] / 'third_party' / 'natural-plan'))
LOCK = threading.Lock(); USAGE = dict(calls=0, prompt=0, completion=0)


def load_official(name):
    import absl.flags
    absl.flags.DEFINE_string = lambda *a, **k: None
    spec = importlib.util.spec_from_file_location(name, NP / f'{name}.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def chat(messages, port, model, max_tokens=2048):
    body = dict(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens, chat_template_kwargs={'enable_thinking': False})
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(3):
        try:
            with opener.open(req, timeout=1200) as h: r = json.load(h); break
        except urllib.error.HTTPError as e:
            if e.code == 400: raise RuntimeError('http400 ' + e.read().decode(errors='replace')[:200])
            if attempt == 2: raise
        except OSError:
            if attempt == 2: raise
            time.sleep(3)
    with LOCK:
        USAGE['calls'] += 1; USAGE['prompt'] += r['usage']['prompt_tokens']; USAGE['completion'] += r['usage']['completion_tokens']
    return r['choices'][0]['message']['content'] or ''


# ---------------------------------------------------------------- calendar
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def t2m(s):
    h, m = s.split(':'); return int(h) * 60 + int(m)


def m2t(m): return f'{m // 60}:{m % 60:02d}'


def cal_problem(prompt):
    task = prompt[prompt.rfind('TASK:'):]
    head = task.split('\n')[0]
    dur = 60 if 'one hour' in head else 30 if 'half an hour' in head else (int(re.search(r'(\d+) minutes', head).group(1)) if re.search(r'(\d+) minutes', head) else 60)
    if re.search(r'for (\w+) hours', head) and 'one hour' not in head:
        w = re.search(r'for (\w+) hours', head).group(1); dur = {'two': 120, 'three': 180}.get(w, dur)
    lo, hi = [t2m(x) for x in re.search(r'work hours of (\d+:\d+) to (\d+:\d+)', head).groups()]
    days = [d for d in DAYS if d in head.split('work hours')[1]]
    busy = collections.defaultdict(list)
    for line in task.split('\n')[1:]:
        cur = None
        for tok in re.finditer(r'(Monday|Tuesday|Wednesday|Thursday|Friday)|(\d+:\d+) to (\d+:\d+)', line):
            if tok.group(1): cur = tok.group(1)
            elif cur and ('during' in line): busy[cur].append((t2m(tok.group(2)), t2m(tok.group(3))))
    return dict(dur=dur, lo=lo, hi=hi, days=days, busy=busy)


def cal_free(p):
    out = {}
    for d in p['days']:
        slots = []
        for s in range(p['lo'], p['hi'] - p['dur'] + 1, 30):
            if all(e <= s or b >= s + p['dur'] for b, e in p['busy'][d]): slots.append(s)
        out[d] = slots
    return out


def cal_check(item, resp, constructive, ranked=True):
    p = cal_problem(item['prompt_0shot'])
    m = re.findall(r'([A-Za-z]+), ([0-9]+:[0-9]+) - ([0-9]+:[0-9]+)', resp)
    issues = []
    if not m: issues.append('No proposed time in the required format "Day, HH:MM - HH:MM" was found.')
    else:
        d, s, e = m[0][0], t2m(m[0][1]), t2m(m[0][2])
        if d not in p['days']: issues.append(f'{d} is not one of the allowed days ({", ".join(p["days"])}).')
        if e - s != p['dur']: issues.append(f'The meeting must last {p["dur"]} minutes, but {m2t(s)}-{m2t(e)} lasts {e - s}.')
        if s < p['lo'] or e > p['hi']: issues.append(f'{m2t(s)}-{m2t(e)} is outside work hours {m2t(p["lo"])}-{m2t(p["hi"])}.')
        clash = [f'{m2t(b)}-{m2t(x)}' for b, x in p['busy'].get(d, []) if not (x <= s or b >= e)]
        if clash: issues.append(f'{d} {m2t(s)}-{m2t(e)} overlaps existing busy blocks: {", ".join(clash[:6])}.')
    hints = []
    if constructive and issues:
        free = cal_free(p)
        if ranked:
            for d in p['days']:
                if free[d]: hints.append(f'{d}: slots of {p["dur"]} minutes free for every participant start at ' + ', '.join(m2t(s) for s in free[d][:12]))
        else:
            import random
            rng = random.Random(len(item['prompt_0shot']))
            for d in p['days']:
                cand = list(range(p['lo'], p['hi'] - p['dur'] + 1, 30)); rng.shuffle(cand)
                hints.append(f'{d}: possible start times ' + ', '.join(m2t(s) for s in sorted(cand[:len(free[d]) or 3][:12])))
    return issues, hints


# ---------------------------------------------------------------- meeting
def tm(s): return datetime.datetime.strptime(s, '%I:%M%p')


def tstr(t): return t.strftime('%-I:%M%p')


def meet_problem(item):
    c = item['constraints']; start_loc, start_time = c[0]
    people = {name: dict(loc=loc, s=tm(times.split('to')[0].strip()), e=tm(times.split('to')[1].strip()), dur=int(dur)) for name, loc, times, dur in c[1:]}
    return start_loc, tm(start_time), people, item['dist_matrix']


def meet_check(item, resp, constructive, ranked=True):
    start_loc, cur_t, people, dist = meet_problem(item)
    plan = resp[resp.find('SOLUTION:') + 9:] if 'SOLUTION:' in resp else resp
    steps = [s.strip() for s in plan.split('.') if s.strip()]
    loc, met, issues = start_loc, [], []
    for step in steps:
        try:
            if step.startswith('You start'): continue
            if step.startswith('You travel'):
                dest = step.split('travel to ')[1].split(' in')[0].strip()
                if dest not in dist.get(loc, {}): raise ValueError(f'there is no travel time from {loc} to "{dest}"')
                cur_t += datetime.timedelta(minutes=dist[loc][dest]); loc = dest
            elif step.startswith('You wait'):
                end = tm(step.split('wait until ')[1].split('.')[0].strip())
                if end <= cur_t: raise ValueError(f'cannot wait until {tstr(end)} because it is already {tstr(cur_t)}')
                cur_t = end
            elif step.startswith('You meet'):
                who = step.split('meet ')[1].split(' for')[0].strip()
                if who not in people: raise ValueError(f'"{who}" is not one of the friends')
                if who in met: raise ValueError(f'{who} is met twice')
                p = people[who]; new_t = cur_t + datetime.timedelta(minutes=p['dur'])
                if loc != p['loc']: raise ValueError(f'{who} is at {p["loc"]}, but you are at {loc}')
                if cur_t < p['s']: raise ValueError(f'{who} is only available from {tstr(p["s"])}, but the meeting starts at {tstr(cur_t)} (wait first)')
                if new_t > p['e']: raise ValueError(f'meeting {who} for {p["dur"]} minutes from {tstr(cur_t)} ends after {who} leaves at {tstr(p["e"])}')
                met.append(who); cur_t = new_t
            else: raise ValueError(f'unrecognized step "{step[:60]}"')
        except (ValueError, IndexError, KeyError) as e:
            issues.append(f'Step "{step[:90]}": {e}. The plan is valid only up to this step.'); break
    hints = []
    if constructive:
        feas = []
        for who, p in people.items():
            if who in met or p['loc'] not in dist.get(loc, {}) and p['loc'] != loc: continue
            arrive = cur_t + datetime.timedelta(minutes=dist[loc][p['loc']] if p['loc'] != loc else 0)
            st = max(arrive, p['s'])
            if st + datetime.timedelta(minutes=p['dur']) <= p['e']: feas.append((st, who, p, arrive))
        if ranked: feas.sort()
        else:
            import random; random.Random(len(resp)).shuffle(feas)
        if feas:
            where = f'after your last valid step (at {loc}, {tstr(cur_t)})'
            hints.append(f'Friends you can still meet {where}, earliest feasible start first: ' + '; '.join(
                f'{w} at {p["loc"]} (arrive {tstr(a)}, can start {tstr(s)}, needs {p["dur"]} min, leaves {tstr(p["e"])})' for s, w, p, a in feas[:8]))
            if not issues: issues.append(f'The plan is valid, but {len(feas)} more friend(s) can still be met after its last step; the goal is to meet as many friends as possible.')
    return issues, hints


# ---------------------------------------------------------------- trip
def trip_problem(prompt):
    head, rest = prompt.split('Here are the cities that have direct flights:')
    total = int(re.search(r'European cities for (\d+) days', head).group(1))
    body = head.split('commute between cities.', 1)[-1]
    stay = {}
    for pat in [r'stay in (\w[\w ]*?) for (\d+) days', r'spend (\d+) days in (\w[\w ]*?)\.', r'visit (\w[\w ]*?) for (\d+) days']:
        for m in re.finditer(pat, body):
            a, b = m.groups(); city, d = (b, a) if a.isdigit() else (a, b); stay[city.strip()] = int(d)
    events = []
    for m in re.finditer(r'in (\w[\w ]*?) between day (\d+) and day (\d+)', head): events.append((m.group(1).strip(), int(m.group(2)), int(m.group(3))))
    for m in re.finditer(r'[Ff]rom day (\d+) to day (\d+), there is a annual show you want to attend in (\w[\w ]*?)\.', head): events.append((m.group(3).strip(), int(m.group(1)), int(m.group(2))))
    for m in re.finditer(r'[Dd]uring day (\d+) and day (\d+), you have to attend a conference in (\w[\w ]*?)\.', head): events.append((m.group(3).strip(), int(m.group(1)), int(m.group(2))))
    flights = collections.defaultdict(set)
    fl = rest.split('Find a trip plan')[0]
    for m in re.finditer(r'from (\w[\w ]*?) to (\w[\w ]*?)(?:,|\.|\n|$)', fl): flights[m.group(1).strip()].add(m.group(2).strip())
    for m in re.finditer(r'(\w[\w ]*?) and (\w[\w ]*?)(?:,|\.|\n|$)', fl.replace('from ', '')):
        a, b = m.group(1).strip(), m.group(2).strip(); flights[a].add(b); flights[b].add(a)
    return total, stay, events, flights


def trip_check(item, resp, constructive, ranked=True, parse=None):
    total, stay, events, flights = trip_problem(item['prompt_0shot'])
    plan = parse(resp + '\n') if parse else []
    issues, hints = [], []
    if not plan: return ['No plan in the required format ("**Day X-Y:** Visit CITY for N days." / "**Day X:** Fly from A to B.") was found.'], hints
    cities = [c for c, _ in plan]
    for c in stay:
        if c not in cities: issues.append(f'{c} is missing; it must be visited for {stay[c]} days.')
    day = 1; spans = {}
    for c, n in plan:
        if c not in stay: issues.append(f'{c} is not one of the requested cities.')
        elif n != stay[c]: issues.append(f'{c} is visited for {n} days but must be visited for {stay[c]} days (the flight day counts for both cities).')
        spans[c] = (day, day + n - 1); day = day + n - 1
    if day != total: issues.append(f'The plan ends on day {day}, but the trip must last exactly {total} days.')
    for a, b in zip(cities, cities[1:]):
        if b not in flights.get(a, set()):
            issues.append(f'There is no direct flight from {a} to {b}.')
            if constructive: hints.append(f'Direct flights from {a} go to: ' + ', '.join(sorted(flights.get(a, set()))))
    for c, s, e in events:
        if c in spans and (spans[c][1] < s or spans[c][0] > e):
            issues.append(f'You must be in {c} between day {s} and day {e}, but the plan has {c} on days {spans[c][0]}-{spans[c][1]}.')
            if constructive: hints.append(f'{c} ({stay.get(c)} days) must overlap days {s}-{e}; for example start {c} on day {max(1, e - stay.get(c, 1) + 1)} to {s}.')
    if constructive and issues and ranked:
        fixed = sorted(events, key=lambda x: x[1])
        if fixed: hints.append('Date-anchored cities in chronological order: ' + '; '.join(f'{c} (days {s}-{e})' for c, s, e in fixed))
    if constructive and issues and not ranked:
        import random
        rng = random.Random(len(resp)); cs = list(stay); rng.shuffle(cs)
        hints = [f'Cities and connections: ' + '; '.join(f'{c} <-> {", ".join(sorted(flights.get(c, set())))}' for c in cs)]
    return issues, hints


# ---------------------------------------------------------------- driver
FIX = ('\n\nA checker that verified your answer against the problem statement found these problems:\n{issues}{hints}\n\n'
       'Give a corrected complete answer in exactly the same format as before.')


def solve(task, item, method, port, model, trip_parse):
    prompt = item['prompt_5shot']
    msgs = [dict(role='user', content=prompt)]
    resp = chat(msgs, port, model); log = []
    if method == 'direct': return resp, log
    constructive = method in ('cgf', 'cgf_unranked'); ranked = method != 'cgf_unranked'
    check = dict(calendar=cal_check, meeting=meet_check, trip=lambda i, r, c, ranked=True: trip_check(i, r, c, ranked, trip_parse))[task]
    best, (bi, _) = resp, check(item, resp, False)
    issues, hints = check(item, resp, constructive, ranked) if constructive else (bi, [])
    log.append(len(bi))
    for _ in range(3):
        if not issues: break
        fb = FIX.format(issues='\n'.join('- ' + x for x in issues), hints=('\nCandidates computed from the problem statement:\n' + '\n'.join('- ' + h for h in hints)) if hints else '')
        resp = chat(msgs + [dict(role='assistant', content=resp), dict(role='user', content=fb)], port, model)
        ci, _ = check(item, resp, False); log.append(len(ci))
        issues, hints = check(item, resp, constructive, ranked) if constructive else (ci, [])
        if len(ci) <= len(bi): best, bi = resp, ci
    return best, log


def score(task, data, preds, mods):
    ok = []
    for (k, item), pred in zip(data, preds):
        if task == 'calendar':
            ok.append(mods['cal'].compute_solve_rate([pred], [item['golden_plan']]) == 1.0)
        elif task == 'trip':
            ok.append(mods['trip'].compute_example_score(item['cities'], item['durations'], mods['trip'].parse_response(pred)) == 1.0)
        else:
            m = mods['meet']; sl, it = item['constraints'][0]; cons = m.process_constraints(item['constraints'][1:])
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                g = m.validator_from_text(item['golden_plan'], cons, sl, it, item['dist_matrix'])
                try: s = m.validator_from_text(m.parse_text_plan(pred), cons, sl, it, item['dist_matrix'])
                except Exception: s = -1  # malformed plan crashes the official validator -> counted as unsolved
            ok.append(s == g)
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--task'); ap.add_argument('--method'); ap.add_argument('--model'); ap.add_argument('--ports')
    ap.add_argument('--out'); ap.add_argument('--limit', type=int); ap.add_argument('--workers', type=int, default=256)
    a = ap.parse_args(); ports = [int(x) for x in a.ports.split(',')]
    mods = dict(cal=load_official('evaluate_calendar_scheduling'), meet=load_official('evaluate_meeting_planning'), trip=load_official('evaluate_trip_planning'))
    fname = dict(calendar='calendar_scheduling', meeting='meeting_planning', trip='trip_planning')[a.task]
    data = list(json.load(open(NP / 'data' / f'{fname}.json')).items())
    if a.limit: data = data[:a.limit]
    t0 = time.time(); res = [None] * len(data)

    def job(i):
        try: return i, *solve(a.task, data[i][1], a.method, ports[i % len(ports)], a.model, mods['trip'].parse_response)
        except Exception as e: return i, '', ['error:' + repr(e)[:200]]
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for i, pred, log in pool.map(job, range(len(data))): res[i] = (pred, log)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out + '.preds.json', 'w') as f: json.dump([dict(id=k, pred=r[0], log=r[1]) for (k, _), r in zip(data, res)], f)
    ok = score(a.task, data, [r[0] for r in res], mods)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w') as f:
        for (k, item), (pred, log), o in zip(data, res, ok): f.write(json.dumps(dict(id=k, pred=pred, log=log, ok=o, n=item.get('num_people') or item.get('num_cities'))) + '\n')
    meta = dict(task=a.task, method=a.method, model=a.model, n=len(data), acc=round(100 * sum(ok) / len(ok), 2), seconds=round(time.time() - t0),
                errors=sum(1 for r in res if any(str(x).startswith('error') for x in r[1])), **USAGE)
    Path(a.out + '.meta.json').write_text(json.dumps(meta)); print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
