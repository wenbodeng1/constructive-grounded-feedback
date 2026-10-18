"""Staged and feedback-gated staged (FGS+CGF) delivery on NATURAL PLAN meeting / trip.

python np_staged.py --task trip --method staged|fgs_cgf --model np-qwen35-9b --ports ... --out runs/q35/trip_staged.jsonl
Same official 5-shot prompt as Direct in every stage (information-matched); the final plan
is assembled in the official answer format and scored with the official evaluator.
"""
import argparse, concurrent.futures as cf, datetime, json, re, time, urllib.request
from pathlib import Path
import np_run as R

SKEL_SCHEMA = dict(type='object', properties=dict(order=dict(type='array', items=dict(type='string'))), required=['order'], additionalProperties=False)
DAYS_SCHEMA = dict(type='object', properties=dict(days=dict(type='integer')), required=['days'], additionalProperties=False)
STEPS_SCHEMA = dict(type='object', properties=dict(steps=dict(type='string')), required=['steps'], additionalProperties=False)


def chat_json(messages, schema, port, model, max_tokens=1024):
    body = dict(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens, chat_template_kwargs={'enable_thinking': False},
                response_format=dict(type='json_schema', json_schema=dict(name='out', schema=schema)))
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
    with R.LOCK:
        R.USAGE['calls'] += 1; R.USAGE['prompt'] += r['usage']['prompt_tokens']; R.USAGE['completion'] += r['usage']['completion_tokens']
    return json.loads(r['choices'][0]['message']['content'] or '{}')


def u(prompt, extra): return [dict(role='user', content=prompt + '\n\n' + extra)]


def hint_block(issues, hints):
    s = '\nA checker that verified this against the problem statement found these problems:\n' + '\n'.join('- ' + x for x in issues)
    if hints: s += '\nCandidates computed from the problem statement:\n' + '\n'.join('- ' + h for h in hints)
    return s


# ---------------------------------------------------------------- trip
def trip_render(order, stays, total):
    lines = [f'Here is the trip plan for visiting the {len(order)} European cities for {total} days:', '']
    day = 1
    for i, (c, n) in enumerate(zip(order, stays)):
        end = day + n - 1
        lines.append(f'**Day {day}-{end}:** ' + (f'Arriving in {c} and visit {c} for {n} days.' if i == 0 else f'Visit {c} for {n} days.'))
        if i + 1 < len(order): lines.append(f'**Day {end}:** Fly from {c} to {order[i + 1]}.')
        day = end
    return '\n'.join(lines)


def trip_skeleton_issues(item, order):
    total, stay, events, flights = R.trip_problem(item['prompt_0shot'])
    issues, hints = [], []
    missing = [c for c in stay if c not in order]; extra = [c for c in order if c not in stay]
    if missing: issues.append('The order is missing: ' + ', '.join(missing) + '.')
    if extra: issues.append('These are not requested cities: ' + ', '.join(extra) + '.')
    for a, b in zip(order, order[1:]):
        if b not in flights.get(a, set()):
            issues.append(f'There is no direct flight from {a} to {b}.'); hints.append(f'Direct flights from {a} go to: ' + ', '.join(sorted(flights.get(a, set()))))
    day, spans = 1, {}
    for c in order:
        n = stay.get(c, 1); spans[c] = (day, day + n - 1); day = day + n - 1
    for c, s, e in events:
        if c in spans and (spans[c][1] < s or spans[c][0] > e):
            issues.append(f'With the required stay lengths, {c} would fall on days {spans[c][0]}-{spans[c][1]}, but you must be there between day {s} and day {e}.')
    if issues and events: hints.append('Date-anchored cities in chronological order: ' + '; '.join(f'{c} (days {s}-{e})' for c, s, e in sorted(events, key=lambda x: x[1])))
    return issues, hints


def solve_trip(item, method, port, model, parse):
    prompt = item['prompt_5shot']; total = R.trip_problem(item['prompt_0shot'])[0]
    ask0 = 'Stage 1 of the planning task: decide only the order in which to visit the cities. Output JSON {"order": [city, ...]}.'
    order = chat_json(u(prompt, ask0), SKEL_SCHEMA, port, model)['order']; log = []
    if method == 'fgs_cgf':
        for _ in range(2):
            iss, hints = trip_skeleton_issues(item, order); log.append(len(iss))
            if not iss: break
            order = chat_json(u(prompt, ask0 + '\n\nYour previous order: ' + json.dumps(order) + hint_block(iss, hints) + '\nOutput a corrected order as JSON.'), SKEL_SCHEMA, port, model)['order']
    stays = []
    _, stay_req, _, _ = R.trip_problem(item['prompt_0shot'])
    for k, c in enumerate(order):
        done = '; '.join(f'{x}: {n} days' for x, n in zip(order, stays)) or 'none'
        ask = (f'The order of cities is fixed: {" -> ".join(order)}. The flight day counts as a day in both cities. '
               f'Stays already decided: {done}. Now decide only how many days to stay in {c}. Output JSON {{"days": n}}.')
        n = int(chat_json(u(prompt, ask), DAYS_SCHEMA, port, model)['days'])
        if method == 'fgs_cgf' and c in stay_req and n != stay_req[c]:
            n2 = int(chat_json(u(prompt, ask + hint_block([f'{c} must be visited for {stay_req[c]} days, not {n}.'], [])), DAYS_SCHEMA, port, model)['days'])
            n = n2
        stays.append(n)
    plan = trip_render(order, stays, total)
    if method == 'fgs_cgf': plan, rl = final_cgf('trip', item, prompt, plan, port, model, parse); log += rl
    return plan, log


# ---------------------------------------------------------------- meeting
def meet_greedy(item, order):
    """Greedy earliest schedule for a fixed order (used only to report infeasible friends in the skeleton)."""
    start_loc, t, people, dist = R.meet_problem(item)
    loc, bad = start_loc, []
    for w in order:
        p = people.get(w)
        if p is None: bad.append(f'"{w}" is not one of the friends.'); continue
        if p['loc'] != loc and p['loc'] not in dist.get(loc, {}): bad.append(f'No travel time from {loc} to {p["loc"]} for {w}.'); continue
        arrive = t + datetime.timedelta(minutes=dist[loc][p['loc']] if p['loc'] != loc else 0)
        st = max(arrive, p['s'])
        if st + datetime.timedelta(minutes=p['dur']) > p['e']: bad.append(f'{w} cannot be met for {p["dur"]} minutes in this order (earliest start {R.tstr(st)}, leaves {R.tstr(p["e"])}).'); continue
        t, loc = st + datetime.timedelta(minutes=p['dur']), p['loc']
    return bad


def solve_meeting(item, method, port, model, parse=None):
    prompt = item['prompt_5shot']; start_loc, t0, people, _ = R.meet_problem(item)
    ask0 = 'Stage 1 of the planning task: decide only which friends to meet and in which order. Output JSON {"order": [name, ...]}.'
    order = chat_json(u(prompt, ask0), SKEL_SCHEMA, port, model)['order']; log = []
    if method == 'fgs_cgf':
        for _ in range(2):
            bad = meet_greedy(item, order); log.append(len(bad))
            if not bad: break
            _, hints = R.meet_check(item, 'SOLUTION: You start at ' + start_loc + ' at ' + R.tstr(t0) + '.', True)
            order = chat_json(u(prompt, ask0 + '\n\nYour previous order: ' + json.dumps(order) + hint_block(bad, hints) + '\nOutput a corrected order as JSON.'), SKEL_SCHEMA, port, model)['order']
    text = f'SOLUTION: You start at {start_loc} at {R.tstr(t0)}.'
    for w in order:
        ask = (f'Friends to meet, in this fixed order: {", ".join(order)}. Schedule written so far: "{text}" '
               f'Now write only the next steps (travel, wait, meet) for meeting {w}, using exactly the sentence format of the examples. Output JSON {{"steps": "..."}}.')
        steps = chat_json(u(prompt, ask), STEPS_SCHEMA, port, model)['steps'].strip()
        if method == 'fgs_cgf':
            before, _ = R.meet_check(item, text, False); after, hints = R.meet_check(item, text + ' ' + steps, True)
            new = [x for x in after if x not in before and 'more friend' not in x]
            if new: steps = chat_json(u(prompt, ask + '\nYour previous steps: "' + steps + '"' + hint_block(new, hints)), STEPS_SCHEMA, port, model)['steps'].strip()
        text = text + ' ' + steps
    if method == 'fgs_cgf': text, rl = final_cgf('meeting', item, prompt, text, port, model, parse); log += rl
    return text, log


def final_cgf(task, item, prompt, plan, port, model, parse):
    check = (lambda i, r, c: R.meet_check(i, r, c)) if task == 'meeting' else (lambda i, r, c: R.trip_check(i, r, c, True, parse))
    msgs = [dict(role='user', content=prompt)]
    best, (bi, _) = plan, check(item, plan, False); issues, hints = check(item, plan, True); log = [len(bi)]; resp = plan
    for _ in range(3):
        if not issues: break
        fb = R.FIX.format(issues='\n'.join('- ' + x for x in issues), hints=('\nCandidates computed from the problem statement:\n' + '\n'.join('- ' + h for h in hints)) if hints else '')
        resp = R.chat(msgs + [dict(role='assistant', content=resp), dict(role='user', content=fb)], port, model)
        ci, _ = check(item, resp, False); log.append(len(ci)); issues, hints = check(item, resp, True)
        if len(ci) <= len(bi): best, bi = resp, ci
    return best, log


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--task'); ap.add_argument('--method'); ap.add_argument('--model'); ap.add_argument('--ports')
    ap.add_argument('--out'); ap.add_argument('--limit', type=int); ap.add_argument('--workers', type=int, default=256)
    a = ap.parse_args(); ports = [int(x) for x in a.ports.split(',')]
    mods = dict(cal=R.load_official('evaluate_calendar_scheduling'), meet=R.load_official('evaluate_meeting_planning'), trip=R.load_official('evaluate_trip_planning'))
    fname = dict(meeting='meeting_planning', trip='trip_planning')[a.task]
    data = list(json.load(open(R.NP / 'data' / f'{fname}.json')).items())
    if a.limit: data = data[:a.limit]
    solve = solve_trip if a.task == 'trip' else solve_meeting
    t0 = time.time(); res = [None] * len(data)

    def job(i):
        try: return i, *solve(data[i][1], a.method, ports[i % len(ports)], a.model, mods['trip'].parse_response)
        except Exception as e: return i, '', ['error:' + repr(e)[:200]]
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for i, pred, log in pool.map(job, range(len(data))): res[i] = (pred, log)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out + '.preds.json', 'w') as f: json.dump([dict(id=k, pred=r[0], log=r[1]) for (k, _), r in zip(data, res)], f)
    ok = R.score(a.task, data, [r[0] for r in res], mods)
    with open(a.out, 'w') as f:
        for (k, item), (pred, log), o in zip(data, res, ok): f.write(json.dumps(dict(id=k, pred=pred, log=log, ok=o, n=item.get('num_people') or item.get('num_cities'))) + '\n')
    meta = dict(task=a.task, method=a.method, model=a.model, n=len(data), acc=round(100 * sum(ok) / len(ok), 2), seconds=round(time.time() - t0),
                errors=sum(1 for r in res if any(str(x).startswith('error') for x in r[1])), **R.USAGE)
    Path(a.out + '.meta.json').write_text(json.dumps(meta)); print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
