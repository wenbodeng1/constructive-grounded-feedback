"""TravelPlanner sole-planning experiments: all-at-once vs staged delivery (Qwen3.5-9B).

python run_tp.py --method direct --out runs/direct.jsonl [--limit N]
Methods are defined in METHODS below. Plans are written in the official evaluation
format (one JSON line per validation query, same order as the HF split).
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

from datasets import load_dataset

MODEL = os.environ.get('TP_MODEL', 'qwen3.5-9b')  # served model name (OpenAI-compatible vLLM endpoint)
PORTS = [int(x) for x in os.environ.get('TP_PORTS', '8000').split(',')]
FIELDS = ['days', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']
LOCK = threading.Lock()
USAGE = dict(calls=0, prompt=0, completion=0)

EXAMPLE = [
    dict(days=1, current_city='from Ithaca to Charlotte', transportation='Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46',
         breakfast="Nagaland's Kitchen, Charlotte", attraction='The Charlotte Museum of History, Charlotte;', lunch='Cafe Maple Street, Charlotte',
         dinner='Bombay Vada Pav, Charlotte', accommodation='Affordable Spacious Refurbished Room in Bushwick!, Charlotte'),
    dict(days=2, current_city='Charlotte', transportation='-', breakfast='Olive Tree Cafe, Charlotte',
         attraction='The Mint Museum, Charlotte;Romare Bearden Park, Charlotte;', lunch='Birbal Ji Dhaba, Charlotte',
         dinner='Pind Balluchi, Charlotte', accommodation='Affordable Spacious Refurbished Room in Bushwick!, Charlotte'),
    dict(days=3, current_city='from Charlotte to Ithaca', transportation='Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26',
         breakfast='Subway, Charlotte', attraction='Books Monument, Charlotte;', lunch='Olive Tree Cafe, Charlotte', dinner='Kylin Skybar, Charlotte', accommodation='-'),
]

# Official PLANNER_INSTRUCTION wording, with the example rendered in the JSON output format.
INSTRUCTION = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'current_city' field as in the example (i.e., from A to B). Self-driving or taxi transportation is written like "Self-driving, from A to B, duration: ..., distance: ..., cost: ..." or "Taxi, from A to B, ...".

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Travel Plan (JSON):
""" + json.dumps(EXAMPLE, indent=1) + """
***** Example Ends *****

Given information: {text}
Query: {query}"""


def day_schema(n=None, fixed_day=None):
    item = dict(type='object', properties={f: (dict(type='integer') if f == 'days' else dict(type='string')) for f in FIELDS},
                required=FIELDS, additionalProperties=False)
    if fixed_day is not None: return item
    return dict(type='array', items=item, minItems=n, maxItems=n)


def chat(messages, schema=None, max_tokens=6000, port=None):
    body = dict(model=MODEL, messages=messages, temperature=0.0, max_tokens=max_tokens, chat_template_kwargs={'enable_thinking': False})
    if schema is not None: body['response_format'] = dict(type='json_schema', json_schema=dict(name='plan', schema=schema))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            with opener.open(req, timeout=1200) as h: r = json.load(h)
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors='replace')
            if e.code == 400: raise RuntimeError('http400: ' + detail[:300])
            if attempt == 2: raise
        except OSError:
            if attempt == 2: raise
            time.sleep(5)
    with LOCK:
        USAGE['calls'] += 1; USAGE['prompt'] += r['usage']['prompt_tokens']; USAGE['completion'] += r['usage']['completion_tokens']
    return r['choices'][0]['message']['content'], r['choices'][0]['finish_reason']


def base_prompt(q):
    return INSTRUCTION.replace('{text}', q['reference_information']).replace('{query}', q['query'])


# ---------------- methods ----------------

def m_direct(q, port):
    out, fr = chat([dict(role='user', content=base_prompt(q) + '\nTravel Plan (JSON):')], day_schema(q['days']), port=port)
    return json.loads(out), dict(finish=fr)


def m_staged(q, port, local=False):
    """Stage 0: route skeleton for all days; stage k: details of day k given the skeleton
    and the days already planned (feedback from previous stages). Same information as direct."""
    n = q['days']
    skel_schema = dict(type='array', minItems=n, maxItems=n, items=dict(type='object', additionalProperties=False,
                       required=['days', 'current_city', 'transportation', 'accommodation'],
                       properties=dict(days=dict(type='integer'), current_city=dict(type='string'), transportation=dict(type='string'), accommodation=dict(type='string'))))
    p0 = base_prompt(q) + ('\n\nStage 1 of the planning task: decide only the route skeleton for every day: '
                           'days, current_city, transportation and accommodation (same formats as the example). Output JSON.')
    skel, _ = chat([dict(role='user', content=p0)], skel_schema, port=port)
    skel = json.loads(skel); plan = []
    for k in range(n):
        p = base_prompt(q) + '\n\nThe route skeleton has been fixed:\n' + json.dumps(skel, indent=1)
        if plan: p += '\n\nDays already planned (do not change them):\n' + json.dumps(plan, indent=1)
        p += f'\n\nNow plan only Day {k+1}: fill in all fields for day {k+1}, keeping its current_city, transportation and accommodation from the skeleton. Output one JSON object.'
        d, _ = chat([dict(role='user', content=p)], day_schema(fixed_day=k + 1), max_tokens=1500, port=port)
        d = json.loads(d); d['days'] = k + 1; plan.append(d)
    return plan, dict(skeleton=skel)


# ---------------- ledger / feedback methods ----------------
import ledger as L

WORDS = dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10, a=1, single=1, solo=1)


def parse_query(text):
    """Budget and party size from the query text itself (not from benchmark annotations)."""
    b = re.search(r'\$\s?([\d,]+)', text); budget = int(b.group(1).replace(',', '')) if b else None
    people = 1
    pats = [r'(?:group|party) of (\d+|\w+)', r'\b(\d+|\w+)\s+(?:people|persons|individuals|travelers|travellers|adults)\b', r'\bfor (\d+|two|three|four|five|six|seven|eight|nine|ten)\b(?!\s*(?:days|day|nights))']
    if re.search(r'\b(duo|pair|couple|both of us|two of us)\b', text, re.I): return budget, 2
    for pat in pats:
        m = re.search(pat, text, re.I)
        if m:
            w = m.group(1).lower(); v = int(w) if w.isdigit() else WORDS.get(w)
            if v: people = v; break
    return budget, people


def repair(q, plan, port, rounds=3, constructive=False, ranked=True):
    budget, people = parse_query(q['query'])
    best = plan; _, best_issues = L.audit(plan, people, budget); log = [len(best_issues)]
    cur, issues = plan, best_issues
    for _ in range(rounds):
        if not issues: break
        try:
            hints = ('\n\nCandidate replacements looked up in the given data:\n' + '\n'.join('- ' + x for x in L.suggestions(cur, q['query'], people, issues, ranked=ranked))) if constructive else ''
        except Exception as e:
            hints = ''; log.append('hint_error:' + repr(e)[:80])
        p = (base_prompt(q) + '\n\nCurrent travel plan (JSON):\n' + json.dumps(cur, indent=1) +
             '\n\nA checker that looked up every item in the given data found these problems:\n' + '\n'.join('- ' + x for x in issues) +
             hints +
             '\n\nRevise the plan so that all problems are fixed, keeping all other requirements of the query satisfied and using only items from the given information. Output the complete revised plan as JSON.')
        try:
            out, _ = chat([dict(role='user', content=p)], day_schema(q['days']), port=port)
            cur = json.loads(out)
        except Exception:
            break
        _, issues = L.audit(cur, people, budget); log.append(len(issues))
        if len(issues) <= len(best_issues): best, best_issues = cur, issues
    return best, log


def m_staged_ledger(q, port):
    n = q['days']; budget, people = parse_query(q['query'])
    skel_schema = dict(type='array', minItems=n, maxItems=n, items=dict(type='object', additionalProperties=False,
                       required=['days', 'current_city', 'transportation', 'accommodation'],
                       properties=dict(days=dict(type='integer'), current_city=dict(type='string'), transportation=dict(type='string'), accommodation=dict(type='string'))))
    p0 = base_prompt(q) + ('\n\nStage 1 of the planning task: decide only the route skeleton for every day: '
                           'days, current_city, transportation and accommodation (same formats as the example). Output JSON.')
    skel, _ = chat([dict(role='user', content=p0)], skel_schema, port=port)
    skel = json.loads(skel); plan = []
    for k in range(n):
        led, _ = L.audit(plan, people, budget)
        p = base_prompt(q) + '\n\nThe route skeleton has been fixed:\n' + json.dumps(skel, indent=1)
        if plan: p += '\n\nDays already planned (do not change them):\n' + json.dumps(plan, indent=1)
        p += '\n\n' + L.render(led, n, k)
        p += f'\n\nNow plan only Day {k+1}: fill in all fields for day {k+1}, keeping its current_city, transportation and accommodation from the skeleton, respecting the ledger. Output one JSON object.'
        d, _ = chat([dict(role='user', content=p)], day_schema(fixed_day=k + 1), max_tokens=1500, port=port)
        d = json.loads(d); d['days'] = k + 1; plan.append(d)
    return plan, dict(skeleton=skel)


def m_direct_repair(q, port):
    plan, info = m_direct(q, port); plan, log = repair(q, plan, port); return plan, dict(info, repair=log)


def m_staged_ledger_repair(q, port):
    plan, info = m_staged_ledger(q, port); plan, log = repair(q, plan, port); return plan, dict(info, repair=log)


METHODS = dict(direct=m_direct, staged=m_staged, staged_ledger=m_staged_ledger, direct_repair=m_direct_repair,
               staged_ledger_repair=m_staged_ledger_repair)


SKEL_FIELDS = ['days', 'current_city', 'transportation', 'accommodation']


def skel_schema(n):
    return dict(type='array', minItems=n, maxItems=n, items=dict(type='object', additionalProperties=False, required=SKEL_FIELDS,
                properties=dict(days=dict(type='integer'), current_city=dict(type='string'), transportation=dict(type='string'), accommodation=dict(type='string'))))


def as_plan(skel):
    return [dict(d, breakfast='-', lunch='-', dinner='-', attraction='-') for d in skel]


def m_staged_verified(q, port, skel_rounds=2, day_rounds=1, final_rounds=3, constructive=False):
    """Feedback-gated staging: each stage's output is checked against the data before the
    next stage is issued; the next stage receives the ledger of verified commitments."""
    n = q['days']; budget, people = parse_query(q['query']); log = dict(skeleton=[], days=[])
    p0 = base_prompt(q) + ('\n\nStage 1 of the planning task: decide only the route skeleton for every day: '
                           'days, current_city, transportation and accommodation (same formats as the example). '
                           'Leave enough budget for meals and attractions. Output JSON.')
    skel = json.loads(chat([dict(role='user', content=p0)], skel_schema(n), port=port)[0])
    for _ in range(skel_rounds + 1):
        led, issues = L.audit(as_plan(skel), people, budget); log['skeleton'].append(len(issues))
        if not issues or len(log['skeleton']) > skel_rounds: break
        p = (p0 + '\n\nCurrent skeleton:\n' + json.dumps(skel, indent=1) + f'\n\nFixed costs of this skeleton (transportation + accommodation): ${led["spent"]}.'
             '\nA checker that looked up every item in the given data found these problems:\n' + '\n'.join('- ' + x for x in issues) +
             (('\n\nCandidate replacements looked up in the given data:\n' + '\n'.join('- ' + x for x in L.suggestions(as_plan(skel), q['query'], people, issues))) if constructive else '') +
             '\n\nRevise the skeleton to fix all problems. Output the complete revised skeleton as JSON.')
        skel = json.loads(chat([dict(role='user', content=p)], skel_schema(n), port=port)[0])
    plan = []
    for k in range(n):
        led, _ = L.audit(plan, people, budget)
        base = base_prompt(q) + '\n\nThe verified route skeleton:\n' + json.dumps(skel, indent=1)
        if plan: base += '\n\nDays already planned and verified (do not change them):\n' + json.dumps(plan, indent=1)
        fixed_rest, _ = L.audit(as_plan(skel[k:]), people, None)
        base += '\n\n' + L.render(led, n, k) + f'\n- Remaining fixed costs from the skeleton (transportation + accommodation of day {k+1} onward): ${fixed_rest["spent"]}.'
        ask = f'\n\nNow plan only Day {k+1}: fill in all fields for day {k+1}, keeping its current_city, transportation and accommodation from the skeleton, respecting the ledger. Output one JSON object.'
        day = json.loads(chat([dict(role='user', content=base + ask)], day_schema(fixed_day=k + 1), max_tokens=1500, port=port)[0]); day['days'] = k + 1
        for r in range(day_rounds):
            _, before = L.audit(plan, people, None); _, after = L.audit(plan + [day], people, None)
            new_issues = [x for x in after if x not in before]
            if not new_issues: break
            hints = ('\nCandidate replacements looked up in the given data:\n' + '\n'.join('- ' + x for x in L.suggestions(plan + [day], q['query'], people, new_issues))) if constructive else ''
            fix = ask + '\n\nYour previous Day ' + str(k + 1) + ' proposal:\n' + json.dumps(day, indent=1) + '\nA checker found these problems:\n' + '\n'.join('- ' + x for x in new_issues) + hints + '\nOutput a corrected Day ' + str(k + 1) + ' JSON object.'
            day = json.loads(chat([dict(role='user', content=base + fix)], day_schema(fixed_day=k + 1), max_tokens=1500, port=port)[0]); day['days'] = k + 1
        plan.append(day)
    plan, rlog = repair(q, plan, port, rounds=final_rounds, constructive=constructive); log['final'] = rlog
    return plan, log


METHODS['staged_verified'] = m_staged_verified


def m_direct_cgf(q, port):
    plan, info = m_direct(q, port); plan, log = repair(q, plan, port, constructive=True); return plan, dict(info, repair=log)


METHODS['direct_cgf'] = m_direct_cgf


def m_direct_cgf_unranked(q, port):
    plan, info = m_direct(q, port); plan, log = repair(q, plan, port, constructive=True, ranked=False); return plan, dict(info, repair=log)


METHODS['direct_cgf_unranked'] = m_direct_cgf_unranked
METHODS['staged_verified_cgf'] = lambda q, port: m_staged_verified(q, port, constructive=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--method', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int); ap.add_argument('--split', default='validation'); ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    data = list(load_dataset('osunlp/TravelPlanner', a.split)[a.split])
    if a.limit: data = data[:a.limit]
    fn = METHODS[a.method]; results = [None] * len(data); started = time.time()

    def job(i):
        q = data[i]
        try: plan, info = fn(q, PORTS[i % len(PORTS)])
        except Exception as e: plan, info = None, dict(error=repr(e)[:400])
        if plan is not None:
            for d in plan:
                for f in FIELDS:
                    d.setdefault(f, '-')
        return i, plan, info
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for i, plan, info in pool.map(job, range(len(data))):
            results[i] = dict(idx=i + 1, query=data[i]['query'], plan=plan, info=info)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w') as f:
        for r in results: f.write(json.dumps(r) + '\n')
    meta = dict(method=a.method, n=len(data), seconds=round(time.time() - started), errors=sum(r['plan'] is None for r in results), **USAGE)
    Path(a.out + '.meta.json').write_text(json.dumps(meta, indent=1)); print(meta)


if __name__ == '__main__':
    main()
