"""Grounded commitment ledger and validator for TravelPlanner plans.

Uses only the sandbox database (the same data the benchmark's search tools expose to
agents) and the plan itself. It does not read gold plans, the parsed local_constraint
annotation, or the official evaluator. Hard preference constraints (cuisine, room
rule, room type, transport preference) are deliberately NOT checked here.
"""
import math
import os
import re
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP_ROOT = os.environ.get('TP_ROOT', os.path.join(_REPO, 'third_party', 'TravelPlanner'))
DB = os.environ.get('TP_DB', os.path.join(TP_ROOT, 'database'))
REST = pd.read_csv(f'{DB}/restaurants/clean_restaurant_2022.csv')
ATTR = pd.read_csv(f'{DB}/attractions/attractions.csv')
ACC = pd.read_csv(f'{DB}/accommodations/clean_accommodations_2022.csv')
ACC_OK = ACC.dropna(subset=['price', 'minimum nights', 'maximum occupancy'])
FLT = pd.read_csv(f'{DB}/flights/clean_Flights_2022.csv')
DIST = pd.read_csv(f'{DB}/googleDistanceMatrix/distance.csv')
FLT_BY_NO = {r['Flight Number']: r for r in FLT.to_dict('records')}


def name_city(text):
    text = re.sub(r'\s*\(.*?\)\s*$', '', text.strip().rstrip(';').strip())
    if ',' not in text: return text.strip(), None
    name, city = text.rsplit(',', 1)
    return name.strip(), city.strip()


def from_to(text):
    m = re.search(r'from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)', text or '')
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)


def lookup(df, name_col, city_col, name, city):
    if not name or not city: return None
    hit = df[(df[name_col].astype(str) == name) & (df[city_col] == city)]
    if len(hit) == 0: hit = df[(df[name_col].astype(str).str.contains(re.escape(name), regex=True)) & (df[city_col] == city)]
    return hit.iloc[0].to_dict() if len(hit) else None


def transport_cost(text, current_city, people):
    t = (text or '').strip()
    if not t or t == '-': return 0, None, None
    low = t.lower()
    if 'flight number' in low:
        no = t.split('Flight Number:')[1].split(',')[0].strip()
        f = FLT_BY_NO.get(no)
        if f is None: return 0, 'Flight', f'flight {no} does not exist in the data'
        o, d = from_to(t)
        if o and d and (f['OriginCityName'] != o or f['DestCityName'] != d):
            return f['Price'] * people, 'Flight', f'flight {no} goes {f["OriginCityName"]}->{f["DestCityName"]}, not {o}->{d}'
        return f['Price'] * people, 'Flight', None
    mode = 'Self-driving' if 'self-driving' in low else 'Taxi' if 'taxi' in low else None
    if mode is None: return 0, None, f'unrecognized transportation "{t[:60]}"'
    o, d = from_to(t)
    if not o: o, d = from_to(current_city)
    hit = DIST[(DIST['origin'] == o) & (DIST['destination'] == d)] if o else DIST.iloc[0:0]
    if len(hit) == 0 or not isinstance(hit.iloc[0]['distance'], str) or 'day' in str(hit.iloc[0]['duration']):
        return 0, mode, f'no {mode.lower()} route from {o} to {d} in the data'
    km = float(hit.iloc[0]['distance'].replace('km', '').replace(',', ''))
    per = int(km * 0.05) if mode == 'Self-driving' else int(km)
    return per * math.ceil(people / (5 if mode == 'Self-driving' else 4)), mode, None


def audit(plan, people, budget):
    """Return (ledger dict, list of violation strings) for a (possibly partial) plan."""
    used_rest, used_attr, items, issues, modes, total = {}, {}, [], [], set(), 0
    for d in plan:
        day = d.get('days')
        cc = d.get('current_city', '')
        cost, mode, err = transport_cost(d.get('transportation'), cc, people)
        total += cost; modes.add(mode) if mode else None
        if cost: items.append(f'day {day} transportation ${cost}')
        if err: issues.append(f'Day {day} transportation: {err}.')
        for meal in ('breakfast', 'lunch', 'dinner'):
            v = (d.get(meal) or '-').strip()
            if v in ('-', ''): continue
            n, c = name_city(v); r = lookup(REST, 'Name', 'City', n, c)
            if r is None: issues.append(f'Day {day} {meal}: "{v}" is not a restaurant listed for that city in the given data.')
            elif r['Average Cost'] == r['Average Cost']: total += r['Average Cost'] * people; items.append(f'day {day} {meal} ${r["Average Cost"] * people}')
            if v in used_rest: issues.append(f'Day {day} {meal}: restaurant "{v}" was already used on {used_rest[v]}; every meal must use a different restaurant.')
            else: used_rest[v] = f'day {day} {meal}'
        for a in [x.strip() for x in (d.get('attraction') or '-').split(';') if x.strip() and x.strip() != '-']:
            n, c = name_city(a)
            if lookup(ATTR, 'Name', 'City', n, c) is None: issues.append(f'Day {day} attraction "{a}" is not listed for that city in the given data.')
            if a in used_attr: issues.append(f'Day {day}: attraction "{a}" was already visited on day {used_attr[a]}.')
            else: used_attr[a] = day
        v = (d.get('accommodation') or '-').strip()
        if v not in ('-', ''):
            n, c = name_city(v); r = lookup(ACC, 'NAME', 'city', n, c)
            if r is None: issues.append(f'Day {day} accommodation "{v}" is not listed for that city in the given data.')
            elif r['maximum occupancy'] == r['maximum occupancy'] and r['price'] == r['price']:
                rooms = math.ceil(people / r['maximum occupancy']); total += r['price'] * rooms; items.append(f'day {day} accommodation ${r["price"] * rooms}')
    # accommodation stays and minimum nights
    stays, prev = [], None
    for d in plan:
        v = (d.get('accommodation') or '-').strip()
        if stays and v == prev: stays[-1][1] += 1
        else: stays.append([v, 1])
        prev = v
    for v, nights in stays:
        if v in ('-', ''): continue
        n, c = name_city(v); r = lookup(ACC, 'NAME', 'city', n, c)
        if r is not None and nights < r['minimum nights']:
            issues.append(f'Accommodation "{v}" is booked for {nights} consecutive night(s) but requires a minimum of {int(r["minimum nights"])} nights.')
    if 'Self-driving' in modes and ({'Flight', 'Taxi'} & modes):
        issues.append('Transportation modes conflict: self-driving cannot be combined with flights or taxis in one trip.')
    if budget is not None and total > budget:
        issues.append(f'Total cost ${total:.0f} exceeds the budget ${budget}.')
    ledger = dict(used_restaurants=list(used_rest), used_attractions=list(used_attr), spent=round(float(total)), budget=budget,
                  remaining=None if budget is None else round(float(budget - total)), modes=sorted(modes), stays=stays)
    return ledger, issues


def render(ledger, n_days, k):
    lines = ['LEDGER of commitments made so far (computed from the given data):',
             f'- Restaurants already used (must not be used again): {ledger["used_restaurants"] or "none"}',
             f'- Attractions already visited (must not be visited again): {ledger["used_attractions"] or "none"}',
             f'- Spent so far: ${ledger["spent"]}' + (f'; budget ${ledger["budget"]}; remaining ${ledger["remaining"]} for the remaining {n_days - k} day(s).' if ledger['budget'] is not None else ''),
             f'- Transportation modes used: {ledger["modes"] or "none"} (self-driving cannot be mixed with flights or taxis).']
    return '\n'.join(lines)


# ---------------- constructive feedback: candidate repairs looked up in the data ----------------
import datetime as _dt

MONTHS = {m: i for i, m in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'], 1)}


def start_date(query):
    m = re.search(r'(' + '|'.join(MONTHS) + r')\s+(\d{1,2})(?:st|nd|rd|th)?', query, re.I)
    y = re.search(r'\b(20\d\d)\b', query)
    if not m: return None
    return _dt.date(int(y.group(1)) if y else 2022, MONTHS[m.group(1).lower()], int(m.group(2)))


def legs_and_stays(plan):
    legs, stays, cur = [], {}, None
    for d in plan:
        o, t = from_to(d.get('current_city', ''))
        if o: legs.append((d.get('days'), o, t)); cur = t
        else: cur = (d.get('current_city') or '').strip() or cur
        v = (d.get('accommodation') or '-').strip()
        if v not in ('-', '') and cur: stays[cur] = stays.get(cur, 0) + 1
    return legs, stays


def _money(x): return f'${int(round(float(x)))}'


def suggestions(plan, query, people, issues, k=5, ranked=True):
    """Short lists of data-grounded alternatives relevant to the reported problems."""
    text = ' '.join(issues).lower(); out = []
    legs, stays = legs_and_stays(plan)
    used = {(d.get(m) or '').strip() for d in plan for m in ('breakfast', 'lunch', 'dinner')}
    budget_issue = 'exceeds the budget' in text
    if budget_issue or 'minimum of' in text or 'accommodation' in text:
        for city, nights in stays.items():
            acc = (ACC_OK[(ACC_OK['city'] == city) & (ACC_OK['minimum nights'] <= nights)] if ranked else ACC_OK[ACC_OK['city'] == city]).copy()
            if len(acc) == 0: continue
            acc['cost'] = acc['price'] * ((people + acc['maximum occupancy'] - 1) // acc['maximum occupancy'])
            rows = acc.sort_values('cost').head(k) if ranked else acc.sample(min(k, len(acc)), random_state=0)
            if ranked:
                out.append(f'Accommodations in {city} allowing a {nights}-night stay (cheapest first, total per night for {people} people): ' +
                           '; '.join(f'"{r.NAME}, {city}" {_money(r.cost)}/night ({r._3}, max {int(r._8)} guests, rules: {r.house_rules})' for r in rows.itertuples()))
            else:
                out.append(f'Accommodations in {city} (total per night for {people} people): ' +
                           '; '.join(f'"{r.NAME}, {city}" {_money(r.cost)}/night ({r._3}, min {int(r._5)} nights, max {int(r._8)} guests, rules: {r.house_rules})' for r in rows.itertuples()))
    if budget_issue or 'restaurant' in text:
        for city in stays or {t for _, _, t in legs}:
            rest = REST[(REST['City'] == city) & REST['Average Cost'].notna()]
            rest = rest.sort_values('Average Cost') if ranked else rest.sample(frac=1, random_state=0)
            rest = (rest[~rest.apply(lambda r: f'{r.Name}, {city}' in used, axis=1)] if ranked else rest).head(k + 3)
            if len(rest): out.append((f'Unused restaurants in {city} (cheapest first, per person): ' if ranked else f'Restaurants in {city} (per person): ') + '; '.join(f'"{r.Name}, {city}" {_money(r._5)} ({r.Cuisines})' for r in rest.itertuples()))
    if budget_issue or 'transportation' in text or 'flight' in text:
        s = start_date(query)
        for day, o, t in legs:
            opts = []
            if s and isinstance(day, int):
                date = (s + _dt.timedelta(days=day - 1)).isoformat()
                fl = FLT[(FLT['OriginCityName'] == o) & (FLT['DestCityName'] == t) & (FLT['FlightDate'] == date) & FLT['Price'].notna()]
                fl = fl.sort_values('Price').head(3) if ranked else fl.sample(min(3, len(fl)), random_state=0)
                opts += [f'Flight Number: {r._1}, {r.DepTime}-{r.ArrTime}, {_money(r.Price * people)} total' for r in fl.itertuples()]
            for mode in ('Self-driving', 'Taxi'):
                c, _, err = transport_cost(f'{mode}, from {o} to {t}', '', people)
                if not err: opts.append(f'{mode} {_money(c)} total')
            out.append(f'Day {day} options from {o} to {t}: ' + ('; '.join(opts) if opts else 'none found in the data'))
    if 'attraction' in text:
        used_a = {a.strip() for d in plan for a in (d.get('attraction') or '').split(';')}
        for city in stays:
            at = [f'{n}, {city}' for n in ATTR[ATTR['City'] == city]['Name'].astype(str) if f'{n}, {city}' not in used_a][:k + 3]
            if at: out.append(f'Unvisited attractions in {city}: ' + '; '.join(at))
    return out
