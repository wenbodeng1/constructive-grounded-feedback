"""Expand a compact high-level plan (LLM output) into an official ChinaTravel
plan JSON (chinatravel/evaluation/output_schema.json), filling every field
the official schema / commonsense / logical evaluators read.

HIGH-LEVEL PLAN (what the planner LLM outputs)
----------------------------------------------
{
  "itinerary": [
    {"day": 1,
     "activities": [
       {"type": "train" | "airplane" | "attraction" | "breakfast" | "lunch" |
                "dinner" | "accommodation",
        "name": "<exact sandbox POI name; for train/airplane the TrainID/FlightID>",
        "start_time": "HH:MM",
        "end_time": "HH:MM",
        "transport_mode": "walk" | "metro" | "taxi"}
     ]}
  ]
}
* transport_mode = how the traveller gets from the PREVIOUS activity's
  location to THIS activity's location; for a return train/airplane it is the
  mode used to reach its departure station.  Ignored for the first activity of
  the trip and when both locations are identical.  Missing/invalid -> "taxi"
  (logged in notes).
* For train/airplane, start_time/end_time are ignored (official times are
  taken from the sandbox by ID).  For POIs the planner's times are kept
  verbatim (only "9:05" -> "09:05" zero-padding in lenient mode).
* breakfast may be taken at a hotel: give the hotel's exact name (price 0).
* Days are numbered by list order (the planner's "day" values are ignored).
* Extra keys are ignored.  rooms / taxi cars / tickets are NOT planner
  choices: they are derived (rooms = ceil(people/beds), cars =
  ceil(people/4), tickets = people) -- see SEMANTICS.md section 4.
* HL_JSON_SCHEMA below is the exact JSON Schema of this format.

expand(query, high_level_plan, lenient=True) -> official plan dict
expand_with_notes(...) -> (official plan dict, notes list[str])

Filling rules (they mirror the official RuleNeSy agent and the evaluator):
* plan-level: people_number, start_city, target_city from the query.
* train/airplane: looked up by ID in the sandbox table of the direction
  (first intercity activity of the plan = outbound start->target, any later
  one = return target->start) and the declared type; start/end =
  From/To station, start_time/end_time = BeginTime/EndTime, price = Cost,
  tickets = people_number, cost = price*tickets; TrainID/FlightID key by type.
* attraction: price = sandbox ticket price, tickets = people_number,
  cost = price*tickets.
* breakfast/lunch/dinner at a restaurant: price = sandbox price/person,
  cost = price*people_number.  breakfast at a hotel: price = cost = 0.
* accommodation: price = sandbox price/room/night, room_type = numbed,
  rooms = ceil(people_number/numbed), cost = price*rooms.
* transports (inner-city, computed with the environment's own
  Transportation.goto of the English sandbox, exactly as the evaluator
  recomputes them):
    - origin = previous activity's position (or the arrival station of a
      preceding train/airplane), destination = this activity's position (or
      the departure station of a return train/airplane); identical -> [].
    - departure = previous activity's end_time when the previous activity is
      on the same day; for the first activity of a day >= 2 the traveller
      leaves "just in time": departure = start_time - travel duration
      (durations are time-invariant in goto), floored at 00:00.
    - metro -> 3 legs walk/metro/walk; the metro leg gets price = goto cost,
      tickets = people_number, cost = price*tickets; walk legs price=cost=0.
      If goto(metro) has no solution (both endpoints share the nearest
      station) we fall back to walk -- the behaviour the environment code
      itself intends ("Too near. Walk.") -- and log it.
    - taxi -> price = goto cost, cars = ceil(people_number/4), cost = price*cars.
    - walk -> price = cost = 0.
  Arrival later than the planner's start_time is NOT corrected: the official
  evaluator flags it.
* Robustness: never raises.  Unknown names / IDs are emitted anyway (POI:
  position=<name>, price=cost=0; intercity: start=end="") so the official
  evaluator marks them ungrounded; failing goto -> transports [] (logged).
* lenient=True (default) additionally: case/whitespace/quote-insensitive
  canonicalisation of POI names to the sandbox spelling (first sandbox
  row with that normalised name), "K335/K338" merged train IDs -> first ID that
  exists, type synonyms hotel->accommodation, flight/plane->airplane,
  H:MM -> HH:MM.  Every such repair is logged in notes.  lenient=False
  disables all of them.
"""

import math
import re
import unicodedata

import ct_common
from ct_common import py_num

# JSON Schema (draft-07) of the high-level plan; usable for validation or guided decoding.
HL_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["itinerary"],
    "properties": {
        "itinerary": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["day", "activities"],
                "properties": {
                    "day": {"type": "integer"},
                    "activities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type", "name", "start_time", "end_time"],
                            "properties": {
                                "type": {"type": "string", "enum": ["train", "airplane", "attraction", "breakfast",
                                                                     "lunch", "dinner", "accommodation"]},
                                "name": {"type": "string"},
                                "start_time": {"type": "string", "pattern": "^\\d{2}:\\d{2}$"},
                                "end_time": {"type": "string", "pattern": "^\\d{2}:\\d{2}$"},
                                "transport_mode": {"type": "string", "enum": ["walk", "metro", "taxi"]},
                            },
                        },
                    },
                },
            },
        },
    },
}

POI_TYPES = ("attraction", "breakfast", "lunch", "dinner", "accommodation")
MEAL_TYPES = ("breakfast", "lunch", "dinner")
INTERCITY_TYPES = ("train", "airplane")
MODES = ("walk", "metro", "taxi")
DEFAULT_MODE = "taxi"
TYPE_SYNONYMS = {"hotel": "accommodation", "flight": "airplane", "plane": "airplane"}
_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*$")


# ------------------------------------------------------------ lookups
class _CityIndex:
    def __init__(self, city):
        t = ct_common.tools()
        self.attr = self._index(t["attractions"].data[city])
        self.rest = self._index(t["restaurants"].data[city])
        self.hotel = self._index(t["accommodations"].data[city])

    @staticmethod
    def _index(df):
        exact, norm = {}, {}
        for _, r in df.iterrows():
            n = r["name"]
            if n not in exact:  # evaluator uses .values[0] = first match
                exact[n] = r
            norm.setdefault(_norm(n), [])
            if n not in norm[_norm(n)]:
                norm[_norm(n)].append(n)
        return exact, norm


_CITY_CACHE = {}
_INTERCITY_CACHE = {}


def _city(city):
    if city not in _CITY_CACHE:
        _CITY_CACHE[city] = _CityIndex(city)
    return _CITY_CACHE[city]


def _intercity_index(start_city, end_city):
    key = (start_city, end_city)
    if key not in _INTERCITY_CACHE:
        trains, flights = ct_common.intercity_tables(start_city, end_city)
        idx = {"train": {}, "airplane": {}}
        for kind, df, col in (("train", trains, "TrainID"), ("airplane", flights, "FlightID")):
            for _, r in df.iterrows():
                idx[kind].setdefault(str(r[col]), r)  # first row (evaluator iterates in this order)
        _INTERCITY_CACHE[key] = idx
    return _INTERCITY_CACHE[key]


def _norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip().casefold()


def _resolve(index, name, lenient, notes, what):
    exact, norm = index
    if name in exact:
        return name, exact[name]
    if lenient and isinstance(name, str):
        cands = norm.get(_norm(name), [])
        if len(cands) >= 1:
            notes.append("canonicalised %s name %r -> %r" % (what, name, cands[0]))
            return cands[0], exact[cands[0]]
    return None, None


# ------------------------------------------------------------ time utils
def _fix_time(t, lenient, notes, ctx):
    """Return (string to emit, minutes or None)."""
    if not isinstance(t, str):
        notes.append("%s: missing/non-string time %r" % (ctx, t))
        return ("" if t is None else str(t)), None
    m = _TIME_RE.match(t)
    if not m:
        notes.append("%s: unparsable time %r" % (ctx, t))
        return t, None
    h, mi = int(m.group(1)), int(m.group(2))
    s = "%02d:%02d" % (h, mi)
    if s != t:
        if lenient:
            notes.append("%s: time %r -> %r" % (ctx, t, s))
        else:
            return t, h * 60 + mi
    return s, h * 60 + mi


def _hhmm(minutes):
    minutes = max(0, int(minutes))
    return "%02d:%02d" % (minutes // 60, minutes % 60)


def _to_min(s):
    try:
        h, m = str(s).split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _r2(x):
    return round(float(x) + 0.0, 2)


# ------------------------------------------------------------ transports
def _goto(city, start, end, start_time, mode):
    return ct_common.tools()["transportation"].goto(city, start, end, start_time, mode)


def _build_transports(city, start, end, depart, mode, people, notes, ctx):
    """Inner-city legs from start to end departing at `depart` (HH:MM)."""
    if start == end:
        return []
    if not start or not end or depart is None:
        notes.append("%s: cannot compute transport (start=%r end=%r depart=%r)" % (ctx, start, end, depart))
        return []
    try:
        legs = _goto(city, start, end, depart, mode)
        if not isinstance(legs, list) and mode == "metro":
            notes.append("%s: metro has no solution %r -> %r (same nearest station); fell back to walk" % (ctx, start, end))
            mode = "walk"
            legs = _goto(city, start, end, depart, mode)
        if not isinstance(legs, list):
            notes.append("%s: goto returned %r" % (ctx, legs))
            return []
    except Exception as e:  # unknown POI etc.
        notes.append("%s: goto failed %r -> %r (%s: %s)" % (ctx, start, end, type(e).__name__, e))
        return []
    out = []
    for leg in legs:
        leg = {k: py_num(v) if k in ("cost", "distance") else v for k, v in leg.items()}
        out.append(leg)
    if len(out) == 3:  # metro: walk / metro / walk
        for i in (0, 2):
            out[i]["price"] = out[i]["cost"]
        out[1]["price"] = out[1]["cost"]
        out[1]["tickets"] = int(people)
        out[1]["cost"] = _r2(out[1]["price"] * people)
    elif len(out) == 1 and out[0]["mode"] == "taxi":
        out[0]["price"] = out[0]["cost"]
        out[0]["cars"] = int((people - 1) // 4 + 1)
        out[0]["cost"] = _r2(out[0]["price"] * out[0]["cars"])
    else:
        for leg in out:
            leg["price"] = leg["cost"]
    return out


def _travel_minutes(city, start, end, mode):
    """Duration of the goto route (time-invariant); (minutes, mode_used) or (None, mode)."""
    try:
        legs = _goto(city, start, end, "00:00", mode)
        if not isinstance(legs, list) and mode == "metro":
            mode = "walk"
            legs = _goto(city, start, end, "00:00", mode)
        if not isinstance(legs, list) or not legs:
            return None, mode
        return _to_min(legs[-1]["end_time"]), mode
    except Exception:
        return None, mode


# ------------------------------------------------------------ main
def expand_with_notes(query, high_level_plan, lenient=True):
    notes = []
    people = int(query["people_number"])
    start_city, target_city = query["start_city"], query["target_city"]
    plan = {"people_number": people, "start_city": start_city, "target_city": target_city, "itinerary": []}
    try:
        days = _get_days(high_level_plan, notes)
        cidx = _city(target_city)
        seen_intercity = False
        prev_loc = None          # location where the previous activity ends
        prev_end = None          # minutes
        prev_end_str = None
        prev_day = None
        for d_i, day in enumerate(days):
            acts_out = []
            acts = day.get("activities", []) if isinstance(day, dict) else []
            if not isinstance(acts, list):
                notes.append("day %d: activities is not a list" % (d_i + 1))
                acts = []
            for a_i, act in enumerate(acts):
                ctx = "day %d act %d" % (d_i + 1, a_i + 1)
                if not isinstance(act, dict):
                    notes.append("%s: activity is not an object, skipped" % ctx)
                    continue
                out, here_start, here_end = _expand_activity(
                    act, ctx, cidx, query, people, seen_intercity, lenient, notes)
                if out.get("type") in INTERCITY_TYPES:
                    seen_intercity = True
                # inbound transport
                mode = act.get("transport_mode")
                if isinstance(mode, str):
                    mode = mode.strip().lower() if lenient else mode
                is_first = prev_loc is None
                transports = []
                if not is_first and here_start != prev_loc:
                    if mode not in MODES:
                        notes.append("%s: transport_mode %r invalid/missing -> %s" % (ctx, act.get("transport_mode"), DEFAULT_MODE))
                        mode = DEFAULT_MODE
                    if prev_day == d_i and prev_end is not None:
                        depart = prev_end_str
                    else:  # first activity of a new day: leave just in time
                        st = _to_min(out.get("start_time"))
                        dur, _ = _travel_minutes(target_city, prev_loc, here_start, mode)
                        if st is None or dur is None:
                            depart = prev_end_str if prev_end is not None and prev_day == d_i else "00:00"
                        else:
                            depart = _hhmm(max(0, st - dur))
                    transports = _build_transports(target_city, prev_loc, here_start, depart, mode, people, notes, ctx)
                out["transports"] = transports
                acts_out.append(out)
                prev_loc = here_end
                prev_end_str = out.get("end_time")
                prev_end = _to_min(prev_end_str)
                prev_day = d_i
            plan["itinerary"].append({"day": d_i + 1, "activities": acts_out})
    except Exception as e:  # never crash
        notes.append("expand internal error %s: %s" % (type(e).__name__, e))
    return plan, notes


def expand(query, high_level_plan, lenient=True):
    return expand_with_notes(query, high_level_plan, lenient=lenient)[0]


def _get_days(hl, notes):
    if isinstance(hl, str):
        import json
        try:
            hl = json.loads(hl)
        except Exception:
            notes.append("high-level plan is not valid JSON")
            return []
    if isinstance(hl, dict):
        days = hl.get("itinerary", [])
    elif isinstance(hl, list):
        days = hl
    else:
        days = []
    if not isinstance(days, list):
        notes.append("itinerary is not a list")
        return []
    return days


def _expand_activity(act, ctx, cidx, query, people, seen_intercity, lenient, notes):
    """Return (official activity without transports, location_at_start, location_at_end)."""
    raw_type = act.get("type")
    typ = raw_type.strip().lower() if (lenient and isinstance(raw_type, str)) else raw_type
    if lenient and typ in TYPE_SYNONYMS:
        notes.append("%s: type %r -> %r" % (ctx, raw_type, TYPE_SYNONYMS[typ]))
        typ = TYPE_SYNONYMS[typ]
    name = act.get("name", act.get("position", act.get("id")))
    if isinstance(name, str) and lenient:
        name = name.strip()

    if typ in INTERCITY_TYPES:
        return _expand_intercity(act, ctx, typ, name, query, people, seen_intercity, lenient, notes)

    st, _ = _fix_time(act.get("start_time"), lenient, notes, ctx)
    et, _ = _fix_time(act.get("end_time"), lenient, notes, ctx)
    out = {"type": typ, "position": name if isinstance(name, str) else str(name),
           "start_time": st, "end_time": et}
    if typ == "attraction":
        canon, row = _resolve(cidx.attr, name, lenient, notes, "attraction")
        if row is None:
            notes.append("%s: unknown attraction %r" % (ctx, name))
            out.update({"price": 0, "cost": 0, "tickets": people})
        else:
            price = py_num(row["price"])
            out.update({"position": canon, "price": price, "tickets": people, "cost": _r2(price * people)})
    elif typ in MEAL_TYPES:
        canon, row = _resolve(cidx.rest, name, lenient, notes, "restaurant")
        if row is not None:
            price = py_num(row["price"])
            out.update({"position": canon, "price": price, "cost": _r2(price * people)})
        else:
            hcanon, hrow = _resolve(cidx.hotel, name, lenient, notes, "hotel")
            if hrow is not None:
                out.update({"position": hcanon, "price": 0, "cost": 0})
                if typ != "breakfast":
                    notes.append("%s: %s at hotel %r (only breakfast may be at a hotel)" % (ctx, typ, hcanon))
            else:
                notes.append("%s: unknown restaurant %r" % (ctx, name))
                out.update({"price": 0, "cost": 0})
    elif typ == "accommodation":
        canon, row = _resolve(cidx.hotel, name, lenient, notes, "hotel")
        if row is None:
            notes.append("%s: unknown hotel %r" % (ctx, name))
            out.update({"price": 0, "cost": 0})
        else:
            price, beds = py_num(row["price"]), int(row["numbed"])
            rooms = int(math.ceil(people / beds))
            out.update({"position": canon, "price": price, "room_type": beds, "rooms": rooms,
                        "cost": _r2(price * rooms)})
    else:
        notes.append("%s: unknown activity type %r" % (ctx, raw_type))
        out.update({"price": 0, "cost": 0})
    return out, out["position"], out["position"]


def _expand_intercity(act, ctx, typ, name, query, people, seen_intercity, lenient, notes):
    start_city, target_city = query["start_city"], query["target_city"]
    direction = (target_city, start_city) if seen_intercity else (start_city, target_city)
    other_dir = (direction[1], direction[0])
    other_typ = "airplane" if typ == "train" else "train"
    ids = [str(name)] if name is not None else []
    if lenient and isinstance(name, str) and "/" in name:
        ids = [x.strip() for x in name.split("/") if x.strip()]
    row, found_typ, found_dir = None, typ, direction
    for cand_typ, cand_dir in ((typ, direction), (other_typ, direction), (typ, other_dir), (other_typ, other_dir)):
        idx = _intercity_index(*cand_dir)[cand_typ]
        hit = next((i for i in ids if i in idx), None)
        if hit is not None:
            row, found_typ, found_dir = idx[hit], cand_typ, cand_dir
            if len(ids) > 1:
                notes.append("%s: merged ID %r -> %r" % (ctx, name, hit))
            break
    if row is None:
        notes.append("%s: unknown %s ID %r for %s->%s" % (ctx, typ, name, *direction))
        st, _ = _fix_time(act.get("start_time"), lenient, notes, ctx)
        et, _ = _fix_time(act.get("end_time"), lenient, notes, ctx)
        out = {"type": typ, "start": "", "end": "", "start_time": st, "end_time": et,
               "price": 0, "cost": 0, "tickets": people}
        out["FlightID" if typ == "airplane" else "TrainID"] = "" if name is None else str(name)
        return out, "", ""
    if found_typ != typ:
        if lenient:
            notes.append("%s: ID %r is a %s, declared %s -> using %s" % (ctx, name, found_typ, typ, found_typ))
            typ = found_typ
        else:
            notes.append("%s: ID %r is a %s but declared %s" % (ctx, name, found_typ, typ))
    if found_dir != direction:
        notes.append("%s: ID %r belongs to %s->%s, expected %s->%s" % (ctx, name, *found_dir, *direction))
    price = py_num(row["Cost"])
    out = {"type": typ, "start": str(row["From"]), "end": str(row["To"]),
           "start_time": str(row["BeginTime"]), "end_time": str(row["EndTime"]),
           "price": price, "tickets": people, "cost": _r2(price * people) if price == price else price}
    id_val = str(row["FlightID"] if found_typ == "airplane" else row["TrainID"])
    out["FlightID" if typ == "airplane" else "TrainID"] = id_val
    return out, out["start"], out["end"]


# ------------------------------------------------------------ inverse
def to_high_level(official_plan):
    """Official plan -> high-level plan (used for round-trip validation).
    transport_mode is the primary mode of the activity's inbound transports
    (metro if 3 legs), omitted when there is no transport."""
    hl = {"itinerary": []}
    for day in official_plan.get("itinerary", []):
        acts = []
        for a in day.get("activities", []):
            t = a.get("type")
            h = {"type": t,
                 "name": a.get("TrainID") or a.get("FlightID") if t in INTERCITY_TYPES else a.get("position"),
                 "start_time": a.get("start_time"), "end_time": a.get("end_time")}
            tr = a.get("transports") or []
            if len(tr) == 3:
                h["transport_mode"] = tr[1].get("mode")
            elif len(tr) == 1:
                h["transport_mode"] = tr[0].get("mode")
            acts.append(h)
        hl["itinerary"].append({"day": day.get("day"), "activities": acts})
    return hl
