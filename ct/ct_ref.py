"""Tool-free reference block for ChinaTravel sole-planning.

    build_reference(query, max_poi_tokens=DEFAULT_MAX_POI_TOKENS,
                    max_tokens=None, max_chars=None) -> str
    build_reference_with_meta(...) -> (str, meta)
    estimate_tokens(text) -> int     (calibrated to the Qwen3 tokenizer)

The block lists, for the query's target city, attractions / restaurants /
hotels of the English sandbox and ALL usable trains and flights
start_city->target_city and target_city->start_city.  Data come from the very
same sandbox tool objects the official evaluator uses, so names, prices and
times are byte-identical to what is checked.

Step 1 -- deterministic, query-independent compaction (no usable entity dropped)
  * planner-relevant columns only:
      attractions : name | opening hours | ticket price/person |
                    recommended visit hours (recommendmintime-recommendmaxtime);
                    attraction type = group header
      restaurants : name | opening hours | price/person; cuisine = group header
      hotels      : name | price per room per night;
                    (hotel feature type, beds per room) = group header
      trains      : ID depart-arrive; (from station, to station, price/person)
                    = group header; trains with identical from/to/times/price
                    (same physical train listed under several numbers) are
                    merged as "K335/K338 01:16-05:02"
      flights     : ID depart-arrive price/person; (from airport, to airport)
                    = group header
    dropped columns: ids, lat/lon, hotelname_en (== name in all 10 cities),
    restaurant recommendedfood, TrainType (= ID letter), Duration (derivable).
  * 20 intercity rows of the whole sandbox have a missing (NaN) price; the
    evaluator compares price with ==, so a plan using them can never pass --
    they are omitted; so is the one row whose station ("Station", Guangzhou->
    Beijing Z112) is unknown to goto.  (No POI has missing fields, and every
    POI name resolves in the goto position table.)
  * POI hours drop ":00" ("8:30-17" = 08:30-17:00); "00:00-24:00" -> "24h";
    a closing time <= opening time (the evaluator then adds 24h) is printed
    as e.g. "17-2(next day)".  Intercity times stay HH:MM.
  * groups appear in order of first appearance in the sandbox table, rows
    keep sandbox order (POIs) / departure-time order (intercity).

Step 2 -- only if the POI sections exceed max_poi_tokens: neutral subsampling.
  Only attraction / restaurant / hotel lists are subsampled (intercity lists
  stay complete, because departure-time constraints depend on them).  Inside
  every group (attraction type / cuisine / hotel feature+beds) rows are ranked
  by sha1(salt|table|name) -- a fixed, query-independent pseudo-random order --
  and the first ceil(f*n) rows (>=1 per group) are kept, PLUS every
  attraction/restaurant/hotel whose exact sandbox name occurs literally in the
  query's nature_language.  f is the largest value on the grid
  1.00, 0.95, ..., 0.05 for which the POI sections fit max_poi_tokens
  (subsets are nested across f), so the subset depends only on the target
  city and the named entities.  The block then says it is a subset.

Budget: max_poi_tokens (default 12000) counts estimate_tokens() of the POI
sections (Qwen3-calibrated; Qwen splits every digit, so this data is ~2.3
chars/token; Llama-3.1 needs ~15% fewer).  Optional whole-block caps:
max_tokens / max_chars.  max_poi_tokens=None gives the complete compact list.
"""

import hashlib
import math
import re

import ct_common
from ct_common import fmt_num

DEFAULT_MAX_POI_TOKENS = 12000
_SALT = "ct-sole-planning-v1"


# ----------------------------------------------------------------- helpers
_ALPHA_RE = re.compile(r"[A-Za-z]+")


def estimate_tokens(text):
    """Qwen3-tokenizer estimate (least-squares fit on 400 reference blocks of
    this format, relative error within +-2%; Llama-3.1 needs ~12-15% fewer)."""
    digits = sum(c.isdigit() for c in text)
    runs = 0
    alpha_chars = 0
    for m in _ALPHA_RE.finditer(text):
        runs += 1
        alpha_chars += len(m.group(0))
    spaces = sum(c.isspace() for c in text)
    other = len(text) - digits - alpha_chars - spaces
    est = 1.161 * digits + 1.3703 * runs - 0.0609 * alpha_chars + 0.1727 * other + 0.9133 * spaces
    return int(round(est))


def _tables(city):
    t = ct_common.tools()
    return (t["attractions"].data[city], t["restaurants"].data[city],
            t["accommodations"].data[city])


def _short_time(hhmm):
    h, m = str(hhmm).split(":")
    h = str(int(h))
    return h if m == "00" else "%s:%s" % (h, m)


def _tmin(hhmm):
    h, m = str(hhmm).split(":")
    return int(h) * 60 + int(m)


def _hours(row, style):
    o, e = str(row["opentime"]), str(row["endtime"])
    if style == "full":
        return "%s-%s" % (o, e)
    try:
        if o == "00:00" and e in ("24:00", "00:00"):
            return "24h"
        if e == "00:00":
            return "%s-24" % _short_time(o)
        if _tmin(e) <= _tmin(o):  # evaluator adds 24h to the closing time
            return "%s-%s(next day)" % (_short_time(o), _short_time(e))
        return "%s-%s" % (_short_time(o), _short_time(e))
    except Exception:
        return "%s-%s" % (o, e)


def _rec(row):
    return "%s-%sh" % (fmt_num(row["recommendmintime"]), fmt_num(row["recommendmaxtime"]))


def _group_rows(rows, keyfunc):
    order, groups = [], {}
    for r in rows:
        k = keyfunc(r)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)
    return [(k, groups[k]) for k in order]


def _rows(df):
    return [r for _, r in df.iterrows()]


def _isnan(x):
    try:
        return x != x
    except Exception:
        return False


# ----------------------------------------------------------- sections
def usable_intercity(start_city, end_city):
    """(trains_rows, flights_rows) usable rows, API order: price not NaN and both
    stations resolvable by the POI table used by goto (one sandbox row,
    Guangzhou->Beijing Z112, has From="Station")."""
    trains, flights = ct_common.intercity_tables(start_city, end_city)
    poi = ct_common.tools()["transportation"].poi_search

    def ok(r):
        return (not _isnan(r["Cost"]) and not isinstance(poi.search(start_city, r["From"]), str)
                and not isinstance(poi.search(end_city, r["To"]), str))
    out = []
    for df in (trains, flights):
        out.append([r for r in _rows(df) if ok(r)] if len(df) else [])
    return out[0], out[1]


def _intercity_lines(start_city, end_city, label, style):
    trains, flights = usable_intercity(start_city, end_city)
    lines = ["## %s: %s -> %s" % (label, start_city, end_city)]
    for kind, rows, idcol in (("Trains", trains, "TrainID"), ("Flights", flights, "FlightID")):
        if len(rows) == 0:
            lines.append("%s: none" % kind)
            continue
        if style == "full":
            lines.append("%s (ID | from | to | depart | arrive | price/person):" % kind)
            for r in rows:
                lines.append("%s | %s | %s | %s | %s | %s" % (
                    r[idcol], r["From"], r["To"], r["BeginTime"], r["EndTime"], fmt_num(r["Cost"])))
            continue
        if kind == "Trains":  # train prices repeat within a station pair -> put price in the header
            lines.append("Trains (ID depart-arrive), grouped by [from -> to, price/person]:")
            for (fr, to, cost), grp in _group_rows(rows, lambda r: (r["From"], r["To"], fmt_num(r["Cost"]))):
                merged = _group_rows(grp, lambda r: (r["BeginTime"], r["EndTime"]))
                items = ["%s %s-%s" % ("/".join(str(x[idcol]) for x in g), bt, et) for (bt, et), g in merged]
                lines.append("[%s -> %s, %s] %s" % (fr, to, cost, "; ".join(items)))
        else:  # flight prices are all distinct
            lines.append("Flights (ID depart-arrive price/person), grouped by [from -> to]:")
            for (fr, to), grp in _group_rows(rows, lambda r: (r["From"], r["To"])):
                items = ["%s %s-%s %s" % (r[idcol], r["BeginTime"], r["EndTime"], fmt_num(r["Cost"])) for r in grp]
                lines.append("[%s -> %s] %s" % (fr, to, "; ".join(items)))
    return lines


def _attr_key(r):
    return str(r["type"])


def _rest_key(r):
    return str(r["cuisine"])


def _hotel_key(r):
    return (str(r["featurehoteltype"]), int(r["numbed"]))


def _attraction_lines(rows, style):
    if style == "full":
        out = ["## Attractions (name | type | opening hours | ticket price/person | recommended visit hours)"]
        out += ["%s | %s | %s | %s | %s" % (r["name"], r["type"], _hours(r, style), fmt_num(r["price"]), _rec(r)) for r in rows]
        return out
    out = ["## Attractions, grouped by [type] (name | opening hours | ticket price/person | recommended visit hours)"]
    for g, grp in _group_rows(rows, _attr_key):
        out.append("[%s]" % g)
        out += ["%s | %s | %s | %s" % (r["name"], _hours(r, style), fmt_num(r["price"]), _rec(r)) for r in grp]
    return out


def _restaurant_lines(rows, style):
    if style == "full":
        out = ["## Restaurants (name | cuisine | opening hours | price/person)"]
        out += ["%s | %s | %s | %s" % (r["name"], r["cuisine"], _hours(r, style), fmt_num(r["price"])) for r in rows]
        return out
    out = ["## Restaurants, grouped by [cuisine] (name | opening hours | price/person)"]
    for g, grp in _group_rows(rows, _rest_key):
        out.append("[%s]" % g)
        out += ["%s | %s | %s" % (r["name"], _hours(r, style), fmt_num(r["price"])) for r in grp]
    return out


def _hotel_lines(rows, style):
    if style == "full":
        out = ["## Hotels (name | feature type | price per room per night | beds per room)"]
        out += ["%s | %s | %s | %s" % (r["name"], r["featurehoteltype"], fmt_num(r["price"]), fmt_num(r["numbed"])) for r in rows]
        return out
    out = ["## Hotels, grouped by [feature type; beds per room] (name | price per room per night)"]
    for (feat, beds), grp in _group_rows(rows, _hotel_key):
        out.append("[%s; %d bed%s/room]" % (feat, beds, "" if beds == 1 else "s"))
        out += ["%s | %s" % (r["name"], fmt_num(r["price"])) for r in grp]
    return out


HEADER_COMPACT = (
    "# Reference data for a trip {start} -> {target} (sandbox data; use only entities listed here)\n"
    "Copy names and IDs exactly. Prices in CNY. Intercity times are HH:MM; an arrival earlier "
    "than the departure means arrival after midnight. POI opening hours are open-close in 24h "
    "clock with ':00' omitted (e.g. 8:30-17 = 08:30-17:00); 24h = open all day; "
    "'(next day)' = closes after midnight."
)
HEADER_FULL = (
    "# Reference data for a trip {start} -> {target} (sandbox data; use only entities listed here)\n"
    "Copy names and IDs exactly. Prices in CNY. Times are 24h HH:MM; an arrival earlier than the "
    "departure means arrival after midnight."
)


def _subsample(rows, keyfunc, table, frac, keep_names):
    chosen = set()
    for _, grp in _group_rows(rows, keyfunc):
        ranked = sorted(grp, key=lambda r: hashlib.sha1(
            ("%s|%s|%s" % (_SALT, table, r["name"])).encode("utf-8")).hexdigest())
        k = max(1, math.ceil(frac * len(grp)))
        chosen.update(id(r) for r in ranked[:k])
        chosen.update(id(r) for r in grp if r["name"] in keep_names)
    return [r for r in rows if id(r) in chosen]


def named_entities(query):
    """Sandbox POI names (target city) occurring literally in nature_language."""
    attr, rest, hotel = _tables(query["target_city"])
    nl = query.get("nature_language", "")
    names = set(attr["name"]) | set(rest["name"]) | set(hotel["name"])
    return {n for n in names if isinstance(n, str) and n and n in nl}


def _fits(text, max_tokens, max_chars):
    if max_chars is not None and len(text) > max_chars:
        return False
    if max_tokens is not None and estimate_tokens(text) > max_tokens:
        return False
    return True


def build_reference_with_meta(query, max_poi_tokens=DEFAULT_MAX_POI_TOKENS, max_tokens=None,
                              max_chars=None, style="compact"):
    """max_poi_tokens: budget (estimate_tokens) for the attraction+restaurant+hotel
    sections only -- intercity lists are always complete, so the POI subset
    depends only on the target city (+ entities named in the request).
    max_tokens / max_chars: optional caps on the whole block."""
    start, target = query["start_city"], query["target_city"]
    attr, rest, hotel = _tables(target)
    a_rows, r_rows, h_rows = _rows(attr), _rows(rest), _rows(hotel)
    header = (HEADER_FULL if style == "full" else HEADER_COMPACT).format(start=start, target=target)
    intercity = "\n".join(_intercity_lines(start, target, "Outbound intercity", style)) + "\n\n" + \
        "\n".join(_intercity_lines(target, start, "Return intercity", style))

    def poi_text(a, r, h):
        return "\n\n".join(["\n".join(_attraction_lines(a, style)),
                             "\n".join(_restaurant_lines(r, style)), "\n".join(_hotel_lines(h, style))])

    def render(pois, note=None):
        parts = [header] + ([note] if note else []) + [intercity, pois]
        return "\n\n".join(parts) + "\n"

    meta = {"frac": 1.0, "n_attr": len(a_rows), "n_rest": len(r_rows), "n_hotel": len(h_rows),
            "n_attr_total": len(a_rows), "n_rest_total": len(r_rows), "n_hotel_total": len(h_rows),
            "subsampled": False, "named_in_request": None}
    pois = poi_text(a_rows, r_rows, h_rows)
    text = render(pois)
    if style == "full" or (_fits(pois, max_poi_tokens, None) and _fits(text, max_tokens, max_chars)):
        meta.update({"chars": len(text), "est_tokens": estimate_tokens(text),
                     "est_tokens_intercity": estimate_tokens(intercity), "est_tokens_poi": estimate_tokens(pois)})
        return text, meta

    keep = named_entities(query)
    frac = 1.0
    while frac > 0.05 + 1e-9:
        frac = round(frac - 0.05, 2)
        a = _subsample(a_rows, _attr_key, "attr", frac, keep)
        r = _subsample(r_rows, _rest_key, "rest", frac, keep)
        h = _subsample(h_rows, _hotel_key, "hotel", frac, keep)
        note = ("NOTE: to fit the context, the attraction/restaurant/hotel lists below are a fixed "
                "subset (about %d%% of each category) of the sandbox; every place named in the "
                "request is included. Intercity lists are complete." % round(frac * 100))
        pois = poi_text(a, r, h)
        text = render(pois, note)
        if _fits(pois, max_poi_tokens, None) and _fits(text, max_tokens, max_chars):
            break
    meta.update({"frac": frac, "n_attr": len(a), "n_rest": len(r), "n_hotel": len(h),
                 "subsampled": True, "named_in_request": len(keep),
                 "chars": len(text), "est_tokens": estimate_tokens(text),
                 "est_tokens_intercity": estimate_tokens(intercity), "est_tokens_poi": estimate_tokens(pois)})
    return text, meta


def build_reference(query, max_poi_tokens=DEFAULT_MAX_POI_TOKENS, max_tokens=None, max_chars=None,
                    style="compact"):
    return build_reference_with_meta(query, max_poi_tokens=max_poi_tokens, max_tokens=max_tokens,
                                     max_chars=max_chars, style=style)[0]


def _stats(vals):
    v = sorted(vals)
    n = len(v)
    return "min=%d p25=%d median=%d mean=%.0f p75=%d p90=%d max=%d" % (
        v[0], v[n // 4], v[n // 2], sum(v) / n, v[(3 * n) // 4], v[int(0.9 * n)], v[-1])


if __name__ == "__main__":
    import argparse
    import json
    import os

    import ct_data

    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", default=None, help="write one reference .txt per uid (+ _meta.json)")
    ap.add_argument("--max_poi_tokens", type=int, default=DEFAULT_MAX_POI_TOKENS, help="<=0: none")
    ap.add_argument("--max_tokens", type=int, default=0, help="<=0: none")
    ap.add_argument("--max_chars", type=int, default=0, help="<=0: none")
    ap.add_argument("--show", type=int, default=None)
    ap.add_argument("--baselines", action="store_true")
    args = ap.parse_args()
    mp = args.max_poi_tokens if args.max_poi_tokens > 0 else None
    mt = args.max_tokens if args.max_tokens > 0 else None
    mc = args.max_chars if args.max_chars > 0 else None

    uids, data = ct_data.load_queries()
    if args.baselines:
        full = [len(build_reference(data[u], style="full")) for u in uids]
        comp = [build_reference_with_meta(data[u], max_poi_tokens=None)[1] for u in uids]
        print("full rows, no compaction : chars", _stats(full))
        print("compact, no subsampling  : chars", _stats([m["chars"] for m in comp]),
              "| est_tokens", _stats([m["est_tokens"] for m in comp]))
    metas = []
    for u in uids:
        text, meta = build_reference_with_meta(data[u], max_poi_tokens=mp, max_tokens=mt, max_chars=mc)
        meta["uid"] = u
        metas.append(meta)
        if args.dump_dir:
            os.makedirs(args.dump_dir, exist_ok=True)
            with open(os.path.join(args.dump_dir, u + ".txt"), "w", encoding="utf-8") as f:
                f.write(text)
    print("budget max_poi_tokens=%s max_tokens=%s max_chars=%s:" % (mp, mt, mc))
    print("  chars      ", _stats([m["chars"] for m in metas]))
    print("  est_tokens ", _stats([m["est_tokens"] for m in metas]))
    print("  est_tokens intercity part", _stats([m["est_tokens_intercity"] for m in metas]))
    sub = [m for m in metas if m["subsampled"]]
    print("subsampled queries: %d/%d" % (len(sub), len(metas)))
    if sub:
        print("  frac(%%): %s" % _stats([int(round(100 * m["frac"])) for m in sub]))
        print("  POIs shown as %% of city total (all 100): %s" % _stats(
            [int(100 * (m["n_attr"] + m["n_rest"] + m["n_hotel"]) / (m["n_attr_total"] + m["n_rest_total"] + m["n_hotel_total"])) for m in metas]))
        print("  queries with >=1 named POI forced in: %d; named POIs per query: %s" % (
            sum(1 for m in sub if m["named_in_request"]), _stats([m["named_in_request"] for m in sub])))
    if args.dump_dir:
        with open(os.path.join(args.dump_dir, "_meta.json"), "w") as f:
            json.dump(metas, f, indent=1)
    if args.show is not None:
        print(build_reference(data[uids[args.show]], max_poi_tokens=mp, max_tokens=mt, max_chars=mc))
