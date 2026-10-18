"""Shared bootstrap for the tool-free ChinaTravel ("sole-planning") pipeline.

* Puts the ChinaTravel repo on sys.path (default third_party/ChinaTravel, override with
  env CT_REPO).
* The official evaluator modules instantiate *Chinese* sandbox tools at import
  time (chinatravel/symbol_verification/commonsense_constraint.py).  Only the
  English sandbox (database_en) is unpacked in the repo copy, so if the
  Chinese `database/` directory is missing we redirect the zh data path to
  database_en *in-process* (nothing on disk is modified).  The zh tool objects
  are never used when lang='en'; every evaluation here runs with lang='en'.
* Provides lazily-constructed, cached English sandbox tool objects
  (the very same classes the official evaluator uses), so expand() and the
  reference builder see exactly the evaluator's data.
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CT_REPO = os.environ.get("CT_REPO", os.path.join(_REPO, "third_party", "ChinaTravel"))
QUERY_CSV = os.environ.get("CT_QUERY_CSV", os.path.join(_REPO, "third_party", "ChinaTravel_phase2", "competition_test.csv"))
LANG = "en"

if CT_REPO not in sys.path:
    sys.path.insert(0, CT_REPO)

import chinatravel.environment.language as _ct_lang  # noqa: E402

_ZH_DB = os.path.join(CT_REPO, "chinatravel", "environment", "database")
if not os.path.isdir(_ZH_DB) and not getattr(_ct_lang, "_ct_zh_redirect", False):
    _orig_rdp = _ct_lang.relative_database_path

    def _relative_database_path_en_only(lang, *parts):
        return _orig_rdp("en", *parts)

    _ct_lang.relative_database_path = _relative_database_path_en_only
    _ct_lang._ct_zh_redirect = True

CITY_NAMES_EN = list(_ct_lang.CITY_NAMES["en"])

_TOOLS = {}


def tools():
    """Return dict of English sandbox tool objects (cached)."""
    if not _TOOLS:
        from chinatravel.environment.tools.accommodations.apis import Accommodations
        from chinatravel.environment.tools.attractions.apis import Attractions
        from chinatravel.environment.tools.restaurants.apis import Restaurants
        from chinatravel.environment.tools.intercity_transport.apis import IntercityTransport
        from chinatravel.environment.tools.transportation.apis import Transportation

        _TOOLS["attractions"] = Attractions(lang=LANG)
        _TOOLS["accommodations"] = Accommodations(lang=LANG)
        _TOOLS["restaurants"] = Restaurants(lang=LANG)
        _TOOLS["intercity"] = IntercityTransport(lang=LANG)
        _TOOLS["transportation"] = Transportation(lang=LANG)
    return _TOOLS


def py_num(x):
    """numpy scalar -> plain python int/float (int if integral)."""
    try:
        f = float(x)
    except Exception:
        return x
    if f != f:  # NaN
        return f
    if f.is_integer():
        return int(f)
    return f


def fmt_num(x):
    """Compact human-readable number (no trailing .0)."""
    v = py_num(x)
    if isinstance(v, float):
        return ("%.2f" % v).rstrip("0").rstrip(".")
    return str(v)


def intercity_tables(start_city, end_city):
    """Return (trains_df, flights_df) for the direction start_city->end_city,
    exactly as the evaluator's IntercityTransport.select returns them
    (sorted by BeginTime).  Either may be empty."""
    import pandas as pd

    it = tools()["intercity"]
    out = []
    for kind in ("train", "airplane"):
        try:
            df = it.select(start_city, end_city, kind)
        except Exception:
            df = None
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame([])
        out.append(df.reset_index(drop=True))
    return out[0], out[1]
