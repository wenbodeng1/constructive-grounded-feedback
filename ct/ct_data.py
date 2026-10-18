"""Load the 100 ChinaTravel Phase-2 competition_test queries (English).

load_queries() -> (uid_list, {uid: symbolic_input})

symbolic_input has exactly the structure the official evaluators expect for
lang='en' (same field set/types as the repo's English query JSON records and
scripts/export_phase2_hf.py PUBLIC_FIELDS):

    uid (str), tag (str), start_city (str), target_city (str),
    days (int), people_number (int),
    hard_logic_py (list[str])   -- ORACLE ONLY (evaluator input)
    hard_logic_nl (list[str])   -- ORACLE ONLY
    nature_language (str)       -- the ONLY agent-facing request text
    constraint_keys (list[str]) -- ORACLE ONLY (constraint template ids)
    official_phase2_evaluation (bool)

The planner must only ever see `planner_view(query)` (nature_language plus
the trip header fields that are already stated in nature_language).
"""

import csv
import json

import ct_common

ORACLE_FIELDS = ("hard_logic_py", "hard_logic_nl", "constraint_keys")


def _to_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def load_queries(path=None):
    csv.field_size_limit(1 << 30)
    path = path or ct_common.QUERY_CSV
    uids, data = [], {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            q = {
                "uid": row["uid"],
                "tag": row["tag"],
                "start_city": row["start_city"],
                "target_city": row["target_city"],
                "days": int(row["days"]),
                "people_number": int(row["people_number"]),
                "hard_logic_py": json.loads(row["hard_logic_py"]),
                "hard_logic_nl": json.loads(row["hard_logic_nl"]),
                "nature_language": row["nature_language"],
                "constraint_keys": json.loads(row["constraint_keys"]),
                "official_phase2_evaluation": _to_bool(row["official_phase2_evaluation"]),
            }
            assert isinstance(q["hard_logic_py"], list)
            assert len(q["hard_logic_py"]) == len(q["hard_logic_nl"]) == len(q["constraint_keys"])
            if q["uid"] in data:
                raise ValueError("duplicate uid " + q["uid"])
            uids.append(q["uid"])
            data[q["uid"]] = q
    return uids, data


def planner_view(query):
    """Fields that may be shown to the planner (no oracle DSL / NL lists)."""
    return {
        "uid": query["uid"],
        "start_city": query["start_city"],
        "target_city": query["target_city"],
        "days": query["days"],
        "people_number": query["people_number"],
        "nature_language": query["nature_language"],
    }


if __name__ == "__main__":
    uids, data = load_queries()
    print(len(uids), "queries")
    q = data[uids[0]]
    print({k: (v if k not in ORACLE_FIELDS else "<%d items>" % len(v)) for k, v in q.items()})
