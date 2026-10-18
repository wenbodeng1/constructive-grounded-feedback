"""Official ChinaTravel evaluation (lang='en') of official-format plans.

evaluate(plans, uids=None, query_data=None, quiet=True, tpc=True) -> dict

plans: {uid: official plan dict}.  A missing uid (or a non-dict plan) is
scored as {} -- exactly like eval_exp.py / eval_tpc.py do for a missing file.
uids default to all 100 competition_test queries (ct_data.load_queries()).

Uses, unchanged:
  chinatravel.evaluation.schema_constraint.evaluate_schema_constraints
  chinatravel.evaluation.commonsense_constraint.evaluate_commonsense_constraints(lang='en')
  chinatravel.evaluation.hard_constraint.evaluate_hard_constraints_v2(lang='en')
and, if tpc=True, eval_tpc.cal_default_pr_score + the TPC overall-score formula.

Returned dict (all rates in percent, as the official functions return them):
  n, schema_pass_rate,
  commonsense_micro, commonsense_macro,          (Mic.EPR / Mac.EPR)
  hard_micro, hard_macro,                        (logical, python DSL v2)
  hard_conditional_micro, hard_conditional_macro (C-LPR: only commonsense-passing plans count)
  final_pass_rate                                (schema & commonsense & hard; FPR)
  [tpc_DAV, tpc_ATT, tpc_DDR, tpc_overall]       (if tpc=True)
  commonsense_failure_counts {evaluator column: #plans failing it}
  hard_failure_counts_by_key {constraint_key: #failed}, hard_total_by_key {constraint_key: #instances}
  per_uid {uid: {schema, commonsense, hard, final (bools), commonsense_failures [cols],
                 hard_failures [{index, key, nl}], hard_passed, hard_total}}
"""

import contextlib
import io
import json
import os

import ct_common  # noqa: F401  (sys.path + zh-database redirect)

SCHEMA_PATH = os.path.join(ct_common.CT_REPO, "chinatravel", "evaluation", "output_schema.json")


def _load_official():
    from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
    from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
    from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
    return evaluate_schema_constraints, evaluate_commonsense_constraints, evaluate_hard_constraints_v2


def evaluate(plans, uids=None, query_data=None, quiet=True, tpc=True):
    if query_data is None or uids is None:
        import ct_data
        all_uids, all_data = ct_data.load_queries()
        uids = uids if uids is not None else all_uids
        query_data = query_data if query_data is not None else all_data
    uids = list(uids)
    plan_dict = {u: (plans.get(u) if isinstance(plans.get(u), dict) else {}) for u in uids}
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    ev_schema, ev_comm, ev_hard = _load_official()
    sink_out, sink_err = io.StringIO(), io.StringIO()
    ctx_out = contextlib.redirect_stdout(sink_out) if quiet else contextlib.nullcontext()
    ctx_err = contextlib.redirect_stderr(sink_err) if quiet else contextlib.nullcontext()
    with ctx_out, ctx_err:
        schema_rate, schema_agg, schema_pass = ev_schema(uids, plan_dict, schema=schema)
        c_macro, c_micro, comm_agg, comm_pass = ev_comm(uids, query_data, plan_dict, verbose=False, lang="en")
        h_macro, h_micro, hc_macro, hc_micro, hard_agg, hard_pass = ev_hard(
            uids, query_data, plan_dict, env_pass_id=comm_pass, verbose=False, lang="en")
        final_pass = set(schema_pass) & set(comm_pass) & set(hard_pass)
        res = {
            "n": len(uids),
            "schema_pass_rate": float(schema_rate),
            "commonsense_micro": float(c_micro),
            "commonsense_macro": float(c_macro),
            "hard_micro": float(h_micro),
            "hard_macro": float(h_macro),
            "hard_conditional_micro": float(hc_micro),
            "hard_conditional_macro": float(hc_macro),
            "final_pass_rate": 100.0 * len(final_pass) / len(uids),
        }
        if tpc:
            res.update(_tpc_scores(uids, query_data, plan_dict, final_pass, res))
    if quiet:
        res["evaluator_stderr_lines"] = [l for l in sink_err.getvalue().splitlines()
                                         if "evaluation failed" in l][:50]

    # ---- per-constraint failure counts and per-uid flags
    comm_cols = [c for c in comm_agg.columns if c != "data_id"]
    comm_counts = {c: int(comm_agg[c].fillna(0).astype(float).gt(0).sum()) for c in comm_cols}
    res["commonsense_failure_counts"] = dict(sorted(comm_counts.items(), key=lambda kv: -kv[1]))
    s_pass, c_pass, h_pass = set(schema_pass), set(comm_pass), set(hard_pass)
    per_uid, key_fail, key_total = {}, {}, {}
    hard_rows = {row["data_id"]: row for _, row in hard_agg.iterrows()}
    for ii, u in enumerate(uids):
        crow = comm_agg.iloc[ii]
        c_fail = [c for c in comm_cols if _as_int(crow[c]) > 0]
        q = query_data[u]
        h_fail = []
        hrow = hard_rows.get(u)
        n_logic = len(q["hard_logic_py"])
        keys = q.get("constraint_keys") or ["logic_%d" % i for i in range(n_logic)]
        for i in range(n_logic):
            ok = hrow is not None and _as_int(hrow.get("logic_py_%d" % i, 0)) == 1
            key_total[keys[i]] = key_total.get(keys[i], 0) + 1
            if not ok:
                key_fail[keys[i]] = key_fail.get(keys[i], 0) + 1
                h_fail.append({"index": i, "key": keys[i], "nl": q["hard_logic_nl"][i] if "hard_logic_nl" in q else None})
        per_uid[u] = {"schema": u in s_pass, "commonsense": u in c_pass, "hard": u in h_pass,
                      "final": u in final_pass, "commonsense_failures": c_fail,
                      "hard_failures": h_fail, "hard_passed": n_logic - len(h_fail), "hard_total": n_logic}
    res["hard_failure_counts_by_key"] = dict(sorted(key_fail.items(), key=lambda kv: -kv[1]))
    res["hard_total_by_key"] = dict(sorted(key_total.items()))
    res["per_uid"] = per_uid
    return res


def _as_int(v):
    try:
        f = float(v)
        return int(f) if f == f else 0
    except Exception:
        return 0


def _tpc_scores(uids, query_data, plan_dict, final_pass, res):
    """Same formula as eval_tpc.py (TPC@AIC2025 overall score)."""
    try:
        import eval_tpc  # repo root is on sys.path
        pr = eval_tpc.cal_default_pr_score(uids, query_data, plan_dict, list(final_pass))
        dav, att, ddr = float(pr[0]) * 100, float(pr[1]) * 100, float(pr[2]) * 100
    except Exception as e:  # pragma: no cover
        return {"tpc_error": "%s: %s" % (type(e).__name__, e)}
    overall = (0.1 * res["commonsense_micro"] + 0.1 * res["commonsense_macro"]
               + 0.25 * res["hard_conditional_micro"] + 0.05 * dav + 0.05 * att + 0.05 * ddr
               + 0.4 * res["final_pass_rate"])
    return {"tpc_DAV": dav, "tpc_ATT": att, "tpc_DDR": ddr, "tpc_overall": overall}


def summary_line(res):
    return ("n=%d schema=%.1f comm_micro=%.1f comm_macro=%.1f hard_micro=%.1f hard_macro=%.1f "
            "C-LPR=%.1f final=%.1f" % (res["n"], res["schema_pass_rate"], res["commonsense_micro"],
                                       res["commonsense_macro"], res["hard_micro"], res["hard_macro"],
                                       res["hard_conditional_micro"], res["final_pass_rate"]))


if __name__ == "__main__":
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Evaluate a directory of <uid>.json official plans")
    ap.add_argument("plan_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    plans = {}
    for f in glob.glob(os.path.join(args.plan_dir, "*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                plans[os.path.basename(f)[:-5]] = json.load(fh)
        except Exception:
            pass
    r = evaluate(plans)
    print(summary_line(r))
    print(json.dumps({k: v for k, v in r.items() if k != "per_uid"}, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(r, fh, indent=1)
