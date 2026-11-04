"""HM3 and the reported comparisons on the 4B test runs, by the plan written in research/log/2026-10-30-0855.md
before any 4B test result: per task the last row, the recorded verdict; tasks whose last row is a server error in
either arm of a comparison are dropped from that comparison and counted. Paired bootstrap as paired_compare.py.
Usage: python -B hm3.py <runs_q4b_test dir> <tasks_test.jsonl> <tasks_ml_test.jsonl> <A>:<B> ...  (A, B like A1_H1)"""
import collections
import json
import os
import sys

import numpy as np

runs, wt, mt = sys.argv[1:4]
tmpl = {t["task_id"]: t["template"] for t in map(json.loads, open(wt))}
tmpl.update({t["task_id"]: t["template"] for t in map(json.loads, open(mt))})


def arm(name):
    res = {}
    for part in ("wealth", "ml"):
        p = f"{runs}/{name}_{part}/results.jsonl"
        if not os.path.exists(p):
            return None
        for r in map(json.loads, open(p)):
            res[r["task_id"]] = None if r["end"] == "api_error" else bool(r["passed"])
    return res


for pair in sys.argv[4:]:
    a, b = pair.split(":")
    A, B = arm(a), arm(b)
    if A is None or B is None:
        print(json.dumps({"comparison": f"{b} - {a}", "missing": [n for n, x in ((a, A), (b, B)) if x is None]}))
        continue
    ids = sorted(set(A) & set(B))
    dropped = [i for i in ids if A[i] is None or B[i] is None]
    ids = [i for i in ids if i not in dropped]
    d = np.array([B[i] for i in ids], float) - np.array([A[i] for i in ids], float)
    boot = d[np.random.default_rng(20261025).integers(0, len(ids), size=(10000, len(ids)))].mean(1)
    by = collections.defaultdict(lambda: [0, 0, 0])
    for i in ids:
        x = by[tmpl.get(i, "?")]
        x[0] += A[i]; x[1] += B[i]; x[2] += 1
    print(json.dumps({"comparison": f"{b} - {a}", "tasks": len(ids), "dropped_server_error": len(dropped),
                      "missing_in_one_arm": len(set(A) ^ set(B)),
                      a: round(100 * np.mean([A[i] for i in ids]), 1), b: round(100 * np.mean([B[i] for i in ids]), 1),
                      "diff": round(100 * d.mean(), 1), "ci95": [round(100 * np.percentile(boot, 2.5), 1), round(100 * np.percentile(boot, 97.5), 1)],
                      "p_one_sided": float((boot <= 0).mean()),
                      "by_template": {k: f"{v[0]}/{v[2]} -> {v[1]}/{v[2]}" for k, v in sorted(by.items())}}))
