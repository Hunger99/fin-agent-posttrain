"""Paired comparison of two harness v3 run directories with several runs per task (research/log/2026-11-01-0010.md,
11-02 00:1x UTC rule): per task the last row of each (task, run), rows that ended on a server or harness error dropped;
a task's score is its share of passed valid runs; tasks scored in both arms are paired; the difference B - A in the mean
score is resampled over tasks (10,000 resamples, seed 20261025), 95% interval from the 2.5th and 97.5th percentiles,
p_one_sided the share of resampled differences <= 0. Prints the comparison over all tasks, over --subset templates, and
per template.
Usage: python3 -B paired_runs.py <runs dir> <tasks.jsonl> <A> <B> [--subset t1,t2,...]"""
import argparse
import collections
import json
import os

import numpy as np

ERR = ("api_error", "budget", "harness_error")
ap = argparse.ArgumentParser()
ap.add_argument("runs")
ap.add_argument("tasks")
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--subset", default="")
a = ap.parse_args()
tpl = {}
for line in open(a.tasks):
    t = json.loads(line)
    tpl[t["task_id"]] = t["template"]


def scores(name):
    last = {}
    for line in open(os.path.join(a.runs, name, "results.jsonl")):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r["task_id"] in tpl:
            last[(r["task_id"], r["run"])] = r
    ok, n = collections.Counter(), collections.Counter()
    for (tid, _), r in last.items():
        if r.get("end") in ERR:
            continue
        n[tid] += 1
        ok[tid] += bool(r["passed"])
    return {tid: ok[tid] / n[tid] for tid in n}


A, B = scores(a.a), scores(a.b)
common = sorted(set(A) & set(B))
rng = np.random.default_rng(20261025)


def compare(ids):
    if not ids:
        return None
    x, y = np.array([A[i] for i in ids]), np.array([B[i] for i in ids])
    d = y - x
    boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    return {"n": len(ids), "A": round(100 * x.mean(), 1), "B": round(100 * y.mean(), 1), "diff": round(100 * d.mean(), 1),
            "ci95": [round(100 * np.percentile(boot, 2.5), 1), round(100 * np.percentile(boot, 97.5), 1)],
            "p_one_sided": float((boot <= 0).mean())}


subset = set(s for s in a.subset.split(",") if s)
print(json.dumps({"comparison": f"{a.a} -> {a.b}", "tasks_in_both": len(common), "missing_in_one_arm": len(set(A) ^ set(B))}))
print(json.dumps({"all": compare(common)}))
if subset:
    print(json.dumps({"subset": sorted(subset), "result": compare([i for i in common if tpl[i] in subset])}))
for t in sorted({tpl[i] for i in common}):
    print(json.dumps({"template": t, "result": compare([i for i in common if tpl[i] == t])}))
