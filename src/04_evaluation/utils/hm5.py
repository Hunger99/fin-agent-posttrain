"""HM5 on the new dataset's v3 files (research/log/2026-10-31-0150.md, 23:3x UTC) with hm3.py's rules: per task the
last row and its recorded verdict; a task whose last row ended on a server or harness error (api_error, budget,
harness_error) in either arm is dropped from that comparison and counted; paired bootstrap of the difference in pass
rate, 10,000 resamples, seed 20261025, p_one_sided the share of resampled differences <= 0.
  pairs:   <A>:<B> run directory names under <runs dir>, compared as B - A on the tasks of <tasks.jsonl>;
  --passk: <run dir> with k runs per task: pass@1 as the mean over tasks of (passed runs / valid runs), pass@k as the
           share of tasks with at least one passed run, over the tasks with k valid runs (the others are counted);
  --exclude: task ids left out of every comparison (reported as a second set, not instead of the first).
Usage: python -B hm5.py <runs dir> <tasks.jsonl> [A:B ...] [--passk <run dir> --k 4] [--exclude id[,id]]"""
import argparse
import collections
import json
import os

import numpy as np

ERR = ("api_error", "budget", "harness_error")
ap = argparse.ArgumentParser()
ap.add_argument("runs")
ap.add_argument("tasks")
ap.add_argument("pairs", nargs="*")
ap.add_argument("--passk", default=None)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--exclude", default="")
a = ap.parse_args()
tasks = {t["task_id"]: t.get("template", "-") for t in map(json.loads, open(a.tasks))}
excl = set(filter(None, a.exclude.split(",")))


def rows(name):
    p = f"{a.runs}/{name}/results.jsonl"
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r["task_id"] in tasks:
            d[(r["task_id"], r["run"])] = r
    return d


def arm(name):
    d = rows(name)
    if d is None:
        return None
    last = {}
    for (tid, run), r in sorted(d.items()):
        if run == 1:
            last[tid] = None if r["end"] in ERR else bool(r["passed"])
    return last


def compare(A, B, ids, a_name, b_name):
    dropped = [i for i in ids if A[i] is None or B[i] is None]
    ids = [i for i in ids if i not in dropped]
    d = np.array([B[i] for i in ids], float) - np.array([A[i] for i in ids], float)
    boot = d[np.random.default_rng(20261025).integers(0, len(ids), size=(10000, len(ids)))].mean(1)
    by = collections.defaultdict(lambda: [0, 0, 0])
    for i in ids:
        x = by[tasks[i]]
        x[0] += A[i]; x[1] += B[i]; x[2] += 1
    return {"tasks": len(ids), "dropped_server_error": len(dropped),
            a_name: round(100 * np.mean([A[i] for i in ids]), 1), b_name: round(100 * np.mean([B[i] for i in ids]), 1),
            "diff": round(100 * d.mean(), 1), "ci95": [round(100 * np.percentile(boot, 2.5), 1), round(100 * np.percentile(boot, 97.5), 1)],
            "p_one_sided": float((boot <= 0).mean()),
            "by_template": {k: f"{v[0]}/{v[2]} -> {v[1]}/{v[2]}" for k, v in sorted(by.items())}}


for pair in a.pairs:
    an, bn = pair.split(":")
    A, B = arm(an), arm(bn)
    if A is None or B is None:
        print(json.dumps({"comparison": f"{bn} - {an}", "missing": [n for n, x in ((an, A), (bn, B)) if x is None]}))
        continue
    common = sorted(set(A) & set(B))
    out = {"comparison": f"{bn} - {an}", "missing_in_one_arm": len(set(A) ^ set(B)), "all": compare(A, B, common, an, bn)}
    if excl:
        out["without_excluded"] = compare(A, B, [i for i in common if i not in excl], an, bn)
    print(json.dumps(out))

if a.passk:
    d = rows(a.passk)
    per = collections.defaultdict(list)
    for (tid, run), r in d.items():
        if r["end"] not in ERR:
            per[tid].append(bool(r["passed"]))
    full = {t: v for t, v in per.items() if len(v) >= a.k}
    p1 = 100 * np.mean([sum(v) / len(v) for v in full.values()])
    pk = 100 * np.mean([any(v) for v in full.values()])
    print(json.dumps({"passk": a.passk, "k": a.k, "tasks": len(full), "tasks_short_of_k": len(tasks) - len(full),
                      "pass@1": round(p1, 1), f"pass@{a.k}": round(pk, 1), "diff": round(pk - p1, 1)}))
