"""Whether M1's gain over the official Qwen3.5-4B survives the reworded market test set (research/log/2026-11-01-0010.md,
11-02 23:1x UTC): the per-task difference in differences (M1 - official on the reworded set) - (M1 - official on the
original set), with the scores and the bootstrap of paired_runs.py: per task the last row of each (task, run), rows that
ended on a server or harness error dropped, a task's score its share of passed valid runs, only tasks scored in all four
arms; the mean resampled over tasks (10,000 resamples, seed 20261025), 95% interval from the 2.5th and 97.5th
percentiles. share_le_0 is paired_runs.py's p_one_sided; share_ge_0, the share of resampled differences >= 0, is the
one-sided p of "the gain shrinks under rewording". Prints one JSON line over all tasks and one per template.
Usage: python3 -B did_reworded.py <runs dir> <tasks.jsonl> <M1 original> <official original> <M1 reworded>
       <official reworded>"""
import argparse
import collections
import json
import os

import numpy as np

ERR = ("api_error", "budget", "harness_error")
ap = argparse.ArgumentParser()
ap.add_argument("runs")
ap.add_argument("tasks")
ap.add_argument("m1_orig")
ap.add_argument("off_orig")
ap.add_argument("m1_rw")
ap.add_argument("off_rw")
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


S = {k: scores(getattr(a, k)) for k in ("m1_orig", "off_orig", "m1_rw", "off_rw")}
common = sorted(set.intersection(*(set(v) for v in S.values())))
rng = np.random.default_rng(20261025)


def did(ids):
    if not ids:
        return None
    v = {k: np.array([S[k][i] for i in ids]) for k in S}
    g0, g1 = v["m1_orig"] - v["off_orig"], v["m1_rw"] - v["off_rw"]
    d = g1 - g0
    boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    pct = lambda x: round(100 * float(x), 1)
    return {"n": len(ids), **{k: pct(x.mean()) for k, x in v.items()}, "gain_orig": pct(g0.mean()),
            "gain_rw": pct(g1.mean()), "did": pct(d.mean()),
            "ci95": [pct(np.percentile(boot, 2.5)), pct(np.percentile(boot, 97.5))],
            "share_le_0": float((boot <= 0).mean()), "share_ge_0": float((boot >= 0).mean())}


print(json.dumps({"comparison": "did %s - %s -> %s - %s" % (a.m1_orig, a.off_orig, a.m1_rw, a.off_rw),
                  "tasks_in_all": len(common), "tasks_in_some": len(set.union(*(set(v) for v in S.values()))) - len(common)}))
print(json.dumps({"all": did(common)}))
for t in sorted({tpl[i] for i in common}):
    print(json.dumps({"template": t, "result": did([i for i in common if tpl[i] == t])}))
