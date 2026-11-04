"""Summarize harness runs and compare two configurations, per docs/history/04_evaluation.md and docs/results.md.

A run directory holds src/02_harness/code/legacy/v1/run.py's results.jsonl; when an episode appears more than once (rerun after a
server error), its last row counts. The success rate is the mean over runs of each run's pass rate; pass^k is the
share of tasks passed in all k runs. Also reported: success by task type, mean steps and tokens, tool errors,
rule rejections, verification rejections, episodes ended by a server error (run.py reruns them, so a final report
should have none), replies sampled again after a parse error, and the failure categories (docs/03_harness.md
section 6) with the share that the constraint, verification and correction parts mainly reduce (HE1's second
condition).
With two directories, a paired bootstrap over tasks (each task the mean of its runs, 10,000 resamples) gives the
difference B - A, its 95% interval and the one-sided p value of difference <= 0.
A comma-separated list of directories counts as one configuration: the wealth and ML test tasks of a configuration run
as separate groups, and the success rate is over both.

Usage: python harness_report.py <run dir>[,<run dir>...] [<run dir B>[,...]]
"""
import collections
import json
import os
import sys

import numpy as np

HE1_CATEGORIES = {"argument_error", "rule_violation", "calculation_error", "premature_finish", "missed_subtask", "loop",
                  "step_limit"}


def load(dirs):
    rows = {}
    for d in dirs.split(","):
        for r in map(json.loads, open(os.path.join(d, "results.jsonl"), encoding="utf-8")):
            rows[(r["task_id"], r["run"])] = r
    return list(rows.values())


def summary(rows):
    by_run = collections.defaultdict(list)
    by_task = collections.defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r["passed"])
        by_task[r["task_id"]].append(r["passed"])
    k = max(len(v) for v in by_task.values())
    fails = [r for r in rows if not r["passed"]]
    cats = collections.Counter(r["failure"] for r in fails)
    types = collections.defaultdict(list)
    for r in rows:
        types[r["type"]].append(r["passed"])
    return {"episodes": len(rows), "tasks": len(by_task), "runs": len(by_run),
            "success_rate": round(100 * float(np.mean([np.mean(v) for v in by_run.values()])), 2),
            f"pass^{k}": round(100 * float(np.mean([all(v) for v in by_task.values() if len(v) == k])), 2),
            "by_type": {t: round(100 * float(np.mean(v)), 2) for t, v in sorted(types.items())},
            "mean_steps": round(float(np.mean([r["steps"] for r in rows])), 2),
            "mean_tokens": round(float(np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in rows])), 1),
            "mean_completion_tokens": round(float(np.mean([r["completion_tokens"] for r in rows])), 1),
            "tool_error_rate": round(float(np.mean([r["tool_errors"] / max(1, r["steps"]) for r in rows])), 4),
            "constraint_blocks": sum(r["constraint_blocks"] for r in rows),
            "verification_rejects": sum(r["verification_rejects"] for r in rows),
            "server_errors": sum(r["end"] == "api_error" for r in rows),
            "resamples": sum(r.get("resamples", 0) for r in rows),
            "ends": dict(collections.Counter(r["end"] for r in rows)),
            "failures": dict(cats.most_common()),
            "failures_in_he1_categories_pct": round(100 * sum(cats[c] for c in HE1_CATEGORIES) / max(1, len(fails)), 2)}


def paired(a_rows, b_rows, n=10_000, seed=20261027):
    ta, tb = collections.defaultdict(list), collections.defaultdict(list)
    for r in a_rows:
        ta[r["task_id"]].append(r["passed"])
    for r in b_rows:
        tb[r["task_id"]].append(r["passed"])
    ids = sorted(set(ta) & set(tb))
    d = np.array([np.mean(tb[i]) - np.mean(ta[i]) for i in ids]) * 100
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return {"tasks": len(ids), "diff_b_minus_a": round(float(d.mean()), 2),
            "ci95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)],
            "p_one_sided": float((boots <= 0).mean())}


if __name__ == "__main__":
    a = load(sys.argv[1])
    print(json.dumps({"A": sys.argv[1], **summary(a)}, ensure_ascii=False))
    if len(sys.argv) > 2:
        b = load(sys.argv[2])
        print(json.dumps({"B": sys.argv[2], **summary(b)}, ensure_ascii=False))
        print(json.dumps({"paired": paired(a, b)}))
