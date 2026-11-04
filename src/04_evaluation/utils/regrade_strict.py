"""Regrade recorded harness v3 market runs with the original and the strict grader (strict_grade.py;
research/log/2026-11-01-0010.md, 11-02 22:3x UTC), per run and as paired comparisons of two runs.

A run directory holds run.py's results.jsonl, whose rows carry "task_id", "run", "end", "passed", "grade" and
"answer" (the finish text, null without one). Rows of tasks outside <tasks.jsonl> are left out; per (task, run) the
last row counts, and one that ended on a server or harness error (ERR) is dropped, as the evaluation chain's
paired_runs.py and hm5.py do; the rest are the valid rows. Each valid row is graded again with market.grade_keys,
which must give the recorded "passed" (the mismatches are listed, and the run's verdicts are suspect when there are
any), and with strict_grade.grade_task_strict. Per run: rows, valid rows, passes under both graders, the rows that
pass only the original (with their failed rules and answers) and those that pass only the strict one (0 by
construction), the same per template, and with several runs per task pass@1 and pass@k as hm5.py --passk computes
them (over the tasks with k valid runs).

A comparison B - A under either grader scores a task as its share of passed valid runs and pairs the tasks scored in
both, as paired_runs.py does; the mean difference is resampled over tasks (10,000 resamples, seed 20261025, a fresh
generator for each comparison, so that the all-task numbers equal paired_runs.py's and hm5.py's), with the 95%
interval from the 2.5th and 97.5th percentiles and p_one_sided the share of resampled differences <= 0. --exclude
adds the comparison without the given tasks; --subset adds one over the given templates.

Usage: python -B regrade_strict.py <runs dir> <tasks.jsonl> <run> [<run> ...] [--pair A:B ...] [--exclude id[,id]]
       [--subset t1,t2] [--out <dir>]
       (writes <out>/<run>.json for each run and <out>/compare_<tasks file stem>.json; prints one line per run and
       per comparison)
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "code"))
import strict_grade  # noqa: E402

ERR = ("api_error", "budget", "harness_error")
GRADERS = ("original", "strict")


def valid_rows(path, tasks):
    """(rows read, {(task_id, run): row} of the valid rows, ends of the dropped rows, unreadable lines)."""
    last, n, bad = {}, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            n += 1
            if r["task_id"] in tasks:
                last[(r["task_id"], r["run"])] = r
    dropped = collections.Counter(r.get("end") for r in last.values() if r.get("end") in ERR)
    return n, {k: r for k, r in last.items() if r.get("end") not in ERR}, dropped, bad


def regrade(runs_dir, name, tasks):
    n, rows, dropped, bad = valid_rows(os.path.join(runs_dir, name, "results.jsonl"), tasks)
    verdicts, mismatch, only_orig, only_strict = {}, [], [], []
    tpl = collections.defaultdict(lambda: collections.Counter())
    rules = collections.Counter()
    for (tid, run), r in sorted(rows.items()):
        t = tasks[tid]
        orig = strict_grade.market.grade_keys(t["keys"], r.get("answer"))["passed"]
        g = strict_grade.grade_task_strict(t, r.get("answer"))
        verdicts[(tid, run)] = {"original": orig, "strict": g["passed"]}
        if orig != bool(r["passed"]):
            mismatch.append({"task_id": tid, "run": run, "recorded": r["passed"], "recomputed": orig})
        c = tpl[t.get("template", "-")]
        c["valid"] += 1
        c["original"] += orig
        c["strict"] += g["passed"]
        if orig and not g["passed"]:
            c["pass_original_fail_strict"] += 1
            rules.update(g["failed_rules"])
            only_orig.append({"task_id": tid, "run": run, "template": t.get("template"),
                              "failed_rules": g["failed_rules"], "keys": t["keys"], "answer": r.get("answer")})
        elif g["passed"] and not orig:
            only_strict.append({"task_id": tid, "run": run})
    out = {"run": name, "rows": n, "unreadable_lines": bad, "valid_rows": len(rows), "dropped_by_end": dict(dropped),
           "tasks_with_valid_rows": len({tid for tid, _ in rows}),
           "passed": {gr: sum(v[gr] for v in verdicts.values()) for gr in GRADERS},
           "recorded_passed": sum(bool(r["passed"]) for r in rows.values()), "recorded_mismatches": mismatch,
           "pass_original_fail_strict": len(only_orig), "pass_strict_fail_original": len(only_strict),
           "failed_rules_of_flips": dict(rules),
           "by_template": {k: dict(v) for k, v in sorted(tpl.items())}}
    k = max((run for _, run in rows), default=1)
    if k > 1:
        per = collections.defaultdict(list)
        for (tid, _), v in verdicts.items():
            per[tid].append(v)
        full = {tid: v for tid, v in per.items() if len(v) >= k}
        out["passk"] = {"k": k, "tasks": len(full), "tasks_short_of_k": len(tasks) - len(full)}
        for gr in GRADERS:
            p1 = np.mean([sum(x[gr] for x in v) / len(v) for v in full.values()])
            pk = np.mean([any(x[gr] for x in v) for v in full.values()])
            out["passk"][gr] = {"pass@1": round(100 * float(p1), 1), f"pass@{k}": round(100 * float(pk), 1)}
    out["flips"] = only_orig + only_strict
    return out, verdicts


def scores(verdicts, grader):
    ok, n = collections.Counter(), collections.Counter()
    for (tid, _), v in verdicts.items():
        n[tid] += 1
        ok[tid] += v[grader]
    return {tid: ok[tid] / n[tid] for tid in n}


def compare(A, B, ids):
    if not ids:
        return None
    x, y = np.array([A[i] for i in ids]), np.array([B[i] for i in ids])
    d = y - x
    boot = d[np.random.default_rng(20261025).integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    return {"n": len(ids), "A": round(100 * x.mean(), 1), "B": round(100 * y.mean(), 1),
            "diff": round(100 * d.mean(), 1),
            "ci95": [round(100 * np.percentile(boot, 2.5), 1), round(100 * np.percentile(boot, 97.5), 1)],
            "p_one_sided": float((boot <= 0).mean())}


def pair(va, vb, tasks, exclude, subset):
    out = {}
    for gr in GRADERS:
        A, B = scores(va, gr), scores(vb, gr)
        common = sorted(set(A) & set(B))
        res = {"tasks_in_both": len(common), "missing_in_one_arm": len(set(A) ^ set(B)), "all": compare(A, B, common)}
        if exclude:
            res["without_excluded"] = compare(A, B, [i for i in common if i not in exclude])
        if subset:
            res["subset"] = compare(A, B, [i for i in common if tasks[i].get("template") in subset])
        res["by_template"] = {}
        for t in sorted({tasks[i].get("template", "-") for i in common}):
            ids = [i for i in common if tasks[i].get("template", "-") == t]
            res["by_template"][t] = {"n": len(ids), "A": round(100 * np.mean([A[i] for i in ids]), 1),
                                     "B": round(100 * np.mean([B[i] for i in ids]), 1)}
        out[gr] = res
    return out


def fmt(r):
    lo, hi = r["ci95"]
    return f"{r['A']} -> {r['B']}, {r['diff']:+.1f} [{lo}, {hi}], p {r['p_one_sided']:.4f}, n {r['n']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("tasks")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pair", action="append", default=[], help="A:B, compared as B - A; both must be among the runs")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--subset", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with open(a.tasks, encoding="utf-8") as f:
        tasks = {t["task_id"]: t for t in map(json.loads, f)}
    exclude, subset = set(filter(None, a.exclude.split(","))), set(filter(None, a.subset.split(",")))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    verdicts = {}
    for name in a.runs:
        if not os.path.exists(os.path.join(a.runs_dir, name, "results.jsonl")):
            print(json.dumps({"run": name, "missing": True}))
            continue
        out, verdicts[name] = regrade(a.runs_dir, name, tasks)
        out["tasks_file"] = a.tasks
        if a.out:
            with open(os.path.join(a.out, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=1)
        print(f"{name}: {out['valid_rows']}/{out['rows']} valid rows, passed original {out['passed']['original']} "
              f"(recorded {out['recorded_passed']}, {len(out['recorded_mismatches'])} mismatches), strict "
              f"{out['passed']['strict']}; flips {out['pass_original_fail_strict']} original-only, "
              f"{out['pass_strict_fail_original']} strict-only {out['failed_rules_of_flips']}"
              + (f"; passk {out['passk']}" if "passk" in out else ""))
    comps = {}
    for p in a.pair:
        an, bn = p.split(":")
        if an not in verdicts or bn not in verdicts:
            comps[f"{bn} - {an}"] = {"missing": [n for n in (an, bn) if n not in verdicts]}
            print(json.dumps({"comparison": f"{bn} - {an}", **comps[f"{bn} - {an}"]}))
            continue
        c = comps[f"{bn} - {an}"] = pair(verdicts[an], verdicts[bn], tasks, exclude, subset)
        for gr in GRADERS:
            line = f"{bn} - {an} [{gr}]: all {fmt(c[gr]['all'])}"
            if exclude:
                line += f"; without excluded {fmt(c[gr]['without_excluded'])}"
            if subset and c[gr]["subset"]:
                line += f"; subset {fmt(c[gr]['subset'])}"
            print(line)
    if a.out and comps:
        stem = os.path.splitext(os.path.basename(a.tasks))[0]
        with open(os.path.join(a.out, f"compare_{stem}.json"), "w", encoding="utf-8") as f:
            json.dump({"tasks_file": a.tasks, "exclude": sorted(exclude), "subset": sorted(subset),
                       "comparisons": comps}, f, indent=1)


if __name__ == "__main__":
    main()
