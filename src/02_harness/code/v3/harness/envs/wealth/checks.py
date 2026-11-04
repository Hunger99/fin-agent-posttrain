"""Checks of generated wealth task files, version 2's env/check_tasks.py and env/verify_composite.py as functions.

check_tasks (research/log/2026-10-27-0307.md): replay each task's reference trades, answer with its key values, and
grade; every task must pass, and a failure means a template, its gold state or the grader is wrong.

verify_composite (research/log/2026-10-30-0855.md, D6), for t5_review_switch files:
1. the reference calls break no rule (core.check) and reach the gold state, and an answer made of the keys (and the
   broken rules, when there are any) passes grade();
2. doing part of it fails: the reference calls without the last one;
3. a naive policy fails whenever the rules change the request: redeem, then subscribe the requested amount as asked.
It prints the counts per case and fails if check 1 or 2 fails on any task.

Usage: python -B envtool.py check-tasks <instance.sqlite> <tasks.jsonl>
       python -B envtool.py verify-composite <instance.sqlite> <tasks.jsonl>
"""
import collections
import json
import re

from harness.envs.wealth import core, grade

FMT = {"money": lambda v: f"${v:,.2f}", "percent": lambda v: f"{v:.2f}%", "units": lambda v: f"{v:,.4f} units",
       "id": lambda v: v, "number": lambda v: f"{v} business days"}


def fresh(base):
    e = core.Env.__new__(core.Env)
    e.db = core.sqlite3.connect(":memory:")
    base.db.backup(e.db)
    e.db.row_factory = core.sqlite3.Row
    return e


def check_tasks(db_path, tasks_path):
    tasks = [json.loads(line) for line in open(tasks_path, encoding="utf-8")]
    fails = collections.Counter()
    for t in tasks:
        env = core.Env(db_path)
        for name, args in t["reference_calls"]:
            getattr(env, name)(**args)
        answer = "; ".join(FMT[k["kind"]](k["value"]) for k in t["keys"])
        if t["violated_rules"]:
            answer += f". Not executed: it would break rule {t['violated_rules'][0]}."
        g = grade.grade(t, env.state(t["customers"]), answer)
        if not g["passed"]:
            fails[t["template"]] += 1
            if fails[t["template"]] <= 2:
                print(t["task_id"], t["template"], g, answer[:200])
    print(json.dumps({"tasks": len(tasks), "failed": sum(fails.values()), "by_template": dict(fails)}))
    return not fails


def verify_composite(db_path, tasks_path):
    base = core.Env(db_path)

    def run(task, calls):
        e = fresh(base)
        for tool, args in calls:
            getattr(e, tool)(**args)
        return e.state(task["customers"])

    bad, counts = [], collections.Counter()
    for task in map(json.loads, open(tasks_path, encoding="utf-8")):
        calls = task["reference_calls"]
        answer = " ".join(str(k["value"]) for k in task["keys"]) + "".join(f" rule {r}" for r in task["violated_rules"][:1])
        e = fresh(base)
        broke = []
        for tool, args in calls:
            broke += core.check(e.db, tool, args)
            getattr(e, tool)(**args)
        ref = grade.grade(task, e.state(task["customers"]), answer)
        part = grade.grade(task, run(task, calls[:-1]), answer)
        a = int(re.search(r"invest \$([\d,]+)", task["prompt"]).group(1).replace(",", ""))  # the amount asked for
        case = "blocked" if task["violated_rules"] else ("full" if any(c[0] == "subscribe" and c[1]["amount"] == a for c in calls) else "capped")
        counts[case] += 1
        if broke or not ref["passed"] or part["passed"]:
            bad.append((task["task_id"], case, broke, ref, part["passed"]))
        if case != "full":
            bond = task["keys"][2]["value"]
            naive = run(task, calls[:1] + [["subscribe", {"customer_id": calls[0][1]["customer_id"], "product_id": bond, "amount": a}]])
            counts[f"naive fails ({case})"] += not grade.grade(task, naive, answer)["passed"]
    print(json.dumps({"tasks": sum(counts[k] for k in ("full", "capped", "blocked")), "counts": counts, "failed_checks": len(bad)}))
    for b in bad[:10]:
        print(b)
    return not bad
