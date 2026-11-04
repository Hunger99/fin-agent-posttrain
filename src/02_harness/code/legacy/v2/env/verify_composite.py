"""Checks every task of a generated composite task file (research/log/2026-10-30-0855.md, D6) before it is used:
1. the reference calls break no rule (core.check) and reach the gold state, and an answer made of the keys (and the
   broken rules, when there are any) passes grade();
2. doing part of it fails: the reference calls without the last one;
3. a naive policy fails whenever the rules change the request: redeem, then subscribe the requested amount as asked.
Prints the counts per case and exits 1 if check 1 or 2 fails on any task.
Usage: python -B verify_composite.py <db.sqlite> <tasks.jsonl>"""
import collections
import json
import re
import sys

import core
import grade

db_path, tasks_path = sys.argv[1:3]
base = core.Env(db_path)


def fresh():
    e = core.Env.__new__(core.Env)
    e.db = core.sqlite3.connect(":memory:")
    base.db.backup(e.db)
    e.db.row_factory = core.sqlite3.Row
    return e


def run(task, calls):
    e = fresh()
    for tool, args in calls:
        getattr(e, tool)(**args)
    return e.state(task["customers"])


bad, counts = [], collections.Counter()
for task in map(json.loads, open(tasks_path, encoding="utf-8")):
    calls = task["reference_calls"]
    answer = " ".join(str(k["value"]) for k in task["keys"]) + "".join(f" rule {r}" for r in task["violated_rules"][:1])
    e = fresh()
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
sys.exit(1 if bad else 0)
