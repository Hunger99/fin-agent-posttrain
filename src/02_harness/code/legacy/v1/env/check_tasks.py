"""Oracle check of a task split: replay each task's reference trades, answer with its key values, and grade.

research/log/2026-10-27-0307.md. Every task must pass; a failure means a template, its gold state or the grader is wrong.

Usage: python check_tasks.py <instance.sqlite> <tasks.jsonl>
"""
import collections
import json
import sys

import core
import grade

db_path, tasks = sys.argv[1], [json.loads(line) for line in open(sys.argv[2], encoding="utf-8")]
FMT = {"money": lambda v: f"${v:,.2f}", "percent": lambda v: f"{v:.2f}%", "units": lambda v: f"{v:,.4f} units",
       "id": lambda v: v, "number": lambda v: f"{v} business days"}
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
