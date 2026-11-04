"""verl parquet files of market tasks for fin_agent_loop.py (research/log/2026-11-01-0010.md, 20:xx UTC).

One row per task: the task's prompt as the chat prompt (verl's dataset only uses it for its length filter; the agent loop
builds the episode's own messages), agent_name "fin_harness", and the task's JSON line in extra_info["task"].
With --screen, a harness v3 results.jsonl of the start model on these tasks (several runs each), tasks are kept by how
many runs passed: every task some run failed, and a seeded fraction --solved-frac of the tasks every run passed.

Version 2 (research/log/2026-11-01-0010.md, 11-02 22:xx UTC): selection by the pass rate of a k-run screen, as
public RL recipes keep the sometimes-solved tasks and drop the never-solved ones, whose groups all score 0 and give
GRPO no gradient. --drop-unsolved also drops the tasks no run passed (0/k); --runs K counts each (task, run) pair
once, by its last usable row, and only tasks with all K runs usable as screened, so a task short of runs (server
errors) is dropped rather than selected on a coarser pass rate; --manifest writes the per-task runs and passes, each
task's selection class and the counts per template. Without these flags the selection, the rng draws and the parquet
are as before (run1_train.parquet rebuilt byte-identical). Rows run.py marks api_error or harness_error never count.

Usage: python -B make_grpo_data.py --tasks <tasks.jsonl> --out <file.parquet> [--screen <results.jsonl> --solved-frac 0.25]
           [--drop-unsolved] [--runs K] [--manifest <file.json>] [--limit N] [--seed 20261101]
"""
import argparse
import collections
import json
import random

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--screen")
ap.add_argument("--solved-frac", type=float, default=0.25)
ap.add_argument("--drop-unsolved", action="store_true", help="drop the tasks no screening run passed")
ap.add_argument("--runs", type=int, help="the screen's k: a task counts as screened only with k distinct usable runs")
ap.add_argument("--manifest", help="JSON with the per-task pass counts and the selection")
ap.add_argument("--limit", type=int)
ap.add_argument("--seed", type=int, default=20261101)
a = ap.parse_args()

tasks = [json.loads(line) for line in open(a.tasks)]
rng = random.Random(a.seed)
if a.screen:
    runs, passed, last = collections.Counter(), collections.Counter(), {}
    for line in open(a.screen):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("end") in ("api_error", "harness_error"):
            continue
        if a.runs:
            last[(r["task_id"], r["run"])] = bool(r["passed"])
            continue
        runs[r["task_id"]] += 1
        passed[r["task_id"]] += bool(r["passed"])
    for (tid, _), ok in last.items():
        runs[tid] += 1
        passed[tid] += ok
    screened = [t for t in tasks if runs[t["task_id"]] and (not a.runs or runs[t["task_id"]] == a.runs)]
    solved = [t for t in screened if passed[t["task_id"]] == runs[t["task_id"]]]
    keep = {t["task_id"] for t in screened if passed[t["task_id"]] < runs[t["task_id"]]
            and not (a.drop_unsolved and passed[t["task_id"]] == 0)}
    sampled = {t["task_id"] for t in rng.sample(solved, round(len(solved) * a.solved_frac))}
    keep |= sampled
    unsolved = sum(1 for t in screened if passed[t["task_id"]] == 0)
    print(f"screened {len(screened)} of {len(tasks)}; every run passed {len(solved)}; kept {len(keep)}"
          + (f"; no run passed {unsolved}, dropped" if a.drop_unsolved else ""))
    if a.manifest:
        def cls(t):
            n, p = runs[t["task_id"]], passed[t["task_id"]]
            return "unscreened" if not n or (a.runs and n != a.runs) else "none" if p == 0 else "all" if p == n else "some"
        per = [{"task_id": t["task_id"], "template": t.get("template"), "runs": runs[t["task_id"]],
                "passed": passed[t["task_id"]], "class": cls(t), "kept": t["task_id"] in keep} for t in tasks]
        by = collections.defaultdict(collections.Counter)
        for x in per:
            by[x["template"]][x["class"] + ("_kept" if x["kept"] else "_dropped")] += 1
        json.dump({"args": vars(a), "tasks": len(tasks), "screened": len(screened), "kept": len(keep),
                   "classes": dict(collections.Counter(x["class"] for x in per)),
                   "runs_histogram": dict(sorted(collections.Counter(x["runs"] for x in per).items())),
                   "passed_histogram": dict(sorted(collections.Counter(x["passed"] for x in per if x["class"] != "unscreened").items())),
                   "by_template": {k: dict(sorted(v.items())) for k, v in sorted(by.items())}, "per_task": per},
                  open(a.manifest, "w"), indent=1)
    tasks = [t for t in tasks if t["task_id"] in keep]
rng.shuffle(tasks)
if a.limit:
    tasks = tasks[: a.limit]
rows = [{"data_source": "fin_market", "agent_name": "fin_harness",
         "prompt": [{"role": "user", "content": t["prompt"]}],
         "reward_model": {"style": "rule", "ground_truth": t["task_id"]},
         "extra_info": {"index": i, "task_id": t["task_id"], "template": t.get("template") or "", "task": json.dumps(t)}}
        for i, t in enumerate(tasks)]
pd.DataFrame(rows).to_parquet(a.out)
print(f"{len(rows)} rows -> {a.out}; templates {dict(collections.Counter(t.get('template') for t in tasks))}")
