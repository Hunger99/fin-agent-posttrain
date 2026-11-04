"""Run one harness configuration on a task split: every task RUNS times, against one or more OpenAI-compatible servers.

docs/history/04_evaluation.md: per-episode rows (task id, instance, type, configuration, model, run, passed, steps, tokens,
tool errors, rule rejections, verification rejections, failure category) go to <out>/results.jsonl and the messages
to <out>/trajectories.jsonl; an episode already in results.jsonl is not run again, so a stopped run resumes (one
that ended on a server error is run again, and the analysis keeps only each episode's last row).

Usage: python run.py --tasks <tasks.jsonl> --db <instance.sqlite> --config H0 --runs 3 --out <dir>
       [--limit N] [--concurrency 32] [--model gpt-oss-120b] [--keep-results 5] [--max-steps 60] [--summary-chars 200] --base-url http://127.0.0.1:18003/v1 [...]
"""
import argparse
import asyncio
import itertools
import json
import os
import sys

from openai import AsyncOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))
import agent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True)
ap.add_argument("--db", required=True)
ap.add_argument("--config", required=True, choices=list(agent.CONFIGS))
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--model", default="gpt-oss-120b")
ap.add_argument("--base-url", action="append", required=True)
ap.add_argument("--keep-results", type=int, default=agent.KEEP_RESULTS,
                help="context management: how many latest tool results stay whole (H1 and its ablations)")
ap.add_argument("--max-steps", type=int, default=agent.MAX_STEPS, help="model calls per task, in every configuration alike")
ap.add_argument("--summary-chars", type=int, default=agent.SUMMARY_CHARS,
                help="context management: characters kept of an older tool result (H1 and its ablations)")
a = ap.parse_args()
agent.KEEP_RESULTS, agent.MAX_STEPS, agent.SUMMARY_CHARS = a.keep_results, a.max_steps, a.summary_chars

tasks = [json.loads(line) for line in open(a.tasks, encoding="utf-8")][:a.limit]
os.makedirs(a.out, exist_ok=True)
res_path, traj_path = os.path.join(a.out, "results.jsonl"), os.path.join(a.out, "trajectories.jsonl")
done = set()
if os.path.exists(res_path):
    # an episode cut short by a server error (e.g. the server was stopped) is run again
    done = {(r["task_id"], r["run"]) for r in map(json.loads, open(res_path, encoding="utf-8")) if r["end"] != "api_error"}
todo = [(t, k) for k in range(1, a.runs + 1) for t in tasks if (t["task_id"], k) not in done]
KINDS = {"wealth": agent.Episode}
try:  # the other environments, once built
    import docfinqa  # noqa: E402
    KINDS["docfinqa"] = docfinqa.DocEpisode
    import ml_tasks  # noqa: E402
    KINDS["ml"] = ml_tasks.MLEpisode
except ImportError:
    pass
clients = itertools.cycle([AsyncOpenAI(base_url=u, api_key="none", timeout=600, max_retries=2) for u in a.base_url])


async def main():
    sem = asyncio.Semaphore(a.concurrency)
    res_f, traj_f = open(res_path, "a", encoding="utf-8"), open(traj_path, "a", encoding="utf-8")
    n = {"done": 0, "passed": 0}

    async def one(t, k):
        async with sem:
            row, msgs = await agent.run_one(t, a.config, a.db, next(clients), a.model, k, KINDS[t.get("env", "wealth")])
        res_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        traj_f.write(json.dumps({"task_id": t["task_id"], "run": k, "config": a.config, "messages": msgs}, ensure_ascii=False) + "\n")
        res_f.flush()
        traj_f.flush()
        n["done"] += 1
        n["passed"] += row["passed"]
        if n["done"] % 20 == 0:
            print(f"{n['done']}/{len(todo)} episodes, {n['passed']} passed", flush=True)

    await asyncio.gather(*(one(t, k) for t, k in todo))
    print(json.dumps({"config": a.config, "episodes": n["done"], "passed": n["passed"]}))


asyncio.run(main())
