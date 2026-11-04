"""Run one harness configuration on a task split: every task RUNS times, against one or more OpenAI-compatible servers.

docs/history/04_evaluation.md: per-episode rows (task id, instance, type, configuration, model, run, passed, steps, tokens,
tool errors, rule rejections, verification rejections, failure category) go to <out>/results.jsonl and the messages
to <out>/trajectories.jsonl; an episode already in results.jsonl is not run again, so a stopped run resumes (one
that ended on a server error is run again, and the analysis keeps only each episode's last row).

Version 2 (research/log/2026-10-29-0836.md) runs the v2 agent and environment, and adds a paid provider: with
--provider deepseek the requests go to DeepSeek's API in thinking mode, the key is read from --api-key-file (never
printed), and every call's usage is kept in its episode's row. The spend is computed from that usage at the given prices
per million tokens; before each model call an episode stops ("budget") once the spend plus a reserve for the calls in
flight (concurrency x the largest one call can cost) reaches --budget-usd. An episode stopped so is run again on resume.

Usage: python run.py --tasks <tasks.jsonl> --db <instance.sqlite> --config H0 --runs 3 --out <dir>
       [--limit N] [--concurrency 32] [--model gpt-oss-120b] [--keep-results 5] [--max-steps 60] [--summary-chars 200] --base-url http://127.0.0.1:18003/v1 [...]
       [--hint]  (wealth tasks with their template's steps as a hint, research/log/2026-10-29-1725.md)
       [--provider deepseek --api-key-file <path> --budget-usd 10 --price-in-miss .. --price-in-hit .. --price-out ..
        --max-prompt-tokens 131072 --max-tokens 4096 --reasoning-effort high]
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
ap.add_argument("--provider", choices=["vllm", "qwen", "deepseek"], default="vllm", help="vllm: gpt-oss; qwen: Qwen3.8 on vLLM")
ap.add_argument("--api-key-file", default=None, help="a file holding the API key (deepseek)")
ap.add_argument("--budget-usd", type=float, default=None, help="spending cap of this run (deepseek)")
for k in ("in-miss", "in-hit", "out"):
    ap.add_argument(f"--price-{k}", type=float, default=None, help="US dollars per million tokens (deepseek)")
ap.add_argument("--max-prompt-tokens", type=int, default=131072, help="largest prompt, for the budget's reserve")
ap.add_argument("--max-tokens", type=int, default=agent.MAX_TOKENS, help="output tokens per model call, reasoning included")
ap.add_argument("--reasoning-effort", default=None, help="default: medium for vllm and qwen, high for deepseek")
ap.add_argument("--hint", action="store_true", help="wealth tasks get their template's steps as a hint (hint.py)")
a = ap.parse_args()
agent.KEEP_RESULTS, agent.MAX_STEPS, agent.SUMMARY_CHARS = a.keep_results, a.max_steps, a.summary_chars
agent.PROVIDER, agent.MAX_TOKENS = a.provider, a.max_tokens
agent.REASONING_EFFORT = a.reasoning_effort or ("high" if a.provider == "deepseek" else "medium")
api_key = "none"
if a.provider == "deepseek":
    prices = (a.price_in_miss, a.price_in_hit, a.price_out)
    if a.api_key_file is None or a.budget_usd is None or None in prices:
        sys.exit("--provider deepseek needs --api-key-file, --budget-usd and the three --price-* values")
    api_key = open(a.api_key_file, encoding="utf-8").read().strip()
    reserve = a.concurrency * (a.max_prompt_tokens * a.price_in_miss + a.max_tokens * a.price_out) / 1e6
    if reserve >= a.budget_usd:
        sys.exit(f"the reserve for {a.concurrency} calls in flight (${reserve:.2f}) leaves nothing of the cap")
    agent.BUDGET = agent.Budget(a.budget_usd - reserve, *prices)

tasks = [json.loads(line) for line in open(a.tasks, encoding="utf-8")][:a.limit]
os.makedirs(a.out, exist_ok=True)
res_path, traj_path = os.path.join(a.out, "results.jsonl"), os.path.join(a.out, "trajectories.jsonl")
done = set()
if os.path.exists(res_path):
    # an episode cut short by a server error (e.g. the server was stopped) is run again
    done = {(r["task_id"], r["run"]) for r in map(json.loads, open(res_path, encoding="utf-8"))
            if r["end"] not in ("api_error", "budget")}
todo = [(t, k) for k in range(1, a.runs + 1) for t in tasks if (t["task_id"], k) not in done]
KINDS = {"wealth": agent.Episode}
try:  # the other environments, once built
    import docfinqa  # noqa: E402
    KINDS["docfinqa"] = docfinqa.DocEpisode
    import ml_tasks  # noqa: E402
    KINDS["ml"] = ml_tasks.MLEpisode
    import report_gate  # noqa: E402
    KINDS["report_gate"] = report_gate.GateEpisode
except ImportError:
    pass
if a.hint:
    import hint  # noqa: E402
    KINDS["wealth"] = hint.HintEpisode
# a 4,096-token reply at the ~7 tokens/s of a 4B server with 64 episodes in flight takes about 600 s, so the wait is
# 1,800 s (research/log/2026-10-30-1145.md)
clients = itertools.cycle([AsyncOpenAI(base_url=u, api_key=api_key, timeout=1800, max_retries=2) for u in a.base_url])


async def main():
    sem = asyncio.Semaphore(a.concurrency)
    res_f, traj_f = open(res_path, "a", encoding="utf-8"), open(traj_path, "a", encoding="utf-8")
    n = {"done": 0, "passed": 0}

    async def one(t, k):
        async with sem:
            if agent.BUDGET is not None and agent.BUDGET.exhausted():
                return
            row, msgs = await agent.run_one(t, a.config, a.db, next(clients), a.model, k, KINDS[t.get("env", "wealth")])
        if a.hint:
            row["hint"] = True
        res_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        traj_f.write(json.dumps({"task_id": t["task_id"], "run": k, "config": a.config, "messages": msgs}, ensure_ascii=False) + "\n")
        res_f.flush()
        traj_f.flush()
        n["done"] += 1
        n["passed"] += row["passed"]
        if n["done"] % 20 == 0:
            print(f"{n['done']}/{len(todo)} episodes, {n['passed']} passed", flush=True)

    await asyncio.gather(*(one(t, k) for t, k in todo))
    spend = {} if agent.BUDGET is None else {"calls": agent.BUDGET.calls, "spent_usd": round(agent.BUDGET.spent, 4)}
    print(json.dumps({"config": a.config, "episodes": n["done"], "passed": n["passed"]} | spend))


asyncio.run(main())
