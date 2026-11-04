"""Run one harness configuration on a task file: every task RUNS times, against one or more OpenAI-compatible servers.

docs/history/04_evaluation.md: per-episode rows (task id, instance, type, configuration, model, run, passed, steps, tokens,
tool errors, rule rejections, verification rejections, failure category) go to <out>/results.jsonl and the
trajectories to <out>/trajectories.jsonl, each line written and flushed as its episode ends. An episode already in
results.jsonl is not run again, so a stopped run resumes; one that ended on a server error ("api_error"), on the
budget ("budget") or on an error of the harness itself ("harness_error") is run again, and the analysis keeps only each
episode's last row.

Version 2 (research/log/2026-10-29-0836.md) added a paid provider: with --provider deepseek the requests go to
DeepSeek's API in thinking mode, the key is read from --api-key-file (never printed), and every call's usage is kept in
its episode's row. The spend is computed from that usage at the given prices per million tokens; before each model call
an episode stops ("budget") once the spend plus a reserve for the calls in flight (concurrency x the largest one call
can cost) reaches --budget-usd.

Version 3 (research/log/2026-10-31-0150.md): one run.py for every environment kind (each task's "env" field selects
it: wealth, ml, docfinqa, report_gate, market), the v3 loop (harness/agent.py), "harness": "v3" in every row, and a
trajectory line that holds the messages exactly as sent, the tools, the sampling settings and one record per model
turn with its tokens. Resuming into a directory whose rows are not v3's is refused. --keep-results and
--summary-chars are accepted for version 2's command lines and ignored, since v3 never summarizes. An exception in one
episode is written as its row (end "harness_error") instead of stopping the run.

Version 4 (research/log/2026-10-31-0150.md, 19:3x UTC): with several --base-url, each episode goes to the server with
the fewest episodes in flight (ties to the earliest one given) instead of to the next one in turn; in turn, a slower
server kept collecting the slots its long episodes held until its cache overflowed.

Usage: python -B run.py --tasks <tasks.jsonl> --db <instance.sqlite or docs dir> --config H1 --runs 3 --out <dir>
       --base-url http://127.0.0.1:18003/v1 [--base-url ...] [--limit N] [--concurrency 32] [--model gpt-oss-120b]
       [--max-steps 100] [--max-tokens 4096] [--provider vllm|qwen|deepseek] [--reasoning-effort medium]
       [--tool-result-chars 8000] [--max-checklist-calls 20] [--redo-steps 10] [--no-token-records] [--hint]
       [--provider deepseek --api-key-file <path> --budget-usd 10 --price-in-miss .. --price-in-hit .. --price-out ..
        --max-prompt-tokens 131072]
"""
import argparse
import asyncio
import json
import os
import sys
import traceback

from harness import agent, hint

RERUN = ("api_error", "budget", "harness_error")


def parse(argv):
    ap = argparse.ArgumentParser(prog="run.py")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--db", required=True, help="the database instance (wealth, report_gate), the documents directory "
                                                "(docfinqa) or the market snapshot; a task's own \"db\" wins")
    ap.add_argument("--config", required=True, choices=list(agent.CONFIGS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--base-url", action="append", default=[])
    ap.add_argument("--max-steps", type=int, default=agent.Settings.max_steps, help="task turns per episode")
    ap.add_argument("--max-tokens", type=int, default=agent.Settings.max_tokens, help="output tokens per model call")
    ap.add_argument("--tool-result-chars", type=int, default=agent.Settings.tool_result_chars,
                    help="a tool result is cut at this many characters when it is produced")
    ap.add_argument("--max-checklist-calls", type=int, default=agent.Settings.max_checklist_calls)
    ap.add_argument("--redo-steps", type=int, default=agent.Settings.redo_steps,
                    help="turns after a rejected finish that do not count toward --max-steps")
    ap.add_argument("--no-token-records", action="store_true", help="do not ask the server for token ids and logprobs")
    ap.add_argument("--keep-results", type=int, default=None, help=argparse.SUPPRESS)  # version 2's, ignored
    ap.add_argument("--summary-chars", type=int, default=None, help=argparse.SUPPRESS)  # version 2's, ignored
    ap.add_argument("--provider", choices=["vllm", "qwen", "deepseek"], default="vllm", help="vllm: gpt-oss; qwen: Qwen on vLLM")
    ap.add_argument("--api-key-file", default=None, help="a file holding the API key (deepseek)")
    ap.add_argument("--budget-usd", type=float, default=None, help="spending cap of this run (deepseek)")
    for k in ("in-miss", "in-hit", "out"):
        ap.add_argument(f"--price-{k}", type=float, default=None, help="US dollars per million tokens (deepseek)")
    ap.add_argument("--max-prompt-tokens", type=int, default=131072, help="largest prompt, for the budget's reserve")
    ap.add_argument("--reasoning-effort", default=None, help="default: medium for vllm and qwen, high for deepseek")
    ap.add_argument("--hint", action="store_true", help="wealth tasks get their template's steps as a hint (hint.py)")
    return ap.parse_args(argv)


def settings_of(a):
    s = agent.Settings(provider=a.provider, max_steps=a.max_steps, max_tokens=a.max_tokens,
                       tool_result_chars=a.tool_result_chars, max_checklist_calls=a.max_checklist_calls,
                       redo_steps=a.redo_steps, token_records=not a.no_token_records,
                       reasoning_effort=a.reasoning_effort or ("high" if a.provider == "deepseek" else "medium"))
    api_key = "none"
    if a.provider == "deepseek":
        prices = (a.price_in_miss, a.price_in_hit, a.price_out)
        if a.api_key_file is None or a.budget_usd is None or None in prices:
            sys.exit("--provider deepseek needs --api-key-file, --budget-usd and the three --price-* values")
        api_key = open(a.api_key_file, encoding="utf-8").read().strip()
        reserve = a.concurrency * (a.max_prompt_tokens * a.price_in_miss + a.max_tokens * a.price_out) / 1e6
        if reserve >= a.budget_usd:
            sys.exit(f"the reserve for {a.concurrency} calls in flight (${reserve:.2f}) leaves nothing of the cap")
        s.budget = agent.Budget(a.budget_usd - reserve, *prices)
    return s, api_key


def todo_of(a, tasks, res_path):
    """The (task, run) pairs still to run; refuses a results file written by another harness version."""
    done = set()
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        other = {r.get("harness", "v2") for r in rows} - {"v3"}
        if other:
            sys.exit(f"{res_path} holds rows of harness {sorted(other)}; write v3 runs to a new --out")
        # an episode cut short by a server error (e.g. the server was stopped) is run again
        done = {(r["task_id"], r["run"]) for r in rows if r["end"] not in RERUN}
    return [(t, k) for k in range(1, a.runs + 1) for t in tasks if (t["task_id"], k) not in done]


def main(argv=None, clients=None):
    """clients: OpenAI-compatible clients to use instead of --base-url (the tests' stubs)."""
    a = parse(argv)
    if a.keep_results is not None or a.summary_chars is not None:
        print("note: --keep-results and --summary-chars are ignored in v3 (the context is append-only)", file=sys.stderr)
    settings, api_key = settings_of(a)
    if clients is None:
        if not a.base_url:
            sys.exit("give at least one --base-url")
        from openai import AsyncOpenAI
        # a 4,096-token reply at the ~7 tokens/s of a 4B server with 64 episodes in flight takes about 600 s, so the
        # wait is 1,800 s (research/log/2026-10-30-1145.md)
        clients = [AsyncOpenAI(base_url=u, api_key=api_key, timeout=1800, max_retries=2) for u in a.base_url]
    inflight = [0] * len(clients)
    with open(a.tasks, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f][:a.limit]
    os.makedirs(a.out, exist_ok=True)
    res_path, traj_path = os.path.join(a.out, "results.jsonl"), os.path.join(a.out, "trajectories.jsonl")
    todo = todo_of(a, tasks, res_path)
    n = {"done": 0, "passed": 0}

    async def run_all():
        sem = asyncio.Semaphore(a.concurrency)
        res_f, traj_f = open(res_path, "a", encoding="utf-8"), open(traj_path, "a", encoding="utf-8")

        async def one(t, k):
            async with sem:
                if settings.budget is not None and settings.budget.exhausted():
                    return
                wealth = t.get("env", "wealth") == "wealth"
                suffix = hint.hint_text(t["template"]) if a.hint and wealth and t.get("template") in hint.HINTS else ""
                c = min(range(len(clients)), key=inflight.__getitem__)
                inflight[c] += 1
                try:
                    row, traj = await agent.run_one(t, a.config, a.db, clients[c], a.model, k, settings, suffix)
                except Exception as e:  # one episode's bug does not stop the others
                    row = {"task_id": t["task_id"], "env": t.get("env", "wealth"), "config": a.config, "model": a.model,
                           "run": k, "harness": "v3", "passed": False, "end": "harness_error", "failure": "harness_error",
                           "harness_error": "".join(traceback.format_exception_only(type(e), e))[-500:]}
                    traj = {"task_id": t["task_id"], "run": k, "config": a.config, "harness": "v3", "messages": None}
                finally:
                    inflight[c] -= 1
            if a.hint:
                row["hint"] = bool(suffix)
            res_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            traj_f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            res_f.flush()
            traj_f.flush()
            n["done"] += 1
            n["passed"] += row["passed"]
            if n["done"] % 20 == 0:
                print(f"{n['done']}/{len(todo)} episodes, {n['passed']} passed", flush=True)

        await asyncio.gather(*(one(t, k) for t, k in todo))
        res_f.close()
        traj_f.close()

    asyncio.run(run_all())
    b = settings.budget
    spend = {} if b is None else {"calls": b.calls, "spent_usd": round(b.spent, 4)}
    print(json.dumps({"config": a.config, "episodes": n["done"], "passed": n["passed"]} | spend))
    return n
