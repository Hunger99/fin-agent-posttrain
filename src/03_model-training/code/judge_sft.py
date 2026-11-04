"""Stage 8 of filter_sft.py: an LLM judge of whether a passed market episode got its answer by the right method
(research/log/2026-10-31-0150.md).

Market answers are graded by key values alone, so an episode can pass with a wrong method whose number happens to fall
within the tolerance (a small count, a split-adjusted price where the raw one was asked and no split intervened, a
recalled rather than computed value). The judge reads the question, the generator's reference method (the task's
solution, operations and parameters), every tool call the agent made with its arguments, each tool result cut to
--result-chars, and the final answer, and returns {"verdict": "ok" | "flawed", "reason": ...}. Since the judge is the
same model that wrote most of these episodes, it is told to compare with the reference method rather than to re-solve
the task. Episodes whose reply does not parse are judged again once, then counted as "unparsed" (not "ok").

Output: one JSON line per episode (source, task_id, run, verdict, reason), appended and resumable.

Usage: python -B judge_sft.py --source NAME:RUN_DIR:TASKS --out <judge.jsonl> --base-url <url> [--base-url <url> ...]
       --model <name>
       [--concurrency 32] [--result-chars 800] [--only-passed] [--limit N]
"""
import argparse
import asyncio
import itertools
import json
import os
import re

SYSTEM = ("You review how an AI agent answered a quantitative finance question with data tools. You are given the "
          "question, the reference method that defines the correct answer, and the agent's tool calls, tool results "
          "and final answer. The final answer is already known to match the reference value within tolerance. Decide "
          "whether the agent got it by a method consistent with the reference: the right instrument or series, the "
          "price type (raw, split-adjusted or total return), the date rule (trading days, the window, the stated "
          "rule for non-trading days), the units and the aggregation. Small implementation differences that cannot "
          "change the value are fine. Answer \"flawed\" if the agent computed a different quantity that happens to "
          "be close, used a convention the question rules out, or wrote a value it did not obtain from the tools "
          "(recalled or guessed). Do not re-solve the task. Reply with one JSON object and nothing else: "
          "{\"verdict\": \"ok\" or \"flawed\", \"reason\": \"one sentence\"}.")


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="NAME:RUN_DIR:TASKS[+TASKS]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", action="append", required=True, help="one or more servers of the same model, one per episode in turn")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--result-chars", type=int, default=800)
    ap.add_argument("--max-tokens", type=int, default=8192)  # 4,096 left 3 of 20 long xbrl transcripts unparsed
    ap.add_argument("--only-passed", action="store_true", help="judge only the episodes that passed grading")
    ap.add_argument("--limit", type=int, default=None, help="judge at most this many (for a check)")
    return ap.parse_args(argv)


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def transcript(task, msgs, cut):
    meta = task.get("meta") or {}
    out = [f"QUESTION:\n{task['prompt']}",
           f"REFERENCE METHOD: {task.get('solution', '')}\nOPERATIONS: {meta.get('operations')}\n"
           f"PARAMETERS: {json.dumps(meta.get('params'), default=str)}",
           "AGENT:"]
    for m in msgs:
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                out.append(f"[call] {c['function']['name']} {c['function']['arguments'][:4000]}")
        elif m["role"] == "tool":
            text = m.get("content") or ""
            out.append(f"[result] {text[:cut]}" + (" ...(cut)" if len(text) > cut else ""))
    return "\n".join(out)


def verdict_of(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group())
    except ValueError:
        return None
    return j if j.get("verdict") in ("ok", "flawed") else None


async def main_async(a):
    from openai import AsyncOpenAI
    clients = itertools.cycle([AsyncOpenAI(base_url=u, api_key="none", timeout=1800, max_retries=2) for u in a.base_url])
    name, run_dir, task_files = a.source.split(":", 2)
    tasks = {t["task_id"]: t for p in task_files.split("+") for t in jsonl(p)}
    res = {}
    for r in jsonl(os.path.join(run_dir, "results.jsonl")):
        res[(r["task_id"], r["run"])] = r
    done = {(j["task_id"], j["run"]) for j in jsonl(a.out)} if os.path.exists(a.out) else set()
    todo = []
    for t in jsonl(os.path.join(run_dir, "trajectories.jsonl")):
        k = (t["task_id"], t["run"])
        if k in done or k not in res or (a.only_passed and not res[k]["passed"]):
            continue
        todo.append((k, t["messages"]))
    todo = todo[:a.limit]
    sem, out = asyncio.Semaphore(a.concurrency), open(a.out, "a", encoding="utf-8")
    n = {"ok": 0, "flawed": 0, "unparsed": 0}

    async def one(k, msgs):
        async with sem:
            client = next(clients)
            user = transcript(tasks[k[0]], msgs, a.result_chars)
            j = None
            for _ in range(2):
                r = await client.chat.completions.create(
                    model=a.model, max_tokens=a.max_tokens, temperature=0.0,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}})
                j = verdict_of(r.choices[0].message.content)
                if j:
                    break
            row = {"source": name, "task_id": k[0], "run": k[1], "verdict": j["verdict"] if j else "unparsed",
                   "reason": (j or {}).get("reason", "")}
            n[row["verdict"]] += 1
            out.write(json.dumps(row) + "\n")
            out.flush()

    await asyncio.gather(*(one(k, m) for k, m in todo))
    print(json.dumps({"judged": len(todo), **n}))


if __name__ == "__main__":
    asyncio.run(main_async(parse()))
