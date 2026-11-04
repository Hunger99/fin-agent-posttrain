"""Regression check by replay: version 2's sampled episodes, replayed through v3, are graded as version 2 graded them.

Each v2 episode that ended "finished" without a text call is replayed: a stub client returns v2's assistant messages
in order, so v3 executes the same calls in the same environment kind. The replay is in sync while every tool result
v3 returns equals v2's (ignoring the example of a valid call, which v3 takes from each kind's own examples); v3 may
answer differently where it changed the shared parts (get_fund's 20 ids, the cut at 8,000 characters, the cap of 20
set_checklist calls), and then the model's later calls no longer answer what v3 showed, so only the episodes in sync
to the end are compared. For those, v3's final state and answer are v2's, and v3's grade must give the same pass/fail.

Usage: PYTHONPATH=<v3 dir> python -B tests/replay_v2.py <v2 run dir> <tasks.jsonl> <db or docs dir> [<limit>]
"""
import asyncio
import collections
import json
import os
import sys
from types import SimpleNamespace

from harness import agent

DEC = json.JSONDecoder()


def norm(text):
    """A tool result without its example of a valid call: (the leading JSON, the rest of the text)."""
    try:
        obj, end = DEC.raw_decode(text)
    except json.JSONDecodeError:
        return text, ""
    if isinstance(obj, dict):
        obj.pop("example_of_a_valid_call", None)
    return obj, text[end:]


class Replay:
    def __init__(self, v2_msgs):
        self.assistant = [m for m in v2_msgs if m["role"] == "assistant"]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, messages, **_):
        k = sum(m["role"] == "assistant" for m in messages)
        if k >= len(self.assistant):
            raise RuntimeError("replay: no more v2 replies")
        m = self.assistant[k]
        tcs = [SimpleNamespace(id=c["id"], type="function", function=SimpleNamespace(**c["function"]))
               for c in m.get("tool_calls") or []] or None
        msg = SimpleNamespace(content=m.get("content") or None, tool_calls=tcs, reasoning=m.get("reasoning_content"))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)


def in_sync(v2_msgs, v3_msgs):
    a = [m["content"] for m in v2_msgs if m["role"] == "tool"]
    b = [m["content"] for m in v3_msgs if m["role"] == "tool"]
    return len(a) == len(b) and all(norm(x) == norm(y) for x, y in zip(a, b))


async def main(run_dir, tasks_path, db, limit):
    with open(tasks_path, encoding="utf-8") as f:
        tasks = {t["task_id"]: t for t in map(json.loads, f)}
    rows = {}
    with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[(r["task_id"], r["run"])] = r
    counts, differ = collections.Counter(), []
    settings = agent.Settings(provider="qwen", max_steps=100, token_records=False)
    with open(os.path.join(run_dir, "trajectories.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                tr = json.loads(line)
            except json.JSONDecodeError:  # a line still being written by a live run
                continue
            r = rows.get((tr["task_id"], tr["run"]))
            if r is None or r["end"] != "finished" or tr["task_id"] not in tasks:
                continue
            counts["v2_finished"] += 1
            if r.get("text_calls") or any("_text_call" in m for m in tr["messages"]):
                counts["skipped_text_call"] += 1
                continue
            row, traj = await agent.run_one(tasks[tr["task_id"]], r["config"], db, Replay(tr["messages"]), "replay",
                                            tr["run"], settings)
            if not in_sync(tr["messages"], traj["messages"]) or row["end"] != "finished":
                counts["out_of_sync"] += 1
                continue
            counts["compared"] += 1
            counts["v2_passed"] += r["passed"]
            counts["v3_passed"] += row["passed"]
            if row["passed"] != r["passed"]:
                differ.append(tr["task_id"])
            if limit and counts["compared"] >= limit:
                break
    counts["pass_fail_differs"] = len(differ)
    print(json.dumps({"run": run_dir, **counts, "differ_examples": differ[:5]}))
    return not differ


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)) else 1)
