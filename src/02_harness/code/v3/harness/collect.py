"""Collect agent SFT trajectories on one task file with the sampling plan of research/log/2026-10-29-1725.md.

1. Sampled: under H1, every task is run with seeds 1 and 2; a task with fewer than 2 passing runs gets seed 3, then 4;
   a task with none after 4 gets seeds 5 to 8, one at a time, until one passes. Each round is one run.py call on the
   tasks still short, into <out>/sampled, so a stopped collection resumes from its results.jsonl.
2. Hint (wealth only): a task with no pass after 8 runs gets up to 2 runs with its template's steps as a hint
   (hint.py), into <out>/hint.
3. Reference (wealth only): a task still without a pass gets golden.py's trajectory, into <out>/golden.
<out>/collect.status records each round's task count and time.

Version 3 (research/log/2026-10-31-0150.md): the rounds call v3's run.py and golden.py (the scripts next to the harness
package); --keep-results and --summary-chars are gone with the summarizing, and a template without a hint (such as
t5_review_switch) gets no hint round.

Usage: python -B collect.py --tasks <tasks.jsonl> --db <instance.sqlite or none> --out <dir> --base-url <url> [...]
       [--concurrency 18] [--max-steps 100] [--provider vllm|qwen] [--model gpt-oss-120b] [--reasoning-effort medium]
"""
import argparse
import collections
import datetime
import json
import os
import subprocess
import sys

from harness import golden, hint

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the directory of run.py and golden.py


def passes(out):
    """Passing runs per task in a run.py output directory, from each episode's last row."""
    path = os.path.join(out, "results.jsonl")
    last = {}
    if os.path.exists(path):
        for r in map(json.loads, open(path, encoding="utf-8")):
            last[(r["task_id"], r["run"])] = r
    n = collections.Counter()
    for (t, _), r in last.items():
        n[t] += r["passed"]
    return n


def main(a):
    tasks = [json.loads(line) for line in open(a.tasks, encoding="utf-8")]
    os.makedirs(a.out, exist_ok=True)
    st = open(os.path.join(a.out, "collect.status"), "a", encoding="utf-8")

    def log(msg):
        st.write(f"{msg} {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S}\n")
        st.flush()

    def run(name, subset, runs, extra=()):
        if not subset:
            return
        path = os.path.join(a.out, f"{name}_tasks.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for t in subset:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        log(f"{name}: {len(subset)} tasks, runs up to {runs}")
        cmd = [sys.executable, "-B", os.path.join(HERE, "run.py"), "--tasks", path, "--db", a.db, "--config", "H1",
               "--runs", str(runs), "--concurrency", str(a.concurrency), "--max-steps", str(a.max_steps),
               "--provider", a.provider, "--model", a.model,
               "--out", os.path.join(a.out, "hint" if "--hint" in extra else "sampled"), *extra]
        if a.reasoning_effort:
            cmd += ["--reasoning-effort", a.reasoning_effort]
        for u in a.base_url:
            cmd += ["--base-url", u]
        code = subprocess.call(cmd, stdout=open(os.path.join(a.out, f"{name}.log"), "a"), stderr=subprocess.STDOUT)
        log(f"{name}: exit {code}")
        if code:
            sys.exit(code)

    sampled = os.path.join(a.out, "sampled")
    run("round1", tasks, 2)
    for k in (3, 4):
        n = passes(sampled)
        run(f"round{k - 1}", [t for t in tasks if n[t["task_id"]] < 2], k)
    for k in range(5, 9):
        n = passes(sampled)
        run(f"round{k - 1}", [t for t in tasks if n[t["task_id"]] == 0], k)
    n = passes(sampled)
    left = [t for t in tasks if n[t["task_id"]] == 0 and t.get("env", "wealth") == "wealth"]
    hintable = [t for t in left if t["template"] in hint.HINTS]
    run("hint1", hintable, 1, ["--hint"])
    h = passes(os.path.join(a.out, "hint"))
    run("hint2", [t for t in hintable if h[t["task_id"]] == 0], 2, ["--hint"])
    h = passes(os.path.join(a.out, "hint"))
    rest = [t for t in left if h[t["task_id"]] == 0 and t["template"] in golden.POLICIES]
    if rest:
        path = os.path.join(a.out, "golden_tasks.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for t in rest:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        log(f"golden: {len(rest)} tasks")
        code = subprocess.call([sys.executable, "-B", os.path.join(HERE, "golden.py"), "--tasks", path, "--db", a.db,
                                "--out", os.path.join(a.out, "golden"), "--max-steps", str(a.max_steps),
                                "--provider", a.provider])
        log(f"golden: exit {code}")
    n = passes(sampled)
    log(f"collection done: {sum(1 for t in tasks if n[t['task_id']])} of {len(tasks)} tasks with a sampled pass, "
        f"{sum(1 for t in left if h[t['task_id']])} with a hinted pass, {len(rest)} with a reference trajectory")


def cli(argv=None):
    ap = argparse.ArgumentParser(prog="collect.py")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", action="append", required=True)
    ap.add_argument("--concurrency", type=int, default=18)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--provider", choices=["vllm", "qwen"], default="vllm", help="vllm: gpt-oss; qwen: Qwen3.8 on vLLM")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--reasoning-effort", default=None, help="default: run.py's")
    main(ap.parse_args(argv))
