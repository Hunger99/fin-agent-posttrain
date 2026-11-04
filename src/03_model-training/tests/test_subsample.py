"""Tests of src/03_model-training/utils/subsample_grpo_data.py on synthetic data, CPU only (rl_chain_c.sh stage 2b).

A synthetic pool of 2,000 tasks over 10 templates and a synthetic 8-run screen go through the real make_grpo_data.py
(--runs 8 --drop-unsolved --solved-frac 0.1 --seed 20261103, as screen_m1_k8.sh calls it), and its parquet through
subsample_grpo_data.py. The allocation is recomputed here independently with exact fractions.
Usage: python -B test_subsample.py <work dir>
"""
import collections
import fractions
import json
import os
import random
import subprocess
import sys

import pandas as pd

S = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
PY = sys.executable
W = sys.argv[1]
os.makedirs(W, exist_ok=True)
for f in os.listdir(W):
    if f.endswith((".parquet", ".json", ".jsonl")):
        os.remove(os.path.join(W, f))
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
failures = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        failures.append(what)


def run(*args, ok=True):
    p = subprocess.run([PY, "-B", *args], capture_output=True, text=True, env=env)
    if ok and p.returncode:
        print(p.stdout, p.stderr)
        raise SystemExit(f"command failed: {args}")
    return p


# 1. a synthetic pool and 8-run screen; uneven pass rates per template so the kept set is uneven too
rng = random.Random(7)
templates = [f"tpl{i}" for i in range(10)]
tasks, rows = [], []
for k, t in enumerate(templates):
    for j in range(200):
        tid = f"{t}-{j:04d}"
        tasks.append({"task_id": tid, "template": t, "prompt": f"question {tid}"})
        p = [0.05, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95][k]
        for r in range(8):
            rows.append({"task_id": tid, "run": r, "passed": rng.random() < p, "end": "finished"})
rows.append({"task_id": "tpl0-0000", "run": 0, "passed": True, "end": "api_error"})  # never counts
with open(f"{W}/tasks.jsonl", "w") as f:
    f.writelines(json.dumps(t) + "\n" for t in tasks)
with open(f"{W}/results.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
run(f"{S}/make_grpo_data.py", "--tasks", f"{W}/tasks.jsonl", "--screen", f"{W}/results.jsonl", "--runs", "8",
    "--drop-unsolved", "--solved-frac", "0.1", "--seed", "20261103", "--manifest", f"{W}/run2_train.manifest.json",
    "--out", f"{W}/run2_train.parquet")
src = pd.read_parquet(f"{W}/run2_train.parquet")
print(f"synthetic run2_train.parquet: {len(src)} rows, {dict(sorted(collections.Counter(e['template'] for e in src['extra_info']).items()))}")
check(len(src) > 939, "the synthetic selection keeps more than 939 tasks")

# 2. the subsample
args = [f"{S}/subsample_grpo_data.py", "--in", f"{W}/run2_train.parquet", "--n", "939", "--seed", "20261103",
        "--out", f"{W}/run2_train_939.parquet", "--manifest", f"{W}/run2_train_939.manifest.json"]
p = run(*args)
print(p.stdout.strip())
check(p.stdout.strip().splitlines()[-1] == f"TRAIN={W}/run2_train_939.parquet", "last line names the subsample")
sub = pd.read_parquet(f"{W}/run2_train_939.parquet")
check(len(sub) == 939, "939 rows")
check(list(sub.columns) == list(src.columns), "same columns as the input")

# independent allocation: exact quotas, floors, the largest remainders (ties by name)
size = collections.Counter(e["template"] for e in src["extra_info"])
total = sum(size.values())
q = {t: fractions.Fraction(939 * size[t], total) for t in size}
fl = {t: q[t].numerator // q[t].denominator for t in q}
extra = 939 - sum(fl.values())
for t in sorted(q, key=lambda t: (-(q[t] - fl[t]), t))[:extra]:
    fl[t] += 1
got = collections.Counter(e["template"] for e in sub["extra_info"])
check(dict(got) == {t: n for t, n in fl.items() if n}, f"per-template counts match the independent allocation {dict(sorted(fl.items()))}")

# rows are input rows, unchanged but for the index, in the input's order
pos = {e["task_id"]: i for i, e in enumerate(src["extra_info"])}
ids = [e["task_id"] for e in sub["extra_info"]]
check(len(set(ids)) == 939 and all(i in pos for i in ids), "every row is an input task, each once")
check([pos[i] for i in ids] == sorted(pos[i] for i in ids), "input row order kept")
check([e["index"] for e in sub["extra_info"]] == list(range(939)), "extra_info.index is 0..938")
same = all(sub.iloc[k]["prompt"][0]["content"] == src.iloc[pos[i]]["prompt"][0]["content"]
           and sub.iloc[k]["reward_model"]["ground_truth"] == i
           and json.loads(sub.iloc[k]["extra_info"]["task"]) == json.loads(src.iloc[pos[i]]["extra_info"]["task"])
           for k, i in enumerate(ids))
check(same, "prompt, ground truth and task JSON unchanged")

# the within-template draw is the documented one
r2 = random.Random(20261103)
want = set()
by = collections.defaultdict(list)
for e in src["extra_info"]:
    by[e["template"]].append(e["task_id"])
for t in sorted(by):
    want.update(r2.sample(sorted(by[t]), fl[t]))
check(set(ids) == want, "the draw is random.Random(20261103).sample over each template's sorted task ids, templates sorted")

m = json.load(open(f"{W}/run2_train_939.manifest.json"))
check(m["rows_in"] == len(src) and m["rows_out"] == 939 and m["task_ids"] == ids, "manifest counts and ids")
check({t: v["out"] for t, v in m["by_template"].items()} == fl, "manifest per-template allocation")

# 3. rerun: identical file kept; determinism; a different file refused; small input used whole
sha = m["output_sha256"]
p = run(*args)
check("kept" in p.stdout and p.stdout.strip().endswith(f"TRAIN={W}/run2_train_939.parquet"), "a rerun accepts the identical file")
os.rename(f"{W}/run2_train_939.parquet", f"{W}/first.parquet")
os.remove(f"{W}/run2_train_939.manifest.json")
run(*args)
check(json.load(open(f"{W}/run2_train_939.manifest.json"))["output_sha256"] == sha, "a fresh rebuild is byte-identical (sha256)")
sub.iloc[::-1].reset_index(drop=True).to_parquet(f"{W}/run2_train_939.parquet")
p = run(*args, ok=False)
check(p.returncode != 0 and "not replaced" in p.stderr, "a file with other rows is refused, not replaced")
small = src.iloc[:900].reset_index(drop=True)
small.to_parquet(f"{W}/small.parquet")
p = run(f"{S}/subsample_grpo_data.py", "--in", f"{W}/small.parquet", "--n", "939", "--seed", "20261103",
        "--out", f"{W}/small_939.parquet", "--manifest", f"{W}/small_939.manifest.json")
check(p.stdout.strip().endswith(f"TRAIN={W}/small.parquet") and not os.path.exists(f"{W}/small_939.parquet"),
      "900 rows: no subsample, the input is used whole, nothing written")

# 4. ties in the remainders go to the template whose name sorts first
sys.path.insert(0, S)
sys.argv = ["x"]
from subsample_grpo_data import allocate  # noqa: E402
alloc, _ = allocate({"b": 5, "a": 5, "c": 5}, 7)
check(alloc == {"a": 3, "b": 2, "c": 2}, f"tie-break by name: {alloc}")
alloc, _ = allocate({"x": 1, "y": 99}, 10)
check(sum(alloc.values()) == 10 and alloc["x"] <= 1, f"a small stratum never exceeds its size: {alloc}")

print("FAILURES: " + (", ".join(failures) if failures else "none"))
sys.exit(1 if failures else 0)
