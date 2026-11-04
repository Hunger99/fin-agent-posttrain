"""Stratified subsample of a GRPO training parquet to a fixed number of tasks (research/log/2026-11-01-0010.md, 11-02 22:4x UTC).

The second GRPO round and the PPO round train on the tasks the 8-run screen keeps (make_grpo_data.py --runs 8
--drop-unsolved); when it keeps more tasks than the first round's 939, they are cut to 939 so that the rounds see a
similar number of epochs. The rule was fixed before any screen result: strata are the templates (extra_info.template);
each template gets n * size / total tasks (proportional allocation), rounded down, and the tasks left over go one each
to the templates with the largest fractional parts (largest-remainder rounding; ties to the template whose name sorts
first); within a template the tasks are drawn with random.Random(seed).sample from its rows sorted by task id, the
templates taken in sorted order from one generator. The selected rows keep the input's order (make_grpo_data.py's
seeded shuffle) and get extra_info.index 0..n-1, as make_grpo_data.py numbers its rows. With no more rows than n, the
input is used whole and nothing is written.

The last line printed is TRAIN=<parquet to train on>, which rl_chain_c.sh reads. An existing --out is never
replaced: it is accepted when it holds exactly the rows this rule selects, in order, and refused otherwise.

Usage: python -B subsample_grpo_data.py --in <train.parquet> --n 939 --seed 20261103 --out <subsample.parquet>
           --manifest <subsample.manifest.json>
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

import pandas as pd


def allocate(sizes, n):
    """{template: tasks} with sum n by proportional allocation and largest-remainder rounding; sizes is {template: rows}."""
    total = sum(sizes.values())
    quota = {t: n * s / total for t, s in sizes.items()}
    alloc = {t: int(q) for t, q in quota.items()}   # floor: every quota is non-negative
    left = n - sum(alloc.values())
    order = sorted(sizes, key=lambda t: (-(quota[t] - alloc[t]), t))
    for t in order[:left]:
        alloc[t] += 1
    return alloc, quota


def select(df, n, seed):
    """The row positions (in the input's order) of the stratified subsample, and the allocation."""
    tpl = [e.get("template") or "" for e in df["extra_info"]]
    tid = [e["task_id"] for e in df["extra_info"]]
    if len(set(tid)) != len(tid):
        sys.exit("subsample_grpo_data: duplicate task ids in the input")
    by = collections.defaultdict(list)
    for i, t in enumerate(tpl):
        by[t].append(i)
    alloc, quota = allocate({t: len(v) for t, v in by.items()}, n)
    rng, keep = random.Random(seed), set()
    for t in sorted(by):
        rows = sorted(by[t], key=lambda i: tid[i])
        keep.update(rng.sample(rows, alloc[t]))
    return sorted(keep), alloc, quota, tpl, tid


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    if a.n < 1:
        sys.exit("subsample_grpo_data: --n must be positive")
    df = pd.read_parquet(a.inp)
    if len(df) <= a.n:
        print(f"{len(df)} rows <= {a.n}: no subsample, the input is used whole")
        print(f"TRAIN={a.inp}")
        return
    pos, alloc, quota, tpl, tid = select(df, a.n, a.seed)
    out = df.iloc[pos].reset_index(drop=True)
    out["extra_info"] = [dict(e, index=i) for i, e in enumerate(out["extra_info"])]
    ids = [tid[i] for i in pos]
    wrote = False
    if os.path.exists(a.out):
        have = [e["task_id"] for e in pd.read_parquet(a.out)["extra_info"]]
        if have != ids:
            sys.exit(f"subsample_grpo_data: {a.out} exists and holds other rows than this rule selects; not replaced")
        print(f"{a.out} exists and holds exactly the {len(ids)} rows this rule selects: kept")
    else:
        part = f"{a.out}.part{os.getpid()}"
        out.to_parquet(part)
        os.replace(part, a.out)
        wrote = True
    if wrote or not os.path.exists(a.manifest):
        sizes = collections.Counter(tpl)
        part = f"{a.manifest}.part{os.getpid()}"
        with open(part, "w") as f:
            json.dump({"args": vars(a), "input_sha256": sha256(a.inp), "output_sha256": sha256(a.out),
                       "rule": "strata = extra_info.template; proportional allocation n * size / total, largest-"
                               "remainder rounding (ties: template name ascending); random.Random(seed).sample within "
                               "each template from its rows sorted by task_id, templates in sorted order; input row "
                               "order kept, extra_info.index renumbered",
                       "rows_in": len(df), "rows_out": len(out),
                       "by_template": {t: {"in": sizes[t], "quota": round(quota[t], 4), "out": alloc[t]}
                                       for t in sorted(sizes)},
                       "task_ids": ids}, f, indent=1)
        os.replace(part, a.manifest)
    print(f"{len(df)} rows -> {len(out)} rows -> {a.out}; per template {dict(sorted(alloc.items()))}; manifest {a.manifest}")
    print(f"TRAIN={a.out}")


if __name__ == "__main__":
    main()
