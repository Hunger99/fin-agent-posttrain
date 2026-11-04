"""Compare two models' per-item results from mc_eval.py by the paired bootstrap of goal file Metrics section 6.

Items are paired by id; the difference is model B's accuracy minus model A's. The one-sided p value is the
share of resamples whose difference is <= 0 (Metrics section 6); the 95% interval is the 2.5th to 97.5th
percentile of the resampled differences. B resamples, fixed seed (docs/history/04_evaluation.md section 3).

Usage: python paired_compare.py <A.jsonl> <B.jsonl> [--resamples 10000] [--seed 20261025]
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--resamples", type=int, default=10000)
ap.add_argument("--seed", type=int, default=20261025)
args = ap.parse_args()


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["correct"] for r in map(json.loads, f)}


a, b = load(args.a), load(args.b)
ids = sorted(set(a) & set(b))
d = np.array([b[i] for i in ids], dtype=float) - np.array([a[i] for i in ids], dtype=float)
boot = d[np.random.default_rng(args.seed).integers(0, len(ids), size=(args.resamples, len(ids)))].mean(1)
print(json.dumps({"items": len(ids), "acc_a": round(float(np.mean([a[i] for i in ids])), 4),
                  "acc_b": round(float(np.mean([b[i] for i in ids])), 4), "diff": round(float(d.mean()), 4),
                  "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                  "p_one_sided": float((boot <= 0).mean())}))
