"""Choose the starting model by docs/history/02_model-training.md section 1's rule, from mc_eval.py outputs.

Each group is a starting-model candidate and its reference model; the group whose gap (reference accuracy
minus candidate accuracy) on the same items is larger supplies the starting model. Gaps, and the difference
between the two groups' gaps, get 95% intervals from a paired bootstrap over items (B resamples, fixed seed),
as in goal file Metrics section 6.

Usage: python select_start_model.py <report.md> <name>=<candidate.jsonl>,<reference.jsonl> <name>=<...>
"""
import json
import sys

import numpy as np

REPORT, GROUPS = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
B, SEED = 10000, 20261025


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["correct"] for r in map(json.loads, f)}


runs = {name: [load(p) for p in pair.split(",")] for name, pair in GROUPS}
ids = sorted(set.intersection(*(set(d) for pair in runs.values() for d in pair)))
mat = {name: np.array([[d[i] for i in ids] for d in pair], dtype=float) for name, pair in runs.items()}
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(ids), size=(B, len(ids)))
gap = {name: m[1] - m[0] for name, m in mat.items()}


def ci(x):
    boot = x[idx].mean(1)
    return x.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


lines = ["# 起点模型的选择", "", f"由 src/04_evaluation/utils/select_start_model.py 生成。{len(ids):,} 道题，四个模型都答过；区间为配对 bootstrap"
         f"（{B:,} 次，种子 {SEED}）的 95% 区间，单位为百分点。", "",
         "| 组 | 起点候选正确率 | 参照模型正确率 | 差距（参照 − 候选） |", "| --- | --- | --- | --- |"]
for name, m in mat.items():
    g, lo, hi = ci(gap[name])
    lines.append(f"| {name} | {m[0].mean():.1%} | {m[1].mean():.1%} | {g * 100:.1f}（{lo * 100:.1f} 到 {hi * 100:.1f}） |")
(n1, g1), (n2, g2) = gap.items()
d, lo, hi = ci(g1 - g2)
best = n1 if d > 0 else n2
lines += ["", f"两组差距之差（{n1} − {n2}）：{d * 100:.1f} 个百分点（{lo * 100:.1f} 到 {hi * 100:.1f}）。按规则选差距更大的一组：{best}。"]
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
