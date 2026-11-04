"""Tabulate a score_docs.py output: score distribution and keep share (>= 3) per source.

Usage: python score_report.py <scored.jsonl> <report.md>
"""
import collections
import json
import statistics
import sys

SRC, REPORT = sys.argv[1], sys.argv[2]
dist = collections.defaultdict(collections.Counter)
tokens = []
with open(SRC, encoding="utf-8") as f:
    for r in map(json.loads, f):
        dist[r["source"]][r["score"] if r.get("score") is not None else "无效"] += 1
        dist[r["source"]]["截断"] += r.get("finish") == "length"
        tokens.append(r.get("completion_tokens") or 0)
total = sum(dist.values(), collections.Counter())
lines = ["# 教师打分的分布", "", f"由 src/01_data-processing/utils/score_docs.py 的输出 `{SRC}` 经 score_report.py 生成。"
         "每格为篇数；无效为回答中取不到分数，截断为生成达到长度上限。", "",
         "| 来源 | 篇数 | 0 | 1 | 2 | 3 | 4 | 5 | 无效 | 截断 | ≥3 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for source, c in sorted(dist.items()) + [("合计", total)]:
    n = sum(v for k, v in c.items() if k != "截断")
    keep = sum(c[s] for s in (3, 4, 5))
    lines.append(f"| {source} | {n:,} | " + " | ".join(f"{c[s]:,}" for s in (0, 1, 2, 3, 4, 5, "无效", "截断"))
                 + f" | {keep / max(n, 1):.1%} |")
lines += ["", f"生成长度中位数 {statistics.median(tokens):.0f} token。"]
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
