"""Summarize a doc_qa.py output against the pilot rule of docs/history/decisions.md item 3.

The rule, fixed before the pilot: closed-book continues when self-contained kept questions average at least
0.25 per chunk and at most half of them are a single entity's numbers. A kept question counts as a single
entity's number when the self check labels it 具体事实 and its answer contains an Arabic digit; the
self check's label alone also covers non-numeric facts, so the report gives both shares.

Usage: python doc_qa_report.py <output.jsonl> <report.md>
"""
import collections
import json
import re
import sys

SRC, REPORT = sys.argv[1], sys.argv[2]
STAGES = ("questions", "evidence_ok", "no_deictic", "rules_ok", "self_contained", "kept")
by_source = collections.defaultdict(collections.Counter)
grid = collections.Counter()
consistency = collections.Counter()
with open(SRC, encoding="utf-8") as f:
    for row in map(json.loads, f):
        c = by_source[row["source"]]
        c["chunks"] += 1
        c["chunk_errors"] += "error" in row
        for q in row["questions"]:
            c["questions"] += 1
            c["evidence_ok"] += q["evidence_ok"]
            c["no_deictic"] += q["evidence_ok"] and not q["deictic"]
            # year and language checks exist from v3 on; older outputs count as passing them
            c["rules_ok"] += q["evidence_ok"] and not q["deictic"] and q.get("year_ok", True) and q.get("chinese", True)
            c["self_contained"] += bool(q.get("self_contained"))
            c["question_errors"] += "error" in q
            if q.get("consistent"):
                consistency[q["consistent"]] += 1
            if q["kept"]:
                c["kept"] += 1
                numeric = q.get("category") == "具体事实" and bool(re.search(r"\d", q["answer"]))
                c["entity_numeric"] += numeric
                c["specific"] += q.get("category") == "具体事实"
                grid[(q["type"], q.get("category") or "无标签")] += 1
total = sum(by_source.values(), collections.Counter())
per_chunk = total["kept"] / max(total["chunks"], 1)
numeric_share = total["entity_numeric"] / max(total["kept"], 1)
passed = per_chunk >= 0.25 and numeric_share <= 0.5

lines = ["# 文档问答的产量和题型", "", f"由 src/01_data-processing/utils/doc_qa_report.py 从 `{SRC}` 生成。每列为到这一步为止还留下的题数："
         "证据逐字出现在块中、题目中没有指代词、题目中的年份都出现在块中且以中文为主（v3 起才查）、教师判为自含"
         "、一致性检查通过（即保留；v3 起只对判为通用知识的题做这一步）。", "",
         "| 来源 | 块 | 出题 | 证据通过 | 无指代词 | 年份和语言 | 自含 | 保留 | 每块保留 | 保留中具体事实 | 保留中单个主体的数字 | 出错的块 / 题 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for source, c in sorted(by_source.items()) + [("合计", total)]:
    lines.append(f"| {source} | {c['chunks']:,} | " + " | ".join(f"{c[k]:,}" for k in STAGES)
                 + f" | {c['kept'] / max(c['chunks'], 1):.2f} | {c['specific'] / max(c['kept'], 1):.0%} | "
                 f"{c['entity_numeric'] / max(c['kept'], 1):.0%} | {c['chunk_errors']} / {c['question_errors']} |")
types = sorted({t for t, _ in grid})
cats = sorted({k for _, k in grid})
lines += ["", "保留的题按题型（出题时的标签）和类别（自含检查的标签）：", "", "| 题型 | " + " | ".join(cats) + " | 合计 |",
          "| --- | " + " | ".join("---" for _ in cats) + " | --- |"]
for t in types:
    lines.append(f"| {t} | " + " | ".join(f"{grid[(t, k)]:,}" for k in cats) + f" | {sum(grid[(t, k)] for k in cats):,} |")
lines += ["", "一致性检查（只对自含的题做）：" + "，".join(f"{k} {v:,}" for k, v in sorted(consistency.items())) + "。", "",
          f"试跑规则（裁决第 3 项）：每块保留 {per_chunk:.2f} 道（要求 ≥0.25），单个主体的数字类占保留的 {numeric_share:.0%}"
          f"（要求 ≤50%）。结论：{'继续用闭卷' if passed else '重新裁决'}。"]
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
