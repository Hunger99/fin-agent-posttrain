"""Draw the English E4 human check of document QA: a sample stratified by the kind of calculation, as a sheet.

research/log/2026-10-26-2222.md and the E4 standard in docs/results.md (Evaluation Metrics): 100 items, each
judged against its document chunk by the author or someone who knows finance. Items are drawn from the SFT
samples that would enter the domain pool (docqa_to_sft_en.py's output after decontamination and dedup), and each
kind of calculation gets its share of N in proportion to its count, by a seeded shuffle. The kind is read from the
question's wording (KINDS, first match), only to spread the sample; it is not a judgment of the item.

Output: a CSV with one row per item (id, kind, question, calculation, answer, document chunk) and empty columns
for the reviewer. The 10 control items are added by e4_add_controls.py's procedure, separately.

Usage: python e4_sample_en.py <doc_qa_en.jsonl> --exclude <ids file> --n 100 --out <sheet.csv>
"""
import argparse
import collections
import csv
import json
import random
import re

KINDS = [("变化率", r"percent(age)? change|growth rate|grew by|increase(d)? by .*percent|decrease(d)? by .*percent|cagr"),
         ("比率", r"ratio|margin|as a percent(age)? of|proportion|share of|percent(age)? of|per share|per unit|yield"),
         ("差额", r"difference|change in|increase in|decrease in|how much (more|less)"),
         ("合计", r"total|sum|combined|altogether"),
         ("平均", r"average|mean"),
         ("其他", r".")]
ap = argparse.ArgumentParser()
ap.add_argument("sft")
ap.add_argument("--exclude", default=None, help="file of ids removed by decontamination or dedup, one per line")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=20261027)
ap.add_argument("--out", required=True)
args = ap.parse_args()

drop = set(open(args.exclude).read().split()) if args.exclude else set()
items = [r for r in map(json.loads, open(args.sft, encoding="utf-8")) if r["id"] not in drop]
by_kind = collections.defaultdict(list)
for r in items:
    q = r["messages"][0]["content"].rsplit("\n\nQuestion: ", 1)[1]
    kind = next(k for k, pat in KINDS if re.search(pat, q.lower()))
    by_kind[kind].append((r, q))
rng = random.Random(args.seed)
total = sum(len(v) for v in by_kind.values())
quota = {k: round(args.n * len(v) / total) for k, v in by_kind.items()}
while sum(quota.values()) != args.n:  # rounding: adjust the largest kind
    k = max(by_kind, key=lambda k: len(by_kind[k]))
    quota[k] += 1 if sum(quota.values()) < args.n else -1
rows = []
for k in sorted(by_kind):
    pool = by_kind[k][:]
    rng.shuffle(pool)
    rows += [(k, r, q) for r, q in pool[:quota[k]]]
rng.shuffle(rows)
with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["序号", "id", "计算类型", "题目", "计算过程", "答案", "原文块", "答案对错（对/错）", "题目确实要从原文计算（是/否）", "备注"])
    for i, (k, r, q) in enumerate(rows, 1):
        calc, ans = r["messages"][1]["content"].rsplit("\nAnswer: ", 1)
        chunk = r["messages"][0]["content"].split("\n\nDocument:\n", 1)[1].rsplit("\n\nQuestion: ", 1)[0]
        w.writerow([i, r["id"], k, q, calc.replace("Calculation: ", "", 1), ans, chunk, "", "", ""])
print(json.dumps({"items": total, "by_kind": {k: len(v) for k, v in by_kind.items()}, "quota": quota}, ensure_ascii=False))
