"""Draw the E4 human check of document QA: a sample stratified by question type, as a sheet to fill in.

docs/history/01_data-processing.md section 6 asks for a check stratified by question type, judged against the
source by the author or someone who knows finance; docs/history/decisions.md items 6 and 7 add two counts, a single
company's practice written as a general rule and a rule that may be out of date. The sample size and the
pass threshold are the author's to set (section 9 item 1 suggests 100); N is an argument for that reason.
Items are drawn from docqa_to_sft.py's output, so only questions that would enter the domain pool are
checked, and each type gets its share of N in proportion to its count, by a seeded shuffle.

Output: a CSV with one row per item (id, type, question, answer, evidence sentence, source chunk) and empty
columns for the reviewer.

Usage: python e4_sample.py <doc_qa.jsonl from docqa_to_sft.py> <doc_qa output.jsonl>... --n N --out <sheet.csv>
"""
import argparse
import collections
import csv
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("sft")
ap.add_argument("raw", nargs="+")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=20261029)
ap.add_argument("--out", required=True)
args = ap.parse_args()

with open(args.sft, encoding="utf-8") as f:
    items = [json.loads(line) for line in f]
chunks = {}
for path in args.raw:
    with open(path, encoding="utf-8") as f:
        chunks.update((r["chunk_id"], r["chunk"]) for r in map(json.loads, f))
by_type = collections.defaultdict(list)
for it in items:
    by_type[it["meta"]["type"]].append(it)
rng = random.Random(args.seed)
# Largest remainder, so the shares add up to exactly N (rounding each type gave 101 of 100)
share = {t: args.n * len(v) / len(items) for t, v in by_type.items()}
quota = {t: int(x) for t, x in share.items()}
for t in sorted(share, key=lambda t: share[t] - quota[t], reverse=True)[:args.n - sum(quota.values())]:
    quota[t] += 1
picked = []
for t in sorted(by_type):
    pool = sorted(by_type[t], key=lambda it: it["id"])
    rng.shuffle(pool)
    picked += pool[:quota[t]]
rng.shuffle(picked)
with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["序号", "id", "题型", "题目", "答案", "证据句", "原文块",
                "答案对错（对/错）", "单个公司做法写成通用规则（是/否）", "规定可能已过时（是/否）", "备注"])
    for k, it in enumerate(picked, 1):
        w.writerow([k, it["id"], it["meta"]["type"], it["messages"][0]["content"], it["messages"][1]["content"],
                    it["meta"]["evidence"], chunks.get(it["meta"]["chunk_id"], ""), "", "", "", ""])
print(f"{len(picked)} items from {len(items):,}; per type: " + ", ".join(f"{t} {quota[t]}" for t in sorted(quota)))
