"""Turn lm-evaluation-harness per-sample logs into the per-item lines paired_compare.py reads.

docs/history/04_evaluation.md section 2: GSM8K is scored by the flexible-extract filter's exact match and IFEval by
prompt_level_strict_acc. Ids follow processed/eval: gsm8k-test-<row> and ifeval-<key>, so items pair across
models whatever order lm-eval wrote them in.

Output: one line per item (id, correct) and a summary on stdout.

Usage: python lmeval_items.py <samples_*.jsonl> <out.jsonl>
"""
import json
import os
import sys

src, out = sys.argv[1], sys.argv[2]
task = os.path.basename(src).split("_")[1]
FILTER, METRIC = {"gsm8k": ("flexible-extract", "exact_match"), "ifeval": ("none", "prompt_level_strict_acc")}[task]
rows = []
with open(src, encoding="utf-8") as f:
    for r in map(json.loads, f):
        if r["filter"] != FILTER:
            continue
        rid = f"gsm8k-test-{r['doc_id']}" if task == "gsm8k" else f"ifeval-{r['doc']['key']}"
        rows.append({"id": rid, "correct": bool(r[METRIC])})
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(json.dumps({"task": task, "items": len(rows), "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4)}))
