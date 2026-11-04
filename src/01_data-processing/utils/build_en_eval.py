"""Write the English evaluation sets: FinQA and TAT-QA numeric questions, each with its report excerpt.

The English domain score (research/log/2026-10-26-2006.md) is accuracy on numeric questions answered from a
given excerpt of a company's annual report: all of FinQA, whose gold answer is the executed program's result
(`exe_ans`, a number or yes/no), and TAT-QA's arithmetic and count questions, whose answer is one number in the
question's scale. TAT-QA's span answers are left out, because they are scored by token F1, not by one rule
shared with FinQA. Test splits give the domain score, dev splits are the selection sets, and train splits
supply the few-shot examples and, later, seed questions.

Each row: id, source, split, context (pre-text, the table one row per line with " | " between cells, post-text),
question, answer (float, or "yes"/"no"), scale, and calc, the gold calculation written out for use as an
example: FinQA's program with each step's value, TAT-QA's derivation.

XFinBench (the knowledge cross-check) and MMLU (the English general knowledge set) are written as well; both
are also decontamination references.

Usage: python build_en_eval.py <English data root>
"""
import csv
import json
import math
import os
import sys

import pyarrow.parquet as pq

ROOT = sys.argv[1]
RAW = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "processed", "eval")
os.makedirs(OUT, exist_ok=True)
OPS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}


def table_text(rows):
    return "\n".join(" | ".join(str(c).strip() for c in row) for row in rows)


def num(s):
    s = str(s).replace(",", "").replace("$", "").replace("%", "").strip()
    if s.startswith("const_"):
        s = s[6:].replace("m1", "-1")
    return float(s)


def fmt(x):
    if isinstance(x, str):
        return x
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.5g}" if abs(x) < 1 else f"{round(x, 4):g}"


def finqa_calc(program):
    """Write a FinQA program as arithmetic with each step's value, or None when it uses a table or compare op."""
    steps, vals = [], []
    for part in program.split("),"):
        part = part.strip().rstrip(")")
        if not part:
            continue
        op, args = part.split("(", 1)
        op = op.strip()
        if op not in OPS:
            return None
        a, b = [x.strip() for x in args.split(",")]
        try:
            x = vals[int(a[1:])] if a.startswith("#") else num(a)
            y = vals[int(b[1:])] if b.startswith("#") else num(b)
        except (ValueError, IndexError):
            return None
        if op == "divide" and y == 0:
            return None
        v = {"add": x + y, "subtract": x - y, "multiply": x * y, "divide": x / y if y else math.nan}[op]
        vals.append(v)
        steps.append(f"{fmt(x)} {OPS[op]} {fmt(y)} = {fmt(v)}")
    return "; ".join(steps) if steps else None


def write(name, rows):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"file": name, "items": len(rows)}))


for split in ("train", "dev", "test"):
    rows = []
    for r in json.load(open(os.path.join(RAW, "finqa", "dataset", f"{split}.json"), encoding="utf-8")):
        qa = r["qa"]
        ans = qa["exe_ans"]
        if isinstance(ans, str):
            ans = ans.strip().lower()
            if ans not in ("yes", "no"):
                continue
        context = "\n".join([" ".join(r["pre_text"]), table_text(r["table"]), " ".join(r["post_text"])])
        rows.append({"id": f"finqa-{split}-{r['id']}", "source": "finqa", "split": split, "context": context,
                     "question": qa["question"].strip(), "answer": ans, "scale": "",
                     "calc": finqa_calc(qa["program"]) if not isinstance(ans, str) else None})
    write(f"finqa_{split}.jsonl", rows)

for split, fname in (("train", "train"), ("dev", "dev"), ("test", "test_gold")):
    rows = []
    for c in json.load(open(os.path.join(RAW, "tat-qa", f"tatqa_dataset_{fname}.json"), encoding="utf-8")):
        paras = "\n".join(p["text"] for p in sorted(c["paragraphs"], key=lambda p: p["order"]))
        context = table_text(c["table"]["table"]) + "\n" + paras
        for q in c["questions"]:
            if q["answer_type"] not in ("arithmetic", "count"):
                continue
            try:
                ans = num(q["answer"][0] if isinstance(q["answer"], list) else q["answer"])
            except ValueError:
                continue
            calc = q["derivation"].strip()
            rows.append({"id": f"tatqa-{split}-{q['uid']}", "source": "tatqa", "split": split, "context": context,
                         "question": q["question"].strip(), "answer": ans, "scale": q["scale"],
                         "calc": f"{calc} = {fmt(ans)}" if calc else None})
    write(f"tatqa_{split}.jsonl", rows)

# XFinBench keeps its three tasks (bool, mcq, calcu) and the figure name; an item
# whose figure is not the blank image needs the picture, which a text-only model does not see.
csv.field_size_limit(sys.maxsize)
for split, fname in (("val", "validation_set.csv"), ("test", "test_set.csv")):
    with open(os.path.join(RAW, "xfinbench", fname), encoding="utf-8-sig") as f:
        rows = [{"id": f"xfinbench-{split}-{r['id']}", "source": "xfinbench", "split": split, "task": r["task"],
                 "question": r["question"], "choices": r["choice"], "answer": r["ground_truth"],
                 "figure": r["figure"] if r["figure"] not in ("", "blank_image.png") else None,
                 "capability": r["fin_capability"]} for r in csv.DictReader(f)]
    write(f"xfinbench_{split}.jsonl", rows)

for split in ("dev", "test"):
    rows = []
    for i, r in enumerate(pq.read_table(os.path.join(RAW, "mmlu", "all", f"{split}-00000-of-00001.parquet")).to_pylist()):
        rows.append({"id": f"mmlu-{split}-{r['subject']}-{i}", "source": "mmlu", "split": split, "exam": r["subject"],
                     "task": "multiple_choice", "question": r["question"],
                     "choices": dict(zip("ABCD", r["choices"])), "answer": "ABCD"[r["answer"]]})
    write(f"mmlu_{split}.jsonl", rows)
