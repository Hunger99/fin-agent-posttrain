"""Convert the English SFT sources into the unified JSONL of docs/history/01_data-processing.md section 4.

research/log/2026-10-26-2020.md. The same row format as convert.py (id, source, messages with reasoning moved
out of <think>, meta with license and token count; samples over 4,096 tokens dropped), with English in place
of Chinese: IndustryInstruction and Agentar keep only their English samples, which convert.py dropped.

FinQA and TAT-QA train items become SFT samples with the gold calculation as the answer, in the prompt format
of num_eval.py, and they are split 80/20 by report page (FinQA) or by table (TAT-QA) with a fixed salted hash:
the 20% is held for preference pairs and goes to interim/seed/, never into SFT, as the CFLUE train split was
(docs/history/01_data-processing.md section 5). Splitting by page keeps the two parts from sharing an excerpt.

Sujet-Finance-Instruct-177k drops the rows it took from other benchmarks' evaluation sets (AdaptLLM's
finance-tasks, whose configurations are test splits only, and lighthouzai/finqabench), so no benchmark's test
items are trained on.

Usage: python convert_en.py <English data root>
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = sys.argv[1]
RAW = os.path.join(ROOT, "raw")
MAX_TOKENS = 4096
SALT = "seed-split-20261026"
THINK = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.S)
HAN = re.compile(r"[一-鿿]")
ROLE = {"user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant", "system": "system"}
tok = Tokenizer.from_file(os.path.join(RAW, "qwen2.5-1.5b-tokenizer", "tokenizer.json"))
INSTRUCTION = ("Answer the question from the excerpt of a company's annual report. Write the calculation, then the "
               "final answer as a number on the last line, after \"Answer:\".")
funnel = []


def english(text):
    # No Han character at all and mostly Latin letters among the letters: the mirror of convert.py's 30% Han rule.
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and not HAN.search(text) and sum(c.isascii() for c in letters) / len(letters) >= 0.9


def sft_row(rid, source, license_tag, turns, **meta):
    msgs = []
    for role, content in turns:
        role = ROLE[role.lower()]
        if role == "system" and not content.strip():
            continue
        m = THINK.match(content) if role == "assistant" else None
        msgs.append({"role": role, "content": m.group(2).strip(), "reasoning": m.group(1).strip()} if m
                    else {"role": role, "content": content.strip()})
    return {"id": rid, "source": source, "messages": msgs,
            "meta": {"license": license_tag, "has_reasoning": any(x.get("reasoning") for x in msgs), **meta}}


def write(rel, rows):
    os.makedirs(os.path.dirname(os.path.join(ROOT, rel)), exist_ok=True)
    with open(os.path.join(ROOT, rel), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_sft(name, label, rows, english_only=False, note=""):
    raw = len(rows)
    if english_only:
        rows = [r for r in rows if english(" ".join(m["content"] for m in r["messages"]))]
    non_en = raw - len(rows)
    kept = []
    for i in range(0, len(rows), 20000):
        batch = rows[i:i + 20000]
        texts = [" ".join(m["content"] + m.get("reasoning", "") for m in r["messages"]) for r in batch]
        for r, e in zip(batch, tok.encode_batch(texts, add_special_tokens=False)):
            if len(e.ids) <= MAX_TOKENS and all(m["content"] for m in r["messages"] if m["role"] == "assistant"):
                r["meta"]["n_tokens"] = len(e.ids)
                kept.append(r)
    write(os.path.join("processed", "sft", f"{name}.jsonl"), kept)
    funnel.append({"source": label, "raw": raw, "kept": len(kept), "non_english": non_en if english_only else None,
                   "dropped_long_or_empty": raw - non_en - len(kept), "note": note})
    print(json.dumps(funnel[-1], ensure_ascii=False), flush=True)


def reserved(key):
    return int(hashlib.sha256((SALT + key).encode()).hexdigest(), 16) % 5 == 0


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# FinQA and TAT-QA train: seed questions; 80% become SFT samples, 20% are held for preference pairs.
for src, key in (("finqa", lambda r: r["id"].rsplit("-", 1)[0]), ("tatqa", None)):
    items = read_jsonl(os.path.join(ROOT, "processed", "eval", f"{src}_train.jsonl"))
    if src == "tatqa":  # the table is the unit: map each question to its context's first question id
        ctx = {}
        for r in items:
            ctx.setdefault(r["context"], r["id"])
        key = lambda r, ctx=ctx: ctx[r["context"]]
    held = [r for r in items if reserved(key(r))]
    write(os.path.join("interim", "seed", f"{src}_train_pref.jsonl"), held)
    rows = []
    for r in items:
        if reserved(key(r)) or not r["calc"]:
            continue
        ans = r["answer"] if isinstance(r["answer"], str) else f"{r['answer']:g}"
        user = f"{INSTRUCTION}\n\nReport:\n{r['context']}\n\nQuestion: {r['question']}"
        rows.append(sft_row(r["id"], f"{src}-train80", "CC BY 4.0" if src == "tatqa" else "MIT",
                            [("user", user), ("assistant", f"Calculation: {r['calc']}\nAnswer: {ans}")]))
    write_sft(f"{src}_train80", f"{src} train 80%", rows,
              note=f"20% held for preference pairs: {len(held)} items; items without a written-out calculation dropped")

rows = []
for i, r in enumerate(json.load(open(os.path.join(RAW, "dianjin-r1-finqa", "train", "fin_qa.json"), encoding="utf-8"))):
    rows.append(sft_row(f"dianjin-finqa-{i}", "dianjin-r1-finqa", "MIT", [("user", r["instruction"].strip()), ("assistant", r["output"])]))
write_sft("dianjin_r1_finqa", "DianJin-R1-Data fin_qa", rows)

rows = []
for split in ("train",):
    path = glob.glob(os.path.join(RAW, "bizbench", "data", f"{split}-*.parquet"))[0]
    for i, r in enumerate(pq.read_table(path).to_pylist()):
        q = (f"{r['context']}\n\n" if r.get("context") else "") + r["question"]
        a = (f"```python\n{r['program'].strip()}\n```\n" if r.get("program") else "") + f"Answer: {r['answer']}"
        rows.append(sft_row(f"bizbench-{split}-{i}", "bizbench", "Apache-2.0", [("user", q), ("assistant", a)], task=r["task"]))
write_sft("bizbench", "BizBench train", rows)

rows = [sft_row(f"agentar-{r['id']}", "agentar-deepfinance-100k", "Apache-2.0（GitHub README 徽章，Hub 无标签）",
                [(m["role"], m["content"]) for m in r["messages"]])
        for r in read_jsonl(os.path.join(RAW, "agentar-deepfinance-100k", "data", "Agentar_DeepFinance_01.jsonl"))]
write_sft("agentar_deepfinance_en", "Agentar-DeepFinance-100K (English)", rows, english_only=True)

rows = []
for path in sorted(glob.glob(os.path.join(RAW, "industryinstruction-finance", "*.jsonl"))):
    part = "eval" if path.endswith("_eval.jsonl") else "train"
    for i, r in enumerate(read_jsonl(path)):
        rows.append(sft_row(f"industryinstruction-{part}-{i}", "industryinstruction-finance", "Apache-2.0",
                            [(t["from"], t["value"]) for t in r["conversations"]],
                            lang=r.get("lang"), deita_score=r.get("deita_score"), rw_score=r.get("rw_score")))
write_sft("industryinstruction_finance_en", "IndustryInstruction 金融 (English)", rows, english_only=True)

csv.field_size_limit(sys.maxsize)
rows, skipped = [], 0
with open(os.path.join(RAW, "sujet-finance-instruct-177k", "sujet-ai", "Sujet-Finance-Instruct-177k.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["dataset"].startswith("AdaptLLM/finance-tasks") or r["dataset"] == "lighthouzai/finqabench":
            skipped += 1
            continue
        turns = ([("system", r["system_prompt"])] if r["system_prompt"].strip() else []) + \
                [("user", r["user_prompt"]), ("assistant", r["answer"])]
        rows.append(sft_row(f"sujet-{r['']}", "sujet-finance-instruct-177k", "Apache-2.0", turns, origin=r["dataset"],
                            task=r["task_type"]))
write_sft("sujet_finance", "Sujet-Finance-Instruct-177k", rows, english_only=True,
          note=f"rows from other benchmarks' evaluation sets dropped: {skipped}")

rows = [sft_row(f"finance-alpaca-{i}", "finance-alpaca", "MIT",
                [("user", r["instruction"] + ("\n" + r["input"] if r.get("input") else "")), ("assistant", r["output"])])
        for i, r in enumerate(json.load(open(os.path.join(RAW, "finance-alpaca", "Cleaned_date.json"), encoding="utf-8")))]
write_sft("finance_alpaca", "finance-alpaca", rows, english_only=True)

rows = []
for i, r in enumerate(read_jsonl(os.path.join(RAW, "finance-instruct-500k", "train.json"))):
    turns = ([("system", r["system"])] if (r.get("system") or "").strip() else []) + [("user", r["user"]), ("assistant", r["assistant"])]
    rows.append(sft_row(f"finance-instruct-500k-{i}", "finance-instruct-500k", "Apache-2.0", turns))
write_sft("finance_instruct_500k", "Finance-Instruct-500k", rows, english_only=True)

write(os.path.join("reports", "funnel_convert_en.jsonl"), funnel)
