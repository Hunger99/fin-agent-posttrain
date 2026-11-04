"""Convert the downloaded evaluation sets and SFT sources into the unified JSONL of docs/history/01_data-processing.md section 4.

Evaluation items keep question, choices and answer, so decontamination and scoring read one format.
SFT samples become ``messages`` with the reasoning moved out of ``<think>`` into ``reasoning``.
CFLUE and DianJin-R1-Data carry no item ids, so ids are row indices, stable only at the pinned commits
recorded in MANIFEST.md.

Usage: python convert.py $FP_DATA_ZH
"""
import ast
import csv
import glob
import io
import json
import os
import re
import sys
import zipfile

import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = sys.argv[1]
RAW = os.path.join(ROOT, "raw")
# SFT sequence length in docs/history/02_model-training.md section 2; longer samples are dropped, not
# truncated, because truncation would keep reasoning without its answer (01 section 9, item 5).
MAX_TOKENS = 4096
THINK = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.S)
tok = Tokenizer.from_file(os.path.join(RAW, "qwen2.5-1.5b-tokenizer", "tokenizer.json"))


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


funnel = []  # (source, raw count, kept count, note)

for split in ("test", "dev"):
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, "financeiq", "data", split, "*.csv"))):
        exam = os.path.splitext(os.path.basename(path))[0]
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({"id": f"financeiq-{split}-{exam}-{r['']}", "source": "financeiq", "split": split,
                             "exam": exam, "task": "单项选择题", "question": r["Question"],
                             "choices": {k: r[k] for k in "ABCD"}, "answer": r["Answer"]})
    write_jsonl(os.path.join(ROOT, "processed", "eval", f"financeiq_{split}.jsonl"), rows)
    funnel.append((f"FinanceIQ {split}", len(rows), len(rows), "评测集，不过滤"))

for split in ("train", "val", "test"):
    with open(os.path.join(RAW, "cflue", f"{split}.json"), encoding="utf-8") as f:
        items = json.load(f)
    rows = []
    for i, r in enumerate(items):
        # choices is stored as a Python dict literal in a string, e.g. "{'A': '1980万元', ...}", not as JSON
        c = r["choices"]
        choices = ast.literal_eval(c) if isinstance(c, str) and c.strip() else (c or {})
        rows.append({"id": f"cflue-{split}-{i}", "source": "cflue", "split": split, "exam": r["名称"],
                     "subject": r["科目"], "task": r["task"], "question": r["question"], "choices": choices,
                     "answer": r["answer"], "analysis": r["analysis"]})
    # The train split is training data (seed questions and the reserved preference pool), so it stays
    # out of processed/eval/; its 80/20 split happens when the pools are built.
    out = (os.path.join(ROOT, "interim", "cflue", "train.jsonl") if split == "train"
           else os.path.join(ROOT, "processed", "eval", f"cflue_{split}.jsonl"))
    write_jsonl(out, rows)
    funnel.append((f"CFLUE {split}", len(rows), len(rows), "不过滤"))

for part in ("mcq", "oe"):
    with open(os.path.join(RAW, "dianjin-r1-data", "train", f"cflue_{part}.json"), encoding="utf-8") as f:
        items = json.load(f)
    parsed = []
    for r in items:
        m = THINK.match(r["output"])
        # A sample without a <think> block keeps its whole output as the answer with empty reasoning,
        # so it shows up as has_reasoning=False instead of disappearing from the counts.
        reasoning, answer = (m.group(1).strip(), m.group(2).strip()) if m else ("", r["output"].strip())
        parsed.append((r["instruction"].strip(), reasoning, answer))
    lengths = [len(e.ids) for e in tok.encode_batch([q + r + a for q, r, a in parsed], add_special_tokens=False)]
    rows, no_think = [], 0
    for i, ((q, reasoning, answer), n) in enumerate(zip(parsed, lengths)):
        no_think += not reasoning
        if n > MAX_TOKENS:
            continue
        rows.append({"id": f"dianjin-r1-{part}-{i}", "source": f"dianjin-r1-data/cflue_{part}",
                     "messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": answer, "reasoning": reasoning}],
                     "meta": {"license": "MIT", "original_index": i, "has_reasoning": bool(reasoning),
                              "n_tokens": n}})
    write_jsonl(os.path.join(ROOT, "processed", "sft", f"dianjin_r1_{part}.jsonl"), rows)
    funnel.append((f"DianJin-R1-Data cflue_{part}", len(items), len(rows),
                   f"丢弃超过 {MAX_TOKENS} token 的 {len(items) - len(rows)} 条；没有 <think> 的 {no_think} 条"))

# Second round (research/log/2026-10-25-0039.md): the remaining SFT sources, the preference supplement
# and the general evaluation sets.
ROLE = {"human": "user", "HUMAN": "user", "user": "user", "gpt": "assistant", "ASSISTANT": "assistant",
        "assistant": "assistant", "system": "system", "SYSTEM": "system"}
HAN = re.compile(r"[一-鿿]")


def mostly_chinese(text):
    # The same 30% share of Han characters that section 2 applies to corpus documents.
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and len(HAN.findall(text)) / len(chars) >= 0.3


def sft_row(rid, source, license_tag, turns, **meta):
    msgs = []
    for role, content in turns:
        m = THINK.match(content) if ROLE[role] == "assistant" else None
        msgs.append({"role": ROLE[role], "content": m.group(2).strip(), "reasoning": m.group(1).strip()} if m
                    else {"role": ROLE[role], "content": content.strip()})
    return {"id": rid, "source": source, "messages": msgs,
            "meta": {"license": license_tag, "has_reasoning": any(x.get("reasoning") for x in msgs), **meta}}


def write_sft(name, label, rows, chinese_only):
    raw = len(rows)
    if chinese_only:
        rows = [r for r in rows if mostly_chinese(next(m["content"] for m in r["messages"] if m["role"] == "user"))]
    non_zh = raw - len(rows)
    kept = []
    for i in range(0, len(rows), 20000):
        batch = rows[i:i + 20000]
        texts = [" ".join(m["content"] + m.get("reasoning", "") for m in r["messages"]) for r in batch]
        for r, e in zip(batch, tok.encode_batch(texts, add_special_tokens=False)):
            if len(e.ids) <= MAX_TOKENS:
                r["meta"]["n_tokens"] = len(e.ids)
                kept.append(r)
    write_jsonl(os.path.join(ROOT, "processed", "sft", f"{name}.jsonl"), kept)
    note = f"丢弃超过 {MAX_TOKENS} token 的 {raw - non_zh - len(kept)} 条"
    funnel.append((label, raw, len(kept), note + (f"；非中文 {non_zh} 条" if chinese_only else "")))


def read_jsonl_file(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


rows = [sft_row(f"agentar-{r['id']}", "agentar-deepfinance-100k", "Apache-2.0（GitHub README 徽章，Hub 无标签）",
                [(m["role"], m["content"]) for m in r["messages"]])
        for r in read_jsonl_file(os.path.join(RAW, "agentar-deepfinance-100k", "data", "Agentar_DeepFinance_01.jsonl"))]
write_sft("agentar_deepfinance", "Agentar-DeepFinance-100K", rows, chinese_only=True)

rows = []
for path in sorted(glob.glob(os.path.join(RAW, "industryinstruction-finance", "*.jsonl"))):
    part = "eval" if path.endswith("_eval.jsonl") else "train"
    for i, r in enumerate(read_jsonl_file(path)):
        # The id field names the generation task, not the item, so the row index is the id.
        rows.append(sft_row(f"industryinstruction-{part}-{i}", "industryinstruction-finance", "Apache-2.0",
                            [(t["from"], t["value"]) for t in r["conversations"]],
                            lang=r.get("lang"), deita_score=r.get("deita_score"), rw_score=r.get("rw_score")))
write_sft("industryinstruction_finance", "IndustryInstruction 金融", rows, chinese_only=True)

with open(os.path.join(RAW, "alpaca-zh", "alpaca_gpt4_data_zh.json"), encoding="utf-8") as f:
    rows = [sft_row(f"alpaca-zh-{i}", "alpaca-zh", "CC BY 4.0",
                    [("user", r["instruction"] + ("\n" + r["input"] if r.get("input") else "")), ("assistant", r["output"])])
            for i, r in enumerate(json.load(f))]
write_sft("alpaca_zh", "alpaca-zh", rows, chinese_only=False)

rows = []
for path in sorted(glob.glob(os.path.join(RAW, "smoltalk-chinese", "*.parquet"))):
    part = os.path.basename(path)[:-8]
    for i, conv in enumerate(pq.read_table(path, columns=["conversations"]).column(0).to_pylist()):
        rows.append(sft_row(f"smoltalk-{part}-{i}", "smoltalk-chinese", "Apache-2.0 + OpenCSG 社区许可协议（未读）",
                            [(t["role"], t["content"]) for t in conv]))
write_sft("smoltalk_chinese", "smoltalk-chinese", rows, chinese_only=False)

rows = []
for path in sorted(glob.glob(os.path.join(RAW, "tulu-3-sft-mixture", "data", "*.parquet"))):
    for r in pq.read_table(path, columns=["id", "messages", "source"]).to_pylist():
        rows.append(sft_row(f"tulu3-{r['id']}", "tulu-3-sft-mixture", "ODC-BY",
                            [(m["role"], m["content"]) for m in r["messages"]], origin=r["source"]))
# Tülu 3 is mostly English and stays so: the general score includes GSM8K and IFEval, which are English.
write_sft("tulu3", "Tülu 3 SFT", rows, chinese_only=False)

prm = pq.read_table(os.path.join(RAW, "dianjin-fin-prm-data", "train.parquet")).to_pylist()
rows = [{"id": f"dianjin-prm-{i}", "source": "dianjin-fin-prm-data", "task": r["task"], "exam": r["名称"],
         "question": r["question"], "choices": ast.literal_eval(r["choices"]) if r["choices"] else {},
         "answer": r["answer"], "trace": r["trace"], "final_answer": r["final_answer"],
         "trajectory_label": r["trajectory_label"]} for i, r in enumerate(prm)]
write_jsonl(os.path.join(ROOT, "interim", "prm", "dianjin_fin_prm.jsonl"), rows)
funnel.append(("DianJin-Fin-PRM-Data", len(rows), len(rows), "偏好补充，不进 SFT，不过滤"))

rows = []
with zipfile.ZipFile(os.path.join(RAW, "cmmlu", "cmmlu_v1_0_1.zip")) as z:
    for name in sorted(n for n in z.namelist() if "/test/" in f"/{n}" and n.endswith(".csv")):
        subject = os.path.basename(name)[:-4]
        for r in csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")):
            rows.append({"id": f"cmmlu-test-{subject}-{r['']}", "source": "cmmlu", "split": "test", "exam": subject,
                         "task": "单项选择题", "question": r["Question"], "choices": {k: r[k] for k in "ABCD"},
                         "answer": r["Answer"]})
write_jsonl(os.path.join(ROOT, "processed", "eval", "cmmlu_test.jsonl"), rows)
funnel.append(("CMMLU test（全部科目）", len(rows), len(rows), "评测参照集"))

gsm = pq.read_table(os.path.join(RAW, "gsm8k", "main", "test-00000-of-00001.parquet")).to_pylist()
rows = [{"id": f"gsm8k-test-{i}", "source": "gsm8k", "split": "test", "task": "数学应用题", "question": r["question"],
         "choices": {}, "answer": r["answer"]} for i, r in enumerate(gsm)]
write_jsonl(os.path.join(ROOT, "processed", "eval", "gsm8k_test.jsonl"), rows)
funnel.append(("GSM8K test", len(rows), len(rows), "评测参照集"))

rows = [{"id": f"ifeval-{r['key']}", "source": "ifeval", "split": "test", "task": "指令遵循", "question": r["prompt"],
         "choices": {}, "answer": ""} for r in read_jsonl_file(os.path.join(RAW, "ifeval", "ifeval_input_data.jsonl"))]
write_jsonl(os.path.join(ROOT, "processed", "eval", "ifeval.jsonl"), rows)
funnel.append(("IFEval", len(rows), len(rows), "评测参照集"))

os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
with open(os.path.join(ROOT, "reports", "funnel.md"), "w", encoding="utf-8") as f:
    f.write("# 条数漏斗（第一步：转格式）\n\n由 src/01_data-processing/utils/convert.py 生成；token 数用 Qwen2.5-1.5B 的分词器计算，"
            "不含对话模板。\n\n| 来源 | 原始条数 | 保留条数 | 说明 |\n| --- | --- | --- | --- |\n")
    for name, raw, kept, note in funnel:
        f.write(f"| {name} | {raw:,} | {kept:,} | {note} |\n")
for row in funnel:
    print(*row, sep="\t")
