"""Flag training samples whose prompt is a rewrite of an evaluation item, by bge-m3 cosine similarity.

The second match of docs/history/01_data-processing.md section 3: 8-grams miss rewritten items, such as the
open-ended questions DianJin-R1-Data rewrote from CFLUE. Each SFT sample's first user turn, or the item
itself for CFLUE-style item sources, is embedded with bge-m3's dense head (the normalized CLS vector,
at most 512 tokens) and compared with every evaluation item; cosine >= THRESHOLD is flagged.

The vectors are saved under ``interim/embed/`` and reused while their texts are unchanged, so a later
threshold change costs only the matrix products (see ``embed_cached``).

The reference and source lists repeat those of decontam.py; both scripts run with ``python -I``, which
keeps this directory off sys.path, so they do not import each other.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python embed_decontam.py $FP_DATA_ZH [domain-only]

``domain-only`` skips the three general sources (about 1.5M samples). On 2026-10-25 every working GPU
on host 1 was busy with another user's jobs, so the domain sources ran on CPU in 2 h 44 min
(research/log/2026-10-25-0039.md); all sources were then rerun on a free GPU on host 2 in 55 min
(research/log/2026-10-25-0610.md). With no usable GPU the script runs on CPU.

Pairs across languages score lower, so 0.90 misses Chinese translations of GSM8K items; xling_decontam.py
flags those with its own rule (research/log/2026-10-25-0730.md).
"""
import collections
import hashlib
import json
import os
import random
import re
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = sys.argv[1]
THRESHOLD = 0.90  # docs/history/01_data-processing.md section 3; lowered from 0.92 after the 100-pair review in research/log/2026-10-25-0039.md
REVIEW = 100  # pairs near the threshold written out for reading, per section 3
EVAL_SETS = [("FinanceIQ test", "processed/eval/financeiq_test.jsonl"),
             ("FinanceIQ dev", "processed/eval/financeiq_dev.jsonl"),
             ("CFLUE val", "processed/eval/cflue_val.jsonl"),
             ("CFLUE test", "processed/eval/cflue_test.jsonl"),
             ("CMMLU test", "processed/eval/cmmlu_test.jsonl"),
             ("GSM8K test", "processed/eval/gsm8k_test.jsonl"),
             ("IFEval", "processed/eval/ifeval.jsonl")]
SOURCES = [("DianJin-R1-Data cflue_mcq", "processed/sft/dianjin_r1_mcq.jsonl", set(), "sft"),
           ("DianJin-R1-Data cflue_oe", "processed/sft/dianjin_r1_oe.jsonl", set(), "sft"),
           ("Agentar-DeepFinance-100K", "processed/sft/agentar_deepfinance.jsonl", set(), "sft"),
           ("IndustryInstruction 金融", "processed/sft/industryinstruction_finance.jsonl", set(), "sft"),
           ("CFLUE train", "interim/cflue/train.jsonl", set(), "item"),
           ("DianJin-Fin-PRM-Data", "interim/prm/dianjin_fin_prm.jsonl", set(), "item"),
           ("alpaca-zh", "processed/sft/alpaca_zh.jsonl", set(), "sft"),
           ("smoltalk-chinese", "processed/sft/smoltalk_chinese.jsonl", set(), "sft"),
           ("Tülu 3 SFT", "processed/sft/tulu3.jsonl", set(), "sft"),
           ("CFLUE val（评测集之间）", "processed/eval/cflue_val.jsonl", {"CFLUE val"}, "item"),
           ("CFLUE test（评测集之间）", "processed/eval/cflue_test.jsonl", {"CFLUE test"}, "item")]
# DianJin-R1-Data wraps every question in fixed instruction lines; left in, they pull every DianJin
# prompt toward every other and away from the bare evaluation item.
WRAPPER = re.compile(r"^(假设你是一位金融行业专家，请回答下列问题。|注意：.*|请一步步思考。)$")


def read_jsonl(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def item_text(it):
    return it["question"] + " " + " ".join(f"{k}. {v}" for k, v in sorted(it["choices"].items()))


def prompt_text(r, kind):
    if kind == "item":
        return item_text(r)
    # First non-empty user turn: 40 Tülu 3 samples open with an empty one (counted 2026-10-25), and an
    # empty text would be compared as if it were a prompt. No sample has no user turn at all.
    user = next((m["content"] for m in r["messages"] if m["role"] == "user" and m["content"].strip()), "")
    return "\n".join(line for line in user.splitlines() if not WRAPPER.match(line.strip())).strip()


GENERAL = {"alpaca-zh", "smoltalk-chinese", "Tülu 3 SFT"}
if len(sys.argv) > 2 and sys.argv[2] == "domain-only":
    SOURCES = [s for s in SOURCES if s[0] not in GENERAL]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CPU float16 matmuls are slow; float32 there. 32 threads leave the other users' CPU jobs room.
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
torch.set_num_threads(32)
tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, "raw", "bge-m3"))
model = AutoModel.from_pretrained(os.path.join(ROOT, "raw", "bge-m3"), torch_dtype=DTYPE).to(DEVICE).eval()


@torch.inference_mode()
def embed(texts, batch=64):
    # 64 was first sized for host 1's busy cards, which had about 5 GB free. On a free A6000 it is
    # also the fastest of 64, 256 and 512: timed on 3,000 random prompts per training source, the
    # nine sources extrapolate to 47, 49 and 56 min, with peaks of 1.8, 3.9 and 6.7 GiB
    # (research/log/2026-10-25-0610.md).
    # Sorting by length keeps padding low; results are put back in input order.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = torch.empty(len(texts), model.config.hidden_size, dtype=DTYPE, device=DEVICE)
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        enc = tokenizer([texts[j] for j in idx], padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(DEVICE)
        cls = model(**enc).last_hidden_state[:, 0]
        out[idx] = torch.nn.functional.normalize(cls.float(), dim=-1).to(DTYPE)
        if i // batch % 200 == 0:
            print(f"  embedded {i + len(idx):,}/{len(texts):,}", flush=True)
    return out


EMBED_DIR = os.path.join(ROOT, "interim", "embed")


def embed_cached(name, texts, meta=None):
    """Vectors of ``texts``, read from ``interim/embed/<name>.npy`` when a run already embedded the same texts.

    The 0.92 run kept its vectors only in memory, so lowering the threshold to 0.90 meant embedding
    everything again (research/log/2026-10-25-0039.md). The array is float16 in input order, i.e. file
    row order; ``<name>.json`` holds the row count, a SHA-256 of the texts and ``meta``. The hash covers
    the texts only, so after changing the model or ``max_length`` delete ``interim/embed/`` by hand.
    """
    digest = hashlib.sha256("\0".join(texts).encode("utf-8")).hexdigest()
    npy, info = os.path.join(EMBED_DIR, name + ".npy"), os.path.join(EMBED_DIR, name + ".json")
    if os.path.exists(info):
        with open(info, encoding="utf-8") as f:
            if json.load(f)["sha256"] == digest:
                print(f"  reusing {npy}", flush=True)
                return torch.from_numpy(np.load(npy)).to(DEVICE, DTYPE)
    emb = embed(texts)
    os.makedirs(EMBED_DIR, exist_ok=True)
    np.save(npy, emb.to(torch.float16).cpu().numpy())
    # Written after the array, so an interrupted save leaves no record that would vouch for it.
    with open(info, "w", encoding="utf-8") as f:
        json.dump({"rows": len(texts), "sha256": digest, **(meta or {})}, f, ensure_ascii=False)
    return emb


ref_ids, ref_sets, ref_texts = [], [], []
for set_name, rel in EVAL_SETS:
    for it in read_jsonl(rel):
        ref_ids.append(it["id"])
        ref_sets.append(set_name)
        ref_texts.append(item_text(it))
ref_emb = embed_cached("ref", ref_texts, {"ids": ref_ids, "sets": ref_sets})
set_of = torch.tensor([[s == name for s in ref_sets] for name, _ in EVAL_SETS], device=DEVICE)  # set x ref

results, near = {}, []
# Similarities are taken in float32. In float16, 0.90 is stored as 0.89990234, so the first GPU run in
# research/log/2026-10-25-0610.md also flagged samples just under the threshold (13 more in DianJin-R1-Data
# cflue_mcq than float32 gives), and its flag counts did not match its own similarity bins.
ref32 = ref_emb.float()
os.makedirs(os.path.join(ROOT, "interim", "decontam"), exist_ok=True)
for src, rel, skip, kind in SOURCES:
    rows = read_jsonl(rel)
    texts = [prompt_text(r, kind) for r in rows]
    emb = embed_cached(os.path.basename(rel)[:-6], texts)
    allowed = ~set_of[[i for i, (name, _) in enumerate(EVAL_SETS) if name in skip]].any(0) if skip else None
    flagged, best = [], []
    for i in range(0, len(rows), 4096):
        sims = emb[i:i + 4096].float() @ ref32.T
        if allowed is not None:
            sims[:, ~allowed] = -1
        top, arg = sims.max(1)
        hit_rows, hit_cols = (sims >= THRESHOLD).nonzero(as_tuple=True)
        hits = collections.defaultdict(list)
        for r_i, c_i in zip(hit_rows.tolist(), hit_cols.tolist()):
            hits[r_i].append((ref_sets[c_i], ref_ids[c_i], round(float(sims[r_i, c_i]), 3)))
        for r_i, (t, a) in enumerate(zip(top.tolist(), arg.tolist())):
            best.append(t)
            if 0.88 <= t < 0.96:
                near.append((round(t, 3), src, rows[i + r_i]["id"], texts[i + r_i], ref_ids[a], ref_texts[a]))
            if r_i in hits:
                flagged.append((rows[i + r_i]["id"], sorted(hits[r_i], key=lambda x: -x[2])))
    with open(os.path.join(ROOT, "interim", "decontam", os.path.basename(rel)[:-6] + ".embed.flags.jsonl"), "w",
              encoding="utf-8") as f:
        for sid, hs in flagged:
            f.write(json.dumps({"id": sid, "matches": hs}, ensure_ascii=False) + "\n")
    results[src] = (len(rows), flagged, best, rel)
    print(src, len(rows), len(flagged), flush=True)

short = {"FinanceIQ test": "FIQ test", "FinanceIQ dev": "FIQ dev", "CFLUE val": "CFLUE val", "CFLUE test": "CFLUE test",
         "CMMLU test": "CMMLU", "GSM8K test": "GSM8K", "IFEval": "IFEval"}
lines = ["# 去污染报告：bge-m3 向量相似度", "",
         "由 src/01_data-processing/utils/embed_decontam.py 生成。SFT 样本取第一轮用户提示（去掉 DianJin-R1-Data 的固定说明行），"
         f"CFLUE 类的题取题干加选项；用 bge-m3 的稠密向量（CLS 归一化，最长 512 token）算余弦相似度，≥{THRESHOLD} 记为重叠。"
         f"运行设备：{DEVICE}。只被这一步、没被 8-gram 标记的样本数见 decontam_summary.md。", "",
         "| 来源 | 样本数 | " + " | ".join(short[s] for s, _ in EVAL_SETS) + " | 任一 |",
         "| --- | --- | " + " | ".join("---" for _ in EVAL_SETS) + " | --- |"]
for src, rel, skip, kind in SOURCES:
    n_rows, flagged, best, rel = results[src]
    cells = ["—" if s in skip else f"{sum(any(h[0] == s for h in hs) for _, hs in flagged):,}" for s, _ in EVAL_SETS]
    lines.append(f"| {src} | {n_rows:,} | " + " | ".join(cells) + f" | {len(flagged):,} |")
lines += ["", "## 每个样本的最高相似度", "", "| 来源 | <0.8 | 0.8–0.85 | 0.85–0.9 | 0.9–0.92 | 0.92–0.95 | ≥0.95 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
for src, *_ in SOURCES:
    bins = collections.Counter(sum(b >= e for e in (0.8, 0.85, 0.9, 0.92, 0.95)) for b in results[src][2])
    lines.append(f"| {src} | " + " | ".join(f"{bins[k]:,}" for k in range(6)) + " |")
with open(os.path.join(ROOT, "reports", "decontam_embed.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

random.Random(20261025).shuffle(near)
with open(os.path.join(ROOT, "reports", "embed_review.md"), "w", encoding="utf-8") as f:
    f.write(f"# 阈值附近的 {REVIEW} 对（相似度 0.88–0.96，随机抽取，种子 20261025）\n\n")
    for sim, src, sid, text, rid, rtext in sorted(near[:REVIEW], reverse=True):
        f.write(f"- {sim} | {src} `{sid}`：{' '.join(text.split())[:90]} ‖ `{rid}`：{' '.join(rtext.split())[:90]}\n")
