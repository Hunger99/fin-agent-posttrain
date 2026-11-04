"""Split CFLUE train 80/20 and flag the samples that repeat a held-out item, per docs/history/01_data-processing.md section 5.

The 20% part is kept for preference pairs only, so that DPO learns on items SFT never saw; section 3
lists it as a reference set. A sample is flagged when either match ties it to a held-out item:

- 8-gram: decontam.py already matched every source against the whole train split as its diagnostic
  reference set (coverage >= 0.5), so the hits on held-out items are read from its flags.
- embedding: each source's saved bge-m3 vectors (embed_decontam.py) against the held-out rows of CFLUE
  train's own vectors, which embed question plus choices as the evaluation items are; cosine >= 0.90.

CFLUE train was never 8-gram matched against itself, so its 80% part is checked by embedding only.
An 80% item that repeats a held-out item is flagged and leaves the pool; the 20% part stays whole.
A DianJin-R1-Data cflue_mcq sample whose nearest CFLUE train item has been dropped is flagged too.
Nothing is deleted here: the split files hold every item, and select_sets.py applies the flags.

Usage: python split_cflue.py $FP_DATA_ZH
"""
import collections
import json
import os
import random
import sys

import numpy as np

ROOT = sys.argv[1]
SEED = 20261025  # chosen by the model; the same seed as embed_decontam.py's review sample
HELD_SHARE = 0.2  # section 5
THRESHOLD = 0.90  # embed_decontam.py's threshold (research/log/2026-10-25-0039.md)
DIAG = "CFLUE train（诊断用）"  # decontam.py's name for the whole train split as a reference set
NGRAM, EMBED = "CFLUE 留出 20%（8-gram）", "CFLUE 留出 20%（向量）"
EVAL_SETS = {"FinanceIQ test", "FinanceIQ dev", "CFLUE val", "CFLUE test", "CMMLU test", "GSM8K test", "IFEval"}
# (name, stem of the flags and vectors, file). DianJin-Fin-PRM-Data never enters SFT; it is matched only
# to count the rows that can supplement preference pairs on held-out items (section 6).
SOURCES = [("DianJin-R1-Data cflue_mcq", "dianjin_r1_mcq", "processed/sft/dianjin_r1_mcq.jsonl"),
           ("DianJin-R1-Data cflue_oe", "dianjin_r1_oe", "processed/sft/dianjin_r1_oe.jsonl"),
           ("Agentar-DeepFinance-100K", "agentar_deepfinance", "processed/sft/agentar_deepfinance.jsonl"),
           ("IndustryInstruction 金融", "industryinstruction_finance", "processed/sft/industryinstruction_finance.jsonl"),
           ("alpaca-zh", "alpaca_zh", "processed/sft/alpaca_zh.jsonl"),
           ("smoltalk-chinese", "smoltalk_chinese", "processed/sft/smoltalk_chinese.jsonl"),
           ("Tülu 3 SFT", "tulu3", "processed/sft/tulu3.jsonl"),
           ("DianJin-Fin-PRM-Data", "dianjin_fin_prm", "interim/prm/dianjin_fin_prm.jsonl")]


def read_jsonl(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(rel, rows):
    with open(os.path.join(ROOT, rel), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def file_ids(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return [json.loads(line)["id"] for line in f]


def flag_ids(stem, kind, sets):
    with open(os.path.join(ROOT, "interim", "decontam", f"{stem}.{kind}.flags.jsonl"), encoding="utf-8") as f:
        return {r["id"] for r in map(json.loads, f) if any(m[0] in sets for m in r["matches"])}


def vectors(stem):
    return np.load(os.path.join(ROOT, "interim", "embed", stem + ".npy")).astype(np.float32)


items = read_jsonl("interim/cflue/train.jsonl")
ids = [it["id"] for it in items]
order = ids[:]
random.Random(SEED).shuffle(order)
held = set(order[:round(len(ids) * HELD_SHARE)])
write_jsonl("interim/cflue/heldout20.jsonl", [it for it in items if it["id"] in held])
write_jsonl("interim/cflue/train80.jsonl", [it for it in items if it["id"] not in held])

train_vec = vectors("train")
assert len(train_vec) == len(items), "interim/embed/train.npy does not match interim/cflue/train.jsonl"
held_rows = np.array([i for i, x in enumerate(ids) if x in held])
held_ids = [ids[i] for i in held_rows]
held_vec = train_vec[held_rows]


def embed_hits(vec, row_ids):
    """``{row id: [(EMBED, held-out id, sim)]}`` for rows with cosine >= THRESHOLD to a held-out item."""
    hits = collections.defaultdict(list)
    for i in range(0, len(vec), 16384):
        sims = vec[i:i + 16384] @ held_vec.T
        for r, c in zip(*np.nonzero(sims >= THRESHOLD)):
            hits[row_ids[i + r]].append((EMBED, held_ids[c], round(float(sims[r, c]), 3)))
    return hits


def write_flags(stem, hits):
    with open(os.path.join(ROOT, "interim", "decontam", f"{stem}.heldout.flags.jsonl"), "w", encoding="utf-8") as f:
        for sid, hs in hits.items():
            f.write(json.dumps({"id": sid, "matches": sorted(hs, key=lambda h: -h[2])}, ensure_ascii=False) + "\n")


lines = ["# 去污染报告：CFLUE 训练集留出的 20%", "",
         f"由 src/01_data-processing/utils/split_cflue.py 生成。CFLUE 训练集 {len(ids):,} 题按 id 用种子 {SEED} 随机打乱，"
         f"{len(held):,} 题（{HELD_SHARE:.0%}）留出，只用于偏好对；其余 {len(ids) - len(held):,} 题为 80% 部分。"
         "8-gram 取 decontam.py 对整个训练集的诊断匹配中落在留出部分的题（覆盖 ≥50%）；"
         f"向量用 embed_decontam.py 存下的 bge-m3 向量，对留出部分余弦相似度 ≥{THRESHOLD} 记为重叠。", "",
         "## 各来源与留出部分的重叠", "",
         "DianJin-Fin-PRM-Data 不进 SFT，不删；它与留出部分重叠的行可以补充这些题的偏好对。", "",
         "| 来源 | 条数 | 8-gram | 向量 | 其中 8-gram 没标的 | 任一 |", "| --- | --- | --- | --- | --- | --- |"]
held_any = {}
for name, stem, rel in SOURCES:
    row_ids = file_ids(rel)
    hits = embed_hits(vectors(stem), row_ids)
    em = set(hits)
    with open(os.path.join(ROOT, "interim", "decontam", f"{stem}.ngram.flags.jsonl"), encoding="utf-8") as f:
        for r in map(json.loads, f):
            for m in r["matches"]:
                if m[0] == DIAG and m[1] in held:
                    hits[r["id"]].append((NGRAM, m[1], m[2]))
    ng = {sid for sid, hs in hits.items() if any(h[0] == NGRAM for h in hs)}
    write_flags(stem, hits)
    held_any[stem] = set(hits)
    lines.append(f"| {name} | {len(row_ids):,} | {len(ng):,} | {len(em):,} | {len(em - ng):,} | {len(hits):,} |")
    print(name, len(row_ids), len(ng), len(em), len(hits), flush=True)

# The 80% part against the held-out part, by embedding only (see the module docstring).
part_rows = np.array([i for i, x in enumerate(ids) if x not in held])
part_ids = [ids[i] for i in part_rows]
dup = embed_hits(train_vec[part_rows], part_ids)
write_flags("train", dup)
dup_sims = sorted((top[2], sid, top[1]) for sid, top in ((s, max(hs, key=lambda h: h[2])) for s, hs in dup.items()))
text = {it["id"]: it["question"] for it in items}
lines += ["", "## 80% 部分与留出部分的同题", "",
          f"80% 部分有 {len(dup):,} 题与留出部分的题向量相似度 ≥{THRESHOLD}，从 80% 部分去掉，留出部分不变。"
          "下面是相似度最低的 3 对，题干截到前 80 字：", ""]
for sim, a, b in dup_sims[:3]:
    lines.append(f"- {sim}：`{a}` {' '.join(text[a].split())[:80]} ‖ `{b}` {' '.join(text[b].split())[:80]}")

# What each part keeps once evaluation overlaps (0.90 run, both matches) are removed as well.
evalf = flag_ids("train", "ngram", EVAL_SETS) | flag_ids("train", "embed", EVAL_SETS)
held_ok = [it for it in items if it["id"] in held and it["id"] not in evalf]
part_ok = [x for x in part_ids if x not in evalf and x not in dup]
tasks = collections.Counter(it["task"] for it in held_ok)
prm_eval = flag_ids("dianjin_fin_prm", "ngram", EVAL_SETS) | flag_ids("dianjin_fin_prm", "embed", EVAL_SETS)
prm_ok = held_any["dianjin_fin_prm"] - prm_eval
lines += ["", "## 两部分各剩多少题", "",
          "| 部分 | 题数 | 与评测集重叠 | 与留出部分同题 | 剩下 |", "| --- | --- | --- | --- | --- |",
          f"| 留出 20%（偏好对） | {len(held):,} | {len(held & evalf):,} | — | {len(held_ok):,} |",
          f"| 80% 部分（SFT、种子题） | {len(part_ids):,} | {len(set(part_ids) & evalf):,} | {len(dup):,} | {len(part_ok):,} |",
          "", "留出部分剩下的题按题型：" + "，".join(f"{k} {v:,}" for k, v in tasks.most_common()) + "。",
          f"DianJin-Fin-PRM-Data 中与留出部分重叠、且不与评测集重叠的有 {len(prm_ok):,} 条，可以补充这些题的偏好对。"]

# DianJin-R1-Data cflue_mcq re-renders CFLUE train items as SFT samples, so a sample shares its item's
# fate. A first run (research/log/2026-10-25-0726.md) kept 693 samples whose nearest item (cosine >= 0.90)
# had been dropped: 518 items overlapped an evaluation item and 175 others repeated a held-out item, so
# those samples leaked held-out questions into SFT although each fell just under 0.90 to the held-out
# item itself. Such samples are flagged here too. The nearest item of every sample is written out, so select_sets.py can
# count each CFLUE question once.
INHERIT = "CFLUE 训练题已删"
mcq_ids = file_ids("processed/sft/dianjin_r1_mcq.jsonl")
mcq_vec = vectors("dianjin_r1_mcq")
mcq_drop = (flag_ids("dianjin_r1_mcq", "ngram", EVAL_SETS) | flag_ids("dianjin_r1_mcq", "embed", EVAL_SETS)
            | held_any["dianjin_r1_mcq"])
part_ok_set = set(part_ok)
reason = {}
for x in ids:
    reason[x] = ("留出部分" if x in held else "与评测集重叠" if x in evalf else "与留出部分同题" if x in dup else None)
inherit, why, nearest = {}, collections.Counter(), []
for i in range(0, len(mcq_ids), 16384):
    sims = mcq_vec[i:i + 16384] @ train_vec.T
    for r, (b, a) in enumerate(zip(sims.max(1).tolist(), sims.argmax(1).tolist())):
        sid = mcq_ids[i + r]
        nearest.append({"id": sid, "cflue_id": ids[a], "sim": round(b, 3), "same_item": b >= THRESHOLD})
        if sid not in mcq_drop and b >= THRESHOLD and reason[ids[a]]:
            inherit[sid] = [(INHERIT, ids[a], round(b, 3))]
            why[reason[ids[a]]] += 1
write_jsonl("interim/cflue/dianjin_mcq_nearest.jsonl", nearest)
with open(os.path.join(ROOT, "interim", "decontam", "dianjin_r1_mcq.inherit.flags.jsonl"), "w", encoding="utf-8") as f:
    for sid, hs in inherit.items():
        f.write(json.dumps({"id": sid, "matches": hs}, ensure_ascii=False) + "\n")
kept = [n for n in nearest if n["id"] not in mcq_drop and n["id"] not in inherit]
twins = sum(n["same_item"] and n["cflue_id"] in part_ok_set for n in kept)
lines += ["", "## DianJin-R1-Data 选择题部分对应的 CFLUE 题", "",
          f"去掉与评测集、留出部分重叠的样本后剩 {len(mcq_ids) - len(mcq_drop):,} 条。其中 {len(inherit):,} 条最相近的 CFLUE 训练题"
          f"（向量相似度 ≥{THRESHOLD}）已被删掉，样本随之删掉（"
          + "，".join(f"{k} {v:,} 条" for k, v in why.most_common()) + f"），剩 {len(kept):,} 条。"
          f"其中 {twins:,} 条对应 80% 部分剩下的题，同一道题在领域池中出现两次；"
          f"另 {len(kept) - twins:,} 条与任何 CFLUE 训练题的相似度都低于 {THRESHOLD}。"]
with open(os.path.join(ROOT, "reports", "heldout_decontam.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
