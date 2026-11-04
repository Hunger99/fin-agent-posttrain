"""Flag training samples that overlap evaluation items by the 8-gram rule of docs/history/01_data-processing.md section 3.

A sample overlaps a reference item when 8-grams that also occur in the sample cover at least half of
the item's tokens, the rule Tülu 3 uses. Both texts are NFKC-normalized with whitespace collapsed and
tokenized with the starting model's tokenizer. Rewritten items are caught by embed_decontam.py; this
pass only flags, it removes nothing.

8-grams are hashed with numpy, so the million-sample general pool takes minutes; the first round
(research/log/2026-10-25-0014.md) hashed Python tuples one by one, which would take hours here.

The CFLUE train split is matched as a reference set only to find which split DianJin-R1-Data and
DianJin-Fin-PRM-Data were built from; it is training data, not a reference set.

Usage: python decontam.py $FP_DATA_ZH
"""
import collections
import json
import os
import re
import sys
import unicodedata

import numpy as np
from tokenizers import Tokenizer

ROOT = sys.argv[1]
N = 8
THRESHOLD = 0.5  # decided in research/log/2026-10-25-0039.md
# 8-grams that occur in more than COMMON reference items, such as stock exam phrasing, are not used to
# find candidate items, since each would pull in hundreds of unrelated items per sample. They still
# count toward coverage once an item is a candidate. An item made only of such 8-grams would be
# missed; exam items are long enough that none is expected, and the report counts the items too short
# to have any 8-gram.
COMMON = 100
DIAG = "CFLUE train（诊断用）"
REF_SETS = [("FinanceIQ test", "processed/eval/financeiq_test.jsonl"),
            ("FinanceIQ dev", "processed/eval/financeiq_dev.jsonl"),
            ("CFLUE val", "processed/eval/cflue_val.jsonl"),
            ("CFLUE test", "processed/eval/cflue_test.jsonl"),
            ("CMMLU test", "processed/eval/cmmlu_test.jsonl"),
            ("GSM8K test", "processed/eval/gsm8k_test.jsonl"),
            ("IFEval", "processed/eval/ifeval.jsonl"),
            (DIAG, "interim/cflue/train.jsonl")]
EVAL_SETS = [s for s, _ in REF_SETS if s != DIAG]
# (name, file, reference sets not matched, how the text is built). CFLUE train is also a training
# source (its items become SFT samples and seed questions), so it is checked like any other source but
# never against itself. The CFLUE selection and test sets are checked against the other evaluation
# sets, because a selection set that repeats test items would let selection read the test set.
SOURCES = [("DianJin-R1-Data cflue_mcq", "processed/sft/dianjin_r1_mcq.jsonl", set(), "sft"),
           ("DianJin-R1-Data cflue_oe", "processed/sft/dianjin_r1_oe.jsonl", set(), "sft"),
           ("Agentar-DeepFinance-100K", "processed/sft/agentar_deepfinance.jsonl", set(), "sft"),
           ("IndustryInstruction 金融", "processed/sft/industryinstruction_finance.jsonl", set(), "sft"),
           ("CFLUE train", "interim/cflue/train.jsonl", {DIAG}, "seed"),
           ("DianJin-Fin-PRM-Data", "interim/prm/dianjin_fin_prm.jsonl", set(), "prm"),
           ("alpaca-zh", "processed/sft/alpaca_zh.jsonl", set(), "sft"),
           ("smoltalk-chinese", "processed/sft/smoltalk_chinese.jsonl", set(), "sft"),
           ("Tülu 3 SFT", "processed/sft/tulu3.jsonl", set(), "sft"),
           ("CFLUE val（评测集之间）", "processed/eval/cflue_val.jsonl", {"CFLUE val", DIAG}, "item"),
           ("CFLUE test（评测集之间）", "processed/eval/cflue_test.jsonl", {"CFLUE test", DIAG}, "item")]
EXAMPLES = 2  # pairs per source shown in the report, taken just above the threshold
tok = Tokenizer.from_file(os.path.join(ROOT, "raw", "qwen2.5-1.5b-tokenizer", "tokenizer.json"))


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def read_jsonl(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def item_text(it):
    return it["question"] + " " + " ".join(f"{k}. {v}" for k, v in sorted(it["choices"].items()))


def text_of(r, kind):
    if kind == "sft":
        return " ".join(m["content"] + " " + m.get("reasoning", "") for m in r["messages"])
    if kind == "seed":  # a CFLUE item as an SFT sample carries its answer and analysis too
        return item_text(r) + " 答案：" + r["answer"] + " " + (r.get("analysis") or "")
    if kind == "prm":
        return item_text(r) + " " + (r.get("trace") or "") + " " + (r.get("final_answer") or "")
    return item_text(r)


def gram_hashes(ids):
    """64-bit hash of every 8-token window: a polynomial hash finished with the splitmix64 mixer."""
    a = np.asarray(ids, dtype=np.uint64)
    k = len(a) - N + 1
    if k <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(k, dtype=np.uint64)
    for j in range(N):
        h = h * np.uint64(0x100000001B3) + a[j:j + k]
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return h


def encode(texts, batch=20000):
    for i in range(0, len(texts), batch):
        yield from (e.ids for e in tok.encode_batch([norm(t) for t in texts[i:i + batch]], add_special_tokens=False))


refs, ref_text, ref_stats = [], {}, []  # refs: (set name, item id, token count, 8-gram hashes in order)
for set_name, rel in REF_SETS:
    items = read_jsonl(rel)
    texts = [item_text(it) for it in items]
    short = 0
    for it, text, t in zip(items, texts, encode(texts)):
        g = gram_hashes(t)
        short += not len(g)
        refs.append((set_name, it["id"], len(t), g))
        ref_text[it["id"]] = text
    ref_stats.append((set_name, len(items), short))
postings = collections.defaultdict(list)
for ri, (_, _, _, g) in enumerate(refs):
    for x in np.unique(g).tolist():
        postings[x].append(ri)
ref_sorted = np.sort(np.fromiter(postings.keys(), dtype=np.uint64, count=len(postings)))
ones = np.ones(N, dtype=np.int8)

results = {}  # source -> (sample count, [(sample id, [(set, item id, coverage)])], best eval coverage per sample)
examples = collections.defaultdict(list)
os.makedirs(os.path.join(ROOT, "interim", "decontam"), exist_ok=True)
for src, rel, skip, kind in SOURCES:
    rows = read_jsonl(rel)
    texts = [text_of(r, kind) for r in rows]
    flagged, best = [], []
    for r, text, t in zip(rows, texts, encode(texts)):
        h = gram_hashes(t)
        pos = np.searchsorted(ref_sorted, h)
        pos[pos == len(ref_sorted)] = 0
        shared = np.unique(h[ref_sorted[pos] == h]) if len(h) else h
        cands = set()
        for g in shared.tolist():
            p = postings[g]
            if len(p) <= COMMON:
                cands.update(p)
        hits, top = [], 0.0
        for ri in cands:
            set_name, item_id, n_tok, grams = refs[ri]
            if set_name in skip:
                continue
            cov = np.count_nonzero(np.convolve(np.isin(grams, shared).astype(np.int8), ones)) / n_tok
            if set_name != DIAG:
                top = max(top, cov)
            if cov >= THRESHOLD:
                hits.append((set_name, item_id, round(float(cov), 3)))
                if set_name != DIAG and cov < 0.6 and len(examples[src]) < EXAMPLES:
                    examples[src].append((round(float(cov), 3), item_id, ref_text[item_id], r["id"], text))
        best.append(top)
        if hits:
            flagged.append((r["id"], sorted(hits, key=lambda x: -x[2])))
    with open(os.path.join(ROOT, "interim", "decontam", os.path.basename(rel)[:-6] + ".ngram.flags.jsonl"), "w",
              encoding="utf-8") as f:
        for sid, hits in flagged:
            f.write(json.dumps({"id": sid, "matches": hits}, ensure_ascii=False) + "\n")
    results[src] = (len(rows), flagged, best)
    print(src, len(rows), len(flagged), flush=True)

short_names = {"FinanceIQ test": "FIQ test", "FinanceIQ dev": "FIQ dev", "CFLUE val": "CFLUE val",
               "CFLUE test": "CFLUE test", "CMMLU test": "CMMLU", "GSM8K test": "GSM8K", "IFEval": "IFEval"}
lines = ["# 去污染报告：8-gram 匹配", "",
         "由 src/01_data-processing/utils/decontam.py 生成。规则：文本做 NFKC 并合并空白后，用 Qwen2.5-1.5B 的分词器切 token；"
         f"样本中出现的 {N}-gram 覆盖参照题 {THRESHOLD:.0%} 以上的 token，即算重叠。参照题为题干加选项；"
         "SFT 样本为全部轮次的内容和推理，CFLUE 训练集为题目加答案和解析，DianJin-Fin-PRM-Data 为题目加推理。"
         "向量相似度见 decontam_embed.md。这一步只标记，不删除。", "",
         "## 参照集", "", "| 参照集 | 题数 | 不足 8 个 token、无法匹配的题 |", "| --- | --- | --- |"]
lines += [f"| {s} | {n:,} | {short:,} |" for s, n, short in ref_stats]
lines += ["", "## 与各评测参照集重叠的样本数", "",
          "最后两列是与任一评测参照集重叠的样本，即按这一步会删掉的样本；最后两行是评测集之间的重叠，不删。", "",
          "| 来源 | 样本数 | " + " | ".join(short_names[s] for s in EVAL_SETS) + " | 任一 | 占比 |",
          "| --- | --- | " + " | ".join("---" for _ in EVAL_SETS) + " | --- | --- |"]
for src, rel, skip, kind in SOURCES:
    n_rows, flagged, _ = results[src]
    cells = ["—" if s in skip else f"{sum(any(h[0] == s for h in hs) for _, hs in flagged):,}" for s in EVAL_SETS]
    k = sum(any(h[0] in EVAL_SETS for h in hs) for _, hs in flagged)
    lines.append(f"| {src} | {n_rows:,} | " + " | ".join(cells) + f" | {k:,} | {k / n_rows:.1%} |")
lines += ["", "## 覆盖率 ≥0.9 的重叠", "", "这一档几乎都是同一道题；每格为重叠的样本数 / 被覆盖的参照题数，只列非零的格。", ""]
for src, rel, skip, kind in SOURCES:
    n_rows, flagged, _ = results[src]
    cells = []
    for s in EVAL_SETS:
        samples = sum(any(h[0] == s and h[2] >= 0.9 for h in hs) for _, hs in flagged)
        items = len({h[1] for _, hs in flagged for h in hs if h[0] == s and h[2] >= 0.9})
        if samples:
            cells.append(f"{short_names[s]} {samples:,} / {items:,}")
    lines.append(f"- {src}：" + ("；".join(cells) if cells else "无"))
lines += ["", "## 诊断：由 CFLUE 生成的来源用了哪个划分", "", "每个样本取覆盖率最高的 CFLUE 题，按该题所在的划分计数。", "",
          "| 来源 | train | val | test | 不与 CFLUE 重叠 |", "| --- | --- | --- | --- | --- |"]
for src in ("DianJin-R1-Data cflue_mcq", "DianJin-R1-Data cflue_oe", "DianJin-Fin-PRM-Data"):
    n_rows, flagged, _ = results[src]
    c = collections.Counter()
    for _, hs in flagged:
        cflue = [h for h in hs if h[0].startswith("CFLUE")]
        if cflue:
            c["train" if cflue[0][0] == DIAG else cflue[0][0].split()[1]] += 1
    lines.append(f"| {src} | {c['train']:,} | {c['val']:,} | {c['test']:,} | {n_rows - sum(c.values()):,} |")
lines += ["", "## 每个样本对评测参照集的最高覆盖率", "", "| 来源 | <0.1 | 0.1–0.3 | 0.3–0.5 | 0.5–0.7 | 0.7–0.9 | ≥0.9 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
for src, rel, skip, kind in SOURCES:
    bins = collections.Counter(sum(c >= e for e in (0.1, 0.3, 0.5, 0.7, 0.9)) for c in results[src][2])
    lines.append(f"| {src} | " + " | ".join(f"{bins[k]:,}" for k in range(6)) + " |")
lines += ["", "## 阈值附近的例子", "", f"每个来源取最先遇到的 {EXAMPLES} 对覆盖率在 0.5–0.6 的匹配，文本截到前 150 字。", ""]
for src, ex in examples.items():
    for cov, item_id, rtext, sid, stext in ex:
        lines.append(f"- **{src}**，覆盖率 {cov}。参照题 `{item_id}`：{norm(rtext)[:150]}。样本 `{sid}`：{norm(stext)[:150]}")
with open(os.path.join(ROOT, "reports", "decontam.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
