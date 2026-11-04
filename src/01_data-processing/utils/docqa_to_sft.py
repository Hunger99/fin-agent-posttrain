"""Turn kept document-QA questions into closed-book SFT samples, removing duplicates and contaminated ones.

docs/history/decisions.md item 3: the user turn is the question alone and the assistant turn the answer; item 3 (4)
asks that document QA be deduplicated and decontaminated on question plus answer. A question is taken when
doc_qa.py kept it and, when the exam-scope check ran (item 7), judged it in scope. Then:

- exact duplicates of the normalized question (NFKC, whitespace removed) keep their first occurrence in
  chunk-id order;
- decontam.py's 8-gram rule on question plus answer, against every evaluation set and the CFLUE held-out
  20% (coverage of an item's tokens >= 0.5; 8-grams in more than COMMON items count only toward coverage);
- bge-m3 on question plus answer (normalized CLS, first 512 tokens) against the vectors embed_decontam.py
  and split_cflue.py saved, at cosine >= 0.90, the SFT threshold of research/log/2026-10-25-0039.md.

Each sample keeps its chunk id, generation settings and filter results in "meta", and "source" is
"doc_qa (synthetic)", so R3-synthetic can remove the whole source (section 6).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python docqa_to_sft.py <data root> <output.jsonl> <doc_qa output.jsonl>...
"""
import collections
import json
import os
import re
import sys
import unicodedata

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer

ROOT, OUT, SRCS = sys.argv[1], sys.argv[2], sys.argv[3:]
N, COVER, COMMON, SIM = 8, 0.5, 100, 0.90  # decontam.py's 8-gram rule; embed_decontam.py's threshold
REF_SETS = [("FinanceIQ test", "processed/eval/financeiq_test.jsonl"), ("FinanceIQ dev", "processed/eval/financeiq_dev.jsonl"),
            ("CFLUE val", "processed/eval/cflue_val.jsonl"), ("CFLUE test", "processed/eval/cflue_test.jsonl"),
            ("CMMLU test", "processed/eval/cmmlu_test.jsonl"), ("GSM8K test", "processed/eval/gsm8k_test.jsonl"),
            ("IFEval", "processed/eval/ifeval.jsonl"), ("CFLUE 留出 20%", "interim/cflue/heldout20.jsonl")]


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def gram_hashes(ids):
    """decontam.py's 64-bit hash of every 8-token window."""
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


# Kept questions, deduplicated on the normalized question
samples, seen, counts = [], set(), collections.Counter()
for path in SRCS:
    with open(path, encoding="utf-8") as f:
        rows = sorted(map(json.loads, f), key=lambda r: r["chunk_id"])
    for r in rows:
        for j, q in enumerate(r["questions"]):
            if not q["kept"] or not q.get("in_scope", True):
                continue
            counts["kept"] += 1
            key = re.sub(r"\s+", "", unicodedata.normalize("NFKC", q["question"]))
            if key in seen:
                counts["duplicate"] += 1
                continue
            seen.add(key)
            samples.append((r, j, q))

# 8-gram on question plus answer
tok = Tokenizer.from_file(os.path.join(ROOT, "raw", "qwen2.5-1.5b-tokenizer", "tokenizer.json"))
refs = []
for set_name, rel in REF_SETS:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    texts = [norm(it["question"] + " " + " ".join(f"{k}. {v}" for k, v in sorted(it["choices"].items()))) for it in items]
    for it, e in zip(items, tok.encode_batch(texts, add_special_tokens=False)):
        refs.append((set_name, it["id"], len(e.ids), gram_hashes(e.ids)))
postings = collections.defaultdict(list)
for ri, (_, _, _, g) in enumerate(refs):
    for x in np.unique(g).tolist():
        postings[x].append(ri)
ones = np.ones(N, dtype=np.int8)
qa = [q["question"] + " " + q["answer"] for _, _, q in samples]
ngram = {}
for i, e in enumerate(tok.encode_batch([norm(t) for t in qa], add_special_tokens=False)):
    h = np.unique(gram_hashes(e.ids))
    cands = {ri for x in h.tolist() if 0 < len(postings.get(x, ())) <= COMMON for ri in postings[x]}
    for ri in cands:
        set_name, item_id, n_tok, grams = refs[ri]
        cov = np.count_nonzero(np.convolve(np.isin(grams, h).astype(np.int8), ones)) / n_tok
        if cov >= COVER and (i not in ngram or cov > ngram[i][2]):
            ngram[i] = (set_name, item_id, round(float(cov), 3))

# bge-m3 on question plus answer
meta = json.load(open(os.path.join(ROOT, "interim", "embed", "ref.json"), encoding="utf-8"))
ref = np.load(os.path.join(ROOT, "interim", "embed", "ref.npy")).astype(np.float32)
with open(os.path.join(ROOT, "interim", "cflue", "train.jsonl"), encoding="utf-8") as f:
    train_ids = [json.loads(line)["id"] for line in f]
with open(os.path.join(ROOT, "interim", "cflue", "heldout20.jsonl"), encoding="utf-8") as f:
    held = {json.loads(line)["id"] for line in f}
rows_held = [i for i, x in enumerate(train_ids) if x in held]
ref = torch.from_numpy(np.vstack([ref, np.load(os.path.join(ROOT, "interim", "embed", "train.npy")).astype(np.float32)[rows_held]])).cuda()
ref_ids = meta["ids"] + [train_ids[i] for i in rows_held]
ref_sets = meta["sets"] + ["CFLUE 留出 20%"] * len(rows_held)
btok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "raw", "bge-m3"))
model = AutoModel.from_pretrained(os.path.join(ROOT, "raw", "bge-m3"), torch_dtype=torch.float16).cuda().eval()
embed = {}
with torch.inference_mode():
    for i in range(0, len(qa), 128):
        enc = btok(qa[i:i + 128], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        v = torch.nn.functional.normalize(model(**enc).last_hidden_state[:, 0].float(), dim=-1)
        s, a = (v @ ref.T).max(1)  # float32, as embed_decontam.py: 0.90 is not representable in float16
        for k, (sim, j) in enumerate(zip(s.tolist(), a.tolist())):
            if sim >= SIM:
                embed[i + k] = (ref_sets[j], ref_ids[j], round(sim, 4))

by_set = collections.Counter()
with open(OUT, "w", encoding="utf-8") as out:
    for i, (r, j, q) in enumerate(samples):
        if i in ngram or i in embed:
            counts["contaminated"] += 1
            by_set.update({(ngram.get(i) or embed[i])[0]})
            continue
        out.write(json.dumps({"id": f"{r['chunk_id']}#q{j}", "source": "doc_qa (synthetic)",
                              "messages": [{"role": "user", "content": q["question"]}, {"role": "assistant", "content": q["answer"]}],
                              "meta": {"chunk_id": r["chunk_id"], "doc_source": r["source"], "type": q["type"],
                                       "evidence": q["evidence"], "consistent": q.get("consistent"), **r["meta"]}},
                             ensure_ascii=False) + "\n")
        counts["written"] += 1
lines = ["# 文档问答转成 SFT 样本", "", f"由 src/01_data-processing/utils/docqa_to_sft.py 从 {', '.join('`' + os.path.basename(p) + '`' for p in SRCS)} 生成。"
         "去重按规范化后的题目；去污染按\"题目 + 答案\"，8-gram 覆盖 ≥0.5 或 bge-m3 相似度 ≥0.90 即删。", "",
         f"- 保留且在考试范围内的题：{counts['kept']:,}；题目重复：{counts['duplicate']:,}；与评测题或 CFLUE 留出部分重叠：{counts['contaminated']:,}"
         f"（其中 8-gram {len(ngram):,}、向量 {len(embed):,}，有重合）；写出：{counts['written']:,}。",
         "- 重叠按参照集：" + ("，".join(f"{k} {v:,}" for k, v in by_set.most_common()) or "无") + "。"]
with open(os.path.join(ROOT, "reports", f"docqa_to_sft_{os.path.splitext(os.path.basename(OUT))[0]}.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
