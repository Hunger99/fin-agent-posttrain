"""Find corpus paragraphs that overlap an evaluation item or a held-out CFLUE item by section 3's 8-gram rule.

docs/history/01_data-processing.md section 3: a corpus is judged in paragraphs of at most 2,000 characters and
only the overlapping paragraphs are removed. A paragraph is consecutive lines joined up to 2,000
characters, a longer line being cut into 2,000-character pieces. The rule is decontam.py's: text NFKC-
normalized with whitespace collapsed, Qwen2.5-1.5B tokens, and a paragraph overlaps a reference item when
its 8-grams cover at least half of the item's tokens; 8-grams in more than 100 items only count toward
coverage. The reference sets are decontam.py's evaluation sets plus the held-out 20% of CFLUE train
(split_cflue.py), which section 3 lists as a reference set. IndustryCorpus2 prints every Chinese punctuation
mark in half width, and NFKC leaves 。、“”《》【】 and the like alone, so both sides also map those to one
half-width form (HALF; docs/history/decisions.md item 5 (2)). Nothing is deleted here: the character spans
of overlapping paragraphs go to interim/decontam_corpus/<stem>.spans.jsonl and are cut when the
continued-pretraining corpus is written.

Usage: python corpus_decontam.py <data root> <stem>...
"""
import collections
import gzip
import json
import multiprocessing
import os
import re
import sys
import unicodedata

import numpy as np
from tokenizers import Tokenizer

ROOT, STEMS = sys.argv[1], sys.argv[2:]
N, THRESHOLD, COMMON, PARA = 8, 0.5, 100, 2000  # decontam.py's rule; section 3's paragraph size
REF_SETS = [("FinanceIQ test", "processed/eval/financeiq_test.jsonl"), ("FinanceIQ dev", "processed/eval/financeiq_dev.jsonl"),
            ("CFLUE val", "processed/eval/cflue_val.jsonl"), ("CFLUE test", "processed/eval/cflue_test.jsonl"),
            ("CMMLU test", "processed/eval/cmmlu_test.jsonl"), ("GSM8K test", "processed/eval/gsm8k_test.jsonl"),
            ("IFEval", "processed/eval/ifeval.jsonl"), ("CFLUE 留出 20%", "interim/cflue/heldout20.jsonl")]
tok = Tokenizer.from_file(os.path.join(ROOT, "raw", "qwen2.5-1.5b-tokenizer", "tokenizer.json"))


HALF = str.maketrans({"。": ".", "、": ",", "“": '"', "”": '"', "‘": "'", "’": "'", "《": '"', "》": '"', "〈": '"', "〉": '"',
                      "「": '"', "」": '"', "『": '"', "』": '"', "【": "[", "】": "]"})


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).translate(HALF)).strip()


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


refs = []  # (set name, item id, token count, 8-gram hashes)
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
ref_sorted = np.sort(np.fromiter(postings.keys(), dtype=np.uint64, count=len(postings)))
ones = np.ones(N, dtype=np.int8)


def paragraphs(text):
    """(start, end) character spans of consecutive lines joined up to PARA characters."""
    spans, start, pos = [], 0, 0
    for line in text.splitlines(keepends=True):
        if pos + len(line) - start > PARA and pos > start:
            spans.append((start, pos))
            start = pos
        pos += len(line)
        while pos - start > PARA:
            spans.append((start, start + PARA))
            start += PARA
    if pos > start:
        spans.append((start, pos))
    return spans


def work(lines):
    out, n_para = [], 0
    for line in lines:
        d = json.loads(line)
        spans = paragraphs(d["text"])
        n_para += len(spans)
        enc = tok.encode_batch([norm(d["text"][a:b]) for a, b in spans], add_special_tokens=False)
        hit = []
        for (a, b), e in zip(spans, enc):
            h = gram_hashes(e.ids)
            if not len(h):
                continue
            pos = np.searchsorted(ref_sorted, h)
            pos[pos == len(ref_sorted)] = 0
            shared = np.unique(h[ref_sorted[pos] == h])
            if not len(shared):
                continue
            cands = {ri for g in shared.tolist() if len(postings[g]) <= COMMON for ri in postings[g]}
            best = None
            for ri in cands:
                set_name, item_id, n_tok, grams = refs[ri]
                cov = np.count_nonzero(np.convolve(np.isin(grams, shared).astype(np.int8), ones)) / n_tok
                if cov >= THRESHOLD and (best is None or cov > best[2]):
                    best = (set_name, item_id, round(float(cov), 3))
            if best:
                hit.append([a, b, *best])
        if hit:
            out.append({"id": d["id"], "spans": hit})
    return out, len(lines), n_para


def batches(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        batch = []
        for line in f:
            batch.append(line)
            if len(batch) == 500:
                yield batch
                batch = []
        if batch:
            yield batch


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "interim", "decontam_corpus"), exist_ok=True)
    lines = ["# 语料去污染：8-gram 匹配到的段落", "",
             "由 src/01_data-processing/utils/corpus_decontam.py 生成。段落为不超过 2,000 字的连续行；规则同 decontam.py（覆盖参照题 ≥50% 的 token），"
             "参照集为全部评测集和 CFLUE 训练集留出的 20%。", "",
             "| 来源 | 篇数 | 段落 | 命中的段落 | 含命中段落的文档 | " + " | ".join(s for s, _ in REF_SETS) + " |",
             "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in REF_SETS) + " |"]
    with multiprocessing.Pool(24) as pool:  # forked after the index is built, so workers share it
        for stem in STEMS:
            n_doc = n_para = 0
            by_set = collections.Counter()
            hit_docs = hit_paras = 0
            with open(os.path.join(ROOT, "interim", "decontam_corpus", stem + ".spans.jsonl"), "w", encoding="utf-8") as f:
                for out, nd, npar in pool.imap(work, batches(os.path.join(ROOT, "interim", "corpus", stem + ".jsonl.gz")), chunksize=1):
                    n_doc += nd
                    n_para += npar
                    for r in out:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        hit_docs += 1
                        hit_paras += len(r["spans"])
                        by_set.update(s[2] for s in r["spans"])
            lines.append(f"| {stem} | {n_doc:,} | {n_para:,} | {hit_paras:,} | {hit_docs:,} | "
                         + " | ".join(f"{by_set[s]:,}" for s, _ in REF_SETS) + " |")
            print(lines[-1], flush=True)
    with open(os.path.join(ROOT, "reports", f"corpus_decontam_{'_'.join(STEMS)}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
