"""Embedding decontamination of the corpora at the size of one exam item, per docs/history/decisions.md item 4.

A paragraph embedding cannot see an item inside it: 40 paragraphs the 8-gram rule had matched scored a
median 0.61 and at most 0.86 against their own item, while item-sized chunks of one exam-dump document
matched 16 FinanceIQ test items at >= 0.90 (decisions.md item 4). So windows of WINDOW characters with
stride STRIDE are embedded with bge-m3 (normalized CLS, as embed_decontam.py) and compared with every
evaluation item and the CFLUE held-out 20%, using the vectors embed_decontam.py and split_cflue.py saved.

Two modes:
- ``embed <stem>...``: windows of candidate documents (an 8-gram hit from corpus_decontam.py, or an
  exam-like paragraph) and of an audit sample of AUDIT random other documents across the stems; every
  window at cosine >= LOW is written to interim/decontam_corpus/<stem>.windows.jsonl for the threshold
  reading, with its best item.
- ``mark <threshold> <stem>...``: the paragraph (corpus_decontam.py's segmentation) around each window at
  or above the threshold goes to interim/decontam_corpus/<stem>.vspans.jsonl, which build_cpt.py cuts.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python corpus_window_decontam.py <data root> embed <stem>...
       python corpus_window_decontam.py <data root> mark <threshold> <stem>...
"""
import collections
import gzip
import hashlib
import json
import os
import re
import sys

import numpy as np

WINDOW, STRIDE, LOW, PARA = 300, 150, 0.85, 2000
AUDIT = 2000  # random non-candidate documents across all stems (decisions.md item 4)
OPT = re.compile(r"(^|\n)\s*[A-D][\.．、:：]\s*\S")
CUE = re.compile(r"（\s*）|\(\s*\)|下列.{0,20}(正确|错误|属于|不属于)|以下.{0,10}(正确|错误)的是|答案[:：]\s*[A-D]")


def exam_like(text):
    return any(len(OPT.findall(text[i:i + PARA])) >= 2 and CUE.search(text[i:i + PARA]) for i in range(0, len(text), PARA))


def paragraphs(text):
    """corpus_decontam.py's segmentation: consecutive lines joined up to PARA characters."""
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


def audit_pick(doc_id, rate):
    return int(hashlib.md5(("audit-" + doc_id).encode()).hexdigest()[:8], 16) / 2**32 < rate


def docs(stem):
    with gzip.open(os.path.join(ROOT, "interim", "corpus", stem + ".jsonl.gz"), "rt", encoding="utf-8") as f:
        yield from map(json.loads, f)


if __name__ == "__main__":  # sample_doc_qa.py imports exam_like() and paragraphs()
    ROOT, MODE = sys.argv[1], sys.argv[2]
    DIR = os.path.join(ROOT, "interim", "decontam_corpus")
    if MODE == "embed":
        import torch
        from transformers import AutoModel, AutoTokenizer

        stems = sys.argv[3:]
        meta = json.load(open(os.path.join(ROOT, "interim", "embed", "ref.json"), encoding="utf-8"))
        ref = np.load(os.path.join(ROOT, "interim", "embed", "ref.npy")).astype(np.float32)
        with open(os.path.join(ROOT, "interim", "cflue", "train.jsonl"), encoding="utf-8") as f:
            train_ids = [json.loads(line)["id"] for line in f]
        with open(os.path.join(ROOT, "interim", "cflue", "heldout20.jsonl"), encoding="utf-8") as f:
            held = {json.loads(line)["id"] for line in f}
        tv = np.load(os.path.join(ROOT, "interim", "embed", "train.npy")).astype(np.float32)
        rows = [i for i, x in enumerate(train_ids) if x in held]
        # CPU when no GPU is visible, as for Fineweb files 121-130 while the teacher held the GPU budget
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ref = torch.from_numpy(np.vstack([ref, tv[rows]])).to(dev)
        ref_ids = meta["ids"] + [train_ids[i] for i in rows]
        ref_sets = meta["sets"] + ["CFLUE 留出 20%"] * len(rows)
        tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "raw", "bge-m3"))
        model = AutoModel.from_pretrained(os.path.join(ROOT, "raw", "bge-m3"),
                                          torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

        @torch.inference_mode()
        def best(texts):
            out_s, out_i = [], []
            for i in range(0, len(texts), 128):
                enc = tok(texts[i:i + 128], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
                v = torch.nn.functional.normalize(model(**enc).last_hidden_state[:, 0].float(), dim=-1)
                s, a = (v @ ref.T).max(1)
                out_s += s.tolist()
                out_i += a.tolist()
            return out_s, out_i

        total = sum(1 for s in stems for _ in docs(s))
        rate = AUDIT / total
        report = collections.defaultdict(collections.Counter)
        for stem in stems:
            hits = set()
            path = os.path.join(DIR, stem + ".spans.jsonl")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    hits = {json.loads(line)["id"] for line in f}
            with open(os.path.join(DIR, stem + ".windows.jsonl"), "w", encoding="utf-8") as out:
                buf = []

                def flush():
                    texts = [w for _, _, _, w in buf]
                    for (d_id, kind, a, w), s, j in zip(buf, *best(texts)):
                        report[(stem, kind)]["windows"] += 1
                        for e in (0.85, 0.88, 0.90, 0.92, 0.95):
                            report[(stem, kind)][f">={e}"] += s >= e
                        if s >= LOW:
                            out.write(json.dumps({"id": d_id, "kind": kind, "start": a, "end": a + len(w), "sim": round(s, 4),
                                                  "set": ref_sets[j], "item": ref_ids[j], "text": w}, ensure_ascii=False) + "\n")
                    buf.clear()

                for d in docs(stem):
                    kind = "候选" if d["id"] in hits or exam_like(d["text"]) else ("抽查" if audit_pick(d["id"], rate) else None)
                    if kind is None:
                        continue
                    report[(stem, kind)]["documents"] += 1
                    t = d["text"]
                    for a in range(0, max(len(t) - WINDOW, 0) + 1, STRIDE):
                        buf.append((d["id"], kind, a, t[a:a + WINDOW]))
                    if len(buf) >= 8192:
                        flush()
                if buf:
                    flush()
            print(stem, {k[1]: dict(v) for k, v in report.items() if k[0] == stem}, flush=True)
        lines = ["# 语料的窗口向量匹配", "", f"由 src/01_data-processing/code/corpus_window_decontam.py 生成。窗口 {WINDOW} 字、步长 {STRIDE} 字；"
                 f"候选文档为含 8-gram 命中或像考题段落的文档，抽查为另从其余文档中随机抽的约 {AUDIT} 篇。每格为最高相似度达到该值的窗口数。", "",
                 "| 来源 | 类别 | 文档 | 窗口 | ≥0.85 | ≥0.88 | ≥0.90 | ≥0.92 | ≥0.95 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for (stem, kind), c in sorted(report.items()):
            lines.append(f"| {stem} | {kind} | {c['documents']:,} | {c['windows']:,} | " + " | ".join(f"{c[f'>={e}']:,}" for e in (0.85, 0.88, 0.90, 0.92, 0.95)) + " |")
        with open(os.path.join(ROOT, "reports", f"corpus_windows_{'_'.join(stems)}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
    else:
        threshold, stems = float(sys.argv[3]), sys.argv[4:]
        for stem in stems:
            by_doc = collections.defaultdict(list)
            with open(os.path.join(DIR, stem + ".windows.jsonl"), encoding="utf-8") as f:
                for w in map(json.loads, f):
                    if w["sim"] >= threshold:
                        by_doc[w["id"]].append(w)
            n = 0
            with open(os.path.join(DIR, stem + ".vspans.jsonl"), "w", encoding="utf-8") as out:
                for d in docs(stem):
                    if d["id"] not in by_doc:
                        continue
                    cut = {}
                    for a, b in paragraphs(d["text"]):
                        for w in by_doc[d["id"]]:
                            if w["start"] < b and w["end"] > a:
                                cut[(a, b)] = [a, b, w["set"], w["item"], w["sim"]]
                    out.write(json.dumps({"id": d["id"], "spans": list(cut.values())}, ensure_ascii=False) + "\n")
                    n += len(cut)
            print(stem, "documents", len(by_doc), "paragraphs cut", n)
