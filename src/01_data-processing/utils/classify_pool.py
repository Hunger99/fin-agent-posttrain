"""Apply the FinCorpus quality classifier to the scoring pool and keep documents predicted to score >= 3.

docs/history/01_data-processing.md section 2; the classifier comes from train_score_classifier.py and the pool
from corpus_filter.py (about 3B tokens of FinCorpus, research/log/2026-10-25-0855.md). The pool is split
into shards by document order so that several GPUs embed it at once; a final merge writes the kept
documents with their predicted scores.

Usage:
  CUDA_VISIBLE_DEVICES=<gpu> python classify_pool.py <data root> embed <shard> <shards>
  python classify_pool.py <data root> merge <shards>
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

ROOT, MODE = sys.argv[1], sys.argv[2]
POOL = sorted(glob.glob(os.path.join(ROOT, "interim", "corpus", "fincorpus_*.jsonl.gz")))
POOL = [p for p in POOL if not p.endswith("_kept.jsonl.gz")]
OUT = os.path.join(ROOT, "interim", "scoring")


def docs():
    for path in POOL:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from map(json.loads, f)


if MODE == "embed":
    import torch
    from transformers import AutoModel, AutoTokenizer

    shard, shards = int(sys.argv[3]), int(sys.argv[4])
    w = np.load(os.path.join(OUT, "classifier.npz"))["w"]
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "raw", "bge-m3"))
    model = AutoModel.from_pretrained(os.path.join(ROOT, "raw", "bge-m3"), torch_dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def predict(texts):
        # Within a buffer, sorting by length keeps padding low, as in embed_decontam.py
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        pred = np.empty(len(texts), dtype=np.float32)
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            enc = tok([texts[j][:4000] for j in idx], padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to("cuda")
            v = torch.nn.functional.normalize(model(**enc).last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()
            pred[idx] = v @ w[:-1] + w[-1]
        return pred

    with open(os.path.join(OUT, f"pool_pred_{shard}.jsonl"), "w", encoding="utf-8") as f:
        buf = []
        for i, d in enumerate(docs()):
            if i % shards != shard:
                continue
            buf.append(d)
            if len(buf) == 8192:
                for d2, p in zip(buf, predict([x["text"] for x in buf])):
                    f.write(json.dumps({"id": d2["id"], "pred": round(float(p), 3)}) + "\n")
                buf = []
                print(f"shard {shard}: {i:,} docs read", flush=True)
        if buf:
            for d2, p in zip(buf, predict([x["text"] for x in buf])):
                f.write(json.dumps({"id": d2["id"], "pred": round(float(p), 3)}) + "\n")
else:
    shards = int(sys.argv[3])
    pred = {}
    for k in range(shards):
        with open(os.path.join(OUT, f"pool_pred_{k}.jsonl"), encoding="utf-8") as f:
            pred.update((r["id"], r["pred"]) for r in map(json.loads, f))
    counts = {}
    with gzip.open(os.path.join(ROOT, "interim", "corpus", "fincorpus_kept.jsonl.gz"), "wt", encoding="utf-8") as out:
        for d in docs():
            c = counts.setdefault(d["source"], [0, 0, 0, 0])
            c[0] += 1
            c[1] += d["n_tokens"]
            if round(pred[d["id"]]) >= 3:
                d["pred_score"] = pred[d["id"]]
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
                c[2] += 1
                c[3] += d["n_tokens"]
    lines = ["# FinCorpus 打分池的分类结果", "", "由 src/01_data-processing/utils/classify_pool.py 生成；预测分四舍五入 ≥3 的文档保留。", "",
             "| 来源 | 打分池（篇 / token） | 保留（篇 / token） | 保留比例（按 token） |", "| --- | --- | --- | --- |"]
    for s, (n, t, k, kt) in counts.items():
        lines.append(f"| {s} | {n:,} / {t / 1e9:.2f}B | {k:,} / {kt / 1e9:.2f}B | {kt / max(t, 1):.0%} |")
    with open(os.path.join(ROOT, "reports", "pool_classified.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
