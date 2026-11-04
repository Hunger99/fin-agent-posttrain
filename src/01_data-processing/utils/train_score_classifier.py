"""Fit the FinCorpus quality classifier: a linear regression from bge-m3 vectors to the teacher's 0-5 score.

docs/history/01_data-processing.md section 2 follows FineWeb-Edu: the teacher scores a sample, a linear head on
embedding vectors learns those scores, and a document is kept when its predicted score rounds to 3 or
more. The vector is bge-m3's normalized CLS vector over the first 512 tokens, as embed_decontam.py
computes it. 10% of the scored documents, split by a seeded hash of the id, are held out to measure the
keep decision against the teacher's own >= 3. The ridge strength is chosen by mean absolute error on a
further 10% of the training part, then the head is refitted on the whole training part. Further pairs of
scored and input files are only checked, not fitted: docs/history/decisions.md item 5 asks whether the classifier
still agrees with the teacher on IndustryCorpus2.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python train_score_classifier.py <data root> <scored.jsonl> <input.jsonl> [<scored> <input>]...
  scored.jsonl comes from score_docs.py; input.jsonl holds the texts it scored.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT, SCORED, INPUT, CHECKS = sys.argv[1], sys.argv[2], sys.argv[3], list(zip(sys.argv[4::2], sys.argv[5::2]))
RIDGES = (1e-5, 1e-4, 1e-3, 1e-2)  # L2 strength per document; the vectors have unit norm, so each feature's square averages 1/1,024
OUT = os.path.join(ROOT, "interim", "scoring")


def held_out(doc_id, salt="holdout-"):
    return int(hashlib.md5((salt + doc_id).encode()).hexdigest()[:8], 16) / 2**32 < 0.1


def load(scored, inp):
    with open(scored, encoding="utf-8") as f:
        s = {r["id"]: r["score"] for r in map(json.loads, f) if r.get("score") is not None}
    with open(inp, encoding="utf-8") as f:
        return s, [d for d in map(json.loads, f) if d["id"] in s]


score, docs = load(SCORED, INPUT)
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "raw", "bge-m3"))
model = AutoModel.from_pretrained(os.path.join(ROOT, "raw", "bge-m3"), torch_dtype=torch.float16).cuda().eval()


@torch.inference_mode()
def embed(texts, batch=64):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        cls = model(**enc).last_hidden_state[:, 0].float()
        out[idx] = torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()
    return out


x = embed([d["text"] for d in docs])
y = np.array([score[d["id"]] for d in docs], dtype=np.float64)
test = np.array([held_out(d["id"]) for d in docs])
xa = np.hstack([x, np.ones((len(x), 1), dtype=np.float32)]).astype(np.float64)
valid = np.array([held_out(d["id"], "valid-") for d in docs]) & ~test


def fit(mask, ridge):
    a = xa[mask]
    penalty = ridge * len(a) * np.eye(a.shape[1])
    penalty[-1, -1] = 0  # the bias is not shrunk
    return np.linalg.solve(a.T @ a + penalty, a.T @ y[mask])


mae = {r: np.abs(xa[valid] @ fit(~test & ~valid, r) - y[valid]).mean() for r in RIDGES}
ridge = min(mae, key=mae.get)
w = fit(~test, ridge)
pred = xa @ w
np.savez(os.path.join(OUT, "classifier.npz"), w=w.astype(np.float32))


def report(mask, pred=pred, y=y):
    keep_p, keep_t = np.rint(pred[mask]) >= 3, y[mask] >= 3
    tp = int((keep_p & keep_t).sum())
    prec = tp / max(int(keep_p.sum()), 1)
    rec = tp / max(int(keep_t.sum()), 1)
    return (f"{int(mask.sum()):,} 篇；平均绝对误差 {np.abs(pred[mask] - y[mask]).mean():.2f}；"
            f"教师判 ≥3 的占 {keep_t.mean():.1%}，分类器判 ≥3 的占 {keep_p.mean():.1%}；"
            f"按 ≥3 与教师一致 {(keep_p == keep_t).mean():.1%}；"
            f"精确率 {prec:.2f}，召回率 {rec:.2f}，F1 {2 * prec * rec / max(prec + rec, 1e-9):.2f}")


lines = ["# FinCorpus 质量分类器", "",
         "由 src/01_data-processing/utils/train_score_classifier.py 生成。bge-m3 向量（前 512 个 token 的 CLS，归一化）上的线性回归，"
         "预测分四舍五入 ≥3 即保留；按 id 的哈希留出 10% 检验。", "",
         "- 正则强度：按训练部分中再留出的 10% 上的平均绝对误差选，" + "，".join(f"{r:g} 为 {m:.3f}" for r, m in mae.items())
         + f"；选 {ridge:g}，再在整个训练部分上重新拟合。",
         f"- 训练部分：{report(~test)}", f"- 留出部分：{report(test)}"]
for scored, inp in CHECKS:
    s, d = load(scored, inp)
    xc = np.hstack([embed([e["text"] for e in d]), np.ones((len(d), 1), dtype=np.float32)]).astype(np.float64)
    yc = np.array([s[e["id"]] for e in d], dtype=np.float64)
    lines.append(f"- 只检查不拟合（`{os.path.basename(scored)}`）：{report(np.ones(len(d), bool), xc @ w, yc)}")
with open(os.path.join(ROOT, "reports", "score_classifier.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
