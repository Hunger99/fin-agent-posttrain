"""Deduplicate the corpora, exactly by MD5 and nearly by MinHash-LSH, per docs/history/01_data-processing.md section 2.

Section 2's rule: exact duplicates by the MD5 of the normalized text; near duplicates by MinHash-LSH over
character 5-grams with 256 hash functions at Jaccard >= 0.8, with FinCorpus and IndustryCorpus2 merged for
a second pass. The work is split in two modes, because the signatures cost most of the time and can be
computed before the classifier decides which FinCorpus documents remain:

- ``sign <stem>``: for interim/corpus/<stem>.jsonl.gz, the MD5 of the normalized text (NFKC, lower case,
  whitespace collapsed) and a 256-value MinHash signature, saved under interim/dedup/.
- ``cluster <stem>...``: one pass over the given stems in that order. The first document of each cluster
  stays, so the order sets which source keeps a shared document. LSH uses 32 bands of 8 rows: a pair at
  Jaccard 0.8 shares a band with probability 0.997, at 0.6 with 0.42; candidates are then checked on all
  256 positions. Clusters are connected components, as in dedup_sft.py. Writes interim/dedup/<stem>.dup.jsonl
  (removed id, kept id, kind, estimated Jaccard) and reports/corpus_dedup_<stems>.md.

The MinHash code is dedup_sft.py's; the scripts run with ``python -I`` and do not import each other.

Usage: python corpus_dedup.py <data root> sign <stem> | cluster <stem>...
"""
import collections
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import sys
import unicodedata

import numpy as np

ROOT, MODE, STEMS = sys.argv[1], sys.argv[2], sys.argv[3:]
DIR = os.path.join(ROOT, "interim", "dedup")
SHINGLE, PERMS, BANDS = 5, 256, 32
JACCARD = 0.8  # section 2
BUCKET_CHECK = 50  # within one band bucket, each member is checked against at most this many earlier ones
SEEDS = np.random.default_rng(20261025).integers(0, 2**63, size=PERMS, dtype=np.uint64)  # as dedup_sft.py


def mix(h):
    h = h ^ (h >> np.uint64(30))
    h = h * np.uint64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> np.uint64(27))
    h = h * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def signature(key):
    cp = np.frombuffer(key.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    if len(cp) < SHINGLE:
        grams = np.array([int.from_bytes(hashlib.md5(key.encode()).digest()[:8], "little")], dtype=np.uint64)
    else:
        k = len(cp) - SHINGLE + 1
        grams = np.zeros(k, dtype=np.uint64)
        for j in range(SHINGLE):
            grams = grams * np.uint64(0x100000001B3) + cp[j:j + k]
        grams = np.unique(mix(grams))
    sig = np.full(PERMS, np.iinfo(np.uint64).max, dtype=np.uint64)
    for i in range(0, len(grams), 4096):
        sig = np.minimum(sig, mix(grams[i:i + 4096, None] ^ SEEDS[None, :]).min(0))
    return sig.astype(np.uint32)


def work(lines):
    out = []
    for line in lines:
        d = json.loads(line)
        key = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", d["text"])).strip().lower()
        out.append((d["id"], hashlib.md5(key.encode()).hexdigest(), d["n_tokens"], signature(key)))
    return out


def batches(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        batch = []
        for line in f:
            batch.append(line)
            if len(batch) == 1000:
                yield batch
                batch = []
        if batch:
            yield batch


os.makedirs(DIR, exist_ok=True)
if MODE == "sign":
    stem = STEMS[0]
    ids, md5s, toks, sigs = [], [], [], []
    with multiprocessing.Pool(24) as pool:  # 24 of 96 cores; others use the host too
        for part in pool.imap(work, batches(os.path.join(ROOT, "interim", "corpus", stem + ".jsonl.gz")), chunksize=1):
            for i, m, t, s in part:
                ids.append(i)
                md5s.append(m)
                toks.append(t)
                sigs.append(s)
            if len(ids) % 200000 < 1000:
                print(f"{stem}: {len(ids):,} signed", flush=True)
    np.save(os.path.join(DIR, stem + ".sig.npy"), np.stack(sigs))
    np.save(os.path.join(DIR, stem + ".tokens.npy"), np.array(toks, dtype=np.int64))
    with open(os.path.join(DIR, stem + ".ids.jsonl"), "w", encoding="utf-8") as f:
        for i, m in zip(ids, md5s):
            f.write(json.dumps({"id": i, "md5": m}) + "\n")
    print(f"{stem}: {len(ids):,} documents signed")
else:
    ids, md5s, src, sig, toks = [], [], [], [], []
    for s, stem in enumerate(STEMS):
        with open(os.path.join(DIR, stem + ".ids.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        ids += [r["id"] for r in rows]
        md5s += [r["md5"] for r in rows]
        src += [s] * len(rows)
        sig.append(np.load(os.path.join(DIR, stem + ".sig.npy")))
        toks.append(np.load(os.path.join(DIR, stem + ".tokens.npy")))
    sig, toks, src = np.concatenate(sig), np.concatenate(toks), np.array(src)
    first, dup = {}, {}
    for i, h in enumerate(md5s):
        if h in first:
            dup[i] = (first[h], "精确重复", 1.0)
        else:
            first[h] = i
    uniq = np.array(sorted(first.values()))
    parent = np.arange(len(ids))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rows = PERMS // BANDS
    for b in range(BANDS):
        band = np.ascontiguousarray(sig[uniq, b * rows:(b + 1) * rows]).view(f"V{rows * 4}").ravel()
        _, inv, counts = np.unique(band, return_inverse=True, return_counts=True)
        order = np.argsort(inv, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts)])
        for g in np.nonzero(counts > 1)[0]:
            members = uniq[order[starts[g]:starts[g + 1]]]
            for j in range(1, len(members)):
                a = int(members[j])
                for c in members[max(0, j - BUCKET_CHECK):j]:
                    ra, rc = find(a), find(int(c))
                    if ra != rc and np.mean(sig[a] == sig[c]) >= JACCARD:
                        parent[max(ra, rc)] = min(ra, rc)
        print(f"band {b}", flush=True)
    for i in uniq.tolist():
        root = find(i)
        if root != i:
            dup[i] = (root, "近似重复", round(float(np.mean(sig[i] == sig[root])), 3))
    for i, (j, kind, est) in list(dup.items()):
        if kind == "精确重复" and find(j) != j:
            dup[i] = (find(j), kind, est)
    stats = collections.defaultdict(collections.Counter)
    out = {stem: open(os.path.join(DIR, stem + ".dup.jsonl"), "w", encoding="utf-8") for stem in STEMS}
    for i, (j, kind, est) in dup.items():
        stem = STEMS[src[i]]
        out[stem].write(json.dumps({"id": ids[i], "kept": ids[j], "kind": kind, "est": est}, ensure_ascii=False) + "\n")
        stats[stem][kind] += 1
        stats[stem][kind + "token"] += int(toks[i])
        if src[i] != src[j]:
            stats[stem]["跨来源"] += 1
    for f in out.values():
        f.close()
    lines = ["# 语料去重", "", "由 src/01_data-processing/utils/corpus_dedup.py 生成。精确重复为规范化文本的 MD5 相同；近似重复为按字 5-gram 的 MinHash"
             f"（256 个哈希，32 段 × 8 行）估计的 Jaccard ≥{JACCARD}，按连通分量合并。来源按下表的顺序，每组留最先的一篇。", "",
             "| 来源 | 篇数 | token | 精确重复 | 近似重复 | 其中与另一来源重复 | 去重后 token |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for s, stem in enumerate(STEMS):
        n, t, st = int((src == s).sum()), int(toks[src == s].sum()), stats[stem]
        lines.append(f"| {stem} | {n:,} | {t / 1e9:.2f}B | {st['精确重复']:,} | {st['近似重复']:,} | {st['跨来源']:,} "
                     f"| {(t - st['精确重复token'] - st['近似重复token']) / 1e9:.2f}B |")
    with open(os.path.join(ROOT, "reports", f"corpus_dedup_{'_'.join(STEMS)}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
