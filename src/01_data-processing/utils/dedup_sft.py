"""Deduplicate the SFT sources by user prompt and count personal information, per docs/history/01_data-processing.md section 2.

Only samples that decontamination keeps take part (``interim/decontam/removed_ids.json`` from
select_sets.py), so a cluster's kept sample is never one that decontamination drops anyway.

- **Key:** all user turns of a sample, joined. DianJin-R1-Data's fixed instruction lines are dropped, as in
  embed_decontam.py; then NFKC, OpenCC traditional to simplified, HTML tags and control characters
  removed, whitespace collapsed, lower case. The key only decides what counts as the same prompt; the
  samples keep their own text. All user turns are used, not the first only, so that multi-turn chats
  that open with the same greeting are not merged.
- **Exact:** the same MD5 of the key.
- **Near:** MinHash over character 5-grams with 256 hash functions, LSH in 16 bands of 16 rows, and an
  estimated Jaccard >= 0.9 (section 2's SFT threshold) checked on the full signatures.
- **Which sample stays:** the first in the order of ``SOURCES`` and then of each file, so domain sources come
  before general ones. Every other sample of a cluster is flagged against the kept one.
- **Personal information:** mainland mobile numbers, 18-digit ID numbers, 16-19 digit card numbers and email
  addresses are only counted, in every turn of the kept samples; replacing them is left to the mixture
  build (section 7). Examples in the report are masked.

English mode (research/log/2026-10-26-2020.md): ``en`` after the root dedups the English sources of
convert_en.py with the same rules; the instruction line convert_en.py puts before FinQA and TAT-QA samples and
DianJin-R1-Data's English instruction line are dropped from the key, and FinQA and TAT-QA come first, so a
FinQA item that also arrives through DianJin-R1-Data or BizBench keeps its gold calculation. Tülu 3 is read
through a link to the Chinese root's file.

Usage: python dedup_sft.py $FP_DATA_ZH [en [--doc-qa]]
"""
import collections
import hashlib
import json
import multiprocessing
import os
import re
import sys
import unicodedata

import numpy as np
import opencc

ROOT = sys.argv[1]
SHINGLE, PERMS, BANDS = 5, 256, 16  # section 2: character 5-grams, 256 hashes; 16 bands x 16 rows
JACCARD = 0.9  # section 2: near-duplicate threshold for SFT prompts
# A pair at Jaccard 0.9 shares a band with probability 1 - (1 - 0.9**16)**16 = 0.96; at 0.8, 0.37; at 0.7,
# 0.05. Candidates are then checked on all 256 positions, so a lower band hit rate costs only checks.
BUCKET_CHECK = 50  # within one band bucket, each member is checked against at most this many earlier ones
SOURCES = [("DianJin-R1-Data cflue_mcq", "dianjin_r1_mcq", "领域"), ("DianJin-R1-Data cflue_oe", "dianjin_r1_oe", "领域"),
           ("Agentar-DeepFinance-100K", "agentar_deepfinance", "领域"),
           ("IndustryInstruction 金融", "industryinstruction_finance", "领域"),
           ("alpaca-zh", "alpaca_zh", "通用"), ("smoltalk-chinese", "smoltalk_chinese", "通用"),
           ("Tülu 3 SFT", "tulu3", "通用")]
WRAPPER = re.compile(r"^(假设你是一位金融行业专家，请回答下列问题。|注意：.*|请一步步思考。)$")
if len(sys.argv) > 2 and sys.argv[2] == "en":
    SOURCES = [("FinQA train 80%", "finqa_train80", "领域"), ("TAT-QA train 80%", "tatqa_train80", "领域"),
               ("DianJin-R1-Data fin_qa", "dianjin_r1_finqa", "领域"), ("BizBench train", "bizbench", "领域"),
               ("Agentar-DeepFinance-100K (English)", "agentar_deepfinance_en", "领域"),
               ("IndustryInstruction 金融 (English)", "industryinstruction_finance_en", "领域"),
               ("Sujet-Finance-Instruct-177k", "sujet_finance", "领域"), ("finance-alpaca", "finance_alpaca", "领域"),
               ("Finance-Instruct-500k", "finance_instruct_500k", "领域"), ("Tülu 3 SFT", "tulu3", "通用")]
    WRAPPER = re.compile(r"^(Answer the question from the excerpt of a company's annual report\..*"
                         r"|Answer the question from the excerpt of a financial document\..*"
                         r"|Please answer the given financial question based on the context\.)$")
# --doc-qa (research/log/2026-10-27-0248.md): English document QA, synthesized after the other sources were
# deduplicated, comes last, behind Tülu 3, and only its flags are written. Rerunning the whole English dedup would
# overwrite the four excerpt-QA sources' flags, which excerpt_dedup_en.py replaced, and could change the fixed
# non-synthetic pool; with document QA last, no other source's sample is the one a cluster drops.
DOC_QA = "--doc-qa" in sys.argv[3:]
if DOC_QA:
    SOURCES = SOURCES + [("English document QA (synthetic)", "doc_qa_en", "领域")]
TAG = re.compile(r"<[^<>\n]{1,200}>")
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏﻿]")
PII = {"手机号": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "身份证号": re.compile(r"(?<!\d)\d{17}[\dXx](?![\dXx])"),
       "银行卡号": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
       # ASCII only: with \w, the Chinese text before an address was taken as its user name
       "邮箱": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")}
rng = np.random.default_rng(20261025)
SEEDS = rng.integers(0, 2**63, size=PERMS, dtype=np.uint64)
_cc = None


def mix(h):
    """splitmix64 finalizer, applied elementwise to uint64."""
    h = h ^ (h >> np.uint64(30))
    h = h * np.uint64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> np.uint64(27))
    h = h * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def normalize(text):
    global _cc
    if _cc is None:  # built in each worker process
        _cc = opencc.OpenCC("t2s")
    text = _cc.convert(unicodedata.normalize("NFKC", text))
    text = CONTROL.sub("", TAG.sub(" ", text))
    return re.sub(r"\s+", " ", text).strip().lower()


def signature(key):
    cp = np.frombuffer(key.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    if len(cp) < SHINGLE:
        grams = np.array([int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "little")], dtype=np.uint64)
    else:
        k = len(cp) - SHINGLE + 1
        grams = np.zeros(k, dtype=np.uint64)
        for j in range(SHINGLE):
            grams = grams * np.uint64(0x100000001B3) + cp[j:j + k]
        grams = np.unique(mix(grams))
    sig = np.full(PERMS, np.iinfo(np.uint64).max, dtype=np.uint64)
    for i in range(0, len(grams), 4096):  # bounds memory for long prompts
        sig = np.minimum(sig, mix(grams[i:i + 4096, None] ^ SEEDS[None, :]).min(0))
    return sig.astype(np.uint32)  # the low 32 bits are enough to compare positions


def work(prompts):
    keys = [normalize(p) for p in prompts]
    return [hashlib.md5(k.encode("utf-8")).hexdigest() for k in keys], np.stack([signature(k) for k in keys])


def prompt_of(r):
    users = [m["content"] for m in r["messages"] if m["role"] == "user"]
    text = "\n".join(users)
    return "\n".join(line for line in text.splitlines() if not WRAPPER.match(line.strip()))


def mask(s):
    if "@" in s:
        return s[0] + "***@" + s.split("@", 1)[1]
    return s[:3] + "*" * (len(s) - 5) + s[-2:]


with open(os.path.join(ROOT, "interim", "decontam", "removed_ids.json"), encoding="utf-8") as f:
    removed = {k: set(v) for k, v in json.load(f).items()}
ids, src_of, prompts = [], [], []
pii = collections.defaultdict(collections.Counter)
pii_ex = collections.defaultdict(list)
for s, (name, stem, _) in enumerate(SOURCES):
    with open(os.path.join(ROOT, "processed", "sft", stem + ".jsonl"), encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r["id"] in removed[stem]:
                continue
            ids.append(r["id"])
            src_of.append(s)
            prompts.append(prompt_of(r))
            text = " ".join(m["content"] + " " + m.get("reasoning", "") for m in r["messages"])
            for kind, pat in PII.items():
                hits = pat.findall(text)
                if hits:
                    pii[name][kind] += 1
                    if len(pii_ex[(name, kind)]) < 2:
                        pii_ex[(name, kind)].append(mask(hits[0]))
    print(name, sum(x == s for x in src_of), flush=True)
src_of = np.array(src_of)

chunks = [prompts[i:i + 4000] for i in range(0, len(prompts), 4000)]
with multiprocessing.Pool(16) as pool:  # 16 of the host's 96 cores, leaving room for other users
    parts = pool.map(work, chunks)
md5 = [h for p in parts for h in p[0]]
sig = np.concatenate([p[1] for p in parts])
del parts, chunks
print("signatures", sig.shape, flush=True)

# Exact duplicates: every later sample with an earlier sample's MD5
first, dup = {}, {}
for i, h in enumerate(md5):
    if h in first:
        dup[i] = (first[h], "精确重复", 1.0)
    else:
        first[h] = i
uniq = np.array(sorted(first.values()))

# Near duplicates among the first occurrences, joined with union-find rooted at the earliest index
parent = {int(i): int(i) for i in uniq}


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
                c = int(c)
                ra, rc = find(a), find(c)
                if ra == rc:
                    continue
                est = float(np.mean(sig[a] == sig[c]))
                if est >= JACCARD:
                    parent[max(ra, rc)] = min(ra, rc)
    print("band", b, flush=True)
for i in uniq.tolist():
    root = find(i)
    if root != i:
        dup[i] = (root, "近似重复", round(float(np.mean(sig[i] == sig[root])), 3))
# An exact duplicate of a near duplicate points at the cluster's kept sample
for i, (j, kind, est) in list(dup.items()):
    if kind == "精确重复" and find(j) != j:
        dup[i] = (find(j), kind, est)

flags = collections.defaultdict(list)
stats = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
for i, (j, kind, est) in dup.items():
    name, stem, pool_i = SOURCES[src_of[i]]
    flags[stem].append({"id": ids[i], "matches": [[kind, ids[j], est]]})
    stats[name][kind] += 1
    stats[name]["跨来源" if src_of[i] != src_of[j] else "同来源"] += 1
    if SOURCES[src_of[j]][2] != pool_i:
        stats[name]["跨池"] += 1
    if kind == "近似重复" and len(examples[name]) < 200:
        examples[name].append((est, i, j))
for _, stem, _ in SOURCES:
    if DOC_QA and stem != "doc_qa_en":
        continue
    with open(os.path.join(ROOT, "interim", "decontam", f"{stem}.dedup.flags.jsonl"), "w", encoding="utf-8") as f:
        for r in flags[stem]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

lines = ["# SFT 按用户提示去重", "",
         "由 src/01_data-processing/utils/dedup_sft.py 生成。只对去污染后留下的样本去重。提示取全部用户轮，规范化（NFKC、繁转简、"
         "去 HTML 标签和控制字符、合并空白、转小写）后：MD5 相同为精确重复；按字 5-gram 的 MinHash（256 个哈希，16 段 × 16 行）"
         f"估计的 Jaccard ≥{JACCARD} 为近似重复。每组重复留下按来源顺序（领域来源在前）和文件顺序最先的一条。", "",
         "| 来源 | 去污染后 | 精确重复 | 近似重复 | 其中与另一来源重复 | 其中与另一个池重复 | 去重后 |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
for s, (name, stem, _) in enumerate(SOURCES):
    n = int((src_of == s).sum())
    st = stats[name]
    lines.append(f"| {name} | {n:,} | {st['精确重复']:,} | {st['近似重复']:,} | {st['跨来源']:,} | {st['跨池']:,} "
                 f"| {n - st['精确重复'] - st['近似重复']:,} |")
lines += ["", "## 近似重复的例子", "", "每个来源取估计 Jaccard 最低的 2 对，提示截到前 100 字。", ""]
for name, ex in examples.items():
    for est, i, j in sorted(ex)[:2]:
        lines.append(f"- {name}，{est}：`{ids[i]}` {' '.join(prompts[i].split())[:100]} ‖ `{ids[j]}` {' '.join(prompts[j].split())[:100]}")
lines += ["", "## 个人信息（只计数，未替换）", "",
          "每格为含至少一处该类信息的样本数，按全部轮次的内容和推理计。规则是简单的正则：16–19 位数字也会命中题目中的长数字，"
          "18 位数字会同时计入身份证号和银行卡号，所以这些数是上限。例子已打码。", "",
          "| 来源 | " + " | ".join(PII) + " |", "| --- | " + " | ".join("---" for _ in PII) + " |"]
for name, _, _ in SOURCES:
    lines.append(f"| {name} | " + " | ".join(f"{pii[name][k]:,}" for k in PII) + " |")
lines += [""] + [f"- {name} {kind}：" + "，".join(ex) for (name, kind), ex in pii_ex.items()]
with open(os.path.join(ROOT, "reports", "dedup_sft_doc_qa_en.md" if DOC_QA else "dedup_sft.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[:16]))
