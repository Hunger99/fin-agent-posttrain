"""Write the continued-pretraining corpus: dedup, cut decontaminated paragraphs, replace personal information, sample to budget.

docs/history/01_data-processing.md sections 2, 3 and 7. For each source in turn:

1. drop documents corpus_dedup.py flags as duplicates (interim/dedup/<stem>.dup.jsonl);
2. cut the paragraphs corpus_decontam.py (8-gram, <stem>.spans.jsonl) or corpus_window_decontam.py
   (embedding, <stem>.vspans.jsonl) matched to an evaluation or held-out item, both under
   interim/decontam_corpus/, dropping a document left under 200 characters, the rule's minimum length;
3. replace personal information (section 2) with a bracketed tag: mainland mobile numbers, 18-digit ID
   numbers that pass the ID checksum, 16-19 digit card numbers that pass the Luhn check, and email
   addresses. The checksums are there because a plain 16-19 digit rule mostly matched long numbers in
   exam questions (research/log/2026-10-25-0754.md);
4. keep a document when a salted hash of its id falls under budget / available tokens, so each source
   is sampled uniformly to its token budget. Sources joined by "+" share one budget and so one rate, which
   draws each in proportion to its available tokens (docs/history/decisions.md item 5 (4)).

Output: processed/cpt/<stem>.jsonl.gz ({"id", "source", "text", "n_tokens"}) and
processed/cpt/MANIFEST.json with each source's budget, counts and seed salt, and how many written
documents were also document-QA sources, drawn or used (interim/docqa/*.jsonl; decisions.md item 5 (5)).

Usage: python build_cpt.py <data root> <stem>[+<stem>...]=<budget tokens> ...
  e.g. fincorpus_kept+industrycorpus2=1125000000 fineweb_edu_zh=375000000
"""
import glob
import gzip
import hashlib
import json
import os
import re
import sys

ROOT = sys.argv[1]
GROUPS = [(a.split("=")[0].split("+"), int(float(a.split("=")[1]))) for a in sys.argv[2:]]
OUT = os.path.join(ROOT, "processed", "cpt")
MIN_LEN = 200  # section 2's minimum document length
SALT = "cpt-20261025"
MOBILE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
IDCARD = re.compile(r"(?<!\d)\d{17}[\dXx](?![\dXx])")
CARD = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def id_ok(s):
    w = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    return "10X98765432"[sum(int(c) * k for c, k in zip(s[:17], w)) % 11] == s[17].upper()


def luhn_ok(s):
    total = 0
    for i, c in enumerate(reversed(s)):
        d = int(c) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def scrub(text, counts):
    def sub(pat, tag, check=None):
        def f(m):
            if check and not check(m.group(0)):
                return m.group(0)
            counts[tag] = counts.get(tag, 0) + 1
            return f"[{tag}]"
        return pat.sub(f, text)
    text = sub(IDCARD, "身份证号", id_ok)
    text = sub(CARD, "银行卡号", luhn_ok)
    text = sub(MOBILE, "手机号")
    return sub(EMAIL, "邮箱")


def load_ids(path, key):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r[key] for r in map(json.loads, f)}


os.makedirs(OUT, exist_ok=True)
docqa = set()
for path in glob.glob(os.path.join(ROOT, "interim", "docqa", "*.jsonl")):
    with open(path, encoding="utf-8") as f:
        # doc_qa.py outputs name chunks ("<doc id>#c<k>"); sample_doc_qa.py inputs name documents ("id")
        docqa |= {r["chunk_id"].split("#")[0] if "chunk_id" in r else r["id"] for r in map(json.loads, f)}
manifest = {"salt": SALT, "sources": {}}
# docs/history/decisions.md item 8 (1): training text keeps each source's punctuation as published
PUNCTUATION = {"industrycorpus2": "中文标点为半角，按原样（decisions.md 第 8 项）"}
for stems, budget in GROUPS:
    plan = {}
    for stem in stems:
        dups = load_ids(os.path.join(ROOT, "interim", "dedup", stem + ".dup.jsonl"), "kept")
        spans = load_ids(os.path.join(ROOT, "interim", "decontam_corpus", stem + ".spans.jsonl"), "spans")
        for i, v in load_ids(os.path.join(ROOT, "interim", "decontam_corpus", stem + ".vspans.jsonl"), "spans").items():
            spans[i] = spans.get(i, []) + v
        # Both checks use the same paragraphs, so a paragraph both matched is cut, and counted, once
        spans = {i: sorted({(a, b) for a, b, *_ in v}) for i, v in spans.items()}
        # First pass: the tokens left after dedup and decontamination, to set the sampling rate
        avail = 0
        with gzip.open(os.path.join(ROOT, "interim", "corpus", stem + ".jsonl.gz"), "rt", encoding="utf-8") as f:
            for d in map(json.loads, f):
                if d["id"] in dups:
                    continue
                cut = sum(b - a for a, b in spans.get(d["id"], []))
                avail += d["n_tokens"] * max(len(d["text"]) - cut, 0) / max(len(d["text"]), 1)
        plan[stem] = (dups, spans, avail)
    rate = min(1.0, budget / max(sum(p[2] for p in plan.values()), 1))
    for stem, (dups, spans, avail) in plan.items():
        c = {"documents_in": 0, "duplicates": 0, "with_cut_paragraphs": 0, "too_short_after_cut": 0,
             "documents_out": 0, "tokens_out": 0, "docqa_sources_out": 0, "pii": {}}
        with gzip.open(os.path.join(ROOT, "interim", "corpus", stem + ".jsonl.gz"), "rt", encoding="utf-8") as f, \
                gzip.open(os.path.join(OUT, stem + ".jsonl.gz"), "wt", encoding="utf-8") as out:
            for d in map(json.loads, f):
                c["documents_in"] += 1
                if d["id"] in dups:
                    c["duplicates"] += 1
                    continue
                if int(hashlib.md5((SALT + d["id"]).encode()).hexdigest()[:8], 16) / 2**32 >= rate:
                    continue
                text, n = d["text"], d["n_tokens"]
                if d["id"] in spans:
                    c["with_cut_paragraphs"] += 1
                    keep, pos = [], 0
                    for a, b in spans[d["id"]]:
                        keep.append(text[pos:a])
                        pos = b
                    keep.append(text[pos:])
                    new = "".join(keep)
                    n = round(n * len(new) / max(len(text), 1))  # tokens scale with the characters kept
                    text = new
                    if len(text) < MIN_LEN:
                        c["too_short_after_cut"] += 1
                        continue
                text = scrub(text, c["pii"])
                out.write(json.dumps({"id": d["id"], "source": d["source"], "text": text, "n_tokens": n}, ensure_ascii=False) + "\n")
                c["documents_out"] += 1
                c["tokens_out"] += n
                c["docqa_sources_out"] += d["id"] in docqa
        c.update(group="+".join(stems), group_budget=budget, available_tokens=round(avail), rate=rate,
                 punctuation=PUNCTUATION.get(stem, "中文标点为全角，按原样"))
        manifest["sources"][stem] = c
        print(stem, json.dumps(c, ensure_ascii=False), flush=True)
with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)
