"""Deduplicate the English excerpt-QA sources by question and excerpt, in place of dedup_sft.py's prompt match.

research/log/2026-10-26-2020.md. In FinQA, TAT-QA, DianJin-R1-Data's FinQA part and BizBench a prompt is a report
excerpt followed by one question, and an excerpt carries several questions, so the prompt-level MinHash of
dedup_sft.py (Jaccard >= 0.9) merged different questions about the same excerpt: 2,202 of FinQA's 3,515 samples.
Here two samples are the same item when their questions are the same after normalization and the numbers in
their excerpts overlap by Jaccard >= 0.5; numbers are compared rather than text because each source prints
the excerpt differently. The first in source order stays (FinQA, TAT-QA, DianJin, BizBench), as in dedup_sft.py.

The prompt-level flags of these four sources are kept as <stem>.dedup_prompt.flags.jsonl and replaced by the
flags of this rule; the other sources keep dedup_sft.py's flags.

--doc-qa (research/log/2026-10-27-0256.md): English document QA has the same layout, a document chunk followed by
one question, and a chunk carries up to two questions, so dedup_sft.py --doc-qa merged them the same way. It comes
after the four sources, which are read but whose flags are not rewritten, and only its own flags are replaced.

Usage: python excerpt_dedup_en.py <English data root> [--doc-qa]
"""
import collections
import json
import os
import re
import sys

ROOT = sys.argv[1]
D = os.path.join(ROOT, "interim", "decontam")
SOURCES = ["finqa_train80", "tatqa_train80", "dianjin_r1_finqa", "bizbench"]
DOC_QA = "--doc-qa" in sys.argv[2:]
WRITE = ["doc_qa_en"] if DOC_QA else SOURCES
if DOC_QA:
    SOURCES = SOURCES + ["doc_qa_en"]
NUM = re.compile(r"\d[\d,]*\.?\d*")


def split(stem, user):
    """(excerpt, question) of one prompt, in the layout each source uses."""
    if stem in ("finqa_train80", "tatqa_train80"):
        ctx, q = user.rsplit("\n\nQuestion: ", 1)
        return ctx.split("Report:\n", 1)[-1], q
    if stem == "doc_qa_en":
        ctx, q = user.rsplit("\n\nQuestion: ", 1)
        return ctx.split("Document:\n", 1)[-1], q
    if stem == "dianjin_r1_finqa":
        m = re.search(r"Context:(.*)\n\s*Question:(.*?)(\n\s*Please think step by step.*)?$", user, re.S)
        return (m.group(1), m.group(2)) if m else (user, "")
    ctx, _, q = user.rpartition("\n\n")
    return ctx, q


def norm_q(q):
    return re.sub(r"\s+", " ", q).strip().lower().rstrip("?. ")


def numbers(text):
    return {n.replace(",", "").rstrip(".") for n in NUM.findall(text)}


with open(os.path.join(D, "removed_ids.json"), encoding="utf-8") as f:
    removed = {k: set(v) for k, v in json.load(f).items()}
seen = collections.defaultdict(list)  # normalized question -> [(sample id, excerpt numbers)]
flags, stats = collections.defaultdict(list), {}
for stem in SOURCES:
    n = dup = 0
    with open(os.path.join(ROOT, "processed", "sft", f"{stem}.jsonl"), encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r["id"] in removed[stem]:
                continue
            n += 1
            user = next(m["content"] for m in r["messages"] if m["role"] == "user")
            ctx, q = split(stem, user)
            key, nums = norm_q(q), numbers(ctx)
            match = None
            for other, other_nums in seen[key]:
                union = nums | other_nums
                if not union or len(nums & other_nums) / len(union) >= 0.5:
                    match = other
                    break
            if match:
                dup += 1
                flags[stem].append({"id": r["id"], "matches": [["同题同摘录", match, 1.0]]})
            else:
                seen[key].append((r["id"], nums))
    stats[stem] = {"after_decontam": n, "dedup": dup, "kept": n - dup}
    print(json.dumps({"source": stem, **stats[stem]}), flush=True)
for stem in WRITE:
    path = os.path.join(D, f"{stem}.dedup.flags.jsonl")
    backup = os.path.join(D, f"{stem}.dedup_prompt.flags.jsonl")
    if not os.path.exists(backup):
        os.replace(path, backup)
    with open(path, "w", encoding="utf-8") as f:
        for x in flags[stem]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
with open(os.path.join(ROOT, "reports", "excerpt_dedup_doc_qa_en.json" if DOC_QA else "excerpt_dedup_en.json"), "w",
          encoding="utf-8") as f:
    json.dump(stats, f, indent=1)
