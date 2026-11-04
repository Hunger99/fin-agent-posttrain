"""Draw the documents the teacher scores: a uniform sample of the FinCorpus scoring pool, by a hash of the id.

docs/history/01_data-processing.md section 2 asks for 50,000 scored documents. The pool is corpus_filter.py's
(research/log/2026-10-25-0855.md); a document is drawn when a salted hash of its id falls under
K / pool size, so the draw is reproducible and spread over the three FinCorpus files by their pool sizes.
Texts are cut to the 3,000 characters score_docs.py sends, to keep the file small. Other corpora can be
drawn the same way by naming their stems under interim/corpus/, as docs/history/decisions.md item 5 does for
IndustryCorpus2.

Usage: python sample_for_scoring.py <data root> <K> <output.jsonl> [stem ...]   (default: the FinCorpus pool)
"""
import glob
import gzip
import hashlib
import json
import os
import sys

ROOT, K, OUT, STEMS = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
POOL = [p for p in sorted(glob.glob(os.path.join(ROOT, "interim", "corpus", "fincorpus_*.jsonl.gz")))
        if not p.endswith("_kept.jsonl.gz")]
if STEMS:
    POOL = [os.path.join(ROOT, "interim", "corpus", s + ".jsonl.gz") for s in STEMS]


def lines():
    for path in POOL:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from f


n = sum(1 for _ in lines())
rate = K / n
kept = 0
with open(OUT, "w", encoding="utf-8") as out:
    for line in lines():
        d = json.loads(line)
        if int(hashlib.md5(("score-" + d["id"]).encode()).hexdigest()[:8], 16) / 2**32 < rate:
            out.write(json.dumps({"id": d["id"], "source": d["source"], "text": d["text"][:3000]}, ensure_ascii=False) + "\n")
            kept += 1
print(f"pool {n:,} documents, drawn {kept:,} at rate {rate:.4%}")
