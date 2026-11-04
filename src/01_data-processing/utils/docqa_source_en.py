"""Write the English document-QA source: IndustryCorpus2's English high tier after the rule filter.

research/log/2026-10-26-2222.md. The thresholds of the Chinese corpus filter (docs/history/01_data-processing.md
section 2), with length counted in words: a document is kept when it has 200-100,000 words (the file's
num_words), at most 30% of its non-empty lines repeat another line, and it holds no Han character. The tier is
already BAAI's high-quality tier, so no scoring classifier runs.

Output lines: {"id", "source", "text"}, the input doc_qa_en.py reads, in the order of a salted hash of the id, so
a run that reads the first K documents takes a sample of the tier rather than its first file; a summary on
stdout.

Usage: python docqa_source_en.py <English data root> <out.jsonl>
"""
import glob
import hashlib
import json
import os
import re
import sys

import pyarrow.parquet as pq

ROOT, OUT = sys.argv[1], sys.argv[2]
HAN = re.compile(r"[一-鿿]")
counts = {"docs": 0, "length": 0, "repeated_lines": 0, "han": 0, "kept": 0}
kept = []
for path in sorted(glob.glob(os.path.join(ROOT, "raw", "industrycorpus2-finance-en", "english", "high", "*.parquet"))):
    stem = os.path.basename(path)[:-8]
    f = pq.ParquetFile(path)
    row = 0
    for g in range(f.num_row_groups):
        for r in f.read_row_group(g, columns=["text", "num_words"]).to_pylist():
            rid, row = f"ic2en-{stem}-{row}", row + 1
            counts["docs"] += 1
            if not 200 <= (r["num_words"] or 0) <= 100_000:
                counts["length"] += 1
                continue
            lines = [x.strip() for x in r["text"].splitlines() if x.strip()]
            if lines and 1 - len(set(lines)) / len(lines) > 0.30:
                counts["repeated_lines"] += 1
                continue
            if HAN.search(r["text"]):
                counts["han"] += 1
                continue
            counts["kept"] += 1
            kept.append((hashlib.md5(("docqa-en-20261026" + rid).encode()).hexdigest(), rid, r["text"]))
kept.sort()
with open(OUT, "w", encoding="utf-8") as out:
    for _, rid, text in kept:
        out.write(json.dumps({"id": rid, "source": "industrycorpus2-finance-en", "text": text}, ensure_ascii=False) + "\n")
print(json.dumps(counts))
