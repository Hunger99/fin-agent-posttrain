"""Draw the documents for document QA, as a number of chunks per source.

Two kinds of spec:
- ``pool=<K>``: docs/history/decisions.md item 3 (3)'s pilot. K chunks split over FinCorpus announcements, articles
  and news by each source's share of the scoring pool (research/log/2026-10-25-0855.md), from documents the
  teacher scored >= 3. Nothing else is excluded, so a smaller K takes the first chunks of the same order and
  can be compared chunk by chunk with an earlier run.
- ``<stem>=<chunks>`` or ``<stem>=<chunks>:all``: that many chunks from interim/corpus/<stem>.jsonl.gz, from
  documents scored >= 3, or with ``:all`` from any document. Following docs/history/decisions.md item 6 (2), a
  document is skipped when the 8-gram or window decontamination matched one of its paragraphs
  (interim/decontam_corpus/<stem>.spans.jsonl, .vspans.jsonl) or when it has an exam-like paragraph. Both
  files must exist (item 7 (2)); ``:nodecontam`` takes any document and skips only what files exist, as the
  measurement runs before the window check did.

Within a source, documents are taken in the order of a salted hash of their id, with their chunks cut by
doc_qa.py's chunks(), until the quota is met; the last document may add one chunk over it. Full texts are
read from the corpus files, since score_input_*.jsonl cut them to 3,000 characters.

Usage: python sample_doc_qa.py <data root> <output.jsonl> <scored.jsonl>[,<scored.jsonl>...] <spec>...
"""
import collections
import glob
import gzip
import hashlib
import heapq
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "code"))
from corpus_window_decontam import exam_like  # noqa: E402
from doc_qa import chunks  # noqa: E402

ROOT, OUT, SCORED, SPECS = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4:]
SALT = "docqa-20261025"
POOL = [p for p in sorted(glob.glob(os.path.join(ROOT, "interim", "corpus", "fincorpus_*.jsonl.gz")))
        if not p.endswith("_kept.jsonl.gz")]

good = set()
for path in SCORED:
    with open(path, encoding="utf-8") as f:
        good |= {r["id"] for r in map(json.loads, f) if r.get("score") is not None and r["score"] >= 3}


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        yield from map(json.loads, f)


def matched(stem, required):
    ids = set()
    for ext in (".spans.jsonl", ".vspans.jsonl"):
        path = os.path.join(ROOT, "interim", "decontam_corpus", stem + ext)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                ids |= {json.loads(line)["id"] for line in f}
        elif required:  # docs/history/decisions.md item 7 (2): batch input only from documents both checks have seen
            sys.exit(f"{path} is missing; run corpus_decontam.py and corpus_window_decontam.py on {stem} first")
    return ids


def order(doc_id):
    return hashlib.md5((SALT + doc_id).encode()).hexdigest()


def take(texts, quota, out):
    n = 0
    for i in sorted(texts, key=order):
        if n >= quota:
            break
        c = len(chunks(texts[i][1]))
        if c:
            out.write(json.dumps({"id": i, "source": texts[i][0], "text": texts[i][1]}, ensure_ascii=False) + "\n")
            n += c
    return n


total = 0
with open(OUT, "w", encoding="utf-8") as out:
    for spec in SPECS:
        name, value = spec.split("=")
        if name == "pool":
            pool = collections.Counter()
            texts = collections.defaultdict(dict)
            for path in POOL:
                for d in read(path):
                    pool[d["source"]] += 1
                    if d["id"] in good:
                        texts[d["source"]][d["id"]] = (d["source"], d["text"])
            for source in sorted(pool):
                quota = round(int(value) * pool[source] / sum(pool.values()))
                n = take(texts[source], quota, out)
                total += n
                print(f"{source}: pool {pool[source]:,}, scored >=3 {len(texts[source]):,}, quota {quota}, chunks {n}")
        else:
            quota, _, scope = value.partition(":")
            skip, dropped = matched(name, scope != "nodecontam"), collections.Counter()

            def eligible():
                for d in read(os.path.join(ROOT, "interim", "corpus", name + ".jsonl.gz")):
                    if scope not in ("all", "nodecontam") and d["id"] not in good:
                        continue
                    if d["id"] in skip:
                        dropped["decontamination"] += 1
                    elif exam_like(d["text"]):
                        dropped["exam-like"] += 1
                    else:
                        dropped["eligible"] += 1
                        yield d

            # Only the first documents in hash order can be used, so a source as large as IndustryCorpus2
            # is not held in memory; 4 documents per chunk of quota leaves room for documents without a chunk
            first = heapq.nsmallest(4 * int(quota) + 100, eligible(), key=lambda d: order(d["id"]))
            n = take({d["id"]: (d["source"], d["text"]) for d in first}, int(quota), out)
            total += n
            print(f"{name}: {dict(dropped)}, quota {quota}, chunks {n}")
print("total chunks", total)
