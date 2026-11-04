"""Merge the English decontamination checks into interim/decontam/removed_ids.json, the list the pools drop.

research/log/2026-10-28-0040.md: every check that applies to a source removes what it flags, the union over
checks, after the full-chain audit of the same entry measured each check's misses and false flags. Which checks
apply where:
- every source: the second-version 8-gram, plain and format-stripped (decontam_en.py --rule v2), bge-m3 at 0.90
  (embed_decontam_en.py, the vectors of research/log/2026-10-26-2020.md) and Nemotron at 0.72 against MMLU, GSM8K
  and IFEval, where the audit's judges found paraphrases down to that value, and 0.845 against the other reference
  items, where flags below it were other pages of the same reports, which the author keeps (embed_decontam_en.py
  --encoder nemotron --threshold-general 0.72 --threshold-finance 0.845; earlier flags in _nemotron_v1_0845/ and
  _nemotron_v2_072_078/; Tülu 3's come in four shards, and its multi-turn samples have "_turns" flags over every
  user turn);
- the five sources whose prompts carry report excerpts (FinQA and TAT-QA train 80%, DianJin-R1-Data's FinQA part,
  BizBench, Agentar): the excerpt numbers, first and second version together (number_overlap_decontam_en.py and
  --rule v2; the second drops 8 copies the first flags, its 75% rule for items left with 4-7 numbers), and the three
  copy checks of excerpt_copy_decontam_en.py (same question and answer, ordered number windows, same TAT-QA
  table);
- IndustryInstruction: answers that use material the prompt does not give (grounding_filter_en.py).
The excluded sources (Sujet, Finance-Instruct-500k) keep their earlier lists. The previous list is kept as
removed_ids.<tag>.json before it is replaced.

Output: interim/decontam/removed_ids.json ({stem: [ids]}), removed_reasons.json ({stem: {id: [checks]}}) and one
JSON line per source on stdout with the samples removed, the count per check and the count only that check removes.

Usage: python merge_decontam_en.py <English data root> [--keep-old <tag>] [--dry-run]
"""
import collections
import glob
import json
import os
import shutil
import sys

R = sys.argv[1]
DRY = "--dry-run" in sys.argv[2:]
TAG = sys.argv[sys.argv.index("--keep-old") + 1] if "--keep-old" in sys.argv else "before_v2"
D = os.path.join(R, "interim", "decontam")
ALL = ["finqa_train80", "tatqa_train80", "dianjin_r1_finqa", "bizbench", "agentar_deepfinance_en",
       "industryinstruction_finance_en", "finance_alpaca", "doc_qa_en", "tulu3"]
EXCERPT = ALL[:5]
EVERY = ["ngram_v2", "ngram_strip_v2", "embed", "embed_nemotron", "embed_nemotron_turns"]
FOR_EXCERPT = ["numbers", "numbers_v2", "qa_match", "win5", "table_hash"]


def flagged(stem, check):
    paths = [os.path.join(D, f"{stem}.{check}.flags.jsonl")]
    if stem == "tulu3" and check.startswith("embed_nemotron"):
        paths += sorted(glob.glob(os.path.join(D, f"tulu3.shard*of4.{check}.flags.jsonl")))
    out = set()
    for p in paths:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.update(json.loads(line)["id"] for line in f)
    return out


old = json.load(open(os.path.join(D, "removed_ids.json"), encoding="utf-8"))
ungrounded = json.load(open(os.path.join(D, "ungrounded_ids.json"), encoding="utf-8"))
removed, reasons = {}, {}
for stem in ALL:
    checks = EVERY + (FOR_EXCERPT if stem in EXCERPT else [])
    sets = {c: flagged(stem, c) for c in checks}
    missing = [c for c in checks if not sets[c] and not os.path.exists(os.path.join(D, f"{stem}.{c}.flags.jsonl"))
               and not (stem == "tulu3" and glob.glob(os.path.join(D, f"tulu3.shard*of4.{c}.flags.jsonl")))]
    if stem == "industryinstruction_finance_en":
        sets["grounding"] = set(ungrounded.get(stem, []))
    why = collections.defaultdict(list)
    for c, ids in sets.items():
        for i in ids:
            why[i].append(c)
    removed[stem] = sorted(why)
    reasons[stem] = {i: why[i] for i in removed[stem]}
    only = collections.Counter(v[0] for v in why.values() if len(v) == 1)
    print(json.dumps({"source": stem, "removed": len(why), "before": len(old.get(stem, [])),
                      "per_check": {c: len(s) for c, s in sets.items()}, "only_this_check": dict(only),
                      "missing_flag_files": missing}), flush=True)
for stem, ids in old.items():  # sources the pools do not use keep their earlier lists
    removed.setdefault(stem, ids)
if not DRY:
    shutil.copy2(os.path.join(D, "removed_ids.json"), os.path.join(D, f"removed_ids.{TAG}.json"))
    with open(os.path.join(D, "removed_ids.json"), "w", encoding="utf-8") as f:
        json.dump(removed, f, ensure_ascii=False)
    with open(os.path.join(D, "removed_reasons.json"), "w", encoding="utf-8") as f:
        json.dump(reasons, f, ensure_ascii=False)
