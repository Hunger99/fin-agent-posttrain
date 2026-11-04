"""Flag English training samples that carry most of the numbers of a reference item's report excerpt.

research/log/2026-10-28-0040.md. The 8-gram rules miss an excerpt whose table was rewritten in another layout, such
as JSON, since the words around the numbers change order; the numbers stay. For each reference item with an excerpt
(the FinQA and TAT-QA sets), the distinct numbers of its excerpt are taken (thousands separators, currency and
percent signs dropped; whole numbers under 100 and 1900-2035 left out: a first run that left out only 0-12 flagged
Tülu 3 math problems listing the integers 1 to 100 against tables of two-digit figures); an item with fewer than 8
is skipped. A training sample is flagged when at least half of an item's numbers occur in it and it has at most 4
times as many such numbers as the item (a second run flagged Tülu 3 replies that list hundreds of numbered lines).
Numbers in more than 200 of these items (201, 100, 500 and one-decimal figures such as 1.5) are not counted as
occurring, though they stay in the item's total. Two adjacent years' tables of one company share the prior year's
column, so such pairs are flagged too, which the strictest reading of decontamination accepts.

--rule v2 (the audit of 0040, interim/decontam/audit/full_chain_audit.json) fixes three misses of that rule, found
mostly in BizBench, which rewrites TAT-QA's held-out tables as JSON. Numbers in more than 200 of all the items
(26 of them) are left out of the item's total as well, since otherwise a full copy of an item whose numbers are
mostly common cannot reach half (bizbench-train-7720 carries all 10 numbers of its item, 6 of them common: 0.4).
An item with at least 8 remaining numbers is flagged at half of them, as before. An item with 4-7 is flagged when
at least 75% and at least 4 of them occur: v1 skipped 714 items and with them small tables copied whole
(bizbench-train-6891, 6 numbers), and 3 shared figures also occur by chance (bizbench-train-3487, a Campbell Soup
schedule, has 110, 136 and 140, 3 of the 4 of another company's table). Items with fewer than 4 are skipped. An
item that had 8 or more numbers before common ones were left out now needs 75%, so v2 does not flag 8 samples that
v1 flags (mostly JSON copies of TAT-QA tables whose excerpt adds paragraph numbers); v1's flag files still hold
them. The 4-times cap is dropped: the check is not used on Tülu 3, and on the excerpt sources the cap let through
samples whose reasoning or longer table repeats the item's numbers (agentar-d9f2c0e3 carries the 2013 column of a
FinQA test table in a 45-number sample, one over the cap), at the cost of chance matches it had blocked (11 of the
15 the audit read as false). v2 runs on those five sources only.

Output: interim/decontam/<source>.numbers.flags.jsonl and reports/decontam_en_numbers.json; with --rule v2,
<source>.numbers_v2.flags.jsonl and reports/decontam_en_numbers_v2.json.

Usage: python number_overlap_decontam_en.py <English data root> <Chinese data root> [source ...] [--rule v1|v2]
"""
import collections
import json
import math
import os
import re
import sys

ARGS = sys.argv[1:]
RULE = "v1"
if "--rule" in ARGS:
    i = ARGS.index("--rule")
    RULE = ARGS[i + 1]
    del ARGS[i:i + 2]
if RULE not in ("v1", "v2"):
    sys.exit(f"unknown rule {RULE}")
EN, ZH = ARGS[0], ARGS[1]
THRESHOLD, MIN_NUMBERS, COMMON, MAX_RATIO = 0.5, 8, 200, 4
SMALL_THRESHOLD, MIN_SMALL = 0.75, 4  # v2's items with 4-7 numbers
REF_SETS = [("FinQA test", EN, "processed/eval/finqa_test.jsonl"), ("FinQA dev", EN, "processed/eval/finqa_dev.jsonl"),
            ("TAT-QA test", EN, "processed/eval/tatqa_test.jsonl"), ("TAT-QA dev", EN, "processed/eval/tatqa_dev.jsonl"),
            ("FinQA train 20%", EN, "interim/seed/finqa_train_pref.jsonl"),
            ("TAT-QA train 20%", EN, "interim/seed/tatqa_train_pref.jsonl")]
EXCERPT_SOURCES = ("finqa_train80", "tatqa_train80", "dianjin_r1_finqa", "bizbench", "agentar_deepfinance_en")
SOURCES = [(name, EN, f"processed/sft/{name}.jsonl") for name in
           EXCERPT_SOURCES + ("industryinstruction_finance_en", "finance_alpaca", "doc_qa_en")] + \
          [("tulu3", ZH, "processed/sft/tulu3.jsonl")]
if RULE == "v2":
    SOURCES = SOURCES[:len(EXCERPT_SOURCES)]
if ARGS[2:]:
    SOURCES = [s for s in SOURCES if s[0] in ARGS[2:]]
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(text):
    out = set()
    for m in NUM.finditer(text):
        s = m.group(0).replace(",", "").rstrip(".")
        try:
            v = float(s)
        except ValueError:
            continue
        if not math.isfinite(v):  # a digit string too long for a float, as in Tülu 3's arithmetic
            continue
        if v == int(v) and (v < 100 or 1900 <= v <= 2035):  # whole numbers under 100 occur in any numeric text
            continue
        out.add(f"{v:.6g}")
    return out


def read_jsonl(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


items = [(set_name, it["id"], numbers(it.get("context", ""))) for set_name, root, rel in REF_SETS
         for it in read_jsonl(root, rel)]
common = set()
if RULE == "v2":
    seen = collections.Counter(x for _, _, nums in items for x in nums)
    common = {x for x, c in seen.items() if c > COMMON}
refs, postings, skipped, small = [], collections.defaultdict(list), collections.Counter(), collections.Counter()
for set_name, item_id, nums in items:
    nums = nums - common
    if len(nums) < (MIN_SMALL if RULE == "v2" else MIN_NUMBERS):
        skipped[set_name] += 1
        continue
    if len(nums) < MIN_NUMBERS:
        small[set_name] += 1
    for x in nums:
        postings[x].append(len(refs))
    refs.append((set_name, item_id, len(nums)))

summary = []
for name, root, rel in SOURCES:
    rows = read_jsonl(root, rel)
    flagged, per_set = [], collections.Counter()
    for r in rows:
        text = " ".join(m["content"] + " " + m.get("reasoning", "") for m in r["messages"])
        count, own = collections.Counter(), numbers(text)
        for x in own:
            p = postings.get(x, ())
            if len(p) <= COMMON:
                count.update(p)
        if RULE == "v1":
            # a copied excerpt carries about as many numbers as the item; a list of hundreds of numbered lines does not
            hits = [(refs[i][0], refs[i][1], round(c / refs[i][2], 3)) for i, c in count.items()
                    if c / refs[i][2] >= THRESHOLD and len(own) <= MAX_RATIO * refs[i][2]]
        else:
            hits = [(refs[i][0], refs[i][1], round(c / refs[i][2], 3)) for i, c in count.items() if c >= MIN_SMALL
                    and c / refs[i][2] >= (THRESHOLD if refs[i][2] >= MIN_NUMBERS else SMALL_THRESHOLD)]
        if hits:
            flagged.append({"id": r["id"], "matches": sorted(hits, key=lambda x: -x[2])[:5]})
            per_set.update({s for s, _, _ in hits})
    suffix = "numbers" if RULE == "v1" else "numbers_v2"
    with open(os.path.join(EN, "interim", "decontam", f"{name}.{suffix}.flags.jsonl"), "w", encoding="utf-8") as f:
        for x in flagged:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summary.append({"source": name, "samples": len(rows), "flagged": len(flagged), "by_set": dict(per_set)})
    print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
if RULE == "v1":
    report = {"reference_items": len(refs), "skipped_under_8_numbers": dict(skipped), "sources": summary}
else:
    report = {"rule": "v2", "reference_items": len(refs), "items_with_4_to_7_numbers": dict(small),
              "skipped_under_4_numbers": dict(skipped), "common_numbers": sorted(common, key=float),
              "sources": summary}
with open(os.path.join(EN, "reports", "decontam_en_numbers.json" if RULE == "v1" else "decontam_en_numbers_v2.json"),
          "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
