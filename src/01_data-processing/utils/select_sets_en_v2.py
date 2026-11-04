"""Drop from the English selection sets every item that the second-version checks tie to a test item.

research/log/2026-10-28-0040.md. The selection sets (processed/eval/{finqa,tatqa}_dev_select.jsonl) are the dev
items left after the 8-gram check of research/log/2026-10-26-2108.md; the full-chain audit found about 8 more that
share a test item's excerpt numbers or its calculation and answer, so a choice made on the selection set would read
the test set. A selection item is dropped when, against a test item of the same dataset:
- excerpt numbers: at least half of the test excerpt's distinctive numbers (not whole numbers under 100 or years
  1900-2035), when it has 8 or more, occur in the item's excerpt; or at least 75% and 4 of them when it has 4-7;
- same calculation: the gold calculations are equal after removing spaces, commas and dollar signs and hold 2 or
  more distinctive numbers; or the questions are equal after lowercasing and keeping letters and digits and the
  answers agree within 1%;
- same page (FinQA): both come from the same page of the same filing.
The previous files are kept as <name>.v1.jsonl.

Output: the two selection files rewritten and one JSON line per dataset with the items kept and dropped by reason.

Usage: python select_sets_en_v2.py <English data root> [--dry-run]
"""
import collections
import json
import os
import re
import shutil
import sys

EN = sys.argv[1]
DRY = "--dry-run" in sys.argv[2:]
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def numbers(text):
    out = set()
    for m in NUM.finditer(text or ""):
        s = m.group(0).replace(",", "").rstrip(".")
        try:
            v = float(s)
        except ValueError:
            continue
        if v != v or v in (float("inf"),) or (v == int(v) and (v < 100 or 1900 <= v <= 2035)):
            continue
        out.add(f"{v:.6g}")
    return out


def qnorm(q):
    return re.sub(r"[^a-z0-9]+", " ", (q or "").lower()).strip()


def same_answer(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()
    return abs(a - b) <= 0.01 * max(abs(a), abs(b), 1e-9)


def calc_key(it):
    """The calculation without spaces, commas and dollar signs, only when it has 2 or more distinctive numbers: a dry
    run matched TAT-QA's year counts ("2019 = 1") and short sums such as 19 - 26 across unrelated questions."""
    c = it.get("calc") or it.get("program") or ""
    return re.sub(r"[\s,$]", "", c) if len(numbers(c.split("=")[0])) >= 2 else None


def page(it):
    return it["id"].split("-", 2)[2].rsplit("-", 1)[0] if it["id"].startswith("finqa-") else None


for ds in ("finqa", "tatqa"):
    test = jl(os.path.join(EN, "processed", "eval", f"{ds}_test.jsonl"))
    path = os.path.join(EN, "processed", "eval", f"{ds}_dev_select.jsonl")
    sel = jl(path)
    tn = [(t, numbers(t.get("context"))) for t in test]
    post = collections.defaultdict(list)
    for k, (_, n) in enumerate(tn):
        for x in n:
            post[x].append(k)
    tcalc = {calc_key(t) for t in test if calc_key(t)}
    tq = collections.defaultdict(list)
    for t in test:
        tq[qnorm(t["question"])].append(t)
    tpage = {page(t) for t in test if page(t)}
    keep, why = [], collections.Counter()
    for r in sel:
        own = numbers(r.get("context"))
        cnt = collections.Counter(k for x in own for k in post.get(x, ()))
        reasons = set()
        for k, c in cnt.items():
            n = len(tn[k][1])
            if (n >= 8 and c / n >= 0.5) or (4 <= n < 8 and c >= 4 and c / n >= 0.75):
                reasons.add("excerpt_numbers")
                break
        if calc_key(r) and calc_key(r) in tcalc:
            reasons.add("same_calculation")
        if any(same_answer(r.get("answer"), t.get("answer")) for t in tq.get(qnorm(r["question"]), ())):
            reasons.add("same_question_and_answer")
        if page(r) and page(r) in tpage:
            reasons.add("same_page")
        if reasons:
            why.update(reasons)
            why["dropped"] += 1
        else:
            keep.append(r)
    print(json.dumps({"set": f"{ds}_dev_select", "before": len(sel), "kept": len(keep), "dropped_by": dict(why)}))
    if not DRY:
        shutil.copy2(path, path.replace(".jsonl", ".v1.jsonl"))
        with open(path, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
