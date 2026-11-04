"""Fix the English domain SFT pool in two parts, and write its members for make_mixture.py.

research/log/2026-10-27-0248.md. The pool's seven non-synthetic sources (Sujet and Finance-Instruct-500k left
out, research/log/2026-10-26-2006.md) were decontaminated and deduplicated before any synthesis, so their
members do not depend on E4 and are fixed first, which lets R3-synthetic (make_mixture.py --no-synthetic)
be drawn before the human review. The synthetic source, English document QA (docqa_to_sft_en.py), joins
only once E4 has passed (interim/docqa/e4_result.json with "passed": true); this second run checks that the
non-synthetic members it would write are the ones already fixed. Document QA is deduplicated after every other
source, so its arrival removes none of their samples.

Output: interim/pools/domain_members.json ({"members": {stem: [ids]}, "synthetic": [stems], "synthetic_fixed":
bool}) and interim/pools/domain.lock, which make_mixture.py requires before it draws domain samples; with
"synthetic_fixed": false it draws only with --no-synthetic.

Usage: python lock_domain_pool_en.py <English data root> [--without-synthetic] [--dry-run]
"""
import json
import os
import sys

R = sys.argv[1]
WITHOUT, DRY = "--without-synthetic" in sys.argv[2:], "--dry-run" in sys.argv[2:]
NON_SYNTHETIC = ["finqa_train80", "tatqa_train80", "dianjin_r1_finqa", "bizbench", "agentar_deepfinance_en",
                 "industryinstruction_finance_en", "finance_alpaca"]
SYNTHETIC = "doc_qa_en"
P = os.path.join(R, "interim", "pools")
with open(os.path.join(R, "interim", "decontam", "removed_ids.json"), encoding="utf-8") as f:
    removed = {k: set(v) for k, v in json.load(f).items()}
# answers that use material the prompt does not give (grounding_filter_en.py, research/log/2026-10-27-2138.md)
UNGROUNDED = os.path.join(R, "interim", "decontam", "ungrounded_ids.json")
if os.path.exists(UNGROUNDED):
    with open(UNGROUNDED, encoding="utf-8") as f:
        for k, v in json.load(f).items():
            removed[k] = removed.get(k, set()) | set(v)


def ids(stem):
    drop = set(removed.get(stem, ()))
    with open(os.path.join(R, "interim", "decontam", f"{stem}.dedup.flags.jsonl"), encoding="utf-8") as f:
        drop |= {json.loads(line)["id"] for line in f}
    with open(os.path.join(R, "processed", "sft", f"{stem}.jsonl"), encoding="utf-8") as f:
        return [r["id"] for r in map(json.loads, f) if r["id"] not in drop]


members = {s: ids(s) for s in NON_SYNTHETIC}
if not WITHOUT:
    verdict = os.path.join(R, "interim", "docqa", "e4_result.json")
    if not DRY and (not os.path.exists(verdict) or not json.load(open(verdict, encoding="utf-8")).get("passed")):
        raise SystemExit("E4 has not passed (interim/docqa/e4_result.json); run with --without-synthetic")
    members[SYNTHETIC] = ids(SYNTHETIC)
counts = {s: len(v) for s, v in members.items()}
print(json.dumps({"counts": counts, "total": sum(counts.values()),
                  "non_synthetic": sum(v for s, v in counts.items() if s != SYNTHETIC)}))
if DRY:
    sys.exit(0)
old = os.path.join(P, "domain_members.json")
if os.path.exists(old):
    with open(old, encoding="utf-8") as f:
        fixed = json.load(f)["members"]
    for s in NON_SYNTHETIC:
        if fixed[s] != members[s]:
            raise SystemExit(f"{s}: members differ from the fixed non-synthetic part; nothing written")
os.makedirs(P, exist_ok=True)
with open(old, "w", encoding="utf-8") as f:
    json.dump({"members": members, "synthetic": [SYNTHETIC], "synthetic_fixed": not WITHOUT}, f)
with open(os.path.join(P, "domain.lock"), "w", encoding="utf-8") as f:
    json.dump({"counts": counts, "synthetic_fixed": not WITHOUT}, f)
