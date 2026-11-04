"""Fix the domain SFT pool once E4 has passed, and write its members for make_mixture.py.

docs/history/decisions.md item 2: each CFLUE question appears once. A DianJin-R1-Data cflue_mcq sample kept by
decontamination and dedup whose nearest CFLUE train item is the same question (split_cflue.py's
interim/cflue/dianjin_mcq_nearest.jsonl) is the twin and keeps its R1 reasoning; that CFLUE item's own
version (cflue_to_sft.py, the 80% part) is left out. Every other kept CFLUE item of the 80% part uses its
own answer and analysis. Document QA (docqa_to_sft.py) is the pool's only synthetic source. Condition 2 of
item 2 fixes the pool only after E4, so this script needs the E4 verdict file the main agent writes after
checking the human review (interim/docqa/e4_result.json with "passed": true).

Output: interim/pools/domain_members.json ({stem: [ids]}, with "synthetic" naming the synthetic stems) and
interim/pools/domain.lock, which make_mixture.py requires before it draws domain samples.

Usage: python lock_domain_pool.py <data root> [--dry-run]
"""
import json
import os
import sys

R, DRY = sys.argv[1], "--dry-run" in sys.argv[2:]
OTHER = ["agentar_deepfinance", "dianjin_r1_oe", "industryinstruction_finance"]
with open(os.path.join(R, "interim", "decontam", "removed_ids.json"), encoding="utf-8") as f:
    removed = {k: set(v) for k, v in json.load(f).items()}


def ids(stem, flags_stem=None):
    drop = set(removed.get(flags_stem or stem, ()))
    path = os.path.join(R, "interim", "decontam", f"{flags_stem or stem}.dedup.flags.jsonl")
    if os.path.exists(path):  # CFLUE train is not deduplicated as SFT (select_sets.py NOT_SFT)
        with open(path, encoding="utf-8") as f:
            drop |= {json.loads(line)["id"] for line in f}
    with open(os.path.join(R, "processed", "sft", f"{stem}.jsonl"), encoding="utf-8") as f:
        return [r["id"] for r in map(json.loads, f) if r["id"] not in drop]


members = {s: ids(s) for s in OTHER}
mcq = ids("dianjin_r1_mcq")
kept_mcq = set(mcq)
with open(os.path.join(R, "interim", "cflue", "dianjin_mcq_nearest.jsonl"), encoding="utf-8") as f:
    twin_items = {n["cflue_id"] for n in map(json.loads, f) if n["id"] in kept_mcq and n["same_item"]}
members["dianjin_r1_mcq"] = mcq
members["cflue_train80"] = [i for i in ids("cflue_train80", "train") if i not in twin_items]
with open(os.path.join(R, "processed", "sft", "doc_qa.jsonl"), encoding="utf-8") as f:
    members["doc_qa"] = [json.loads(line)["id"] for line in f]
counts = {s: len(v) for s, v in members.items()}
print(json.dumps({"counts": counts, "twins_left_out_of_cflue_train80": len(twin_items),
                  "total": sum(counts.values()), "non_synthetic": sum(v for s, v in counts.items() if s != "doc_qa")}))
if DRY:
    sys.exit(0)
verdict = os.path.join(R, "interim", "docqa", "e4_result.json")
if not os.path.exists(verdict) or not json.load(open(verdict, encoding="utf-8")).get("passed"):
    raise SystemExit("E4 has not passed (interim/docqa/e4_result.json); the domain pool stays open")
os.makedirs(os.path.join(R, "interim", "pools"), exist_ok=True)
with open(os.path.join(R, "interim", "pools", "domain_members.json"), "w", encoding="utf-8") as f:
    json.dump({"members": members, "synthetic": ["doc_qa"]}, f)
with open(os.path.join(R, "interim", "pools", "domain.lock"), "w", encoding="utf-8") as f:
    json.dump({"counts": counts, "e4": verdict}, f)
