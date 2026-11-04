"""Assemble the hard finance-agent dataset of composition 甲 (research/log/2026-10-31-0150.md, 23:3x UTC).

Sources, each split taken whole from files that were fixed before any model ran on their dev or test parts, except
the two choices below, which use dev results and the generator alone:
  - the hard tier without div_event_study (the official Qwen3.5-4B made 19/20 of its dev tasks): market_tasks_hard_*,
    plus the 700 extra train tasks of market_tasks_hard_train_extra.jsonl;
  - the 3 tier-1 templates the official model did worst on in dev (rolling_vol, extreme_period, xbrl_ratio):
    market_tasks_{train,dev,test}.jsonl;
  - DocFinQA test: the first 200 of the 920 test questions ordered by sha256("docfinqa-test-300|" + task id), the
    order that already fixed the 300-question subset (research/log/2026-10-28-0557.md), so the 200 are its first 200;
  - FinSearchComp: the 98 offline-answerable Global T2/T3 questions (external/finsearchcomp_ext_test.jsonl), test only.

Writes <out>/{train,dev,test}_market.jsonl, test_docfinqa.jsonl, test_finsearchcomp.jsonl and manifest.json (counts
per source, template and split, sha256 of every input and output file).

Usage: python3 -B build_hardset.py --market-dir <market_v1> --docfinqa-test <tasks_docfinqa_test.jsonl> --out <dir>
"""
import argparse
import collections
import hashlib
import json
import os

HARD_DROP = {"div_event_study"}
TIER1_WEAK = ("rolling_vol", "extreme_period", "xbrl_ratio")
DOCFINQA_N = 200


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write(path, rows):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-dir", required=True)
    ap.add_argument("--docfinqa-test", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = os.path.join(a.market_dir, "tasks")
    inputs = {
        "hard_train": os.path.join(T, "market_tasks_hard_train.jsonl"),
        "hard_train_extra": os.path.join(T, "market_tasks_hard_train_extra.jsonl"),
        "hard_dev": os.path.join(T, "market_tasks_hard_dev.jsonl"),
        "hard_test": os.path.join(T, "market_tasks_hard_test.jsonl"),
        "tier1_train": os.path.join(T, "market_tasks_train.jsonl"),
        "tier1_dev": os.path.join(T, "market_tasks_dev.jsonl"),
        "tier1_test": os.path.join(T, "market_tasks_test.jsonl"),
        "docfinqa_test": a.docfinqa_test,
        "finsearchcomp": os.path.join(a.market_dir, "external", "finsearchcomp_ext_test.jsonl"),
    }
    data = {k: jsonl(p) for k, p in inputs.items()}
    hard = lambda rows: [t for t in rows if t["template"] not in HARD_DROP]
    weak = lambda rows: [t for t in rows if t["template"] in TIER1_WEAK]
    dq = sorted(data["docfinqa_test"], key=lambda t: hashlib.sha256(("docfinqa-test-300|" + t["task_id"]).encode())
                .hexdigest())[:DOCFINQA_N]
    out = {
        "train_market": hard(data["hard_train"]) + data["hard_train_extra"] + weak(data["tier1_train"]),
        "dev_market": hard(data["hard_dev"]) + weak(data["tier1_dev"]),
        "test_market": hard(data["hard_test"]) + weak(data["tier1_test"]),
        "test_docfinqa": dq,
        "test_finsearchcomp": data["finsearchcomp"],
    }
    ids = collections.Counter(t["task_id"] for rows in out.values() for t in rows)
    assert max(ids.values()) == 1, "a task id appears twice"
    os.makedirs(a.out, exist_ok=True)
    files = {}
    for name, rows in out.items():
        p = os.path.join(a.out, name + ".jsonl")
        write(p, rows)
        files[name + ".jsonl"] = sha(p)
    counts = {name: dict(sorted(collections.Counter(t["template"] for t in rows).items())) for name, rows in out.items()}
    manifest = {
        "totals": {name: len(rows) for name, rows in out.items()},
        "total": sum(len(rows) for rows in out.values()),
        "counts_by_template": counts,
        "hard_dropped": sorted(HARD_DROP), "tier1_weak": list(TIER1_WEAK), "docfinqa_n": DOCFINQA_N,
        "inputs_sha256": {k: sha(p) for k, p in inputs.items()}, "inputs": inputs, "files_sha256": files,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest["totals"]), manifest["total"])


if __name__ == "__main__":
    main()
