"""Rescore saved English domain-score outputs with src/04_evaluation/code/num_score.py (research/log/2026-10-29-0442.md).

Each file is num_eval.py's output (id, source, gold, pred, correct, text). The answers are read again from the saved
text; nothing is generated. Only an item whose answer carries an exponent ("3.87688e+06") may change; any other item
whose prediction or verdict changes stops the run before anything is written. A file with changes keeps its old
version beside it as <name>_mantissa.jsonl and gets the new scores under its own name; a file without changes is left
as it is.

Output: one JSON line per file with the items, the old and new accuracy, the answers with an exponent and how many
of them turned right or wrong.

Usage: python rescore_domain_en.py <domain jsonl> [...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from num_score import EXPONENT, NUMBER, correct, parse  # noqa: E402


def has_exponent(text):
    tail = text.rsplit("Answer:", 1)[-1] if "Answer:" in text else ""
    m = NUMBER.search(tail)
    return bool(m and EXPONENT.search(m.group(0)))


plans = []
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    new, exp, to_right, to_wrong = [], 0, 0, 0
    for r in rows:
        pred = parse(r["text"], r["gold"])
        c = correct(pred, r["gold"])
        if has_exponent(r["text"]):
            exp += 1
            to_right += c and not r["correct"]
            to_wrong += r["correct"] and not c
        elif pred != r["pred"] or c != r["correct"]:
            sys.exit(f"{path}: {r['id']} changed without an exponent: {r['pred']} -> {pred}")
        new.append({**r, "pred": pred, "correct": c})
    plans.append((path, rows, new, exp, to_right, to_wrong))

for path, rows, new, exp, to_right, to_wrong in plans:
    changed = any(a != b for a, b in zip(rows, new))
    if changed:
        keep = path[: -len(".jsonl")] + "_mantissa.jsonl"
        if os.path.exists(keep):
            sys.exit(f"{keep} exists; not overwriting it")
        os.rename(path, keep)
        with open(path, "w", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    print(json.dumps({"file": path, "items": n, "old_accuracy": round(100 * sum(r["correct"] for r in rows) / n, 2),
                      "new_accuracy": round(100 * sum(r["correct"] for r in new) / n, 2), "exponent_answers": exp,
                      "turned_right": to_right, "turned_wrong": to_wrong, "rewritten": changed}))
