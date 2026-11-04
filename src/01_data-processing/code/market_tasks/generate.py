"""Generate the market-environment task splits from the market_v1 snapshot.

Usage:
  PY=$FP_VENVS/mkt/bin/python
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $PY -B generate.py --cache-dir <local dir>
      [--tier base|hard] [--seed 20261031] [--only t1,t2] [--sizes 200,20,30] [--dry] [--full-check]

Writes $M_DIR/tasks/market_tasks_{train,dev,test}.jsonl, manifest.json (counts,
seed, file hashes, snapshot hash, quality-rule removals, independent-check
agreement) and spotcheck.txt (3 tasks per template with the rows used).

Version 2 (research/log/2026-10-31-0150.md, the hard tier): --tier hard generates
the difficulty-4 and -5 templates of templates_hard.py instead, with task ids
mk-hard-<split>-<template>-NNNN, into market_tasks_hard_{train,dev,test}.jsonl,
manifest_hard.json and spotcheck_hard.txt (default sizes 150,20,30, half of each
split at either difficulty). The default tier's output is unchanged.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import DB, OUT_DIR, SPLITS, Skip, Snapshot, dumps, sha256_file  # noqa: E402
from sec_index import SecIndex  # noqa: E402
from templates import TEMPLATES, Ctx  # noqa: E402
from templates_hard import HARD_TEMPLATES  # noqa: E402
import verify  # noqa: E402


def sub_rng(seed, *parts):
    h = hashlib.sha256(("|".join([str(seed)] + [str(p) for p in parts])).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


TIERS = {  # templates, difficulty levels, task-id prefix, file prefix, manifest, spotcheck file, default sizes
    "base": (TEMPLATES, (1, 2, 3), "mk", "market_tasks", "manifest.json", "spotcheck.txt", "200,20,30"),
    "hard": (HARD_TEMPLATES, (4, 5), "mk-hard", "market_tasks_hard", "manifest_hard.json", "spotcheck_hard.txt",
             "150,20,30"),
}


def per_level(n, levels):
    """Targets per difficulty: ceil(n / k) for each level but the last, which takes the rest."""
    q = math.ceil(n / len(levels))
    return {d: (q if k < len(levels) - 1 else n - q * (len(levels) - 1)) for k, d in enumerate(levels)}


def generate(C, seed, sizes, only, tier="base"):
    templates, levels, id_prefix = TIERS[tier][:3]
    tasks, report = [], {}
    for tname, fn in templates.items():
        if only and tname not in only:
            continue
        rep = {}
        for split, n in zip(SPLITS, sizes):
            rng = sub_rng(seed, tname, split)
            per_d = per_level(n, levels)
            got = {d: 0 for d in levels}
            reasons, seen, out = Counter(), set(), []
            attempts, max_attempts = 0, n * 400
            while len(out) < n and attempts < max_attempts:
                attempts += 1
                open_d = [d for d in levels if got[d] < per_d[d]]
                d = rng.choice(open_d)
                try:
                    t = fn(C, split, rng, d)
                except Skip as e:
                    reasons[str(e)] += 1
                    continue
                except (ValueError, IndexError, KeyError, ZeroDivisionError) as e:
                    reasons["error:%s" % type(e).__name__] += 1
                    continue
                sig = dumps(t["meta"]["params"])
                if sig in seen:
                    reasons["duplicate"] += 1
                    continue
                seen.add(sig)
                got[d] += 1
                out.append(t)
            for k, t in enumerate(out):
                t["task_id"] = "%s-%s-%s-%04d" % (id_prefix, split, tname, k)
                t["db"] = DB
            tasks += out
            rep[split] = {"target": n, "accepted": len(out), "candidates": attempts,
                          "by_difficulty": {str(k): v for k, v in sorted(got.items())},
                          "removed": dict(sorted(reasons.items(), key=lambda x: -x[1]))}
        report[tname] = rep
        print("%-16s %s" % (tname, " ".join("%s=%d/%d(%d cand)" % (s, rep[s]["accepted"], rep[s]["target"],
                                                                     rep[s]["candidates"]) for s in rep)), flush=True)
    return tasks, report


def check(tasks, seed, frac=0.10):
    V = verify.V()
    by_t = defaultdict(list)
    for t in tasks:
        by_t[t["template"]].append(t)
    res, failures, evidence = {}, [], {}
    for tname, ts in by_t.items():
        rng = sub_rng(seed, "verify", tname)
        k = max(3, math.ceil(frac * len(ts)))
        sample = rng.sample(ts, min(k, len(ts)))
        ok = 0
        for t in sample:
            try:
                vals, ev = verify.VERIFIERS[tname](V, t["meta"]["params"])
                good = verify.compare(t["keys"], vals)
            except Exception as e:  # a crash in the second implementation counts as disagreement
                vals, ev, good = None, "error %r" % e, False
            evidence[t["task_id"]] = ev
            ok += good
            if not good:
                failures.append({"task_id": t["task_id"], "keys": t["keys"], "recomputed": vals, "evidence": ev})
        res[tname] = {"checked": len(sample), "agree": ok}
    return res, failures, evidence


def spotcheck(tasks, seed, path):
    V = verify.V()
    by_t = defaultdict(list)
    for t in tasks:
        by_t[t["template"]].append(t)
    lines = []
    for tname, ts in by_t.items():
        rng = sub_rng(seed, "spot", tname)
        for t in rng.sample(ts, min(3, len(ts))):
            try:
                vals, ev = verify.VERIFIERS[tname](V, t["meta"]["params"])
            except Exception as e:
                vals, ev = None, "error %r" % e
            lines += ["=" * 100, "%s  (difficulty %d)" % (t["task_id"], t["meta"]["difficulty"]),
                      "PROMPT: " + t["prompt"], "ROWS USED: " + ev, "ANSWER: " + t["answer_text"],
                      "KEYS: " + json.dumps(t["keys"]), "RECOMPUTED: " + json.dumps(vals, default=str), ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def snapshot_hash(cache_dir):
    st = os.stat(DB)
    p = os.path.join(cache_dir, "snapshot_sha256.json")
    if os.path.exists(p):
        with open(p) as f:
            c = json.load(f)
        if c["size"] == st.st_size and c["mtime"] == st.st_mtime:
            return c["sha256"]
    h = sha256_file(DB)
    with open(p, "w") as f:
        json.dump({"size": st.st_size, "mtime": st.st_mtime, "sha256": h}, f)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tier", default="base", choices=sorted(TIERS))
    ap.add_argument("--sizes", default=None, help="train,dev,test per template (default: the tier's)")
    ap.add_argument("--only", default="")
    ap.add_argument("--dry", action="store_true", help="generate and check, print samples, write nothing")
    ap.add_argument("--full-check", action="store_true", help="also recompute every task with verify.py")
    a = ap.parse_args()
    _, _, _, file_prefix, manifest_name, spot_name, default_sizes = TIERS[a.tier]
    sizes = [int(x) for x in (a.sizes or default_sizes).split(",")]
    only = set(x for x in a.only.split(",") if x)
    t0 = time.time()
    S = Snapshot(a.cache_dir)
    X = SecIndex(S.facts)
    C = Ctx(S, X)
    print("loaded in %.0fs; equities per split %s; SEC equities %s" % (
        time.time() - t0, {k: len(v) for k, v in C.eq.items()}, {k: len(v) for k, v in C.sec_eq.items()}), flush=True)
    tasks, report = generate(C, a.seed, sizes, only, a.tier)
    agreement, failures, _ = check(tasks, a.seed)
    print("independent check:", json.dumps(agreement))
    for f in failures[:30]:
        print("DISAGREE", json.dumps(f, default=str)[:1500])
    full = None
    if a.full_check:
        full, full_fail, _ = check(tasks, a.seed, frac=1.0)
        print("full independent check:", json.dumps(full))
        for f in full_fail[:30]:
            print("DISAGREE(full)", json.dumps(f, default=str)[:1500])
    if a.dry:
        by = defaultdict(list)
        for t in tasks:
            by[t["template"]].append(t)
        for tname, ts in by.items():
            for t in ts[:2]:
                print("-" * 80)
                print(t["task_id"], t["meta"]["difficulty"], t["prompt"])
                print("  ->", t["answer_text"], json.dumps(t["keys"]))
        return
    os.makedirs(a.out_dir, exist_ok=True)
    files = {}
    for split in SPLITS:
        p = os.path.join(a.out_dir, "%s_%s.jsonl" % (file_prefix, split))
        with open(p + ".tmp", "w") as f:
            for t in tasks:
                if t["split"] == split:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
        os.replace(p + ".tmp", p)
        files[os.path.basename(p)] = sha256_file(p)
    spot = os.path.join(a.out_dir, spot_name)
    spotcheck(tasks, a.seed, spot)
    counts = defaultdict(dict)
    for t in tasks:
        counts[t["template"]][t["split"]] = counts[t["template"]].get(t["split"], 0) + 1
    meta = S.meta
    manifest = {
        **({"tier": a.tier, "difficulties": list(TIERS[a.tier][1])} if a.tier != "base" else {}),
        "seed": a.seed, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": os.path.dirname(os.path.abspath(__file__)), "db": DB,
        "snapshot_sha256": snapshot_hash(a.cache_dir),
        "snapshot_meta": {k: meta.get(k) for k in ("cutoff", "download_date", "last_build_finished", "counts")},
        "sizes_target": dict(zip(SPLITS, sizes)), "counts": counts,
        "totals": {s: sum(1 for t in tasks if t["split"] == s) for s in SPLITS},
        "files_sha256": files, "spotcheck_sha256": sha256_file(spot),
        "quality_removals": report, "independent_check": agreement, "independent_check_failures": failures,
        "independent_check_all_tasks": full,
        "sec_fiscal_years_dropped_ambiguous": X.fy_ambiguous,
    }
    with open(os.path.join(a.out_dir, manifest_name), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    print("wrote", a.out_dir, json.dumps(manifest["totals"]), "in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
