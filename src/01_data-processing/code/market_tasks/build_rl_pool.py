"""Fresh RL task pool from the 10 training templates of hardset_v1, apart from every task the SFT data or the evaluation used.

Version 1 (research/log/2026-11-01-0010.md, 11-02 22:xx UTC): GRPO run1/run1b and PPO ppo1 train on 939 of the 2,350
tasks of hardset_v1/train_market.jsonl, whose teacher trajectories M1 was SFT-trained on; Qwen3, Hunyuan-A13B and
MiniMax keep RL tasks apart from SFT tasks. This driver draws new instances of the same 10 templates (the 7 hard ones
without div_event_study, from templates_hard.py as hardened once on 10-31 13:3x UTC, and rolling_vol, extreme_period,
xbrl_ratio from templates.py) for the train split only, with a new seed, the generator's difficulty mix (4/5 or 1/2/3
in equal parts) and its quality rules. It reuses generate.py's per-template rng streams, difficulty targets,
independent check and spotcheck; generate.py itself is unchanged, so its outputs stay byte-identical.

A candidate is also rejected when it repeats a reference task (every --ref file, by default the five files of
hardset_v1's train, hard-train, hard-train-extra, dev and test sources, the full tier files, and the pre-hardening
_hard_v1 files) or another pool task:
  reference_duplicate       same template and the same params (generate.py's own duplicate rule, across files)
  reference_near_duplicate  same template, the same set of meta.entities (tickers, CIKs, benchmark, FRED series) and
                            date windows [meta.start, meta.end] that overlap by at least one day
  pool_near_duplicate       the same near-duplicate rule against the pool's earlier tasks
A looser overlap that is only reported, not removed: same template, at least one shared equity and overlapping windows.
A candidate that passes these is recomputed with verify.py at once and rejected (independent_disagree) when the second
implementation disagrees or fails: hardset_v1 had full agreement, and the first build of this seed had one such task
(q4_margin_rank, Garmin FY2017: verify.py found no matching NetIncomeLoss fact; which implementation is right was not
settled), so the filter keeps the pool's agreement whole while the final full check still reports it.
Every accepted task is checked against the split rules (equities by ticker hash; whenever an index, ETF, FX, future
or FRED series takes part, every date inside 2000-01-01..2016-12-31), which the generator already applies.

Writes <out>/train_rl.jsonl (task ids mk-rl2-train-<template>-NNNN), spotcheck_rl.txt and manifest.json (seed, counts
per template and difficulty, removals per rule and per reference file, the split check, the independent recompute of
every task, and the checks given on the command line: --regen-of compares the written file with an earlier build of
the same seed, --golden summarizes golden.py's results.jsonl under a config, --check-market reads envtool.py's line).

Usage:
  PY=$FP_VENVS/mkt/bin/python
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 nice -n 19 $PY -B build_rl_pool.py --cache-dir <local dir>
      [--out-dir <market_v1>/rl_pool_v2] [--seed 20261102] [--size 200] [--only t1,t2] [--ref <tasks.jsonl> ...]
      [--regen-of <earlier train_rl.jsonl> ...] [--golden H1=<golden out dir> ...] [--check-market <envtool output>]
"""
import argparse
import bisect
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import DB, M_DIR, SPLIT_WINDOWS, Skip, Snapshot, dumps, sha256_file, ticker_split  # noqa: E402
from generate import check, per_level, snapshot_hash, spotcheck, sub_rng  # noqa: E402
from sec_index import SecIndex  # noqa: E402
from templates import TEMPLATES, Ctx  # noqa: E402
from templates_hard import HARD_TEMPLATES  # noqa: E402
import verify  # noqa: E402

POOL = [(t, "hard") for t in HARD_TEMPLATES if t != "div_event_study"] + \
       [(t, "base") for t in TEMPLATES if t in ("rolling_vol", "extreme_period", "xbrl_ratio")]
LEVELS = {"hard": (4, 5), "base": (1, 2, 3)}
FN = {"hard": HARD_TEMPLATES, "base": TEMPLATES}
T = M_DIR + "/tasks"
REFS = [M_DIR + "/hardset_v1/" + f for f in ("train_market.jsonl", "dev_market.jsonl", "test_market.jsonl")] + \
       [T + "/market_tasks_hard_%s.jsonl" % s for s in ("train", "train_extra", "dev", "test")] + \
       [T + "/market_tasks_%s.jsonl" % s for s in ("train", "dev", "test")] + \
       [T + "/_hard_v1/market_tasks_hard_%s.jsonl" % s for s in ("train", "dev", "test")]
SPLIT = "train"
ID_PREFIX = "mk-rl2"


class Index:
    """Exact and near-duplicate lookup of tasks by template; each entry remembers the file it came from."""

    def __init__(self):
        self.exact = collections.defaultdict(set)    # (template, params json) -> files
        self.near = collections.defaultdict(list)    # (template, entity set) -> [(start, end, file)]

    def add(self, t, src):
        m = t["meta"]
        self.exact[(t["template"], dumps(m["params"]))].add(src)
        self.near[(t["template"], frozenset(m["entities"]))].append((m["start"], m["end"], src))

    def match(self, t):
        """(files with the same params, files with a near duplicate)."""
        m = t["meta"]
        near = {f for s, e, f in self.near.get((t["template"], frozenset(m["entities"])), ())
                if s <= m["end"] and m["start"] <= e}
        return self.exact.get((t["template"], dumps(m["params"])), set()), near


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_violations(C, t):
    """The entities of t that break the train split's rules (see the module docstring)."""
    m, lo, hi = t["meta"], *SPLIT_WINDOWS[SPLIT]
    bad = []
    for e in m["entities"]:
        if e.startswith("CIK:"):
            continue  # a filer goes with its ticker, which is in the same list
        if e in C.S.px and C.kind(e) == "equity":
            if ticker_split(e) != SPLIT:
                bad.append(e)
        elif not (lo <= m["start"] and m["end"] <= hi):
            bad.append(e)
    return bad


def shared_equity_overlap(C, pool, refs):
    """Per template, pool tasks sharing an equity with an overlapping window with some reference task (report only)."""
    by = collections.defaultdict(list)    # (template, equity) -> sorted reference starts with their ends
    for t in refs:
        for e in t["meta"]["entities"]:
            if e in C.S.px and C.kind(e) == "equity":
                by[(t["template"], e)].append((t["meta"]["start"], t["meta"]["end"]))
    for v in by.values():
        v.sort()
    out = collections.Counter()
    for t in pool:
        m = t["meta"]
        for e in m["entities"]:
            v = by.get((t["template"], e), [])
            if any(s <= m["end"] and m["start"] <= x for s, x in v[:bisect.bisect_right(v, (m["end"], "9"))]):
                out[t["template"]] += 1
                break
    return dict(out)


def agrees(V, t):
    try:
        vals, _ = verify.VERIFIERS[t["template"]](V, t["meta"]["params"])
        return verify.compare(t["keys"], vals)
    except Exception:  # a crash in the second implementation counts as disagreement, as in generate.check
        return False


def build(C, seed, size, only, refs):
    pool_idx, V = Index(), verify.V()
    tasks, report = [], {}
    for tname, tier in POOL:
        if only and tname not in only:
            continue
        fn, levels = FN[tier][tname], LEVELS[tier]
        rng = sub_rng(seed, tname, SPLIT)
        per_d = per_level(size, levels)
        got = {d: 0 for d in levels}
        reasons, by_file, out = collections.Counter(), collections.Counter(), []
        attempts, max_attempts = 0, size * 400  # generate.py's cap
        while len(out) < size and attempts < max_attempts:
            attempts += 1
            d = rng.choice([x for x in levels if got[x] < per_d[x]])
            try:
                t = fn(C, SPLIT, rng, d)
            except Skip as e:
                reasons[str(e)] += 1
                continue
            except (ValueError, IndexError, KeyError, ZeroDivisionError) as e:
                reasons["error:%s" % type(e).__name__] += 1
                continue
            same, _ = pool_idx.match(t)
            if same:
                reasons["duplicate"] += 1
                continue
            same, near = refs.match(t)
            if same or near:
                reasons["reference_duplicate" if same else "reference_near_duplicate"] += 1
                for f in same | near:
                    by_file[os.path.relpath(f, M_DIR)] += 1
                continue
            if pool_idx.match(t)[1]:
                reasons["pool_near_duplicate"] += 1
                continue
            if not agrees(V, t):
                reasons["independent_disagree"] += 1
                continue
            pool_idx.add(t, "pool")
            got[d] += 1
            out.append(t)
        for k, t in enumerate(out):
            t["task_id"] = "%s-%s-%s-%04d" % (ID_PREFIX, SPLIT, tname, k)
            t["db"] = DB
        tasks += out
        report[tname] = {"tier": tier, "target": size, "accepted": len(out), "candidates": attempts,
                         "by_difficulty": {str(k): v for k, v in sorted(got.items())},
                         "removed": dict(sorted(reasons.items(), key=lambda x: -x[1])),
                         "reference_matches_by_file": dict(sorted(by_file.items()))}
        print("%-16s %d/%d (%d cand) ref_dup %d ref_near %d pool_near %d disagree %d" % (
            tname, len(out), size, attempts, reasons["reference_duplicate"], reasons["reference_near_duplicate"],
            reasons["pool_near_duplicate"], reasons["independent_disagree"]), flush=True)
    return tasks, report


def golden_summary(path):
    rows = jsonl(os.path.join(path, "results.jsonl"))
    sums = {k: sum(r.get(k) or 0 for r in rows) for k in ("tool_errors", "verification_rejects", "internal_errors")}
    return {"dir": path, "rows": len(rows), "passed": sum(bool(r["passed"]) for r in rows),
            "end": dict(collections.Counter(r["end"] for r in rows)), **sums,
            "failed": sorted(r["task_id"] for r in rows if not r["passed"])[:20]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261102, help="unused by every market task file so far")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", default=M_DIR + "/rl_pool_v2")
    ap.add_argument("--size", type=int, default=200, help="tasks per template")
    ap.add_argument("--only", default="")
    ap.add_argument("--ref", action="append", help="reference task files (default: REFS)")
    ap.add_argument("--regen-of", action="append", default=[])
    ap.add_argument("--golden", action="append", default=[], help="CONFIG=<golden.py --out dir>")
    ap.add_argument("--check-market", help="envtool.py check-market's output on the written file")
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)
    t0 = time.time()
    ref_files = a.ref or REFS
    refs, ref_tasks = Index(), []
    for p in ref_files:
        for t in jsonl(p):
            if "meta" in t:
                refs.add(t, p)
                ref_tasks.append(t)
    S = Snapshot(a.cache_dir)
    C = Ctx(S, SecIndex(S.facts))
    print("loaded in %.0fs; %d reference tasks; train equities %d, SEC equities %d" % (
        time.time() - t0, len(ref_tasks), len(C.eq[SPLIT]), len(C.sec_eq[SPLIT])), flush=True)
    tasks, report = build(C, a.seed, a.size, only, refs)
    bad = {t["task_id"]: v for t in tasks for v in [split_violations(C, t)] if v}
    train_ref = jsonl(M_DIR + "/hardset_v1/train_market.jsonl")
    bad_ref = sum(1 for t in train_ref if split_violations(C, t))
    agreement, failures, _ = check(tasks, a.seed, frac=1.0)
    print("split violations: pool %d, train_market %d; independent check: %s" % (len(bad), bad_ref, json.dumps(agreement)))
    for f in failures[:30]:
        print("DISAGREE", json.dumps(f, default=str)[:1500])
    os.makedirs(a.out_dir, exist_ok=True)
    p = os.path.join(a.out_dir, "train_rl.jsonl")
    with open(p + ".tmp", "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    os.replace(p + ".tmp", p)
    spot = os.path.join(a.out_dir, "spotcheck_rl.txt")
    spotcheck(tasks, a.seed, spot)
    sha = sha256_file(p)
    checks = {"split_rules": {"violations": len(bad), "examples": dict(list(bad.items())[:10]),
                              "train_market_violations": bad_ref, "train_market_tasks": len(train_ref)},
              "independent_check": agreement, "independent_check_failures": failures,
              "regeneration": [{"file": q, "sha256": sha256_file(q), "identical": sha256_file(q) == sha}
                               for q in a.regen_of]}
    if a.golden:
        checks["golden"] = {k: golden_summary(v) for k, v in (g.split("=", 1) for g in a.golden)}
    if a.check_market:
        checks["check_market"] = jsonl(a.check_market)
    removed = collections.Counter()
    for r in report.values():
        removed.update({k: v for k, v in r["removed"].items()
                        if k in ("reference_duplicate", "reference_near_duplicate", "pool_near_duplicate",
                                 "independent_disagree")})
    manifest = {
        "seed": a.seed, "split": SPLIT, "size_target": a.size, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": os.path.dirname(os.path.abspath(__file__)), "db": DB, "snapshot_sha256": snapshot_hash(a.cache_dir),
        "templates": {t: tier for t, tier in POOL if not only or t in only},
        "counts": {t: r["accepted"] for t, r in report.items()},
        "by_difficulty": {t: r["by_difficulty"] for t, r in report.items()}, "total": len(tasks),
        "exclusion": {"rules": {
            "reference_duplicate": "same template and identical meta.params",
            "reference_near_duplicate": "same template, same set of meta.entities, and [meta.start, meta.end] windows "
                                        "overlapping by at least one day",
            "pool_near_duplicate": "the near-duplicate rule against earlier pool tasks",
            "independent_disagree": "verify.py's recompute disagrees with the keys or fails"},
            "removed": dict(removed),
            "report_only_shared_equity_overlap": shared_equity_overlap(C, tasks, ref_tasks),
            "references": {q: sha256_file(q) for q in ref_files}, "reference_tasks": len(ref_tasks)},
        "quality_removals": report, "files_sha256": {"train_rl.jsonl": sha, "spotcheck_rl.txt": sha256_file(spot)},
        "checks": checks,
    }
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    print("wrote %s: %d tasks, sha256 %s, in %.0fs" % (p, len(tasks), sha, time.time() - t0))


if __name__ == "__main__":
    main()
