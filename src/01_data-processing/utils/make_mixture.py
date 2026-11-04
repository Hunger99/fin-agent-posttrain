"""Write one run's SFT mixture and its manifest, per docs/history/01_data-processing.md section 7.

A run takes N samples, a share d of them from the domain pool and the rest from the general pool. Within a
pool each source contributes in proportion to its count after decontamination and dedup, no source above
CAP of the pool's part (water-filling, as select_sets.py's capacity()). Within a source, samples are taken in
the order of a salted hash of their id, so a run that needs fewer samples from a source takes a subset of a
run that needs more, and the runs differ only in what the configuration changes.

A sample is kept when select_sets.py's decontamination (interim/decontam/removed_ids.json) and dedup_sft.py
(interim/decontam/<stem>.dedup.flags.jsonl) both kept it. The domain pool is fixed only once E4 has a
result (decisions.md item 2 (2)), so a configuration with d > 0 stops unless interim/pools/domain.lock
exists; the general pool has no such wait.

Output: mixtures/<name>.jsonl in the sharegpt form LLaMA-Factory reads ({"messages": [{"role", "content"}]},
a turn's "reasoning" rendered as <think>...</think> before its content, docs/history/02_model-training.md
section 2), and mixtures/<name>.manifest.json with the configuration, per-source counts, salt and ids.

The domain pool's members come from lock_domain_pool.py (interim/pools/domain_members.json), which also
decides each CFLUE question's single version (decisions.md item 2). --no-synthetic leaves the synthetic
source out, for R3-synthetic, whose domain part keeps the other sources' proportions (section 7).
--dry-run prints the allocation without the lock and writes nothing.

--general names the general pool's sources; the English root's general pool is Tülu 3 alone
(research/log/2026-10-26-2006.md).

Usage: python make_mixture.py <data root> <name> --domain-share D --n N [--no-synthetic] [--dry-run]
       [--general tulu3]
"""
import argparse
import collections
import hashlib
import json
import os

CAP = 0.40  # docs/history/01_data-processing.md section 7
SALT = "mix-20261025"
GENERAL = ["alpaca_zh", "smoltalk_chinese", "tulu3"]

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("name")
ap.add_argument("--domain-share", type=float, required=True)
ap.add_argument("--n", type=int, default=100_000)
ap.add_argument("--no-synthetic", action="store_true")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--general", default=",".join(GENERAL), help="comma-separated general sources")
args = ap.parse_args()
GENERAL = args.general.split(",")
R = args.root
LOCK = os.path.join(R, "interim", "pools", "domain.lock")
if args.domain_share > 0 and not args.dry_run and not os.path.exists(LOCK):
    raise SystemExit("the domain pool is not locked yet: it waits for E4 (decisions.md item 2 (2))")

with open(os.path.join(R, "interim", "decontam", "removed_ids.json"), encoding="utf-8") as f:
    removed = {k: set(v) for k, v in json.load(f).items()}


def domain_rows(stem, members):
    want = set(members)
    with open(os.path.join(R, "processed", "sft", f"{stem}.jsonl"), encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["id"] in want]
    return sorted(rows, key=lambda r: hashlib.md5((SALT + r["id"]).encode()).hexdigest())


def kept(stem):
    drop = set(removed.get(stem, ()))
    with open(os.path.join(R, "interim", "decontam", f"{stem}.dedup.flags.jsonl"), encoding="utf-8") as f:
        drop |= {json.loads(line)["id"] for line in f}
    with open(os.path.join(R, "processed", "sft", f"{stem}.jsonl"), encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["id"] not in drop]
    return sorted(rows, key=lambda r: hashlib.md5((SALT + r["id"]).encode()).hexdigest())


def allocate(counts, total):
    """Water-filling: shares follow counts, no source above CAP of total, rounded to sum to total.

    A pool of one source, such as the English general pool, takes the whole total: the cap limits one source's
    share among several and has no meaning for a single one."""
    if len(counts) == 1:
        (s, c), = counts.items()
        if total > c:
            raise SystemExit(f"{s} has {c:,} samples, {total:,} needed")
        return {s: total}
    share, capped = {}, set()
    while True:
        free = total - sum(share[s] for s in capped)
        rest = {s: c for s, c in counts.items() if s not in capped}
        m = sum(rest.values())
        new = {s: free * c / m for s, c in rest.items()}
        over = {s for s, x in new.items() if x > CAP * total}
        if not over:
            share.update(new)
            break
        for s in over:
            share[s] = CAP * total
        capped |= over
    alloc = {s: int(x) for s, x in share.items()}
    for s in sorted(share, key=lambda s: share[s] - alloc[s], reverse=True)[:total - sum(alloc.values())]:
        alloc[s] += 1
    for s, k in alloc.items():
        if k > counts[s]:
            raise SystemExit(f"{s} has {counts[s]:,} samples, {k:,} needed")
    return alloc


def render(r):
    msgs = []
    for m in r["messages"]:
        content = m["content"]
        if m.get("reasoning"):
            content = f"<think>\n{m['reasoning']}\n</think>\n\n{content}"
        msgs.append({"role": m["role"], "content": content})
    return {"id": r["id"], "messages": msgs}


n_domain = round(args.n * args.domain_share)
plan = {"general": (GENERAL, args.n - n_domain)}
if n_domain:
    if os.path.exists(LOCK):
        with open(os.path.join(R, "interim", "pools", "domain_members.json"), encoding="utf-8") as f:
            pool_file = json.load(f)
        # The English pool is fixed in two parts (lock_domain_pool_en.py): before E4 only the non-synthetic one
        if not pool_file.get("synthetic_fixed", True) and not args.no_synthetic:
            raise SystemExit("only the non-synthetic part of the domain pool is fixed; it waits for E4, or use --no-synthetic")
    else:  # --dry-run before the lock: the same membership lock_domain_pool.py would write
        import subprocess
        import sys
        print("dry run: domain pool not locked; counts from lock_domain_pool.py --dry-run")
        out = subprocess.run([sys.executable, "-B", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lock_domain_pool.py"),
                              R, "--dry-run"], capture_output=True, text=True, check=True).stdout
        counts = json.loads(out.strip().splitlines()[-1])["counts"]
        pool_file = {"members": {s: [f"{s}-{i}" for i in range(c)] for s, c in counts.items()}, "synthetic": ["doc_qa"]}
    domain = {s: v for s, v in pool_file["members"].items() if not (args.no_synthetic and s in pool_file["synthetic"])}
    plan["domain"] = (sorted(domain), n_domain)
manifest = {"name": args.name, "n": args.n, "domain_share": args.domain_share, "no_synthetic": args.no_synthetic,
            "cap": CAP, "salt": SALT, "sources": {}}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# which removal lists and pool members the mixture was drawn from (research/log/2026-10-28-0040.md), so that a run
# can check, before training, that its mixture matches the current lists
manifest["inputs_sha256"] = {rel: sha256(os.path.join(R, rel)) for rel in
                             ["interim/decontam/removed_ids.json", "interim/decontam/ungrounded_ids.json",
                              "interim/pools/domain_members.json"] +
                             [f"interim/decontam/{s}.dedup.flags.jsonl" for s in sorted(set().union(
                                 *(set(st) for st, _ in plan.values())))]
                             if os.path.exists(os.path.join(R, rel))}
out = []
for pool, (stems, total) in plan.items():
    if args.dry_run:
        counts = ({s: len(domain[s]) for s in stems} if pool == "domain" else {s: len(kept(s)) for s in stems})
        print(pool, total, allocate(counts, total))
        continue
    rows = {s: (domain_rows(s, domain[s]) if pool == "domain" else kept(s)) for s in stems}
    alloc = allocate({s: len(v) for s, v in rows.items()}, total)
    for s in stems:
        picked = rows[s][:alloc[s]]
        out += [render(r) for r in picked]
        manifest["sources"][s] = {"pool": pool, "available": len(rows[s]), "taken": len(picked), "ids": [r["id"] for r in picked]}
if args.dry_run:
    raise SystemExit(0)
os.makedirs(os.path.join(R, "mixtures"), exist_ok=True)
out.sort(key=lambda r: hashlib.md5(("order-" + SALT + r["id"]).encode()).hexdigest())
with open(os.path.join(R, "mixtures", f"{args.name}.jsonl"), "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
with open(os.path.join(R, "mixtures", f"{args.name}.manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False)
print(json.dumps({s: {k: v for k, v in d.items() if k != "ids"} for s, d in manifest["sources"].items()}, ensure_ascii=False))
