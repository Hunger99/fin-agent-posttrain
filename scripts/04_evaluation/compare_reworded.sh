#!/bin/bash
# Comparisons of the reworded market test runs (research/log/2026-11-01-0010.md, 11-02 23:1x UTC), CPU only, all paired
# per task with paired_runs.py's scores and bootstrap (10,000 resamples, seed 20261025):
#   1. each model, original -> reworded (paired_runs.py: test_hs_market_<m>_H1 -> test_hsrw_market_<m>_H1);
#   2. M1's gain over the official model on the original and on the reworded set (paired_runs.py, official -> M1);
#   3. the difference in differences, gain on the reworded set minus gain on the original (did_reworded.py).
# The task file of 1 and 2 is hardset_v1/test_market.jsonl, which has the same task ids and templates as the reworded one.
# Writes test_hsrw_market.compare.{json,txt} next to the runs ($MK/runs; under dry/ with DRY=1, where missing runs are
# reported instead of compared).
# Usage: bash compare_reworded.sh   (eval_reworded.sh calls it once both models' runs exist)
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
MK=$FP_DATA_D2/market_v1; RUNS=$MK/runs; HS=$MK/hardset_v1
HERE=$ROOT/scripts/04_evaluation
PY=$FP_VENVS/vllm-0.12/bin/python; export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
if [ -n "$DRY" ]; then mkdir -p $DRY_DIR/reworded; CMP=$DRY_DIR/reworded/test_hsrw_market.compare; else CMP=$RUNS/test_hsrw_market.compare; fi
T=$HS/test_market.jsonl
: > $CMP.raw
section() { echo "== $1" >> $CMP.raw; }
have() { local x; for x in "$@"; do [ -s $RUNS/$x/results.jsonl ] || { echo "{\"missing\": \"$x\"}" >> $CMP.raw; return 1; }; done; }
pair() { have $1 $2 && nice -n 19 $PY -B $ROOT/src/04_evaluation/utils/paired_runs.py $RUNS $T $1 $2 >> $CMP.raw 2>&1; }
for m in M1 qwen3.5-4b; do section "$m original -> reworded"; pair test_hs_market_${m}_H1 test_hsrw_market_${m}_H1; done
section "original: official -> M1"; pair test_hs_market_qwen3.5-4b_H1 test_hs_market_M1_H1
section "reworded: official -> M1"; pair test_hsrw_market_qwen3.5-4b_H1 test_hsrw_market_M1_H1
section "difference in differences"
set -- test_hs_market_M1_H1 test_hs_market_qwen3.5-4b_H1 test_hsrw_market_M1_H1 test_hsrw_market_qwen3.5-4b_H1
have "$@" && nice -n 19 $PY -B $ROOT/src/04_evaluation/utils/did_reworded.py $RUNS $T "$@" >> $CMP.raw 2>&1
python3 - $CMP.raw $CMP.json $CMP.txt <<'EOF'
import json, sys
raw, js, txt = sys.argv[1:4]
out, sec = {}, None
for line in open(raw):
    line = line.rstrip("\n")
    if line.startswith("== "):
        sec = line[3:]; out[sec] = []; continue
    try:
        out[sec].append(json.loads(line))
    except ValueError:
        out[sec].append({"text": line})
json.dump(out, open(js, "w"), indent=1)
lines = []
ci = lambda x: f"[{x['ci95'][0]}, {x['ci95'][1]}]"
for sec, rs in out.items():
    lines.append(f"{sec}:" + ("" if rs else " no rows"))
    for r in rs:
        if "missing" in r:
            lines.append(f"  missing {r['missing']}")
        elif "comparison" in r:
            lines.append(f"  {r['comparison']}: " + ", ".join(f"{k} {v}" for k, v in r.items() if k != "comparison"))
        elif "all" in r and r["all"] and "did" in r["all"]:
            x = r["all"]
            lines.append(f"  all: gain {x['gain_orig']} -> {x['gain_rw']}, did {x['did']:+.1f} {ci(x)}, share >= 0 "
                         f"{x['share_ge_0']:.4f}, {x['n']} tasks (M1 {x['m1_orig']} -> {x['m1_rw']}, official "
                         f"{x['off_orig']} -> {x['off_rw']})")
        elif "all" in r and r["all"]:
            x = r["all"]
            lines.append(f"  all: {x['A']} -> {x['B']}, {x['diff']:+.1f} {ci(x)}, p_one_sided {x['p_one_sided']:.4f}, "
                         f"{x['n']} tasks")
        elif "template" in r and r["result"]:
            x = r["result"]
            lines.append(f"    {r['template']}: " + (f"gain {x['gain_orig']} -> {x['gain_rw']} ({x['did']:+.1f}, n {x['n']})"
                                                  if "did" in x else f"{x['A']} -> {x['B']} ({x['diff']:+.1f}, n {x['n']})"))
        elif "text" in r:
            lines.append(f"  {r['text']}")
open(txt, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
rm -f $CMP.raw
