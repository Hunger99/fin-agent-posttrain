#!/bin/bash
# Comparisons of the RL models' reworded market test runs (rl_chain_c.sh stage 5c; research/log/2026-11-01-0010.md, 11-02
# 23:2x UTC), CPU only, the counterpart of compare_reworded.sh for each model in RL_MODELS (default "grpo1 grpo2 ppo2"),
# with the same scripts and settings (paired per task, 10,000 resamples, seed 20261025):
#   1. original -> reworded (paired_runs.py: test_hs_market_<m>_H1 -> test_hsrw_market_<m>_H1);
#   2. the model's gain over M1 on each set (paired_runs.py, M1 -> <m>);
#   3. the difference in differences, gain on the reworded set minus gain on the original (did_reworded.py with <m>
#      in its "M1" arguments and M1 in its "official" ones, so its m1_* fields are <m>'s and its off_* fields M1's).
# Writes test_hsrw_market_rl.compare.txt (one JSON line per result under "== <section>" lines) next to the runs; under
# DRYDIR (default $DRY_DIR/) with DRY=1, where missing runs are reported instead of compared.
# Usage: [RL_MODELS="grpo1 grpo2 ppo2"] bash compare_reworded_rl.sh
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
MK=$FP_DATA_D2/market_v1; RUNS=$MK/runs; T=$MK/hardset_v1/test_market.jsonl
HERE=$ROOT/scripts/04_evaluation
PY=$FP_VENVS/vllm-0.12/bin/python; export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
if [ -n "$DRY" ]; then D=${DRYDIR:-$DRY_DIR}; mkdir -p $D; CMP=$D/test_hsrw_market_rl.compare.txt; else CMP=$RUNS/test_hsrw_market_rl.compare.txt; fi
: > $CMP
section() { echo "== $1" >> $CMP; }
have() { local x; for x in "$@"; do [ -s $RUNS/$x/results.jsonl ] || { echo "{\"missing\": \"$x\"}" >> $CMP; return 1; }; done; }
pair() { have $1 $2 && nice -n 19 $PY -B $ROOT/src/04_evaluation/utils/paired_runs.py $RUNS $T $1 $2 >> $CMP 2>&1; }
for m in ${RL_MODELS-grpo1 grpo2 ppo2}; do
  section "$m original -> reworded"; pair test_hs_market_${m}_H1 test_hsrw_market_${m}_H1
  section "original: M1 -> $m"; pair test_hs_market_M1_H1 test_hs_market_${m}_H1
  section "reworded: M1 -> $m"; pair test_hsrw_market_M1_H1 test_hsrw_market_${m}_H1
  section "$m gain over M1: difference in differences (did_reworded.py; m1_* = $m, off_* = M1)"
  set -- test_hs_market_${m}_H1 test_hs_market_M1_H1 test_hsrw_market_${m}_H1 test_hsrw_market_M1_H1
  have "$@" && nice -n 19 $PY -B $ROOT/src/04_evaluation/utils/did_reworded.py $RUNS $T "$@" >> $CMP 2>&1
done
echo "written: $CMP"
