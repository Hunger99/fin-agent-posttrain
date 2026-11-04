#!/bin/bash
# Regrade one evaluated model's market runs with the original and the strict grader (src/04_evaluation/
# regrade_strict.py; research/log/2026-11-01-0010.md, 11-02 22:3x UTC), as eval_ckpt_8gpu.sh step 5 compares them:
# test_market and test_finsearchcomp (98 tasks, and 97 without fsc-(T3)Complex_Historical_Investigation_023) against
# M1, the official Qwen3.5-4B, the start model and the teacher q38; dev_market x4 against M1's x4 run, with the
# subset of long templates. CPU only, nice 19; reads the run directories and writes only $OUT/<name>/:
# <run>.json per run, compare_<tasks file>.json per file, and regrade.txt (the printed lines).
# Usage: bash regrade_run.sh <name>   (e.g. grpo1, ppo1: the name eval_ckpt_8gpu.sh was given)
set -u
[ $# -eq 1 ] || { echo "usage: $0 <name>" >&2; exit 2; }
NAME=$1
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
MK=$FP_DATA_D2/market_v1; HS=$MK/hardset_v1
RG=$ROOT/src/04_evaluation/utils/regrade_strict.py
PY=$FP_VENVS/vllm-0.12/bin/python  # harness v3's venv (eval_ckpt_8gpu.sh's PY)
OUT=$STRICT_DIR/$NAME
LONG=xbrl_ratio,q4_margin_rank,filing_reaction,extreme_period,ma_cross
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
rg() { nice -n 19 $PY -B $RG $MK/runs "$@" --out "$OUT"; }
{
  for f in market fsc; do
    T=$HS/test_$f.jsonl; [ $f = fsc ] && T=$HS/test_finsearchcomp.jsonl
    b=test_hs_${f}_${NAME}_H1; x=(); [ $f = fsc ] && x=(--exclude "fsc-(T3)Complex_Historical_Investigation_023")
    [ -s $MK/runs/$b/results.jsonl ] || { echo "no rows: $b"; continue; }
    echo "== $(basename $T .jsonl)"
    rg $T test_hs_${f}_M1_H1 test_hs_${f}_qwen3.5-4b_H1 test_hs_${f}_qwen3.5-4b-base_H1_stop test_hs_${f}_q38_H1 $b \
      --pair test_hs_${f}_M1_H1:$b --pair test_hs_${f}_qwen3.5-4b_H1:$b --pair test_hs_${f}_qwen3.5-4b-base_H1_stop:$b \
      --pair $b:test_hs_${f}_q38_H1 "${x[@]}"
  done
  b=dev_hs_market_${NAME}_H1_x4
  if [ -s $MK/runs/$b/results.jsonl ]; then
    echo "== dev_market x4"
    rg $HS/dev_market.jsonl dev_hs_market_M1_H1_x4 $b --pair dev_hs_market_M1_H1_x4:$b --subset $LONG
  else echo "no rows: $b"; fi
} 2>&1 | tee "$OUT/regrade.txt"
