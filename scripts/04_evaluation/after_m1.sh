#!/bin/bash
# The new dataset's test rows for all four models side by side (research/log/2026-11-01-0010.md, 09:3x UTC), replacing
# refs_4b2.sh (whose start-model run on test_market it adopts) and eval_m1.sh, with the same commands as theirs. Once
# M1's training ends: the official Qwen3.5-4B on GPUs 4, 5 and the teacher (TP2) on GPUs 6, 7 at once, M1 on GPUs 2, 3
# once it trained cleanly, and the start model goes on on GPU 9. Each 4B model runs test_market, test_finsearchcomp and
# test_docfinqa in that order (M1 then dev_market x4 and the v2 wealth/ML test); every 10 minutes it adds a server on
# each free, unlocked GPU of its list and restarts its current file (run.py resumes) if 60 or more tasks remain.
# "refs_4b done" (screen_hard_m1.sh's start) is written once the start model, the official model and the teacher end.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
EN=$FP_DATA_EN; O=$FP_DATA_D2/harness_v2; P=$O/pilot; H=$EN/harness
MK=$FP_DATA_D2/market_v1; HS=$MK/hardset_v1; V3=$ROOT/src/02_harness/code/v3
R=$FP_RUNS/qwen3.5-4b-base-en
MD=$FP_MODELS_D2; PY=$FP_VENVS/vllm-0.12/bin/python; export PYTHONDONTWRITEBYTECODE=1
TS=$EN/logs/train_q35_4b_M1_en.status; CS=$MK/runs/chain_m1b.status; LOG=$MK/runs/after_m1.status; BASE_RP=302830
FLAG=$MK/runs/after_m1.assigned  # no model adds GPUs before the first assignment is made
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
free_gpu() { U=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader); [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $U)" = 0 ]; }
stop_srv() { p=$(head -n 1 $1 | grep -o "pid [0-9]*" | cut -d' ' -f2); [ -n "$p" ] && { kill -- -$p 2>/dev/null; sleep 20; kill -9 -- -$p 2>/dev/null; }; }
up() { for i in $(seq 90); do curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "\"$2\"" && return 0; sleep 10; done; return 1; }
rows() { [ -f $MK/runs/$1/results.jsonl ] && wc -l < $MK/runs/$1/results.jsonl || echo 0; }
file_of() { case $1 in market) echo "$HS/test_market.jsonl $MK/market_v1.sqlite";; fsc) echo "$HS/test_finsearchcomp.jsonl $MK/market_v1.sqlite";;
  docfinqa) echo "$HS/test_docfinqa.jsonl $EN/harness/docfinqa_test";; dev) echo "$HS/dev_market.jsonl $MK/market_v1.sqlite";; esac; }

# A 4B model's v3 runs in a subshell. Set by the caller: ARM (output and log name), DIR, M (served name), GC (GENCFG or
# empty), PFX (port prefix), K (episodes per server), MORE (GPUs it may add); G and PORTS hold its servers.
slog() { echo $EN/logs/serve_4b_tesths_${ARM}_gpu$1.log; }
grab() {  # <gpus>: a server on each that is free and unlocked; returns 0 if any was added
  local x added=1
  for x in $1; do
    echo " $G " | grep -q " $x " && continue
    free_gpu $x && mkdir $P/locks/gpu$x 2>/dev/null || continue
    ( GENCFG=$GC MAXSEQS=$K setsid sh $ROOT/scripts/serve_4b_keep_thinking.sh $x $PFX$x $DIR $M $(slog $x) < /dev/null > /dev/null 2>&1 & )
    if up $PFX$x $M; then G="$G $x"; PORTS="$PORTS $PFX$x"; added=0; say "$ARM server on GPU $x"
    else say "$ARM server on GPU $x did not start"; stop_srv $(slog $x); rmdir $P/locks/gpu$x; fi
  done
  return $added
}
v3file() {  # <file key> <runs> <out name>
  local T db RP c n p urls need
  read T db <<EOF
$(file_of $1)
EOF
  need=$(( $(wc -l < $T) * $2 ))
  while true; do
    n=$(echo $PORTS | wc -w); urls=""; for p in $PORTS; do urls="$urls --base-url http://127.0.0.1:$p/v1"; done
    setsid $PY -B $V3/run.py --tasks $T --db $db --config H1 --runs $2 --max-steps 100 --concurrency $((K * n)) --provider qwen \
      --model $M $urls --out $MK/runs/$3 >> $MK/runs/$3.log 2>&1 &
    RP=$!; c=0
    while kill -0 $RP 2>/dev/null; do
      sleep 60; c=$((c + 1))
      if [ $((c % 10)) = 0 ] && [ -f $FLAG ] && [ $((need - $(rows $3))) -ge 60 ] && grab "$MORE"; then
        kill -- -$RP 2>/dev/null; sleep 5; kill -9 -- -$RP 2>/dev/null; wait $RP; say "$ARM $3 restarts on GPUs$G"; continue 2; fi
    done
    wait $RP; say "$ARM $3 exit $?: $(rows $3) rows, $(grep -c '"passed": true' $MK/runs/$3/results.jsonl) passed"; return
  done
}
stop_all() { local x; for x in $G; do stop_srv $(slog $x); rmdir $P/locks/gpu$x; done; G=""; PORTS=""; }

base_job() {
  ARM=qwen3.5-4b-base_H1_stop; DIR=$MD/Qwen3.5-4B-Base; M=qwen3.5-4b-base; GC=$V3/templates/base_stop; PFX=1819; K=16
  MORE="9 2 3 4 5 6 7"; G=" 9"; PORTS=" 18199"
  while kill -0 $BASE_RP 2>/dev/null; do sleep 60; done
  say "$ARM market (adopted run.py) ended: $(rows test_hs_market_$ARM) rows"
  for f in market fsc docfinqa; do v3file $f 1 test_hs_${f}_$ARM; done
  stop_all; say "$ARM done"
}
official_job() {
  ARM=qwen3.5-4b_H1; DIR=$MD/Qwen3.5-4B; M=qwen3.5-4b; GC=""; PFX=1819; K=16; MORE="4 5 9 2 3 6 7"; G=""; PORTS=""
  until grab "4 5"; do sleep 60; done; sleep 60; grab "4 5"
  for f in market fsc docfinqa; do v3file $f 1 test_hs_${f}_$ARM; done
  stop_all; say "$ARM done"
}
teacher_job() {
  local f T db k SL=$EN/logs/serve_q38_tesths_18011.log
  until free_gpu 6 && free_gpu 7 && mkdir $P/locks/gpu6 2>/dev/null; do sleep 60; done
  until mkdir $P/locks/gpu7 2>/dev/null; do sleep 60; done
  ( setsid sh $ROOT/scripts/serve_qwen38.sh 6,7 18011 $SL < /dev/null > /dev/null 2>&1 & )
  if up 18011 qwen3.8-27b; then
    say "teacher server on GPUs 6, 7"
    for f in market fsc docfinqa; do
      read T db <<EOF
$(file_of $f)
EOF
      k=20; [ $f = docfinqa ] && k=16
      $PY -B $V3/run.py --tasks $T --db $db --config H1 --runs 1 --concurrency $k --provider qwen --model qwen3.8-27b \
        --base-url http://127.0.0.1:18011/v1 --out $MK/runs/test_hs_${f}_q38_H1 >> $MK/runs/test_hs_${f}_q38_H1.log 2>&1
      say "teacher test_hs_${f}_q38_H1 exit $?: $(rows test_hs_${f}_q38_H1) rows, $(grep -c '"passed": true' $MK/runs/test_hs_${f}_q38_H1/results.jsonl) passed"
    done
  else say "teacher server did not start"; fi
  stop_srv $SL; rmdir $P/locks/gpu6 $P/locks/gpu7; say "teacher done"
}
m1_job() {
  ARM=M1; DIR=$R/M1; M=M1; GC=""; PFX=1816; K=24; MORE="2 3 9 4 5 6 7"; G=""; PORTS=""
  until grep -q "M1 trained; chain_m1b done" $CS; do
    grep -q "M1 did not train cleanly\|stopped\|not trained\|failed" $CS && { say "chain_m1b ended without M1: not evaluated"; return; }
    sleep 60
  done
  until grab "2 3"; do sleep 60; done; sleep 60; grab "2 3"
  for f in market fsc docfinqa; do v3file $f 1 test_hs_${f}_M1_H1; done
  v3file dev 4 dev_hs_market_M1_H1_x4
  local g gs="$G" urls="" n=0; for g in $gs; do stop_srv $(slog $g); done
  # the wealth and ML test sets, v2 harness, Qwen3.5's own template (eval_m1.sh (2)); the GPUs it held stay locked
  for g in $gs; do ( MAXSEQS=96 setsid sh $ROOT/scripts/serve_4b.sh $g 1817$g $R/M1 M1 $EN/logs/serve_4b_test_M1_gpu$g.log < /dev/null > /dev/null 2>&1 & ); done
  for g in $gs; do up 1817$g M1 && { urls="$urls --base-url http://127.0.0.1:1817$g/v1"; n=$((n+1)); }; done
  run2() {  # <part> <concurrency>: two passes, as base_test_stop.sh run()
    local part=$1 k=$2 T pass
    if [ $part = wealth ]; then T="--tasks $H/tasks_test.jsonl --db $H/C.sqlite"; else T="--tasks $H/tasks_ml_test.jsonl --db none"; fi
    for pass in 1 2; do
      $PY -B $O/code_collect/run.py $T --concurrency $k --runs 1 --config H1 --max-steps 100 --keep-results 10 --summary-chars 1000 \
        --provider qwen --model M1 $urls --out $O/runs_q4b_test/M1_H1_$part >> $O/logs/q4bt_M1_H1_$part.log 2>&1
      echo "pass $pass exit $?" >> $O/logs/q4bt_M1_H1_$part.log
    done
  }
  if [ $n -gt 0 ]; then
    run2 wealth $((64 * n)) & w=$!; run2 ml $((12 * n)); wait $w
    say "M1 wealth/ML test: $(wc -l < $O/runs_q4b_test/M1_H1_wealth/results.jsonl) wealth rows, $(wc -l < $O/runs_q4b_test/M1_H1_ml/results.jsonl) ml rows"
  else say "M1 v2 servers did not start"; fi
  for g in $gs; do stop_srv $EN/logs/serve_4b_test_M1_gpu$g.log; rmdir $P/locks/gpu$g; done
  say "M1 done"
}

say "after_m1 starts; the start model goes on on GPU 9 (run.py $BASE_RP)"
( base_job ) & B=$!
until grep -q "^M1 exit" $TS; do sleep 60; done
say "M1's training ended: $(grep '^M1 exit' $TS | tail -n 1)"
( official_job ) & F=$!
( teacher_job ) & T=$!
( m1_job ) & M1P=$!
for i in $(seq 20); do a=1; for g in 2 3 4 5 6 7; do [ -d $P/locks/gpu$g ] || a=0; done; [ $a = 1 ] && break; sleep 60; done
touch $FLAG; say "first assignment made: models may add GPUs"
wait $B $F $T
echo "refs_4b done (after_m1.sh) $(date -u '+%F %T')" >> $MK/runs/refs_4b.status
wait $M1P
say "after_m1 done"
