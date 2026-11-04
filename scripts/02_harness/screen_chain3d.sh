#!/bin/bash
# screen_chain3c.sh with the screenings reordered (research/log/2026-10-31-0150.md, 18:3x UTC): the tier-1 training
# tasks first, then wealth and ML, the hard tier last (its pass 1 resumes from the 172 rows chain3c left), since the
# start model solves most tier-1 and wealth tasks and only about a tenth of the hard-tier ones. The dev runs are done
# and left out. It takes over the start-model servers on GPUs 2 and 3 (ports 18182, 18183) with their locks; CANDS
# also holds GPU 4 (after base_test_stop.sh) and the teacher's GPUs 5, 6, 7 and 9, each added once it is free and
# unlocked. Every pass resumes from its results.jsonl.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
EN=$FP_DATA_EN; P=$FP_DATA_D2/harness_v2/pilot
MK=$FP_DATA_D2/market_v1; V3=$ROOT/src/02_harness/code/v3
C=$FP_DATA_D2/agent_sft/snapshot_20261030_0906; H=$EN/harness
PY=$FP_VENVS/vllm-0.12/bin/python; export PYTHONDONTWRITEBYTECODE=1
BASE=$FP_MODELS_D2/Qwen3.5-4B-Base; M=qwen3.5-4b-base; LOG=$MK/runs/screen_chain3.status
export GENCFG=$V3/templates/base_stop; CANDS=${CANDS:-2 3 4 5 6 7 9}; CONC=${CONC:-32}; CC_LONG=${CC_LONG:-16}
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
free_gpu() { U=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader); [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $U)" = 0 ]; }
stop_srv() { p=$(head -n 1 $1 | grep -o "pid [0-9]*" | cut -d' ' -f2); [ -n "$p" ] && { kill -- -$p 2>/dev/null; sleep 20; kill -9 -- -$p 2>/dev/null; }; }
G=" 2 3"; PORTS=" 18182 18183"
grab() {  # adds a server on each free, unlocked GPU in CANDS; returns 0 if any was added
  local x added=1
  for x in $CANDS; do
    echo " $G " | grep -q " $x " && continue
    free_gpu $x && mkdir $P/locks/gpu$x 2>/dev/null || continue
    ( MAXSEQS=$CONC setsid sh $ROOT/scripts/serve_4b_keep_thinking.sh $x 1818$x $BASE $M $EN/logs/serve_4b_stop_gpu$x.log < /dev/null > /dev/null 2>&1 & )
    for i in $(seq 90); do curl -s -m 5 http://127.0.0.1:1818$x/v1/models | grep -q "\"$M\"" && break; sleep 10; done
    if curl -s -m 5 http://127.0.0.1:1818$x/v1/models | grep -q "\"$M\""; then G="$G $x"; PORTS="$PORTS 1818$x"; added=0
    else say "server on GPU $x did not start"; stop_srv $EN/logs/serve_4b_stop_gpu$x.log; rmdir $P/locks/gpu$x; fi
  done
  return $added
}
run_pass() {  # <tasks> <db> <out> <runs> [max steps, 30] [episodes per server, CONC]: run until done, restarting with more servers when GPUs free up
  local T=$1 DB=$2 OUT=$3 k=$4 S=${5:-30} E=${6:-$CONC} RP urls n p
  while true; do
    n=$(echo $PORTS | wc -w); urls=""; for p in $PORTS; do urls="$urls --base-url http://127.0.0.1:$p/v1"; done
    setsid $PY -B $V3/run.py --tasks $T --db $DB --config H1 --runs $k --max-steps $S --concurrency $((E * n)) \
      --provider qwen --model $M $urls --out $OUT >> $OUT.log 2>&1 &
    RP=$!; say "run.py ($T, runs $k, $E per server) on GPUs$G"
    while kill -0 $RP 2>/dev/null; do
      sleep 300
      if [ $(echo $G | wc -w) -lt $(echo $CANDS | wc -w) ] && grab; then kill -- -$RP 2>/dev/null; sleep 5; kill -9 -- -$RP 2>/dev/null; say "restart with GPUs$G"; continue 2; fi
    done
    wait $RP; say "run.py exit $?: $(wc -l < $OUT/results.jsonl) rows"; return
  done
}
screen() {  # <tasks> <db> <out name> [episodes per server, CONC]: pass 1 on every task, pass 2 on the tasks pass 1 solved
  local T=$1 DB=$2 OUT=$MK/runs/$3 E=${4:-$CONC}
  run_pass $T $DB $OUT 1 30 $E
  $PY -B - $T $OUT/results.jsonl $OUT/pass2_tasks.jsonl <<'PYEOF'
import json, sys
tasks, res, out = sys.argv[1:]
ok = {}
for line in open(res):
    try: r = json.loads(line)
    except ValueError: continue
    if r["run"] == 1: ok[r["task_id"]] = r["passed"]
with open(out, "w") as f:
    for line in open(tasks):
        if ok.get(json.loads(line)["task_id"]): f.write(line)
PYEOF
  say "$3 pass 2 on $(wc -l < $OUT/pass2_tasks.jsonl) tasks"
  run_pass $OUT/pass2_tasks.jsonl $DB $OUT 2 30 $E
  say "$3 done"
}
say "chain3d starts (tier-1, wealth, ML, then the hard tier at $CC_LONG per server)"
screen $MK/tasks/market_tasks_train.jsonl $MK/market_v1.sqlite screen_base_train
screen $C/wealth/round1_tasks.jsonl $H/A.sqlite screen_base_wealth
screen $C/ml/round1_tasks.jsonl none screen_base_ml
screen $MK/tasks/market_tasks_hard_train.jsonl $MK/market_v1.sqlite screen_base_hard $CC_LONG
for x in $G; do stop_srv $EN/logs/serve_4b_stop_gpu$x.log; rmdir $P/locks/gpu$x; done
say "screen_chain3 done"
