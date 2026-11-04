#!/bin/bash
# The reworded market test for rl_chain_c.sh's stage 5c (research/log/2026-11-01-0010.md, 11-02 23:1x and 23:2x UTC):
# eval_reworded.sh's runs of M1 and the official Qwen3.5-4B, and the same run of each RL model in RL_MODELS (default
# "grpo1 grpo2 ppo2"; empty for none), one model after another on the GPUs this script holds. eval_reworded.sh cannot
# run inside the chain: its guard refuses while any process of ours names rl_chain.
# Every model's run has eval_reworded.sh's settings, so its rows pair per task with the model's original
# test_hs_market_<model>_H1 rows: hardset_v1_reworded/test_market.jsonl, harness v3, serve_4b_keep_thinking.sh without
# GENCFG, one server per GPU, H1, runs 1, max steps 100, provider qwen; M1 (served M1) at 24 and the official model
# (served qwen3.5-4b) at 16 episodes per server, as eval_reworded.sh; each RL model rl/hf_<name> (served <name>) at 24,
# as eval_ckpt_8gpu.sh served it for its original rows. Outputs runs/test_hsrw_market_<model>_H1 (run.py resumes an
# existing one, so M1's and the official model's are the ones eval_reworded.sh would write). Then compare_reworded.sh
# (M1 and the official model) and compare_reworded_rl.sh (each RL model: original -> reworded, and its gain over M1 on
# both sets).
# - GPUs: KEPT=<GPUs> takes over GPUs kept for this stage; otherwise NG of 0-9 without a free8.sh lock are claimed. NG
#   is required without KEPT: the caller sets it within the cap, since a lock snapshot cannot see a claim of ours still
#   waiting, such as the judge's (rl_chain_c.sh passes 4 while its judge runs, else 6); it is lowered to 6 less the GPUs
#   our jobs already use when that is smaller. Holders of GCHOLD GiB (default 4) until the
#   servers appear: vllm-0.32 needs 0.85 of the GPU's 48 GiB free (40.8 GiB). Between two models: gpu_claim.py keep,
#   up to 10 minutes for the servers' processes to leave, then claim --kept. A model whose servers do not start while
#   another user's process is on its GPUs gets fresh GPUs once; any other failure (no server, a run.py exit other than
#   0) keeps the GPUs for the fixed job (status: restart and release commands) and exits 1. Every other end, a signal
#   included, stops the servers by process group and releases the GPUs.
# Usage: KEPT=<GPUs> | NG=<n> [RL_MODELS="grpo1 grpo2 ppo2"] [GCHOLD=4] bash reworded_c.sh   Stop: kill <pid> (not -9).
#        Status runs/reworded_c.status. DRY=1 prints the commands and starts, claims and signals nothing (status under
#        DRYDIR, default $DRY_DIR/).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
EN=$FP_DATA_EN; MK=$FP_DATA_D2/market_v1
V3=$ROOT/src/02_harness/code/v3; MD=$FP_MODELS_D2
HERE=$ROOT/scripts/04_evaluation
R=$FP_RUNS/qwen3.5-4b-base-en; RL=$FP_RUNS/rl
PY=$FP_VENVS/vllm-0.12/bin/python
T=$MK/hardset_v1_reworded/test_market.jsonl; DB=$MK/market_v1.sqlite; CAP=6; GCHOLD=${GCHOLD:-4}; ALL="0 1 2 3 4 5 6 7 8 9"
RL_MODELS=${RL_MODELS-grpo1 grpo2 ppo2}
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "$DRY" ]; then SD=${DRYDIR:-$DRY_DIR}; mkdir -p $SD; else SD=$MK/runs; fi
LOG=$SD/reworded_c.status; GC_LOG=$SD/reworded_c.gpuclaim.log; PIDF=$SD/reworded_c.pid
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; [ -n "$DRY" ] && echo "# $1"; return 0; }
nap() { sleep $1 & wait $!; }
source $ROOT/scripts/gpuclaim_c_lib.sh  # gc_claim, gc_keep, gc_keep_for_restart, gc_release, contended, ours_on, holder_lines; our_gpus
KEPT=$(spaces $KEPT)
for x in $KEPT; do case $x in [0-9]) ;; *) echo "KEPT: not a GPU of this host (0-9): $x" >&2; exit 2;; esac; done
[ -n "$KEPT" ] || case "$NG" in [1-6]) ;; *) echo "NG (1-6, the GPUs to claim) is required without KEPT: '$NG'" >&2; exit 2;; esac

declare -A DIR K
DIR[M1]=$R/M1; K[M1]=24; DIR[qwen3.5-4b]=$MD/Qwen3.5-4B; K[qwen3.5-4b]=16
for m in $RL_MODELS; do
  case $m in grpo1|grpo2|grpo3|grpo4|ppo2) DIR[$m]=$RL/hf_$m; K[$m]=24;; *) echo "RL_MODELS: grpo1, grpo2, grpo3, grpo4 or ppo2, not $m" >&2; exit 2;; esac
done
MODELS="M1 qwen3.5-4b $RL_MODELS"
slog() { echo $EN/logs/serve_4b_hsrw_${1}_gpu$2.log; }  # <served name> <GPU>, as eval_reworded.sh names them
port() { echo 1823$1; }  # <GPU>: one model at a time, so one port per GPU
up() { for i in $(seq 90); do curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "\"$2\"" && return 0; nap 10; done; return 1; }
port_used() { ss -Hltn "( sport = :$1 )" | grep -q .; }
SLOGS=""; RPID=""; HELD=""
stop_srvs() {  # <serve logs>: each server's process group (pid on the log's first line), TERM, then KILL after 20 s
  local l p ps=""
  [ -n "$DRY" ] && return 0
  for l in $1; do p=$(head -n 1 $l 2>/dev/null | grep -o "pid [0-9]*" | cut -d' ' -f2); [ -n "$p" ] && { kill -- -$p 2>/dev/null; ps="$ps $p"; }; done
  [ -n "$ps" ] && { sleep 20; for p in $ps; do kill -9 -- -$p 2>/dev/null; done; }
  return 0
}
cleanup() {
  trap - EXIT INT TERM HUP
  [ -n "$DRY" ] && return
  [ -n "$RPID" ] && kill -- -$RPID 2>/dev/null
  stop_srvs "$SLOGS"; gc_stop_claim; [ -n "$GC_HELD" ] && say "GPUs released: $GC_HELD"; gc_release
  rm -f $PIDF
}
trap cleanup EXIT; trap 'say "reworded_c stopped by a signal"; exit 143' INT TERM HUP
[ -n "$DRY" ] && : > $LOG
if [ -z "$DRY" ] && [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null && tr '\0' ' ' < /proc/$(cat $PIDF)/cmdline 2>/dev/null | grep -q reworded_c.sh; then
  echo "another reworded_c.sh runs (pid $(cat $PIDF))" >&2; trap - EXIT; exit 2
fi
[ -n "$DRY" ] || echo $$ > $PIDF
[ -f $T ] || { say "no reworded test set: $T"; exit 1; }
for m in $RL_MODELS; do [ -n "$DRY" ] || [ -f ${DIR[$m]}/config.json ] || { say "no merged weights for $m: ${DIR[$m]} (eval_ckpt_c.sh merges them)"; exit 1; }; done

claim() {  # [kept GPUs] [count]: HELD := the GPUs claimed or taken over; when none are kept, count GPUs, by default NG
  local n
  if [ -n "$1" ]; then n=$(echo $1 | wc -w)
  elif [ -n "$2" ]; then n=$2
  else
    if [ -n "$DRY" ]; then n=$NG; else n=$((CAP - $(our_gpus | wc -l))); [ $NG -lt $n ] && n=$NG; fi
    [ $n -ge 1 ] || { say "refused: our jobs already use $CAP GPUs (the project's cap)"; exit 1; }
  fi
  if [ -n "$DRY" ]; then HELD=$(for i in $(seq $n); do printf 'G%s ' $i; done); HELD=$(echo $HELD)
    say "(dry run) would claim $n of GPUs $ALL without a free8.sh lock (now: $(f8_free $ALL)), holders of $GCHOLD GiB${1:+, taking over the kept GPUs $1}; $HELD stand for them"; return 0; fi
  say "claiming $n of GPUs $ALL without a free8.sh lock, holders of $GCHOLD GiB${1:+, taking over the kept GPUs $1}"
  gc_claim $n $GCHOLD "$ALL" "$1" || exit 1
  HELD=$GC_HELD; say "GPUs claimed: $HELD"
}
serve() {  # <model>: one server per held GPU; PORTS := those that answer, SLOGS := their logs; false when none answers
  local m=$1 x l cmd ok=""
  PORTS=""; SLOGS=""
  for x in $HELD; do
    l=$(slog $m $x)
    if [ -z "$DRY" ] && port_used $(port $x); then say "$m: port $(port $x) in use, GPU $x left out"; continue; fi
    cmd="( GENCFG= MAXSEQS=${K[$m]} setsid sh $ROOT/scripts/serve_4b_keep_thinking.sh $x $(port $x) ${DIR[$m]} $m $l < /dev/null > /dev/null 2>&1 & )"
    if [ -n "$DRY" ]; then echo "$cmd"; else eval "$cmd"; fi
    SLOGS="$SLOGS $l"
  done
  for x in $HELD; do
    l=$(slog $m $x); case " $SLOGS " in *" $l "*) ;; *) continue;; esac
    if [ -n "$DRY" ] || up $(port $x) $m; then ok="$ok $x"; PORTS="$PORTS $(port $x)"
    else say "$m server on GPU $x did not start (see $l)"; stop_srvs $l; SLOGS=$(echo $SLOGS | tr ' ' '\n' | grep -vx "$l" | tr '\n' ' '); fi
  done
  [ -n "$DRY" ] || holder_lines $HELD
  [ -n "$ok" ] && say "$m servers on GPUs$ok"
  [ -n "$ok" ]
}
handover() {  # between two models: keep, wait up to 10 minutes for the servers' processes to leave, take over
  local end x busy=""
  [ -n "$DRY" ] && return 0
  gc_keep $GCHOLD
  end=$(( $(date +%s) + 600 ))
  for x in $HELD; do until [ -z "$(ours_on $x)" ] || [ $(date +%s) -ge $end ]; do nap 5; done; [ -z "$(ours_on $x)" ] || busy="$busy $x"; done
  [ -n "$busy" ] && say "processes of ours still on GPUs$busy 10 minutes after the servers stopped"
  claim "$HELD"
}

say "reworded_c starts (pid $$): models $MODELS${KEPT:+, taking over the kept GPUs $KEPT}"
claim "$KEPT"; RETRIED=0; first=1
for m in $MODELS; do
  [ $first = 1 ] || handover; first=0
  if ! serve $m; then
    stop_srvs "$SLOGS"; SLOGS=""
    if [ -z "$DRY" ] && [ $RETRIED = 0 ] && contended $HELD; then
      RETRIED=1; n=$(echo $HELD | wc -w); say "$m: another user's process is on GPUs $HELD; released, claiming $n afresh"
      gc_release; claim "" $n; serve $m || { stop_srvs "$SLOGS"; SLOGS=""; say "reworded_c failed: no $m server started"; gc_keep_for_restart $GCHOLD "bash $0"; exit 1; }
    else say "reworded_c failed: no $m server started"; [ -z "$DRY" ] && { gc_keep_for_restart $GCHOLD "bash $0"; exit 1; }; fi
  fi
  urls=""; for p in $PORTS; do urls="$urls --base-url http://127.0.0.1:$p/v1"; done
  out=test_hsrw_market_${m}_H1
  cmd="$PY -B $V3/run.py --tasks $T --db $DB --config H1 --runs 1 --max-steps 100 --concurrency $(( ${K[$m]} * $(echo $PORTS | wc -w) )) --provider qwen --model $m$urls --out $MK/runs/$out >> $MK/runs/$out.log 2>&1"
  if [ -n "$DRY" ]; then echo "setsid $cmd &"; rc=0
  else eval "setsid $cmd &"; RPID=$!; say "$m run.py started (pid $RPID)"; wait $RPID; rc=$?; RPID=""; fi
  [ -n "$DRY" ] || say "$m $out exit $rc: $(wc -l < $MK/runs/$out/results.jsonl 2>/dev/null || echo 0) rows, $(grep -c '"passed": true' $MK/runs/$out/results.jsonl 2>/dev/null) passed"
  stop_srvs "$SLOGS"; SLOGS=""
  [ $rc = 0 ] || { say "reworded_c failed: $m run.py exit $rc (run.py resumes)"; gc_keep_for_restart $GCHOLD "bash $0"; exit 1; }
done
[ -n "$DRY" ] || { say "GPUs released: $GC_HELD"; gc_release; HELD=""; }
if [ -n "$DRY" ]; then echo "bash $HERE/compare_reworded.sh"; echo "RL_MODELS=\"$RL_MODELS\" bash $HERE/compare_reworded_rl.sh"
else
  bash $HERE/compare_reworded.sh >> $LOG 2>&1; say "compare_reworded.sh exit $?: $MK/runs/test_hsrw_market.compare.txt"
  if [ -n "$RL_MODELS" ]; then RL_MODELS="$RL_MODELS" bash $HERE/compare_reworded_rl.sh >> $LOG 2>&1; say "compare_reworded_rl.sh exit $?: $MK/runs/test_hsrw_market_rl.compare.txt"; fi
fi
say "reworded_c done"
