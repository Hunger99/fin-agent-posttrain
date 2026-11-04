#!/bin/bash
# General-capability evaluations for rl_chain_c.sh's stage 5b (research/log/2026-11-01-0010.md, 11-02 22:4x and 23:2x UTC):
# eval_general.sh's jobs for several models one after another on GPUs this script holds. Each model's jobs are exactly
# eval_general.sh's: its DRY=1 mode writes <out dir>/jobs.sh (MMLU, GSM8K and IFEval, one per GPU, or IFEval alone for
# ifeval_only), which runs here with CUDA_VISIBLE_DEVICES set to the held GPUs and eval_general.sh's environment (its
# line 20: HF_HOME, VLLM_USE_FLASHINFER_SAMPLER=0, and vllm-0.32's bin first on PATH, where FlashInfer's JIT finds ninja),
# and then eval_general.sh's last step, lmeval_items.py on the GSM8K and IFEval samples. Only the GPU handling differs from eval_general.sh's own run, which
# claims fresh GPUs for each model, cannot take over GPUs kept for it, does not see free8.sh's locks, and holds 24 GiB,
# more than vLLM's start check leaves on these 48 GiB GPUs at gpu_memory_utilization 0.6 (28.8 GiB must be free).
# - GPUs: N = 3 (all) or 1 (ifeval_only), through gpuclaim_lib.sh's gc_claim: KEPT=<N GPUs> first takes over GPUs kept
#   for this stage, else N of 0-9 without a free8.sh lock are claimed, each with a holder of GCHOLD GiB (default 4)
#   until a job's process appears. Between two models the GPUs stay ours: gpu_claim.py keep when a model's jobs end, up
#   to 10 minutes for their processes to leave, then claim --kept for the next model; a job that ends before the model's
#   others (GSM8K before MMLU and IFEval) has its GPU kept at once (gpu_claim.py keep), so that the GPU does not sit
#   locked but empty, looking free to other users, until the slowest job ends. KEEP_AT_END=<file>: after the last
#   model the GPUs are kept for the chain's next stage (holders of KEEP_HOLD_GB GiB, default GCHOLD) and listed in <file>;
#   otherwise they are released at the end, on a signal and when the script stops.
# - A model passes when each of its jobs wrote "<name> <job> exit 0" to eval_general's status file after it started.
#   A failed model whose GPUs carry another user's process in two checks a second apart gets fresh GPUs and one more
#   try; any other failure keeps the GPUs for the fixed run (status file: restart and release commands) and exits 1.
#   An earlier attempt's outputs in the model's out dir are moved to <out dir>/_attempt_<UTC time>/ first, since lm-eval
#   names its sample files by time and eval_general.sh's last step reads the first one it lists.
# - A model already recorded "general <name> done" in this status file is skipped; when all are, kept GPUs are still
#   taken over, then handed on (KEEP_AT_END) or released.
# Usage: [KEPT=<GPUs>] [KEEP_AT_END=<file>] [KEEP_HOLD_GB=G] [GCHOLD=4] bash general_c.sh <all|ifeval_only> <name>=<model dir> ...
#        Outputs rl/general_eval/<name>/ (jobs.sh, mmlu.jsonl, gsm8k/, ifeval/, gsm8k.jsonl, ifeval.jsonl), eval_general's
#        status $EN/logs/eval_general_<name>.status, this script's status rl/general_eval/general_c_<all|ifeval_only>.status.
#        Stop: kill <pid> (not -9): the running jobs.sh session gets TERM (its process group), then the GPUs are released.
#        DRY=1: no GPU, lock or job; eval_general.sh's DRY=1 writes jobs.sh under DRYDIR/general/<name>/ for each model
#        whose directory exists, and the commands are printed.
WHAT=$1; shift
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
EN=$FP_DATA_EN; E=$ROOT/src/04_evaluation
RL=$FP_RUNS/rl
FP=$FP_VENVS/fin-posttrain/bin/python; EG=$ROOT/scripts/04_evaluation/eval_general.sh
GCHOLD=${GCHOLD:-4}
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
JOBENV="HF_HOME=$HOME/.cache/huggingface VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$FP_VENVS/vllm-0.32/bin:$PATH"  # eval_general.sh line 20's, for jobs.sh only
case "$WHAT" in all) N=3;; ifeval_only) N=1;; *) echo "usage: $0 <all|ifeval_only> <name>=<model dir> ..." >&2; exit 2;; esac
[ $# -ge 1 ] || { echo "usage: $0 <all|ifeval_only> <name>=<model dir> ..." >&2; exit 2; }
if [ -n "$DRY" ]; then SD=${DRYDIR:-$DRY_DIR}/general; else SD=$RL/general_eval; fi
mkdir -p $SD
LOG=$SD/general_c_$WHAT.status; GC_LOG=$SD/general_c_$WHAT.gpuclaim.log; PIDF=$SD/general_c_$WHAT.pid
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; [ -n "$DRY" ] && echo "# $1"; return 0; }
nap() { sleep $1 & wait $!; }
source $ROOT/scripts/gpuclaim_c_lib.sh  # gc_claim, gc_keep, gc_keep_for_restart, gc_release, contended, ours_on (gpuclaim_lib.sh); gc_keep_handover
KEPT=$(spaces $KEPT)
for x in $KEPT; do case $x in [0-9]) ;; *) echo "KEPT: not a GPU of this host (0-9): $x" >&2; exit 2;; esac; done
[ -z "$KEPT" ] || [ $(echo $KEPT | wc -w) = $N ] || { echo "KEPT: $N GPUs for $WHAT, not $(echo $KEPT | wc -w)" >&2; exit 2; }
for a in "$@"; do case "$a" in [A-Za-z0-9]*=/*) ;; *) echo "not <name>=<model dir>: $a" >&2; exit 2;; esac; done

JP=""; HELD=""
cleanup() {
  trap - EXIT INT TERM HUP
  [ -n "$DRY" ] && return
  [ -n "$JP" ] && { kill -TERM -- -$JP 2>/dev/null; for i in $(seq 30); do kill -0 -- -$JP 2>/dev/null || break; sleep 2; done; kill -KILL -- -$JP 2>/dev/null; }
  gc_stop_claim; [ -n "$GC_HELD" ] && say "GPUs released: $GC_HELD"; gc_release
  rm -f $PIDF
}
trap cleanup EXIT; trap 'say "general_c $WHAT stopped by a signal"; exit 143' INT TERM HUP
[ -n "$DRY" ] && : > $LOG
if [ -z "$DRY" ] && [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null && tr '\0' ' ' < /proc/$(cat $PIDF)/cmdline 2>/dev/null | grep -q general_c.sh; then
  echo "another general_c.sh $WHAT runs (pid $(cat $PIDF))" >&2; trap - EXIT; exit 2
fi
[ -n "$DRY" ] || echo $$ > $PIDF
say "general_c $WHAT starts (pid $$): $*; $N GPU(s), holders of $GCHOLD GiB${KEPT:+, taking over the kept GPUs $KEPT}"

claim() {  # [kept GPUs]: HELD := the GPUs claimed or taken over
  if [ -n "$DRY" ]; then HELD=$(for i in $(seq $N); do printf 'G%s ' $i; done); HELD=$(echo $HELD)
    say "(dry run) would claim $N of GPUs 0-9 without a free8.sh lock (now: $(f8_free 0 1 2 3 4 5 6 7 8 9)), holders of $GCHOLD GiB${1:+, taking over the kept GPUs $1}; $HELD stand for them"; return 0; fi
  gc_claim $N $GCHOLD "0 1 2 3 4 5 6 7 8 9" "$1" || exit 1
  HELD=$GC_HELD; say "GPUs claimed: $HELD"
}
handover() {  # between two models: keep the GPUs, wait up to 10 minutes for the last jobs' processes to leave, take them over
  local end x busy=""
  gc_keep $GCHOLD
  end=$(( $(date +%s) + 600 ))
  for x in $HELD; do until [ -z "$(ours_on $x)" ] || [ $(date +%s) -ge $end ]; do nap 5; done; [ -z "$(ours_on $x)" ] || busy="$busy $x"; done
  [ -n "$busy" ] && say "processes of ours still on GPUs$busy 10 minutes after the last jobs ended"
  claim "$HELD"
}
run_model() {  # <name> <model dir>: eval_general.sh's jobs on HELD; true when every job exited 0
  local name=$1 m=$2 out=$RL/general_eval/$1 q=$EN/logs/eval_general_$1.status n0 t ok=1 rc f a
  [ -n "$DRY" ] && out=$SD/$name
  if [ -n "$DRY" ] && { [ ! -f $m/config.json ] || ! ls $m/*.safetensors >/dev/null 2>&1; }; then
    say "(dry run) $name: $m holds no model yet; would run: DRY=1 bash $EG $name $m $RL/general_eval/$name $WHAT, then bash $RL/general_eval/$name/jobs.sh on its GPUs"; return 0; fi
  if [ -z "$DRY" ]; then
    a=""; for f in mmlu.jsonl gsm8k ifeval gsm8k.jsonl ifeval.jsonl; do [ -e $out/$f ] && a="$a $f"; done
    if [ -n "$a" ]; then d=$out/_attempt_$(date -u +%m%d%H%M%S); mkdir -p $d; for f in $a; do mv $out/$f $d/; done; say "$name: earlier outputs moved to $d:$a"; fi
  fi
  DRY=1 bash $EG $name $m $out $WHAT > $SD/$name.jobs.out 2>&1; rc=$?
  [ $rc = 0 ] && [ -f $out/jobs.sh ] && grep -qx "N=$N" $SD/$name.jobs.out || { say "$name: eval_general.sh DRY=1 did not write $N jobs (exit $rc; see $SD/$name.jobs.out)"; return 1; }
  if [ -n "$DRY" ]; then say "(dry run) $name: would run env CUDA_VISIBLE_DEVICES=<the $N GPUs> ${JOBENV%%PATH=*}PATH=$FP_VENVS/vllm-0.32/bin:\$PATH setsid bash $out/jobs.sh, keeping each job's GPU as soon as the job ends, then lmeval_items.py on gsm8k and ifeval"; cat $SD/$name.jobs.out; return 0; fi
  n0=$(cat $q 2>/dev/null | wc -l)
  local -a jobs=($(sed -n 's/^( echo "[^ ]* \([a-z0-9]*\) start gpu.*/\1/p' $out/jobs.sh)) gpus=($HELD); local i kept=""
  say "$name: jobs ${jobs[*]} start on GPUs $HELD ($out/jobs.sh)"
  env CUDA_VISIBLE_DEVICES=$(commas $HELD) $JOBENV setsid bash $out/jobs.sh < /dev/null > $SD/$name.jobs.log 2>&1 & JP=$!
  while kill -0 $JP 2>/dev/null; do  # jobs.sh gives its i-th job the i-th GPU
    for i in ${!jobs[@]}; do
      case " $kept " in *" ${gpus[i]} "*) continue;; esac
      tail -n +$((n0 + 1)) $q 2>/dev/null | grep -q "^$name ${jobs[i]} exit " || continue
      kill -0 $JP 2>/dev/null || break
      gc keep ${gpus[i]} --hold-gb $GCHOLD 2>> $GC_LOG && say "$name ${jobs[i]} ended: GPU ${gpus[i]} kept (holder of $GCHOLD GiB) while the other jobs run"
      kept="$kept ${gpus[i]}"
    done
    nap 10
  done
  wait $JP; JP=""
  for t in ${jobs[*]}; do
    tail -n +$((n0 + 1)) $q 2>/dev/null | grep -q "^$name $t exit 0 " || { ok=0; say "$name $t: no \"exit 0\" line in $q since the start"; }
  done
  for t in gsm8k ifeval; do f=$(ls $out/$t/*/samples_$t*.jsonl 2>/dev/null | head -n 1); [ -n "$f" ] && $FP -B $E/utils/lmeval_items.py $f $out/$t.jsonl >> $q 2>&1; done
  [ $ok = 1 ]
}

first=1; RETRIED=0
for a in "$@"; do
  name=${a%%=*}; m=${a#*=}
  if [ -z "$DRY" ] && grep -q "^general $name done " $LOG 2>/dev/null; then say "general $name already done: skipped"; continue; fi
  if [ $first = 1 ]; then claim "$KEPT"; first=0; elif [ -z "$DRY" ]; then handover; fi
  if run_model $name $m; then say "general $name done"; continue; fi
  if [ -z "$DRY" ] && [ $RETRIED = 0 ] && contended $HELD; then
    RETRIED=1; say "$name failed, and another user's process is on GPUs $HELD: released, claiming $N afresh, one more try"
    gc_release; claim; run_model $name $m && { say "general $name done"; continue; }
  fi
  say "general $name failed"
  [ -z "$DRY" ] && gc_keep_for_restart $GCHOLD "bash $0 $WHAT $*"
  exit 1
done
[ $first = 1 ] && [ -n "$KEPT" ] && claim "$KEPT"  # every model was done: the kept GPUs are still taken over, to hand on or release
if [ -n "$DRY" ]; then [ -n "$KEEP_AT_END" ] && say "(dry run) would keep GPUs $HELD for the chain's next stage (holders of ${KEEP_HOLD_GB:-$GCHOLD} GiB) and list them in $KEEP_AT_END"
elif [ -n "$GC_HELD" ]; then
  if [ -n "$KEEP_AT_END" ]; then gc_keep_handover "${KEEP_HOLD_GB:-$GCHOLD}" "$KEEP_AT_END" || gc_release
  else say "GPUs released: $GC_HELD"; gc_release; fi
fi
say "general_c $WHAT done"
