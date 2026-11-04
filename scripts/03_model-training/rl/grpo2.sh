#!/bin/bash
# grpo2: the second GRPO run, from M1 on the 8-run screen's selection (research/log/2026-11-01-0010.md, 11-02 22:4x UTC: the
# ablation of patches 2 and 3 against grpo1 = run1 + run1b), launched like ppo_prep/ppo1.sh (gpu-claim, crash keep, stop
# by the driver's PID) with scripts/03_model-training/rl/run_grpo2.sh.
# - Settings, all run1b's (grpo_run1_resume7.sh and the driver's command line, log 11-02 08:3x UTC) but the data, the start
#   weights, the steps and the epochs cap: 6 GPUs, 96 tasks a step x 8 (rollout.n, run_grpo2.sh's default), actor
#   mini-batch MINI (default 96), lr LR (default 8.7e-7), KL loss 0.001 low_var_kl, clip 0.2/0.28, seq-mean-token-mean,
#   response budget 28,672, ppo_max_token_len_per_gpu 32,768, validation (200 dev tasks) and a checkpoint every 4 steps
#   with the last 2 kept, no validation before training, Ray with 64 CPUs and its temp dir in ${FP_TMP:-/tmp}/ray.
#   The data are TRAIN and VAL; the start weights M1 (not run1/hf_step4); 28 steps (run1 4 + run1b 24 updates).
# - DYN=1 turns on run_grpo2.sh's dynamic group filter (groups whose 8 rewards are equal are dropped and refilled, with
#   run_grpo2.sh's defaults REFILL 2.0, CAP 3 x 96 prompts a step, INFLIGHT 1); DYN=0 is run_grpo.sh's sampling. DYN, LR
#   and MINI are required (rl_chain_c.sh reads them from rl_chain_c.conf when grpo2 starts) and recorded in the status file.
# - EPOCHS, only a cap on verl's loop, holds 28 steps: DYN=0 ceil(28 / (rows // 96)), as verl drops an epoch's last partial
#   batch; DYN=1 ceil(28 x 288 / rows) + 1, since a step may draw up to CAP = 288 prompts.
# - GPUs: NG=6 of 0-9 without a free8.sh lock through gpuclaim_lib.sh's gc_claim (gpu_claim.py claim, holders of its
#   default size until the run's processes appear); KEPT=<6 GPUs> first takes over GPUs kept by the chain's previous stage
#   or by a failed earlier run, and claims 6 afresh when gpu_claim.py refuses them. A failure within 20 minutes while
#   another user's process is on the run's GPUs releases them and starts once more on 6 claimed afresh; any other failure
#   keeps them for the fixed run (status file: restart and release commands). KEEP_AT_END=<file>: a run that ends well
#   keeps its GPUs for the chain's next stage (holders of KEEP_HOLD_GB GiB, default gpu_claim.py's) and lists them in
#   <file>; otherwise a success, the stop command and a signal release them.
# - Once the GPUs are claimed and before training starts, the root disk must have more than ROOT_MIN_GB (default 30) GB free, the output disk
#   MIN_DISK2_GB (default 300) and MemAvailable MIN_MEM_GB (default 200) GB, else the run is not started (GPUs released).
# - Without RESUME=1 the run starts fresh (trainer.resume_mode=disable) and refuses an rl/<EXP>/ckpt that holds
#   checkpoints; RESUME=1 resumes from it (auto). An existing metrics.jsonl is copied aside first (verl opens it "wb").
# Usage: DYN=0|1 LR=<lr> MINI=<tasks> TRAIN=<parquet> VAL=<parquet> [RESUME=1] [EXP=grpo2] [KEPT=<GPUs>] [KEEP_AT_END=<file>]
#          [EXTRA='<hydra overrides>'] setsid nohup bash grpo2.sh > grpo2.out 2>&1 < /dev/null &
#        Status rl/<EXP>.status: "<EXP> exit 0", then "<EXP> done", on success; training log rl/<EXP>.log.
#        Stop: bash grpo2.sh stop (same EXP), or kill <grpo2.sh pid>: TERM to the main_ppo driver, up to 3 minutes, then
#        KILL what is left of run_grpo2.sh; our user's processes other than our holders still on the run's GPUs 3 minutes
#        later are killed (Ray workers run in their own sessions); the GPUs are released.
#        DRY=1: no lock, copy or training (G1..G6 stand for the GPUs); run_grpo2.sh is copied under DRYDIR (default
#        $DRY_DIR/) with its python replaced by scripts/03_model-training/rl/py_cfg_job.sh and its output dir moved there, and hydra
#        prints the composed job config to DRYDIR/<EXP>_DYN<n>.yaml; DRY_NTASKS (default 939) rows when TRAIN is missing.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); . "$ROOT/scripts/paths.sh"
RL=$FP_RUNS/rl; S=$ROOT/src/03_model-training/code/verl_grpo
M1=$FP_RUNS/qwen3.5-4b-base-en/M1
source $ROOT/scripts/gpuclaim_c_lib.sh  # gc_claim, gc_keep_for_restart, gc_release, contended, LOCKS (gpuclaim_lib.sh); gc_keep_handover, margins_ok
NG=6; STEPS=28; BATCH=96; RESP=28672
GPUL=auto; EXP=${EXP:-grpo2}; KEPT=$(spaces $KEPT)
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # GPU indices are nvidia-smi's
echo "$EXP" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]*$' || { echo "EXP must be letters, digits, '.', '_' or '-': $EXP" >&2; exit 2; }
if [ "$1" != stop ]; then
  case "$DYN" in 0|1) ;; *) echo "DYN must be 0 or 1: '$DYN'" >&2; exit 2;; esac
  echo "$LR" | grep -Eq '^[0-9]+(\.[0-9]+)?([eE]-?[0-9]+)?$' || { echo "LR must be a positive number: '$LR'" >&2; exit 2; }
  case "$MINI" in ''|*[!0-9]*|0*) echo "MINI must be a whole number: '$MINI'" >&2; exit 2;; esac
  { [ "$MINI" -ge 1 ] && [ $((BATCH % MINI)) = 0 ] && [ $((MINI * 8 % NG)) = 0 ]; } \
    || { echo "MINI must divide $BATCH, and MINI x 8 must divide by $NG GPUs: $MINI" >&2; exit 2; }
  [ -n "$TRAIN" ] && [ -n "$VAL" ] || { echo "TRAIN and VAL (parquet files) are required" >&2; exit 2; }
  if [ -f "$TRAIN" ]; then NT=$(CUDA_VISIBLE_DEVICES= $FP_VENVS/verl/bin/python -B -c 'import sys, pyarrow.parquet as pq; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' $TRAIN)
  elif [ -n "$DRY" ]; then NT=${DRY_NTASKS:-939}
  else echo "no TRAIN file: $TRAIN" >&2; exit 2; fi
  [ -n "$DRY" ] || [ -f "$VAL" ] || { echo "no VAL file: $VAL" >&2; exit 2; }
  case "$NT" in ''|*[!0-9]*) echo "cannot count the rows of $TRAIN" >&2; exit 2;; esac
  [ $((NT / BATCH)) -ge 1 ] || { echo "TRAIN has $NT rows, fewer than one batch of $BATCH" >&2; exit 2; }
  if [ $DYN = 1 ]; then EPOCHS=$(( (STEPS * 3 * BATCH + NT - 1) / NT + 1 )); else EPOCHS=$(( (STEPS + NT / BATCH - 1) / (NT / BATCH) )); fi
fi
if [ -n "$KEPT" ]; then
  for x in $KEPT; do case $x in [0-9]) ;; *) echo "KEPT: not a GPU of this host (0-9): $x" >&2; exit 2;; esac; done
  [ $(echo $KEPT | wc -w) = $NG ] || { echo "KEPT: $NG GPUs, not $(echo $KEPT | wc -w)" >&2; exit 2; }
fi
unset KL N MEM EAGER PREFIX WORKERS AGG FILTER FIN_MAX_STEPS OFFLOAD REFILL CAP INFLIGHT MODEL VAL_BATCH  # run_grpo2.sh's defaults (run1b's), or set below
if [ -n "$DRY" ]; then SD=${DRYDIR:-$DRY_DIR}; mkdir -p $SD; OUT=$SD/${EXP}_outd; else SD=$RL; OUT=$RL/$EXP; fi
LOG=$SD/$EXP.status; PIDF=$SD/$EXP.pid; GF=$SD/$EXP.gpus  # GF: the GPUs this run holds, for the stop command
GC_LOG=$SD/$EXP.gpuclaim.log; STOPF=$SD/$EXP.stopping  # STOPF: written by the stop command, so that its end releases the GPUs
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; [ -n "$DRY" ] && echo "# $1"; return 0; }
free_gpu() { [ -z "$(ours_on $1)" ]; }  # no process of ours on it but our holders
busy() { local x b=""; for x in $GPUL; do free_gpu $x || b="$b $x"; done; echo $b; }
nap() { sleep $1 & wait $!; }  # a signal reaches the trap at once
alive() { kill -0 $1 2>/dev/null; }
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }  # a process's descendants, deepest first
wait_free() {  # <seconds>: until the GPUs held (GPUL) have no process of ours but holders; false when the time runs out
  local end=$(( $(date +%s) + $1 ))
  until [ -z "$(busy)" ]; do [ $(date +%s) -ge $end ] && return 1; nap 5; done
}
clear_gpus() {  # after the driver ended: our user's processes other than our holders still on GPUL after 3 minutes are killed
  local q k=""
  wait_free 180 && return 0
  for q in $(ours_on $GPUL); do kill -9 $q 2>/dev/null && k="$k $q"; done
  say "GPUs $GPUL busy 3 minutes after the driver ended; killed leftover pids of our user:$k"
  wait_free 120 || say "GPUs still busy 2 minutes later:$(busy)"
}
stop_training() {  # TERM the main_ppo driver (run_grpo2.sh's child), wait up to 3 minutes, then KILL run_grpo2.sh's tree
  local tp q i
  [ -n "$RP" ] && alive $RP || return 0
  tp=$(pgrep -P $RP | head -n 1); [ -n "$tp" ] && kill -TERM $tp
  for i in $(seq 36); do alive $RP || return 0; nap 5; done
  for q in $(tree $RP) $RP; do kill -9 $q 2>/dev/null; done; say "killed what was left of run_grpo2.sh"
}

if [ "$1" = stop ]; then
  read MAIN RP 2>/dev/null < $PIDF; GPUL=$(cat $GF 2>/dev/null)  # the GPUs the running instance locked
  case "$(ps -o args= -p ${MAIN:-0} 2>/dev/null)" in *"$(basename $0)"*) ;; *) echo "$EXP is not running (no live pid in $PIDF)" >&2; exit 1;; esac
  : > $STOPF  # the training's end is then a stop, which releases the GPUs, not a failure, which keeps them
  TP=""; [ -n "$RP" ] && alive $RP && TP=$(pgrep -P $RP | head -n 1)
  if [ -n "$TP" ]; then
    say "$EXP stop: TERM to the main_ppo driver (pid $TP)"; stop_training
  else say "$EXP stop: TERM to grpo2.sh (pid $MAIN), no training running"; kill -TERM $MAIN; fi
  for i in $(seq 120); do alive $MAIN || break; sleep 5; done  # its own end clears its GPUs and frees the locks
  if alive $MAIN; then
    for q in $(tree $MAIN) $MAIN; do kill -9 $q 2>/dev/null; done; say "$EXP stop: grpo2.sh still running 10 minutes later, killed"
    [ -n "$RP" ] && clear_gpus
    if [ -n "$GPUL" ]; then gc release $(commas $GPUL); for x in $GPUL; do rmdir $LOCKS/gpu$x 2>/dev/null; done; say "GPUs released: $GPUL"; fi
    rm -f $GF $STOPF
  fi
  say "$EXP stopped"; exit 0
fi

STARTED=""; RP=""
release() {  # on exit: after a started run, its GPUs cleared of leftovers; then the GPUs still claimed are released
  trap - EXIT; trap '' INT TERM HUP  # a further signal must not cut this short
  if [ -z "$DRY" ]; then
    gc_stop_claim
    [ -n "$STARTED" ] && clear_gpus
    [ -n "$GC_HELD" ] && say "GPUs released: $GC_HELD"; gc_release  # GPUs kept for a restart or the next stage are no longer in GC_HELD
  fi
  rm -f $PIDF $GF $STOPF
}
trap release EXIT; trap 'trap "" INT TERM HUP; say "$EXP stopped by a signal"; stop_training; exit 143' INT TERM HUP
[ -n "$DRY" ] && : > $LOG
if [ -z "$DRY" ] && [ -f $PIDF ] && read OP _ < $PIDF && [ "$OP" != $$ ] && case "$(ps -o args= -p ${OP:-0} 2>/dev/null)" in *"$(basename $0)"*) true;; *) false;; esac; then
  echo "another $EXP launcher runs (pid $OP)" >&2; trap - EXIT; exit 2
fi
echo $$ > $PIDF; rm -f $GF $STOPF  # no record of an earlier instance's GPUs or stop survives into this one
if [ "$RESUME" = 1 ]; then RARG=trainer.resume_mode=auto
else
  RARG=trainer.resume_mode=disable
  [ -f $OUT/ckpt/latest_checkpointed_iteration.txt ] && { say "$EXP not started: $OUT/ckpt holds checkpoints (RESUME=1 resumes them; or move the directory aside)"; exit 1; }
fi
[ -n "$DRY" ] || [ -f $GC ] || { say "$EXP not started: gpu-claim is not installed (no $GC)"; exit 1; }
AVAIL=$(df -P / | awk 'NR == 2 {print int($4 / 1048576)}')  # whole GiB rounded down, as margins_ok
[ "$AVAIL" -gt ${ROOT_MIN_GB:-30} ] || { say "root disk has ${AVAIL} GB free (Ray's temp dir is there; needs more than ${ROOT_MIN_GB:-30}): $EXP not started"; exit 1; }
if [ -n "$DRY" ]; then
  RUNG=$SD/run_grpo2_cfgcopy.sh; TLOG=$SD/${EXP}_DYN$DYN.yaml; export CFG_ARGS=$SD/${EXP}_DYN$DYN.args
  sed -e "s#^HERE=.*#HERE=$S#" -e "s#^PY=.*#PY=$ROOT/scripts/03_model-training/rl/py_cfg_job.sh#" -e "s#^OUTD=.*#OUTD=$OUT; mkdir -p \$OUTD#" $ROOT/scripts/03_model-training/rl/run_grpo2.sh > $RUNG
else RUNG=$ROOT/scripts/03_model-training/rl/run_grpo2.sh; TLOG=$RL/$EXP.log; fi
export NCCL_P2P_DISABLE=1 RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"'"$FP_CACHE"'/ray_spill"}}'
say "${DRY:+(dry run) }$EXP settings: DYN=$DYN LR=$LR MINI=$MINI; TRAIN $TRAIN ($NT tasks, $((NT / BATCH)) batches of $BATCH an epoch), VAL $VAL, $STEPS steps, EPOCHS cap $EPOCHS, from $M1"
CANDS="0 1 2 3 4 5 6 7 8 9"
for ATTEMPT in 1 2; do
  if [ -n "$DRY" ]; then
    GPUL="G1 G2 G3 G4 G5 G6"
    say "(dry run) would claim $NG of GPUs $CANDS without a free8.sh lock (now: $(f8_free $CANDS)) with $GC, holders of its default size${KEPT:+, first taking over the kept GPUs $KEPT}; $GPUL stand for them"
  else
    say "$EXP claims $NG of GPUs $CANDS without a free8.sh lock (attempt $ATTEMPT)${KEPT:+, first taking over the kept GPUs $KEPT}"
    gc_claim $NG "" "$CANDS" "$KEPT" || exit 1
    GPUL=$GC_HELD; echo $GPUL > $GF; say "GPUs claimed: $GPUL"
    [ $ATTEMPT = 2 ] && [ -f $TLOG ] && mv $TLOG $TLOG.attempt1
    margins_ok ${ROOT_MIN_GB:-30} ${MIN_DISK2_GB:-300} ${MIN_MEM_GB:-200} || { say "$EXP not started: $MARGINS (needs > ${ROOT_MIN_GB:-30}, ${MIN_DISK2_GB:-300}, ${MIN_MEM_GB:-200}); GPUs released"; exit 1; }
    say "margins: $MARGINS"
  fi
  GL=$(commas $GPUL)  # run_grpo2.sh's GPUS (it counts them for n_gpus_per_node)
  if [ -f $OUT/metrics.jsonl ]; then
    B=$OUT/metrics_before_$(date -u +%m%d%H%M%S).jsonl
    if [ -n "$DRY" ]; then say "(dry run) would copy $OUT/metrics.jsonl to $B"; else cp $OUT/metrics.jsonl $B && say "copied metrics.jsonl to $B"; fi
  fi
  say "${DRY:+(dry run) }$EXP starts on GPUs $GL: DYN=$DYN, $RARG${EXTRA:+, extra overrides: $EXTRA}"
  STARTED=1; T0=$(date +%s)
  MODEL=$M1 GPUS=$GL EXP=$EXP TRAIN=$TRAIN VAL=$VAL BATCH=$BATCH MINI=$MINI LR=$LR STEPS=$STEPS EPOCHS=$EPOCHS DYN=$DYN \
    SAVE_FREQ=4 TEST_FREQ=4 VAL_BEFORE=False VAL_BATCH=200 RESP=$RESP TOKLEN=32768 \
    bash $RUNG ray_kwargs.ray_init.num_cpus=64 +ray_kwargs.ray_init._temp_dir=${FP_TMP:-/tmp}/ray \
    trainer.max_actor_ckpt_to_keep=2 $RARG $EXTRA > $TLOG 2>&1 &
  RP=$!; echo "$$ $RP" > $PIDF; wait $RP; RC=$?
  say "$EXP exit $RC"
  [ -n "$DRY" ] && break
  holder_lines $GPUL
  [ $RC = 0 ] && [ ! -f $STOPF ] && [ -n "$KEEP_AT_END" ] && gc_keep_handover "$KEEP_HOLD_GB" "$KEEP_AT_END"
  { [ $RC = 0 ] || [ -f $STOPF ]; } && break
  if [ $ATTEMPT = 1 ] && [ $(( $(date +%s) - T0 )) -lt 1200 ] && contended $GPUL; then
    say "$EXP failed within 20 minutes, and another user's process is on GPUs $GL: released, claiming again"
    clear_gpus; STARTED=""; gc_release; KEPT=""; continue
  fi
  gc_keep_for_restart "" "DYN=$DYN LR=$LR MINI=$MINI TRAIN=$TRAIN VAL=$VAL EXP=$EXP bash $0 (RESUME=1 to resume from its checkpoints)"
  clear_gpus; STARTED=""; break
done
release
say "$EXP done"
exit $RC
