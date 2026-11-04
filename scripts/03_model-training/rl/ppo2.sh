#!/bin/bash
# ppo2.sh: ppo1.sh for the second PPO run, ppo2 (research/log/2026-11-01-0010.md, 11-02 22:4x UTC: "PPO 就直接按照补丁做来
# 执行"), every setting ppo1.sh's except: (1) the data are TRAIN=<parquet> and VAL=<parquet> (rl_chain_c.sh passes the
# 8-run screen's selection and run2_val.parquet), and EPOCHS, a cap only, is computed from TRAIN's row count so that the
# epochs hold the 31 batches: ceil(31 / (rows // 96)) (verl drops an epoch's last partial batch); (2) EXP defaults to ppo2;
# (3) KEEP_AT_END=<file>: when the run ends well, its GPUs are kept for the chain's next stage (gpu_claim.py keep,
# holders of KEEP_HOLD_GB GiB, default gpu_claim.py's) and listed in <file>, instead of released; (4) once the GPUs are
# claimed and before training starts, the root disk (Ray's temp dir) must have more than 30 GB free, the output disk
# MIN_DISK2_GB (default 300) and the host MemAvailable MIN_MEM_GB (default 200), else the run is not started and the
# GPUs are released; (5) DRY=1 writes under DRYDIR (default $DRY_DIR/), with DRY_NTASKS (default 939) rows when
# TRAIN does not exist yet; (6) the actor's learning rate and mini-batch and the critic's learning rate are ACTOR_LR,
# ACTOR_MINI and CRITIC_LR, by default ppo1.sh's 8.7e-7, 96 and 8.7e-6 (rl_chain_c.sh passes the conf file's PPO_LR,
# PPO_MINI and PPO_CRITIC_LR, so that ppo2's actor can follow a changed grpo2). ppo1.sh's header follows; changed lines
# are marked "ppo2".
# The first PPO run on M1 (research/log/2026-11-01-0010.md, 11-02 05:0x UTC), not started: once 6 GPUs are free
# (optionally after process WAIT_PID exits), lock them and run run_ppo.sh on run1's data.
# - GPUs (the same log, 11-02 07:4x UTC; the author: "改为十个 GPU，只要有八个空，就立刻启动，占住那八个"; 08:3x UTC:
#   "你把用八卡的删掉，后面都改成六卡，PPO 也是"; 07:5x and 08:3x UTC: the GPU-claim tool's holders by default): any of the host's
#   10 GPUs (0-9) may be used when no process of any user is on it. By default (GPUL=auto) the script claims NG=6 of
#   those that have no free8.sh lock with the GPU-claim tool's gpu_claim.py (gpuclaim_lib.sh), each with a
#   holder of gpu_claim.py's default size until the run's own processes appear there, and trains on them (GPUS=<their
#   comma list>); an explicit GPUL="<6 GPUs of 0-9>" is claimed whole once all of it is free. KEPT=<6 GPUs> first takes
#   over the GPUs that a failed earlier run kept; when gpu_claim.py refuses them (no longer kept, or another user's
#   process on one), they are released and 6 claimed afresh. A failure within 20 minutes of the start while another
#   user's process is on the run's GPUs in two checks a second apart releases them, and the run starts once more on 6
#   claimed afresh; any other failure keeps them for the fixed run (holders until a process of ours appears there or
#   gpu_claim.py's keep time passes; the status file gives the restart and release commands). A success, the stop
#   command and a signal release them. The GPUs held are recorded in <EXP>.gpus next to the status file for the stop
#   command; gpu_claim.py's messages go to <EXP>.gpuclaim.log.
# - Batch and learning rates for 6 GPUs (08:3x UTC), run1b's rule: run1's 16 tasks a GPU, so 96 tasks a step x 8 (768
#   trajectories), actor mini-batch 96 (one actor update a step), and each learning rate scaled by sqrt(96/128) as
#   usual for Adam: actor 1e-6 -> 8.7e-7, critic 1e-5 -> 8.7e-6; critic mini-batch 24 tasks, 4 critic updates a step as
#   32 of 128 gave. The other actor, reference, rollout and trainer settings are run1's: KL loss 0.001, clip 0.2/0.28,
#   validation and a checkpoint every 4 steps with the last 2 of each kept, and no validation before training (run1
#   measured M1 at 0.770).
# - 31 steps, the first 3 critic-only, so that the actor gets 28 updates, as many as GRPO run1 + run1b (4 + 24; 2,688
#   tasks against their 2,816; author 11-02 18:0x UTC), on batches 4-31 of the 4 epochs' batches (939 tasks // 96 = 9
#   an epoch, the rest dropped; EPOCHS=4 is only a cap, 36 batches); value clip 10, and the new value head initialised
#   with std 0.001 (run_ppo.sh). verl's rules hold on 6 GPUs: 768, the actor's 768 and the critic's 192 trajectories a
#   mini-batch divide by the data-parallel sizes (6; the actor's 3 under LONG=1's SP 2).
# - LONG=1 is the long-context configuration run1.status's decision line may choose: response budget 61,440, Ulysses
#   SP 2 for the actor and the reference, and the actor's parameters and optimizer offloaded, since the critic, which
#   cannot use SP, holds a whole 65,536-token sequence on one GPU. LONG=0 (default) is run1's: budget 28,672, no SP,
#   no offload.
# - Without RESUME=1 the run starts fresh (trainer.resume_mode=disable) and refuses a ckpt/ that holds checkpoints;
#   RESUME=1 resumes from it (auto). An existing metrics.jsonl is copied aside first (verl's file logger opens it "wb").
# - If GPU memory runs out, in order: OFFLOAD=True (the actor's offload, which LONG=1 already has), then
#   EXTRA='critic.model.enable_activation_offload=True' (a key critic.model has, so no '+'). EXTRA holds further hydra
#   overrides for run_ppo.sh.
# Usage (ppo2: TRAIN=<parquet> VAL=<parquet> [ACTOR_LR=..] [ACTOR_MINI=..] [CRITIC_LR=..] [KEEP_AT_END=<file>] first): [LONG=0|1] [RESUME=1] [WAIT_PID=<pid>] [OFFLOAD=True|False] [EXTRA='<overrides>'] [EXP=ppo1] [GPUL=auto|"<GPUs>"] \
#          [KEPT=<GPUs>] setsid nohup bash ppo1.sh > ppo1.out 2>&1 < /dev/null &
#        Status rl/<EXP>.status: "<EXP> exit 0", then "<EXP> done", on success; training log rl/<EXP>.log.
#        Stop: bash ppo1.sh stop (same EXP), as scripts/03_model-training/rl/pause_eval_step4.sh stopped run1: TERM the main_ppo driver, wait up to
#        3 minutes (then KILL what is left of run_ppo.sh), kill our user's processes other than our holders still on the
#        run's GPUs after 3 more minutes (found with nvidia-smi: Ray workers run in their own sessions) and release the
#        GPUs. kill <ppo1.sh pid> does the same.
#        DRY=1: no wait, lock, copy or training (GPUs G1..G6 stand for the 6 that auto would take); hydra prints the
#        composed job config to ppo_prep/dry/<EXP>_LONG<n>.yaml.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); . "$ROOT/scripts/paths.sh"
RL=$FP_RUNS/rl; G=$FP_DATA_D2/grpo
P=$FP_DATA_D2/harness_v2/pilot; S=$ROOT/src/03_model-training/code/verl_grpo
source $ROOT/scripts/gpuclaim_c_lib.sh  # gpuclaim_lib.sh's gc_claim, gc_keep_for_restart, gc_release, contended, LOCKS; gc_keep_handover, margins (ppo2)
NG=6  # GPUs a run takes; the batch and learning rates below are set for 6
GPUL=$(echo $GPUL | tr ',' ' '); GPUL=${GPUL:-auto}  # "auto" until the GPUs are claimed, then the GPUs held
EXP=${EXP:-ppo2}; LONG=${LONG:-0}; KEPT=$(spaces $KEPT)  # ppo2 (2)
STEPS=31; BATCH=96  # ppo2 (1): the steps and tasks a step of ppo1.sh, which EPOCHS must hold
ACTOR_LR=${ACTOR_LR:-8.7e-7}; ACTOR_MINI=${ACTOR_MINI:-96}; CRITIC_LR=${CRITIC_LR:-8.7e-6}  # ppo2 (6)
if [ "$1" != stop ]; then
  for k in ACTOR_LR CRITIC_LR; do  # ppo2 (6)
    echo "${!k}" | grep -Eq '^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$' && awk -v x="${!k}" 'BEGIN {exit !(x + 0 > 0 && x + 0 <= 1e-4)}' \
      || { echo "$k must be a number with 0 < $k <= 1e-4: '${!k}'" >&2; exit 2; }
  done
  case "$ACTOR_MINI" in ''|*[!0-9]*|0*) echo "ACTOR_MINI must be a whole number: '$ACTOR_MINI'" >&2; exit 2;; esac
  { [ "$ACTOR_MINI" -ge 1 ] && [ $((BATCH % ACTOR_MINI)) = 0 ] && [ $((ACTOR_MINI * 8 % NG)) = 0 ]; } \
    || { echo "ACTOR_MINI must divide $BATCH, and ACTOR_MINI x 8 must divide by $NG GPUs: $ACTOR_MINI" >&2; exit 2; }
  [ -n "$TRAIN" ] && [ -n "$VAL" ] || { echo "TRAIN and VAL (parquet files) are required" >&2; exit 2; }
  if [ -f "$TRAIN" ]; then NT=$(CUDA_VISIBLE_DEVICES= $FP_VENVS/verl/bin/python -B -c 'import sys, pyarrow.parquet as pq; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' $TRAIN)
  elif [ -n "$DRY" ]; then NT=${DRY_NTASKS:-939}
  else echo "no TRAIN file: $TRAIN" >&2; exit 2; fi
  [ -n "$DRY" ] || [ -f "$VAL" ] || { echo "no VAL file: $VAL" >&2; exit 2; }
  case "$NT" in ''|*[!0-9]*) echo "cannot count the rows of $TRAIN" >&2; exit 2;; esac
  [ $((NT / BATCH)) -ge 1 ] || { echo "TRAIN has $NT rows, fewer than one batch of $BATCH" >&2; exit 2; }
  EPOCHS=$(( (STEPS + NT / BATCH - 1) / (NT / BATCH) ))  # ppo2 (1): ceil(31 / (rows // 96))
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # GPU indices are nvidia-smi's
echo "$EXP" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]*$' || { echo "EXP must be letters, digits, '.', '_' or '-': $EXP" >&2; exit 2; }
case $LONG in 0|1) ;; *) echo "LONG must be 0 or 1: $LONG" >&2; exit 2;; esac
if [ "$GPUL" != auto ]; then
  for x in $GPUL; do case $x in [0-9]) ;; *) echo "GPUL: not a GPU of this host (0-9): $x" >&2; exit 2;; esac; done
  [ "$(echo $GPUL | tr ' ' '\n' | sort | uniq -d)" = "" ] || { echo "GPUL: a GPU is given twice: $GPUL" >&2; exit 2; }
  [ $(echo $GPUL | wc -w) = $NG ] || { echo "GPUL: $NG GPUs, not $(echo $GPUL | wc -w): the batch and learning rates are set for $NG" >&2; exit 2; }
fi
if [ -n "$KEPT" ]; then
  for x in $KEPT; do case $x in [0-9]) ;; *) echo "KEPT: not a GPU of this host (0-9): $x" >&2; exit 2;; esac; done
  [ $(echo $KEPT | wc -w) = $NG ] || { echo "KEPT: $NG GPUs, not $(echo $KEPT | wc -w)" >&2; exit 2; }
fi
unset LR KL N MEM EAGER PREFIX WORKERS AGG FILTER FIN_MAX_STEPS  # LR is set below; the others are run_ppo.sh's defaults (run1's)
if [ -n "$DRY" ]; then SD=${DRYDIR:-$DRY_DIR}; mkdir -p $SD; OUT=$SD/${EXP}_outd; else SD=$RL; OUT=$RL/$EXP; fi  # ppo2 (5)
LOG=$SD/$EXP.status; PIDF=$SD/$EXP.pid; GF=$SD/$EXP.gpus  # GF: the GPUs this run holds, for the stop command
GC_LOG=$SD/$EXP.gpuclaim.log; STOPF=$SD/$EXP.stopping  # STOPF: written by the stop command, so that its end releases the GPUs
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; [ -n "$DRY" ] && echo "# $1"; }
free_gpu() { [ -z "$(ours_on $1)" ]; }  # no process of ours on it but our holders (another user's is not ours to wait for)
busy() { local x b=""; for x in $GPUL; do free_gpu $x || b="$b $x"; done; echo $b; }
nap() { sleep $1 & wait $!; }  # a signal reaches the trap at once
alive() { kill -0 $1 2>/dev/null; }
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }  # a process's descendants, deepest first
wait_free() {  # <seconds>: until the GPUs held (GPUL) have no process of ours but holders; false when the time runs out
  local end=$(( $(date +%s) + $1 ))
  until [ -z "$(busy)" ]; do [ $(date +%s) -ge $end ] && return 1; nap 5; done
}
clear_gpus() {  # after the driver ended: our user's processes other than our holders still on the GPUs held (GPUL) after 3 minutes are killed
  local q k=""
  wait_free 180 && return 0
  for q in $(ours_on $GPUL); do kill -9 $q 2>/dev/null && k="$k $q"; done
  say "GPUs $GPUL busy 3 minutes after the driver ended; killed leftover pids of our user:$k"
  wait_free 120 || say "GPUs still busy 2 minutes later:$(busy)"
}
stop_training() {  # TERM the main_ppo driver (run_ppo.sh's child), wait up to 3 minutes, then KILL run_ppo.sh's tree
  local tp q i
  [ -n "$RP" ] && alive $RP || return 0
  tp=$(pgrep -P $RP | head -n 1); [ -n "$tp" ] && kill -TERM $tp
  for i in $(seq 36); do alive $RP || return 0; nap 5; done
  for q in $(tree $RP) $RP; do kill -9 $q 2>/dev/null; done; say "killed what was left of run_ppo.sh"
}

if [ "$1" = stop ]; then
  read MAIN RP 2>/dev/null < $PIDF; GPUL=$(cat $GF 2>/dev/null)  # the GPUs the running instance locked
  case "$(ps -o args= -p ${MAIN:-0} 2>/dev/null)" in *"$(basename $0)"*) ;; *) echo "$EXP is not running (no live pid in $PIDF)" >&2; exit 1;; esac
  : > $STOPF  # the training's end is then a stop, which releases the GPUs, not a failure, which keeps them
  TP=""; [ -n "$RP" ] && alive $RP && TP=$(pgrep -P $RP | head -n 1)
  if [ -n "$TP" ]; then
    say "$EXP stop: TERM to the main_ppo driver (pid $TP)"; stop_training
  else say "$EXP stop: TERM to ppo1.sh (pid $MAIN), no training running"; kill -TERM $MAIN; fi
  for i in $(seq 120); do alive $MAIN || break; sleep 5; done  # its own end clears its GPUs and frees the locks
  if alive $MAIN; then
    for q in $(tree $MAIN) $MAIN; do kill -9 $q 2>/dev/null; done; say "$EXP stop: ppo1.sh still running 10 minutes later, killed"
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
    [ -n "$GC_HELD" ] && say "GPUs released: $GC_HELD"; gc_release  # GPUs kept for a restart are no longer in GC_HELD
  fi
  rm -f $PIDF $GF $STOPF
}
trap release EXIT; trap 'trap "" INT TERM HUP; say "$EXP stopped by a signal"; stop_training; exit 143' INT TERM HUP
[ -n "$DRY" ] && : > $LOG
echo $$ > $PIDF; rm -f $GF $STOPF  # no record of an earlier instance's GPUs or stop survives into this one
if [ "$RESUME" = 1 ]; then RARG=trainer.resume_mode=auto
else
  RARG=trainer.resume_mode=disable
  [ -f $OUT/ckpt/latest_checkpointed_iteration.txt ] && { say "$EXP not started: $OUT/ckpt holds checkpoints (RESUME=1 resumes them; or move the directory aside)"; exit 1; }
fi
if [ $LONG = 1 ]; then
  RESP=61440; SPN=2; OFFLOAD=${OFFLOAD:-True}
  SPARGS="actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=2 actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=2"
else RESP=28672; SPN=1; OFFLOAD=${OFFLOAD:-False}; SPARGS=""; fi
[ -n "$DRY" ] || [ -f $GC ] || { say "$EXP not started: gpu-claim is not installed (no $GC)"; exit 1; }
if [ -n "$WAIT_PID" ] && [ -z "$DRY" ]; then say "$EXP waits for pid $WAIT_PID"; while alive $WAIT_PID; do nap 60; done; fi
AVAIL=$(df -P / | awk 'NR == 2 {print int($4 / 1048576)}')  # ppo2: whole GiB rounded down, as margins_ok
[ "$AVAIL" -gt ${ROOT_MIN_GB:-30} ] || { say "root disk has ${AVAIL} GB free (Ray's temp dir is there; needs more than ${ROOT_MIN_GB:-30}): $EXP not started"; exit 1; }
if [ -n "$DRY" ]; then PYX=$ROOT/scripts/03_model-training/rl/py_cfg_job.sh; TLOG=$SD/${EXP}_LONG$LONG.yaml; export CFG_ARGS=$SD/${EXP}_LONG$LONG.args
else PYX=""; TLOG=$RL/$EXP.log; fi
export NCCL_P2P_DISABLE=1 RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"'"$FP_CACHE"'/ray_spill"}}'
# GPUs and training, at most twice (see the header). gc_claim waits for gpu_claim.py in the background, so that a signal
# reaches the trap at once; the GPUs it claimed but did not yet hand over are released by release too.
if [ "$GPUL" = auto ]; then CANDS="0 1 2 3 4 5 6 7 8 9"; else CANDS=$GPUL; fi
for ATTEMPT in 1 2; do
  if [ -n "$DRY" ]; then
    [ "$GPUL" = auto ] && GPUL=$(for i in $(seq $NG); do printf 'G%s ' $i; done); GPUL=$(echo $GPUL)
    say "(dry run) would claim $NG of GPUs $CANDS without a free8.sh lock (now: $(f8_free $CANDS)) with $GC, holders of its default size${KEPT:+, first taking over the kept GPUs $KEPT}; $GPUL stand for them"
  else
    say "$EXP claims $NG of GPUs $CANDS without a free8.sh lock (attempt $ATTEMPT)${KEPT:+, first taking over the kept GPUs $KEPT}"
    gc_claim $NG "" "$CANDS" "$KEPT" || exit 1
    GPUL=$GC_HELD; echo $GPUL > $GF; say "GPUs claimed: $GPUL"
    [ $ATTEMPT = 2 ] && [ -f $TLOG ] && mv $TLOG $TLOG.attempt1
    margins_ok ${ROOT_MIN_GB:-30} ${MIN_DISK2_GB:-300} ${MIN_MEM_GB:-200} || { say "$EXP not started: $MARGINS (needs > ${ROOT_MIN_GB:-30}, ${MIN_DISK2_GB:-300}, ${MIN_MEM_GB:-200}); GPUs released"; exit 1; }  # ppo2 (4)
    say "margins: $MARGINS"
  fi
  GL=$(commas $GPUL)  # run_ppo.sh's GPUS (it counts them for n_gpus_per_node)
  if [ -f $OUT/metrics.jsonl ]; then
    B=$OUT/metrics_before_$(date -u +%m%d%H%M%S).jsonl
    if [ -n "$DRY" ]; then say "(dry run) would copy $OUT/metrics.jsonl to $B"; else cp $OUT/metrics.jsonl $B && say "copied metrics.jsonl to $B"; fi
  fi
  say "${DRY:+(dry run) }$EXP starts on GPUs $GL: LONG=$LONG (response budget $RESP, actor and reference SP $SPN, actor offload $OFFLOAD), $RARG${EXTRA:+, extra overrides: $EXTRA}; TRAIN $TRAIN ($NT tasks, $((NT / BATCH)) batches an epoch, EPOCHS $EPOCHS), VAL $VAL, $STEPS steps; actor lr $ACTOR_LR, mini-batch $ACTOR_MINI; critic lr $CRITIC_LR"
  STARTED=1; T0=$(date +%s)
  GPUS=$GL EXP=$EXP TRAIN=$TRAIN VAL=$VAL BATCH=$BATCH MINI=$ACTOR_MINI LR=$ACTOR_LR STEPS=$STEPS EPOCHS=$EPOCHS \
    WARMUP=3 CRITIC_LR=$CRITIC_LR CRITIC_MINI=24 VCLIP=10 SAVE_FREQ=4 TEST_FREQ=4 VAL_BEFORE=False VAL_BATCH=200 RESP=$RESP \
    TOKLEN=32768 OFFLOAD=$OFFLOAD OUTD=$OUT PY=$PYX \
    bash $ROOT/scripts/03_model-training/rl/run_ppo.sh ray_kwargs.ray_init.num_cpus=64 +ray_kwargs.ray_init._temp_dir=${FP_TMP:-/tmp}/ray \
    trainer.max_actor_ckpt_to_keep=2 trainer.max_critic_ckpt_to_keep=2 $RARG $SPARGS $EXTRA > $TLOG 2>&1 &
  RP=$!; echo "$$ $RP" > $PIDF; wait $RP; RC=$?
  say "$EXP exit $RC"
  [ -n "$DRY" ] && break
  holder_lines $GPUL
  [ $RC = 0 ] && [ ! -f $STOPF ] && [ -n "$KEEP_AT_END" ] && gc_keep_handover "$KEEP_HOLD_GB" "$KEEP_AT_END"  # ppo2 (3)
  { [ $RC = 0 ] || [ -f $STOPF ]; } && break
  if [ $ATTEMPT = 1 ] && [ $(( $(date +%s) - T0 )) -lt 1200 ] && contended $GPUL; then
    say "$EXP failed within 20 minutes, and another user's process is on GPUs $GL: released, claiming again"
    clear_gpus; STARTED=""; gc_release; KEPT=""; continue
  fi
  gc_keep_for_restart "" "TRAIN=$TRAIN VAL=$VAL LONG=$LONG EXP=$EXP ACTOR_LR=$ACTOR_LR ACTOR_MINI=$ACTOR_MINI CRITIC_LR=$CRITIC_LR bash $0 (RESUME=1 to resume from its checkpoints)"
  clear_gpus; STARTED=""; break
done
release
say "$EXP done"
exit $RC
