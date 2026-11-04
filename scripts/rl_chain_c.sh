#!/bin/bash
# rl_chain_c: the plan after GRPO run1b (research/log/2026-11-01-0010.md, 11-02 22:4x UTC, the author: "补丁二和三做与不做，两轮
# GRPO……PPO 就直接按照补丁做来执行吧，然后评测是放到最后做"; 23:1x and 23:2x UTC). Stages, one after another, each start and end
# written to rl/rl_chain_c.status with its exact command; a stage that fails stops the chain with the reason:
#  1. run1b's end: its resume script (grpo_run1_resume7.sh, pid RUN1B_WRAP) has exited, run1.status's last
#     "run1 resume exit" line is "run1 resume exit 0 ... (run1b)" (rl_chain_b.sh's check, for run1b), and its crash guard
#     (gpu_guard_run1b.sh, pid RUN1B_GUARD) has exited with "nothing held"; else the chain stops and does not screen.
#     RUN1B_OK=1, which the author sets after checking a run1b that did not end so (written to the status file), skips
#     those two line checks; RUN1B_WRAP_NAME and RUN1B_GUARD_NAME name a different resume script or guard. grpo1 =
#     run1b/ckpt/global_step_<latest>/actor.
#  2. the 8-run screen of M1 on rl_pool_v2 and the GRPO data (scripts/03_model-training/rl/screen_c.sh = screen_m1_k8.sh, see its header):
#     grpo/run2_train.parquet and run2_val.parquet. 2b: more than 939 tasks are cut to 939 by subsample_grpo_data.py
#     (seed 20261103, per template, proportional, largest remainder) into grpo/run2_train_939.parquet and its manifest;
#     otherwise all are used. The result is the training data of stages 3, 3b and 4.
#  3. grpo2 (grpo2.sh: run1b's settings from M1 on the stage-2 data, 28 steps). DYN, LR and MINI are read from
#     rl_chain_c.conf when grpo2 first starts, once its GPUs may be taken (defaults DYN=0, LR=8.7e-7, MINI=96 for a
#     missing file or key; a bad line stops the chain), recorded in the status file and in rl/grpo2.settings; a resumed
#     grpo2 keeps those. The file's PPO_* keys are still read and checked, but nothing uses them while ppo2 is not run.
#  3b. grpo3 (grpo2.sh with EXP=grpo3; log 11-03 06:3x UTC, the author: "GRPO 1结束后，紧跟着进行 GRPO 2、GRPO 3"): grpo2
#     with one change, the actor mini-batch GRPO3_MINI = 24 tasks, so 4 updates a step and 112 in all; DYN and LR are
#     grpo2's start settings (rl/grpo2.settings), written with MINI to rl/grpo3.settings; a resumed grpo3 keeps those.
#  4. grpo4 (grpo2.sh with EXP=grpo4; log 11-03 07:1x UTC, the author: "PPO 先不做，先把 gRPO 做彻底，也就是增加一个 gRPO
#     四，它是在三的基础上额外多一个，就是丢掉"): grpo3 with the dynamic group filter on (DYN=1, dyn_filter.py's
#     CappedDapoReplayBuffer); LR and MINI are grpo3's start settings (rl/grpo3.settings), written with DYN to
#     rl/grpo4.settings; a resumed grpo4 keeps those. ppo2 is not run (scripts/03_model-training/rl/ppo2.sh is kept for later).
#  5. evaluations: (a) eval_ckpt_c.sh (= eval_ckpt_8gpu.sh) on grpo1, grpo2, grpo3 and grpo4's last checkpoints, each followed by
#     scripts/04_evaluation/regrade_run.sh <name>, which must write each run's and each comparison's file anew; (b)
#     scripts/04_evaluation/general_c.sh (eval_general.sh's jobs) for M1, grpo1, grpo2, grpo3 and grpo4 on 3 GPUs, and the official
#     Qwen3.5-4B with ifeval_only on 1, beside (d) scripts/04_evaluation/judge_c.sh, the judge calibration on 2 (6 in all); (c) once
#     (b) is done, scripts/04_evaluation/reworded_c.sh: M1 and the official model on the reworded test set, then compare_reworded.sh,
#     and grpo1, grpo2, grpo3 and grpo4 too unless REWORDED_RL=0 in the conf file (read when 5c starts). The chain ends when (c)
#     and (d) are done.
# GPUs: at most 6 of ours at any moment, holders included, over all jobs: before each GPU stage the chain waits until the
# GPUs our jobs use (a process of ours, a gpu-claim or free8.sh lock) outside the GPUs it hands to the stage, plus the
# stage's count, are within 6; while the judge runs, the reworded test claims at most 4, whether or not the judge's claim
# has locked its 2 yet. Every stage claims its GPUs with gpu-claim through gpuclaim_lib.sh (free8.sh's locks respected)
# and, when it ends well, keeps them for the next (KEEP_AT_END: gpu_claim.py keep, holders sized for the next stage's
# first process: gpu_claim.py's default before verl, 4 GiB before vLLM at 0.85, 2 GiB before the judge's vLLM at 0.90),
# which takes them over (KEPT=<GPUs>, claim --kept), so no gap opens between stages. The 6 GPUs kept after the last eval
# are split: 2 to the judge, 1 to the official model's IFEval, 3 to the general evaluation, whose GPUs then go on to the
# reworded test; on a restart, kept GPUs go to the 5b jobs still to run (all of them in that order, or the one job whose
# count they match), and the rest are released.
# Stops: a stage that fails keeps, as gpu-claim does, the GPUs it kept for its restart and those handed to it that it never
# took over; the chain lists them with the chain's restart command (KEPT=<GPUs>), which hands them to that stage again.
# Every other stop (a margin, a check, a missing file) releases the GPUs kept for the next stage. While the judge runs,
# a stop first releases those GPUs and waits for the judge.
# Resuming: run again after a stop and finished stages are skipped (their success lines and outputs exist), the screen
# and the reworded runs resume (run.py), grpo2, grpo3 and grpo4 resume from their checkpoints (RESUME=1, with the settings they
# started with), an eval resumes its rows, the judge its verdicts. KEPT=<GPUs> hands GPUs that the last run kept to the
# first stage that runs (released there when that stage takes a different number).
# Margins (whole GiB, rounded down): before grpo2, grpo3 and grpo4, the root disk (Ray's temp dir) must have more than 30 GB
# free, disk2 300 GB and the host's MemAvailable 200 GB (the launchers check again once their GPUs are claimed); before
# each eval stage 30, 50, 64. ROOT_MIN_GB, when set, replaces the root disk's 30 here and in the stages' launchers.
# Usage: setsid nohup bash rl_chain_c.sh > rl_chain_c.out 2>&1 < /dev/null &
#        Stop: bash rl_chain_c.sh stop (or kill <pid>, which the status file's first line gives; not -9): TERM goes to
#        the running stage(s), each stops as its script says (servers by process group, trainers through their driver,
#        GPUs released), the chain waits for them, releases the GPUs they or a stage that has not started left locked,
#        and exits 143.
#        DRY=1: no waiting, claiming, process start or write under rl/; each stage's launcher runs in its own DRY mode,
#        all writing under $DRY_DIR/, and the chain goes on past failed checks to show every stage.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); . "$ROOT/scripts/paths.sh"
RL=$FP_RUNS/rl
G=$FP_DATA_D2/grpo; MK=$FP_DATA_D2/market_v1
S=$ROOT/src/03_model-training/code/verl_grpo; MD=$FP_MODELS_D2
R=$FP_RUNS/qwen3.5-4b-base-en; VPY=$FP_VENVS/verl/bin/python
CONF=${CONF:-$ROOT/configs/03_model-training/grpo/rl_chain_c.conf}; RUN1B_WRAP=${RUN1B_WRAP:-4005387}; RUN1B_GUARD=${RUN1B_GUARD:-4046774}
W1=${RUN1B_WRAP_NAME:-grpo_run1_resume7}; G1=${RUN1B_GUARD_NAME:-gpu_guard_run1b}; OK1=$RUN1B_OK
NSUB=939; SUBSEED=20261103; GRPO_STEPS=28; CAPG=6; GRPO3_MINI=24
INIT_KEPT=$KEPT
unset KEPT KEEP_AT_END KEEP_HOLD_GB GCHOLD TRAIN VAL DYN LR MINI RESUME EXP LONG GPUL EXTRA RL_MODELS NG WAIT_PID DRY_NTASKS \
  ACTOR_LR ACTOR_MINI CRITIC_LR RUN1B_OK
if [ -n "$DRY" ]; then SD=$DRY_DIR; else SD=$RL; fi
LOG=$SD/rl_chain_c.status; PIDF=$SD/rl_chain_c.pid; GC_LOG=$SD/rl_chain_c.gpuclaim.log
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; [ -n "$DRY" ] && echo "# $1"; return 0; }
nap() { sleep $1 & wait $!; }  # a signal reaches the trap at once
runs() { [ -n "$1" ] && case "$(ps -o args= -p $1 2>/dev/null)" in *"$2"*) true;; *) false;; esac; }  # <pid> <script name>
source $ROOT/scripts/gpuclaim_c_lib.sh  # our_gpus, gc_left_by, margins_ok, commas, spaces, GC, LOCKS (gpuclaim_lib.sh, free8.sh)
rows() { CUDA_VISIBLE_DEVICES= $VPY -B -c 'import sys, pyarrow.parquet as pq; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' $1 2>/dev/null; }
lines() { cat $1 2>/dev/null | wc -l; }
since() { tail -n +$(($2 + 1)) $1 2>/dev/null | grep -q "^$3 "; }  # <file> <lines before> <line start>: written since
same_num() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a + 0 == b + 0)}'; }

num_in() {  # <value> <max>: a plain or e-notation number with 0 < value <= max
  printf '%s' "$1" | grep -Eq '^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$' && awk -v x="$1" -v m="$2" 'BEGIN {exit !(x + 0 > 0 && x + 0 <= m + 0)}'
}
mini_ok() { case $1 in *[!0-9]*|0*|'') false;; *) [ $((96 % $1)) = 0 ] && [ $(($1 * 8 % 6)) = 0 ];; esac; }  # divides 96, x 8 by 6 GPUs
read_conf() {  # <file>: C_DYN C_LR C_MINI C_RWRL C_PLR C_PMINI C_PCLR, and C_FROM (the keys the file set, or "no file");
  # false with C_ERR on a bad line: KEY=VALUE lines, '#' comments, blanks and spaces ignored; keys DYN (0|1), LR (a number,
  # 0 < LR <= 1e-5), MINI (a whole number dividing 96, with MINI x 8 divisible by 6 GPUs), REWORDED_RL (0|1), PPO_LR (as
  # LR), PPO_MINI (as MINI), PPO_CRITIC_LR (0 < x <= 1e-4); each key at most once
  local line k v n=0
  C_DYN=0; C_LR=8.7e-7; C_MINI=96; C_RWRL=1; C_PLR=8.7e-7; C_PMINI=96; C_PCLR=8.7e-6; C_FROM=""; C_ERR=""
  [ -f "$1" ] || { C_FROM="no file"; return 0; }
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1)); line=${line%%#*}; line=$(printf '%s' "$line" | tr -d ' \t\r')
    [ -z "$line" ] && continue
    case "$line" in ?*=?*) k=${line%%=*}; v=${line#*=};; *) C_ERR="line $n: not KEY=VALUE: '$line'"; return 1;; esac
    case " $C_FROM " in *" $k "*) C_ERR="line $n: $k given twice"; return 1;; esac
    case $k in
      DYN) case $v in 0|1) C_DYN=$v;; *) C_ERR="line $n: DYN must be 0 or 1, not '$v'"; return 1;; esac;;
      REWORDED_RL) case $v in 0|1) C_RWRL=$v;; *) C_ERR="line $n: REWORDED_RL must be 0 or 1, not '$v'"; return 1;; esac;;
      LR|PPO_LR) num_in "$v" 1e-5 || { C_ERR="line $n: $k must be a number with 0 < $k <= 1e-5, not '$v'"; return 1; }
            [ $k = LR ] && C_LR=$v || C_PLR=$v;;
      PPO_CRITIC_LR) num_in "$v" 1e-4 && C_PCLR=$v || { C_ERR="line $n: PPO_CRITIC_LR must be a number with 0 < PPO_CRITIC_LR <= 1e-4, not '$v'"; return 1; };;
      MINI|PPO_MINI) case $v in *[!0-9]*|0*) C_ERR="line $n: $k must be a whole number, not '$v'"; return 1;; esac
            mini_ok $v || { C_ERR="line $n: $k must divide 96 with $k x 8 divisible by 6, not $v"; return 1; }
            [ $k = MINI ] && C_MINI=$v || C_PMINI=$v;;
      *) C_ERR="line $n: unknown key '$k' (DYN, LR, MINI, REWORDED_RL, PPO_LR, PPO_MINI, PPO_CRITIC_LR)"; return 1;;
    esac
    C_FROM="$C_FROM $k"
  done < "$1"
  C_FROM=${C_FROM# }; C_FROM=${C_FROM:-no keys}
}
ppo_choice_ok() {  # <grpo2's LR> <grpo2's MINI>: false (C_ERR) when grpo2's LR or MINI is not the default and the conf
  # file just read does not give both PPO_LR and PPO_MINI
  same_num $1 8.7e-7 && [ "$2" = 96 ] && return 0
  case " $C_FROM " in *" PPO_LR "*) case " $C_FROM " in *" PPO_MINI "*) return 0;; esac;; esac
  C_ERR="grpo2's LR=$1 MINI=$2 differ from ppo1.sh's 8.7e-7 and 96, so $CONF must give PPO_LR and PPO_MINI (grpo2's values keep grpo2 and ppo2 apart only in the algorithm; 8.7e-7 and 96 are ppo1.sh's)"
  return 1
}
latest() { cat $1/ckpt/latest_checkpointed_iteration.txt 2>/dev/null; }  # <run dir>: its last checkpoint's step
run_dir() { case $1 in grpo1) echo $RL/run1b;; *) echo $RL/$1;; esac; }  # <grpo1|grpo2|grpo3|grpo4>
st_done() {  # <grpo2|grpo3|grpo4|eval_<name>>: true when an earlier run of the chain finished that stage (never in a dry run)
  local n d k
  [ -n "$DRY" ] && return 1
  case $1 in
    grpo2) [ "$(latest $RL/grpo2)" = $GRPO_STEPS ] && grep -q '^grpo2 exit 0 ' $RL/grpo2.status 2>/dev/null;;
    grpo3) [ "$(latest $RL/grpo3)" = $GRPO_STEPS ] && grep -q '^grpo3 exit 0 ' $RL/grpo3.status 2>/dev/null;;
    grpo4) [ "$(latest $RL/grpo4)" = $GRPO_STEPS ] && grep -q '^grpo4 exit 0 ' $RL/grpo4.status 2>/dev/null;;
    eval_*) n=${1#eval_}; d=$(run_dir $n); k=$(latest $d)
            [ -n "$k" ] && grep -q "^eval $n done " $RL/eval_$n.status 2>/dev/null && grep -Fqx "merged from $d/ckpt/global_step_$k/actor" $RL/eval_$n.status;;
    *) return 1;;
  esac
}
hold_after() {  # <stage>: the hold in GiB (empty: gpu_claim.py's default) for the GPUs that stage keeps, sized for the
  # first later stage that will run: verl (grpo2, grpo3, grpo4) the default, an eval's vLLM at 0.85 4, the judge's at 0.90 2
  local seen="" s
  for s in screen grpo2 grpo3 grpo4 eval_grpo1 eval_grpo2 eval_grpo3 eval_grpo4 final; do
    if [ -z "$seen" ]; then [ $s = $1 ] && seen=1; continue; fi
    [ $s = final ] && { echo 2; return; }
    st_done $s && continue
    case $s in grpo2|grpo3|grpo4) echo ""; return;; eval_*) echo 4; return;; esac
  done
}
regrade_ok() {  # <name> <marker>: regrade_run.sh wrote each run's and each comparison's file after the marker, and no
  # traceback; RG_WHY := what is missing
  local d=$STRICT_DIR/$1 f
  for f in test_hs_market_$1_H1 test_hs_fsc_$1_H1 dev_hs_market_$1_H1_x4 compare_test_market compare_test_finsearchcomp compare_dev_market; do
    [ $d/$f.json -nt $2 ] || { RG_WHY="no $d/$f.json written by this regrade"; return 1; }
  done
  grep -q Traceback $d/regrade.txt 2>/dev/null && { RG_WHY="a traceback in $d/regrade.txt"; return 1; }
  return 0
}
gen_done() {  # <all|ifeval_only> <name>=<dir> ...: every model has its "general <name> done" line (never in a dry run)
  local w=$1 a; shift
  [ -n "$DRY" ] && return 1
  for a in "$@"; do grep -q "^general ${a%%=*} done " $GD5/general_c_$w.status 2>/dev/null || return 1; done
}
CHILDREN=""; HAND=""; STAGE_PID=""  # the stages running; the GPUs kept for the next stage; the last stage's launcher
KJ=""; KO=""; KA=""; PJ=""; JRUN=""; nj=0  # 5b/5d: the GPUs handed to the judge, the official IFEval, the general eval; the judge's pid
drop_child() { CHILDREN=$(echo $CHILDREN | tr ' ' '\n' | grep -vx "$1" | tr '\n' ' '); }
release_hand() {  # releases HAND's GPUs whose lock's owner is gone (kept by a stage that exited, or handed to one and not
  # taken over); HAND := ""
  local mine
  if [ -z "$DRY" ] && [ -n "$HAND" ]; then
    mine=$(gc_left_by "" "$HAND")
    [ -n "$mine" ] && { gc release $(commas $mine) 2>> $GC_LOG; say "GPUs $mine, kept for a stage that will not start on them, released"; }
  fi
  HAND=""
}
fit_hand() {  # <N>: GPUs handed in that are not the N the next stage takes are released
  local n=$(echo $HAND | wc -w)
  [ $n = 0 ] || [ $n = $1 ] && return 0
  say "GPUs $HAND handed on: not the $1 this stage takes; released"; release_hand
}
after_stage() {  # <rc> <handover file> <GPUs handed to the stage>: HAND := on success, the GPUs the stage kept for the next
  # (its handover file); after a failure, those it kept for its restart or was handed and never took over. On success,
  # GPUs handed to it that it neither took over nor kept (none, normally) are released.
  local l x
  HAND=""
  if [ -n "$DRY" ]; then [ $1 = 0 ] && [ -s "$2" ] && HAND=$(cat $2); return 0; fi
  if [ $1 = 0 ]; then
    [ -s "$2" ] && HAND=$(cat $2)
    if [ -n "$3" ]; then
      l=$(for x in $(gc_left_by "" "$3"); do case " $HAND " in *" $x "*) ;; *) printf '%s ' $x;; esac; done)
      [ -n "$l" ] && { gc release $(commas $l) 2>> $GC_LOG; say "GPUs $l, handed to the stage and left unused, released"; }
    fi
  else
    HAND=$(gc_left_by "$STAGE_PID" "$3")
    [ -n "$HAND" ] && say "GPUs $HAND stay locked for the failed stage (kept for its restart, or handed to it and not taken over)"
  fi
  return 0
}
on_signal() {
  trap '' INT TERM HUP
  local p l kids=$CHILDREN
  say "rl_chain_c got a signal${kids:+; TERM to the running stage(s), pids $kids, waiting for them}"
  for p in $kids; do kill -TERM $p 2>/dev/null; done; for p in $kids; do wait $p; done
  if [ -z "$DRY" ]; then
    l=$(gc_left_by "$kids" "$HAND $KJ $KO $KA")
    [ -n "$l" ] && { gc release $(commas $l) 2>> $GC_LOG; say "GPUs $l, left locked by the stopped stages or kept for a stage that had not started, released"; }
  fi
  say "rl_chain_c stopped by a signal"; rm -f $PIDF; exit 143
}
stop() {  # <reason> [keep]: the chain stops (a dry run goes on, to show the later stages). With keep (a stage failed), the
  # GPUs in HAND, which that stage kept for its restart or was handed and never took over, stay kept for the chain's
  # restart, which hands them to the same stage; otherwise they are released. A judge still running is waited for first,
  # with HAND released, since the chain cannot restart before the judge ends; a judge that fails then keeps its GPUs.
  local how=${2:-release} p rj
  if [ -n "$PJ" ]; then
    [ -n "$HAND" ] && { say "GPUs $HAND released: the chain stops once the judge ends"; release_hand; }
    say "waiting for the judge (pid $PJ) to end before stopping (reason: $1)"
    wait $PJ; rj=$?; drop_child $PJ; p=$PJ; PJ=""
    if since $JST $nj "judge_c done"; then say "5d. judge ends: exit $rj"
    else HAND=$(gc_left_by "$p" "$KJ"); how=keep; say "5d. judge calibration failed too: exit $rj (see $JST)"; fi
  fi
  say "rl_chain_c stopped: $1"
  if [ -n "$HAND" ]; then
    if [ "$how" = keep ]; then
      say "GPUs $HAND stay kept, with holders, until gpu_claim.py's keep time passes; restart the chain (not the stage's own restart line), which hands them to that stage: KEPT=\"$HAND\" setsid nohup bash $0 > $CHAIN_DIR/logs/rl_chain_c.out 2>&1 < /dev/null &   release instead: python3 $GC release $(commas $HAND)"
    else release_hand; fi
  fi
  HAND=""
  [ -n "$DRY" ] && return 0
  rm -f $PIDF; exit 1
}
stage() {  # <label> <status file> <success line> <command ...>: the command in the background, waited for by its PID
  # (STAGE_PID); true when the success line was written to the status file after the start
  local label=$1 st=$2 ok=$3 n0 p rc; shift 3
  n0=$(lines $st); [ -n "$DRY" ] && n0=0  # the dry runs start their status files afresh
  say "$label starts: $*"
  if [ -n "$DRY" ]; then "$@" & p=$!; else "$@" < /dev/null & p=$!; fi
  STAGE_PID=$p; CHILDREN="$CHILDREN $p"; wait $p; rc=$?; drop_child $p
  if since $st $n0 "$ok"; then say "$label ends: exit $rc, \"$ok\" written"; return 0; fi
  say "$label failed: exit $rc, no \"$ok\" in $st since it started"; return 1
}
cap_wait() {  # <N> <GPUs handed to the stage>: until our other GPUs plus N are within the cap
  local u t=0
  if [ -n "$DRY" ]; then say "(dry run) cap: our GPUs outside the hand-over now: [$(our_gpus $2 | tr '\n' ' ')]; the stage takes $1 (cap $CAPG)"; return 0; fi
  while u=$(our_gpus $2 | tr '\n' ' '); [ $(( $(echo $u | wc -w) + $1 )) -gt $CAPG ]; do
    [ $((t % 1800)) = 0 ] && say "waiting: our jobs use GPUs [$u] besides the $1 this stage takes (cap $CAPG)"
    nap 30; t=$((t + 30))
  done
  [ $t -gt 0 ] && say "cap: the stage may take $1 GPUs now (others of ours: [$u])"; return 0
}
margins() {  # <root GB> <disk2 GB> <memory GB> <label>; ROOT_MIN_GB, when set, replaces the root threshold (log 11-03 01:0x UTC)
  local r=${ROOT_MIN_GB:-$1}
  if margins_ok $r $2 $3; then say "$4 margins: $MARGINS"; else stop "$4: $MARGINS; needs more than $r, $2 and $3 GB"; fi
}
alloc_5b() {  # NEED (the 5b/5d jobs still to run, of "judge:2 ifeval_only:1 all:3") and HAND: KJ, KO, KA := the GPUs
  # the judge, the official IFEval and the general evaluation take over: every job's count from HAND in that order when
  # it holds enough (the rest released), else HAND whole to the one job whose count it matches, else HAND released
  local j g i m="" sum=0; set -- $HAND
  KJ=""; KO=""; KA=""
  for j in $NEED; do sum=$((sum + ${j#*:})); done
  if [ $# = 0 ]; then HAND=""; return 0; fi
  if [ $# -ge $sum ]; then
    for j in $NEED; do g=""; for i in $(seq ${j#*:}); do g="$g $1"; shift; done; g=${g# }
      case ${j%%:*} in judge) KJ=$g;; ifeval_only) KO=$g;; all) KA=$g;; esac; done
    say "5b. the kept GPUs $HAND:${KJ:+ $KJ to the judge;}${KO:+ $KO to the IFEval of the official model;}${KA:+ $KA to the general evaluation;}${*:+ $* not needed, released}"
    HAND="$*"; release_hand
  else
    for j in $NEED; do [ ${j#*:} = $# ] && m=${j%%:*}; done
    case $m in judge) KJ=$HAND;; ifeval_only) KO=$HAND;; all) KA=$HAND;; esac
    if [ -n "$m" ]; then say "5b. the $# kept GPUs $HAND go to $m, which takes $#; the other jobs claim their own"
    else say "5b. the $# kept GPUs $HAND fit no 5b job still to run: released; each job claims its own"; release_hand; fi
  fi
  HAND=""
}
fit_5c() {  # RG := the GPUs the reworded test may hold: CAPG, or CAPG - 2 while the judge runs, whether or not the judge's
  # claim has locked its 2 yet; HAND beyond RG is released; NGR := the count to claim when HAND is empty, else ""
  local n
  RG=$CAPG; { [ -n "$PJ" ] && kill -0 $PJ 2>/dev/null; } || { [ -n "$DRY" ] && [ -n "$JRUN" ]; } && RG=$((CAPG - 2))
  n=$(echo $HAND | wc -w)
  [ $n -gt $RG ] && { say "5c. the $n GPUs handed on are more than the $RG the reworded test may take beside the judge: released"; release_hand; n=0; }
  if [ $n = 0 ]; then NGR=$RG; else NGR=""; fi
}
[ -n "$CHAIN_LIB_ONLY" ] && return 0  # sourced by a test: the functions only

if [ "$1" = stop ]; then
  P=$(cat $SD/rl_chain_c.pid 2>/dev/null)
  runs "$P" rl_chain_c.sh || { echo "rl_chain_c.sh is not running (pid file $SD/rl_chain_c.pid: ${P:-none})" >&2; exit 1; }
  kill -TERM $P; echo "TERM sent to rl_chain_c.sh (pid $P); its running stage(s) stop as their scripts say (status: $LOG)"
  for i in $(seq 240); do kill -0 $P 2>/dev/null || { echo "rl_chain_c.sh stopped"; exit 0; }; sleep 5; done
  echo "rl_chain_c.sh still runs 20 minutes later: see $LOG" >&2; exit 1
fi

trap on_signal INT TERM HUP
[ -n "$DRY" ] && { mkdir -p $SD; : > $LOG; }
if [ -z "$DRY" ]; then
  runs "$(cat $PIDF 2>/dev/null)" rl_chain_c.sh && { echo "rl_chain_c.sh already runs (pid $(cat $PIDF))" >&2; exit 2; }
  echo $$ > $PIDF
fi
HAND=$(spaces $INIT_KEPT)
say "rl_chain_c starts (pid $$${DRY:+, dry run}; stop: bash $0 stop)${HAND:+; GPUs $HAND handed to the first stage that runs}"

# 1. run1b's end (checked every 2 s, so that the screen claims run1b's GPUs soon after their locks go)
if runs $RUN1B_WRAP $W1; then
  if [ -n "$DRY" ]; then say "1. (dry run) run1b's resume script (pid $RUN1B_WRAP, $W1) runs; the chain would wait for it to exit (checked every 2 s)"
  else say "1. waits for run1b's resume script (pid $RUN1B_WRAP, $W1) to exit"; while runs $RUN1B_WRAP $W1; do nap 2; done; fi
fi
if runs $RUN1B_GUARD $G1; then
  if [ -n "$DRY" ]; then say "1. (dry run) run1b's crash guard (pid $RUN1B_GUARD, $G1) runs; the chain would wait up to 10 minutes for it to exit"
  else
    say "1. waits up to 10 minutes for run1b's crash guard (pid $RUN1B_GUARD, $G1) to exit"; t=0
    while runs $RUN1B_GUARD $G1; do nap 2; t=$((t + 2)); [ $t -ge 600 ] && stop "1. run1b's crash guard (pid $RUN1B_GUARD) still runs 10 minutes after the resume script's end"; done
  fi
fi
X=$(grep '^run1 resume exit ' $RL/run1.status | tail -n 1); Y=$(tail -n 1 $RL/run1b_guard.status 2>/dev/null)
if [ "$OK1" = 1 ]; then
  say "1. RUN1B_OK=1 (the author checked run1b): run1.status's last exit line (${X:-none}) and the guard's last line (${Y:-none}) are not checked"
else
  case "$X" in "run1 resume exit 0 "*"(run1b)"*) say "1. run1b ended: $X";;
    *) stop "1. run1.status's last exit line is not \"run1 resume exit 0 ... (run1b)\": ${X:-none} (run1b failed or has not ended; no screen; RUN1B_OK=1 once the author has checked it)";; esac
  case "$Y" in "run1b ended normally"*"nothing held"*) say "1. run1b's crash guard: $Y";;
    *) stop "1. run1b's crash guard did not end with \"nothing held\": ${Y:-none}";; esac
fi
N1=$(latest $RL/run1b); CK1=$RL/run1b/ckpt/global_step_$N1/actor
[ -n "$N1" ] && [ -f $CK1/fsdp_config.json ] || stop "1. no run1b checkpoint at $CK1"
say "1. grpo1 = run1b step ${N1:-?}: $CK1$([ "$N1" = 24 ] || echo " (run1b was set for 24 steps)")"
L=$(for x in 0 1 2 3 4 5 6 7 8 9; do [ -d $LOCKS/gpu$x ] && printf '%s ' $x; done)
[ -n "$L" ] && say "1. free8.sh locks still on GPUs ${L% } (gc_claim leaves those GPUs out)"

# 2. the 8-run screen and the GRPO data
SCR_ST=$MK/runs/screen_m1_rlpool2_k8.status; T2=$G/run2_train.parquet; V2=$G/run2_val.parquet
if [ -f $T2 ] && [ -f $V2 ] && grep -q '^screen_m1_rlpool2_k8 done ' $SCR_ST 2>/dev/null; then
  say "2. screen already done: $T2, $V2"
else
  fit_hand 6; cap_wait 6 "$HAND"; margins 30 50 64 "2. screen"
  KF=$SD/handover.screen; rm -f $KF; KH=$(hold_after screen); OLD=$HAND
  if [ -n "$DRY" ]; then
    stage "2. screen" $SD/screen_m1_rlpool2_k8.status "screen_m1_rlpool2_k8 done" env DRY=1 DRYDIR=$SD KEPT="$HAND" KEEP_AT_END=$KF KEEP_HOLD_GB="$KH" GCHOLD=4 bash $ROOT/scripts/03_model-training/rl/screen_c.sh
  else stage "2. screen" $SCR_ST "screen_m1_rlpool2_k8 done" env KEPT="$HAND" KEEP_AT_END=$KF KEEP_HOLD_GB="$KH" GCHOLD=4 bash $ROOT/scripts/03_model-training/rl/screen_c.sh; fi
  rc=$?; after_stage $rc $KF "$OLD"
  [ $rc = 0 ] || stop "2. the screen failed (see $SCR_ST)" keep
fi
if [ -f $G/run2_train.manifest.json ]; then
  say "2. selection: $(python3 -c 'import json, sys; m = json.load(open(sys.argv[1])); print("{} of {} tasks kept, {} screened, classes {}".format(m["kept"], m["tasks"], m["screened"], m["classes"]))' $G/run2_train.manifest.json 2>&1)"
fi
# 2b. at most 939 tasks
SUBOUT=$G; [ -n "$DRY" ] && SUBOUT=$SD  # a dry run writes its subsample, if any, under $DRY_DIR/
SUBCMD="$VPY -B $ROOT/src/03_model-training/utils/subsample_grpo_data.py --in $T2 --n $NSUB --seed $SUBSEED --out $SUBOUT/run2_train_$NSUB.parquet --manifest $SUBOUT/run2_train_$NSUB.manifest.json"
if [ -f $T2 ]; then
  NT=$(rows $T2); say "2b. $T2 has ${NT:-?} tasks; subsample: $SUBCMD"
  X=$(CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 19 $SUBCMD 2>&1); rc=$?
  echo "$X" | while IFS= read -r l; do say "2b. $l"; done
  TRAIN2=$(echo "$X" | sed -n 's/^TRAIN=//p' | tail -n 1)
  [ $rc = 0 ] && [ -f "$TRAIN2" ] || stop "2b. subsample_grpo_data.py exit $rc"
else
  TRAIN2=$G/run2_train_$NSUB.parquet
  say "2b. (dry run) $T2 does not exist yet; the chain would run: $SUBCMD, and train on its TRAIN= line's file (shown here as $TRAIN2; $T2 itself when it holds at most $NSUB tasks)"
  [ -n "$DRY" ] || stop "2b. no $T2"
fi
NT2=$(rows $TRAIN2); say "2b. grpo2, grpo3 and grpo4 train on $TRAIN2 (${NT2:-? (dry run: 939 assumed)} tasks), validate on $V2"

# 3. grpo2; its settings are read once its GPUs may be taken, right before it starts
GS=$SD/grpo2.settings; GST=$SD/grpo2.status; N3=$(latest $RL/grpo2)
if st_done grpo2; then say "3. grpo2 already done (step $N3)"
else
  fit_hand 6; cap_wait 6 "$HAND"; margins 30 300 200 "3. grpo2"
  if [ -n "$N3" ] && [ -z "$DRY" ]; then
    RES=1; [ -f $GS ] || stop "3. rl/grpo2/ckpt holds step $N3 but $GS, the settings it started with, is missing"
    read DYN2 LR2 MINI2 < $GS; read_conf $CONF && X="DYN=$C_DYN LR=$C_LR MINI=$C_MINI" || X="unreadable: $C_ERR"
    say "3. grpo2 resumes from step $N3 with its start settings DYN=$DYN2 LR=$LR2 MINI=$MINI2 ($GS; $CONF now: $X)"
  else
    RES=""; read_conf $CONF || stop "3. $CONF: $C_ERR"
    DYN2=$C_DYN; LR2=$C_LR; MINI2=$C_MINI
    echo "$DYN2 $LR2 $MINI2" > $GS
    say "3. grpo2 settings read from $CONF at its start: DYN=$DYN2 LR=$LR2 MINI=$MINI2 (keys from the file: $C_FROM; the others are the defaults DYN=0 LR=8.7e-7 MINI=96); written to $GS"
  fi
  KF=$SD/handover.grpo2; rm -f $KF; KH=$(hold_after grpo2); OLD=$HAND
  if [ -n "$DRY" ]; then
    stage "3. grpo2" $GST "grpo2 exit 0" env DRY=1 DRYDIR=$SD DRY_NTASKS=939 DYN=$DYN2 LR=$LR2 MINI=$MINI2 TRAIN=$TRAIN2 VAL=$V2 KEPT="$HAND" KEEP_AT_END=$KF KEEP_HOLD_GB="$KH" bash $ROOT/scripts/03_model-training/rl/grpo2.sh
  else stage "3. grpo2" $GST "grpo2 exit 0" env DYN=$DYN2 LR=$LR2 MINI=$MINI2 TRAIN=$TRAIN2 VAL=$V2 RESUME=$RES KEPT="$HAND" KEEP_AT_END=$KF KEEP_HOLD_GB="$KH" bash $ROOT/scripts/03_model-training/rl/grpo2.sh; fi
  rc=$?; after_stage $rc $KF "$OLD"
  [ $rc = 0 ] || stop "3. grpo2 failed (see $GST, $RL/grpo2.log)" keep
  N3=$(latest $RL/grpo2); [ -n "$DRY" ] || [ "$N3" = $GRPO_STEPS ] || stop "3. grpo2's last checkpoint is step ${N3:-none}, not $GRPO_STEPS"
fi

# 3b. grpo3: grpo2's start settings but GRPO3_MINI tasks a mini-batch, settled right before it starts
G3S=$SD/grpo3.settings; G3ST=$SD/grpo3.status; N3B=$(latest $RL/grpo3)
if st_done grpo3; then say "3b. grpo3 already done (step $N3B)"
else
  fit_hand 6; cap_wait 6 "$HAND"; margins 30 300 200 "3b. grpo3"
  if [ -n "$N3B" ] && [ -z "$DRY" ]; then
    RES=1; [ -f $G3S ] || stop "3b. rl/grpo3/ckpt holds step $N3B but $G3S, the settings it started with, is missing"
    read DYN3 LR3 MINI3 < $G3S; say "3b. grpo3 resumes from step $N3B with its start settings DYN=$DYN3 LR=$LR3 MINI=$MINI3 ($G3S)"
  else
    RES=""; [ -f $GS ] || stop "3b. no $GS: grpo2's start settings, which grpo3 takes"
    read DYN3 LR3 MG2 < $GS; MINI3=$GRPO3_MINI
    [ "$MG2" = $MINI3 ] && stop "3b. grpo2 ran with mini-batch $MG2 ($GS) already: grpo3 would repeat it"
    echo "$DYN3 $LR3 $MINI3" > $G3S
    say "3b. grpo3 settings: DYN=$DYN3 LR=$LR3 from grpo2's start ($GS), MINI=$MINI3 (grpo2's $MG2), so $((96 / MINI3)) updates a step; written to $G3S"
  fi
  KF=$SD/handover.grpo3; rm -f $KF; KH=$(hold_after grpo3); OLD=$HAND
  if [ -n "$DRY" ]; then
    stage "3b. grpo3" $G3ST "grpo3 exit 0" env DRY=1 DRYDIR=$SD DRY_NTASKS=939 EXP=grpo3 DYN=$DYN3 LR=$LR3 MINI=$MINI3 TRAIN=$TRAIN2 VAL=$V2 KEPT="$HAND" KEEP_AT_END=$KF KEEP_HOLD_GB="$KH" bash $ROOT/scripts/03_model-training/rl/grpo2.sh
  else stage "3b. grpo3" $G3ST "grpo3 exit 0" env EXP=grpo3 DYN=$DYN3 LR=$LR3 MINI=$MINI3 TRAIN=$TRAIN2 VAL=$V2 RESUME=$RES KEPT="$HAND" KEEP_AT_END=$KF KEEP_HOLD_GB="$KH" bash $ROOT/scripts/03_model-training/rl/grpo2.sh; fi
  rc=$?; after_stage $rc $KF "$OLD"
  [ $rc = 0 ] || stop "3b. grpo3 failed (see $G3ST, $RL/grpo3.log)" keep
  N3B=$(latest $RL/grpo3); [ -n "$DRY" ] || [ "$N3B" = $GRPO_STEPS ] || stop "3b. grpo3's last checkpoint is step ${N3B:-none}, not $GRPO_STEPS"
fi

# 4. grpo4: grpo3's start settings with the dynamic group filter on (DYN=1), settled right before it starts
G4S=$SD/grpo4.settings; G4ST=$SD/grpo4.status; N4=$(latest $RL/grpo4)
if st_done grpo4; then say "4. grpo4 already done (step $N4)"
else
  fit_hand 6; cap_wait 6 "$HAND"; margins 30 300 200 "4. grpo4"
  if [ -n "$N4" ] && [ -z "$DRY" ]; then
    RES=1; [ -f $G4S ] || stop "4. rl/grpo4/ckpt holds step $N4 but $G4S, the settings it started with, is missing"
    read DYN4 LR4 MINI4 < $G4S; say "4. grpo4 resumes from step $N4 with its start settings DYN=$DYN4 LR=$LR4 MINI=$MINI4 ($G4S)"
  else
    RES=""; [ -f $G3S ] || stop "4. no $G3S: grpo3's start settings, which grpo4 takes"
    read D3 LR4 MINI4 < $G3S; DYN4=1
    [ "$D3" = 1 ] && stop "4. grpo3 ran with the group filter on ($G3S) already: grpo4 would repeat it"
    echo "$DYN4 $LR4 $MINI4" > $G4S
    say "4. grpo4 settings: LR=$LR4 MINI=$MINI4 from grpo3's start ($G3S), DYN=1 (grpo3's $D3); written to $G4S"
  fi
  KF=$SD/handover.grpo4; rm -f $KF; KH=$(hold_after grpo4); OLD=$HAND
  if [ -n "$DRY" ]; then
    stage "4. grpo4" $G4ST "grpo4 exit 0" env DRY=1 DRYDIR=$SD DRY_NTASKS=939 EXP=grpo4 DYN=$DYN4 LR=$LR4 MINI=$MINI4 TRAIN=$TRAIN2 VAL=$V2 KEPT="$HAND" KEEP_AT_END=$KF KEEP_HOLD_GB="$KH" bash $ROOT/scripts/03_model-training/rl/grpo2.sh
  else stage "4. grpo4" $G4ST "grpo4 exit 0" env EXP=grpo4 DYN=$DYN4 LR=$LR4 MINI=$MINI4 TRAIN=$TRAIN2 VAL=$V2 RESUME=$RES KEPT="$HAND" KEEP_AT_END=$KF KEEP_HOLD_GB="$KH" bash $ROOT/scripts/03_model-training/rl/grpo2.sh; fi
  rc=$?; after_stage $rc $KF "$OLD"
  [ $rc = 0 ] || stop "4. grpo4 failed (see $G4ST, $RL/grpo4.log)" keep
  N4=$(latest $RL/grpo4); [ -n "$DRY" ] || [ "$N4" = $GRPO_STEPS ] || stop "4. grpo4's last checkpoint is step ${N4:-none}, not $GRPO_STEPS"
fi

# 5a. the four checkpoints on the original evaluation, then the strict regrade
for spec in "grpo1 $RL/run1b" "grpo2 $RL/grpo2" "grpo3 $RL/grpo3" "grpo4 $RL/grpo4"; do
  set -- $spec; name=$1; N=$(latest $2); CK=$2/ckpt/global_step_$N/actor; EST=$SD/eval_$name.status
  if [ -z "$N" ] || [ ! -f $CK/fsdp_config.json ]; then
    if [ -n "$DRY" ]; then say "5a. (dry run) eval $name would run: env KEPT=<the GPUs the stage before kept> KEEP_AT_END=$SD/handover.eval_$name KEEP_HOLD_GB=$(hold_after eval_$name) GCHOLD=4 bash $ROOT/scripts/04_evaluation/eval_ckpt_c.sh $2/ckpt/global_step_<latest>/actor $name; then bash $ROOT/scripts/04_evaluation/regrade_run.sh $name"; continue; fi
    stop "5a. no $name checkpoint at $CK"
  fi
  if st_done eval_$name; then say "5a. eval $name already done ($CK)"
  else
    fit_hand 6; cap_wait 6 "$HAND"; margins 30 50 64 "5a. eval $name"
    KF=$SD/handover.eval_$name; rm -f $KF; KH=$(hold_after eval_$name); OLD=$HAND
    if [ -n "$DRY" ]; then stage "5a. eval $name (step $N)" $EST "eval $name done" env DRY=1 DRYDIR=$SD KEPT="$HAND" KEEP_AT_END=$KF KEEP_HOLD_GB=$KH GCHOLD=4 bash $ROOT/scripts/04_evaluation/eval_ckpt_c.sh $CK $name
    else stage "5a. eval $name (step $N)" $EST "eval $name done" env KEPT="$HAND" KEEP_AT_END=$KF KEEP_HOLD_GB=$KH GCHOLD=4 bash $ROOT/scripts/04_evaluation/eval_ckpt_c.sh $CK $name; fi
    rc=$?; after_stage $rc $KF "$OLD"
    [ $rc = 0 ] || stop "5a. eval $name failed (see $EST)" keep
  fi
  MRK=$STRICT_DIR/.regrade_$name.start
  if [ -n "$DRY" ]; then say "5a. (dry run) would run: bash $ROOT/scripts/04_evaluation/regrade_run.sh $name (CPU), and require $STRICT_DIR/$name/{test_hs_market_${name}_H1,test_hs_fsc_${name}_H1,dev_hs_market_${name}_H1_x4,compare_test_market,compare_test_finsearchcomp,compare_dev_market}.json written after its start (marker $MRK) and no traceback in its regrade.txt"
  else
    touch -d "@$(( $(date +%s) - 2 ))" $MRK
    say "5a. regrade $name starts: bash $ROOT/scripts/04_evaluation/regrade_run.sh $name"
    bash $ROOT/scripts/04_evaluation/regrade_run.sh $name > /dev/null 2>&1 < /dev/null & p=$!; CHILDREN="$CHILDREN $p"; wait $p; rr=$?; drop_child $p
    if regrade_ok $name $MRK; then say "5a. regrade $name ends: $STRICT_DIR/$name/regrade.txt"
    else stop "5a. regrade $name (exit $rr) is incomplete: $RG_WHY"; fi
  fi
done

# 5b + 5d. general capability (3 GPUs and 1) beside the judge calibration (2): the jobs still to run
JST=$CHAIN_DIR/judge_cal/judge_c.status; GD5=$RL/general_eval; [ -n "$DRY" ] && { JST=$SD/judge_c.status; GD5=$SD/general; }
GEN_ALL="M1=$R/M1 grpo1=$RL/hf_grpo1 grpo2=$RL/hf_grpo2 grpo3=$RL/hf_grpo3 grpo4=$RL/hf_grpo4"; GEN_OFF="qwen3.5-4b=$MD/Qwen3.5-4B"
NEED=""
if [ -z "$DRY" ] && grep -q '^judge_c done ' $JST 2>/dev/null; then say "5d. judge calibration already done ($JST)"; else NEED="$NEED judge:2"; fi
if gen_done ifeval_only $GEN_OFF; then say "5b. general (official, ifeval_only) already done"; else NEED="$NEED ifeval_only:1"; fi
if gen_done all $GEN_ALL; then say "5b. general (all) already done"; else NEED="$NEED all:3"; fi
NSUM=0; for j in $NEED; do NSUM=$((NSUM + ${j#*:})); done
KF=$SD/handover.general; rm -f $KF
if [ $NSUM = 0 ]; then say "5b. nothing left to run in 5b or 5d${HAND:+; GPUs $HAND go on to 5c}"
else
  cap_wait $NSUM "$HAND"; margins 30 20 64 "5b"
  if [ -n "$HAND" ]; then alloc_5b
  elif [ -n "$DRY" ]; then say "5b. (dry run) a real run splits the 6 GPUs that grpo4's eval keeps: the first 2 to the judge, the 3rd to the official model's IFEval, the last 3 to the general evaluation (each takes its share over with KEPT); without kept GPUs each claims its own"
  else say "5b. no kept GPUs: each job still to run claims its own (the judge 2, the official model's IFEval 1, the general evaluation 3)"; fi
  HAND=""; PO=""; PA=""
  case " $NEED " in *" judge:2 "*)
    JCMD="env KEPT=\"$KJ\" GCHOLD=2 bash $ROOT/scripts/04_evaluation/judge_c.sh"; nj=$(lines $JST); JRUN=1
    if [ -n "$DRY" ]; then stage "5d. judge" $JST "judge_c done" env DRY=1 DRYDIR=$SD KEPT="$KJ" GCHOLD=2 bash $ROOT/scripts/04_evaluation/judge_c.sh
    else say "5d. judge starts: $JCMD"; env KEPT="$KJ" GCHOLD=2 bash $ROOT/scripts/04_evaluation/judge_c.sh < /dev/null & PJ=$!; CHILDREN="$CHILDREN $PJ"; say "5d. judge pid $PJ"; fi;;
  esac
  if [ -n "$DRY" ]; then
    case " $NEED " in *" ifeval_only:1 "*) stage "5b. general (official, ifeval_only)" $GD5/general_c_ifeval_only.status "general_c ifeval_only done" env DRY=1 DRYDIR=$SD KEPT="$KO" GCHOLD=4 bash $ROOT/scripts/04_evaluation/general_c.sh ifeval_only $GEN_OFF;; esac
    case " $NEED " in *" all:3 "*) stage "5b. general (all)" $GD5/general_c_all.status "general_c all done" env DRY=1 DRYDIR=$SD KEPT="$KA" KEEP_AT_END=$KF KEEP_HOLD_GB=4 GCHOLD=4 bash $ROOT/scripts/04_evaluation/general_c.sh all $GEN_ALL;; esac
  else
    no=$(lines $GD5/general_c_ifeval_only.status); na=$(lines $GD5/general_c_all.status)
    case " $NEED " in *" ifeval_only:1 "*)
      say "5b. general (official, ifeval_only) starts: env KEPT=\"$KO\" GCHOLD=4 bash $ROOT/scripts/04_evaluation/general_c.sh ifeval_only $GEN_OFF"
      env KEPT="$KO" GCHOLD=4 bash $ROOT/scripts/04_evaluation/general_c.sh ifeval_only $GEN_OFF < /dev/null & PO=$!; CHILDREN="$CHILDREN $PO";;
    esac
    case " $NEED " in *" all:3 "*)
      say "5b. general (all) starts: env KEPT=\"$KA\" KEEP_AT_END=$KF KEEP_HOLD_GB=4 GCHOLD=4 bash $ROOT/scripts/04_evaluation/general_c.sh all $GEN_ALL"
      env KEPT="$KA" KEEP_AT_END=$KF KEEP_HOLD_GB=4 GCHOLD=4 bash $ROOT/scripts/04_evaluation/general_c.sh all $GEN_ALL < /dev/null & PA=$!; CHILDREN="$CHILDREN $PA";;
    esac
    say "5b. general pids: ifeval_only ${PO:-none}, all ${PA:-none}"
    bad=""
    if [ -n "$PO" ]; then wait $PO; ro=$?; drop_child $PO
      since $GD5/general_c_ifeval_only.status $no "general_c ifeval_only done" && say "5b. general (official, ifeval_only) ends: exit $ro" || bad="$bad ifeval_only (exit $ro)"; fi
    if [ -n "$PA" ]; then wait $PA; ra=$?; drop_child $PA
      since $GD5/general_c_all.status $na "general_c all done" && say "5b. general (all) ends: exit $ra" || bad="$bad all (exit $ra)"; fi
    [ -s $KF ] && HAND=$(cat $KF)
    if [ -n "$bad" ]; then
      L=$(gc_left_by "$PO $PA" "$KO $KA"); HAND=$(echo $HAND $L)
      stop "5b. general evaluation failed:$bad (see $GD5/general_c_*.status)" keep
    fi
  fi
fi

# 5c. the reworded test, on the general evaluation's GPUs; beside a running judge at most CAPG - 2 GPUs
read_conf $CONF || stop "5c. $CONF: $C_ERR"
RLM=""; [ "$C_RWRL" = 1 ] && RLM="grpo1 grpo2 grpo3 grpo4"
say "5c. REWORDED_RL=$C_RWRL ($CONF: ${C_FROM}): RL_MODELS=\"$RLM\""
RST=$MK/runs/reworded_c.status; [ -n "$DRY" ] && RST=$SD/reworded_c.status
if [ -z "$DRY" ] && awk -v m="models M1 qwen3.5-4b $RLM" '/^reworded_c starts / {ok = index($0, m) > 0; d = 0} /^reworded_c done / {d = 1} END {exit !(ok && d)}' $RST 2>/dev/null; then
  say "5c. reworded test already done ($RST)"; release_hand
else
  fit_5c; if [ -n "$NGR" ]; then cap_wait $NGR ""; else cap_wait $(echo $HAND | wc -w) "$HAND"; fi
  JNOTE=""; [ $RG -lt $CAPG ] && JNOTE="; the judge's 2 are left out whether or not its claim has locked them"
  if [ -n "$HAND" ]; then say "5c. the reworded test takes over GPUs $HAND"; else say "5c. the reworded test claims $NGR GPUs (NG=$NGR$JNOTE)"; fi
  OLD=$HAND
  if [ -n "$DRY" ]; then stage "5c. reworded" $RST "reworded_c done" env DRY=1 DRYDIR=$SD KEPT="$HAND" NG=$NGR RL_MODELS="$RLM" GCHOLD=4 bash $ROOT/scripts/04_evaluation/reworded_c.sh
  else stage "5c. reworded" $RST "reworded_c done" env KEPT="$HAND" NG=$NGR RL_MODELS="$RLM" GCHOLD=4 bash $ROOT/scripts/04_evaluation/reworded_c.sh; fi
  rc=$?; after_stage $rc "" "$OLD"
  [ $rc = 0 ] || stop "5c. reworded test failed (see $RST)" keep
fi

# 5d's end
if [ -n "$PJ" ]; then
  say "5d. waits for the judge (pid $PJ)"; wait $PJ; rj=$?; drop_child $PJ; p=$PJ; PJ=""
  if since $JST $nj "judge_c done"; then say "5d. judge ends: exit $rj; $(tail -n +$((nj + 1)) $JST | grep -E 'runner exit|without a verdict after|scorer exit' | tail -n 2 | tr '\n' ';')"
  else HAND=$(gc_left_by "$p" "$KJ"); stop "5d. judge calibration failed: exit $rj (see $JST)" keep; fi
fi
release_hand
say "rl_chain_c done"
rm -f $PIDF
