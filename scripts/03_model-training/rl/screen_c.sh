#!/bin/bash
# screen_c.sh: screen_m1_k8.sh for rl_chain_c.sh's stage 2 (research/log/2026-11-01-0010.md, 11-02 22:4x and 23:2x UTC),
# the same screen, outputs and status file, with four changes: (1) the refusal "RL jobs of ours run" leaves out the
# processes that started this script, since screen_m1_k8.sh's pattern rl_chain*.sh matches rl_chain_c.sh itself (any
# other RL chain, launcher or trainer still refuses); (2) KEEP_AT_END=<file>: when run.py ends well, the GPUs are kept
# for the chain's next stage (gpu_claim.py keep, holders of KEEP_HOLD_GB GiB, default gpu_claim.py's, since the next
# stage is verl) and listed in <file>, instead of released; (3) GCHOLD can be set (default 6); (4) DRY=1 writes under
# DRYDIR (default $DRY_DIR/). The original's header follows; the changed lines are marked "screen_c".
# M1 on the fresh RL pool, 8 times each, to select the next GRPO round's tasks by an 8-run pass rate
# (research/log/2026-11-01-0010.md, 11-02 22:xx UTC). Written, not run: start it only when the author approves the GPU
# time (about 47-52 GPU hours, 8-9 hours on 6 GPUs, from the 2-run screen's 340 and 309 episodes per GPU hour).
# Serving as screen_m1_train.sh: one M1 server per GPU (serve_4b_keep_thinking.sh, 24 episodes each, port 1816<GPU>),
# run.py under H1 with --runs 8 and --max-steps 30 on market_v1/rl_pool_v2/train_rl.jsonl, resuming into
# runs/screen_m1_rlpool2_k8 (run.py does run 1 of every task before run 2, so a stopped screen leaves whole runs).
# Then make_grpo_data.py with the k-run selection: --runs 8 (a task counts only with 8 usable runs), --drop-unsolved
# (0/8 dropped), every 1-7/8 task, and a seeded SOLVED_FRAC (default 0.10, MiMo's share) of the 8/8 tasks, with
# per-task pass counts in grpo/run2_train.manifest.json; the dev file becomes grpo/run2_val.parquet as for run1.
# Refuses to start (exit 3) while any RL job of ours runs (an RL chain, GRPO or PPO launcher, or verl's trainer), while
# any other process of ours is on a GPU (the project may use at most 6 GPUs at once, and this job takes 6), or while
# another instance runs. GPUs: NG=6 of 0-9 without a free8.sh lock, claimed with the GPU-claim tool's
# gpu_claim.py through gpuclaim_lib.sh, each with a holder of GCHOLD=6 GiB until its server appears (vllm-0.32 needs
# 0.85 of the GPU's 47.5 GiB free at start, which a larger hold would not leave). When no server starts while another
# user's process is on the GPUs, they are released and claimed afresh once; any other failure of the servers, and a
# failed run.py, keeps the GPUs for the fixed job (restart, which resumes run.py: KEPT=<GPUs> bash screen_m1_k8.sh) and
# exits 1. Every other end releases them. The project's judge job may still start beside this one later (its own rule
# counts free GPUs, not the project's), so check it before a start.
# Usage: [KEPT=<GPUs>] [SOLVED_FRAC=0.1] bash screen_m1_k8.sh. DRY=1 runs the checks, prints the commands instead of
# running them (no GPU, lock or server; G1..G6 stand for the GPUs) and reports a refusal without exiting.
# Stop: kill <pid> (not -9); the status file is market_v1/runs/screen_m1_rlpool2_k8.status.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); . "$ROOT/scripts/paths.sh"
EN=$FP_DATA_EN; MK=$FP_DATA_D2/market_v1
V3=$ROOT/src/02_harness/code/v3; S=$ROOT/src/03_model-training/code/verl_grpo
ME=$ROOT/scripts/03_model-training/rl
R=$FP_RUNS/qwen3.5-4b-base-en; GD=$FP_DATA_D2/grpo
PY=$FP_VENVS/vllm-0.12/bin/python; VPY=$FP_VENVS/verl/bin/python
TASKS=$MK/rl_pool_v2/train_rl.jsonl; NAME=screen_m1_rlpool2_k8; OUT=$MK/runs/$NAME; K=24; NG=6; GCHOLD=${GCHOLD:-6}; RUNS=8  # screen_c (3)
SOLVED_FRAC=${SOLVED_FRAC:-0.1}
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= CUDA_DEVICE_ORDER=PCI_BUS_ID  # run.py and the parquet build use no GPU
if [ -n "$DRY" ]; then SD=${DRYDIR:-$DRY_DIR}; mkdir -p $SD; else SD=$MK/runs; fi  # screen_c (4)
LOG=$SD/$NAME.status; PIDF=$SD/$NAME.pid; GC_LOG=$SD/$NAME.gpuclaim.log
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; [ -n "$DRY" ] && echo "# $1"; return 0; }
nap() { sleep $1 & wait $!; }
source $ROOT/scripts/gpuclaim_c_lib.sh  # gpuclaim_lib.sh's functions, and gc_keep_handover and ancestors_of (screen_c)
[ -n "$DRY" ] && : > $LOG

# 1. refusals. Patterns match a process's command (its first words), not text inside another command line, so a
# monitor whose command line mentions these names does not count. The processes that started this script (the chain)
# do not count (screen_c (1)).
rl_jobs() {
  ps -u "$(id -un)" -o pid=,args= | awk -v anc=" $(ancestors_of $$) " '
    index(anc, " " $1 " ") { next }
    { cmd = $2; scr = $3 }
    cmd ~ /(^|\/)(ba)?sh$/ && scr ~ /(rl_chain[^\/]*|ppo1|grpo_run[^\/]*|run_grpo|run_ppo|eval_ckpt_8gpu)\.sh$/ { print $1, cmd, scr; next }
    cmd ~ /python[0-9.]*$/ && $0 ~ / -m verl\.trainer\.main_ppo / { print $1, cmd, "-m verl.trainer.main_ppo" }'
}
our_gpu_jobs() {  # the GPUs with a process of ours other than a gpu-claim holder
  local x q out=""
  for x in 0 1 2 3 4 5 6 7 8 9; do
    for q in $(gpu_pids $x); do is_holder $q && continue; [ "$(ps -o user= -p $q 2>/dev/null | tr -d ' ')" = "$(id -un)" ] && { out="$out $x"; break; }; done
  done
  echo $out
}
refuse() { say "refused: $1"; [ -n "$DRY" ] && return 0; exit 3; }
if [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then refuse "another $NAME runs (pid $(cat $PIDF))"; fi
j=$(rl_jobs); [ -n "$j" ] && refuse "RL jobs of ours run: $(echo "$j" | tr '\n' ';')"
g=$(our_gpu_jobs); [ -n "$g" ] && refuse "processes of ours are on GPUs $g (the project's limit is 6 GPUs at once)"
[ -s $TASKS ] || refuse "no task file $TASKS"
avail=$(df -P / | awk 'NR == 2 {print int($4 / 1048576)}'); [ "$avail" -gt ${ROOT_MIN_GB:-30} ] || refuse "root disk has ${avail} GB free (<= ${ROOT_MIN_GB:-30})"
[ -n "$DRY" ] || [ -f $GC ] || refuse "gpu-claim is not installed (no $GC)"
[ -n "$DRY" ] || echo $$ > $PIDF
say "$NAME starts: $(wc -l < $TASKS) tasks x $RUNS runs on any $NG free GPUs of 0-9"

SLOGS=""; RUNPID=""
slog() { echo $EN/logs/serve_4b_rlpool2k8_gpu$1.log; }
up() { for i in $(seq 90); do curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q '"M1"' && return 0; sleep 10; done; return 1; }
stop_srvs() {  # <serve logs>: each server's process group (pid on the log's first line), TERM, then KILL after 20 s
  local l p ps=""
  for l in $1; do p=$(head -n 1 $l 2>/dev/null | grep -o "pid [0-9]*" | cut -d' ' -f2); [ -n "$p" ] && { kill -- -$p 2>/dev/null; ps="$ps $p"; }; done
  [ -n "$ps" ] && { sleep 20; for p in $ps; do kill -9 -- -$p 2>/dev/null; done; }
}
cleanup() {
  trap - EXIT INT TERM HUP
  [ -n "$DRY" ] && return
  [ -n "$RUNPID" ] && kill -- -$RUNPID 2>/dev/null
  stop_srvs "$SLOGS"; gc_stop_claim; gc_release
  rm -f $PIDF
}
trap cleanup EXIT; trap 'say "$NAME stopped by a signal"; exit 143' INT TERM HUP

# 2. GPUs
RETRIED=0; HELD=""
claim_gpus() {  # [kept GPUs]: HELD := the GPUs claimed, or the kept ones taken over
  if [ -n "$DRY" ]; then
    HELD="G1 G2 G3 G4 G5 G6"
    say "(dry run) would claim $NG of GPUs 0-9 without a free8.sh lock (now: $(f8_free 0 1 2 3 4 5 6 7 8 9)) with $GC, holders of $GCHOLD GiB${1:+, first taking over the kept GPUs $1}; $HELD stand for them"
    return 0
  fi
  say "claiming $NG of GPUs 0-9 without a free8.sh lock, holders of $GCHOLD GiB until the servers appear${1:+, first taking over the kept GPUs $1}"
  gc_claim $NG $GCHOLD "0 1 2 3 4 5 6 7 8 9" "$1" || exit 1
  HELD=$GC_HELD; say "GPUs claimed: $HELD"
}
start_servers() {  # <GPUs> -> G (GPUs whose server answers), URLS, SLOGS
  local x ok=""
  G=""; URLS=""; SLOGS=""
  for x in $1; do
    cmd="( MAXSEQS=$K setsid sh $ROOT/scripts/serve_4b_keep_thinking.sh $x 1816$x $R/M1 M1 $(slog $x) < /dev/null > /dev/null 2>&1 & )"
    if [ -n "$DRY" ]; then echo "$cmd"; else eval "$cmd"; fi
    SLOGS="$SLOGS $(slog $x)"; ok="$ok $x"
  done
  for x in $ok; do
    if [ -n "$DRY" ] || up 1816$x; then G="$G $x"; URLS="$URLS --base-url http://127.0.0.1:1816$x/v1"
    else say "server on GPU $x did not start (see $(slog $x))"; stop_srvs $(slog $x); fi
  done
  say "servers on GPUs$G"
}
claim_gpus "$(spaces $KEPT)"
start_servers "$HELD"
if [ -z "$G" ] && [ -z "$DRY" ] && [ $RETRIED = 0 ] && contended $HELD; then
  RETRIED=1; say "no server started, and another user's process is on GPUs $HELD: released, claiming again"
  gc_release; claim_gpus; start_servers "$HELD"
fi
if [ -z "$G" ] && [ -z "$DRY" ]; then say "$NAME failed: no server started"; gc_keep_for_restart $GCHOLD "bash $0"; exit 1; fi
[ -z "$DRY" ] && holder_lines $G

# 3. the screen
n=$(echo $G | wc -w)
cmd="$PY -B $V3/run.py --tasks $TASKS --db $MK/market_v1.sqlite --config H1 --runs $RUNS --max-steps 30 --concurrency $((K * n)) --provider qwen --model M1$URLS --out $OUT >> $OUT.log 2>&1"
if [ -n "$DRY" ]; then echo "setsid $cmd &"; rc=0
else eval "setsid $cmd &"; RUNPID=$!; wait $RUNPID; rc=$?; RUNPID=""; fi
rows=$( [ -f $OUT/results.jsonl ] && wc -l < $OUT/results.jsonl || echo 0)
say "run.py exit $rc: $rows rows"
if [ -z "$DRY" ]; then
  stop_srvs "$SLOGS"; SLOGS=""
  [ $rc = 0 ] || { say "$NAME failed: run.py exit $rc (run.py resumes)"; gc_keep_for_restart $GCHOLD "bash $0"; exit 1; }
  if [ -n "$KEEP_AT_END" ] && gc_keep_handover "$KEEP_HOLD_GB" "$KEEP_AT_END"; then HELD=""; say "servers stopped"  # screen_c (2)
  else gc_release; HELD=""; say "servers stopped, GPUs released"; fi
elif [ -n "$KEEP_AT_END" ]; then say "(dry run) would keep GPUs $HELD for the chain's next stage (gpu_claim.py keep, holders of ${KEEP_HOLD_GB:-the default} GiB) and list them in $KEEP_AT_END"
fi

# 4. the GRPO data by the 8-run pass rate
cmd="$VPY -B $ROOT/src/03_model-training/utils/make_grpo_data.py --tasks $TASKS --screen $OUT/results.jsonl --runs $RUNS --drop-unsolved --solved-frac $SOLVED_FRAC --seed 20261103 --manifest $GD/run2_train.manifest.json --out $GD/run2_train.parquet"
cmd2="$VPY -B $ROOT/src/03_model-training/utils/make_grpo_data.py --tasks $MK/hardset_v1/dev_market.jsonl --out $GD/run2_val.parquet"
if [ -n "$DRY" ]; then echo "$cmd"; echo "$cmd2"
else
  for f in $GD/run2_train.parquet $GD/run2_val.parquet; do [ -e $f ] && { say "$f exists: not overwritten"; exit 1; }; done
  eval "$cmd >> $LOG 2>&1" && eval "$cmd2 >> $LOG 2>&1" || { say "GRPO data failed"; exit 1; }
fi
say "$NAME done"
