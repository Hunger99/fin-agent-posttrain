#!/bin/bash
# Test of rl_chain_c.sh's GPU bookkeeping around a stage's end and the chain's stops: after_stage, stop (keep and release),
# fit_hand, alloc_5b, fit_5c and on_signal, sourced from the chain (CHAIN_LIB_ONLY) and run against the installed
# gpu_claim.py with test_handover.sh's stub nvidia-smi and fake torch (stub GPU UUIDs, a temporary lock dir and free8.sh
# lock dir), so that no real GPU, lock or chain file is touched. Each "stage" is a separate bash process that sources
# gpuclaim_c_lib.sh, as a launcher does.
# Usage: bash test_stop.sh
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); . "$ROOT/scripts/paths.sh"
T=$(mktemp -d ${TEST_TMP:-${TMPDIR:-/tmp}}/rl_chain_c_test.stop.XXXX); mkdir -p $T/state $T/bin $T/fake/torch $T/f8locks
SRC=$GPU_CLAIM_TESTS
/usr/bin/python3 - $SRC $T <<'EOF'
import sys
src, t = sys.argv[1], sys.argv[2]
ns = {}
code = open(src).read()
exec(code[code.index("STUB = r'''"):code.index("def apps(")], ns)   # the skill test's STUB and FAKE_TORCH, verbatim
open(f"{t}/bin/nvidia-smi", "w").write(ns["STUB"].format(python="/usr/bin/python3"))
open(f"{t}/fake/torch/__init__.py", "w").write(ns["FAKE_TORCH"])
open(f"{t}/state/gpus", "w").write("".join(f"{i}, GPU-stub-{i}\n" for i in range(10)))
open(f"{t}/state/apps", "w").write("")
open(f"{t}/stub.conf", "w").write(f"nvidia_smi = {t}/bin/nvidia-smi\npython = /usr/bin/python3\nlock_dir = {t}/locks\npoll_s = 0.05\n")
EOF
chmod +x $T/bin/nvidia-smi
export GPU_STUB_DIR=$T/state GPU_CLAIM_CONFIG=$T/stub.conf PYTHONPATH=$T/fake PATH=$T/bin:$PATH F8_LOCKS=$T/f8locks CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
[ "$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | head -n 1)" = "0, GPU-stub-0" ] || { echo "FAIL the stub nvidia-smi is not first on PATH"; exit 1; }
F=0; ok() { echo "ok   $1"; }; bad() { echo "FAIL $1"; F=$((F + 1)); }
LIB='GC_LOG='$T'/gc_$GCN.log; say() { echo "  [$$] $1" >> '$T'/say.log; }; nap() { sleep $1 & wait $!; }; source '"$ROOT"'/scripts/gpuclaim_c_lib.sh'
nlocks() { ls -d $T/locks/gpu? 2>/dev/null | wc -l; }
locked() { ls -d $T/locks/gpu? 2>/dev/null | sed 's#.*/gpu##' | sort -n | tr '\n' ' ' | sed 's/ $//'; }
srt() { echo $* | tr ' ' '\n' | sort -n | tr '\n' ' ' | sed 's/ $//'; }
( m=0; while [ ! -f $T/stop_watch ]; do n=$(nlocks); [ $n -gt $m ] && { m=$n; echo $m > $T/max_locks; }; sleep 0.02; done ) &

CHAIN_LIB_ONLY=1 source $ROOT/scripts/rl_chain_c.sh
case "$GC_LOCK_DIR" in $T/*) ;; *) echo "FAIL the lock dir is not the test's: $GC_LOCK_DIR"; exit 1;; esac
SD=$T; LOG=$T/chain.status; PIDF=$T/chain.pid; GC_LOG=$T/chain_gc.log; JST=$T/judge.status; : > $JST
stage_bg() { bash -c "$LIB; $1" & STAGE_PID=$!; wait $STAGE_PID; }  # <commands>: a launcher-like stage, waited for

# 1. a stage handed 6 kept GPUs exits before taking them over (eval_ckpt_c.sh's merge failure): they stay in HAND
stage_bg 'gc_claim 6 "" "0 1 2 3 4 5 6 7 8 9" "" || exit 9; gc_keep_handover 4 '$T'/hand_a'
HAND=$(cat $T/hand_a); OLD=$HAND
stage_bg 'exit 1'; after_stage 1 "" "$OLD"
[ "$(srt $HAND)" = "$(srt $OLD)" ] && ok "after_stage: GPUs handed to a stage that failed before its claim stay in HAND [$HAND]" || bad "after a pre-claim failure HAND=[$HAND], handed [$OLD]"
( stop "5a. eval grpo2 failed" keep ); rc=$?
[ $rc = 1 ] && [ $(nlocks) = 6 ] && grep -q "stay kept, with holders.*KEPT=\"$HAND\" setsid nohup bash" $LOG \
  && ok "stop keep: exit 1, the 6 GPUs still locked, the chain's restart command with KEPT=\"$HAND\" written" || bad "stop keep: rc $rc, locks [$(locked)], status: $(tail -n 2 $LOG)"

# 2. the failed stage's restart takes them over and fails again, keeping them for its restart (grpo2.sh's crash keep)
OLD=$HAND
stage_bg 'gc_claim 6 "" "0 1 2 3 4 5 6 7 8 9" "'"$OLD"'" || exit 9; gc_keep_for_restart "" "bash grpo2.sh"; exit 1'
after_stage 1 "" "$OLD"
[ "$(srt $HAND)" = "$(srt $OLD)" ] && grep -q "took over the kept GPUs" $T/gc_.log && ok "after_stage: a stage that took them over and kept them for its restart leaves them in HAND [$HAND]" || bad "after a crash keep HAND=[$HAND]"

# 3. a stop that is not a stage's failure (a margin) releases them
( stop "4. ppo2: root disk 29 GB free" ); rc=$?
[ $rc = 1 ] && [ $(nlocks) = 0 ] && ok "stop without keep: exit 1, the kept GPUs released" || bad "stop release: rc $rc, locks [$(locked)]"
HAND=""

# 4. a stage that ends well hands on its handover file; fit_hand releases a hand-over of the wrong size
stage_bg 'gc_claim 3 "" "0 1 2 3 4 5 6 7 8 9" "" || exit 9; gc_keep_handover 4 '$T'/hand_d'
after_stage 0 $T/hand_d ""
[ "$HAND" = "$(cat $T/hand_d)" ] && ok "after_stage success: HAND = the handover file [$HAND]" || bad "after success HAND=[$HAND]"
fit_hand 6
[ -z "$HAND" ] && [ $(nlocks) = 0 ] && ok "fit_hand 6 releases 3 handed GPUs" || bad "fit_hand: HAND [$HAND], locks [$(locked)]"

# 5. alloc_5b: 6 kept GPUs split 2/1/3; only the general evaluation left: 3 to it, 3 released; partial hand-overs
stage_bg 'gc_claim 6 "" "0 1 2 3 4 5 6 7 8 9" "" || exit 9; gc_keep_handover 2 '$T'/hand_e'
HE=$(cat $T/hand_e); set -- $HE
HAND=$HE; NEED=" judge:2 ifeval_only:1 all:3"; alloc_5b
[ "$KJ|$KO|$KA" = "$1 $2|$3|$4 $5 $6" ] && [ -z "$HAND" ] && [ $(nlocks) = 6 ] && ok "alloc_5b, all three jobs: [$KJ] [$KO] [$KA], nothing released" || bad "alloc_5b all: [$KJ] [$KO] [$KA], HAND [$HAND], locks [$(locked)]"
HAND=$HE; NEED=" all:3"; alloc_5b
[ "$KJ|$KO|$KA" = "||$1 $2 $3" ] && [ "$(locked)" = "$(srt $1 $2 $3)" ] && ok "alloc_5b, only the general evaluation: [$KA], the other 3 released" || bad "alloc_5b all only: [$KJ] [$KO] [$KA], locks [$(locked)]"
HAND="$1 $2 $3"; NEED=" judge:2 all:3"; alloc_5b
[ "$KJ|$KO|$KA" = "||$1 $2 $3" ] && [ $(nlocks) = 3 ] && ok "alloc_5b, 3 kept with the judge and the general evaluation to run: all 3 to the general evaluation" || bad "alloc_5b 3 of 5: [$KJ] [$KO] [$KA], locks [$(locked)]"
HAND="$1 $2 $3"; NEED=" judge:2 ifeval_only:1"; alloc_5b
[ "$KJ|$KO|$KA" = "$1 $2|$3|" ] && [ $(nlocks) = 3 ] && ok "alloc_5b, 3 kept with the judge and IFEval to run: 2 and 1" || bad "alloc_5b judge+ifeval: [$KJ] [$KO] [$KA], locks [$(locked)]"
HAND="$1 $2"; NEED=" ifeval_only:1 all:3"; alloc_5b
[ "$KJ|$KO|$KA" = "||" ] && [ "$(locked)" = "$3" ] && ok "alloc_5b, 2 kept that fit no job still to run (IFEval 1, general 3): released" || bad "alloc_5b no fit: [$KJ] [$KO] [$KA], locks [$(locked)]"
python3 -B $GC release $3; KJ=""; KO=""; KA=""

# 6. fit_5c: while the judge runs, the reworded test claims 4 (or takes over at most 4); otherwise 6
sleep 30 & PJ=$!; DRY=""; JRUN=1
HAND="7 8 9"; fit_5c; [ "$RG $NGR $HAND" = "4  7 8 9" ] && ok "fit_5c, judge running, 3 handed: they are taken over" || bad "fit_5c 3 handed: RG $RG NGR [$NGR] HAND [$HAND]"
HAND=""; fit_5c; [ "$RG $NGR" = "4 4" ] && ok "fit_5c, judge running, none handed: NG=4" || bad "fit_5c none handed: RG $RG NGR [$NGR]"
HAND="4 5 6 7 8 9"; fit_5c; [ "$RG $NGR [$HAND]" = "4 4 []" ] && ok "fit_5c, judge running, 6 handed: released, NG=4" || bad "fit_5c 6 handed: RG $RG NGR [$NGR] HAND [$HAND]"
kill $PJ; wait $PJ 2>/dev/null; PJ=""; JRUN=""
HAND=""; fit_5c; [ "$RG $NGR" = "6 6" ] && ok "fit_5c, no judge: NG=6" || bad "fit_5c no judge: RG $RG NGR [$NGR]"

# 7. a stop while the judge runs: the failed stage's GPUs are released first, then the judge is waited for; a judge that
# fails keeps its 2 for the restart
stage_bg 'gc_claim 3 "" "0 1 2 3 4 5 6 7 8 9" "" || exit 9; gc_keep_for_restart 4 "bash reworded_c.sh"; exit 1'; after_stage 1 "" ""; H7=$HAND
( bash -c "$LIB"'; gc_claim 2 "" "0 1 2 3 4 5 6 7 8 9" "" || exit 9; echo "$GC_HELD" > '$T'/judge_claimed; sleep 1; gc_keep_for_restart 2 "bash judge_c.sh"; exit 1' & PJ=$!; nj=0
  for i in $(seq 100); do [ -s $T/judge_claimed ] && break; sleep 0.1; done
  stop "5c. reworded test failed" keep ); rc=$?  # in one subshell, so that stop can wait for the judge, its child
JG=$(cat $T/judge_claimed)
[ $rc = 1 ] && [ "$(locked)" = "$(srt $JG)" ] && grep -q "GPUs $H7 released: the chain stops once the judge ends" $LOG && grep -q "judge calibration failed too" $LOG && grep -q "KEPT=\"$(gc_left_by "" "$JG")\"" $LOG \
  && ok "stop with the judge running: the reworded GPUs [$H7] released first, the failed judge's [$JG] kept with the restart command" || bad "stop with judge: rc $rc, locks [$(locked)], judge [$JG], status: $(tail -n 4 $LOG | tr '\n' '|')"
python3 -B $GC release $(commas $JG); PJ=""; HAND=""

# 8. on_signal: TERM to the chain while a stage holds 2 GPUs (and dies on TERM without releasing) and 1 kept GPU waits
# to be handed on: the stage is stopped and all three are released; exit 143
stage_bg 'gc_claim 1 "" "0 1 2 3 4 5 6 7 8 9" "" || exit 9; gc_keep_handover 4 '$T'/hand_h'
( trap on_signal INT TERM HUP
  bash -c "$LIB"'; gc_claim 2 "" "0 1 2 3 4 5 6 7 8 9" "" || exit 9; echo "$GC_HELD" > '$T'/sig_claimed; exec sleep 300' & p=$!
  CHILDREN=" $p"; HAND=$(cat $T/hand_h); wait $p ) & S=$!
for i in $(seq 100); do [ -s $T/sig_claimed ] && break; sleep 0.1; done
[ $(nlocks) = 3 ] || bad "before the signal: locks [$(locked)]"
kill -TERM $S; wait $S; rc=$?
[ $rc = 143 ] && [ $(nlocks) = 0 ] && grep -q "left locked by the stopped stages or kept for a stage that had not started, released" $LOG \
  && ok "on_signal: exit 143, the stage's 2 GPUs and the kept 1 released" || bad "on_signal: rc $rc, locks [$(locked)], status: $(tail -n 3 $LOG | tr '\n' '|')"

touch $T/stop_watch; sleep 0.1
M=$(cat $T/max_locks); [ "$M" -le 6 ] && ok "at most $M GPUs locked at any moment (cap 6)" || bad "$M GPUs locked at once"
sleep 1; pgrep -f "gpu_holder.py --uuid GPU-stub" > /dev/null && bad "stub holders still run: $(pgrep -f 'gpu_holder.py --uuid GPU-stub' | tr '\n' ' ')" || ok "no holder left running"
echo "FAILURES: $F  (work dir $T)"; exit $F
