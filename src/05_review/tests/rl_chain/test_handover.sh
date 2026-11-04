#!/bin/bash
# Test of the chain's GPU hand-over (gpuclaim_c_lib.sh's gc_keep_handover, then gpuclaim_lib.sh's gc_claim with kept
# GPUs) with the installed gpu_claim.py against a stub nvidia-smi and a fake torch, as the GPU-claim tool's tests do:
# the configuration file names the stub, the holders run a Python whose torch is the fake one, and the stub's GPU UUIDs
# (GPU-stub-<i>) match no real GPU, so no holder can touch a real GPU. Locks go to a temporary lock dir, free8.sh's
# locks to another (F8_LOCKS). Each "stage" is a separate bash process that sources the library, as a launcher does.
# Usage: bash test_handover.sh
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); . "$ROOT/scripts/paths.sh"
T=$(mktemp -d ${TEST_TMP:-${TMPDIR:-/tmp}}/rl_chain_c_test.handover.XXXX); mkdir -p $T/state $T/bin $T/fake/torch $T/f8locks
SRC=$GPU_CLAIM_TESTS
/usr/bin/python3 - $SRC $T <<'EOF'
import re, sys
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
case "$(/usr/bin/python3 -c 'import torch; print(torch.__file__)')" in $T/fake/*) ;; *) echo "FAIL torch is not the fake one"; exit 1;; esac
F=0; ok() { echo "ok   $1"; }; bad() { echo "FAIL $1"; F=$((F + 1)); }
LIB='GC_LOG='$T'/gc_$GCN.log; say() { echo "  [$$] $1" >> '$T'/say.log; }; nap() { sleep $1 & wait $!; }; source '"$ROOT"'/scripts/gpuclaim_c_lib.sh'
nlocks() { ls -d $T/locks/gpu? 2>/dev/null | wc -l; }
status() { python3 -B $GPU_CLAIM_PY status --gpus $1 | cut -f2,4; }
# a watcher records the most locks of ours at any moment (stopped by $T/stop_watch; never waited for)
( m=0; while [ ! -f $T/stop_watch ]; do n=$(nlocks); [ $n -gt $m ] && { m=$n; echo $m > $T/max_locks; }; sleep 0.02; done ) &

# 1. stage A claims 6 afresh (free8.sh's lock on GPU 0 leaves it out), then keeps them for the next stage and exits
mkdir $T/f8locks/gpu0
bash -c "$LIB"'; gc_claim 6 "" "0 1 2 3 4 5 6 7 8 9" "" || exit 9; echo "$GC_HELD" > '$T'/a_claimed; gc_keep_handover 4 '$T'/hand_a; echo "rc $? held [$GC_HELD]" > '$T'/a_after'
A=$(cat $T/a_claimed 2>/dev/null); H=$(cat $T/hand_a 2>/dev/null)
[ "$A" = "1 2 3 4 5 6" ] && ok "stage A claimed 1-6 (GPU 0 has a free8.sh lock)" || bad "stage A claimed [$A]"
[ "$H" = "$A" ] && [ "$(cat $T/a_after)" = "rc 0 held []" ] && ok "stage A kept them for the next stage: hand-over file [$H], nothing left to release" || bad "hand-over [$H], $(cat $T/a_after)"
[ "$(ls $T/f8locks)" = gpu0 ] && ok "stage A dropped its free8.sh locks on keeping" || bad "free8.sh locks: $(ls $T/f8locks | tr '\n' ' ')"
s=$(status 1); case "$s" in *"owner "*" gone,"*) ok "after stage A's exit the kept GPU 1 is locked for it, owner gone: $s";; *) bad "status of GPU 1: $s";; esac
sleep 0.5; grep -q "holding" $T/locks/gpu1.holder.log 2>/dev/null && ok "a holder holds kept GPU 1 (fake torch)" || bad "no holding line for GPU 1: $(tail -n 2 $T/locks/gpu1.holder.log 2>/dev/null)"

# 2. stage B takes the kept GPUs over, then keeps them in turn
bash -c "$LIB"'; gc_claim 6 4 "0 1 2 3 4 5 6 7 8 9" "'"$H"'" || exit 9; echo "$GC_HELD $$" > '$T'/b_claimed; gc_keep_handover 2 '$T'/hand_b'
read B1 B2 B3 B4 B5 B6 BP < $T/b_claimed
[ "$B1 $B2 $B3 $B4 $B5 $B6" = "$H" ] && ok "stage B took over exactly the kept GPUs [$H] (claim --kept)" || bad "stage B got [$B1 $B2 $B3 $B4 $B5 $B6]"
grep -q "took over the kept GPUs 1,2,3,4,5,6" $T/gc_.log && ok "gpu_claim.py logged the take-over" || bad "no take-over line in gc.log"
H2=$(cat $T/hand_b); [ "$H2" = "$H" ] && ok "stage B kept them for the next stage" || bad "stage B hand-over [$H2]"

# 3. the 5b split: three stages take over 2, 1 and 3 of them at once, each with its own gpu_claim.py log (GC_LOG.claimed
# holds a claim's GPUs, so concurrent launchers must not share one, as judge_c.sh and the two general_c.sh do not)
set -- $H2; PIDS=""
for part in "$1 $2" "$3" "$4 $5 $6"; do
  n=$(echo $part | wc -w)
  GCN=split$n bash -c "$LIB"'; gc_claim '$n' 2 "0 1 2 3 4 5 6 7 8 9" "'"$part"'" && echo "$GC_HELD" >> '$T'/split; sleep 1; gc_release' & PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done
[ "$(sort $T/split | tr '\n' '|')" = "$(printf '%s\n' "$1 $2" "$3" "$4 $5 $6" | sort | tr '\n' '|')" ] && ok "three concurrent take-overs of disjoint subsets: $(tr '\n' '|' < $T/split)" || bad "split: $(tr '\n' '|' < $T/split)"
[ $(nlocks) = 0 ] && ok "all released after the split" || bad "locks left: $(ls $T/locks)"

# 4. a kept GPU gets another user's process (PID 1) beside its holder: the next stage's take-over is refused, the kept
# GPUs are released and 6 claimed afresh, never more than 6 locked at once
bash -c "$LIB"'; gc_claim 6 "" "1 2 3 4 5 6 7 8 9" "" || exit 9; gc_keep_handover 4 '$T'/hand_c'
H3=$(cat $T/hand_c); echo "GPU-stub-${H3%% *}, 1" > $T/state/apps
bash -c "$LIB"'; gc_claim 6 4 "1 2 3 4 5 6 7 8 9" "'"$H3"'" || exit 9; echo "$GC_HELD" > '$T'/d_claimed; gc_release'
D=$(cat $T/d_claimed 2>/dev/null)
grep -q "were refused" $T/say.log && ok "the take-over was refused (another user's process on GPU ${H3%% *})" || bad "no refusal line"
case " $D " in *" ${H3%% } "*) bad "the busy GPU ${H3%% *} was claimed again: [$D]";; *) [ $(echo $D | wc -w) = 6 ] && ok "6 claimed afresh without the busy GPU: [$D]" || bad "claimed afresh: [$D]";; esac
: > $T/state/apps

# 5. our_gpus counts locks of both kinds and processes of ours; a chain-style release of kept GPUs whose owner is gone
bash -c "$LIB"'; gc_claim 2 "" "7 8" "" || exit 9; gc_keep_handover 4 '$T'/hand_e'
echo "GPU-stub-3, $$" > $T/state/apps
u=$(bash -c "$LIB"'; our_gpus | tr "\n" " "'); [ "$u" = "0 3 7 8 " ] && ok "our_gpus: free8.sh lock 0, our process on 3, kept 7 8 -> [$u]" || bad "our_gpus [$u]"
u=$(bash -c "$LIB"'; our_gpus 7 8 | tr "\n" " "'); [ "$u" = "0 3 " ] && ok "our_gpus leaving out the hand-over 7 8 -> [$u]" || bad "our_gpus without 7 8 [$u]"
for x in 7 8; do case "$(status $x)" in *"owner "*" gone,"*) python3 -B $GPU_CLAIM_PY release $x;; esac; done
[ $(nlocks) = 0 ] && ok "kept GPUs whose owner is gone released" || bad "locks left: $(ls $T/locks)"

# 6. gc_left_by, which the chain uses after a stage: the GPUs a failed stage kept for its restart (owner its PID, gone)
# plus those handed to it that it never took over (any owner gone), but not a lock whose owner still runs
: > $T/state/apps
bash -c "$LIB"'; gc_claim 2 "" "1 2 3 4 5 6 7 8 9" "" || exit 9; gc_keep_handover 4 '$T'/hand_f'
HF=$(cat $T/hand_f)
bash -c "$LIB"'; gc_claim 3 "" "1 2 3 4 5 6 7 8 9" "" || exit 9; echo "$$ $GC_HELD" > '$T'/f_claimed; gc_keep_for_restart "" "bash stage_f.sh"'
read FP F1 F2 F3 < $T/f_claimed
GCN=live bash -c "$LIB"'; gc_claim 1 "" "1 2 3 4 5 6 7 8 9" "" || exit 9; echo "$$ $GC_HELD" > '$T'/live_claimed; while [ ! -f '$T'/live_stop ]; do sleep 0.1; done; gc_release' & LP=$!
for i in $(seq 100); do [ -s $T/live_claimed ] && break; sleep 0.1; done; read LO L1 < $T/live_claimed
got=$(bash -c "$LIB"'; gc_left_by "'"$FP $LO"'" "'"$HF"'"')
want=$(echo $HF $F1 $F2 $F3 | tr ' ' '\n' | sort -n | tr '\n' ' ' | sed 's/ $//')
[ "$got" = "$want" ] && ok "gc_left_by: stage F's restart-kept $F1 $F2 $F3 and the unclaimed hand-over $HF, not the live claim's $L1 -> [$got]" || bad "gc_left_by [$got], want [$want] (live $L1)"
got=$(bash -c "$LIB"'; gc_left_by "" ""'); [ -z "$got" ] && ok "gc_left_by with no PIDs and no GPUs: none" || bad "gc_left_by with nothing given: [$got]"
touch $T/live_stop; wait $LP
python3 -B $GPU_CLAIM_PY release $(echo $HF $F1 $F2 $F3 | tr ' ' ',')
[ $(nlocks) = 0 ] && ok "all released after the gc_left_by test" || bad "locks left: $(ls $T/locks)"

touch $T/stop_watch; sleep 0.1
M=$(cat $T/max_locks); [ "$M" -le 6 ] && ok "at most $M GPUs locked at any moment (cap 6)" || bad "$M GPUs locked at once"
sleep 1; pgrep -f "gpu_holder.py --uuid GPU-stub" > /dev/null && bad "stub holders still run: $(pgrep -f 'gpu_holder.py --uuid GPU-stub' | tr '\n' ' ')" || ok "no holder left running"
rmdir $T/f8locks/gpu0
echo "FAILURES: $F  (work dir $T)"; exit $F
