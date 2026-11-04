#!/bin/bash
# eval_ckpt_c.sh: eval_ckpt_8gpu.sh for rl_chain_c.sh's final evaluations (research/log/2026-11-01-0010.md, 11-02 22:4x
# and 23:2x UTC): the same evaluation, outputs, status and comparison files, with five changes: (1) KEEP_AT_END=<file>:
# when the v2 phase ends, the GPUs are kept for the chain's next stage (gpu_claim.py keep, holders of KEEP_HOLD_GB GiB,
# default GCHOLD) and listed in <file>, instead of released; (2) GCHOLD can be set (default 6, as the original); (3)
# DRY=1 writes under DRYDIR (default $DRY_DIR/); (4) "eval <name> done" is written only when every run.py exited
# 0 (each v3 file, and each v2 set's last pass); otherwise the GPUs are kept for the eval's restart, which resumes the
# runs' rows, the comparisons are still written, and the script ends with "eval <name> failed: ..." and exit 1; (5) the
# merge writes into rl/hf_<name>.part and renames it to rl/hf_<name> once the merge and the copied files are complete,
# so that a merge cut short is redone on the next start (its .part is moved aside to rl/_trash_hf_<name>_part_<time>/)
# instead of refused. The original's header follows; changed lines are marked "eval_c".
# A checkpoint's evaluation on 6 GPUs by M1's evaluation (research/log/2026-11-01-0010.md, 11-02 04:0x UTC), so that
# its rows compare with M1, the official Qwen3.5-4B, the start model and the teacher: after_m1.sh's m1_job with the
# model directory and name as parameters and every given GPU from the start (one server per GPU, no adding or
# restarting). Harness v3 (serve_4b_keep_thinking.sh, 24 episodes per server, H1, 100 steps, provider qwen):
# test_market, test_finsearchcomp and test_docfinqa once each, then dev_market 4 times; then the v2 wealth and ML test
# sets (code_collect/run.py, serve_4b.sh with Qwen3.5's own template at 96 sequences, wealth 64 and ML 12 episodes per
# server, two passes). A verl actor checkpoint is first merged into rl/hf_<name> with verl's model_merger, the files
# it lacks (tokenizer, chat template, processor, generation config) copied from M1, and it is stopped unless its
# RoPE base is M1's 1e7 under the eval venv's transformers too (the RoPE-base fault, research/log/2026-10-28-1937.md).
# GPUs (research/log/2026-11-01-0010.md, 11-02 07:4x UTC; the author: "改为十个 GPU，只要有八个空，就立刻启动，占住那八
# 个"; 08:3x UTC: "你把用八卡的删掉，后面都改成六卡"; 07:5x and 08:3x UTC: the GPU-claim tool's holders by default): any of the
# host's 10 GPUs (0-9) may be used when no process of any user is on it. By default ("auto") the script claims NG=6 of
# those that have no free8.sh lock with the GPU-claim tool's gpu_claim.py (gpuclaim_lib.sh), each with a
# holder of GCHOLD=6 GiB until its server appears: vllm-0.32 refuses to start with less free than 0.85 of the GPU's
# 47.5 GiB (40.4 GiB), which the default 36 GB hold would not leave. A given list (any number) is claimed whole once all
# of it is free. KEPT=<GPUs> first takes over the GPUs that a failed earlier instance kept; when gpu_claim.py refuses
# them (no longer kept, or another user's process on one), they are released and NG claimed afresh. The v3 and the v2
# phase both run on these GPUs: holders keep them from the v3 servers' stop until the v2 servers appear (gpu_claim.py
# keep, then claim --kept). A phase in which no server starts fails the eval: once, when another user's process is on
# its GPUs in two checks a second apart, they are released, NG claimed afresh and the phase's servers started again;
# otherwise the GPUs stay kept for the fixed eval (status file: the restart and release commands) and the script exits
# 1. Every other end releases them. The file keeps its 8gpu name, which rl_chain_b.sh calls. Servers stop by process
# group, and the GPUs go at the end, or when the script gets INT/TERM/HUP. Then hm5.py (each v3 test file: M1, official and start model -> <name>,
# <name> -> teacher), paired_runs.py and hm5.py --passk (dev x4) and hm3.py (wealth + ML, as for M1) write
# rl/eval_<name>.compare.{json,txt} next to the status file rl/eval_<name>.status.
# Usage: [KEPT=<GPUs>] eval_ckpt_8gpu.sh <verl actor checkpoint dir (.../global_step_N/actor) or HF model dir> <name>
#        [GPUs: "auto" (default) or a list of 0-9, e.g. "3 4 5 6 7 8"]. DRY=1 prints the commands instead of running them
#        (no GPU, lock or merge; GPUs G1..G6 stand for the 6 that auto would take) and writes the comparisons of whatever
#        rows exist under ppo_prep/dry/. Stop it with kill <pid> (not -9). gpu_claim.py's messages: eval_<name>.gpuclaim.log.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
EN=$FP_DATA_EN; O=$FP_DATA_D2/harness_v2; P=$O/pilot; H=$EN/harness
MK=$FP_DATA_D2/market_v1; HS=$MK/hardset_v1; V3=$ROOT/src/02_harness/code/v3
R=$FP_RUNS/qwen3.5-4b-base-en
RL=$FP_RUNS/rl; M1=$R/M1; PY=$FP_VENVS/vllm-0.12/bin/python
VPY=$FP_VENVS/verl/bin/python; V32PY=$FP_VENVS/vllm-0.32/bin/python; K=24; NG=6
LONG=xbrl_ratio,q4_margin_rank,filing_reaction,extreme_period,ma_cross
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= CUDA_DEVICE_ORDER=PCI_BUS_ID  # the clients, merger and checks use no GPU; each server sets its own, by nvidia-smi's index
source $ROOT/scripts/gpuclaim_c_lib.sh  # gpuclaim_lib.sh's gc_claim, gc_keep, gc_keep_for_restart, gc_release, contended; gc_keep_handover (eval_c)

usage() { echo "usage: $0 <verl actor checkpoint dir or HF model dir> <name> [GPUs: \"auto\" (default, any $NG free of 0-9) or a list of 0-9]" >&2; exit 2; }
[ $# -ge 2 ] && [ $# -le 3 ] || usage
ARGS="$*"; SRC=${1%/}; NAME=$2; GPUS=$(echo "${3:-auto}" | tr ',' ' '); KEPT=$(spaces $KEPT)
for g in $KEPT; do case $g in [0-9]) ;; *) echo "KEPT: not a GPU of this host (0-9): $g" >&2; exit 2;; esac; done
echo "$NAME" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]*$' || { echo "name must be letters, digits, '.', '_' or '-': $NAME" >&2; exit 2; }
if [ "$GPUS" != auto ]; then
  GPUS=$(echo $GPUS); [ -n "$GPUS" ] || usage
  for g in $GPUS; do case $g in [0-9]) ;; *) echo "not a GPU of this host (0-9): $g" >&2; exit 2;; esac; done
  [ "$(echo $GPUS | tr ' ' '\n' | sort | uniq -d)" = "" ] || { echo "a GPU is given twice: $GPUS" >&2; exit 2; }
fi
[ -f "$SRC/actor/fsdp_config.json" ] && SRC=$SRC/actor  # global_step_N given instead of its actor/
if [ -f "$SRC/fsdp_config.json" ]; then KIND=verl; HF=$RL/hf_$NAME
elif [ -f "$SRC/config.json" ]; then KIND=hf; HF=$SRC
else echo "neither a verl actor checkpoint (fsdp_config.json) nor a HF model dir (config.json): $SRC" >&2; exit 2; fi
if [ -n "$DRY" ]; then SD=${DRYDIR:-$DRY_DIR}; mkdir -p $SD; else SD=$RL; fi  # eval_c (3)
LOG=$SD/eval_$NAME.status; PIDF=$SD/eval_$NAME.pids; CMP=$SD/eval_$NAME.compare; GC_LOG=$SD/eval_$NAME.gpuclaim.log
GCHOLD=${GCHOLD:-6}; RETRIED=0  # GiB a holder takes (see the header; eval_c (2)); whether the one contention retry has been used
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; [ -n "$DRY" ] && echo "# $1"; return 0; }
idle_gpu() { [ -z "$(ours_on $1)" ]; }  # no process of ours on it but our holders
up() { for i in $(seq 90); do curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "\"$2\"" && return 0; sleep 10; done; return 1; }
port_used() { ss -Hltn "( sport = :$1 )" | grep -q .; }
rows() { [ -f $MK/runs/$1/results.jsonl ] && wc -l < $MK/runs/$1/results.jsonl || echo 0; }
file_of() { case $1 in market) echo "$HS/test_market.jsonl $MK/market_v1.sqlite";; fsc) echo "$HS/test_finsearchcomp.jsonl $MK/market_v1.sqlite";;
  docfinqa) echo "$HS/test_docfinqa.jsonl $EN/harness/docfinqa_test";; dev) echo "$HS/dev_market.jsonl $MK/market_v1.sqlite";; esac; }
slog3() { echo $EN/logs/serve_4b_tesths_${NAME}_gpu$1.log; }
slog2() { echo $EN/logs/serve_4b_test_${NAME}_gpu$1.log; }
nap() { sleep $1 & wait $!; }
# bg <pid file> <command>: the command in its own session, waited for so that a signal reaches the trap at once
bg() {
  if [ -n "$DRY" ]; then echo "setsid $2 &"; return 0; fi
  eval "setsid $2 &"; local p=$! rc; echo $p >> $1; wait $p; rc=$?; sed -i "/^$p\$/d" $1; return $rc
}
stop_srvs() {  # <serve logs>: each server's process group (pid on the log's first line), TERM, then KILL after 20 s
  local l p ps=""
  for l in $1; do p=$(head -n 1 $l 2>/dev/null | grep -o "pid [0-9]*" | cut -d' ' -f2); [ -n "$p" ] && { kill -- -$p 2>/dev/null; ps="$ps $p"; }; done
  [ -n "$ps" ] && { sleep 20; for p in $ps; do kill -9 -- -$p 2>/dev/null; done; }
}
HELD=""; SLOGS=""; SUBS=""
cleanup() {
  trap - EXIT INT TERM HUP
  [ -n "$DRY" ] && return
  local p g
  for p in $SUBS; do kill $p 2>/dev/null; done                      # the v2 pass loops, before their run.py
  for p in $(cat $PIDF $PIDF.* 2>/dev/null); do kill -- -$p 2>/dev/null; done  # run.py / merger sessions
  stop_srvs "$SLOGS"; gc_stop_claim; gc_release  # GPUs kept for a restart are no longer in GC_HELD
  rm -f $PIDF $PIDF.*
}
trap cleanup EXIT; trap 'say "$NAME stopped by a signal"; exit 143' INT TERM HUP

# outputs, as M1's with M1 replaced by <name>; a name whose outputs exist from another model is refused
V3OUT="test_hs_market_${NAME}_H1 test_hs_fsc_${NAME}_H1 test_hs_docfinqa_${NAME}_H1 dev_hs_market_${NAME}_H1_x4"
V2OUT="$O/runs_q4b_test/${NAME}_H1_wealth $O/runs_q4b_test/${NAME}_H1_ml"
ex=""; for d in $V3OUT; do [ -e $MK/runs/$d ] && ex="$ex $d"; done; for d in $V2OUT; do [ -e $d ] && ex="$ex $(basename $d)"; done
if [ -n "$ex" ] && ! grep -Fqx "model $HF" $RL/eval_$NAME.status 2>/dev/null; then
  if [ -z "$DRY" ]; then echo "outputs of another model already carry the name $NAME:$ex" >&2; trap - EXIT; exit 2; fi
  say "(dry run) outputs exist under this name and run.py would resume them:$ex"
fi
[ -n "$DRY" ] && : > $LOG
if [ "$GPUS" = auto ]; then say "eval $NAME starts: $KIND $SRC on any $NG free GPUs of 0-9"; else say "eval $NAME starts: $KIND $SRC on GPUs $GPUS"; fi
echo "model $HF" >> $LOG
avail=$(df -P / | awk 'NR == 2 {print int($4 / 1048576)}')
[ "$avail" -gt ${ROOT_MIN_GB:-30} ] || { say "root disk has ${avail} GB free (<= ${ROOT_MIN_GB:-30}): stopped"; exit 1; }
[ -n "$DRY" ] || [ -f $GC ] || { say "gpu-claim is not installed (no $GC): stopped"; exit 1; }

# 1. merge a verl checkpoint and check it against M1
if [ $KIND = verl ]; then
  if [ -f $HF/config.json ] && grep -Fqx "merged from $SRC" $LOG; then say "already merged: $HF"
  elif [ -e $HF ] && [ -z "$DRY" ]; then say "$HF exists but was not merged from $SRC here: stopped"; exit 1
  else
    room=$(df -P $RL | awk 'NR == 2 {print int($4 / 1048576)}'); [ "$room" -gt 20 ] || { say "${room} GB free for the merge: stopped"; exit 1; }
    PART=$HF.part  # eval_c (5)
    if [ -e $PART ] && [ -z "$DRY" ]; then a=$RL/_trash_hf_${NAME}_part_$(date -u +%m%d%H%M%S); mv $PART $a && say "an earlier merge's $PART, cut short, moved to $a"; fi
    bg $PIDF "$VPY -m verl.model_merger merge --backend fsdp --local_dir $SRC --target_dir $PART > $RL/merge_$NAME.log 2>&1" \
      || { say "merge failed (see $RL/merge_$NAME.log)"; exit 1; }
    for f in chat_template.jinja tokenizer.json tokenizer_config.json processor_config.json generation_config.json preprocessor_config.json video_preprocessor_config.json; do
      if [ -n "$DRY" ]; then echo "[ -f $PART/$f ] || cp $M1/$f $PART/"; continue; fi
      [ -f $PART/$f ] || { [ -f $M1/$f ] && { cp $M1/$f $PART/ || { say "copying $f from M1 failed: stopped"; exit 1; }; } && say "copied $f from M1"; }
    done
    echo "merged from $SRC" >> $LOG  # before the rename: a stop between the two leaves no $HF, so the next start merges again
    if [ -n "$DRY" ]; then echo "mv $PART $HF"; else mv $PART $HF || { say "renaming $PART to $HF failed: stopped"; exit 1; }; fi
  fi
fi
# the RoPE base in config.json and as the eval venv's transformers reads it, the architecture, vocabulary and tied
# embeddings as M1's; a generation config that differs from M1's in any field but the version is replaced by M1's
check=$(python3 - $HF $M1 $KIND 2>&1 <<'EOF'
import hashlib, json, os, shutil, sys
hf, m1, kind = sys.argv[1:4]
c, m = json.load(open(f"{hf}/config.json")), json.load(open(f"{m1}/config.json"))
t, tm = c.get("text_config", c), m.get("text_config", m)
rope = (t.get("rope_parameters") or {}).get("rope_theta", t.get("rope_theta"))
bad = [k for k, ok in (("rope_theta", rope == 1e7), ("architectures", c.get("architectures") == m["architectures"]),
                       ("vocab_size", t.get("vocab_size") == tm["vocab_size"]),
                       ("tie_word_embeddings", c.get("tie_word_embeddings", t.get("tie_word_embeddings")) == m["tie_word_embeddings"])) if not ok]
note = ""
skip = ("transformers_version", "_from_model_config")
g, gm = (json.load(open(p)) if os.path.exists(p) else {} for p in (f"{hf}/generation_config.json", f"{m1}/generation_config.json"))
if {k: v for k, v in g.items() if k not in skip} != {k: v for k, v in gm.items() if k not in skip}:
    if kind == "verl" and not os.environ.get("DRY"):  # only the directory this script merged
        if g:
            os.replace(f"{hf}/generation_config.json", f"{hf}/generation_config.merged.json")
        shutil.copy(f"{m1}/generation_config.json", f"{hf}/generation_config.json")
        note = "; generation_config.json replaced by M1's (the merged one kept as generation_config.merged.json)"
    else:
        note = f"; generation_config.json differs from M1's: {g}"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
if sha(f"{hf}/tokenizer.json") != sha(f"{m1}/tokenizer.json"):
    note += "; tokenizer.json differs from M1's"
print(("ok" if not bad else "BAD " + ",".join(bad)) + f": {c.get('architectures')} rope_theta {rope} vocab {t.get('vocab_size')}" + note)
EOF
)
if [ -n "$DRY" ] && [ $KIND = verl ] && [ ! -f $HF/config.json ]; then check="ok (dry run: not merged yet)"; fi
say "config check: $check"
case "$check" in ok*) ;; *) say "$NAME not evaluated: its config differs from M1's"; exit 1;; esac
if [ -z "$DRY" ] || [ -f $HF/config.json ]; then
  vr=$($V32PY -I -c "import sys; from transformers import AutoConfig; c = AutoConfig.from_pretrained(sys.argv[1]); print(getattr(c, 'text_config', c).rope_parameters['rope_theta'])" $HF 2>/dev/null)
  say "rope_theta under vllm-0.32's transformers: $vr"
  [ "$vr" = 10000000 ] || [ "$vr" = 10000000.0 ] || { say "$NAME not evaluated: the eval venv reads RoPE base $vr"; exit 1; }
fi

# 2. GPUs: gc_claim waits for gpu_claim.py in the background, so that a signal reaches the trap at once; the GPUs it
# claimed but did not yet hand over are released by cleanup too.
if [ "$GPUS" = auto ]; then GN=$NG; CANDS="0 1 2 3 4 5 6 7 8 9"; else GN=$(echo $GPUS | wc -w); CANDS=$GPUS; fi
claim_gpus() {  # [kept GPUs]: HELD := the GPUs claimed, or the kept ones taken over (see the header)
  if [ -n "$DRY" ]; then
    if [ "$GPUS" = auto ]; then HELD=$(for i in $(seq $NG); do printf 'G%s ' $i; done); else HELD=$GPUS; fi; HELD=$(echo $HELD)
    say "(dry run) would claim $GN of GPUs $CANDS without a free8.sh lock (now: $(f8_free $CANDS)) with $GC, holders of $GCHOLD GiB${1:+, first taking over the kept GPUs $1}; $HELD stand for them"
    return 0
  fi
  say "claiming $GN of GPUs $CANDS without a free8.sh lock, holders of $GCHOLD GiB until the servers appear${1:+, first taking over the kept GPUs $1}"
  gc_claim $GN $GCHOLD "$CANDS" "$1" || exit 1
  HELD=$GC_HELD; say "GPUs claimed: $HELD"
}
claim_gpus "$KEPT"

# 3. harness v3: one server per GPU, then the four files one after another on all of them
start_servers() {  # <v3|v2> <GPUs> -> G (GPUs whose server answers), PORTS, SLOGS
  local x l port ok=""
  G=""; PORTS=""; SLOGS=""
  for x in $2; do
    if [ $1 = v3 ]; then port=1816$x; l=$(slog3 $x); else port=1817$x; l=$(slog2 $x); fi
    if [ -z "$DRY" ] && port_used $port; then say "port $port in use: GPU $x left out"; continue; fi
    if [ $1 = v3 ]; then cmd="( GENCFG= MAXSEQS=$K setsid sh $ROOT/scripts/serve_4b_keep_thinking.sh $x $port $HF $NAME $l < /dev/null > /dev/null 2>&1 & )"
    else cmd="( MAXSEQS=96 setsid sh $ROOT/scripts/serve_4b.sh $x $port $HF $NAME $l < /dev/null > /dev/null 2>&1 & )"; fi
    if [ -n "$DRY" ]; then echo "$cmd"; else eval "$cmd"; fi
    SLOGS="$SLOGS $l"; ok="$ok $x"
  done
  for x in $ok; do
    if [ $1 = v3 ]; then port=1816$x; l=$(slog3 $x); else port=1817$x; l=$(slog2 $x); fi
    if [ -n "$DRY" ] || up $port $NAME; then G="$G $x"; PORTS="$PORTS $port"
    else say "$1 server on GPU $x did not start (see $l)"; stop_srvs $l; SLOGS=$(echo $SLOGS | tr ' ' '\n' | grep -vx "$l" | tr '\n' ' '); fi
  done
  say "$1 servers on GPUs$G"
}
v3file() {  # <file key> <runs> <out name>
  local T db n p urls rc
  read T db <<EOF
$(file_of $1)
EOF
  n=$(echo $PORTS | wc -w); urls=""; for p in $PORTS; do urls="$urls --base-url http://127.0.0.1:$p/v1"; done
  bg $PIDF "$PY -B $V3/run.py --tasks $T --db $db --config H1 --runs $2 --max-steps 100 --concurrency $((K * n)) --provider qwen --model $NAME$urls --out $MK/runs/$3 >> $MK/runs/$3.log 2>&1"
  rc=$?; [ -n "$DRY" ] && return
  say "$NAME $3 exit $rc: $(rows $3) rows, $(grep -c '"passed": true' $MK/runs/$3/results.jsonl) passed"
  [ $rc = 0 ] || BADRUNS="$BADRUNS $3 (exit $rc)"  # eval_c (4)
}
BADRUNS=""
servers() {  # <v3|v2> <GPUs>: start_servers; a phase in which no server starts fails the eval (see the header)
  start_servers $1 "$2"
  if [ -z "$G" ] && [ -z "$DRY" ] && [ $RETRIED = 0 ] && contended $HELD; then
    RETRIED=1; say "no $1 server started, and another user's process is on GPUs $HELD: released, claiming again"
    gc_release; claim_gpus; start_servers $1 "$HELD"
  fi
  [ -n "$DRY" ] && return 0
  [ -z "$G" ] && { say "$NAME failed: no $1 server started"; gc_keep_for_restart $GCHOLD "bash $0 $ARGS"; exit 1; }
  holder_lines $G
}
servers v3 "$HELD"
for f in market fsc docfinqa; do v3file $f 1 test_hs_${f}_${NAME}_H1; done
v3file dev 4 dev_hs_market_${NAME}_H1_x4
[ -z "$DRY" ] && stop_srvs "$SLOGS"; SLOGS=""

# 4. the wealth and ML test sets, v2 harness, Qwen3.5's own template, on the same GPUs, which holders keep from the v3
# servers' stop until the v2 servers appear: every GPU gets up to 10 minutes in all for the v3 servers' processes to
# leave, and one still busy then is recorded and left out; then the v2 phase takes the GPUs over (claim --kept), or
# claims NG afresh when gpu_claim.py refuses them
G2=$HELD
if [ -n "$DRY" ]; then say "(dry run) would keep GPUs $HELD (gpu_claim.py keep, holders of $GCHOLD GiB), wait up to 10 minutes for them to have no process but those holders, and take them over (claim --kept)"
else
  gc_keep $GCHOLD
  end=$(( $(date +%s) + 600 )); busy=""
  for x in $HELD; do until idle_gpu $x || [ $(date +%s) -ge $end ]; do nap 5; done; idle_gpu $x || busy="$busy $x"; done
  claim_gpus "$HELD"
  G2=$(for x in $HELD; do case " $busy " in *" $x "*) ;; *) echo $x;; esac; done | tr '\n' ' ')
  [ -n "$busy" ] && say "GPUs still busy 10 minutes after the v3 servers stopped, left out of the v2 phase:$busy"
fi
servers v2 "$G2"
n=$(echo $PORTS | wc -w); urls=""; for p in $PORTS; do urls="$urls --base-url http://127.0.0.1:$p/v1"; done
run2() {  # <part> <concurrency>: two passes, as M1's; returns the last pass's exit status (eval_c (4))
  local part=$1 k=$2 T pass rc
  if [ $part = wealth ]; then T="--tasks $H/tasks_test.jsonl --db $H/C.sqlite"; else T="--tasks $H/tasks_ml_test.jsonl --db none"; fi
  for pass in 1 2; do
    bg $PIDF.$part "$PY -B $O/code_collect/run.py $T --concurrency $k --runs 1 --config H1 --max-steps 100 --keep-results 10 --summary-chars 1000 --provider qwen --model $NAME$urls --out $O/runs_q4b_test/${NAME}_H1_$part >> $O/logs/q4bt_${NAME}_H1_$part.log 2>&1"
    rc=$?; [ -z "$DRY" ] && echo "pass $pass exit $rc" >> $O/logs/q4bt_${NAME}_H1_$part.log
  done
  return $rc
}
if [ $n -gt 0 ]; then
  if [ -n "$DRY" ]; then run2 wealth $((64 * n)); run2 ml $((12 * n))
  else
    run2 wealth $((64 * n)) & W=$!; run2 ml $((12 * n)) & MLP=$!; SUBS="$W $MLP"; wait $W; rw=$?; wait $MLP; rml=$?; SUBS=""
    say "$NAME wealth/ML test: $(cat $O/runs_q4b_test/${NAME}_H1_wealth/results.jsonl 2>/dev/null | wc -l) wealth rows, $(cat $O/runs_q4b_test/${NAME}_H1_ml/results.jsonl 2>/dev/null | wc -l) ml rows; last passes exit $rw and $rml"
    [ $rw = 0 ] || BADRUNS="$BADRUNS ${NAME}_H1_wealth (exit $rw)"; [ $rml = 0 ] || BADRUNS="$BADRUNS ${NAME}_H1_ml (exit $rml)"  # eval_c (4)
  fi
else say "$NAME: no v2 server started"; fi
if [ -n "$DRY" ]; then
  [ -n "$KEEP_AT_END" ] && say "(dry run) would keep GPUs $HELD for the chain's next stage (gpu_claim.py keep, holders of ${KEEP_HOLD_GB:-$GCHOLD} GiB) and list them in $KEEP_AT_END"
else
  stop_srvs "$SLOGS"; SLOGS=""
  if [ -n "$BADRUNS" ]; then gc_keep_for_restart $GCHOLD "bash $0 $ARGS"; HELD=""; say "servers stopped; runs with an exit other than 0:$BADRUNS"  # eval_c (4)
  elif [ -n "$KEEP_AT_END" ] && gc_keep_handover "${KEEP_HOLD_GB:-$GCHOLD}" "$KEEP_AT_END"; then HELD=""; say "servers stopped"  # eval_c (1)
  else gc_release; HELD=""; say "servers stopped, GPUs released"; fi
fi

# 5. comparisons with M1 (and the other reference models) on the same tasks
: > $CMP.raw
section() { echo "== $1" >> $CMP.raw; }
for f in market fsc docfinqa; do
  T=$(file_of $f | cut -d' ' -f1); b=test_hs_${f}_${NAME}_H1; x=""; [ $f = fsc ] && x="--exclude fsc-(T3)Complex_Historical_Investigation_023"
  section "$(basename $T .jsonl) hm5"
  [ -s $MK/runs/$b/results.jsonl ] && $PY -B $ROOT/src/04_evaluation/utils/hm5.py $MK/runs $T test_hs_${f}_M1_H1:$b test_hs_${f}_qwen3.5-4b_H1:$b \
    test_hs_${f}_qwen3.5-4b-base_H1_stop:$b $b:test_hs_${f}_q38_H1 $x >> $CMP.raw 2>&1
done
b=dev_hs_market_${NAME}_H1_x4
section "dev_market_x4 paired_runs"
[ -s $MK/runs/$b/results.jsonl ] && $PY -B $ROOT/src/04_evaluation/utils/paired_runs.py $MK/runs $HS/dev_market.jsonl dev_hs_market_M1_H1_x4 $b --subset $LONG >> $CMP.raw 2>&1
section "dev_market_x4 passk"
for a in dev_hs_market_M1_H1_x4 $b; do [ -s $MK/runs/$a/results.jsonl ] && $PY -B $ROOT/src/04_evaluation/utils/hm5.py $MK/runs $HS/dev_market.jsonl --passk $a --k 4 >> $CMP.raw 2>&1; done
section "wealth_ml hm3"
[ -s $O/runs_q4b_test/${NAME}_H1_wealth/results.jsonl ] && $PY -B $ROOT/src/04_evaluation/utils/hm3.py $O/runs_q4b_test $H/tasks_test.jsonl $H/tasks_ml_test.jsonl \
  M1_H1:${NAME}_H1 A1-base_H1:${NAME}_H1 qwen3.5-4b-base_H1_stop:${NAME}_H1 qwen3.5-4b_H1:${NAME}_H1 >> $CMP.raw 2>&1
python3 - $CMP.raw $CMP.json $CMP.txt "$NAME" "$HF" <<'EOF'
import json, sys
raw, js, txt, name, model = sys.argv[1:6]
out, sec = {"name": name, "model": model}, None
for line in open(raw):
    line = line.rstrip("\n")
    if line.startswith("== "):
        sec = line[3:]; out[sec] = []; continue
    try:
        out[sec].append(json.loads(line))
    except ValueError:
        out[sec].append({"text": line})
json.dump(out, open(js, "w"), indent=1)
lines = [f"{name} ({model})"]
fmt = lambda r: f"{r['diff']:+.1f} [{r['ci95'][0]}, {r['ci95'][1]}], p_one_sided {r['p_one_sided']:.4f}"
for sec, rs in out.items():
    if not isinstance(rs, list):
        continue
    lines.append(f"{sec}:" + ("" if rs else " no rows"))
    for r in rs:
        if "missing" in r:
            lines.append(f"  {r['comparison']}: missing {r['missing']}")
        elif "all" in r and isinstance(r["all"], dict) and "tasks" in r["all"]:  # hm5
            b, a = r["comparison"].split(" - ")
            for key in ("all", "without_excluded"):
                if key in r:
                    x = r[key]
                    lines.append(f"  {key}: {a} {x[a]} -> {b} {x[b]}, {fmt(x)}, {x['tasks']} tasks, {x['dropped_server_error']} dropped")
        elif "passk" in r:
            lines.append(f"  {r['passk']}: pass@1 {r['pass@1']}, pass@{r['k']} {r[f'pass@' + str(r['k'])]}, {r['tasks']} tasks")
        elif "tasks" in r and "diff" in r:  # hm3
            b, a = r["comparison"].split(" - ")
            lines.append(f"  {a} {r[a]} -> {b} {r[b]}, {fmt(r)}, {r['tasks']} tasks, {r['dropped_server_error']} dropped")
        elif "comparison" in r:
            lines.append(f"  {r['comparison']}: {r['tasks_in_both']} tasks in both, {r['missing_in_one_arm']} in one arm only")
        elif "all" in r or "subset" in r:
            x, label = (r["all"], "all") if "all" in r else (r["result"], "subset " + ",".join(r["subset"]))
            if x:
                lines.append(f"  {label}: M1 {x['A']} -> {name} {x['B']}, {fmt(x)}, {x['n']} tasks")
        elif "template" in r and r["result"]:
            x = r["result"]
            lines.append(f"    {r['template']}: {x['A']} -> {x['B']} ({x['diff']:+.1f}, n {x['n']})")
        elif "text" in r:
            lines.append(f"  {r['text']}")
open(txt, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
rm -f $CMP.raw
say "comparisons written: $CMP.json, $CMP.txt"
[ -n "$BADRUNS" ] && { say "eval $NAME failed: runs with an exit other than 0:$BADRUNS (a restart resumes their rows)"; exit 1; }  # eval_c (4)
say "eval $NAME done"
