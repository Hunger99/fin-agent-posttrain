#!/bin/bash
# Judge calibration on exactly 2 GPUs claimed with gpu-claim, for rl_chain_c.sh's final stage (research/log/2026-11-01-0010.md,
# 11-02 23:1x UTC: judge_cal_job.sh was stopped because its start rule counts free GPUs, not the project's cap of 6, so
# it could make our total 8 beside a 6-GPU stage; the chain runs this launcher beside the 3-GPU general evaluation).
# The server and the judge commands are judge_cal_job.sh's: serve_qwen38.sh (Qwen3.8-27B, TP 2, port 18013) on the
# pair, then judge_calibration.py run (concurrency 32; it appends its verdicts to judge_cal_verdicts.jsonl and resumes
# past the ids already there), up to 3 starts while items lack a verdict, then judge_calibration.py score with the same
# --earlier files, writing judge_cal_report.{json,md}. Only the GPU rule differs: no yielding; GPUs through
# gpuclaim_lib.sh's gc_claim (KEPT=<2 GPUs> takes over GPUs kept for this stage, else 2 of 0-9 without a free8.sh lock
# are claimed), each with a holder of GCHOLD GiB (default 2) until the server appears: serve_qwen38.sh's vLLM needs 0.90
# of the GPU's 48 GiB free at start (43.2 GiB), which leaves room for a holder of at most about 4 GiB.
# - Refuses to start while judge_cal_job.sh or another judge_c.sh runs (one writer of the verdict file).
# - A server that does not answer within 30 minutes, or dies, fails the job: once, when another user's process is on its
#   GPUs in two checks a second apart, they are released, 2 claimed afresh and the server started again; otherwise the
#   GPUs are kept for the fixed job (status: restart and release commands) and the script exits 1. Every other end, the
#   scorer included, stops the server and releases the GPUs; items still without a verdict after 3 starts are scored
#   without, as judge_cal_job.sh does, and counted in the status file.
# - The server stops by its process group (TERM, KILL after 20 s), then whatever of ours is left in its session.
# Usage: [KEPT=<GPUs>] [GCHOLD=2] setsid nohup bash judge_c.sh < /dev/null > <log> 2>&1 &   Stop: kill <pid> (not -9).
#        Status judge_cal/judge_c.status ("judge_c done" at the end). DRY=1 prints the commands and starts, claims and
#        signals nothing (status under DRYDIR, default $DRY_DIR/).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
J=$CHAIN_DIR/judge_cal; OUT=$FP_DATA_D2/judge_cal
MK=$FP_DATA_D2/market_v1; SRC=$ROOT/src/03_model-training
PY=$FP_VENVS/vllm-0.12/bin/python; SERVE=$ROOT/scripts/serve_qwen38.sh; PORT=18013; NAP=5
SET=$OUT/judge_cal_set.jsonl; VER=$OUT/judge_cal_verdicts.jsonl; GCHOLD=${GCHOLD:-2}
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "$DRY" ]; then SD=${DRYDIR:-$DRY_DIR}; mkdir -p $SD; else SD=$J; fi
ST=$SD/judge_c.status; SL=$SD/serve_judge_c.log; RLOG=$SD/judge_c_run.log; PIDF=$SD/judge_c.pid; GC_LOG=$SD/judge_c.gpuclaim.log
say() { echo "$1 $(date -u '+%F %T')" >> $ST; [ -n "$DRY" ] && echo "# $1"; return 0; }
nap() { sleep $1 & wait $!; }
source $ROOT/scripts/gpuclaim_c_lib.sh  # gc_claim, gc_keep_for_restart, gc_release, contended, holder_lines (gpuclaim_lib.sh)
KEPT=$(spaces $KEPT)
for x in $KEPT; do case $x in [0-9]) ;; *) echo "KEPT: not a GPU of this host (0-9): $x" >&2; exit 2;; esac; done
[ -z "$KEPT" ] || [ $(echo $KEPT | wc -w) = 2 ] || { echo "KEPT: 2 GPUs, not $(echo $KEPT | wc -w)" >&2; exit 2; }
running() { [ -f $1 ] && kill -0 "$(cat $1)" 2>/dev/null && tr '\0' ' ' < /proc/$(cat $1)/cmdline 2>/dev/null | grep -q "$2"; }  # <pid file> <script name>
if [ -z "$DRY" ]; then
  running $J/judge_cal.pid judge_cal_job && { echo "judge_cal_job.sh runs (pid $(cat $J/judge_cal.pid)): not started" >&2; exit 2; }
  running $PIDF judge_c.sh && { echo "another judge_c.sh runs (pid $(cat $PIDF)): not started" >&2; exit 2; }
fi
[ -f $SET ] || { echo "no calibration set $SET" >&2; exit 2; }

SPID=""; RPID=""
stop_runner() { [ -n "$RPID" ] || return 0; kill -TERM $RPID 2>/dev/null
  for i in $(seq 10); do kill -0 $RPID 2>/dev/null || break; sleep 1; done; kill -KILL $RPID 2>/dev/null; RPID=""; }
stop_server() {  # judge_cal_job.sh's: TERM to the server's process group, KILL after 20 s, then our leftovers in its session
  [ -n "$SPID" ] || return 0; kill -TERM -- -$SPID 2>/dev/null
  for i in $(seq 20); do kill -0 -- -$SPID 2>/dev/null || break; sleep 1; done
  kill -KILL -- -$SPID 2>/dev/null
  for q in $(ps -eo pid=,sid=,user= | awk -v s=$SPID -v u=$(id -un) '$2 == s && $3 == u {print $1}'); do kill -KILL $q 2>/dev/null; done
  SPID=""; }
cleanup() {
  trap - EXIT INT TERM HUP
  [ -n "$DRY" ] && return
  stop_runner; stop_server; gc_stop_claim; [ -n "$GC_HELD" ] && say "GPUs released: $GC_HELD"; gc_release
  [ "$(cat $PIDF 2>/dev/null)" = $$ ] && rm -f $PIDF
}
trap cleanup EXIT; trap 'say "judge_c stopped by a signal"; exit 143' INT TERM HUP
[ -n "$DRY" ] && : > $ST
[ -n "$DRY" ] || echo $$ > $PIDF
say "judge_c starts (pid $$): set $(wc -l < $SET) items, verdicts $VER ($( [ -f $VER ] && wc -l < $VER || echo 0) so far); 2 GPUs, holders of $GCHOLD GiB${KEPT:+, taking over the kept GPUs $KEPT}"

claim() {  # [kept GPUs]: PAIR := the two GPUs, "a,b"
  if [ -n "$DRY" ]; then PAIR="G1,G2"; say "(dry run) would claim 2 of GPUs 0-9 without a free8.sh lock (now: $(f8_free 0 1 2 3 4 5 6 7 8 9)), holders of $GCHOLD GiB${1:+, taking over the kept GPUs $1}; $PAIR stand for them"; return 0; fi
  gc_claim 2 $GCHOLD "0 1 2 3 4 5 6 7 8 9" "$1" || exit 1
  PAIR=$(commas $GC_HELD); say "GPUs claimed: $GC_HELD"
}
serve() {  # true when the server on PAIR answers within 30 minutes
  local t0
  if [ -n "$DRY" ]; then echo "( setsid sh $SERVE $PAIR $PORT $SL < /dev/null > /dev/null 2>&1 & )"; return 0; fi
  [ -f $SL ] && mv $SL $SL.$(date +%s)
  ( setsid sh $SERVE $PAIR $PORT $SL < /dev/null > /dev/null 2>&1 & )
  for i in $(seq 60); do SPID=$(head -1 $SL 2>/dev/null | grep -o 'pid [0-9]*' | cut -d' ' -f2); [ -n "$SPID" ] && break; nap 0.5; done
  [ -n "$SPID" ] || { say "server log $SL has no pid line"; return 1; }
  say "server on GPUs $PAIR port $PORT, pid/pgid $SPID; waiting for /v1/models"
  t0=$(date +%s)
  until curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q qwen3.8-27b; do
    kill -0 $SPID 2>/dev/null || { say "server exited during start (see $SL)"; stop_server; return 1; }
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { say "server not up after 30 min"; stop_server; return 1; }
    nap $NAP
  done
  say "server up after $(( $(date +%s) - t0 ))s"; holder_lines $GC_HELD
}
fail() { say "judge_c failed: $1"; stop_runner; stop_server; gc_keep_for_restart $GCHOLD "bash $0"; exit 1; }

claim "$KEPT"
if ! serve; then
  if contended $GC_HELD; then say "another user's process is on GPUs $PAIR: released, claiming 2 afresh"; gc_release; claim; serve || fail "the server did not start"
  else fail "the server did not start"; fi
fi
missing() { CUDA_VISIBLE_DEVICES= $PY -B -c "import json,sys; s={json.loads(l)['id'] for l in open(sys.argv[1])}; v={json.loads(l)['id'] for l in open(sys.argv[2])} if __import__('os').path.exists(sys.argv[2]) else set(); print(len(s - v))" $SET $VER 2>/dev/null; }
RUNCMD="CUDA_VISIBLE_DEVICES= $PY -B $SRC/utils/judge_calibration.py run --set $SET --out $VER --base-url http://127.0.0.1:$PORT/v1 --model qwen3.8-27b --concurrency 32"
SCORE="CUDA_VISIBLE_DEVICES= $PY -B $SRC/utils/judge_calibration.py score --set $SET --verdicts $VER --out $OUT/judge_cal_report.json --earlier $MK/runs/judge_market_train.jsonl,$MK/runs/judge_hard_train.jsonl"
if [ -n "$DRY" ]; then
  echo "$RUNCMD >> $RLOG 2>&1 &   (up to 3 starts while items lack a verdict; now $(missing) of $(wc -l < $SET) lack one)"
  echo "$SCORE > $OUT/judge_cal_report.md 2>> $RLOG"
  say "judge_c done"; exit 0
fi
for start in 1 2 3; do
  eval "$RUNCMD >> $RLOG 2>&1 &"; RPID=$!
  say "judging (start $start, runner pid $RPID, $( [ -f $VER ] && wc -l < $VER || echo 0) verdicts so far)"
  while kill -0 $RPID 2>/dev/null; do kill -0 $SPID 2>/dev/null || fail "server died during the run (see $SL)"; nap $NAP; done
  wait $RPID; rc=$?; RPID=""
  n=$(missing); say "runner exit $rc, ${n:-?} items without a verdict"
  [ "$n" = 0 ] && break
done
[ "$n" = 0 ] || say "$n items still without a verdict after 3 starts (see $RLOG); scored without them, as judge_cal_job.sh does"
eval "$SCORE > $OUT/judge_cal_report.md 2>> $RLOG"; rc=$?
say "scorer exit $rc: $OUT/judge_cal_report.md"
stop_server; say "GPUs released: $GC_HELD"; gc_release
[ $rc = 0 ] || { say "judge_c failed: scorer exit $rc"; exit 1; }
say "judge_c done"
