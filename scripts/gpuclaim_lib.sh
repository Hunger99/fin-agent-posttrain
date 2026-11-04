# GPU claims of eval_ckpt_8gpu.sh and ppo_prep/ppo1.sh through the GPU-claim tool (research/log/2026-11-01-0010.md,
# 11-02 07:5x UTC; the author: "以后凡是所有 RL 相关的训练，去抢卡或者是我们自己跑的时候，哪怕是安全的时候。也默认使用这个站位进程"; 08:3x UTC: if
# our run crashes, hold its GPUs at once and keep them until the fixed code runs again). GC is the installed gpu_claim.py;
# its lock dir and the holders' Python come from the GPU-claim tool's config. free8.sh's jobs (run1b, its guard,
# judge_cal_job.sh) lock GPUs in free8.sh's NFS lock dir LOCKS instead, which gpu_claim.py does not see, so gc_claim
# claims only GPUs without such a lock and then takes that lock on each too; when another job got one first, it lets
# them all go and claims again. The free8.sh locks are dropped whenever the GPUs are released or kept for a restart (a
# kept GPU shows its holder's process, which free8.sh counts as busy).
# The caller defines say and nap and sets GC_LOG, the file that gets gpu_claim.py's messages (and GC_LOG.claimed, the
# GPUs a claim printed). Overridable for a stub test: GC, F8_LOCKS (instead of free8.sh's LOCKS), GPU_CLAIM_CONFIG.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); . "$ROOT/scripts/paths.sh"
source $ROOT/scripts/free8.sh  # LOCKS
LOCKS=${F8_LOCKS:-$LOCKS}
GC=${GC:-$GPU_CLAIM_PY}
GC_HELD=""; GC_F8=""; GC_PID=""  # the GPUs claimed and neither released nor kept; the free8.sh locks taken; a claim waiting
gc() { python3 -B $GC "$@"; }
commas() { echo $* | tr ' ' ','; }
spaces() { echo $(echo "$*" | tr ',' ' '); }
f8_free() { local x out=""; for x in "$@"; do [ -d $LOCKS/gpu$x ] || out="$out $x"; done; echo $out; }  # <GPUs>: those without a free8.sh lock

gc_claim() {  # <N> <hold GiB, empty for gpu_claim.py's default> <candidate GPUs> [kept GPUs]: GC_HELD := the kept GPUs
  # taken over (claim --kept) or, when that is refused or none are given, N candidates without a free8.sh lock claimed
  # (claim -n), each with a holder, for this script ($$); false when gpu_claim.py fails otherwise. A claim -n still
  # waiting after 10 minutes is stopped (it then releases what it locked) and started again with the candidates that
  # have no free8.sh lock by then.
  local n=$1 hold=$2 cands=$(spaces $3) kept=$(spaces $4) c x ok rc t
  : > $GC_LOG.claimed
  while true; do
    if [ -n "$kept" ]; then set -- --kept $(commas $kept)
    else
      until c=$(f8_free $cands); [ $(echo $c | wc -w) -ge $n ]; do nap 5; done
      set -- -n $n --gpus $(commas $c)
    fi
    python3 -B $GC claim "$@" --owner $$ ${hold:+--hold-gb $hold} > $GC_LOG.claimed 2>> $GC_LOG & GC_PID=$!  # not through gc: $! must be the claim
    t=0; while kill -0 $GC_PID 2>/dev/null && { [ -n "$kept" ] || [ $t -lt 600 ]; }; do nap 1; t=$((t + 1)); done
    kill -0 $GC_PID 2>/dev/null && { kill -TERM $GC_PID; t=stopped; }
    wait $GC_PID; rc=$?; GC_PID=""
    [ $t = stopped ] && [ $rc != 0 ] && { : > $GC_LOG.claimed; continue; }  # stopped while waiting; one that ended first counts
    if [ $rc != 0 ]; then
      [ -n "$kept" ] || { say "gpu_claim.py claim exited $rc (see $GC_LOG)"; return 1; }
      say "the kept GPUs $kept were refused (see $GC_LOG): released, claiming $n afresh"
      gc_release_kept $kept; gc_drop_f8; GC_HELD=""; kept=""; continue
    fi
    GC_HELD=$(spaces $(cat $GC_LOG.claimed)); ok=1
    for x in $GC_HELD; do
      case " $GC_F8 " in *" $x "*) continue;; esac
      if mkdir $LOCKS/gpu$x 2>/dev/null; then GC_F8="$GC_F8 $x"; else ok=0; fi
    done
    [ $ok = 1 ] && return 0
    say "GPUs $GC_HELD: another job took a free8.sh lock on one of them first; released, claiming again"
    gc_release; kept=""; nap 5
  done
}
gc_release_kept() {  # <GPUs>: releases those whose lock belongs to this script or to a process that has exited
  local x mine=""
  for x in "$@"; do
    case "$(gc status --gpus $x 2>/dev/null | cut -f4)" in "owner $$ "*|"owner "*" gone,"*) mine="$mine $x";; esac
  done
  [ -n "$mine" ] && gc release $(commas $mine) 2>> $GC_LOG
  return 0
}
gc_keep() {  # <hold GiB or empty>: holders keep GC_HELD, still this script's, from a stage's end until the next stage's
  # processes appear (gpu_claim.py keep); the next stage takes them over with gc_claim <N> <hold> <candidates> "$GC_HELD"
  [ -z "$GC_HELD" ] && return 0
  gc keep $(commas $GC_HELD) ${1:+--hold-gb $1} 2>> $GC_LOG || say "gpu_claim.py keep failed (see $GC_LOG)"
}
gc_keep_for_restart() {  # <hold GiB or empty> <restart command>: after a failure, keeps GC_HELD for the fixed job, whose
  # restart takes them over with KEPT=<GPUs>; their gpu-claim locks stay, so this script's exit does not release them
  local g=$(commas $GC_HELD)
  [ -z "$g" ] && return 0
  if gc keep $g ${1:+--hold-gb $1} 2>> $GC_LOG; then
    say "GPUs $g kept for the fixed job until a process of ours is on them, they are released, or the keep time passes ($(gc status --gpus $g 2>/dev/null | head -n 1 | grep -o 'limit [^ ]*')); restart: KEPT=$g $2; release: python3 $GC release $g"
    GC_HELD=""; : > $GC_LOG.claimed
  else say "gpu_claim.py keep failed (see $GC_LOG): GPUs $g will be released"; fi
  gc_drop_f8
}
gc_release() {  # releases GC_HELD and the GPUs of a claim not yet read, and drops the free8.sh locks taken
  local g=$(commas $GC_HELD $(spaces $(cat $GC_LOG.claimed 2>/dev/null)))
  [ -n "$g" ] && gc release $g 2>> $GC_LOG
  gc_drop_f8; GC_HELD=""; : > $GC_LOG.claimed
}
gc_drop_f8() { local x; for x in $GC_F8; do rmdir $LOCKS/gpu$x 2>/dev/null; done; GC_F8=""; }
gc_stop_claim() {  # stops a waiting claim, which releases what it locked; the GPUs of one that ended first stay for gc_release
  [ -z "$GC_PID" ] && return 0
  kill -TERM $GC_PID 2>/dev/null; wait $GC_PID 2>/dev/null || : > $GC_LOG.claimed; GC_PID=""
}

# GPU processes, by nvidia-smi's index
gpu_pids() {  # <GPU>: the PIDs nvidia-smi shows on it ([N/A] for one it cannot show)
  local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ' | grep "^$1," | cut -d, -f2)
  [ -n "$u" ] && nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | tr -d ' ' | grep "^$u," | cut -d, -f2
}
is_holder() { [ -O /proc/$1 ] && tr '\0' ' ' 2>/dev/null < /proc/$1/cmdline | grep -q 'gpu_holder.py --uuid'; }  # <PID>: a gpu-claim holder of ours
ours_on() {  # <GPUs>: the processes of ours on them other than our holders, such as a stopped job's leftovers
  local x q; for x in "$@"; do for q in $(gpu_pids $x); do is_holder $q || [ "$(ps -o user= -p $q 2>/dev/null | tr -d ' ')" != "$(id -un)" ] || echo $q; done; done
}
foreign_pids() {  # <GPUs>: the PIDs on them of another user, or that ps cannot show
  local x q; for x in "$@"; do for q in $(gpu_pids $x); do [ "$(ps -o user= -p $q 2>/dev/null | tr -d ' ')" = "$(id -un)" ] || echo $q; done; done | sort -u
}
contended() { local a; a=$(foreign_pids "$@"); [ -n "$a" ] || return 1; sleep 1; [ -n "$(comm -12 <(echo "$a") <(foreign_pids "$@"))" ]; }  # <GPUs>: one such PID in two checks a second apart
holder_lines() {  # <GPUs>: each one's latest holding, waiting, freed or not holding line, written with say
  local d x l
  d=$(python3 -B -c 'import getpass, os, sys; sys.path.insert(0, os.path.dirname(sys.argv[1])); import gpu_claim; print(os.path.realpath(gpu_claim.read_config().get("lock_dir", "/tmp/gpu-claim-" + getpass.getuser())))' $GC 2>/dev/null)
  for x in "$@"; do
    l=$(grep -E 's (holding|waiting|freed|not holding) ' $d/gpu$x.holder.log 2>/dev/null | tail -n 1)
    say "GPU $x latest holder line: ${l:-none}"
  done
}
