# Additions to gpuclaim_lib.sh for rl_chain_c.sh and its stage launchers (research/log/2026-11-01-0010.md, 11-02 22:4x and
# 23:2x UTC). gpuclaim_lib.sh is sourced unchanged; this file only adds:
# - gc_keep_handover: a stage that ends well keeps its GPUs for the chain's next stage (gpu_claim.py keep, whose holders
#   wait for this stage's processes to leave and then hold until a process of ours appears, the GPUs are released, or
#   gpu_claim.py's keep time passes), writes them to a file for the chain, and no longer releases them on exit; the next
#   stage takes them over with KEPT=<GPUs> (claim --kept), so no other user can take them between two stages.
# - our_gpus: the GPUs that the project's jobs use now, for the cap of 6 GPUs at once (holders and locks included).
# - gc_left_by: the GPUs a stage left locked when it exited (kept for its restart or the next stage), and those handed to
#   it that it never took over, so that the chain can report, hand on or release them.
# - ancestors_of: the PIDs that started a process, so that a stage's refusal checks can leave out the chain that runs it.
# - margins_ok: the disk and memory margins a stage needs before it starts.
# The caller defines say and nap and sets GC_LOG, as gpuclaim_lib.sh needs.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); . "$ROOT/scripts/paths.sh"
source $ROOT/scripts/gpuclaim_lib.sh  # gc_claim, gc_keep, gc_keep_for_restart, gc_release, gc_drop_f8, gpu_pids, is_holder, LOCKS, GC
GC_LOCK_DIR=$(python3 -B -c 'import getpass, os, sys; sys.path.insert(0, os.path.dirname(sys.argv[1])); import gpu_claim; print(os.path.realpath(gpu_claim.read_config().get("lock_dir", "/tmp/gpu-claim-" + getpass.getuser())))' $GC 2>/dev/null)
GC_LOCK_DIR=${GC_LOCK_DIR:-/tmp/gpu-claim-$(id -un)}

gc_keep_handover() {  # <hold GiB or empty> <file>: keeps GC_HELD for the chain's next stage and writes them to <file>;
  # false (GC_HELD unchanged, so the exit releases them) when gpu_claim.py keep fails
  local g=$(commas $GC_HELD)
  [ -z "$g" ] && return 1
  if gc keep $g ${1:+--hold-gb $1} 2>> $GC_LOG; then
    echo $GC_HELD > $2
    say "GPUs $g kept for the chain's next stage (holders of ${1:-the default} GiB until a process of ours is on them, they are released, or the keep time passes; listed in $2); release: python3 $GC release $g"
    GC_HELD=""; : > $GC_LOG.claimed; gc_drop_f8; return 0
  fi
  say "gpu_claim.py keep failed (see $GC_LOG): GPUs $g will be released"; return 1
}

our_gpus() {  # [GPUs to leave out]: GPUs 0-9 with a process of this user on them (holders included) or a lock of ours on
  # them (the gpu-claim lock dir or free8.sh's NFS lock dir), one per line
  local x q
  for x in 0 1 2 3 4 5 6 7 8 9; do
    case " $* " in *" $x "*) continue;; esac
    if [ -d $GC_LOCK_DIR/gpu$x ] || [ -d $LOCKS/gpu$x ]; then echo $x; continue; fi
    for q in $(gpu_pids $x); do [ "$(ps -o user= -p $q 2>/dev/null | tr -d ' ')" = "$(id -un)" ] && { echo $x; break; }; done
  done
}

gc_left_by() {  # <PIDs> [GPUs]: the GPUs whose gpu-claim lock's owner is gone and either is one of the PIDs (a stage that
  # kept its GPUs for its restart or the next stage, or claimed them and exited) or, for the GPUs given, any process (GPUs
  # handed to a stage that never took them over); "x y ..." on one line
  gc status 2>/dev/null | awk -F'\t' -v p=" $(echo $1) " -v g=" $(echo $2) " '
    $4 ~ /^owner [0-9]+ gone,/ { split($4, a, " "); if (index(p, " " a[2] " ") || index(g, " " $1 " ")) printf "%s ", $1 }' | sed 's/ $//'
}

margins_ok() {  # <root GB> <disk2 GB> <memory GB>: true when the root disk (Ray's temp dir, /tmp), the output disk and
  # the host's MemAvailable each have more than that free, each in whole GiB rounded down, as eval_ckpt_c.sh and
  # screen_c.sh read the root disk; MARGINS := the readings
  local r d m
  r=$(df -P / | awk 'NR == 2 {print int($4 / 1048576)}')
  d=$(df -P "$FP_RUNS" | awk 'NR == 2 {print int($4 / 1048576)}')
  m=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
  MARGINS="root disk ${r:-?} GB free, disk2 ${d:-?} GB free, MemAvailable ${m:-?} GB"
  [ "${r:-0}" -gt $1 ] && [ "${d:-0}" -gt $2 ] && [ "${m:-0}" -gt $3 ]
}

ancestors_of() {  # <PID>: the PIDs that started it, its parent first
  local p=$1 out=""
  while p=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' '); [ -n "$p" ] && [ "$p" -gt 1 ]; do out="$out $p"; done
  echo $out
}
