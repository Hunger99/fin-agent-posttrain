# GPU helpers for taking any N free GPUs of the host's 10 (research/log/2026-11-01-0010.md, 11-02 07:4x and 07:5x UTC; the
# author: "改为十个 GPU，只要有八个空，就立刻启动，占住那八个", then "用机械脚本快速刷新检测，只要看到八张卡是空的，就立刻启动，然后抢进去").
# Six GPUs since 11-02 08:3x UTC (the author: "你把用八卡的删掉，后面都改成六卡，PPO 也是"): every later job takes 6 with
# grabn 6 (eval_ckpt_8gpu.sh and ppo_prep/ppo1.sh keep their file names; grab8 is gone, its scripts are in old_8gpu/).
# A GPU is free when no process of any user is on it and no lock of ours holds it. grabn N checks every second (two
# nvidia-smi calls a round), locks the first N free GPUs as soon as there are N, checks once more that none got a process,
# starts gpu_holder.py on each unless HOLD=0 (it holds HOLD_GB until a process of ours appears on that GPU, then frees it;
# at most 10 min), and prints the N GPUs; otherwise it frees what it took and goes on checking.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); . "$ROOT/scripts/paths.sh"
LOCKS=$FP_DATA_D2/harness_v2/pilot/locks
HOLDER=$ROOT/src/00_shared/utils/gpu_holder.py
HPY=$FP_VENVS/verl/bin/python
busy_uuids() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | sort -u; }
free_list() {  # [busy uuids]: the free GPUs
  local b i u out=""; b=${1-$(busy_uuids)}
  while IFS=, read i u; do i=$(echo $i); u=$(echo $u)
    [ -n "$b" ] && echo "$b" | grep -qx "$u" && continue
    [ -d $LOCKS/gpu$i ] && continue
    out="$out $i"
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  echo $out
}
grabn() {  # <N>: the N GPUs taken (see the header)
  local want=$1 pick got x ok n
  case "$want" in [1-9]|10) ;; *) echo "grabn: N must be 1-10, not '$want'" >&2; return 2;; esac
  while true; do
    set -- $(free_list)
    if [ $# -ge $want ]; then
      pick=$(echo "$@" | tr ' ' '\n' | head -n $want | tr '\n' ' '); pick=$(echo $pick); got=""; ok=1
      for x in $pick; do if mkdir $LOCKS/gpu$x 2>/dev/null; then got="$got $x"; else ok=0; break; fi; done
      if [ $ok = 1 ]; then n=$(rmdir_free_check $pick); [ "$n" = $want ] || ok=0; fi
      if [ $ok = 1 ]; then
        [ "${HOLD:-1}" = 1 ] && for x in $pick; do
          ( CUDA_VISIBLE_DEVICES=$x HOLD_GB=${HOLD_GB:-36} MAX_S=600 setsid $HPY -B $HOLDER $x < /dev/null > /dev/null 2>&1 & ); done
        echo $pick; return 0
      fi
      for x in $got; do rmdir $LOCKS/gpu$x 2>/dev/null; done
    fi
    sleep 1
  done
}
rmdir_free_check() {  # <gpus...>: how many of them still have no process (the locks are ours now, so only processes count)
  local b x u k=0; b=$(busy_uuids)
  while IFS=, read x u; do x=$(echo $x); u=$(echo $u)
    case " $* " in *" $x "*) { [ -n "$b" ] && echo "$b" | grep -qx "$u"; } || k=$((k + 1));; esac
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  echo $k
}
others_on() {  # <gpus...>: true when a process of a user other than ours is on one of them
  local x u q
  for x in "$@"; do u=$(nvidia-smi -i $x --query-gpu=uuid --format=csv,noheader | tr -d ' ')
    for q in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | tr -d ' ' | grep "^$u," | cut -d, -f2); do
      [ "$(ps -o user= -p $q 2>/dev/null | tr -d ' ')" != "$(id -un)" ] && return 0; done; done
  return 1
}
