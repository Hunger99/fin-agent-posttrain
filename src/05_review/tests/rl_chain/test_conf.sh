#!/bin/bash
# Tests of rl_chain_c.sh's read_conf (the config file of grpo2's DYN, LR and MINI, ppo2's PPO_LR, PPO_MINI and
# PPO_CRITIC_LR, and REWORDED_RL), ppo_choice_ok, regrade_ok, hold_after and
# st_done (the stage-done checks that size each hand-over's holders), on files in a work dir; no GPU, lock or process.
# Usage: bash test_conf.sh <work dir>
W=$1; mkdir -p $W; rm -f $W/*.conf
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); . "$ROOT/scripts/paths.sh"
CHAIN_LIB_ONLY=1 source $ROOT/scripts/rl_chain_c.sh
F=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; F=$((F + 1)); }
expect() {  # <name> <file content or @missing> <want: "DYN LR MINI RWRL FROM" or "ERR:<substring>">
  local f=$W/$1.conf got
  if [ "$2" = @missing ]; then f=$W/none.conf; else printf '%b' "$2" > $f; fi
  if read_conf $f; then got="$C_DYN $C_LR $C_MINI $C_RWRL $C_FROM"; else got="ERR:$C_ERR"; fi
  case "$3" in
    ERR:*) case "$got" in ERR:*"${3#ERR:}"*) ok "$1 -> $got";; *) bad "$1: want $3, got $got";; esac;;
    *) [ "$got" = "$3" ] && ok "$1 -> $got" || bad "$1: want '$3', got '$got'";;
  esac
}
expect missing @missing "0 8.7e-7 96 1 no file"
expect empty "" "0 8.7e-7 96 1 no keys"
expect comments "# DYN=1\n\n   # LR=1e-6\n" "0 8.7e-7 96 1 no keys"
expect dyn1 "DYN=1\n" "1 8.7e-7 96 1 DYN"
expect spaces_crlf_comment "  DYN = 1   # the author's decision\r\n" "1 8.7e-7 96 1 DYN"
expect no_final_newline "DYN=1" "1 8.7e-7 96 1 DYN"
expect all_keys "DYN=0\nLR=1e-6\nMINI=48\nREWORDED_RL=0\n" "0 1e-6 48 0 DYN LR MINI REWORDED_RL"
expect lr_decimal "LR=0.00000087\n" "0 0.00000087 96 1 LR"
expect mini24 "MINI=24\n" "0 8.7e-7 24 1 MINI"
expect dyn2 "DYN=2\n" "ERR:DYN must be 0 or 1"
expect dyn_empty "DYN=\n" "ERR:not KEY=VALUE"
expect lr_text "LR=abc\n" "ERR:LR must be a number"
expect lr_zero "LR=0\n" "ERR:LR must be a number"
expect lr_big "LR=1e-4\n" "ERR:LR must be a number"
expect lr_negative "LR=-1e-6\n" "ERR:LR must be a number"
expect mini32 "MINI=32\n" "ERR:MINI must divide 96"
expect mini40 "MINI=40\n" "ERR:MINI must divide 96"
expect mini_zero_lead "MINI=048\n" "ERR:MINI must be a whole number"
expect mini_text "MINI=9x\n" "ERR:MINI must be a whole number"
expect unknown "STEPS=30\n" "ERR:unknown key 'STEPS'"
expect lowercase "dyn=1\n" "ERR:unknown key 'dyn'"
expect twice "DYN=1\nDYN=0\n" "ERR:DYN given twice"
expect no_equals "DYN 1\n" "ERR:not KEY=VALUE"
expect shell_code 'DYN=$(touch '"$W"'/pwned)\n' "ERR:DYN must be 0 or 1"
[ -e $W/pwned ] && bad "a value was executed" || ok "no value is executed"
# ppo2's keys (C_PLR C_PMINI C_PCLR) and the rule that a changed grpo2 LR or MINI needs PPO_LR and PPO_MINI
expect_ppo() {  # <name> <file content> <want: "PLR PMINI PCLR FROM" or "ERR:<substring>">
  local f=$W/$1.conf got; printf '%b' "$2" > $f
  if read_conf $f; then got="$C_PLR $C_PMINI $C_PCLR $C_FROM"; else got="ERR:$C_ERR"; fi
  case "$3" in
    ERR:*) case "$got" in ERR:*"${3#ERR:}"*) ok "$1 -> $got";; *) bad "$1: want $3, got $got";; esac;;
    *) [ "$got" = "$3" ] && ok "$1 -> $got" || bad "$1: want '$3', got '$got'";;
  esac
}
expect_ppo ppo_defaults "DYN=1\n" "8.7e-7 96 8.7e-6 DYN"
expect_ppo ppo_all "PPO_LR=1e-6\nPPO_MINI=48\nPPO_CRITIC_LR=1e-5\n" "1e-6 48 1e-5 PPO_LR PPO_MINI PPO_CRITIC_LR"
expect_ppo ppo_lr_big "PPO_LR=2e-5\n" "ERR:PPO_LR must be a number"
expect_ppo ppo_mini32 "PPO_MINI=32\n" "ERR:PPO_MINI must divide 96"
expect_ppo ppo_mini_lead "PPO_MINI=048\n" "ERR:PPO_MINI must be a whole number"
expect_ppo ppo_critic_big "PPO_CRITIC_LR=1e-3\n" "ERR:PPO_CRITIC_LR must be a number"
expect_ppo ppo_twice "PPO_LR=1e-6\nPPO_LR=1e-6\n" "ERR:PPO_LR given twice"
choice() {  # <name> <file content> <grpo2 LR> <grpo2 MINI> <want ok|stop>
  local f=$W/$1.conf got; printf '%b' "$2" > $f; read_conf $f || { bad "$1: read_conf: $C_ERR"; return; }
  if ppo_choice_ok $3 $4; then got=ok; else got=stop; fi
  [ $got = $5 ] && ok "ppo_choice_ok $1 (grpo2 LR $3 MINI $4) -> $got" || bad "ppo_choice_ok $1: want $5, got $got ($C_ERR)"
}
choice defaults_no_file "" 8.7e-7 96 ok
choice same_written_otherwise "LR=0.00000087\n" 0.00000087 96 ok
choice lr_changed_no_ppo "LR=1e-6\n" 1e-6 96 stop
choice mini_changed_no_ppo "MINI=48\n" 8.7e-7 48 stop
choice lr_changed_only_ppo_lr "LR=1e-6\nPPO_LR=1e-6\n" 1e-6 96 stop
choice lr_changed_ppo_both "LR=1e-6\nPPO_LR=1e-6\nPPO_MINI=96\n" 1e-6 96 ok
choice lr_changed_ppo_kept_ppo1 "LR=1e-6\nPPO_LR=8.7e-7\nPPO_MINI=96\n" 1e-6 96 ok

# hold sizes: in a dry run no stage counts as done
DRY=1
[ "$(hold_after screen)" = "" ] && [ "$(hold_after grpo2)" = "" ] && [ "$(hold_after grpo3)" = "" ] && [ "$(hold_after grpo4)" = 4 ] && [ "$(hold_after eval_grpo1)" = 4 ] \
  && [ "$(hold_after eval_grpo2)" = 4 ] && [ "$(hold_after eval_grpo3)" = 4 ] && [ "$(hold_after eval_grpo4)" = 2 ] && ok "hold_after (nothing done): screen '', grpo2 '', grpo3 '', grpo4 4, evals 4 4 4 2" \
  || bad "hold_after (nothing done): $(for s in screen grpo2 grpo3 grpo4 eval_grpo1 eval_grpo2 eval_grpo3 eval_grpo4; do printf '%s=%s ' $s "$(hold_after $s)"; done)"
# a fake rl/ in which grpo2, grpo3 and grpo4 are done, and grpo1's, grpo2's and grpo3's evals; grpo4's eval not
DRY=""; RL=$W/rl; rm -rf $RL
for r in run1b:24 grpo2:28 grpo3:28 grpo4:28; do d=$RL/${r%%:*}; mkdir -p $d/ckpt; echo ${r#*:} > $d/ckpt/latest_checkpointed_iteration.txt; done
echo "grpo2 exit 0 2026-11-04 00:00:00" > $RL/grpo2.status; echo "grpo3 exit 0 2026-11-04 12:00:00" > $RL/grpo3.status; echo "grpo4 exit 0 2026-11-05 00:00:00" > $RL/grpo4.status
printf 'merged from %s\neval grpo1 done 2026-11-05 01:00:00\n' $RL/run1b/ckpt/global_step_24/actor > $RL/eval_grpo1.status
printf 'merged from %s\neval grpo2 done 2026-11-05 02:00:00\n' $RL/grpo2/ckpt/global_step_28/actor > $RL/eval_grpo2.status
printf 'merged from %s\neval grpo3 done 2026-11-05 02:30:00\n' $RL/grpo3/ckpt/global_step_28/actor > $RL/eval_grpo3.status
printf 'merged from %s\n' $RL/grpo4/ckpt/global_step_28/actor > $RL/eval_grpo4.status
st_done grpo2 && st_done grpo3 && st_done grpo4 && st_done eval_grpo1 && st_done eval_grpo2 && st_done eval_grpo3 && ! st_done eval_grpo4 && ok "st_done reads the fake rl/" || bad "st_done"
[ "$(hold_after screen)" = 4 ] && [ "$(hold_after grpo2)" = 4 ] && [ "$(hold_after eval_grpo1)" = 4 ] && [ "$(hold_after eval_grpo4)" = 2 ] \
  && ok "hold_after skips finished stages: screen -> grpo4's eval 4, eval_grpo1 -> grpo4's eval 4, eval_grpo4 -> judge 2" \
  || bad "hold_after with finished stages: $(for s in screen grpo2 eval_grpo1 eval_grpo4; do printf '%s=%s ' $s "$(hold_after $s)"; done)"
echo "eval grpo4 done 2026-11-05 03:00:00" >> $RL/eval_grpo4.status
[ "$(hold_after screen)" = 2 ] && ok "everything done: screen -> judge 2" || bad "everything done: screen=$(hold_after screen)"
echo 27 > $RL/grpo4/ckpt/latest_checkpointed_iteration.txt
! st_done grpo4 && ! st_done eval_grpo4 && [ "$(hold_after grpo2)" = "" ] && ok "grpo4 at step 27 is not done: grpo2 -> grpo4 (default hold)" || bad "grpo4 at step 27"
# regrade_ok: every run's and comparison's file newer than the marker, and no traceback
STRICT_DIR=$W/sp/strict_grade; d=$STRICT_DIR/g1; mkdir -p $d; M=$W/marker; touch -d "@$(( $(date +%s) - 100 ))" $M
FS="test_hs_market_g1_H1 test_hs_fsc_g1_H1 dev_hs_market_g1_H1_x4 compare_test_market compare_test_finsearchcomp compare_dev_market"
for f in $FS; do touch $d/$f.json; done; echo "== test_market" > $d/regrade.txt
regrade_ok g1 $M && ok "regrade_ok: all six files new" || bad "regrade_ok with all files: $RG_WHY"
touch -d "@$(( $(date +%s) - 200 ))" $d/test_hs_fsc_g1_H1.json
regrade_ok g1 $M && bad "regrade_ok passed with an old fsc file" || ok "regrade_ok refuses an old file: $RG_WHY"
touch $d/test_hs_fsc_g1_H1.json; printf '== test_market\nTraceback (most recent call last):\n' > $d/regrade.txt
regrade_ok g1 $M && bad "regrade_ok passed with a traceback" || ok "regrade_ok refuses a traceback: $RG_WHY"
rm $d/compare_dev_market.json; echo ok > $d/regrade.txt
regrade_ok g1 $M && bad "regrade_ok passed without compare_dev_market.json" || ok "regrade_ok refuses a missing file: $RG_WHY"
echo "FAILURES: $F"; exit $F
