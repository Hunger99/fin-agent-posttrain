#!/bin/bash
# Candidate M1 under composition 甲 and screening scope (b) (research/log/2026-11-01-0010.md, 01:3x UTC): once the
# teacher's hard-tier collection and judge are done and screen_after_tier1.sh has stopped the screening after tier-1,
# (1) filter_sft.py over the four sources with --max-per-task 2 and one FinSearchComp question exempt from
# decontamination (01:4x UTC), checking that its wealth and ML entries are the ones M1_wa was built from; (2) build_agent_sft_v3.py renders the kept market trajectories at 32,768 and adds M1_wa's samples
# unchanged; (3) the gradient accumulation for ~64 samples per step and the card-hour gate; (4) M1 trained on GPUs 2-5
# once all four are free and unlocked. Evaluation is a separate script.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); . "$ROOT/scripts/paths.sh"
EN=$FP_DATA_EN; P=$FP_DATA_D2/harness_v2/pilot
MK=$FP_DATA_D2/market_v1; HS=$MK/hardset_v1; T=$MK/tasks
SRC=$ROOT/src/03_model-training; CF=$ROOT/configs/03_model-training/sft/sft_q35_4b_M1_en.yaml
D=$FP_DATA_D2/agent_sft; C=$D/snapshot_20261030_0906
R=$FP_RUNS/qwen3.5-4b-base-en; TP=$FP_VENVS/train-q35/bin/python
export PYTHONDONTWRITEBYTECODE=1; LOG=$MK/runs/chain_m1b.status; TS=$EN/logs/train_q35_4b_M1_en.status
USED=${USED:-75.7}; CAP=${CAP:-152}; RATE=${RATE:-4300000}  # card-hours used before M1, the cap, token positions per card-hour (pilot P1)
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
free_gpu() { U=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader); [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $U)" = 0 ]; }
echo $$ > $MK/runs/chain_m1b.pid
say "chain_m1b waits for teacher_hard done and the tier-1 screening"
until grep -q "teacher_hard done" $MK/runs/teacher_hard.status && grep -q "screen_after_tier1: chain3d stopped" $MK/runs/screen_chain3.status; do sleep 120; done
grep -q "judge exit 0" $MK/runs/teacher_hard.status || { say "the hard-tier judge did not exit 0: stopped"; exit 1; }
grep -q "screen_base_train done" $MK/runs/screen_chain3.status || { say "the tier-1 screening did not finish: stopped"; exit 1; }
# (1) filter
$TP -B $SRC/code/filter_sft.py \
  --source market:v3:$MK/runs/train_q38_H1:$T/market_tasks_train_shuffled.jsonl \
  --source hard:v3:$MK/runs/train_hard_q38_H1:$HS/train_market.jsonl \
  --source wealth_sampled:v2:$C/wealth/sampled:$C/wealth/round1_tasks.jsonl \
  --source ml_sampled:v2:$C/ml/sampled:$C/ml/round1_tasks.jsonl \
  --screen $MK/runs/screen_base_train --judge $MK/runs/judge_market_train.jsonl,$MK/runs/judge_hard_train.jsonl \
  --heldout $T/market_tasks_dev.jsonl,$T/market_tasks_test.jsonl,$T/market_tasks_hard_dev.jsonl,$T/market_tasks_hard_test.jsonl,$MK/external/finsearchcomp_ext_test.jsonl,$HS/dev_market.jsonl,$HS/test_market.jsonl,$HS/test_finsearchcomp.jsonl \
  --heldout-exempt 'fsc-(T3)Complex_Historical_Investigation_023' \
  --max-per-task 2 --out $D/M1_keep.jsonl > $D/M1_filter.log 2>&1 || { say "filter failed"; exit 1; }
say "M1 filter: $(tail -n 1 $D/M1_filter.log | cut -c1-600)"
a=$(grep -E '"(wealth|ml)_sampled"' $D/M1_keep.jsonl | sort | sha256sum | cut -c1-16); b=$(sort $D/M1_wa_keep.jsonl | sha256sum | cut -c1-16)
[ "$a" = "$b" ] || { say "the wealth/ML entries of M1_keep differ from M1_wa_keep ($a vs $b): stopped"; exit 1; }
# (2) data
$TP -B $SRC/code/build_agent_sft_v3.py --runs market=$MK/runs/train_q38_H1,hard=$MK/runs/train_hard_q38_H1 --keep $D/M1_keep.jsonl \
  --add-packed $D/M1_wa --cutoff 32768 --max-per-task 2 --tokenizer $FP_MODELS_D2/Qwen3.5-4B-Base \
  --out $D/M1 > $D/M1.log 2>&1 || { say "M1 data failed"; exit 1; }
say "M1 data: $(grep -v Saving $D/M1.log | tail -n 1 | cut -c1-600)"
# (3) gradient accumulation and the card-hour gate
read acc est <<EOF
$($TP -c "import json; m=json.load(open('$D/M1.manifest.json')); s=m['trajectories']+m['general']+m['added_from_packed']; n=m['packed_sequences']; print(max(1, round(64 / (4 * s / n))), round(n * 32768 * 2 / $RATE, 1))")
EOF
say "M1 estimate: $est card-hours for 2 epochs (used $USED of $CAP)"
python3 -c "import sys; sys.exit(0 if $est <= $CAP - $USED else 1)" || { say "M1 would exceed the card-hour cap: not trained"; exit 1; }
cur=$(grep -o "^gradient_accumulation_steps: [0-9]*" $CF | grep -o "[0-9]*$")
if [ "$acc" != "$cur" ]; then sed -i "s/^gradient_accumulation_steps: $cur /gradient_accumulation_steps: $acc /" $CF; say "gradient accumulation set to $acc (was $cur)"; else say "gradient accumulation $acc"; fi
grep -q "^gradient_accumulation_steps: $acc " $CF || { say "the config's gradient accumulation is not $acc: stopped"; exit 1; }
# (4) training on GPUs 2-5
all_free() { for g in 2 3 4 5; do free_gpu $g && [ ! -d $P/locks/gpu$g ] || return 1; done; }
until all_free; do sleep 60; done
for g in 2 3 4 5; do mkdir -p $P/locks/gpu$g; done
say "training M1 on GPUs 2-5"
echo "M1 start gpus 2,3,4,5 $(date -u '+%F %T')" >> $TS
PATH=$FP_VENVS/train-q35/bin:$PATH CUDA_VISIBLE_DEVICES=2,3,4,5 DISABLE_VERSION_CHECK=1 NCCL_P2P_DISABLE=1 HF_HOME=$HOME/.cache/huggingface \
  HF_DATASETS_CACHE=${FP_HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets} $FP_VENVS/train-q35/bin/torchrun \
  --nproc_per_node 4 --master_port 29673 $SRC/code/train_q35.py $CF >> $EN/logs/train_q35_4b_M1_en.log 2>&1
rc=$?; echo "M1 exit $rc $(date -u '+%F %T')" >> $TS
for g in 2 3 4 5; do rmdir $P/locks/gpu$g; done
if [ $rc != 0 ] || ! ls $R/M1/*.safetensors > /dev/null 2>&1 || ! grep -q '"rope_theta": 10000000' $R/M1/config.json; then
  say "M1 did not train cleanly (exit $rc) or its config lacks rope_theta 10000000"; exit 1; fi
say "M1 trained; chain_m1b done"
