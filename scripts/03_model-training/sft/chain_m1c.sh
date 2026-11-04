#!/bin/bash
# chain_m1b.sh from its step (2), with M1 trained on 6 GPUs (2-7) instead of 4 (research/log/2026-11-01-0010.md,
# 07:3x UTC): chain_m1b.sh was stopped after its filter step; the gradient accumulation follows the same rule with
# 6 cards (6 x accumulation x samples per packed sequence ~ 64), so each step still sees ~64 samples.
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
echo $$ > $MK/runs/chain_m1c.pid
say "chain_m1c starts from the build (6 GPUs)"
grep -q "M1 filter:" $LOG || { say "no M1 filter line: stopped"; exit 1; }
a=$(grep -E '"(wealth|ml)_sampled"' $D/M1_keep.jsonl | sort | sha256sum | cut -c1-16); b=$(sort $D/M1_wa_keep.jsonl | sha256sum | cut -c1-16)
[ "$a" = "$b" ] || { say "the wealth/ML entries of M1_keep differ from M1_wa_keep ($a vs $b): stopped"; exit 1; }
[ -d $D/M1 ] && mv $D/M1 $D/_trash_M1_partial_20261101_0733
# (2) data
$TP -B $SRC/code/build_agent_sft_v3.py --runs market=$MK/runs/train_q38_H1,hard=$MK/runs/train_hard_q38_H1 --keep $D/M1_keep.jsonl \
  --add-packed $D/M1_wa --cutoff 32768 --max-per-task 2 --tokenizer $FP_MODELS_D2/Qwen3.5-4B-Base \
  --out $D/M1 > $D/M1.log 2>&1 || { say "M1 data failed"; exit 1; }
say "M1 data: $(grep -v Saving $D/M1.log | tail -n 1 | cut -c1-600)"
# (3) gradient accumulation and the card-hour gate
read acc est <<EOF
$($TP -c "import json; m=json.load(open('$D/M1.manifest.json')); s=m['trajectories']+m['general']+m['added_from_packed']; n=m['packed_sequences']; print(max(1, round(64 / (6 * s / n))), round(n * 32768 * 2 / $RATE, 1))")
EOF
say "M1 estimate: $est card-hours for 2 epochs (used $USED of $CAP)"
python3 -c "import sys; sys.exit(0 if $est <= $CAP - $USED else 1)" || { say "M1 would exceed the card-hour cap: not trained"; exit 1; }
cur=$(grep -o "^gradient_accumulation_steps: [0-9]*" $CF | grep -o "[0-9]*$")
if [ "$acc" != "$cur" ]; then sed -i "s/^gradient_accumulation_steps: $cur /gradient_accumulation_steps: $acc /" $CF; say "gradient accumulation set to $acc (was $cur)"; else say "gradient accumulation $acc"; fi
grep -q "^gradient_accumulation_steps: $acc " $CF || { say "the config's gradient accumulation is not $acc: stopped"; exit 1; }
# (4) training on GPUs 2-7
all_free() { for g in 2 3 4 5 6 7; do free_gpu $g && [ ! -d $P/locks/gpu$g ] || return 1; done; }
until all_free; do sleep 60; done
for g in 2 3 4 5 6 7; do mkdir -p $P/locks/gpu$g; done
say "training M1 on GPUs 2-7"
echo "M1 start gpus 2,3,4,5,6,7 $(date -u '+%F %T')" >> $TS
PATH=$FP_VENVS/train-q35/bin:$PATH CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 NCCL_P2P_DISABLE=1 HF_HOME=$HOME/.cache/huggingface \
  HF_DATASETS_CACHE=${FP_HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets} $FP_VENVS/train-q35/bin/torchrun \
  --nproc_per_node 6 --master_port 29673 $SRC/code/train_q35.py $CF >> $EN/logs/train_q35_4b_M1_en.log 2>&1
rc=$?; echo "M1 exit $rc $(date -u '+%F %T')" >> $TS
for g in 2 3 4 5 6 7; do rmdir $P/locks/gpu$g; done
if [ $rc != 0 ] || ! ls $R/M1/*.safetensors > /dev/null 2>&1 || ! grep -q '"rope_theta": 10000000' $R/M1/config.json; then
  say "M1 did not train cleanly (exit $rc) or its config lacks rope_theta 10000000"; exit 1; fi
say "M1 trained; chain_m1b done"
