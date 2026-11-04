#!/bin/bash
# General-capability check for the agent models (research/log/2026-11-01-0010.md, 11-02 22:3x UTC, patch 5): M1 and later
# the RL checkpoints against their start model Qwen3.5-4B-Base, whose scores were measured by outputs/archive/legacy/scratchpad/q35/eval_q35_en_v2.sh
# (base_eval: MMLU 72.8, GSM8K 80.5). MMLU and GSM8K repeat that protocol exactly, so the rows compare with the start
# model: MMLU 5-shot without a template, one item at a time (mc_eval.py --max-tokens 1, the 13,018 items without the four
# economics/accounting subjects); GSM8K 5-shot without a template, greedy, lm-evaluation-harness. IFEval differs from
# eval_q35_en_v2.sh, whose sft kind renders LLaMA-Factory's qwen template: M1 was trained with Qwen3.5's own template and
# thinks before answering, so IFEval uses the model's own chat template with thinking on, and lm-eval scores only the text
# after </think> (think_end_token) with room for the thinking (max_gen_toks 8192); the start model cannot chat, so the
# IFEval comparison is with the official Qwen3.5-4B under the same protocol (ifeval_only).
# GPUs: gpu-claim's run takes N free GPUs (3, or 1 for ifeval_only) and holds them for the vLLM processes (hold below
# 0.4 x 80 GB, since vLLM wants 60% free); it refuses to start while one of our RL runs or the chain is alive, since the
# project may use 6 GPUs at once and RL takes all 6.
# Usage: eval_general.sh <name> <model dir> <out dir> [all|ifeval_only]     DRY=1 writes <out dir>/jobs.sh, prints it and exits.
NAME=$1; M=$2; O=$3; WHAT=${4:-all}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
EN=$FP_DATA_EN; E=$ROOT/src/04_evaluation
RL=$FP_RUNS/rl
VP=$FP_VENVS/vllm-0.32/bin/python; TP=$FP_VENVS/train-q35/bin/python; FP=$FP_VENVS/fin-posttrain/bin/python
GC="python3 $GPU_CLAIM_PY"
export HF_HOME=$HOME/.cache/huggingface PYTHONDONTWRITEBYTECODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$FP_VENVS/vllm-0.32/bin:$PATH
Q=$EN/logs/eval_general_${NAME}.status
say() { echo "$* $(date -u '+%F %T')" | tee -a $Q; }
ls $M/config.json $M/tokenizer.json >/dev/null 2>&1 && ls $M/*.safetensors >/dev/null 2>&1 || { say "$NAME model files missing"; exit 1; }
LM="$FP_VENVS/vllm-0.32/bin/lm_eval --model vllm --batch_size auto --log_samples"
MA="pretrained=$M,tokenizer=$M,dtype=bfloat16,gpu_memory_utilization=0.6"
MMLU="$TP -B $E/utils/mc_eval.py --model $M --items $EN/processed/eval/mmlu_test.jsonl --shots $EN/processed/eval/mmlu_dev.jsonl --task multiple_choice --lang en --exclude econometrics,high_school_macroeconomics,high_school_microeconomics,professional_accounting --max-tokens 1 --out $O/mmlu.jsonl"
GSM="$LM --model_args $MA,max_model_len=4096 --tasks gsm8k --num_fewshot 5 --output_path $O/gsm8k"
IFE="$LM --model_args $MA,max_model_len=16384,enable_thinking=True,think_end_token=</think> --tasks ifeval --apply_chat_template --gen_kwargs max_gen_toks=8192 --output_path $O/ifeval"
if [ $WHAT = ifeval_only ]; then N=1; JOBS=("ifeval|$IFE"); else N=3; JOBS=("mmlu|$MMLU" "gsm8k|$GSM" "ifeval|$IFE"); fi
[ -z "$DRY" ] && for p in $(cat $RL/*.pid 2>/dev/null); do kill -0 $p 2>/dev/null && { say "$NAME refused: our RL job $p is alive"; exit 1; }; done
[ -z "$DRY" ] && pgrep -u "$(id -un)" -f "rl_chain[_b]*\.sh|main_ppo" >/dev/null && { say "$NAME refused: an RL chain or trainer is alive"; exit 1; }
mkdir -p $O
# one job per claimed GPU, each in its own subshell; the claimed GPUs come in CUDA_VISIBLE_DEVICES
cat > $O/jobs.sh <<EOS
#!/bin/bash
IFS=, read -ra G <<< "\$CUDA_VISIBLE_DEVICES"; i=0
$(for j in "${JOBS[@]}"; do t=${j%%|*}; c=${j#*|}; echo "( echo \"$NAME $t start gpu \${G[\$i]} \$(date -u '+%F %T')\" >> $Q; CUDA_VISIBLE_DEVICES=\${G[\$i]} sh -c '$c' > $EN/logs/eval_general_${NAME}_$t.log 2>&1; echo \"$NAME $t exit \$? \$(date -u '+%F %T')\" >> $Q ) & i=\$((i+1))"; done)
wait
EOS
if [ -n "$DRY" ]; then echo "N=$N"; cat $O/jobs.sh; exit 0; fi
say "$NAME waits for $N free GPUs ($WHAT)"
$GC run -n $N --gpus 0,1,2,3,4,5,6,7,8,9 --hold-gb 24 --max-hold-s 1800 -- bash $O/jobs.sh >> $O/gpu_claim.log 2>&1
say "$NAME jobs exit $?"
for t in gsm8k ifeval; do f=$(ls $O/$t/*/samples_$t*.jsonl 2>/dev/null | head -n 1); [ -n "$f" ] && $FP -B $E/utils/lmeval_items.py $f $O/$t.jsonl >> $Q 2>&1; done
say "eval_general_${NAME} done"
