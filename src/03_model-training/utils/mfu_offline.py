"""Offline actor MFU for run1b: verl's own qwen3_5 FLOPs estimate over each step's episode sequence lengths, divided by
the update_actor time, 6 GPUs and the RTX A6000's dense BF16 tensor peak (154.8 TFLOPS, NVIDIA datasheet), because
verl's _DEVICE_FLOPS has no A6000 entry and logs perf/mfu/actor = 0."""
import json, sys, glob, collections
from transformers import AutoConfig
from verl.utils import flops_counter as fc
cfg = AutoConfig.from_pretrained(sys.argv[1])
E, M = sys.argv[2], sys.argv[3]
per_step = collections.defaultdict(list)
for f in sorted(glob.glob(E + "/*.jsonl")):
    step, n = 1, 0
    for l in open(f):
        r = json.loads(l)
        if "-dev-" in r["task_id"]:
            continue
        per_step[step].append(r["prompt_tokens"] + r["response_tokens"]); n += 1
        if n == 96:
            step, n = step + 1, 0
metrics = {json.loads(l)["step"]: json.loads(l)["data"] for l in open(M)}
print("device peak table has A6000:", any("A6000" in k for k in fc._DEVICE_FLOPS))
for s in sorted(metrics):
    lens = per_step[s]
    d = metrics[s]
    t = d["timing_s/update_actor"]
    tflops = fc.FlopsCounter(cfg).estimate_flops(lens, t)[0]  # achieved TFLOPS summed over GPUs (fwd+bwd, 6N rule)
    print(s, "tokens", sum(lens), "vs perf/total_num_tokens", int(d["perf/total_num_tokens"]),
          "update_actor_s", round(t), "TFLOPS/GPU", round(tflops / 6, 1), "MFU", round(tflops / 6 / 154.8, 3))
