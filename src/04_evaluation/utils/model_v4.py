"""Write a view of a trained run's model directory that vLLM 0.12, which runs on transformers 4.x, reads with the
RoPE base the run was trained with.

LLaMA-Factory saves with transformers 5, whose config.json keeps the RoPE base only inside "rope_parameters".
transformers 4.57 does not know that key, so its Qwen3 config falls back to rope_theta = 10000, and vLLM 0.12
then overwrites rope_parameters["rope_theta"] with that default; the model still writes fluent text but its
answers get worse (research/log/2026-10-28-1937.md). The view links every file of the run except config.json,
whose copy also carries the top-level "rope_theta" taken from "rope_parameters". The run's own directory is not
touched; vLLM and lm-eval are pointed at the view.

Usage: python model_v4.py <run dir> <out dir>
"""
import json
import os
import sys

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
src, dst = os.path.abspath(sys.argv[1]), sys.argv[2]
os.makedirs(dst, exist_ok=True)
for name in os.listdir(src):
    if name in ("config.json", "eval"):
        continue
    link = os.path.join(dst, name)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.join(src, name), link)
with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
rope = cfg.get("rope_parameters") or {}
if "rope_theta" not in cfg and "rope_theta" in rope:
    cfg["rope_theta"] = rope["rope_theta"]
with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print(json.dumps({"src": src, "dst": dst, "rope_theta": cfg.get("rope_theta")}))
