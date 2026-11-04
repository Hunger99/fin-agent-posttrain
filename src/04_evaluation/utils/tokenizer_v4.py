"""Write a copy of a trained run's tokenizer that transformers 4.x, which vLLM 0.12 runs on, can load.

LLaMA-Factory saves with transformers 5, whose tokenizer_config.json lists the special tokens under
"extra_special_tokens" as a list; transformers 4.57 expects a dict there and fails. The copy keeps
tokenizer.json byte for byte and only renames that list to "additional_special_tokens", dropping the
5.x-only keys "backend" and "is_local". Its chat template is the training one written by
train_chat_template.py, not the base model's template the run directory keeps. The run's own directory is
not touched; vLLM and lm-eval are pointed at the copy with their tokenizer argument.

--dump writes what a tokenizer makes of fixed probe texts and one chat, so the copy can be checked against
the original under each transformers version (research/log/2026-10-26-1108.md).

Usage: python tokenizer_v4.py <run dir> <out dir> <training template .jinja>
       python tokenizer_v4.py --dump <tokenizer dir> <out.json>
"""
import json
import os
import shutil
import sys

if sys.argv[1] == "--dump":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sys.argv[2])
    probes = ["以下是关于金融的单项选择题，请选出正确答案。\n答案：B", "Question: 16 - 3 - 4 = ?\nAnswer: 9",
              "<|im_start|>user\nhi<|im_end|>\n", "<think>\n推理\n</think>\n\n结论"]
    chat = [{"role": "user", "content": "Write a haiku about rain, all in lowercase."}]
    chats = [chat, [{"role": "system", "content": "只用中文回答。"}] + chat]
    out = {"ids": [tok(p, add_special_tokens=False)["input_ids"] for p in probes],
           "chat": [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats],
           "eos": tok.eos_token, "pad": tok.pad_token}
    with open(sys.argv[3], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    sys.exit(0)

src, dst = sys.argv[1], sys.argv[2]
if len(sys.argv) != 4:
    raise SystemExit(__doc__)
os.makedirs(dst, exist_ok=True)
shutil.copyfile(sys.argv[3], os.path.join(dst, "chat_template.jinja"))
for name in ["tokenizer.json", "special_tokens_map.json"]:
    if os.path.exists(os.path.join(src, name)):
        shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
with open(os.path.join(src, "tokenizer_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
if isinstance(cfg.get("extra_special_tokens"), list):
    cfg["additional_special_tokens"] = cfg.pop("extra_special_tokens")
for key in ["backend", "is_local"]:
    cfg.pop(key, None)
with open(os.path.join(dst, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print(json.dumps({"src": src, "dst": dst, "keys": sorted(cfg)}))
