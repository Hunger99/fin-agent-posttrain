"""Write the chat template the SFT runs were trained with, as Jinja, for evaluation with a chat template.

LLaMA-Factory's "qwen" template (docs/history/02_model-training.md section 2) puts its default system prompt, "You
are Qwen, created by Alibaba Cloud. You are a helpful assistant.", before every sample that has none, but the
run directory keeps the base model's own Qwen3 template, which adds no system prompt and handles <think>
differently. Evaluating with that saved template would change the input format, so this writes the training
template instead and checks that it renders a prompt to exactly the token ids LLaMA-Factory trains on.
Run it in the LLaMA-Factory environment.

Usage: python train_chat_template.py <run dir> <out.jinja> [--template qwen]
"""
import argparse

from llamafactory.data.template import TEMPLATES
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--template", default="qwen")
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.run)
template = TEMPLATES[args.template]
jinja = template._get_jinja_template(tok)
tok.chat_template = jinja
turn = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
for system in (None, "只用中文回答。"):  # LLaMA-Factory takes the system prompt apart from the messages
    prompt, _ = template.encode_oneturn(tok, turn, system=system)
    chat = ([{"role": "system", "content": system}] if system else []) + turn[:1]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    if tok(text, add_special_tokens=False)["input_ids"] != prompt:
        raise SystemExit("the Jinja template does not reproduce the training prompt")
with open(args.out, "w", encoding="utf-8") as f:
    f.write(jinja)
print(f"wrote {args.out}; it reproduces the training prompt with and without a system message")
