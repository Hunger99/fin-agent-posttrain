"""A stub client that serves like vLLM with Qwen3.5's tokenizer and a given chat template, for the template checks.

It stands in for `vllm serve ... --chat-template <template> --tool-call-parser qwen3_coder --reasoning-parser qwen3`
(templates/serve_4b_keep_thinking.sh): each request's messages are turned into the template's conversation as vLLM
does (to_conversation: reasoning_content kept, tool-call arguments parsed from JSON into objects), rendered with
add_generation_prompt and enable_thinking and tokenized without special tokens; that is the prompt. The policy's reply
{"reasoning", "content", "calls": [(name, arguments as a dict)]} becomes the completion text a model trained on the
template writes for it (canonical: the template's own rendering of that assistant message, through <|im_end|>), whose
token ids are returned like vLLM's return_token_ids; the reply message is parsed back from that text as the qwen3
parsers do (parse_completion: reasoning before </think>, content, each <tool_call> with its parameters converted by
the tool's JSON Schema). render and parse can be replaced, e.g. by vLLM's own functions.

Needs transformers and the tokenizer at TOKENIZER; the tests skip when either is missing.
"""
import copy
import json
import os
import re
from types import SimpleNamespace

TOKENIZER = os.environ.get("FP_MODELS_D2", "<FP_MODELS_D2 unset>") + "/Qwen3.5-4B-Base"
GEN_HEAD = "<|im_start|>assistant\n<think>\n"
IM_END = "<|im_end|>"


def to_conversation(messages):
    """The messages as vLLM hands them to the chat template: arguments as objects; everything else as sent."""
    out = []
    for m in copy.deepcopy(messages):
        for c in m.get("tool_calls") or []:
            a = c["function"].get("arguments")
            c["function"]["arguments"] = (json.loads(a) if a else {}) if isinstance(a, str) else (a or {})
        out.append(m)
    return out


def canonical_completion(tok, template, tools, reasoning, content, calls):
    """The completion text a model writes for this reply under the template: the template's rendering of the
    assistant message after its "<|im_start|>assistant\\n<think>\\n", through <|im_end|>."""
    msg = {"role": "assistant", "content": content or "", "reasoning_content": reasoning or ""}
    if calls:
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}}
                             for i, (n, a) in enumerate(calls)]
    text = tok.apply_chat_template([{"role": "user", "content": "x"}, msg], tools=tools, chat_template=template,
                                   tokenize=False)
    i = text.rindex(GEN_HEAD) + len(GEN_HEAD)
    j = text.rindex(IM_END) + len(IM_END)
    return text[i:j]


def schema_value(raw, prop):
    """A parameter's text as a value of its schema type (strings stay text; others are read as JSON when they can)."""
    types = prop.get("type", "string")
    types = types if isinstance(types, list) else [types]
    if types == ["string"]:
        return raw
    if "boolean" in types and raw in ("True", "False", "true", "false"):
        return raw.lower() == "true"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>]+)>\n?(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_completion(text, tools):
    """(reasoning, content, [(name, arguments JSON)]) of a completion without its <|im_end|>, as the qwen3 reasoning
    parser and the qwen3_coder tool parser split it."""
    reasoning, _, rest = text.partition("</think>") if "</think>" in text else ("", "", text)
    props = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools or []}
    calls = []
    for m in TOOL_CALL.finditer(rest):
        name = m[1].strip()
        args = {k: schema_value(v, props.get(name, {}).get(k, {})) for k, v in PARAM.findall(m[2])}
        calls.append((name, json.dumps(args, ensure_ascii=False)))
    content = rest[:rest.index("<tool_call>")] if "<tool_call>" in rest else rest
    return reasoning, content.strip(), calls


class QwenStub:
    """policy(messages) -> reply; completion(reasoning, content, calls) may override the canonical text (to test
    replies a model writes differently from the template)."""

    def __init__(self, policy, tok, template, render=None, parse=None, completion=None):
        self.policy, self.tok, self.template = policy, tok, template
        self.render = render or self.render_hf
        self.parse = parse or parse_completion
        self.completion = completion
        self.prompts, self.completions, self.texts = [], [], []  # token ids per request; completion texts
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def render_hf(self, messages, tools):
        return self.tok.apply_chat_template(to_conversation(messages), tools=tools, chat_template=self.template,
                                            tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                            preserve_thinking=True)

    def ids(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    async def create(self, messages, tools=None, **kw):
        prompt = self.ids(self.render(messages, tools))
        reply = self.policy(messages)
        calls = [(n, a if isinstance(a, dict) else json.loads(a)) for n, a in reply.get("calls", [])]
        text = (self.completion or (lambda *x: canonical_completion(self.tok, self.template, tools, *x)))(
            reply.get("reasoning"), reply.get("content"), calls)
        ids = self.ids(text)
        reasoning, content, parsed = self.parse(text[:-len(IM_END)] if text.endswith(IM_END) else text, tools)
        self.prompts.append(prompt)
        self.completions.append(ids)
        self.texts.append(text)
        n = len(self.completions)
        tcs = [SimpleNamespace(id=f"chatcmpl-tool-{n}-{j}", type="function",
                               function=SimpleNamespace(name=name, arguments=a)) for j, (name, a) in enumerate(parsed)]
        msg = SimpleNamespace(content=content or None, tool_calls=tcs or None, reasoning=reasoning or None)
        want = bool(kw.get("logprobs"))
        lp = SimpleNamespace(content=[SimpleNamespace(token=f"token_id:{t}", logprob=-0.5) for t in ids])
        choice = SimpleNamespace(message=msg, finish_reason="tool_calls" if tcs else "stop",
                                 token_ids=ids if want else None, logprobs=lp if want else None)
        usage = SimpleNamespace(prompt_tokens=len(prompt), completion_tokens=len(ids),
                                model_dump=lambda: {"prompt_tokens": len(prompt), "completion_tokens": len(ids)})
        return SimpleNamespace(choices=[choice], usage=usage, prompt_token_ids=prompt if want else None)


def training_encoder(path, tok):
    """rl_data.py's encode() and norm() (the training-side encoder, copied from build_agent_sft.py), taken from the
    source by name, since rl_data.py parses its command line on import; bound to this tokenizer as render() binds
    them. Returns (encode(messages, tools) -> (ids, labels), norm(message) -> message)."""
    import ast
    with open(path, encoding="utf-8") as f:
        src = f.read()
    render = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "render")
    fns = [n for n in render.body if isinstance(n, ast.FunctionDef) and n.name in ("encode", "norm")]
    scope = {"tok": tok, "json": json, "IGNORE": -100, "im_start": tok.convert_tokens_to_ids("<|im_start|>"),
             "im_end": tok.convert_tokens_to_ids("<|im_end|>"),
             "head": tok("assistant\n", add_special_tokens=False).input_ids}
    exec(compile(ast.Module(body=fns, type_ignores=[]), path, "exec"), scope)
    return scope["encode"], scope["norm"]
