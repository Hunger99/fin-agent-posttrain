"""A stub OpenAI-compatible client for the v3 tests: replies come from a policy, and token ids from a toy renderer.

The policy is called with the request's messages and returns a reply: {"content", "reasoning", "calls": [(name,
arguments)]}, arguments a dict or a raw string. The stub keeps a deep copy of every request (sent) and answers like
vLLM with return_token_ids and logprobs: the prompt's token ids, the completion's ids, and "token_id:<id>" logprob
entries. Its renderer gives every message a segment of ids; an assistant message renders as exactly the ids sampled
for it when its content and tool calls are unchanged, so the prompt of each turn begins with the previous prompt and
completion unless a message was rewritten. With drop_reasoning, an assistant message that had reasoning and comes
before the last user message renders without it, as the Qwen3.5 chat template does.
"""
import copy
import json
import zlib
from types import SimpleNamespace

GEN = 1  # the generation prompt ("<|im_start|>assistant")
EOS = 2


def tokenize(text):
    """Toy token ids: one per 4 characters, from a checksum, above the special ids."""
    return [3 + zlib.crc32(text[i:i + 4].encode()) % 50000 for i in range(0, len(text), 4)]


class Stub:
    def __init__(self, policy, token_ids=True, drop_reasoning=False, refuse_token_options=False):
        self.policy, self.token_ids, self.drop_reasoning = policy, token_ids, drop_reasoning
        self.refuse_token_options = refuse_token_options
        self.sent, self.requests, self.produced = [], [], []  # produced: (content, calls, had reasoning, ids)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def segment(self, i, m, last_user, n_assistant):
        if m["role"] != "assistant":
            return tokenize(json.dumps(m, sort_keys=True))
        content, calls, had_reasoning, ids = self.produced[n_assistant]
        same = (m.get("content") or "") == content and [(c["function"]["name"], c["function"]["arguments"])
                                                         for c in m.get("tool_calls") or []] == calls
        if same and not (self.drop_reasoning and had_reasoning and i < last_user):
            return [GEN] + ids
        return [GEN] + tokenize(json.dumps(m, sort_keys=True))

    def render(self, messages):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        out, k = [], 0
        for i, m in enumerate(messages):
            out += self.segment(i, m, last_user, k)
            k += m["role"] == "assistant"
        return out + [GEN]

    async def create(self, messages, tools=None, **kw):
        if self.refuse_token_options and (kw.get("logprobs") or "return_token_ids" in (kw.get("extra_body") or {})):
            raise RuntimeError("Error code: 400 - logprobs are not supported by this server")
        self.sent.append(copy.deepcopy(messages))
        self.requests.append(copy.deepcopy(dict(kw, tools=tools)))
        reply = self.policy(messages)
        calls = [(name, args if isinstance(args, str) else json.dumps(args)) for name, args in reply.get("calls", [])]
        n = len(self.produced)
        content = reply.get("content", "")
        ids = tokenize(json.dumps([reply.get("reasoning"), content, calls])) + [EOS]
        self.produced.append((content, calls, bool(reply.get("reasoning")), ids))
        prompt = self.render(messages)
        tcs = [SimpleNamespace(id=f"call-{n}-{j}", type="function", function=SimpleNamespace(name=name, arguments=a))
               for j, (name, a) in enumerate(calls)] or None
        msg = SimpleNamespace(content=content or None, tool_calls=tcs, reasoning=reply.get("reasoning"))
        lp = SimpleNamespace(content=[SimpleNamespace(token=f"token_id:{t}", logprob=-(j % 4) * 0.25)
                                      for j, t in enumerate(ids)])
        want = self.token_ids and kw.get("logprobs")
        choice = SimpleNamespace(message=msg, finish_reason="tool_calls" if tcs else "stop",
                                 token_ids=ids if want else None, logprobs=lp if want else None)
        usage = SimpleNamespace(prompt_tokens=len(prompt), completion_tokens=len(ids),
                                model_dump=lambda: {"prompt_tokens": len(prompt), "completion_tokens": len(ids)})
        return SimpleNamespace(choices=[choice], usage=usage, prompt_token_ids=prompt if want else None)


def tool_results(messages):
    """The tool results since the last assistant message, parsed (the JSON before any appended note)."""
    out = []
    for m in reversed(messages):
        if m["role"] != "tool":
            break
        out.append(json.JSONDecoder().raw_decode(m["content"])[0])
    return out[::-1]


def scripted(steps):
    """A policy that plays the replies in order, one per request; a step may be a function of the messages. After the
    last step it finishes with "done" (and keeps doing so)."""
    def policy(messages):
        k = sum(m["role"] == "assistant" for m in messages)
        step = steps[k] if k < len(steps) else {"calls": [("finish", {"answer": "done"})]}
        return step(messages) if callable(step) else step
    return policy
