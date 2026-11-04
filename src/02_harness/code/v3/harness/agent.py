"""The agent loop and the five harness parts, per docs/03_harness.md sections 4-5.

Every configuration shares the system prompt (role, business rules), the tool interface (native function calling
described by JSON Schema, structured JSON results, paged lists, search returning a list with details fetched
separately) and the loop: at most MAX_STEPS model calls per task, a redo after a correction counting as a step. Three
parts of the full harness H1 can be switched off one at a time, and context management only comes with H1:

- context management: tool results older than the last KEEP_RESULTS are replaced by a short summary (errors are kept
  whole), and the model keeps a checklist with set_checklist, restated after every tool result;
- constraints: before a write, the arguments are checked against the schema and the business rules 1-4
  and 7 (env/core.check); a write that breaks a rule is rejected without being executed;
- verification: when the model calls finish, every number in the answer must appear in a tool result or the request,
  every checklist item must be done, and each trade made in the episode must satisfy the rules on the final state;
  a failed check rejects the finish;
- correction: a rejected finish comes back with the failing checks, at most MAX_REDOS times (without correction,
  only a bare "verification failed", once); a rule rejection comes with its reason and the largest allowed amount;
  tool errors come with an example of a valid call; the third identical call gets a hint to change approach.

H0 is the shared parts alone. A rejected write and an error are returned to the model in every configuration, since
the environment answers every call. A model reply that the server cannot parse into a message is sampled again with
another seed, at most RESAMPLES times, the last time with a tool call required, in every configuration alike.

Version 2 (research/log/2026-10-29-0836.md), after the tool diagnosis of research/log/2026-10-29-0743.md, changes only
what every configuration shares, so the harness parts are version 1's:
- rule 2 is worded as it is checked: the subscribed product, not every product, stays at most 30% of total assets;
- a reply without a tool call whose whole text is one tool's arguments as a JSON object (189 dev steps wrote finish's
  arguments as text) is taken as that call, and the message is rewritten as the call;
- get_fund takes up to 50 ids, and verification reads ids and dates written with any dash, as the grader does;
- PROVIDER "qwen" (Qwen3.8-27B on vLLM, research/log/2026-10-29-1815.md) sends each turn's reasoning back, since the
  model keeps the thinking of earlier turns (preserved thinking), with the model card's thinking-mode sampling
  (temperature 1.0, top_p 0.95, top_k 20) and the reasoning effort as a request parameter; a server error ends the
  episode at once (before, the loop asked the server up to RESAMPLES more times and still ended the episode, dropping
  any reply it got);
- PROVIDER "deepseek" sends each turn's reasoning back, which DeepSeek's thinking mode requires when tools are used,
  drops the sampling parameters it ignores, and stops before a model call once the spend reaches BUDGET's cap.

Version 2.1 (research/log/2026-10-29-1236.md), from the tool-design advice of research/log/2026-10-29-1036.md: the
descriptions of get_fund and subscribe say that the minimum initial subscription is in the prospectus, to be looked up
before a first subscription, and the system prompt asks for all arithmetic on the calculator. The constraints part
is described as it runs: it also checks rule 7.

Version 3 (research/log/2026-10-31-0150.md): the context is append-only, steps are counted per kind of turn, and every
model turn keeps its tokens, so that a trajectory is exactly what the model saw and can be trained on as sampled.
- Append-only context, in every configuration: no message is changed after it is sent, so the messages of each model
  call are a prefix of the next call's and of the stored trajectory. Older tool results are no longer summarized;
  instead every tool result is cut when it is produced at Settings.tool_result_chars characters (8,000: a whole
  get_fund call of 20 products, about 6,200, and read_chunk with its neighbours, about 7,100, fit), with a note on how
  to ask for less, and list tools keep their pages. A reply whose text is a tool's arguments is not executed or
  rewritten: it stays as written and the model is told to make the call through the tool interface. A tool name with
  a harmony suffix is still executed as the tool it starts with, but the message keeps the name as the model wrote it.
  The reasoning goes into the assistant message only where the provider takes it back (qwen, deepseek), and is kept
  in the turn record otherwise (vllm, gpt-oss); a tool result carries its tool's name except for deepseek.
- Step accounting: Settings.max_steps (100) counts task turns only. A turn whose calls are all set_checklist calls is
  a checklist turn, with at most Settings.max_checklist_calls (20) set_checklist calls in an episode, after which the
  call is refused as a tool error and the turn counts as a task turn; a turn with a rejected finish and the
  Settings.redo_steps (10) turns after it are redo turns, with at most MAX_REDOS rejections (as before). Checklist and
  redo turns do not count toward max_steps; the episode ends at max_steps task turns or, as a guard, at max_steps +
  max_checklist_calls + MAX_REDOS x (redo_steps + 1) turns in all ("hard_cap"), which the counts above never exceed.
- Token records: on a vLLM server each request asks for the sampled tokens (logprobs, return_token_ids,
  return_tokens_as_token_ids), and each turn records the prompt's token count, the completion's token ids and
  logprobs (harness/tokens.py) and whether the prompt began with the previous turn's prompt and completion tokens
  (prefix_ok), which fails where the server's chat template renders an earlier turn differently from how it was
  sampled. A server that refuses these options gets requests without them for the rest of the episode.
- Environments are kinds behind one interface (harness/envs/base.py), selected by each task's "env" field, and every
  module is imported as part of the harness package, so each environment's ToolError is the class the loop catches,
  and any other exception from a tool also comes back to the model as a tool error ("internal_errors").
- Per-kind verification and slow tools (research/log/2026-10-31-0150.md, W7): which numbers of a finish answer count
  as supported by the tool results, and the message that names the others, come from the kind
  (EnvKind.unsupported_numbers and unsupported_message; the default is the rule above, so wealth and the other kinds
  verify as before); the checklist check stays shared. The tools a kind lists in ASYNC_TOOLS (market's run_python,
  a subprocess of up to 30 s) run in a worker thread (asyncio.to_thread), so they no longer stall the other episodes
  of the run; every other tool, and every use of an episode's database, stays on the event loop's thread.
H0 keeps the shared parts only: system prompt, tools, the result cap, the loop and resampling. H1 adds the checklist
(set_checklist, the system prompt's note, the checklist restated after every tool result), constraints, verification
and correction as in version 2.1; H1-constraints, H1-verification and H1-correction each drop one of the last three.
"""
import asyncio
import collections
import dataclasses
import json
import re

from harness import tokens
from harness.envs import base
from harness.envs.base import S, schema
from harness.envs.registry import kind_of
from harness.envs.wealth import grade

CONFIGS = {"H0": dict(context=False, constraints=False, verification=False, correction=False),
           "H1": dict(context=True, constraints=True, verification=True, correction=True),
           "H1-constraints": dict(context=True, constraints=False, verification=True, correction=True),
           "H1-verification": dict(context=True, constraints=True, verification=False, correction=True),
           "H1-correction": dict(context=True, constraints=True, verification=True, correction=False)}
CHECKLIST_NOTE = "\n\nKeep a checklist of the steps the request needs with set_checklist, and mark steps done as you go."
CHECKLIST_TOOL = schema("set_checklist", "Write or update the checklist of steps for this request.",
                        {"items": {"type": "array", "items": S}, "done": {"type": "array", "items": {"type": "integer"},
                                                                         "description": "numbers (from 1) of the done items"}},
                        ["items"])
CHECKLIST_EXAMPLE = {"items": ["look up the customer", "..."], "done": [1]}
NUDGE = "Use the tools to act on the request, and call finish with the final answer when done."
TEXT_CALL_NUDGE = ("Your reply has no tool call: its text reads as the arguments of {name}. Make the call through the "
                   "tool interface instead; call finish with the final answer when done.")
CUT_NOTE = ("\n[cut: the result has {n:,} characters and only the first {cap:,} are shown; ask for less at a time, such "
            "as a later page, fewer ids, a shorter date range or fewer neighbouring passages]")
D = "[-‐-―−]"  # a hyphen as gpt-oss writes it ("B‑P0023", U+2011)
ID_OR_DATE = re.compile(rf"\b[ABC]{D}[CP]\d{{4}}\b|\b\d{{4}}{D}\d{{2}}{D}\d{{2}}\b|\brule\s*\d\b|\bR\d\b|\bC\d\b", re.I)
RETRYABLE = re.compile(r"message header|Unknown role|Unexpected token")
OVERFLOW = re.compile(r"context|max_model_len|too long|maximum|max_tokens must be at least", re.I)
TOKEN_OPTIONS = re.compile(r"logprob|token_ids|return_token", re.I)


@dataclasses.dataclass
class Settings:
    """Everything the loop takes from run.py besides the task; one object per run, shared by its episodes."""
    provider: str = "vllm"  # "vllm" (gpt-oss), "qwen" (Qwen on vLLM) or "deepseek": see Episode.request
    reasoning_effort: str = "medium"
    max_steps: int = 100  # task turns
    max_tokens: int = 4096  # per model call, reasoning included
    tool_result_chars: int = 8000
    max_checklist_calls: int = 20
    max_redos: int = 2
    redo_steps: int = 10  # turns after a rejected finish that do not count toward max_steps
    resamples: int = 3
    token_records: bool = True
    budget: object = None  # with provider "deepseek", a Budget

    def hard_cap(self):
        return self.max_steps + self.max_checklist_calls + self.max_redos * (self.redo_steps + 1)


class Budget:
    """Spend of the paid calls, in US dollars, from each response's usage at the given prices per million tokens."""

    def __init__(self, cap_usd, price_in_miss, price_in_hit, price_out):
        self.cap, self.prices, self.spent, self.calls = cap_usd, (price_in_miss, price_in_hit, price_out), 0.0, 0

    def cost(self, usage):
        hit = usage.get("prompt_cache_hit_tokens") or 0
        miss = usage.get("prompt_cache_miss_tokens")
        miss = (usage.get("prompt_tokens") or 0) - hit if miss is None else miss
        a, b, c = self.prices
        return (miss * a + hit * b + (usage.get("completion_tokens") or 0) * c) / 1e6

    def add(self, usage):
        self.spent += self.cost(usage)
        self.calls += 1

    def exhausted(self):
        return self.spent >= self.cap


def text_call(content, tools):
    """The (name, arguments) of a reply whose whole text is one tool's arguments as a JSON object, or None: required
    keys all present, no key the tool lacks, and no other tool fitting as well."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip())
    if not text.startswith("{"):
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not obj:
        return None
    fits = [t["function"]["name"] for t in tools
            if set(t["function"]["parameters"]["required"]) <= set(obj) <= set(t["function"]["parameters"]["properties"])]
    return (fits[0], obj) if len(fits) == 1 else None


def valid_args(tools, name, args):
    """Schema check of a write's arguments: the problem, or None (version 2's check: keys, and positive amounts)."""
    spec = next(t["function"]["parameters"] for t in tools if t["function"]["name"] == name)
    missing = [k for k in spec["required"] if k not in args]
    extra = [k for k in args if k not in spec["properties"]]
    if missing or extra:
        return f"missing {missing} / unexpected {extra}"
    for k, p in spec["properties"].items():
        if k in args and "exclusiveMinimum" in p and (not isinstance(args[k], (int, float)) or args[k] <= p["exclusiveMinimum"]):
            return f"{k} must be a positive number" if p["exclusiveMinimum"] == 0 else f"{k} must be above {p['exclusiveMinimum']}"
    return None


class Episode:
    def __init__(self, task, cfg_name, db_path, client, model, seed, settings=None, prompt_suffix=""):
        self.task, self.cfg_name, self.cfg = task, cfg_name, CONFIGS[cfg_name]
        self.s = settings or Settings()
        self.kind = kind_of(task)
        self.env = self.kind.make_env(task, db_path)
        self.client, self.model, self.seed = client, model, seed
        system = self.kind.SYSTEM + (CHECKLIST_NOTE if self.cfg["context"] else "")
        self.msgs = [{"role": "system", "content": system}, {"role": "user", "content": task["prompt"] + prompt_suffix}]
        self.tools = list(self.kind.TOOLS) + ([CHECKLIST_TOOL] if self.cfg["context"] else [])
        self.known = {t["function"]["name"] for t in self.tools}
        self.examples = dict(self.kind.EXAMPLES, set_checklist=CHECKLIST_EXAMPLE)
        self.checklist, self.done = [], set()
        self.calls = collections.Counter()
        self.seen_numbers = set()
        self.results = []  # the text of every successful tool result other than set_checklist, in order
        self.trades = []  # (write tool, arguments, kind.trade_info) of each executed write
        self.usage = []  # each paid call's usage, as the provider returned it
        self.stats = collections.Counter()
        self.turns = []  # one record per model turn (see read_reply)
        self.end = None
        self.task_steps = self.checklist_steps = self.redo_steps = 0
        self.redo_allowance, self.checklist_calls = 0, 0
        self.redos_left = self.s.max_redos if self.cfg["correction"] else 1
        self.send_reasoning = self.s.provider in ("qwen", "deepseek")
        self.token_records = self.s.token_records and self.s.provider != "deepseek"
        self._last_tokens = None  # (prompt ids, completion ids) of the last turn, for prefix_ok

    # -- verification -----------------------------------------------------------------------------------------------
    def checklist_text(self):
        if not self.checklist:
            return "\n[checklist] none yet; write one with set_checklist."
        return "\n[checklist] " + "; ".join(f"{i}. [{'x' if i in self.done else ' '}] {s}" for i, s in enumerate(self.checklist, 1))

    def note_numbers(self, text):
        for n in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")):
            self.seen_numbers.add(round(float(n), 2))

    def untraced(self, answer):
        out = []
        for n in re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", ID_OR_DATE.sub(" ", grade.normalize(answer))):
            v = float(n.replace("$", "").replace(",", ""))
            if abs(v) < 10 and v == int(v):  # small whole numbers: counts, rule numbers
                continue
            if not any(abs(v - s) <= 0.011 or abs(v - s * 100) <= 0.011 for s in self.seen_numbers):
                out.append(n)
        return out

    def verify(self, answer):
        problems = []
        bad = self.kind.unsupported_numbers(self, answer)
        if bad:
            problems.append(self.kind.unsupported_message(bad))
        if self.cfg["context"]:
            todo = [s for i, s in enumerate(self.checklist, 1) if i not in self.done]
            if not self.checklist:
                problems.append("no checklist was written")
            elif todo:
                problems.append(f"checklist items not done: {'; '.join(todo)}")
        return problems + self.kind.verify(self, answer)

    # -- tools ------------------------------------------------------------------------------------------------------
    def tool_name(self, name):
        """gpt-oss now and then writes a tool's name with a harmony suffix, such as "calculator<|channel|>commentary"
        (research/log/2026-10-27-1945.md); a name that starts with a known tool's name followed by a character no tool
        name has is executed as that tool, in every configuration alike (the message keeps the name as written)."""
        m = re.match(r"\w+", name or "")
        if name not in self.known and m and m.group(0) in self.known:
            self.stats["renamed_tools"] += 1
            return m.group(0)
        return name

    def cap(self, text):
        n, c = len(text), self.s.tool_result_chars
        if n <= c:
            return text
        self.stats["cut_results"] += 1
        return text[:c] + CUT_NOTE.format(n=n, cap=c)

    def tool_error(self, name, message):
        self.stats["tool_errors"] += 1
        err = {"error": message}
        if self.cfg["correction"]:
            err["example_of_a_valid_call"] = {name: self.examples.get(name, {})}
        return json.dumps(err)

    def set_checklist(self, args):
        self.checklist_calls += 1
        if self.checklist_calls > self.s.max_checklist_calls:
            return self.tool_error("set_checklist", f"the checklist can be set at most {self.s.max_checklist_calls} times "
                                                    f"in a task; go on with the request"), False
        try:  # a malformed call, such as "done": 3, is answered like any other tool's (research/log/2026-10-30-1145.md)
            items = [str(s) for s in args.get("items", [])]
            done = {int(i) for i in args.get("done", []) if str(i).isdigit()}
        except (TypeError, ValueError, AttributeError) as e:
            return self.tool_error("set_checklist", str(e)), True
        self.checklist, self.done = items or self.checklist, done
        return json.dumps({"status": "ok"}), True

    async def call_tool(self, name, args):
        """Execute one call other than finish; returns (result text, whether it was a set_checklist call within the cap).
        A tool in the kind's ASYNC_TOOLS runs in a worker thread; the others run here, on the event loop's thread."""
        key = (name, json.dumps(args, sort_keys=True))
        self.calls[key] += 1
        hint = ""
        if self.cfg["correction"] and self.calls[key] >= 3:
            self.stats["loop_hints"] += 1
            hint = "\nNote: this exact call has now been made 3 or more times; try a different approach."
        if name == "set_checklist" and self.cfg["context"]:
            text, free = self.set_checklist(args)
            return text + hint, free
        if name in self.kind.WRITES and self.cfg["constraints"]:
            problem = valid_args(self.tools, name, args)
            broken = [] if problem else self.kind.check_write(self.env, name, args)
            if problem or broken:
                self.stats["constraint_blocks"] += 1
                out = {"status": "rejected", "violated_rules": [r for r, _ in broken]} if broken else {"status": "rejected", "error": problem}
                if self.cfg["correction"] and broken:
                    out["reasons"] = [why for _, why in broken]
                    out |= self.kind.rejection_details(self.env, name, args, broken)
                return json.dumps(out) + hint, False
        fn = getattr(self.env, name, None) if name in self.known and name != "finish" else None
        if fn is None:
            self.stats["tool_errors"] += 1
            return json.dumps({"error": f"unknown tool {name}"}) + hint, False
        info = self.kind.trade_info(self.env, name, args) if name in self.kind.WRITES else None
        try:
            res = await asyncio.to_thread(fn, **args) if name in self.kind.ASYNC_TOOLS else fn(**args)
        except (base.ToolError, TypeError, ValueError) as e:
            return self.cap(self.tool_error(name, str(e))) + hint, False
        except Exception as e:  # a bug in the environment: the model is told, the run goes on, the row counts it
            self.stats["internal_errors"] += 1
            self.stats["internal_error"] = self.stats.get("internal_error") or f"{name}: {type(e).__name__}: {e}"[:300]
            return self.cap(self.tool_error(name, f"the tool failed ({type(e).__name__}: {e})")) + hint, False
        if name in self.kind.WRITES:
            self.trades.append((name, args, info))
        text = self.cap(json.dumps(res))
        self.note_numbers(text)
        self.results.append(text)
        return text + hint, False

    async def execute(self, raw_name, arguments):
        """One tool call of a turn: (result text, kind) where kind is "checklist" for a set_checklist call within the
        cap, "rejected" for a finish that verification rejected, and None otherwise."""
        name = self.tool_name(raw_name)
        try:
            args = json.loads(arguments or "{}")
            assert isinstance(args, dict)
        except (json.JSONDecodeError, AssertionError, TypeError):
            self.stats["tool_errors"] += 1
            return json.dumps({"error": "arguments are not a JSON object"}), None
        kind = None
        if name == "finish":
            problems = self.verify(str(args.get("answer", ""))) if self.cfg["verification"] else []
            if problems and self.redos_left > 0:
                self.redos_left -= 1
                self.stats["verification_rejects"] += 1
                text, kind = json.dumps({"status": "rejected", "verification_failed": problems} if self.cfg["correction"]
                                        else {"status": "rejected", "verification_failed": True}), "rejected"
            else:
                text = json.dumps(self.env.finish(args.get("answer", "")))
        else:
            text, free = await self.call_tool(name, args)
            kind = "checklist" if free else None
        if self.cfg["context"] and not self.env.finished:
            text += self.checklist_text()
        return text, kind

    # -- model calls ------------------------------------------------------------------------------------------------
    def request(self, attempt, step):
        """The chat-completions arguments of one model call; the messages are the episode's, never edited."""
        p = self.s.provider
        if p == "deepseek":
            # thinking mode ignores temperature and the penalties, and takes its effort as reasoning_effort
            return dict(model=self.model, messages=list(self.msgs), tools=self.tools, max_tokens=self.s.max_tokens,
                        reasoning_effort=self.s.reasoning_effort, extra_body={"thinking": {"type": "enabled"}})
        seed = self.seed * 1000 + step + 100_000 * attempt
        tool_choice = "required" if attempt == self.s.resamples else "auto"
        if p == "qwen":
            kw = dict(temperature=1.0, top_p=0.95, reasoning_effort=self.s.reasoning_effort,
                      extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}})
        else:
            kw = dict(temperature=1.0, top_p=1.0, extra_body={"reasoning_effort": self.s.reasoning_effort})
        if self.token_records:
            kw["logprobs"], kw["top_logprobs"] = True, 0
            kw["extra_body"] = dict(kw["extra_body"], return_token_ids=True, return_tokens_as_token_ids=True)
        return dict(model=self.model, messages=list(self.msgs), tools=self.tools, tool_choice=tool_choice,
                    max_tokens=self.s.max_tokens, seed=seed, **kw)

    def sampling(self):
        """The request's settings without the conversation, for the trajectory."""
        r = self.request(0, 0)
        return {k: v for k, v in r.items() if k not in ("messages", "tools", "seed", "model")}

    async def sample(self, step):
        attempt = 0
        while True:
            try:
                return await self.client.chat.completions.create(**self.request(attempt, step)), attempt
            except Exception as e:
                msg = str(e)
                if self.token_records and TOKEN_OPTIONS.search(msg) and not OVERFLOW.search(msg):
                    self.token_records = False  # a server without the token options: go on without them
                    self.stats["token_records_off"] = msg[:300]
                    continue
                # gpt-oss now and then writes a reply vLLM's harmony parser rejects (HTTP 500 "unexpected tokens
                # remaining in message header", "Unknown role: final", "Unexpected token ... while"); the same seed
                # gives the same reply, so sample another; the last try requires a tool call
                if attempt < self.s.resamples and RETRYABLE.search(msg):
                    self.stats["resamples"] += 1
                    attempt += 1
                    continue
                # context overflow and other server errors end the episode; vLLM reports a prompt that leaves no room
                # to generate as "max_tokens must be at least 1, got -n"
                self.end = "context_overflow" if OVERFLOW.search(msg) else "api_error"
                self.stats["api_error"] = msg[:300]
                return None, attempt

    def read_reply(self, r, attempt, step):
        """The assistant message as the model wrote it, its tool calls, and the turn's record."""
        ch = r.choices[0]
        m = ch.message
        u = r.usage
        self.stats["prompt_tokens"] += (u.prompt_tokens or 0) if u else 0
        self.stats["completion_tokens"] += (u.completion_tokens or 0) if u else 0
        if u is not None and self.s.provider == "deepseek":
            usage = u.model_dump()
            self.usage.append(usage)
            if self.s.budget is not None:
                self.s.budget.add(usage)
        msg = {"role": "assistant", "content": m.content or ""}
        rec = {"msg": len(self.msgs), "seed": self.seed * 1000 + step + 100_000 * attempt, "attempt": attempt,
               "finish_reason": getattr(ch, "finish_reason", None),
               "n_prompt": u.prompt_tokens if u else None, "n_completion": u.completion_tokens if u else None}
        reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
        if reasoning:
            if self.send_reasoning:
                msg["reasoning_content"] = reasoning
            else:
                rec["reasoning"] = reasoning  # not sent back (gpt-oss on vLLM renders no earlier reasoning)
        calls = []
        if m.tool_calls:
            msg["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name,
                                                                                "arguments": tc.function.arguments}}
                                 for tc in m.tool_calls]
            calls = [(tc.id, tc.function.name, tc.function.arguments) for tc in m.tool_calls]
        rec["hit_max_tokens"] = bool(u and (u.completion_tokens or 0) >= self.s.max_tokens)
        self.stats["hit_max_tokens"] += rec["hit_max_tokens"]
        prompt_ids, ids, logprobs = tokens.from_response(r)
        if ids is not None:
            rec["token_ids"] = tokens.pack_ids(ids)
            if logprobs is not None:
                rec["logprobs"] = tokens.pack_logprobs(logprobs)
        if prompt_ids is not None:
            rec["n_prompt"] = len(prompt_ids)
            if self._last_tokens is not None:
                p, c = self._last_tokens
                rec["prefix_ok"] = prompt_ids[:len(p) + len(c)] == p + c
                self.stats["token_prefix_breaks"] += not rec["prefix_ok"]
        self._last_tokens = (prompt_ids, ids) if prompt_ids is not None and ids is not None else None
        return msg, calls, rec

    def charge(self, rec, kinds, calls):
        """Which count the turn goes to: checklist, redo or task (Version 3 above)."""
        if calls and all(k == "checklist" for k in kinds) and len(kinds) == len(calls):
            self.checklist_steps += 1
            rec["step"] = "checklist"
        elif "rejected" in kinds:
            self.redo_steps += 1
            self.redo_allowance = self.s.redo_steps
            rec["step"] = "redo"
        elif self.redo_allowance > 0:
            self.redo_allowance -= 1
            self.redo_steps += 1
            rec["step"] = "redo"
        else:
            self.task_steps += 1
            rec["step"] = "task"

    async def run(self):
        self.note_numbers(self.task["prompt"])
        self.note_numbers(self.kind.SYSTEM)  # the rules' own numbers (30%, 5%, 7 days) may be quoted in an answer
        while True:
            if self.task_steps >= self.s.max_steps:
                self.end = "step_limit"
                break
            if len(self.turns) >= self.s.hard_cap():
                self.end = "hard_cap"
                break
            if self.s.budget is not None and self.s.budget.exhausted():
                self.end = "budget"
                break
            step = len(self.turns)
            r, attempt = await self.sample(step)
            if self.end:
                break
            msg, calls, rec = self.read_reply(r, attempt, step)
            self.msgs.append(msg)
            self.turns.append(rec)
            kinds = []
            if not calls:
                tc = text_call(msg["content"], self.tools)
                if tc:
                    self.stats["text_calls"] += 1
                self.msgs.append({"role": "user", "content": TEXT_CALL_NUDGE.format(name=tc[0]) if tc else NUDGE})
            for call_id, raw_name, arguments in calls:
                text, kind = await self.execute(raw_name, arguments)
                kinds.append(kind)
                tool_msg = {"role": "tool", "tool_call_id": call_id, "content": text}
                if self.s.provider != "deepseek":  # vLLM renders a gpt-oss tool result as authored by functions.<name>
                    tool_msg["name"] = raw_name
                self.msgs.append(tool_msg)
                if self.env.finished:
                    break
            self.charge(rec, kinds, calls)
            if self.env.finished:
                self.end = "finished"
                break
        return self


ROW_STATS = ("tool_errors", "constraint_blocks", "verification_rejects", "loop_hints", "resamples", "renamed_tools",
             "text_calls", "cut_results", "internal_errors", "hit_max_tokens", "token_prefix_breaks")


async def run_one(task, cfg_name, db_path, client, model, seed, settings=None, prompt_suffix=""):
    """One episode: its result row and its trajectory record."""
    ep = await Episode(task, cfg_name, db_path, client, model, seed, settings, prompt_suffix).run()
    g = ep.kind.grade(task, ep.env)
    row = {"task_id": task["task_id"], "env": ep.kind.name, "instance": task.get("instance"), "type": task.get("type"),
           "template": task.get("template"), "feasible": task.get("feasible", True), "config": cfg_name, "model": model,
           "run": seed, "harness": "v3", "passed": g["passed"], "end": ep.end, "steps": len(ep.turns),
           "task_steps": ep.task_steps, "checklist_steps": ep.checklist_steps, "redo_steps": ep.redo_steps,
           "prompt_tokens": ep.stats["prompt_tokens"], "completion_tokens": ep.stats["completion_tokens"],
           **{k: ep.stats[k] for k in ROW_STATS},
           "failure": ep.end if ep.end in ("api_error", "budget") else ep.kind.categorize(ep, g), "grade": g,
           "answer": ep.env.answer, "api_error": ep.stats.get("api_error"), "internal_error": ep.stats.get("internal_error"),
           "token_records": ep.token_records and any("token_ids" in t for t in ep.turns), "usage": ep.usage or None}
    traj = {"task_id": task["task_id"], "run": seed, "config": cfg_name, "harness": "v3", "model": model,
            "provider": ep.s.provider, "sampling": ep.sampling(), "tools": ep.tools, "messages": ep.msgs,
            "turns": ep.turns}
    return row, traj
