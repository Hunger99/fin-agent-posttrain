"""The agent loop and the five harness parts, per docs/03_harness.md sections 4-5.

Every configuration shares the system prompt (role, business rules), the tool interface (native function calling
described by JSON Schema, structured JSON results, paged lists, search returning a list with details fetched
separately) and the loop: at most MAX_STEPS model calls per task, a redo after a correction counting as a step. Three
parts of the full harness H1 can be switched off one at a time, and context management only comes with H1:

- context management: tool results older than the last KEEP_RESULTS are replaced by a short summary (errors are kept
  whole), and the model keeps a checklist with set_checklist, restated after every tool result;
- constraints: before a write, the arguments are checked against the schema and the business rules 1-4
  (env/core.check); a write that breaks a rule is rejected without being executed;
- verification: when the model calls finish, every number in the answer must appear in a tool result or the request,
  every checklist item must be done, and each trade made in the episode must satisfy the rules on the final state;
  a failed check rejects the finish;
- correction: a rejected finish comes back with the failing checks, at most MAX_REDOS times (without correction,
  only a bare "verification failed", once); a rule rejection comes with its reason and the largest allowed amount;
  tool errors come with an example of a valid call; the third identical call gets a hint to change approach.

H0 is the shared parts alone. A rejected write and an error are returned to the model in every configuration, since
the environment answers every call. A model reply that the server cannot parse into a message is sampled again with
another seed, at most RESAMPLES times, the last time with a tool call required, in every configuration alike.
"""
import asyncio
import collections
import json
import re

from env import core, grade

MAX_STEPS, KEEP_RESULTS, MAX_REDOS, MAX_TOKENS, RESAMPLES = 60, 5, 2, 4096, 3
SUMMARY_CHARS = 200  # how much of an older tool result context management keeps (research/log/2026-10-28-0135.md)
CONFIGS = {"H0": dict(context=False, constraints=False, verification=False, correction=False),
           "H1": dict(context=True, constraints=True, verification=True, correction=True),
           "H1-constraints": dict(context=True, constraints=False, verification=True, correction=True),
           "H1-verification": dict(context=True, constraints=True, verification=False, correction=True),
           "H1-correction": dict(context=True, constraints=True, verification=True, correction=False)}

SYSTEM = f"""You are an operations assistant on a fund platform, serving requests about customers' accounts with the tools provided. Today is {core.TODAY.isoformat()}; every NAV the tools return is today's unless dated otherwise. Amounts are US dollars.

Business rules:
1. A customer may subscribe only to products whose risk level (R1-R5) is at most the customer's risk tolerance (C1-C5).
2. After a subscription, no single product's market value may exceed 30% of the customer's total assets.
3. After a subscription, the customer's available cash must be at least 5% of total assets.
4. A product whose subscription (or redemption) is suspended cannot be subscribed (or redeemed).
5. Redemption fee by holding period since the purchase date: under 7 days 1.5%, 7 days to under 1 year 0.5%, 1 year or more none.
6. An operation that breaks a rule must not be executed; say which rule prevents it.
7. A first subscription to a product (the customer holds none of it) must be at least the minimum initial subscription in the product's prospectus; prospectuses are in the knowledge base.
Total assets are available cash plus all holdings at today's NAV. A subscription fee is charged on the amount; the rest buys units at today's NAV.

Act on the request as stated without asking questions. Use as few tool calls as you can: one get_fund call can look up many products, one calculator call can evaluate many expressions, and one expression can combine many values, such as a customer's total assets. When done, call finish with the final answer, giving every value the request asks for (amounts to the cent)."""
CHECKLIST_NOTE = ("\n\nKeep a checklist of the steps the request needs with set_checklist, and mark steps done as you go; "
                  "older tool results are shortened to save space, so note the values you will need.")


def schema(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required, "additionalProperties": False}}}


S, I, N, B = {"type": "string"}, {"type": "integer", "minimum": 1}, {"type": "number"}, {"type": "boolean"}
DATE = {"type": "string", "description": "YYYY-MM-DD"}
TOOLS = [
    schema("get_customer", "A customer's name, risk tolerance and available cash.", {"customer_id": S}, ["customer_id"]),
    schema("list_holdings", "A customer's holdings (units, cost basis, purchase date), 20 per page.",
           {"customer_id": S, "page": I}, ["customer_id"]),
    schema("get_fund", "Products' type, risk level, today's NAV, fees and suspensions; up to 20 products in one call, e.g. "
           "all of a customer's holdings.", {"product_ids": {"type": "array", "items": S, "minItems": 1, "maxItems": 20}},
           ["product_ids"]),
    schema("search_funds", "List products, 20 per page, optionally filtered and sorted; use get_fund for details.",
           {"type": {"type": "string", "enum": ["money_market", "bond", "mixed", "index", "equity"]},
            "max_risk_level": {"type": "string", "enum": ["R1", "R2", "R3", "R4", "R5"]}, "open_for_subscription": B,
            "sort_by": {"type": "string", "enum": ["product_id", "management_fee", "subscription_fee"]}, "page": I}, []),
    schema("get_nav_history", "A product's daily NAVs between two dates, 60 per page.",
           {"product_id": S, "start_date": DATE, "end_date": DATE, "page": I}, ["product_id", "start_date", "end_date"]),
    schema("fund_metrics", "Return, maximum drawdown and annualized volatility (percent) between two dates.",
           {"product_id": S, "start_date": DATE, "end_date": DATE}, ["product_id", "start_date", "end_date"]),
    schema("calculator", "Evaluate one or more arithmetic expressions in one call, e.g. [\"(2.7124*1196.65)*(1-0.005)\", "
           "\"5055.3 + 48744.1891*1.3714 + 737.5851*3.1643\"]; put every value you need at this point into one call.",
           {"expressions": {"type": "array", "items": S, "minItems": 1}}, ["expressions"]),
    schema("search_knowledge", "Search the knowledge base of fund prospectuses; returns passage ids, titles and snippets.",
           {"query": S, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
    schema("get_document", "The full text of a knowledge-base passage.", {"doc_id": S}, ["doc_id"]),
    schema("subscribe", "Buy a product for a customer with a dollar amount (fee included). Executes immediately.",
           {"customer_id": S, "product_id": S, "amount": {"type": "number", "exclusiveMinimum": 0}},
           ["customer_id", "product_id", "amount"]),
    schema("redeem", "Sell units of a customer's holding at today's NAV, minus the redemption fee. Executes immediately.",
           {"customer_id": S, "product_id": S, "units": {"type": "number", "exclusiveMinimum": 0}},
           ["customer_id", "product_id", "units"]),
    schema("finish", "End the task with the final answer to the request.", {"answer": S}, ["answer"]),
]
CHECKLIST_TOOL = schema("set_checklist", "Write or update the checklist of steps for this request.",
                        {"items": {"type": "array", "items": S}, "done": {"type": "array", "items": {"type": "integer"},
                                                                         "description": "numbers (from 1) of the done items"}},
                        ["items"])
EXAMPLES = {"get_customer": {"customer_id": "B-C0001"}, "list_holdings": {"customer_id": "B-C0001", "page": 1},
            "get_fund": {"product_ids": ["B-P0006", "B-P0170"]},
            "search_funds": {"type": "bond", "max_risk_level": "R2", "open_for_subscription": True, "sort_by": "management_fee"},
            "get_nav_history": {"product_id": "B-P0006", "start_date": "2026-06-01", "end_date": "2026-06-30"},
            "fund_metrics": {"product_id": "B-P0006", "start_date": "2025-06-30", "end_date": "2026-06-30"},
            "calculator": {"expressions": ["1196.65 * 2.7124 * (1 - 0.005)", "5055.3 + 48744.1891 * 1.3714"]},
            "search_knowledge": {"query": "B-P0006 minimum initial subscription", "top_k": 5},
            "get_document": {"doc_id": "B-P0006-2"},
            "subscribe": {"customer_id": "B-C0001", "product_id": "B-P0170", "amount": 5000},
            "redeem": {"customer_id": "B-C0001", "product_id": "B-P0006", "units": 1196.65},
            "finish": {"answer": "..."}, "set_checklist": {"items": ["look up the customer", "..."], "done": [1]}}
WRITES = ("subscribe", "redeem")
ID_OR_DATE = re.compile(r"\b[ABC]-[CP]\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\brule\s*\d\b|\bR\d\b|\bC\d\b", re.I)


def valid_args(name, args):
    """Schema check of a write's arguments: the problem, or None."""
    spec = next(t["function"]["parameters"] for t in TOOLS if t["function"]["name"] == name)
    missing = [k for k in spec["required"] if k not in args]
    extra = [k for k in args if k not in spec["properties"]]
    if missing or extra:
        return f"missing {missing} / unexpected {extra}"
    for k in ("amount", "units"):
        if k in args and (not isinstance(args[k], (int, float)) or args[k] <= 0):
            return f"{k} must be a positive number"
    return None


def largest_allowed(db, c, p):
    lo, hi = 0, int(db.execute("SELECT cash FROM customers WHERE customer_id=?", (c,)).fetchone()[0] * 100)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        broken = {r for r, _ in core.check(db, "subscribe", {"customer_id": c, "product_id": p, "amount": mid / 100})}
        lo, hi = (mid, hi) if not broken & {2, 3} else (lo, mid - 1)  # rules 2 and 3 only get harder as the amount grows
    return lo / 100


class Episode:
    # An environment other than the wealth-management one (DocFinQA, the ML tasks) subclasses Episode and sets these
    SYSTEM, TOOLS = SYSTEM, TOOLS

    def make_env(self, db_path):
        return core.Env(db_path)

    def __init__(self, task, cfg_name, db_path, client, model, seed):
        self.task, self.cfg, self.cfg_name = task, CONFIGS[cfg_name], cfg_name
        self.env, self.client, self.model, self.seed = self.make_env(db_path), client, model, seed
        system = self.SYSTEM + (CHECKLIST_NOTE if self.cfg["context"] else "")
        self.msgs = [{"role": "system", "content": system}, {"role": "user", "content": task["prompt"]}]
        self.tools = self.TOOLS + ([CHECKLIST_TOOL] if self.cfg["context"] else [])
        self.checklist, self.done = [], set()
        self.calls = collections.Counter()
        self.seen_numbers = set()
        self.trades = []
        self.stats = collections.Counter()
        self.end = None

    def grade(self):
        return grade.grade(self.task, self.env.state(self.task["customers"]), self.env.answer)

    def categorize(self, g):
        return categorize(self, g)

    def view(self):
        """The messages the model sees: under context management, older tool results are summarized.

        The reasoning of earlier turns stays in the trajectory but is not sent: vLLM drops it when it renders a gpt-oss
        conversation (entrypoints/harmony_utils.py), and an unknown key could fail request validation."""
        if not self.cfg["context"]:
            return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in self.msgs]
        tool_idx = [i for i, m in enumerate(self.msgs) if m["role"] == "tool"]
        old = set(tool_idx[:-KEEP_RESULTS])
        out = []
        for i, m in enumerate(self.msgs):
            m = {k: v for k, v in m.items() if k != "reasoning_content"}
            if i in old and not m["content"].startswith('{"error"'):
                m = dict(m, content="[summarized] " + m["content"][:SUMMARY_CHARS] +
                         ("…" if len(m["content"]) > SUMMARY_CHARS else ""))
            out.append(m)
        return out

    def checklist_text(self):
        if not self.checklist:
            return "\n[checklist] none yet; write one with set_checklist."
        return "\n[checklist] " + "; ".join(f"{i}. [{'x' if i in self.done else ' '}] {s}" for i, s in enumerate(self.checklist, 1))

    def note_numbers(self, text):
        for n in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")):
            self.seen_numbers.add(round(float(n), 2))

    def untraced(self, answer):
        out = []
        for n in re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", ID_OR_DATE.sub(" ", answer)):
            v = float(n.replace("$", "").replace(",", ""))
            if abs(v) < 10 and v == int(v):  # small whole numbers: counts, rule numbers
                continue
            if not any(abs(v - s) <= 0.011 or abs(v - s * 100) <= 0.011 for s in self.seen_numbers):
                out.append(n)
        return out

    def verify(self, answer):
        problems = []
        bad = self.untraced(answer)
        if bad:
            problems.append(f"these numbers in the answer appear in no tool result: {', '.join(bad[:8])}; compute them "
                            f"with the calculator or correct them")
        if self.cfg["context"]:
            todo = [s for i, s in enumerate(self.checklist, 1) if i not in self.done]
            if not self.checklist:
                problems.append("no checklist was written")
            elif todo:
                problems.append(f"checklist items not done: {'; '.join(todo)}")
        for name, args, first in self.trades:
            c, p = args["customer_id"], args["product_id"]
            cust = self.env.db.execute("SELECT * FROM customers WHERE customer_id=?", (c,)).fetchone()
            prod = self.env.db.execute("SELECT * FROM products WHERE product_id=?", (p,)).fetchone()
            if name == "subscribe":
                t = core.total_assets(self.env.db, c)
                v = (self.env.db.execute("SELECT units FROM holdings WHERE customer_id=? AND product_id=?", (c, p)).fetchone() or [0])[0] * prod["nav"]
                if prod["risk_level"] > cust["risk_level"]:
                    problems.append(f"the subscription of {p} for {c} broke rule 1")
                if prod["subscription_suspended"]:
                    problems.append(f"the subscription of {p} for {c} broke rule 4")
                if v > core.CONCENTRATION * t + 1e-6:
                    problems.append(f"after the subscription, {p} is {v / t:.1%} of {c}'s total assets (rule 2)")
                if cust["cash"] < core.CASH_FLOOR * t - 1e-6:
                    problems.append(f"after the subscription, {c}'s cash is {cust['cash'] / t:.1%} of total assets (rule 3)")
                m = self.env.db.execute("SELECT min_subscription FROM prospectus WHERE product_id=?", (p,)).fetchone()[0]
                if first and float(args["amount"]) < m - 1e-9:
                    problems.append(f"the first subscription of {p} for {c} is below its prospectus minimum (rule 7)")
            elif prod["redemption_suspended"]:
                problems.append(f"the redemption of {p} for {c} broke rule 4")
        return problems

    def tool_name(self, name):
        """gpt-oss now and then writes a tool's name with a harmony suffix, such as "calculator<|channel|>commentary"
        (research/log/2026-10-27-1945.md); a name that starts with a known tool's name followed by a character no tool
        name has is read as that tool, in every configuration alike."""
        known = {t["function"]["name"] for t in self.TOOLS}
        m = re.match(r"\w+", name or "")
        if name not in known and m and m.group(0) in known:
            self.stats["renamed_tools"] += 1
            return m.group(0)
        return name

    def call_tool(self, name, args):
        """Execute one call; returns the result text."""
        key = (name, json.dumps(args, sort_keys=True))
        self.calls[key] += 1
        hint = ""
        if self.cfg["correction"] and self.calls[key] >= 3:
            self.stats["loop_hints"] += 1
            hint = "\nNote: this exact call has now been made 3 or more times; try a different approach."
        if name == "set_checklist" and self.cfg["context"]:
            self.checklist = [str(s) for s in args.get("items", [])] or self.checklist
            self.done = {int(i) for i in args.get("done", []) if str(i).isdigit()}
            return json.dumps({"status": "ok"}) + hint
        if name in WRITES and self.cfg["constraints"]:
            problem = valid_args(name, args)
            broken = [] if problem else core.check(self.env.db, name, args)
            if problem or broken:
                self.stats["constraint_blocks"] += 1
                out = {"status": "rejected", "violated_rules": [r for r, _ in broken]} if broken else {"status": "rejected", "error": problem}
                if self.cfg["correction"] and broken:
                    out["reasons"] = [why for _, why in broken]
                    if name == "subscribe" and {r for r, _ in broken} <= {2, 3}:  # rule 7 has a floor, not a ceiling
                        out["largest_allowed_amount"] = largest_allowed(self.env.db, args["customer_id"], args["product_id"])
                return json.dumps(out) + hint
        fn = getattr(self.env, name, None) if name in {t["function"]["name"] for t in self.TOOLS} else None
        if fn is None:
            self.stats["tool_errors"] += 1
            return json.dumps({"error": f"unknown tool {name}"}) + hint
        first = name == "subscribe" and hasattr(self.env, "db") and self.env.db.execute(
            "SELECT 1 FROM holdings WHERE customer_id=? AND product_id=?", (args.get("customer_id"), args.get("product_id"))).fetchone() is None
        try:
            res = fn(**args)
        except (core.ToolError, TypeError, ValueError) as e:
            self.stats["tool_errors"] += 1
            err = {"error": str(e)}
            if self.cfg["correction"]:
                err["example_of_a_valid_call"] = {name: EXAMPLES.get(name, {})}
            return json.dumps(err) + hint
        if name in WRITES:
            self.trades.append((name, args, first))
        text = json.dumps(res)
        self.note_numbers(text)
        return text + hint

    async def run(self):
        self.note_numbers(self.task["prompt"])
        self.note_numbers(self.SYSTEM)  # the rules' own numbers (30%, 5%, 7 days) may be quoted in an answer
        redos_left = MAX_REDOS if self.cfg["correction"] else 1
        for step in range(MAX_STEPS):
            self.stats["steps"] += 1
            for attempt in range(RESAMPLES + 1):
                try:
                    r = await self.client.chat.completions.create(
                        model=self.model, messages=self.view(), tools=self.tools,
                        tool_choice="required" if attempt == RESAMPLES else "auto", temperature=1.0,
                        top_p=1.0, max_tokens=MAX_TOKENS, seed=self.seed * 1000 + step + 100_000 * attempt,
                        extra_body={"reasoning_effort": "medium"})
                    break
                except Exception as e:
                    # gpt-oss now and then writes a reply vLLM's harmony parser rejects (HTTP 500 "unexpected tokens
                    # remaining in message header", "Unknown role: final", "Unexpected token ... while"); the same
                    # seed gives the same reply, so sample another; the last try requires a tool call, since in the pilot
                    # one state failed on every seed where the model should have called finish
                    if attempt < RESAMPLES and re.search(r"message header|Unknown role|Unexpected token", str(e)):
                        self.stats["resamples"] += 1
                        continue
                    # context overflow and other server errors end the episode; vLLM reports a prompt that leaves no
                    # room to generate as "max_tokens must be at least 1, got -n"
                    overflow = r"context|max_model_len|too long|maximum|max_tokens must be at least"
                    self.end = "context_overflow" if re.search(overflow, str(e), re.I) else "api_error"
                    self.stats["api_error"] = str(e)[:300]
            if self.end:
                break
            u = r.usage
            self.stats["prompt_tokens"] += u.prompt_tokens if u else 0
            self.stats["completion_tokens"] += u.completion_tokens if u else 0
            m = r.choices[0].message
            msg = {"role": "assistant", "content": m.content or ""}
            reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
            if reasoning:
                msg["reasoning_content"] = reasoning
            if m.tool_calls:
                names = [self.tool_name(tc.function.name) for tc in m.tool_calls]
                msg["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": name,
                                                                                     "arguments": tc.function.arguments}}
                                     for tc, name in zip(m.tool_calls, names)]
            self.msgs.append(msg)
            if not m.tool_calls:
                self.msgs.append({"role": "user", "content": "Use the tools to act on the request, and call finish with "
                                                             "the final answer when done."})
                continue
            for tc, name in zip(m.tool_calls, names):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    assert isinstance(args, dict)
                except (json.JSONDecodeError, AssertionError):
                    self.stats["tool_errors"] += 1
                    text = json.dumps({"error": "arguments are not a JSON object"})
                else:
                    if name == "finish":
                        problems = self.verify(str(args.get("answer", ""))) if self.cfg["verification"] else []
                        if problems and redos_left > 0:
                            redos_left -= 1
                            self.stats["verification_rejects"] += 1
                            text = json.dumps({"status": "rejected", "verification_failed": problems}
                                              if self.cfg["correction"] else {"status": "rejected", "verification_failed": True})
                        else:
                            text = json.dumps(self.env.finish(args.get("answer", "")))
                    else:
                        text = self.call_tool(name, args)
                if self.cfg["context"] and not self.env.finished:
                    text += self.checklist_text()
                # vLLM renders a gpt-oss tool result as authored by functions.<name>, so the name goes with it
                self.msgs.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": text})
                if self.env.finished:
                    break
            if self.env.finished:
                self.end = "finished"
                break
        else:
            self.end = "step_limit"
        return self


def categorize(ep, g):
    """The main failure category of a failed episode from its log (docs/03_harness.md section 6); None if it passed."""
    if g["passed"]:
        return None
    t = ep.task
    if ep.end == "context_overflow":
        return "context_overflow"
    if ep.end == "step_limit":
        return "step_limit"
    if max(ep.calls.values(), default=0) >= 3:
        return "loop"
    forbidden = {c for c in t["customers"] if t["gold_state"][c] == t["initial_state"][c]}
    if any(a["customer_id"] in forbidden and not t["feasible"] for _, a, _ in ep.trades):
        return "rule_violation"
    gold_trades = sum(len(s["transactions"]) for s in t["gold_state"].values())
    if g["state"] and len(ep.trades) < gold_trades:
        return "missed_subtask" if ep.trades else "premature_finish"
    if g["state"]:
        return "argument_error"
    if g["missing_keys"] or g["names_rule"] is False:
        return "calculation_error" if any(k["kind"] != "id" for k in g["missing_keys"]) else "other"
    return "other"


async def run_one(task, cfg_name, db_path, client, model, seed, cls=None):
    ep = await (cls or Episode)(task, cfg_name, db_path, client, model, seed).run()
    g = ep.grade()
    return {"task_id": task["task_id"], "instance": task["instance"], "type": task["type"], "template": task["template"],
            "feasible": task["feasible"], "config": cfg_name, "model": model, "run": seed, "passed": g["passed"],
            "end": ep.end, "steps": ep.stats["steps"], "prompt_tokens": ep.stats["prompt_tokens"],
            "completion_tokens": ep.stats["completion_tokens"], "tool_errors": ep.stats["tool_errors"],
            "constraint_blocks": ep.stats["constraint_blocks"], "verification_rejects": ep.stats["verification_rejects"],
            "loop_hints": ep.stats["loop_hints"], "resamples": ep.stats["resamples"],
            "renamed_tools": ep.stats["renamed_tools"], "failure": "api_error" if ep.end == "api_error" else ep.categorize(g), "grade": g, "answer": ep.env.answer,
            "api_error": ep.stats.get("api_error")}, ep.msgs
