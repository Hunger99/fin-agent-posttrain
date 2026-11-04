"""The served chat template and the token prefix (docs/03_harness.md, 9): templates/qwen35_keep_thinking.jinja keeps
every assistant turn's thinking after the first user message, so that with v3's nudges, tool errors and multi-call
turns (a) each prompt starts with the previous prompt's tokens followed by the previous completion's tokens, and (b)
the training encoder (src/03_model-training/utils/rl_data.py's encode, given this template) gives the token ids the model
saw. The native Qwen3.5 template breaks both at a nudge. The client is tests/qwen_stub.py (Qwen3.5's tokenizer, canonical
completions); no model is used.

Run: python -B -m unittest tests.test_template (from <v3>, or discover as in test_v3.py); needs transformers and the
Qwen3.5-4B-Base tokenizer, and is skipped without them.
"""
import asyncio
import json
import os
import unittest

from harness import agent, tokens
from tests import market_tasks, qwen_stub
from tests.stub import tool_results

V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = os.path.join(V3, "templates", "qwen35_keep_thinking.jinja")
RL_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(V3))), "03_model-training", "utils",
                       "rl_data.py")
MARKET_DB = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1/market_v1.sqlite"
WEALTH = os.environ.get("FP_DATA_EN", "<FP_DATA_EN unset>") + "/harness"

try:
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(qwen_stub.TOKENIZER)
except Exception:  # no transformers or no tokenizer here
    TOK = None


def first(path):
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())


def by_turn(steps):
    """A policy playing the steps in order, one per model call (a step may be a function of the messages)."""
    def policy(messages):
        k = sum(m["role"] == "assistant" for m in messages)
        step = steps[min(k, len(steps) - 1)]
        return step(messages) if callable(step) else step
    return policy


def market_steps(h1):
    """Two calls in one turn, a tool error, a reply without a tool call (nudge), two calls again, a reply that is a
    tool's arguments as text (the other nudge), a finish H1 rejects, and a finish after marking the checklist."""
    raw, split = (k["value"] for k in first(market_tasks.path(MARKET_DB))["keys"])  # read from the snapshot
    look = ("get_prices", {"symbol": "AAPL", "start": "2020-08-28", "end": "2020-08-28", "fields": ["close"]})
    plan = [("set_checklist", {"items": ["look up the closes", "answer"]})] if h1 else []
    done = [("set_checklist", {"items": ["look up the closes", "answer"], "done": [1, 2]})] if h1 else []
    return [
        {"reasoning": "Plan, then look up the unadjusted close.", "content": "I will look up the close first.",
         "calls": plan + [look]},
        {"reasoning": "Now the split-adjusted close; try the symbol.",
         "calls": [("get_prices", {"symbol": "NOPE", "start": "2020-08-28", "end": "2020-08-28", "adjustment": "split",
                                   "page": 1})]},
        {"reasoning": "That symbol does not exist (≈ a typo); Apple is AAPL.", "content": "The symbol was wrong."},
        {"reasoning": "Compute the split-adjusted close in Python and check the next trading day.",
         "calls": [("run_python", {"code": "df = data.get_prices('AAPL', '2020-08-28', '2020-08-28', adjustment='split', "
                                           "fields=['close'])\nprint(round(df.close.iloc[0], 2))"}),
                   ("trading_day", {"symbol": "AAPL", "date": "2020-08-29", "offset": 1, "rule": "on_or_after"})]},
        {"reasoning": "Count the trading days of that week.",
         "content": json.dumps({"symbol": "AAPL", "start": "2020-08-24", "end": "2020-08-28"})},
        {"reasoning": "Answer.", "calls": [("finish", {"answer": f"Unadjusted ${raw:.2f}, split-adjusted ${split:.2f}, and 777.77."})]},
        {"reasoning": "Mark the checklist done and answer again.",
         "calls": done + [("finish", {"answer": f"Unadjusted close ${raw:.2f}; split-adjusted close ${split:.2f}."})]},
    ]


def wealth_steps():
    """Wealth (H1): an array and a boolean argument, a calculator call with several expressions, a nudge."""
    return [
        {"reasoning": "Plan.", "calls": [("set_checklist", {"items": ["look up", "answer"]}),
                                         ("get_customer", {"customer_id": "C-C1613"})]},
        {"reasoning": "Find open bond funds.", "calls": [("search_funds", {"type": "bond", "max_risk_level": "R3",
                                                                           "open_for_subscription": True,
                                                                           "sort_by": "management_fee"})]},
        {"reasoning": "Read a passage that does not exist.", "calls": [("get_document", {"doc_id": "C-P9999-9"})]},
        {"reasoning": "Let me think without a tool.", "content": "Thinking it over."},
        {"reasoning": "Compute.", "calls": [("calculator", {"expressions": ["1+1", "round(2.5*3, 2)"]}),
                                            ("get_fund", {"product_ids": ["C-P0001", "C-P0002"]})]},
        lambda m: {"reasoning": "Done.", "calls": [("set_checklist", {"items": ["look up", "answer"], "done": [1, 2]}),
                                                   ("finish", {"answer": "done"})]},
    ]


def results_tail(messages):
    """The text the template renders after the last completion: the newline after its <|im_end|>, then the results of
    the last turn's calls (the finish result last) as one user turn."""
    k = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    return ("\n<|im_start|>user" + "".join(f"\n<tool_response>\n{m['content']}\n</tool_response>" for m in messages[k + 1:])
            + "<|im_end|>\n")


def run(task, db, steps, cfg, template):
    stub = qwen_stub.QwenStub(by_turn(steps), TOK, template)
    row, traj = asyncio.run(agent.run_one(task, cfg, db, stub, "stub", 1, agent.Settings(provider="qwen")))
    return row, json.loads(json.dumps(traj)), stub


def prefix_breaks(stub):
    """Turns k whose prompt does not start with prompt k-1 followed by completion k-1."""
    p, c = stub.prompts, stub.completions
    return [k for k in range(1, len(p)) if p[k][:len(p[k - 1]) + len(c[k - 1])] != p[k - 1] + c[k - 1]]


def cases():
    return [("market H1", first(market_tasks.path(MARKET_DB)), MARKET_DB, market_steps(True), "H1"),
            ("market H0", first(market_tasks.path(MARKET_DB)), MARKET_DB, market_steps(False), "H0"),
            ("wealth H1", first(f"{WEALTH}/tasks_test.jsonl"), f"{WEALTH}/C.sqlite", wealth_steps(), "H1")]


@unittest.skipIf(TOK is None, "transformers or the Qwen3.5 tokenizer is not available")
class TestTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(NEW, encoding="utf-8") as f:
            cls.new = f.read()
        cls.native = TOK.chat_template  # the template the tokenizer ships, which serve_4b.sh serves

    def test_one_user_message_renders_as_native(self):
        """Where there is one user message (every trajectory A1 was trained on), the two templates agree."""
        task = first(market_tasks.path(MARKET_DB))
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": task["prompt"]},
                {"role": "assistant", "content": "", "reasoning_content": "r1",
                 "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"a": [1, "x"], "b": True}}}]},
                {"role": "tool", "content": "{}"}, {"role": "assistant", "content": "ok", "reasoning_content": "r2"}]
        for gen in (False, True):
            self.assertEqual(TOK.apply_chat_template(msgs, chat_template=self.new, tokenize=False, add_generation_prompt=gen),
                             TOK.apply_chat_template(msgs, chat_template=self.native, tokenize=False, add_generation_prompt=gen))
        nudged = msgs + [{"role": "user", "content": agent.NUDGE}]
        text = TOK.apply_chat_template(nudged, chat_template=self.new, tokenize=False, add_generation_prompt=True)
        self.assertEqual(text.count("<think>\nr1\n</think>"), 1)  # kept after the nudge; the native template drops it
        self.assertNotIn("r1", TOK.apply_chat_template(nudged, chat_template=self.native, tokenize=False,
                                                       add_generation_prompt=True))

    def test_prefix_holds_with_nudges(self):
        """(a) Every prompt begins with the previous prompt and completion, as the harness's prefix_ok also says."""
        for name, task, db, steps, cfg in cases():
            with self.subTest(case=name):
                row, traj, stub = run(task, db, steps, cfg, self.new)
                self.assertEqual(row["end"], "finished")
                users = sum(m["role"] == "user" for m in traj["messages"])
                self.assertGreaterEqual(users, 2)  # nudged at least once
                self.assertGreaterEqual(row["tool_errors"], 1)
                self.assertEqual(prefix_breaks(stub), [])
                self.assertTrue(all(t["prefix_ok"] for t in traj["turns"][1:]))
                self.assertEqual(row["token_prefix_breaks"], 0)

    def test_native_template_breaks_at_the_nudge(self):
        row, traj, stub = run(*cases()[0][1:4], "H1", self.native)
        msgs = traj["messages"]
        nudges = [i for i, m in enumerate(msgs) if m["role"] == "user"][1:]
        # the first prompt after the first nudge is the first that breaks
        turn_after = sum(m["role"] == "assistant" for m in msgs[:nudges[0]])
        self.assertEqual(prefix_breaks(stub)[0], turn_after)
        self.assertGreater(row["token_prefix_breaks"], 0)

    @unittest.skipUnless(os.path.exists(RL_DATA), "src/03_model-training/utils/rl_data.py is not here")
    def test_training_encoder_matches_sampling(self):
        """(b) rl_data.py's encode on the whole trajectory gives the sampled ids when the tokenizer carries the v3
        template: the last prompt and completion, then the last turn's results; the native template it ships does not."""
        encode, norm = qwen_stub.training_encoder(RL_DATA, TOK)
        for name, task, db, steps, cfg in cases():
            with self.subTest(case=name):
                row, traj, stub = run(task, db, steps, cfg, self.new)
                msgs = [norm(m) for m in traj["messages"]]
                sampled = stub.prompts[-1] + stub.completions[-1]
                saved = TOK.chat_template
                try:
                    TOK.chat_template = self.new
                    ids, labels = encode(msgs, traj["tools"])
                    TOK.chat_template = self.native
                    native_ids, _ = encode(msgs, traj["tools"])
                finally:
                    TOK.chat_template = saved
                tail = TOK(results_tail(traj["messages"]), add_special_tokens=False).input_ids
                self.assertEqual(ids, sampled + tail)
                self.assertNotEqual(native_ids[:len(sampled)], sampled)
                # the stored records are the sampled tokens
                for t, c in zip(traj["turns"], stub.completions):
                    self.assertEqual(tokens.unpack_ids(t["token_ids"]), c)
                # every sampled completion carries the loss; so does the "<think>\n" that ends each prompt
                think = TOK("<think>\n", add_special_tokens=False).input_ids
                want = set()
                for p, c in zip(stub.prompts, stub.completions):
                    self.assertEqual(p[-len(think):], think)
                    want |= set(range(len(p) - len(think), len(p) + len(c)))
                self.assertEqual({i for i, x in enumerate(labels) if x != -100}, want)


if __name__ == "__main__":
    unittest.main()
