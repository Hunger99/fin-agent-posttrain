"""Unit tests of form_errors.py: which tool results count as form errors, which call arguments the builders cannot
render, how results are paired with calls, and how the turn flags are combined with a builder's own
(research/log/2026-11-01-0010.md, 20:2x UTC). The result texts are written as the two harnesses write them, json.dumps
of the error object followed by the checklist; the run_python, ToolError and checklist texts are shaped after those in
the collected trajectories, and the cut result after v3's CUT_NOTE.

Run: python -B src/03_model-training/tests/test_form_errors.py   (standard library only)
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
from form_errors import call_args, form_error, form_error_turns, mask_form_errors  # noqa: E402

CHECKLIST = "\n[checklist] none yet; write one with set_checklist."
HINT = "\nNote: this exact call has now been made 3 or more times; try a different approach."


def err(message, example=None):
    """A tool error as both harnesses return it (with correction on, the example of a valid call follows)."""
    e = {"error": message}
    if example is not None:
        e["example_of_a_valid_call"] = example
    return json.dumps(e) + CHECKLIST


def turn(*calls):
    """An assistant message with the given (name, arguments) calls, ids c0, c1, ..."""
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a) if isinstance(a, dict) else a}}
        for i, (n, a) in enumerate(calls)]}


def results(*texts, ids=True):
    return [dict({"role": "tool", "content": t}, **({"tool_call_id": f"c{i}"} if ids else {})) for i, t in enumerate(texts)]


class TestFormError(unittest.TestCase):
    def test_unknown_tool(self):
        for name in ("set_checklook", "redemption", "checklist"):
            with self.subTest(name=name):
                self.assertEqual(form_error(json.dumps({"error": f"unknown tool {name}"}) + CHECKLIST, name), "unknown_tool")
        self.assertEqual(form_error(json.dumps({"error": "unknown tool redem"}) + HINT + CHECKLIST, "redem"), "unknown_tool")
        self.assertEqual(form_error(" " + json.dumps({"error": "unknown tool x"}), "x"), "unknown_tool")

    def test_arguments_not_a_json_object(self):
        self.assertEqual(form_error(json.dumps({"error": "arguments are not a JSON object"}), "get_prices"), "args_not_json")
        self.assertEqual(form_error(json.dumps({"error": "arguments are not a JSON object"}) + CHECKLIST, None),
                         "args_not_json")

    def test_arguments_that_do_not_fit(self):
        cases = [("get_corporate_actions", "MarketEnv.get_corporate_actions() got an unexpected keyword argument 'type'"),
                 ("ml_train", "MLEnv.ml_train() got an unexpected keyword argument 'k'"),
                 ("calculator", "Env.calculator() missing 1 required positional argument: 'expressions'"),
                 ("get_prices", "MarketEnv.get_prices() missing 2 required positional arguments: 'start' and 'end'"),
                 ("f", "E.f() missing 1 required keyword-only argument: 'k'"),
                 ("get_fund", "Env.get_fund() got multiple values for argument 'self'"),
                 ("g", "E.g() got some positional-only arguments passed as keyword arguments: 'a'"),
                 ("get_prices", "get_prices() got an unexpected keyword argument 'begin'"),  # without the class
                 # the direct call of run_python with the wrong argument name is a form error like any other tool's
                 ("run_python", "MarketEnv.run_python() got an unexpected keyword argument 'items'"),
                 ("run_python", "MarketEnv.run_python() missing 1 required positional argument: 'code'"),
                 # a name with a harmony suffix is executed, and named in the message, as the tool it starts with
                 ("get_prices<|channel|>commentary", "MarketEnv.get_prices() got an unexpected keyword argument 'x'"),
                 # the qualified name of a class defined inside a function
                 ("get_prices", "make_env.<locals>.Env.get_prices() got an unexpected keyword argument 'x'")]
        for name, message in cases:
            with self.subTest(message=message):
                self.assertEqual(form_error(err(message, {name: {}}), name), "bad_args")
                self.assertEqual(form_error(err(message), name), "bad_args")  # H0: no example

    def test_errors_from_running_a_well_formed_call(self):
        sandbox = ("the code raised an error:\nTraceback (most recent call last):\n  File \"<run_python>\", line 11, "
                   "in <module>\nTypeError: Rolling.apply() got an unexpected keyword argument 'axis'")
        cases = [("run_python", sandbox),
                 ("run_python", "the code raised an error:\nTraceback (most recent call last):\n  File \"<run_python>\","
                                " line 3, in <module>\nTypeError: f() missing 1 required positional argument: 'x'"),
                 ("run_python", "the code ran longer than 30 seconds and was stopped; load less data or avoid slow loops"),
                 ("get_prices", "unknown symbol 'NOPE' Find symbols with search_securities."),
                 ("search_securities", "no security matches 'Apple Inc'; try a ticker, fewer words, or another kind"),
                 ("ml_train", "missing values in ['age']; set impute to mean or median"),
                 ("ml_train", "unknown or no features: ['x']"),
                 ("set_checklist", "'int' object is not iterable"),  # a value the harness cannot read, not a binding
                 ("get_prices", "the tool failed (KeyError: 'close')"),
                 # a TypeError binding some other function inside the tool, not the tool's own parameters
                 ("get_prices", "MarketEnv._page() got an unexpected keyword argument 'size'"),
                 ("get_prices", "MarketEnv.get_prices_v2() got an unexpected keyword argument 'x'"),
                 ("get_prices", "make_env.<locals>.Env._page() got an unexpected keyword argument 'size'")]
        for name, message in cases:
            with self.subTest(message=message[:60]):
                self.assertIsNone(form_error(err(message, {name: {}}), name))

    def test_only_the_leading_error_object_counts(self):
        cases = ['{"rows": [["unknown tool", 1]], "note": "{\\"error\\": \\"unknown tool x\\"}"}',
                 json.dumps({"stdout": json.dumps({"error": "unknown tool x"})}),
                 json.dumps({"text": "arguments are not a JSON object"}),
                 '[cut] {"error": "unknown tool x"}',
                 json.dumps([{"error": "unknown tool x"}]),
                 json.dumps({"status": "rejected", "verification_failed": ["no checklist was written"]}),
                 json.dumps({"status": "ok"}), "", None]
        for text in cases:
            with self.subTest(text=str(text)[:60]):
                self.assertIsNone(form_error(text, "x"))
        # an escaped quote inside the message is read as part of it
        self.assertIsNone(form_error(json.dumps({"error": 'no security matches "unknown tool"'}), "search_securities"))
        self.assertEqual(form_error(json.dumps({"error": 'unknown tool a"b'}), 'a"b'), "unknown_tool")

    def test_the_leading_object_is_decoded_whole(self):
        """Any JSON layout of the leading object is read; a result cut off inside it falls back to its first key."""
        self.assertEqual(form_error(json.dumps({"error": "unknown tool x"}, indent=1) + CHECKLIST, "x"), "unknown_tool")
        full = err("MarketEnv.get_prices() got an unexpected keyword argument 'begin'", {"get_prices": {"q": "x" * 500}})
        cut = full[:200] + "\n[cut: the result has 1,000 characters and only the first 200 are shown; ask for less]"
        self.assertRaises(ValueError, json.loads, cut)
        self.assertEqual(form_error(cut, "get_prices"), "bad_args")
        self.assertIsNone(form_error('{"error": "unknown tool x' + "\n[cut: ...]", "x"))  # cut inside the message

    def test_write_schema_rejection(self):
        """A write rejected for a missing or unexpected key is bad_args; one rejected for a rule or a value is not."""
        def rejected(**out):
            return json.dumps({"status": "rejected", **out}) + CHECKLIST
        for message in ("missing ['amount'] / unexpected ['amt']", "missing [] / unexpected ['amt']",
                        "missing ['customer_id', 'product_id'] / unexpected []"):
            with self.subTest(message=message):
                self.assertEqual(form_error(rejected(error=message), "subscribe"), "bad_args")
                self.assertEqual(form_error(rejected(error=message) + HINT, "redeem"), "bad_args")
        cases = [rejected(error="missing [] / unexpected []"), rejected(error="amount must be a positive number"),
                 rejected(error="units must be above 0"),
                 rejected(error="missing ['amount'] / unexpected ['amt'] and more"),
                 rejected(violated_rules=[2], reasons=["over 30% of total assets"]),
                 rejected(verification_failed=["no checklist was written"]), rejected(verification_failed=True),
                 json.dumps({"status": "ok", "error": "missing ['amount'] / unexpected []"}),
                 json.dumps({"error": "missing ['amount'] / unexpected []"})]
        for text in cases:
            with self.subTest(text=text[:60]):
                self.assertIsNone(form_error(text, "subscribe"))


class TestCallArgs(unittest.TestCase):
    def test_objects_and_the_rest(self):
        call = lambda args: {"function": {"name": "f", "arguments": args}}  # noqa: E731
        self.assertEqual(call_args(call('{"symbol": "AAPL"}')), {"symbol": "AAPL"})
        self.assertEqual(call_args(call("")), {})  # both harnesses read an empty string as no arguments
        self.assertEqual(call_args(call(None)), {})
        for args in ('{"symbol": "AAPL"', "[1, 2]", '"AAPL"', "3", "null", "{'symbol': 'AAPL'}"):
            with self.subTest(args=args):
                self.assertIsNone(call_args(call(args)))


class TestTurns(unittest.TestCase):
    def msgs(self, ids=True):
        """Six assistant turns: fine; unknown tool beside a fine call; a run_python traceback; bad arguments to
        get_prices; a reply without a call (nudged); finish."""
        return ([{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                 turn(("set_checklist", {"items": ["a"]})), *results(json.dumps({"status": "ok"}) + CHECKLIST, ids=ids),
                 turn(("get_prices", {"symbol": "AAPL", "start": "2020-01-02", "end": "2020-01-03"}), ("checklist", {})),
                 *results(json.dumps({"rows": [1]}), json.dumps({"error": "unknown tool checklist"}), ids=ids),
                 turn(("run_python", {"code": "x.apply(axis=1)"})),
                 *results(err("the code raised an error:\nTraceback (most recent call last):\nTypeError: Rolling.apply() "
                              "got an unexpected keyword argument 'axis'", {"run_python": {"code": "..."}}), ids=ids),
                 turn(("get_prices", {"symbol": "AAPL", "begin": "2020-01-02"})),
                 *results(err("MarketEnv.get_prices() got an unexpected keyword argument 'begin'", {}), ids=ids),
                 {"role": "assistant", "content": "The close was 123.45."}, {"role": "user", "content": "Use the tools."},
                 turn(("finish", {"answer": "123.45"})), *results(json.dumps({"status": "ok"}), ids=ids)])

    def test_kinds_per_turn(self):
        want = [[], ["unknown_tool"], [], ["bad_args"], [], []]
        self.assertEqual(form_error_turns(self.msgs()), want)
        self.assertEqual(form_error_turns(self.msgs(ids=False)), want)  # paired by position

    def test_pairing_by_id(self):
        """The name checked against a TypeError message is the one of the call the result answers."""
        m = [{"role": "user", "content": "u"}, turn(("get_fund", {"x": 1}), ("calculator", {"expressions": ["1+1"]})),
             {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"results": [2]})},
             {"role": "tool", "tool_call_id": "c0", "content": err("Env.get_fund() got an unexpected keyword argument 'x'")}]
        self.assertEqual(form_error_turns(m), [["bad_args"]])
        m[3]["tool_call_id"] = "c1"  # now said to answer calculator, whose name the message does not give
        self.assertEqual(form_error_turns(m), [[]])

    def test_mask(self):
        msgs = self.msgs()
        self.assertEqual(mask_form_errors(msgs), ([True, False, True, False, True, True], 2))
        # with a builder's own flags (the nudged reply is off): only turns that were on are counted
        own = [True, True, True, True, False, True]
        self.assertEqual(mask_form_errors(msgs, own), ([True, False, True, False, False, True], 2))
        self.assertEqual(mask_form_errors(msgs, [True, False, True, True, False, True]),
                         ([True, False, True, False, False, True], 1))
        self.assertEqual(own, [True, True, True, True, False, True])  # not changed in place
        self.assertEqual(mask_form_errors(msgs[:3]), ([True], 0))


if __name__ == "__main__":
    unittest.main()
