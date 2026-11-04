"""Unit tests of harness v3 with the stub client (tests/stub.py); no model server is used.

Run: python -B -m unittest discover -s <v3>/tests -t <v3>   (the wealth and DocFinQA data under
$FP_DATA_EN/harness and the composite tasks of harness_v2 must be readable)
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest

from harness import agent, tokens
from harness import run as run_mod
from harness.envs import base, market
from harness.envs.registry import KINDS
from harness.envs.wealth import core
from tests import market_tasks
from tests.stub import Stub, scripted, tool_results

D = os.environ.get("FP_DATA_EN", "<FP_DATA_EN unset>") + "/harness"
H = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/harness_v2"
V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="harness_v3_test_")
MARKET_DB = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1/market_v1.sqlite"
MARKET_GOOD = ("get_prices", {"symbol": "AAPL", "start": "2020-08-28", "end": "2020-08-28", "fields": ["close"]})


def market_task():
    """The first hand-written market task, AAPL's close on 2020-08-28 (keys read from the snapshot)."""
    return first(market_tasks.path(MARKET_DB))


def market_answer():
    raw, split = (k["value"] for k in market_task()["keys"])
    return f"{raw:.2f} unadjusted, {split:.2f} split-adjusted"


def first(path):
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def finish(answer="done"):
    return {"calls": [("finish", {"answer": answer})]}


def finisher(answer="done"):
    """Finish; after a rejection, mark the checklist done and finish again."""
    def step(messages):
        res = tool_results(messages)
        if res and res[-1].get("status") == "rejected":
            return {"reasoning": "fix it", "calls": [("set_checklist", {"items": ["a"], "done": [1]}), ("finish", {"answer": answer})]}
        return {"reasoning": "report", "calls": [("finish", {"answer": answer})]}
    return step


# one task per environment kind, with a call the environment refuses (raising ToolError) and a valid call
CASES = {
    "wealth": (lambda: (first(f"{D}/tasks_test.jsonl"), f"{D}/C.sqlite"),
               ("get_document", {"doc_id": "C-P9999-9"}), ("get_customer", {"customer_id": "C-C1613"})),
    "t5_review_switch": (lambda: (first(f"{H}/composite/tasks_t5_dev.jsonl"), f"{D}/B.sqlite"),
                         ("redeem", {"customer_id": "B-C0911", "product_id": "B-P9999", "units": 1}),
                         ("list_holdings", {"customer_id": "B-C0911"})),
    "ml": (lambda: (first(f"{D}/tasks_ml_dev.jsonl"), None), ("ml_column_stats", {"column": "no_such_column"}),
           ("ml_describe_data", {})),
    "docfinqa": (lambda: (first(f"{D}/tasks_docfinqa_dev.jsonl"), f"{D}/docfinqa_dev"),
                 ("read_chunk", {"chunk_id": 10 ** 9}), ("search_document", {"query": "revenue"})),
    "report_gate": (lambda: (first(f"{H}/composite/tasks_t6_dev.jsonl"), f"{D}/B.sqlite"),
                    ("get_document", {"doc_id": "B-P9999-9"}), ("read_chunk", {"chunk_id": 10 ** 9})),
    "market": (lambda: (market_task(), MARKET_DB),
               ("get_prices", {"symbol": "NOPE", "start": "2020-01-02", "end": "2020-01-03"}), MARKET_GOOD),
}


def episode(task, db, policy, cfg="H1", stub=None, **settings):
    stub = stub or Stub(policy)
    s = agent.Settings(**settings)
    row, traj = asyncio.run(agent.run_one(task, cfg, db, stub, "stub", 1, s))
    return row, traj, stub


def standard_policy(bad, good):
    return scripted([
        {"reasoning": "plan", "calls": [("set_checklist", {"items": ["look", "answer"]})]},
        {"reasoning": "try", "calls": [bad]},
        {"content": "Let me think about this without a tool.", "reasoning": "hmm"},
        {"reasoning": "look", "calls": [good, ("calculator", {"expressions": ["1+1"]})]},
        finisher(), finisher(), finisher(),
    ])


class TestPackage(unittest.TestCase):
    def test_one_module_each(self):
        """Every environment raises the one ToolError the loop catches, and no module is loaded under a bare name."""
        code = ("import sys; import harness.run, harness.golden, harness.collect; from harness.envs import base, ml, "
                "docfinqa, report_gate, market; from harness.envs.wealth import core; "
                "bare = sorted(m for m in sys.modules if m in ('core', 'grade', 'tasks', 'agent', 'docfinqa', 'ml_tasks', "
                "'report_gate', 'hint', 'golden', 'env')); "
                "same = core.ToolError is base.ToolError is ml.ToolError is docfinqa.ToolError is market.ToolError; "
                "print(bare, same)")
        out = subprocess.run([sys.executable, "-B", "-c", code], cwd=V3, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "[] True")

    def test_no_path_tricks(self):
        for root, _, files in os.walk(os.path.join(V3, "harness")):
            for f in files:
                if f.endswith(".py"):
                    text = read(os.path.join(root, f))
                    self.assertNotIn("sys.path", text, f)
                    self.assertNotRegex(text, r"(?m)^\s*import (core|grade|tasks|agent)\b", f)


class TestPerKind(unittest.TestCase):
    """Prefix property and tool errors for every environment kind, under H0 and H1 and two providers."""

    def check_prefix(self, stub, traj):
        sent, final = stub.sent, traj["messages"]
        for k, msgs in enumerate(sent):
            nxt = sent[k + 1] if k + 1 < len(sent) else final
            self.assertEqual(msgs, nxt[:len(msgs)], f"request {k} is not a prefix of the next")
            self.assertEqual(msgs, final[:len(msgs)], f"request {k} is not a prefix of the trajectory")
        self.assertTrue(all(r["tools"] == stub.requests[0]["tools"] for r in stub.requests))
        self.assertEqual(traj["tools"], stub.requests[0]["tools"])

    def test_kinds(self):
        for name, (load, bad, good) in CASES.items():
            for cfg in ("H0", "H1"):
                for provider in ("vllm", "qwen"):
                    with self.subTest(kind=name, cfg=cfg, provider=provider):
                        task, db = load()
                        row, traj, stub = episode(task, db, standard_policy(bad, good), cfg, provider=provider)
                        self.check_prefix(stub, traj)
                        self.assertEqual(row["end"], "finished")
                        self.assertEqual(row["harness"], "v3")
                        msgs = traj["messages"]
                        # the refused call came back as a tool error, and the model was asked again after it
                        i = next(i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("tool_calls")
                                 and m["tool_calls"][0]["function"]["name"] == bad[0])
                        err = json.JSONDecoder().raw_decode(msgs[i + 1]["content"])[0]
                        self.assertIn("error", err)
                        self.assertTrue(any(m["role"] == "assistant" for m in msgs[i + 2:]))
                        self.assertGreaterEqual(row["tool_errors"], 1)
                        self.assertEqual(row["internal_errors"], 0)
                        # reasoning goes into the message only for a provider that takes it back
                        has = any("reasoning_content" in m for m in msgs)
                        self.assertEqual(has, provider == "qwen")
                        if provider == "vllm":
                            self.assertTrue(any("reasoning" in t for t in traj["turns"]))
                        # the reply written as text got a nudge, not a rewrite
                        j = next(i for i, m in enumerate(msgs) if m["role"] == "assistant" and not m.get("tool_calls"))
                        self.assertEqual(msgs[j]["content"], "Let me think about this without a tool.")
                        self.assertEqual(msgs[j + 1]["role"], "user")
                        self.assertTrue(all(t.get("prefix_ok", True) for t in traj["turns"]))

    def test_internal_error(self):
        """Any other exception from a tool is a tool error for the model and an internal error in the row."""
        task, db = CASES["market"][0]()
        orig = market.MarketEnv.get_prices
        market.MarketEnv.get_prices = lambda self, **kw: {}["boom"]
        try:
            row, traj, _ = episode(task, db, scripted([{"calls": [MARKET_GOOD]}]))
        finally:
            market.MarketEnv.get_prices = orig
        self.assertEqual(row["end"], "finished")
        self.assertEqual(row["internal_errors"], 1)
        self.assertIn("KeyError", traj["messages"][3]["content"])

    def test_text_call_kept(self):
        """A reply whose text is a tool's arguments is neither executed nor rewritten."""
        task, db = CASES["market"][0]()
        text = json.dumps({"symbol": "AAPL", "date": "2020-08-29"})
        row, traj, stub = episode(task, db, scripted([{"content": text}]), "H0")
        self.assertEqual(traj["messages"][2], {"role": "assistant", "content": text})
        self.assertIn("trading_day", traj["messages"][3]["content"])
        self.assertEqual(row["text_calls"], 1)
        self.check_prefix(stub, traj)

    def test_renamed_tool_kept(self):
        task, db = CASES["market"][0]()
        row, traj, stub = episode(task, db, scripted([{"calls": [("get_prices<|channel|>commentary", MARKET_GOOD[1])]}]),
                                  "H0")
        self.assertEqual(traj["messages"][2]["tool_calls"][0]["function"]["name"], "get_prices<|channel|>commentary")
        self.assertIn(f'["2020-08-28", {market_task()["keys"][0]["value"]}]', traj["messages"][3]["content"])
        self.assertEqual(row["renamed_tools"], 1)

    def test_cap(self):
        task, db = CASES["market"][0]()
        row, traj, stub = episode(task, db, scripted([{"calls": [MARKET_GOOD]}]), "H0", tool_result_chars=20)
        full = json.dumps(market.MarketEnv(db).get_prices(**MARKET_GOOD[1]))
        self.assertEqual(traj["messages"][3]["content"], full[:20] + agent.CUT_NOTE.format(n=len(full), cap=20))
        self.assertEqual(row["cut_results"], 1)


class TestSteps(unittest.TestCase):
    def setUp(self):
        self.task, self.db = CASES["market"][0]()
        self.good = MARKET_GOOD

    def test_checklist_and_redo_turns_are_not_task_steps(self):
        checklist = {"calls": [("set_checklist", {"items": ["a"]})]}
        steps = [checklist] * 25 + [{"calls": [self.good]}] * 3
        steps += [finish("123456.78"), {"calls": [self.good]}, {"calls": [self.good]}, finish("123456.78"), finish("123456.78")]
        row, traj, _ = episode(self.task, self.db, scripted(steps), "H1", max_steps=10)
        self.assertEqual(row["end"], "finished")
        self.assertEqual((row["checklist_steps"], row["task_steps"], row["redo_steps"]), (20, 8, 5))
        self.assertEqual(row["verification_rejects"], 2)
        msgs = traj["messages"]
        self.assertIn("at most 20 times", msgs[2 * 21 + 1]["content"])  # the 21st set_checklist call is refused
        self.assertEqual([t["step"] for t in traj["turns"]][19:22], ["checklist", "task", "task"])

    def test_step_limit(self):
        row, traj, _ = episode(self.task, self.db, scripted([{"calls": [self.good]}] * 50), "H1", max_steps=4)
        self.assertEqual((row["end"], row["task_steps"], row["steps"]), ("step_limit", 4, 4))

    def test_checklist_forever_ends(self):
        s = agent.Settings(max_steps=3)
        row, traj, _ = episode(self.task, self.db, scripted([{"calls": [("set_checklist", {"items": ["a"]})]}] * 100),
                               "H1", max_steps=3)
        self.assertEqual((row["end"], row["checklist_steps"], row["task_steps"]), ("step_limit", 20, 3))
        self.assertLessEqual(row["steps"], s.hard_cap())

    def test_h0_has_no_free_turns(self):
        row, _, _ = episode(self.task, self.db, scripted([{"calls": [("set_checklist", {"items": ["a"]})]}] * 10), "H0",
                            max_steps=3)
        self.assertEqual((row["end"], row["checklist_steps"], row["task_steps"]), ("step_limit", 0, 3))


class TestTokens(unittest.TestCase):
    def setUp(self):
        self.task, self.db = CASES["market"][0]()

    def test_records(self):
        policy = scripted([{"reasoning": "look it up", "calls": [MARKET_GOOD]},
                           {"reasoning": "answer", "calls": [("finish", {"answer": market_answer()})]}])
        row, traj, stub = episode(self.task, self.db, policy, "H0", provider="qwen")
        self.assertTrue(row["passed"])
        self.assertTrue(row["token_records"])
        traj = json.loads(json.dumps(traj))  # as stored in trajectories.jsonl
        for t, (content, calls, _, ids), msgs in zip(traj["turns"], stub.produced, stub.sent):
            self.assertEqual(tokens.unpack_ids(t["token_ids"]), ids)
            self.assertEqual(tokens.unpack_logprobs(t["logprobs"]), [-(j % 4) * 0.25 for j in range(len(ids))])
            self.assertEqual(t["n_prompt"], len(stub.render(msgs)))
            self.assertEqual(t["n_completion"], len(ids))
        self.assertNotIn("prefix_ok", traj["turns"][0])
        self.assertTrue(traj["turns"][1]["prefix_ok"])
        self.assertEqual(stub.requests[0]["logprobs"], True)
        self.assertEqual(stub.requests[0]["extra_body"]["return_token_ids"], True)
        self.assertTrue(traj["sampling"]["logprobs"])

    def test_prefix_break_is_recorded(self):
        """A server whose template drops earlier reasoning after a user message breaks the token prefix there."""
        policy = scripted([{"reasoning": "look", "calls": [MARKET_GOOD]},
                           {"reasoning": "hmm", "content": "thinking aloud"},
                           finish("99.5")])
        row, traj, _ = episode(self.task, self.db, policy, "H0", provider="qwen",
                               stub=Stub(policy, drop_reasoning=True))
        self.assertEqual([t.get("prefix_ok") for t in traj["turns"]], [None, True, False])
        self.assertEqual(row["token_prefix_breaks"], 1)

    def test_server_without_token_options(self):
        policy = scripted([finish("99.5")])
        row, traj, stub = episode(self.task, self.db, policy, "H0", provider="qwen",
                                  stub=Stub(policy, refuse_token_options=True))
        self.assertEqual(row["end"], "finished")
        self.assertFalse(row["token_records"])
        self.assertNotIn("logprobs", stub.requests[0])

    def test_no_token_records_option(self):
        policy = scripted([finish("99.5")])
        row, traj, stub = episode(self.task, self.db, policy, "H0", token_records=False)
        self.assertNotIn("logprobs", stub.requests[0])
        self.assertNotIn("token_ids", traj["turns"][0])


class TestRun(unittest.TestCase):
    def tasks(self, n=3):
        t = market_task()
        path = os.path.join(TMP, f"tasks_{n}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps(dict(t, task_id=f"market-hand-{i:04d}")) + "\n")
        return path

    def policy(self, messages):
        if not tool_results(messages):
            return {"calls": [MARKET_GOOD]}
        return {"calls": [("finish", {"answer": market_answer()})]}

    def test_resume(self):
        out = os.path.join(TMP, "run_resume")
        argv = ["--tasks", self.tasks(), "--db", MARKET_DB, "--config", "H1", "--runs", "2", "--out", out]
        n = run_mod.main(argv, clients=[Stub(self.policy)])
        self.assertEqual(n, {"done": 6, "passed": 6})
        res = os.path.join(out, "results.jsonl")
        rows = read_jsonl(res)
        self.assertTrue(all(r["harness"] == "v3" and r["passed"] for r in rows))
        trajs = read_jsonl(os.path.join(out, "trajectories.jsonl"))
        self.assertEqual(len(trajs), 6)
        self.assertTrue(all(t["turns"] and "token_ids" in t["turns"][0] for t in trajs))
        rows[0]["end"], rows[3]["end"], rows[4]["end"] = "api_error", "budget", "step_limit"
        with open(res, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = run_mod.main(argv, clients=[Stub(self.policy)])
        self.assertEqual(n["done"], 2)  # the api_error and budget episodes again; step_limit stays
        again = read_jsonl(res)[6:]
        self.assertEqual({(r["task_id"], r["run"]) for r in again},
                         {(rows[0]["task_id"], rows[0]["run"]), (rows[3]["task_id"], rows[3]["run"])})
        self.assertEqual(run_mod.main(argv, clients=[Stub(self.policy)])["done"], 0)

    def test_refuses_v2_rows(self):
        out = os.path.join(TMP, "run_v2rows")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "results.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"task_id": "x", "run": 1, "end": "finished"}) + "\n")
        with self.assertRaises(SystemExit):
            run_mod.main(["--tasks", self.tasks(), "--db", MARKET_DB, "--config", "H0", "--runs", "1", "--out", out],
                         clients=[Stub(self.policy)])

    def test_harness_error_row(self):
        """An episode that raises becomes a row with end "harness_error"; the others still run."""
        out = os.path.join(TMP, "run_error")
        path = os.path.join(TMP, "tasks_bad.jsonl")
        t = market_task()
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(dict(t, task_id="bad", env="no_such_kind")) + "\n")
            f.write(json.dumps(t) + "\n")
        n = run_mod.main(["--tasks", path, "--db", MARKET_DB, "--config", "H0", "--runs", "1", "--out", out],
                         clients=[Stub(self.policy)])
        rows = {r["task_id"]: r for r in read_jsonl(os.path.join(out, "results.jsonl"))}
        self.assertEqual(n["done"], 2)
        self.assertEqual(rows["bad"]["end"], "harness_error")
        self.assertTrue(rows[t["task_id"]]["passed"])


class TestGrading(unittest.TestCase):
    def test_market_keys(self):
        from harness.envs.wealth import grade
        keys = [{"kind": "number", "value": 99.5, "tol": 0.005}, {"kind": "date", "value": "2020-01-02"},
                {"kind": "ticker", "value": "DEMO"}]
        self.assertTrue(grade.grade_keys(keys, "DEMO closed at $99.50 on Jan 3; it was 101.25 on January 2, 2020.")["passed"])
        self.assertTrue(grade.grade_keys(keys, "DEMO: 99.504 (2020‑01‑02)")["passed"])
        self.assertFalse(grade.grade_keys(keys, "DEMO: 99.6 (2020-01-02)")["passed"])
        self.assertFalse(grade.grade_keys(keys, "DEMOX: 99.5 (2020-01-02)")["passed"])
        self.assertFalse(grade.grade_keys(keys, "DEMO: 99.5 (2020-01-03)")["passed"])
        for answer in ("DEMO closed at $99.50 on Jan 3; it was 101.25 on January 2, 2020.", "DEMO: 99.6 (2020-01-02)",
                       "DEMOX: 99.5 (2020-01-02)", None):  # the market kind's grader agrees where keys have no extras
            self.assertEqual(market.grade_keys(keys, answer), grade.grade_keys(keys, answer))
        s = [{"kind": "set", "value": ["AAPL", {"kind": "number", "value": 3, "rel_tol": 0.1}]}]
        self.assertTrue(grade.grade_keys(s, "AAPL, 3.2")["passed"])
        self.assertFalse(grade.grade_keys(s, "AAPL, 3.4")["passed"])
        self.assertFalse(grade.grade_keys(keys, None)["passed"])

    def test_registry(self):
        self.assertEqual(sorted(KINDS), ["docfinqa", "market", "ml", "report_gate", "wealth"])
        for k in KINDS.values():
            names = [t["function"]["name"] for t in k.TOOLS]
            self.assertIn("finish", names, k.name)
            self.assertTrue(set(k.EXAMPLES) <= set(names), k.name)
        self.assertIs(core.ToolError, base.ToolError)


if __name__ == "__main__":
    unittest.main()
