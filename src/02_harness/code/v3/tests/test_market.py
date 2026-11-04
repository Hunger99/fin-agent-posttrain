"""Tests of the market environment kind against the market_v1 snapshot (read-only), with the stub client.

Run: python -B -m unittest tests.test_market (from <v3>, or discover as in test_v3.py); run_python needs
$FP_VENVS/mkt/bin/python (pandas, numpy).
"""
import asyncio
import json
import os
import sqlite3
import tempfile
import time
import unittest

from harness import agent
from harness.envs import market
from harness.envs.base import ToolError
from tests import market_tasks
from tests.stub import Stub

DB = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1/market_v1.sqlite"
CUTOFF = "2025-10-30"


def env():
    return market.MarketEnv(DB)


def size(x):
    return len(json.dumps(x))


class TestLookups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = env()

    def test_search_securities(self):
        r = self.e.search_securities("apple")
        self.assertEqual(r["results"][0]["symbol"], "AAPL")
        self.assertEqual(r["results"][0]["cik"], 320193)
        self.assertEqual(self.e.search_securities("AAPL")["results"][0]["symbol"], "AAPL")
        self.assertTrue(all(x["kind"] == "future" for x in self.e.search_securities("gold", kind="future")["results"]))
        self.assertEqual(self.e.search_securities("BRK.B")["results"][0]["symbol"], "BRK-B")
        for bad in ({"query": ""}, {"query": "apple", "kind": "bond"}, {"query": "zzzzqqq"}):
            with self.assertRaises(ToolError):
                self.e.search_securities(**bad)

    def test_aapl_split(self):
        want_raw, want_split = (k["value"] for k in market_tasks.hand_tasks(DB)[0]["keys"])  # read from the snapshot
        raw = self.e.get_prices("AAPL", "2020-08-28", "2020-08-28", fields=["close"])
        self.assertEqual(raw["rows"], [["2020-08-28", want_raw]])
        split = self.e.get_prices("AAPL", "2020-08-28", "2020-08-28", adjustment="split", fields=["close"])
        self.assertAlmostEqual(split["rows"][0][1], want_split, delta=0.005)
        self.assertAlmostEqual(split["rows"][0][1], raw["rows"][0][1] / 4, delta=0.01)  # the 4:1 split of 2020-08-31
        total = self.e.get_prices("AAPL", "2020-08-28", "2020-08-28", adjustment="total", fields=["close"])
        self.assertLess(total["rows"][0][1], split["rows"][0][1])  # dividends since then lower the total-return price
        last = [self.e.get_prices("AAPL", CUTOFF, CUTOFF, adjustment=a, fields=["close"])["rows"][0][1]
                for a in ("raw", "split", "total")]
        self.assertEqual(len(set(last)), 1)  # on the cutoff day all three agree (no split since)
        r = self.e.get_prices("AAPL", "2020-08-27", "2020-09-01")
        self.assertEqual(r["columns"], ["date", "open", "high", "low", "close", "volume"])
        self.assertEqual([x[0] for x in r["rows"]], ["2020-08-27", "2020-08-28", "2020-08-31", "2020-09-01"])
        self.assertIsInstance(r["rows"][0][5], int)
        with self.assertRaises(ToolError):
            self.e.get_prices("AAPL", "2020-08-28", "2020-08-28", adjustment="adjusted")
        with self.assertRaises(ToolError):
            self.e.get_prices("AAPL", "2020-08-28", "2020-08-28", fields=["adj_close", "vwap"])

    def test_monthly_and_weekly_bars(self):
        d = self.e.get_prices("^GSPC", "2022-01-01", "2022-02-28")["rows"]
        m = self.e.get_prices("^GSPC", "2022-01-01", "2022-02-28", interval="month")
        self.assertEqual(m["total_rows"], 2)
        jan = [x for x in d if x[0] < "2022-02-01"]
        bar = m["rows"][0]
        self.assertEqual(bar[0], jan[-1][0])  # labelled by the last trading day
        self.assertEqual(bar[1], jan[0][1])
        self.assertEqual(bar[2], max(x[2] for x in jan))
        self.assertEqual(bar[3], min(x[3] for x in jan))
        self.assertEqual(bar[4], jan[-1][4])
        self.assertAlmostEqual(bar[5], sum(x[5] for x in jan), delta=len(jan))
        self.assertNotIn("partial_bars", m)
        cut = self.e.get_prices("^GSPC", "2022-01-15", "2022-03-10", interval="month", fields=["close"])
        self.assertEqual(cut["partial_bars"], ["2022-01-31", "2022-03-10"])
        w = self.e.get_prices("^GSPC", "2022-01-03", "2022-01-14", interval="week", fields=["close"])
        self.assertEqual([x[0] for x in w["rows"]], ["2022-01-07", "2022-01-14"])

    def test_paging(self):
        one = self.e.get_prices("AAPL", "2000-01-01", CUTOFF, fields=["close"])
        self.assertEqual(len(one["rows"]), 200)
        self.assertEqual(one["total_rows"], 6497)
        self.assertEqual(one["pages"], 33)
        last = self.e.get_prices("AAPL", "2000-01-01", CUTOFF, fields=["close"], page=33)
        self.assertEqual(last["rows"][-1][0], CUTOFF)
        for sym in ("BTC-USD", "^N225", "AAPL"):
            r = self.e.get_prices(sym, "2015-01-01", CUTOFF)
            self.assertLess(size(r), 7500, sym)
        with self.assertRaises(ToolError):
            self.e.get_prices("AAPL", "2000-01-01", CUTOFF, fields=["close"], page=34)
        self.assertLess(size(self.e.get_financials("AAPL", "Assets")), 7500)
        self.assertLess(size(self.e.get_corporate_actions("HYG")), 7500)
        self.assertLess(size(self.e.get_fred("DGS10", "2000-01-01", CUTOFF)), 7500)
        self.assertLess(size(self.e.search_financial_tags("AAPL", "revenue")), 7500)
        self.assertLess(size(self.e.search_fred("rate")), 7500)

    def test_trading_day_weekend(self):
        r = self.e.trading_day("AAPL", "2020-08-29")  # a Saturday
        self.assertEqual((r["trading_day"], r["date_is_trading_day"]), ("2020-08-28", False))
        self.assertEqual(self.e.trading_day("AAPL", "2020-08-29", rule="on_or_after")["trading_day"], "2020-08-31")
        self.assertEqual(self.e.trading_day("AAPL", "2020-08-28", offset=1)["trading_day"], "2020-08-31")
        self.assertEqual(self.e.trading_day("AAPL", "2020-08-31", offset=-1)["trading_day"], "2020-08-28")
        self.assertEqual(self.e.count_trading_days("^GSPC", "2020-01-01", "2020-12-31")["count"], 253)
        with self.assertRaises(ToolError):
            self.e.trading_day("AAPL", "2020-08-29", rule="nearest")
        with self.assertRaises(ToolError):
            self.e.trading_day("AAPL", CUTOFF, offset=1)

    def test_corporate_actions(self):
        r = self.e.get_corporate_actions("AAPL", "2020-01-01", "2020-12-31")
        self.assertEqual(r["splits"]["rows"], [["2020-08-31", 4.0, "4:1", "split"]])
        divs = dict((d, (a, b)) for d, a, b in r["dividends"]["rows"])
        raw, adj = divs["2020-08-07"]
        self.assertAlmostEqual(raw, 4 * adj, places=9)  # before the split: the raw amount is 4 post-split shares'
        raw, adj = divs["2020-11-06"]
        self.assertEqual(raw, adj)
        s = self.e.get_corporate_actions("HON", kind="split")
        self.assertNotIn("dividends", s)
        self.assertEqual({x[3] for x in s["splits"]["rows"]}, {"adjustment"})

    def test_fred(self):
        r = self.e.get_fred("DGS10", "2023-10-19", "2023-10-19")
        with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as c:
            want = c.execute("select value_vintage from fred where series_id='DGS10' and date='2023-10-19'").fetchone()[0]
        self.assertEqual(r["rows"], [["2023-10-19", want]])
        self.assertEqual(r["vintage"], "as_of_cutoff")
        self.assertEqual(r["units"], "%")
        self.assertEqual(self.e.search_fred("10-year treasury")["results"][0]["series_id"], "DGS10")
        # GDP for 2025 Q3 was first published after the cutoff: only the latest vintage has it
        cut = self.e.get_fred("GDP", "2025-01-01", CUTOFF)
        late = self.e.get_fred("GDP", "2025-01-01", CUTOFF, vintage="latest")
        self.assertNotIn("2025-07-01", [x[0] for x in cut["rows"]])
        self.assertIn("2025-07-01", [x[0] for x in late["rows"]])
        with self.assertRaises(ToolError):
            self.e.get_fred("DGS100", "2023-10-19", "2023-10-19")

    def test_financials(self):
        r = self.e.get_financials("NVDA", "Goodwill", fiscal_year=2022, form="10-K")
        rows = {x[1]: x for x in r["rows"]}
        self.assertEqual(rows["2022-01-30"][3], 4349000000)
        self.assertEqual(rows["2022-01-30"][5:8], [2022, "FY", "10-K"])
        self.assertEqual(self.e.get_financials("nvidia", "us-gaap:Goodwill", fiscal_year=2022, form="10-K")["rows"],
                         r["rows"])
        dflt = self.e.get_financials("NVDA", "Goodwill")
        allf = self.e.get_financials("NVDA", "Goodwill", filing="all")
        self.assertGreater(allf["total_rows"], dflt["total_rows"])
        orig = self.e.get_financials("NVDA", "Goodwill", filing="original")
        self.assertEqual(orig["total_rows"], dflt["total_rows"])
        first = {(x[0], x[1]): x for x in orig["rows"]}[(None, "2022-01-30")]
        self.assertEqual(first[7:9], ["10-K", "2022-03-18"])  # first reported in the FY2022 10-K
        latest = {(x[0], x[1]): x for x in dflt["rows"]}[(None, "2022-01-30")]
        self.assertGreater(latest[8], first[8])  # repeated in later filings
        self.assertEqual(len(first[9]), 20)  # the accession number
        ends = []
        for p in range(1, dflt["pages"] + 1):
            ends += [(x[0], x[1], x[4]) for x in self.e.get_financials("NVDA", "Goodwill", page=p)["rows"]]
        self.assertEqual(len(ends), len(set(ends)))  # one row per period by default
        flows = self.e.get_financials("NVDA", "Revenues", fiscal_year=2022, form="10-K")["rows"]
        self.assertTrue(any(x[2] == 12 for x in flows))  # months of an annual flow
        tags = self.e.search_financial_tags("NVDA", "goodwill")["results"]
        self.assertEqual(tags[0]["tag"], "Goodwill")
        xom = self.e.get_financials("XOM", "Assets", fiscal_year=2024, form="10-K")  # filed by the predecessor
        self.assertEqual(len(xom["registrants"]), 2)
        self.assertTrue(xom["rows"])
        self.assertEqual(self.e.get_financials("TSM", "Revenue", form="20-F")["taxonomy"], "ifrs-full")
        for bad in ({"tag": "Goodwil"}, {"tag": "Goodwill", "period": "Q4"}, {"tag": "Goodwill", "form": "8-K"},
                    {"tag": "Goodwill", "filing": "first"}):
            with self.assertRaises(ToolError):
                self.e.get_financials("NVDA", **bad)

    def test_cutoff(self):
        with self.assertRaises(ToolError):
            self.e.get_prices("AAPL", "2025-10-31", "2025-12-31")
        r = self.e.get_prices("AAPL", "2025-10-20", "2026-06-30", fields=["close"])
        self.assertEqual(r["rows"][-1][0], CUTOFF)
        self.assertIn("after the snapshot's last day", r["note"])
        self.assertTrue(all(x[0] <= CUTOFF for x in self.e.get_fred("DGS10", "2025-10-01", "2027-01-01")["rows"]))
        with self.assertRaises(ToolError):
            self.e.trading_day("AAPL", "2025-11-03")
        with self.assertRaises(ToolError):
            self.e.count_trading_days("AAPL", "2025-11-01", "2025-11-30")
        a = self.e.get_financials("AAPL", "Assets", filing="all")
        for p in range(1, a["pages"] + 1):
            for x in self.e.get_financials("AAPL", "Assets", filing="all", page=p)["rows"]:
                self.assertLessEqual(x[1], CUTOFF)
                self.assertLessEqual(x[8], CUTOFF)
        out = self.e.run_python("print(data.get_prices('AAPL').index.max().date(), "
                                "data.get_fred('DGS10', '2025-01-01', '2030-01-01').index.max().date())")
        prices_end, fred_end = out["stdout"].split()
        self.assertEqual(prices_end, CUTOFF)
        self.assertLessEqual(fred_end, CUTOFF)  # 2025-10-29: the cutoff-day vintage did not yet hold 2025-10-30

    def test_bad_input(self):
        for call in (lambda: self.e.get_prices("AAPL", "08/28/2020", "2020-08-28"),
                     lambda: self.e.get_prices("AAPL", "2020-02-30", "2020-03-01"),
                     lambda: self.e.get_prices("AAPL", "2020-03-01", "2020-02-01"),
                     lambda: self.e.get_prices("APPLE", "2020-01-01", "2020-02-01")):
            with self.assertRaises(ToolError):
                call()
        try:
            self.e.get_prices("APPLE", "2020-01-01", "2020-02-01")
        except ToolError as err:
            self.assertIn("AAPL", str(err))
        empty = self.e.get_prices("AAPL", "2020-08-29", "2020-08-30")
        self.assertEqual(empty["rows"], [])
        self.assertIn("trading_day", empty["note"])


class TestPython(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = env()

    def test_computes(self):
        code = ("df = data.get_prices('^GSPC', '2021-12-01', '2022-12-31', fields=['close'], interval='month')\n"
                "r = df.close.pct_change().dropna() * 100\n"
                "print(r.idxmin().strftime('%Y-%m'), round(r.min(), 2))")
        month, drop = (k["value"] for k in market_tasks.hand_tasks(DB)[1]["keys"])  # read from the snapshot
        self.assertEqual(self.e.run_python(code)["stdout"].split(), [month, str(drop)])
        out = self.e.run_python("g = data.get_financials('NVDA', 'Goodwill', fiscal_year=2022, form='10-K')\n"
                                "int(g.loc[g.end == '2022-01-30', 'value'].iloc[0])")
        self.assertEqual(out["stdout"].strip(), "4349000000")  # a last expression is printed

    def test_timeout(self):
        t = time.time()
        with self.assertRaises(ToolError) as cm:
            self.e.run_python("while True:\n    pass")
        self.assertIn("longer than 30 seconds", str(cm.exception))
        self.assertLess(time.time() - t, 40)

    def test_no_socket(self):
        with self.assertRaises(ToolError) as cm:
            self.e.run_python("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)")
        self.assertIn("network access is disabled", str(cm.exception))
        with self.assertRaises(ToolError):
            self.e.run_python("import socket\ns = socket.socket()")
        if market.bwrap_works():  # below the prelude: a child process has no network either
            out = self.e.run_python("import subprocess, sys\nr = subprocess.run([sys.executable, '-c', 'import socket; "
                                    "socket.create_connection((\"1.1.1.1\", 80), timeout=3)'], capture_output=True, "
                                    "text=True)\nprint(r.returncode, r.stderr.strip().splitlines()[-1])")
            self.assertTrue(out["stdout"].startswith("1 OSError: [Errno 101] Network is unreachable"), out)

    def test_no_writes_outside(self):
        target = os.path.join(tempfile.gettempdir(), f"market_escape_{os.getpid()}.txt")
        with self.assertRaises(ToolError) as cm:
            self.e.run_python(f"open({target!r}, 'w').write('x')")
        self.assertIn("only inside its working directory", str(cm.exception))
        with self.assertRaises(ToolError):
            self.e.run_python(f"import pandas as pd\npd.DataFrame({{'a': [1]}}).to_csv({target!r})")
        if market.bwrap_works():  # below the prelude: the file system is read-only for a child process too
            out = self.e.run_python(f"import subprocess\nr = subprocess.run(['sh', '-c', 'echo x > {target}'], "
                                    f"capture_output=True, text=True)\nprint(r.returncode, r.stderr)")
            self.assertIn("Read-only file system", out["stdout"])
        self.assertFalse(os.path.exists(target))
        ok = self.e.run_python("import os\nopen('a.csv', 'w').write('1')\nprint(sorted(os.listdir('.')))")
        self.assertEqual(ok["stdout"].strip(), "['a.csv']")
        again = self.e.run_python("import os\nprint(os.listdir('.'))")  # a fresh directory each call
        self.assertEqual(again["stdout"].strip(), "[]")

    def test_errors(self):
        with self.assertRaises(ToolError) as cm:
            self.e.run_python("print('before')\n1/0")
        self.assertIn("ZeroDivisionError", str(cm.exception))
        self.assertIn("before", str(cm.exception))
        self.assertNotIn("market_sandbox.py", str(cm.exception))
        with self.assertRaises(ToolError) as cm:
            self.e.run_python("data.get_prices('NOPE')")
        self.assertIn("unknown symbol", str(cm.exception))
        with self.assertRaises(ToolError):
            self.e.run_python("   ")
        self.assertIn("nothing printed", self.e.run_python("x = 1")["stdout"])
        long = self.e.run_python("print('x' * 50000)")["stdout"]
        self.assertLess(len(long), market.STDOUT_CHARS + 200)


class TestGrader(unittest.TestCase):
    def test_keys(self):
        g = market.grade_keys
        k = [{"kind": "text", "value": "2021-03", "aliases": ["March 2021"]}, {"kind": "number", "value": -7.15, "tol": 0.005}]
        self.assertTrue(g(k, "2021-03: -7.15%")["passed"])
        self.assertTrue(g(k, "March 2021 (−7.15%)")["passed"])
        self.assertFalse(g(k, "March 2021 (7.15%)")["passed"])
        self.assertFalse(g(k, "2021-04: -7.15%")["passed"])
        free = [{"kind": "number", "value": -7.15, "tol": 0.005, "sign_free": True}]
        self.assertTrue(g(free, "a decline of 7.15%")["passed"])
        self.assertTrue(g(free, "-7.15%")["passed"])
        d = [{"kind": "date", "value": "2021-03-05"}]
        for text in ("2021-03-05", "2021/3/5", "March 5, 2021", "5 March 2021"):
            self.assertTrue(g(d, text)["passed"], text)
        self.assertFalse(g(d, "2021/3/6")["passed"])
        s = [{"kind": "set", "value": [{"kind": "ticker", "value": "GOOGL", "aliases": ["GOOG"]}, "AAPL"]}]
        self.assertTrue(g(s, "AAPL and GOOG")["passed"])
        self.assertFalse(g(s, "GOOG")["passed"])
        self.assertFalse(g(k, None)["passed"])

    def test_own_answers(self):
        ok, bad = market.check_answers(market_tasks.path(DB))
        self.assertEqual((ok, bad), (3, []))


def results(messages):
    """Every tool result so far, parsed (the JSON before any appended note), in order."""
    out = []
    for m in messages:
        if m["role"] == "tool":
            try:
                out.append(json.JSONDecoder().raw_decode(m["content"])[0])
            except json.JSONDecodeError:
                out.append(None)
    return out


def reference(task_id, checklist=True):
    """The scripted reference policy of a hand-written task: it reads the values from the tool results."""
    if task_id == "market-hand-lookup":
        calls = [("get_prices", {"symbol": "AAPL", "start": "2020-08-28", "end": "2020-08-28", "fields": ["close"]}),
                 ("get_prices", {"symbol": "AAPL", "start": "2020-08-28", "end": "2020-08-28", "adjustment": "split",
                                 "fields": ["close"]})]

        def answer(res):
            return f"Unadjusted close: ${res[1]['rows'][0][1]:.2f}; split-adjusted close: ${res[2]['rows'][0][1]:.2f}."
    elif task_id == "market-hand-monthly":
        calls = [("run_python", {"code": "df = data.get_prices('^GSPC', '2021-12-01', '2022-12-31', fields=['close'], "
                                         "interval='month')\nr = df.close.pct_change().dropna() * 100\n"
                                         "print(r.idxmin().strftime('%Y-%m'), round(r.min(), 2))"})]

        def answer(res):
            month, change = res[1]["stdout"].split()
            return f"{month}, a change of {change}%."
    else:
        calls = [("get_financials", {"company": "NVDA", "tag": "Goodwill", "fiscal_year": 2022, "form": "10-K"}),
                 ("get_financials", {"company": "NVDA", "tag": "Assets", "fiscal_year": 2022, "form": "10-K"}),
                 lambda res: ("run_python", {"code": "print(round({} / {} * 100, 2))".format(
                     *[next(x[3] for x in r["rows"] if x[1] == "2022-01-30") for r in res[1:3]])})]

        def answer(res):
            return f"Goodwill was {res[3]['stdout'].strip()}% of total assets at 2022-01-30."
    n, items = len(calls), ["look up", "compute", "answer"]

    def policy(messages):
        res = results(messages)
        if not checklist:
            res = [None] + res  # the same result numbering as with the checklist call first
        k = sum(m["role"] == "assistant" for m in messages) + (not checklist)
        if k == 0:
            return {"reasoning": "plan", "calls": [("set_checklist", {"items": items})]}
        if k <= n:
            c = calls[k - 1]
            return {"reasoning": "look up", "calls": [c(res) if callable(c) else c]}
        done = [("set_checklist", {"items": items, "done": [1, 2, 3]})] if checklist else []
        return {"reasoning": "answer", "calls": done + [("finish", {"answer": answer(res)})]}
    return policy


class TestReference(unittest.TestCase):
    def test_hand_tasks(self):
        with open(market_tasks.path(DB), encoding="utf-8") as f:
            tasks = [json.loads(line) for line in f]
        for t in tasks:
            for cfg in ("H0", "H1"):
                with self.subTest(task=t["task_id"], cfg=cfg):
                    stub = Stub(reference(t["task_id"], checklist=cfg == "H1"))
                    row, traj = asyncio.run(agent.run_one(t, cfg, DB, stub, "stub", 1, agent.Settings(provider="qwen")))
                    self.assertTrue(row["passed"], (row["answer"], row["grade"]))
                    self.assertEqual((row["end"], row["tool_errors"], row["internal_errors"]), ("finished", 0, 0))
                    self.assertEqual(row["verification_rejects"], 0)


M_DIR = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1"
TASK_FILES = [f"{M_DIR}/tasks/market_tasks_{s}.jsonl" for s in ("train", "dev", "test")] + \
    [f"{M_DIR}/external/finsearchcomp_ext_test.jsonl"]


def read_tasks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class TestTaskFiles(unittest.TestCase):
    def test_every_answer_text_passes_its_keys(self):
        """The regenerated files: every task's own answer_text passes its keys (market_cap gives millions of shares)."""
        for path in TASK_FILES:
            with self.subTest(path=os.path.basename(path)):
                ok, bad = market.check_answers(path)
                self.assertEqual(bad, [])
                self.assertGreater(ok, 0)


class TestVerification(unittest.TestCase):
    """H1's number rule for market answers (market.unsupported_in) and its use at finish."""

    RESULTS = ['{"rows": [[null, "2022-01-30", null, 4349000000, "USD"]]}', '{"stdout": "0.09843\\n2.3700001\\n"}',
               '{"rows": [["2021-03-05", 512.34]]}']

    def test_rule(self):
        f = lambda a, q="": market.unsupported_in(a, self.RESULTS, q)
        self.assertEqual(f("Goodwill was $4.349 billion (4,349 million; 4349000 thousand)."), [])
        self.assertEqual(f("9.84% of assets, 0.098 as a fraction; dividends totalled $2.37."), [])
        self.assertEqual(f("A decline of 512.34 (written -512.34 or 512.3)."), [])  # sign and coarser rounding
        self.assertEqual(f("On March 31, 2022 (2022-03-31, 31 March 2022, 2022/3/31, 2022-09, FY2023, Q3 2023), in 2021."), [])
        self.assertEqual(f("Over 20 trading days the value was 7.5.", "Over 20 trading days, what ..."), ["7.5"])
        self.assertEqual(f("A computed 3.17% and 371.25 points"), ["3.17", "371.25"])
        self.assertEqual(f("4349.5 million"), ["4349.5"])  # the answer's own precision is checked

    def episode(self, answer_calls, cfg="H1"):
        task = read_tasks(market_tasks.path(DB))[2]  # NVDA goodwill share of assets
        look = [("get_financials", {"company": "NVDA", "tag": "Goodwill", "fiscal_year": 2022, "form": "10-K"})]
        steps = [{"calls": [("set_checklist", {"items": ["look up", "answer"]})] + look}] + answer_calls
        def policy(messages):
            k = sum(m["role"] == "assistant" for m in messages)
            return steps[min(k, len(steps) - 1)]
        return asyncio.run(agent.run_one(task, cfg, DB, Stub(policy), "stub", 1, agent.Settings(provider="qwen")))

    def test_scale_words_pass_and_mental_arithmetic_is_rejected(self):
        done = ("set_checklist", {"items": ["look up", "answer"], "done": [1, 2]})
        row, traj = self.episode([{"calls": [done, ("finish", {"answer": "Goodwill was $4.349 billion on "
                                                                              "January 30, 2022."})]}])
        self.assertEqual((row["end"], row["verification_rejects"]), ("finished", 0))
        row, traj = self.episode([{"calls": [done, ("finish", {"answer": "Goodwill was 9.84% of assets."})]},
                                  {"calls": [("finish", {"answer": "Goodwill was $4.349 billion."})]}])
        self.assertEqual(row["verification_rejects"], 1)
        msg = next(m["content"] for m in traj["messages"] if m["role"] == "tool" and "verification_failed" in m["content"])
        self.assertIn("9.84", msg)
        self.assertIn("run_python", msg)
        self.assertNotIn("calculator", msg)

    def test_other_kinds_keep_the_shared_rule(self):
        from harness.envs.base import EnvKind
        from harness.envs.registry import KINDS
        for name, k in KINDS.items():
            if name != "market":
                self.assertIs(k.unsupported_numbers, EnvKind.unsupported_numbers, name)
                self.assertIs(k.unsupported_message, EnvKind.unsupported_message, name)
        self.assertIn("with the calculator", EnvKind().unsupported_message(["1.5"]))


class TestAsyncTools(unittest.TestCase):
    """run_python (the kind's ASYNC_TOOLS) runs in a worker thread, so two episodes' slow calls overlap; the other
    tools run on the event loop's thread."""

    def test_slow_calls_overlap(self):
        import threading
        spans, where = [], {}
        orig_py, orig_px = market.MarketEnv.run_python, market.MarketEnv.get_prices

        def slow(self, code):
            t0 = time.monotonic()
            where.setdefault("run_python", set()).add(threading.get_ident())
            time.sleep(1.0)
            spans.append((t0, time.monotonic()))
            return {"stdout": "1\n"}

        def prices(self, *a, **k):
            where.setdefault("get_prices", set()).add(threading.get_ident())
            return orig_px(self, *a, **k)

        task = read_tasks(market_tasks.path(DB))[0]
        steps = [{"calls": [("get_prices", {"symbol": "AAPL", "start": "2020-08-28", "end": "2020-08-28"})]},
                 {"calls": [("run_python", {"code": "print(1)"})]},
                 {"calls": [("finish", {"answer": "512.34 and 128.09"})]}]

        def policy(messages):
            return steps[min(sum(m["role"] == "assistant" for m in messages), 2)]

        async def two():
            return await asyncio.gather(*(agent.run_one(task, "H0", DB, Stub(policy), "stub", k, agent.Settings())
                                          for k in (1, 2)))
        market.MarketEnv.run_python, market.MarketEnv.get_prices = slow, prices
        try:
            t = time.monotonic()
            rows = asyncio.run(two())
            wall = time.monotonic() - t
        finally:
            market.MarketEnv.run_python, market.MarketEnv.get_prices = orig_py, orig_px
        self.assertEqual([r["end"] for r, _ in rows], ["finished", "finished"])
        (a0, a1), (b0, b1) = spans
        self.assertLess(max(a0, b0), min(a1, b1))  # the two 1-second calls overlapped
        self.assertLess(wall, 1.9)
        self.assertEqual(where["get_prices"], {threading.main_thread().ident})
        self.assertNotIn(threading.main_thread().ident, where["run_python"])
        self.assertEqual(market.MarketKind.ASYNC_TOOLS, ("run_python",))


class TestMarketReference(unittest.TestCase):
    """The reference policy (harness/market_reference.py via golden.py) solves a dev task of every template and some
    external tasks with run_python alone, under H1 (its answer passing the verification) and H0."""

    def test_reference_policy(self):
        from harness import golden
        tasks, seen = [], set()
        for t in read_tasks(f"{M_DIR}/tasks/market_tasks_dev.jsonl"):
            if t["template"] not in seen:
                seen.add(t["template"])
                tasks.append(t)
        ext = read_tasks(TASK_FILES[-1])
        tasks += [ext[i] for i in (0, 6, 31, 63, 70, 80, 92)]  # FRED, SEC, prices, T3 with SEC quarters, BTC, ratios

        async def run(cfg):
            async def one(t):
                client = golden.Scripted(golden.market_episode(t, cfg == "H1"))
                row, _ = await agent.run_one(t, cfg, DB, client, "golden", 1, agent.Settings(provider="qwen",
                                                                                         token_records=False))
                return t["task_id"], row, client.error
            return await asyncio.gather(*(one(t) for t in tasks))
        for cfg in ("H1", "H0"):
            for tid, row, err in asyncio.run(run(cfg)):
                with self.subTest(task=tid, cfg=cfg):
                    self.assertIsNone(err)
                    self.assertTrue(row["passed"], (row["answer"], row["grade"]))
                    self.assertEqual((row["verification_rejects"], row["tool_errors"]), (0, 0))



HARD_FILES = [f"{M_DIR}/tasks/market_tasks_hard_{s}.jsonl" for s in ("train", "dev", "test")]
HARD_TEMPLATES = ("risk_rank", "rolling_beta", "ma_cross", "reinvest_return", "div_event_study", "q4_margin_rank",
                  "filing_reaction", "month_chain")
SPLIT_WINDOWS = {"train": ("2000-01-01", "2016-12-31"), "dev": ("2017-01-01", "2019-12-31"),
                 "test": ("2020-01-01", "2025-10-30")}


def ticker_split(ticker):
    """The generator's equity split (src/01_data-processing/code/market_tasks/common.py), restated for the test."""
    import hashlib
    h = int(hashlib.sha256(ticker.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("dev" if h < 80 else "test")


class TestHardTier(unittest.TestCase):
    """The hard tier (templates_hard.py, difficulty 4 and 5; research/log/2026-10-31-0150.md): the files' own
    answers, the independent check recorded in manifest_hard.json, the split rules, the hardened templates' prompts
    and key layout, and the reference policy."""

    def test_answer_texts_and_layout(self):
        kinds = {r["symbol"]: r["kind"] for r in market.snapshot(DB).securities().values()}
        for path in HARD_FILES:
            with self.subTest(path=os.path.basename(path)):
                ok, bad = market.check_answers(path)
                self.assertEqual(bad, [])
                tasks = read_tasks(path)
                self.assertEqual(ok, len(tasks))
                self.assertEqual({t["template"] for t in tasks}, set(HARD_TEMPLATES))
                for t in tasks:
                    sp = t["split"]
                    self.assertTrue(t["task_id"].startswith(f"mk-hard-{sp}-{t['template']}-"))
                    self.assertIn(t["meta"]["difficulty"], (4, 5))
                    ents = [e for e in t["meta"]["entities"] if not e.startswith("CIK:")]
                    # equities stay in their ticker split; any index, ETF or FRED series keeps every date in the window
                    self.assertTrue(all(ticker_split(e) == sp for e in ents if kinds.get(e) == "equity"), t["task_id"])
                    if any(kinds.get(e) != "equity" for e in ents):
                        lo, hi = SPLIT_WINDOWS[sp]
                        self.assertTrue(lo <= t["meta"]["start"] <= t["meta"]["end"] <= hi, t["task_id"])

    def test_hardened_templates(self):
        """Version 2 of five templates (research/log/2026-10-31-0150.md, the hardening): companies named without
        tickers at both difficulties, no spelled-out formula, and the added keys in their order."""
        kinds = {r["symbol"]: r["kind"] for r in market.snapshot(DB).securities().values()}
        layout = {  # (template, difficulty) -> key kinds; "count" is a number key with tolerance 0.5
            ("rolling_beta", 4): ["date", "number", "date", "number"],
            ("rolling_beta", 5): ["date", "number", "date", "number", "number"],
            ("ma_cross", 4): ["date", "date", "date", "number"],
            ("ma_cross", 5): ["count", "number", "number", "date", "number"],
            ("reinvest_return", 4): ["number"] * 3,
            ("reinvest_return", 5): ["number"] * 4,
            ("div_event_study", 4): ["count", "number", "date", "number"],
            ("div_event_study", 5): ["count", "number", "date", "number", "date", "number"],
            ("q4_margin_rank", 4): ["ticker", "number", "number", "number"],
            ("q4_margin_rank", 5): ["ticker"] + ["number"] * 4,
        }
        spelled = ("sample covariance", "divided by the sample variance", "SMA50 > SMA200", "<= SMA", "minus 1)",
                   "final shares times", "this difference, in percentage points", "Fourth-quarter amounts are not")
        seen = set()
        for path in HARD_FILES:
            for t in read_tasks(path):
                k = (t["template"], t["meta"]["difficulty"])
                if k not in layout:
                    continue
                seen.add(k)
                with self.subTest(task=t["task_id"]):
                    got = ["count" if x["kind"] == "number" and x.get("tol") == 0.5 else x["kind"] for x in t["keys"]]
                    self.assertEqual(got, layout[k])
                    for e in t["meta"]["entities"]:
                        if kinds.get(e) == "equity":
                            self.assertNotIn(f"({e})", t["prompt"])
                    self.assertFalse([s for s in spelled if s in t["prompt"]])
        self.assertEqual(seen, set(layout))

    def test_independent_check_agrees_on_every_task(self):
        with open(f"{M_DIR}/tasks/manifest_hard.json", encoding="utf-8") as f:
            man = json.load(f)
        full = man["independent_check_all_tasks"]
        self.assertEqual(set(full), set(HARD_TEMPLATES))
        self.assertEqual(sum(v["checked"] for v in full.values()), sum(man["totals"].values()))
        self.assertTrue(all(v["checked"] == v["agree"] for v in full.values()), full)
        self.assertEqual(man["independent_check_failures"], [])

    def test_reference_policy(self):
        """One dev task per template and difficulty, under H1 (the answer passing the verification) and H0."""
        from harness import golden
        tasks, seen = [], set()
        for t in read_tasks(HARD_FILES[1]):
            k = (t["template"], t["meta"]["difficulty"])
            if k not in seen:
                seen.add(k)
                tasks.append(t)
        self.assertEqual(len(tasks), 2 * len(HARD_TEMPLATES))

        async def run(cfg):
            async def one(t):
                client = golden.Scripted(golden.market_episode(t, cfg == "H1"))
                row, _ = await agent.run_one(t, cfg, DB, client, "golden", 1, agent.Settings(provider="qwen",
                                                                                         token_records=False))
                return t["task_id"], row, client.error
            return await asyncio.gather(*(one(t) for t in tasks))
        for cfg in ("H1", "H0"):
            for tid, row, err in asyncio.run(run(cfg)):
                with self.subTest(task=tid, cfg=cfg):
                    self.assertIsNone(err)
                    self.assertTrue(row["passed"], (row["answer"], row["grade"]))
                    self.assertEqual((row["verification_rejects"], row["tool_errors"]), (0, 0))

if __name__ == "__main__":
    unittest.main()
