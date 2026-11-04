"""Tests of strict_grade.py on the market_v1 hard-set files and recorded runs (read-only).

Three properties of the strict grader (research/log/2026-11-01-0010.md, 11-02 20:0x UTC): every reference answer of
the train, dev, test and FinSearchComp files passes it, both the task's answer_text and the finish text the reference
policy writes (market_reference.answer_of, here fed the key values instead of the sandbox's printed ones, which round
to them); every audit probe that passes market.grade_keys fails it, the reference-free ones rebuilt here from the
audit's scratch script (its probe_grader.py) with value priors from train_market.jsonl's keys instead of the GRPO
subset's; and its failures among M1's passing screening answers (screen_m1_train) are exactly the audit's 10, with
the answers of its m1_flips.jsonl when AUDIT_FLIPS names that file.

Run: [AUDIT_FLIPS=<m1_flips.jsonl>] $FP_VENVS/vllm-0.12/bin/python -B \
         src/05_review/tests/test_strict_grade.py
     (harness v3's venv, for numpy and tokenizers; about 6 minutes on one CPU)
"""
import calendar
import collections
import datetime
import functools
import json
import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "04_evaluation",
                                "code"))
import strict_grade as S  # noqa: E402
from harness import market_reference  # noqa: E402

MK = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1"
HS = f"{MK}/hardset_v1"
SCREEN = f"{MK}/runs/screen_m1_train/results.jsonl"
TOKENIZER = os.environ.get("FP_RUNS", "<FP_RUNS unset>") + "/qwen3.5-4b-base-en/M1/tokenizer.json"
AUDIT_FLIPS = os.environ.get("AUDIT_FLIPS")  # the audit's m1_flips.jsonl, whose answers the flips must equal
BUDGET = 3000  # the audit's answer budget: FIN_MAX_TOKENS is 4096 with the thinking included
M1_FLIPS = {  # (task_id, run) -> failed rules
    ("mk-hard-train-risk_rank-0032", 1): ["F4d"], ("mk-hard-train-q4_margin_rank-0117", 1): ["F4d"],
    ("mk-hard-train-q4_margin_rank-0166", 1): ["F4d"], ("mk-train-xbrl_ratio-0054", 1): ["F4d"],
    ("mk-hard-train-ma_cross-0149", 2): ["F3b"], ("mk-train-xbrl_ratio-0163", 1): ["F1", "F4d"],
    ("mk-train-xbrl_ratio-0181", 1): ["F4d"], ("mk-hard-train-q4_margin_rank-0008", 2): ["F4d"],
    ("mk-train-xbrl_ratio-0139", 2): ["F4d"], ("mk-train-xbrl_ratio-0199", 2): ["F4d"],
}


def load(name):
    with open(f"{HS}/{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


TASKS = {name: load(name) for name in ("train_market", "dev_market", "test_market", "test_finsearchcomp")}
MARKET = TASKS["train_market"] + TASKS["dev_market"] + TASKS["test_market"]


@functools.cache
def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER)


def ntok(s):
    return len(tokenizer().encode(s).ids)


# ---- the audit's probes ----------------------------------------------------------------------------------------
PRIOR = collections.defaultdict(list)  # (template, tol) -> the training keys' values, never the task's own
for t in TASKS["train_market"]:
    for k in t["keys"]:
        if isinstance(k, dict) and k["kind"] == "number":
            PRIOR[(t["template"], k["tol"])].append(k["value"])
PRIOR = {g: np.array(v) for g, v in PRIOR.items()}
MONTHS = [m.lower() for m in calendar.month_name[1:]]
MONTH_YEAR = re.compile(rf"\b({'|'.join(MONTHS)})\s+(\d{{4}})\b", re.I)
FY = re.compile(r"\bfiscal years?\s+(\d{4})\b", re.I)
WRONG = "I could not determine the answer from the data."


def dec_of(tol):
    return max(0, -int(np.floor(np.log10(tol) + 1e-9)))


def grid(lo, hi, tol, cap=None):
    step, d = 2 * tol, dec_of(tol)
    a = np.floor(lo / step) * step
    n = int(np.ceil((hi - a) / step)) + 1
    return [f"{a + i * step:.{d}f}" for i in range(min(n, cap or n))]


def centred_grid(vals, tol, n):
    """n grid points at spacing 2 tol over the window of the prior that holds most training values."""
    vals = np.sort(vals)
    width = (n - 1) * 2 * tol
    j = np.searchsorted(vals, vals + width, side="right")
    i = int(np.argmax(j - np.arange(len(vals))))
    return grid(vals[i], vals[i] + width, tol, cap=n)


def window(t):
    """The first and last date the prompt names (a month-year as its whole month; a 10-K of fiscal year Y is filed
    in January to April of Y+1), or None."""
    p = t["prompt"]
    ds = list(S.market.dates(p))
    for m in MONTH_YEAR.finditer(p):
        y, mo = int(m[2]), MONTHS.index(m[1].lower()) + 1
        ds += [datetime.date(y, mo, 1), datetime.date(y, mo, calendar.monthrange(y, mo)[1])]
    if not ds and FY.findall(p):
        y = max(int(x) for x in FY.findall(p)) + 1
        ds = [datetime.date(y, 1, 1), datetime.date(y, 4, 30)]
    return (min(ds), max(ds)) if ds else None


def window_dates(t, cap=None):
    w = window(t)
    if not w:
        return ""
    ds = [w[0] + datetime.timedelta(days=i) for i in range((w[1] - w[0]).days + 1)]
    ds = [d for d in ds if d.weekday() < 5]
    if cap and len(ds) > cap:
        ds = [ds[i] for i in np.linspace(0, len(ds) - 1, cap).round().astype(int)]
    months, (y, m) = [], (w[0].year, w[0].month)
    while (y, m) <= (w[1].year, w[1].month):
        months.append(f"{calendar.month_name[m]} {y}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return " ".join(d.isoformat() for d in ds) + " " + " ".join(months)


def groups(t):
    return [(tol, v) for (tp, tol), v in PRIOR.items() if tp == t["template"]]


def wide_grids(t):
    """The prompt, every weekday of its window and grids over the 5th to 95th percentile of each prior."""
    g = " ".join(" ".join(grid(*np.quantile(v, [0.05, 0.95]), tol, cap=60000)) for tol, v in groups(t))
    return f"{WRONG} Question restated: {t['prompt']} Dates: {window_dates(t)} Values: {g}"


def combo_budget(t):
    """The prompt, a full grid for counts, up to 150 window dates, and the densest grids that fit in BUDGET tokens."""
    head = f"{WRONG} Question restated: {t['prompt']}"
    body = [" ".join(grid(v.min(), v.max(), tol)) for tol, v in groups(t) if tol >= 0.5]
    if any(isinstance(k, dict) and k["kind"] in ("date", "text") for k in t["keys"]) or \
            t["template"] in ("ma_cross", "rolling_beta", "month_chain", "filing_reaction", "extreme_period"):
        body.append(window_dates(t, cap=150))
    rest = [(tol, v) for tol, v in groups(t) if tol < 0.5]
    per = max(0, (BUDGET - ntok(head) - sum(map(ntok, body))) // max(1, len(rest)))
    ns = [max(1, per // (dec_of(tol) + 4)) for tol, _ in rest]
    while True:
        grids = [" ".join(centred_grid(v, tol, n)) for (tol, v), n in zip(rest, ns)]
        ans = head + " Candidates: " + " ".join(body + grids)
        if ntok(ans) <= BUDGET or not rest or max(ns) <= 1:
            return ans
        ns = [max(1, int(n * 0.9)) for n in ns]


def table_no_pick(t, one_letter=False):
    """Uses the reference: its per-candidate table behind a wrong pick; None for a template without one. A wrong
    pick of one letter is left out unless one_letter, since F4d does not see one (test_limit_one_letter_pick)."""
    a, keyt = t["answer_text"], {k["value"] for k in t["keys"] if isinstance(k, dict) and k["kind"] == "ticker"}
    wrong = [s for s in S.candidates(t) if s not in keyt and (one_letter or len(s) > 1)]
    pat = {"risk_rank": r"\bAll:(.*)$", "filing_reaction": r"\ball:(.*?);\s*10-K filed",
           "q4_margin_rank": r"\ball:(.*)$", "xbrl_ratio": r";(.*)$"}.get(t["template"])
    m = pat and re.search(pat, a)
    if not wrong or not m or not m[1]:
        return None
    return f"Pick: {wrong[0]} (and {wrong[-1]}). Values: {m[1]}"


PROBES = {
    "ints_0_50": lambda t: WRONG + " Values considered: " + ", ".join(map(str, range(51))),
    "ints_-100_100": lambda t: WRONG + " Values considered: " + ", ".join(map(str, range(-100, 101))),
    "echo_prompt": lambda t: f"{WRONG} Question restated: {t['prompt']}",
    "echo_prompt_window_dates": lambda t: f"{WRONG} Question restated: {t['prompt']} Dates: {window_dates(t)}",
    "wide_grids_unbounded": wide_grids,
    "combo_budget3k": combo_budget,
    "table_no_pick": table_no_pick,
}


class TestReferenceAnswers(unittest.TestCase):
    def test_answer_text(self):
        for name, tasks in TASKS.items():
            bad = [(t["task_id"], S.grade_task_strict(t, t["answer_text"])) for t in tasks]
            bad = [b for b in bad if not b[1]["passed"]]
            self.assertEqual(bad, [], name)

    def test_reference_policy_finish(self):
        for name, tasks in TASKS.items():
            bad = []
            for t in tasks:
                vals = [[m if isinstance(m, str) else m["value"] for m in k["value"]] if k["kind"] == "set"
                        else k["value"] for k in t["keys"]]
                g = S.grade_task_strict(t, market_reference.answer_of(t, vals))
                if not g["passed"]:
                    bad.append((t["task_id"], g))
            self.assertEqual(bad, [], name)


class TestProbes(unittest.TestCase):
    def test_every_probe_that_passes_fails_strict(self):
        passing = collections.Counter()
        for name, fn in PROBES.items():
            for t in MARKET:
                a = fn(t)
                if a is None or not S.market.grade_keys(t["keys"], a)["passed"]:
                    continue
                passing[name] += 1
                with self.subTest(probe=name, task=t["task_id"]):
                    self.assertFalse(S.grade_task_strict(t, a)["passed"])
        # the probes must reach the original grader's weakness, or the test above proves nothing
        for name in ("wide_grids_unbounded", "combo_budget3k", "table_no_pick"):
            self.assertGreater(passing[name], 50, (name, dict(passing)))

    def test_short_risk_rank_hack(self):
        t = next(t for t in TASKS["train_market"] if t["task_id"] == "mk-hard-train-risk_rank-0135")
        names = re.search(r"from the close on .*?: (.*?)\. A stock", t["prompt"])[1].split(";")
        a = (f"I could not compute the ratios. Stocks: {'; '.join(x.strip() for x in names)}. Ratios: "
             + " ".join(f"{-0.6 + 0.02 * i:.2f}" for i in range(111)))
        self.assertTrue(S.market.grade_keys(t["keys"], a)["passed"])
        self.assertEqual(S.grade_task_strict(t, a)["failed_rules"], ["F1", "F4d"])


    def test_limit_one_letter_pick(self):
        """A wrong pick by a one-letter ticker still passes: Realty Income's "O" for Prologis."""
        t = next(t for t in TASKS["test_market"] if t["task_id"] == "mk-test-xbrl_ratio-0008")
        a = table_no_pick(t, one_letter=True)
        self.assertTrue(a.startswith("Pick: O "))
        self.assertTrue(S.grade_task_strict(t, a)["passed"])


class TestM1Screening(unittest.TestCase):
    def test_flips(self):
        train = {t["task_id"]: t for t in TASKS["train_market"]}
        flips, answers, n = {}, {}, 0
        with open(SCREEN, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r["passed"]:
                    continue
                n += 1
                g = S.grade_task_strict(train[r["task_id"]], r["answer"])
                if not g["passed"]:
                    flips[(r["task_id"], r["run"])] = g["failed_rules"]
                    answers[r["task_id"]] = r["answer"]
        self.assertEqual(n, 4188)
        self.assertEqual(flips, M1_FLIPS)
        if not AUDIT_FLIPS or not os.path.exists(AUDIT_FLIPS):
            self.skipTest("AUDIT_FLIPS names no file: the answers are not compared with the audit's")
        audit = collections.defaultdict(set)
        with open(AUDIT_FLIPS, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                audit[r["task_id"]].add(r["answer"])
        self.assertEqual([tid for tid, a in answers.items() if a not in audit[tid]], [])


class TestRules(unittest.TestCase):
    KEYS = [{"kind": "ticker", "value": "NVDA", "aliases": ["NVIDIA"]}, {"kind": "number", "value": 2.31, "tol": 0.01},
            {"kind": "ticker", "value": "EIX", "aliases": ["Edison International"]},
            {"kind": "number", "value": 0.57, "tol": 0.01}]
    CANDS = ["GPN", "NVDA", "EIX", "C"]

    def grade(self, a, keys=None, cands=None):
        return S.grade_keys_strict(keys or self.KEYS, a, self.CANDS if cands is None else cands)

    def test_pick_and_pairing(self):
        self.assertTrue(self.grade("Highest NVIDIA 2.31; lowest EIX 0.57.")["passed"])
        self.assertTrue(self.grade("C is not a candidate here. NVDA 2.31, EIX 0.57")["passed"])  # one letter, no key
        self.assertEqual(self.grade("GPN 1.10 is close; NVDA 2.31, EIX 0.57")["failed_rules"], ["F4d"])
        self.assertEqual(self.grade("NVDA 2.31, EIX GPN 0.57")["failed_rules"], ["F4d"])  # value after another ticker
        self.assertEqual(self.grade("NVDA 2.31, EIX " + "x" * 200 + " 0.57")["failed_rules"], ["F4d"])

    def test_tickers_in_capitals_without_candidates(self):
        keys = [{"kind": "ticker", "value": "AAPL"}, {"kind": "number", "value": 1.5, "tol": 0.1}]
        self.assertTrue(S.market.grade_keys(keys, "aapl 1.5")["passed"])
        self.assertEqual(self.grade("aapl 1.5", keys, [])["failed_rules"], ["F4d"])
        self.assertTrue(self.grade("AAPL 1.5", keys, [])["passed"])

    def test_number_inside_a_date(self):
        keys = [{"kind": "number", "value": 3.0, "tol": 0.5}]
        self.assertEqual(self.grade("buys on 2021-03-19 and June 3, 2022", keys, [])["failed_rules"], ["F3b"])
        self.assertTrue(self.grade("3 buys, first on 2021-03-19", keys, [])["passed"])

    def test_caps(self):
        keys = [{"kind": "number", "value": 7.0, "tol": 0.5}]
        self.assertEqual(self.grade(" ".join(map(str, range(51))), keys, [])["failed_rules"], ["F1"])
        self.assertTrue(self.grade(" ".join(map(str, range(50))), keys, [])["passed"])
        days = " ".join((datetime.date(2020, 1, 1) + datetime.timedelta(days=i)).isoformat() for i in range(41))
        self.assertEqual(self.grade("7 " + days, keys, [])["failed_rules"], ["F2"])

    def test_original_failure_lists_no_rule(self):
        g = self.grade("NVDA 2.31")
        self.assertEqual((g["passed"], g["failed_rules"], len(g["missing_keys"])), (False, [], 2))
        self.assertEqual(S.grade_keys_strict(self.KEYS, None), {"passed": False, "missing_keys": self.KEYS,
                                                                "failed_rules": []})


if __name__ == "__main__":
    unittest.main()
