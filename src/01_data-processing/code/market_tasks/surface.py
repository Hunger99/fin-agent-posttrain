"""Reworded copies of the market tasks, each request in new words (research/log/2026-11-01-0010.md, 11-02 22:5x UTC).

Every market template renders its request with one of two fixed wordings, and M1's SFT trajectories saw only those, so
its gain over the official Qwen3.5-4B (+31.0 points on the market test) may depend on the trained wording. This module
writes the request of each task of the 10 templates of hardset_v1 in one of the wordings below, at least 3 per template,
written by hand, none of them one of the generator's own; sha256("<seed>|<task id>") picks the wording and its
sub-phrases. A reworded test set then measures the dependence with an evaluation only, without retraining.

A rewording is rendered from the task's meta.params and difficulty, the same inputs that defined the original, with
entity names from the snapshot's securities table written as templates.Ctx.ent writes them (tickers only where the
original shows them). Only what the stored parameters do not record is read off the original prompt: which dates
fmt_date wrote long, xbrl_ratio's fiscal-year end list (difficulty 1-2) and whether it names "(SEC XBRL data)",
rolling_vol's "(e.g. 1.234)", and whether filing_reaction and month_chain number their steps, since the numerals then
occur in the request. Everything but "prompt" stays byte-identical. The request also keeps exactly the numbers of the
original, because harness v3 reads them: unsupported_in (H1's finish check) does not check an answer number that the
question contains, and Episode.note_numbers lets the checklist trace answer numbers to the question.

check_one() compares each rewording with its original and fails the build on any of:
  - values: the sets of ISO dates, long dates and month labels, of numbers (as unsupported_in and as note_numbers read
    them), of us-gaap tags and of rounding words ("two decimal...") differ; a symbol, FRED series or option phrase
    (OPTIONS: Sharpe/Sortino, highest/lowest, gain/loss, vintage, adjustment, unit words) is shown in one and not in
    the other; an entity string rendered here is missing from either (so a name-only original keeps its names only);
  - fields: the row with the original prompt put back is not byte-identical to the original line;
  - distance: the word-level edit distance, 1 - difflib ratio over the words of the fixed text (every value above
    removed, punctuation ignored), is below MIN_DIST = 0.30, chosen so that at least 3 words in 10 of the fixed text
    change while the finance terms, which no wording may replace, stay.
It also reports, without a threshold, the share of the masked 5-grams of the request found in the train prompts of the
same template (an original scores about 1).

Writes <out>/{dev,test}_market.jsonl and manifest.json (seed, wording counts per template, distances, 5-gram overlap,
failures, the wording of every task, sha256 of inputs and outputs); nothing when any task fails a check.

Usage: PYTHONDONTWRITEBYTECODE=1 $FP_VENVS/mkt/bin/python -B surface.py --tasks-dir <hardset_v1> --out <dir>
           [--seed 20261102] [--review <file> --review-n 2]
"""
import argparse
import collections
import difflib
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import DB, MONTHS, long_date, month_label, ro_connect  # noqa: E402
from templates import FUT_NAMES, INDEX_NAMES  # noqa: E402
from templates_hard import FX_HOME, MARGINS  # noqa: E402

FILES = ("dev_market", "test_market")
MIN_DIST = 0.30
NWORD = 3  # frames per template; each frame also draws its sub-phrases from the same rng


def task_rng(seed, task_id):
    return random.Random(int(hashlib.sha256(("%s|%s" % (seed, task_id)).encode()).hexdigest()[:16], 16))


class Names:
    """Entity strings as templates.Ctx.ent writes them, from the snapshot's securities table."""

    def __init__(self, db=DB):
        con = ro_connect(db)
        rows = con.execute("select symbol, kind, name from securities").fetchall()
        con.close()
        self.kind = {s: k for s, k, _ in rows}
        self.name = {s: str(n).strip() for s, _, n in rows}

    def ent(self, sym, d):
        k, nm = self.kind[sym], self.name[sym]
        if k == "equity":
            return nm if d == 3 else "%s (%s)" % (nm, sym)
        if k == "index":
            n = INDEX_NAMES.get(sym, nm)
            n = n if n.lower().endswith("index") else n + " index"
            return ("the %s" % n) if d == 3 else ("the %s (Yahoo symbol %s)" % (n, sym))
        if k == "etf":
            return ("the %s" % nm) if d == 3 else ("the %s (%s)" % (nm, sym))
        if k == "future":
            return "the Yahoo continuous front-month %s futures series (%s)" % (FUT_NAMES[sym], sym)
        if k == "fx":
            return "the %s exchange rate (Yahoo symbol %s)" % (nm, sym)
        return "%s (%s)" % (nm, sym)


class R:
    """One task being reworded: its params, difficulty, original prompt, rng and the entity strings it rendered."""

    def __init__(self, t, N, seed):
        self.t, self.N, self.p, self.d = t, N, t["meta"]["params"], t["meta"]["difficulty"]
        self.orig = t["prompt"]
        self.rng = task_rng(seed, t["task_id"])
        self.k = self.rng.randrange(NWORD)
        self.ents = []

    def ent(self, sym, d):
        e = self.N.ent(sym, d)
        self.ents.append(e)
        return e

    def date(self, iso):
        """The date as the original wrote it (fmt_date chose ISO or long form)."""
        if long_date(iso) in self.orig:
            return long_date(iso)
        if iso in self.orig:
            return iso
        raise ValueError("%s: date %s not in the prompt" % (self.t["task_id"], iso))

    def pick(self, *alts):
        return self.rng.choice(alts)

    def frame(self, *alts):
        assert len(alts) == NWORD
        return alts[self.k]


def cap(s):
    return s[0].upper() + s[1:]


def pct(r):
    return r.pick("in percent rounded to two decimals (12.34 for 12.34%; negative for a decline)",
                  "as a percentage with two decimals (write 12.34 for 12.34%, and a negative number for a decline)",
                  "as a percentage to two decimal places (e.g. 12.34 means 12.34%; negative for a decline)")


def nt_rule(r):
    return r.pick("Any date given here that is not a trading day for the instrument stands for the last trading day on "
                  "or before it.",
                  "Where a date given here is not a trading day for the instrument, use the most recent trading day on "
                  "or before it.")


def rev_rule(r, doc):
    """The revenue tag precedence of templates_hard.REV_RULE, for the filing named by doc."""
    return r.pick("revenue means us-gaap:Revenues when %s reports it for the year, else "
                  "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax, else us-gaap:SalesRevenueNet" % doc,
                  "revenue is taken as us-gaap:Revenues if %s reports it for the year, failing that as "
                  "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax, and failing that as "
                  "us-gaap:SalesRevenueNet" % doc,
                  "revenue is the first of us-gaap:Revenues, "
                  "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax and us-gaap:SalesRevenueNet, in that "
                  "order, reported for the year in %s" % doc)


def steps(r, parts, numbered):
    """Joined steps; numbered (as the original was) with the step numerals, else with words, so the numbers match."""
    if numbered:
        return " ".join("Step %d: %s." % (i + 1, cap(s)) for i, s in enumerate(parts))
    words = r.pick(["First", "Next", "Finally"], ["To begin", "After that", "Last"], ["First", "Then", "Finally"])
    return " ".join("%s, %s." % (w, s) for w, s in zip(words, parts))


# ============================================================================
# Hard tier (templates_hard.py)

def risk_rank(r):
    p, d = r.p, r.d
    K, rf, ask = len(p["symbols"]), p["rf"], p["ask"]
    nk = "%d " % K if "these %d stocks" % K in r.orig else ""  # the generator's second wording gives no count
    names = "; ".join(r.ent(s, 3 if d == 5 else 2) for s in p["symbols"])
    S, E = r.date(p["start"]), r.date(p["end"])
    T = "annualized %s ratio" % ("Sharpe" if p["metric"] == "sharpe" else "Sortino")
    x = "excess returns" if rf else "total returns"
    ret = r.pick("Use daily total returns from closes adjusted for splits and dividends: on every trading day after "
                 "the start date up to and including the end date, a stock's daily total return is that day's adjusted "
                 "close divided by the previous trading day's adjusted close, minus 1.",
                 "Returns are daily total returns: for each trading day after the start date, up to and including the "
                 "end date, divide the stock's close adjusted for splits and dividends by its adjusted close on the "
                 "previous trading day and subtract 1 to get that day's total return.")
    if rf:
        rfs = r.pick("A day's excess return is its total return less a daily risk-free rate: the 3-month Treasury bill "
                     "rate (FRED series DTB3, in percent) on that trading day, or its most recent earlier observation, "
                     "divided by 100 and by 252.",
                     "The daily excess return subtracts from each daily total return the daily risk-free rate, which "
                     "is the 3-month Treasury bill rate from FRED series DTB3 (in percent) for that trading day, or "
                     "the most recent earlier observation, divided by 100 and by 252.")
    else:
        rfs = r.pick("Use a risk-free rate of zero.", "Treat the risk-free rate as zero.")
    if p["metric"] == "sharpe":
        body = r.pick("the average of the daily %s, multiplied by the square root of 252 and divided by their sample "
                      "standard deviation (n-1 denominator)" % x,
                      "the average of the daily %s over their sample standard deviation (with the n-1 denominator), "
                      "times the square root of 252" % x)
    else:
        body = r.pick("the average of the daily %s, multiplied by the square root of 252 and divided by their downside "
                      "deviation, the downside deviation being the square root of the average of min(daily %s, 0) "
                      "squared taken over all days of the window" % (x, x[:-1]),
                      "the square root of 252 times the ratio of the average daily %s to their downside deviation, "
                      "which is the square root of the mean, over all days of the window, of min(daily %s, 0) squared"
                      % (x, x[:-1]))
    q = {"highest": r.pick("Report the stock with the highest %s and its value." % T,
                           "Name the stock whose %s was highest and give that ratio." % T),
         "lowest": r.pick("Report the stock with the lowest %s and its value." % T,
                          "Name the stock whose %s was lowest and give that ratio." % T),
         "both": r.pick("Report the stock with the highest %s and the stock with the lowest, with both values." % T,
                        "Name the stocks with the highest and the lowest %s and give each of the two ratios." % T)}[ask]
    rnd = r.pick("Round each ratio to two decimal places.", "Give the ratios to two decimals.")
    nt = " " + nt_rule(r) if d == 5 else ""
    return r.frame(
        "%s Consider these %sstocks: %s. The window runs from the close on %s to the close on %s. %s %s The %s is %s. "
        "%s%s" % (q, nk, names, S, E, ret, rfs, T, body, rnd, nt),
        "%s %s Define the %s as %s. With these definitions, take the following %sstocks, measured between the close on "
        "%s and the close on %s: %s. %s %s%s" % (ret, rfs, T, body, nk, S, E, names, q, rnd, nt),
        "Rank these %sstocks by %s over the period that starts at the close on %s and ends at the close on %s: %s.%s "
        "%s %s Here the %s is %s. %s %s" % (nk, T, S, E, names, nt, ret, rfs, T, body, q, rnd),
    )


def rolling_beta(r):
    p = r.p
    ent, bent, W, S0, E0 = r.ent(p["symbol"], 3), r.ent(p["bench"], 2), p["window"], p["start"], p["end"]
    if p["total"]:
        rdef = r.pick("simple daily total returns computed from closes adjusted for splits and dividends (for an "
                      "index, from its closing levels)",
                      "simple day-over-day total returns of the closes adjusted for splits and dividends (closing "
                      "levels where the series is an index)")
    else:
        rdef = r.pick("simple daily returns computed from split-adjusted closes (for an index, from its closing "
                      "levels)",
                      "simple day-over-day returns of the split-adjusted closes (closing levels where the series is an "
                      "index)")
    wdef = r.pick("the beta of a given day is estimated from the %d daily returns that end on that day, using returns "
                  "from before %s whenever the window needs them" % (W, S0),
                  "a day's beta comes from the window of %d daily returns ending on that day, which reaches back "
                  "before %s where needed" % (W, S0))
    q = r.pick("Report the trading day with the highest beta and the trading day with the lowest beta, with each of "
               "those betas.",
               "Find the trading day on which the beta was highest and the one on which it was lowest, and give the "
               "beta on each.")
    if p["ask_mean"]:
        q += " " + r.pick("Report as well the mean of the daily betas across all of those trading days.",
                          "Add the average of the daily betas over all of those trading days.")
    fmt = r.pick("Write dates in YYYY-MM-DD form and round betas to three decimals.",
                 "Dates in YYYY-MM-DD format; betas to three decimal places.")
    return r.frame(
        "Track the rolling %d-trading-day beta of %s relative to %s for every trading day between %s and %s, both days "
        "included, based on %s; %s. %s %s" % (W, ent, bent, S0, E0, rdef, wdef, q, fmt),
        "Using %s, calculate for each trading day in the range %s to %s (inclusive) the rolling %d-trading-day beta of "
        "%s, with %s serving as the benchmark; %s. %s %s" % (rdef, S0, E0, W, ent, bent, wdef, q, fmt),
        "%s Here the beta is the rolling %d-trading-day beta of %s against %s, computed on each trading day from %s "
        "through %s inclusive and based on %s, and %s. %s" % (q, W, ent, bent, S0, E0, rdef, wdef, fmt),
    )


def ma_cross(r):
    p, sym = r.p, r.p["symbol"]
    eq = r.N.kind[sym] == "equity"
    ent, ns, nl = r.ent(sym, 3 if eq else 2), p["short"], p["long"]
    series = "split-adjusted daily closes" if r.N.kind[sym] in ("equity", "etf") else "daily closing levels"
    defs = r.pick(
        "Work with the %d-day and %d-day simple moving averages of the %s of %s; each average includes the day it is "
        "dated and runs over the full price history, so closes before any date named here count too. A golden cross is "
        "dated on the first trading day on which the %d-day average is above the %d-day average, and a death cross on "
        "the first trading day on which it is below the %d-day average." % (ns, nl, series, ent, ns, nl, nl),
        "The signals come from two simple moving averages of the %s of %s, over %d and over %d days, each including "
        "its own day and computed on the whole price history (closes before any date named here included). A golden "
        "cross falls on the first trading day with the %d-day average above the %d-day average; a death cross falls on "
        "the first trading day with the %d-day average below the %d-day "
        "average." % (series, ent, ns, nl, ns, nl, ns, nl))
    P = pct(r)
    if p["mode"] == "next":
        D = r.date(p["date"])
        return r.frame(
            "%s Three crossings are needed: the earliest golden cross on a trading day after %s; then the earliest "
            "death cross after that golden cross; then the earliest golden cross after that death cross. Give their "
            "dates (YYYY-MM-DD) and the price return between the closes of the first golden-cross day and of the "
            "death-cross day, %s." % (defs, D, P),
            "What are the dates of the first golden cross on a trading day after %s, of the earliest death cross that "
            "comes after it, and of the golden cross that follows that death cross? Also give the price return between "
            "the close of the first golden-cross day and the close of the death-cross day, %s, with dates as "
            "YYYY-MM-DD. %s" % (D, P, defs),
            "Give three dates in YYYY-MM-DD form and one return: the first golden cross on a trading day after %s, the "
            "first death cross after that golden cross, the next golden cross after that death cross, and the price "
            "return from the close on the first golden-cross day to the close on the death-cross day (%s). %s"
            % (D, P, defs),
        )
    S0, E0 = p["start"], p["end"]
    rule = r.pick(
        "Run a long-only crossover backtest over %s to %s, both days included: begin in cash, buy at the close of "
        "every golden-cross day and sell at the close of every death-cross day within that period, close any position "
        "still open at the close on %s, and assume no costs and no dividends." % (S0, E0, E0),
        "Backtest this long-only rule from %s through %s inclusive, with no costs and no dividends: start with cash, "
        "buy at the close on each golden-cross day and sell at the close on each death-cross day within that period, "
        "and sell whatever is still held at the close on %s." % (S0, E0, E0))
    q = r.pick(
        "Report how many buys the rule makes, its compounded return, the buy-and-hold price return from the close on "
        "%s to the close on %s, and, for the rule's most profitable round trip (a position sold at the end counts as "
        "one), its buy date (YYYY-MM-DD) and price return; give each return %s." % (S0, E0, P),
        "Give the number of buys, the compounded return of the rule, the buy-and-hold price return from the close on "
        "%s to the close on %s, and the buy date (YYYY-MM-DD) and price return of the most profitable round trip of "
        "the rule (counting a position sold at the end as one), every return %s." % (S0, E0, P))
    return r.frame("%s %s %s" % (rule, defs, q), "%s %s %s" % (rule, q, defs), "%s %s %s" % (q, rule, defs))


def reinvest_return(r):
    p = r.p
    ent, S, E = r.ent(p["symbol"], 3), r.date(p["start"]), r.date(p["end"])
    nt = nt_rule(r)
    split = "every split multiplies the number of shares by its ratio (new shares per old share)"
    ledger = r.frame(
        "The position is one share of %s, purchased at the unadjusted close on %s and still held at the unadjusted "
        "close on %s. %s. Dividends go back into the stock manually: for a cash dividend whose ex-dividend date is "
        "later than the purchase day but not later than the last day, take the per-share amount as paid at the time on "
        "the shares then held and buy fractional shares with it at the unadjusted close of that ex-dividend date. %s"
        % (ent, S, E, cap(split), nt),
        "Keep a hand-made ledger of reinvested dividends for one share of %s, bought at the unadjusted close on %s and "
        "kept through the unadjusted close on %s. %s %s. Each cash dividend with an ex-dividend date after the "
        "purchase day, up to and including the last day, is reinvested in fractional shares at the unadjusted close on "
        "its ex-dividend date, at the per-share amount as paid at the time." % (ent, S, E, nt, cap(split)),
        "%s An investor holds one share of %s from the unadjusted close on %s through the unadjusted close on %s and "
        "reinvests dividends manually. For each cash dividend with an ex-dividend date after the purchase day and on "
        "or before the last day, the amount per share as paid at the time is reinvested at the unadjusted close of the "
        "ex-dividend date (in fractional shares); %s." % (nt, ent, S, E, split),
    )
    P = pct(r)
    if not p["fx"]:
        q = r.pick("Give three numbers: the share count at the end, rounded to four decimals; the total dividend cash "
                   "reinvested, in US dollars rounded to two decimals; and the position's total return %s." % P,
                   "What are the final number of shares (to four decimals), the dividend cash reinvested in total (in "
                   "US dollars, to two decimals) and the total return of the position, %s?" % P)
        return ledger + " " + q
    sid = p["fx"]
    home, qdesc, _ = FX_HOME[sid]
    q = r.pick(
        "The investor measures results in %s: translate the purchase cost and the final value with FRED series %s "
        "(%s), at its observation on the purchase day and on the last day, or the most recent earlier observation. "
        "Report the final number of shares (four decimals), the total dividend cash reinvested in US dollars (two "
        "decimals), and the total return both in US dollars and in %s, each return %s." % (home, sid, qdesc, home, P),
        "Results are counted in %s, so convert the cost on the purchase day and the final value on the last day with "
        "FRED series %s (%s), taking that date's observation or the most recent earlier one. Give the share count at "
        "the end (four decimals), the dividend cash reinvested in total in US dollars (two decimals), the total return "
        "in US dollars and the total return in %s, every return %s." % (home, sid, qdesc, home, P))
    return ledger + " " + q


def q4_margin_rank(r):
    p = r.p
    names = "; ".join(r.ent(s, 3) for s in p["symbols"])
    T, _, ndesc = MARGINS[p["metric"]]
    F, dr = p["fy"], p["direction"]
    rev = rev_rule(r, "the 10-K")
    defs = r.pick(
        "Q4 figures are implied ones: from the full-year amount reported in the original Form 10-K for fiscal year %d, "
        "subtract the nine-month year-to-date amount reported in the first-filed Q3 Form 10-Q of that fiscal year, "
        "with fiscal years labelled the way each company labels them. Margin numerator: %s. %s; keep that same tag in "
        "both periods." % (F, ndesc, cap(rev)),
        "Derive each implied Q4 figure by subtraction, annual amount minus nine-month year-to-date amount, where the "
        "annual amount comes from the original Form 10-K for fiscal year %d and the year-to-date amount from the "
        "first-filed Q3 Form 10-Q of that fiscal year; fiscal years follow each company's own labels. The margin "
        "divides %s by revenue, and %s, one tag serving both periods." % (F, ndesc, rev),
        "Each Q4 amount here is an implied one: subtract what the first-filed Q3 Form 10-Q of fiscal year %d reports "
        "for the nine-month year-to-date period from what the original Form 10-K of fiscal year %d reports for the "
        "full year (fiscal years as each company labels them). Numerator: %s. %s (the same tag for both "
        "periods)." % (F, F, ndesc, cap(rev)))
    more = (r.pick(", its %s for the full fiscal year (from the 10-K amounts)" % T,
                   ", its full-fiscal-year %s (from the 10-K amounts)" % T) if p["ask_more"] else "")
    q = r.pick(
        "Find the company whose Q4 %s for fiscal year %d is the %s and give that margin; also report its Q4 revenue in "
        "millions of US dollars (one decimal)%s, and the Q4 %s of the runner-up, the company with the second-%s one. "
        "Express margins %s." % (T, F, dr, more, T, dr, pct(r)),
        "Answer with the company that had the %s Q4 %s in fiscal year %d together with that margin, its Q4 revenue in "
        "millions of US dollars rounded to one decimal%s, and the Q4 %s of the company in second place (the "
        "second-%s). Margins go %s." % (dr, T, F, more, T, dr, pct(r)),
        "Report the company with the %s Q4 %s in fiscal year %d and that margin, its Q4 revenue in millions of US "
        "dollars to one decimal%s, and the Q4 %s of the company ranked second-%s. Margins %s." % (dr, T, F, more, T, dr,
                                                                                               pct(r)))
    return r.frame(
        "%s The companies are: %s. %s" % (q, names, defs),
        "%s %s The companies to compare: %s." % (q, defs, names),
        "Here are the companies: %s. %s %s" % (names, q, defs),
    )


def filing_reaction(r):
    p, d = r.p, r.d
    names = "; ".join(r.ent(s, 3 if d == 5 else 2) for s in p["symbols"])
    F, N, dr = p["fy"], p["n"], p["direction"]
    tdays = "1 trading day" if N == 1 else "%d trading days" % N
    s1 = r.pick(
        "compare revenue growth across the companies, fiscal year %d against fiscal year %d (year-over-year), and "
        "select the company where it is %s; take each year's revenue from that year's own original Form 10-K annual "
        "report, and in each year %s" % (F, F - 1, dr, rev_rule(r, "that year's 10-K")),
        "rank the companies by how much their revenue grew year-over-year from fiscal year %d to fiscal year %d and "
        "keep the one with the %s growth, each year's revenue coming from the original Form 10-K filed for that year "
        "(in each year, %s)" % (F - 1, F, dr, rev_rule(r, "that year's 10-K")),
        "work out, for every company, the year-over-year change in revenue between fiscal year %d and fiscal year %d, "
        "and keep the company whose growth is the %s; read each year's revenue from that same year's own original "
        "annual report on Form 10-K (%s)" % (F - 1, F, dr, rev_rule(r, "that year's 10-K")))
    tot = r.pick("total return with dividends included (based on the close adjusted for both splits and dividends)",
                 "total return including dividends (computed from closes adjusted for both splits and dividends)")
    s2 = r.pick(
        "measure that company's stock against the S&P 500 index (^GSPC) around the SEC filing of its original Form "
        "10-K for fiscal year %d: day 0 is the filing date, or the next trading day if the filing date is not one; day "
        "-1 is the trading day before day 0 and day +%d is %s after day 0; the excess return is the stock's %s from "
        "the close on day -1 to the close on day +%d, less the index's price return between the same two closes"
        % (F, N, tdays, tot, N),
        "compare that company's stock with the S&P 500 index (^GSPC) over a window around its filing: with day 0 the "
        "date its original Form 10-K for fiscal year %d was filed with the SEC, or the following trading day when no "
        "trading took place that date, the window runs from the close on day -1 (the trading day before day 0) to the "
        "close on day +%d (%s after day 0); compute the stock's %s over the window and deduct the index's price return "
        "over that same window" % (F, N, tdays, tot))
    grow = ("its revenue growth %s, " % pct(r)) if p["ask_growth"] else ""
    ask = r.pick("Answer with the company, %sthe filing date (YYYY-MM-DD) and, in percentage points to two decimals, "
                 "the excess return (negative if the stock lagged the index)." % grow,
                 "The answer names the company, gives %sthe filing date in YYYY-MM-DD form, and states the excess "
                 "return in percentage points to two decimals, negative when the stock lagged the index." % grow)
    st = steps(r, [s1, s2], "(1) " in r.orig)
    return r.frame(
        "%s The companies are %s, and the answer takes two steps. %s" % (ask, names, st),
        "%s %s The companies are: %s." % (st, ask, names),
        "This task has two steps and covers the following companies: %s. %s %s" % (names, ask, st),
    )


def month_chain(r):
    p, d = r.p, r.d
    sym = p["symbol"]
    kind = r.N.kind[sym]
    ent = r.ent(sym, 3 if d == 5 and kind == "equity" else 2)
    first, last, dr = month_label(p["first_month"]), month_label(p["last_month"]), p["direction"]
    adj = ("closes adjusted for splits and dividends (total return)" if p["total"] else
           "split-adjusted closes" if kind in ("equity", "etf") else "closing levels")
    if p["vintage"] == "as_of_cutoff":
        vt = r.pick("the FRED vintage of 2025-10-30, the values as published then (the snapshot's as-of-cutoff values, "
                    "which are the default)",
                    "the values published in the FRED vintage of 2025-10-30 (the default: the snapshot's as-of-cutoff "
                    "values)")
    else:
        vt = r.pick("the latest vintage (vintage='latest', today's values with the revisions after 2025-10-30 "
                    "included)",
                    "the latest vintage (vintage='latest'), which holds today's values including revisions after "
                    "2025-10-30")
    s1 = r.pick(
        "look at the monthly returns of %s in each calendar month from %s through %s and pick the month with the %s "
        "one, a month's return being computed from %s as the close on the month's final trading day divided by the "
        "close on the previous month's final trading day, minus 1" % (ent, first, last, dr, adj),
        "find which calendar month between %s and %s, both included, gave %s its %s monthly return, using %s and "
        "dividing the close on each month's last trading day by the close at the end of the month before, minus 1"
        % (first, last, ent, dr, adj))
    s2 = r.pick(
        "convert that nominal return into an inflation-adjusted (real) one with (1 + nominal) / (1 + inflation) - 1, "
        "where inflation is the month-over-month change in CPIAUCSL (FRED, monthly), that month's value over the "
        "previous month's value minus 1, as given by %s" % vt,
        "deflate that month's return: the inflation-adjusted return is (1 + nominal) / (1 + inflation) - 1, where "
        "inflation is CPIAUCSL (FRED, monthly) of that month over CPIAUCSL of the previous month, minus 1, using "
        "%s" % vt)
    parts = [s1, s2]
    if p["ask_yield"]:
        parts.append(r.pick(
            "measure the move in the 10-year Treasury constant-maturity yield (FRED DGS10) from the final trading day "
            "of the previous month to the final trading day of that month, reading each date's observation or else the "
            "most recent earlier one, in basis points rounded to a whole number, negative for a fall",
            "report the change in the 10-year Treasury constant-maturity yield (FRED DGS10) between the previous "
            "month's last trading day and that month's last trading day, taking the observation on each of those dates "
            "or the most recent earlier one, in basis points rounded to a whole number, negative for a fall",
            "give how much the 10-year Treasury constant-maturity yield (FRED DGS10) moved from the previous month's "
            "last trading day to that month's last trading day, with each date's observation or the most recent "
            "earlier one, in basis points rounded to a whole number (negative for a fall)"))
    ident = r.pick("Answer with the date of the month's last trading day (YYYY-MM-DD) to identify it, plus its nominal "
                   "and its real monthly return %s%s." % (pct(r), " and the yield change" if p["ask_yield"] else ""),
                   "Name the month by the date of its last trading day (YYYY-MM-DD), and report its nominal and real "
                   "monthly returns %s%s." % (pct(r), ", plus the yield change" if p["ask_yield"] else ""),
                   "Identify the month through the date of its last trading day (YYYY-MM-DD) and give the nominal and "
                   "the real monthly return %s%s." % (pct(r), ", together with the yield change" if p["ask_yield"] else
                                                      ""))
    st = steps(r, parts, "(1) " in r.orig)
    return r.frame(
        "%s Work it out in %s steps. %s" % (ident, "three" if p["ask_yield"] else "two", st),
        "%s Proceed as follows. %s" % (ident, st),
        "%s When you have the month, %s" % (st, ident[0].lower() + ident[1:]),
    )


# ============================================================================
# Tier 1 (templates.py)

def extreme_period(r):
    p, d = r.p, r.d
    sym = p["symbol"]
    ent = r.ent(sym, d)
    px = "split-adjusted close" if r.N.kind[sym] in ("equity", "etf") else "closing level"
    gain = p["direction"] == "gain"
    word = "largest gain" if gain else r.pick("largest loss (the most negative return)",
                                              "largest loss, meaning the most negative return")
    if p["freq"] == "month":
        f, l = month_label(p["first"][:7]), month_label(p["last"][:7])
        span = r.pick("among the months %s through %s" % (f, l), "out of the months from %s through %s" % (f, l))
        defn = r.pick("Returns run from month-end to month-end: the month's final %s over the final %s of the month "
                      "before, minus 1." % (px, px),
                      "Take each month's return from month-end to month-end: divide the month's last %s by the last %s "
                      "of the month before and subtract 1." % (px, px))
        ident = r.pick("State the month", "Give the name of the month")
        which = r.pick("State the month", "Give the name of the month")
    else:
        span = r.pick("among the calendar weeks (each running Monday to Sunday) with their last trading day between %s "
                      "and %s inclusive" % (p["first"], p["last"]),
                      "out of the calendar weeks (Monday to Sunday) whose final trading day is on or after %s and on "
                      "or before %s" % (p["first"], p["last"]))
        defn = r.pick("Returns run from week-end to week-end: the %s at the week's final trading day over the %s at "
                      "the final trading day of the week before, minus 1." % (px, px),
                      "Take each week's return from week-end to week-end: divide the %s of the week's final trading "
                      "day by the %s of the previous week's final trading day and subtract 1." % (px, px))
        ident = r.pick("Give the week as the date of its last trading day (YYYY-MM-DD)",
                       "Label the week with the date of its last trading day (YYYY-MM-DD)")
        which = r.pick("Give, as the date of its last trading day (YYYY-MM-DD), the week",
                       "Label with the date of its last trading day (YYYY-MM-DD) the week")
    P = pct(r)
    return r.frame(
        "%s, in which %s did %s have its %s? %s %s and give that return %s." % (cap(span), p["freq"], ent, word, defn,
                                                                             ident, P),
        "Which %s %s shows the %s for %s? %s and give that return %s. %s" % (p["freq"], span, word, ent, ident, P,
                                                                         defn),
        "%s in which %s posted its %s %s, and report that return %s. %s" % (which, ent, word, span, P, defn),
    )


def rolling_vol(r):
    p, d = r.p, r.d
    syms = p["symbols"]
    spread = len(syms) == 2
    adj = spread or r.N.kind[syms[0]] in ("equity", "etf")
    base = r.pick(
        "Work with daily returns in percent, each equal to the close divided by the previous trading day's close, "
        "minus 1%s. On every trading day, compute the sample standard deviation (n-1 denominator) of the 20 most "
        "recent daily returns up to and including that day; these may reach back before the window starts."
        % (" (using split-adjusted closes)" if adj else ""),
        "A daily return is close / previous trading day's close - 1%s, expressed in percent. For each trading day, the "
        "rolling value is the sample standard deviation (n-1 denominator) of the 20 latest daily returns ending on "
        "that day, which may go back before the start of the window." % (", from split-adjusted closes" if adj else ""))
    if spread:
        ea, eb, s, e = r.ent(syms[0], 2), r.ent(syms[1], 2), p["start"], p["end"]
        sp = r.pick("The daily spread is the daily return of %s minus the daily return of %s, on dates when both trade."
                    % (ea, eb),
                    "Take as the daily spread the daily return of %s less that of %s, keeping only dates on which both "
                    "trade." % (ea, eb))
        out = r.pick("Report the average in percentage points, rounded to three decimals.",
                     "Give that average in percentage points to three decimals.")
        return r.frame(
            "%s %s Average the rolling 20-day standard deviation of that spread across every trading day from %s to "
            "%s. %s" % (sp, base, s, e, out),
            "What is the average, over the trading days %s through %s, of the rolling 20-day standard deviation of a "
            "daily return spread? %s %s %s" % (s, e, sp, base, out),
            "Report, in percentage points to three decimals, the average over all trading days from %s to %s of the "
            "rolling 20-day standard deviation of a daily return spread. %s %s" % (s, e, sp, base),
        )
    ent, s, e = r.ent(syms[0], d), r.date(p["start"]), r.date(p["end"])
    eg = " (for example 1.234)" if "(e.g. 1.234)" in r.orig else ""
    out = r.pick("Give the average in percent to three decimals%s." % eg,
                 "Report that average in percent, rounded to three decimals%s." % eg)
    return r.frame(
        "Take %s and average its rolling 20-day standard deviations of daily returns over every trading day from %s "
        "through %s. %s %s" % (ent, s, e, base, out),
        "%s Averaged over all trading days from %s through %s, what is the rolling 20-day volatility of %s? %s"
        % (base, s, e, ent, out),
        "Compute the mean rolling 20-day volatility of the daily returns of %s across the trading days from %s through "
        "%s. %s %s" % (ent, s, e, base, out),
    )


RATIO_ALT = {
    "net_margin": ("net profit margin",
                   ("net income (us-gaap:NetIncomeLoss) over revenue, revenue being us-gaap:Revenues when the filing "
                    "reports that tag for the year, else us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax, "
                    "else us-gaap:SalesRevenueNet",
                    "net income (us-gaap:NetIncomeLoss) as a share of revenue, with revenue taken as us-gaap:Revenues "
                    "if the filing reports that tag for the year, failing that "
                    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax, failing that "
                    "us-gaap:SalesRevenueNet"),
                   True),
    "roe": ("return on average equity",
            ("net income (us-gaap:NetIncomeLoss) over the mean of total stockholders' equity "
             "(us-gaap:StockholdersEquity) at the beginning and at the end of the fiscal year, both taken as presented "
             "on that year's 10-K balance sheet",
             "net income (us-gaap:NetIncomeLoss) divided by the average of the opening and closing total stockholders' "
             "equity (us-gaap:StockholdersEquity) of the fiscal year, both as shown on that year's 10-K balance sheet"),
            True),
    "inv_turnover": ("inventory turnover",
                     ("cost of sales over the mean of net inventory (us-gaap:InventoryNet) at the beginning and at the "
                      "end of the fiscal year, both from that year's 10-K balance sheet, with cost of sales taken as "
                      "us-gaap:CostOfGoodsAndServicesSold if reported, else us-gaap:CostOfRevenue, else "
                      "us-gaap:CostOfGoodsSold",
                      "cost of sales divided by the average of the opening and closing net inventory "
                      "(us-gaap:InventoryNet) of the fiscal year, both from that year's 10-K balance sheet; cost of "
                      "sales is us-gaap:CostOfGoodsAndServicesSold when reported, otherwise us-gaap:CostOfRevenue, "
                      "otherwise us-gaap:CostOfGoodsSold"),
                     False),
}
ENDS = re.compile(r" Fiscal years are as each company labels them: (.+?)\.(?= Which company| Average each)")
END_ITEM = re.compile(r"FY(\d{4}) ended (\d{4}-\d{2}-\d{2})")


def and_list(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def xbrl_ratio(r):
    p, d = r.p, r.d
    names = ", ".join(r.ent(s, d) for s in p["symbols"])
    T, defs, is_pct = RATIO_ALT[p["ratio"]]
    defn = r.pick(*defs)
    ys, dr = p["years"], p["direction"]
    ytxt = (r.pick("fiscal years %d to %d inclusive" % (ys[0], ys[-1]), "each fiscal year from %d through %d"
                   % (ys[0], ys[-1])) if len(ys) == 3 else
            r.pick("the two fiscal years %d and %d" % (ys[0], ys[-1]), "fiscal years %d and %d" % (ys[0], ys[-1])))
    unit = pct(r) if is_pct else r.pick("as a multiple (times), to two decimals", "in times, rounded to two decimals")
    src = " (SEC XBRL data)" if "(SEC XBRL data)" in r.orig else ""
    ends = ""
    m = ENDS.search(r.orig)
    if m:
        items = []
        for chunk in m.group(1).split("; "):
            sym, rest = chunk.split(" ", 1)
            items.append("for %s, %s" % (sym, and_list(["FY%s ended %s" % x for x in END_ITEM.findall(rest)])))
        ends = " " + r.pick("Fiscal years follow each company's own labels: ", "Each company's own fiscal-year labels "
                                                                               "apply: ") + "; ".join(items) + "."
    elif d < 3:
        raise ValueError("%s: no fiscal-year end list" % r.t["task_id"])
    return r.frame(
        "Which of these companies had the %s average %s over %s: %s? The %s is %s. Take each year's value from that "
        "fiscal year's original annual report on Form 10-K%s and average the yearly values with a simple mean.%s "
        "Report the company and its average %s." % (dr, T, ytxt, names, T, defn, src, ends, unit),
        "Using the original annual report on Form 10-K of each fiscal year%s, work out the %s of %s in %s, the %s "
        "being %s.%s Take the simple mean of each company's yearly values, then give the company with the %s average "
        "and that average, %s." % (src, T, names, ytxt, T, defn, ends, dr, unit),
        "Rank %s by their average %s across %s (the simple mean of the yearly values), with the %s defined as %s and "
        "each year's figure from that year's original Form 10-K%s.%s Which company comes out %s, and what is its "
        "average (%s)?" % (names, T, ytxt, T, defn, src, ends, dr, unit),
    )


RENDER = {"risk_rank": risk_rank, "rolling_beta": rolling_beta, "ma_cross": ma_cross,
          "reinvest_return": reinvest_return, "q4_margin_rank": q4_margin_rank, "filing_reaction": filing_reaction,
          "month_chain": month_chain, "extreme_period": extreme_period, "rolling_vol": rolling_vol,
          "xbrl_ratio": xbrl_ratio}


def reword(t, N, seed):
    """(new prompt, wording index, entity strings rendered) of task t."""
    r = R(t, N, seed)
    s = RENDER[t["template"]](r)
    return s.replace("..", "."), r.k, r.ents  # a name ending in "Inc." before a full stop, as templates.mk does


# ============================================================================
# Checks

ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
LONG = re.compile(r"\b(?:%s) \d{1,2}, \d{4}\b" % "|".join(MONTHS))
MONTH = re.compile(r"\b(?:%s) \d{4}\b" % "|".join(MONTHS))
RESULT_NUM = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?")  # harness/envs/market.py's
TAG = re.compile(r"\bus-gaap:\w+")
ROUND = re.compile(r"\b(one|two|three|four) decimal", re.I)
FIXED_IDS = ["^GSPC", "DTB3", "CPIAUCSL", "DGS10"] + sorted(FX_HOME)
# Phrases that state an option, a convention or a unit: each is in both the original and the rewording, or in neither.
OPTIONS = ["Sharpe", "Sortino", "highest", "lowest", "second-highest", "second-lowest", "gain", "loss", "month",
           "monthly", "week", "golden", "death", "operating margin", "net margin", "gross margin",
           "net profit margin", "return on average equity", "inventory turnover", "excess return", "total return",
           "price return", "risk-free", "split-adjusted", "unadjusted", "adjusted for splits and dividends",
           "closing level", "closing levels", "whole number", "percentage points", "basis points", "millions",
           "US dollars", "YYYY-MM-DD", "n-1", "square root of 252", "Monday to Sunday", "as-of-cutoff",
           "vintage='latest'", "latest vintage", "ex-dividend", "fractional shares", "new shares per old share",
           "long-only", "buy-and-hold", "round trip", "first-filed", "nine-month", "year-to-date", "implied", "Q3",
           "Q4", "10-K", "10-Q", "constant-maturity", "inflation-adjusted", "nominal", "real", "FRED", "SEC",
           "S&P 500", "simple moving averages", "sample standard deviation", "downside deviation", "rolling", "beta",
           "Treasury bill", "3-month", "10-year", "20-day", "day 0", "day -1", "filing date", "year-over-year", "XBRL",
           "Yahoo", "front-month", "futures", "exchange rate", "trading day", "calendar", "spread", "no costs",
           "dividends", "inflation", "CPI"]


def note_numbers(text):
    """The numbers Episode.note_numbers records (harness/agent.py)."""
    return {round(float(n), 2) for n in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))}


def result_numbers(text):
    return {float(m.replace(",", "")) for m in RESULT_NUM.findall(text)}


def shows(sym, text):
    if len(sym) > 1:
        return re.search(r"(?<![\w^=.-])%s(?![\w=-])" % re.escape(sym), text) is not None
    return re.search(r"\(%s\)|for %s,|(?<![\w.-])%s FY" % ((re.escape(sym),) * 3), text) is not None


def phrase(ph, text):
    return re.search(r"(?<![\w-])%s(?![\w])" % re.escape(ph), text, re.I) is not None


def mask(text, ents):
    for e in sorted(set(ents), key=len, reverse=True):
        text = text.replace(e, " ENT ").replace(e.rstrip("."), " ENT ")
    for rx, tok in ((TAG, " TAG "), (LONG, " DATE "), (ISO, " DATE "), (MONTH, " DATE "), (RESULT_NUM, " NUM ")):
        text = rx.sub(tok, text)
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in ("ent", "tag", "date", "num")]


def distance(a, b, ents):
    return 1.0 - difflib.SequenceMatcher(None, mask(a, ents), mask(b, ents), autojunk=False).ratio()


def check_one(line, t, new, ents):
    """(failed checks, empty when the rewording passes; its distance to the original)."""
    a, b, bad = t["prompt"], new, []
    for name, f in (("iso dates", lambda s: set(ISO.findall(s))), ("long dates", lambda s: set(LONG.findall(s))),
                    ("months", lambda s: set(MONTH.findall(s))), ("numbers", result_numbers),
                    ("noted numbers", note_numbers), ("tags", lambda s: set(TAG.findall(s))),
                    ("rounding", lambda s: {x.lower() for x in ROUND.findall(s)})):
        x, y = f(a), f(b)
        if x != y:
            bad.append("%s: original only %s, reworded only %s" % (name, sorted(x - y), sorted(y - x)))
    p = t["meta"]["params"]
    syms = set(t["meta"]["entities"]) | set(p.get("symbols", [])) | set(FIXED_IDS)
    syms |= {p[k] for k in ("symbol", "bench", "fx") if p.get(k)}
    for s in sorted(x for x in syms if not x.startswith("CIK:")):
        if shows(s, a) != shows(s, b):
            bad.append("symbol %s shown in %s only" % (s, "original" if shows(s, a) else "rewording"))
    for ph in OPTIONS:
        x, y = phrase(ph, a), phrase(ph, b)
        if x != y:
            bad.append("phrase %r in %s only" % (ph, "original" if x else "rewording"))
    for e in ents:
        if not (e in a or e.rstrip(".") in a):
            bad.append("entity %r not in the original" % e)
        if not (e in b or e.rstrip(".") in b):
            bad.append("entity %r not in the rewording" % e)
    back = json.dumps(dict(json.loads(json.dumps(dict(t, prompt=new), ensure_ascii=False)), prompt=a),
                      ensure_ascii=False)
    if back != line.rstrip("\n"):
        bad.append("fields differ from the original line")
    dist = distance(a, b, ents)
    if dist < MIN_DIST:
        bad.append("distance %.3f < %.2f" % (dist, MIN_DIST))
    return bad, dist


def ngrams(words, n=5):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def train_overlap(rows, train_rows):
    """Per template: mean share of a request's masked 5-grams that occur in the train prompts of the same template."""
    seen = collections.defaultdict(set)
    for t in train_rows:
        if t["template"] in RENDER:
            seen[t["template"]] |= ngrams(mask(t["prompt"], []))
    out = collections.defaultdict(list)
    for t, new, ents in rows:
        g_new, g_old = ngrams(mask(new, [])), ngrams(mask(t["prompt"], []))
        out[t["template"]].append((len(g_new & seen[t["template"]]) / max(1, len(g_new)),
                                   len(g_old & seen[t["template"]]) / max(1, len(g_old))))
    mean = lambda xs: round(sum(xs) / len(xs), 3)
    return {k: {"reworded": mean([x for x, _ in v]), "original": mean([y for _, y in v])}
            for k, v in sorted(out.items())}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", required=True, help="hardset_v1 (dev_market.jsonl, test_market.jsonl, "
                                                       "train_market.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--review", default="", help="write original and reworded requests side by side to this file")
    ap.add_argument("--review-n", type=int, default=2, help="tasks per template and file in --review")
    a = ap.parse_args()
    N = Names()
    with open(os.path.join(a.tasks_dir, "train_market.jsonl"), encoding="utf-8") as f:
        train = [json.loads(x) for x in f if x.strip()]
    outs, manifest, failed, review = {}, {"seed": a.seed, "min_distance": MIN_DIST, "wordings_per_template": NWORD,
                                          "code": os.path.abspath(__file__), "source": os.path.abspath(a.tasks_dir),
                                          "files": {}}, 0, []
    for name in FILES:
        src = os.path.join(a.tasks_dir, name + ".jsonl")
        with open(src, encoding="utf-8") as f:
            lines = [x for x in f if x.strip()]
        rows, counts, fails, dists, by_task = [], collections.defaultdict(collections.Counter), [], \
            collections.defaultdict(list), {}
        for line in lines:
            t = json.loads(line)
            assert json.dumps(t, ensure_ascii=False) == line.rstrip("\n"), "a line does not re-serialize as read"
            new, k, ents = reword(t, N, a.seed)
            bad, dist = check_one(line, t, new, ents)
            counts[t["template"]][k] += 1
            dists[t["template"]].append(dist)
            by_task[t["task_id"]] = k
            if bad:
                fails.append({"task_id": t["task_id"], "failed": bad})
            rows.append((t, new, ents))
        failed += len(fails)
        why = collections.Counter((f_["task_id"].split("-")[-2], re.sub(r"[\d.]+|\[.*?\]", "#", x))
                                  for f_ in fails for x in f_["failed"])
        for (tpl, x), n in sorted(why.items()):
            print("FAIL %s %s %s: %d" % (name, tpl, x, n))
        seen = collections.Counter()
        for t, new, _ in rows:
            if seen[t["template"]] < a.review_n:
                seen[t["template"]] += 1
                review += ["=" * 100, "%s  %s  (difficulty %d, wording %d)" % (
                               name, t["task_id"], t["meta"]["difficulty"], by_task[t["task_id"]]),
                           "PARAMS: " + json.dumps(t["meta"]["params"]), "ORIGINAL: " + t["prompt"], "REWORDED: " + new,
                           ""]
        outs[name] = rows
        manifest["files"][name + ".jsonl"] = {
            "source_sha256": sha(src), "tasks": len(rows), "failed_checks": len(fails), "failures": fails,
            "wording_counts": {k: {str(i): v[i] for i in range(NWORD)} for k, v in sorted(counts.items())},
            "distance": {k: {"min": round(min(v), 3), "mean": round(sum(v) / len(v), 3)}
                         for k, v in sorted(dists.items())},
            "train_5gram_overlap": train_overlap(rows, train), "wording_by_task": by_task}
    if a.review:
        with open(a.review, "w") as f:
            f.write("\n".join(review))
    if failed:
        print("%d reworded tasks failed a check; nothing written" % failed)
        sys.exit(1)
    os.makedirs(a.out, exist_ok=True)
    for name, rows in outs.items():
        p = os.path.join(a.out, name + ".jsonl")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            for t, new, _ in rows:
                f.write(json.dumps(dict(t, prompt=new), ensure_ascii=False) + "\n")
        os.replace(p + ".tmp", p)
        manifest["files"][name + ".jsonl"]["sha256"] = sha(p)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: {"tasks": v["tasks"], "failed_checks": v["failed_checks"]}
                      for k, v in manifest["files"].items()}))


if __name__ == "__main__":
    main()
