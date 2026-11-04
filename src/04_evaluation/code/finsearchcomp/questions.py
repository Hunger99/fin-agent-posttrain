"""Reference computations against market_v1 for the FinSearchComp Global T2/T3 candidates.

Each function is registered under the short id (T2_536, T3_b004) and returns a dict:
  keys     grading keys built from the benchmark's reference answer (harness v3 grade_keys schema); their values are
           read at run time from a local copy of the FinSearchComp data (SRC below), so this file holds no answers;
  snap     the snapshot's value for each key, in the same order (a float, or a string for text/ticker/date keys);
  append   sentences appended to the question (a fact the question assumes, or a convention clarification);
  sol      a short description of the computation;
  src      citations of the snapshot rows and of any public fact appended;
  note     optional remarks (what was investigated, why a part is not graded).
"""
import calendar
import datetime as dt
import json
import os
import re

from snap import (bar, bars, cite, first_of, fred, fred_range, next_days, on_or_before, pct, prev_bar, q, value)

Q = {}


def reg(pid):
    def deco(f):
        def run():  # the readers below take the values of the keys from this question's reference answer
            _CUR.append(pid)
            try:
                return f()
            finally:
                _CUR.pop()
        Q[pid] = run
        return f
    return deco


def N(v, tol):
    return {"kind": "number", "value": float(v), "tol": float(tol)}


def NR(lo, hi):
    """A reference range [lo, hi] as a midpoint with half-width tolerance."""
    return {"kind": "number", "value": round((lo + hi) / 2, 6), "tol": round((hi - lo) / 2, 6)}


def NREL(v, rel):
    return {"kind": "number", "value": float(v), "rel_tol": float(rel)}


def T(v, *aliases):
    return {"kind": "text", "value": v, "aliases": list(aliases)}


def TK(v, *aliases):
    return {"kind": "ticker", "value": v, "aliases": list(aliases)}


def D(v):
    return {"kind": "date", "value": v}


def R(keys, snap, sol, src, append=(), note=""):
    return {"keys": keys, "snap": snap, "sol": sol, "src": src, "append": list(append), "note": note}


# -- the reference answers, read from the local FinSearchComp copy ---------------------------------------------------
# The keys' values are not written in this file: each comes from the question's response_reference in the benchmark's
# own data file (SRC), as its i-th number (A), its i-th date (RD), month (RM) or company (RTK), or a pattern's group (RE).

SRC = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/finsearchcomp/finsearchcomp_data.json"
_REF = {}
_CUR = []


def short(pid):
    return pid.replace("(T2)Simple_Historical_Lookup_", "T2_").replace("(T3)Complex_Historical_Investigation_", "T3_")


def ref():
    """The response_reference of the question being built."""
    if not _REF:
        with open(SRC, encoding="utf-8") as f:
            _REF.update((short(r["prompt_id"]), r.get("response_reference") or "") for r in json.load(f))
    return _REF[_CUR[-1]]


NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
MONTHS = {m: i for i, m in enumerate(calendar.month_name) if m}
DATE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})|(%s) (\d{1,2}), (\d{4})" % "|".join(MONTHS))
MONTH = re.compile(r"\b(%s),? (\d{4})\b" % "|".join(MONTHS))


def A(i):
    """The i-th number of the reference answer, commas dropped."""
    return float(NUM.findall(ref())[i].replace(",", ""))


def RE(pattern):
    return re.search(pattern, ref(), re.M).group(1)


def RD(i):
    """A date key for the i-th date of the reference answer (2021/3/5, 2021-03-05 or March 5, 2021)."""
    m = list(DATE.finditer(ref()))[i]
    y, mo, d = (m.group(1), m.group(2), m.group(3)) if m.group(1) else (m.group(6), MONTHS[m.group(4)], m.group(5))
    return D(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}")


def RMS():
    """The months ("March 2021") of the reference answer, as YYYY-MM."""
    return [f"{int(y):04d}-{MONTHS[n]:02d}" for n, y in MONTH.findall(ref())]


def RM(i):
    return RMS()[i]


def month_label(ym):
    return f"{calendar.month_name[int(ym[5:])]} {ym[:4]}"


def month_key(ym):
    name, ab, y = calendar.month_name[int(ym[5:])], calendar.month_abbr[int(ym[5:])], ym[:4]
    return T(ym, f"{name} {y}", f"{ab} {y}", f"{name}, {y}", f"{y}/{ym[5:]}", f"{ab}. {y}")


# the companies of the ticker questions: the names a ticker key accepts besides the ticker
COMPANIES = {"BAC": ("Bank of America",), "WFC": ("Wells Fargo",), "TSM": ("TSMC", "Taiwan Semiconductor"),
             "BA": ("Boeing",), "GS": ("Goldman Sachs",), "MRK": ("Merck",), "IBM": ("International Business Machines",),
             "NVDA": ("NVIDIA",), "META": ("Facebook", "Meta Platforms", "Meta"), "TSLA": ("Tesla",),
             "AAPL": ("Apple",)}


def TKN(name):
    """The ticker key of the company with this name."""
    tk = next(t for t, ns in COMPANIES.items() if name.lower() in [x.lower() for x in (t,) + ns])
    return TK(tk, *COMPANIES[tk])


def RTK(i):
    """The ticker key of the i-th company the reference answer names (by first mention)."""
    text, first = ref().lower(), {}
    for tk, ns in COMPANIES.items():
        hits = [m.start() for n in (tk,) + ns for m in re.finditer(r"\b%s\b" % re.escape(n.lower()), text)]
        if hits:
            first[tk] = min(hits)
    tk = sorted(first, key=first.get)[i]
    return TK(tk, *COMPANIES[tk])


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def ratio_key(r):
    a, b = r.split(":")
    return T(r, f"{a}-for-{b}", f"{a} for {b}", f"{WORDS[int(a)]}-for-{WORDS[int(b)]}", f"{b}:{a}", f"{a}-to-{b}",
             f"{b}-for-{a}")


INDICES = {"Nasdaq": ("NASDAQ", "IXIC"), "S&P 500": ("S&P", "SPX", "GSPC")}


def index_key(name):
    return T(name, *INDICES[name])


ORD = ["", "first", "second", "third", "fourth"]


def quarter_key(s):
    y, q = s[:4], int(s[5])
    return T(s, f"Q{q} {y}", f"{y} Q{q}", f"{ORD[q]} quarter of {y}", f"Q{q}'{y[2:]}", f"{q}Q{y[2:]}", f"Q{q} of {y}",
             f"Q{q} FY{y}", f"{y} {ORD[q]} quarter", f"Q{q} fiscal {y}")


M = 1e6
K = 1e3


def fredsrc(sid, date):
    return f"FRED {sid} {date} value_vintage (cutoff 2025-10-30)"


def pxsrc(sym, *dates):
    return f"Yahoo {sym} daily bars {', '.join(dates)}"


# =====================================================================
# T2 FRED
# =====================================================================

@reg("T2_503")
def _():
    rows = fred_range("IORB", "2023-03-01", "2023-03-31")
    return R([N(A(0), 0.1)], [rows[-1][1]], "IORB level at the end of March 2023 (raised from 4.65 on 2023-03-23).",
             [fredsrc("IORB", "2023-03-01..2023-03-31")])


@reg("T2_504")
def _():
    lo, hi = fred("DFEDTARL", "2022-09-30"), fred("DFEDTARU", "2022-09-30")
    return R([N(A(0), 0.0), N(A(1), 0.0)], [lo, hi], "Fed funds target range in force at 2022-09-30 (DFEDTARL, DFEDTARU).",
             [fredsrc("DFEDTARL", "2022-09-30"), fredsrc("DFEDTARU", "2022-09-30")])


def fred_one(sid, date, key, scale=1.0, sol=""):
    v = fred(sid, date) / scale
    return R([key], [v], sol or f"FRED {sid} at {date}" + (f", divided by {scale:g}" if scale != 1 else "") + ".",
             [fredsrc(sid, date)])


@reg("T2_507")
def _(): return fred_one("A191RL1A225NBEA", "2015-01-01", N(A(0), 0.1))


@reg("T2_508")
def _(): return fred_one("NZLNGDPRPCPPPT", "2005-01-01", N(A(0), 0.1))


@reg("T2_513")
def _(): return fred_one("FPCPITOTLZGAUS", "2022-01-01", N(A(0), 0.1))


@reg("T2_514")
def _(): return fred_one("PCAGDPILA646NWDB", "2020-01-01", N(A(0), 0.1))


@reg("T2_567")
def _(): return fred_one("MRTSSM44000USN", "2014-12-01", N(A(0), 1))


@reg("T2_570")
def _(): return fred_one("RUSCPIALLMINMEI", "2015-04-01", N(A(0), 0.01))


@reg("T2_576")
def _(): return fred_one("LRHUTTTTJPM156S", "2009-08-01", NR(A(0), A(1)))


@reg("T2_578")
def _(): return fred_one("BUSINV", "2021-09-01", N(A(0), 0.1), scale=1e3)


@reg("T2_580")
def _(): return fred_one("DSPI", "2025-04-01", N(A(0), 0.1))


@reg("T2_581")
def _(): return fred_one("DGS10", "2023-05-22", N(A(0), 0.01))


@reg("T2_582")
def _(): return fred_one("DGS3MO", "2024-07-03", N(A(0), 0.01))


@reg("T2_597")
def _(): return fred_one("M2SL", "2024-12-01", N(A(0), 1))


@reg("T2_598")
def _(): return fred_one("M2V", "2020-07-01", N(A(0), 0.001))


@reg("T2_599")
def _(): return fred_one("BORROW", "2023-04-01", NR(A(0), A(1)))


@reg("T2_600")
def _(): return fred_one("TOTRESNS", "2021-12-01", N(A(0), 0.1))


@reg("T2_601")
def _(): return fred_one("DTB6", "2022-07-15", NR(A(0), A(1)))


# =====================================================================
# T2 Yahoo
# =====================================================================

def px_one(sym, date, field, key, sol=None):
    b = bar(sym, date)
    return R([key], [float(b[field])], sol or f"{sym} {field} on {date}.", [pxsrc(sym, date)])


@reg("T2_572")
def _(): return px_one("DX-Y.NYB", "2023-03-08", "raw_close", NR(A(0), A(1)))


@reg("T2_583")
def _(): return px_one("^FTSE", "2024-07-04", "raw_close", N(A(0), 0.01))


@reg("T2_584")
def _(): return px_one("^N225", "2024-11-15", "raw_high", N(A(0), 0.01))


@reg("T2_585")
def _(): return px_one("^GDAXI", "2022-05-30", "raw_close", N(A(0), 0.01))


@reg("T2_586")
def _(): return px_one("^VIX", "2022-04-25", "raw_close", N(A(0), 0.01))


@reg("T2_593")
def _(): return px_one("^DJI", "2023-06-07", "raw_close", NR(A(0), A(1)))


# =====================================================================
# T2 SEC
# =====================================================================

def sec_one(ticker, tags, end, key, scale, period="instant", sol=None, sign=1.0, **kw):
    tags = [tags] if isinstance(tags, str) else tags
    v, row = first_of(ticker, tags, end, period=period, **kw)
    return R([key], [sign * v / scale],
             sol or f"{ticker} {row['tag']} for the period ending {end} ({period}), as first reported, / {scale:g}.",
             [cite(row)])


@reg("T2_536")
def _(): return sec_one("NVDA", "RetainedEarningsAccumulatedDeficit", "2022-01-30", N(A(0), 1), M)


@reg("T2_537")
def _():
    r = sec_one("NVDA", "DeferredTaxAssetsNet", "2022-01-30", N(A(0), 1), M)
    r["note"] = ("The reference is DeferredTaxAssetsNet (gross less valuation allowance), not the annotation's first "
                 "candidate DeferredTaxAssetsGross.")
    return r


@reg("T2_540")
def _(): return sec_one("NVDA", "Goodwill", "2022-01-30", N(A(0), 1), M)


@reg("T2_541")
def _(): return sec_one("LE", "InterestExpense", "2021-01-29", N(A(0), 1), K, period="year")


@reg("T2_542")
def _(): return sec_one("LE", "PrepaidExpenseAndOtherAssetsCurrent", "2021-01-29", N(A(0), 1), K)


@reg("T2_543")
def _(): return sec_one("LE", ["AdditionalPaidInCapital", "AdditionalPaidInCapitalCommonStock"], "2021-01-29",
                        N(A(0), 1), K)


@reg("T2_544")
def _(): return sec_one("SOFI", "NetCashProvidedByUsedInFinancingActivities", "2020-12-31", N(A(0), 1), 1,
                        period="any", start="2020-07-10")


@reg("T2_545")
def _(): return sec_one("SOFI", ["Cash", "CashAndCashEquivalentsAtCarryingValue"], "2020-12-31", N(A(0), 1), 1)


@reg("T2_546")
def _(): return sec_one("SOFI", "AccruedLiabilitiesCurrent", "2020-12-31", N(A(0), 1), 1)


@reg("T2_547")
def _(): return sec_one("PLUG", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                        "2016-12-31", N(-A(0), 1), K, period="year")


@reg("T2_549")
def _(): return sec_one("PLUG", ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"], "2016-12-31",
                        N(A(0), 1), K)


@reg("T2_550")
def _():
    r = sec_one("QCOM", ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "2016-09-25",
                N(A(0), 1), M)
    r["note"] = ("QCOM's balance-sheet line 'Total stockholders' equity' includes noncontrolling interests, so the tag "
                 "with them is used, not StockholdersEquity (Qualcomm stockholders only).")
    return r


@reg("T2_551")
def _(): return sec_one("QCOM", ["CommercialPaper", "DebtCurrent"], "2016-09-25", NR(A(0), A(1)), M)


@reg("T2_552")
def _(): return sec_one("QCOM", "LiabilitiesCurrent", "2016-09-25", N(A(0), 1), M)


def kscp_rev():
    return first_of("KSCP", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"], "2021-12-31", period="year")


@reg("T2_553")
def _():
    s, rs = value("KSCP", "SellingAndMarketingExpense", "2021-12-31", period="year")
    r, rr = kscp_rev()
    return R([N(A(0), 1)], [s / r * 100], "KSCP FY2021 SellingAndMarketingExpense / revenue x 100.", [cite(rs), cite(rr)])


@reg("T2_554")
def _():
    g, rg = value("KSCP", "GeneralAndAdministrativeExpense", "2021-12-31", period="year")
    r, rr = kscp_rev()
    return R([N(A(0), 1)], [g / r * 100], "KSCP FY2021 GeneralAndAdministrativeExpense / revenue x 100.", [cite(rg), cite(rr)])


@reg("T2_555")
def _(): return sec_one("KSCP", "AccountsPayableCurrent", "2021-12-31", NR(A(0), A(1)), K)


@reg("T2_556")
def _(): return sec_one("KSCP", ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
                        "2021-12-31", N(-A(0), 1), K)


@reg("T2_557")
def _(): return sec_one("UPST", ["TemporaryEquityCarryingAmountAttributableToParent", "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests"],
                        "2019-12-31", N(A(0), 1), K)


@reg("T2_558")
def _(): return sec_one("UPST", "LiabilitiesAndStockholdersEquity", "2020-12-31", N(A(0), 1), K)


@reg("T2_561")
def _(): return sec_one("PL", "NetCashProvidedByUsedInInvestingActivities", "2023-01-31", N(A(0), 1), K, period="year")


@reg("T2_563")
def _(): return sec_one("PL", "GrossProfit", "2023-01-31", N(A(0), 1), K, period="year")


@reg("T2_566")
def _():
    a, ra = value("GME", "AssetsCurrent", "2016-01-30")
    l, rl = value("GME", "LiabilitiesCurrent", "2016-01-30")
    return R([N(A(0), 0.1)], [(a - l) / M], "GME AssetsCurrent - LiabilitiesCurrent at 2016-01-30, in millions.",
             [cite(ra), cite(rl)])


@reg("T2_607")
def _(): return sec_one("META", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"], "2019-12-31",
                        N(A(0), 1), M, period="year")


@reg("T2_608")
def _(): return sec_one("META", "CashAndCashEquivalentsAtCarryingValue", "2019-12-31", N(A(0), 1), M)


@reg("T2_609")
def _(): return sec_one("SBUX", "Assets", "2020-09-27", N(A(0), 0.1), M)


@reg("T2_610")
def _():
    try:
        return sec_one("SBUX", "LongTermDebt", "2020-09-27", N(A(0), 0.1), M)
    except KeyError:
        c, rc = value("SBUX", "LongTermDebtCurrent", "2020-09-27")
        n, rn = value("SBUX", "LongTermDebtNoncurrent", "2020-09-27")
        return R([N(A(0), 0.1)], [(c + n) / M], "SBUX LongTermDebtCurrent + LongTermDebtNoncurrent at 2020-09-27.",
                 [cite(rc), cite(rn)])


@reg("T2_611")
def _(): return sec_one("MSFT", "StockholdersEquity", "2018-06-30", N(A(0), 1), M)


@reg("T2_612")
def _(): return sec_one("MSFT", "OperatingIncomeLoss", "2018-06-30", N(A(0), 1), M, period="year")


@reg("T2_613")
def _(): return sec_one("AMC", "EarningsPerShareDiluted", "2013-12-31", N(A(0), 0.01), 1, period="year")


@reg("T2_614")
def _(): return sec_one("AMC", ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"], "2013-12-31", N(A(0), 1), K)


@reg("T2_615")
def _(): return sec_one("AAPL", "NetCashProvidedByUsedInOperatingActivities", "2021-09-25", N(A(0), 1), M, period="year")


@reg("T2_616")
def _():
    r = sec_one("AAPL", "NetCashProvidedByUsedInInvestingActivities", "2021-09-25", N(A(0), 1), M, period="year")
    r["keys"][0]["sign_free"] = True
    r["note"] = "Reference accepts either sign; key marked sign_free."
    return r


@reg("T2_617")
def _():
    r = sec_one("AAPL", "NetCashProvidedByUsedInFinancingActivities", "2021-09-25", N(A(0), 1), M, period="year", sign=-1)
    r["keys"][0]["sign_free"] = True
    r["note"] = ("Reference gives the magnitude of cash used; the tag is negative. Key is the magnitude, marked "
                 "sign_free.")
    return r


@reg("T2_618")
def _(): return sec_one("GME", "AssetsCurrent", "2022-01-29", N(A(0), 0.1), M)


@reg("T2_619")
def _(): return sec_one("GME", "PropertyPlantAndEquipmentNet", "2022-01-29", N(A(0), 0.1), M)


@reg("T2_620")
def _():
    try:
        return sec_one("GME", "Liabilities", "2022-01-29", N(A(0), 0.1), M)
    except KeyError:
        t, rt = value("GME", "LiabilitiesAndStockholdersEquity", "2022-01-29")
        e, re_ = value("GME", "StockholdersEquity", "2022-01-29")
        return R([N(A(0), 0.1)], [(t - e) / M],
                 "GME has no Liabilities tag at 2022-01-29: LiabilitiesAndStockholdersEquity - StockholdersEquity.",
                 [cite(rt), cite(re_)])


@reg("T2_621")
def _(): return sec_one("WMT", ["CostOfRevenue", "CostOfGoodsAndServicesSold"], "2018-01-31", N(A(0), 1), M, period="year")


@reg("T2_622")
def _(): return sec_one("WMT", ["InventoryNet"], "2018-01-31", N(A(0), 1), M)


@reg("T2_623")
def _(): return sec_one("WMT", "PrepaidExpenseAndOtherAssetsCurrent", "2018-01-31", N(A(0), 1), M)


@reg("T2_624")
def _(): return sec_one("UBER", "CostsAndExpenses", "2019-12-31", N(A(0), 1), M, period="year")


@reg("T2_625")
def _(): return sec_one("UBER", ["MarketingExpense", "SellingAndMarketingExpense"], "2019-12-31", N(A(0), 1), M, period="year")


@reg("T2_626")
def _(): return sec_one("UBER", "ResearchAndDevelopmentExpense", "2019-12-31", N(A(0), 1), M, period="year")


# =====================================================================
# T3 price questions
# =====================================================================

@reg("T3_001")
def _():
    c = bar("AAPL", "2019-02-20").raw_close
    o = bar("AAPL", "2019-02-21").raw_open
    return R([dict(N(-A(0), 0.01), sign_free=True)], [o - c], "AAPL raw open 2019-02-21 minus raw close 2019-02-20 (event day).",
             [pxsrc("AAPL", "2019-02-20", "2019-02-21"),
              "Samsung Galaxy Unpacked 2019 (Galaxy Fold unveiled), Bill Graham Civic Auditorium, San Francisco, "
              "2019-02-20: https://www.macrumors.com/2019/02/20/samsung-galaxy-fold-smartphone/ ; "
              "https://abc7news.com/amp/samsung-product-launch-foldable-phone-patent/5147233"],
             append=["The launch event (Samsung Galaxy Unpacked, where the Galaxy Fold was unveiled) took place in San "
                     "Francisco on February 20, 2019."],
             note="Reference states the sign in words ('... lower'); the negative key is marked sign_free so "
                  "'<x> lower' can pass.")


@reg("T3_006")
def _():
    df = bars("MSFT", "2011-01-01", "2011-12-31")
    lo = df.raw_close.min()
    return R([N(A(0), 10)], [10e6 / lo],
             f"10,000,000 / lowest MSFT raw close of 2011 ({lo:.4f} on {df.raw_close.idxmin()}).",
             [pxsrc("MSFT", "2011-01-01..2011-12-31"),
              "Kaiming He received his PhD from the Chinese University of Hong Kong in 2011: "
              "https://people.csail.mit.edu/kaiming/ ; https://en.wikipedia.org/wiki/Kaiming_He"],
             append=["Dr. Kaiming He received his PhD from the Chinese University of Hong Kong in 2011."])


@reg("T3_011")
def _():
    df = bars("TSLA", "2023-12-29", "2024-07-24")
    ch = df.raw_close.pct_change() * 100
    hits = [d for d, v in ch.items() if v < -10 and "2024-01-01" <= d < "2024-07-24"]
    d = hits[-1]
    days = (dt.date(2024, 7, 24) - dt.date.fromisoformat(d)).days
    return R([RD(0), N(A(2), 0)], [d, days],
             f"TSLA raw close-to-close drops < -10% in 2024-01-01..2024-07-23: {hits}; latest one and calendar days to "
             "2024-07-24.", [pxsrc("TSLA", "2023-12-29..2024-07-24")])


@reg("T3_023")
def _():
    df = bars("^GSPC", "2009-12-01", "2025-04-30")
    me = df.raw_close.groupby(df.index.str[:7]).last()
    r = me.pct_change() * 100
    r = r[r.index >= "2010-01"]
    m = r.idxmax()
    return R([month_key(RM(0)), N(A(1), 0.1)],
             [m, r.max()], "Month-end ^GSPC closes, monthly % change 2010-01..2025-04, the largest.",
             [pxsrc("^GSPC", "2009-12-31..2025-04-30")])


@reg("T3_037")
def _():
    out = []
    for s in ("BA", "LMT"):
        out.append(pct(bar(s, "2024-06-06").raw_close, bar(s, "2024-06-07").raw_close))
    return R([N(A(0), 0.01), N(A(1), 0.01)], out, "BA and LMT raw close 2024-06-07 vs 2024-06-06, % change.",
             [pxsrc("BA", "2024-06-06", "2024-06-07"), pxsrc("LMT", "2024-06-06", "2024-06-07")])


@reg("T3_046")
def _():
    out = [pct(bar(s, "2023-02-01").raw_close, bar(s, "2023-02-02").raw_close) for s in ("BAC", "WFC")]
    larger = "BAC" if abs(out[0]) > abs(out[1]) else "WFC"
    return R([N(A(0), 0.01), N(A(1), 0.01), TKN(RE(r"([A-Z][\w ]*?) experienced a larger"))], out + [larger],
             "BAC and WFC raw close 2023-02-02 vs 2023-02-01, % change; larger magnitude.",
             [pxsrc("BAC", "2023-02-01", "2023-02-02"), pxsrc("WFC", "2023-02-01", "2023-02-02")],
             note="The 'larger' ticker key is weak: the answer names both banks.")


@reg("T3_047")
def _():
    d, ratio = q("select date, ratio_text from actions where symbol='AAPL' and kind='split' and split_type='split' "
                 "and date<='2025-10-30' order by date desc limit 1")[0]
    pd_, pb = prev_bar("AAPL", d)
    ch = bar("AAPL", d).raw_open - pb.raw_close
    return R([N(-A(0), 0.01), ratio_key(RE(r"split ratio was (\d+:\d+)"))],
             [ch, ratio], f"Latest AAPL split ({d}, {ratio}): raw open on {d} minus raw close on {pd_}.",
             [pxsrc("AAPL", pd_, d), f"Yahoo AAPL actions split {d} {ratio}"])


@reg("T3_050")
def _():
    a = bars("^IXIC", "2020-10-01", "2025-05-09").raw_close
    b = bars("^GSPC", "2020-10-01", "2025-05-09").raw_close
    s = (a - b).dropna()
    v1 = s.rolling(20).std(ddof=1).mean()
    v0 = s.rolling(20).std(ddof=0).mean()
    return R([N(A(0), 0.01)], [v1], f"Mean of the 20-day rolling std of (IXIC - GSPC) close spread within the window: "
             f"ddof=1 {v1:.2f}, ddof=0 {v0:.2f}.", [pxsrc("^IXIC", "2020-10-01..2025-05-09"),
                                                    pxsrc("^GSPC", "2020-10-01..2025-05-09")])


@reg("T3_060")
def _():
    df = bars("^VIX", "2024-03-10", "2025-05-31")
    w = df[df.index <= "2025-05-16"]
    pk = w.raw_close.idxmax()
    after = next_days("^VIX", pk, 10)
    drop = (1 - after.raw_close.iloc[-1] / w.raw_close.max()) * 100
    hi = w.raw_high.idxmax()
    return R([RD(0), N(A(3), 0.01)], [pk, drop],
             f"Peak by close {pk}; close 10 trading days later ({after.index[-1]}) vs peak close, % drop. Peak by "
             f"intraday high would be {hi}.", [pxsrc("^VIX", "2024-03-10..2025-05-16")])


@reg("T3_111")
def _():
    out = [bar(s, "2021-01-11").raw_close - bar(s, "2021-01-08").raw_close for s in ("NVDA", "AAPL", "TSLA")]
    return R([N(A(0), 0.01), N(A(1), 0.01), N(A(2), 0.01)], out,
             "NVDA, AAPL, TSLA raw close 2021-01-11 minus raw close 2021-01-08.",
             [pxsrc(s, "2021-01-08", "2021-01-11") for s in ("NVDA", "AAPL", "TSLA")],
             note="The requested ordering by % change is not graded.")


@reg("T3_207")
def _():
    b = bar("^AXJO", "2024-12-18")
    pdd, pb = prev_bar("^AXJO", "2024-12-18")
    oc = pct(b.raw_open, b.raw_close)
    cc = pct(pb.raw_close, b.raw_close)
    return R([N(A(0), 0.01), N(A(1), 0.01), N(A(2), 0.01)], [b.raw_open, b.raw_close, oc],
             f"^AXJO 2024-12-18 open, close, open-to-close % change (close-to-close vs {pdd}: {cc:.4f}).",
             [pxsrc("^AXJO", pdd, "2024-12-18")])


@reg("T3_211")
def _():
    c0 = bar("AMD", "2023-11-13").raw_close
    nd = next_days("AMD", "2023-11-13", 10)
    return R([N(A(0), 0.01)], [pct(c0, nd.raw_close.iloc[-1])],
             f"AMD raw close on 2023-11-13 (announcement) to raw close on the 10th trading day after ({nd.index[-1]}).",
             [pxsrc("AMD", "2023-11-13", nd.index[-1]),
              "NVIDIA announced the H200 (HGX H200) at SC23 on 2023-11-13: "
              "https://nvidianews.nvidia.com/news/nvidia-supercharges-hopper-the-worlds-leading-ai-computing-platform ; "
              "https://insidehpc.com/2023/11/nvidia-announces-hgx-h200-systems-and-cloud-instances"],
             append=["NVIDIA announced the H200 chip on November 13, 2023."])


@reg("T3_212")
def _():
    b = bar("^IXIC", "2025-03-07")
    return R([N(A(0), 1)], [b.raw_high - b.raw_low], "^IXIC high minus low on 2025-03-07.",
             [pxsrc("^IXIC", "2025-03-07"),
              "Powell, 'Economic Outlook', Chicago Booth 2025 U.S. Monetary Policy Forum, 2025-03-07: "
              "https://fraser.stlouisfed.org/title/statements-speeches-jerome-h-powell-1164/economic-outlook-717393"],
             append=["The speech was delivered on March 7, 2025."])


@reg("T3_216")
def _():
    ch = {s: pct(bar(s, "2024-12-30").raw_close, bar(s, "2024-12-31").raw_close) for s in ("^NDX", "^GSPC", "^DJI")}
    return R([N(A(0), 0.01)], [max(ch.values()) - min(ch.values())],
             f"Close-to-close % change on 2024-12-31 of ^NDX, ^GSPC, ^DJI ({', '.join(f'{k} {v:.3f}' for k, v in ch.items())}); "
             "max minus min.", [pxsrc(s, "2024-12-30", "2024-12-31") for s in ch])


@reg("T3_225")
def _():
    df = bars("BTC-USD", "2022-12-25", "2025-05-20")
    c = df.raw_close
    idx = list(c.index)
    res = {"prev": [], "first": []}
    for i in range(1, len(idx) - 4):
        win = [c.iloc[i + k] > c.iloc[i + k - 1] for k in range(5)]
        if not all(win) or idx[i] < "2023-01-01":
            continue
        if c.iloc[i + 4] / c.iloc[i - 1] - 1 > 0.20:
            res["prev"].append((idx[i], idx[i + 4]))
        if c.iloc[i + 4] / c.iloc[i] - 1 > 0.20:
            res["first"].append((idx[i], idx[i + 4]))
    w = res["prev"]
    keys = [RD(i) for i in range(4)]
    snap = [d for pair in w for d in pair] if len(w) == 2 else ["n/a"] * 4
    return R(keys, snap, f"BTC-USD (UTC days) windows of 5 consecutive higher closes with total rise > 20%: from the close "
             f"before day 1: {res['prev']}; from day 1's close: {res['first']}.", [pxsrc("BTC-USD", "2022-12-31..2025-05-20")],
             note="Both baselines are checked; the ambiguity is immaterial only if they give the same windows. "
                  "Count of instances (2) is implied by the four dates.")


@reg("T3_227")
def _():
    c = bars("^GSPC", "2022-03-16", "2022-06-16").raw_close
    best = (0, None, None)
    for i, d in enumerate(c.index):
        for e in c.index[i:]:
            dd = (1 - c[e] / c[d]) * 100
            if dd > best[0]:
                best = (dd, d, e)
    return R([RD(0), RD(1), N(A(4), 0.01)], [best[1], best[2], best[0]],
             "Close-based maximum drawdown of ^GSPC within 2022-03-16..2022-06-16.", [pxsrc("^GSPC", "2022-03-16..2022-06-16")])


@reg("T3_228")
def _():
    m7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    r = {s: pct(on_or_before(s, "2022-12-31")[1].raw_close, on_or_before(s, "2023-12-31")[1].raw_close) for s in m7}
    return R([N(A(0), 0.01)], [max(r.values()) - min(r.values())],
             f"2023 raw-close returns (2022-12-30 to 2023-12-29): {', '.join(f'{k} {v:.2f}' for k, v in r.items())}; "
             "max minus min.", [pxsrc(s, "2022-12-30", "2023-12-29") for s in m7] +
             ["Magnificent 7 = Apple, Microsoft, Alphabet, Amazon, Nvidia, Meta, Tesla: "
              "https://www.statista.com/chart/31509/magnificent-seven-stock-performance-in-2023/ ; "
              "https://www.fool.com/investing/2024/03/07/trillion-dollar-magnificent-seven-stocks-buy-march"],
             append=["The \"Magnificent 7\" are Apple (AAPL), Microsoft (MSFT), Alphabet (GOOGL), Amazon (AMZN), NVIDIA "
                     "(NVDA), Meta Platforms (META) and Tesla (TSLA)."],
             note="Alphabet is taken as GOOGL; GOOG would give a slightly different return but neither is the max or min.")


@reg("T3_229")
def _():
    nd = next_days("TSLA", "2024-11-05", 10)
    return R([N(A(0), 0.01)], [nd.raw_high.max()],
             f"Max TSLA raw intraday high over the 10 trading days after 2024-11-05 ({nd.index[0]}..{nd.index[-1]}).",
             [pxsrc("TSLA", nd.index[0] + ".." + nd.index[-1]),
              "2024 US presidential election held 2024-11-05; Trump declared winner early 2024-11-06: "
              "https://www.aljazeera.com/program/newsfeed/2024/11/6/donald-trump-declared-us-election-winner"],
             append=["The 2024 US presidential election day was November 5, 2024."])


@reg("T3_231")
def _():
    df = bars("NVDA", "2024-12-31", "2025-03-31")
    w = df[df.index >= "2025-01-01"]
    d = w.raw_volume.idxmax()
    i = list(df.index).index(d)
    ch = pct(df.raw_close.iloc[i - 1], df.raw_close.iloc[i])
    return R([N(A(0), 0.01)], [ch], f"NVDA max raw volume day in Q1 2025 ({d}), close-to-close % change.",
             [pxsrc("NVDA", "2024-12-31..2025-03-31")])


@reg("T3_233")
def _():
    df = bars("TSLA", "2025-04-01", "2025-04-30")
    sh = 200000 + 10000 * ((df.raw_volume < 80e6).sum() - (df.raw_volume > 120e6).sum())
    return R([N(A(0), 0)], [sh], "200,000 + 10,000 x (#days raw volume < 80M - #days > 120M), TSLA April 2025.",
             [pxsrc("TSLA", "2025-04-01..2025-04-30")])


@reg("T3_239")
def _():
    df = bars("^N225", "2024-04-01", "2025-04-30")
    me = df.raw_close.groupby(df.index.str[:7]).last()
    r = me.pct_change()
    pos = [m for m, v in r.items() if m >= "2024-05" and v > 0]
    want = RMS()
    names = {m: month_label(m) for m in want}
    keys = [{"kind": "set", "value": [T(names[m], m, names[m][:3] + " " + m[:4], m.replace("-", "/")) for m in want]}]
    note = ""
    return R(keys, [pos == want], f"^N225 month-end closes, positive monthly returns 2024-05..2025-04: {pos}.",
             [pxsrc("^N225", "2024-04-30..2025-04-30")],
             note="Set key; the grader only checks that each month is named, not that no extra month is.")


@reg("T3_240")
def _():
    df = bars("^KS11", "2020-01-01", "2024-12-31")
    yh = df.raw_high.groupby(df.index.str[:4]).max()
    y = yh.idxmax()
    return R([T(RE(r"^(\d{4}):")), N(A(1), 0.01)], [y, yh.max()], "^KS11 annual max of daily highs 2020-2024; the largest.",
             [pxsrc("^KS11", "2020-01-01..2024-12-31")])


def month_ret(s, m_prev_end, m_end):
    return pct(on_or_before(s, m_prev_end)[1].raw_close, on_or_before(s, m_end)[1].raw_close)


@reg("T3_b012")
def _():
    r = {s: month_ret(s, "2025-03-31", "2025-04-30") for s in ("MRK", "TSM", "BA", "IBM", "GS")}
    pos = sorted(s for s, v in r.items() if v > 0)
    keys = [RTK(0), N(A(0), 0.01), RTK(1), N(A(1), 0.01), RTK(2), N(A(2), 0.01)]
    t = [keys[0]["value"], keys[2]["value"], keys[4]["value"]]
    return R(keys, [x for s in t for x in (s if s in pos else "-", r[s])],
             f"April 2025 raw-close returns (2025-03-31 to 2025-04-30): {', '.join(f'{k} {v:.2f}' for k, v in r.items())}.",
             [pxsrc(s, "2025-03-31", "2025-04-30") for s in r],
             note=f"Positive set from snapshot: {pos}.")


@reg("T3_b014")
def _():
    df = bars("BA", "2025-04-01", "2025-04-30")
    return R([N(A(0), 0.01), N(A(1), 0.01)], [df.raw_high.max(), pct(df.raw_low.min(), df.raw_high.max())],
             "BA April 2025: max raw high; % above min raw low.", [pxsrc("BA", "2025-04-01..2025-04-30")])


@reg("T3_b017")
def _():
    t = bar("TSLA", "2025-05-23").raw_close
    a = bar("AMZN", "2025-05-23").raw_close
    return R([N(A(0), 0.01), N(A(1), 0.01)], [t, t - a], "TSLA and AMZN raw close on Friday 2025-05-23; difference.",
             [pxsrc("TSLA", "2025-05-23"), pxsrc("AMZN", "2025-05-23")])


@reg("T3_b020")
def _():
    a = bar("AAL", "2025-05-16").raw_close
    b = bar("AAL", "2025-05-23").raw_close
    r = R([N(A(0), 0.01)], [abs(b - a)], f"AAL raw close 2025-05-23 ({b:.2f}) vs 2025-05-16 ({a:.2f}), absolute difference.",
          [pxsrc("AAL", "2025-05-16", "2025-05-23")])
    r["keys"][0]["sign_free"] = True
    r["note"] = f"Signed difference (later minus earlier) is {b - a:.4f}; the reference gives a magnitude."
    return r


@reg("T3_b030")
def _():
    n = pct(bar("^IXIC", "2023-12-29").raw_close, bar("^IXIC", "2024-12-31").raw_close)
    s = pct(bar("^GSPC", "2023-12-29").raw_close, bar("^GSPC", "2024-12-31").raw_close)
    return R([index_key(RE(r"^(Nasdaq|S&P 500)")), N(A(0), 0.01)], ["Nasdaq" if n > s else "S&P 500", n - s],
             f"2024 returns from 2023-12-29 close to 2024-12-31 close: IXIC {n:.4f}, GSPC {s:.4f}; difference.",
             [pxsrc("^IXIC", "2023-12-29", "2024-12-31"), pxsrc("^GSPC", "2023-12-29", "2024-12-31")],
             note="The text key is weak: the question itself names both indices.")


@reg("T3_220")
def _():
    d = "2025-01-24"
    vol = bar("PFE", d).raw_volume
    rows = q("select val, end, filed, accn from facts where cik=78003 and tag='EntityCommonStockSharesOutstanding' "
             "and filed<=? order by filed desc limit 1", d)
    sh, end, filed, accn = rows[0]
    return R([N(A(0), 0.01)], [vol / sh * 100],
             f"PFE raw volume on {d} / latest cover-page shares outstanding filed before it ({sh:.0f} as of {end}, filed {filed}).",
             [pxsrc("PFE", d), f"SEC dei:EntityCommonStockSharesOutstanding cik=78003 end={end} filed={filed} accn={accn}",
              "DOJ announced Pfizer's ~$59.7M False Claims Act settlement (Biohaven, Nurtec ODT) on 2025-01-24: "
              "https://www.justice.gov/opa/pr/pfizer-agrees-pay-nearly-60m-resolve-false-claims-allegations-relating-improper-physician ; "
              "https://oig.hhs.gov/fraud/enforcement/pfizer-agrees-to-pay-nearly-60m-to-resolve-false-claims-allegations-relating-to-improper-physician-payments-by-subsidiary"],
             append=["Pfizer agreed to the settlement on January 24, 2025."])


# =====================================================================
# T3 SEC questions
# =====================================================================

def fy_end(ticker, fy):
    """The period end of a company's fiscal year `fy` (the 10-K whose own fy and fp=FY match; latest end in it)."""
    rows = q("select end, count(*) from facts where cik=? and form in ('10-K','10-K/A') and fy=? and fp='FY' "
             "and tag in ('Assets','Revenues','NetIncomeLoss','StockholdersEquity') and start is null group by end "
             "order by end desc", cik_of(ticker), fy)
    if not rows:
        raise KeyError(f"{ticker} FY{fy}: no 10-K")
    return rows[0][0]


def cik_of(ticker):
    from snap import ciks
    return ciks(ticker)[0]


@reg("T3_205")
def _():
    def quarters(tk, rev_tags, years):
        out = {}
        for fy in years:
            # quarterly duration facts in this fiscal year's 10-Qs plus Q4 = FY - 9M
            end_fy = fy_end(tk, fy)
            start_fy = (dt.date.fromisoformat(end_fy) - dt.timedelta(days=371)).isoformat()
            ends = [r[0] for r in q("select distinct end from facts where cik=? and tag='GrossProfit' and end>? and end<? "
                                    "and julianday(end)-julianday(start) between 80 and 100 order by end",
                                    cik_of(tk), start_fy, end_fy)]
            for i, e in enumerate(ends[:3]):
                gp, _ = value(tk, "GrossProfit", e, period="quarter")
                rv, _ = first_of(tk, rev_tags, e, period="quarter")
                out[f"{fy}Q{i + 1}"] = (gp, rv, e)
            gpy, _ = value(tk, "GrossProfit", end_fy, period="year")
            rvy, _ = first_of(tk, rev_tags, end_fy, period="year")
            gp3 = sum(out[f"{fy}Q{i}"][0] for i in (1, 2, 3))
            rv3 = sum(out[f"{fy}Q{i}"][1] for i in (1, 2, 3))
            out[f"{fy}Q4"] = (gpy - gp3, rvy - rv3, end_fy)
        return {k: (v[0] / v[1] * 100, v[2]) for k, v in out.items()}
    revs = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]
    amd = quarters("AMD", revs, [2022, 2023, 2024])
    intc = quarters("INTC", revs, [2022, 2023, 2024])
    qmin = min(amd, key=lambda k: amd[k][0])
    imin = min(intc, key=lambda k: intc[k][0])
    return R([quarter_key(RE(r"(\d{4}Q[1-4])")),
              N(A(4), 0.01), N(A(5), 0.01)],
             [qmin, amd[qmin][0], intc[qmin][0]],
             f"Quarterly GAAP gross margin = GrossProfit / revenue (Q4 = FY - 9M). AMD: "
             f"{', '.join(f'{k} {v[0]:.2f}' for k, v in amd.items())}. INTC: {', '.join(f'{k} {v[0]:.2f}' for k, v in intc.items())}. "
             f"Intel's minimum is {imin}.",
             [f"SEC us-gaap:GrossProfit and revenue quarterly facts, AMD and INTC FY2022-FY2024"],
             note="The yes/no part (Intel's 2022Q3 margin is not its minimum) is not graded.")


@reg("T3_213")
def _():
    a, ra = value("WMT", "Goodwill", "2022-01-31")
    b, rb = value("WMT", "Goodwill", "2024-01-31")
    return R([N(A(0), 1)], [(b - a) / M], "WMT Goodwill 2024-01-31 minus 2022-01-31, in millions.", [cite(ra), cite(rb)])


@reg("T3_214")
def _():
    a, ra = value("AMZN", "NetCashProvidedByUsedInFinancingActivities", "2020-12-31", period="year")
    b, rb = value("AMZN", "NetCashProvidedByUsedInFinancingActivities", "2024-12-31", period="year")
    return R([dict(N(-A(0), 1), sign_free=True)], [(b - a) / M], "AMZN financing cash flow FY2024 minus FY2020, in millions.",
             [cite(ra), cite(rb)],
             note="Reference states the sign in words ('Decreased by ...'); the negative key is marked sign_free.")


@reg("T3_221")
def _():
    vals, src = {}, []
    for y in range(2019, 2024):
        v, r = value("ORCL", "Assets", f"{y}-05-31")
        vals[y] = v
        src.append(cite(r))
    g = [pct(vals[y - 1], vals[y]) for y in (2020, 2021, 2022, 2023)]
    return R([N(A(7), 0.01), N(A(5), 0.01), N(A(3), 0.01), N(A(1), 0.01)], g,
             "ORCL Assets at May 31 of each fiscal year (as first reported), YoY % for FY2020-FY2023.", src)


def ratio_lia(tk, end):
    a, ra = value(tk, "Assets", end)
    try:
        l, rl = value(tk, "Liabilities", end)
    except KeyError:
        t, rt = value(tk, "LiabilitiesAndStockholdersEquity", end)
        e, re_ = first_of(tk, ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                               "StockholdersEquity"], end)
        return (t - e) / a * 100, [cite(ra), cite(rt), cite(re_)]
    return l / a * 100, [cite(ra), cite(rl)]


@reg("T3_224")
def _():
    out, src = [], []
    for y in (2022, 2023, 2024):
        x, s1 = ratio_lia("LLY", f"{y}-12-31")
        g, s2 = ratio_lia("GOOGL", f"{y}-12-31")
        out.append(x - g)
        src += s1 + s2
    return R([N(A(1), 0.01), N(A(3), 0.01), N(A(5), 0.01)], out,
             "Liabilities / Assets (%) of LLY minus GOOGL at each Dec 31, 2022-2024.", src)


def fy_flow(tk, tags, fy, period="year"):
    end = fy_end(tk, fy)
    v, r = first_of(tk, tags, end, period=period)
    return v, r, end


REV = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenuesNetOfInterestExpense"]
NI = ["NetIncomeLoss"]


@reg("T3_b004")
def _():
    npm, src = {}, []
    for tk in ("TSLA", "NVDA", "WFC", "META", "AAPL"):
        ni, rn, end = fy_flow(tk, NI, 2021)
        rv, rr, _ = fy_flow(tk, REV, 2021)
        npm[tk] = ni / rv * 100
        src += [cite(rn), cite(rr)]
    hi = max(npm, key=npm.get)
    lo = min(npm, key=npm.get)
    return R([RTK(0), N(A(0), 0.01), RTK(1), N(A(1), 0.01)],
             [hi, npm[hi], lo, npm[lo]],
             f"FY2021 (each company's own fiscal year) NetIncomeLoss / revenue: "
             f"{', '.join(f'{k} {v:.2f}' for k, v in npm.items())}.", src)


@reg("T3_b005")
def _():
    out, src = [], []
    for fy in (2021, 2022, 2023, 2024):
        m = {}
        for tk in ("AAPL", "NVDA"):
            gp, rg, _ = fy_flow(tk, ["GrossProfit"], fy)
            rv, rr, _ = fy_flow(tk, REV, fy)
            m[tk] = gp / rv * 100
            src += [cite(rg), cite(rr)]
        out.append(m["NVDA"] - m["AAPL"])
    return R([N(A(1), 0.01), N(A(3), 0.01), N(A(5), 0.01), N(A(7), 0.01)], out,
             "NVDA gross margin minus AAPL gross margin (percentage points), each company's own FY2021-FY2024.", src)


@reg("T3_b006")
def _():
    r, src = {}, []
    for tk in ("TSLA", "NVDA", "WFC", "META", "AAPL"):
        end = fy_end(tk, 2024)
        r[tk], s = ratio_lia(tk, end)
        src += s
    top = sorted(r, key=r.get, reverse=True)[:3]
    keys = [RTK(0), N(A(0), 0.01), RTK(1), N(A(1), 0.01), RTK(2), N(A(2), 0.01)]
    t = [keys[0]["value"], keys[2]["value"], keys[4]["value"]]
    return R(keys, [top[0], r[t[0]], top[1], r[t[1]], top[2], r[t[2]]],
             f"FY2024 Liabilities / Assets (%): {', '.join(f'{k} {v:.2f}' for k, v in r.items())}.", src,
             note="Ranking order is not graded; the three tickers and their ratios are.")


@reg("T3_b007")
def _():
    out, src = [], []
    for fy in (2021, 2022, 2023, 2024):
        w, rw, _ = fy_flow("WFC", NI, fy)
        n, rn, _ = fy_flow("NVDA", NI, fy)
        out.append(w / n)
        src += [cite(rw), cite(rn)]
    return R([N(A(1), 0.01), N(A(3), 0.01), N(A(5), 0.01), N(A(7), 0.01)], out,
             "WFC NetIncomeLoss / NVDA NetIncomeLoss, each company's own FY2021-FY2024.", src)


@reg("T3_b008")
def _():
    out, src, det = [], [], []
    for fy in (2021, 2022, 2023, 2024):
        t = {}
        for tk in ("AAPL", "TSLA"):
            cg, rc, end = fy_flow(tk, ["CostOfGoodsAndServicesSold", "CostOfRevenue"], fy)
            prev_end = fy_end(tk, fy - 1)
            i1, r1 = value(tk, "InventoryNet", end)
            i0, r0 = value(tk, "InventoryNet", prev_end)
            t[tk] = cg / ((i0 + i1) / 2)
            src += [cite(rc), cite(r1), cite(r0)]
        out.append(t["AAPL"] / t["TSLA"])
        det.append(f"FY{fy} AAPL {t['AAPL']:.3f} TSLA {t['TSLA']:.3f}")
    return R([N(A(1), 0.01), N(A(3), 0.01), N(A(5), 0.01), N(A(7), 0.01)], out,
             "Inventory turnover = cost of revenue / average(begin, end) InventoryNet; AAPL / TSLA. " + "; ".join(det), src)


@reg("T3_b009")
def _():
    r, src = {}, []
    for fy in (2021, 2022, 2023, 2024):
        da, rd, _ = fy_flow("TSLA", ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndImpairment"], fy)
        ni, rn, _ = fy_flow("TSLA", NI, fy)
        r[fy] = da / ni * 100
        src += [cite(rd), cite(rn)]
    y = min(r, key=r.get)
    return R([T(RE(r"in (\d{4})")), N(A(1), 0.01)], [str(y), r[y]],
             f"TSLA D&A / NetIncomeLoss (%): {', '.join(f'{k} {v:.2f}' for k, v in r.items())}; the lowest.", src)
