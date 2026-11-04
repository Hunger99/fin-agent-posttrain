"""Task templates for the offline market environment (reference implementation, numpy/pandas).

Each generator draws one candidate task for a split and difficulty from the
seeded rng, raises Skip(<rule>) when the candidate breaks a quality rule, and
returns a task dict without task_id. `params` in meta fully defines the
computation so that verify.py can recompute the answer independently.
"""
import numpy as np

from common import (CUTOFF, GLOBAL_START, SPLIT_WINDOWS, Skip, add_days, count_key, date_key,
                    fmt_date, long_date, minus_one_year, month_aliases, month_label, num_key,
                    round_ok, set_key, short_names, text_key, ticker_split)

INDEX_NAMES = {
    "^GSPC": "S&P 500", "^DJI": "Dow Jones Industrial Average", "^IXIC": "Nasdaq Composite",
    "^NDX": "Nasdaq-100", "^RUT": "Russell 2000", "^VIX": "Cboe Volatility Index (VIX)",
    "^N225": "Nikkei 225", "^FTSE": "FTSE 100", "^GDAXI": "DAX", "^FCHI": "CAC 40",
    "^STOXX50E": "Euro Stoxx 50", "^HSI": "Hang Seng", "^GSPTSE": "S&P/TSX Composite",
    "^KS11": "KOSPI Composite", "^BSESN": "BSE Sensex", "^AXJO": "S&P/ASX 200",
    "^TWII": "Taiwan Capitalization Weighted Stock Index (TAIEX)", "^JKSE": "IDX Composite",
    "DX-Y.NYB": "US Dollar Index",
}
FUT_NAMES = {"GC=F": "gold", "SI=F": "silver", "CL=F": "WTI crude oil", "BZ=F": "Brent crude oil",
             "NG=F": "natural gas", "HG=F": "copper", "ZC=F": "corn", "ZS=F": "soybean",
             "ZW=F": "wheat"}
US_INDICES = ["^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT"]
SPREAD_PAIRS = [("^GSPC", "^NDX"), ("^GSPC", "^RUT"), ("^DJI", "^IXIC"), ("^GSPC", "^DJI"),
                ("^NDX", "^RUT"), ("SPY", "QQQ"), ("XLK", "XLF"), ("XLE", "XLV"), ("IWM", "SPY"),
                ("TLT", "IEF"), ("XLK", "XLE"), ("LQD", "HYG"), ("GLD", "SLV"), ("EFA", "EEM")]
# FRED dollar series for foreign indices: (series, True if quoted as local units per USD)
FX_FOR_INDEX = {"^N225": ("DEXJPUS", True, "Japanese yen per US dollar"),
                "^FTSE": ("DEXUSUK", False, "US dollars per British pound"),
                "^GDAXI": ("DEXUSEU", False, "US dollars per euro"),
                "^FCHI": ("DEXUSEU", False, "US dollars per euro"),
                "^STOXX50E": ("DEXUSEU", False, "US dollars per euro"),
                "^HSI": ("DEXHKUS", True, "Hong Kong dollars per US dollar"),
                "^GSPTSE": ("DEXCAUS", True, "Canadian dollars per US dollar"),
                "^KS11": ("DEXKOUS", True, "South Korean won per US dollar"),
                "^BSESN": ("DEXINUS", True, "Indian rupees per US dollar"),
                "^AXJO": ("DEXUSAL", False, "US dollars per Australian dollar")}
LOCAL_CCY = {"^N225": "yen", "^FTSE": "pounds", "^GDAXI": "euros", "^FCHI": "euros",
             "^STOXX50E": "euros", "^HSI": "Hong Kong dollars", "^GSPTSE": "Canadian dollars",
             "^KS11": "won", "^BSESN": "rupees", "^AXJO": "Australian dollars"}

TAG_DESC = {
    "Revenues": "total revenues", "RevenueFromContractWithCustomerExcludingAssessedTax":
    "revenue from contracts with customers, excluding assessed tax", "SalesRevenueNet": "net sales",
    "NetIncomeLoss": "net income attributable to the company", "OperatingIncomeLoss":
    "operating income", "GrossProfit": "gross profit", "CostOfRevenue": "cost of revenue",
    "CostOfGoodsAndServicesSold": "cost of goods and services sold", "CostOfGoodsSold":
    "cost of goods sold", "ResearchAndDevelopmentExpense": "research and development expense",
    "NetCashProvidedByUsedInOperatingActivities": "net cash provided by (used in) operating activities",
    "PaymentsToAcquirePropertyPlantAndEquipment": "payments to acquire property, plant and equipment",
    "IncomeTaxExpenseBenefit": "income tax expense (benefit)", "EarningsPerShareDiluted":
    "diluted earnings per share", "Assets": "total assets", "Liabilities": "total liabilities",
    "StockholdersEquity": "total stockholders' equity", "CashAndCashEquivalentsAtCarryingValue":
    "cash and cash equivalents", "InventoryNet": "net inventory", "LongTermDebtNoncurrent":
    "long-term debt, noncurrent", "Goodwill": "goodwill", "AssetsCurrent": "total current assets",
    "LiabilitiesCurrent": "total current liabilities"}
DUR_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "NetIncomeLoss",
            "OperatingIncomeLoss", "GrossProfit", "CostOfRevenue", "ResearchAndDevelopmentExpense",
            "NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment",
            "IncomeTaxExpenseBenefit", "EarningsPerShareDiluted"]
INST_TAGS = ["Assets", "Liabilities", "StockholdersEquity", "CashAndCashEquivalentsAtCarryingValue",
             "InventoryNet", "LongTermDebtNoncurrent", "Goodwill", "AssetsCurrent", "LiabilitiesCurrent"]
REV_PREC = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
COGS_PREC = ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"]
SIGNED_TAGS = {"NetIncomeLoss", "OperatingIncomeLoss", "NetCashProvidedByUsedInOperatingActivities",
               "IncomeTaxExpenseBenefit", "EarningsPerShareDiluted", "GrossProfit"}

PCT = "as a percentage rounded to two decimals (e.g. 12.34 for 12.34%)"
PCT_SIGNED = PCT[:-1] + "; negative for a decline)"
NT_RULE = ("If a date given here is not a trading day for that instrument, use the last trading "
           "day on or before it.")
SPLIT_ADJ = "split-adjusted (adjusted for all splits up to 2025-10-30, not for dividends)"
UNADJ = "unadjusted (the price actually quoted that day, not adjusted for later splits)"
TOTAL_DESC = ("total return including dividends (use the close adjusted for both splits and "
              "dividends)")
PRICE_DESC = "price return excluding dividends (use split-adjusted closes)"


def ordinal(n):
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return "%d%s" % (n, suf)


class Ctx:
    """Entity pools per split and naming helpers."""

    def __init__(self, S, X):
        self.S, self.X = S, X
        sec = S.sec
        eqs = sorted(s for s in S.px if sec.loc[s, "kind"] == "equity"
                     and sec.loc[s, "currency"] == "USD" and len(S.px[s]) >= 260)
        names = {s: str(sec.loc[s, "name"]).strip() for s in S.px}
        dup = {}
        for s in eqs:
            dup.setdefault(names[s], []).append(s)
        self.name = names
        self.name_unique = {s: len(dup[names[s]]) == 1 for s in eqs}
        self.eq = {sp: [s for s in eqs if ticker_split(s) == sp] for sp in SPLIT_WINDOWS}
        self.etf = sorted(s for s in S.px if sec.loc[s, "kind"] == "etf")
        self.index = sorted(s for s in S.px if sec.loc[s, "kind"] == "index")
        self.fut = sorted(s for s in S.px if sec.loc[s, "kind"] == "future")
        self.fx = sorted(s for s in S.px if sec.loc[s, "kind"] == "fx" and s != "TWD=X")
        self.crypto = sorted(s for s in S.px if sec.loc[s, "kind"] == "crypto")
        cik = sec["cik"]
        multi = set()
        for c, row in S.companies.iterrows():
            if row["tickers"] and "," in str(row["tickers"]):
                multi.add(int(c))
        self.cik = {}
        for s in eqs:
            c = cik.get(s)
            if c is not None and np.isfinite(c) and int(c) not in multi and \
                    any(True for _ in X.years(int(c))[:1]):
                self.cik[s] = int(c)
        self.sec_eq = {sp: [s for s in self.eq[sp] if s in self.cik] for sp in SPLIT_WINDOWS}
        self.sector = {s: sec.loc[s, "sector"] for s in eqs}
        self.true_splits = {s: [x for x in S.splits.get(s, []) if x[3] in ("split", "reverse_split")]
                            for s in S.px}

    # -- naming ------------------------------------------------------------
    def kind(self, sym):
        return self.S.px[sym].kind

    def ent(self, sym, d):
        k = self.kind(sym)
        nm = self.name[sym]
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

    def ticker_key(self, sym):
        nm = self.name[sym]
        al = [nm] + short_names(nm)
        if sym in INDEX_NAMES:
            al += [INDEX_NAMES[sym], sym.lstrip("^")]
        if sym.startswith("^"):
            al.append(sym.lstrip("^"))
        return {"kind": "ticker", "value": sym, "aliases": sorted(set(a for a in al if a != sym))}

    def unit_of(self, sym):
        k = self.kind(sym)
        if k == "index":
            return "index points", 0.01
        if k == "fx":
            return "units of the quote currency", 0.0001
        if sym in ("ZC=F", "ZS=F", "ZW=F"):
            return "US cents per bushel", 0.01
        if sym in ("NG=F", "HG=F"):
            return "US dollars", 0.001
        return "US dollars", 0.01

    # -- windows -----------------------------------------------------------
    def window(self, sym, split, needs_time_window=False):
        """Allowed data window: whole history for equities, split window otherwise."""
        if self.kind(sym) == "equity" and not needs_time_window:
            return GLOBAL_START, CUTOFF
        return SPLIT_WINDOWS[split]

    def pick_sym(self, rng, split, pool):
        if pool == "eq":
            return rng.choice(self.eq[split])
        return rng.choice(pool)

    def name_ok(self, sym, d):
        if d == 3 and self.kind(sym) == "equity" and not self.name_unique.get(sym, False):
            raise Skip("ambiguous_name")


def pick_span(rng, px, lo, hi, L, lookback=0):
    a = px.i_ge(lo) + lookback
    b = px.i_le(hi)
    if b - a + 1 < L:
        raise Skip("short_history")
    i0 = rng.randint(a, b - L + 1)
    return i0, i0 + L - 1


def cal_date_near(rng, px, i, p_cal):
    """A prompt date that resolves to trading day i: i itself or a later non-trading day."""
    if rng.random() >= p_cal or i + 1 >= len(px):
        return str(px.d[i]), False
    gap = (np_date(px.d[i + 1]) - np_date(px.d[i])).days
    if gap < 2:
        return str(px.d[i]), False
    return add_days(str(px.d[i]), rng.randint(1, gap - 1)), True


def np_date(s):
    return np.datetime64(str(s), "D").astype(object)


def need_rule(nontrading, rule_stated):
    if nontrading and not rule_stated:
        raise Skip("non_trading_date_without_rule")


def plain(o):
    """Recursively convert numpy scalars so tasks serialize as plain JSON."""
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def mk(template, split, d, prompt, keys, answer_text, entities, start, end, ops, rows, solution, params):
    prompt = prompt.replace("..", ".")  # a name ending in "Inc." before a full stop
    return plain({"env": "market", "split": split, "template": template, "prompt": prompt, "keys": keys,
            "answer_text": answer_text,
            "meta": {"entities": entities, "start": start, "end": end, "operations": ops,
                     "rows_needed": int(rows), "difficulty": d, "params": params},
            "solution": solution})


def pick_entity(C, rng, split, eq_share=0.6, kinds=("index", "etf"), needs_time_window=False):
    if rng.random() < eq_share:
        sym = rng.choice(C.eq[split])
    else:
        pool = []
        for k in kinds:
            pool += {"index": [s for s in C.index if s not in ("^VIX", "^TWII", "^JKSE")],
                     "etf": C.etf, "future": C.fut, "fx": C.fx, "crypto": C.crypto,
                     "us": US_INDICES + C.etf}[k]
        sym = rng.choice(pool)
    lo, hi = C.window(sym, split, needs_time_window)
    return sym, lo, hi


def daily_ret(px, i0, i1, total=False):
    c = px.adj_close if total else px.close
    return c[i0:i1 + 1] / c[i0 - 1:i1] - 1


# ============================================================================
# Lookups

def gen_price_lookup(C, split, rng, d):
    sym, lo, hi = pick_entity(C, rng, split, 0.6, ("index", "etf", "future", "fx", "crypto"))
    C.name_ok(sym, d)
    px = C.S.px[sym]
    a, b = px.i_ge(lo), px.i_le(hi)
    k = px.kind
    adjustable = k in ("equity", "etf")
    sp = C.S.splits.get(sym, [])
    if adjustable and d >= 2 and sp and rng.random() < 0.75:
        b = min(b, px.i_le(sp[-1][0]) - 1)  # a date with a later split, so raw != adjusted
    if b < a:
        raise Skip("short_history")
    i = rng.randint(a, b)
    qdate, nontrading = cal_date_near(rng, px, i, {1: 0.15, 2: 0.4, 3: 0.7}[d])
    rule = d == 3 or (d == 2 and rng.random() < 0.6)
    need_rule(nontrading, rule)
    if px.bad[i]:
        raise Skip("bad_row")
    adj = rng.choice(["raw", "split"]) if adjustable else "level"
    if d == 3:
        fields = rng.choice([["high", "low"], ["open", "close"], ["low", "close"]])
    else:
        fields = [rng.choice(["open", "high", "low", "close", "close"] + (["volume"] if adjustable else []))]
    unit_txt, unit = C.unit_of(sym)
    keys, vals, parts = [], [], []
    vol_scale = None
    for f in fields:
        col = ("raw_" + f) if adj == "raw" else f
        v = float(getattr(px, col)[i])
        if f == "volume":
            if v <= 0:
                raise Skip("bad_row")
            vol_scale = rng.choice(["shares", "millions"]) if d >= 2 else "shares"
            if vol_scale == "millions":
                v, u = v / 1e6, 0.01
            else:
                u = 1
        else:
            u = unit
        if not round_ok(v, u):
            raise Skip("rounding_boundary")
        keys.append(num_key(v, u))
        vals.append(v)
    adj_txt = {"raw": UNADJ, "split": SPLIT_ADJ, "level": ""}[adj]
    ent = C.ent(sym, d)
    dt = fmt_date(qdate, rng)
    if fields == ["volume"]:
        scale_txt = ("in shares (a whole number)" if vol_scale == "shares"
                     else "in millions of shares, rounded to two decimals")
        what = "%s trading volume" % ("unadjusted" if adj == "raw" else "split-adjusted")
        prompt = rng.choice([
            "What was the %s of %s on %s? Report it %s." % (what, ent, dt, scale_txt),
            "Look up %s's %s for the session of %s and give it %s." % (ent, what, dt, scale_txt),
        ])
        if adj == "split":
            prompt += " Split-adjusted means adjusted for all splits up to 2025-10-30."
    else:
        dec = {0.01: "two", 0.001: "three", 0.0001: "four"}[unit]
        fl = " and ".join(fields)
        many = len(fields) > 1
        adj_part = (" Use %s prices." % adj_txt) if adj_txt else ""
        prompt = rng.choice([
            "What %s the daily %s %s of %s on %s?%s Give %s in %s, rounded to %s decimals." % (
                "were" if many else "was", fl, "prices" if many else "price", ent, dt, adj_part,
                "each" if many else "it", unit_txt, dec),
            "Find the %s for %s on %s.%s Report %s in %s rounded to %s decimal places." % (
                fl, ent, dt, adj_part, "both" if many else "the answer", unit_txt, dec),
            "On %s, what %s did the daily bar of %s record?%s Answer in %s, %s decimals." % (
                dt, fl, ent, adj_part, unit_txt, dec),
        ])
    if rule:
        prompt += " " + NT_RULE
    td = str(px.d[i])
    ans = "; ".join("%s %s" % (f, k["value"]) for f, k in zip(fields, keys))
    return mk("price_lookup", split, d, prompt, keys,
              "On %s (trading day used: %s) the %s%s for %s was: %s." % (
                  qdate, td, adj + " " if adj != "level" else "", "/".join(fields), sym, ans),
              [sym], td, td, ["calendar_resolve", "price_lookup"] + (["split_adjust"] if adj != "level" else []),
              1, "prices.%s for %s on %s" % ("/".join(("raw_" + f) if adj == "raw" else f for f in fields), sym, td),
              {"symbol": sym, "date": qdate, "fields": fields, "adj": adj, "vol_scale": vol_scale})


FRED_DAILY = {"DGS10": "10-year Treasury constant-maturity yield", "DGS2": "2-year Treasury constant-maturity yield",
              "DGS5": "5-year Treasury constant-maturity yield", "DGS30": "30-year Treasury constant-maturity yield",
              "DGS1": "1-year Treasury constant-maturity yield", "DGS3MO": "3-month Treasury constant-maturity yield",
              "DFF": "effective federal funds rate (daily)", "DTB3": "3-month Treasury bill secondary-market rate (discount basis)",
              "T10Y2Y": "10-year minus 2-year Treasury spread", "SOFR": "Secured Overnight Financing Rate",
              "DFEDTARU": "upper limit of the federal funds target range", "IORB": "interest rate on reserve balances",
              "IOER": "interest rate on excess reserves"}
FRED_FX = {"DEXJPUS": ("Japanese yen per US dollar", "yen", True), "DEXUSEU": ("US dollars per euro", "euro", False),
           "DEXUSUK": ("US dollars per British pound", "pound", False), "DEXCAUS": ("Canadian dollars per US dollar", "Canadian dollar", True),
           "DEXSZUS": ("Swiss francs per US dollar", "Swiss franc", True), "DEXCHUS": ("Chinese yuan per US dollar", "yuan", True),
           "DEXKOUS": ("South Korean won per US dollar", "won", True), "DEXUSAL": ("US dollars per Australian dollar", "Australian dollar", False)}
FRED_MONTHLY = {"CPIAUCSL": ("CPI for all urban consumers, all items (seasonally adjusted index)", "index points", 1, 0.001),
                "CPILFESL": ("core CPI, all items less food and energy (seasonally adjusted index)", "index points", 1, 0.001),
                "UNRATE": ("civilian unemployment rate", "percent", 1, 0.1),
                "FEDFUNDS": ("monthly average effective federal funds rate", "percent", 1, 0.01),
                "PAYEMS": ("total nonfarm payroll employment", "millions of persons", 1e-3, 0.01),
                "M2SL": ("M2 money stock", "trillions of US dollars", 1e-3, 0.01),
                "INDPRO": ("industrial production index", "index points", 1, 0.01)}
FRED_OTHER = {"GDP": ("nominal gross domestic product (quarterly, seasonally adjusted annual rate)", "trillions of US dollars", 1e-3, 0.01, "Q"),
              "GDPC1": ("real gross domestic product (quarterly, chained 2017 dollars, annual rate)", "trillions of chained 2017 US dollars", 1e-3, 0.01, "Q"),
              "WALCL": ("total assets of the Federal Reserve (Wednesday level)", "trillions of US dollars", 1e-6, 0.001, "W"),
              "MORTGAGE30US": ("30-year fixed mortgage average rate (weekly)", "percent", 1, 0.01, "W")}
NO_OBS_RULE = "If the series has no observation on that date, use its most recent earlier observation."


def fred_avail(C, sid, lo, hi):
    f = C.S.fred.get(sid)
    if f is None:
        raise Skip("missing_row")
    m = (f.d >= lo) & (f.d <= hi) & np.isfinite(f.v)
    idx = np.nonzero(m)[0]
    if len(idx) == 0:
        raise Skip("short_history")
    return f, idx


def gen_fred_lookup(C, split, rng, d):
    lo, hi = SPLIT_WINDOWS[split]
    kind = rng.choice(["daily", "daily", "fx", "monthly", "other"])
    if d == 3:
        kind = rng.choice(["yoy", "chg_bp", "month_avg"])
    if kind in ("daily", "chg_bp", "month_avg"):
        sid = rng.choice(sorted(FRED_DAILY))
        f, idx = fred_avail(C, sid, lo, hi)
        if kind == "daily":
            j = int(rng.choice(idx))
            qdate = str(f.d[j])
            nonobs = False
            if d >= 2 and rng.random() < 0.5:
                qdate = add_days(qdate, rng.randint(1, 3))
                nonobs = not (qdate in set(f.d[j:j + 4]))
            if qdate > hi:
                raise Skip("outside_split_window")
            rule = nonobs or (d >= 2 and rng.random() < 0.5)
            need_rule(nonobs, rule)
            od, v = C.S.fred[sid].le(qdate)
            scale = "pct" if d == 1 else rng.choice(["pct", "bp"])
            if scale == "bp":
                val, u, ut = v * 100, 1, "in basis points, rounded to the nearest whole basis point"
            else:
                val, u, ut = v, 0.01, "in percent, rounded to two decimals (e.g. 4.25 for 4.25%)"
            if not round_ok(val, u):
                raise Skip("rounding_boundary")
            prompt = rng.choice([
                "What was the %s (FRED series %s) on %s? Give it %s." % (FRED_DAILY[sid], sid, fmt_date(qdate, rng), ut),
                "Using FRED series %s (%s), report the value for %s %s." % (sid, FRED_DAILY[sid], fmt_date(qdate, rng), ut),
            ]) + ((" " + NO_OBS_RULE) if rule else "")
            return mk("fred_lookup", split, d, prompt, [num_key(val, u)],
                      "%s on %s (observation %s) was %s%s." % (sid, qdate, od, round(val, 2), " bp" if scale == "bp" else "%"),
                      [sid], od, od, ["fred_lookup"] + (["unit_convert"] if scale == "bp" else []), 1,
                      "fred.value for %s on or before %s" % (sid, qdate),
                      {"op": "daily", "series": sid, "date": qdate, "scale": scale})
        if kind == "chg_bp":
            j0, j1 = sorted(rng.sample(list(idx), 2))
            d0, d1_ = str(f.d[j0]), str(f.d[j1])
            if d0 == d1_:
                raise Skip("duplicate")
            _, v0 = C.S.fred[sid].le(d0)
            _, v1 = C.S.fred[sid].le(d1_)
            val = (v1 - v0) * 100
            if not round_ok(val, 1):
                raise Skip("rounding_boundary")
            prompt = rng.choice([
                "By how many basis points did the %s (FRED %s) change from %s to %s? Report the change in basis points rounded to the nearest whole number, negative for a decrease." % (FRED_DAILY[sid], sid, d0, d1_),
                "Compute the change in FRED series %s (%s) between %s and %s, in basis points (whole number; negative if it fell)." % (sid, FRED_DAILY[sid], long_date(d0), long_date(d1_)),
            ])
            return mk("fred_lookup", split, d, prompt, [num_key(val, 1)],
                      "%s moved from %s to %s, a change of %d bp." % (sid, v0, v1, round(val)),
                      [sid], d0, d1_, ["fred_lookup", "difference", "unit_convert"], 2,
                      "(value(%s) - value(%s)) * 100" % (d1_, d0),
                      {"op": "chg_bp", "series": sid, "d0": d0, "d1": d1_})
        # month_avg
        months = sorted(set(str(x)[:7] for x in f.d[idx]))
        ym = rng.choice(months[1:-1] if len(months) > 2 else months)
        m = np.array([str(x).startswith(ym) for x in f.d]) & np.isfinite(f.v)
        js = np.nonzero(m)[0]
        vals = [C.S.fred[sid]._val(j) for j in js]
        val = float(np.mean(vals))
        if not round_ok(val, 0.001):
            raise Skip("rounding_boundary")
        prompt = rng.choice([
            "What was the average of the daily observations of FRED series %s (%s) during %s? Average only the days that have an observation, and give the result in percent rounded to three decimals." % (sid, FRED_DAILY[sid], month_label(ym)),
            "Take every daily value of the %s (FRED %s) published for %s and report their simple mean in percent, rounded to three decimals." % (FRED_DAILY[sid], sid, month_label(ym)),
        ])
        return mk("fred_lookup", split, d, prompt, [num_key(val, 0.001)],
                  "Mean of %d daily %s values in %s = %.3f%%." % (len(vals), sid, ym, val),
                  [sid], str(f.d[js[0]]), str(f.d[js[-1]]), ["fred_range", "mean"], len(vals),
                  "mean of fred.value for %s with date in %s" % (sid, ym),
                  {"op": "month_avg", "series": sid, "month": ym})
    if kind == "fx":
        sid = rng.choice(sorted(FRED_FX))
        desc, unitname, per_usd = FRED_FX[sid]
        f, idx = fred_avail(C, sid, lo, hi)
        j = int(rng.choice(idx))
        qdate = str(f.d[j])
        od, v = C.S.fred[sid].le(qdate)
        invert = d >= 2 and rng.random() < 0.6
        if invert:
            if per_usd:
                val = 100.0 / v if sid in ("DEXJPUS", "DEXKOUS") else 1.0 / v
                base = "100 %s" % unitname if sid in ("DEXJPUS", "DEXKOUS") else "one %s" % unitname
                if sid == "DEXKOUS":
                    val, base = 1000.0 / v, "1,000 won"
                ask = "how many US dollars did %s buy" % base
            else:
                val = 1.0 / v
                ask = "how many %ss did one US dollar buy" % unitname
            u = 0.0001
        else:
            val, u = v, 0.0001
            ask = "what was the exchange rate quoted as %s" % desc
        if not round_ok(val, u):
            raise Skip("rounding_boundary")
        prompt = rng.choice([
            "According to FRED series %s (noon buying rates in New York), %s on %s? Round to four decimals." % (sid, ask, fmt_date(qdate, rng)),
            "Using the FRED exchange-rate series %s for %s, %s? Give four decimals." % (sid, fmt_date(qdate, rng), ask),
        ])
        return mk("fred_lookup", split, d, prompt, [num_key(val, u)],
                  "%s on %s was %s; the requested figure is %.4f." % (sid, qdate, v, val),
                  [sid], od, od, ["fred_lookup"] + (["invert_rate"] if invert else []), 1,
                  "fred.value %s on %s%s" % (sid, qdate, ", inverted" if invert else ""),
                  {"op": "fx", "series": sid, "date": qdate, "invert": invert})
    if kind == "monthly":
        sid = rng.choice(sorted(FRED_MONTHLY))
        desc, unit_txt, mult, u = FRED_MONTHLY[sid]
        f, idx = fred_avail(C, sid, lo, hi)
        j = int(rng.choice(idx))
        ym = str(f.d[j])[:7]
        v = C.S.fred[sid].get(ym + "-01")
        val = v * mult
        if not round_ok(val, u):
            raise Skip("rounding_boundary")
        dec = {0.1: "one", 0.01: "two", 0.001: "three"}[u]
        prompt = rng.choice([
            "What was the %s (FRED series %s) for %s? Report it in %s, rounded to %s decimals." % (desc, sid, month_label(ym), unit_txt, dec),
            "Look up FRED %s (%s) for the month of %s and give the value in %s with %s decimals." % (sid, desc, month_label(ym), unit_txt, dec),
        ])
        return mk("fred_lookup", split, d, prompt, [num_key(val, u)],
                  "%s for %s = %s (%s %s)." % (sid, ym, v, round(val, 3), unit_txt),
                  [sid], ym + "-01", ym + "-01", ["fred_lookup"] + (["unit_convert"] if mult != 1 else []), 1,
                  "fred.value %s on %s-01 x %g" % (sid, ym, mult),
                  {"op": "monthly", "series": sid, "month": ym, "mult": mult})
    if kind == "other":
        sid = rng.choice(sorted(FRED_OTHER))
        desc, unit_txt, mult, u, freq = FRED_OTHER[sid]
        f, idx = fred_avail(C, sid, lo, hi)
        j = int(rng.choice(idx))
        od = str(f.d[j])
        v = C.S.fred[sid].get(od)
        val = v * mult
        if not round_ok(val, u):
            raise Skip("rounding_boundary")
        dec = {0.01: "two", 0.001: "three"}[u]
        if freq == "Q":
            q = (int(od[5:7]) - 1) // 3 + 1
            when = "the %s quarter of %s" % (["first", "second", "third", "fourth"][q - 1], od[:4])
            whenk = "%sQ%d" % (od[:4], q)
        else:
            when = "the week of the observation dated %s" % od
            whenk = od
        prompt = rng.choice([
            "What was the %s (FRED series %s) for %s? Give it in %s, rounded to %s decimals." % (desc, sid, when, unit_txt, dec),
            "Report FRED %s (%s) for %s in %s (%s decimals)." % (sid, desc, when, unit_txt, dec),
        ])
        return mk("fred_lookup", split, d, prompt, [num_key(val, u)],
                  "%s for %s = %s, i.e. %s %s." % (sid, whenk, v, round(val, 3), unit_txt),
                  [sid], od, od, ["fred_lookup", "unit_convert"] if mult != 1 else ["fred_lookup"], 1,
                  "fred.value %s on %s x %g" % (sid, od, mult),
                  {"op": "other", "series": sid, "date": od, "mult": mult})
    # yoy: 12-month percent change of a monthly index
    sid = rng.choice(["CPIAUCSL", "CPILFESL", "PAYEMS", "INDPRO", "M2SL"])
    f, idx = fred_avail(C, sid, lo, hi)
    j = int(rng.choice(idx))
    ym = str(f.d[j])[:7]
    y0 = "%04d-%s" % (int(ym[:4]) - 1, ym[5:7])
    if y0 + "-01" < lo:
        raise Skip("outside_split_window")
    v1 = C.S.fred[sid].get(ym + "-01")
    v0 = C.S.fred[sid].get(y0 + "-01")
    val = (v1 / v0 - 1) * 100
    if not round_ok(val, 0.01):
        raise Skip("rounding_boundary")
    desc = FRED_MONTHLY[sid][0]
    prompt = rng.choice([
        "What was the year-over-year percent change in the %s (FRED %s) in %s, comparing that month with the same month a year earlier? Answer %s." % (desc, sid, month_label(ym), PCT_SIGNED),
        "Compute the 12-month change of FRED series %s (%s) for %s versus %s, %s." % (sid, desc, month_label(ym), month_label(y0), PCT_SIGNED),
    ])
    return mk("fred_lookup", split, d, prompt, [num_key(val, 0.01)],
              "%s %s = %s vs %s = %s: %.2f%%." % (sid, ym, v1, y0, v0, val),
              [sid], y0 + "-01", ym + "-01", ["fred_lookup", "pct_change"], 2,
              "value(%s)/value(%s)-1" % (ym, y0), {"op": "yoy", "series": sid, "month": ym})


def scale_spec(rng, d, tag):
    if tag.startswith("EarningsPerShare"):
        return "eps", 1.0, 0.01, "in US dollars per share, rounded to two decimals"
    if d == 1:
        return "millions", 1e-6, 0.1, "in millions of US dollars, rounded to one decimal"
    return rng.choice([
        ("millions", 1e-6, 0.1, "in millions of US dollars, rounded to one decimal"),
        ("billions", 1e-9, 0.001, "in billions of US dollars, rounded to three decimals"),
        ("thousands", 1e-3, 1, "in thousands of US dollars, rounded to the nearest thousand dollars (a whole number of thousands)"),
    ])


def fy_phrase(C, cik, fy, d, inst=False):
    E = C.X.info(cik, fy)["E"]
    if d == 3:
        return "fiscal year %d" % fy if not inst else "the end of fiscal year %d" % fy
    if inst:
        return "the end of fiscal year %d (balance-sheet date %s)" % (fy, E)
    return "fiscal year %d (the fiscal year ended %s)" % (fy, E)


def gen_xbrl_lookup(C, split, rng, d):
    pool = C.sec_eq[split]
    sym = rng.choice(pool)
    C.name_ok(sym, d)
    cik = C.cik[sym]
    years = C.X.years(cik)
    if not years:
        raise Skip("sec_no_fiscal_year")
    fy = rng.choice(years)
    op = "value" if d < 3 else rng.choice(["yoy", "yoy_pct", "q4"])
    tag = rng.choice(DUR_TAGS + INST_TAGS if op in ("value", "yoy", "yoy_pct") else DUR_TAGS[:-1])
    inst = tag in INST_TAGS
    kind = "inst" if inst else "dur"
    v, end, start = C.X.value(cik, fy, tag, kind)
    restated = C.X.restated(cik, tag, start, end, v)
    if restated and d == 1:
        raise Skip("sec_restated")
    name, mult, u, ut = scale_spec(rng, d, tag)
    ent = C.ent(sym, d)
    src = "as reported in the company's original annual report on Form 10-K for that fiscal year (not later restatements)"
    sign = " (negative for a loss or outflow)" if tag in SIGNED_TAGS else ""
    tdesc = "%s (XBRL tag us-gaap:%s)" % (TAG_DESC[tag], tag)
    rows = 1
    if op == "value":
        val = v * mult
        when = fy_phrase(C, cik, fy, d, inst)
        prompt = rng.choice([
            "What was %s's %s %s %s, %s? Give the value %s%s." % (ent, tdesc, "at" if inst else "for", when, src, ut, sign),
            "From SEC XBRL data, report the %s of %s %s %s, %s. Express it %s%s." % (tdesc, ent, "at" if inst else "for", when, src, ut, sign),
            "Using %s's 10-K financial data, find %s %s %s (%s). Answer %s%s." % (ent, tdesc, "at" if inst else "for", when, src.replace("as reported in ", ""), ut, sign),
        ])
        ops, sol = ["sec_fiscal_year_resolve", "sec_fact_lookup", "scale_convert"], "facts %s end=%s in accn %s" % (tag, end, C.X.info(cik, fy)["accn"])
        ans = "%s %s FY%d (%s): %s USD -> %s" % (sym, tag, fy, end, v, round(val, 3))
        start_d, end_d = end, end
    elif op in ("yoy", "yoy_pct"):
        v0, end0, start0 = C.X.value(cik, fy - 1, tag, kind)
        if op == "yoy":
            if tag.startswith("EarningsPerShare"):
                mult, u, ut = 1.0, 0.01, "in US dollars per share, rounded to two decimals"
            else:
                mult, u, ut = 1e-6, 0.1, "in millions of US dollars, rounded to one decimal"
            val = (v - v0) * mult
            prompt = rng.choice([
                "By how much did %s's %s change from fiscal year %d to fiscal year %d? Use each year's value as reported in that year's own original Form 10-K. Answer %s, negative for a decrease." % (ent, tdesc, fy - 1, fy, ut),
                "Compute the change in %s for %s between fiscal %d and fiscal %d (later minus earlier), taking each year's figure from its own original 10-K; give it %s (negative if it fell)." % (tdesc, ent, fy - 1, fy, ut),
            ])
        else:
            if v0 == 0 or (v0 < 0):
                raise Skip("degenerate_value")
            val, u = (v / v0 - 1) * 100, 0.01
            prompt = rng.choice([
                "What was the year-over-year growth of %s's %s in fiscal year %d versus fiscal year %d? Use each year's value from that year's own original Form 10-K and answer %s." % (ent, tdesc, fy, fy - 1, PCT_SIGNED),
                "Using each fiscal year's original 10-K, compute the percentage change in %s for %s from fiscal %d to fiscal %d, %s." % (tdesc, ent, fy - 1, fy, PCT_SIGNED),
            ])
        restated = restated or C.X.restated(cik, tag, start0, end0, v0)
        ops = ["sec_fiscal_year_resolve", "sec_fact_lookup", "sec_fact_lookup", "difference" if op == "yoy" else "pct_change"]
        sol = "%s FY%d (%s) vs FY%d (%s) from each original 10-K" % (tag, fy, end, fy - 1, end0)
        ans = "%s %s FY%d %s vs FY%d %s -> %s" % (sym, tag, fy, v, fy - 1, v0, round(val, 3))
        rows, start_d, end_d = 2, end0, end
    else:  # q4 = FY - 9M YTD from the Q3 10-Q
        v9, end9, _ = C.X.nine_month(cik, fy, tag)
        mult, u, ut = 1e-6, 0.1, "in millions of US dollars, rounded to one decimal"
        val = (v - v9) * mult
        prompt = rng.choice([
            "Derive %s's fourth-quarter %s for fiscal year %d as the full-year value in the original 10-K minus the nine-month year-to-date value in the original third-quarter 10-Q of the same fiscal year. Answer %s%s." % (ent, tdesc, fy, ut, sign),
            "%s does not report Q4 separately. Compute fiscal %d Q4 %s = annual figure (original 10-K) minus nine-month figure (original Q3 10-Q of fiscal %d), %s%s." % (ent, fy, tdesc, fy, ut, sign),
        ])
        ops = ["sec_fiscal_year_resolve", "sec_fact_lookup", "sec_quarter_lookup", "difference"]
        sol = "%s FY%d %s minus 9M to %s" % (tag, fy, v, end9)
        ans = "%s Q4 FY%d %s = %s - %s -> %s" % (sym, fy, tag, v, v9, round(val, 3))
        rows, start_d, end_d = 2, end9, end
    if not round_ok(val, u):
        raise Skip("rounding_boundary")
    return mk("xbrl_lookup", split, d, prompt, [num_key(val, u)], ans,
              [sym, "CIK:%d" % cik], start_d, end_d, ops, rows, sol,
              {"op": op, "symbol": sym, "cik": cik, "fy": fy, "tag": tag, "kind": kind, "mult": mult,
               "restated": bool(restated)})


# ============================================================================
# Single-entity computations

def gen_period_return(C, split, rng, d):
    sym, lo, hi = pick_entity(C, rng, split, 0.6, ("index", "etf", "future", "crypto"))
    C.name_ok(sym, d)
    px = C.S.px[sym]
    L = {1: (20, 130), 2: (120, 760), 3: (250, 2500)}[d]
    L = rng.randint(*L)
    sp = C.S.splits.get(sym, [])
    straddle = px.kind in ("equity", "etf") and sp and d >= 2 and rng.random() < 0.6
    if straddle:
        sd = rng.choice(sp)[0]
        k = px.i_ge(sd)
        a, b = px.i_ge(lo), px.i_le(hi)
        if not (a < k <= b):
            raise Skip("short_history")
        if max(a, k - L + 1) > k - 1:
            raise Skip("short_history")
        i0 = rng.randint(max(a, k - L + 1), k - 1)
        i1 = min(b, i0 + L - 1)
        if i1 < k:
            raise Skip("short_history")
    else:
        i0, i1 = pick_span(rng, px, lo, hi, L)
    if not px.clean(i0, i1) or px.close[i0:i1 + 1].min() <= 0:
        raise Skip("bad_row")
    has_div = px.kind in ("equity", "etf")
    if not has_div:
        measures = ["price"]
    elif d == 3:
        measures = ["price", "total"]
    else:
        measures = [rng.choice(["price", "total"])]
    p_cal = {1: 0.0, 2: 0.4, 3: 0.6}[d]
    s_date, nt0 = cal_date_near(rng, px, i0, p_cal)
    e_date, nt1 = cal_date_near(rng, px, i1, p_cal)
    rule = d >= 2
    need_rule(nt0 or nt1, rule)
    keys, vals = [], []
    for m in measures:
        c = px.adj_close if m == "total" else px.close
        v = (c[i1] / c[i0] - 1) * 100
        if not round_ok(v, 0.01):
            raise Skip("rounding_boundary")
        keys.append(num_key(v, 0.01))
        vals.append(v)
    ent = C.ent(sym, d)
    if not has_div:
        mdesc = "price change (percentage change in the closing level)"
    elif len(measures) == 2:
        mdesc = "%s, and its %s" % (PRICE_DESC, TOTAL_DESC)
    else:
        mdesc = PRICE_DESC if measures[0] == "price" else TOTAL_DESC
    prompt = rng.choice([
        "What was the %s of %s from the close on %s to the close on %s? Give %s %s." % (
            mdesc, ent, fmt_date(s_date, rng), fmt_date(e_date, rng), "each" if len(measures) == 2 else "it", PCT_SIGNED),
        "Compute the %s for %s between the closing prices of %s and %s, %s." % (
            mdesc, ent, fmt_date(s_date, rng), fmt_date(e_date, rng), PCT_SIGNED),
        "Measured close to close from %s to %s, what %s did %s deliver? Answer %s." % (
            fmt_date(s_date, rng), fmt_date(e_date, rng), mdesc, ent, PCT_SIGNED),
    ])
    if rule:
        prompt += " " + NT_RULE
    ops = ["calendar_resolve", "price_lookup", "pct_change"] + (["split_adjust"] if straddle else []) + \
          (["dividend_adjust"] if "total" in measures else [])
    return mk("period_return", split, d, prompt, keys,
              "%s from %s to %s: %s." % (sym, px.d[i0], px.d[i1], ", ".join(
                  "%s return %.2f%%" % (m, v) for m, v in zip(measures, vals))),
              [sym], str(px.d[i0]), str(px.d[i1]), ops, 2,
              "close(or adj_close)[%s]/[%s]-1" % (px.d[i1], px.d[i0]),
              {"symbol": sym, "start": s_date, "end": e_date, "measures": measures, "straddles_split": bool(straddle)})


def gen_nth_day_move(C, split, rng, d):
    sym, lo, hi = pick_entity(C, rng, split, 0.6, ("index", "etf", "future", "crypto"))
    C.name_ok(sym, d)
    px = C.S.px[sym]
    n = {1: 1, 2: rng.randint(2, 5), 3: rng.randint(5, 15)}[d]
    a, b = px.i_ge(lo) + 1, px.i_le(hi) - n
    if b <= a:
        raise Skip("short_history")
    i = rng.randint(a, b)
    D, _ = cal_date_near(rng, px, i, 0.5)  # D need not be a trading day: counting starts after it
    j = px.i_le(D) + n
    mode = rng.choice(["cc", "oc"])
    if not px.clean(j - 1, j):
        raise Skip("bad_row")
    if mode == "cc":
        v = (px.close[j] / px.close[j - 1] - 1) * 100
        mdesc = "close-to-close return (split-adjusted close of that day versus the previous trading day's split-adjusted close)"
    else:
        v = (px.close[j] / px.open[j] - 1) * 100
        mdesc = "open-to-close return (that day's close versus that day's open)"
    if px.kind not in ("equity", "etf"):
        mdesc = mdesc.replace("split-adjusted ", "")
    if not round_ok(v, 0.01):
        raise Skip("rounding_boundary")
    keys = [num_key(v, 0.01)]
    target = str(px.d[j])
    if d == 3:
        keys.append(date_key(target))
    ent = C.ent(sym, d)
    nth = "next trading day" if n == 1 else "%s trading day" % ordinal(n)
    ask_date = " Also give that day's date (YYYY-MM-DD)." if d == 3 else ""
    prompt = rng.choice([
        "For %s, take the %s after %s (count trading days of that instrument only, starting after %s). What was its %s on that day? Answer %s.%s" % (
            ent, nth, fmt_date(D, rng), D, mdesc, PCT_SIGNED, ask_date),
        "What %s did %s post on the %s following %s? (Trading days are counted from the first one after that date.) Give it %s.%s" % (
            mdesc, ent, nth, fmt_date(D, rng), PCT_SIGNED, ask_date),
    ])
    return mk("nth_day_move", split, d, prompt, keys,
              "The %s after %s for %s is %s; %s return %.2f%%." % (nth, D, sym, target, mode, v),
              [sym], str(px.d[j - 1]), target, ["nth_trading_day", "pct_change"], 2,
              "trading day index(last<=%s)+%d -> %s" % (D, n, target),
              {"symbol": sym, "date": D, "n": n, "mode": mode})


def month_ends(px, i_lo, i_hi):
    """Indices of the last trading day of each month in [i_lo, i_hi]."""
    ym = np.array([s[:7] for s in px.d[i_lo:i_hi + 2]]) if i_hi + 1 < len(px) else \
        np.array([s[:7] for s in px.d[i_lo:i_hi + 1]] + ["9999-99"])
    last = np.nonzero(ym[:-1] != ym[1:])[0] + i_lo
    return last


def week_ends(px, i_lo, i_hi):
    wk = [np_date(s).isocalendar()[:2] for s in px.d[i_lo:min(i_hi + 2, len(px))]]
    if i_hi + 1 >= len(px):
        wk.append((9999, 99))
    return np.array([i_lo + k for k in range(len(wk) - 1) if wk[k] != wk[k + 1]])


def gen_extreme_period(C, split, rng, d):
    sym, lo, hi = pick_entity(C, rng, split, 0.5, ("index", "etf", "future", "crypto"))
    C.name_ok(sym, d)
    px = C.S.px[sym]
    freq = "month" if d == 1 else rng.choice(["month", "week"]) if d == 2 else "week" if rng.random() < 0.6 else "month"
    direction = rng.choice(["gain", "loss"])
    hi = min(hi, "2025-09-30")
    a, b = px.i_ge(lo), px.i_le(hi)
    if freq == "month":
        ends = month_ends(px, a, b)
        nper = {1: 12, 2: rng.randint(24, 48), 3: rng.randint(60, 120)}[d]
    else:
        ends = week_ends(px, a, b)
        nper = {2: rng.randint(26, 78), 3: rng.randint(104, 260)}.get(d, 52)
    if len(ends) < nper + 2:
        raise Skip("short_history")
    k0 = rng.randint(1, len(ends) - nper - 1)  # ends[k0-1] is the base close
    sel = ends[k0 - 1:k0 + nper]
    if freq == "month":
        first_m, last_m = px.d[sel[1]][:7], px.d[sel[-1]][:7]
        mi = [int(px.d[x][:4]) * 12 + int(px.d[x][5:7]) for x in sel]
        if any(b_ - a_ != 1 for a_, b_ in zip(mi[:-1], mi[1:])):
            raise Skip("missing_row")
    if not px.clean(int(sel[0]), int(sel[-1])) or px.close[sel].min() <= 0:
        raise Skip("bad_row")
    r = (px.close[sel[1:]] / px.close[sel[:-1]] - 1) * 100
    order = np.argsort(r) if direction == "loss" else np.argsort(-r)
    best, second = r[order[0]], r[order[1]]
    if abs(best - second) < 0.02:
        raise Skip("near_tie")
    if not round_ok(best, 0.01):
        raise Skip("rounding_boundary")
    win = sel[1 + order[0]]
    ent = C.ent(sym, d)
    word = "largest gain" if direction == "gain" else "largest loss (most negative return)"
    if freq == "month":
        ym = px.d[win][:7]
        k_per = text_key(month_label(ym), month_aliases(ym))
        span = "from %s through %s" % (month_label(first_m), month_label(last_m))
        defn = ("A month's return is its last split-adjusted close divided by the previous month's last "
                "split-adjusted close, minus 1.")
        ident = "Name the month"
        per_txt = month_label(ym)
    else:
        wk_s = px.d[sel[1]]
        wk_e = px.d[sel[-1]]
        k_per = date_key(str(px.d[win]))
        span = "over the calendar weeks (Monday to Sunday) whose last trading day falls between %s and %s inclusive" % (wk_s, wk_e)
        defn = ("A week's return is the split-adjusted close on its last trading day divided by the "
                "split-adjusted close on the last trading day of the previous week, minus 1.")
        ident = "Identify the week by the date of its last trading day (YYYY-MM-DD)"
        per_txt = "week ending %s" % px.d[win]
    if px.kind not in ("equity", "etf"):
        defn = defn.replace("split-adjusted close", "closing level")
    prompt = rng.choice([
        "Which %s had the %s for %s %s? %s %s and give that return %s." % (
            freq, word, ent, span, defn, ident, PCT_SIGNED),
        "%s Looking at %s %s, find the %s with the %s. %s and report its return %s." % (
            defn, ent, span, freq, word, ident, PCT_SIGNED),
    ])
    return mk("extreme_period", split, d, prompt, [k_per, num_key(best, 0.01)],
              "%s's %s %s: %s with %.2f%% (runner-up %.2f%%)." % (sym, freq, direction, per_txt, best, second),
              [sym], str(px.d[sel[0]]), str(px.d[sel[-1]]), ["resample_%sly" % freq, "pct_change", "argmax"],
              int(sel[-1] - sel[0] + 1), "%s-end closes, %s of returns" % (freq, "min" if direction == "loss" else "max"),
              {"symbol": sym, "freq": freq, "direction": direction, "first": str(px.d[sel[1]]),
               "last": str(px.d[sel[-1]])})


def gen_threshold_scan(C, split, rng, d):
    sym, lo, hi = pick_entity(C, rng, split, 0.6, ("index", "etf", "future", "crypto"))
    C.name_ok(sym, d)
    px = C.S.px[sym]
    L = rng.randint(*{1: (60, 250), 2: (250, 750), 3: (500, 1500)}[d])
    i0, i1 = pick_span(rng, px, lo, hi, L, lookback=1)
    if not px.clean(i0 - 1, i1) or px.close[i0 - 1:i1 + 1].min() <= 0:
        raise Skip("bad_row")
    modes = ["pct_drop", "pct_gain"]
    if px.kind in ("equity", "etf"):
        modes.append("usd_drop")
    mode = rng.choice(modes)
    if mode == "usd_drop":
        fac = px.raw_close[i0 - 1:i1 + 1] / px.close[i0 - 1:i1 + 1]
        if np.ptp(fac) > 1e-6 * fac.mean():
            raise Skip("split_in_window")
        chg = np.diff(px.raw_close[i0 - 1:i1 + 1])
        typical = np.median(np.abs(chg))
        X = max(0.5, round(float(np.quantile(np.abs(chg), rng.uniform(0.9, 0.99))) * 2) / 2)
        hit = chg < -X
        near = np.abs(chg + X) < 0.005
        thr_txt = "a decline of more than $%.2f in the unadjusted closing price from the previous trading day's unadjusted close" % X
    else:
        r = daily_ret(px, i0, i1) * 100
        X = rng.choice([1, 1.5, 2, 3, 4, 5]) if px.kind != "crypto" else rng.choice([5, 7, 10])
        if mode == "pct_drop":
            hit, near = r < -X, np.abs(r + X) < 0.005
            thr_txt = "a daily return below -%g%% (close versus previous close, split-adjusted)" % X
        else:
            hit, near = r > X, np.abs(r - X) < 0.005
            thr_txt = "a daily return above +%g%% (close versus previous close, split-adjusted)" % X
    if near.any():
        raise Skip("near_threshold")
    cnt = int(hit.sum())
    if cnt == 0:
        raise Skip("empty_answer")
    dates = [str(x) for x in px.d[i0:i1 + 1][hit]]
    list_mode = d >= 2 and cnt <= 6 and rng.random() < 0.6
    keys = [count_key(cnt)]
    if list_mode:
        keys.append(set_key([date_key(x) for x in dates]))
    ent = C.ent(sym, d)
    s, e = str(px.d[i0]), str(px.d[i1])
    ask = "How many trading days" if not list_mode else "How many trading days, and which ones (list their dates as YYYY-MM-DD)"
    prompt = rng.choice([
        "%s between %s and %s (inclusive) did %s record %s?" % (ask, fmt_date(s, rng), fmt_date(e, rng), ent, thr_txt),
        "Scan %s's daily closes from %s through %s. %s show %s?" % (ent, fmt_date(s, rng), fmt_date(e, rng), ask, thr_txt),
    ])
    if px.kind not in ("equity", "etf"):
        prompt = prompt.replace(", split-adjusted", "")
    return mk("threshold_scan", split, d, prompt, keys,
              "%d days: %s." % (cnt, ", ".join(dates[:20]) + (" ..." if len(dates) > 20 else "")),
              [sym], str(px.d[i0 - 1]), e, ["price_range", "daily_return", "threshold_filter", "count"],
              i1 - i0 + 2, "count of days with %s in [%s,%s]" % (mode, s, e),
              {"symbol": sym, "start": s, "end": e, "mode": mode, "X": X, "list": list_mode})


def drawdown(c):
    peak = np.maximum.accumulate(c)
    return c / peak - 1, peak


def gen_max_drawdown(C, split, rng, d):
    sym, lo, hi = pick_entity(C, rng, split, 0.6, ("index", "etf", "future", "crypto"))
    C.name_ok(sym, d)
    px = C.S.px[sym]
    L = rng.randint(*{1: (60, 250), 2: (250, 1000), 3: (750, 3000)}[d])
    i0, i1 = pick_span(rng, px, lo, hi, L)
    if not px.clean(i0, i1) or px.close[i0:i1 + 1].min() <= 0:
        raise Skip("bad_row")
    measure = "price" if (d == 1 or px.kind not in ("equity", "etf")) else rng.choice(["price", "total"])
    c = (px.adj_close if measure == "total" else px.close)[i0:i1 + 1]
    dd, peak = drawdown(c)
    k = int(np.argmin(dd))
    mdd = -dd[k] * 100
    if mdd < 1:
        raise Skip("degenerate_value")
    if (np.abs(dd * 100 + mdd) < 0.02).sum() > 1:
        raise Skip("near_tie")
    if not round_ok(mdd, 0.01):
        raise Skip("rounding_boundary")
    pk = int(np.nonzero(c[:k + 1] == peak[k])[0][0])
    if (np.abs(c[:k + 1] / peak[k] - 1) < 1e-4).sum() > 1:
        if d == 3:
            raise Skip("near_tie")
    trough, peakd = str(px.d[i0 + k]), str(px.d[i0 + pk])
    keys = [num_key(mdd, 0.01), date_key(trough)]
    if d == 3:
        keys.append(date_key(peakd))
    s, e = str(px.d[i0]), str(px.d[i1])
    ent = C.ent(sym, d)
    series = {"price": "split-adjusted daily closes" if px.kind in ("equity", "etf") else "daily closes",
              "total": "daily closes adjusted for both splits and dividends (total return)"}[measure]
    extra = " and the date of the peak that preceded it" if d == 3 else ""
    prompt = rng.choice([
        "Using %s of %s from %s to %s (inclusive), what was the maximum drawdown (largest percentage decline from a running peak close to a later close within the window)? Report it as a positive percentage rounded to two decimals, along with the date of the trough%s (YYYY-MM-DD)." % (
            series, ent, fmt_date(s, rng), fmt_date(e, rng), extra),
        "Compute the maximum peak-to-trough drawdown of %s between %s and %s based on %s. Give the drawdown as a positive percentage with two decimals and the trough date%s." % (
            ent, fmt_date(s, rng), fmt_date(e, rng), series, extra),
    ])
    return mk("max_drawdown", split, d, prompt, keys,
              "Max drawdown %.2f%% from peak %s to trough %s." % (mdd, peakd, trough),
              [sym], s, e, ["price_range", "running_max", "drawdown", "argmin"], i1 - i0 + 1,
              "min(c/cummax(c)-1) over [%s,%s]" % (s, e),
              {"symbol": sym, "start": s, "end": e, "measure": measure, "with_peak": d == 3})


def gen_rolling_vol(C, split, rng, d):
    spread = d == 3 and rng.random() < 0.7
    if spread:
        A, B = rng.choice(SPREAD_PAIRS)
        lo, hi = SPLIT_WINDOWS[split]
        pa, pb = C.S.px[A], C.S.px[B]
        L = rng.randint(60, 250)
        i0, i1 = pick_span(rng, pa, lo, hi, L, lookback=21)
        j0, j1 = pb.i_ge(pa.d[i0 - 20]), pb.i_le(pa.d[i1])
        if not np.array_equal(pa.d[i0 - 20:i1 + 1], pb.d[j0:j1 + 1]):
            raise Skip("calendar_mismatch")
        if not (pa.clean(i0 - 20, i1) and pb.clean(j0, j1)):
            raise Skip("bad_row")
        ra = pa.close[i0 - 19:i1 + 1] / pa.close[i0 - 20:i1] - 1
        rb = pb.close[j0 + 1:j1 + 1] / pb.close[j0:j1] - 1
        r = (ra - rb) * 100
        syms = [A, B]
        px = pa
    else:
        sym, lo, hi = pick_entity(C, rng, split, 0.5, ("us", "future", "crypto"))
        C.name_ok(sym, d)
        px = C.S.px[sym]
        L = rng.randint(*{1: (20, 60), 2: (60, 250), 3: (120, 500)}[d])
        i0, i1 = pick_span(rng, px, lo, hi, L, lookback=21)
        if not px.clean(i0 - 20, i1) or px.close[i0 - 20:i1 + 1].min() <= 0:
            raise Skip("bad_row")
        r = (px.close[i0 - 19:i1 + 1] / px.close[i0 - 20:i1] - 1) * 100
        syms = [sym]
    # rolling 20-observation sample std ending at each day of the window
    sd = np.array([np.std(r[t - 19:t + 1], ddof=1) for t in range(19, len(r))])
    val = float(sd.mean())
    if not round_ok(val, 0.001):
        raise Skip("rounding_boundary")
    s, e = str(px.d[i0]), str(px.d[i1])
    base = ("Daily returns are close / previous trading day's close - 1 (split-adjusted closes), in percent. "
            "For each trading day, take the sample standard deviation (n-1 denominator) of the 20 most recent "
            "daily returns ending on that day (these may reach back before the window start).")
    if spread:
        ea, eb = C.ent(syms[0], 2), C.ent(syms[1], 2)
        prompt = rng.choice([
            "%s Define the daily spread as the daily return of %s minus the daily return of %s, using only dates on which both trade. Compute the rolling 20-day standard deviation of this spread for every trading day from %s to %s and report its average, in percentage points rounded to three decimals." % (base, ea, eb, s, e),
            "Consider the return spread between %s and %s (first minus second, daily). %s Average the rolling 20-day standard deviation of the spread over the trading days %s through %s; answer in percentage points with three decimals." % (ea, eb, base, s, e),
        ])
    else:
        ent = C.ent(syms[0], d)
        if px.kind not in ("equity", "etf"):
            base = base.replace(" (split-adjusted closes)", "")
        prompt = rng.choice([
            "%s For %s, average these rolling 20-day standard deviations over all trading days from %s to %s inclusive. Report the average in percent, rounded to three decimals (e.g. 1.234)." % (base, ent, fmt_date(s, rng), fmt_date(e, rng)),
            "What was the average rolling 20-day volatility of %s's daily returns between %s and %s? %s Average over the trading days in the window and give the result in percent with three decimals." % (ent, fmt_date(s, rng), fmt_date(e, rng), base),
        ])
    return mk("rolling_vol", split, d, prompt, [num_key(val, 0.001)],
              "Mean rolling 20-day std over %d days = %.3f%%." % (len(sd), val),
              syms, str(px.d[i0 - 20]), e, ["price_range", "daily_return"] + (["spread"] if spread else []) +
              ["rolling_std", "mean"], (i1 - i0 + 21) * len(syms), "mean of rolling(20).std(ddof=1)",
              {"symbols": syms, "start": s, "end": e})


def gen_split_event(C, split, rng, d):
    pool = [(s, x) for s in C.eq[split] for x in C.true_splits.get(s, [])]
    lo, hi = SPLIT_WINDOWS[split]
    pool += [(s, x) for s in C.etf for x in C.true_splits.get(s, []) if lo <= x[0] <= hi]
    if not pool:
        raise Skip("short_history")
    sym, (E, ratio, rtxt, stype) = rng.choice(pool)
    C.name_ok(sym, d)
    px = C.S.px[sym]
    if not px.is_td(E):
        raise Skip("missing_row")
    i = px.i_le(E)
    if i < 1 or not px.clean(i - 1, i):
        raise Skip("bad_row")
    if C.kind(sym) == "etf" and px.d[i - 1] < lo:
        raise Skip("outside_split_window")
    pr = px.raw_close[i - 1] / px.raw_open[i]
    if not (0.6 < pr / ratio < 1.6):
        raise Skip("split_price_inconsistent")
    diff = px.raw_open[i] - px.raw_close[i - 1]
    keys = [num_key(ratio, 0.001), num_key(diff, 0.01)]
    if not (round_ok(ratio, 0.001) and round_ok(diff, 0.01)):
        raise Skip("rounding_boundary")
    want_pr = d == 3 or (d == 2 and rng.random() < 0.5)
    if want_pr:
        if not round_ok(pr, 0.001):
            raise Skip("rounding_boundary")
        keys.append(num_key(pr, 0.001))
    ym = E[:7]
    same_month = [x for x in C.S.splits.get(sym, []) if x[0][:7] == ym]
    ent = C.ent(sym, d)
    if d == 1:
        ident = "the stock split that took effect on %s" % fmt_date(E, rng)
    elif d == 2:
        if len(same_month) != 1:
            raise Skip("ambiguous_event")
        ident = "the stock split that took effect in %s" % month_label(ym)
    else:
        nxt = [x[0] for x in C.true_splits[sym] if x[0] > E]
        lim = min([CUTOFF] + [add_days(n, -1) for n in nxt])
        if lim <= E:
            raise Skip("ambiguous_event")
        D = add_days(E, rng.randint(0, min(400, (np_date(lim) - np_date(E)).days)))
        ident = ("its most recent stock split (an ordinary split or reverse split; ignore spin-off and "
                 "stock-dividend adjustments) effective on or before %s" % fmt_date(D, rng))
        keys.insert(0, date_key(E))
    pr_txt = (", and the previous unadjusted close divided by the unadjusted open on the effective date (three decimals)"
              if want_pr else "")
    date_txt = "the effective date (YYYY-MM-DD), " if d == 3 else ""
    prompt = rng.choice([
        "For %s, consider %s. Report %sthe split ratio as new shares per old share (e.g. 2 for a 2-for-1 split, 0.1 for a 1-for-10 reverse split; three decimals), and the unadjusted opening price on the effective date minus the unadjusted closing price on the previous trading day, in US dollars rounded to two decimals (negative if the open was lower)%s." % (
            ent, ident, date_txt, pr_txt),
        "Look at %s for %s. Give %sits ratio in new shares per old share (three decimals), plus the gap between the effective date's unadjusted open and the prior trading day's unadjusted close (open minus close, US dollars, two decimals)%s." % (
            ident, ent, date_txt, pr_txt),
    ])
    return mk("split_event", split, d, prompt, keys,
              "%s split %s (%s) effective %s; raw open %.2f vs prior raw close %.2f: diff %.2f%s." % (
                  sym, rtxt, ratio, E, px.raw_open[i], px.raw_close[i - 1], diff,
                  ", close/open %.3f" % pr if want_pr else ""),
              [sym], str(px.d[i - 1]), E, ["action_lookup", "price_lookup", "difference"] + (["ratio"] if want_pr else []),
              3, "actions split %s; raw_open[%s]-raw_close[%s]" % (E, E, px.d[i - 1]),
              {"symbol": sym, "effective": E, "ratio": ratio, "want_pr": want_pr, "ask_date": d == 3})


def gen_dividends(C, split, rng, d):
    if rng.random() < 0.75:
        sym = rng.choice(C.eq[split])
        lo, hi = GLOBAL_START, CUTOFF
    else:
        sym = rng.choice(C.etf)
        lo, hi = SPLIT_WINDOWS[split]
    C.name_ok(sym, d)
    if sym not in C.S.divs:
        raise Skip("empty_answer")
    dd, dv, dr = C.S.divs[sym]
    px = C.S.px[sym]
    op = "year_raw" if d == 1 else rng.choice(["year_raw", "year_adj", "ttm"]) if d == 2 else rng.choice(["year_count", "ttm"])
    ent = C.ent(sym, d)
    if op.startswith("year"):
        y0 = max(int(lo[:4]), int(px.d[0][:4]) + 1)
        y1 = min(int(hi[:4]), 2024)
        if y1 < y0:
            raise Skip("short_history")
        Y = rng.randint(y0, y1)
        m = np.array([x.startswith(str(Y)) for x in dd])
        if not m.any():
            raise Skip("empty_answer")
        if any(x[5:] in ("01-01", "01-02", "01-03", "12-29", "12-30", "12-31") for x in dd[m]):
            raise Skip("boundary_sensitive")
        vals = dr[m] if op in ("year_raw", "year_count") else dv[m]
        if op == "year_adj" and np.allclose(dr[m], dv[m]):
            raise Skip("no_adjustment_difference")
        tot = float(vals.sum())
        if not round_ok(tot, 0.0001):
            raise Skip("rounding_boundary")
        keys = [num_key(tot, 0.0001)]
        basis = ("unadjusted (the per-share amounts as declared at the time)" if op != "year_adj" else
                 "split-adjusted (restated for all splits up to 2025-10-30)")
        if op == "year_count":
            keys.insert(0, count_key(int(m.sum())))
            prompt = rng.choice([
                "How many cash dividends with ex-dividend dates in calendar year %d did %s pay, and what was their total per share, %s? Give the total in US dollars rounded to four decimals." % (Y, ent, basis),
                "Count the dividends of %s whose ex-dividend date falls in %d and add up their %s per-share amounts (US dollars, four decimals)." % (ent, Y, basis.split(" (")[0]),
            ])
        else:
            prompt = rng.choice([
                "What was the total %s cash dividend per share of %s with ex-dividend dates in calendar year %d? Answer in US dollars rounded to four decimals." % (basis, ent, Y),
                "Sum the per-share dividends of %s that went ex-dividend during %d, using %s amounts. Give US dollars to four decimals." % (ent, Y, basis),
            ])
        return mk("dividends", split, d, prompt, keys,
                  "%s %d: %d dividends, total %.4f (%s)." % (sym, Y, int(m.sum()), tot, op),
                  [sym], str(dd[m][0]), str(dd[m][-1]), ["action_range", "sum"] + (["split_adjust"] if op == "year_adj" else []),
                  int(m.sum()), "sum actions.%s for dividend ex-dates in %d" % ("value" if op == "year_adj" else "value_raw", Y),
                  {"op": op, "symbol": sym, "year": Y})
    # trailing twelve-month yield
    a = px.i_ge(max(lo, add_days(px.d[0], 400)))
    b = px.i_le(hi)
    if b <= a:
        raise Skip("short_history")
    i = rng.randint(a, b)
    D, nt = cal_date_near(rng, px, i, 0.4)
    start = minus_one_year(D)
    if start < lo:
        raise Skip("outside_split_window")
    m = (dd > start) & (dd <= D)
    if not m.any():
        raise Skip("empty_answer")
    near = [x for x in dd if abs((np_date(x) - np_date(start)).days) <= 3 or abs((np_date(x) - np_date(D)).days) <= 1]
    if near:
        raise Skip("boundary_sensitive")
    if px.bad[i]:
        raise Skip("bad_row")
    y = float(dv[m].sum() / px.close[i] * 100)
    if not round_ok(y, 0.01):
        raise Skip("rounding_boundary")
    prompt = rng.choice([
        "What was the trailing twelve-month dividend yield of %s on %s? Define it as the sum of split-adjusted dividends per share with ex-dividend dates after %s and on or before %s, divided by the split-adjusted close on %s. %s Answer %s." % (
            ent, fmt_date(D, rng), start, D, D, NT_RULE, PCT),
        "Compute %s's trailing 12-month dividend yield as of %s: total split-adjusted dividends per share going ex in the window (%s, %s] over the split-adjusted closing price on that date. %s Give it %s." % (
            ent, fmt_date(D, rng), start, D, NT_RULE, PCT),
    ])
    return mk("dividends", split, d, prompt, [num_key(y, 0.01)],
              "%s TTM dividends %.4f over close %.2f on %s = %.2f%%." % (sym, dv[m].sum(), px.close[i], px.d[i], y),
              [sym], str(dd[m][0]), str(px.d[i]), ["action_range", "sum", "calendar_resolve", "price_lookup", "ratio"],
              int(m.sum()) + 1, "sum(actions.value in (D-1y,D]) / close(D)",
              {"op": "ttm", "symbol": sym, "date": D})


# ============================================================================
# Multi-entity / multi-source

def gen_cross_rank(C, split, rng, d):
    K = {1: 3, 2: rng.randint(4, 5), 3: rng.randint(5, 6)}[d]
    use_eq = rng.random() < 0.7
    if use_eq:
        syms = rng.sample(C.eq[split], K)
        lo, hi = GLOBAL_START, CUTOFF
    else:
        syms = rng.sample(US_INDICES + C.etf, K)
        lo, hi = SPLIT_WINDOWS[split]
    for s in syms:
        C.name_ok(s, d)
    metric = rng.choice(["total", "price", "vol"])
    if metric == "total" and any(C.kind(s) == "index" for s in syms):
        metric = "price"
    direction = rng.choice(["highest", "lowest"])
    L = rng.randint(*{1: (60, 250), 2: (120, 750), 3: (250, 1250)}[d])
    ref = C.S.px["^GSPC"]
    i0, i1 = pick_span(rng, ref, max(lo, max(C.S.px[s].d[0] for s in syms)), hi, L, lookback=1)
    S0, E0 = str(ref.d[i0]), str(ref.d[i1])
    vals = []
    rows = 0
    for s in syms:
        px = C.S.px[s]
        a, b = px.i_le(S0), px.i_le(E0)
        if a < 1 or px.d[a] != S0 or px.d[b] != E0:
            raise Skip("missing_row")
        if not px.clean(a, b):
            raise Skip("bad_row")
        if metric == "vol":
            r = px.close[a + 1:b + 1] / px.close[a:b] - 1
            v = float(np.std(r, ddof=1) * np.sqrt(252) * 100)
        else:
            c = px.adj_close if metric == "total" else px.close
            v = float((c[b] / c[a] - 1) * 100)
        vals.append(v)
        rows += b - a + 1
    vals = np.array(vals)
    order = np.argsort(-vals) if direction == "highest" else np.argsort(vals)
    if abs(vals[order[0]] - vals[order[1]]) < 0.02:
        raise Skip("near_tie")
    w = syms[int(order[0])]
    if not round_ok(vals[order[0]], 0.01):
        raise Skip("rounding_boundary")
    names = ", ".join(C.ent(s, d) for s in syms)
    mdesc = {"total": TOTAL_DESC, "price": PRICE_DESC,
             "vol": ("annualized volatility (sample standard deviation, n-1, of daily close-to-close returns "
                     "on split-adjusted closes from the start date's close through the end date's close, times "
                     "the square root of 252)")}[metric]
    prompt = rng.choice([
        "Among %s, which had the %s %s from the close on %s to the close on %s? Name it and give its value %s." % (
            names, direction, mdesc, fmt_date(S0, rng), fmt_date(E0, rng), PCT),
        "Compare the %s of these instruments between %s and %s (close to close): %s. Which one was %s, and what was its value, %s?" % (
            mdesc, fmt_date(S0, rng), fmt_date(E0, rng), names, direction, PCT),
    ])
    if metric != "vol":
        prompt = prompt.replace(PCT, PCT_SIGNED)
    return mk("cross_rank", split, d, prompt, [C.ticker_key(w), num_key(vals[order[0]], 0.01)],
              "%s: %s (%s); all: %s." % (direction, w, round(vals[order[0]], 2),
                                         ", ".join("%s %.2f" % (s, v) for s, v in zip(syms, vals))),
              syms, S0, E0, ["price_range"] * K + (["daily_return", "std"] if metric == "vol" else ["pct_change"]) + ["argmax"],
              rows if metric == "vol" else 2 * K, "%s per symbol, %s" % (metric, direction),
              {"symbols": syms, "start": S0, "end": E0, "metric": metric, "direction": direction})


def gen_fx_index_return(C, split, rng, d):
    sym = rng.choice(sorted(FX_FOR_INDEX))
    sid, per_usd, qdesc = FX_FOR_INDEX[sym]
    lo, hi = SPLIT_WINDOWS[split]
    px = C.S.px[sym]
    L = rng.randint(*{1: (60, 250), 2: (120, 750), 3: (250, 1500)}[d])
    i0, i1 = pick_span(rng, px, add_days(lo, 7), hi, L)
    if not px.clean(i0, i1):
        raise Skip("bad_row")
    p_cal = {1: 0.0, 2: 0.5, 3: 0.5}[d]
    sD, nt0 = cal_date_near(rng, px, i0, p_cal)
    eD, nt1 = cal_date_near(rng, px, i1, p_cal)
    need_rule(nt0 or nt1, d >= 2)
    F = C.S.fred[sid]
    fd0, f0 = F.le(str(px.d[i0]))
    fd1, f1 = F.le(str(px.d[i1]))
    if (np_date(px.d[i0]) - np_date(fd0)).days > 7 or (np_date(px.d[i1]) - np_date(fd1)).days > 7:
        raise Skip("missing_row")
    if fd0 < lo:
        raise Skip("outside_split_window")
    u0 = px.close[i0] / f0 if per_usd else px.close[i0] * f0
    u1 = px.close[i1] / f1 if per_usd else px.close[i1] * f1
    usd = (u1 / u0 - 1) * 100
    loc = (px.close[i1] / px.close[i0] - 1) * 100
    if abs(usd - loc) < 0.05:
        raise Skip("fx_irrelevant")
    keys = [num_key(usd, 0.01)]
    if not round_ok(usd, 0.01):
        raise Skip("rounding_boundary")
    if d == 3:
        if not round_ok(loc, 0.01):
            raise Skip("rounding_boundary")
        keys.insert(0, num_key(loc, 0.01))
    ent = C.ent(sym, d)
    rule = ("%s For the exchange rate, use the FRED %s observation on the same date as the index close, or "
            "the most recent earlier observation if that date has none." % ((" " + NT_RULE) if d >= 2 else "", sid))
    both = " Give the return in %s first, then in US dollars." % LOCAL_CCY[sym] if d == 3 else ""
    prompt = rng.choice([
        "What was the US-dollar price return of %s from the close on %s to the close on %s? Convert each closing level to US dollars with FRED series %s (%s).%s%s Answer %s." % (
            ent, fmt_date(sD, rng), fmt_date(eD, rng), sid, qdesc, rule, both, PCT_SIGNED),
        "An American investor tracks %s. Using FRED %s (%s) to translate the index into US dollars, compute the return between the closes of %s and %s.%s%s Report %s." % (
            ent, sid, qdesc, fmt_date(sD, rng), fmt_date(eD, rng), rule, both, PCT_SIGNED),
    ])
    return mk("fx_index_return", split, d, prompt, keys,
              "%s %s->%s local %.2f%%, FX %s %s->%s, USD return %.2f%%." % (sym, px.d[i0], px.d[i1], loc, sid, f0, f1, usd),
              [sym, sid], min(fd0, str(px.d[i0])), str(px.d[i1]),
              ["calendar_resolve", "price_lookup", "fred_lookup", "fx_convert", "pct_change"], 4,
              "(idx1 fx-converted)/(idx0 fx-converted)-1",
              {"symbol": sym, "series": sid, "per_usd": per_usd, "start": sD, "end": eD})


def gen_real_return(C, split, rng, d):
    sym, lo, hi = pick_entity(C, rng, split, 0.5, ("us",), needs_time_window=True)
    C.name_ok(sym, d)
    px = C.S.px[sym]
    hi = min(hi, "2025-09-30")
    nm = rng.randint(*{1: (12, 36), 2: (12, 60), 3: (24, 120)}[d])
    a, b = px.i_ge(lo), px.i_le(hi)
    ends = month_ends(px, a, b)
    if len(ends) < nm + 2:
        raise Skip("short_history")
    k = rng.randint(1, len(ends) - nm - 1)
    i0, i1 = int(ends[k]), int(ends[k + nm])
    m0, m1 = px.d[i0][:7], px.d[i1][:7]
    if m0 + "-01" < lo:
        raise Skip("outside_split_window")
    if not px.clean(i0, i1):
        raise Skip("bad_row")
    total = px.kind != "index"
    c = px.adj_close if total else px.close
    nom = c[i1] / c[i0] - 1
    cpi0, cpi0_latest = C.S.fred["CPIAUCSL"].pair(m0 + "-01")
    cpi1, cpi1_latest = C.S.fred["CPIAUCSL"].pair(m1 + "-01")
    infl = cpi1 / cpi0 - 1
    real = ((1 + nom) / (1 + infl) - 1) * 100
    infl_l = cpi1_latest / cpi0_latest - 1
    real_l = ((1 + nom) / (1 + infl_l) - 1) * 100
    nom *= 100
    infl *= 100
    infl_l *= 100
    keys = [num_key(real, 0.01)]
    for v in [real] + ([nom] if d >= 2 else []) + ([infl] if d == 3 else []):
        if not round_ok(v, 0.01):
            raise Skip("rounding_boundary")
    # CPI is revised after first release; the answer uses the vintage published by the cutoff and the
    # prompt names that vintage whenever the latest revision would round differently.
    vintage = round(real, 2) != round(real_l, 2) or (d == 3 and round(infl, 2) != round(infl_l, 2))
    if d >= 2:
        keys.insert(0, num_key(nom, 0.01))
    if d == 3:
        keys.insert(1, num_key(infl, 0.01))
    ent = C.ent(sym, d)
    nd = TOTAL_DESC if total else "price return of the index level"
    ask = {1: "the inflation-adjusted (real) return",
           2: "the nominal return and the inflation-adjusted (real) return",
           3: "the nominal return, the CPI inflation over the same months, and the inflation-adjusted (real) return"}[d]
    defn = ("Nominal return: %s from the close on the last trading day of %s to the close on the last trading day "
            "of %s. Inflation: CPIAUCSL (FRED, monthly) for %s versus %s. Real return = (1 + nominal) / (1 + inflation) - 1." % (
                nd, month_label(m0), month_label(m1), month_label(m1), month_label(m0)))
    prompt = rng.choice([
        "What were %s of %s from the end of %s to the end of %s? %s Give each %s." % (ask, ent, month_label(m0), month_label(m1), defn, PCT_SIGNED),
        "%s Using these definitions, report %s for %s between %s and %s, %s." % (defn, ask, ent, month_label(m0), month_label(m1), PCT_SIGNED),
    ])
    if d == 1:
        prompt = prompt.replace("What were", "What was").replace("Give each", "Give it")
    if vintage:
        prompt += (" Use the CPIAUCSL values as published in the FRED vintage of 2025-10-30 (the snapshot's "
                   "as-of-cutoff values, before later revisions).")
    return mk("real_return", split, d, prompt, keys,
              "%s nominal %.2f%% (%s->%s), CPI %s->%s inflation %.2f%%, real %.2f%%." % (sym, nom, px.d[i0], px.d[i1], cpi0, cpi1, infl, real),
              [sym, "CPIAUCSL"], m0 + "-01", str(px.d[i1]),
              ["month_end_resolve", "price_lookup", "fred_lookup", "deflate"], 4,
              "(1+nominal)/(CPI1/CPI0)-1", {"symbol": sym, "m0": m0, "m1": m1, "total": total, "vintage": vintage})


RATIOS = {
    "net_margin": ("net profit margin", "net income (us-gaap:NetIncomeLoss) divided by revenue, where revenue is us-gaap:Revenues if that tag is reported for the year in the filing, otherwise us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax, otherwise us-gaap:SalesRevenueNet", True),
    "roe": ("return on average equity", "net income (us-gaap:NetIncomeLoss) divided by the average of total stockholders' equity (us-gaap:StockholdersEquity) at the start and end of the fiscal year, both as presented in that year's 10-K balance sheet", True),
    "inv_turnover": ("inventory turnover", "cost of sales divided by the average of net inventory (us-gaap:InventoryNet) at the start and end of the fiscal year (both from that year's 10-K balance sheet), where cost of sales is us-gaap:CostOfGoodsAndServicesSold if reported, otherwise us-gaap:CostOfRevenue, otherwise us-gaap:CostOfGoodsSold", False),
}


def ratio_value(X, cik, fy, ratio):
    rows = 0
    if ratio == "net_margin":
        ni, _, _ = X.value(cik, fy, "NetIncomeLoss", "dur")
        rev = None
        for t in ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]:
            try:
                rev, _, _ = X.value(cik, fy, t, "dur")
                break
            except Skip as e:
                if str(e) != "missing_row":
                    raise
        if rev is None or rev <= 0:
            raise Skip("missing_row")
        return ni / rev * 100, 2
    if ratio == "roe":
        ni, _, _ = X.value(cik, fy, "NetIncomeLoss", "dur")
        e1, _, _ = X.value(cik, fy, "StockholdersEquity", "inst", "current")
        e0, _, _ = X.value(cik, fy, "StockholdersEquity", "inst", "prior")
        if e0 <= 0 or e1 <= 0:
            raise Skip("degenerate_value")
        return ni / ((e0 + e1) / 2) * 100, 3
    cogs = None
    for t in COGS_PREC:
        try:
            cogs, _, _ = X.value(cik, fy, t, "dur")
            break
        except Skip as e:
            if str(e) != "missing_row":
                raise
    if cogs is None or cogs <= 0:
        raise Skip("missing_row")
    i1, _, _ = X.value(cik, fy, "InventoryNet", "inst", "current")
    i0, _, _ = X.value(cik, fy, "InventoryNet", "inst", "prior")
    if i0 <= 0 or i1 <= 0:
        raise Skip("degenerate_value")
    return cogs / ((i0 + i1) / 2), 3


def gen_xbrl_ratio(C, split, rng, d):
    K = {1: 2, 2: rng.choice([2, 3]), 3: 3}[d]
    ny = {1: 2, 2: rng.choice([2, 3]), 3: 3}[d]
    ratio = rng.choice(sorted(RATIOS))
    pool = C.sec_eq[split]
    first = rng.choice(pool)
    same = [s for s in pool if C.sector.get(s) == C.sector.get(first) and s != first]
    others = same if len(same) >= K - 1 and rng.random() < 0.7 else [s for s in pool if s != first]
    syms = [first] + rng.sample(others, K - 1)
    for s in syms:
        C.name_ok(s, d)
    common_years = None
    for s in syms:
        ys = set(C.X.years(C.cik[s]))
        common_years = ys if common_years is None else common_years & ys
    cands = sorted(y for y in common_years if all(y - k in common_years for k in range(ny)))
    if not cands:
        raise Skip("sec_no_fiscal_year")
    last = rng.choice(cands)
    years = list(range(last - ny + 1, last + 1))
    avg = []
    per = {}
    rows = 0
    for s in syms:
        vs = []
        for y in years:
            v, r = ratio_value(C.X, C.cik[s], y, ratio)
            vs.append(v)
            rows += r
        per[s] = vs
        avg.append(float(np.mean(vs)))
    avg = np.array(avg)
    direction = rng.choice(["highest", "lowest"])
    order = np.argsort(-avg) if direction == "highest" else np.argsort(avg)
    if abs(avg[order[0]] - avg[order[1]]) < 0.02:
        raise Skip("near_tie")
    if not round_ok(avg[order[0]], 0.01):
        raise Skip("rounding_boundary")
    w = syms[int(order[0])]
    title, defn, pct = RATIOS[ratio]
    unit_txt = PCT_SIGNED if pct else "as a multiple (times) rounded to two decimals"
    names = ", ".join(C.ent(s, d) for s in syms)
    ytxt = "fiscal years %d through %d" % (years[0], years[-1]) if ny == 3 else "fiscal years %d and %d" % (years[0], years[-1])
    ends = ""
    if d < 3:
        ends = " Fiscal years are as each company labels them: " + "; ".join(
            "%s %s" % (s, ", ".join("FY%d ended %s" % (y, C.X.info(C.cik[s], y)["E"]) for y in years)) for s in syms) + "."
    prompt = rng.choice([
        "Compute the %s of %s for each of %s, where the %s is %s. Use each fiscal year's original annual report on Form 10-K (SEC XBRL data).%s Which company had the %s average %s across these years (simple mean of the yearly values), and what was that average, %s?" % (
            title, names, ytxt, title, defn, ends, direction, title, unit_txt),
        "For %s, calculate %s (%s) in %s from each year's original Form 10-K.%s Average each company's yearly values. Which company has the %s average, and what is it (%s)?" % (
            names, title, defn, ytxt, ends, direction, unit_txt),
    ])
    return mk("xbrl_ratio", split, d, prompt, [C.ticker_key(w), num_key(avg[order[0]], 0.01)],
              "%s average %s: %s; " % (direction, title, w) + "; ".join(
                  "%s %s avg %.2f" % (s, [round(v, 3) for v in per[s]], a) for s, a in zip(syms, avg)),
              syms + ["CIK:%d" % C.cik[s] for s in syms],
              min(C.X.info(C.cik[s], years[0])["S"] for s in syms),
              max(C.X.info(C.cik[s], years[-1])["E"] for s in syms),
              ["sec_fiscal_year_resolve", "sec_fact_lookup", "tag_fallback", "ratio", "mean", "argmax"], rows,
              "%s per company-year from original 10-K, mean over years, %s" % (ratio, direction),
              {"symbols": syms, "ciks": [C.cik[s] for s in syms], "years": years, "ratio": ratio,
               "direction": direction})


def gen_market_cap(C, split, rng, d):
    sym = rng.choice(C.sec_eq[split])
    C.name_ok(sym, d)
    cik = C.cik[sym]
    px = C.S.px[sym]
    a = px.i_ge("2010-01-01")
    b = px.i_le(CUTOFF)
    if b <= a:
        raise Skip("short_history")
    i = rng.randint(a, b)
    D, nt = cal_date_near(rng, px, i, {1: 0.0, 2: 0.5, 3: 0.5}[d])
    need_rule(nt, d >= 2)
    if px.bad[i]:
        raise Skip("bad_row")
    sh, cover, filed, accn = C.X.shares(cik, D)
    if sh % 1e6 == 0:
        raise Skip("rounded_share_count")  # e.g. a cover page stating "approximately 1.0 billion shares"
    if (np_date(D) - np_date(cover)).days > 200:
        raise Skip("stale_shares")
    if any(cover < x[0] <= str(px.d[i]) for x in C.S.splits.get(sym, [])):
        raise Skip("split_between_cover_and_date")
    mcap = px.raw_close[i] * sh / 1e9
    keys = [num_key(mcap, 0.01)]
    if not round_ok(mcap, 0.01):
        raise Skip("rounding_boundary")
    if d == 3:
        shm = sh / 1e6
        if not round_ok(shm, 0.1):
            raise Skip("rounding_boundary")
        keys.insert(0, num_key(shm, 0.1))
    ent = C.ent(sym, d)
    rule = ("Market capitalization = the unadjusted closing price on %s%s times dei:EntityCommonStockSharesOutstanding "
            "(cover-page shares outstanding) from the company's latest SEC filing filed on or before %s." % (
                D, " (or the last trading day on or before it)" if d >= 2 else "", D))
    ask = ("the shares-outstanding figure used, in millions of shares rounded to one decimal, and the market "
           "capitalization" if d == 3 else "the market capitalization")
    prompt = rng.choice([
        "What was the market capitalization of %s on %s? %s Report %s in billions of US dollars, rounded to two decimals." % (ent, fmt_date(D, rng), rule, ask),
        "%s Using that rule, give %s for %s as of %s, in billions of US dollars with two decimals." % (rule, ask, ent, fmt_date(D, rng)),
    ])
    return mk("market_cap", split, d, prompt, keys,
              # the shares in millions as the prompt asks for them (with d == 3 they are a key), then the raw count
              "%s raw close %.2f on %s x %.1f million shares (%d; cover %s, filed %s) = $%.2f bn." % (
                  sym, px.raw_close[i], px.d[i], sh / 1e6, sh, cover, filed, mcap),
              [sym, "CIK:%d" % cik], cover, str(px.d[i]), ["calendar_resolve", "price_lookup", "sec_latest_filing", "multiply", "scale_convert"],
              2, "raw_close(%s) x shares(accn %s)" % (px.d[i], accn), {"symbol": sym, "cik": cik, "date": D, "ask_shares": d == 3})


def gen_two_hop(C, split, rng, d):
    chain = rng.choice(["mdd_dgs10", "worstday_vix", "bestmonth_unrate", "peak_dff"])
    sym, lo, hi = pick_entity(C, rng, split, 0.5, ("us",), needs_time_window=True)
    if sym == "^VIX":
        raise Skip("duplicate")
    C.name_ok(sym, d)
    px = C.S.px[sym]
    L = rng.randint(*{1: (60, 250), 2: (250, 750), 3: (500, 1500)}[d])
    hi = min(hi, "2025-09-30")
    i0, i1 = pick_span(rng, px, lo, hi, L, lookback=1)
    if not px.clean(i0 - 1, i1) or px.close[i0:i1 + 1].min() <= 0:
        raise Skip("bad_row")
    s, e = str(px.d[i0]), str(px.d[i1])
    ent = C.ent(sym, d)
    adj = "split-adjusted " if px.kind in ("equity", "etf") else ""
    if chain == "mdd_dgs10":
        c = px.close[i0:i1 + 1]
        dd, _ = drawdown(c)
        k = int(np.argmin(dd))
        if (np.abs(dd - dd[k]) < 2e-4).sum() > 1 or dd[k] > -0.01:
            raise Skip("near_tie")
        hop = str(px.d[i0 + k])
        od, v = C.S.fred["DGS10"].le(hop)
        u = 0.01
        q1 = "find the trough date of its maximum drawdown (largest decline from a running peak of %sdaily closes) over %s to %s" % (adj, s, e)
        q2 = "the 10-year Treasury constant-maturity yield (FRED DGS10) on that date (most recent earlier observation if none that day), in percent with two decimals"
        k_hop = date_key(hop)
    elif chain == "worstday_vix":
        r = daily_ret(px, i0, i1)
        o = np.argsort(r)
        if r[o[1]] - r[o[0]] < 2e-4:
            raise Skip("near_tie")
        hop = str(px.d[i0 + o[0]])
        vx = C.S.px["^VIX"]
        if not vx.is_td(hop):
            raise Skip("missing_row")
        v = float(vx.close[vx.i_le(hop)])
        od = hop
        u = 0.01
        q1 = "find the trading day with its largest one-day percentage decline (%sclose versus previous close) between %s and %s" % (adj, s, e)
        q2 = "the closing level of the Cboe Volatility Index (Yahoo symbol ^VIX) on that day, rounded to two decimals"
        k_hop = date_key(hop)
    elif chain == "bestmonth_unrate":
        ends = month_ends(px, i0 - 1, i1)
        if len(ends) < 6:
            raise Skip("short_history")
        r = px.close[ends[1:]] / px.close[ends[:-1]] - 1
        o = np.argsort(-r)
        if r[o[0]] - r[o[1]] < 2e-4:
            raise Skip("near_tie")
        ym = px.d[ends[1 + o[0]]][:7]
        od = ym + "-01"
        v = C.S.fred["UNRATE"].get(od)
        u = 0.1
        s, e = px.d[ends[1]][:7], px.d[ends[-1]][:7]
        if px.d[ends[0]][:7] == s:
            raise Skip("short_history")
        q1 = ("find the calendar month with its highest monthly return (last %sclose of the month versus the last "
              "close of the previous month) among the months %s through %s" % (adj, month_label(s), month_label(e)))
        q2 = "the US unemployment rate (FRED UNRATE) for that month, in percent with one decimal"
        k_hop = text_key(month_label(ym), month_aliases(ym))
        hop = ym
    else:
        c = px.close[i0:i1 + 1]
        o = np.argsort(-c)
        if c[o[0]] / c[o[1]] - 1 < 2e-4:
            raise Skip("near_tie")
        hop = str(px.d[i0 + o[0]])
        od, v = C.S.fred["DFF"].le(hop)
        u = 0.01
        q1 = "find the date of its highest %sclose between %s and %s" % (adj, s, e)
        q2 = "the daily effective federal funds rate (FRED DFF) on that date, in percent with two decimals"
        k_hop = date_key(hop)
    if od < lo:
        raise Skip("outside_split_window")
    if not round_ok(v, u):
        raise Skip("rounding_boundary")
    prompt = rng.choice([
        "For %s, first %s. Then report %s. Give both the %s and the value." % (ent, q1, q2, "month" if chain == "bestmonth_unrate" else "date (YYYY-MM-DD)"),
        "Two steps: (1) for %s, %s; (2) look up %s. Answer with the %s from step 1 and the value from step 2." % (ent, q1, q2, "month" if chain == "bestmonth_unrate" else "date"),
    ])
    return mk("two_hop", split, d, prompt, [k_hop, num_key(v, u)],
              "Step 1 (%s): %s; step 2: %s." % (chain, hop, v),
              [sym, {"mdd_dgs10": "DGS10", "worstday_vix": "^VIX", "bestmonth_unrate": "UNRATE", "peak_dff": "DFF"}[chain]],
              str(px.d[i0 - 1]), str(px.d[i1]), ["price_range", chain.split("_")[0], "fred_lookup" if chain != "worstday_vix" else "price_lookup"],
              i1 - i0 + 3, chain, {"chain": chain, "symbol": sym, "start": str(px.d[i0]), "end": str(px.d[i1]),
                                   "first_month": s if chain == "bestmonth_unrate" else None,
                                   "last_month": e if chain == "bestmonth_unrate" else None})


TEMPLATES = {
    "price_lookup": gen_price_lookup,
    "fred_lookup": gen_fred_lookup,
    "xbrl_lookup": gen_xbrl_lookup,
    "period_return": gen_period_return,
    "nth_day_move": gen_nth_day_move,
    "extreme_period": gen_extreme_period,
    "threshold_scan": gen_threshold_scan,
    "max_drawdown": gen_max_drawdown,
    "rolling_vol": gen_rolling_vol,
    "split_event": gen_split_event,
    "dividends": gen_dividends,
    "cross_rank": gen_cross_rank,
    "fx_index_return": gen_fx_index_return,
    "real_return": gen_real_return,
    "xbrl_ratio": gen_xbrl_ratio,
    "market_cap": gen_market_cap,
    "two_hop": gen_two_hop,
}
