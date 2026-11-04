"""Independent recomputation of task answers straight from SQLite.

This is a second implementation of every template: it reads only the task's
`meta.params`, queries the snapshot with SQL (window functions, aggregates and
row lookups) and recomputes the answer in plain Python, without the pandas
arrays, cached tables or SEC fiscal-year index used by templates.py.
Each function returns (values, evidence): values in the order of the task's
keys (a set key becomes a sorted list), evidence a short text of the rows used.

Version 2 (research/log/2026-10-31-0150.md, the hard tier): adds a verifier for
each template of templates_hard.py, again from params and SQL only (the risk
ratios, rolling betas, moving averages, the reinvestment ledger, the event
windows, Q4 amounts and the month chain are recomputed in plain Python).

Version 3 (research/log/2026-10-31-0150.md, the hardening of five hard templates): the verifiers of rolling_beta,
ma_cross, reinvest_return, div_event_study and q4_margin_rank also recompute the added keys (the mean beta, the next
golden cross and the best round trip, the dividend cash reinvested, the highest event, the runner-up's Q4 margin).
"""
import math
import os
import sqlite3
from datetime import date, timedelta

DB = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1/market_v1.sqlite"


class V:
    def __init__(self, path=DB):
        self.con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)

    def q(self, sql, *p):
        return self.con.execute(sql, p).fetchall()

    def one(self, sql, *p):
        r = self.con.execute(sql, p).fetchone()
        if r is None:
            raise LookupError(sql)
        return r

    # ---------------------------------------------------------------- prices
    def bar_le(self, sym, d, cols="date, open, high, low, close, adj_close, volume, raw_open, raw_high, raw_low, raw_close, raw_volume"):
        return self.one("select %s from prices where symbol=? and date<=? order by date desc limit 1" % cols, sym, d)

    def bar_lt(self, sym, d, cols="date, close"):
        return self.one("select %s from prices where symbol=? and date<? order by date desc limit 1" % cols, sym, d)

    def rng(self, sym, d0, d1, cols="date, close"):
        return self.q("select %s from prices where symbol=? and date between ? and ? order by date" % cols, sym, d0, d1)

    def fred_le(self, sid, d):
        return self.one("select date, value from fred where series_id=? and date<=? and value is not null "
                        "order by date desc limit 1", sid, d)

    def fred_eq(self, sid, d):
        return self.one("select value from fred where series_id=? and date=?", sid, d)[0]

    # ---------------------------------------------------------------- SEC
    def annual_filing(self, cik, fy):
        rows = self.q("select accn, min(filed) from facts where cik=? and form='10-K' and fp='FY' and fy=? "
                      "and taxonomy='us-gaap' group by accn order by 2 limit 1", cik, fy)
        accn, filed = rows[0]
        # Fiscal year end = latest end of a one-year us-gaap period in the filing, dated before filing.
        E = self.one("select max(\"end\") from facts where cik=? and accn=? and taxonomy='us-gaap' "
                     "and start is not null and julianday(\"end\")-julianday(start) between 350 and 380 "
                     "and \"end\"<=?", cik, accn, filed)[0]
        return accn, E

    def sec_val(self, cik, fy, tag, kind, which="current"):
        accn, E = self.annual_filing(cik, fy)
        unit = "USD/shares" if tag.startswith("EarningsPerShare") else "USD"
        if kind == "dur":
            rows = self.q("select distinct val, start from facts where cik=? and tag=? and \"end\"=? and accn=? and unit=? "
                          "and start is not null and julianday(\"end\")-julianday(start) between 350 and 380",
                          cik, tag, E, accn, unit)
        elif which == "current":
            rows = self.q("select distinct val, start from facts where cik=? and tag=? and \"end\"=? and accn=? and unit=? "
                          "and start is null", cik, tag, E, accn, unit)
        else:
            rows = self.q("select distinct val, \"end\" from facts where cik=? and tag=? and accn=? and unit=? and start is null "
                          "and julianday(?)-julianday(\"end\") between 350 and 380", cik, tag, accn, unit, E)
        if len(rows) != 1:
            raise LookupError("sec %s %s %s %s: %d rows" % (cik, fy, tag, which, len(rows)))
        return rows[0][0], E, rows[0][1], accn

    def sec_first(self, cik, fy, tags):
        for t in tags:
            try:
                return self.sec_val(cik, fy, t, "dur")[0]
            except LookupError:
                continue
        raise LookupError("no tag")


def pct(a, b):
    return (a / b - 1.0) * 100.0


def sdev(xs):
    n = len(xs)
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def iso_week(s):
    return date.fromisoformat(s).isocalendar()[:2]


def v_price_lookup(v, p):
    out = []
    bar = dict(zip(["date", "open", "high", "low", "close", "adj_close", "volume", "raw_open", "raw_high",
                    "raw_low", "raw_close", "raw_volume"], v.bar_le(p["symbol"], p["date"])))
    for f in p["fields"]:
        x = bar[("raw_" + f) if p["adj"] == "raw" else f]
        if f == "volume" and p["vol_scale"] == "millions":
            x = x / 1e6
        out.append(float(x))
    return out, "bar %s: %s" % (bar["date"], {f: bar[f] for f in bar if f.endswith(tuple(p["fields"]))})


def v_fred_lookup(v, p):
    op, sid = p["op"], p["series"]
    if op == "daily":
        d, x = v.fred_le(sid, p["date"])
        return [x * 100 if p["scale"] == "bp" else x], "%s %s=%s" % (sid, d, x)
    if op == "chg_bp":
        a = v.fred_le(sid, p["d0"])
        b = v.fred_le(sid, p["d1"])
        return [(b[1] - a[1]) * 100], "%s %s %s" % (sid, a, b)
    if op == "month_avg":
        r = v.one("select avg(value), count(*) from fred where series_id=? and substr(date,1,7)=? and value is not null",
                  sid, p["month"])
        return [r[0]], "%s %s avg of %d" % (sid, p["month"], r[1])
    if op == "fx":
        d, x = v.fred_le(sid, p["date"])
        if p["invert"]:
            scale = {"DEXJPUS": 100.0, "DEXKOUS": 1000.0}.get(sid, 1.0)
            y = scale / x
        else:
            y = x
        return [y], "%s %s=%s" % (sid, d, x)
    if op == "monthly":
        x = v.fred_eq(sid, p["month"] + "-01")
        return [x * p["mult"]], "%s %s=%s" % (sid, p["month"], x)
    if op == "other":
        x = v.fred_eq(sid, p["date"])
        return [x * p["mult"]], "%s %s=%s" % (sid, p["date"], x)
    ym = p["month"]
    y0 = "%04d-%s" % (int(ym[:4]) - 1, ym[5:7])
    a, b = v.fred_eq(sid, y0 + "-01"), v.fred_eq(sid, ym + "-01")
    return [pct(b, a)], "%s %s=%s %s=%s" % (sid, y0, a, ym, b)


def v_xbrl_lookup(v, p):
    cik, fy, tag, kind = p["cik"], p["fy"], p["tag"], p["kind"]
    x, E, _, accn = v.sec_val(cik, fy, tag, kind)
    if p["op"] == "value":
        return [x * p["mult"]], "%s FY%d end %s accn %s val %s" % (tag, fy, E, accn, x)
    if p["op"] in ("yoy", "yoy_pct"):
        x0, E0, _, accn0 = v.sec_val(cik, fy - 1, tag, kind)
        if p["op"] == "yoy":
            return [(x - x0) * p["mult"]], "%s FY%d %s (%s) FY%d %s (%s)" % (tag, fy, x, accn, fy - 1, x0, accn0)
        return [pct(x, x0)], "%s FY%d %s FY%d %s" % (tag, fy, x, fy - 1, x0)
    # q4: annual minus nine-month YTD of the original Q3 10-Q
    _, _, S, _ = v.sec_val(cik, fy, tag, kind)
    r = v.q("select val, filed from facts where cik=? and tag=? and form='10-Q' and fy=? and fp='Q3' and start=? "
            "and julianday(\"end\")-julianday(start) between 250 and 290 and unit='USD' order by filed", cik, tag, fy, S)
    nine = r[0][0]
    return [(x - nine) * 1e-6], "%s FY%d %s minus 9M %s" % (tag, fy, x, nine)


def v_period_return(v, p):
    a = v.bar_le(p["symbol"], p["start"], "date, close, adj_close")
    b = v.bar_le(p["symbol"], p["end"], "date, close, adj_close")
    out = []
    for m in p["measures"]:
        k = 2 if m == "total" else 1
        out.append(pct(b[k], a[k]))
    return out, "start %s end %s" % (a, b)


def v_nth_day_move(v, p):
    rows = v.q("select date, open, close from prices where symbol=? and date>? order by date limit ?",
               p["symbol"], p["date"], p["n"])
    d, o, c = rows[-1]
    if p["mode"] == "cc":
        prev = v.bar_lt(p["symbol"], d)
        val = pct(c, prev[1])
        ev = "prev %s, day %s close %s" % (prev, d, c)
    else:
        val = pct(c, o)
        ev = "day %s open %s close %s" % (d, o, c)
    return [val, d], ev


def v_extreme_period(v, p):
    sym = p["symbol"]
    if p["freq"] == "month":
        m0 = p["first"][:7]
        start = m0 + "-01"
        base = v.bar_lt(sym, start)
        rows = v.rng(sym, start, p["last"])
        last = {}
        for d, c in rows:
            last[d[:7]] = (d, c)
        keys = sorted(last)
        prev = base[1]
        rets = []
        for k in keys:
            rets.append((pct(last[k][1], prev), k))
            prev = last[k][1]
        ev = "base %s, %d months" % (base, len(keys))
    else:
        d0 = date.fromisoformat(p["first"])
        monday = d0 - timedelta(days=d0.weekday())
        base = v.bar_lt(sym, monday.isoformat())
        rows = v.rng(sym, monday.isoformat(), p["last"])
        last = {}
        for d, c in rows:
            last[iso_week(d)] = (d, c)
        keys = sorted(last)
        prev = base[1]
        rets = []
        for k in keys:
            rets.append((pct(last[k][1], prev), last[k][0]))
            prev = last[k][1]
        ev = "base %s, %d weeks" % (base, len(keys))
    best = min(rets) if p["direction"] == "loss" else max(rets)
    return [best[1], best[0]], ev + ", best %s" % (best,)


def v_threshold_scan(v, p):
    sym = p["symbol"]
    prev = v.bar_lt(sym, p["start"])[0]
    col = "raw_close" if p["mode"] == "usd_drop" else "close"
    rows = v.q("select date, chg, ret from (select date, %s - lag(%s) over w as chg, %s / lag(%s) over w - 1 as ret "
               "from prices where symbol=? and date between ? and ? window w as (order by date)) where date>=?"
               % (col, col, col, col), sym, prev, p["end"], p["start"])
    X = p["X"]
    if p["mode"] == "usd_drop":
        hits = [d for d, c, r in rows if c < -X]
    elif p["mode"] == "pct_drop":
        hits = [d for d, c, r in rows if r * 100 < -X]
    else:
        hits = [d for d, c, r in rows if r * 100 > X]
    out = [float(len(hits))]
    if p["list"]:
        out.append(sorted(hits))
    return out, "%d rows scanned, hits %s" % (len(rows), hits[:8])


def v_max_drawdown(v, p):
    col = "adj_close" if p["measure"] == "total" else "close"
    rows = v.rng(p["symbol"], p["start"], p["end"], "date, " + col)
    peak, peak_d, best, trough, best_peak = -1, None, 0.0, None, None
    for d, c in rows:
        if c > peak:
            peak, peak_d = c, d
        dd = c / peak - 1
        if dd < best:
            best, trough, best_peak = dd, d, peak_d
    out = [-best * 100, trough]
    if p["with_peak"]:
        out.append(best_peak)
    return out, "%d rows; peak %s trough %s" % (len(rows), best_peak, trough)


def v_rolling_vol(v, p):
    syms = p["symbols"]
    series = []
    for s in syms:
        n_in = v.one("select count(*) from prices where symbol=? and date between ? and ?", s, p["start"], p["end"])[0]
        rows = v.q("select date, close from prices where symbol=? and date<=? order by date desc limit ?",
                   s, p["end"], n_in + 20)[::-1]
        series.append(rows)
    rets = []
    for rows in series:
        rets.append([(rows[k][0], rows[k][1] / rows[k - 1][1] - 1) for k in range(1, len(rows))])
    if len(syms) == 2:
        assert [a[0] for a in rets[0]] == [b[0] for b in rets[1]]
        r = [(a[0], (a[1] - b[1]) * 100) for a, b in zip(rets[0], rets[1])]
    else:
        r = [(a[0], a[1] * 100) for a in rets[0]]
    sds = []
    for t in range(19, len(r)):
        if r[t][0] >= p["start"]:
            sds.append(sdev([x[1] for x in r[t - 19:t + 1]]))
    return [sum(sds) / len(sds)], "%d windows from %s" % (len(sds), r[19][0])


def v_split_event(v, p):
    sym, E = p["symbol"], p["effective"]
    ratio = v.one("select value from actions where symbol=? and date=? and kind='split'", sym, E)[0]
    o = v.one("select raw_open from prices where symbol=? and date=?", sym, E)[0]
    prev = v.bar_lt(sym, E, "date, raw_close")
    out = [ratio, o - prev[1]]
    if p["want_pr"]:
        out.append(prev[1] / o)
    if p["ask_date"]:
        out = [E] + out
    return out, "split %s ratio %s, raw open %s, prev %s" % (E, ratio, o, prev)


def v_dividends(v, p):
    sym = p["symbol"]
    if p["op"] in ("year_raw", "year_adj", "year_count"):
        col = "value" if p["op"] == "year_adj" else "value_raw"
        n, tot = v.one("select count(*), sum(%s) from actions where symbol=? and kind='dividend' and date like ?" % col,
                       sym, "%d-%%" % p["year"])
        out = [tot] if p["op"] != "year_count" else [float(n), tot]
        return out, "%d dividends, sum %s" % (n, tot)
    D = p["date"]
    d = date.fromisoformat(D)
    try:
        start = d.replace(year=d.year - 1).isoformat()
    except ValueError:
        start = d.replace(year=d.year - 1, day=28).isoformat()
    tot = v.one("select sum(value) from actions where symbol=? and kind='dividend' and date>? and date<=?", sym, start, D)[0]
    bar = v.bar_le(sym, D, "date, close")
    return [tot / bar[1] * 100], "divs %s over close %s" % (tot, bar)


def v_cross_rank(v, p):
    vals = []
    for s in p["symbols"]:
        if p["metric"] == "vol":
            rows = v.q("select r from (select date, close / lag(close) over (order by date) - 1 as r from prices "
                       "where symbol=? and date between ? and ?) where date>?", s, p["start"], p["end"], p["start"])
            vals.append(sdev([r[0] for r in rows]) * math.sqrt(252) * 100)
        else:
            col = "adj_close" if p["metric"] == "total" else "close"
            a = v.one("select %s from prices where symbol=? and date=?" % col, s, p["start"])[0]
            b = v.one("select %s from prices where symbol=? and date=?" % col, s, p["end"])[0]
            vals.append(pct(b, a))
    pick = max if p["direction"] == "highest" else min
    best = pick(zip(vals, p["symbols"]))
    return [best[1], best[0]], "values %s" % ["%s %.3f" % (s, x) for x, s in zip(vals, p["symbols"])]


def v_fx_index_return(v, p):
    sym, sid = p["symbol"], p["series"]
    a = v.bar_le(sym, p["start"], "date, close")
    b = v.bar_le(sym, p["end"], "date, close")
    fa, fb = v.fred_le(sid, a[0]), v.fred_le(sid, b[0])
    conv = (lambda x, f: x / f) if p["per_usd"] else (lambda x, f: x * f)
    usd = pct(conv(b[1], fb[1]), conv(a[1], fa[1]))
    loc = pct(b[1], a[1])
    return [loc, usd], "idx %s %s fx %s %s" % (a, b, fa, fb)


def v_real_return(v, p):
    sym = p["symbol"]
    col = "adj_close" if p["total"] else "close"
    out = []
    a = v.one("select date, %s from prices where symbol=? and substr(date,1,7)=? order by date desc limit 1" % col, sym, p["m0"])
    b = v.one("select date, %s from prices where symbol=? and substr(date,1,7)=? order by date desc limit 1" % col, sym, p["m1"])
    col = "value_vintage" if p.get("vintage") else "value"
    c0 = v.one("select %s from fred where series_id='CPIAUCSL' and date=?" % col, p["m0"] + "-01")[0]
    c1 = v.one("select %s from fred where series_id='CPIAUCSL' and date=?" % col, p["m1"] + "-01")[0]
    nom = b[1] / a[1]
    infl = c1 / c0
    return [(nom - 1) * 100, (infl - 1) * 100, (nom / infl - 1) * 100], "px %s %s cpi %s %s" % (a, b, c0, c1)


def v_ratio(v, cik, fy, ratio):
    if ratio == "net_margin":
        ni = v.sec_val(cik, fy, "NetIncomeLoss", "dur")[0]
        rev = v.sec_first(cik, fy, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"])
        return ni / rev * 100
    if ratio == "roe":
        ni = v.sec_val(cik, fy, "NetIncomeLoss", "dur")[0]
        e1 = v.sec_val(cik, fy, "StockholdersEquity", "inst")[0]
        e0 = v.sec_val(cik, fy, "StockholdersEquity", "inst", "prior")[0]
        return ni / ((e0 + e1) / 2) * 100
    cogs = v.sec_first(cik, fy, ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"])
    i1 = v.sec_val(cik, fy, "InventoryNet", "inst")[0]
    i0 = v.sec_val(cik, fy, "InventoryNet", "inst", "prior")[0]
    return cogs / ((i0 + i1) / 2)


def v_xbrl_ratio(v, p):
    avgs = []
    for s, cik in zip(p["symbols"], p["ciks"]):
        xs = [v_ratio(v, cik, y, p["ratio"]) for y in p["years"]]
        avgs.append((sum(xs) / len(xs), s))
    best = max(avgs) if p["direction"] == "highest" else min(avgs)
    return [best[1], best[0]], "averages %s" % ["%s %.4f" % (s, a) for a, s in avgs]


def v_market_cap(v, p):
    bar = v.bar_le(p["symbol"], p["date"], "date, raw_close")
    rows = v.q("select val, \"end\", filed from facts where cik=? and tag='EntityCommonStockSharesOutstanding' "
               "and unit='shares' and filed<=? order by filed desc, val", p["cik"], p["date"])
    last = rows[0][2]
    vals = sorted(set(r[0] for r in rows if r[2] == last))
    assert len(vals) == 1
    sh = vals[0]
    out = [bar[1] * sh / 1e9]
    if p["ask_shares"]:
        out = [sh / 1e6] + out
    return out, "close %s, shares %s filed %s" % (bar, sh, last)


def v_two_hop(v, p):
    sym, ch = p["symbol"], p["chain"]
    if ch == "mdd_dgs10":
        rows = v.rng(sym, p["start"], p["end"])
        peak, best, hop = -1, 0, None
        for d, c in rows:
            peak = max(peak, c)
            if c / peak - 1 < best:
                best, hop = c / peak - 1, d
        val = v.fred_le("DGS10", hop)
        return [hop, val[1]], "trough %s, DGS10 %s" % (hop, val)
    if ch == "worstday_vix":
        rows = v.q("select date, r from (select date, close / lag(close) over (order by date) - 1 as r from prices "
                   "where symbol=? and date between ? and ?) where date>=? order by r limit 1",
                   sym, v.bar_lt(sym, p["start"])[0], p["end"], p["start"])
        hop = rows[0][0]
        vix = v.one("select close from prices where symbol='^VIX' and date=?", hop)[0]
        return [hop, vix], "worst day %s, VIX %s" % (rows[0], vix)
    if ch == "bestmonth_unrate":
        best = None
        y, m = int(p["first_month"][:4]), int(p["first_month"][5:7])
        while "%04d-%02d" % (y, m) <= p["last_month"]:
            ym = "%04d-%02d" % (y, m)
            c1 = v.one("select close from prices where symbol=? and substr(date,1,7)=? order by date desc limit 1", sym, ym)[0]
            c0 = v.bar_lt(sym, ym + "-01")[1]
            if best is None or c1 / c0 - 1 > best[0]:
                best = (c1 / c0 - 1, ym)
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        un = v.fred_eq("UNRATE", best[1] + "-01")
        return [best[1], un], "best month %s, UNRATE %s" % (best, un)
    rows = v.q("select date, close from prices where symbol=? and date between ? and ? order by close desc limit 1",
               sym, p["start"], p["end"])
    hop = rows[0][0]
    val = v.fred_le("DFF", hop)
    return [hop, val[1]], "peak %s, DFF %s" % (rows[0], val)



# ---------------------------------------------------------------------------- hard tier

REV_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
FX_PER_USD = {"DEXJPUS": True, "DEXSZUS": True, "DEXCAUS": True, "DEXUSEU": False, "DEXUSUK": False, "DEXUSAL": False}


def stepwise(obs, days):
    """For each day, the value of the last observation dated on or before it (obs sorted by date)."""
    out, k, last = [], 0, None
    for t in days:
        while k < len(obs) and obs[k][0] <= t:
            last = obs[k][1]
            k += 1
        out.append(last)
    return out


def v_risk_rank(v, p):
    s0 = v.bar_le("^GSPC", p["start"], "date, close")[0]
    e0 = v.bar_le("^GSPC", p["end"], "date, close")[0]
    days = [r[0] for r in v.q("select date from prices where symbol='^GSPC' and date>? and date<=? order by date", s0, e0)]
    rf = [0.0] * len(days)
    if p["rf"]:
        obs = v.q("select date, value from fred where series_id='DTB3' and date<=? and value is not null order by date", e0)
        rf = [x / 100.0 / 252.0 for x in stepwise(obs, days)]
    vals = []
    for s in p["symbols"]:
        rows = v.q("select date, r from (select date, adj_close / lag(adj_close) over (order by date) - 1 as r from prices "
                   "where symbol=? and date between ? and ?) where date>? order by date", s, s0, e0, s0)
        assert [r[0] for r in rows] == days, s
        e = [r[1] - f for r, f in zip(rows, rf)]
        m = sum(e) / len(e)
        if p["metric"] == "sharpe":
            x = m / sdev(e) * math.sqrt(252)
        else:
            x = m / math.sqrt(sum(min(y, 0.0) ** 2 for y in e) / len(e)) * math.sqrt(252)
        vals.append((x, s))
    hi, lo = max(vals), min(vals)
    out = {"highest": [hi[1], hi[0]], "lowest": [lo[1], lo[0]], "both": [hi[1], hi[0], lo[1], lo[0]]}[p["ask"]]
    return out, "%s..%s, %d days; %s" % (s0, e0, len(days), ["%s %.3f" % (s, x) for x, s in vals])


def v_rolling_beta(v, p):
    col, W = ("adj_close" if p["total"] else "close"), p["window"]
    n_in = v.one("select count(*) from prices where symbol=? and date between ? and ?", p["bench"], p["start"], p["end"])[0]
    series = []
    for s in (p["symbol"], p["bench"]):
        rows = v.q("select date, %s from prices where symbol=? and date<=? order by date desc limit ?" % col,
                   s, p["end"], n_in + W)[::-1]
        series.append(rows)
    assert [r[0] for r in series[0]] == [r[0] for r in series[1]]
    rs = [series[0][k][1] / series[0][k - 1][1] - 1 for k in range(1, len(series[0]))]
    rm = [series[1][k][1] / series[1][k - 1][1] - 1 for k in range(1, len(series[1]))]
    dates = [r[0] for r in series[1][1:]]
    betas = []
    for t in range(W - 1, len(rm)):
        xs, ys = rs[t - W + 1:t + 1], rm[t - W + 1:t + 1]
        mx, my = sum(xs) / W, sum(ys) / W
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (W - 1)
        var = sum((y - my) ** 2 for y in ys) / (W - 1)
        betas.append((cov / var, dates[t]))
    assert betas[0][1] == p["start"] and len(betas) == n_in
    hi, lo = max(betas), min(betas)
    out = {"max": [hi[1], hi[0]], "min": [lo[1], lo[0]], "both": [hi[1], hi[0], lo[1], lo[0]]}[p["ask"]]
    if p.get("ask_mean"):
        out.append(sum(b for b, _ in betas) / len(betas))
    return out, "%d betas from %s; max %s min %s" % (len(betas), betas[0][1], hi, lo)


def v_ma_cross(v, p):
    rows = v.q("select date, close from prices where symbol=? order by date", p["symbol"])
    ns, nl = p["short"], p["long"]
    acc, s_ma, l_ma = [0.0], [], []
    for _, c in rows:
        acc.append(acc[-1] + c)
    for k in range(len(rows)):
        s_ma.append((acc[k + 1] - acc[k + 1 - ns]) / ns if k >= ns - 1 else None)
        l_ma.append((acc[k + 1] - acc[k + 1 - nl]) / nl if k >= nl - 1 else None)

    def sig(k):
        if k < nl:
            return 0
        if s_ma[k] > l_ma[k] and s_ma[k - 1] <= l_ma[k - 1]:
            return 1
        if s_ma[k] < l_ma[k] and s_ma[k - 1] >= l_ma[k - 1]:
            return -1
        return 0
    if p["mode"] == "next":
        g = next(k for k in range(len(rows)) if rows[k][0] > p["date"] and sig(k) == 1)
        x = next(k for k in range(g + 1, len(rows)) if sig(k) == -1)
        g2 = next(k for k in range(x + 1, len(rows)) if sig(k) == 1)
        return [rows[g][0], rows[x][0], rows[g2][0], pct(rows[x][1], rows[g][1])], "golden %s death %s golden %s" % (
            rows[g], rows[x], rows[g2])
    ks = [k for k in range(len(rows)) if p["start"] <= rows[k][0] <= p["end"]]
    held, buys, growth, trades = None, 0, 1.0, []
    for k in ks:
        if held is None and sig(k) == 1:
            held, buys = k, buys + 1
        elif held is not None and sig(k) == -1:
            growth *= rows[k][1] / rows[held][1]
            trades.append((rows[held][0], rows[k][0]))
            held = None
    if held is not None:
        growth *= rows[ks[-1]][1] / rows[held][1]
        trades.append((rows[held][0], rows[ks[-1]][0]))
    bh = pct(rows[ks[-1]][1], rows[ks[0]][1])
    px = dict(rows)
    best = max((pct(px[q], px[b]), b) for b, q in trades)
    return [float(buys), (growth - 1) * 100, bh, best[1], best[0]], "trades %s" % trades


def v_reinvest_return(v, p):
    sym = p["symbol"]
    s = v.bar_le(sym, p["start"], "date, raw_close")
    e = v.bar_le(sym, p["end"], "date, raw_close")
    acts = v.q("select a.date, a.kind, a.value, a.value_raw, p.raw_close from actions a join prices p on "
               "p.symbol=a.symbol and p.date=a.date where a.symbol=? and a.date>? and a.date<=? order by a.date",
               sym, s[0], e[0])
    n_all = v.one("select count(*) from actions where symbol=? and date>? and date<=?", sym, s[0], e[0])[0]
    assert n_all == len(acts)
    shares, cash = 1.0, 0.0
    for d, kind, val, raw, close in acts:
        if kind == "split":
            shares *= val
        else:
            cash += shares * raw
            shares += shares * raw / close
    out = [shares, cash, pct(shares * e[1], s[1])]
    ev = "buy %s, sell %s, %d actions" % (s, e, len(acts))
    if p["fx"]:
        f0, f1 = v.fred_le(p["fx"], s[0]), v.fred_le(p["fx"], e[0])
        conv = (lambda x, f: x * f) if FX_PER_USD[p["fx"]] else (lambda x, f: x / f)
        out.append(pct(conv(shares * e[1], f1[1]), conv(s[1], f0[1])))
        ev += ", fx %s %s" % (f0, f1)
    return out, ev


def v_div_event_study(v, p):
    sym, N = p["symbol"], p["n"]
    col = "adj_close" if p["total"] else "close"
    exs = [r[0] for r in v.q("select date from actions where symbol=? and kind='dividend' and date like ? order by date",
                             sym, "%d-%%" % p["year"])]
    xs = []
    for x in exs:
        prev = v.bar_lt(sym, x, "date, " + col)
        after = v.q("select date, %s from prices where symbol=? and date>=? order by date limit ?" % col, sym, x, N + 1)[-1]
        g0 = v.one("select close from prices where symbol='^GSPC' and date=?", prev[0])[0]
        g1 = v.one("select close from prices where symbol='^GSPC' and date=?", after[0])[0]
        xs.append((pct(after[1], prev[1]) - pct(g1, g0), x))
    out = [float(len(exs)), sum(x for x, _ in xs) / len(xs)]
    if p["ask_min"]:
        lo = min(xs)
        out += [lo[1], lo[0]]
    if p.get("ask_max"):
        hi = max(xs)
        out += [hi[1], hi[0]]
    return out, "events %s" % ["%s %.3f" % (d, x) for x, d in xs]


def q4_parts(v, cik, fy, tag):
    """(annual value from the original 10-K, nine-month value from the first-filed Q3 10-Q of the same fiscal year)."""
    x, E, S, accn = v.sec_val(cik, fy, tag, "dur")
    r = v.q("select val from facts where cik=? and tag=? and form='10-Q' and fy=? and fp='Q3' and start=? "
            "and julianday(\"end\")-julianday(start) between 250 and 290 and unit='USD' order by filed", cik, tag, fy, S)
    return x, r[0][0]


def v_q4_margin_rank(v, p):
    ntag = {"operating": "OperatingIncomeLoss", "net": "NetIncomeLoss", "gross": "GrossProfit"}[p["metric"]]
    rows = []
    for s, cik in zip(p["symbols"], p["ciks"]):
        for t in REV_TAGS:
            try:
                rev = q4_parts(v, cik, p["fy"], t)
                break
            except LookupError:
                continue
        num = q4_parts(v, cik, p["fy"], ntag)
        q4r = rev[0] - rev[1]
        rows.append(((num[0] - num[1]) / q4r * 100, s, q4r / 1e6, num[0] / rev[0] * 100))
    ranked = sorted(rows, reverse=p["direction"] == "highest")
    best = ranked[0]
    out = [best[1], best[0]] + ([best[2]] if p["ask_more"] or p.get("ask_rev") else []) + \
        ([best[3]] if p["ask_more"] else []) + ([ranked[1][0]] if p.get("ask_second") else [])
    return out, "Q4 %s margins %s" % (p["metric"], ["%s %.3f" % (r[1], r[0]) for r in rows])


def v_filing_reaction(v, p):
    gr = []
    for s, cik in zip(p["symbols"], p["ciks"]):
        gr.append((pct(v.sec_first(cik, p["fy"], REV_TAGS), v.sec_first(cik, p["fy"] - 1, REV_TAGS)), s, cik))
    g, sym, cik = max(gr) if p["direction"] == "highest" else min(gr)
    accn, _ = v.annual_filing(cik, p["fy"])
    filed = v.one("select min(filed) from facts where cik=? and accn=?", cik, accn)[0]
    rows = v.q("select date, adj_close from prices where symbol=? and date>=? order by date limit ?", sym, filed, p["n"] + 1)
    prev = v.bar_lt(sym, rows[0][0], "date, adj_close")
    g0 = v.one("select close from prices where symbol='^GSPC' and date=?", prev[0])[0]
    g1 = v.one("select close from prices where symbol='^GSPC' and date=?", rows[-1][0])[0]
    ex = pct(rows[-1][1], prev[1]) - pct(g1, g0)
    out = [sym] + ([g] if p["ask_growth"] else []) + [filed, ex]
    return out, "growth %s; filed %s; %s -> %s" % (["%s %.3f" % (s, x) for x, s, _ in gr], filed, prev, rows[-1])


def v_month_chain(v, p):
    sym = p["symbol"]
    col = "adj_close" if p["total"] else "close"
    y, m = int(p["first_month"][:4]), int(p["first_month"][5:7])
    rets = []
    while "%04d-%02d" % (y, m) <= p["last_month"]:
        ym = "%04d-%02d" % (y, m)
        d1, c1 = v.one("select date, %s from prices where symbol=? and substr(date,1,7)=? order by date desc limit 1"
                       % col, sym, ym)
        d0, c0 = v.bar_lt(sym, ym + "-01", "date, " + col)
        rets.append((pct(c1, c0), ym, d0, d1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    nom, ym, d0, d1 = min(rets) if p["direction"] == "lowest" else max(rets)
    vcol = "value_vintage" if p["vintage"] == "as_of_cutoff" else "value"
    y, m = int(ym[:4]), int(ym[5:7])
    prev = "%04d-%02d" % ((y - 1, 12) if m == 1 else (y, m - 1))
    c_now = v.one("select %s from fred where series_id='CPIAUCSL' and date=?" % vcol, ym + "-01")[0]
    c_prev = v.one("select %s from fred where series_id='CPIAUCSL' and date=?" % vcol, prev + "-01")[0]
    real = ((1 + nom / 100) / (c_now / c_prev) - 1) * 100
    out = [d1, nom, real]
    ev = "month %s (%s..%s) cpi %s/%s" % (ym, d0, d1, c_now, c_prev)
    if p["ask_yield"]:
        a, b = v.fred_le("DGS10", d0), v.fred_le("DGS10", d1)
        out.append((b[1] - a[1]) * 100)
        ev += ", DGS10 %s %s" % (a, b)
    return out, ev


VERIFIERS = {k[2:]: f for k, f in globals().items() if k.startswith("v_") and callable(f) and k != "v_ratio"}


def _eq(key, val):
    k = key["kind"]
    if k == "number":
        # Same value after rounding to the requested unit (keys are stored rounded).
        tol = key["tol"] if "tol" in key else key["rel_tol"] * abs(key["value"])
        return abs(float(val) - key["value"]) <= 0.5 * tol + 1e-9
    if k in ("date",):
        return key["value"] == val
    if k in ("text", "ticker"):
        return val == key["value"] or val in key.get("aliases", [])
    if k == "set":
        return sorted(x["value"] for x in key["value"]) == sorted(val)
    return False


def compare(keys, vals):
    """True when the task keys appear, in order, as a subsequence of the recomputed values."""
    j = 0
    for key in keys:
        while j < len(vals) and not _eq(key, vals[j]):
            j += 1
        if j == len(vals):
            return False
        j += 1
    return True
