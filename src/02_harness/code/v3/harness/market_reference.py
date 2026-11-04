"""The reference policy of the market kind (docs/03_harness.md, 9.1; research/log/2026-10-31-0150.md, W7).

It shows that the environment's tools can solve every market task: for each task, one run_python call whose code
computes the answer with the sandbox's `data` object only (the same queries as the lookup tools), then a finish whose
answer gives the printed values in the prompt's units and rounding. The code is written from the task's
meta.params for the in-house templates (src/01_data-processing/code/market_tasks/templates.py; the computations follow
verify.py's independent implementation, translated from SQL to `data` calls) and by hand from each task's solution
description for the FinSearchComp external tasks (EXTERNAL below). The code prints a JSON list with one value per key
of the task, in the keys' order; each number is then rounded to the decimals the prompt asks for, which the key's
tolerance records (tol 0.01 -> 2 decimals, 1 -> 0; a tolerance of 0 takes the decimals the key's value is written
with), dates and names are given as printed. The values themselves come only from the tool output.

golden.py runs it through agent.Episode like the wealth policies: under H1 a checklist first and a done mark before
finish, so the finish passes H1's verification only if every number of the answer is supported by run_python's
printed output (market.unsupported_in).

Version 2 (research/log/2026-10-31-0150.md, the hard tier): code for the eight difficulty-4/5 templates of
src/01_data-processing/code/market_tasks/templates_hard.py (risk_rank ... month_chain below), still one run_python call
per task with the data object only, following the hard-tier verifiers of verify.py.

Version 3 (research/log/2026-10-31-0150.md, the hardening of five hard templates): rolling_beta, ma_cross,
reinvest_return, div_event_study and q4_margin_rank also print the added keys, as verify.py's verifiers compute them.
"""
import json
import math

PRELUDE = r'''
import json, math
import numpy as np
import pandas as pd

def shift(d, days):
    return (pd.Timestamp(d) + pd.Timedelta(days=days)).strftime('%Y-%m-%d')

def bar_le(sym, d, adj='split'):
    """(date, row) of the last trading day on or before d."""
    t = data.trading_day(sym, d)
    df = data.get_prices(sym, t, t, adjustment=adj)
    return t, df.iloc[0]

def day_before(sym, d):
    """The last trading day strictly before d."""
    return data.trading_day(sym, d, offset=-1, rule='on_or_after')

def closes(sym, start, end, adj='split', field='close'):
    s = data.get_prices(sym, start, end, adjustment=adj, fields=[field])[field]
    s.index = s.index.strftime('%Y-%m-%d')
    return s

def pct(a, b):
    return (a / b - 1.0) * 100.0

def fred_le(sid, d):
    df = data.get_fred(sid, None, d)
    return df.index[-1].strftime('%Y-%m-%d'), float(df.value.iloc[-1])

def fred_at(sid, d):
    df = data.get_fred(sid, d, d)
    return float(df.value.iloc[0])

def facts(cik, tag, **kw):
    df = data.get_financials(str(cik), tag if ':' in tag else 'us-gaap:' + tag, filing='all', **kw)
    for c in ('start', 'end', 'filed'):
        df[c] = df[c].dt.strftime('%Y-%m-%d')
    return df

def days(a, b):
    return (pd.Timestamp(b) - pd.Timestamp(a)).days

def span(df, lo, hi, prior_of=None):
    """The rows of df whose period lasts lo..hi days (with prior_of: whose end is lo..hi days before it)."""
    if df.empty:
        return df
    if prior_of is None:
        keep = [s is not None and s == s and lo <= days(s, e) <= hi for s, e in zip(df.start, df.end)]
    else:
        keep = [lo <= days(e, prior_of) <= hi for e in df.end]
    return df[np.array(keep, dtype=bool)]

def annual(cik, fy, tag, kind, which='current'):
    """A value of fiscal year fy's original 10-K (verify.py's sec_val): the earliest-filed 10-K with fp FY and that
    fy; its fiscal-year end E is the latest end of a one-year period in it; a duration value covers the year ending
    E, an instant value is at E ('current') or about a year before it ('prior')."""
    unit = 'USD/shares' if tag.startswith('EarningsPerShare') else 'USD'
    try:
        df = facts(cik, tag, fiscal_year=fy, period='FY', form='10-K')
    except Exception:
        raise LookupError(tag)
    df = df[df.unit == unit]
    if df.empty:
        raise LookupError(tag)
    accn = df.sort_values(['filed', 'accn']).accn.iloc[0]
    f = df[df.accn == accn]
    filed = f.filed.iloc[0]
    yr = span(f[f.start.notna()], 330, 400)
    E = fiscal_end(cik, fy, accn, filed) if yr.empty else max(e for e in yr.end if e <= filed)
    if kind == 'dur':
        rows = span(yr[yr.end == E], 350, 380)
    elif which == 'current':
        rows = f[f.start.isna() & (f.end == E)]
    else:
        rows = span(f[f.start.isna()], 350, 380, prior_of=E)
    vals = sorted(set(float(v) for v in rows.value))
    if len(vals) != 1:
        raise LookupError('%s %s %s: %d values' % (tag, fy, which, len(vals)))
    return vals[0], E

def fiscal_end(cik, fy, accn, filed):
    """The fiscal-year end of a filing from its one-year flows (net income, revenue, costs or cash flows)."""
    for t in ('NetIncomeLoss', 'Revenues', 'NetCashProvidedByUsedInOperatingActivities', 'ProfitLoss', 'SalesRevenueNet',
              'RevenueFromContractWithCustomerExcludingAssessedTax', 'CostOfGoodsAndServicesSold', 'CostOfRevenue',
              'CostOfGoodsSold', 'GrossProfit', 'OperatingIncomeLoss', 'IncomeTaxExpenseBenefit'):
        try:
            df = facts(cik, t, fiscal_year=fy, period='FY', form='10-K')
        except Exception:
            continue
        df = span(df[(df.accn == accn) & df.start.notna()], 330, 400)
        ends = [e for e in df.end if e <= filed]
        if ends:
            return max(ends)
    raise LookupError('no fiscal-year end')

def annual_first(cik, fy, tags):
    for t in tags:
        try:
            return annual(cik, fy, t, 'dur')[0]
        except LookupError:
            continue
    raise LookupError('none of %s' % tags)

def fact(cik, tag, end, start=None, accn=None, unit=None):
    """One reported value: the fact of tag with this period (start None: an instant), from filing accn."""
    df = facts(cik, tag)
    df = df[(df.end == end) & ((df.start == start) if start else df.start.isna())]
    if accn:
        df = df[df.accn == accn]
    if unit:
        df = df[df.unit == unit]
    vals = sorted(set(float(v) for v in df.value))
    assert len(vals) == 1, (tag, end, vals)
    return vals[0]
'''


# -- in-house templates: code from meta.params ---------------------------------------------------------------------

def _c_price_lookup(p, d):
    adj = "raw" if p["adj"] == "raw" else "split"
    return f'''
t, row = bar_le({p['symbol']!r}, {p['date']!r}, {adj!r})
out = [float(row[f]) / (1e6 if f == 'volume' and {p['vol_scale']!r} == 'millions' else 1) for f in {p['fields']!r}]
'''


def _c_fred_lookup(p, d):
    op, sid = p["op"], p["series"]
    if op == "daily":
        return f"_, x = fred_le({sid!r}, {p['date']!r})\nout = [x * 100 if {p.get('scale')!r} == 'bp' else x]\n"
    if op == "chg_bp":
        return f"a = fred_le({sid!r}, {p['d0']!r})[1]\nb = fred_le({sid!r}, {p['d1']!r})[1]\nout = [(b - a) * 100]\n"
    if op == "month_avg":
        m = p["month"]
        return (f"df = data.get_fred({sid!r}, {m + '-01'!r}, shift({m + '-01'!r}, 40))\n"
                f"df = df[df.index.strftime('%Y-%m') == {m!r}]\nout = [float(df.value.mean())]\n")
    if op == "fx":
        scale = {"DEXJPUS": 100.0, "DEXKOUS": 1000.0}.get(sid, 1.0)
        return f"_, x = fred_le({sid!r}, {p['date']!r})\nout = [{scale!r} / x if {p['invert']!r} else x]\n"
    if op == "monthly":
        return f"out = [fred_at({sid!r}, {p['month'] + '-01'!r}) * {p['mult']!r}]\n"
    if op == "other":
        return f"out = [fred_at({sid!r}, {p['date']!r}) * {p['mult']!r}]\n"
    ym = p["month"]
    y0 = "%04d-%s" % (int(ym[:4]) - 1, ym[5:7])
    return f"out = [pct(fred_at({sid!r}, {ym + '-01'!r}), fred_at({sid!r}, {y0 + '-01'!r}))]\n"


def _c_xbrl_lookup(p, d):
    cik, fy, tag, kind, mult = p["cik"], p["fy"], p["tag"], p["kind"], p["mult"]
    if p["op"] == "value":
        return f"x, E = annual({cik}, {fy}, {tag!r}, {kind!r})\nout = [x * {mult!r}]\n"
    if p["op"] == "yoy":
        return (f"x, _ = annual({cik}, {fy}, {tag!r}, {kind!r})\nx0, _ = annual({cik}, {fy - 1}, {tag!r}, {kind!r})\n"
                f"out = [(x - x0) * {mult!r}]\n")
    if p["op"] == "yoy_pct":
        return (f"x, _ = annual({cik}, {fy}, {tag!r}, {kind!r})\nx0, _ = annual({cik}, {fy - 1}, {tag!r}, {kind!r})\n"
                f"out = [pct(x, x0)]\n")
    # Q4 = the annual value minus the nine-month value of the original Q3 10-Q of the same fiscal year
    return f'''
x, E = annual({cik}, {fy}, {tag!r}, 'dur')
f = facts({cik}, {tag!r}, fiscal_year={fy}, form='10-K')
S = [s for s, e, v in zip(f.start, f.end, f.value) if e == E and s and 350 <= days(s, e) <= 380][0]
q = facts({cik}, {tag!r}, fiscal_year={fy}, period='Q3', form='10-Q')
q = span(q[(q.start == S) & (q.unit == 'USD')], 250, 290).sort_values('filed')
out = [(x - float(q.value.iloc[0])) * 1e-6]
'''


def _c_period_return(p, d):
    return f'''
out = []
for m in {p['measures']!r}:
    adj = 'total' if m == 'total' else 'split'
    a = bar_le({p['symbol']!r}, {p['start']!r}, adj)[1].close
    b = bar_le({p['symbol']!r}, {p['end']!r}, adj)[1].close
    out.append(pct(b, a))
'''


def _c_nth_day_move(p, d):
    return f'''
sym = {p['symbol']!r}
td = [x.strftime('%Y-%m-%d') for x in data.trading_days(sym, {p['date']!r})]
day = [x for x in td if x > {p['date']!r}][{p['n']} - 1]
row = data.get_prices(sym, day, day, adjustment='split').iloc[0]
if {p['mode']!r} == 'cc':
    v = pct(row.close, bar_le(sym, day_before(sym, day))[1].close)
else:
    v = pct(row.close, row.open)
out = [v] + ([day] if {d} == 3 else [])
'''


def _c_extreme_period(p, d):
    sym, pick = p["symbol"], "min" if p["direction"] == "loss" else "max"
    if p["freq"] == "month":
        start = p["first"][:7] + "-01"
        return f'''
c = closes({sym!r}, {start!r}, {p['last']!r})
prev = bar_le({sym!r}, day_before({sym!r}, {start!r}))[1].close
last = c.groupby(c.index.str[:7]).last()
rets = []
for k, v in last.items():
    rets.append((pct(v, prev), k))
    prev = v
best = {pick}(rets)
out = [best[1], best[0]]
'''
    return f'''
d0 = pd.Timestamp({p['first']!r})
monday = (d0 - pd.Timedelta(days=d0.weekday())).strftime('%Y-%m-%d')
c = closes({sym!r}, monday, {p['last']!r})
prev = bar_le({sym!r}, day_before({sym!r}, monday))[1].close
wk = [tuple(pd.Timestamp(x).isocalendar()[:2]) for x in c.index]
rets = []
for k in sorted(set(wk)):
    sel = [i for i, w in enumerate(wk) if w == k]
    v, lab = c.iloc[sel[-1]], c.index[sel[-1]]
    rets.append((pct(v, prev), lab))
    prev = v
best = {pick}(rets)
out = [best[1], best[0]]
'''


def _c_threshold_scan(p, d):
    adj = "raw" if p["mode"] == "usd_drop" else "split"
    return f'''
sym, X = {p['symbol']!r}, {p['X']!r}
c = closes(sym, day_before(sym, {p['start']!r}), {p['end']!r}, {adj!r})
chg, ret = c.diff(), c.pct_change() * 100
keep = c.index >= {p['start']!r}
mode = {p['mode']!r}
hit = (chg < -X) if mode == 'usd_drop' else (ret < -X) if mode == 'pct_drop' else (ret > X)
hits = sorted(c.index[keep & hit.values])
out = [float(len(hits))] + ([hits] if {p['list']!r} else [])
'''


def _c_max_drawdown(p, d):
    adj = "total" if p["measure"] == "total" else "split"
    return f'''
c = closes({p['symbol']!r}, {p['start']!r}, {p['end']!r}, {adj!r})
peak, peak_d, best, trough, best_peak = -1, None, 0.0, None, None
for day, v in c.items():
    if v > peak:
        peak, peak_d = v, day
    if v / peak - 1 < best:
        best, trough, best_peak = v / peak - 1, day, peak_d
out = [-best * 100, trough] + ([best_peak] if {p['with_peak']!r} else [])
'''


def _c_rolling_vol(p, d):
    return f'''
rets = []
for s in {p['symbols']!r}:
    n_in = data.count_trading_days(s, {p['start']!r}, {p['end']!r})
    c = closes(s, None, {p['end']!r}).iloc[-(n_in + 21):]
    rets.append(c.pct_change().iloc[1:])
r = (rets[0] - rets[1]) * 100 if len(rets) == 2 else rets[0] * 100
sd = r.rolling(20).std(ddof=1)
out = [float(sd[sd.index >= {p['start']!r}].mean())]
'''


def _c_split_event(p, d):
    return f'''
sym, E = {p['symbol']!r}, {p['effective']!r}
a = data.get_corporate_actions(sym, E, E, kind='split')
ratio = float(a.ratio.iloc[0])
o = float(data.get_prices(sym, E, E, adjustment='raw').open.iloc[0])
prev = float(bar_le(sym, day_before(sym, E), 'raw')[1].close)
out = ([E] if {p['ask_date']!r} else []) + [ratio, o - prev] + ([prev / o] if {p['want_pr']!r} else [])
'''


def _c_dividends(p, d):
    sym = p["symbol"]
    if p["op"] in ("year_raw", "year_adj", "year_count"):
        col = "amount_split_adjusted" if p["op"] == "year_adj" else "amount_raw"
        y = p["year"]
        return f'''
a = data.get_corporate_actions({sym!r}, '{y}-01-01', '{y}-12-31', kind='dividend')
tot = float(a.{col}.sum())
out = ([float(len(a))] if {p['op']!r} == 'year_count' else []) + [tot]
'''
    D = p["date"]
    return f'''
D = pd.Timestamp({D!r})
try:
    start = D.replace(year=D.year - 1)
except ValueError:
    start = D.replace(year=D.year - 1, day=28)
a = data.get_corporate_actions({sym!r}, start.strftime('%Y-%m-%d'), {D!r}, kind='dividend')
a = a[a.index > start]
out = [float(a.amount_split_adjusted.sum()) / float(bar_le({sym!r}, {D!r})[1].close) * 100]
'''


def _c_cross_rank(p, d):
    pick = "max" if p["direction"] == "highest" else "min"
    return f'''
vals = []
for s in {p['symbols']!r}:
    if {p['metric']!r} == 'vol':
        c = closes(s, {p['start']!r}, {p['end']!r})
        vals.append((float(c.pct_change().iloc[1:].std(ddof=1) * math.sqrt(252) * 100), s))
    else:
        adj = 'total' if {p['metric']!r} == 'total' else 'split'
        a = data.get_prices(s, {p['start']!r}, {p['start']!r}, adjustment=adj).close.iloc[0]
        b = data.get_prices(s, {p['end']!r}, {p['end']!r}, adjustment=adj).close.iloc[0]
        vals.append((pct(b, a), s))
best = {pick}(vals)
out = [best[1], best[0]]
'''


def _c_fx_index_return(p, d):
    conv = "x / f" if p["per_usd"] else "x * f"
    return f'''
sym, sid = {p['symbol']!r}, {p['series']!r}
ta, a = bar_le(sym, {p['start']!r})
tb, b = bar_le(sym, {p['end']!r})
fa, fb = fred_le(sid, ta)[1], fred_le(sid, tb)[1]
conv = lambda x, f: {conv}
usd = pct(conv(b.close, fb), conv(a.close, fa))
out = ([pct(b.close, a.close)] if {d} == 3 else []) + [usd]
'''


def _c_real_return(p, d):
    adj = "total" if p["total"] else "split"
    return f'''
def month_end(m):
    c = closes({p['symbol']!r}, m + '-01', shift(m + '-01', 31), {adj!r})
    return c[c.index.str[:7] == m].iloc[-1]
nom = month_end({p['m1']!r}) / month_end({p['m0']!r})
infl = fred_at('CPIAUCSL', {p['m1'] + '-01'!r}) / fred_at('CPIAUCSL', {p['m0'] + '-01'!r})
out = ([(nom - 1) * 100] if {d} >= 2 else []) + ([(infl - 1) * 100] if {d} == 3 else []) + [(nom / infl - 1) * 100]
'''


def _c_xbrl_ratio(p, d):
    pick = "max" if p["direction"] == "highest" else "min"
    return f'''
REV = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet']
COGS = ['CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfGoodsSold']
def ratio(cik, fy, r):
    if r == 'net_margin':
        return annual(cik, fy, 'NetIncomeLoss', 'dur')[0] / annual_first(cik, fy, REV) * 100
    if r == 'roe':
        ni = annual(cik, fy, 'NetIncomeLoss', 'dur')[0]
        e1, e0 = annual(cik, fy, 'StockholdersEquity', 'inst')[0], annual(cik, fy, 'StockholdersEquity', 'inst', 'prior')[0]
        return ni / ((e0 + e1) / 2) * 100
    i1, i0 = annual(cik, fy, 'InventoryNet', 'inst')[0], annual(cik, fy, 'InventoryNet', 'inst', 'prior')[0]
    return annual_first(cik, fy, COGS) / ((i0 + i1) / 2)
avgs = []
for s, cik in zip({p['symbols']!r}, {p['ciks']!r}):
    xs = [ratio(cik, y, {p['ratio']!r}) for y in {p['years']!r}]
    avgs.append((sum(xs) / len(xs), s))
best = {pick}(avgs)
out = [best[1], best[0]]
'''


def _c_market_cap(p, d):
    return f'''
t, row = bar_le({p['symbol']!r}, {p['date']!r}, 'raw')
f = facts({p['cik']}, 'dei:EntityCommonStockSharesOutstanding')
f = f[(f.unit == 'shares') & (f.filed <= {p['date']!r})]
last = f[f.filed == f.filed.max()]
vals = sorted(set(float(v) for v in last.value))
assert len(vals) == 1, vals
sh = vals[0]
out = ([sh / 1e6] if {p['ask_shares']!r} else []) + [float(row.close) * sh / 1e9]
'''


def _c_two_hop(p, d):
    sym, ch = p["symbol"], p["chain"]
    if ch == "mdd_dgs10":
        return f'''
c = closes({sym!r}, {p['start']!r}, {p['end']!r})
dd = c / c.cummax() - 1
hop = dd.idxmin() if dd.min() < 0 else None
out = [hop, fred_le('DGS10', hop)[1]]
'''
    if ch == "worstday_vix":
        return f'''
c = closes({sym!r}, day_before({sym!r}, {p['start']!r}), {p['end']!r})
r = c.pct_change()
r = r[r.index >= {p['start']!r}]
hop = r.idxmin()
out = [hop, float(data.get_prices('^VIX', hop, hop).close.iloc[0])]
'''
    if ch == "bestmonth_unrate":
        return f'''
c = closes({sym!r}, None, shift({p['last_month'] + '-01'!r}, 31))
months = pd.period_range({p['first_month']!r}, {p['last_month']!r}, freq='M').strftime('%Y-%m')
best = None
for ym in months:
    c1 = c[c.index.str[:7] == ym].iloc[-1]
    c0 = c[c.index < ym + '-01'].iloc[-1]
    if best is None or c1 / c0 - 1 > best[0]:
        best = (c1 / c0 - 1, ym)
out = [best[1], fred_at('UNRATE', best[1] + '-01')]
'''
    return f'''
c = closes({sym!r}, {p['start']!r}, {p['end']!r})
hop = c.idxmax()
out = [hop, fred_le('DFF', hop)[1]]
'''


# -- hard tier (templates_hard.py): code from meta.params, following verify.py's hard-tier verifiers ---------------

REV_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
FX_PER_USD = {"DEXJPUS": True, "DEXSZUS": True, "DEXCAUS": True, "DEXUSEU": False, "DEXUSUK": False, "DEXUSAL": False}


def _c_risk_rank(p, d):
    hi, lo = "[hi[1], hi[0]]", "[lo[1], lo[0]]"
    out = {"highest": hi, "lowest": lo, "both": hi + " + " + lo}[p["ask"]]
    return f'''
s0 = data.trading_day('^GSPC', {p['start']!r})
e0 = data.trading_day('^GSPC', {p['end']!r})
if {p['rf']!r}:
    f = data.get_fred('DTB3', None, e0).value
    f.index = f.index.strftime('%Y-%m-%d')
vals = []
for s in {p['symbols']!r}:
    r = closes(s, s0, e0, 'total').pct_change().iloc[1:]
    if {p['rf']!r}:
        r = r - f.reindex(f.index.union(r.index)).ffill().reindex(r.index) / 100 / 252
    if {p['metric']!r} == 'sharpe':
        x = r.mean() / r.std(ddof=1) * math.sqrt(252)
    else:
        x = r.mean() / math.sqrt((np.minimum(r, 0) ** 2).mean()) * math.sqrt(252)
    vals.append((float(x), s))
hi, lo = max(vals), min(vals)
out = {out}
'''


def _c_rolling_beta(p, d):
    adj = "total" if p["total"] else "split"
    out = {"max": "[hi, float(b.max())]", "min": "[lo, float(b.min())]",
           "both": "[hi, float(b.max()), lo, float(b.min())]"}[p["ask"]]
    if p.get("ask_mean"):
        out += " + [float(b.mean())]"
    return f'''
sym, bench, W = {p['symbol']!r}, {p['bench']!r}, {p['window']}
n_in = data.count_trading_days(bench, {p['start']!r}, {p['end']!r})
cb = closes(bench, None, {p['end']!r}, {adj!r}).iloc[-(n_in + W):]
cs = closes(sym, None, {p['end']!r}, {adj!r}).iloc[-(n_in + W):]
assert list(cb.index) == list(cs.index)
rm, rs = cb.pct_change().iloc[1:], cs.pct_change().iloc[1:]
b = rs.rolling(W).cov(rm) / rm.rolling(W).var()
b = b[b.index >= {p['start']!r}]
hi, lo = b.idxmax(), b.idxmin()
out = {out}
'''


def _c_ma_cross(p, d):
    head = f'''
c = closes({p['symbol']!r}, None, None)
s, l = c.rolling({p['short']}).mean(), c.rolling({p['long']}).mean()
up = (s > l) & (s.shift(1) <= l.shift(1))
dn = (s < l) & (s.shift(1) >= l.shift(1))
'''
    if p["mode"] == "next":
        return head + f'''
g = up[up & (up.index > {p['date']!r})].index[0]
x = dn[dn & (dn.index > g)].index[0]
g2 = up[up & (up.index > x)].index[0]
out = [g, x, g2, pct(c[x], c[g])]
'''
    return head + f'''
days = [t for t in c.index if {p['start']!r} <= t <= {p['end']!r}]
held, buys, growth, trades = None, 0, 1.0, []
for t in days:
    if held is None and up[t]:
        held, buys = t, buys + 1
    elif held is not None and dn[t]:
        growth *= c[t] / c[held]
        trades.append((pct(c[t], c[held]), held))
        held = None
if held is not None:
    growth *= c[days[-1]] / c[held]
    trades.append((pct(c[days[-1]], c[held]), held))
best = max(trades)
out = [float(buys), (growth - 1) * 100, pct(c[days[-1]], c[days[0]]), best[1], best[0]]
'''


def _c_reinvest_return(p, d):
    fx = ""
    if p["fx"]:
        conv = "x * f" if FX_PER_USD[p["fx"]] else "x / f"
        fx = f'''
f0, f1 = fred_le({p['fx']!r}, t0)[1], fred_le({p['fx']!r}, t1)[1]
conv = lambda x, f: {conv}
out.append(pct(conv(shares * float(b1.close), f1), conv(float(b0.close), f0)))
'''
    return f'''
sym = {p['symbol']!r}
t0, b0 = bar_le(sym, {p['start']!r}, 'raw')
t1, b1 = bar_le(sym, {p['end']!r}, 'raw')
c = closes(sym, t0, t1, 'raw')
a = data.get_corporate_actions(sym, shift(t0, 1), t1)
shares, cash = 1.0, 0.0
for day, row in a.iterrows():
    if row.kind == 'split':
        shares *= float(row.ratio)
    else:
        cash += shares * float(row.amount_raw)
        shares += shares * float(row.amount_raw) / float(c[day.strftime('%Y-%m-%d')])
out = [shares, cash, pct(shares * float(b1.close), float(b0.close))]
''' + fx


def _c_div_event_study(p, d):
    y, adj = p["year"], ("total" if p["total"] else "split")
    return f'''
sym, N = {p['symbol']!r}, {p['n']}
a = data.get_corporate_actions(sym, '{y}-01-01', '{y}-12-31', kind='dividend')
c = closes(sym, '{y - 1}-11-01', '{y + 1}-02-28', {adj!r})
g = closes('^GSPC', '{y - 1}-11-01', '{y + 1}-02-28')
days = list(c.index)
xs = []
for x in a.index.strftime('%Y-%m-%d'):
    k = days.index(x)
    d0, d1 = days[k - 1], days[k + N]
    xs.append((pct(c[d1], c[d0]) - pct(g[d1], g[d0]), x))
out = [float(len(xs)), sum(v for v, _ in xs) / len(xs)]
if {p['ask_min']!r}:
    lo = min(xs)
    out += [lo[1], lo[0]]
if {p.get('ask_max', False)!r}:
    hi = max(xs)
    out += [hi[1], hi[0]]
'''


Q4_HELPERS = '''
REV = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet']
def q4(cik, fy, tag):
    """(annual value of the original 10-K, nine-month value of the first-filed Q3 10-Q of the same fiscal year)."""
    x, E = annual(cik, fy, tag, 'dur')
    f = facts(cik, tag, fiscal_year=fy, form='10-K')
    S = [s for s, e in zip(f.start, f.end) if e == E and isinstance(s, str) and 350 <= days(s, e) <= 380][0]
    q = facts(cik, tag, fiscal_year=fy, period='Q3', form='10-Q')
    q = span(q[(q.start == S) & (q.unit == 'USD')], 250, 290).sort_values('filed')
    return x, float(q.value.iloc[0])
'''


def _c_q4_margin_rank(p, d):
    ntag = {"operating": "OperatingIncomeLoss", "net": "NetIncomeLoss", "gross": "GrossProfit"}[p["metric"]]
    rev = "[best[2]]" if p["ask_more"] or p.get("ask_rev") else "[]"
    return Q4_HELPERS + f'''
rows = []
for s, cik in zip({p['symbols']!r}, {p['ciks']!r}):
    for t in REV:
        try:
            annual(cik, {p['fy']}, t, 'dur')
            break
        except LookupError:
            continue
    rev, num = q4(cik, {p['fy']}, t), q4(cik, {p['fy']}, {ntag!r})
    q4r = rev[0] - rev[1]
    rows.append(((num[0] - num[1]) / q4r * 100, s, q4r / 1e6, num[0] / rev[0] * 100))
rows.sort(reverse={p['direction'] == 'highest'!r})
best = rows[0]
out = [best[1], best[0]] + {rev} + ([best[3]] if {p['ask_more']!r} else []) + \\
    ([rows[1][0]] if {p.get('ask_second', False)!r} else [])
'''


def _c_filing_reaction(p, d):
    pick = "max" if p["direction"] == "highest" else "min"
    fy = p["fy"]
    return f'''
REV = {REV_TAGS!r}
gr = []
for s, cik in zip({p['symbols']!r}, {p['ciks']!r}):
    gr.append((pct(annual_first(cik, {fy}, REV), annual_first(cik, {fy - 1}, REV)), s, cik))
g, sym, cik = {pick}(gr)
for t in REV:
    try:
        f = facts(cik, t, fiscal_year={fy}, period='FY', form='10-K')
        break
    except Exception:
        continue
filed = f.sort_values(['filed', 'accn']).filed.iloc[0]
day0 = data.trading_day(sym, filed, rule='on_or_after')
dm1, dn = data.trading_day(sym, day0, offset=-1), data.trading_day(sym, day0, offset={p['n']})
c, m = closes(sym, dm1, dn, 'total'), closes('^GSPC', dm1, dn)
ex = pct(c[dn], c[dm1]) - pct(m[dn], m[dm1])
out = [sym] + ([g] if {p['ask_growth']!r} else []) + [filed, ex]
'''


def _c_month_chain(p, d):
    adj = "total" if p["total"] else "split"
    pick = "min" if p["direction"] == "lowest" else "max"
    return f'''
c = closes({p['symbol']!r}, shift({p['first_month'] + '-01'!r}, -40), shift({p['last_month'] + '-01'!r}, 31), {adj!r})
rets = []
for ym in pd.period_range({p['first_month']!r}, {p['last_month']!r}, freq='M').strftime('%Y-%m'):
    cm, prev = c[c.index.str[:7] == ym], c[c.index < ym + '-01']
    rets.append((pct(cm.iloc[-1], prev.iloc[-1]), ym, prev.index[-1], cm.index[-1]))
nom, ym, d0, d1 = {pick}(rets)
pm = (pd.Period(ym, freq='M') - 1).strftime('%Y-%m')
cpi = data.get_fred('CPIAUCSL', pm + '-01', ym + '-01', vintage={p['vintage']!r}).value
cpi.index = cpi.index.strftime('%Y-%m')
real = ((1 + nom / 100) / (cpi[ym] / cpi[pm]) - 1) * 100
out = [d1, nom, real]
if {p['ask_yield']!r}:
    out.append((fred_le('DGS10', d1)[1] - fred_le('DGS10', d0)[1]) * 100)
'''


INHOUSE = {k[3:]: f for k, f in globals().items() if k.startswith("_c_")}


def code_of(task):
    """The run_python code that prints the task's answer values as a JSON list, one per key."""
    if task["task_id"] in EXTERNAL:
        body = EXTERNAL[task["task_id"]]
    else:
        body = INHOUSE[task["template"]](task["meta"]["params"], task["meta"]["difficulty"])
    return PRELUDE.strip() + "\n\n" + body.strip() + "\nprint(json.dumps(out, default=float))\n"


def decimals(key):
    """The decimals the prompt asks for, as the key's tolerance records them."""
    tol = key.get("tol")
    if tol is None and "rel_tol" in key:
        return 2
    if tol == 0.5:  # a count
        return 0
    if not tol:
        s = repr(float(key["value"]))
        return 0 if s.endswith(".0") else len(s.split(".")[1])
    return max(0, -math.floor(math.log10(tol) + 1e-9))


def fmt(key, v):
    if key["kind"] == "set":
        return ", ".join(map(str, v))
    if key["kind"] in ("text", "ticker", "date") or isinstance(v, str):
        return str(v)
    n = decimals(key)
    return f"{float(v):.{n}f}"


def answer_of(task, values):
    if len(values) != len(task["keys"]):
        raise ValueError(f"{len(values)} values for {len(task['keys'])} keys")
    return "Answer: " + "; ".join(fmt(k, v) for k, v in zip(task["keys"], values)) + "."


# -- the FinSearchComp external tasks: code by hand from each solution description ---------------------------------
# helper shorthands used below: fact(cik, tag, end, start, accn) is one reported value; bar_le and closes are prices.

def _sec(cik, tag, end, start=None, accn=None, scale=1.0):
    return f"fact({cik}, {tag!r}, {end!r}, {start!r}, {accn!r}) * {scale!r}"


def _one(expr):
    return f"out = [{expr}]"


EXTERNAL = {
    # T2: FRED values
    "fsc-(T2)Simple_Historical_Lookup_503":
        "df = data.get_fred('IORB', '2023-03-01', '2023-03-31')\nout = [float(df.value.iloc[-1])]",
    "fsc-(T2)Simple_Historical_Lookup_504":
        "out = [fred_le('DFEDTARL', '2022-09-30')[1], fred_le('DFEDTARU', '2022-09-30')[1]]",
    "fsc-(T2)Simple_Historical_Lookup_507": _one("fred_at('A191RL1A225NBEA', '2015-01-01')"),
    "fsc-(T2)Simple_Historical_Lookup_508": _one("fred_at('NZLNGDPRPCPPPT', '2005-01-01')"),
    "fsc-(T2)Simple_Historical_Lookup_513": _one("fred_at('FPCPITOTLZGAUS', '2022-01-01')"),
    "fsc-(T2)Simple_Historical_Lookup_514": _one("fred_at('PCAGDPILA646NWDB', '2020-01-01')"),
    "fsc-(T2)Simple_Historical_Lookup_567": _one("fred_at('MRTSSM44000USN', '2014-12-01')"),
    "fsc-(T2)Simple_Historical_Lookup_570": _one("fred_at('RUSCPIALLMINMEI', '2015-04-01')"),
    "fsc-(T2)Simple_Historical_Lookup_576": _one("fred_at('LRHUTTTTJPM156S', '2009-08-01')"),
    "fsc-(T2)Simple_Historical_Lookup_578": _one("fred_at('BUSINV', '2021-09-01') / 1000"),
    "fsc-(T2)Simple_Historical_Lookup_581": _one("fred_at('DGS10', '2023-05-22')"),
    "fsc-(T2)Simple_Historical_Lookup_582": _one("fred_at('DGS3MO', '2024-07-03')"),
    "fsc-(T2)Simple_Historical_Lookup_598": _one("fred_at('M2V', '2020-07-01')"),
    "fsc-(T2)Simple_Historical_Lookup_599": _one("fred_at('BORROW', '2023-04-01')"),
    "fsc-(T2)Simple_Historical_Lookup_600": _one("fred_at('TOTRESNS', '2021-12-01')"),
    "fsc-(T2)Simple_Historical_Lookup_601": _one("fred_at('DTB6', '2022-07-15')"),
    # T2: prices (unadjusted, the day's bar)
    "fsc-(T2)Simple_Historical_Lookup_572": _one("float(data.get_prices('DX-Y.NYB', '2023-03-08', '2023-03-08').close.iloc[0])"),
    "fsc-(T2)Simple_Historical_Lookup_584": _one("float(data.get_prices('^N225', '2024-11-15', '2024-11-15').high.iloc[0])"),
    "fsc-(T2)Simple_Historical_Lookup_585": _one("float(data.get_prices('^GDAXI', '2022-05-30', '2022-05-30').close.iloc[0])"),
    "fsc-(T2)Simple_Historical_Lookup_586": _one("float(data.get_prices('^VIX', '2022-04-25', '2022-04-25').close.iloc[0])"),
    "fsc-(T2)Simple_Historical_Lookup_593": _one("float(data.get_prices('^DJI', '2023-06-07', '2023-06-07').close.iloc[0])"),
    # T2: SEC values as first reported in the named filing
    "fsc-(T2)Simple_Historical_Lookup_536": _one(_sec(1045810, "RetainedEarningsAccumulatedDeficit", "2022-01-30", None, "0001045810-22-000036", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_537": _one(_sec(1045810, "DeferredTaxAssetsNet", "2022-01-30", None, "0001045810-22-000036", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_540": _one(_sec(1045810, "Goodwill", "2022-01-30", None, "0001045810-22-000036", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_541": _one(_sec(799288, "InterestExpense", "2021-01-29", "2020-02-01", "0001564590-21-015555", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_542": _one(_sec(799288, "PrepaidExpenseAndOtherAssetsCurrent", "2021-01-29", None, "0001564590-21-015555", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_543": _one(_sec(799288, "AdditionalPaidInCapitalCommonStock", "2021-01-29", None, "0001564590-21-015555", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_544": _one(_sec(1818874, "NetCashProvidedByUsedInFinancingActivities", "2020-12-31", "2020-07-10", "0001104659-21-037483")),
    "fsc-(T2)Simple_Historical_Lookup_545": _one(_sec(1818874, "Cash", "2020-12-31", None, "0001104659-21-037483")),
    "fsc-(T2)Simple_Historical_Lookup_546": _one(_sec(1818874, "AccruedLiabilitiesCurrent", "2020-12-31", None, "0001104659-21-037483")),
    "fsc-(T2)Simple_Historical_Lookup_547": _one(_sec(1093691, "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "2016-12-31", "2016-01-01", "0001558370-17-001631", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_549": _one(_sec(1093691, "IntangibleAssetsNetExcludingGoodwill", "2016-12-31", None, "0001558370-17-001631", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_550": _one(_sec(804328, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "2016-09-25", None, "0001234452-16-000552", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_551": _one(_sec(804328, "CommercialPaper", "2016-09-25", None, "0001234452-16-000552", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_552": _one(_sec(804328, "LiabilitiesCurrent", "2016-09-25", None, "0001234452-16-000552", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_553": _one(_sec(1600983, "SellingAndMarketingExpense", "2021-12-31", "2021-01-01", "0001410578-22-000800") + " / " + _sec(1600983, "RevenueFromContractWithCustomerExcludingAssessedTax", "2021-12-31", "2021-01-01", "0001410578-22-000800") + " * 100"),
    "fsc-(T2)Simple_Historical_Lookup_554": _one(_sec(1600983, "GeneralAndAdministrativeExpense", "2021-12-31", "2021-01-01", "0001410578-22-000800") + " / " + _sec(1600983, "RevenueFromContractWithCustomerExcludingAssessedTax", "2021-12-31", "2021-01-01", "0001410578-22-000800") + " * 100"),
    "fsc-(T2)Simple_Historical_Lookup_555": _one(_sec(1600983, "AccountsPayableCurrent", "2021-12-31", None, "0001410578-22-000800", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_556": _one(_sec(1600983, "StockholdersEquity", "2021-12-31", None, "0001410578-22-000800", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_557": _one(_sec(1647639, "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests", "2019-12-31", None, "0001647639-21-000022", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_558": _one(_sec(1647639, "LiabilitiesAndStockholdersEquity", "2020-12-31", None, "0001647639-21-000022", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_561": _one(_sec(1836833, "NetCashProvidedByUsedInInvestingActivities", "2023-01-31", "2022-02-01", "0001836833-23-000014", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_563": _one(_sec(1836833, "GrossProfit", "2023-01-31", "2022-02-01", "0001836833-23-000014", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_566": _one("(" + _sec(1326380, "AssetsCurrent", "2016-01-30", None, "0001326380-16-000320") + " - " + _sec(1326380, "LiabilitiesCurrent", "2016-01-30", None, "0001326380-16-000320") + ") / 1e6"),
    "fsc-(T2)Simple_Historical_Lookup_607": _one(_sec(1326801, "RevenueFromContractWithCustomerExcludingAssessedTax", "2019-12-31", "2019-01-01", "0001326801-20-000013", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_608": _one(_sec(1326801, "CashAndCashEquivalentsAtCarryingValue", "2019-12-31", None, "0001326801-20-000013", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_609": _one(_sec(829224, "Assets", "2020-09-27", None, "0000829224-20-000078", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_610": _one(_sec(829224, "LongTermDebt", "2020-09-27", None, "0000829224-20-000078", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_611": _one(_sec(789019, "StockholdersEquity", "2018-06-30", None, "0001564590-18-019062", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_612": _one(_sec(789019, "OperatingIncomeLoss", "2018-06-30", "2017-07-01", "0001564590-18-019062", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_613": _one(_sec(1411579, "EarningsPerShareDiluted", "2013-12-31", "2013-01-01", "0001047469-15-001918")),
    "fsc-(T2)Simple_Historical_Lookup_614": _one(_sec(1411579, "ReceivablesNetCurrent", "2013-12-31", None, "0001047469-14-004674", 1e-3)),
    "fsc-(T2)Simple_Historical_Lookup_615": _one(_sec(320193, "NetCashProvidedByUsedInOperatingActivities", "2021-09-25", "2020-09-27", "0000320193-21-000105", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_616": _one(_sec(320193, "NetCashProvidedByUsedInInvestingActivities", "2021-09-25", "2020-09-27", "0000320193-21-000105", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_617": _one("abs(" + _sec(320193, "NetCashProvidedByUsedInFinancingActivities", "2021-09-25", "2020-09-27", "0000320193-21-000105", 1e-6) + ")"),
    "fsc-(T2)Simple_Historical_Lookup_618": _one(_sec(1326380, "AssetsCurrent", "2022-01-29", None, "0001326380-22-000021", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_619": _one(_sec(1326380, "PropertyPlantAndEquipmentNet", "2022-01-29", None, "0001326380-22-000021", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_620": _one(_sec(1326380, "Liabilities", "2022-01-29", None, "0001326380-22-000021", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_621": _one(_sec(104169, "CostOfRevenue", "2018-01-31", "2017-02-01", "0000104169-18-000028", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_622": _one(_sec(104169, "InventoryNet", "2018-01-31", None, "0000104169-18-000028", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_623": _one(_sec(104169, "PrepaidExpenseAndOtherAssetsCurrent", "2018-01-31", None, "0000104169-18-000028", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_624": _one(_sec(1543151, "CostsAndExpenses", "2019-12-31", "2019-01-01", "0001543151-20-000010", 1e-6)),
    "fsc-(T2)Simple_Historical_Lookup_626": _one(_sec(1543151, "ResearchAndDevelopmentExpense", "2019-12-31", "2019-01-01", "0001543151-20-000010", 1e-6)),
    # T3
    "fsc-(T3)Complex_Historical_Investigation_001": '''
p = data.get_prices('AAPL', '2019-02-20', '2019-02-21')
out = [float(p.open.iloc[1] - p.close.iloc[0])]''',
    "fsc-(T3)Complex_Historical_Investigation_011": '''
c = closes('TSLA', '2023-12-29', '2024-07-24', 'raw')
r = c.pct_change() * 100
drops = [d for d, x in r.items() if d >= '2024-01-01' and d < '2024-07-24' and x < -10]
last = max(drops)
out = [last, float(days(last, '2024-07-24'))]''',
    "fsc-(T3)Complex_Historical_Investigation_023": '''
c = closes('^GSPC', '2009-12-01', '2025-04-30', 'raw')
m = c.groupby(c.index.str[:7]).last()
r = m.pct_change() * 100
r = r[(r.index >= '2010-01') & (r.index <= '2025-04')]
out = [r.idxmax(), float(r.max())]''',
    "fsc-(T3)Complex_Historical_Investigation_037": '''
out = []
for s in ('BA', 'LMT'):
    c = closes(s, '2024-06-06', '2024-06-07', 'raw')
    out.append(pct(c.iloc[1], c.iloc[0]))''',
    "fsc-(T3)Complex_Historical_Investigation_046": '''
ch = {}
for s in ('BAC', 'WFC'):
    c = closes(s, '2023-02-01', '2023-02-02', 'raw')
    ch[s] = pct(c.iloc[1], c.iloc[0])
out = [ch['BAC'], ch['WFC'], max(ch, key=lambda s: abs(ch[s]))]''',
    "fsc-(T3)Complex_Historical_Investigation_047": '''
a = data.get_corporate_actions('AAPL', kind='split')
a = a[a.split_type == 'split']
E = a.index[-1].strftime('%Y-%m-%d')
o = float(data.get_prices('AAPL', E, E).open.iloc[0])
prev = float(bar_le('AAPL', day_before('AAPL', E), 'raw')[1].close)
out = [o - prev, a.ratio_text.iloc[-1]]''',
    "fsc-(T3)Complex_Historical_Investigation_111": '''
out = []
for s in ('NVDA', 'AAPL', 'TSLA'):
    c = closes(s, '2021-01-08', '2021-01-11', 'raw')
    out.append(float(c.iloc[1] - c.iloc[0]))''',
    "fsc-(T3)Complex_Historical_Investigation_205": '''
def quarters(cik, fys):
    """GrossProfit / revenue per fiscal quarter, Q4 = FY - nine months, from each filing as first reported."""
    out = {}
    for tag_rev in ('RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues'):
        try:
            rev = facts(cik, tag_rev)
            break
        except Exception:
            continue
    gp = facts(cik, 'GrossProfit')
    def first(df, s, e):
        x = df[(df.start == s) & (df.end == e)].sort_values('filed')
        return float(x.value.iloc[0]) if len(x) else None
    for fy in fys:
        a = gp[(gp.fy == fy) & (gp.form == '10-K') & (gp.fp == 'FY')]
        a = span(a, 350, 380)
        S, E = a.sort_values('end').iloc[-1][['start', 'end']]
        q = gp[(gp.form == '10-Q') & (gp.fy == fy)]
        q = span(q, 80, 100)
        q = q[(q.start >= S) & (q.end <= E)]
        spans = sorted(set(zip(q.start, q.end)), key=lambda x: x[1])
        g3 = r3 = 0.0
        for k, (s, e) in enumerate(spans[:3], 1):
            g, r = first(gp, s, e), first(rev, s, e)
            out['%dQ%d' % (fy, k)] = g / r * 100
            g3, r3 = g3 + g, r3 + r
        out['%dQ4' % fy] = (first(gp, S, E) - g3) / (first(rev, S, E) - r3) * 100
    return out
amd, intc = quarters(2488, (2022, 2023, 2024)), quarters(50863, (2022, 2023, 2024))
q = min(amd, key=amd.get)
out = [q, amd[q], intc[q]]''',
    "fsc-(T3)Complex_Historical_Investigation_207": '''
p = data.get_prices('^AXJO', '2024-12-18', '2024-12-18').iloc[0]
out = [float(p.open), float(p.close), pct(p.close, p.open)]''',
    "fsc-(T3)Complex_Historical_Investigation_211": '''
t = data.trading_day('AMD', '2023-11-13', offset=10)
c = closes('AMD', '2023-11-13', t, 'raw')
out = [pct(c.iloc[-1], c.iloc[0])]''',
    "fsc-(T3)Complex_Historical_Investigation_212": '''
p = data.get_prices('^IXIC', '2025-03-07', '2025-03-07').iloc[0]
out = [float(p.high - p.low)]''',
    "fsc-(T3)Complex_Historical_Investigation_213": _one("(" + _sec(104169, "Goodwill", "2024-01-31", None, "0000104169-24-000056") + " - " + _sec(104169, "Goodwill", "2022-01-31", None, "0000104169-22-000012") + ") / 1e6"),
    "fsc-(T3)Complex_Historical_Investigation_214": _one("(" + _sec(1018724, "NetCashProvidedByUsedInFinancingActivities", "2024-12-31", "2024-01-01", "0001018724-25-000004") + " - " + _sec(1018724, "NetCashProvidedByUsedInFinancingActivities", "2020-12-31", "2020-01-01", "0001018724-21-000004") + ") / 1e6"),
    "fsc-(T3)Complex_Historical_Investigation_216": '''
ch = []
for s in ('^NDX', '^GSPC', '^DJI'):
    c = closes(s, '2024-12-30', '2024-12-31', 'raw')
    ch.append(pct(c.iloc[1], c.iloc[0]))
out = [max(ch) - min(ch)]''',
    "fsc-(T3)Complex_Historical_Investigation_220": '''
v = float(data.get_prices('PFE', '2025-01-24', '2025-01-24').volume.iloc[0])
f = facts(78003, 'dei:EntityCommonStockSharesOutstanding')
f = f[f.filed <= '2025-01-24']
sh = float(f[f.filed == f.filed.max()].value.iloc[0])
out = [v / sh * 100]''',
    "fsc-(T3)Complex_Historical_Investigation_221": '''
filings = {2019: '0001564590-19-023119', 2020: '0001564590-20-030125', 2021: '0001564590-21-033616',
           2022: '0001564590-22-023675', 2023: '0000950170-23-028914'}
a = {y: fact(1341439, 'Assets', '%d-05-31' % y, None, accn) for y, accn in filings.items()}
out = [pct(a[y], a[y - 1]) for y in (2020, 2021, 2022, 2023)]''',
    "fsc-(T3)Complex_Historical_Investigation_224": '''
lly = {2022: '0000059478-23-000082', 2023: '0000059478-24-000065', 2024: '0000059478-25-000067'}
goog = {2022: '0001652044-23-000016', 2023: '0001652044-24-000022', 2024: '0001652044-25-000014'}
out = []
for y in (2022, 2023, 2024):
    e = '%d-12-31' % y
    la = fact(59478, 'Assets', e, None, lly[y])
    ll = fact(59478, 'LiabilitiesAndStockholdersEquity', e, None, lly[y]) - fact(59478, 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', e, None, lly[y])
    ga, gl = fact(1652044, 'Assets', e, None, goog[y]), fact(1652044, 'Liabilities', e, None, goog[y])
    out.append((ll / la - gl / ga) * 100)''',
    "fsc-(T3)Complex_Historical_Investigation_225": '''
c = closes('BTC-USD', '2022-12-31', '2025-05-20', 'raw')
up = c.diff() > 0
d = list(c.index)
hits = []
for i in range(1, len(c) - 4):
    if d[i] < '2023-01-01':
        continue
    if all(up.iloc[i:i + 5]) and c.iloc[i + 4] / c.iloc[i - 1] - 1 > 0.20:
        hits.append((d[i], d[i + 4]))
out = [x for h in hits for x in h]''',
    "fsc-(T3)Complex_Historical_Investigation_227": '''
c = closes('^GSPC', '2022-03-16', '2022-06-16', 'raw')
peak = c.cummax()
dd = c / peak - 1
trough = dd.idxmin()
pk = c[:trough].idxmax()
out = [pk, trough, float(-dd.min() * 100)]''',
    "fsc-(T3)Complex_Historical_Investigation_228": '''
r = []
for s in ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'):
    a = bar_le(s, '2022-12-30', 'raw')[1].close
    b = bar_le(s, '2023-12-29', 'raw')[1].close
    r.append(pct(b, a))
out = [max(r) - min(r)]''',
    "fsc-(T3)Complex_Historical_Investigation_229": '''
end = data.trading_day('TSLA', '2024-11-05', offset=10)
p = data.get_prices('TSLA', '2024-11-06', end)
out = [float(p.high.max())]''',
    "fsc-(T3)Complex_Historical_Investigation_231": '''
p = data.get_prices('NVDA', '2024-12-31', '2025-03-31')
c = p.close.pct_change() * 100
q = p[p.index >= '2025-01-01']
out = [float(c[q.volume.idxmax()])]''',
    "fsc-(T3)Complex_Historical_Investigation_233": '''
v = data.get_prices('TSLA', '2025-04-01', '2025-04-30').volume
out = [200000.0 + 10000 * (int((v < 80e6).sum()) - int((v > 120e6).sum()))]''',
    "fsc-(T3)Complex_Historical_Investigation_239": '''
c = closes('^N225', '2024-04-01', '2025-04-30', 'raw')
m = c.groupby(c.index.str[:7]).last()
r = m.pct_change()
out = [[k for k, x in r.items() if k >= '2024-05' and x > 0]]''',
    "fsc-(T3)Complex_Historical_Investigation_240": '''
h = closes('^KS11', '2020-01-01', '2024-12-31', 'raw', 'high')
y = h.groupby(h.index.str[:4]).max()
out = [y.idxmax(), float(y.max())]''',
    "fsc-(T3)Complex_Historical_Investigation_b004": '''
m = {'TSLA': (1318605, 'Revenues', '2021-12-31', '2021-01-01'),
     'NVDA': (1045810, 'Revenues', '2021-01-31', '2020-01-27'),
     'WFC': (72971, 'RevenuesNetOfInterestExpense', '2021-12-31', '2021-01-01'),
     'META': (1326801, 'RevenueFromContractWithCustomerExcludingAssessedTax', '2021-12-31', '2021-01-01'),
     'AAPL': (320193, 'RevenueFromContractWithCustomerExcludingAssessedTax', '2021-09-25', '2020-09-27')}
def first(cik, tag, e, s):
    f = facts(cik, tag)
    f = f[(f.end == e) & (f.start == s) & (f.form == '10-K')].sort_values('filed')
    return float(f.value.iloc[0])
npm = {k: first(c, 'NetIncomeLoss', e, s) / first(c, t, e, s) * 100 for k, (c, t, e, s) in m.items()}
hi, lo = max(npm, key=npm.get), min(npm, key=npm.get)
out = [hi, npm[hi], lo, npm[lo]]''',
    "fsc-(T3)Complex_Historical_Investigation_b005": '''
def first(cik, tag, e, s):
    f = facts(cik, tag)
    f = f[(f.end == e) & (f.start == s) & (f.form == '10-K')].sort_values('filed')
    return float(f.value.iloc[0])
aapl = [('2021-09-25', '2020-09-27'), ('2022-09-24', '2021-09-26'), ('2023-09-30', '2022-09-25'), ('2024-09-28', '2023-10-01')]
nvda = [('2021-01-31', '2020-01-27'), ('2022-01-30', '2021-02-01'), ('2023-01-29', '2022-01-31'), ('2024-01-28', '2023-01-30')]
out = []
for (ea, sa), (en, sn) in zip(aapl, nvda):
    ga = first(320193, 'GrossProfit', ea, sa) / first(320193, 'RevenueFromContractWithCustomerExcludingAssessedTax', ea, sa)
    gn = first(1045810, 'GrossProfit', en, sn) / first(1045810, 'Revenues', en, sn)
    out.append((gn - ga) * 100)''',
    "fsc-(T3)Complex_Historical_Investigation_b006": '''
m = {'TSLA': (1318605, '2024-12-31', '0001628280-25-003063'), 'NVDA': (1045810, '2024-01-28', '0001045810-24-000029'),
     'WFC': (72971, '2024-12-31', '0000072971-25-000066'), 'META': (1326801, '2024-12-31', '0001326801-25-000017'),
     'AAPL': (320193, '2024-09-28', '0000320193-24-000123')}
r = {k: fact(c, 'Liabilities', e, None, a) / fact(c, 'Assets', e, None, a) * 100 for k, (c, e, a) in m.items()}
top = sorted(r, key=r.get, reverse=True)[:3]
out = [x for k in top for x in (k, r[k])]''',
    "fsc-(T3)Complex_Historical_Investigation_b007": '''
def first(cik, e, s):
    f = facts(cik, 'NetIncomeLoss')
    f = f[(f.end == e) & (f.start == s) & (f.form == '10-K')].sort_values('filed')
    return float(f.value.iloc[0])
wfc = [('2021-12-31', '2021-01-01'), ('2022-12-31', '2022-01-01'), ('2023-12-31', '2023-01-01'), ('2024-12-31', '2024-01-01')]
nvda = [('2021-01-31', '2020-01-27'), ('2022-01-30', '2021-02-01'), ('2023-01-29', '2022-01-31'), ('2024-01-28', '2023-01-30')]
out = [first(72971, *w) / first(1045810, *n) for w, n in zip(wfc, nvda)]''',
    "fsc-(T3)Complex_Historical_Investigation_b008": '''
def first(cik, tag, e, s=None):
    f = facts(cik, tag)
    f = f[(f.end == e) & ((f.start == s) if s else f.start.isna()) & (f.form == '10-K')].sort_values('filed')
    return float(f.value.iloc[0])
def turnover(cik, cogs, e, s, e0):
    return first(cik, cogs, e, s) / ((first(cik, 'InventoryNet', e) + first(cik, 'InventoryNet', e0)) / 2)
aapl = [('2021-09-25', '2020-09-27', '2020-09-26'), ('2022-09-24', '2021-09-26', '2021-09-25'),
        ('2023-09-30', '2022-09-25', '2022-09-24'), ('2024-09-28', '2023-10-01', '2023-09-30')]
tsla = [('%d-12-31' % y, '%d-01-01' % y, '%d-12-31' % (y - 1)) for y in (2021, 2022, 2023, 2024)]
out = [turnover(320193, 'CostOfGoodsAndServicesSold', *a) / turnover(1318605, 'CostOfRevenue', *t) for a, t in zip(aapl, tsla)]''',
    "fsc-(T3)Complex_Historical_Investigation_b012": '''
r = {}
for s in ('MRK', 'TSM', 'BA', 'IBM', 'GS'):
    c = closes(s, '2025-03-31', '2025-04-30', 'raw')
    r[s] = pct(c.iloc[-1], c.iloc[0])
out = [x for s in r if r[s] > 0 for x in (s, r[s])]''',
    "fsc-(T3)Complex_Historical_Investigation_b014": '''
p = data.get_prices('BA', '2025-04-01', '2025-04-30')
out = [float(p.high.max()), pct(p.high.max(), p.low.min())]''',
    "fsc-(T3)Complex_Historical_Investigation_b017": '''
t = float(data.get_prices('TSLA', '2025-05-23', '2025-05-23').close.iloc[0])
a = float(data.get_prices('AMZN', '2025-05-23', '2025-05-23').close.iloc[0])
out = [t, t - a]''',
    "fsc-(T3)Complex_Historical_Investigation_b020": '''
c = closes('AAL', '2025-05-16', '2025-05-23', 'raw')
out = [float(abs(c.iloc[-1] - c.iloc[0]))]''',
    "fsc-(T3)Complex_Historical_Investigation_b030": '''
r = {}
for name, s in (('Nasdaq', '^IXIC'), ('S&P 500', '^GSPC')):
    c = closes(s, '2023-12-29', '2024-12-31', 'raw')
    r[name] = pct(c.iloc[-1], c.iloc[0])
out = [max(r, key=r.get), abs(r['Nasdaq'] - r['S&P 500'])]''',
}
