"""Hard-tier task templates (difficulty 4 and 5) for the offline market environment.

Version 1 (research/log/2026-10-31-0150.md, the hard tier): on the 17 templates of templates.py (difficulty 1-3)
the teacher passes 339/340 dev tasks, so these templates compose several of the operations those templates use
singly: a cross-section over 8-12 stocks, rolling regressions, moving-average signals and a backtest, a manual
dividend-reinvestment ledger with splits and a currency, an event study around ex-dividend dates, Q4 values derived
from 10-K minus nine-month 10-Q across companies with different fiscal years, and three-hop chains (pick by one
criterion, then look up or compute in another source). Difficulty 5 adds conventions (risk-free rate, total
returns, a pinned FRED vintage, names without tickers, more keys) to the difficulty-4 form.

Version 2 (research/log/2026-10-31-0150.md, the hardening after the teacher's dev calibration): rolling_beta,
ma_cross, reinvest_return, div_event_study and q4_margin_rank, which the teacher solved at >= 95%, name every company
without its ticker at difficulty 4 too, name the measure by its finance term instead of spelling out its formula
(keeping only the conventions that make the answer unique: price series, window, benchmark, date rules, sources,
rounding), and ask for more keys: both beta extremes (and at 5 the mean beta), the next golden cross (at 5 the best
round trip's buy date and return), the dividend cash reinvested, the highest event (at 5 the lowest too), and the Q4
revenue and the runner-up's margin. The other three templates are unchanged; each template draws from its own rng.

Each generator follows templates.py: it draws one candidate from the seeded rng, raises Skip(<rule>) when a quality
rule fails (rounding boundary, near ties and near-threshold days, missing or conflicting facts, calendar mismatch),
and returns mk(...) with `params` that fully define the computation, recomputed independently by verify.py.
Entities follow the existing split rules: equities by ticker hash (any date), and whenever an index, ETF or FRED
series takes part, every date used lies inside the split's time window.
"""
import numpy as np

from common import (CUTOFF, GLOBAL_START, SPLIT_WINDOWS, Skip, count_key, date_key, fmt_date, month_label, num_key,
                    round_ok)
from templates import (NT_RULE, PCT_SIGNED, REV_PREC, TOTAL_DESC, cal_date_near, mk, month_ends, need_rule,
                       np_date, pick_entity, pick_span)

SECTOR_ETF = {"Information Technology": "XLK", "Financials": "XLF", "Energy": "XLE", "Health Care": "XLV"}
FX_HOME = {"DEXUSEU": ("euros", "US dollars per euro", False), "DEXUSUK": ("British pounds", "US dollars per British pound", False),
           "DEXJPUS": ("Japanese yen", "Japanese yen per US dollar", True), "DEXSZUS": ("Swiss francs", "Swiss francs per US dollar", True),
           "DEXCAUS": ("Canadian dollars", "Canadian dollars per US dollar", True),
           "DEXUSAL": ("Australian dollars", "US dollars per Australian dollar", False)}
MARGINS = {"operating": ("operating margin", "OperatingIncomeLoss", "operating income (us-gaap:OperatingIncomeLoss)"),
           "net": ("net margin", "NetIncomeLoss", "net income attributable to the company (us-gaap:NetIncomeLoss)"),
           "gross": ("gross margin", "GrossProfit", "gross profit (us-gaap:GrossProfit)")}
REV_RULE = ("revenue is us-gaap:Revenues if that filing reports it for the period, otherwise "
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax, otherwise us-gaap:SalesRevenueNet")
SEC_TIE = 0.02  # percentage points between the chosen company and the runner-up


def same_calendar(px, ref, i0, i1):
    """Indices (a, b) of px for ref's trading days i0..i1; Skip unless px trades on exactly those days."""
    a, b = px.i_le(ref.d[i0]), px.i_le(ref.d[i1])
    if a < 0 or px.d[a] != ref.d[i0] or px.d[b] != ref.d[i1] or not np.array_equal(px.d[a:b + 1], ref.d[i0:i1 + 1]):
        raise Skip("calendar_mismatch")
    if not px.clean(a, b):
        raise Skip("bad_row")
    return a, b


def name_of(C, sym, names_only):
    if names_only:
        C.name_ok(sym, 3)
    return C.ent(sym, 3 if names_only else 2)


def fred_near(F, s, lo, max_gap=7):
    """(date, value) of a FRED series on s or its most recent earlier observation, at most max_gap days earlier."""
    od, v = F.le(s)
    if (np_date(s) - np_date(od)).days > max_gap:
        raise Skip("missing_row")
    if od < lo:
        raise Skip("outside_split_window")
    return od, v


def pct(a, b):
    return (a / b - 1.0) * 100.0


def days_txt(n):
    return "1 trading day" if n == 1 else "%d trading days" % n


# ============================================================================
# Cross-section of risk-adjusted returns

RISK_DESC = {
    "sharpe": ("annualized Sharpe ratio", "the mean of the daily {r} divided by their sample standard deviation "
               "(n-1 denominator), times the square root of 252"),
    "sortino": ("annualized Sortino ratio", "the mean of the daily {r} divided by their downside deviation, times the "
                "square root of 252, where the downside deviation is the square root of the mean, over all days of "
                "the window, of min(daily {r1}, 0) squared"),
}


def risk_value(e, metric):
    if metric == "sharpe":
        return float(e.mean() / e.std(ddof=1) * np.sqrt(252))
    return float(e.mean() / np.sqrt(np.mean(np.minimum(e, 0.0) ** 2)) * np.sqrt(252))


def gen_risk_rank(C, split, rng, d):
    K, rf = (8, False) if d == 4 else (12, True)
    lo, hi = SPLIT_WINDOWS[split] if rf else (GLOBAL_START, CUTOFF)
    syms = rng.sample(C.eq[split], K)
    names = [name_of(C, s, d == 5) for s in syms]
    metric = rng.choice(["sharpe", "sortino"])
    ref = C.S.px["^GSPC"]
    L = rng.randint(250, 750)
    i0, i1 = pick_span(rng, ref, max(lo, max(C.S.px[s].d[0] for s in syms)), hi, L, lookback=1)
    days = ref.d[i0 + 1:i1 + 1]
    rfd = np.zeros(len(days))
    if rf:
        F = C.S.fred["DTB3"]
        rfd = np.array([fred_near(F, str(t), lo)[1] for t in days]) / 100.0 / 252.0
    vals = []
    for s in syms:
        px = C.S.px[s]
        a, b = same_calendar(px, ref, i0, i1)
        c = px.adj_close[a:b + 1]
        if c.min() <= 0:
            raise Skip("bad_row")
        vals.append(risk_value(c[1:] / c[:-1] - 1 - rfd, metric))
    vals = np.array(vals)
    order = np.argsort(-vals)
    ask = rng.choice(["highest", "lowest"]) if d == 4 else "both"
    picks = {"highest": [order[0]], "lowest": [order[-1]], "both": [order[0], order[-1]]}[ask]
    if ask in ("highest", "both") and vals[order[0]] - vals[order[1]] < 0.02:
        raise Skip("near_tie")
    if ask in ("lowest", "both") and vals[order[-2]] - vals[order[-1]] < 0.02:
        raise Skip("near_tie")
    keys = []
    for k in picks:
        if not round_ok(vals[k], 0.01):
            raise Skip("rounding_boundary")
        keys += [C.ticker_key(syms[k]), num_key(vals[k], 0.01)]
    S0, E0 = str(ref.d[i0]), str(ref.d[i1])
    sD, nt0 = cal_date_near(rng, ref, i0, 0.5 if d == 5 else 0.0)
    eD, nt1 = cal_date_near(rng, ref, i1, 0.5 if d == 5 else 0.0)
    need_rule(nt0 or nt1, d == 5)
    title, defn = RISK_DESC[metric]
    rname = "excess returns" if rf else "total returns"
    defn = defn.format(r=rname, r1=rname[:-1])
    ret_def = ("A stock's daily total return is its close adjusted for splits and dividends divided by the previous "
               "trading day's adjusted close, minus 1, for every trading day after the start date through the end "
               "date.")
    if rf:
        ret_def += (" Its daily excess return is that total return minus the daily risk-free rate, which is the "
                    "3-month Treasury bill rate (FRED series DTB3, in percent) on that trading day, or its most recent "
                    "earlier observation, divided by 100 and by 252.")
    else:
        ret_def += " Take the risk-free rate as zero."
    q = {"highest": "Which stock had the highest %s, and what was it?" % title,
         "lowest": "Which stock had the lowest %s, and what was it?" % title,
         "both": "Which stock had the highest %s and which the lowest, and what were their values?" % title}[ask]
    prompt = rng.choice([
        "Compare the %s of these %d stocks from the close on %s to the close on %s: %s. %s The %s is %s. %s Round "
        "each ratio to two decimals." % (title, K, fmt_date(sD, rng), fmt_date(eD, rng), "; ".join(names), ret_def,
                                         title, defn, q),
        "%s %s Over the period from the close on %s to the close on %s, with the %s defined as %s, rank the "
        "following stocks: %s. Give ratios rounded to two decimals." % (ret_def, q, fmt_date(sD, rng),
                                                                       fmt_date(eD, rng), title, defn, "; ".join(names)),
    ])
    if d == 5:
        prompt += " " + NT_RULE
    desc = ", ".join("%s %.3f" % (s, v) for s, v in zip(syms, vals))
    return mk("risk_rank", split, d, prompt, keys,
              "%s: %s. All: %s." % (title, "; ".join("%s %s %.2f" % ("highest" if k == order[0] else "lowest",
                                                                    syms[k], vals[k]) for k in picks), desc),
              syms + (["DTB3"] if rf else []), S0, E0,
              ["price_range"] * K + ["daily_return", "dividend_adjust"] + (["fred_range", "align"] if rf else []) +
              ["mean", "std" if metric == "sharpe" else "downside_dev", "annualize", "rank"],
              K * (L + 1) + (L if rf else 0), "%s of daily %s per stock, %s" % (metric, rname, ask),
              {"symbols": syms, "start": sD, "end": eD, "metric": metric, "rf": rf, "ask": ask})


# ============================================================================
# Rolling beta against a benchmark

def gen_rolling_beta(C, split, rng, d):
    sym = rng.choice(C.eq[split])
    ent = name_of(C, sym, True)
    lo, hi = SPLIT_WINDOWS[split]
    if d == 4:
        bench = "^GSPC"
    else:
        bench = SECTOR_ETF.get(C.sector.get(sym)) if rng.random() < 0.6 else None
        bench = bench or rng.choice(["^GSPC", "SPY", "QQQ", "IWM", "^NDX", "^DJI"])
    W = rng.choice([60, 120]) if d == 4 else rng.choice([60, 90, 120, 250])
    total = d == 5
    pb, px = C.S.px[bench], C.S.px[sym]
    L = rng.randint(120, 500)
    i0, i1 = pick_span(rng, pb, max(lo, px.d[0]), hi, L, lookback=W)
    if not pb.clean(i0 - W, i1):
        raise Skip("bad_row")
    a, b = same_calendar(px, pb, i0 - W, i1)
    cb = (pb.adj_close if total else pb.close)[i0 - W:i1 + 1]
    cs = (px.adj_close if total else px.close)[a:b + 1]
    if cb.min() <= 0 or cs.min() <= 0:
        raise Skip("bad_row")
    rm, rs = cb[1:] / cb[:-1] - 1, cs[1:] / cs[:-1] - 1
    wm = np.lib.stride_tricks.sliding_window_view(rm, W)
    ws = np.lib.stride_tricks.sliding_window_view(rs, W)
    dm, ds = wm - wm.mean(axis=1, keepdims=True), ws - ws.mean(axis=1, keepdims=True)
    beta = (dm * ds).sum(axis=1) / (dm * dm).sum(axis=1)  # the n-1 denominators cancel
    assert len(beta) == i1 - i0 + 1
    ask, mean = "both", float(beta.mean())
    order = np.argsort(-beta)
    picks = [order[0], order[-1]]
    if beta[order[0]] - beta[order[1]] < 2e-4 or beta[order[-2]] - beta[order[-1]] < 2e-4:
        raise Skip("near_tie")
    if {0, len(beta) - 1} & {int(k) for k in picks}:
        raise Skip("extreme_on_named_date")  # a date the prompt states would match the date key by chance
    keys = []
    for k in picks:
        if not round_ok(beta[k], 0.001):
            raise Skip("rounding_boundary")
        keys += [date_key(str(pb.d[i0 + k])), num_key(beta[k], 0.001)]
    if d == 5:
        if not round_ok(mean, 0.001):
            raise Skip("rounding_boundary")
        keys.append(num_key(mean, 0.001))
    S0, E0 = str(pb.d[i0]), str(pb.d[i1])
    bent = C.ent(bench, 2)
    # Version 2: the beta is named, not defined (covariance over variance does not depend on the ddof); the stated
    # conventions are the price series, simple returns and a window of W returns ending on the day.
    rdef = ("simple daily total returns of closes adjusted for splits and dividends" if total else
            "simple daily returns of split-adjusted closes") + " (closing levels for an index)"
    wdef = "each day's beta uses the %d daily returns ending on that day, reaching back before %s where needed" % (W, S0)
    q = "On which trading days was the beta highest and lowest, and what were those betas?"
    if d == 5:
        q += " Also give the average of the daily betas over those trading days."
    prompt = rng.choice([
        "Compute the rolling %d-trading-day beta of %s against %s on every trading day from %s to %s inclusive, from "
        "%s; %s. %s Give dates as YYYY-MM-DD and betas rounded to three decimals." % (W, ent, bent, S0, E0, rdef, wdef,
                                                                                     q),
        "For %s, with %s as the benchmark, take the rolling %d-trading-day beta on each trading day from %s through "
        "%s, using %s; %s. %s Report dates (YYYY-MM-DD) and betas to three decimals." % (ent, bent, W, S0, E0, rdef,
                                                                                        wdef, q),
    ])
    return mk("rolling_beta", split, d, prompt, keys,
              "Rolling %d-day beta of %s vs %s: %s%s." % (W, sym, bench, "; ".join(
                  "%s %s %.3f" % ("max" if k == order[0] else "min", pb.d[i0 + k], beta[k]) for k in picks),
                  "; mean %.3f" % mean if d == 5 else ""),
              [sym, bench], str(pb.d[i0 - W]), E0,
              ["price_range", "price_range", "daily_return", "align", "rolling_cov", "rolling_var", "argmax", "argmin"]
              + (["mean"] if d == 5 else []),
              2 * (L + W), "rolling(%d) cov/var, max and min%s" % (W, ", mean" if d == 5 else ""),
              {"symbol": sym, "bench": bench, "start": S0, "end": E0, "window": W, "total": total, "ask": ask,
               "ask_mean": d == 5})


# ============================================================================
# Moving-average crossover signals

def sma(c, n):
    cs = np.r_[0.0, np.cumsum(c)]
    out = np.full(len(c), np.nan)
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def crosses(s, l, j):
    """+1 for a golden cross on index j, -1 for a death cross, 0 otherwise."""
    if s[j] > l[j] and s[j - 1] <= l[j - 1]:
        return 1
    if s[j] < l[j] and s[j - 1] >= l[j - 1]:
        return -1
    return 0


def gen_ma_cross(C, split, rng, d):
    sym, lo, hi = pick_entity(C, rng, split, 0.6, ("index", "etf"))
    ent = name_of(C, sym, C.kind(sym) == "equity")
    px = C.S.px[sym]
    ns, nl = (50, 200) if d == 4 else rng.choice([(20, 100), (50, 200), (10, 50), (20, 50)])
    a, b = px.i_ge(lo), px.i_le(hi)
    if b - a < nl + 400:
        raise Skip("short_history")
    c = px.close[a:b + 1]
    s, l = sma(c, ns), sma(c, nl)
    adj = px.kind in ("equity", "etf")
    series = "split-adjusted daily closes" if adj else "daily closing levels"
    # Version 2: the crosses are named, not defined by inequalities; the stated conventions are the price series, an
    # average that includes its day over the full history, and the day a cross is dated (ties never occur: the gap
    # between the averages is at least 1e-5 of the longer one on every day used).
    defs = ("with the %d- and %d-day simple moving averages of its %s (each average includes its own day and uses "
            "the full price history, closes before any date named here included), and a golden or death cross dated "
            "on the first trading day on which the %d-day average is above or below the %d-day average" % (
                ns, nl, series, ns, nl))
    if d == 4:
        i = rng.randint(nl, b - a - 250)
        D, _ = cal_date_near(rng, px, a + i, 0.5)
        g = next((j for j in range(i + 1, min(i + 750, b - a + 1)) if crosses(s, l, j) == 1), None)
        if g is None:
            raise Skip("empty_answer")
        x = next((j for j in range(g + 1, b - a + 1) if crosses(s, l, j) == -1), None)
        if x is None:
            raise Skip("empty_answer")
        g2 = next((j for j in range(x + 1, b - a + 1) if crosses(s, l, j) == 1), None)
        if g2 is None:
            raise Skip("empty_answer")
        j0, j1 = i, g2
        if not px.clean(a + i + 1 - nl, a + g2):
            raise Skip("bad_row")
        ret = pct(c[x], c[g])
        if not round_ok(ret, 0.01):
            raise Skip("rounding_boundary")
        keys = [date_key(str(px.d[a + g])), date_key(str(px.d[a + x])), date_key(str(px.d[a + g2])),
                num_key(ret, 0.01)]
        prompt = rng.choice([
            "For %s, %s: find the first golden cross on a trading day after %s, the first death cross after it, and "
            "the next golden cross after that death cross. Give the three dates (YYYY-MM-DD) and the price return "
            "from the close on the first golden-cross day to the close on the death-cross day %s." % (
                ent, defs, fmt_date(D, rng), PCT_SIGNED),
            "Take %s, %s. What are the dates of the first golden cross after %s, of the first death cross that "
            "follows it and of the golden cross after that, and the price return from the first golden-cross close "
            "to the death-cross close? Dates as YYYY-MM-DD; the return %s." % (ent, defs, fmt_date(D, rng), PCT_SIGNED),
        ])
        ans = "golden %s, death %s, next golden %s, return %.2f%%" % (px.d[a + g], px.d[a + x], px.d[a + g2], ret)
        params = {"symbol": sym, "short": ns, "long": nl, "mode": "next", "date": D}
        S0, E0 = str(px.d[a + i + 1 - nl]), str(px.d[a + g2])
    else:
        L = rng.randint(500, 1500)
        i0, i1 = pick_span(rng, px, lo, hi, L, lookback=nl)
        j0, j1 = i0 - a, i1 - a
        if not px.clean(i0 - nl, i1):
            raise Skip("bad_row")
        held, buys, growth, trades = None, 0, 1.0, []
        for j in range(j0, j1 + 1):
            sig = crosses(s, l, j)
            if held is None and sig == 1:
                held, buys = j, buys + 1
            elif held is not None and sig == -1:
                growth *= c[j] / c[held]
                trades.append((held, j))
                held = None
        if held is not None:
            growth *= c[j1] / c[held]
            trades.append((held, j1))
        if not 2 <= buys <= 10:
            raise Skip("empty_answer" if buys == 0 else "degenerate_value")
        strat, bh = (growth - 1) * 100, pct(c[j1], c[j0])
        tr = sorted(((pct(c[q], c[p]), p) for p, q in trades), reverse=True)
        if tr[0][0] - tr[1][0] < 0.02:
            raise Skip("near_tie")
        best, best_buy = tr[0]
        for v in (strat, bh, best):
            if not round_ok(v, 0.01):
                raise Skip("rounding_boundary")
        keys = [count_key(buys), num_key(strat, 0.01), num_key(bh, 0.01), date_key(str(px.d[a + best_buy])),
                num_key(best, 0.01)]
        S0, E0 = str(px.d[i0]), str(px.d[i1])
        rule = ("backtest a long-only crossover rule from %s to %s inclusive: start in cash, buy at the close of each "
                "golden-cross day and sell at the close of each death-cross day in that period, and sell an open "
                "position at the close on %s; no costs, no dividends" % (S0, E0, E0))
        q = ("the number of buys, the rule's compounded return, the buy-and-hold price return from the close on %s to "
             "the close on %s, and the buy date (YYYY-MM-DD) and price return of the rule's most profitable round trip "
             "(a position sold at the end counts as one)" % (S0, E0))
        prompt = rng.choice([
            "For %s, %s, %s. Report %s; returns %s." % (ent, defs, rule, q, PCT_SIGNED),
            "Take %s, %s, and %s. Give %s, each return %s." % (ent, defs, rule, q, PCT_SIGNED),
        ])
        ans = "%d buys %s; strategy %.2f%%, buy and hold %.2f%%; best trade bought %s, %.2f%%" % (
            buys, ", ".join("%s->%s" % (px.d[a + p], px.d[a + q]) for p, q in trades), strat, bh,
            px.d[a + best_buy], best)
        params = {"symbol": sym, "short": ns, "long": nl, "mode": "backtest", "start": S0, "end": E0}
        S0 = str(px.d[i0 - nl])  # the first close the moving averages use
        j0 = j0 - 1
    gap = np.abs(s[j0:j1 + 1] - l[j0:j1 + 1]) / l[j0:j1 + 1]
    if not np.all(np.isfinite(gap)) or gap.min() < 1e-5:
        raise Skip("near_threshold")
    return mk("ma_cross", split, d, prompt, keys, "SMA%d/SMA%d of %s: %s." % (ns, nl, sym, ans),
              [sym], S0, E0, ["price_range", "rolling_mean", "rolling_mean", "signal_scan"] +
              (["backtest", "compound", "argmax"] if d == 5 else ["pct_change"]), j1 - j0 + nl,
              "SMA%d vs SMA%d crossings, %s" % (ns, nl, params["mode"]), params)


# ============================================================================
# Total return with manual dividend reinvestment, splits and (difficulty 5) a home currency

def gen_reinvest_return(C, split, rng, d):
    sym = rng.choice(C.eq[split])
    ent = name_of(C, sym, True)
    fx = d == 5
    lo, hi = SPLIT_WINDOWS[split] if fx else (GLOBAL_START, CUTOFF)
    if sym not in C.S.divs:
        raise Skip("empty_answer")
    px = C.S.px[sym]
    L = rng.randint(250, 1250)
    a, b = px.i_ge(lo), px.i_le(hi)
    true_sp = [x for x in C.true_splits.get(sym, []) if px.d[a] < x[0] <= px.d[b]]
    if true_sp and rng.random() < 0.8:
        k = px.i_ge(rng.choice(true_sp)[0])
        if max(a, k - L + 1) > k - 1:
            raise Skip("short_history")
        i0 = rng.randint(max(a, k - L + 1), k - 1)
        i1 = min(b, i0 + L - 1)
    else:
        i0, i1 = pick_span(rng, px, lo, hi, L)
    if not px.clean(i0, i1) or px.raw_close[i0:i1 + 1].min() <= 0:
        raise Skip("bad_row")
    S0, E0 = str(px.d[i0]), str(px.d[i1])
    sp = [x for x in C.S.splits.get(sym, []) if S0 < x[0] <= E0]
    if any(x[3] not in ("split", "reverse_split") for x in sp):
        raise Skip("ambiguous_event")
    dd, _, dr = C.S.divs[sym]
    m = (dd > S0) & (dd <= E0)
    divs = list(zip(dd[m], dr[m]))
    if len(divs) < 2:
        raise Skip("empty_answer")
    sp_days = {x[0] for x in sp}
    if any(x in sp_days for x, _ in divs):
        raise Skip("ambiguous_event")
    events = sorted([(x[0], "split", x[1]) for x in sp] + [(x, "div", v) for x, v in divs])
    shares, cash = 1.0, 0.0
    for x, kind, v in events:
        if not px.is_td(x) or not np.isfinite(v) or v <= 0:
            raise Skip("missing_row")
        if kind == "split":
            shares *= v
        else:
            cash += shares * v
            shares += shares * v / px.raw_close[px.i_le(x)]
    usd = (shares * px.raw_close[i1] / px.raw_close[i0] - 1) * 100
    keys = [num_key(shares, 0.0001), num_key(cash, 0.01), num_key(usd, 0.01)]
    if not round_ok(shares, 0.0001) or not round_ok(cash, 0.01) or not round_ok(usd, 0.01):
        raise Skip("rounding_boundary")
    sD, nt0 = cal_date_near(rng, px, i0, 0.4)
    eD, nt1 = cal_date_near(rng, px, i1, 0.4)
    need_rule(nt0 or nt1, True)
    # Version 2: the ledger and the total return are named, not spelled out; the stated conventions are unadjusted
    # prices, the dividend as paid, reinvestment at the ex-date's close, which ex-dates count, and splits.
    ledger = ("An investor buys one share of %s at the unadjusted close on %s and holds the position through the "
              "unadjusted close on %s, reinvesting dividends by hand: each cash dividend with an ex-dividend date "
              "after the purchase day, up to and including the last day, is reinvested at the unadjusted close of its "
              "ex-dividend date (fractional shares), using the amount per share as paid at the time, and each split "
              "multiplies the share count by its ratio (new shares per old share). %s" % (ent, fmt_date(sD, rng), fmt_date(eD, rng), NT_RULE))
    if not fx:
        prompt = ledger + rng.choice([
            " How many shares does the investor hold at the end (four decimals), how much dividend cash was "
            "reinvested in total (US dollars, two decimals), and what is the position's total return %s?" % PCT_SIGNED,
            " Report the final share count rounded to four decimals, the total dividend cash reinvested in US dollars "
            "rounded to two decimals, and the position's total return %s." % PCT_SIGNED,
        ])
        ans = "%s: %d dividends, %d splits; %.4f shares; dividends reinvested %.2f USD; total return %.2f%%" % (
            sym, len(divs), len(sp), shares, cash, usd)
        params = {"symbol": sym, "start": sD, "end": eD, "fx": None}
        ents = [sym]
    else:
        sid = rng.choice(sorted(FX_HOME))
        home, qdesc, per_usd = FX_HOME[sid]
        F = C.S.fred[sid]
        f0 = fred_near(F, S0, lo)[1]
        f1 = fred_near(F, E0, lo)[1]
        conv = (lambda x, f: x * f) if per_usd else (lambda x, f: x / f)
        loc = (conv(shares * px.raw_close[i1], f1) / conv(px.raw_close[i0], f0) - 1) * 100
        if abs(loc - usd) < 0.05:
            raise Skip("fx_irrelevant")
        if not round_ok(loc, 0.01):
            raise Skip("rounding_boundary")
        keys.append(num_key(loc, 0.01))
        prompt = ledger + rng.choice([
            " The investor counts in %s: convert the purchase cost and the final value with FRED series %s (%s), using "
            "its observation on the purchase day and on the last day (or the most recent earlier observation). Give the "
            "final share count (four decimals), the total dividend cash reinvested in US dollars (two decimals), the "
            "total return in US dollars and the total return in %s, each return %s." % (home, sid, qdesc, home,
                                                                                         PCT_SIGNED),
            " Measured in %s (translate the cost on the purchase day and the final value on the last day with FRED %s, "
            "%s, taking the observation of that date or the most recent earlier one), what are the final number of "
            "shares (four decimals), the total dividend cash reinvested in US dollars (two decimals), the US-dollar "
            "total return and the %s total return? Returns %s." % (home, sid, qdesc, home, PCT_SIGNED),
        ])
        ans = "%s: %d dividends, %d splits; %.4f shares; dividends reinvested %.2f USD; USD %.2f%%, %s %s->%s: %.2f%%" % (
            sym, len(divs), len(sp), shares, cash, usd, sid, f0, f1, loc)
        params = {"symbol": sym, "start": sD, "end": eD, "fx": sid, "per_usd": per_usd}
        ents = [sym, sid]
    return mk("reinvest_return", split, d, prompt, keys, ans, ents, S0, E0,
              ["calendar_resolve", "action_range", "price_lookup", "split_adjust", "reinvest_ledger", "pct_change"] +
              (["fred_lookup", "fx_convert"] if fx else []), len(events) + 2,
              "1 share, splits multiply, dividends buy at the raw close of the ex-date", params)


# ============================================================================
# Event study around ex-dividend dates

def gen_div_event_study(C, split, rng, d):
    sym = rng.choice(C.eq[split])
    ent = name_of(C, sym, True)
    lo, hi = SPLIT_WINDOWS[split]
    if sym not in C.S.divs:
        raise Skip("empty_answer")
    px, g = C.S.px[sym], C.S.px["^GSPC"]
    y0, y1 = max(int(lo[:4]), int(px.d[0][:4]) + 1), min(int(hi[:4]), 2024)
    if y1 < y0:
        raise Skip("short_history")
    Y = rng.randint(y0, y1)
    dd = C.S.divs[sym][0]
    ex = [str(x) for x in dd if str(x).startswith(str(Y))]
    if len(ex) < 2:
        raise Skip("empty_answer")
    N = rng.choice([5, 10]) if d == 4 else rng.randint(3, 10)
    total = d == 5
    col = px.adj_close if total else px.close
    xs, span = [], []
    for x in ex:
        if not px.is_td(x):
            raise Skip("missing_row")
        k = px.i_le(x)
        if k < 1 or k + N >= len(px):
            raise Skip("short_history")
        if px.d[k - 1] < lo or px.d[k + N] > hi:
            raise Skip("outside_split_window")
        j = g.i_le(x)
        if g.d[j] != x or not np.array_equal(px.d[k - 1:k + N + 1], g.d[j - 1:j + N + 1]):
            raise Skip("calendar_mismatch")
        if not (px.clean(k - 1, k + N) and g.clean(j - 1, j + N)):
            raise Skip("bad_row")
        xs.append(pct(col[k + N], col[k - 1]) - pct(g.close[j + N], g.close[j - 1]))
        span += [str(px.d[k - 1]), str(px.d[k + N])]
    xs = np.array(xs)
    mean = float(xs.mean())
    if not round_ok(mean, 0.01):
        raise Skip("rounding_boundary")
    keys = [count_key(len(ex)), num_key(mean, 0.01)]
    o = np.argsort(xs)
    ends = ([("lowest", o[0], xs[o[1]] - xs[o[0]])] if d == 5 else []) + [("highest", o[-1], xs[o[-1]] - xs[o[-2]])]
    for _, k, margin in ends:
        if margin < 0.02:
            raise Skip("near_tie")
        if not round_ok(xs[k], 0.01):
            raise Skip("rounding_boundary")
        keys += [date_key(ex[k]), num_key(xs[k], 0.01)]
    # Version 2: the excess return is named (market-adjusted, against ^GSPC), not spelled out; the stated
    # conventions are the stock's price series, the event window in trading days and the benchmark.
    sret = ("total return (closes adjusted for splits and dividends)" if total else
            "price return from split-adjusted closes (excluding the dividend)")
    defs = ("For each ex-dividend date (day 0) of %s in calendar year %d, take the market-adjusted excess return over "
            "the S&P 500 index (^GSPC) from the close on day -1 to the close on day +%d, counting trading days, with the "
            "stock's %s and the index's price return, in percentage points." % (ent, Y, N, sret))
    ext = "lowest and the highest" if d == 5 else "highest"
    rnd = "Round excess returns to two decimals (negative if the stock lagged the index)."
    prompt = rng.choice([
        "Event study. %s How many ex-dividend dates are there, what is the mean excess return across them, and on "
        "which ex-dividend date%s (YYYY-MM-DD) was the excess return the %s, with %s? %s" % (
            defs, "s" if d == 5 else "", ext, "those values" if d == 5 else "its value", rnd),
        "%s Report the number of ex-dividend dates, the mean excess return across them, and the ex-dividend date%s "
        "(YYYY-MM-DD) with the %s excess return, with %s. %s" % (
            defs, "s" if d == 5 else "", ext, "those values" if d == 5 else "its value", rnd),
    ])
    return mk("div_event_study", split, d, prompt, keys,
              "%s %d, %d ex-dates, N=%d: excess %s; mean %.2f; %s." % (
                  sym, Y, len(ex), N, ", ".join("%s %.2f" % (x, v) for x, v in zip(ex, xs)), mean,
                  "; ".join("%s %s %.2f" % (w, ex[k], xs[k]) for w, k, _ in ends)),
              [sym, "^GSPC"], min(span), max(span), ["action_range", "nth_trading_day", "price_lookup", "pct_change",
                                               "difference", "mean", "argmax"] + (["argmin"] if d == 5 else []),
              len(ex) * 4 + 1, "mean over ex-dates of r(stock, -1..+N) - r(^GSPC), extremes",
              {"symbol": sym, "year": Y, "n": N, "total": total, "ask_min": d == 5, "ask_max": True})


# ============================================================================
# SEC: Q4 margins across companies with different fiscal years; growth leader's filing-date reaction

def sec_pick(C, rng, split, K):
    pool = C.sec_eq[split]
    first = rng.choice(pool)
    same = [s for s in pool if C.sector.get(s) == C.sector.get(first) and s != first]
    others = same if len(same) >= K - 1 and rng.random() < 0.6 else [s for s in pool if s != first]
    return [first] + rng.sample(others, K - 1)


def first_tag(X, cik, fy, tags):
    for t in tags:
        try:
            return t, X.value(cik, fy, t, "dur")[0]
        except Skip as e:
            if str(e) != "missing_row":
                raise
    raise Skip("missing_row")


def gen_q4_margin_rank(C, split, rng, d):
    K = 3 if d == 4 else 4
    syms = sec_pick(C, rng, split, K)
    names = [name_of(C, s, True) for s in syms]
    metric = rng.choice(sorted(MARGINS))
    title, ntag, ndesc = MARGINS[metric]
    common = None
    for s in syms:
        ys = set(C.X.years(C.cik[s]))
        common = ys if common is None else common & ys
    if not common:
        raise Skip("sec_no_fiscal_year")
    F = rng.choice(sorted(common))
    rows = []
    for s in syms:
        cik = C.cik[s]
        rtag, rfy = first_tag(C.X, cik, F, REV_PREC)
        r9 = C.X.nine_month(cik, F, rtag)[0]
        nfy = C.X.value(cik, F, ntag, "dur")[0]
        n9 = C.X.nine_month(cik, F, ntag)[0]
        q4r = rfy - r9
        if rfy <= 0 or not 0.1 < q4r / rfy < 0.5:
            raise Skip("degenerate_value")
        rows.append(((nfy - n9) / q4r * 100, q4r, nfy / rfy * 100, rtag, rfy, r9, nfy, n9))
    m = np.array([r[0] for r in rows])
    direction = rng.choice(["highest", "lowest"])
    order = np.argsort(-m) if direction == "highest" else np.argsort(m)
    if abs(m[order[0]] - m[order[1]]) < SEC_TIE or abs(m[order[1]] - m[order[2]]) < SEC_TIE:
        raise Skip("near_tie")
    w, w2 = int(order[0]), int(order[1])
    keys = [C.ticker_key(syms[w]), num_key(m[w], 0.01), num_key(rows[w][1] * 1e-6, 0.1)]
    checks = [(m[w], 0.01), (rows[w][1] * 1e-6, 0.1)]
    if d == 5:
        keys.append(num_key(rows[w][2], 0.01))
        checks.append((rows[w][2], 0.01))
    keys.append(num_key(m[w2], 0.01))
    checks.append((m[w2], 0.01))
    for v, u in checks:
        if not round_ok(v, u):
            raise Skip("rounding_boundary")
    # Version 2: the Q4 derivation is named (implied Q4), not explained; the stated conventions are the source
    # filings as originally filed, the fiscal-year labels, the numerator tag and the revenue tag order.
    defs = ("Use implied Q4 amounts: the full-year amount in the original Form 10-K for fiscal year %d less the "
            "nine-month year-to-date amount in the first-filed Q3 Form 10-Q of that fiscal year, with fiscal years as "
            "each company labels them. The numerator is %s; %s, the same tag for both periods." % (
                F, ndesc, REV_RULE.replace("that filing reports it for the period", "the 10-K reports it for the year")))
    q = ("Which company had the %s Q4 %s in fiscal year %d, and what was it? Also give that company's Q4 revenue in "
         "millions of US dollars rounded to one decimal%s, and the Q4 %s of the company with the second-%s one. Give "
         "margins %s." % (direction, title, F, ", its full-fiscal-year %s (from the 10-K amounts)" % title if d == 5 else "",
                          title, direction, PCT_SIGNED))
    prompt = rng.choice([
        "Compare the fourth-quarter %s of %s for fiscal year %d. %s %s" % (title, "; ".join(names), F, defs, q),
        "%s For %s: %s" % (defs, "; ".join(names), q),
    ])
    return mk("q4_margin_rank", split, d, prompt, keys,
              "%s Q4 %s FY%d: %s %.2f%% (Q4 revenue %.1f m, FY %s %.2f%%); second %s %.2f%%; all: %s." % (
                  direction, title, F, syms[w], m[w], rows[w][1] * 1e-6, title, rows[w][2], syms[w2], m[w2],
                  "; ".join("%s %s Q4 %.3f%% (FY %s-9M %s, %s %s-%s)" % (s, r[3], r[0], r[4], r[5], ntag, r[6], r[7])
                            for s, r in zip(syms, rows))),
              syms + ["CIK:%d" % C.cik[s] for s in syms],
              min(C.X.info(C.cik[s], F)["S"] for s in syms), max(C.X.info(C.cik[s], F)["E"] for s in syms),
              ["sec_fiscal_year_resolve", "sec_fact_lookup", "sec_quarter_lookup", "tag_fallback", "difference",
               "ratio", "rank"], 4 * K,
              "Q4 = FY(10-K) - 9M(Q3 10-Q) for %s and revenue, ratio, %s; Q4 revenue, runner-up margin" % (
                  ntag, direction),
              {"symbols": syms, "ciks": [C.cik[s] for s in syms], "fy": F, "metric": metric, "direction": direction,
               "ask_more": d == 5, "ask_rev": True, "ask_second": True})


def gen_filing_reaction(C, split, rng, d):
    K = 3 if d == 4 else 5
    lo, hi = SPLIT_WINDOWS[split]
    syms = sec_pick(C, rng, split, K)
    names = [name_of(C, s, d == 5) for s in syms]
    common = None
    for s in syms:
        ys = set(C.X.years(C.cik[s]))
        common = ys if common is None else common & ys
    cands = sorted(y for y in (common or ()) if y - 1 in common)
    if not cands:
        raise Skip("sec_no_fiscal_year")
    F = rng.choice(cands)
    gr = []
    for s in syms:
        cik = C.cik[s]
        r1 = first_tag(C.X, cik, F, REV_PREC)[1]
        r0 = first_tag(C.X, cik, F - 1, REV_PREC)[1]
        if r0 <= 0 or r1 <= 0:
            raise Skip("degenerate_value")
        gr.append(pct(r1, r0))
    gr = np.array(gr)
    direction = rng.choice(["highest", "lowest"])
    order = np.argsort(-gr) if direction == "highest" else np.argsort(gr)
    if abs(gr[order[0]] - gr[order[1]]) < SEC_TIE:
        raise Skip("near_tie")
    w = int(order[0])
    sym = syms[w]
    filed = C.X.info(C.cik[sym], F)["filed"]
    px, g = C.S.px[sym], C.S.px["^GSPC"]
    N = rng.choice([1, 5, 10]) if d == 4 else rng.choice([1, 3, 5, 10, 20])
    k = px.i_ge(filed)
    if k < 1 or k + N >= len(px):
        raise Skip("short_history")
    if px.d[k - 1] < lo or px.d[k + N] > hi:
        raise Skip("outside_split_window")
    j = g.i_ge(filed)
    if j < 1 or j + N >= len(g) or not np.array_equal(px.d[k - 1:k + N + 1], g.d[j - 1:j + N + 1]):
        raise Skip("calendar_mismatch")
    if not (px.clean(k - 1, k + N) and g.clean(j - 1, j + N)):
        raise Skip("bad_row")
    ex = pct(px.adj_close[k + N], px.adj_close[k - 1]) - pct(g.close[j + N], g.close[j - 1])
    keys = [C.ticker_key(sym)]
    checks = [(ex, 0.01)]
    if d == 5:
        keys.append(num_key(gr[w], 0.01))
        checks.append((gr[w], 0.01))
    keys += [date_key(filed), num_key(ex, 0.01)]
    for v, u in checks:
        if not round_ok(v, u):
            raise Skip("rounding_boundary")
    step1 = ("find the company with the %s year-over-year revenue growth in fiscal year %d versus fiscal year %d, "
             "taking each year's revenue from that year's own original annual report on Form 10-K (for each year, %s)"
             % (direction, F, F - 1, REV_RULE.replace("that filing reports it for the period",
                                                     "that year's 10-K reports it for the year")))
    step2 = ("take the date on which that company's original Form 10-K for fiscal year %d was filed with the SEC as "
             "day 0 (if it is not a trading day, the next trading day is day 0) and compute the stock's %s from the "
             "close on day -1 (the trading day before day 0) to the close on day +%d (%s after day 0), minus the "
             "price return of the S&P 500 index (^GSPC) over the same two closes" % (F, TOTAL_DESC, N, days_txt(N)))
    ask = ("Report the company, %sthe filing date (YYYY-MM-DD) and the excess return in percentage points rounded to "
           "two decimals (negative if the stock lagged the index)." % (
               "its revenue growth %s, " % PCT_SIGNED if d == 5 else ""))
    prompt = rng.choice([
        "Among %s: (1) %s; (2) %s. %s" % ("; ".join(names), step1, step2, ask),
        "Two steps for these companies: %s. First, %s. Second, %s. %s" % ("; ".join(names), step1, step2, ask),
    ])
    return mk("filing_reaction", split, d, prompt, keys,
              "%s revenue growth FY%d: %s (%.2f%%); all: %s; 10-K filed %s; day -1 %s to day +%d %s; excess %.2f." % (
                  direction, F, sym, gr[w], ", ".join("%s %.2f" % (s, v) for s, v in zip(syms, gr)), filed,
                  px.d[k - 1], N, px.d[k + N], ex),
              syms + ["CIK:%d" % C.cik[s] for s in syms] + ["^GSPC"], str(px.d[k - 1]), str(px.d[k + N]),
              ["sec_fiscal_year_resolve", "sec_fact_lookup", "tag_fallback", "pct_change", "argmax",
               "sec_filing_date", "nth_trading_day", "price_lookup", "difference"], 2 * K + 4,
              "growth leader -> original 10-K filed date -> excess return over ^GSPC",
              {"symbols": syms, "ciks": [C.cik[s] for s in syms], "fy": F, "direction": direction, "n": N,
               "ask_growth": d == 5})


# ============================================================================
# Three hops: extreme month -> CPI-deflated return (pinned vintage) -> Treasury yield change

def gen_month_chain(C, split, rng, d):
    sym, lo, hi = pick_entity(C, rng, split, 0.5, ("us",), needs_time_window=True)
    ent = name_of(C, sym, d == 5 and C.kind(sym) == "equity")
    px = C.S.px[sym]
    hi = min(hi, "2025-09-30")
    nm = rng.randint(12, 36) if d == 4 else rng.randint(24, 60)
    a, b = px.i_ge(lo), px.i_le(hi)
    ends = month_ends(px, a, b)
    if len(ends) < nm + 3:
        raise Skip("short_history")
    k0 = rng.randint(2, len(ends) - nm - 1)
    sel = ends[k0 - 1:k0 + nm]
    mi = [int(px.d[x][:4]) * 12 + int(px.d[x][5:7]) for x in sel]
    if any(q - p != 1 for p, q in zip(mi[:-1], mi[1:])):
        raise Skip("missing_row")
    total = d == 5 and px.kind != "index"
    c = (px.adj_close if total else px.close)
    if not px.clean(int(sel[0]), int(sel[-1])) or c[sel].min() <= 0:
        raise Skip("bad_row")
    r = (c[sel[1:]] / c[sel[:-1]] - 1) * 100
    direction = rng.choice(["lowest", "highest"])
    o = np.argsort(r) if direction == "lowest" else np.argsort(-r)
    if abs(r[o[0]] - r[o[1]]) < 0.02:
        raise Skip("near_tie")
    t = int(o[0])
    i_base, i_end = int(sel[t]), int(sel[t + 1])
    ym, ym0 = px.d[i_end][:7], px.d[i_base][:7]
    if ym0 + "-01" < lo:
        raise Skip("outside_split_window")
    vintage = "as_of_cutoff" if d == 4 else rng.choice(["as_of_cutoff", "latest"])
    cpi = C.S.fred["CPIAUCSL"]
    p1, p0 = cpi.pair(ym + "-01"), cpi.pair(ym0 + "-01")
    vi = 0 if vintage == "as_of_cutoff" else 1
    if not (np.isfinite(p1[vi]) and np.isfinite(p0[vi])):
        raise Skip("missing_row")
    infl = pct(p1[vi], p0[vi])
    nom = float(r[t])
    real = pct((1 + nom / 100) / (1 + infl / 100), 1)
    other = pct((1 + nom / 100) / (p1[1 - vi] / p0[1 - vi]), 1)
    keys = [date_key(str(px.d[i_end])), num_key(nom, 0.01), num_key(real, 0.01)]
    checks = [(nom, 0.01), (real, 0.01)]
    bp = None
    if d == 5:
        D = C.S.fred["DGS10"]
        od0, y0 = fred_near(D, str(px.d[i_base]), lo)
        od1, y1 = fred_near(D, str(px.d[i_end]), lo)
        bp = (y1 - y0) * 100
        keys.append(num_key(bp, 1))
        checks.append((bp, 1))
    for v, u in checks:
        if not round_ok(v, u):
            raise Skip("rounding_boundary")
    first_m, last_m = px.d[sel[1]][:7], px.d[sel[-1]][:7]
    adj = ("closes adjusted for splits and dividends (total return)" if total else
           "split-adjusted closes" if px.kind in ("equity", "etf") else "closing levels")
    vtxt = ("the values as published in the FRED vintage of 2025-10-30 (the snapshot's as-of-cutoff values, the "
            "default)" if vintage == "as_of_cutoff" else
            "the latest vintage (vintage='latest': today's values, including revisions after 2025-10-30)")
    s1 = ("among the calendar months %s through %s, find the month with the %s monthly return of %s, where a "
          "month's return is its last trading day's close divided by the previous month's last close, minus 1, using "
          "%s" % (month_label(first_m), month_label(last_m), direction, ent, adj))
    s2 = ("for that month compute the inflation-adjusted return (1 + nominal) / (1 + inflation) - 1, where inflation "
          "is CPIAUCSL (FRED, monthly) for that month divided by CPIAUCSL for the previous month, minus 1, using %s"
          % vtxt)
    s3 = ("give the change in the 10-year Treasury constant-maturity yield (FRED DGS10) from the previous month's last "
          "trading day to that month's last trading day, using the observation on each date or the most recent "
          "earlier one, in basis points rounded to a whole number (negative for a fall)")
    steps = [s1, s2] + ([s3] if d == 5 else [])
    ident = ("Identify the month by the date of its last trading day (YYYY-MM-DD) and give the nominal and real "
             "monthly returns %s%s." % (PCT_SIGNED, " and the yield change" if d == 5 else ""))
    prompt = rng.choice([
        "%s steps: %s. %s" % ("Three" if d == 5 else "Two", "; ".join("(%d) %s" % (i + 1, s) for i, s in enumerate(steps)),
                              ident),
        "First, %s. Then %s.%s %s" % (s1, s2, (" Finally, %s." % s3) if d == 5 else "", ident),
    ])
    return mk("month_chain", split, d, prompt, keys,
              "%s month %s (last day %s): nominal %.2f%%, CPI %s %s->%s, inflation %.3f%%, real %.2f%% (other vintage "
              "%.2f%%)%s." % (direction, ym, px.d[i_end], nom, vintage, p0[vi], p1[vi], infl, real, other,
                              "; DGS10 %s->%s: %+.0f bp" % (y0, y1, bp) if d == 5 else ""),
              [sym, "CPIAUCSL"] + (["DGS10"] if d == 5 else []), str(px.d[sel[0]]), str(px.d[sel[-1]]),
              ["resample_monthly", "pct_change", "argmin", "fred_lookup", "fred_vintage", "deflate"] +
              (["fred_lookup", "difference", "unit_convert"] if d == 5 else []), int(sel[-1] - sel[0]) + 6,
              "month-end returns -> extreme month -> CPI m/m (%s) -> real%s" % (vintage, " -> DGS10 bp" if d == 5 else ""),
              {"symbol": sym, "first_month": first_m, "last_month": last_m, "direction": direction, "total": total,
               "vintage": vintage, "ask_yield": d == 5})


HARD_TEMPLATES = {
    "risk_rank": gen_risk_rank,
    "rolling_beta": gen_rolling_beta,
    "ma_cross": gen_ma_cross,
    "reinvest_return": gen_reinvest_return,
    "div_event_study": gen_div_event_study,
    "q4_margin_rank": gen_q4_margin_rank,
    "filing_reaction": gen_filing_reaction,
    "month_chain": gen_month_chain,
}
