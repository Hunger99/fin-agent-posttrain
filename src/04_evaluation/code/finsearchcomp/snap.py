"""Read-only lookups on the market_v1 snapshot for the FinSearchComp reference computations.

Prices are the snapshot's reconstructed unadjusted values (raw_*) unless a function says otherwise; FRED values are the
cutoff vintage (value_vintage); SEC facts are read by period end and duration, never by the filing's fy/fp alone.
"""
import datetime as dt
import os
import sqlite3

import pandas as pd

DB = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1/market_v1.sqlite"
_con = None


def con():
    global _con
    if _con is None:
        _con = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    return _con


def q(sql, *args):
    return con().execute(sql, args).fetchall()


# ---------- prices ----------

def bars(sym, start="2000-01-01", end="2025-10-30"):
    """Daily bars of sym in [start, end] as a DataFrame indexed by date string, with raw and split-adjusted columns."""
    df = pd.read_sql_query(
        "select date, open, high, low, close, adj_close, volume, raw_open, raw_high, raw_low, raw_close, raw_volume "
        "from prices where symbol=? and date between ? and ? order by date", con(), params=(sym, start, end))
    if df.empty:
        raise KeyError(f"no bars for {sym} {start}..{end}")
    return df.set_index("date")


def bar(sym, date):
    df = bars(sym, date, date)
    return df.iloc[0]


def on_or_before(sym, date):
    """The bar of the last trading day on or before date."""
    r = q("select max(date) from prices where symbol=? and date<=?", sym, date)[0][0]
    return r, bar(sym, r)


def prev_bar(sym, date):
    """The bar of the trading day before date."""
    r = q("select max(date) from prices where symbol=? and date<?", sym, date)[0][0]
    return r, bar(sym, r)


def next_days(sym, date, n):
    """The n trading days strictly after date, as a DataFrame."""
    end = (dt.date.fromisoformat(date) + dt.timedelta(days=3 * n + 15)).isoformat()
    df = bars(sym, date, end)
    return df[df.index > date].iloc[:n]


def pct(a, b):
    """Percentage change from a to b."""
    return (b / a - 1) * 100


# ---------- FRED ----------

def fred(sid, date, vintage=True):
    r = q("select value, value_vintage from fred where series_id=? and date=?", sid, date)
    if not r:
        raise KeyError(f"no FRED {sid} {date}")
    return r[0][1] if vintage else r[0][0]


def fred_range(sid, start, end, vintage=True):
    col = "value_vintage" if vintage else "value"
    return q(f"select date, {col} from fred where series_id=? and date between ? and ? order by date", sid, start, end)


# ---------- SEC ----------

def ciks(ticker):
    """CIKs registered for a ticker (current first, then predecessors)."""
    rows = q("select cik, role from sec_companies where ticker=? or ','||tickers||',' like ?", ticker, f"%,{ticker},%")
    rows.sort(key=lambda r: r[1] != "current")
    return [r[0] for r in rows]


def facts(ticker, tag, end=None, taxonomy="us-gaap"):
    """All facts of a tag for a ticker (optionally one period end), as a DataFrame with a duration column (days)."""
    out = []
    for cik in ciks(ticker):
        sql = "select cik, taxonomy, tag, unit, val, fy, fp, form, start, end, filed, frame, accn from facts " \
              "where cik=? and tag=? and taxonomy=?"
        args = [cik, tag, taxonomy]
        if end:
            sql += " and end=?"
            args.append(end)
        out += q(sql, *args)
    df = pd.DataFrame(out, columns=["cik", "taxonomy", "tag", "unit", "val", "fy", "fp", "form", "start", "end",
                                    "filed", "frame", "accn"])
    if not df.empty:
        df["days"] = [None if s is None else (dt.date.fromisoformat(e) - dt.date.fromisoformat(s)).days
                      for s, e in zip(df["start"], df["end"])]
    return df


def value(ticker, tag, end, period="instant", start=None, first=True, taxonomy="us-gaap", forms=None, unit=None):
    """One fact value for a period ending on `end`.

    period: "instant" (balance sheet), "year" (a 350-380 day flow), "quarter" (80-100 days), "ytd9" (9 months) or
    "any" (match start exactly). first=True takes the earliest filing (as originally reported), else the latest.
    Raises KeyError when no fact matches; returns (value, row) so callers can cite the filing.
    """
    df = facts(ticker, tag, end, taxonomy)
    if df.empty:
        raise KeyError(f"{ticker} {tag} {end}: no facts")
    if unit:
        df = df[df.unit == unit]
    if forms:
        df = df[df.form.isin(forms)]
    if start:
        df = df[df.start == start]
    elif period == "instant":
        df = df[df.start.isna()]
    elif period == "year":
        df = df[(df.days >= 350) & (df.days <= 380)]
    elif period == "quarter":
        df = df[(df.days >= 80) & (df.days <= 100)]
    elif period == "ytd9":
        df = df[(df.days >= 260) & (df.days <= 285)]
    if df.empty:
        raise KeyError(f"{ticker} {tag} {end} {period}: no matching facts")
    df = df.sort_values("filed", ascending=first)
    row = df.iloc[0]
    return float(row.val), row.to_dict()


def first_of(ticker, tags, end, **kw):
    """value() for the first tag in tags that has a matching fact; returns (value, row)."""
    err = None
    for t in tags:
        try:
            return value(ticker, t, end, **kw)
        except KeyError as e:
            err = e
    raise err


def cite(row):
    """A short citation of a SEC fact row."""
    return f"SEC {row['taxonomy']}:{row['tag']} cik={row['cik']} end={row['end']} start={row['start']} " \
           f"form={row['form']} filed={row['filed']} accn={row['accn']} val={row['val']:.0f}"
