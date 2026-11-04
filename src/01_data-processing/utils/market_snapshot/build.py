"""Build the offline market-data snapshot (market_v1) as one SQLite file.

Downloads daily prices, splits and dividends from the keyless Yahoo chart
API, macro series from FRED's CSV endpoint and XBRL company facts from SEC,
normalizes them to the cutoff date and stores them in one SQLite database,
with every raw response appended to one gzip JSONL file per source.

The build is resumable and incremental: each fetched id is recorded in the
``fetch_log`` table, and a re-run only fetches ids that are missing or
failed, so passing a new ``--symbols-file``, ``--fred-file`` or
``--sec-file`` adds only what is not yet stored.

Usage:
    python -B build.py [--db PATH] [--raw-dir DIR] [--sources yahoo,fred,sec]
        [--symbols-file F] [--fred-file F] [--sec-file F]
        [--only-symbols A,B] [--only-fred A,B] [--only-sec A,B]
        [--skip-universe] [--retry-failed]
"""

import argparse
import datetime as dt
import gzip
import io
import json
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
import requests

M = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1"
START = "2000-01-01"
CUTOFF = "2025-10-30"

YAHOO_UA = "Mozilla/5.0"  # A full Chrome UA string gets HTTP 429 from Yahoo.
SEC_UA = "fin-posttrain academic research"
YAHOO_URL = ("https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"
             "?period1={p1}&period2={p2}&interval=1d&events=div%7Csplit"
             "&includeAdjustedClose=true")
FRED_CSV = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
            "&cosd={start}&coed={end}")
ALFRED_CSV = ("https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={id}"
              "&vintage_date={end}&cosd={start}&coed={end}")
FRED_META = "https://fred.stlouisfed.org/graph/api/series/?id={id}"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SEARCH = "https://efts.sec.gov/LATEST/search-index?keysTyped={q}"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

INDICES = ("^GSPC ^IXIC ^DJI ^NDX ^RUT ^VIX ^FTSE ^GDAXI ^FCHI ^STOXX50E "
           "^N225 ^HSI ^GSPTSE ^AXJO ^KS11 ^BSESN ^TWII ^JKSE").split()
ETFS = "SPY QQQ IWM DIA GLD SLV TLT IEF HYG LQD EEM EFA VNQ XLF XLE XLK XLV".split()
FUTURES = "GC=F SI=F CL=F BZ=F NG=F HG=F ZC=F ZS=F ZW=F".split()
FX = ("EURUSD=X GBPUSD=X USDJPY=X USDCNY=X AUDUSD=X USDCAD=X USDCHF=X "
      "USDINR=X USDKRW=X USDHKD=X").split()
CRYPTO = "BTC-USD ETH-USD".split()
FRED_IDS = (
    "DFF FEDFUNDS EFFR SOFR IORB IOER DFEDTARU DFEDTARL DTB3 DGS1MO DGS3MO "
    "DGS6MO DGS1 DGS2 DGS5 DGS10 DGS30 T10Y2Y MORTGAGE30US CPIAUCSL CPILFESL "
    "PCEPI PCEPILFE GDP GDPC1 UNRATE PAYEMS INDPRO M2SL WALCL TOTBKCR "
    "TLAACBW027SBOG VIXCLS DCOILWTICO DCOILBRENTEU DEXUSEU DEXJPUS DEXUSUK "
    "DEXCHUS DEXCAUS DEXSZUS DEXKOUS DEXINUS DEXUSAL DEXHKUS DTWEXBGS").split()
SEC_TAXONOMIES = ("us-gaap", "ifrs-full", "dei")
# Registrants that filed a ticker's reports before a later reorganization or
# acquisition gave the ticker a new CIK; their facts are fetched as well.
PREDECESSOR_CIKS = {34088: "XOM", 1364742: "BLK", 813828: "SKYD",
                    1144519: "BG", 97517: "TPL", 1832433: "FERG"}
SEC_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "40-F"}
KIND_BY_TYPE = {"EQUITY": "equity", "ETF": "etf", "INDEX": "index",
                "FUTURE": "future", "CURRENCY": "fx",
                "CRYPTOCURRENCY": "crypto", "MUTUALFUND": "fund"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS securities(
  symbol TEXT PRIMARY KEY, name TEXT, kind TEXT, exchange TEXT,
  currency TEXT, cik INTEGER, first_date TEXT, last_date TEXT,
  instrument_type TEXT, timezone TEXT, list TEXT, sector TEXT,
  sub_industry TEXT, sp500_date_added TEXT, n_days INTEGER,
  post_cutoff_split_factor REAL);
CREATE TABLE IF NOT EXISTS prices(
  symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
  adj_close REAL, volume INTEGER, raw_open REAL, raw_high REAL,
  raw_low REAL, raw_close REAL, raw_volume INTEGER,
  PRIMARY KEY(symbol, date)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS actions(
  symbol TEXT, date TEXT, kind TEXT, value REAL, value_raw REAL,
  ratio_text TEXT, split_type TEXT,
  PRIMARY KEY(symbol, date, kind)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS fred_series(
  series_id TEXT PRIMARY KEY, title TEXT, units TEXT, frequency TEXT,
  seasonal_adjustment TEXT, notes TEXT, copyright_notice INTEGER,
  first_date TEXT, last_date TEXT, n_obs INTEGER,
  vintage_last_date TEXT, vintage_n_obs INTEGER);
CREATE TABLE IF NOT EXISTS fred(
  series_id TEXT, date TEXT, value REAL, value_vintage REAL,
  PRIMARY KEY(series_id, date)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS sec_companies(
  cik INTEGER PRIMARY KEY, ticker TEXT, tickers TEXT, name TEXT,
  n_facts INTEGER, role TEXT);
CREATE TABLE IF NOT EXISTS sec_tags(
  taxonomy TEXT, tag TEXT, label TEXT, description TEXT,
  PRIMARY KEY(taxonomy, tag)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS facts(
  cik INTEGER, taxonomy TEXT, tag TEXT, unit TEXT, val REAL, fy INTEGER,
  fp TEXT, form TEXT, start TEXT, "end" TEXT, filed TEXT, frame TEXT,
  accn TEXT);
CREATE INDEX IF NOT EXISTS facts_cik_tag_end ON facts(cik, tag, "end");
CREATE TABLE IF NOT EXISTS universe(
  symbol TEXT PRIMARY KEY, list TEXT, name TEXT, cik INTEGER, sector TEXT,
  sub_industry TEXT, date_added TEXT);
CREATE TABLE IF NOT EXISTS fetch_log(
  source TEXT, id TEXT, status TEXT, n_rows INTEGER, error TEXT,
  fetched_at TEXT, PRIMARY KEY(source, id));
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(*a):
    print(now(), *a, flush=True)


class Http:
    """Per-host rate-limited GET with retries and exponential backoff."""

    def __init__(self):
        self.s = requests.Session()
        self.last = {}
        self.min_gap = {"yahoo": 0.6, "fred": 0.6, "sec": 0.25, "wiki": 1.0}

    def get(self, host_key, url, headers=None, timeout=90, tries=6,
            ok_404=False):
        """Return (status, text); raises after the last failed try."""
        err = None
        for i in range(tries):
            gap = self.min_gap[host_key] - (time.time() - self.last.get(host_key, 0))
            if gap > 0:
                time.sleep(gap)
            self.last[host_key] = time.time()
            u = url
            if host_key == "yahoo" and i % 2 == 1:
                u = url.replace("//query1.", "//query2.")
            try:
                r = self.s.get(u, headers=headers or {}, timeout=timeout)
                if r.status_code == 200 or (ok_404 and r.status_code == 404):
                    return r.status_code, r.text
                err = f"HTTP {r.status_code}: {r.text[:200]!r}"
                if r.status_code == 404 and host_key == "yahoo":
                    return r.status_code, r.text
            except requests.RequestException as e:
                err = f"{type(e).__name__}: {e}"
            wait = min(60, 2 ** (i + 1))
            log(f"  retry {i + 1}/{tries} {host_key} {url[:120]} -> {err[:160]}; sleep {wait}s")
            time.sleep(wait)
        raise RuntimeError(err)


class RawSink:
    """Buffers raw responses and appends them as one gzip member per flush."""

    def __init__(self, raw_dir):
        self.dir = raw_dir
        os.makedirs(raw_dir, exist_ok=True)
        self.buf = {}

    def add(self, source, rec_id, url, status, body):
        rec = {"source": source, "id": rec_id, "url": url, "status": status,
               "fetched_at": now(), "body": body}
        self.buf.setdefault(source, []).append(json.dumps(rec, separators=(",", ":")))

    def flush(self):
        for source, lines in self.buf.items():
            if lines:
                with open(os.path.join(self.dir, f"{source}.jsonl.gz"), "ab") as f:
                    f.write(gzip.compress(("\n".join(lines) + "\n").encode()))
        self.buf = {}


def open_db(path):
    new = not os.path.exists(path)
    con = sqlite3.connect(path, timeout=600)
    if new:
        con.execute("PRAGMA page_size=32768")
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-262144")
    con.executescript(SCHEMA)
    cols = [r[1] for r in con.execute("PRAGMA table_info(actions)")]
    if "split_type" not in cols:  # databases built before the column existed
        con.execute("ALTER TABLE actions ADD COLUMN split_type TEXT")
        rows = con.execute("SELECT symbol, date, value FROM actions WHERE kind='split'").fetchall()
        con.executemany("UPDATE actions SET split_type=? WHERE symbol=? AND date=? AND kind='split'",
                        [(split_type(v), sym, d) for sym, d, v in rows])
    cols = [r[1] for r in con.execute("PRAGMA table_info(sec_companies)")]
    if "role" not in cols:
        con.execute("ALTER TABLE sec_companies ADD COLUMN role TEXT")
        con.execute("UPDATE sec_companies SET role='current'")
    con.commit()
    return con


def split_type(ratio):
    """'split' / 'reverse_split' for small whole-share ratios, else 'adjustment'.

    Yahoo encodes spin-offs, stock dividends and similar distributions as
    splits with ratios such as 1061:1000; those become 'adjustment'.
    """
    from fractions import Fraction
    f = Fraction(ratio).limit_denominator(100000)
    if f.denominator <= 10 and f.numerator <= 100 and abs(float(f) - ratio) < 1e-9:
        return "split" if f >= 1 else "reverse_split"
    return "adjustment"


def done_ids(con, source, retry_failed):
    q = "SELECT id, status FROM fetch_log WHERE source=?"
    rows = con.execute(q, (source,)).fetchall()
    if retry_failed:
        return {i for i, s in rows if s in ("ok", "empty", "not_found")}
    return {i for i, s in rows}


def mark(con, source, rec_id, status, n_rows=0, error=None):
    con.execute("INSERT OR REPLACE INTO fetch_log VALUES(?,?,?,?,?,?)",
                (source, rec_id, status, n_rows, error, now()))


def read_id_file(path):
    """One id per line; text after '#', a tab or a comma is ignored."""
    if not path or not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        tok = line.split("#")[0].replace("\t", " ").replace(",", " ").strip().split()
        if tok and tok[0] not in out:
            out.append(tok[0])
    return out


# ---------------------------------------------------------------- universe
def load_universe(con, http, sink, refresh):
    have = con.execute("SELECT COUNT(*) FROM universe WHERE list='sp500'").fetchone()[0]
    if have and not refresh:
        return
    st, html = http.get("wiki", WIKI_SP500,
                        headers={"User-Agent": "fin-posttrain academic research (Mozilla/5.0)"})
    sink.add("misc", "wikipedia_sp500", WIKI_SP500, st, html)
    t = pd.read_html(io.StringIO(html))[0]
    rows = []
    for r in t.itertuples(index=False):
        sym = str(r.Symbol).strip().replace(".", "-")
        rows.append((sym, "sp500", r.Security, int(r.CIK), r[2], r[3], str(r[5])))
    con.execute("DELETE FROM universe WHERE list='sp500'")
    con.executemany("INSERT OR REPLACE INTO universe VALUES(?,?,?,?,?,?,?)", rows)
    for lst, syms in (("index", INDICES), ("etf", ETFS), ("future", FUTURES),
                      ("fx", FX), ("crypto", CRYPTO)):
        con.executemany("INSERT OR IGNORE INTO universe(symbol, list) VALUES(?,?)",
                        [(s, lst) for s in syms])
    con.execute("INSERT OR REPLACE INTO meta VALUES('sp500_source', ?)",
                (f"{WIKI_SP500} fetched {now()} ({len(rows)} rows)",))
    con.commit()
    sink.flush()
    log(f"universe: {len(rows)} S&P 500 rows from Wikipedia")


# ------------------------------------------------------------------- yahoo
def yahoo_frame(js, cutoff):
    """Normalize one chart response to cutoff-basis rows.

    Yahoo's open/high/low/close/volume are adjusted for every split up to the
    download day, and adjclose also for every dividend up to that day. The
    snapshot is as of ``cutoff``, so prices are re-based to the splits and
    dividends known on the cutoff: close_c = close_y * S_post and
    adj_c = adj_y * S_post / D_post, where S_post is the product of split
    ratios after the cutoff and D_post = adj_y / close_y on the last trading
    day on or before the cutoff. Raw (unadjusted) values multiply by the
    product of all split ratios effective after each date.
    """
    res = js["chart"]["result"][0]
    meta = res["meta"]
    tz = meta.get("exchangeTimezoneName") or "UTC"
    ts = res.get("timestamp") or []
    q = (res.get("indicators", {}).get("quote") or [{}])[0]
    adjl = (res.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    n = len(ts)
    col = lambda k: q.get(k) if q.get(k) is not None else [None] * n
    df = pd.DataFrame({"ts": ts, "open": col("open"), "high": col("high"),
                       "low": col("low"), "close": col("close"),
                       "volume": col("volume"),
                       "adj": adjl if adjl is not None else [None] * n},
                      dtype="float64")
    df["date"] = (pd.to_datetime(df.ts.astype("int64"), unit="s", utc=True)
                  .dt.tz_convert(tz).dt.strftime("%Y-%m-%d")) if n else []
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")
    df = df.drop_duplicates("date", keep="last").sort_values("date")

    ev = res.get("events", {}) or {}
    splits = []
    for s in (ev.get("splits") or {}).values():
        num, den = s.get("numerator"), s.get("denominator")
        if not num or not den:
            continue
        d = pd.Timestamp(s["date"], unit="s", tz="UTC").tz_convert(tz).strftime("%Y-%m-%d")
        splits.append((d, float(num) / float(den), s.get("splitRatio")))
    splits.sort()
    divs = []
    for v in (ev.get("dividends") or {}).values():
        d = pd.Timestamp(v["date"], unit="s", tz="UTC").tz_convert(tz).strftime("%Y-%m-%d")
        divs.append((d, float(v["amount"])))
    divs.sort()

    def s_after(dates):
        f = np.ones(len(dates))
        for d, r, _ in splits:
            f[np.asarray(dates) < d] *= r
        return f

    s_post = float(np.prod([r for d, r, _ in splits if d > cutoff])) if splits else 1.0
    df = df[(df.date >= START) & (df.date <= cutoff)].copy()
    s_all = s_after(df.date.values)
    last = df.dropna(subset=["close", "adj"])
    last = last[last.close > 0]
    d_post = float(last.adj.iloc[-1] / last.close.iloc[-1]) if len(last) else None
    out = pd.DataFrame({"date": df.date})
    for k in ("open", "high", "low", "close"):
        out[k] = df[k] * s_post
        out["raw_" + k] = df[k] * s_all
    out["adj_close"] = df.adj * s_post / d_post if d_post else df.close * s_post
    out["volume"] = (df.volume / s_post).round()
    out["raw_volume"] = (df.volume / s_all).round()

    acts = []
    for d, r, txt in splits:
        if START <= d <= cutoff:
            acts.append((d, "split", r, r, txt, split_type(r)))
    if divs:
        f = s_after([d for d, _ in divs])
        for (d, a), fa in zip(divs, f):
            if START <= d <= cutoff:
                acts.append((d, "dividend", a * s_post, a * fa, None, None))
    return meta, out, acts, s_post


def nn(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return x


def fetch_yahoo(con, http, sink, symbols, retry_failed, p2):
    done = done_ids(con, "yahoo", retry_failed)
    todo = [s for s in symbols if s not in done]
    log(f"yahoo: {len(symbols)} symbols, {len(todo)} to fetch")
    uni = {r[0]: r for r in con.execute(
        "SELECT symbol, list, name, cik, sector, sub_industry, date_added FROM universe")}
    p1 = int(pd.Timestamp(START, tz="UTC").timestamp()) - 86400 * 2
    for i, sym in enumerate(todo):
        url = YAHOO_URL.format(host="query1", sym=requests.utils.quote(sym, safe=""), p1=p1, p2=p2)
        try:
            st, text = http.get("yahoo", url, headers={"User-Agent": YAHOO_UA})
        except RuntimeError as e:
            mark(con, "yahoo", sym, "fail", 0, str(e)[:500])
            log(f"yahoo {sym}: FAIL {e}")
            continue
        try:
            js = json.loads(text)
        except ValueError:
            js = None
        sink.add("yahoo_chart", sym, url, st, js if js is not None else text)
        res = (js or {}).get("chart", {}).get("result")
        if st == 404 or not res:
            errmsg = json.dumps((js or {}).get("chart", {}).get("error"))[:500]
            mark(con, "yahoo", sym, "not_found", 0, errmsg)
            log(f"yahoo {sym}: not found {errmsg}")
        else:
            try:
                meta, df, acts, s_post = yahoo_frame(js, CUTOFF)
            except Exception as e:  # malformed payloads are logged, not fatal
                mark(con, "yahoo", sym, "fail", 0, f"parse: {type(e).__name__}: {e}"[:500])
                log(f"yahoo {sym}: parse FAIL {e}")
                continue
            con.execute("DELETE FROM prices WHERE symbol=?", (sym,))
            con.execute("DELETE FROM actions WHERE symbol=?", (sym,))
            cols = ["open", "high", "low", "close", "adj_close", "volume",
                    "raw_open", "raw_high", "raw_low", "raw_close", "raw_volume"]
            rows = [(sym, r[0]) + tuple(nn(v) for v in r[1:])
                    for r in df[["date"] + cols].itertuples(index=False, name=None)]
            rows = [r[:7] + (None if r[7] is None else int(r[7]),) + r[8:12]
                    + (None if r[12] is None else int(r[12]),) for r in rows]
            con.executemany("INSERT INTO prices VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            con.executemany("INSERT OR REPLACE INTO actions VALUES(?,?,?,?,?,?,?)",
                            [(sym,) + a for a in acts])
            u = uni.get(sym, (sym, "extra", None, None, None, None, None))
            itype = meta.get("instrumentType")
            kind = KIND_BY_TYPE.get(itype, (itype or "unknown").lower())
            con.execute(
                "INSERT OR REPLACE INTO securities VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sym, meta.get("longName") or meta.get("shortName") or u[2], kind,
                 meta.get("fullExchangeName") or meta.get("exchangeName"),
                 meta.get("currency"), u[3],
                 rows[0][1] if rows else None, rows[-1][1] if rows else None,
                 itype, meta.get("exchangeTimezoneName"), u[1], u[4], u[5], u[6],
                 len(rows), s_post))
            status = "ok" if rows else "empty"
            mark(con, "yahoo", sym, status, len(rows))
            log(f"yahoo {sym}: {status} {len(rows)} days, {len(acts)} actions"
                + (f", post-cutoff split factor {s_post}" if s_post != 1 else ""))
        if (i + 1) % 20 == 0:
            sink.flush()
            con.commit()
    sink.flush()
    con.commit()


# -------------------------------------------------------------------- fred
def fred_csv(text):
    """Parse a FRED/ALFRED CSV into (date, value) rows within the range."""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df = df.iloc[:, :2]
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df.value, errors="coerce")
    df = df.dropna()
    return df[(df.date >= START) & (df.date <= CUTOFF)]


def fetch_fred(con, http, sink, ids, retry_failed):
    done = done_ids(con, "fred", retry_failed)
    todo = [s for s in ids if s not in done]
    log(f"fred: {len(ids)} series, {len(todo)} to fetch")
    for sid in todo:
        url = FRED_CSV.format(id=sid, start=START, end=CUTOFF)
        try:
            st, text = http.get("fred", url, ok_404=True)
        except RuntimeError as e:
            mark(con, "fred", sid, "fail", 0, str(e)[:500])
            log(f"fred {sid}: FAIL {e}")
            continue
        sink.add("fred", sid, url, st, text)
        if st == 404 or not text.startswith("observation_date"):
            mark(con, "fred", sid, "not_found", 0, text[:300])
            log(f"fred {sid}: not found")
            continue
        df = fred_csv(text)
        # ALFRED gives the series as published on the cutoff day: values
        # revised later keep their old value, and observations first
        # released after the cutoff are absent (value_vintage NULL).
        vurl = ALFRED_CSV.format(id=sid, start=START, end=CUTOFF)
        vdf = None
        try:
            vst, vtext = http.get("fred", vurl, tries=3, ok_404=True)
            sink.add("fred_alfred", sid, vurl, vst, vtext)
            if vst == 200 and vtext.startswith("observation_date"):
                vdf = fred_csv(vtext)
        except RuntimeError as e:
            log(f"fred {sid}: ALFRED vintage unavailable ({str(e)[:120]})")
        vals = dict(zip(df.date, df.value))
        vvals = dict(zip(vdf.date, vdf.value)) if vdf is not None else {}
        meta = {}
        murl = FRED_META.format(id=sid)
        try:
            mst, mtext = http.get("fred", murl, tries=3)
            mj = json.loads(mtext)
            sink.add("fred_meta", sid, murl, mst, mj)
            meta = mj["chart_series"][0]["series_objects"]["a"]
        except Exception as e:  # metadata is optional; keep the id only
            log(f"fred {sid}: metadata unavailable ({e})")
        notes = meta.get("notes")
        con.execute("DELETE FROM fred WHERE series_id=?", (sid,))
        con.executemany("INSERT INTO fred VALUES(?,?,?,?)",
                        [(sid, d, vals.get(d), vvals.get(d))
                         for d in sorted(set(vals) | set(vvals))])
        con.execute("INSERT OR REPLACE INTO fred_series VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sid, meta.get("title"), meta.get("units_short") or meta.get("units"),
                     meta.get("frequency"), meta.get("season"), notes,
                     int("copyright" in (notes or "").lower()) if meta else None,
                     df.date.min() if len(df) else None,
                     df.date.max() if len(df) else None, len(df),
                     max(vvals) if vvals else None, len(vvals) if vdf is not None else None))
        status = "ok" if len(df) else "empty"
        mark(con, "fred", sid, status, len(df),
             None if vdf is not None else "no ALFRED vintage")
        log(f"fred {sid}: {status} {len(df)} obs, vintage {len(vvals) if vdf is not None else 'n/a'}")
    sink.flush()
    con.commit()


# --------------------------------------------------------------------- sec
def sec_ticker_map(con, http, sink):
    """Ticker -> CIK from SEC's company_tickers.json; empty dict if blocked."""
    try:
        st, text = http.get("sec", SEC_TICKERS, headers={"User-Agent": SEC_UA}, tries=2)
        js = json.loads(text)
        sink.add("misc", "sec_company_tickers", SEC_TICKERS, st, js)
        return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in js.values()}
    except Exception as e:
        log(f"sec: company_tickers.json unavailable ({str(e)[:120]}); using fallbacks")
        return {}


def sec_lookup(http, sink, ticker):
    """CIK of an exact ticker match from EDGAR's entity search, or None."""
    q = ticker.replace("-", ".")
    url = SEC_SEARCH.format(q=requests.utils.quote(q))
    try:
        st, text = http.get("sec", url, headers={"User-Agent": SEC_UA}, tries=3)
        js = json.loads(text)
    except Exception:
        return None
    sink.add("misc", f"sec_search:{ticker}", url, st, js)
    for h in js.get("hits", {}).get("hits", []):
        ts = [t.strip().upper() for t in (h["_source"].get("tickers") or "").split(",")]
        if q.upper() in ts or ticker.upper() in ts:
            return int(h["_id"])
    return None


def fetch_sec(con, http, sink, targets, retry_failed):
    """targets: dict cik -> list of tickers."""
    done = done_ids(con, "sec", retry_failed)
    todo = [c for c in targets if str(c) not in done]
    log(f"sec: {len(targets)} companies, {len(todo)} to fetch")
    for i, cik in enumerate(todo):
        url = SEC_FACTS.format(cik=cik)
        try:
            st, text = http.get("sec", url, headers={"User-Agent": SEC_UA}, ok_404=True)
        except RuntimeError as e:
            mark(con, "sec", str(cik), "fail", 0, str(e)[:500])
            log(f"sec {cik}: FAIL {e}")
            continue
        if st == 404:
            sink.add("sec_companyfacts", str(cik), url, st, text[:2000])
            mark(con, "sec", str(cik), "not_found", 0, "HTTP 404")
            log(f"sec {cik} {targets[cik]}: no companyfacts (404)")
            continue
        js = json.loads(text)
        sink.add("sec_companyfacts", str(cik), url, st, js)
        rows, tags = [], []
        for tax in SEC_TAXONOMIES:
            for tag, f in (js.get("facts", {}).get(tax) or {}).items():
                tags.append((tax, tag, f.get("label"), f.get("description")))
                for unit, obs in (f.get("units") or {}).items():
                    for o in obs:
                        # Filings after the cutoff were not public on it.
                        if o.get("form") not in SEC_FORMS or (o.get("filed") or "") > CUTOFF:
                            continue
                        val = o.get("val")
                        if isinstance(val, int) and abs(val) >= 2 ** 63:
                            val = float(val)  # beyond SQLite INTEGER range
                        rows.append((cik, tax, tag, unit, val, o.get("fy"),
                                     o.get("fp"), o.get("form"), o.get("start"),
                                     o.get("end"), o.get("filed"), o.get("frame"),
                                     o.get("accn")))
        con.execute("DELETE FROM facts WHERE cik=?", (cik,))
        con.executemany("INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.executemany("INSERT OR IGNORE INTO sec_tags VALUES(?,?,?,?)", tags)
        tk = targets[cik]
        role = (f"predecessor registrant of {PREDECESSOR_CIKS[cik]}"
                if cik in PREDECESSOR_CIKS else "current")
        con.execute("INSERT OR REPLACE INTO sec_companies VALUES(?,?,?,?,?,?)",
                    (cik, tk[0], ",".join(tk), js.get("entityName"), len(rows), role))
        mark(con, "sec", str(cik), "ok" if rows else "empty", len(rows))
        log(f"sec {cik} {','.join(tk)}: {len(rows)} facts")
        if (i + 1) % 10 == 0:
            sink.flush()
            con.commit()
    # A ticker whose current registrant filed nothing by the cutoff points to
    # its predecessor registrant instead (XOM -> Exxon Mobil Corp).
    for pcik, t in PREDECESSOR_CIKS.items():
        con.execute("UPDATE securities SET cik=? WHERE symbol=? AND cik IN "
                    "(SELECT cik FROM sec_companies WHERE n_facts=0)", (pcik, t))
    sink.flush()
    con.commit()


def sec_targets(con, http, sink, extra_sec, extra_us_symbols):
    """CIK -> tickers for S&P 500 members plus extra tickers or CIKs."""
    targets = {}
    for sym, cik in con.execute(
            "SELECT symbol, cik FROM universe WHERE list='sp500' AND cik IS NOT NULL"):
        targets.setdefault(int(cik), []).append(sym)
    have = {s for ss in targets.values() for s in ss}
    for pcik, t in PREDECESSOR_CIKS.items():
        if t in have:
            targets.setdefault(pcik, [t])
    need = [t for t in list(extra_sec) + list(extra_us_symbols)]
    tmap = None
    known = {s: c for c, ss in targets.items() for s in ss}
    for tok in need:
        if tok.isdigit():
            targets.setdefault(int(tok), [tok])
            continue
        t = tok.upper().replace(".", "-")
        cik = known.get(t)
        if cik is None:
            if tmap is None:
                tmap = sec_ticker_map(con, http, sink)
            cik = tmap.get(t) or sec_lookup(http, sink, t)
        if cik is None:
            mark(con, "sec_cik", t, "not_found", 0, "no CIK for ticker")
            log(f"sec: no CIK for {t}")
            continue
        known[t] = cik
        if t not in targets.setdefault(cik, []):
            targets[cik].append(t)
        con.execute("UPDATE securities SET cik=? WHERE symbol=? AND cik IS NULL", (cik, t))
    con.commit()
    return targets


# -------------------------------------------------------------------- main
def write_meta(con, started, args):
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("securities", "prices", "actions", "fred_series", "fred",
                        "sec_companies", "sec_tags", "facts")}
    fails = {}
    for src, i, st, err in con.execute(
            "SELECT source, id, status, error FROM fetch_log WHERE status NOT IN ('ok')"):
        fails.setdefault(src, {})[i] = f"{st}: {err}"[:300]
    kv = {
        "cutoff": CUTOFF, "start": START,
        "cutoff_note": ("Prices, actions and FRED observations are dated "
                        f"{START}..{CUTOFF}; prices and dividends are adjusted for "
                        "splits and dividends known on the cutoff only."),
        "last_build_started": started, "last_build_finished": now(),
        "source_urls": json.dumps({"yahoo": YAHOO_URL, "fred_csv": FRED_CSV,
                                   "fred_meta": FRED_META,
                                   "fred_alfred_vintage": ALFRED_CSV, "sec_facts": SEC_FACTS,
                                   "sec_tickers": SEC_TICKERS, "sec_search": SEC_SEARCH,
                                   "sp500": WIKI_SP500}),
        "counts": json.dumps(counts), "failures": json.dumps(fails),
        "sec_taxonomies": ",".join(SEC_TAXONOMIES), "sec_forms": ",".join(sorted(SEC_FORMS)),
    }
    if con.execute("SELECT 1 FROM meta WHERE key='first_build_started'").fetchone() is None:
        kv["first_build_started"] = started
    con.executemany("INSERT OR REPLACE INTO meta VALUES(?,?)", list(kv.items()))
    con.commit()
    log("counts", json.dumps(counts))
    log("failures", json.dumps({k: len(v) for k, v in fails.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(M, "market_v1.sqlite"))
    ap.add_argument("--raw-dir", default=os.path.join(M, "raw"))
    ap.add_argument("--sources", default="yahoo,fred,sec")
    ap.add_argument("--symbols-file", action="append", default=[])
    ap.add_argument("--fred-file", action="append", default=[])
    ap.add_argument("--sec-file", action="append", default=[])
    ap.add_argument("--only-symbols", help="comma list; replaces the default universe")
    ap.add_argument("--only-fred", help="comma list; replaces the default FRED list")
    ap.add_argument("--only-sec", help="comma list of tickers/CIKs; replaces S&P 500 for SEC")
    ap.add_argument("--retry-failed", action="store_true", default=True)
    ap.add_argument("--no-retry-failed", dest="retry_failed", action="store_false")
    ap.add_argument("--refresh-universe", action="store_true")
    args = ap.parse_args()

    started = now()
    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    con = open_db(args.db)
    http = Http()
    sink = RawSink(args.raw_dir)
    sources = set(args.sources.split(","))
    load_universe(con, http, sink, args.refresh_universe)

    extra_syms = [s for f in args.symbols_file for s in read_id_file(f)]
    extra_fred = [s for f in args.fred_file for s in read_id_file(f)]
    extra_sec = [s for f in args.sec_file for s in read_id_file(f)]
    con.executemany("INSERT OR IGNORE INTO universe(symbol, list) VALUES(?, 'extra')",
                    [(s,) for s in extra_syms])
    con.commit()

    if args.only_symbols:
        symbols = args.only_symbols.split(",")
    else:
        symbols = [r[0] for r in con.execute(
            "SELECT symbol FROM universe ORDER BY CASE list WHEN 'sp500' THEN 1 "
            "WHEN 'extra' THEN 9 ELSE 0 END, symbol")]
    symbols += [s for s in extra_syms if s not in symbols]
    p2 = int(time.time())

    if "yahoo" in sources:
        fetch_yahoo(con, http, sink, symbols, args.retry_failed, p2)
    if "fred" in sources:
        ids = args.only_fred.split(",") if args.only_fred else list(FRED_IDS)
        ids += [s for s in extra_fred if s not in ids]
        fetch_fred(con, http, sink, ids, args.retry_failed)
    if "sec" in sources:
        extra_us = [s for (s,) in con.execute(
            "SELECT s.symbol FROM securities s JOIN universe u USING(symbol) "
            "WHERE u.list='extra' AND s.kind='equity' AND s.currency='USD'")]
        if args.only_sec:
            targets = sec_targets(con, http, sink, args.only_sec.split(","), [])
            keep = set()
            for tok in args.only_sec.split(","):
                keep.update(c for c, ts in targets.items()
                            if tok.upper().replace(".", "-") in ts or tok == str(c))
            targets = {c: t for c, t in targets.items() if c in keep}
        else:
            targets = sec_targets(con, http, sink, extra_sec, extra_us)
        fetch_sec(con, http, sink, targets, args.retry_failed)
    write_meta(con, started, args)
    con.close()
    log("done")


if __name__ == "__main__":
    main()
