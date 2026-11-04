"""Read-only queries over the market_v1 snapshot, shared by the market tools (harness/envs/market.py) and the data
object of run_python's sandbox (harness/envs/market_sandbox.py).

The module uses the standard library only, because the sandbox loads it by file path in another interpreter (the
one with pandas). Every method returns plain rows: the tools page them, the sandbox wraps them in DataFrames. A bad
request raises QueryError with a message meant for the model; the tools turn it into ToolError.

Conventions (the snapshot's README, src/01_data-processing/utils/market_snapshot/README.md):
- nothing dated after the cutoff (meta "cutoff", 2025-10-30) is returned: a start after it is an error, an end after
  it is clamped, SEC facts filed or ending after it are dropped;
- prices: "raw" is what traded that day (raw_* columns), "split" is adjusted for every split up to the cutoff but
  not for dividends (open..close, volume), "total" is Yahoo's adjusted close (splits and dividends, multiplicative),
  with open, high and low scaled by the same day's adj_close/close and the split-adjusted volume; on the last
  trading day split = total;
- prices are rounded to 7 significant digits but at least 2 decimals (the source values are float32), volumes are
  whole numbers;
- weekly bars group by ISO week, monthly bars by calendar month, over the trading days inside start..end, each
  labelled by its last trading day: open of the first day, highest high, lowest low, close of the last day, summed
  volume;
- SEC values are in base units (USD, shares, USD per share); fy and fp describe the filing that reported a value,
  not the value's own period, which is start..end (start empty for a balance-sheet value).
"""
import bisect
import datetime
import math
import re
import sqlite3

FIELDS = ("open", "high", "low", "close", "volume")
ADJUSTMENTS = {"raw": "raw", "unadjusted": "raw", "none": "raw", "split": "split", "split_adjusted": "split",
               "total": "total", "total_return": "total"}
INTERVALS = {"day": "day", "daily": "day", "1d": "day", "week": "week", "weekly": "week", "1wk": "week",
             "month": "month", "monthly": "month", "1mo": "month"}
KINDS = ("equity", "index", "etf", "future", "fx", "crypto")
FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "40-F")
PERIODS = ("FY", "Q1", "Q2", "Q3")


class QueryError(Exception):
    """A request the snapshot cannot serve; the message tells the model what to change."""


def rnd(x):
    """A price as shown: 7 significant digits, at least 2 decimals (123.4500030517578 -> 123.45)."""
    if x is None or not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return round(x, max(2, 6 - int(math.floor(math.log10(abs(x))))))


def num(v):
    """An SEC or FRED value: a whole number as int (4349000000.0 -> 4349000000)."""
    if v is None:
        return None
    return int(v) if float(v).is_integer() and abs(v) < 1e15 else v


def check_date(value, name):
    s = str(value if value is not None else "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        raise QueryError(f"{name} must be a date written YYYY-MM-DD, such as 2020-08-28; got {value!r}")
    try:
        return datetime.date.fromisoformat(s).isoformat()
    except ValueError:
        raise QueryError(f"{name} {value!r} is not a calendar date") from None


def words(text):
    """Lower-case words of a query or name; a CamelCase tag splits into its words."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text or ""))
    return re.findall(r"[a-z0-9]+", text.lower())


def period_key(d, interval):
    y, m, day = int(d[:4]), int(d[5:7]), int(d[8:10])
    if interval == "month":
        return (y, m)
    return tuple(datetime.date(y, m, day).isocalendar()[:2])


class Snapshot:
    def __init__(self, path):
        # immutable=1: the snapshot never changes, so SQLite takes no locks on the shared disk
        self.db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self.path = path
        meta = dict(self.db.execute("SELECT key, value FROM meta WHERE key IN ('cutoff', 'start')").fetchall())
        self.cutoff, self.start = meta.get("cutoff", "2025-10-30"), meta.get("start", "2000-01-01")
        self._securities = self._series = self._companies = None
        self._days = {}

    def q(self, sql, *args):
        return self.db.execute(sql, args).fetchall()

    # -- dates ------------------------------------------------------------------------------------------------------
    def window(self, start, end, required=True):
        """(start, end, note) of a date range, end clamped to the cutoff; start after the cutoff is an error."""
        start = self.start if start is None and not required else check_date(start, "start")
        end = self.cutoff if end is None and not required else check_date(end, "end")
        if start > self.cutoff:
            raise QueryError(f"start {start} is after the snapshot's last day {self.cutoff}; the database holds no "
                             f"later data")
        note = None
        if end > self.cutoff:
            note = f"end {end} is after the snapshot's last day; data ends on {self.cutoff}"
            end = self.cutoff
        if start > end:
            raise QueryError(f"start {start} is after end {end}")
        return start, end, note

    # -- securities -------------------------------------------------------------------------------------------------
    def securities(self):
        if self._securities is None:
            cols = ("symbol", "name", "kind", "exchange", "currency", "first_date", "last_date", "cik")
            self._securities = {r[0]: dict(zip(cols, r)) for r in self.q(f"SELECT {', '.join(cols)} FROM securities")}
        return self._securities

    def search_securities(self, query, kind=None):
        """Matching securities, best first: exact symbol, symbol prefix, then names holding every query word."""
        if kind is not None:
            kind = str(kind).strip().lower()
            if kind not in KINDS:
                raise QueryError(f"kind must be one of {', '.join(KINDS)} (or left out); got {kind!r}")
        qs, qw = str(query or "").strip().upper(), words(query)
        if not qs:
            raise QueryError("query is empty; give a ticker such as AAPL or words of a name such as 'apple'")
        out = []
        for s in self.securities().values():
            if kind and s["kind"] != kind:
                continue
            name_w = set(words(s["name"])) | {s["symbol"].lower()}
            if s["symbol"] == qs or s["symbol"] == qs.replace(".", "-"):
                score = 0
            elif s["symbol"].startswith(qs):
                score = 1
            elif qw and all(any(w == x or (len(w) >= 3 and x.startswith(w)) for x in name_w) for w in qw):
                score = 2
            else:
                continue
            out.append((score, len(s["symbol"]), s["symbol"]))
        return [self.securities()[sym] for _, _, sym in sorted(out)]

    def symbol(self, symbol):
        s = str(symbol or "").strip().upper()
        sec = self.securities()
        for c in (s, s.replace(".", "-"), s.replace("/", "")):
            if c in sec:
                return c
        hits = self.search_securities(symbol)[:5] if s else []
        hint = ("; did you mean " + ", ".join(f"{h['symbol']} ({h['name']})" for h in hits) + "?") if hits else ""
        raise QueryError(f"unknown symbol {symbol!r}{hint} Find symbols with search_securities.")

    def trading_days(self, symbol):
        if symbol not in self._days:
            self._days[symbol] = [r[0] for r in self.q("SELECT date FROM prices WHERE symbol=? ORDER BY date", symbol)]
        return self._days[symbol]

    # -- prices -----------------------------------------------------------------------------------------------------
    def prices(self, symbol, start, end, adjustment="raw", fields=None, interval="day", required=True):
        """{"symbol", "adjustment", "interval", "columns", "rows", "note", "partial"} with every row of the range."""
        sym = self.symbol(symbol)
        adj = ADJUSTMENTS.get(str(adjustment or "raw").strip().lower())
        if adj is None:
            raise QueryError(f"adjustment must be 'raw' (unadjusted, as traded that day), 'split' (adjusted for splits "
                             f"only) or 'total' (adjusted for splits and dividends); got {adjustment!r}")
        iv = INTERVALS.get(str(interval or "day").strip().lower())
        if iv is None:
            raise QueryError(f"interval must be 'day', 'week' or 'month'; got {interval!r}")
        if fields is None or fields == [] or fields == "":
            fields = list(FIELDS)
        elif isinstance(fields, str):
            fields = [f for f in re.split(r"[,\s]+", fields) if f]
        fields = [str(f).strip().lower() for f in fields]
        fields = ["close" if f in ("adj_close", "adjclose", "price") else f for f in fields if f != "date"]  # date: always
        fields = fields or list(FIELDS)
        bad = [f for f in fields if f not in FIELDS]
        if bad:
            raise QueryError(f"unknown fields {bad}; fields are a list from {list(FIELDS)} (the adjustment argument, "
                             f"not the field, chooses raw, split-adjusted or total-return prices)")
        fields = list(dict.fromkeys(fields))
        start, end, note = self.window(start, end, required)
        sec = self.securities()[sym]
        if start < sec["first_date"] and end >= sec["first_date"]:
            extra = f"{sym} data begins on {sec['first_date']}"
            note = f"{note}; {extra}" if note else extra
        rows = self.q("SELECT date, open, high, low, close, adj_close, volume, raw_open, raw_high, raw_low, raw_close, "
                      "raw_volume FROM prices WHERE symbol=? AND date BETWEEN ? AND ? ORDER BY date", sym, start, end)
        days = []
        for d, o, h, lo, c, ac, v, ro, rh, rl, rc, rv in rows:
            if adj == "raw":
                vals = (ro, rh, rl, rc, rv)
            elif adj == "split":
                vals = (o, h, lo, c, v)
            else:
                f = ac / c if (ac is not None and c) else None
                vals = tuple(None if (x is None or f is None) else x * f for x in (o, h, lo)) + (ac, v)
            days.append((d,) + vals)
        partial = []
        if iv != "day" and days:
            days, partial = self.bars(sym, days, iv)
        idx = [FIELDS.index(f) + 1 for f in fields]
        out = [[r[0]] + [(None if r[i] is None else int(r[i])) if i == 5 else rnd(r[i]) for i in idx] for r in days]
        if not out:
            extra = f"no trading days of {sym} between {start} and {end}; trading_day finds the nearest trading day"
            note = f"{note}; {extra}" if note else extra
        return {"symbol": sym, "name": sec["name"], "currency": sec["currency"], "adjustment": adj, "interval": iv,
                "columns": ["date"] + fields, "rows": out, "note": note, "partial_bars": partial}

    def bars(self, sym, days, iv):
        """Weekly or monthly bars of the day rows, and the labels of bars cut by the range (partial)."""
        groups = {}
        for r in days:
            groups.setdefault(period_key(r[0], iv), []).append(r)
        out = []
        for g in groups.values():
            hs, ls = [r[2] for r in g if r[2] is not None], [r[3] for r in g if r[3] is not None]
            vs = [r[5] for r in g if r[5] is not None]
            out.append((g[-1][0], g[0][1], max(hs) if hs else None, min(ls) if ls else None, g[-1][4],
                        sum(vs) if vs else None))
        td = self.trading_days(sym)
        partial = []
        i = bisect.bisect_left(td, days[0][0])
        if i > 0 and period_key(td[i - 1], iv) == period_key(days[0][0], iv):
            partial.append(out[0][0])
        j = bisect.bisect_right(td, days[-1][0])
        if j < len(td) and period_key(td[j], iv) == period_key(days[-1][0], iv) and out[-1][0] not in partial:
            partial.append(out[-1][0])
        return out, partial

    # -- corporate actions ------------------------------------------------------------------------------------------
    def actions(self, symbol, start=None, end=None, kind="all"):
        """{"symbol", "splits": [[date, ratio, ratio_text, split_type]], "dividends": [[date, raw, split-adjusted]]}"""
        sym = self.symbol(symbol)
        k = str(kind or "all").strip().lower().rstrip("s")
        if k not in ("all", "split", "dividend"):
            raise QueryError(f"kind must be 'all', 'split' or 'dividend'; got {kind!r}")
        start, end, note = self.window(start, end, required=False)
        rows = self.q("SELECT date, kind, value, value_raw, ratio_text, split_type FROM actions WHERE symbol=? AND date "
                      "BETWEEN ? AND ? ORDER BY date", sym, start, end)
        splits = [[d, v, t, st] for d, kd, v, _, t, st in rows if kd == "split" and k in ("all", "split")]
        divs = [[d, round(vr, 6) if vr is not None else None, round(v, 6) if v is not None else None]
                for d, kd, v, vr, _, _ in rows if kd == "dividend" and k in ("all", "dividend")]
        return {"symbol": sym, "start": start, "end": end, "splits": splits, "dividends": divs, "note": note}

    # -- trading calendar -------------------------------------------------------------------------------------------
    def trading_day(self, symbol, date, offset=0, rule="on_or_before"):
        sym = self.symbol(symbol)
        d = check_date(date, "date")
        r = str(rule or "on_or_before").strip().lower().replace(" ", "_").replace("-", "_")
        if r not in ("on_or_before", "on_or_after"):
            raise QueryError(f"rule must be 'on_or_before' or 'on_or_after'; got {rule!r}")
        try:
            off = int(offset or 0)
        except (TypeError, ValueError):
            raise QueryError(f"offset must be a whole number of trading days (negative for earlier); got {offset!r}")
        if d > self.cutoff:
            raise QueryError(f"date {d} is after the snapshot's last day {self.cutoff}")
        td = self.trading_days(sym)
        if r == "on_or_before":
            i = bisect.bisect_right(td, d) - 1
            if i < 0:
                raise QueryError(f"no trading day of {sym} on or before {d}; its data begins on {td[0]}")
        else:
            i = bisect.bisect_left(td, d)
            if i >= len(td):
                raise QueryError(f"no trading day of {sym} on or after {d}; its data ends on {td[-1]}")
        j = i + off
        if not 0 <= j < len(td):
            raise QueryError(f"{off:+d} trading days from {td[i]} is outside {sym}'s data ({td[0]} to {td[-1]})")
        return {"symbol": sym, "date": d, "date_is_trading_day": td[i] == d, "rule": r, "base_trading_day": td[i],
                "offset": off, "trading_day": td[j],
                "weekday": datetime.date.fromisoformat(td[j]).strftime("%A")}

    def count_trading_days(self, symbol, start, end):
        sym = self.symbol(symbol)
        start, end, note = self.window(start, end)
        td = self.trading_days(sym)
        i, j = bisect.bisect_left(td, start), bisect.bisect_right(td, end)
        return {"symbol": sym, "start": start, "end": end, "count": max(0, j - i),
                "first_trading_day": td[i] if j > i else None, "last_trading_day": td[j - 1] if j > i else None,
                "note": note}

    # -- FRED -------------------------------------------------------------------------------------------------------
    def series(self):
        if self._series is None:
            cols = ("series_id", "title", "units", "frequency", "seasonal_adjustment", "first_date", "last_date",
                    "vintage_last_date")
            self._series = {r[0]: dict(zip(cols, r)) for r in self.q(f"SELECT {', '.join(cols)} FROM fred_series")}
        return self._series

    def search_fred(self, query):
        qs, qw = str(query or "").strip().upper(), words(query)
        if not qs:
            raise QueryError("query is empty; give a series id such as DGS10 or words of a title such as 'treasury 10'")
        out = []
        for s in self.series().values():
            tw = set(words(s["title"])) | {s["series_id"].lower()}
            if s["series_id"] == qs:
                score = 0
            elif s["series_id"].startswith(qs):
                score = 1
            else:
                hit = sum(any(w == x or (len(w) >= 3 and x.startswith(w)) for x in tw) for w in qw)
                if not hit:
                    continue
                score = 2 + len(qw) - hit
            out.append((score, s["series_id"]))
        return [self.series()[i] for _, i in sorted(out)]

    def fred(self, series_id, start=None, end=None, vintage="as_of_cutoff", required=True):
        sid = str(series_id or "").strip().upper()
        if sid not in self.series():
            hits = self.search_fred(series_id)[:5] if sid else []
            hint = ("; did you mean " + ", ".join(f"{h['series_id']} ({h['title']})" for h in hits) + "?") if hits else ""
            raise QueryError(f"unknown FRED series {series_id!r}{hint} Find series with search_fred.")
        v = str(vintage or "as_of_cutoff").strip().lower()
        if v not in ("latest", "as_of_cutoff"):
            raise QueryError(f"vintage must be 'as_of_cutoff' (default) or 'latest'; got {vintage!r}")
        start, end, note = self.window(start, end, required)
        col = "value" if v == "latest" else "value_vintage"
        rows = self.q(f"SELECT date, {col} FROM fred WHERE series_id=? AND date BETWEEN ? AND ? AND {col} IS NOT NULL "
                      f"ORDER BY date", sid, start, end)
        if not rows:
            extra = f"no observations of {sid} between {start} and {end}"
            note = f"{note}; {extra}" if note else extra
        return dict(self.series()[sid], vintage=v, start=start, end=end, rows=[[d, num(x)] for d, x in rows], note=note)

    # -- SEC --------------------------------------------------------------------------------------------------------
    def companies(self):
        if self._companies is None:
            cols = ("cik", "ticker", "tickers", "name", "n_facts", "role")
            self._companies = [dict(zip(cols, r)) for r in self.q(f"SELECT {', '.join(cols)} FROM sec_companies")]
        return self._companies

    def company(self, company):
        """The registrants of a company, given by ticker, CIK or name: every CIK filed under its ticker, the current
        registrant first and predecessor registrants after it."""
        s = str(company or "").strip()
        if not s:
            raise QueryError("company is empty; give a ticker such as NVDA or a name such as 'nvidia'")
        cs = self.companies()
        u = s.upper().replace(".", "-")
        tick = None
        if s.isdigit():
            hit = [c for c in cs if c["cik"] == int(s)]
            tick = hit[0]["ticker"] if hit else None
            if not hit:
                raise QueryError(f"no SEC registrant with CIK {s} in the snapshot")
        elif any(u in (c["tickers"] or c["ticker"] or "").split(",") for c in cs):
            tick = u
        elif u in self.securities() and self.securities()[u]["cik"]:
            cik = self.securities()[u]["cik"]
            tick = next((c["ticker"] for c in cs if c["cik"] == cik), None)
        if tick is None:
            qw = words(s)
            by_cik = {c["cik"]: c["ticker"] for c in cs}
            names = {c["ticker"] for c in cs if qw and all(w in words(c["name"]) for w in qw)}
            names |= {by_cik.get(sec["cik"]) for sec in self.search_securities(s) if sec["kind"] == "equity"}
            names = sorted(n for n in names if n)
            if len(names) == 1:
                tick = names[0]
            elif names:
                raise QueryError(f"company {company!r} matches several tickers: {', '.join(names[:10])}; give the ticker")
            else:
                raise QueryError(f"no SEC registrant matches {company!r}; give a ticker (search_securities finds it)")
        regs = [c for c in cs if tick in (c["tickers"] or c["ticker"] or "").split(",") or c["ticker"] == tick]
        if s.isdigit():
            regs = [c for c in regs if c["cik"] == int(s)] or regs
        regs.sort(key=lambda c: (c["role"] != "current", -(c["n_facts"] or 0)))
        return tick, regs

    def tags(self, ciks):
        rows = self.q(f"SELECT taxonomy, tag, COUNT(*), MIN(\"end\"), MAX(\"end\"), GROUP_CONCAT(DISTINCT unit) FROM facts "
                      f"WHERE cik IN ({','.join('?' * len(ciks))}) AND filed <= ? AND \"end\" <= ? GROUP BY taxonomy, tag",
                      *ciks, self.cutoff, self.cutoff)
        labels = dict(((t, g), lb) for t, g, lb in self.q("SELECT taxonomy, tag, label FROM sec_tags"))
        return [{"taxonomy": t, "tag": g, "label": labels.get((t, g)), "units": u, "n_facts": n, "first_end": a,
                 "last_end": b} for t, g, n, a, b, u in rows]

    def search_financial_tags(self, company, query):
        tick, regs = self.company(company)
        ciks = [c["cik"] for c in regs]
        qw = words(query)
        out = []
        for t in self.tags(ciks):
            tw = set(words(t["tag"])) | set(words(t["label"]))
            hit = sum(any(w == x or (len(w) >= 4 and x.startswith(w)) for x in tw) for w in qw) if qw else 1
            if hit:  # most query words first, then the tag with fewest other words (Assets before AssetsCurrent)
                out.append((-hit, len(set(words(t["tag"]))) - hit, -t["n_facts"], t["tag"], t))
        out.sort(key=lambda x: x[:4])
        return tick, regs, [t for *_, t in out]

    def financials(self, company, tag, fiscal_year=None, period=None, form=None, filing="latest"):
        """filing: "latest" (one row per period: the latest filing reporting it), "original" (the first filing
        reporting it) or "all" (every filing)."""
        fl = str(filing or "latest").strip().lower()
        if fl not in ("latest", "original", "all"):
            raise QueryError(f"filing must be 'latest' (default), 'original' or 'all'; got {filing!r}")
        tick, regs = self.company(company)
        ciks = [c["cik"] for c in regs]
        raw = str(tag or "").strip()
        tax, _, name = raw.rpartition(":")
        if not name:
            raise QueryError("tag is empty; give an XBRL tag such as Goodwill (search_financial_tags finds them)")
        where, args = [f"cik IN ({','.join('?' * len(ciks))})", "tag=?", "filed <= ?", "\"end\" <= ?"], [*ciks, name,
                                                                                                         self.cutoff, self.cutoff]
        if tax:
            where.append("taxonomy=?")
            args.append(tax.lower())
        if fiscal_year is not None and fiscal_year != "":
            m = re.fullmatch(r"(?:FY)?\s*(\d{4})", str(fiscal_year).strip(), re.I)
            if not m:
                raise QueryError(f"fiscal_year must be a year such as 2022; got {fiscal_year!r}")
            where.append("fy=?")
            args.append(int(m.group(1)))
        if period is not None and period != "":
            p = str(period).strip().upper()
            if p == "Q4":
                raise QueryError("no filing has fp Q4: the fourth quarter is reported in the annual 10-K (fp FY); a "
                                 "Q4 flow is the FY value minus the 9-month (Q3 year-to-date) value")
            if p not in PERIODS:
                raise QueryError(f"period must be one of {', '.join(PERIODS)} (the filing's fp); got {period!r}")
            where.append("fp=?")
            args.append(p)
        if form is not None and form != "":
            f = str(form).strip().upper().replace(" ", "")
            f = {"10K": "10-K", "10Q": "10-Q", "20F": "20-F", "40F": "40-F", "10K/A": "10-K/A", "10Q/A": "10-Q/A"}.get(f, f)
            if f not in FORMS:
                raise QueryError(f"form must be one of {', '.join(FORMS)}; got {form!r}")
            where.append("form=?")
            args.append(f)
        rows = self.q(f"SELECT taxonomy, unit, start, \"end\", val, fy, fp, form, filed, accn FROM facts WHERE "
                      f"{' AND '.join(where)} ORDER BY \"end\", start, filed, accn", *args)
        if not rows:
            has = self.q(f"SELECT COUNT(*) FROM facts WHERE cik IN ({','.join('?' * len(ciks))}) AND tag=?", *ciks, name)[0][0]
            if has:
                raise QueryError(f"{tick} has {name} facts, but none with these filters (fiscal_year={fiscal_year!r}, "
                                 f"period={period!r}, form={form!r}); fiscal_year is the filing's fiscal year")
            near = [t["tag"] for t in self.search_financial_tags(tick, name)[2][:8]]
            raise QueryError(f"{tick} has no facts tagged {raw!r}" + (f"; tags with similar words: {', '.join(near)}"
                                                                      if near else "") + ". Use search_financial_tags.")
        if fl != "all":
            pick = {}
            for r in rows:  # ordered by filed: the last one per period is the latest filing, the first the original
                k = (r[0], r[1], r[2], r[3])
                if fl == "latest" or k not in pick:
                    pick[k] = r
            rows = sorted(pick.values(), key=lambda r: (r[3], r[2] or "", r[0], r[1]))
        taxonomies = sorted({r[0] for r in rows})
        label = self.q("SELECT label FROM sec_tags WHERE taxonomy=? AND tag=?", taxonomies[0], name)
        out = []
        for t, unit, s, e, v, fy, fp, fm, filed, accn in rows:
            months = None
            if s:
                months = round((datetime.date.fromisoformat(e) - datetime.date.fromisoformat(s)).days / 30.44)
            row = [s, e, months, num(v), unit, fy, fp, fm, filed, accn]
            out.append(row + ([t] if len(taxonomies) > 1 else []))
        cols = ["start", "end", "months", "value", "unit", "fy", "fp", "form", "filed", "accn"]
        return {"ticker": tick, "registrants": [{"cik": c["cik"], "name": c["name"], "role": c["role"]} for c in regs],
                "tag": name, "taxonomy": taxonomies[0] if len(taxonomies) == 1 else taxonomies,
                "label": label[0][0] if label else None, "columns": cols + (["taxonomy"] if len(taxonomies) > 1 else []),
                "rows": out}
