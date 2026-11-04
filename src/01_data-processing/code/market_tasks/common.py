"""Shared data access, split rules and answer-key helpers for the market task generator.

The generator reads the read-only market_v1 snapshot once, caches the tables it
needs in a pickle on local disk, and exposes per-symbol price arrays, FRED
series, corporate actions and a subset of SEC XBRL facts.
"""
import hashlib
import json
import os
import pickle
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd

M_DIR = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1"
DB = M_DIR + "/market_v1.sqlite"
OUT_DIR = M_DIR + "/tasks"
CUTOFF = "2025-10-30"
GLOBAL_START = "2000-01-01"

# Non-equity series are split by time window; equities by ticker hash.
SPLIT_WINDOWS = {
    "train": ("2000-01-01", "2016-12-31"),
    "dev": ("2017-01-01", "2019-12-31"),
    "test": ("2020-01-01", "2025-10-30"),
}
SPLITS = ("train", "dev", "test")

# Duration tags are annual flows; instant tags are balance-sheet points.
SEC_DURATION_TAGS = [
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
    "NetIncomeLoss", "OperatingIncomeLoss", "GrossProfit", "CostOfRevenue",
    "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "ResearchAndDevelopmentExpense",
    "NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment",
    "IncomeTaxExpenseBenefit", "EarningsPerShareDiluted",
]
SEC_INSTANT_TAGS = [
    "Assets", "Liabilities", "StockholdersEquity", "CashAndCashEquivalentsAtCarryingValue",
    "InventoryNet", "LongTermDebtNoncurrent", "Goodwill", "AssetsCurrent", "LiabilitiesCurrent",
]
DEI_SHARES = "EntityCommonStockSharesOutstanding"


def ro_connect(path=DB):
    return sqlite3.connect("file:%s?mode=ro" % path, uri=True)


def ticker_split(ticker):
    """Stable split of an equity ticker: sha256 mod 100, <70 train, <80 dev, else test."""
    h = int(hashlib.sha256(ticker.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("dev" if h < 80 else "test")


def d2s(d):
    return d.isoformat()


def s2d(s):
    return date(int(s[:4]), int(s[5:7]), int(s[8:10]))


def add_days(s, n):
    return d2s(s2d(s) + timedelta(days=n))


def minus_one_year(s):
    d = s2d(s)
    try:
        return d2s(d.replace(year=d.year - 1))
    except ValueError:  # 29 February
        return d2s(d.replace(year=d.year - 1, day=28))


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def long_date(s):
    d = s2d(s)
    return "%s %d, %d" % (MONTHS[d.month - 1], d.day, d.year)


def fmt_date(s, rng):
    """Date in the prompt: ISO or long form, chosen by the seed."""
    return s if rng.random() < 0.5 else long_date(s)


def month_label(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    return "%s %d" % (MONTHS[m - 1], y)


def month_aliases(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    return [ym, "%s %d" % (MONTHS[m - 1][:3], y), "%d/%d" % (m, y), "%02d/%d" % (m, y),
            "%s, %d" % (MONTHS[m - 1], y)]


# ----------------------------------------------------------------------------
# Answer keys (harness v3 market grader schema)

def round_ok(v, unit):
    """False when v lies within 10% of a rounding boundary of the requested unit."""
    q = v / unit
    return abs(q - round(q)) <= 0.4


def num_key(v, unit, ndp=None):
    if ndp is None:
        ndp = max(0, -int(np.floor(np.log10(unit) + 1e-9)))
    return {"kind": "number", "value": round(float(v), ndp), "tol": unit}


def count_key(n):
    # Integer counts need the exact integer; 0.5 accepts only it.
    return {"kind": "number", "value": float(n), "tol": 0.5}


def date_key(s):
    return {"kind": "date", "value": s}


def text_key(v, aliases):
    return {"kind": "text", "value": v, "aliases": [a for a in aliases if a and a != v]}


def set_key(items):
    return {"kind": "set", "value": items}


SUFFIXES = [", Inc.", " Inc.", " Incorporated", " Corporation", " Corp.", " Company", " Co.",
            " plc", " PLC", " N.V.", " Ltd.", " Limited", " Holdings", " Group", " L.P.",
            " S.A.", " AG", " SE", " Inc", " Corp", ",", " (The)", " & Co."]


def short_names(name):
    out = []
    s = name.strip()
    for _ in range(3):
        for suf in SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)].strip()
        if s.startswith("The "):
            s = s[4:]
    if len(s) >= 3 and s != name:
        out.append(s)
    return out


class Skip(Exception):
    """Raised by a generator when a candidate violates a quality rule; the message is the rule."""


# ----------------------------------------------------------------------------
# Snapshot loading

class Px:
    """Daily prices of one symbol as numpy arrays indexed by trading day."""

    COLS = ["open", "high", "low", "close", "adj_close", "volume", "raw_open", "raw_high",
            "raw_low", "raw_close", "raw_volume"]

    def __init__(self, sym, kind, df):
        self.sym = sym
        self.kind = kind
        self.d = df["date"].to_numpy().astype("U10")
        for c in self.COLS:
            setattr(self, c, df[c].to_numpy(dtype=float))
        bad = (self.high < self.low) | ~np.isfinite(self.close) | ~np.isfinite(self.raw_close)
        if sym != "CL=F":
            bad |= (self.close <= 0) | (self.open <= 0)
        if kind == "equity":
            bad |= self.raw_volume <= 0
        if kind in ("index", "etf", "fx"):
            r = np.r_[0.0, self.close[1:] / self.close[:-1] - 1]
            jump = np.abs(r) > 0.2
            bad |= jump | np.r_[jump[1:], False]
        self.bad = bad
        self.badcum = np.r_[0, np.cumsum(bad)]

    def __len__(self):
        return len(self.d)

    def i_le(self, s):
        return int(np.searchsorted(self.d, s, side="right")) - 1

    def i_ge(self, s):
        return int(np.searchsorted(self.d, s, side="left"))

    def is_td(self, s):
        i = self.i_le(s)
        return i >= 0 and self.d[i] == s

    def clean(self, i0, i1):
        if i0 < 0 or i1 >= len(self.d) or i0 > i1:
            return False
        return self.badcum[i1 + 1] - self.badcum[i0] == 0


class Fred:
    def __init__(self, sid, df):
        self.sid = sid
        self.d = df["date"].to_numpy().astype("U10")
        self.v = df["value"].to_numpy(dtype=float)
        self.vv = df["value_vintage"].to_numpy(dtype=float)

    def get(self, s):
        """Value on exactly date s; Skip when missing or the two vintages disagree."""
        i = int(np.searchsorted(self.d, s))
        if i >= len(self.d) or self.d[i] != s:
            raise Skip("missing_row")
        return self._val(i)

    def _val(self, i):
        v, vv = self.v[i], self.vv[i]
        if not (np.isfinite(v) and np.isfinite(vv)):
            raise Skip("missing_row")
        if abs(v - vv) > 1e-9 * max(1.0, abs(v)):
            raise Skip("fred_vintage_disagree")
        return float(v)

    def pair(self, s):
        """(as-of-cutoff vintage value, latest value) on exactly date s."""
        i = int(np.searchsorted(self.d, s))
        if i >= len(self.d) or self.d[i] != s or not np.isfinite(self.vv[i]):
            raise Skip("missing_row")
        return float(self.vv[i]), float(self.v[i])

    def le(self, s):
        """(date, value) of the latest observation on or before s, consistent across vintages."""
        j = int(np.searchsorted(self.d, s, side="right")) - 1
        # Rows can exist with a value in only one vintage; the latest usable date must agree.
        while j >= 0 and not np.isfinite(self.v[j]) and not np.isfinite(self.vv[j]):
            j -= 1
        if j < 0:
            raise Skip("missing_row")
        return str(self.d[j]), self._val(j)


class Snapshot:
    def __init__(self, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "market_v1_cache.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                raw = pickle.load(f)
        else:
            raw = self._read_db()
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(raw, f, protocol=4)
            os.replace(tmp, path)
        self.meta = raw["meta"]
        self.sec = raw["securities"].set_index("symbol")
        self.px = {}
        for sym, g in raw["prices"].groupby("symbol", sort=True):
            self.px[sym] = Px(sym, self.sec.loc[sym, "kind"], g)
        acts = raw["actions"]
        self.splits = {}
        self.divs = {}
        for sym, g in acts.groupby("symbol", sort=True):
            sp = g[g.kind == "split"].sort_values("date")
            dv = g[g.kind == "dividend"].sort_values("date")
            self.splits[sym] = list(sp[["date", "value", "ratio_text", "split_type"]].itertuples(index=False, name=None))
            self.divs[sym] = (dv["date"].to_numpy().astype("U10"), dv["value"].to_numpy(float),
                              dv["value_raw"].to_numpy(float))
        self.fred_series = raw["fred_series"].set_index("series_id")
        self.fred = {sid: Fred(sid, g) for sid, g in raw["fred"].groupby("series_id", sort=True)}
        self.companies = raw["sec_companies"].set_index("cik")
        self.facts = raw["facts"]

    @staticmethod
    def _read_db():
        con = ro_connect()
        q = lambda s, p=(): pd.read_sql_query(s, con, params=p)
        raw = {
            "meta": dict(con.execute("select key, value from meta").fetchall()),
            "securities": q("select * from securities"),
            "prices": q("select * from prices order by symbol, date"),
            "actions": q("select * from actions"),
            "fred_series": q("select * from fred_series"),
            "fred": q("select * from fred order by series_id, date"),
            "sec_companies": q("select * from sec_companies"),
        }
        tags = SEC_DURATION_TAGS + SEC_INSTANT_TAGS + [DEI_SHARES]
        ph = ",".join("?" * len(tags))
        raw["facts"] = q(
            "select cik, taxonomy, tag, unit, val, fy, fp, form, start, \"end\", filed, accn "
            "from facts where tag in (%s) and taxonomy in ('us-gaap','dei')" % ph, tags)
        con.close()
        return raw


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dumps(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False)
