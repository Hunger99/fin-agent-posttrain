"""Fiscal-year index over the SEC XBRL facts in the snapshot.

The `fy` and `fp` fields of a fact describe the filing, not the period of the
value, so a fiscal year is resolved to the company's original 10-K for that
year and values are then selected inside that filing by period end date.
"""
import numpy as np
import pandas as pd

from common import DEI_SHARES, Skip

ANNUAL_DAYS = (350, 380)
NINE_MONTH_DAYS = (250, 290)


class SecIndex:
    def __init__(self, facts):
        f = facts.copy()
        f["start"] = f["start"].fillna("")
        sd = pd.to_datetime(f["start"].where(f["start"] != ""), errors="coerce")
        ed = pd.to_datetime(f["end"], errors="coerce")
        fd = pd.to_datetime(f["filed"], errors="coerce")
        f["dur"] = (ed - sd).dt.days
        f["lag"] = (fd - ed).dt.days
        self.by = {k: g for k, g in f.groupby(["cik", "tag"], sort=True)}
        ann = f[(f.form == "10-K") & (f.fp == "FY") & (f.taxonomy == "us-gaap")
                & f.dur.between(*ANNUAL_DAYS) & (f.lag >= 0) & (f.lag <= 200)]
        per_accn = ann.groupby("accn").agg(cik=("cik", "first"), E=("end", "max"),
                                          filed=("filed", "min"),
                                          fy=("fy", lambda s: int(s.mode().iloc[0])))
        self.fy = {}
        self.fy_ambiguous = 0
        for (cik, fy), g in per_accn.groupby(["cik", "fy"], sort=True):
            if g.E.nunique() != 1:
                self.fy_ambiguous += 1
                continue
            row = g.sort_values("filed").iloc[0]
            E = row.E
            y, m = int(E[:4]), int(E[5:7])
            # A label must name the year the period ends in, or the prior year for Jan-Mar ends.
            if not (fy == y or (m <= 3 and fy == y - 1)):
                self.fy_ambiguous += 1
                continue
            accn = g.sort_values("filed").index[0]
            a = ann[(ann.accn == accn) & (ann.end == E)]
            S = a.start.mode().iloc[0]
            self.fy[(int(cik), int(fy))] = {"accn": accn, "E": E, "S": S, "filed": row.filed}

    def years(self, cik):
        if not hasattr(self, "_years"):
            self._years = {}
            for (c, fy) in self.fy:
                self._years.setdefault(c, []).append(fy)
            for c in self._years:
                self._years[c].sort()
        return self._years.get(cik, [])

    def info(self, cik, fy):
        i = self.fy.get((cik, fy))
        if i is None:
            raise Skip("sec_no_fiscal_year")
        return i

    def _rows(self, cik, tag):
        g = self.by.get((cik, tag))
        if g is None:
            raise Skip("missing_row")
        unit = "USD/shares" if tag.startswith("EarningsPerShare") else "USD"
        return g[g.unit == unit]

    def value(self, cik, fy, tag, kind, which="current"):
        """Value of `tag` in the original 10-K for fiscal year `fy`.

        kind 'dur' selects the annual period ending at the fiscal year end; kind
        'inst' selects the balance at the year end ('current') or the prior year
        end ('prior') as presented in the same filing. Returns (value, end, start).
        """
        i = self.info(cik, fy)
        g = self._rows(cik, tag)
        g = g[g.accn == i["accn"]]
        E = i["E"]
        if kind == "dur":
            r = g[(g.end == E) & g.dur.between(*ANNUAL_DAYS)]
        elif which == "current":
            r = g[(g.end == E) & (g.start == "")]
        else:
            lo = (pd.Timestamp(E) - pd.Timedelta(days=380)).strftime("%Y-%m-%d")
            hi = (pd.Timestamp(E) - pd.Timedelta(days=350)).strftime("%Y-%m-%d")
            r = g[(g.start == "") & (g.end >= lo) & (g.end <= hi)]
            if r.end.nunique() > 1:
                raise Skip("sec_conflict")
        if len(r) == 0:
            raise Skip("missing_row")
        vals = np.unique(r.val.to_numpy(float))
        if len(vals) != 1 or not np.isfinite(vals[0]):
            raise Skip("sec_conflict")
        return float(vals[0]), r.end.iloc[0], r.start.iloc[0]

    def restated(self, cik, tag, start, end, val):
        """True when any filing reports a different value for the same tag and period."""
        g = self._rows(cik, tag)
        r = g[(g.start == start) & (g.end == end)]
        return bool((np.abs(r.val.to_numpy(float) - val) > 0.5).any())

    def nine_month(self, cik, fy, tag):
        """Year-to-date value through Q3 from the original Q3 10-Q of fiscal year fy."""
        i = self.info(cik, fy)
        g = self._rows(cik, tag)
        r = g[(g.form == "10-Q") & (g.fy == fy) & (g.fp == "Q3") & (g.start == i["S"])
              & g.dur.between(*NINE_MONTH_DAYS)]
        if len(r) == 0:
            raise Skip("missing_row")
        first = r.filed.min()
        r = r[r.filed == first]
        vals = np.unique(r.val.to_numpy(float))
        if len(vals) != 1:
            raise Skip("sec_conflict")
        return float(vals[0]), r.end.iloc[0], r.accn.iloc[0]

    def shares(self, cik, on_or_before):
        """Cover-page shares outstanding from the latest filing filed on or before a date."""
        g = self.by.get((cik, DEI_SHARES))
        if g is None:
            raise Skip("missing_row")
        g = g[(g.unit == "shares") & (g.filed <= on_or_before)]
        if len(g) == 0:
            raise Skip("missing_row")
        last = g.filed.max()
        r = g[g.filed == last]
        vals = np.unique(r.val.to_numpy(float))
        if len(vals) != 1:
            raise Skip("sec_multi_class_or_conflict")
        return float(vals[0]), r.end.iloc[0], last, r.accn.iloc[0]
