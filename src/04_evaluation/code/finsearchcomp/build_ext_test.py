"""Build the FinSearchComp external test set for the market environment.

Candidates: Global T2/T3 rows annotated coverable_v1 = "yes", plus "partial" rows whose only gap is a fact the prompt
can state (needs_world_knowledge set). Each candidate's reference computation (questions.py) runs against the snapshot;
a question is kept only when every snapshot value agrees with its key under the key's tolerance and no drop decision
(DECISIONS) applies. Writes, under market_v1/external/:
  finsearchcomp_ext_test.jsonl   kept tasks, in the shared market task format;
  finsearchcomp_decisions.jsonl  every candidate with keep/drop, reason category, snapshot and reference values.

Usage: python -B build_ext_test.py [--dry]
"""
import json
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snap  # noqa: E402
from questions import Q, SRC  # noqa: E402  (SRC: the local FinSearchComp data file)

M_DIR = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1"
ANN = f"{M_DIR}/finsearchcomp_needs/annotations.jsonl"
OUT = f"{M_DIR}/external"

# Drops decided before or regardless of the comparison: (category, reason).
PRE_DROP = {
    "T3_b013": ("duplicate", "Same question and answer as T3_221 (kept)."),
    "T3_218": ("convention unclear", "Besides DJIA membership (statable), PE-TTM's EPS window and share base are not "
                                     "defined; the annotation marks the convention vague."),
    "T3_050": ("convention unclear", "Annotation flags the computation as ambiguous: the std's ddof and whether the "
                                     "first 19 days use a warm-up window are not stated, and ddof alone moves the "
                                     "result by about 2.6% (far beyond the 0.01 tolerance)."),
    "T3_060": ("convention unclear", "Annotation flags the wording as ambiguous: 'peak' can be the closing or the "
                                     "intraday peak (different dates in the snapshot) and '10 days' can be calendar or "
                                     "trading days."),
}


def short(pid):
    return pid.replace("(T2)Simple_Historical_Lookup_", "T2_").replace("(T3)Complex_Historical_Investigation_", "T3_")


def decimals(tol):
    """Rounding decimals implied by a power-of-ten tolerance (0.01 -> 2, 1 -> 0), else None."""
    if tol <= 0:
        return None
    e = math.log10(tol)
    return max(0, -int(round(e))) if abs(e - round(e)) < 1e-9 else None


def agrees(key, s):
    """Does the snapshot value s satisfy key (s rounded to the key's unit when the tolerance is a power of ten)?"""
    k = key["kind"]
    if k == "number":
        try:
            s = float(s)
        except (TypeError, ValueError):
            return False
        if math.isnan(s):
            return False
        v = key["value"]
        tol = max(key.get("tol", 0.0), key.get("rel_tol", 0.0) * abs(v))
        d = decimals(key["tol"]) if "tol" in key else None
        x = round(s, d) if d is not None else s
        if key.get("sign_free"):
            x, v = abs(x), abs(v)
        return abs(x - v) <= tol + 1e-9
    if k == "set":
        return bool(s)
    if k == "date":
        return str(s) == key["value"]
    vals = [key["value"]] + key.get("aliases", [])
    return str(s).lower() in [str(a).lower() for a in vals]


def main(dry):
    ann = [json.loads(l) for l in open(ANN)]
    src = {r["prompt_id"]: r for r in json.load(open(SRC))}
    cands = [r for r in ann if r["coverable_v1"] == "yes" or
             (r["coverable_v1"] == "partial" and r["needs_world_knowledge"])]
    tasks, decisions = [], []
    for r in cands:
        pid, sid = r["prompt_id"], short(r["prompt_id"])
        o = src[pid]
        tier = "T2" if pid.startswith("(T2)") else "T3"
        sources = sorted({d["source"] for d in r["required_data"]})
        dec = {"prompt_id": pid, "short_id": sid, "tier": tier, "coverable_v1": r["coverable_v1"], "sources": sources}
        res, err = None, None
        if sid in Q:
            try:
                res = Q[sid]()
            except Exception as e:  # a computation error is recorded, not hidden
                err = f"{type(e).__name__}: {e}"
                traceback.print_exc(limit=1)
        if res:
            ok = [agrees(k, s) for k, s in zip(res["keys"], res["snap"])]
            assert len(res["keys"]) == len(res["snap"]), sid
            dec.update(keys=res["keys"], snapshot_values=res["snap"], key_agrees=ok, solution=res["sol"], note=res["note"])
        if sid in PRE_DROP:
            dec.update(decision="drop", category=PRE_DROP[sid][0], reason=PRE_DROP[sid][1])
        elif err or not res:
            cat, why = POST_DROP.get(sid, ("other", "no reference computation"))
            dec.update(decision="drop", category=cat, reason=why + (f" (computation error: {err})" if err else ""))
        elif all(ok):
            dec.update(decision="keep", category="", reason="")
        else:
            cat, why = POST_DROP.get(sid, ("unreviewed", "snapshot disagrees with reference"))
            dec.update(decision="drop", category=cat, reason=why)
        decisions.append(dec)
        if dec["decision"] == "keep":
            prompt = o["prompt"].rstrip()
            if res["append"]:
                prompt += "\n" + " ".join(res["append"])
            tasks.append({
                "task_id": f"fsc-{pid}", "env": "market", "split": "ext_test", "template": f"finsearchcomp_{tier}",
                "prompt": prompt, "db": snap.DB, "keys": res["keys"], "answer_text": o["response_reference"],
                "meta": {"prompt_id": pid, "label": r["label"], "snapshot_value": res["snap"],
                         "reference_value": [k["value"] if k["kind"] != "set" else [x["value"] for x in k["value"]]
                                             for k in res["keys"]],
                         "appended": ("; ".join(res["append"]) + (" (source: see sources)" if res["append"] else "")),
                         "sources": res["src"], "data_sources": sources, "note": res["note"],
                         "license": "FinSearchComp question and reference answer: CC BY 4.0 (ByteSeedXpert/FinSearchComp)"},
                "solution": res["sol"]})
        flag = dec["decision"].upper()
        print(f"{sid:9s} {flag:4s} {dec.get('category', ''):18s} snap={fmt(dec.get('snapshot_values'))} "
              f"ref={fmt([k.get('value') for k in dec.get('keys', [])])}")
    if not dry:
        os.makedirs(OUT, exist_ok=True)
        for name, rows in (("finsearchcomp_ext_test.jsonl", tasks), ("finsearchcomp_decisions.jsonl", decisions)):
            tmp = f"{OUT}/{name}.tmp"
            with open(tmp, "w") as f:
                for x in rows:
                    f.write(json.dumps(x, ensure_ascii=False, default=float) + "\n")
            os.replace(tmp, f"{OUT}/{name}")
    print(f"candidates {len(cands)} kept {len(tasks)}")


def fmt(v):
    if v is None:
        return "-"
    out = []
    for x in v:
        if isinstance(x, float):
            out.append(f"{x:.4f}")
        elif isinstance(x, list):
            out.append("[" + ",".join(str(y.get("value")) if isinstance(y, dict) else str(y) for y in x) + "]")
        else:
            out.append(str(x))
    return "|".join(out)


# Drops after the comparison and one investigation: (category, reason). Filled after review of each mismatch.
POST_DROP = {
    "T2_580": ("vintage", "DSPI 2025-04 is 22,918.1 in the cutoff vintage (22,947.9 latest) vs reference 22,579.6: the "
                          "reference predates BEA's 2025 annual update; the snapshot has no earlier vintage."),
    "T2_597": ("vintage", "M2SL 2024-12 is 21,424.5 in the cutoff vintage (21,462.0 latest) vs reference 21,442: M2 "
                          "history is revised; the snapshot has no earlier vintage."),
    "T2_583": ("vendor/data difference", "Yahoo ^FTSE close 2024-07-04 is 8,241.30 vs reference 8,241.26 (tolerance 0.01)."),
    "T3_006": ("vendor/data difference", "Lowest 2011 MSFT close in Yahoo is 23.71 (2011-06-10), giving 421,763 shares; the "
                                         "reference 421,852 implies a close of about 23.705, which no Yahoo bar has "
                                         "(tolerance 10 shares)."),
    "T2_625": ("convention unclear", "The snapshot has two facts matching 'marketing expenses': MarketingExpense 2,500 and "
                                     "SellingAndMarketingExpense 4,626 (the reference, the income-statement line); an agent "
                                     "reading tag names can defensibly answer 2,500."),
    "T3_b009": ("other", "Tesla tags its cash-flow line 'Depreciation, amortization and impairment' with a company "
                         "extension tag, which the snapshot (us-gaap, ifrs-full, dei only) does not hold; us-gaap "
                         "Depreciation alone gives a different ratio."),
}

if __name__ == "__main__":
    main("--dry" in sys.argv)
