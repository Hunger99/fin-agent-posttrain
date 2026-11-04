"""Simple machine-learning tasks in the wealth-management environment (docs/results.md, Overview Version 4).

The agent builds a model on a training table and submits it; the program scores it on a hidden test table. There is
no code execution: the agent works through structured tools, so model-written code never runs on the shared host
(research/log/2026-10-27-0321.md):
- ml_describe_data: columns, types, missing values, sizes, target and metric;
- ml_column_stats: one column's summary and its relation to the target;
- ml_train: a model (linear_regression, ridge, logistic_regression or knn) on chosen features, with optional
  imputation, log1p transforms and standardization; categorical columns are one-hot encoded; returns the 5-fold
  cross-validated metric on the training table;
- ml_submit: choose a trained model; the last submission counts.
Two templates draw a fresh table from the task's seed: ml_redemption (will a customer redeem within 30 days, AUC) and
ml_volatility (a fund's next-quarter volatility, R^2). Each table has a skewed column that wants a log transform,
missing values that need imputation, a categorical column and noise columns. The generating model is known, so the
oracle metric (the true probabilities or the noiseless target on the test table) bounds what is reachable.
Version 2 (research/log/2026-10-29-0836.md): ml_describe_data also gives each numeric column's skewness and its
correlation with the target, as is and after log1p; in the diagnosis a log1p of the skewed column was what passing
models had, and only ml_column_stats, called one column at a time, showed it (research/log/2026-10-29-0743.md).
A task passes when a model was submitted and its test metric is at least the oracle's minus MARGIN; the margins were
chosen by the model in unattended mode before any run, for the author to review.

Usage: python ml_tasks.py make <split> <n> <seed> <out.jsonl>
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))
import agent  # noqa: E402
from env import core  # noqa: E402

MARGIN = {"auc": 0.015, "r2": 0.03}
MODELS = ("linear_regression", "ridge", "logistic_regression", "knn")


def table(template, seed):
    """(train columns, train target, test columns, test target, oracle prediction on test, target name, metric)."""
    rng = np.random.default_rng(seed)
    n = 4500 if template == "ml_redemption" else 1800
    if template == "ml_redemption":
        assets = np.exp(rng.normal(math.log(60_000), 1.3, n))
        cash = rng.beta(2, 6, n)
        ret = rng.normal(5, 12, n)
        logins = rng.poisson(6, n).astype(float)
        channel = rng.choice(["app", "web", "branch"], n, p=[0.5, 0.3, 0.2])
        age = rng.integers(22, 85, n).astype(float)
        z = lambda x: (x - x.mean()) / x.std()
        logit = (-1.2 - 1.5 * z(np.log(assets)) + 0.7 * z(cash) - 0.8 * z(ret) - 0.6 * z(logins)
                 + 0.5 * (channel == "branch") + 0.2 * z(age))
        p = 1 / (1 + np.exp(-logit))
        y = (rng.random(n) < p).astype(float)
        logins[rng.random(n) < 0.15] = np.nan
        cols = {"total_assets": assets, "cash_share": cash, "return_12m_pct": ret, "logins_30d": logins,
                "channel": channel, "age": age, "tenure_months": rng.integers(1, 240, n).astype(float),
                "n_products": rng.integers(1, 9, n).astype(float),
                **{f"noise_{k}": rng.normal(0, 1, n) for k in (1, 2, 3)}}
        target, metric, oracle = "redeemed_within_30d", "auc", p
    else:
        vol3 = rng.gamma(4, 3.5, n)
        vol12 = vol3 * rng.uniform(0.7, 1.3, n)
        ftype = rng.choice(["bond", "mixed", "index", "equity"], n)
        aum = np.exp(rng.normal(math.log(400), 1.4, n))
        effect = {"bond": -2.0, "mixed": 0.0, "index": 1.0, "equity": 2.5}
        mean = 0.5 * vol3 + 0.3 * vol12 + np.vectorize(effect.get)(ftype) - 1.5 * np.log(aum) + 10
        y = mean + rng.normal(0, 1.5, n)
        vol3_obs = vol3.copy()
        vol3_obs[rng.random(n) < 0.1] = np.nan
        cols = {"vol_3m_pct": vol3_obs, "vol_12m_pct": vol12, "type": ftype, "aum_musd": aum,
                "mgmt_fee_pct": rng.uniform(0.2, 1.8, n), "age_years": rng.uniform(1, 30, n),
                **{f"noise_{k}": rng.normal(0, 1, n) for k in (1, 2)}}
        target, metric, oracle = "vol_next_quarter_pct", "r2", mean
    k = int(n * 2 / 3)
    tr = {c: v[:k] for c, v in cols.items()}
    te = {c: v[k:] for c, v in cols.items()}
    return tr, y[:k], te, y[k:], oracle[k:], target, metric


def auc(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks of ties
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


SCORE = {"auc": auc, "r2": r2}


class Prep:
    """Feature preparation fitted on training rows: imputation, log1p, one-hot, standardization."""

    def __init__(self, cols, features, log_features, impute, standardize):
        self.features, self.log, self.impute, self.std = features, set(log_features), impute, standardize
        self.cats = {f: sorted(set(cols[f])) for f in features if cols[f].dtype.kind in "OU"}
        x = self.raw(cols)
        self.mu, self.sd = np.nanmean(x, 0), np.nanstd(x, 0) + 1e-9
        self.fill = np.nanmedian(x, 0) if impute == "median" else self.mu

    def raw(self, cols):
        parts = []
        for f in self.features:
            v = cols[f]
            if f in self.cats:
                parts += [(v == c).astype(float) for c in self.cats[f][1:]]
            else:
                v = v.astype(float)
                if f in self.log:
                    if np.nanmin(v) <= -1:
                        raise core.ToolError(f"log1p needs values above -1; {f} has smaller ones")
                    v = np.log1p(v)
                parts.append(v)
        return np.stack(parts, 1)

    def __call__(self, cols):
        x = self.raw(cols)
        if np.isnan(x).any():
            if not self.impute:
                bad = [f for f in self.features if f not in self.cats and np.isnan(cols[f].astype(float)).any()]
                raise core.ToolError(f"missing values in {bad}; set impute to mean or median")
            x = np.where(np.isnan(x), self.fill, x)
        return (x - self.mu) / self.sd if self.std else x


def fit_predict(model, params, x, y, xt, classification):
    if model in ("linear_regression", "ridge"):
        lam = float(params.get("alpha", 1.0)) if model == "ridge" else 1e-8
        a = np.c_[np.ones(len(x)), x]
        w = np.linalg.solve(a.T @ a + lam * np.diag([0] + [1] * x.shape[1]), a.T @ y)
        return np.c_[np.ones(len(xt)), xt] @ w
    if model == "logistic_regression":
        if not classification:
            raise core.ToolError("logistic_regression is for a 0/1 target")
        lam = 1 / float(params.get("C", 1.0))
        a, w = np.c_[np.ones(len(x)), x], np.zeros(x.shape[1] + 1)
        for _ in range(25):  # Newton steps with an L2 penalty on the weights
            p = 1 / (1 + np.exp(-np.clip(a @ w, -30, 30)))
            g = a.T @ (p - y) + lam * np.r_[0, w[1:]]
            h = (a * (p * (1 - p))[:, None]).T @ a + lam * np.diag([0] + [1] * x.shape[1])
            w -= np.linalg.solve(h, g)
        return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(xt)), xt] @ w, -30, 30)))
    if model == "knn":
        k = int(params.get("k", 25))
        out = np.empty(len(xt))
        for i in range(0, len(xt), 256):
            d = ((xt[i:i + 256, None, :] - x[None]) ** 2).sum(-1)
            out[i:i + 256] = y[np.argsort(d, 1)[:, :k]].mean(1)
        return out
    raise core.ToolError(f"model must be one of {', '.join(MODELS)}")


class MLEnv:
    def __init__(self, task):
        self.tr, self.y, self.te, self.yt, self.oracle, self.target, self.metric = table(task["template"], task["seed"])
        self.models, self.submitted, self.finished, self.answer = {}, None, False, None

    def ml_describe_data(self):
        cols = []
        for c, v in self.tr.items():
            if v.dtype.kind in "OU":
                cols.append({"column": c, "type": "categorical", "categories": sorted(set(v))})
            else:
                m = ~np.isnan(v)
                x, y = v[m], self.y[m]
                cols.append({"column": c, "type": "numeric", "missing": int((~m).sum()),
                             "mean": round(float(x.mean()), 4), "std": round(float(x.std()), 4),
                             "min": round(float(x.min()), 4), "max": round(float(x.max()), 4),
                             "skewness": round(float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12)), 3),
                             "correlation_with_target": round(float(np.corrcoef(x, y)[0, 1]), 4),
                             "correlation_of_log1p_with_target": (round(float(np.corrcoef(np.log1p(x), y)[0, 1]), 4)
                                                                  if x.min() > -1 else None)})
        return {"target": self.target, "metric": self.metric, "train_rows": len(self.y), "test_rows": len(self.yt),
                "columns": cols, "target_mean": round(float(self.y.mean()), 4)}

    def ml_column_stats(self, column):
        if column not in self.tr:
            raise core.ToolError(f"no column {column!r}")
        v = self.tr[column]
        if v.dtype.kind in "OU":
            return {"column": column, "target_mean_by_category": {c: round(float(self.y[v == c].mean()), 4) for c in sorted(set(v))}}
        m = ~np.isnan(v)
        q = np.percentile(v[m], [1, 25, 50, 75, 99])
        return {"column": column, "missing": int((~m).sum()), "percentiles_1_25_50_75_99": [round(float(x), 4) for x in q],
                "correlation_with_target": round(float(np.corrcoef(v[m], self.y[m])[0, 1]), 4),
                "correlation_of_log1p_with_target": (round(float(np.corrcoef(np.log1p(v[m]), self.y[m])[0, 1]), 4)
                                                     if v[m].min() > -1 else None)}

    def ml_train(self, model, features, log_features=(), impute=None, standardize=True, params=None):
        bad = [f for f in list(features) + list(log_features) if f not in self.tr]
        if bad or not features:
            raise core.ToolError(f"unknown or no features: {bad}")
        params, classification = params or {}, self.metric == "auc"
        folds, scores = np.arange(len(self.y)) % 5, []
        for k in range(5):
            tr = {c: v[folds != k] for c, v in self.tr.items()}
            va = {c: v[folds == k] for c, v in self.tr.items()}
            prep = Prep(tr, list(features), log_features, impute, standardize)
            pred = fit_predict(model, params, prep(tr), self.y[folds != k], prep(va), classification)
            scores.append(SCORE[self.metric](self.y[folds == k], pred))
        mid = f"m{len(self.models) + 1}"
        self.models[mid] = (model, list(features), list(log_features), impute, standardize, params)
        return {"model_id": mid, f"cv_{self.metric}": round(float(np.mean(scores)), 4), "folds": 5}

    def ml_submit(self, model_id):
        if model_id not in self.models:
            raise core.ToolError(f"no model {model_id!r}; train one with ml_train")
        self.submitted = model_id
        return {"status": "submitted", "model_id": model_id}

    def finish(self, answer):
        self.finished, self.answer = True, str(answer)
        return {"status": "finished"}

    def test_score(self):
        if self.submitted is None:
            return None
        model, features, log_features, impute, standardize, params = self.models[self.submitted]
        prep = Prep(self.tr, features, log_features, impute, standardize)
        pred = fit_predict(model, params, prep(self.tr), self.y, prep(self.te), self.metric == "auc")
        return SCORE[self.metric](self.yt, pred)

    def oracle_score(self):
        return SCORE[self.metric](self.yt, self.oracle)


SYSTEM = """You build a predictive model for a fund platform with the machine-learning tools provided. ml_describe_data shows the training table, its target and the metric; ml_column_stats shows one column; ml_train fits a model on chosen features and returns its 5-fold cross-validated metric; ml_submit chooses the model that is scored on a hidden test table (the last submission counts). Act without asking questions. When done, call finish with a short answer naming the submitted model and its cross-validated metric."""
ARR = {"type": "array", "items": agent.S}
TOOLS = [
    agent.schema("ml_describe_data", "The training table: columns, types, missing values, skewness, correlations with the "
                 "target (as is and after log1p), target and metric.", {}, []),
    agent.schema("ml_column_stats", "One column's distribution and its relation to the target.", {"column": agent.S}, ["column"]),
    agent.schema("ml_train", "Fit a model with 5-fold cross-validation on the training table.",
                 {"model": {"type": "string", "enum": list(MODELS)}, "features": ARR, "log_features": ARR,
                  "impute": {"type": "string", "enum": ["mean", "median"]}, "standardize": agent.B,
                  "params": {"type": "object", "properties": {"alpha": agent.N, "C": agent.N, "k": agent.I},
                             "additionalProperties": False}}, ["model", "features"]),
    agent.schema("ml_submit", "Choose the trained model to be scored on the hidden test table.", {"model_id": agent.S},
                 ["model_id"]),
    agent.schema("finish", "End the task with a short answer.", {"answer": agent.S}, ["answer"]),
]


class MLEpisode(agent.Episode):
    SYSTEM, TOOLS = SYSTEM, TOOLS

    def make_env(self, _):
        return MLEnv(self.task)

    def verify(self, answer):
        problems = super().verify(answer)
        if self.env.submitted is None:
            problems.append("no model has been submitted with ml_submit")
        return problems

    def grade(self):
        score, oracle = self.env.test_score(), self.env.oracle_score()
        ok = self.env.answer is not None and score is not None and score >= oracle - MARGIN[self.env.metric]
        return {"passed": ok, "test_metric": score, "oracle_metric": oracle, "metric": self.env.metric}

    def categorize(self, g):
        if g["passed"]:
            return None
        if self.end in ("context_overflow", "step_limit"):
            return self.end
        if max(self.calls.values(), default=0) >= 3:
            return "loop"
        return "no_submission" if g["test_metric"] is None else "weak_model"


PROMPT = {"ml_redemption": "Build a model that predicts whether a customer will redeem within 30 days (the target "
                           "redeemed_within_30d), from the customer table. It is scored by AUC on a hidden test table. "
                           "Explore the data, compare a few models by cross-validation, and submit the best one.",
          "ml_volatility": "Build a model that predicts a fund's annualized volatility over the next quarter (the target "
                           "vol_next_quarter_pct, in percent), from the fund table. It is scored by R^2 on a hidden test "
                           "table. Explore the data, compare a few models by cross-validation, and submit the best one."}


def make(split, n, seed, out):
    rng = np.random.default_rng(seed)
    with open(out, "w", encoding="utf-8") as f:
        for i in range(n):
            tpl = ["ml_redemption", "ml_volatility"][i % 2]
            f.write(json.dumps({"task_id": f"ml-{split}-{i:04d}", "env": "ml", "instance": "ml", "split": split, "type": "ML",
                                "template": tpl, "feasible": True, "seed": int(rng.integers(2**31)),
                                "prompt": PROMPT[tpl]}) + "\n")


if __name__ == "__main__" and sys.argv[1] == "make":
    make(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
