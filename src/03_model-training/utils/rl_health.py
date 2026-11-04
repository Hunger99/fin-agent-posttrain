#!/usr/bin/env python3
"""RL health monitor for verl GRPO / PPO training runs.  Read-only: it never writes, signals or starts anything.

Usage
  python -B rl_health.py RUN_DIR [--prev EARLIER_RUN_DIR] [--log TRAIN_LOG] [--json] [options]

Inputs
  RUN_DIR/metrics.jsonl   one JSON line per finished step, {"step": s, "data": {...}} (a flat {"step": s, key: v} line is
                          accepted too).  A partially written last line is ignored; duplicate steps keep the last line.
  RUN_DIR/episodes/*.jsonl optional, one row per rollout episode, one file per agent-loop worker, written in step order
                          (fields: task_id, template, reward, end, turns, task_steps, prompt_tokens, response_tokens,
                          model_tokens, loss_masked, tool_errors, verification_rejects, seconds).  Rows whose task_id
                          contains --dev-marker ("-dev-") are validation rows; a cluster of them after step s is the
                          validation round of step s.  Rows are mapped to steps by count: the rows per worker per step are
                          detected from step 1 (prompt+response token sum == perf/total_num_tokens) unless given.  The
                          files are streamed and parsed only up to the cut-off step; later rows are only counted.
  --prev EARLIER_RUN_DIR  an earlier run of the same model lineage to prepend (e.g. run1 before its continuation run1b).
                          Baselines and trends then span both runs; KL-to-reference, learning-rate and timing rules stay
                          within one run because the reference, the schedule and the hardware may change between runs.
  --log TRAIN_LOG         optional training log: error signatures, prompt-truncation warnings, unknown tool names, the
                          Qwen3.5 packing patch, and config values (max_prompt_length, grad_clip, total_training_steps,
                          test_freq, save_freq, trainer_mode, adv_estimator, rollout_is) when not given as options.
  --live                  also run host checks for a live run: metrics.jsonl age (hang), --pid liveness, free disk,
                          foreign processes on --gpus, orphaned stub-test loops.

Output
  One line per rule: STATUS RULE title | value | threshold | step | note.  STATUS is ok, watch or alert, or skip when the
  rule's keys or inputs are absent.  Each rule is judged at the latest complete step and over its trend; an alert that
  happened at an earlier step but no longer holds is reported as watch, naming that step.  --json prints the same as
  JSON, with each rule's healthy range, source, matching pitfall record and fix.
  Exit code: 0 when no rule is alert, 2 when any is (1 for usage errors).

Works for GRPO and PPO: PPO runs (critic/vf_loss present) get the critic rules; every rule skips keys that are absent.
"""
import argparse
import datetime
import glob
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict

OK, WATCH, ALERT, SKIP = "ok", "watch", "alert", "skip"
RANK = {SKIP: -1, OK: 0, WATCH: 1, ALERT: 2}
EP_FIELDS = ("task_id", "template", "reward", "end", "turns", "task_steps", "prompt_tokens", "response_tokens",
             "model_tokens", "loss_masked", "tool_errors", "verification_rejects", "seconds")
KNOWN_ENDS = {"finished", "overlong", "step_limit", "hard_cap"}
TRAIN_MARKERS = ("timing_s/step", "actor/entropy", "actor/pg_loss", "critic/vf_loss", "training/global_step")


# ----------------------------------------------------------------------------------------------------------------- utils
def worst(*ss):
    ss = [s for s in ss if s is not None]
    return max(ss, key=lambda s: RANK[s]) if ss else OK


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def hi(v, w, a):
    """Status of a value where larger is worse; a non-finite value is alert."""
    if not fin(v):
        return ALERT
    return ALERT if v > a else WATCH if v > w else OK


def lo(v, w, a):
    """Status of a value where smaller is worse; a non-finite value is alert."""
    if not fin(v):
        return ALERT
    return ALERT if v < a else WATCH if v < w else OK


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, bool) or isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1e5 or a < 1e-3:
        return f"{x:.3g}"
    return f"{x:.{nd}g}"


def pct(x):
    return "-" if x is None else (f"{100 * x:.1f}%" if fin(x) else str(x))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = list(xs)
    return statistics.median(xs) if xs else None


def boot_ci(diffs, B, seed):
    """Mean and percentile 95% CI of the mean by bootstrap (numpy when available)."""
    n = len(diffs)
    if n == 0:
        return None
    m = sum(diffs) / n
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        a = np.asarray(diffs, dtype=float)
        means = []
        for start in range(0, B, 1000):
            b = min(1000, B - start)
            means.append(a[rng.integers(0, n, size=(b, n))].mean(axis=1))
        means = np.concatenate(means)
        lo_, hi_ = np.percentile(means, [2.5, 97.5])
        return m, float(lo_), float(hi_)
    except ImportError:
        rng = random.Random(seed)
        ms = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
        return m, ms[int(0.025 * B)], ms[min(B - 1, int(0.975 * B))]


def mcnemar_p(b, c):
    """Exact two-sided McNemar p for b and c discordant pairs."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# --------------------------------------------------------------------------------------------------------------- loading
def is_train(d):
    return any(k in d for k in TRAIN_MARKERS)


def load_metrics(path):
    info = {"bad_lines": 0, "partial_last": False, "duplicates": [], "exists": os.path.exists(path)}
    rows = {}
    if not info["exists"]:
        return rows, info
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            if i == len(lines) - 1:  # last line without its newline: still being written
                info["partial_last"] = True
            else:
                info["bad_lines"] += 1
            continue
        if not isinstance(obj, dict) or "step" not in obj:
            info["bad_lines"] += 1
            continue
        d = obj["data"] if isinstance(obj.get("data"), dict) else {k: v for k, v in obj.items() if k != "step"}
        s = int(obj["step"])
        if s in rows:
            info["duplicates"].append(s)
        rows[s] = d
    return rows, info


def compact(o, fi):
    r = {k: o.get(k) for k in EP_FIELDS}
    r["task_id"] = str(r["task_id"] or "")
    r["reward"] = float(r["reward"]) if fin(r["reward"]) else 0.0
    r["loss_masked"] = bool(r["loss_masked"])
    r["f"] = fi
    return r


class EpisodeReader:
    """Streams one worker file, keeping compact train rows and validation clusters up to what is asked for."""

    def __init__(self, path, fi, dev_marker):
        self.path, self.fi, self.dev = path, fi, dev_marker
        self.f = open(path, "rb")
        self.train, self.clusters = [], []  # clusters: [train rows before the cluster, [rows]]
        self.prev_dev, self.look, self.eof = False, None, False
        self.bad, self.partial, self.rest_train = 0, False, 0

    def _next(self):
        if self.look is not None:
            r, self.look = self.look, None
            return r
        while not self.eof:
            raw = self.f.readline()
            if not raw:
                self.close()
                return None
            if not raw.strip():
                continue
            try:
                return compact(json.loads(raw), self.fi)
            except ValueError:
                if raw.endswith(b"\n"):
                    self.bad += 1
                    continue
                self.partial = True  # last line still being written
                self.close()
                return None
        return None

    def _take(self, r):
        if self.dev in r["task_id"]:
            if not self.prev_dev or not self.clusters:
                self.clusters.append([len(self.train), []])
            self.clusters[-1][1].append(r)
            self.prev_dev = True
        else:
            self.train.append(r)
            self.prev_dev = False

    def read_train(self, k):
        while len(self.train) < k:
            r = self._next()
            if r is None:
                break
            self._take(r)
        return len(self.train) >= k

    def read_dev_tail(self):
        """Read the validation cluster that directly follows the train rows read so far."""
        while True:
            r = self._next()
            if r is None:
                return
            if self.dev in r["task_id"]:
                self._take(r)
            else:
                self.look = r
                return

    def count_rest(self):
        """Count the remaining train rows without parsing them (rows after the cut-off)."""
        n = 1 if self.look is not None else 0
        self.look = None
        if not self.eof:
            mark = self.dev.encode()
            for raw in self.f:
                if raw.strip() and raw.endswith(b"\n") and mark not in raw:
                    n += 1
            self.close()
        self.rest_train = n
        return n

    def close(self):
        if not self.eof:
            self.eof = True
            self.f.close()


def detect_n(rows):
    """Rollouts per task: within one step of one worker file every task appears exactly n times (rows are written in
    completion order, so a group's rows need not be adjacent)."""
    cnt = Counter(r["task_id"] for r in rows)
    return Counter(cnt.values()).most_common(1)[0][0] if cnt else None


def detect_p(readers, n, m1):
    """Rows per worker per step: the row count whose step-1 rows reproduce step-1 metrics (n, when known, is the step)."""
    if not m1:
        return None
    t_tok = m1.get("perf/total_num_tokens")
    t_score, t_resp = m1.get("critic/score/mean"), m1.get("response_length/mean")
    inc = n or 1
    k, prev, tok = inc, 0, 0
    while k <= 65536:
        if not all(r.read_train(k) for r in readers):
            return None
        for r in readers:
            for x in r.train[prev:k]:
                tok += (x["prompt_tokens"] or 0) + (x["response_tokens"] or 0)
        prev = k
        if fin(t_tok):
            if tok == int(t_tok):
                return k
            if tok > t_tok:
                return None
        elif fin(t_score) and fin(t_resp):
            rows = [x for r in readers for x in r.train[:k]]
            cap = m1.get("response_length/max") or 1e18
            if abs(mean(x["reward"] for x in rows) - t_score) < 1e-5 and \
                    abs(mean(min(x["response_tokens"] or 0, cap) for x in rows) - t_resp) < 0.5:
                return k
        else:
            return None
        k += inc
    return None


def agg_step(rows, budget):
    """Per-step episode statistics used by the rules (rows are discarded afterwards)."""
    N = len(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["f"], r["task_id"]), []).append(r)
    zero = two = pos_only = carry = 0
    sum_a = tw_num = tw_den = 0.0
    tmpl = defaultdict(Counter)
    task_pass, task_tmpl = defaultdict(list), {}
    for (_, tid), g in groups.items():
        rs = [x["reward"] for x in g]
        m = sum(rs) / len(rs)
        sd = statistics.stdev(rs) if len(rs) > 1 else 0.0
        zero += sd == 0
        pos = neg = False
        for x in g:
            if x["loss_masked"]:
                continue
            a = (x["reward"] - m) / (sd + 1e-6)
            sum_a += a
            mt = x["model_tokens"] or 0
            tw_num += a * mt
            tw_den += mt
            carry += abs(a) > 1e-6
            pos |= a > 1e-6
            neg |= a < -1e-6
        two += pos and neg
        pos_only += pos and not neg
        t = g[0]["template"] or "?"
        tmpl[t]["groups"] += 1
        tmpl[t]["allpass"] += all(v > 0.5 for v in rs)
        task_pass[tid].extend(rs)
        task_tmpl[tid] = t
    cap = budget or 10 ** 12
    masked = [r for r in rows if r["loss_masked"]]
    resp_cap = [min(r["response_tokens"] or 0, cap) for r in rows]
    passes = [r for r in rows if r["reward"] > 0.5]
    has = lambda k: any(r[k] is not None for r in rows)
    for r in rows:
        c = tmpl[r["template"] or "?"]
        c["n"] += 1
        c["rew"] += r["reward"]
        c["over"] += r["loss_masked"] or r["end"] == "overlong"
        c["resp"] += min(r["response_tokens"] or 0, cap)
        c["terr"] += r["tool_errors"] or 0
        if r["reward"] > 0.5:
            c["passes"] += 1
            c["unver"] += (r["verification_rejects"] or 0) >= 2
    valid_tok = sum(r["model_tokens"] or 0 for r in rows if not r["loss_masked"])
    n_valid = N - len(masked)
    secs = [r["seconds"] for r in rows if fin(r["seconds"])]
    return {
        "N": N, "groups": len(groups),
        "zero_std": zero / len(groups), "two_sided": two / len(groups), "pos_only": pos_only / len(groups),
        "carry": carry / N, "pred_pg": -sum_a / N, "pred_adv": tw_num / tw_den if tw_den else None,
        "reward": mean(r["reward"] for r in rows), "masked": len(masked) / N,
        "masked_tok": sum(min(r["response_tokens"] or 0, cap) for r in masked) / max(sum(resp_cap), 1),
        "n_masked": len(masked), "resp": mean(resp_cap),
        "tokens": sum((r["prompt_tokens"] or 0) + (r["response_tokens"] or 0) for r in rows),
        "valid_tok": valid_tok, "mean_valid_len": valid_tok / n_valid if n_valid else None,
        "prompt_max": max((r["prompt_tokens"] or 0) for r in rows),
        "tool_err": mean((r["tool_errors"] or 0) for r in rows) if has("tool_errors") else None,
        "tool_gt2": mean(((r["tool_errors"] or 0) > 2) for r in rows) if has("tool_errors") else None,
        "passes": len(passes),
        "unver": sum((r["verification_rejects"] or 0) >= 2 for r in passes) if has("verification_rejects") else None,
        "ends": Counter(r["end"] for r in rows if r["end"] is not None),
        "task_steps": mean(r["task_steps"] for r in rows if fin(r["task_steps"])) if has("task_steps") else None,
        "model_tok_median": median((r["model_tokens"] or 0) for r in rows) if has("model_tokens") else None,
        "sec_median": median(secs), "sec_max": max(secs) if secs else None,
        "task_pass": {t: sum(v) / len(v) for t, v in task_pass.items()}, "task_tmpl": task_tmpl,
        "tmpl": {t: dict(c) for t, c in tmpl.items()},
    }


class Step:
    def __init__(self, seg, run, step, m, ep):
        self.seg, self.run, self.step, self.m, self.ep = seg, run, step, m, ep
        self.label = f"{run}:{step}"
        self.g = None
        self.pairs = []  # same-task pass-rate differences vs the task's previous occurrence in the chain


class Segment:
    def __init__(self, idx, run_dir, label):
        self.idx, self.dir, self.label = idx, run_dir, label
        self.metrics, self.minfo = load_metrics(os.path.join(run_dir, "metrics.jsonl"))
        self.train_steps = sorted(s for s, d in self.metrics.items() if is_train(d))
        self.n = self.p = self.W = None
        self.ep_steps, self.dev_rounds, self.ep_note = {}, {}, ""
        self.ep_info = {"bad": 0, "partial": 0, "misaligned_dev": 0, "rest_train": 0, "incomplete_rows": 0}
        self.cutoff = None


def load_episodes(seg, args, cutoff, budget, is_main):
    files = sorted(glob.glob(os.path.join(seg.dir, "episodes", "*.jsonl")))
    if not files or args.no_episodes:
        seg.ep_note = "no episodes"
        return
    readers = [EpisodeReader(p, i, args.dev_marker) for i, p in enumerate(files)]
    try:
        seg.W = len(readers)
        s_first = seg.train_steps[0] if seg.train_steps else 1
        p = (args.rows_per_worker if is_main else args.prev_rows_per_worker) or \
            detect_p(readers, args.n, seg.metrics.get(s_first))
        if not p:
            seg.ep_note = "episodes not aligned to steps (pass --rows-per-worker)"
            return
        seg.p = p
        readers[0].read_train(p)
        seg.n = args.n or detect_n(readers[0].train[:p])
        if is_main and cutoff is not None:
            limit = p * (cutoff - s_first + 1)
            for r in readers:
                r.read_train(limit)
                r.read_dev_tail()
                seg.ep_info["rest_train"] += r.count_rest()
        else:
            for r in readers:
                r.read_train(10 ** 12)
        for r in readers:
            seg.ep_info["bad"] += r.bad
            seg.ep_info["partial"] += r.partial
        full = min(len(r.train) for r in readers) // p
        if is_main and cutoff is not None:
            full = min(full, cutoff - s_first + 1)
        seg.ep_info["incomplete_rows"] = sum(len(r.train) - full * p for r in readers)
        for i in range(full):
            rows = [x for r in readers for x in r.train[i * p:(i + 1) * p]]
            seg.ep_steps[s_first + i] = agg_step(rows, budget)
        for r in readers:
            for t_before, rows in r.clusters:
                if t_before % p:
                    seg.ep_info["misaligned_dev"] += 1
                step = s_first - 1 + t_before // p
                if is_main and cutoff is not None and step > cutoff:
                    continue
                d = seg.dev_rounds.setdefault(step, {"rows": []})
                d["rows"].extend({k: x[k] for k in ("task_id", "template", "reward", "end", "verification_rejects")}
                                 for x in rows)
            r.train = []
    finally:
        for r in readers:
            r.close()


def parse_log(path):
    out = {"counts": {}, "config": {}}
    with open(path, "rb") as f:
        data = f.read()
    pats = {
        "errors": rb"Traceback|out of memory|CUDA error|NCCL (?:error|timeout)|ActorDiedError|RayActorError|"
                  rb"RaySystemError|OutOfDiskError|preempted|\bnan\b|failure: [1-9]",
        "prompt_trunc": rb"exceeds rollout\.prompt_length",
        "unknown_tool": rb"is not defined in the tools list",
        "qwen35_patch": rb"Monkey patch Qwen3_5\w* model forward",
        "qwen35": rb"Qwen3_5",
        "progress": rb"failure: \d+",
    }
    for k, p in pats.items():
        out["counts"][k] = len(re.findall(p, data))
    m = re.findall(rb"(Traceback|out of memory|CUDA error|NCCL (?:error|timeout)|ActorDiedError|RayActorError|"
                   rb"RaySystemError|OutOfDiskError|preempted|\bnan\b|failure: [1-9])", data)
    out["error_kinds"] = dict(Counter(x.decode() for x in m))
    head = data[:6_000_000]
    for key, rx, typ in (("max_prompt_length", rb"'max_prompt_length': (\d+)", int),
                         ("max_response_length", rb"'max_response_length': (\d+)", int),
                         ("grad_clip", rb"'grad_clip': ([\d.]+)", float),
                         ("total_training_steps", rb"'total_training_steps': (\d+)", int),
                         ("test_freq", rb"'test_freq': (-?\d+)", int),
                         ("save_freq", rb"'save_freq': (-?\d+)", int),
                         ("trainer_mode", rb"'trainer_mode': '(\w+)'", str),
                         ("adv_estimator", rb"'adv_estimator': '(\w+)'", str),
                         ("rollout_is", rb"'rollout_is': '?(\w+)'?", str),
                         ("n_gpus_per_node", rb"'n_gpus_per_node': (\d+)", int)):
        mm = re.search(rx, head)
        if mm:
            v = mm.group(1).decode()
            out["config"][key] = typ(v) if typ is not str else v
    return out


# --------------------------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, args):
        self.args = args
        self.log = parse_log(args.log) if args.log else None
        cfg = self.log["config"] if self.log else {}
        segs = []
        if args.prev:
            segs.append(Segment(0, args.prev, args.prev_label or os.path.basename(os.path.normpath(args.prev))))
        segs.append(Segment(len(segs), args.run_dir, args.label or os.path.basename(os.path.normpath(args.run_dir))))
        self.segs, self.main = segs, segs[-1]
        if not self.main.train_steps:
            raise SystemExit("no training step in " + os.path.join(args.run_dir, "metrics.jsonl"))
        cut = args.cutoff if args.cutoff is not None else self.main.train_steps[-1]
        self.main.cutoff = cut
        all_max = [d.get("response_length/max") for s in segs for d in s.metrics.values()]
        self.budget = args.budget or cfg.get("max_response_length") or \
            (int(max(v for v in all_max if fin(v))) if any(fin(v) for v in all_max) else None)
        self.max_prompt = args.max_prompt_length or cfg.get("max_prompt_length")
        self.grad_clip = args.grad_clip or cfg.get("grad_clip") or 1.0
        self.total_steps = args.total_steps or cfg.get("total_training_steps")
        self.test_freq = args.test_freq or cfg.get("test_freq")
        self.trainer_mode = cfg.get("trainer_mode")
        self.rollout_is = cfg.get("rollout_is")
        for s in segs:
            load_episodes(s, args, s.cutoff, self.budget, s is self.main)
        self.n = args.n or next((s.n for s in reversed(segs) if s.n), None)
        # chain of steps
        self.steps = []
        for s in segs:
            keys = set(st for st in s.train_steps if s is not self.main or st <= cut) | set(s.ep_steps)
            for st in sorted(keys):
                if s is self.main and st > cut:
                    continue
                m = s.metrics.get(st)
                self.steps.append(Step(s.idx, s.label, st, m if m is not None and is_train(m) else None,
                                       s.ep_steps.get(st)))
        for i, st in enumerate(self.steps):
            st.g = i + 1
        self.msteps = [s for s in self.steps if s.m is not None]
        self.esteps = [s for s in self.steps if s.ep is not None]
        self.L = [s for s in self.msteps if s.seg == self.main.idx][-1]
        self.is_ppo = any(("critic/vf_loss" in s.m or "critic/values/mean" in s.m) for s in self.msteps)
        self.estimator = args.adv_estimator or cfg.get("adv_estimator") or ("gae" if self.is_ppo else "grpo")
        # single update per step: given, or ppo_kl and clip fractions exactly 0 at the first 3 steps of a run
        self.single = {}
        for s in segs:
            if args.single_update != "auto":
                self.single[s.idx] = args.single_update == "yes"
                continue
            first = [x.m for x in self.msteps if x.seg == s.idx and "actor/ppo_kl" in x.m][:3]
            self.single[s.idx] = bool(first) and all(
                d.get("actor/ppo_kl") == 0 and d.get("actor/pg_clipfrac", 0) == 0 and
                d.get("actor/pg_clipfrac_lower", 0) == 0 for d in first)
        # same-task pairs along the chain
        last = {}
        for st in self.esteps:
            for t, p in st.ep["task_pass"].items():
                if t in last:
                    st.pairs.append((t, st.ep["task_tmpl"].get(t), p - last[t]))
                last[t] = p
        # validation rounds along the chain
        self.dev = []
        for s in segs:
            for step in sorted(s.dev_rounds):
                rows = s.dev_rounds[step]["rows"]
                res = defaultdict(list)
                for r in rows:
                    res[r["task_id"]].append(r["reward"])
                self.dev.append({"seg": s.idx, "step": step, "label": f"{s.label}:{step}", "n": len(rows),
                                 "res": {t: sum(v) / len(v) for t, v in res.items()},
                                 "tmpl": {r["task_id"]: r["template"] for r in rows},
                                 "passes": sum(r["reward"] > 0.5 for r in rows),
                                 "unver": sum(r["reward"] > 0.5 and (r["verification_rejects"] or 0) >= 2
                                              for r in rows),
                                 "overlong": sum(r["end"] == "overlong" for r in rows)})
        nmax = max((d["n"] for d in self.dev), default=0)
        self.dev = [d for d in self.dev if d["n"] == nmax]
        # validation metric rows (val-core/*), including validation-only rows such as step 0
        self.val = []
        for s in segs:
            for st in sorted(s.metrics):
                if s is self.main and st > cut:
                    continue
                vk = {k: v for k, v in s.metrics[st].items() if k.startswith("val-core/") and "mean@" in k}
                if vk:
                    self.val.append((f"{s.label}:{st}", vk))

    def seg_msteps(self, seg):
        return [s for s in self.msteps if s.seg == seg]

    def series(self, key, steps=None):
        return [(s, s.m[key]) for s in (steps if steps is not None else self.msteps) if key in s.m]

    def N(self, seg_idx):
        s = self.segs[seg_idx]
        return s.p * s.W if s.p and s.W else None


def R(status, value="-", thr="-", step="-", note="", num=None):
    return {"status": status, "value": value, "threshold": thr, "step": step, "note": note, "num": num}


def history(evals):
    """evals: [(Step, (status, value, note, num))]; latest status, earlier alerts downgraded to watch."""
    evals = [(s, e) for s, e in evals if e is not None]
    if not evals:
        return None
    s_l, (st, val, note, num) = evals[-1]
    earlier = [s.label for s, e in evals[:-1] if e[0] == ALERT]
    if earlier:
        more = f" (+{len(earlier) - 4} more)" if len(earlier) > 4 else ""
        note = (note + "; " if note else "") + "earlier alert at " + ",".join(earlier[-4:]) + more
        if st == OK:
            st = WATCH
    return st, val, note, num, s_l


def per_step(c, steps, fn, thr):
    h = history([(s, fn(s)) for s in steps])
    if h is None:
        return None
    st, val, note, num, s_l = h
    if s_l.g < c.L.g and s_l.m is not None:
        note = (note + "; " if note else "") + f"latest evaluable step {s_l.label}"
    return R(st, val, thr, s_l.label, note, num)


RULES = []


def rule(rid, area, title, keys, rng, src, pitfall, fix):
    def deco(fn):
        RULES.append({"id": rid, "area": area, "title": title, "keys": keys, "healthy_range": rng, "source": src,
                      "pitfall": pitfall, "fix": fix, "fn": fn})
        return fn
    return deco


# ---------------------------------------------------------------------------------------------------- optimization rules
@rule("OPT-01", "optimization", "policy ratio and clip fractions",
      ["actor/ppo_kl", "actor/pg_clipfrac", "actor/pg_clipfrac_lower"],
      "one update per step: |ppo_kl|<=1e-6 and both clip fractions 0; several mini-batches: ppo_kl<=0.01, "
      "pg_clipfrac<=0.10",
      "verl 0.10 compute_policy_loss_vanilla (ratio vs old_log_prob recomputed by the same FSDP actor); research log "
      "11-01 20:1x and 21:26 smoke test (ppo_kl 0, clip 0); OpenAI Spinning Up PPO target_kl 0.01-0.05",
      "Log 11-01 20:1x rule 'turn prefix caching off if ppo_kl != 0' can never fire (ppo_kl compares FSDP with FSDP); "
      "APO table 'entropy falls fast -> DAPO Clip-Higher': the 0.2/0.28 clip acts only with >=2 mini-batches",
      "Nonzero with one update per step: old_log_prob no longer comes from the current weights (check the recompute, "
      "micro-batching, determinism). Detect vLLM/FSDP mismatch with CON-01..05, not ppo_kl.")
def r_ratio(c):
    steps = [s for s in c.msteps if "actor/ppo_kl" in s.m]
    if not steps:
        return None

    def f(s):
        k, cf, cl = s.m.get("actor/ppo_kl"), s.m.get("actor/pg_clipfrac", 0.0), s.m.get("actor/pg_clipfrac_lower", 0.0)
        v = f"ppo_kl {fmt(k)}, clip {fmt(cf)}/{fmt(cl)}"
        if not all(fin(x) for x in (k, cf, cl)):
            return ALERT, v, "non-finite", k
        if c.single[s.seg]:
            return (ALERT if abs(k) > 1e-6 or cf > 0 or cl > 0 else OK), v, "one update per step", k
        return worst(hi(k, 0.01, 0.05), hi(cf, 0.10, 0.30)), v, "several updates per step", k
    thr = ("alert |ppo_kl|>1e-6 or clipfrac>0" if c.single[c.main.idx]
           else "watch ppo_kl>0.01|clip>0.10; alert >0.05|>0.30")
    return per_step(c, steps, f, thr)


@rule("OPT-02", "optimization", "pg_loss vs advantage identity (masking bias)", ["actor/pg_loss"],
      "ratio 1 with overlong masking: -0.20<=pg_loss<=0 and within 0.005 of -sum(unmasked A)/N",
      "derived from verl GRPO advantage (unbiased std, eps 1e-6) and seq-mean-token-mean agg_loss dividing by the "
      "global batch N incl. masked rows (v1 trainer_base _update_actor); log 11-01 21:5x masking decision; -0.20 "
      "judgment",
      "APO [R2] collapse root causes row 2 (token-mean: pg_loss stays positive, entropy explodes); APO Agentic RL #2 / "
      "DeepSWE: do not score truncated trajectories plain 0; Cui et al. 2025 (arXiv 2505.22617): entropy declines "
      "when advantage covaries with log-prob",
      "Keep the masking; if entropy falls >15% (OPT-08) or the overlong share passes 20% (ROL-02): compute group "
      "mean/std over unmasked episodes only (judgment) or raise the response budget (log 21:5x option C).")
def r_pg(c):
    steps = [s for s in c.msteps if "actor/pg_loss" in s.m]
    if not steps:
        return None
    grpo = not c.is_ppo and c.estimator == "grpo"
    ent = dict((s.g, s.m.get("actor/entropy")) for s in steps)

    def f(s):
        pg = s.m["actor/pg_loss"]
        if not fin(pg):
            return ALERT, fmt(pg), "non-finite", pg
        if grpo and c.single[s.seg]:
            st = OK
            note = "one update per step"
            v = f"pg_loss {fmt(pg)}"
            if pg > 0.01:
                st = ALERT
            elif pg < -0.20:
                st = WATCH
            if s.ep is not None:
                d = abs(pg - s.ep["pred_pg"])
                v += f", pred {fmt(s.ep['pred_pg'])}, masked {pct(s.ep['masked'])}"
                if d >= 0.005:
                    st = ALERT
                    note += f"; |pg-pred| {fmt(d)}"
                elif s.ep["masked"] > 0.15:
                    st = worst(st, WATCH)
            return st, v, note, pg
        # several updates or PPO: positive pg_loss for 3 steps while entropy rises (token-mean blow-up signature)
        i = steps.index(s)
        last3 = steps[max(0, i - 2):i + 1]
        st = OK
        if len(last3) == 3 and all(x.m["actor/pg_loss"] > 0 for x in last3) and \
                all(fin(ent.get(x.g)) for x in last3) and ent[last3[-1].g] > ent[last3[0].g]:
            st = ALERT
        return st, f"pg_loss {fmt(pg)}", "several updates or PPO", pg
    thr = ("alert |pg-pred|>=0.005 or pg>0.01; watch pg<-0.20 or masked>0.15" if grpo and c.single[c.main.idx]
           else "alert pg_loss>0 on 3 steps while entropy rises")
    return per_step(c, steps, f, thr)


@rule("OPT-03", "optimization", "groups with two-sided advantage", ["episodes"],
      ">=0.55 of groups hold both a positive and a negative unmasked advantage",
      "research log 11-02 23:0x (groups whose failures are all masked carry only positive advantage: 14.5%; "
      "effective policy-gradient scale 0.53-0.70); threshold judgment",
      "Overlong masking keeps masked failures in the group baseline, so a group whose failures are all masked pushes "
      "only toward its sampled responses (OPT-02)",
      "grpo2 patch 3 (log 22:3x: drop 0/k tasks, keep 10% of k/k tasks, drop zero-std groups; DAPO dynamic "
      "sampling); compute group statistics over unmasked episodes.")
def r_twosided(c):
    if not c.esteps or c.is_ppo:
        return None
    return per_step(c, c.esteps, lambda s: (lo(s.ep["two_sided"], 0.55, -1),
                                            f"two-sided {pct(s.ep['two_sided'])}, positive-only "
                                            f"{pct(s.ep['pos_only'])}", "", s.ep["two_sided"]),
                    "watch two-sided share <0.55")


@rule("OPT-04", "optimization", "training reward trend", ["critic/score/mean"],
      "rolling 4-step mean within 0.05 of the chain's first 4 steps (tasks differ per step: +-0.03 noise)",
      "rollout analysis R3 (a 0.05 score fall marks premature submission); APO 'quality falls late in training -> "
      "early stop, monitor KL'; 0.15 judgment",
      "APO 'training quality falls in late stage -> early stop'",
      "Check grader/reward changes and KL growth (OPT-10); a collapse calls for stopping and rolling back to the last "
      "good checkpoint.")
def r_score(c):
    ser = c.series("critic/score/mean")
    if len(ser) < 2:
        return None
    base = mean(v for _, v in ser[:4])
    w = c.args.window

    def f(s):
        i = [x for x, _ in ser].index(s)
        if i < 4:
            return None
        roll = mean(v for _, v in ser[max(0, i - w + 1):i + 1])
        if not fin(roll):
            return ALERT, fmt(roll), "non-finite", roll
        drop = base - roll
        return worst(hi(drop, 0.05 - 1e-9, 0.15 - 1e-9)), f"rolling {fmt(roll)} vs base {fmt(base)}", \
            f"latest {fmt(ser[i][1])}", roll
    r = per_step(c, [s for s, _ in ser], f, "watch drop>=0.05; alert drop>=0.15")
    return r or R(OK, f"{fmt(ser[-1][1])}", "needs >4 steps", ser[-1][0].label)


@rule("OPT-05", "optimization", "same-task pass-rate gain", ["episodes"],
      "paired gain on repeated tasks >0, CI excluding 0 by the end of the run",
      "APO 现象排查 'KL very low, reward not rising -> constraint too strong or lr too low'; APO Agentic RL #4 "
      "(saturation vs capability ceiling); bootstrap 10,000 resamples seed 20261025; thresholds judgment",
      "APO 'KL 很低，reward 不涨'; the KL term is not binding here (0.001 x kl_loss <= 1e-6 vs |pg_loss| ~0.1)",
      "Larger update per step: 2-4 mini-batches (also activates the 0.2/0.28 clip; DAPO makes 16 updates per rollout "
      "step) or lr toward 1e-6..3e-6 with warmup (APO default table); grpo2 patch 3. Decide after the run ends.")
def r_repeat(c):
    if not c.esteps:
        return None
    w = c.args.window
    win = c.esteps[-w:]
    pairs = [d for s in win for _, _, d in s.pairs]
    if len(pairs) < 20:
        return R(OK, f"{len(pairs)} repeated tasks", "needs >=20 repeated tasks in the window",
                 f"{win[0].label}..{win[-1].label}")
    m, l, h = boot_ci(pairs, c.args.boot, c.args.seed)
    updates = len([s for s in c.steps if s.g <= c.L.g])
    st = ALERT if h < 0 else (WATCH if updates >= 12 and l <= 0 else OK)
    prev = c.esteps[-2 * w:-w]
    pp = [d for s in prev for _, _, d in s.pairs]
    note = f"{len(pairs)} tasks, {updates} updates"
    if len(pp) >= 20:
        se = lambda xs: statistics.stdev(xs) / math.sqrt(len(xs))
        if mean(pp) < -se(pp) and m < -se(pairs):
            st = worst(st, WATCH)
            note += "; below 0 by >1 SE in 2 windows"
    return R(st, f"{m:+.4f} [{l:+.4f}, {h:+.4f}]", "alert CI upper<0; watch CI lower<=0 after >=12 updates",
             f"{win[0].label}..{win[-1].label}", note, m)


def dev_paired(c, d, base):
    common = sorted(set(d["res"]) & set(base["res"]))
    if not common:
        return None
    return boot_ci([d["res"][t] - base["res"][t] for t in common], c.args.boot, c.args.seed), len(common)


@rule("OPT-06", "optimization", "reward-hacking gap (train gain - dev gain)", ["episodes"],
      "paired train gain at most 0.05 above paired dev gain",
      "optimization analysis OD-3; grader audit (log 11-01 19:3x-20:0x: 22.3% of blind adversarial answers pass on "
      "train tasks); threshold judgment",
      "Grader audit: the grader accepts a matching value anywhere in the answer",
      "Regrade stored answers with the strict grader (log 22:3x patch 1); raise kl_loss_coef (APO table).")
def r_gap(c):
    if len(c.dev) < 2 or not c.esteps:
        return None
    base = c.dev[0]
    w = c.args.window
    gaps = []
    for d in c.dev[1:]:
        st = next((s for s in c.esteps if s.seg == d["seg"] and s.step == d["step"]), None)
        if st is None:
            continue
        win = [s for s in c.esteps if st.g - w < s.g <= st.g]
        pairs = [x for s in win for _, _, x in s.pairs]
        dp = dev_paired(c, d, base)
        if len(pairs) < 20 or dp is None:
            continue
        gaps.append((d["label"], mean(pairs) - dp[0][0]))
    if not gaps:
        return R(OK, "no round with >=20 repeated tasks", "watch gap>0.05; alert on 2 rounds", c.dev[-1]["label"])
    lab, g = gaps[-1]
    st = ALERT if len(gaps) >= 2 and gaps[-2][1] > 0.05 and g > 0.05 else (WATCH if g > 0.05 else OK)
    return R(st, f"{g:+.4f}", "watch gap>0.05; alert >0.05 at 2 consecutive rounds", lab,
             "rounds: " + ", ".join(f"{a} {b:+.3f}" for a, b in gaps[-3:]), g)


@rule("OPT-07", "optimization", "dev reward vs the first validation round", ["val-core/*", "episodes (dev rows)"],
      "paired dev diff vs the chain's first round: CI upper>=0 and point>-0.04; val-core not falling twice",
      "research log 11-02 00:1x drop rule (paired-bootstrap CI entirely below 0); 0.04/0.08 judgment (~1.3/2.7 SE "
      "for 200 tasks x 1 sample)",
      "results.md Appendix §5 'each test task is run only once' (single-sample noise); APO 'quality falls late -> "
      "early stop'",
      "Confirm with the chain evaluation (dev x4, eval settings); before evaluating a merged checkpoint on another "
      "vLLM/transformers, check rope_theta (memory note vllm-rope-theta-pitfall).")
def r_dev(c):
    vals = []
    if c.val:
        key = sorted(c.val[-1][1])[0]
        vals = [(lab, d[key]) for lab, d in c.val if key in d]
    fell2 = len(vals) >= 3 and vals[-1][1] < vals[-2][1] < vals[-3][1]
    trend = " -> ".join(f"{fmt(v, 3)}" for _, v in vals[-4:])
    if len(c.dev) >= 2:
        base, d = c.dev[0], c.dev[-1]
        (m, l, h), k = dev_paired(c, d, base)
        st = ALERT if h < 0 or m <= -0.08 else (WATCH if m <= -0.04 or fell2 else OK)
        return R(st, f"{m:+.4f} [{l:+.4f}, {h:+.4f}] vs {base['label']}",
                 "alert CI upper<0 or point<=-0.08; watch point<=-0.04 or 2 falls", d["label"],
                 f"{k} tasks; val-core {trend}", m)
    if len(vals) >= 2:
        drop = vals[0][1] - vals[-1][1]
        st = ALERT if drop >= 0.08 else (WATCH if drop >= 0.04 or fell2 else OK)
        return R(st, f"val-core {trend}", "no dev rows: alert drop>=0.08; watch >=0.04 or 2 falls", vals[-1][0],
                 "", vals[-1][1])
    return None


@rule("OPT-08", "optimization", "policy entropy", ["actor/entropy"],
      "0.85-1.25x the chain's first-3-step mean",
      "judgment anchored on DAPO (arXiv 2503.14476, entropy collapse without Clip-Higher) and the APO rows 熵快速下降 "
      "/ 熵增崩溃 (entropy falls fast / rise-collapse)",
      "APO 'entropy falls fast, outputs homogenize'; APO [R2] row 2 token-mean entropy explosion (prevented by "
      "seq-mean-token-mean, log 20:1x)",
      "Falling: entropy bonus 0.001 decaying to 0.0001 (APO), or Clip-Higher with >=2 mini-batches. Rising: check "
      "train-inference mismatch (CON rules), fp32 attention, loss_agg_mode.")
def r_entropy(c):
    ser = c.series("actor/entropy")
    if not ser:
        return None
    base = mean(v for _, v in ser[:3])
    vals = [v for _, v in ser]

    def f(s):
        i = [x for x, _ in ser].index(s)
        v = vals[i]
        if not fin(v) or not fin(base) or base <= 0:
            return ALERT, fmt(v), "non-finite", v
        ratio = v / base
        st = ALERT if ratio < 0.70 or ratio > 1.50 else WATCH if ratio < 0.85 or ratio > 1.25 else OK
        if i >= 3 and vals[i] < vals[i - 1] < vals[i - 2] < vals[i - 3] and vals[i] / vals[i - 3] < 0.90:
            st = worst(st, WATCH)
        return st, f"{fmt(v)} ({ratio:.3f}x base {fmt(base)})", "", ratio
    return per_step(c, [s for s, _ in ser], f, "watch <0.85x|>1.25x|3 falls >10%; alert <0.70x|>1.50x")


def spike_rule(c, key, clip, steps=None):
    ser = c.series(key, steps)
    if not ser:
        return None
    hist = []

    def f(s):
        v = s.m[key]
        med = median(hist[-10:]) if hist else None
        if fin(v):
            hist.append(v)
        if not fin(v):
            return ALERT, fmt(v), "non-finite", v
        r = v / med if med else None
        st = ALERT if (clip and v >= clip) or (r and r > 3) else WATCH if r and r > 2 else OK
        return st, fmt(v) + (f" ({r:.2f}x median)" if r else ""), "", v
    return per_step(c, [s for s, _ in ser], f,
                    (f"alert non-finite, >=clip {fmt(clip)} or >3x median of previous 10; watch >2x" if clip else
                     "alert non-finite or >3x median of previous 10; watch >2x"))


@rule("OPT-09", "optimization", "actor gradient norm", ["actor/grad_norm"],
      "finite, below grad_clip, below 2x the median of the previous 10 steps",
      "verl grad_clip (1.0 here); APO rows 梯度范数爆炸 (gradient norm explodes) and async-RL gradient spikes; "
      "multipliers judgment",
      "APO 'gradient norm explodes -> lower lr, clip, standardize reward'; APO async-RL spikes from ~10-step staleness",
      "Lower lr, keep clipping, standardize reward (APO); check staleness (CON-08) and mismatch (CON-01).")
def r_grad(c):
    return spike_rule(c, "actor/grad_norm", c.grad_clip)


@rule("OPT-10", "optimization", "KL to the reference policy", ["actor/kl_loss", "actor/kl_coef", "actor/loss"],
      "low_var_kl kl_loss <0.01 and step ratio <1.5, within one reference",
      "judgment with the APO default table (kl_loss ~0.001 from SFT) and the APO row 'reward rising, KL exploding'",
      "APO 'reward 涨，KL 暴涨 -> fooling the RM -> raise the KL coefficient'; grader audit (grader exploitable)",
      "If the ratio passes 1.5 while train reward rises and dev does not: regrade with the strict grader and raise "
      "kl_loss_coef. A continuation run resets the reference, so compare kl_loss within one run only.")
def r_kl(c):
    ser = c.series("actor/kl_loss")
    if not ser:
        return None
    prev = {}
    ratios = defaultdict(list)

    def f(s):
        v = s.m["actor/kl_loss"]
        if not fin(v):
            return ALERT, fmt(v), "non-finite", v
        p = prev.get(s.seg)
        r = v / p if p and p >= 1e-4 else None
        prev[s.seg] = v
        rs = ratios[s.seg]
        rs.append(r)
        st = ALERT if v > 0.05 or (r and r > 2.0) else OK
        if st == OK and (v > 0.01 or (r and r > 1.5) or
                         (len(rs) >= 3 and all(x and x > 1.25 for x in rs[-3:]))):
            st = WATCH
        note = ""
        lo_, pg, kc = s.m.get("actor/loss"), s.m.get("actor/pg_loss"), s.m.get("actor/kl_coef")
        if all(fin(x) for x in (lo_, pg, kc)):
            resid = abs(lo_ - pg - kc * v)
            if resid > 1e-6 + 1e-6 * abs(pg):
                st = worst(st, WATCH)
                note = f"loss-pg-coef*kl residual {fmt(resid)} (entropy bonus or aggregation changed?)"
        if c.total_steps and r and s.seg == c.main.idx:
            gm = [x for x in rs if x]
            if gm:
                g = math.exp(mean(math.log(x) for x in gm))
                proj = v * g ** max(0, c.total_steps - s.step)
                note = (note + "; " if note else "") + f"projected {fmt(proj)} at step {c.total_steps} (x{g:.2f}/step)"
        return st, fmt(v) + (f" (x{r:.2f})" if r else ""), note, v
    return per_step(c, [s for s, _ in ser], f,
                    "watch >0.01, ratio>1.5 or 3 ratios>1.25; alert >0.05 or ratio>2.0")


@rule("OPT-11", "optimization", "GRPO advantage bounds and zero-std groups",
      ["critic/advantages/*", "critic/returns/*", "episodes"],
      "|A|<=(n-1)/sqrt(n); returns==advantages; advantages/mean == token-weighted unmasked mean; zero-std groups "
      "<=0.40; gradient-carrying episodes >=0.50",
      "GRPO z-scores with verl's unbiased std (Samuelson bound); APO (LLMCat) 'frac_reward_zero_std >50% means a "
      "reward problem'; log 11-02 step-8 entry (72.7% mixed groups)",
      "APO 组内 reward 全同、梯度为 0 (all rewards in a group equal, zero gradient); APO Agentic RL #4",
      "grpo2 patch 3 (k-run task selection, DAPO dynamic sampling).")
def r_adv(c):
    if c.is_ppo or c.estimator != "grpo":
        return None
    steps = [s for s in c.steps if (s.m and "critic/advantages/max" in s.m) or s.ep]
    if not steps:
        return None
    n = c.n
    bound = (n - 1) / math.sqrt(n) if n else None
    zs = []

    def f(s):
        st, parts, note = OK, [], []
        if s.m and "critic/advantages/max" in s.m:
            mx, mn = s.m["critic/advantages/max"], s.m.get("critic/advantages/min")
            if not fin(mx) or not fin(mn):
                return ALERT, f"max {fmt(mx)} min {fmt(mn)}", "non-finite", mx
            parts.append(f"max {fmt(mx)} min {fmt(mn)}")
            if bound and (mx > bound + 1e-3 or mn < -bound - 1e-3):
                st = ALERT
                note.append(f"outside +-{bound:.4f}")
            for k in ("mean", "max", "min"):
                a, b = s.m.get(f"critic/advantages/{k}"), s.m.get(f"critic/returns/{k}")
                if fin(a) and fin(b) and abs(a - b) > 1e-6:
                    st = ALERT
                    note.append(f"returns/{k} != advantages/{k}")
            if s.ep and fin(s.ep["pred_adv"]) and fin(s.m.get("critic/advantages/mean")):
                d = abs(s.m["critic/advantages/mean"] - s.ep["pred_adv"])
                if d >= 1e-3:
                    st = ALERT
                    note.append(f"advantages/mean off episodes by {fmt(d)}")
        if s.ep:
            z = s.ep["zero_std"]
            zs.append(z)
            parts.append(f"zero-std {pct(z)}, carrying {pct(s.ep['carry'])}")
            if z > 0.50 and len(zs) >= 2 and zs[-2] > 0.50:
                st = ALERT
            elif z > 0.40 or s.ep["carry"] < 0.50:
                st = worst(st, WATCH)
        return st, ", ".join(parts), "; ".join(note), (s.ep or {}).get("zero_std")
    return per_step(c, steps, f, f"alert |A|>{fmt(bound)}, identity broken or zero-std>0.50 twice; "
                                 f"watch zero-std>0.40 or carrying<0.50")


@rule("OPT-12", "optimization", "learning rate", ["actor/lr", "critic/lr"],
      "equal to the configured value at every step when the schedule is constant",
      "research log 11-01 20:1x (constant 1e-6) and 11-02 08:2x (sqrt scaling to 8.7e-7 for batch 96)",
      "none",
      "Check the scheduler config and the resumed optimizer state.")
def r_lr(c):
    out = []
    for key in ("actor/lr", "critic/lr"):
        ser = c.series(key)
        if not ser:
            continue
        first = {}

        def f(s, key=key):
            v = s.m[key]
            segser = [x.m[key] for x in c.seg_msteps(s.seg) if key in x.m]
            if not fin(v) or v < 0:
                return ALERT, f"{v:.3e}" if fin(v) else str(v), "non-finite", v
            const = len(segser) >= 2 and segser[0] == segser[1]
            f0 = first.setdefault(s.seg, segser[0])
            if const and f0 and abs(v / f0 - 1) > 1e-6:
                return ALERT, f"{v:.3e} vs {f0:.3e}", "constant schedule changed", v
            return OK, f"{v:.3e}", "constant" if const else "scheduled", v
        out.append(per_step(c, [s for s, _ in ser], f, "alert |lr/lr0-1|>1e-6 (constant schedule)"))
    out = [o for o in out if o]
    if not out:
        return None
    r = max(out, key=lambda o: RANK[o["status"]])
    if len(out) > 1:
        r["value"] = " / ".join(o["value"] for o in out)
    return r


@rule("OPT-13", "optimization", "non-finite metric values", ["all keys"],
      "every logged value finite",
      "judgment",
      "none",
      "Inspect that step's loss, gradient and reward; resume from the last finite checkpoint.")
def r_nan(c):
    def f(s):
        bad = sorted(k for k, v in s.m.items() if isinstance(v, float) and not math.isfinite(v))
        if s.m.get("response/aborted_ratio") == 1.0:
            bad = [k for k in bad if not k.startswith("response_length_non_aborted/")]
        return (ALERT if bad else OK), (", ".join(bad[:4]) + (f" (+{len(bad) - 4})" if len(bad) > 4 else "")
                                        if bad else "all finite"), "", len(bad)
    return per_step(c, c.msteps, f, "alert any NaN/inf")


# ----------------------------------------------------------------------------------------------------- consistency rules
@rule("CON-01", "consistency", "vLLM vs FSDP log-prob mismatch (k3_kl)", ["rollout_corr/k3_kl", "rollout_corr/kl"],
      "k3_kl <=0.008 and not rising; |kl|<=0.1 (verl hard limit); step-1 kl <0.05",
      "verl rollout_corr docs health check (|kl|>0.1 warning, https://verl.readthedocs.io/en/latest/algo/"
      "rollout_corr.html); verl FP8 docs (BF16 k3 0.00127, FP8+KV 0.00534, all with token TIS); 0.008 judgment from "
      "run1+run1b",
      "results.md §3 'prefix cache on Qwen3.5 is not bitwise identical' and 'Qwen3.5 scores change with batch shape'; "
      "memory note vllm-rope-theta-pitfall (a wrong RoPE base would show at step 1)",
      "Rising: token TIS or two-sided IcePop/RS (CON-10); A/B the prefix cache and concurrency (CON-09).")
def r_k3(c):
    key = "rollout_corr/k3_kl" if c.series("rollout_corr/k3_kl") else "rollout_corr/kl"
    ser = c.series(key)
    if not ser:
        return None
    vals = [v for _, v in ser]
    base = median(vals[:4])
    firsts = {}
    for s in c.msteps:
        firsts.setdefault(s.seg, s)

    def f(s):
        i = [x for x, _ in ser].index(s)
        v = vals[i]
        if not fin(v):
            return ALERT, fmt(v), "non-finite", v
        st = hi(v, 0.008, 0.02)
        note = []
        if i >= 4 and base:
            tr = mean(vals[max(0, i - 3):i + 1]) / base
            st = worst(st, hi(tr, 1.25, 1.5))
            note.append(f"trend {tr:.2f}x")
        kl = s.m.get("rollout_corr/kl")
        if fin(kl) and abs(kl) > 0.1:
            st = ALERT
            note.append("|kl|>0.1")
        if firsts.get(s.seg) is s and fin(kl) and kl >= 0.05:
            st = ALERT
            note.append("step-1 kl>=0.05 (check rope_theta/weights)")
        return st, fmt(v), ", ".join(note), v
    return per_step(c, [s for s, _ in ser], f, "watch >0.008 or trend>1.25x; alert >0.02, trend>1.5x, |kl|>0.1")


@rule("CON-02", "consistency", "token importance-weight variance (chi2_token)", ["rollout_corr/chi2_token"],
      "chi2_token <=0.03 (verl: >1.0 severe shift); chi2/(2 kl) <=2.5",
      "verl rollout_corr docs (chi2_token >1.0 severe distribution shift); 0.03/0.1 judgment from run1+run1b",
      "none",
      "Tail widening: IcePop (rollout_is_threshold '0.5_5.0') or token TIS (threshold 2.0).")
def r_chi2(c):
    ser = c.series("rollout_corr/chi2_token")
    if not ser:
        return None

    def f(s):
        v, kl = s.m["rollout_corr/chi2_token"], s.m.get("rollout_corr/kl")
        st = hi(v, 0.03, 0.1)
        note = ""
        if fin(v) and fin(kl) and kl > 0:
            ratio = v / (2 * kl)
            note = f"chi2/(2kl) {ratio:.2f}"
            if ratio > 2.5:
                st = worst(st, WATCH)
        return st, fmt(v), note, v
    return per_step(c, [s for s, _ in ser], f, "watch >0.03 or chi2/(2kl)>2.5; alert >0.1")


@rule("CON-03", "consistency", "token alignment of rollout and training log-probs",
      ["rollout_corr/k3_kl", "rollout_corr/kl", "training/rollout_actor_probs_pearson_corr",
       "training/rollout_logprobs_diff_mean"],
      "|E[rho]-1| = |k3_kl-kl| <=3e-4; Pearson >=0.993; mean |dlogp| <=0.045",
      "IS unbiasedness E_q[p/q]=1; arXiv 2506.13585 (Pearson as mismatch diagnostic, cited by verl debug metrics); "
      "thresholds judgment from run1+run1b and the smoke test",
      "results.md §4 'harness context management rewrote earlier tool results' (fixed: token-in/token-out loop)",
      "Misalignment: check token ids, rewritten context, off-by-one in the agent loop.")
def r_align(c):
    steps = [s for s in c.msteps if any(k in s.m for k in ("rollout_corr/k3_kl",
                                                          "training/rollout_actor_probs_pearson_corr"))]
    if not steps:
        return None

    def f(s):
        st, parts = OK, []
        k3, kl = s.m.get("rollout_corr/k3_kl"), s.m.get("rollout_corr/kl")
        if k3 is not None and kl is not None:
            e = (k3 - kl) if fin(k3) and fin(kl) else float("nan")
            st = worst(st, hi(abs(e), 3e-4, 1e-3))
            parts.append(f"E[rho]-1 {fmt(e)}")
        p = s.m.get("training/rollout_actor_probs_pearson_corr")
        if p is not None:
            st = worst(st, lo(p, 0.993, 0.98))
            parts.append(f"pearson {fmt(p)}")
        d = s.m.get("training/rollout_logprobs_diff_mean")
        if d is not None:
            st = worst(st, hi(d, 0.045, 0.08))
            parts.append(f"|dlogp| {fmt(d)}")
        return st, ", ".join(parts), "", p
    return per_step(c, steps, f, "alert |E[rho]-1|>1e-3, pearson<0.98, |dlogp|>0.08; watch >3e-4, <0.993, >0.045")


@rule("CON-04", "consistency", "worst single-token log-prob gap", ["training/rollout_logprobs_diff_max"],
      "<=16 nats (verl clamps log-ratios at +-20); probs_diff_max near 1 is normal in bf16",
      "Yao et al. 2025 'Your efficient RL framework secretly brings you off-policy RL training'; Li et al. 2025 "
      "arXiv 2512.23087 (tail tokens); 16 judgment",
      "none",
      "Two-sided correction (IcePop '0.5_5.0' or RS with a lower bound); token TIS caps only rho>2.")
def r_tail(c):
    ser = c.series("training/rollout_logprobs_diff_max")
    if not ser:
        return None
    return per_step(c, [s for s, _ in ser],
                    lambda s: (hi(s.m["training/rollout_logprobs_diff_max"], 16, 20),
                               fmt(s.m["training/rollout_logprobs_diff_max"]), "",
                               s.m["training/rollout_logprobs_diff_max"]), "watch >16; alert >20")


@rule("CON-05", "consistency", "sequence-level mismatch", ["rollout_corr/log_ppl_diff_max", "rollout_corr/log_ppl_diff_min",
                                                           "rollout_corr/log_ppl_diff"],
      "log_ppl_diff_max <=0.04, min >=-0.02, masked-row-corrected log_ppl_diff <=0.008",
      "verl rollout_corr docs (ppl_ratio 1 is ideal); results.md §2 packing-leak scale; thresholds judgment",
      "results.md §2 'pass sample boundaries to linear-attention and conv layers when packing Qwen3.5' (a leak raises "
      "log_ppl_diff_max); all-masked overlong rows count as log_ppl_diff 0 (dilution)",
      "Check packing boundaries (cu_seqlens patch, CON-11) and prefix-cache state.")
def r_seq(c):
    steps = [s for s in c.msteps if "rollout_corr/log_ppl_diff_max" in s.m]
    if not steps:
        return None

    def f(s):
        mx, mn, avg = (s.m.get(k) for k in ("rollout_corr/log_ppl_diff_max", "rollout_corr/log_ppl_diff_min",
                                            "rollout_corr/log_ppl_diff"))
        st = worst(hi(mx, 0.04, 0.08), lo(mn, -0.02, -0.05) if mn is not None else OK)
        parts = [f"max {fmt(mx)}", f"min {fmt(mn)}"]
        if s.ep and fin(avg) and s.ep["N"] > s.ep["n_masked"]:
            corr = avg * s.ep["N"] / (s.ep["N"] - s.ep["n_masked"])
            st = worst(st, hi(corr, 0.008, 0.02))
            parts.append(f"corrected mean {fmt(corr)}")
        return st, ", ".join(parts), "", mx
    return per_step(c, steps, f, "watch max>0.04|min<-0.02|corr>0.008; alert >0.08|<-0.05|>0.02")


@rule("CON-06", "consistency", "sequence importance weights (chi2_seq, informational)", ["rollout_corr/chi2_seq"],
      "no threshold: product sequence IS is degenerate when mean loss tokens x kl >=5",
      "verl rollout_corr docs; Liu, Li et al. 2025 'When Speed Kills Stability' (sequence IS: zero bias, high "
      "variance)",
      "none",
      "Use geometric sequence RS (rollout_rs=seq_mean_k1/k3), never product sequence IS, on long multi-turn "
      "trajectories.")
def r_chi2seq(c):
    s = c.L
    if "rollout_corr/chi2_seq" not in s.m:
        return None
    v = s.m["rollout_corr/chi2_seq"]
    st, note = OK, "informational"
    if s.ep and fin(s.m.get("rollout_corr/kl")) and s.ep["mean_valid_len"]:
        prod = s.ep["mean_valid_len"] * s.m["rollout_corr/kl"]
        note = f"length x kl {prod:.1f}"
        if c.rollout_is == "sequence" and prod >= 5:
            st = WATCH
            note += " with rollout_is=sequence"
    return R(st, fmt(v), "watch only if rollout_is=sequence and length x kl>=5", s.label, note, v)


@rule("CON-07", "consistency", "bitwise mismatch fraction", ["training/rollout_logprobs_mismatch_count",
                                                            "training/rollout_probs_diff_valid"],
      "0.85-0.97 of loss tokens differ bitwise (non-deterministic bf16); rollout_probs_diff_valid == 1",
      "judgment from run1+run1b; deterministic inference gives ~0 (arXiv 2511.17826)",
      "none",
      "Near 0: rollout log-probs are copied or recomputed by the trainer (bypass mode), so mismatch metrics are "
      "meaningless; check the config.")
def r_bitwise(c):
    steps = [s for s in c.msteps if "training/rollout_logprobs_mismatch_count" in s.m or
             "training/rollout_probs_diff_valid" in s.m]
    if not steps:
        return None

    def f(s):
        st, parts = OK, []
        mc = s.m.get("training/rollout_logprobs_mismatch_count")
        if s.ep and fin(mc) and s.ep["valid_tok"]:
            frac = mc / s.ep["valid_tok"]
            st = ALERT if frac < 0.5 else (WATCH if not 0.85 <= frac <= 0.97 else OK)
            parts.append(f"fraction {frac:.4f}")
        pv = s.m.get("training/rollout_probs_diff_valid")
        if pv is not None:
            if pv != 1:
                st = ALERT
            parts.append(f"valid {fmt(pv)}")
        return st, ", ".join(parts), "", pv
    return per_step(c, steps, f, "watch outside 0.85-0.97; alert <0.5 or diff_valid!=1")


@rule("CON-08", "consistency", "policy-version staleness", ["training/off_policy/trajectory_staleness/max",
                                                           "training/off_policy/trajectory_spans/max"],
      "sync trainer: staleness 0 and spans 1 at every step",
      "verl v1 trainer_base _compute_metrics (staleness = global_steps-1 - max version; spans 1 = one weight version)",
      "APO async-RL gradient spikes / KL blow-up from ~10-step staleness",
      "Check the weight sync to vLLM (timing_s/update_weights); in async modes cover staleness with rollout "
      "correction.")
def r_stale(c):
    steps = [s for s in c.msteps if "training/off_policy/trajectory_staleness/max" in s.m]
    if not steps:
        return None
    k = c.args.max_staleness

    def f(s):
        a = s.m.get("training/off_policy/trajectory_staleness/max")
        b = s.m.get("training/off_policy/trajectory_spans/max", 1)
        st = ALERT if not (fin(a) and fin(b)) or a > k or b > k + 1 else OK
        return st, f"staleness {fmt(a)}, spans {fmt(b)}", "", a
    return per_step(c, steps, f, f"alert staleness>{k} or spans>{k + 1}")


@rule("CON-09", "consistency", "mismatch vs a same-weights reference run", ["rollout_corr/k3_kl"],
      "median k3_kl of the first 4 steps <=2x the reference (the smoke test's step-1 k3 on the same weights)",
      "consistency analysis C12 (smoke test 8 rollouts per server vs full run 128 per server, identical vLLM "
      "settings); thresholds judgment",
      "results.md §3 'fixed seed does not reproduce under concurrent load' and 'prefix cache stops working when the "
      "KV cache cannot hold the concurrent long contexts'",
      "Before the next run: one smoke step at full concurrency with the prefix cache on vs off and 48 vs 768 "
      "rollouts; keep prefix caching only if k3_kl changes by <=0.002.")
def r_ref(c):
    ref = c.args.ref_k3
    ser = c.series("rollout_corr/k3_kl")
    if not ser or not ref:
        return None
    first = median(v for _, v in ser[:4])
    last = median(v for _, v in ser[-4:])
    rf, rl = first / ref, last / ref
    st = worst(hi(rf, 2, 4), hi(rl, 2, 4))
    return R(st, f"first-4 {rf:.2f}x, last-4 {rl:.2f}x ref {fmt(ref)}", "watch >2x; alert >4x", c.L.label,
             "", rf)


@rule("CON-10", "consistency", "importance-sampling correction need / health",
      ["rollout_corr/k3_kl", "rollout_corr/chi2_token", "actor/entropy", "rollout_corr/rollout_is_*"],
      "no correction needed while k3_kl<=0.008 flat and chi2_token<=0.03; with IS on: IS mean 0.5-2.0, ESS>=0.3, "
      "IS std<=1.0",
      "verl rollout_corr docs (presets, health-check bands); verl FP8 docs (token TIS C=2); Yao et al. 2025 (TIS)",
      "none in the records (no IS decision recorded)",
      "algorithm.rollout_correction.rollout_is=token, rollout_is_threshold=2.0, or IcePop '0.5_5.0' for both tails; "
      "identical in every arm; a user decision before grpo2/ppo2.")
def r_is(c):
    ser = c.series("rollout_corr/k3_kl")
    if not ser:
        return None
    s = c.L
    d = s.m
    vals = [v for _, v in ser]
    trig = []
    if len(vals) >= 2 and all(fin(x) and x > 0.01 for x in vals[-2:]):
        trig.append("k3>0.01 on 2 steps")
    if len(vals) >= 5 and median(vals[:4]) and mean(vals[-4:]) / median(vals[:4]) > 1.5:
        trig.append("k3 trend>1.5x")
    if fin(d.get("rollout_corr/chi2_token")) and d["rollout_corr/chi2_token"] > 0.05:
        trig.append("chi2_token>0.05")
    ent = [v for _, v in c.series("actor/entropy")]
    if len(ent) >= 5 and fin(ent[-1]) and ent[-1] < 0.8 * median(ent[:4]) and vals[-1] > median(vals[:4]):
        trig.append("entropy <0.8x while k3 rises")
    st = WATCH if trig else OK
    value = "trigger: " + ", ".join(trig) if trig else "no trigger"
    if "rollout_corr/rollout_is_mean" in d:
        m, ess, sd = d.get("rollout_corr/rollout_is_mean"), d.get("rollout_corr/rollout_is_eff_sample_size"), \
            d.get("rollout_corr/rollout_is_std")
        if not (fin(m) and 0.5 <= m <= 2.0) or (ess is not None and not (fin(ess) and ess >= 0.3)) or \
                (sd is not None and not (fin(sd) and sd <= 1.0)):
            st = ALERT
        for k in ("rollout_corr/rollout_is_ratio_fraction_high", "rollout_corr/rollout_is_ratio_fraction_low"):
            if k in d:
                st = worst(st, hi(d[k], 0.001, 0.01))
        value += f"; IS mean {fmt(m)}, ESS {fmt(ess)}, std {fmt(sd)}"
    return R(st, value, "watch on a trigger (enable correction); with IS on alert outside verl's bands", s.label,
             "", len(trig))


@rule("CON-11", "consistency", "Qwen3.5 packing-boundary patch (log)", ["--log"],
      "one 'Monkey patch Qwen3_5... model forward' line per actor and ref worker",
      "results.md §2 (without boundaries loss 2.0838 vs 2.0243, max token diff 6.30); verl models/transformers/"
      "qwen3_5.py",
      "results.md §2 'pass sample boundaries to linear-attention and conv layers when packing Qwen3.5'",
      "Make sure verl applies the Qwen3.5 GDN patch (cu_seqlens); PPO critic: one sequence per micro-batch.")
def r_patch(c):
    if not c.log or not c.log["counts"]["qwen35"]:
        return None
    k = c.log["counts"]["qwen35_patch"]
    g = c.log["config"].get("n_gpus_per_node")
    return R(ALERT if k == 0 else OK, f"{k} patch lines", "alert 0 in a Qwen3.5 run", c.L.label,
             f"expected 2 x GPUs ({2 * g})" if g else "", k)


# --------------------------------------------------------------------------------------------------------- rollout rules
@rule("ROL-01", "rollout", "prompt truncation", ["prompt_length/max", "prompt_length/clip_ratio", "episodes"],
      "prompt_length/max < max_prompt_length; prompt clip_ratio = k x n / N (ties at the padded batch max)",
      "verl metric_utils.py (clip_ratio vs the padded batch max), v1 trainer_base to_padded_tensor",
      "none",
      "Raise max_prompt_length or shorten the fixed system prompt / tool schemas.")
def r_prompt(c):
    steps = [s for s in c.msteps if "prompt_length/max" in s.m]
    if not steps:
        return None
    L = c.max_prompt

    def f(s):
        pm = s.m["prompt_length/max"]
        em = s.ep["prompt_max"] if s.ep else None
        st, note = OK, []
        top = max(x for x in (pm, em) if x is not None)
        if L:
            if top >= L:
                st = ALERT
            elif top >= L - 64:
                st = WATCH
        cr, N, n = s.m.get("prompt_length/clip_ratio"), c.N(s.seg), c.n
        if fin(cr) and N and n:
            k = cr * N
            if abs(k - round(k)) > 0.01 or round(k) % n:
                st = worst(st, WATCH)
                note.append(f"clip_ratio x N = {k:.2f} not a multiple of n")
        return st, f"max {fmt(top)}" + (f" / limit {L}" if L else ""), ", ".join(note), top
    r = per_step(c, steps, f, f"alert max>=limit; watch >=limit-64" if L else "pass --max-prompt-length")
    if c.log and c.log["counts"]["prompt_trunc"]:
        r["status"] = ALERT
        r["note"] = (r["note"] + "; " if r["note"] else "") + \
            f"log has {c.log['counts']['prompt_trunc']} 'exceeds rollout.prompt_length' lines"
    return r


@rule("ROL-02", "rollout", "overlong (loss-masked) share", ["episodes.loss_masked", "response_length/clip_ratio"],
      "loss-masked share <0.20 sustained (<=0.15 per step), masked tokens <0.35, masked share >= budget-hit share",
      "research log 11-01 21:5x (20% rule); DAPO overlong filtering (arXiv 2503.14476); 0.15/0.35 judgment",
      "results.md §2 'the length limit must follow the hardest template'; log 21:5x option B; the log tracked "
      "response_length/clip_ratio, which misses ~40% of overlong episodes",
      "Track the 20% rule on loss_masked. If exceeded: raise the budget (~45k) or use the SP-2 / 61,440 "
      "configuration once its smoke test passes; DAPO soft overlong punishment.")
def r_overlong(c):
    steps = [s for s in c.steps if s.ep or (s.m and "response_length/clip_ratio" in s.m)]
    if not steps:
        return None
    seq = []

    def f(s):
        if s.ep:
            v = s.ep["masked"]
            seq.append(v)
            st = OK
            if v > 0.20 and len(seq) >= 2 and seq[-2] > 0.20:
                st = ALERT
            elif v > 0.15 or s.ep["masked_tok"] > 0.35:
                st = WATCH
            note = f"masked tokens {pct(s.ep['masked_tok'])}"
            if s.m and fin(s.m.get("response_length/clip_ratio")):
                cr = s.m["response_length/clip_ratio"]
                note += f", budget-hit {pct(cr)}"
                if c.budget and s.m.get("response_length/max") == c.budget and v < cr - 0.005:
                    st = worst(st, WATCH)
                    note += " (masked < budget-hit)"
            return st, pct(v), note, v
        cr = s.m["response_length/clip_ratio"]
        seq.append(cr)
        st = ALERT if fin(cr) and cr > 0.20 and len(seq) >= 2 and seq[-2] > 0.20 else hi(cr, 0.15, 10)
        return st, f"budget-hit {pct(cr)} (proxy)", "no episodes: clip_ratio understates overlong", cr
    return per_step(c, steps, f, "watch >0.15 or masked tokens >0.35; alert >0.20 on 2 consecutive steps")


def window_tmpl(steps):
    agg = defaultdict(Counter)
    for s in steps:
        for t, cnt in s.ep["tmpl"].items():
            agg[t].update(cnt)
    return agg


@rule("ROL-03", "rollout", "overlong share by template (4-step window)", ["episodes"],
      "each template's overlong share <=0.25 in a 4-step window",
      "research log 11-02 00:1x (templates with >=10% overlong are 'long'); 0.25 judgment",
      "results.md §2 'the length limit must follow the hardest template'",
      "Per-template budget, or the long-context configuration for the long templates (log 21:5x option C).")
def r_overlong_t(c):
    if len(c.esteps) < 1:
        return None
    win = c.esteps[-c.args.window:]
    agg = window_tmpl(win)
    shares = {t: v["over"] / v["n"] for t, v in agg.items() if v["n"] >= 16}
    if not shares:
        return None
    t, v = max(shares.items(), key=lambda kv: kv[1])
    top = sorted(shares.items(), key=lambda kv: -kv[1])[:3]
    return R(WATCH if v > 0.25 else OK, f"max {t} {pct(v)}", "watch any template >0.25",
             f"{win[0].label}..{win[-1].label}", "top: " + ", ".join(f"{a} {pct(b)}" for a, b in top), v)


@rule("ROL-04", "rollout", "response length", ["response_length/mean", "episodes.task_steps", "episodes.model_tokens"],
      "rolling 4-step mean within 0.75-1.25x the chain's first 4 steps; task_steps and model-token median stable",
      "judgment anchored on the run's first 4 steps; log 21:5x risk (zero-reward truncation pushes early submission); "
      "DAPO length growth (arXiv 2503.14476); 1.50x explosion bound judgment",
      "log 21:5x pitfall '截断轨迹不要直接给 0，否则倾向提前提交' (truncated trajectories at 0 push early submission)",
      "Falling with pass rate: re-examine the zero reward masked overlong rollouts keep in the group baseline. "
      "Growing to the budget: ROL-02 fixes.")
def r_len(c):
    ser = c.series("response_length/mean")
    if len(ser) < 2:
        return None
    vals = [v for _, v in ser]
    base = mean(vals[:4])
    sc = dict((s.g, s.m.get("critic/score/mean")) for s, _ in ser)
    sbase = mean(v for v in [sc[s.g] for s, _ in ser[:4]] if fin(v)) if any(fin(sc[s.g]) for s, _ in ser[:4]) else None
    ebase = c.esteps[:4]
    ts0 = mean(s.ep["task_steps"] for s in ebase if s.ep["task_steps"] is not None) if ebase and \
        ebase[0].ep["task_steps"] is not None else None
    mt0 = mean(s.ep["model_tok_median"] for s in ebase if s.ep["model_tok_median"] is not None) if ebase and \
        ebase[0].ep["model_tok_median"] is not None else None
    w = c.args.window

    def f(s):
        i = [x for x, _ in ser].index(s)
        if i < 4:
            return None
        roll = mean(vals[max(0, i - w + 1):i + 1])
        if not fin(roll):
            return ALERT, fmt(roll), "non-finite", roll
        r = roll / base
        st = ALERT if r > 1.50 else WATCH if r > 1.25 or r < 0.75 else OK
        note = []
        sroll = [sc[x.g] for x, _ in ser[max(0, i - w + 1):i + 1] if fin(sc[x.g])]
        if r < 0.85 and sbase and sroll and sbase - mean(sroll) >= 0.05:
            st = ALERT
            note.append("shorter with reward falling")
        win = [x for x in c.esteps if s.g - w < x.g <= s.g]
        if win and ts0 and win[-1].ep["task_steps"] is not None:
            ts = mean(x.ep["task_steps"] for x in win)
            if ts < 0.8 * ts0:
                st = ALERT
                note.append(f"task_steps {ts:.2f} vs {ts0:.2f}")
        if win and mt0 and win[-1].ep["model_tok_median"] is not None:
            mt = mean(x.ep["model_tok_median"] for x in win) / mt0
            if mt > 1.25 or mt < 0.75:
                st = worst(st, WATCH)
            note.append(f"model-token median {mt:.2f}x")
        return st, f"rolling {fmt(roll)} ({r:.2f}x base {fmt(base)})", ", ".join(note), r
    return per_step(c, [s for s, _ in ser], f,
                    "watch <0.75x|>1.25x; alert >1.50x, or <0.85x with reward -0.05, or task_steps -20%")


@rule("ROL-05", "rollout", "aborted rollouts", ["response/aborted_ratio"],
      "0 (verl counts a zero-length response as aborted)",
      "verl metric_utils.py (aborted_mask = response_length == 0)",
      "results.md §6 'service processes can die silently; resume logic must catch it'",
      "Check the vLLM servers and rollout failures in the log (SYS-10).")
def r_abort(c):
    ser = c.series("response/aborted_ratio")
    if not ser:
        return None
    return per_step(c, [s for s, _ in ser],
                    lambda s: (hi(s.m["response/aborted_ratio"], 0, 0.01), fmt(s.m["response/aborted_ratio"]), "",
                               s.m["response/aborted_ratio"]), "watch >0; alert >0.01")


@rule("ROL-06", "rollout", "turns and step caps", ["num_turns/mean", "episodes.end"],
      "num_turns/mean within 0.8-1.2x the first 4 steps; step_limit <=2%; hard_cap 0; only known end values",
      "judgment with M1 screening as baseline (8.31 turns, step_limit 1.6%); harness v3 agent.py (hard_cap never "
      "reached by design)",
      "results.md §4 'checklist calls and redo turns used the same 100-step limit' (fixed in harness v3)",
      "Check harness limits and the redo/checklist accounting.")
def r_turns(c):
    steps = [s for s in c.steps if (s.m and "num_turns/mean" in s.m) or s.ep]
    if not steps:
        return None
    nt = [s.m["num_turns/mean"] for s in c.msteps if "num_turns/mean" in s.m]
    base = mean(nt[:4]) if nt else None

    def f(s):
        st, parts = OK, []
        if s.m and "num_turns/mean" in s.m and base:
            r = s.m["num_turns/mean"] / base if fin(s.m["num_turns/mean"]) else float("nan")
            st = worst(st, ALERT if not fin(r) else (WATCH if not 0.8 <= r <= 1.2 else OK))
            parts.append(f"turns {fmt(s.m['num_turns/mean'])} ({r:.2f}x)")
        if s.ep:
            e, N = s.ep["ends"], s.ep["N"]
            sl = e.get("step_limit", 0) / N
            st = worst(st, hi(sl, 0.02, 0.05))
            parts.append(f"step_limit {pct(sl)}")
            if e.get("hard_cap"):
                st = ALERT
                parts.append(f"hard_cap {e['hard_cap']}")
            unknown = {k: v for k, v in e.items() if k not in KNOWN_ENDS}
            if unknown:
                st = ALERT
                parts.append(f"unknown ends {unknown}")
        return st, ", ".join(parts), "", None
    return per_step(c, steps, f, "watch turns outside 0.8-1.2x or step_limit>2%; alert step_limit>5%, hard_cap, "
                                 "unknown end")


@rule("ROL-07", "rollout", "sequence-length balance across ranks", ["global_seqlen/balanced_max",
                                                                   "global_seqlen/balanced_min",
                                                                   "perf/total_num_tokens"],
      "balanced_max/balanced_min <=1.05; total tokens <=1.3x the run's first 4 steps",
      "judgment from verl's balance_batch purpose (trainer.balance_batch=True)",
      "none",
      "Check trainer.balance_batch; growing tokens predict longer steps (SYS-01).")
def r_bal(c):
    steps = [s for s in c.msteps if "global_seqlen/balanced_max" in s.m or "perf/total_num_tokens" in s.m]
    if not steps:
        return None

    def f(s):
        st, parts = OK, []
        a, b = s.m.get("global_seqlen/balanced_max"), s.m.get("global_seqlen/balanced_min")
        if fin(a) and fin(b) and b > 0:
            r = a / b
            st = worst(st, ALERT if r > 1.05 else OK)
            parts.append(f"balance {r:.4f}")
        t = s.m.get("perf/total_num_tokens")
        tb = [x.m["perf/total_num_tokens"] for x in c.seg_msteps(s.seg) if "perf/total_num_tokens" in x.m][:4]
        if fin(t) and tb:
            r2 = t / mean(tb)
            st = worst(st, WATCH if r2 > 1.3 else OK)
            parts.append(f"tokens {r2:.2f}x")
        return st, ", ".join(parts), "", None
    return per_step(c, steps, f, "alert balance>1.05; watch tokens>1.3x first 4 steps of the run")


@rule("ROL-08", "rollout", "tool errors", ["episodes.tool_errors"],
      "per-step mean <=1.3x the chain's first 4 steps; share with >2 errors <=0.40; per template <=1.5x its first "
      "window",
      "judgment with M1 screening as baseline (1.85 errors, 20.9% with >2); log 11-01 20:1x unknown-name calls",
      "log 20:1x 'turns calling a non-existent tool name' (model's own sampling); results.md §3 gpt-oss misspells "
      "tool names",
      "No RL change; for the next SFT round mask format-error and unknown-name turns out of the loss "
      "(form_errors.py patch).")
def r_tools(c):
    steps = [s for s in c.esteps if s.ep["tool_err"] is not None]
    if not steps:
        return None
    base = mean(s.ep["tool_err"] for s in steps[:4])
    w = c.args.window
    first, last = window_tmpl(steps[:w]), window_tmpl(steps[-w:])
    worst_t = None
    for t, v in last.items():
        if v["n"] >= 16 and first.get(t, {}).get("n", 0) >= 16:
            r = (v["terr"] / v["n"]) / max(first[t]["terr"] / first[t]["n"], 1e-9)
            if worst_t is None or r > worst_t[1]:
                worst_t = (t, r)

    def f(s):
        v, g2 = s.ep["tool_err"], s.ep["tool_gt2"]
        st = WATCH if v > 1.3 * base or g2 > 0.40 else OK
        note = ""
        if s is steps[-1] and worst_t:
            note = f"max template {worst_t[0]} {worst_t[1]:.2f}x first window"
            if worst_t[1] > 1.5:
                st = worst(st, WATCH)
        return st, f"{v:.2f}/episode ({v / base:.2f}x), >2 errors {pct(g2)}", note, v
    r = per_step(c, steps, f, "watch >1.3x base, >2-error share >0.40, or a template >1.5x")
    if c.log and c.log["counts"]["unknown_tool"]:
        r["note"] += f"; log: {c.log['counts']['unknown_tool']} unknown-tool-name warnings"
    return r


@rule("ROL-09", "rollout", "unverified finishes (reward-hacking proxy)", ["episodes.verification_rejects"],
      "passes after 2 verification rejections <=3.2% of passes per 4-step window (2x M1's 1.6%); per template <=8%",
      "grader audit, research log 11-01 19:3x-20:0x (M1 68/4,188 = 1.6%); 2x factor and 8% judgment",
      "grader audit and author reply (log 21:5x): the grader can be gamed and H1 accepts the 3rd finish unverified "
      "(max_redos = 2); results.md §4 H1's rejection message leaks information",
      "Strict regrading of stored answers (log 22:3x patch 1); log answers (added from PPO on); chain evaluation "
      "under harness v3 with the strict grader.")
def r_unver(c):
    steps = [s for s in c.esteps if s.ep["unver"] is not None]
    if not steps:
        return None
    w = c.args.window
    win = steps[-w:]
    u_n, p_n = sum(s.ep["unver"] for s in win), sum(s.ep["passes"] for s in win)
    u = u_n / p_n if p_n else 0.0
    st = hi(u, 0.032, 0.05)
    agg = window_tmpl(win)
    note = []
    for t, v in sorted(agg.items()):
        if v.get("unver", 0) >= 5 and v["passes"] and v["unver"] / v["passes"] > 0.08:
            st = worst(st, WATCH)
            note.append(f"{t} {v['unver']}/{v['passes']}")
    if c.dev and c.dev[-1]["passes"]:
        d = c.dev[-1]
        du = d["unver"] / d["passes"]
        note.append(f"dev {d['label']} {d['unver']}/{d['passes']}")
        if d["unver"] >= 5 and du > 0.05:
            st = worst(st, WATCH)
    return R(st, f"{u_n}/{p_n} = {pct(u)}", "watch >3.2% (template >8% with >=5; dev >5% with >=5); alert >5%",
             f"{win[0].label}..{win[-1].label}", ", ".join(note), u)


@rule("ROL-10", "rollout", "template saturation and collapse", ["episodes"],
      "no template collapsed (pass <=0.10 or -0.15 vs its first window); saturated templates (pass >=0.95 or "
      "all-pass groups >=0.80) noted",
      "research log 11-01 20:0x ('only mid pass-rate tasks carry gradient') and 11-02 18:4x; DAPO dynamic sampling "
      "(arXiv 2503.14476)",
      "log 22:3x patch 3 (select tasks by k-run pass rate); the 25% k/k selection brought in rolling_vol tasks",
      "k-run task selection (make_grpo_data.py --drop-unsolved/--runs/--solved-frac) for the next run; DAPO dynamic "
      "sampling.")
def r_sat(c):
    if not c.esteps:
        return None
    w = c.args.window
    first, last = window_tmpl(c.esteps[:w]), window_tmpl(c.esteps[-w:])
    sat, col = [], []
    for t, v in sorted(last.items()):
        if v["n"] < 16:
            continue
        p = v["rew"] / v["n"]
        ap = v["allpass"] / v["groups"] if v["groups"] else 0
        if p >= 0.95 or ap >= 0.80:
            sat.append(f"{t} pass {p:.3f}, all-pass {pct(ap)}")
        f0 = first.get(t)
        p0 = f0["rew"] / f0["n"] if f0 and f0["n"] >= 16 else None
        if p <= 0.10 or (p0 is not None and p0 - p >= 0.15):
            col.append(f"{t} {p:.3f}" + (f" (was {p0:.3f})" if p0 is not None else ""))
    st = ALERT if col else (WATCH if sat else OK)
    val = ("collapsed: " + "; ".join(col)) if col else ("saturated: " + "; ".join(sat) if sat else "none")
    return R(st, val, "alert collapse; watch saturation", f"{c.esteps[-w:][0].label}..{c.esteps[-1].label}", "",
             len(col))


@rule("ROL-11", "rollout", "template drift (train window, same tasks, dev)", ["episodes"],
      "template window pass within -0.08 of its first window; same-task delta not < -0.10 at >2 SE; dev template "
      "loses <4 tasks and McNemar p>=0.05",
      "judgment; research log 11-02 13:4x / 18:4x (single-sample 200-task dev rounds are within noise); results.md "
      "§5 each test task is run once",
      "none directly; long templates (log 00:1x) point to the overlong budget (ROL-02)",
      "Confirm with the chain evaluation (dev x4, paired bootstrap); if a long template keeps drifting, apply the "
      "ROL-02 budget fixes to it.")
def r_drift(c):
    if not c.esteps:
        return None
    w = c.args.window
    win = c.esteps[-w:]
    first, last = window_tmpl(c.esteps[:w]), window_tmpl(win)
    st, notes = OK, []
    for t, v in sorted(last.items()):
        f0 = first.get(t)
        if v["n"] < 16 or not f0 or f0["n"] < 16:
            continue
        p, p0 = v["rew"] / v["n"], f0["rew"] / f0["n"]
        o, o0 = v["over"] / v["n"], f0["over"] / f0["n"]
        r, r0 = v["resp"] / v["n"], f0["resp"] / f0["n"]
        if p0 - p >= 0.08 or (o - o0 >= 0.05 and r0 and r / r0 >= 1.15):
            st = worst(st, WATCH)
            notes.append(f"{t} pass {p0:.3f}->{p:.3f}, overlong {pct(o0)}->{pct(o)}")
    by_t = defaultdict(list)
    for s in win:
        for _, t, d in s.pairs:
            by_t[t].append(d)
    for t, ds in sorted(by_t.items()):
        if len(ds) >= 10:
            m = mean(ds)
            se = statistics.stdev(ds) / math.sqrt(len(ds))
            if m < -0.10 and abs(m) > 2 * se:
                st = ALERT
                notes.append(f"{t} same-task {m:+.3f} (SE {se:.3f})")
    if len(c.dev) >= 2:
        base, d = c.dev[0], c.dev[-1]
        common = set(base["res"]) & set(d["res"])
        b = sum(base["res"][x] > 0.5 and d["res"][x] <= 0.5 for x in common)
        cc = sum(base["res"][x] <= 0.5 and d["res"][x] > 0.5 for x in common)
        if b > cc and mcnemar_p(b, cc) < 0.05:  # a loss only: a significant dev gain is not drift
            st = ALERT
            notes.append(f"dev overall -{b - cc} (lost {b}, gained {cc}), McNemar p {mcnemar_p(b, cc):.3f}")
        for t in sorted(set(base["tmpl"].values())):
            ts = [x for x in common if base["tmpl"].get(x) == t]
            bt = sum(base["res"][x] > 0.5 and d["res"][x] <= 0.5 for x in ts)
            ct = sum(base["res"][x] <= 0.5 and d["res"][x] > 0.5 for x in ts)
            if bt > ct and mcnemar_p(bt, ct) < 0.05:
                st = ALERT
                notes.append(f"dev {t} McNemar p {mcnemar_p(bt, ct):.3f}")
            elif bt - ct >= 4:
                st = worst(st, WATCH)
                notes.append(f"dev {t} -{bt - ct}/{len(ts)} vs {base['label']}")
    return R(st, "; ".join(notes[:4]) + (f" (+{len(notes) - 4})" if len(notes) > 4 else "") if notes else "none",
             "watch pass -0.08, overlong +0.05 with length +15%, dev template -4; alert same-task <-0.10 at 2 SE or "
             "McNemar p<0.05", f"{win[0].label}..{win[-1].label}" + (f", dev {c.dev[-1]['label']}" if c.dev else ""),
             "", len(notes))


@rule("ROL-12", "rollout", "episode/metrics bookkeeping", ["metrics.jsonl", "episodes", "ckpt/"],
      "one metrics line per finished step; batch x n episode rows per step; episodes reproduce critic/score/mean "
      "(1e-4), response_length/mean (0.5) and perf/total_num_tokens",
      "judgment from the per-step identity verified on run1b; results.md Appendix §6",
      "results.md §6 'stopping and resuming can leave half-written or duplicate rows' (停止再续跑可能留下写了一半的行或"
      "重复的行); run1's step-4 metrics line was lost by the step-4 pause",
      "Pause scripts should wait for the step's metrics line; recover reward/advantage statistics from episodes.")
def r_books(c):
    st, notes = OK, []
    for seg in c.segs:
        mi = seg.minfo
        if mi["bad_lines"]:
            st = ALERT
            notes.append(f"{seg.label}: {mi['bad_lines']} unparsable metrics lines")
        if mi["duplicates"]:
            st = worst(st, WATCH)
            notes.append(f"{seg.label}: duplicate steps {sorted(set(mi['duplicates']))} (last kept)")
        if mi["partial_last"]:
            notes.append(f"{seg.label}: partial last metrics line ignored")
        if seg.ep_info["bad"]:
            st = ALERT
            notes.append(f"{seg.label}: {seg.ep_info['bad']} unparsable episode rows")
        if seg.ep_info["misaligned_dev"]:
            st = worst(st, WATCH)
            notes.append(f"{seg.label}: {seg.ep_info['misaligned_dev']} dev clusters not at a step boundary")
        tsteps = [s for s in seg.train_steps if seg is not c.main or s <= c.main.cutoff]
        if tsteps:
            gaps = sorted(set(range(tsteps[0], tsteps[-1] + 1)) - set(tsteps))
            if gaps:
                st = ALERT if seg is c.main else worst(st, WATCH)
                notes.append(f"{seg.label}: steps {gaps} have no metrics line")
        if seg.ep_note and seg.ep_note != "no episodes":
            st = worst(st, WATCH)
            notes.append(f"{seg.label}: {seg.ep_note}")
        # finished steps with episodes but no metrics line
        for s in sorted(seg.ep_steps):
            if s not in seg.metrics or not is_train(seg.metrics[s]):
                if seg is not c.main:
                    st = worst(st, WATCH)
                    ck_note = " (its checkpoint exists)" if os.path.isdir(os.path.join(seg.dir, "ckpt",
                                                                                       f"global_step_{s}")) else ""
                    notes.append(f"{seg.label}:{s} has {seg.ep_steps[s]['N']} episodes and no metrics line{ck_note}")
        # checkpoints without a metrics line
        ck = os.path.join(seg.dir, "ckpt")
        if os.path.isdir(ck):
            for name in os.listdir(ck):
                mm = re.fullmatch(r"global_step_(\d+)", name)
                if mm:
                    k = int(mm.group(1))
                    if k not in seg.metrics and k not in seg.ep_steps and (seg is not c.main or k < c.main.cutoff):
                        st = worst(st, WATCH)
                        notes.append(f"{seg.label}: ckpt global_step_{k} has no metrics line")
    if c.main.p:
        rest, p = c.main.ep_info["rest_train"], c.main.p * c.main.W
        if rest >= p + c.main.W and c.main.cutoff + 1 not in c.main.metrics:
            st = worst(st, WATCH)
            notes.append(f"step {c.main.cutoff + 1} rollouts done and the next step started, no metrics line")
        elif rest and c.main.cutoff + 1 in c.main.metrics:
            notes.append(f"{rest} episode rows after the cut-off ignored")
        elif rest:
            notes.append(f"{rest} rows of step {c.main.cutoff + 1} in progress (ignored)")
    bad = []
    for s in c.steps:
        if s.ep is None:
            continue
        N = c.N(s.seg)
        if N and s.ep["N"] != N:
            bad.append(f"{s.label} rows {s.ep['N']}!={N}")
        if s.m:
            if fin(s.m.get("critic/score/mean")) and abs(s.ep["reward"] - s.m["critic/score/mean"]) > 1e-4:
                bad.append(f"{s.label} reward")
            if fin(s.m.get("response_length/mean")) and abs(s.ep["resp"] - s.m["response_length/mean"]) > 0.5:
                bad.append(f"{s.label} length")
            t = s.m.get("perf/total_num_tokens")
            if fin(t) and abs(s.ep["tokens"] - t) > 1e-3 * t:
                bad.append(f"{s.label} tokens")
    if bad:
        st = ALERT
        notes.append("episode/metrics mismatch: " + ", ".join(bad[:4]))
    ok_steps = sum(1 for s in c.steps if s.ep and s.m)
    return R(st, f"{ok_steps} steps cross-checked", "alert mismatch, gap or corrupt line; watch history gaps",
             c.L.label, "; ".join(notes), len(bad))


@rule("ROL-13", "rollout", "data-order replay within an epoch", ["episodes", "training/epoch"],
      "no task trained twice within one data epoch across the run chain",
      "systems verification of S14 (run1b restarted the dataloader with the same seed: steps 1-6 replayed run1's 512 "
      "tasks); research log 11-02 'another finding' after 22:4x",
      "research log 11-02 (grpo1 limitation: replayed tasks are trained twice before the rest of epoch 0)",
      "When restarting from merged weights: exclude already-trained tasks from the first epoch or change data.seed; "
      "with the same world size resume with resume_mode=auto.")
def r_replay(c):
    if not c.esteps:
        return None
    seen = defaultdict(set)
    last_key = {}
    epochs = {}
    hist = []
    prev_seg_key = None
    cur_e = {}
    latest_dup = 0
    for s in c.steps:
        e = s.m.get("training/epoch") if s.m else None
        if e is None:
            e = cur_e.get(s.seg, 0)
        cur_e[s.seg] = e
        if s.seg not in epochs:
            epochs[s.seg] = e
            key = prev_seg_key if (prev_seg_key is not None and s.seg > 0) else (s.seg, e)
            last_key[(s.seg, e)] = key
        key = last_key.setdefault((s.seg, e), (s.seg, e))
        prev_seg_key = key
        if s.ep is None:
            continue
        tasks = set(s.ep["task_pass"])
        dup = len(tasks & seen[key])
        seen[key] |= tasks
        if dup:
            hist.append(f"{s.label} {dup}")
        if s is c.esteps[-1]:
            latest_dup = dup
    st = WATCH if latest_dup or hist else OK
    note = ("history: " + ", ".join(hist[:6]) + (f" (+{len(hist) - 6})" if len(hist) > 6 else "")) if hist else ""
    return R(st, f"{latest_dup} repeated tasks at the latest step", "watch any task repeated within an epoch",
             c.esteps[-1].label, note, latest_dup)


# --------------------------------------------------------------------------------------------------------- systems rules
@rule("SYS-01", "systems", "step time and throughput", ["timing_s/step", "timing_s/save_checkpoint", "perf/throughput"],
      "step time without the save within 1.15x the run's running median; throughput >=0.85x the run median",
      "judgment; verl metric_utils compute_timing_metrics / compute_throughout_metrics",
      "results.md §7 'compute and time underestimated about 4x' (project from measured step times)",
      "Look for GPU co-tenants, throttling (SYS-02) or longer responses (ROL-04).")
def r_time(c):
    steps = [s for s in c.msteps if "timing_s/step" in s.m]
    if not steps:
        return None

    def base(s):
        return s.m["timing_s/step"] - (s.m.get("timing_s/save_checkpoint") or 0)

    def f(s):
        prev = [base(x) for x in steps if x.seg == s.seg and x.g < s.g]
        x = base(s)
        if not fin(x):
            return ALERT, fmt(x), "non-finite", x
        st, note = OK, ""
        if len(prev) >= 3:
            r = x / median(prev)
            st = ALERT if r > 1.30 or r < 0.60 else WATCH if r > 1.15 else OK
            note = f"{r:.2f}x median"
        tp = s.m.get("perf/throughput")
        tps = [y.m["perf/throughput"] for y in steps if y.seg == s.seg and "perf/throughput" in y.m and
               not y.m.get("timing_s/save_checkpoint")]
        if fin(tp) and tps and not s.m.get("timing_s/save_checkpoint") and tp < 0.85 * median(tps):
            st = worst(st, WATCH)
            note += f", throughput {tp:.0f} <0.85x median"
        return st, f"{x:.0f} s" + (f", {tp:.0f} tok/s/GPU" if fin(tp) else ""), note, x
    return per_step(c, steps, f, "watch >1.15x median; alert >1.30x or <0.60x")


@rule("SYS-02", "systems", "per-token compute cost", ["timing_per_token_ms/update_actor", "timing_per_token_ms/ref",
                                                     "timing_s/old_log_prob", "timing_s/update_weights"],
      "per-token update/ref/old-logprob cost within +-10% of the run's median (after step 1); update_weights <30 s",
      "judgment: fixed-shape compute on fixed hardware costs the same per token (drift = throttling, a co-tenant, "
      "host-routed collectives or fragmentation)",
      "results.md §6 GPU contention (shared GPUs taken between check and launch)",
      "Check nvidia-smi for foreign processes (SYS-13), clocks and temperatures.")
def r_cost(c):
    steps = [s for s in c.msteps if "timing_per_token_ms/update_actor" in s.m or
             "timing_per_token_ms/update_critic" in s.m]
    if not steps:
        return None

    def costs(s):
        d = {k.split("/")[-1]: s.m[k] for k in ("timing_per_token_ms/update_actor", "timing_per_token_ms/ref",
                                                 "timing_per_token_ms/update_critic", "timing_per_token_ms/values")
             if k in s.m}
        if fin(s.m.get("timing_s/old_log_prob")) and fin(s.m.get("perf/total_num_tokens")) and \
                s.m["perf/total_num_tokens"]:
            d["old_log_prob"] = s.m["timing_s/old_log_prob"] * 1000 / s.m["perf/total_num_tokens"]
        return d

    def f(s):
        seg = [x for x in steps if x.seg == s.seg]
        prev = [costs(x) for x in seg[1:] if x.g < s.g]
        cur = costs(s)
        st, parts = OK, []
        for k, v in cur.items():
            if not fin(v):
                st = ALERT
                parts.append(f"{k} {v}")
                continue
            hist_k = [p[k] for p in prev if k in p and fin(p[k])]
            if s is seg[0] or not hist_k:
                continue
            r = v / median(hist_k)
            if abs(r - 1) > 0.10:
                st = ALERT
            parts.append(f"{k} {r:.3f}x")
        uw = s.m.get("timing_s/update_weights")
        if uw is not None:
            st = worst(st, hi(uw, 30, 30))
            parts.append(f"update_weights {fmt(uw)} s")
        return st, ", ".join(parts) or "first step of the run", "", None
    return per_step(c, steps, f, "alert |cost/median-1|>0.10 or update_weights>30 s")


@rule("SYS-03", "systems", "rollout straggler tail", ["timing_s/gen", "episodes.seconds"],
      "gen / median episode time <=3.5; gen share of the step <=0.65",
      "research log 11-02 00:4x (synchronous GRPO waits for the slowest trajectory); thresholds judgment",
      "log 00:4x straggler note; results.md §7",
      "Efficiency only: async/partial rollout (needs rollout correction), a per-episode wall-clock or token cap for "
      "long templates, or cutting overlong episodes earlier (they carry no loss).")
def r_tailtime(c):
    steps = [s for s in c.steps if s.m and "timing_s/gen" in s.m]
    if not steps:
        return None

    def f(s):
        g = s.m["timing_s/gen"]
        if not fin(g):
            return ALERT, fmt(g), "non-finite", g
        st, parts = OK, []
        if s.ep and s.ep["sec_median"]:
            r = g / s.ep["sec_median"]
            st = worst(st, hi(r, 3.5, 5))
            parts.append(f"gen/median {r:.2f}")
        if fin(s.m.get("timing_s/step")) and s.m["timing_s/step"]:
            sh = g / s.m["timing_s/step"]
            st = worst(st, WATCH if sh > 0.65 else OK)
            parts.append(f"share {sh:.2f}")
        return st, ", ".join(parts), "", g
    return per_step(c, steps, f, "watch ratio>3.5 or share>0.65; alert ratio>5")


@rule("SYS-04", "systems", "model FLOPs utilization", ["perf/mfu/actor", "perf/mfu/critic"],
      "a 0 on every step means the GPU is missing from verl's flops table (dead metric); otherwise >=0.85x the run "
      "median",
      "verl utils/flops_counter.py (_DEVICE_FLOPS has no A6000 entry -> inf peak -> 0); offline 0.31 on A6000",
      "none",
      "Read efficiency from timing_per_token_ms/update_actor (SYS-02) or mfu_offline.py; add the GPU to "
      "_DEVICE_FLOPS in a future venv.")
def r_mfu(c):
    out = []
    for key in ("perf/mfu/actor", "perf/mfu/critic"):
        ser = c.series(key)
        if not ser:
            continue
        vals = [v for _, v in ser]
        if all(v == 0 for v in vals):
            out.append(R(OK, f"{key.split('/')[-1]} 0 on all steps", "dead metric", c.L.label,
                         "device missing from verl's flops table"))
            continue
        segv = [s.m[key] for s in c.seg_msteps(c.L.seg) if key in s.m and fin(s.m[key])]
        v = c.L.m.get(key)
        st = ALERT if not fin(v) or (segv and v < 0.85 * median(segv)) else OK
        out.append(R(st, f"{key.split('/')[-1]} {fmt(v)}", "alert <0.85x run median", c.L.label, "", v))
    if not out:
        return None
    return max(out, key=lambda o: RANK[o["status"]])


@rule("SYS-05", "systems", "GPU memory high-water mark", ["actor/perf/max_memory_reserved_gb",
                                                         "critic/perf/max_memory_reserved_gb"],
      "reserved high-water <=0.85x card memory",
      "judgment; results.md §2 (248k vocabulary vs memory, fixed by chunked entropy); log 11-02 05:0x (smoke_sp2 "
      "vLLM wake-up OOM)",
      "results.md §2 Qwen3.5 vocabulary memory; log 05:0x smoke_sp2 OOM",
      "PPO on 6 GPUs and larger budgets must re-check the margin at step 1; keep entropy chunking and checkpointing.")
def r_mem(c):
    G = c.args.gpu_mem_gib
    keys = [k for k in ("actor/perf/max_memory_reserved_gb", "critic/perf/max_memory_reserved_gb")
            if c.series(k)]
    if not keys:
        return None
    if not G:
        return R(SKIP, fmt(c.L.m.get(keys[0])), "pass --gpu-mem-gib", c.L.label)

    def f(s):
        vs = [s.m[k] for k in keys if k in s.m]
        v = max(vs)
        return hi(v, 0.85 * G, 0.90 * G), f"{fmt(v)} GiB of {fmt(G)}", "", v
    return per_step(c, [s for s in c.msteps if any(k in s.m for k in keys)], f,
                    f"watch >{0.85 * G:.1f}; alert >{0.90 * G:.1f} GiB")


@rule("SYS-06", "systems", "host memory in use", ["actor/perf/cpu_memory_used_gb"],
      "host-wide used memory <0.6x RAM (all users)",
      "verl engine_workers.py (psutil.virtual_memory().used); Ray OOM monitor kills at 0.95 (Ray docs); judgment",
      "none",
      "Find the process holding host memory (other users count too); Ray spilling settings.")
def r_host(c):
    ser = c.series("actor/perf/cpu_memory_used_gb")
    if not ser:
        return None
    ram = c.args.host_ram_gib
    if not ram:
        try:
            with open("/proc/meminfo") as f:
                ram = int(f.readline().split()[1]) / 1024 ** 2
        except OSError:
            ram = None
    if not ram:
        return None
    return per_step(c, [s for s, _ in ser],
                    lambda s: (hi(s.m["actor/perf/cpu_memory_used_gb"], 0.6 * ram, 0.85 * ram),
                               f"{fmt(s.m['actor/perf/cpu_memory_used_gb'])} of {ram:.0f} GiB", "",
                               s.m["actor/perf/cpu_memory_used_gb"]),
                    f"watch >{0.6 * ram:.0f}; alert >{0.85 * ram:.0f} GiB")


@rule("SYS-07", "systems", "checkpoint save", ["timing_s/save_checkpoint", "ckpt/latest_checkpointed_iteration.txt"],
      "save <900 s; latest_checkpointed_iteration == the latest save step; every rank's model and optim shard present",
      "judgment; the shared disk's usage rules (shared NFS: heavy I/O slows every user)",
      "none",
      "Check the NFS load and free space on the checkpoint disk (SYS-12).")
def r_ckpt(c):
    steps = [s for s in c.msteps if "timing_s/save_checkpoint" in s.m]
    if not steps:
        return None
    s = steps[-1]
    v = s.m["timing_s/save_checkpoint"]
    st = hi(v, 900, 1800)
    notes = []
    ck = os.path.join(c.main.dir, "ckpt")
    lt = os.path.join(ck, "latest_checkpointed_iteration.txt")
    if s.seg == c.main.idx and os.path.exists(lt):
        try:
            k = int(open(lt).read().strip())
        except ValueError:
            k = None
        notes.append(f"latest_checkpointed_iteration {k}")
        if k is None or k < s.step:
            st = ALERT
        actor = os.path.join(ck, f"global_step_{k}", "actor")
        if k is not None and os.path.isdir(actor):
            names = os.listdir(actor)
            nm = sum(n.startswith("model_") for n in names)
            no = sum(n.startswith("optim_") for n in names)
            notes.append(f"{nm} model / {no} optim shards")
            if nm == 0 or nm != no:
                st = ALERT
    return R(st, f"{v:.0f} s", "watch >900 s; alert >1800 s or missing shards", s.label, ", ".join(notes), v)


@rule("SYS-08", "systems", "epoch counter", ["training/epoch"],
      "epoch == floor((step-1) / floor(n_train / batch))",
      "research log 11-02 08:2x (24 steps x 96); judgment",
      "none",
      "Check data.train_files, drop_last and the dataloader state after a resume.")
def r_epoch(c):
    steps = [s for s in c.seg_msteps(c.main.idx) if "training/epoch" in s.m]
    if not steps:
        return None
    spe = None
    N, n = c.N(c.main.idx), c.n
    if c.args.n_train and N and n:
        spe = c.args.n_train // (N // n)
    else:
        ch = next((s for i, s in enumerate(steps[1:], 1) if s.m["training/epoch"] != steps[i - 1].m["training/epoch"]),
                  None)
        if ch:
            spe = ch.step - steps[0].step
    if not spe:
        return R(OK, f"epoch {steps[-1].m['training/epoch']}", "no epoch boundary yet", steps[-1].label)
    bad = [s.label for s in steps if s.m["training/epoch"] != (s.step - 1) // spe]
    return R(ALERT if bad else OK, f"epoch {steps[-1].m['training/epoch']}, {spe} steps/epoch",
             "alert epoch != floor((step-1)/steps_per_epoch)", steps[-1].label, ", ".join(bad[:4]), len(bad))


@rule("SYS-09", "systems", "projected end time", ["timing_s/step", "timing_s/save_checkpoint", "timing_s/testing"],
      "mean step time of the last 4 steps <=1.15x the run mean",
      "results.md §7 (estimates run several times low); research log 11-02 00:4x and 22:4x",
      "results.md §7 underestimated compute",
      "Plan the successor chain from the projected end, not the earlier estimate.")
def r_eta(c):
    steps = [s for s in c.seg_msteps(c.main.idx) if "timing_s/step" in s.m]
    if len(steps) < 2:
        return None
    base = [s.m["timing_s/step"] - (s.m.get("timing_s/save_checkpoint") or 0) for s in steps]
    m_all, m4 = mean(base), mean(base[-4:])
    st = WATCH if m4 > 1.15 * m_all else OK
    note = ""
    T = c.total_steps
    if T and T > c.L.step and c.L.step == c.main.train_steps[-1]:
        saves = [s.m["timing_s/save_checkpoint"] for s in steps if s.m.get("timing_s/save_checkpoint")]
        tests = [s.m["timing_s/testing"] for s in steps if s.m.get("timing_s/testing")]
        f = c.test_freq if c.test_freq and c.test_freq > 0 else None
        evals = sum(1 for k in range(c.L.step + 1, T + 1) if f and (k % f == 0 or k == T))
        rem = (T - c.L.step) * m_all + evals * ((mean(saves) or 0) + (mean(tests) or 0))
        t0 = os.path.getmtime(os.path.join(c.main.dir, "metrics.jsonl"))
        eta = datetime.datetime.fromtimestamp(t0 + rem, datetime.timezone.utc)
        note = f"ETA {eta:%Y-%m-%d %H:%M} UTC ({rem / 3600:.1f} h for {T - c.L.step} steps, {evals} eval steps)"
    elif T:
        note = "run reached total_training_steps"
    return R(st, f"last-4 mean {m4:.0f} s vs run mean {m_all:.0f} s", "watch last-4 >1.15x run mean", c.L.label,
             note, m4)


@rule("SYS-10", "systems", "error signatures in the training log", ["--log"],
      "0 Traceback / OOM / CUDA / NCCL / Ray actor death / preemption / nan / rollout failure lines",
      "judgment",
      "results.md §6 'service processes die silently'",
      "Read the first matching log line; check the guard status and the process list.")
def r_logerr(c):
    if not c.log:
        return None
    k = c.log["counts"]["errors"]
    return R(ALERT if k else OK, f"{k} lines", "alert >0", c.L.label,
             (str(c.log["error_kinds"]) if k else f"{c.log['counts']['progress']} rollout progress lines, "
                                                  f"failure 0"), k)


@rule("SYS-11", "systems", "liveness and hang (live)", ["metrics.jsonl mtime", "--pid"],
      "a new metrics line within 1.5x (median step + testing time); the driver process alive until the last step",
      "results.md §6 (vLLM idle SIGTERM, silent service death); research log 11-02 08:3x crash guard",
      "results.md §6 'service processes can die silently'",
      "Check the driver, the vLLM servers and the guard status; resume from the latest checkpoint with the same "
      "world size and resume_mode=auto.")
def r_live(c):
    if not c.args.live:
        return None
    steps = [s for s in c.seg_msteps(c.main.idx) if "timing_s/step" in s.m]
    if c.total_steps and c.L.step >= c.total_steps:
        return R(OK, "finished", "-", c.L.label)
    age = time.time() - os.path.getmtime(os.path.join(c.main.dir, "metrics.jsonl"))
    tests = [s.m["timing_s/testing"] for s in steps if s.m.get("timing_s/testing")]
    lim = 1.5 * ((median(s.m["timing_s/step"] for s in steps) if steps else 3600) + (max(tests) if tests else 0))
    st = ALERT if age > lim else OK
    note = ""
    if c.args.pid:
        alive = os.path.exists(f"/proc/{c.args.pid}")
        note = f"pid {c.args.pid} {'alive' if alive else 'gone'}"
        if not alive:
            st = ALERT
    return R(st, f"metrics age {age / 60:.0f} min", f"alert >{lim / 60:.0f} min or pid gone", c.L.label, note, age)


@rule("SYS-12", "systems", "free disk (live)", ["df /", "df RUN_DIR"],
      "root disk >=44 GB free; checkpoint disk holds >=1.5 checkpoints more",
      "results.md §6 (host 2 root disk chronically tight, 44-64 GB free); systems verification of S7 (Ray spills to "
      "disk2, so the Ray 99.5% threshold does not bind on /); judgment",
      "results.md §6 root disk; log 11-01 21:00 Ray temp dir on a nearly full root disk",
      "Send outputs, model copies and caches to host 1's disk2; keep Ray _temp_dir on a local filesystem with "
      "headroom (e.g. a local data disk of the training host), never on NFS.")
def r_disk(c):
    if not c.args.live:
        return None
    sv = os.statvfs("/")
    root = sv.f_bavail * sv.f_frsize / 1e9
    st = ALERT if root < 10 else WATCH if root < 44 else OK
    note = ""
    try:
        sv2 = os.statvfs(c.main.dir)
        free2 = sv2.f_bavail * sv2.f_frsize / 1e9
        ck = os.path.join(c.main.dir, "ckpt")
        size = 0
        if os.path.isdir(ck):
            steps = sorted((int(m.group(1)), n) for n in os.listdir(ck) if (m := re.fullmatch(r"global_step_(\d+)", n)))
            if steps:
                actor = os.path.join(ck, steps[-1][1], "actor")
                if os.path.isdir(actor):
                    size = sum(e.stat().st_size for e in os.scandir(actor) if e.is_file()) / 1e9
        note = f"run disk {free2:.0f} GB free, checkpoint {size:.1f} GB"
        if size and free2 < 1.5 * size:
            st = ALERT
    except OSError:
        pass
    return R(st, f"/ {root:.1f} GB free", "watch </44 GB; alert <10 GB or run disk <1.5 checkpoints", "now", note,
             root)


@rule("SYS-13", "systems", "foreign processes on the run's GPUs (live)", ["nvidia-smi", "--gpus"],
      "only the run owner's processes on the run's GPUs",
      "results.md §6 (shared GPUs taken between the check and the launch); research log 11-02 05:1x and 07:54",
      "results.md §6 GPU contention",
      "Ask the other user / administrator; never use other users' GPUs (memory note fin-posttrain-gpu-cap).")
def r_gpus(c):
    if not c.args.live or not c.args.gpus:
        return None
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=30).stdout
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return R(SKIP, "nvidia-smi unavailable")
    idx = {u.strip(): int(i) for i, u in (l.split(",") for l in q.strip().splitlines() if "," in l)}
    want = {int(x) for x in c.args.gpus.split(",")}
    owner = c.args.owner or os.environ.get("USER", "")
    foreign = []
    for line in a.strip().splitlines():
        if "," not in line:
            continue
        u, pid = (x.strip() for x in line.split(","))
        if idx.get(u) not in want:
            continue
        try:
            import pwd
            who = pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
        except (OSError, KeyError, ImportError):
            who = "?"
        if who != owner:
            foreign.append(f"gpu{idx.get(u)} pid {pid} {who}")
    return R(ALERT if foreign else OK, f"{len(foreign)} foreign", "alert >0", "now", "; ".join(foreign[:4]),
             len(foreign))


@rule("SYS-14", "systems", "orphaned stub-test loops (live)", ["/proc"],
      "no stub-test process outlives its test (1 h)",
      "research log 11-02 07:54 incident (stub tests must not leave real effects); judgment",
      "log 11-02 07:54 incident (GPU-claim stub tests)",
      "Stop them by PID (not pkill -f, which matches the calling shell); gpu-claim tests should kill their loops on "
      "teardown.")
def r_stub(c):
    if not c.args.live:
        return None
    uid, found = os.getuid(), []
    try:
        hz = os.sysconf("SC_CLK_TCK")
        up = float(open("/proc/uptime").read().split()[0])
    except (OSError, ValueError):
        return None
    pat = c.args.stub_pattern.encode()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            cmd = open(f"/proc/{pid}/cmdline", "rb").read()
            if pat not in cmd:
                continue
            start = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19]) / hz
            if up - start > 3600:
                found.append(pid)
        except (OSError, ValueError, IndexError):
            continue
    return R(WATCH if found else OK, f"{len(found)} loops", "watch any older than 1 h", "now",
             ("PIDs " + " ".join(found[:8])) if found else "", len(found))


# ------------------------------------------------------------------------------------------------------------ PPO rules
@rule("PPO-01", "ppo", "value loss", ["critic/vf_loss"],
      "finite, below 2x the median of the previous 10 steps",
      "judgment (mirrors OPT-09)",
      "none",
      "Lower the critic lr, check reward scale and the critic's packing (one sequence per micro-batch for Qwen3.5).")
def r_vf(c):
    if not c.is_ppo:
        return None
    return spike_rule(c, "critic/vf_loss", None)


@rule("PPO-02", "ppo", "value explained variance", ["critic/vf_explained_var"],
      ">=0 after the critic's first 5 steps",
      "stable-baselines3 explained_variance (ev=0: no better than a constant; ev<0: worse); thresholds judgment",
      "log 11-02 04:5x (critic packing leak without sample boundaries)",
      "Check critic warmup, value-head init, lr and sample boundaries in packed critic forwards.")
def r_ev(c):
    ser = c.series("critic/vf_explained_var")
    if not ser or not c.is_ppo:
        return None
    vals = []

    def f(s):
        v = s.m["critic/vf_explained_var"]
        vals.append(v)
        if not fin(v):
            return ALERT, fmt(v), "non-finite (response mask all False?)", v
        if len(vals) <= 5:
            return OK, fmt(v), "critic warm-up", v
        st = ALERT if v < -1.0 else WATCH if v < 0 and len(vals) >= 3 and all(x < 0 for x in vals[-3:]) else OK
        return st, fmt(v), "", v
    return per_step(c, [s for s, _ in ser], f, "watch <0 on 3 steps after step 5; alert <-1")


@rule("PPO-03", "ppo", "value clip fraction", ["critic/vf_clipfrac"],
      "<=0.3",
      "judgment (verl cliprange_value)",
      "none",
      "Lower the critic lr or widen cliprange_value.")
def r_vfclip(c):
    ser = c.series("critic/vf_clipfrac")
    if not ser or not c.is_ppo:
        return None
    return per_step(c, [s for s, _ in ser], lambda s: (hi(s.m["critic/vf_clipfrac"], 0.3, 0.6),
                                                       fmt(s.m["critic/vf_clipfrac"]), "", s.m["critic/vf_clipfrac"]),
                    "watch >0.3; alert >0.6")


@rule("PPO-04", "ppo", "critic gradient norm", ["critic/grad_norm"],
      "finite, below grad_clip, below 2x the median of the previous 10 steps",
      "judgment (mirrors OPT-09)",
      "none",
      "Lower the critic lr; keep gradient clipping.")
def r_cgrad(c):
    if not c.is_ppo:
        return None
    return spike_rule(c, "critic/grad_norm", c.grad_clip)


@rule("PPO-05", "ppo", "value bias", ["critic/values/mean", "critic/returns/mean", "critic/vpred_mean"],
      "|values/mean - returns/mean| <=0.25x the returns range",
      "judgment",
      "none",
      "Check value-head init and critic warmup; a persistent bias makes GAE advantages one-signed.")
def r_vbias(c):
    steps = [s for s in c.msteps if "critic/values/mean" in s.m and "critic/returns/mean" in s.m]
    if not steps or not c.is_ppo:
        return None
    k = []

    def f(s):
        v, r = s.m["critic/values/mean"], s.m["critic/returns/mean"]
        rng = (s.m.get("critic/returns/max", 1) or 1) - (s.m.get("critic/returns/min", 0) or 0) or 1
        if not (fin(v) and fin(r)):
            return ALERT, f"{fmt(v)} vs {fmt(r)}", "non-finite", None
        b = abs(v - r) / abs(rng)
        k.append(b)
        st = ALERT if b > 0.5 else WATCH if len(k) > 5 and len(k) >= 3 and all(x > 0.25 for x in k[-3:]) else OK
        return st, f"values {fmt(v)} vs returns {fmt(r)} ({b:.2f} of range)", "", b
    return per_step(c, steps, f, "watch >0.25 of range on 3 steps after step 5; alert >0.5")


@rule("PPO-06", "ppo", "whitened advantage mean", ["critic/advantages/mean"],
      "|mean| <=0.05 (GAE advantages are whitened over response tokens)",
      "verl core_algos compute_gae_advantage_return (masked_whiten); threshold judgment",
      "none",
      "Check response_mask (overlong masking) and the whitening.")
def r_ppoadv(c):
    if not c.is_ppo:
        return None
    ser = c.series("critic/advantages/mean")
    if not ser:
        return None
    return per_step(c, [s for s, _ in ser], lambda s: (hi(abs(s.m["critic/advantages/mean"]), 0.05, 0.5),
                                                       fmt(s.m["critic/advantages/mean"]), "",
                                                       s.m["critic/advantages/mean"]), "watch |mean|>0.05; alert >0.5")


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--prev", help="earlier run directory of the same lineage, prepended to the trend")
    ap.add_argument("--label"), ap.add_argument("--prev-label")
    ap.add_argument("--log", help="training log of RUN_DIR")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--cutoff", type=int, help="last step of RUN_DIR to evaluate (default: last metrics line)")
    ap.add_argument("--n", type=int, help="rollouts per task (default: detected from episodes)")
    ap.add_argument("--rows-per-worker", type=int, help="episode rows per worker file per step for RUN_DIR")
    ap.add_argument("--prev-rows-per-worker", type=int)
    ap.add_argument("--no-episodes", action="store_true")
    ap.add_argument("--dev-marker", default="-dev-")
    ap.add_argument("--budget", type=int, help="response token budget (default: log or max response_length/max)")
    ap.add_argument("--max-prompt-length", type=int)
    ap.add_argument("--grad-clip", type=float)
    ap.add_argument("--single-update", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--adv-estimator")
    ap.add_argument("--ref-k3", type=float, help="k3_kl of a same-weights reference run (e.g. smoke test step 1)")
    ap.add_argument("--max-staleness", type=int, default=0)
    ap.add_argument("--total-steps", type=int), ap.add_argument("--test-freq", type=int)
    ap.add_argument("--n-train", type=int, help="number of training tasks (epoch-counter check)")
    ap.add_argument("--gpu-mem-gib", type=float), ap.add_argument("--host-ram-gib", type=float)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--boot", type=int, default=10000), ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--live", action="store_true", help="host checks for a running job")
    ap.add_argument("--pid", type=int), ap.add_argument("--gpus"), ap.add_argument("--owner")
    ap.add_argument("--stub-pattern", default="GPU_STUB_DIR")
    args = ap.parse_args()
    c = Ctx(args)
    results = []
    for r in RULES:
        try:
            out = r["fn"](c)
        except Exception as e:  # a rule bug must not hide the other rules
            out = R(WATCH, "rule error", "-", "-", f"{type(e).__name__}: {e}")
        if out is None:
            out = R(SKIP, "-", "-", "-", "keys or inputs absent")
        meta = {k: v for k, v in r.items() if k != "fn"}
        results.append({**meta, **out})
    n_alert = sum(r["status"] == ALERT for r in results)
    code = 2 if n_alert else 0
    seg_desc = []
    for s in c.segs:
        tr = [x for x in s.train_steps if s is not c.main or x <= c.main.cutoff]
        seg_desc.append({"run": s.label, "dir": s.dir, "metric_steps": f"{tr[0]}-{tr[-1]}" if tr else "-",
                         "episode_steps": f"{min(s.ep_steps)}-{max(s.ep_steps)}" if s.ep_steps else "-",
                         "rows_per_worker": s.p, "workers": s.W, "dev_rounds": sorted(s.dev_rounds),
                         "single_update": c.single.get(s.idx), "note": s.ep_note})
    meta = {"cutoff": c.L.label, "chain_steps": c.L.g, "algorithm": "PPO" if c.is_ppo else c.estimator.upper(),
            "n": c.n, "budget": c.budget, "max_prompt_length": c.max_prompt, "grad_clip": c.grad_clip,
            "total_steps": c.total_steps, "runs": seg_desc,
            "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    counts = Counter(r["status"] for r in results)
    if args.json:
        print(json.dumps({"meta": meta, "rules": results, "counts": counts, "exit_code": code}, indent=1,
                         default=str))
        return code
    runs = ", ".join(f"{d['run']} steps {d['metric_steps']} (episodes {d['episode_steps']}, "
                     f"{d['rows_per_worker']}x{d['workers']} rows/step)" for d in seg_desc)
    print(f"RL health {meta['generated']} | cut-off {meta['cutoff']} (chain step {meta['chain_steps']}) | "
          f"{meta['algorithm']} n={c.n} budget={c.budget} | {runs}")
    for r in results:
        line = f"{r['status']:<5} {r['id']:<6} {r['title']} | value: {r['value']} | threshold: {r['threshold']} | " \
               f"step: {r['step']}"
        if r["note"]:
            line += f" | {r['note']}"
        print(line)
    print(f"summary: {counts.get(OK, 0)} ok, {counts.get(WATCH, 0)} watch, {counts.get(ALERT, 0)} alert, "
          f"{counts.get(SKIP, 0)} skip -> exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
