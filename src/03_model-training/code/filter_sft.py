"""Multi-stage filtering of agent-SFT trajectories and their tasks (research/log/2026-10-31-0150.md): which collected
episodes may enter a training set, and a funnel of how many each stage removes.

Trajectory stages, in order (each counts only what the stages before it kept):
  1 passed          the episode passed grading;
  2 clean_end       it ended by finishing (not on the step limit, context, a server or harness error);
  3 not_truncated   no reply hit max_tokens (v3 rows record it; v2 rows do not, so v2 episodes pass unchecked);
  4 grounded        the accepted answer passed the harness's own verification: fewer rejections than the 2 redos
                    allowed, since after the last redo a finish is accepted unverified;
  5 no_loop         no tool called 3 or more times with the same arguments, and no loop hint from the harness;
  6 few_tool_errors at most 2 tool errors as the harness counts them (the row's tool_errors): an episode with a form
                    error among them (a tool name it was not offered, or arguments that do not bind to the tool's
                    parameters; form_errors.py) is kept on the same terms, since the builders take that turn out of
                    the loss and keep it in the context, so the recovery after it is still learned
                    (research/log/2026-11-01-0010.md, 20:2x UTC); the builders drop an episode with a call whose
                    arguments are not a JSON object, which the chat template cannot render. A write's schema
                    rejection for a missing or unexpected key ({"status": "rejected", "error": "missing [...] /
                    unexpected [...]"}), which form_errors.py also counts as bad_args, is not among them: both
                    harnesses log it under constraint_blocks, so this stage never counts it
                    (research/log/2026-11-01-0010.md, 21:4x UTC; none of M1's training episodes has one);
  7 efficient       at most twice the median steps of the episodes of the same source and template that passed
                    stages 1-6 (a fixed percentile would always drop a share, and drop the harder instances first);
  8 judged          for sources with a judge file (judge_sft.py): the judge found the method consistent with the
                    task's reference solution.
Task stages:
  9 decontaminated  the task shares no entity with an overlapping date window with any dev, test or external task
                    (market tasks only; the wealth tasks' database instances are disjoint by design), except the
                    held-out tasks --heldout-exempt names;
 10 base_screened   a task the start model solved in both screening runs (screen_base.sh) is kept with probability
                    --keep-solved-share (stratified by template, seeded); tasks it failed at least once are kept;
 11 max_per_task    at most --max-per-task episodes per (source, task), the fewest steps first.

Sources are given as NAME:FORMAT:RUN_DIR:TASKS, FORMAT v2 or v3, TASKS one or more task files joined by '+'; a run
directory's episodes on tasks outside its task files are left out before stage 1 and counted per source in the funnel
(outside_task_files), so a task file can select part of a run, such as some of its templates. The output
is a keep list (one JSON line per kept episode: source, task_id, run) for build_agent_sft.py / build_agent_sft_v3.py
--keep, and <out>.funnel.json with the counts per stage, source and template.

Usage: python -B filter_sft.py --source market:v3:<run dir>:<tasks.jsonl> [--source ...] --out <keep.jsonl>
       [--screen <screen run dir>[,...]] [--judge <judge.jsonl>[,...]] [--heldout <tasks.jsonl>[,...]]
       [--heldout-exempt <task id>[,...]]
       [--keep-solved-share 0.10] [--max-per-task 1] [--seed 20261031]
"""
import argparse
import collections
import json
import os
import random
import re
import statistics

MAX_REDOS = 2  # both harness versions: a finish rejected twice is accepted the third time unverified
STAGES = ["passed", "clean_end", "not_truncated", "grounded", "no_loop", "few_tool_errors", "efficient", "judged",
          "decontaminated", "base_screened", "max_per_task"]


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="NAME:FORMAT:RUN_DIR:TASKS[+TASKS]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--screen", default="")
    ap.add_argument("--judge", default="")
    ap.add_argument("--heldout", default="")
    ap.add_argument("--heldout-exempt", default="", help="held-out task ids not used for decontamination")
    ap.add_argument("--keep-solved-share", type=float, default=0.10)
    ap.add_argument("--max-per-task", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261031)
    return ap.parse_args(argv)


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:  # a line still being written
                continue


def last_rows(path):
    d = {}
    for r in jsonl(path):
        d[(r["task_id"], r["run"])] = r
    return d


def calls_of(msgs):
    out = []
    for m in msgs:
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                out.append((c["function"]["name"], c["function"]["arguments"] or ""))
    return out


def episode_checks(r, msgs, fmt):
    """Stages 1-6 for one episode: the name of the first stage it fails, or None."""
    if not r["passed"]:
        return "passed"
    if r["end"] != "finished" or r.get("api_error") or r.get("internal_errors") or r.get("internal_error"):
        return "clean_end"
    if fmt == "v3" and r.get("hit_max_tokens"):
        return "not_truncated"
    if r.get("verification_rejects", 0) >= MAX_REDOS:
        return "grounded"
    if r.get("loop_hints") or max(collections.Counter(calls_of(msgs)).values(), default=0) >= 3:
        return "no_loop"
    if r.get("tool_errors", 0) > 2:
        return "few_tool_errors"
    return None


# -- decontamination ---------------------------------------------------------------------------------------------
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def footprint(t):
    """[(entities, first date, last date), ...] a market task touches. Generated tasks give theirs in meta; external
    (FinSearchComp) tasks name their snapshot rows in meta.sources, as "SEC <tag> cik=<n> end=<d> start=<d|None> ...",
    "Yahoo <ticker> daily bars <d>, <d>" (or <d>..<d>) and "FRED <series> <d>[..<d>] value_vintage (cutoff ...)";
    the world-knowledge lines appended to some are skipped."""
    m = t.get("meta") or {}
    if m.get("entities"):
        return [(set(m["entities"]), m.get("start"), m.get("end"))]
    out = []
    for s in m.get("sources") or []:
        s = s.split("(cutoff")[0]
        w = s.split()
        if not w or w[0] not in ("SEC", "Yahoo", "FRED") or len(w) < 2:
            continue
        if w[0] == "SEC":
            cik = re.search(r"cik=(\d+)", s)
            ents = {f"CIK:{cik.group(1)}"} if cik else set()
            dates = [d for k in ("start", "end") for d in re.findall(rf"{k}=(\d{{4}}-\d{{2}}-\d{{2}})", s)]
        else:
            ents, dates = {w[1]}, DATE.findall(s)
        if ents:
            out.append((ents, min(dates) if dates else None, max(dates) if dates else None))
    return out


def overlaps(fa, fb):
    """Some footprint of one shares an entity and an overlapping date window with some footprint of the other."""
    for ea, sa, xa in fa:
        for eb, sb, xb in fb:
            if not ea & eb:
                continue
            if not (sa and xa and sb and xb):
                return True  # an undated footprint on a shared entity counts as overlapping
            if sa <= xb and sb <= xa:
                return True
    return False


def main(argv=None):
    a = parse(argv)
    rng = random.Random(a.seed)
    funnel = collections.defaultdict(collections.Counter)  # (source, template) -> stage -> kept count
    survivors = []  # (source, task_id, run, template, steps)
    outside = collections.Counter()  # source -> episodes on tasks outside its task files
    exempt = set(filter(None, a.heldout_exempt.split(",")))
    held_tasks = [t for p in a.heldout.split(",") if p for t in jsonl(p)]
    missing = exempt - {t["task_id"] for t in held_tasks}
    if missing:
        raise SystemExit(f"--heldout-exempt names tasks that are not held out: {sorted(missing)}")
    held = [footprint(t) for t in held_tasks if t["task_id"] not in exempt]
    judged = {}
    for p in a.judge.split(","):
        if p:
            for j in jsonl(p):
                judged[(j["source"], j["task_id"], j["run"])] = j["verdict"] == "ok"
    judged_sources = {k[0] for k in judged}
    screen = collections.defaultdict(list)
    for p in a.screen.split(","):
        if p:
            for r in last_rows(os.path.join(p, "results.jsonl")).values():
                screen[r["task_id"]].append(bool(r["passed"]))

    for spec in a.source:
        name, fmt, run_dir, task_files = spec.split(":", 3)
        tasks = {t["task_id"]: t for p in task_files.split("+") for t in jsonl(p)}
        res = last_rows(os.path.join(run_dir, "results.jsonl"))
        traj = {(t["task_id"], t["run"]): t["messages"] for t in jsonl(os.path.join(run_dir, "trajectories.jsonl"))}
        stage16 = []
        for k, r in sorted(res.items()):
            if k[0] not in tasks:
                outside[name] += 1
                continue
            tmpl = r.get("template") or tasks[k[0]].get("template") or "-"
            funnel[(name, tmpl)]["collected"] += 1
            if k not in traj:
                continue
            failed = episode_checks(r, traj[k], fmt)
            for s in STAGES[:6]:
                if failed == s:
                    break
                funnel[(name, tmpl)][s] += 1
            if failed is None:
                stage16.append((k, tmpl, r["steps"]))
        cap = {}
        by_t = collections.defaultdict(list)
        for _, tmpl, steps in stage16:
            by_t[tmpl].append(steps)
        for tmpl, v in by_t.items():
            cap[tmpl] = 2 * statistics.median(v)
        for k, tmpl, steps in stage16:
            if steps > cap[tmpl]:
                continue
            funnel[(name, tmpl)]["efficient"] += 1
            if name in judged_sources and not judged.get((name,) + k, False):
                continue
            funnel[(name, tmpl)]["judged"] += 1
            t = tasks.get(k[0])
            if held and t is not None and t.get("env") == "market" and any(overlaps(footprint(t), h) for h in held):
                continue
            funnel[(name, tmpl)]["decontaminated"] += 1
            survivors.append((name, k[0], k[1], tmpl, steps))

    # 10: base screening, stratified by template among the tasks the start model solved twice
    solved = sorted({(s, tid, tmpl) for s, tid, _, tmpl, _ in survivors if screen.get(tid) and len(screen[tid]) >= 2
                     and all(screen[tid])})
    keep_solved = set()
    by_t = collections.defaultdict(list)
    for s, tid, tmpl in solved:
        by_t[(s, tmpl)].append(tid)
    for key in sorted(by_t):
        ids = by_t[key]
        n = round(a.keep_solved_share * len(ids))
        keep_solved |= {(key[0], tid) for tid in rng.sample(ids, n)}
    solved_ids = {(s, tid) for s, tid, _ in solved}
    survivors = [x for x in survivors if (x[0], x[1]) not in solved_ids or (x[0], x[1]) in keep_solved]
    for s, tid, _, tmpl, _ in survivors:
        funnel[(s, tmpl)]["base_screened"] += 1
    # 11: at most --max-per-task per (source, task), the fewest steps first
    per = collections.defaultdict(list)
    for x in survivors:
        per[(x[0], x[1])].append(x)
    kept = []
    for key in sorted(per):
        for x in sorted(per[key], key=lambda x: (x[4], x[2]))[:a.max_per_task]:
            kept.append(x)
            funnel[(x[0], x[3])]["max_per_task"] += 1
    with open(a.out, "w", encoding="utf-8") as f:
        for s, tid, run, _, _ in kept:
            f.write(json.dumps({"source": s, "task_id": tid, "run": run}) + "\n")
    total = collections.defaultdict(collections.Counter)
    for (s, _), c in funnel.items():
        total[s].update(c)
    out = {"stages": ["collected"] + STAGES, "args": vars(a),
           "by_source": {s: {st: c[st] for st in ["collected"] + STAGES} for s, c in sorted(total.items())},
           "by_template": {f"{s}/{t}": {st: c[st] for st in ["collected"] + STAGES} for (s, t), c in sorted(funnel.items())},
           "outside_task_files": dict(outside),
           "screened_tasks": len(screen), "base_solved_twice": len(solved), "base_solved_kept": len(keep_solved)}
    json.dump(out, open(a.out + ".funnel.json", "w"), indent=1)
    print(json.dumps(out["by_source"]))


if __name__ == "__main__":
    main()
