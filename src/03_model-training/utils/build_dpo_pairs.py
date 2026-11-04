"""DPO preference pairs from harness v3 trajectories of the 2,350 market training tasks, from M1's two screening runs
per task and the teacher's runs (research/log/2026-11-01-0010.md, 11-02 09:4x UTC).

Two subsets, written apart so that training can take A alone or A + B:
  onpolicy (A)  tasks M1 passed in exactly one of its two runs (screen_m1_train): chosen is the passed run, rejected
                the failed one;
  teacher (B)   tasks M1 failed at least once: chosen is a passed teacher run of the same task (train_q38_H1 for a
                tier-1 task, train_hard_q38_H1 for a hard-tier one), rejected an M1 failed run, the one with fewer
                steps when both are usable.
A chosen run must pass filter_sft.py's trajectory stages 1-7: episode_checks, then the cap of twice the median steps
of its source's episodes of the same template that pass stages 1-6, over the same task files as M1's filter run, so
the caps are M1's. The judge (stage 8) was run on teacher episodes only: a teacher chosen run is taken from those it
found ok, else (--teacher-judge prefer) from the other runs passing stages 1-7, the fewest steps first as stage 11
orders them; an M1 chosen run records its verdict as "not_run". A rejected run must be a genuine failure, a wrong
answer or a loop at a normal finish, or the step limit. A run that ended on context overflow, an API or harness
error, or that has a reply cut at max_tokens is dropped and counted by reason (--keep-cut-rejected keeps the last
kind): its failure may come from the length or the server rather than from the model's choices, and DPO would push
down whatever the run did.

Both sides are rendered with build_agent_sft_v3.py's Encoder (the keep-thinking template, the trajectory's own tools)
and get that builder's labels: the assistant turns' tokens through <|im_end|>, less the turns that drew a nudge, had a
call the harness rejected or got a form error back (form_errors.mask_form_errors). On the rejected side the turns with
a rejected call or a form error keep their labels too, since they are among the actions DPO should push down
(research/log/2026-11-01-0010.md, 11-02 18:0x UTC); --mask-rejected-calls masks them as on the chosen side, as the
first build did. System, user and tool turns carry no label, so no tool result is trained. The two sides must render
the same prompt, the system turn with the tools and the user turn through the first <|im_start|>assistant\\n
(prompt_len tokens), which is checked. A pair is dropped when its task fails filter_sft.py's stage 9 test against
M1's held-out files (--keep-heldout-overlap keeps it, as the first build did, recording heldout_overlap), when a side
is longer than --cutoff tokens, has no labelled token, or has a call whose arguments are not a JSON object (the
template cannot render it).

Output per subset under --out: <subset>/, a DatasetDict (split train, one row per pair, columns in COLUMNS);
<subset>.messages.jsonl, the same pairs as messages with each assistant turn's label state, for reading; and
<subset>.manifest.json with the funnel, token counts, the template mix, the inputs' sha256 and the command.

Usage: python -B build_dpo_pairs.py --out <dir> [--tokenizer <dir>] [--template <jinja>] [--cutoff 32768]
       [--teacher-judge prefer|require] [--keep-cut-rejected] [--keep-heldout-overlap] [--mask-rejected-calls]
       (the run directories, task files, judge and held-out files default to the ones M1 and its screening used)
"""
import argparse
import collections
import hashlib
import json
import os
import statistics
import sys

import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "code"))
from build_agent_sft_v3 import IGNORE, TEMPLATE, Encoder, norm, trained_turns  # noqa: E402
from filter_sft import episode_checks, footprint, overlaps  # noqa: E402
from form_errors import call_args, mask_form_errors  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>")
MARKET = f"{DATA}/market_v1"
TASKS = f"{MARKET}/tasks"
# M1_keep.jsonl.funnel.json's args: the held-out files and the one exempt task of M1's stage 9
HELDOUT = ",".join([f"{TASKS}/market_tasks_dev.jsonl", f"{TASKS}/market_tasks_test.jsonl",
                    f"{TASKS}/market_tasks_hard_dev.jsonl", f"{TASKS}/market_tasks_hard_test.jsonl",
                    f"{MARKET}/external/finsearchcomp_ext_test.jsonl", f"{MARKET}/hardset_v1/dev_market.jsonl",
                    f"{MARKET}/hardset_v1/test_market.jsonl", f"{MARKET}/hardset_v1/test_finsearchcomp.jsonl"])
EXEMPT = "fsc-(T3)Complex_Historical_Investigation_023"
# (end, failure) of a run that failed by its own choices; harness v3's categorize() gives the failure
GENUINE = {("finished", "wrong_answer"), ("finished", "loop"), ("step_limit", "step_limit")}
INT = Sequence(Value("int32"))
COLUMNS = {"pair_id": Value("string"), "subset": Value("string"), "task_id": Value("string"),
           "template": Value("string"), "tier": Value("string"),
           "chosen_source": Value("string"), "chosen_model": Value("string"), "chosen_run": Value("int32"),
           "chosen_steps": Value("int32"), "chosen_judge": Value("string"), "chosen_pick": Value("string"),
           "rejected_source": Value("string"), "rejected_model": Value("string"), "rejected_run": Value("int32"),
           "rejected_steps": Value("int32"), "rejected_end": Value("string"), "rejected_failure": Value("string"),
           "rejected_pick": Value("string"), "heldout_overlap": Value("bool"),
           "prompt_len": Value("int32"), "chosen_len": Value("int32"), "rejected_len": Value("int32"),
           "chosen_loss_tokens": Value("int32"), "rejected_loss_tokens": Value("int32"),
           "chosen_input_ids": INT, "chosen_labels": INT, "rejected_input_ids": INT, "rejected_labels": INT}


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{DATA}/dpo")
    ap.add_argument("--tasks", default=f"{MARKET}/hardset_v1/train_market.jsonl", help="the tasks pairs are built on")
    ap.add_argument("--m1", default=f"{MARKET}/runs/screen_m1_train", help="M1's run directory, 2 runs per task")
    ap.add_argument("--teacher", action="append",
                    help="NAME:RUN_DIR:TASKS[+TASKS] as filter_sft.py's --source names them for the judge files; "
                         "TASKS sets the episodes stage 7's median is taken over (default: M1's two market sources)")
    ap.add_argument("--judge", default=f"{MARKET}/runs/judge_market_train.jsonl,{MARKET}/runs/judge_hard_train.jsonl")
    ap.add_argument("--heldout", default=HELDOUT)
    ap.add_argument("--heldout-exempt", default=EXEMPT)
    ap.add_argument("--tokenizer", default=os.environ.get("FP_MODELS_D2", "<FP_MODELS_D2 unset>") + "/Qwen3.5-4B-Base")
    ap.add_argument("--template", default=os.path.normpath(TEMPLATE))
    ap.add_argument("--cutoff", type=int, default=32768)
    ap.add_argument("--teacher-judge", choices=["prefer", "require"], default="prefer",
                    help="prefer: a teacher run the judge found ok, else another passing stages 1-7; require: only ok")
    ap.add_argument("--keep-cut-rejected", action="store_true",
                    help="keep rejected runs that have a reply cut at max_tokens")
    ap.add_argument("--keep-heldout-overlap", action="store_true",
                    help="keep pairs whose task overlaps the held-out files (the first build's behaviour)")
    ap.add_argument("--mask-rejected-calls", action="store_true",
                    help="no label on the rejected side's rejected-call and form-error turns (the first build's)")
    a = ap.parse_args(argv)
    a.teacher = a.teacher or [f"market:{MARKET}/runs/train_q38_H1:{TASKS}/market_tasks_train_shuffled.jsonl",
                              f"hard:{MARKET}/runs/train_hard_q38_H1:{MARKET}/hardset_v1/train_market.jsonl"]
    return a


def lines(path, digests):
    """The JSON lines of path, read once; the file's sha256 is put in digests[path] when they are used up."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for line in f:
            h.update(line)
            yield json.loads(line)
    digests[path] = h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(run_dir, tasks, wanted, digests):
    """The last row of each (task, run) on tasks, the first filter stage of 1-7 it fails (None when it passes all),
    and the trajectories of the tasks in wanted.

    Stage 7 needs the steps of every episode of the run directory that passes stages 1-6, so all of them are checked,
    in one pass over the trajectory file; filter_sft.main computes the cap the same way.
    """
    rows = {}
    for r in lines(f"{run_dir}/results.jsonl", digests):
        if r["task_id"] in tasks:
            rows[(r["task_id"], r["run"])] = r
    stage, traj = {}, {}
    for t in lines(f"{run_dir}/trajectories.jsonl", digests):
        k = (t["task_id"], t["run"])
        if k in rows:
            stage[k] = episode_checks(rows[k], t["messages"], "v3")
            if k[0] in wanted:
                traj[k] = t
    tmpl = {k: r.get("template") or tasks[k[0]].get("template") or "-" for k, r in rows.items()}
    steps = collections.defaultdict(list)
    for k, s in stage.items():
        if s is None:
            steps[tmpl[k]].append(rows[k]["steps"])
    cap = {t: 2 * statistics.median(v) for t, v in steps.items()}
    for k, s in stage.items():
        if s is None and rows[k]["steps"] > cap[tmpl[k]]:
            stage[k] = "efficient"
    for k in rows:
        stage.setdefault(k, "no trajectory")
    return rows, stage, traj, cap


def not_genuine(r, keep_cut):
    """None when a failed run counts as a genuine failure, else why it does not."""
    if r.get("api_error") or (r["end"], r["failure"]) not in GENUINE:
        return f"ended on {r['end']}"
    if r.get("internal_errors") or r.get("internal_error"):
        return "harness error in the episode"
    if r.get("hit_max_tokens") and not keep_cut:
        return "a reply cut at max_tokens"
    return None


def render(enc, t, cutoff, train_bad=False):
    """One side of a pair: (ids, labels, label state per assistant turn), or the reason it cannot be used. train_bad
    keeps the labels of the turns with a rejected call or a form error (the rejected side's), marked "+loss"."""
    msgs = t["messages"]
    if any(call_args(c) is None for m in msgs for c in m.get("tool_calls") or []):
        return "arguments not a JSON object"
    accepted = trained_turns(msgs)
    train, _ = mask_form_errors(msgs, accepted)
    asst = [m for m in msgs if m["role"] == "assistant"]
    state = ["trained" if on else "form_error" if ok else "rejected_call" if m.get("tool_calls") else "nudged"
             for m, ok, on in zip(asst, accepted, train)]
    if train_bad:
        train = [on or s != "nudged" for on, s in zip(train, state)]
        state = [s + " +loss" if s in ("form_error", "rejected_call") else s for s in state]
    ids, labels, n_asst = enc([norm(m) for m in msgs], t["tools"], train)
    if n_asst != len(train):
        raise SystemExit(f"{t['task_id']} run {t['run']}: {n_asst} assistant turns rendered, {len(train)} in the "
                         "messages")
    if len(ids) > cutoff:
        return f"longer than {cutoff}"
    if not any(train):
        return "no turn with a loss"
    return ids, labels, state


def prompt_len(enc, ids):
    """Tokens before the first assistant turn's content: system (with the tools), user, <|im_start|>assistant\\n."""
    n = len(enc.assistant)
    for i, x in enumerate(ids):
        if x == enc.im_start and ids[i + 1:i + 1 + n] == enc.assistant:
            return i + 1 + n
    raise SystemExit("no assistant turn rendered")


def view(t, state):
    """A trajectory after its system and user turns, for reading, with each assistant turn's label state."""
    out, k = [], 0
    for m in t["messages"][2:]:
        m = {key: v for key, v in m.items() if not key.startswith("_")}
        if m["role"] == "assistant":
            m["label"] = state[k]
            k += 1
        out.append(m)
    return out


def stats(v):
    v = sorted(v)
    return {"n": len(v), "total": sum(v), "min": v[0], "median": statistics.median(v),
            "p90": v[int(0.9 * (len(v) - 1))], "max": v[-1]} if v else {"n": 0}


def main(argv=None):
    a = parse(argv)
    digests = {}
    tasks = {t["task_id"]: t for t in lines(a.tasks, digests)}
    m1_rows = {}
    for r in lines(f"{a.m1}/results.jsonl", {}):  # read again inside load(), which records its sha256
        if r["task_id"] in tasks:
            m1_rows.setdefault(r["task_id"], {})[r["run"]] = r
    runs_ok = {tid: {k: bool(r["passed"]) for k, r in v.items()} for tid, v in m1_rows.items()}
    if any(len(v) != 2 for v in runs_ok.values()) or len(runs_ok) != len(tasks):
        raise SystemExit(f"{a.m1}: expected 2 runs on each of the {len(tasks)} tasks")
    task_a = sorted(tid for tid, v in runs_ok.items() if sum(v.values()) == 1)
    task_b = sorted(tid for tid, v in runs_ok.items() if sum(v.values()) < 2)
    wanted = set(task_b)
    m1, m1_stage, m1_traj, m1_cap = load(a.m1, tasks, wanted, digests)

    teacher, judged = {}, {}
    for spec in a.teacher:
        name, run_dir, task_files = spec.split(":", 2)
        src_tasks = {t["task_id"]: t for p in task_files.split("+") for t in lines(p, digests)}
        rows, stage, traj, cap = load(run_dir, src_tasks, wanted, digests)
        teacher[name] = {"dir": run_dir, "rows": rows, "stage": stage, "traj": traj, "cap": cap}
    for p in filter(None, a.judge.split(",")):
        for j in lines(p, digests):
            judged[(j["source"], j["task_id"], j["run"])] = j["verdict"]
    exempt = set(filter(None, a.heldout_exempt.split(",")))
    held = [footprint(t) for p in filter(None, a.heldout.split(",")) for t in lines(p, digests)
            if t["task_id"] not in exempt]

    def overlap(tid):  # filter_sft.py's stage 9
        t = tasks[tid]
        return bool(held) and t.get("env") == "market" and any(overlaps(footprint(t), h) for h in held)

    def teacher_source(tid):
        hits = [n for n, s in teacher.items() if any(k[0] == tid for k in s["rows"])]
        if len(hits) != 1:
            raise SystemExit(f"{tid}: in {len(hits)} teacher run directories, expected 1")
        return hits[0]

    enc = Encoder(a.tokenizer, a.template)
    m1_name = os.path.basename(a.m1.rstrip("/"))
    out = {"onpolicy": [], "teacher": []}
    funnel = {"onpolicy": collections.Counter(), "teacher": collections.Counter()}
    detail = {"onpolicy": collections.Counter(), "teacher": collections.Counter()}

    def pair(subset, tid, c, r, ctx):
        """Render both sides; append the pair, or count why it is dropped."""
        if not a.keep_heldout_overlap and overlap(tid):
            funnel[subset]["dropped: task overlaps the held-out files"] += 1
            return
        sides = {}
        for side, t in (("chosen", c), ("rejected", r)):
            x = render(enc, t, a.cutoff, train_bad=side == "rejected" and not a.mask_rejected_calls)
            if isinstance(x, str):
                funnel[subset][f"dropped: {side} {x}"] += 1
                return
            sides[side] = x
        (ci, cl, cs), (ri, rl, rs) = sides["chosen"], sides["rejected"]
        n = prompt_len(enc, ci)
        if prompt_len(enc, ri) != n or ci[:n] != ri[:n]:
            raise SystemExit(f"{subset} {tid}: chosen and rejected render different prompts")
        if any(x != IGNORE for x in cl[:n] + rl[:n]) or any(x not in (IGNORE, i) for x, i in zip(cl + rl, ci + ri)):
            raise SystemExit(f"{subset} {tid}: a label outside the assistant turns or not its token")
        funnel[subset]["kept"] += 1
        row = dict(ctx, pair_id=f"{subset}:{tid}", subset=subset, task_id=tid, template=tasks[tid]["template"],
                   tier="hard" if tid.startswith("mk-hard-") else "tier1", heldout_overlap=overlap(tid),
                   prompt_len=n, chosen_len=len(ci), rejected_len=len(ri),
                   chosen_loss_tokens=sum(x != IGNORE for x in cl), rejected_loss_tokens=sum(x != IGNORE for x in rl),
                   chosen_input_ids=np.asarray(ci, np.int32), chosen_labels=np.asarray(cl, np.int32),
                   rejected_input_ids=np.asarray(ri, np.int32), rejected_labels=np.asarray(rl, np.int32))
        msgs = c["messages"]
        row["_view"] = {"pair_id": row["pair_id"], "task_id": tid, "template": row["template"], "prompt_len": n,
                        "system": msgs[0]["content"], "user": msgs[1]["content"],
                        "tools": [x["function"]["name"] for x in c["tools"]],
                        "chosen": {k[7:]: v for k, v in ctx.items() if k.startswith("chosen_")} | {
                            "turn_labels": collections.Counter(cs), "messages": view(c, cs)},
                        "rejected": {k[9:]: v for k, v in ctx.items() if k.startswith("rejected_")} | {
                            "turn_labels": collections.Counter(rs), "messages": view(r, rs)}}
        out[subset].append(row)

    def unusable(r):
        """Why an M1 failed run cannot be a rejected side, or None."""
        why = not_genuine(r, a.keep_cut_rejected)
        return why or (None if (r["task_id"], r["run"]) in m1_traj else "has no trajectory")

    def rejected_ctx(rr, pick):
        return {"rejected_source": m1_name, "rejected_model": rr["model"], "rejected_run": rr["run"],
                "rejected_steps": rr["steps"], "rejected_end": rr["end"], "rejected_failure": rr["failure"],
                "rejected_pick": pick}

    # A: M1's passed run against its failed run on the same task
    for tid in task_a:
        funnel["onpolicy"]["candidate tasks"] += 1
        ok_run = next(k for k, v in runs_ok[tid].items() if v)
        bad_run = next(k for k, v in runs_ok[tid].items() if not v)
        cr, rr = m1[(tid, ok_run)], m1[(tid, bad_run)]
        s = m1_stage[(tid, ok_run)]
        if s is not None:
            funnel["onpolicy"][f"dropped: chosen fails stage {s}"] += 1
            continue
        why = unusable(rr)
        if why:
            funnel["onpolicy"][f"dropped: rejected {why}"] += 1
            continue
        ctx = {"chosen_source": m1_name, "chosen_model": cr["model"], "chosen_run": ok_run, "chosen_steps": cr["steps"],
               "chosen_judge": "not_run", "chosen_pick": "the passed run (stages 1-7; no judge for M1)"}
        pair("onpolicy", tid, m1_traj[(tid, ok_run)], m1_traj[(tid, bad_run)],
             ctx | rejected_ctx(rr, "the failed run"))

    # B: a teacher run against an M1 failed run
    for tid in task_b:
        funnel["teacher"]["candidate tasks"] += 1
        failed = sorted((m1[(tid, k)] for k, v in runs_ok[tid].items() if not v), key=lambda r: (r["steps"], r["run"]))
        usable = [r for r in failed if not unusable(r)]
        if not usable:
            for r in failed:
                detail["teacher"][f"M1 failed run: {unusable(r)}"] += 1
            funnel["teacher"]["dropped: no genuine M1 failure"] += 1
            continue
        rr = usable[0]
        if len(failed) == 1:
            pick = "the failed run"
        elif len(usable) == 2:
            pick = f"fewer steps of the two failed runs (other: run {usable[1]['run']}, {usable[1]['steps']} steps)"
        else:
            other = next(r for r in failed if r is not rr)
            pick = f"the other failed run (run {other['run']}) {unusable(other)}"
        name = teacher_source(tid)
        src = teacher[name]
        runs = sorted((k for k in src["rows"] if k[0] == tid), key=lambda k: (src["rows"][k]["steps"], k[1]))
        passing = [k for k in runs if src["stage"][k] is None]
        ok = [k for k in passing if judged.get((name,) + k) == "ok"]
        if not passing:
            for k in runs:
                detail["teacher"][f"teacher run fails stage {src['stage'][k]}"] += 1
            funnel["teacher"]["dropped: no teacher run passes stages 1-7"] += 1
            continue
        if not ok and a.teacher_judge == "require":
            funnel["teacher"]["dropped: no teacher run the judge found ok"] += 1
            continue
        k = (ok or passing)[0]
        verdict = judged.get((name,) + k, "missing")
        how = "stages 1-8, judge ok" if ok else f"stages 1-7, judge {verdict} (no run with judge ok)"
        cr = src["rows"][k]
        ctx = {"chosen_source": os.path.basename(src["dir"].rstrip("/")), "chosen_model": cr["model"],
               "chosen_run": k[1], "chosen_steps": cr["steps"], "chosen_judge": verdict,
               "chosen_pick": f"{how}; fewest steps of {len(ok or passing)} such run(s) of {len(runs)}"}
        pair("teacher", tid, src["traj"][k], m1_traj[(tid, rr["run"])], ctx | rejected_ctx(rr, pick))

    os.makedirs(a.out, exist_ok=True)
    fixed = [a.template] + [os.path.join(a.tokenizer, f) for f in ("tokenizer.json", "tokenizer_config.json")]
    fixed += [os.path.join(HERE, "build_dpo_pairs.py")]
    fixed += [os.path.normpath(os.path.join(HERE, os.pardir, "code", f))
              for f in ("build_agent_sft_v3.py", "filter_sft.py", "form_errors.py")]
    for p in fixed:
        digests[p] = sha256(p)
    for subset, rows in out.items():
        base = os.path.join(a.out, subset)
        with open(base + ".messages.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.pop("_view"), ensure_ascii=False) + "\n")
        cols = {c: [row[c] for row in rows] for c in COLUMNS}
        DatasetDict({"train": Dataset.from_dict(cols, features=Features(COLUMNS))}).save_to_disk(base)
        manifest = {
            "subset": subset, "pairs": len(rows), "funnel": dict(funnel[subset]), "drop_detail": dict(detail[subset]),
            "tokens": {"prompt_len": stats([r["prompt_len"] for r in rows]),
                       "chosen_len": stats([r["chosen_len"] for r in rows]),
                       "rejected_len": stats([r["rejected_len"] for r in rows]),
                       "chosen_loss_tokens": stats([r["chosen_loss_tokens"] for r in rows]),
                       "rejected_loss_tokens": stats([r["rejected_loss_tokens"] for r in rows])},
            "template_mix": dict(sorted(collections.Counter(r["template"] for r in rows).items())),
            "tier_mix": dict(collections.Counter(r["tier"] for r in rows)),
            "chosen_source_mix": dict(collections.Counter(r["chosen_source"] for r in rows)),
            "chosen_judge_mix": dict(collections.Counter(r["chosen_judge"] for r in rows)),
            "rejected_failure_mix": dict(collections.Counter(f"{r['rejected_end']}/{r['rejected_failure']}"
                                                             for r in rows)),
            "rejected_pick_mix": dict(collections.Counter(r["rejected_pick"].split(" (")[0] for r in rows)),
            "heldout_overlap": sum(r["heldout_overlap"] for r in rows),
            "stage7_cap": {m1_name: m1_cap} | {n: s["cap"] for n, s in teacher.items()},
            "stage1_7_pass": {m1_name: sum(v is None for v in m1_stage.values())}
                             | {n: sum(v is None for v in s["stage"].values()) for n, s in teacher.items()},
            "stage1_8_pass": {n: sum(v is None and judged.get((n,) + k) == "ok" for k, v in s["stage"].items())
                              for n, s in teacher.items()},
            "columns": {c: str(t) for c, t in COLUMNS.items()},
            "args": vars(a), "inputs_sha256": digests,
            "command": " ".join([sys.executable, "-B"] + [os.path.abspath(sys.argv[0])] + sys.argv[1:])}
        json.dump(manifest, open(base + ".manifest.json", "w"), indent=1)
        print(json.dumps({k: manifest[k] for k in ("subset", "pairs", "funnel", "drop_detail", "chosen_judge_mix",
                                                   "heldout_overlap")}))


if __name__ == "__main__":
    main()
