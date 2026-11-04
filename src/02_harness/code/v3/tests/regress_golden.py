"""Regression check: the reference policy graded under v3 passes and fails exactly the tasks it does under v2.

For each suite, golden.py of v3 runs on the task file into <out>/<suite> and its rows are compared, task by task, with
version 2's rows of the same tasks (a results.jsonl written by v2's golden.py). Reported per suite: tasks, passes under
each version, tasks whose pass/fail differs (must be 0), and tasks whose final answer text differs (the policies are
the same, so 0 is expected but only pass/fail is required).

The v2 rows of the wealth test split and of t5 dev and test are made first with v2's golden.py into <v2 dir>/test_H1,
test_H0 (--config H0), t5_dev_H1 and t5_test_H1; the others are the stored v2 reference runs.

Usage: python -B tests/regress_golden.py <out dir> <v2 dir>
"""
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("FP_DATA_EN", "<FP_DATA_EN unset>") + "/harness"
H = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/harness_v2"


def suites(v2_scratch):
    return [  # name, tasks, db, config, v2 results
        ("wealth_test_H1", f"{D}/tasks_test.jsonl", f"{D}/C.sqlite", "H1", f"{v2_scratch}/test_H1/results.jsonl"),
        ("wealth_test_H0", f"{D}/tasks_test.jsonl", f"{D}/C.sqlite", "H0", f"{v2_scratch}/test_H0/results.jsonl"),
        ("t4m_pilot_H1", f"{H}/pilot/tasks_t4m.jsonl", f"{D}/A.sqlite", "H1", f"{H}/pilot/golden_t4m/results.jsonl"),
        ("t5_train_H1", f"{H}/composite/tasks_t5_train.jsonl", f"{D}/A.sqlite", "H1", f"{H}/composite/golden_t5_train/results.jsonl"),
        ("t5_collect_H1", f"{H}/collect_t5/golden_tasks.jsonl", f"{D}/A.sqlite", "H1", f"{H}/collect_t5/golden/results.jsonl"),
        ("t5_dev_H1", f"{H}/composite/tasks_t5_dev.jsonl", f"{D}/B.sqlite", "H1", f"{v2_scratch}/t5_dev_H1/results.jsonl"),
        ("t5_test_H1", f"{H}/composite/tasks_t5_test.jsonl", f"{D}/C.sqlite", "H1", f"{v2_scratch}/t5_test_H1/results.jsonl"),
    ]


def rows(path):
    return {r["task_id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def main(out, v2_scratch):
    report, ok = {}, True
    for name, tasks, db, config, v2 in suites(v2_scratch):
        o = os.path.join(out, name)
        subprocess.run([sys.executable, "-B", os.path.join(HERE, "golden.py"), "--tasks", tasks, "--db", db, "--out", o,
                        "--config", config], check=True)
        a, b = rows(v2), rows(os.path.join(o, "results.jsonl"))
        common = sorted(set(a) & set(b))
        differ = [t for t in common if a[t]["passed"] != b[t]["passed"]]
        report[name] = {"v2_rows": len(a), "v3_rows": len(b), "compared": len(common),
                        "v2_passed": sum(a[t]["passed"] for t in common), "v3_passed": sum(b[t]["passed"] for t in common),
                        "pass_fail_differs": len(differ), "answer_differs": sum(a[t]["answer"] != b[t]["answer"] for t in common),
                        "only_v2": len(set(a) - set(b)), "only_v3": len(set(b) - set(a)),
                        "v3_ends": dict(collections.Counter(b[t]["end"] for t in common)),
                        "v3_max_task_steps": max(b[t]["task_steps"] for t in common)}
        ok &= not differ and len(common) == len(a) == len(b)
        print(name, json.dumps(report[name]), flush=True)
    print(json.dumps({"all_suites_agree": ok}))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
