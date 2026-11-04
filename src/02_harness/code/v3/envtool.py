"""Environment tools of harness v3: make and check task files, build a database instance, prepare DocFinQA.

Usage: python -B envtool.py <command> <arguments>
  generate-db <out.sqlite> <prefix A|B|C> <seed> [--customers 2000] [--products 300]
  make-tasks <instance.sqlite> <split> <n> <seed> <out.jsonl>
  make-tasks --template <t4_rebalance_one|t5_review_switch> <instance.sqlite> <split> <n> <seed> <out.jsonl>
  check-tasks <instance.sqlite> <tasks.jsonl>                    (exits 1 when a task fails)
  verify-composite <instance.sqlite> <tasks_t5.jsonl>            (exits 1 when a check fails)
  make-gate <instance.sqlite> <split> <tasks_docfinqa.jsonl> <docs dir> <n> <seed> <out.jsonl>
  verify-gate <instance.sqlite> <tasks_t6.jsonl>                 (exits 1 when a check fails)
  make-ml <split> <n> <seed> <out.jsonl>
  prepare-docfinqa <raw dev.json or test.json> <split> <out dir>
  check-market <tasks.jsonl> [...]                               (grade each market task's own answer_text;
                                                                  exits 1 when one fails)
"""
import collections
import json
import sys


def main(argv):
    cmd, args = argv[0], argv[1:]
    if cmd == "generate-db":
        from harness.envs.wealth import generate_db
        generate_db.main(args)
    elif cmd == "make-tasks":
        from harness.envs.wealth import tasks
        tasks.main(args)
    elif cmd == "check-tasks":
        from harness.envs.wealth import checks
        return 0 if checks.check_tasks(*args[:2]) else 1
    elif cmd == "verify-composite":
        from harness.envs.wealth import checks
        return 0 if checks.verify_composite(*args[:2]) else 1
    elif cmd == "make-gate":
        from harness.envs import report_gate
        db_path, split, doc_tasks, docs_dir, n, seed, out_path = args[:7]
        made = report_gate.make(db_path, split, doc_tasks, docs_dir, int(n), int(seed))
        with open(out_path, "w", encoding="utf-8") as f:
            for x in made:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(json.dumps({"tasks": len(made), "by_case": collections.Counter(x["case"] for x in made),
                          "filings": len({x["doc"] for x in made})}))
    elif cmd == "verify-gate":
        from harness.envs import report_gate
        return 0 if report_gate.verify(*args[:2]) else 1
    elif cmd == "make-ml":
        from harness.envs import ml
        ml.make(args[0], int(args[1]), int(args[2]), args[3])
    elif cmd == "prepare-docfinqa":
        from harness.envs import docfinqa
        docfinqa.prepare(*args[:3])
    elif cmd == "check-market":
        from harness.envs import market
        failed = 0
        for path in args:
            ok, bad = market.check_answers(path)
            failed += len(bad)
            print(json.dumps({"tasks": path, "passed": ok, "failed": len(bad), "failures": bad[:20]}, ensure_ascii=False))
        return 1 if failed else 0
    else:
        sys.exit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) if len(sys.argv) > 1 else __doc__)
