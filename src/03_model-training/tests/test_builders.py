"""End-to-end tests of the agent-SFT builders' labels on synthetic trajectories (research/log/2026-11-01-0010.md, 20:2x
UTC): build_agent_sft_v3.py, imported, on a harness v3 run directory, build_agent_sft.py, run as a script on a v2
collect directory with harness_v2's code snapshot for the tools, and rl_data.py's render and pack, run as a script on a
v2 rollout directory, each with and without --train-form-errors. A turn's flag is read back from the packed output:
whether the tokens of its span, after <|im_start|>assistant\\n through its <|im_end|>, carry labels. A trajectory with
a call whose arguments are not a JSON object cannot be rendered (Qwen3.5's template takes the arguments' items), so
each builder is checked to drop it. build_agent_sft_v3.py's check of the --add-packed sets (21:4x UTC) is tested on
sets packed here, one of them given a manifest like M1_wa's.

Run: python -B src/03_model-training/tests/test_builders.py   (in the train-q35 environment: transformers, datasets and
LLaMA-Factory, with the Qwen3.5-4B-Base tokenizer and harness_v2's code snapshots readable; skipped without them)
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = os.environ.get("FP_MODELS_D2", "<FP_MODELS_D2 unset>") + "/Qwen3.5-4B-Base"
V2_CODE = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/harness_v2/code_collect"
# rl_data.py render imports report_gate, which only the t6 snapshot has
V2_CODE_T6 = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/harness_v2/code_collect_t6"
sys.path.insert(0, os.path.join(HERE, "code"))
try:
    import datasets
    import build_agent_sft_v3 as b3
    from form_errors import mask_form_errors
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(TOKENIZER)
    datasets.disable_progress_bars()
except Exception:  # not the training environment, or no tokenizer here
    b3 = TOK = None
CHECKLIST = "\n[checklist] 1. [ ] look up the close; 2. [ ] answer"


def schema(name, *params):
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {
        "type": "object", "properties": {p: {"type": "string"} for p in params}, "required": list(params)}}}


def call(i, name, args):
    """An assistant turn with one call; args is an object, or a string sent as the arguments as it is."""
    args = args if isinstance(args, str) else json.dumps(args)
    return {"role": "assistant", "content": "", "reasoning_content": f"step {i}",
            "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args}}]}


def result(i, text, name=None):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": text, **({"name": name} if name else {})}


def err(message, tool):
    return json.dumps({"error": message, "example_of_a_valid_call": {tool: {}}}) + CHECKLIST


def episode(steps, users=1):
    """System and user messages, then one assistant turn and its result per step (name, args, result text); a step
    without a name is a reply without a call, nudged when users > 1."""
    msgs = [{"role": "system", "content": "You answer with the tools."}, {"role": "user", "content": "What was the close?"}]
    for i, (name, args, text) in enumerate(steps):
        if name is None:
            msgs += [{"role": "assistant", "content": text, "reasoning_content": f"step {i}"}]
            if users > 1:
                msgs += [{"role": "user", "content": "Use the tools to act on the request, and call finish with the "
                                                     "final answer when done."}]
        else:
            msgs += [call(i, name, args), result(i, text, name)]
    return msgs


V3_TOOLS = [schema("get_prices", "symbol", "start", "end"), schema("run_python", "code"),
            schema("set_checklist", "items"), schema("finish", "answer")]
SANDBOX = ("the code raised an error:\nTraceback (most recent call last):\n  File \"<run_python>\", line 2, in <module>\n"
           "TypeError: Rolling.apply() got an unexpected keyword argument 'axis'")
# (name, args, result, loss by default, loss with --train-form-errors)
V3_STEPS = [
    ("set_checklook", {"items": ["a"]}, json.dumps({"error": "unknown tool set_checklook"}) + CHECKLIST, False, True),
    ("get_prices", {"symbol": "AAPL", "begin": "2020-01-02"},
     err("MarketEnv.get_prices() got an unexpected keyword argument 'begin'", "get_prices"), False, True),
    ("run_python", {"code": "s.rolling(5).apply(f, axis=1)"}, err(SANDBOX, "run_python"), True, True),
    ("get_prices", {"symbol": "NOPE", "start": "2020-01-02", "end": "2020-01-02"},
     err("unknown symbol 'NOPE' Find symbols with search_securities.", "get_prices"), True, True),
    ("get_prices", {"symbol": "AAPL", "start": "2020-01-02", "end": "2020-01-02"},
     json.dumps({"rows": [["2020-01-02", 123.45]], "note": "{\"error\": \"unknown tool x\"} is not this"}) + CHECKLIST,
     True, True),
    (None, None, "The close was 123.45.", False, False),  # nudged
    ("finish", {"answer": "123.45"}, json.dumps({"status": "rejected", "verification_failed": ["no checklist was written"]}),
     False, False),
    ("finish", {"answer": "123.45"}, json.dumps({"status": "finished"}), True, True)]
V3_PLAIN = [V3_STEPS[3], V3_STEPS[4], V3_STEPS[7]]  # no form error
NOT_JSON = json.dumps({"error": "arguments are not a JSON object"}) + CHECKLIST
# arguments that are not JSON, and JSON that is not an object: each trajectory is dropped
BAD_ARGS = [("get_prices", '{"symbol": "AAPL", "start": "2020-01-02"', NOT_JSON), ("get_prices", '["AAPL"]', NOT_JSON)]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def spans(ids):
    """(start, end) of each assistant turn's labelled tokens, as the builders' encoders find them."""
    im_start, im_end = TOK.convert_tokens_to_ids("<|im_start|>"), TOK.convert_tokens_to_ids("<|im_end|>")
    head = TOK("assistant\n", add_special_tokens=False).input_ids
    out, i = [], 0
    while i < len(ids):
        if ids[i] == im_start and ids[i + 1:i + 1 + len(head)] == head:
            j = i + 1 + len(head)
            while ids[j] != im_end:
                j += 1
            out.append((i + 1 + len(head), j + 1))
            i = j
        i += 1
    return out


def turn_flags(ids, labels):
    """Whether each assistant span carries a loss; a span is labelled whole (labels = ids) or not at all, and nothing
    outside the spans is labelled."""
    flags, inside = [], set()
    for s, e in spans(ids):
        on = [labels[q] != b3.IGNORE for q in range(s, e)]
        assert all(on) or not any(on), (s, e)
        if all(on):
            assert labels[s:e] == ids[s:e]
        flags.append(all(on))
        inside |= set(range(s, e))
    assert all(labels[q] == b3.IGNORE for q in range(len(ids)) if q not in inside)
    return flags


@unittest.skipIf(b3 is None, "transformers, datasets, LLaMA-Factory or the Qwen3.5 tokenizer is not available")
class TestBuilderV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="agent_sft_v3_test_")
        run = os.path.join(cls.tmp, "run")
        cls.msgs = {"t1": episode([s[:3] for s in V3_STEPS], users=2), "t2": episode([s[:3] for s in V3_PLAIN])}
        bad = {f"t{3 + i}": episode([b, V3_STEPS[7][:3]]) for i, b in enumerate(BAD_ARGS)}
        write_jsonl(f"{run}/results.jsonl", [{"task_id": t, "run": 1, "passed": True, "harness": "v3", "template": "x"}
                                             for t in {**cls.msgs, **bad}])
        write_jsonl(f"{run}/trajectories.jsonl", [{"task_id": t, "run": 1, "tools": V3_TOOLS, "messages": m}
                                                  for t, m in {**cls.msgs, **bad}.items()])
        cls.out = {}
        for name, extra in (("default", []), ("compat", ["--train-form-errors"])):
            out = os.path.join(cls.tmp, name)
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)  # the builder leaves its input files to the collector
                b3.main(["--runs", f"market={run}", "--tokenizer", TOKENIZER, "--out", out] + extra)
            with open(out + ".manifest.json", encoding="utf-8") as f:
                cls.out[name] = (b3.unpack(out), json.load(f))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            enc = b3.Encoder(TOKENIZER, b3.TEMPLATE)
        # the labels version 1 gave: the nudge and rejection rule alone
        cls.old = {t: enc([b3.norm(x) for x in m], V3_TOOLS, b3.trained_turns(m))[:2] for t, m in cls.msgs.items()}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def sample(self, name, task):
        ids = self.old[task][0]
        return next(s for s in self.out[name][0] if s[0] == ids)

    def test_form_error_turns_carry_no_loss(self):
        """Unknown tool and unexpected keyword: no loss; run_python's traceback, a ToolError and a result whose data
        mentions "unknown tool": a loss; the nudged reply and the rejected finish: no loss, as before."""
        self.assertEqual(turn_flags(*self.sample("default", "t1")), [s[3] for s in V3_STEPS])
        self.assertEqual(self.out["default"][1]["assistant_turns"]["masked for a form error"], 2)
        self.assertEqual(self.out["default"][1]["assistant_turns"]["masked"], 4)

    def test_compat_flag_gives_the_old_labels(self):
        self.assertEqual(turn_flags(*self.sample("compat", "t1")), [s[4] for s in V3_STEPS])
        for t in self.msgs:
            self.assertEqual(self.sample("compat", t), self.old[t])
        self.assertEqual(self.out["compat"][1]["assistant_turns"]["masked for a form error"], 0)
        self.assertEqual(self.out["compat"][1]["assistant_turns"]["masked"], 2)

    def test_only_the_form_error_spans_differ(self):
        ids, new = self.sample("default", "t1")
        old = self.old["t1"][1]
        masked = [sp for sp, s in zip(spans(ids), V3_STEPS) if s[4] and not s[3]]
        differ = {q for q in range(len(ids)) if new[q] != old[q]}
        self.assertEqual(differ, {q for s, e in masked for q in range(s, e)})
        self.assertEqual(self.sample("default", "t2"), self.old["t2"])

    def test_arguments_not_a_json_object_drop_the_trajectory(self):
        for name in ("default", "compat"):
            with self.subTest(run=name):
                m = self.out[name][1]
                self.assertEqual((m["trajectories"], m["dropped"]), (2, {"arguments not a JSON object": 2}))


def write_v3_run(path, msgs):
    write_jsonl(f"{path}/results.jsonl", [{"task_id": t, "run": 1, "passed": True, "harness": "v3", "template": "x"}
                                          for t in msgs])
    write_jsonl(f"{path}/trajectories.jsonl", [{"task_id": t, "run": 1, "tools": V3_TOOLS, "messages": m}
                                               for t, m in msgs.items()])


@unittest.skipIf(b3 is None, "transformers, datasets, LLaMA-Factory or the Qwen3.5 tokenizer is not available")
class TestAddPackedGuard(unittest.TestCase):
    """build_agent_sft_v3.py --add-packed takes a packed set only when its manifest shows form-error masking, unless
    --allow-unmasked-packed or --train-form-errors is given, and records which added sets were masked. The sets to add
    are built from t1 (two form-error turns): "masked" by default, "prefix" with --train-form-errors and then given a
    manifest like M1_wa's, without train_form_errors in its args or a "masked for a form error" count."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="add_packed_guard_test_")
        write_v3_run(f"{cls.tmp}/src_run", {"t1": episode([s[:3] for s in V3_STEPS], users=2)})
        write_v3_run(f"{cls.tmp}/run", {"t2": episode([s[:3] for s in V3_PLAIN])})
        cls.masked, cls.prefix = f"{cls.tmp}/masked", f"{cls.tmp}/prefix"
        cls.build(cls.masked, "--runs", f"market={cls.tmp}/src_run")
        cls.build(cls.prefix, "--runs", f"market={cls.tmp}/src_run", "--train-form-errors")
        m = cls.read(cls.prefix + ".manifest.json")
        del m["args"]["train_form_errors"], m["assistant_turns"]["masked for a form error"]
        del m["add_packed_form_errors_masked"]
        cls.write(cls.prefix + ".manifest.json", m)
        cls.out = {name: cls.build(f"{cls.tmp}/{name}", "--add-packed", src, *extra) for name, src, extra in (
            ("allow", cls.prefix, ["--allow-unmasked-packed"]), ("compat", cls.prefix, ["--train-form-errors"]),
            ("with_masked", cls.masked, []))}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def build(cls, out, *args):
        """Run the builder (on the t2 run unless args name --runs); the output's samples and manifest."""
        runs = [] if "--runs" in args else ["--runs", f"market={cls.tmp}/run"]
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            b3.main([*runs, "--tokenizer", TOKENIZER, "--out", out, *args])
        return b3.unpack(out), cls.read(out + ".manifest.json")

    @staticmethod
    def read(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def write(path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def test_a_pre_fix_set_stops_the_build(self):
        out = os.path.join(self.tmp, "refused")
        r = subprocess.run([sys.executable, "-B", os.path.join(HERE, "code", "build_agent_sft_v3.py"), "--runs",
                            f"market={self.tmp}/run", "--tokenizer", TOKENIZER, "--out", out, "--add-packed",
                            self.prefix], capture_output=True, text=True, check=False,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(f"--add-packed {self.prefix}: its manifest's args have no train_form_errors", r.stderr)
        self.assertIn("--allow-unmasked-packed", r.stderr)
        self.assertFalse(os.path.exists(out) or os.path.exists(out + ".manifest.json"))

    def test_allowed_compat_and_masked_sets_are_added_unchanged(self):
        for name, src, masked in (("allow", self.prefix, False), ("compat", self.prefix, False),
                                  ("with_masked", self.masked, True)):
            with self.subTest(run=name):
                samples, manifest = self.out[name]
                self.assertEqual(manifest["add_packed_form_errors_masked"], {src: masked})
                self.assertEqual((manifest["trajectories"], manifest["added_from_packed"]), (1, 1))
                self.assertIn(b3.unpack(src)[0], samples)

    def test_unmasked_reason(self):
        """The test behind the guard on each kind of manifest, and through sets an added set took in."""
        self.assertIsNone(b3.unmasked_reason(self.masked))
        self.assertIsNone(b3.unmasked_reason(f"{self.tmp}/with_masked/"))
        cases = {"no_manifest": None, "compat_build": {"train_form_errors": True},
                 "opd": {"train_form_errors": False, "mode": "opd"},
                 # written before add_packed_form_errors_masked: the sets in args.add_packed are tested in turn
                 "before_record": {"train_form_errors": False, "add_packed": self.prefix},
                 "before_record_masked": {"train_form_errors": False, "add_packed": self.masked}}
        for name, args in cases.items():
            if args is not None:
                self.write(f"{self.tmp}/{name}.manifest.json", {"args": args})
        why = {name: b3.unmasked_reason(f"{self.tmp}/{name}") for name in cases}
        self.assertEqual(why, {
            "no_manifest": f"{self.tmp}/no_manifest.manifest.json does not exist",
            "compat_build": "it was built with --train-form-errors",
            "opd": "rl_data.py pack --mode opd keeps the form-error turns' labels",
            "before_record": f"it took in {self.prefix}, where its manifest's args have no train_form_errors, so it "
                             "was built before form-error masking",
            "before_record_masked": None})
        # a set built with --allow-unmasked-packed is refused in turn, by its record
        self.assertEqual(b3.unmasked_reason(f"{self.tmp}/allow"),
                         f"it took in {self.prefix} unmasked (--allow-unmasked-packed)")
        with self.assertRaises(SystemExit) as e:
            b3.main(["--runs", f"market={self.tmp}/run", "--tokenizer", TOKENIZER, "--out", f"{self.tmp}/refused2",
                     "--add-packed", f"{self.tmp}/allow"])
        self.assertIn(f"--add-packed {self.tmp}/allow: it took in {self.prefix} unmasked", str(e.exception.code))


# (name, args, result, loss by default, with --mask-rejected, with --train-form-errors, with both)
V2_STEPS = [
    ("redemption", {"customer_id": "A-C0001"}, json.dumps({"error": "unknown tool redemption"}) + CHECKLIST,
     False, False, True, True),
    ("calculator", {}, err("Env.calculator() missing 1 required positional argument: 'expressions'", "calculator"),
     False, False, True, True),
    ("get_fund", {"product_ids": ["A-P9999"]}, err("no product A-P9999", "get_fund"), True, True, True, True),
    ("get_document", {"doc_id": "A-P0001-2"},
     json.dumps({"doc_id": "A-P0001-2", "text": "A wrong name gets {\"error\": \"unknown tool\"} back."}) + CHECKLIST,
     True, True, True, True),
    ("subscribe", {"customer_id": "A-C0001", "product_id": "A-P0001", "amount": 1e9},
     json.dumps({"status": "rejected", "violated_rules": [2], "reasons": ["over 30% of total assets"]}) + CHECKLIST,
     True, False, True, False),
    # a write the schema check rejected for a missing key is a form error; one rejected for its value is not
    ("subscribe", {"customer_id": "A-C0001", "amount": 100.0},
     json.dumps({"status": "rejected", "error": "missing ['product_id'] / unexpected []"}) + CHECKLIST,
     False, False, True, False),
    ("redeem", {"customer_id": "A-C0001", "product_id": "A-P0001", "units": -5},
     json.dumps({"status": "rejected", "error": "units must be a positive number"}) + CHECKLIST,
     True, False, True, False),
    ("finish", {"answer": "done"}, json.dumps({"status": "rejected", "verification_failed": ["checklist items not done"]}),
     True, False, True, False),
    ("finish", {"answer": "done"}, json.dumps({"status": "finished"}), True, True, True, True)]
V2_PLAIN = [V2_STEPS[2], V2_STEPS[3], V2_STEPS[-1]]
V2_RUNS = (("default", []), ("mask_rejected", ["--mask-rejected"]), ("compat", ["--train-form-errors"]),
           ("compat_mask_rejected", ["--train-form-errors", "--mask-rejected"]))
# the turns each run takes out of the loss for a form error alone: under --mask-rejected the schema rejection already is
V2_FORM_MASKED = {"default": 3, "mask_rejected": 2, "compat": 0, "compat_mask_rejected": 0}


@unittest.skipIf(b3 is None or not os.path.exists(f"{V2_CODE}/agent.py"),
                 "the training environment, the Qwen3.5 tokenizer or harness_v2's code snapshot is not available")
class TestBuilderV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="agent_sft_v2_test_")
        c = os.path.join(cls.tmp, "collect")
        cls.msgs = {"w1": episode([s[:3] for s in V2_STEPS]), "w2": episode([s[:3] for s in V2_PLAIN]),
                    "w3": episode([(None, None, "Done."), V2_STEPS[-1][:3]], users=2),  # nudged: dropped as before
                    **{f"w{4 + i}": episode([b, V2_STEPS[-1][:3]]) for i, b in enumerate(BAD_ARGS)}}
        write_jsonl(f"{c}/wealth/round1_tasks.jsonl", [{"task_id": t, "template": "t1_x"} for t in cls.msgs])
        write_jsonl(f"{c}/wealth/sampled/results.jsonl", [{"task_id": t, "run": 1, "passed": True} for t in cls.msgs])
        write_jsonl(f"{c}/wealth/sampled/trajectories.jsonl", [{"task_id": t, "run": 1, "messages": m}
                                                               for t, m in cls.msgs.items()])
        write_jsonl(f"{c}/ml/round1_tasks.jsonl", [])
        cls.out = {}
        for name, extra in V2_RUNS:
            out = os.path.join(cls.tmp, name)
            r = subprocess.run([sys.executable, "-B", os.path.join(HERE, "utils", "build_agent_sft.py"), "--collect", c,
                                "--code", V2_CODE, "--tokenizer", TOKENIZER, "--out", out] + extra, capture_output=True,
                               text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            assert r.returncode == 0, r.stderr[-3000:]
            with open(out + ".manifest.json", encoding="utf-8") as f:
                cls.out[name] = (b3.unpack(out), json.load(f))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def by_turns(self, name, n):
        """The output sample with n assistant turns."""
        return next(s for s in self.out[name][0] if len(spans(s[0])) == n)

    def test_labels_per_option(self):
        for k, (name, _) in enumerate(V2_RUNS):
            with self.subTest(run=name):
                self.assertEqual(turn_flags(*self.by_turns(name, len(V2_STEPS))), [s[3 + k] for s in V2_STEPS])
                self.assertEqual(turn_flags(*self.by_turns(name, len(V2_PLAIN))), [s[3 + k] for s in V2_PLAIN])
                m = self.out[name][1]
                self.assertEqual(m["assistant_turns"]["masked for a form error"], V2_FORM_MASKED[name])
                self.assertEqual((m["trajectories"], m["dropped"]),
                                 (2, {"second user turn": 1, "arguments not a JSON object": 2}))

    def test_only_the_form_error_spans_differ(self):
        for new, old, col_new, col_old in (("default", "compat", 3, 5),
                                           ("mask_rejected", "compat_mask_rejected", 4, 6)):
            with self.subTest(run=new):
                ids, a = self.by_turns(new, len(V2_STEPS))
                ids_old, b = self.by_turns(old, len(V2_STEPS))
                self.assertEqual(ids, ids_old)
                masked = [sp for sp, s in zip(spans(ids), V2_STEPS) if s[col_old] and not s[col_new]]
                self.assertEqual({q for q in range(len(ids)) if a[q] != b[q]}, {q for s, e in masked for q in range(s, e)})
                self.assertEqual(self.by_turns(new, len(V2_PLAIN)), self.by_turns(old, len(V2_PLAIN)))


@unittest.skipIf(b3 is None or not os.path.exists(f"{V2_CODE_T6}/report_gate.py"),
                 "the training environment, the Qwen3.5 tokenizer or harness_v2's t6 code snapshot is not available")
class TestRlData(unittest.TestCase):
    """rl_data.py: render records the form-error spans, pack --mode sft masks them unless --train-form-errors, and
    --mode opd keeps the rendered labels."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="rl_data_test_")
        roll = os.path.join(cls.tmp, "roll")
        cls.msgs = {"r1": episode([s[:3] for s in V2_STEPS]), "r2": episode([s[:3] for s in V2_PLAIN]),
                    "r3": episode([V2_STEPS[0][:3], V2_STEPS[-1][:3]]),  # failed: only --mode opd keeps it
                    **{f"r{4 + i}": episode([b, V2_STEPS[-1][:3]]) for i, b in enumerate(BAD_ARGS)}}
        write_jsonl(f"{cls.tmp}/tasks.jsonl", [{"task_id": t, "template": "t1_x", "env": "wealth"} for t in cls.msgs])
        write_jsonl(f"{roll}/results.jsonl", [{"task_id": t, "run": 1, "end": "finish", "passed": t != "r3"}
                                              for t in cls.msgs])
        write_jsonl(f"{roll}/trajectories.jsonl", [{"task_id": t, "run": 1, "messages": m}
                                                   for t, m in cls.msgs.items()])
        cls.rendered = os.path.join(cls.tmp, "rendered.jsonl")
        cls.render_out = json.loads(cls.run_rl("render", "--rollouts", roll, "--tasks", f"{cls.tmp}/tasks.jsonl",
                                               "--code", V2_CODE_T6, "--tokenizer", TOKENIZER, "--out", cls.rendered))
        with open(cls.rendered, encoding="utf-8") as f:
            cls.rows = {x["task_id"]: x for x in map(json.loads, f)}
        scored = os.path.join(cls.tmp, "scored.jsonl")  # as score would add them: a constant advantage of 0.5
        write_jsonl(scored, [dict(x, lp_old=[-1.0] * len(x["ids"]), lp_teacher=[-0.5] * len(x["ids"]))
                             for x in cls.rows.values()])
        cls.out = {}
        for name, rendered, extra in (("sft", cls.rendered, ["--mode", "sft"]),
                                      ("sft_compat", cls.rendered, ["--mode", "sft", "--train-form-errors"]),
                                      ("opd", scored, ["--mode", "opd"])):
            out = os.path.join(cls.tmp, name)
            cls.run_rl("pack", "--rendered", rendered, "--out", out, "--tokenizer", TOKENIZER, *extra)
            with open(out + ".manifest.json", encoding="utf-8") as f:
                cls.out[name] = (datasets.load_from_disk(out)["train"], b3.unpack(out), json.load(f))

    @classmethod
    def run_rl(cls, *args):
        r = subprocess.run([sys.executable, "-B", os.path.join(HERE, "utils", "rl_data.py"), *args], capture_output=True,
                           text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        assert r.returncode == 0, r.stderr[-3000:]
        return r.stdout.strip().splitlines()[-1]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def sample(self, name, task):
        return next(s for s in self.out[name][1] if s[0] == self.rows[task]["ids"])

    def test_render(self):
        self.assertEqual(self.render_out, {"kept": 3, "turns with a form error": 4,
                                           "dropped": {"arguments not a JSON object": 2}})
        for t, x in self.rows.items():
            with self.subTest(task=t):
                self.assertTrue(all(turn_flags(x["ids"], x["labels"])))  # every assistant turn, as before
                off = [list(sp) for sp, f in zip(spans(x["ids"]), mask_form_errors(self.msgs[t])[0]) if not f]
                self.assertEqual(x["form_error_spans"], off)
        self.assertEqual(len(self.rows["r1"]["form_error_spans"]), 3)

    def test_sft_masks_the_form_error_turns(self):
        self.assertEqual(turn_flags(*self.sample("sft", "r1")), [s[3] for s in V2_STEPS])
        self.assertEqual(turn_flags(*self.sample("sft", "r2")), [s[3] for s in V2_PLAIN])
        self.assertEqual(self.out["sft"][2]["trajectories"], 2)
        self.assertEqual(self.out["sft"][2]["assistant_turns_masked_for_a_form_error"], 3)

    def test_sft_compat_and_opd_keep_the_rendered_labels(self):
        for t in ("r1", "r2"):
            self.assertEqual(self.sample("sft_compat", t), (self.rows[t]["ids"], self.rows[t]["labels"]))
        self.assertEqual(self.out["sft_compat"][2]["assistant_turns_masked_for_a_form_error"], 0)
        rows, samples, manifest = self.out["opd"]
        self.assertEqual(manifest["trajectories"], 3)
        self.assertNotIn("assistant_turns_masked_for_a_form_error", manifest)
        for t in ("r1", "r2", "r3"):
            self.assertEqual(self.sample("opd", t), (self.rows[t]["ids"], self.rows[t]["labels"]))
        for row in rows:  # the advantage lies on every labelled token, the form-error turns' included
            self.assertEqual([v != 0 for v in row["token_adv"]], [v != b3.IGNORE for v in row["labels"]])

    def test_a_file_rendered_before_stops_sft(self):
        old = os.path.join(self.tmp, "rendered_v1.jsonl")
        write_jsonl(old, [{k: v for k, v in x.items() if k != "form_error_spans"} for x in self.rows.values()])
        r = subprocess.run([sys.executable, "-B", os.path.join(HERE, "utils", "rl_data.py"), "pack", "--rendered", old, "--mode",
                            "sft", "--out", os.path.join(self.tmp, "sft_v1"), "--tokenizer", TOKENIZER],
                           capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("has no form_error_spans", r.stderr)


if __name__ == "__main__":
    unittest.main()
