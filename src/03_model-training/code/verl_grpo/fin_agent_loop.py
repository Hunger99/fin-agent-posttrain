"""A verl agent loop that runs harness v3's episodes under H1, token in and token out, for GRPO on the market tasks
(research/log/2026-11-01-0010.md, 20:xx UTC).

Each dataset row carries one task (extra_info["task"], its JSON line). The loop builds harness v3's Episode for the task,
so the system prompt, the tools, the checklist restated after each result, verification and correction at finish, the
nudge after a reply without a tool call, tool errors and the step accounting are the ones the SFT trajectories and the
evaluations had; only the chat-completions call is replaced by verl's rollout server:
- the prompt (system, user, tools) is rendered once with the model's chat template, and every turn's sampled tokens are
  appended as sampled, so each earlier turn keeps its thinking, as harness v3's keep-thinking template renders it;
- tool results and nudges are rendered as appended messages by verl's Continuous Token builder, with loss mask 0;
- a turn's tool calls are read from its tokens with verl's qwen3_coder parser (Qwen3.5's XML calls).
The episode ends as harness v3 ends it (finish accepted, FIN_MAX_STEPS task turns, the hard cap) or when the next turn
would pass rollout.response_length, the budget for every response and context token after the prompt ("overlong").
The reward is the task's grade, 1.0 when passed and 0.0 otherwise, set as reward_score so verl's reward loop is skipped.
An overlong episode keeps its reward 0 in its group but carries no loss (FIN_MASK_OVERLONG=1, the default; research/log/
2026-11-01-0010.md, 21:5x UTC): the budget cut it, not the model, and a 0 on it would push the policy to answer early.
One JSON line per episode (task, reward, end, turns, tokens, tool errors, the finish answer and the keys it misses) goes
to FIN_EPISODE_LOG_DIR/<pid>.jsonl, so the answers can be regraded offline (research/log/2026-11-01-0010.md, 21:5x UTC).

Settings come from environment variables, since verl instantiates the loop in its own workers: FIN_HARNESS_V3 (harness v3's
directory), FIN_DB (the market snapshot), FIN_MAX_STEPS (task turns, 30), FIN_MAX_TOKENS (tokens per model call, 4096,
harness v3's Settings.max_tokens) and FIN_EPISODE_LOG_DIR.
"""
import json
import logging
import os
import sys
import time
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.profiler import simple_timer

V3 = os.environ.get("FIN_HARNESS_V3", os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, "02_harness", "code", "v3")))
if V3 not in sys.path:
    sys.path.insert(0, V3)
from harness import agent as hv3  # noqa: E402

DB = os.environ.get("FIN_DB", os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1/market_v1.sqlite")
MAX_STEPS = int(os.environ.get("FIN_MAX_STEPS", "30"))
MAX_TOKENS = int(os.environ.get("FIN_MAX_TOKENS", "4096"))
LOG_DIR = os.environ.get("FIN_EPISODE_LOG_DIR")
MASK_OVERLONG = os.environ.get("FIN_MASK_OVERLONG", "1") == "1"

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FinHarnessAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.tool_parser = ToolParser.get_tool_parser("qwen3_coder", self.tokenizer)
        self.im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _log(self, rec):
        if not LOG_DIR:
            return
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, f"{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        task = json.loads((kwargs.get("extra_info") or {})["task"])
        settings = hv3.Settings(provider="qwen", max_steps=MAX_STEPS, max_tokens=MAX_TOKENS, token_records=False)
        ep = hv3.Episode(task, "H1", DB, client=None, model="policy", seed=0, settings=settings)
        tools, schemas = ep.tools, [OpenAIFunctionToolSchema.model_validate(t) for t in ep.tools]
        ep.note_numbers(task["prompt"])
        ep.note_numbers(ep.kind.SYSTEM)
        metrics, request_id, t0 = {}, uuid4().hex, time.time()

        ids = await self.ct_build_initial_tokens(list(ep.msgs), tools=tools)
        n_prompt = len(ids)
        mask, logprobs, extra = [], [], {}
        while True:
            if ep.task_steps >= ep.s.max_steps:
                ep.end = "step_limit"
                break
            if len(ep.turns) >= ep.s.hard_cap():
                ep.end = "hard_cap"
                break
            room = self.response_length - len(mask)
            if room <= 1:
                ep.end = "overlong"
                break
            with simple_timer("generate_sequences", metrics):
                out = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=ids,
                    sampling_params={**sampling_params, "max_tokens": min(ep.s.max_tokens, room)})
            # the server's per-request fields (the policy versions the trainer tracks), as ToolAgentLoop keeps them
            if not extra:
                extra.update(out.extra_fields or {})
            elif (out.extra_fields or {}).get("max_global_steps") is not None:
                extra["max_global_steps"] = out.extra_fields["max_global_steps"]
            merged, mask, lp = await self.ct_merge_assistant_token(
                ids, out.token_ids, mask, logprobs if (logprobs or out.log_probs) else None,
                assistant_logprobs=out.log_probs or None)
            ids, logprobs = merged.token_ids, (lp or [])
            rec = {"n_completion": len(out.token_ids)}
            ep.turns.append(rec)
            text, calls = await self.tool_parser.extract_tool_calls(out.token_ids, schemas)
            content = text.split("</think>", 1)[-1].replace("<|im_end|>", "").strip()
            msg = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = [{"id": f"call_{len(ep.turns)}_{i}", "type": "function",
                                      "function": {"name": c.name, "arguments": c.arguments}} for i, c in enumerate(calls)]
            previous = list(ep.msgs) + [msg]
            ep.msgs.append(msg)
            kinds = []
            if len(out.token_ids) >= room:  # cut by the budget: no complete reply to act on
                ep.end = "overlong"
                break
            if not calls:
                tc = hv3.text_call(content, tools)
                if tc:
                    ep.stats["text_calls"] += 1
                ep.msgs.append({"role": "user", "content": hv3.TEXT_CALL_NUDGE.format(name=tc[0]) if tc else hv3.NUDGE})
            with simple_timer("tool_calls", metrics):
                for call in msg.get("tool_calls", []):
                    result, kind = await ep.execute(call["function"]["name"], call["function"]["arguments"])
                    kinds.append(kind)
                    ep.msgs.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                                    "content": result})
                    if ep.env.finished:
                        break
            ep.charge(rec, kinds, msg.get("tool_calls", []))
            if ep.env.finished:
                ep.end = "finished"
                break
            if not out.token_ids or out.token_ids[-1] != self.im_end:  # cut at FIN_MAX_TOKENS: close the turn, no loss
                ids, mask = ids + [self.im_end], mask + [0]
                logprobs = logprobs + [0.0] if logprobs else logprobs
            merged, new_mask, lp = await self.ct_merge_context_msg(
                previous, ep.msgs, ids, mask, logprobs or None, tools=tools)
            if len(new_mask) >= self.response_length:
                ep.end = "overlong"
                break
            ids, mask, logprobs = merged.token_ids, new_mask, (lp or [])

        grade = ep.kind.grade(task, ep.env)
        reward = 1.0 if grade["passed"] else 0.0
        response_ids = ids[n_prompt:]
        assert len(response_ids) == len(mask), (len(response_ids), len(mask))
        n_model, masked = sum(mask), MASK_OVERLONG and ep.end == "overlong"
        if masked:  # cut by the budget, not by the model: reward 0 in the group, no loss (DeepSWE, DAPO's overlong filter)
            mask = [0] * len(mask)
        self._log({"task_id": task["task_id"], "template": task.get("template"), "reward": reward, "end": ep.end,
                   "turns": len(ep.turns), "task_steps": ep.task_steps, "prompt_tokens": n_prompt,
                   "response_tokens": len(mask), "model_tokens": n_model, "loss_masked": masked,
                   "tool_errors": ep.stats["tool_errors"],
                   "verification_rejects": ep.stats["verification_rejects"], "seconds": round(time.time() - t0, 1),
                   "answer": ep.env.answer, "missing_keys": grade.get("missing_keys")})
        return AgentLoopOutput(
            prompt_ids=ids[:n_prompt],
            response_ids=response_ids[: self.response_length],
            response_mask=mask[: self.response_length],
            response_logprobs=logprobs[: self.response_length] if logprobs else None,
            reward_score=reward,
            num_turns=len(ep.msgs) - 1,
            metrics=AgentLoopMetrics(**{k: v for k, v in metrics.items() if k in AgentLoopMetrics.model_fields}),
            extra_fields=extra,
        )
