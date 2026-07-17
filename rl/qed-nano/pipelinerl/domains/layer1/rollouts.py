"""Layer1 delta Stage-4 rollout policy (in-process reward + Azure judge fuse).

Mirrors ``pipelinerl.domains.math.generate_math_rollout`` but:
- builds the prompt from the per-record chat messages (system + user) instead of
  a single ``task_template`` (keeps the exact SFT prompt);
- scores with the in-process Stage-4 reward
  ``r = gate * (0.5 * r_rule + 0.5 * r_llm)``;
- ``r_llm`` comes from the Azure M2 interest judge (realtime + cached). When the
  judge is disabled (``LAYER1_JUDGE_ENABLED`` != "1") the reward is rule-only,
  which lets us smoke-test the loop without burning judge quota.
"""

from __future__ import annotations

import time
import logging

import aiohttp
from omegaconf import DictConfig, OmegaConf
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import RolloutResult, BaseMetrics

from . import judge as judge_mod
from .reward import build_config, compute_rule_reward, fuse_stage4

logger = logging.getLogger(__name__)


_REWARD_CFG = None


def _get_reward_cfg(cfg: DictConfig):
    global _REWARD_CFG
    if _REWARD_CFG is None:
        raw = OmegaConf.to_container(cfg.reward, resolve=True) if "reward" in cfg else {}
        _REWARD_CFG = build_config(raw)
    return _REWARD_CFG


async def generate_layer1_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
    rc_actor: bool = False,
) -> RolloutResult:
    reward_cfg = _get_reward_cfg(cfg)
    actor_cfg = cfg.rc_actor if rc_actor else cfg.actor

    # ---- build prompt from the per-record chat messages ----
    messages = list(problem.get("messages_prompt") or [])
    if not messages:
        # Fallback: system_prompt + task_template (math-style)
        messages = []
        if actor_cfg.get("system_prompt"):
            messages.append({"role": "system", "content": actor_cfg.system_prompt})
        messages.append({"role": "user", "content": problem.get("task", "")})
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    completion = llm_call.output.content or ""
    trace = make_training_text(llm, llm_call)

    # ---- rule reward (gate + 2b anti-collapse + anti-hallucination + fidelity) ----
    out = compute_rule_reward(
        completion=completion,
        input_signals=list(problem.get("input_signals") or []),
        teacher_count=problem.get("teacher_count"),
        teacher_length=problem.get("teacher_length"),
        cfg=reward_cfg,
    )
    gate_value = float(out.components.get("gate", 0.0))

    # ---- judge fuse (r_llm) ----
    utility = precision = None
    recall_info: dict | None = None
    if judge_mod.judge_enabled() and gate_value > 0.0 and out.parsed:
        utility, precision = await judge_mod.judge_interests(out.parsed)
        if judge_mod.recall_judge_enabled():
            grounded = await judge_mod.load_grounded(problem.get("user_id", ""))
            if grounded:
                recall_info = await judge_mod.judge_recall(out.parsed, grounded)

    recall_overall = recall_info.get("overall") if recall_info else None
    out = fuse_stage4(
        out, gate_value, utility, precision, reward_cfg, recall=recall_overall
    )

    discount_factor = actor_cfg.get("discount_factor", 1)
    reward = out.reward * (discount_factor ** llm_call.output_length_tokens)
    trace.reward = reward

    metrics = BaseMetrics(
        reward=reward,
        success=out.success,
        no_error=out.components.get("parse_ok", False),
        no_answer=(out.metadata.get("n_interests", 0) == 0),
    )

    verifier_metrics: dict[str, float | int] = {
        "verifier/gate": gate_value,
        "verifier/r_rule": out.r_rule,
        "verifier/anti_collapse": out.components.get("anti_collapse", 0.0),
        "verifier/anti_hallucination": out.components.get("anti_hallucination", 0.0),
        "verifier/fidelity_score": out.components.get("fidelity_score", 0.0),
        "verifier/n_interests": out.metadata.get("n_interests", 0),
        "verifier/rollouts/success_frac": 1.0 if out.success else 0.0,
    }
    if out.r_llm is not None:
        verifier_metrics["verifier/r_llm"] = out.r_llm
        if utility is not None:
            verifier_metrics["verifier/judge_utility"] = utility
        if precision is not None:
            verifier_metrics["verifier/judge_precision"] = precision
    if recall_info is not None:
        # overall feeds the reward; matched/broad are diagnostics only.
        if recall_info.get("overall") is not None:
            verifier_metrics["verifier/judge_recall_overall"] = recall_info["overall"]
        if recall_info.get("matched") is not None:
            verifier_metrics["verifier/judge_recall_matched"] = recall_info["matched"]
        if recall_info.get("broad") is not None:
            verifier_metrics["verifier/judge_recall_broad"] = recall_info["broad"]
        verifier_metrics["verifier/recall_n_grounded"] = recall_info.get("n_grounded", 0)
        verifier_metrics["verifier/recall_n_covered"] = recall_info.get("n_covered", 0)

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        verifier_metrics=verifier_metrics,
    )
