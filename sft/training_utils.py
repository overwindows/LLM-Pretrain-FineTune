"""Dependency-light SFT validation, chat masking, and model-family selection."""
from __future__ import annotations

import logging
import math
from numbers import Real
from typing import Any

logger = logging.getLogger("sft_train")


def warmup_arguments(training_arguments_class, ratio: float) -> dict[str, float]:
    if not 0 <= ratio <= 1:
        raise ValueError("warmup_ratio must be between 0 and 1")
    if "warmup_ratio" in training_arguments_class.__dataclass_fields__:
        return {"warmup_ratio": ratio}
    # Modern Trainer represents a ratio <1 directly in warmup_steps.
    if ratio == 1:
        raise ValueError("This Trainer requires warmup_ratio <1; 1 means one absolute step")
    return {"warmup_steps": float(ratio)}


def validate_sample_weight(rec: dict[str, Any]) -> None:
    weight = rec.get("sample_weight", 1)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, Real)
        or not math.isfinite(weight)
        or weight != 1
    ):
        raise ValueError(
            "sample_weight must be numeric 1 or absent. This trainer uses uniform "
            "row sampling; materialize positive integer task frequencies upstream "
            "and remove sample_weight. Non-unit weights are not silently discarded."
        )


def model_class_for_config(config, transformers_module):
    """Keep the full Qwen3.5 checkpoint rather than extracting its text tower."""
    model_type = config.model_type
    if model_type == "qwen3_5":
        model_class = getattr(
            transformers_module, "Qwen3_5ForConditionalGeneration", None
        )
        if model_class is None:
            raise RuntimeError(
                "qwen3_5 requires Transformers with "
                "Qwen3_5ForConditionalGeneration support; use the isolated "
                "scripts/setup_env.sh --profile qwen35 environment."
            )
        architectures = getattr(config, "architectures", None)
        if architectures and architectures != ["Qwen3_5ForConditionalGeneration"]:
            raise ValueError(f"Unsupported qwen3_5 architectures: {architectures!r}")
        return model_class
    if model_type in {"qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}:
        raise ValueError(
            f"Unsupported model_type {model_type!r}: this Qwen3.5 SFT path "
            "requires the full dense qwen3_5 conditional-generation checkpoint."
        )
    return transformers_module.AutoModelForCausalLM


def configure_qwen35_text_training(model) -> None:
    if model.__class__.__name__ != "Qwen3_5ForConditionalGeneration":
        raise RuntimeError(f"Unexpected Qwen3.5 model class: {type(model).__name__}")
    if model.config.model_type != "qwen3_5":
        raise RuntimeError("Qwen3.5 must retain its full export configuration")
    # Text batches do not exercise the vision tower; freeze it for DDP.
    model.model.visual.requires_grad_(False)
    model.config.use_cache = False
    model.config.text_config.use_cache = False


def _render_and_encode(
    tokenizer, messages, add_generation_prompt: bool, enable_thinking: bool | None = None
) -> list[int]:
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **kwargs,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def find_last_assistant_span(
    tokenizer, messages, full_ids, enable_thinking: bool | None = None
) -> tuple[int, int]:
    prefix_ids = _render_and_encode(
        tokenizer, messages[:-1], True, enable_thinking=enable_thinking
    )
    start = len(prefix_ids)
    if prefix_ids != full_ids[:start]:
        raise ValueError(
            "Generation prompt is not an exact prefix of the training chat. "
            "Use the same explicit chat-template settings for training and serving "
            "(Qwen3.5 nonreasoning targets require enable_thinking: false). "
            "Refusing to guess the assistant loss boundary."
        )
    return start, len(full_ids)


def find_think_close_token_ids(tokenizer) -> list[int]:
    return tokenizer.encode("</think>", add_special_tokens=False)


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def tokenize_record(
    rec: dict[str, Any],
    tokenizer,
    max_seq_len: int,
    mask_think_in_loss: bool = False,
    think_close_ids: list[int] | None = None,
    enable_thinking: bool | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    validate_sample_weight(rec)
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2 for causal loss")
    messages = rec.get("messages", [])
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        if stats is not None:
            stats["skip_reason"] = "no_final_assistant"
        logger.warning("Skipping record without a final assistant target")
        return None
    if any(not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("Text SFT requires string content; multimodal rows are unsupported")
    if not messages[-1]["content"].strip():
        if stats is not None:
            stats["skip_reason"] = "empty_target"
        logger.warning("Skipping empty assistant target")
        return None

    full_ids = _render_and_encode(tokenizer, messages, False, enable_thinking)
    start, end = find_last_assistant_span(
        tokenizer, messages, full_ids, enable_thinking
    )
    # Retain a predecessor for the first target token's causal prediction.
    offset = max(0, len(full_ids) - max_seq_len)
    if stats is not None:
        stats.update({"original_length": end, "context_tokens_removed": offset})
    if start < 1 or offset >= start or end <= start:
        if stats is not None:
            stats["skip_reason"] = "target_does_not_fit"
        logger.warning("Skipping target that cannot fit intact with a context token")
        return None

    train_start = start
    if mask_think_in_loss and think_close_ids:
        rel = find_subsequence(full_ids[start:end], think_close_ids)
        if rel >= 0:
            train_start = start + rel + len(think_close_ids)
    if train_start >= end:
        if stats is not None:
            stats["skip_reason"] = "no_supervised_tokens"
        logger.warning("Skipping record with no supervised tokens after think masking")
        return None

    labels = [-100] * train_start + full_ids[train_start:]
    return {
        "input_ids": full_ids[offset:],
        "attention_mask": [1] * (end - offset),
        "labels": labels[offset:],
    }
