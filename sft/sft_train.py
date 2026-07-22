"""SFT training script for MAIProfile layer steps.

Reads {messages} format JSONL (curated by GPT-5.2), tokenizes via the base
model's chat template, and trains with HuggingFace Trainer + DeepSpeed ZeRO-3.

Loss is computed ONLY on tokens of the LAST assistant message. All earlier
messages (system, few-shot demo user/assistant, real user input) are masked
with label = -100.

Step-agnostic: the same script is used for every layer; per-step hyperparams
live in configs/sft/{step_key}.yaml.

Usage:
    accelerate launch --config_file configs/accelerate/accelerate_ds3.yaml \
        scripts/sft_train.py --config configs/sft/layer0_signal.yaml

Smoke test on 1 GPU:
    python scripts/sft_train.py --config configs/sft/layer0_signal.yaml --debug
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Compat: PyTorch 2.6+ made `weights_only=True` the default for torch.load,
# but DeepSpeed 0.15.4's torch_checkpoint_engine calls torch.load() without
# specifying weights_only, breaking checkpoint reload (e.g. the
# load_best_model_at_end=True path at end of Trainer.train()).
# We trust checkpoints we wrote ourselves — restore prior default.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat  # type: ignore[assignment]

import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger("sft_train")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SFTConfig:
    step_key: str
    base_model: str
    train_jsonl: str
    val_jsonl: str
    output_dir: str
    max_seq_len: int = 5120
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 3.0
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    save_strategy: str = "steps"
    save_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 50
    logging_steps: int = 5
    save_total_limit: int = 3
    gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"   # "flash_attention_2" if available
    deepspeed_config: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    resume_from_checkpoint: str | None = None  # "auto" or path
    # v2: when true, mask the <think>...</think> span inside the assistant
    # target so only post-</think> tokens (i.e. the JSON answer) contribute to
    # the cross-entropy loss. Records that have no </think> are unaffected
    # (the full assistant span still contributes — which for those records is
    # just the JSON answer).
    mask_think_in_loss: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        # ignore unknown keys (forward compat)
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Data loading + tokenization + loss mask
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _render_and_encode(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> list[int]:
    """Render the chat template to a string, then encode to int ids.

    Works around transformers >= 5.x behaviour where
    apply_chat_template(..., tokenize=True) returns a BatchEncoding wrapping
    [Encoding(...)] (one element per message) instead of a flat int list.
    """
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def find_last_assistant_span(
    tokenizer,
    messages: list[dict[str, str]],
    full_ids: list[int],
) -> tuple[int, int]:
    """Return (start, end) token indices of the last assistant message's tokens
    in `full_ids` produced by render+encode of the full message list.

    Strategy: render messages[:-1] with add_generation_prompt=True. The length
    of that prefix is the start; end = len(full_ids).
    """
    prefix_messages = messages[:-1]
    prefix_ids = _render_and_encode(tokenizer, prefix_messages, add_generation_prompt=True)
    start = len(prefix_ids)
    end = len(full_ids)
    # sanity check: prefix should be a prefix of full_ids
    if prefix_ids != full_ids[:start]:
        # fall back: find longest matching prefix
        for i in range(min(len(prefix_ids), len(full_ids)) - 1, 0, -1):
            if prefix_ids[:i] == full_ids[:i]:
                start = i
                break
    return start, end


def find_think_close_token_ids(tokenizer) -> list[int]:
    """Return the token ids for the literal string '</think>'. Qwen3 chat
    template uses a single special-token id for this; other tokenizers may
    split it into 2-3 sub-tokens.

    Returns [] if the tokenizer does not recognise the string at all (in which
    case think masking is a no-op for that base model).
    """
    ids = tokenizer.encode("</think>", add_special_tokens=False)
    return ids if ids else []


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Return the FIRST index i such that haystack[i:i+len(needle)] == needle,
    or -1 if not found."""
    if not needle:
        return -1
    nlen = len(needle)
    for i in range(len(haystack) - nlen + 1):
        if haystack[i : i + nlen] == needle:
            return i
    return -1


def tokenize_record(
    rec: dict[str, Any],
    tokenizer,
    max_seq_len: int,
    mask_think_in_loss: bool = False,
    think_close_ids: list[int] | None = None,
) -> dict[str, Any] | None:
    """Returns {input_ids, attention_mask, labels} for one record.
    Labels are -100 except on the last assistant message tokens.
    Returns None if the record can't be used (no assistant message, truncated
    away, etc.).

    When `mask_think_in_loss=True` and the assistant span contains a closing
    `</think>` token (id sequence provided in `think_close_ids`), label
    positions from `start` up to and including the `</think>` token are also
    set to -100. Only tokens AFTER `</think>` (the JSON answer) contribute to
    the loss. If `</think>` is not present, the full assistant span trains.
    """
    messages = rec.get("messages", [])
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        return None

    try:
        full_ids = _render_and_encode(tokenizer, messages, add_generation_prompt=False)
    except Exception as exc:
        logger.warning("chat_template failed: %s", exc)
        return None

    if len(full_ids) > max_seq_len:
        # Truncate from the LEFT, keeping the assistant target intact at the end.
        full_ids = full_ids[-max_seq_len:]

    start, end = find_last_assistant_span(tokenizer, messages, full_ids)
    if end - start <= 0:
        # After truncation, the assistant tokens may have been cut. Skip.
        return None

    input_ids = full_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(input_ids)

    train_start = start
    if mask_think_in_loss and think_close_ids:
        # Search for </think> only inside the assistant span
        rel = find_subsequence(input_ids[start:end], think_close_ids)
        if rel >= 0:
            # Mask through the </think> tokens themselves; train starts after.
            train_start = start + rel + len(think_close_ids)
            if train_start >= end:
                # </think> at the very end leaves nothing to train on; fall
                # back to the safe behaviour of training the whole span.
                train_start = start

    for i in range(train_start, end):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_dataset(
    jsonl_path: str,
    tokenizer,
    max_seq_len: int,
    debug_subsample: int | None = None,
    mask_think_in_loss: bool = False,
    num_proc: int | None = None,
) -> Dataset:
    """Build a tokenized HuggingFace Dataset from a JSONL file.

    Tokenization runs via ``Dataset.map(num_proc=N)`` — each record is
    processed in a separate worker process, bypassing the GIL and using all
    available CPU cores.  This replaces the old single-threaded for-loop which
    took 20–40 min on 50k × 14336-token records with the slow tokenizer.

    ``num_proc`` defaults to ``os.cpu_count()`` (all cores).  Pass 1 to
    reproduce the old single-process behaviour (useful for debugging).
    """
    records = load_jsonl(jsonl_path)
    if debug_subsample:
        records = records[:debug_subsample]

    think_close_ids: list[int] = []
    if mask_think_in_loss:
        think_close_ids = find_think_close_token_ids(tokenizer)
        logger.info(
            "mask_think_in_loss=True; </think> token ids = %r", think_close_ids
        )
        if not think_close_ids:
            logger.warning(
                "tokenizer does not recognise '</think>' string; think masking "
                "will be a no-op for every record"
            )

    raw_ds = Dataset.from_list(records)

    # Bind tokenizer + hyperparams into the map fn.  Each worker process gets
    # its own copy via pickle — safe with use_fast=False (pure-Python tokenizer).
    _SKIP = {"input_ids": [], "attention_mask": [], "labels": []}

    def _map_fn(rec):
        # tokenize_record returns None when the record should be skipped.
        # dataset.map requires consistent output columns across all rows, so
        # we return empty lists instead of None; the filter step removes them.
        result = tokenize_record(
            rec,
            tokenizer,
            max_seq_len,
            mask_think_in_loss=mask_think_in_loss,
            think_close_ids=think_close_ids,
        )
        return result if result is not None else _SKIP

    n_workers = num_proc if num_proc is not None else min(8, os.cpu_count() or 8)
    logger.info(
        "build_dataset(%s): tokenizing %d records with num_proc=%d ...",
        jsonl_path, len(records), n_workers,
    )

    tokenized_ds = raw_ds.map(
        _map_fn,
        num_proc=n_workers,
        remove_columns=raw_ds.column_names,
        desc=f"Tokenizing {os.path.basename(jsonl_path)}",
        # Keep only records where input_ids was produced (skip → empty dict)
        load_from_cache_file=False,
    )

    # Filter out skipped records (empty lists from _SKIP sentinel)
    tokenized_ds = tokenized_ds.filter(
        lambda x: len(x["input_ids"]) > 0,
        num_proc=n_workers,
        desc="Filtering skipped",
    )

    # Compute stats for logging
    n_kept = len(tokenized_ds)
    n_total = len(records)
    n_skipped = n_total - n_kept
    n_overlong = sum(1 for ids in tokenized_ds["input_ids"] if len(ids) == max_seq_len)

    logger.info(
        "build_dataset(%s): kept=%d skipped=%d overlong=%d",
        jsonl_path, n_kept, n_skipped, n_overlong,
    )
    return tokenized_ds


# ---------------------------------------------------------------------------
# Collator: dynamic padding to longest in batch
# ---------------------------------------------------------------------------
class DataCollatorForSFT:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Custom Trainer: compute extra metric "token_accuracy" on eval
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """eval_pred has predictions (logits) and label_ids.
    Compute token-level accuracy on the non-(-100) positions.
    Note: HF Trainer requires `preprocess_logits_for_metrics` to argmax logits,
    otherwise predictions array is huge.
    """
    preds, labels = eval_pred
    # preds shape: (batch, seq_len) after preprocess_logits_for_metrics
    # labels shape: (batch, seq_len)
    # Shift: model predicts token t+1 from position t, but Trainer already shifts
    # labels internally for CE loss. For metrics we mirror: predictions[i] should
    # match labels[i] where labels[i] != -100 (after the same shift).
    # We approximate: ignore shift here (counts both sides), which is fine for trend.
    mask = labels != -100
    if mask.sum() == 0:
        return {"token_accuracy": 0.0}
    correct = ((preds == labels) & mask).sum()
    total = mask.sum()
    return {"token_accuracy": float(correct) / float(total)}


def preprocess_logits_for_metrics(logits, labels):
    # logits: (batch, seq_len, vocab) → argmax → (batch, seq_len)
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to configs/sft/{step}.yaml")
    ap.add_argument("--debug", action="store_true", help="tiny subsample, 1 GPU, 5 steps")
    ap.add_argument("--smoke-test", action="store_true",
                    help="train 5 steps on 32 records, no eval; just verify loop works")
    ap.add_argument("--no-wandb", action="store_true",
                    help="force report_to=[none], skip WandB even if yaml sets wandb_project")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="if > 0, override num_train_epochs and stop after this many optimizer steps "
                         "(useful for full-path DS smoke tests)")
    ap.add_argument("--override-output-dir", default=None,
                    help="override output_dir (useful for smoke test)")
    args = ap.parse_args()

    setup_logging()
    cfg = SFTConfig.from_yaml(args.config)
    if args.override_output_dir:
        cfg.output_dir = args.override_output_dir

    set_seed(cfg.seed)
    logger.info("=" * 60)
    logger.info("Config:")
    for k, v in cfg.__dict__.items():
        logger.info("  %s = %r", k, v)
    logger.info("=" * 60)

    # --- WandB ---
    if cfg.wandb_project and not args.smoke_test and not args.debug and not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
        if cfg.wandb_run_name:
            os.environ.setdefault("WANDB_NAME", cfg.wandb_run_name)
        if cfg.wandb_tags:
            os.environ.setdefault("WANDB_TAGS", ",".join(cfg.wandb_tags))
        report_to = ["wandb"]
    else:
        report_to = ["none"]

    # --- Tokenizer ---
    logger.info("Loading tokenizer: %s", cfg.base_model)
    # use_fast=False: the ptca AML conda env has tokenizers 0.19.x (Python 3.8 ceiling).
    # Qwen3's tokenizer.json uses the ModelWrapper format introduced in tokenizers>=0.20.0,
    # so fast tokenizer loading always fails with "data did not match any variant of
    # untagged enum ModelWrapper". The slow tokenizer is pure-Python, reads vocab/merges
    # files directly, and has no version dependency on the Rust tokenizers extension.
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Datasets ---
    debug_n = 32 if (args.debug or args.smoke_test) else None
    logger.info("Building train dataset ...")
    train_ds = build_dataset(
        cfg.train_jsonl, tokenizer, cfg.max_seq_len, debug_n,
        mask_think_in_loss=cfg.mask_think_in_loss,
    )
    logger.info("Building val dataset ...")
    val_ds = build_dataset(
        cfg.val_jsonl, tokenizer, cfg.max_seq_len, debug_n,
        mask_think_in_loss=cfg.mask_think_in_loss,
    )

    if len(train_ds) == 0:
        logger.error("Empty train dataset")
        sys.exit(2)

    # --- Model ---
    logger.info("Loading model: %s (attn=%s)", cfg.base_model, cfg.attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=True,
    )
    logger.info("Model loaded.")
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required for DeepSpeed ZeRO-3 + gradient checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    logger.info("Gradient checkpointing applied.")

    # --- TrainingArguments ---
    logger.info("Building TrainingArguments ...")
    if args.smoke_test:
        max_steps = 5
        eval_strategy = "no"
        save_strategy = "no"
        logging_steps = 1
        report_to = ["none"]
    elif args.max_steps > 0:
        max_steps = args.max_steps
        # keep yaml's eval/save behaviour; but if max_steps < save_steps,
        # save_steps would never fire — that's intentional for short smokes.
        eval_strategy = cfg.eval_strategy
        save_strategy = cfg.save_strategy
        logging_steps = max(1, min(cfg.logging_steps, max_steps))
    else:
        max_steps = -1
        eval_strategy = cfg.eval_strategy
        save_strategy = cfg.save_strategy
        logging_steps = cfg.logging_steps

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs if max_steps == -1 else 1,
        max_steps=max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=cfg.eval_steps if eval_strategy == "steps" else None,
        save_strategy=save_strategy,
        save_steps=cfg.save_steps if save_strategy == "steps" else None,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=(save_strategy != "no" and eval_strategy != "no"),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=cfg.deepspeed_config if not args.debug and not args.smoke_test else None,
        seed=cfg.seed,
        report_to=report_to,
        run_name=cfg.wandb_run_name,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        logging_first_step=True,
    )
    logger.info("TrainingArguments built.")

    collator = DataCollatorForSFT(pad_token_id=tokenizer.pad_token_id)

    logger.info("Constructing Trainer ...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if eval_strategy != "no" else None,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics if eval_strategy != "no" else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_strategy != "no" else None,
    )
    logger.info("Trainer constructed.")

    # --- Resume ---
    resume = None
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint == "auto":
            # auto-detect
            ckpts = list(Path(cfg.output_dir).glob("checkpoint-*"))
            if ckpts:
                resume = str(sorted(ckpts, key=lambda p: int(p.name.split("-")[-1]))[-1])
                logger.info("Auto-resuming from %s", resume)
        else:
            resume = cfg.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=resume)

    # --- Save final ---
    logger.info("Saving final model to %s", cfg.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(cfg.output_dir)
    if trainer.is_world_process_zero():
        with open(os.path.join(cfg.output_dir, "training_summary.json"), "w") as f:
            json.dump(
                {
                    "step_key": cfg.step_key,
                    "base_model": cfg.base_model,
                    "train_jsonl": cfg.train_jsonl,
                    "val_jsonl": cfg.val_jsonl,
                    "best_metric": trainer.state.best_metric,
                    "best_model_checkpoint": trainer.state.best_model_checkpoint,
                    "global_step": trainer.state.global_step,
                },
                f, indent=2,
            )
    logger.info("Done.")


if __name__ == "__main__":
    main()
