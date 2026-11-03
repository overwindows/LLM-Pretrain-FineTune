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
import importlib.metadata
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import yaml
import transformers
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

if __package__:
    from .bounded_validation import (
        BoundedValidationConfig, BoundedValidationTrainer, ValidationMeasurements,
        collective_action, collective_check, exchange,
        deepspeed_config_snapshot, file_sha256, state_sha256, validate_checkpoint,
    )
    from .training_utils import (
        configure_qwen35_text_training,
        find_think_close_token_ids,
        model_class_for_config,
        tokenize_record,
        validate_sample_weight,
        warmup_arguments,
    )
else:
    from bounded_validation import (
        BoundedValidationConfig, BoundedValidationTrainer, ValidationMeasurements,
        collective_action, collective_check, exchange,
        deepspeed_config_snapshot, file_sha256, state_sha256, validate_checkpoint,
    )
    from training_utils import (
        configure_qwen35_text_training,
        find_think_close_token_ids,
        model_class_for_config,
        tokenize_record,
        validate_sample_weight,
        warmup_arguments,
    )

logger = logging.getLogger("sft_train")


def setup_deepspeed() -> None:
    import deepspeed
    import deepspeed.runtime.lr_schedules  # noqa: F401

    # Only the historical DS checkpoint engine needs the old torch.load default.
    if deepspeed.__version__ == "0.15.4":
        original_load = torch.load

        def load_legacy_checkpoint(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = load_legacy_checkpoint


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
    optim: str | None = None
    adam_beta1: float | None = None
    adam_beta2: float | None = None
    adam_epsilon: float | None = None
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
    # None preserves legacy defaults. Qwen3.5 requires an explicit policy.
    enable_thinking: bool | None = None
    model_revision: str | None = None
    bounded_validation: dict[str, Any] | None = None
    validation_fixed_order: bool = False
    validation_checkpoint_step: int | None = None
    full_corpus_contract: str | None = None
    full_corpus_segment_end_step: int | None = None

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
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return records


def build_dataset(
    jsonl_path: str,
    tokenizer,
    max_seq_len: int,
    debug_subsample: int | None = None,
    mask_think_in_loss: bool = False,
    num_proc: int | None = None,
    enable_thinking: bool | None = None,
    validation_stats: dict[str, Any] | None = None,
    record_context_crops: bool = False,
) -> Dataset:
    """Build a tokenized HuggingFace Dataset from a JSONL file.

    Tokenization stays in-process on each rank to avoid a per-rank worker
    explosion. ``num_proc`` is retained for call compatibility, not used.
    """
    if record_context_crops and validation_stats is None:
        raise ValueError("Full-corpus crop evidence requires tokenization statistics")
    if record_context_crops and debug_subsample is not None:
        raise ValueError("Full-corpus construction cannot subsample approved records")

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

    raw_temporary = None
    if record_context_crops:
        if __package__:
            from .full_corpus import full_corpus_jsonl_dataset
        else:
            from full_corpus import full_corpus_jsonl_dataset
        raw_temporary = tempfile.TemporaryDirectory(prefix="sft-full-raw-")
        try:
            raw_ds = full_corpus_jsonl_dataset(jsonl_path, Path(raw_temporary.name) / "raw.arrow")
            if not len(raw_ds):
                del raw_ds
                raise ValueError(f"Empty dataset: {jsonl_path}")
        except Exception:
            raw_temporary.cleanup()
            raise
    else:
        records = load_jsonl(jsonl_path)
        if not records:
            raise ValueError(f"Empty dataset: {jsonl_path}")
        # Check all rows before subsampling or projecting away the raw columns.
        for rec in records:
            validate_sample_weight(rec)
            rec.pop("sample_weight", None)
        if debug_subsample:
            records = records[:debug_subsample]
        raw_ds = Dataset.from_list(records)
    n_total = len(raw_ds)
    if validation_stats is not None:
        validation_stats.update({
            "records": n_total, "kept": 0, "skipped": 0,
            "left_truncated_records": 0, "context_tokens_removed": 0,
            "skip_reasons": {},
        })

    _SKIP = {"input_ids": [], "attention_mask": [], "labels": []}

    def _map_fn(rec, index):
        # tokenize_record returns None when the record should be skipped.
        # dataset.map requires consistent output columns across all rows, so
        # we return empty lists instead of None; the filter step removes them.
        stats = {} if validation_stats is not None else None
        result = tokenize_record(
            rec,
            tokenizer,
            max_seq_len,
            mask_think_in_loss=mask_think_in_loss,
            think_close_ids=think_close_ids,
            enable_thinking=enable_thinking,
            stats=stats,
        )
        if validation_stats is not None:
            if result is None:
                validation_stats["skipped"] += 1
                reason = stats["skip_reason"]
                reasons = validation_stats["skip_reasons"]
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                validation_stats["kept"] += 1
                removed = stats["context_tokens_removed"]
                validation_stats["left_truncated_records"] += int(removed > 0)
                validation_stats["context_tokens_removed"] += removed
        output = result if result is not None else dict(_SKIP)
        if validation_stats is not None:
            output["_validation_row_index"] = index
        if record_context_crops:
            output["_full_corpus_context_tokens_removed"] = stats.get("context_tokens_removed", 0)
        return output

    # New datasets versions also spawn a worker for num_proc=1.
    n_workers = None
    logger.info(
        "[SFT] Tokenizing %s (%d records) in-process to prevent multi-rank OOM",
        jsonl_path, n_total,
    )

    try:
        tokenized_ds = raw_ds.map(
            _map_fn,
            with_indices=True,
            num_proc=n_workers,
            remove_columns=raw_ds.column_names,
            desc=f"Tokenizing {os.path.basename(jsonl_path)}",
            # Keep only records where input_ids was produced (skip → empty dict)
            load_from_cache_file=validation_stats is None,
            **({"keep_in_memory": True} if record_context_crops else {}),
        )
    finally:
        del raw_ds
        if raw_temporary is not None:
            raw_temporary.cleanup()

    # Filter out skipped records (empty lists from _SKIP sentinel)
    tokenized_ds = tokenized_ds.filter(
        lambda x: len(x["input_ids"]) > 0,
        num_proc=n_workers,
        desc="Filtering skipped",
        **({"keep_in_memory": True} if record_context_crops else {}),
    )

    # Compute stats for logging
    n_kept = len(tokenized_ds)
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
            validate_sample_weight(f)
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if "_validation_row_index" in features[0]:
            batch["_validation_row_index"] = torch.tensor(
                [f["_validation_row_index"] for f in features], dtype=torch.long
            )
        return batch


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
    preds = preds[:, :-1]
    labels = labels[:, 1:]
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
    full_corpus = None
    if cfg.full_corpus_contract is not None:
        if (
            args.debug or args.smoke_test or args.max_steps != -1 or args.override_output_dir
            or cfg.bounded_validation is not None or cfg.validation_fixed_order
            or cfg.validation_checkpoint_step is not None or not cfg.deepspeed_config
            or cfg.eval_strategy != "no" or cfg.save_strategy != "no"
            or cfg.resume_from_checkpoint == "auto"
            or cfg.model_revision != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        ):
            raise ValueError(
                "Full-corpus mode requires its separate contract, pinned Qwen3.5 revision, active DS, "
                "no diagnostic/debug/max-steps overrides, no scheduled saves/eval and explicit resume"
            )
        if __package__:
            from .full_corpus import (
                COLLECTIVES, FullCorpusContract, FullCorpusMeasurements, FullCorpusTrainer,
                contract_sha256, validate_full_checkpoint,
            )
        else:
            from full_corpus import (
                COLLECTIVES, FullCorpusContract, FullCorpusMeasurements, FullCorpusTrainer,
                contract_sha256, validate_full_checkpoint,
            )
        full_corpus = FullCorpusContract(cfg.full_corpus_contract, cfg)
        if (
            isinstance(cfg.full_corpus_segment_end_step, bool)
            or not isinstance(cfg.full_corpus_segment_end_step, int)
            or cfg.full_corpus_segment_end_step not in full_corpus.value["segment_end_steps"]
        ):
            raise ValueError("Full-corpus segment end must be an explicit contracted optimizer-step boundary")
        destination = Path(cfg.output_dir) / f"checkpoint-{cfg.full_corpus_segment_end_step}"
        if destination.exists():
            raise ValueError("Refusing to overwrite an existing full-corpus segment checkpoint")
        if (Path(cfg.output_dir) / "full_corpus_inflight.json").exists():
            raise ValueError("Unresolved full-corpus in-flight update; refusing resume/replay")
        if not cfg.resume_from_checkpoint and (Path(cfg.output_dir) / "full_corpus_coverage.jsonl").exists():
            raise ValueError("Refusing to replay an existing full-corpus coverage journal")
    elif cfg.full_corpus_segment_end_step is not None:
        raise ValueError("full_corpus_segment_end_step requires full_corpus_contract")
    if not isinstance(cfg.validation_fixed_order, bool):
        raise ValueError("validation_fixed_order must be boolean")
    if cfg.deepspeed_config and not args.debug and not args.smoke_test:
        setup_deepspeed()
    validation = (
        BoundedValidationConfig.parse(cfg.bounded_validation)
        if cfg.bounded_validation is not None else None
    )
    if cfg.validation_fixed_order and validation is None:
        raise ValueError("validation_fixed_order requires bounded_validation")
    if validation is not None:
        if (
            not cfg.deepspeed_config or args.debug or args.smoke_test
            or args.max_steps != validation.expected_steps
            or (cfg.resume_from_checkpoint and cfg.validation_checkpoint_step is None)
        ):
            raise ValueError("Bounded validation requires active DeepSpeed, exact --max-steps and no resume/debug")
    if cfg.validation_checkpoint_step is not None:
        if (
            validation is None or validation.expected_world_size != 8
            or validation.expected_steps > 8
            or isinstance(cfg.validation_checkpoint_step, bool)
            or not isinstance(cfg.validation_checkpoint_step, int)
            or not validation.warmup_steps < cfg.validation_checkpoint_step < validation.expected_steps
            or cfg.resume_from_checkpoint == "auto"
            or cfg.eval_strategy != "no" or cfg.save_strategy != "no"
        ):
            raise ValueError(
                "Checkpoint validation requires world8, total steps <= 8, warmup_steps < validation_checkpoint_step < expected_steps, "
                "explicit resume path (not auto), eval_strategy/save_strategy: no"
            )
        expected_checkpoint = Path(cfg.output_dir) / f"checkpoint-{cfg.validation_checkpoint_step}"
        if cfg.resume_from_checkpoint:
            if Path(cfg.resume_from_checkpoint).resolve() != expected_checkpoint.resolve():
                raise ValueError("Checkpoint validation must resume the explicit checkpoint in the same output directory")
        elif expected_checkpoint.exists():
            raise ValueError("Refusing to overwrite an existing bounded checkpoint")
    optimizer_args = {
        key: getattr(cfg, key) for key in ("optim", "adam_beta1", "adam_beta2", "adam_epsilon")
        if getattr(cfg, key) is not None
    }
    for key in ("adam_beta1", "adam_beta2", "adam_epsilon"):
        if key in optimizer_args:
            value = optimizer_args[key]
            if (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (key != "adam_epsilon" and not 0 <= value < 1)
                or (key == "adam_epsilon" and value <= 0)
            ):
                raise ValueError(f"Invalid optimizer setting {key}")
    if validation is not None and validation.expected_world_size == 8:
        if cfg.optim not in (None, "adamw_torch"):
            raise ValueError("Eight-GPU bounded validation requires optim: adamw_torch")
        optimizer_args["optim"] = "adamw_torch"

    set_seed(cfg.seed)
    logger.info("=" * 60)
    logger.info("Config:")
    for k, v in cfg.__dict__.items():
        logger.info("  %s = %r", k, v)
    logger.info("=" * 60)

    # --- WandB ---
    if cfg.wandb_project and not args.smoke_test and not args.debug and not args.no_wandb:
        # Only enable W&B if an API key is available — avoids a fatal
        # UsageError crash on AML nodes where the key wasn't forwarded.
        if os.environ.get("WANDB_API_KEY"):
            os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
            if cfg.wandb_run_name:
                os.environ.setdefault("WANDB_NAME", cfg.wandb_run_name)
            if cfg.wandb_tags:
                os.environ.setdefault("WANDB_TAGS", ",".join(cfg.wandb_tags))
            report_to = ["wandb"]
        else:
            logger.warning(
                "WANDB_API_KEY not set — disabling W&B logging (report_to=none). "
                "Set WANDB_API_KEY in your secrets file or AML job env vars to enable."
            )
            os.environ["WANDB_DISABLED"] = "true"
            report_to = ["none"]
    else:
        report_to = ["none"]

    model_config = AutoConfig.from_pretrained(
        cfg.base_model, revision=cfg.model_revision, trust_remote_code=True
    )
    model_class = model_class_for_config(model_config, transformers)
    is_qwen35 = model_config.model_type == "qwen3_5"
    if full_corpus is not None and not is_qwen35:
        raise ValueError("Full-corpus contract requires the full Qwen3.5 model family")
    if is_qwen35 and not isinstance(cfg.enable_thinking, bool):
        raise ValueError("Qwen3.5 requires explicit enable_thinking: false or true")

    # --- Tokenizer ---
    logger.info("Loading tokenizer: %s", cfg.base_model)
    # Preserve the slow tokenizer for legacy ptca; Qwen3.5 uses its native tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, revision=cfg.model_revision,
        trust_remote_code=not is_qwen35, use_fast=is_qwen35
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Datasets ---
    debug_n = 32 if (args.debug or args.smoke_test) else None
    dataset_stats = {"train": {}, "val": {}} if validation is not None or full_corpus is not None else None
    logger.info("Building train dataset ...")
    train_ds = build_dataset(
        cfg.train_jsonl, tokenizer, cfg.max_seq_len, debug_n,
        mask_think_in_loss=cfg.mask_think_in_loss,
        enable_thinking=cfg.enable_thinking,
        validation_stats=dataset_stats["train"] if dataset_stats is not None else None,
        record_context_crops=full_corpus is not None,
    )
    logger.info("Building val dataset ...")
    val_ds = build_dataset(
        cfg.val_jsonl, tokenizer, cfg.max_seq_len, debug_n,
        mask_think_in_loss=cfg.mask_think_in_loss,
        enable_thinking=cfg.enable_thinking,
        validation_stats=dataset_stats["val"] if dataset_stats is not None else None,
    )
    if full_corpus is not None:
        full_corpus.validate_dataset(train_ds, dataset_stats["train"])

    if len(train_ds) == 0:
        logger.error("Empty train dataset")
        sys.exit(2)

    # --- TrainingArguments ---
    logger.info("Building TrainingArguments ...")
    if full_corpus is not None:
        max_steps = full_corpus.value["total_max_steps"]
        eval_strategy = cfg.eval_strategy
        save_strategy = cfg.save_strategy
        logging_steps = cfg.logging_steps
    elif args.smoke_test:
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
        **warmup_arguments(transformers.TrainingArguments, cfg.warmup_ratio),
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
        logging_nan_inf_filter=False,
        **optimizer_args,
        **({
            "ignore_data_skip": False, "data_seed": cfg.seed,
            "accelerator_config": {"use_seedable_sampler": True},
            "save_only_model": False,
        } if cfg.validation_checkpoint_step is not None or full_corpus is not None else {}),
    )
    logger.info("TrainingArguments built.")
    checkpoint_contract = None
    resume_manifest = None
    full_runtime_contract = None
    full_resume_manifest = None
    if full_corpus is not None:
        if training_args.world_size != 8:
            raise RuntimeError("Full-corpus training requires actual world size eight before loading")
        full_ds_input = json.loads(Path(cfg.deepspeed_config).read_text())
        if (
            full_ds_input.get("zero_optimization", {}).get("stage") != 3
            or "optimizer" in full_ds_input or "scheduler" in full_ds_input
            or full_ds_input.get("zero_optimization", {}).get("offload_optimizer")
            or full_ds_input.get("zero_optimization", {}).get("offload_param")
        ):
            raise ValueError("Full-corpus resume requires ZeRO3/native AdamW/external HF scheduler/no offload")
        config_identity = dict(cfg.__dict__)
        config_identity.pop("resume_from_checkpoint")
        config_identity.pop("full_corpus_segment_end_step")
        full_runtime_contract = {
            "corpus": full_corpus.value, "sft_config": config_identity,
            "tokenized_train_sha256": full_corpus.tokenized_sha256,
            "model_config": model_config.to_dict(),
            "deepspeed_input": deepspeed_config_snapshot(full_ds_input),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "accelerate", "deepspeed", "datasets", "tokenizers")
            },
            "trainer_code_sha256": {
                name: file_sha256(Path(__file__).with_name(name))
                for name in ("sft_train.py", "bounded_validation.py", "training_utils.py", "full_corpus.py")
            },
        }
        required_packages = {"torch": "2.6.0+cu124", "transformers": "5.19.0", "accelerate": "1.15.0", "deepspeed": "0.16.9"}
        if any(full_runtime_contract["packages"][name] != version for name, version in required_packages.items()):
            raise ValueError("Full-corpus contract requires the unchanged pinned torch/TF/Accelerate/DS stack")
        full_runtime_contract = json.loads(json.dumps(full_runtime_contract, allow_nan=False))
        if len(set(exchange(contract_sha256(full_runtime_contract), COLLECTIVES))) != 1:
            raise ValueError("Ranks disagree on full-corpus runtime/data/package/source contract")
        if cfg.resume_from_checkpoint:
            def validate_full_resume():
                nonlocal full_resume_manifest
                full_resume_manifest = validate_full_checkpoint(
                    Path(cfg.resume_from_checkpoint), full_runtime_contract, full_corpus,
                    verify_files=training_args.process_index == 0,
                )
                saved_step = full_resume_manifest["checkpoint_step"]
                boundaries = full_corpus.value["segment_end_steps"]
                if boundaries.index(cfg.full_corpus_segment_end_step) != boundaries.index(saved_step) + 1:
                    raise ValueError("Full-corpus resume must continue the next planned segment without replay/skips")
                if Path(cfg.resume_from_checkpoint).resolve() != (
                    Path(cfg.output_dir) / f"checkpoint-{saved_step}"
                ).resolve():
                    raise ValueError("Full-corpus resume must use the explicit same-output checkpoint")
                journal = Path(cfg.output_dir) / "full_corpus_coverage.jsonl"
                saved_journal = Path(cfg.resume_from_checkpoint) / "coverage.jsonl"
                if journal.exists() and file_sha256(journal) != file_sha256(saved_journal):
                    raise ValueError("Live coverage advanced beyond checkpoint; refusing update replay")
            collective_action(validate_full_resume, COLLECTIVES, "full-corpus checkpoint preflight")
            manifest_hashes = exchange(
                file_sha256(Path(cfg.resume_from_checkpoint) / "full_corpus_checkpoint.json"), COLLECTIVES,
            )
            if len(set(manifest_hashes)) != 1:
                raise ValueError("Ranks read different full-corpus checkpoint completion manifests")
        elif cfg.full_corpus_segment_end_step != full_corpus.value["segment_end_steps"][0]:
            raise ValueError("Fresh full-corpus training must begin at the first planned segment")
    if validation is not None and validation.expected_world_size == 8:
        if training_args.world_size != 8:
            raise RuntimeError("Expected eight distributed ranks before model loading")
        if cfg.validation_fixed_order:
            expected_rows = (
                validation.expected_steps * 8 * cfg.per_device_train_batch_size
                * cfg.gradient_accumulation_steps
            )
            if len(train_ds) != expected_rows or dataset_stats["train"]["skipped"]:
                raise ValueError(
                    f"Eight-rank fixed-order validation requires exactly {expected_rows} usable rows without skips"
                )
    if cfg.validation_checkpoint_step is not None:
        config_identity = dict(cfg.__dict__)
        config_identity.pop("resume_from_checkpoint")
        deepspeed_input = json.loads(Path(cfg.deepspeed_config).read_text())
        if "optimizer" in deepspeed_input or "scheduler" in deepspeed_input:
            raise ValueError("Checkpoint validation requires native torch AdamW and external HF scheduler, not DS config blocks")
        checkpoint_contract = {
            "config": config_identity, "max_steps": max_steps,
            "model_config": model_config.to_dict(),
            "tokenized_train_sha256": state_sha256(list(train_ds)),
            "input_sha256": {
                "train": file_sha256(Path(cfg.train_jsonl)), "val": file_sha256(Path(cfg.val_jsonl)),
            },
            "deepspeed_input": deepspeed_config_snapshot(deepspeed_input),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "accelerate", "deepspeed", "datasets", "tokenizers")
            },
            "trainer_code_sha256": {
                name: file_sha256(Path(__file__).with_name(name))
                for name in ("sft_train.py", "bounded_validation.py", "training_utils.py")
            },
        }
        # JSON normalization also prevents tuple/list representation mismatches on reload.
        checkpoint_contract = json.loads(json.dumps(checkpoint_contract, allow_nan=False))
        if len(set(exchange(state_sha256(checkpoint_contract), validation))) != 1:
            raise ValueError("Ranks disagree on the checkpoint training/data/configuration contract")
        if cfg.resume_from_checkpoint:
            def validate_resume():
                nonlocal resume_manifest
                resume_manifest = validate_checkpoint(
                    Path(cfg.resume_from_checkpoint), checkpoint_contract, validation, cfg.validation_checkpoint_step,
                    verify_files=training_args.process_index == 0,
                )
            collective_action(validate_resume, validation, "checkpoint preflight")
            manifest_hashes = exchange(
                file_sha256(Path(cfg.resume_from_checkpoint) / "bounded_checkpoint.json"), validation,
            )
            if len(set(manifest_hashes)) != 1:
                raise ValueError("Ranks read different checkpoint completion manifests")

    # TrainingArguments retains HfTrainerDeepSpeedConfig's weakref before
    # from_pretrained checks it to enter deepspeed.zero.Init for ZeRO-3.
    if validation is not None or full_corpus is not None:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
        if not is_deepspeed_zero3_enabled():
            raise RuntimeError("ZeRO-3 configuration must be active before model loading")

    logger.info("Loading model: %s (attn=%s)", cfg.base_model, cfg.attn_implementation)
    model = model_class.from_pretrained(
        cfg.base_model,
        config=model_config,
        revision=cfg.model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=not is_qwen35,
    )
    if is_qwen35:
        configure_qwen35_text_training(model)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    measurements = None
    full_measurements = None
    if full_corpus is not None:
        partitioned = sum(hasattr(parameter, "ds_id") for parameter in model.parameters())
        collective_check(
            None if partitioned else "Full-corpus model loading did not use ZeRO3 partitions",
            COLLECTIVES, "full-corpus pre-load ZeRO3",
        )
        full_measurements = FullCorpusMeasurements(
            full_corpus, cfg.output_dir, cfg.full_corpus_segment_end_step, full_runtime_contract,
            cfg.resume_from_checkpoint, full_resume_manifest,
        )
        full_measurements.report["zero3_initialization"] = {
            "active_before_from_pretrained": True, "partitioned_parameter_count": partitioned,
        }
    if validation is not None:
        partitioned = sum(hasattr(p, "ds_id") for p in model.parameters())
        collective_check(
            "ZeRO-3 model loading did not produce partitioned parameters" if not partitioned else None,
            validation, "partitioned model loading",
        )
        measurements = ValidationMeasurements(validation, cfg.output_dir)
        measurements.report.update({
            "dataset_stats": dataset_stats,
            "zero3_initialization": {
                "active_before_from_pretrained": True,
                "partitioned_parameter_count": partitioned,
            },
            "validation_fixed_order": cfg.validation_fixed_order,
        })
        if cfg.validation_checkpoint_step is not None:
            measurements.configure_checkpoint(
                cfg.validation_checkpoint_step, checkpoint_contract,
                cfg.resume_from_checkpoint, resume_manifest,
            )

    collator = DataCollatorForSFT(pad_token_id=tokenizer.pad_token_id)

    logger.info("Constructing Trainer ...")
    trainer_class = BoundedValidationTrainer if measurements is not None else Trainer
    validation_kwargs = (
        {"measurements": measurements, "fixed_order": cfg.validation_fixed_order}
        if measurements is not None else {}
    )
    if full_measurements is not None:
        trainer_class = FullCorpusTrainer
        validation_kwargs = {"measurements": full_measurements}
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if eval_strategy != "no" else None,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics if eval_strategy != "no" else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_strategy != "no" else None,
        **validation_kwargs,
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

    try:
        train_result = trainer.train(resume_from_checkpoint=resume)
        if full_measurements is not None:
            full_measurements.finish(trainer.state.global_step, train_result.training_loss)
            if (
                not full_measurements.report["corpus_complete"]
                and not full_corpus.value["export_at_segment_end"]
            ):
                logger.info("Full-state segment saved; corpus incomplete, final export deferred")
                return
        if measurements is not None:
            if cfg.validation_checkpoint_step is not None and resume_manifest is None:
                measurements.finish_checkpoint_phase(trainer.state.global_step, train_result.training_loss)
                logger.info("Full checkpoint saved; final export deferred until explicit resume phase")
                return
            measurements.finish(trainer.state.global_step, train_result.training_loss)
    except Exception as exc:
        if full_measurements is not None:
            full_measurements.fail(exc)
        if measurements is not None:
            measurements.fail(exc)
        raise
    if not math.isfinite(train_result.training_loss):
        raise RuntimeError("Training loss is not finite; refusing to export the model")

    # --- Save final ---
    distributed_validation = (
        measurements is not None and validation.expected_world_size == 8
    ) or full_measurements is not None

    def save_weights():
        # ZeRO-3's 16-bit gather requires every rank, not only rank zero.
        trainer.save_model()
        trainer.save_state()

    def save_tokenizer():
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(cfg.output_dir)

    def save_summary():
        if trainer.is_world_process_zero():
            with open(os.path.join(cfg.output_dir, "training_summary.json"), "w") as f:
                json.dump(
                    {
                        "step_key": cfg.step_key,
                        "base_model": cfg.base_model,
                        "model_revision": cfg.model_revision,
                        "model_class": model.__class__.__name__,
                        "model_type": model.config.model_type,
                        "enable_thinking": cfg.enable_thinking,
                        "train_jsonl": cfg.train_jsonl,
                        "val_jsonl": cfg.val_jsonl,
                        "best_metric": trainer.state.best_metric,
                        "best_model_checkpoint": trainer.state.best_model_checkpoint,
                        "global_step": trainer.state.global_step,
                        "training_loss": train_result.training_loss,
                        **({"bounded_validation": measurements.report} if measurements is not None else {}),
                        **({"full_corpus": full_measurements.report} if full_measurements is not None else {}),
                    },
                    f, indent=2,
                )

    logger.info(
        "Saving %s model to %s",
        "segment" if full_measurements is not None and not full_measurements.report["corpus_complete"] else "final",
        cfg.output_dir,
    )
    try:
        if distributed_validation:
            export_collectives = COLLECTIVES if full_measurements is not None else validation
            export_measurements = full_measurements if full_measurements is not None else measurements
            torch.distributed.barrier()
            collective_action(save_weights, export_collectives, "full model export")
            torch.distributed.barrier()
            collective_action(save_tokenizer, export_collectives, "rank-zero tokenizer export")
            if full_measurements is not None:
                full_measurements.record_export()
                export_measurements.report["segment_export_complete"] = True
                if full_measurements.report["corpus_complete"]:
                    export_measurements.report["export_complete"] = True
            else:
                export_measurements.report["export_complete"] = True
            export_measurements.persist()
            collective_action(save_summary, export_collectives, "rank-zero final summary")
            torch.distributed.barrier()
        else:
            save_weights()
            save_tokenizer()
            save_summary()
    except Exception as exc:
        if full_measurements is not None:
            full_measurements.fail(exc)
        if measurements is not None:
            measurements.fail(exc)
        raise
    logger.info("Done.")


if __name__ == "__main__":
    main()
