"""Opt-in, single-rank DeepSpeed validation evidence; not a general profiler."""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainerCallback


@dataclass(frozen=True)
class BoundedValidationConfig:
    expected_steps: int = 8
    warmup_steps: int = 2
    expected_zero_stage: int = 3
    expected_world_size: int = 1

    @classmethod
    def parse(cls, value: dict[str, Any]) -> "BoundedValidationConfig":
        config = cls(**value)
        for key, item in asdict(config).items():
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"bounded_validation.{key} must be an integer")
        if not 1 <= config.expected_steps <= 10:
            raise ValueError("bounded_validation expected_steps must be between 1 and 10")
        if not 0 <= config.warmup_steps < config.expected_steps:
            raise ValueError("bounded_validation warmup_steps must leave measured steps")
        if config.expected_zero_stage != 3 or config.expected_world_size != 1:
            raise ValueError("bounded_validation supports only ZeRO stage 3, world size 1")
        return config


def deepspeed_proof(trainer, config: BoundedValidationConfig) -> dict[str, Any]:
    import deepspeed

    engine = trainer.model_wrapped
    if not trainer.is_deepspeed_enabled or not isinstance(engine, deepspeed.DeepSpeedEngine):
        raise RuntimeError("Requested DeepSpeed validation but actual DeepSpeedEngine is inactive")
    if not torch.distributed.is_initialized():
        raise RuntimeError("DeepSpeed validation requires initialized torch.distributed")
    world_size = torch.distributed.get_world_size()
    stage = int(engine.zero_optimization_stage())
    if stage != config.expected_zero_stage or world_size != config.expected_world_size:
        raise RuntimeError(f"DeepSpeed runtime mismatch: stage={stage}, world_size={world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("Bounded DeepSpeed measurements require an actual CUDA device")
    if not engine.bfloat16_enabled():
        raise RuntimeError("Bounded DeepSpeed validation requires actual BF16 execution")
    if (
        engine.dp_world_size != config.expected_world_size
        or engine.gradient_accumulation_steps() != trainer.args.gradient_accumulation_steps
        or engine.train_micro_batch_size_per_gpu() != trainer.args.per_device_train_batch_size
        or engine.train_batch_size() != (
            trainer.args.gradient_accumulation_steps
            * trainer.args.per_device_train_batch_size * world_size
        )
    ):
        raise RuntimeError("DeepSpeed engine batch/parallel settings differ from TrainingArguments")
    # This public HF/Accelerate config is finalized before engine preparation.
    # It records resolved input values, not every engine-internal default.
    resolved_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config.config
    resolved_config = json.loads(json.dumps(resolved_config, allow_nan=False))
    return {
        "engine_class": f"{type(engine).__module__}.{type(engine).__name__}",
        "zero_stage": stage,
        "world_size": world_size,
        "rank": torch.distributed.get_rank(),
        "engine_dp_world_size": engine.dp_world_size,
        "gradient_accumulation_steps": engine.gradient_accumulation_steps(),
        "micro_batch_size": engine.train_micro_batch_size_per_gpu(),
        "train_batch_size": engine.train_batch_size(),
        "bf16_enabled": bool(engine.bfloat16_enabled()),
        "resolved_hf_deepspeed_config": resolved_config,
        "engine_optimizer_class": f"{type(engine.optimizer).__module__}.{type(engine.optimizer).__name__}",
        "optimizer_name": engine.optimizer_name(),
        "deepspeed_version": deepspeed.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "accelerate", "deepspeed")
        },
    }


class ValidationMeasurements(TrainerCallback):
    def __init__(self, config: BoundedValidationConfig, output_dir: str):
        self.config = config
        self.path = Path(output_dir) / "bounded_validation.json"
        self.trainer = None
        self.started = None
        self.pending: list[dict[str, Any]] = []
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "status": "pending",
            "config": asdict(config),
            "timing_scope": (
                "CUDA-synchronized optimizer-update interval: forward/backward, "
                "gradient accumulation, optimizer update and counter overhead; "
                "excludes dataset preparation, batch prefetch, logging, eval and export"
            ),
            "steps": [],
        }

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, self.path)

    def on_train_begin(self, args, state, control, **kwargs):
        self.report["runtime"] = deepspeed_proof(self.trainer, self.config)
        self.report["engine_global_steps_start"] = self.trainer.model_wrapped.global_steps
        if state.global_step != 0 or self.trainer.model_wrapped.global_steps != 0:
            raise RuntimeError("Bounded validation must start at step zero, not resume")
        self.report["status"] = "running"
        self.persist()
        self.training_started = time.perf_counter()

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step >= self.config.expected_steps:
            raise RuntimeError("Bounded validation exceeded the configured step cap")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.pending = []
        self.engine_step_start = self.trainer.model_wrapped.global_steps
        self.started = time.perf_counter()

    def record_batch(self, inputs: dict[str, torch.Tensor]) -> None:
        if self.started is None:
            raise RuntimeError("Observed a training batch outside an optimizer-update interval")
        mask = inputs["attention_mask"]
        labels = inputs["labels"]
        lengths = mask.sum(dim=-1).detach().cpu().tolist()
        self.pending.append({
            "input_lengths": lengths,
            "padded_sequence_length": mask.shape[-1],
            "attended_tokens": int(sum(lengths)),
            "padded_tokens": mask.numel() - int(sum(lengths)),
            # labels[:, 0] never participates in a causal next-token loss.
            "supervised_tokens": int((labels[:, 1:] != -100).sum().item()),
        })

    def record_loss(self, loss: torch.Tensor) -> None:
        value = float(loss.detach().float().item())
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite training microbatch loss in bounded validation")
        self.pending[-1]["trainer_returned_loss"] = value

    def on_step_end(self, args, state, control, **kwargs):
        if self.started is None or state.global_step != len(self.report["steps"]) + 1:
            raise RuntimeError("Bounded validation optimizer-step ordering is invalid")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        if not self.pending or not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("Missing or invalid bounded optimizer-step measurements")
        lengths = [length for batch in self.pending for length in batch["input_lengths"]]
        item = {
            "global_step": state.global_step,
            "is_warmup": state.global_step <= self.config.warmup_steps,
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "optimizer_step_skipped": bool(self.trainer.accelerator.optimizer_step_was_skipped),
            "microbatch_count": len(self.pending),
            "examples": len(lengths),
            "input_lengths": lengths,
            "max_input_length": max(lengths),
            "attended_tokens": sum(batch["attended_tokens"] for batch in self.pending),
            "supervised_tokens": sum(batch["supervised_tokens"] for batch in self.pending),
            "padded_tokens": sum(batch["padded_tokens"] for batch in self.pending),
            "microbatches": self.pending,
            "engine_global_steps": self.trainer.model_wrapped.global_steps,
            "engine_optimizer_updates": self.trainer.model_wrapped.global_steps - self.engine_step_start,
        }
        self.report["steps"].append(item)
        self.started = None
        self.persist()
        if item["optimizer_step_skipped"]:
            raise RuntimeError("DeepSpeed skipped an optimizer update during bounded validation")
        if item["engine_optimizer_updates"] != 1:
            raise RuntimeError("DeepSpeed did not perform exactly one engine update for this optimizer step")
        if item["microbatch_count"] != args.gradient_accumulation_steps:
            raise RuntimeError("Bounded validation requires a full gradient-accumulation group per optimizer step")

    def finish(self, global_step: int, training_loss: float) -> None:
        steps = self.report["steps"]
        if global_step != self.config.expected_steps or len(steps) != global_step:
            raise RuntimeError("Bounded validation did not produce the exact expected optimizer steps")
        if not math.isfinite(training_loss):
            raise RuntimeError("Bounded validation training loss is not finite")
        measured = [step for step in steps if not step["is_warmup"]]
        seconds = sum(step["elapsed_seconds"] for step in measured)
        self.report.update({
            "status": "training_complete",
            "global_step": global_step,
            "training_loss": training_loss,
            "training_wall_seconds": time.perf_counter() - self.training_started,
            "measured_steps": len(measured),
            "measured_seconds": seconds,
            "attended_tokens_per_second": sum(s["attended_tokens"] for s in measured) / seconds,
            "supervised_tokens_per_second": sum(s["supervised_tokens"] for s in measured) / seconds,
            "examples_per_second": sum(s["examples"] for s in measured) / seconds,
            "max_peak_allocated_bytes": max(s["peak_allocated_bytes"] for s in steps),
            "max_peak_reserved_bytes": max(s["peak_reserved_bytes"] for s in steps),
        })
        self.persist()

    def fail(self, exc: Exception) -> None:
        self.report.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        self.persist()


class BoundedValidationTrainer(Trainer):
    def __init__(self, *args, measurements: ValidationMeasurements, fixed_order: bool, **kwargs):
        self.measurements = measurements
        self.validation_fixed_order = fixed_order
        super().__init__(*args, **kwargs)
        measurements.trainer = self
        self.add_callback(measurements)

    def _get_train_sampler(self, train_dataset=None):
        if self.validation_fixed_order:
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
        return super()._get_train_sampler(train_dataset)

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.measurements.record_batch(inputs)
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.measurements.record_loss(loss)
        return loss
