"""Opt-in bounded DeepSpeed validation evidence; not a general profiler."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import socket
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
        if config.expected_zero_stage != 3 or config.expected_world_size not in (1, 8):
            raise ValueError("bounded_validation supports only ZeRO stage 3, world size 1 or 8")
        return config


def exchange(value, config: BoundedValidationConfig):
    if config.expected_world_size == 1:
        return [value]
    if not torch.distributed.is_initialized():
        raise RuntimeError("Distributed validation requires initialized torch.distributed")
    values = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(values, value)
    if len(values) != config.expected_world_size:
        raise RuntimeError("Collective world size differs from bounded validation configuration")
    return values


def collective_check(error: str | None, config: BoundedValidationConfig, phase: str):
    errors = exchange(error, config)
    if any(errors):
        raise RuntimeError(f"{phase} failed across ranks: {errors}")


def collective_action(action, config: BoundedValidationConfig, phase: str):
    """Exchange ordinary local failures after a phase all ranks must enter."""
    error = None
    try:
        action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    collective_check(error, config, phase)


def optimizer_hyperparameters(optimizer) -> list[dict[str, Any]]:
    return json.loads(json.dumps([
        {key: group[key] for key in ("lr", "initial_lr", "weight_decay", "betas", "eps", "amsgrad", "fused")
         if key in group}
        for group in optimizer.param_groups
    ], allow_nan=False))


def deepspeed_config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Encode only Accelerate's verified logging sentinel, never measurements."""
    tag = "__deepspeed_config_float__"

    def encode(value, path):
        if isinstance(value, dict):
            if tag in value:
                raise ValueError(f"Reserved DeepSpeed configuration tag at {path!r}")
            if any(not isinstance(key, str) for key in value):
                raise ValueError(f"DeepSpeed configuration keys must be strings at {path!r}")
            return {key: encode(item, path + (key,)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item, path + (index,)) for index, item in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            if path == ("steps_per_print",) and value == math.inf:
                return {tag: "+Infinity"}
            raise ValueError(f"Unexpected nonfinite DeepSpeed configuration at {path!r}: {value!r}")
        return value

    encoded = encode(config, ())
    canonical = json.dumps(
        encoded, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )
    return {
        "resolved_hf_deepspeed_config": json.loads(canonical),
        "resolved_hf_deepspeed_config_encoding": "accelerate-steps-per-print-positive-infinity-v1",
        "resolved_hf_deepspeed_config_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
    }


def validate_topology(ranks: list[dict[str, Any]], expected_world_size: int) -> None:
    if len(ranks) != expected_world_size or sorted(r["rank"] for r in ranks) != list(range(expected_world_size)):
        raise RuntimeError("Runtime proof does not contain exactly the expected ranks")
    if expected_world_size == 8:
        uuids = [r["cuda_uuid"] for r in ranks]
        if any(not uuid for uuid in uuids) or len(set(uuids)) != 8:
            raise RuntimeError("Eight ranks must use eight distinct physical CUDA UUIDs")
        if any(
            "A100" not in r["cuda_device"] or "MIG" in r["cuda_device"].upper()
            or "MIG" in r["cuda_uuid"].upper() or r["distributed_backend"] != "nccl"
            for r in ranks
        ):
            raise RuntimeError("Eight-GPU bounded validation requires A100 devices and NCCL")
        reference = ranks[0]
        for rank in ranks:
            for key in (
                "resolved_hf_deepspeed_config", "packages", "torch_version",
                "resolved_hf_deepspeed_config_encoding", "resolved_hf_deepspeed_config_sha256",
                "zero_stage", "bf16_enabled", "gradient_accumulation_steps",
                "micro_batch_size", "train_batch_size", "optimizer_name",
                "optimizer_class", "optimizer_group_proof", "underlying_optimizer_groups",
            ):
                if rank[key] != reference[key]:
                    raise RuntimeError(f"Ranks disagree on runtime setting {key}")
            if rank["local_rank"] < 0 or not 0 <= rank["cuda_device_index"] < rank["cuda_visible_device_count"]:
                raise RuntimeError("Invalid local CUDA device identity")
        # LOCAL_RANK is a per-host launch rank, not a global GPU index.
        host_ranks = [(r["hostname"], r["local_rank"]) for r in ranks]
        if len(set(host_ranks)) != 8:
            raise RuntimeError("Duplicate per-host local ranks in distributed topology")


def nccl_collective_proof(config: BoundedValidationConfig) -> dict[str, Any]:
    rank = torch.distributed.get_rank()
    value = torch.tensor(float(rank + 1), dtype=torch.float64, device=torch.cuda.current_device())
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = config.expected_world_size * (config.expected_world_size + 1) / 2
    actual = float(value.item())
    if not math.isfinite(actual) or actual != expected:
        raise RuntimeError("Actual CUDA all-reduce failed the exact rank-sum check")
    return {"operation": "SUM", "actual": actual, "expected": expected, "device": "cuda"}


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
    config_proof = deepspeed_config_snapshot(resolved_config)
    proof = {
        "engine_class": f"{type(engine).__module__}.{type(engine).__name__}",
        "zero_stage": stage,
        "world_size": world_size,
        "rank": torch.distributed.get_rank(),
        "engine_dp_world_size": engine.dp_world_size,
        "gradient_accumulation_steps": engine.gradient_accumulation_steps(),
        "micro_batch_size": engine.train_micro_batch_size_per_gpu(),
        "train_batch_size": engine.train_batch_size(),
        "bf16_enabled": bool(engine.bfloat16_enabled()),
        **config_proof,
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
    if config.expected_world_size == 8:
        if engine.optimizer_name() != "AdamW" or not isinstance(engine.basic_optimizer, torch.optim.AdamW):
            raise RuntimeError("Eight-GPU bounded validation requires explicit AdamW, not an unsupported optimizer")
        group_proof = trainer.measurements.report.get("optimizer_group_proof")
        if not group_proof or group_proof.get("coverage_exact") is not True:
            raise RuntimeError("Missing pre-ZeRO AdamW trainable-parameter group coverage proof")
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        proof.update({
            "local_rank": int(os.environ["LOCAL_RANK"]),
            "hostname": socket.gethostname(),
            "cuda_device_index": device,
            "cuda_visible_device_count": torch.cuda.device_count(),
            "cuda_uuid": str(properties.uuid),
            "cuda_total_memory_bytes": properties.total_memory,
            "distributed_backend": str(torch.distributed.get_backend()),
            "optimizer_class": f"{type(engine.basic_optimizer).__module__}.{type(engine.basic_optimizer).__name__}",
            "optimizer_group_proof": group_proof,
            "underlying_optimizer_groups": optimizer_hyperparameters(engine.basic_optimizer),
        })
    return proof


class ValidationMeasurements(TrainerCallback):
    def __init__(self, config: BoundedValidationConfig, output_dir: str):
        self.config = config
        self.path = Path(output_dir) / "bounded_validation.json"
        self.trainer = None
        self.started = None
        self.pending: list[dict[str, Any]] = []
        self.report: dict[str, Any] = {
            "schema_version": 1 if config.expected_world_size == 1 else 2,
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
        rank = self.report.get("local_rank_global", 0)
        paths = [self.path]
        if self.config.expected_world_size == 8:
            if "local_rank_global" not in self.report:
                rank = (
                    torch.distributed.get_rank() if torch.distributed.is_initialized()
                    else int(os.environ["RANK"])
                )
                self.report["local_rank_global"] = rank
            paths = [self.path.with_name(f"bounded_validation.rank{rank}.json")]
            if rank == 0:
                paths.append(self.path)
        for path in paths:
            self._write_report(path)

    def _write_report(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)

    def on_train_begin(self, args, state, control, **kwargs):
        error = None
        proof = None
        try:
            proof = deepspeed_proof(self.trainer, self.config)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if self.config.expected_world_size == 1 and error:
            raise RuntimeError(error)
        if self.config.expected_world_size == 8:
            self.report["local_rank_global"] = torch.distributed.get_rank()
            collective_check(error, self.config, "runtime proof")
            ranks = exchange(proof, self.config)
            validate_topology(ranks, self.config.expected_world_size)
            proof = {**proof, "ranks": ranks}
            init_ranks = exchange(self.report.get("zero3_initialization"), self.config)
            if any(
                not item or item.get("active_before_from_pretrained") is not True
                or item.get("partitioned_parameter_count", 0) <= 0 for item in init_ranks
            ):
                raise RuntimeError("Every rank must prove pre-load ZeRO-3 parameter partitioning")
            self.report["zero3_initialization_ranks"] = init_ranks
            self.report["collective_proof"] = nccl_collective_proof(self.config)
        self.report["runtime"] = proof
        self.report["engine_global_steps_start"] = self.trainer.model_wrapped.global_steps
        collective_check(
            "Bounded validation must start at step zero, not resume"
            if state.global_step != 0 or self.trainer.model_wrapped.global_steps != 0
            or (self.config.expected_world_size == 8 and self.trainer.model_wrapped.skipped_steps != 0)
            else None,
            self.config, "start",
        )
        self.report["status"] = "running"
        self.persist()
        self.training_started = time.perf_counter()

    def on_step_begin(self, args, state, control, **kwargs):
        collective_check(
            "Bounded validation exceeded the configured step cap"
            if state.global_step >= self.config.expected_steps else None,
            self.config, "step begin",
        )
        torch.cuda.synchronize()
        if self.config.expected_world_size == 8:
            torch.distributed.barrier()
        torch.cuda.reset_peak_memory_stats()
        self.pending = []
        self.engine_step_start = self.trainer.model_wrapped.global_steps
        if self.config.expected_world_size == 8:
            self.engine_skipped_start = self.trainer.model_wrapped.skipped_steps
        self.started = time.perf_counter()

    def record_batch(self, inputs: dict[str, torch.Tensor], row_indices=None) -> None:
        if self.started is None:
            raise RuntimeError("Observed a training batch outside an optimizer-update interval")
        if self.config.expected_world_size == 8 and self.trainer.validation_fixed_order:
            args = self.trainer.args
            offset = (
                (len(self.report["steps"]) * args.gradient_accumulation_steps + len(self.pending))
                * self.config.expected_world_size * args.per_device_train_batch_size
                + torch.distributed.get_rank() * args.per_device_train_batch_size
            )
            expected = list(range(offset, offset + args.per_device_train_batch_size))
            collective_check(
                "Fixed-order local microbatch indices differ from the distributed shard plan"
                if row_indices is None or row_indices.tolist() != expected else None,
                self.config, "data shard before forward",
            )
        mask = inputs["attention_mask"]
        labels = inputs["labels"]
        lengths = mask.sum(dim=-1).detach().cpu().tolist()
        self.pending.append({
            "row_indices": row_indices.detach().cpu().tolist() if row_indices is not None else [],
            "input_lengths": lengths,
            "padded_sequence_length": mask.shape[-1],
            "attended_tokens": int(sum(lengths)),
            "padded_tokens": mask.numel() - int(sum(lengths)),
            # labels[:, 0] never participates in a causal next-token loss.
            "supervised_tokens": int((labels[:, 1:] != -100).sum().item()),
        })

    def record_loss(self, loss: torch.Tensor) -> None:
        value = float(loss.detach().float().item())
        collective_check(
            "Nonfinite training microbatch loss in bounded validation" if not math.isfinite(value) else None,
            self.config, "microbatch loss",
        )
        self.pending[-1]["trainer_returned_loss"] = value

    def on_step_end(self, args, state, control, **kwargs):
        collective_check(
            "Bounded validation optimizer-step ordering is invalid"
            if self.started is None or state.global_step != len(self.report["steps"]) + 1 else None,
            self.config, "step end",
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        collective_check(
            "Missing or invalid bounded optimizer-step measurements"
            if not self.pending or not math.isfinite(elapsed) or elapsed <= 0 else None,
            self.config, "measurements",
        )
        lengths = [length for batch in self.pending for length in batch["input_lengths"]]
        item = {
            "rank": self.report.get("local_rank_global", 0),
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
            "row_indices": [index for batch in self.pending for index in batch.get("row_indices", [])],
        }
        if self.config.expected_world_size == 8:
            item["engine_skipped_steps"] = self.trainer.model_wrapped.skipped_steps
            item["engine_skipped_updates"] = item["engine_skipped_steps"] - self.engine_skipped_start
            item["engine_optimizer_updates"] -= item["engine_skipped_updates"]
        rank_items = exchange(item, self.config)
        errors = []
        for rank_item in rank_items:
            if rank_item["global_step"] != state.global_step or rank_item["engine_global_steps"] != state.global_step:
                errors.append("Ranks disagree on actual engine/global step")
            if rank_item["optimizer_step_skipped"]:
                errors.append("DeepSpeed skipped an optimizer update during bounded validation")
            if self.config.expected_world_size == 8 and rank_item["engine_skipped_updates"] != 0:
                errors.append("DeepSpeed engine skipped-step counter changed")
            if rank_item["engine_optimizer_updates"] != 1:
                errors.append("DeepSpeed did not perform exactly one engine update for this optimizer step")
            if rank_item["microbatch_count"] != args.gradient_accumulation_steps:
                errors.append("Bounded validation requires a full gradient-accumulation group per optimizer step")
        if self.config.expected_world_size == 8:
            item = self.aggregate_step(rank_items, args)
        self.report["steps"].append(item)
        self.started = None
        self.persist()
        if errors:
            raise RuntimeError("; ".join(errors))

    def aggregate_step(self, ranks, args):
        if sorted(r["rank"] for r in ranks) != list(range(self.config.expected_world_size)):
            raise RuntimeError("Step measurements do not contain exactly the expected ranks")
        item = dict(ranks[0])
        item["ranks"] = ranks
        item["elapsed_seconds"] = max(r["elapsed_seconds"] for r in ranks)
        for key in ("attended_tokens", "supervised_tokens", "padded_tokens", "examples", "microbatch_count"):
            item[key] = sum(r[key] for r in ranks)
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "max_input_length"):
            item[key] = max(r[key] for r in ranks)
        for key in ("input_lengths", "row_indices"):
            item[key] = [value for rank in ranks for value in rank[key]]
        item["microbatches"] = [batch for rank in ranks for batch in rank["microbatches"]]
        if self.trainer.validation_fixed_order:
            global_batch = args.gradient_accumulation_steps * args.per_device_train_batch_size * self.config.expected_world_size
            expected = list(range((item["global_step"] - 1) * global_batch, item["global_step"] * global_batch))
            if sorted(item["row_indices"]) != expected:
                raise RuntimeError("Distributed fixed-order data indices were duplicated, skipped or reordered across updates")
        return item

    def finish(self, global_step: int, training_loss: float) -> None:
        steps = self.report["steps"]
        collective_check(
            "Bounded validation did not produce the exact expected optimizer steps"
            if global_step != self.config.expected_steps or len(steps) != global_step else
            "Bounded validation training loss is not finite" if not math.isfinite(training_loss) else None,
            self.config, "finish",
        )
        final_ranks = exchange({
            "training_loss": training_loss,
            "training_wall_seconds": time.perf_counter() - self.training_started,
            "global_step": global_step,
        }, self.config)
        measured = [step for step in steps if not step["is_warmup"]]
        seconds = sum(step["elapsed_seconds"] for step in measured)
        self.report.update({
            "status": "training_complete",
            "global_step": global_step,
            "training_loss": sum(r["training_loss"] for r in final_ranks) / len(final_ranks),
            "training_wall_seconds": max(r["training_wall_seconds"] for r in final_ranks),
            "final_ranks": final_ranks,
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

    def create_optimizer(self):
        result = super().create_optimizer()
        if self.measurements.config.expected_world_size == 8:
            if not isinstance(self.optimizer, torch.optim.AdamW):
                raise RuntimeError("Eight-GPU bounded validation must construct torch AdamW")
            expected = {id(p) for p in self.model.parameters() if p.requires_grad}
            grouped = [p for group in self.optimizer.param_groups for p in group["params"]]
            actual = [id(p) for p in grouped]
            if not expected or set(actual) != expected or len(actual) != len(expected):
                raise RuntimeError("AdamW groups must cover each trainable parameter exactly once, excluding frozen parameters")
            groups = optimizer_hyperparameters(self.optimizer)
            for group, source in zip(groups, self.optimizer.param_groups):
                group["parameter_tensors"] = len(source["params"])
                group["full_parameter_elements"] = sum(
                    int(getattr(p, "ds_numel", p.numel())) for p in source["params"]
                )
            self.measurements.report["optimizer_group_proof"] = {
                "scope": "before DeepSpeed replaces original parameters with flat partition buffers",
                "optimizer_class": f"{type(self.optimizer).__module__}.{type(self.optimizer).__name__}",
                "coverage_exact": True,
                "trainable_parameter_tensors": len(expected),
                "frozen_parameter_tensors": sum(not p.requires_grad for p in self.model.parameters()),
                "group_count": len(groups),
                "groups": groups,
            }
        return result

    def _get_train_sampler(self, train_dataset=None):
        if self.validation_fixed_order:
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
        return super()._get_train_sampler(train_dataset)

    def training_step(self, model, inputs, num_items_in_batch=None):
        row_indices = inputs.pop("_validation_row_index", None)
        self.measurements.record_batch(inputs, row_indices)
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.measurements.record_loss(loss)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("_validation_row_index", None)
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if model.training:
            value = loss[0] if return_outputs else loss
            collective_check(
                "Nonfinite forward loss before backward" if not bool(torch.isfinite(value).all().item()) else None,
                self.measurements.config, "forward loss",
            )
        return loss
