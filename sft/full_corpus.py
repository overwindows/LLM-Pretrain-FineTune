"""Explicit full-corpus contracts and checkpoint-separated coverage evidence."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

import torch
from torch.utils.data import Sampler
from transformers import Trainer, TrainerCallback

if __package__:
    from .bounded_validation import (
        BoundedValidationConfig, atomic_json, checkpoint_inventory, checkpoint_state_proof,
        collective_action, collective_check, deepspeed_proof, exchange, file_sha256,
        nccl_collective_proof, optimizer_hyperparameters, rng_sha256, validate_topology,
    )
else:
    from bounded_validation import (
        BoundedValidationConfig, atomic_json, checkpoint_inventory, checkpoint_state_proof,
        collective_action, collective_check, deepspeed_proof, exchange, file_sha256,
        nccl_collective_proof, optimizer_hyperparameters, rng_sha256, validate_topology,
    )


COLLECTIVES = BoundedValidationConfig(expected_world_size=8)
FREQUENCIES = {"gem": 1, "hpf": 2, "mai_profile": 1}
SAMPLER_ALGORITHM = "torch-randperm-seed-plus-pass-v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def full_corpus_raw_dataset(records, arrow_path=None):
    """Construct the approved raw text schema with 64-bit byte/child offsets."""
    import pyarrow as pa
    from datasets import Dataset

    for record in records:
        exact_keys(record, {"task", "messages"}, "full-corpus raw row")
        if record["task"] not in FREQUENCIES or not isinstance(record["messages"], list):
            raise ValueError("Invalid full-corpus raw task/messages")
        for message in record["messages"]:
            exact_keys(message, {"role", "content"}, "full-corpus raw message")
            if not isinstance(message["role"], str) or not isinstance(message["content"], str):
                raise ValueError("Full-corpus raw messages require exact text role/content")
    schema = pa.schema([
        pa.field("task", pa.large_string()),
        pa.field("messages", pa.large_list(pa.struct([
            pa.field("role", pa.large_string()), pa.field("content", pa.large_string()),
        ]))),
    ])
    if arrow_path is None:
        return Dataset(pa.Table.from_pylist(records, schema=schema))
    # Standard memory-mapped Dataset fingerprinting avoids pickling the entire
    # multi-GiB raw payload while preserving every record in native Arrow.
    with pa.OSFile(str(arrow_path), "wb") as stream, pa.ipc.new_stream(stream, schema) as writer:
        for start in range(0, len(records), 1024):
            writer.write_batch(pa.RecordBatch.from_pylist(records[start:start + 1024], schema=schema))
    return Dataset.from_file(str(arrow_path))


def strict_json_text(text):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=invalid, object_pairs_hook=pairs)


def strict_json(path):
    return strict_json_text(Path(path).read_text(encoding="utf-8"))


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Invalid {label} fields; required: {sorted(keys)}")


def positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def verify_file(value):
    exact_keys(value, {"path", "sha256"}, "file identity")
    if not isinstance(value["path"], str) or not SHA256.fullmatch(value["sha256"]):
        raise ValueError("Invalid file path/SHA256 identity")
    path = Path(value["path"])
    if not path.is_file() or file_sha256(path) != value["sha256"]:
        raise ValueError(f"File identity mismatch: {path}")
    return path


class PassSampler(Sampler):
    def __init__(self, size, seed):
        self.size, self.seed, self.epoch = size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def permutation(self):
        return torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch))

    def __iter__(self):
        return iter(self.permutation().tolist())

    def __len__(self):
        return self.size


def permutation_sha256(permutation):
    if permutation.dtype != torch.int64 or permutation.ndim != 1 or permutation.device.type != "cpu":
        raise ValueError("Permutation digest requires a contiguous CPU int64 vector")
    return hashlib.sha256(permutation.contiguous().numpy().tobytes()).hexdigest()


def contract_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()


class FullCorpusContract:
    def __init__(self, path, cfg):
        self.path = Path(path)
        self.value = strict_json(path)
        value = self.value
        exact_keys(value, {
            "schema_version", "model", "input_files", "artifacts", "source_counts", "frequencies",
            "exposure_counts", "full_passes", "total_max_steps", "segment_end_steps",
            "training", "sampler", "quality_train", "export_at_segment_end",
        }, "full-corpus contract")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["quality_train"] is not False:
            raise ValueError("Full-corpus contract must be schema1 with Quality evaluation-only")
        if not isinstance(value["export_at_segment_end"], bool):
            raise ValueError("export_at_segment_end must be an explicit boolean")
        exact_keys(value["model"], {"base_model", "revision"}, "model")
        if value["model"] != {"base_model": cfg.base_model, "revision": cfg.model_revision}:
            raise ValueError("Model path/revision differs from full-corpus contract")
        exact_keys(value["input_files"], {"train", "val", "ledger"}, "inputs")
        self.inputs = {name: verify_file(identity) for name, identity in value["input_files"].items()}
        if self.inputs["train"].resolve() != Path(cfg.train_jsonl).resolve() or self.inputs["val"].resolve() != Path(cfg.val_jsonl).resolve():
            raise ValueError("Training/validation paths differ from full-corpus contract")
        exact_keys(value["artifacts"], {"sources", "instructions", "tokenizer"}, "artifacts")
        if not set(FREQUENCIES).issubset(value["artifacts"]["sources"]):
            raise ValueError("Source artifacts must bind each of gem/hpf/mai_profile explicitly")
        self.sources = {}
        for kind, artifacts in value["artifacts"].items():
            if not isinstance(artifacts, dict) or not artifacts:
                raise ValueError(f"Missing {kind} artifact identities")
            for identity in artifacts.values():
                artifact = verify_file(identity)
                if kind == "sources":
                    self.sources[identity["sha256"]] = artifact
                if kind == "tokenizer" and not artifact.resolve().is_relative_to(Path(cfg.base_model).resolve()):
                    raise ValueError("Tokenizer artifacts must bind the actual local base-model directory")
        tokenizer_files = {
            path.resolve() for path in Path(cfg.base_model).rglob("*")
            if path.is_file() and (
                path.name in {
                    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                    "vocab.json", "merges.txt", "tokenizer.model", "chat_template.jinja",
                } or path.parent.name == "chat_templates" and path.suffix == ".jinja"
            )
        }
        bound_tokenizer_files = {Path(identity["path"]).resolve() for identity in value["artifacts"]["tokenizer"].values()}
        if not tokenizer_files or not tokenizer_files.issubset(bound_tokenizer_files):
            raise ValueError("Contract must bind every local tokenizer/chat-template asset")
        if value["frequencies"] != FREQUENCIES:
            raise ValueError("Full-corpus frequencies must be GEM1/HPF2/MAI1")
        if any(type(frequency) is not int for frequency in value["frequencies"].values()):
            raise ValueError("Frequencies must be exact integers")
        for key in ("source_counts", "exposure_counts"):
            exact_keys(value[key], FREQUENCIES, key)
            for task, count in value[key].items():
                positive_integer(count, f"{key}.{task}")
        if value["exposure_counts"] != {
            task: value["source_counts"][task] * frequency for task, frequency in FREQUENCIES.items()
        }:
            raise ValueError("Exposure counts differ from materialized source frequencies")
        positive_integer(value["full_passes"], "full_passes")
        positive_integer(value["total_max_steps"], "total_max_steps")
        if cfg.num_train_epochs != value["full_passes"]:
            raise ValueError("num_train_epochs must equal the explicit full_passes contract")
        training_fields = {
            "world_size", "per_device_train_batch_size", "gradient_accumulation_steps", "seed",
            "learning_rate", "weight_decay", "optim", "adam_beta1", "adam_beta2", "adam_epsilon",
            "lr_scheduler_type", "warmup_ratio", "max_seq_len", "enable_thinking", "mask_think_in_loss",
        }
        exact_keys(value["training"], training_fields, "training")
        expected_training = {name: getattr(cfg, name) for name in training_fields - {"world_size"}}
        expected_training["world_size"] = 8
        if value["training"] != expected_training:
            raise ValueError("Training configuration differs from the full-corpus contract")
        for key in ("world_size", "per_device_train_batch_size", "gradient_accumulation_steps", "max_seq_len"):
            positive_integer(value["training"][key], key)
        if type(value["training"]["seed"]) is not int:
            raise ValueError("seed must be an integer")
        for key in ("learning_rate", "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon", "warmup_ratio"):
            parameter = value["training"][key]
            if isinstance(parameter, bool) or not isinstance(parameter, (float, int)) or not math.isfinite(parameter):
                raise ValueError(f"{key} must be finite numeric")
        if (
            cfg.learning_rate <= 0 or cfg.weight_decay < 0 or cfg.adam_epsilon <= 0
            or not 0 <= cfg.adam_beta1 < 1 or not 0 <= cfg.adam_beta2 < 1
            or not 0 <= cfg.warmup_ratio < 1 or cfg.lr_scheduler_type != "cosine"
        ):
            raise ValueError("Invalid full-corpus AdamW/cosine scheduler settings")
        if (
            cfg.per_device_train_batch_size != 1 or cfg.gradient_accumulation_steps != 4
            or cfg.optim != "adamw_torch" or cfg.enable_thinking is not False
            or cfg.mask_think_in_loss is not False or cfg.seed != 42
        ):
            raise ValueError("Full-corpus contract requires world8/micro1/accum4/AdamW/seed42/nonthinking")
        self.size = sum(value["exposure_counts"].values())
        self.global_batch = 8 * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
        if self.size % self.global_batch:
            raise ValueError("Full-corpus remainder is unsupported: no padding, dropping or partial accumulation")
        self.steps_per_pass = self.size // self.global_batch
        if value["total_max_steps"] != self.steps_per_pass * value["full_passes"]:
            raise ValueError("Total scheduler horizon does not cover exact whole weighted passes")
        boundaries = value["segment_end_steps"]
        if not isinstance(boundaries, list) or not boundaries:
            raise ValueError("Explicit segment boundaries are required")
        for step in boundaries:
            positive_integer(step, "segment_end_steps")
        if boundaries != sorted(set(boundaries)) or boundaries[-1] != value["total_max_steps"]:
            raise ValueError("Segment boundaries must increase and end at the unchanged total horizon")
        exact_keys(value["sampler"], {"algorithm", "permutation_sha256"}, "sampler")
        if value["sampler"]["algorithm"] != SAMPLER_ALGORITHM:
            raise ValueError("Unsupported full-corpus sampler algorithm")
        self.sampler = PassSampler(self.size, cfg.seed)
        digests = []
        for epoch in range(value["full_passes"]):
            self.sampler.set_epoch(epoch)
            digests.append(permutation_sha256(self.sampler.permutation()))
        self.sampler.set_epoch(0)
        if digests != value["sampler"]["permutation_sha256"]:
            raise ValueError("Actual per-pass permutation differs from the contract")
        self._validate_ledger()

    def _validate_ledger(self):
        unique = {task: Counter() for task in FREQUENCIES}
        exposures = Counter()
        self.rows = []
        with ExitStack() as stack:
            sources = {sha: stack.enter_context(path.open("rb")) for sha, path in self.sources.items()}
            with self.inputs["ledger"].open(encoding="utf-8") as ledger:
                for index, line in enumerate(ledger):
                    if not line.strip():
                        raise ValueError("Blank ledger rows are not allowed")
                    row = strict_json_text(line)
                    exact_keys(row, {"task", "source_id", "exposure_index"}, "ledger row")
                    task = row["task"]
                    if task not in FREQUENCIES or type(row["exposure_index"]) is not int or row["exposure_index"] != index:
                        raise ValueError("Ledger exposure indices/tasks are not exact")
                    identity = row["source_id"]
                    if not isinstance(identity, str):
                        raise ValueError("Ledger source_id must be a reversible string")
                    parts = identity.split(":")
                    if (
                        len(parts) != 4 or parts[0] != task
                        or parts[1] != self.value["artifacts"]["sources"][task]["sha256"]
                        or not parts[2].isdigit() or str(int(parts[2])) != parts[2]
                        or not SHA256.fullmatch(parts[3])
                    ):
                        raise ValueError("Invalid source_id: expected TASK:sourceSHA:byteOffset:rawLineSHA")
                    unique[task][identity] += 1
                    exposures[task] += 1
                    self.rows.append(row)
            for task, counts in unique.items():
                sha = self.value["artifacts"]["sources"][task]["sha256"]
                stream = sources[sha]
                stream.seek(0)
                retained = 0
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    raw = line.rstrip(b"\r\n")
                    identity = f"{task}:{sha}:{offset}:{hashlib.sha256(raw).hexdigest()}"
                    if not raw or identity not in counts:
                        raise ValueError("Every retained raw source record must appear in the full-corpus ledger")
                    retained += 1
                if retained != len(counts):
                    raise ValueError("Ledger contains an invalid raw-record offset/identity")
        if len(self.rows) != self.size or dict(exposures) != self.value["exposure_counts"]:
            raise ValueError("Ledger physical exposure counts differ from the contract")
        if {task: len(counts) for task, counts in unique.items()} != self.value["source_counts"]:
            raise ValueError("Ledger distinct source counts differ from the contract")
        if any(count != FREQUENCIES[task] for task, counts in unique.items() for count in counts.values()):
            raise ValueError("Ledger replicas do not follow exact per-source frequencies")

    def indices(self, step):
        epoch, offset = divmod(step - 1, self.steps_per_pass)
        if epoch >= self.value["full_passes"] or step <= 0:
            raise ValueError("Optimizer step is outside the full-corpus horizon")
        if getattr(self, "_epoch", None) != epoch:
            self.sampler.set_epoch(epoch)
            self._permutation = self.sampler.permutation()
            self._epoch = epoch
        start = offset * self.global_batch
        return self._permutation[start:start + self.global_batch].tolist()

    def validate_dataset(self, dataset, stats):
        if len(dataset) != self.size or stats["records"] != self.size or stats["kept"] != self.size or stats["skipped"]:
            raise ValueError("Full-corpus training cannot skip or subsample any approved exposure")
        preprocessing = {task: {
            "exposures": 0, "context_cropped_exposures": 0, "context_tokens_removed": 0,
            "supervised_tokens": 0,
        } for task in FREQUENCIES}
        digest = hashlib.sha256()
        for index, row in enumerate(dataset):
            if row["_validation_row_index"] != index:
                raise ValueError("Tokenized exposure indices changed")
            labels = row["labels"]
            if (
                len(row["input_ids"]) < 2 or len(labels) != len(row["input_ids"])
                or labels[0] != -100 or not any(label != -100 for label in labels[1:])
            ):
                raise ValueError("Every full-corpus exposure needs intact supervised target tokens")
            removed = row["_full_corpus_context_tokens_removed"]
            if type(removed) is not int or removed < 0:
                raise ValueError("Missing/invalid exact context crop evidence")
            task_stats = preprocessing[self.rows[index]["task"]]
            task_stats["exposures"] += 1
            task_stats["context_cropped_exposures"] += int(removed > 0)
            task_stats["context_tokens_removed"] += removed
            task_stats["supervised_tokens"] += sum(label != -100 for label in labels[1:])
            digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii"))
            digest.update(b"\n")
        self.dataset_stats = stats
        if (
            sum(item["context_cropped_exposures"] for item in preprocessing.values()) != stats["left_truncated_records"]
            or sum(item["context_tokens_removed"] for item in preprocessing.values()) != stats["context_tokens_removed"]
        ):
            raise ValueError("Per-exposure crop evidence differs from tokenization totals")
        self.preprocessing_per_task = preprocessing
        self.tokenized_sha256 = digest.hexdigest()


class CoverageCursor:
    def __init__(self, contract):
        self.contract = contract
        self.step = 0
        self.digests = [hashlib.sha256() for _ in range(8)]
        self.task_exposures = Counter()
        self.task_sources = {task: set() for task in FREQUENCIES}

    def advance(self, item):
        step = self.step + 1
        if type(item["global_step"]) is not int or item["global_step"] != step:
            raise ValueError("Completed optimizer-step journal is not contiguous")
        expected = self.contract.indices(step)
        ranks = item["ranks"]
        if [rank["rank"] for rank in ranks] != list(range(8)):
            raise ValueError("Coverage journal requires ordered proof from all eight ranks")
        for rank, evidence in enumerate(ranks):
            indices = expected[rank::8]
            if (
                any(type(index) is not int for index in evidence["row_indices"])
                or evidence["row_indices"] != indices or type(evidence["engine_global_steps"]) is not int
                or evidence["engine_global_steps"] != step or type(evidence["rank"]) is not int
            ):
                raise ValueError("Completed-update row coverage differs from the contracted permutation")
            if (
                type(evidence["optimizer_updates"]) is not int or type(evidence["skipped_updates"]) is not int
                or evidence["optimizer_updates"] != 1 or evidence["skipped_updates"] != 0
            ):
                raise ValueError("Coverage requires one actual unskipped optimizer update per rank")
            if any(
                isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss)
                for loss in evidence["losses"]
            ) or len(evidence["losses"]) != 4:
                raise ValueError("Coverage requires four finite microbatch losses per rank")
        for rank in range(8):
            self.digests[rank].update(json.dumps(
                {"step": step, "indices": expected[rank::8]}, separators=(",", ":"),
            ).encode("ascii"))
        for index in expected:
            row = self.contract.rows[index]
            self.task_exposures[row["task"]] += 1
            self.task_sources[row["task"]].add(row["source_id"])
        self.step = step

    def consumed_counts(self):
        return {
            "source_counts": {task: len(sources) for task, sources in self.task_sources.items()},
            "exposure_counts": {task: self.task_exposures[task] for task in FREQUENCIES},
        }

    def rank_state(self, rank):
        completed, partial = divmod(self.step, self.contract.steps_per_pass)
        return {
            "rank": rank, "global_step": self.step, "completed_full_passes": completed,
            "partial_pass_optimizer_steps": partial,
            "consumed_exposures": self.step * 4,
            "coverage_sha256": self.digests[rank].hexdigest(),
        }

    def restore_journal(self, path, expected_step):
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                self.advance(strict_json_text(line))
        if self.step != expected_step:
            raise ValueError("Coverage journal does not end at the saved checkpoint step")


def full_inventory(path, step):
    inventory = checkpoint_inventory(path, 8, step)
    inventory.pop("full_corpus_checkpoint.json", None)
    if "coverage.jsonl" not in inventory:
        raise ValueError("Missing checkpoint coverage journal")
    return inventory


def validate_full_checkpoint(path, runtime_contract, contract, verify_files=True, for_resume=True):
    manifest = strict_json(Path(path) / "full_corpus_checkpoint.json")
    step = manifest["checkpoint_step"]
    if (
        manifest["schema_version"] != 1 or manifest["runtime_contract"] != runtime_contract
        or step not in contract.value["segment_end_steps"]
        or for_resume and step >= contract.value["total_max_steps"]
    ):
        raise ValueError("Incompatible full-corpus checkpoint/unchanged scheduler horizon")
    if verify_files and full_inventory(Path(path), step) != manifest["files"]:
        raise ValueError("Full-corpus checkpoint inventory/hash mismatch")
    if strict_json(Path(path) / "trainer_state.json")["max_steps"] != contract.value["total_max_steps"]:
        raise ValueError("Saved full-corpus scheduler horizon changed")
    cursor = CoverageCursor(contract)
    cursor.restore_journal(Path(path) / "coverage.jsonl", step)
    for rank in range(8):
        proof = strict_json(Path(path) / f"checkpoint_rank{rank}.json")
        if proof["rank"] != rank or proof["coverage"] != cursor.rank_state(rank):
            raise ValueError("Saved rank coverage cursor does not match completed-update journal")
    return manifest


def segment_export_inventory(path):
    path = Path(path)
    files = [
        file for file in path.rglob("*") if file.is_file() and (
            file.parent == path and (
                file.name in {"config.json", "generation_config.json", "model.safetensors.index.json"}
                or file.name.startswith("tokenizer") or file.name in {"special_tokens_map.json", "added_tokens.json", "chat_template.jinja"}
                or file.suffix == ".safetensors"
            ) or file.parent.name == "chat_templates" and file.suffix == ".jinja"
        )
    ]
    names = {file.relative_to(path).as_posix() for file in files}
    if not {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(names):
        raise ValueError("Missing native HF segment export model/generation/tokenizer files")
    if not any(name.endswith(".safetensors") for name in names) or not any(name.endswith(".jinja") for name in names):
        raise ValueError("Missing native HF segment export weights/chat template")
    return {file.relative_to(path).as_posix(): {
        "bytes": file.stat().st_size, "sha256": file_sha256(file),
    } for file in sorted(files)}


def verify_full_segment(output_dir, contract, expected_runtime_contract=None):
    """Read-only raw-file and completed-ID verifier, not GPU-state reconstruction."""
    directory = Path(output_dir)
    report = strict_json(directory / "full_corpus.json")
    step = report["global_step"]
    if type(step) is not int or step not in contract.value["segment_end_steps"]:
        raise ValueError("Segment report has an uncontracted optimizer boundary")
    checkpoint = directory / f"checkpoint-{step}"
    marker = strict_json(checkpoint / "full_corpus_checkpoint.json")
    runtime = marker["runtime_contract"]
    if runtime.get("corpus") != contract.value or (
        expected_runtime_contract is not None and runtime != expected_runtime_contract
    ):
        raise ValueError("Segment runtime/corpus contract differs from the audited expectation")
    validate_full_checkpoint(checkpoint, runtime, contract, for_resume=False)
    if (
        report["contract_sha256"] != contract_sha256(contract.value)
        or report["runtime_contract_sha256"] != contract_sha256(runtime)
        or report["checkpoint_manifest_sha256"] != file_sha256(checkpoint / "full_corpus_checkpoint.json")
        or report["segment_end_step"] != step or report["total_max_steps"] != contract.value["total_max_steps"]
    ):
        raise ValueError("Segment report contract/checkpoint/horizon identity mismatch")
    complete = step == contract.value["total_max_steps"]
    position = contract.value["segment_end_steps"].index(step)
    expected_start = contract.value["segment_end_steps"][position - 1] if position else 0
    if (
        report["corpus_complete"] is not complete
        or report["status"] != ("corpus_complete" if complete else "segment_complete")
        or report["start_global_step"] != expected_start
        or report["completed_full_passes"] != step // contract.steps_per_pass
    ):
        raise ValueError("Incomplete/full-corpus completion flags do not match actual horizon")
    cursor = CoverageCursor(contract)
    cursor.restore_journal(checkpoint / "coverage.jsonl", step)
    if report["consumed_counts"] != cursor.consumed_counts():
        raise ValueError("Reported source/task exposure counts differ from actual completed-update replay")
    coverage = [cursor.rank_state(rank) for rank in range(8)]
    saved = [strict_json(checkpoint / f"checkpoint_rank{rank}.json") for rank in range(8)]
    if report["coverage_ranks"] != coverage or report["saved_state_ranks"] != saved or marker["saved_state_ranks"] != saved:
        raise ValueError("Report rank state/coverage hashes do not match the immutable checkpoint")
    for rank, proof in enumerate(saved):
        state = proof["state"]
        if (
            proof["rank"] != rank or state["global_step"] != step
            or state["optimizer_partition_count"] != len(state["optimizer_steps"])
            or not state["optimizer_steps"] or any(type(value) is not int or value != step for value in state["optimizer_steps"])
            or any(not SHA256.fullmatch(state[key]) for key in ("model_sha256", "optimizer_sha256", "scheduler_sha256"))
            or not SHA256.fullmatch(proof["rng_sha256"])
        ):
            raise ValueError("Missing/inconsistent saved actual rank-state fingerprints/counters")
        rank_report = strict_json(directory / f"full_corpus.rank{rank}.json")
        for key in (
            "status", "contract_sha256", "runtime_contract_sha256", "global_step", "corpus_complete",
            "checkpoint_manifest_sha256", "coverage_ranks", "saved_state_ranks", "consumed_counts",
        ):
            if rank_report[key] != report[key]:
                raise ValueError(f"Rank {rank} segment report differs on {key}")
    if complete:
        if (
            report["full_pass_source_counts"] != contract.value["source_counts"]
            or report["full_pass_exposure_counts"] != contract.value["exposure_counts"]
            or report["total_completed_exposures"] != contract.size * contract.value["full_passes"]
        ):
            raise ValueError("Final source/exposure counts differ from exact replayed full passes")
    if complete or contract.value["export_at_segment_end"]:
        if complete and report.get("export_complete") is not True:
            raise ValueError("Final native HF export did not complete")
        if report.get("segment_export_complete") is not True:
            raise ValueError("Requested native segment export did not complete")
        if report["segment_export_files"] != segment_export_inventory(directory):
            raise ValueError("Native HF segment export file inventory/hash mismatch")
        for rank in range(8):
            rank_report = strict_json(directory / f"full_corpus.rank{rank}.json")
            if rank_report.get("segment_export_files") != report["segment_export_files"] or rank_report.get("segment_export_complete") is not True:
                raise ValueError("Rank reports disagree on native export hashes/completion")
    return {
        "global_step": step, "corpus_complete": complete, "coverage_ranks": coverage,
        "completed_exposures": step * contract.global_batch,
        "consumed_counts": cursor.consumed_counts(),
        "checkpoint_manifest_sha256": report["checkpoint_manifest_sha256"],
        "segment_export_files": report.get("segment_export_files"),
    }


class FullCorpusMeasurements(TrainerCallback):
    def __init__(self, contract, output_dir, end_step, runtime_contract, resume_path=None, manifest=None):
        self.contract, self.end_step, self.runtime_contract = contract, end_step, runtime_contract
        self.resume_path, self.manifest = resume_path, manifest
        self.output_dir = Path(output_dir)
        self.journal = self.output_dir / "full_corpus_coverage.jsonl"
        self.cursor = CoverageCursor(contract)
        if manifest is not None:
            self.cursor.restore_journal(Path(resume_path) / "coverage.jsonl", manifest["checkpoint_step"])
        self.start_step = self.cursor.step
        self.pending = []
        self.report = {
            "schema_version": 1, "status": "pending", "contract_sha256": contract_sha256(contract.value),
            "runtime_contract_sha256": contract_sha256(runtime_contract),
            "start_global_step": self.start_step, "segment_end_step": end_step,
            "total_max_steps": contract.value["total_max_steps"],
            "corpus_complete": False, "dataset_stats": contract.dataset_stats,
            "preprocessing_per_task": contract.preprocessing_per_task,
        }

    def persist(self):
        rank = self.trainer.args.process_index
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.output_dir / f"full_corpus.rank{rank}.json", self.report)
        if rank == 0:
            atomic_json(self.output_dir / "full_corpus.json", self.report)

    def on_train_begin(self, args, state, control, **kwargs):
        engine = self.trainer.model_wrapped
        def verify():
            if state.global_step != self.start_step or engine.global_steps != self.start_step or engine.skipped_steps != 0:
                raise RuntimeError("Full-corpus engine/Trainer start counters disagree")
            self.local_runtime = deepspeed_proof(self.trainer, COLLECTIVES)
            if self.manifest is not None:
                saved = strict_json(Path(self.resume_path) / f"checkpoint_rank{args.process_index}.json")
                actual = checkpoint_state_proof(self.trainer)
                if saved["state"] != actual or saved["coverage"] != self.cursor.rank_state(args.process_index):
                    raise RuntimeError("Actual full-state restore/coverage differs from saved checkpoint")
                self.report["restored_state"] = actual
        collective_action(verify, COLLECTIVES, "full-corpus engine/state initialization")
        ranks = exchange(self.local_runtime, COLLECTIVES)
        validate_topology(ranks, 8)
        self.report["runtime"] = {"ranks": ranks}
        if self.manifest is not None:
            self.report["restored_state_ranks"] = exchange({
                "rank": args.process_index, "state": self.report["restored_state"],
                "coverage": self.cursor.rank_state(args.process_index),
            }, COLLECTIVES)
        initialization = exchange(self.report["zero3_initialization"], COLLECTIVES)
        if any(
            item["active_before_from_pretrained"] is not True or item["partitioned_parameter_count"] <= 0
            for item in initialization
        ):
            raise RuntimeError("Every full-corpus rank must prove pre-load ZeRO3 partitioning")
        self.report["zero3_initialization_ranks"] = initialization
        self.report["collective_proof"] = nccl_collective_proof(COLLECTIVES)
        def prepare_journal():
            if args.process_index == 0:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                if self.manifest is None:
                    with self.journal.open("x", encoding="utf-8"):
                        pass
                elif not self.journal.exists():
                    shutil.copyfile(Path(self.resume_path) / "coverage.jsonl", self.journal)
        collective_action(prepare_journal, COLLECTIVES, "full-corpus journal initialization")
        self.report["status"] = "running"
        self.persist()

    def on_step_begin(self, args, state, control, **kwargs):
        collective_check(
            "Full-corpus step ceiling/RNG restore violated"
            if state.global_step >= self.end_step or (
                self.manifest is not None and not self.report.get("rng_restored")
            ) else None, COLLECTIVES, "full-corpus step begin",
        )
        self.pending = []
        self.engine_start = self.trainer.model_wrapped.global_steps
        self.skipped_start = self.trainer.model_wrapped.skipped_steps
        def mark_inflight():
            if args.process_index == 0:
                path = self.output_dir / "full_corpus_inflight.json"
                if path.exists():
                    raise RuntimeError("Unresolved full-corpus in-flight update; refusing replay")
                atomic_json(path, {
                    "global_step": state.global_step + 1,
                    "runtime_contract_sha256": self.report["runtime_contract_sha256"],
                })
                with path.open("r+b") as stream:
                    os.fsync(stream.fileno())
        collective_action(mark_inflight, COLLECTIVES, "full-corpus update intent")

    def record_batch(self, row_indices):
        rank = self.trainer.args.process_index
        expected = self.contract.indices(self.cursor.step + 1)[rank::8]
        actual = row_indices.detach().cpu().tolist() if row_indices is not None else []
        collective_check(
            "Actual full-corpus microbatch identity/order differs from the exact pass plan"
            if len(actual) != 1 or len(self.pending) >= 4 or actual[0] != expected[len(self.pending)]
            else None, COLLECTIVES, "full-corpus input identity",
        )
        self.pending.append({"row_index": actual[0]})

    def record_loss(self, loss):
        value = float(loss.detach().float().item())
        collective_check(
            "Nonfinite full-corpus microbatch loss" if not math.isfinite(value) else None,
            COLLECTIVES, "full-corpus loss",
        )
        self.pending[-1]["loss"] = value

    def on_step_end(self, args, state, control, **kwargs):
        engine = self.trainer.model_wrapped
        skipped = engine.skipped_steps - self.skipped_start
        evidence = {
            "rank": args.process_index, "row_indices": [row["row_index"] for row in self.pending],
            "losses": [row["loss"] for row in self.pending], "engine_global_steps": engine.global_steps,
            "optimizer_updates": engine.global_steps - self.engine_start - skipped, "skipped_updates": skipped,
        }
        collective_check(
            "Accelerate reports a skipped full-corpus optimizer update"
            if self.trainer.accelerator.optimizer_step_was_skipped else None, COLLECTIVES, "full-corpus update",
        )
        item = {"global_step": state.global_step, "ranks": exchange(evidence, COLLECTIVES)}
        collective_action(lambda: self.cursor.advance(item), COLLECTIVES, "full-corpus completed update coverage")
        def append():
            if args.process_index == 0:
                with self.journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                (self.output_dir / "full_corpus_inflight.json").unlink()
        collective_action(append, COLLECTIVES, "completed full-corpus update journal")
        if state.global_step == self.end_step:
            control.should_save = True
            control.should_training_stop = True

    def finish(self, step, loss):
        collective_check(
            "Full-corpus segment did not finish at its exact saved boundary with finite loss"
            if step != self.end_step or self.cursor.step != step or not math.isfinite(loss)
            or self.report["status"] != "checkpoint_saved" else None, COLLECTIVES, "full-corpus segment finish",
        )
        complete = step == self.contract.value["total_max_steps"]
        if complete:
            counts = self.cursor.consumed_counts()
            if (
                counts["source_counts"] != self.contract.value["source_counts"]
                or counts["exposure_counts"] != {
                    task: count * self.contract.value["full_passes"]
                    for task, count in self.contract.value["exposure_counts"].items()
                }
            ):
                raise RuntimeError("Actual completed-update full-corpus counts differ from the exact contract")
        self.report.update({
            "status": "corpus_complete" if complete else "segment_complete",
            "corpus_complete": complete, "global_step": step,
            "completed_full_passes": step // self.contract.steps_per_pass,
            "segment_training_loss": loss, "training_loss_scope": "this process segment only",
            "coverage_ranks": exchange(self.cursor.rank_state(self.trainer.args.process_index), COLLECTIVES),
            "consumed_counts": self.cursor.consumed_counts(),
        })
        if complete:
            self.report["full_pass_source_counts"] = self.contract.value["source_counts"]
            self.report["full_pass_exposure_counts"] = self.contract.value["exposure_counts"]
            self.report["total_completed_exposures"] = self.contract.size * self.contract.value["full_passes"]
        self.persist()

    def fail(self, exc):
        self.report.update({"status": "failed", "corpus_complete": False, "error": f"{type(exc).__name__}: {exc}"})
        self.persist()

    def record_export(self):
        inventory = None
        def inspect():
            nonlocal inventory
            if self.trainer.is_world_process_zero():
                inventory = segment_export_inventory(self.output_dir)
        collective_action(inspect, COLLECTIVES, "native HF segment export file hashes")
        inventories = exchange(inventory, COLLECTIVES)
        if not inventories[0] or any(value is not None for value in inventories[1:]):
            raise RuntimeError("Native export inventory must be produced by global rank zero only")
        self.report["segment_export_files"] = inventories[0]


class FullCorpusTrainer(Trainer):
    def __init__(self, *args, measurements, **kwargs):
        self.measurements = measurements
        super().__init__(*args, **kwargs)
        measurements.trainer = self
        self.add_callback(measurements)

    def _get_train_sampler(self, train_dataset=None):
        return PassSampler(len(self.train_dataset if train_dataset is None else train_dataset), self.args.data_seed)

    def create_optimizer(self):
        result = super().create_optimizer()
        expected = {id(parameter) for parameter in self.model.parameters() if parameter.requires_grad}
        actual = [id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]]
        if not isinstance(self.optimizer, torch.optim.AdamW) or not expected or set(actual) != expected or len(actual) != len(expected):
            raise RuntimeError("Full-corpus AdamW must cover every trainable parameter exactly once")
        self.measurements.report["optimizer_group_proof"] = {
            "coverage_exact": True, "trainable_parameter_tensors": len(expected),
            "frozen_parameter_tensors": sum(not parameter.requires_grad for parameter in self.model.parameters()),
            "optimizer_class": f"{type(self.optimizer).__module__}.{type(self.optimizer).__name__}",
            "groups": optimizer_hyperparameters(self.optimizer),
            "scope": "native groups before DeepSpeed FP32 partition replacement",
        }
        for group, source in zip(self.measurements.report["optimizer_group_proof"]["groups"], self.optimizer.param_groups):
            group["parameter_tensors"] = len(source["params"])
            group["full_parameter_elements"] = sum(int(getattr(parameter, "ds_numel", parameter.numel())) for parameter in source["params"])
        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.measurements.record_batch(inputs.pop("_validation_row_index", None))
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.measurements.record_loss(loss)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("_validation_row_index", None)
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if model.training:
            value = loss[0] if return_outputs else loss
            collective_check(
                "Nonfinite full-corpus forward loss" if not bool(torch.isfinite(value).all().item()) else None,
                COLLECTIVES, "full-corpus forward",
            )
        return loss

    def _save_rng_state(self, output_dir):
        super()._save_rng_state(output_dir)
        self._saved_rng_sha256 = rng_sha256()

    def _load_rng_state(self, checkpoint):
        def restore():
            super(FullCorpusTrainer, self)._load_rng_state(checkpoint)
            actual = rng_sha256()
            saved = strict_json(Path(checkpoint) / f"checkpoint_rank{self.args.process_index}.json")
            if actual != saved["rng_sha256"]:
                raise RuntimeError("Actual full-corpus rank RNG restore differs from checkpoint")
            self.measurements.report.update({"rng_restored": True, "rng_sha256": actual})
        collective_action(restore, COLLECTIVES, "full-corpus RNG restore")
        self.measurements.report["rng_restored_ranks"] = exchange({
            "rank": self.args.process_index, "rng_sha256": self.measurements.report["rng_sha256"],
        }, COLLECTIVES)

    def _save_checkpoint(self, model, trial):
        observer = self.measurements
        if self.state.global_step != observer.end_step or self.args.save_only_model or self.args.ignore_data_skip:
            raise RuntimeError("Unexpected/incomplete full-corpus checkpoint boundary")
        if self.args.max_steps != observer.contract.value["total_max_steps"]:
            raise RuntimeError("Full-corpus checkpoint total scheduler horizon changed")
        collective_action(
            lambda: super(FullCorpusTrainer, self)._save_checkpoint(model, trial),
            COLLECTIVES, "native full-corpus training checkpoint",
        )
        path = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        def save_rank():
            self._rank_state = {
                "rank": self.args.process_index, "state": checkpoint_state_proof(self),
                "rng_sha256": self._saved_rng_sha256,
                "coverage": observer.cursor.rank_state(self.args.process_index),
            }
            atomic_json(path / f"checkpoint_rank{self.args.process_index}.json", self._rank_state)
        collective_action(save_rank, COLLECTIVES, "full-corpus rank checkpoint proof")
        observer.report["saved_state_ranks"] = exchange(self._rank_state, COLLECTIVES)
        torch.distributed.barrier()
        def save_manifest():
            if self.is_world_process_zero():
                shutil.copyfile(observer.journal, path / "coverage.jsonl")
                atomic_json(path / "full_corpus_checkpoint.json", {
                    "schema_version": 1, "checkpoint_step": self.state.global_step,
                    "runtime_contract": observer.runtime_contract,
                    "files": full_inventory(path, self.state.global_step),
                    "saved_state_ranks": observer.report["saved_state_ranks"],
                })
        collective_action(save_manifest, COLLECTIVES, "full-corpus completed checkpoint manifest")
        torch.distributed.barrier()
        observer.report.update({
            "status": "checkpoint_saved", "checkpoint_path": str(path),
            "checkpoint_manifest_sha256": file_sha256(path / "full_corpus_checkpoint.json"),
        })
        observer.persist()
