import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler, SequentialSampler
from transformers import Qwen3_5ForConditionalGeneration, Trainer, TrainingArguments

from sft.bounded_validation import (
    BoundedValidationConfig,
    BoundedValidationTrainer,
    ValidationMeasurements,
    collective_check,
    collective_action,
    deepspeed_proof,
    exchange,
    nccl_collective_proof,
    validate_topology,
)
from test_bounded_validation import FakeEngine
from sft import sft_train
from test_sft_training import CharacterTokenizer, record
import test_sft_training


def config():
    return BoundedValidationConfig.parse({
        "expected_steps": 2, "warmup_steps": 1, "expected_world_size": 8,
    })


def topology():
    return [{
        "rank": rank, "local_rank": rank, "hostname": "one-host",
        "cuda_device": "NVIDIA A100-SXM4-80GB", "cuda_uuid": f"uuid-{rank}",
        "distributed_backend": "nccl", "resolved_hf_deepspeed_config": {"stage": 3},
        "resolved_hf_deepspeed_config_encoding": "accelerate-steps-per-print-positive-infinity-v1",
        "resolved_hf_deepspeed_config_sha256": "mock-hash",
        "packages": {"deepspeed": "0.16.9"}, "torch_version": "2.6.0+cu124",
        "zero_stage": 3, "bf16_enabled": True, "gradient_accumulation_steps": 2,
        "micro_batch_size": 2, "train_batch_size": 32, "cuda_device_index": rank,
        "cuda_visible_device_count": 8, "optimizer_name": "AdamW",
        "optimizer_class": "torch.optim.adamw.AdamW",
        "optimizer_group_proof": {"coverage_exact": True},
        "underlying_optimizer_groups": [{"lr": 0, "betas": [0.9, 0.999], "eps": 1e-8}],
    } for rank in range(8)]


def step_items(step):
    return [{
        "rank": rank,
        "global_step": step, "is_warmup": step == 1, "elapsed_seconds": float(rank + 1),
        "peak_allocated_bytes": 100 + rank, "peak_reserved_bytes": 200 + rank,
        "optimizer_step_skipped": False, "microbatch_count": 2, "examples": 4,
        "input_lengths": [2, 4, 2, 4], "max_input_length": 4,
        "attended_tokens": 12, "supervised_tokens": 6, "padded_tokens": 4,
        "microbatches": [], "engine_global_steps": step, "engine_optimizer_updates": 1,
        "engine_skipped_steps": 0, "engine_skipped_updates": 0,
        "row_indices": [
            (step - 1) * 32 + micro * 16 + rank * 2 + within
            for micro in range(2) for within in range(2)
        ],
    } for rank in range(8)]


class DistributedValidationTests(unittest.TestCase):
    def test_config_retains_one_default_and_accepts_eight(self):
        self.assertEqual(BoundedValidationConfig.parse({}).expected_world_size, 1)
        self.assertEqual(config().expected_world_size, 8)
        for size in (0, 2, 7, 9):
            with self.assertRaises(ValueError):
                BoundedValidationConfig.parse({"expected_world_size": size})

    def test_eight_real_rank_topology_contract_and_negative_cases(self):
        validate_topology(topology(), 8)
        for key, value in (
            ("cuda_uuid", "uuid-0"), ("rank", 0), ("local_rank", 0),
            ("cuda_device", "NVIDIA T4"), ("distributed_backend", "gloo"),
            ("cuda_device", "NVIDIA A100 MIG 1g.10gb"),
            ("train_batch_size", 4), ("bf16_enabled", False),
            ("resolved_hf_deepspeed_config_sha256", "different-hash"),
        ):
            ranks = topology()
            ranks[7][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_topology(ranks, 8)
        ranks = topology()
        for rank in ranks:
            rank["hostname"] = f"host-{rank['rank']}"
            rank["local_rank"] = 0
        validate_topology(ranks, 8)

    def test_collective_status_reports_remote_nonfinite_or_step_failure(self):
        with patch("torch.distributed.is_initialized", return_value=True), patch(
            "torch.distributed.get_world_size", return_value=8
        ), patch("torch.distributed.all_gather_object") as gather:
            def remote_error(values, value):
                values[:] = [None] * 7 + ["remote rank loss nonfinite"]
            gather.side_effect = remote_error
            with self.assertRaisesRegex(RuntimeError, "remote rank loss nonfinite"):
                collective_check(None, config(), "forward loss")
            gather.side_effect = lambda values, value: values.__setitem__(slice(None), [value] * 8)
            self.assertEqual(exchange(42, config()), [42] * 8)
            collective_check(None, config(), "success")
        with patch("torch.distributed.is_initialized", return_value=True), patch(
            "torch.distributed.get_world_size", return_value=1
        ), patch("torch.distributed.all_gather_object"):
            with self.assertRaisesRegex(RuntimeError, "Collective world size"):
                exchange(None, config())

    def test_actual_cuda_rank_sum_guard_with_mocked_transport(self):
        with patch("torch.distributed.get_rank", return_value=3), patch(
            "torch.cuda.current_device", return_value=0
        ), patch("torch.tensor", return_value=torch.tensor(4.0)), patch(
            "torch.distributed.all_reduce", side_effect=lambda value, op: value.fill_(36)
        ), patch("torch.cuda.synchronize"):
            proof = nccl_collective_proof(config())
            self.assertEqual(proof["actual"], 36)
            self.assertEqual(proof["expected"], 36)
        with patch("torch.distributed.get_rank", return_value=3), patch(
            "torch.cuda.current_device", return_value=0
        ), patch("torch.tensor", return_value=torch.tensor(4.0)), patch(
            "torch.distributed.all_reduce"
        ), patch("torch.cuda.synchronize"):
            with self.assertRaisesRegex(RuntimeError, "rank-sum"):
                nccl_collective_proof(config())

    def test_global_metrics_sum_tokens_max_time_memory_and_no_index_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer = ValidationMeasurements(config(), tmp)
            observer.trainer = SimpleNamespace(validation_fixed_order=True)
            args = SimpleNamespace(gradient_accumulation_steps=2, per_device_train_batch_size=2)
            item = observer.aggregate_step(step_items(2), args)
            self.assertEqual(item["elapsed_seconds"], 8)
            self.assertEqual(item["attended_tokens"], 96)
            self.assertEqual(item["supervised_tokens"], 48)
            self.assertEqual(item["padded_tokens"], 32)
            self.assertEqual(item["microbatch_count"], 16)
            self.assertEqual(item["examples"], 32)
            self.assertEqual(item["peak_allocated_bytes"], 107)
            self.assertEqual(item["peak_reserved_bytes"], 207)
            self.assertEqual(sorted(item["row_indices"]), list(range(32, 64)))
            wrong = step_items(2)
            wrong[-1]["row_indices"] = wrong[0]["row_indices"]
            with self.assertRaisesRegex(RuntimeError, "duplicated"):
                observer.aggregate_step(wrong, args)

    def test_rank_artifacts_do_not_collide_with_shared_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank in range(8):
                observer = ValidationMeasurements(config(), tmp)
                observer.report["local_rank_global"] = rank
                observer.persist()
            root = Path(tmp)
            self.assertEqual(len(list(root.glob("bounded_validation.rank*.json"))), 8)
            aggregate = json.loads((root / "bounded_validation.json").read_text())
            self.assertEqual(aggregate["local_rank_global"], 0)
            self.assertEqual(aggregate["schema_version"], 2)

    def test_accelerate_shards_sequential_materialized_rows_once(self):
        # Two optimizer groups, batch=2, accumulation=2, world=8 =>64 rows.
        # Materialized HPF repetition is retained as distinct source row indices.
        rows = ["gem", "hpf", "hpf", "mai"] * 16
        shards = []
        for rank in range(8):
            sampler = BatchSampler(SequentialSampler(rows), batch_size=2, drop_last=False)
            shards.append(list(BatchSamplerShard(
                sampler, num_processes=8, process_index=rank,
                split_batches=False, even_batches=True,
            )))
        for step in range(2):
            indices = [
                index for shard in shards for batch in shard[step * 2:(step + 1) * 2]
                for index in batch
            ]
            self.assertEqual(sorted(indices), list(range(step * 32, (step + 1) * 32)))
            self.assertEqual(sum(rows[index] == "hpf" for index in indices), 16)

    def test_root_256_row_plan_gives_each_rank_one_gem_two_hpf_one_mai(self):
        pattern = ["gem", "hpf", "hpf", "mai"] * 2
        rows = [
            pattern[(rank + micro) % 8]
            for step in range(8) for micro in range(4) for rank in range(8)
        ]
        for rank in range(8):
            shard = list(BatchSamplerShard(
                BatchSampler(SequentialSampler(rows), batch_size=1, drop_last=False),
                num_processes=8, process_index=rank, split_batches=False, even_batches=True,
            ))
            self.assertEqual(len(shard), 32)
            for step in range(8):
                indices = [batch[0] for batch in shard[step * 4:(step + 1) * 4]]
                self.assertEqual(indices, [step * 32 + micro * 8 + rank for micro in range(4)])
                tasks = [rows[index] for index in indices]
                self.assertEqual([tasks.count(task) for task in ("gem", "hpf", "mai")], [1, 2, 1])

    def test_forward_loss_checks_all_ranks_before_backward(self):
        trainer = object.__new__(BoundedValidationTrainer)
        trainer.measurements = SimpleNamespace(config=config())
        model = SimpleNamespace(training=True)
        with patch.object(Trainer, "compute_loss", return_value=torch.tensor(1.0)), patch(
            "sft.bounded_validation.collective_check", side_effect=RuntimeError("rank7 nonfinite")
        ):
            with self.assertRaisesRegex(RuntimeError, "rank7 nonfinite"):
                trainer.compute_loss(model, {"_validation_row_index": torch.tensor([1])})

    def test_eight_rank_engine_requires_actual_world_batch_bf16_and_adamw(self):
        engine = FakeEngine()
        engine.dp_world_size = 8
        engine.train_batch_size = lambda: 32
        engine.basic_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
        trainer = SimpleNamespace(
            model_wrapped=engine, is_deepspeed_enabled=True,
            args=SimpleNamespace(gradient_accumulation_steps=2, per_device_train_batch_size=2),
            accelerator=SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(
                hf_ds_config=SimpleNamespace(config={"zero_optimization": {"stage": 3}})
            ))),
            measurements=SimpleNamespace(report={"optimizer_group_proof": {"coverage_exact": True}}),
        )
        with patch.dict("sys.modules", {"deepspeed": SimpleNamespace(
            DeepSpeedEngine=FakeEngine, __version__="0.16.9"
        )}), patch.dict("os.environ", {"LOCAL_RANK": "7"}), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.get_world_size", return_value=8), patch(
            "torch.distributed.get_rank", return_value=7
        ), patch("torch.distributed.get_backend", return_value="nccl"), patch(
            "torch.cuda.is_available", return_value=True
        ), patch("torch.cuda.current_device", return_value=7), patch(
            "torch.cuda.device_count", return_value=8
        ), patch("torch.cuda.get_device_name", return_value="NVIDIA A100"), patch(
            "torch.cuda.get_device_properties", return_value=SimpleNamespace(uuid="uuid-7", total_memory=80000)
        ), patch("importlib.metadata.version", return_value="mock-version"):
            proof = deepspeed_proof(trainer, config())
            self.assertEqual(proof["rank"], 7)
            self.assertEqual(proof["local_rank"], 7)
            self.assertEqual(proof["cuda_uuid"], "uuid-7")
            self.assertEqual(proof["train_batch_size"], 32)
            self.assertTrue(proof["optimizer_group_proof"]["coverage_exact"])
            self.assertEqual(proof["underlying_optimizer_groups"][0]["betas"], [0.9, 0.999])
            with patch.object(engine, "basic_optimizer", torch.optim.SGD(
                [torch.nn.Parameter(torch.ones(1))], lr=0.1
            )):
                with self.assertRaisesRegex(RuntimeError, "AdamW"):
                    deepspeed_proof(trainer, config())
            with patch.object(engine, "optimizer_name", return_value="Muon"):
                with self.assertRaisesRegex(RuntimeError, "AdamW"):
                    deepspeed_proof(trainer, config())
            with patch("torch.distributed.get_world_size", return_value=1):
                with self.assertRaisesRegex(RuntimeError, "runtime mismatch"):
                    deepspeed_proof(trainer, config())

    def test_all_rank_runtime_initialization_can_be_serialized_without_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer = ValidationMeasurements(config(), tmp)
            observer.report["zero3_initialization"] = {
                "active_before_from_pretrained": True, "partitioned_parameter_count": 723,
            }
            observer.trainer = SimpleNamespace(model_wrapped=FakeEngine())
            def gather(value, cfg):
                return topology() if isinstance(value, dict) and "rank" in value else [value] * 8
            with patch("sft.bounded_validation.deepspeed_proof", return_value=topology()[0]), patch(
                "sft.bounded_validation.exchange", side_effect=gather
            ), patch("torch.distributed.get_rank", return_value=0), patch(
                "sft.bounded_validation.nccl_collective_proof", return_value={"actual": 36, "expected": 36}
            ):
                observer.on_train_begin(None, SimpleNamespace(global_step=0), None)
            report = json.loads(observer.path.read_text())
            self.assertEqual(len(report["runtime"]["ranks"]), 8)
            self.assertEqual(len(report["zero3_initialization_ranks"]), 8)
            self.assertEqual(report["status"], "running")

    def test_wrong_shard_rejected_before_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer = ValidationMeasurements(config(), tmp)
            observer.trainer = SimpleNamespace(
                validation_fixed_order=True,
                args=SimpleNamespace(gradient_accumulation_steps=2, per_device_train_batch_size=2),
            )
            observer.started = 1
            batch = {
                "attention_mask": torch.ones((2, 4), dtype=torch.long),
                "labels": torch.ones((2, 4), dtype=torch.long),
            }
            with patch("torch.distributed.get_rank", return_value=7), patch(
                "sft.bounded_validation.exchange", side_effect=lambda value, cfg: [value] * 8
            ):
                observer.record_batch(batch, torch.tensor([14, 15]))
                self.assertEqual(observer.pending[0]["row_indices"], [14, 15])
                with self.assertRaisesRegex(RuntimeError, "shard plan"):
                    observer.record_batch(batch, torch.tensor([14, 15]))

    def test_remote_skipped_partial_or_mismatched_update_fails_every_rank(self):
        for key, value in (
            ("optimizer_step_skipped", True), ("engine_optimizer_updates", 0),
            ("engine_global_steps", 0), ("global_step", 2), ("microbatch_count", 1),
            ("engine_skipped_updates", 1),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                observer = ValidationMeasurements(config(), tmp)
                observer.report["local_rank_global"] = 0
                observer.trainer = SimpleNamespace(
                    validation_fixed_order=True, model_wrapped=SimpleNamespace(global_steps=1, skipped_steps=0),
                    accelerator=SimpleNamespace(optimizer_step_was_skipped=False),
                    args=SimpleNamespace(gradient_accumulation_steps=2, per_device_train_batch_size=2),
                )
                observer.started = 0
                observer.engine_step_start = 0
                observer.engine_skipped_start = 0
                observer.pending = [{
                    "input_lengths": [2, 4], "attended_tokens": 6,
                    "supervised_tokens": 3, "padded_tokens": 2, "row_indices": [0, 1],
                }] * 2
                ranks = step_items(1)
                ranks[-1][key] = value
                def gather(item, cfg):
                    return ranks if isinstance(item, dict) else [item] * 8
                with patch("sft.bounded_validation.exchange", side_effect=gather), patch(
                    "torch.cuda.synchronize"
                ), patch("torch.cuda.max_memory_allocated", return_value=100), patch(
                    "torch.cuda.max_memory_reserved", return_value=200
                ), patch("sft.bounded_validation.time.perf_counter", return_value=1):
                    with self.assertRaises(RuntimeError):
                        observer.on_step_end(observer.trainer.args, SimpleNamespace(global_step=1), None)

    def test_collective_action_propagates_local_and_remote_export_failure(self):
        with patch("sft.bounded_validation.exchange", side_effect=lambda item, cfg: [item] * 8):
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                collective_action(Mock(side_effect=OSError("disk full")), config(), "export")
        with patch("sft.bounded_validation.exchange", return_value=[None] * 7 + ["rank7 export failure"]):
            with self.assertRaisesRegex(RuntimeError, "rank7 export failure"):
                collective_action(lambda: None, config(), "export")

    def test_actual_dataset_collator_keeps_indices_outside_model_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data.jsonl"
            rows = [record(prompt="gem"), record(prompt="hpf"), record(prompt="hpf")]
            data.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            dataset = sft_train.build_dataset(
                str(data), CharacterTokenizer(), 1000, validation_stats={},
            )
            batch = sft_train.DataCollatorForSFT(0)(list(dataset))
            self.assertEqual(batch["_validation_row_index"].tolist(), [0, 1, 2])
            trainer = object.__new__(BoundedValidationTrainer)
            trainer.measurements = Mock()
            with patch.object(Trainer, "training_step", return_value=torch.tensor(1.0)) as forward:
                trainer.training_step(None, batch)
            self.assertNotIn("_validation_row_index", forward.call_args.args[1])
            self.assertEqual(trainer.measurements.record_batch.call_args.args[1].tolist(), [0, 1, 2])

    def test_global_throughput_excludes_warmup_and_uses_max_rank_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer = ValidationMeasurements(config(), tmp)
            observer.report["local_rank_global"] = 0
            observer.trainer = SimpleNamespace(validation_fixed_order=True)
            observer.training_started = 0
            args = SimpleNamespace(gradient_accumulation_steps=2, per_device_train_batch_size=2)
            observer.report["steps"] = [observer.aggregate_step(step_items(step), args) for step in (1, 2)]
            def gather(value, cfg):
                if value is None:
                    return [None] * 8
                return [{
                    "training_loss": rank + 1.0, "training_wall_seconds": 30 + rank,
                    "global_step": 2,
                } for rank in range(8)]
            with patch("sft.bounded_validation.exchange", side_effect=gather), patch(
                "sft.bounded_validation.time.perf_counter", return_value=30
            ):
                observer.finish(2, 1.0)
            report = json.loads(observer.path.read_text())
            self.assertEqual(report["status"], "training_complete")
            self.assertEqual(report["measured_steps"], 1)
            self.assertEqual(report["measured_seconds"], 8)
            self.assertEqual(report["attended_tokens_per_second"], 12)
            self.assertEqual(report["supervised_tokens_per_second"], 6)
            self.assertEqual(report["examples_per_second"], 4)
            self.assertEqual(report["training_wall_seconds"], 37)
            self.assertEqual(report["training_loss"], 4.5)

    def test_main_eight_rank_export_and_dataset_guards_with_mock_engine(self):
        for rank, count, skipped, save_error in (
            (0, 64, 0, False), (7, 64, 0, False),
            (0, 63, 0, False), (0, 64, 1, False), (0, 64, 0, True),
        ):
            with self.subTest(rank=rank, count=count, skipped=skipped, save_error=save_error):
                with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                    cfg = sft_train.SFTConfig(
                        step_key="test", base_model="unused", train_jsonl="unused",
                        val_jsonl="unused", output_dir=tmp, deepspeed_config="unused",
                        enable_thinking=False, gradient_checkpointing=False,
                        per_device_train_batch_size=2, gradient_accumulation_steps=2,
                        bounded_validation={"expected_steps": 2, "warmup_steps": 1, "expected_world_size": 8},
                        validation_fixed_order=True, eval_strategy="no", save_strategy="no",
                        optim="adamw_torch", adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                    )
                    model = SimpleNamespace(
                        config=SimpleNamespace(model_type="qwen3_5"),
                        parameters=lambda: [SimpleNamespace(ds_id=1)],
                    )
                    model_class = Mock()
                    model_class.from_pretrained.return_value = model
                    tokenizer = Mock(pad_token="pad", pad_token_id=0)
                    observer = Mock(report={})
                    trainer = SimpleNamespace(
                        train=Mock(return_value=SimpleNamespace(training_loss=1.0)),
                        state=SimpleNamespace(global_step=2, best_metric=None, best_model_checkpoint=None),
                        save_model=Mock(side_effect=OSError("disk full") if save_error else None),
                        save_state=Mock(), is_world_process_zero=Mock(return_value=rank == 0),
                    )
                    def dataset(*args, validation_stats, **kwargs):
                        validation_stats.update({"skipped": skipped})
                        return [{}] * count
                    stack.enter_context(patch("sys.argv", [
                        "sft_train.py", "--config", "unused", "--max-steps", "2", "--no-wandb",
                    ]))
                    for target, replacement in (
                        ("SFTConfig.from_yaml", Mock(return_value=cfg)),
                        ("setup_deepspeed", Mock()),
                        ("AutoConfig.from_pretrained", Mock(return_value=model.config)),
                        ("AutoTokenizer.from_pretrained", Mock(return_value=tokenizer)),
                        ("model_class_for_config", Mock(return_value=model_class)),
                        ("configure_qwen35_text_training", Mock()),
                        ("build_dataset", Mock(side_effect=dataset)),
                        ("TrainingArguments", Mock(return_value=SimpleNamespace(world_size=8))),
                        ("ValidationMeasurements", Mock(return_value=observer)),
                        ("BoundedValidationTrainer", Mock(return_value=trainer)),
                    ):
                        stack.enter_context(patch(f"sft.sft_train.{target}", new=replacement))
                    barrier = stack.enter_context(patch("torch.distributed.barrier"))
                    stack.enter_context(patch(
                        "transformers.integrations.deepspeed.is_deepspeed_zero3_enabled", return_value=True,
                    ))
                    stack.enter_context(patch(
                        "sft.bounded_validation.exchange", side_effect=lambda item, cfg: [item] * 8,
                    ))
                    invalid_rows = count != 64 or skipped
                    if invalid_rows or save_error:
                        with self.assertRaises(ValueError if invalid_rows else RuntimeError):
                            sft_train.main()
                        self.assertFalse((Path(tmp) / "training_summary.json").exists())
                        if invalid_rows:
                            model_class.from_pretrained.assert_not_called()
                        else:
                            observer.fail.assert_called_once()
                    else:
                        sft_train.main()
                        trainer.save_model.assert_called_once()
                        trainer.save_state.assert_called_once()
                        self.assertEqual(barrier.call_count, 3)
                        self.assertTrue(observer.report["export_complete"])
                        if rank == 0:
                            tokenizer.save_pretrained.assert_called_once_with(tmp)
                            self.assertTrue((Path(tmp) / "training_summary.json").exists())
                        else:
                            tokenizer.save_pretrained.assert_not_called()
                            self.assertFalse((Path(tmp) / "training_summary.json").exists())
                        self.assertEqual(sft_train.TrainingArguments.call_args.kwargs["optim"], "adamw_torch")
                        for key, value in (("adam_beta1", 0.9), ("adam_beta2", 0.999), ("adam_epsilon", 1e-8)):
                            self.assertEqual(sft_train.TrainingArguments.call_args.kwargs[key], value)

    def test_actual_adamw_groups_hybrid_update_frozen_vision_and_state_reload(self):
        tiny = test_sft_training.ModelTests.tiny_config()
        tiny.tie_word_embeddings = True
        tiny.text_config.tie_word_embeddings = True
        model = Qwen3_5ForConditionalGeneration(tiny)
        sft_train.configure_qwen35_text_training(model)
        self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
        for parameter in model.parameters():
            parameter.ds_numel = parameter.numel()
        with tempfile.TemporaryDirectory() as tmp:
            measurements = ValidationMeasurements(config(), tmp)
            trainer = BoundedValidationTrainer(
                model=model, measurements=measurements, fixed_order=True,
                args=TrainingArguments(
                    output_dir=tmp, use_cpu=True, report_to="none", optim="adamw_torch",
                    learning_rate=5e-6, weight_decay=0.0,
                    adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                ),
            )
            optimizer = trainer.create_optimizer()
            proof = measurements.report["optimizer_group_proof"]
            self.assertTrue(proof["coverage_exact"])
            self.assertEqual(proof["trainable_parameter_tensors"], sum(p.requires_grad for p in model.parameters()))
            self.assertGreater(proof["frozen_parameter_tensors"], 0)
            self.assertEqual(sum(g["full_parameter_elements"] for g in proof["groups"]),
                             sum(p.numel() for p in model.parameters() if p.requires_grad))
            for group in proof["groups"]:
                self.assertEqual(group["lr"], 5e-6)
                self.assertEqual(group["weight_decay"], 0)
                self.assertEqual(group["betas"], [0.9, 0.999])
                self.assertEqual(group["eps"], 1e-8)
            before = model.get_input_embeddings().weight.detach().clone()
            vision_before = [p.detach().clone() for p in model.model.visual.parameters()]
            loss = model(
                input_ids=torch.tensor([[4, 5, 6, 7]]),
                labels=torch.tensor([[-100, -100, 6, 7]]),
            ).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.get_input_embeddings().weight))
            for old, parameter in zip(vision_before, model.model.visual.parameters()):
                torch.testing.assert_close(old, parameter)
            state_path = Path(tmp) / "optimizer.pt"
            torch.save(optimizer.state_dict(), state_path)
            trainer.optimizer = None
            reloaded = trainer.create_optimizer()
            reloaded.load_state_dict(torch.load(state_path, weights_only=True))
            self.assertEqual(len(reloaded.state), len(optimizer.state))
            for state in reloaded.state.values():
                self.assertEqual(state["step"].item(), 1)
                self.assertTrue(torch.isfinite(state["exp_avg"]).all())
                self.assertTrue(torch.isfinite(state["exp_avg_sq"]).all())
            group = reloaded.param_groups[0]["params"]
            group.append(group[0])
            with self.assertRaisesRegex(RuntimeError, "exactly once"):
                trainer.create_optimizer()

    def test_yaml_optimizer_settings_and_invalid_choices_fail_before_loading(self):
        base = {
            "step_key": "test", "base_model": "unused", "train_jsonl": "unused",
            "val_jsonl": "unused", "output_dir": "unused", "deepspeed_config": "unused",
            "bounded_validation": {"expected_world_size": 8},
            "optim": "adamw_torch", "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(base), encoding="utf-8")
            config_from_yaml = sft_train.SFTConfig.from_yaml(str(path))
            self.assertEqual(config_from_yaml.optim, "adamw_torch")
            self.assertEqual(config_from_yaml.adam_beta1, 0.9)
            self.assertEqual(config_from_yaml.adam_beta2, 0.999)
            self.assertEqual(config_from_yaml.adam_epsilon, 1e-8)
            for key, value in (
                ("optim", "muon"), ("adam_beta1", True), ("adam_beta2", 1.0),
                ("adam_beta1", float("nan")), ("adam_epsilon", 0),
                ("adam_epsilon", float("inf")), ("adam_epsilon", "1e-8"),
            ):
                path.write_text(yaml.safe_dump({**base, key: value}), encoding="utf-8")
                with self.subTest(key=key, value=value), patch("sys.argv", [
                    "sft_train.py", "--config", str(path), "--max-steps", "8",
                ]), patch.object(sft_train, "setup_deepspeed"), patch.object(
                    sft_train.AutoConfig, "from_pretrained"
                ) as load:
                    with self.assertRaises(ValueError):
                        sft_train.main()
                    load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
