import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import Trainer

from sft.bounded_validation import (
    BoundedValidationConfig,
    BoundedValidationTrainer,
    ValidationMeasurements,
    deepspeed_proof,
)
from sft.sft_train import build_dataset
from sft import sft_train
from test_sft_training import CharacterTokenizer, record


class FakeEngine:
    dp_world_size = 1
    global_steps = 0
    optimizer = SimpleNamespace()

    def bfloat16_enabled(self):
        return True

    def optimizer_name(self):
        return "AdamW"

    def zero_optimization_stage(self):
        return 3

    def gradient_accumulation_steps(self):
        return 2

    def train_micro_batch_size_per_gpu(self):
        return 2

    def train_batch_size(self):
        return 4


class BoundedValidationTests(unittest.TestCase):
    def config(self):
        return BoundedValidationConfig.parse({
            "expected_steps": 2, "warmup_steps": 1,
            "expected_zero_stage": 3, "expected_world_size": 1,
        })

    def trainer(self):
        return SimpleNamespace(
            model_wrapped=FakeEngine(), is_deepspeed_enabled=True,
            args=SimpleNamespace(gradient_accumulation_steps=2, per_device_train_batch_size=2),
            accelerator=SimpleNamespace(
                optimizer_step_was_skipped=False,
                state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(
                    hf_ds_config=SimpleNamespace(config={
                        "zero_optimization": {"stage": 3, "reduce_bucket_size": 1024},
                        "bf16": {"enabled": True},
                    })
                )),
            ),
        )

    def test_config_caps_and_invalid_settings(self):
        for settings in (
            {"expected_steps": 11}, {"expected_steps": 0},
            {"warmup_steps": 8}, {"expected_zero_stage": 2},
            {"expected_world_size": 2}, {"warmup_steps": True},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                BoundedValidationConfig.parse(settings)
        with self.assertRaises(TypeError):
            BoundedValidationConfig.parse({"unused": True})

    def test_runtime_proof_checks_actual_engine_and_parallel_settings(self):
        trainer = self.trainer()
        fake_ds = SimpleNamespace(DeepSpeedEngine=FakeEngine, __version__="0.16.9")
        with patch.dict("sys.modules", {"deepspeed": fake_ds}), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.get_world_size", return_value=1), patch(
            "torch.distributed.get_rank", return_value=0
        ), patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.current_device", return_value=0
        ), patch("torch.cuda.get_device_name", return_value="mock-device"), patch(
            "importlib.metadata.version", return_value="mock-version"
        ):
            proof = deepspeed_proof(trainer, self.config())
            self.assertEqual(proof["zero_stage"], 3)
            self.assertEqual(proof["world_size"], 1)
            self.assertEqual(proof["train_batch_size"], 4)
            self.assertIs(proof["bf16_enabled"], True)
            self.assertEqual(
                proof["resolved_hf_deepspeed_config"]["zero_optimization"]["reduce_bucket_size"],
                1024,
            )
            self.assertEqual(proof["optimizer_name"], "AdamW")
            self.assertEqual(proof["engine_optimizer_class"], "types.SimpleNamespace")
            trainer.accelerator.state.deepspeed_plugin.hf_ds_config.config["bf16"]["enabled"] = False
            self.assertIs(proof["resolved_hf_deepspeed_config"]["bf16"]["enabled"], True)
            trainer.accelerator.state.deepspeed_plugin.hf_ds_config.config["bf16"]["enabled"] = True
            with patch.object(trainer.model_wrapped, "bfloat16_enabled", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "BF16"):
                    deepspeed_proof(trainer, self.config())
            trainer.is_deepspeed_enabled = False
            with self.assertRaisesRegex(RuntimeError, "inactive"):
                deepspeed_proof(trainer, self.config())
            trainer.is_deepspeed_enabled = True
            trainer.model_wrapped = object()
            with self.assertRaisesRegex(RuntimeError, "inactive"):
                deepspeed_proof(trainer, self.config())
            trainer.model_wrapped = FakeEngine()
            trainer.model_wrapped.dp_world_size = 2
            with self.assertRaisesRegex(RuntimeError, "batch/parallel"):
                deepspeed_proof(trainer, self.config())
            trainer.model_wrapped.dp_world_size = 1
            with patch("torch.distributed.get_world_size", return_value=2):
                with self.assertRaisesRegex(RuntimeError, "runtime mismatch"):
                    deepspeed_proof(trainer, self.config())
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    deepspeed_proof(trainer, self.config())

    def test_accumulated_token_counts_cuda_timing_and_persistent_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer = ValidationMeasurements(self.config(), tmp)
            observer.trainer = self.trainer()
            state = SimpleNamespace(global_step=0)
            batch = {
                "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
                "labels": torch.tensor([[-100, 2, -100, -100], [-100, -100, 3, 4]]),
            }
            with patch(
                "sft.bounded_validation.deepspeed_proof", return_value={"zero_stage": 3}
            ), patch("torch.cuda.synchronize") as synchronize, patch(
                "torch.cuda.reset_peak_memory_stats"
            ) as reset, patch(
                "torch.cuda.max_memory_allocated", return_value=100
            ), patch(
                "torch.cuda.max_memory_reserved", return_value=200
            ), patch(
                "sft.bounded_validation.time.perf_counter", side_effect=[0, 1, 3, 5, 9, 10]
            ):
                observer.on_train_begin(observer.trainer.args, state, None)
                self.assertEqual(json.loads(observer.path.read_text())["status"], "running")
                for step in (1, 2):
                    observer.on_step_begin(observer.trainer.args, state, None)
                    for _ in range(2):
                        observer.record_batch(batch)
                        observer.record_loss(torch.tensor(1.5))
                    state.global_step = step
                    observer.trainer.model_wrapped.global_steps = step
                    observer.on_step_end(observer.trainer.args, state, None)
                observer.finish(2, 1.5)
                self.assertEqual(synchronize.call_count, 4)
                self.assertEqual(reset.call_count, 2)
            report = json.loads((Path(tmp) / "bounded_validation.json").read_text())
            self.assertEqual(report["status"], "training_complete")
            self.assertEqual(report["global_step"], 2)
            self.assertEqual(report["training_wall_seconds"], 10)
            self.assertEqual(report["steps"][0]["elapsed_seconds"], 2)
            self.assertEqual(report["steps"][1]["elapsed_seconds"], 4)
            self.assertEqual(report["steps"][1]["input_lengths"], [2, 4, 2, 4])
            self.assertEqual(report["steps"][1]["attended_tokens"], 12)
            self.assertEqual(report["steps"][1]["padded_tokens"], 4)
            self.assertEqual(report["steps"][1]["supervised_tokens"], 6)
            self.assertEqual(report["steps"][1]["engine_optimizer_updates"], 1)
            self.assertEqual(report["steps"][1]["peak_allocated_bytes"], 100)
            self.assertEqual(report["measured_steps"], 1)
            self.assertEqual(report["attended_tokens_per_second"], 3)
            self.assertEqual(report["supervised_tokens_per_second"], 1.5)
            self.assertEqual(report["examples_per_second"], 1)
            with self.assertRaisesRegex(RuntimeError, "exact expected"):
                observer.finish(1, 1.5)
            observer.fail(RuntimeError("test failure"))
            self.assertEqual(json.loads(observer.path.read_text())["status"], "failed")

    def test_invalid_measurements_and_nonfinite_loss_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer = ValidationMeasurements(self.config(), tmp)
            observer.trainer = self.trainer()
            with self.assertRaisesRegex(RuntimeError, "outside"):
                observer.record_batch({})
            observer.started = 0
            observer.pending = [{}]
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                observer.record_loss(torch.tensor(float("nan")))
            with patch("torch.cuda.synchronize"), patch(
                "sft.bounded_validation.time.perf_counter", return_value=1
            ):
                observer.pending = []
                with self.assertRaisesRegex(RuntimeError, "Missing"):
                    observer.on_step_end(observer.trainer.args, SimpleNamespace(global_step=1), None)

    def test_skipped_or_unperformed_optimizer_update_rejected(self):
        for engine_delta, skipped, accumulation in ((0, False, 1), (1, True, 1), (1, False, 2)):
            with self.subTest(delta=engine_delta, skipped=skipped, accumulation=accumulation):
                with tempfile.TemporaryDirectory() as tmp:
                    observer = ValidationMeasurements(self.config(), tmp)
                    observer.trainer = self.trainer()
                    observer.trainer.accelerator.optimizer_step_was_skipped = skipped
                    observer.trainer.args.gradient_accumulation_steps = accumulation
                    observer.engine_step_start = 0
                    observer.trainer.model_wrapped.global_steps = engine_delta
                    observer.started = 0
                    observer.pending = [{
                        "input_lengths": [2], "attended_tokens": 2,
                        "supervised_tokens": 1, "padded_tokens": 0,
                    }]
                    with patch("torch.cuda.synchronize"), patch(
                        "torch.cuda.max_memory_allocated", return_value=100
                    ), patch("torch.cuda.max_memory_reserved", return_value=200), patch(
                        "sft.bounded_validation.time.perf_counter", return_value=1
                    ):
                        with self.assertRaises(RuntimeError):
                            observer.on_step_end(
                                observer.trainer.args, SimpleNamespace(global_step=1), None
                            )

    def test_fixed_order_is_opt_in_and_keeps_repeated_rows(self):
        trainer = object.__new__(BoundedValidationTrainer)
        trainer.train_dataset = ["gem", "hpf", "hpf", "mai"]
        trainer.validation_fixed_order = True
        self.assertEqual(list(trainer._get_train_sampler()), [0, 1, 2, 3])
        trainer.validation_fixed_order = False
        with patch.object(Trainer, "_get_train_sampler", return_value="default") as sampler:
            self.assertEqual(trainer._get_train_sampler(), "default")
            sampler.assert_called_once()

    def test_inactive_zero3_fails_before_model_loading(self):
        cfg = sft_train.SFTConfig(
            step_key="test", base_model="unused", train_jsonl="unused",
            val_jsonl="unused", output_dir="unused", deepspeed_config="unused",
            enable_thinking=False,
            bounded_validation={"expected_steps": 2, "warmup_steps": 1},
        )
        model_class = Mock()
        with patch("sys.argv", [
            "sft_train.py", "--config", "unused", "--max-steps", "2", "--no-wandb",
        ]), patch.object(
            sft_train.SFTConfig, "from_yaml", return_value=cfg
        ), patch.object(
            sft_train, "setup_deepspeed"
        ), patch.object(
            sft_train.AutoConfig, "from_pretrained",
            return_value=SimpleNamespace(model_type="qwen3_5"),
        ), patch.object(
            sft_train.AutoTokenizer, "from_pretrained", return_value=SimpleNamespace(pad_token="pad")
        ), patch.object(
            sft_train, "model_class_for_config", return_value=model_class
        ), patch.object(
            sft_train, "build_dataset", return_value=[{}]
        ), patch.object(
            sft_train, "TrainingArguments"
        ) as arguments, patch(
            "transformers.integrations.deepspeed.is_deepspeed_zero3_enabled",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "before model loading"):
                sft_train.main()
            arguments.assert_called_once()
            model_class.from_pretrained.assert_not_called()

    def test_dataset_skip_and_context_truncation_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data.jsonl"
            rows = [record(), record(answer=""), record(answer="x" * 100)]
            data.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
            stats = {}
            dataset = build_dataset(str(data), CharacterTokenizer(), 20, validation_stats=stats)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(stats["records"], 3)
            self.assertEqual(stats["kept"], 1)
            self.assertEqual(stats["skipped"], 2)
            self.assertEqual(stats["left_truncated_records"], 1)
            self.assertGreater(stats["context_tokens_removed"], 0)
            self.assertEqual(stats["skip_reasons"], {"empty_target": 1, "target_does_not_fit": 1})


if __name__ == "__main__":
    unittest.main()
