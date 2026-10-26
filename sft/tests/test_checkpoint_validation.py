import copy
import importlib.util
import io
import json
import random
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import numpy as np
from transformers import Qwen3_5ForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments, set_seed

from sft import bounded_validation as bv
from sft import sft_train
from sft.sft_train import DataCollatorForSFT
from sft.training_utils import configure_qwen35_text_training
import test_sft_training
import test_distributed_validation


def config():
    return bv.BoundedValidationConfig(expected_steps=8, warmup_steps=2, expected_world_size=8)


def checkpoint_files(path, step=4):
    path.mkdir(parents=True)
    (path / "latest").write_text(f"global_step{step}")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": 8}))
    for name in ("scheduler.pt", "training_args.bin"):
        (path / name).write_bytes(b"fixture-not-a-real-state")
    tag = path / f"global_step{step}"
    tag.mkdir()
    for rank in range(8):
        (path / f"rng_state_{rank}.pth").write_bytes(b"fixture-rng")
        (path / f"checkpoint_rank{rank}.json").write_text(json.dumps({
            "rank": rank, "state": {"global_step": step}, "rng_sha256": "fixture-rng-hash",
        }))
        for kind, prefix in (("model", ""), ("optim", "bf16_")):
            (tag / f"{prefix}zero_pp_rank_{rank}_mp_rank_00_{kind}_states.pt").write_bytes(b"fixture-shard")


def manifest(path, contract):
    report = {
        "status": "checkpoint_saved", "steps": [
            {"global_step": step, "phase": "initial"} for step in range(1, 5)
        ],
        "phases": [{"phase": "initial", "start_global_step": 0, "end_global_step": 4}],
    }
    value = {
        "schema_version": 1, "checkpoint_step": 4, "contract": contract,
        "files": bv.checkpoint_inventory(path, 8, 4), "measurements": report,
    }
    bv.atomic_json(path / "bounded_checkpoint.json", value)
    return value


class CheckpointValidationTests(unittest.TestCase):
    def test_complete_sharded_checkpoint_rejects_missing_mutated_or_incompatible_state(self):
        contract = {"model": "full-qwen35", "world": 8, "max_steps": 8, "seed": 42}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint-4"
            checkpoint_files(path)
            value = manifest(path, contract)
            self.assertEqual(bv.validate_checkpoint(path, contract, config(), 4), value)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                bv.validate_checkpoint(path, {**contract, "world": 1}, config(), 4)
            for name in (
                "scheduler.pt", "rng_state_7.pth", "trainer_state.json",
                "global_step4/zero_pp_rank_7_mp_rank_00_model_states.pt",
                "global_step4/bf16_zero_pp_rank_7_mp_rank_00_optim_states.pt",
            ):
                data = (path / name).read_bytes()
                (path / name).unlink()
                with self.subTest(name=name), self.assertRaises((ValueError, FileNotFoundError)):
                    bv.validate_checkpoint(path, contract, config(), 4)
                (path / name).write_bytes(data)
            (path / "scheduler.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hashes"):
                bv.validate_checkpoint(path, contract, config(), 4)
            (path / "scheduler.pt").write_bytes(b"fixture-not-a-real-state")
            (path / "latest").write_text("global_step3")
            with self.assertRaisesRegex(ValueError, "latest"):
                bv.validate_checkpoint(path, contract, config(), 4)

    def test_scheduler_total_horizon_and_preceding_step_history_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint-4"
            checkpoint_files(path)
            value = manifest(path, {})
            state_path = path / "trainer_state.json"
            state_path.write_text(json.dumps({"global_step": 4, "max_steps": 12}))
            value["files"] = bv.checkpoint_inventory(path, 8, 4)
            bv.atomic_json(path / "bounded_checkpoint.json", value)
            with self.assertRaisesRegex(ValueError, "horizon"):
                bv.validate_checkpoint(path, {}, config(), 4)
            state_path.write_text(json.dumps({"global_step": 4, "max_steps": 8}))
            value["files"] = bv.checkpoint_inventory(path, 8, 4)
            value["measurements"]["steps"].pop()
            bv.atomic_json(path / "bounded_checkpoint.json", value)
            with self.assertRaisesRegex(ValueError, "preceding"):
                bv.validate_checkpoint(path, {}, config(), 4)

    def test_actual_adam_model_scheduler_state_hashes_detect_warmstart_or_changed_moments(self):
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 8)
        for _ in range(4):
            optimizer.zero_grad()
            model(torch.ones(2, 4)).square().sum().backward()
            optimizer.step()
            scheduler.step()
        parameters = list(model.parameters())
        trainer = SimpleNamespace(
            model=model, state=SimpleNamespace(global_step=4), lr_scheduler=scheduler,
            model_wrapped=SimpleNamespace(
                global_steps=4, skipped_steps=0, basic_optimizer=optimizer,
                optimizer=SimpleNamespace(fp32_partitioned_groups_flat=parameters),
            ),
        )
        proof = bv.checkpoint_state_proof(trainer)
        self.assertEqual(proof["optimizer_steps"], [4, 4])
        saved = copy.deepcopy(optimizer.state_dict())
        optimizer.state[parameters[0]]["exp_avg"].add_(1)
        changed = bv.checkpoint_state_proof(trainer)
        self.assertNotEqual(proof["optimizer_sha256"], changed["optimizer_sha256"])
        optimizer.load_state_dict(saved)
        self.assertEqual(proof, bv.checkpoint_state_proof(trainer))
        optimizer.state[parameters[0]]["step"].zero_()
        with self.assertRaisesRegex(RuntimeError, "optimizer step"):
            bv.checkpoint_state_proof(trainer)
        optimizer.state[parameters[0]]["step"].fill_(4.5)
        with self.assertRaisesRegex(RuntimeError, "optimizer step"):
            bv.checkpoint_state_proof(trainer)
        # Hash actual BF16 tensor bytes, not a lossy float conversion or repr.
        value = torch.ones(5, dtype=torch.bfloat16)
        digest = bv.state_sha256(value)
        value[0] = 2
        self.assertNotEqual(digest, bv.state_sha256(value))
        value[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite tensor"):
            bv.state_sha256(value)

    def test_checkpoint_save_uses_native_full_checkpoint_then_atomic_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint-4"
            observer = bv.ValidationMeasurements(config(), tmp)
            observer.configure_checkpoint(4, {"contract": "same"})
            observer.training_started = 0
            observer.report["local_rank_global"] = 0
            observer.report["steps"] = [{"global_step": i} for i in range(1, 5)]
            trainer = object.__new__(bv.BoundedValidationTrainer)
            trainer.measurements = observer
            trainer.args = SimpleNamespace(
                output_dir=tmp, process_index=0, save_only_model=False, ignore_data_skip=False, max_steps=8,
            )
            trainer.state = SimpleNamespace(global_step=4)
            trainer._saved_rng_sha256 = "fixture-rng-hash"
            trainer.is_world_process_zero = lambda: True
            with patch.object(Trainer, "_save_checkpoint", side_effect=lambda *args: checkpoint_files(path)) as native, patch(
                "sft.bounded_validation.exchange", side_effect=lambda value, cfg: [value] * 8
            ), patch("torch.distributed.barrier"), patch(
                "sft.bounded_validation.checkpoint_state_proof", return_value={"global_step": 4}
            ):
                trainer._save_checkpoint(None, None)
            native.assert_called_once()
            value = bv.validate_checkpoint(path, {"contract": "same"}, config(), 4)
            self.assertEqual(value["measurements"]["status"], "checkpoint_saved")
            self.assertEqual(observer.report["checkpoint_validation"]["checkpoint_manifest_sha256"],
                             bv.file_sha256(path / "bounded_checkpoint.json"))
            trainer.state.global_step = 5
            with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                trainer._save_checkpoint(None, None)

    def test_resume_rng_restore_hash_and_total_step_continuity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint-4"
            checkpoint_files(path)
            value = manifest(path, {})
            observer = bv.ValidationMeasurements(config(), tmp)
            observer.configure_checkpoint(4, {}, str(path), value)
            self.assertEqual(len(observer.report["steps"]), 4)
            trainer = object.__new__(bv.BoundedValidationTrainer)
            trainer.measurements = observer
            trainer.args = SimpleNamespace(process_index=7, gradient_accumulation_steps=4, per_device_train_batch_size=1)
            observer.trainer = trainer
            trainer.validation_fixed_order = True
            batch = {"attention_mask": torch.ones((1, 4)), "labels": torch.ones((1, 4))}
            with patch("sft.bounded_validation.exchange", side_effect=lambda item, cfg: [item] * 8), patch(
                "torch.distributed.get_rank", return_value=7
            ), patch("sft.bounded_validation.rng_sha256", return_value="fixture-rng-hash"), patch.object(
                Trainer, "_load_rng_state"
            ) as native:
                trainer._load_rng_state(str(path))
                native.assert_called_once_with(str(path))
                self.assertTrue(observer.report["checkpoint_validation"]["rng_restored"])
                observer.started = 1
                observer.record_batch(batch, torch.tensor([135]))  # global step5, micro0, rank7
                with self.assertRaisesRegex(RuntimeError, "shard plan"):
                    observer.record_batch(batch, torch.tensor([7]))  # replay of phase1
            with patch("sft.bounded_validation.exchange", side_effect=lambda item, cfg: [item] * 8), patch(
                "sft.bounded_validation.rng_sha256", return_value="wrong"
            ), patch.object(Trainer, "_load_rng_state"):
                with self.assertRaisesRegex(RuntimeError, "RNG fingerprint"):
                    trainer._load_rng_state(str(path))

    def test_runtime_restore_checks_actual_saved_state_before_resumed_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint-4"
            checkpoint_files(path)
            value = manifest(path, {})
            for actual in ({"global_step": 4}, {"global_step": 0}):
                observer = bv.ValidationMeasurements(config(), tmp)
                observer.report["zero3_initialization"] = {
                    "active_before_from_pretrained": True, "partitioned_parameter_count": 723,
                }
                observer.configure_checkpoint(4, {}, str(path), value)
                observer.trainer = SimpleNamespace(model_wrapped=SimpleNamespace(global_steps=4, skipped_steps=0))
                def exchange(item, cfg):
                    if isinstance(item, dict) and "cuda_uuid" in item:
                        return test_distributed_validation.topology()
                    return [item] * 8
                with patch("sft.bounded_validation.exchange", side_effect=exchange), patch(
                    "sft.bounded_validation.deepspeed_proof", return_value=test_distributed_validation.topology()[0]
                ), patch("sft.bounded_validation.nccl_collective_proof", return_value={"actual": 36}), patch(
                    "torch.distributed.get_rank", return_value=0
                ), patch("sft.bounded_validation.checkpoint_state_proof", return_value=actual):
                    args = SimpleNamespace(process_index=0)
                    if actual["global_step"] == 0:
                        with self.assertRaisesRegex(RuntimeError, "state hashes"):
                            observer.on_train_begin(args, SimpleNamespace(global_step=4), None)
                    else:
                        observer.on_train_begin(args, SimpleNamespace(global_step=4), None)
                        self.assertEqual(observer.report["engine_global_steps_start"], 4)
                        self.assertEqual(len(observer.report["checkpoint_validation"]["restored_state_ranks"]), 8)
                        with self.assertRaisesRegex(RuntimeError, "RNG must be restored"):
                            observer.on_step_begin(args, SimpleNamespace(global_step=4), None)

    def test_initial_phase_stops_at_four_and_resumed_final_retains_eight_phase_labeled_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer = bv.ValidationMeasurements(config(), tmp)
            observer.configure_checkpoint(4, {})
            observer.report["local_rank_global"] = 0
            observer.report["steps"] = [
                {"global_step": step, "phase": "initial"} for step in range(1, 4)
            ]
            observer.started = 0
            observer.engine_step_start = 3
            observer.engine_skipped_start = 0
            observer.pending = [{
                "input_lengths": [2], "attended_tokens": 2, "supervised_tokens": 1,
                "padded_tokens": 0, "row_indices": [96 + micro * 8],
            } for micro in range(4)]
            args = SimpleNamespace(gradient_accumulation_steps=4, per_device_train_batch_size=1)
            observer.trainer = SimpleNamespace(
                args=args, validation_fixed_order=True,
                model_wrapped=SimpleNamespace(global_steps=4, skipped_steps=0),
                accelerator=SimpleNamespace(optimizer_step_was_skipped=False),
            )
            def exchange(item, cfg):
                if not isinstance(item, dict):
                    return [item] * 8
                return [{
                    **item, "rank": rank, "row_indices": [96 + micro * 8 + rank for micro in range(4)],
                } for rank in range(8)]
            control = SimpleNamespace(should_save=False, should_training_stop=False)
            with patch("sft.bounded_validation.exchange", side_effect=exchange), patch(
                "torch.cuda.synchronize"
            ), patch("torch.cuda.max_memory_allocated", return_value=100), patch(
                "torch.cuda.max_memory_reserved", return_value=200
            ), patch("sft.bounded_validation.time.perf_counter", return_value=1):
                observer.on_step_end(args, SimpleNamespace(global_step=4), control)
            self.assertTrue(control.should_save)
            self.assertTrue(control.should_training_stop)
            self.assertEqual(observer.report["steps"][-1]["phase"], "initial")
            observer.report["status"] = "checkpoint_saved"
            with patch("sft.bounded_validation.exchange", side_effect=lambda item, cfg: [item] * 8):
                observer.finish_checkpoint_phase(4, 1.0)
                with self.assertRaisesRegex(RuntimeError, "exactly"):
                    observer.finish_checkpoint_phase(8, 1.0)
            resumed = bv.ValidationMeasurements(config(), tmp)
            resumed.configure_checkpoint(4, {}, "checkpoint-4", {
                "measurements": {
                    "steps": [], "phases": [{
                        "phase": "initial", "start_global_step": 0, "end_global_step": 4,
                        "training_wall_seconds": 12,
                    }],
                },
            })
            resumed.report["local_rank_global"] = 0
            resumed.report["steps"] = [{
                "global_step": step, "is_warmup": step <= 2,
                "phase": "initial" if step <= 4 else "resumed",
                "elapsed_seconds": 2, "attended_tokens": 64, "supervised_tokens": 32,
                "examples": 32, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
            } for step in range(1, 9)]
            resumed.training_started = 0
            with patch("sft.bounded_validation.exchange", side_effect=lambda item, cfg: [item] * 8), patch(
                "sft.bounded_validation.time.perf_counter", return_value=20
            ):
                resumed.finish(8, 0.9)
            self.assertEqual(resumed.report["measured_steps"], 6)
            self.assertEqual(resumed.report["measured_seconds"], 12)
            self.assertEqual(resumed.report["phase_training_wall_seconds_sum"], 32)
            self.assertIn("not a contiguous", resumed.report["throughput_scope"])
            self.assertEqual([step["global_step"] for step in resumed.report["steps"]], list(range(1, 9)))

    def test_main_checkpoint_api_preserves_eight_total_steps_and_defers_initial_export(self):
        for resume_phase, ds_block in ((False, None), (True, None), (False, "optimizer"), (False, "scheduler")):
            with self.subTest(resume=resume_phase, ds_block=ds_block), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                path = Path(tmp) / "checkpoint-4"
                if resume_phase:
                    checkpoint_files(path)
                    manifest(path, {})
                train = Path(tmp) / "train.jsonl"
                train.write_text("{}\n")
                ds = Path(tmp) / "zero3.json"
                ds.write_text(json.dumps({
                    "zero_optimization": {"stage": 3}, **({ds_block: {}} if ds_block else {}),
                }))
                cfg = sft_train.SFTConfig(
                    step_key="test", base_model="unused", train_jsonl=str(train), val_jsonl=str(train),
                    output_dir=tmp, deepspeed_config=str(ds), enable_thinking=False,
                    gradient_checkpointing=False, per_device_train_batch_size=1, gradient_accumulation_steps=4,
                    eval_strategy="no", save_strategy="no", validation_fixed_order=True,
                    validation_checkpoint_step=4,
                    resume_from_checkpoint=str(path) if resume_phase else None,
                    bounded_validation={"expected_steps": 8, "warmup_steps": 2, "expected_world_size": 8},
                )
                model_config = SimpleNamespace(model_type="qwen3_5", to_dict=lambda: {"model_type": "qwen3_5"})
                model = SimpleNamespace(config=model_config, parameters=lambda: [SimpleNamespace(ds_id=1)])
                model_class = Mock()
                model_class.from_pretrained.return_value = model
                observer = Mock(report={})
                trainer = SimpleNamespace(
                    train=Mock(return_value=SimpleNamespace(training_loss=1.0)),
                    state=SimpleNamespace(global_step=8 if resume_phase else 4, best_metric=None, best_model_checkpoint=None),
                    save_model=Mock(), save_state=Mock(), is_world_process_zero=lambda: True,
                )
                def dataset(*args, validation_stats, **kwargs):
                    validation_stats.update({"skipped": 0})
                    return [{"input_ids": [1, 2], "_validation_row_index": i} for i in range(256)]
                def validate(path_arg, contract, config_arg, checkpoint_step, verify_files=True):
                    self.assertEqual(contract["max_steps"], 8)
                    self.assertEqual(contract["config"]["validation_checkpoint_step"], 4)
                    self.assertEqual(checkpoint_step, 4)
                    return {"measurements": {"steps": []}}
                stack.enter_context(patch("sys.argv", [
                    "sft_train.py", "--config", "unused", "--max-steps", "8", "--no-wandb",
                ]))
                for name, replacement in (
                    ("SFTConfig.from_yaml", Mock(return_value=cfg)), ("setup_deepspeed", Mock()),
                    ("AutoConfig.from_pretrained", Mock(return_value=model_config)),
                    ("AutoTokenizer.from_pretrained", Mock(return_value=Mock(pad_token="pad", pad_token_id=0))),
                    ("model_class_for_config", Mock(return_value=model_class)),
                    ("configure_qwen35_text_training", Mock()),
                    ("build_dataset", Mock(side_effect=dataset)),
                    ("TrainingArguments", Mock(return_value=SimpleNamespace(world_size=8, process_index=0))),
                    ("ValidationMeasurements", Mock(return_value=observer)),
                    ("BoundedValidationTrainer", Mock(return_value=trainer)),
                    ("validate_checkpoint", Mock(side_effect=validate)),
                    ("exchange", Mock(side_effect=lambda item, cfg: [item] * 8)),
                ):
                    stack.enter_context(patch(f"sft.sft_train.{name}", new=replacement))
                stack.enter_context(patch("importlib.metadata.version", return_value="mock-version"))
                stack.enter_context(patch("torch.distributed.barrier"))
                stack.enter_context(patch("sft.bounded_validation.exchange", side_effect=lambda item, cfg: [item] * 8))
                stack.enter_context(patch(
                    "transformers.integrations.deepspeed.is_deepspeed_zero3_enabled", return_value=True,
                ))
                if ds_block:
                    with self.assertRaisesRegex(ValueError, "external HF scheduler"):
                        sft_train.main()
                    model_class.from_pretrained.assert_not_called()
                    continue
                sft_train.main()
                self.assertEqual(sft_train.TrainingArguments.call_args.kwargs["max_steps"], 8)
                self.assertIs(sft_train.TrainingArguments.call_args.kwargs["ignore_data_skip"], False)
                self.assertEqual(trainer.train.call_args.kwargs["resume_from_checkpoint"], str(path) if resume_phase else None)
                if resume_phase:
                    observer.finish.assert_called_once_with(8, 1.0)
                    trainer.save_model.assert_called_once()
                else:
                    observer.finish_checkpoint_phase.assert_called_once_with(4, 1.0)
                    observer.finish.assert_not_called()
                    trainer.save_model.assert_not_called()
                    self.assertFalse((Path(tmp) / "training_summary.json").exists())

    def test_actual_native_rng_save_restore_matches_production_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            set_seed(42)
            model = Qwen3_5ForConditionalGeneration(test_sft_training.ModelTests.tiny_config())
            observer = bv.ValidationMeasurements(bv.BoundedValidationConfig(), tmp)
            trainer = bv.BoundedValidationTrainer(
                model=model, measurements=observer, fixed_order=True,
                args=TrainingArguments(output_dir=tmp, use_cpu=True, report_to="none"),
            )
            observer.configure_checkpoint(4, {})
            trainer._save_rng_state(tmp)
            saved = trainer._saved_rng_sha256
            bv.atomic_json(Path(tmp) / "checkpoint_rank0.json", {"rng_sha256": saved})
            random.random()
            np.random.random()
            torch.rand(5)
            self.assertNotEqual(saved, bv.rng_sha256())
            observer.resume_manifest = {}
            trainer._load_rng_state(tmp)
            self.assertEqual(saved, bv.rng_sha256())
            self.assertTrue(observer.report["checkpoint_validation"]["rng_restored"])

    def test_invalid_checkpoint_configuration_fails_before_any_model_load(self):
        base = dict(
            step_key="test", base_model="unused", train_jsonl="unused", val_jsonl="unused",
            output_dir="unused", deepspeed_config="unused", validation_checkpoint_step=4,
            eval_strategy="no", save_strategy="no",
            bounded_validation={"expected_steps": 8, "warmup_steps": 2, "expected_world_size": 8},
        )
        for changes in (
            {"validation_checkpoint_step": True}, {"validation_checkpoint_step": 2},
            {"validation_checkpoint_step": 8}, {"resume_from_checkpoint": "auto"},
            {"resume_from_checkpoint": "different-output\\checkpoint-4"},
            {"eval_strategy": "steps"}, {"save_strategy": "steps"},
            {"bounded_validation": {"expected_steps": 9, "warmup_steps": 2, "expected_world_size": 8}},
            {"bounded_validation": {"expected_steps": 8, "warmup_steps": 2, "expected_world_size": 1}},
        ):
            cfg = sft_train.SFTConfig(**{**base, **changes})
            max_steps = cfg.bounded_validation["expected_steps"]
            with self.subTest(changes=changes), patch("sys.argv", [
                "sft_train.py", "--config", "unused", "--max-steps", str(max_steps),
            ]), patch.object(sft_train.SFTConfig, "from_yaml", return_value=cfg), patch.object(
                sft_train, "setup_deepspeed"
            ), patch.object(sft_train.AutoConfig, "from_pretrained") as load:
                with self.assertRaises(ValueError):
                    sft_train.main()
                load.assert_not_called()

    @unittest.skipUnless(shutil.which("git"), "Git archive required")
    def test_exact_committed_archive_validates_full_checkpoint_inventory(self):
        root = Path(__file__).resolve().parents[2]
        raw = subprocess.check_output([
            "git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
            "archive", "--format=tar", "HEAD", "sft/bounded_validation.py",
        ], cwd=root, timeout=30)
        with tempfile.TemporaryDirectory() as tmp, tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            source = archive.extractfile("sft/bounded_validation.py").read()
            blob = subprocess.check_output(["git", "show", "HEAD:sft/bounded_validation.py"], cwd=root, timeout=30)
            self.assertEqual(source, blob)
            path = Path(tmp) / "bounded_validation.py"
            path.write_bytes(source)
            spec = importlib.util.spec_from_file_location("archived_checkpoint_validation", path)
            module = importlib.util.module_from_spec(spec)
            with patch.dict("sys.modules", {spec.name: module}):
                spec.loader.exec_module(module)
                checkpoint = Path(tmp) / "checkpoint-4"
                checkpoint_files(checkpoint)
                expected = manifest(checkpoint, {"world_size": 8})
                actual = module.validate_checkpoint(checkpoint, {"world_size": 8}, config(), 4)
                self.assertEqual(actual, expected)
                self.assertEqual(module.state_sha256(torch.ones(5, dtype=torch.bfloat16)),
                                 bv.state_sha256(torch.ones(5, dtype=torch.bfloat16)))
                (checkpoint / "rng_state_7.pth").unlink()
                with self.assertRaisesRegex(ValueError, "rng_state_7"):
                    module.validate_checkpoint(checkpoint, {"world_size": 8}, config(), 4)


class QuietCPUObserver(bv.ValidationMeasurements):
    """Real native CPU resume test; explicitly no DS/CUDA evidence."""
    def on_train_begin(self, *args, **kwargs):
        pass

    def on_step_begin(self, *args, **kwargs):
        pass

    def on_step_end(self, *args, **kwargs):
        pass

    def record_batch(self, *args, **kwargs):
        pass

    def record_loss(self, *args, **kwargs):
        pass


class StopAtTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_save = True
            control.should_training_stop = True


class RecordingCPUTrainer(bv.BoundedValidationTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        self.observed_rows.extend(inputs["_validation_row_index"].tolist())
        return super().training_step(model, inputs, num_items_in_batch)


class ActualCPUResumeTests(unittest.TestCase):
    def test_native_optimizer_scheduler_rng_and_fixed_or_shuffled_resume_match_uninterrupted(self):
        for fixed in (True, False):
            with self.subTest(fixed=fixed), tempfile.TemporaryDirectory() as tmp:
                rows = [{
                    "input_ids": [index + 4, 5, 6, 7], "attention_mask": [1] * 4,
                    "labels": [-100, -100, 6, 7], "_validation_row_index": index,
                } for index in range(8)]
                def make_trainer(output, stop=False):
                    set_seed(42)
                    tiny = test_sft_training.ModelTests.tiny_config()
                    tiny.text_config.attention_dropout = 0.1
                    model = Qwen3_5ForConditionalGeneration(tiny)
                    configure_qwen35_text_training(model)
                    observer = QuietCPUObserver(bv.BoundedValidationConfig(expected_steps=4, warmup_steps=1), output)
                    trainer = RecordingCPUTrainer(
                        model=model, measurements=observer, fixed_order=fixed,
                        args=TrainingArguments(
                            output_dir=output, use_cpu=True, max_steps=4,
                            per_device_train_batch_size=1, gradient_accumulation_steps=2,
                            learning_rate=5e-4, warmup_steps=1, lr_scheduler_type="cosine",
                            save_strategy="no", report_to="none", disable_tqdm=True,
                            dataloader_num_workers=0, remove_unused_columns=False,
                            data_seed=42, accelerator_config={"use_seedable_sampler": True},
                        ),
                        train_dataset=rows, data_collator=DataCollatorForSFT(0),
                        callbacks=[StopAtTwo()] if stop else [],
                    )
                    trainer.observed_rows = []
                    return trainer
                full = make_trainer(str(Path(tmp) / "full"))
                full.train()
                first = make_trainer(str(Path(tmp) / "split"), stop=True)
                first.train()
                self.assertEqual(first.state.global_step, 2)
                checkpoint = Path(first.args.output_dir) / "checkpoint-2"
                for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
                    self.assertTrue((checkpoint / name).is_file())
                resumed = make_trainer(first.args.output_dir)
                resumed.train(resume_from_checkpoint=str(checkpoint))
                self.assertEqual(resumed.state.global_step, 4)
                self.assertEqual(first.observed_rows + resumed.observed_rows, full.observed_rows)
                if not fixed:
                    self.assertNotEqual(full.observed_rows, list(range(8)))
                for expected, actual in zip(full.model.parameters(), resumed.model.parameters()):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                self.assertEqual(bv.state_sha256(full.optimizer.state_dict()),
                                 bv.state_sha256(resumed.optimizer.state_dict()))
                self.assertEqual(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
                self.assertEqual(full.state.max_steps, resumed.state.max_steps)


if __name__ == "__main__":
    unittest.main()
