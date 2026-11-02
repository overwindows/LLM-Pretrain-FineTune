import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.data import BatchSampler
from accelerate.data_loader import BatchSamplerShard
from transformers import Qwen3_5ForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments, set_seed

from sft import full_corpus as fc
from sft import sft_train
from sft.bounded_validation import atomic_json, file_sha256, state_sha256
from sft.sft_train import build_dataset, SFTConfig
import test_checkpoint_validation
from test_sft_training import CharacterTokenizer, record
import test_sft_training
from sft.training_utils import configure_qwen35_text_training


def file_identity(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def fixture(directory, passes=2):
    root = Path(directory)
    base = root / "base"
    base.mkdir()
    tokenizer = base / "tokenizer.json"
    tokenizer.write_text('{"fixture":"not-a-real-tokenizer"}')
    instruction = root / "instruction.txt"
    instruction.write_text("fixture instruction")
    sources, identities = {}, {}
    for task in fc.FREQUENCIES:
        source = root / f"{task}.jsonl"
        offsets = []
        with source.open("wb") as stream:
            for index in range(16):
                raw = json.dumps({"task": task, "index": index}).encode("ascii")
                offsets.append((stream.tell(), hashlib.sha256(raw).hexdigest()))
                stream.write(raw + b"\n")
        identity = file_identity(source)
        sources[task] = identity
        identities[task] = [f"{task}:{identity['sha256']}:{offset}:{sha}" for offset, sha in offsets]
    ledger_rows = []
    for task, frequency in fc.FREQUENCIES.items():
        for identity in identities[task]:
            for _ in range(frequency):
                ledger_rows.append({"task": task, "source_id": identity, "exposure_index": len(ledger_rows)})
    ledger = root / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in ledger_rows))
    train = root / "train.jsonl"
    train.write_text("".join(json.dumps({"task": row["task"], **record()}) + "\n" for row in ledger_rows))
    val = root / "val.jsonl"
    val.write_text(json.dumps(record()) + "\n")
    ds = root / "zero3.json"
    ds.write_text('{"zero_optimization":{"stage":3}}')
    cfg = SFTConfig(
        step_key="fixture", base_model=str(base), train_jsonl=str(train), val_jsonl=str(val),
        output_dir=str(root / "output"), num_train_epochs=passes, max_seq_len=17,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, seed=42,
        learning_rate=2.5e-6, weight_decay=0, optim="adamw_torch",
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        enable_thinking=False, eval_strategy="no", save_strategy="no", deepspeed_config=str(ds),
        model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        full_corpus_contract=str(root / "contract.json"), full_corpus_segment_end_step=1,
        gradient_checkpointing=False,
    )
    value = {
        "schema_version": 1, "model": {"base_model": cfg.base_model, "revision": cfg.model_revision},
        "input_files": {name: file_identity(path) for name, path in (("train", train), ("val", val), ("ledger", ledger))},
        "artifacts": {
            "sources": sources, "instructions": {"fixture": file_identity(instruction)},
            "tokenizer": {"tokenizer.json": file_identity(tokenizer)},
        },
        "source_counts": {task: 16 for task in fc.FREQUENCIES},
        "frequencies": fc.FREQUENCIES,
        "exposure_counts": {task: 16 * frequency for task, frequency in fc.FREQUENCIES.items()},
        "full_passes": passes, "total_max_steps": 2 * passes,
        "segment_end_steps": list(range(1, 2 * passes + 1)),
        "training": {
            name: getattr(cfg, name) for name in (
                "per_device_train_batch_size", "gradient_accumulation_steps", "seed",
                "learning_rate", "weight_decay", "optim", "adam_beta1", "adam_beta2", "adam_epsilon",
                "lr_scheduler_type", "warmup_ratio", "max_seq_len", "enable_thinking", "mask_think_in_loss",
            )
        },
        "sampler": {"algorithm": fc.SAMPLER_ALGORITHM, "permutation_sha256": [
            fc.permutation_sha256(torch.randperm(64, generator=torch.Generator().manual_seed(42 + epoch)))
            for epoch in range(passes)
        ]},
        "quality_train": False,
        "export_at_segment_end": False,
    }
    value["training"]["world_size"] = 8
    atomic_json(Path(cfg.full_corpus_contract), value)
    return cfg, value


def evidence(contract, step):
    indices = contract.indices(step)
    return {"global_step": step, "ranks": [{
        "rank": rank, "row_indices": indices[rank::8], "engine_global_steps": step,
        "optimizer_updates": 1, "skipped_updates": 0, "losses": [1.0] * 4,
    } for rank in range(8)]}


class FullCorpusContractTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("SFT_ARROW64_LARGE_REGRESSION") == "1",
        "Opt-in real >2GiB Arrow fingerprint regression requires measured host RAM",
    )
    def test_real_over_2gib_reused_string_buffers_reject_arrow32_and_fingerprint_arrow64(self):
        import pyarrow as pa
        from datasets import Dataset

        payload = "x" * (1024 * 1024)
        chunk = pa.array([payload], type=pa.string())
        count = 2049
        narrow = pa.chunked_array([chunk] * count)
        self.assertGreater(count * len(payload), 2 ** 31 - 1)
        with self.assertRaisesRegex(pa.ArrowInvalid, "offset overflow"):
            narrow.combine_chunks()
        wide = pa.chunked_array([chunk.cast(pa.large_string())] * count)
        dataset = Dataset(pa.Table.from_arrays([wide], names=["content"]))
        self.assertTrue(dataset._fingerprint)
        self.assertEqual(dataset.num_rows, count)
        self.assertEqual(dataset[0]["content"], payload)
        self.assertEqual(dataset[count - 1]["content"], payload)
        combined = dataset.data.column("content").combine_chunks()
        self.assertEqual(combined.buffers()[2].size, count * len(payload))
        self.assertTrue(pa.types.is_large_string(combined.type))

    def test_arrow64_raw_dataset_preserves_every_field_order_and_native_fingerprint(self):
        import pyarrow as pa
        rows = [{"task": task, "messages": [
            {"role": "system", "content": "instruction"},
            {"role": "user", "content": f"unicode \u4e2d {index}"},
            {"role": "assistant", "content": ""},
        ]} for index, task in enumerate(("gem", "hpf", "mai_profile"))]
        dataset = fc.full_corpus_raw_dataset(rows)
        self.assertEqual(list(dataset), rows)
        self.assertTrue(pa.types.is_large_string(dataset.data.schema.field("task").type))
        messages = dataset.data.schema.field("messages").type
        self.assertTrue(pa.types.is_large_list(messages))
        self.assertTrue(pa.types.is_large_string(messages.value_type.field("content").type))
        self.assertTrue(dataset._fingerprint)
        self.assertEqual(fc.full_corpus_raw_dataset(rows)._fingerprint, dataset._fingerprint)
        for bad in (
            [{**rows[0], "extra": "must-not-be-discarded"}],
            [{"task": "gem", "messages": [{"role": "user", "content": "x", "extra": 1}]}],
            [{"task": "gem", "messages": [{"role": "user", "content": None}]}],
        ):
            with self.assertRaises(ValueError):
                fc.full_corpus_raw_dataset(bad)

    def test_exact_committed_full_module_archive_consumes_fixture_without_checkout_imports(self):
        root = Path(__file__).resolve().parents[2]
        raw = subprocess.check_output([
            "git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
            "archive", "--format=tar", "HEAD", "sft/full_corpus.py", "sft/bounded_validation.py",
        ], cwd=root, timeout=30)
        with tempfile.TemporaryDirectory() as tmp, tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            cfg, value = fixture(tmp)
            modules = {}
            for name in ("bounded_validation", "full_corpus"):
                source = archive.extractfile(f"sft/{name}.py").read()
                blob = subprocess.check_output(["git", "show", f"HEAD:sft/{name}.py"], cwd=root, timeout=30)
                self.assertEqual(source, blob)
                path = Path(tmp) / f"{name}.py"
                path.write_bytes(source)
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                modules[name] = module
                with patch.dict("sys.modules", modules):
                    spec.loader.exec_module(module)
            archived = modules["full_corpus"]
            raw_rows = [{"task": "gem", **record()}]
            self.assertEqual(list(archived.full_corpus_raw_dataset(raw_rows)), raw_rows)
            contract = archived.FullCorpusContract(cfg.full_corpus_contract, cfg)
            cursor = archived.CoverageCursor(contract)
            for step in range(1, 5):
                cursor.advance(evidence(contract, step))
            self.assertEqual(cursor.rank_state(0)["completed_full_passes"], 2)
            self.assertEqual(archived.contract_sha256(value), fc.contract_sha256(value))

    def test_complete_ledger_and_actual_sampler_shards_cover_two_full_passes_without_remainder(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, value = fixture(tmp)
            contract = fc.FullCorpusContract(cfg.full_corpus_contract, cfg)
            self.assertEqual(len(contract.rows), 64)
            self.assertEqual(sum(value["source_counts"].values()), 48)
            cursor = fc.CoverageCursor(contract)
            for epoch in range(2):
                sampler = fc.PassSampler(contract.size, cfg.seed)
                sampler.set_epoch(epoch)
                shards = [list(BatchSamplerShard(
                    BatchSampler(sampler, batch_size=1, drop_last=False),
                    num_processes=8, process_index=rank, split_batches=False, even_batches=True,
                )) for rank in range(8)]
                seen = []
                for offset in range(2):
                    step = epoch * 2 + offset + 1
                    actual = [[batch[0] for batch in shard[offset * 4:offset * 4 + 4]] for shard in shards]
                    self.assertEqual(actual, [contract.indices(step)[rank::8] for rank in range(8)])
                    seen.extend(index for row in actual for index in row)
                    cursor.advance(evidence(contract, step))
                self.assertEqual(sorted(seen), list(range(64)))
            self.assertEqual(cursor.rank_state(7)["completed_full_passes"], 2)
            self.assertEqual(cursor.rank_state(7)["consumed_exposures"], 16)
            self.assertEqual(sorted(contract.indices(1) + contract.indices(2)), list(range(64)))
            self.assertNotEqual(contract.indices(1), contract.indices(3))

    def test_skip_subsample_rejected_but_existing_safe_context_crop_target_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = fixture(tmp)
            contract = fc.FullCorpusContract(cfg.full_corpus_contract, cfg)
            stats = {}
            dataset = build_dataset(
                cfg.train_jsonl, CharacterTokenizer(), cfg.max_seq_len,
                enable_thinking=False, validation_stats=stats,
                record_context_crops=True,
            )
            self.assertEqual(stats["left_truncated_records"], 64)
            self.assertGreater(stats["context_tokens_removed"], 0)
            contract.validate_dataset(dataset, stats)
            self.assertEqual([label for label in dataset[0]["labels"] if label != -100], list(b"gold!"))
            for bad_stats in ({**stats, "skipped": 1}, {**stats, "kept": 63}):
                with self.assertRaisesRegex(ValueError, "skip or subsample"):
                    contract.validate_dataset(dataset, bad_stats)
            with self.assertRaisesRegex(ValueError, "skip or subsample"):
                contract.validate_dataset(dataset.select(range(32)), stats)

    def test_contract_rejects_wrong_hash_horizon_frequency_permutation_remainder_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, original = fixture(tmp)
            variants = []
            for key, replacement in (
                ("quality_train", True), ("total_max_steps", 3), ("full_passes", True),
                ("export_at_segment_end", "true"),
                ("segment_end_steps", [2, 1, 4]), ("frequencies", {"gem": 1, "hpf": 1, "mai_profile": 1}),
            ):
                variants.append({**original, key: replacement})
            variants.append({**original, "sampler": {**original["sampler"], "permutation_sha256": ["0" * 64] * 2}})
            changed = copy.deepcopy(original)
            changed["source_counts"]["gem"] += 1
            changed["exposure_counts"]["gem"] += 1
            variants.append(changed)
            changed = copy.deepcopy(original)
            changed["input_files"]["train"]["sha256"] = "0" * 64
            variants.append(changed)
            variants.append({**original, "ignored_typo": True})
            for variant in variants:
                with self.subTest(keys=variant.keys()):
                    atomic_json(Path(cfg.full_corpus_contract), variant)
                    with self.assertRaises(ValueError):
                        fc.FullCorpusContract(cfg.full_corpus_contract, cfg)

    def test_source_identity_and_exact_replica_counts_are_verified_not_just_total_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, original = fixture(tmp)
            ledger = Path(original["input_files"]["ledger"]["path"])
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            for change in ("replicas", "offset", "offset_alias", "index", "task"):
                bad = copy.deepcopy(rows)
                if change == "replicas":
                    bad[1]["source_id"] = bad[0]["source_id"]
                elif change == "offset":
                    parts = bad[0]["source_id"].split(":")
                    parts[2] = "1"
                    bad[0]["source_id"] = ":".join(parts)
                elif change == "offset_alias":
                    parts = bad[0]["source_id"].split(":")
                    parts[2] = "00"
                    bad[0]["source_id"] = ":".join(parts)
                elif change == "index":
                    bad[0]["exposure_index"] = 1
                else:
                    bad[0]["task"] = "quality"
                ledger.write_text("".join(json.dumps(row) + "\n" for row in bad))
                value = copy.deepcopy(original)
                value["input_files"]["ledger"] = file_identity(ledger)
                atomic_json(Path(cfg.full_corpus_contract), value)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    fc.FullCorpusContract(cfg.full_corpus_contract, cfg)

    def test_retained_source_record_cannot_be_omitted_by_declaring_a_smaller_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, value = fixture(tmp)
            source = Path(value["artifacts"]["sources"]["gem"]["path"])
            source.write_bytes(source.read_bytes() + b'{"extra":"retained-but-omitted"}\n')
            original_sha = value["artifacts"]["sources"]["gem"]["sha256"]
            value["artifacts"]["sources"]["gem"] = file_identity(source)
            new_sha = value["artifacts"]["sources"]["gem"]["sha256"]
            ledger = Path(value["input_files"]["ledger"]["path"])
            ledger.write_text(ledger.read_text().replace(original_sha, new_sha))
            value["input_files"]["ledger"] = file_identity(ledger)
            atomic_json(Path(cfg.full_corpus_contract), value)
            with self.assertRaisesRegex(ValueError, "Every retained"):
                fc.FullCorpusContract(cfg.full_corpus_contract, cfg)

    def test_actual_update_journal_cursor_restore_detects_replay_skip_and_failed_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = fixture(tmp)
            contract = fc.FullCorpusContract(cfg.full_corpus_contract, cfg)
            cursor = fc.CoverageCursor(contract)
            journal = Path(tmp) / "coverage.jsonl"
            first = evidence(contract, 1)
            cursor.advance(first)
            journal.write_text(json.dumps(first) + "\n")
            restored = fc.CoverageCursor(contract)
            restored.restore_journal(journal, 1)
            self.assertEqual(cursor.rank_state(0), restored.rank_state(0))
            self.assertEqual(restored.rank_state(0)["completed_full_passes"], 0)
            for mutation in ("step", "row", "update", "nan", "rank"):
                item = evidence(contract, 2)
                if mutation == "step":
                    item["global_step"] = 3
                elif mutation == "row":
                    item["ranks"][0]["row_indices"][0] = first["ranks"][0]["row_indices"][0]
                elif mutation == "update":
                    item["ranks"][0]["skipped_updates"] = 1
                elif mutation == "nan":
                    item["ranks"][0]["losses"][0] = float("nan")
                else:
                    item["ranks"][0]["rank"] = 7
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    fc.CoverageCursor(contract).advance(item)
            restored.advance(evidence(contract, 2))
            self.assertEqual(restored.rank_state(0)["completed_full_passes"], 1)

    def test_complete_full_state_checkpoint_requires_rank_shards_horizon_and_coverage_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = fixture(tmp)
            contract = fc.FullCorpusContract(cfg.full_corpus_contract, cfg)
            path = Path(tmp) / "checkpoint-1"
            test_checkpoint_validation.checkpoint_files(path, step=1)
            atomic_json(path / "trainer_state.json", {"global_step": 1, "max_steps": 4})
            cursor = fc.CoverageCursor(contract)
            item = evidence(contract, 1)
            cursor.advance(item)
            (path / "coverage.jsonl").write_text(json.dumps(item) + "\n")
            for rank in range(8):
                proof_path = path / f"checkpoint_rank{rank}.json"
                proof = json.loads(proof_path.read_text())
                proof["coverage"] = cursor.rank_state(rank)
                atomic_json(proof_path, proof)
            runtime = {"exact_fixture_runtime": True}
            manifest = {
                "schema_version": 1, "checkpoint_step": 1, "runtime_contract": runtime,
                "files": fc.full_inventory(path, 1),
            }
            atomic_json(path / "full_corpus_checkpoint.json", manifest)
            self.assertEqual(fc.validate_full_checkpoint(path, runtime, contract), manifest)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                fc.validate_full_checkpoint(path, {}, contract)
            (path / "rng_state_7.pth").unlink()
            with self.assertRaisesRegex(ValueError, "rng_state_7"):
                fc.validate_full_checkpoint(path, runtime, contract)

    def test_read_only_downloaded_segment_verifier_recomputes_counts_rank_hashes_and_hf_export(self):
        for step in (1, 2):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as tmp:
                cfg, value = fixture(tmp, passes=1)
                value["export_at_segment_end"] = True
                atomic_json(Path(cfg.full_corpus_contract), value)
                contract = fc.FullCorpusContract(cfg.full_corpus_contract, cfg)
                output = Path(cfg.output_dir)
                path = output / f"checkpoint-{step}"
                test_checkpoint_validation.checkpoint_files(path, step=step)
                atomic_json(path / "trainer_state.json", {"global_step": step, "max_steps": 2})
                cursor = fc.CoverageCursor(contract)
                journal = []
                for update in range(1, step + 1):
                    item = evidence(contract, update)
                    cursor.advance(item)
                    journal.append(json.dumps(item) + "\n")
                (path / "coverage.jsonl").write_text("".join(journal))
                saved = []
                for rank in range(8):
                    proof = {
                        "rank": rank, "rng_sha256": "0" * 64, "coverage": cursor.rank_state(rank),
                        "state": {
                            "global_step": step, "optimizer_partition_count": 1, "optimizer_steps": [step],
                            "model_sha256": "0" * 64, "optimizer_sha256": "1" * 64, "scheduler_sha256": "2" * 64,
                        },
                    }
                    saved.append(proof)
                    atomic_json(path / f"checkpoint_rank{rank}.json", proof)
                runtime = {"corpus": contract.value, "scope": "fixture-not-GPU-runtime"}
                marker = {
                    "schema_version": 1, "checkpoint_step": step, "runtime_contract": runtime,
                    "files": fc.full_inventory(path, step), "saved_state_ranks": saved,
                }
                atomic_json(path / "full_corpus_checkpoint.json", marker)
                for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
                    atomic_json(output / name, {"scope": "fixture"})
                (output / "chat_template.jinja").write_text("fixture")
                (output / "model.safetensors").write_bytes(b"fixture-not-loadable-model")
                complete = step == 2
                report = {
                    "schema_version": 1, "status": "corpus_complete" if complete else "segment_complete",
                    "corpus_complete": complete, "global_step": step, "segment_end_step": step,
                    "start_global_step": step - 1, "completed_full_passes": int(complete), "total_max_steps": 2,
                    "contract_sha256": fc.contract_sha256(contract.value),
                    "runtime_contract_sha256": fc.contract_sha256(runtime),
                    "checkpoint_manifest_sha256": file_sha256(path / "full_corpus_checkpoint.json"),
                    "coverage_ranks": [cursor.rank_state(rank) for rank in range(8)],
                    "consumed_counts": cursor.consumed_counts(), "saved_state_ranks": saved,
                    "segment_export_complete": True, "segment_export_files": fc.segment_export_inventory(output),
                }
                if complete:
                    report.update({
                        "export_complete": True, "full_pass_source_counts": value["source_counts"],
                        "full_pass_exposure_counts": value["exposure_counts"], "total_completed_exposures": 64,
                    })
                for name in ("full_corpus.json", *(f"full_corpus.rank{rank}.json" for rank in range(8))):
                    atomic_json(output / name, report)
                verified = fc.verify_full_segment(output, contract, runtime)
                self.assertEqual(verified["completed_exposures"], step * 32)
                self.assertIs(verified["corpus_complete"], complete)
                self.assertEqual(verified["consumed_counts"], cursor.consumed_counts())
                if complete:
                    with self.assertRaisesRegex(ValueError, "Incompatible"):
                        fc.validate_full_checkpoint(path, runtime, contract)
                with self.assertRaisesRegex(ValueError, "audited"):
                    fc.verify_full_segment(output, contract, {})
                bad = copy.deepcopy(report)
                bad["consumed_counts"]["exposure_counts"]["hpf"] += 1
                atomic_json(output / "full_corpus.json", bad)
                with self.assertRaisesRegex(ValueError, "completed-update replay"):
                    fc.verify_full_segment(output, contract)
                atomic_json(output / "full_corpus.json", report)
                (output / "model.safetensors").write_bytes(b"renamed-old-model")
                with self.assertRaisesRegex(ValueError, "export file inventory/hash"):
                    fc.verify_full_segment(output, contract)


class FullCorpusRuntimeTests(unittest.TestCase):
    def test_only_completed_collective_update_clears_intent_and_advances_durable_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = fixture(tmp)
            contract = fc.FullCorpusContract(cfg.full_corpus_contract, cfg)
            contract.dataset_stats, contract.preprocessing_per_task = {}, {}
            observer = fc.FullCorpusMeasurements(contract, cfg.output_dir, 1, {})
            observer.output_dir.mkdir()
            observer.journal.write_text("")
            engine = SimpleNamespace(global_steps=0, skipped_steps=0)
            args = SimpleNamespace(process_index=0)
            observer.trainer = SimpleNamespace(
                args=args, model_wrapped=engine, accelerator=SimpleNamespace(optimizer_step_was_skipped=False),
            )
            with patch("sft.bounded_validation.exchange", side_effect=lambda item, config: [item] * 8):
                observer.on_step_begin(args, SimpleNamespace(global_step=0), None)
                self.assertTrue((observer.output_dir / "full_corpus_inflight.json").exists())
                self.assertEqual(observer.cursor.step, 0)
                with self.assertRaisesRegex(RuntimeError, "in-flight"):
                    observer.on_step_begin(args, SimpleNamespace(global_step=0), None)
                for index in contract.indices(1)[0::8]:
                    observer.record_batch(torch.tensor([index]))
                    observer.record_loss(torch.tensor(1.0))
                self.assertEqual(observer.cursor.step, 0)
                self.assertEqual(observer.journal.read_text(), "")
                engine.global_steps = 1
                control = SimpleNamespace(should_save=False, should_training_stop=False)
                with patch("sft.full_corpus.exchange", return_value=evidence(contract, 1)["ranks"]):
                    observer.on_step_end(args, SimpleNamespace(global_step=1), control)
                self.assertTrue(control.should_save)
                self.assertTrue(control.should_training_stop)
                self.assertFalse((observer.output_dir / "full_corpus_inflight.json").exists())
                restored = fc.CoverageCursor(contract)
                restored.restore_journal(observer.journal, 1)
                self.assertEqual(restored.rank_state(0), observer.cursor.rank_state(0))

    def test_native_checkpoint_calls_all_rank_save_then_persists_full_state_and_cursor_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = fixture(tmp)
            contract = fc.FullCorpusContract(cfg.full_corpus_contract, cfg)
            contract.dataset_stats, contract.preprocessing_per_task = {}, {}
            observer = fc.FullCorpusMeasurements(contract, cfg.output_dir, 1, {"runtime": "fixture"})
            trainer = object.__new__(fc.FullCorpusTrainer)
            trainer.args = SimpleNamespace(
                output_dir=cfg.output_dir, process_index=0, save_only_model=False,
                ignore_data_skip=False, max_steps=4,
            )
            trainer.state = SimpleNamespace(global_step=1)
            trainer.measurements = observer
            trainer.is_world_process_zero = lambda: True
            trainer._saved_rng_sha256 = "fixture-rng"
            observer.trainer = trainer
            observer.output_dir.mkdir()
            item = evidence(contract, 1)
            observer.cursor.advance(item)
            observer.journal.write_text(json.dumps(item) + "\n")
            path = observer.output_dir / "checkpoint-1"
            def native(*args):
                test_checkpoint_validation.checkpoint_files(path, step=1)
                atomic_json(path / "trainer_state.json", {"global_step": 1, "max_steps": 4})
                for rank in range(8):
                    atomic_json(path / f"checkpoint_rank{rank}.json", {
                        "rank": rank, "state": {"global_step": 1}, "rng_sha256": "fixture-rng",
                        "coverage": observer.cursor.rank_state(rank),
                    })
            with patch.object(Trainer, "_save_checkpoint", side_effect=native) as save, patch(
                "sft.full_corpus.checkpoint_state_proof", return_value={"global_step": 1}
            ), patch("sft.bounded_validation.exchange", side_effect=lambda item, config: [item] * 8), patch(
                "sft.full_corpus.exchange", side_effect=lambda item, config: [item] * 8
            ), patch("torch.distributed.barrier"):
                trainer._save_checkpoint(None, None)
            save.assert_called_once()
            manifest = fc.validate_full_checkpoint(path, observer.runtime_contract, contract)
            self.assertEqual(manifest["checkpoint_step"], 1)
            self.assertEqual(observer.report["status"], "checkpoint_saved")
            self.assertEqual(observer.report["checkpoint_manifest_sha256"], file_sha256(path / "full_corpus_checkpoint.json"))
            trainer.args.save_only_model = True
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                trainer._save_checkpoint(None, None)

    def test_full_mode_actual_state_and_rng_restore_fail_closed_before_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = fixture(tmp)
            contract = fc.FullCorpusContract(cfg.full_corpus_contract, cfg)
            contract.dataset_stats, contract.preprocessing_per_task = {}, {}
            path = Path(tmp) / "checkpoint-1"
            path.mkdir()
            item = evidence(contract, 1)
            (path / "coverage.jsonl").write_text(json.dumps(item) + "\n")
            observer = fc.FullCorpusMeasurements(
                contract, cfg.output_dir, 2, {}, str(path), {"checkpoint_step": 1},
            )
            trainer = object.__new__(fc.FullCorpusTrainer)
            trainer.args = SimpleNamespace(process_index=0)
            trainer.measurements = observer
            observer.trainer = trainer
            saved = {
                "state": {"global_step": 1, "model_sha256": "expected"},
                "rng_sha256": "expected-rng", "coverage": observer.cursor.rank_state(0),
            }
            atomic_json(path / "checkpoint_rank0.json", saved)
            with patch.object(Trainer, "_load_rng_state") as native, patch(
                "sft.full_corpus.rng_sha256", return_value="wrong"
            ), patch("sft.bounded_validation.exchange", side_effect=lambda item, config: [item] * 8):
                with self.assertRaisesRegex(RuntimeError, "RNG restore"):
                    trainer._load_rng_state(str(path))
                native.assert_called_once()
            trainer.model_wrapped = SimpleNamespace(global_steps=1, skipped_steps=0)
            with patch("sft.full_corpus.deepspeed_proof", return_value={}), patch(
                "sft.full_corpus.checkpoint_state_proof", return_value={"global_step": 1, "model_sha256": "warmstart"}
            ), patch("sft.bounded_validation.exchange", side_effect=lambda item, config: [item] * 8):
                with self.assertRaisesRegex(RuntimeError, "full-state restore"):
                    observer.on_train_begin(trainer.args, SimpleNamespace(global_step=1), None)
            with patch("sft.bounded_validation.exchange", side_effect=lambda item, config: [item] * 8):
                with self.assertRaisesRegex(RuntimeError, "RNG restore"):
                    observer.on_step_begin(trainer.args, SimpleNamespace(global_step=1), None)

    def test_segment_finish_never_labels_partial_pass_as_corpus_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = fixture(tmp, passes=1)
            contract = fc.FullCorpusContract(cfg.full_corpus_contract, cfg)
            contract.dataset_stats = {"records": 64, "kept": 64, "skipped": 0, "left_truncated_records": 0}
            contract.preprocessing_per_task = {}
            observer = fc.FullCorpusMeasurements(contract, cfg.output_dir, 1, {})
            observer.trainer = SimpleNamespace(args=SimpleNamespace(process_index=0))
            observer.cursor.advance(evidence(contract, 1))
            observer.report["status"] = "checkpoint_saved"
            with patch("sft.bounded_validation.exchange", side_effect=lambda item, config: [item] * 8), patch(
                "sft.full_corpus.exchange", side_effect=lambda item, config: [item] * 8
            ):
                observer.finish(1, 1.0)
                self.assertFalse(observer.report["corpus_complete"])
                self.assertEqual(observer.report["status"], "segment_complete")
                self.assertEqual(observer.report["completed_full_passes"], 0)
                observer.cursor.advance(evidence(contract, 2))
                observer.end_step = 2
                observer.report["status"] = "checkpoint_saved"
                observer.finish(2, 1.0)
                self.assertTrue(observer.report["corpus_complete"])
                self.assertEqual(observer.report["total_completed_exposures"], 64)
                self.assertEqual(observer.report["full_pass_source_counts"], contract.value["source_counts"])

    def test_full_mode_rejects_diagnostic_overrides_before_loading_model_or_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = fixture(tmp)
            for flags in (["--max-steps", "8"], ["--debug"], ["--smoke-test"]):
                with patch("sys.argv", ["sft_train.py", "--config", "unused", *flags]), patch.object(
                    SFTConfig, "from_yaml", return_value=cfg
                ), patch.object(sft_train.AutoConfig, "from_pretrained") as model_load:
                    with self.assertRaisesRegex(ValueError, "no diagnostic"):
                        sft_train.main()
                    model_load.assert_not_called()

    def test_actual_main_uses_whole_horizon_and_defers_partial_segment_export(self):
        self._assert_main_export(False)

    def test_contracted_intermediate_export_does_not_claim_corpus_completion(self):
        self._assert_main_export(True)

    def _assert_main_export(self, export):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, value = fixture(tmp)
            value["export_at_segment_end"] = export
            atomic_json(Path(cfg.full_corpus_contract), value)
            Path(cfg.output_dir).mkdir()
            tokenizer = CharacterTokenizer()
            tokenizer.pad_token, tokenizer.pad_token_id = "pad", 0
            tokenizer.save_pretrained = Mock()
            model_config = SimpleNamespace(model_type="qwen3_5", to_dict=lambda: {"model_type": "qwen3_5"})
            model_class = Mock()
            model_class.from_pretrained.return_value = SimpleNamespace(
                config=model_config, parameters=lambda: [SimpleNamespace(ds_id=1)],
            )
            observer = Mock(report={"corpus_complete": False})
            trainer = Mock(state=SimpleNamespace(global_step=1, best_metric=None, best_model_checkpoint=None))
            trainer.train.return_value = SimpleNamespace(training_loss=1.0)
            versions = {"torch": "2.6.0+cu124", "transformers": "5.19.0", "accelerate": "1.15.0", "deepspeed": "0.16.9"}
            with patch("sys.argv", ["sft_train.py", "--config", "unused", "--no-wandb"]), patch.object(
                SFTConfig, "from_yaml", return_value=cfg
            ), patch.object(sft_train, "setup_deepspeed"), patch.object(
                sft_train.AutoConfig, "from_pretrained", return_value=model_config
            ), patch.object(sft_train.AutoTokenizer, "from_pretrained", return_value=tokenizer), patch.object(
                sft_train, "model_class_for_config", return_value=model_class
            ), patch.object(sft_train, "configure_qwen35_text_training"), patch.object(
                sft_train, "TrainingArguments", return_value=SimpleNamespace(world_size=8, process_index=0)
            ) as training_args, patch.object(fc, "FullCorpusMeasurements", return_value=observer), patch.object(
                fc, "FullCorpusTrainer", return_value=trainer
            ), patch.object(sft_train, "exchange", side_effect=lambda item, config: [item] * 8), patch(
                "sft.bounded_validation.exchange", side_effect=lambda item, config: [item] * 8
            ), patch("importlib.metadata.version", side_effect=lambda name: versions.get(name, "fixture")), patch(
                "transformers.integrations.deepspeed.is_deepspeed_zero3_enabled", return_value=True
            ), patch("torch.distributed.barrier"):
                sft_train.main()
            self.assertEqual(training_args.call_args.kwargs["max_steps"], 4)
            self.assertIs(training_args.call_args.kwargs["save_only_model"], False)
            self.assertIs(training_args.call_args.kwargs["ignore_data_skip"], False)
            self.assertEqual(training_args.call_args.kwargs["data_seed"], 42)
            observer.finish.assert_called_once_with(1, 1.0)
            if export:
                trainer.save_model.assert_called_once()
                trainer.save_state.assert_called_once()
                tokenizer.save_pretrained.assert_called_once_with(cfg.output_dir)
                self.assertTrue(observer.report["segment_export_complete"])
                self.assertIs(observer.report["corpus_complete"], False)
                self.assertNotIn("export_complete", observer.report)
                summary = json.loads((Path(cfg.output_dir) / "training_summary.json").read_text())
                self.assertIs(summary["full_corpus"]["corpus_complete"], False)
                self.assertTrue(summary["full_corpus"]["segment_export_complete"])
            else:
                trainer.save_model.assert_not_called()
            self.assertEqual(trainer.train.call_args.kwargs["resume_from_checkpoint"], None)


class NativeCPUObserver(TrainerCallback):
    """CPU native HF continuation only, not full-mode DS runtime evidence."""
    def __init__(self):
        self.report = {}

    def record_batch(self, indices):
        pass

    def record_loss(self, loss):
        pass


class StopAtOne(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_save = True
            control.should_training_stop = True


class NativeCPUFullSamplerTrainer(fc.FullCorpusTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        self.rows_seen.extend(inputs["_validation_row_index"].tolist())
        return super().training_step(model, inputs, num_items_in_batch)

    def _save_checkpoint(self, model, trial):
        return Trainer._save_checkpoint(self, model, trial)

    def _load_rng_state(self, checkpoint):
        return Trainer._load_rng_state(self, checkpoint)


class NativeFullSamplerResumeTests(unittest.TestCase):
    def test_actual_qwen_native_state_resume_matches_uninterrupted_across_pass_boundary(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "sft.bounded_validation.exchange", side_effect=lambda item, config: [item] * 8
        ):
            rows = [{
                "input_ids": [index % 20 + 4, 5, 6, 7], "attention_mask": [1] * 4,
                "labels": [-100, -100, 6, 7], "_validation_row_index": index,
            } for index in range(64)]
            def make(output, stop=False):
                set_seed(42)
                tiny = test_sft_training.ModelTests.tiny_config()
                tiny.text_config.attention_dropout = 0.1
                model = Qwen3_5ForConditionalGeneration(tiny)
                configure_qwen35_text_training(model)
                trainer = NativeCPUFullSamplerTrainer(
                    model=model, measurements=NativeCPUObserver(),
                    args=TrainingArguments(
                        output_dir=output, use_cpu=True, max_steps=4,
                        per_device_train_batch_size=16, gradient_accumulation_steps=2,
                        learning_rate=2.5e-4, warmup_steps=1, lr_scheduler_type="cosine",
                        optim="adamw_torch", save_strategy="no", report_to="none", disable_tqdm=True,
                        dataloader_num_workers=0, remove_unused_columns=False, data_seed=42,
                    ),
                    train_dataset=rows, data_collator=sft_train.DataCollatorForSFT(0),
                    callbacks=[StopAtOne()] if stop else [],
                )
                trainer.rows_seen = []
                return trainer
            full = make(str(Path(tmp) / "full"))
            full.train()
            first = make(str(Path(tmp) / "split"), stop=True)
            first.train()
            self.assertEqual(first.state.global_step, 1)
            first.save_model()
            self.assertTrue((Path(first.args.output_dir) / "generation_config.json").is_file())
            self.assertTrue((Path(first.args.output_dir) / "config.json").is_file())
            exported = Qwen3_5ForConditionalGeneration.from_pretrained(first.args.output_dir, local_files_only=True)
            self.assertEqual(state_sha256(first.model.state_dict()), state_sha256(exported.state_dict()))
            self.assertEqual(exported.config.model_type, "qwen3_5")
            self.assertTrue(hasattr(exported.model, "visual"))
            resumed = make(first.args.output_dir)
            resumed.train(resume_from_checkpoint=str(Path(first.args.output_dir) / "checkpoint-1"))
            self.assertEqual(resumed.state.global_step, 4)
            self.assertEqual(first.rows_seen + resumed.rows_seen, full.rows_seen)
            for epoch in range(2):
                planned = torch.randperm(64, generator=torch.Generator().manual_seed(42 + epoch)).tolist()
                self.assertEqual(full.rows_seen[epoch * 64:(epoch + 1) * 64], planned)
            self.assertEqual(state_sha256(full.model.state_dict()), state_sha256(resumed.model.state_dict()))
            self.assertEqual(state_sha256(full.optimizer.state_dict()), state_sha256(resumed.optimizer.state_dict()))
            self.assertEqual(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())


if __name__ == "__main__":
    unittest.main()
