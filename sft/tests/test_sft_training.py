"""CPU regressions; no checkpoint or dataset downloads."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import transformers
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from sft.sft_train import DataCollatorForSFT, build_dataset, compute_metrics
from sft import sft_train
from sft.training_utils import (
    _render_and_encode,
    configure_qwen35_text_training,
    find_last_assistant_span,
    model_class_for_config,
    tokenize_record,
    validate_sample_weight,
    warmup_arguments,
)


class CharacterTokenizer:
    def apply_chat_template(
        self, messages, add_generation_prompt, tokenize, **kwargs
    ):
        text = "".join(f"<{m['role']}>{m['content']}!" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}

    def encode(self, text, **kwargs):
        return list(text.encode("ascii"))


def record(prompt="long prompt " * 10, answer="gold"):
    return {"messages": [
        {"role": "system", "content": "instruction"},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]}


class TokenizationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_only_last_assistant_supervised(self):
        rec = record()
        rec["messages"][1:1] = [
            {"role": "user", "content": "demo"},
            {"role": "assistant", "content": "example"},
        ]
        out = tokenize_record(rec, self.tokenizer, 1000)
        target = self.tokenizer.encode("gold!")
        self.assertEqual([x for x in out["labels"] if x != -100], target)
        self.assertEqual(out["labels"][:-len(target)], [-100] * (len(out["labels"]) - len(target)))

    def test_left_truncation_preserves_exact_target_and_mask(self):
        rec = record()
        full = tokenize_record(rec, self.tokenizer, 1000)
        truncated = tokenize_record(rec, self.tokenizer, 17)
        self.assertEqual(len(truncated["input_ids"]), 17)
        self.assertEqual(truncated["input_ids"], full["input_ids"][-17:])
        self.assertEqual(truncated["labels"], full["labels"][-17:])
        self.assertEqual(truncated["labels"][:12], [-100] * 12)
        self.assertEqual(truncated["labels"][12:], self.tokenizer.encode("gold!"))

    def test_target_must_fit_with_preceding_context(self):
        rec = record(answer="x" * 12)
        self.assertIsNone(tokenize_record(rec, self.tokenizer, 13))
        self.assertIsNone(tokenize_record(rec, self.tokenizer, 12))
        out = tokenize_record(rec, self.tokenizer, 14)
        self.assertEqual(out["labels"][0], -100)
        self.assertEqual(out["labels"][1:], self.tokenizer.encode("x" * 12 + "!"))

    def test_prefix_mismatch_is_not_guessed(self):
        class MismatchedTokenizer(CharacterTokenizer):
            def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
                rendered = super().apply_chat_template(
                    messages, add_generation_prompt, tokenize, **kwargs
                )
                return rendered + ("<think>" if add_generation_prompt else "")
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            tokenize_record(record(), MismatchedTokenizer(), 20)

    def test_think_mask_moves_with_truncation(self):
        rec = record(answer="<think>reason</think>gold")
        out = tokenize_record(
            rec, self.tokenizer, 40, True, self.tokenizer.encode("</think>")
        )
        self.assertEqual(
            [x for x in out["labels"] if x != -100], self.tokenizer.encode("gold!")
        )

    def test_invalid_target_and_multimodal(self):
        self.assertIsNone(tokenize_record(record(answer=""), self.tokenizer, 50))
        rec = record()
        rec["messages"][-1]["content"] = [{"type": "text", "text": "gold"}]
        with self.assertRaisesRegex(ValueError, "string content"):
            tokenize_record(rec, self.tokenizer, 50)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            tokenize_record(record(), self.tokenizer, 1)

    def test_row_weights_never_silently_disappear(self):
        for weight in (2, 0, -1, 0.5, float("nan"), float("inf"), True, "1", None):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(ValueError, "sample_weight"):
                    tokenize_record({**record(), "sample_weight": weight}, self.tokenizer, 50)
        for weight in (1, 1.0):
            validate_sample_weight({"sample_weight": weight})
        validate_sample_weight(record())

    def test_dataset_rejects_weights_outside_debug_subsample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(
                json.dumps(record()) + "\n" +
                json.dumps({**record(), "sample_weight": 2}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sample_weight"):
                build_dataset(str(path), self.tokenizer, 50, debug_subsample=1)

    def test_materialized_frequencies_preserved_without_double_weighting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            gem = {**record(answer="GEM"), "task": "gem", "sample_weight": 1}
            hpf = {**record(answer="HPF"), "task": "hpf"}
            path.write_text(
                "\n".join(map(json.dumps, [gem, hpf, hpf])) + "\n", encoding="utf-8"
            )
            ds = build_dataset(str(path), self.tokenizer, 50)
            self.assertEqual(len(ds), 3)
            self.assertEqual(ds[1], ds[2])
            self.assertNotIn("sample_weight", ds.column_names)
            self.assertEqual(
                [x for x in ds[0]["labels"] if x != -100], self.tokenizer.encode("GEM!")
            )

    def test_collator_padding_and_weight_guard(self):
        features = [
            tokenize_record(record(), self.tokenizer, 20),
            tokenize_record(record(prompt="p"), self.tokenizer, 100),
        ]
        batch = DataCollatorForSFT(0)(features)
        self.assertTrue((batch["labels"][0, 20:] == -100).all())
        self.assertTrue((batch["attention_mask"][0, 20:] == 0).all())
        with self.assertRaisesRegex(ValueError, "sample_weight"):
            DataCollatorForSFT(0)([{**features[0], "sample_weight": 2}])

    def test_accuracy_uses_causal_shift(self):
        labels = np.array([[-100, -100, 2, 3]])
        predictions = np.array([[9, 2, 3, 8]])
        self.assertEqual(compute_metrics((predictions, labels)), {"token_accuracy": 1.0})

    def test_malformed_json_fails_with_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jsonl"
            path.write_text(json.dumps(record()) + "\nnot json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                build_dataset(str(path), self.tokenizer, 50)

    def test_empty_dataset_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Empty dataset"):
                build_dataset(str(path), self.tokenizer, 50)

    def test_actual_cached_qwen35_nonthinking_template(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3.5-4B",
                revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                local_files_only=True,
            )
        except OSError:
            self.skipTest("Official tokenizer metadata not cached; no network downloads")
        rec = record(prompt="Return JSON.", answer='{"ok":true}')
        ids = _render_and_encode(tokenizer, rec["messages"], False, False)
        start, end = find_last_assistant_span(tokenizer, rec["messages"], ids, False)
        full = tokenize_record(rec, tokenizer, 1000, enable_thinking=False)
        target = ids[start:end]
        truncated = tokenize_record(
            rec, tokenizer, len(target) + 1, enable_thinking=False
        )
        self.assertEqual(full["labels"][start:], target)
        self.assertEqual(truncated["labels"], [-100] + target)
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            tokenize_record(rec, tokenizer, 1000, enable_thinking=True)


class ModelTests(unittest.TestCase):
    def test_warmup_ratio_preserved_across_trainer_apis(self):
        old_api = SimpleNamespace(__dataclass_fields__={"warmup_ratio": None})
        new_api = SimpleNamespace(__dataclass_fields__={"warmup_steps": None})
        self.assertEqual(warmup_arguments(old_api, 0.03), {"warmup_ratio": 0.03})
        self.assertEqual(warmup_arguments(new_api, 0.03), {"warmup_steps": 0.03})
        self.assertEqual(warmup_arguments(old_api, 1), {"warmup_ratio": 1})
        with self.assertRaisesRegex(ValueError, "one absolute step"):
            warmup_arguments(new_api, 1)

    def test_model_family_selection_and_legacy_preservation(self):
        self.assertIs(
            model_class_for_config(Qwen3_5Config(), transformers),
            Qwen3_5ForConditionalGeneration,
        )
        self.assertIs(
            model_class_for_config(SimpleNamespace(model_type="qwen3"), transformers),
            transformers.AutoModelForCausalLM,
        )
        with self.assertRaisesRegex(RuntimeError, "requires Transformers"):
            model_class_for_config(Qwen3_5Config(), SimpleNamespace())
        for model_type in ("qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
            with self.assertRaisesRegex(ValueError, "full dense"):
                model_class_for_config(SimpleNamespace(model_type=model_type), transformers)
        config = Qwen3_5Config()
        config.architectures = ["WrongExport"]
        with self.assertRaisesRegex(ValueError, "architectures"):
            model_class_for_config(config, transformers)

    @staticmethod
    def tiny_config():
        return Qwen3_5Config(
            text_config={
                "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 16,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {
                    "rope_type": "default", "rope_theta": 10000,
                    "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2],
                },
            },
            vision_config={
                "depth": 1, "hidden_size": 32, "intermediate_size": 64,
                "num_heads": 2, "out_hidden_size": 32,
                "num_position_embeddings": 16,
            },
        )
    def test_tiny_hybrid_training_save_reload_without_deepspeed(self):
        config = self.tiny_config()
        torch.manual_seed(42)
        model = Qwen3_5ForConditionalGeneration(config)
        configure_qwen35_text_training(model)
        self.assertFalse(any(p.requires_grad for p in model.model.visual.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.model.language_model.parameters()))
        self.assertFalse(model.config.text_config.use_cache)
        before = model.get_input_embeddings().weight.detach().clone()
        sample = {
            "input_ids": [4, 5, 6, 7, 8],
            "attention_mask": [1] * 5, "labels": [-100, -100, 6, 7, 8],
        }
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=tmp, use_cpu=True, max_steps=1,
                    per_device_train_batch_size=1,
                    learning_rate=1e-3, save_strategy="no",
                    report_to="none", disable_tqdm=True,
                    remove_unused_columns=False,
                    gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                ),
                train_dataset=[sample],
                data_collator=DataCollatorForSFT(0),
            )
            result = trainer.train()
            self.assertTrue(np.isfinite(result.training_loss))
            self.assertFalse(torch.equal(before, model.get_input_embeddings().weight))
            trainer.save_model()
            trainer.save_state()
            saved_state = json.loads((Path(tmp) / "trainer_state.json").read_text())
            self.assertEqual(saved_state["global_step"], 1)
            self.assertTrue(saved_state["log_history"])
            saved_config = AutoConfig.from_pretrained(tmp)
            self.assertEqual(saved_config.model_type, "qwen3_5")
            self.assertEqual(saved_config.architectures, ["Qwen3_5ForConditionalGeneration"])
            self.assertIsNotNone(saved_config.vision_config)
            reloaded = model_class_for_config(saved_config, transformers).from_pretrained(tmp)
            model.eval()
            reloaded.eval()
            batch = DataCollatorForSFT(0)([sample])
            with torch.no_grad():
                expected = model(**batch).logits
                actual = reloaded(**batch).logits
            torch.testing.assert_close(expected, actual)

    def test_main_bounded_training_exports_evidence_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            output = root / "output"
            Qwen3_5ForConditionalGeneration(self.tiny_config()).save_pretrained(base)
            vocab = {
                "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
                "system": 4, "user": 5, "assistant": 6,
                "instruction": 7, "prompt": 8, "gold": 9,
            }
            backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                bos_token="[BOS]", eos_token="[EOS]",
                chat_template=(
                    "{% for message in messages %}"
                    "{{ message['role'] + ' ' + message['content'] + ' [EOS] ' }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
                ),
            )
            tokenizer.save_pretrained(base)
            data = root / "data.jsonl"
            data.write_text(json.dumps(record(prompt="prompt")) + "\n", encoding="utf-8")
            config_path = root / "train.yaml"
            config_path.write_text(yaml.safe_dump({
                "step_key": "tiny-test", "base_model": str(base),
                "train_jsonl": str(data), "val_jsonl": str(data),
                "output_dir": str(output), "enable_thinking": False,
                "max_seq_len": 32, "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": True,
                "eval_strategy": "no", "save_strategy": "no",
            }), encoding="utf-8")

            def cpu_arguments(**kwargs):
                kwargs["use_cpu"] = True
                kwargs["dataloader_num_workers"] = 0
                kwargs["dataloader_pin_memory"] = False
                kwargs["disable_tqdm"] = True
                return TrainingArguments(**kwargs)

            with patch("sys.argv", [
                "sft_train.py", "--config", str(config_path),
                "--max-steps", "2", "--no-wandb",
            ]), patch.object(sft_train, "TrainingArguments", side_effect=cpu_arguments):
                sft_train.main()
            summary = json.loads((output / "training_summary.json").read_text())
            self.assertEqual(summary["global_step"], 2)
            self.assertTrue(np.isfinite(summary["training_loss"]))
            self.assertEqual(summary["model_class"], "Qwen3_5ForConditionalGeneration")
            self.assertEqual(summary["model_type"], "qwen3_5")
            self.assertIs(summary["enable_thinking"], False)
            self.assertTrue((output / "trainer_state.json").is_file())
            self.assertEqual(
                AutoTokenizer.from_pretrained(output).chat_template,
                tokenizer.chat_template,
            )
            self.assertFalse(list(output.glob("checkpoint-*")))

            failed_output = root / "nonfinite-output"
            failed_config = yaml.safe_load(config_path.read_text())
            failed_config["output_dir"] = str(failed_output)
            config_path.write_text(yaml.safe_dump(failed_config), encoding="utf-8")
            with patch("sys.argv", [
                "sft_train.py", "--config", str(config_path),
                "--max-steps", "1", "--no-wandb",
            ]), patch.object(
                sft_train, "TrainingArguments", side_effect=cpu_arguments
            ), patch.object(
                Trainer, "train", return_value=SimpleNamespace(training_loss=float("nan"))
            ):
                with self.assertRaisesRegex(RuntimeError, "not finite"):
                    sft_train.main()
            self.assertFalse((failed_output / "training_summary.json").exists())
            self.assertFalse((failed_output / "model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
