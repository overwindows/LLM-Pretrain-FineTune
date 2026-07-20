"""Eval stage: runs model evaluation against M1 (rule-based) and M2 (LLM judge) metrics.

Evaluation harness:
  1. Serve the checkpoint with vLLM (background process)
  2. Sample N completions from the test set
  3. Score with M1 (deterministic rule checker)
  4. Score with M2 (LLM judge — calls Azure OpenAI)
  5. Aggregate and write eval_summary.json to Cosmos
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .cosmos import cosmos_path, rsync_to_cosmos
from .env_setup import build_subprocess_env, conda_bin, setup_training_env
from .utils import run_cmd

logger = logging.getLogger(__name__)

_DEFAULT_EASYTRAIN = os.environ.get(
    "EASYTRAIN_ROOT",
    str(Path(__file__).parent.parent.parent / "LLM-Pretrain-FineTune"),
)


@dataclass
class EvalResult:
    output_dir: str
    cosmos_eval_dir: str
    checkpoint_path: str
    n_samples: int
    m1_score: Optional[float]
    m2_score: Optional[float]
    metrics: dict = field(default_factory=dict)


class EvalStage:
    """Evaluate a checkpoint using the EasyPosttrain eval harness.

    Parameters
    ----------
    cfg : dict
        The ``eval`` sub-dict from project.yaml.
    rl_checkpoint_dir : str
        Path to the RL checkpoint directory (preferred eval target).
    sft_checkpoint : str
        Fallback SFT checkpoint path if no RL checkpoint is available.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        cosmos_root: str,
        run_name: str,
        rl_checkpoint_dir: str,
        sft_checkpoint: str,
        local_scratch_root: str,
        cosmos_persist_root: str,
    ):
        self.cfg = cfg
        self.cosmos_root = cosmos_root
        self.run_name = run_name
        self.rl_checkpoint_dir = rl_checkpoint_dir
        self.sft_checkpoint = sft_checkpoint
        self.local_output_dir = os.path.join(local_scratch_root, "eval")
        self.cosmos_eval_dir = cosmos_path(cosmos_persist_root, "eval")
        easytrain_root = cfg.get("easytrain_root") or _DEFAULT_EASYTRAIN
        eval_dir = cfg.get("easytrain_eval_dir") or str(Path(easytrain_root) / "eval")
        self._easytrain_eval = Path(eval_dir)
        self._repo_root = Path(__file__).parent.parent

    def run(self) -> EvalResult:
        logger.info("=== Eval stage: %s ===", self.run_name)
        os.makedirs(self.local_output_dir, exist_ok=True)

        checkpoint = self._pick_checkpoint()
        logger.info("Evaluating checkpoint: %s", checkpoint)

        metrics = self.cfg.get("metrics", ["m1_rule", "m2_llm"])
        n_samples = self.cfg.get("n_test_samples", 1000)

        all_results: dict = {}

        if "m1_rule" in metrics:
            m1_result = self._run_m1(checkpoint, n_samples)
            all_results.update(m1_result)

        if "m2_llm" in metrics:
            m2_result = self._run_m2(checkpoint, n_samples)
            all_results.update(m2_result)

        summary = {
            "run_name": self.run_name,
            "checkpoint": checkpoint,
            "n_samples": n_samples,
            **all_results,
        }
        summary_path = os.path.join(self.local_output_dir, "eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Eval summary: %s", summary)

        rsync_to_cosmos(self.local_output_dir, self.cosmos_eval_dir)

        return EvalResult(
            output_dir=self.local_output_dir,
            cosmos_eval_dir=self.cosmos_eval_dir,
            checkpoint_path=checkpoint,
            n_samples=n_samples,
            m1_score=all_results.get("m1_score"),
            m2_score=all_results.get("m2_score"),
            metrics=all_results,
        )

    # ------------------------------------------------------------------

    def _pick_checkpoint(self) -> str:
        """Pick best checkpoint: prefer RL output, fall back to SFT."""
        from .cosmos import latest_checkpoint
        rl_ckpt = latest_checkpoint(self.rl_checkpoint_dir)
        if rl_ckpt and os.path.isdir(rl_ckpt):
            logger.info("Using RL checkpoint: %s", rl_ckpt)
            return rl_ckpt
        if self.sft_checkpoint and os.path.isdir(self.sft_checkpoint):
            logger.info("Falling back to SFT checkpoint: %s", self.sft_checkpoint)
            return self.sft_checkpoint
        raise FileNotFoundError(
            f"No valid checkpoint found.\n"
            f"  RL dir: {self.rl_checkpoint_dir}\n"
            f"  SFT ckpt: {self.sft_checkpoint}"
        )

    def _run_m1(self, checkpoint: str, n_samples: int) -> dict:
        """Run rule-based M1 eval."""
        script = str(self._easytrain_eval / "eval_m1_rule.py")
        output_file = os.path.join(self.local_output_dir, "m1_results.json")

        test_data = self.cfg.get("test_jsonl") or str(
            Path(_DEFAULT_EASYTRAIN) / "eval" / "test_data" / "layer1_test.jsonl"
        )
        conda_env = self.cfg.get("conda_env", "ptca")
        env_overlay = setup_training_env(
            secrets_file=self.cfg.get("secrets_file", "~/.secrets/maiprofile_sft.env"),
        )
        env = build_subprocess_env(env_overlay)
        python = conda_bin(conda_env, "python")

        cmd = [
            python, script,
            "--model", checkpoint,
            "--test-data", test_data,
            "--output", output_file,
            "--n-samples", str(n_samples),
            "--max-model-len", str(self.cfg.get("max_model_len", 40960)),
        ]
        run_cmd(cmd, env=env, cwd=str(self._easytrain_eval), log_prefix="EvalM1")

        score = None
        if os.path.isfile(output_file):
            with open(output_file) as f:
                data = json.load(f)
            score = data.get("overall_score") or data.get("f1_score")
        return {"m1_score": score, "m1_output": output_file}

    def _run_m2(self, checkpoint: str, n_samples: int) -> dict:
        """Run LLM-judge M2 eval."""
        script = str(self._easytrain_eval / "eval_m2_llm_judge.py")
        output_file = os.path.join(self.local_output_dir, "m2_results.json")

        test_data = self.cfg.get("test_jsonl") or str(
            Path(_DEFAULT_EASYTRAIN) / "eval" / "test_data" / "layer1_test.jsonl"
        )
        conda_env = self.cfg.get("conda_env", "ptca")
        env_overlay = setup_training_env(
            secrets_file=self.cfg.get("secrets_file", "~/.secrets/maiprofile_sft.env"),
            required_secrets=["AZURE_OPENAI_KEY"],
        )
        env = build_subprocess_env(env_overlay)
        python = conda_bin(conda_env, "python")

        cmd = [
            python, script,
            "--model", checkpoint,
            "--test-data", test_data,
            "--output", output_file,
            "--n-samples", str(n_samples),
            "--judge-model", self.cfg.get("judge_model", "gpt-4o"),
        ]
        run_cmd(cmd, env=env, cwd=str(self._easytrain_eval), log_prefix="EvalM2")

        score = None
        if os.path.isfile(output_file):
            with open(output_file) as f:
                data = json.load(f)
            score = data.get("overall_score") or data.get("win_rate")
        return {"m2_score": score, "m2_output": output_file}
