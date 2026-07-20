"""Data cleaning stage: Pass A (teacher quality filter) + Pass B (rollout difficulty).

Pass A: CPU-only quality filters on the SFT data.
  - F1: remove records with JSON-parse errors in teacher output
  - F2: remove records whose teacher output fails schema validation
  - F3: remove all-hallucination records (0% fidelity)
  - F4: remove records exceeding max_prompt_tokens

Pass B: GPU rollout-based difficulty filter.
  - Generates N rollouts per record using the SFT checkpoint via vLLM
  - Drops records that are too easy (mean reward > sat_mean) or too hard
    (std < easy_std on near-zero-reward records)

Output: rl_data/{v2,v3}/train.jsonl written to Cosmos.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .cosmos import cosmos_path, rsync_to_cosmos
from .env_setup import build_subprocess_env, conda_bin, setup_training_env
from .utils import run_cmd

logger = logging.getLogger(__name__)

EASYTRAIN_ROOT = os.environ.get(
    "EASYTRAIN_ROOT",
    str(Path(__file__).parent.parent.parent / "LLM-Pretrain-FineTune"),
)


@dataclass
class DataCleanResult:
    pass_a_jsonl: str   # path to Pass A output
    rl_train_jsonl: str  # path to Pass B output (or Pass A if Pass B skipped)
    pass_a_kept: int
    pass_a_dropped: int
    pass_b_kept: Optional[int]
    pass_b_dropped: Optional[int]


class DataCleanStage:
    """Run Pass A and/or Pass B data cleaning.

    Parameters
    ----------
    cfg : dict
        The ``data_clean`` sub-dict from project.yaml.
    sft_checkpoint : str
        Path to the SFT checkpoint used for Pass B rollouts.
    sft_train_jsonl : str
        Path to the raw SFT training JSONL (input to Pass A).
    """

    def __init__(
        self,
        cfg: dict,
        *,
        cosmos_root: str,
        run_name: str,
        sft_checkpoint: str,
        sft_train_jsonl: str,
        local_scratch_root: str,
        cosmos_persist_root: str,
    ):
        self.cfg = cfg
        self.cosmos_root = cosmos_root
        self.run_name = run_name
        self.sft_checkpoint = sft_checkpoint
        self.sft_train_jsonl = sft_train_jsonl
        self.local_scratch_root = local_scratch_root
        self.cosmos_persist_root = cosmos_persist_root

        self.local_output_dir = os.path.join(local_scratch_root, "data_clean")
        self.cosmos_rl_data_dir = cosmos_path(cosmos_persist_root, "rl_data")
        self._easytrain_dc = Path(EASYTRAIN_ROOT) / "data_cleaning"

    def run(self) -> DataCleanResult:
        logger.info("=== Data cleaning stage: %s ===", self.run_name)
        os.makedirs(self.local_output_dir, exist_ok=True)

        # Pass A
        pass_a_out = os.path.join(self.local_output_dir, "v2_train.jsonl")
        kept_a, dropped_a = self._run_pass_a(self.sft_train_jsonl, pass_a_out)
        logger.info("Pass A: kept=%d dropped=%d", kept_a, dropped_a)

        # Pass B (optional — skip if sft_checkpoint unavailable or cfg disables it)
        skip_b = self.cfg.get("pass_b", {}).get("skip", False)
        pass_b_out = None
        kept_b = dropped_b = None
        if not skip_b and os.path.exists(self.sft_checkpoint):
            pass_b_jsonl = os.path.join(self.local_output_dir, "v3_train.jsonl")
            kept_b, dropped_b = self._run_pass_b(pass_a_out, pass_b_jsonl)
            logger.info("Pass B: kept=%d dropped=%d", kept_b, dropped_b)
            pass_b_out = pass_b_jsonl
        else:
            logger.info("Pass B skipped (skip=%s, ckpt_exists=%s)", skip_b, os.path.exists(self.sft_checkpoint))

        # Persist to Cosmos
        rsync_to_cosmos(self.local_output_dir, self.cosmos_rl_data_dir)

        rl_jsonl = pass_b_out or pass_a_out
        # Map to Cosmos path
        rl_cosmos = cosmos_path(
            self.cosmos_rl_data_dir,
            os.path.basename(rl_jsonl),
        )

        return DataCleanResult(
            pass_a_jsonl=cosmos_path(self.cosmos_rl_data_dir, "v2_train.jsonl"),
            rl_train_jsonl=rl_cosmos,
            pass_a_kept=kept_a,
            pass_a_dropped=dropped_a,
            pass_b_kept=kept_b,
            pass_b_dropped=dropped_b,
        )

    # ------------------------------------------------------------------

    def _run_pass_a(self, input_jsonl: str, output_jsonl: str) -> tuple[int, int]:
        """Run pass_a_teacher_quality.py — CPU filter."""
        script = str(self._easytrain_dc / "pass_a_teacher_quality.py")
        cfg_a = self.cfg.get("pass_a", {})

        verifier_dir = cfg_a.get("verifier_dir") or str(
            Path(EASYTRAIN_ROOT) / "data_cleaning" / "verifier_layer1"
        )

        env = build_subprocess_env(setup_training_env())
        env["PYTHONPATH"] = verifier_dir + ":" + env.get("PYTHONPATH", "")

        conda_env = self.cfg.get("conda_env", "ptca")
        python = conda_bin(conda_env, "python")

        cmd = [
            python, script,
            "--input", input_jsonl,
            "--output", output_jsonl,
        ]
        if cfg_a.get("max_prompt_tokens"):
            cmd += ["--max-prompt-tokens", str(cfg_a["max_prompt_tokens"])]

        run_cmd(cmd, env=env, cwd=str(self._easytrain_dc), log_prefix="PassA")

        # Count records
        kept = _count_lines(output_jsonl)
        dropped = _count_lines(input_jsonl) - kept
        return kept, dropped

    def _run_pass_b(self, input_jsonl: str, output_jsonl: str) -> tuple[int, int]:
        """Run pass_b_rollout_difficulty.py — GPU rollout filter."""
        script = str(self._easytrain_dc / "pass_b_rollout_difficulty.py")
        cfg_b = self.cfg.get("pass_b", {})
        conda_env = cfg_b.get("conda_env") or self.cfg.get("conda_env", "ptca")

        n_gpus = cfg_b.get("num_shards", 1)
        gpu_ids = ",".join(str(i) for i in range(n_gpus))

        env = build_subprocess_env(
            setup_training_env(cuda_visible_devices=gpu_ids)
        )
        python = conda_bin(conda_env, "python")

        cmd = [
            python, script,
            "--input", input_jsonl,
            "--output", output_jsonl,
            "--model", self.sft_checkpoint,
            "--n-rollouts", str(cfg_b.get("n_rollouts", 16)),
            "--temperature", str(cfg_b.get("temperature", 1.0)),
            "--top-p", str(cfg_b.get("top_p", 0.95)),
            "--max-tokens", str(cfg_b.get("max_tokens", 4096)),
            "--easy-std", str(cfg_b.get("easy_std", 0.05)),
            "--sat-mean", str(cfg_b.get("sat_mean", 0.95)),
            "--sat-std", str(cfg_b.get("sat_std", 0.03)),
            "--max-model-len", str(cfg_b.get("max_model_len", 40960)),
        ]
        run_cmd(cmd, env=env, cwd=str(self._easytrain_dc), log_prefix="PassB")

        kept = _count_lines(output_jsonl)
        dropped = _count_lines(input_jsonl) - kept
        return kept, dropped


def _count_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)
