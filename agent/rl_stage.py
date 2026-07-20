"""RL stage: GRPO via PipelineRL (QED-Nano).

Launches ``python -m pipelinerl.launch --config-name=<name>`` in the
``qed-rl`` conda env with the roles distributed across 7 GPUs:
  GPU 0   — Actor (vLLM inference, serves rollouts)
  GPU 1   — Preprocessor (reward scoring)
  GPUs 2-3-4 — Finetune (DeepSpeed ZeRO-3)
  GPU 5-6 — (extra actors or as finetune overflow — controlled by Hydra config)

GPU assignment is controlled through the ``PIPELINERL_GPUS`` env var
(comma-separated) which is read by the patched ``world.py`` in QED-Nano.

Hydra overrides are passed via command-line ``++key=value`` syntax.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
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
class RLResult:
    output_dir: str
    cosmos_ckpt_dir: str
    best_checkpoint_dir: Optional[str]
    final_reward: Optional[float]
    global_step: Optional[int]
    config_name: str


class RLStage:
    """Run one RL training stage using PipelineRL / QED-Nano.

    Parameters
    ----------
    cfg : dict
        The ``rl`` sub-dict from project.yaml.
    train_jsonl : str
        Path to the RL training data (output of data_clean Pass B, or raw SFT data).
    """

    def __init__(
        self,
        cfg: dict,
        *,
        cosmos_root: str,
        run_name: str,
        train_jsonl: str,
        local_scratch_root: str,
        cosmos_persist_root: str,
    ):
        self.cfg = cfg
        self.cosmos_root = cosmos_root
        self.run_name = run_name
        self.train_jsonl = train_jsonl
        self.local_output_dir = os.path.join(local_scratch_root, "rl")
        self.cosmos_ckpt_dir = cosmos_path(cosmos_persist_root, "rl")
        easytrain_root = cfg.get("easytrain_root") or _DEFAULT_EASYTRAIN
        rl_dir = cfg.get("easytrain_rl_dir") or str(Path(easytrain_root) / "rl" / "qed-nano")
        self._easytrain_rl = Path(rl_dir)
        self._repo_root = Path(__file__).parent.parent

    def run(self) -> RLResult:
        logger.info("=== RL stage (GRPO/PipelineRL): %s ===", self.run_name)
        os.makedirs(self.local_output_dir, exist_ok=True)

        config_name = self.cfg.get("config_name", "layer1_rl")
        num_gpus = self.cfg.get("num_gpus", 7)
        gpu_offset = self.cfg.get("gpu_offset", 0)
        gpu_ids = ",".join(str(i + gpu_offset) for i in range(num_gpus))

        init_model = self.cfg.get("init_model", "")
        if not init_model:
            raise ValueError("rl.init_model must be set (SFT checkpoint path)")

        # Build Hydra overrides from cfg
        overrides = self._build_hydra_overrides(init_model)

        # Env setup
        env_overlay = setup_training_env(
            secrets_file=self.cfg.get("secrets_file", "~/.secrets/maiprofile_sft.env"),
            strip_aml_mpi=False,   # PipelineRL manages its own process group
            cuda_visible_devices=None,  # pipelinerl assigns GPUs internally
        )
        env_overlay.update({
            "PIPELINERL_GPUS": gpu_ids,
            "WANDB_PROJECT": self.cfg.get("wandb_project", "maiprofile-rl"),
            "WANDB_NAME": self.run_name,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        env = build_subprocess_env(env_overlay)

        conda_env = self.cfg.get("conda_env", "ptca")
        python = conda_bin(conda_env, "python")

        cmd = [
            python, "-m", "pipelinerl.launch",
            f"--config-name={config_name}",
        ] + overrides

        run_cmd(cmd, env=env, cwd=str(self._easytrain_rl), log_prefix="RL")
        rsync_to_cosmos(self.local_output_dir, self.cosmos_ckpt_dir)

        result = self._read_summary(config_name)
        logger.info(
            "RL complete: best_ckpt=%s reward=%s step=%s",
            result.best_checkpoint_dir, result.final_reward, result.global_step,
        )
        return result

    # ------------------------------------------------------------------

    def _build_hydra_overrides(self, init_model: str) -> list[str]:
        """Build Hydra CLI overrides from project.yaml rl section."""
        cfg = self.cfg
        ov: list[str] = []

        def add(key: str, val):
            if val is not None:
                ov.append(f"++{key}={val}")

        # Core training params
        add("train_data_path", self.train_jsonl)
        add("output_dir", self.local_output_dir)
        add("actor.model_name_or_path", init_model)
        add("finetuner.model_name_or_path", init_model)

        # Hydra params (map from project.yaml keys to Hydra paths)
        param_map = {
            "kl_coef":            "finetuner.kl_coef",
            "entropy_bonus":      "finetuner.entropy_bonus",
            "lr":                 "finetuner.learning_rate",
            "max_train_steps":    "finetuner.max_train_steps",
            "attempts":           "rollout.n_samples",
            "max_seq_len":        "actor.max_model_len",
            "actor_fraction":     "roles.actor_gpu_count",
            "preprocessor_fraction": "roles.preprocessor_gpu_count",
            "finetune_fraction":  "roles.finetune_gpu_count",
        }
        for proj_key, hydra_key in param_map.items():
            if proj_key in cfg:
                add(hydra_key, cfg[proj_key])

        return ov

    def _read_summary(self, config_name: str) -> RLResult:
        """Try to read training summary from the output directory."""
        # PipelineRL writes trainer_state.json inside the checkpoint dir
        # Look for the most recent checkpoint
        ckpt_dir = None
        final_reward = None
        global_step = None

        from .cosmos import list_checkpoints, latest_checkpoint
        ckpt_dir = latest_checkpoint(self.local_output_dir)

        summary_path = os.path.join(self.local_output_dir, "rl_summary.json")
        if os.path.isfile(summary_path):
            try:
                with open(summary_path) as f:
                    s = json.load(f)
                final_reward = s.get("final_reward")
                global_step = s.get("global_step")
            except Exception as exc:
                logger.warning("Could not read rl_summary.json: %s", exc)

        return RLResult(
            output_dir=self.local_output_dir,
            cosmos_ckpt_dir=self.cosmos_ckpt_dir,
            best_checkpoint_dir=ckpt_dir,
            final_reward=final_reward,
            global_step=global_step,
            config_name=config_name,
        )
