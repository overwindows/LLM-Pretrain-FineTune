# SFT stage runner.
# Renders the SFT config template, launches accelerate + deepspeed multi-GPU
# training, and rsyncs checkpoints to Cosmos on completion.
#
# Design: mirrors LLM-Pretrain-FineTune/sft/launch_repro_sft.sh but in Python
# so the agentic pipeline can call it programmatically and inspect results.
from __future__ import annotations

import json
import logging
import os
import string
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
class SFTResult:
    output_dir: str
    cosmos_ckpt_dir: str
    best_checkpoint_dir: Optional[str]
    eval_loss: Optional[float]
    global_step: Optional[int]


class SFTStage:
    # Run one SFT training stage.
    # cfg: the ``sft`` sub-dict from project.yaml (path-resolved).
    # cosmos_root: discovered Cosmos mount root.
    # run_name: unique experiment name for output dirs and W&B.

    def __init__(
        self,
        cfg: dict,
        *,
        cosmos_root: str,
        run_name: str,
        base_model: str,
        train_jsonl: str,
        val_jsonl: str,
        local_scratch_root: str,
        cosmos_persist_root: str,
    ):
        self.cfg = cfg
        self.cosmos_root = cosmos_root
        self.run_name = run_name
        self.base_model = base_model
        self.train_jsonl = train_jsonl
        self.val_jsonl = val_jsonl
        self.local_output_dir = os.path.join(local_scratch_root, "sft")
        self.cosmos_ckpt_dir = cosmos_path(cosmos_persist_root, "sft")
        self._repo_root = Path(__file__).parent.parent
        self._easytrain_sft = Path(EASYTRAIN_ROOT) / "sft"

    def run(self) -> SFTResult:
        # Execute: render config -> train -> rsync.
        logger.info("=== SFT stage: %s ===", self.run_name)
        os.makedirs(self.local_output_dir, exist_ok=True)

        rendered_config = self._render_config()

        env_overlay = setup_training_env(
            secrets_file=self.cfg.get("secrets_file", "~/.secrets/maiprofile_sft.env"),
            strip_aml_mpi=True,
            cuda_visible_devices=self._cuda_visible_devices(),
        )
        env_overlay["WANDB_PROJECT"] = self.cfg.get("wandb_project", "maiprofile-sft")
        env_overlay["WANDB_NAME"] = self.run_name
        env = build_subprocess_env(env_overlay)

        conda_env = self.cfg.get("conda_env", "ptca")
        accelerate = conda_bin(conda_env, "accelerate")
        accel_config = str(self._repo_root / self.cfg["accelerate_config"])
        sft_script = str(self._easytrain_sft / "sft_train.py")

        cmd = [
            accelerate, "launch",
            "--config_file", accel_config,
            "--num_processes", str(self.cfg.get("num_gpus", 8)),
            sft_script,
            "--config", rendered_config,
        ]
        run_cmd(cmd, env=env, cwd=str(self._easytrain_sft), log_prefix="SFT")
        rsync_to_cosmos(self.local_output_dir, self.cosmos_ckpt_dir)

        result = self._read_summary()
        logger.info(
            "SFT complete: best_ckpt=%s eval_loss=%s step=%s",
            result.best_checkpoint_dir, result.eval_loss, result.global_step,
        )
        return result

    def _render_config(self) -> str:
        # Render ${VAR} template -> concrete YAML file.
        template_path = self._repo_root / self.cfg.get(
            "config_template", "configs/sft/sft_template.yaml.tmpl"
        )
        if not template_path.exists():
            template_path = (
                self._easytrain_sft
                / "configs/sft/layer1_delta_thinking_50k_4o-v2.repro.yaml.tmpl"
            )
        template = template_path.read_text()

        deepspeed_config = str(self._repo_root / self.cfg["deepspeed_config"])
        vars_ = {
            "STEP_KEY": self.run_name,
            "BASE_MODEL": self.base_model,
            "TRAIN_JSONL": self.train_jsonl,
            "VAL_JSONL": self.val_jsonl,
            "OUTPUT_DIR": self.local_output_dir,
            "DEEPSPEED_CONFIG": deepspeed_config,
        }
        rendered = string.Template(template).safe_substitute(vars_)

        # Write rendered config to /tmp (writable) — _repo_root may be an RO_MOUNT on AML
        out_dir = Path("/tmp/agentic_sft_rendered")
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered_path = str(out_dir / f"{self.run_name}_sft.yaml")
        with open(rendered_path, "w") as fout:
            fout.write(rendered)
        logger.info("Rendered SFT config -> %s", rendered_path)
        return rendered_path

    def _cuda_visible_devices(self) -> str:
        n = self.cfg.get("num_gpus", 8)
        return ",".join(str(i) for i in range(n))

    def _read_summary(self) -> SFTResult:
        summary_path = os.path.join(self.local_output_dir, "training_summary.json")
        best_ckpt = eval_loss = global_step = None
        if os.path.isfile(summary_path):
            try:
                with open(summary_path) as f:
                    s = json.load(f)
                best_ckpt = s.get("best_model_checkpoint")
                eval_loss = s.get("best_metric")
                global_step = s.get("global_step")
            except Exception as exc:
                logger.warning("Could not read training_summary.json: %s", exc)
        return SFTResult(
            output_dir=self.local_output_dir,
            cosmos_ckpt_dir=self.cosmos_ckpt_dir,
            best_checkpoint_dir=best_ckpt,
            eval_loss=eval_loss,
            global_step=global_step,
        )
