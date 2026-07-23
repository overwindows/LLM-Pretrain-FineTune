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
import subprocess
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

        # Do NOT pass cuda_visible_devices here — let AML infrastructure control
        # CUDA_VISIBLE_DEVICES.  Passing an explicit list caused
        # "CUDA error: invalid device ordinal" when the node exposes fewer GPUs
        # than the hardcoded list (e.g. 1 GPU node getting devices 0..7).
        env_overlay = setup_training_env(
            secrets_file=self.cfg.get("secrets_file", "~/.secrets/maiprofile_sft.env"),
            strip_aml_mpi=True,
            cuda_visible_devices=None,
        )
        env_overlay["WANDB_PROJECT"] = self.cfg.get("wandb_project", "maiprofile-sft")
        env_overlay["WANDB_NAME"] = self.run_name
        env = build_subprocess_env(env_overlay)

        conda_env = self.cfg.get("conda_env", "ptca")
        accelerate = conda_bin(conda_env, "accelerate")
        accel_config = str(self._repo_root / self.cfg["accelerate_config"])
        sft_script = str(self._easytrain_sft / "sft_train.py")

        # Detect actual GPU count at runtime so --num_processes matches reality.
        num_processes = self._actual_gpu_count()
        logger.info("[SFT] Detected %d GPU(s); launching %d accelerate process(es).",
                    num_processes, num_processes)

        cmd = [
            accelerate, "launch",
            "--config_file", accel_config,
            "--num_processes", str(num_processes),
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

    def _actual_gpu_count(self) -> int:
        """Detect the number of CUDA-visible GPUs available on this node.

        Priority:
        1. CUDA_VISIBLE_DEVICES env var (already set by AML infrastructure).
        2. nvidia-smi --list-gpus count.
        3. Fall back to cfg["num_gpus"] (default 1 for safety).
        """
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd and cvd not in ("NoDevFiles", "-1", ""):
            count = len(cvd.split(","))
            logger.info("[SFT] CUDA_VISIBLE_DEVICES=%r -> %d GPU(s)", cvd, count)
            return count
        # nvidia-smi fallback
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--list-gpus"], text=True, timeout=10,
            )
            count = len([ln for ln in out.splitlines() if ln.strip()])
            if count > 0:
                logger.info("[SFT] nvidia-smi detected %d GPU(s)", count)
                return count
        except Exception as exc:
            logger.warning("[SFT] nvidia-smi failed (%s); using cfg num_gpus fallback", exc)
        fallback = self.cfg.get("num_gpus", 1)
        logger.info("[SFT] GPU count fallback from cfg: %d", fallback)
        return fallback

    def _cuda_visible_devices(self) -> Optional[str]:
        # Intentionally returns None — do NOT override AML's CUDA_VISIBLE_DEVICES.
        # Kept for API compatibility; no longer called from run().
        return None

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
