"""Top-level Pipeline orchestrator.

Reads project.yaml, discovers the Cosmos mount, then runs stages in order.
Each stage is independently re-runnable: if a stage already produced its
checkpoint in Cosmos, it is skipped (or re-run with --force).

Usage (on AML node)
-------------------
from agent.pipeline import Pipeline

pipe = Pipeline.from_yaml("project.yaml")
pipe.run()          # all stages
pipe.run(["sft"])   # one stage
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from string import Template
from typing import Optional

import yaml

from .cosmos import discover_cosmos_mount, cosmos_path
from .utils import setup_logging

logger = logging.getLogger(__name__)


def _resolve_vars(obj, ctx: dict):
    """Recursively resolve ${VAR} placeholders in a nested dict/list/str."""
    if isinstance(obj, str):
        try:
            return Template(obj).safe_substitute(ctx)
        except Exception:
            return obj
    if isinstance(obj, dict):
        return {k: _resolve_vars(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_vars(v, ctx) for v in obj]
    return obj


class Pipeline:
    """Runs the SFT → data_clean → RL → eval pipeline.

    Parameters
    ----------
    cfg : dict
        Resolved config dict (from project.yaml with vars substituted).
    cosmos_root : str
        Absolute Cosmos mount path.
    """

    def __init__(self, cfg: dict, cosmos_root: str):
        self.cfg = cfg
        self.cosmos_root = cosmos_root
        self.run_name: str = cfg["run_name"]

        # Resolve ${cosmos_root} and ${run_name} placeholders
        ctx = {"cosmos_root": cosmos_root, "run_name": self.run_name}
        self.cfg = _resolve_vars(cfg, ctx)

        self.base_model: str = self.cfg["base_model"]
        self.data: dict = self.cfg.get("data", {})
        self.local_scratch_root: str = self.cfg.get(
            "local_scratch_root", f"/scratch/agentic_runs/{self.run_name}"
        )
        self.cosmos_persist_root: str = self.cfg.get(
            "cosmos_persist_root",
            cosmos_path(cosmos_root, "agentic_runs", self.run_name),
        )
        self._all_stages: list[str] = self.cfg.get(
            "stages", ["sft", "data_clean", "rl", "eval"]
        )

    @classmethod
    def from_yaml(
        cls,
        project_yaml: str,
        cosmos_root: Optional[str] = None,
    ) -> "Pipeline":
        """Load config from YAML and auto-discover Cosmos mount."""
        with open(project_yaml) as f:
            cfg = yaml.safe_load(f)

        root = cosmos_root or cfg.get("cosmos_root") or discover_cosmos_mount()
        return cls(cfg, root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        stages: Optional[list[str]] = None,
        force: bool = False,
    ) -> dict:
        """Run requested stages in order.

        Parameters
        ----------
        stages : list[str] | None
            Stages to run, e.g. ``["sft", "rl"]``.  None = all stages
            defined in ``project.yaml``.
        force : bool
            Re-run stages even if their Cosmos output already exists.

        Returns
        -------
        dict
            Mapping stage → result object.
        """
        setup_logging()
        stages = stages or self._all_stages
        results: dict = {}

        for stage in stages:
            if stage not in self._all_stages:
                logger.warning("Unknown stage %r — skipping", stage)
                continue
            logger.info("===== Stage: %s =====", stage)
            results[stage] = self._run_stage(stage, force=force)
            logger.info("===== Stage %s complete =====", stage)

        return results

    # ------------------------------------------------------------------
    # Private dispatch
    # ------------------------------------------------------------------

    def _run_stage(self, stage: str, force: bool = False):
        if stage == "sft":
            return self._run_sft(force)
        if stage == "data_clean":
            return self._run_data_clean(force)
        if stage == "rl":
            return self._run_rl(force)
        if stage == "eval":
            return self._run_eval(force)
        raise ValueError(f"Unknown stage: {stage!r}")

    def _run_sft(self, force: bool = False):
        from .sft_stage import SFTStage

        cfg = self.cfg.get("sft", {})
        train_jsonl = self.data.get("sft_train_jsonl", "")
        val_jsonl = self.data.get("sft_val_jsonl", "")
        if not train_jsonl or not val_jsonl:
            raise ValueError("data.sft_train_jsonl and data.sft_val_jsonl must be set")

        stage = SFTStage(
            cfg,
            cosmos_root=self.cosmos_root,
            run_name=self.run_name,
            base_model=self.base_model,
            train_jsonl=train_jsonl,
            val_jsonl=val_jsonl,
            local_scratch_root=self.local_scratch_root,
            cosmos_persist_root=self.cosmos_persist_root,
        )
        result = stage.run()
        # Pass best checkpoint forward: RL init model = SFT best checkpoint
        if result.best_checkpoint_dir:
            self.cfg.setdefault("rl", {})
            if not self.cfg["rl"].get("init_model"):
                self.cfg["rl"]["init_model"] = result.best_checkpoint_dir
                logger.info("RL init_model auto-set to SFT best: %s", result.best_checkpoint_dir)
        return result

    def _run_data_clean(self, force: bool = False):
        from .data_clean_stage import DataCleanStage

        cfg = self.cfg.get("data_clean", {})
        sft_ckpt = self.cfg.get("rl", {}).get("init_model") or cosmos_path(
            self.cosmos_persist_root, "sft"
        )
        train_jsonl = self.data.get("sft_train_jsonl", "")
        stage = DataCleanStage(
            cfg,
            cosmos_root=self.cosmos_root,
            run_name=self.run_name,
            sft_checkpoint=sft_ckpt,
            sft_train_jsonl=train_jsonl,
            local_scratch_root=self.local_scratch_root,
            cosmos_persist_root=self.cosmos_persist_root,
        )
        result = stage.run()
        if result.rl_train_jsonl:
            self.data["rl_train_jsonl"] = result.rl_train_jsonl
            logger.info("RL data set to Pass B output: %s", result.rl_train_jsonl)
        return result

    def _run_rl(self, force: bool = False):
        from .rl_stage import RLStage

        cfg = self.cfg.get("rl", {})
        rl_train_jsonl = self.data.get("rl_train_jsonl", "")
        if not rl_train_jsonl:
            logger.warning("rl_train_jsonl not set — using sft_train_jsonl as fallback")
            rl_train_jsonl = self.data.get("sft_train_jsonl", "")
        stage = RLStage(
            cfg,
            cosmos_root=self.cosmos_root,
            run_name=self.run_name,
            train_jsonl=rl_train_jsonl,
            local_scratch_root=self.local_scratch_root,
            cosmos_persist_root=self.cosmos_persist_root,
        )
        return stage.run()

    def _run_eval(self, force: bool = False):
        from .eval_stage import EvalStage

        cfg = self.cfg.get("eval", {})
        # Best checkpoint: prefer RL output, fall back to SFT
        rl_ckpt = cosmos_path(self.cosmos_persist_root, "rl")
        sft_ckpt = self.cfg.get("rl", {}).get("init_model", "")
        stage = EvalStage(
            cfg,
            cosmos_root=self.cosmos_root,
            run_name=self.run_name,
            rl_checkpoint_dir=rl_ckpt,
            sft_checkpoint=sft_ckpt,
            local_scratch_root=self.local_scratch_root,
            cosmos_persist_root=self.cosmos_persist_root,
        )
        return stage.run()
