"""Cosmos mount path discovery and artifact persistence helpers.

Background
----------
On AzureML, the Cosmos blob share is mounted at a path of the form:
    /scratch/azureml/cr/j/<JOBID>/cap/data-capability/wd/INPUT_<DATASTORE>/...

Both <JOBID> and INPUT_<DATASTORE> change per-person / per-job — **never
hardcode them**.  This module discovers the mount root at runtime by globbing
the known prefix.

Key functions
-------------
discover_cosmos_mount()       -> str path to the INPUT_* share root
cosmos_path(cfg, *parts)      -> absolute path under cosmos_root
rsync_to_cosmos(src, dst)     -> persist a local directory to cosmos
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# The glob pattern that identifies the Cosmos mount root on AML nodes.
_AML_COSMOS_GLOB = "/scratch/azureml/cr/j/*/cap/data-capability/wd/INPUT_*"


def discover_cosmos_mount() -> str:
    """Return the first Cosmos INPUT_* mount dir found on this AML node.

    Falls back to the current working directory with a warning if running
    off-AML (e.g. local dev).

    Raises
    ------
    RuntimeError
        If the pattern matches more than one mount (unexpected) and
        AGENTIC_COSMOS_ROOT is not set.
    """
    # Explicit override: AGENTIC_COSMOS_ROOT env var.
    override = os.environ.get("AGENTIC_COSMOS_ROOT")
    if override:
        logger.info("cosmos_root (env override): %s", override)
        return override

    matches = sorted(glob.glob(_AML_COSMOS_GLOB))
    if not matches:
        cwd = os.getcwd()
        logger.warning(
            "No AML Cosmos mount found at %s; "
            "falling back to cwd=%s.  Set AGENTIC_COSMOS_ROOT to silence this.",
            _AML_COSMOS_GLOB, cwd,
        )
        return cwd

    if len(matches) > 1:
        logger.warning(
            "Multiple Cosmos mounts found: %s — using the first.  "
            "Set AGENTIC_COSMOS_ROOT to pin a specific one.",
            matches,
        )
        # Prefer the INPUT_cosmos_data mount if present (avoid picking RO_MOUNT
        # inputs like INPUT_agent_pkg or INPUT_project_yaml_file).
        preferred = [m for m in matches if "INPUT_cosmos_data" in m]
        if preferred:
            root = preferred[0]
            logger.info("cosmos_root (auto-discovered, preferred cosmos_data): %s", root)
            return root

    root = matches[0]
    logger.info("cosmos_root (auto-discovered): %s", root)
    return root


def cosmos_path(cosmos_root: str, *parts: str) -> str:
    """Return an absolute path under cosmos_root.

    Example
    -------
    >>> cosmos_path("/scratch/azureml/.../INPUT_foo", "models", "sft", "ckpt")
    '/scratch/azureml/.../INPUT_foo/models/sft/ckpt'
    """
    return str(Path(cosmos_root).joinpath(*parts))


def rsync_to_cosmos(
    src_dir: str,
    dst_dir: str,
    exclude_patterns: Optional[list[str]] = None,
    dry_run: bool = False,
) -> None:
    """Rsync a local training directory to the Cosmos share.

    Excludes DeepSpeed ZeRO shard directories (``global_step*``) by default
    — they are transient and not needed for inference or resumption from a
    safetensors checkpoint.

    Parameters
    ----------
    src_dir:
        Local source directory (e.g. /scratch/local_sft_runs/my_run).
    dst_dir:
        Cosmos destination (discovered via cosmos_path()).
    exclude_patterns:
        Additional rsync --exclude patterns.  ``global_step*`` is always
        excluded unless explicitly overridden with an empty list.
    dry_run:
        If True, pass --dry-run to rsync (no writes).
    """
    if exclude_patterns is None:
        exclude_patterns = ["global_step*"]

    cmd = ["rsync", "-a", "--info=progress2"]
    for pat in exclude_patterns:
        cmd += [f"--exclude={pat}"]
    if dry_run:
        cmd.append("--dry-run")
    # trailing slash on src = contents, not the directory itself
    cmd += [src_dir.rstrip("/") + "/", dst_dir]

    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    logger.info("rsync %s -> %s (excludes=%s)", src_dir, dst_dir, exclude_patterns)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("rsync failed with exit code %d", result.returncode)
        raise RuntimeError(f"rsync {src_dir} -> {dst_dir} failed (rc={result.returncode})")
    logger.info("rsync complete -> %s", dst_dir)


def list_checkpoints(ckpt_dir: str) -> list[str]:
    """Return sorted list of checkpoint-* subdirectories in ckpt_dir."""
    p = Path(ckpt_dir)
    if not p.is_dir():
        return []
    ckpts = sorted(
        [str(c) for c in p.iterdir() if c.is_dir() and c.name.startswith("checkpoint-")],
        key=lambda x: int(Path(x).name.split("-")[-1]),
    )
    return ckpts


def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return the path to the highest-numbered checkpoint-* directory, or None."""
    ckpts = list_checkpoints(ckpt_dir)
    return ckpts[-1] if ckpts else None
