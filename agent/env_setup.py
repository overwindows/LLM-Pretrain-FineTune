"""Environment setup: secrets, conda env paths, NCCL stability vars.

All secrets come from environment variables.  This module:
- Sources a secrets file (e.g. ~/.secrets/maiprofile_sft.env) if present.
- Validates that required secrets are set.
- Exports NCCL / CUDA / PyTorch stability vars needed for multi-GPU training.
- Strips AzureML multi-node MPI env vars so accelerate runs single-node.

Usage
-----
from agent.env_setup import setup_training_env

env = setup_training_env(
    secrets_file="~/.secrets/maiprofile_sft.env",
    required_secrets=["AZURE_OPENAI_KEY", "WANDB_API_KEY"],
)
# env is a dict suitable for subprocess.run(..., env={**os.environ, **env})
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Environment variables set by AzureML for multi-node MPI jobs.
# Stripping these makes accelerate / deepspeed treat the job as single-node,
# which is what we want when running all RL roles on one node.
_AML_MPI_VARS = [
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
    "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_NAME",
    "AZUREML_EXPERIMENT_ID", "AZUREML_RUN_ID",
]

# NCCL / CUDA / PyTorch stability settings proven in EasyPosttrain.
# Source: docs/process/sft_process.md + AML_OPERATIONS.md
_STABILITY_ENV = {
    "NCCL_TIMEOUT": "14400",          # 4 h (GRPO trains slowly; default 1800s trips watchdog)
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
}


def _source_env_file(path: str) -> dict[str, str]:
    """Source a bash env file and return the exported variables as a dict.

    Uses a subprocess shell trick: `source <file>; env` so that bash
    variable expansions inside the file are resolved correctly.
    """
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        logger.debug("secrets file not found (skipping): %s", expanded)
        return {}
    try:
        result = subprocess.run(
            ["bash", "-c", f"set -a; source {expanded}; set +a; env"],
            capture_output=True, text=True, check=True,
        )
        env: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                env[k] = v
        logger.info("Sourced secrets from %s (%d vars)", expanded, len(env))
        return env
    except subprocess.CalledProcessError as exc:
        logger.warning("Failed to source %s: %s", expanded, exc)
        return {}


def setup_training_env(
    secrets_file: Optional[str] = None,
    required_secrets: Optional[list[str]] = None,
    strip_aml_mpi: bool = True,
    cuda_visible_devices: Optional[str] = None,
) -> dict[str, str]:
    """Build the environment dict for training subprocesses.

    Parameters
    ----------
    secrets_file:
        Path to a bash env file to source (e.g. ``~/.secrets/maiprofile_sft.env``).
        Skipped silently if the file does not exist.
    required_secrets:
        List of env var names that must be non-empty after setup.  Raises
        ``EnvironmentError`` if any are missing.
    strip_aml_mpi:
        If True, unset AzureML MPI variables so single-node accelerate runs
        correctly on a multi-node AML allocation.
    cuda_visible_devices:
        If set, override CUDA_VISIBLE_DEVICES in the returned env.

    Returns
    -------
    dict[str, str]
        Env vars to overlay on ``os.environ`` for subprocess calls.
        Caller does: ``subprocess.run(..., env={**os.environ, **env})``.
    """
    env: dict[str, str] = {}

    # 1. Source secrets file
    if secrets_file:
        sourced = _source_env_file(secrets_file)
        env.update({k: v for k, v in sourced.items() if k not in os.environ})

    # 2. Stability vars
    env.update(_STABILITY_ENV)

    # 3. Strip AML MPI vars (unset = remove from subprocess env)
    if strip_aml_mpi:
        for var in _AML_MPI_VARS:
            env[var] = ""  # empty string signals removal in build_subprocess_env()

    # 4. CUDA_VISIBLE_DEVICES override
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    # 5. Validate required secrets
    for key in (required_secrets or []):
        val = env.get(key) or os.environ.get(key)
        if not val:
            raise EnvironmentError(
                f"Required secret {key!r} is not set.  "
                "Export it or add it to your secrets file."
            )

    return env


def build_subprocess_env(overlay: dict[str, str]) -> dict[str, str]:
    """Merge overlay into os.environ, removing keys whose value is empty string.

    This is the correct way to both set and unset vars in a subprocess call:
    ``env={**os.environ, **overlay}`` but with empties removed.
    """
    merged = {**os.environ}
    for k, v in overlay.items():
        if v == "":
            merged.pop(k, None)
        else:
            merged[k] = v
    return merged


def _conda_env_root(conda_env: str) -> str:
    """Discover the root directory of a named conda environment.

    Probes the two common locations on AML nodes in order:
      1. /opt/conda/envs/<name>  — default Singularity / managed images
      2. /home/aiscuser/.conda/envs/<name>  — user-installed envs

    Falls back to /opt/conda/envs/<name> (will fail loudly at subprocess time
    if the env truly doesn't exist, giving a clear error message).
    """
    import os
    candidates = [
        f"/opt/conda/envs/{conda_env}",
        f"/home/aiscuser/.conda/envs/{conda_env}",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    # Fall back to the most common location; missing env will surface clearly
    logger.warning(
        "conda env %r not found in %s — using first candidate", conda_env, candidates
    )
    return candidates[0]


def conda_python(conda_env: str) -> str:
    """Return the absolute path to Python in the given conda environment."""
    return f"{_conda_env_root(conda_env)}/bin/python"


def conda_bin(conda_env: str, binary: str) -> str:
    """Return path to a binary in the given conda environment's bin/."""
    return f"{_conda_env_root(conda_env)}/bin/{binary}"
