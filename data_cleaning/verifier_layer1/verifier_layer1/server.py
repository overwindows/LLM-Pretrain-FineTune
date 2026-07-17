"""FastAPI verifier server for Layer1 delta RL (Stage 1).

This server mimics QED-Nano's ``training/pipelinerl/entrypoints/verifier.py``
contract: ``POST /`` accepts a rollout body and returns a reward dict.

Run locally for smoke testing:

    LAYER1_STAGE=1 uvicorn server:app --host 0.0.0.0 --port 8001

Config loading order (later wins):
1. Built-in defaults from :mod:`reward.compose`.
2. JSON file at ``LAYER1_REWARD_CONFIG`` env var, if set.
3. Per-request override in the body's ``reward_config`` field (NOT recommended
   in production; provided for ad-hoc debugging only).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException

from .reward.compose import (
    RewardConfig,
    build_config,
    compute_reward,
    reward_config_to_dict,
    reward_output_to_dict,
)


logger = logging.getLogger("verifier_layer1")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Layer1 Delta Verifier", version="0.1.0")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_default_config() -> RewardConfig:
    cfg_path = os.environ.get("LAYER1_REWARD_CONFIG")
    if cfg_path and os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        logger.info("Loaded reward config from %s", cfg_path)
        return build_config(raw)
    stage = int(os.environ.get("LAYER1_STAGE", "1"))
    return build_config({"stage": stage})


_DEFAULT_CFG = _load_default_config()
logger.info("Default reward config: %s", json.dumps(reward_config_to_dict(_DEFAULT_CFG)))


# ---------------------------------------------------------------------------
# Request validation (lenient — PipelineRL bodies are dict-typed)
# ---------------------------------------------------------------------------


def _validate_body(body: dict[str, Any]) -> tuple[str, list[Any]]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    completion = body.get("completion")
    if not isinstance(completion, str):
        raise HTTPException(
            status_code=400,
            detail="Body must contain 'completion' as a string",
        )
    meta = body.get("metadata") or {}
    if not isinstance(meta, dict):
        raise HTTPException(status_code=400, detail="'metadata' must be an object")
    signals = meta.get("input_signals") or []
    if not isinstance(signals, list):
        raise HTTPException(
            status_code=400,
            detail="'metadata.input_signals' must be a list when provided",
        )
    return completion, signals


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "stage": _DEFAULT_CFG.stage}


@app.get("/config")
async def get_config() -> dict[str, Any]:
    return reward_config_to_dict(_DEFAULT_CFG)


@app.post("/")
async def verify(body: dict[str, Any]) -> dict[str, Any]:
    completion, signals = _validate_body(body)

    # Per-request override (debug only).
    override = body.get("reward_config")
    cfg = build_config(override) if isinstance(override, dict) else _DEFAULT_CFG

    out = compute_reward(completion=completion, input_signals=signals, cfg=cfg)

    # Echo record_id when present so the trainer side can correlate.
    rid = (body.get("metadata") or {}).get("record_id")
    resp = reward_output_to_dict(out)
    if rid is not None:
        resp["record_id"] = rid
    return resp
