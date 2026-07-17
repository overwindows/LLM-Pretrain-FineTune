"""Minimal async Azure OpenAI client for offline judge / RM-distillation work.

This is a self-contained subset of ``maiprofilev3dev/modules/llm_client.py``
so the scripts under :mod:`rl_layer1.rm_distill` can run inside the QED-Nano
clone without depending on the maiprofile repo. The patterns
(`DefaultAzureCredential` fallback, ``api_key`` env var, ``response_format``
JSON mode, HTTP/2) match the production client.

Usage::

    client = build_judge_client("gpt54-eval")
    text, usage = await invoke_judge(
        client, model="gpt-5.4-evaluation",
        system_prompt="...",
        user_prompt="...",
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncAzureOpenAI


logger = logging.getLogger("rl_layer1.rm_distill.llm_judge_client")


# ---------------------------------------------------------------------------
# Model registry (kept in sync with MaiProfile config.py)
# ---------------------------------------------------------------------------
# Mirrors the production deployments; override via env if needed.

API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

JUDGE_MODELS: dict[str, dict[str, Any]] = {
    "gpt54-eval": {
        "model": "gpt-5.4-evaluation",
        "endpoint": "https://agentic-retrieval-ae-resource.cognitiveservices.azure.com/",
        "api_key_env": "GPT54_EVAL_API_KEY",
        "max_output_tokens": 100000,
        "timeout": 300.0,
        "is_reasoning": True,
    },
    "gpt52": {
        "model": "gpt-5.2",
        "endpoint": "https://csnf-singularity-aoai-eastus2.openai.azure.com/",
        "api_key_env": None,        # token provider only
        "max_output_tokens": 100000,
        "timeout": 300.0,
        "is_reasoning": True,
    },
    "gpt4o": {
        "model": "gpt-4o",
        "endpoint": "https://csnf-singularity-aoai-northcentralus.openai.azure.com/",
        "api_key_env": None,
        "max_output_tokens": 16384,
        "timeout": 120.0,
        "is_reasoning": False,
    },
}


def _shared_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        http2=True,
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
    )


def _resolve_api_key(cfg: dict[str, Any]) -> str | None:
    env_name = cfg.get("api_key_env")
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]
    return None


def _bearer_token_provider() -> Any:
    """Return an Azure AD bearer-token provider."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    return get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )


def build_judge_client(
    model_key: str,
    http_client: httpx.AsyncClient | None = None,
) -> AsyncAzureOpenAI:
    """Construct an :class:`AsyncAzureOpenAI` client for a given judge model.

    Auth precedence: env API key → DefaultAzureCredential bearer token.
    """
    if model_key not in JUDGE_MODELS:
        raise KeyError(
            f"Unknown judge model {model_key!r}; "
            f"available: {sorted(JUDGE_MODELS.keys())}"
        )
    cfg = JUDGE_MODELS[model_key]
    http = http_client or _shared_http_client()
    api_key = _resolve_api_key(cfg)
    if api_key:
        logger.info("judge=%s: using api key from %s", model_key, cfg["api_key_env"])
        return AsyncAzureOpenAI(
            api_version=API_VERSION,
            azure_endpoint=cfg["endpoint"],
            api_key=api_key,
            max_retries=2,
            timeout=cfg["timeout"],
            http_client=http,
        )
    logger.info("judge=%s: using DefaultAzureCredential token provider", model_key)
    return AsyncAzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=cfg["endpoint"],
        azure_ad_token_provider=_bearer_token_provider(),
        max_retries=2,
        timeout=cfg["timeout"],
        http_client=http,
    )


# ---------------------------------------------------------------------------
# Single-call helper
# ---------------------------------------------------------------------------


@dataclass
class JudgeResponse:
    text: str
    usage: dict[str, Any]
    finish_reason: str
    elapsed_s: float


async def invoke_judge(
    client: AsyncAzureOpenAI,
    *,
    model_key: str,
    system_prompt: str,
    user_prompt: str,
    response_format: str | None = "json_object",
    max_completion_tokens: int | None = None,
    reasoning_effort: str | None = "medium",
) -> JudgeResponse:
    """Call the judge once and return the parsed response object.

    Notes
    -----
    - For reasoning models (``gpt-5.x``) ``temperature`` / ``top_p`` are
      forbidden by the API; we omit them and pass ``reasoning_effort``.
    - For non-reasoning models (``gpt-4o``) we set ``temperature=0.2`` so the
      judge is mostly-deterministic.
    """
    import time

    cfg = JUDGE_MODELS[model_key]
    max_tokens = min(
        max_completion_tokens or cfg["max_output_tokens"],
        cfg["max_output_tokens"],
    )
    kwargs: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": max_tokens,
    }
    if cfg["is_reasoning"]:
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = 0.2
    if response_format:
        kwargs["response_format"] = {"type": response_format}

    started = time.perf_counter()
    resp = await client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - started

    choice = resp.choices[0]
    text = choice.message.content or ""
    finish_reason = getattr(choice, "finish_reason", "unknown")
    usage_obj = getattr(resp, "usage", None)
    usage: dict[str, Any]
    if usage_obj is None:
        usage = {}
    elif hasattr(usage_obj, "model_dump"):
        usage = usage_obj.model_dump()
    else:
        usage = dict(usage_obj)
    return JudgeResponse(text=text, usage=usage, finish_reason=finish_reason, elapsed_s=elapsed)
