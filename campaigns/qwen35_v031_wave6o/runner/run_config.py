#!/usr/bin/env python3
"""Run one Gemma4 offline benchmark from an AML-generated JSON config."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "datasets" / "sc1_delta_v2.jsonl"
RESULTS_DIR = SCRIPT_DIR / "run_results"


def load_prompts(limit: int) -> list[str]:
    prompts = []
    with DATASET_PATH.open(encoding="utf-8") as dataset:
        for line in dataset:
            record = json.loads(line)
            prompts.append(record["prompt"])
            if len(prompts) == limit:
                break
    if not prompts:
        raise RuntimeError(f"No prompts found in {DATASET_PATH}")
    return prompts


def build_llm_kwargs(config: dict) -> dict:
    kwargs = {
        "model": os.environ["GEMMA4_TEXT_ONLY_MODEL_PATH"],
        "trust_remote_code": True,
        "quantization": config["quantization"],
        "max_model_len": config["max_model_len"],
        "gpu_memory_utilization": config["gpu_memory_utilization"],
        "enforce_eager": config["enforce_eager"],
        "enable_prefix_caching": config["enable_prefix_caching"],
        "enable_chunked_prefill": config["enable_chunked_prefill"],
    }
    for key in (
        "max_num_seqs",
        "max_num_batched_tokens",
        "optimization_level",
        "performance_mode",
        "async_scheduling",
    ):
        if key in config:
            kwargs[key] = config[key]
    if config.get("kv_cache_dtype", "auto") != "auto":
        kwargs["kv_cache_dtype"] = config["kv_cache_dtype"]
    if config.get("moe_backend", "auto") != "auto":
        kwargs["moe_backend"] = config["moe_backend"]
    if "speculative_config" in config:
        kwargs["speculative_config"] = config["speculative_config"]
    return kwargs


def truncate_prompts(
    tokenizer, prompts: list[str], max_model_len: int, max_tokens: int
) -> list[dict[str, list[int]]]:
    max_prompt_tokens = max_model_len - max_tokens
    if max_prompt_tokens <= 0:
        raise ValueError("max_model_len must exceed max_tokens")
    prompt_token_ids = []
    for prompt in prompts:
        token_ids = [int(token_id) for token_id in tokenizer.encode(prompt)]
        prompt_token_ids.append({"prompt_token_ids": token_ids[-max_prompt_tokens:]})
    return prompt_token_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", type=int, default=20)
    parser.add_argument("--reps", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = os.environ["GEMMA4_TEXT_ONLY_MODEL_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    rendered_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in load_prompts(args.prompts)
    ]
    max_tokens = config.get("max_tokens", 1024)
    prompts = truncate_prompts(
        tokenizer, rendered_prompts, config["max_model_len"], max_tokens
    )

    print(f"Gemma4 benchmark model: {model_path}", flush=True)
    print(f"Gemma4 benchmark prompts: {len(prompts)}", flush=True)
    llm = LLM(**build_llm_kwargs(config))
    samples = []
    totals = []
    for repetition in range(1, args.reps + 1):
        sampling = SamplingParams(
            temperature=0.7, top_p=0.95, max_tokens=max_tokens, seed=repetition
        )
        start = time.time()
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        elapsed = time.time() - start
        output_tokens = sum(
            len(completion.token_ids)
            for output in outputs
            for completion in output.outputs
        )
        prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        samples.append(output_tokens / elapsed)
        totals.append((prompt_tokens + output_tokens) / elapsed)

    result = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "mean_output_tps": round(statistics.mean(samples), 2),
        "mean_total_tps": round(statistics.mean(totals), 2),
        "stdev_output_tps": round(statistics.stdev(samples), 2) if len(samples) > 1 else 0.0,
        "status": "ok",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    result_path = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"output_tps: {result['mean_output_tps']:.2f} +/- {result['stdev_output_tps']:.2f}")
    print(f"total_tps: {result['mean_total_tps']:.2f}")
    print(f"Results saved: {result_path}")
    del llm
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
