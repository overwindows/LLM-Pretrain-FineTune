"""serve_sft.py — Thin wrapper around `vllm serve` for evaluation.

Starts a vLLM OpenAI-compatible server hosting either:
  - the SFT'd model        (configs/eval/layer0_signal.yaml :: subject_models.sft)
  - the zero-shot baseline (configs/eval/layer0_signal.yaml :: subject_models.zero_shot)

Usage (single-GPU is enough for a 4B bf16 model):
    CUDA_VISIBLE_DEVICES=0 python scripts/serve_sft.py \
        --config configs/eval/layer0_signal.yaml --subject sft

The script execs vllm and never returns; kill the server with Ctrl-C or
`pkill -f 'vllm.entrypoints.openai.api_server'`.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subject", required=True, choices=["sft", "zero_shot", "rl", "rl_repro"])
    ap.add_argument("--port", type=int, default=None,
                    help="override vllm.port from the YAML")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    subj = cfg["subject_models"][args.subject]
    vcfg = cfg["vllm"]
    port = args.port if args.port is not None else vcfg["port"]

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", subj["model_path"],
        "--served-model-name", subj["served_name"],
        "--port", str(port),
        "--host", "127.0.0.1",
        "--max-model-len", str(vcfg["max_model_len"]),
        "--gpu-memory-utilization", str(vcfg["gpu_memory_utilization"]),
        "--dtype", vcfg["dtype"],
        "--trust-remote-code",
        # Disable the chat-template prepended generation prompt logging spam
        "--disable-log-requests",
    ]
    # Optional: disable prefix caching (observed to occasionally corrupt
    # long generations under high concurrency on Qwen3 SFT checkpoints in
    # vllm 0.8.5). Yaml: vllm.no_prefix_cache: true
    if vcfg.get("no_prefix_cache"):
        cmd.append("--no-enable-prefix-caching")
    # AzureML injects RANK/WORLD_SIZE for its 4-node MPI topology. vLLM tries
    # to honour those and gets confused on a single-process serve. Strip them.
    for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
              "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK"):
        os.environ.pop(v, None)

    # Compatibility shim: vLLM 0.8.5 reads `tokenizer.all_special_tokens_extended`
    # (see vllm/transformers_utils/tokenizer.py get_cached_tokenizer), but
    # transformers 5.5.4 dropped that attribute on slow tokenizers like
    # Qwen2Tokenizer. Fall back to `all_special_tokens` (close enough for
    # vLLM's caching purposes) by wrapping the vllm invocation in a tiny
    # prelude that patches PreTrainedTokenizerBase before the api_server starts.
    vllm_argv = ["vllm-serve"] + cmd[3:]  # drop [python, -m, module_name]
    prelude = (
        "import transformers.tokenization_utils_base as _t; "
        "_B = _t.PreTrainedTokenizerBase; "
        "_B.all_special_tokens_extended = getattr(_B, 'all_special_tokens_extended', None) "
        "or property(lambda self: self.all_special_tokens); "
        f"import sys; sys.argv = {vllm_argv!r}; "
        "import runpy; runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')"
    )
    wrapped = [sys.executable, "-c", prelude]
    print("[serve_sft] exec (wrapped, vllm argv):", " ".join(vllm_argv), flush=True)
    os.execvp(wrapped[0], wrapped)


if __name__ == "__main__":
    sys.exit(main())
