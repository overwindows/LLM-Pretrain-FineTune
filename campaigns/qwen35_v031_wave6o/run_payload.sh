#!/bin/bash
set +e
JOB_WORKDIR="$PWD"
AML_OUTPUTS_RESULTS_DIR="$JOB_WORKDIR/outputs/results"
mkdir -p "$AML_OUTPUTS_RESULTS_DIR"

echo "=== Agentic vLLM Job Start ==="
echo "Job ID: 20261118_114824"
echo "Experiments: 1"
echo "Runner branch: main"
echo "Patch branch: none (param-only)"
echo "Patches: 5"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

# 1. Clone the benchmark runner independently from the installed vLLM source,
#    OR use a pre-baked runner already present in the image (baked_runner_path).
BAKED_RUNNER=""
if [ -n "$BAKED_RUNNER" ]; then
    echo "[Step 1/7] Using baked runner at $BAKED_RUNNER (no clone / no patches)..."
    RUNNER_DIR="$BAKED_RUNNER"
    RUNNER_COMMAND="python run_config.py"
    RUNNER_PYTHONPATH=""
    cd "$RUNNER_DIR"
    echo "  Runner directory (baked): $RUNNER_DIR"
    if [ ! -f "$RUNNER_DIR/run_config.py" ]; then
        echo "ERROR: baked runner entrypoint not found: $RUNNER_DIR/run_config.py"
        exit 1
    fi
else
echo "[Step 1/7] Staging runner from job code bundle..."
mkdir -p /tmp/runner_src/benchmarks/gemma4_moe_benchmarks
cp -R "$JOB_WORKDIR/runner/." /tmp/runner_src/benchmarks/gemma4_moe_benchmarks/
if [ -n "" ]; then
    RUNNER_DIR="/tmp/runner_src/src/agentic_vllm/autobench"
    RUNNER_COMMAND="python -m "
    RUNNER_PYTHONPATH="/tmp/runner_src/src"
elif [ -n "benchmarks/gemma4_moe_benchmarks" ]; then
    RUNNER_DIR="/tmp/runner_src/benchmarks/gemma4_moe_benchmarks"
    RUNNER_COMMAND="python run_config.py"
    RUNNER_PYTHONPATH=""
else
    RUNNER_DIR=$(find /tmp/runner_src/benchmarks -type f -name run_experiment.py -path '*/autobench/*' -printf '%h
' | head -1)
    RUNNER_COMMAND="python run_experiment.py"
    RUNNER_PYTHONPATH=""
fi
if [ -z "$RUNNER_DIR" ] || [ ! -d "$RUNNER_DIR" ]; then
    echo "ERROR: benchmark runner is unavailable in https://github.com/MS-Feeds/vllm-msn.git:main"
    exit 1
fi
cd "$RUNNER_DIR"
echo "  Runner cloned at $(date)"
echo "  Runner directory: $RUNNER_DIR"
echo "  Runner commit: $(git -C /tmp/runner_src rev-parse HEAD)"
if [ "$RUNNER_COMMAND" = "python run_config.py" ]; then
    python - <<'PY'
from pathlib import Path

path = Path("run_config.py")
text = path.read_text()
text = text.replace(
    'import gc\n',
    'import gc\nimport hashlib\n',
    1,
)
old = "json.dumps(result, indent=2)"
new = "json.dumps(result, indent=2, default=str)"
if old in text:
    path.write_text(text.replace(old, new, 1))
    print("  Applied JSON serialization compatibility fix to benchmark runner")
PY
fi
if [ "$RUNNER_COMMAND" = "python run_config.py" ]; then
    python - <<'PY'
from pathlib import Path

path = Path("run_config.py") if Path("run_config.py").exists() else Path("run_experiment.py")
text = path.read_text()
text = text.replace(
    'def build_llm_kwargs(config: dict) -> dict:\n    kwargs = {',
    'def build_llm_kwargs(config: dict) -> dict:\n    from vllm.engine.arg_utils import EngineArgs\n    import inspect\n\n    supported_engine_args = inspect.signature(EngineArgs).parameters\n    kwargs = {',
    1,
)
text = text.replace(
    '"quantization": config["quantization"],\n',
    '"quantization": config["quantization"] if config.get("quantization", "auto") not in ("auto", None, "") else None,\n',
    1,
)
text = text.replace(
    '    samples = []\n    totals = []\n',
    '    samples = []\n    totals = []\n    qps_samples = []\n    elapsed_samples = []\n    out_lens = []\n    parity_digest = None\n',
    1,
)
text = text.replace(
    '        "async_scheduling",\n    ):',
    '        "async_scheduling",\n        "max_num_partial_prefills",\n        "long_prefill_token_threshold",\n        "attention_backend",\n        "mamba_ssm_cache_dtype",\n        "language_model_only",\n        "hf_overrides",\n        "dtype",\n        "per_request_spec_decode_metrics",\n    ):',
    1,
)
text = text.replace(
    '            add_generation_prompt=True,\n            tokenize=False,\n        )\n',
    '            add_generation_prompt=True,\n            tokenize=False,\n            **(config.get("chat_template_kwargs") or {"enable_thinking": config["enable_thinking"]} if "enable_thinking" in config or "chat_template_kwargs" in config else {}),\n        )\n',
    1,
)
text = text.replace(
    '        if key in config:\n            kwargs[key] = config[key]\n',
    '        if key in config:\n            if key not in supported_engine_args:\n                print(f"Skipping unsupported EngineArgs option: TRANSFORMERS_OFFLINE")\n                continue\n            kwargs[key] = config[key]\n',
    1,
)
text = text.replace(
    '    return kwargs\n',
    '    if os.environ.get("AGENTIC_VLLM_TORCH_PROFILE") == "1":\n'
    '        kwargs["profiler_config"] = {\n'
    '            "profiler": "torch",\n'
    '            "torch_profiler_dir": "/tmp/vllm_torch_profile",\n'
    '            "torch_profiler_with_stack": False,\n'
    '            "torch_profiler_record_shapes": False,\n'
    '            "torch_profiler_use_gzip": True,\n'
    '            "ignore_frontend": True,\n'
    '            "wait_iterations": 5,\n'
    '            "warmup_iterations": 3,\n'
    '            "active_iterations": 20,\n'
    '            "max_iterations": 20,\n'
    '        }\n'
    '    return kwargs\n',
    1,
)
text = text.replace(
    '    llm = LLM(**build_llm_kwargs(config))\n',
    '    llm = LLM(**build_llm_kwargs(config))\n'
    '    profile_enabled = os.environ.get("AGENTIC_VLLM_TORCH_PROFILE") == "1"\n'
    '    if profile_enabled:\n'
    '        print("Starting bounded vLLM Torch profiler capture", flush=True)\n'
    '        llm.start_profile("v027_greedy_winner")\n',
    1,
)
text = text.replace(
    '        outputs = llm.generate(prompts, sampling, use_tqdm=True)\n',
    '        outputs = llm.generate(prompts, sampling, use_tqdm=True)\n'
    '        if repetition == 1 and config.get("_agentic_latency_path"):\n'
    '            _collect_latency(outputs, config["_agentic_latency_path"])\n'
    '        parity_output_path = config.get("_agentic_parity_output_path")\n'
    '        if parity_output_path and repetition == 1:\n'
    '            records = [\n'
    '                [{\n'
    '                    "token_ids": completion.token_ids,\n'
    '                    "text": completion.text,\n'
    '                } for completion in output.outputs]\n'
    '                for output in outputs\n'
    '            ]\n'
    '            Path(parity_output_path).write_text(json.dumps(records), encoding="utf-8")\n'
    '            print(f"Wrote deterministic parity output: {parity_output_path}", flush=True)\n'
    '        if profile_enabled and repetition == 1:\n'
    '            llm.stop_profile()\n'
    '            print("Stopped bounded vLLM Torch profiler capture", flush=True)\n',
    1,
)
text = text.replace(
    '        samples.append(output_tokens / elapsed)\n        totals.append((prompt_tokens + output_tokens) / elapsed)\n',
    '        samples.append(output_tokens / elapsed)\n        totals.append((prompt_tokens + output_tokens) / elapsed)\n        qps_samples.append(len(prompts) / elapsed)\n        elapsed_samples.append(elapsed)\n        out_lens.extend(len(c.token_ids) for o in outputs for c in o.outputs)\n        if repetition == 1 and os.environ.get("AGENTIC_VLLM_OUTPUT_PARITY") == "1":\n            parity_tokens = [[int(token_id) for token_id in output.outputs[0].token_ids] for output in outputs[:32]]\n            parity_payload = json.dumps(parity_tokens, separators=(",", ":")).encode("utf-8")\n            parity_digest = hashlib.sha256(parity_payload).hexdigest()\n',
    1,
)
text = text.replace(
    '            temperature=0.7, top_p=0.95, max_tokens=max_tokens, seed=repetition\n',
    '            temperature=config.get("temperature", 0.0), top_p=config.get("top_p", 1.0), max_tokens=max_tokens, seed=repetition, ignore_eos=False\n',
    1,
)
text = text.replace(
    '        "stdev_output_tps": round(statistics.stdev(samples), 2) if len(samples) > 1 else 0.0,\n',
    '        "stdev_output_tps": round(statistics.stdev(samples), 2) if len(samples) > 1 else 0.0,\n        "mean_qps": round(statistics.mean(qps_samples), 2),\n        "mean_elapsed_time": round(statistics.mean(elapsed_samples), 2),\n        "mean_out": round(statistics.mean(out_lens), 1) if out_lens else 0,\n        "sampling_temperature": config.get("temperature", 0.0),\n        "sampling_top_p": config.get("top_p", 1.0),\n        "sampling_ignore_eos": False,\n        "parity_digest_32": parity_digest,\n',
    1,
)
text = text.replace(
    '''    print(f"total_tps: {result['mean_total_tps']:.2f}")\n''',
    '''    print(f"total_tps: {result['mean_total_tps']:.2f}")\n    print("mean_out: %.1f" % result["mean_out"])\n    print("sampling: temp=%.2f top_p=%.2f ignore_eos=%s" % (result["sampling_temperature"], result["sampling_top_p"], result["sampling_ignore_eos"]))\n    print("qps: %.2f" % result["mean_qps"])\n    print("elapsed_time: %.2fs" % result["mean_elapsed_time"])\n    if result["parity_digest_32"]:\n        print("parity_digest_32: %s" % result["parity_digest_32"])\n''',
    1,
)
path.write_text(text)
if '"mean_qps"' in text and '"mean_elapsed_time"' in text:
    print("  Applied Gemma4 runner QPS and elapsed-time result patch")
else:
    print("  WARNING: Gemma4 runner QPS and elapsed-time patch did not match runner source")
if '"max_num_partial_prefills"' in text and 'supported_engine_args' in text:
    print("  Applied Gemma4 runner scheduler-setting forwarding compatibility patch")
else:
    print("  WARNING: Gemma4 runner scheduler-setting patch did not match runner source")
if 'temperature=config.get("temperature", 0.0)' in text and 'ignore_eos=False' in text:
    print("  Applied Gemma4 runner sampling-parameter forwarding patch (greedy default)")
else:
    print("  WARNING: Gemma4 runner sampling-parameter patch did not match runner source")
if 'AGENTIC_VLLM_TORCH_PROFILE' in text and 'llm.start_profile' in text:
    print("  Applied vLLM-native bounded Torch profiler integration")
else:
    print("  WARNING: vLLM-native Torch profiler integration did not match runner source")
if 'parity_digest_32' in text and 'hashlib.sha256' in text:
    print("  Applied deterministic 32-prompt output-parity digest")
else:
    print("  WARNING: output-parity digest patch did not match runner source")
# OPT-IN per-request latency capture (TTFT/TPOT/e2e/spec-accept). Shared helper
# always present in the runner, but only wired to write output when the config
# carries `_agentic_latency_path` (so runs without the key behave identically).
if 'def _collect_latency' not in text:
    inject = '''
def _collect_latency(outputs, latency_path):
    ttft, tpot, e2e, queued, prefill = [], [], [], [], []
    acc_len, acc_rate, gen_toks = [], [], []
    from collections import defaultdict
    per_pos_acc = defaultdict(int)
    per_pos_drafted = defaultdict(int)
    for output in outputs:
        m = getattr(output, "metrics", None)
        if m is not None and m.arrival_time and m.first_token_ts:
            ttft.append(m.first_token_latency if m.first_token_latency else (m.first_token_ts - m.arrival_time))
            e2e.append(m.last_token_ts - m.arrival_time if m.last_token_ts else None)
            if m.scheduled_ts and m.queued_ts:
                queued.append(m.scheduled_ts - m.queued_ts)
            if m.scheduled_ts and m.first_token_ts:
                prefill.append(m.first_token_ts - m.scheduled_ts)
            ng = m.num_generation_tokens
            if ng and m.last_token_ts and m.first_token_ts:
                if ng - 1 > 0:
                    tpot.append((m.last_token_ts - m.first_token_ts) / (ng - 1))
            gen_toks.append(ng)
        for completion in output.outputs:
            sd = getattr(completion, "spec_decode_metrics", None)
            if sd is not None:
                acc_len.append(sd.mean_acceptance_length)
                acc_rate.append(sd.draft_acceptance_rate)
                # detailed mode exposes per-step accepted/drafted arrays;
                # accumulate a per-position acceptance histogram (kv: step->rate).
                psa = getattr(sd, "per_step_accepted", None)
                psd = getattr(sd, "per_step_drafted", None)
                if psa and psd:
                    for pos, (a, d) in enumerate(zip(psa, psd)):
                        per_pos_acc[pos] += a
                        per_pos_drafted[pos] += d
    def _s(x):
        xs = [v for v in x if v is not None]
        return statistics.mean(xs) if xs else None
    def _r(x):
        # Safe rounder: returns None when the metric list is empty/all-None
        # (offline llm.generate may not populate per-request timing metrics),
        # instead of round(None) which would crash and block the parity dump.
        v = _s(x)
        return round(v, 4) if v is not None else None
    per_position_acceptance = {
        str(pos): (per_pos_acc[pos] / per_pos_drafted[pos] if per_pos_drafted[pos] else None)
        for pos in sorted(set(per_pos_acc) | set(per_pos_drafted))
    }
    payload = {
        "num_requests": len(outputs),
        "mean_ttft_s": _r(ttft), "mean_tpot_s": _r(tpot),
        "mean_e2e_s": _r(e2e), "mean_queued_s": _r(queued),
        "mean_prefill_s": _r(prefill), "mean_gen_tokens": _r(gen_toks),
        "mean_spec_accept_len": _r(acc_len),
        "mean_spec_accept_rate": _r(acc_rate),
        "per_position_acceptance_rates": per_position_acceptance,
        "per_position_drafted_counts": {
            str(pos): per_pos_drafted[pos]
            for pos in sorted(set(per_pos_acc) | set(per_pos_drafted))
        },
    }
    Path(latency_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote per-request latency report: {latency_path}", flush=True)

'''
    marker = "def build_llm_kwargs(config: dict) -> dict:"
    if marker in text:
        text = text.replace(marker, inject + marker, 1)
if 'def _run_http_online' not in text:
    http_inject = '''
def _mtp_metrics_parse(text: str) -> dict:
    import re
    res = {"num_drafts": 0.0, "num_draft_tokens": 0.0, "num_accepted_tokens": 0.0, "per_position": {}}
    for line in text.splitlines():
        m = re.fullmatch(r'(vllm:spec_decode_\\w+)(?:\\{([^}]*)\\})?\\s+([0-9.eE+-]+)', line)
        if not m:
            continue
        name, labels, val = m.groups()
        for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
            if name in (f"vllm:spec_decode_{k}", f"vllm:spec_decode_{k}_total"):
                res[k] += float(val)
        if name in ("vllm:spec_decode_num_accepted_tokens_per_pos", "vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            pos = re.search(r'(?:position|pos)="(\\d+)"', labels or "")
            if pos:
                res["per_position"][pos.group(1)] = float(val)
    return res

def _mtp_counter_delta(before: str, after: str, depth: int) -> dict:
    ini, fin = _mtp_metrics_parse(before), _mtp_metrics_parse(after)
    delta = {k: fin[k] - ini[k] for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")}
    delta["accepted_tokens_per_position"] = {
        k: v - ini["per_position"].get(k, 0.0) for k, v in fin["per_position"].items()
    }
    if delta["num_drafts"] > 0 and delta["num_draft_tokens"] > 0:
        delta["draft_acceptance_rate"] = delta["num_accepted_tokens"] / delta["num_draft_tokens"]
        delta["mean_acceptance_length_including_bonus"] = 1.0 + delta["num_accepted_tokens"] / delta["num_drafts"]
        delta["mean_drafted_tokens_per_event"] = delta["num_draft_tokens"] / delta["num_drafts"]
        delta["per_position_acceptance_rates"] = {
            k: v / delta["num_drafts"] for k, v in delta["accepted_tokens_per_position"].items()
        }
    delta["engagement_verified"] = (
        delta["num_drafts"] > 0 and delta["num_draft_tokens"] > 0
        and set(delta["accepted_tokens_per_position"]) == {str(i) for i in range(depth)}
        and all(v > 0 for v in delta["accepted_tokens_per_position"].values())
        and abs(sum(delta["accepted_tokens_per_position"].values()) - delta["num_accepted_tokens"]) < 1e-3
    )
    return delta

def _run_http_online(args, config: dict) -> int:
    import asyncio
    from collections import Counter
    import subprocess
    import sys
    import threading
    import aiohttp

    model_path = os.environ["GEMMA4_TEXT_ONLY_MODEL_PATH"]
    served_model = config.get("served_model_name", "maiprofile-qwen35-sft")
    port = int(config.get("_agentic_http_port", 8100))
    max_tokens = int(config.get("max_tokens", 8192))
    default_temp = float(config.get("temperature", 0.0))
    override_temp = bool(config.get("_agentic_override_temperature", default_temp == 0.0))
    spec_cfg = config.get("speculative_config") or {
        "method": config.get("speculative_method", "mtp"),
        "num_speculative_tokens": int(config.get("spec_tokens", 3)),
        "attention_backend": config.get("speculative_attention_backend", "TRITON_ATTN"),
    }
    mtp_depth = int(spec_cfg.get("num_speculative_tokens", config.get("spec_tokens", 3)))

    r9_data_cand = Path(model_path).parent / "data" / "l1_v2_gpt56_test.jsonl"
    explicit_ds = config.get("_agentic_http_dataset")
    if explicit_ds and Path(os.path.expandvars(explicit_ds)).exists():
        ds_path = Path(os.path.expandvars(explicit_ds))
    elif r9_data_cand.exists():
        ds_path = r9_data_cand
    else:
        ds_path = DATASET_PATH

    requests = []
    with ds_path.open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "messages" in row:
                req = {
                    "messages": row["messages"],
                    "max_tokens": int(row.get("max_tokens", max_tokens)),
                    "temperature": default_temp if override_temp else float(row.get("temperature", default_temp)),
                }
                if "top_p" in config and (config.get("_agentic_override_top_p") or override_temp):
                    req["top_p"] = float(config["top_p"])
            else:
                req = {
                    "messages": [{"role": "user", "content": row["prompt"]}],
                    "max_tokens": max_tokens,
                    "temperature": default_temp,
                }
                if "top_p" in config:
                    req["top_p"] = float(config["top_p"])
            requests.append(req)
            if len(requests) == args.prompts:
                break
    eff_temps = sorted({float(r.get("temperature", 0.0)) for r in requests})
    eff_top_ps = sorted({float(r.get("top_p", 1.0)) for r in requests})
    print(f"HTTP online benchmark dataset: {ds_path} ({len(requests)} requests, effective_temperatures={eff_temps}, effective_top_ps={eff_top_ps}, greedy={eff_temps == [0.0]})", flush=True)

    srv_args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--served-model-name", served_model,
        "--tensor-parallel-size", str(config.get("tensor_parallel_size", 1)),
        "--data-parallel-size", str(config.get("data_parallel_size", 1)),
        "--max-model-len", str(config.get("max_model_len", 25600)),
        "--gpu-memory-utilization", str(config.get("gpu_memory_utilization", 0.9)),
        "--max-num-seqs", str(config.get("max_num_seqs", 128)),
        "--max-num-batched-tokens", str(config.get("max_num_batched_tokens", 16384)),
        "--dtype", str(config.get("dtype", "bfloat16")),
        "--trust-remote-code",
        "--hf-overrides", json.dumps(config.get("hf_overrides", {"architectures": ["Qwen3_5ForConditionalGeneration"]})),
        "--no-enable-log-requests",
        "--mamba-ssm-cache-dtype", str(config.get("mamba_ssm_cache_dtype", "bfloat16")),
        "--attention-backend", str(config.get("attention_backend", "FLASHINFER")),
        "--reasoning-parser", str(config.get("reasoning_parser", "qwen3")),
        "--default-chat-template-kwargs", json.dumps({"enable_thinking": bool(config.get("enable_thinking", False))}),
    ]
    if config.get("async_scheduling", True):
        srv_args.append("--async-scheduling")
    if config.get("language_model_only", True):
        srv_args.append("--language-model-only")
    if config.get("enable_prefix_caching", False):
        srv_args.append("--enable-prefix-caching")
        if config.get("mamba_cache_mode"):
            srv_args.extend(["--mamba-cache-mode", str(config["mamba_cache_mode"])])
    else:
        srv_args.append("--no-enable-prefix-caching")
    if config.get("api_server_count") and int(config["api_server_count"]) > 1:
        srv_args.extend(["--api-server-count", str(int(config["api_server_count"]))])
    if config.get("max_num_partial_prefills"):
        srv_args.extend(["--max-num-partial-prefills", str(int(config["max_num_partial_prefills"]))])
    if config.get("long_prefill_token_threshold"):
        srv_args.extend(["--long-prefill-token-threshold", str(int(config["long_prefill_token_threshold"]))])
    if config.get("disable_uvicorn_access_log", False):
        srv_args.append("--disable-uvicorn-access-log")
    if config.get("compilation_config"):
        cc = config["compilation_config"]
        srv_args.extend(["--compilation-config", json.dumps(cc) if isinstance(cc, dict) else str(cc)])
    if spec_cfg.get("method", "mtp") == "mtp":
        clean_spec = {k: v for k, v in spec_cfg.items() if k != "model"}
    else:
        clean_spec = {k: v for k, v in spec_cfg.items() if v is not None}
    srv_args.extend(["--speculative-config", json.dumps(clean_spec)])

    print("Launching HTTP online server:", " ".join(srv_args), flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    srv_log_path = RESULTS_DIR / "http_server.log"
    srv_proc = subprocess.Popen(
        srv_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    def _drain_log():
        with srv_log_path.open("w", encoding="utf-8") as lf:
            assert srv_proc.stdout is not None
            for line in srv_proc.stdout:
                lf.write(line)
                lf.flush()
                sys.stdout.write("[vllm-http] " + line)
                sys.stdout.flush()
    log_thread = threading.Thread(target=_drain_log, daemon=True)
    log_thread.start()

    async def _eval_once():
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=3600)
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=max(2000, len(requests))), timeout=timeout
        ) as session:
            ready = False
            for _ in range(900):
                if srv_proc.poll() is not None:
                    raise RuntimeError(f"vLLM HTTP server exited during init: {srv_proc.returncode}")
                try:
                    async with session.get(base + "/v1/models", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        ready = resp.status == 200
                except Exception:
                    ready = False
                if ready:
                    break
                await asyncio.sleep(2)
            if not ready:
                raise TimeoutError("vLLM HTTP server init timed out.")

            async def _get_metrics() -> str:
                try:
                    async with session.get(base + "/metrics", timeout=aiohttp.ClientTimeout(total=15)) as r:
                        return await r.text() if r.status == 200 else ""
                except Exception:
                    return ""

            warmup_reqs = int(config.get("http_warmup_requests", 0) or 0)
            if warmup_reqs > 0 and requests:
                warmup_max_tokens = int(config.get("http_warmup_max_tokens", 16) or 16)
                print(
                    f"Running {warmup_reqs} HTTP warmup requests (max_tokens={warmup_max_tokens}) before timer start...",
                    flush=True,
                )
                async def _warm_one(b: dict) -> None:
                    wb = dict(b)
                    wb["max_tokens"] = warmup_max_tokens
                    try:
                        async with session.post(
                            base + "/v1/chat/completions",
                            json={"model": served_model, **wb},
                        ) as resp:
                            await resp.read()
                    except Exception:
                        pass
                await asyncio.gather(*(_warm_one(requests[i % len(requests)]) for i in range(warmup_reqs)))
                print("HTTP warmup requests complete.", flush=True)

            before_m = await _get_metrics()
            completed = 0

            async def _one(idx: int, body: dict) -> dict:
                nonlocal completed
                t0 = time.perf_counter()
                row_res = {"index": idx, "success": False}
                try:
                    async with session.post(
                        base + "/v1/chat/completions",
                        json={"model": served_model, **body},
                    ) as resp:
                        row_res["http_status"] = resp.status
                        payload = await resp.json()
                        if resp.status != 200:
                            row_res["error"] = payload
                        else:
                            ch = payload["choices"][0]
                            content = ch["message"]["content"]
                            try:
                                json.loads(content)
                                valid_json = True
                            except Exception:
                                valid_json = False
                            row_res.update(
                                success=True,
                                json_parseable=valid_json,
                                prompt_tokens=payload["usage"]["prompt_tokens"],
                                output_tokens=payload["usage"]["completion_tokens"],
                                finish_reason=ch.get("finish_reason"),
                                reasoning_content_present=bool(ch["message"].get("reasoning_content")),
                            )
                except Exception as exc:
                    row_res["error"] = f"{type(exc).__name__}: {exc}"
                row_res["latency_seconds"] = time.perf_counter() - t0
                completed += 1
                if completed % 100 == 0 or completed == len(requests):
                    print(f"Completed HTTP requests: {completed}/{len(requests)}", flush=True)
                return row_res

            t_start = time.perf_counter()
            rows = await asyncio.gather(*(_one(i, b) for i, b in enumerate(requests)))
            elapsed = time.perf_counter() - t_start
            after_m = await _get_metrics()
            return rows, elapsed, _mtp_counter_delta(before_m, after_m, mtp_depth)

    try:
        rows, elapsed, mtp_delta = asyncio.run(_eval_once())
    finally:
        if srv_proc.poll() is None:
            srv_proc.terminate()
            try:
                srv_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                srv_proc.kill()
                srv_proc.wait(timeout=10)
        log_thread.join(timeout=10)

    succ = [r for r in rows if r["success"]]
    if len(succ) != len(rows):
        print(f"WARNING: {len(rows) - len(succ)} HTTP requests failed!", flush=True)
    prompt_toks = sum(r["prompt_tokens"] for r in succ)
    output_toks = sum(r["output_tokens"] for r in succ)
    out_tps = output_toks / elapsed if elapsed > 0 else 0.0
    tot_tps = (prompt_toks + output_toks) / elapsed if elapsed > 0 else 0.0
    qps = len(succ) / elapsed if elapsed > 0 else 0.0
    mean_in = prompt_toks / len(succ) if succ else 0.0
    mean_out = output_toks / len(succ) if succ else 0.0
    mean_lat = sum(r["latency_seconds"] for r in succ) / len(succ) if succ else 0.0
    json_ok = sum(1 for r in succ if r.get("json_parseable"))

    summary = {
        "requests": len(rows),
        "successful_requests": len(succ),
        "failed_requests": len(rows) - len(succ),
        "wall_seconds": elapsed,
        "prompt_tokens": prompt_toks,
        "output_tokens": output_toks,
        "qps": qps,
        "output_tps": out_tps,
        "total_tps": tot_tps,
        "mean_input_tokens": mean_in,
        "mean_output_tokens": mean_out,
        "mean_latency_seconds_including_queueing": mean_lat,
        "json_parseable": json_ok,
        "json_parseable_fraction_of_successes": json_ok / len(succ) if succ else 0.0,
        "finish_reason_counts": dict(Counter(r.get("finish_reason") for r in succ)),
        "output_cap_hits": sum(1 for r in succ if r["output_tokens"] == max_tokens),
        "reasoning_content_requests": sum(1 for r in succ if r.get("reasoning_content_present")),
        "requested_mtp_depth": mtp_depth,
        "resolved_mtp_depth": mtp_depth,
        "mtp_counters": mtp_delta,
        "mtp_engagement_verified": mtp_delta.get("engagement_verified", False),
        "effective_temperatures": eff_temps,
        "effective_top_ps": eff_top_ps,
        "greedy_decoding_verified": eff_temps == [0.0],
    }
    result = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": "http_online_chat_completions",
        "dataset": str(ds_path),
        "config": config,
        "mean_output_tps": round(out_tps, 2),
        "mean_total_tps": round(tot_tps, 2),
        "stdev_output_tps": 0.0,
        "mean_qps": round(qps, 2),
        "mean_elapsed_time": round(elapsed, 2),
        "mean_out": round(mean_out, 1),
        "summary": summary,
        "status": "ok" if len(succ) == len(rows) else "partial_failure",
    }
    result_path = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"output_tps: {result['mean_output_tps']:.2f} +/- 0.00", flush=True)
    print(f"total_tps: {result['mean_total_tps']:.2f}", flush=True)
    print(f"mean_out: {result['mean_out']:.1f}", flush=True)
    print(f"qps: {result['mean_qps']:.2f}", flush=True)
    print(f"elapsed_time: {result['mean_elapsed_time']:.2f}s", flush=True)
    print(f"json_parseable: {json_ok}/{len(succ)} ({summary['json_parseable_fraction_of_successes']:.4f})", flush=True)
    print(f"draft_acceptance_rate: {mtp_delta.get('draft_acceptance_rate', 0.0):.4f}", flush=True)
    print(f"Results saved: {result_path}", flush=True)
    return 0 if len(succ) == len(rows) else 1

'''
    marker = "def build_llm_kwargs(config: dict) -> dict:"
    if marker in text:
        text = text.replace(marker, http_inject + marker, 1)
    main_hook_old = '    with open(args.config, encoding="utf-8") as config_file:\n        config = json.load(config_file)\n'
    main_hook_new = (
        '    with open(args.config, encoding="utf-8") as config_file:\n'
        '        config = json.load(config_file)\n'
        '    if config.get("_agentic_http_online"):\n'
        '        return _run_http_online(args, config)\n'
    )
    if main_hook_old in text:
        text = text.replace(main_hook_old, main_hook_new, 1)
        print("  Applied HTTP /v1/chat/completions online-service benchmark mode patch")
# Persist the latency + HTTP online helpers: the earlier path.write_text(text) happened BEFORE
# this injection, so without this second write the helper was compiled in memory
# but never landed on disk -> NameError at runtime (wedged ~66% through gen).
path.write_text(text)
compile(text, str(path), "exec")
PY
if [ $? -ne 0 ]; then
    echo "ERROR: MTP benchmark runner compatibility patch failed"
    exit 1
fi
fi
fi

# 1b. Activate the requested vLLM runtime.
# Strategy: copy compiled .so files from Docker's vllm into our fork's source
# tree, then use PYTHONPATH to shadow Docker's Python files with our fork's.
# This avoids ~30-60 min CUDA recompilation while ensuring our scheduler /
# attention / MoE Python changes are the ones that run.
# For patch branches: clone that branch separately and install it instead.
echo "[Step 1b] Activating vllm-msn fork (main)..."
# No separate clone needed — runner and vLLM source are identical
echo "  Source: /tmp/runner_src"
echo "  Branch: main"
echo "  Start: $(date)"

if [ "pip_isolated" = "system" ]; then
    echo "  Using the original vLLM installation from the AML image."
    unset PYTHONPATH
    python -c "import importlib.metadata, vllm; print('  vLLM version:', importlib.metadata.version('vllm'), '| path:', vllm.__file__)"
elif [ "pip_isolated" = "pip" ]; then
    echo "  Installing official vLLM 0.31.0 package from PyPI..."
    sudo -H -E python -m pip uninstall -y vllm 2>&1 || true
    sudo rm -f /usr/local/bin/vllm 2>/dev/null || true
    sudo -H -E python -m pip install --upgrade "vllm==0.31.0"
    # The base image's CUDA 12.9 Torchaudio and stale FlashInfer JIT cache / cubin
    # packages (0.6.x) are incompatible with vLLM 0.28+/0.31's Torch 2.13/CUDA 13
    # and FlashInfer 0.7+ dependencies.
    sudo python -m pip uninstall -y torchaudio flashinfer-jit-cache flashinfer-cubin 2>&1 || python -m pip uninstall -y torchaudio flashinfer-jit-cache flashinfer-cubin 2>&1 || true
    # PyPI does not publish a flashinfer-cubin package matching vLLM 0.28's
    # FlashInfer dependency. A100 cannot use the affected FlashInfer paths.
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    VLLM_NVIDIA_LIBS=$(python - <<'PY'
import site
from pathlib import Path

roots = [Path(path) for path in site.getsitepackages()]
roots.append(Path(site.getusersitepackages()))
print(":".join(str(path) for root in roots for path in root.glob("nvidia/*/lib") if path.is_dir()))
PY
)
    export LD_LIBRARY_PATH="$VLLM_NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
    unset PYTHONPATH
    sudo -H -E python - <<'PY' || true
from pathlib import Path
import vllm

path = Path(vllm.__file__).parent / "transformers_utils/model_arch_config_convertor.py"
text = path.read_text()
base_old = '''    def get_head_size(self) -> int:
        if self.is_deepseek_mla():
'''
base_new = '''    def get_head_size(self) -> int:
        try:
            setattr(self.hf_text_config, "allow_global_per_layer_attribute_access", True)
        except Exception:
            pass
        if self.is_deepseek_mla():
'''
if base_old in text:
    path.write_text(text.replace(base_old, base_new, 1))
    print("  Applied generic heterogeneous head_dim compatibility patch")
PY
    python -c "import importlib.metadata, vllm; assert importlib.metadata.version('vllm') == '0.31.0'; print('  vLLM version:', vllm.__version__, '| path:', vllm.__file__)"
    echo "  Official vLLM 0.31.0 package installed at $(date)"
elif [ "pip_isolated" = "pip_isolated" ]; then
    echo "  Installing official vLLM 0.31.0 in an isolated virtual environment..."
    python -m venv /tmp/vllm_venv
    export PATH="/tmp/vllm_venv/bin:$PATH"
    unset PYTHONPATH
    python -m pip install --upgrade pip
    python -m pip install "vllm==0.31.0"
    # vLLM 0.27 allows an unbounded Transformers 5 dependency. Transformers
    # 5.16.1 regresses Gemma4 heterogeneous-head loading (512-wide checkpoint
    # weight into a 256-wide parameter); 5.14.1 is the version used by the
    # successful exact-v0.27 matrix and global-pip retries.
    if [ "0.31.0" = "0.27.0" ]; then
        python -m pip install "transformers==5.14.1"
    fi
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    VLLM_NVIDIA_LIBS=$(python - <<'PY'
import site
from pathlib import Path

roots = [Path(path) for path in site.getsitepackages()]
roots.append(Path(site.getusersitepackages()))
print(":".join(str(path) for root in roots for path in root.glob("nvidia/*/lib") if path.is_dir()))
PY
)
    export LD_LIBRARY_PATH="$VLLM_NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
    python - <<'PY'
from pathlib import Path
import vllm

path = Path(vllm.__file__).parent / "transformers_utils/model_arch_config_convertor.py"
text = path.read_text()
base_old = '''    def get_head_size(self) -> int:
        if self.is_deepseek_mla():
'''
base_new = '''    def get_head_size(self) -> int:
        try:
            setattr(self.hf_text_config, "allow_global_per_layer_attribute_access", True)
        except Exception:
            pass
        if self.is_deepseek_mla():
'''
gemma_old = '''    def get_head_size(self) -> int:
        # Gemma4 uses dual head dimensions: head_dim (sliding attention)
'''
gemma_new = '''    def get_head_size(self) -> int:
        try:
            setattr(self.hf_text_config, "allow_global_per_layer_attribute_access", True)
        except Exception:
            pass
        # Gemma4 uses dual head dimensions: head_dim (sliding attention)
'''
if base_old in text:
    text = text.replace(base_old, base_new, 1)
if gemma_old in text:
    text = text.replace(gemma_old, gemma_new, 1)
path.write_text(text)
print("  Applied Transformers heterogeneous head_dim compatibility patch")

if "0.31.0" in ('0.23.0', '0.24.0'):
    target_path = Path(vllm.__file__).parent / "model_executor/models/gemma4.py"
    target_text = target_path.read_text()
    target_old = '''        # Gemma4 uses different head dimensions for sliding vs full attention
        layer_type = config.layer_types[layer_idx]
        self.is_full_attention = layer_type == "full_attention"
        if self.is_full_attention:
            head_dim = getattr(config, "global_head_dim", config.head_dim)
        else:
            head_dim = config.head_dim
'''
    target_new = '''        # Transformers 5 stores Gemma4 attention dimensions per layer.
        layer_config = (
            config.per_layer_config[layer_idx]
            if hasattr(config, "per_layer_config")
            else config
        )
        layer_type = config.layer_types[layer_idx]
        self.is_full_attention = layer_type == "full_attention"
        head_dim = layer_config.head_dim
        num_heads = (
            getattr(config, "num_global_attention_heads", layer_config.num_attention_heads)
            if self.is_full_attention
            else layer_config.num_attention_heads
        )
        num_kv_heads = layer_config.num_key_value_heads
        if __import__("os").getenv("VLLM_GEMMA4_LEGACY_DEBUG_SHAPES"):
            print(
                "Gemma4 v0.23 target attention shape:",
                layer_idx, layer_type, head_dim, num_heads, num_kv_heads,
                getattr(config, "num_global_attention_heads", None),
            )
'''
    if target_old not in target_text:
        raise RuntimeError("vLLM legacy Gemma4 target layer patch context not found")
    target_text = target_text.replace(target_old, target_new, 1)
    target_old = "            num_heads=config.num_attention_heads,\n"
    if target_old not in target_text:
        raise RuntimeError("vLLM legacy Gemma4 target attention patch context not found")
    target_path.write_text(target_text.replace(target_old, "            num_heads=num_heads,\n", 1))

    draft_path = Path(vllm.__file__).parent / "model_executor/models/gemma4_mtp.py"
    draft_text = draft_path.read_text()
    draft_old = '''        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        is_full_attention = layer_type == "full_attention"
        head_dim = (
            getattr(config, "global_head_dim", config.head_dim)
            if is_full_attention
            else config.head_dim
        )
'''
    draft_new = '''        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        is_full_attention = layer_type == "full_attention"
        layer_config = (
            config.per_layer_config[layer_idx]
            if hasattr(config, "per_layer_config")
            else config
        )
        head_dim = layer_config.head_dim
        num_heads = (
            getattr(config, "num_global_attention_heads", layer_config.num_attention_heads)
            if is_full_attention
            else layer_config.num_attention_heads
        )
        num_kv_heads = layer_config.num_key_value_heads
        if __import__("os").getenv("VLLM_GEMMA4_LEGACY_DEBUG_SHAPES"):
            print(
                "Gemma4 v0.23 MTP attention shape:",
                layer_idx, layer_type, head_dim, num_heads, num_kv_heads,
                getattr(config, "num_global_attention_heads", None),
            )
'''
    if draft_old not in draft_text:
        raise RuntimeError("vLLM legacy Gemma4 MTP layer patch context not found")
    draft_text = draft_text.replace(draft_old, draft_new, 1)
    draft_old = "            num_heads=config.num_attention_heads,\n"
    if draft_old not in draft_text:
        raise RuntimeError("vLLM legacy Gemma4 MTP attention patch context not found")
    draft_path.write_text(draft_text.replace(draft_old, "            num_heads=num_heads,\n", 1))
    print("  Applied legacy vLLM Gemma4 per-layer attention config backport")
PY
    python -c "import importlib.metadata, vllm; assert importlib.metadata.version('vllm') == '0.31.0'; print('  vLLM version:', vllm.__version__, '| Transformers:', importlib.metadata.version('transformers'), '| path:', vllm.__file__)"
    echo "  Isolated official vLLM 0.31.0 package installed at $(date)"
elif [ "pip_isolated" = "source" ]; then
    echo "  Building and installing vLLM directly from main..."
    unset PYTHONPATH
    cd /tmp/runner_src

    # Deterministically force a real csrc recompile. The vllm_gemma4:9 image
    # pins VLLM_USE_PRECOMPILED / VLLM_PRECOMPILED_WHEEL_LOCATION as immutable
    # env vars, so shell `export` cannot override them (envs.py:666-668 treats a
    # set VLLM_PRECOMPILED_WHEEL_LOCATION as precompiled regardless of the flag).
    # Instead of fighting the environment, patch the cloned source so the
    # precompiled path is impossible: envs.VLLM_USE_PRECOMPILED -> False and
    # setup.py USE_PRECOMPILED_EXTENSIONS -> False => cmake_build_ext compiles
    # the fork's own csrc (6-arg gptq_marlin_repack), fixing the ABI mismatch.
    if grep -q "VLLM_USE_PRECOMPILED" vllm/envs.py; then
        python - <<'PY'
from pathlib import Path
p = Path("vllm/envs.py")
t = p.read_text()
old = '''        os.environ.get("VLLM_USE_PRECOMPILED", "").strip().lower() in ("1", "true")
        or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION"))'''
new = '''        os.environ.get("VLLM_USE_PRECOMPILED_FORCE_0", "").strip().lower() in ("1", "true")
        or (False and bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION")))'''
if old in t:
    t = t.replace(old, new, 1)
    p.write_text(t)
    print("  Patched envs.py: VLLM_USE_PRECOMPILED always False")
else:
    print("  WARN envs.py precompiled lambda pattern not found; check manually")
PY
    fi
    python - <<'PY'
from pathlib import Path
p = Path("setup.py")
t = p.read_text()
old = "USE_PRECOMPILED_EXTENSIONS = envs.VLLM_USE_PRECOMPILED"
new = "USE_PRECOMPILED_EXTENSIONS = False  # force real csrc compile"
if old in t:
    t = t.replace(old, new, 1)
    p.write_text(t)
    print("  Patched setup.py: USE_PRECOMPILED_EXTENSIONS = False")
else:
    print("  WARN setup.py USE_PRECOMPILED_EXTENSIONS line not found; check manually")
# On A100 (sm_80) the Hopper/SM100-only extension units (_vllm_fa3_C: FA3,
# _flashmla_C/_flashmla_extension_C, _flashkda_C, _deep_gemm_C, _qutlass_C)
# are each guarded by a `get_nvcc_cuda_version() >= Version("12.x")` gate that
# the image (CUDA 12.9) satisfies, so they compile as real .so under a forced
# source build and FAIL because cusparse.h is missing from the image's
# runtime-only CUDA include tree (CUDAContext.h -> #include <cusparse.h>).
# STRUCTURE-SAFE FIX: bump every such version gate to Version("99.0") so those
# units are never added -- leaves the surrounding multiline ext_modules.append()
# expressions syntactically intact (a plain `pass` rewrite breaks the FA4
# multiline append). FA4-CuteDSL / fmha_sm100 / tml_fa4 are Python-file-copy
# optional packages (never compile .so / don't touch cusparse.h) and stay.
# Keep SM80-relevant units (_vllm_fa2_C, the main _C, _moe) intact.
import re as _re
t = _re.sub(r'get_nvcc_cuda_version\(\)\s*>=\s*Version\("[0-9.]+"\)',
            'get_nvcc_cuda_version() >= Version("99.0")', t)
p.write_text(t)
print("  Patched setup.py: neutered SM90+/SM100-only extension version gates")
PY
    python - <<'PY'
# The vllm_gemma4:9 image bakes TORCH_CUDA_ARCH_LIST / MAX_JOBS as immutable env
# vars that are re-injected into every spawned subprocess, so shell `export`
# before `pip install` does NOT survive into the cmake build. Force them inside
# setup.py (which executes at build time, after any env reset) so a source build
# compiles ONLY for the A100's sm_80 arch instead of the image's full
# 7.5-12.0 matrix (which gratuitously builds SM90/SM100 units and risks more
# missing-header failures on those arches).
from pathlib import Path
p = Path("setup.py")
t = p.read_text()
_pin = 'os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"\n'
if 'os.environ["TORCH_CUDA_ARCH_LIST"]' not in t:
    # Insert on the line after `import os` so os is definitely in scope. If the
    # fork has no bare `import os`, fall back to after the first import line but
    # ensure os is available by prepending an import if needed.
    _os_import = "import os\n"
    if _os_import in t:
        idx = t.index(_os_import) + len(_os_import)
    else:
        _anchor = t.find("\n", t.find("import"))
        if _anchor == -1:
            raise RuntimeError("setup.py has no import block; cannot pin arch list.")
        idx = _anchor + 1
        t = t[:idx] + _os_import + t[idx:]
        idx = idx + len(_os_import)
    t = t[:idx] + _pin + t[idx:]
    p.write_text(t)
    print("  Patched setup.py: pinned TORCH_CUDA_ARCH_LIST to 8.0 (survives image env re-injection)")
else:
    print("  setup.py arch pin already present")
PY
    python - <<'PY'
from pathlib import Path

path = Path(r"/tmp/runner_src") / "vllm/transformers_utils/model_arch_config_convertor.py"
old = '''    def get_head_size(self) -> int:
        # Gemma4 uses dual head dimensions: head_dim (sliding attention)
        # and global_head_dim (full attention).  Return the largest so
        # that attention backends allocate buffers large enough for both.
        head_dim = getattr(self.hf_text_config, "head_dim", 0)
        global_head_dim = getattr(self.hf_text_config, "global_head_dim", 0)
        return max(head_dim, global_head_dim) or super().get_head_size()
'''
new = '''    def get_head_size(self) -> int:
        # Gemma4 configs can be heterogeneous; allow global fallback for sizing.
        cfg = self.hf_text_config
        try:
            setattr(cfg, "allow_global_per_layer_attribute_access", True)
        except Exception:
            pass
        head_dim = getattr(cfg, "head_dim", 0)
        global_head_dim = getattr(cfg, "global_head_dim", 0)
        return max(head_dim, global_head_dim) or super().get_head_size()
'''
if path.exists():
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new, 1))
        print("  Applied Gemma4 heterogeneous head_dim compatibility patch")
        text = path.read_text()
    base_old = '''    def get_head_size(self) -> int:
        if self.is_deepseek_mla():
'''
    base_new = '''    def get_head_size(self) -> int:
        try:
            setattr(self.hf_text_config, "allow_global_per_layer_attribute_access", True)
        except Exception:
            pass
        if self.is_deepseek_mla():
'''
    if base_old in text:
        path.write_text(text.replace(base_old, base_new, 1))
        print("  Applied generic heterogeneous head_dim compatibility patch")
PY
    echo "  Ensuring build dependencies for the editable source install are present..."
    # nvidia-cuda-nvrtc-cu12 provides libnvrtc.so; the *cusparse/*cublas/*cusolver
    # etc. -cu12 packages additionally ship the CUDA dev include/ headers that the
    # runtime-only base image lacks. The fork recompiles the main _C extension,
    # which needs cusparse.h (via ATen/cuda/CUDAContextLight.h), so install the
    # header-carrying runtime packages and surface their include/ dirs on CPATH
    # (nvcc follows CPATH for system headers).
    if ! sudo -H -E python -m pip install --upgrade setuptools-rust cmake ninja         nvidia-cuda-nvrtc-cu12 nvidia-cusparse-cu12 nvidia-cublas-cu12         nvidia-cusolver-cu12 nvidia-curand-cu12 nvidia-cufft-cu12         nvidia-cuda-runtime-cu12 nvidia-nccl-cu12 2>&1; then
        echo "ERROR: failed to install build dependencies for source build"
        exit 1
    fi
    VLLM_NVIDIA_LIBS=$(python - <<'PY'
import site
from pathlib import Path

roots = [Path(path) for path in site.getsitepackages()]
roots.append(Path(site.getusersitepackages()))
print(":".join(str(path) for root in roots for path in root.glob("nvidia/*/lib") if path.is_dir()))
PY
)
    VLLM_NVIDIA_INCLUDES=$(python - <<'PY'
import site
from pathlib import Path

roots = [Path(path) for path in site.getsitepackages()]
roots.append(Path(site.getusersitepackages()))
print(":".join(str(path) for root in roots for path in root.glob("nvidia/*/include") if path.is_dir()))
PY
)
    export LD_LIBRARY_PATH="$VLLM_NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$VLLM_NVIDIA_LIBS:${LIBRARY_PATH:-}"
    export CMAKE_LIBRARY_PATH="$VLLM_NVIDIA_LIBS:${CMAKE_LIBRARY_PATH:-}"
    export CPATH="$VLLM_NVIDIA_INCLUDES:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="$VLLM_NVIDIA_INCLUDES:${CPLUS_INCLUDE_PATH:-}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
    export MAX_JOBS="${MAX_JOBS:-8}"
    export NVCC_THREADS="${NVCC_THREADS:-1}"
    if [ "${VLLM_USE_PRECOMPILED:-1}" = "0" ]; then
        # Force a real csrc recompile: envs.py treats VLLM_USE_PRECOMPILED as
        # true if EITHER the flag is 1 OR VLLM_PRECOMPILED_WHEEL_LOCATION is set
        # (fork vllm/envs.py:666-668). The vllm_gemma4:9 image sets the wheel
        # location, so exporting the flag alone is insufficient. Blank both so
        # USE_PRECOMPILED_EXTENSIONS is False and setup.py runs cmake_build_ext.
        # Use empty-string exports (not unset) because they survive a possible
        # docker-env reinjection at process spawn, and bool("") is False.
        export VLLM_USE_PRECOMPILED="0"
        export VLLM_PRECOMPILED_WHEEL_LOCATION=""
        export VLLM_PRECOMPILED_WHEEL_VARIANT=""
        export VLLM_PRECOMPILED_WHEEL_COMMIT=""
        export VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX="0"
    else
        export VLLM_USE_PRECOMPILED="1"
    fi
    export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
    if [ "$TORCH_COMPILE_DISABLE" = "0" ]; then
        unset TORCH_COMPILE_DISABLE
    fi
    NVRTC_LIBRARY=$(python - <<'PY'
import site
from pathlib import Path

roots = [Path(path) for path in site.getsitepackages()]
roots.append(Path(site.getusersitepackages()))
candidates = [path for root in roots for path in root.glob("nvidia/cuda_nvrtc/lib/libnvrtc.so*")]
print(candidates[0] if candidates else "")
PY
)
    if [ -n "$NVRTC_LIBRARY" ]; then
        export CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDA_nvrtc_LIBRARY=$NVRTC_LIBRARY"
        echo "  CUDA nvrtc library: $NVRTC_LIBRARY"
    else
        echo "  WARNING: CUDA nvrtc library not found in Python NVIDIA packages"
    fi
    echo "  Removing any stale vLLM install before the source build..."
    sudo -H -E python -m pip uninstall -y vllm 2>&1 || true
    sudo rm -f /usr/local/bin/vllm 2>/dev/null || true
    echo "  Installing with VLLM_USE_PRECOMPILED=$VLLM_USE_PRECOMPILED, TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    echo "  Runtime safeguard: TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-unset}"
    if ! sudo -H -E python -m pip install --no-build-isolation --no-deps -e .; then
        echo "ERROR: vLLM source install failed"
        exit 1
    fi
    python -m pip install -q sentencepiece tiktoken gguf
    if ! python -c "
from pathlib import Path
import vllm
source = Path(r'/tmp/runner_src').resolve()
loaded = Path(vllm.__file__).resolve()
assert loaded.is_relative_to(source), f'vLLM loaded from {loaded}, expected source under {source}'
print('  vLLM source install:', loaded)
"; then
        echo "ERROR: vLLM source import verification failed"
        exit 1
    fi
    cd "$RUNNER_DIR"
    echo "  Source build complete at $(date)"
else
# Strategy: use the Docker image's pre-compiled CUDA extensions (.so files)
# with our fork's Python source.
#
# WHY: pip install -e . --no-build-isolation requires ~30-60 min to recompile
# CUDA extensions, and often fails if CUDA env vars or build tools are missing.
# The Docker image ships vllm with all .so files already compiled for this GPU.
# Our fork only changes Python files (scheduler, attention routing, MoE logic)
# — the compiled extensions are ABI-compatible.
#
# APPROACH:
#   1. Copy compiled .so files from Docker's vllm into our fork's source tree
#   2. retain the Docker distribution metadata for platform detection
#   3. export PYTHONPATH to our fork — Python finds our .py files first,
#      and the .so files are already in place within the same tree
echo "  Copying compiled CUDA extensions from Docker image to fork..."
SYSTEM_VLLM=$(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "")
if [ -n "$SYSTEM_VLLM" ]; then
    echo "  System vllm found at: $SYSTEM_VLLM"
    # Copy all compiled .so files into our fork's vllm directory
    find "$SYSTEM_VLLM" -name "*.so" | while read sofile; do
        relpath="${sofile#$SYSTEM_VLLM/}"
        case "$relpath" in
            _flashmla_C.abi3.so|_flashmla_extension_C.abi3.so)
                echo "    Skipped unsupported A100 FlashMLA extension: $relpath"
                continue
                ;;
        esac
        destdir="/tmp/runner_src/vllm/$(dirname "$relpath")"
        mkdir -p "$destdir"
        cp -f "$sofile" "$destdir/" 2>/dev/null && echo "    Copied: $relpath" || true
    done
    SO_COUNT=$(find "$SYSTEM_VLLM" -name "*.so" | wc -l)
    echo "  Copied $SO_COUNT .so file(s) into /tmp/runner_src/vllm/"
    # Some vLLM wheels expose the stable extension as _C_stable_libtorch while
    # the source package expects the editable-install shim vllm._C. If the
    # branch cannot be installed, create the equivalent local shim so Python
    # can still load the Docker image's compiled operators.
    if [ ! -e "/tmp/runner_src/vllm/_C.py" ] && [ -e "/tmp/runner_src/vllm/_C_stable_libtorch.abi3.so" ]; then
        printf 'from ._C_stable_libtorch import *
' > "/tmp/runner_src/vllm/_C.py"
        echo "  Created vllm._C shim for _C_stable_libtorch"
    fi
    if [ ! -e "/tmp/runner_src/vllm/version.py" ]; then
        printf '__version__ = "0.26.0"
__version_tuple__ = (0, 26, 0)
' > "/tmp/runner_src/vllm/version.py"
        echo "  Created vllm.version shim"
    fi
else
    echo "  [WARN] Could not locate system vllm — .so files not copied"
fi

echo "  Retaining Docker vllm distribution metadata for platform detection."

# No recompilation needed — .so files already copied above.
# Just install Python deps that aren't in the image (no build isolation needed
# since we're not building extensions).
cd /tmp/runner_src
python - <<'PY'
from pathlib import Path

path = Path(r"/tmp/runner_src") / "vllm/transformers_utils/model_arch_config_convertor.py"
old = '''    def get_head_size(self) -> int:
        # Gemma4 uses dual head dimensions: head_dim (sliding attention)
        # and global_head_dim (full attention).  Return the largest so
        # that attention backends allocate buffers large enough for both.
        head_dim = getattr(self.hf_text_config, "head_dim", 0)
        global_head_dim = getattr(self.hf_text_config, "global_head_dim", 0)
        return max(head_dim, global_head_dim) or super().get_head_size()
'''
new = '''    def get_head_size(self) -> int:
        # Gemma4 configs can be heterogeneous; allow global fallback for sizing.
        cfg = self.hf_text_config
        try:
            setattr(cfg, "allow_global_per_layer_attribute_access", True)
        except Exception:
            pass
        head_dim = getattr(cfg, "head_dim", 0)
        global_head_dim = getattr(cfg, "global_head_dim", 0)
        return max(head_dim, global_head_dim) or super().get_head_size()
'''
if path.exists():
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new, 1))
        print("  Applied Gemma4 heterogeneous head_dim compatibility patch")
        text = path.read_text()
    base_old = '''    def get_head_size(self) -> int:
        if self.is_deepseek_mla():
'''
    base_new = '''    def get_head_size(self) -> int:
        try:
            setattr(self.hf_text_config, "allow_global_per_layer_attribute_access", True)
        except Exception:
            pass
        if self.is_deepseek_mla():
'''
    if base_old in text:
        path.write_text(text.replace(base_old, base_new, 1))
        print("  Applied generic heterogeneous head_dim compatibility patch")
PY
# Fork .py source is shadowed via PYTHONPATH below and the compiled .so files
# were copied from the Docker image above, so an editable build is NOT required.
# Attempt a metadata-only develop install (a lightweight setuptools "egg-link"),
# but never fail the job on it -- PYTHONPATH + copied .so is the load path.
python -m pip install --no-build-isolation --no-deps -e . 2>&1 | tail -3 || echo "  [WARN] editable install skipped (relying on PYTHONPATH + copied .so)"
echo "  Install complete at $(date)"

# Prepend the installed vLLM source and optional Python-module runner.
if [ -n "$RUNNER_PYTHONPATH" ]; then
    export PYTHONPATH=/tmp/runner_src:$RUNNER_PYTHONPATH:$PYTHONPATH
else
    export PYTHONPATH=/tmp/runner_src:$PYTHONPATH
fi
echo "  PYTHONPATH set: $PYTHONPATH"
python -c "import vllm; print('  vLLM version:', vllm.__version__, '| path:', vllm.__file__)"
# Verify .so extensions are loadable (platform detection requires vllm._C)
python -c "
import sys
try:
    import vllm._C
    print('  vllm._C OK')
except ImportError as e:
    print('  [WARN] vllm._C not loadable:', e, file=sys.stderr)
"
# Hard-fail if Python source is still loading from the system dist-packages.
# The retained distribution metadata is intentional: source overlays consult
# importlib.metadata during CUDA platform detection.
python -c "
import vllm, sys
p = vllm.__file__
if ('dist-packages' in p or 'site-packages' in p) and '/tmp/' not in p:
    print('ERROR: vLLM still loading from system install!', p, file=sys.stderr)
    sys.exit(1)
print('  vLLM path OK:', p)
"
# Install tokenizer dependencies missing from base image (use python -m pip to target the correct interpreter)
python -m pip install -q sentencepiece tiktoken gguf
echo "  sentencepiece/tiktoken/gguf installed"
cd "$RUNNER_DIR"
fi

# 2. Copy data from Cosmos mount to local disk
echo "[Step 2/7] Copying dataset to local disk..."
MOUNT="${AZURE_ML_INPUT_datastore:-}"
if [ -z "$MOUNT" ]; then
    MOUNT="${AZURE_ML_INPUT_msndni:-}"
fi
mkdir -p datasets
DATA_TARGET="datasets/sc1_delta_v2.jsonl"
if [ -z "$MOUNT" ]; then
    echo "ERROR: datastore mount is unavailable (checked AZURE_ML_INPUT_datastore and AZURE_ML_INPUT_msndni)"
    exit 1
fi
DATA_SOURCE="$MOUNT/shares/users/zxy/autobench/data/qwen35-dflash-bench/sc1_delta_v2.jsonl"
echo "  Dataset source: $DATA_SOURCE"
for attempt in 1 2 3 4 5; do
    if timeout 30 cp "$DATA_SOURCE" "$DATA_TARGET" 2>/dev/null && [ -s "$DATA_TARGET" ]; then
        break
    fi
    echo "  Dataset copy attempt $attempt/5 failed or timed out; retrying..."
    rm -f "$DATA_TARGET"
    sleep 5
done
if [ ! -s "$DATA_TARGET" ]; then
    echo "ERROR: dataset copy failed after 5 attempts: $DATA_SOURCE"
    exit 1
fi
echo "  Data copied: $(wc -l < datasets/sc1_delta_v2.jsonl) lines"


# 3. Load model from dtlake mount (primary) or HF fallback
echo "[Step 3/7] Loading model weights..."
MODEL_MOUNT="$AZURE_ML_INPUT_model_store"
MODEL_LOADED=0

if [ -n "$MODEL_MOUNT" ] && [ -n "$(ls -A "$MODEL_MOUNT/text_only" 2>/dev/null)" ]; then
    echo "  Using dtlake mount: $MODEL_MOUNT"
    ls -lh "$MODEL_MOUNT/" 2>&1 | head -10
    echo "  Model accessible at $(date)"
    MODEL_LOADED=1
elif [ -n "$MODEL_MOUNT" ] && [ -n "$(ls -A "$MODEL_MOUNT/dflash" 2>/dev/null)" ]; then
    echo "  Using dtlake mount (DFlash mode, no text_only required): $MODEL_MOUNT"
    ls -lh "$MODEL_MOUNT/" 2>&1 | head -10
    echo "  Model accessible at $(date)"
    MODEL_LOADED=1
else
    echo "  [WARN] dtlake mount unavailable or empty (AZURE_ML_INPUT_model_store='$MODEL_MOUNT')"
fi

# Fallback: HuggingFace download
if [ "$MODEL_LOADED" -eq 0 ]; then
    echo "  Falling back to HuggingFace download..."
    mkdir -p /tmp/models/gemma4
    echo "  Repo: Xinyi0625/Gemma-4-26B-A4B-it-deploy"
    echo "  Start: $(date)"
    python -m pip install -q "huggingface_hub>=0.23"
    python -c "
import subprocess, sys
result = subprocess.run(
    [sys.executable, '-m', 'huggingface_hub.commands.huggingface_cli', 'download',
     '--repo-type', 'model', '--local-dir', '/tmp/models/gemma4', '--local-dir-use-symlinks', 'False',
     'Xinyi0625/Gemma-4-26B-A4B-it-deploy'],
    check=False
)
sys.exit(result.returncode)
" 2>&1 | tail -20
    HF_RC=$?
    if [ $HF_RC -ne 0 ]; then
        echo "  ERROR: HF download failed (rc=$HF_RC)"
        exit 1
    fi
    echo "  HF download complete at $(date)"
    # Verify
    if [ -z "$(ls -A /tmp/models/gemma4/text_only 2>/dev/null)" ] || [ -z "$(ls -A /tmp/models/gemma4/assistant 2>/dev/null)" ]; then
        echo "ERROR: model dirs empty after HF download — aborting"
        exit 1
    fi
    MODEL_LOADED=1
    # Use local dirs for env vars below
    MODEL_MOUNT="/tmp/models/gemma4"
fi

# 4. Set environment variables
echo "[Step 4/7] Setting environment variables..."
export GEMMA4_TEXT_ONLY_MODEL_PATH="$MODEL_MOUNT/text_only"
export GEMMA4_ASSISTANT_MODEL_PATH="$MODEL_MOUNT/assistant"
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_MOE_USE_DEEP_GEMM=0
echo "  GEMMA4_TEXT_ONLY_MODEL_PATH=$GEMMA4_TEXT_ONLY_MODEL_PATH"
echo "  GEMMA4_ASSISTANT_MODEL_PATH=$GEMMA4_ASSISTANT_MODEL_PATH"


# 5. Apply patches (if any)
echo "[Step 5/7] Applying Python patches to vLLM site-packages..."
VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])")
echo "  vLLM path: $VLLM_PATH"
if git apply --unsafe-paths -p1 --directory="$(dirname "$VLLM_PATH")" << 'PATCH_EOF'
diff --git a/vllm/envs.py b/vllm/envs.py
index 64b5ca2c4e..d2006e09e3 100755
--- a/vllm/envs.py
+++ b/vllm/envs.py
@@ -93,6 +93,9 @@ if TYPE_CHECKING:
     VLLM_FLOAT32_MATMUL_PRECISION: Literal["highest", "high", "medium"] = "highest"
     VLLM_BATCH_INVARIANT: bool = False
     VLLM_TRITON_USE_TD: bool | None = None
+    D2G_TILE: int = 32
+    D2G_STAGES: int = 2
+    D2G_WARPS: int = 0
     VLLM_GPU_SYNC_CHECK: Literal["warn", "error"] | None = None
     MAX_JOBS: str | None = None
     NVCC_THREADS: str | None = None
@@ -651,6 +654,11 @@ environment_variables: dict[str, Callable[[], Any]] = {
     "VLLM_GPU_SYNC_CHECK": env_with_choices(
         "VLLM_GPU_SYNC_CHECK", None, ["warn", "error"]
     ),
+    # Fork launch controls: tile affects the 2D path; stages/warps affect both
+    # 2D and 3D. Zero retains the upstream choice for each control.
+    "D2G_TILE": lambda: int(os.getenv("D2G_TILE", "32")),
+    "D2G_STAGES": lambda: int(os.getenv("D2G_STAGES", "2")),
+    "D2G_WARPS": lambda: int(os.getenv("D2G_WARPS", "0")),
     # Maximum number of compilation jobs to run in parallel.
     # By default this is the number of CPUs
     "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
diff --git a/vllm/v1/attention/ops/triton_unified_attention.py b/vllm/v1/attention/ops/triton_unified_attention.py
index b0231dd0c0..ab4313d6f5 100644
--- a/vllm/v1/attention/ops/triton_unified_attention.py
+++ b/vllm/v1/attention/ops/triton_unified_attention.py
@@ -1092,6 +1092,13 @@ def unified_attention(
     if launch_num_stages is not None:
         launch_kwargs["num_stages"] = launch_num_stages

+    if envs.D2G_TILE and not use_3d:
+        tile_size = min(envs.D2G_TILE, block_size) if use_td else envs.D2G_TILE
+    if envs.D2G_STAGES:
+        launch_kwargs["num_stages"] = envs.D2G_STAGES
+    if envs.D2G_WARPS:
+        launch_kwargs["num_warps"] = envs.D2G_WARPS
+
     kernel_unified_attention[grid](
         output_ptr=out,
         segm_output_ptr=segm_output_ptr,

PATCH_EOF
then echo "  Applied unified patch: attention/qwen35_d2g_v031"
else echo "[FATAL] git apply FAILED for unified patch attention/qwen35_d2g_v031"; exit 1; fi
if git apply --unsafe-paths -p1 --directory="$(dirname "$VLLM_PATH")" << 'PATCH_EOF'
diff --git a/vllm/v1/attention/backends/triton_attn.py b/vllm/v1/attention/backends/triton_attn.py
index 8a2fab4..9ef53e7 100644
--- a/vllm/v1/attention/backends/triton_attn.py
+++ b/vllm/v1/attention/backends/triton_attn.py
@@ -128,6 +128,11 @@ class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMet
                 CUDAGraphMode.FULL_DECODE_ONLY,
                 CUDAGraphMode.FULL,
             )
+            or (
+                self.vllm_config.speculative_config is not None
+                and self.vllm_config.compilation_config.cudagraph_mode
+                != CUDAGraphMode.NONE
+            )
         )
 
         # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
diff --git a/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py b/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
index 897e7af..23dba01 100644
--- a/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
+++ b/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
@@ -130,7 +130,16 @@ class AutoRegressiveSpeculator(DraftModelSpeculator):
         )

         # PIECEWISE cudagraphs are not supported for draft decodes.
-        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
+        draft_attn_backend = getattr(
+            self.speculative_config, "attention_backend", None
+        )
+        draft_attn_backend_name = getattr(
+            draft_attn_backend, "name", draft_attn_backend
+        )
+        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL or (
+            cudagraph_mode != CUDAGraphMode.NONE
+            and draft_attn_backend_name in (None, "TRITON_ATTN", "FLASH_ATTN")
+        ):
             cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
         else:
             cudagraph_mode = CUDAGraphMode.NONE
@@ -510,6 +516,15 @@ class AutoRegressiveSpeculator(DraftModelSpeculator):
         query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
         idx_mapping = self.idx_mapping[:num_reqs]
 
+        if (
+            not skip_attn
+            and batch_desc.num_reqs is not None
+            and batch_desc.num_reqs > num_reqs
+        ):
+            self.block_tables.gather_block_tables(
+                idx_mapping, num_reqs_padded=batch_desc.num_reqs
+            )
+
         attn_metadata = None
         slot_mappings_by_layer = None
         for step in range(1, self.num_speculative_steps):
@@ -560,6 +575,15 @@ class AutoRegressiveSpeculator(DraftModelSpeculator):
         query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
         idx_mapping = self.idx_mapping[:num_reqs]
 
+        if (
+            not skip_attn
+            and batch_desc.num_reqs is not None
+            and batch_desc.num_reqs > num_reqs
+        ):
+            self.block_tables.gather_block_tables(
+                idx_mapping, num_reqs_padded=batch_desc.num_reqs
+            )
+
         attn_metadata = None
         slot_mappings_by_layer = None
         if not skip_attn:

PATCH_EOF
then echo "  Applied unified patch: attention/vllm_v031_eagle_full_cudagraph"
else echo "[FATAL] git apply FAILED for unified patch attention/vllm_v031_eagle_full_cudagraph"; exit 1; fi
if git apply --unsafe-paths -p1 --directory="$(dirname "$VLLM_PATH")" << 'PATCH_EOF'
diff --git a/vllm/v1/worker/gpu/sample/gumbel.py b/vllm/v1/worker/gpu/sample/gumbel.py
index a34169b733..f37343ebfe 100644
--- a/vllm/v1/worker/gpu/sample/gumbel.py
+++ b/vllm/v1/worker/gpu/sample/gumbel.py
@@ -62,6 +62,56 @@ def apply_temperature(
     )
 
 
+@triton.jit
+def _temperature_upcast_kernel(
+    out_ptr,
+    out_stride,
+    logits_ptr,
+    logits_stride,
+    expanded_idx_mapping_ptr,
+    temperature_ptr,
+    vocab_size,
+    BLOCK_SIZE: tl.constexpr,
+):
+    token_idx = tl.program_id(0).to(tl.int64)
+    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
+    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
+
+    block_idx = tl.program_id(1)
+    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+    mask = block < vocab_size
+
+    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
+    logits = logits.to(tl.float32)
+    if temperature != 0.0 and temperature != 1.0:
+        logits = logits / temperature
+    tl.store(out_ptr + token_idx * out_stride + block, logits, mask=mask)
+
+
+def apply_temperature_upcast(
+    logits: torch.Tensor,
+    expanded_idx_mapping: torch.Tensor,
+    temperature: torch.Tensor,
+) -> torch.Tensor:
+    num_tokens, vocab_size = logits.shape
+    out = torch.empty_like(logits, dtype=torch.float32)
+    if num_tokens == 0:
+        return out
+    BLOCK_SIZE = 8192
+    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
+    _temperature_upcast_kernel[(num_tokens, num_blocks)](
+        out,
+        out.stride(0),
+        logits,
+        logits.stride(0),
+        expanded_idx_mapping,
+        temperature,
+        vocab_size,
+        BLOCK_SIZE=BLOCK_SIZE,
+    )
+    return out
+
+
 @triton.jit
 def tl_rand32(seed, offset, includes_zero: tl.constexpr):
     u = tl.rand(seed, offset)
diff --git a/vllm/v1/worker/gpu/sample/sampler.py b/vllm/v1/worker/gpu/sample/sampler.py
index 7ffc5bf493..0341f66848 100644
--- a/vllm/v1/worker/gpu/sample/sampler.py
+++ b/vllm/v1/worker/gpu/sample/sampler.py
@@ -83,6 +83,8 @@ class Sampler:
             TraceReplayState(req_states) if enable_trace_replay else None
         )
         self.needs_logits_processing = np.zeros(max_num_reqs, dtype=bool)
+        self.needs_non_temp_logits_processing = np.zeros(max_num_reqs, dtype=bool)
+        self.needs_custom_logits_processing = np.zeros(max_num_reqs, dtype=bool)
         self.num_speculative_tokens = num_speculative_tokens
         self.return_sampling_mask = return_sampling_mask
         self.use_flashinfer = (
@@ -95,12 +97,22 @@ class Sampler:
 
     def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
         needs_processing = self.sampling_states.add_request(req_idx, sampling_params)
-        needs_processing |= self.thinking_budget_state.add_request(
+        needs_custom = self.thinking_budget_state.add_request(
             req_idx, sampling_params
         )
         for processor in self.logits_processors:
-            needs_processing |= processor.add_request(req_idx, sampling_params)
-        self.needs_logits_processing[req_idx] = needs_processing
+            needs_custom |= processor.add_request(req_idx, sampling_params)
+        self.needs_logits_processing[req_idx] = needs_processing or needs_custom
+        self.needs_custom_logits_processing[req_idx] = needs_custom
+        has_temp = self.sampling_states.temperature.np[req_idx] != 0.0
+        has_min_p = has_temp and self.sampling_states.min_p.np[req_idx] != 0.0
+        has_top_k_or_top_p = has_temp and (
+            self.sampling_states.top_k.np[req_idx] != self.sampling_states.vocab_size
+            or self.sampling_states.top_p.np[req_idx] != 1.0
+        )
+        self.needs_non_temp_logits_processing[req_idx] = (
+            needs_custom or has_min_p or has_top_k_or_top_p
+        )
 
         self.logprob_token_ids_state.add_request(req_idx, sampling_params)
         if self.trace_replay_state is not None:
@@ -131,6 +143,16 @@ class Sampler:
         num_logprobs = max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0
         return num_logprobs, max_token_ids
 
+    def can_fuse_temperature(self, idx_mapping_np: np.ndarray) -> bool:
+        return not bool(
+            np.any(self.needs_non_temp_logits_processing[idx_mapping_np])
+        )
+
+    def can_fuse_temperature_upcast(self, idx_mapping_np: np.ndarray) -> bool:
+        return not bool(
+            np.any(self.needs_custom_logits_processing[idx_mapping_np])
+        )
+
     def __call__(
         self,
         logits: torch.Tensor,
@@ -238,31 +260,37 @@ class Sampler:
         if not np.any(self.needs_logits_processing[idx_mapping_np]):
             return logits
 
-        # Copy logits to a new FP32 tensor.
-        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
-
-        ctx = LogitsContext(
-            expanded_idx_mapping=expanded_idx_mapping,
-            idx_mapping=idx_mapping,
-            idx_mapping_np=idx_mapping_np,
-            expanded_local_pos=expanded_local_pos,
-            input_ids=input_ids,
-            pos=pos,
-            seq_lens_upper_bound_np=seq_lens_upper_bound_np,
-        )
+        if self.can_fuse_temperature_upcast(idx_mapping_np):
+            # Fuse BF16->FP32 upcast and temperature scaling into a single kernel pass.
+            logits = self.sampling_states.apply_temperature_upcast(
+                logits, expanded_idx_mapping, idx_mapping_np
+            )
+        else:
+            # Copy logits to a new FP32 tensor.
+            logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
+
+            ctx = LogitsContext(
+                expanded_idx_mapping=expanded_idx_mapping,
+                idx_mapping=idx_mapping,
+                idx_mapping_np=idx_mapping_np,
+                expanded_local_pos=expanded_local_pos,
+                input_ids=input_ids,
+                pos=pos,
+                seq_lens_upper_bound_np=seq_lens_upper_bound_np,
+            )
 
-        # Apply logits processors (native + any custom).
-        for processor in self.logits_processors:
-            logits = processor.apply(logits, ctx)
+            # Apply logits processors (native + any custom).
+            for processor in self.logits_processors:
+                logits = processor.apply(logits, ctx)
 
-        # Forcing runs last so no stage can overwrite the forced end marker
-        # or weaken it by scaling.
-        self.thinking_budget_state.apply(logits, ctx)
+            # Forcing runs last so no stage can overwrite the forced end marker
+            # or weaken it by scaling.
+            self.thinking_budget_state.apply(logits, ctx)
 
-        # Apply temperature in place.
-        self.sampling_states.apply_temperature(
-            logits, expanded_idx_mapping, idx_mapping_np
-        )
+            # Apply temperature in place.
+            self.sampling_states.apply_temperature(
+                logits, expanded_idx_mapping, idx_mapping_np
+            )
 
         # Apply min_p in place.
         self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)
@@ -287,6 +315,23 @@ class Sampler:
         seq_lens_upper_bound_np: np.ndarray,
         return_logprobs: bool = False,
     ) -> tuple[torch.Tensor, torch.Tensor]:
+        if (
+            self.can_fuse_temperature(idx_mapping_np)
+            and not (return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES)
+            and not self.return_sampling_mask
+        ):
+            sampled = gumbel_sample(
+                logits,
+                expanded_idx_mapping,
+                self.sampling_states.temperature.gpu,
+                self.sampling_states.seeds.gpu,
+                pos,
+                apply_temperature=True,
+                is_drafting=False,
+                use_fp64=self.use_fp64_gumbel,
+            )
+            return sampled, logits
+
         processed_logits = self.apply_sampling_params(
             logits,
             expanded_idx_mapping,
diff --git a/vllm/v1/worker/gpu/sample/states.py b/vllm/v1/worker/gpu/sample/states.py
index 47cf5de856..2966402268 100644
--- a/vllm/v1/worker/gpu/sample/states.py
+++ b/vllm/v1/worker/gpu/sample/states.py
@@ -6,7 +6,7 @@ import torch
 from vllm.sampling_params import SamplingParams
 from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
 from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
-from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
+from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, apply_temperature_upcast
 from vllm.v1.worker.gpu.sample.min_p import apply_min_p
 
 NO_LOGPROBS = -1
@@ -91,6 +91,20 @@ class SamplingStates:
 
         apply_temperature(logits, expanded_idx_mapping, self.temperature.gpu)
 
+    def apply_temperature_upcast(
+        self,
+        logits: torch.Tensor,
+        expanded_idx_mapping: torch.Tensor,
+        idx_mapping_np: np.ndarray,
+    ) -> torch.Tensor:
+        if logits.dtype == torch.float32:
+            out = logits.clone()
+            self.apply_temperature(out, expanded_idx_mapping, idx_mapping_np)
+            return out
+        return apply_temperature_upcast(
+            logits, expanded_idx_mapping, self.temperature.gpu
+        )
+
     def apply_min_p(
         self,
         logits: torch.Tensor,
diff --git a/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py b/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
index b96007b6c1..f5fa419f53 100644
--- a/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
+++ b/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
@@ -255,16 +255,28 @@ class RejectionSampler:
         seq_lens_upper_bound_np: np.ndarray,
         verify_draft_sampled: torch.Tensor,
     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
-        processed_logits = self.sampler.apply_sampling_params(
-            logits,
-            expanded_idx_mapping,
-            idx_mapping,
-            idx_mapping_np,
-            pos,
-            draft_sampled,
-            expanded_local_pos,
-            seq_lens_upper_bound_np,
+        needs_processed_logprobs = (
+            self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
+            and self.sampler.sampling_states.max_num_logprobs(idx_mapping_np)
+            != NO_LOGPROBS
+        )
+        fuse_temperature = (
+            self.sampler.can_fuse_temperature(idx_mapping_np)
+            and not needs_processed_logprobs
         )
+        if fuse_temperature:
+            processed_logits = logits
+        else:
+            processed_logits = self.sampler.apply_sampling_params(
+                logits,
+                expanded_idx_mapping,
+                idx_mapping,
+                idx_mapping_np,
+                pos,
+                draft_sampled,
+                expanded_local_pos,
+                seq_lens_upper_bound_np,
+            )
         sampled, num_sampled = rejection_sample(
             processed_logits,
             draft_logits,
@@ -280,6 +292,7 @@ class RejectionSampler:
             self.synthetic_conditional_rates,
             use_fp64=self.sampler.use_fp64_gumbel,
             use_block_verification=self.use_block_verification,
+            apply_temperature=fuse_temperature,
             **self._watermarking_kwargs(
                 draft_sampled, expanded_idx_mapping, expanded_local_pos
             ),
diff --git a/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py b/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
index b2efa6432e..3018d74ca3 100644
--- a/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
+++ b/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
@@ -59,8 +59,10 @@ def _compute_global_residual_mass(
     draft_token,
     logit_idx,
     vocab_num_blocks,
+    temp,
     PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
     HAS_DRAFT_LOGITS: tl.constexpr,
+    APPLY_TEMPERATURE: tl.constexpr = False,
 ):
     if HAS_DRAFT_LOGITS:
         blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
@@ -87,6 +89,8 @@ def _compute_global_residual_mass(
         target_logit = tl.load(
             target_logits_ptr + logit_idx * target_logits_stride + draft_token,
         ).to(tl.float32)
+        if APPLY_TEMPERATURE and temp != 1.0:
+            target_logit = target_logit / temp
         m_b = tl.exp(target_logit - target_lse)
         return prefix_joint_ratio * (1.0 - m_b)
 
@@ -144,12 +148,15 @@ def _compute_global_logprobs_and_logsumexp(
     vocab_num_blocks,
     PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
     HAS_DRAFT_LOGITS: tl.constexpr,
+    APPLY_TEMPERATURE: tl.constexpr = False,
 ):
     target_logit = tl.load(
         target_logits_ptr + logit_idx * target_logits_stride + token,
         mask=mask,
         other=float("-inf"),
     ).to(tl.float32)
+    if APPLY_TEMPERATURE and temp != 1.0:
+        target_logit = target_logit / temp
     target_lse = _compute_global_logsumexp(
         target_local_max_ptr,
         target_local_max_stride,
@@ -224,6 +231,7 @@ def _compute_local_logits_stats_kernel(
     num_speculative_steps,
     BLOCK_SIZE: tl.constexpr,
     HAS_DRAFT_LOGITS: tl.constexpr,
+    APPLY_TEMPERATURE: tl.constexpr = False,
 ):
     logit_idx = tl.program_id(0).to(tl.int64)
     draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
@@ -265,6 +273,8 @@ def _compute_local_logits_stats_kernel(
             mask=mask,
             other=float("-inf"),
         ).to(tl.float32)
+        if APPLY_TEMPERATURE and temp != 1.0:
+            target_logits = target_logits / temp
         target_max, target_sumexp = _compute_max_and_sumexp(target_logits)
         tl.store(
             target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
@@ -337,6 +347,7 @@ def _compute_cumulative_log_p_kernel(
     vocab_num_blocks,
     PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
     HAS_DRAFT_LOGITS: tl.constexpr,
+    APPLY_TEMPERATURE: tl.constexpr = False,
 ):
     req_idx = tl.program_id(0)
     req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
@@ -378,6 +389,7 @@ def _compute_cumulative_log_p_kernel(
                     vocab_num_blocks,
                     PADDED_VOCAB_NUM_BLOCKS,
                     HAS_DRAFT_LOGITS,
+                    APPLY_TEMPERATURE,
                 )
             )
             log_p = tl.minimum(log_p + (target_logprob - draft_logprob), 0.0)
@@ -423,6 +435,7 @@ def _compute_local_residual_mass_kernel(
     vocab_num_blocks,
     BLOCK_SIZE: tl.constexpr,
     PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
+    APPLY_TEMPERATURE: tl.constexpr = False,
 ):
     logit_idx = tl.program_id(0).to(tl.int64)
     draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
@@ -468,6 +481,7 @@ def _compute_local_residual_mass_kernel(
         vocab_num_blocks,
         PADDED_VOCAB_NUM_BLOCKS,
         True,  # HAS_DRAFT_LOGITS
+        APPLY_TEMPERATURE,
     )
 
     # Compute the residual mass: max(p_i * M_b(x|x_{<i}) - M_s(x|x_{<i}), 0)
@@ -538,6 +552,7 @@ def _rejection_kernel(
     HAS_DRAFT_LOGITS: tl.constexpr,
     SYNTHETIC_MODE: tl.constexpr,
     USE_BLOCK_VERIFICATION: tl.constexpr,
+    APPLY_TEMPERATURE: tl.constexpr = False,
 ):
     req_idx = tl.program_id(0)
     req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
@@ -616,8 +631,10 @@ def _rejection_kernel(
                         next_draft_token,
                         logit_idx + 1,
                         vocab_num_blocks,
+                        temp,
                         PADDED_VOCAB_NUM_BLOCKS,
                         HAS_DRAFT_LOGITS,
+                        APPLY_TEMPERATURE,
                     )
                     denom = residual_mass + 1.0 - prefix_joint_ratio
                     h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
@@ -651,6 +668,7 @@ def _rejection_kernel(
                         vocab_num_blocks,
                         PADDED_VOCAB_NUM_BLOCKS,
                         HAS_DRAFT_LOGITS,
+                        APPLY_TEMPERATURE,
                     )
                 )
                 if SYNTHETIC_MODE:
@@ -778,6 +796,7 @@ def _resample_kernel(
     CONTEXT_WIDTH: tl.constexpr,
     WATERMARK: tl.constexpr,
     DEDUPLICATE_CONTEXTS: tl.constexpr,
+    APPLY_TEMPERATURE: tl.constexpr = False,
 ):
     req_idx = tl.program_id(0)
     resample_idx = tl.load(rejected_step_ptr + req_idx)
@@ -808,6 +827,8 @@ def _resample_kernel(
         mask=mask,
         other=float("-inf"),
     ).to(tl.float32)
+    if APPLY_TEMPERATURE and temp != 0.0 and temp != 1.0:
+        target_logits = target_logits / temp
 
     # Compute the residual logits to resample the rejected token from.
     is_watermarked = False
@@ -1039,6 +1060,7 @@ def rejection_sample(
     watermarking: torch.Tensor | None = None,
     watermarking_skip_mask: torch.Tensor | None = None,
     watermark_key: int | None = None,
+    apply_temperature: bool = False,
 ) -> tuple[torch.Tensor, torch.Tensor]:
     assert target_logits.ndim == 2 and target_logits.stride(-1) == 1
     assert draft_logits is None or (
@@ -1133,6 +1155,7 @@ def rejection_sample(
         num_speculative_steps,
         BLOCK_SIZE=VOCAB_BLOCK_SIZE,
         HAS_DRAFT_LOGITS=has_draft_logits,
+        APPLY_TEMPERATURE=apply_temperature,
     )
 
     # Precompute the running joint ratio and residual mass for block
@@ -1168,6 +1191,7 @@ def rejection_sample(
             vocab_num_blocks,
             PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
             HAS_DRAFT_LOGITS=has_draft_logits,
+            APPLY_TEMPERATURE=apply_temperature,
             num_warps=1,
         )
 
@@ -1205,6 +1229,7 @@ def rejection_sample(
                 vocab_num_blocks,
                 BLOCK_SIZE=VOCAB_BLOCK_SIZE,
                 PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
+                APPLY_TEMPERATURE=apply_temperature,
             )
         else:
             local_residual_mass = None
@@ -1256,6 +1281,7 @@ def rejection_sample(
         HAS_DRAFT_LOGITS=has_draft_logits,
         SYNTHETIC_MODE=synthetic_conditional_rates is not None,
         USE_BLOCK_VERIFICATION=use_block_verification,
+        APPLY_TEMPERATURE=apply_temperature,
         num_warps=1,
     )
 
@@ -1305,6 +1331,7 @@ def rejection_sample(
         CONTEXT_WIDTH=context_width,
         WATERMARK=watermark,
         DEDUPLICATE_CONTEXTS=watermarking_skip_mask is not None,
+        APPLY_TEMPERATURE=apply_temperature,
     )
 
     # Insert the resampled tokens into the output sampled.

PATCH_EOF
then echo "  Applied unified patch: attention/vllm_v031_fast_spec_sample"
else echo "[FATAL] git apply FAILED for unified patch attention/vllm_v031_fast_spec_sample"; exit 1; fi
if git apply --unsafe-paths -p1 --directory="$(dirname "$VLLM_PATH")" << 'PATCH_EOF'
diff --git a/vllm/v1/sample/ops/topk_topp_triton.py b/vllm/v1/sample/ops/topk_topp_triton.py
--- a/vllm/v1/sample/ops/topk_topp_triton.py
+++ b/vllm/v1/sample/ops/topk_topp_triton.py
@@ -668,7 +668,7 @@
                 idx = tl.cast(p * 200, tl.int32)
                 idx = tl.maximum(0, tl.minimum(idx, 199))
                 sigma = tl.load(NORMAL_CDF_TO_SIGMA_TABLE + idx)
-                sigma = sigma + tl.abs(sigma) * -0.25
+                sigma = sigma + tl.abs(sigma) * -0.10
                 outlier_pivot = avg_logit + std_logit * sigma
 
                 outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
@@ -1673,6 +1673,40 @@
         topp_enabled and batch_size <= _SPLIT_MAX_BATCH and logits.device.type == "cuda"
     )
     if use_split and not topk_enabled:
+        buf_key = (logits.device, logits.dtype, vocab_size)
+        if buf_key not in _TRITON_BUFFER_CACHE:
+            # Pre-allocate the full num_sm buffer and pre-compile the large-batch
+            # monolithic _topk_topp_kernel during small-batch HTTP warmup so
+            # ramping concurrency past _SPLIT_MAX_BATCH (64) never triggers a
+            # mid-benchmark 107 MB allocation or Triton JIT compilation spike.
+            buffer = logits.new_empty((num_sm, vocab_size))
+            _TRITON_BUFFER_CACHE[buf_key] = buffer
+            tables = _TRITON_TABLE_CACHE.get(logits.device)
+            if tables is None:
+                with gpu_sync_allowed():
+                    normal_cdf_to_sigma_table = logits.new_tensor(
+                        _NORMAL_CDF_TO_SIGMA_TABLE
+                    )
+                    percentile_to_std_table = logits.new_tensor(
+                        _PERCENTILE_TO_STD_TABLE
+                    )
+                    _TRITON_TABLE_CACHE[logits.device] = (
+                        normal_cdf_to_sigma_table,
+                        percentile_to_std_table,
+                    )
+            else:
+                normal_cdf_to_sigma_table, percentile_to_std_table = tables
+            dummy_logits = logits[:1].clone()
+            _topk_topp(
+                dummy_logits,
+                buffer[:1],
+                percentile_to_std_table,
+                normal_cdf_to_sigma_table,
+                None,
+                p_ptr[:1],
+                mask_value,
+                num_sm,
+            )
         _apply_topp_split(logits, None, p_ptr, mask_value, num_sm)
         return logits
 
diff --git a/vllm/v1/worker/gpu/sample/gumbel.py b/vllm/v1/worker/gpu/sample/gumbel.py
--- a/vllm/v1/worker/gpu/sample/gumbel.py
+++ b/vllm/v1/worker/gpu/sample/gumbel.py
@@ -239,6 +239,11 @@
     if USE_FP64:
         logits = logits.to(tl.float64)
     if temp != 0.0:
+        # When top_p/top_k or grammar masking eliminates every token in this
+        # 1024-token block, skip the 4-round Murmur3 hash and log1p/log Gumbel
+        # transform completely.
+        if tl.max(logits, axis=0) == float("-inf"):
+            return float("-inf"), 0
         if IS_DRAFTING:
             pos = pos + _DRAFT_NOISE_SALT
         if USE_FP64:
diff --git a/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py b/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
--- a/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
+++ b/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
@@ -235,17 +235,30 @@
 ):
     logit_idx = tl.program_id(0).to(tl.int64)
     draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
-
-    if draft_step_idx >= num_speculative_steps:
-        # Bonus token. Max/argmax and summed exponentials are not needed.
-        return
-
     req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
     temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
 
     block_idx = tl.program_id(1)
     block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = block_offsets < vocab_size
+
+    if draft_step_idx >= num_speculative_steps:
+        # Bonus token does not need logsumexp, but storing its 8192-block max
+        # when temp != 0.0 lets _resample_kernel skip all -inf 8192-element
+        # chunks after top_p without reading target_logits again.
+        if temp != 0.0:
+            target_logits = tl.load(
+                target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
+                mask=mask,
+                other=float("-inf"),
+            ).to(tl.float32)
+            if APPLY_TEMPERATURE and temp != 1.0:
+                target_logits = target_logits / temp
+            tl.store(
+                target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
+                tl.max(target_logits, axis=0),
+            )
+        return
 
     if temp == 0.0:
         # Greedy sampling. Only the target max/argmax are needed.
@@ -456,6 +469,15 @@
         return
 
     block_idx = tl.program_id(1)
+    if (
+        tl.load(target_local_max_ptr + logit_idx * target_local_max_stride + block_idx)
+        == float("-inf")
+    ):
+        tl.store(
+            local_residual_mass_ptr + logit_idx * local_residual_mass_stride + block_idx,
+            0.0,
+        )
+        return
     block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = block_offsets < vocab_size
     target_log_probs, draft_log_probs, _, _ = _compute_global_logprobs_and_logsumexp(
@@ -756,6 +778,9 @@
     # [num_logits, V]
     target_logits_ptr,
     target_logits_stride,
+    # [num_logits, vocab_num_blocks]
+    target_local_max_ptr,
+    target_local_max_stride,
     # [num_reqs]
     target_rejected_logsumexp_ptr,
     # [max_num_reqs, num_speculative_steps, V]
@@ -820,6 +845,34 @@
     is_valid_rejected_draft = rejected_draft_token >= 0
 
     block_idx = tl.program_id(1)
+    can_skip_masked_block = not WATERMARK and (
+        is_bonus or HAS_DRAFT_LOGITS or not is_valid_rejected_draft
+    )
+    if can_skip_masked_block and temp != 0.0:
+        # Level-1 coarse skip: check the 8192-vocab block max already computed
+        # in _compute_local_logits_stats_kernel before reading 1024 target_logits
+        # or 1024 draft_logits from HBM.
+        parent_block_idx = block_idx // 8
+        parent_max = tl.load(
+            target_local_max_ptr
+            + resample_token_idx * target_local_max_stride
+            + parent_block_idx
+        )
+        if parent_max == float("-inf"):
+            tl.store(
+                resampled_local_argmax_ptr
+                + req_idx * resampled_local_argmax_stride
+                + block_idx,
+                block_idx * BLOCK_SIZE,
+            )
+            tl.store(
+                resampled_local_max_ptr
+                + req_idx * resampled_local_max_stride
+                + block_idx,
+                float("-inf"),
+            )
+            return
+
     block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = block < vocab_size
     target_logits = tl.load(
@@ -829,6 +882,24 @@
     ).to(tl.float32)
     if APPLY_TEMPERATURE and temp != 0.0 and temp != 1.0:
         target_logits = target_logits / temp
+
+    if can_skip_masked_block and tl.max(target_logits, axis=0) == float("-inf"):
+        # Level-2 fine skip: within a surviving 8192-token block, skip 1024-token
+        # sub-blocks that contain no top_p survivors before loading draft_logits
+        # or computing exp/log1p/Murmur3 Gumbel noise.
+        tl.store(
+            resampled_local_argmax_ptr
+            + req_idx * resampled_local_argmax_stride
+            + block_idx,
+            block_idx * BLOCK_SIZE,
+        )
+        tl.store(
+            resampled_local_max_ptr
+            + req_idx * resampled_local_max_stride
+            + block_idx,
+            float("-inf"),
+        )
+        return
 
     # Compute the residual logits to resample the rejected token from.
     is_watermarked = False
@@ -1304,6 +1375,8 @@
         resampled_local_max.stride(0),
         target_logits,
         target_logits.stride(0),
+        target_local_max,
+        target_local_max.stride(0),
         target_rejected_logsumexp,
         draft_logits,
         draft_logits_stride_0,
@@ -1348,5 +1421,6 @@
         expanded_idx_mapping,
         temperature,
         PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
+        num_warps=1,
     )
     return sampled, num_sampled

PATCH_EOF
then echo "  Applied unified patch: attention/vllm_v031_compact_topk_resample"
else echo "[FATAL] git apply FAILED for unified patch attention/vllm_v031_compact_topk_resample"; exit 1; fi
if git apply --unsafe-paths -p1 --directory="$(dirname "$VLLM_PATH")" << 'PATCH_EOF'
diff --git a/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py b/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
index 9d5bcc8..a6997a7 100644
--- a/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
+++ b/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
@@ -914,7 +914,7 @@ class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
             and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
         )
         if use_fused_gdn_decode:
-            core_attn_out = torch.zeros(
+            core_attn_out = torch.empty(
                 (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                 dtype=hidden_states.dtype,
                 device=hidden_states.device,
@@ -1163,6 +1163,40 @@ class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
                 T,
                 self.prefix,
             )
+            try:
+                conv_dim = qkv_or_qkvz.shape[-1] - v_dim
+                conv_w = self.conv1d.weight.view(
+                    self.conv1d.weight.size(0), self.conv1d.weight.size(2)
+                )
+                conv_state_dtype, _ = self.get_state_dtype()
+                dummy_conv_state = torch.zeros(
+                    2,
+                    conv_dim,
+                    self.conv_kernel_size - 1,
+                    device=device,
+                    dtype=conv_state_dtype,
+                )
+                dummy_conv_indices = torch.tensor([0], device=device, dtype=torch.int32)
+                dummy_acc = torch.tensor([1], device=device, dtype=torch.int32)
+                dummy_qsl = torch.tensor([0, 4], device=device, dtype=torch.int32)
+                causal_conv1d_update(
+                    dummy_mixed_qkv[:4],
+                    dummy_conv_state,
+                    conv_w,
+                    self.conv1d.bias,
+                    self.activation,
+                    conv_state_indices=dummy_conv_indices,
+                    num_accepted_tokens=dummy_acc,
+                    query_start_loc=dummy_qsl,
+                    max_query_len=4,
+                    validate_data=False,
+                )
+                del dummy_conv_state, dummy_conv_indices, dummy_acc, dummy_qsl
+            except Exception:
+                logger.debug(
+                    "GDN causal_conv1d_update warmup skipped for layer %s",
+                    self.prefix,
+                )
         finally:
             del (
                 dummy_mixed_qkv,

PATCH_EOF
then echo "  Applied unified patch: attention/vllm_v031_gdn_minimal"
else echo "[FATAL] git apply FAILED for unified patch attention/vllm_v031_gdn_minimal"; exit 1; fi
echo "  0 file(s) and 5 unified patch(es) applied"

# 6. Prepare results directory
echo "[Step 6/7] Preparing results directory..."
RESULTS_DIR="$AML_OUTPUTS_RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
echo "  Results will be saved to: $RESULTS_DIR/"

# 7. Run experiments
echo "[Step 7/7] Running 1 experiment(s)..."

echo ""
echo "============================================================"
echo "  [20261118_114824] Experiment 1/1: auto | CG | MTP-k3 | mns=128 | mnbt=16384 | mml=25600 | no-prefix-cache | async-sched"
echo "============================================================"
echo "  Start time: $(date)"
echo "  Prompts: 2000, Reps: 1"
PARITY_FAILED=0

# Set/unset env vars for this experiment
unset VLLM_HUMMING_MOE_GEMM_TYPE 2>/dev/null || true
unset VLLM_HUMMING_USE_F16_ACCUM 2>/dev/null || true
unset VLLM_TEST_FORCE_FP8_MARLIN 2>/dev/null || true
export D2G_STAGES="${PARAM_D2G_STAGES:-2}"
export D2G_TILE="${PARAM_D2G_TILE:-64}"
export D2G_WARPS="${PARAM_D2G_WARPS:-8}"
export GEMMA4_ASSISTANT_MODEL_PATH="$AZURE_ML_INPUT_datastore/shares/users/zxy/autobench/data/qwen35-posted-reproduction-20261023-step6000/checkpoint"
export GEMMA4_TEXT_ONLY_MODEL_PATH="$AZURE_ML_INPUT_datastore/shares/users/zxy/autobench/data/qwen35-posted-reproduction-20261023-step6000/checkpoint"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

python - << 'CFGEOF'
import json, os
spec_tokens = int(os.environ.get("PARAM_SPEC_TOKENS", "3"))
cfg = {
  "_agentic_http_online": True,
  "served_model_name": "maiprofile-qwen35-sft",
  "quantization": "auto",
  "kv_cache_dtype": "auto",
  "gpu_memory_utilization": float(os.environ.get("PARAM_GPU_MEM_UTIL", "0.9")),
  "spec_tokens": spec_tokens,
  "attention_backend": "FLASHINFER",
  "mamba_ssm_cache_dtype": "bfloat16",
  "language_model_only": True,
  "enable_thinking": False,
  "reasoning_parser": "qwen3",
  "dtype": "bfloat16",
  "async_scheduling": True,
  "hf_overrides": {
    "architectures": [
      "Qwen3_5ForConditionalGeneration"
    ]
  },
  "max_tokens": 8192,
  "enforce_eager": False,
  "max_num_seqs": int(os.environ.get("PARAM_MAX_NUM_SEQS", "128")),
  "max_num_batched_tokens": int(os.environ.get("PARAM_MAX_NUM_BATCHED_TOKENS", "16384")),
  "max_model_len": 25600,
  "enable_chunked_prefill": True,
  "enable_prefix_caching": False,
  "_agentic_override_temperature": True,
  "_agentic_override_top_p": True,
  "temperature": float(os.environ.get("PARAM_TEMPERATURE", "0.6")),
  "top_p": float(os.environ.get("PARAM_TOP_P", "0.8")),
  "disable_uvicorn_access_log": True,
  "http_warmup_requests": 8,
  "http_warmup_max_tokens": 16,
}
if spec_tokens > 0:
    cfg["_speculative_config_partial"] = {
        "method": "mtp",
        "num_speculative_tokens": spec_tokens,
        "attention_backend": "TRITON_ATTN"
    }
with open("/tmp/config_1.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
CFGEOF

# Inject runtime spec model path into speculative_config (if partial spec config present)
python - << 'PYEOF'
import json, os, shutil
cfg_path = "/tmp/config_1.json"
with open(cfg_path) as f:
    cfg = json.load(f)
partial = cfg.pop("_speculative_config_partial", None)
if partial:
    draft_hf_repo = cfg.pop("_draft_hf_repo", None) or os.environ.get("AGENTIC_DRAFT_HF_REPO", "")
    draft_model_subdir = cfg.pop("_draft_model_subdir", None)
    if draft_hf_repo:
        local_draft_dir = os.path.join("/tmp/hf_drafts", draft_hf_repo.replace("/", "__"))
        if not os.path.isdir(local_draft_dir) or not os.path.isfile(os.path.join(local_draft_dir, "config.json")):
            print(f"  Staging HuggingFace draft model {draft_hf_repo} -> {local_draft_dir} ...", flush=True)
            old_hf_off = os.environ.pop("HF_HUB_OFFLINE", None)
            old_tr_off = os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=draft_hf_repo, local_dir=local_draft_dir)
            finally:
                if old_hf_off is not None:
                    os.environ["HF_HUB_OFFLINE"] = old_hf_off
                if old_tr_off is not None:
                    os.environ["TRANSFORMERS_OFFLINE"] = old_tr_off
        draft_model = local_draft_dir
        if cfg.pop("_draft_force_full_attention", False):
            dcfg_p = os.path.join(draft_model, "config.json")
            if os.path.isfile(dcfg_p):
                with open(dcfg_p, "r", encoding="utf-8") as df:
                    dcfg = json.load(df)
                dcfg["layer_types"] = ["full_attention"] * int(dcfg.get("num_hidden_layers", len(dcfg.get("layer_types", [2]))))
                dcfg["sliding_window"] = None
                dcfg["use_sliding_window"] = False
                with open(dcfg_p, "w", encoding="utf-8") as df:
                    json.dump(dcfg, df, indent=2)
                print(f"  Patched draft config.json -> full_attention (collapses 34 KV groups -> 4): {dcfg_p}", flush=True)
    elif draft_model_subdir:
        model_store = os.environ.get("AZURE_ML_INPUT_model_store", "")
        draft_model = os.path.join(model_store, draft_model_subdir)
        if not os.path.isdir(draft_model):
            raise RuntimeError(f"Draft model directory is unavailable: {draft_model}")
    else:
        draft_model = os.environ.get("GEMMA4_ASSISTANT_MODEL_PATH", "")
    partial["model"] = draft_model
    cfg["speculative_config"] = partial
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  speculative_config injected: method={partial['method']}, model={draft_model}")
PYEOF

# Run experiment (validated run_experiment.py from vllm-msn)
# vLLM treats TORCH_COMPILE_DISABLE as presence-based, so remove it from the
# benchmark process environment when this YAML explicitly enables compilation.
(
    set -o pipefail
    $RUNNER_COMMAND --config /tmp/config_1.json --prompts ${PARAM_PROMPTS:-2000} --reps 1         2>&1 | tee "/tmp/run_1.log"
)
EXP_EXIT=$?
if [ $EXP_EXIT -eq 0 ]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[WARN] Experiment 1 FAILED (exit code: $EXP_EXIT)"
fi

if [ $EXP_EXIT -eq 0 ] && [ -n "" ]; then
    python - <<'PYEOF'
import json
from pathlib import Path

reference_path = Path("")
candidate_path = Path("")
tolerance = float("0.98")
abort_on_fail = int("1")
if not reference_path.is_file() or not candidate_path.is_file():
    raise RuntimeError("Parity output is missing")

reference = json.loads(reference_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
if len(reference) != len(candidate):
    raise RuntimeError("Parity: completion count mismatch")

# Writer (aml.py) emits one completion per prompt, each completion a full token list:
#   [[completion.token_ids] for each prompt]
# Greedy decode is chaotic: a single logit-boundary token flip (FP8/bf16 tensor-core
# reduction variance on A100, proven in self-probe) reroutes the entire tail, so
# whole-sequence equality would score a 1023/1024-agreeing pair as 0. Use the
# tolerant per-token metric Chen approved: per completion, score the fraction of
# the shared prefix agreed up to the first divergence (mismatches only near
# logit ties are tolerated), then average across completions. A 1023/1024 pair
# scores 0.999, while a behavior-changing patch that diverges early scores low.
agree_fracs = []
diff_positions = []
for i, (ref_comp, cand_comp) in enumerate(zip(reference, candidate)):
    if not isinstance(ref_comp, list) or not isinstance(cand_comp, list):
        raise RuntimeError(f"Parity: prompt {i} is not a list of completions")
    if len(ref_comp) != len(cand_comp):
        raise RuntimeError(f"Parity: completion {i} count mismatch")
    for j, (rt, ct) in enumerate(zip(ref_comp, cand_comp)):
        # Records may be dicts of token_ids+text (new) or bare token lists (legacy).
        # bare token lists (legacy). Normalize both to a token list for the
        # length-tolerant greedy compare.
        def _tokens(rec):
            if isinstance(rec, dict):
                return rec.get("token_ids") or []
            return rec if isinstance(rec, list) else []
        rt = _tokens(rt)
        ct = _tokens(ct)
        # Length-tolerant greedy: a single logit-boundary flip can reroute the
        # tail so one run hits EOS earlier/later, making lengths differ (e.g.
        # 1209 vs 1237). That is exactly the tolerated numerics variance, not a
        # quality regression, so compare the common prefix up to the shorter
        # length and score against that shared length.
        common = min(len(rt), len(ct))
        d = 0
        while d < common and rt[d] == ct[d]:
            d += 1
        agree_fracs.append(d / common if common else 1.0)
        if d < common:
            diff_positions.append((i, j, d))

total = len(agree_fracs)
identical = sum(1 for f in agree_fracs if f == 1.0)
agree_frac = sum(agree_fracs) / total if total else 1.0

report = {
    "completions_compared": total,
    "fully_identical_completions": identical,
    "mean_agreement": round(agree_frac, 6),
    "agree_fraction": round(agree_frac, 6),
    "tolerance": tolerance,
    "passed": agree_frac >= tolerance,
    "diff_positions": diff_positions[:100],
}
report_path = Path("" + ".report.json")
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"Tolerant parity: mean_agree={agree_frac:.4f} (tolerance={tolerance}) "
      f"diffs={total - identical} total={total}", flush=True)
if agree_frac < tolerance:
    if abort_on_fail:
        raise RuntimeError(f"Tolerant parity check failed: agree={agree_frac:.4f} < {tolerance}")
    print(f"Diagnostic parity: agree={agree_frac:.4f} below tolerance {tolerance}; "
          "NOT aborting (diagnostic mode)", flush=True)
else:
    print("Tolerant parity check passed", flush=True)
PYEOF
    if [ $? -ne 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT - 1))
        FAIL_COUNT=$((FAIL_COUNT + 1))
        EXP_EXIT=1
        PARITY_FAILED=1
    fi
fi

echo "  Experiment 1 finished at $(date)"


# Persist deterministic-parity dumps + comparison report (with diff_positions)
for _pf in     ""     ".report.json"     ""; do
    if [ -n "$_pf" ] && [ -f "$_pf" ]; then
        _sb=$(basename "$_pf")
        cp "$_pf" "$RESULTS_DIR/parity__1__${_sb}" 2>/dev/null || true
    fi
done


# Copy results
if [ -d run_results ]; then
    for f in run_results/*.json; do
        if [ -f "$f" ]; then
            if [ "$(basename "$f")" = "parity_sample.json" ]; then
                cp "$f" "$RESULTS_DIR/parity_sample_1.json" 2>/dev/null || true
                cp "$f" "$AML_OUTPUTS_RESULTS_DIR/parity_sample_1.json" 2>/dev/null || true
            else
                cp "$f" "$RESULTS_DIR/result_1_$(basename "$f")" 2>/dev/null || true
                cp "$f" "$AML_OUTPUTS_RESULTS_DIR/result_1_$(basename "$f")" 2>/dev/null || true
            fi
        fi
    done
    if [ -f "run_results/http_server.log" ]; then
        cp "run_results/http_server.log" "$RESULTS_DIR/http_server_1.log" 2>/dev/null || true
        cp "run_results/http_server.log" "$AML_OUTPUTS_RESULTS_DIR/http_server_1.log" 2>/dev/null || true
    fi
    rm -rf run_results
fi
if [ -f "/tmp/run_1.log" ]; then
    cp "/tmp/run_1.log" "$RESULTS_DIR/" 2>/dev/null || true
    cp "/tmp/run_1.log" "$AML_OUTPUTS_RESULTS_DIR/" 2>/dev/null || true
fi
# Copy vLLM-native Torch profiler outputs when this experiment enabled them.
if [ -d "/tmp/vllm_torch_profile" ]; then
    mkdir -p "$RESULTS_DIR/vllm_torch_profile_1"
    cp -r /tmp/vllm_torch_profile/. "$RESULTS_DIR/vllm_torch_profile_1/" 2>/dev/null || true
fi
if [ "$PARITY_FAILED" -eq 1 ]; then
    echo "ERROR: parity gate failed; aborting remaining experiments"
    exit 1
fi


# Save aggregated results
if [ -f results.tsv ]; then
    cp results.tsv "$RESULTS_DIR/results.tsv"
fi

echo ""
echo "=== All experiments done ==="
echo "Results: $SUCCESS_COUNT succeeded, $FAIL_COUNT failed out of 1"

if [ "$SUCCESS_COUNT" -eq 0 ]; then
    echo "ERROR: All 1 experiments failed."
    exit 1
fi
