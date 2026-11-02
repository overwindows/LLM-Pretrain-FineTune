# SFT — Execution Process (layer1-delta, Thinking 50K, GPT-4o teacher)

## Qwen3.5 text-task compatibility (separate from the historical run below)

The dense `Qwen/Qwen3.5-4B` checkpoint uses the full
`Qwen3_5ForConditionalGeneration` class and `qwen3_5` config with nested text
and vision configs. The trainer now selects that concrete native class,
freezes its unused vision tower for text-only DDP, disables caching, and retains
the full model configuration/weights on save. Other existing model families
still use `AutoModelForCausalLM`. Some modern Transformers releases also map
Qwen3.5 through that Auto class; the problem is **not** universal Auto-class
incompatibility. The old Transformers 4.51.3 setup is incompatible with Qwen3.5.
Text SFT accepts string message content only; it does not process images/videos.

Use an isolated Python **3.10+** environment rather than altering the historical
SFT/RL/vLLM installation:

```bash
# PIP_CONSTRAINT may point at an approved exact CUDA/torch stack lock.
# Resolve and validate that GPU lock before any actual training submission.
bash scripts/setup_env.sh --profile qwen35 --python /path/to/python3 \
  --env-dir /new/path/qwen35-sft
```

This profile refuses existing environments, uses
`env/qwen35-sft-requirements.txt`, checks dependency consistency, and records
`resolved-requirements.txt` inside the environment. It never applies Python
version-gate patches. The default `legacy` profile remains unchanged.
The Qwen3.5 constraints pin Transformers **5.19.0** but do **not** claim a
validated CUDA/driver/DeepSpeed stack; inherited `PIP_CONSTRAINT` is respected.
The historical `env/pipeline-rl-pip-freeze.txt` is not a Qwen3.5 runtime lock.

CPU regression evidence: Python 3.12.10, torch 2.14.1 (CUDA unavailable),
Transformers 5.19.0, accelerate 1.15.0, datasets 5.1.0, tokenizers 0.23.2,
safetensors 0.8.0, PyYAML 6.0.3. Tests include a tiny randomly initialized
Qwen3.5 model with both linear and full attention, an actual optimizer update,
full HF save/reload and equal reloaded logits. They do not download model weights.
An additional local-cache-only tokenizer test uses official HF revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; it skips if metadata is absent.
The existing shared CPU host has unrelated OpenTelemetry dependency conflicts
reported by `pip check`; it is not a clean environment lock. The new Linux venv
bootstrap itself still requires runtime validation.

Render `sft/configs/sft/qwen35_text.yaml.tmpl` to a YAML with real paths. For
an orchestration-approved bounded GPU smoke, use a fresh output directory,
`max_seq_len: 1024`, batch/accumulation 1, no DeepSpeed and no resume:

```bash
unset PYTHONPATH PYTHONUSERBASE
export PYTHONNOUSERSITE=1
/new/path/qwen35-sft/bin/python sft/sft_train.py \
  --config /path/to/rendered-smoke.yaml --max-steps 3 --no-wandb
```

`base_model` should be the fully staged checkpoint directory with immutable
source provenance. For direct Hub loading, optional YAML `model_revision`
is passed to config, tokenizer, and weights. Always set
`enable_thinking: false` for the current nonreasoning gold targets and use the
same chat-template flag for evaluation/inference. Qwen3.5 requires an explicit
boolean setting. A generation prefix that does not exactly match the full
training rendering is an error, not a guessed loss boundary.

Assistant-only labels are computed **before** truncation, then shifted with the
left-truncated input. Targets that cannot fit intact together with a preceding
context token are skipped with a warning. Think masking never falls back to
training a span it was asked to mask. Token accuracy now uses causal shifting.
The warmup ratio is adapted to the installed Trainer API (`warmup_steps` ratios
in modern releases, `warmup_ratio` in legacy releases); modern ratios must be
less than 1 to avoid treating 1 as one absolute step.

Task weighting is a **preparation contract**: the caller materializes positive
integer frequency replication (currently HPF 2, GEM 1, MAI Profile 1) and removes
`sample_weight`. The trainer preserves those repeated rows with uniform
sampling; it does not also apply a weighted loss. Any remaining non-unit,
nonfinite or invalid row weight fails before dataset column projection, even
outside a debug subsample. Quality stays evaluation-only under the caller's
split selection policy. Malformed JSON and unsafe template boundaries fail
explicitly rather than silently reducing the corpus.

Run CPU regressions with:

```bash
python -m unittest discover -s sft/tests -v
```

The final HF model and tokenizer are saved at `output_dir`, including the
tokenizer chat template. Final `trainer_state.json` records step/log history;
`training_summary.json` records finite training loss, concrete model class/type,
thinking policy and source revision. Nonfinite training loss fails before final
export. Load the export using
`Qwen3_5ForConditionalGeneration.from_pretrained(output_dir)` and
`AutoTokenizer.from_pretrained(output_dir)` for **text-only** inference.
Image/video processing is outside this trainer's contract; it does not export
an `AutoProcessor`. Serving also needs an architecture-compatible runtime and
the same nonthinking flag. The historical vLLM 0.8.5 wrapper/shim is **not**
proof of Qwen3.5 support; do not mix it into this training environment or mark
GPU/serving readiness true from these CPU tests.

---

### Opt-in 8192-token ZeRO-3 bounded validation

Bounded validation supports the original **one-rank** mode and an opt-in
**eight-A100 jointly distributed** mode, not full training readiness. Retain
the tested native-HF stack and install the exact candidate
`deepspeed==0.16.9` after torch is installed, with `DS_BUILD_OPS=0` and
`--no-build-isolation`. Keep the root job's exact `PIP_CONSTRAINT`; run `pip check`
and persist the resolved freeze before model loading. No DeepSpeed package is
added to the default legacy or native-HF environment.

Candidate evidence (not proof of this model's GPU runtime compatibility):
[DeepSpeed 0.16.9's upstream CUDA CI](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.9/.github/workflows/nv-torch-latest-v100.yml)
explicitly tests torch 2.6 / CUDA 12.4. Its
[package requirements](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.9/requirements/requirements.txt)
allow torch 2.6 and pydantic >=2. The PyPI sdist SHA256 is
`88dc08986cd321047c37c2d0f1edd6faac498a1b52eb486d559164362c6b9011`.
The pinned source exposes `DeepSpeedEngine.zero_optimization_stage`,
`set_gradient_accumulation_boundary`, batch-size accessors and `global_steps`,
which are used by Accelerate 1.15 and the runtime proof. Its checkpoint loader
already specifies `weights_only=False`; no global torch-load patch is applied
to this version. This session's CPU/mock tests do not establish Linux dependency
resolution, CUDA kernels or 8192-token memory safety.

`TrainingArguments` is now constructed **before** `from_pretrained`, retaining
the HF DeepSpeed config weak reference used for `deepspeed.zero.Init`.
HF 5.19 also resolves automatic ZeRO buckets from Qwen3.5's nested
`text_config.hidden_size`. The full conditional-generation model is preserved,
with its unused visual tower frozen before the optimizer is constructed.

Use the orchestration-owned stage-3 JSON (BF16 and batch sizes automatic,
gather 16-bit weights on save, no offload/optimizer/scheduler blocks). Add these
keys to the rendered trainer YAML:

```yaml
max_seq_len: 8192
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
optim: adamw_torch
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1.0e-8
learning_rate: 5.0e-6
weight_decay: 0.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
seed: 42
enable_thinking: false
deepspeed_config: /path/to/zero3_bf16.json
resume_from_checkpoint: null
validation_fixed_order: true
bounded_validation:
  expected_steps: 8
  warmup_steps: 2
  expected_zero_stage: 3
  expected_world_size: 8  # 1 remains the default
```

After the orchestration owner verifies an actual eight-A100 allocation,
launch with `torchrun --standalone --nnodes=1 --nproc_per_node=8 sft/sft_train.py
--config <yaml> --max-steps 8 --no-wandb` from the isolated environment.
Use `expected_world_size: 1` and `--nproc_per_node=1` for the original mode.
The exact step cap must match `expected_steps` (at most 10); no debug/smoke flags
or resume are allowed. The outer job owns the GPU count and hard timeout.
`validation_fixed_order` is rejected outside bounded validation and selects
`SequentialSampler`, preserving the caller's repeated task-frequency rows and
planned near-cap coverage. Default training still uses the ordinary Trainer
and sampler.

In eight-rank mode, the optimizer is explicitly `adamw_torch`, and the actual
engine must report `AdamW`. Muon is not implemented on this fixed stack:
torch 2.6 does not provide it, and DeepSpeed 0.16.9 ZeRO-3 passes flattened
partition buffers to the underlying optimizer, not the complete matrices
required by the verified Muon implementation. An untested-optimizer switch
does not repair that mathematical incompatibility.
`optim`, `adam_beta1`, `adam_beta2`, and `adam_epsilon` are optional YAML
pass-throughs to TrainingArguments; when omitted they leave legacy defaults
unchanged (except the explicit eight-rank optimizer choice). Invalid or nonfinite
beta/epsilon settings and any non-`adamw_torch` eight-rank choice are rejected.

Accelerate shards the sequential batch sampler; do not install a second
distributed sampler. With batch 1 and accumulation 4, rank `r` consumes indices
`step*32 + micro*8 + r` (zero-based step/micro). Exactly 256 usable materialized
rows and no tokenization skips are required for eight fixed-order updates.
HPF's two copies remain distinct indices, not loss weights. Every microbatch's
actual source indices are collectively checked **before forward**, and each
update must cover its exact contiguous global group once. This proves
materialized row ordering, not distinct source users or source-content
uniqueness; those remain data-audit responsibilities.

`bounded_validation.json` is atomically updated at training start and after
each optimizer update, and embedded in the final `training_summary.json`.
It records actual engine class, ZeRO stage, initialized distributed world size,
batch/accumulation settings, device/package versions, and pre-load ZeRO-3
configuration/partitioned-parameter proof. Inactive/mismatched DeepSpeed,
skipped updates, partial accumulation groups, nonfinite microbatch losses or
an incorrect final step count fail explicitly. A failure during training leaves
a failed-status report rather than success-shaped evidence.
Runtime proof also requires `bf16_enabled: true` from the actual engine, records
its optimizer wrapper class and optimizer name, and snapshots
`resolved_hf_deepspeed_config` from the public HF/Accelerate integration after
automatic bucket/batch resolution. That snapshot represents the finalized
configuration passed to DeepSpeed, not all of DeepSpeed's internal defaults.

The resolved HF config includes Accelerate 1.15's deliberate
`steps_per_print = float("inf")` sentinel to disable DeepSpeed stdout logging
([upstream `DeepSpeedPlugin.__post_init__`](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py)).
It shares the `hf_ds_config.config` dictionary; this is not a nonfinite loss
or a source-JSON setting. DeepSpeed 0.16.9 accepts this sentinel in its actual
CPU configuration constructor. A strict JSON snapshot previously failed here,
before runtime proof persistence or any optimizer update.

Runtime proof now uses a narrow, lossless JSON representation **only** at
`$.steps_per_print`: a positive float infinity becomes
`{"__deepspeed_config_float__": "+Infinity"}`. Decode that exact tag at that
exact path as positive float infinity, not a string/zero/null. Finite values
and all other fields retain their existing JSON representation. The raw
HF/engine configuration is never mutated. Any negative infinity, NaN, or
infinity elsewhere (including nested buckets) is rejected with its path,
as is the reserved `__deepspeed_config_float__` key anywhere in raw config,
so a user-provided object cannot impersonate the sentinel.

Every runtime/rank proof also records:
`resolved_hf_deepspeed_config_encoding =
"accelerate-steps-per-print-positive-infinity-v1"` and
`resolved_hf_deepspeed_config_sha256`. The hash is SHA256 over the **encoded**
config dictionary serialized with Python
`json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False).encode("ascii")`, not a hash of the descriptor or full proof.
Verifiers must validate the descriptor/tag path and all remaining finite
config guards, recompute this hash, and compare both snapshot and hash across
ranks. The descriptor is included even for fully finite configs. Ordinary
losses, timings, optimizer hyperparameters and report persistence remain
strictly finite; this encoding is never applied to those measurements.
Regressions reproduce the original failure using the actual Accelerate
plugin and exercise the proof/persistence path imported from the exact
committed Git archive. The optional real-DeepSpeed test checks only its CPU
config constructor, not a distributed engine or GPU compatibility.

Eight-rank reports use `schema_version: 2`; one-rank reports retain version 1.
Every rank writes only `bounded_validation.rankN.json`; only rank zero also
writes the aggregate filename and final training summary. `runtime.ranks`
contains all eight rank proofs, ordered by global rank: actual rank/world size,
launcher local rank, hostname, CUDA index/visible-device count/physical UUID,
device name/memory, backend, engine and optimizer identity, BF16, resolved
configuration and package versions. Eight distinct physical UUIDs, A100 names,
unique host/local-rank pairs, NCCL and consistent runtime settings are mandatory.
Identified MIG devices are rejected.
`zero3_initialization_ranks` preserves every rank's pre-load configuration and
partitioned-parameter proof. `collective_proof` records an actual CUDA NCCL SUM
of ranks plus one, with `actual == expected == 36`, before any optimizer update.
Mocked tests cannot supply this runtime evidence.

`runtime.optimizer_class` identifies the actual underlying `engine.basic_optimizer`,
which must be a torch AdamW instance. `optimizer_group_proof` captures groups
at `Trainer.create_optimizer`, **before** ZeRO replaces their original
parameters with flattened partition buffers. It requires exact, duplicate-free
identity coverage of all and only trainable parameters (tied weights counted
once; frozen vision excluded), and reports tensor/full-element counts,
group count, LR, weight decay, betas and epsilon without parameter values.
Full element counts use ZeRO's `ds_numel` metadata when parameters are sharded.
`underlying_optimizer_groups` separately records actual runtime hyperparameters
after scheduler preparation; warmup can make its current `lr` zero while the
pre-wrap proof retains the configured base LR. All ranks must agree on both
snapshots. The CPU regression exercises a real tiny hybrid Qwen3.5 AdamW
update and optimizer-state reload, not a distributed ZeRO update.

Each optimizer update has CUDA synchronization at both ends, peak allocated and
reserved bytes reset at its start, and actual microbatch input lengths,
attended/padded tokens, causally supervised tokens and engine update count.
Microbatch losses are the values returned by Trainer (not an independently
normalized per-example metric). Preparation separately counts kept/skipped rows,
skip reasons and valid left-context truncation. Dataset diagnostics do not
include per-example content. Warmup updates are excluded from the aggregate
attended-token, supervised-token and example throughput. These update intervals
exclude dataset preparation, batch prefetch, logging, evaluation and export;
`training_wall_seconds` additionally includes inter-update overhead. Do not
extrapolate a full-run cost as though those excluded stages or sufficient
evaluation had been measured. GPU validation and long-generation evaluation
remain the orchestration session's responsibility.

For version 2, each `steps` entry contains explicit per-rank `ranks` evidence
and `row_indices`. Aggregate attended/supervised/padded tokens, examples and
microbatch counts are **sums across ranks**; elapsed seconds are the **maximum**
CUDA-synchronized rank interval after a start barrier. Peak allocated/reserved
bytes and maximum sequence length are **per-device maxima**, not total GPU
memory. Unpadded lengths and microbatch details retain all ranks' observations.
Each rank must have one actual engine update, the same global step, a full
accumulation group, no skipped update and finite losses. Forward losses are
collectively checked before backward. The last six of eight global updates
feed throughput; `final_ranks` preserves finite final losses/exact step counts
and the maximum rank training wall time is reported.
Version 2 also persists `engine_skipped_steps` and `engine_skipped_updates`;
the actual engine's cumulative skipped-update counter must start at zero and
never increase. `engine_optimizer_updates` subtracts that skip delta from
the engine global-step delta, rather than assuming every counter increment
was a successful update.

Final model gathering/saving executes on **all** ranks, with barriers around
the collective export; tokenizer and summary writes execute only on rank zero.
`export_complete: true` is added only after collective model/state and
rank-zero tokenizer save phases succeed. It does not claim reload or tensor
validation: orchestration must check those after torchrun exits successfully.
Ordinary local export failures are exchanged and persisted as failed reports.
Hard process death or a failure inside a collective still requires torchrun's
process supervision and the outer job's hard timeout; no in-process error
exchange can guarantee recovery from a dead peer.

### Opt-in bounded full checkpoint/resume validation

Final `save_model()` plus `save_state()` exports weights and Trainer metadata;
it is **not** a resumable DeepSpeed optimizer checkpoint. For a two-phase
validation within one orchestration-owned bounded job, add
`validation_checkpoint_step: 4` to **both** rendered YAMLs. Keep
`bounded_validation.expected_steps: 8` and `--max-steps 8` in both phases:
the optimizer/scheduler horizon is eight total updates, not four initially
followed by eight new updates. This opt-in supports world eight and at most
eight total updates, with `warmup_steps < checkpoint_step < expected_steps`,
`eval_strategy: no` and `save_strategy: no`. This validation uses native
torch AdamW and the external HF scheduler; DeepSpeed `optimizer` and
`scheduler` config blocks are rejected before model loading.

Phase one uses `resume_from_checkpoint: null`. At exactly update four, the
callback requests a native HF full checkpoint and stops. HF calls
`DeepSpeedEngine.save_checkpoint` on every rank, saves the external HF
`scheduler.pt`, rank-specific RNG state, `training_args.bin`, and
`trainer_state.json` with `max_steps: 8`. The trainer writes
`checkpoint-4/bounded_checkpoint.json` **only after** all native saves and
rank-state proof writes complete collectively. The phase report becomes
`checkpoint_saved`, `global_step: 4`; no final model/tokenizer/summary export
is performed. A fresh phase refuses to overwrite an existing checkpoint.

Phase two changes **only** `resume_from_checkpoint` to the explicit
`<same output_dir>/checkpoint-4` path. `auto`, a different output directory,
missing/incomplete checkpoints, changed configuration/data/packages, or a
different total step horizon are rejected rather than treated as warmstarts.
The exact contract covers every SFT config field except the resume path,
the original model config, raw train/validation file SHA256s, tokenized train
row/order fingerprint, DeepSpeed input config snapshot, package versions,
and the three trainer Python source file SHA256s.
Both phases must therefore retain the same base path/revision, output path,
seed, sampler policy, batching, optimizer/scheduler and preprocessing settings.
Contracts and completed-manifest hashes must agree across all ranks.
Rank zero verifies the full checkpoint inventory/hash once before model load,
avoiding eight redundant reads of every large optimizer shard; each rank
then independently checks its actual restored state.

The completion manifest contains `schema_version: 1`, `checkpoint_step`,
`contract`, `files` (relative path to `{bytes, sha256}`), and `measurements`
(the immutable first-phase report). Required artifacts include all eight
`global_step4/zero_pp_rank_N_mp_rank_00_model_states.pt`,
`global_step4/bf16_zero_pp_rank_N_mp_rank_00_optim_states.pt`,
`rng_state_N.pth`, `checkpoint_rankN.json`, `latest`, scheduler/args/Trainer
state. Inventory hashing covers all checkpoint files except the completion
manifest itself. The DS `latest` tag and saved Trainer global step/horizon
must match; deleting or changing an artifact fails preflight.

`checkpoint_rankN.json` stores `{rank, state, rng_sha256}`. `state` contains
`global_step`, `optimizer_partition_count`, `optimizer_steps`, and
`model_sha256`, `optimizer_sha256`, `scheduler_sha256`. These are actual-state
fingerprints, not just hashes of filenames: local BF16 parameter partitions
and persistent buffers, FP32 master partitions and AdamW moments/group
hyperparameters, and scheduler state respectively. Nonfinite tensor state
is rejected. Per-rank RNG fingerprints cover Python, NumPy, torch CPU and
CUDA generators. Hashing uses structured type/shape/dtype metadata and exact
tensor bytes in bounded CPU chunks, preserving BF16 values without conversion.
Fingerprinting and checkpoint I/O add explicit wall-time/memory overhead;
they are outside optimizer-update timing and must fit the outer job budget.

HF resumes via its strict DeepSpeed model load with optimizer/scheduler
restoration enabled; the original total horizon is retained.
Before resumed updates, the trainer requires actual engine/Trainer/AdamW
partition/scheduler step counters equal four and saved/restored
model/optimizer/scheduler hashes match. HF restores rank RNG after dataloader
skip and before forward; that restored fingerprint must match too.
`ignore_data_skip` is forced false. Sequential diagnostic sharding therefore
continues at global rows 128..159 for update five, not rows 0..31 again.
For opt-in shuffled checkpoint validation, Accelerate's seedable sampler with
`data_seed: seed` reconstructs epoch order; ordinary non-opt-in full training
retains its existing native shuffled sampler behavior.

Reports add `checkpoint_validation` with `checkpoint_step`, `total_max_steps`,
`phase` (`initial`/`resumed`), `resume_from_checkpoint`, `start_global_step`,
and `rng_restored`. Phase one additionally records `checkpoint_path`,
`checkpoint_manifest_sha256` and `saved_state_ranks` (rank/state/RNG proofs).
Phase two records `checkpoint_manifest_sha256`, local `restored_state`,
`restored_state_ranks` (rank/state), local `rng_sha256`, and
`rng_restored_ranks` (rank/RNG hash). All rank arrays use global-rank order.
Each optimizer-step entry is labeled `phase`; phase-two reports retain the
original four entries and append only steps five through eight. `phases`
records start/end global steps and each phase's maximum rank training wall
time. `engine_global_steps_start` truthfully records zero in phase one and
four in phase two; it does not pretend the resumed engine starts at zero.
Throughput still uses updates three through eight, but
`throughput_scope` explicitly marks checkpoint-separated intervals, **not
a contiguous six-step baseline**. Final `training_loss` and
`training_wall_seconds` describe the resumed phase only, with corresponding
scope fields; `phase_training_wall_seconds_sum` adds the two training phases
without claiming end-to-end checkpoint save/model reload time. Resumed
callback wall time includes dataloader skipping and the deferred native RNG
restore/fingerprint; model reload and state hashing before the train-begin
callback completes are excluded.

CPU regressions compare actual tiny hybrid Qwen3.5 uninterrupted training
against native save/resume with fixed and shuffled order, requiring bitwise
matching final model/AdamW state, scheduler state and exact dataloader row
continuity. Mocked eight-rank tests cover full-shard inventories, incomplete/
incompatible checkpoints, restored-state/RNG mismatch, total-step guards,
phase stop/export behavior and carried telemetry. Neither establishes an
actual eight-GPU DeepSpeed checkpoint/reload; that remains the bounded
cloud validation owner's responsibility.

### Separately opt-in full-corpus training

`full_corpus_contract: <local JSON path>` and
`full_corpus_segment_end_step: <planned absolute optimizer step>` select a
separate trainer in `sft/full_corpus.py`. They do **not** relax any bounded
eight/ten-update diagnostic guard. Do not supply `--max-steps`, `--debug`,
`--smoke-test`, output overrides, or bounded-validation settings in this mode.
Keep `eval_strategy: no`, `save_strategy: no`, and explicit resume paths.
`num_train_epochs` must equal the manifest's `full_passes`; the scheduler uses
the manifest's unchanged **whole-corpus** `total_max_steps` in every segment.
Neither the trainer nor the manifest grants job submission or runtime/cost
approval. The orchestration owner must separately enforce those budgets;
expiration before the last step means an incomplete corpus, not completion.

The current approved source inventory has `gem: 142608`, `hpf: 271009`,
`mai_profile: 143278` retained sources (556895 total). Frequencies 1/2/1
materialize 827904 physical exposures, not 556895 uniformly weighted rows.
With eight ranks, microbatch one, accumulation four, each complete weighted
pass requires 25872 updates. The initial contract rejects a nondivisible
remainder explicitly: no dropped rows, shard padding/repeated initial rows,
or silently shortened final accumulation groups. Any future partial-group
policy requires separate implementation and proof. Native loss/masking is
unchanged; HPF is approximately 65.47% of weighted row contributions, **not**
the old balanced diagnostic's 50% or a token-weight share. A candidate
learning rate such as `2.5e-6` is configuration, not evidence of quality.
Cosine warmup ratio `.03` with horizon 25872 resolves to 777 warmup updates,
not the old short diagnostic horizon.

The strict schema1 JSON requires exactly these top-level fields:

| Field | Required content |
|---|---|
| `schema_version`, `quality_train` | Integer `1`, boolean `false` |
| `export_at_segment_end` | Explicit boolean; `true` enables native HF retention export after each completed segment checkpoint |
| `model` | `{base_model, revision}` matching actual local model path and pinned revision |
| `input_files` | `train`, `val`, `ledger`, each `{path, sha256}` |
| `artifacts` | Nonempty `sources`, `instructions`, `tokenizer` maps, each named entry `{path, sha256}` |
| `source_counts`, `exposure_counts`, `frequencies` | Exact `gem`, `hpf`, `mai_profile` maps; frequencies 1/2/1; exposure counts equal source counts times frequencies |
| `full_passes`, `total_max_steps` | Explicit positive integers; horizon equals passes times exposures/global batch |
| `segment_end_steps` | Explicit strictly increasing positive absolute steps ending at total horizon |
| `training` | All fields listed below, equal to the rendered SFT YAML |
| `sampler` | `{algorithm: "torch-randperm-seed-plus-pass-v1", permutation_sha256: [one digest per pass]}` |

`training` requires `world_size`, `per_device_train_batch_size`,
`gradient_accumulation_steps`, `seed`, `learning_rate`, `weight_decay`,
`optim`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, `lr_scheduler_type`,
`warmup_ratio`, `max_seq_len`, `enable_thinking`, `mask_think_in_loss`.
The supported runtime is world8/micro1/accum4/seed42, native `adamw_torch`,
cosine, explicit nonthinking and no think masking. The full dense
Qwen3.5 revision remains `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
with unused visual parameters frozen and retained in full checkpoints.
Torch `2.6.0+cu124`, Transformers `5.19.0`, Accelerate `1.15.0`, and
DeepSpeed `0.16.9` must match exactly. ZeRO3/native AdamW/external HF
scheduler/no offload is required; no DeepSpeed optimizer/scheduler blocks,
Muon adapter, stack upgrade, packing, replay, or LoRA is introduced.

`artifacts.sources` must contain `gem`, `hpf`, `mai_profile` approved derived
kept files; additional canonical/exclusion provenance artifacts are allowed.
Every instruction artifact is hashed. Tokenizer artifacts must reside in
the actual local base-model directory and cover every tokenizer vocabulary/
config/special-token/chat-template asset there. All SHA256s are lowercase
64-character hex; paths refer to actual immutable worker-local inputs.
Unknown fields, duplicate JSON fields, and nonfinite JSON constants fail.

The ordered ledger is JSONL with exactly `{task, source_id, exposure_index}`.
Indices are integers 0..N-1 without gaps. `source_id` is
`task:derivedSourceSHA:zeroBasedByteOffset:rawLineSHA`; offset must be a line
boundary in that task's bound derived file, and raw-line SHA excludes only
trailing CR/LF bytes, not whitespace. Each distinct source must occur exactly
its task frequency. Source files, raw records, per-task distinct counts and
physical exposure counts are verified, not inferred from a total row count.
The prepared train-file SHA binds the instruction-normalized exposure rows
in the same order; the orchestration audit owns that source-to-prepared-row
derivation.

Full-corpus raw rows have exactly `{task, messages}`, with each message
exactly `{role, content}` containing text. `full_corpus_raw_dataset(records)`
constructs this schema directly with Arrow `large_string` for task, role
and content, and `large_list(struct)` for messages, **before** ordinary
datasets fingerprinting. Arrow32 string offsets can overflow when the
fingerprinter combines message-content chunks exceeding 2 GiB even though
individual records are valid. Explicit 64-bit offsets preserve every field,
row, message and content byte without disabling fingerprints or skipping
rows. Unexpected raw fields/types fail instead of being projected away.
Ordinary and bounded-diagnostic `Dataset.from_list` construction is unchanged.
For the full build path, 1024-row Arrow IPC batches are written to a
per-process owned temporary raw file under the standard `TMPDIR` policy.
Native `Dataset.from_file` memory-mapped fingerprinting avoids pickling/copying
the entire multi-GiB raw payload. The Python raw list is released before
tokenization. Full-mode map/filter explicitly retain the original in-memory
tokenized-output behavior, so their result never references the temporary
raw IPC file; the raw mapping is released and the file/directory removed
on success or error. Transient paths can affect native dataset fingerprints
but not tokenized values, the contract's content hash, or coverage identity.
No persistent cache or new orchestration configuration is introduced.
This fixes Arrow offset representability, not full-corpus tokenization RAM,
throughput, or GPU continuation; those still require observed evidence.

Pass `p` uses actual CPU `torch.randperm(N, generator=manual_seed(42+p))`.
The sampler supports native `set_epoch`; Accelerate shards this permutation
without replacements. Permutation SHA is over the contiguous CPU int64
vector's raw NumPy bytes; generate it in the **pinned worker Torch version**,
not an assumed equivalent Windows/other-Torch RNG. Runtime compares each
actual microbatch exposure index to the contracted pass/rank/micro index.
All approved rows must tokenize successfully. Safe **left-context** cropping
retains the entire gold target and predecessor with the existing exact chat
prefix/mask algorithm; target cutting, skip, splitting, or repair is not
allowed. Exact per-exposure context-removal counts are included in the
tokenized hash, with weighted per-task crop counts/removed tokens/supervised
token totals in `preprocessing_per_task`. `tokenized_train_sha256` hashes
canonical sorted-key compact ASCII JSON per tokenized row plus a newline,
including IDs, attention mask, labels, exposure index, and context-removal
count. Contract/runtime hashes use sorted-key compact ASCII JSON SHA256.

Every completed update collectively proves all eight ranks, four finite
microbatch losses/rank, exact expected indices, one actual engine update,
and no skipped update. Only then is rank-zero
`full_corpus_coverage.jsonl` appended and fsynced. Coverage is **not** granted
merely for a forward, planned file list, or Trainer sample count.
`full_corpus_inflight.json` records update intent before forward and is
removed only after durable completed-update evidence. An unresolved intent,
live journal beyond the saved checkpoint, existing destination checkpoint,
or automatic resume fails closed rather than replaying uncertain updates.

At each explicit segment boundary native HF saves all DS model/AdamW shards,
external scheduler, rank RNG, args, and Trainer state. After all-rank saves,
`checkpoint-N/checkpoint_rankN.json` contains `{rank, state, rng_sha256,
coverage}`. Actual model/FP32 masters/AdamW moments/scheduler hashes and step
counters use the existing full-state helper. `coverage` contains `rank`,
`global_step`, `completed_full_passes`, `partial_pass_optimizer_steps`,
local `consumed_exposures`, and a cumulative `coverage_sha256` over canonical
`{step, indices}` records for that rank. The rank-zero atomic
`full_corpus_checkpoint.json` marker contains `schema_version: 1`,
`checkpoint_step`, `runtime_contract`, `files: {path: {bytes, sha256}}`,
and `saved_state_ranks`. Inventory includes immutable `coverage.jsonl`
and all native rank state files; only the marker excludes itself.

Resume must use the explicit same-output checkpoint at the immediately
preceding planned boundary. Rank zero verifies full file hashes; all ranks
replay/validate the completed-update journal, compare contract and manifest
hashes, reconstruct their coverage cursor, and verify actual restored
model/AdamW/scheduler state before updates. Native RNG restore is checked
after data skipping and before the next forward. Changed inputs, instructions,
tokenizer, packages, trainer source, batching, seed, optimizer, scheduler
horizon, or missing/corrupt state is rejected, never treated as warmstart.

`full_corpus.json` and `full_corpus.rankN.json` report runtime/topology/NCCL
proof, actual optimizer group coverage, preload partition proof, saved/
restored state and RNG rank proofs, segment boundaries, and coverage cursors.
Intermediate segments report `segment_complete`, `corpus_complete: false`.
With contracted `export_at_segment_end: false`, they do not export. With
`true`, after the complete native checkpoint they execute the same all-rank
HF `Trainer.save_model`/state and rank-zero tokenizer path, including native
HF generation config, and record `segment_export_complete: true` separately
from corpus completion. This allows orchestration-owned retention evaluation
of that actual snapshot after DS exits; it grants neither evaluation/job
authorization nor a full-coverage claim. Only the exact whole horizon reports
`corpus_complete`, full-pass source/exposure counts and total completed
exposures; final full-model export still requires all-rank barriers.
Exact-horizon export additionally records `export_complete: true`.
Segment training loss is labeled as segment-only. This is coverage/state
evidence, not a quality result, throughput forecast, or promise of finishing
within a runtime ceiling. `segment_export_files` records actual native HF
model/generation/tokenizer/chat-template filenames and streamed `{bytes,
sha256}` identities after all writes, including actual single/sharded
safetensors layout rather than assuming a filename. Hashing reads at most
8 MiB per chunk; it adds real checkpoint/export wall-time overhead.

Public read-only integration helpers are:

```python
FullCorpusContract(path, cfg)
validate_full_checkpoint(path, runtime_contract, contract,
                         verify_files=True, for_resume=True)
verify_full_segment(output_dir, contract, expected_runtime_contract=None)
segment_export_inventory(path)
```

The first validates actual staged worker-local data/source/ledger/tokenizer/
instruction/permutation identities without initializing a model. `cfg` is
the rendered `SFTConfig` or an equivalent field-bearing object. It does not
itself tokenize the full corpus; runtime `validate_dataset(dataset, stats)`
adds no-skip/target/predecessor/crop checks and the tokenized identity.
`validate_full_checkpoint` rejects the final horizon for resume; a read-only
consumer can set `for_resume=False` to validate a final checkpoint too.
`verify_full_segment` derives `checkpoint-N` from the downloaded report,
recomputes full file hashes, replays completed-update IDs, reconstructs
actual per-task distinct-source/exposure counts and all rank coverage
digests, compares saved rank fingerprints and report consistency, and
verifies native HF export hashes/completion independently of summary counts.
Supply the audited expected runtime contract when available; otherwise its
worker-specific package/model/DS/code fields require additional consumer
validation. This is artifact/coverage consistency verification, **not**
offline reconstruction of original GPU tensor-state hashes or a model
quality/strict-load test. Those remain the evaluator's responsibility.

For a new job, copy only the immutable immediately preceding completed
`checkpoint-N` (including its journal/marker/all native shards) into the same
stable output path. The live `full_corpus_coverage.jsonl` may be absent: rank
zero restores the exact checkpoint journal after initialization proof.
If present, it must match byte-for-byte; do not create an empty journal or
truncate an advanced journal. Any unresolved in-flight intent stops admission.
All model/input/output/contract/config paths must remain identical across
segments; do not silently rebase a contract when validating downloaded files.

CPU regressions cover actual Accelerate eight-shard full-pass permutations,
ledger/raw-source/hash/frequency failures, safe target-preserving crops,
completed-update coverage and checkpoint inventories. Native tiny hybrid
Qwen CPU training compares uninterrupted versus segmented resume across a
pass boundary, with identical row order and bitwise model/AdamW state and
exact scheduler state. DeepSpeed/CUDA save/restore tests use explicit mocks;
actual eight-GPU full-corpus save/resume, memory and retention durability
remain unverified locally and must not be claimed from these tests.

---

How the Layer1-delta SFT model (`Qwen3-4B-Thinking-2507` → distilled on GPT-4o
teacher labels) was actually trained, with the **real, recovered hyperparameters**.
This is the "how we trained it" companion to
[`evaluation_process.md`](evaluation_process.md) (how we benchmarked it).
Everything referenced below lives under `MAIDistillation0623/sft/` and
`MAIDistillation0623/models/`.

> **Base model:** `Qwen3-4B-Thinking-2507` (persisted at `models/base/`).
> **Teacher:** `gpt-4o` (reference labels in the training data).
> **Best checkpoint:** `checkpoint-2260`, eval_loss **0.0833**, persisted at
> `models/sft/layer1_delta_thinking_50k_4o-v1/`.
> **Repro run:** `layer1_delta_thinking_50k_4o-v2-repro` (online wandb project
> `maiprofile-sft`).

---

## Pipeline at a glance

```
  data/splits (v1)                  base model (cosmos)
  train/val.jsonl                   models/base/Qwen3-4B-Thinking-2507
        │                                   │
        └───────────────┬───────────────────┘
                        ▼
       scripts/launch_repro_sft.sh   (renders template → concrete config)
                        │
                        ▼
       accelerate launch (8×A100) → scripts/sft_train.py
         · DeepSpeed ZeRO-3 (configs/deepspeed/ds_config_zero3.json)
         · HF Trainer, loss masked to last assistant message
         · logs metrics → online wandb (project maiprofile-sft)
                        │
        ┌───────────────┴────────────────┐
        ▼                                ▼
  OUTPUT_DIR (local /scratch)      checkpoint-N every 200 steps
        │  rsync (exclude global_step*)
        ▼
  models/sft/<run>/  (cosmos persistent: best merged ckpt)
```

---

## Step 0 — Inputs (data + base model)

### Training data (v1 = the SFT split)
`MAIProfileSFT_50k/data/splits/layer1_delta_thinking_50k_4o/`

| file | lines | size | role |
|---|---|---|---|
| `train.jsonl` | 36 252 | 524 MB | training |
| `val.jsonl`   | 4 532  | 64 MB  | eval (every 200 steps) |
| `test.jsonl`  | 4 531  | 66 MB  | held out (used by eval, not training) |
| `manifest.json` | — | — | provenance + split stats |

- **Source curation:** `curation_data_thinking_layer1_delta_50k.jsonl`
  (`source_sha256 = 3e12f51d…e44aee8c`), split **seed=42**, ratios **80/10/10**.
- **Record schema:** `{"messages": [system, user, assistant], "metadata": {...}}`
  - `system` — Layer1 delta task spec (naming rules, schema).
  - `user`   — `Date / User ID / Demographics / Today's Denoised Signals (JSON array of
    {Date, Source, DetailedSource, Action, intent})`.
  - `assistant` — GPT-4o teacher target: `<think>…</think>` + ` ```json {"interests":[…]} ``` `.
  - `metadata` — `{model: gpt-4o, user_id, date, delta_index, prompt_tokens,
    completion_tokens, temperature 0.2, …}`.
- **Teacher assistant length:** mean 4 985 chars, max 82 518 (manifest).

### Base model
`models/base/Qwen3-4B-Thinking-2507/` — re-downloaded + persisted to cosmos
(config + generation_config + 3 safetensors shards + tokenizer). This is the
exact base the SFT/repro run loads.

---

## Step 1 — Render the config + launch (8×A100)

Driver: `scripts/launch_repro_sft.sh` (portable; resolves repo/cosmos roots from
its own location, so it is **job-id agnostic** — the original v1 config hardcoded
the now-dead AML job id `fc096b74…`).

```bash
PRL=/home/aiscuser/.conda/envs/pipeline-rl/bin
cd MAIDistillation0623/sft
export WANDB_API_KEY=...            # online wandb (never commit; rotate after)
export WANDB_MODE=online
# unset any leaked path envs from prior dry-runs first:
unset BASE_MODEL TRAIN_JSONL VAL_JSONL OUTPUT_DIR DEEPSPEED_CONFIG RUN_NAME
setsid nohup bash scripts/launch_repro_sft.sh > logs/repro_sft.log 2>&1 < /dev/null &
```

What the launcher does:
1. `envsubst` the template `configs/sft/layer1_delta_thinking_50k_4o-v2.repro.yaml.tmpl`
   → concrete `configs/sft/rendered/<run>.yaml` (fills `BASE_MODEL / TRAIN_JSONL /
   VAL_JSONL / OUTPUT_DIR / DEEPSPEED_CONFIG`).
2. Defaults: `BASE_MODEL=models/base/Qwen3-4B-Thinking-2507`,
   `DATA_DIR=…/MAIProfileSFT_50k/data/splits/layer1_delta_thinking_50k_4o`,
   `OUTPUT_DIR=/scratch/local_sft_runs/<run>` (local fast disk),
   `PERSIST_CKPT_DIR=models/sft/<run>` (cosmos).
3. Strips AzureML MPI envs (`RANK/WORLD_SIZE/MASTER_ADDR/…`) so accelerate runs
   single-node 8-GPU DDP; sets NCCL/CUDA stability envs;
   `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.
4. `accelerate launch --config_file configs/accelerate/accelerate_ds3.yaml
   --num_processes 8 scripts/sft_train.py --config <rendered>`.
5. On finish: `rsync -a --exclude='global_step*' OUTPUT_DIR/ PERSIST_CKPT_DIR/`
   (keeps the merged HF checkpoints, drops the bulky DeepSpeed shards).

---

## Step 2 — Training (`scripts/sft_train.py`)

HF `Trainer` + DeepSpeed ZeRO-3, full-parameter (no LoRA/PEFT).

### Hyperparameters (frozen — match the recovered original run)

| param | value | note |
|---|---|---|
| max_seq_len | **14336** | 99.57% coverage on the Qwen tokenizer |
| per_device_train_batch_size | 1 | |
| gradient_accumulation_steps | 4 | 8 GPU × 1 × 4 = **effective batch 32** |
| num_train_epochs | **2** | confirmed via best ckpt `trainer_state.json` (epoch=2.0, step=2260) |
| learning_rate | 5.0e-6 | |
| lr_scheduler_type | cosine | warmup_ratio 0.03 |
| weight_decay | 0.0 | |
| seed | 42 | |
| mask_think_in_loss | **false** | think tokens DO contribute to loss |
| attn_implementation | sdpa | flash-attn ABI mismatch on this box; SDPA works |
| gradient_checkpointing | true | |
| eval / save | every **200** steps | save_total_limit 3 |
| logging_steps | 10 | |
| resume_from_checkpoint | auto | |
| **total steps** | **2260** | ~1130 steps/epoch × 2 |

### Loss masking
`tokenize_record()` renders the full chat with the Qwen chat template, then sets
`labels = -100` everywhere **except the last assistant message span** (the teacher
target). Overlong records are **left-truncated** to `max_seq_len` keeping the
assistant target intact; if the target itself gets cut, the record is skipped.
With `mask_think_in_loss=false` the `<think>…</think>` block trains as part of the
target.

> Build stats (repro run): train kept **36 142** (skip_truncated_target 110,
> overlong 66), val kept **4 522** (skip 10, overlong 8). Tokenization is a
> single-thread python loop redundantly run on all 8 ranks → ~10 min CPU-bound
> with GPUs idle before the first step (expected, not a hang).

### wandb logging (online)
`sft_train.py` enables `report_to=["wandb"]` when `wandb_project` is set (and not
`--no-wandb/--smoke-test/--debug`). It records:
- **metrics** every 10 steps: `loss`, `grad_norm`, `learning_rate`, `epoch`;
  eval `eval_loss` every 200 steps;
- **config**: all hyperparams above (HF Trainer auto-uploads);
- **system metrics**: 8× GPU util/mem/power/temp, CPU, disk, network (wandb auto).
- The **training data** is separately versioned as a wandb **dataset artifact**
  (see Step 4).

---

## Step 3 — Checkpoints & persistence

- Trainer writes `checkpoint-200, 400, …` to `OUTPUT_DIR` (local
  `/scratch/local_sft_runs/<run>`), keeping the latest 3 + the best by eval_loss.
- The **best** checkpoint (original run: `checkpoint-2260`, eval_loss 0.0833) is
  rsynced to cosmos `models/sft/<run>/` as a flat merged HF checkpoint:
  `model.safetensors` (8.0 GB), `config.json`, `generation_config.json`,
  tokenizer files, `chat_template.jinja`, `training_args.bin`, `training_summary.json`.
  The DeepSpeed `global_step*` shards are **excluded** (not needed to serve/eval).

---

## Step 4 — Training data → online wandb artifact (`scripts/log_data_to_wandb.py`)

Stores the exact training data durably + versioned alongside the run:

```bash
$PRL/python scripts/log_data_to_wandb.py \
  --data-dir <…>/layer1_delta_thinking_50k_4o \
  --config   configs/sft/rendered/<run>.yaml \
  --project  maiprofile-sft \
  --artifact layer1_delta_thinking_50k_4o-data
```

- Logs `train/val/test.jsonl` + `manifest.json` + the rendered SFT config as a
  wandb **`dataset`** artifact, each file tagged with its **sha256**
  (train `df821863…`, val `ae384d5e…`, test `1000bdb3…`) for integrity/versioning.
- Result: project `maiprofile-sft`, artifact `layer1_delta_thinking_50k_4o-data`.

---

## Files in this repo (all required to reproduce)

| path | role |
|---|---|
| `scripts/sft_train.py` | HF Trainer + DeepSpeed ZeRO-3 trainer; loss masking; wandb wiring |
| `scripts/launch_repro_sft.sh` | **portable** launcher: render template → 8-GPU accelerate → rsync to cosmos |
| `scripts/launch_layer1_delta_thinking_50k_sft.sh` | original launcher (references the dead job id; kept for provenance) |
| `scripts/log_data_to_wandb.py` | upload train/val/test + manifest + config as a versioned wandb dataset artifact |
| `configs/sft/layer1_delta_thinking_50k_4o-v2.repro.yaml.tmpl` | **portable** config template (paths injected at launch) |
| `configs/sft/layer1_delta_thinking_50k_4o-v1.yaml` | original recovered config (dead-job paths; provenance) |
| `configs/sft/_ref_layer1_delta.yaml` | reference config |
| `configs/sft/rendered/<run>.yaml` | auto-generated concrete config (by launcher) |
| `configs/accelerate/accelerate_ds3.yaml` | accelerate: 8-proc MULTI_GPU bf16, no hardcoded paths |
| `configs/deepspeed/ds_config_zero3.json` | DeepSpeed ZeRO-3, no offload, auto batch/grad |
| `env/pipeline-rl-pip-freeze.txt` | exact training deps (torch 2.6.0, transformers 5.5.4, accelerate 1.8.1, deepspeed 0.15.4, datasets 4.8.4, wandb 0.27.0) |

---

## Environment
- conda `pipeline-rl` at `/home/aiscuser/.conda/envs/pipeline-rl/bin` (invoked by
  absolute path; `conda activate` is broken on this box). Training deps were
  installed to match `env/pipeline-rl-pip-freeze.txt`.
- 8× A100-80GB, single node. bf16. DeepSpeed ZeRO-3, no offload.

## Secrets
- `WANDB_API_KEY` read from env (or `/home/aiscuser/.secrets/maiprofile_sft.env`
  if present). **Never committed. Rotate after any session where it was exported.**

## Gotchas (learned the hard way)
1. `accelerate`/`deepspeed`/`datasets`/`wandb` were **missing** from `pipeline-rl`
   (it was an eval-only env) — install to the freeze versions before launching.
2. `run_in_terminal` shares one persistent bash → dry-run `export OUTPUT_DIR=…`
   leaks into the next launch. **`unset` the path envs before launching.**
3. `du` reports 0 for files on the cosmos mount — use `ls -la` to verify sizes.
4. The original v1 config/launcher reference the dead job id `fc096b74…`; use the
   portable `*.repro.yaml.tmpl` + `launch_repro_sft.sh` instead.
