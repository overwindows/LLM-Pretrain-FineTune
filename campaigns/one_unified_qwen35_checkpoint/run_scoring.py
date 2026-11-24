"""Run the One Unified pipeline against a fresh, job-local MAI validation layout."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from emit_metric import CHECKPOINTS, metric_line

PROJECT_DIR = Path("one_unified_slm/one_unified_qwen3_5_4b_v1")
OLD_MOUNT = re.compile(r"/scratch/azureml/cr/j/[a-f0-9]+/cap/data-capability/wd/INPUT_cosmos_data(?=/|$)")
ORIGINAL_CHECKPOINT = re.compile(r"(?<![\w-])checkpoint-3000(?![\w-])")
VALIDATION_EXAMPLES = 1000


def rebase_project(value, mount, runtime, code, checkpoint):
    if isinstance(value, dict):
        return {
            key: str(code) if key == "easytrain_root" else rebase_project(
                item, mount, runtime, code, checkpoint
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rebase_project(item, mount, runtime, code, checkpoint) for item in value]
    if isinstance(value, str):
        value = value.replace(f"{mount}/", f"{runtime}/")
        value = OLD_MOUNT.sub(lambda _: str(runtime), value)
        value = value.replace("/tmp/gpu_code", str(code))
        return ORIGINAL_CHECKPOINT.sub(checkpoint, value)
    return value


def prepare_runtime(mount, runtime, code, project_path, checkpoint):
    source = mount / PROJECT_DIR
    project_root = runtime / PROJECT_DIR
    data = project_root / "data"
    sft = project_root / "sft"
    data.mkdir(parents=True)
    sft.mkdir()

    adapter = source / "sft" / checkpoint
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file() or (adapter / name).stat().st_size == 0:
            raise ValueError(f"Missing {name} for {checkpoint}")
    (sft / checkpoint).symlink_to(adapter, target_is_directory=True)

    validation = source / "data" / "mai_profile_val.jsonl"
    if not validation.is_file() or validation.stat().st_size == 0:
        raise ValueError("Missing MAI Profile validation split")

    registry = json.loads((source / "data" / "task_registry.json").read_text(encoding="utf-8"))
    task = registry["tasks"]["mai_profile"]
    validation_split = task["splits"]["val"]
    if validation_split.get("examples") != VALIDATION_EXAMPLES:
        raise ValueError(f"Expected {VALIDATION_EXAMPLES} MAI Profile validation examples")
    (data / validation.name).symlink_to(validation)
    # EvalStage reads the registry's test slot. Put the validation split there
    # for checkpoint selection and leave the held-out test file untouched.
    eval_input = data / "mai_profile_test.jsonl"
    eval_input.symlink_to(validation)
    task.update(test=str(eval_input), val=str(data / validation.name))
    task["splits"]["test"] = {**validation_split, "source": str(eval_input)}
    registry["tasks"] = {"mai_profile": task}
    mai_contract = next(
        item for item in registry["task_contract"] if item["task_name"] == "mai_profile"
    )
    mai_contract["data"].update(
        test_jsonl=str(eval_input), val_jsonl=str(data / validation.name)
    )
    registry["task_contract"] = [mai_contract]
    (data / "task_registry.json").write_text(json.dumps(registry), encoding="utf-8")

    options = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    if not isinstance(options, dict):
        raise ValueError("Expected the project YAML to contain a mapping")
    options = rebase_project(options, mount, runtime, code, checkpoint)
    options.update(
        easytrain_root=str(code),
        cosmos_root=str(runtime),
        cosmos_persist_root=str(project_root),
        local_scratch_root=str(runtime / "scratch"),
    )
    project = runtime / "project_runtime.yaml"
    project.write_text(yaml.safe_dump(options, sort_keys=False), encoding="utf-8")
    return project


def run_scoring(args):
    code = args.work / "code"
    runtime = args.work / "cosmos"
    evaluator = code / "tasks" / "mai_profile" / "eval_mai_profile.py"
    if not evaluator.is_file():
        raise ValueError("The mounted agent package lacks the MAI Profile evaluator")
    project = prepare_runtime(args.mount, runtime, code, args.project, args.checkpoint)
    env = {
        **os.environ,
        "AGENTIC_COSMOS_ROOT": str(runtime),
        "AGENTIC_RUN_NAME": PROJECT_DIR.name,
        "AZURE_ML_INPUT_COSMOS_DATA": str(runtime),
        "EASYTRAIN_ROOT": str(code),
    }
    subprocess.run(
        [sys.executable, str(code / "scripts" / "run_pipeline.py"),
         "--project", str(project), "--stages", "lora_merge", "eval"],
        cwd=code,
        env=env,
        check=True,
    )

    reports = [
        path for path in runtime.rglob("mai_profile_eval.jsonl")
        if path.is_file() and path.resolve().is_relative_to(runtime)
    ]
    if len(reports) != 1:
        raise ValueError(f"Expected one MAI Profile score from this run, got {len(reports)}")
    report = reports[0]
    line = metric_line(report, args.checkpoint, VALIDATION_EXAMPLES)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report, args.output / "mai_profile_eval.json")
    print(line, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", choices=CHECKPOINTS)
    parser.add_argument("mount", type=lambda value: Path(value).resolve())
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("work", type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    try:
        run_scoring(args)
    except (OSError, ValueError, KeyError, StopIteration, subprocess.CalledProcessError, yaml.YAMLError) as error:
        parser.exit(1, f"MAI Profile scoring failed: {error}\n")


if __name__ == "__main__":
    main()
