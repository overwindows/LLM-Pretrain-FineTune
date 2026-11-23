"""Run and measure one fresh Qwen3.5 SFT candidate on Azure ML."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.pipeline import _resolve_vars


def training_config(lr, run_name, cosmos_root, project_yaml, output_dir, smoke_data=False):
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    project = yaml.safe_load(project_yaml.read_text(encoding="utf-8"))
    project = _resolve_vars(project, {"cosmos_root": str(cosmos_root), "run_name": run_name})
    if smoke_data:
        smoke_dir = Path(__file__).resolve().parent
        data = {
            "sft_train_jsonl": str(smoke_dir / "smoke-train.jsonl"),
            "sft_val_jsonl": str(smoke_dir / "smoke-val.jsonl"),
        }
    else:
        data = project["data"]
    config = yaml.safe_load(
        (ROOT / "sft/configs/sft/qwen35_text.yaml.tmpl").read_text(encoding="utf-8")
    )
    for key in ("sft_train_jsonl", "sft_val_jsonl"):
        if not Path(data[key]).is_file():
            raise ValueError(f"Project data.{key} must point to a readable JSONL file")
    config.update(
        base_model=project["base_model"],
        model_revision=project.get("model_revision"),
        train_jsonl=data["sft_train_jsonl"],
        val_jsonl=data["sft_val_jsonl"],
        output_dir=str(output_dir),
        learning_rate=lr,
        seed=1,
        eval_strategy="epoch",
        lora_enabled=True,
        load_in_4bit=True,
    )
    return config


def final_eval_loss(history):
    loss = next((entry["eval_loss"] for entry in reversed(history) if "eval_loss" in entry), None)
    if type(loss) not in (int, float) or not math.isfinite(loss):
        raise ValueError("Trainer did not report a finite final eval_loss")
    return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lr", type=float)
    parser.add_argument("run_name")
    parser.add_argument("cosmos_root", type=Path)
    parser.add_argument("project_yaml", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--smoke-data", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("The candidate output directory must be empty before training")
    config = training_config(
        args.lr, args.run_name, args.cosmos_root, args.project_yaml, output_dir, args.smoke_data,
    )
    with tempfile.TemporaryDirectory() as work_dir:
        config_path = Path(work_dir) / "sft.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        subprocess.run(
            [sys.executable, str(ROOT / "sft/sft_train.py"), "--config", str(config_path), "--no-wandb"],
            cwd=ROOT, check=True,
        )
    state = json.loads((output_dir / "trainer_state.json").read_text(encoding="utf-8"))
    loss = final_eval_loss(state["log_history"])
    print("AML_RESULT_JSON=" + json.dumps({"eval_loss": loss, "lr": args.lr, "seed": config["seed"]}), flush=True)


if __name__ == "__main__":
    main()
