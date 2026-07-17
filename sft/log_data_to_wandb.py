#!/usr/bin/env python
"""Log the layer1_delta SFT training data + config to Weights & Biases as a
versioned artifact, for durable provenance / reproducibility.

Creates a small `upload-dataset` run in project `maiprofile-sft` that owns a
`dataset` artifact containing the train/val/test splits, the split manifest
(with source sha256), and the rendered training config. The training run can
later reference it via `run.use_artifact(...)`.

Usage:
    WANDB_API_KEY=... python scripts/log_data_to_wandb.py \
        --data-dir   /.../MAIProfileSFT_50k/data/splits/layer1_delta_thinking_50k_4o \
        --config     configs/sft/rendered/layer1_delta_thinking_50k_4o-v2-repro.yaml \
        --project    maiprofile-sft \
        --artifact   layer1_delta_thinking_50k_4o-data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import wandb


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default=None, help="rendered SFT config to attach")
    ap.add_argument("--project", default="maiprofile-sft")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--artifact", default="layer1_delta_thinking_50k_4o-data")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = [p for p in ["train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"]
             if (data_dir / p).exists()]
    if not files:
        raise SystemExit(f"no split files found under {data_dir}")

    # Pull source provenance from manifest if present
    manifest = {}
    mpath = data_dir / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        job_type="upload-dataset",
        name=f"upload-{args.artifact}",
        tags=["layer1_delta", "sft", "dataset", "50k", "gpt4o-teacher"],
        config={
            "data_dir": str(data_dir),
            "source": manifest.get("source"),
            "source_sha256": manifest.get("source_sha256"),
            "seed": manifest.get("seed"),
            "splits": manifest.get("splits"),
            "teacher_models": manifest.get("teacher_models"),
        },
    )

    art = wandb.Artifact(
        name=args.artifact,
        type="dataset",
        description="Layer1-delta SFT curation data (GPT-4o teacher, thinking, "
                    "50K) — by-user 80/10/10 split, seed 42.",
        metadata={
            "source": manifest.get("source"),
            "source_sha256": manifest.get("source_sha256"),
            "seed": manifest.get("seed"),
            "ratios": manifest.get("ratios"),
            "splits": manifest.get("splits"),
        },
    )

    for fname in files:
        fpath = data_dir / fname
        digest = sha256_of(fpath)
        size_mb = fpath.stat().st_size / (1 << 20)
        print(f"adding {fname:16s} {size_mb:8.1f} MB  sha256={digest[:16]}…")
        art.add_file(str(fpath), name=fname)

    if args.config and Path(args.config).exists():
        art.add_file(args.config, name="sft_config.yaml")
        print(f"adding sft_config.yaml  <- {args.config}")

    run.log_artifact(art)
    art.wait()
    print(f"DONE: logged artifact '{args.artifact}' to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
