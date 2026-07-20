"""CLI entry point for running the agentic SFT/RL pipeline on an AML node.

This script is cloned from GitHub onto the AML GPU node and executed by
the aml_submit.py job command.  It reads project.yaml, discovers the Cosmos
mount, and runs the requested pipeline stages.

Usage (on AML node)
-------------------
python scripts/run_pipeline.py --project project_runtime.yaml [--stages sft rl] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import Pipeline
from agent.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Run the agentic SFT/RL pipeline"
    )
    parser.add_argument(
        "--project", default="project.yaml",
        help="Path to project.yaml (default: project.yaml in cwd)"
    )
    parser.add_argument(
        "--stages", nargs="+",
        choices=["sft", "data_clean", "rl", "eval"],
        help="Stages to run (default: all stages defined in project.yaml)"
    )
    parser.add_argument(
        "--cosmos-root", default=None,
        help="Override Cosmos mount root (auto-discovered if not set)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run stages even if Cosmos output already exists"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    pipeline = Pipeline.from_yaml(args.project, cosmos_root=args.cosmos_root)
    results = pipeline.run(stages=args.stages, force=args.force)

    # Print a brief summary
    print("\n" + "=" * 60)
    print("Pipeline complete")
    print("=" * 60)
    for stage, result in results.items():
        print(f"  {stage}: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
