"""Entry point for agentic pipeline runs on AML GPU nodes.

This script is submitted by the brain-side aml_submit.py in
Q:\LLM-Fine-Tuning\agentic_sft_rl\scripts\aml_submit.py.

The full implementation lives in the agent brain repo.
On GPU nodes: the brain injects the resolved project.yaml via AML input,
and this script calls Pipeline.from_yaml(...).run(stages).
"""
# Placeholder — the actual version is injected from the agentic_sft_rl brain
# See: https://github.com/overwindows/LLM-Pretrain-FineTune (this repo)
# Brain: Q:\LLM-Fine-Tuning\agentic_sft_rl\scripts\run_pipeline.py
if __name__ == "__main__":
    import sys
    print("This placeholder should be overridden by the brain's run_pipeline.py")
    sys.exit(1)
