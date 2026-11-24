"""Score synthetic predictions to verify the AML and AgentX metric handoff."""

import argparse
import json
from pathlib import Path


LABELS = ("cat", "dog", "dog", "cat")
PREDICTIONS = {
    "checkpoint-3000": ("cat", "cat", "dog", "dog"),
    "checkpoint-2000": ("cat", "dog", "dog", "dog"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", choices=PREDICTIONS)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    predictions = PREDICTIONS[args.checkpoint]
    matches = sum(label == prediction for label, prediction in zip(LABELS, predictions))
    result = {
        "smoke_exact_match": matches / len(LABELS),
        "checkpoint": args.checkpoint,
        "examples": len(LABELS),
        "synthetic": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    line = json.dumps(result, separators=(",", ":"))
    (args.output / "smoke_score.json").write_text(line + "\n", encoding="utf-8")
    print("AML_RESULT_JSON=" + line, flush=True)


if __name__ == "__main__":
    main()
