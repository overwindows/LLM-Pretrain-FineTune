"""Publish the MAI Profile evaluator's aggregate score in the AML user log."""

import argparse
import json
from pathlib import Path


CHECKPOINTS = ("checkpoint-2000", "checkpoint-3000")


def metric_line(score_file, checkpoint, expected_examples):
    report = json.loads(score_file.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("task") != "mai_profile":
        raise ValueError("Expected the aggregate result for the mai_profile task")

    examples = report.get("examples")
    parse_fail = report.get("parse_fail")
    score = report.get("exact_match")
    if type(examples) is not int or examples <= 0 or examples != expected_examples:
        raise ValueError("Evaluation did not score the expected number of examples")
    if type(parse_fail) is not int or not 0 <= parse_fail < examples:
        raise ValueError("Evaluation produced no usable predictions or no parse-failure count")
    if type(score) not in (int, float) or not 0 <= score <= 1:
        raise ValueError("Evaluation did not report a finite exact_match between 0 and 1")

    result = {
        "mai_profile_exact_match": score,
        "checkpoint": checkpoint,
        "examples": examples,
        "parse_fail": parse_fail,
    }
    return "AML_RESULT_JSON=" + json.dumps(result, allow_nan=False, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_file", type=Path, help="MAI Profile score from this job")
    parser.add_argument("--checkpoint", required=True, choices=CHECKPOINTS)
    parser.add_argument("--expected-examples", required=True, type=int)
    args = parser.parse_args()

    try:
        print(metric_line(args.score_file, args.checkpoint, args.expected_examples))
    except (OSError, ValueError) as error:
        parser.error(f"Cannot publish the MAI Profile evaluation result: {error}")


if __name__ == "__main__":
    main()
