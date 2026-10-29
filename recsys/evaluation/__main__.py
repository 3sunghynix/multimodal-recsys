"""Run from repository root: python -m recsys.evaluation INPUT.json --output OUTPUT.json"""

import argparse
import json
from pathlib import Path

from .metrics import evaluate_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate prepared rankings, not a model or database.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="New JSON file; existing results are never overwritten")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as file:
        payload = json.load(file)
    # JSON object keys are strings; do not silently mismatch integer customer
    # IDs in eligible_users against string keys in rankings and targets.
    if any(not isinstance(user, str) for user in payload["eligible_users"]):
        parser.error("JSON eligible_users must contain string IDs, matching rankings/targets keys")
    report = evaluate_report(**payload)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        # Exclusive creation avoids accidental replacement of saved experiments.
        with args.output.open("x", encoding="utf-8") as file:
            file.write(text + "\n")
        print(f"Saved: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
