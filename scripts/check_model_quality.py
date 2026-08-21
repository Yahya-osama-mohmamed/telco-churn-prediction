#!/usr/bin/env python
"""Fail the build if the shipped model no longer clears its floor.

A retrain that quietly degrades is the failure mode this catches: the notebook
still runs, the tests still pass, and a worse model reaches the image. The
floors below sit just under the numbers the README publishes, so genuine noise
between retrains passes while a real regression stops the pipeline.

Reads the metadata the notebook writes; the test set is never re-scored here.
"""

import json
import sys
from pathlib import Path

METADATA = Path(__file__).resolve().parent.parent / "models" / "model_metadata.json"

# (metric, comparison, floor/ceiling, why this number)
CHECKS = [
    ("test_roc_auc", "min", 0.83, "published 0.854; below 0.83 is a real regression"),
    ("val_roc_auc", "min", 0.82, "published 0.837"),
    ("decision_threshold", "min", 0.01, "a threshold of 0 means tuning silently failed"),
    ("decision_threshold", "max", 0.99, "a threshold of 1 flags nobody"),
]


def main() -> int:
    if not METADATA.exists():
        print(f"FAIL: {METADATA} not found - run the notebook before shipping.")
        return 1

    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    print(f"model    : {meta.get('best_model')}")
    print(f"trained  : {meta.get('trained_at_utc')}")
    print()

    failures = []
    for metric, kind, bound, why in CHECKS:
        value = meta.get(metric)
        if value is None:
            failures.append(f"{metric} missing from metadata")
            continue
        ok = value >= bound if kind == "min" else value <= bound
        symbol = ">=" if kind == "min" else "<="
        print(f"{'PASS' if ok else 'FAIL'}  {metric:20} {value:<10} {symbol} {bound}   ({why})")
        if not ok:
            failures.append(f"{metric}={value} violates {symbol} {bound}")

    print()
    if failures:
        print("Model quality gate FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Model quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
