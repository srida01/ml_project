#!/usr/bin/env python3
"""Evaluation entry point; consumes frozen predictions and labels."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ids.evaluation.metrics import classification_metrics
from ids.utils.io import read_table, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Table containing binary_label and predicted_binary_label")
    parser.add_argument("--output", default="reports/tables/evaluation.json")
    args = parser.parse_args()
    frame = read_table(args.input)
    result = classification_metrics(frame["binary_label"], frame["predicted_binary_label"], frame.get("svm_score"))
    write_json(result, args.output)
    print(f"Wrote evaluation metrics to {args.output}")


if __name__ == "__main__":
    main()
