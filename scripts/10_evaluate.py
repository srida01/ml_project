#!/usr/bin/env python3
"""Evaluation entry point; consumes frozen predictions and labels."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ids.evaluation.metrics import classification_metrics
from ids.utils.io import read_table, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Predictions table (e.g. from 11_predict_batch.py)")
    parser.add_argument("--labels", help="Table with flow_id and binary_label, joined when --input has no labels")
    parser.add_argument("--output", default="reports/tables/evaluation.json")
    args = parser.parse_args()
    frame = read_table(args.input)
    if args.labels:
        labels = read_table(args.labels)[["flow_id", "binary_label"]]
        frame = frame.drop(columns=["binary_label"], errors="ignore")
        frame["flow_id"] = frame["flow_id"].astype(str)
        labels["flow_id"] = labels["flow_id"].astype(str)
        frame = frame.merge(labels, on="flow_id", how="inner", validate="one_to_one")
    if "binary_label" not in frame.columns:
        raise SystemExit("No binary_label column: pass --labels with a flow_id/binary_label table")
    if "predicted_binary_label" not in frame.columns:
        if "final_alert_label" not in frame.columns:
            raise SystemExit("Input needs predicted_binary_label or final_alert_label")
        frame["predicted_binary_label"] = (frame["final_alert_label"] != "benign").astype("int8")
    result = classification_metrics(frame["binary_label"], frame["predicted_binary_label"], frame.get("svm_score"))
    write_json(result, args.output)
    print(f"Evaluated {len(frame)} rows; wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
