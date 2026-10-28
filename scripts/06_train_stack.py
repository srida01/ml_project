#!/usr/bin/env python3
"""Train the OOF known-attack stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

import _bootstrap  # noqa: F401

from ids.features.preprocess import FeaturePreprocessor
from ids.models.stack_classifier import StackedClassifier
from ids.utils.io import read_table, read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.parquet")
    parser.add_argument("--preprocessor", default="artifacts/preprocess/preprocessor.joblib")
    parser.add_argument("--output", default="artifacts/models/stack_classifier.joblib")
    parser.add_argument("--config", default="configs/stack.yaml")
    args = parser.parse_args()
    frame = read_table(args.train)
    frame = frame[frame["binary_label"] == 1]
    preprocessor = FeaturePreprocessor.load(args.preprocessor)
    config = read_yaml(args.config)
    model = StackedClassifier.fit(
        preprocessor.transform(frame),
        frame["multiclass_label"].to_numpy(),
        n_splits=int(config.get("n_splits", 5)),
        seed=int(config.get("seed", 42)),
        random_forest_estimators=int(config.get("random_forest_estimators", 200)),
        xgboost_estimators=int(config.get("xgboost_estimators", 200)),
        max_rows=int(config["max_rows"]) if config.get("max_rows") is not None else None,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print("Wrote OOF stacking classifier")


if __name__ == "__main__":
    main()
