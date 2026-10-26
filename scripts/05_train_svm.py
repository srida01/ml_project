#!/usr/bin/env python3
"""Train and save the binary SVM gate."""

from __future__ import annotations

import argparse

import joblib

import _bootstrap  # noqa: F401

from ids.features.preprocess import FeaturePreprocessor
from ids.models.svm_filter import SVMFilter
from ids.utils.io import read_table, read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.parquet")
    parser.add_argument("--validation", default="data/processed/validation.parquet")
    parser.add_argument("--preprocessor", default="artifacts/preprocess/preprocessor.joblib")
    parser.add_argument("--output", default="artifacts/models/svm_filter.joblib")
    parser.add_argument("--config", default="configs/svm.yaml")
    args = parser.parse_args()
    train, validation = read_table(args.train), read_table(args.validation)
    preprocessor = FeaturePreprocessor.load(args.preprocessor)
    config = read_yaml(args.config)
    model = SVMFilter.fit(
        preprocessor.transform(train),
        train["binary_label"],
        kernel=str(config.get("kernel", "rbf")),
        C=float(config.get("C", 2.0)),
        gamma=config.get("gamma", "scale"),
        class_weight=config.get("class_weight", "balanced"),
        max_exact_rows=int(config.get("max_exact_rows", 100000)),
        approx_components=int(config.get("approx_components", 512)),
        transform_batch_size=int(config.get("transform_batch_size", 4096)),
        seed=int(config.get("seed", 42)),
    )
    model.select_high_recall_threshold(
        preprocessor.transform(validation),
        validation["binary_label"],
        target_recall=float(config.get("validation_recall_target", 0.98)),
    )
    joblib.dump(model, args.output)
    print(f"Wrote SVM gate with threshold {model.threshold:.6g}")


if __name__ == "__main__":
    main()
