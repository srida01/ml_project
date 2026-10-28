#!/usr/bin/env python3
"""Train lightweight binary baselines; full experiments should use recorded configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from ids.data.validate import feature_columns
from ids.features.preprocess import FeaturePreprocessor
from ids.utils.io import read_table


def main() -> None:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.parquet")
    parser.add_argument("--output-dir", default="artifacts/models")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    frame = read_table(args.train)
    preprocessor = FeaturePreprocessor.fit(frame)
    X, y = preprocessor.transform(frame), frame["binary_label"]
    joblib.dump(LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y), f"{args.output_dir}/binary_logistic.joblib")
    joblib.dump(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1).fit(X, y), f"{args.output_dir}/binary_random_forest.joblib")
    print("Wrote binary baseline models")


if __name__ == "__main__":
    main()
