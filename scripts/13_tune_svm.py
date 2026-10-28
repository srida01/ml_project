#!/usr/bin/env python3
"""Validation-only hyperparameter search for the approximate RBF SVM gate.

Each candidate is trained on the training split and scored on the validation
split. The selection criterion matches deployment: the benign false-positive
rate at the threshold that keeps validation recall at the configured target.
The test split is never read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

import _bootstrap  # noqa: F401

from ids.features.preprocess import FeaturePreprocessor
from ids.models.svm_filter import SVMFilter
from ids.utils.io import read_table, read_yaml, write_json

DEFAULT_GRID = {
    "approx_components": [512, 2048],
    "alpha": [1e-4, 1e-5, 1e-6],
    "epochs": [1, 5],
    "gamma": ["scale"],
    "max_exact_rows": [100000],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/train.parquet")
    parser.add_argument("--validation", default="data/processed/validation.parquet")
    parser.add_argument("--preprocessor", default="artifacts/preprocess/preprocessor.joblib")
    parser.add_argument("--config", default="configs/svm.yaml")
    parser.add_argument("--output", default="reports/tables/svm_tuning.json")
    parser.add_argument("--grid", default=json.dumps(DEFAULT_GRID), help="JSON object of hyperparameter lists")
    args = parser.parse_args()
    from sklearn.metrics import roc_auc_score

    config = read_yaml(args.config)
    target = float(config.get("validation_recall_target", 0.98))
    preprocessor = FeaturePreprocessor.load(args.preprocessor)
    columns = list(preprocessor.columns) + ["binary_label"]
    train = read_table(args.train, columns=columns)
    validation = read_table(args.validation, columns=columns)
    X_train, y_train = preprocessor.transform(train), train["binary_label"].to_numpy()
    X_validation, y_validation = preprocessor.transform(validation), validation["binary_label"].to_numpy()
    del train, validation

    grid = {**DEFAULT_GRID, **json.loads(args.grid)}
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        start = time.perf_counter()
        model = SVMFilter.fit(
            X_train, y_train, kernel="rbf", gamma=params["gamma"], class_weight=config.get("class_weight", "balanced"),
            max_exact_rows=params["max_exact_rows"], approx_components=params["approx_components"],
            transform_batch_size=int(config.get("transform_batch_size", 4096)), alpha=params["alpha"], epochs=params["epochs"],
            seed=int(config.get("seed", 42)),
        )
        fit_seconds = time.perf_counter() - start
        scores = model.score(X_validation)
        model.select_high_recall_threshold(X_validation, y_validation, target_recall=target)
        flagged = scores >= model.threshold
        row = {
            **params,
            "validation_recall": float(flagged[y_validation == 1].mean()),
            "validation_fpr": float(flagged[y_validation == 0].mean()),
            "validation_roc_auc": float(roc_auc_score(y_validation, scores)),
            "fit_seconds": fit_seconds,
        }
        results.append(row)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    best = min(results, key=lambda r: (r["validation_fpr"], -r["validation_roc_auc"]))
    write_json({"recall_target": target, "grid": grid, "results": results, "selected": best}, args.output)
    print("Selected:", best)


if __name__ == "__main__":
    main()
