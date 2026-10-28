#!/usr/bin/env python3
"""Train the benign-only autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

import _bootstrap  # noqa: F401

from ids.features.preprocess import FeaturePreprocessor
from ids.models.autoencoder import AutoencoderScorer
from ids.utils.io import read_table, read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.parquet")
    parser.add_argument("--validation", default="data/processed/validation.parquet")
    parser.add_argument("--preprocessor", default="artifacts/preprocess/preprocessor.joblib")
    parser.add_argument("--output", default="artifacts/models/autoencoder.pt")
    parser.add_argument("--config", default="configs/autoencoder.yaml")
    args = parser.parse_args()
    config = read_yaml(args.config)
    preprocessor = FeaturePreprocessor.load(args.preprocessor)
    train, validation = read_table(args.train), read_table(args.validation)
    train = train[train["binary_label"] == 0]
    validation = validation[validation["binary_label"] == 0]
    X_train = preprocessor.transform(train)
    X_validation = preprocessor.transform(validation)
    X_train = _bounded_rows(X_train, config.get("max_train_rows"), int(config.get("seed", 42)))
    X_validation = _bounded_rows(X_validation, config.get("max_validation_rows"), int(config.get("seed", 42)) + 1)
    scorer = AutoencoderScorer.fit(
        X_train,
        X_validation,
        hidden_dims=tuple(config.get("hidden_dims", [128, 64, 16])),
        epochs=int(config.get("epochs", 100)),
        patience=int(config.get("patience", 10)),
        batch_size=int(config.get("batch_size", 256)),
        learning_rate=float(config.get("learning_rate", 1e-3)),
        seed=int(config.get("seed", 42)),
        fpr=float(config.get("benign_validation_fpr", 0.01)),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scorer.save(args.output)
    joblib.dump(scorer, args.output + ".joblib")
    print(f"Wrote benign-only autoencoder with threshold {scorer.threshold:.6g}")


def _bounded_rows(values, limit, seed: int):
    if limit is None or len(values) <= int(limit):
        return values
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(values), size=int(limit), replace=False)
    return values[np.sort(indices)]


if __name__ == "__main__":
    main()
