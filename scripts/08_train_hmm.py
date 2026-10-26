#!/usr/bin/env python3
"""Train exploratory HMM temporal enrichment when valid sequences exist."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

import _bootstrap  # noqa: F401

from ids.models.temporal_hmm import TemporalHMM, build_sequences
from ids.features.preprocess import FeaturePreprocessor
from ids.utils.io import read_table, read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.parquet")
    parser.add_argument("--preprocessor", default="artifacts/preprocess/preprocessor.joblib")
    parser.add_argument("--svm", default="artifacts/models/svm_filter.joblib")
    parser.add_argument("--stack", default="artifacts/models/stack_classifier.joblib")
    parser.add_argument("--autoencoder", default="artifacts/models/autoencoder.pt.joblib")
    parser.add_argument("--output", default="artifacts/models/temporal_hmm.joblib")
    parser.add_argument("--config", default="configs/hmm.yaml")
    args = parser.parse_args()
    config = read_yaml(args.config)
    frame = read_table(args.input)
    if "sequence_id" not in frame.columns or frame["sequence_id"].notna().sum() == 0:
        Path(args.output).unlink(missing_ok=True)
        print("No valid sequences: HMM is unavailable for this dataset/schema")
        return

    preprocessor = FeaturePreprocessor.load(args.preprocessor)
    X = preprocessor.transform(frame)
    svm_filter = joblib.load(args.svm)
    frame["svm_score"] = svm_filter.score(X)
    if args.stack and Path(args.stack).is_file():
        stack = joblib.load(args.stack)
        _, frame["known_attack_confidence"] = stack.predict_with_confidence(X)
    else:
        frame["known_attack_confidence"] = 0.0
    if args.autoencoder and Path(args.autoencoder).is_file():
        autoencoder = joblib.load(args.autoencoder)
        frame["anomaly_score"] = autoencoder.score(X)
    else:
        frame["anomaly_score"] = 0.0

    configured_columns = config.get("emission_columns", ["svm_score", "known_attack_confidence", "anomaly_score"])
    columns = [column for column in configured_columns if column in frame]
    sequences = build_sequences(
        frame,
        min_length=int(config.get("min_sequence_length", 2)),
        emission_columns=columns,
    )
    if not sequences:
        Path(args.output).unlink(missing_ok=True)
        print("No valid sequences: HMM is unavailable for this dataset/schema")
        return
    states = config.get("states", [3])
    state_count = int(config.get("state_count", states[0] if states else 3))
    model = TemporalHMM.fit(sequences, state_count=state_count, emission_columns=columns)
    joblib.dump(model, args.output)
    print(f"Wrote HMM from {len(sequences)} sequence(s)")


if __name__ == "__main__":
    main()
