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
    configured_columns = config.get("emission_columns", ["Flow Duration", "Flow Packets/s", "Packet Length Mean"])
    # Use inference-available raw features only.  In-sample supervised model
    # outputs would leak training-row information into HMM parameters.
    feature_positions = {column: position for position, column in enumerate(preprocessor.columns)}
    columns = []
    for column in configured_columns:
        if column in feature_positions:
            frame[column] = X[:, feature_positions[column]]
            columns.append(column)
    if not columns:
        Path(args.output).unlink(missing_ok=True)
        print("No configured inference-available HMM emissions: HMM is unavailable")
        return
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
