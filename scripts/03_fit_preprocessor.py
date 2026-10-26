#!/usr/bin/env python3
"""Fit preprocessing on training data only."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ids.features.preprocess import FeaturePreprocessor
from ids.utils.io import read_table, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.parquet")
    parser.add_argument("--output", default="artifacts/preprocess/preprocessor.joblib")
    parser.add_argument("--metadata", default="artifacts/preprocess/feature_schema.json")
    args = parser.parse_args()
    preprocessor = FeaturePreprocessor.fit(read_table(args.input))
    preprocessor.save(args.output)
    write_json(preprocessor.to_metadata(), args.metadata)
    print(f"Fitted {len(preprocessor.columns)} features on training data")


if __name__ == "__main__":
    main()
